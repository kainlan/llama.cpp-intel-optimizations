#include "sycl-kernel-profiler.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

struct profile_key {
    std::string name;
    std::string category;
    std::string metadata;

    bool operator<(const profile_key & other) const {
        if (name != other.name) {
            return name < other.name;
        }
        if (category != other.category) {
            return category < other.category;
        }
        return metadata < other.metadata;
    }
};

struct profile_label_snapshot {
    profile_key key;
    std::string queue_kind;
    int         device = -1;
    size_t      bytes  = 0;
};

struct profile_aggregate {
    profile_key key;
    std::string queue_kind = "unknown";
    int         device     = -1;

    std::vector<uint64_t> durations_ns;
    uint64_t              total_ns          = 0;
    uint64_t              bytes             = 0;
    uint64_t              failed_timestamps = 0;
    bool                  graph_recorded    = false;
};

struct pending_profile_event {
    profile_label_snapshot label;
    sycl::event            event;
};

struct profile_row {
    profile_key key;
    std::string queue_kind;
    int         device            = -1;
    uint64_t    count             = 0;
    uint64_t    total_ns          = 0;
    uint64_t    mean_ns           = 0;
    uint64_t    min_ns            = 0;
    uint64_t    p50_ns            = 0;
    uint64_t    p95_ns            = 0;
    uint64_t    max_ns            = 0;
    uint64_t    bytes             = 0;
    uint64_t    failed_timestamps = 0;
    bool        graph_recorded    = false;
};

struct profiler_state {
    std::mutex mutex;

    bool                            env_config_loaded = false;
    ggml_sycl_kernel_profile_config env_config;

    bool                            test_config_enabled = false;
    ggml_sycl_kernel_profile_config test_config;

    std::map<profile_key, profile_aggregate> aggregates;
    std::vector<pending_profile_event>       pending_events;

    std::atomic<int> enabled_cache{ -1 };
};

profiler_state & get_profiler_state() {
    static profiler_state * state = new profiler_state();
    return *state;
}

std::string string_from_cstr(const char * value, const char * fallback) {
    return value != nullptr ? std::string(value) : std::string(fallback);
}

std::string to_lower_ascii(std::string value) {
    for (char & ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

bool parse_bool_env(const char * value, bool default_value) {
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }

    std::string lowered = to_lower_ascii(value);
    if (lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on") {
        return true;
    }
    if (lowered == "0" || lowered == "false" || lowered == "no" || lowered == "off") {
        return false;
    }
    return true;
}

int parse_positive_int_env(const char * value, int default_value) {
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }

    char *     end    = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed <= 0 || parsed > 1000000) {
        return default_value;
    }
    return static_cast<int>(parsed);
}

ggml_sycl_kernel_profile_output_format parse_output_format(const char * value) {
    if (value == nullptr || value[0] == '\0') {
        return ggml_sycl_kernel_profile_output_format::CSV;
    }

    const std::string lowered = to_lower_ascii(value);
    if (lowered == "stderr" || lowered == "summary" || lowered == "text") {
        return ggml_sycl_kernel_profile_output_format::STDERR;
    }
    if (lowered == "csv") {
        return ggml_sycl_kernel_profile_output_format::CSV;
    }
    if (lowered == "json") {
        return ggml_sycl_kernel_profile_output_format::JSON;
    }
    if (lowered == "both" || lowered == "csv,json" || lowered == "json,csv") {
        return ggml_sycl_kernel_profile_output_format::BOTH;
    }
    return ggml_sycl_kernel_profile_output_format::CSV;
}

ggml_sycl_kernel_profile_flush_mode parse_flush_mode(const char * value) {
    if (value == nullptr || value[0] == '\0') {
        return ggml_sycl_kernel_profile_flush_mode::FINAL;
    }

    const std::string lowered = to_lower_ascii(value);
    if (lowered == "window") {
        return ggml_sycl_kernel_profile_flush_mode::WINDOW;
    }
    if (lowered == "none") {
        return ggml_sycl_kernel_profile_flush_mode::NONE;
    }
    return ggml_sycl_kernel_profile_flush_mode::FINAL;
}

profile_label_snapshot snapshot_label(const ggml_sycl_profile_label & label) {
    profile_label_snapshot snapshot;
    snapshot.key.name     = string_from_cstr(label.name, "unknown");
    snapshot.key.category = string_from_cstr(label.category, "unknown");
    snapshot.key.metadata = string_from_cstr(label.metadata, "");
    snapshot.queue_kind   = string_from_cstr(label.queue_kind, "unknown");
    snapshot.device       = label.device;
    snapshot.bytes        = label.bytes;
    return snapshot;
}

profile_aggregate & ensure_aggregate_locked(profiler_state & state, const profile_label_snapshot & label) {
    auto                inserted  = state.aggregates.emplace(label.key, profile_aggregate{});
    profile_aggregate & aggregate = inserted.first->second;
    if (inserted.second) {
        aggregate.key        = label.key;
        aggregate.queue_kind = label.queue_kind;
        aggregate.device     = label.device;
    }
    return aggregate;
}

void add_sample_locked(profiler_state & state, const profile_label_snapshot & label, uint64_t duration_ns) {
    profile_aggregate & aggregate = ensure_aggregate_locked(state, label);
    aggregate.durations_ns.push_back(duration_ns);
    aggregate.total_ns += duration_ns;
    aggregate.bytes += static_cast<uint64_t>(label.bytes);
}

void add_failed_timestamp_locked(profiler_state & state, const profile_label_snapshot & label, bool graph_recorded) {
    profile_aggregate & aggregate = ensure_aggregate_locked(state, label);
    aggregate.failed_timestamps++;
    aggregate.graph_recorded = aggregate.graph_recorded || graph_recorded;
}

uint64_t percentile_floor_index(const std::vector<uint64_t> & sorted_values, int percentile) {
    if (sorted_values.empty()) {
        return 0;
    }

    const size_t index = ((sorted_values.size() - 1) * static_cast<size_t>(percentile)) / 100;
    return sorted_values[index];
}

std::vector<profile_row> collect_rows_locked(const profiler_state & state) {
    std::vector<profile_row> rows;
    rows.reserve(state.aggregates.size());

    for (const auto & entry : state.aggregates) {
        const profile_aggregate & aggregate = entry.second;
        profile_row               row;
        row.key               = aggregate.key;
        row.queue_kind        = aggregate.queue_kind;
        row.device            = aggregate.device;
        row.count             = static_cast<uint64_t>(aggregate.durations_ns.size());
        row.total_ns          = aggregate.total_ns;
        row.bytes             = aggregate.bytes;
        row.failed_timestamps = aggregate.failed_timestamps;
        row.graph_recorded    = aggregate.graph_recorded;

        if (!aggregate.durations_ns.empty()) {
            std::vector<uint64_t> sorted = aggregate.durations_ns;
            std::sort(sorted.begin(), sorted.end());
            row.mean_ns = aggregate.total_ns / row.count;
            row.min_ns  = sorted.front();
            row.p50_ns  = percentile_floor_index(sorted, 50);
            row.p95_ns  = percentile_floor_index(sorted, 95);
            row.max_ns  = sorted.back();
        }

        rows.push_back(std::move(row));
    }

    std::sort(rows.begin(), rows.end(), [](const profile_row & lhs, const profile_row & rhs) {
        if (lhs.total_ns != rhs.total_ns) {
            return lhs.total_ns > rhs.total_ns;
        }
        if (lhs.key.name != rhs.key.name) {
            return lhs.key.name < rhs.key.name;
        }
        if (lhs.key.category != rhs.key.category) {
            return lhs.key.category < rhs.key.category;
        }
        return lhs.key.metadata < rhs.key.metadata;
    });

    return rows;
}

std::vector<profile_row> collect_rows() {
    profiler_state &            state = get_profiler_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    return collect_rows_locked(state);
}

std::string csv_sanitize(std::string value) {
    for (char & ch : value) {
        if (ch == ',' || ch == '\n' || ch == '\r') {
            ch = ';';
        }
    }
    return value;
}

std::string json_escape(const std::string & value) {
    std::ostringstream out;
    for (const unsigned char ch : value) {
        switch (ch) {
            case '\\':
                out << "\\\\";
                break;
            case '"':
                out << "\\\"";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                if (ch < 0x20) {
                    static const char hex[] = "0123456789abcdef";
                    out << "\\u00" << hex[(ch >> 4) & 0xf] << hex[ch & 0xf];
                } else {
                    out << static_cast<char>(ch);
                }
                break;
        }
    }
    return out.str();
}

std::string format_csv_rows(const std::vector<profile_row> & rows) {
    std::ostringstream out;
    out << "name,category,metadata,device,queue_kind,count,total_ns,mean_ns,min_ns,p50_ns,p95_ns,max_ns,bytes,failed_"
           "timestamps,graph_recorded\n";
    for (const profile_row & row : rows) {
        out << csv_sanitize(row.key.name) << ',' << csv_sanitize(row.key.category) << ','
            << csv_sanitize(row.key.metadata) << ',' << row.device << ',' << csv_sanitize(row.queue_kind) << ','
            << row.count << ',' << row.total_ns << ',' << row.mean_ns << ',' << row.min_ns << ',' << row.p50_ns << ','
            << row.p95_ns << ',' << row.max_ns << ',' << row.bytes << ',' << row.failed_timestamps << ','
            << (row.graph_recorded ? 1 : 0) << '\n';
    }
    return out.str();
}

std::string format_json_rows(const std::vector<profile_row> & rows) {
    std::ostringstream out;
    out << "{\"kernels\":[";
    for (size_t i = 0; i < rows.size(); ++i) {
        const profile_row & row = rows[i];
        if (i > 0) {
            out << ',';
        }
        out << '{' << "\"name\":\"" << json_escape(row.key.name) << "\","
            << "\"category\":\"" << json_escape(row.key.category) << "\","
            << "\"metadata\":\"" << json_escape(row.key.metadata) << "\","
            << "\"device\":" << row.device << ',' << "\"queue_kind\":\"" << json_escape(row.queue_kind) << "\","
            << "\"count\":" << row.count << ',' << "\"total_ns\":" << row.total_ns << ','
            << "\"mean_ns\":" << row.mean_ns << ',' << "\"min_ns\":" << row.min_ns << ',' << "\"p50_ns\":" << row.p50_ns
            << ',' << "\"p95_ns\":" << row.p95_ns << ',' << "\"max_ns\":" << row.max_ns << ','
            << "\"bytes\":" << row.bytes << ',' << "\"failed_timestamps\":" << row.failed_timestamps << ','
            << "\"graph_recorded\":" << (row.graph_recorded ? 1 : 0) << '}';
    }
    out << "]}\n";
    return out.str();
}

std::string format_summary_rows(const std::vector<profile_row> & rows, int top_n) {
    const int          limit = top_n > 0 ? top_n : static_cast<int>(rows.size());
    std::ostringstream out;
    out << "total_ns count mean_ns failed_timestamps name category metadata\n";
    int emitted = 0;
    for (const profile_row & row : rows) {
        if (emitted >= limit) {
            break;
        }
        out << row.total_ns << ' ' << row.count << ' ' << row.mean_ns << ' ' << row.failed_timestamps << ' '
            << row.key.name << ' ' << row.key.category << ' ' << row.key.metadata << '\n';
        emitted++;
    }
    return out.str();
}

ggml_sycl_kernel_profile_config current_config() {
    profiler_state &            state = get_profiler_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.test_config_enabled) {
        return state.test_config;
    }
    if (!state.env_config_loaded) {
        state.env_config        = ggml_sycl_kernel_profile_config_from_env();
        state.env_config_loaded = true;
    }
    return state.env_config;
}

void drain_pending_events(bool wait_for_events) {
    profiler_state & state = get_profiler_state();

    std::vector<pending_profile_event> pending;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        pending.swap(state.pending_events);
    }

    std::vector<pending_profile_event> still_pending;
    for (pending_profile_event & pending_event : pending) {
        if (!wait_for_events) {
            try {
                const auto status = pending_event.event.get_info<sycl::info::event::command_execution_status>();
                if (status != sycl::info::event_command_status::complete) {
                    still_pending.push_back(pending_event);
                    continue;
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(state.mutex);
                add_failed_timestamp_locked(state, pending_event.label, false);
                continue;
            }
        } else {
            try {
                pending_event.event.wait_and_throw();
            } catch (...) {
                std::lock_guard<std::mutex> lock(state.mutex);
                add_failed_timestamp_locked(state, pending_event.label, false);
                continue;
            }
        }

        try {
            const uint64_t start = pending_event.event.get_profiling_info<sycl::info::event_profiling::command_start>();
            const uint64_t end   = pending_event.event.get_profiling_info<sycl::info::event_profiling::command_end>();
            std::lock_guard<std::mutex> lock(state.mutex);
            if (end >= start) {
                add_sample_locked(state, pending_event.label, end - start);
            } else {
                add_failed_timestamp_locked(state, pending_event.label, false);
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(state.mutex);
            add_failed_timestamp_locked(state, pending_event.label, false);
        }
    }

    if (!still_pending.empty()) {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.pending_events.insert(state.pending_events.end(), still_pending.begin(), still_pending.end());
    }
}

bool has_suffix(const std::string & value, const char * suffix) {
    const std::string suffix_string(suffix);
    return value.size() >= suffix_string.size() &&
           value.compare(value.size() - suffix_string.size(), suffix_string.size(), suffix_string) == 0;
}

std::string output_path_for_format(const std::string & base_path, ggml_sycl_kernel_profile_output_format format) {
    if (base_path.empty()) {
        return std::string();
    }

    if (format == ggml_sycl_kernel_profile_output_format::CSV) {
        return has_suffix(base_path, ".csv") ? base_path : base_path + ".csv";
    }
    if (format == ggml_sycl_kernel_profile_output_format::JSON) {
        return has_suffix(base_path, ".json") ? base_path : base_path + ".json";
    }
    return base_path;
}

void write_text_file_noexcept(const std::string & path, const std::string & text) {
    if (path.empty()) {
        return;
    }

    try {
        std::ofstream out(path);
        if (out) {
            out << text;
        }
    } catch (...) {
    }
}

bool effective_wait_for_flush(const ggml_sycl_kernel_profile_config & cfg, bool requested_wait) {
    if (cfg.flush_mode == ggml_sycl_kernel_profile_flush_mode::WINDOW) {
        return true;
    }
    if (cfg.flush_mode == ggml_sycl_kernel_profile_flush_mode::NONE) {
        return false;
    }
    return requested_wait;
}

}  // namespace

ggml_sycl_kernel_profile_config ggml_sycl_kernel_profile_config_from_env() {
    ggml_sycl_kernel_profile_config cfg;

    cfg.enabled = parse_bool_env(std::getenv("GGML_SYCL_KERNEL_PROFILE"), false);
    if (const char * output = std::getenv("GGML_SYCL_KERNEL_PROFILE_OUTPUT")) {
        cfg.output_path = output;
    }
    cfg.output_format = parse_output_format(std::getenv("GGML_SYCL_KERNEL_PROFILE_FORMAT"));
    cfg.top_n         = parse_positive_int_env(std::getenv("GGML_SYCL_KERNEL_PROFILE_TOP_N"), cfg.top_n);
    cfg.raw_events    = parse_bool_env(std::getenv("GGML_SYCL_KERNEL_PROFILE_RAW"), false);
    cfg.flush_mode    = parse_flush_mode(std::getenv("GGML_SYCL_KERNEL_PROFILE_FLUSH"));

    return cfg;
}

bool ggml_sycl_kernel_profile_enabled() {
    profiler_state & state  = get_profiler_state();
    const int        cached = state.enabled_cache.load(std::memory_order_acquire);
    if (cached >= 0) {
        return cached != 0;
    }

    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.test_config_enabled) {
        state.enabled_cache.store(state.test_config.enabled ? 1 : 0, std::memory_order_release);
        return state.test_config.enabled;
    }
    if (!state.env_config_loaded) {
        state.env_config        = ggml_sycl_kernel_profile_config_from_env();
        state.env_config_loaded = true;
    }
    state.enabled_cache.store(state.env_config.enabled ? 1 : 0, std::memory_order_release);
    return state.env_config.enabled;
}

void ggml_sycl_kernel_profile_record_event(const ggml_sycl_profile_label & label, const sycl::event & event) {
    if (!ggml_sycl_kernel_profile_enabled()) {
        return;
    }

    profiler_state &            state = get_profiler_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.pending_events.push_back(pending_profile_event{ snapshot_label(label), event });
}

void ggml_sycl_kernel_profile_flush(bool wait_for_events, const char * reason) {
    if (!ggml_sycl_kernel_profile_enabled()) {
        return;
    }

    try {
        ggml_sycl_kernel_profile_config cfg = current_config();
        wait_for_events                     = effective_wait_for_flush(cfg, wait_for_events);

        drain_pending_events(wait_for_events);

        const std::vector<profile_row> rows    = collect_rows();
        const std::string              summary = format_summary_rows(rows, cfg.top_n);
        std::fprintf(stderr, "SYCL kernel profile (%s):\n%s", reason != nullptr ? reason : "flush", summary.c_str());

        if (!cfg.output_path.empty()) {
            if (cfg.output_format == ggml_sycl_kernel_profile_output_format::CSV ||
                cfg.output_format == ggml_sycl_kernel_profile_output_format::BOTH) {
                write_text_file_noexcept(
                    output_path_for_format(cfg.output_path, ggml_sycl_kernel_profile_output_format::CSV),
                    format_csv_rows(rows));
            }
            if (cfg.output_format == ggml_sycl_kernel_profile_output_format::JSON ||
                cfg.output_format == ggml_sycl_kernel_profile_output_format::BOTH) {
                write_text_file_noexcept(
                    output_path_for_format(cfg.output_path, ggml_sycl_kernel_profile_output_format::JSON),
                    format_json_rows(rows));
            }
        }
    } catch (...) {
    }
}

void ggml_sycl_kernel_profile_reset_for_test() {
    profiler_state &            state = get_profiler_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.env_config_loaded   = false;
    state.env_config          = ggml_sycl_kernel_profile_config{};
    state.test_config_enabled = false;
    state.test_config         = ggml_sycl_kernel_profile_config{};
    state.aggregates.clear();
    state.pending_events.clear();
    state.enabled_cache.store(-1, std::memory_order_release);
}

void ggml_sycl_kernel_profile_set_config_for_test(const ggml_sycl_kernel_profile_config & cfg) {
    profiler_state &            state = get_profiler_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.test_config_enabled = true;
    state.test_config         = cfg;
    state.enabled_cache.store(cfg.enabled ? 1 : 0, std::memory_order_release);
}

void ggml_sycl_kernel_profile_add_sample_for_test(const ggml_sycl_profile_label & label, uint64_t duration_ns) {
    profiler_state &            state = get_profiler_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    add_sample_locked(state, snapshot_label(label), duration_ns);
}

void ggml_sycl_kernel_profile_add_failed_timestamp_for_test(const ggml_sycl_profile_label & label,
                                                            bool                            graph_recorded) {
    profiler_state &            state = get_profiler_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    add_failed_timestamp_locked(state, snapshot_label(label), graph_recorded);
}

std::string ggml_sycl_kernel_profile_format_csv_for_test() {
    return format_csv_rows(collect_rows());
}

std::string ggml_sycl_kernel_profile_format_json_for_test() {
    return format_json_rows(collect_rows());
}

std::string ggml_sycl_kernel_profile_format_summary_for_test(int top_n) {
    return format_summary_rows(collect_rows(), top_n);
}

bool ggml_sycl_kernel_profile_effective_wait_for_test(bool requested_wait) {
    return effective_wait_for_flush(current_config(), requested_wait);
}
