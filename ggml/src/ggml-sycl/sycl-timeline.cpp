#include "sycl-timeline.hpp"

#include <cctype>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace ggml_sycl {
namespace {

struct sycl_timeline_span_event {
    std::string category;
    std::string name;
    std::string metadata;
    std::string file;
    int         line = 0;
    std::string function;
    uint64_t    pid    = 1;
    uint64_t    tid    = 0;
    int64_t     ts_us  = 0;
    int64_t     dur_us = 0;
};

struct sycl_timeline_state {
    std::mutex mutex;

    bool                                  env_config_loaded   = false;
    sycl_timeline_config                  env_config          = {};
    bool                                  test_config_enabled = false;
    sycl_timeline_config                  test_config         = {};
    std::vector<sycl_timeline_span_event> events;
};

sycl_timeline_state & get_timeline_state() {
    static sycl_timeline_state * state = new sycl_timeline_state();
    return *state;
}

bool is_space(char ch) {
    return std::isspace(static_cast<unsigned char>(ch)) != 0;
}

std::string_view trim_ascii(const char * value) {
    if (value == nullptr) {
        return {};
    }

    const char * first = value;
    while (*first != '\0' && is_space(*first)) {
        ++first;
    }

    const char * last = first + std::strlen(first);
    while (last > first && is_space(*(last - 1))) {
        --last;
    }

    return { first, static_cast<size_t>(last - first) };
}

bool equals_ignore_case(std::string_view value, const char * literal) {
    const size_t literal_len = std::strlen(literal);
    if (value.size() != literal_len) {
        return false;
    }

    for (size_t i = 0; i < literal_len; ++i) {
        const unsigned char lhs = static_cast<unsigned char>(value[i]);
        const unsigned char rhs = static_cast<unsigned char>(literal[i]);
        if (std::tolower(lhs) != std::tolower(rhs)) {
            return false;
        }
    }

    return true;
}

sycl_timeline_mode parse_mode(const char * value) {
    const std::string_view mode = trim_ascii(value);
    if (mode.empty() || equals_ignore_case(mode, "0") || equals_ignore_case(mode, "off") ||
        equals_ignore_case(mode, "false") || equals_ignore_case(mode, "no") || equals_ignore_case(mode, "disabled")) {
        return sycl_timeline_mode::OFF;
    }
    if (equals_ignore_case(mode, "summary") || equals_ignore_case(mode, "1") || equals_ignore_case(mode, "on") ||
        equals_ignore_case(mode, "true") || equals_ignore_case(mode, "yes") || equals_ignore_case(mode, "enabled")) {
        return sycl_timeline_mode::SUMMARY;
    }
    if (equals_ignore_case(mode, "timeline")) {
        return sycl_timeline_mode::TIMELINE;
    }
    if (equals_ignore_case(mode, "timeline+events") || equals_ignore_case(mode, "events")) {
        return sycl_timeline_mode::TIMELINE_EVENTS;
    }
    return sycl_timeline_mode::OFF;
}

int parse_int_or_default(const char * value, int default_value, int min_value) {
    const char * first = value;
    if (first == nullptr) {
        return default_value;
    }
    while (*first != '\0' && is_space(*first)) {
        ++first;
    }
    if (*first == '\0') {
        return default_value;
    }

    char * end        = nullptr;
    errno             = 0;
    const long parsed = std::strtol(first, &end, 10);
    if (end == first || errno == ERANGE || parsed < min_value || parsed > INT_MAX) {
        return default_value;
    }
    while (*end != '\0' && is_space(*end)) {
        ++end;
    }
    if (*end != '\0') {
        return default_value;
    }
    return static_cast<int>(parsed);
}

sycl_timeline_config read_env_config() {
    return sycl_timeline_config_from_values(std::getenv("GGML_SYCL_TIMELINE"), std::getenv("GGML_SYCL_TIMELINE_OUTPUT"),
                                            std::getenv("GGML_SYCL_TIMELINE_TOKEN_START"),
                                            std::getenv("GGML_SYCL_TIMELINE_TOKEN_COUNT"),
                                            std::getenv("GGML_SYCL_TIMELINE_MAX_EVENTS"));
}

const sycl_timeline_config & current_config(sycl_timeline_state & state) {
    if (state.test_config_enabled) {
        return state.test_config;
    }
    if (!state.env_config_loaded) {
        state.env_config        = read_env_config();
        state.env_config_loaded = true;
    }
    return state.env_config;
}

bool config_records_spans(const sycl_timeline_config & cfg) {
    return cfg.enabled && (cfg.mode == sycl_timeline_mode::TIMELINE || cfg.mode == sycl_timeline_mode::TIMELINE_EVENTS);
}

bool timeline_records_spans() {
    sycl_timeline_state &       state = get_timeline_state();
    std::lock_guard<std::mutex> lock(state.mutex);

    const sycl_timeline_config & cfg = current_config(state);
    return config_records_spans(cfg) && cfg.max_events > 0 && static_cast<int>(state.events.size()) < cfg.max_events;
}

std::string string_or_empty(const char * value) {
    return value != nullptr ? value : "";
}

int64_t time_point_to_us(std::chrono::steady_clock::time_point time_point) {
    return std::chrono::duration_cast<std::chrono::microseconds>(time_point.time_since_epoch()).count();
}

int64_t duration_to_us(std::chrono::steady_clock::duration duration) {
    const int64_t duration_us = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    return duration_us >= 0 ? duration_us : 0;
}

void append_json_escaped(std::string & out, std::string_view value) {
    static constexpr char hex[] = "0123456789abcdef";

    out.push_back('"');
    for (const char ch : value) {
        const unsigned char uch = static_cast<unsigned char>(ch);
        switch (ch) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (uch < 0x20) {
                    out += "\\u00";
                    out.push_back(hex[(uch >> 4) & 0x0f]);
                    out.push_back(hex[uch & 0x0f]);
                } else {
                    out.push_back(ch);
                }
                break;
        }
    }
    out.push_back('"');
}

std::string format_trace_json(const std::vector<sycl_timeline_span_event> & events) {
    std::string out;
    out.reserve(events.size() * 192 + 18);
    out += "{\"traceEvents\":[";
    for (size_t i = 0; i < events.size(); ++i) {
        const sycl_timeline_span_event & event = events[i];
        if (i != 0) {
            out.push_back(',');
        }
        out += "{\"ph\":\"X\",\"cat\":";
        append_json_escaped(out, event.category);
        out += ",\"name\":";
        append_json_escaped(out, event.name);
        out += ",\"pid\":";
        out += std::to_string(event.pid);
        out += ",\"tid\":";
        out += std::to_string(event.tid);
        out += ",\"ts\":";
        out += std::to_string(event.ts_us);
        out += ",\"dur\":";
        out += std::to_string(event.dur_us);
        out += ",\"args\":{\"file\":";
        append_json_escaped(out, event.file);
        out += ",\"line\":";
        out += std::to_string(event.line);
        out += ",\"function\":";
        append_json_escaped(out, event.function);
        out += ",\"metadata\":";
        append_json_escaped(out, event.metadata);
        out += "}}";
    }
    out += "]}";
    return out;
}

}  // namespace

bool sycl_timeline_enabled_from_env(const char * value) {
    return parse_mode(value) != sycl_timeline_mode::OFF;
}

bool sycl_timeline_enabled() {
    sycl_timeline_state &       state = get_timeline_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    return current_config(state).enabled;
}

sycl_timeline_config sycl_timeline_config_from_env() {
    return read_env_config();
}

sycl_timeline_config sycl_timeline_config_from_values(const char * mode,
                                                      const char * output,
                                                      const char * token_start,
                                                      const char * token_count,
                                                      const char * max_events) {
    sycl_timeline_config cfg;

    cfg.mode        = parse_mode(mode);
    cfg.enabled     = cfg.mode != sycl_timeline_mode::OFF;
    cfg.output_path = output != nullptr ? output : "";
    cfg.token_start = parse_int_or_default(token_start, cfg.token_start, 0);
    cfg.token_count = parse_int_or_default(token_count, cfg.token_count, 0);
    cfg.max_events  = parse_int_or_default(max_events, cfg.max_events, 0);

    return cfg;
}

sycl_timeline_scope::sycl_timeline_scope(const char *           category,
                                         const char *           name,
                                         const char *           metadata,
                                         sycl_timeline_callsite callsite) {
    active_ = timeline_records_spans();
    if (!active_) {
        return;
    }

    start_time_ = std::chrono::steady_clock::now();
    category_   = string_or_empty(category);
    name_       = string_or_empty(name);
    metadata_   = string_or_empty(metadata);
    callsite_   = callsite;
}

sycl_timeline_scope::~sycl_timeline_scope() {
    if (!active_) {
        return;
    }

    try {
        sycl_timeline_record_span(category_.c_str(), name_.c_str(), metadata_.c_str(), callsite_, start_time_,
                                  std::chrono::steady_clock::now());
    } catch (...) {
    }
}

void sycl_timeline_record_span(const char *                          category,
                               const char *                          name,
                               const char *                          metadata,
                               sycl_timeline_callsite                callsite,
                               std::chrono::steady_clock::time_point start_time,
                               std::chrono::steady_clock::time_point end_time) {
    sycl_timeline_state &       state = get_timeline_state();
    std::lock_guard<std::mutex> lock(state.mutex);

    const sycl_timeline_config & cfg = current_config(state);
    if (!config_records_spans(cfg) || cfg.max_events <= 0 || static_cast<int>(state.events.size()) >= cfg.max_events) {
        return;
    }

    sycl_timeline_span_event event;
    event.category = string_or_empty(category);
    event.name     = string_or_empty(name);
    event.metadata = string_or_empty(metadata);
    event.file     = string_or_empty(callsite.file);
    event.line     = callsite.line;
    event.function = string_or_empty(callsite.function);
    event.pid      = 1;
    event.tid      = static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    event.ts_us    = time_point_to_us(start_time);
    event.dur_us   = duration_to_us(end_time - start_time);

    state.events.push_back(std::move(event));
}

std::string sycl_timeline_format_json_for_tests() {
    sycl_timeline_state &       state = get_timeline_state();
    std::lock_guard<std::mutex> lock(state.mutex);

    if (!config_records_spans(current_config(state)) || state.events.empty()) {
        return "{\"traceEvents\":[]}";
    }

    return format_trace_json(state.events);
}

void sycl_timeline_reset_for_tests() {
    sycl_timeline_state &       state = get_timeline_state();
    std::lock_guard<std::mutex> lock(state.mutex);

    state.env_config_loaded   = false;
    state.env_config          = {};
    state.test_config_enabled = false;
    state.test_config         = {};
    state.events.clear();
}

void sycl_timeline_set_config_for_tests(const sycl_timeline_config & cfg) {
    sycl_timeline_state &       state = get_timeline_state();
    std::lock_guard<std::mutex> lock(state.mutex);

    state.test_config_enabled = true;
    state.test_config         = cfg;
    state.events.clear();
}

}  // namespace ggml_sycl
