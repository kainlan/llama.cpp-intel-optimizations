#include "moe-layer-ids-cache.hpp"

#include <cctype>
#include <condition_variable>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifndef GGML_SYCL_CPP_SOURCE_PATH
#    error "GGML_SYCL_CPP_SOURCE_PATH must name the production translation unit"
#endif

static thread_local moe_layer_ids_cache g_moe_layer_ids_cache;

static void check(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

class persistent_worker {
  public:
    persistent_worker() : thread_([this] { run(); }) {}

    ~persistent_worker() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            ready_    = true;
        }
        cv_.notify_one();
        thread_.join();
    }

    void dispatch(std::function<void()> work) {
        std::unique_lock<std::mutex> lock(mutex_);
        done_  = false;
        error_ = nullptr;
        work_  = std::move(work);
        ready_ = true;
        cv_.notify_one();
        cv_.wait(lock, [this] { return done_; });
        if (error_) {
            std::rethrow_exception(error_);
        }
    }

  private:
    void run() {
        for (;;) {
            std::function<void()> work;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return ready_; });
                if (stopping_) {
                    return;
                }
                work   = std::move(work_);
                ready_ = false;
            }
            try {
                work();
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex_);
                error_ = std::current_exception();
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                done_ = true;
            }
            cv_.notify_one();
        }
    }

    std::thread             thread_;
    std::mutex              mutex_;
    std::condition_variable cv_;
    std::function<void()>   work_;
    std::exception_ptr      error_;
    bool                    ready_    = false;
    bool                    done_     = false;
    bool                    stopping_ = false;
};

struct worker_snapshot {
    std::thread::id thread_id;
    size_t          bucket_count  = 0;
    const void *    entry_address = nullptr;
    size_t          ids_capacity  = 0;
    size_t          cpu_capacity  = 0;
    size_t          sec_capacity  = 0;
    size_t          gpu_capacity  = 0;
};

static void seed_worker(worker_snapshot & snapshot, int worker_index) {
    check(g_moe_layer_ids_cache.empty(), "new persistent worker did not start with empty TLS state");
    g_moe_layer_ids_cache.reserve(8);
    auto & entry = g_moe_layer_ids_cache[17];
    entry.ids_host.reserve(32);
    entry.partition.cpu_indices.reserve(16);
    entry.partition.sec_indices.reserve(16);
    entry.partition.gpu0_cached_indices.reserve(16);
    entry.ids_host              = { 100 + worker_index, 110 + worker_index };
    entry.partition.cpu_indices = { static_cast<size_t>(worker_index + 1) };
    entry.partition.sec_indices = {
        { static_cast<size_t>(worker_index + 2), worker_index }
    };
    entry.partition.gpu0_cached_indices = { static_cast<size_t>(worker_index + 3) };
    entry.partition.valid               = true;

    snapshot.thread_id     = std::this_thread::get_id();
    snapshot.bucket_count  = g_moe_layer_ids_cache.bucket_count();
    snapshot.entry_address = &entry;
    snapshot.ids_capacity  = entry.ids_host.capacity();
    snapshot.cpu_capacity  = entry.partition.cpu_indices.capacity();
    snapshot.sec_capacity  = entry.partition.sec_indices.capacity();
    snapshot.gpu_capacity  = entry.partition.gpu0_cached_indices.capacity();
}

static void cross_graph_reset_then_refill(const worker_snapshot & snapshot, int worker_index) {
    check(std::this_thread::get_id() == snapshot.thread_id, "task migrated away from its persistent worker");

    // This is the exact synchronous helper called at all production graph boundaries.
    ggml_sycl_moe_layer_ids_cache_new_graph(g_moe_layer_ids_cache);

    check(g_moe_layer_ids_cache.size() == 1, "graph reset erased the retained map node");
    check(g_moe_layer_ids_cache.bucket_count() == snapshot.bucket_count, "graph reset replaced map allocation");
    auto it = g_moe_layer_ids_cache.find(17);
    check(it != g_moe_layer_ids_cache.end(), "graph reset erased layer key");
    auto & entry = it->second;
    check(&entry == snapshot.entry_address, "graph reset replaced retained map entry");
    check(entry.ids_host.empty(), "stale IDs were readable after graph reset");
    check(entry.partition.cpu_indices.empty(), "stale CPU partition was readable after graph reset");
    check(entry.partition.sec_indices.empty(), "stale secondary partition was readable after graph reset");
    check(entry.partition.gpu0_cached_indices.empty(), "stale GPU partition was readable after graph reset");
    check(!entry.partition.valid, "stale partition remained valid after graph reset");
    check(entry.ids_host.capacity() == snapshot.ids_capacity, "IDs vector capacity was released");
    check(entry.partition.cpu_indices.capacity() == snapshot.cpu_capacity, "CPU vector capacity was released");
    check(entry.partition.sec_indices.capacity() == snapshot.sec_capacity, "secondary vector capacity was released");
    check(entry.partition.gpu0_cached_indices.capacity() == snapshot.gpu_capacity, "GPU vector capacity was released");

    // Refill and consume within the same graph, proving reset did not disable bounded reuse.
    entry.ids_host              = { 200 + worker_index, 210 + worker_index };
    entry.partition.cpu_indices = { static_cast<size_t>(20 + worker_index) };
    entry.partition.sec_indices = {
        { static_cast<size_t>(30 + worker_index), worker_index + 4 }
    };
    entry.partition.gpu0_cached_indices = { static_cast<size_t>(40 + worker_index) };
    entry.partition.valid               = true;
    check(entry.ids_host[0] == 200 + worker_index && entry.partition.valid, "same-graph refill was not readable");
}

static void probe_same_worker(int worker_index) {
    auto it = g_moe_layer_ids_cache.find(17);
    check(it != g_moe_layer_ids_cache.end(), "worker lost its retained TLS layer");
    check(it->second.ids_host.size() == 2 && it->second.ids_host[1] == 210 + worker_index,
          "worker observed another worker's graph-local IDs");
    check(it->second.partition.valid && it->second.partition.cpu_indices[0] == static_cast<size_t>(20 + worker_index),
          "worker observed another worker's graph-local partition");
}

static size_t count_occurrences(const std::string & text, const std::string & needle) {
    size_t count = 0;
    for (size_t position = 0; (position = text.find(needle, position)) != std::string::npos;
         position += needle.size()) {
        ++count;
    }
    return count;
}

static std::string bounded_scope(const std::string & source,
                                 const std::string & begin_marker,
                                 const std::string & end_marker,
                                 const char *        contract) {
    const size_t begin = source.find(begin_marker);
    check(begin != std::string::npos, std::string("missing bounded source path: ") + contract);
    const size_t end = source.find(end_marker, begin + begin_marker.size());
    check(end != std::string::npos, std::string("missing end of bounded source path: ") + contract);
    return source.substr(begin, end - begin);
}

// Replace comments and literals with spaces while preserving offsets and
// control-flow punctuation. This keeps brace/return checks immune to examples
// in comments or braces embedded in diagnostics without pretending to parse C++.
static std::string control_flow_source(const std::string & source) {
    enum class state { code, line_comment, block_comment, string_literal, char_literal };
    std::string clean   = source;
    state       current = state::code;
    bool        escaped = false;
    for (size_t i = 0; i < source.size(); ++i) {
        const char c    = source[i];
        const char next = i + 1 < source.size() ? source[i + 1] : '\0';
        if (current == state::code) {
            if (c == '/' && next == '/') {
                clean[i] = clean[i + 1] = ' ';
                ++i;
                current = state::line_comment;
            } else if (c == '/' && next == '*') {
                clean[i] = clean[i + 1] = ' ';
                ++i;
                current = state::block_comment;
            } else if (c == '"') {
                clean[i] = ' ';
                current  = state::string_literal;
                escaped  = false;
            } else if (c == '\'') {
                clean[i] = ' ';
                current  = state::char_literal;
                escaped  = false;
            }
            continue;
        }
        if (current == state::line_comment) {
            if (c == '\n') {
                current = state::code;
            } else {
                clean[i] = ' ';
            }
            continue;
        }
        if (current == state::block_comment) {
            clean[i] = ' ';
            if (c == '*' && next == '/') {
                clean[i + 1] = ' ';
                ++i;
                current = state::code;
            }
            continue;
        }
        clean[i] = ' ';
        if (escaped) {
            escaped = false;
        } else if (c == '\\') {
            escaped = true;
        } else if ((current == state::string_literal && c == '"') || (current == state::char_literal && c == '\'')) {
            current = state::code;
        }
    }
    return clean;
}

static int brace_depth_at(const std::string & source, size_t position) {
    int depth = 0;
    for (size_t i = 0; i < position; ++i) {
        depth += source[i] == '{' ? 1 : source[i] == '}' ? -1 : 0;
    }
    return depth;
}

static bool has_return_at_depth_before(const std::string & source, size_t position, int required_depth) {
    int depth = 0;
    for (size_t i = 0; i < position;) {
        if (source[i] == '{') {
            ++depth;
            ++i;
        } else if (source[i] == '}') {
            --depth;
            ++i;
        } else if (std::isalpha(static_cast<unsigned char>(source[i])) || source[i] == '_') {
            const size_t begin = i++;
            while (i < position && (std::isalnum(static_cast<unsigned char>(source[i])) || source[i] == '_')) {
                ++i;
            }
            if (depth == required_depth && source.compare(begin, i - begin, "return") == 0) {
                return true;
            }
        } else {
            ++i;
        }
    }
    return false;
}

static bool has_unbraced_control_immediately_before(const std::string & source, size_t position) {
    size_t i = position;
    while (i > 0 && std::isspace(static_cast<unsigned char>(source[i - 1]))) {
        --i;
    }
    if (i == 0 || source[i - 1] != ')') {
        return false;
    }
    int parentheses = 0;
    do {
        --i;
        parentheses += source[i] == ')' ? 1 : source[i] == '(' ? -1 : 0;
    } while (i > 0 && parentheses != 0);
    while (i > 0 && std::isspace(static_cast<unsigned char>(source[i - 1]))) {
        --i;
    }
    const size_t end = i;
    while (i > 0 && (std::isalnum(static_cast<unsigned char>(source[i - 1])) || source[i - 1] == '_')) {
        --i;
    }
    const std::string keyword = source.substr(i, end - i);
    return keyword == "if" || keyword == "for" || keyword == "while" || keyword == "switch";
}

static void check_reset_contract(const std::string & scope, const char * path, int required_branch_depth) {
    const std::string reset      = "ggml_sycl_moe_layer_ids_cache_new_graph(g_moe_layer_ids_cache);";
    const std::string cache_name = "g_moe_layer_ids_cache";
    check(count_occurrences(scope, reset) == 1,
          std::string(path) + " path must contain exactly one worker-local reset");
    const size_t      reset_position = scope.find(reset);
    const std::string control_source = control_flow_source(scope);
    check(brace_depth_at(control_source, reset_position) == required_branch_depth &&
              !has_unbraced_control_immediately_before(control_source, reset_position),
          std::string(path) + " reset is conditional instead of unconditional at required branch depth");
    check(!has_return_at_depth_before(control_source, reset_position, required_branch_depth),
          std::string(path) + " path can return before worker-local reset at required branch depth");
    const size_t first_access = scope.find(cache_name);
    const size_t reset_access = reset_position + reset.find(cache_name);
    check(first_access == reset_access, std::string(path) + " path accesses worker-local cache before reset");
}

static void check_all_dispatches_after_reset(const std::string & scope,
                                             const std::string & dispatch,
                                             const char *        failure) {
    const std::string reset             = "ggml_sycl_moe_layer_ids_cache_new_graph(g_moe_layer_ids_cache);";
    const size_t      reset_position    = scope.find(reset);
    size_t            dispatch_position = 0;
    bool              found             = false;
    while ((dispatch_position = scope.find(dispatch, dispatch_position)) != std::string::npos) {
        found = true;
        check(reset_position < dispatch_position, failure);
        dispatch_position += dispatch.size();
    }
    check(found, std::string("missing bounded dispatch contract: ") + dispatch);
}

// Checked-in mutation matrix and observed host-only results (all exit 1):
// - remove normal reset -> "normal path must contain exactly one worker-local reset"
// - put a g_moe_layer_ids_cache read before the moved reset
//   -> "normal path accesses worker-local cache before reset"
// - move block reset after replay but before record
//   -> "block graphlet replays before worker-local reset"
// - move block reset after record
//   -> "block graphlet records before worker-local reset"
// - move segmented reset after replay but before record
//   -> "segmented replay dispatches before worker-local reset"
// - move segmented reset after record
//   -> "segmented replay records before worker-local reset"
// - replace the helper loop with cache.clear()
//   -> "graph reset erased the retained map node"
// - remove thread_local from the production declaration
//   -> "production MoE layer cache is not worker-local TLS"
// - wrap each of normal/block/segmented reset in if (false) { ... }
//   -> "<path> reset is conditional instead of unconditional at required branch depth"
// - put each reset after a same-branch unconditional return
//   -> "<path> path can return before worker-local reset at required branch depth"
// The source mutations use bounded function/branch slices, so a reset or
// dispatch in a later production path cannot accidentally satisfy the check.
static void check_production_source_order() {
    std::ifstream input(GGML_SYCL_CPP_SOURCE_PATH);
    check(input.good(), "could not open ggml-sycl.cpp for source-order contracts");
    const std::string source((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    const std::string tls_declaration = "static thread_local moe_layer_ids_cache g_moe_layer_ids_cache;";
    check(count_occurrences(source, tls_declaration) == 1, "production MoE layer cache is not worker-local TLS");

    const std::string normal = bounded_scope(source, "static void ggml_backend_sycl_graph_compute_impl(",
                                             "static bool moe_graph_pair_can_fuse_segment(", "normal compute");
    check_reset_contract(normal, "normal", 1);
    check_all_dispatches_after_reset(normal, "ggml_sycl_compute_forward(",
                                     "normal compute dispatches before worker-local reset");

    const std::string block = bounded_scope(source, "static bool moe_graph_try_block_graphlets(",
                                            "static bool check_graph_compatibility(", "block graphlet");
    check_reset_contract(block, "block graphlet", 1);
    // Record is checked first intentionally: moving reset after both record and
    // replay must name the later-record mutation rather than failing on replay.
    check_all_dispatches_after_reset(block, "moe_graph_record_block_graphs(",
                                     "block graphlet records before worker-local reset");
    check_all_dispatches_after_reset(block, "moe_graph_replay_block_graphs(",
                                     "block graphlet replays before worker-local reset");

    const std::string segmented = bounded_scope(source, "if (!block_graphlet_executed) {",
                                                "} else if (sycl_ctx->exec_graph &&", "segmented replay");
    check_reset_contract(segmented, "segmented replay", 1);
    check_all_dispatches_after_reset(segmented, "moe_graph_record_segments(",
                                     "segmented replay records before worker-local reset");
    check_all_dispatches_after_reset(segmented, "moe_graph_replay_segments(",
                                     "segmented replay dispatches before worker-local reset");
}

int main() {
    try {
        persistent_worker workers[2];
        worker_snapshot   snapshots[2];

        // Main-thread handoff is intentionally alternating and synchronous.
        workers[0].dispatch([&] { seed_worker(snapshots[0], 0); });
        workers[1].dispatch([&] { seed_worker(snapshots[1], 1); });
        check(snapshots[0].thread_id != snapshots[1].thread_id, "fixture did not create two persistent workers");
        workers[0].dispatch([&] { cross_graph_reset_then_refill(snapshots[0], 0); });
        workers[1].dispatch([&] { cross_graph_reset_then_refill(snapshots[1], 1); });
        workers[0].dispatch([&] { probe_same_worker(0); });
        workers[1].dispatch([&] { probe_same_worker(1); });

        check_production_source_order();
        std::cout << "two-worker TLS graph-boundary reset and source-order contracts passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
