// Execution gate + stress test for unified_cache::unpin_on_event via binbcast ops.
//
// Usage:
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-unified-cache-unpin-event --mode=safe
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-unified-cache-unpin-event --mode=barrier
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-unified-cache-unpin-event --mode=reuse
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-unified-cache-unpin-event --mode=compare
//
// `compare` covers all three GGML_SYCL_BINBCAST_EVENT_MODE values and is both
// what ctest passes and what a bare invocation defaults to, so no mode is left
// ungated by either entry point.
//
// Two independent checks run per mode:
//
//   1. run_unpin_lease_gate()   -- the pin/unpin LEASE property.  Constructs a
//      genuinely cache-resident weight so binbcast's `maybe_pin_cached` actually
//      populates `pins[]`, then proves the lease is released only after the
//      gating event completes.  See the comment block on that function.
//   2. run_binbcast_stress()    -- the pre-existing result-correctness stress
//      loop.  It does NOT reach the unpin path (its weight is host-buffer
//      allocated, so it carries no `tensor->layout` and is never cache-resident);
//      it is kept because its result check is non-vacuous on its own terms and
//      caught a real ggml_gallocr aliasing bug.  It reports its own blindness so
//      a reader cannot mistake a green stress loop for lease coverage.

#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml.h"

// Internal SYCL backend headers.  The test target already puts ggml/src on the
// include path and compiles with -fsycl, so the unified cache's own API is
// reachable in-process.  This is what makes a POSITIVE assertion on the lease
// possible: `is_pinned()` reads the same `unified_cache_entry::pinned` flag that
// `unpin_on_event` clears, instead of inferring the path ran from a passing
// result.
#include "ggml-sycl/unified-cache.hpp"

#include <unistd.h>

#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if !defined(GGML_USE_SYCL)
int main() {
    fprintf(stderr, "GGML_USE_SYCL not enabled; skipping test.\n");
    return 0;
}
#else

// Test hooks exported by ggml/src/ggml-sycl/binbcast.cpp.  Both read the
// GGML_SYCL_BINBCAST_EVENT_MODE environment variable through the same parser the
// backend uses, so they exercise the real policy rather than a copy of it.
const char * ggml_sycl_test_binbcast_event_mode_name();
const char * ggml_sycl_test_binbcast_event_source_name(bool kernel_event_valid);

namespace {

enum class event_mode {
    SAFE,
    BARRIER,
    REUSE,
    COMPARE,
};

const char * mode_name(event_mode mode) {
    switch (mode) {
        case event_mode::SAFE:
            return "safe";
        case event_mode::BARRIER:
            return "barrier";
        case event_mode::REUSE:
            return "reuse";
        case event_mode::COMPARE:
            return "compare";
        default:
            return "unknown";
    }
}

event_mode parse_mode(const char * mode_str) {
    if (!mode_str) {
        return event_mode::COMPARE;
    }
    if (std::strcmp(mode_str, "safe") == 0) {
        return event_mode::SAFE;
    }
    if (std::strcmp(mode_str, "barrier") == 0) {
        return event_mode::BARRIER;
    }
    if (std::strcmp(mode_str, "reuse") == 0) {
        return event_mode::REUSE;
    }
    if (std::strcmp(mode_str, "compare") == 0 || std::strcmp(mode_str, "both") == 0) {
        return event_mode::COMPARE;
    }
    return event_mode::COMPARE;
}

int parse_iters(const char * iters_str, int default_iters) {
    if (!iters_str || !*iters_str) {
        return default_iters;
    }
    const int value = std::atoi(iters_str);
    return value > 0 ? value : default_iters;
}

const char * get_arg(int argc, char ** argv, const char * prefix) {
    const size_t prefix_len = std::strlen(prefix);
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], prefix, prefix_len) == 0) {
            return argv[i] + prefix_len;
        }
    }
    return nullptr;
}

// Host-only gate on the completion-event policy: no device work, so it also runs
// on a busy or absent GPU.
//
// The invariant: every (mode, kernel_event_valid) combination must resolve to a
// REAL completion event.  Both consumers of the binbcast completion event need
// one -- `ggml_sycl_set_tensor_ready_event` for GGML_OP_MUL, and
// `unified_cache::unpin_on_event`, which parks the weight-cache lease release on
// it.  A default-constructed sycl::event reads as ALREADY COMPLETE, so "no
// event" would unpin a weight the GPU is still reading (DEVICE_LOST or silent
// corruption) -- and it would look like a speedup.  "reuse" without a captured
// kernel event must therefore still fall back to a real submission.
static bool check_event_source_policy() {
    struct expectation {
        const char * env;                 // GGML_SYCL_BINBCAST_EVENT_MODE value, nullptr = unset
        const char * expect_mode;         // mode the backend parses out of it
        const char * expect_with_kernel;  // event source when a kernel event was captured
        const char * expect_no_kernel;    // event source when none was captured
    };

    const expectation cases[] = {
        { nullptr,   "barrier", "submission", "submission" },
        { "barrier", "barrier", "submission", "submission" },
        { "safe",    "safe",    "submission", "submission" },
        { "reuse",   "reuse",   "kernel",     "submission" },
        { "bogus",   "barrier", "submission", "submission" },
    };

    const char * saved     = std::getenv("GGML_SYCL_BINBCAST_EVENT_MODE");
    std::string  saved_str = saved ? saved : "";
    const bool   had_saved = saved != nullptr;

    bool ok = true;
    for (const expectation & c : cases) {
        if (c.env) {
            setenv("GGML_SYCL_BINBCAST_EVENT_MODE", c.env, 1);
        } else {
            unsetenv("GGML_SYCL_BINBCAST_EVENT_MODE");
        }

        const char * got_mode        = ggml_sycl_test_binbcast_event_mode_name();
        const char * got_with_kernel = ggml_sycl_test_binbcast_event_source_name(true);
        const char * got_no_kernel   = ggml_sycl_test_binbcast_event_source_name(false);

        if (std::strcmp(got_mode, c.expect_mode) != 0) {
            fprintf(stderr, "[policy] env=%s: mode is '%s', expected '%s'\n", c.env ? c.env : "(unset)", got_mode,
                    c.expect_mode);
            ok = false;
        }
        if (std::strcmp(got_with_kernel, c.expect_with_kernel) != 0) {
            fprintf(stderr, "[policy] env=%s kernel_event_valid=1: source is '%s', expected '%s'\n",
                    c.env ? c.env : "(unset)", got_with_kernel, c.expect_with_kernel);
            ok = false;
        }
        if (std::strcmp(got_no_kernel, c.expect_no_kernel) != 0) {
            fprintf(stderr, "[policy] env=%s kernel_event_valid=0: source is '%s', expected '%s'\n",
                    c.env ? c.env : "(unset)", got_no_kernel, c.expect_no_kernel);
            ok = false;
        }
    }

    if (had_saved) {
        setenv("GGML_SYCL_BINBCAST_EVENT_MODE", saved_str.c_str(), 1);
    } else {
        unsetenv("GGML_SYCL_BINBCAST_EVENT_MODE");
    }

    printf("Event source policy: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// The lease gate: real execution coverage for unified_cache::unpin_on_event.
// ---------------------------------------------------------------------------
//
// WHAT WAS MISSING.  `binbcast.cpp`'s post-op unpin loop only runs when
// `maybe_pin_cached` populated `pins[]`, which needs THREE things at once:
//
//   a. `ggml_sycl_should_skip_pin_unpin(device)` false.  It is true when the
//      placement planner is authoritative AND the eviction guard is up, which is
//      the steady state in real decode -- there the guard already protects the
//      weight and per-op pinning is redundant by design.  A bare test backend has
//      no placement plan, so this is false and the per-op path is live.
//   b. `ggml_sycl_get_layout_info(tensor)` non-null with a non-null `data_ptr`.
//      `tensor->layout` is only assigned by the DEVICE buffer_type
//      init_tensor/set_tensor path -- a `ggml_backend_sycl_host_buffer_type()`
//      weight can never acquire one.
//   c. `cache->is_cached(key, layout->mode)` true.
//
// This function constructs all three: the weight lives in a DEVICE buffer marked
// GGML_BACKEND_BUFFER_USAGE_WEIGHTS (gives b), and `register_ready` publishes the
// tensor's own device pointer as a non-owning READY DENSE_WEIGHT/AOS entry (gives
// c).  `register_ready` is the same call the backend uses for MoE expert weights
// in `ggml_backend_sycl_buffer_set_tensor`, so the entry is shaped like a real one
// and the kernel reads exactly the memory the lease protects.
//
// GRAPH CAPTURE MUST BE OFF.  The per-op pin path is unreachable under SYCL graph
// replay in BOTH phases: while recording, `g_ggml_sycl_graph_recording` makes
// `maybe_pin_cached` return immediately (pin ops are not graph-safe), and on
// replay the op is never dispatched at all.  main() therefore sets
// GGML_SYCL_DISABLE_GRAPH=1 before backend init, where that flag is latched.
//
// WHY THE OBSERVATION IS MADE FROM THE BACKEND'S TRACE, NOT FROM OUTSIDE THE OP.
// `unpin_on_event` does not unpin; it parks the event on `inflight_unpins_`, and a
// later drain (`process_deferred_frees`) polls `event_complete()` and releases only
// when the event has completed.  The property under test is therefore: the release
// happens after the gating event completes, and not before.
//
// That whole sequence is over before `ggml_backend_graph_compute` returns.  Its
// exit path drains the queue (`graph-compute-exit` wait) for any graph that is not
// decode-shaped, and the drain that follows releases the lease.  So an observer
// outside the op ALWAYS sees the post-release state -- `is_pinned()` is already
// false, which proves neither that the pin was taken nor that it was held.  The
// only place the in-flight moment is visible is the backend's own trace, so the
// gate enables SYCL debug and reads these four lines back:
//
//   [SYCL-BINBCAST] unpin event mode=M source=S pins=N      (binbcast.cpp)
//   [UNIFIED-CACHE] in-flight pin   ... ready=R             (unpin_on_event)
//   [UNIFIED-CACHE] unpin check     ... ready=R             (the drain's poll)
//   [UNIFIED-CACHE] in-flight unpin ...                     (the release)
//
// and asserts, filtered to this weight's own name_hash:
//
//   1. pins >= 1               -- pin_count > 0 was reached: the path EXECUTED.
//                                 This is the assertion the test previously lacked
//                                 entirely; `pins` came out 0 in every config.
//   2. source == expected(M)   -- the live op resolved the event source the policy
//                                 says it should, `reuse` -> kernel event included.
//   3. release ordering        -- the `in-flight unpin` for this key is preceded by
//                                 a poll of that key, so the release came from the
//                                 event-gated drain and not from some other path
//                                 (`unpin`, `unpin_all`) that bypasses the event.
//   4. release happened        -- the lease is not simply leaked, which would
//                                 satisfy 3 for the wrong reason.
//
// WHY THE `ready=` FLAG IN THAT TRACE IS REPORTED BUT NOT ASSERTED ON.  It looks
// like the natural non-vacuity check -- "the event really was in flight when the
// release was parked".  It cannot serve as one for a binbcast event, and the reason
// is a property of this stack worth writing down.
//
// `unified_cache::event_complete()` is a bare
// `event::get_info<sycl::info::event::command_execution_status>()`.  On a queue
// created with `sycl::property::queue::enable_profiling` -- which is every backend
// stream, see `default_queue_properties()` in common.hpp -- that query does not poll:
// it BLOCKS until the command finishes and then answers `complete`.  Measured on the
// Arc Pro B50 with an 825 ms kernel, four queue configurations:
//
//   in_order only                  status returned @+0.012 ms = submitted
//   in_order + enable_profiling    status returned @+826.2  ms = complete
//   out-of-order only              status returned @+0.956 ms = submitted
//   out-of-order + enable_profiling status returned @+825.2  ms = complete
//
// So for any event produced on a backend stream, `event_complete()` can only ever
// return true, after stalling for the outstanding GPU work.  That is CONSERVATIVE --
// it cannot cause an early release -- but it means ready=0 is unobservable for a
// binbcast event no matter how long the gated kernel runs, and gating on it would
// fail this test forever for a reason unrelated to the lease.
//
// STAGE 2 therefore tests the property on a queue where the poll behaves: the
// cache's own DMA queue is created `in_order` WITHOUT enable_profiling
// (unified-cache.cpp), so `event_complete()` there is a genuine non-blocking poll.
// `unpin_on_event` is public, so it can be driven directly:
//
//   pin -> submit a calibrated ~60 ms kernel -> unpin_on_event(its event)
//     assert still pinned                 (parked, never released synchronously)
//   -> drain
//     assert STILL PINNED                 <-- the early-unpin detector
//   -> wait out the kernel, and require that wait to have taken real time
//                                         <-- non-vacuity: the event WAS in flight
//                                             during the drain above
//   -> drain
//     assert now unpinned                 (released once the event completed)
//
// That is the full "released only after the gating event completes, and not before"
// property, with its own proof that the window was open.
//
// What none of this proves: an early unpin only corrupts data if the cache then
// evicts or overwrites the entry, which needs VRAM pressure a single-entry cache
// cannot produce.  The assertions are on the lease state machine itself, which is
// the mechanism, and are strictly stronger than waiting for corruption that this
// setup would never generate.
static constexpr int64_t k_gate_ne0 = 4096;
static constexpr int64_t k_gate_ne1 = 16384;

// Stage 2 tuning.  The busy kernel is bounded work, never a spin on a host flag:
// an unbounded GPU kernel risks the driver's hang detector and a GT reset.
static constexpr int    k_busy_items        = 1 << 20;
static constexpr int    k_busy_cal_iters    = 1024;
static constexpr double k_busy_target_ms    = 60.0;
static constexpr int    k_busy_max_iters    = 1 << 22;
// After the in-flight drain, waiting out the busy kernel must still take at least
// this long -- that is the proof the gating event had not completed when the drain
// looked at it.  Generous, since the target is ~60 ms.
static constexpr double k_busy_remaining_ms = 10.0;

// Bounded busy kernel: a dependent FMA chain, so its duration scales with `iters`
// and cannot be optimised away (the result is stored).
static sycl::event submit_busy_kernel(sycl::queue & q, float * scratch, int items, int iters) {
    return q.parallel_for(sycl::range<1>(static_cast<size_t>(items)), [=](sycl::id<1> i) {
        float v = static_cast<float>(i[0] & 0xff) * 1e-3f;
        for (int k = 0; k < iters; ++k) {
            v = v * 1.0000001f + 1e-7f;
        }
        scratch[i[0]] = v;
    });
}

// Captures the SYCL backend's debug trace.  GGML_SYCL_DEBUG writes with a plain
// fprintf(stderr, ...) rather than through ggml's log callback, so a
// `ggml_log_set` hook cannot see it -- the fd has to be redirected.  The capture
// stays installed for the whole run so the debug flood never reaches ctest's log;
// `dump_tail` prints it only when something failed.
class stderr_capture {
  public:
    bool begin() {
        char path[] = "/tmp/unpin-event-trace-XXXXXX";
        fd_         = mkstemp(path);
        if (fd_ < 0) {
            return false;
        }
        path_ = path;
        unlink(path_.c_str());  // anonymous: nothing to clean up on any exit path
        fflush(stderr);
        saved_fd_ = dup(STDERR_FILENO);
        if (saved_fd_ < 0 || dup2(fd_, STDERR_FILENO) < 0) {
            return false;
        }
        active_ = true;
        return true;
    }

    void end() {
        if (!active_) {
            return;
        }
        fflush(stderr);
        dup2(saved_fd_, STDERR_FILENO);
        close(saved_fd_);
        saved_fd_ = -1;
        active_   = false;
    }

    bool active() const { return active_; }

    // Byte offset to slice from, so each gate reads only its own trace.
    off_t mark() const {
        fflush(stderr);
        return lseek(fd_, 0, SEEK_CUR);
    }

    std::string read_from(off_t offset) const {
        fflush(stderr);
        const off_t end_off = lseek(fd_, 0, SEEK_CUR);
        if (end_off <= offset) {
            return std::string();
        }
        std::string   out(static_cast<size_t>(end_off - offset), '\0');
        const ssize_t got = pread(fd_, &out[0], out.size(), offset);
        out.resize(got > 0 ? static_cast<size_t>(got) : 0);
        return out;
    }

    void dump_tail(size_t max_lines) const {
        if (fd_ < 0) {
            return;
        }
        const off_t   end_off = lseek(fd_, 0, SEEK_CUR);
        const off_t   window  = end_off > 512 * 1024 ? 512 * 1024 : end_off;
        std::string   tail(static_cast<size_t>(window), '\0');
        const ssize_t got = pread(fd_, &tail[0], tail.size(), end_off - window);
        if (got <= 0) {
            return;
        }
        tail.resize(static_cast<size_t>(got));
        std::vector<std::string> lines;
        size_t                   pos = 0;
        while (pos < tail.size()) {
            const size_t nl = tail.find('\n', pos);
            const size_t to = nl == std::string::npos ? tail.size() : nl;
            lines.push_back(tail.substr(pos, to - pos));
            pos = to + 1;
        }
        const size_t first = lines.size() > max_lines ? lines.size() - max_lines : 0;
        fprintf(stderr, "--- captured SYCL trace (last %zu lines) ---\n", lines.size() - first);
        for (size_t i = first; i < lines.size(); ++i) {
            fprintf(stderr, "%s\n", lines[i].c_str());
        }
        fprintf(stderr, "--- end captured SYCL trace ---\n");
    }

  private:
    int         fd_       = -1;
    int         saved_fd_ = -1;
    bool        active_   = false;
    std::string path_;
};

static stderr_capture g_trace;

// What the gate found in the backend's trace for one weight.
struct trace_findings {
    bool        saw_unpin_event = false;
    int         pins            = 0;
    std::string source;
    bool        saw_park         = false;
    bool        park_ready       = true;   // ready flag at unpin_on_event time
    bool        saw_release      = false;
    bool        saw_check_ready0 = false;  // a drain polled and correctly did not release
    bool        released_early   = false;  // a release with no preceding ready=1 check
};

static bool line_has(const std::string & line, const char * needle) {
    return line.find(needle) != std::string::npos;
}

// Parse "<key>=<int>" out of a trace line. Returns false if absent.
static bool parse_kv_int(const std::string & line, const char * key, long & out) {
    const size_t at = line.find(key);
    if (at == std::string::npos) {
        return false;
    }
    out = std::strtol(line.c_str() + at + std::strlen(key), nullptr, 0);
    return true;
}

static std::string parse_kv_word(const std::string & line, const char * key) {
    const size_t at = line.find(key);
    if (at == std::string::npos) {
        return std::string();
    }
    const size_t start = at + std::strlen(key);
    size_t       end   = start;
    while (end < line.size() && !std::isspace(static_cast<unsigned char>(line[end]))) {
        ++end;
    }
    return line.substr(start, end - start);
}

static trace_findings scan_trace(const std::string & trace, uint64_t name_hash) {
    char hash_tag[64];
    std::snprintf(hash_tag, sizeof(hash_tag), "name_hash=0x%llx", (unsigned long long) name_hash);

    trace_findings f;
    bool           last_check_ready = false;
    bool           saw_any_check    = false;

    size_t pos = 0;
    while (pos < trace.size()) {
        const size_t      nl   = trace.find('\n', pos);
        const size_t      to   = nl == std::string::npos ? trace.size() : nl;
        const std::string line = trace.substr(pos, to - pos);
        pos                    = to + 1;

        if (line_has(line, "[SYCL-BINBCAST] unpin event")) {
            long pins = 0;
            if (parse_kv_int(line, "pins=", pins)) {
                f.saw_unpin_event = true;
                if (pins > f.pins) {
                    f.pins = static_cast<int>(pins);
                }
                f.source = parse_kv_word(line, "source=");
            }
            continue;
        }
        if (!line_has(line, hash_tag)) {
            continue;
        }
        long ready = 1;
        if (line_has(line, "[UNIFIED-CACHE] in-flight pin")) {
            f.saw_park = true;
            if (parse_kv_int(line, "ready=", ready)) {
                f.park_ready = ready != 0;
            }
        } else if (line_has(line, "[UNIFIED-CACHE] unpin check")) {
            if (parse_kv_int(line, "ready=", ready)) {
                saw_any_check    = true;
                last_check_ready = ready != 0;
                if (ready == 0) {
                    f.saw_check_ready0 = true;
                }
            }
        } else if (line_has(line, "[UNIFIED-CACHE] in-flight unpin")) {
            f.saw_release = true;
            // The release must be immediately preceded by a poll that saw the
            // gating event complete.  Anything else is an early release.
            if (!saw_any_check || !last_check_ready) {
                f.released_early = true;
            }
        }
    }
    return f;
}

static bool run_unpin_lease_gate(event_mode mode) {
    const char * env_mode = mode_name(mode);
    setenv("GGML_SYCL_BINBCAST_EVENT_MODE", env_mode, 1);

    const int64_t ne1 = parse_iters(std::getenv("GGML_SYCL_UNPIN_GATE_ROWS"), static_cast<int>(k_gate_ne1));

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        fprintf(stderr, "[gate %s] failed to init SYCL backend\n", env_mode);
        return false;
    }

    // The DEVICE buffer type.  Using the host buffer type here is what left the
    // path uncovered: it cannot produce a tensor->layout.
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    ggml_init_params params = {
        /*.mem_size   =*/16 * 1024 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "[gate %s] failed to init ggml context\n", env_mode);
        ggml_backend_free(backend);
        return false;
    }

    ggml_tensor * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k_gate_ne0);
    ggml_set_name(weight, (std::string("gate.weight.") + env_mode).c_str());

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_gate_ne0, ne1);
    ggml_set_name(input, (std::string("gate.input.") + env_mode).c_str());

    // The single gated op.  Deliberately large: its own kernel duration is what
    // keeps the completion event in flight when unpin_on_event parks it.
    ggml_tensor * out = ggml_mul(ctx, input, weight);
    ggml_set_name(out, (std::string("gate.out.") + env_mode).c_str());

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, out);

    // Weight in its own DEVICE buffer.  Usage must be set BEFORE the alloc: it is
    // what init_tensor keys `ggml_sycl_tensor_is_weight` off, and therefore what
    // gets tensor->layout initialised.
    const size_t          weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t weight_buf      = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    if (!weight_buf) {
        fprintf(stderr, "[gate %s] failed to allocate device weight buffer\n", env_mode);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }
    ggml_backend_buffer_set_usage(weight_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buf, weight, ggml_backend_buffer_get_base(weight_buf));

    ggml_gallocr_t galloc = ggml_gallocr_new(buft);
    ggml_gallocr_reserve(galloc, graph);
    ggml_gallocr_alloc_graph(galloc, graph);

    std::vector<float> weight_data(ggml_nelements(weight));
    for (size_t i = 0; i < weight_data.size(); ++i) {
        weight_data[i] = 1.0f + static_cast<float>(i % 7) * 0.01f;
    }
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));

    // One tensor_set for the whole input: set_tensor drains the device queue by
    // default, so uploading row by row would cost ne1 full drains.
    std::vector<float> input_data(static_cast<size_t>(k_gate_ne0) * static_cast<size_t>(ne1));
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = static_cast<float>(i % 13) * 0.02f;
    }
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    bool ok = true;

    // --- precondition (a)/(b)/(c): the state maybe_pin_cached needs ---
    const ggml_tensor_layout * layout = weight->layout;
    if (!layout || !layout->data_ptr) {
        fprintf(stderr,
                "[gate %s] weight has no usable tensor->layout (layout=%p data_ptr=%p): maybe_pin_cached would "
                "early-return and pin_count would stay 0\n",
                env_mode, (const void *) layout, layout ? layout->data_ptr : nullptr);
        ok = false;
    }

    const ggml_layout_mode     lmode = layout ? layout->mode : GGML_LAYOUT_AOS;
    const ggml_sycl_cache_id   key   = ggml_backend_sycl_get_weight_cache_key(weight, 0);
    ggml_sycl::unified_cache * cache = ggml_sycl::get_unified_cache_for_device(0);

    if (ok && !key.valid) {
        fprintf(stderr, "[gate %s] weight cache key is invalid\n", env_mode);
        ok = false;
    }
    if (ok && !cache) {
        fprintf(stderr, "[gate %s] no unified cache for device 0\n", env_mode);
        ok = false;
    }

    if (ok) {
        // Publish the weight's own device allocation as a non-owning READY entry,
        // exactly as set_tensor does for MoE expert weights.  This is the step the
        // old test had no equivalent of, and the reason pin_count was always 0.
        cache->register_ready(key, weight->data, lmode, ggml_nbytes(weight), ggml_sycl::cache_entry_type::DENSE_WEIGHT);
        if (!cache->is_cached(key, lmode)) {
            fprintf(stderr, "[gate %s] register_ready did not make the weight cache-resident\n", env_mode);
            ok = false;
        }
    }
    if (ok) {
        // Baseline: NOT pinned. Without this, a later `is_pinned() == true` would
        // not prove binbcast pinned it -- it could have arrived pinned.
        cache->unpin(key, lmode);
        if (cache->is_pinned(key, lmode)) {
            fprintf(stderr, "[gate %s] weight is pinned before compute; baseline is not clean\n", env_mode);
            ok = false;
        }
    }

    // --- run the gated op and read the lease state machine off the trace ---
    trace_findings found;
    double         compute_ms   = 0.0;
    bool           pinned_after = true;

    if (ok) {
        const off_t trace_mark = g_trace.mark();
        const auto  t0         = std::chrono::steady_clock::now();
        ggml_status status     = ggml_backend_graph_compute_async(backend, graph);
        ggml_backend_synchronize(backend);
        compute_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "[gate %s] graph compute failed (%d)\n", env_mode, (int) status);
            ok = false;
        } else {
            // Drain explicitly.  ggml_backend_sycl_synchronize only drains when
            // has_pending_deferred_frees() is true, and that predicate covers
            // deferred_frees_/deferred_host_frees_ but NOT inflight_unpins_ -- so a
            // parked lease release can outlive synchronize and wait for an
            // unrelated allocation to trigger a drain.  Safe here: synchronize has
            // cleared the eviction guard and waited on the queue.
            cache->process_deferred_frees_public();
            found        = scan_trace(g_trace.read_from(trace_mark), key.name_hash);
            pinned_after = cache->is_pinned(key, lmode);
        }
    }

    const char * expect_source = (mode == event_mode::REUSE) ? "kernel" : "submission";

    if (ok) {
        // 1. The path EXECUTED.  This is what was missing.
        if (!found.saw_unpin_event || found.pins < 1) {
            fprintf(stderr,
                    "[gate %s] no '[SYCL-BINBCAST] unpin event ... pins=N' with N >= 1 in the trace: maybe_pin_cached "
                    "did not populate pins[], so unpin_on_event was never called and the path still has no coverage.\n",
                    env_mode);
            ok = false;
        } else if (found.source != expect_source) {
            fprintf(stderr, "[gate %s] live op resolved event source '%s', expected '%s'\n", env_mode,
                    found.source.c_str(), expect_source);
            ok = false;
        }
        // 2. unpin_on_event reached the cache for THIS weight.
        if (!found.saw_park) {
            fprintf(stderr,
                    "[gate %s] no '[UNIFIED-CACHE] in-flight pin' for this weight: the lease release was not parked on "
                    "an event.\n",
                    env_mode);
            ok = false;
        }
        // 3. The release came from the event-gated drain, not from some other path
        //    that bypasses the event entirely.
        if (found.released_early) {
            fprintf(stderr,
                    "[gate %s] the lease was released without the event-gated drain ever polling it: the release came "
                    "from a path that bypasses the gating event.\n",
                    env_mode);
            ok = false;
        }
        // 4. And it was released, so 3 is not satisfied by a leak.
        if (!found.saw_release) {
            fprintf(stderr,
                    "[gate %s] no '[UNIFIED-CACHE] in-flight unpin' for this weight: the lease was never released, so "
                    "it is leaked rather than deferred.\n",
                    env_mode);
            ok = false;
        }
        if (pinned_after) {
            fprintf(stderr, "[gate %s] weight is still pinned after the gated op completed\n", env_mode);
            ok = false;
        }
    }

    // --- result check: the op the lease was protecting produced the right data ---
    if (ok) {
        std::vector<float> out_row(k_gate_ne0);
        ggml_backend_tensor_get(out, out_row.data(), 0, out_row.size() * sizeof(float));
        int bad = 0;
        for (int64_t i0 = 0; i0 < k_gate_ne0 && bad < 8; ++i0) {
            const float expected = input_data[i0] * weight_data[i0];
            const float diff     = std::fabs(out_row[i0] - expected);
            if (diff > 1e-5f * (1.0f + std::fabs(expected))) {
                fprintf(stderr, "[gate %s] mismatch at [%lld]: got %.9g expected %.9g\n", env_mode, (long long) i0,
                        out_row[i0], expected);
                ++bad;
            }
        }
        if (bad > 0) {
            ok = false;
        }
    }

    // --- stage 2: drive unpin_on_event directly with a provably in-flight event ---
    //
    // This is the assertion that the lease is released only AFTER the gating event
    // completes.  It does not read the completion poll (which is degraded on this
    // driver -- see the note above); it measures whether the drain blocks until the
    // GPU is done, which is what the `wait_and_throw()` in the drain guarantees.
    double busy_remaining_ms = 0.0;
    double drain_ms          = 0.0;
    bool   pinned_after_park = false;
    bool   pinned_in_flight  = false;
    bool   pinned_at_end     = true;
    bool   stage2_ran        = false;

    if (ok) {
        sycl::queue & q       = cache->get_dma_queue();
        float *       scratch = static_cast<float *>(out->data);  // consumed above; safe to overwrite
        try {
            // Warm up first (the initial submit pays JIT / command-list creation and
            // would blow up the calibration), then size `iters` for this device.
            submit_busy_kernel(q, scratch, k_busy_items, 1).wait();
            const auto t_cal = std::chrono::steady_clock::now();
            submit_busy_kernel(q, scratch, k_busy_items, k_busy_cal_iters).wait();
            const double cal_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_cal).count();
            int iters = k_busy_max_iters;
            if (cal_ms > 0.05) {
                const double scaled = static_cast<double>(k_busy_cal_iters) * (k_busy_target_ms / cal_ms);
                iters               = scaled < 1.0 ? 1 : static_cast<int>(scaled);
            }
            if (iters > k_busy_max_iters) {
                iters = k_busy_max_iters;
            }

            cache->pin(key, lmode);
            if (!cache->is_pinned(key, lmode)) {
                fprintf(stderr, "[gate %s] stage 2: could not re-pin the entry\n", env_mode);
                ok = false;
            } else {
                sycl::event busy = submit_busy_kernel(q, scratch, k_busy_items, iters);
                cache->unpin_on_event(key, lmode, busy);

                // unpin_on_event must PARK the release, never perform it.
                pinned_after_park = cache->is_pinned(key, lmode);

                // A drain while the gating event is in flight must leave the pin alone.
                const auto t_drain = std::chrono::steady_clock::now();
                cache->process_deferred_frees_public();
                drain_ms =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_drain).count();
                pinned_in_flight = cache->is_pinned(key, lmode);

                // How much of the kernel was still outstanding during that drain.
                const auto t_rest = std::chrono::steady_clock::now();
                q.wait();
                busy_remaining_ms =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_rest).count();

                // And now the release must happen.
                cache->process_deferred_frees_public();
                pinned_at_end = cache->is_pinned(key, lmode);
                stage2_ran    = true;
            }
        } catch (const sycl::exception & e) {
            fprintf(stderr, "[gate %s] stage 2: SYCL exception: %s\n", env_mode, e.what());
            ok = false;
        }
    }

    if (ok && stage2_ran) {
        if (!pinned_after_park) {
            fprintf(stderr,
                    "[gate %s] stage 2: unpin_on_event released the lease SYNCHRONOUSLY; it must only park the release "
                    "on the event.\n",
                    env_mode);
            ok = false;
        }
        // Non-vacuity FIRST: without a real in-flight window the next check is empty.
        if (busy_remaining_ms < k_busy_remaining_ms) {
            fprintf(stderr,
                    "[gate %s] stage 2: only %.1f ms of the busy kernel was left after the drain (want >= %.1f ms), so "
                    "'still pinned' would not prove the event was in flight. Raise k_busy_target_ms.\n",
                    env_mode, busy_remaining_ms, k_busy_remaining_ms);
            ok = false;
        } else if (!pinned_in_flight) {
            fprintf(
                stderr,
                "[gate %s] stage 2: EARLY UNPIN. A drain released the lease while the gating event still had %.1f ms "
                "of GPU work outstanding. The weight can be evicted while the GPU is still reading it.\n",
                env_mode, busy_remaining_ms);
            ok = false;
        }
        if (pinned_at_end) {
            fprintf(stderr,
                    "[gate %s] stage 2: the lease was still pinned after the gating event completed and the cache was "
                    "drained: leaked, not deferred.\n",
                    env_mode);
            ok = false;
        }
    }

    printf(
        "[gate %s] stage1: pins=%d source=%s released=%d bypass=%d compute=%.1f ms rows=%lld (poll ready=%d, always "
        "1 on a profiling queue) | stage2: parked=%d held_in_flight=%d remaining=%.1f ms drain=%.1f ms released=%d: "
        "%s\n",
        env_mode, found.pins, found.source.empty() ? "-" : found.source.c_str(), found.saw_release ? 1 : 0,
        found.released_early ? 1 : 0, compute_ms, (long long) ne1, found.park_ready ? 1 : 0, pinned_after_park ? 1 : 0,
        pinned_in_flight ? 1 : 0, busy_remaining_ms, drain_ms, pinned_at_end ? 0 : 1, ok ? "PASS" : "FAIL");

    // Drop the non-owning entry before the buffer backing it goes away.
    if (cache && key.valid) {
        cache->remove(key, ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, lmode);
    }

    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(weight_buf);
    ggml_free(ctx);
    ggml_backend_free(backend);

    return ok;
}

static bool run_binbcast_stress(event_mode mode, int iters) {
    const char * env_mode = mode_name(mode);
    setenv("GGML_SYCL_BINBCAST_EVENT_MODE", env_mode, 1);

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        fprintf(stderr, "Failed to init SYCL backend\n");
        return false;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    ggml_backend_buffer_type_t host_buft = ggml_backend_sycl_host_buffer_type();
    if (!host_buft) {
        fprintf(stderr, "Failed to get SYCL host buffer type\n");
        ggml_backend_free(backend);
        return false;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Failed to init ggml context\n");
        ggml_backend_free(backend);
        return false;
    }

    const int64_t ne0 = 1024;
    const int64_t ne1 = 4;

    ggml_tensor * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
    ggml_set_name(weight, (std::string("weight.") + env_mode).c_str());

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    ggml_set_name(input, (std::string("input.") + env_mode).c_str());

    ggml_tensor * out = ggml_mul(ctx, input, weight);
    ggml_set_name(out, (std::string("out.") + env_mode).c_str());

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, out);

    const size_t weight_buf_size = ggml_backend_buft_get_alloc_size(host_buft, weight);
    ggml_backend_buffer_t weight_buf = ggml_backend_buft_alloc_buffer(host_buft, weight_buf_size);
    if (!weight_buf) {
        fprintf(stderr, "Failed to allocate weight buffer\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }
    ggml_backend_buffer_set_usage(weight_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buf, weight, ggml_backend_buffer_get_base(weight_buf));

    ggml_gallocr_t galloc = ggml_gallocr_new(buft);
    ggml_gallocr_reserve(galloc, graph);
    ggml_gallocr_alloc_graph(galloc, graph);

    std::vector<float> weight_data(ggml_nelements(weight));
    for (size_t i = 0; i < weight_data.size(); ++i) {
        weight_data[i] = 1.0f + static_cast<float>(i % 7) * 0.01f;
    }
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));

    std::vector<float> input_data(ggml_nelements(input));
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = static_cast<float>(i % 13) * 0.02f;
    }
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    // Report -- do not assert -- that this loop cannot reach unpin_on_event.  The
    // host-buffer weight carries no tensor->layout and is not cache-resident, so
    // maybe_pin_cached early-returns and pin_count stays 0 for every iteration and
    // every event mode.  Printed so a green stress result is never misread as
    // lease coverage; run_unpin_lease_gate() is what covers that.
    {
        const ggml_tensor_layout *       stress_layout = weight->layout;
        const ggml_sycl_cache_id         stress_key    = ggml_backend_sycl_get_weight_cache_key(weight, 0);
        const ggml_sycl::unified_cache * stress_cache  = ggml_sycl::get_unified_cache_for_device(0);
        const bool                       stress_cached =
            stress_cache && stress_key.valid &&
            stress_cache->is_cached(stress_key, stress_layout ? stress_layout->mode : GGML_LAYOUT_AOS);
        printf("[stress %s] host-buffer weight: layout=%s cached=%d -> reaches unpin_on_event: no\n", env_mode,
               (stress_layout && stress_layout->data_ptr) ? "yes" : "no", stress_cached ? 1 : 0);
    }

    bool ok = true;
    for (int i = 0; i < iters; ++i) {
        // Re-upload `input` every iteration. `ggml_gallocr` allocates `out`
        // IN-PLACE over `input` (same shape and type, and the mul is input's only
        // consumer), so without this each compute multiplies the PREVIOUS result
        // again and the tensor ends up holding input*weight^iters, which no fixed
        // expected value can check. Resetting it makes every iteration's result
        // exactly input*weight while still driving `iters` compute cycles.
        ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

        const ggml_status status = ggml_backend_graph_compute(backend, graph);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "[%s] graph compute failed at iter %d (%d)\n", env_mode, i, (int) status);
            ok = false;
            break;
        }
    }

    ggml_backend_sycl_submit_barrier(backend);
    ggml_backend_synchronize(backend);

    // Verify the result. An early pin release lets the cache evict or overwrite a
    // weight the kernel is still reading, which shows up here as wrong output --
    // the only symptom this test can observe from outside the backend. Throughput
    // alone would read an early unpin as a win.
    if (ok) {
        std::vector<float> out_data(ggml_nelements(out));
        ggml_backend_tensor_get(out, out_data.data(), 0, out_data.size() * sizeof(float));

        int bad = 0;
        for (int64_t i1 = 0; i1 < ne1 && bad < 8; ++i1) {
            for (int64_t i0 = 0; i0 < ne0 && bad < 8; ++i0) {
                const size_t idx      = static_cast<size_t>(i1) * static_cast<size_t>(ne0) + static_cast<size_t>(i0);
                const float  expected = input_data[idx] * weight_data[i0];
                const float  actual   = out_data[idx];
                const float  diff     = std::fabs(actual - expected);
                if (diff > 1e-5f * (1.0f + std::fabs(expected))) {
                    fprintf(stderr, "[%s] mismatch at [%lld,%lld]: got %.9g expected %.9g\n", env_mode, (long long) i0,
                            (long long) i1, actual, expected);
                    ++bad;
                }
            }
        }
        if (bad > 0) {
            ok = false;
        }
    }

    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(weight_buf);
    ggml_free(ctx);
    ggml_backend_free(backend);

    return ok;
}

}  // namespace

int main(int argc, char ** argv) {
    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {
        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:0", 1);
    }

    // Graph replay makes the per-op pin/unpin path unreachable: while recording,
    // `maybe_pin_cached` returns early because pin ops are not graph-recordable,
    // and on replay the op is not dispatched at all.  The flag is latched inside
    // ggml_check_sycl, so it has to be set before the first backend init -- after
    // that, setting it has no effect.  Covering unpin_on_event therefore requires
    // eager dispatch for the whole process.
    setenv("GGML_SYCL_DISABLE_GRAPH", "1", 1);

    // Same story for the debug level: `ggml_check_sycl` latches it at the first
    // backend init from an override that this setter writes.  It has to be on
    // before then, and it is what emits the four trace lines the gate asserts on.
    // Set via the API rather than GGML_SYCL_DEBUG=1 because the backend unsets
    // that variable at init (it can suppress Level-Zero device discovery).
    ggml_backend_sycl_set_debug(1);

    const char * mode_arg  = get_arg(argc, argv, "--mode=");
    const char * iters_arg = get_arg(argc, argv, "--iters=");

    event_mode mode = parse_mode(mode_arg ? mode_arg : std::getenv("GGML_SYCL_UNPIN_EVENT_MODE"));
    int        iters = parse_iters(iters_arg ? iters_arg : std::getenv("GGML_SYCL_UNPIN_ITERS"), 200);

    // Host-only, so it runs first and reports even if device work later fails.
    bool ok = check_event_source_policy();

    if (!g_trace.begin()) {
        fprintf(stderr,
                "FAIL: could not redirect stderr to capture the SYCL trace. The lease gate reads its assertions from "
                "that trace, so without it the gate could only pass vacuously.\n");
        return 1;
    }

    // The backend's own startup banner now goes into the capture, and this host has
    // two discrete GPUs whose numbering is not intuitive, so name the card on stdout.
    // tests/CMakeLists.txt pins ONEAPI_DEVICE_SELECTOR=level_zero:1 (the Arc Pro B50
    // validation card) precisely because device 0 unfiltered is the measurement card.
    {
        char desc[256] = { 0 };
        ggml_backend_sycl_get_device_description(0, desc, sizeof(desc));
        printf("Device 0 after ONEAPI_DEVICE_SELECTOR filtering: %s\n", desc);
    }

    if (mode == event_mode::COMPARE) {
        printf("Mode: compare (safe, barrier, reuse), iters=%d\n", iters);
        const event_mode modes[] = { event_mode::SAFE, event_mode::BARRIER, event_mode::REUSE };
        for (event_mode m : modes) {
            if (!ok) {
                break;
            }
            ok = run_unpin_lease_gate(m);
        }
        for (event_mode m : modes) {
            if (!ok) {
                break;
            }
            ok = run_binbcast_stress(m, iters);
        }
    } else if (ok) {
        printf("Mode: %s, iters=%d\n", mode_name(mode), iters);
        ok = run_unpin_lease_gate(mode);
        if (ok) {
            ok = run_binbcast_stress(mode, iters);
        }
    }

    g_trace.end();
    if (!ok) {
        g_trace.dump_tail(200);
    }

    printf("\nUnified cache unpin event test: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

#endif
