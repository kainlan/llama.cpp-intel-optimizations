#pragma once

#include "ggml.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

namespace ggml_sycl {

enum class e2e_tg_stage : uint8_t {
    DISPATCH = 0,
    CPU_DISPATCH,
    NON_MOE_MATMUL,
    MOE,
    ATTENTION,
    KV,
    ELEMENTWISE,
    GRAPH,
    CACHE,
    TRANSFER,
    OTHER,
    COUNT,
};

struct e2e_tg_stage_accum {
    uint64_t    calls     = 0;
    double      host_us   = 0.0;
    // HOST-ONLY INSTRUMENT: no call site supplies a device timestamp. The
    // accumulator and its printed field work (test-sycl-e2e-profile.cpp feeds
    // and reads back a non-zero value), and the one non-zero producer in the
    // tree is a host-clock span -- e2e_tg_profile_record_transfer(
    // "peer_host_bounce_measure", ...) passes the chrono time of a
    // host-mediated peer-link bounce copy as `device_us` -- so this field is
    // not universally 0.0, but it is never a SYCL device timestamp. Reading
    // sycl::event::get_profiling_info<command_start/command_end> at the
    // per-op e2e_tg_scope bracket would be an implicit host wait on an
    // incomplete event, which the no-host-waits rule forbids on the dispatch
    // path (the assert_no_waits source guards in
    // tests/test-sycl-e2e-profile-compute-forward-source.py and
    // -fattn-source.py fail that region on any wait). GGML_SYCL_KERNEL_PROFILE
    // (sycl-kernel-profiler.hpp) gets real device time by deferring the
    // profiling read to its own flush point; this instrument has no such
    // mechanism. Use the kernel profiler for device time; treat `device_us`
    // here as host-only unless a caller documents otherwise.
    double      device_us = 0.0;
    uint64_t    bytes     = 0;
    std::string last_path = "unknown";
};

struct e2e_tg_profile_snapshot {
    uint64_t                                                                 tokens    = 0;
    uint64_t                                                                 ops       = 0;
    uint64_t                                                                 moe_calls = 0;
    // P4 TG-cost-visibility (llama.cpp-os8k): wall-clock time between this
    // graph_compute's start and the PREVIOUS one's start, i.e. one full
    // token's (TG) or one ubatch's (PP) host-observed wall time -- distinct
    // from `stages[*].host_us`, which sums only the e2e_tg_scope-bracketed
    // host spans and can be less than true wall time (unbracketed work,
    // scheduling gaps between them) or double-count (nested scopes). Zero
    // on the very first graph_compute of a run (no previous timestamp to
    // diff against).
    double                                                                   wall_us   = 0.0;
    std::array<e2e_tg_stage_accum, static_cast<size_t>(e2e_tg_stage::COUNT)> stages{};
};

bool         e2e_tg_profile_enabled_from_env(const char * env);
bool         e2e_tg_profile_enabled();
const char * e2e_tg_stage_name(e2e_tg_stage stage);
e2e_tg_stage e2e_tg_stage_from_op(ggml_op op, const char * tensor_name);
// `device_us`: see e2e_tg_stage_accum::device_us above -- no caller supplies
// a device timestamp today (the one non-zero producer forwards a host chrono
// span through e2e_tg_profile_record_transfer). A future caller with a
// non-blocking way to measure device time (a deferred/pending-event flush
// like the kernel profiler's) can supply one without further plumbing.
void         e2e_tg_profile_record(e2e_tg_stage stage,
                                   const char * path,
                                   double       host_us,
                                   double       device_us = 0.0,
                                   uint64_t     bytes     = 0,
                                   uint64_t     calls     = 1);
void         e2e_tg_profile_record_cache_event(const char * path, uint64_t bytes, double host_us);
void         e2e_tg_profile_record_transfer(const char * path, uint64_t bytes, double host_us, double device_us);
void         e2e_tg_profile_flush_if_ready(FILE * out = stderr);
// P4 TG-cost-visibility (llama.cpp-os8k): call once at the START of every
// ggml_backend_sycl_graph_compute() (same call site as
// sycl_timeline_note_graph_compute()) to flush the PREVIOUS graph's
// accumulated per-stage host/device split before the next one starts
// accumulating. Unlike e2e_tg_profile_flush_if_ready (gated on
// moe_calls>=72, so it never fires mid-run for a dense/non-MoE model --
// Mistral, gemma -- silently making "tokens" and per-token arithmetic wrong
// by the whole run's token count), this is model-shape-independent: one
// graph_compute call is one token during TG and one ubatch during PP, so
// each flush corresponds to exactly one of those. No-op (and no snapshot
// copy) when the profile is disabled.
void         e2e_tg_profile_note_new_graph_compute(FILE * out = stderr);
void         e2e_tg_profile_force_flush(FILE * out = stderr);
#if defined(GGML_SYCL_PRIVATE_TESTING)
void         e2e_tg_profile_reset_for_tests();
void         e2e_tg_profile_flush_for_tests(FILE * out);
#endif
// Read-only diagnostic snapshot: safe ABI allowlist entry.
e2e_tg_profile_snapshot e2e_tg_profile_snapshot_for_tests();

// llama.cpp-qmen: this scope is HOST-CLOCK ONLY -- its destructor always
// records device_us=0.0 (see e2e_tg_stage_accum::device_us above for why
// that is a deliberate, documented limitation rather than an oversight).
// host_us is real and is the primary signal this instrument provides.
class e2e_tg_scope {
  public:
    e2e_tg_scope(e2e_tg_stage stage, const char * path, bool enabled = e2e_tg_profile_enabled()) :
        enabled_(enabled),
        stage_(stage) {
        if (enabled_) {
            path_  = path && path[0] != '\0' ? path : "unknown";
            start_ = clock::now();
        }
    }

    ~e2e_tg_scope() {
        if (!enabled_) {
            return;
        }
        const auto   end     = clock::now();
        const double host_us = std::chrono::duration<double, std::micro>(end - start_).count();
        e2e_tg_profile_record(stage_, path_.c_str(), host_us, 0.0, 0, 1);
    }

    e2e_tg_scope(const e2e_tg_scope &)             = delete;
    e2e_tg_scope & operator=(const e2e_tg_scope &) = delete;

  private:
    using clock = std::chrono::high_resolution_clock;

    bool              enabled_ = false;
    e2e_tg_stage      stage_   = e2e_tg_stage::OTHER;
    std::string       path_;
    clock::time_point start_{};
};

}  // namespace ggml_sycl
