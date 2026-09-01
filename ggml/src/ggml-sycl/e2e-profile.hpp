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
    // llama.cpp-qmen (S6/I1 profiler completeness, spike S6 of
    // docs/plans/2026-09-01-sycl-utilization-plan.md): the ACCUMULATOR and
    // its printed field are real and tested (test-sycl-e2e-profile.cpp feeds
    // and reads back a non-zero value end to end) -- what is missing is a
    // producer. Every production call site in the tree passes a literal 0.0
    // today, so in practice this field always reads 0.0. That is NOT
    // fixable by reading `sycl::event::get_profiling_info<command_start/
    // command_end>` at the call sites that matter most (the per-op
    // `e2e_tg_scope` bracket in ggml_sycl_compute_forward, which accounts
    // for the bulk of dispatched ops): querying profiling info on an event
    // that has not yet completed is an IMPLICIT HOST WAIT (see the
    // `mxfp4_pp_batched_profile_enabled()` comment in ggml-sycl.cpp, which
    // explicitly disables that instrument during graph recording for this
    // reason), and the no-host-waits architecture rule (CLAUDE.md's "No
    // host waits -- event-chain everything") forbids adding one to the
    // hot per-op dispatch path -- confirmed live by the `assert_no_waits`
    // guards in tests/test-sycl-e2e-profile-compute-forward-source.py and
    // -fattn-source.py, which fail this exact region on any `.wait(`-family
    // call. GGML_SYCL_KERNEL_PROFILE (sycl-kernel-profiler.hpp) solves the
    // same problem correctly by deferring the get_profiling_info() read to
    // its own flush/drain point instead of the submit site; this instrument
    // has no such deferred/pending-event mechanism, and adding one is a
    // real redesign, not a light touch -- out of scope for this task. Read
    // GGML_SYCL_KERNEL_PROFILE for genuine per-kernel device time (now
    // covering the oneDNN WOQ GEMM and RMS_NORM family as of this task);
    // treat every `device_us` value here as 0.0/host-only unless a specific
    // caller is documented otherwise.
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
// `device_us`: see e2e_tg_stage_accum::device_us above -- every current
// production caller passes 0.0 (the default). The parameter and its
// accumulation are exercised end to end by test-sycl-e2e-profile.cpp with a
// synthetic non-zero value, so a future caller that has a legitimately
// non-blocking way to measure device time (e.g. a deferred/pending-event
// flush like the kernel profiler's) can supply one without further plumbing.
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
