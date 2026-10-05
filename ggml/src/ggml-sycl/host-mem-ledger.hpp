//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// host-mem-ledger.hpp -- per-consumer host-memory breakdown (llama.cpp-z5fn).
//
// RssAnon cannot say which consumer holds the bytes, and the pinned pool's own
// counters only say how many chunks exist. This ledger counts the bytes the
// backend knowingly puts in host memory per consumer, and
// ggml_sycl_log_host_mem() prints them next to the process counters and the
// pinned pool's zone figures, with an explicit residual, so one WARN line per
// phase can be scored against RssAnon.
//
// WARN, not INFO: GGML_LOG_INFO is dropped at default verbosity in every tool.
//
// Cost model. The default is two cheap lines per run (load end, first PP->TG),
// reading only /proc/self/status, /proc/meminfo, the ledger and the pool
// counters. The expensive readers -- /proc/self/smaps (a page-table walk that
// holds mmap_lock on a ~70 GB process) and mallinfo2() (takes every arena lock)
// -- and the per-graph_compute ladder run only with GGML_SYCL_HOSTMEM=1.
// host_mem_plan_for() is the single decision point, and it is pure so a
// host-only test can pin it.

#pragma once

#include <atomic>
#include <cstddef>

namespace ggml_sycl {

struct host_mem_ledger {
    // Live SYCL_Host-family buffers (at load: every non-CPU weight, device-tier and host-tier).
    std::atomic<size_t> sycl_host_buffer_bytes{ 0 };
    // Host-tier expert / dense weights copied into a second pinned allocation. CUMULATIVE: added at
    // load, never decremented (the copies are released through cache eviction, which does not
    // report back here), so they are not a live figure like their siblings.
    std::atomic<size_t> host_expert_copy_cumulative_bytes{ 0 };
    std::atomic<size_t> host_dense_copy_cumulative_bytes{ 0 };
    // Per-thread CPU-dispatch weight-dequant scratch (cpu_dispatch_buffers::scratch_nk), all threads.
    std::atomic<size_t> cpu_dispatch_scratch_bytes{ 0 };
};

enum class host_mem_phase {
    LOAD_END,         // end of model load
    PP_TO_TG_BEFORE,  // first/each PP->TG transition, before refresh_moe_after_pp
    PP_TO_TG_AFTER,   // same transition, after the refresh
    LADDER,           // graph_compute call-count ladder (power-of-two call numbers)
};

struct host_mem_plan {
    bool emit;          // write a line at all
    bool full;          // include smaps + mallinfo2 (the expensive readers)
    bool rate_limited;  // apply the per-interval rate limit
};

// first_pp_to_tg: this is the first PP_TO_TG_BEFORE call of the process.
inline host_mem_plan host_mem_plan_for(host_mem_phase phase, bool full_mode, bool first_pp_to_tg) {
    switch (phase) {
        case host_mem_phase::LOAD_END:
            return { true, full_mode, false };
        case host_mem_phase::PP_TO_TG_BEFORE:
            // Default: once. Full: every transition, rate limited.
            return { full_mode || first_pp_to_tg, full_mode, full_mode };
        case host_mem_phase::PP_TO_TG_AFTER:
        case host_mem_phase::LADDER:
            return { full_mode, full_mode, false };
    }
    return { false, false, false };
}

inline host_mem_ledger & host_mem_ledger_get() {
    static host_mem_ledger ledger;
    return ledger;
}

}  // namespace ggml_sycl

// GGML_SYCL_HOSTMEM=1 selects the full diagnostic mode (read once).
bool ggml_sycl_host_mem_full();

// Prints one `[HOSTMEM] <label> ...` WARN line when host_mem_plan_for() says so.
// Safe at any phase; never constructs a cache. Returns whether it logged.
bool ggml_sycl_log_host_mem(ggml_sycl::host_mem_phase phase, const char * label);
