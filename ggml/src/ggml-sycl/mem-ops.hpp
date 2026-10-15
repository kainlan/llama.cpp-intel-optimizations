// Canonical event-returning memory operations over mem_handle operands.

#pragma once

#include "mem-handle.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ggml_sycl {

// mem_copy_async and mem_copy default-capture their CALLER's file, line and
// function (llama.cpp-th86). Every copy below reaches mem_copy_direct_submit
// from inside mem-ops.cpp, so a site captured there would always name
// mem-ops.cpp; capturing it here names the code that asked for the copy, which
// is what the GGML_SYCL_MEMCPY_SITE_CENSUS table and the kernel profiler's
// callsite need. A wrapper that copies on behalf of its own callers should take
// the same three defaulted parameters and pass them on.
sycl::event mem_copy_async(const mem_handle &               dst,
                           const mem_handle &               src,
                           size_t                           size,
                           sycl::queue &                    queue,
                           const std::vector<sycl::event> & deps     = {},
                           const char *                     file     = __builtin_FILE(),
                           int                              line     = __builtin_LINE(),
                           const char *                     function = __builtin_FUNCTION());

// Async copy over raw transient views (canonical contract: raw pointers are
// ABI views resolved from a mem_handle for immediate submission only). The
// caller MUST hold the owning mem_handle(s) for both endpoints until the
// returned event completes; this is the single sanctioned site for
// pointer-level queue copies outside common.hpp/unified-cache.cpp.
sycl::event mem_copy_ptr_async(void *                           dst,
                               const void *                     src,
                               size_t                           size,
                               sycl::queue &                    queue,
                               const std::vector<sycl::event> & deps = {});

sycl::event mem_copy_async(const mem_handle &               dst,
                           size_t                           dst_offset,
                           const mem_handle &               src,
                           size_t                           src_offset,
                           size_t                           size,
                           sycl::queue &                    queue,
                           const std::vector<sycl::event> & deps     = {},
                           const char *                     file     = __builtin_FILE(),
                           int                              line     = __builtin_LINE(),
                           const char *                     function = __builtin_FUNCTION());

sycl::event mem_fill_async(const mem_handle &               h,
                           int                              value,
                           size_t                           size,
                           sycl::queue &                    queue,
                           const std::vector<sycl::event> & deps = {});

sycl::event mem_fill_async(const mem_handle &               h,
                           size_t                           offset,
                           int                              value,
                           size_t                           size,
                           sycl::queue &                    queue,
                           const std::vector<sycl::event> & deps = {});

#if defined(GGML_SYCL_PRIVATE_TESTING)
uint64_t mem_fill_test_profile_error_after_submit_count();
void     mem_fill_set_profile_error_after_submit_for_test(bool enabled);
#endif

void mem_copy(const mem_handle &               dst,
              const mem_handle &               src,
              size_t                           size,
              sycl::queue &                    queue,
              const std::vector<sycl::event> & deps     = {},
              const char *                     file     = __builtin_FILE(),
              int                              line     = __builtin_LINE(),
              const char *                     function = __builtin_FUNCTION());

void mem_copy(const mem_handle &               dst,
              size_t                           dst_offset,
              const mem_handle &               src,
              size_t                           src_offset,
              size_t                           size,
              sycl::queue &                    queue,
              const std::vector<sycl::event> & deps     = {},
              const char *                     file     = __builtin_FILE(),
              int                              line     = __builtin_LINE(),
              const char *                     function = __builtin_FUNCTION());

void mem_fill(const mem_handle &               h,
              int                              value,
              size_t                           size,
              sycl::queue &                    queue,
              const std::vector<sycl::event> & deps = {});

void mem_fill(const mem_handle &               h,
              size_t                           offset,
              int                              value,
              size_t                           size,
              sycl::queue &                    queue,
              const std::vector<sycl::event> & deps = {});

// Diagnostic for llama.cpp-480a, off unless GGML_SYCL_STAGE_TRACE=1.
//
// Emit a boundary marker so pinned-staging occupancy can be read PER CASE
// rather than as one undifferentiated stream.  The whole question is whether
// staging returns to baseline between units of work (a high-water / drain
// cadence problem) or climbs monotonically (a real leak), and those two have
// opposite fixes -- so the trace is useless without something to segment it.
//
// No-op and allocation-free when tracing is disabled.
void stage_trace_mark(const char * tag);

// Diagnostic for llama.cpp-th86, off unless GGML_SYCL_MEMCPY_SITE_CENSUS=1.
//
// Print the per-call-site copy census to stderr: one row per (caller
// file:line, direction, profile path) with its call and byte counts, sorted by
// calls. Every copy that reaches mem_copy_direct_submit is counted, including
// the host-host std::memcpy branch (direction H2H), which the kernel profiler
// never sees. The table is cumulative over the process. The SYCL backend's free
// path calls this right after the kernel-profile flush, and the first recorded
// copy registers an exit printer that prints again only if copies were
// recorded since the last print. No-op when the census is disabled.
void mem_copy_site_census_flush(const char * reason);

// The synchronous waits mem_copy_submit makes on the calling thread, by kind (llama.cpp-qwn5): the staged
// host->device copy's wait for its deps (H2D_DEPS), for the previous chunk before it refills the stage (H2D_CHUNK)
// and for the last chunk when the caller does not retain the copy (H2D_FINAL); the staged device->host copy's wait
// for each chunk (D2H); the host->host copy's wait for its deps (H2H_DEPS); and the cross-device copy's waits
// (CROSS). Each one holds the thread until the device or the dependency has caught up.
enum mem_copy_staged_wait_kind {
    MEM_COPY_STAGED_WAIT_KIND_H2D_DEPS = 0,
    MEM_COPY_STAGED_WAIT_KIND_H2D_CHUNK,
    MEM_COPY_STAGED_WAIT_KIND_H2D_FINAL,
    MEM_COPY_STAGED_WAIT_KIND_D2H,
    MEM_COPY_STAGED_WAIT_KIND_H2H_DEPS,
    MEM_COPY_STAGED_WAIT_KIND_CROSS,
    MEM_COPY_STAGED_WAIT_KIND_COUNT,
};

// One (copy caller's file, line, kind) row of the census. `file` is the caller's __builtin_FILE() literal, which
// lives for the process; null marks a free slot.
struct mem_copy_staged_wait_site {
    const char * file  = nullptr;
    int          line  = 0;
    int          kind  = MEM_COPY_STAGED_WAIT_KIND_COUNT;
    uint64_t     count = 0;
    double       us    = 0.0;
};

constexpr size_t k_mem_copy_staged_wait_site_slots = 16;

// The calling thread's waits inside mem_copy_submit while ggml_sycl_cpu_expert_trace_enabled() (llama.cpp-qwn5): how
// many of each kind and their summed wall time, the same per copy-caller site in a fixed table (a site that finds no
// free slot counts into overflow), and how many copies took the staged branch in each direction. The submitting
// thread prints its own on each [MOE-HOSTPATH-WAITS] line and then resets it. Off, a wait costs one cached flag test
// and reads no clock.
struct mem_copy_staged_wait_census {
    uint64_t                  count[MEM_COPY_STAGED_WAIT_KIND_COUNT]   = {};
    double                    us[MEM_COPY_STAGED_WAIT_KIND_COUNT]      = {};
    mem_copy_staged_wait_site sites[k_mem_copy_staged_wait_site_slots] = {};
    uint64_t                  overflow_count                           = 0;
    double                    overflow_us                              = 0.0;
    uint64_t                  staged_d2h                               = 0;
    uint64_t                  staged_h2d                               = 0;
};

mem_copy_staged_wait_census & mem_copy_staged_wait_census_this_thread();

// The census as the [MOE-HOSTPATH-WAITS] line prints it: `copies` is d2h:N,h2d:N; `waits` is kind:N/USus for each
// kind that waited, comma separated; `sites` is file:line:kind=N/USus for each site that waited, then
// overflow=N/USus when the table was full. `waits` and `sites` read none when nothing waited.
struct mem_copy_staged_wait_text {
    char copies[64];
    char waits[256];
    char sites[1024];
};

void mem_copy_staged_wait_census_format(const mem_copy_staged_wait_census & census, mem_copy_staged_wait_text & out);

// Allocate a host-pinned staging buffer for a TERMINAL attempt -- one whose
// caller has no further fallback and will abort or fail if it returns false.
//
// Exported so payload-staging callers outside this TU share the ONE bounded
// retry (llama.cpp-480a / f8ws) rather than growing a second copy of it, and so
// they inherit the always-on `ok=0` [STAGE-TRACE] failure print: a site that
// hand-rolls its own alloc_request gets neither, which is how
// ggml-sycl.cpp:7070 came to abort on a 256-byte allocation while five sibling
// staging failures in the same run were rescued (llama.cpp-13u6).
//
// TERMINAL is in the name because the retry belongs only to a last attempt.  A
// caller that can try another device first should do that instead of sleeping
// on this one, so non-terminal sites deliberately pass no retries and are not
// served by this entry point.
//
// Unrelated to moe-mmid-workspace.hpp's `terminal` vocabulary (device_terminal,
// terminal_release, queue_submission_authority::mint_terminal), which is about
// submission-completion authority, not allocation attempts.  Same English word,
// different subsystem; there is no relationship to infer.
//
// ONLY THE ggml-sycl.cpp PAYLOAD-STAGING SITE USES THIS.  The four staging
// sites inside mem-ops.cpp still call the static alloc_pinned_stage_handle()
// directly with k_stage_alloc_retries, and that is deliberate rather than a
// half-finished migration: they are in the same TU as the static, so routing
// them through this exported wrapper would add a call layer and buy nothing.
// This entry point exists for callers that CANNOT see the static.  Do not
// "complete" the migration on the assumption that uniformity is the goal.
//
// Recovers from TRANSIENT pressure only -- it performs no drain, reap or
// eviction between attempts, because reclaiming memory that still has a live
// handle is forbidden by the unified-cache ownership contract.
bool alloc_pinned_stage_handle_terminal(size_t        size,
                                        sycl::queue & queue,
                                        int           device,
                                        const char *  cohort_id,
                                        bool          require_host_usm_base,
                                        mem_handle *  out);

}  // namespace ggml_sycl
