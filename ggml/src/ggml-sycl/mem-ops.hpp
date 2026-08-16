// Canonical event-returning memory operations over mem_handle operands.

#pragma once

#include "mem-handle.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ggml_sycl {

sycl::event mem_copy_async(const mem_handle &               dst,
                           const mem_handle &               src,
                           size_t                           size,
                           sycl::queue &                    queue,
                           const std::vector<sycl::event> & deps = {});

sycl::event mem_copy_async(const mem_handle &               dst,
                           size_t                           dst_offset,
                           const mem_handle &               src,
                           size_t                           src_offset,
                           size_t                           size,
                           sycl::queue &                    queue,
                           const std::vector<sycl::event> & deps = {});

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
              const std::vector<sycl::event> & deps = {});

void mem_copy(const mem_handle &               dst,
              size_t                           dst_offset,
              const mem_handle &               src,
              size_t                           src_offset,
              size_t                           size,
              sycl::queue &                    queue,
              const std::vector<sycl::event> & deps = {});

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
