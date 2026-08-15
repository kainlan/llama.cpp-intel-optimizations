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

}  // namespace ggml_sycl
