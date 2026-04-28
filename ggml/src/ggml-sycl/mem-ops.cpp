#include "mem-ops.hpp"

#include "ggml-impl.h"

#include <cstring>
#include <utility>

namespace ggml_sycl {

static void add_deps(sycl::handler & cgh, const std::vector<sycl::event> & deps) {
    if (!deps.empty()) {
        cgh.depends_on(deps);
    }
}

static sycl::event mem_copy_submit(const mem_handle &               dst,
                                   const mem_handle &               src,
                                   size_t                           size,
                                   sycl::queue &                    queue,
                                   const std::vector<sycl::event> & deps,
                                   bool                             retain_until_event) {
    resolved_ptr d = dst.resolve();
    resolved_ptr s = src.resolve();
    GGML_ASSERT(d && s && "mem_copy_async on unresolved handle");

    sycl::event event;
    if (d.on_device || s.on_device) {
        event = queue.submit([&](sycl::handler & cgh) {
            add_deps(cgh, deps);
            cgh.memcpy(d.ptr, s.ptr, size);
        });
    } else {
        event = queue.submit([&](sycl::handler & cgh) {
            add_deps(cgh, deps);
            cgh.host_task([dst_ptr = d.ptr, src_ptr = s.ptr, size]() { std::memcpy(dst_ptr, src_ptr, size); });
        });
    }

    if (retain_until_event) {
        retain_handles_until_event({ dst, src }, event);
    }
    return event;
}

static sycl::event mem_fill_submit(const mem_handle &               h,
                                   int                              value,
                                   size_t                           size,
                                   sycl::queue &                    queue,
                                   const std::vector<sycl::event> & deps,
                                   bool                             retain_until_event) {
    resolved_ptr r = h.resolve();
    GGML_ASSERT(r && "mem_fill_async on unresolved handle");

    sycl::event event;
    if (r.on_device) {
        event = queue.submit([&](sycl::handler & cgh) {
            add_deps(cgh, deps);
            cgh.memset(r.ptr, value, size);
        });
    } else {
        event = queue.submit([&](sycl::handler & cgh) {
            add_deps(cgh, deps);
            cgh.host_task([ptr = r.ptr, value, size]() { std::memset(ptr, value, size); });
        });
    }

    if (retain_until_event) {
        retain_handles_until_event({ h }, event);
    }
    return event;
}

sycl::event mem_copy_async(const mem_handle &               dst,
                           const mem_handle &               src,
                           size_t                           size,
                           sycl::queue &                    queue,
                           const std::vector<sycl::event> & deps) {
    return mem_copy_submit(dst, src, size, queue, deps, true);
}

sycl::event mem_fill_async(const mem_handle &               h,
                           int                              value,
                           size_t                           size,
                           sycl::queue &                    queue,
                           const std::vector<sycl::event> & deps) {
    return mem_fill_submit(h, value, size, queue, deps, true);
}

void mem_copy(const mem_handle &               dst,
              const mem_handle &               src,
              size_t                           size,
              sycl::queue &                    queue,
              const std::vector<sycl::event> & deps) {
    mem_copy_submit(dst, src, size, queue, deps, false).wait_and_throw();
}

void mem_fill(const mem_handle &               h,
              int                              value,
              size_t                           size,
              sycl::queue &                    queue,
              const std::vector<sycl::event> & deps) {
    mem_fill_submit(h, value, size, queue, deps, false).wait_and_throw();
}

}  // namespace ggml_sycl
