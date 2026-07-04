#include "mem-ops.hpp"

#include "common.hpp"
#include "ggml-impl.h"
#include "sycl-kernel-profiler.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ggml_sycl {

static void add_deps(sycl::handler & cgh, const std::vector<sycl::event> & deps) {
    if (!deps.empty()) {
        cgh.depends_on(deps);
    }
}

static bool queues_share_context(sycl::queue & a, sycl::queue & b) {
    try {
        return a.get_context() == b.get_context();
    } catch (...) {
        return false;
    }
}

static int queue_device_or_host(sycl::queue & queue) {
    try {
        return ggml_sycl_get_device_id_from_queue(queue);
    } catch (...) {
        return mem_handle::HOST_DEVICE;
    }
}

static ggml_sycl_profile_label make_memcpy_profile_label(sycl::queue & queue,
                                                         const char *  name,
                                                         const char *  metadata,
                                                         const char *  queue_kind,
                                                         size_t        bytes) {
    ggml_sycl_profile_label label{};
    label.name       = name;
    label.category   = "memory";
    label.queue_kind = queue_kind;
    label.metadata   = metadata;
    label.device     = queue_device_or_host(queue);
    label.bytes      = bytes;
    return label;
}

template <typename SubmitFn>
static sycl::event ggml_sycl_memcpy_profile_submit(sycl::queue & queue,
                                                   const char *  name,
                                                   const char *  metadata,
                                                   const char *  queue_kind,
                                                   size_t        bytes,
                                                   SubmitFn &&   submit_fn,
                                                   const char *  file     = __builtin_FILE(),
                                                   int           line     = __builtin_LINE(),
                                                   const char *  function = __builtin_FUNCTION()) {
    if (!ggml_sycl_kernel_profile_enabled()) {
        return submit_fn(queue);
    }

    ggml_sycl_profile_label label = make_memcpy_profile_label(queue, name, metadata, queue_kind, bytes);
    return ggml_sycl_profile_submit(queue, label, static_cast<SubmitFn &&>(submit_fn), file, line, function);
}

static sycl::queue & queue_for_device_or_fallback(int device, sycl::queue & fallback) {
    if (device >= 0) {
        if (unified_cache * cache = get_unified_cache_for_device(device)) {
            return cache->get_queue();
        }
        if (sycl::queue * q = get_shared_context_queue(device)) {
            return *q;
        }
        return ggml_sycl_get_device(device).default_queue();
    }
    return fallback;
}

static bool host_ptr_is_usm_accessible(const void * ptr) {
    const sycl::usm::alloc alloc = ggml_sycl_get_alloc_type(ptr);
    return alloc == sycl::usm::alloc::host || alloc == sycl::usm::alloc::shared;
}

static bool mem_copy_trace_enabled() {
    static const bool enabled = []() {
        const char * v = std::getenv("GGML_SYCL_MEM_COPY_TRACE");
        return v && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

static const char * alloc_type_name(alloc_type type) {
    switch (type) {
        case alloc_type::DEVICE:      return "device";
        case alloc_type::HOST_PINNED: return "host_pinned";
        case alloc_type::SHARED:      return "shared";
        case alloc_type::MMAP:        return "mmap";
        case alloc_type::UNKNOWN:     return "unknown";
    }
    return "unknown";
}

static const char * usm_alloc_name(sycl::usm::alloc type) {
    switch (type) {
        case sycl::usm::alloc::host:    return "host";
        case sycl::usm::alloc::device:  return "device";
        case sycl::usm::alloc::shared:  return "shared";
        case sycl::usm::alloc::unknown: return "unknown";
    }
    return "unknown";
}

static sycl::usm::alloc probe_pointer_type_in_queue_context(const void * ptr, sycl::queue & queue) {
    if (!ptr) {
        return sycl::usm::alloc::unknown;
    }
    try {
        return sycl::get_pointer_type(ptr, queue.get_context());
    } catch (...) {
        return sycl::usm::alloc::unknown;
    }
}

static void trace_mem_copy_endpoint(const char * label,
                                    const char * endpoint,
                                    const mem_handle & handle,
                                    size_t offset,
                                    sycl::queue & queue) {
    if (!mem_copy_trace_enabled()) {
        return;
    }

    const int          queue_device = queue_device_or_host(queue);
    const resolved_ptr r            = handle.resolve(queue_device);
    const void *       ptr          = r ? static_cast<const char *>(r.ptr) + offset : nullptr;
    const auto *       reg          = ggml_sycl::alloc_registry::instance().lookup(ptr);
    const sycl::usm::alloc fast     = ptr ? ggml_sycl_get_alloc_type(ptr) : sycl::usm::alloc::unknown;
    const sycl::usm::alloc probed   = probe_pointer_type_in_queue_context(ptr, queue);

    std::fprintf(stderr,
                 "[MEM-COPY] %s %s queue_dev=%d handle_dev=%d on_dev=%d base=%p ptr=%p off=%zu "
                 "reg=%s reg_dev=%d reg_base=%p reg_size=%zu fast=%s probed=%s\n",
                 label,
                 endpoint,
                 queue_device,
                 handle.device(),
                 r.on_device ? 1 : 0,
                 r.ptr,
                 ptr,
                 offset,
                 reg ? alloc_type_name(reg->type) : "none",
                 reg ? reg->device_id : -99,
                 reg ? reinterpret_cast<void *>(reg->base) : nullptr,
                 reg ? reg->size : size_t(0),
                 usm_alloc_name(fast),
                 usm_alloc_name(probed));
}

static void trace_mem_copy_submit(const char * label,
                                  const mem_handle & dst,
                                  size_t dst_offset,
                                  const mem_handle & src,
                                  size_t src_offset,
                                  size_t size,
                                  sycl::queue & queue) {
    if (!mem_copy_trace_enabled()) {
        return;
    }
    const int queue_device = queue_device_or_host(queue);
    std::fprintf(stderr, "[MEM-COPY] %s submit queue_dev=%d size=%zu deps=?\n", label, queue_device, size);
    trace_mem_copy_endpoint(label, "dst", dst, dst_offset, queue);
    trace_mem_copy_endpoint(label, "src", src, src_offset, queue);
}

static void wait_deps(const std::vector<sycl::event> & deps) {
    for (const sycl::event & dep : deps) {
        const_cast<sycl::event &>(dep).wait_and_throw();
    }
}

static bool alloc_pinned_stage_handle(size_t           size,
                                      sycl::queue &    queue,
                                      int              device,
                                      const char *     cohort_id,
                                      bool             require_host_usm_base,
                                      mem_handle *     out) {
    alloc_request req{};
    req.queue                               = &queue;
    req.device                              = device;
    req.size                                = size;
    // The returned mem_handle may be retained by an async copy event.  Use the
    // scoped staging role so unified_free() releases this allocation
    // individually instead of relying on a reset-scoped host STAGING zone.
    req.intent.role                         = alloc_role::EXPERT_STAGING;
    req.intent.category                     = runtime_category::STAGING;
    req.intent.cohort_id                    = cohort_id ? cohort_id : "mem-copy-stage";
    req.intent.constraints.must_host_pinned = true;
    req.intent.constraints.use_pinned_pool  = true;
    // A command graph captures the staging pointer in its memcpy node and may
    // replay it after graph-boundary host-zone resets.  Keep graph-recorded
    // staging allocations as standalone unified-cache-owned host USM bases
    // rather than reset-scoped SCRATCH/STAGING zone slices.
    req.intent.constraints.require_host_usm_base = require_host_usm_base || ggml_sycl_graph_recording_active();

    *out = unified_allocate(req);
    if (!out->valid()) {
        return false;
    }
    return true;
}

static sycl::event mem_copy_direct_submit(const mem_handle &               dst,
                                          size_t                           dst_offset,
                                          const mem_handle &               src,
                                          size_t                           src_offset,
                                          size_t                           size,
                                          sycl::queue &                    queue,
                                          const std::vector<sycl::event> & deps,
                                          const char *                     profile_name = "sycl.memcpy.mem_ops",
                                          const char * profile_metadata = "role=memcpy;path=mem_ops",
                                          const char * profile_queue_kind = "copy",
                                          const char * file               = __builtin_FILE(),
                                          int          line               = __builtin_LINE(),
                                          const char * function           = __builtin_FUNCTION()) {
    const int    queue_device = queue_device_or_host(queue);
    resolved_ptr d            = dst.resolve(queue_device);
    resolved_ptr s            = src.resolve(queue_device);
    GGML_ASSERT(d && s && "mem_copy_async on unresolved handle");

    void *       dst_ptr = static_cast<char *>(d.ptr) + dst_offset;
    const void * src_ptr = static_cast<const char *>(s.ptr) + src_offset;

    if (!d.on_device && !s.on_device) {
        wait_deps(deps);
        std::memcpy(dst_ptr, src_ptr, size);
        return sycl::event{};
    }

    return ggml_sycl_memcpy_profile_submit(
        queue, profile_name, profile_metadata, profile_queue_kind, size, [&](sycl::queue & profiled_q) {
            return profiled_q.submit([&](sycl::handler & cgh) {
                add_deps(cgh, deps);
                cgh.memcpy(dst_ptr, src_ptr, size);
            });
        }, file, line, function);
}

static sycl::event mem_copy_submit(const mem_handle &               dst,
                                   size_t                           dst_offset,
                                   const mem_handle &               src,
                                   size_t                           src_offset,
                                   size_t                           size,
                                   sycl::queue &                    queue,
                                   const std::vector<sycl::event> & deps,
                                   bool                             retain_until_event) {
    resolved_ptr d = dst.resolve();
    resolved_ptr s = src.resolve();
    GGML_ASSERT(d && s && "mem_copy_async on unresolved handle");

    const int dst_device = d.on_device ? dst.device() : mem_handle::HOST_DEVICE;
    const int src_device = s.on_device ? src.device() : mem_handle::HOST_DEVICE;

    if (d.on_device && s.on_device && dst_device >= 0 && src_device >= 0 && dst_device != src_device) {
        sycl::queue & src_queue = queue_for_device_or_fallback(src_device, queue);
        sycl::queue & dst_queue = queue_for_device_or_fallback(dst_device, queue);

        mem_handle src_stage;
        const int fallback_device = queue_device_or_host(queue);
        if (!alloc_pinned_stage_handle(size, src_queue, src_device, "mem-copy-cross-device",
                                       /*require_host_usm_base=*/true, &src_stage) &&
            !alloc_pinned_stage_handle(size, dst_queue, dst_device, "mem-copy-cross-device",
                                       /*require_host_usm_base=*/true, &src_stage) &&
            !alloc_pinned_stage_handle(size, queue, fallback_device, "mem-copy-cross-device",
                                       /*require_host_usm_base=*/true, &src_stage)) {
            GGML_ABORT("[MEM-OPS] failed to allocate %zu byte host-pinned staging buffer for device %d -> %d copy",
                       size, src_device, dst_device);
        }

        trace_mem_copy_submit("cross-d2h", src_stage, 0, src, src_offset, size, src_queue);
        sycl::event d2h = mem_copy_direct_submit(src_stage, 0, src, src_offset, size, src_queue, deps,
                                                 "sycl.memcpy.cross_device", "role=memcpy;path=cross_device");

        if (queues_share_context(src_queue, dst_queue)) {
            std::vector<sycl::event> h2d_deps;
            h2d_deps.push_back(d2h);
            trace_mem_copy_submit("cross-h2d", dst, dst_offset, src_stage, 0, size, dst_queue);
            sycl::event h2d = mem_copy_direct_submit(dst, dst_offset, src_stage, 0, size, dst_queue, h2d_deps,
                                                     "sycl.memcpy.cross_device", "role=memcpy;path=cross_device");
            if (retain_until_event) {
                retain_handles_until_event({ dst, src, src_stage }, h2d);
            } else {
                h2d.wait_and_throw();
            }
            return h2d;
        }

        d2h.wait_and_throw();

        mem_handle dst_stage;
        if (!alloc_pinned_stage_handle(size, dst_queue, dst_device, "mem-copy-cross-device-dst",
                                       /*require_host_usm_base=*/true, &dst_stage) &&
            !alloc_pinned_stage_handle(size, queue, fallback_device, "mem-copy-cross-device-dst",
                                       /*require_host_usm_base=*/true, &dst_stage)) {
            GGML_ABORT("[MEM-OPS] failed to allocate %zu byte destination host-pinned staging buffer for device %d -> %d copy",
                       size, src_device, dst_device);
        }
        resolved_ptr src_stage_ptr = src_stage.resolve();
        resolved_ptr dst_stage_ptr = dst_stage.resolve();
        GGML_ASSERT(src_stage_ptr && dst_stage_ptr && !src_stage_ptr.on_device && !dst_stage_ptr.on_device);
        std::memcpy(dst_stage_ptr.ptr, src_stage_ptr.ptr, size);

        trace_mem_copy_submit("cross-h2d", dst, dst_offset, dst_stage, 0, size, dst_queue);
        sycl::event h2d = mem_copy_direct_submit(dst, dst_offset, dst_stage, 0, size, dst_queue, {},
                                                 "sycl.memcpy.cross_device", "role=memcpy;path=cross_device");
        if (retain_until_event) {
            retain_handles_until_event({ dst, src, src_stage, dst_stage }, h2d);
        } else {
            h2d.wait_and_throw();
        }
        return h2d;
    }

    sycl::queue * copy_queue = &queue;
    const int     requested_queue_device = queue_device_or_host(queue);
    if (s.on_device && src_device >= 0 && !d.on_device) {
        copy_queue = requested_queue_device == src_device ? &queue : &queue_for_device_or_fallback(src_device, queue);
    } else if (d.on_device && dst_device >= 0 && !s.on_device) {
        copy_queue = requested_queue_device == dst_device ? &queue : &queue_for_device_or_fallback(dst_device, queue);
    } else if (d.on_device && s.on_device && dst_device >= 0 && dst_device == src_device) {
        copy_queue = requested_queue_device == dst_device ? &queue : &queue_for_device_or_fallback(dst_device, queue);
    }

    if (d.on_device && !s.on_device && !host_ptr_is_usm_accessible(s.ptr)) {
        constexpr size_t max_stage_bytes = 64ull * 1024ull * 1024ull;
        const size_t     stage_bytes     = std::min(size, max_stage_bytes);
        mem_handle stage;
        if (!alloc_pinned_stage_handle(stage_bytes, *copy_queue, dst_device, "mem-copy-host-to-device",
                                       /*require_host_usm_base=*/false, &stage)) {
            GGML_ABORT("[MEM-OPS] failed to allocate %zu byte host-pinned staging buffer for host -> device %d copy",
                       stage_bytes, dst_device);
        }
        wait_deps(deps);
        resolved_ptr stage_ptr = stage.resolve();
        GGML_ASSERT(stage_ptr && !stage_ptr.on_device);
        sycl::event event;
        size_t      copied = 0;
        while (copied < size) {
            if (copied != 0) {
                event.wait_and_throw();
            }
            const size_t cur = std::min(stage_bytes, size - copied);
            std::memcpy(stage_ptr.ptr, static_cast<const char *>(s.ptr) + src_offset + copied, cur);
            event = mem_copy_direct_submit(dst, dst_offset + copied, stage, 0, cur, *copy_queue, {});
            copied += cur;
        }
        if (retain_until_event) {
            retain_handles_until_event({ dst, src, stage }, event);
        } else {
            event.wait_and_throw();
        }
        return event;
    }

    if (!d.on_device && s.on_device && !host_ptr_is_usm_accessible(d.ptr)) {
        constexpr size_t max_stage_bytes = 64ull * 1024ull * 1024ull;
        const size_t     stage_bytes     = std::min(size, max_stage_bytes);
        mem_handle stage;
        if (!alloc_pinned_stage_handle(stage_bytes, *copy_queue, src_device, "mem-copy-device-to-host",
                                       /*require_host_usm_base=*/false, &stage)) {
            GGML_ABORT("[MEM-OPS] failed to allocate %zu byte host-pinned staging buffer for device %d -> host copy",
                       stage_bytes, src_device);
        }
        resolved_ptr stage_ptr = stage.resolve();
        GGML_ASSERT(stage_ptr && !stage_ptr.on_device);
        size_t copied = 0;
        while (copied < size) {
            const size_t cur = std::min(stage_bytes, size - copied);
            sycl::event event =
                mem_copy_direct_submit(stage, 0, src, src_offset + copied, cur, *copy_queue,
                                       copied == 0 ? deps : std::vector<sycl::event>{});
            event.wait_and_throw();
            std::memcpy(static_cast<char *>(d.ptr) + dst_offset + copied, stage_ptr.ptr, cur);
            copied += cur;
        }
        return sycl::event{};
    }

    sycl::event event = mem_copy_direct_submit(dst, dst_offset, src, src_offset, size, *copy_queue, deps);
    if (retain_until_event) {
        retain_handles_until_event({ dst, src }, event);
    }
    return event;
}

static sycl::event mem_copy_submit(const mem_handle &               dst,
                                   const mem_handle &               src,
                                   size_t                           size,
                                   sycl::queue &                    queue,
                                   const std::vector<sycl::event> & deps,
                                   bool                             retain_until_event) {
    return mem_copy_submit(dst, 0, src, 0, size, queue, deps, retain_until_event);
}

static sycl::event mem_fill_direct_submit(const mem_handle &               h,
                                          size_t                           offset,
                                          int                              value,
                                          size_t                           size,
                                          sycl::queue &                    queue,
                                          const std::vector<sycl::event> & deps,
                                          const char *                     file     = __builtin_FILE(),
                                          int                              line     = __builtin_LINE(),
                                          const char *                     function = __builtin_FUNCTION()) {
    const int    queue_device = queue_device_or_host(queue);
    resolved_ptr r            = h.resolve(queue_device);
    GGML_ASSERT(r && "mem_fill_async on unresolved handle");

    void * ptr = static_cast<char *>(r.ptr) + offset;
    if (!r.on_device) {
        for (const sycl::event & dep : deps) {
            const_cast<sycl::event &>(dep).wait_and_throw();
        }
        std::memset(ptr, value, size);
        return sycl::event{};
    }

    return ggml_sycl_memcpy_profile_submit(
        queue, "sycl.memcpy.mem_fill", "role=memfill;path=mem_ops", "copy", size, [&](sycl::queue & profiled_q) {
            return profiled_q.submit([&](sycl::handler & cgh) {
                add_deps(cgh, deps);
                cgh.memset(ptr, value, size);
            });
        }, file, line, function);
}

static sycl::event mem_fill_submit(const mem_handle &               h,
                                   size_t                           offset,
                                   int                              value,
                                   size_t                           size,
                                   sycl::queue &                    queue,
                                   const std::vector<sycl::event> & deps,
                                   bool                             retain_until_event) {
    resolved_ptr r = h.resolve();
    GGML_ASSERT(r && "mem_fill_async on unresolved handle");

    sycl::queue * fill_queue = &queue;
    if (r.on_device && h.device() >= 0) {
        const int requested_queue_device = queue_device_or_host(queue);
        fill_queue = requested_queue_device == h.device() ? &queue : &queue_for_device_or_fallback(h.device(), queue);
    }

    sycl::event event = mem_fill_direct_submit(h, offset, value, size, *fill_queue, deps);

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

sycl::event mem_copy_async(const mem_handle &               dst,
                           size_t                           dst_offset,
                           const mem_handle &               src,
                           size_t                           src_offset,
                           size_t                           size,
                           sycl::queue &                    queue,
                           const std::vector<sycl::event> & deps) {
    return mem_copy_submit(dst, dst_offset, src, src_offset, size, queue, deps, true);
}

sycl::event mem_fill_async(const mem_handle &               h,
                           int                              value,
                           size_t                           size,
                           sycl::queue &                    queue,
                           const std::vector<sycl::event> & deps) {
    return mem_fill_submit(h, 0, value, size, queue, deps, true);
}

sycl::event mem_fill_async(const mem_handle &               h,
                           size_t                           offset,
                           int                              value,
                           size_t                           size,
                           sycl::queue &                    queue,
                           const std::vector<sycl::event> & deps) {
    return mem_fill_submit(h, offset, value, size, queue, deps, true);
}

void mem_copy(const mem_handle &               dst,
              const mem_handle &               src,
              size_t                           size,
              sycl::queue &                    queue,
              const std::vector<sycl::event> & deps) {
    mem_copy_submit(dst, src, size, queue, deps, false).wait_and_throw();
}

void mem_copy(const mem_handle &               dst,
              size_t                           dst_offset,
              const mem_handle &               src,
              size_t                           src_offset,
              size_t                           size,
              sycl::queue &                    queue,
              const std::vector<sycl::event> & deps) {
    mem_copy_submit(dst, dst_offset, src, src_offset, size, queue, deps, false).wait_and_throw();
}

void mem_fill(const mem_handle &               h,
              int                              value,
              size_t                           size,
              sycl::queue &                    queue,
              const std::vector<sycl::event> & deps) {
    mem_fill_submit(h, 0, value, size, queue, deps, false).wait_and_throw();
}

void mem_fill(const mem_handle &               h,
              size_t                           offset,
              int                              value,
              size_t                           size,
              sycl::queue &                    queue,
              const std::vector<sycl::event> & deps) {
    mem_fill_submit(h, offset, value, size, queue, deps, false).wait_and_throw();
}

}  // namespace ggml_sycl
