//
// a7l5w-probe.hpp — host-pointer bad-pointer bisect instrumentation
//
// Bead: llama.cpp-a7l5w
//
// Signature: gpt-oss-20b TG SEGVs at token 10 in a DNNL JIT F32 GEMM microkernel
// (`vbroadcastss -0x80(%rcx), %ymm2` + `vfmadd231ps`).  `rcx` is the A-matrix
// pointer and lies in an unmapped VA gap between two adjacent 2 GB
// `/dev/dri/renderD129` mappings.  Predecessor investigations ruled out TLSF,
// CpuExpertPool lifetime, and allocator fragmentation.  The bad pointer is
// constructed by caller-side arithmetic (`base + i02*nb02 + i03*nb03`) or
// returned from a cache whose backing was resized/evicted.
//
// This probe fires an informative assertion BEFORE the DNNL JIT kernel is
// entered with a pointer that the alloc_registry cannot account for.  When an
// assertion fires, its site, tensor name, offset, registered size and access
// extent are logged with std::abort() so the coredump points at the actual bad
// caller rather than the downstream DNNL microkernel frame.
//
// Compile-time gate: GGML_SYCL_A7L5W_INSTRUMENT.  Default: ON when NDEBUG is
// not defined.  Can be forced on in a release build by passing
// -DGGML_SYCL_A7L5W_INSTRUMENT=1 at configure time.
//
// Remove in the fix commit; keep the documented signature in
// docs/plans/2026-04-19-a7l5w-rootcause.md so future investigators can re-add.
//
// MIT license
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_A7L5W_PROBE_HPP
#define GGML_SYCL_A7L5W_PROBE_HPP

// The probe is OPT-IN.  To run the bisect, rebuild with
// `-DGGML_SYCL_A7L5W_INSTRUMENT=1` (both on the compile command line for
// cpu-dispatch.cpp and ggml-sycl.cpp) or flip the default below to 1.  The
// shipping build defaults to disabled so the registry lookups and debug
// prints are not on the hot path.
#ifndef GGML_SYCL_A7L5W_INSTRUMENT
#    define GGML_SYCL_A7L5W_INSTRUMENT 0
#endif

#if GGML_SYCL_A7L5W_INSTRUMENT

#    include "alloc-registry.hpp"
#    include "ggml.h"

#    include <cstddef>
#    include <cstdint>
#    include <cstdio>
#    include <cstdlib>

namespace ggml_sycl {
namespace a7l5w {

// Assert that [ptr, ptr + access_extent) lies entirely within a single
// registered allocation.  Logs the caller site, tensor identity, offset into
// the allocation, the registered (base, size), and the claimed access extent.
// Fires std::abort() on failure so the coredump points at THIS site, not the
// downstream DNNL microkernel.
inline void assert_host_ptr_range(const char * site,
                                  const char * tensor_name,
                                  const void * ptr,
                                  std::size_t  access_extent) {
    if (ptr == nullptr) {
        std::fprintf(stderr,
                     "[A7L5W] ABORT %s: nullptr (tensor=%s extent=%zu)\n",
                     site,
                     tensor_name ? tensor_name : "(null)",
                     access_extent);
        std::fflush(stderr);
        std::abort();
    }
    if (access_extent == 0) {
        return;  // zero-byte access is trivially in-range
    }
    const auto * info = alloc_registry::instance().lookup(ptr);
    if (info == nullptr) {
        // Not a SYCL-registered allocation.  This is common for stack/heap/mmap
        // ggml compute buffers that never go through register_alloc.  We log and
        // return; a SEGV here would need a different mechanism to catch (e.g.
        // a probing load) and we do not want to abort in the normal case where
        // the fault is downstream.  Only log the first N hits per site to avoid
        // flooding stderr, and always flush so a downstream SEGV doesn't hide
        // us.
        static std::atomic<int> unreg_log_count{ 0 };
        if (unreg_log_count.fetch_add(1, std::memory_order_relaxed) < 40) {
            std::fprintf(stderr,
                         "[A7L5W] UNREG %s: tensor=%s ptr=%p extent=%zu (not in alloc_registry)\n",
                         site,
                         tensor_name ? tensor_name : "(null)",
                         ptr,
                         access_extent);
            std::fflush(stderr);
        }
        return;
    }
    // Registered allocation — verify the access extent fits.
    const auto        base   = reinterpret_cast<std::uintptr_t>(ptr);
    const std::size_t offset = static_cast<std::size_t>(base - info->base);
    if (offset > info->size || offset + access_extent > info->size) {
        std::fprintf(stderr,
                     "[A7L5W] ABORT %s: OOB tensor=%s ptr=%p extent=%zu "
                     "alloc_base=0x%lx alloc_size=%zu offset=%zu over_by=%zd\n",
                     site,
                     tensor_name ? tensor_name : "(null)",
                     ptr,
                     access_extent,
                     static_cast<unsigned long>(info->base),
                     info->size,
                     offset,
                     static_cast<std::ptrdiff_t>(offset + access_extent) -
                         static_cast<std::ptrdiff_t>(info->size));
        std::fflush(stderr);
        std::abort();
    }
}

// Convenience wrapper: access extent implied by a tensor's [ne, nb] is
// ggml_nbytes(tensor), which is the minimal "the whole tensor fits" check.
// For batched indexing within a stacked MUL_MAT (e.g. MoE experts), the
// caller should pass the effective extent they are about to read from `ptr`
// (usually `ne00 * ne01 * elem_size` for a single slab).
inline void assert_host_ptr_tensor(const char *        site,
                                   const ggml_tensor * tensor,
                                   const void *        ptr,
                                   std::size_t         access_extent) {
    assert_host_ptr_range(site,
                          tensor ? tensor->name : "(null-tensor)",
                          ptr,
                          access_extent);
}

}  // namespace a7l5w
}  // namespace ggml_sycl

#    define GGML_SYCL_A7L5W_ASSERT_PTR(site, name, ptr, extent) \
        ::ggml_sycl::a7l5w::assert_host_ptr_range((site), (name), (ptr), (extent))
#    define GGML_SYCL_A7L5W_ASSERT_TENSOR(site, tensor, ptr, extent) \
        ::ggml_sycl::a7l5w::assert_host_ptr_tensor((site), (tensor), (ptr), (extent))

#else  // GGML_SYCL_A7L5W_INSTRUMENT == 0

#    define GGML_SYCL_A7L5W_ASSERT_PTR(site, name, ptr, extent) ((void) 0)
#    define GGML_SYCL_A7L5W_ASSERT_TENSOR(site, tensor, ptr, extent) ((void) 0)

#endif  // GGML_SYCL_A7L5W_INSTRUMENT

#endif  // GGML_SYCL_A7L5W_PROBE_HPP
