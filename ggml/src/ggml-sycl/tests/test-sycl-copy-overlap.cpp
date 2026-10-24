// test-sycl-copy-overlap: do H2D copies on a cache queue overlap a compute kernel?
// (docs/plans/2026-10-24-prefill-window-ring.md, Task 4 spike; llama.cpp-efck)
//
// Times 16 x 4 MiB pinned host-to-device copies alone, then the same copies
// submitted while a ~100 ms spin kernel runs on the backend's compute queue,
// for the cache's BCS queue and for the cache's compute queue. Prints one
// [COPY-OVERLAP] row per queue at default verbosity. overlap=yes means the
// copies did not queue behind the kernel. Lead-run only (it takes the device);
// exits 77 without one.
//
// The copy source is the prefill ring's own source shape: a pinned-pool WEIGHT
// slice of the cache's host zone, which mem_copy_async trusts as host USM and
// copies directly. Any other host source is staged through a synchronous host
// memcpy, and the row would time that instead of the DMA.
//
// No host thread waits on a copy-queue event: the copies are drained with a
// queue wait on the copy queue itself. The only event waited is the spin
// kernel's, which is on the compute queue.

#include "../common.hpp"
#include "../mem-handle.hpp"
#include "../mem-ops.hpp"
#include "../unified-cache.hpp"
#include "sycl-test-skip.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <optional>
#include <string>
#include <sycl/sycl.hpp>
#include <utility>

using ggml_sycl::mem_handle;

static constexpr size_t kChunkBytes = 4ull * 1024ull * 1024ull;
static constexpr size_t kChunks     = 16;
static constexpr size_t kCopyBytes  = kChunkBytes * kChunks;

static const char * adapter_name() {
    std::ifstream maps("/proc/self/maps");
    std::string   line;
    bool          v1 = false;
    bool          v2 = false;
    while (std::getline(maps, line)) {
        v2 |= line.find("libur_adapter_level_zero_v2.so") != std::string::npos;
        v1 |= line.find("libur_adapter_level_zero.so") != std::string::npos;
    }
    return v2 ? "v2" : v1 ? "v1" : "unknown";
}

// Device busy work on `q`: one work-group iterating an LCG `iters` times.
static sycl::event spin_kernel(sycl::queue & q, uint32_t * counter, uint64_t iters) {
    return q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::nd_range<1>(256, 256), [=](sycl::nd_item<1> it) {
            uint32_t acc = static_cast<uint32_t>(it.get_global_id(0));
            for (uint64_t i = 0; i < iters; ++i) {
                acc = acc * 1664525u + 1013904223u;
            }
            if (it.get_global_id(0) == 0) {
                *counter = acc;
            }
        });
    });
}

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// 16 x 4 MiB pinned H2D on `copy_q`; returns wall ms from the first submit to copy_q.wait().
static double time_copies(sycl::queue & copy_q, const mem_handle & dst, const mem_handle & src) {
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < kChunks; ++i) {
        ggml_sycl::mem_copy_async(dst, i * kChunkBytes, src, i * kChunkBytes, kChunkBytes, copy_q);
    }
    copy_q.wait();
    return ms_since(t0);
}

// A bytes-long device buffer in the RUNTIME zone of cq's device.
static mem_handle alloc_runtime_device(sycl::queue & cq, int device, size_t bytes) {
    ggml_sycl::alloc_request req{};
    req.queue                                     = &cq;
    req.device                                    = device;
    req.size                                      = bytes;
    req.intent.role                               = ggml_sycl::alloc_role::STAGING;
    req.intent.category                           = ggml_sycl::runtime_category::STAGING;
    req.intent.constraints.must_device            = true;
    req.intent.constraints.prefer_vram_zone       = ggml_sycl::vram_zone_id::RUNTIME;
    req.intent.constraints.forbid_vram_zone_spill = true;
    ggml_sycl::allocation_result result           = ggml_sycl::unified_allocate_owner(req);
    if (!result) {
        return mem_handle();
    }
    return mem_handle::from_owned_alloc(std::move(result.owner), GGML_LAYOUT_AOS);
}

// A bytes-long pinned-pool WEIGHT slice of the cache's host zone: the ring's copy source.
static mem_handle alloc_pinned_weight(sycl::queue & cq, int device, size_t bytes) {
    ggml_sycl::alloc_request req{};
    req.queue                               = &cq;
    req.device                              = device;
    req.size                                = bytes;
    req.intent.role                         = ggml_sycl::alloc_role::WEIGHT;
    req.intent.category                     = ggml_sycl::runtime_category::OTHER;
    req.intent.constraints.must_host_pinned = true;
    req.intent.constraints.use_pinned_pool  = true;
    ggml_sycl::allocation_result result     = ggml_sycl::unified_allocate_owner(req);
    if (!result) {
        return mem_handle();
    }
    return mem_handle::from_owned_alloc(std::move(result.owner), GGML_LAYOUT_AOS);
}

int main() {
    const std::optional<sycl::device> dev = sycl_test_require_gpu("the copy-overlap spike");
    if (!dev) {
        return SYCL_TEST_SKIP;
    }
    if (ggml_sycl_info().total_gpu_count <= 0) {
        sycl_test_print_skip("the copy-overlap spike backend owner selection");
        return SYCL_TEST_SKIP;
    }
    sycl::queue & ccs    = ggml_sycl_get_gpu_device(0).default_queue();
    const int     device = ggml_sycl_get_device_id_from_queue(ccs);
    std::printf("Device: %s (logical %d)\n", ccs.get_device().get_info<sycl::info::device::name>().c_str(), device);

    ggml_sycl::unified_cache * cache = ggml_sycl::get_unified_cache_for_device(device);
    if (!cache) {
        std::printf("FAIL: no unified cache for device %d\n", device);
        return 1;
    }
    if (!cache->host_zones_configured()) {
        cache->configure_host_zones(2 * kCopyBytes, 2ull << 20, 4ull << 20, 2ull << 20);
    }
    if (!cache->host_zones_configured()) {
        std::printf("FAIL: the cache's host zones could not be configured\n");
        return 1;
    }
    sycl::queue & cq = cache->get_queue();

    const mem_handle src = alloc_pinned_weight(cq, device, kCopyBytes);
    const mem_handle dst = alloc_runtime_device(cq, device, kCopyBytes);
    const mem_handle ctr = alloc_runtime_device(cq, device, 64);
    if (!src.valid() || !dst.valid() || !ctr.valid()) {
        std::printf("FAIL: allocation refused (src=%d dst=%d ctr=%d)\n", src.valid() ? 1 : 0, dst.valid() ? 1 : 0,
                    ctr.valid() ? 1 : 0);
        return 1;
    }
    // Without a zone-managed pinned WEIGHT owner the copies are staged, and the rows would time a host memcpy.
    ggml_sycl::alloc_metadata owner{};
    if (!src.owner_metadata(&owner) || owner.tier != ggml_sycl::alloc_tier::HOST_PINNED ||
        owner.host_zone == ggml_sycl::host_zone_id::COUNT || owner.role != ggml_sycl::alloc_role::WEIGHT) {
        std::printf("FAIL: the copy source is not a zone-managed pinned-pool WEIGHT owner (tier=%d host_zone=%d)\n",
                    (int) owner.tier, (int) owner.host_zone);
        return 1;
    }
    const ggml_sycl::resolved_ptr ctr_ptr = ctr.resolve(device);
    if (!ctr_ptr || !ctr_ptr.on_device) {
        std::printf("FAIL: the counter did not resolve to device memory\n");
        return 1;
    }
    uint32_t * counter = static_cast<uint32_t *>(ctr_ptr.ptr);

    // JIT the spin kernel before timing it, then calibrate it to ~100 ms.
    spin_kernel(ccs, counter, 1).wait();
    uint64_t iters     = 1ull << 20;
    double   kernel_ms = 0.0;
    for (;;) {
        const auto t0 = std::chrono::steady_clock::now();
        spin_kernel(ccs, counter, iters).wait();
        kernel_ms = ms_since(t0);
        if (kernel_ms >= 80.0 || iters > (1ull << 36)) {
            break;
        }
        iters *= 2;
    }

    struct arm {
        const char *  name;
        sycl::queue * q;
    };

    const arm arms[] = {
        { "bcs",     &cache->get_bcs_queue() },
        { "compute", &cq                     }
    };
    const char * adapter = adapter_name();
    for (const arm & a : arms) {
        time_copies(*a.q, dst, src);  // warm-up
        const double alone = time_copies(*a.q, dst, src);
        const auto   t0    = std::chrono::steady_clock::now();
        sycl::event  k     = spin_kernel(ccs, counter, iters);
        const double with  = time_copies(*a.q, dst, src);
        k.wait();
        const double total   = ms_since(t0);
        const bool   overlap = with < alone + 0.5 * kernel_ms;
        std::printf(
            "[COPY-OVERLAP] queue=%s alone_ms=%.1f with_kernel_ms=%.1f kernel_ms=%.1f total_ms=%.1f overlap=%s "
            "adapter=%s\n",
            a.name, alone, with, kernel_ms, total, overlap ? "yes" : "no", adapter);
        std::fflush(stdout);
    }
    ccs.wait();
    cq.wait();
    return 0;
}
