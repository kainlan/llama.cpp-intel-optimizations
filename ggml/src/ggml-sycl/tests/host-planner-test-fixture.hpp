// The host-only planner tests' shared fixture: the check counters they report from, and a mocked SYCL device table the
// real planner plans against without touching a device. Not a production header.
#pragma once

#include "../common.hpp"

#include <cstddef>
#include <cstdio>

namespace host_planner_test {

inline int g_checks = 0;
inline int g_fail   = 0;

inline void check(bool cond, const char * what) {
    g_checks++;
    if (!cond) {
        g_fail++;
        std::printf("  FAIL: %s\n", what);
    }
}

inline void check_eq(const char * what, size_t got, size_t want) {
    g_checks++;
    if (got != want) {
        g_fail++;
        std::printf("  FAIL: %s: got %zu, want %zu\n", what, got, want);
    }
}

// `device_count` discrete cards of `total_vram` bytes and `nsm` compute units each, enough to plan with and nothing
// touched. Each card is named "<name>-<index>".
inline ggml_sycl_device_info make_mock_sycl_info(int device_count, size_t total_vram, int nsm, const char * name) {
    constexpr size_t mib = 1024ull * 1024ull;

    ggml_sycl_device_info info{};
    info.device_count        = device_count;
    info.total_gpu_count     = device_count;
    info.host_max_alloc_size = 128ull * 1024ull * mib;

    for (int d = 0; d < device_count; ++d) {
        sycl_device_info & dev          = info.devices[d];
        dev.cc                          = 1200;
        dev.nsm                         = nsm;
        dev.smpbo                       = 64ull * 1024ull;
        dev.vmm                         = true;
        dev.total_vram                  = total_vram;
        dev.free_vram_at_init           = total_vram;
        dev.max_alloc_size              = total_vram - 512ull * mib;
        dev.safe_max_alloc_size         = dev.max_alloc_size;
        dev.supports_soa_reorder        = true;
        dev.xmx_caps.compute_units      = dev.nsm;
        dev.xmx_caps.global_mem_size    = total_vram;
        dev.xmx_caps.max_mem_alloc_size = dev.max_alloc_size;
        std::snprintf(dev.device_name, sizeof(dev.device_name), "%s-%d", name, d);
        info.max_work_group_sizes[d] = 256;
        info.default_tensor_split[d] = static_cast<float>(d + 1) / static_cast<float>(device_count);
        info.gpu_dpct_ids[d]         = d;
    }
    for (int src = 0; src < device_count; ++src) {
        for (int dst = 0; dst < device_count; ++dst) {
            sycl_peer_link_info & link = info.peer_links[src][dst];
            link.valid                 = true;
            link.src_device            = src;
            link.dst_device            = dst;
            link.same_device           = src == dst;
            link.same_backend          = true;
            link.same_platform         = true;
            link.same_sycl_context     = src == dst;
        }
    }
    return info;
}

}  // namespace host_planner_test
