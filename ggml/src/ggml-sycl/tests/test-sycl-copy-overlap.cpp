// test-sycl-copy-overlap: do H2D copies on a cache queue overlap a compute kernel?
// (docs/plans/2026-10-24-prefill-window-ring.md, Task 4 spike; llama.cpp-efck)
//
// Times 16 x 4 MiB pinned host-to-device copies alone, then the same copies
// submitted while a ~100 ms spin kernel runs on the backend's compute queue,
// for the cache's BCS queue and for the cache's compute queue. Prints one
// [COPY-OVERLAP] row per queue and spin shape at default verbosity.
// overlap=yes means the copies did not queue behind the kernel. Lead-run only
// (it takes the device); exits 77 without one.
//
// Two spin shapes: spin=1wg is one work-group and leaves most EUs idle, so a
// copy the compute engine runs by time-slicing can still finish on time;
// spin=full launches 4096 work-groups to saturate the EUs, closer to the
// dequant+GEMM load the ring's copies meet.
//
// overlap= does not say which engine ran the copies, so each row also prints
// this process's per-engine busy deltas from the xe driver's DRM fdinfo
// (drm-cycles-<class>, GPU timestamp ticks this client kept the engine class
// busy; ticks= is the drm-total-cycles delta, the elapsed GPU timestamp over
// the same window). alone_*: around the alone copies, which no kernel shares,
// so they attribute the copies' engine directly. The unprefixed fields are
// around the kernel submit and the with-kernel copies, and ccs_cycles there
// includes the spin kernel's own ticks. Only render-node fds whose drm-pdev is
// the device under test count, each DRM client once. NA: no drm-cycles key
// (not the xe driver, or no matching fd). One raw fdinfo is printed first as
// [COPY-OVERLAP-FDINFO] lines so the keys this kernel prints are on record.
//
// The copy source is the prefill ring's own source shape: a pinned-pool WEIGHT
// slice of the cache's host zone, which mem_copy_async trusts as host USM and
// copies directly. Any other host source is staged through a synchronous host
// memcpy, and the row would time that instead of the DMA. So the source's
// owner is checked against that trust rule before timing, and every timed call
// reads this thread's staged-H2D census (counted with the trace off) around its
// copies: staged_h2d= on the row is the count over the row's three copy rounds,
// and a non-zero count prints a FAIL line and makes the test exit 1.
//
// No host thread waits on a copy-queue event: the copies are drained with a
// queue wait on the copy queue itself. The only event waited is the spin
// kernel's, which is on the compute queue.

#include "../common.hpp"
#include "../mem-handle.hpp"
#include "../mem-ops.hpp"
#include "../unified-cache.hpp"
#include "sycl-test-skip.hpp"

#include <dirent.h>
#include <unistd.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <sycl/sycl.hpp>
#include <utility>
#include <vector>

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

// The fds of this process open on a DRM render node, with their /proc/self/fd number.
static std::vector<std::string> render_node_fds() {
    std::vector<std::string> fds;
    DIR *                    dir = opendir("/proc/self/fd");
    if (!dir) {
        return fds;
    }
    for (dirent * e = readdir(dir); e != nullptr; e = readdir(dir)) {
        if (e->d_name[0] == '.') {
            continue;
        }
        const std::string link = std::string("/proc/self/fd/") + e->d_name;
        char              target[256];
        const ssize_t     n = readlink(link.c_str(), target, sizeof(target) - 1);
        if (n <= 0) {
            continue;
        }
        target[n] = '\0';
        if (std::string(target).rfind("/dev/dri/renderD", 0) == 0) {
            fds.push_back(e->d_name);
        }
    }
    closedir(dir);
    return fds;
}

// The `key:\tvalue` lines of one fd's fdinfo.
static std::vector<std::pair<std::string, std::string>> read_fdinfo(const std::string & fd) {
    std::vector<std::pair<std::string, std::string>> kv;
    std::ifstream                                    in("/proc/self/fdinfo/" + fd);
    std::string                                      line;
    while (std::getline(in, line)) {
        const size_t colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        const size_t at = line.find_first_not_of(" \t", colon + 1);
        kv.emplace_back(line.substr(0, colon), at == std::string::npos ? std::string() : line.substr(at));
    }
    return kv;
}

// Engine-busy counters summed over this process's DRM clients on the device whose PCI address is `pdev`
// (every render-node client when `pdev` is empty).
struct drm_snapshot {
    std::map<std::string, unsigned long long> cycles;     // drm-cycles-<class>, keyed by <class>
    unsigned long long                        ticks = 0;  // max drm-total-cycles-<class> over the clients
};

static drm_snapshot read_drm_snapshot(const std::string & pdev) {
    drm_snapshot          snap;
    std::set<std::string> clients;
    for (const std::string & fd : render_node_fds()) {
        const std::vector<std::pair<std::string, std::string>> kv = read_fdinfo(fd);
        std::string                                            fd_pdev;
        std::string                                            client;
        for (const auto & p : kv) {
            if (p.first == "drm-pdev") {
                fd_pdev = p.second;
            } else if (p.first == "drm-client-id") {
                client = p.second;
            }
        }
        if ((!pdev.empty() && fd_pdev != pdev) || client.empty() || !clients.insert(client).second) {
            continue;
        }
        for (const auto & p : kv) {
            const unsigned long long v = std::strtoull(p.second.c_str(), nullptr, 10);
            if (p.first.rfind("drm-cycles-", 0) == 0) {
                snap.cycles[p.first.substr(11)] += v;
            } else if (p.first.rfind("drm-total-cycles-", 0) == 0 && v > snap.ticks) {
                snap.ticks = v;
            }
        }
    }
    return snap;
}

// " <prefix>bcs_cycles=<d> <prefix>ccs_cycles=<d> [<prefix><class>_cycles=<d> ...] <prefix>ticks=<d>": bcs and ccs
// always, any other class only when it moved; "<prefix>bcs_cycles=NA <prefix>ccs_cycles=NA" without cycles keys.
static std::string drm_delta_fields(const char * prefix, const drm_snapshot & before, const drm_snapshot & after) {
    std::string out;
    char        buf[128];
    if (after.cycles.empty()) {
        std::snprintf(buf, sizeof(buf), " %sbcs_cycles=NA %sccs_cycles=NA", prefix, prefix);
        return buf;
    }
    const char * always[] = { "bcs", "ccs" };
    for (const char * cls : always) {
        const auto a = after.cycles.find(cls);
        const auto b = before.cycles.find(cls);
        if (a == after.cycles.end()) {
            std::snprintf(buf, sizeof(buf), " %s%s_cycles=NA", prefix, cls);
        } else {
            const unsigned long long d = a->second - (b == before.cycles.end() ? 0ull : b->second);
            std::snprintf(buf, sizeof(buf), " %s%s_cycles=%llu", prefix, cls, d);
        }
        out += buf;
    }
    for (const auto & a : after.cycles) {
        if (a.first == "bcs" || a.first == "ccs") {
            continue;
        }
        const auto               b = before.cycles.find(a.first);
        const unsigned long long d = a.second - (b == before.cycles.end() ? 0ull : b->second);
        if (d != 0) {
            std::snprintf(buf, sizeof(buf), " %s%s_cycles=%llu", prefix, a.first.c_str(), d);
            out += buf;
        }
    }
    std::snprintf(buf, sizeof(buf), " %sticks=%llu", prefix, after.ticks - before.ticks);
    out += buf;
    return out;
}

// Device busy work on `q`: `groups` work-groups of 256 iterating an LCG `iters` times.
static sycl::event spin_kernel(sycl::queue & q, uint32_t * counter, size_t groups, uint64_t iters) {
    return q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::nd_range<1>(groups * 256, 256), [=](sycl::nd_item<1> it) {
            uint32_t acc = static_cast<uint32_t>(it.get_global_id(0));
            for (uint64_t i = 0; i < iters; ++i) {
                acc = acc * 1664525u + 1013904223u;
            }
            // Every work-item's result is observable (acc == 0 is the rare store), so the compiler cannot sink the
            // loop into work-item 0's branch and leave the other work-items idle.
            if (it.get_global_id(0) == 0 || acc == 0u) {
                *counter = acc;
            }
        });
    });
}

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// 16 x 4 MiB pinned H2D on `copy_q`; returns wall ms from the first submit to copy_q.wait() and adds to `staged`
// how many of the copies took the staged host-to-device branch. The census is per thread and the copies are
// submitted on this one, so the difference is exactly these copies; it is read outside the timed window.
static double time_copies(sycl::queue & copy_q, const mem_handle & dst, const mem_handle & src, uint64_t & staged) {
    const uint64_t staged_before = ggml_sycl::mem_copy_staged_wait_census_this_thread().staged_h2d;
    const auto     t0            = std::chrono::steady_clock::now();
    for (size_t i = 0; i < kChunks; ++i) {
        ggml_sycl::mem_copy_async(dst, i * kChunkBytes, src, i * kChunkBytes, kChunkBytes, copy_q);
    }
    copy_q.wait();
    const double ms = ms_since(t0);
    staged += ggml_sycl::mem_copy_staged_wait_census_this_thread().staged_h2d - staged_before;
    return ms;
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
    // Without a zone-managed pinned WEIGHT owner on a device (mem-ops.cpp's host_endpoint_owner_in_scope) the copies
    // are staged, and the rows would time a host memcpy. The census check on every row is the backstop.
    ggml_sycl::alloc_metadata owner{};
    if (!src.owner_metadata(&owner) || owner.tier != ggml_sycl::alloc_tier::HOST_PINNED ||
        owner.host_zone == ggml_sycl::host_zone_id::COUNT || owner.device < 0 ||
        owner.role != ggml_sycl::alloc_role::WEIGHT) {
        std::printf(
            "FAIL: the copy source is not a zone-managed pinned-pool WEIGHT owner (tier=%d host_zone=%d device=%d)\n",
            (int) owner.tier, (int) owner.host_zone, owner.device);
        return 1;
    }
    const ggml_sycl::resolved_ptr ctr_ptr = ctr.resolve(device);
    if (!ctr_ptr || !ctr_ptr.on_device) {
        std::printf("FAIL: the counter did not resolve to device memory\n");
        return 1;
    }
    uint32_t * counter = static_cast<uint32_t *>(ctr_ptr.ptr);

    // The device's PCI address selects its DRM clients; empty (every render-node client) when not reported.
    std::string pdev;
    if (ccs.get_device().has(sycl::aspect::ext_intel_pci_address)) {
        pdev = ccs.get_device().get_info<sycl::ext::intel::info::device::pci_address>();
    }
    std::printf("[COPY-OVERLAP-FDINFO] pdev=%s render_fds=%zu\n", pdev.empty() ? "unknown" : pdev.c_str(),
                render_node_fds().size());
    for (const std::string & fd : render_node_fds()) {
        const std::vector<std::pair<std::string, std::string>> kv    = read_fdinfo(fd);
        bool                                                   match = pdev.empty();
        for (const auto & p : kv) {
            match |= p.first == "drm-pdev" && p.second == pdev;
        }
        if (!match) {
            continue;
        }
        for (const auto & p : kv) {
            std::printf("[COPY-OVERLAP-FDINFO] fd=%s %s: %s\n", fd.c_str(), p.first.c_str(), p.second.c_str());
        }
        break;
    }

    struct spin {
        const char * name;
        size_t       groups;
        uint64_t     iters;
        double       kernel_ms;
    };

    // JIT the spin kernel before timing it, then calibrate each shape to ~100 ms.
    spin_kernel(ccs, counter, 1, 1).wait();
    spin spins[] = {
        { "1wg",  1,    1ull << 20, 0.0 },
        { "full", 4096, 1ull << 8,  0.0 },
    };
    for (spin & sp : spins) {
        for (;;) {
            const auto t0 = std::chrono::steady_clock::now();
            spin_kernel(ccs, counter, sp.groups, sp.iters).wait();
            sp.kernel_ms = ms_since(t0);
            if (sp.kernel_ms >= 80.0 || sp.iters > (1ull << 36)) {
                break;
            }
            sp.iters *= 2;
        }
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
    int          failed  = 0;
    for (const spin & sp : spins) {
        for (const arm & a : arms) {
            uint64_t staged = 0;
            time_copies(*a.q, dst, src, staged);  // warm-up
            const drm_snapshot alone_before = read_drm_snapshot(pdev);
            const double       alone        = time_copies(*a.q, dst, src, staged);
            const drm_snapshot alone_after  = read_drm_snapshot(pdev);
            const drm_snapshot with_before  = read_drm_snapshot(pdev);
            const auto         t0           = std::chrono::steady_clock::now();
            sycl::event        k            = spin_kernel(ccs, counter, sp.groups, sp.iters);
            const double       with         = time_copies(*a.q, dst, src, staged);
            const drm_snapshot with_after   = read_drm_snapshot(pdev);
            k.wait();
            const double total   = ms_since(t0);
            const bool   overlap = with < alone + 0.5 * sp.kernel_ms;
            std::printf(
                "[COPY-OVERLAP] queue=%s alone_ms=%.1f with_kernel_ms=%.1f kernel_ms=%.1f total_ms=%.1f overlap=%s "
                "adapter=%s spin=%s%s%s staged_h2d=%llu\n",
                a.name, alone, with, sp.kernel_ms, total, overlap ? "yes" : "no", adapter, sp.name,
                drm_delta_fields("alone_", alone_before, alone_after).c_str(),
                drm_delta_fields("", with_before, with_after).c_str(), (unsigned long long) staged);
            if (staged != 0) {
                std::printf(
                    "FAIL: queue=%s spin=%s: %llu of the row's %zu copies took the staged host-to-device branch, so "
                    "its times are a host memcpy, not the DMA\n",
                    a.name, sp.name, (unsigned long long) staged, 3 * kChunks);
                ++failed;
            }
            std::fflush(stdout);
        }
    }
    ccs.wait();
    cq.wait();
    return failed == 0 ? 0 : 1;
}
