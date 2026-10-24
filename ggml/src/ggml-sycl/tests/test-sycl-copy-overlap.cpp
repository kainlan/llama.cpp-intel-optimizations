// test-sycl-copy-overlap: do H2D copies on a cache queue overlap a compute kernel?
// (docs/plans/2026-10-24-prefill-window-ring.md, Task 4 spike; llama.cpp-efck)
//
// Times 16 x 4 MiB pinned host-to-device copies alone, then the same copies
// submitted while a spin kernel calibrated to 80-160 ms runs on the backend's
// compute queue, for the cache's BCS queue and for the cache's compute queue.
// Prints one [COPY-OVERLAP] row per queue and spin shape at default
// verbosity. overlap=yes means the copies did not queue behind the kernel.
// resolved= names the queue object the arm's accessor returned: the BCS
// accessor falls back to the DMA queue and then to the compute queue, and a
// bcs arm that resolved to the compute queue fails. A spin that cannot reach
// the calibration floor fails too, since every row would then read
// overlap=yes having measured nothing. Lead-run only (it takes the device);
// exits 77 without one.
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
// the same window). alone_*: from just before the alone copies' first submit
// to just after their queue wait; no kernel shares that window, so it
// attributes the copies' engine directly. The unprefixed fields: from just
// before the spin kernel's submit to just after both the with-kernel copies'
// queue wait and the kernel's own completion, so ccs_cycles there holds the
// whole spin kernel on every row plus any copy the compute engine ran. No
// fdinfo read falls inside a timed window (alone_ms, with_kernel_ms,
// total_ms). Only render-node fds whose drm-pdev is the device under test
// count, each DRM client once. NA: no drm-cycles key on a matching fd (not the
// xe driver, or no matching fd), or the device reports no PCI address, so its
// DRM clients cannot be told from the other cards'. One raw fdinfo is printed
// first as [COPY-OVERLAP-FDINFO] lines so the keys this kernel prints are on
// record.
//
// The copy source is the prefill ring's own source shape: a pinned-pool WEIGHT
// slice of the cache's host zone, which mem_copy_async trusts as host USM and
// copies directly. Any other host source is staged through a synchronous host
// memcpy, and the row would time that instead of the DMA. So the source's
// owner is checked against that trust rule before timing, and every timed call
// reads this thread's staged-H2D census (counted with the trace off) around its
// copies: staged_h2d= on the row is the count over all the row's copy rounds,
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

// The copy shape the plan's spike names: 16 copies of 4 MiB, 64 MiB in all, about 2.7 ms at the probed ~25 GB/s.
static constexpr size_t kChunkBytes   = 4ull * 1024ull * 1024ull;
static constexpr size_t kChunks       = 16;
static constexpr size_t kCopyBytes    = kChunkBytes * kChunks;
// The spin counter: one uint32 the kernel stores, padded to a 64-byte line.
static constexpr size_t kCounterBytes = 64;

// Host zones, configured only when the process has none: a WEIGHT zone with room for two copy sources, and the
// small KV, STAGING and SCRATCH zones test-mem-ops configures (this test allocates nothing from them).
static constexpr size_t kHostWeightZoneBytes  = 2 * kCopyBytes;
static constexpr size_t kHostKvZoneBytes      = 2ull << 20;
static constexpr size_t kHostStagingZoneBytes = 4ull << 20;
static constexpr size_t kHostScratchZoneBytes = 2ull << 20;

// The spin kernel's work-group size: the plan's shape, a multiple of every Battlemage sub-group width.
static constexpr size_t   kSpinGroupSize         = 256;
// spin=full's work-group count: many times the work-groups the 256-CU B70 can hold at once, so every EU stays busy.
static constexpr size_t   kSpinFullGroups        = 4096;
// Calibration doubles a shape's iteration count until one launch takes at least this long, so it lands in
// [floor, 2 x floor) = 80-160 ms: long enough that queueing behind the kernel dwarfs a ~2.7 ms copy round.
static constexpr double   kSpinFloorMs           = 80.0;
// Starting iteration counts, far below the floor for each shape on both cards so calibration climbs to it. spin=full
// starts lower because each of its iterations runs 4096 times as many work-items.
static constexpr uint64_t kSpin1wgStartIters     = 1ull << 20;
static constexpr uint64_t kSpinFullStartIters    = 1ull << 8;
// Calibration stops here: a spin that still misses the floor after 2^36 iterations is not running as written (the
// loop was compiled away, or the device is not executing it), and the test fails rather than time against it.
static constexpr uint64_t kSpinMaxIters          = 1ull << 36;
// overlap=yes when the with-kernel copies took less than alone + this fraction of kernel_ms. Copies queued behind the
// kernel take at least alone + kernel_ms, overlapping ones about alone; half the kernel splits the two.
static constexpr double   kOverlapKernelFraction = 0.5;

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

// The value of `key` in one fdinfo, or "" when it has none.
static std::string fdinfo_value(const std::vector<std::pair<std::string, std::string>> & kv, const char * key) {
    for (const auto & p : kv) {
        if (p.first == key) {
            return p.second;
        }
    }
    return std::string();
}

// Engine-busy counters summed over this process's DRM clients on one device.
struct drm_snapshot {
    std::map<std::string, unsigned long long> cycles;             // drm-cycles-<class>, keyed by <class>
    bool                                      has_ticks = false;  // any drm-total-cycles-<class> was read
    unsigned long long                        ticks     = 0;      // max drm-total-cycles-<class> over the clients
};

// The counters of this process's DRM clients on the device whose PCI address is `pdev`: render-node fds whose
// drm-pdev matches, each drm-client-id once (dup'd fds share a client). Empty, so every cycle field prints NA, when
// `pdev` is empty: summing every render client would count the other cards'.
static drm_snapshot read_drm_snapshot(const std::string & pdev, const std::vector<std::string> & fds) {
    drm_snapshot          snap;
    std::set<std::string> clients;
    if (pdev.empty()) {
        return snap;
    }
    for (const std::string & fd : fds) {
        const std::vector<std::pair<std::string, std::string>> kv     = read_fdinfo(fd);
        const std::string                                      client = fdinfo_value(kv, "drm-client-id");
        if (fdinfo_value(kv, "drm-pdev") != pdev || client.empty() || !clients.insert(client).second) {
            continue;
        }
        for (const auto & p : kv) {
            const unsigned long long v = std::strtoull(p.second.c_str(), nullptr, 10);
            if (p.first.rfind("drm-cycles-", 0) == 0) {
                snap.cycles[p.first.substr(11)] += v;
            } else if (p.first.rfind("drm-total-cycles-", 0) == 0) {
                snap.ticks     = snap.has_ticks && snap.ticks > v ? snap.ticks : v;
                snap.has_ticks = true;
            }
        }
    }
    return snap;
}

// after - before of one counter as text, or "NA" when either side lacks it or it went backwards (a client that
// closed between the snapshots), never a wrapped unsigned difference. `moved` reports a non-zero delta.
static std::string counter_delta_text(bool               has_before,
                                      unsigned long long before,
                                      bool               has_after,
                                      unsigned long long after,
                                      bool *             moved = nullptr) {
    const bool ok = has_before && has_after && after >= before;
    if (moved) {
        *moved = !ok || after != before;
    }
    return ok ? std::to_string(after - before) : std::string("NA");
}

static std::string cycles_delta_text(const drm_snapshot & before,
                                     const drm_snapshot & after,
                                     const std::string &  cls,
                                     bool *               moved = nullptr) {
    const auto b = before.cycles.find(cls);
    const auto a = after.cycles.find(cls);
    return counter_delta_text(b != before.cycles.end(), b == before.cycles.end() ? 0ull : b->second,
                              a != after.cycles.end(), a == after.cycles.end() ? 0ull : a->second, moved);
}

// " <prefix>bcs_cycles=<d> <prefix>ccs_cycles=<d> [<prefix><class>_cycles=<d> ...] <prefix>ticks=<d>": bcs and ccs
// always, any other class only when it moved (or cannot be compared); "<prefix>bcs_cycles=NA <prefix>ccs_cycles=NA"
// without cycles keys.
static std::string drm_delta_fields(const char * prefix, const drm_snapshot & before, const drm_snapshot & after) {
    if (after.cycles.empty()) {
        return std::string(" ") + prefix + "bcs_cycles=NA " + prefix + "ccs_cycles=NA";
    }
    std::string  out;
    const char * always[] = { "bcs", "ccs" };
    for (const char * cls : always) {
        out += std::string(" ") + prefix + cls + "_cycles=" + cycles_delta_text(before, after, cls);
    }
    for (const auto & a : after.cycles) {
        if (a.first == "bcs" || a.first == "ccs") {
            continue;
        }
        bool              moved = false;
        const std::string d     = cycles_delta_text(before, after, a.first, &moved);
        if (moved) {
            out += std::string(" ") + prefix + a.first + "_cycles=" + d;
        }
    }
    out += std::string(" ") + prefix +
           "ticks=" + counter_delta_text(before.has_ticks, before.ticks, after.has_ticks, after.ticks);
    return out;
}

// Which of the cache's queue objects `q` is. Compute first: the DMA accessor itself falls back to the compute queue.
static const char * queue_identity(const sycl::queue * q, ggml_sycl::unified_cache & cache) {
    if (q == &cache.get_queue()) {
        return "compute";
    }
    if (q == &cache.get_dma_queue()) {
        return "dma";
    }
    return "bcs";
}

// Device busy work on `q`: `groups` work-groups of kSpinGroupSize iterating an LCG `iters` times.
static sycl::event spin_kernel(sycl::queue & q, uint32_t * counter, size_t groups, uint64_t iters) {
    return q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::nd_range<1>(groups * kSpinGroupSize, kSpinGroupSize), [=](sycl::nd_item<1> it) {
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

// A row's copy tally over all its rounds: copies submitted, and how many of them took the staged branch.
struct copy_tally {
    uint64_t copies = 0;
    uint64_t staged = 0;
};

// 16 x 4 MiB pinned H2D on `copy_q`; returns wall ms from the first submit to copy_q.wait() and adds to `tally` the
// copies submitted and how many of them took the staged host-to-device branch. The census is per thread and the
// copies are submitted on this one, so the difference is exactly these copies; it is read outside the timed window.
static double time_copies(sycl::queue & copy_q, const mem_handle & dst, const mem_handle & src, copy_tally & tally) {
    const uint64_t staged_before = ggml_sycl::mem_copy_staged_wait_census_this_thread().staged_h2d;
    const auto     t0            = std::chrono::steady_clock::now();
    for (size_t i = 0; i < kChunks; ++i) {
        ggml_sycl::mem_copy_async(dst, i * kChunkBytes, src, i * kChunkBytes, kChunkBytes, copy_q);
    }
    copy_q.wait();
    const double ms = ms_since(t0);
    tally.copies += kChunks;
    tally.staged += ggml_sycl::mem_copy_staged_wait_census_this_thread().staged_h2d - staged_before;
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
        cache->configure_host_zones(kHostWeightZoneBytes, kHostKvZoneBytes, kHostStagingZoneBytes,
                                    kHostScratchZoneBytes);
    }
    if (!cache->host_zones_configured()) {
        std::printf("FAIL: the cache's host zones could not be configured\n");
        return 1;
    }
    sycl::queue & cq = cache->get_queue();

    const mem_handle src = alloc_pinned_weight(cq, device, kCopyBytes);
    const mem_handle dst = alloc_runtime_device(cq, device, kCopyBytes);
    const mem_handle ctr = alloc_runtime_device(cq, device, kCounterBytes);
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

    // The device's PCI address selects its DRM clients. Not reported: every cycle field prints NA, and the dump
    // below shows the first render fd's keys only.
    std::string pdev;
    if (ccs.get_device().has(sycl::aspect::ext_intel_pci_address)) {
        pdev = ccs.get_device().get_info<sycl::ext::intel::info::device::pci_address>();
    }
    // The render-node fds are opened during backend and cache initialisation, all done by now: one scan serves the
    // dump and every snapshot.
    const std::vector<std::string> fds = render_node_fds();
    if (pdev.empty()) {
        std::printf(
            "[COPY-OVERLAP-FDINFO] pdev=unknown render_fds=%zu cycles=NA: the device reports no PCI address, so its "
            "DRM clients cannot be told from the other cards'; the dump below is the first render fd, for its keys\n",
            fds.size());
    } else {
        std::printf("[COPY-OVERLAP-FDINFO] pdev=%s render_fds=%zu\n", pdev.c_str(), fds.size());
    }
    for (const std::string & fd : fds) {
        const std::vector<std::pair<std::string, std::string>> kv = read_fdinfo(fd);
        if (!pdev.empty() && fdinfo_value(kv, "drm-pdev") != pdev) {
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

    // JIT the spin kernel before timing it, then calibrate each shape to [kSpinFloorMs, 2 x kSpinFloorMs).
    spin_kernel(ccs, counter, 1, 1).wait();
    spin spins[] = {
        { "1wg",  1,               kSpin1wgStartIters,  0.0 },
        { "full", kSpinFullGroups, kSpinFullStartIters, 0.0 },
    };
    for (spin & sp : spins) {
        for (;;) {
            const auto t0 = std::chrono::steady_clock::now();
            spin_kernel(ccs, counter, sp.groups, sp.iters).wait();
            sp.kernel_ms = ms_since(t0);
            if (sp.kernel_ms >= kSpinFloorMs || sp.iters >= kSpinMaxIters) {
                break;
            }
            sp.iters *= 2;
        }
        if (sp.kernel_ms < kSpinFloorMs) {
            std::printf(
                "FAIL: spin=%s reached only kernel_ms=%.1f at %llu iterations, under the %.0f ms calibration floor; "
                "every row would read overlap=yes having measured nothing\n",
                sp.name, sp.kernel_ms, (unsigned long long) sp.iters, kSpinFloorMs);
            return 1;
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
            copy_tally tally;
            time_copies(*a.q, dst, src, tally);  // warm-up
            const drm_snapshot alone_before = read_drm_snapshot(pdev, fds);
            const double       alone        = time_copies(*a.q, dst, src, tally);
            const drm_snapshot alone_after  = read_drm_snapshot(pdev, fds);
            const drm_snapshot with_before  = read_drm_snapshot(pdev, fds);
            const auto         t0           = std::chrono::steady_clock::now();
            sycl::event        k            = spin_kernel(ccs, counter, sp.groups, sp.iters);
            const double       with         = time_copies(*a.q, dst, src, tally);
            k.wait();
            const double       total      = ms_since(t0);
            // After the kernel's completion and after total_ms: the window holds the whole kernel on every row, and
            // the read is outside every timed window.
            const drm_snapshot with_after = read_drm_snapshot(pdev, fds);
            const bool         overlap    = with < alone + kOverlapKernelFraction * sp.kernel_ms;
            const char *       resolved   = queue_identity(a.q, *cache);
            std::printf(
                "[COPY-OVERLAP] queue=%s alone_ms=%.1f with_kernel_ms=%.1f kernel_ms=%.1f total_ms=%.1f overlap=%s "
                "adapter=%s spin=%s%s%s staged_h2d=%llu resolved=%s\n",
                a.name, alone, with, sp.kernel_ms, total, overlap ? "yes" : "no", adapter, sp.name,
                drm_delta_fields("alone_", alone_before, alone_after).c_str(),
                drm_delta_fields("", with_before, with_after).c_str(), (unsigned long long) tally.staged, resolved);
            if (tally.staged != 0) {
                std::printf(
                    "FAIL: queue=%s spin=%s: %llu of the row's %llu copies took the staged host-to-device branch, so "
                    "its times are a host memcpy, not the DMA\n",
                    a.name, sp.name, (unsigned long long) tally.staged, (unsigned long long) tally.copies);
                ++failed;
            }
            if (std::string(a.name) == "bcs" && std::string(resolved) == "compute") {
                std::printf(
                    "FAIL: queue=bcs spin=%s: the cache built neither a BCS nor a DMA queue, so the bcs arm timed the "
                    "compute queue\n",
                    sp.name);
                ++failed;
            }
            std::fflush(stdout);
        }
    }
    ccs.wait();
    cq.wait();
    return failed == 0 ? 0 : 1;
}
