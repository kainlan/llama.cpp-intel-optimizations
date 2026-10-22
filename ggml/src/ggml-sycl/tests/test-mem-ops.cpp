#include "../../../../tests/test-skip.h"  // LLAMA_TEST_EXIT_SKIP: the one definition of "77 means skip"
#include "../common.hpp"
#include "../mem-handle.hpp"
#include "../mem-ops.hpp"
#include "../unified-cache.hpp"

#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

static bool expect_fatal(const char * label, const std::function<void()> & operation) {
    const pid_t child = fork();
    if (child == 0) {
        operation();
        _exit(0);
    }
    if (child < 0) {
        std::fprintf(stderr, "FAIL: %s fork failed\n", label);
        return false;
    }
    int status = 0;
    if (waitpid(child, &status, 0) != child || (WIFEXITED(status) && WEXITSTATUS(status) == 0)) {
        std::fprintf(stderr, "FAIL: %s did not terminate fatally\n", label);
        return false;
    }
    return true;
}

static bool check_bytes(const uint8_t * data, size_t size, uint8_t expected, const char * label) {
    for (size_t i = 0; i < size; ++i) {
        if (data[i] != expected) {
            std::fprintf(stderr, "FAIL: %s byte %zu expected 0x%02x got 0x%02x\n", label, i, expected, data[i]);
            return false;
        }
    }
    return true;
}

// The th86 copy site census as mem_copy_site_census_flush prints it to stderr, read back through a temporary file.
static std::string capture_site_census() {
    std::string text;
    FILE *      f = std::tmpfile();
    if (!f) {
        return text;
    }
    std::fflush(stderr);
    const int saved = dup(fileno(stderr));
    if (saved < 0 || dup2(fileno(f), fileno(stderr)) < 0) {
        std::fclose(f);
        return text;
    }
    ggml_sycl::mem_copy_site_census_flush("test-mem-ops");
    std::fflush(stderr);
    dup2(saved, fileno(stderr));
    close(saved);
    std::rewind(f);
    char   buf[4096];
    size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) {
        text.append(buf, n);
    }
    std::fclose(f);
    return text;
}

// llama.cpp-36j0: a pinned-pool host buffer with role WEIGHT, the shape S1-PRELOAD gives a host-planned expert
// (register_host_aos_expert in ggml-sycl.cpp: role WEIGHT, no category), is a trusted USM copy endpoint, so its
// host->device copy is one direct DMA. Taking the staged branch instead would count staged_h2d and wait the copy's
// dependency on this thread (H2D_DEPS). Category OTHER, not EXPERT_CACHE: the expert-path category was already in
// scope, so a copy from it would pass without the change this case checks. Runs first in main: host zones configured
// after the pinned pool has handed out runtime chunks (every staged copy's stage) can fail on the pool budget.
static int check_pinned_weight_copy_is_direct(sycl::queue & q) {
    constexpr size_t       mib       = 1024ull * 1024ull;
    constexpr size_t       bytes     = 16 * mib;
    constexpr const char * site_file = "test-mem-ops-pinned-weight";
    constexpr int          site_line = 1;
    const char * const     site_tag  = "site=test-mem-ops-pinned-weight:1 ";

    if (!ggml_sycl_cpu_expert_trace_enabled()) {
        std::fprintf(stderr,
                     "FAIL: pinned WEIGHT copy: GGML_SYCL_MOE_IDS_COPY_TRACE is off, so the staged-copy census "
                     "would read zero whatever branch ran\n");
        return 1;
    }
    ggml_sycl::unified_cache * cache = ggml_sycl::get_unified_cache(q);
    if (!cache) {
        std::fprintf(stderr, "FAIL: pinned WEIGHT copy: no unified cache for the device\n");
        return 1;
    }
    if (!cache->host_zones_configured()) {
        cache->configure_host_zones(2 * bytes, 2 * mib, 4 * mib, 2 * mib);
    }
    if (!cache->host_zones_configured()) {
        std::fprintf(stderr, "FAIL: pinned WEIGHT copy: host zones could not be configured\n");
        return 1;
    }
    sycl::queue & cq     = cache->get_queue();
    const int     device = ggml_sycl_get_device_id_from_queue(cq);

    ggml_sycl::alloc_request host_req{};
    host_req.queue                               = &cq;
    host_req.device                              = device;
    host_req.size                                = bytes;
    host_req.intent.role                         = ggml_sycl::alloc_role::WEIGHT;
    host_req.intent.category                     = ggml_sycl::runtime_category::OTHER;
    host_req.intent.constraints.must_host_pinned = true;
    host_req.intent.constraints.use_pinned_pool  = true;
    ggml_sycl::mem_handle     src                = ggml_sycl::unified_allocate(host_req);
    ggml_sycl::alloc_metadata owner{};
    if (!src.valid() || !src.owner_metadata(&owner) || owner.tier != ggml_sycl::alloc_tier::HOST_PINNED ||
        owner.host_zone == ggml_sycl::host_zone_id::COUNT || owner.role != ggml_sycl::alloc_role::WEIGHT ||
        owner.device < 0) {
        std::fprintf(stderr,
                     "FAIL: pinned WEIGHT copy: the source is not a zone-managed pinned-pool WEIGHT owner "
                     "(valid=%d tier=%d host_zone=%d role=%d device=%d)\n",
                     src.valid() ? 1 : 0, (int) owner.tier, (int) owner.host_zone, (int) owner.role, owner.device);
        return 1;
    }
    const ggml_sycl::resolved_ptr src_ptr = src.resolve();
    if (!src_ptr || src_ptr.on_device) {
        std::fprintf(stderr, "FAIL: pinned WEIGHT copy: the source did not resolve to host memory\n");
        return 1;
    }
    uint8_t * host = static_cast<uint8_t *>(src_ptr.ptr);
    for (size_t i = 0; i < bytes; ++i) {
        host[i] = static_cast<uint8_t>((i * 31u + 7u) & 0xffu);
    }

    ggml_sycl::alloc_request dev_req{};
    dev_req.queue                                     = &cq;
    dev_req.device                                    = device;
    dev_req.size                                      = bytes;
    dev_req.intent.role                               = ggml_sycl::alloc_role::STAGING;
    dev_req.intent.category                           = ggml_sycl::runtime_category::STAGING;
    dev_req.intent.constraints.must_device            = true;
    dev_req.intent.constraints.prefer_vram_zone       = ggml_sycl::vram_zone_id::RUNTIME;
    dev_req.intent.constraints.forbid_vram_zone_spill = true;
    ggml_sycl::mem_handle dst                         = ggml_sycl::unified_allocate(dev_req);
    if (!dst.valid()) {
        std::fprintf(stderr,
                     "FAIL: pinned WEIGHT copy: the RUNTIME zone refused the %zu-byte device buffer "
                     "(arena_active=%d zone_available=%zu)\n",
                     bytes, cache->arena_active() ? 1 : 0, cache->zone_available(ggml_sycl::vram_zone_id::RUNTIME));
        return 1;
    }

    int                                          failed = 0;
    sycl::event                                  fill   = ggml_sycl::mem_fill_async(dst, 0, bytes, cq);
    const ggml_sycl::mem_copy_staged_wait_census before = ggml_sycl::mem_copy_staged_wait_census_this_thread();
    // The copy carries a dependency so that the staged branch, had it run, would also have waited it on this thread.
    sycl::event h2d = ggml_sycl::mem_copy_async(dst, src, bytes, cq, { fill }, site_file, site_line, __func__);
    h2d.wait_and_throw();
    const ggml_sycl::mem_copy_staged_wait_census after = ggml_sycl::mem_copy_staged_wait_census_this_thread();

    if (after.staged_h2d != before.staged_h2d) {
        std::fprintf(stderr, "FAIL: pinned WEIGHT copy took the staged host->device branch (%llu staged copies)\n",
                     (unsigned long long) (after.staged_h2d - before.staged_h2d));
        failed++;
    }
    const int staged_kinds[] = { ggml_sycl::MEM_COPY_STAGED_WAIT_KIND_H2D_DEPS,
                                 ggml_sycl::MEM_COPY_STAGED_WAIT_KIND_H2D_CHUNK,
                                 ggml_sycl::MEM_COPY_STAGED_WAIT_KIND_H2D_FINAL };
    for (int kind : staged_kinds) {
        if (after.count[kind] != before.count[kind]) {
            std::fprintf(stderr, "FAIL: pinned WEIGHT copy recorded %llu staged wait(s) of kind %d\n",
                         (unsigned long long) (after.count[kind] - before.count[kind]), kind);
            failed++;
        }
    }

    // Exactly one copy left this site, a host->device submission of all the bytes: one census row, one call.
    const std::string table = capture_site_census();
    if (table.find("[SYCL-MEMCPY-SITE-CENSUS]") == std::string::npos) {
        std::fprintf(stderr,
                     "FAIL: pinned WEIGHT copy: the copy site census printed nothing "
                     "(GGML_SYCL_MEMCPY_SITE_CENSUS off?)\n");
        failed++;
    } else {
        int         rows = 0;
        std::string row;
        for (size_t at = 0; at < table.size();) {
            size_t end = table.find('\n', at);
            if (end == std::string::npos) {
                end = table.size();
            }
            const std::string line = table.substr(at, end - at);
            if (line.find(site_tag) != std::string::npos) {
                rows++;
                row = line;
            }
            at = end + 1;
        }
        const std::string want = " calls=1 bytes=" + std::to_string(bytes) + " dir=H2D path=sycl.memcpy.mem_ops ";
        if (rows != 1 || row.find(want) == std::string::npos) {
            std::fprintf(stderr, "FAIL: pinned WEIGHT copy: expected one census row with%s, got %d (last: %s)\n",
                         want.c_str(), rows, row.c_str());
            failed++;
        }
    }

    std::vector<uint8_t> out(bytes, 0);
    ggml_sycl::mem_copy(ggml_sycl::mem_handle::from_direct(out.data(), GGML_LAYOUT_AOS, false,
                                                           ggml_sycl::mem_handle::HOST_DEVICE, bytes),
                        dst, bytes, cq);
    for (size_t i = 0; i < bytes; ++i) {
        if (out[i] != static_cast<uint8_t>((i * 31u + 7u) & 0xffu)) {
            std::fprintf(stderr, "FAIL: pinned WEIGHT copy: byte %zu read back 0x%02x\n", i, out[i]);
            failed++;
            break;
        }
    }
    return failed;
}

// llama.cpp-36j0: the positive control for the real copy source. S1-PRELOAD mints a host-planned expert's storage
// owner-first (ggml_sycl_host_expert_alloc: role WEIGHT, pinned pool, no category) and registers it with
// register_host_expert's typed overload. The lease that registration hands out must answer owner_metadata() with
// that allocation's HOST_PINNED/WEIGHT provenance on its own, after the caller's owner handle is gone, and a copy
// from the lease must be one direct DMA: no staged_h2d and no staged wait, with a dependency passed.
static int check_registered_host_expert_lease_copy_is_direct(sycl::queue & q) {
    constexpr size_t bytes = 4ull * 1024ull * 1024ull;

    if (!ggml_sycl_cpu_expert_trace_enabled()) {
        std::fprintf(stderr, "FAIL: host expert lease copy: GGML_SYCL_MOE_IDS_COPY_TRACE is off\n");
        return 1;
    }
    ggml_sycl::unified_cache * cache = ggml_sycl::get_unified_cache(q);
    if (!cache) {
        std::fprintf(stderr, "FAIL: host expert lease copy: no unified cache for the device\n");
        return 1;
    }
    // Configured here as check_pinned_weight_copy_is_direct does, so this case does not rely on running after it.
    if (!cache->host_zones_configured()) {
        cache->configure_host_zones(2 * bytes, 2ull * 1024ull * 1024ull, 4ull * 1024ull * 1024ull,
                                    2ull * 1024ull * 1024ull);
    }
    if (!cache->host_zones_configured()) {
        std::fprintf(stderr, "FAIL: host expert lease copy: host zones could not be configured\n");
        return 1;
    }
    sycl::queue & cq     = cache->get_queue();
    const int     device = ggml_sycl_get_device_id_from_queue(cq);

    // The request S1-PRELOAD's ggml_sycl_host_expert_alloc makes.
    ggml_sycl::alloc_request host_req{};
    host_req.device                              = device;
    host_req.size                                = bytes;
    host_req.intent.role                         = ggml_sycl::alloc_role::WEIGHT;
    host_req.intent.constraints.must_host_pinned = true;
    host_req.intent.constraints.use_pinned_pool  = true;
    ggml_sycl::allocation_result allocation      = ggml_sycl::unified_allocate_owner(host_req);
    if (!allocation) {
        std::fprintf(stderr, "FAIL: host expert lease copy: the pinned pool refused the %zu-byte expert\n", bytes);
        return 1;
    }
    ggml_sycl::mem_handle owner_handle =
        ggml_sycl::mem_handle::from_owned_alloc(std::move(allocation.owner), GGML_LAYOUT_AOS);
    const ggml_sycl::resolved_ptr host_ptr = owner_handle.resolve();
    if (!host_ptr || host_ptr.on_device) {
        std::fprintf(stderr, "FAIL: host expert lease copy: the expert did not resolve to host memory\n");
        return 1;
    }
    uint8_t * host = static_cast<uint8_t *>(host_ptr.ptr);
    for (size_t i = 0; i < bytes; ++i) {
        host[i] = static_cast<uint8_t>((i * 13u + 5u) & 0xffu);
    }

    static const char  key_tag = 0;
    ggml_sycl_cache_id key{};
    key.valid    = true;
    key.model_id = 1;
    key.aux_id   = reinterpret_cast<uintptr_t>(&key_tag);
    ggml_sycl::mem_handle lease;
    bool                  registered = false;
    {
        ggml_sycl::scoped_planned_materialization planned(cache, "test-mem-ops/host expert registration");
        registered = cache->register_host_expert(key, host, bytes, GGML_LAYOUT_AOS, &lease, owner_handle);
    }
    owner_handle = {};  // From here only the entry and the lease hold the expert.
    if (!registered || !lease.valid() || lease.resolve().ptr != host) {
        std::fprintf(stderr, "FAIL: host expert lease copy: registration failed or the lease does not resolve to it\n");
        return 1;
    }

    int                       failed = 0;
    ggml_sycl::alloc_metadata owner{};
    if (!lease.owner_metadata(&owner) || owner.tier != ggml_sycl::alloc_tier::HOST_PINNED ||
        owner.host_zone == ggml_sycl::host_zone_id::COUNT || owner.role != ggml_sycl::alloc_role::WEIGHT ||
        owner.device < 0) {
        std::fprintf(stderr,
                     "FAIL: host expert lease copy: the lease does not report a zone-managed pinned-pool WEIGHT owner "
                     "(tier=%d host_zone=%d role=%d device=%d)\n",
                     (int) owner.tier, (int) owner.host_zone, (int) owner.role, owner.device);
        failed++;
    }

    ggml_sycl::alloc_request dev_req{};
    dev_req.queue                                     = &cq;
    dev_req.device                                    = device;
    dev_req.size                                      = bytes;
    dev_req.intent.role                               = ggml_sycl::alloc_role::STAGING;
    dev_req.intent.category                           = ggml_sycl::runtime_category::STAGING;
    dev_req.intent.constraints.must_device            = true;
    dev_req.intent.constraints.prefer_vram_zone       = ggml_sycl::vram_zone_id::RUNTIME;
    dev_req.intent.constraints.forbid_vram_zone_spill = true;
    ggml_sycl::mem_handle dst                         = ggml_sycl::unified_allocate(dev_req);
    if (!dst.valid()) {
        std::fprintf(stderr, "FAIL: host expert lease copy: the RUNTIME zone refused the %zu-byte device buffer\n",
                     bytes);
        failed++;
    } else {
        sycl::event                                  fill   = ggml_sycl::mem_fill_async(dst, 0, bytes, cq);
        const ggml_sycl::mem_copy_staged_wait_census before = ggml_sycl::mem_copy_staged_wait_census_this_thread();
        sycl::event                                  h2d =
            ggml_sycl::mem_copy_async(dst, lease, bytes, cq, { fill }, "test-mem-ops-host-expert-lease", 1, __func__);
        h2d.wait_and_throw();
        const ggml_sycl::mem_copy_staged_wait_census after = ggml_sycl::mem_copy_staged_wait_census_this_thread();
        if (after.staged_h2d != before.staged_h2d) {
            std::fprintf(stderr, "FAIL: host expert lease copy took the staged host->device branch (%llu)\n",
                         (unsigned long long) (after.staged_h2d - before.staged_h2d));
            failed++;
        }
        const int staged_kinds[] = { ggml_sycl::MEM_COPY_STAGED_WAIT_KIND_H2D_DEPS,
                                     ggml_sycl::MEM_COPY_STAGED_WAIT_KIND_H2D_CHUNK,
                                     ggml_sycl::MEM_COPY_STAGED_WAIT_KIND_H2D_FINAL };
        for (int kind : staged_kinds) {
            if (after.count[kind] != before.count[kind]) {
                std::fprintf(stderr, "FAIL: host expert lease copy recorded %llu staged wait(s) of kind %d\n",
                             (unsigned long long) (after.count[kind] - before.count[kind]), kind);
                failed++;
            }
        }
        std::vector<uint8_t> out(bytes, 0);
        ggml_sycl::mem_copy(ggml_sycl::mem_handle::from_direct(out.data(), GGML_LAYOUT_AOS, false,
                                                               ggml_sycl::mem_handle::HOST_DEVICE, bytes),
                            dst, bytes, cq);
        for (size_t i = 0; i < bytes; ++i) {
            if (out[i] != static_cast<uint8_t>((i * 13u + 5u) & 0xffu)) {
                std::fprintf(stderr, "FAIL: host expert lease copy: byte %zu read back 0x%02x\n", i, out[i]);
                failed++;
                break;
            }
        }
    }

    lease = {};
    if (cache->retire_expert_entry_exact(key, GGML_LAYOUT_AOS, "test-mem-ops") !=
        ggml_sycl::expert_retire_status::WITHDRAWN) {
        std::fprintf(stderr, "FAIL: host expert lease copy: the registered expert was not withdrawn\n");
        failed++;
    }
    cache->process_deferred_frees_public();
    return failed;
}

int main() {
    // Before any copy: both diagnostics latch on their first read. The staged-copy census is the trace's
    // (llama.cpp-qwn5) and the copy site census is th86's; check_pinned_weight_copy_is_direct reads both.
    setenv("GGML_SYCL_MOE_IDS_COPY_TRACE", "1", 1);
    setenv("GGML_SYCL_MEMCPY_SITE_CENSUS", "1", 1);

    std::vector<sycl::device> gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (gpus.empty()) {
        // Exit 77, not 0. A device test that cannot reach a device has verified NOTHING,
        // and returning 0 made that indistinguishable from a pass -- run this binary
        // directly in a shell where setvars.sh was never sourced and it printed one line
        // and exited green in 0.4 s. 77 is ctest's SKIP_RETURN_CODE, so ctest reports
        // *skipped* rather than *passed*, and a bare shell run gets a non-zero status.
        // The point is to make the skip visible as a skip, not to forbid skipping: a
        // CPU-only runner still legitimately lands here. See llama.cpp-k208.
        //
        // ⚠️ 77 only reads as "skipped" once this test's registration carries
        // SKIP_RETURN_CODE 77; ggml/src/ggml-sycl/CMakeLists.txt:1274 does not yet.
        std::fprintf(stderr,
                     "SKIP: no SYCL GPU devices available -- NO DEVICE WORK WAS PERFORMED.\n"
                     "      On a machine that has a GPU this means the oneAPI runtime was not on the\n"
                     "      library path: source /opt/intel/oneapi/setvars.sh --force and re-run.\n");
        return LLAMA_TEST_EXIT_SKIP;
    }

    sycl::queue      q(gpus.front(), sycl::property::queue::in_order{});
    constexpr size_t size = 4096;

    int failed = check_pinned_weight_copy_is_direct(q);
    failed += check_registered_host_expert_lease_copy_is_direct(q);

    uint8_t * host_a = sycl::malloc_host<uint8_t>(size, q);
    uint8_t * host_b = sycl::malloc_host<uint8_t>(size, q);
    uint8_t * dev_a  = sycl::malloc_device<uint8_t>(size, q);
    uint8_t * dev_b  = sycl::malloc_device<uint8_t>(size, q);
    if (!host_a || !host_b || !dev_a || !dev_b) {
        std::fprintf(stderr, "FAIL: allocation failed\n");
        sycl::free(host_a, q);
        sycl::free(host_b, q);
        sycl::free(dev_a, q);
        sycl::free(dev_b, q);
        return 1;
    }

    auto host_a_h = ggml_sycl::mem_handle::from_direct(
        host_a, GGML_LAYOUT_AOS, false, ggml_sycl::mem_handle::HOST_DEVICE, size);
    auto host_b_h = ggml_sycl::mem_handle::from_direct(
        host_b, GGML_LAYOUT_AOS, false, ggml_sycl::mem_handle::HOST_DEVICE, size);
    auto dev_a_h = ggml_sycl::mem_handle::from_direct(dev_a, GGML_LAYOUT_AOS, true, 0, size);
    auto dev_b_h = ggml_sycl::mem_handle::from_direct(dev_b, GGML_LAYOUT_AOS, true, 0, size);

    std::memset(host_a, 0x11, size);
    std::memset(host_b, 0x00, size);
    ggml_sycl::mem_copy(host_b_h, host_a_h, size, q);
    failed += !check_bytes(host_b, size, 0x11, "H2H copy");

    std::memset(host_a, 0x22, size);
    auto h2d = ggml_sycl::mem_copy_async(dev_a_h, host_a_h, size, q);
    auto d2h = ggml_sycl::mem_copy_async(host_b_h, dev_a_h, size, q, { h2d });
    d2h.wait_and_throw();
    failed += !check_bytes(host_b, size, 0x22, "H2D/D2H copy");

    auto fill = ggml_sycl::mem_fill_async(dev_a_h, 0x33, size, q);
    auto d2d  = ggml_sycl::mem_copy_async(dev_b_h, dev_a_h, size, q, { fill });
    auto back = ggml_sycl::mem_copy_async(host_b_h, dev_b_h, size, q, { d2d });
    back.wait_and_throw();
    failed += !check_bytes(host_b, size, 0x33, "fill+D2D+D2H chain");

    // Temporary handles may die immediately after submission.  mem_ops must
    // keep any handle-owned lease alive until the returned event completes.
    auto temp_fill = ggml_sycl::mem_fill_async(
        ggml_sycl::mem_handle::from_direct(dev_a, GGML_LAYOUT_AOS, true, 0, size), 0x44, size, q);
    auto temp_copy = ggml_sycl::mem_copy_async(
        ggml_sycl::mem_handle::from_direct(
            host_b, GGML_LAYOUT_AOS, false, ggml_sycl::mem_handle::HOST_DEVICE, size),
        ggml_sycl::mem_handle::from_direct(dev_a, GGML_LAYOUT_AOS, true, 0, size), size, q, { temp_fill });
    temp_copy.wait_and_throw();
    failed += !check_bytes(host_b, size, 0x44, "temporary handle event lifetime");

    // Exact boundary succeeds; one-byte offset and SIZE_MAX offset arithmetic
    // are rejected before any queue submission.
    ggml_sycl::mem_copy(host_b_h, size - 1, host_a_h, size - 1, 1, q);
    failed += !expect_fatal("source extent overflow", [&] {
        ggml_sycl::mem_copy(host_b_h, 0, host_a_h, size - 1, 2, q);
    });
    failed += !expect_fatal("destination offset overflow", [&] {
        ggml_sycl::mem_fill(host_b_h, SIZE_MAX, 0, 1, q);
    });

    ggml_sycl::alloc_request req{};
    req.queue                               = &q;
    req.size                                = size;
    req.intent.role                         = ggml_sycl::alloc_role::STAGING;
    req.intent.category                     = ggml_sycl::runtime_category::STAGING;
    req.intent.constraints.must_device      = true;
    req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::WEIGHT;

    ggml_sycl::alloc_handle unified_alloc{};
    if (!ggml_sycl::unified_alloc(req, &unified_alloc) || !unified_alloc.ptr) {
        std::fprintf(stderr, "FAIL: unified_alloc device buffer failed\n");
        failed++;
    } else {
        ggml_sycl::mem_handle unified_h    = ggml_sycl::detail::from_legacy_owned_alloc(std::move(unified_alloc));
        auto                  unified_fill = ggml_sycl::mem_fill_async(unified_h, 0x55, size, q);
        auto                  unified_back = ggml_sycl::mem_copy_async(host_b_h, unified_h, size, q, { unified_fill });
        unified_back.wait_and_throw();
        failed += !check_bytes(host_b, size, 0x55, "unified_alloc-owned fill+D2H");
    }

    // llama.cpp-4do9 (TKV-9): the census over tiered_kv_buffer_set_tensor and
    // the tp-buffer get/set functions found every ggml_sycl_copy_handle_for_raw_ptr
    // call site already declaring its byte contract (fixed by c1f4504c8; see
    // that commit's message for the family of sites this guards). This locks
    // in the underlying primitive's trusted-extent resolution directly: an
    // unregistered external pointer with NO declared byte contract must
    // resolve to extent 0 (the exact fxrg failure shape -- llama.cpp-fxrg),
    // and the same pointer WITH a byte contract must resolve to that declared
    // extent. It does NOT exercise the call sites above, and dropping the
    // `, size` argument at any of them would still leave this test green --
    // call-site discipline is source-review (census) territory, not something
    // this test can catch structurally.
    {
        std::vector<uint8_t> unregistered_buf(128, 0xAB);

        ggml_sycl::mem_handle no_contract =
            ggml_sycl_memcpy_handle_for_raw_ptr(unregistered_buf.data(), /*fallback_device=*/-1, GGML_LAYOUT_AOS,
                                                /*fallback_on_device=*/false, /*fallback_unknown=*/false,
                                                /*trusted_extent=*/0);
        ggml_sycl::resolved_ptr no_contract_resolved = no_contract.resolve();
        if (no_contract_resolved.extent != 0) {
            std::fprintf(stderr, "FAIL: unregistered pointer with no byte contract must resolve to extent 0, got %zu\n",
                         no_contract_resolved.extent);
            failed++;
        }

        constexpr size_t      contract_bytes = 128;
        ggml_sycl::mem_handle with_contract =
            ggml_sycl_memcpy_handle_for_raw_ptr(unregistered_buf.data(), /*fallback_device=*/-1, GGML_LAYOUT_AOS,
                                                /*fallback_on_device=*/false, /*fallback_unknown=*/false,
                                                /*trusted_extent=*/contract_bytes);
        ggml_sycl::resolved_ptr with_contract_resolved = with_contract.resolve();
        if (with_contract_resolved.ptr != unregistered_buf.data() || with_contract_resolved.extent != contract_bytes) {
            std::fprintf(stderr,
                         "FAIL: unregistered pointer with a %zu-byte contract must resolve to that extent "
                         "(got ptr=%p extent=%zu)\n",
                         contract_bytes, with_contract_resolved.ptr, with_contract_resolved.extent);
            failed++;
        }
    }

    sycl::free(host_a, q);
    sycl::free(host_b, q);
    sycl::free(dev_a, q);
    sycl::free(dev_b, q);

    if (failed) {
        std::fprintf(stderr, "FAIL: %d mem_ops checks failed\n", failed);
        return 1;
    }
    std::printf("PASS: mem_ops H2H/H2D/D2H/D2D/dependency checks passed\n");
    return 0;
}
