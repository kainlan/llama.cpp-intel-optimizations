// Gate for llama.cpp-fz2u: a zone reset issued while that zone still holds a
// live registered allocation must be REFUSED, not fatal.
//
// Background, because the correct behaviour here is narrow and two adjacent
// behaviours are both wrong:
//
//   * 9a0670712 made both host_zone_reset() and zone_reset() GGML_ABORT on a
//     live allocation. That is wrong because several llama_model objects may be
//     loaded at once (tests/test-thread-safety.cpp exercises exactly that), so a
//     live allocation belonging to another still-running model is CORRECT, not a
//     leak.
//   * Before 9a0670712 both purged the registry and reset anyway. That is wrong
//     in the other direction: it recycles addresses still owned by live handles.
//
// The answer is to refuse -- which is what zone_reset() already does for the
// WEIGHT zone and for KV in a shared KV+WEIGHT arena.
//
// Two properties are asserted. The second is the one that is easy to lose and
// impossible to notice:
//
//   1. REFUSAL  -- a reset issued while an allocation in that zone is live
//                  returns without resetting; the zone's accounting is
//                  unchanged and the allocation is still usable.
//   2. NO LATCH -- once the owner releases, the NEXT reset must succeed.
//                  The refusal is only safe because unified_free() drops the
//                  registry record for every zone, including the reset-only
//                  SCRATCH/STAGING host zones that take no per-block free. If
//                  that erase ever becomes conditional on zone, the zone would
//                  never reset again for the lifetime of the process -- silently,
//                  with no abort to point at. Property 1 alone still passes in
//                  that broken world, which is why property 2 is here.
//
// This test deliberately does NOT hardcode which zone an allocation lands in.
// It allocates through the real unified_alloc() path and reads the resulting
// zone back off the handle, so it keeps testing the right zone if routing
// changes. If an allocation turns out not to be zone-managed at all, the test
// FAILS rather than skipping -- a vacuous pass is the failure mode this
// repository keeps hitting.

#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml-sycl/unified-cache.hpp"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>

#if !defined(GGML_USE_SYCL)
int main() {
    fprintf(stderr, "GGML_USE_SYCL not enabled; skipping test.\n");
    return 0;
}
#else

using ggml_sycl::alloc_handle;
using ggml_sycl::alloc_request;
using ggml_sycl::alloc_role;
using ggml_sycl::get_unified_cache_for_device;
using ggml_sycl::host_zone_id;
using ggml_sycl::runtime_category;
using ggml_sycl::unified_alloc;
using ggml_sycl::unified_cache;
using ggml_sycl::unified_cache_host_zone_reset;
using ggml_sycl::unified_cache_host_zone_used;
using ggml_sycl::unified_cache_zone_reset;
using ggml_sycl::unified_free;
using ggml_sycl::vram_zone_id;

namespace {

int g_failures = 0;

void check(bool ok, const char * what) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) {
        g_failures++;
    }
}

// Allocate through the real tracked path so the allocation lands in
// g_runtime_alloc_registry -- which is what both reset paths scan.
//
// Zone routing is driven by alloc_constraints, NOT by role/category: a VRAM zone
// needs `prefer_vram_zone` (unified-cache.cpp, the prefer_vram_zone branch) and a
// host zone needs `must_host_pinned` to reach the host branch at all. Role and
// category only select WHICH host zone once there. Getting this wrong produces an
// allocation that is not zone-managed and reaches neither reset path -- which is
// what the first version of this test did, and why both halves assert
// zone-managed instead of skipping.
bool alloc_tracked(int                                  device,
                   sycl::queue *                        queue,
                   alloc_role                           role,
                   runtime_category                     category,
                   const char *                         cohort,
                   size_t                               size,
                   const ggml_sycl::alloc_constraints & constraints,
                   alloc_handle *                       out) {
    alloc_request req;
    req.queue              = queue;
    req.device             = device;
    req.size               = size;
    req.intent.role        = role;
    req.intent.category    = category;
    req.intent.cohort_id   = cohort;
    req.intent.constraints = constraints;
    return unified_alloc(req, out);
}

// --- VRAM zone -------------------------------------------------------------
//
// zone_used() gives a positive observable: a refused reset must leave it
// unchanged, and a successful reset must drive it to zero. Without that, "did
// not abort" would be the only signal and a reset that silently did nothing
// would look identical to a correct refusal.
void test_vram_zone(int device, unified_cache * cache, sycl::queue * queue) {
    printf("VRAM zone_reset refusal:\n");

    // RUNTIME is a real TLSF zone in the arena and is what arena_reserve() resets
    // on a new context -- the same zone_reset() call this test is gating.
    ggml_sycl::alloc_constraints constraints;
    constraints.prefer_vram_zone = vram_zone_id::RUNTIME;

    alloc_handle handle;
    if (!alloc_tracked(device, queue, alloc_role::COMPUTE, runtime_category::COMPUTE, "fz2u_vram_probe", 64 * 1024,
                       constraints, &handle)) {
        check(false, "tracked VRAM allocation succeeded");
        return;
    }

    const vram_zone_id zone = handle.vram_zone;
    if (zone == vram_zone_id::COUNT) {
        // Not zone-managed: this test cannot exercise zone_reset() at all, and
        // reporting success would be a vacuous pass.
        check(false, "allocation is zone-managed (required to reach zone_reset)");
        unified_free(handle);
        return;
    }

    const size_t used_live = cache->zone_used(zone);
    check(used_live > 0, "zone_used() is non-zero while the allocation is live");

    // Property 1: refusal. Reaching the next line at all is half the assertion --
    // before the fix this aborted the process.
    unified_cache_zone_reset(device, zone);
    check(cache->zone_used(zone) == used_live, "refused reset left zone_used() unchanged");

    // Property 2: no latch.
    unified_free(handle);
    unified_cache_zone_reset(device, zone);
    check(cache->zone_used(zone) == 0, "reset succeeds once the owner has released (no latch)");
}

// --- Host zone -------------------------------------------------------------
//
// Uses the same role/category/cohort shape as the allocation that actually
// triggered the abort in test-thread-safety (role=STAGING, category=STAGING,
// host_zone=STAGING, cohort=get_rows_indices_small_host).
void test_host_zone(int device, sycl::queue * queue) {
    printf("host_zone_reset refusal:\n");

    // must_host_pinned is what routes into the host branch; role/category then
    // select which host zone. use_pinned_pool matches how the real staging
    // allocations that triggered the abort are made.
    ggml_sycl::alloc_constraints constraints;
    constraints.must_host_pinned = true;
    constraints.use_pinned_pool  = true;

    alloc_handle handle;
    if (!alloc_tracked(device, queue, alloc_role::STAGING, runtime_category::STAGING, "fz2u_host_probe", 4096,
                       constraints, &handle)) {
        check(false, "tracked host-zone allocation succeeded");
        return;
    }

    const host_zone_id zone = handle.host_zone;
    if (zone == host_zone_id::COUNT) {
        check(false, "allocation is host-zone-managed (required to reach host_zone_reset)");
        unified_free(handle);
        return;
    }
    if (zone == host_zone_id::WEIGHT) {
        // host_zone_reset() asserts on WEIGHT by contract; nothing to test.
        check(false, "allocation landed outside the WEIGHT zone");
        unified_free(handle);
        return;
    }

    const size_t used_live = unified_cache_host_zone_used(zone);
    check(used_live > 0, "host_zone_used() is non-zero while the allocation is live");

    // Property 1: refusal. Surviving this call is half the assertion -- before
    // the fix it aborted -- but "did not abort" cannot tell a correct refusal
    // apart from a reset that silently did nothing, so assert the zone's
    // accounting is untouched as well.
    unified_cache_host_zone_reset(zone);
    check(unified_cache_host_zone_used(zone) == used_live, "refused reset left host_zone_used() unchanged");

    // Property 2: no latch. The reset-only host zones (SCRATCH/STAGING) take no
    // per-block free, so this leans entirely on unified_free() still dropping
    // the registry record for them.
    unified_free(handle);
    unified_cache_host_zone_reset(zone);
    check(unified_cache_host_zone_used(zone) == 0, "reset succeeds once the owner has released (no latch)");
}

}  // namespace

int main() {
    if (!getenv("ONEAPI_DEVICE_SELECTOR")) {
        // Match the sibling SYCL gates: pin the validation card so a bare ctest
        // run cannot perturb a measurement on the other GPU. ctest also sets
        // this via ENVIRONMENT; this is only a fallback for bare invocation and
        // must be set before the SYCL runtime enumerates devices.
        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:1", 1);
    }

    const int device = 0;  // in-process index after selector filtering

    // Bring the backend up. Without this the device/queues do not exist.
    ggml_backend_t backend = ggml_backend_sycl_init(device);
    if (backend == nullptr) {
        fprintf(stderr, "ggml_backend_sycl_init(%d) failed; cannot run.\n", device);
        return 1;
    }

    unified_cache * cache = get_unified_cache_for_device(device);
    if (cache == nullptr) {
        fprintf(stderr, "no unified cache for device %d; cannot run.\n", device);
        ggml_backend_free(backend);
        return 1;
    }
    sycl::queue * queue = &cache->get_queue();

    // Zones are normally configured as a side effect of a model load, which this
    // test deliberately does not do -- loading a model would make it as heavy as
    // test-thread-safety, which is the gate this one exists to replace. Activate
    // them explicitly instead. Without this, unified_alloc() returns allocations
    // that are not zone-managed, and neither reset path is reachable: the first
    // version of this test failed for exactly that reason rather than passing
    // vacuously, which is why both halves assert zone-managed rather than skip.
    if (!cache->host_zones_configured()) {
        cache->configure_host_zones(/*weight*/ 0, /*kv*/ 16u << 20, /*staging*/ 16u << 20,
                                    /*scratch*/ 16u << 20);
    }
    if (!cache->arena_active()) {
        ggml_sycl::unified_cache_reserve_compute_arena(device, 64u << 20);
    }
    printf("setup: host_zones_configured=%d arena_active=%d\n", cache->host_zones_configured() ? 1 : 0,
           cache->arena_active() ? 1 : 0);

    test_vram_zone(device, cache, queue);
    test_host_zone(device, queue);

    ggml_backend_free(backend);

    printf("%s: %d failure(s)\n", g_failures == 0 ? "OK" : "FAILED", g_failures);
    return g_failures == 0 ? 0 : 1;
}

#endif  // GGML_USE_SYCL
