// Gate for llama.cpp-kpjw: a compute buffer the RUNTIME zone will not serve is placed in the arena's KV zone, and it
// is published and freed as a KV-zone allocation.
//
// What the host-only source-contract gate cannot see, and the reason this test exists: that the allocation actually
// lands where the allocator says, is accounted against the zone it landed in, and returns every byte to THAT zone when
// it is freed. A placement published under the wrong zone id (the request's RUNTIME zone instead of the KV zone it
// physically sits in) frees through the wrong allocator, which at best leaks the KV bytes and at worst corrupts the
// RUNTIME zone; no source pattern distinguishes the two as well as one allocate / free round trip does.
//
// Asserted, all through the real unified_alloc() path (no model is loaded):
//
//   1. ZONE-FULL  -- a flagged RUNTIME request larger than the whole RUNTIME zone lands in the KV zone: the handle
//                    reports the KV zone, KV zone_used grows by at least the request, RUNTIME zone_used does not move,
//                    and the placement is counted as a zone-full KV placement (not a hold spill).
//   2. FREE       -- freeing it returns KV zone_used to exactly where it started, RUNTIME untouched.
//   3. HOLD       -- a flagged request the planned-scratch hold keeps out of the RUNTIME zone also lands in the KV
//                    zone, and is counted as an in-arena hold spill, never a raw one.
//   4. CONTROL    -- the same requests WITHOUT the flag never land in the KV zone (the flag is the only thing that
//                    sends a request there), so a pass above is the flag's doing and not a coincidence of routing.
//
// A precondition that cannot be established (no arena, no KV room) FAILS the test rather than skipping it: a vacuous
// pass is the failure mode this repository keeps hitting.

#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml-sycl/unified-cache.hpp"
#include "ggml.h"
#include "sycl-selector-fallback.hpp"

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
using ggml_sycl::planned_hold_spill_totals;
using ggml_sycl::runtime_category;
using ggml_sycl::unified_alloc;
using ggml_sycl::unified_cache;
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

// A RUNTIME-zone compute-buffer request, the shape ggml_backend_sycl_buffer_type_alloc_buffer makes for a scheduler
// compute buffer; `kv_first` is the flag under test.
bool alloc_compute(int device, sycl::queue * queue, size_t size, bool kv_first, alloc_handle * out) {
    alloc_request req;
    req.queue                                      = queue;
    req.device                                     = device;
    req.size                                       = size;
    req.alignment                                  = 64;
    req.intent.role                                = alloc_role::COMPUTE;
    req.intent.category                            = runtime_category::COMPUTE;
    req.intent.cohort_id                           = "kpjw_kv_zone_probe";
    req.intent.constraints.must_device             = true;
    req.intent.constraints.prefer_vram_zone        = vram_zone_id::RUNTIME;
    req.intent.constraints.spill_to_kv_zone_before_raw = kv_first;
    return unified_alloc(req, out);
}

}  // namespace

int main(int, char ** argv) {
    // Match the sibling SYCL gates: pin the validation card; ctest also sets this via ENVIRONMENT.
    sycl_test_selector_fallback(argv, "level_zero:1");

    const int device = 0;  // in-process index after selector filtering

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
    if (!cache->arena_active()) {
        ggml_sycl::unified_cache_reserve_compute_arena(device, 64u << 20);
    }
    printf("setup: arena_active=%d\n", cache->arena_active() ? 1 : 0);
    check(cache->arena_active(), "the VRAM arena is active (required: the KV zone lives in it)");

    // A request larger than the whole RUNTIME zone can never be served by it, whatever is live in it.
    const size_t runtime_cap = cache->zone_capacity(vram_zone_id::RUNTIME);
    const size_t big         = runtime_cap + (1u << 20);
    const size_t small       = 1u << 20;
    check(runtime_cap > 0, "the RUNTIME zone exists");
    check(cache->zone_available(vram_zone_id::KV) >= big + small,
          "the KV zone has room for the probe (required: nothing is proven without it)");
    if (g_failures != 0) {
        ggml_backend_free(backend);
        printf("FAILED: %d failure(s) (a precondition did not hold)\n", g_failures);
        return 1;
    }

    // The counters are per owner: publish an empty hold as a fresh context would, and start an epoch.
    const uint64_t owner = ggml_sycl::unified_cache_mint_planned_scratch_owner();
    ggml_sycl::unified_cache_set_planned_scratch_hold(device, 0, owner);
    ggml_sycl::unified_cache_begin_planned_hold_epoch(device, owner, 512);

    printf("zone-full request, flagged:\n");
    {
        const size_t kv_before  = cache->zone_used(vram_zone_id::KV);
        const size_t run_before = cache->zone_used(vram_zone_id::RUNTIME);
        alloc_handle handle;
        const bool   ok = alloc_compute(device, queue, big, /*kv_first=*/true, &handle);
        check(ok && handle.ptr != nullptr, "the flagged request was served");
        check(handle.vram_zone == vram_zone_id::KV, "the handle reports the KV zone");
        check(cache->zone_used(vram_zone_id::KV) >= kv_before + big, "KV zone_used grew by at least the request");
        check(cache->zone_used(vram_zone_id::RUNTIME) == run_before, "RUNTIME zone_used did not move");
        planned_hold_spill_totals totals;
        ggml_sycl::unified_cache_get_recent_planned_hold_spills(device, owner, &totals);
        check(totals.zone_full_count == 1 && totals.zone_full_bytes >= big,
              "counted as one zone-full KV placement of at least the request");
        check(totals.arena_count == 0 && totals.raw_count == 0, "not counted as a hold spill of either kind");

        unified_free(handle);
        check(cache->zone_used(vram_zone_id::KV) == kv_before, "freeing returned KV zone_used to where it started");
        check(cache->zone_used(vram_zone_id::RUNTIME) == run_before, "freeing left RUNTIME zone_used untouched");
    }

    printf("zone-full request, not flagged (control):\n");
    {
        const size_t kv_before = cache->zone_used(vram_zone_id::KV);
        alloc_handle handle;
        const bool   ok = alloc_compute(device, queue, big, /*kv_first=*/false, &handle);
        check(!ok || handle.vram_zone != vram_zone_id::KV, "an unflagged request is never placed in the KV zone");
        check(cache->zone_used(vram_zone_id::KV) == kv_before, "KV zone_used did not move");
        if (ok) {
            unified_free(handle);
        }
    }

    printf("held-back request, flagged:\n");
    ggml_sycl::unified_cache_begin_planned_hold_epoch(device, owner, 512);
    // A hold as large as the whole zone keeps every spill-capable request out of it, however small.
    ggml_sycl::unified_cache_set_planned_scratch_hold(device, runtime_cap, owner);
    {
        const size_t kv_before = cache->zone_used(vram_zone_id::KV);
        alloc_handle handle;
        const bool   ok = alloc_compute(device, queue, small, /*kv_first=*/true, &handle);
        check(ok && handle.ptr != nullptr, "the flagged request was served");
        check(handle.vram_zone == vram_zone_id::KV, "the held-back request lands in the KV zone");
        check(cache->zone_used(vram_zone_id::KV) >= kv_before + small, "KV zone_used grew by at least the request");
        planned_hold_spill_totals totals;
        ggml_sycl::unified_cache_get_recent_planned_hold_spills(device, owner, &totals);
        check(totals.arena_count == 1 && totals.arena_bytes >= small,
              "counted as one in-arena hold spill of at least the request");
        check(totals.raw_count == 0 && totals.zone_full_count == 0,
              "not counted as a raw spill or a zone-full placement");
        unified_free(handle);
        check(cache->zone_used(vram_zone_id::KV) == kv_before, "freeing returned KV zone_used to where it started");
    }

    printf("held-back request, not flagged (control):\n");
    {
        const size_t kv_before = cache->zone_used(vram_zone_id::KV);
        alloc_handle handle;
        const bool   ok = alloc_compute(device, queue, small, /*kv_first=*/false, &handle);
        check(!ok || handle.vram_zone != vram_zone_id::KV, "an unflagged held-back request is never placed in the KV zone");
        check(cache->zone_used(vram_zone_id::KV) == kv_before, "KV zone_used did not move");
        if (ok) {
            unified_free(handle);
        }
    }

    ggml_sycl::unified_cache_release_planned_scratch_hold(device, owner);
    ggml_backend_free(backend);

    printf("%s: %d failure(s)\n", g_failures == 0 ? "OK" : "FAILED", g_failures);
    return g_failures == 0 ? 0 : 1;
}

#endif  // GGML_USE_SYCL
