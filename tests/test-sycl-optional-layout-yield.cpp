// Gate for unified_cache::yield_optional_layouts() on a real device cache.
//
// A dense weight's optional layout copy (S1-PRELOAD's oneDNN WOQ second copy,
// unified_cache_entry::optional_layout) yields to runtime KV: the runtime-
// context transaction asks the cache to release copies, then re-reads the
// KV-shared zone's free space and fits the KV against it. So the claim this
// test checks is about the ZONE, not about the cache's bookkeeping: after a
// yield reports N bytes freed, zone_available() of the zone the copy lived in
// must have risen by N. A yield that retires entries and only queues their
// frees reports success while the zone reads the same, and the transaction then
// demotes KV exactly as if nothing had been released. That was observed on the
// B50 (Mistral Q4_0, GGML_SYCL_VRAM_BUDGET_PCT=60, -c 2048): "released 4 ...
// (192.3 MB)" followed by "22 layer(s) demoted ... 84.4 MB free for KV".
//
// The other two properties are the ownership class's limits:
//   - a copy someone other than the cache's own direct-stage mirror still
//     leases is not yieldable, and becomes yieldable when that lease drops;
//   - a primary (never marked optional) is never released.
//
// No model is loaded: the copies are staged directly into the device cache, as
// S1-PRELOAD does, so the run costs a few MB of device memory.
//
// Usage (GPU; ctest supplies the selector):
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-sycl-optional-layout-yield

#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml-sycl/ggml-sycl-test.hpp"
#include "ggml-sycl/unified-cache.hpp"
#include "ggml.h"
#include "sycl-selector-fallback.hpp"
#include "test-skip.h"

#include <cstdint>
#include <cstdio>
#include <sycl/sycl.hpp>
#include <vector>

#if !defined(GGML_USE_SYCL)
int main() {
    fprintf(stderr, "SKIP: GGML_USE_SYCL not enabled; this run proves NOTHING about optional layout yields.\n");
    return LLAMA_TEST_EXIT_SKIP;
}
#else

using ggml_sycl::unified_cache;
using ggml_sycl::vram_zone_id;

namespace {

int g_failures = 0;

void check(bool ok, const char * what) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) {
        g_failures++;
    }
}

constexpr size_t COPY_BYTES = 4u << 20;

// Stages `data` as a dense weight in `layout`, the way S1-PRELOAD stages a
// primary or its WOQ copy. `lease` (optional) receives a lease of its own, as
// S1-PRELOAD's out_handle does while it is still pinning.
bool stage(unified_cache *              cache,
           sycl::queue *                queue,
           const std::vector<uint8_t> & data,
           ggml_layout_mode             layout,
           void **                      ptr,
           ggml_sycl::mem_handle *      lease = nullptr) {
    const ggml_sycl_cache_id key = ggml_sycl::test_make_cache_id(data.data());
    auto                     result =
        cache->direct_stage_weight(key, data.data(), data.size(), data.size(), layout, nullptr, nullptr, queue, lease);
    if (!result.ok || !result.ptr) {
        return false;
    }
    result.event.wait();
    *ptr = result.ptr;
    return true;
}

// The zone whose free space the transaction re-reads is the one the copy
// occupies; in single-chunk mode WEIGHT delegates to KV's allocator, so this
// is the KV headroom as well.
size_t copy_zone_available(unified_cache * cache) {
    return cache->zone_available(vram_zone_id::WEIGHT);
}

void test_yield_returns_bytes_to_zone(unified_cache * cache, sycl::queue * queue) {
    printf("yield returns a copy's bytes to its zone:\n");
    std::vector<uint8_t>     data(COPY_BYTES, 0x3c);
    const ggml_sycl_cache_id key = ggml_sycl::test_make_cache_id(data.data());

    const size_t before = copy_zone_available(cache);
    void *       ptr    = nullptr;
    if (!stage(cache, queue, data, GGML_LAYOUT_ONEDNN_WOQ, &ptr)) {
        check(false, "staged a WOQ copy");
        return;
    }
    // Without this the zone observable below says nothing about the copy.
    const bool in_zone = cache->zone_owns(vram_zone_id::WEIGHT, ptr);
    check(in_zone, "the copy lives in the WEIGHT zone (the one KV admission re-reads)");
    const size_t staged = copy_zone_available(cache);
    printf("  zone_available(WEIGHT): before=%zu staged=%zu\n", before, staged);
    check(before >= staged + COPY_BYTES, "staging took the copy's bytes from the zone");

    const size_t optional_before = cache->optional_layout_bytes();
    check(cache->mark_optional_layout(key, GGML_LAYOUT_ONEDNN_WOQ), "mark_optional_layout found the copy");
    check(cache->optional_layout_bytes() == optional_before + COPY_BYTES,
          "optional_layout_bytes() counts it (only the cache's own mirror leases it)");

    const auto   result = cache->yield_optional_layouts(COPY_BYTES);
    const size_t after  = copy_zone_available(cache);
    printf("  yield: retired=%zu freed=%zu freed_bytes=%zu pending_bytes=%zu; zone_available after=%zu\n",
           result.retired, result.freed, result.freed_bytes, result.pending_bytes, after);
    check(result.retired == 1 && result.freed == 1 && result.freed_bytes == COPY_BYTES,
          "the yield reports the copy retired and freed");
    check(after >= staged + result.freed_bytes, "zone_available() rose by the bytes the yield reports freed");
    check(!cache->is_cached(key, GGML_LAYOUT_ONEDNN_WOQ), "the copy no longer resolves");
}

void test_leased_copy_is_not_yielded(unified_cache * cache, sycl::queue * queue) {
    printf("a leased copy is not yielded until its lease drops:\n");
    std::vector<uint8_t>     data(COPY_BYTES, 0x5a);
    const ggml_sycl_cache_id key = ggml_sycl::test_make_cache_id(data.data());

    ggml_sycl::mem_handle lease;
    void *                ptr = nullptr;
    if (!stage(cache, queue, data, GGML_LAYOUT_ONEDNN_WOQ, &ptr, &lease) || !lease.valid()) {
        check(false, "staged a WOQ copy with a lease held");
        return;
    }
    check(cache->mark_optional_layout(key, GGML_LAYOUT_ONEDNN_WOQ), "mark_optional_layout found the copy");
    const size_t optional_leased = cache->optional_layout_bytes();

    const auto held = cache->yield_optional_layouts(COPY_BYTES);
    check(held.retired == 0 && held.freed_bytes == 0, "a yield passes over the leased copy");
    check(cache->is_cached(key, GGML_LAYOUT_ONEDNN_WOQ), "the leased copy still resolves");

    lease = {};
    check(cache->optional_layout_bytes() == optional_leased + COPY_BYTES,
          "once the lease drops, optional_layout_bytes() counts the copy");
    const auto released = cache->yield_optional_layouts(COPY_BYTES);
    check(released.freed == 1 && released.freed_bytes == COPY_BYTES, "and a yield then frees it");
}

void test_primary_is_never_yielded(unified_cache * cache, sycl::queue * queue) {
    printf("a primary is never yielded:\n");
    std::vector<uint8_t>     data(COPY_BYTES, 0x11);
    const ggml_sycl_cache_id key = ggml_sycl::test_make_cache_id(data.data());

    void * ptr = nullptr;
    if (!stage(cache, queue, data, GGML_LAYOUT_AOS, &ptr)) {
        check(false, "staged an AOS primary");
        return;
    }
    const auto result = cache->yield_optional_layouts(SIZE_MAX);
    check(result.retired == 0, "a yield asking for everything retires nothing unmarked");
    check(cache->is_cached(key, GGML_LAYOUT_AOS), "the primary still resolves");
}

}  // namespace

int main(int, char ** argv) {
    sycl_test_selector_fallback(argv, "level_zero:1");

    const int      device  = 0;  // in-process index after selector filtering
    ggml_backend_t backend = ggml_backend_sycl_init(device);
    if (backend == nullptr) {
        fprintf(stderr, "ggml_backend_sycl_init(%d) failed; cannot run.\n", device);
        return 1;
    }
    unified_cache * cache = ggml_sycl::get_unified_cache_for_device(device);
    if (cache == nullptr) {
        fprintf(stderr, "no unified cache for device %d; cannot run.\n", device);
        ggml_backend_free(backend);
        return 1;
    }
    sycl::queue * queue = &cache->get_queue();
    // The arena is normally reserved by a model load, which this test does not
    // do. Reserve a small one the way the backend's own staging tests do
    // (test_ensure_weight_zone in ggml-sycl.cpp).
    if (!cache->arena_active()) {
        constexpr size_t arena_bytes = 64u << 20;
        (void) cache->arena_reserve(*queue, arena_bytes, arena_bytes, arena_bytes, 0, 0, 0, 0);
    }
    printf("setup: arena_active=%d zone_capacity(WEIGHT)=%zu\n", cache->arena_active() ? 1 : 0,
           cache->zone_capacity(vram_zone_id::WEIGHT));
    // Without an arena there is no zone to observe, and every zone check below
    // would pass or fail for reasons unrelated to the yield.
    check(cache->arena_active() && cache->zone_capacity(vram_zone_id::WEIGHT) >= 4 * COPY_BYTES,
          "the device cache has an arena with room in the WEIGHT zone");

    if (g_failures == 0) {
        test_yield_returns_bytes_to_zone(cache, queue);
        test_leased_copy_is_not_yielded(cache, queue);
        test_primary_is_never_yielded(cache, queue);
    }

    ggml_backend_free(backend);

    printf("%s: %d failure(s)\n", g_failures == 0 ? "OK" : "FAILED", g_failures);
    return g_failures == 0 ? 0 : 1;
}

#endif  // GGML_USE_SYCL
