// Gate for unified_cache::yield_optional_layouts() on a real device cache.
//
// A dense weight's optional layout copy (S1-PRELOAD's oneDNN WOQ second copy,
// unified_cache_entry::optional_layout) yields to runtime KV: the runtime-
// context transaction asks the cache to release copies for the device's KV
// layers, and holds the KV to the layers the zone can then place. So the claims
// this test checks are about the ZONE, not about the cache's bookkeeping:
//   - after a yield reports N bytes freed, zone_available() of the zone the
//     copy lived in has risen by N. A yield that retires entries and only
//     queues their frees reports success while the zone reads the same; that
//     was observed on the B50 (Mistral Q4_0, GGML_SYCL_VRAM_BUDGET_PCT=60,
//     -c 2048): "released 4 ... (192.3 MB)" then "22 layer(s) demoted ...
//     84.4 MB free for KV";
//   - a copy is released only when its room lets another KV layer land. Copies
//     staged between live weights free into holes no larger than one copy;
//     releasing them for a larger layer loses their layout and gives the KV
//     nothing. On the B50 at -c 32768 every copy was released, the fit counted
//     the 2954.1 MB by bytes, and all 23 device-resident 128 MB layers missed
//     the arena and went to raw device memory ([EXT-ALLOC]
//     cohort=kv-tier-layer-device-zone, llama.cpp-moua);
//   - adjacent copies free into one extent, so a layer larger than any one
//     copy lands once enough of them go.
// And the ownership class's limits: a copy someone other than the cache's own
// direct-stage mirror leases is not released until that lease drops, and a
// primary (never marked optional) is never released.
//
// No model is loaded: weights are staged directly into the device cache, as
// S1-PRELOAD does, so the run costs a few tens of MB of device memory. The
// zone's own free space is whatever the backend reserved (GBs), so each case
// asks for one more layer than the zone can place right now -- counted by the
// cache itself -- rather than for a size relative to that free space.
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
#include <list>
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

// Every staged weight's host bytes stay alive for the whole run: a cache key is
// derived from the data pointer, and a freed buffer's address coming back for
// a later weight would alias a key still in the cache.
std::list<std::vector<uint8_t>> g_weights;

struct staged {
    ggml_sycl_cache_id key{};
    void *             ptr = nullptr;
};

// Stages a weight in `layout`, the way S1-PRELOAD stages a primary or its WOQ
// copy; a WOQ copy is also marked optional, as S1-PRELOAD marks it. `lease`
// (optional) receives a lease of its own, as S1-PRELOAD's out_handle does while
// it is still pinning.
bool stage(unified_cache *         cache,
           sycl::queue *           queue,
           ggml_layout_mode        layout,
           staged *                out,
           ggml_sycl::mem_handle * lease = nullptr) {
    g_weights.emplace_back(COPY_BYTES, static_cast<uint8_t>(g_weights.size() + 1));
    const std::vector<uint8_t> & data = g_weights.back();
    out->key                          = ggml_sycl::test_make_cache_id(data.data());
    auto result = cache->direct_stage_weight(out->key, data.data(), data.size(), data.size(), layout, nullptr, nullptr,
                                             queue, lease);
    if (!result.ok || !result.ptr) {
        return false;
    }
    result.event.wait();
    out->ptr = result.ptr;
    return layout != GGML_LAYOUT_ONEDNN_WOQ || cache->mark_optional_layout(out->key, GGML_LAYOUT_ONEDNN_WOQ);
}

// One more KV layer of `layer_bytes` than the zone can place now, and how many
// it can: the request no copy's release can help unless its room is usable.
std::vector<size_t> one_more_layer(unified_cache * cache, size_t layer_bytes, size_t * placeable) {
    const size_t        probe = cache->zone_available(vram_zone_id::WEIGHT) / layer_bytes + 2;
    std::vector<size_t> layers(probe, layer_bytes);
    *placeable = cache->kv_layers_allocatable(layers);
    layers.resize(*placeable + 1);
    return layers;
}

void test_yield_returns_bytes_to_zone(unified_cache * cache, sycl::queue * queue) {
    printf("a yield returns a copy's bytes to its zone, and a layer lands in them:\n");
    staged copy;
    if (!stage(cache, queue, GGML_LAYOUT_ONEDNN_WOQ, &copy)) {
        check(false, "staged and marked a WOQ copy");
        return;
    }
    check(cache->zone_owns(vram_zone_id::WEIGHT, copy.ptr),
          "the copy lives in the WEIGHT zone (the one KV admission re-reads)");
    size_t       placeable = 0;
    const auto   layers    = one_more_layer(cache, COPY_BYTES, &placeable);
    const size_t staged    = cache->zone_available(vram_zone_id::WEIGHT);

    const auto   result = cache->yield_optional_layouts(layers);
    const size_t after  = cache->zone_available(vram_zone_id::WEIGHT);
    printf(
        "  placeable before=%zu; yield: retired=%zu freed=%zu freed_bytes=%zu pending_bytes=%zu kv_layers=%zu; "
        "zone_available %zu -> %zu\n",
        placeable, result.retired, result.freed, result.freed_bytes, result.pending_bytes, result.kv_layers, staged,
        after);
    check(result.retired == 1 && result.freed == 1 && result.freed_bytes == COPY_BYTES,
          "the yield reports the copy retired and freed");
    check(after >= staged + result.freed_bytes, "zone_available() rose by the bytes the yield reports freed");
    check(result.kv_layers == layers.size(), "and the layer that did not land before does now");
    check(!cache->is_cached(copy.key, GGML_LAYOUT_ONEDNN_WOQ), "the copy no longer resolves");
}

void test_leased_copy_is_not_yielded(unified_cache * cache, sycl::queue * queue) {
    printf("a leased copy is not yielded until its lease drops:\n");
    ggml_sycl::mem_handle lease;
    staged                copy;
    if (!stage(cache, queue, GGML_LAYOUT_ONEDNN_WOQ, &copy, &lease) || !lease.valid()) {
        check(false, "staged and marked a WOQ copy with a lease held");
        return;
    }
    size_t     placeable = 0;
    const auto layers    = one_more_layer(cache, COPY_BYTES, &placeable);

    const auto held = cache->yield_optional_layouts(layers);
    check(held.retired == 0 && held.freed_bytes == 0, "a yield passes over the leased copy");
    check(held.kv_layers == placeable, "and says the extra layer still does not land");
    check(cache->is_cached(copy.key, GGML_LAYOUT_ONEDNN_WOQ), "the leased copy still resolves");

    lease               = {};
    const auto released = cache->yield_optional_layouts(layers);
    check(released.freed == 1 && released.freed_bytes == COPY_BYTES, "once the lease drops, a yield frees it");
    check(released.kv_layers == layers.size(), "and the extra layer lands");
}

// S1-PRELOAD's order: each copy beside its own primary. Three 4 MB copies free
// 12 MB, more than an 8 MB layer, but as three holes between primaries.
void test_holes_between_primaries_are_not_yielded(unified_cache *       cache,
                                                  sycl::queue *         queue,
                                                  std::vector<staged> * primaries) {
    printf("copies between live primaries are not yielded for a layer larger than one of them:\n");
    std::vector<staged> copies;
    for (int k = 0; k < 3; ++k) {
        staged primary;
        staged copy;
        if (!stage(cache, queue, GGML_LAYOUT_AOS, &primary) || !stage(cache, queue, GGML_LAYOUT_ONEDNN_WOQ, &copy)) {
            check(false, "staged primary/copy pairs");
            return;
        }
        primaries->push_back(primary);
        copies.push_back(copy);
    }
    staged seal;
    if (!stage(cache, queue, GGML_LAYOUT_AOS, &seal)) {
        check(false, "staged the sealing primary");
        return;
    }
    primaries->push_back(seal);

    size_t       placeable = 0;
    const auto   layers    = one_more_layer(cache, 2 * COPY_BYTES, &placeable);
    const size_t optional  = cache->optional_layout_bytes();
    // The byte count the fit reads says the layer fits once the copies go;
    // without this the case would not show that a byte count is wrong here.
    check(optional >= 3 * COPY_BYTES && optional >= layers.back(),
          "by bytes, releasing the copies makes room for the extra layer");

    const auto result = cache->yield_optional_layouts(layers);
    printf("  placeable before=%zu optional=%zu; yield: retired=%zu kv_layers=%zu\n", placeable, optional,
           result.retired, result.kv_layers);
    check(result.retired == 0, "no copy is released: none of their holes holds the layer");
    check(result.kv_layers == placeable, "and the zone says so: the extra layer does not land");
    bool resident = true;
    for (const staged & copy : copies) {
        resident = resident && cache->is_cached(copy.key, GGML_LAYOUT_ONEDNN_WOQ);
    }
    check(resident, "every copy still resolves");
}

// Copies staged next to each other: two 4 MB copies free into one 8 MB extent.
void test_adjacent_copies_are_yielded_together(unified_cache *       cache,
                                               sycl::queue *         queue,
                                               std::vector<staged> * primaries) {
    printf("copies staged together are yielded together for a layer larger than one of them:\n");
    std::vector<staged> copies(2);
    staged              seal;
    if (!stage(cache, queue, GGML_LAYOUT_ONEDNN_WOQ, &copies[0]) ||
        !stage(cache, queue, GGML_LAYOUT_ONEDNN_WOQ, &copies[1]) || !stage(cache, queue, GGML_LAYOUT_AOS, &seal)) {
        check(false, "staged two adjacent copies and a sealing primary");
        return;
    }
    primaries->push_back(seal);
    const uintptr_t lo = reinterpret_cast<uintptr_t>(copies[0].ptr);
    const uintptr_t hi = reinterpret_cast<uintptr_t>(copies[1].ptr);
    // Without adjacency the case tests nothing about coalescing.
    check(hi == lo + COPY_BYTES, "the two copies are adjacent in the zone");

    size_t     placeable = 0;
    const auto layers    = one_more_layer(cache, 2 * COPY_BYTES, &placeable);
    const auto result    = cache->yield_optional_layouts(layers);
    printf("  placeable before=%zu; yield: retired=%zu freed_bytes=%zu kv_layers=%zu\n", placeable, result.retired,
           result.freed_bytes, result.kv_layers);
    check(result.retired == 2 && result.freed_bytes == 2 * COPY_BYTES, "both adjacent copies are released");
    check(result.kv_layers == layers.size(), "and the 8 MB layer lands in the extent they free");
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
    check(cache->arena_active() && cache->zone_capacity(vram_zone_id::WEIGHT) >= 16 * COPY_BYTES,
          "the device cache has an arena with room in the WEIGHT zone");

    if (g_failures == 0) {
        std::vector<staged> primaries;
        test_yield_returns_bytes_to_zone(cache, queue);
        test_leased_copy_is_not_yielded(cache, queue);
        test_holes_between_primaries_are_not_yielded(cache, queue, &primaries);
        test_adjacent_copies_are_yielded_together(cache, queue, &primaries);

        printf("a primary is never yielded:\n");
        bool resident = !primaries.empty();
        for (const staged & primary : primaries) {
            resident = resident && cache->is_cached(primary.key, GGML_LAYOUT_AOS);
        }
        check(resident, "every primary staged above still resolves");
    }

    ggml_backend_free(backend);

    printf("%s: %d failure(s)\n", g_failures == 0 ? "OK" : "FAILED", g_failures);
    return g_failures == 0 ? 0 : 1;
}

#endif  // GGML_USE_SYCL
