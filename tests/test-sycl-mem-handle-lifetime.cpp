#include "mem-handle.hpp"
#include "unified-cache.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <future>
#include <cstring>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#define CHECK(cond, msg)                             \
    do {                                             \
        if (!(cond)) {                               \
            std::fprintf(stderr, "FAIL: %s\n", msg); \
            return 1;                                \
        }                                            \
    } while (0)

static int test_direct_handle_debug_snapshot() {
    alignas(64) float     storage[16] = {};
    ggml_sycl::mem_handle h           = ggml_sycl::mem_handle::from_direct(storage, GGML_LAYOUT_AOS, false);
    h.set_debug_owner("unit-direct");
    ggml_sycl::mem_handle_debug_info info = h.debug_info();
    CHECK(info.valid, "direct handle must be valid");
    CHECK(info.kind == ggml_sycl::mem_handle_kind::DIRECT, "kind must be DIRECT");
    CHECK(std::strcmp(info.owner_tag, "unit-direct") == 0, "owner tag must round trip");
    CHECK(info.stable_identity_hash != 0, "stable identity hash must be non-zero");
    CHECK(!info.has_stable_identity, "external direct handles are pointer-identity only");
    CHECK(!info.has_ready_event, "new direct handle must not have ready event");
    return 0;
}

static int test_bounded_direct_views_report_and_enforce_extent() {
    alignas(64) unsigned char storage[64] = {};
    auto bounded = ggml_sycl::mem_handle::from_direct(
        storage, GGML_LAYOUT_AOS, false, ggml_sycl::mem_handle::HOST_DEVICE, sizeof(storage));
    const auto root = bounded.resolve();
    CHECK(root.ptr == storage && root.extent == sizeof(storage), "bounded root must report minted extent");

    const auto view = bounded.slice(16, 24);
    const auto resolved_view = view.resolve();
    CHECK(resolved_view.ptr == storage + 16 && resolved_view.extent == 24,
          "slice must report its view extent");
    CHECK(!bounded.slice(63, 2).valid(), "slice must reject offset+size beyond parent");
    CHECK(!bounded.slice(SIZE_MAX, 1).valid(), "slice must reject overflowing offset");

    auto unknown = ggml_sycl::mem_handle::from_direct(storage, GGML_LAYOUT_AOS, false);
    CHECK(unknown.resolve().extent == 0, "unknown direct extent must remain unknown");
    CHECK(!unknown.slice(0, 1).valid(), "unknown extent must not mint slicing authority");
    return 0;
}

static int test_copy_move_preserve_stable_identity_and_owner() {
    alignas(64) float     storage[16] = {};
    ggml_sycl::mem_handle a           = ggml_sycl::mem_handle::from_direct(storage, GGML_LAYOUT_AOS, false);
    a.set_debug_owner("copy-move-source");
    const size_t hash_a = a.stable_identity_hash();

    ggml_sycl::mem_handle b = a;
    CHECK(b.stable_identity_hash() == hash_a, "copy must preserve stable identity");
    CHECK(b.stable_identity_equal(a), "copy must compare stable-equal");
    CHECK(std::strcmp(b.debug_info().owner_tag, "copy-move-source") == 0, "copy must preserve owner tag");

    ggml_sycl::mem_handle c = std::move(b);
    CHECK(c.stable_identity_hash() == hash_a, "move must preserve stable identity");
    CHECK(c.debug_info().valid, "moved-to handle must be valid");
    CHECK(std::strcmp(c.debug_info().owner_tag, "copy-move-source") == 0, "move must transfer owner tag");
    return 0;
}

static int test_arena_authority_invalidation_and_chunk_bounds() {
    alignas(64) unsigned char a[64] = {}, b[64] = {};
    auto authority = std::make_shared<ggml_sycl::arena_authority>();
    authority->generation = 11;
    authority->zone_id = static_cast<int>(ggml_sycl::vram_zone_id::RUNTIME);
    authority->chunks = { { a, sizeof(a) }, { b, sizeof(b) } };
    authority->allowed_ranges = { { 0, sizeof(a) + sizeof(b) } };
    CHECK(authority->register_allocation(authority->zone_id, 101, 48, 16),
          "first fake allocator record must register");
    CHECK(authority->register_allocation(authority->zone_id, 102, 64, 8),
          "second fake allocator record must register");
    auto first = ggml_sycl::mem_handle::from_arena_zone(
        static_cast<int>(ggml_sycl::vram_zone_id::RUNTIME), 48, 16, 0, 11, 101, 16, authority);
    auto second = ggml_sycl::mem_handle::from_arena_zone(
        static_cast<int>(ggml_sycl::vram_zone_id::RUNTIME), 64, 8, 0, 11, 102, 8, authority);
    CHECK(first.resolve().ptr == a + 48 && first.resolve().extent == 16, "first exact extent");
    CHECK(second.resolve().ptr == b && second.resolve().extent == 8, "second physical chunk boundary");
    CHECK(authority->resolve_allocation(authority->zone_id, 11, 101, 60, 8) == nullptr,
          "non-exact sibling/cross-chunk tuple rejected");
    std::atomic<bool> stop{ false }, stale{ false };
    std::thread resolver([&] { while (!stop.load()) (void) first.resolve(); if (first.resolve()) stale = true; });
    authority->close_and_invalidate(12);
    CHECK(!authority->acquire_allocation(authority->zone_id, 11, 101, 48, 16),
          "closed incarnation admitted a new exact lease");
    stop = true;
    resolver.join();
    CHECK(!stale.load() && !first.resolve() && !second.resolve(), "concurrent invalidation is terminal");

    auto drain = std::async(std::launch::async, [&] { authority->wait_for_terminal_leases(); });
    CHECK(drain.wait_for(std::chrono::milliseconds(10)) == std::future_status::timeout,
          "settle did not wait for retained terminal leases");
    first = {};
    second = {};
    CHECK(drain.wait_for(std::chrono::seconds(1)) == std::future_status::ready,
          "terminal lease release did not unblock settle");
    CHECK(authority->unregister_allocation(authority->zone_id, 48), "first fake record must unregister");
    CHECK(authority->unregister_allocation(authority->zone_id, 64), "second fake record must unregister");
    return 0;
}

static std::shared_ptr<ggml_sycl::arena_authority> make_authority(
    int zone, uint64_t generation, void * ptr, size_t size, size_t logical_start = 0) {
    auto authority = std::make_shared<ggml_sycl::arena_authority>();
    authority->zone_id = zone;
    authority->generation = generation;
    authority->chunks = { { ptr, size } };
    authority->allowed_ranges = { { logical_start, size } };
    authority->lease_counts.resize(1, 0);
    return authority;
}

static uint32_t allocation_lease_count(const std::shared_ptr<ggml_sycl::arena_authority> & authority,
                                       uint64_t allocation_id) {
    std::lock_guard<std::mutex> guard(authority->mutex);
    for (const auto & record : authority->allocations) {
        if (record.allocation_id == allocation_id) return record.lease_count;
    }
    return 0;
}

// llama.cpp-09ts: the tiered KV buffer is one arena sub-allocation PER LAYER,
// and its "base" pointer is just the first layer's.  This pins the two facts
// tiered_kv_buffer_clear() has to respect: a root minted over layer 0 resolves
// layer 0 only, and no view of it can be widened to cover the siblings.  A
// whole-buffer fill through that root is therefore never expressible, which is
// why the clear fills through each layer's own owning handle.  Uses the KV
// zone id specifically — it is the zone with no dedicated mem_handle_kind, so
// this also covers that KV resolves through zone_id_ rather than kind_.
static int test_kv_zone_root_does_not_cover_sibling_layers() {
    constexpr size_t          layer_bytes                     = 16;
    constexpr size_t          n_layers                        = 4;
    alignas(64) unsigned char storage[layer_bytes * n_layers] = {};

    const int kv_zone   = static_cast<int>(ggml_sycl::vram_zone_id::KV);
    auto      authority = make_authority(kv_zone, 31, storage, sizeof(storage));
    for (size_t l = 0; l < n_layers; ++l) {
        CHECK(authority->register_allocation(kv_zone, 300 + l, l * layer_bytes, layer_bytes),
              "per-layer KV fake allocator record must register");
    }

    std::vector<ggml_sycl::mem_handle> layers;
    for (size_t l = 0; l < n_layers; ++l) {
        layers.push_back(ggml_sycl::mem_handle::from_arena_zone(kv_zone, l * layer_bytes, layer_bytes, 0, 31, 300 + l,
                                                                layer_bytes, authority));
        const auto resolved = layers.back().resolve();
        CHECK(resolved.ptr == storage + l * layer_bytes && resolved.extent == layer_bytes,
              "each KV layer handle must resolve its own allocation exactly");
    }

    // The buffer "base" is layer 0's pointer, and its authority stops there.
    CHECK(layers.front().resolve().extent == layer_bytes,
          "KV root must not inherit the extent of the whole layer span");
    CHECK(!layers.front().slice(0, sizeof(storage)).valid(), "KV root must not be wideable to a whole-buffer view");
    CHECK(authority->resolve_allocation(kv_zone, 31, 300, 0, sizeof(storage)) == nullptr,
          "whole-span tuple must not resolve against a single layer's record");

    layers.clear();
    for (size_t l = 0; l < n_layers; ++l) {
        CHECK(authority->unregister_allocation(kv_zone, l * layer_bytes), "per-layer KV fake record must unregister");
    }
    return 0;
}

static int test_arena_allocation_vs_settle_linearization() {
    alignas(64) unsigned char storage[64] = {};
    auto authority = make_authority(static_cast<int>(ggml_sycl::vram_zone_id::RUNTIME), 3,
                                    storage, sizeof(storage));

    // Allocation/mint wins: close observes its lease and settle cannot finish
    // until the alias is released.
    constexpr uint64_t allocation_id = 201;
    const int zone = static_cast<int>(ggml_sycl::vram_zone_id::RUNTIME);
    CHECK(authority->register_allocation(zone, allocation_id, 0, 16),
          "linearization fake allocator record must register");
    auto admitted = authority->acquire_allocation(zone, 3, allocation_id, 0, 16);
    CHECK(admitted, "pre-close exact allocation must be admitted");
    CHECK(allocation_lease_count(authority, allocation_id) == 1 && authority->terminal_lease_count() == 1,
          "exact admission must increment allocation and chunk counts once");
    authority->close_and_invalidate(4);
    auto drain = std::async(std::launch::async, [&] { authority->wait_for_terminal_leases(); });
    CHECK(drain.wait_for(std::chrono::milliseconds(10)) == std::future_status::timeout,
          "settle passed an allocation linearized before close");
    admitted.lease.reset();
    CHECK(drain.wait_for(std::chrono::seconds(1)) == std::future_status::ready,
          "settle did not complete after terminal allocation lease");
    CHECK(allocation_lease_count(authority, allocation_id) == 0 && authority->terminal_lease_count() == 0,
          "exact lease release must decrement allocation and chunk counts");

    // Settle wins: no allocation or handle mint is admitted after close.
    CHECK(!authority->acquire_allocation(zone, 3, allocation_id, 0, 16),
          "exact allocation admitted after close");
    CHECK(authority->unregister_allocation(zone, 0), "linearization fake record must unregister");
    return 0;
}

static int test_unrelated_zone_incarnations_are_independent() {
    alignas(64) unsigned char runtime_bytes[64] = {}, scratch_bytes[64] = {};
    auto runtime = make_authority(static_cast<int>(ggml_sycl::vram_zone_id::RUNTIME), 8,
                                  runtime_bytes, sizeof(runtime_bytes));
    auto scratch = make_authority(static_cast<int>(ggml_sycl::vram_zone_id::SCRATCH), 13,
                                  scratch_bytes, sizeof(scratch_bytes));
    const int runtime_zone = static_cast<int>(ggml_sycl::vram_zone_id::RUNTIME);
    const int scratch_zone = static_cast<int>(ggml_sycl::vram_zone_id::SCRATCH);
    CHECK(runtime->register_allocation(runtime_zone, 1, 0, 8), "runtime fake record must register");
    CHECK(scratch->register_allocation(scratch_zone, 2, 0, 8), "scratch root fake record must register");
    CHECK(scratch->register_allocation(scratch_zone, 4, 8, 8), "scratch sibling fake record must register");
    auto runtime_handle = ggml_sycl::mem_handle::from_arena_zone(
        static_cast<int>(ggml_sycl::vram_zone_id::RUNTIME), 0, 8, 0, 8, 1, 8, runtime);
    auto scratch_handle = ggml_sycl::mem_handle::from_arena_zone(
        static_cast<int>(ggml_sycl::vram_zone_id::SCRATCH), 0, 8, 0, 13, 2, 8, scratch);
    runtime->close_and_invalidate(9);
    CHECK(!runtime_handle.resolve(), "settled zone handle remained valid");
    CHECK(scratch_handle.resolve().ptr == scratch_bytes,
          "settling runtime invalidated unrelated scratch incarnation");
    auto sibling = scratch->acquire_allocation(scratch_zone, 13, 4, 8, 8);
    CHECK(sibling, "unrelated zone exact admission was blocked");
    sibling.lease.reset();
    runtime_handle = {};
    scratch_handle = {};
    CHECK(runtime->unregister_allocation(runtime_zone, 0), "runtime fake record must unregister");
    CHECK(scratch->unregister_allocation(scratch_zone, 0), "scratch root fake record must unregister");
    CHECK(scratch->unregister_allocation(scratch_zone, 8), "scratch sibling fake record must unregister");
    return 0;
}

static int test_arena_alias_count_and_publish_race() {
    alignas(64) unsigned char storage[64] = {};
    auto old_authority = make_authority(static_cast<int>(ggml_sycl::vram_zone_id::SCRATCH), 21,
                                        storage, sizeof(storage));
    const int scratch_zone = static_cast<int>(ggml_sycl::vram_zone_id::SCRATCH);
    CHECK(old_authority->register_allocation(scratch_zone, 3, 0, 16),
          "old fake allocator record must register");
    auto root = ggml_sycl::mem_handle::from_arena_zone(
        static_cast<int>(ggml_sycl::vram_zone_id::SCRATCH), 0, 16, 0, 21, 3, 16, old_authority);
    auto alias_a = root;
    auto alias_b = alias_a.slice(4, 4);
    CHECK(old_authority->terminal_lease_count() == 1 && allocation_lease_count(old_authority, 3) == 1,
          "handle aliases must share one allocation/chunk lease, not double-count");

    std::mutex mirror_mutex;
    std::shared_ptr<ggml_sycl::arena_authority> mirror = old_authority;
    std::atomic<bool> stop{ false }, close_complete{ false }, stale_admission{ false };
    std::thread minter([&] {
        while (!stop.load(std::memory_order_acquire)) {
            std::shared_ptr<ggml_sycl::arena_authority> snapshot;
            { std::lock_guard<std::mutex> lock(mirror_mutex); snapshot = mirror; }
            const bool observed_closed = close_complete.load(std::memory_order_acquire);
            if (observed_closed && snapshot == old_authority &&
                snapshot->acquire_allocation(scratch_zone, 21, 3, 0, 16)) {
                stale_admission.store(true, std::memory_order_release);
            }
        }
    });
    old_authority->close_and_invalidate(22);
    close_complete.store(true, std::memory_order_release);
    auto fresh = make_authority(static_cast<int>(ggml_sycl::vram_zone_id::SCRATCH), 22,
                                storage, sizeof(storage));
    CHECK(fresh->register_allocation(scratch_zone, 4, 0, 1),
          "fresh fake allocator record must register before publication");
    { std::lock_guard<std::mutex> lock(mirror_mutex); mirror = fresh; }
    stop.store(true, std::memory_order_release);
    minter.join();
    CHECK(!old_authority->acquire_allocation(scratch_zone, 21, 3, 0, 16),
          "stale authority published an exact admission");
    auto fresh_admission = fresh->acquire_allocation(scratch_zone, 22, 4, 0, 1);
    CHECK(fresh_admission, "fresh incarnation exact record was not publishable");
    CHECK(!stale_admission.load(std::memory_order_acquire),
          "concurrent publication admitted through the closed incarnation");
    fresh_admission.lease.reset();
    alias_b = {};
    alias_a = {};
    root = {};
    CHECK(old_authority->terminal_lease_count() == 0 && allocation_lease_count(old_authority, 3) == 0,
          "final alias release must clear allocation and chunk counts");
    CHECK(old_authority->unregister_allocation(scratch_zone, 0), "old fake record must unregister");
    CHECK(fresh->unregister_allocation(scratch_zone, 0), "fresh fake record must unregister");
    return 0;
}

static int test_arena_debug_identity_includes_generation() {
    // e4c665c88 made exact allocation identity mandatory: from_arena_zone
    // refuses a mint whose allocation_id or allocation_extent is zero, so the
    // defaulted five-argument form these calls used returned an empty handle
    // and every assertion below read zeros.
    const int             arena_zone = static_cast<int>(ggml_sycl::vram_zone_id::RUNTIME);
    ggml_sycl::mem_handle a          = ggml_sycl::mem_handle::from_arena_zone(arena_zone, 4096, 1024,
                                                                              ggml_sycl::mem_handle::HOST_DEVICE, 7, 501, 1024);
    ggml_sycl::mem_handle b          = ggml_sycl::mem_handle::from_arena_zone(arena_zone, 4096, 1024,
                                                                              ggml_sycl::mem_handle::HOST_DEVICE, 8, 501, 1024);
    a.set_debug_owner("arena-a");
    CHECK(a.debug_info().generation == 7, "generation 7 must be visible");
    CHECK(b.debug_info().generation == 8, "generation 8 must be visible");
    CHECK(a.debug_info().zone_id == arena_zone, "zone must be visible");
    CHECK(a.debug_info().size == 1024, "size must be visible");
    CHECK(a.debug_info().offset == 0, "arena canonical root offset must be allocation-relative");
    CHECK(a.has_stable_owner_identity(), "arena handles must have stable owner identity");
    CHECK(!a.stable_identity_equal(b), "different arena generations must not compare stable-equal");
    CHECK(std::strcmp(a.debug_info().owner_tag, "arena-a") == 0, "arena owner tag must round trip");
    return 0;
}

int main() {
    if (int rc = test_direct_handle_debug_snapshot()) {
        return rc;
    }
    if (int rc = test_bounded_direct_views_report_and_enforce_extent()) {
        return rc;
    }
    if (int rc = test_copy_move_preserve_stable_identity_and_owner()) {
        return rc;
    }
    if (int rc = test_arena_debug_identity_includes_generation()) return rc;
    if (int rc = test_arena_authority_invalidation_and_chunk_bounds()) return rc;
    if (int rc = test_kv_zone_root_does_not_cover_sibling_layers()) {
        return rc;
    }
    if (int rc = test_arena_allocation_vs_settle_linearization()) return rc;
    if (int rc = test_unrelated_zone_incarnations_are_independent()) return rc;
    if (int rc = test_arena_alias_count_and_publish_race()) return rc;
    std::puts("PASS: mem_handle lifetime diagnostics");
    return 0;
}
