#include "mem-handle.hpp"
#include "unified-cache.hpp"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <memory>
#include <thread>
#include <utility>

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
    authority->chunks = { { a, sizeof(a) }, { b, sizeof(b) } };
    auto first = ggml_sycl::mem_handle::from_arena_zone(
        static_cast<int>(ggml_sycl::vram_zone_id::RUNTIME), 48, 16, 0, 11, 101, 16, authority);
    auto second = ggml_sycl::mem_handle::from_arena_zone(
        static_cast<int>(ggml_sycl::vram_zone_id::RUNTIME), 64, 8, 0, 11, 102, 8, authority);
    CHECK(first.resolve().ptr == a + 48 && first.resolve().extent == 16, "first exact extent");
    CHECK(second.resolve().ptr == b && second.resolve().extent == 8, "second physical chunk boundary");
    CHECK(authority->resolve_offset(11, 60, 8) == nullptr, "cross-chunk extent rejected");
    std::atomic<bool> stop{ false }, stale{ false };
    std::thread resolver([&] { while (!stop.load()) (void) first.resolve(); if (first.resolve()) stale = true; });
    authority->invalidate(12);
    stop = true;
    resolver.join();
    CHECK(!stale.load() && !first.resolve() && !second.resolve(), "concurrent invalidation is terminal");
    return 0;
}

static int test_arena_debug_identity_includes_generation() {
    ggml_sycl::mem_handle a =
        ggml_sycl::mem_handle::from_arena_zone(static_cast<int>(ggml_sycl::vram_zone_id::RUNTIME), 4096, 1024, 0, 7);
    ggml_sycl::mem_handle b =
        ggml_sycl::mem_handle::from_arena_zone(static_cast<int>(ggml_sycl::vram_zone_id::RUNTIME), 4096, 1024, 0, 8);
    a.set_debug_owner("arena-a");
    CHECK(a.debug_info().generation == 7, "generation 7 must be visible");
    CHECK(b.debug_info().generation == 8, "generation 8 must be visible");
    CHECK(a.debug_info().zone_id == static_cast<int>(ggml_sycl::vram_zone_id::RUNTIME), "zone must be visible");
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
    std::puts("PASS: mem_handle lifetime diagnostics");
    return 0;
}
