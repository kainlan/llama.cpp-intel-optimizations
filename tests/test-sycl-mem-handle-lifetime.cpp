#include "mem-handle.hpp"
#include "unified-cache.hpp"

#include <cstdio>
#include <cstring>
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
    CHECK(a.has_stable_owner_identity(), "arena handles must have stable owner identity");
    CHECK(!a.stable_identity_equal(b), "different arena generations must not compare stable-equal");
    CHECK(std::strcmp(a.debug_info().owner_tag, "arena-a") == 0, "arena owner tag must round trip");
    return 0;
}

int main() {
    if (int rc = test_direct_handle_debug_snapshot()) {
        return rc;
    }
    if (int rc = test_copy_move_preserve_stable_identity_and_owner()) {
        return rc;
    }
    if (int rc = test_arena_debug_identity_includes_generation()) {
        return rc;
    }
    std::puts("PASS: mem_handle lifetime diagnostics");
    return 0;
}
