//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//
// Unit tests for the TLSF (Two-Level Segregated Fit) sub-allocator.
// Tests the external-metadata (offset-based) design — no writes to
// the managed region.  Operates on a CPU malloc region as a stand-in
// for VRAM; the allocator never touches that memory.
// Validates O(1) alloc/free semantics, coalescing, reset, and stress.

#include "../ggml/src/ggml-sycl/shared-zone-tags.hpp"
#include "../ggml/src/ggml-sycl/tlsf-allocator.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <map>
#include <vector>

// Every check goes through REQUIRE below, which aborts whether or not NDEBUG
// is defined: the project builds Release with -DNDEBUG, which would compile a
// bare assert() out and leave a test that cannot fail. The test is host-only
// (the allocator keeps all metadata on the host) and runs as the hostonly
// ctest test-tlsf-allocator.
//
// Must precede <cassert>, which binds assert at include time.
#undef NDEBUG
#include <cassert>

// Active test macro that works regardless of NDEBUG (assert() is compiled
// away under NDEBUG, silently turning the entire test suite into a no-op).
#define REQUIRE(cond)                                                                    \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            fprintf(stderr, "REQUIRE FAILED: %s at %s:%d\n", #cond, __FILE__, __LINE__); \
            abort();                                                                     \
        }                                                                                \
    } while (0)

using namespace ggml_sycl;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static constexpr size_t ONE_MB = 1024 * 1024;

// Allocate a 1 MB region with malloc (simulates VRAM — allocator never
// touches it).  Construct a tlsf_allocator over the SIZE only.
struct test_arena {
    void *           mem;  // Simulated VRAM (never accessed by TLSF)
    size_t           size;
    tlsf_allocator * alloc;

    explicit test_arena(size_t sz = ONE_MB) : size(sz) {
        // Align to 256 bytes so that offsets + arena_base would be aligned.
        // aligned_alloc requires the size to be a multiple of the alignment.
        mem = std::aligned_alloc(256, (sz + 255) & ~size_t(255));
        REQUIRE(mem != nullptr && "aligned_alloc failed");
        // Constructor takes SIZE only — no pointer to managed region.
        alloc = new tlsf_allocator(sz);
    }

    ~test_arena() {
        delete alloc;
        std::free(mem);
    }

    // Convert allocator offset to pointer (simulates what unified-cache does).
    void * offset_to_ptr(size_t offset) const { return static_cast<uint8_t *>(mem) + offset; }

    // Convert pointer back to offset.
    size_t ptr_to_offset(const void * ptr) const {
        auto p    = reinterpret_cast<uintptr_t>(ptr);
        auto base = reinterpret_cast<uintptr_t>(mem);
        REQUIRE(p >= base && p < base + size && "pointer out of arena");
        return static_cast<size_t>(p - base);
    }

    test_arena(const test_arena &)             = delete;
    test_arena & operator=(const test_arena &) = delete;
};

// ---------------------------------------------------------------------------
// 1. Basic allocation and free
// ---------------------------------------------------------------------------
static void test_basic_alloc_free() {
    test_arena arena;

    // Single allocation — returns offset (not pointer)
    size_t offset = arena.alloc->allocate(256);
    REQUIRE(offset != SIZE_MAX && "first alloc should succeed");
    REQUIRE(arena.alloc->used() > 0 && "used() should be non-zero after alloc");

    // Offset must be within the arena size
    REQUIRE(offset < arena.size && "offset within arena");

    // Free and verify
    size_t used_before = arena.alloc->used();
    (void) used_before;
    arena.alloc->free(offset);
    REQUIRE(arena.alloc->used() < used_before && "used() should decrease after free");

    std::cout << "test_basic_alloc_free: PASSED\n";
}

// ---------------------------------------------------------------------------
// 2. Alignment — all returned offsets must be 256-byte aligned
// ---------------------------------------------------------------------------
static void test_alignment() {
    test_arena arena;

    for (int i = 0; i < 50; ++i) {
        size_t sz     = 256 * (i + 1);
        size_t offset = arena.alloc->allocate(sz);
        REQUIRE(offset != SIZE_MAX && "allocation should succeed");
        REQUIRE((offset % 256) == 0 && "returned offset must be 256-byte aligned");
    }

    std::cout << "test_alignment: PASSED\n";
}

// ---------------------------------------------------------------------------
// 3. Coalescing — free two adjacent blocks, verify they merge
// ---------------------------------------------------------------------------
static void test_coalescing() {
    test_arena arena;

    // Allocate two blocks adjacent in offset order
    size_t o1 = arena.alloc->allocate(4096);
    size_t o2 = arena.alloc->allocate(4096);
    REQUIRE(o1 != SIZE_MAX && o2 != SIZE_MAX);

    size_t avail_before = arena.alloc->available();
    (void) avail_before;

    // Free o1 first (non-coalesce, o2 is allocated)
    arena.alloc->free(o1);
    size_t avail_after1 = arena.alloc->available();
    (void) avail_after1;

    // Free o2 — should coalesce with the freed o1
    arena.alloc->free(o2);
    size_t avail_after2 = arena.alloc->available();
    (void) avail_after2;

    // After coalescing, available should be strictly greater than after freeing
    // just o1, and should equal the original available (minus header overhead)
    REQUIRE(avail_after2 > avail_after1 && "coalescing should increase available");

    // All freed — used() should be zero and largest_free_block should
    // recover to the full arena (no header overhead in external-metadata design).
    REQUIRE(arena.alloc->used() == 0 && "all memory should be reclaimed after full free");
    REQUIRE(arena.alloc->largest_free_block() == arena.size &&
            "coalesced block should span entire arena (no header overhead)");

    std::cout << "test_coalescing: PASSED\n";
}

// ---------------------------------------------------------------------------
// 4. Reset — bulk deallocation returns all memory
// ---------------------------------------------------------------------------
static void test_reset() {
    test_arena arena;

    // Allocate a bunch
    size_t o1 = arena.alloc->allocate(1024 * 64);
    size_t o2 = arena.alloc->allocate(1024 * 32);
    size_t o3 = arena.alloc->allocate(1024 * 128);
    (void) o1;
    (void) o2;
    (void) o3;

    REQUIRE(arena.alloc->used() > 0 && "should have allocations");

    // Reset
    arena.alloc->reset();
    REQUIRE(arena.alloc->used() == 0 && "used() must be 0 after reset");

    // Should be able to allocate a large block after reset.
    // Note: allocate() uses mapping_search() which rounds up to the next SL
    // boundary, so requesting exactly largest_free_block() may search a class
    // above the block's actual class. Use 90% of the largest block instead.
    size_t largest = arena.alloc->largest_free_block();
    REQUIRE(largest > 0 && "largest_free_block should be positive after reset");

    size_t big = arena.alloc->allocate(largest * 9 / 10);
    REQUIRE(big != SIZE_MAX && "should allocate large block after reset");
    (void) big;

    std::cout << "test_reset: PASSED\n";
}

// ---------------------------------------------------------------------------
// 5. Allocation failure — request larger than available
// ---------------------------------------------------------------------------
static void test_alloc_failure() {
    // Tiny arena: smaller than MIN_BLOCK_SIZE (256) — no valid allocation possible.
    test_arena arena(128);

    // No valid allocation should be possible
    size_t offset = arena.alloc->allocate(256);
    REQUIRE(offset == SIZE_MAX && "allocation from tiny arena should fail");
    (void) offset;

    // A size whose granularity rounding would wrap to 0 is refused, not
    // handed back as a zero-size block at offset 0 that the next allocation
    // then aliases.
    test_arena big;
    REQUIRE(big.alloc->allocate(SIZE_MAX) == SIZE_MAX && "a wrapping size must be refused");
    REQUIRE(big.alloc->used() == 0 && big.alloc->check_invariants());
    REQUIRE(big.alloc->allocate(4096) == 0 && "the arena is still whole after the refusal");
    REQUIRE(big.alloc->check_invariants());

    std::cout << "test_alloc_failure: PASSED\n";
}

// ---------------------------------------------------------------------------
// 6. Exhaust and recycle — allocate all, free all, allocate again
// ---------------------------------------------------------------------------
static void test_exhaust_recycle() {
    test_arena arena;

    // Determine largest block — use 90% to account for TLSF rounding in
    // mapping_search(), which may search a size class above the block's class.
    size_t largest = arena.alloc->largest_free_block();
    REQUIRE(largest > 0);
    size_t alloc_size = largest * 9 / 10;

    // Allocate a large block
    size_t o1 = arena.alloc->allocate(alloc_size);
    REQUIRE(o1 != SIZE_MAX && "large block alloc should succeed");

    // Now available should be reduced
    size_t avail_after = arena.alloc->available();
    REQUIRE(avail_after < arena.size && "available should decrease after large alloc");
    (void) avail_after;

    // Try another large allocation — should fail
    size_t o2 = arena.alloc->allocate(alloc_size);
    // This may or may not succeed depending on remaining space, but it
    // exercises the failure path.

    // Free and verify recovery
    arena.alloc->free(o1);
    if (o2 != SIZE_MAX) {
        arena.alloc->free(o2);
    }

    // After freeing, we should be able to allocate again
    size_t o3 = arena.alloc->allocate(256);
    REQUIRE(o3 != SIZE_MAX && "should allocate after free");
    arena.alloc->free(o3);

    std::cout << "test_exhaust_recycle: PASSED\n";
}

// ---------------------------------------------------------------------------
// 7. Stress test — allocate/free many blocks randomly on 1 MB region
// ---------------------------------------------------------------------------
static void test_stress() {
    test_arena       arena;
    constexpr int    N_ROUNDS   = 200;
    constexpr int    MAX_SLOTS  = 64;
    constexpr size_t BLOCK_SIZE = 4096;

    size_t slots[MAX_SLOTS] = {};
    // Initialize all slots to SIZE_MAX (meaning "not allocated")
    for (int i = 0; i < MAX_SLOTS; ++i) {
        slots[i] = SIZE_MAX;
    }

    for (int round = 0; round < N_ROUNDS; ++round) {
        // Pick a random slot to either alloc or free
        int idx = round % MAX_SLOTS;

        if (slots[idx] == SIZE_MAX) {
            // Allocate
            slots[idx] = arena.alloc->allocate(BLOCK_SIZE);
            // May fail if arena is full — that's fine
        } else {
            // Free
            arena.alloc->free(slots[idx]);
            slots[idx] = SIZE_MAX;
        }
    }

    // Free remaining slots
    for (auto & slot : slots) {
        if (slot != SIZE_MAX) {
            arena.alloc->free(slot);
            slot = SIZE_MAX;
        }
    }

    // After freeing everything, verify all memory reclaimed
    REQUIRE(arena.alloc->used() == 0 && "all memory should be reclaimed after stress test");
    REQUIRE(arena.alloc->largest_free_block() == arena.size &&
            "coalesced block should span entire arena after stress test");

    std::cout << "test_stress: PASSED\n";
}

// ---------------------------------------------------------------------------
// 8. Free SIZE_MAX — should be a no-op (not crash)
// ---------------------------------------------------------------------------
static void test_free_invalid() {
    test_arena arena;
    // Free SIZE_MAX (invalid offset) — should be a no-op
    arena.alloc->free(SIZE_MAX);

    std::cout << "test_free_invalid: PASSED\n";
}

// ---------------------------------------------------------------------------
// 9. Allocate zero — should return SIZE_MAX
// ---------------------------------------------------------------------------
static void test_alloc_zero() {
    test_arena arena;
    size_t     offset = arena.alloc->allocate(0);
    REQUIRE(offset == SIZE_MAX && "alloc(0) should return SIZE_MAX");
    (void) offset;

    std::cout << "test_alloc_zero: PASSED\n";
}

// ---------------------------------------------------------------------------
// 10. Largest free block tracking
// ---------------------------------------------------------------------------
static void test_largest_free_block() {
    test_arena arena;

    // Initially, largest should be the entire arena (no header overhead)
    size_t initial_largest = arena.alloc->largest_free_block();
    REQUIRE(initial_largest > 0 && "initial largest_free_block should be positive");
    REQUIRE(initial_largest <= arena.size && "largest_free_block cannot exceed arena size");
    (void) initial_largest;

    // Allocate some memory
    size_t o1 = arena.alloc->allocate(4096);
    REQUIRE(o1 != SIZE_MAX);

    // After allocation, largest free block should be <= initial
    size_t after_largest = arena.alloc->largest_free_block();
    REQUIRE(after_largest <= initial_largest && "largest_free_block should not increase after alloc");
    (void) after_largest;

    arena.alloc->free(o1);

    std::cout << "test_largest_free_block: PASSED\n";
}

// ---------------------------------------------------------------------------
// 11. Multiple alloc/free cycles — verify no memory leak
// ---------------------------------------------------------------------------
static void test_no_leak() {
    test_arena arena;

    for (int i = 0; i < 100; ++i) {
        size_t offset = arena.alloc->allocate(256 * (i % 32 + 1));
        if (offset != SIZE_MAX) {
            arena.alloc->free(offset);
        }
    }

    // After all alloc/free cycles, all memory should be reclaimed
    REQUIRE(arena.alloc->used() == 0 && "no memory leak after alloc/free cycles");
    REQUIRE(arena.alloc->largest_free_block() == arena.size && "full arena recovered after alloc/free cycles");

    std::cout << "test_no_leak: PASSED\n";
}

// ---------------------------------------------------------------------------
// 12. Splitting — allocate small blocks and verify they don't overlap
// ---------------------------------------------------------------------------
static void test_splitting() {
    test_arena arena;

    // Allocate many small blocks
    constexpr int N = 16;
    size_t        offsets[N];
    for (int i = 0; i < N; ++i) {
        offsets[i] = arena.alloc->allocate(256);
        REQUIRE(offsets[i] != SIZE_MAX && "small alloc should succeed");
    }

    // Verify no two offsets overlap (each allocation should be at least
    // 256 bytes apart since they're 256-byte aligned)
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            size_t dist = offsets[i] > offsets[j] ? offsets[i] - offsets[j] : offsets[j] - offsets[i];
            REQUIRE(dist >= 256 && "allocations must not overlap");
            (void) dist;
        }
    }

    // Free all
    for (int i = 0; i < N; ++i) {
        arena.alloc->free(offsets[i]);
    }

    std::cout << "test_splitting: PASSED\n";
}

// ---------------------------------------------------------------------------
// Lifetime-segregated placement (llama.cpp-moua)
//
// Weights front-carve with allocate(). Context-lifetime tenants (KV regions)
// top-carve the gap from above with allocate_top()/allocate_below(). Yieldable
// tenants front-carve the gap from below with allocate_gap_front(), so they
// sit at the weight frontier. The tests below pin where each primitive places
// a block, and that releasing yieldable tenants highest-first grows the gap
// by exactly what frontier_walk() reported.
//
// These run single-threaded against a private allocator.  They do not test
// locking: tlsf_allocator has none, and every call here, the const reads
// included, must run under the owner's allocator lock in production
// (unified_cache::arena_allocator_group_mutex, llama.cpp-044k).  That is a
// property of the zone wrappers that call these, and belongs to their tests.
// ---------------------------------------------------------------------------
static constexpr uint8_t TAG_WEIGHT   = SHARED_ZONE_TAG_WEIGHT;
static constexpr uint8_t TAG_OPTIONAL = SHARED_ZONE_TAG_OPTIONAL;
static constexpr uint8_t TAG_CONTEXT  = SHARED_ZONE_TAG_CONTEXT;

// 13. allocate_top carves the upper end of the physically last block.
static void test_allocate_top() {
    test_arena arena;

    const size_t w = arena.alloc->allocate(4096, 256, TAG_WEIGHT);
    REQUIRE(w == 0 && "a front carve from an empty arena starts at offset 0");

    const size_t top = arena.alloc->allocate_top(8192, 256, TAG_CONTEXT);
    REQUIRE(top == arena.size - 8192 && "allocate_top must place the block at the arena's high end");
    REQUIRE(arena.alloc->used() == 4096 + 8192);
    REQUIRE(arena.alloc->tag_at(top) == TAG_CONTEXT);
    REQUIRE(arena.alloc->tag_at(w) == TAG_WEIGHT);

    // The gap between the two is one free block.
    REQUIRE(arena.alloc->gap_below(top) == arena.size - 4096 - 8192);
    // The last block is now allocated, so there is no gap at the top.
    REQUIRE(arena.alloc->gap_below(tlsf_allocator::no_anchor) == 0);
    REQUIRE(arena.alloc->allocate_top(256) == SIZE_MAX && "no free last block to carve");

    // A later front carve still goes to the bottom of the gap, not above the top block.
    const size_t w2 = arena.alloc->allocate(4096, 256, TAG_WEIGHT);
    REQUIRE(w2 == 4096);

    arena.alloc->free(top);
    REQUIRE(arena.alloc->tag_at(top) == 0 && "a freed block carries no tag");
    REQUIRE(arena.alloc->gap_below(tlsf_allocator::no_anchor) == arena.size - 8192 &&
            "freeing the top block coalesces it back into the gap");

    std::cout << "test_allocate_top: PASSED\n";
}

// 14. allocate_below carves the upper end of the free block just below an anchor.
static void test_allocate_below() {
    test_arena arena;

    const size_t top = arena.alloc->allocate_top(64 * 1024, 256, TAG_CONTEXT);
    REQUIRE(top == arena.size - 64 * 1024);

    const size_t b1 = arena.alloc->allocate_below(top, 16 * 1024, 256, TAG_CONTEXT);
    REQUIRE(b1 == top - 16 * 1024 && "allocate_below must place the block directly under its anchor");
    const size_t b2 = arena.alloc->allocate_below(b1, 4096, 256, TAG_CONTEXT);
    REQUIRE(b2 == b1 - 4096);

    // Too large for the gap: refused, and nothing moves.
    const size_t used_before = arena.alloc->used();
    REQUIRE(arena.alloc->allocate_below(b2, arena.size, 256, TAG_CONTEXT) == SIZE_MAX);
    REQUIRE(arena.alloc->used() == used_before);

    // An offset that is not an allocated block is not an anchor.
    REQUIRE(arena.alloc->allocate_below(b2 + 256, 256) == SIZE_MAX);
    REQUIRE(arena.alloc->gap_below(b2 + 256) == 0);

    // Fill the gap exactly: the anchor's predecessor is then allocated, so no gap.
    const size_t rest = arena.alloc->gap_below(b2);
    REQUIRE(rest == b2);
    const size_t fill = arena.alloc->allocate_below(b2, rest, 256, TAG_CONTEXT);
    REQUIRE(fill == 0);
    REQUIRE(arena.alloc->gap_below(b2) == 0);
    REQUIRE(arena.alloc->allocate_below(b2, 256) == SIZE_MAX && "no free block below the anchor");

    std::cout << "test_allocate_below: PASSED\n";
}

// 15. allocate_gap_front carves the lower end of the gap, i.e. at the weight frontier.
static void test_allocate_gap_front() {
    test_arena arena;

    const size_t w   = arena.alloc->allocate(100 * 1024, 256, TAG_WEIGHT);
    const size_t top = arena.alloc->allocate_top(200 * 1024, 256, TAG_CONTEXT);
    REQUIRE(w == 0);

    const size_t o1 = arena.alloc->allocate_gap_front(top, 10 * 1024, 256, TAG_OPTIONAL);
    REQUIRE(o1 == 100 * 1024 && "a gap-front carve must sit directly above the weights");
    const size_t o2 = arena.alloc->allocate_gap_front(top, 20 * 1024, 256, TAG_OPTIONAL);
    REQUIRE(o2 == o1 + 10 * 1024);
    REQUIRE(arena.alloc->gap_below(top) == arena.size - (100 + 200 + 10 + 20) * 1024);

    // With no anchor the gap is the last block.
    test_arena   bare;
    const size_t f = bare.alloc->allocate_gap_front(tlsf_allocator::no_anchor, 4096, 256, TAG_OPTIONAL);
    REQUIRE(f == 0);
    REQUIRE(bare.alloc->gap_below(tlsf_allocator::no_anchor) == bare.size - 4096);

    std::cout << "test_allocate_gap_front: PASSED\n";
}

// 16. Carved offsets keep MIN_BLOCK_SIZE alignment for odd sizes and small alignments.
static void test_carve_alignment() {
    test_arena arena;

    size_t anchor = tlsf_allocator::no_anchor;
    for (int i = 0; i < 20; ++i) {
        const size_t sz  = 1000 + 37 * i;  // never a multiple of 256
        const size_t off = anchor == tlsf_allocator::no_anchor ?
                               arena.alloc->allocate_top(sz, 64, TAG_CONTEXT) :
                               arena.alloc->allocate_below(anchor, sz, 64, TAG_CONTEXT);
        REQUIRE(off != SIZE_MAX);
        REQUIRE((off % 256) == 0 && "a top carve must return a 256-aligned offset");
        anchor = off;

        const size_t front = arena.alloc->allocate_gap_front(anchor, sz, 64, TAG_OPTIONAL);
        REQUIRE(front != SIZE_MAX);
        REQUIRE((front % 256) == 0 && "a gap-front carve must return a 256-aligned offset");
    }

    std::cout << "test_carve_alignment: PASSED\n";
}

// 17. A gap too small to split is taken whole, from its start.
static void test_carve_whole_gap() {
    test_arena arena;

    const size_t top = arena.alloc->allocate_top(arena.size - 4096 - 512, 256, TAG_CONTEXT);
    const size_t w   = arena.alloc->allocate(4096, 256, TAG_WEIGHT);
    REQUIRE(top == 4096 + 512);
    REQUIRE(w == 0);
    REQUIRE(arena.alloc->gap_below(top) == 512);

    // 300 B rounds to 512 = the whole gap: no remainder to leave free.
    const size_t whole = arena.alloc->allocate_below(top, 300, 256, TAG_CONTEXT);
    REQUIRE(whole == 4096);
    REQUIRE(arena.alloc->used() == arena.size && "the whole gap is accounted as used");
    REQUIRE(arena.alloc->gap_below(top) == 0);

    std::cout << "test_carve_whole_gap: PASSED\n";
}

// 18. frontier_walk reports the yield ladder, and a highest-first release grows
//     the gap by exactly the reported sizes.
static void test_frontier_walk_ladder() {
    test_arena arena;

    const size_t w  = arena.alloc->allocate(64 * 1024, 256, TAG_WEIGHT);
    const size_t kv = arena.alloc->allocate_top(256 * 1024, 256, TAG_CONTEXT);
    size_t       opt[3];
    const size_t opt_size[3] = { 8 * 1024, 16 * 1024, 32 * 1024 };
    for (int i = 0; i < 3; ++i) {
        opt[i] = arena.alloc->allocate_gap_front(kv, opt_size[i], 256, TAG_OPTIONAL);
        REQUIRE(opt[i] != SIZE_MAX);
    }
    REQUIRE(w == 0);

    const size_t gap = arena.alloc->gap_below(kv);
    auto         lad = arena.alloc->frontier_walk(kv, TAG_OPTIONAL);
    REQUIRE(lad.size() == 4 && "the gap plus three optional tenants, stopping at the weight");
    REQUIRE(lad[0].free && lad[0].size == gap);
    for (int i = 0; i < 3; ++i) {
        const auto & e = lad[1 + i];
        REQUIRE(!e.free && e.tag == TAG_OPTIONAL);
        REQUIRE(e.offset == opt[2 - i] && e.size == opt_size[2 - i] && "the ladder runs highest address first");
    }

    // Release highest-first: each release extends the gap by exactly its size.
    size_t expected = gap;
    for (int i = 2; i >= 0; --i) {
        arena.alloc->free(opt[i]);
        expected += opt_size[i];
        REQUIRE(arena.alloc->gap_below(kv) == expected);
    }
    REQUIRE(arena.alloc->gap_below(kv) == arena.size - 64 * 1024 - 256 * 1024);

    std::cout << "test_frontier_walk_ladder: PASSED\n";
}

// 19. An interior release shows up in the walk as a free rung, and the
//     cumulative walk still predicts the gap after the rest are released.
static void test_frontier_walk_interior_free() {
    test_arena arena;

    (void) arena.alloc->allocate(64 * 1024, 256, TAG_WEIGHT);
    const size_t kv = arena.alloc->allocate_top(64 * 1024, 256, TAG_CONTEXT);
    const size_t o1 = arena.alloc->allocate_gap_front(kv, 4096, 256, TAG_OPTIONAL);
    const size_t o2 = arena.alloc->allocate_gap_front(kv, 8192, 256, TAG_OPTIONAL);
    const size_t o3 = arena.alloc->allocate_gap_front(kv, 4096, 256, TAG_OPTIONAL);

    arena.alloc->free(o2);
    auto lad = arena.alloc->frontier_walk(kv, TAG_OPTIONAL);
    REQUIRE(lad.size() == 4);
    REQUIRE(lad[1].offset == o3 && !lad[1].free);
    REQUIRE(lad[2].offset == o2 && lad[2].free && lad[2].size == 8192);
    REQUIRE(lad[3].offset == o1 && !lad[3].free);

    size_t sum = 0;
    for (const auto & e : lad) {
        sum += e.size;
    }
    arena.alloc->free(o3);
    arena.alloc->free(o1);
    REQUIRE(arena.alloc->gap_below(kv) == sum && "the walk's total is the gap after the ladder is released");

    std::cout << "test_frontier_walk_interior_free: PASSED\n";
}

// 20. A weight placed above the optional tenants buries them: the walk stops at it.
static void test_frontier_walk_stops_at_weight() {
    test_arena arena;

    (void) arena.alloc->allocate(4096, 256, TAG_WEIGHT);
    const size_t kv = arena.alloc->allocate_top(64 * 1024, 256, TAG_CONTEXT);
    (void) arena.alloc->allocate_gap_front(kv, 4096, 256, TAG_OPTIONAL);
    const size_t w2 = arena.alloc->allocate_gap_front(kv, 4096, 256, TAG_WEIGHT);

    auto lad = arena.alloc->frontier_walk(kv, TAG_OPTIONAL);
    REQUIRE(lad.size() == 1 && "only the gap: the weight above the optional tenant ends the walk");
    REQUIRE(lad[0].free && lad[0].offset == w2 + 4096);

    // A context block directly on top of the walk start: the walk is empty.
    const size_t all = arena.alloc->gap_below(kv);
    (void) arena.alloc->allocate_below(kv, all, 256, TAG_CONTEXT);
    REQUIRE(arena.alloc->frontier_walk(kv, TAG_OPTIONAL).empty());

    std::cout << "test_frontier_walk_stops_at_weight: PASSED\n";
}

// 21. Randomized: every carve lands where an interval model predicts, gap_below
//     and frontier_walk agree with the model for every anchor, tags persist,
//     and check_invariants() holds after every step.  allocate() and free() of
//     allocate()'d blocks are in the mix, because the weight side (allocate)
//     and the context side (carves) share one allocator: a carve that leaves a
//     stale free-list entry shows up as allocate() handing out a live block,
//     or as a failed invariant, not in any physical-list read.
static void test_carve_against_interval_model() {
    const size_t   size = ONE_MB;
    tlsf_allocator alloc(size);

    struct live {
        size_t  size;
        uint8_t tag;
    };

    std::map<size_t, live> model;  // offset -> block, allocated only
    uint32_t               rng  = 12345u;
    auto                   next = [&]() {
        rng = rng * 1664525u + 1013904223u;
        return rng >> 8;
    };
    auto model_gap = [&](size_t anchor, size_t & lo) -> size_t {
        const size_t hi = anchor == tlsf_allocator::no_anchor ? size : anchor;
        lo              = 0;
        for (const auto & kv : model) {
            if (kv.first < hi) {
                lo = std::max(lo, kv.first + kv.second.size);
            }
        }
        return hi - lo;
    };
    auto model_walk = [&](size_t anchor, uint8_t pass_tag) {
        std::vector<tlsf_allocator::extent> walk;
        size_t                              hi = anchor == tlsf_allocator::no_anchor ? size : anchor;
        while (true) {
            auto         it       = model.lower_bound(hi);  // first block at or above hi
            const bool   has_prev = it != model.begin();
            const size_t lo       = has_prev ? std::prev(it)->first + std::prev(it)->second.size : 0;
            if (lo < hi) {
                walk.push_back({ lo, hi - lo, true, 0 });
            }
            if (!has_prev || std::prev(it)->second.tag != pass_tag) {
                return walk;
            }
            --it;
            walk.push_back({ it->first, it->second.size, false, it->second.tag });
            hi = it->first;
        }
    };
    auto pick_anchor = [&]() -> size_t {
        if (model.empty() || next() % 4 == 0) {
            return tlsf_allocator::no_anchor;
        }
        auto it = model.begin();
        std::advance(it, next() % model.size());
        return it->first;
    };

    int n_allocate = 0;
    for (int step = 0; step < 4000; ++step) {
        const uint32_t op = next() % 4;
        if (op == 2 && !model.empty()) {
            auto it = model.begin();
            std::advance(it, next() % model.size());
            alloc.free(it->first);
            model.erase(it);
        } else if (op == 3) {
            const size_t  req  = 256 * (1 + next() % 64);
            const uint8_t tag  = static_cast<uint8_t>(1 + next() % 3);
            const size_t  used = alloc.used();
            const size_t  got  = alloc.allocate(req, 256, tag);
            if (got == SIZE_MAX) {
                REQUIRE(alloc.used() == used && "a refused allocate() changes nothing");
            } else {
                const size_t bytes = alloc.used() - used;
                REQUIRE(got % 256 == 0 && got + bytes <= size && bytes >= req && bytes < req + 256);
                auto above = model.lower_bound(got);
                REQUIRE((above == model.end() || got + bytes <= above->first) &&
                        "allocate() returned a block overlapping a live block above it");
                REQUIRE((above == model.begin() || std::prev(above)->first + std::prev(above)->second.size <= got) &&
                        "allocate() returned a block overlapping a live block below it");
                model[got] = { bytes, tag };
                n_allocate++;
            }
        } else {
            const size_t  anchor = pick_anchor();
            const size_t  req    = 256 * (1 + next() % 64);
            const bool    front  = op == 1;
            const uint8_t tag    = static_cast<uint8_t>(1 + next() % 3);
            size_t        lo     = 0;
            const size_t  gap    = model_gap(anchor, lo);
            const size_t  hi     = lo + gap;
            const size_t  got    = front ?
                                       alloc.allocate_gap_front(anchor, req, 256, tag) :
                                       (anchor == tlsf_allocator::no_anchor ? alloc.allocate_top(req, 256, tag) :
                                                                              alloc.allocate_below(anchor, req, 256, tag));
            if (gap < req) {
                REQUIRE(got == SIZE_MAX && "a carve larger than the gap must be refused");
            } else if (gap < req + 256) {
                REQUIRE(got == lo && "a gap too small to split is taken whole");
                model[got] = { gap, tag };
            } else {
                REQUIRE(got == (front ? lo : hi - req) && "carve placed where the interval model predicts");
                model[got] = { req, tag };
            }
        }

        REQUIRE(alloc.check_invariants());
        size_t used = 0;
        for (const auto & kv : model) {
            used += kv.second.size;
            REQUIRE(alloc.tag_at(kv.first) == kv.second.tag);
            size_t lo = 0;
            REQUIRE(alloc.gap_below(kv.first) == model_gap(kv.first, lo));
        }
        REQUIRE(alloc.used() == used);
        size_t lo = 0;
        REQUIRE(alloc.gap_below(tlsf_allocator::no_anchor) == model_gap(tlsf_allocator::no_anchor, lo));

        const size_t  walk_anchor = pick_anchor();
        const uint8_t pass_tag    = static_cast<uint8_t>(1 + next() % 3);
        const auto    walk        = alloc.frontier_walk(walk_anchor, pass_tag);
        const auto    expect      = model_walk(walk_anchor, pass_tag);
        REQUIRE(walk.size() == expect.size() && "frontier_walk length differs from the model");
        for (size_t i = 0; i < walk.size(); ++i) {
            REQUIRE(walk[i].offset == expect[i].offset && walk[i].size == expect[i].size &&
                    walk[i].free == expect[i].free && walk[i].tag == expect[i].tag);
        }
    }
    REQUIRE(n_allocate > 100 && "the random mix exercised allocate()");

    std::cout << "test_carve_against_interval_model: PASSED\n";
}

// 21b. Placement after reset(): reset() must re-establish the physical last
//      block, or allocate_top() finds no gap.
static void test_carve_after_reset() {
    test_arena arena;

    (void) arena.alloc->allocate(4096, 256, TAG_WEIGHT);
    (void) arena.alloc->allocate_top(8192, 256, TAG_CONTEXT);
    arena.alloc->reset();
    REQUIRE(arena.alloc->check_invariants());

    const size_t top = arena.alloc->allocate_top(8192, 256, TAG_CONTEXT);
    REQUIRE(top == arena.size - 8192 && "allocate_top after reset() carves the top of the whole region");
    REQUIRE(arena.alloc->check_invariants());

    std::cout << "test_carve_after_reset: PASSED\n";
}

// 21c. Refusals change nothing: a size whose rounding would wrap, and a top
//      carve of a region whose end is off the MIN_BLOCK_SIZE grid.
static void test_carve_refusals() {
    test_arena arena;

    const size_t anchor = arena.alloc->allocate_top(8192, 256, TAG_CONTEXT);
    const size_t used   = arena.alloc->used();
    REQUIRE(arena.alloc->allocate_top(SIZE_MAX - 10) == SIZE_MAX);
    REQUIRE(arena.alloc->allocate_below(anchor, SIZE_MAX) == SIZE_MAX);
    REQUIRE(arena.alloc->allocate_gap_front(anchor, SIZE_MAX - 100) == SIZE_MAX);
    REQUIRE(arena.alloc->used() == used && arena.alloc->check_invariants());
    REQUIRE(arena.alloc->tag_at(anchor) == TAG_CONTEXT && "the anchor's own record is untouched");
    arena.alloc->free(anchor);
    REQUIRE(arena.alloc->used() == 0 && "the anchor is still freeable");

    // A region of ONE_MB + 100 bytes: its last block ends off the grid.
    tlsf_allocator odd(ONE_MB + 100);
    REQUIRE(odd.allocate_top(4096) == SIZE_MAX && "a top carve would start off the grid; refuse, do not abort");
    REQUIRE(odd.used() == 0 && odd.check_invariants());
    // Front carves and a whole-gap top take start on the grid, so they still work.
    REQUIRE(odd.allocate_gap_front(tlsf_allocator::no_anchor, 4096) == 0);
    REQUIRE(odd.check_invariants());

    std::cout << "test_carve_refusals: PASSED\n";
}

// 22. The llama.cpp-moua A2 shape: B50, GGML_SYCL_VRAM_BUDGET_PCT=60, Mistral
//     Q4_0, -c 32768. A 6872 MiB shared zone, 3917.9 MiB of primaries, and
//     one optional WOQ copy per eligible weight. When each copy is staged next
//     to its primary, releasing all of them frees 2954.1 MiB in holes too
//     small for one 128 MiB KV layer (the 23 EXT-ALLOC lines on hardware).
//     When the copies are staged after every primary at the gap front, the
//     same release gives a gap that holds all 23 layers, top-carved.
//     What the segregated arm proves is the two-pass ORDERING: during a load
//     there is one free block, so allocate() would place those copies at the
//     same addresses allocate_gap_front() does.  The difference between the
//     two calls shows only with several free blocks, which case 21 covers.
static void test_segregated_layout_a2_shape() {
    const size_t MiB    = ONE_MB;
    const size_t shared = 6872 * MiB;
    const size_t kv_lay = 128 * MiB;

    std::vector<size_t> primaries;
    std::vector<bool>   eligible;
    auto                add = [&](size_t bytes, bool woq) {
        primaries.push_back(bytes);
        eligible.push_back(woq);
    };
    add(32000ull * 4096 * 18 / 32, false);   // token_embd
    add(32000ull * 4096 * 210 / 256, true);  // output
    for (int i = 0; i < 65; ++i) {
        add(4096 * 4, false);                // norms
    }
    for (int l = 0; l < 32; ++l) {
        add(4096ull * 4096 * 18 / 32, true);  // q
        add(4096ull * 4096 * 18 / 32, true);  // o
        add(1024ull * 4096 * 18 / 32, true);  // k
        add(1024ull * 4096 * 18 / 32, true);  // v
    }
    for (int l = 0; l < 32 * 3; ++l) {
        add(14336ull * 4096 * 18 / 32, true);  // gate / up / down
    }
    size_t plan = 64 * MiB;                    // the planner's KV at load (n_ctx=512)
    for (size_t b : primaries) {
        plan += b;
    }

    for (int segregated = 0; segregated < 2; ++segregated) {
        tlsf_allocator      a(shared);
        std::vector<size_t> copies;
        size_t              staged     = 0;
        // S1-PRELOAD's guard: stage a copy only while the plan's remaining bytes stay free.
        auto                stage_copy = [&](size_t bytes) {
            if (a.largest_free_block() >= bytes && a.available() >= bytes + (plan - staged)) {
                const size_t off = segregated ?
                                                      a.allocate_gap_front(tlsf_allocator::no_anchor, bytes, 64, TAG_OPTIONAL) :
                                                      a.allocate(bytes, 64, TAG_OPTIONAL);
                if (off != SIZE_MAX) {
                    copies.push_back(off);
                }
            }
        };
        for (size_t i = 0; i < primaries.size(); ++i) {
            REQUIRE(a.allocate(primaries[i], 64, TAG_WEIGHT) != SIZE_MAX);
            staged += primaries[i];
            if (!segregated && eligible[i]) {
                stage_copy(primaries[i]);
            }
        }
        if (segregated) {
            for (size_t i = 0; i < primaries.size(); ++i) {
                if (eligible[i]) {
                    stage_copy(primaries[i]);
                }
            }
        }
        REQUIRE(copies.size() > 150);

        // Release every copy highest address first.
        std::sort(copies.begin(), copies.end());
        for (auto it = copies.rbegin(); it != copies.rend(); ++it) {
            a.free(*it);
        }
        const size_t free_bytes = a.available();
        const size_t largest    = a.largest_free_block();
        const size_t admitted   = free_bytes / kv_lay;
        REQUIRE(admitted == 23 && "the byte count admits 23 layers either way");

        size_t landed = 0;
        size_t anchor = tlsf_allocator::no_anchor;
        for (size_t l = 0; l < admitted; ++l) {
            const size_t off = anchor == tlsf_allocator::no_anchor ? a.allocate_top(kv_lay, 256, TAG_CONTEXT) :
                                                                     a.allocate_below(anchor, kv_lay, 256, TAG_CONTEXT);
            if (off == SIZE_MAX) {
                break;
            }
            anchor = off;
            landed++;
        }
        std::cout << "  a2 shape " << (segregated ? "segregated" : "interleaved") << ": free " << free_bytes / MiB
                  << " MiB, largest " << largest / MiB << " MiB, landed " << landed << "/" << admitted << "\n";
        if (segregated) {
            REQUIRE(landed == admitted && "segregated: every admitted KV layer lands in the zone");
        } else {
            REQUIRE(landed == 0 && "interleaved: the freed bytes are holes no KV layer fits (the A2 spill)");
        }
    }

    std::cout << "test_segregated_layout_a2_shape: PASSED\n";
}

// ---------------------------------------------------------------------------
// block_census() (llama.cpp-moua L4 1b): the whole-TLSF read primitive beside
// frontier_walk(), for the buried-optional / weight-hole census of
// kv_region_fit's geometry.  Read-only, group-mutex-only like frontier_walk().
// ---------------------------------------------------------------------------

// The census is the physical block list low to high: it tiles [0, size) with
// no gap or overlap, no two free blocks touch, free blocks carry tag 0, the
// allocated sum is used() and the free sum is available().
static void require_census_exact(const tlsf_allocator & a, size_t region_size) {
    const std::vector<tlsf_allocator::extent> census = a.block_census();
    REQUIRE(a.check_invariants());
    size_t at        = 0;
    size_t sum_used  = 0;
    size_t sum_free  = 0;
    bool   prev_free = false;
    for (const tlsf_allocator::extent & e : census) {
        REQUIRE(e.offset == at && "the census tiles the region low to high");
        REQUIRE(e.size > 0);
        REQUIRE(!(e.free && prev_free) && "no two free blocks are adjacent");
        REQUIRE(!e.free || e.tag == 0);
        if (e.free) {
            sum_free += e.size;
        } else {
            sum_used += e.size;
            REQUIRE(a.tag_at(e.offset) == e.tag && "an allocated block carries the tag the allocator holds");
        }
        prev_free = e.free;
        at        = e.offset + e.size;
    }
    REQUIRE(at == region_size && "the census ends at the region end");
    REQUIRE(sum_used == a.used());
    REQUIRE(sum_free == a.available());
}

static void test_block_census_tiles() {
    test_arena arena;

    {
        const auto c = arena.alloc->block_census();
        REQUIRE(c.size() == 1 && c[0].offset == 0 && c[0].size == arena.size && c[0].free && c[0].tag == 0 &&
                "a fresh allocator is one free block");
    }

    const size_t w  = arena.alloc->allocate(64 * 1024, 256, TAG_WEIGHT);
    const size_t kv = arena.alloc->allocate_top(128 * 1024, 256, TAG_CONTEXT);
    const size_t o1 = arena.alloc->allocate_gap_front(kv, 4096, 256, TAG_OPTIONAL);
    const size_t w2 = arena.alloc->allocate_gap_front(kv, 8192, 256, TAG_WEIGHT);
    const size_t o2 = arena.alloc->allocate_gap_front(kv, 16384, 256, TAG_OPTIONAL);
    REQUIRE(w == 0 && o1 != SIZE_MAX && w2 != SIZE_MAX && o2 != SIZE_MAX);
    require_census_exact(*arena.alloc, arena.size);

    // Interior releases: a buried optional (o1, under the weight w2) and a free
    // hole next to it, which the frontier walk cannot see past w2.
    arena.alloc->free(o1);
    require_census_exact(*arena.alloc, arena.size);
    {
        const auto c        = arena.alloc->block_census();
        bool       saw_hole = false;
        for (const auto & e : c) {
            if (e.offset == o1) {
                saw_hole = e.free && e.size == 4096;
            }
        }
        REQUIRE(saw_hole && "a released block below the frontier shows up as a free block");
        REQUIRE(c.front().offset == 0 && !c.front().free && c.front().tag == TAG_WEIGHT);
        REQUIRE(c.back().offset + c.back().size == arena.size && !c.back().free && c.back().tag == TAG_CONTEXT);
    }

    // The census contains the frontier walk as its high end, in reverse.
    {
        const auto c    = arena.alloc->block_census();
        const auto walk = arena.alloc->frontier_walk(kv, TAG_OPTIONAL);
        REQUIRE(!walk.empty());
        size_t pos = 0;
        while (pos < c.size() && c[pos].offset != walk.back().offset) {
            pos++;
        }
        REQUIRE(pos + walk.size() <= c.size());
        for (size_t i = 0; i < walk.size(); ++i) {
            const auto & e = c[pos + i];
            const auto & f = walk[walk.size() - 1 - i];
            REQUIRE(e.offset == f.offset && e.size == f.size && e.free == f.free && e.tag == f.tag);
        }
    }

    arena.alloc->free(w2);
    arena.alloc->free(kv);
    arena.alloc->free(o2);
    arena.alloc->free(w);
    require_census_exact(*arena.alloc, arena.size);
    {
        const auto c = arena.alloc->block_census();
        REQUIRE(c.size() == 1 && c[0].free && c[0].size == arena.size && "everything freed coalesces to one block");
    }

    std::cout << "test_block_census_tiles: PASSED\n";
}

// A random mix of every placement call against a shadow of what each call
// returned: the census's allocated blocks are exactly the shadow's (offset and
// tag; size is the used() delta, which includes whole-gap absorption), and its
// free blocks are exactly the stretches between them.
static void test_block_census_random_mix() {
    test_arena                arena;
    std::map<size_t, size_t>  sizes;  // allocated offset -> block size
    std::map<size_t, uint8_t> tags;
    std::vector<size_t>       live;
    uint32_t                  seed = 12345u;
    auto                      rnd  = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return seed >> 8;
    };
    size_t anchor = tlsf_allocator::no_anchor;

    for (int step = 0; step < 3000; ++step) {
        const uint32_t op = rnd() % 8;
        if (op < 5 || live.empty()) {
            const size_t  size = 256 + (rnd() % 24) * 256;
            const uint8_t tag  = (uint8_t) (1 + rnd() % 3);
            const size_t  used = arena.alloc->used();
            size_t        off  = SIZE_MAX;
            switch (op % 4) {
                case 0:
                    off = arena.alloc->allocate(size, 256, tag);
                    break;
                case 1:
                    off = arena.alloc->allocate_below(anchor, size, 256, tag);
                    if (off != SIZE_MAX) {
                        anchor = off;
                    }
                    break;
                case 2:
                    off = arena.alloc->allocate_gap_front(anchor, size, 256, tag);
                    break;
                default:
                    off = arena.alloc->allocate_top(size, 256, tag);
                    break;
            }
            if (off != SIZE_MAX) {
                sizes[off] = arena.alloc->used() - used;
                tags[off]  = tag;
                live.push_back(off);
            }
        } else {
            const size_t i   = rnd() % live.size();
            const size_t off = live[i];
            live[i]          = live.back();
            live.pop_back();
            if (off == anchor) {
                anchor = tlsf_allocator::no_anchor;
            }
            arena.alloc->free(off);
            sizes.erase(off);
            tags.erase(off);
        }

        if (step % 25 == 0) {
            require_census_exact(*arena.alloc, arena.size);
            const auto c = arena.alloc->block_census();
            size_t     n = 0;
            for (const auto & e : c) {
                if (e.free) {
                    continue;
                }
                n++;
                REQUIRE(sizes.count(e.offset) == 1 && sizes[e.offset] == e.size && tags[e.offset] == e.tag);
            }
            REQUIRE(n == sizes.size() && "the census holds every allocated block and no other");
        }
    }

    std::cout << "test_block_census_random_mix: PASSED\n";
}

// The census reads and changes nothing: an allocator interrogated after every
// operation places every later block exactly where its untouched twin does.
static void test_block_census_is_read_only() {
    test_arena watched;
    test_arena twin;
    uint32_t   seed = 777u;
    auto       rnd  = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return seed >> 8;
    };
    std::vector<size_t> live;
    for (int step = 0; step < 800; ++step) {
        const bool do_free = !live.empty() && rnd() % 3 == 0;
        if (do_free) {
            const size_t i   = rnd() % live.size();
            const size_t off = live[i];
            live[i]          = live.back();
            live.pop_back();
            watched.alloc->free(off);
            twin.alloc->free(off);
        } else {
            const size_t  size = 256 + (rnd() % 16) * 256;
            const uint8_t tag  = (uint8_t) (1 + rnd() % 3);
            const size_t  a    = watched.alloc->allocate(size, 256, tag);
            const size_t  b    = twin.alloc->allocate(size, 256, tag);
            REQUIRE(a == b && "an interrogated allocator places blocks where its twin does");
            if (a != SIZE_MAX) {
                live.push_back(a);
            }
        }
        (void) watched.alloc->block_census();
        REQUIRE(watched.alloc->used() == twin.alloc->used());
        REQUIRE(watched.alloc->largest_free_block() == twin.alloc->largest_free_block());
    }

    std::cout << "test_block_census_is_read_only: PASSED\n";
}

// A region too small to hold a block, and a reset region, are the empty and the
// single-free-block census.
static void test_block_census_degenerate() {
    tlsf_allocator tiny(100);
    REQUIRE(tiny.block_census().empty() && "a region under MIN_BLOCK_SIZE has no blocks");

    test_arena arena;
    (void) arena.alloc->allocate(4096, 256, TAG_WEIGHT);
    (void) arena.alloc->allocate_top(4096, 256, TAG_CONTEXT);
    arena.alloc->reset();
    const auto c = arena.alloc->block_census();
    REQUIRE(c.size() == 1 && c[0].free && c[0].offset == 0 && c[0].size == arena.size);

    std::cout << "test_block_census_degenerate: PASSED\n";
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    test_basic_alloc_free();
    test_alignment();
    test_coalescing();
    test_reset();
    test_alloc_failure();
    test_exhaust_recycle();
    test_stress();
    test_free_invalid();
    test_alloc_zero();
    test_largest_free_block();
    test_no_leak();
    test_splitting();
    test_allocate_top();
    test_allocate_below();
    test_allocate_gap_front();
    test_carve_alignment();
    test_carve_whole_gap();
    test_frontier_walk_ladder();
    test_frontier_walk_interior_free();
    test_frontier_walk_stops_at_weight();
    test_carve_against_interval_model();
    test_carve_after_reset();
    test_carve_refusals();
    test_segregated_layout_a2_shape();
    test_block_census_tiles();
    test_block_census_random_mix();
    test_block_census_is_read_only();
    test_block_census_degenerate();

    std::cout << "\nAll tlsf_allocator tests PASSED!\n";
    return 0;
}
