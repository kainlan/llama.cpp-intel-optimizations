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

#include "../ggml/src/ggml-sycl/expert-slot.hpp"
#include "../ggml/src/ggml-sycl/shared-zone-tags.hpp"
#include "../ggml/src/ggml-sycl/tlsf-allocator.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <map>
#include <random>
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

    // Should be able to allocate the whole largest block after reset.
    // allocate() places any size up to largest_free_block(), except where the
    // block ends off the grain (only a region whose size is not a multiple of
    // it) and the rounded request passes its end; this arena is on the grain.
    size_t largest = arena.alloc->largest_free_block();
    REQUIRE(largest > 0 && "largest_free_block should be positive after reset");

    size_t big = arena.alloc->allocate(largest);
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

    // Determine largest block and take 90% of it, so a second request of the
    // same size exercises the failure path.
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

    // Another large allocation fails: the largest free block is the 10% left.
    REQUIRE(arena.alloc->allocate(alloc_size) == SIZE_MAX && "a second 90% block does not fit");

    // Free and verify recovery
    arena.alloc->free(o1);

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
// allocate_at / allocate_excluding (llama.cpp-moua L4 pending-range primitive)
// ---------------------------------------------------------------------------

// allocate_at carves exactly the asked range out of the one free block that
// holds it, and the front and back remainders stay whole free blocks.
static void test_allocate_at_carves_exact() {
    test_arena       arena(16 * ONE_MB);
    tlsf_allocator & t = *arena.alloc;

    REQUIRE(t.allocate_at(4 * ONE_MB, ONE_MB, 3) == 4 * ONE_MB);
    REQUIRE(t.check_invariants());
    REQUIRE(t.block_size_at(4 * ONE_MB) == ONE_MB && t.tag_at(4 * ONE_MB) == 3);
    auto census = t.block_census();
    REQUIRE(census.size() == 3 && census[0].free && census[0].size == 4 * ONE_MB);
    REQUIRE(!census[1].free && census[2].free && census[2].offset == 5 * ONE_MB);

    // Both ends of the region and the whole block.
    REQUIRE(t.allocate_at(0, ONE_MB, 4) == 0);
    REQUIRE(t.allocate_at(16 * ONE_MB - ONE_MB, ONE_MB, 5) == 16 * ONE_MB - ONE_MB);
    REQUIRE(t.check_invariants());
    t.reset();
    REQUIRE(t.allocate_at(0, 16 * ONE_MB, 6) == 0 && t.used() == 16 * ONE_MB);
    REQUIRE(t.check_invariants());

    // The size rounds up to the block grain like allocate()'s.
    t.reset();
    REQUIRE(t.allocate_at(2 * ONE_MB, 300, 1) == 2 * ONE_MB && t.block_size_at(2 * ONE_MB) == 512);
    REQUIRE(t.check_invariants());
    t.reset();
    REQUIRE(t.allocate_at(0, 16 * ONE_MB, 6) == 0);

    // The block it carved frees and coalesces like any other.
    t.free(0);
    REQUIRE(t.block_census().size() == 1 && t.check_invariants());

    std::cout << "test_allocate_at_carves_exact: PASSED\n";
}

// Every refusal returns SIZE_MAX and changes nothing.
static void test_allocate_at_refusals() {
    test_arena       arena(16 * ONE_MB);
    tlsf_allocator & t = *arena.alloc;
    REQUIRE(t.allocate_at(4 * ONE_MB, ONE_MB, 3) == 4 * ONE_MB);
    REQUIRE(t.allocate_at(8 * ONE_MB, ONE_MB, 3) == 8 * ONE_MB);
    const auto   before = t.block_census();
    const size_t used   = t.used();

    REQUIRE(t.allocate_at(3 * ONE_MB, 2 * ONE_MB, 1) == SIZE_MAX && "a range reaching into an allocated block");
    REQUIRE(t.allocate_at(4 * ONE_MB, ONE_MB, 1) == SIZE_MAX && "an allocated range");
    REQUIRE(t.allocate_at(5 * ONE_MB - 256, 512, 1) == SIZE_MAX && "a range straddling a free and an allocated block");
    REQUIRE(t.allocate_at(2 * ONE_MB, 8 * ONE_MB, 1) == SIZE_MAX &&
            "a range across two free blocks and the one between");
    REQUIRE(t.allocate_at(6 * ONE_MB + 100, 256, 1) == SIZE_MAX && "an off-grain offset");
    REQUIRE(t.allocate_at(6 * ONE_MB, 0, 1) == SIZE_MAX && "a zero size");
    REQUIRE(t.allocate_at(16 * ONE_MB, 256, 1) == SIZE_MAX && "an offset at the region end");
    REQUIRE(t.allocate_at(16 * ONE_MB - 256, 512, 1) == SIZE_MAX && "a range past the region end");
    REQUIRE(t.allocate_at(SIZE_MAX - 255, 512, 1) == SIZE_MAX && "an offset that wraps");
    REQUIRE(t.allocate_at(6 * ONE_MB, SIZE_MAX, 1) == SIZE_MAX && "a size that wraps");

    const auto after = t.block_census();
    REQUIRE(after.size() == before.size() && t.used() == used);
    for (size_t i = 0; i < before.size(); ++i) {
        REQUIRE(before[i].offset == after[i].offset && before[i].size == after[i].size &&
                before[i].free == after[i].free);
    }
    REQUIRE(t.check_invariants());
    std::cout << "test_allocate_at_refusals: PASSED\n";
}

// A back remainder under MIN_BLOCK_SIZE (an off-grid region end) is absorbed
// into the carved block, as carve_gap does, and the sizes still tile.
static void test_allocate_at_absorbs_sub_min_remainder() {
    tlsf_allocator t(1000);
    REQUIRE(t.block_census().size() == 1);
    REQUIRE(t.allocate_at(256, 512, 2) == 256);
    REQUIRE(t.check_invariants());
    // 1000 - 256 - 512 = 232 < 256: absorbed into the carved block.
    REQUIRE(t.block_size_at(256) == 744);
    REQUIRE(t.used() == 744);
    const auto c = t.block_census();
    REQUIRE(c.size() == 2 && c[0].free && c[0].size == 256 && !c[1].free);
    t.free(256);
    REQUIRE(t.block_census().size() == 1 && t.check_invariants());
    std::cout << "test_allocate_at_absorbs_sub_min_remainder: PASSED\n";
}

// allocate_extent_at takes its size as the extent: a tail block that absorbed a
// sub-grain remainder goes back at its recorded size, which allocate_at() would
// round past the region end.
static void test_allocate_extent_at() {
    tlsf_allocator t(1000);
    REQUIRE(t.allocate_at(0, 768, 4) == 0 && t.block_size_at(0) == 1000);
    t.free(0);
    REQUIRE(t.allocate_at(0, 1000, 4) == SIZE_MAX && "allocate_at rounds 1000 to 1024, past the region end");
    REQUIRE(t.allocate_extent_at(0, 1000, 4) == 0 && t.block_size_at(0) == 1000 && t.tag_at(0) == 4);
    REQUIRE(t.check_invariants() && t.used() == 1000);
    t.free(0);
    REQUIRE(t.allocate_extent_at(100, 256, 1) == SIZE_MAX && "an off-grain offset");
    REQUIRE(t.allocate_extent_at(0, 0, 1) == SIZE_MAX && "a zero extent");
    REQUIRE(t.allocate_extent_at(0, 1001, 1) == SIZE_MAX && "an extent past the free block");
    REQUIRE(t.allocate_extent_at(256, SIZE_MAX - 100, 1) == SIZE_MAX && "an extent that wraps");
    REQUIRE(t.allocate_extent_at(256, 512, 2) == 256 && t.block_size_at(256) == 744 &&
            "a sub-grain remainder is still absorbed, as in allocate_at");
    REQUIRE(t.check_invariants());
    tlsf_allocator u(4096);
    REQUIRE(u.allocate_extent_at(256, 512, 2) == 256 && u.block_size_at(256) == 512 && "an on-grain extent is kept");
    REQUIRE(u.check_invariants());
    // The grid rule: an extent off the grain would leave the block after it off the grain, so it is
    // refused unless it ends exactly at the free block's end (a tail that absorbed a sub-grain remainder).
    tlsf_allocator v(4096);
    REQUIRE(v.allocate_extent_at(0, 300, 1) == SIZE_MAX && "an off-grain extent inside a larger free block");
    REQUIRE(v.check_invariants() && v.used() == 0 && v.allocate(256, 256, 2) == 0 && "and nothing was carved");
    tlsf_allocator w(1000);
    REQUIRE(w.allocate_extent_at(0, 300, 1) == SIZE_MAX && "an off-grain extent short of the free block's end");
    REQUIRE(w.allocate_extent_at(0, 800, 1) == SIZE_MAX &&
            "an off-grain extent is refused even when the 200-byte remainder would be absorbed: the extent is exact");
    REQUIRE(w.allocate_extent_at(256, 744, 1) == 256 && w.block_size_at(256) == 744 &&
            "an off-grain extent that ends exactly at the free block's end");
    REQUIRE(w.check_invariants());
    std::cout << "test_allocate_extent_at: PASSED\n";
}

// round_request is the one rounding rule: the larger of alignment and grain, 0 for
// a zero size and for a wrap.
static void test_round_request() {
    REQUIRE(tlsf_allocator::round_request(0, 256) == 0);
    REQUIRE(tlsf_allocator::round_request(1, 64) == 256);
    REQUIRE(tlsf_allocator::round_request(256, 256) == 256);
    REQUIRE(tlsf_allocator::round_request(257, 256) == 512);
    REQUIRE(tlsf_allocator::round_request(SIZE_MAX, 256) == 0);
    // An alignment over the grain asserts only for a request that gets as far as the rounding; a zero size or
    // one the guard refuses answers 0 first, as allocate() always has.
    REQUIRE(tlsf_allocator::round_request(0, 4096) == 0);
    REQUIRE(tlsf_allocator::round_request(SIZE_MAX, 4096) == 0);
    {
        tlsf_allocator t(4096);
        REQUIRE(t.allocate(0, 4096) == SIZE_MAX && "a zero size at an over-grain alignment is refused, not an abort");
    }
    // SIZE_MAX - 255 is on the grain and would round to itself; the guard `size > SIZE_MAX - granularity` is
    // conservative by one grain, so it is refused although it does not wrap.
    REQUIRE(tlsf_allocator::round_request(SIZE_MAX - 255, 256) == 0 &&
            "refused one grain early, though it does not wrap");
    REQUIRE(tlsf_allocator::round_request(SIZE_MAX - 256, 256) == ((SIZE_MAX - 256 + 255) & ~size_t(255)) &&
            "the largest representable round-up is kept");
    std::cout << "test_round_request: PASSED\n";
}

// allocate_excluding is first fit by offset over the parts of free blocks that
// lie outside every excluded range.
static void test_allocate_excluding_geometry() {
    test_arena       arena(16 * ONE_MB);
    tlsf_allocator & t = *arena.alloc;
    using R            = tlsf_allocator::excluded_range;

    // A range at the low end: the first fit starts above it.
    REQUIRE(t.allocate_excluding(
                {
                    R{ 0, 4 * ONE_MB }
    },
                ONE_MB, 256, 1) == 4 * ONE_MB);
    REQUIRE(t.check_invariants());
    // A gap whose top is a range still serves below it, and the free block the
    // range cuts is not split at its boundary.
    REQUIRE(t.allocate_excluding(
                {
                    R{ 0,          4 * ONE_MB },
                    R{ 6 * ONE_MB, 2 * ONE_MB }
    },
                ONE_MB, 256, 1) == 5 * ONE_MB);
    REQUIRE(t.check_invariants());
    // A request that fits only across a range misses.
    REQUIRE(t.allocate_excluding(
                {
                    R{ 0,          4 * ONE_MB },
                    R{ 6 * ONE_MB, 2 * ONE_MB }
    },
                2 * ONE_MB, 256, 1) == 8 * ONE_MB);
    REQUIRE(t.allocate_excluding(
                {
                    R{ 0,           4 * ONE_MB },
                    R{ 10 * ONE_MB, 6 * ONE_MB }
    },
                3 * ONE_MB, 256, 1) == SIZE_MAX);
    REQUIRE(t.check_invariants());
    t.reset();

    // A free block a range cuts keeps its remainders whole: the carve below the
    // range leaves one free block that STRADDLES it, so no two free blocks are
    // adjacent.
    REQUIRE(t.allocate_excluding(
                {
                    R{ 8 * ONE_MB, 4 * ONE_MB }
    },
                ONE_MB, 256, 1) == 0);
    auto c = t.block_census();
    REQUIRE(c.size() == 2 && !c[0].free && c[1].free && c[1].offset == ONE_MB && c[1].size == 15 * ONE_MB);
    REQUIRE(t.check_invariants());

    // Overlapping, unsorted, off-grain ranges: the end of the union decides.
    t.reset();
    REQUIRE(t.allocate_excluding(
                {
                    R{ 3 * ONE_MB, ONE_MB           },
                    R{ 0,          2 * ONE_MB + 100 },
                    R{ ONE_MB,     3 * ONE_MB       }
    },
                256, 256, 1) == 4 * ONE_MB);
    // An off-grain range end rounds UP to the next grain: that offset is the candidate.
    t.reset();
    REQUIRE(t.allocate_excluding(
                {
                    R{ 0, 2 * ONE_MB + 100 }
    },
                256, 256, 1) == 2 * ONE_MB + 256);
    // No exclusion is plain first fit; an empty size is refused.
    t.reset();
    REQUIRE(t.allocate_excluding({}, ONE_MB, 256, 1) == 0);
    REQUIRE(t.allocate_excluding({}, 0, 256, 1) == SIZE_MAX);
    REQUIRE(t.check_invariants());

    std::cout << "test_allocate_excluding_geometry: PASSED\n";
}

// allocate_excluding against an interval model: the offset it returns is the
// lowest one a brute-force scan over the grain finds, over random layouts.
static void test_allocate_excluding_interval_model() {
    uint32_t seed = 4242u;
    auto     rnd  = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return seed >> 8;
    };
    constexpr size_t grain = 256;
    constexpr size_t nblk  = 256;  // a 64 KB region
    for (int round = 0; round < 200; ++round) {
        tlsf_allocator    t(nblk * grain);
        std::vector<bool> used(nblk, false);
        for (int i = 0; i < 12; ++i) {
            const size_t at = rnd() % nblk;
            const size_t n  = 1 + rnd() % 8;
            if (at + n > nblk) {
                continue;
            }
            bool clear = true;
            for (size_t k = at; k < at + n; ++k) {
                clear = clear && !used[k];
            }
            const size_t got = t.allocate_at(at * grain, n * grain, 1);
            REQUIRE((got != SIZE_MAX) == clear && "allocate_at succeeds exactly when the range is free");
            if (clear) {
                for (size_t k = at; k < at + n; ++k) {
                    used[k] = true;
                }
            }
        }
        std::vector<tlsf_allocator::excluded_range> ex;
        std::vector<bool>                           barred = used;
        for (int i = 0; i < 4; ++i) {
            const size_t at = rnd() % nblk;
            const size_t n  = 1 + rnd() % 40;
            ex.push_back({ at * grain, n * grain });
            for (size_t k = at; k < at + n && k < nblk; ++k) {
                barred[k] = true;
            }
        }
        const size_t n    = 1 + rnd() % 10;
        size_t       want = SIZE_MAX;
        for (size_t at = 0; at + n <= nblk && want == SIZE_MAX; ++at) {
            bool ok = true;
            for (size_t k = at; k < at + n; ++k) {
                ok = ok && !barred[k];
            }
            if (ok) {
                want = at * grain;
            }
        }
        REQUIRE(t.allocate_excluding(ex, n * grain, 256, 2) == want && "the lowest offset outside every range");
        REQUIRE(t.check_invariants());
    }
    std::cout << "test_allocate_excluding_interval_model: PASSED\n";
}

// ---------------------------------------------------------------------------
// The exact-class fallback of allocate().  A MoE expert triplet's members are
// 921600, 921600 and 921816 bytes; 921816 rounds to 921856, and both sizes
// map to one second-level class, [917504, 950272).  A request the rounded
// search misses falls back to that class, whose free list can hold a fitting
// block behind a smaller head.
// ---------------------------------------------------------------------------
static constexpr size_t TRIPLET_SMALL         = 921600;  // two per triplet
static constexpr size_t TRIPLET_LARGE         = 921816;  // one per triplet
static constexpr size_t TRIPLET_LARGE_ROUNDED = 921856;
static constexpr size_t TRIPLET_BYTES         = 2 * TRIPLET_SMALL + TRIPLET_LARGE_ROUNDED;

static bool census_equal(const std::vector<tlsf_allocator::extent> & a, const std::vector<tlsf_allocator::extent> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].offset != b[i].offset || a[i].size != b[i].size || a[i].free != b[i].free || a[i].tag != b[i].tag) {
            return false;
        }
    }
    return true;
}

// Two free blocks of one class, separated by allocated 256-byte blocks so they
// cannot coalesce, and nothing else free.  `head` is freed last, so it heads
// the class's free list and `behind` sits after it.  Returns the offsets.
struct class_pair {
    size_t head_offset;
    size_t behind_offset;
};

static class_pair class_pair_setup(tlsf_allocator & t, size_t head, size_t behind) {
    const size_t head_offset   = t.allocate(head);
    const size_t sep_a         = t.allocate(256);
    const size_t behind_offset = t.allocate(behind);
    const size_t sep_b         = t.allocate(256);
    REQUIRE(head_offset != SIZE_MAX && sep_a != SIZE_MAX && behind_offset != SIZE_MAX && sep_b != SIZE_MAX);
    REQUIRE(t.available() == 0 && "the region holds exactly the four blocks");
    t.free(behind_offset);
    t.free(head_offset);
    REQUIRE(t.check_invariants());
    return { head_offset, behind_offset };
}

// A fitting block behind a smaller head is found.
static void test_allocate_class_walk_finds_block_behind_head() {
    REQUIRE(tlsf_allocator::round_request(TRIPLET_LARGE, 256) == TRIPLET_LARGE_ROUNDED);
    tlsf_allocator   t(TRIPLET_SMALL + 256 + TRIPLET_LARGE_ROUNDED + 256);
    const class_pair p = class_pair_setup(t, TRIPLET_SMALL, TRIPLET_LARGE_ROUNDED);
    REQUIRE(t.largest_free_block() == TRIPLET_LARGE_ROUNDED && "the fitting block is free, behind the head");
    const size_t used = t.used();
    const size_t got  = t.allocate(TRIPLET_LARGE);
    REQUIRE(got == p.behind_offset && "the request takes the fitting block behind the smaller head");
    REQUIRE(t.used() == used + TRIPLET_LARGE_ROUNDED);
    REQUIRE(t.check_invariants());
    std::cout << "test_allocate_class_walk_finds_block_behind_head: PASSED\n";
}

// The walk takes the smallest fitting block, not the first: the head would
// fit, but taking it would leave the larger request no room later.
static void test_allocate_class_walk_takes_smallest_fit() {
    tlsf_allocator   t(TRIPLET_LARGE_ROUNDED + 256 + TRIPLET_SMALL + 256);
    const class_pair p = class_pair_setup(t, TRIPLET_LARGE_ROUNDED, TRIPLET_SMALL);
    REQUIRE(t.largest_free_block() == TRIPLET_LARGE_ROUNDED && "the larger block is free, at the head");
    REQUIRE(t.allocate(TRIPLET_SMALL) == p.behind_offset && "the exact-size block behind the head is taken");
    REQUIRE(t.allocate(TRIPLET_LARGE) == p.head_offset && "the larger request still finds its block");
    REQUIRE(t.available() == 0);
    REQUIRE(t.check_invariants());
    std::cout << "test_allocate_class_walk_takes_smallest_fit: PASSED\n";
}

// No block of the class fits: the request is refused and nothing changes.
static void test_allocate_class_walk_refuses_when_none_fits() {
    tlsf_allocator   t(TRIPLET_SMALL + 256 + TRIPLET_SMALL + 256);
    const class_pair p      = class_pair_setup(t, TRIPLET_SMALL, TRIPLET_SMALL);
    const size_t     used   = t.used();
    const auto       before = t.block_census();
    REQUIRE(t.allocate(TRIPLET_LARGE) == SIZE_MAX && "no block of the class is large enough");
    REQUIRE(t.used() == used);
    REQUIRE(census_equal(t.block_census(), before) && "a refusal changes no block");
    REQUIRE(t.check_invariants());
    const size_t first  = t.allocate(TRIPLET_SMALL);
    const size_t second = t.allocate(TRIPLET_SMALL);
    REQUIRE(((first == p.head_offset && second == p.behind_offset) ||
             (first == p.behind_offset && second == p.head_offset)) &&
            "both blocks are still free and still whole");
    REQUIRE(t.check_invariants());
    std::cout << "test_allocate_class_walk_refuses_when_none_fits: PASSED\n";
}

// largest_free_block() is the largest block of the top class, not its head:
// the head is smaller, the larger block sits behind it, and a smaller block
// in a lower class changes nothing.
static void test_largest_free_block_walks_top_class() {
    tlsf_allocator t(4096 + 256 + TRIPLET_SMALL + 256 + TRIPLET_LARGE_ROUNDED + 256);
    const size_t   low = t.allocate(4096);
    REQUIRE(low != SIZE_MAX && t.allocate(256) != SIZE_MAX);
    const class_pair p = class_pair_setup(t, TRIPLET_SMALL, TRIPLET_LARGE_ROUNDED);
    t.free(low);
    REQUIRE(t.check_invariants());
    REQUIRE(t.largest_free_block() == TRIPLET_LARGE_ROUNDED && "the block behind the top class's head is the largest");
    // The answer is a size allocate() honours, with the head still smaller.
    REQUIRE(t.allocate(t.largest_free_block()) == p.behind_offset);
    REQUIRE(t.largest_free_block() == TRIPLET_SMALL && "the head is now the only block of the top class");
    REQUIRE(t.check_invariants());
    std::cout << "test_largest_free_block_walks_top_class: PASSED\n";
}

// With every byte allocated there is no free block: largest_free_block() is 0,
// and it returns to the freed size once a block is released.
static void test_largest_free_block_empty_pool() {
    tlsf_allocator t(TRIPLET_SMALL + TRIPLET_LARGE_ROUNDED);
    const size_t   a = t.allocate(TRIPLET_SMALL);
    const size_t   b = t.allocate(TRIPLET_LARGE);
    REQUIRE(a != SIZE_MAX && b != SIZE_MAX && t.available() == 0);
    REQUIRE(t.largest_free_block() == 0 && "a full pool has no free block");
    t.free(a);
    REQUIRE(t.largest_free_block() == TRIPLET_SMALL);
    REQUIRE(t.check_invariants());
    std::cout << "test_largest_free_block_empty_pool: PASSED\n";
}

// A region off the grain ends in a block off the grain.  largest_free_block()
// reports its whole size, a request rounded up past its end is refused, and
// segment_piece() (how the host pinned pool sizes each piece of a segmented
// allocation) gives the block's on-grain part, which is placed and takes the
// whole block.
static void test_largest_free_block_off_grain_tail() {
    const size_t   region = 16 * 256 + 100;
    tlsf_allocator t(region);
    REQUIRE(t.largest_free_block() == region);
    REQUIRE(t.allocate(region - 50) == SIZE_MAX && "rounded up to the grain, the request passes the block's end");
    REQUIRE(t.used() == 0);
    const size_t piece = tlsf_allocator::segment_piece(region - 50, t.largest_free_block(), 256);
    REQUIRE(piece == 16 * 256 && "the piece is the block's on-grain part");
    REQUIRE(t.allocate(piece) == 0 && "the on-grain part of the block is placed");
    REQUIRE(t.used() == region && "the sub-grain tail is absorbed into the block");
    REQUIRE(t.largest_free_block() == 0);
    REQUIRE(t.check_invariants());
    std::cout << "test_largest_free_block_off_grain_tail: PASSED\n";
}

// segment_piece(): the whole remainder rounded up to the granularity when that
// fits the block, otherwise the block's on-grain part.  The off-grain rows are
// the ones a piece rounded up past the block would get wrong.
static void test_segment_piece() {
    const size_t off = 16 * 256 + 100;  // a block ending 100 bytes past the grain
    // Fits: rounded up, never past the block.
    REQUIRE(tlsf_allocator::segment_piece(1000, 4096, 256) == 1024);
    REQUIRE(tlsf_allocator::segment_piece(100, 4096, 64) == 256 && "a small alignment still rounds to the grain");
    REQUIRE(tlsf_allocator::segment_piece(4096, 4096, 256) == 4096);
    REQUIRE(tlsf_allocator::segment_piece(5000, 8192, 256) == 5120);
    REQUIRE(tlsf_allocator::segment_piece(16 * 256, off, 256) == 16 * 256);
    // Larger than an on-grain block: the whole block.
    REQUIRE(tlsf_allocator::segment_piece(10000, 4096, 256) == 4096);
    // Off-grain block, remainder past its on-grain part: the on-grain part.
    REQUIRE(tlsf_allocator::segment_piece(16 * 256 + 1, off, 256) == 16 * 256);
    REQUIRE(tlsf_allocator::segment_piece(off - 50, off, 256) == 16 * 256);
    REQUIRE(tlsf_allocator::segment_piece(off, off, 256) == 16 * 256);
    REQUIRE(tlsf_allocator::segment_piece(10000, off, 256) == 16 * 256);
    // Nothing to place.
    REQUIRE(tlsf_allocator::segment_piece(0, 4096, 256) == 0);
    REQUIRE(tlsf_allocator::segment_piece(1000, 200, 256) == 0 && "a block below one granule holds no piece");
    // Every piece the helper gives for an off-grain region is placed by allocate().
    for (size_t remaining = 1; remaining <= 2 * off; remaining += 37) {
        tlsf_allocator t(off);
        const size_t   piece = tlsf_allocator::segment_piece(remaining, t.largest_free_block(), 256);
        REQUIRE(piece != 0 && t.allocate(piece) == 0 && "a segment piece always fits the block it was sized for");
    }
    std::cout << "test_segment_piece: PASSED\n";
}

// The last size class has no upper bound: every block from about 62 GiB up
// maps to it.  A request in it must not get a smaller block from the fast
// path, and a larger block behind that head must be found.  The region is
// bookkeeping only, so a region this size costs nothing.
static void test_allocate_unbounded_last_class() {
    const size_t     GiB   = size_t(1) << 30;
    const size_t     small = 63 * GiB;  // both blocks map to the last class
    const size_t     large = 66 * GiB;
    tlsf_allocator   t(small + 256 + large + 256);
    const class_pair p = class_pair_setup(t, small, large);
    REQUIRE(t.largest_free_block() == large);
    const size_t used = t.used();
    REQUIRE(t.allocate(67 * GiB) == SIZE_MAX && "no block of the last class holds the request");
    REQUIRE(t.used() == used);
    const size_t got = t.allocate(65 * GiB);
    REQUIRE(got == p.behind_offset && "the larger block behind the smaller head is taken");
    REQUIRE(t.block_size_at(got) >= 65 * GiB && "the block holds the request");
    REQUIRE(t.allocate(small) == p.head_offset);
    REQUIRE(t.check_invariants());
    std::cout << "test_allocate_unbounded_last_class: PASSED\n";
}

// A replay of expert triplets in a pool sized for exactly n of them: allocate
// every triplet with members (small, small, large), free the triplets in
// `freed` (members in `member_free_order`), then re-allocate as many triplets
// with members in `order`.  member_major allocates every triplet's member 0,
// then every member 1, then every member 2, so a freed triplet leaves three
// separate holes of the members' own sizes rather than one triplet-sized hole.
struct triplet_replay {
    bool                member_major;
    size_t              n;
    std::vector<size_t> freed;
    int                 member_free_order[3];
    size_t              order[3];
};

struct triplet_replay_result {
    size_t refused;
    bool   used_restored;
};

// Every refusal must be a shape refusal: no free block anywhere in the pool is
// as large as the rounded request.  A refusal while such a block exists is a
// search that missed a fit.
static triplet_replay_result triplet_replay_run(const triplet_replay & r) {
    tlsf_allocator                   t(r.n * TRIPLET_BYTES);
    const size_t                     first[3] = { TRIPLET_SMALL, TRIPLET_SMALL, TRIPLET_LARGE };
    std::vector<std::vector<size_t>> trip(r.n, std::vector<size_t>(3, SIZE_MAX));
    for (size_t step = 0; step < 3 * r.n; ++step) {
        const size_t i = r.member_major ? step % r.n : step / 3;
        const size_t k = r.member_major ? step / r.n : step % 3;
        trip[i][k]     = t.allocate(first[k]);
        REQUIRE(trip[i][k] != SIZE_MAX && "the first placement fits the pool exactly");
    }
    REQUIRE(t.available() == 0);
    const size_t used = t.used();
    for (const size_t i : r.freed) {
        for (const int k : r.member_free_order) {
            t.free(trip[i][k]);
        }
    }
    REQUIRE(t.check_invariants());
    size_t refused = 0;
    for (size_t j = 0; j < r.freed.size(); ++j) {
        for (const size_t s : r.order) {
            if (t.allocate(s) != SIZE_MAX) {
                continue;
            }
            refused++;
            const size_t want = tlsf_allocator::round_request(s, 256);
            for (const tlsf_allocator::extent & e : t.block_census()) {
                REQUIRE(!(e.free && e.size >= want) && "a refusal while a free block fits the request");
            }
        }
    }
    REQUIRE(t.check_invariants());
    return { refused, t.used() == used };
}

static std::vector<size_t> triplet_indices(size_t n, size_t mod, size_t rem, bool descending) {
    std::vector<size_t> v;
    for (size_t i = 0; i < n; ++i) {
        if (i % mod == rem) {
            v.push_back(i);
        }
    }
    if (descending) {
        std::reverse(v.begin(), v.end());
    }
    return v;
}

// Freed triplets that are not adjacent leave holes of exactly the member
// sizes, all in one class.  Re-allocating in another member order succeeds in
// full only if each request takes a hole of its own size: the class head
// alone refuses requests that a hole further down fits, and a first fit puts
// a small member into a large member's hole and refuses that large member.
static void test_allocate_class_walk_triplet_replay() {
    const size_t         S       = TRIPLET_SMALL;
    const size_t         L       = TRIPLET_LARGE;
    const size_t         n       = 64;
    const triplet_replay cases[] = {
        // Half the triplets, every other one.
        { true, n, triplet_indices(n, 2, 0, false), { 0, 1, 2 }, { S, L, S } },
        // The other half, freed from the top down.
        { true, n, triplet_indices(n, 2, 1, true),  { 0, 1, 2 }, { S, L, S } },
        // A third of them, large member first.
        { true, n, triplet_indices(n, 3, 0, false), { 0, 1, 2 }, { L, S, S } },
    };
    for (const triplet_replay & r : cases) {
        const triplet_replay_result res = triplet_replay_run(r);
        REQUIRE(res.refused == 0 && "every re-allocated triplet fits the room its predecessor freed");
        REQUIRE(res.used_restored && "used() returns to its value before the frees");
    }
    std::cout << "test_allocate_class_walk_triplet_replay: PASSED\n";
}

// Over a fixed set of replays, adjacent and member-major, with freed sets that
// do and do not coalesce, allocate() refuses only when no free block fits.
// Some of these replays do refuse: when the freed bytes equal the re-requested
// bytes, a 256-byte sliver left by a placement can refuse a later request
// under every policy.  Those refusals are the pool's shape, not the search.
static void test_allocate_refuses_only_without_a_fitting_block() {
    const size_t S = TRIPLET_SMALL;
    const size_t L = TRIPLET_LARGE;
    const size_t n = 64;

    std::vector<std::vector<size_t>> freed_sets;
    freed_sets.push_back(triplet_indices(n, 2, 0, false));
    freed_sets.push_back(triplet_indices(n, 2, 1, true));
    freed_sets.push_back(triplet_indices(n, 3, 0, false));
    std::vector<size_t> not_3_mod_4;
    std::vector<size_t> all;
    for (size_t i = 0; i < n; ++i) {
        if (i % 4 != 3) {
            not_3_mod_4.push_back(i);
        }
        all.push_back(i);
    }
    freed_sets.push_back(not_3_mod_4);
    freed_sets.push_back(all);
    std::vector<size_t> stride;
    for (size_t i = 0; i < n * 5 / 8; ++i) {
        stride.push_back((i * 37) % n);
    }
    freed_sets.push_back(stride);

    const size_t orders[3][3] = {
        { S, S, L },
        { S, L, S },
        { L, S, S }
    };
    const int free_orders[2][3] = {
        { 0, 1, 2 },
        { 2, 1, 0 }
    };
    size_t runs    = 0;
    size_t refused = 0;
    for (const bool member_major : { false, true }) {
        for (const std::vector<size_t> & freed : freed_sets) {
            for (const auto & order : orders) {
                for (const auto & free_order : free_orders) {
                    const triplet_replay r = {
                        member_major,
                        n,
                        freed,
                        { free_order[0], free_order[1], free_order[2] },
                        { order[0],      order[1],      order[2]      },
                    };
                    refused += triplet_replay_run(r).refused;
                    runs++;
                }
            }
        }
    }
    REQUIRE(runs == 72);
    // Shape refusals occur in this set, so the census check above has run.
    REQUIRE(refused > 0);
    std::cout << "test_allocate_refuses_only_without_a_fitting_block: PASSED\n";
}

// ---------------------------------------------------------------------------
// A WEIGHT zone through a MoE expert swap.  The load stages each expert
// tensor's device experts one after another, so the zone holds a run of one
// member size per tensor (down, gate and up of each layer, in the order the
// loader creates them).  The swap evicts whole triplets, then admits as many
// others, each minting its gate, then its up, then its down, and giving back
// what it took when a member finds no room.  `staged` is what each member asks
// of the zone at the load and at the admit, by role (gate, up, down).
// ---------------------------------------------------------------------------
struct expert_swap_result {
    size_t admitted  = 0;
    size_t refused   = 0;
    size_t available = 0;  // free bytes after the admits
    size_t largest   = 0;  // largest free block after the admits
};

static expert_swap_result expert_swap_run(const size_t (&staged)[3], size_t layers, size_t per_layer, size_t evicted) {
    const int    load_order[3] = { 2, 0, 1 };  // down, gate, up: the loader's tensor order within a layer
    const size_t triplets      = layers * per_layer;
    size_t       triplet_bytes = 0;
    for (const size_t s : staged) {
        triplet_bytes += tlsf_allocator::round_request(s, 256);
    }
    // Room for every triplet and a little more: the zone the plan filled, and a remainder it left.
    tlsf_allocator t(triplets * triplet_bytes + 16 * triplet_bytes);

    std::vector<std::array<size_t, 3>> offset(triplets);
    for (size_t l = 0; l < layers; ++l) {
        for (const int r : load_order) {
            for (size_t x = 0; x < per_layer; ++x) {
                offset[l * per_layer + x][r] = t.allocate(staged[r]);
                REQUIRE(offset[l * per_layer + x][r] != SIZE_MAX && "the load stages every planned expert");
            }
        }
    }
    std::vector<size_t> order(triplets);
    for (size_t i = 0; i < triplets; ++i) {
        order[i] = i;
    }
    std::mt19937 rng(12345);
    std::shuffle(order.begin(), order.end(), rng);
    for (size_t k = 0; k < evicted; ++k) {
        for (int r = 0; r < 3; ++r) {
            t.free(offset[order[k]][r]);
        }
    }

    expert_swap_result res;
    for (size_t k = 0; k < evicted; ++k) {
        size_t got[3] = { SIZE_MAX, SIZE_MAX, SIZE_MAX };
        bool   fits   = true;
        for (int r = 0; r < 3 && fits; ++r) {
            got[r] = t.allocate(staged[r]);
            fits   = got[r] != SIZE_MAX;
        }
        if (fits) {
            res.admitted++;
            continue;
        }
        res.refused++;
        for (const size_t o : got) {
            t.free(o);
        }
    }
    REQUIRE(t.check_invariants());
    res.available = t.available();
    res.largest   = t.largest_free_block();
    return res;
}

// The 640 x 2560 IQ4_NL expert of the swap the WEIGHT zone refused: gate and
// up stage 921600 bytes, down 921816 (its row-padding tail).
static constexpr size_t SWAP_GATE_UP = 921600;
static constexpr size_t SWAP_DOWN    = 921816;

// Each member at its own size: the zone refuses admits while it holds room for
// many triplets, in blocks a few grains short of a member.  No allocator
// policy fixes this (a search of the request's own class first refuses as
// many); the members' sizes do.  Characterization, kept so the slot test below
// reads against it.
static void test_expert_swap_mixed_member_sizes_fragment() {
    const size_t             staged[3] = { SWAP_GATE_UP, SWAP_GATE_UP, SWAP_DOWN };
    const size_t             triplet   = 2 * SWAP_GATE_UP + tlsf_allocator::round_request(SWAP_DOWN, 256);
    const expert_swap_result res       = expert_swap_run(staged, 24, 64, 1152);
    REQUIRE(res.refused > 0 && "admits are refused");
    REQUIRE(res.available >= 10 * triplet && "while the zone holds room for ten triplets and more");
    REQUIRE(res.largest < 3 * SWAP_GATE_UP && "in blocks too small for one triplet");
    std::cout << "test_expert_swap_mixed_member_sizes_fragment: PASSED (refused " << res.refused << " of 1152, "
              << res.available << " bytes free, largest block " << res.largest << ")\n";
}

// The slot rule: the largest staged member, rounded up to the grain.
static void test_expert_triplet_slot_bytes() {
    const size_t mixed[3] = { SWAP_GATE_UP, SWAP_GATE_UP, SWAP_DOWN };
    REQUIRE(expert_triplet_slot_bytes(mixed) == 921856 && "the padded down member, rounded to the grain");
    const size_t down_first[3] = { SWAP_DOWN, SWAP_GATE_UP, SWAP_GATE_UP };
    REQUIRE(expert_triplet_slot_bytes(down_first) == 921856 && "the largest member, wherever it is");
    const size_t uniform[3] = { 4096, 4096, 4096 };
    REQUIRE(expert_triplet_slot_bytes(uniform) == 4096 && "members already on the grain keep their size");
    // The device loan fixtures' Q4_0 expert of 640 columns: 1 MiB and 216 bytes of row padding round up one grain, to
    // 1 MiB and 256 bytes; the 1 MiB expert itself, already on the grain, is not bumped a grain.
    const size_t q4_0_640[3] = { 1048576 + 216, 1048576 + 216, 1048576 + 216 };
    REQUIRE(expert_triplet_slot_bytes(q4_0_640) == 1048576 + 256 && "216 bytes past the grain round up to one grain");
    const size_t aligned[3] = { 1048576, 1048576, 1048576 };
    REQUIRE(expert_triplet_slot_bytes(aligned) == 1048576 && "an aligned size stays as it is");
    const size_t sub_grain[3] = { 1, 100, 200 };
    REQUIRE(expert_triplet_slot_bytes(sub_grain) == 256 && "a slot is at least one grain");
    const size_t unstaged[3] = { SWAP_GATE_UP, 0, SWAP_DOWN };
    REQUIRE(expert_triplet_slot_bytes(unstaged) == 0 && "a member with no staged size: no slot");
    const size_t wraps[3] = { SWAP_GATE_UP, SIZE_MAX - 1, SWAP_DOWN };
    REQUIRE(expert_triplet_slot_bytes(wraps) == 0 && "a size the rounding would wrap: no slot");
    std::cout << "test_expert_triplet_slot_bytes: PASSED\n";
}

// A Q4_K gate and up beside a Q5_K down, 2048 x 512 and 512 x 2048: rows on
// the padding multiple, so each member's staged bytes are its own.
static constexpr size_t MIXED_GATE_UP = 512 * (2048 / 256) * 144;
static constexpr size_t MIXED_DOWN    = 2048 * (512 / 256) * 176;

// The slot only where it is nearly free: the padding it adds to a triplet,
// three slots less the members' grain-rounded bytes, is at most 1/128 of
// those bytes.  A triplet past that keeps each member's own size.
static void test_expert_triplet_slot_padding_bound() {
    const size_t iq4[3] = { SWAP_GATE_UP, SWAP_GATE_UP, SWAP_DOWN };
    REQUIRE(expert_triplet_slot_bytes(iq4) == 921856 && "a padded down member: 512 bytes of padding, under 1/128");
    const size_t mixed[3] = { MIXED_GATE_UP, MIXED_GATE_UP, MIXED_DOWN };
    REQUIRE(expert_triplet_slot_bytes(mixed) == 0 && "a Q4_K/Q4_K/Q5_K triplet would pad 14%: no slot");
    // 65280 x 2 + 66048 = 196608 bytes; padding 2 x 768 = 1536 = 196608 / 128.
    const size_t at_bound[3] = { 65280, 65280, 66048 };
    REQUIRE(expert_triplet_slot_bytes(at_bound) == 66048 && "padding exactly 1/128 of the triplet takes the slot");
    // One grain more on the down member: padding 2048 against 196864 / 128 = 1538.
    const size_t past_bound[3] = { 65280, 65280, 66304 };
    REQUIRE(expert_triplet_slot_bytes(past_bound) == 0 && "one grain past 1/128: no slot");
    std::cout << "test_expert_triplet_slot_padding_bound: PASSED\n";
}

// What the zone hands each member: the slot when the triplet has one, the
// member's own staged bytes when it has none.
static void test_expert_triplet_member_bytes() {
    const size_t iq4[3] = { SWAP_GATE_UP, SWAP_GATE_UP, SWAP_DOWN };
    for (int r = 0; r < 3; ++r) {
        REQUIRE(expert_triplet_member_bytes(iq4, r) == 921856 && "every member of a slotted triplet takes the slot");
    }
    const size_t mixed[3] = { MIXED_GATE_UP, MIXED_GATE_UP, MIXED_DOWN };
    for (int r = 0; r < 3; ++r) {
        REQUIRE(expert_triplet_member_bytes(mixed, r) == mixed[r] && "a triplet with no slot: each its own bytes");
    }
    std::cout << "test_expert_triplet_member_bytes: PASSED\n";
}

// What a triplet asks of the zone together: three slots when it has one, else
// its members' own sizes summed, and the slot it has (0 for none).
static void test_expert_triplet_zone_ask() {
    const size_t                  iq4[3]  = { SWAP_GATE_UP, SWAP_GATE_UP, SWAP_DOWN };
    const expert_triplet_zone_ask slotted = expert_triplet_zone_ask_of(iq4);
    REQUIRE(slotted.slot == 921856 && "a slotted triplet asks its slot");
    REQUIRE(slotted.total == 3 * 921856 && "and three of them");
    const size_t                  mixed[3] = { MIXED_GATE_UP, MIXED_GATE_UP, MIXED_DOWN };
    const expert_triplet_zone_ask own      = expert_triplet_zone_ask_of(mixed);
    REQUIRE(own.slot == 0 && "a triplet with no slot says so");
    REQUIRE(own.total == 2 * MIXED_GATE_UP + MIXED_DOWN && "and asks its members' own bytes");
    REQUIRE(own.total != 3 * MIXED_DOWN && "not three of its largest member");
    std::cout << "test_expert_triplet_zone_ask: PASSED\n";
}

// A triplet with no slot fragments the zone another way: a gate or up admit
// splits a freed down-sized block and leaves a remainder no member fits.
// Characterization of a known gap the slot rule leaves open on purpose.
static void test_expert_swap_unslotted_mixed_quant_fragments() {
    const size_t             staged[3] = { MIXED_GATE_UP, MIXED_GATE_UP, MIXED_DOWN };
    const size_t             triplet   = 2 * MIXED_GATE_UP + MIXED_DOWN;
    const expert_swap_result res       = expert_swap_run(staged, 24, 64, 1152);
    REQUIRE(res.refused > 0 && "admits are refused");
    REQUIRE(res.available >= 10 * triplet && "while the zone holds room for ten triplets and more");
    REQUIRE(res.largest < triplet && "in blocks too small for one triplet");
    std::cout << "test_expert_swap_unslotted_mixed_quant_fragments: PASSED (refused " << res.refused << " of 1152, "
              << res.available << " bytes free, largest block " << res.largest << ")\n";
}

// Every member at its triplet's slot: the same load and the same swap admit
// every triplet the swap evicted room for.
static void test_expert_swap_at_triplet_slot_admits_all() {
    const size_t members[3] = { SWAP_GATE_UP, SWAP_GATE_UP, SWAP_DOWN };
    const size_t slot       = expert_triplet_slot_bytes(members);
    REQUIRE(slot >= tlsf_allocator::round_request(SWAP_DOWN, 256) && "the slot holds every member");
    const size_t             staged[3] = { slot, slot, slot };
    const expert_swap_result res       = expert_swap_run(staged, 24, 64, 1152);
    REQUIRE(res.refused == 0 && res.admitted == 1152 && "every admit fits the room an evict freed");
    std::cout << "test_expert_swap_at_triplet_slot_admits_all: PASSED\n";
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
    test_allocate_at_carves_exact();
    test_allocate_at_refusals();
    test_allocate_at_absorbs_sub_min_remainder();
    test_allocate_extent_at();
    test_round_request();
    test_allocate_excluding_geometry();
    test_allocate_excluding_interval_model();
    test_allocate_class_walk_finds_block_behind_head();
    test_allocate_class_walk_takes_smallest_fit();
    test_allocate_class_walk_refuses_when_none_fits();
    test_largest_free_block_walks_top_class();
    test_largest_free_block_empty_pool();
    test_largest_free_block_off_grain_tail();
    test_segment_piece();
    test_allocate_unbounded_last_class();
    test_allocate_class_walk_triplet_replay();
    test_allocate_refuses_only_without_a_fitting_block();
    test_expert_swap_mixed_member_sizes_fragment();
    test_expert_triplet_slot_bytes();
    test_expert_triplet_slot_padding_bound();
    test_expert_triplet_member_bytes();
    test_expert_triplet_zone_ask();
    test_expert_swap_unslotted_mixed_quant_fragments();
    test_expert_swap_at_triplet_slot_admits_all();

    std::cout << "\nAll tlsf_allocator tests PASSED!\n";
    return 0;
}
