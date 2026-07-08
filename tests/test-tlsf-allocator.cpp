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

#include "../ggml/src/ggml-sycl/tlsf-allocator.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

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
        mem = std::aligned_alloc(256, sz);
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

    std::cout << "\nAll tlsf_allocator tests PASSED!\n";
    return 0;
}
