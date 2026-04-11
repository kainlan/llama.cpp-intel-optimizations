//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//
// Unit tests for the TLSF (Two-Level Segregated Fit) sub-allocator.
// Operates entirely on a CPU malloc region — no SYCL device required.
// Validates O(1) alloc/free semantics, coalescing, reset, and stress.

#include "../ggml/src/ggml-sycl/tlsf-allocator.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

// Active test macro that works regardless of NDEBUG (assert() is compiled
// away under NDEBUG, silently turning the entire test suite into a no-op).
#define REQUIRE(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "REQUIRE FAILED: %s at %s:%d\n", #cond, __FILE__, __LINE__); \
        abort(); \
    } \
} while(0)

using namespace ggml_sycl;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static constexpr size_t ONE_MB = 1024 * 1024;

// Allocate a 1 MB region with malloc and construct a tlsf_allocator over it.
struct test_arena {
    void *           mem;
    size_t           size;
    tlsf_allocator * alloc;

    explicit test_arena(size_t sz = ONE_MB) : size(sz) {
        mem = std::malloc(sz);
        REQUIRE(mem != nullptr && "malloc failed");
        alloc = new tlsf_allocator(mem, sz);
    }

    ~test_arena() {
        delete alloc;
        std::free(mem);
    }

    test_arena(const test_arena &)             = delete;
    test_arena & operator=(const test_arena &) = delete;
};

// ---------------------------------------------------------------------------
// 1. Basic allocation and free
// ---------------------------------------------------------------------------
static void test_basic_alloc_free() {
    test_arena arena;

    // Single allocation
    void * p = arena.alloc->allocate(256);
    REQUIRE(p != nullptr && "first alloc should succeed");
    REQUIRE(arena.alloc->used() > 0 && "used() should be non-zero after alloc");

    // Pointer should be within the arena
    auto addr = reinterpret_cast<uintptr_t>(p);
    REQUIRE(addr >= reinterpret_cast<uintptr_t>(arena.mem) &&
            addr < reinterpret_cast<uintptr_t>(arena.mem) + arena.size && "pointer within arena");
    (void) addr;

    // Free and verify
    size_t used_before = arena.alloc->used();
    (void) used_before;
    arena.alloc->free(p);
    REQUIRE(arena.alloc->used() < used_before && "used() should decrease after free");

    std::cout << "test_basic_alloc_free: PASSED\n";
}

// ---------------------------------------------------------------------------
// 2. Alignment — all returned pointers must be 256-byte aligned
// ---------------------------------------------------------------------------
static void test_alignment() {
    test_arena arena;

    for (int i = 0; i < 50; ++i) {
        size_t sz = 256 * (i + 1);
        void * p  = arena.alloc->allocate(sz);
        REQUIRE(p != nullptr && "allocation should succeed");
        auto addr = reinterpret_cast<uintptr_t>(p);
        (void) addr;
        REQUIRE((addr % 256) == 0 && "returned pointer must be 256-byte aligned");
    }

    std::cout << "test_alignment: PASSED\n";
}

// ---------------------------------------------------------------------------
// 3. Coalescing — free two adjacent blocks, verify they merge
// ---------------------------------------------------------------------------
static void test_coalescing() {
    test_arena arena;

    // Allocate two blocks adjacent in address order
    void * p1 = arena.alloc->allocate(4096);
    void * p2 = arena.alloc->allocate(4096);
    REQUIRE(p1 != nullptr && p2 != nullptr);

    size_t avail_before = arena.alloc->available();
    (void) avail_before;

    // Free p1 first (non-coalesce, p2 is allocated)
    arena.alloc->free(p1);
    size_t avail_after1 = arena.alloc->available();
    (void) avail_after1;

    // Free p2 — should coalesce with the freed p1
    arena.alloc->free(p2);
    size_t avail_after2 = arena.alloc->available();
    (void) avail_after2;

    // After coalescing, available should be strictly greater than after freeing
    // just p1, and should equal the original available (minus header overhead)
    REQUIRE(avail_after2 > avail_after1 && "coalescing should increase available");

    // All freed — available should be close to total minus header overhead
    size_t total_avail = arena.alloc->available();
    size_t overhead    = arena.size - total_avail;
    REQUIRE(overhead == tlsf_allocator::header_overhead() && "all memory should be reclaimed after full free");
    (void) overhead;

    std::cout << "test_coalescing: PASSED\n";
}

// ---------------------------------------------------------------------------
// 4. Reset — bulk deallocation returns all memory
// ---------------------------------------------------------------------------
static void test_reset() {
    test_arena arena;

    // Allocate a bunch
    void * p1 = arena.alloc->allocate(1024 * 64);
    void * p2 = arena.alloc->allocate(1024 * 32);
    void * p3 = arena.alloc->allocate(1024 * 128);
    (void) p1;
    (void) p2;
    (void) p3;

    REQUIRE(arena.alloc->used() > 0 && "should have allocations");

    // Reset
    arena.alloc->reset();
    REQUIRE(arena.alloc->used() == 0 && "used() must be 0 after reset");

    // Should be able to allocate the full arena (minus header) again
    size_t largest = arena.alloc->largest_free_block();
    REQUIRE(largest > 0 && "largest_free_block should be positive after reset");

    void * big = arena.alloc->allocate(largest);
    REQUIRE(big != nullptr && "should allocate largest block after reset");
    (void) big;

    std::cout << "test_reset: PASSED\n";
}

// ---------------------------------------------------------------------------
// 5. Allocation failure — request larger than available
// ---------------------------------------------------------------------------
static void test_alloc_failure() {
    // Tiny arena
    test_arena arena(512);  // Too small for even one block (needs header + 256)

    // No valid allocation should be possible
    void * p = arena.alloc->allocate(256);
    REQUIRE(p == nullptr && "allocation from tiny arena should fail");
    (void) p;

    std::cout << "test_alloc_failure: PASSED\n";
}

// ---------------------------------------------------------------------------
// 6. Exhaust and recycle — allocate all, free all, allocate again
// ---------------------------------------------------------------------------
static void test_exhaust_recycle() {
    test_arena arena;

    // Determine largest block
    size_t largest = arena.alloc->largest_free_block();
    REQUIRE(largest > 0);

    // Allocate the largest block
    void * p = arena.alloc->allocate(largest);
    REQUIRE(p != nullptr && "largest block alloc should succeed");

    // Now available should be reduced
    size_t avail_after = arena.alloc->available();
    REQUIRE(avail_after < arena.size && "available should decrease after large alloc");
    (void) avail_after;

    // Try another large allocation — should fail (fragmented)
    void * p2 = arena.alloc->allocate(largest);
    // This may or may not succeed depending on remaining space, but it
    // exercises the failure path.

    // Free and verify recovery
    arena.alloc->free(p);
    if (p2) {
        arena.alloc->free(p2);
    }

    // After freeing, we should be able to allocate again
    void * p3 = arena.alloc->allocate(256);
    REQUIRE(p3 != nullptr && "should allocate after free");
    arena.alloc->free(p3);

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

    void * slots[MAX_SLOTS] = {};

    for (int round = 0; round < N_ROUNDS; ++round) {
        // Pick a random slot to either alloc or free
        int idx = round % MAX_SLOTS;

        if (slots[idx] == nullptr) {
            // Allocate
            slots[idx] = arena.alloc->allocate(BLOCK_SIZE);
            // May fail if arena is full — that's fine
        } else {
            // Free
            arena.alloc->free(slots[idx]);
            slots[idx] = nullptr;
        }
    }

    // Free remaining slots
    for (auto & slot : slots) {
        if (slot != nullptr) {
            arena.alloc->free(slot);
            slot = nullptr;
        }
    }

    // After freeing everything, verify all memory reclaimed
    size_t total_avail = arena.alloc->available();
    size_t overhead    = arena.size - total_avail;
    REQUIRE(overhead == tlsf_allocator::header_overhead() && "all memory should be reclaimed after stress test");
    (void) overhead;

    std::cout << "test_stress: PASSED\n";
}

// ---------------------------------------------------------------------------
// 8. Free nullptr — should be a no-op (not crash)
// ---------------------------------------------------------------------------
static void test_free_null() {
    test_arena arena;
    // Should not crash
    arena.alloc->free(nullptr);

    std::cout << "test_free_null: PASSED\n";
}

// ---------------------------------------------------------------------------
// 9. Allocate zero — should return nullptr
// ---------------------------------------------------------------------------
static void test_alloc_zero() {
    test_arena arena;
    void *     p = arena.alloc->allocate(0);
    REQUIRE(p == nullptr && "alloc(0) should return nullptr");
    (void) p;

    std::cout << "test_alloc_zero: PASSED\n";
}

// ---------------------------------------------------------------------------
// 10. Largest free block tracking
// ---------------------------------------------------------------------------
static void test_largest_free_block() {
    test_arena arena;

    // Initially, largest should be nearly the entire arena (minus header)
    size_t initial_largest = arena.alloc->largest_free_block();
    REQUIRE(initial_largest > 0 && "initial largest_free_block should be positive");
    REQUIRE(initial_largest <= arena.size && "largest_free_block cannot exceed arena size");
    (void) initial_largest;

    // Allocate some memory
    void * p1 = arena.alloc->allocate(4096);
    REQUIRE(p1 != nullptr);

    // After allocation, largest free block should be <= initial
    size_t after_largest = arena.alloc->largest_free_block();
    REQUIRE(after_largest <= initial_largest && "largest_free_block should not increase after alloc");
    (void) after_largest;

    arena.alloc->free(p1);

    std::cout << "test_largest_free_block: PASSED\n";
}

// ---------------------------------------------------------------------------
// 11. Multiple alloc/free cycles — verify no memory leak
// ---------------------------------------------------------------------------
static void test_no_leak() {
    test_arena arena;

    for (int i = 0; i < 100; ++i) {
        void * p = arena.alloc->allocate(256 * (i % 32 + 1));
        if (p) {
            arena.alloc->free(p);
        }
    }

    // After all alloc/free cycles, all memory should be reclaimed
    size_t total_avail = arena.alloc->available();
    size_t overhead    = arena.size - total_avail;
    REQUIRE(overhead == tlsf_allocator::header_overhead() && "no memory leak after alloc/free cycles");
    (void) overhead;

    std::cout << "test_no_leak: PASSED\n";
}

// ---------------------------------------------------------------------------
// 12. Splitting — allocate small blocks and verify they don't overlap
// ---------------------------------------------------------------------------
static void test_splitting() {
    test_arena arena;

    // Allocate many small blocks
    constexpr int N = 16;
    void *        ptrs[N];
    for (int i = 0; i < N; ++i) {
        ptrs[i] = arena.alloc->allocate(256);
        REQUIRE(ptrs[i] != nullptr && "small alloc should succeed");
    }

    // Verify no two pointers overlap (each allocation should be at least
    // 256 + header_size apart)
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            auto   a    = reinterpret_cast<uintptr_t>(ptrs[i]);
            auto   b    = reinterpret_cast<uintptr_t>(ptrs[j]);
            // Two 256-byte allocations with 256-byte alignment should be
            // at least 256 bytes apart
            size_t dist = a > b ? a - b : b - a;
            assert(dist >= 256 && "allocations must not overlap");
            (void) dist;
        }
    }

    // Free all
    for (int i = 0; i < N; ++i) {
        arena.alloc->free(ptrs[i]);
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
    test_free_null();
    test_alloc_zero();
    test_largest_free_block();
    test_no_leak();
    test_splitting();

    std::cout << "\nAll tlsf_allocator tests PASSED!\n";
    return 0;
}
