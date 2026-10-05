// Host-only gate for the runtime allocation registry's containment lookup (llama.cpp-ii25):
// unified_lookup_runtime_allocation() answers through the registry's ordered range index instead of scanning every row.
//
// What it pins, on the real registry (rows published through the host-only test seams, no physical allocation):
//   - bounds: first byte, last byte hit; one past the end and one before the base miss;
//   - nesting: a chunk row and the suballocation rows inside it all contain an address, and the answer is the INNERMOST
//     one, deterministically, on every call (the scan this replaced returned whichever row unordered_map iteration
//     reached first);
//   - the index follows every registry mutation: release/erase, a stale RELEASING row displaced by a claim, a rekey
//     (erase then publish elsewhere), and a refused publish leaves nothing behind;
//   - a few thousand random rows agree with a brute-force oracle, and the index/registry consistency audit holds
//     throughout.
//
// The pure index logic is tests/test-address-range-index.cpp; this proves the registry is wired to it.

#include "ggml.h"
#include "unified-cache.hpp"

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using namespace ggml_sycl;

namespace {

int g_failures = 0;

void check(bool ok, const char * what) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) {
        g_failures++;
    }
}

void * at(uintptr_t a) {
    return reinterpret_cast<void *>(a);
}

// The row's identity rides in the `device` field of its metadata.
int owner_of(uintptr_t addr) {
    alloc_metadata m{};
    if (!unified_lookup_runtime_allocation(at(addr), &m, nullptr)) {
        return -1;
    }
    return m.device;
}

bool publish(uintptr_t base, size_t bytes, int tag) {
    return allocation_registry_test_publish_raw(at(base), tag, bytes, false, false, false);
}

void test_bounds_and_nesting() {
    printf("bounds and nesting:\n");
    const uintptr_t chunk = 0x100000;
    check(publish(chunk, 0x4000, 1), "chunk [0x100000,0x104000)");
    check(publish(chunk + 0x100, 0x100, 2), "sub [+0x100,+0x200)");
    check(publish(chunk + 0x300, 0x100, 3), "sub [+0x300,+0x400)");
    check(publish(chunk + 0x310, 0x10, 4), "subsub [+0x310,+0x320)");
    check(allocation_registry_test_index_consistent(), "index consistent after the publishes");

    check(owner_of(chunk - 1) == -1, "one byte before the chunk misses");
    check(owner_of(chunk) == 1, "the chunk's first byte -> chunk (it is also its base)");
    check(owner_of(chunk + 0x100) == 2, "a sub's first byte -> the sub, not the chunk");
    check(owner_of(chunk + 0x1ff) == 2, "a sub's last byte -> the sub");
    check(owner_of(chunk + 0x200) == 1, "one past the sub -> the enclosing chunk");
    check(owner_of(chunk + 0x310) == 4, "subsub's first byte -> subsub");
    check(owner_of(chunk + 0x31f) == 4, "subsub's last byte -> subsub");
    check(owner_of(chunk + 0x320) == 3, "one past the subsub -> its parent sub");
    check(owner_of(chunk + 0x3fff) == 1, "the chunk's last byte, past every sub, walks back to the chunk");
    check(owner_of(chunk + 0x4000) == -1, "one past the chunk misses");

    bool stable = true;
    for (int i = 0; i < 1000; i++) {
        stable = stable && owner_of(chunk + 0x315) == 4 && owner_of(chunk + 0x350) == 3;
    }
    check(stable, "the answer is the same on every call");

    alloc_metadata m{};
    sycl::queue *  q = reinterpret_cast<sycl::queue *>(0x1);
    check(unified_lookup_runtime_allocation(at(chunk + 0x150), &m, &q) && m.ptr == at(chunk + 0x100) &&
              m.size == 0x100 && q == nullptr,
          "the row's own metadata and queue are returned");
    check(unified_lookup_runtime_allocation(at(chunk + 0x150), nullptr, nullptr), "null out-params are accepted");
    check(!unified_lookup_runtime_allocation(nullptr, &m, nullptr) && m.ptr == nullptr,
          "null ptr misses and clears out");

    allocation_registry_test_erase(at(chunk + 0x310));
    check(owner_of(chunk + 0x315) == 3, "erasing the subsub: its bytes answer with the parent");
    allocation_registry_test_erase(at(chunk + 0x300));
    check(owner_of(chunk + 0x315) == 1, "erasing the sub: its bytes answer with the chunk");
    allocation_registry_test_erase(at(chunk));
    check(owner_of(chunk + 0x315) == -1 && owner_of(chunk + 0x150) == 2,
          "erasing the chunk: only the other sub remains");
    check(allocation_registry_test_index_consistent(), "index consistent after the erases");
    allocation_registry_test_erase(at(chunk + 0x100));
    check(allocation_registry_test_size() == 0 && allocation_registry_test_index_consistent(), "registry empty again");
}

void test_mutations_follow() {
    printf("index follows every mutation:\n");
    const uintptr_t base = 0x200000;
    check(publish(base, 0x1000, 10), "publish");
    check(!publish(base, 0x2000, 11), "a second row at the same pointer is refused");
    check(owner_of(base + 0x800) == 10 && owner_of(base + 0x1800) == -1, "and the refused row left no range behind");

    // A stale RELEASING row displaced by a claim: the recycled address is handed to a new allocation.
    const uintptr_t stale = 0x300000;
    check(allocation_registry_test_publish_raw(at(stale), 20, 0x1000, false, false, true), "publish a RELEASING row");
    check(owner_of(stale + 8) == 20, "a RELEASING row is still a row");
    check(allocation_registry_test_claim_ptr(at(stale)), "claim displaces the stale row");
    check(owner_of(stale + 8) == -1, "its range is gone with it");
    check(publish(stale, 0x80, 21), "the address is reused by a smaller allocation");
    check(owner_of(stale + 0x7f) == 21 && owner_of(stale + 0x80) == -1, "the new row's own bounds apply");

    // Rekey: the same pointer, new size.
    allocation_registry_test_erase(at(base));
    check(publish(base, 0x40, 12), "rekey: erase then publish the same base at a new size");
    check(owner_of(base + 0x3f) == 12 && owner_of(base + 0x40) == -1, "the old extent no longer answers");
    check(allocation_registry_test_index_consistent(), "index consistent");
    allocation_registry_test_erase(at(base));
    allocation_registry_test_erase(at(stale));
    check(allocation_registry_test_size() == 0 && allocation_registry_test_index_consistent(), "registry empty again");
}

void test_random_against_oracle() {
    printf("random rows vs a brute-force oracle:\n");

    struct row {
        uintptr_t base;
        uintptr_t end;
        int       tag;
    };

    std::mt19937_64  rng(11);
    std::vector<row> rows;
    int              mismatches = 0;
    int              next_tag   = 100;

    auto oracle = [&](uintptr_t addr) {
        const row * best = nullptr;
        for (const auto & r : rows) {
            if (addr >= r.base && addr < r.end && (best == nullptr || r.base > best->base)) {
                best = &r;
            }
        }
        return best ? best->tag : -1;
    };

    for (int i = 0; i < 6000; i++) {
        const int kind = static_cast<int>(rng() % 10);
        if (kind < 5) {
            const uintptr_t base = 0x1000000 + (rng() % 0x4000) * 16;
            const size_t    size = 16 * (1 + rng() % 256);
            const int       tag  = next_tag++;
            bool            dup  = false;
            for (const auto & r : rows) {
                dup = dup || r.base == base;
            }
            const bool ok = publish(base, size, tag);
            if (ok == dup) {
                mismatches++;
            }
            if (ok) {
                rows.push_back({ base, base + size, tag });
            }
        } else if (kind < 7 && !rows.empty()) {
            const size_t pick = static_cast<size_t>(rng() % rows.size());
            allocation_registry_test_erase(at(rows[pick].base));
            rows[pick] = rows.back();
            rows.pop_back();
        } else {
            uintptr_t addr = 0x1000000 + rng() % (0x4000 * 16 + 4096);
            if (!rows.empty() && (rng() & 1)) {
                const row &     r      = rows[static_cast<size_t>(rng() % rows.size())];
                const uintptr_t cand[] = { r.base, r.base - 1, r.end - 1, r.end };
                addr                   = cand[rng() % 4];
            }
            if (owner_of(addr) != oracle(addr)) {
                mismatches++;
            }
        }
        if (i % 1000 == 0 && !allocation_registry_test_index_consistent()) {
            mismatches++;
        }
    }
    printf("    %zu live rows at the end, %d mismatches\n", rows.size(), mismatches);
    check(mismatches == 0, "the registry lookup agrees with the oracle on every operation");
    check(allocation_registry_test_size() == rows.size() && allocation_registry_test_index_consistent(),
          "row count and index consistency hold");
    for (const auto & r : rows) {
        allocation_registry_test_erase(at(r.base));
    }
    check(allocation_registry_test_size() == 0 && allocation_registry_test_index_consistent(), "drained");
}

}  // namespace

int main() {
    test_bounds_and_nesting();
    test_mutations_follow();
    test_random_against_oracle();
    if (g_failures != 0) {
        printf("FAILED: %d check(s)\n", g_failures);
        return 1;
    }
    printf("ALL PASS\n");
    return 0;
}
