// Host-only gate for the ordered address-range index behind the runtime allocation registry's containment lookup
// (range-index.hpp). unified_lookup_runtime_allocation() used to scan every registry row per call; with a 512-expert
// MoE the registry holds tens of thousands of rows and that scan was 61.65% of decode CPU.
//
// The contract under test:
//   - find_innermost(addr) returns the range with the GREATEST base among those containing addr, which in a properly
//     nested family (a chunk and its suballocations) is the smallest enclosing range. It is deterministic; the scan it
//     replaces returned whichever row unordered_map iteration reached first.
//   - [base, base + size) is half-open: the first byte hits, the last byte hits, one past the end misses.
//   - It stays correct, and O(log n), for arbitrary overlap, not only nesting.
//   - erase()/insert() keep it in step (a rekey is an erase plus an insert).
//   - find_first_base_in(lo, hi) returns the range with the smallest base in [lo, hi), or nothing: "does any registered
//     row START inside this span" (the zone-settle liveness test, llama.cpp-rriv). A range that begins below lo and
//     reaches into the span is NOT answered; that is containment, which find_innermost() owns.
//
// The oracle is a brute-force vector scan. Plain C++, no SYCL: the header has no backend dependency.

#include "range-index.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <unordered_map>
#include <vector>

using ggml_sycl::address_range_index;
using entry = address_range_index::entry;

namespace {

int g_failures = 0;

void check(bool ok, const char * what) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) {
        g_failures++;
    }
}

// Brute-force reference: the container with the greatest base.
struct oracle_row {
    uintptr_t base;
    uintptr_t end;
    void *    key;
};

struct oracle {
    std::vector<oracle_row> rows;

    bool insert(uintptr_t base, size_t size, void * key) {
        const uintptr_t end = base + size < base ? UINTPTR_MAX : base + size;
        if (size == 0 || end <= base) {
            return false;
        }
        for (const auto & r : rows) {
            if (r.base == base) {
                return false;
            }
        }
        rows.push_back({ base, end, key });
        return true;
    }

    bool resize(uintptr_t base, void * key, size_t size) {
        const uintptr_t end = base + size < base ? UINTPTR_MAX : base + size;
        if (size == 0 || end <= base) {
            return false;
        }
        for (auto & r : rows) {
            if (r.base == base && r.key == key) {
                r.end = end;
                return true;
            }
        }
        return false;
    }

    bool erase(uintptr_t base, void * key) {
        for (size_t i = 0; i < rows.size(); i++) {
            if (rows[i].base == base && rows[i].key == key) {
                rows.erase(rows.begin() + static_cast<std::ptrdiff_t>(i));
                return true;
            }
        }
        return false;
    }

    bool find_first_base_in(uintptr_t lo, uintptr_t hi, entry * out) const {
        const oracle_row * best = nullptr;
        for (const auto & r : rows) {
            if (r.base >= lo && r.base < hi && (best == nullptr || r.base < best->base)) {
                best = &r;
            }
        }
        if (best == nullptr) {
            return false;
        }
        *out = { best->base, best->end, best->key };
        return true;
    }

    bool find(uintptr_t addr, entry * out) const {
        const oracle_row * best = nullptr;
        for (const auto & r : rows) {
            if (addr >= r.base && addr < r.end && (best == nullptr || r.base > best->base)) {
                best = &r;
            }
        }
        if (best == nullptr) {
            return false;
        }
        *out = { best->base, best->end, best->key };
        return true;
    }
};

void * key_of(uintptr_t n) {
    return reinterpret_cast<void *>(n * 16 + 8);
}

bool same(const entry & a, const entry & b) {
    return a.base == b.base && a.end == b.end && a.key == b.key;
}

void test_basic_bounds() {
    printf("bounds:\n");
    address_range_index idx;
    check(idx.size() == 0, "empty index has size 0");
    entry e;
    check(!idx.find_innermost(100, &e), "empty index finds nothing");
    check(!idx.insert(100, 0, key_of(1)), "a zero-size range is refused");
    check(idx.insert(100, 50, key_of(1)), "insert [100,150)");
    check(!idx.insert(100, 10, key_of(2)), "a second range at the same base is refused");
    check(idx.size() == 1, "size is 1");
    check(!idx.insert(UINTPTR_MAX, 5, key_of(3)), "a range whose clamped end equals its base is empty and is refused");
    check(idx.size() == 1, "and it was not added");
    check(!idx.find_innermost(99, &e), "one byte before the base misses");
    check(idx.find_innermost(100, &e) && e.base == 100 && e.end == 150 && e.key == key_of(1), "first byte hits");
    check(idx.find_innermost(149, &e) && e.key == key_of(1), "last byte hits");
    check(!idx.find_innermost(150, &e), "one past the end misses");
    check(idx.find_exact(100, &e) && e.key == key_of(1), "find_exact at the base");
    check(!idx.find_exact(101, &e), "find_exact is not containment");
    check(!idx.erase(100, key_of(2)), "erase with another row's key leaves the row");
    check(idx.find_innermost(120, &e), "the row survived the mismatched erase");
    check(idx.erase(100, key_of(1)), "erase");
    check(!idx.find_innermost(120, &e), "erased range misses");
    check(idx.size() == 0 && idx.check_invariants(), "empty again, invariants hold");
}

void test_address_space_end() {
    printf("address-space end:\n");
    address_range_index idx;
    entry               e;
    const uintptr_t     top = UINTPTR_MAX - 15;
    check(idx.insert(top, 4096, key_of(1)), "a range running past UINTPTR_MAX is clamped, not wrapped");
    check(idx.find_innermost(UINTPTR_MAX - 1, &e) && e.key == key_of(1), "its last addressable byte hits");
    check(!idx.find_innermost(top - 1, &e), "the byte before it misses");
    check(idx.check_invariants(), "invariants hold");
}

void test_nesting() {
    printf("nesting:\n");
    address_range_index idx;
    entry               e;
    // A chunk and its suballocations, one of them split again.
    check(idx.insert(1000, 4000, key_of(1)), "chunk [1000,5000)");
    check(idx.insert(1100, 100, key_of(2)), "sub [1100,1200)");
    check(idx.insert(1300, 100, key_of(3)), "sub [1300,1400)");
    check(idx.insert(1310, 10, key_of(4)), "subsub [1310,1320)");

    check(idx.find_innermost(1000, &e) && e.key == key_of(1), "chunk-only byte -> chunk");
    check(idx.find_innermost(1100, &e) && e.key == key_of(2), "sub's first byte -> sub, not the chunk");
    check(idx.find_innermost(1199, &e) && e.key == key_of(2), "sub's last byte -> sub");
    check(idx.find_innermost(1200, &e) && e.key == key_of(1), "one past the sub -> the enclosing chunk");
    check(idx.find_innermost(1305, &e) && e.key == key_of(3), "between sub and subsub -> sub");
    check(idx.find_innermost(1310, &e) && e.key == key_of(4), "subsub's first byte -> subsub");
    check(idx.find_innermost(1319, &e) && e.key == key_of(4), "subsub's last byte -> subsub");
    check(idx.find_innermost(1320, &e) && e.key == key_of(3), "one past the subsub -> its parent sub");
    check(idx.find_innermost(4999, &e) && e.key == key_of(1),
          "chunk's last byte past all subs walks back to the chunk");
    check(!idx.find_innermost(5000, &e), "one past the chunk misses");

    check(idx.erase(1310, key_of(4)), "erase the subsub");
    check(idx.find_innermost(1315, &e) && e.key == key_of(3), "its bytes now answer with the parent sub");
    check(idx.erase(1300, key_of(3)), "erase the sub");
    check(idx.find_innermost(1315, &e) && e.key == key_of(1), "and then with the chunk");
    check(idx.erase(1000, key_of(1)), "erase the chunk");
    check(!idx.find_innermost(1315, &e), "and then with nothing");
    check(idx.find_innermost(1150, &e) && e.key == key_of(2), "the surviving sub is untouched");
    check(idx.check_invariants(), "invariants hold");
}

void test_resize() {
    printf("resize:\n");
    address_range_index idx;
    entry               e;
    check(idx.insert(1000, 4000, key_of(1)) && idx.insert(1100, 100, key_of(2)),
          "chunk [1000,5000) and sub [1100,1200)");
    check(idx.find_innermost(1250, &e) && e.key == key_of(1), "past the sub the chunk answers");
    check(idx.resize(1100, key_of(2), 400), "grow the sub to [1100,1500)");
    check(idx.find_innermost(1250, &e) && e.key == key_of(2) && e.end == 1500, "its new extent answers");
    check(idx.find_innermost(1500, &e) && e.key == key_of(1), "one past it is the chunk again");
    check(idx.resize(1100, key_of(2), 10), "shrink the sub to [1100,1110)");
    check(idx.find_innermost(1109, &e) && e.key == key_of(2) && idx.find_innermost(1110, &e) && e.key == key_of(1),
          "the shrunk bounds apply");
    check(idx.resize(1000, key_of(1), 100), "shrink the chunk below its sub's end");
    check(idx.find_innermost(4000, &e) == false, "the chunk's old tail no longer answers (max_end was recomputed)");
    check(!idx.resize(1100, key_of(9), 50), "the wrong key resizes nothing");
    check(!idx.resize(1101, key_of(2), 50), "a base that is not present resizes nothing");
    check(!idx.resize(1100, key_of(2), 0), "an empty new size is refused");
    check(idx.find_innermost(1105, &e) && e.end == 1110, "refusals changed nothing");
    check(idx.size() == 2 && idx.check_invariants(), "size 2, invariants hold");
}

void test_rekey() {
    printf("rekey:\n");
    address_range_index idx;
    entry               e;
    check(idx.insert(2000, 100, key_of(7)), "insert [2000,2100)");
    check(idx.erase(2000, key_of(7)) && idx.insert(3000, 400, key_of(7)), "rekey = erase + insert at a new base/size");
    check(!idx.find_innermost(2050, &e), "the old range is gone");
    check(idx.find_innermost(3399, &e) && e.key == key_of(7) && e.base == 3000, "the new range answers");
    check(idx.size() == 1 && idx.check_invariants(), "size 1, invariants hold");
}

void test_first_base_in() {
    printf("first base in [lo, hi):\n");
    address_range_index idx;
    entry               e;
    check(!idx.find_first_base_in(0, UINTPTR_MAX, &e), "an empty index has no row in any span");
    check(idx.insert(1000, 100, key_of(1)) && idx.insert(2000, 100, key_of(2)) && idx.insert(3000, 100, key_of(3)),
          "three rows at 1000, 2000, 3000");
    check(idx.find_first_base_in(0, UINTPTR_MAX, &e) && e.base == 1000 && e.key == key_of(1),
          "the whole space answers the lowest base");
    check(idx.find_first_base_in(1000, 1001, &e) && e.base == 1000, "lo is inclusive");
    check(!idx.find_first_base_in(1001, 2000, &e), "hi is exclusive, and a base below lo is not found");
    check(idx.find_first_base_in(1001, 2001, &e) && e.base == 2000 && e.end == 2100 && e.key == key_of(2),
          "the span (1001, 2001) answers the row at 2000, with its full entry");
    check(!idx.find_first_base_in(2500, 2500, &e) && !idx.find_first_base_in(2600, 2500, &e),
          "an empty or inverted span answers nothing");
    check(!idx.find_first_base_in(1050, 1090, &e),
          "a span wholly INSIDE a range does not answer it: only a base inside the span counts");
    check(!idx.find_first_base_in(3001, UINTPTR_MAX, &e), "above the highest base");
    check(idx.erase(2000, key_of(2)) && !idx.find_first_base_in(1500, 2500, &e), "an erased row is not found");
    check(idx.insert(UINTPTR_MAX - 100, 50, key_of(4)) && idx.find_first_base_in(UINTPTR_MAX - 100, UINTPTR_MAX, &e) &&
              e.key == key_of(4),
          "a row near the top of the address space");
    check(idx.check_invariants(), "the query changed nothing");
}

// Random ranges, both laminar (nested) and arbitrarily overlapping, against the brute-force oracle.
void run_random(const char * name, bool laminar, uint32_t seed, int ops, uintptr_t space) {
    printf("random vs oracle (%s, seed %u, %d ops):\n", name, seed, ops);
    std::mt19937_64                           rng(seed);
    address_range_index                       idx;
    oracle                                    ref;
    int                                       mismatches = 0;
    int                                       refusals   = 0;
    int                                       resizes    = 0;
    int                                       spans      = 0;
    uintptr_t                                 next_key   = 1;
    std::vector<std::pair<uintptr_t, void *>> live;

    for (int i = 0; i < ops; i++) {
        const int kind = static_cast<int>(rng() % 10);
        if (kind < 4) {
            uintptr_t base = rng() % space;
            size_t    size = 1 + rng() % (laminar ? 64 : 4096);
            if (laminar) {
                // Snap to a power-of-two grid so ranges are either nested or disjoint.
                const unsigned  shift = static_cast<unsigned>(rng() % 12);
                const uintptr_t span  = static_cast<uintptr_t>(1) << shift;
                base                  = base / span * span;
                size                  = span;
            }
            void *     key = key_of(next_key++);
            const bool a   = idx.insert(base, size, key);
            const bool b   = ref.insert(base, size, key);
            if (a != b) {
                mismatches++;
            } else if (a) {
                live.push_back({ base, key });
            } else {
                refusals++;  // both refused: a duplicate base
            }
        } else if (kind < 6 && !live.empty()) {
            const size_t pick = static_cast<size_t>(rng() % live.size());
            const auto   row  = live[pick];
            live[pick]        = live.back();
            live.pop_back();
            const bool a = idx.erase(row.first, row.second);
            const bool b = ref.erase(row.first, row.second);
            if (a != b) {
                mismatches++;
            }
        } else if (kind == 6) {
            // Resize a live row, or try with the wrong key / a missing base; both must agree with the oracle.
            uintptr_t base = rng() % space;
            void *    key  = key_of(next_key++);
            if (!live.empty() && (rng() & 3) != 0) {
                const auto & row = live[static_cast<size_t>(rng() % live.size())];
                base             = row.first;
                key              = (rng() & 7) != 0 ? row.second : key;
            }
            const size_t size = (rng() % 8) == 0 ? 0 : 1 + rng() % 4096;
            const bool   a    = idx.resize(base, key, size);
            const bool   b    = ref.resize(base, key, size);
            if (a != b) {
                mismatches++;
            } else if (a) {
                resizes++;
            }
        } else {
            // Probe a row's boundaries as well as random addresses.
            uintptr_t addr = rng() % (space + 4096);
            if (!ref.rows.empty() && (rng() & 1)) {
                const oracle_row & r      = ref.rows[static_cast<size_t>(rng() % ref.rows.size())];
                const uintptr_t    cand[] = { r.base, r.base - 1, r.end - 1, r.end };
                addr                      = cand[rng() % 4];
            }
            entry      a, b;
            const bool fa = idx.find_innermost(addr, &a);
            const bool fb = ref.find(addr, &b);
            if (fa != fb || (fa && !same(a, b))) {
                mismatches++;
            }
            // The same probe as a span: [addr, addr + width), and one whose bounds sit on row bases.
            uintptr_t lo = addr;
            uintptr_t hi = addr + rng() % 512;
            if (!ref.rows.empty() && (rng() & 1)) {
                const oracle_row & r = ref.rows[static_cast<size_t>(rng() % ref.rows.size())];
                lo                   = (rng() & 1) ? r.base : r.base + 1;
                hi                   = (rng() & 1) ? r.base + 1 : r.end;
            }
            entry      c, d;
            const bool fc = idx.find_first_base_in(lo, hi, &c);
            const bool fd = ref.find_first_base_in(lo, hi, &d);
            spans++;
            if (fc != fd || (fc && !same(c, d))) {
                mismatches++;
            }
        }
        if (i % 4096 == 0 && !idx.check_invariants()) {
            mismatches++;
        }
        if (idx.size() != ref.rows.size()) {
            mismatches++;
        }
    }
    printf("    %zu live rows at the end, %d refused duplicates, %d resizes, %d mismatches\n", ref.rows.size(),
           refusals, resizes, mismatches);
    check(mismatches == 0, "index agrees with the oracle on every operation");
    check(spans > 1000, "the run probed spans as well as addresses");
    check(refusals > 0, "the run exercised duplicate-base refusals");
    check(resizes > 0, "the run exercised resizes");
    check(idx.check_invariants(), "invariants hold at the end");
    check(ref.rows.size() > 100, "the run kept a population worth checking");
}

// What the old lookup was: a scan over every row of an unordered_map.
struct fat_row {
    uintptr_t base;
    size_t    size;
    char      pad[200];  // runtime_alloc_record is a couple hundred bytes; keep the cache behaviour comparable
};

void microbench() {
    printf("microbench (100k rows):\n");
    constexpr size_t                    N = 100000;
    std::unordered_map<void *, fat_row> linear;
    address_range_index                 idx;
    std::vector<uintptr_t>              bases;
    linear.reserve(N);

    // 64 big chunks, each holding an equal share of 4 KiB..64 KiB suballocations; one-in-four addresses fall in a chunk
    // gap so the chunk-fallback path is exercised as well.
    std::mt19937_64 rng(7);
    uintptr_t       cursor = 1u << 20;
    size_t          made   = 0;
    for (size_t c = 0; c < 64 && made < N; c++) {
        const uintptr_t chunk_base = cursor;
        uintptr_t       off        = 0;
        const size_t    per_chunk  = N / 64;
        for (size_t s = 0; s < per_chunk; s++) {
            const size_t    size = 4096 * (1 + rng() % 16);
            const uintptr_t b    = chunk_base + off;
            off += size + 4096 * (rng() % 2);
            linear.emplace(reinterpret_cast<void *>(b + 1), fat_row{ b + 1, size, {} });
            idx.insert(b + 1, size, reinterpret_cast<void *>(b + 1));
            bases.push_back(b + 1);
            made++;
        }
        // The chunk row itself encloses them all (its base is below every sub's).
        linear.emplace(reinterpret_cast<void *>(chunk_base), fat_row{ chunk_base, off + 4096, {} });
        idx.insert(chunk_base, off + 4096, reinterpret_cast<void *>(chunk_base));
        cursor = chunk_base + off + 8192;
    }
    printf("    index rows: %zu, registry-shaped rows: %zu\n", idx.size(), linear.size());

    std::vector<uintptr_t> probes;
    for (int i = 0; i < 4096; i++) {
        const uintptr_t b = bases[rng() % bases.size()];
        probes.push_back(b + 1 + rng() % 4096);
    }

    using clk               = std::chrono::steady_clock;
    volatile uintptr_t sink = 0;

    const int linear_n = 50;
    auto      t0       = clk::now();
    for (int i = 0; i < linear_n; i++) {
        const uintptr_t addr = probes[static_cast<size_t>(i) % probes.size()];
        for (const auto & kv : linear) {
            if (addr >= kv.second.base && addr < kv.second.base + kv.second.size) {
                sink = sink + kv.second.base;
                break;
            }
        }
    }
    auto t1 = clk::now();

    const int indexed_n = 200000;
    entry     e;
    for (int i = 0; i < indexed_n; i++) {
        if (idx.find_innermost(probes[static_cast<size_t>(i) % probes.size()], &e)) {
            sink = sink + e.base;
        }
    }
    auto t2 = clk::now();

    const double old_ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / linear_n;
    const double new_ns = std::chrono::duration<double, std::nano>(t2 - t1).count() / indexed_n;
    printf("    linear scan : %12.0f ns/lookup (%d lookups)\n", old_ns, linear_n);
    printf("    range index : %12.1f ns/lookup (%d lookups)\n", new_ns, indexed_n);
    printf("    speedup     : %12.0fx\n", old_ns / new_ns);
    (void) sink;
    check(new_ns * 10.0 < old_ns, "the index is at least 10x faster than the scan at 100k rows");
}

}  // namespace

int main() {
    test_basic_bounds();
    test_address_space_end();
    test_nesting();
    test_rekey();
    test_resize();
    test_first_base_in();
    run_random("nested", true, 1, 100000, 1u << 16);
    run_random("overlapping", false, 2, 100000, 1u << 16);
    run_random("sparse overlapping", false, 3, 60000, 1u << 22);
    microbench();
    if (g_failures != 0) {
        printf("FAILED: %d check(s)\n", g_failures);
        return 1;
    }
    printf("ALL PASS\n");
    return 0;
}
