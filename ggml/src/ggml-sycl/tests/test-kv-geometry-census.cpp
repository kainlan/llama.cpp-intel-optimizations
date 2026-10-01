// Host tests of the production geometry census (llama.cpp-moua L4 1b): the
// whole-TLSF block census cut into kv_region_fit's frontier and side runs.
//
// Scored here:
//   C1  hand-computed geometry on scripted zones, from the layout alone, so a
//       reading that the zone model and the production builder share cannot
//       pass by being wrong the same way;
//   C2  agreement with tests/kv-region-test-model.hpp::snapshot() on scripted
//       and random zones: the frontier and every side run, block for block
//       (the model builds them from its own bookkeeping, the builder from the
//       allocator's block list);
//   C3  the yieldability predicate is asked about allocated optional tenants
//       only, once per tenant it reaches, never about a free block or another
//       tag (H7q);
//   C4  the edges: no anchor, an anchor with a weight directly below it, an
//       empty frontier, a zone with nothing buried.
//
// The build is -DNDEBUG, so CHECK is explicit and always runs.

#include "../kv-geometry-census.hpp"
#include "kv-region-test-model.hpp"

#include <cstdio>
#include <cstdlib>
#include <set>
#include <vector>

#define CHECK(cond, msg)                                                         \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

using namespace kv_region_test;
using ggml_sycl::kv_geometry_from_tlsf;

namespace {

bool same_block(const zone_block & a, const zone_block & b) {
    return a.offset == b.offset && a.size == b.size && a.free == b.free && a.optional_tenant == b.optional_tenant &&
           a.yieldable == b.yieldable;
}

bool same_blocks(const std::vector<zone_block> & a, const std::vector<zone_block> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (!same_block(a[i], b[i])) {
            return false;
        }
    }
    return true;
}

bool same_geometry(const tlsf_geometry & a, const tlsf_geometry & b) {
    if (!same_blocks(a.frontier, b.frontier) || a.side_runs.size() != b.side_runs.size()) {
        return false;
    }
    for (size_t i = 0; i < a.side_runs.size(); ++i) {
        if (!same_blocks(a.side_runs[i], b.side_runs[i])) {
            return false;
        }
    }
    return true;
}

tlsf_geometry census_of(const zone_model & z) {
    return kv_geometry_from_tlsf(z.allocator(), z.anchor(), [&](size_t off) { return z.optional_yieldable(off); });
}

constexpr size_t K = 1024;

// C1: W0 | O1 | hole | W1 | O2 | OL(leased) | O3 | W2 | O4 | OL2(leased) | ctx   (low to high)
// The frontier walk passes every optional tenant, leased or not (the fit truncates
// the ladder at the first one that is not yieldable, not the walk), and stops at W2.
void case_hand_computed_geometry() {
    zone_model   z(1024 * K);
    const size_t w0  = z.weight(64 * K);
    const size_t o1  = z.optional_tenant(8 * K);
    const size_t hp  = z.optional_tenant(4 * K);
    const size_t w1  = z.weight(16 * K);
    const size_t o2  = z.optional_tenant(8 * K);
    const size_t ol  = z.optional_tenant(8 * K, /*leased=*/true);
    const size_t o3  = z.optional_tenant(8 * K);
    const size_t w2  = z.weight(16 * K);
    const size_t o4  = z.optional_tenant(8 * K);
    const size_t ol2 = z.optional_tenant(8 * K, /*leased=*/true);
    const size_t cx  = z.context(256 * K);
    CHECK(w0 == 0 && o1 == 64 * K && hp == 72 * K && w1 == 76 * K && o2 == 92 * K && ol == 100 * K && o3 == 108 * K &&
              w2 == 116 * K && o4 == 132 * K && ol2 == 140 * K && cx != SIZE_MAX,
          "C1: layout is what the comment says");
    z.free(hp);

    const tlsf_geometry g = census_of(z);

    CHECK(g.frontier.size() == 3, "C1: the frontier is the gap, ol2 and o4");
    CHECK(g.frontier[0].free && g.frontier[0].offset == 148 * K && g.frontier[0].size == cx - 148 * K,
          "C1: the top rung is the gap below the context side");
    CHECK(g.frontier[1].offset == ol2 && g.frontier[1].optional_tenant && !g.frontier[1].yieldable,
          "C1: the leased tenant is a rung the walk passes, marked not yieldable");
    CHECK(g.frontier[2].offset == o4 && g.frontier[2].optional_tenant && g.frontier[2].yieldable,
          "C1: o4 is a yieldable optional rung");

    // Below the frontier floor (o4): the weights and the leased OL break runs.
    CHECK(g.side_runs.size() == 3, "C1: three runs below the frontier");
    CHECK(g.side_runs[0].size() == 2 && g.side_runs[0][0].offset == o1 && g.side_runs[0][0].yieldable &&
              g.side_runs[0][1].offset == hp && g.side_runs[0][1].free && g.side_runs[0][1].size == 4 * K,
          "C1: run A is the buried o1 and the weight hole, between w0 and w1");
    CHECK(g.side_runs[1].size() == 1 && g.side_runs[1][0].offset == o2 && g.side_runs[1][0].yieldable,
          "C1: run B is o2 alone, ended by the leased ol");
    CHECK(g.side_runs[2].size() == 1 && g.side_runs[2][0].offset == o3 && g.side_runs[2][0].yieldable,
          "C1: run C is o3 alone, between ol and w2");
    std::printf("case_hand_computed_geometry: PASSED\n");
}

// C2, scripted: the shapes the fit's own cases build.
void case_agreement_scripted() {
    {
        zone_model z(1024 * K);  // empty zone, no anchor
        CHECK(same_geometry(z.snapshot(), census_of(z)), "C2: an empty zone agrees");
    }
    {
        zone_model z(1024 * K);  // context side only
        (void) z.context(128 * K);
        (void) z.context(64 * K);
        CHECK(same_geometry(z.snapshot(), census_of(z)), "C2: a context-only zone agrees");
    }
    {
        zone_model z(1024 * K);  // weights only, nothing buried
        (void) z.weight(64 * K);
        (void) z.weight(32 * K);
        CHECK(same_geometry(z.snapshot(), census_of(z)), "C2: a weights-only zone agrees");
    }
    {
        zone_model z(1024 * K);  // a weight directly below the anchor: empty frontier past the gap
        (void) z.optional_tenant(8 * K);
        (void) z.weight(8 * K);
        const size_t cx = z.context(900 * K);
        CHECK(cx != SIZE_MAX, "C2: layout");
        const tlsf_geometry g = census_of(z);
        CHECK(same_geometry(z.snapshot(), g), "C2: a weight under the gap agrees");
        CHECK(g.frontier.size() == 1 && g.frontier[0].free, "C4: the frontier is the gap alone under a weight");
        CHECK(g.side_runs.size() == 1 && g.side_runs[0].size() == 1 && g.side_runs[0][0].optional_tenant,
              "C4: the optional below the weight is buried, in its own run");
    }
    std::printf("case_agreement_scripted: PASSED\n");
}

// C2, random: every placement kind, frees anywhere, leased and unleased
// tenants, with the context side growing and shrinking.
void case_agreement_random() {
    uint32_t seed = 4242u;
    auto     rnd  = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return seed >> 8;
    };
    size_t n_runs   = 0;
    size_t n_buried = 0;
    size_t n_holes  = 0;
    size_t n_leased = 0;
    for (int zone = 0; zone < 60; ++zone) {
        zone_model          z(2048 * K);
        std::vector<size_t> live;
        for (int step = 0; step < 120; ++step) {
            const uint32_t op = rnd() % 10;
            if (op < 7 || live.empty()) {
                const size_t size = 256 * (1 + rnd() % 40);
                size_t       off  = SIZE_MAX;
                switch (op % 4) {
                    case 0:
                        off = z.weight(size);
                        break;
                    case 1:
                        off = z.optional_tenant(size, rnd() % 4 == 0);
                        break;
                    case 2:
                        off = z.optional_tenant(size);
                        break;
                    default:
                        off = z.context(size);
                        break;
                }
                if (off != SIZE_MAX) {
                    live.push_back(off);
                }
            } else {
                // The anchor is always a live allocated block in production; the model
                // keeps a freed anchor's stale offset, which is not a state to compare.
                const size_t i = rnd() % live.size();
                if (live[i] == z.anchor()) {
                    continue;
                }
                z.free(live[i]);
                live[i] = live.back();
                live.pop_back();
            }
            const tlsf_geometry want = z.snapshot();
            const tlsf_geometry got  = census_of(z);
            CHECK(same_geometry(want, got), "C2: the census agrees with the model's snapshot");
            CHECK(z.invariants(), "C2: the zone stays valid");
            if (step % 20 == 0) {
                n_runs += got.side_runs.size();
                for (const auto & run : got.side_runs) {
                    for (const zone_block & b : run) {
                        n_buried += b.optional_tenant ? 1 : 0;
                        n_holes += b.free ? 1 : 0;
                    }
                }
                for (const zone_block & b : got.frontier) {
                    n_leased += (b.optional_tenant && !b.yieldable) ? 1 : 0;
                }
            }
        }
    }
    CHECK(n_runs > 20 && n_buried > 10 && n_holes > 10 && n_leased > 0,
          "C2: the random zones produced side runs, buried optionals, holes and leased rungs (not vacuous)");
    std::printf("case_agreement_random: PASSED (runs %zu, buried %zu, holes %zu)\n", n_runs, n_buried, n_holes);
}

// C3: the predicate sees allocated optional tenants only, each at most once.
void case_predicate_scope() {
    zone_model z(1024 * K);
    (void) z.weight(32 * K);
    const size_t o1 = z.optional_tenant(8 * K);
    const size_t h  = z.optional_tenant(4 * K);
    (void) z.weight(16 * K);
    const size_t o2 = z.optional_tenant(8 * K, true);
    const size_t o3 = z.optional_tenant(8 * K);
    (void) z.context(64 * K);
    z.free(h);

    std::vector<size_t>    asked;
    const tlsf_geometry    g = kv_geometry_from_tlsf(z.allocator(), z.anchor(), [&](size_t off) {
        asked.push_back(off);
        return z.optional_yieldable(off);
    });
    const std::set<size_t> distinct(asked.begin(), asked.end());
    CHECK(distinct.size() == asked.size(), "C3: no tenant is asked about twice");
    CHECK(distinct == (std::set<size_t>{ o1, o2, o3 }), "C3: exactly the allocated optional tenants are asked");
    CHECK(same_geometry(g, z.snapshot()), "C3: the counting predicate changes nothing");
    std::printf("case_predicate_scope: PASSED\n");
}

// C5: the run cutter is total over any block list.  A hand-built census can end in
// a releasable block under the floor, which an allocator never produces (the walk
// takes those into the frontier); the open run must still be reported, and a block
// that ends exactly at the floor is below it.
void case_run_cutter_is_total() {
    using ext                     = tlsf_allocator::extent;
    const std::vector<ext> census = {
        { 0,      4 * K, false, ggml_sycl::SHARED_ZONE_TAG_WEIGHT   },
        { 4 * K,  4 * K, true,  0                                   },
        { 8 * K,  4 * K, false, ggml_sycl::SHARED_ZONE_TAG_OPTIONAL },
        { 12 * K, 4 * K, true,  0                                   },
    };
    const auto yes = [](size_t) {
        return true;
    };
    const auto runs = ggml_sycl::kv_side_runs_from_census(census, 16 * K, yes);
    CHECK(runs.size() == 1 && runs[0].size() == 3, "C5: the run open at the floor is reported whole");
    CHECK(runs[0][2].offset == 12 * K && runs[0][2].free, "C5: the block ending exactly at the floor is in the run");

    const auto none = [](size_t) {
        return false;
    };
    const auto cut = ggml_sycl::kv_side_runs_from_census(census, 16 * K, none);
    CHECK(cut.size() == 2 && cut[0].size() == 1 && cut[1].size() == 1,
          "C5: a refused optional tenant splits the run it sits in");

    CHECK(ggml_sycl::kv_side_runs_from_census(census, 0, yes).empty(), "C5: nothing lies below a floor of 0");
    CHECK(ggml_sycl::kv_side_runs_from_census({}, SIZE_MAX, yes).empty(), "C5: an empty census has no runs");
    std::printf("case_run_cutter_is_total: PASSED\n");
}

}  // namespace

int main() {
    case_hand_computed_geometry();
    case_agreement_scripted();
    case_agreement_random();
    case_predicate_scope();
    case_run_cutter_is_total();
    return 0;
}
