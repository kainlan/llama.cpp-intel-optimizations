// Host-only unit test for the Q8_0 SOA MMVQ small-N occupancy geometry
// (ggml/src/ggml-sycl/mmvq-launch-geometry.hpp:ggml_sycl_mmvq_q8_0_soa_geometry,
// llama.cpp-6cgq).
//
// What is checked, against the current production dispatch code in
// ggml/src/ggml-sycl/mmvq.cpp (reorder_mul_mat_vec_q8_0_q8_1_sycl):
//   1. REGRESSION ANCHOR: for the large-N shapes profiled on llama.cpp-6cgq
//      (already at or near the bandwidth ceiling), the helper returns
//      subgroups_per_workgroup == 16 -- byte-identical to today's fixed
//      geometry -- and the resulting work-group count matches what the
//      unmodified dispatch code computes for those shapes.
//   2. OCCUPANCY FLOOR: for the small-N shapes profiled with a measured
//      efficiency deficit, the helper raises the launched work-group count
//      to AT LEAST the documented floor -- round(2 * n_cores * min(1,
//      4096/ncols)), independently re-derived in expected_occupancy_floor()
//      below from the header's own comment, not by calling into the header
//      -- and never DROPS below today's work-group count for the same
//      shape. This is a floor check, not merely "more than today": a helper
//      that always raises occupancy a LITTLE (e.g. a fixed subgroups=8
//      regardless of nrows/n_cores) would pass a "more than today" check on
//      every shape below while still missing the actual target.
//   3. COVERAGE: mmvq_pad_rows_to_workgroups never returns fewer rows than
//      requested, for every (nrows, subgroups_per_workgroup) this test
//      exercises -- i.e. the geometry the helper picks is safe to launch.
//   4. TINY-N EDGE CASE: for a shape where NO granularity (down to
//      subgroups_per_workgroup=1) can reach the floor, the helper falls back
//      to the finest available granularity (maximum achievable work-groups)
//      rather than silently keeping today's 16.
//   5. POSITIVE CONTROL (occupancy floor): a deliberately wrong "always 16"
//      stand-in for the helper is provably rejected on every small-N shape
//      below -- so a helper that silently no-ops would fail this test, not
//      pass it vacuously. (The corresponding NEGATIVE mutant -- breaking the
//      real helper to always return {16, 1} -- was run by hand against this
//      file per the task's TDD requirement; see the implementer's report on
//      llama.cpp-6cgq for that RED/GREEN record.)
//   6. POSITIVE CONTROL (K-relief term): the wide-K "down" projection shape
//      (N=4096, K=14336 -- same N as the "q,o" shape above it, which the
//      relief term must NOT spare) is asserted unchanged, AND a mutant that
//      drops the relief term entirely (geometry_without_k_relief) is
//      confirmed to land on a DIFFERENT, over-aggressive geometry there --
//      so the unchanged-check on that shape actually exercises the relief
//      term, not just its absence of effect on shapes the term never
//      touches. Run by hand as a full-implementation swap (production
//      geometry function's k_relief hardcoded to 1.0): the whole test suite
//      goes RED on this shape's unchanged-check, moving it from 256 to 512
//      work-groups; see the implementer's report for that record.
//
// Pure C++: no SYCL, no device, no ggml link -- mmvq-launch-geometry.hpp is
// deliberately dependency-free for exactly this reason. Exit 1 on any
// failure.

#include "ggml-sycl/mmvq-launch-geometry.hpp"

#include <cstdio>

static int failures = 0;

#define CHECK(cond, ...)                                     \
    do {                                                     \
        if (!(cond)) {                                       \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                        \
            std::printf("\n");                               \
            ++failures;                                      \
        }                                                    \
    } while (0)

// What ggml-sycl/mmvq.cpp's UNMODIFIED (today's) dispatch code computes:
// GGML_SYCL_MMV_Y is always 1 (presets.hpp), num_subgroups is the fixed 16,
// so block_num_y == nrows and the launched work-group count is
// mmvq_pad_rows_to_workgroups(nrows, 16) / 16.
static int today_workgroups(int nrows) {
    const int padded = ggml_sycl::mmvq_pad_rows_to_workgroups(nrows, 16);
    return padded / 16;
}

static int workgroups_for(int nrows, int subgroups_per_workgroup) {
    if (subgroups_per_workgroup <= 0) {
        return 0;
    }
    const int padded = ggml_sycl::mmvq_pad_rows_to_workgroups(nrows, subgroups_per_workgroup);
    return padded / subgroups_per_workgroup;
}

// A deliberately wrong stand-in for the helper: always keeps today's
// geometry, exactly as if the small-N fix were never applied. Used as a
// positive control below -- the occupancy-floor check must reject it on
// every small-N shape, or the check itself has no teeth.
static ggml_sycl::mmvq_q8_0_soa_geometry_t always_16(int /*nrows*/, int /*ncols*/, int /*n_cores*/) {
    return { 16, 1 };
}

// Independent re-derivation of the DOCUMENTED floor formula from
// mmvq-launch-geometry.hpp's ggml_sycl_mmvq_q8_0_soa_geometry comment:
//   floor = round(2 * n_cores * min(1, 4096/ncols))
// Written from the doc, not by calling the header, so this test is scoring
// the implementation against its published contract rather than against
// itself (score-the-exact-answer-against-the-oracle).
static int expected_occupancy_floor(int ncols, int n_cores) {
    constexpr int kReferenceCols       = 4096;
    constexpr int kOccupancyMultiplier = 2;
    const double  k_relief             = (ncols > kReferenceCols) ? (double) kReferenceCols / (double) ncols : 1.0;
    int           floor_wgs            = (int) (kOccupancyMultiplier * n_cores * k_relief + 0.5);
    return floor_wgs < 1 ? 1 : floor_wgs;
}

// A mutant that drops the K-relief term entirely (as if ncols were always
// <= the K=4096 reference, i.e. k_relief hardcoded to 1.0). Used as a
// positive control for the K-relief term specifically: it must diverge from
// the real helper on a wide-K shape (see the down-projection shape below),
// or the "large-N unchanged" check on that shape has no teeth against a
// broken/dropped relief term.
static ggml_sycl::mmvq_q8_0_soa_geometry_t geometry_without_k_relief(int nrows, int /*ncols*/, int n_cores) {
    if (nrows <= 0 || n_cores <= 0) {
        return { 16, 1 };
    }
    int floor_wgs = (int) (2.0 * n_cores + 0.5);  // k_relief hardcoded to 1.0
    if (floor_wgs < 1) {
        floor_wgs = 1;
    }
    for (int subgroups = 16; subgroups >= 1; subgroups /= 2) {
        const int padded     = ggml_sycl::mmvq_pad_rows_to_workgroups(nrows, subgroups);
        const int workgroups = padded / subgroups;
        if (workgroups >= floor_wgs) {
            return { subgroups, 1 };
        }
    }
    return { 1, 1 };
}

struct shape_t {
    const char * label;
    int          nrows;
    int          ncols;
    int          n_cores;
    bool         expect_unchanged;  // true: large-N (or wide-K) shapes that must stay at subgroups==16
    // Hardcoded LITERAL work-group count for expect_unchanged shapes only (0
    // otherwise, unused): a regression anchor that does NOT re-derive its
    // expectation from mmvq_pad_rows_to_workgroups, so a bug that broke both
    // that helper AND this test in the same direction could not slip through
    // by drifting them together (spec review round 1, finding 8). This is
    // ADDITIONAL to, not a replacement for, the `workgroups == baseline`
    // check below (baseline is still useful as a live cross-check against
    // today's actual dispatch code).
    int          expected_workgroups_literal;
};

// The exact shapes llama.cpp-6cgq's task profiled plus the synthetic edge
// cases the ticket and the spec review asked for: a mid-size non-Mistral K,
// a row count too small to ever reach the occupancy floor, and the wide-K
// "down" projection shape that exercises the K-relief term (K=14336 > the
// 4096 reference, at the SAME nrows=4096 as the "q,o" shape above it, which
// K-relief must NOT spare).
static const shape_t kShapes[] = {
    { "gate/up-ish small N, B70",                     1024,  4096,  256, false, 0    },
    { "gate/up-ish small N, B50",                     1024,  4096,  128, false, 0    },
    { "q,o N=4096, B70",                              4096,  4096,  256, false, 0    },
    { "gate/up N=14336, B70 (large N, unchanged)",    14336, 4096,  256, true,  896  },
    { "lm-head N=32000, B70 (large N, unchanged)",    32000, 4096,  256, true,  2000 },
    { "gemma4-ish N=2048 K=2560, B50",                2048,  2560,  128, false, 0    },
    { "tiny edge case N=37, B70",                     37,    4096,  256, false, 0    },
    { "down N=4096 K=14336, B70 (wide-K, unchanged)", 4096,  14336, 256, true,  256  },
};

int main() {
    for (const shape_t & s : kShapes) {
        const auto geometry   = ggml_sycl::ggml_sycl_mmvq_q8_0_soa_geometry(s.nrows, s.ncols, s.n_cores);
        const int  padded     = ggml_sycl::mmvq_pad_rows_to_workgroups(s.nrows, geometry.subgroups_per_workgroup);
        const int  workgroups = workgroups_for(s.nrows, geometry.subgroups_per_workgroup);
        const int  baseline   = today_workgroups(s.nrows);

        // (3) COVERAGE: padding never drops a requested row, whatever
        // granularity the helper picked.
        CHECK(ggml_sycl::mmvq_launch_covers_rows(padded, s.nrows), "%s: padded rows %d < requested %d", s.label, padded,
              s.nrows);
        CHECK(ggml_sycl::mmvq_launch_is_uniform(padded, geometry.subgroups_per_workgroup),
              "%s: padded rows %d not a whole multiple of subgroups_per_workgroup %d", s.label, padded,
              geometry.subgroups_per_workgroup);

        // The helper must never choose a geometry that launches FEWER
        // work-groups than today's fixed-16 code for the same shape --
        // that would be a regression, not an improvement.
        CHECK(workgroups >= baseline,
              "%s: helper launches %d work-groups, fewer than today's %d (nrows=%d) -- regression", s.label, workgroups,
              baseline, s.nrows);

        if (s.expect_unchanged) {
            // (1) REGRESSION ANCHOR.
            CHECK(geometry.subgroups_per_workgroup == 16,
                  "%s: large-N geometry changed -- subgroups_per_workgroup=%d, expected unchanged 16", s.label,
                  geometry.subgroups_per_workgroup);
            CHECK(workgroups == baseline, "%s: large-N work-group count changed -- got %d, today's dispatch is %d",
                  s.label, workgroups, baseline);
            CHECK(workgroups == s.expected_workgroups_literal,
                  "%s: large-N work-group count %d does not match the hardcoded regression-anchor literal %d", s.label,
                  workgroups, s.expected_workgroups_literal);

            // K-RELIEF POSITIVE CONTROL (only meaningful where ncols exceeds
            // the K=4096 reference, i.e. the wide-K "down" shape): a mutant
            // that drops the relief term entirely must land on a DIFFERENT,
            // over-aggressive geometry here -- confirming this shape's
            // unchanged-check actually exercises the relief term rather than
            // passing regardless of it.
            if (s.ncols > 4096) {
                const auto mutant_geometry   = geometry_without_k_relief(s.nrows, s.ncols, s.n_cores);
                const int  mutant_workgroups = workgroups_for(s.nrows, mutant_geometry.subgroups_per_workgroup);
                CHECK(mutant_workgroups != workgroups,
                      "%s: K-relief positive control void -- a relief-blind mutant matches the real helper (both %d "
                      "workgroups); the unchanged-check on this shape would not catch a dropped relief term",
                      s.label, workgroups);
            }
        } else {
            // (2) OCCUPANCY FLOOR: assert the ACTUAL documented floor, not
            // merely "more than today" -- a helper that under-shoots the
            // floor (e.g. always picking subgroups=8 regardless of nrows and
            // n_cores) still satisfies "raised over today" on these shapes
            // while launching far fewer work-groups than the floor requires.
            const int  floor_wgs         = expected_occupancy_floor(s.ncols, s.n_cores);
            const int  finest_workgroups = workgroups_for(s.nrows, 1);
            const bool floor_unreachable = finest_workgroups < floor_wgs;
            const bool finest_available  = geometry.subgroups_per_workgroup == 1;
            if (floor_unreachable) {
                // (4) TINY-N EDGE CASE: no granularity can reach the floor --
                // the only acceptable behaviour is the finest granularity
                // (maximum achievable work-groups), not a silent fallback to
                // today's 16.
                CHECK(finest_available,
                      "%s: floor %d is unreachable (finest granularity gives only %d work-groups), so the helper "
                      "must fall back to subgroups_per_workgroup=1, got %d",
                      s.label, floor_wgs, finest_workgroups, geometry.subgroups_per_workgroup);
                CHECK(workgroups == finest_workgroups,
                      "%s: floor unreachable -- expected the maximum achievable %d work-groups, got %d", s.label,
                      finest_workgroups, workgroups);
            } else {
                CHECK(workgroups >= floor_wgs, "%s: work-groups %d below the documented floor %d (n_cores=%d ncols=%d)",
                      s.label, workgroups, floor_wgs, s.n_cores, s.ncols);
            }

            // (5) POSITIVE CONTROL: the "always 16" stand-in must be
            // distinguishable from the real helper on every small-N shape --
            // otherwise the checks above would pass vacuously on a no-op
            // helper too.
            const auto mutant_geometry   = always_16(s.nrows, s.ncols, s.n_cores);
            const int  mutant_workgroups = workgroups_for(s.nrows, mutant_geometry.subgroups_per_workgroup);
            CHECK(mutant_workgroups == baseline, "%s: positive control setup wrong: mutant workgroups=%d != %d",
                  s.label, mutant_workgroups, baseline);
            CHECK(workgroups != mutant_workgroups || finest_available,
                  "%s: positive control void -- real helper matches the always-16 mutant on a small-N shape", s.label);
        }

        std::printf("%s: nrows=%d ncols=%d n_cores=%d -> subgroups_per_workgroup=%d workgroups=%d (today=%d)\n",
                    s.label, s.nrows, s.ncols, s.n_cores, geometry.subgroups_per_workgroup, workgroups, baseline);
    }

    // Non-positive inputs must not crash and must not invent a geometry --
    // they fall back to today's fixed value.
    {
        const auto g0 = ggml_sycl::ggml_sycl_mmvq_q8_0_soa_geometry(0, 4096, 256);
        CHECK(g0.subgroups_per_workgroup == 16, "nrows=0: expected fallback subgroups_per_workgroup=16, got %d",
              g0.subgroups_per_workgroup);
        const auto g1 = ggml_sycl::ggml_sycl_mmvq_q8_0_soa_geometry(1024, 4096, 0);
        CHECK(g1.subgroups_per_workgroup == 16, "n_cores=0: expected fallback subgroups_per_workgroup=16, got %d",
              g1.subgroups_per_workgroup);
    }

    if (failures) {
        std::printf("FAILED: %d check(s)\n", failures);
        return 1;
    }
    std::printf("PASS: q8_0 SOA MMVQ small-N occupancy geometry\n");
    return 0;
}
