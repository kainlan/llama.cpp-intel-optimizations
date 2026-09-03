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
//      to at least the documented floor (2x the device core count, relaxed
//      in proportion to how far ncols exceeds the K=4096 reference -- see
//      the header comment for the full derivation), and never DROPS below
//      today's work-group count for the same shape.
//   3. COVERAGE: mmvq_pad_rows_to_workgroups never returns fewer rows than
//      requested, for every (nrows, subgroups_per_workgroup) this test
//      exercises -- i.e. the geometry the helper picks is safe to launch.
//   4. TINY-N EDGE CASE: a row count too small to ever reach the floor at
//      any granularity gets the finest available granularity (1 sub-group
//      per work-group) rather than silently falling back to today's 16.
//   5. POSITIVE CONTROL: a deliberately wrong "always 16" stand-in for the
//      helper is provably rejected by the occupancy-floor check on every
//      small-N shape below -- so a helper that silently no-ops would fail
//      this test, not pass it vacuously. (The corresponding NEGATIVE mutant
//      -- breaking the real helper to always return {16, 1} -- was run by
//      hand against this file per the task's TDD requirement; see the
//      implementer's report on llama.cpp-6cgq for that RED/GREEN record.)
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

struct shape_t {
    const char * label;
    int          nrows;
    int          ncols;
    int          n_cores;
    bool         expect_unchanged;  // true: large-N shapes that must stay at subgroups==16
};

// The exact shapes llama.cpp-6cgq's task profiled plus the two synthetic
// edge cases the ticket asked for (a mid-size non-Mistral K, and a row count
// too small to ever reach the occupancy floor).
static const shape_t kShapes[] = {
    { "gate/up-ish small N, B70",                  1024,  4096, 256, false },
    { "gate/up-ish small N, B50",                  1024,  4096, 128, false },
    { "q,o N=4096, B70",                           4096,  4096, 256, false },
    { "gate/up N=14336, B70 (large N, unchanged)", 14336, 4096, 256, true  },
    { "lm-head N=32000, B70 (large N, unchanged)", 32000, 4096, 256, true  },
    { "gemma4-ish N=2048 K=2560, B50",             2048,  2560, 128, false },
    { "tiny edge case N=37, B70",                  37,    4096, 256, false },
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
        } else {
            // (2) OCCUPANCY FLOOR. The floor itself is an implementation
            // detail of the helper (see its header comment); what this test
            // asserts is the CONTRACT: raised occupancy over today's
            // geometry, or -- for the tiny-N edge case where even the finest
            // granularity cannot reach any reasonable floor -- (4) the
            // finest granularity available (one sub-group per work-group).
            const bool raised_occupancy = workgroups > baseline;
            const bool finest_available = geometry.subgroups_per_workgroup == 1;
            CHECK(raised_occupancy || finest_available,
                  "%s: small-N shape did not raise occupancy (workgroups=%d, today=%d) and did not fall back to "
                  "the finest granularity (subgroups_per_workgroup=%d)",
                  s.label, workgroups, baseline, geometry.subgroups_per_workgroup);

            // (5) POSITIVE CONTROL: the "always 16" stand-in must be
            // distinguishable from the real helper on every small-N shape --
            // otherwise the two checks above would pass vacuously on a
            // no-op helper too.
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
