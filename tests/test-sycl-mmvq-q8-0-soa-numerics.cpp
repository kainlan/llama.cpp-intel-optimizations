// GPU numerics test for the Q8_0 SOA MMVQ fast path (llama.cpp-6cgq):
// reorder_mul_mat_vec_q8_0_q8_1_sycl's fast path (mmvq.cpp, gated by
// GGML_SYCL_MMVQ_Q8_SOA_FASTPATH, default ON -- D-plane SLM staging plus the
// occupancy geometry from mmvq-launch-geometry.hpp, applied together) must
// produce output that agrees with a CPU reference AND with the COALESCED
// kernel's output for the same weights and input.
//
// This is intentionally NOT run by the implementer (no GPU access in this
// role, per the fork's Hard-Won Rules); it is written for `main` to build
// and run. ctest registers this binary TWICE (tests/CMakeLists.txt) --
// default ON, and again with GGML_SYCL_MMVQ_Q8_SOA_FASTPATH=0 under the name
// test-sycl-mmvq-q8-0-soa-smalln-numerics-off -- since the env accessor
// caches its value once per process (mmvq_q8_0_soa_fastpath_enabled() in
// mmvq.cpp) and ctest runs each registration in its own process. Direct
// invocation covers the same two arms:
//
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-sycl-mmvq-q8-0-soa-smalln-numerics
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 GGML_SYCL_MMVQ_Q8_SOA_FASTPATH=0 \
//       ./build/bin/test-sycl-mmvq-q8-0-soa-smalln-numerics
//
// Strategy per shape: build ONE weight tensor's worth of random Q8_0 blocks
// and ONE input tensor's worth of random F32 activations, materialize the
// SAME bytes as two SEPARATE device weight tensors -- one forced SOA, one
// forced COALESCED, via the ggml-sycl-private-fixtures layout-override hook
// (the same mechanism tests/test-q8-0-layout-cache-path-mmvq.cpp uses to
// force SOA caching) -- run MUL_MAT on each, and compare:
//   (a) SOA GPU output vs a CPU dequant+dot reference,
//   (b) COALESCED GPU output vs the same CPU reference,
//   (c) SOA GPU output vs COALESCED GPU output directly.
// (a)+(b) catch either kernel drifting from correct; (c) is the direct
// "SOA vs COALESCED" comparison the ticket asked for, and is the tightest of
// the three since both read literally the same weight bytes.
//
// CPU REFERENCE CORRECTNESS (spec review round 2, finding 2 -- a real bug,
// not a nitpick): MMVQ computes Q8_0(weight) . Q8_1(activation), NOT
// Q8_0(weight) . F32(activation). An earlier version of this file dot-
// producted the weight's dequantized int8 values directly against the RAW
// F32 input, skipping the activation-side quantization the production
// kernel actually performs -- and was caught exactly the way a broken
// reference should be caught: the UNCHANGED, pre-existing coalesced kernel
// failed it by nearly the same magnitude as SOA (max_rel ~9x and ~5x), while
// the real model emits correct tokens through both paths, so the reference
// was self-evidently the thing that was wrong, not either kernel.
// quantize_q8_1_ref() below is transcribed from ggml_quantize_row_q8_1_ref's
// actual CPU implementation (ggml/src/ggml-quants.c:302 at the time this was
// written -- amax/127 scale, round() quants, one block of 32 per scale,
// scale round-tripped through fp16 since block_q8_1.d is stored fp16 on
// both host and device) rather than re-derived from memory, and is applied
// to each batch column of the F32 input before the dot product, matching
// what the production dispatch's activation-quantization kernel does ahead
// of every MMVQ call.
//
// BRANCH ENGAGEMENT / GEOMETRY DECISION (spec review round 1, finding 4,
// updated for round 2's redesign): the fast path's SLM kernel is now
// dispatched whenever GGML_SYCL_MMVQ_Q8_SOA_FASTPATH is on, UNCONDITIONALLY
// -- not merely when the occupancy geometry differs from today's fixed 16
// (round 1's design; the D-plane fix helps at every N, so gating it on the
// geometry decision would have left large-N shapes on the unfixed kernel for
// no reason). So "did the fast-path KERNEL run" is now a simple per-process
// fact (mirrors fastpath_env_enabled() below), not a per-shape question --
// but "which GEOMETRY did it use for this shape" still is, and is still
// worth checking independently for the same reason round 1 gave: correct
// numerics prove nothing about which launch geometry produced them, and a
// geometry helper that silently always returns {16,1} (e.g. from a broken
// env read) would still pass every numerics comparison below while
// exercising nothing this ticket's geometry lever added.
// check_geometry_decision() below computes what the production dispatch's
// geometry decision WOULD be for this shape on THIS PROCESS's actual device
// (real core count via ggml_sycl_info(), read once the backend is
// initialized) using the same pure geometry helper the production dispatch
// calls, and compares it against an INDEPENDENT expectation re-derived from
// the header's documented floor formula (not by calling the header's own
// floor logic -- there isn't a separate accessor for it -- but by
// re-deriving round(2*n_cores*min(1,4096/ncols)) from its prose, the same
// re-derivation the host-only geometry test uses): a shape must show
// subgroups_per_workgroup != 16 exactly when today's fixed-16 launch would
// fall below that floor on THIS device. This is device-adaptive on purpose
// -- whether a given shape crosses the floor depends on the actual core
// count of whichever card ONEAPI_DEVICE_SELECTOR points at (256 on the B70,
// 128 on the B50), so a shape near the boundary is expected to cross on one
// card and not the other; the assertion follows the real device rather than
// a hardcoded per-shape table that would go stale or wrong on a different
// card. A geometry helper that silently never changes anything FAILS here
// with an explicit "geometry decision mismatch ... expected ... but the
// helper picked subgroups_per_workgroup=16" message and a non-zero exit,
// distinguishable from a numerics failure (which names the comparison that
// diverged instead). The `subgroups=` value printed here is the same one
// the fast path's profile_label metadata carries (mmvq.cpp), so a
// GGML_SYCL_KERNEL_PROFILE=1 capture can be cross-checked against this
// test's own prediction by hand.
//
// PRODUCTION DISPATCH OBSERVATION (spec review round 2, finding 1 -- C4
// still open from round 1): check_geometry_decision() above only reasons
// about what the geometry helper WOULD return; it re-reads the env var and
// calls the same pure function the production code calls, entirely from
// this test's own code, so it observes nothing about whether the actual
// PRODUCTION dispatch in mmvq.cpp ever ran. Because the fast-path kernel is
// bit-identical to the original by construction (same QS reads, same dp4a
// math -- only the source of `d` differs), every numerics comparison in
// this file is INVARIANT to which kernel actually executed: a
// mmvq_q8_0_soa_fastpath_enabled() that always returned false (e.g. from a
// broken env read, or the two-registration ctest wiring somehow losing the
// FASTPATH=0 arm's environment) would still pass every OK line below on
// BOTH ctest arms, because the untouched original kernel produces the same
// numbers. run_soa_with_dispatch_observation() closes this gap by observing
// a REAL artifact of the production dispatch: it force-enables the kernel
// profiler for the test process (ggml_sycl_kernel_profile_set_config_for_test,
// GGML_SYCL_PRIVATE_TESTING-gated, sycl-kernel-profiler.hpp -- this binary
// already links ggml-sycl-private-fixtures with that macro defined), runs
// the SOA layout, flushes, and inspects the recorded CSV rows directly.
// Both mmvq.cpp branches label their profile_label "mulmat.mmvq.q8_0_soa"
// (deliberate profiler continuity across this ticket), so the label name
// alone cannot distinguish them; the metadata string can: only the fast
// path's includes ";subgroups=" (mmvq.cpp's fast-path branch builds
// "ncols=...;nrows=...;subgroups=..."), while the unchanged original
// kernel's is "ncols=...;nrows=..." with no such field, since `metadata` is
// part of the profiler's aggregation key (sycl-kernel-profiler.cpp's
// profile_key), so the two branches' records never collide or get merged.
// A run where the fast path silently never dispatches FAILS with an
// explicit "production dispatch observation mismatch ... expected a
// mulmat.mmvq.q8_0_soa profiler record WITH/WITHOUT a 'subgroups=' metadata
// field, but ..." message, distinguishable from both a numerics failure
// (names the diverging comparison) and a geometry-decision mismatch (names
// the expected vs actual subgroups_per_workgroup).

#include "common.hpp"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml-sycl/ggml-sycl-test.hpp"
#include "ggml-sycl/mmvq-launch-geometry.hpp"
#include "ggml-sycl/sycl-kernel-profiler.hpp"
#include "ggml.h"
#include "test-skip.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define QK8_0 32
#define QK8_1 32  // Q8_1 activation blocks are also 32 elements, same as Q8_0.

typedef struct {
    ggml_fp16_t d;
    int8_t      qs[QK8_0];
} block_q8_0_test;

static_assert(sizeof(block_q8_0_test) == 34, "block_q8_0 size mismatch");

static int g_failures = 0;

static ggml_backend_buffer_t alloc_tensor_buffer(ggml_backend_buffer_type_t buft,
                                                 ggml_tensor *              tensor,
                                                 ggml_backend_buffer_usage  usage) {
    const size_t          size   = ggml_backend_buft_get_alloc_size(buft, tensor);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, size);
    if (!buffer) {
        return nullptr;
    }
    ggml_backend_buffer_set_usage(buffer, usage);
    ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer));
    return buffer;
}

static void fill_q8_0_random(block_q8_0_test * blocks, int nblocks, std::mt19937 & rng) {
    std::uniform_int_distribution<int>    qdist(-120, 120);
    std::uniform_real_distribution<float> ddist(0.001f, 0.05f);
    for (int i = 0; i < nblocks; ++i) {
        blocks[i].d = ggml_fp32_to_fp16(ddist(rng));
        for (int j = 0; j < QK8_0; ++j) {
            blocks[i].qs[j] = static_cast<int8_t>(qdist(rng));
        }
    }
}

// Transcribed from quantize_row_q8_1_ref (ggml/src/ggml-quants.c) -- see the
// file header comment for why this must match that function exactly rather
// than being approximated.
static void quantize_q8_1_ref(const float * x, int n, std::vector<int8_t> & qs_out, std::vector<float> & d_out) {
    const int nb = n / QK8_1;
    qs_out.resize((size_t) n);
    d_out.resize((size_t) nb);
    for (int i = 0; i < nb; ++i) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_1; ++j) {
            amax = std::max(amax, std::fabs(x[(size_t) i * QK8_1 + j]));
        }
        const float d  = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        for (int j = 0; j < QK8_1; ++j) {
            qs_out[(size_t) i * QK8_1 + j] = (int8_t) std::lround((double) (x[(size_t) i * QK8_1 + j] * id));
        }
        // block_q8_1.d is stored fp16 on both host and device; round-trip so
        // the reference uses the same truncated scale the GPU kernel reads.
        d_out[(size_t) i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(d));
    }
}

static void compute_reference(const block_q8_0_test * weights,
                              const float *           input,
                              float *                 output,
                              int                     ncols,
                              int                     nrows,
                              int                     batch) {
    const int blocks_per_row = ncols / QK8_0;

    std::vector<int8_t> act_qs;
    std::vector<float>  act_d;
    for (int b = 0; b < batch; ++b) {
        const float * input_col = input + (size_t) b * ncols;
        quantize_q8_1_ref(input_col, ncols, act_qs, act_d);

        float * out_col = output + (size_t) b * nrows;
        for (int row = 0; row < nrows; ++row) {
            double                  sum        = 0.0;
            const block_q8_0_test * row_blocks = weights + (size_t) row * blocks_per_row;
            for (int blk = 0; blk < blocks_per_row; ++blk) {
                const block_q8_0_test * w   = row_blocks + blk;
                const float             wd  = ggml_fp16_to_fp32(w->d);
                const float             ad  = act_d[(size_t) blk];
                int32_t                 dot = 0;
                for (int j = 0; j < QK8_0; ++j) {
                    dot += (int32_t) w->qs[j] * (int32_t) act_qs[(size_t) blk * QK8_0 + j];
                }
                sum += (double) wd * (double) ad * (double) dot;
            }
            out_col[row] = (float) sum;
        }
    }
}

// Runs one MUL_MAT under the given forced layout and returns the GPU output
// (empty on failure, with a printed reason). weight_data/input_data are the
// SAME bytes for every call in a shape so SOA and COALESCED read identical
// weights.
static std::vector<float> run_layout(ggml_backend_t                       backend,
                                     ggml_layout_mode                     layout,
                                     const char *                         layout_name,
                                     int                                  ncols,
                                     int                                  nrows,
                                     int                                  batch,
                                     const std::vector<block_q8_0_test> & weight_data,
                                     const std::vector<float> &           input_data) {
    std::vector<float> empty;

    ggml_init_params params = { 16 * 1024 * 1024, nullptr, true };
    ggml_context *   ctx    = ggml_init(params);
    if (!ctx) {
        std::printf("FAIL [%s]: ggml_init failed\n", layout_name);
        ++g_failures;
        return empty;
    }

    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, ncols, nrows);
    ggml_set_name(weight, "mmvq_q8_0_soa_fastpath_weight");
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, batch);
    ggml_set_name(input, "mmvq_q8_0_soa_fastpath_input");
    ggml_tensor * output = ggml_mul_mat(ctx, weight, input);

    ggml_backend_buffer_type_t weight_buft = ggml_backend_sycl_host_buffer_type();
    ggml_backend_buffer_type_t dev_buft    = ggml_backend_get_default_buffer_type(backend);

    ggml_backend_buffer_t weight_buf = alloc_tensor_buffer(weight_buft, weight, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_t input_buf  = alloc_tensor_buffer(dev_buft, input, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_buffer_t output_buf = alloc_tensor_buffer(dev_buft, output, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    auto cleanup = [&](ggml_sycl_model_token * model) {
        if (weight_buf) {
            ggml_backend_buffer_free(weight_buf);
        }
        if (input_buf) {
            ggml_backend_buffer_free(input_buf);
        }
        if (output_buf) {
            ggml_backend_buffer_free(output_buf);
        }
        ggml_free(ctx);
        if (model) {
            (void) ggml_backend_sycl_model_unloaded_token(*model);
        }
    };

    if (!weight_buf || !input_buf || !output_buf) {
        std::printf("FAIL [%s]: buffer allocation failed\n", layout_name);
        ++g_failures;
        cleanup(nullptr);
        return empty;
    }

    // See test-q8-0-layout-cache-path-mmvq.cpp: host-weight registration is
    // only honoured inside a model-load transaction, and that transaction is
    // the only producer of a dense-weight cache entry -- so registration and
    // upload must both happen inside it (llama.cpp-43uy).
    ggml_sycl::test_clear_host_weight_registry();
    ggml_sycl::test_layout_override_guard layout_guard(layout);

    ggml_sycl_load_txn    load{};
    ggml_sycl_model_token model{};
    if (ggml_backend_sycl_model_load_begin(&load) != GGML_SYCL_LIFECYCLE_OK) {
        std::printf("FAIL [%s]: model_load_begin failed\n", layout_name);
        ++g_failures;
        cleanup(nullptr);
        return empty;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev) {
        ggml_backend_sycl_register_host_weight_tensor(dev, weight);
    }
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(block_q8_0_test));

    if (ggml_backend_sycl_model_load_end(load, true, &model) != GGML_SYCL_LIFECYCLE_OK) {
        std::printf("FAIL [%s]: model_load_end failed\n", layout_name);
        ++g_failures;
        cleanup(nullptr);
        return empty;
    }

    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    auto resolved = ggml_sycl_resolve(weight, 0);
    if (!resolved || resolved.layout != layout) {
        std::printf("FAIL [%s]: expected layout not resolved (got %d)\n", layout_name,
                    resolved ? (int) resolved.layout : -1);
        ++g_failures;
        cleanup(&model);
        return empty;
    }

    // Overwrite the AoS staging bytes so a kernel that (incorrectly) reads
    // them instead of the resolved/cached layout cannot pass by accident.
    std::memset(weight->data, 0, ggml_nbytes(weight));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    const enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::printf("FAIL [%s]: graph compute failed (status=%d)\n", layout_name, (int) status);
        ++g_failures;
        cleanup(&model);
        return empty;
    }

    std::vector<float> gpu_output((size_t) nrows * batch, 0.0f);
    ggml_backend_tensor_get(output, gpu_output.data(), 0, gpu_output.size() * sizeof(float));

    cleanup(&model);
    return gpu_output;
}

static void compare(const char *               what,
                    const std::vector<float> & got,
                    const std::vector<float> & ref,
                    float                      rel_tol,
                    float                      abs_tol) {
    if (got.empty() || ref.empty() || got.size() != ref.size()) {
        std::printf("FAIL [%s]: size mismatch or missing output (got=%zu ref=%zu)\n", what, got.size(), ref.size());
        ++g_failures;
        return;
    }
    float max_diff = 0.0f;
    float max_rel  = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) {
        const float diff = std::fabs(got[i] - ref[i]);
        max_diff         = std::max(max_diff, diff);
        const float rel  = std::fabs(ref[i]) > 1e-6f ? diff / std::fabs(ref[i]) : diff;
        max_rel          = std::max(max_rel, rel);
    }
    const bool ok = max_rel < rel_tol || max_diff < abs_tol;
    std::printf("%s [%s]: max_diff=%.6e max_rel=%.6e (tol rel=%.1e abs=%.1e)\n", ok ? "OK" : "FAIL", what, max_diff,
                max_rel, rel_tol, abs_tol);
    if (!ok) {
        ++g_failures;
    }
}

// Independent re-read of the same env var mmvq_q8_0_soa_fastpath_enabled()
// (mmvq.cpp, static/file-local, not reachable from here) consults: default
// ON, GGML_SYCL_MMVQ_Q8_SOA_FASTPATH=0 disables. If this copy and the
// production accessor's semantics ever drift apart, the production dispatch
// still does whatever IT reads -- this copy only decides what THIS TEST
// expects and prints, so a drift shows up as this test's geometry-decision
// check failing, not as it silently agreeing with a broken accessor.
static bool fastpath_env_enabled() {
    const char * env = std::getenv("GGML_SYCL_MMVQ_Q8_SOA_FASTPATH");
    return !(env && std::atoi(env) == 0);
}

// Independent re-derivation of the DOCUMENTED floor formula from
// mmvq-launch-geometry.hpp's ggml_sycl_mmvq_q8_0_soa_geometry comment --
// the SAME re-derivation tests/test-sycl-mmvq-q8-0-soa-geometry.cpp's
// expected_occupancy_floor() uses, kept in sync with the header's prose, not
// with its code, so a shape's expected geometry decision below is not
// merely re-running the code under test against itself.
static int expected_occupancy_floor(int ncols, int n_cores) {
    constexpr int kReferenceCols       = 4096;
    constexpr int kOccupancyMultiplier = 2;
    const double  k_relief             = (ncols > kReferenceCols) ? (double) kReferenceCols / (double) ncols : 1.0;
    const int     floor_wgs            = (int) (kOccupancyMultiplier * n_cores * k_relief + 0.5);
    return floor_wgs < 1 ? 1 : floor_wgs;
}

// See "BRANCH ENGAGEMENT / GEOMETRY DECISION" in the file header comment.
static void check_geometry_decision(const char * label, int ncols, int nrows) {
    const int  device_id = 0;  // ggml_backend_sycl_init(0) below -- the same index production code would resolve.
    const int  n_cores   = (int) ggml_sycl_info().devices[device_id].xmx_caps.compute_units;
    const bool env_on    = fastpath_env_enabled();
    const auto geometry  = env_on ? ggml_sycl::ggml_sycl_mmvq_q8_0_soa_geometry(nrows, ncols, n_cores) :
                                    ggml_sycl::mmvq_q8_0_soa_geometry_t{ 16, 1 };

    const int  baseline_wgs     = ggml_sycl::mmvq_pad_rows_to_workgroups(nrows, 16) / 16;
    const int  floor_wgs        = expected_occupancy_floor(ncols, n_cores);
    const bool expect_changed   = env_on && (baseline_wgs < floor_wgs);
    const bool geometry_changed = geometry.subgroups_per_workgroup != 16;

    std::printf("  geometry decision [%s]: fastpath=%s n_cores=%d baseline_wgs=%d floor=%d -> subgroups=%d (%s)\n",
                label, env_on ? "on" : "off", n_cores, baseline_wgs, floor_wgs, geometry.subgroups_per_workgroup,
                geometry_changed ? "CHANGED" : "today's 16");

    if (geometry_changed != expect_changed) {
        std::printf(
            "FAIL [%s]: geometry decision mismatch -- today's fixed geometry gives %d work-groups against a floor "
            "of %d (GGML_SYCL_MMVQ_Q8_SOA_FASTPATH=%s), so the geometry was expected to %s, but the helper picked "
            "subgroups_per_workgroup=%d\n",
            label, baseline_wgs, floor_wgs, env_on ? "on" : "off", expect_changed ? "change" : "stay at 16",
            geometry.subgroups_per_workgroup);
        ++g_failures;
    }
}

// See "PRODUCTION DISPATCH OBSERVATION" in the file header comment. Runs the
// SOA layout under forced profiler capture and reports whether the
// production dispatch code path (not this test's own prediction) actually
// took the fast-path branch. Both mmvq.cpp branches label their
// ggml_sycl_profile_label "mulmat.mmvq.q8_0_soa" (profiler continuity), so
// the distinguishing observable is the metadata string: only the fast
// path's includes ";subgroups=" (mmvq.cpp's fast-path branch); the
// unchanged original kernel's is "ncols=...;nrows=..." with no such field.
struct soa_run_result {
    std::vector<float> output;
    bool               fastpath_dispatch_observed = false;
};

static soa_run_result run_soa_with_dispatch_observation(ggml_backend_t                       backend,
                                                        int                                  ncols,
                                                        int                                  nrows,
                                                        int                                  batch,
                                                        const std::vector<block_q8_0_test> & weight_data,
                                                        const std::vector<float> &           input_data) {
    ggml_sycl_kernel_profile_reset_for_test();
    ggml_sycl_kernel_profile_config cfg;
    cfg.enabled = true;
    ggml_sycl_kernel_profile_set_config_for_test(cfg);

    soa_run_result result;
    result.output = run_layout(backend, GGML_LAYOUT_SOA, "soa", ncols, nrows, batch, weight_data, input_data);

    ggml_sycl_kernel_profile_flush(/*wait_for_events=*/true, "llama.cpp-6cgq dispatch observation");
    const std::string csv = ggml_sycl_kernel_profile_format_csv_for_test();
    result.fastpath_dispatch_observed =
        csv.find("mulmat.mmvq.q8_0_soa") != std::string::npos && csv.find("subgroups=") != std::string::npos;

    // Reset so profiler state does not leak into later shapes' unprofiled
    // GPU work (and so a later shape's capture starts from an empty table).
    ggml_sycl_kernel_profile_reset_for_test();
    return result;
}

static void run_shape(ggml_backend_t backend, const char * label, int ncols, int nrows, int batch) {
    std::printf("== shape %s: ncols=%d nrows=%d batch=%d ==\n", label, ncols, nrows, batch);
    check_geometry_decision(label, ncols, nrows);
    const int blocks_per_row = ncols / QK8_0;
    const int nblocks        = nrows * blocks_per_row;

    std::mt19937                 rng(0x6c9au ^ (ncols * 2654435761u) ^ (nrows * 40503u) ^ (unsigned) batch);
    std::vector<block_q8_0_test> weight_data(nblocks);
    fill_q8_0_random(weight_data.data(), nblocks, rng);

    std::uniform_real_distribution<float> xdist(-1.0f, 1.0f);
    std::vector<float>                    input_data((size_t) ncols * batch);
    for (float & v : input_data) {
        v = xdist(rng);
    }

    std::vector<float> ref((size_t) nrows * batch);
    compute_reference(weight_data.data(), input_data.data(), ref.data(), ncols, nrows, batch);

    const soa_run_result soa_result =
        run_soa_with_dispatch_observation(backend, ncols, nrows, batch, weight_data, input_data);
    const std::vector<float> & soa = soa_result.output;
    const std::vector<float>   coalesced =
        run_layout(backend, GGML_LAYOUT_COALESCED, "coalesced", ncols, nrows, batch, weight_data, input_data);

    // PRODUCTION DISPATCH OBSERVATION: correct numerics prove nothing about
    // which kernel produced them (the fast path is bit-identical to the
    // original by design), so a broken mmvq_q8_0_soa_fastpath_enabled()
    // (e.g. always reading off) would otherwise pass every comparison below
    // on the ON ctest arm while never actually exercising this ticket's
    // kernel. This checks a REAL profiler record from the production
    // dispatch, not a self-computed prediction.
    const bool expect_fastpath_dispatched = fastpath_env_enabled();
    if (soa_result.fastpath_dispatch_observed != expect_fastpath_dispatched) {
        std::printf(
            "FAIL [%s]: production dispatch observation mismatch -- GGML_SYCL_MMVQ_Q8_SOA_FASTPATH=%s, expected a "
            "mulmat.mmvq.q8_0_soa profiler record %s a 'subgroups=' metadata field, but %s\n",
            label, expect_fastpath_dispatched ? "on" : "off", expect_fastpath_dispatched ? "WITH" : "WITHOUT",
            soa_result.fastpath_dispatch_observed ? "one WITH subgroups= was recorded (fast path ran)" :
                                                    "no such record was found (fast path did NOT run)");
        ++g_failures;
    }

    // Q8_0 x Q8_1 accumulation: generous but not vacuous tolerance, matching
    // test-q8-0-layout-cache-path-mmvq.cpp's precedent for this same op.
    const float rel_tol = 1e-3f;
    const float abs_tol = 5e-2f;
    compare("soa-vs-cpu-reference", soa, ref, rel_tol, abs_tol);
    compare("coalesced-vs-cpu-reference", coalesced, ref, rel_tol, abs_tol);
    // Tighter: both kernels read the SAME bytes, so their outputs should
    // agree closely regardless of either's absolute accuracy against the
    // reference.
    compare("soa-vs-coalesced", soa, coalesced, 1e-4f, 1e-2f);
}

int main() {
    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        std::printf("SKIP: no SYCL GPU device available\n");
        return LLAMA_TEST_EXIT_SKIP;
    }

    // Shapes from the llama.cpp-6cgq profile table plus the ticket's and the
    // spec review's requested extra coverage (a non-Mistral K, a tiny row
    // count, and the LM-head shape -- spec review round 2, finding 7).
    run_shape(backend, "k,v N=1024 K=4096 b1", 4096, 1024, 1);
    run_shape(backend, "q,o N=4096 K=4096 b1", 4096, 4096, 1);
    run_shape(backend, "gate/up N=14336 K=4096 b1", 4096, 14336, 1);
    run_shape(backend, "down N=4096 K=14336 b1", 14336, 4096, 1);
    run_shape(backend, "lm-head N=32000 K=4096 b1", 4096, 32000, 1);
    run_shape(backend, "gemma4-ish N=2048 K=2560 b1", 2560, 2048, 1);
    run_shape(backend, "N=10240 K=2560 b1", 2560, 10240, 1);
    run_shape(backend, "tiny N=37 K=4096 b1", 4096, 37, 1);
    // Batch 2 and 8 for the shape the fast path's geometry changes most.
    run_shape(backend, "k,v N=1024 K=4096 b2", 4096, 1024, 2);
    run_shape(backend, "k,v N=1024 K=4096 b8", 4096, 1024, 8);

    ggml_backend_free(backend);

    if (g_failures) {
        std::printf("FAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("PASS: q8_0 SOA MMVQ fast path numerics\n");
    return 0;
}
