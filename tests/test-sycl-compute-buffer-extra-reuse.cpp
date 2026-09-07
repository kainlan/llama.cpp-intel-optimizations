// GPU test for llama.cpp-dfo0 (plan task L2): the SYCL COMPUTE-usage buffer's
// tensor_extras vector must stay bounded across repeated GRAPH REBUILDS, not
// just repeated graph REUSES.
//
// Background: ggml_backend_sycl_buffer_reset's COMPUTE branch used to
// unconditionally preserve ctx->tensor_extras on every reset, on the premise
// that init_tensor is never called again for a buffer once allocated. That premise
// holds for a REUSED graph (ggml_backend_sched_graph_compute called again on the
// SAME ggml_cgraph / ggml_context) but not for a REBUILT one: llm_graph_result::
// reset() (src/llama-graph.cpp) re-runs ggml_init over the SAME backing
// mem_buffer for every decode, minting fresh ggml_tensor structs at the same
// addresses old ones occupied. init_tensor then allocates a brand-new extra per
// tensor, and the previously "preserved" entries become permanently unreachable
// -- measured on hardware (llama.cpp-dfo0 c-wx3o/c-pozz) at 838 extras/rebuild,
// ~465 MB host RSS growth per pp1024 decode, with no bound. That measurement
// predates llama.cpp-h9uv's struct split (pre-h9uv sizeof(ggml_tensor_extra_gpu)
// = 277,712 B); post-split the same 838-extra/rebuild leak would be
// sizeof(ggml_tensor_extra_gpu) = 25,048 B each, ~21 MB per rebuild -- see
// test-sycl-extra-gpu-size.cpp for the compile-verified current size.
//
// This test alternates two DIFFERENTLY-SHAPED graphs through the same
// ggml_backend_sched, each built in a FRESH ggml_context over the SAME raw
// mem_buffer (mirroring llm_graph_result::reset exactly -- see its comment
// above for the citation), so every iteration is a genuine rebuild: the new
// ggml_tensor structs alias the exact byte ranges the previous iteration's
// tensor structs occupied. It asserts the COMPUTE buffer's tensor_extras count
// (read via the GGML_SYCL_PRIVATE_TESTING-only debug accessor
// ggml_backend_sycl_debug_last_compute_buffer_extra_count(), since
// ggml_backend_sched does not hand test code the compute buffer it allocates
// internally) stays bounded across 20 rebuilds instead of growing linearly.
//
// RED/GREEN evidence (recorded here rather than reproduced by this file, since
// RED requires a scratch source edit that must never land -- see the commit
// message for what was actually observed): the release condition in
// ggml_backend_sycl_buffer_reset's COMPUTE branch,
//
//     if (extra->alloc_generation + 1 < ctx->alloc_generation) { ... release ... }
//
// is the fix under test. A scratch revert of just that condition to `if
// (false)` (keeping every other line, INCLUDING the debug-accessor bookkeeping,
// intact) reproduces the pre-fix behaviour without needing a separately built
// binary: nothing is ever released, so the count this test prints grows by
// ~N_TENSORS_PER_GRAPH (3) every iteration instead of staying
// <= 2*N_TENSORS_PER_GRAPH.
//
// ggml_backend_sched_new() asserts its LAST backend entry is a CPU device
// (ggml-backend.cpp:2518); this test still runs everything on SYCL (the CPU
// backend is present only to satisfy that structural requirement, matching
// how llama.cpp itself always registers a CPU backend), and asserts the
// per-iteration extras count is > 0 once steady state is reached so a
// misplacement onto the CPU backend (where the SYCL debug accessor would
// silently read 0 forever, making the "count stayed bounded" assertion pass
// for the wrong reason) fails loudly instead of passing vacuously.
//
// Usage:
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-sycl-compute-buffer-extra-reuse

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-sycl.h"
#include "ggml.h"
#include "sycl-selector-fallback.hpp"
#include "test-skip.h"  // LLAMA_TEST_EXIT_SKIP: the one definition of "77 means skip"

#include <cstdio>
#include <cstdlib>
#include <vector>

#if !defined(GGML_SYCL_PRIVATE_TESTING)
#    error "this test requires GGML_SYCL_PRIVATE_TESTING (link against ggml-sycl-private-fixtures)"
#endif

namespace {

constexpr int64_t HIDDEN              = 4096;
constexpr int64_t N_TOKENS_A          = 512;
constexpr int64_t N_TOKENS_B          = 1024;
constexpr int     N_ITERS             = 20;
// Tensors gallocr actually owns and allocates per graph iteration: the input
// leaf (x), the rms_norm output, and the mul_mat output. The weight tensor is
// pre-allocated in its own WEIGHTS-usage buffer and is never touched by the
// COMPUTE buffer's tensor_extras bookkeeping.
constexpr size_t  N_TENSORS_PER_GRAPH = 3;

// Builds "x[HIDDEN, n_tokens] -> rms_norm -> mul_mat(weight, .)" in ctx/gf and
// returns the input leaf so the caller can upload data into it once allocated.
ggml_tensor * build_graph(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * weight, int64_t n_tokens) {
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HIDDEN, n_tokens);
    ggml_set_name(x, "x");
    ggml_set_input(x);

    ggml_tensor * normed = ggml_rms_norm(ctx, x, 1e-5f);
    ggml_set_name(normed, "normed");

    ggml_tensor * out = ggml_mul_mat(ctx, weight, normed);
    ggml_set_name(out, "out");
    ggml_set_output(out);

    ggml_build_forward_expand(gf, out);
    return x;
}

}  // namespace

int main(int, char ** argv) {
    // See tests/sycl-selector-fallback.hpp for why a plain setenv() here does
    // not work (libccl's static initializer memoizes the selector before
    // main() runs, llama.cpp-2x3m) and why the fix is a re-exec.
    sycl_test_selector_fallback(argv, "level_zero:1");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        fprintf(stderr,
                "SKIP: no SYCL GPU device available -- NO DEVICE WORK WAS PERFORMED.\n"
                "      source /opt/intel/oneapi/setvars.sh --force and re-run.\n");
        return LLAMA_TEST_EXIT_SKIP;
    }

    // ggml_backend_sched_new() requires its last backend to be a CPU device
    // (ggml-backend.cpp:2518, GGML_ASSERT) -- present as a structural
    // fallback only; every op in this graph (rms_norm, quantized mul_mat) is
    // SYCL-supported, so nothing is expected to actually dispatch here.
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        fprintf(stderr, "FAIL: ggml_backend_cpu_init failed\n");
        ggml_backend_free(backend);
        return 1;
    }

    // Persistent Q8_0 weight, allocated once in its own WEIGHTS-usage buffer --
    // mirrors how llama.cpp's model weights outlive every ephemeral per-decode
    // graph context and are referenced by pointer from each rebuild.
    ggml_init_params wparams = {
        /*.mem_size   =*/ggml_tensor_overhead() + 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * wctx = ggml_init(wparams);
    if (!wctx) {
        fprintf(stderr, "FAIL: ggml_init (weight context) failed\n");
        ggml_backend_free(cpu);
        ggml_backend_free(backend);
        return 1;
    }
    ggml_tensor * weight = ggml_new_tensor_2d(wctx, GGML_TYPE_Q8_0, HIDDEN, HIDDEN);
    ggml_set_name(weight, "weight");

    ggml_backend_buffer_t weight_buf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, ggml_backend_sycl_buffer_type(0));
    if (!weight_buf) {
        fprintf(stderr, "FAIL: failed to allocate the weight buffer\n");
        ggml_free(wctx);
        ggml_backend_free(cpu);
        ggml_backend_free(backend);
        return 1;
    }
    ggml_backend_buffer_set_usage(weight_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    {
        // Zero-initialise so the quantized scale/qs bytes form a valid (if
        // meaningless) Q8_0 block -- this test checks bookkeeping, not
        // numerics, but a zeroed weight avoids any chance of a stray NaN/Inf
        // bit pattern from uninitialised device memory.
        std::vector<uint8_t> zeros(ggml_nbytes(weight), 0);
        ggml_backend_tensor_set(weight, zeros.data(), 0, zeros.size());
    }

    ggml_backend_t       backends[2] = { backend, cpu };  // CPU MUST be last -- see the comment above.
    ggml_backend_sched_t sched       = ggml_backend_sched_new(backends, nullptr, 2, 4096, false, true);
    if (!sched) {
        fprintf(stderr, "FAIL: ggml_backend_sched_new failed\n");
        ggml_backend_buffer_free(weight_buf);
        ggml_free(wctx);
        ggml_backend_free(cpu);
        ggml_backend_free(backend);
        return 1;
    }

    // Fresh ggml_init over the SAME raw mem_buffer every iteration, exactly
    // mirroring llm_graph_result::reset() (src/llama-graph.cpp): this is what
    // makes each iteration a genuine graph REBUILD rather than a graph REUSE
    // -- the new tensor structs alias the byte ranges the previous iteration's
    // structs occupied, so a use of a stale ggml_tensor* would be reading
    // freshly-reinitialised memory, not merely stale-but-valid memory.
    const size_t         max_nodes = 16;
    const size_t         mem_size  = ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false);
    std::vector<uint8_t> mem_buffer(mem_size);

    bool   ok          = true;
    size_t max_extras  = 0;
    size_t last_extras = 0;

    for (int iter = 0; iter < N_ITERS && ok; ++iter) {
        // The LARGER shape must run first (llama.cpp-dfo0 plan task L2, spec
        // review c-cnko #2): iteration 0 is what sizes the compute buffer, and a
        // later iteration whose tensors exceed that recorded size_max forces
        // ggml_gallocr_alloc_graph to reallocate (ggml-alloc.c ggml_gallocr_
        // needs_realloc -> ggml_backend_sched's reserve fallback) -- which frees
        // the old ggml_backend_sycl_buffer_context and replaces it with a fresh
        // one at generation 0, an empty tensor_extras, invalidating both the
        // GREEN/RED arithmetic below and the iter>=1 positive control (a freshly
        // reallocated buffer legitimately reads 0). Running N_TOKENS_B (1024)
        // first reserves a buffer both shapes fit for the rest of the run.
        const int64_t n_tokens = (iter % 2 == 0) ? N_TOKENS_B : N_TOKENS_A;

        // ggml_backend_sched_alloc_graph() asserts !sched->is_alloc
        // (ggml-backend.cpp:2656), set true by the previous iteration's alloc and
        // cleared ONLY by ggml_backend_sched_reset() -- mirroring how
        // llama_context resets its scheduler before every rebuild
        // (src/llama-context.cpp:1830). Without this, iteration 1 aborts
        // (llama.cpp-dfo0 plan task L2, spec review c-cnko #1).
        ggml_backend_sched_reset(sched);

        ggml_init_params iparams = {
            /*.mem_size   =*/mem_size,
            /*.mem_buffer =*/mem_buffer.data(),
            /*.no_alloc   =*/true,
        };
        ggml_context * ctx = ggml_init(iparams);
        if (!ctx) {
            fprintf(stderr, "FAIL: ggml_init failed at iteration %d\n", iter);
            ok = false;
            break;
        }

        ggml_cgraph * gf = ggml_new_graph_custom(ctx, max_nodes, false);
        ggml_tensor * x  = build_graph(ctx, gf, weight, n_tokens);

        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            fprintf(stderr, "FAIL: ggml_backend_sched_alloc_graph failed at iteration %d\n", iter);
            ggml_free(ctx);
            ok = false;
            break;
        }

        {
            std::vector<float> host_x(static_cast<size_t>(HIDDEN * n_tokens), 1.0f);
            ggml_backend_tensor_set(x, host_x.data(), 0, host_x.size() * sizeof(float));
        }

        const enum ggml_status status = ggml_backend_sched_graph_compute(sched, gf);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "FAIL: ggml_backend_sched_graph_compute failed at iteration %d (status=%d)\n", iter,
                    (int) status);
            ggml_free(ctx);
            ok = false;
            break;
        }

        last_extras = ggml_backend_sycl_debug_last_compute_buffer_extra_count();
        if (last_extras > max_extras) {
            max_extras = last_extras;
        }
        printf("iter=%d n_tokens=%lld compute_buffer_extras=%zu\n", iter, (long long) n_tokens, last_extras);

        ggml_free(ctx);

        if (last_extras > 2 * N_TENSORS_PER_GRAPH) {
            fprintf(stderr,
                    "FAIL: compute buffer tensor_extras count %zu exceeds 2*N_TENSORS_PER_GRAPH=%zu at iteration "
                    "%d -- extras from a rebuilt graph are not being released\n",
                    last_extras, 2 * N_TENSORS_PER_GRAPH, iter);
            ok = false;
            break;
        }
        // Positive control for the accessor itself: from iteration 1 onward
        // (iteration 0's pre-reset snapshot is legitimately 0, taken before
        // any tensor has ever been stamped) the count must be > 0. A silent
        // misplacement onto the CPU backend added above would never touch
        // the SYCL COMPUTE buffer's tensor_extras at all, leaving this
        // accessor stuck at 0 forever -- which the `<= bound` check above
        // would otherwise pass vacuously, exactly the "an empty probe is not
        // a measurement" trap.
        if (iter >= 1 && last_extras == 0) {
            fprintf(stderr,
                    "FAIL: compute buffer tensor_extras count is 0 at iteration %d -- graph tensors are not "
                    "landing on the SYCL COMPUTE buffer this accessor reads (misassigned to the CPU backend?)\n",
                    iter);
            ok = false;
            break;
        }
    }

    printf("max_extras_observed=%zu (bound=%zu)\n", max_extras, 2 * N_TENSORS_PER_GRAPH);

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(weight_buf);
    ggml_free(wctx);
    ggml_backend_free(cpu);
    ggml_backend_free(backend);

    if (!ok) {
        fprintf(stderr, "test-sycl-compute-buffer-extra-reuse: FAIL\n");
        return 1;
    }
    printf("test-sycl-compute-buffer-extra-reuse: PASS\n");
    return 0;
}
