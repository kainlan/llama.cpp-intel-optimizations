// Minimal MoE graph test: embedding -> RMS norm -> MoE MUL_MAT_ID (MXFP4).
// Compares SYCL vs CPU outputs to isolate backend divergence.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml-cpu.h"
#include "ggml-sycl/ggml-sycl-test.hpp"
#include "ggml-quants.h"

// ctest SKIP_RETURN_CODE (house convention, matching the test-sycl-*-policy.sh
// family). Every skip path in this file exits with this instead of 0 so that a
// skip is VISIBLE as a skip: llama.cpp-v9ue found that returning 0 from a skip
// made every historical green indistinguishable from a real comparison.
#define TEST_EXIT_SKIP 77

#if !defined(GGML_USE_SYCL)
int main() {
    fprintf(stderr, "SKIP: GGML_USE_SYCL not enabled; this run proves NOTHING about the SYCL MoE graph path.\n");
    return TEST_EXIT_SKIP;
}
#else

// Outcome of running the graph on one backend. SKIP and FAIL must stay distinct:
// llama.cpp-v9ue found that the previous bool return collapsed them, so a real
// failure in the SYCL leg was reported by main() as "SKIP: SYCL graph path
// unavailable or disabled" with exit 0, and the CPU-vs-SYCL comparison this test
// exists for never ran in any run.
enum class leg_result {
    OK,
    SKIP,
    FAIL,
};

// The SYCL backend runs a warmup pass the first time it sees a given
// (phase, n_nodes) pair -- it dispatches per-op, releases the transient graph
// leases and returns before recording anything (ggml-sycl.cpp, the
// `warmup_n_nodes != cgraph->n_nodes` branch in graph_compute). A single compute
// call therefore can never reach the record/replay path. Three passes give
// warmup, record and replay, so the output compared below is the one the graph
// path actually produced.
static const int SYCL_COMPUTE_PASSES = 3;

struct moe_graph_inputs {
    int vocab;
    int n_embd;
    int n_tokens;
    int n_used;
    int n_experts;
    int out_dim;
    std::vector<float> tok_embd_f32;
    std::vector<int32_t> token_ids;
    std::vector<int32_t> expert_ids;
    std::vector<block_mxfp4> moe_weights;
};

static double nmse(const float * a, const float * b, size_t n) {
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double da = a[i];
        const double db = b[i];
        const double diff = da - db;
        mse_a_b += diff * diff;
        mse_a_0 += da * da;
    }
    return mse_a_b / (mse_a_0 > 0.0 ? mse_a_0 : 1.0);
}

static float max_diff(const float * a, const float * b, size_t n) {
    float max_d = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float d = fabsf(a[i] - b[i]);
        if (d > max_d) {
            max_d = d;
        }
    }
    return max_d;
}

static ggml_backend_buffer_t alloc_tensor_buffer(ggml_backend_buffer_type_t buft, ggml_tensor * tensor,
                                                 ggml_backend_buffer_usage usage) {
    const size_t size = ggml_backend_buft_get_alloc_size(buft, tensor);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, size);
    if (!buffer) {
        return nullptr;
    }
    ggml_backend_buffer_set_usage(buffer, usage);
    ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer));
    return buffer;
}

static void fill_random(std::vector<float> & data, float scale, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (float & v : data) {
        v = dist(rng);
    }
}

static void build_inputs(moe_graph_inputs & inputs) {
    std::mt19937 rng(2025);
    inputs.tok_embd_f32.resize(static_cast<size_t>(inputs.vocab) * inputs.n_embd);
    fill_random(inputs.tok_embd_f32, 0.5f, rng);

    inputs.token_ids.resize(inputs.n_tokens);
    for (int i = 0; i < inputs.n_tokens; ++i) {
        inputs.token_ids[i] = i % inputs.vocab;
    }

    inputs.expert_ids.resize(static_cast<size_t>(inputs.n_used) * inputs.n_tokens);
    for (int t = 0; t < inputs.n_tokens; ++t) {
        for (int i = 0; i < inputs.n_used; ++i) {
            inputs.expert_ids[t * inputs.n_used + i] = (t + i) % inputs.n_experts;
        }
    }

    if (inputs.n_embd % QK_MXFP4 != 0) {
        fprintf(stderr, "FAIL: n_embd must be divisible by QK_MXFP4\n");
        std::abort();
    }

    const int blocks_per_row = inputs.n_embd / QK_MXFP4;
    inputs.moe_weights.resize(static_cast<size_t>(inputs.n_experts) * inputs.out_dim * blocks_per_row);
    std::vector<float> weight_f32(static_cast<size_t>(inputs.n_experts) * inputs.out_dim * inputs.n_embd);
    fill_random(weight_f32, 0.4f, rng);
    for (int e = 0; e < inputs.n_experts; ++e) {
        for (int r = 0; r < inputs.out_dim; ++r) {
            const size_t row_idx = static_cast<size_t>(e) * inputs.out_dim + r;
            const float * row = weight_f32.data() + row_idx * inputs.n_embd;
            block_mxfp4 * out = inputs.moe_weights.data() + row_idx * blocks_per_row;
            quantize_row_mxfp4_ref(row, out, inputs.n_embd);
        }
    }
}

static leg_result run_moe_graph_backend(ggml_backend_t backend,
                                        bool use_host_moe_weights,
                                        const moe_graph_inputs & inputs,
                                        bool require_graphs,
                                        int n_passes,
                                        std::vector<float> & output) {
    const ggml_init_params params = {
        64 * 1024 * 1024,
        nullptr,
        true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "FAIL: ggml_init failed\n");
        return leg_result::FAIL;
    }

    ggml_tensor * tok_embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, inputs.n_embd, inputs.vocab);
    ggml_set_name(tok_embd, "tok_embd.weight");
    ggml_tensor * token_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, inputs.n_tokens);
    ggml_set_name(token_ids, "token_ids");
    ggml_tensor * embd = ggml_get_rows(ctx, tok_embd, token_ids);
    ggml_set_name(embd, "embd");
    ggml_tensor * norm = ggml_rms_norm(ctx, embd, 1e-5f);
    ggml_set_name(norm, "rms_norm");
    ggml_tensor * norm3d = ggml_reshape_3d(ctx, norm, inputs.n_embd, 1, inputs.n_tokens);
    ggml_set_name(norm3d, "rms_norm_3d");
    ggml_tensor * moe_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, inputs.n_used, inputs.n_tokens);
    ggml_set_name(moe_ids, "moe_ids");
    ggml_tensor * moe_w = ggml_new_tensor_3d(ctx, GGML_TYPE_MXFP4, inputs.n_embd, inputs.out_dim, inputs.n_experts);
    ggml_set_name(moe_w, "moe_weights");
    ggml_tensor * moe_out = ggml_mul_mat_id(ctx, moe_w, norm3d, moe_ids);
    ggml_set_name(moe_out, "moe_out");
    ggml_tensor * final_out = moe_out;
    std::vector<ggml_tensor *> extra_ops;
    for (int i = 0; i < 6; ++i) {
        final_out = ggml_scale(ctx, final_out, 1.0f);
        extra_ops.push_back(final_out);
    }

    ggml_backend_buffer_type_t dev_buft = ggml_backend_get_default_buffer_type(backend);
    ggml_backend_buffer_type_t moe_buft = dev_buft;

    if (use_host_moe_weights) {
        moe_buft = ggml_backend_sycl_host_buffer_type();
    }

    std::vector<ggml_backend_buffer_t> buffers;
    auto alloc_or_fail = [&](ggml_backend_buffer_type_t buft, ggml_tensor * t, ggml_backend_buffer_usage usage) {
        if (t->view_src) {
            return ggml_backend_view_init(t) == GGML_STATUS_SUCCESS;
        }
        ggml_backend_buffer_t buf = alloc_tensor_buffer(buft, t, usage);
        if (!buf) {
            return false;
        }
        buffers.push_back(buf);
        return true;
    };

    if (!alloc_or_fail(dev_buft, tok_embd, GGML_BACKEND_BUFFER_USAGE_WEIGHTS) ||
        !alloc_or_fail(dev_buft, token_ids, GGML_BACKEND_BUFFER_USAGE_COMPUTE) ||
        !alloc_or_fail(dev_buft, embd, GGML_BACKEND_BUFFER_USAGE_COMPUTE) ||
        !alloc_or_fail(dev_buft, norm, GGML_BACKEND_BUFFER_USAGE_COMPUTE) ||
        !alloc_or_fail(dev_buft, norm3d, GGML_BACKEND_BUFFER_USAGE_COMPUTE) ||
        !alloc_or_fail(dev_buft, moe_ids, GGML_BACKEND_BUFFER_USAGE_COMPUTE) ||
        !alloc_or_fail(moe_buft, moe_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS) ||
        !alloc_or_fail(dev_buft, moe_out, GGML_BACKEND_BUFFER_USAGE_COMPUTE)) {
        fprintf(stderr, "FAIL: tensor allocation failed\n");
        for (ggml_backend_buffer_t buf : buffers) {
            ggml_backend_buffer_free(buf);
        }
        ggml_free(ctx);
        return leg_result::FAIL;
    }
    for (ggml_tensor * extra : extra_ops) {
        if (!alloc_or_fail(dev_buft, extra, GGML_BACKEND_BUFFER_USAGE_COMPUTE)) {
            fprintf(stderr, "FAIL: tensor allocation failed for chained scale node\n");
            for (ggml_backend_buffer_t buf : buffers) {
                ggml_backend_buffer_free(buf);
            }
            ggml_free(ctx);
            return leg_result::FAIL;
        }
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (use_host_moe_weights && dev) {
        ggml_backend_sycl_register_host_weight_tensor(dev, moe_w);
    }

    ggml_backend_tensor_set(tok_embd, inputs.tok_embd_f32.data(), 0,
                            inputs.tok_embd_f32.size() * sizeof(float));
    ggml_backend_tensor_set(token_ids, inputs.token_ids.data(), 0,
                            inputs.token_ids.size() * sizeof(int32_t));
    ggml_backend_tensor_set(moe_ids, inputs.expert_ids.data(), 0,
                            inputs.expert_ids.size() * sizeof(int32_t));
    ggml_backend_tensor_set(moe_w, inputs.moe_weights.data(), 0,
                            inputs.moe_weights.size() * sizeof(block_mxfp4));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, final_out);

    auto release = [&]() {
        for (ggml_backend_buffer_t buf : buffers) {
            ggml_backend_buffer_free(buf);
        }
        ggml_free(ctx);
    };

    if (require_graphs) {
        if (!ggml_sycl::test_backend_supports_graphs(backend)) {
            fprintf(stderr, "SKIP: backend does not support graphs\n");
            release();
            return leg_result::SKIP;
        }
        if (ggml_sycl::test_backend_graphs_disabled(backend)) {
            fprintf(stderr, "SKIP: backend graphs disabled by configuration\n");
            release();
            return leg_result::SKIP;
        }
    }

    for (int pass = 0; pass < n_passes; ++pass) {
        const ggml_status status = ggml_backend_graph_compute(backend, graph);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "FAIL: graph compute returned %d on pass %d/%d\n", (int) status, pass + 1, n_passes);
            release();
            return leg_result::FAIL;
        }
    }

    if (require_graphs) {
        // Observation, not an assertion. This block used to require
        // test_graph_pinned_entry_count() > 0 and treat zero as a failure --
        // which main() then reported as a skip, hiding the whole test. That
        // count is structurally unreachable for this graph: graph_preload_weights
        // skips device-VRAM weights ("already on GPU") and MoE expert weights, and
        // those are the only two weights here, so graph_weight_leases is never
        // populated no matter how the run goes. Print the graph state so a run
        // says what the backend actually did, and let the CPU-vs-SYCL comparison
        // in main() be the thing that passes or fails (llama.cpp-v9ue).
        fprintf(stderr, "MoE mini-graph SYCL state: exec_graph=%d pinned_entries=%zu replays=%llu passes=%d\n",
                ggml_sycl::test_backend_has_exec_graph(backend) ? 1 : 0,
                ggml_sycl::test_graph_pinned_entry_count(backend),
                (unsigned long long) ggml_sycl::test_backend_graph_replay_count(backend), n_passes);
    }

    output.resize(static_cast<size_t>(inputs.out_dim) * inputs.n_used * inputs.n_tokens);
    ggml_backend_tensor_get(final_out, output.data(), 0, output.size() * sizeof(float));

    release();
    return leg_result::OK;
}

int main() {
    setenv("GGML_SYCL_XMX_MOE", "1", 1);
    setenv("GGML_SYCL_XMX_MOE_TILED", "1", 1);

    ggml_backend_t sycl_backend = ggml_backend_sycl_init(0);
    if (!sycl_backend) {
        fprintf(stderr,
                "SKIP: could not initialize SYCL backend (no device, or oneAPI not sourced);"
                " this run proves NOTHING about the SYCL MoE graph path.\n");
        return TEST_EXIT_SKIP;
    }

    ggml_backend_t cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu_backend) {
        fprintf(stderr, "FAIL: Could not initialize CPU backend\n");
        ggml_backend_free(sycl_backend);
        return 1;
    }

    moe_graph_inputs inputs{};
    inputs.vocab = 64;
    inputs.n_embd = 256;
    inputs.n_tokens = 4;
    inputs.n_used = 4;
    inputs.n_experts = 8;
    inputs.out_dim = 128;
    build_inputs(inputs);

    std::vector<float> cpu_out;
    std::vector<float> sycl_out;

    const leg_result cpu_leg  = run_moe_graph_backend(cpu_backend, false, inputs, false, 1, cpu_out);
    const leg_result sycl_leg = run_moe_graph_backend(sycl_backend, true, inputs, true, SYCL_COMPUTE_PASSES, sycl_out);

    ggml_backend_free(cpu_backend);
    ggml_backend_free(sycl_backend);

    // The CPU leg has no skip path -- it does not ask for graphs -- so anything
    // other than OK is a failure of the baseline itself.
    if (cpu_leg != leg_result::OK) {
        fprintf(stderr, "FAIL: CPU baseline did not produce a result\n");
        return 1;
    }

    // A SYCL failure is a failure. Only a genuinely unavailable graph path is a
    // skip, and it exits 77 so ctest reports it as skipped rather than passed.
    if (sycl_leg == leg_result::FAIL) {
        fprintf(stderr, "FAIL: SYCL leg failed (see the FAIL line above)\n");
        return 1;
    }
    if (sycl_leg == leg_result::SKIP) {
        fprintf(stderr,
                "SKIP: SYCL graph path unavailable or disabled;"
                " this run proves NOTHING about MoE graph correctness.\n");
        return TEST_EXIT_SKIP;
    }

    if (cpu_out.size() != sycl_out.size()) {
        fprintf(stderr, "FAIL: output size mismatch cpu=%zu sycl=%zu\n", cpu_out.size(), sycl_out.size());
        return 1;
    }

    const double err = nmse(cpu_out.data(), sycl_out.data(), cpu_out.size());
    const float max_d = max_diff(cpu_out.data(), sycl_out.data(), cpu_out.size());
    fprintf(stderr, "MoE mini-graph: nmse=%.6e max_diff=%.6f\n", err, max_d);

    const double nmse_tol = 5e-4;
    const float max_tol = 1.0f;
    if (!std::isfinite(err) || err > nmse_tol || !std::isfinite(max_d) || max_d > max_tol) {
        fprintf(stderr, "FAIL: mismatch beyond tolerance (nmse>%.1e or max_diff>%.2f)\n", nmse_tol, max_tol);
        return 1;
    }

    fprintf(stderr, "PASS\n");
    return 0;
}
#endif
