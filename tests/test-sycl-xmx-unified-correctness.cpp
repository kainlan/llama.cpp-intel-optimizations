// Regression test for XMX-enabled unified kernel correctness.
// Reproduces incorrect output when GGML_SYCL_XMX_UNIFIED=1 interacts
// with layout finalization and reordered XMX layouts.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-sycl.h"
#include "ggml-sycl/ggml-sycl-test.hpp"
#include "ggml-quants.h"

#if !defined(GGML_USE_SYCL)
int main() {
    std::fprintf(stderr, "SKIP: GGML_USE_SYCL not enabled; this run proves NOTHING about the XMX path\n");
    return 77;  // ctest SKIP_RETURN_CODE -- a skip must not read as a pass
}
#else

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

static bool max_abs_diff(const std::vector<float> & a, const std::vector<float> & b, float & out_diff) {
    if (a.size() != b.size()) {
        return false;
    }
    float max_d = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        const float da = a[i];
        const float db = b[i];
        if (!std::isfinite(da) || !std::isfinite(db)) {
            out_diff = std::numeric_limits<float>::infinity();
            return true;
        }
        const float d = std::fabs(da - db);
        if (d > max_d) {
            max_d = d;
        }
    }
    out_diff = max_d;
    return true;
}

struct matmul_case {
    int K;  // reduction dim
    int N;  // output columns (weight rows)
    int M;  // batch/output rows (tokens)
    std::vector<block_q4_0> weights_q4;
    std::vector<float>      input_f32;
};

static void build_case(matmul_case & tc) {
    tc.K = 128;
    tc.N = 64;
    tc.M = 16;  // MEDIUM bucket -> dpas + XMX layouts

    if (tc.K % QK4_0 != 0) {
        std::fprintf(stderr, "SKIP: K must be divisible by QK4_0 (K=%d QK4_0=%d)\n", tc.K, QK4_0);
        std::abort();
    }

    const int blocks_per_row = tc.K / QK4_0;

    std::vector<float> weights_f32(static_cast<size_t>(tc.K) * tc.N);
    for (int row = 0; row < tc.N; ++row) {
        for (int col = 0; col < tc.K; ++col) {
            // Deterministic, non-trivial pattern.
            weights_f32[static_cast<size_t>(row) * tc.K + col] = 0.01f * float((row + 1) - (col % 17));
        }
    }

    tc.weights_q4.resize(static_cast<size_t>(tc.N) * blocks_per_row);
    for (int row = 0; row < tc.N; ++row) {
        const float * row_ptr = weights_f32.data() + static_cast<size_t>(row) * tc.K;
        block_q4_0 *  out_ptr = tc.weights_q4.data() + static_cast<size_t>(row) * blocks_per_row;
        quantize_row_q4_0_ref(row_ptr, out_ptr, tc.K);
    }

    tc.input_f32.resize(static_cast<size_t>(tc.K) * tc.M);
    for (int m = 0; m < tc.M; ++m) {
        for (int k = 0; k < tc.K; ++k) {
            tc.input_f32[static_cast<size_t>(m) * tc.K + k] = 0.02f * float((m + 3) + (k % 11));
        }
    }
}

// A run has three outcomes, and collapsing them into one bool is what made this
// test report `FAIL: SYCL backend run failed` for a compute that returned
// GGML_STATUS_SUCCESS.  SKIPPED means a capability the test needs is genuinely
// absent (ctest scores it skipped via exit 77); FAILED means a property the
// test asserts did not hold.
enum class run_status {
    OK,
    SKIPPED,
    FAILED,
};

static run_status run_backend_matmul(ggml_backend_t       backend,
                                     const matmul_case &  tc,
                                     bool                 require_gpu_graph_path,
                                     std::vector<float> & out) {
    const ggml_init_params params = {
        32 * 1024 * 1024,
        nullptr,
        true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "FAIL: ggml_init failed\n");
        return run_status::FAILED;
    }

    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, tc.K, tc.N);
    ggml_set_name(weight, "blk.0.attn_q.weight");
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, tc.K, tc.M);
    ggml_set_name(input, "xmx_input");
    ggml_set_input(input);

    ggml_tensor * out_mat = ggml_mul_mat(ctx, weight, input);
    ggml_set_name(out_mat, "xmx_out");

    // Enlarge the graph to exceed MIN_GRAPH_NODES and exercise graph replay.
    // 13 nodes total (1 MUL_MAT + 12 SCALE).  Do not grow this past 50 without
    // reading the exec-graph precondition below -- its soundness depends on the
    // node count staying under MIN_GPU_PREFIX_NODES.
    ggml_tensor * final_out = out_mat;
    std::vector<ggml_tensor *> extra_ops;
    for (int i = 0; i < 12; ++i) {
        final_out = ggml_scale(ctx, final_out, 1.0f);
        extra_ops.push_back(final_out);
    }

    ggml_backend_buffer_type_t dev_buft = ggml_backend_get_default_buffer_type(backend);
    if (!dev_buft) {
        std::fprintf(stderr, "FAIL: backend has no default buffer type\n");
        ggml_free(ctx);
        return run_status::FAILED;
    }

    // The weight goes in the backend's own buffer, NOT in the SYCL host-pinned
    // buffer this test used to ask for.  A host-resident weight makes
    // ggml_sycl_mul_mat_weight_resolves_to_host() report `resolved-host-handle`,
    // which routes MUL_MAT to CPU dispatch by design (host weights are faster on
    // the CPU than over PCIe).  The XMX unified kernel then never runs and the
    // comparison below degenerates to CPU-vs-CPU.
    ggml_backend_buffer_type_t weight_buft = dev_buft;

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

    auto finish = [&](run_status status) {
        for (ggml_backend_buffer_t buf : buffers) {
            ggml_backend_buffer_free(buf);
        }
        ggml_free(ctx);
        return status;
    };

    if (!alloc_or_fail(weight_buft, weight, GGML_BACKEND_BUFFER_USAGE_WEIGHTS) ||
        !alloc_or_fail(dev_buft, input, GGML_BACKEND_BUFFER_USAGE_COMPUTE) ||
        !alloc_or_fail(dev_buft, out_mat, GGML_BACKEND_BUFFER_USAGE_COMPUTE)) {
        std::fprintf(stderr, "FAIL: tensor buffer allocation failed\n");
        return finish(run_status::FAILED);
    }
    for (ggml_tensor * extra : extra_ops) {
        if (!alloc_or_fail(dev_buft, extra, GGML_BACKEND_BUFFER_USAGE_COMPUTE)) {
            std::fprintf(stderr, "FAIL: scale-chain buffer allocation failed\n");
            return finish(run_status::FAILED);
        }
    }

    ggml_backend_tensor_set(weight, tc.weights_q4.data(), 0, tc.weights_q4.size() * sizeof(block_q4_0));
    ggml_backend_tensor_set(input, tc.input_f32.data(), 0, tc.input_f32.size() * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, final_out);

    if (require_gpu_graph_path) {
        if (!ggml_sycl::test_backend_supports_graphs(backend)) {
            std::fprintf(stderr, "SKIP: backend does not support graphs\n");
            return finish(run_status::SKIPPED);
        }
        if (ggml_sycl::test_backend_graphs_disabled(backend)) {
            std::fprintf(stderr, "SKIP: backend graphs disabled by configuration\n");
            return finish(run_status::SKIPPED);
        }
    }

    // The backend's FIRST compute of any topology is a deliberate warmup pass
    // (ggml-sycl.cpp, warmup_n_nodes gate): it runs without graph recording to
    // populate the oneDNN primitive cache, and only a SECOND compute of the
    // same graph records an executable graph.  One compute can therefore never
    // satisfy the exec-graph precondition below -- run the graph twice, and
    // compare against the second run's output.
    for (int pass = 0; pass < 2; ++pass) {
        const ggml_status status = ggml_backend_graph_compute(backend, graph);
        if (status != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "FAIL: ggml_backend_graph_compute returned status=%d (pass %d)\n", (int) status, pass);
            return finish(run_status::FAILED);
        }
    }

    if (require_gpu_graph_path) {
        // A recorded executable graph is this test's proof that the MUL_MAT ran
        // on the DEVICE.  compute_forward suppresses CPU dispatch while a graph
        // is recording, so every node of a graph that exists executed on the
        // GPU -- which is the precondition the comparison below depends on and
        // which nothing here used to check.
        //
        // The previous gate counted graph-pinned weight leases instead.  That
        // count is populated only by graph_preload_weights(), which by design
        // skips device-VRAM weights ("weights already on GPU"), so it is zero on
        // a correct run of this fixture and never meant "the graph path ran".
        //
        // "Every node ran on the GPU" holds only because this graph is small.
        // The GPU-prefix/CPU-suffix split records a PARTIAL graph -- nodes
        // [0, first_cpu_node) -- and dispatches the tail on the CPU, so an exec
        // graph can exist while later nodes ran on the host.  That split is
        // taken only at first_cpu_node >= MIN_GPU_PREFIX_NODES (50); this
        // fixture is 13 nodes, well under it, so a graph here is always the
        // whole graph.  Re-check this if the fixture ever grows past 50 nodes.
        // The split path also needs GGML_SYCL_CPU_OFFLOAD=1 (off by default),
        // but rely on the node-count margin, not the env default: the invariant
        // is structural and the env var is not ours to depend on.
        if (!ggml_sycl::test_backend_has_exec_graph(backend)) {
            std::fprintf(stderr, "SKIP: backend recorded no executable graph; GPU XMX path not proven\n");
            return finish(run_status::SKIPPED);
        }
        std::fprintf(stderr, "INFO: executable graph recorded, replays=%llu\n",
                     (unsigned long long) ggml_sycl::test_backend_graph_replay_count(backend));
    }

    out.resize(static_cast<size_t>(tc.N) * tc.M);
    ggml_backend_tensor_get(final_out, out.data(), 0, out.size() * sizeof(float));

    return finish(run_status::OK);
}

// ctest's SKIP_RETURN_CODE: a skip must be visible AS a skip, so it can never be
// read as "this run verified the XMX path".
static const int TEST_SKIP_RC = 77;

int main() {
    // Enable XMX unified path BEFORE any can_use_xmx() checks.
    setenv("GGML_SYCL_XMX_UNIFIED", "1", 1);

    matmul_case tc{};
    build_case(tc);

    std::vector<float> cpu_out;
    {
        ggml_backend_t cpu_backend = ggml_backend_cpu_init();
        if (!cpu_backend) {
            std::fprintf(stderr, "SKIP: CPU backend unavailable; this run proves NOTHING about the XMX path\n");
            return TEST_SKIP_RC;
        }
        const run_status cpu_status = run_backend_matmul(cpu_backend, tc, false, cpu_out);
        ggml_backend_free(cpu_backend);
        if (cpu_status != run_status::OK) {
            std::fprintf(stderr, "FAIL: CPU reference did not run\n");
            return 1;
        }
    }

    ggml_backend_t sycl_backend = ggml_backend_sycl_init(0);
    if (!sycl_backend) {
        std::fprintf(stderr, "SKIP: SYCL backend unavailable; this run proves NOTHING about the XMX path\n");
        return TEST_SKIP_RC;
    }

    std::vector<float> sycl_out;
    const run_status   sycl_status = run_backend_matmul(sycl_backend, tc, true, sycl_out);
    ggml_backend_free(sycl_backend);
    if (sycl_status == run_status::SKIPPED) {
        // The reason was already printed at the point it was detected.  Exiting
        // 77 rather than 1 keeps an unavailable capability from masquerading as
        // a correctness failure -- and rather than 0, from masquerading as a pass.
        return TEST_SKIP_RC;
    }
    if (sycl_status != run_status::OK) {
        return 1;
    }

    float diff = 0.0f;
    if (!max_abs_diff(cpu_out, sycl_out, diff)) {
        std::fprintf(stderr, "FAIL: output size mismatch\n");
        return 1;
    }

    // XMX path dequantizes to half precision before joint_matrix, so allow
    // a slightly looser tolerance than the scalar reference.
    const float tol = 8e-2f;
    if (diff > tol) {
        std::fprintf(stderr, "FAIL: XMX unified mismatch diff=%g tol=%g\n", diff, tol);
        return 1;
    }

    std::fprintf(stderr, "PASS: XMX unified correctness diff=%g tol=%g\n", diff, tol);
    return 0;
}

#endif  // GGML_USE_SYCL
