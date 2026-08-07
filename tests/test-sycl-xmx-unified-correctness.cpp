// Regression test for XMX-enabled unified kernel correctness.
// Reproduces incorrect output when GGML_SYCL_XMX_UNIFIED=1 interacts
// with layout finalization and reordered XMX layouts.
//
// What a PASS asserts: a Q4_0 MUL_MAT at the medium (PP) bucket shape
// K=128 N=64 M=16, with the weight resident in the SYCL backend's own device
// buffer, matches the ggml CPU backend to within tol=8e-2 -- and the SYCL side
// ran on the device rather than being routed back to the CPU.  The second half
// is established by the two-part precondition in run_backend_matmul(); read
// that before weakening either half.  It does NOT assert which GPU kernel
// variant was selected; that is a function of GGML_SYCL_XMX_UNIFIED (set in
// main()) and can_use_xmx(M,N,K), which this shape satisfies (M>=8, N>=16,
// K%16==0).
//
// Exit codes are tri-state: 0 = the comparison ran and passed, 1 = a property
// this test asserts did not hold, 77 = a capability or configuration it needs
// is genuinely absent (ctest SKIP_RETURN_CODE).  Never collapse 77 into 0.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-sycl.h"
#include "ggml-quants.h"
// For ggml_sycl_weight_is_currently_device_resident() -- the exact predicate the
// production dispatcher consults before routing a MUL_MAT to the CPU.  Included
// the same way, and under the same target_include_directories /
// target_compile_definitions, as tests/test-dmmv-q4-0-coalesced.cpp
// (ggml/src/ggml-sycl/CMakeLists.txt: the two registrations are identical).
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/ggml-sycl-test.hpp"

#if !defined(GGML_USE_SYCL)
int main() {
    std::fprintf(stderr, "SKIP: GGML_USE_SYCL not enabled; this run proves NOTHING about the XMX path\n");
    return 77;  // ctest SKIP_RETURN_CODE -- a skip must not read as a pass
}
#else

// The SYCL device this test drives.  ggml_backend_sycl_init() takes the
// in-process device index -- the position AFTER ONEAPI_DEVICE_SELECTOR
// filtering, not a physical card id.  The ctest registration pins the selector
// to a single device, so this is that device on any host.
static const int SYCL_DEVICE_INDEX = 0;

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
// GGML_STATUS_SUCCESS.  SKIPPED means a capability or configuration the test
// needs is genuinely absent (ctest scores it skipped via exit 77); FAILED means
// a property the test asserts did not hold.
enum class run_status {
    OK,
    SKIPPED,
    FAILED,
};

static run_status run_backend_matmul(ggml_backend_t       backend,
                                     const matmul_case &  tc,
                                     bool                 require_gpu_execution,
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

    // Enlarge the graph past MIN_GRAPH_NODES so it takes the same graph-eligible
    // path production does.  13 nodes total (1 MUL_MAT + 12 SCALE).  The node
    // count is no longer load-bearing for the precondition -- that moved to
    // weight residency below -- so growing it is safe; it will still never
    // record, for the prompt-phase reason documented there.
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

    if (require_gpu_execution) {
        // ============== GPU-execution preconditions: the enumeration ==========
        // Enumerated at 9857619f3 over every site that can route a plain
        // GGML_OP_MUL_MAT to the CPU.  They fall into three groups; parts 1 and
        // 2 below close the first two, and the third closes on this fixture's
        // own shape and name.
        //
        // GROUP A -- downstream of ggml_sycl_cpu_offload_available(), which is
        // ggml_sycl_info().has_cpu_device, true only when GGML_SYCL_CPU_OFFLOAD=1
        // AND a CPU SYCL device exists.  Closed by part 1:
        //   * should_dispatch_to_cpu()'s `if (!cpu_offload_available) return
        //     false` early-out, and therefore its whole layer classifier, the
        //     name-less/host-buffer branch, and the batch-threshold cost model;
        //   * its preclassified fast path -- g_preclassified_cpu_flags is set
        //     only when ggml_sycl_cpu_offload_active_for_compute() holds, so it
        //     stays null and the branch is dead;
        //   * the GPU-prefix/CPU-suffix split in graph_compute(), whose entire
        //     block sits inside `if (cpu_offload_enabled() && has_cpu_device)`;
        //   * the HOST_COMPUTE batched CPU dispatch, which needs that same flag
        //     table AND GGML_SYCL_HOST_COMPUTE=1.
        //   This group also covers the 12 SCALE nodes: for a non-MUL_MAT op
        //   should_dispatch_to_cpu() routes only if the layer was ALREADY
        //   classified CPU, which only the classifier above can do.
        //
        // GROUP B -- ggml_sycl_mul_mat_weight_resolves_to_host(), which runs
        // ahead of the cpu-offload early-out and needs no CPU device at all.
        // Closed by part 2.
        //
        // GROUP C -- try_output_weight_cpu_vecdot() inside ggml_sycl_mul_mat().
        // Closed three times over by the fixture itself, no runtime check
        // needed: it requires GGML_SYCL_OUTPUT_WEIGHT_CPU_FALLBACK != 0 (default
        // 0), a src0 name containing "output.weight" (this weight is
        // "blk.0.attn_q.weight"), and src1->ne[1] == 1 (this is M = 16).
        //
        // Not reachable at all: the CPU routes in ggml_sycl_mul_mat_id(), since
        // this op is MUL_MAT and not MUL_MAT_ID.
        //
        // -------------------------------- part 1 --------------------------------
        // With the offload flag set, group A reopens and this test cannot
        // establish where the MUL_MAT ran.  That is a configuration the test
        // cannot see through, not a wrong answer -- exit 77, not 1.
        //
        // Be clear about what this does under ctest: nothing.  The registration
        // pins ONEAPI_DEVICE_SELECTOR to one level_zero device, and the CPU
        // queue is built by scanning the VISIBLE platforms for a CPU device, so
        // has_cpu_device is already false and group A is already closed.  This
        // check earns its place on a DIRECT invocation, where nothing pins the
        // selector.  To confirm it is not vacuous, make a CPU device visible:
        //   GGML_SYCL_CPU_OFFLOAD=1 GGML_SYCL_CPU_DEVICE_SELECTOR=opencl:cpu \
        //   ONEAPI_DEVICE_SELECTOR='level_zero:0;opencl:cpu' \
        //   ./build/bin/test-sycl-xmx-unified-correctness   # expect 77 + the SKIP
        if (ggml_backend_sycl_cpu_offload_available()) {
            std::fprintf(stderr,
                         "SKIP: SYCL CPU offload is available (GGML_SYCL_CPU_OFFLOAD); this run cannot prove the "
                         "MUL_MAT executed on the GPU\n");
            return finish(run_status::SKIPPED);
        }

        // -------------------------------- part 2 --------------------------------
        // Group B, and the one that catches the original hazard.
        // ggml_sycl_mul_mat_weight_resolves_to_host() returns false -- GPU -- at
        // its `if (ggml_sycl_weight_is_currently_device_resident(src0,
        // ctx.device)) return false` early-out.  The call below is that same
        // predicate, not a proxy for it.  Device residency also independently
        // closes the authoritative-residency branch further down
        // should_dispatch_to_cpu(): for a MUL_MAT it consults `weight_on_vram`,
        // the same query, and falls through to the GPU.  (That branch is doubly
        // dead here anyway -- it needs a non-empty placement plan, and this test
        // loads no model.)
        //
        // Positive control: this check is not vacuous.  Before llama.cpp-sfe9
        // this test allocated `weight` from the SYCL host-pinned buffer type; a
        // host-resident weight makes resolve report `resolved-host-handle`, the
        // dispatcher routes MUL_MAT to CPU by design, and the comparison below
        // degenerates to CPU-vs-CPU.  Restore that buffer type and this line
        // fires.  A FAIL (not a skip) is correct: the weight was allocated in
        // the backend's own device buffer, so failing to be device-resident is a
        // defect in the path under test.
        if (!ggml_sycl_weight_is_currently_device_resident(weight, SYCL_DEVICE_INDEX)) {
            std::fprintf(stderr,
                         "FAIL: weight '%s' is not device-resident on device %d before compute; MUL_MAT would be "
                         "routed to CPU and the comparison would be CPU-vs-CPU\n",
                         weight->name, SYCL_DEVICE_INDEX);
            return finish(run_status::FAILED);
        }
    }

    // The backend's FIRST compute of any topology is a deliberate warmup pass
    // (ggml-sycl.cpp, warmup_n_nodes gate): it runs without graph recording, to
    // populate the oneDNN primitive cache.  Compare against the SECOND pass so
    // the numbers come from the steady-state dispatch production sees, not from
    // the warmup.  (This loop was added to reach the exec-graph precondition
    // that used to sit below; that gate is gone, but the warmup reason stands on
    // its own.)
    for (int pass = 0; pass < 2; ++pass) {
        const ggml_status status = ggml_backend_graph_compute(backend, graph);
        if (status != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "FAIL: ggml_backend_graph_compute returned status=%d (pass %d)\n", (int) status, pass);
            return finish(run_status::FAILED);
        }
    }

    if (require_gpu_execution) {
        // The residency gate above is a precondition; re-check it here so the
        // assertion covers BOTH passes.  The output read below is the second
        // pass's, and a weight evicted to host between the two would have let
        // that pass take the very CPU route part 2 excludes.
        if (!ggml_sycl_weight_is_currently_device_resident(weight, SYCL_DEVICE_INDEX)) {
            std::fprintf(stderr,
                         "FAIL: weight '%s' stopped being device-resident on device %d during the run; the second "
                         "pass may have been routed to CPU\n",
                         weight->name, SYCL_DEVICE_INDEX);
            return finish(run_status::FAILED);
        }

        // ============ Why the exec graph is INFORMATIONAL, not a gate ============
        // A recorded executable graph would be attractive proof of device
        // execution (compute_forward suppresses CPU dispatch while recording:
        // ggml-sycl.cpp `!ggml_sycl_graph_dispatch_recording_active(&ctx) &&
        // should_dispatch_to_cpu(...)`).  It is also UNREACHABLE for this
        // fixture, twice over, and both were measured at 9857619f3 -- do not
        // restore it as a SKIP/FAIL condition:
        //
        //  1. Recording is DECODE-ONLY by policy.  graph_compute() computes
        //     `cached_is_decode = (first MUL_MAT's src[1]->ne[1] == 1)` and the
        //     `else if (!is_decode_phase)` branch logs
        //         "[SYCL-GRAPH] skipping graph recording for PP phase"
        //     (oneDNN primitive.execute() in the FP16 PP path is incompatible
        //     with Level Zero graph recording).  This fixture is prompt-phase by
        //     construction: src[1]->ne[1] is M = 16.  It MUST be -- the XMX/DPAS
        //     unified kernel is the medium-bucket PP path, and an M = 1 decode
        //     shape would dispatch MMVQ/DMMV instead, missing the kernel under
        //     test.  So exec-graph proof and XMX execution are mutually
        //     exclusive by dispatch policy; no fixture can have both.
        //     Single-device call #2 first logs
        //         "[SYCL-GRAPH-STATE] no executable graph cached, will record"
        //     and then hits the PP-phase skip anyway.
        //  2. Multi-device never records at all.  Under
        //     ONEAPI_DEVICE_SELECTOR=level_zero:0,1 graph_compute() logs
        //         "[SYCL] Multi-device SYCL execution active: disabling full
        //          graph replay ..."
        //     whenever ggml_sycl_routable_device_count() > 1.  The ctest
        //     registration pins a single device, which retires this one -- but
        //     it returns the moment someone runs the binary by hand without the
        //     pin, so it is recorded here rather than in CMake alone.
        //
        // An earlier gate counted graph-pinned weight leases instead.  That
        // count is populated only by graph_preload_weights(), which by design
        // skips device-VRAM weights ("weights already on GPU"), so it is zero on
        // a correct run of this fixture and never meant "the graph path ran".
        std::fprintf(stderr,
                     "INFO: graphs supported=%d disabled=%d exec_graph=%d replays=%llu (informational; PP "
                     "phase never records)\n",
                     ggml_sycl::test_backend_supports_graphs(backend) ? 1 : 0,
                     ggml_sycl::test_backend_graphs_disabled(backend) ? 1 : 0,
                     ggml_sycl::test_backend_has_exec_graph(backend) ? 1 : 0,
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

    ggml_backend_t sycl_backend = ggml_backend_sycl_init(SYCL_DEVICE_INDEX);
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
