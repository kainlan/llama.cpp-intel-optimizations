// Regression test for XMX-enabled unified kernel correctness.
// Reproduces incorrect output when GGML_SYCL_XMX_UNIFIED=1 interacts
// with layout finalization and reordered XMX layouts.
//
// What a PASS asserts: a Q4_0 MUL_MAT at the medium (PP) bucket shape
// K=128 N=64 M=16, with the weight resident in the SYCL backend's own device
// buffer, matches the ggml CPU backend to within tol=8e-2 -- and the SYCL side
// ran on the device rather than being routed back to the CPU.  The second half
// is established by the four-part precondition in run_backend_matmul(), backed
// by the static_asserts on the CASE_* shape constants below; read the
// four-group enumeration there before weakening any of it.  It does NOT assert
// which GPU kernel variant was selected; that is a function of
// GGML_SYCL_XMX_UNIFIED (set in main()) and can_use_xmx(M,N,K), whose
// thresholds the CASE_* static_asserts pin.
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

// The fixture shape.  These are compile-time constants because three separate
// dispatch decisions hang off them, and a future edit to any one of the numbers
// must re-trigger the analysis rather than silently move the test off the path
// it names.  See the static_asserts below for what each one is holding shut.
static constexpr int CASE_K = 128;  // reduction dim
static constexpr int CASE_N = 64;   // output columns (weight rows)
static constexpr int CASE_M = 16;   // batch/output rows (tokens); MEDIUM bucket -> dpas + XMX

// Quantization: the weight is built with quantize_row_q4_0_ref, which requires
// whole Q4_0 blocks.  This was a runtime `SKIP` + std::abort(), which exits 134
// while printing the word SKIP -- a skip that ctest scores as a crash.  It is a
// property of a literal, so the compiler can settle it.
static_assert(CASE_K % QK4_0 == 0, "CASE_K must be a whole number of Q4_0 blocks");

// XMX eligibility, mirroring can_use_xmx(M, N, K) in unified-kernel.hpp: M >=
// XMX_TILE_M (8), N >= XMX_TILE_N (16), K % XMX_TILE_K (16) == 0.  The argument
// mapping is UnifiedKernelArgs (unified-kernel.hpp:1405-1407): M = src1->ne[1],
// N = src0->ne[1], K = src0->ne[0] -- the same roles as the fields here.
static_assert(CASE_M >= 8, "CASE_M below XMX_TILE_M: shape is no longer XMX-eligible");
static_assert(CASE_N >= 16, "CASE_N below XMX_TILE_N: shape is no longer XMX-eligible");
static_assert(CASE_K % 16 == 0, "CASE_K not a multiple of XMX_TILE_K: shape is no longer XMX-eligible");

// Group D closures -- these two are load-bearing for the GPU-execution claim,
// not merely for kernel selection.  Both CPU short-circuits inside
// ggml_sycl_mul_mat() key off src1->ne[1], which is CASE_M:
//   D1 CPU-HOST-MAT (ggml-sycl.cpp:54774-54840) fires only at src1->ne[1] == 1.
//      It is DEFAULT-ACTIVE (opt-out GGML_SYCL_CPU_HOST_MAT=0), so this bound is
//      the only unconditional thing standing between this fixture and it.
//   D2 CPU-PP (ggml-sycl.cpp:54850-54966) fires only at src1->ne[1] >=
//      GGML_SYCL_CPU_PP_MIN_BATCH (default 64).  That default is env-tunable, so
//      this assert is a guard on the DEFAULT configuration only; D2's
//      unconditional closure is its residency gate, asserted at runtime below.
// A shape change that trips either assert must go re-read GROUP D.
static_assert(CASE_M != 1, "CASE_M == 1 opens the default-active CPU-HOST-MAT path (see GROUP D)");
static_assert(CASE_M < 64, "CASE_M >= 64 opens the CPU-PP path at its default threshold (see GROUP D)");

struct matmul_case {
    int K;  // reduction dim
    int N;  // output columns (weight rows)
    int M;  // batch/output rows (tokens)
    std::vector<block_q4_0> weights_q4;
    std::vector<float>      input_f32;
};

static void build_case(matmul_case & tc) {
    tc.K = CASE_K;
    tc.N = CASE_N;
    tc.M = CASE_M;

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
        // GGML_OP_MUL_MAT to the CPU.  They fall into FOUR groups.
        //
        // Read this before adding or weakening a check, and note what the
        // grouping is really tracking: each group is closed by a DIFFERENT
        // residency mechanism, and they do not agree with each other.  Four
        // distinct predicates appear below --
        //   ggml_sycl_weight_is_currently_device_resident() (common.hpp:4546),
        //   ggml_sycl_weight_is_planned_on_host()           (common.hpp:4499),
        //   ggml_sycl_is_host_resident_weight()             (mmvq.cpp:2115),
        //   ggml_backend_buffer_is_host()                   (ggml-backend)
        // -- and asserting one of them says nothing about the others.  An
        // earlier revision of this test asserted only the first and treated the
        // rest as covered; they were not.
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
        // GROUP D -- two CPU short-circuits INSIDE ggml_sycl_mul_mat() itself,
        // each with its own `return` before any GPU kernel is submitted.  They
        // sit upstream of nothing in groups A-C and are gated on residency
        // mechanisms DIFFERENT from the one part 2 asserts:
        //   D1 CPU-HOST-MAT (ggml-sycl.cpp:54774-54840).  DEFAULT-ACTIVE -- it
        //     is an opt-OUT (GGML_SYCL_CPU_HOST_MAT=0), not an opt-in, so no env
        //     default is protecting us here.  Gates: !src0_on_device (where
        //     src0_on_device = !ggml_sycl_is_host_resident_weight(src0,
        //     ctx.stream()), the mmvq.cpp:2115 probe), src0_planned_host (=
        //     ggml_sycl_weight_is_planned_on_host(), the PLACEMENT PLAN -- a
        //     third mechanism again), src1->ne[1] == 1, and quantized f32.
        //     Closed three independent ways, all now asserted or static_asserted:
        //     the CASE_M != 1 static_assert, the planned-host check in part 3,
        //     and the residency checks in part 4.
        //   D2 CPU-PP (ggml-sycl.cpp:54850-54966).  Opt-in
        //     (GGML_SYCL_CPU_PP=1).  Gates: M >= GGML_SYCL_CPU_PP_MIN_BATCH
        //     (default 64) and ggml_sycl_is_host_resident_weight(src0, stream).
        //     Note the threshold is ENV-TUNABLE, so "M < 64" is a guard on the
        //     default configuration only -- D2's unconditional closure is the
        //     residency gate, which part 4 asserts.
        //
        // Not reachable at all: the CPU routes in ggml_sycl_mul_mat_id(), since
        // this op is MUL_MAT and not MUL_MAT_ID.
        //
        // -------------------------------- part 1 --------------------------------
        // With the offload flag set, group A reopens and this test cannot
        // establish where the MUL_MAT ran.  That is a configuration the test
        // cannot see through, not a wrong answer -- exit 77, not 1.
        //
        // Be clear about what this does under ctest: nothing, and about WHY.
        // The primary closure is that GGML_SYCL_CPU_OFFLOAD is unset --
        // ggml_sycl_cpu_offload_enabled() is false, so the CPU queue is never
        // even attempted and has_cpu_device stays false.  That holds on any
        // host, with or without a selector.  Selector pinning is a SECONDARY,
        // belt-and-braces closure: even with the env var set, the CPU queue is
        // built by scanning the VISIBLE platforms for a CPU device, and the
        // registration pins ONEAPI_DEVICE_SELECTOR to one level_zero device.
        // This check therefore earns its place on a DIRECT invocation with the
        // env var set.  To confirm it is not vacuous you must defeat BOTH
        // closures -- set the var AND make a CPU device visible:
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

        // -------------------------------- part 3 --------------------------------
        // Group D1's placement-plan gate.  This is the SECOND residency
        // mechanism and part 2 does not imply it: a weight can be VRAM-resident
        // right now and still be planned onto the host, which is exactly the
        // combination D1 keys on.  The call below is D1's own `src0_planned_host`
        // expression, not a proxy.
        //
        // It holds today because this test loads no model, so the placement plan
        // has no entries and ggml_sycl_get_planned_weight_residency() answers
        // UNKNOWN.  That is a property of the fixture, not of the backend, so it
        // is asserted rather than assumed -- a future test that stages a model
        // (as tests/test-dmmv-q4-0-coalesced.cpp now does) would populate a plan
        // and could land here without anyone noticing.
        if (ggml_sycl_weight_is_planned_on_host(weight, SYCL_DEVICE_INDEX)) {
            std::fprintf(stderr,
                         "FAIL: weight '%s' is planned onto the host on device %d; the default-active CPU-HOST-MAT "
                         "path (GROUP D1) is open\n",
                         weight->name, SYCL_DEVICE_INDEX);
            return finish(run_status::FAILED);
        }

        // -------------------------------- part 4 --------------------------------
        // The residency gate shared by D1 (via !src0_on_device) and D2, which is
        // ggml_sycl_is_host_resident_weight() at mmvq.cpp:2115.  That function is
        // `static` in mmvq.cpp and so is NOT callable from here -- this is a
        // MIRROR of its two checks, not the function itself.  If it ever grows a
        // third check, this mirror silently stops covering the whole predicate;
        // it will not start passing wrongly, but it will cover less.  Keep them
        // in step.
        //
        //   its check 1: ggml_backend_buffer_is_host(src0->buffer)
        //   its check 2: ggml_sycl_host_data(src0) is non-null AND its USM
        //                allocation type is not `device`
        //
        // Both must be false for the weight to count as device-resident there.
        if (ggml_backend_buffer_is_host(weight->buffer)) {
            std::fprintf(stderr,
                         "FAIL: weight '%s' sits in a host buffer; CPU-HOST-MAT/CPU-PP (GROUP D) see it as "
                         "host-resident\n",
                         weight->name);
            return finish(run_status::FAILED);
        }

        // Check 2 folds two distinct causes into one answer, and the message
        // below separates them because they need different responses.
        // ggml_sycl_get_alloc_type() is an alloc_registry lookup that returns
        // `unknown` for an UNREGISTERED pointer as well as for a genuinely
        // non-USM one (alloc-registry.hpp; interior pointers into a registered
        // range do resolve, so an offset into the buffer's arena is fine).
        // A device-resident weight whose arena was never registered would land
        // here reporting `unknown`.  That is still worth failing on rather than
        // ignoring -- production asks this exact question and would answer
        // "host-resident", opening D1's !src0_on_device gate -- but it is a
        // registration defect, not a wrong-buffer defect, so say which.
        const void *           weight_host_data = ggml_sycl_host_data(weight);
        const sycl::usm::alloc weight_alloc =
            weight_host_data ? ggml_sycl_get_alloc_type(weight_host_data) : sycl::usm::alloc::unknown;
        if (weight_host_data != nullptr && weight_alloc != sycl::usm::alloc::device) {
            std::fprintf(stderr,
                         "FAIL: weight '%s' data pointer resolves to USM alloc kind %d, not `device` (%d); "
                         "CPU-HOST-MAT/CPU-PP (GROUP D) see it as host-resident. kind `unknown` (%d) means the "
                         "pointer is non-USM OR its allocation was never registered in alloc_registry -- check "
                         "which before assuming the weight is misplaced\n",
                         weight->name, (int) weight_alloc, (int) sycl::usm::alloc::device,
                         (int) sycl::usm::alloc::unknown);
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
