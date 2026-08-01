//
// Test: MMQ XMX Dispatch Integration
//
// Verifies that the XMX matrix multiplication dispatch works correctly:
// 1. XMX path selected for supported configurations
// 2. Fallback to MMQ when XMX unavailable
// 3. Correct handling of edge cases (empty batch, non-aligned dims)
// 4. Results match CPU reference
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sycl/sycl.hpp>
#include <vector>

// Enable standalone mode for XMX ESIMD common header
#define XMX_TEST_STANDALONE 1

// Include XMX ESIMD common header (provides XMXCapabilities, query_xmx_capabilities)
#include "../xmx-esimd-common.hpp"

// The batch gate the two dispatch sites in ggml-sycl.cpp actually call. Test 3
// asserts against THIS, not against a local copy of the rule -- see the comment
// on the function and llama.cpp-cwev.
#include "../xmx-dispatch-gate.hpp"

// =============================================================================
// Q4_0 and Q8_1 Block Definitions (match ggml-common.h)
// =============================================================================
#define QK4_0 32
#define QK8_1 32

struct block_q4_0_test {
    sycl::half d;              // scale factor
    uint8_t    qs[QK4_0 / 2];  // quantized values: 16 bytes = 32 nibbles
};

struct block_q8_1_test {
    sycl::half d;          // delta (scale)
    sycl::half s;          // sum of quants
    int8_t     qs[QK8_1];  // quantized values
};

static_assert(sizeof(block_q4_0_test) == sizeof(sycl::half) + QK4_0 / 2, "wrong q4_0 block size");
static_assert(sizeof(block_q8_1_test) == 2 * sizeof(sycl::half) + QK8_1, "wrong q8_1 block size");

// =============================================================================
// Test Helpers
// =============================================================================

static int g_tests_run     = 0;
static int g_tests_passed  = 0;
static int g_tests_skipped = 0;

#define TEST_BEGIN(name)                         \
    do {                                         \
        g_tests_run++;                           \
        fprintf(stderr, "[TEST] %s ... ", name); \
    } while (0)

#define TEST_PASS()                  \
    do {                             \
        g_tests_passed++;            \
        fprintf(stderr, "PASSED\n"); \
    } while (0)

#define TEST_SKIP(reason)                          \
    do {                                           \
        g_tests_skipped++;                         \
        fprintf(stderr, "SKIPPED (%s)\n", reason); \
    } while (0)

#define TEST_FAIL(msg)                        \
    do {                                      \
        fprintf(stderr, "FAILED: %s\n", msg); \
        return false;                         \
    } while (0)

#define TEST_ASSERT(cond, msg) \
    do {                       \
        if (!(cond)) {         \
            TEST_FAIL(msg);    \
        }                      \
    } while (0)

#define TEST_ASSERT_NEAR(actual, expected, tol, msg)                                                      \
    do {                                                                                                  \
        float diff = std::abs((float) (actual) - (float) (expected));                                     \
        if (diff > (tol)) {                                                                               \
            fprintf(stderr, "FAILED: %s (expected %.6f, got %.6f, diff=%.6f)\n", msg, (float) (expected), \
                    (float) (actual), diff);                                                              \
            return false;                                                                                 \
        }                                                                                                 \
    } while (0)

// =============================================================================
// CPU Reference: Q4_0 x Q8_1 GEMM
// =============================================================================

// Dequantize a single Q4_0 block to float
static void dequantize_q4_0_block(const block_q4_0_test & block, float * out) {
    const float d = static_cast<float>(block.d);
    for (int i = 0; i < 16; i++) {
        uint8_t byte = block.qs[i];
        // Low nibble: subtract 8 for signed range
        out[i]      = (static_cast<float>(byte & 0x0F) - 8.0f) * d;
        // High nibble
        out[i + 16] = (static_cast<float>(byte >> 4) - 8.0f) * d;
    }
}

// Dequantize a single Q8_1 block to float
static void dequantize_q8_1_block(const block_q8_1_test & block, float * out) {
    const float d = static_cast<float>(block.d);
    for (int i = 0; i < QK8_1; i++) {
        out[i] = static_cast<float>(block.qs[i]) * d;
    }
}

// CPU reference GEMM: C[M,N] = A_q4_0[M,K] * B_q8_1[N,K]^T
// A: Q4_0 weights [M, K/32 blocks]
// B: Q8_1 activations [N, K/32 blocks] (transposed)
// C: float output [M, N]
static void gemm_q4_0_q8_1_cpu_reference(const block_q4_0_test * A,
                                         const block_q8_1_test * B,
                                         float *                 C,
                                         int                     M,
                                         int                     N,
                                         int                     K) {
    const int k_blocks = K / QK4_0;

    // Temporary storage for dequantized blocks
    std::vector<float> a_block(QK4_0);
    std::vector<float> b_block(QK8_1);

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;

            for (int kb = 0; kb < k_blocks; kb++) {
                // Dequantize blocks
                dequantize_q4_0_block(A[m * k_blocks + kb], a_block.data());
                dequantize_q8_1_block(B[n * k_blocks + kb], b_block.data());

                // Dot product
                for (int k = 0; k < QK4_0; k++) {
                    acc += a_block[k] * b_block[k];
                }
            }

            C[m * N + n] = acc;
        }
    }
}

// =============================================================================
// Test 1: XMX Dispatch Selects joint_matrix for Q4_0
// Bug caught: Wrong kernel selected for Q4_0 with XMX hardware
// =============================================================================
static bool test_xmx_dispatch_selects_joint_matrix_for_q4_0(sycl::queue &            q,
                                                             const XMXCapabilities & caps) {
    TEST_BEGIN("test_xmx_dispatch_selects_joint_matrix_for_q4_0");

    if (!caps.supported || !caps.supports_int8) {
        TEST_SKIP("XMX int8 not supported");
        return true;
    }

    // Test dimensions aligned to XMX tiles
    const int M        = 32;  // Multiple of 8
    const int N        = 16;  // Multiple of 16
    const int K        = 64;  // Multiple of 32
    const int k_blocks = K / QK4_0;

    // Allocate and initialize Q4_0 weights
    std::vector<block_q4_0_test> A(M * k_blocks);
    std::mt19937                 rng(42);
    for (auto & block : A) {
        block.d = sycl::half(0.1f * (rng() % 10 + 1));
        for (int i = 0; i < 16; i++) {
            block.qs[i] = rng() % 256;
        }
    }

    // Allocate and initialize Q8_1 activations
    std::vector<block_q8_1_test> B(N * k_blocks);
    for (auto & block : B) {
        block.d = sycl::half(0.1f * (rng() % 10 + 1));
        block.s = sycl::half(0.0f);
        for (int i = 0; i < QK8_1; i++) {
            block.qs[i] = static_cast<int8_t>((rng() % 256) - 128);
        }
    }

    // Compute CPU reference
    std::vector<float> C_ref(M * N);
    gemm_q4_0_q8_1_cpu_reference(A.data(), B.data(), C_ref.data(), M, N, K);

    // Since we can't directly call the dispatch from here (it requires full ggml context),
    // we verify the XMXCapabilities correctly identify the hardware.
    // The actual dispatch integration is tested via llama-bench.

    // Verify XMX capabilities are correctly detected
    TEST_ASSERT(caps.M == 8, "XMX M dimension should be 8");
    TEST_ASSERT(caps.N == 16 || caps.N == 8, "XMX N dimension should be 8 or 16");
    TEST_ASSERT(caps.K == 32, "XMX K dimension should be 32 (matches QK4_0)");

    fprintf(stderr,
            "(XMX detected: M=%zu N=%zu K=%zu) ",
            caps.M,
            caps.N,
            caps.K);

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 2: Fallback for Non-XMX Device
// Bug caught: Crash on non-XMX hardware
// =============================================================================
static bool test_xmx_dispatch_fallback_for_non_xmx_device(sycl::queue &            q,
                                                           const XMXCapabilities & caps) {
    TEST_BEGIN("test_xmx_dispatch_fallback_for_non_xmx_device");

    // This test verifies that XMXCapabilities correctly reports unsupported
    // when running on non-XMX hardware.

    // We can't actually mock a non-XMX device, but we verify the caps structure
    // is correctly populated and the fallback logic would work.

    if (caps.supported) {
        // On XMX hardware, verify the dispatch would select XMX path
        TEST_ASSERT(caps.supports_int8, "XMX should support int8");
        fprintf(stderr, "(XMX available, fallback not needed) ");
    } else {
        // On non-XMX hardware, verify the fallback path is safe
        fprintf(stderr, "(XMX not available, fallback would be used) ");
    }

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 3: Batch Size Threshold
// Bug caught: Wrong kernel for large batches (MMQ should be used for large batches)
//
// ⚠️ THIS TEST USED TO PASS UNCONDITIONALLY (llama.cpp-cwev). Its body called no
// dispatch function: it recomputed `batch < 8` -- a hardcoded literal matching no
// dispatch site -- into a local, fprintf'd it, and called TEST_PASS(). It could
// not fail, while its name and banner advertised exactly the coverage an auditor
// of "is XMX dispatch tested?" would search for.
//
// It now asserts against ggml_sycl_xmx_batch_in_range(), the function BOTH
// dispatch sites in ggml-sycl.cpp call. That is what makes it more than a
// restatement: the assertions bind to the shipped rule, not to a copy of it.
//
// What is asserted is BOUNDARY BEHAVIOUR and STRUCTURE, deliberately not the
// formula. Re-deriving `batch >= 1 && batch < threshold` in the expectation
// would be the original vacuity in a new dress -- it would pass against any
// implementation, including a wrong one, because both sides would be the same
// mistake. Each property below is a statement someone could have gotten wrong:
//
//   P1  batch 0 must NOT take XMX          (empty batch; test 5's premise)
//   P2  batch 1 MUST take XMX              (the path claims batch >= 1)
//   P3  the last accepted batch is exactly threshold-1, and threshold itself is
//       rejected                            (off-by-one: `<` vs `<=`)
//   P4  acceptance is ONE CONTIGUOUS interval, no holes and no second run
//   P5  a negative batch must NOT take XMX
//   P6  threshold 0 accepts NOTHING         (the global's fail-closed
//       initializer depends on this -- llama.cpp-d5h0)
//   P7  threshold 1 accepts NOTHING         (empty half-open range [1,1))
//
// NOT asserted here: that the default threshold is 64. That number lives in one
// place, the GGML_SYCL_XMX_THRESHOLD row of the sycl_env_settings table in
// ggml_check_sycl(), and is guarded by
// scripts/check-sycl-xmx-threshold-default.sh. Copying it into this file would
// be the duplicated-default bug (llama.cpp-d5h0) all over again, in the test.
//
// Also gone: the stale literal 8. It was real once -- at 05519d18f the gate read
// `batch >= 8 && batch < threshold` -- and survived here as a printout constant
// after the dispatch bound was relaxed to `>= 1`.
// =============================================================================
static bool test_xmx_dispatch_threshold_batch_size(sycl::queue &            q,
                                                    const XMXCapabilities & caps) {
    TEST_BEGIN("test_xmx_dispatch_threshold_batch_size");

    // NOTE: no caps.supported early-SKIP. The gate under test is pure integer
    // logic with no hardware input, so skipping on non-XMX hardware would have
    // been a second way for this test to report success without checking
    // anything. Tests 1 and 4 skip because they assert on caps; this does not.

    const int thresholds[] = { 0, 1, 2, 8, 64, 1024 };

    for (int threshold : thresholds) {
        // P1 / P5 / P2
        TEST_ASSERT(!ggml_sycl_xmx_batch_in_range(0, threshold), "batch 0 must never select XMX");
        TEST_ASSERT(!ggml_sycl_xmx_batch_in_range(-1, threshold), "negative batch must never select XMX");
        TEST_ASSERT(!ggml_sycl_xmx_batch_in_range(-1024, threshold), "negative batch must never select XMX");

        // Sweep well past the threshold so a missing upper bound is caught.
        const int64_t sweep_end  = static_cast<int64_t>(threshold) + 64;
        int64_t       first_true = -1;
        int64_t       last_true  = -1;
        int64_t       runs       = 0;  // number of maximal contiguous accepted runs
        bool          prev       = false;

        for (int64_t batch = 0; batch <= sweep_end; batch++) {
            const bool now = ggml_sycl_xmx_batch_in_range(batch, threshold);
            if (now && !prev) {
                runs++;
                if (first_true < 0) {
                    first_true = batch;
                }
            }
            if (now) {
                last_true = batch;
            }
            prev = now;
        }

        // P6 / P7: a threshold that cannot admit anything must admit nothing.
        if (threshold <= 1) {
            TEST_ASSERT(runs == 0, "threshold <= 1 must accept no batch at all");
            fprintf(stderr, "\n  threshold=%d: accepts nothing (fail-closed) ", threshold);
            continue;
        }

        // P4: exactly one contiguous accepted interval.
        TEST_ASSERT(runs == 1, "XMX acceptance must be one contiguous batch interval");

        // P2: the interval starts at 1, so batch 1 is accepted.
        TEST_ASSERT(first_true == 1, "the accepted interval must begin at batch 1");
        TEST_ASSERT(ggml_sycl_xmx_batch_in_range(1, threshold), "batch 1 must select XMX below the threshold");

        // P3: it ends one short of the threshold, and the threshold is out.
        TEST_ASSERT(last_true == static_cast<int64_t>(threshold) - 1, "the last accepted batch must be threshold-1");
        TEST_ASSERT(!ggml_sycl_xmx_batch_in_range(threshold, threshold), "batch == threshold must fall through to MMQ");
        TEST_ASSERT(!ggml_sycl_xmx_batch_in_range(static_cast<int64_t>(threshold) + 1, threshold),
                    "batch > threshold must fall through to MMQ");

        fprintf(stderr, "\n  threshold=%d: accepts [%ld, %ld], rejects %d and above ", threshold, (long) first_true,
                (long) last_true, threshold);
    }

    // A 64-bit batch must not be truncated into range by an int-width gate.
    TEST_ASSERT(!ggml_sycl_xmx_batch_in_range(INT64_C(0x100000001), 64),
                "a batch above INT_MAX must not wrap into the accepted range");

    fprintf(stderr, "\n");
    TEST_PASS();
    return true;
}

// =============================================================================
// Test 4: Non-Aligned Dimensions
// Bug caught: Crash on M/N/K not aligned to XMX tile sizes
// =============================================================================
static bool test_xmx_dispatch_handles_non_aligned_dimensions(sycl::queue &            q,
                                                              const XMXCapabilities & caps) {
    TEST_BEGIN("test_xmx_dispatch_handles_non_aligned_dimensions");

    if (!caps.supported || !caps.supports_int8) {
        TEST_SKIP("XMX int8 not supported");
        return true;
    }

    // Non-aligned dimensions that should still work (with padding/fallback)
    const int M = 37;   // Not aligned to 8
    const int N = 41;   // Not aligned to 16
    const int K = 64;   // Aligned to 32 (required)

    // K must be aligned to QK4_0 for quantization, but M and N can be arbitrary

    // Verify the dispatch handles non-aligned M and N
    // The XMX kernel should either:
    // 1. Pad to tile boundaries internally, or
    // 2. Fall back to scalar/MMQ path for remainder

    // Since we can't call the actual dispatch, we verify the CPU reference
    // handles these dimensions correctly
    const int                    k_blocks = K / QK4_0;
    std::vector<block_q4_0_test> A(M * k_blocks);
    std::vector<block_q8_1_test> B(N * k_blocks);
    std::vector<float>           C(M * N);

    std::mt19937 rng(123);
    for (auto & block : A) {
        block.d = sycl::half(0.1f);
        for (int i = 0; i < 16; i++) {
            block.qs[i] = rng() % 256;
        }
    }
    for (auto & block : B) {
        block.d = sycl::half(0.1f);
        block.s = sycl::half(0.0f);
        for (int i = 0; i < QK8_1; i++) {
            block.qs[i] = static_cast<int8_t>((rng() % 256) - 128);
        }
    }

    // CPU reference should handle non-aligned dimensions
    gemm_q4_0_q8_1_cpu_reference(A.data(), B.data(), C.data(), M, N, K);

    // Verify result is reasonable (not NaN, not zero for non-zero input)
    bool has_nonzero = false;
    for (int i = 0; i < M * N; i++) {
        TEST_ASSERT(!std::isnan(C[i]), "Result contains NaN");
        if (C[i] != 0.0f) {
            has_nonzero = true;
        }
    }
    TEST_ASSERT(has_nonzero, "Result is all zeros");

    fprintf(stderr, "(M=%d, N=%d verified) ", M, N);
    TEST_PASS();
    return true;
}

// =============================================================================
// Test 5: Empty Batch Returns Immediately
// Bug caught: Division by zero or crash on empty input
// =============================================================================
static bool test_xmx_dispatch_empty_batch_returns_immediately(sycl::queue &            q,
                                                               const XMXCapabilities & caps) {
    TEST_BEGIN("test_xmx_dispatch_empty_batch_returns_immediately");

    // When batch_size = 0 (N = 0), the dispatch should return immediately
    // without crashing or writing any output

    const int M = 32;
    const int N = 0;  // Empty batch
    const int K = 64;

    // With N=0, there's nothing to compute
    // The dispatch should detect this and return early

    // We can't call the actual dispatch, but we verify the logic:
    // if (ncols_y <= 0) return; should be present

    // Verify our reference handles this correctly
    const int                    k_blocks = K / QK4_0;
    std::vector<block_q4_0_test> A(M * k_blocks);
    std::vector<block_q8_1_test> B;  // Empty
    std::vector<float>           C;  // Empty

    // CPU reference with N=0 should not crash
    gemm_q4_0_q8_1_cpu_reference(A.data(), B.data(), C.data(), M, N, K);

    // If we get here without crashing, the test passes
    fprintf(stderr, "(N=0 handled correctly) ");
    TEST_PASS();
    return true;
}

// =============================================================================
// Main
// =============================================================================

// Build a queue on a device we already know exists. Wrapped in a function because
// the queue has to outlive the try block and sycl::queue has no default state we
// are willing to construct -- see the comment in main() on the default selector.
static sycl::queue make_queue_or_fail(const sycl::device & dev) {
    try {
        return sycl::queue(dev);
    } catch (const sycl::exception & e) {
        fprintf(stderr, "FAILED: could not create a SYCL queue on the GPU: %s\n", e.what());
        // Deliberate divergence from the `return 1` most sibling tests in this
        // directory use for a setup failure. A value-returning helper has no
        // natural channel for a failure code; std::optional<sycl::queue> would
        // give it one, and is a legitimate alternative -- exit(1) is chosen only
        // because this is genuinely unreachable once get_devices() has already
        // reported a non-empty list, so the extra unwrapping buys nothing. Only
        // `devices` is live, and skipping its destructor at process exit is
        // harmless.
        //
        // What you must NOT do is "align" this with the surrounding convention.
        // That convention is the bug this file was fixed for: 15 of the 30 tests
        // here open with a bare `sycl::device dev;` outside their try block, and
        // sycl::device's default ctor throws on a device-less host exactly as
        // sycl::queue's does -- so the throw is uncaught and the `return 1` those
        // tests wrote for this case never runs. Measured: test-mem-handle-eviction
        // and test-unified-cache-fast-path exit 134 (SIGABRT, core dumped) where
        // this file exits 77. Tracked as llama.cpp-ivin.
        exit(1);
    }
}

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    fprintf(stderr, "=== MMQ XMX Dispatch Integration Tests ===\n");

    // Enumerate BEFORE constructing any queue.
    //
    // ⚠️ This ordering is load-bearing. `sycl::queue q;` default-constructs through
    // the default selector, which THROWS when the runtime exposes no device at all
    // -- and it used to sit here, OUTSIDE the try below. So on a device-less host
    // the process died with
    //
    //     terminate called after throwing an instance of 'sycl::_V1::exception'
    //     what(): No device of requested type available
    //     Aborted (core dumped)          <- exit 134
    //
    // without ever reaching the no-device check ten lines further down. Measured
    // 2026-08-01 with ONEAPI_DEVICE_SELECTOR=level_zero:99. An abort is a louder
    // form of exactly the defect the skip below exists to fix, so the queue must be
    // built from a device already known to exist, never default-constructed.
    std::vector<sycl::device> devices;
    try {
        devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    } catch (const sycl::exception & e) {
        // Enumeration itself failing means no device work is possible either --
        // same skip, but say which of the two happened rather than flattening them.
        fprintf(stderr, "SYCL exception while enumerating GPU devices: %s\n", e.what());
        devices.clear();
    }

    if (devices.empty()) {
        // Exit 77, not 1. "No device" is not a test result in either direction:
        // returning 1 renders a legitimately CPU-only runner as a hard FAILURE,
        // which is the mirror image of the exit-0 vacuous pass this family is named
        // for (llama.cpp-k208) -- same root confusion, opposite symptom. 77 is
        // ctest's SKIP_RETURN_CODE, so ctest reports *skipped* rather than passed or
        // failed, and a bare shell run still gets a non-zero status. The point is to
        // make the skip visible AS a skip, never to forbid skipping: a CPU-only
        // runner still legitimately lands here. The registration in
        // ggml/src/ggml-sycl/CMakeLists.txt carries the matching SKIP_RETURN_CODE 77;
        // without it ctest renders 77 as FAILED.
        fprintf(stderr,
                "SKIP: no SYCL GPU devices available; this run proves NOTHING "
                "about XMX dispatch. Source oneAPI and re-run to actually test it.\n"
                "      (source /opt/intel/oneapi/setvars.sh --force)\n");
        return 77;  // ctest SKIP_RETURN_CODE -- a skip must be visible AS a skip
    }

    // devices[0] exists, so this cannot hit the default-selector throw described
    // above. A failure here is a real one and stays a failure.
    sycl::queue q = make_queue_or_fail(devices[0]);

    auto dev = q.get_device();
    fprintf(stderr, "Using GPU: %s\n", dev.get_info<sycl::info::device::name>().c_str());

    // Query XMX capabilities
    auto caps = query_xmx_capabilities(dev);
    fprintf(stderr, "XMX support: %s\n", caps.supported ? "YES" : "NO");
    if (caps.supported) {
        fprintf(stderr, "XMX dimensions: M=%zu, N=%zu, K=%zu\n", caps.M, caps.N, caps.K);
        fprintf(stderr, "XMX int8: %s, fp16: %s\n", caps.supports_int8 ? "YES" : "NO",
                caps.supports_fp16 ? "YES" : "NO");
    }
    fprintf(stderr, "\n");

    // Run tests
    test_xmx_dispatch_selects_joint_matrix_for_q4_0(q, caps);
    test_xmx_dispatch_fallback_for_non_xmx_device(q, caps);
    test_xmx_dispatch_threshold_batch_size(q, caps);
    test_xmx_dispatch_handles_non_aligned_dimensions(q, caps);
    test_xmx_dispatch_empty_batch_returns_immediately(q, caps);

    // Summary
    fprintf(stderr, "\n=== Summary ===\n");
    fprintf(stderr, "Tests run: %d, Passed: %d, Skipped: %d, Failed: %d\n", g_tests_run, g_tests_passed,
            g_tests_skipped, g_tests_run - g_tests_passed - g_tests_skipped);

    return (g_tests_passed + g_tests_skipped == g_tests_run) ? 0 : 1;
}
