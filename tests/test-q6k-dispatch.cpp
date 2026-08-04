// Test for Q6_K MMVQ dispatch using ACTUAL ggml backend API
// This test exercises the full dispatch path including SoA/AoS handling
//
// Build: cmake --build build --target test-q6k-dispatch
// Run: ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-q6k-dispatch
//
// Environment variables:
//   GGML_SYCL_DISABLE_GRAPH=1 - Disable SYCL graphs
//   (default)                - SoA optimization enabled
//
// Positive-control options (an exit failure is the expected RED result):
//   --reference=f32          - restore the former, mismatched F32 oracle
//   --corrupt-q8-reference   - perturb the matching Q8_1 oracle

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>

// Use actual ggml headers
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml-cpu.h"
#include "ggml-sycl/ggml-sycl-test.hpp"

// Include quants header for quantization/dequantization functions
#include "ggml-quants.h"

// Constants from ggml-common.h
#define QK_K 256
#define QK8_1 32
#ifndef QI6_K
#define QI6_K 32
#endif
#ifndef QR6_K
#define QR6_K 2
#endif
#ifndef QI8_1
#define QI8_1 (QK8_1 / 4)
#endif

// Portable packed-byte helpers for the Q8_1 positive control.
static constexpr uint32_t assemble_u32_le(uint8_t b0, uint8_t b1, uint8_t b2, uint8_t b3) {
    return static_cast<uint32_t>(b0) |
           (static_cast<uint32_t>(b1) << 8) |
           (static_cast<uint32_t>(b2) << 16) |
           (static_cast<uint32_t>(b3) << 24);
}

static inline uint32_t load_u32_le(const uint8_t * bytes, int i32) {
    bytes += 4 * i32;
    return assemble_u32_le(bytes[0], bytes[1], bytes[2], bytes[3]);
}

static inline uint32_t load_u32_le(const int8_t * bytes, int i32) {
    bytes += 4 * i32;
    return assemble_u32_le(
        static_cast<uint8_t>(bytes[0]), static_cast<uint8_t>(bytes[1]),
        static_cast<uint8_t>(bytes[2]), static_cast<uint8_t>(bytes[3]));
}

static constexpr int signed_byte_lane(uint32_t packed, int lane) {
    return ((packed >> (8 * lane)) & 0xffu) < 0x80u
        ? static_cast<int>((packed >> (8 * lane)) & 0xffu)
        : static_cast<int>((packed >> (8 * lane)) & 0xffu) - 0x100;
}

static constexpr int q6_lane(uint32_t vl, uint32_t vh, int nibble, int lane) {
    return static_cast<int>(
        ((vl >> (8 * lane + 4 * nibble)) & 0x0fu) |
        (((vh >> (8 * lane + 4 * nibble)) & 0x03u) << 4)) - 32;
}

static bool q6k_positive_control_helper_known_vector() {
    const uint8_t ql[4] = { 0x10, 0x32, 0x54, 0x76 };
    const uint8_t qh[4] = { 0x20, 0x10, 0x30, 0x00 };
    const int8_t qs[4] = { -128, -1, 0, 127 };
    const uint32_t packed_ql = load_u32_le(ql, 0);
    const uint32_t packed_qh = load_u32_le(qh, 0);
    const uint32_t packed_qs = load_u32_le(qs, 0);

    return packed_ql == 0x76543210u && packed_qh == 0x00301020u && packed_qs == 0x7f00ff80u &&
           q6_lane(packed_ql, packed_qh, 1, 0) == 1 &&
           q6_lane(packed_ql, packed_qh, 1, 1) == -13 &&
           q6_lane(packed_ql, packed_qh, 1, 2) == 21 &&
           q6_lane(packed_ql, packed_qh, 1, 3) == -25 &&
           signed_byte_lane(packed_qs, 0) == -128 && signed_byte_lane(packed_qs, 1) == -1 &&
           signed_byte_lane(packed_qs, 2) == 0 && signed_byte_lane(packed_qs, 3) == 127;
}

static float cpu_vec_dot_q6_K_q8_1(const block_q6_K* bq6_K, const block_q8_1* bq8_1, int iqs) {
    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    const uint32_t vl = load_u32_le(bq6_K->ql, iqs);
    const uint32_t vh = load_u32_le(
        bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;

    const int8_t* scs = bq6_K->scales + scale_offset;

    float sumf = 0.0f;
    for (int i = 0; i < QR6_K; ++i) {
        const int sc = scs[4 * i];
        const uint32_t u = load_u32_le(bq8_1[bq8_offset + 2*i].qs, iqs % QI8_1);
        const float d8 = ggml_fp16_to_fp32(bq8_1[bq8_offset + 2*i].d);

        int dp4a_result = 0;
        for (int j = 0; j < 4; ++j) {
            const int vi_j = q6_lane(vl, vh, i, j);
            dp4a_result += vi_j * signed_byte_lane(u, j);
        }

        sumf += d8 * (dp4a_result * sc);
    }
    return ggml_fp16_to_fp32(bq6_K->d) * sumf;
}

static float cpu_row_dot_q6_K_q8_1(const block_q6_K* x_row, const block_q8_1* y, int ncols) {
    const int blocks_per_row = ncols / QK_K;
    float sum = 0.0f;

    for (int ib = 0; ib < blocks_per_row; ++ib) {
        const block_q6_K* bx = &x_row[ib];
        const block_q8_1* by = &y[ib * (QK_K / QK8_1)];

        for (int iqs = 0; iqs < QI6_K; ++iqs) {
            sum += cpu_vec_dot_q6_K_q8_1(bx, by, iqs);
        }
    }
    return sum;
}

static bool parse_layout_arg(const char * arg, ggml_layout_mode & out) {
    if (!arg) {
        return false;
    }
    if (strcmp(arg, "aos") == 0) {
        out = GGML_LAYOUT_AOS;
        return true;
    }
    if (strcmp(arg, "soa") == 0) {
        out = GGML_LAYOUT_SOA;
        return true;
    }
    if (strcmp(arg, "coalesced") == 0) {
        out = GGML_LAYOUT_COALESCED;
        return true;
    }
    if (strcmp(arg, "xmx_tiled") == 0) {
        out = GGML_LAYOUT_XMX_TILED;
        return true;
    }
    if (strcmp(arg, "xmx_gemm_tiled") == 0) {
        out = GGML_LAYOUT_XMX_GEMM_TILED;
        return true;
    }
    return false;
}

static const char * layout_mode_name(ggml_layout_mode mode) {
    switch (mode) {
        case GGML_LAYOUT_AOS:
            return "aos";
        case GGML_LAYOUT_SOA:
            return "soa";
        case GGML_LAYOUT_COALESCED:
            return "coalesced";
        case GGML_LAYOUT_XMX_TILED:
            return "xmx_tiled";
        case GGML_LAYOUT_XMX_GEMM_TILED:
            return "xmx_gemm_tiled";
        default:
            return "unknown";
    }
}

// CPU reference: compute dot product of Q6_K row with F32 vector
// Uses dequantization for accuracy
static float cpu_dot_q6k_f32(const void* x_data, const float* y, int ncols) {
    // Dequantize Q6_K to float
    std::vector<float> x_f32(ncols);
    dequantize_row_q6_K((const block_q6_K*)x_data, x_f32.data(), ncols);

    // Compute dot product
    float sum = 0.0f;
    for (int i = 0; i < ncols; i++) {
        sum += x_f32[i] * y[i];
    }
    return sum;
}

static constexpr float diagnostic_abs(float value) {
    return value < 0.0f ? -value : value;
}

static constexpr float gpu_q8_tolerance(float cpu_q8) {
    return 1e-3f > 1e-4f * diagnostic_abs(cpu_q8) ? 1e-3f : 1e-4f * diagnostic_abs(cpu_q8);
}

static constexpr bool gpu_q8_contract_match(float gpu, float cpu_q8) {
    return diagnostic_abs(gpu - cpu_q8) <= gpu_q8_tolerance(cpu_q8);
}

static constexpr bool gpu_f32_legacy_match(float gpu, float cpu_f32) {
    const float difference = diagnostic_abs(gpu - cpu_f32);
    const float error = diagnostic_abs(cpu_f32) > 1e-6f ? difference / diagnostic_abs(cpu_f32) : difference;
    return error <= 0.01f;
}

static constexpr float corrupted_q8_reference(float cpu_q8) {
    const float mutation = 8.0f * gpu_q8_tolerance(cpu_q8);
    return cpu_q8 < 0.0f ? cpu_q8 - mutation : cpu_q8 + mutation;
}

static constexpr bool q8_accounts_for_f32_delta(float gpu, float cpu_q8, float cpu_f32) {
    const float q8_f32_abs = diagnostic_abs(cpu_q8 - cpu_f32);
    const float q8_f32_error = diagnostic_abs(cpu_f32) > 1e-6f ? q8_f32_abs / diagnostic_abs(cpu_f32) : q8_f32_abs;
    return gpu_q8_contract_match(gpu, cpu_q8) && q8_f32_error > 0.01f;
}

enum class q8_diagnostic_state {
    not_evaluated,
    consistent,
    not_consistent,
};

static constexpr q8_diagnostic_state q8_diagnostic_summary(int f32_failures, int consistent_rows) {
    return f32_failures == 0 ? q8_diagnostic_state::not_evaluated :
           consistent_rows == f32_failures ? q8_diagnostic_state::consistent :
                                             q8_diagnostic_state::not_consistent;
}

// Host-only contract checks: the Q8 oracle accepts the known matching result, the former F32
// oracle rejects the observed 1.444% delta, and a corrupted Q8 oracle is rejected specifically.
static_assert(gpu_q8_contract_match(101.444f, 101.444f), "matching Q8 oracle must pass");
static_assert(!gpu_f32_legacy_match(101.444f, 100.0f), "1.444% GPU-F32 positive control must RED");
static_assert(!gpu_q8_contract_match(101.444f, corrupted_q8_reference(101.444f)),
              "corrupting the Q8 oracle must RED the Q8 contract");
static_assert(gpu_q8_contract_match(0.001f, 0.0f), "absolute threshold boundary must match");
static_assert(!gpu_q8_contract_match(0.0011f, 0.0f), "values outside absolute threshold must not match");
static_assert(gpu_q8_contract_match(10001.0f, 10000.0f), "relative threshold boundary must match");
static_assert(!gpu_q8_contract_match(10001.1f, 10000.0f), "values outside relative threshold must not match");
static_assert(q8_accounts_for_f32_delta(0.0205f, 0.02f, 0.0f),
              "Q8-F32 crossing the F32 gate must account for a tightly matching GPU delta");
static_assert(!q8_accounts_for_f32_delta(0.0105f, 0.01f, 0.0f),
              "Q8-F32 at the F32 gate boundary must not account for a failing delta");
static_assert(q8_diagnostic_summary(0, 0) == q8_diagnostic_state::not_evaluated,
              "zero F32 failures must not produce a conclusion");

static void format_relative_metric(char * buffer, size_t size, float reference, float relative_error) {
    if (std::abs(reference) <= 1e-6f) {
        snprintf(buffer, size, "N/A");
    } else {
        snprintf(buffer, size, "%.3f%%", relative_error * 100.0f);
    }
}

static bool use_f32_positive_control = false;
static bool corrupt_q8_positive_control = false;

static bool reference_contract_match(float gpu, float cpu_q8_reference, float cpu_f32) {
    return use_f32_positive_control ? gpu_f32_legacy_match(gpu, cpu_f32)
                                    : gpu_q8_contract_match(gpu, cpu_q8_reference);
}

static const char * reference_contract_name() {
    return use_f32_positive_control ? "CPU_F32 positive control (legacy 1% gate)"
                                    : "CPU_Q8_1 (abs <= max(1e-3, 1e-4*abs(CPU_Q8_1)))";
}

static void print_q8_diagnostic_summary(int f32_failures, int consistent_rows) {
    const q8_diagnostic_state state = q8_diagnostic_summary(f32_failures, consistent_rows);
    if (state == q8_diagnostic_state::not_evaluated) {
        printf("  Activation-quantization observation: not evaluated (no F32-failing sampled rows)\n");
        return;
    }

    if (state == q8_diagnostic_state::consistent) {
        printf("  Activation-quantization observation: all %d F32-failing sampled rows are consistent with activation quantization\n",
               f32_failures);
    } else {
        printf("  Activation-quantization observation: %d/%d F32-failing sampled rows are consistent with activation quantization\n",
               consistent_rows, f32_failures);
    }
    printf("    Observational only (does not establish causation): abs(GPU-CPU_Q8) <= max(1e-3, 1e-4*abs(CPU_Q8))\n");
    printf("    and Q8-F32 independently crosses the same F32 failure gate\n");
}

// Test 1: Basic Q6_K MUL_MAT with single token (MMVQ path)
bool test_q6k_mul_mat_single_token() {
    printf("Test 1: Q6_K MUL_MAT single token (MMVQ dispatch path)\n");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: Could not initialize SYCL backend\n");
        return true;
    }
    printf("  Backend: %s\n", ggml_backend_name(backend));

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    // Use realistic dimensions from Mistral 7B
    const int n_embd = 4096;
    const int n_vocab = 32000;
    const int n_tokens = 1;  // Single token = MMVQ path

    // Create context
    struct ggml_init_params params = {
        .mem_size   = 32 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context* ctx = ggml_init(params);

    // Create weight tensor (Q6_K)
    struct ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, n_embd, n_vocab);
    ggml_set_name(weight, "lm_head");

    // Create input tensor (F32)
    struct ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_name(input, "hidden_state");

    // Create output tensor
    struct ggml_tensor* output = ggml_mul_mat(ctx, weight, input);
    ggml_set_name(output, "logits");

    // Allocate weight buffer (with SoA reordering if enabled)
    size_t weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t weight_buffer = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buffer, weight, (void*)ggml_backend_buffer_get_base(weight_buffer));

    // Allocate compute buffer
    size_t input_size = ggml_backend_buft_get_alloc_size(buft, input);
    size_t output_size = ggml_backend_buft_get_alloc_size(buft, output);
    ggml_backend_buffer_t compute_buffer = ggml_backend_buft_alloc_buffer(buft, input_size + output_size + 4096);
    ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    uint8_t* base = (uint8_t*)ggml_backend_buffer_get_base(compute_buffer);
    ggml_backend_tensor_alloc(compute_buffer, input, base);
    ggml_backend_tensor_alloc(compute_buffer, output, base + input_size);

    // Create random test data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Create float data for quantization
    const int weight_floats = n_vocab * n_embd;
    std::vector<float> weight_f32(weight_floats);
    for (int i = 0; i < weight_floats; i++) {
        weight_f32[i] = dist(rng);
    }

    // Quantize to Q6_K using production function
    const int blocks_per_row = n_embd / QK_K;
    const int total_blocks = n_vocab * blocks_per_row;
    std::vector<block_q6_K> weight_q6k(total_blocks);
    quantize_q6_K(weight_f32.data(), weight_q6k.data(), n_vocab, n_embd, nullptr);

    // Set weight data
    ggml_backend_tensor_set(weight, weight_q6k.data(), 0, total_blocks * sizeof(block_q6_K));

    // Create input data
    std::vector<float> input_f32(n_embd);
    for (int i = 0; i < n_embd; i++) {
        input_f32[i] = dist(rng);
    }
    ggml_backend_tensor_set(input, input_f32.data(), 0, n_embd * sizeof(float));

    // Quantize the activation once for the Q8_1 positive control used by every sampled row.
    std::vector<block_q8_1> input_q8(n_embd / QK8_1);
    quantize_row_q8_1_ref(input_f32.data(), input_q8.data(), n_embd);

    // Build and execute graph
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    printf("  Executing MUL_MAT graph...\n");
    enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        printf("  FAIL: Graph compute failed with status %d\n", status);
        ggml_backend_buffer_free(weight_buffer);
        ggml_backend_buffer_free(compute_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    // Get GPU output
    std::vector<float> gpu_output(n_vocab);
    ggml_backend_tensor_get(output, gpu_output.data(), 0, n_vocab * sizeof(float));

    // Compute CPU reference for first few rows
    printf("  Computing CPU reference...\n");
    const int test_rows = std::min(16, n_vocab);
    int contract_mismatches = 0;
    int f32_observation_failures = 0;
    int q8_consistent_f32_failures = 0;
    float max_rel_error = 0.0f;
    bool has_relative_error = false;

    printf("  Sample row comparisons (enforced contract plus CPU_F32/Q8 provenance):\n");
    for (int row = 0; row < test_rows; row++) {
        float cpu_val = cpu_dot_q6k_f32(&weight_q6k[row * blocks_per_row], input_f32.data(), n_embd);
        float cpu_q8_val = cpu_row_dot_q6_K_q8_1(
            &weight_q6k[row * blocks_per_row], input_q8.data(), n_embd);
        float gpu_val = gpu_output[row];
        const float cpu_q8_reference = corrupt_q8_positive_control
            ? corrupted_q8_reference(cpu_q8_val) : cpu_q8_val;

        float abs_diff = std::abs(gpu_val - cpu_val);
        float rel_error = (std::abs(cpu_val) > 1e-6f) ? abs_diff / std::abs(cpu_val) : abs_diff;
        float gpu_q8_abs = std::abs(gpu_val - cpu_q8_reference);
        float gpu_q8_rel = (std::abs(cpu_q8_reference) > 1e-6f)
            ? gpu_q8_abs / std::abs(cpu_q8_reference) : gpu_q8_abs;
        float q8_f32_abs = std::abs(cpu_q8_val - cpu_val);
        float q8_f32_rel = (std::abs(cpu_val) > 1e-6f) ? q8_f32_abs / std::abs(cpu_val) : q8_f32_abs;
        if (std::abs(cpu_val) > 1e-6f) {
            max_rel_error = std::max(max_rel_error, rel_error);
            has_relative_error = true;
        }

        char gpu_f32_rel_text[32];
        char gpu_q8_rel_text[32];
        char q8_f32_rel_text[32];
        format_relative_metric(gpu_f32_rel_text, sizeof(gpu_f32_rel_text), cpu_val, rel_error);
        format_relative_metric(gpu_q8_rel_text, sizeof(gpu_q8_rel_text), cpu_q8_reference, gpu_q8_rel);
        format_relative_metric(q8_f32_rel_text, sizeof(q8_f32_rel_text), cpu_val, q8_f32_rel);
        const bool contract_match = reference_contract_match(gpu_val, cpu_q8_reference, cpu_val);
        printf("    Row %2d: GPU=%10.6f CPU_F32=%10.6f CPU_Q8=%10.6f MATCH_REF=%10.6f "
               "GPU-F32 abs=%9.6f rel=%7s GPU-REF abs=%9.6f rel=%7s "
               "Q8-F32 abs=%9.6f rel=%7s contract=%s\n",
               row, gpu_val, cpu_val, cpu_q8_val, cpu_q8_reference,
               abs_diff, gpu_f32_rel_text, gpu_q8_abs, gpu_q8_rel_text,
               q8_f32_abs, q8_f32_rel_text, contract_match ? "OK" : "FAIL");

        if (!contract_match) {
            contract_mismatches++;
        }
        if (!gpu_f32_legacy_match(gpu_val, cpu_val)) {
            f32_observation_failures++;
            if (q8_accounts_for_f32_delta(gpu_val, cpu_q8_val, cpu_val)) {
                q8_consistent_f32_failures++;
            }
        }
    }

    if (has_relative_error) {
        printf("  Max observed GPU-F32 relative error: %.4f%%\n", max_rel_error * 100);
    } else {
        printf("  Max observed GPU-F32 relative error: N/A\n");
    }
    printf("  Enforced reference contract: %s\n", reference_contract_name());
    print_q8_diagnostic_summary(f32_observation_failures, q8_consistent_f32_failures);
    printf("  Weight extra ptr: %p\n", weight->extra);

    // Cleanup
    ggml_backend_buffer_free(weight_buffer);
    ggml_backend_buffer_free(compute_buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (contract_mismatches > 0) {
        printf("  FAIL: %d/%d rows violate the enforced %s contract\n",
               contract_mismatches, test_rows, reference_contract_name());
        return false;
    }

    printf("  PASS: All %d test rows satisfy the enforced %s contract\n",
           test_rows, reference_contract_name());
    return true;
}

// Test 2: Q6_K with dimensions matching actual Mistral model layers
bool test_q6k_mistral_dimensions() {
    printf("\nTest 2: Q6_K with Mistral 7B layer dimensions\n");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: Could not initialize SYCL backend\n");
        return true;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    // Mistral 7B FFN dimensions
    const int n_embd = 4096;
    const int n_ff = 14336;  // Mistral 7B FFN intermediate size
    const int n_tokens = 1;

    struct ggml_init_params params = {
        .mem_size   = 32 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context* ctx = ggml_init(params);

    // FFN gate weight (Q6_K)
    struct ggml_tensor* gate = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, n_embd, n_ff);
    ggml_set_name(gate, "ffn.gate");

    struct ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_name(input, "hidden");

    struct ggml_tensor* output = ggml_mul_mat(ctx, gate, input);
    ggml_set_name(output, "gate_out");

    // Allocate buffers
    size_t weight_buf_size = ggml_backend_buft_get_alloc_size(buft, gate);
    ggml_backend_buffer_t weight_buffer = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buffer, gate, (void*)ggml_backend_buffer_get_base(weight_buffer));

    size_t input_size = ggml_backend_buft_get_alloc_size(buft, input);
    size_t output_size = ggml_backend_buft_get_alloc_size(buft, output);
    ggml_backend_buffer_t compute_buffer = ggml_backend_buft_alloc_buffer(buft, input_size + output_size + 4096);
    ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    uint8_t* base = (uint8_t*)ggml_backend_buffer_get_base(compute_buffer);
    ggml_backend_tensor_alloc(compute_buffer, input, base);
    ggml_backend_tensor_alloc(compute_buffer, output, base + input_size);

    // Generate test data
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    const int weight_floats = n_ff * n_embd;
    std::vector<float> weight_f32(weight_floats);
    for (int i = 0; i < weight_floats; i++) {
        weight_f32[i] = dist(rng);
    }

    const int blocks_per_row = n_embd / QK_K;
    const int total_blocks = n_ff * blocks_per_row;
    std::vector<block_q6_K> weight_q6k(total_blocks);
    quantize_q6_K(weight_f32.data(), weight_q6k.data(), n_ff, n_embd, nullptr);

    ggml_backend_tensor_set(gate, weight_q6k.data(), 0, total_blocks * sizeof(block_q6_K));

    std::vector<float> input_f32(n_embd);
    for (int i = 0; i < n_embd; i++) {
        input_f32[i] = dist(rng);
    }
    ggml_backend_tensor_set(input, input_f32.data(), 0, n_embd * sizeof(float));

    // Quantize the activation once for the Q8_1 positive control used by every sampled row.
    std::vector<block_q8_1> input_q8(n_embd / QK8_1);
    quantize_row_q8_1_ref(input_f32.data(), input_q8.data(), n_embd);

    // Execute
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        printf("  FAIL: Graph compute failed\n");
        ggml_backend_buffer_free(weight_buffer);
        ggml_backend_buffer_free(compute_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    // Get GPU output and compare
    std::vector<float> gpu_output(n_ff);
    ggml_backend_tensor_get(output, gpu_output.data(), 0, n_ff * sizeof(float));

    // Test specific rows (first, middle, last)
    int test_indices[] = {0, 100, 1000, 5000, 10000, n_ff - 1};
    int contract_mismatches = 0;
    int f32_observation_failures = 0;
    int q8_consistent_f32_failures = 0;
    float max_rel_error = 0.0f;
    bool has_relative_error = false;

    printf("  Sample row comparisons (enforced contract plus CPU_F32/Q8 provenance):\n");
    for (int idx : test_indices) {
        if (idx >= n_ff) continue;

        float cpu_val = cpu_dot_q6k_f32(&weight_q6k[idx * blocks_per_row], input_f32.data(), n_embd);
        float cpu_q8_val = cpu_row_dot_q6_K_q8_1(
            &weight_q6k[idx * blocks_per_row], input_q8.data(), n_embd);
        float gpu_val = gpu_output[idx];
        const float cpu_q8_reference = corrupt_q8_positive_control
            ? corrupted_q8_reference(cpu_q8_val) : cpu_q8_val;

        float abs_diff = std::abs(gpu_val - cpu_val);
        float rel_error = (std::abs(cpu_val) > 1e-6f) ? abs_diff / std::abs(cpu_val) : abs_diff;
        float gpu_q8_abs = std::abs(gpu_val - cpu_q8_reference);
        float gpu_q8_rel = (std::abs(cpu_q8_reference) > 1e-6f)
            ? gpu_q8_abs / std::abs(cpu_q8_reference) : gpu_q8_abs;
        float q8_f32_abs = std::abs(cpu_q8_val - cpu_val);
        float q8_f32_rel = (std::abs(cpu_val) > 1e-6f) ? q8_f32_abs / std::abs(cpu_val) : q8_f32_abs;
        if (std::abs(cpu_val) > 1e-6f) {
            max_rel_error = std::max(max_rel_error, rel_error);
            has_relative_error = true;
        }

        char gpu_f32_rel_text[32];
        char gpu_q8_rel_text[32];
        char q8_f32_rel_text[32];
        format_relative_metric(gpu_f32_rel_text, sizeof(gpu_f32_rel_text), cpu_val, rel_error);
        format_relative_metric(gpu_q8_rel_text, sizeof(gpu_q8_rel_text), cpu_q8_reference, gpu_q8_rel);
        format_relative_metric(q8_f32_rel_text, sizeof(q8_f32_rel_text), cpu_val, q8_f32_rel);
        const bool contract_match = reference_contract_match(gpu_val, cpu_q8_reference, cpu_val);
        printf("    Row %5d: GPU=%10.4f CPU_F32=%10.4f CPU_Q8=%10.4f MATCH_REF=%10.4f "
               "GPU-F32 abs=%9.4f rel=%7s GPU-REF abs=%9.4f rel=%7s "
               "Q8-F32 abs=%9.4f rel=%7s contract=%s\n",
               idx, gpu_val, cpu_val, cpu_q8_val, cpu_q8_reference,
               abs_diff, gpu_f32_rel_text, gpu_q8_abs, gpu_q8_rel_text,
               q8_f32_abs, q8_f32_rel_text, contract_match ? "OK" : "FAIL");

        if (!contract_match) {
            contract_mismatches++;
        }
        if (!gpu_f32_legacy_match(gpu_val, cpu_val)) {
            f32_observation_failures++;
            if (q8_accounts_for_f32_delta(gpu_val, cpu_q8_val, cpu_val)) {
                q8_consistent_f32_failures++;
            }
        }
    }

    if (has_relative_error) {
        printf("  Max observed GPU-F32 relative error: %.4f%%\n", max_rel_error * 100);
    } else {
        printf("  Max observed GPU-F32 relative error: N/A\n");
    }
    printf("  Enforced reference contract: %s\n", reference_contract_name());
    print_q8_diagnostic_summary(f32_observation_failures, q8_consistent_f32_failures);

    // Cleanup
    ggml_backend_buffer_free(weight_buffer);
    ggml_backend_buffer_free(compute_buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (contract_mismatches > 0) {
        printf("  FAIL: %d sample rows violate the enforced %s contract\n",
               contract_mismatches, reference_contract_name());
        return false;
    }

    printf("  PASS: All sample rows satisfy the enforced %s contract\n", reference_contract_name());
    return true;
}

// Test 3: Compare output determinism (run twice, compare)
bool test_q6k_determinism() {
    printf("\nTest 3: Q6_K output determinism\n");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: Could not initialize SYCL backend\n");
        return true;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    const int n_embd = 4096;
    const int n_rows = 1024;
    const int n_tokens = 1;

    struct ggml_init_params params = {
        .mem_size   = 32 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context* ctx = ggml_init(params);

    struct ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, n_embd, n_rows);
    struct ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    struct ggml_tensor* output = ggml_mul_mat(ctx, weight, input);

    // Allocate buffers
    size_t weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t weight_buffer = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buffer, weight, (void*)ggml_backend_buffer_get_base(weight_buffer));

    size_t input_size = ggml_backend_buft_get_alloc_size(buft, input);
    size_t output_size = ggml_backend_buft_get_alloc_size(buft, output);
    ggml_backend_buffer_t compute_buffer = ggml_backend_buft_alloc_buffer(buft, input_size + output_size + 4096);
    ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    uint8_t* base = (uint8_t*)ggml_backend_buffer_get_base(compute_buffer);
    ggml_backend_tensor_alloc(compute_buffer, input, base);
    ggml_backend_tensor_alloc(compute_buffer, output, base + input_size);

    // Generate and set data
    std::mt19937 rng(999);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const int weight_floats = n_rows * n_embd;
    std::vector<float> weight_f32(weight_floats);
    for (int i = 0; i < weight_floats; i++) weight_f32[i] = dist(rng);

    const int blocks_per_row = n_embd / QK_K;
    const int total_blocks = n_rows * blocks_per_row;
    std::vector<block_q6_K> weight_q6k(total_blocks);
    quantize_q6_K(weight_f32.data(), weight_q6k.data(), n_rows, n_embd, nullptr);
    ggml_backend_tensor_set(weight, weight_q6k.data(), 0, total_blocks * sizeof(block_q6_K));

    std::vector<float> input_f32(n_embd);
    for (int i = 0; i < n_embd; i++) input_f32[i] = dist(rng);
    ggml_backend_tensor_set(input, input_f32.data(), 0, n_embd * sizeof(float));

    // Run twice
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    std::vector<float> output1(n_rows), output2(n_rows);

    ggml_backend_graph_compute(backend, graph);
    ggml_backend_tensor_get(output, output1.data(), 0, n_rows * sizeof(float));

    ggml_backend_graph_compute(backend, graph);
    ggml_backend_tensor_get(output, output2.data(), 0, n_rows * sizeof(float));

    // Compare
    int diffs = 0;
    for (int i = 0; i < n_rows; i++) {
        if (output1[i] != output2[i]) {
            if (diffs < 5) {
                printf("  Row %d: run1=%.8f run2=%.8f\n", i, output1[i], output2[i]);
            }
            diffs++;
        }
    }

    // Cleanup
    ggml_backend_buffer_free(weight_buffer);
    ggml_backend_buffer_free(compute_buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (diffs > 0) {
        printf("  FAIL: %d/%d rows differ between runs (non-deterministic!)\n", diffs, n_rows);
        return false;
    }

    printf("  PASS: Output is deterministic\n");
    return true;
}

// Test 4: Small dimensions edge case
bool test_q6k_small_dimensions() {
    printf("\nTest 4: Q6_K small dimension edge case\n");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: Could not initialize SYCL backend\n");
        return true;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    // Minimum valid Q6_K dimensions (256 elements = 1 block)
    const int n_embd = 256;  // QK_K
    const int n_rows = 16;
    const int n_tokens = 1;

    struct ggml_init_params params = {
        .mem_size   = 4 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context* ctx = ggml_init(params);

    struct ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, n_embd, n_rows);
    struct ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    struct ggml_tensor* output = ggml_mul_mat(ctx, weight, input);

    // Allocate
    size_t weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t weight_buffer = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buffer, weight, (void*)ggml_backend_buffer_get_base(weight_buffer));

    size_t input_size = ggml_backend_buft_get_alloc_size(buft, input);
    size_t output_size = ggml_backend_buft_get_alloc_size(buft, output);
    ggml_backend_buffer_t compute_buffer = ggml_backend_buft_alloc_buffer(buft, input_size + output_size + 1024);
    ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    uint8_t* base = (uint8_t*)ggml_backend_buffer_get_base(compute_buffer);
    ggml_backend_tensor_alloc(compute_buffer, input, base);
    ggml_backend_tensor_alloc(compute_buffer, output, base + input_size);

    // Use simple predictable values
    std::vector<float> weight_f32(n_rows * n_embd);
    for (int i = 0; i < n_rows * n_embd; i++) {
        weight_f32[i] = 0.01f * ((i % 100) - 50);  // -0.5 to 0.49
    }

    const int blocks_per_row = n_embd / QK_K;  // = 1
    std::vector<block_q6_K> weight_q6k(n_rows * blocks_per_row);
    quantize_q6_K(weight_f32.data(), weight_q6k.data(), n_rows, n_embd, nullptr);
    ggml_backend_tensor_set(weight, weight_q6k.data(), 0, n_rows * blocks_per_row * sizeof(block_q6_K));

    std::vector<float> input_f32(n_embd, 1.0f);  // All ones
    ggml_backend_tensor_set(input, input_f32.data(), 0, n_embd * sizeof(float));

    // Execute
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_backend_graph_compute(backend, graph);

    std::vector<float> gpu_output(n_rows);
    ggml_backend_tensor_get(output, gpu_output.data(), 0, n_rows * sizeof(float));

    // Compare all rows
    int mismatches = 0;
    float max_rel_error = 0.0f;

    for (int row = 0; row < n_rows; row++) {
        float cpu_val = cpu_dot_q6k_f32(&weight_q6k[row * blocks_per_row], input_f32.data(), n_embd);
        float gpu_val = gpu_output[row];

        float abs_diff = std::abs(gpu_val - cpu_val);
        float rel_error = (std::abs(cpu_val) > 1e-6f) ? abs_diff / std::abs(cpu_val) : abs_diff;
        max_rel_error = std::max(max_rel_error, rel_error);

        if (rel_error > 0.01f) {
            printf("  Row %d: GPU=%.6f CPU=%.6f err=%.2f%%\n", row, gpu_val, cpu_val, rel_error * 100);
            mismatches++;
        }
    }

    printf("  Max relative error: %.4f%%\n", max_rel_error * 100);

    // Cleanup
    ggml_backend_buffer_free(weight_buffer);
    ggml_backend_buffer_free(compute_buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (mismatches > 0) {
        printf("  FAIL: %d/%d rows have >1%% error\n", mismatches, n_rows);
        return false;
    }

    printf("  PASS: Small dimension edge case works correctly\n");
    return true;
}

int main(int argc, char** argv) {
    printf("========================================\n");
    printf("Q6_K Dispatch Unit Test\n");
    printf("========================================\n\n");

    if (!q6k_positive_control_helper_known_vector()) {
        printf("FAIL: Q6_K positive-control packed-byte helper known vector failed\n");
        return 1;
    }

    // Check environment
    const char * disable_graph = getenv("GGML_SYCL_DISABLE_GRAPH");
    ggml_layout_mode override_layout = GGML_LAYOUT_AOS;
    bool has_override = false;
    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];
        if (!arg) {
            continue;
        }
        const char * value = nullptr;
        if (strncmp(arg, "--layout=", 9) == 0) {
            value = arg + 9;
        } else if (strcmp(arg, "--layout") == 0 && i + 1 < argc) {
            value = argv[++i];
        } else if (strcmp(arg, "--reference=f32") == 0) {
            use_f32_positive_control = true;
            continue;
        } else if (strcmp(arg, "--reference=q8") == 0) {
            continue;
        } else if (strcmp(arg, "--corrupt-q8-reference") == 0) {
            corrupt_q8_positive_control = true;
            continue;
        }
        if (value && parse_layout_arg(value, override_layout)) {
            ggml_sycl::test_set_layout_override(override_layout);
            has_override = true;
        } else if (value) {
            printf("WARNING: unknown --layout=%s (ignoring)\n", value);
        } else {
            printf("WARNING: unknown option %s (ignoring)\n", arg);
        }
    }
    if (use_f32_positive_control && corrupt_q8_positive_control) {
        printf("FAIL: --reference=f32 and --corrupt-q8-reference are independent positive controls; run one at a time\n");
        if (has_override) {
            ggml_sycl::test_clear_layout_override();
        }
        return 2;
    }
    printf("Environment:\n");
    printf("  Layout override: %s\n", has_override ? layout_mode_name(override_layout) : "(auto)");
    printf("  GGML_SYCL_DISABLE_GRAPH: %s\n", disable_graph ? disable_graph : "(not set, graphs enabled)");
    printf("  Enforced reference: %s\n", reference_contract_name());
    printf("  Q8 reference mutation: %s\n", corrupt_q8_positive_control ? "ENABLED (expected RED)" : "disabled");
    if (use_f32_positive_control) {
        printf("  F32 oracle positive control: ENABLED (expected RED; known sampled delta 1.444%%)\n");
    }
    printf("\n");

    int passed = 0;
    int failed = 0;
    int skipped = 0;

    // Run tests
    bool result;

    result = test_q6k_mul_mat_single_token();
    if (result) passed++; else failed++;

    result = test_q6k_mistral_dimensions();
    if (result) passed++; else failed++;

    result = test_q6k_determinism();
    if (result) passed++; else failed++;

    result = test_q6k_small_dimensions();
    if (result) passed++; else failed++;

    // Summary
    printf("\n========================================\n");
    printf("Results: %d passed, %d failed, %d skipped\n", passed, failed, skipped);
    printf("========================================\n");

    if (has_override) {
        ggml_sycl::test_clear_layout_override();
    }

    return (failed > 0) ? 1 : 0;
}
