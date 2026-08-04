// Test for MMVQ Q6_K SoA with row slicing
// This test verifies the actual MMVQ kernel produces correct results with SoA layout.
//
// MMVQ is used for batch sizes 2-32 (between DMMV batch=1 and MMQ batch>32).
//
// Build: cmake --build build --target test-mmvq-q6k-soa-rowslice
// Run: ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-mmvq-q6k-soa-rowslice

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <random>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml-quants.h"

#define QK_K 256
#ifndef QK8_1
#define QK8_1 32
#endif
#ifndef QI6_K
#define QI6_K 32
#endif
#ifndef QR6_K
#define QR6_K 2
#endif
#ifndef QI8_1
#define QI8_1 (QK8_1 / 4)
#endif

// Helpers for Q6_K x Q8_1 CPU reference (matches MMVQ's little-endian packed-lane math).
// Assemble words byte-by-byte so the reference is independent of host alignment and byte order.
static inline uint32_t load_u32_le(const uint8_t * x8, const int i32) {
    const uint8_t * bytes = x8 + sizeof(uint32_t) * i32;
    return uint32_t(bytes[0])
         | (uint32_t(bytes[1]) <<  8)
         | (uint32_t(bytes[2]) << 16)
         | (uint32_t(bytes[3]) << 24);
}

static inline uint32_t load_u32_le(const int8_t * x8, const int i32) {
    const int8_t * bytes = x8 + sizeof(uint32_t) * i32;
    return uint32_t(uint8_t(bytes[0]))
         | (uint32_t(uint8_t(bytes[1])) <<  8)
         | (uint32_t(uint8_t(bytes[2])) << 16)
         | (uint32_t(uint8_t(bytes[3])) << 24);
}

static inline int extract_i8_lane(const uint32_t word, const int lane) {
    const uint32_t byte = (word >> (8 * lane)) & 0xffu;
    return byte < 0x80u ? int(byte) : int(byte) - 0x100;
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
            const int low = int((vl >> (8 * j + 4 * i)) & 0x0fu);
            const int high = int((vh >> (8 * j + 4 * i)) & 0x03u);
            const int vi_j = low | (high << 4);
            dp4a_result += (vi_j - 32) * extract_i8_lane(u, j);
        }

        sumf += d8 * (dp4a_result * sc);
    }
    return ggml_fp16_to_fp32(bq6_K->d) * sumf;
}

static bool test_portable_reference_known_vectors() {
    const uint8_t packed_u8[8] = { 0xa5, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd };
    const int8_t packed_i8[4] = { 1, -2, 3, -4 };
    const uint32_t u8_word = load_u32_le(packed_u8 + 1, 0);
    const uint32_t i8_word = load_u32_le(packed_i8, 0);
    const bool packing_ok = u8_word == 0x67452301u && i8_word == 0xfc03fe01u
                         && extract_i8_lane(i8_word, 0) == 1
                         && extract_i8_lane(i8_word, 1) == -2
                         && extract_i8_lane(i8_word, 2) == 3
                         && extract_i8_lane(i8_word, 3) == -4;

    block_q6_K q6 = {};
    block_q8_1 q8[QK_K / QK8_1] = {};

    // iqs=0 contains two groups of four Q6 values.  Their unsigned encodings are
    // {0,31,32,63} and {63,32,31,0}, covering both extremes and the sign boundary.
    const uint8_t ql[4] = { 0xf0, 0x0f, 0xf0, 0x0f };
    const uint8_t qh[4] = { 0x30, 0x21, 0x12, 0x03 };
    for (int lane = 0; lane < 4; ++lane) {
        q6.ql[lane] = ql[lane];
        q6.qh[lane] = qh[lane];
    }
    q6.scales[0] = 3;
    q6.scales[4] = -2;
    q6.d = ggml_fp32_to_fp16(2.0f);

    const int8_t q8_group0[4] = { 1, -2, 3, -4 };
    const int8_t q8_group1[4] = { -5, 6, -7, 8 };
    for (int lane = 0; lane < 4; ++lane) {
        q8[0].qs[lane] = q8_group0[lane];
        q8[2].qs[lane] = q8_group1[lane];
    }
    q8[0].d = ggml_fp32_to_fp16(0.5f);
    q8[2].d = ggml_fp32_to_fp16(0.25f);

    const float dot = cpu_vec_dot_q6_K_q8_1(&q6, q8, 0);
    const bool dot_ok = dot == -58.0f;
    printf("Portable reference known vectors: %s\n", packing_ok && dot_ok ? "PASS" : "FAIL");
    if (!packing_ok) {
        printf("  Packed words: u8=0x%08x i8=0x%08x\n", unsigned(u8_word), unsigned(i8_word));
    }
    if (!dot_ok) {
        printf("  Q6_K x Q8_1 dot: got %.6f expected -58.000000\n", dot);
    }
    return packing_ok && dot_ok;
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

// CPU reference: compute full matmul Q6_K @ Q8_1 (matches MMVQ quantization)
static void cpu_mul_mat_q6k_q8_1(const void* x_data, const float* y,
                                 float* dst, int nrows, int ncols, int n_tokens) {
    const block_q6_K* x = (const block_q6_K*)x_data;
    const int blocks_per_row = ncols / QK_K;
    const int y_blocks_per_row = ncols / QK8_1;

    std::vector<block_q8_1> y_q8(n_tokens * y_blocks_per_row);
    for (int tok = 0; tok < n_tokens; tok++) {
        quantize_row_q8_1_ref(y + tok * ncols,
                              y_q8.data() + tok * y_blocks_per_row,
                              ncols);
    }

    for (int row = 0; row < nrows; row++) {
        for (int tok = 0; tok < n_tokens; tok++) {
            // Output is transposed: dst[tok, row]
            dst[tok * nrows + row] = cpu_row_dot_q6_K_q8_1(
                &x[row * blocks_per_row],
                y_q8.data() + tok * y_blocks_per_row,
                ncols);
        }
    }
}

// Test MMVQ with specific batch size
static bool test_mmvq_q6k_batch(int n_tokens, int n_rows, int n_cols, bool verbose) {
    printf("\n--- MMVQ Q6_K batch=%d rows=%d cols=%d ---\n", n_tokens, n_rows, n_cols);

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: Could not initialize SYCL backend\n");
        return true;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    struct ggml_init_params params = {
        .mem_size   = 32 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        printf("  FAIL: Could not create ggml context\n");
        ggml_backend_free(backend);
        return false;
    }

    // Weight: [n_cols, n_rows] Q6_K
    struct ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, n_cols, n_rows);
    ggml_set_name(weight, "test_q6k_weight");

    // Input: [n_cols, n_tokens] F32
    struct ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_cols, n_tokens);
    ggml_set_name(input, "test_input");

    // Output: MUL_MAT(weight, input) -> [n_rows, n_tokens]
    struct ggml_tensor* output = ggml_mul_mat(ctx, weight, input);
    ggml_set_name(output, "test_output");

    // Allocate weight buffer (triggers SoA reordering)
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

    // Generate weight data - use proper quantization like test-mmq-q6k-gpu
    const int blocks_per_row = n_cols / QK_K;
    const int total_blocks = n_rows * blocks_per_row;
    std::mt19937 rng(42 + n_tokens);  // Different seed per test
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    // Create float data first
    const int weight_floats = n_rows * n_cols;
    std::vector<float> weight_f32(weight_floats);
    for (int i = 0; i < weight_floats; i++) {
        weight_f32[i] = dist(rng);
    }

    // Quantize to Q6_K properly
    std::vector<block_q6_K> weight_data(total_blocks);
    quantize_q6_K(weight_f32.data(), weight_data.data(), n_rows, n_cols, nullptr);

    // Upload weights (triggers SoA transformation)
    ggml_backend_tensor_set(weight, weight_data.data(), 0, total_blocks * sizeof(block_q6_K));

    // Generate input
    std::vector<float> input_data(n_cols * n_tokens);
    std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
    for (auto& v : input_data) v = input_dist(rng);
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    // Build and execute graph
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    ggml_backend_graph_compute(backend, graph);

    // Get GPU output
    std::vector<float> gpu_output(n_rows * n_tokens);
    ggml_backend_tensor_get(output, gpu_output.data(), 0, gpu_output.size() * sizeof(float));

    // CPU reference
    std::vector<float> cpu_output(n_rows * n_tokens);
    cpu_mul_mat_q6k_q8_1(weight_data.data(), input_data.data(),
                         cpu_output.data(), n_rows, n_cols, n_tokens);

    // Compare
    int mismatches = 0;
    float max_diff = 0.0f;
    int first_bad_tok = -1, first_bad_row = -1;

    const float abs_tol = (n_cols >= 4096 && n_rows >= 4096) ? 0.2f : 0.1f;
    const float rel_tol = (n_cols >= 4096 && n_rows >= 4096) ? 0.3f : 0.15f;

    for (int tok = 0; tok < n_tokens; tok++) {
        for (int row = 0; row < n_rows; row++) {
            int idx = tok * n_rows + row;
            float diff = std::fabs(gpu_output[idx] - cpu_output[idx]);
            float ref_mag = std::fabs(cpu_output[idx]);
            float rel_err = ref_mag > 1e-6 ? diff / ref_mag : diff;

            if (diff > abs_tol && rel_err > rel_tol) {
                if (first_bad_tok < 0) {
                    first_bad_tok = tok;
                    first_bad_row = row;
                }
                mismatches++;
                max_diff = std::max(max_diff, diff);
            }
        }
    }

    bool passed = (mismatches == 0);
    printf("  Result: %s (%d/%d mismatches, max_diff=%.4f)\n",
           passed ? "PASS" : "FAIL", mismatches, n_rows * n_tokens, max_diff);

    if (!passed && verbose && first_bad_tok >= 0) {
        printf("  First mismatch: tok=%d row=%d\n", first_bad_tok, first_bad_row);
        int idx = first_bad_tok * n_rows + first_bad_row;
        printf("    GPU=%.6f CPU=%.6f diff=%.6f\n",
               gpu_output[idx], cpu_output[idx],
               std::fabs(gpu_output[idx] - cpu_output[idx]));
    }

    ggml_backend_buffer_free(weight_buffer);
    ggml_backend_buffer_free(compute_buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    return passed;
}

int main() {
    printf("=== Q6_K MMVQ SoA Row Slice Test ===\n");
    printf("Tests actual MMVQ kernel (batch 2-32) with various tensor sizes.\n");

    int passed = 0, failed = 0;

    if (!test_portable_reference_known_vectors()) {
        return 1;
    }

    // Test 1: Small tensor, batch=4 (MMVQ range)
    if (test_mmvq_q6k_batch(4, 256, 512, true)) passed++; else failed++;

    // Test 2: Medium tensor, batch=8
    if (test_mmvq_q6k_batch(8, 512, 1024, true)) passed++; else failed++;

    // Test 3: Large tensor, batch=16
    if (test_mmvq_q6k_batch(16, 2048, 1024, true)) passed++; else failed++;

    // Test 4: Batch=2 (edge of MMVQ range)
    if (test_mmvq_q6k_batch(2, 256, 512, true)) passed++; else failed++;

    // Test 5: Batch=32 (upper edge of MMVQ range)
    if (test_mmvq_q6k_batch(32, 512, 1024, true)) passed++; else failed++;

    // Test 6: Production-like dimensions (Mistral 7B)
    if (test_mmvq_q6k_batch(8, 4096, 4096, true)) passed++; else failed++;

    printf("\n=================================================\n");
    printf("Results: %d passed, %d failed\n", passed, failed);

    return failed > 0 ? 1 : 0;
}
