// Unit test for CPU→GPU data path with SoA reordering.
// Calls production dequantizers from dequantize.hpp and the production Q4_0
// reorder path from ggml-sycl.cpp. CPU GET_ROWS and DMMV remain test harnesses.
//
// Available after Task 17 registers this test target:
// Build: cmake --build build --target test-cpu-gpu-soa-interaction
// Run: ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-cpu-gpu-soa-interaction  # rc 0
// Positive control: append --corrupt-post-copy  # [SOA-REACH], [SOA-POSITIVE-CONTROL], rc 1

#include "dequantize.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static bool configure_bounded_runtime() {
    // reorder_rows_to_soa() intentionally enters the unified allocator for its
    // temporary device copy.  This unit-sized fixture must not reserve the
    // default full-VRAM arena to do that.  Do not override the pinned chunk
    // contract: the raw device reorder path must not allocate host staging.
    return setenv("GGML_SYCL_VRAM_ARENA", "0", 1) == 0;
}

template <typename T>
class usm_allocation {
public:
    usm_allocation(T * ptr, sycl::queue & q) : ptr_(ptr), q_(q) {}
    ~usm_allocation() noexcept {
        if (ptr_ != nullptr) {
            try {
                sycl::free(ptr_, q_);
            } catch (...) {
                // Cleanup must not replace an exception already in flight.
            }
        }
    }

    usm_allocation(const usm_allocation &) = delete;
    usm_allocation & operator=(const usm_allocation &) = delete;

private:
    T * ptr_;
    sycl::queue & q_;
};

// Exercise the production Q4_0 reorder entry point. This dispatches through
// reorder_rows_to_soa() to reorder_qw_q4_0() in ggml-sycl.cpp.
bool reorder_q4_0_to_soa_actual(sycl::queue& stream, uint8_t* data_device,
                                 int ncols, int nrows) {
    const size_t size = nrows * (ncols / QK4_0) * sizeof(block_q4_0);
    return reorder_rows_to_soa(data_device, GGML_TYPE_Q4_0, ncols, nrows, size, &stream);
}

// ============================================================================
// Test-local DMMV reference patterned after dequantize_mul_mat_vec_reorder.
// This is not production DMMV coverage; it does call the production dequantizer.
// ============================================================================
void dmmv_q4_0_soa_reference(sycl::queue& q, const uint8_t* soa_data,
                             const float* y_vec, float* result,
                             int ncols, int nrows) {
    const int d_offset = nrows * ncols / 2;  // SoA d offset

    const sycl::half* d_base = (const sycl::half*)(soa_data + d_offset);

    q.parallel_for(sycl::nd_range<1>(nrows * WARP_SIZE, WARP_SIZE),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            const int row = item.get_group(0);
            const int tid = item.get_local_id(0);
            const int blocks_per_row = ncols / QK4_0;

            float tmp = 0.0f;

            // Each work-item handles quant pairs, while the production
            // dequantizer supplies the low/high halves of each Q4_0 block.
            const int pairs_per_row = blocks_per_row * (QK4_0 / 2);
            for (int pair = tid; pair < pairs_per_row; pair += WARP_SIZE) {
                const int block_in_row = pair / (QK4_0 / 2);
                const int iqs = pair % (QK4_0 / 2);
                const int ib = row * blocks_per_row + block_in_row;
                const uint8_t* qs = soa_data + ib * (QK4_0 / 2);

                dfloat2 v;
                dequantize_q4_0_reorder(d_base, ib, qs, iqs, v);

                const int col = block_in_row * QK4_0 + iqs;
                tmp += v.x() * y_vec[col];
                tmp += v.y() * y_vec[col + QK4_0 / 2];
            }

            // Reference sub-group reduction.
            auto sg = item.get_sub_group();
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                tmp += sycl::shift_group_left(sg, tmp, offset);
            }

            if (tid == 0) {
                result[row] = tmp;
            }
        }).wait();
}

// CPU dequantization (simulates GET_ROWS on CPU - this is what the real code does)
void cpu_dequantize_q4_0(const block_q4_0* src, float* dst, int nblocks) {
    for (int ib = 0; ib < nblocks; ib++) {
        float d = (float)src[ib].d;
        for (int j = 0; j < QK4_0/2; j++) {
            const int x0 = (src[ib].qs[j] & 0x0F) - 8;
            const int x1 = (src[ib].qs[j] >> 4) - 8;
            dst[ib * QK4_0 + j] = x0 * d;
            dst[ib * QK4_0 + j + QK4_0/2] = x1 * d;
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

// Test 1: Verify production dequantize functions match
bool test_actual_dequantize_functions() {
    printf("Test 1: Verify production dequantize_q4_0 vs dequantize_q4_0_reorder\n");

    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    printf("  Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    const int nblocks = 128;
    const int ncols = nblocks * QK4_0;
    const int nrows = 1;

    // Create AoS test data
    std::vector<block_q4_0> aos_data(nblocks);
    for (int i = 0; i < nblocks; i++) {
        aos_data[i].d = sycl::half(0.1f + (float)(i % 100) / 1000.0f);
        for (int j = 0; j < QK4_0/2; j++) {
            aos_data[i].qs[j] = (uint8_t)((i * 17 + j * 3) & 0xFF);
        }
    }

    // Create SoA version
    std::vector<uint8_t> soa_data(nblocks * sizeof(block_q4_0));
    const size_t d_offset = ncols * nrows / 2;

    // Manual SoA conversion
    uint8_t* qs_ptr = soa_data.data();
    sycl::half* d_ptr = (sycl::half*)(soa_data.data() + d_offset);
    for (int i = 0; i < nblocks; i++) {
        for (int j = 0; j < QK4_0/2; j++) {
            qs_ptr[i * (QK4_0/2) + j] = aos_data[i].qs[j];
        }
        d_ptr[i] = aos_data[i].d;
    }

    // Compare dequantize outputs
    int errors = 0;
    for (int ib = 0; ib < nblocks; ib++) {
        for (int iqs = 0; iqs < QK4_0/2; iqs++) {
            dfloat2 v_aos, v_soa;

            // Production dequantize_q4_0 (AoS)
            dequantize_q4_0(aos_data.data(), ib, iqs, v_aos);

            // Production dequantize_q4_0_reorder (SoA)
            const uint8_t* qs = soa_data.data() + ib * (QK4_0/2);
            dequantize_q4_0_reorder(d_ptr, ib, qs, iqs, v_soa);

            if (!std::isfinite(v_aos.x()) || !std::isfinite(v_aos.y()) ||
                !std::isfinite(v_soa.x()) || !std::isfinite(v_soa.y()) ||
                fabsf(v_aos.x() - v_soa.x()) > 1e-5f ||
                fabsf(v_aos.y() - v_soa.y()) > 1e-5f) {
                if (errors < 5) {
                    printf("    Error: ib=%d iqs=%d: AoS=(%.6f,%.6f) SoA=(%.6f,%.6f)\n",
                           ib, iqs, v_aos.x(), v_aos.y(), v_soa.x(), v_soa.y());
                }
                errors++;
            }
        }
    }

    if (errors > 0) {
        printf("  FAIL: %d dequantize mismatches\n", errors);
        return false;
    }

    printf("  PASS: All %d dequantize comparisons match\n", nblocks * (QK4_0/2));
    return true;
}

// Test 2: Verify the production GPU reorder path
bool test_actual_reorder_kernel(bool corrupt_post_copy) {
    printf("Test 2: Verify production GPU reorder path\n");

    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};

    const int nrows = 128;
    const int ncols = 4096;
    const int nblocks = nrows * (ncols / QK4_0);
    const size_t size = nblocks * sizeof(block_q4_0);

    // Create AoS test data
    std::vector<block_q4_0> aos_data(nblocks);
    for (int i = 0; i < nblocks; i++) {
        aos_data[i].d = sycl::half(1.0f + (float)(i % 1000) / 10000.0f);
        for (int j = 0; j < QK4_0/2; j++) {
            aos_data[i].qs[j] = (uint8_t)(i + j);
        }
    }

    // Copy to GPU and reorder
    uint8_t* gpu_data = sycl::malloc_device<uint8_t>(size, q);
    usm_allocation<uint8_t> gpu_data_owner(gpu_data, q);
    if (gpu_data == nullptr) {
        printf("  FAIL: device allocation for reorder data failed\n");
        return false;
    }
    q.memcpy(gpu_data, aos_data.data(), size).wait();

    // Production reorder kernel
    if (!reorder_q4_0_to_soa_actual(q, gpu_data, ncols, nrows)) {
        printf("  FAIL: production Q4_0 reorder rejected the request\n");
        return false;
    }
    q.wait_and_throw();

    // Copy back
    std::vector<uint8_t> soa_result(size);
    q.memcpy(soa_result.data(), gpu_data, size).wait();
    fprintf(stderr, "[SOA-REACH] production reorder copy reached post-copy oracle\n");
    if (corrupt_post_copy) {
        soa_result[0] ^= 0xFF;
        fprintf(stderr, "[SOA-POSITIVE-CONTROL] corrupted post-copy result; oracle must fail\n");
    }

    // Verify SoA layout
    const size_t d_offset = nrows * ncols / 2;
    const uint8_t* qs_ptr = soa_result.data();
    const sycl::half* d_ptr = (const sycl::half*)(soa_result.data() + d_offset);

    int errors = 0;
    for (int i = 0; i < nblocks && errors < 5; i++) {
        // Check qs values
        for (int j = 0; j < QK4_0/2; j++) {
            if (qs_ptr[i * (QK4_0/2) + j] != aos_data[i].qs[j]) {
                printf("    Error: block %d qs[%d]: got=%02x expected=%02x\n",
                       i, j, qs_ptr[i * (QK4_0/2) + j], aos_data[i].qs[j]);
                errors++;
            }
        }
        // Check d values
        if (!std::isfinite((float)d_ptr[i]) ||
            fabsf((float)d_ptr[i] - (float)aos_data[i].d) > 1e-4f) {
            printf("    Error: block %d d: got=%.6f expected=%.6f\n",
                   i, (float)d_ptr[i], (float)aos_data[i].d);
            errors++;
        }
    }

    if (errors > 0) {
        printf("  FAIL: Reorder kernel produced %d errors\n", errors);
        return false;
    }

    printf("  PASS: GPU reorder kernel verified for %d blocks\n", nblocks);
    return true;
}

// Test 3: CPU GET_ROWS → GPU copy → GPU read with SoA reorder
bool test_cpu_gpu_path_with_soa() {
    printf("Test 3: CPU GET_ROWS → GPU copy → SoA reorder → GPU read\n");

    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};

    // Setup: token_embd on CPU (Q4_0)
    const int vocab_size = 1024;
    const int embed_dim = 4096;
    const int nblocks_per_row = embed_dim / QK4_0;
    const int total_blocks = vocab_size * nblocks_per_row;

    std::vector<block_q4_0> cpu_token_embd(total_blocks);
    for (int i = 0; i < total_blocks; i++) {
        cpu_token_embd[i].d = sycl::half(0.1f + (float)(i % 100) / 1000.0f);
        for (int j = 0; j < QK4_0/2; j++) {
            cpu_token_embd[i].qs[j] = (uint8_t)((i + j) & 0xFF);
        }
    }

    // Setup: layer weights on GPU (will be SoA reordered)
    const int weight_rows = 128;
    const int weight_cols = 4096;
    const int weight_nblocks = weight_rows * (weight_cols / QK4_0);
    const size_t weight_size = weight_nblocks * sizeof(block_q4_0);

    uint8_t* gpu_weights = sycl::malloc_device<uint8_t>(weight_size, q);
    usm_allocation<uint8_t> gpu_weights_owner(gpu_weights, q);
    if (gpu_weights == nullptr) {
        printf("  FAIL: device allocation for weights failed\n");
        return false;
    }
    std::vector<block_q4_0> cpu_weights(weight_nblocks);
    for (int i = 0; i < weight_nblocks; i++) {
        cpu_weights[i].d = sycl::half(0.5f);
        for (int j = 0; j < QK4_0/2; j++) {
            cpu_weights[i].qs[j] = 0x77;
        }
    }
    q.memcpy(gpu_weights, cpu_weights.data(), weight_size).wait();

    // GPU buffer for inp_embd (F32)
    float* gpu_inp_embd = sycl::malloc_device<float>(embed_dim, q);
    usm_allocation<float> gpu_inp_embd_owner(gpu_inp_embd, q);
    if (gpu_inp_embd == nullptr) {
        printf("  FAIL: device allocation for inp_embd failed\n");
        return false;
    }

    int pass_count = 0;
    int fail_count = 0;

    printf("  Testing 10 decode tokens...\n");

    for (int token = 1; token <= 10; token++) {
        int token_id = token * 10;

        // Step 1: CPU GET_ROWS - dequantize token embedding
        std::vector<float> cpu_result(embed_dim);
        cpu_dequantize_q4_0(&cpu_token_embd[token_id * nblocks_per_row],
                            cpu_result.data(), nblocks_per_row);

        // Step 2: Copy F32 to GPU (simulates inp_embd transfer)
        q.memcpy(gpu_inp_embd, cpu_result.data(), embed_dim * sizeof(float)).wait();

        // Step 3: SoA reorder on weights (first token only, simulates prompt phase)
        if (token == 1) {
            printf("  Token 1: production SoA reorder on weights...\n");
            if (!reorder_q4_0_to_soa_actual(q, gpu_weights, weight_cols, weight_rows)) {
                printf("  FAIL: production Q4_0 reorder rejected the request\n");
                return false;
            }
            q.wait_and_throw();
        }

        // Step 4: GPU reads inp_embd - copy back and verify
        std::vector<float> gpu_result(embed_dim);
        q.memcpy(gpu_result.data(), gpu_inp_embd, embed_dim * sizeof(float)).wait();

        // Verify
        int zeros = 0;
        int errors = 0;
        for (int i = 0; i < embed_dim; i++) {
            if (gpu_result[i] == 0.0f && cpu_result[i] != 0.0f) zeros++;
            if (!std::isfinite(gpu_result[i]) ||
                fabsf(gpu_result[i] - cpu_result[i]) > fabsf(cpu_result[i]) * 0.01f + 1e-6f) {
                errors++;
            }
        }

        if (zeros > embed_dim / 2) {
            printf("    Token %d: ZEROS detected (%d/%d)!\n", token, zeros, embed_dim);
            fail_count++;
        } else if (errors > 0) {
            printf("    Token %d: %d value mismatches\n", token, errors);
            fail_count++;
        } else {
            pass_count++;
        }
    }

    printf("  Results: %d/10 passed, %d/10 failed\n", pass_count, fail_count);

    if (fail_count > 0) {
        printf("  FAIL: Bug detected in CPU→GPU path with SoA\n");
        return false;
    }

    printf("  PASS: Local CPU→GPU transfer harness passed with production Q4_0 reorder\n");
    return true;
}

// Test 4: USM host memory (actual inp_embd allocation type)
bool test_usm_host_memory() {
    printf("Test 4: USM host memory (actual inp_embd allocation)\n");

    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};

    const int embed_dim = 4096;
    const int nblocks = embed_dim / QK4_0;

    // USM host allocation (this is how inp_embd is actually allocated)
    float* usm_inp_embd = sycl::malloc_host<float>(embed_dim, q);
    usm_allocation<float> usm_inp_embd_owner(usm_inp_embd, q);
    if (usm_inp_embd == nullptr) {
        printf("  FAIL: host allocation for inp_embd failed\n");
        return false;
    }

    // Q4_0 source
    std::vector<block_q4_0> token_data(nblocks);
    for (int i = 0; i < nblocks; i++) {
        token_data[i].d = sycl::half(1.0f);
        for (int j = 0; j < QK4_0/2; j++) {
            token_data[i].qs[j] = (uint8_t)((i + j + 1) & 0xFF);
        }
    }

    // Q4_0 weights on GPU
    const size_t weight_size = 128 * 4096 / QK4_0 * sizeof(block_q4_0);
    uint8_t* gpu_weights = sycl::malloc_device<uint8_t>(weight_size, q);
    usm_allocation<uint8_t> gpu_weights_owner(gpu_weights, q);
    if (gpu_weights == nullptr) {
        printf("  FAIL: device allocation for weights failed\n");
        return false;
    }
    std::vector<uint8_t> weight_init(weight_size, 0x55);
    q.memcpy(gpu_weights, weight_init.data(), weight_size).wait();

    // Expected
    std::vector<float> expected(embed_dim);
    cpu_dequantize_q4_0(token_data.data(), expected.data(), nblocks);

    int pass_count = 0;
    int fail_count = 0;

    for (int token = 1; token <= 10; token++) {
        // CPU writes to USM host memory (GET_ROWS result)
        cpu_dequantize_q4_0(token_data.data(), usm_inp_embd, nblocks);

        // SoA reorder on first token
        if (token == 1) {
            if (!reorder_q4_0_to_soa_actual(q, gpu_weights, 4096, 128)) {
                printf("  FAIL: production Q4_0 reorder rejected the request\n");
                return false;
            }
            q.wait_and_throw();
        }

        // GPU reads USM host memory directly
        float* gpu_sum_ptr = sycl::malloc_device<float>(1, q);
        usm_allocation<float> gpu_sum_owner(gpu_sum_ptr, q);
        if (gpu_sum_ptr == nullptr) {
            printf("  FAIL: device allocation for sum failed\n");
            return false;
        }
        q.memset(gpu_sum_ptr, 0, sizeof(float)).wait();

        q.parallel_for(1, [=](auto) {
            float sum = 0.0f;
            for (int i = 0; i < embed_dim; i++) {
                sum += usm_inp_embd[i];
            }
            *gpu_sum_ptr = sum;
        }).wait();

        float actual_sum;
        q.memcpy(&actual_sum, gpu_sum_ptr, sizeof(float)).wait();

        float expected_sum = 0.0f;
        for (int i = 0; i < embed_dim; i++) {
            expected_sum += expected[i];
        }

        if (!std::isfinite(actual_sum)) {
            printf("    Token %d: non-finite GPU sum\n", token);
            fail_count++;
        } else if (fabsf(actual_sum) < 1e-6f && fabsf(expected_sum) > 1e-6f) {
            printf("    Token %d: GPU saw ZEROS (expected_sum=%.2f)\n", token, expected_sum);
            fail_count++;
        } else if (fabsf(actual_sum - expected_sum) > fabsf(expected_sum) * 0.01f) {
            printf("    Token %d: Mismatch (%.2f vs %.2f)\n", token, actual_sum, expected_sum);
            fail_count++;
        } else {
            pass_count++;
        }
    }

    printf("  Results: %d/10 passed, %d/10 failed\n", pass_count, fail_count);

    if (fail_count > 0) {
        printf("  FAIL: USM host memory path has issues\n");
        return false;
    }

    printf("  PASS: USM host memory works correctly\n");
    return true;
}

// Test 5: test-local DMMV reference with production dequantization
bool test_dmmv_soa_reference() {
    printf("Test 5: DMMV reference with production SoA dequantization\n");

    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};

    const int nrows = 64;
    const int ncols = 4096;
    const int nblocks = nrows * (ncols / QK4_0);
    const size_t size = nblocks * sizeof(block_q4_0);

    // Varied Q4_0 scales and nibbles exercise non-neutral dequantized values.
    std::vector<block_q4_0> aos_data(nblocks);
    for (int i = 0; i < nblocks; i++) {
        aos_data[i].d = sycl::half(0.03125f * (1 + i % 7));
        for (int j = 0; j < QK4_0 / 2; j++) {
            const uint8_t low = (uint8_t)((i * 3 + j * 5 + 1) % 16);
            const uint8_t high = (uint8_t)((i * 7 + j * 2 + 4) % 16);
            aos_data[i].qs[j] = (uint8_t)(low | (high << 4));
        }
    }

    std::vector<float> y_vec(ncols);
    for (int col = 0; col < ncols; col++) {
        y_vec[col] = (float)((col * 13) % 29 - 14) / 16.0f;
    }

    // Independent CPU dequantize-and-dot oracle, row by row.
    std::vector<float> expected(nrows);
    std::vector<float> dequantized_row(ncols);
    const int blocks_per_row = ncols / QK4_0;
    for (int row = 0; row < nrows; row++) {
        cpu_dequantize_q4_0(aos_data.data() + row * blocks_per_row,
                            dequantized_row.data(), blocks_per_row);
        double dot = 0.0;
        for (int col = 0; col < ncols; col++) {
            dot += (double)dequantized_row[col] * (double)y_vec[col];
        }
        expected[row] = (float)dot;
    }

    // GPU allocations
    uint8_t* gpu_weights = sycl::malloc_device<uint8_t>(size, q);
    usm_allocation<uint8_t> gpu_weights_owner(gpu_weights, q);
    if (gpu_weights == nullptr) {
        printf("  FAIL: device allocation for DMMV weights failed\n");
        return false;
    }
    float* gpu_y = sycl::malloc_device<float>(ncols, q);
    usm_allocation<float> gpu_y_owner(gpu_y, q);
    if (gpu_y == nullptr) {
        printf("  FAIL: device allocation for DMMV Y failed\n");
        return false;
    }
    float* gpu_result = sycl::malloc_device<float>(nrows, q);
    usm_allocation<float> gpu_result_owner(gpu_result, q);
    if (gpu_result == nullptr) {
        printf("  FAIL: device allocation for DMMV result failed\n");
        return false;
    }

    q.memcpy(gpu_weights, aos_data.data(), size).wait();
    q.memcpy(gpu_y, y_vec.data(), ncols * sizeof(float)).wait();

    // Reorder to SoA through the production entry point.
    if (!reorder_q4_0_to_soa_actual(q, gpu_weights, ncols, nrows)) {
        printf("  FAIL: production Q4_0 reorder rejected the request\n");
        return false;
    }
    q.wait_and_throw();

    // Run the test-local DMMV reference.
    dmmv_q4_0_soa_reference(q, gpu_weights, gpu_y, gpu_result, ncols, nrows);

    // Get result
    std::vector<float> result(nrows);
    q.memcpy(result.data(), gpu_result, nrows * sizeof(float)).wait();

    int errors = 0;
    for (int row = 0; row < nrows; row++) {
        const float tolerance = fabsf(expected[row]) * 0.01f + 1e-3f;
        if (!std::isfinite(result[row]) ||
            fabsf(result[row] - expected[row]) > tolerance) {
            if (errors < 5) {
                printf("    Row %d: got %.6f, expected %.6f\n",
                       row, result[row], expected[row]);
            }
            errors++;
        }
    }

    if (errors > 0) {
        printf("  FAIL: DMMV reference produced %d incorrect values\n", errors);
        return false;
    }

    printf("  PASS: DMMV reference works correctly with production SoA dequantization\n");
    return true;
}

int main(int argc, char ** argv) {
    bool corrupt_post_copy = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--corrupt-post-copy") == 0) {
            corrupt_post_copy = true;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            return 2;
        }
    }
    if (!configure_bounded_runtime()) {
        fprintf(stderr, "FAIL: could not configure bounded SYCL test allocations\n");
        return 1;
    }

    printf("=== CPU→GPU SoA Interaction Tests ===\n");
    printf("Using production dequantization and Q4_0 reorder paths\n");
    printf("Bounded runtime: VRAM arena disabled; raw device reorder requires no host staging\n\n");

    try {
        int passed = 0;
        int failed = 0;

        if (test_actual_dequantize_functions()) passed++; else failed++;
        printf("\n");

        if (test_actual_reorder_kernel(corrupt_post_copy)) passed++; else failed++;
        printf("\n");

        if (test_cpu_gpu_path_with_soa()) passed++; else failed++;
        printf("\n");

        if (test_usm_host_memory()) passed++; else failed++;
        printf("\n");

        if (test_dmmv_soa_reference()) passed++; else failed++;
        printf("\n");

        printf("=================================\n");
        printf("Results: %d passed, %d failed\n", passed, failed);

        if (failed > 0) {
            printf("\nBug detected in one of the tests.\n");
        } else {
            printf("\nAll local transfer harness, production Q4_0 reorder, and inline dequantization checks passed.\n");
        }

        return failed > 0 ? 1 : 0;

    } catch (const sycl::exception& e) {
        fprintf(stderr, "SYCL exception: %s\n", e.what());
        return 1;
    }
}
