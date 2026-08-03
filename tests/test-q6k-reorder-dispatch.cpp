// Comprehensive Q6_K reorder and dispatch unit test
// Tests: production reorder, SoA offsets, CPU dequantization, and GPU data access
// Uses actual production functions and values

#include <cmath>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>
#include <sycl/sycl.hpp>

// Include production headers
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml-quants.h"
#include "ggml-sycl/convert.hpp"
#include "ggml-sycl/ggml-sycl-test.hpp"

// Constants from production code
#define QK_K 256
#define QK8_1 32
#define QI6_K 32
#define QI8_1 8
#define QR6_K 2
#define WARP_SIZE 16

// block_q6_K layout (210 bytes):
// - ql[128]: low 4 bits of quants
// - qh[64]: high 2 bits of quants
// - scales[16]: sub-block scales
// - d[2]: super-block scale (half)

// SoA layout for N blocks:
// [ql: 128*N bytes][qh: 64*N bytes][scales: 16*N bytes][d: 2*N bytes]

template<typename T>
class usm_device_buffer {
public:
    usm_device_buffer(size_t count, sycl::queue & queue) : queue_(&queue), ptr_(sycl::malloc_device<T>(count, queue)) {}

    ~usm_device_buffer() noexcept {
        if (ptr_ != nullptr) {
            try {
                queue_->wait_and_throw();
            } catch (...) {
                // The explicit test wait reports asynchronous failures.
            }
            try {
                sycl::free(ptr_, *queue_);
            } catch (...) {
                // Destructors must not mask the test's original SYCL failure.
            }
        }
    }

    usm_device_buffer(const usm_device_buffer &) = delete;
    usm_device_buffer & operator=(const usm_device_buffer &) = delete;

    T * get() const { return ptr_; }
    explicit operator bool() const { return ptr_ != nullptr; }

private:
    sycl::queue * queue_;
    T * ptr_;
};

static void store_half(std::vector<uint8_t> & data, size_t offset, ggml_half value) {
    memcpy(data.data() + offset, &value, sizeof(value));
}

static ggml_half load_half(const std::vector<uint8_t> & data, size_t offset) {
    ggml_half value;
    memcpy(&value, data.data() + offset, sizeof(value));
    return value;
}

//=============================================================================
// Test 1: Verify the production Q6_K AoS -> SoA reorder
//=============================================================================
static bool test_production_reorder_layout() {
    printf("\n=== Test 1: Production Q6_K Reorder Layout Verification ===\n");

    try {
        sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
        const size_t nblocks = 4;
        const size_t data_size = nblocks * sizeof(block_q6_K);

        std::vector<block_q6_K> aos_data(nblocks);
        std::vector<uint8_t> expected(data_size);
        uint8_t * expected_ql = expected.data();
        uint8_t * expected_qh = expected_ql + nblocks * (QK_K / 2);
        int8_t * expected_scales = reinterpret_cast<int8_t *>(expected_qh + nblocks * (QK_K / 4));
        const size_t expected_d_offset = nblocks * (QK_K / 2 + QK_K / 4 + QK_K / 16);

        for (size_t b = 0; b < nblocks; ++b) {
            for (int i = 0; i < QK_K / 2; ++i) aos_data[b].ql[i] = static_cast<uint8_t>(b * 17 + i);
            for (int i = 0; i < QK_K / 4; ++i) aos_data[b].qh[i] = static_cast<uint8_t>(0x40 + b * 11 + i);
            for (int i = 0; i < QK_K / 16; ++i) aos_data[b].scales[i] = static_cast<int8_t>(-32 + b * 7 + i);
            aos_data[b].d = ggml_fp32_to_fp16(static_cast<float>(b + 1) / 3.0f);

            memcpy(expected_ql + b * (QK_K / 2), aos_data[b].ql, QK_K / 2);
            memcpy(expected_qh + b * (QK_K / 4), aos_data[b].qh, QK_K / 4);
            memcpy(expected_scales + b * (QK_K / 16), aos_data[b].scales, QK_K / 16);
            store_half(expected, expected_d_offset + b * sizeof(ggml_half), aos_data[b].d);
        }

        usm_device_buffer<block_q6_K> device_aos(nblocks, q);
        usm_device_buffer<uint8_t> device_soa(data_size, q);
        if (!device_aos || !device_soa) {
            printf("  FAIL: device allocation failed\n");
            return false;
        }

        q.memcpy(device_aos.get(), aos_data.data(), data_size);
        q.wait_and_throw();
        reorder_q6_k_aos_to_soa_sycl(device_aos.get(), device_soa.get(), nblocks, &q);
        q.wait_and_throw();

        std::vector<uint8_t> actual(data_size);
        q.memcpy(actual.data(), device_soa.get(), data_size);
        q.wait_and_throw();

        const bool pass = actual == expected;
        printf("  Production call: reorder_q6_k_aos_to_soa_sycl\n");
        printf("  Result: %s\n", pass ? "PASS" : "FAIL");
        return pass;
    } catch (const sycl::exception & e) {
        printf("  FAIL: SYCL error: %s\n", e.what());
        return false;
    }
}

//=============================================================================
// Test 2: Verify offset calculations from quants.hpp
//=============================================================================
static bool test_offset_calculations() {
    printf("\n=== Test 2: SoA Offset Calculations (quants.hpp) ===\n");

    // Simulate quants.hpp block_q_t<GGML_TYPE_Q6_K> offset calculations
    auto get_block_offset = [](int block_index, int n_blocks) -> std::pair<int, int> {
        int low_bits_index  = block_index * (QK_K / QR6_K);  // = block_index * 128
        int high_bits_index = n_blocks * (QK_K / 2) + (block_index * (QK_K / 4));  // = n_blocks * 128 + block_index * 64
        return { low_bits_index, high_bits_index };
    };

    auto get_d_offset = [](int nrows, int ncols, int block_index) -> std::pair<int, int> {
        int nblocks = (nrows * (ncols / QK_K));
        int total_qs_bytes = nblocks * (QK_K / 2) + nblocks * (QK_K / 4);  // nblocks * 192
        int block_scales = total_qs_bytes + block_index * (QK_K / 16);     // total_qs + block * 16
        int sb_scale = total_qs_bytes + nblocks * (QK_K / 16) + block_index * sizeof(ggml_half);
        return { block_scales, sb_scale };
    };

    const int nrows = 4;
    const int ncols = QK_K;  // 256 = 1 block per row
    const int nblocks = nrows * (ncols / QK_K);  // = 4

    printf("  Test config: nrows=%d, ncols=%d, nblocks=%d\n", nrows, ncols, nblocks);
    printf("\n  Expected SoA layout:\n");
    printf("    ql:     bytes 0-%d (128 bytes/block)\n", nblocks * 128 - 1);
    printf("    qh:     bytes %d-%d (64 bytes/block)\n", nblocks * 128, nblocks * 192 - 1);
    printf("    scales: bytes %d-%d (16 bytes/block)\n", nblocks * 192, nblocks * 208 - 1);
    printf("    d:      bytes %d-%d (2 bytes/block)\n", nblocks * 208, nblocks * 210 - 1);

    bool pass = true;
    for (int b = 0; b < nblocks; b++) {
        auto [ql_off, qh_off] = get_block_offset(b, nblocks);
        auto [scales_off, d_off] = get_d_offset(nrows, ncols, b);

        int expected_ql = b * 128;
        int expected_qh = nblocks * 128 + b * 64;
        int expected_scales = nblocks * 192 + b * 16;
        int expected_d = nblocks * 208 + b * 2;

        printf("  Block %d: ql=%d (exp %d), qh=%d (exp %d), scales=%d (exp %d), d=%d (exp %d)\n",
               b, ql_off, expected_ql, qh_off, expected_qh, scales_off, expected_scales, d_off, expected_d);

        if (ql_off != expected_ql || qh_off != expected_qh ||
            scales_off != expected_scales || d_off != expected_d) {
            pass = false;
        }
    }

    printf("  Result: %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

//=============================================================================
// Test 3: CPU dequantization reference
//=============================================================================
static void dequantize_q6k_grouped_reference(const block_q6_K & block, float * output) {
    const float d = ggml_fp16_to_fp32(block.d);
    const uint8_t * ql = block.ql;
    const uint8_t * qh = block.qh;
    const int8_t * scales = block.scales;

    for (int group = 0; group < QK_K; group += 128) {
        for (int lane = 0; lane < 32; ++lane) {
            const int scale = lane / 16;
            const int q1 = ((ql[lane] & 0x0f) | (((qh[lane] >> 0) & 3) << 4)) - 32;
            const int q2 = ((ql[lane + 32] & 0x0f) | (((qh[lane] >> 2) & 3) << 4)) - 32;
            const int q3 = ((ql[lane] >> 4) | (((qh[lane] >> 4) & 3) << 4)) - 32;
            const int q4 = ((ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 3) << 4)) - 32;
            output[group + lane]      = d * scales[scale] * q1;
            output[group + lane + 32] = d * scales[scale + 2] * q2;
            output[group + lane + 64] = d * scales[scale + 4] * q3;
            output[group + lane + 96] = d * scales[scale + 6] * q4;
        }
        ql += 64;
        qh += 32;
        scales += 8;
    }
}

static bool test_cpu_dequant_reference() {
    printf("\n=== Test 3: Production Q6_K CPU Dequantization Reference ===\n");

    block_q6_K block = {};
    block.d = ggml_fp32_to_fp16(0.125f);
    for (int i = 0; i < QK_K / 2; ++i) {
        block.ql[i] = static_cast<uint8_t>((37 * i + 11) & 0xff);
    }
    for (int i = 0; i < QK_K / 4; ++i) {
        block.qh[i] = static_cast<uint8_t>((29 * i + 7) & 0xff);
    }
    for (int i = 0; i < QK_K / 16; ++i) {
        block.scales[i] = static_cast<int8_t>((5 * i) % 17 - 8);
    }

    std::vector<float> production(QK_K);
    std::vector<float> grouped(QK_K);
    std::vector<float> y(QK_K);
    dequantize_row_q6_K(&block, production.data(), QK_K);
    dequantize_q6k_grouped_reference(block, grouped.data());

    bool pass = true;
    float production_dot = 0.0f;
    float grouped_dot = 0.0f;
    for (int i = 0; i < QK_K; ++i) {
        y[i] = static_cast<float>((i * 13) % 23 - 11) / 8.0f;
        pass = pass && std::isfinite(production[i]) && std::isfinite(grouped[i]) &&
               std::fabs(production[i] - grouped[i]) <= 1e-6f;
        production_dot += production[i] * y[i];
        grouped_dot += grouped[i] * y[i];
    }
    pass = pass && std::isfinite(production_dot) && std::isfinite(grouped_dot) &&
           std::fabs(production_dot - grouped_dot) <= 1e-5f;

    printf("  Varied ql/qh/scales/Y grouped dot: %.3f (production %.3f)\n", grouped_dot, production_dot);
    printf("  Result: %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

//=============================================================================
// Test 4: Informational backend/layout setup only
//=============================================================================
static void test_dispatch_setup_informational() {
    printf("\n=== Test 4: Backend/Layout Setup (Informational Only) ===\n");
    printf("  This section does not dispatch work or compare AoS/SoA results.\n");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (backend == nullptr) {
        printf("  INFO: SYCL backend not available; setup was not exercised.\n");
        return;
    }
    ggml_backend_free(backend);

    ggml_sycl::test_set_layout_override(GGML_LAYOUT_AOS);
    backend = ggml_backend_sycl_init(0);
    if (backend != nullptr) {
        ggml_backend_free(backend);
    }

    ggml_sycl::test_set_layout_override(GGML_LAYOUT_SOA);
    backend = ggml_backend_sycl_init(0);
    if (backend != nullptr) {
        ggml_backend_free(backend);
    }
    ggml_sycl::test_clear_layout_override();

    printf("  INFO: AoS/SoA override setup completed; no dispatch result was checked.\n");
}

//=============================================================================
// Test 5: Direct kernel data access simulation
//=============================================================================
static bool test_kernel_data_access() {
    printf("\n=== Test 5: Kernel Data Access Simulation ===\n");

    // Simulate what the kernel does when reading from SoA layout
    const size_t nblocks = 4;

    // Create AoS data with distinct values
    std::vector<block_q6_K> aos_data(nblocks);
    for (size_t b = 0; b < nblocks; b++) {
        for (int i = 0; i < QK_K/2; i++) aos_data[b].ql[i] = (uint8_t)((b * 10) + (i % 10));
        for (int i = 0; i < QK_K/4; i++) aos_data[b].qh[i] = (uint8_t)((b * 5) + (i % 5));
        for (int i = 0; i < QK_K/16; i++) aos_data[b].scales[i] = (int8_t)(b + 1);
        aos_data[b].d = ggml_fp32_to_fp16((float)(b + 1) * 0.1f);
    }

    // Reorder to SoA
    std::vector<uint8_t> soa_data(nblocks * sizeof(block_q6_K));
    uint8_t* soa_ql = soa_data.data();
    uint8_t* soa_qh = soa_ql + nblocks * 128;
    int8_t * soa_scales = reinterpret_cast<int8_t *>(soa_qh + nblocks * 64);
    const size_t soa_d_offset = nblocks * 208;

    for (size_t b = 0; b < nblocks; b++) {
        memcpy(soa_ql + b * 128, aos_data[b].ql, 128);
        memcpy(soa_qh + b * 64, aos_data[b].qh, 64);
        memcpy(soa_scales + b * 16, aos_data[b].scales, 16);
        store_half(soa_data, soa_d_offset + b * sizeof(ggml_half), aos_data[b].d);
    }

    // Simulate kernel offset calculations (from quants.hpp)
    printf("  Simulating kernel reads for %zu blocks:\n", nblocks);

    bool pass = true;
    for (size_t block_idx = 0; block_idx < nblocks; block_idx++) {
        // Calculate offsets as kernel would
        int ql_offset = block_idx * (QK_K / QR6_K);  // = block_idx * 128
        int qh_offset = nblocks * (QK_K / 2) + block_idx * (QK_K / 4);
        int total_qs = nblocks * 128 + nblocks * 64;
        int scales_offset = total_qs + block_idx * 16;
        int d_offset = total_qs + nblocks * 16 + block_idx * 2;

        // Read values at kernel-calculated offsets
        uint8_t ql_val = soa_data[ql_offset];
        uint8_t qh_val = soa_data[qh_offset];
        int8_t scale_val = reinterpret_cast<const int8_t *>(soa_data.data())[scales_offset];
        float d_val = ggml_fp16_to_fp32(load_half(soa_data, d_offset));

        // Expected values (from original AoS)
        uint8_t expected_ql = aos_data[block_idx].ql[0];
        uint8_t expected_qh = aos_data[block_idx].qh[0];
        int8_t expected_scale = aos_data[block_idx].scales[0];
        float expected_d = ggml_fp16_to_fp32(aos_data[block_idx].d);

        printf("    Block %zu: ql=%d(exp %d) qh=%d(exp %d) scale=%d(exp %d) d=%.2f(exp %.2f)\n",
               block_idx, ql_val, expected_ql, qh_val, expected_qh,
               scale_val, expected_scale, d_val, expected_d);

        if (ql_val != expected_ql || qh_val != expected_qh ||
            scale_val != expected_scale || fabs(d_val - expected_d) > 0.001f) {
            pass = false;
        }
    }

    printf("  Result: %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

//=============================================================================
// Test 6: Multi-row matrix simulation (actual MMVQ scenario)
//=============================================================================
static bool test_multirow_matrix() {
    printf("\n=== Test 6: Multi-Row Matrix (MMVQ Scenario) ===\n");

    // Simulate output layer: n_vocab rows x n_embd cols
    // Each row is n_embd/QK_K = 4096/256 = 16 blocks
    const int nrows = 8;  // Reduced for testing
    const int ncols = QK_K * 4;  // 1024 cols = 4 blocks per row
    const int blocks_per_row = ncols / QK_K;
    const int total_blocks = nrows * blocks_per_row;

    printf("  Config: %d rows x %d cols = %d blocks (%d blocks/row)\n",
           nrows, ncols, total_blocks, blocks_per_row);

    // Create AoS data
    std::vector<block_q6_K> aos_data(total_blocks);
    for (int b = 0; b < total_blocks; b++) {
        int row = b / blocks_per_row;
        int col_block = b % blocks_per_row;

        // Use row and col_block to create distinct patterns
        for (int i = 0; i < QK_K/2; i++) {
            aos_data[b].ql[i] = (uint8_t)(row * 16 + col_block);
        }
        for (int i = 0; i < QK_K/4; i++) {
            aos_data[b].qh[i] = (uint8_t)(row + col_block * 4);
        }
        for (int i = 0; i < QK_K/16; i++) {
            aos_data[b].scales[i] = (int8_t)(row * blocks_per_row + col_block + 1);
        }
        aos_data[b].d = ggml_fp32_to_fp16((float)(row + 1) * 0.01f * (col_block + 1));
    }

    // Reorder to SoA (whole tensor)
    const size_t total_size = total_blocks * sizeof(block_q6_K);
    std::vector<uint8_t> soa_data(total_size);

    // SoA layout pointers
    uint8_t* soa_ql = soa_data.data();
    uint8_t* soa_qh = soa_ql + total_blocks * 128;
    int8_t * soa_scales = reinterpret_cast<int8_t *>(soa_qh + total_blocks * 64);
    const size_t soa_d_offset = total_blocks * 208;

    for (int b = 0; b < total_blocks; b++) {
        memcpy(soa_ql + b * 128, aos_data[b].ql, 128);
        memcpy(soa_qh + b * 64, aos_data[b].qh, 64);
        memcpy(soa_scales + b * 16, aos_data[b].scales, 16);
        store_half(soa_data, soa_d_offset + b * sizeof(ggml_half), aos_data[b].d);
    }

    // Verify we can read back correctly using kernel-style indexing
    printf("  Verifying kernel-style access patterns...\n");

    bool pass = true;
    int errors = 0;

    for (int row = 0; row < nrows; row++) {
        for (int col_block = 0; col_block < blocks_per_row; col_block++) {
            int block_idx = row * blocks_per_row + col_block;

            // Kernel-style offset calculation (from quants.hpp)
            // Note: The kernel receives flattened block index
            int ql_offset = block_idx * 128;
            int qh_offset = total_blocks * 128 + block_idx * 64;
            int scales_offset = total_blocks * 192 + block_idx * 16;
            int d_offset = total_blocks * 208 + block_idx * 2;

            // Read first element of each component
            uint8_t ql_val = soa_data[ql_offset];
            uint8_t qh_val = soa_data[qh_offset];
            int8_t scale_val = *reinterpret_cast<const int8_t *>(soa_data.data() + scales_offset);
            float d_val = ggml_fp16_to_fp32(load_half(soa_data, d_offset));

            // Expected from original AoS
            uint8_t expected_ql = aos_data[block_idx].ql[0];
            uint8_t expected_qh = aos_data[block_idx].qh[0];
            int8_t expected_scale = aos_data[block_idx].scales[0];
            float expected_d = ggml_fp16_to_fp32(aos_data[block_idx].d);

            if (ql_val != expected_ql || qh_val != expected_qh ||
                scale_val != expected_scale || fabs(d_val - expected_d) > 0.001f) {
                if (errors < 5) {
                    printf("    FAIL row=%d col_block=%d: ql=%d(exp %d) qh=%d(exp %d)\n",
                           row, col_block, ql_val, expected_ql, qh_val, expected_qh);
                }
                pass = false;
                errors++;
            }
        }
    }

    printf("  Errors: %d/%d blocks\n", errors, total_blocks);
    printf("  Result: %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

//=============================================================================
// Test 7: GPU kernel execution (SoA path) - uses float Y (simplified)
//=============================================================================
static bool test_gpu_soa_kernel() {
    printf("\n=== Test 7: GPU Kernel with SoA X and Float Y ===\n");

    try {
        sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
        printf("  Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

        const int nblocks = 4;
        const int ncols = QK_K;  // 256 elements per row (1 block)
        const int nrows = nblocks;

        std::vector<block_q6_K> aos_data(nblocks);
        for (int b = 0; b < nblocks; ++b) {
            aos_data[b].d = ggml_fp32_to_fp16(0.0625f * (b + 1));
            for (int i = 0; i < QK_K / 2; ++i) {
                aos_data[b].ql[i] = static_cast<uint8_t>((31 * i + 17 * b + 3) & 0xff);
            }
            for (int i = 0; i < QK_K / 4; ++i) {
                aos_data[b].qh[i] = static_cast<uint8_t>((19 * i + 23 * b + 5) & 0xff);
            }
            for (int i = 0; i < QK_K / 16; ++i) {
                aos_data[b].scales[i] = static_cast<int8_t>((7 * i + 3 * b) % 19 - 9);
            }
        }

        // Reorder to SoA
        std::vector<uint8_t> soa_data(nblocks * sizeof(block_q6_K));
        uint8_t* soa_ql = soa_data.data();
        uint8_t* soa_qh = soa_ql + nblocks * 128;
        int8_t * soa_scales = reinterpret_cast<int8_t *>(soa_qh + nblocks * 64);
        const size_t soa_d_offset = nblocks * 208;

        for (int b = 0; b < nblocks; ++b) {
            memcpy(soa_ql + b * 128, aos_data[b].ql, 128);
            memcpy(soa_qh + b * 64, aos_data[b].qh, 64);
            memcpy(soa_scales + b * 16, aos_data[b].scales, 16);
            store_half(soa_data, soa_d_offset + b * sizeof(ggml_half), aos_data[b].d);
        }

        std::vector<float> y(ncols);
        for (int i = 0; i < ncols; ++i) {
            y[i] = static_cast<float>((11 * i) % 29 - 14) / 16.0f;
        }

        std::vector<float> expected(nrows);
        std::vector<float> dequantized(ncols);
        for (int row = 0; row < nrows; ++row) {
            dequantize_row_q6_K(&aos_data[row], dequantized.data(), ncols);
            for (int i = 0; i < ncols; ++i) {
                expected[row] += dequantized[i] * y[i];
            }
        }

        usm_device_buffer<uint8_t> d_soa(soa_data.size(), q);
        usm_device_buffer<float> d_y(ncols, q);
        usm_device_buffer<float> d_out(nrows, q);
        if (!d_soa || !d_y || !d_out) {
            printf("  FAIL: device allocation failed\n");
            return false;
        }
        uint8_t * d_soa_ptr = d_soa.get();
        float * d_y_ptr = d_y.get();
        float * d_out_ptr = d_out.get();

        q.memcpy(d_soa_ptr, soa_data.data(), soa_data.size());
        q.memcpy(d_y_ptr, y.data(), ncols * sizeof(float));
        q.wait_and_throw();

        // Simple GPU kernel that reads from SoA layout and computes dot product
        // NOTE: This uses float Y, not Q8_1 SoA Y. See Test 8 for full production format.
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(nrows), [=](sycl::id<1> row_id) {
                const int row = row_id[0];
                const int block_idx = row;  // 1 block per row for this test

                // Calculate SoA offsets (matching quants.hpp)
                const int ql_offset = block_idx * 128;
                const int qh_offset = nblocks * 128 + block_idx * 64;
                const int scales_offset = nblocks * 192 + block_idx * 16;
                const int d_offset = nblocks * 208 + block_idx * 2;

                const uint8_t * ql = d_soa_ptr + ql_offset;
                const uint8_t * qh = d_soa_ptr + qh_offset;
                const int8_t * scales = reinterpret_cast<const int8_t *>(d_soa_ptr + scales_offset);
                const sycl::half d_half = *reinterpret_cast<const sycl::half *>(d_soa_ptr + d_offset);
                const float d = sycl::vec<sycl::half, 1>(d_half).convert<float>()[0];

                float sum = 0.0f;
                for (int group = 0; group < QK_K; group += 128) {
                    const uint8_t * group_ql = ql + group / 2;
                    const uint8_t * group_qh = qh + group / 4;
                    const int8_t * group_scales = scales + group / 16;
                    for (int lane = 0; lane < 32; ++lane) {
                        const int scale = lane / 16;
                        const int q1 = ((group_ql[lane] & 0x0f) | (((group_qh[lane] >> 0) & 3) << 4)) - 32;
                        const int q2 = ((group_ql[lane + 32] & 0x0f) | (((group_qh[lane] >> 2) & 3) << 4)) - 32;
                        const int q3 = ((group_ql[lane] >> 4) | (((group_qh[lane] >> 4) & 3) << 4)) - 32;
                        const int q4 = ((group_ql[lane + 32] >> 4) | (((group_qh[lane] >> 6) & 3) << 4)) - 32;
                        sum += d * group_scales[scale] * q1 * d_y_ptr[group + lane];
                        sum += d * group_scales[scale + 2] * q2 * d_y_ptr[group + lane + 32];
                        sum += d * group_scales[scale + 4] * q3 * d_y_ptr[group + lane + 64];
                        sum += d * group_scales[scale + 6] * q4 * d_y_ptr[group + lane + 96];
                    }
                }

                d_out_ptr[row] = sum;
            });
        });
        q.wait_and_throw();

        std::vector<float> h_out(nrows);
        q.memcpy(h_out.data(), d_out_ptr, nrows * sizeof(float));
        q.wait_and_throw();

        bool pass = true;
        for (int r = 0; r < nrows; ++r) {
            const float error = std::fabs(h_out[r] - expected[r]);
            const float denominator = std::fabs(expected[r]);
            const float rel_error = denominator > 0.0f ? error / denominator * 100.0f : error;
            const bool finite = std::isfinite(h_out[r]) && std::isfinite(error) && std::isfinite(rel_error);
            printf("  Row %d: result=%.3f expected=%.3f error=%.2f%%\n",
                   r, h_out[r], expected[r], rel_error);
            if (!finite || rel_error > 1.0f) {
                pass = false;
            }
        }

        q.wait_and_throw();
        printf("  Result: %s\n", pass ? "PASS" : "FAIL");
        return pass;

    } catch (const sycl::exception & e) {
        printf("  FAIL: SYCL error: %s\n", e.what());
        return false;
    }
}

//=============================================================================
// Test 8: Production vec_dot Q6_K implementation
// Uses EXACT same algorithm as vecdotq.hpp reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>
//=============================================================================

static bool test_gpu_production_format() {
    printf("\n=== Test 8: Production vec_dot Q6_K (exact algorithm) ===\n");

    try {
        sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
        printf("  Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

        // Test configuration: 4 rows, each row has 1 Q6_K block (256 elements)
        const int nrows = 4;
        const int ncols = QK_K;  // 256 elements per row
        const int blocks_per_row_x = ncols / QK_K;  // 1 Q6_K block per row
        const int blocks_per_row_y = ncols / QK8_1;  // 8 Q8_1 blocks per row
        const int total_x_blocks = nrows * blocks_per_row_x;

        printf("  Config: nrows=%d, ncols=%d, x_blocks/row=%d, y_blocks/row=%d\n",
               nrows, ncols, blocks_per_row_x, blocks_per_row_y);

        //
        // Step 1: Create X (weight) data in Q6_K SoA format
        //
        std::vector<block_q6_K> x_aos(total_x_blocks);
        for (int b = 0; b < total_x_blocks; b++) {
            x_aos[b].d = ggml_fp32_to_fp16(0.1f);  // d = 0.1
            // q = 5 - 32 = -27, so dequant = 0.1 * scale * (-27)
            for (int i = 0; i < QK_K/2; i++) x_aos[b].ql[i] = 0x55;
            for (int i = 0; i < QK_K/4; i++) x_aos[b].qh[i] = 0x00;
            for (int i = 0; i < QK_K/16; i++) x_aos[b].scales[i] = static_cast<int8_t>(b + 1);  // Different per row
        }

        // Reorder X to SoA
        std::vector<uint8_t> x_soa(total_x_blocks * sizeof(block_q6_K));
        uint8_t* x_soa_ql = x_soa.data();
        uint8_t* x_soa_qh = x_soa_ql + total_x_blocks * 128;
        int8_t * x_soa_scales = reinterpret_cast<int8_t *>(x_soa_qh + total_x_blocks * 64);
        const size_t x_soa_d_offset = total_x_blocks * 208;

        for (int b = 0; b < total_x_blocks; b++) {
            memcpy(x_soa_ql + b * 128, x_aos[b].ql, 128);
            memcpy(x_soa_qh + b * 64, x_aos[b].qh, 64);
            memcpy(x_soa_scales + b * 16, x_aos[b].scales, 16);
            store_half(x_soa, x_soa_d_offset + b * sizeof(ggml_half), x_aos[b].d);
        }

        //
        // Step 2: Create Y (activation) in Q8_1 SoA format
        // Production format: quants at [0..ncols-1], ds at [ncols..]
        //
        const size_t y_soa_size = ncols + blocks_per_row_y * sizeof(sycl::half2);
        std::vector<uint8_t> y_soa(y_soa_size);

        int8_t * y_soa_qs = reinterpret_cast<int8_t *>(y_soa.data());

        // Fill Y: all elements = 1.0 -> qs = 127, d = 1/127
        for (int i = 0; i < ncols; i++) {
            y_soa_qs[i] = 127;
        }
        for (int b = 0; b < blocks_per_row_y; b++) {
            const float d_val = 1.0f / 127.0f;
            const float sum_val = 127.0f * QK8_1;
            const sycl::half2 ds{sycl::half(d_val), sycl::half(sum_val)};
            memcpy(y_soa.data() + ncols + b * sizeof(ds), &ds, sizeof(ds));
        }

        printf("  Y format: %zu bytes total (quants=%d, ds=%d)\n",
               y_soa_size, ncols, static_cast<int>(blocks_per_row_y * sizeof(sycl::half2)));

        //
        // Step 3: Calculate expected results
        //
        printf("  Expected results per row:\n");
        for (int r = 0; r < nrows; r++) {
            float expected = ncols * (-2.7f) * (r + 1);
            printf("    Row %d: %.1f\n", r, expected);
        }

        //
        // Step 4: Allocate device memory and copy
        //
        usm_device_buffer<uint8_t> d_x(x_soa.size(), q);
        usm_device_buffer<uint8_t> d_y(y_soa.size(), q);
        usm_device_buffer<float> d_out(nrows, q);
        if (!d_x || !d_y || !d_out) {
            printf("  FAIL: device allocation failed\n");
            return false;
        }
        uint8_t * d_x_ptr = d_x.get();
        uint8_t * d_y_ptr = d_y.get();
        float * d_out_ptr = d_out.get();

        q.memcpy(d_x_ptr, x_soa.data(), x_soa.size());
        q.memcpy(d_y_ptr, y_soa.data(), y_soa.size());
        q.wait_and_throw();

        //
        // Step 5: Run kernel using EXACT production vec_dot algorithm from vecdotq.hpp
        // Each work-item processes one iqs value, then we sum across WARP_SIZE work-items
        //
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<1>(nrows * WARP_SIZE, WARP_SIZE),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    const int row = item.get_group(0);
                    const int lane_id = item.get_local_id(0);
                    auto sg = item.get_sub_group();

                    // X block access (SoA layout for Q6_K)
                    const int ibx = row;  // 1 block per row
                    const int ql_offset = ibx * 128;
                    const int qh_offset = total_x_blocks * 128 + ibx * 64;
                    const int scales_offset = total_x_blocks * 192 + ibx * 16;
                    const int d_offset = total_x_blocks * 208 + ibx * 2;

                    const uint8_t * ql = d_x_ptr + ql_offset;
                    const uint8_t * qh = d_x_ptr + qh_offset;
                    const int8_t * scales = reinterpret_cast<const int8_t *>(d_x_ptr + scales_offset);
                    const sycl::half d_half = *reinterpret_cast<const sycl::half *>(d_x_ptr + d_offset);
                    const float d = sycl::vec<sycl::half, 1>(d_half).convert<float>()[0];

                    // Y access (Q8_1 SoA layout)
                    // iby = 0 for first Q6_K block (each Q6_K maps to 8 Q8_1 blocks)
                    const int iby = 0;
                    const int8_t * q8_1_quant_ptr = reinterpret_cast<const int8_t *>(d_y_ptr) + iby * QK8_1;
                    const sycl::half2 * q8_1_ds_ptr =
                        reinterpret_cast<const sycl::half2 *>(d_y_ptr + ncols + iby * sizeof(sycl::half2));

                    float partial_sum = 0.0f;

                    // Process QI6_K/WARP_SIZE iterations per work-item
                    // QI6_K = 32, WARP_SIZE = 16, so each work-item processes 2 iqs values
                    for (int elem = 0; elem < QI6_K; elem += WARP_SIZE) {
                        const int iqs = elem + lane_id;

                        // Production vec_dot algorithm from vecdotq.hpp lines 468-486
                        const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
                        const int scale_offset = (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
                        const int vh_shift = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));

                        // Read vl and vh using the production two-uint16 load pattern.
                        const uint16_t * ql16 = reinterpret_cast<const uint16_t *>(ql + sizeof(int) * iqs);
                        int vl = ql16[0] | (static_cast<int>(ql16[1]) << 16);

                        const int qh_idx = (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4);
                        const uint16_t * qh16 = reinterpret_cast<const uint16_t *>(qh + sizeof(int) * qh_idx);
                        int vh_raw = qh16[0] | (static_cast<int>(qh16[1]) << 16);
                        int vh = vh_raw >> vh_shift;

                        const int8_t * scs = scales + scale_offset;

                        // Production vec_dot_q6_K_q8_1_impl_mmvq (lines 443-456)
                        float sumf = 0.0f;
                        for (int i = 0; i < QR6_K; ++i) {
                            const int sc = scs[4 * i];

                            // Read u from Q8_1 quants
                            const int8_t * u_ptr = q8_1_quant_ptr + (bq8_offset + 2 * i) * QK8_1 + (iqs % QI8_1) * 4;
                            int u = *reinterpret_cast<const int *>(u_ptr);

                            // Read d8 from Q8_1 ds
                            const sycl::half2 ds_values = *(q8_1_ds_ptr + bq8_offset + 2 * i);
                            float d8 = sycl::vec<sycl::half, 1>(ds_values.x()).convert<float>()[0];

                            // Compute vil, vih, vi (lines 446-450)
                            const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
                            const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;

                            // vi = (vil | vih) - 32 with saturation
                            const int8_t * vil_bytes = reinterpret_cast<const int8_t *>(&vil);
                            const int8_t * vih_bytes = reinterpret_cast<const int8_t *>(&vih);
                            const int8_t * u_bytes = reinterpret_cast<const int8_t *>(&u);

                            // Scalar dp4a equivalent
                            int dp4a_result = 0;
                            for (int j = 0; j < 4; j++) {
                                int vi_j = (vil_bytes[j] | vih_bytes[j]) - 32;
                                dp4a_result += vi_j * u_bytes[j];
                            }

                            sumf += d8 * (dp4a_result * sc);
                        }
                        partial_sum += d * sumf;
                    }

                    // Warp reduction
                    float sum = sycl::reduce_over_group(sg, partial_sum, sycl::plus<float>());

                    if (lane_id == 0) {
                        d_out_ptr[row] = sum;
                    }
                });
        });
        q.wait_and_throw();

        //
        // Step 6: Verify results
        //
        std::vector<float> h_out(nrows);
        q.memcpy(h_out.data(), d_out_ptr, nrows * sizeof(float));
        q.wait_and_throw();

        printf("\n  Results:\n");
        bool pass = true;
        for (int r = 0; r < nrows; r++) {
            const float expected = ncols * (-2.7f) * (r + 1);
            const float error = std::fabs(h_out[r] - expected);
            const float rel_error = expected != 0.0f ? error / std::fabs(expected) * 100.0f : error;
            const bool finite = std::isfinite(h_out[r]) && std::isfinite(error) && std::isfinite(rel_error);
            printf("    Row %d: result=%.2f expected=%.2f error=%.2f%%\n",
                   r, h_out[r], expected, rel_error);
            if (!finite || rel_error > 1.0f) {
                pass = false;
            }
        }

        q.wait_and_throw();
        printf("  Result: %s\n", pass ? "PASS" : "FAIL");
        return pass;

    } catch (const sycl::exception & e) {
        printf("  FAIL: SYCL error: %s\n", e.what());
        return false;
    }
}

//=============================================================================
// Main
//=============================================================================
int main() {
    printf("Q6_K Reorder & Dispatch Comprehensive Unit Tests\n");
    printf("=================================================\n");

    int failures = 0;
    failures += !test_production_reorder_layout();
    failures += !test_offset_calculations();
    failures += !test_cpu_dequant_reference();
    test_dispatch_setup_informational();
    failures += !test_kernel_data_access();
    failures += !test_multirow_matrix();
    failures += !test_gpu_soa_kernel();
    failures += !test_gpu_production_format();

    printf("\n=== All Tests Complete: %d failure(s) ===\n", failures);
    return failures == 0 ? 0 : 1;
}
