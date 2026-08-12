// Executable device regression test for the allocation-free admitted adapter.
#include "mmvq.hpp"
#include "ggml-quants.h"

#include <climits>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename T> struct usm_owner {
    sycl::queue * q = nullptr;
    T * ptr = nullptr;
    usm_owner(sycl::queue & queue, size_t count) : q(&queue), ptr(sycl::malloc_shared<T>(count, queue)) {
        if (!ptr) throw std::bad_alloc();
    }
    ~usm_owner() { if (ptr) sycl::free(ptr, *q); }
};

void run_case(sycl::queue & q, ggml_type type, int ne11) {
    constexpr int experts = 4, top_k = 3, tokens = 2, rows = 5;
    const int K = type == GGML_TYPE_Q1_0 ? QK1_0 : QK_NVFP4;
    const size_t block_bytes = type == GGML_TYPE_Q1_0 ? sizeof(block_q1_0) : sizeof(block_nvfp4);
    const size_t blocks_per_row = static_cast<size_t>(K / (type == GGML_TYPE_Q1_0 ? QK1_0 : QK_NVFP4));
    const size_t weight_bytes = static_cast<size_t>(experts) * rows * blocks_per_row * block_bytes;
    const size_t activation_rows = static_cast<size_t>(ne11) * tokens;
    const size_t q8_bytes = activation_rows * static_cast<size_t>(K / QK8_1) * sizeof(block_q8_1);
    const size_t output_values = static_cast<size_t>(top_k) * tokens * rows;

    usm_owner<unsigned char> weights(q, weight_bytes);
    usm_owner<const void *> table(q, experts);
    usm_owner<float> activation(q, activation_rows * K);
    usm_owner<int32_t> ids(q, top_k * tokens);
    usm_owner<unsigned char> q8(q, q8_bytes + alignof(block_q8_1));
    usm_owner<float> output(q, output_values + 1);

    std::vector<float> weight_row(K);
    for (int e = 0; e < experts; ++e) {
        table.ptr[e] = weights.ptr + static_cast<size_t>(e) * rows * blocks_per_row * block_bytes;
        for (int r = 0; r < rows; ++r) {
            for (int k = 0; k < K; ++k) weight_row[k] = 0.25f + float((e + r + k) % 11) / 13.0f;
            void * row = weights.ptr + (static_cast<size_t>(e) * rows + r) * blocks_per_row * block_bytes;
            if (type == GGML_TYPE_Q1_0) quantize_row_q1_0_ref(weight_row.data(), static_cast<block_q1_0 *>(row), K);
            else quantize_row_nvfp4_ref(weight_row.data(), static_cast<block_nvfp4 *>(row), K);
        }
    }
    for (size_t i = 0; i < activation_rows * static_cast<size_t>(K); ++i) activation.ptr[i] = 0.5f + float(i % 17) / 19.0f;
    const int32_t snapshot[] = { 3, 1, 3, 0, 2, 0 }; // repeated and nonmonotonic
    std::fill(ids.ptr, ids.ptr + top_k * tokens, -1); // adapter must upload snapshot, not trust this
    std::memset(q8.ptr, 0x5a, q8_bytes);
    std::fill(output.ptr, output.ptr + output_values, 123.0f);

    float * activation_ptr = activation.ptr;
    sycl::event delayed = q.submit([&](sycl::handler & h) {
        h.single_task([=]() {
            volatile int spin = 0;
            for (int i = 0; i < 100000; ++i) spin += i;
            activation_ptr[0] = 2.0f + float(spin == -1);
        });
    });
    mmvq_q1_nvfp4_admitted_buffers buffers{ q8.ptr, q8_bytes, output.ptr, output_values * sizeof(float) };
    sycl::event terminal;
    require(mmvq_submit_q1_nvfp4_aos_id_admitted(q, type, GGML_LAYOUT_AOS, table.ptr, activation.ptr, ids.ptr,
                                                  snapshot, experts, K, rows, top_k, tokens, ne11, sizeof(int32_t),
                                                  top_k * sizeof(int32_t), buffers, &delayed, &terminal),
            "valid admitted submit rejected");
    terminal.wait_and_throw();
    bool any_nonzero = false;
    for (size_t i = 0; i < output_values; ++i) {
        require(std::isfinite(output.ptr[i]), "non-finite output");
        any_nonzero |= output.ptr[i] != 0.0f;
    }
    require(any_nonzero, "nonzero weights produced only zero output");
    require(std::memcmp(ids.ptr, snapshot, sizeof(snapshot)) == 0, "retained ID snapshot was not uploaded exactly");

    auto expect_pre_submit_refusal = [&](const char * name, ggml_type bad_type, int bad_k, const int32_t * bad_ids,
                                         int64_t nb0, int64_t nb1, void * q8_ptr, size_t q8_size,
                                         size_t output_size) {
        std::memset(q8.ptr, 0x6b, q8_bytes);
        std::fill(output.ptr, output.ptr + output_values, 77.0f);
        mmvq_q1_nvfp4_admitted_buffers bad{ q8_ptr, q8_size, output.ptr, output_size };
        require(!mmvq_submit_q1_nvfp4_aos_id_admitted(q, bad_type, GGML_LAYOUT_AOS, table.ptr, activation.ptr, ids.ptr,
                                                       bad_ids, experts, bad_k, rows, top_k, tokens, ne11, nb0, nb1,
                                                       bad), name);
        for (size_t i = 0; i < q8_bytes; ++i) require(q8.ptr[i] == 0x6b, "refusal changed Q8 bytes");
        for (size_t i = 0; i < output_values; ++i) require(output.ptr[i] == 77.0f, "refusal changed output");
    };
    int32_t invalid_ids[] = { 3, 1, experts, 0, 2, 0 };
    expect_pre_submit_refusal("unsupported type submitted", GGML_TYPE_F32, K, snapshot, sizeof(int32_t),
                              top_k * sizeof(int32_t), q8.ptr, q8_bytes, output_values * sizeof(float));
    expect_pre_submit_refusal("zero K submitted", type, 0, snapshot, sizeof(int32_t), top_k * sizeof(int32_t), q8.ptr,
                              q8_bytes, output_values * sizeof(float));
    expect_pre_submit_refusal("T-1 Q8 submitted", type, K, snapshot, sizeof(int32_t), top_k * sizeof(int32_t), q8.ptr,
                              q8_bytes - 1, output_values * sizeof(float));
    expect_pre_submit_refusal("output mismatch submitted", type, K, snapshot, sizeof(int32_t),
                              top_k * sizeof(int32_t), q8.ptr, q8_bytes, output_values * sizeof(float) - 1);
    expect_pre_submit_refusal("invalid ID submitted", type, K, invalid_ids, sizeof(int32_t),
                              top_k * sizeof(int32_t), q8.ptr, q8_bytes, output_values * sizeof(float));
    expect_pre_submit_refusal("invalid slot stride submitted", type, K, snapshot, 2 * sizeof(int32_t),
                              top_k * sizeof(int32_t), q8.ptr, q8_bytes, output_values * sizeof(float));
    expect_pre_submit_refusal("invalid token stride submitted", type, K, snapshot, sizeof(int32_t),
                              (top_k + 1) * sizeof(int32_t), q8.ptr, q8_bytes, output_values * sizeof(float));
    expect_pre_submit_refusal("misaligned Q8 submitted", type, K, snapshot, sizeof(int32_t),
                              top_k * sizeof(int32_t), q8.ptr + 1, q8_bytes, output_values * sizeof(float));

    mmvq_q1_nvfp4_admitted_buffers overflow{ q8.ptr, q8_bytes, output.ptr, output_values * sizeof(float) };
    require(!mmvq_submit_q1_nvfp4_aos_id_admitted(q, type, GGML_LAYOUT_AOS, table.ptr, activation.ptr, ids.ptr,
                                                   snapshot, experts, K, rows, INT_MAX, INT_MAX, ne11,
                                                   sizeof(int32_t), top_k * sizeof(int32_t), overflow),
            "overflow shape submitted");
}

} // namespace

int main() {
    try {
        sycl::queue q{ sycl::gpu_selector_v };
        run_case(q, GGML_TYPE_Q1_0, 1);
        run_case(q, GGML_TYPE_Q1_0, 3);
        run_case(q, GGML_TYPE_NVFP4, 1);
        run_case(q, GGML_TYPE_NVFP4, 3);
        std::cout << "Q1/NVFP4 admitted device adapter: PASS\n";
        return 0;
    } catch (const sycl::exception & e) {
        std::cerr << "SKIP: no usable SYCL GPU: " << e.what() << '\n';
        return 77;
    } catch (const std::exception & e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
