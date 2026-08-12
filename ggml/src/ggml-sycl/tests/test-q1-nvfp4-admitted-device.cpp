// Executable device smoke/regression test for the allocation-free admitted adapter.
#include "mmvq.hpp"

#include <cassert>
#include <climits>
#include <cmath>
#include <cstring>
#include <iostream>

namespace {

template <typename T> struct usm_owner {
    sycl::queue * q = nullptr;
    T * ptr = nullptr;
    usm_owner(sycl::queue & queue, size_t count) : q(&queue), ptr(sycl::malloc_shared<T>(count, queue)) {
        if (!ptr) throw std::bad_alloc();
    }
    ~usm_owner() { if (ptr) sycl::free(ptr, *q); }
};

void run_case(sycl::queue & q, ggml_type type, int ne11) {
    constexpr int experts = 4;
    constexpr int top_k = 3;
    constexpr int tokens = 2;
    constexpr int rows = 5;
    const int K = type == GGML_TYPE_Q1_0 ? QK1_0 : QK_NVFP4;
    const size_t block_bytes = type == GGML_TYPE_Q1_0 ? sizeof(block_q1_0) : sizeof(block_nvfp4);
    const size_t weight_bytes = static_cast<size_t>(experts) * rows * block_bytes;
    const size_t activation_rows = static_cast<size_t>(ne11) * tokens;
    const size_t q8_bytes = activation_rows * sizeof(block_q8_1);
    const size_t output_values = static_cast<size_t>(top_k) * tokens * rows;

    usm_owner<unsigned char> weights(q, weight_bytes);
    usm_owner<const void *> table(q, experts);
    usm_owner<float> activation(q, activation_rows * K);
    usm_owner<int32_t> ids(q, top_k * tokens);
    usm_owner<unsigned char> q8(q, q8_bytes);
    usm_owner<float> output(q, output_values);
    std::memset(weights.ptr, 0, weight_bytes); // zero scales make both weight formats exact zero matrices
    for (int e = 0; e < experts; ++e) table.ptr[e] = weights.ptr + static_cast<size_t>(e) * rows * block_bytes;
    for (size_t i = 0; i < activation_rows * static_cast<size_t>(K); ++i) activation.ptr[i] = float(i % 17) - 8.0f;
    const int32_t nonmonotonic_repeated[] = { 3, 1, 3, 0, 2, 0 };
    std::memcpy(ids.ptr, nonmonotonic_repeated, sizeof(nonmonotonic_repeated));
    std::memset(q8.ptr, 0x5a, q8_bytes);
    std::fill(output.ptr, output.ptr + output_values, 123.0f);

    float * activation_ptr = activation.ptr;
    sycl::event dependency = q.submit([&](sycl::handler & h) { h.single_task([=]() { activation_ptr[0] = 7.0f; }); });
    mmvq_q1_nvfp4_admitted_buffers buffers{ q8.ptr, q8_bytes, output.ptr, output_values * sizeof(float) };
    sycl::event terminal;
    assert(mmvq_submit_q1_nvfp4_aos_id_admitted(q, type, GGML_LAYOUT_AOS, table.ptr, activation.ptr, ids.ptr,
                                                 nonmonotonic_repeated, experts, K, rows, top_k, tokens, ne11,
                                                 sizeof(int32_t), top_k * sizeof(int32_t), buffers, &dependency,
                                                 &terminal));
    terminal.wait_and_throw();
    for (size_t i = 0; i < output_values; ++i) assert(std::isfinite(output.ptr[i]) && output.ptr[i] == 0.0f);

    // Every refusal is before quantization: sentinel Q8 bytes must remain unchanged.
    std::memset(q8.ptr, 0x6b, q8_bytes);
    auto rejected = [&](ggml_type rejected_type, int rejected_k, size_t q8_size, int rejected_top_k,
                        int rejected_tokens) {
        mmvq_q1_nvfp4_admitted_buffers bad{ q8.ptr, q8_size, output.ptr, output_values * sizeof(float) };
        return mmvq_submit_q1_nvfp4_aos_id_admitted(
            q, rejected_type, GGML_LAYOUT_AOS, table.ptr, activation.ptr, ids.ptr, nonmonotonic_repeated, experts,
            rejected_k, rows, rejected_top_k, rejected_tokens, ne11, sizeof(int32_t), top_k * sizeof(int32_t), bad);
    };
    assert(!rejected(GGML_TYPE_F32, K, q8_bytes, top_k, tokens));
    assert(!rejected(type, 0, q8_bytes, top_k, tokens));
    assert(!rejected(type, K, q8_bytes - 1, top_k, tokens));
    assert(!rejected(type, K, q8_bytes, INT_MAX, INT_MAX));
    for (size_t i = 0; i < q8_bytes; ++i) assert(q8.ptr[i] == 0x6b);
}

} // namespace

int main() {
    try {
        sycl::queue q;
        run_case(q, GGML_TYPE_Q1_0, 1);
        run_case(q, GGML_TYPE_Q1_0, 3);
        run_case(q, GGML_TYPE_NVFP4, 1);
        run_case(q, GGML_TYPE_NVFP4, 3);
        std::cout << "Q1/NVFP4 admitted device adapter: PASS\n";
        return 0;
    } catch (const sycl::exception & e) {
        std::cerr << "SKIP: no usable SYCL device: " << e.what() << '\n';
        return 77;
    }
}
