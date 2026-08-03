#include "cpu-traits-support.hpp"

#include "ggml-quants.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>

namespace {

static_assert(QK_K % QK1_0 == 0 && QK_K % QK2_0 == 0 && QK_K % QK4_0 == 0 &&
                  QK_K % QK4_1 == 0 && QK_K % QK5_0 == 0 && QK_K % QK5_1 == 0 &&
                  QK_K % QK8_0 == 0 && QK_K % QK8_1 == 0 && QK_K % QK_MXFP4 == 0 &&
                  QK_K % QK_NVFP4 == 0 && QK_K % QK4_NL == 0,
              "portable vec-dot stack tile must be a multiple of every supported quant block");
constexpr int k_vec_dot_tile = QK_K;

static void from_float_f32(const float * src, void * dst, int64_t n) {
    std::memcpy(dst, src, static_cast<size_t>(n) * sizeof(float));
}

static void from_float_q8_k(const float * src, void * dst, int64_t n) {
    quantize_row_q8_K_ref(src, static_cast<block_q8_K *>(dst), n);
}

template<enum ggml_type Type>
static void from_float_ref(const float * src, void * dst, int64_t n) {
    const auto * traits = ggml_get_type_traits(Type);
    if (traits && traits->from_float_ref) {
        traits->from_float_ref(src, dst, n);
    }
}

template<enum ggml_type Type>
static void row_to_float(const ggml_type_traits * traits, const void * src, float * dst, int64_t n) {
    if constexpr (Type == GGML_TYPE_F32) {
        std::memcpy(dst, src, static_cast<size_t>(n) * sizeof(float));
    } else if constexpr (Type == GGML_TYPE_Q8_1) {
        const auto * blocks = static_cast<const block_q8_1 *>(src);
        for (int64_t i = 0; i < n / QK8_1; ++i) {
            const float d = ggml_fp16_to_fp32(blocks[i].d);
            for (int j = 0; j < QK8_1; ++j) {
                dst[i * QK8_1 + j] = d * blocks[i].qs[j];
            }
        }
    } else if constexpr (Type == GGML_TYPE_Q8_K) {
        dequantize_row_q8_K(static_cast<const block_q8_K *>(src), dst, n);
    } else if (traits && traits->to_float) {
        traits->to_float(src, dst, n);
    }
}

template<enum ggml_type Type>
static size_t element_offset(int element) {
    return static_cast<size_t>(element / ggml_blck_size(Type)) * ggml_type_size(Type);
}

template<enum ggml_type X, enum ggml_type Y>
static void vec_dot_ref(int n, float * s, size_t bs, const void * vx, size_t bx,
                        const void * vy, size_t by, int nrc) {
    // Resolve immutable ggml-base conversion traits once, before every row and
    // block loop. All scratch is fixed-size stack storage: every supported
    // block size divides QK_K, so each conversion receives complete blocks.
    const ggml_type_traits * x_traits = ggml_get_type_traits(X);
    const ggml_type_traits * y_traits = ggml_get_type_traits(Y);
    const size_t xrow = bx ? bx : ggml_row_size(X, n);
    const size_t yrow = by ? by : ggml_row_size(Y, n);
    const size_t sout = bs ? bs : sizeof(float);
    std::array<float, k_vec_dot_tile> x{};
    std::array<float, k_vec_dot_tile> y{};

    for (int row = 0; row < nrc; ++row) {
        const auto * xbase = static_cast<const uint8_t *>(vx) + row * xrow;
        const auto * ybase = static_cast<const uint8_t *>(vy) + row * yrow;
        float sum = 0.0f;
        for (int offset = 0; offset < n; offset += k_vec_dot_tile) {
            const int count = std::min(k_vec_dot_tile, n - offset);
            row_to_float<X>(x_traits, xbase + element_offset<X>(offset), x.data(), count);
            row_to_float<Y>(y_traits, ybase + element_offset<Y>(offset), y.data(), count);
            for (int i = 0; i < count; ++i) {
                sum += x[static_cast<size_t>(i)] * y[static_cast<size_t>(i)];
            }
        }
        *reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(s) + row * sout) = sum;
    }
}

static std::array<ggml_type_traits_cpu, GGML_TYPE_COUNT> make_baseline() {
    std::array<ggml_type_traits_cpu, GGML_TYPE_COUNT> table{};
#define TRAIT(T, FROM, DOT, VDT) table[GGML_TYPE_##T] = { FROM, DOT, GGML_TYPE_##VDT, 1 }
    TRAIT(F32, from_float_f32, (vec_dot_ref<GGML_TYPE_F32, GGML_TYPE_F32>), F32);
    TRAIT(F16, (from_float_ref<GGML_TYPE_F16>), (vec_dot_ref<GGML_TYPE_F16, GGML_TYPE_F16>), F16);
    TRAIT(Q1_0, (from_float_ref<GGML_TYPE_Q1_0>), (vec_dot_ref<GGML_TYPE_Q1_0, GGML_TYPE_Q8_0>), Q8_0);
    TRAIT(Q2_0, (from_float_ref<GGML_TYPE_Q2_0>), (vec_dot_ref<GGML_TYPE_Q2_0, GGML_TYPE_Q8_0>), Q8_0);
    TRAIT(Q4_0, (from_float_ref<GGML_TYPE_Q4_0>), (vec_dot_ref<GGML_TYPE_Q4_0, GGML_TYPE_Q8_0>), Q8_0);
    TRAIT(Q4_1, (from_float_ref<GGML_TYPE_Q4_1>), (vec_dot_ref<GGML_TYPE_Q4_1, GGML_TYPE_Q8_1>), Q8_1);
    TRAIT(Q5_0, (from_float_ref<GGML_TYPE_Q5_0>), (vec_dot_ref<GGML_TYPE_Q5_0, GGML_TYPE_Q8_0>), Q8_0);
    TRAIT(Q5_1, (from_float_ref<GGML_TYPE_Q5_1>), (vec_dot_ref<GGML_TYPE_Q5_1, GGML_TYPE_Q8_1>), Q8_1);
    TRAIT(Q8_0, (from_float_ref<GGML_TYPE_Q8_0>), (vec_dot_ref<GGML_TYPE_Q8_0, GGML_TYPE_Q8_0>), Q8_0);
    TRAIT(Q8_1, (from_float_ref<GGML_TYPE_Q8_1>), nullptr, Q8_1);
    TRAIT(MXFP4, (from_float_ref<GGML_TYPE_MXFP4>), (vec_dot_ref<GGML_TYPE_MXFP4, GGML_TYPE_Q8_0>), Q8_0);
    TRAIT(NVFP4, (from_float_ref<GGML_TYPE_NVFP4>), (vec_dot_ref<GGML_TYPE_NVFP4, GGML_TYPE_Q8_0>), Q8_0);
    TRAIT(Q2_K, (from_float_ref<GGML_TYPE_Q2_K>), (vec_dot_ref<GGML_TYPE_Q2_K, GGML_TYPE_Q8_K>), Q8_K);
    TRAIT(Q3_K, (from_float_ref<GGML_TYPE_Q3_K>), (vec_dot_ref<GGML_TYPE_Q3_K, GGML_TYPE_Q8_K>), Q8_K);
    TRAIT(Q4_K, (from_float_ref<GGML_TYPE_Q4_K>), (vec_dot_ref<GGML_TYPE_Q4_K, GGML_TYPE_Q8_K>), Q8_K);
    TRAIT(Q5_K, (from_float_ref<GGML_TYPE_Q5_K>), (vec_dot_ref<GGML_TYPE_Q5_K, GGML_TYPE_Q8_K>), Q8_K);
    TRAIT(Q6_K, (from_float_ref<GGML_TYPE_Q6_K>), (vec_dot_ref<GGML_TYPE_Q6_K, GGML_TYPE_Q8_K>), Q8_K);
    TRAIT(IQ2_XXS, nullptr, (vec_dot_ref<GGML_TYPE_IQ2_XXS, GGML_TYPE_Q8_K>), Q8_K);
    TRAIT(IQ2_XS, nullptr, (vec_dot_ref<GGML_TYPE_IQ2_XS, GGML_TYPE_Q8_K>), Q8_K);
    TRAIT(IQ3_XXS, nullptr, (vec_dot_ref<GGML_TYPE_IQ3_XXS, GGML_TYPE_Q8_K>), Q8_K);
    TRAIT(IQ3_S, nullptr, (vec_dot_ref<GGML_TYPE_IQ3_S, GGML_TYPE_Q8_K>), Q8_K);
    TRAIT(IQ2_S, nullptr, (vec_dot_ref<GGML_TYPE_IQ2_S, GGML_TYPE_Q8_K>), Q8_K);
    TRAIT(IQ1_S, nullptr, (vec_dot_ref<GGML_TYPE_IQ1_S, GGML_TYPE_Q8_K>), Q8_K);
    TRAIT(IQ1_M, nullptr, (vec_dot_ref<GGML_TYPE_IQ1_M, GGML_TYPE_Q8_K>), Q8_K);
    TRAIT(IQ4_NL, (from_float_ref<GGML_TYPE_IQ4_NL>), (vec_dot_ref<GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_0>), Q8_0);
    TRAIT(IQ4_XS, (from_float_ref<GGML_TYPE_IQ4_XS>), (vec_dot_ref<GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_K>), Q8_K);
    // Canonical Q8_K metadata has only from_float; all other fields are zero.
    table[GGML_TYPE_Q8_K] = { from_float_q8_k, nullptr, static_cast<ggml_type>(0), 0 };
    TRAIT(BF16, (from_float_ref<GGML_TYPE_BF16>), (vec_dot_ref<GGML_TYPE_BF16, GGML_TYPE_BF16>), BF16);
    TRAIT(TQ1_0, (from_float_ref<GGML_TYPE_TQ1_0>), (vec_dot_ref<GGML_TYPE_TQ1_0, GGML_TYPE_Q8_K>), Q8_K);
    TRAIT(TQ2_0, (from_float_ref<GGML_TYPE_TQ2_0>), (vec_dot_ref<GGML_TYPE_TQ2_0, GGML_TYPE_Q8_K>), Q8_K);
#undef TRAIT
    return table;
}

const auto baseline = make_baseline();

} // namespace

const ggml_type_traits_cpu * ggml_sycl_get_baseline_type_traits_cpu(enum ggml_type type) noexcept {
    const int index = static_cast<int>(type);
    return index >= 0 && index < GGML_TYPE_COUNT ? &baseline[static_cast<size_t>(index)] : nullptr;
}

const ggml_type_traits_cpu * ggml_sycl_get_type_traits_cpu(enum ggml_type type) noexcept {
    return ggml_sycl_get_baseline_type_traits_cpu(type);
}
