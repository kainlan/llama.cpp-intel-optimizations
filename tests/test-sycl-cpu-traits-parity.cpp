#include "cpu-traits-support.hpp"

#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static const ggml_type types[] = {
    GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q1_0, GGML_TYPE_Q2_0, GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_1,
    GGML_TYPE_MXFP4, GGML_TYPE_NVFP4, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_XXS,
    GGML_TYPE_IQ3_S, GGML_TYPE_IQ2_S, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M, GGML_TYPE_IQ4_NL,
    GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_K, GGML_TYPE_BF16, GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0,
};

int main() {
    ggml_cpu_init();
    constexpr int n = 256;
    std::vector<float> input(n);
    for (int i = 0; i < n; ++i) {
        input[i] = std::sin(float(i) * 0.17f) * 3.0f;
    }

    for (ggml_type type : types) {
        const auto * base = ggml_sycl_get_baseline_type_traits_cpu(type);
        const auto * cpu  = ggml_get_type_traits_cpu(type);
        if (!base || !cpu || base->vec_dot_type != cpu->vec_dot_type || base->nrows != 1 ||
            bool(base->from_float) != bool(cpu->from_float) || bool(base->vec_dot) != bool(cpu->vec_dot)) {
            std::fprintf(stderr, "field mismatch for %s\n", ggml_type_name(type));
            return 1;
        }
        if (!base->from_float) {
            continue;
        }
        std::vector<unsigned char> qb(ggml_row_size(type, n));
        std::vector<unsigned char> qc(qb.size());
        base->from_float(input.data(), qb.data(), n);
        cpu->from_float(input.data(), qc.data(), n);
        if (std::memcmp(qb.data(), qc.data(), qb.size()) != 0) {
            std::fprintf(stderr, "quantizer mismatch for %s\n", ggml_type_name(type));
            return 1;
        }
        if (!base->vec_dot) {
            continue;
        }
        const ggml_type vdt = base->vec_dot_type;
        const auto * base_vdt = ggml_sycl_get_baseline_type_traits_cpu(vdt);
        if (!base_vdt || !base_vdt->from_float) {
            std::fprintf(stderr, "missing vec-dot quantizer for %s\n", ggml_type_name(type));
            return 1;
        }
        std::vector<unsigned char> qv(ggml_row_size(vdt, n));
        base_vdt->from_float(input.data(), qv.data(), n);
        float sb = 0.0f, sc = 0.0f;
        base->vec_dot(n, &sb, 0, qb.data(), 0, qv.data(), 0, 1);
        cpu->vec_dot(n, &sc, 0, qb.data(), 0, qv.data(), 0, 1);
        const float tolerance = 2e-5f * std::max(1.0f, std::fabs(sc));
        if (!std::isfinite(sb) || std::fabs(sb - sc) > tolerance) {
            std::fprintf(stderr, "vec_dot mismatch for %s: %.9g vs %.9g\n", ggml_type_name(type), sb, sc);
            return 1;
        }
    }

    if (ggml_sycl_get_baseline_type_traits_cpu(static_cast<ggml_type>(-1)) != nullptr ||
        ggml_sycl_get_baseline_type_traits_cpu(static_cast<ggml_type>(GGML_TYPE_COUNT)) != nullptr) {
        std::fprintf(stderr, "bounds check failed\n");
        return 1;
    }
    return 0;
}
