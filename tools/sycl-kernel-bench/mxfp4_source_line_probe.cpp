#include "data_generators.hpp"
#include "ggml.h"
#include "kernels/reference/reference_kernels.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sycl/sycl.hpp>

namespace {

struct params {
    std::string kernel     = "mxfp4_pair_glu_xmx_tiled_packed_r8_m2_sparse32_bias";
    int64_t     dim_m      = 2880;
    int64_t     dim_n      = 4;
    int64_t     dim_k      = 2880;
    int         warmup     = 10;
    int         iterations = 100;
    bool        validate   = true;
    bool        json       = true;
};

bool parse_i64_arg(const char * arg, const char * prefix, int64_t & out) {
    const size_t len = std::strlen(prefix);
    if (std::strncmp(arg, prefix, len) != 0) {
        return false;
    }
    char *          end   = nullptr;
    const long long value = std::strtoll(arg + len, &end, 10);
    if (end == arg + len || *end != '\0') {
        return false;
    }
    out = static_cast<int64_t>(value);
    return true;
}

bool parse_int_arg(const char * arg, const char * prefix, int & out) {
    int64_t parsed = 0;
    if (!parse_i64_arg(arg, prefix, parsed)) {
        return false;
    }
    out = static_cast<int>(parsed);
    return true;
}

bool parse_args(int argc, char ** argv, params & p) {
    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];
        if (std::strncmp(arg, "--kernel=", 9) == 0) {
            p.kernel = arg + 9;
        } else if (std::strcmp(arg, "--quant=MXFP4") == 0) {
            continue;
        } else if (parse_i64_arg(arg, "--dim_m=", p.dim_m) || parse_i64_arg(arg, "--dim_n=", p.dim_n) ||
                   parse_i64_arg(arg, "--dim_k=", p.dim_k) || parse_int_arg(arg, "--warmup=", p.warmup) ||
                   parse_int_arg(arg, "--iterations=", p.iterations)) {
            continue;
        } else if (std::strcmp(arg, "--validate") == 0) {
            p.validate = true;
        } else if (std::strcmp(arg, "--no-validate") == 0) {
            p.validate = false;
        } else if (std::strcmp(arg, "--output=json") == 0) {
            p.json = true;
        } else if (std::strcmp(arg, "--output=csv") == 0) {
            p.json = false;
        } else if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            std::fprintf(stderr,
                         "Usage: %s [--kernel=mxfp4_pair_glu_xmx_tiled_packed_r8_m2_sparse32_bias] "
                         "[--dim_m=N --dim_n=N --dim_k=N] [--warmup=N --iterations=N] [--validate] "
                         "[--output=json]\n",
                         argv[0]);
            return false;
        } else {
            std::fprintf(stderr, "unsupported argument for MXFP4 source-line probe: %s\n", arg);
            return false;
        }
    }
    return true;
}

int parse_rows_per_wg(const std::string & kernel_name) {
    if (kernel_name.find("_r16") != std::string::npos) {
        return 16;
    }
    if (kernel_name.find("_r8") != std::string::npos) {
        return 8;
    }
    if (kernel_name.find("_r4") != std::string::npos) {
        return 4;
    }
    if (kernel_name.find("_r2") != std::string::npos) {
        return 2;
    }
    if (kernel_name.find("_r1") != std::string::npos) {
        return 1;
    }
    return 4;
}

int64_t parse_token_rows(const std::string & kernel_name) {
    for (size_t pos = 0; (pos = kernel_name.find("_t", pos)) != std::string::npos; pos += 2) {
        const size_t digit_pos = pos + 2;
        if (digit_pos >= kernel_name.size() || kernel_name[digit_pos] < '0' || kernel_name[digit_pos] > '9') {
            continue;
        }
        char *       end   = nullptr;
        const char * start = kernel_name.c_str() + digit_pos;
        const long   value = std::strtol(start, &end, 10);
        if (end != start && value > 0) {
            return value;
        }
    }
    return 1;
}

int parse_tiles_n(const std::string & kernel_name) {
    const size_t pos = kernel_name.find("_tn");
    if (pos == std::string::npos) {
        return 1;
    }
    char *       end   = nullptr;
    const char * start = kernel_name.c_str() + pos + 3;
    const long   value = std::strtol(start, &end, 10);
    return end != start && value > 0 ? static_cast<int>(value) : 1;
}

int parse_subgroup_size(const std::string & kernel_name) {
    const size_t pos = kernel_name.find("_sg");
    if (pos == std::string::npos) {
        return 32;
    }
    char *       end   = nullptr;
    const char * start = kernel_name.c_str() + pos + 3;
    const long   value = std::strtol(start, &end, 10);
    return end != start && value > 0 ? static_cast<int>(value) : 32;
}

int parse_scale_stride_blocks(const std::string & kernel_name) {
    if (kernel_name.find("_scale128") != std::string::npos) {
        return 128;
    }
    if (kernel_name.find("_scale96") != std::string::npos) {
        return 96;
    }
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    params p;
    if (!parse_args(argc, argv, p)) {
        return 2;
    }

    sycl::queue queue{ sycl::gpu_selector_v, sycl::property::queue::in_order{} };

    sycl_bench::GeneratedWeights weights;
    if (!sycl_bench::generate_quantized_weights(GGML_TYPE_MXFP4, GGML_LAYOUT_SOA, p.dim_m, p.dim_k, false, weights)) {
        std::fprintf(stderr, "failed to generate MXFP4 weights\n");
        return 1;
    }

    const int64_t                    token_rows = parse_token_rows(p.kernel);
    sycl_bench::GeneratedActivations activations =
        sycl_bench::generate_activations(token_rows, p.dim_k, p.dim_k, false, false, true);

    const int  rows_per_wg = parse_rows_per_wg(p.kernel);
    const bool cache_y = p.kernel.find("_cache") != std::string::npos && p.kernel.find("_nocache") == std::string::npos;
    const bool direct_xmx                    = p.kernel.find("_xmx_soa") != std::string::npos;
    const bool xmx_tiled                     = p.kernel.find("_xmx_tiled") != std::string::npos;
    const bool xmx_tiled_grouped             = p.kernel.find("_grouped") != std::string::npos;
    const bool xmx_tiled_pack_q8             = p.kernel.find("_packed") != std::string::npos;
    const bool xmx_tiled_prefetch            = p.kernel.find("_prefetch") != std::string::npos;
    const int  xmx_tiled_m_tiles             = p.kernel.find("_m4") != std::string::npos ? 4 :
                                               p.kernel.find("_m2") != std::string::npos ? 2 :
                                                                                           1;
    const bool xmx_tiled_v2                  = p.kernel.find("_xmx_tiled_v2") != std::string::npos;
    const int  xmx_tiled_v2_group_bytes      = 320;
    const bool xmx_tiled_bundle4             = p.kernel.find("_xmx_tiled_bundle4") != std::string::npos;
    const int  xmx_tiled_bundle4_group_bytes = xmx_tiled_bundle4 ? 1088 : 0;
    const bool split_gate_up                 = p.kernel.find("_split") != std::string::npos;
    const bool single_column_gateup          = p.kernel.find("_singlecol") != std::string::npos;
    const bool multi_rhs_gateup              = p.kernel.find("_multirhs") != std::string::npos;
    const int  multi_rhs_cols                = p.kernel.find("_multirhs_n4") != std::string::npos ? 4 :
                                               p.kernel.find("_multirhs_n2") != std::string::npos ? 2 :
                                                                                                    1;
    const bool predecoded_i8                 = p.kernel.find("_predecoded") != std::string::npos;
    const int  xmx_tiles_n                   = parse_tiles_n(p.kernel);
    const bool vector_qs_load                = p.kernel.find("_vecq") != std::string::npos;
    const bool ignore_weight_scale           = p.kernel.find("_noscale") != std::string::npos;
    const int  scale_stride_blocks           = parse_scale_stride_blocks(p.kernel);
    const int  subgroup_size                 = parse_subgroup_size(p.kernel);
    const bool sparse_expert_slots           = p.kernel.find("_sparse32") != std::string::npos;
    const bool use_bias                      = p.kernel.find("_bias") != std::string::npos;

    sycl_bench::ReferenceMetrics metrics;
    std::string                  error;
    const bool                   ok = sycl_bench::run_mxfp4_pair_glu(
        weights, activations, p.dim_m, p.dim_n, p.dim_k, token_rows, rows_per_wg, cache_y, direct_xmx, xmx_tiled,
        xmx_tiled_grouped, xmx_tiled_pack_q8, xmx_tiled_prefetch, xmx_tiled_m_tiles, xmx_tiled_v2,
        xmx_tiled_v2_group_bytes, xmx_tiled_bundle4, xmx_tiled_bundle4_group_bytes, split_gate_up, single_column_gateup,
        multi_rhs_gateup, multi_rhs_cols, predecoded_i8, xmx_tiles_n, vector_qs_load, ignore_weight_scale,
        scale_stride_blocks, subgroup_size, sparse_expert_slots, use_bias, p.validate, p.warmup, p.iterations, queue,
        metrics, error);
    queue.wait_and_throw();

    if (p.json) {
        std::printf(
            "{\"kernel\":\"%s\",\"ok\":%s,\"latency_us\":%.6f,\"throughput_tps\":%.6f,\"throughput_tops\":%.6f,"
            "\"bandwidth_gbps\":%.6f,\"error\":\"%s\"}\n",
            p.kernel.c_str(), ok ? "true" : "false", metrics.total_us,
            metrics.total_us > 0.0 ? 1.0e6 / metrics.total_us : 0.0, metrics.tops, metrics.bandwidth_gbps,
            error.c_str());
    } else {
        std::printf("kernel,ok,latency_us,throughput_tps,throughput_tops,bandwidth_gbps,error\n");
        std::printf("%s,%d,%.6f,%.6f,%.6f,%.6f,%s\n", p.kernel.c_str(), ok ? 1 : 0, metrics.total_us,
                    metrics.total_us > 0.0 ? 1.0e6 / metrics.total_us : 0.0, metrics.tops, metrics.bandwidth_gbps,
                    error.c_str());
    }

    if (!ok) {
        std::fprintf(stderr, "MXFP4 source-line probe failed: %s\n", error.c_str());
        return 1;
    }
    return 0;
}
