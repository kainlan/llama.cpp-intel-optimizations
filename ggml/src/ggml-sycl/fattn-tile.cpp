#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/work_group_static.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "fattn-common.hpp"
#include "fattn-tile.hpp"
#include <cmath>
#include <float.h>
namespace syclex = sycl::ext::oneapi::experimental;

bool ggml_sycl_fattn_tile_d512_use_gqa_opt(const fattn_params & params) {
    return params.mask != nullptr && params.max_bias == 0.0f && (params.ne11 % FATTN_KQ_STRIDE == 0);
}

void ggml_sycl_flash_attn_ext_tile(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("legacy SYCL flash-attention tile wrapper is not wired; use ggml_sycl_flash_attn_ext");
}
