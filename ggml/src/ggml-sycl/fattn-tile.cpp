#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/work_group_static.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "fattn-common.hpp"
#include "fattn-tile.hpp"
#include <cmath>
#include <float.h>
namespace syclex = sycl::ext::oneapi::experimental;

// DV==512 forces launch_fattn_tile_switch_ncols2's own
// `gqa_ratio<=4 && DV<=256 ? 16 : INT_MAX` ternary to INT_MAX unconditionally
// (the DV<=256 half is always false here) -- so unlike the D<=256 kernels,
// ne01 never gates GQA-grouping eligibility at D=512.
// A mask with a head axis (ne32 > 1) is not shared by the Q heads of one K/V
// head, and flash_attn_tile reads one mask plane per block, so such a mask
// runs ungrouped (ncols2 == 1), as in ggml-cuda/fattn-tile.cuh (llama.cpp-44d0).
bool ggml_sycl_fattn_tile_d512_use_gqa_opt(const fattn_params & params) {
    return params.mask != nullptr && params.ne32 == 1 && params.max_bias == 0.0f &&
           (params.ne11 % FATTN_KQ_STRIDE == 0);
}

void ggml_sycl_flash_attn_ext_tile(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("legacy SYCL flash-attention tile wrapper is not wired; use ggml_sycl_flash_attn_ext");
}
