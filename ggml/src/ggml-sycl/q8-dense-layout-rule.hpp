//
// Q8_0 dense-weight coalesced/SOA tile-alignment predicate (llama.cpp-pktr).
//
// Pure C++ -- NO SYCL, NO ggml includes -- so a host-only unit test
// (tests/test-sycl-q8-dense-layout-rule.cpp) can check the predicate without
// a device or a SYCL toolchain, the same pattern q8-scale-plane.hpp
// established for the nz1k scale-plane index mapping.
//
// The coalesced Q8_0 MMVQ decode kernel pads the last tile of a row up to a
// full MMVQ_COALESCED_TILE_BLOCKS-block tile (ggml-sycl/common.hpp,
// ggml_sycl_coalesced_fixed_tile_count) -- it always processes whole tiles,
// never a partial one. A non-tile-aligned row pays for that padding on every
// token: measured on gemma4-E4B's K=2560 dense attention/FFN weights
// (80 blocks/row = 2 full tiles + a 16-block padded tail), SOA beats
// COALESCED by up to 14 points of B50/B70 achieved bandwidth, while
// tile-aligned dense shapes (Mistral K=4096/14336, gemma4 down-proj K=10240,
// gemma4 q/k/v K=2048) keep COALESCED ahead by 2-9 points (llama.cpp-pktr
// ticket description). ggml_sycl_adjust_layout_for_tensor (ggml-sycl.cpp) and
// planner_default_device_layout (unified-cache.cpp) both call the predicate
// below to demote a COALESCED resolution to SOA for a non-tile-aligned row --
// this header is their single definition of the arithmetic.
//
// The two constants below are LOCAL COPIES of ggml-sycl/common.hpp's QK8_0
// (from ggml-common.h) and MMVQ_COALESCED_TILE_BLOCKS (== GGML_SYCL_WARP_SIZE,
// ggml-sycl/presets.hpp), kept local so this header stays free of ggml/SYCL
// includes. common.hpp static_asserts they still match the canonical values
// it uses, so this file cannot silently drift from what the kernel dispatch
// code actually does.
//
#pragma once

#include <cstdint>

// Q8_0 block width; mirrors ggml-common.h's QK8_0.
constexpr int64_t GGML_SYCL_Q8_DENSE_LAYOUT_RULE_QK = 32;

// Blocks per coalesced warp tile; mirrors ggml-sycl/common.hpp's
// MMVQ_COALESCED_TILE_BLOCKS (== GGML_SYCL_WARP_SIZE).
constexpr int64_t GGML_SYCL_Q8_DENSE_LAYOUT_RULE_TILE_BLOCKS = 32;

// True iff a Q8_0 row of width ne00 is a whole number of coalesced tiles --
// i.e. the coalesced MMVQ decode kernel pays no padding for this row.
constexpr bool ggml_sycl_q8_0_coalesced_tile_aligned(int64_t ne00) {
    return ne00 > 0 && (ne00 % GGML_SYCL_Q8_DENSE_LAYOUT_RULE_QK) == 0 &&
           ((ne00 / GGML_SYCL_Q8_DENSE_LAYOUT_RULE_QK) % GGML_SYCL_Q8_DENSE_LAYOUT_RULE_TILE_BLOCKS) == 0;
}
