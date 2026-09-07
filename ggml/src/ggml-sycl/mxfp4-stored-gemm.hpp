// Full-N, small-M (M <= 8) MXFP4 GEMM reading the STORED SOA expert weight
// layout directly (ggml_sycl_reordered::block_q_t<GGML_TYPE_MXFP4>,
// quants.hpp:190-211), with int8 x int8 DPAS accumulation.
// (llama.cpp-vtfs, option C step 2 -- plan Task G4)
//
// This is a standalone kernel family with its own exported entry point. It
// is deliberately NOT wired into any dispatch path -- ggml-sycl.cpp is
// unchanged by this task -- and exists to isolate "does the DPAS
// accumulation generalise from the decode-only, persistent-TG target-subset
// kernels (mmvq.cpp's mxfp4_pair_glu_soa_dpas_m4_sycl /
// mxfp4_dpas_down_single_col_sycl) to the FULL N of a real expert weight
// matrix, at M > 1" before large-M tiling (Task G5) and the XMX_TILED
// gate/up layout (Task G6).
//
// Math:
//   Y[m][n] = sum_k X[m][k] * W[n][k]     for m in [0, M), n in [0, n_out),
//                                              k in [0, n_k)
// where W is one expert's MXFP4 weight matrix, stored SOA on device exactly
// as quants.hpp documents: qs bytes for every row contiguous
// ([qs0..qsN]), then one E8M0 scale byte per (row, k-block)
// ([scale0..scaleN]), block_index = row * n_k_blocks + k_block. This is the
// SAME byte layout tests/mxfp4-stored-layout-oracle.hpp's
// build_soa_from_aos/decode_soa_mxfp4 build and decode, and the same formula
// mmvq.cpp's mxfp4_soa_load_a_vec (mmvq.cpp:7566) already reads on the
// decode path -- this file does not change that formula, only which DPAS
// operand the weight lands in (see the .cpp for why).
//
// Activation representation: q8_1-block-quantized (block size 32 =
// QK_MXFP4), bit-identical to ggml's own quantize_row_q8_1_ref -- see
// quantize_activations_q8_1 below. This is the SAME activation
// representation tests/mxfp4-stored-layout-oracle.hpp's
// preprocess_activation(..., ACT_Q8_1) reconstructs in double, so a real
// device kernel and the oracle are comparing the same quantity, not two
// different lossy paths.
//
// Epilogue: for each 32-element K block, the int32 DPAS partial product is
// scaled by activation_scale[m][k_block] * weight_scale[n][k_block] (E8M0,
// GGML_E8M0_TO_FP32_HALF / "halved" convention, matching
// mxfp4_soa_load_a_vec's mxfp4_e8m0_to_fp32_esimd and this backend's
// sycl_e8m0_to_fp32_half in common.hpp) and accumulated into a float32
// output cell -- both scales vary per K block, so the multiply happens
// INSIDE the K loop, exactly as mxfp4_dpas_down_single_col_sycl's epilogue
// does, not once after the full reduction.

#ifndef GGML_SYCL_MXFP4_STORED_GEMM_HPP
#define GGML_SYCL_MXFP4_STORED_GEMM_HPP

#include <cstdint>
#include <sycl/sycl.hpp>
#include <vector>

namespace ggml_sycl_mxfp4_stored_gemm {

// Host-side activation packer: quantizes `M x K` row-major f32 activations
// into ggml's own block_q8_1 32-element-block int8 format (bit-identical to
// quantize_row_q8_1_ref -- calls it directly, not a reimplementation), and
// unpacks the two output arrays the GEMM kernel below consumes. Kept
// host-side and reusing the exact ggml reference path rather than a device
// kernel: this task's scope is the GEMM kernel itself (see the file header
// and the plan's Task G4 description), and routing activation packing
// through the same function the CPU oracle's ACT_Q8_1 arm calls means a
// packing bug cannot silently diverge from what the oracle assumes about the
// activation representation.
//
// K must be a positive multiple of QK8_1 (32) -- GGML_ASSERT-checked.
struct q8_1_activation_pack {
    std::vector<int8_t> qs;      // [M][K], row-major int8 codes
    std::vector<float>  scales;  // [M][K / 32], per-block scale (block_q8_1.d, widened to f32)
};

q8_1_activation_pack quantize_activations_q8_1(const float * x, int64_t M, int64_t K);

// Runs the full-N, small-M SOA MXFP4 GEMM for ONE expert. The caller
// allocates and populates the four device buffers below
// (soa_weight_device, act_qs_device, act_scales_device, dst_device)
// through the ordinary ggml/SYCL backend allocator (which is the unified
// cache -- see docs/backend/sycl-memory-design.md) and passes device
// pointers in.
//
// This function itself MAY allocate (llama.cpp-kcya round 6): when the
// internal K-split factor resolves above 1, it allocates a device-side
// partial-sum scratch buffer through the SAME unified cache
// (ggml_sycl::unified_allocate, never a raw sycl::malloc_device -- see
// mxfp4-stored-gemm.cpp's mxfp4_stored_gemm_ksplit_get_or_alloc_scratch),
// owned end-to-end by a single mem_handle. That handle is cached
// thread-local and reused/grown across calls -- a growth retires the old
// handle against the combine kernel's completion event
// (retain_handles_until_event) rather than dropping it while a submission
// may still be reading it; the caller never sees or manages this scratch.
// If the allocator refuses (budget/allocator failure), the call
// transparently degrades to a single unsplit pass writing straight to
// dst_device instead -- correct but slower, never a hard failure.
//
// The returned event completes the WHOLE dispatch: waiting on it
// transitively waits for every kernel this call submitted, including an
// internal partial pass a caller never sees directly. Its OWN profiling
// timestamps, however, cover only the LAST kernel submitted (SYCL event
// profiling has no notion of "this event's dependency chain's total
// time") -- see mxfp4-stored-gemm.cpp's dispatcher comment and this
// kernel's own two profiler rows ("mxfp4.stored_gemm.soa.partial" /
// ".combine") for the per-launch bandwidth this file's own tests derive
// instead of trusting the returned event's timestamps directly.
//
// `soa_weight_device` and `act_qs_device` must each be at least 32-byte
// aligned (GGML_ASSERT-checked): the kernel's `block_load<uint8_t, 16>` /
// `block_load<int8_t, 32>` ESIMD reads off these two bases carry vector-
// alignment requirements that are satisfied only because every offset the
// kernel computes is itself a multiple of the 32-element block width -- a
// caller-supplied base that is not itself 32-byte aligned would silently
// misalign every load from it. `act_scales_device` carries no such
// requirement (only ever scalar-subscripted, never block_loaded) and is
// deliberately not asserted (llama.cpp-6f73 c-py5n should-fix 2, correcting
// an over-broad round-1 claim).
//
//   soa_weight_device: device pointer to the expert's SOA MXFP4 weight
//                       buffer (quants.hpp layout), n_out rows x n_k cols.
//   act_qs_device:     device pointer to `M x n_k` row-major int8 activation
//                       codes (quantize_activations_q8_1's `qs`, uploaded).
//   act_scales_device: device pointer to `M x (n_k / 32)` row-major f32
//                       per-block activation scales (quantize_activations_q8_1's
//                       `scales`, uploaded).
//   dst_device:        device pointer to `M x n_out` row-major f32 output.
//   M:                 number of activation rows. Must be one of
//                       {1, 2, 4, 8} -- the DPAS repeat-count values this
//                       initial generalisation instantiates and Task G4's
//                       numerics test exercises; other values GGML_ABORT.
//                       Task G5 covers M in {32, 128, 512} with a different,
//                       work-group-cooperative kernel, not this one.
//   n_out:             number of weight rows (output columns of Y). Any
//                       positive value -- need not be a multiple of the
//                       kernel's internal 16-row N-tile; a partial tile is
//                       boundary-checked, mirroring mxfp4_soa_load_a_vec's
//                       `row < nrows_per_expert` guard.
//   n_k:                reduction dimension. Must be a positive multiple of
//                       32 (QK_MXFP4).
sycl::event ggml_sycl_mxfp4_soa_gemm_dpas(sycl::queue &                    queue,
                                          const void *                     soa_weight_device,
                                          const int8_t *                   act_qs_device,
                                          const float *                    act_scales_device,
                                          float *                          dst_device,
                                          int                              M,
                                          int                              n_out,
                                          int                              n_k,
                                          const std::vector<sycl::event> & deps = {});

}  // namespace ggml_sycl_mxfp4_stored_gemm

#endif  // GGML_SYCL_MXFP4_STORED_GEMM_HPP
