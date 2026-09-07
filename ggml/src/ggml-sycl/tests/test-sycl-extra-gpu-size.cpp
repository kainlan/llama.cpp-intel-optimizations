// Host-only, no GPU required: right-sizes ggml_tensor_extra_gpu (llama.cpp-h9uv).
//
// RED on master (measured via a standalone `icpx -fsycl` compile of
// common.hpp + a sizeof probe, 2026-09-07, no build/ needed):
//   sizeof(ggml_tensor_extra_gpu) = 277,712 B
// GGML_SYCL_MAX_DEVICES=48 (ggml-sycl.h) sizes 33 device-indexed arrays on
// this one struct. A Mistral pp1024 decode allocates/releases 838 such
// extras per graph rebuild -- 838 x 277,712 B = ~222 MB of host RSS churned
// through glibc's brk heap every rebuild, fragmenting it (llama.cpp-dfo0/L1,
// llama.cpp-asdt c-871e). The single biggest contributor is
// moe_grouped_scratch[48] (5 mem_handle + 5 size_t per device, and
// sizeof(ggml_sycl::mem_handle) is 488 B, not a few words) at 119,040 B --
// 43% of the whole struct -- behind exactly one external call site.
//
// GREEN: the events/XMX/MoE cluster -- confirmed by grep audit (see
// llama.cpp-h9uv ticket comment c-9rhk) to be touched ONLY by weight/MoE
// tensor code paths, never by an activation or KV-view extra -- moves into a
// separate, lazily-allocated ggml_tensor_extra_gpu_weight_ext. An activation
// extra's weight_ext stays null for its entire life, so it only ever pays
// for the core struct.
//
// This test intentionally avoids constructing a ggml_tensor_extra_gpu (that
// needs mem-handle.cpp/unified-cache.cpp linked in, like the sibling
// test-sycl-runtime-alloc/test-mem-handle-* targets do, for a much heavier
// build); every check below is a compile-time static_assert over common.hpp
// alone, so this binary links against nothing but the SYCL runtime and
// builds in seconds. Real construction/destruction of these extras is
// already exercised by test-sycl-compute-buffer-extra-reuse and
// test-sycl-kv-view-extra-reuse (GPU-gated).
#include "common.hpp"

#include <cstdio>
#include <type_traits>

// The activation-only budget. data_handle[GGML_SYCL_MAX_DEVICES] alone (a
// ggml_sycl::mem_handle per device, 488 B each) is 23,424 B and cannot move
// out of the core struct: it is the storage handle for EVERY tensor, weight
// or activation, so it is not part of this ticket's fix. Shrinking it
// further would mean sizing ggml_tensor_extra_gpu's per-device tables by
// runtime device count instead of GGML_SYCL_MAX_DEVICES=48 -- a much larger,
// riskier change touching hundreds of raw `[]` call sites across the
// 60k-line ggml-sycl.cpp, flagged as a possible follow-up rather than done
// here. 32 KB comfortably covers the achieved ~25 KB core struct with
// headroom for the small scalar/TP fields that stay on it.
#define GGML_TENSOR_EXTRA_GPU_ACTIVATION_BUDGET_BYTES (32 * 1024)

static_assert(sizeof(ggml_tensor_extra_gpu) <= GGML_TENSOR_EXTRA_GPU_ACTIVATION_BUDGET_BYTES,
              "ggml_tensor_extra_gpu grew past its activation-only budget -- "
              "see llama.cpp-h9uv for the per-device array audit that explains "
              "why events/xmx_*/moe_* belong on ggml_tensor_extra_gpu_weight_ext, "
              "not on the core struct");

// This is the actual gate: a >= 10x reduction from the measured RED baseline
// (see the header comment). Written against the literal RED number rather
// than a re-derived one so a future change to GGML_SYCL_MAX_DEVICES or
// mem_handle's size does not silently move this goalpost.
static_assert(sizeof(ggml_tensor_extra_gpu) * 10 <= 277712,
              "ggml_tensor_extra_gpu no longer clears the >= 10x reduction "
              "llama.cpp-h9uv's acceptance criterion (the per-reset MB probe, "
              "GGML_SYCL_EXTRA_LEAK_PROBE=1) requires");

// The weight-only cluster must have moved, not been deleted: the extension
// struct should still carry (most of) the original footprint.
static_assert(sizeof(ggml_tensor_extra_gpu_weight_ext) >= 200 * 1024,
              "ggml_tensor_extra_gpu_weight_ext lost its MoE/XMX footprint -- "
              "the weight-only cluster must be relocated, not deleted");

// weight_ext must be a small, lazily-allocated handle (e.g. std::unique_ptr),
// never inline storage -- otherwise every activation extra pays for the
// extension again and the split accomplishes nothing.
static_assert(sizeof(decltype(ggml_tensor_extra_gpu::weight_ext)) <= 16,
              "ggml_tensor_extra_gpu::weight_ext must stay a small pointer-like "
              "handle, not inline storage");

// L2 (llama.cpp-dfo0) and L2b (llama.cpp-asdt) fields gate graph-reuse-vs-
// rebuild detection and the KV-view leak counter on the hot activation/view
// path; they must stay on the CORE struct (never behind weight_ext) with
// their original types, surviving this refactor untouched.
static_assert(std::is_same<decltype(ggml_tensor_extra_gpu::alloc_generation), uint64_t>::value,
              "alloc_generation (L2) must stay a uint64_t on the core struct");
static_assert(std::is_same<decltype(ggml_tensor_extra_gpu::debug_is_kv_view_extra), bool>::value,
              "debug_is_kv_view_extra (L2b) must stay a bool on the core struct");

int main() {
    printf("[test-sycl-extra-gpu-size] sizeof(ggml_tensor_extra_gpu) = %zu bytes (budget %d, RED baseline 277712)\n",
           sizeof(ggml_tensor_extra_gpu), GGML_TENSOR_EXTRA_GPU_ACTIVATION_BUDGET_BYTES);
    printf("[test-sycl-extra-gpu-size] sizeof(ggml_tensor_extra_gpu_weight_ext) = %zu bytes\n",
           sizeof(ggml_tensor_extra_gpu_weight_ext));
    printf("[test-sycl-extra-gpu-size] sizeof(ggml_tensor_extra_gpu::weight_ext) = %zu bytes\n",
           sizeof(decltype(ggml_tensor_extra_gpu::weight_ext)));
    printf("[test-sycl-extra-gpu-size] PASS\n");
    return 0;
}
