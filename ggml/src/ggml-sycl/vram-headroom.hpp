#pragma once
// Driver-reserved VRAM headroom outside the unified-cache arena.
// (perf-recovery epic, track D -- llama.cpp-seno; spec: llama.cpp-dp5i c-ni04/c-gkai)
//
// The arena's external headroom leaves room for driver/runtime working memory
// that never flows through the unified cache: Level Zero command lists and
// queues, oneDNN JIT/kernel residency, graph and event bookkeeping. A flat
// 630 MB default was proven insufficient when the oneDNN batched MoE PP
// pipeline coexists with the tiled executor: on the B50,
// GGML_SYCL_VRAM_ARENA_EXTERNAL_HEADROOM_MB=2048 eliminated a deterministic
// [MUL_MAT_ID-FAIL] error 40 (UR_RESULT_ERROR_OUT_OF_RESOURCES) that the
// 630 MB default could not (llama.cpp-dp5i c-gkai). The B70's default route
// hits the same error-40 class from a second, independent aggravator (the
// single-chunk arena's raw-vs-safe per-allocation cap window; see the sibling
// fix at unified-cache.cpp's arena_reserve()).
//
// This function raises the DEFAULT floor when the batched pipeline is
// planned, without touching an explicit env override -- the override always
// wins verbatim. Percentages (not a flat MB bump) keep the reservation
// proportional on a larger card (the B70 has roughly double the B50's VRAM)
// without a card-name/device-id check (owner ruling, llama.cpp-dp5i: no
// card-name tables).
//
// Deliberately header-only and dependency-free (no SYCL, no unified-cache.hpp)
// so it can be pure-logic unit-tested without a device, mirroring the
// moe-route-table.hpp precedent.

#include <algorithm>
#include <cstddef>
#include <cstdlib>

namespace ggml_sycl {

// total_vram_bytes:        the device's total (or budgeted) VRAM, in bytes.
// onednn_pipeline_planned: true when the oneDNN batched MoE PP pipeline is
//                          planned to run alongside other executors on this
//                          device (heavier driver-side working set).
// env_override:            GGML_SYCL_VRAM_ARENA_EXTERNAL_HEADROOM_MB's raw
//                          string (or nullptr). A positive parse wins
//                          verbatim over every default below.
inline size_t ggml_sycl_vram_external_headroom_bytes(size_t       total_vram_bytes,
                                                     bool         onednn_pipeline_planned,
                                                     const char * env_override) {
    if (env_override != nullptr) {
        const long parsed = std::strtol(env_override, nullptr, 10);
        if (parsed > 0) {
            return static_cast<size_t>(parsed) * 1024ull * 1024ull;
        }
    }
    if (total_vram_bytes == 0) {
        return 0;
    }
    constexpr size_t k_mib            = 1024ull * 1024ull;
    constexpr size_t k_pipeline_floor = 1024ull * k_mib;  // 1 GB
    constexpr size_t k_plain_floor    = 630ull * k_mib;   // prior proven-adequate flat default
    if (onednn_pipeline_planned) {
        return std::max(k_pipeline_floor, total_vram_bytes * 6 / 100);
    }
    return std::max(k_plain_floor, total_vram_bytes * 4 / 100);
}

// Composes the caller's pre-existing default headroom (arena_reserve()'s
// arena_default_external_headroom() result) with the pipeline-aware floor
// above, honoring an explicit env override. This is the wiring arena_reserve()
// actually applies -- extracted as its own pure function (rather than left
// inline at the call site) so the composition itself is unit-testable, not
// just ggml_sycl_vram_external_headroom_bytes() in isolation.
//
// A POSITIVE override wins verbatim, even when lower than either the default
// or the floor -- an operator is allowed to deliberately lower headroom, not
// just raise it. A zero/negative/unparsable override (or no override at all)
// is treated as UNSET and falls through to max(default_headroom, floor):
// discarding default_headroom or hardcoding onednn_pipeline_planned=false on
// an unparsable override would silently drop below both the pre-existing
// default and the pipeline floor on GGML_SYCL_VRAM_ARENA_EXTERNAL_HEADROOM_MB
// values like "0", "-5", "abc", or "" -- reintroducing the error-40 class
// this fix exists for (llama.cpp-seno spec-review finding 1).
inline size_t ggml_sycl_vram_external_headroom_effective(size_t       total_vram_bytes,
                                                         size_t       default_headroom,
                                                         bool         onednn_pipeline_planned,
                                                         const char * env_override) {
    if (env_override != nullptr) {
        const long parsed = std::strtol(env_override, nullptr, 10);
        if (parsed > 0) {
            return static_cast<size_t>(parsed) * 1024ull * 1024ull;
        }
    }
    return std::max(default_headroom, ggml_sycl_vram_external_headroom_bytes(total_vram_bytes, onednn_pipeline_planned,
                                                                             /*env_override=*/nullptr));
}

}  // namespace ggml_sycl
