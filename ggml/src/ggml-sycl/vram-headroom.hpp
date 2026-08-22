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
// vram_external_headroom_bytes() raises the DEFAULT floor when the batched
// pipeline is planned. Percentages (not a flat MB bump) keep the reservation
// proportional on a larger card (the B70 has roughly double the B50's VRAM)
// without a card-name/device-id check (owner ruling, llama.cpp-dp5i: no
// card-name tables).
//
// Deliberately header-only and dependency-free (no SYCL, no unified-cache.hpp)
// so it can be pure-logic unit-tested without a device, mirroring the
// moe-route-table.hpp precedent.

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdlib>

namespace ggml_sycl {

// Parses GGML_SYCL_VRAM_ARENA_EXTERNAL_HEADROOM_MB's raw string into a byte
// count, or returns 0 -- meaning "no real override; treat as unset" -- on ANY
// of: null, empty, a partial/garbage parse (trailing characters strtol did
// not consume, e.g. "1e9" stops at 'e'), a non-positive value, strtol's own
// ERANGE overflow, or a value so large its MB->bytes conversion would wrap
// size_t or plainly can't be a real device's VRAM. Callers must not try to
// distinguish "unset" from "rejected as invalid" -- both compose through the
// same fallback, never a smaller floor (llama.cpp-seno spec-review finding 1
// and finding 3).
//
// Upper-bound choice: rejects any parsed value >= 2^24 MB (16 TiB in bytes),
// an explicit sane cap rather than a total_vram-relative one (the parser has
// no total_vram in scope, and keeping it a standalone string->bytes function
// is simpler to reuse and test). 16 TiB is far beyond any real device's VRAM,
// so a legitimate override is never rejected, while this bound catches
// exactly the failure modes measured pre-fix: env="2147483648" (a
// bytes-instead-of-MB typo, meant as 2 GiB in bytes) produced a headroom of
// 2147483648 MB that skipped the arena cap entirely (device_total_vram never
// exceeds an absurd headroom that large); env="9999999999999999999" did the
// same via strtol's own ERANGE; env="17592186044416" (2^44) wrapped the
// *1024*1024 multiply to exactly 0, silently disabling the cap outright.
inline size_t vram_headroom_parse_positive_mb_to_bytes(const char * env_override) {
    if (env_override == nullptr || *env_override == '\0') {
        return 0;
    }
    errno             = 0;
    char *     endptr = nullptr;
    const long parsed = std::strtol(env_override, &endptr, 10);
    if (endptr == env_override || *endptr != '\0') {
        return 0;  // no digits consumed, or unconsumed trailing characters (also rejects "1e9")
    }
    if (errno == ERANGE || parsed <= 0) {
        return 0;
    }
    constexpr long k_max_sane_mb = 1L << 24;  // 16 TiB in bytes
    if (parsed >= k_max_sane_mb) {
        return 0;
    }
    return static_cast<size_t>(parsed) * 1024ull * 1024ull;
}

// total_vram_bytes:        the device's total (or budgeted) VRAM, in bytes.
// onednn_pipeline_planned: true when the oneDNN batched MoE PP pipeline is
//                          planned to run alongside other executors on this
//                          device (heavier driver-side working set).
inline size_t vram_external_headroom_bytes(size_t total_vram_bytes, bool onednn_pipeline_planned) {
    if (total_vram_bytes == 0) {
        return 0;
    }
    constexpr size_t k_mib            = 1024ull * 1024ull;
    constexpr size_t k_pipeline_floor = 1024ull * k_mib;  // 1 GB
    constexpr size_t k_plain_floor =
        630ull * k_mib;  // prior proven-adequate flat default, WITHOUT the batched pipeline
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
// just vram_external_headroom_bytes() in isolation.
//
// A POSITIVE, fully-valid override (see vram_headroom_parse_positive_mb_to_bytes
// for the full rejection list) wins verbatim, even when lower than either the
// default or the floor -- an operator is allowed to deliberately lower
// headroom, not just raise it. Anything the parser rejects -- unset, empty,
// "0", negative, unparsable, or out-of-range -- is treated as UNSET and falls
// through to max(default_headroom, floor): discarding default_headroom or
// hardcoding onednn_pipeline_planned=false on a rejected override would
// silently drop below both the pre-existing default and the pipeline floor,
// reintroducing the error-40 class this fix exists for (llama.cpp-seno
// spec-review finding 1).
//
// total_vram_bytes == 0 is asymmetric versus vram_external_headroom_bytes(0,
// ...), which returns 0 outright: here the pipeline-aware floor collapses to
// 0 at zero total, so the result is max(default_headroom, 0) ==
// default_headroom, unchanged. Unreachable in production today -- the sole
// caller (arena_reserve(), unified-cache.cpp) only reaches this behind its
// own `device_total_vram > 0` guard -- documented here so a future caller
// without that guard is not surprised (spec-review finding 5).
inline size_t vram_external_headroom_effective(size_t       total_vram_bytes,
                                               size_t       default_headroom,
                                               bool         onednn_pipeline_planned,
                                               const char * env_override) {
    const size_t parsed_override = vram_headroom_parse_positive_mb_to_bytes(env_override);
    if (parsed_override > 0) {
        return parsed_override;
    }
    return std::max(default_headroom, vram_external_headroom_bytes(total_vram_bytes, onednn_pipeline_planned));
}

}  // namespace ggml_sycl
