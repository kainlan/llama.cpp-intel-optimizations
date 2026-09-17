//
// GGML_SYCL_HOST_RESERVE_MB parsing (llama.cpp-16el).
//
// This variable became live with llama.cpp-glkg as the byte-count override of
// the pinned-pool BUDGET, and a budget deserves a stricter parse than the
// tuning knobs that share the lenient strtol helper in unified-cache.cpp:
// `6144junk` must not become 6144, ERANGE must not be ignored, and a MiB count
// whose byte conversion wraps size_t must not become a tiny (or zero) budget.
//
// The parser lives in its own SYCL-header-free TU so a host-only test can
// compile it directly without libggml-sycl, a device, or -fsycl -- the same
// arrangement as zone-sizing.cpp / tests/test-zone-sizing.cpp. It is pure: no
// getenv, no logging. The production call site (the host-arena constructor in
// unified-cache.cpp) reads the variable, calls this, and emits the one WARN on
// rejection; that keeps the decision testable and the diagnostic in the log.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace ggml_sycl {
namespace detail {

enum class host_reserve_parse_status : uint8_t {
    UNSET    = 0,  // variable absent or empty: the auto calculation runs
    AUTO     = 1,  // "0": explicitly leaves the auto calculation enabled (documented, kept)
    OVERRIDE = 2,  // a positive MiB count; `bytes` is the pinned-pool budget
    REJECTED = 3,  // malformed or out of range; the caller warns and runs the auto calculation as if unset
};

struct host_reserve_parse_result {
    host_reserve_parse_status status = host_reserve_parse_status::UNSET;
    size_t                    mb     = 0;        // parsed MiB count (OVERRIDE / AUTO)
    size_t                    bytes  = 0;        // mb * 2^20 (OVERRIDE only)
    const char *              reason = nullptr;  // static string, non-null exactly when REJECTED
};

// Largest MiB count whose byte conversion still fits size_t. Anything above it
// is rejected rather than wrapped (17592186044415 on a 64-bit host).
constexpr size_t k_host_reserve_mb_max = std::numeric_limits<size_t>::max() / (1024u * 1024u);

// Accepted syntax: an unsigned decimal integer, optionally surrounded by
// whitespace. Everything else -- a sign character, trailing or embedded
// non-digits, hex, a whitespace-only string, a value that does not fit
// unsigned long long, or one above k_host_reserve_mb_max -- is REJECTED with a
// reason. `raw` may be null (variable unset).
host_reserve_parse_result parse_host_reserve_mb(const char * raw);

}  // namespace detail
}  // namespace ggml_sycl
