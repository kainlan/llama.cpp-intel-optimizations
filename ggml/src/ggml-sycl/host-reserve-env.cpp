//
// GGML_SYCL_HOST_RESERVE_MB parsing implementation (llama.cpp-16el).
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "host-reserve-env.hpp"

#include <cerrno>
#include <cstdlib>
#include <cstring>

// This TU deliberately depends on nothing beyond the C library: that is what
// lets the host-only test target link it without libggml-sycl.

namespace ggml_sycl {
namespace detail {

// ASCII-only on purpose: std::isspace/std::isdigit are locale-dependent and
// take an int that must not be a negative char.
static bool reserve_is_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r';
}

static bool reserve_is_digit(char c) {
    return c >= '0' && c <= '9';
}

static host_reserve_parse_result reserve_rejected(const char * reason) {
    host_reserve_parse_result result;
    result.status = host_reserve_parse_status::REJECTED;
    result.reason = reason;
    return result;
}

host_reserve_parse_result parse_host_reserve_mb(const char * raw) {
    host_reserve_parse_result result;
    if (!raw || raw[0] == '\0') {
        return result;  // UNSET: the auto calculation runs
    }

    // Trim surrounding whitespace ourselves rather than relying on strtoull's
    // leading skip: the trailing side has to be checked explicitly anyway, and
    // doing both here keeps the digit scan below a plain [begin, end) walk.
    const char * begin = raw;
    const char * end   = raw + std::strlen(raw);
    while (begin < end && reserve_is_space(*begin)) {
        ++begin;
    }
    while (end > begin && reserve_is_space(end[-1])) {
        --end;
    }
    if (begin == end) {
        return reserve_rejected("no digits (whitespace only)");
    }

    // Digits only. A sign is rejected before the digit scan so `-1` names its
    // real problem instead of reading as "junk"; `+4096` is refused too -- the
    // documented syntax is an unsigned decimal integer, nothing else.
    if (*begin == '-') {
        return reserve_rejected("negative values are not allowed");
    }
    if (*begin == '+') {
        return reserve_rejected("sign characters are not allowed");
    }
    for (const char * p = begin; p < end; ++p) {
        if (!reserve_is_digit(*p)) {
            return reserve_rejected(
                "trailing or embedded non-digit characters (expected an unsigned decimal MiB count)");
        }
    }

    // The digit run is validated, so strtoull can only stop at `end` or on
    // ERANGE; a ULLONG_MAX result with errno set is the overflow signal the
    // lenient helper used to ignore.
    errno                          = 0;
    char *                   stop  = nullptr;
    const unsigned long long value = std::strtoull(begin, &stop, 10);
    if (errno == ERANGE || stop != end) {
        return reserve_rejected("out of range (does not fit unsigned long long)");
    }
    if (value > static_cast<unsigned long long>(k_host_reserve_mb_max)) {
        return reserve_rejected("MiB count too large: the byte conversion would overflow size_t");
    }

    result.mb = static_cast<size_t>(value);
    if (result.mb == 0) {
        result.status = host_reserve_parse_status::AUTO;  // documented: 0 leaves the auto calculation enabled
        return result;
    }
    result.status = host_reserve_parse_status::OVERRIDE;
    result.bytes  = result.mb * (1024u * 1024u);  // cannot wrap: mb <= k_host_reserve_mb_max
    return result;
}

}  // namespace detail
}  // namespace ggml_sycl
