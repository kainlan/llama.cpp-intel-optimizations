//
// Element width for the kernel copy that stands in for a memcpy node while a SYCL graph is recording.
//
// Pure C++ (no SYCL header) so a host-only test can pin the arithmetic.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cstddef>
#include <cstdint>

// Widest element, in bytes (4, 2 or 1), that a single copy kernel can use over
// [src, src + nbytes) -> [dst, dst + nbytes) without a misaligned access. It is the largest power of two
// (capped at 4) dividing dst, src and nbytes together, so a copy whose operands are 4-aligned keeps the 4-byte
// body and anything else degrades to a narrower element instead of a mis-addressed one. An int32 body followed
// by a byte tail is NOT equivalent: it still hands the int32 loads and stores an operand that is not 4-aligned
// (a dim==3 CONCAT of F16 or I8 data whose src0 byte count is odd places the second copy at exactly such an
// offset).
inline size_t ggml_sycl_graph_safe_memcpy_width(const void * dst, const void * src, size_t nbytes) {
    const uintptr_t bits = reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src) | nbytes;
    if ((bits & 3) == 0) {
        return 4;
    }
    if ((bits & 1) == 0) {
        return 2;
    }
    return 1;
}
