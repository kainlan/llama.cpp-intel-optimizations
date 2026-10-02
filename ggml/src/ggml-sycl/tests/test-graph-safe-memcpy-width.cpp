//
// Test: element width of the recording-time copy kernel (llama.cpp-qhfp)
//
// While a SYCL graph records, ggml_sycl_graph_safe_memcpy submits a kernel copy instead of a memcpy node.
// The kernel used to copy nbytes / 4 int32 elements and then the tail bytes, which misaligns the loads and
// stores whenever dst or src is not 4-aligned -- e.g. the second copy of a dim==3 contiguous CONCAT of
// element size < 4 with an odd src0 byte count. The width helper is the pure rule that replaced it; this pins
// it. Host-only: the helper includes no SYCL header.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "graph-safe-memcpy-width.hpp"

#include <cstdio>

// The build is -DNDEBUG (Release), so assert() would compile away and the test would pass vacuously.
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

int main() {
    alignas(16) static char buf[64];
    char * const            a = buf;  // 16-aligned

    // Everything 4-aligned keeps the wide copy.
    CHECK(ggml_sycl_graph_safe_memcpy_width(a, a + 32, 16) == 4, "aligned operands use 4-byte elements");
    CHECK(ggml_sycl_graph_safe_memcpy_width(a + 4, a + 36, 28) == 4, "4-aligned offsets still use 4-byte elements");

    // A byte count that is not a multiple of 4 cannot be covered by 4-byte elements, tail or no tail.
    CHECK(ggml_sycl_graph_safe_memcpy_width(a, a + 32, 6) == 2, "a 2-multiple byte count uses 2-byte elements");
    CHECK(ggml_sycl_graph_safe_memcpy_width(a, a + 32, 7) == 1, "an odd byte count uses 1-byte elements");

    // A misaligned operand pulls the whole copy down even when nbytes is a multiple of 4: this is the CONCAT
    // case (dst = base + size0 with size0 odd) the old int32 body mishandled.
    CHECK(ggml_sycl_graph_safe_memcpy_width(a + 1, a + 32, 8) == 1, "an odd dst forces byte elements");
    CHECK(ggml_sycl_graph_safe_memcpy_width(a, a + 33, 8) == 1, "an odd src forces byte elements");
    CHECK(ggml_sycl_graph_safe_memcpy_width(a + 2, a + 32, 8) == 2, "a 2-aligned dst forces 2-byte elements");
    CHECK(ggml_sycl_graph_safe_memcpy_width(a + 2, a + 34, 8) == 2, "2-aligned operands use 2-byte elements");

    // The width always divides the byte count, so a single element loop covers the copy exactly.
    for (size_t off = 0; off < 4; ++off) {
        for (size_t n = 1; n <= 16; ++n) {
            const size_t w = ggml_sycl_graph_safe_memcpy_width(a + off, a + 32, n);
            CHECK(w == 1 || w == 2 || w == 4, "width is 1, 2 or 4");
            CHECK(n % w == 0, "width divides the byte count");
            CHECK((reinterpret_cast<uintptr_t>(a + off) % w) == 0, "dst is aligned to the width");
            CHECK((reinterpret_cast<uintptr_t>(a + 32) % w) == 0, "src is aligned to the width");
        }
    }

    std::printf("PASS: graph-safe memcpy width\n");
    return 0;
}
