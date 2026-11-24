//
// Host-only gate for the inverse of the AOS->SOA weight reorder (ggml/src/ggml-sycl/soa-readback.hpp).
//
// The SYCL backend may store a quantized weight as SOA: one plane per block field, each plane holding that field
// of every block in block order. A reader of the tensor must get ggml's block bytes back, so get_tensor rebuilds
// them with ggml_sycl_soa_to_aos_window. This test builds the SOA image from the block structs of ggml-common.h
// (field by field, never from the header's numbers), asks the inverse for windows of the AOS image, and compares.
// It also holds the header's plane table to the one list of SOA types (soa-reorder-types.hpp).
//
// No SYCL device, queue or backend library is touched.
//

#include "ggml-sycl/soa-readback.hpp"
#include "ggml-sycl/soa-reorder-types.hpp"
#include "ggml.h"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const char * type_name, const char * what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s: %s\n", type_name, what);
        ++g_failures;
    }
}

void append(std::vector<uint8_t> & out, const void * src, size_t n) {
    const uint8_t * p = static_cast<const uint8_t *>(src);
    out.insert(out.end(), p, p + n);
}

template <typename block_t> block_t block_at(const std::vector<uint8_t> & aos, size_t i) {
    block_t blk;
    std::memcpy(&blk, aos.data() + i * sizeof(block_t), sizeof(block_t));
    return blk;
}

// The SOA images, stated from the structs. Plane order as the fills in ggml-sycl.cpp document it:
// Q4_0 and Q8_0 [qs][d], Q4_K [qs][scales][dm], Q6_K [ql][qh][scales][d], MXFP4 [qs][e].
std::vector<uint8_t> soa_q4_0(const std::vector<uint8_t> & aos) {
    const size_t         n = aos.size() / sizeof(block_q4_0);
    std::vector<uint8_t> out;
    for (size_t i = 0; i < n; ++i) {
        const block_q4_0 b = block_at<block_q4_0>(aos, i);
        append(out, b.qs, sizeof(b.qs));
    }
    for (size_t i = 0; i < n; ++i) {
        const block_q4_0 b = block_at<block_q4_0>(aos, i);
        append(out, &b.d, sizeof(b.d));
    }
    return out;
}

std::vector<uint8_t> soa_q8_0(const std::vector<uint8_t> & aos) {
    const size_t         n = aos.size() / sizeof(block_q8_0);
    std::vector<uint8_t> out;
    for (size_t i = 0; i < n; ++i) {
        const block_q8_0 b = block_at<block_q8_0>(aos, i);
        append(out, b.qs, sizeof(b.qs));
    }
    for (size_t i = 0; i < n; ++i) {
        const block_q8_0 b = block_at<block_q8_0>(aos, i);
        append(out, &b.d, sizeof(b.d));
    }
    return out;
}

std::vector<uint8_t> soa_q4_k(const std::vector<uint8_t> & aos) {
    const size_t         n = aos.size() / sizeof(block_q4_K);
    std::vector<uint8_t> out;
    for (size_t i = 0; i < n; ++i) {
        const block_q4_K b = block_at<block_q4_K>(aos, i);
        append(out, b.qs, sizeof(b.qs));
    }
    for (size_t i = 0; i < n; ++i) {
        const block_q4_K b = block_at<block_q4_K>(aos, i);
        append(out, b.scales, sizeof(b.scales));
    }
    for (size_t i = 0; i < n; ++i) {
        const block_q4_K b = block_at<block_q4_K>(aos, i);
        append(out, &b.data.dm, sizeof(b.data.dm));
    }
    return out;
}

std::vector<uint8_t> soa_q6_k(const std::vector<uint8_t> & aos) {
    const size_t         n = aos.size() / sizeof(block_q6_K);
    std::vector<uint8_t> out;
    for (size_t i = 0; i < n; ++i) {
        const block_q6_K b = block_at<block_q6_K>(aos, i);
        append(out, b.ql, sizeof(b.ql));
    }
    for (size_t i = 0; i < n; ++i) {
        const block_q6_K b = block_at<block_q6_K>(aos, i);
        append(out, b.qh, sizeof(b.qh));
    }
    for (size_t i = 0; i < n; ++i) {
        const block_q6_K b = block_at<block_q6_K>(aos, i);
        append(out, b.scales, sizeof(b.scales));
    }
    for (size_t i = 0; i < n; ++i) {
        const block_q6_K b = block_at<block_q6_K>(aos, i);
        append(out, &b.d, sizeof(b.d));
    }
    return out;
}

std::vector<uint8_t> soa_mxfp4(const std::vector<uint8_t> & aos) {
    const size_t         n = aos.size() / sizeof(block_mxfp4);
    std::vector<uint8_t> out;
    for (size_t i = 0; i < n; ++i) {
        const block_mxfp4 b = block_at<block_mxfp4>(aos, i);
        append(out, b.qs, sizeof(b.qs));
    }
    for (size_t i = 0; i < n; ++i) {
        const block_mxfp4 b = block_at<block_mxfp4>(aos, i);
        append(out, &b.e, sizeof(b.e));
    }
    return out;
}

struct type_case {
    ggml_type    type;
    const char * name;
    std::vector<uint8_t> (*to_soa)(const std::vector<uint8_t> &);
};

const type_case CASES[] = {
    { GGML_TYPE_Q4_0,  "q4_0",  soa_q4_0  },
    { GGML_TYPE_Q8_0,  "q8_0",  soa_q8_0  },
    { GGML_TYPE_Q4_K,  "q4_K",  soa_q4_k  },
    { GGML_TYPE_Q6_K,  "q6_K",  soa_q6_k  },
    { GGML_TYPE_MXFP4, "mxfp4", soa_mxfp4 },
};

constexpr size_t  N_BLOCKS   = 7;
constexpr size_t  GUARD      = 16;
constexpr uint8_t GUARD_BYTE = 0xA5;

void run_type(const type_case & tc) {
    const size_t block_bytes = ggml_type_size(tc.type);

    ggml_sycl_soa_plane_table table{};
    check(ggml_sycl_soa_plane_table_for_type(tc.type, &table), tc.name, "the type has no plane table");
    check(table.block_bytes == block_bytes, tc.name, "block_bytes is not ggml_type_size");
    // The planes are a partition of the block: every byte of the block belongs to exactly one plane.
    std::vector<int> owners(block_bytes, 0);
    for (size_t p = 0; p < table.n_planes; ++p) {
        for (size_t i = 0; i < table.planes[p].width; ++i) {
            const size_t at = table.planes[p].aos_offset + i;
            if (at < block_bytes) {
                ++owners[at];
            } else {
                check(false, tc.name, "a plane leaves the block");
            }
        }
    }
    for (size_t i = 0; i < block_bytes; ++i) {
        check(owners[i] == 1, tc.name, "the planes do not partition the block");
    }

    std::vector<uint8_t> aos(N_BLOCKS * block_bytes);
    uint32_t             s = 0x9E3779B9u + (uint32_t) tc.type;
    for (auto & b : aos) {
        s = s * 1664525u + 1013904223u;
        b = (uint8_t) (s >> 24);
    }
    const std::vector<uint8_t> soa = tc.to_soa(aos);
    check(soa.size() == aos.size(), tc.name, "the SOA image is not as large as the AOS image");
    if (soa.size() != aos.size()) {
        return;
    }

    const size_t offsets[] = { 0, 1, block_bytes - 1, block_bytes, block_bytes + 5, 3 * block_bytes, aos.size() };
    const size_t sizes[]   = { 0, 1, block_bytes - 1, block_bytes, block_bytes + 1, 2 * block_bytes + 3, aos.size() };
    for (size_t offset : offsets) {
        for (size_t size : sizes) {
            if (offset + size > aos.size()) {
                continue;
            }
            std::vector<uint8_t> got(size + 2 * GUARD, GUARD_BYTE);
            uint8_t *            dst = got.data() + GUARD;
            check(ggml_sycl_soa_to_aos_window(tc.type, soa.data(), N_BLOCKS, offset, size, dst), tc.name,
                  "a window inside the image was refused");
            check(std::memcmp(got.data() + GUARD, aos.data() + offset, size) == 0, tc.name,
                  "the window does not hold the AOS bytes");
            for (size_t i = 0; i < GUARD; ++i) {
                check(got[i] == GUARD_BYTE && got[GUARD + size + i] == GUARD_BYTE, tc.name,
                      "the inverse wrote outside the window");
            }
        }
    }

    uint8_t untouched = GUARD_BYTE;
    check(!ggml_sycl_soa_to_aos_window(tc.type, soa.data(), N_BLOCKS, aos.size(), 1, &untouched), tc.name,
          "a window past the image was accepted");
    check(untouched == GUARD_BYTE, tc.name, "a refused window wrote to dst");
}

}  // namespace

int main() {
    for (const type_case & tc : CASES) {
        run_type(tc);
    }
    // One list of SOA types: the plane table has exactly the types the reorder table has.
    for (int t = 0; t < (int) GGML_TYPE_COUNT; ++t) {
        ggml_sycl_soa_plane_table table{};
        const bool                has_planes = ggml_sycl_soa_plane_table_for_type((ggml_type) t, &table);
        if (has_planes != ggml_sycl_soa_reorder_supported_type((ggml_type) t)) {
            std::fprintf(stderr, "FAIL: ggml_type %d: plane table %d, reorder table %d\n", t, (int) has_planes,
                         (int) ggml_sycl_soa_reorder_supported_type((ggml_type) t));
            ++g_failures;
        }
    }
    uint8_t byte = 0;
    if (ggml_sycl_soa_to_aos_window(GGML_TYPE_F32, &byte, 1, 0, 1, &byte)) {
        std::fprintf(stderr, "FAIL: f32 has no SOA layout and was accepted\n");
        ++g_failures;
    }
    if (g_failures > 0) {
        std::fprintf(stderr, "test-sycl-soa-readback: %d checks failed\n", g_failures);
        return 1;
    }
    std::fprintf(stderr, "test-sycl-soa-readback: PASS\n");
    return 0;
}
