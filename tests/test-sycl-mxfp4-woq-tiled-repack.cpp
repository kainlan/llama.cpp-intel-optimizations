// XMX_TILED MXFP4 -> WOQ repack kernel device test: the production
// repack_mxfp4_xmx_tiled_to_woq (perf-recovery epic, track C2b Option T,
// llama.cpp-ntfx) must invert the tiled materializer's byte layout
// (ggml_sycl::moe_tile_convert::reorder_mxfp4_to_xmx_tiled /
// reorder_mxfp4_aos_to_xmx_tiled, ggml/src/ggml-sycl/moe-tile-convert.cpp:
// 24-173) into the SAME destination WOQ layout the C1 spike proved on
// hardware (llama.cpp-4m9p comment c-a5by) and that C2's
// repack_mxfp4_soa_to_woq already produces: sequential nibble packing for
// weights {K,N} strides {N,1}, and e8m0 scales {K/QK_MXFP4,N} strides
// {N,1}.
//
// Two independent checks (the C2 lesson, repo memory `a-positive-control-
// can-itself-be-void`: a shared formula bug between the kernel and its
// checker would go undetected by either check alone):
//   1. A HAND-COMPUTED known-answer fixture at the smallest meaningful
//      tiled shape -- one tile group (blocks_per_row=1 so K=32, and
//      nrows <= tile_n_total so there is exactly one tile group along N
//      too). Literal source bytes + a literal expected array, both derived
//      by hand from the layout definitions above (worked examples for the
//      first 4 destination elements are in the comments below) -- NO
//      shared index logic with the kernel.
//   2. A randomized fixture at a larger, irregular shape (multiple K
//      tile-groups, N not a multiple of tile_n_total so the last N tile
//      group is padded). This does NOT reuse the kernel's inverse-map
//      arithmetic OR check 1's hand values: it starts from a random
//      LOGICAL weight matrix (an independent nibble value per (k,n) and an
//      independent scale byte per (k_block,n)), writes that matrix into
//      the XMX_TILED byte layout using a from-scratch FORWARD writer that
//      mirrors the production materializer's own arithmetic (cited by
//      file:line at the writer below -- moe-tile-convert.cpp:38-103,
//      specifically reorder_mxfp4_to_xmx_tiled's group_offset / scale-plane
//      / qs-plane loops), then compares the kernel's output against the
//      DIRECT logical->sequential encoding of that same logical matrix
//      (trivial packing, sharing no code with either the kernel or the
//      forward writer). This triangulates through the logical definition
//      of the weight matrix rather than through any shared index formula.
#include "ggml-common.h"
#include "ggml-sycl/convert.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <sycl/sycl.hpp>
#include <vector>

// Uploads src/dst buffers, runs the production kernel, downloads both planes,
// and memcmp's them against the caller-supplied expected arrays. Returns 0 on
// match, 1 on mismatch (after printing the first differing byte of whichever
// plane failed first).
static int run_and_check(sycl::queue &   q,
                         const char *    label,
                         const uint8_t * src_tiled,
                         size_t          src_bytes,
                         int             blocks_per_row,
                         int             nrows,
                         int             tile_n_total,
                         const uint8_t * expected_nibbles,
                         size_t          nibbles_bytes,
                         const uint8_t * expected_scales,
                         size_t          scales_bytes) {
    uint8_t * src_dev   = (uint8_t *) sycl::malloc_device(src_bytes, q);
    uint8_t * nib_dev   = (uint8_t *) sycl::malloc_device(nibbles_bytes, q);
    uint8_t * scale_dev = (uint8_t *) sycl::malloc_device(scales_bytes, q);
    q.memcpy(src_dev, src_tiled, src_bytes).wait();

    repack_mxfp4_xmx_tiled_to_woq(src_dev, nib_dev, scale_dev, blocks_per_row, nrows, tile_n_total, &q);
    q.wait();

    std::vector<uint8_t> got_nibbles(nibbles_bytes);
    std::vector<uint8_t> got_scales(scales_bytes);
    q.memcpy(got_nibbles.data(), nib_dev, nibbles_bytes).wait();
    q.memcpy(got_scales.data(), scale_dev, scales_bytes).wait();

    sycl::free(src_dev, q);
    sycl::free(nib_dev, q);
    sycl::free(scale_dev, q);

    if (std::memcmp(got_nibbles.data(), expected_nibbles, nibbles_bytes) != 0) {
        for (size_t i = 0; i < nibbles_bytes; ++i) {
            if (got_nibbles[i] != expected_nibbles[i]) {
                std::printf("FAIL[%s]: nibble-plane mismatch at byte %zu: got=0x%02x want=0x%02x\n", label, i,
                            got_nibbles[i], expected_nibbles[i]);
                return 1;
            }
        }
    }
    if (std::memcmp(got_scales.data(), expected_scales, scales_bytes) != 0) {
        for (size_t i = 0; i < scales_bytes; ++i) {
            if (got_scales[i] != expected_scales[i]) {
                std::printf("FAIL[%s]: scale-plane mismatch at index %zu: got=0x%02x want=0x%02x\n", label, i,
                            got_scales[i], expected_scales[i]);
                return 1;
            }
        }
    }
    std::printf("OK[%s]: repack_mxfp4_xmx_tiled_to_woq matches (%zu nibble bytes, %zu scale bytes)\n", label,
                nibbles_bytes, scales_bytes);
    return 0;
}

// ---------------------------------------------------------------------------
// Check 1: hand-computed known-answer fixture.
//
// Shape: blocks_per_row=1 (K=32), nrows=3 (N=3), tile_n_total=4. nrows <=
// tile_n_total and blocks_per_row=1 means the whole source is exactly ONE
// tile group (tg_k=0, tg_n=0): group_bytes = tile_n_total*(1+16) = 68 bytes.
//
// Tile-group byte layout (moe-tile-convert.cpp:50-99, verbatim-copied per
// row -- the qs bytes are the SAME j/j+16-interleaved 16-byte block the SOA
// source and dequantize_tile_mxfp4_soa_rowmajor use):
//   bytes[0..3]   = scale bytes for output rows tn=0,1,2,3 (tn=3 is padding
//                   for row index 3, which is >= nrows=3 and never read).
//   bytes[4..19]  = 16-byte qs block for row 0 (tn=0).
//   bytes[20..35] = 16-byte qs block for row 1 (tn=1).
//   bytes[36..51] = 16-byte qs block for row 2 (tn=2).
//   bytes[52..67] = 16-byte qs block for the padding row (tn=3, unused).
//
// Row 0 qs bytes: byte[j] = j | ((15-j)<<4) -- so row0's 32 elements are
// [0,1,...,15, 15,14,...,0] (ascending then descending).
// Row 1 qs bytes: byte[j] = v | (v<<4), v=(j+8)%16 -- so row1's 32 elements
// are [8,9,...,15,0,1,...,7] repeated twice (period 16).
// Row 2 qs bytes: byte[j] = j | (j<<4) -- so row2's 32 elements are
// [0,1,...,15] repeated twice (period 16, low nibble == high nibble).
//
// Destination: el = k*N+n (N=3), dst byte el/2, low half if el even else
// high; value = the source row-`n`'s element at position k (k in [0,32),
// single k-block so k_local=k directly).
//
// Hand-derivation of the first 4 destination elements:
//   el=0 (k=0,n=0/row0): row0 element 0 = qs_row0[0] low nibble.
//     qs_row0[0] = 0 | (15<<4) = 0xF0 -> low nibble 0x0. el even -> dst
//     byte 0, LOW half = 0x0.
//   el=1 (k=0,n=1/row1): row1 element 0 = qs_row1[0] low nibble.
//     qs_row1[0]: v=(0+8)%16=8 -> byte = 8|(8<<4) = 0x88 -> low nibble 0x8.
//     el odd -> dst byte 0, HIGH half = 0x8.
//     => dst_nibbles[0] = 0x0 | (0x8<<4) = 0x80.
//   el=2 (k=0,n=2/row2): row2 element 0 = qs_row2[0] low nibble.
//     qs_row2[0] = 0 | (0<<4) = 0x00 -> low nibble 0x0. el even -> dst
//     byte 1, LOW half = 0x0.
//   el=3 (k=1,n=0/row0): row0 element 1 = qs_row0[1] low nibble.
//     qs_row0[1] = 1 | (14<<4) = 0xE1 -> low nibble 0x1. el odd -> dst
//     byte 1, HIGH half = 0x1.
//     => dst_nibbles[1] = 0x0 | (0x1<<4) = 0x10.
// Both spot-checks match the literal arrays below (dst_nibbles[0]=0x80,
// dst_nibbles[1]=0x10).
//
// Scales: single k-group (kg=0), so dst_scales[n] is simply the scale byte
// of row n from the (only) tile group: {row0, row1, row2} = {0xAB,0xCD,0xEF}
// (the padding-row scale byte, 0x00 at src[3], is never read).
// ---------------------------------------------------------------------------
static int check_hand_fixture(sycl::queue & q) {
    constexpr int blocks_per_row = 1;
    constexpr int nrows          = 3;
    constexpr int tile_n_total   = 4;

    static const uint8_t qs_row0[16] = { 0xF0, 0xE1, 0xD2, 0xC3, 0xB4, 0xA5, 0x96, 0x87,
                                         0x78, 0x69, 0x5A, 0x4B, 0x3C, 0x2D, 0x1E, 0x0F };
    static const uint8_t qs_row1[16] = { 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF,
                                         0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77 };
    static const uint8_t qs_row2[16] = { 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
                                         0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF };
    static const uint8_t qs_pad[16]  = { 0 };  // never read (row index 3 >= nrows=3)

    uint8_t src[68];
    src[0] = 0xAB;  // scale, row0
    src[1] = 0xCD;  // scale, row1
    src[2] = 0xEF;  // scale, row2
    src[3] = 0x00;  // scale, padding row -- never read
    std::memcpy(src + 4, qs_row0, 16);
    std::memcpy(src + 20, qs_row1, 16);
    std::memcpy(src + 36, qs_row2, 16);
    std::memcpy(src + 52, qs_pad, 16);

    // Full expected nibble plane (48 bytes = K*N/2 = 32*3/2), derived by hand
    // by walking el=0..95 through the row0/row1/row2 element sequences above
    // (the first two bytes are spot-checked in the comment block above; the
    // remaining bytes follow the identical mechanical rule).
    static const uint8_t expected_nibbles[48] = {
        0x80, 0x10, 0x19, 0xA2, 0x32, 0x3B, 0xC4, 0x54, 0x5D, 0xE6, 0x76, 0x7F, 0x08, 0x98, 0x91, 0x2A,
        0xBA, 0xB3, 0x4C, 0xDC, 0xD5, 0x6E, 0xFE, 0xF7, 0x8F, 0xE0, 0x19, 0xAD, 0xC2, 0x3B, 0xCB, 0xA4,
        0x5D, 0xE9, 0x86, 0x7F, 0x07, 0x68, 0x91, 0x25, 0x4A, 0xB3, 0x43, 0x2C, 0xD5, 0x61, 0x0E, 0xF7,
    };
    static const uint8_t expected_scales[3] = { 0xAB, 0xCD, 0xEF };

    return run_and_check(q, "hand-fixture", src, sizeof(src), blocks_per_row, nrows, tile_n_total, expected_nibbles,
                         sizeof(expected_nibbles), expected_scales, sizeof(expected_scales));
}

// ---------------------------------------------------------------------------
// Check 2: randomized fixture, triangulated through an independent logical
// definition (does NOT share index arithmetic with the kernel, nor with
// check 1's hand values).
//
// Shape: blocks_per_row=3 (K=96), nrows=11 (N=11), tile_n_total=4 ->
// n_tile_groups_k=3, n_tile_groups_n=ceil(11/4)=3 (the last N tile group
// holds rows 8,9,10 plus one padding row) -- exercises multiple K
// tile-groups, multiple N tile-groups, and tile-group padding together.
// ---------------------------------------------------------------------------
static int check_randomized(sycl::queue & q) {
    constexpr int blocks_per_row    = 3;
    constexpr int nrows             = 11;
    constexpr int tile_n_total      = 4;
    const int64_t K                 = (int64_t) blocks_per_row * QK_MXFP4;
    const int64_t N                 = nrows;
    const int64_t n_tile_groups_k   = blocks_per_row;
    const int64_t n_tile_groups_n   = (N + tile_n_total - 1) / tile_n_total;
    const int64_t group_bytes       = (int64_t) tile_n_total * (1 + QK_MXFP4 / 2);
    const int64_t total_tiled_bytes = n_tile_groups_k * n_tile_groups_n * group_bytes;

    // Step 1: an independent random LOGICAL weight matrix -- one e2m1 nibble
    // value (0..15) per (k,n) and one opaque e8m0 scale byte per
    // (k_block,n). This is the ground truth; nothing below derives from the
    // kernel or from check 1.
    std::mt19937                       rng(43);
    std::uniform_int_distribution<int> nib_dist(0, 15);
    std::uniform_int_distribution<int> byte_dist(0, 255);

    std::vector<uint8_t> logical_nibble((size_t) (K * N));  // [k*N+n]
    for (auto & v : logical_nibble) {
        v = (uint8_t) nib_dist(rng);
    }
    std::vector<uint8_t> logical_scale((size_t) (blocks_per_row * N));  // [k_block*N+n]
    for (auto & v : logical_scale) {
        v = (uint8_t) byte_dist(rng);
    }

    // Step 2: an independent FORWARD writer that packs the logical matrix
    // into the XMX_TILED byte layout, mirroring the production
    // materializer's OWN arithmetic (ggml_sycl::moe_tile_convert::
    // reorder_mxfp4_to_xmx_tiled, ggml/src/ggml-sycl/moe-tile-convert.cpp:
    // 38-103): group_offset = (tg_k*n_tile_groups_n+tg_n)*group_bytes
    // (moe-tile-convert.cpp:52-53), scale plane is tile_n_total bytes in row
    // order (moe-tile-convert.cpp:72-82), qs plane is tile_n_total 16-byte
    // blocks in row order with byte b's low/high nibble holding elements b
    // and b+16 of that row's 32-element k-block (moe-tile-convert.cpp:84-98
    // copies the source block's 16 bytes verbatim, and the source block's
    // own j/j+16 packing is dequantize_tile_mxfp4_soa_rowmajor's, i.e. byte
    // b already IS low=element b, high=element b+16). Out-of-range rows
    // (padding) are zero-filled, matching the production writer.
    std::vector<uint8_t> tiled(total_tiled_bytes, 0);
    for (int64_t tg_k = 0; tg_k < n_tile_groups_k; ++tg_k) {
        for (int64_t tg_n = 0; tg_n < n_tile_groups_n; ++tg_n) {
            const int64_t group_offset = (tg_k * n_tile_groups_n + tg_n) * group_bytes;
            for (int64_t tn = 0; tn < tile_n_total; ++tn) {
                const int64_t out_col    = tg_n * tile_n_total + tn;
                tiled[group_offset + tn] = (out_col < N) ? logical_scale[(size_t) (tg_k * N + out_col)] : 0;
            }
            for (int64_t tn = 0; tn < tile_n_total; ++tn) {
                const int64_t out_col    = tg_n * tile_n_total + tn;
                const int64_t block_base = group_offset + tile_n_total + tn * (QK_MXFP4 / 2);
                if (out_col >= N) {
                    for (int b = 0; b < QK_MXFP4 / 2; ++b) {
                        tiled[block_base + b] = 0;
                    }
                    continue;
                }
                for (int b = 0; b < QK_MXFP4 / 2; ++b) {
                    const int64_t k_lo    = tg_k * QK_MXFP4 + b;
                    const int64_t k_hi    = tg_k * QK_MXFP4 + b + QK_MXFP4 / 2;
                    const uint8_t lo      = logical_nibble[(size_t) (k_lo * N + out_col)];
                    const uint8_t hi      = logical_nibble[(size_t) (k_hi * N + out_col)];
                    tiled[block_base + b] = (uint8_t) (lo | (hi << 4));
                }
            }
        }
    }

    // Step 3: the expected repack output is the DIRECT logical->sequential
    // encoding of the same logical matrix -- trivial packing per the WOQ
    // destination definition (el=k*N+n, dst byte el/2; scale dst[k_block*N+n]
    // = logical_scale), sharing no code with either the kernel or the
    // forward writer above.
    std::vector<uint8_t> expected_nibbles((size_t) (K * N / 2));
    for (int64_t el = 0; el < K * N; el += 2) {
        const uint8_t lo                    = logical_nibble[(size_t) el];
        const uint8_t hi                    = logical_nibble[(size_t) el + 1];
        expected_nibbles[(size_t) (el / 2)] = (uint8_t) (lo | (hi << 4));
    }
    const std::vector<uint8_t> & expected_scales = logical_scale;

    return run_and_check(q, "randomized", tiled.data(), tiled.size(), blocks_per_row, nrows, tile_n_total,
                         expected_nibbles.data(), expected_nibbles.size(), expected_scales.data(),
                         expected_scales.size());
}

int main() {
    try {
        sycl::queue q{ sycl::gpu_selector_v };

        // Fixture check first: smaller, and its failure messages are directly
        // traceable to the hand derivation in the comments above.
        int rc = check_hand_fixture(q);
        if (rc != 0) {
            return rc;
        }
        rc = check_randomized(q);
        if (rc != 0) {
            return rc;
        }
        return 0;
    } catch (const sycl::exception & ex) {
        std::printf("SKIP: no SYCL GPU (%s)\n", ex.what());
        return 77;
    }
}
