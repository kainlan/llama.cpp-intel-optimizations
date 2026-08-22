// SOA MXFP4 -> WOQ repack kernel device test: the production
// repack_mxfp4_soa_to_woq (perf-recovery epic, track C, llama.cpp-8rug) must
// reproduce the exact source SOA layout (dequantize_tile_mxfp4_soa_rowmajor's
// inverse) and the destination WOQ layout the C1 spike proved on hardware
// (llama.cpp-4m9p comment c-a5by): sequential nibble packing for weights
// {K,N} strides {N,1}, and e8m0 scales {K/QK_MXFP4,N} strides {N,1}.
//
// Two independent checks:
//   1. A HAND-COMPUTED known-answer fixture (see derivation comments below).
//      Its expected arrays are literal constants worked out by hand from the
//      layout definitions, NOT via code that shares the kernel's index
//      arithmetic -- a formula bug shared between the kernel and a CPU
//      "reference" that reimplements the same formula would pass check #2
//      below undetected; this check exists specifically to catch that.
//   2. A randomized fixture at a larger, less regular shape (odd N, multiple
//      k-blocks), checked against a CPU reference that DOES mirror the
//      kernel's index derivation -- this catches launch/offset/OOB bugs that
//      a tiny hand fixture is too small to exercise.
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
                         const uint8_t * src,
                         size_t          src_bytes,
                         int             blocks_per_row,
                         int             nrows,
                         const uint8_t * expected_nibbles,
                         size_t          nibbles_bytes,
                         const uint8_t * expected_scales,
                         size_t          scales_bytes) {
    uint8_t * src_dev   = (uint8_t *) sycl::malloc_device(src_bytes, q);
    uint8_t * nib_dev   = (uint8_t *) sycl::malloc_device(nibbles_bytes, q);
    uint8_t * scale_dev = (uint8_t *) sycl::malloc_device(scales_bytes, q);
    q.memcpy(src_dev, src, src_bytes).wait();

    repack_mxfp4_soa_to_woq(src_dev, nib_dev, scale_dev, blocks_per_row, nrows, &q);
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
    std::printf("OK[%s]: repack_mxfp4_soa_to_woq matches (%zu nibble bytes, %zu scale bytes)\n", label, nibbles_bytes,
                scales_bytes);
    return 0;
}

// ---------------------------------------------------------------------------
// Check 1: hand-computed known-answer fixture.
//
// Shape: blocks_per_row=1, nrows=2 -> K=32, N=2. blocks_per_row=1 collapses
// the general transform to something hand-tractable: with only one k-block,
// every row is exactly one SOA block (block_i = row), and dst el = k*N+n =
// 2k+n means the destination is simply "row0's 32 elements interleaved with
// row1's 32 elements" -- dst byte b's low nibble is row0 element b, high
// nibble is row1 element b. Scales: only one k-group (kg=0), so the {K/32,N}
// transpose is a straight copy of the two per-row e8m0 bytes.
//
// Source qs layout (SOA, dequantize_tile_mxfp4_soa_rowmajor's inverse): row r
// occupies qs[16r .. 16r+15]; byte j (j=0..15) holds element j in its low
// nibble and element j+16 in its high nibble.
//
// Row 0 bytes are chosen as byte[j] = j | ((15-j)<<4), so row0's 32 elements
// are [0,1,...,15, 15,14,...,0] (ascending then descending).
// Row 1 bytes are chosen as byte[j] = v | (v<<4) with v=(j+8)%16, so row1's
// 32 elements are [8,9,...,15,0,1,...,7] repeated twice (period 16, so
// elements 16..31 equal elements 0..15).
//
// Hand-derivation of 4 representative elements (block_i, source byte, which
// nibble half, destination byte, which nibble half):
//   el=0  (even, k=0 <16,  n=0/row0): block_i=0*1+(0/32)=0, k_local=0<16 ->
//     LOW nibble of qs[0*16+0]=qs[0]=0xF0 -> nibble 0x0. el even -> dst byte
//     0, LOW half. Row0 element 0 = 0.  [dst_nibbles[0] low nibble = 0x0]
//   el=1  (odd,  k=0 <16,  n=1/row1): block_i=1*1+0=1, k_local=0<16 -> LOW
//     nibble of qs[1*16+0]=qs[16]=0x88 -> nibble 0x8. el odd -> dst byte 0,
//     HIGH half. Row1 element 0 = 8.  [dst_nibbles[0] high nibble = 0x8]
//     => dst_nibbles[0] = 0x0 | (0x8<<4) = 0x80.
//   el=32 (even, k=16>=16, n=0/row0): block_i=0*1+(16/32)=0, k_local=16-0=16
//     >=16 -> HIGH nibble of qs[0*16+(16-16)]=qs[0]=0xF0 -> nibble 0xF. Row0
//     element 16 = 15 (start of the descending half). el even -> dst byte
//     32/2=16, LOW half.  [dst_nibbles[16] low nibble = 0xF]
//   el=33 (odd,  k=16>=16, n=1/row1): block_i=1*1+0=1, k_local=16 -> HIGH
//     nibble of qs[1*16+0]=qs[16]=0x88 -> nibble 0x8. Row1 element 16 = 8
//     (period-16 repeat). el odd -> dst byte 16, HIGH half.
//     [dst_nibbles[16] high nibble = 0x8] => dst_nibbles[16] = 0xF | (0x8<<4) = 0x8F.
// Both spot-checks match the literal arrays below.
// ---------------------------------------------------------------------------
static int check_hand_fixture(sycl::queue & q) {
    constexpr int blocks_per_row = 1;
    constexpr int nrows          = 2;

    // qs plane: row0 (block_i=0) then row1 (block_i=1), 16 bytes each.
    // Row 0: byte[j] = j | ((15-j)<<4).
    static const uint8_t qs_row0[16] = { 0xF0, 0xE1, 0xD2, 0xC3, 0xB4, 0xA5, 0x96, 0x87,
                                         0x78, 0x69, 0x5A, 0x4B, 0x3C, 0x2D, 0x1E, 0x0F };
    // Row 1: byte[j] = v | (v<<4), v=(j+8)%16.
    static const uint8_t qs_row1[16] = { 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF,
                                         0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77 };
    uint8_t              qs[32];
    std::memcpy(qs, qs_row0, sizeof(qs_row0));
    std::memcpy(qs + sizeof(qs_row0), qs_row1, sizeof(qs_row1));
    // e8m0 plane: one raw byte per row. Values are arbitrary and opaque to
    // this kernel (pure byte transpose, no float decode) -- chosen distinct
    // for traceability only.
    static const uint8_t e_plane[2] = { 0xAB, 0xCD };
    uint8_t              src[34];
    std::memcpy(src, qs, sizeof(qs));
    std::memcpy(src + sizeof(qs), e_plane, sizeof(e_plane));

    // Expected dst nibble plane: dst[b] = row0_elem[b] | (row1_elem[b] << 4).
    static const uint8_t expected_nibbles[32] = {
        0x80, 0x91, 0xA2, 0xB3, 0xC4, 0xD5, 0xE6, 0xF7, 0x08, 0x19, 0x2A, 0x3B, 0x4C, 0x5D, 0x6E, 0x7F,
        0x8F, 0x9E, 0xAD, 0xBC, 0xCB, 0xDA, 0xE9, 0xF8, 0x07, 0x16, 0x25, 0x34, 0x43, 0x52, 0x61, 0x70,
    };
    // Expected dst scale plane: single k-group (kg=0) -> straight copy of
    // the two per-row e8m0 bytes.
    static const uint8_t expected_scales[2] = { 0xAB, 0xCD };

    return run_and_check(q, "hand-fixture", src, sizeof(src), blocks_per_row, nrows, expected_nibbles,
                         sizeof(expected_nibbles), expected_scales, sizeof(expected_scales));
}

// ---------------------------------------------------------------------------
// Check 2: randomized fixture at a larger, irregular shape, checked against
// a CPU reference that mirrors the kernel's own index derivation. This does
// NOT substitute for check 1 (a formula bug shared by both copies would pass
// here) -- it exists to exercise launch geometry and boundary handling that
// the tiny hand fixture above is too small to reach.
// ---------------------------------------------------------------------------
static int check_randomized(sycl::queue & q) {
    // Multi-block, multi-row, odd N -- exercises the k-block boundary and the
    // el/2 nibble-pair boundary crossing between rows (N odd means a byte's
    // low/high nibble can belong to different output rows).
    constexpr int blocks_per_row = 5;   // K = 160
    constexpr int nrows          = 11;  // N = 11
    const int64_t K              = (int64_t) blocks_per_row * QK_MXFP4;
    const int64_t N              = nrows;

    const int64_t nblocks   = (int64_t) nrows * blocks_per_row;
    const size_t  qs_bytes  = (size_t) nblocks * (QK_MXFP4 / 2);
    const size_t  e_bytes   = (size_t) nblocks;
    const size_t  src_bytes = qs_bytes + e_bytes;

    std::vector<uint8_t>               src_host(src_bytes);
    std::mt19937                       rng(42);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    for (size_t i = 0; i < src_bytes; ++i) {
        src_host[i] = (uint8_t) byte_dist(rng);
    }
    const uint8_t * qs      = src_host.data();
    const uint8_t * e_plane = qs + qs_bytes;

    // CPU reference: mirrors repack_mxfp4_soa_to_woq's index derivation exactly
    // (deliberately -- see the file-level comment on why check 1 is needed too).
    std::vector<uint8_t> ref_nibbles((size_t) (K * N / 2));
    auto                 nib_at = [&](int64_t el) -> uint8_t {
        const int64_t k       = el / N;
        const int64_t n       = el - k * N;
        const int64_t b       = k / QK_MXFP4;
        const int64_t k_local = k - b * QK_MXFP4;
        const int64_t block_i = n * blocks_per_row + b;
        const bool    lo      = k_local < QK_MXFP4 / 2;
        const uint8_t byte    = qs[block_i * (QK_MXFP4 / 2) + (lo ? k_local : k_local - QK_MXFP4 / 2)];
        return lo ? (byte & 0xf) : (byte >> 4);
    };
    for (int64_t el = 0; el < K * N; el += 2) {
        const uint8_t lo_nib = nib_at(el);
        const uint8_t hi_nib = nib_at(el + 1);
        ref_nibbles[el / 2]  = (uint8_t) (lo_nib | (hi_nib << 4));
    }

    std::vector<uint8_t> ref_scales((size_t) (blocks_per_row * N));
    for (int64_t kg = 0; kg < blocks_per_row; ++kg) {
        for (int64_t n = 0; n < N; ++n) {
            ref_scales[(size_t) (kg * N + n)] = e_plane[(size_t) (n * blocks_per_row + kg)];
        }
    }

    return run_and_check(q, "randomized", src_host.data(), src_bytes, blocks_per_row, nrows, ref_nibbles.data(),
                         ref_nibbles.size(), ref_scales.data(), ref_scales.size());
}

// ---------------------------------------------------------------------------
// Check 3 (llama.cpp-1lon): repack_mxfp4_soa_to_woq_batched -- ONE launch
// covering every active expert slot via an added grid dimension -- must
// reproduce N sequential calls to the pre-1lon per-slot
// repack_mxfp4_soa_to_woq (already verified by checks 1/2 above)
// byte-identically. This needs no independent reference: it compares the
// new kernel's output to the old kernel's output on the SAME random
// per-slot inputs, written into the SAME slot-strided destination layout
// ggml-sycl.cpp's batched PP executor uses (nibble_dst = weight_base +
// slot*weight_slot_bytes, scale_dst = nibble_dst + woq_nibble_slot_bytes).
// ---------------------------------------------------------------------------
static int check_batched_vs_per_slot(sycl::queue & q) {
    constexpr int blocks_per_row = 3;  // K = 96
    constexpr int nrows          = 7;  // N = 7
    constexpr int n_slots        = 5;  // exercises the added expert-slot grid dim
    const int64_t K              = (int64_t) blocks_per_row * QK_MXFP4;
    const int64_t N              = nrows;

    const int64_t nblocks   = (int64_t) nrows * blocks_per_row;
    const size_t  qs_bytes  = (size_t) nblocks * (QK_MXFP4 / 2);
    const size_t  e_bytes   = (size_t) nblocks;
    const size_t  src_bytes = qs_bytes + e_bytes;

    const size_t nibble_bytes      = (size_t) (K * N / 2);
    const size_t scale_bytes       = (size_t) (blocks_per_row * N);
    const size_t weight_slot_bytes = nibble_bytes + scale_bytes;

    std::mt19937                       rng(44);
    std::uniform_int_distribution<int> byte_dist(0, 255);

    std::vector<std::vector<uint8_t>> src_host(n_slots);
    std::vector<uint8_t *>            src_dev(n_slots);
    for (int s = 0; s < n_slots; ++s) {
        src_host[s].resize(src_bytes);
        for (auto & b : src_host[s]) {
            b = (uint8_t) byte_dist(rng);
        }
        src_dev[s] = (uint8_t *) sycl::malloc_device(src_bytes, q);
        q.memcpy(src_dev[s], src_host[s].data(), src_bytes).wait();
    }

    uint8_t * ref_base     = (uint8_t *) sycl::malloc_device(weight_slot_bytes * n_slots, q);
    uint8_t * batched_base = (uint8_t *) sycl::malloc_device(weight_slot_bytes * n_slots, q);

    // Reference: n_slots sequential calls to the pre-1lon per-slot kernel,
    // exactly mirroring the pre-1lon per-slot loop in ggml-sycl.cpp.
    for (int s = 0; s < n_slots; ++s) {
        uint8_t * nibble_dst = ref_base + (size_t) s * weight_slot_bytes;
        uint8_t * scale_dst  = nibble_dst + nibble_bytes;
        repack_mxfp4_soa_to_woq(src_dev[s], nibble_dst, scale_dst, blocks_per_row, nrows, &q);
    }
    q.wait();

    // Batched: ONE launch covering every slot.
    std::vector<const void *> srcs(src_dev.begin(), src_dev.end());
    repack_mxfp4_soa_to_woq_batched(srcs.data(), n_slots, batched_base, weight_slot_bytes, nibble_bytes, blocks_per_row,
                                    nrows, &q);
    q.wait();

    std::vector<uint8_t> got_ref(weight_slot_bytes * n_slots);
    std::vector<uint8_t> got_batched(weight_slot_bytes * n_slots);
    q.memcpy(got_ref.data(), ref_base, got_ref.size()).wait();
    q.memcpy(got_batched.data(), batched_base, got_batched.size()).wait();

    for (int s = 0; s < n_slots; ++s) {
        sycl::free(src_dev[s], q);
    }
    sycl::free(ref_base, q);
    sycl::free(batched_base, q);

    if (std::memcmp(got_ref.data(), got_batched.data(), got_ref.size()) != 0) {
        for (size_t i = 0; i < got_ref.size(); ++i) {
            if (got_ref[i] != got_batched[i]) {
                std::printf(
                    "FAIL[batched-vs-per-slot]: mismatch at byte %zu (slot %zu): per_slot=0x%02x batched=0x%02x\n", i,
                    i / weight_slot_bytes, got_ref[i], got_batched[i]);
                return 1;
            }
        }
    }
    std::printf(
        "OK[batched-vs-per-slot]: repack_mxfp4_soa_to_woq_batched matches %d sequential repack_mxfp4_soa_to_woq "
        "calls (%zu bytes/slot)\n",
        n_slots, weight_slot_bytes);
    return 0;
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
        rc = check_batched_vs_per_slot(q);
        if (rc != 0) {
            return rc;
        }
        return 0;
    } catch (const sycl::exception & ex) {
        std::printf("SKIP: no SYCL GPU (%s)\n", ex.what());
        return 77;
    }
}
