// SOA MXFP4 -> WOQ repack kernel device test: the production
// repack_mxfp4_soa_to_woq (perf-recovery epic, track C, llama.cpp-8rug) must
// reproduce a CPU reference computed directly from the source SOA layout
// (dequantize_tile_mxfp4_soa_rowmajor's inverse) and the destination WOQ
// layout the C1 spike proved on hardware (llama.cpp-4m9p comment c-a5by):
// sequential nibble packing for weights {K,N} strides {N,1}, and e8m0 scales
// {K/QK_MXFP4,N} strides {N,1}. Random SOA input is generated host-side,
// uploaded, repacked on-device, downloaded, and memcmp'd byte-for-byte
// against the reference for both planes.
#include "ggml-common.h"
#include "ggml-sycl/convert.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <sycl/sycl.hpp>
#include <vector>

int main() {
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

    // CPU reference: mirrors repack_mxfp4_soa_to_woq's index derivation exactly.
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

    try {
        sycl::queue q{ sycl::gpu_selector_v };

        uint8_t * src_dev   = (uint8_t *) sycl::malloc_device(src_bytes, q);
        uint8_t * nib_dev   = (uint8_t *) sycl::malloc_device(ref_nibbles.size(), q);
        uint8_t * scale_dev = (uint8_t *) sycl::malloc_device(ref_scales.size(), q);
        q.memcpy(src_dev, src_host.data(), src_bytes).wait();

        repack_mxfp4_soa_to_woq(src_dev, nib_dev, scale_dev, blocks_per_row, nrows, &q);
        q.wait();

        std::vector<uint8_t> got_nibbles(ref_nibbles.size());
        std::vector<uint8_t> got_scales(ref_scales.size());
        q.memcpy(got_nibbles.data(), nib_dev, got_nibbles.size()).wait();
        q.memcpy(got_scales.data(), scale_dev, got_scales.size()).wait();

        sycl::free(src_dev, q);
        sycl::free(nib_dev, q);
        sycl::free(scale_dev, q);

        if (std::memcmp(got_nibbles.data(), ref_nibbles.data(), ref_nibbles.size()) != 0) {
            for (size_t i = 0; i < ref_nibbles.size(); ++i) {
                if (got_nibbles[i] != ref_nibbles[i]) {
                    std::printf("FAIL: nibble-plane mismatch at byte %zu: got=0x%02x want=0x%02x\n", i, got_nibbles[i],
                                ref_nibbles[i]);
                    return 1;
                }
            }
        }
        if (std::memcmp(got_scales.data(), ref_scales.data(), ref_scales.size()) != 0) {
            for (size_t i = 0; i < ref_scales.size(); ++i) {
                if (got_scales[i] != ref_scales[i]) {
                    std::printf("FAIL: scale-plane mismatch at index %zu: got=0x%02x want=0x%02x\n", i, got_scales[i],
                                ref_scales[i]);
                    return 1;
                }
            }
        }
        std::printf("OK: repack_mxfp4_soa_to_woq matches CPU reference (%zu nibble bytes, %zu scale bytes)\n",
                    ref_nibbles.size(), ref_scales.size());
        return 0;
    } catch (const sycl::exception & ex) {
        std::printf("SKIP: no SYCL GPU (%s)\n", ex.what());
        return 77;
    }
}
