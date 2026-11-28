// Stored NVFP4 -> stored-order WOQ repack kernel device test (Inkling Task 26b-1, llama.cpp-7mub): the production
// repack_nvfp4_to_woq_nk_batched (ggml-sycl/convert.cpp/.hpp) must write the planes oneDNN reads as f4_e2m1 weights
// {B,K,N} with strides {K*N,1,K} and f16 scales {B,K/16,N} with strides {(K/16)*N,N,1}, byte for byte:
//   nibble (k, n) of slot s at byte s*(N*K/2) + (n*K + k)/2, even k in the low nibble;
//   scale  (g, n) of slot s at half s*(K/16)*N + g*N + n, value = the true E4M3 value of the stored scale byte.
//
// Three checks:
//   1. A hand fixture (one expert, K 64, N 2) whose expected arrays are literal constants worked out by hand below,
//      not by code that shares the kernel's arithmetic.
//   2. A randomized fixture (3 experts, K 1152, N 72) against a CPU mirror that decodes every (k, n) from the stored
//      order, the way make_nvfp4_problem in test-sycl-onednn-woq-nvfp4.cpp does, and places it by the formula above.
//      The sources are permuted, so a kernel that ignores the source table fails.
//   3. 130 slots of K 64 and N 8, which crosses GGML_SYCL_MXFP4_WOQ_REPACK_MAX_SLOTS (128): the entry launches
//      twice, and slots 128 and 129 must land at their own plane offsets.
// Every destination is prefilled with a sentinel and carries a guard tail, so a missed or out-of-bounds write fails.
#include "ggml-common.h"
#include "ggml-sycl/convert.hpp"
#include "test-skip.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <sycl/sycl.hpp>
#include <vector>

namespace {

constexpr uint8_t  k_sentinel_byte = 0xa5;
constexpr uint16_t k_sentinel_half = 0xa5a5;
constexpr size_t   k_guard_bytes   = 64;
constexpr size_t   k_guard_halves  = 32;

// The true E4M3 value of a stored NVFP4 scale byte, written independently of ggml_sycl_ue4m3_to_fp32: the sign bit is
// ignored and 0x7f / 0xff read as 0, as the SYCL decode does.
float e4m3_true(uint8_t x) {
    if ((x & 0x7f) == 0x7f) {
        return 0.0f;
    }
    const int exp = (x >> 3) & 0xf;
    const int man = x & 0x7;
    if (exp == 0) {
        return std::ldexp((float) man, -9);
    }
    return std::ldexp(1.0f + (float) man / 8.0f, exp - 7);
}

uint16_t f16_bits(float x) {
    return sycl::bit_cast<uint16_t>(sycl::half(x));
}

struct planes {
    std::vector<uint8_t>  nibbles;
    std::vector<uint16_t> scales;
};

// CPU mirror: slot s reads expert slot_expert[s] of `stored` (experts of N rows of K/64 block_nvfp4 each) and places
// every (k, n) by the interface formula. Element e of a block sits in sub-block e/16 at qs[(e/16)*8 + (e%16)%8], low
// nibble for e%16 < 8 and high nibble otherwise.
planes mirror(const std::vector<uint8_t> & stored, const std::vector<int> & slot_expert, int K, int N) {
    const int    nb           = K / QK_NVFP4;
    const int    G            = K / QK_NVFP4_SUB;
    const size_t expert_bytes = (size_t) N * nb * sizeof(block_nvfp4);
    const size_t nib_slot     = (size_t) N * K / 2;
    const size_t sc_slot      = (size_t) G * N;
    const size_t n_slots      = slot_expert.size();
    planes       p;
    p.nibbles.assign(nib_slot * n_slots, 0);
    p.scales.assign(sc_slot * n_slots, 0);
    for (size_t s = 0; s < n_slots; ++s) {
        const uint8_t * expert = stored.data() + expert_bytes * slot_expert[s];
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                const uint8_t * blk = expert + ((size_t) n * nb + k / QK_NVFP4) * sizeof(block_nvfp4);
                const int       e   = k % QK_NVFP4;
                const int       sb  = e / QK_NVFP4_SUB;
                const int       j   = e % QK_NVFP4_SUB;
                const uint8_t   q   = blk[QK_NVFP4 / QK_NVFP4_SUB + sb * 8 + (j & 7)];
                const uint8_t   c   = j < 8 ? (uint8_t) (q & 0xf) : (uint8_t) (q >> 4);
                const size_t    off = (size_t) n * K + k;
                p.nibbles[nib_slot * s + off / 2] |= (uint8_t) (c << ((off & 1) * 4));
            }
            for (int g = 0; g < G; ++g) {
                const uint8_t d = expert[((size_t) n * nb + g / 4) * sizeof(block_nvfp4) + g % 4];

                p.scales[sc_slot * s + (size_t) g * N + n] = f16_bits(e4m3_true(d));
            }
        }
    }
    return p;
}

// Uploads the experts, points slot s at expert slot_expert[s], runs the production entry once, and compares both
// planes (and their sentinel guard tails) against `want`. Returns 0 on match, 1 on the first mismatch. The slots in
// `own_offset_slots` are compared first, each at its own plane offset, so a slot written elsewhere fails by name.
int run_and_check(sycl::queue &                q,
                  const char *                 label,
                  const std::vector<uint8_t> & stored,
                  const std::vector<int> &     slot_expert,
                  int                          K,
                  int                          N,
                  const planes &               want,
                  const std::vector<int> &     own_offset_slots = {}) {
    const int    nb           = K / QK_NVFP4;
    const size_t expert_bytes = (size_t) N * nb * sizeof(block_nvfp4);
    const int    n_slots      = (int) slot_expert.size();
    const size_t nib_bytes    = want.nibbles.size();
    const size_t sc_halves    = want.scales.size();

    // 64-byte alignment covers the entry's 8-byte asserts on both destination planes.
    uint8_t *    src_dev = (uint8_t *) sycl::aligned_alloc_device(64, stored.size(), q);
    uint8_t *    nib_dev = (uint8_t *) sycl::aligned_alloc_device(64, nib_bytes + k_guard_bytes, q);
    sycl::half * sc_dev =
        (sycl::half *) sycl::aligned_alloc_device(64, (sc_halves + k_guard_halves) * sizeof(sycl::half), q);
    q.memcpy(src_dev, stored.data(), stored.size()).wait();
    q.memset(nib_dev, k_sentinel_byte, nib_bytes + k_guard_bytes).wait();
    q.memset(sc_dev, k_sentinel_byte, (sc_halves + k_guard_halves) * sizeof(sycl::half)).wait();

    std::vector<const void *> srcs(n_slots);
    for (int s = 0; s < n_slots; ++s) {
        srcs[s] = src_dev + expert_bytes * slot_expert[s];
    }
    repack_nvfp4_to_woq_nk_batched(srcs.data(), n_slots, nib_dev, sc_dev, K, N, &q);
    q.wait();

    std::vector<uint8_t>  got_nib(nib_bytes + k_guard_bytes);
    std::vector<uint16_t> got_sc(sc_halves + k_guard_halves);
    q.memcpy(got_nib.data(), nib_dev, got_nib.size()).wait();
    q.memcpy(got_sc.data(), sc_dev, got_sc.size() * sizeof(uint16_t)).wait();
    sycl::free(src_dev, q);
    sycl::free(nib_dev, q);
    sycl::free(sc_dev, q);

    const size_t nib_slot = nib_bytes / n_slots;
    const size_t sc_slot  = sc_halves / n_slots;
    for (const int s : own_offset_slots) {
        if (std::memcmp(got_nib.data() + nib_slot * s, want.nibbles.data() + nib_slot * s, nib_slot) != 0 ||
            std::memcmp(got_sc.data() + sc_slot * s, want.scales.data() + sc_slot * s, sc_slot * sizeof(uint16_t)) !=
                0) {
            std::printf("FAIL[%s]: slot %d is not at its own plane offset\n", label, s);
            return 1;
        }
    }
    for (size_t i = 0; i < got_nib.size(); ++i) {
        const uint8_t w = i < nib_bytes ? want.nibbles[i] : k_sentinel_byte;
        if (got_nib[i] != w) {
            std::printf("FAIL[%s]: nibble-plane mismatch at byte %zu (slot %zu%s): got=0x%02x want=0x%02x\n", label, i,
                        i / nib_slot, i < nib_bytes ? "" : ", guard tail", got_nib[i], w);
            return 1;
        }
    }
    for (size_t i = 0; i < got_sc.size(); ++i) {
        const uint16_t w = i < sc_halves ? want.scales[i] : k_sentinel_half;
        if (got_sc[i] != w) {
            std::printf("FAIL[%s]: scale-plane mismatch at half %zu (slot %zu%s): got=0x%04x want=0x%04x\n", label, i,
                        i / sc_slot, i < sc_halves ? "" : ", guard tail", got_sc[i], w);
            return 1;
        }
    }
    std::printf("OK[%s]: repack_nvfp4_to_woq_nk_batched matches (%d slots, %zu nibble bytes, %zu scale halves)\n",
                label, n_slots, nib_bytes, sc_halves);
    return 0;
}

std::vector<uint8_t> random_experts(int n_experts, int K, int N, uint32_t seed) {
    const size_t                       bytes = (size_t) n_experts * N * (K / QK_NVFP4) * sizeof(block_nvfp4);
    std::vector<uint8_t>               v(bytes);
    std::mt19937                       rng(seed);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    for (auto & b : v) {
        b = (uint8_t) byte_dist(rng);
    }
    return v;
}

// ---------------------------------------------------------------------------------------------------------------------
// Check 1: hand fixture. One expert, K 64 and N 2: each row is one block_nvfp4, and both rows hold the same bytes.
//
// Block bytes: d = { 0x38, 0x40, 0xb8, 0x7f }, then qs[i] = (uint8_t) (i * 0x11 + 1):
//   qs[ 0.. 7] = 01 12 23 34 45 56 67 78    qs[ 8..15] = 89 9a ab bc cd de ef 00
//   qs[16..23] = 11 22 33 44 55 66 77 88    qs[24..31] = 99 aa bb cc dd ee ff 10
//
// Stored sub-block s holds qs[s*8 + j], element j (j < 8) in the low nibble and element j+8 in the high nibble. Output
// byte t of the sub-block holds elements 2t (low) and 2t+1 (high), so with q = the sub-block's 8 bytes:
//   t < 4:  out[t] = (q[2t] & 0xf) | ((q[2t+1] & 0xf) << 4)
//   t >= 4: out[t] = (q[2t-8] >> 4) | (q[2t-7] & 0xf0)
// Sub-block 0, q = 01 12 23 34 45 56 67 78:
//   t=0: 1 | 2<<4 = 0x21   t=1: 3 | 4<<4 = 0x43   t=2: 5 | 6<<4 = 0x65   t=3: 7 | 8<<4 = 0x87
//   t=4: 0 | 0x10 = 0x10   t=5: 2 | 0x30 = 0x32   t=6: 4 | 0x50 = 0x54   t=7: 6 | 0x70 = 0x76
// Sub-block 1, q = 89 9a ab bc cd de ef 00:
//   t=0: 9 | a<<4 = 0xa9   t=1: b | c<<4 = 0xcb   t=2: d | e<<4 = 0xed   t=3: f | 0<<4 = 0x0f
//   t=4: 8 | 0x90 = 0x98   t=5: a | 0xb0 = 0xba   t=6: c | 0xd0 = 0xdc   t=7: e | 0x00 = 0x0e
// Sub-block 2, q = 11 22 33 44 55 66 77 88:
//   t=0: 1 | 2<<4 = 0x21   t=1: 3 | 4<<4 = 0x43   t=2: 5 | 6<<4 = 0x65   t=3: 7 | 8<<4 = 0x87
//   t=4: 1 | 0x20 = 0x21   t=5: 3 | 0x40 = 0x43   t=6: 5 | 0x60 = 0x65   t=7: 7 | 0x80 = 0x87
// Sub-block 3, q = 99 aa bb cc dd ee ff 10:
//   t=0: 9 | a<<4 = 0xa9   t=1: b | c<<4 = 0xcb   t=2: d | e<<4 = 0xed   t=3: f | 0<<4 = 0x0f
//   t=4: 9 | 0xa0 = 0xa9   t=5: b | 0xc0 = 0xcb   t=6: d | 0xe0 = 0xed   t=7: f | 0x10 = 0x1f
// Row n's 64 elements start at byte (n*64)/2 = 32n, so the nibble plane is those 32 bytes twice.
//
// Scales: 0x38 is exp 7, man 0 -> 1.0; 0x40 is exp 8 -> 2.0; 0xb8 has the sign set and decodes as 0x38 -> 1.0; 0x7f
// is the NaN pattern -> 0. Scale (g, n) sits at half g*2 + n: 1.0 1.0 2.0 2.0 1.0 1.0 0.0 0.0, that is
// 0x3c00 0x3c00 0x4000 0x4000 0x3c00 0x3c00 0x0000 0x0000.
// ---------------------------------------------------------------------------------------------------------------------
int check_hand_fixture(sycl::queue & q) {
    constexpr int K = 64;
    constexpr int N = 2;

    std::vector<uint8_t> stored;
    for (int n = 0; n < N; ++n) {
        const uint8_t d[4] = { 0x38, 0x40, 0xb8, 0x7f };
        stored.insert(stored.end(), d, d + 4);
        for (int i = 0; i < QK_NVFP4 / 2; ++i) {
            stored.push_back((uint8_t) (i * 0x11 + 1));
        }
    }

    static const uint8_t row_nibbles[32] = {
        0x21, 0x43, 0x65, 0x87, 0x10, 0x32, 0x54, 0x76, 0xa9, 0xcb, 0xed, 0x0f, 0x98, 0xba, 0xdc, 0x0e,
        0x21, 0x43, 0x65, 0x87, 0x21, 0x43, 0x65, 0x87, 0xa9, 0xcb, 0xed, 0x0f, 0xa9, 0xcb, 0xed, 0x1f,
    };
    static const uint16_t scales[8] = { 0x3c00, 0x3c00, 0x4000, 0x4000, 0x3c00, 0x3c00, 0x0000, 0x0000 };

    planes want;
    want.nibbles.assign(row_nibbles, row_nibbles + 32);
    want.nibbles.insert(want.nibbles.end(), row_nibbles, row_nibbles + 32);
    want.scales.assign(scales, scales + 8);
    return run_and_check(q, "hand-fixture", stored, { 0 }, K, N, want);
}

// Check 2: 3 experts, K 1152 and N 72 (not a multiple of 64), random bytes, one call with n_slots = 3. Slot s reads
// expert 2 - s, so a kernel that ignores the source table and strides from the first pointer fails.
int check_randomized(sycl::queue & q) {
    constexpr int              K           = 1152;
    constexpr int              N           = 72;
    const std::vector<uint8_t> stored      = random_experts(3, K, N, 0x4e564634u);
    const std::vector<int>     slot_expert = { 2, 1, 0 };
    return run_and_check(q, "randomized", stored, slot_expert, K, N, mirror(stored, slot_expert, K, N));
}

// Check 3: 130 slots of K 64 and N 8, crossing GGML_SYCL_MXFP4_WOQ_REPACK_MAX_SLOTS (128), so the entry launches
// twice. Slot s reads expert 129 - s, and each expert's first qs byte is its index, so every slot's planes differ.
// Before the whole-plane comparison, slots 128 and 129 are checked at their own offsets by name: a second launch that
// restarts its slot index at 0 writes them over slots 0 and 1 and leaves their own offsets as sentinel.
int check_crosses_max_slots(sycl::queue & q) {
    constexpr int K       = 64;
    constexpr int N       = 8;
    constexpr int n_slots = GGML_SYCL_MXFP4_WOQ_REPACK_MAX_SLOTS + 2;

    std::vector<uint8_t> stored       = random_experts(n_slots, K, N, 0x31333000u);
    const size_t         expert_bytes = (size_t) N * (K / QK_NVFP4) * sizeof(block_nvfp4);
    for (int e = 0; e < n_slots; ++e) {
        stored[expert_bytes * e + QK_NVFP4 / QK_NVFP4_SUB] = (uint8_t) e;
    }
    std::vector<int> slot_expert(n_slots);
    for (int s = 0; s < n_slots; ++s) {
        slot_expert[s] = n_slots - 1 - s;
    }
    const planes want = mirror(stored, slot_expert, K, N);

    const size_t     nib_slot = (size_t) N * K / 2;
    std::vector<int> high_slots;
    for (int s = GGML_SYCL_MXFP4_WOQ_REPACK_MAX_SLOTS; s < n_slots; ++s) {
        const int low = s - GGML_SYCL_MXFP4_WOQ_REPACK_MAX_SLOTS;
        if (std::memcmp(want.nibbles.data() + nib_slot * s, want.nibbles.data() + nib_slot * low, nib_slot) == 0) {
            std::printf("FAIL[crosses-max-slots]: fixture slots %d and %d are identical, the offset check is blind\n",
                        s, low);
            return 1;
        }
        high_slots.push_back(s);
    }
    if (run_and_check(q, "crosses-max-slots", stored, slot_expert, K, N, want, high_slots) != 0) {
        return 1;
    }
    std::printf("OK[crosses-max-slots]: slots %d and %d landed at their own offsets\n",
                GGML_SYCL_MXFP4_WOQ_REPACK_MAX_SLOTS, n_slots - 1);
    return 0;
}

}  // namespace

int main() {
    try {
        sycl::queue q{ sycl::gpu_selector_v };

        int rc = check_hand_fixture(q);
        if (rc != 0) {
            return rc;
        }
        rc = check_randomized(q);
        if (rc != 0) {
            return rc;
        }
        rc = check_crosses_max_slots(q);
        if (rc != 0) {
            return rc;
        }
        return 0;
    } catch (const sycl::exception & ex) {
        std::printf("SKIP: no SYCL GPU (%s)\n", ex.what());
        return LLAMA_TEST_EXIT_SKIP;
    }
}
