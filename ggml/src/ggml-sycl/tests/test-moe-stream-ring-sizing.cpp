// Host-only test of moe_stream_ring_size() (llama.cpp-36j0, Task 6 of
// docs/plans/2026-10-19-moe-executor-cost-model-prefill-ring.md).
//
// The ring has two regions, each holding the heaviest layer: two slots per
// host matrix (AOS copy destination, SOA reorder destination), one slot per
// largest matrix. If those bytes exceed 10 % of the resident expert set there
// is no ring. The model pins re-derive the plan's base figures.

#include "moe-stream-ring.hpp"

#include <cstdint>
#include <cstdio>

using ggml_sycl::moe_stream_ring_geometry;
using ggml_sycl::moe_stream_ring_size;

static int g_failures = 0;

static void check_geometry(const char *                     label,
                           const moe_stream_ring_geometry & got,
                           size_t                           slot_bytes,
                           uint32_t                         n_slots,
                           size_t                           bytes,
                           uint32_t                         region_slots) {
    if (got.slot_bytes == slot_bytes && got.n_slots == n_slots && got.bytes == bytes &&
        got.region_slots == region_slots) {
        std::printf("PASS: %s (slot_bytes=%zu n_slots=%u bytes=%zu region_slots=%u)\n", label, got.slot_bytes,
                    got.n_slots, got.bytes, got.region_slots);
        return;
    }
    std::printf(
        "FAIL: %s: got slot_bytes=%zu n_slots=%u bytes=%zu region_slots=%u, want slot_bytes=%zu n_slots=%u "
        "bytes=%zu region_slots=%u\n",
        label, got.slot_bytes, got.n_slots, got.bytes, got.region_slots, slot_bytes, n_slots, bytes, region_slots);
    ++g_failures;
}

int main() {
    constexpr size_t KiB = 1024;
    constexpr size_t MiB = 1024 * KiB;
    constexpr size_t GiB = 1024 * MiB;

    // Qwen3.8 on the B70: 410 MiB of host experts per layer in triplets of
    // 1741 KiB is 241 triplets, 723 matrices. One matrix is 1741 KiB / 3 =
    // 594261.3 bytes, rounded up to 594262. A region is 1446 slots, the ring
    // 2892 slots = 1718605704 bytes (1639 MiB), under 10 % of 22 GiB
    // (2362232012 bytes), so it fits.
    const uint32_t qwen_matrices = 3 * static_cast<uint32_t>((410 * MiB) / (1741 * KiB));
    const size_t   qwen_matrix   = (1741 * KiB + 2) / 3;
    check_geometry("Qwen/B70 fits", moe_stream_ring_size(qwen_matrices, qwen_matrix, 22 * GiB), 594262, 2892,
                   1718605704ull, 1446);

    // GPT-OSS 20B on the B50 at default context: 712 MB of host experts per
    // layer in triplets of 37.5 MB is 19 triplets, 57 matrices of 12.5 MB. A
    // region is 114 slots, the ring 228 slots = 2850 MB, over 10 % of ~12 GB
    // (1.2 GB). No ring until the loan grows the budget: RING_UNAVAILABLE.
    check_geometry("GPT-OSS/B50 does not fit", moe_stream_ring_size(57, 12500000, 12000000000ull), 12500000, 0, 0, 0);

    // 10 matrices of 1000 bytes: regions of 20 slots, 40 slots, 40000 bytes.
    check_geometry("exactly at the 10 % cap", moe_stream_ring_size(10, 1000, 400000), 1000, 40, 40000, 20);
    check_geometry("one slot over the cap", moe_stream_ring_size(10, 1000, 390000), 1000, 0, 0, 0);
    check_geometry("one byte over the cap", moe_stream_ring_size(10, 1000, 399999), 1000, 0, 0, 0);

    // Degenerate inputs give no ring rather than a zero-sized or overflowed one.
    check_geometry("no host matrices", moe_stream_ring_size(0, 1000, 400000), 1000, 0, 0, 0);
    check_geometry("matrix_bytes_max 0", moe_stream_ring_size(10, 0, 400000), 0, 0, 0, 0);
    check_geometry("slot count overflows uint32", moe_stream_ring_size(UINT32_MAX, 1, SIZE_MAX), 1, 0, 0, 0);

    if (g_failures != 0) {
        std::printf("FAILED: %d case(s)\n", g_failures);
        return 1;
    }
    std::printf("ALL PASSED\n");
    return 0;
}
