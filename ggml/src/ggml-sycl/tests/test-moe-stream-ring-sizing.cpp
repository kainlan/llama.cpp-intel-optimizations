// Host-only test of moe_stream_ring_size() (llama.cpp-36j0, Task 6 of
// docs/plans/2026-10-19-moe-executor-cost-model-prefill-ring.md).
//
// The ring has two regions, each holding the heaviest layer: two slots per
// host matrix (AOS copy destination, SOA reorder destination), one slot per
// largest matrix, each rounded up to 256 bytes. If those bytes exceed 10 % of the resident expert set there
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
    // 594261.3 bytes, rounded up to 594262 and then to the 256-byte slot
    // alignment, 594432. A region is 1446 slots, the ring 2892 slots =
    // 1719097344 bytes (1639 MiB), under 10 % of 22 GiB (2362232012 bytes), so
    // it fits.
    const uint32_t qwen_matrices = 3 * static_cast<uint32_t>((410 * MiB) / (1741 * KiB));
    const size_t   qwen_matrix   = (1741 * KiB + 2) / 3;
    check_geometry("Qwen/B70 fits", moe_stream_ring_size(qwen_matrices, qwen_matrix, 22 * GiB), 594432, 2892,
                   1719097344ull, 1446);

    // GPT-OSS 20B on the B50 at default context: 712 MB of host experts per
    // layer in triplets of 37.5 MB is 19 triplets, 57 matrices of 12.5 MB. A
    // region is 114 slots of 12500224 bytes (12.5 MB rounded up to 256), the
    // ring 228 slots = 2850051072 bytes, over 10 % of ~12 GB
    // (1.2 GB). No ring until the loan grows the budget: RING_UNAVAILABLE, with
    // every field 0.
    check_geometry("GPT-OSS/B50 does not fit", moe_stream_ring_size(57, 12500000, 12000000000ull), 0, 0, 0, 0);

    // 10 matrices of 1024 bytes (already aligned): regions of 20 slots, 40
    // slots, 40960 bytes.
    check_geometry("exactly at the 10 % cap", moe_stream_ring_size(10, 1024, 409600), 1024, 40, 40960, 20);
    check_geometry("one slot over the cap", moe_stream_ring_size(10, 1024, 399360), 0, 0, 0, 0);
    check_geometry("one byte over the cap", moe_stream_ring_size(10, 1024, 409599), 0, 0, 0, 0);

    // An unaligned matrix is rounded up to the slot alignment, and the cap counts
    // the padding: 1000 bytes becomes 1024, so 40000 bytes of resident / 10 no
    // longer holds the ring that 1000-byte slots would have fitted.
    check_geometry("unaligned matrix rounds up", moe_stream_ring_size(10, 1000, 409600), 1024, 40, 40960, 20);
    check_geometry("padding counts against the cap", moe_stream_ring_size(10, 1000, 400000), 0, 0, 0, 0);
    check_geometry("already aligned matrix is unchanged", moe_stream_ring_size(10, 256, 102400), 256, 40, 10240, 20);

    // Degenerate inputs give no ring rather than a zero-sized or overflowed one.
    // Every no-ring answer is the default geometry: every field is 0.
    check_geometry("no host matrices", moe_stream_ring_size(0, 1000, 400000), 0, 0, 0, 0);
    check_geometry("matrix_bytes_max 0", moe_stream_ring_size(10, 0, 400000), 0, 0, 0, 0);
    check_geometry("slot count overflows uint32", moe_stream_ring_size(UINT32_MAX, 1, SIZE_MAX), 0, 0, 0, 0);
    // 2^30 slots fit in uint32, but 2^30 slots of 2^34 + 256 bytes overflow size_t.
    check_geometry("slot bytes overflow size_t", moe_stream_ring_size(1u << 28, (size_t(1) << 34) + 1, SIZE_MAX), 0, 0,
                   0, 0);
    check_geometry("rounding up overflows size_t", moe_stream_ring_size(1, SIZE_MAX, SIZE_MAX), 0, 0, 0, 0);

    if (g_failures != 0) {
        std::printf("FAILED: %d case(s)\n", g_failures);
        return 1;
    }
    std::printf("ALL PASSED\n");
    return 0;
}
