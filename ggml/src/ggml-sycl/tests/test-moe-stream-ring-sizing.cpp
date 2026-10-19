// Host-only test of moe_stream_ring_size() (llama.cpp-36j0, Task 6 of
// docs/plans/2026-10-19-moe-executor-cost-model-prefill-ring.md).
//
// The ring's byte budget is min(ring_layers x one layer's host bytes, 10 % of
// the resident expert set), cut into whole slots of the largest host triplet.
// The pins are the two canonical models' figures from the plan.

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
                           size_t                           bytes) {
    if (got.slot_bytes == slot_bytes && got.n_slots == n_slots && got.bytes == bytes) {
        std::printf("PASS: %s (slot_bytes=%zu n_slots=%u bytes=%zu)\n", label, got.slot_bytes, got.n_slots, got.bytes);
        return;
    }
    std::printf("FAIL: %s: got slot_bytes=%zu n_slots=%u bytes=%zu, want slot_bytes=%zu n_slots=%u bytes=%zu\n", label,
                got.slot_bytes, got.n_slots, got.bytes, slot_bytes, n_slots, bytes);
    ++g_failures;
}

int main() {
    constexpr size_t KiB = 1024;
    constexpr size_t MiB = 1024 * KiB;
    constexpr size_t GiB = 1024 * MiB;

    // Qwen3.8 on the B70: 2 x 410 MiB = 820 MiB is under 10 % of 22 GiB
    // (2252.8 MiB), so the per-layer term decides: 820 MiB / 1741 KiB = 482.3.
    const size_t qwen_layer    = 410 * MiB;
    const size_t qwen_resident = 22 * GiB;
    const size_t qwen_triplet  = 1741 * KiB;
    check_geometry("Qwen/B70 two layers", moe_stream_ring_size(qwen_layer, qwen_resident, qwen_triplet, 2),
                   qwen_triplet, 482, 482 * qwen_triplet);

    // ring_layers 1 halves it: 410 MiB / 1741 KiB = 241.2, and one layer's
    // slots are those 241 whole slots, so the one-layer ring is not refused.
    check_geometry("Qwen/B70 one layer is half", moe_stream_ring_size(qwen_layer, qwen_resident, qwen_triplet, 1),
                   qwen_triplet, 241, 241 * qwen_triplet);

    // GPT-OSS 20B on the B50 at default context: 2 x 712 MB = 1424 MB is over
    // 10 % of ~12 GB, so the resident cap decides: 1.2 GB / 37.5 MB = 32.
    const size_t gptoss_layer    = 712000000;
    const size_t gptoss_resident = 12000000000ull;
    const size_t gptoss_triplet  = 37500000;
    check_geometry("GPT-OSS/B50 two layers (resident cap)",
                   moe_stream_ring_size(gptoss_layer, gptoss_resident, gptoss_triplet, 2), gptoss_triplet, 32,
                   1200000000ull);

    // One layer is under the cap, so the layer term decides again: 712 / 37.5 = 18.99.
    check_geometry("GPT-OSS/B50 one layer", moe_stream_ring_size(gptoss_layer, gptoss_resident, gptoss_triplet, 1),
                   gptoss_triplet, 18, 18 * gptoss_triplet);

    // A resident set too small for one layer: 10 % of 2 GiB is 204.8 MiB, 120
    // slots, fewer than one layer's 241, so there is no ring.
    check_geometry("resident set too small for one layer", moe_stream_ring_size(qwen_layer, 2 * GiB, qwen_triplet, 2),
                   qwen_triplet, 0, 0);

    // Degenerate inputs give no ring rather than a division by zero.
    check_geometry("ring_layers 0", moe_stream_ring_size(qwen_layer, qwen_resident, qwen_triplet, 0), qwen_triplet, 0,
                   0);
    check_geometry("triplet_bytes_max 0", moe_stream_ring_size(qwen_layer, qwen_resident, 0, 2), 0, 0, 0);

    if (g_failures != 0) {
        std::printf("FAILED: %d case(s)\n", g_failures);
        return 1;
    }
    std::printf("ALL PASSED\n");
    return 0;
}
