// Host-only test of moe_stream_ring_size() (llama.cpp-36j0, Task 6 of
// docs/plans/2026-10-19-moe-executor-cost-model-prefill-ring.md).
//
// The ring has two regions, each holding one executor window's slots
// (moe_stream_plan_index::window_slots), not a whole layer's: two per host
// matrix staged as SOA (AOS copy destination, SOA reorder destination), one
// per matrix staged AOS-only (c-hmey), each slot the largest matrix rounded up
// to 256 bytes. The caller counts the slots, so every call here passes a
// region's slot count. If those bytes exceed 10 % of what the resident expert
// set keeps once the ring's own bytes are lent out of it (bytes > resident /
// 11) there is no ring. The caller passes the expert set with any loan to the
// ring counted back, so the figure does not move with a loan; that half is
// test-moe-stream-plan's (llama.cpp-7999). The model pins re-derive the plan's
// base figures.
//
// It also checks the short reserve's WEIGHT zone line: the ranking of a zone's tags (zone_tag_census_rank) and the
// line moe_stream_ring_weight_zone_line() builds from a census, both pure. The census itself reads the live registry
// and is the lead's device run's to show.

#include "moe-stream-ring.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using ggml_sycl::moe_stream_ring_geometry;
using ggml_sycl::moe_stream_ring_size;
using ggml_sycl::zone_tag_census;
using ggml_sycl::zone_tag_total;

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

static void check_text(const char * label, const std::string & got, const std::string & want) {
    if (got == want) {
        std::printf("PASS: %s\n", label);
        return;
    }
    std::printf("FAIL: %s:\n  got  \"%s\"\n  want \"%s\"\n", label, got.c_str(), want.c_str());
    ++g_failures;
}

static std::string tag_order(const std::vector<zone_tag_total> & tags) {
    std::string out;
    for (const zone_tag_total & t : tags) {
        out += out.empty() ? "" : ",";
        out += t.tag;
    }
    return out;
}

// Five tags with every tie the rank breaks: two of 300000 bytes (more allocations first) and two of 10 bytes with one
// allocation each (tag order). They arrive in neither byte nor name order, as a hash map hands them over.
static std::vector<zone_tag_total> census_tags() {
    return {
        { "b/other",                          1, 10     },
        { "scratch_pool/compute",             1, 300000 },
        { "exact_zone_allocate_owner/weight", 5, 400000 },
        { "a/other",                          1, 10     },
        { "kv_cache/kv",                      2, 300000 },
    };
}

static void check_weight_zone_census() {
    {
        std::vector<zone_tag_total> tags = census_tags();
        ggml_sycl::zone_tag_census_rank(tags, 12);
        check_text("census ranks largest bytes first, then more allocations, then the tag", tag_order(tags),
                   "exact_zone_allocate_owner/weight,kv_cache/kv,scratch_pool/compute,a/other,b/other");
    }
    {
        std::vector<zone_tag_total> tags = census_tags();
        ggml_sycl::zone_tag_census_rank(tags, 2);
        check_text("census keeps only the top tags", tag_order(tags), "exact_zone_allocate_owner/weight,kv_cache/kv");
    }
    {
        std::vector<zone_tag_total> tags;
        ggml_sycl::zone_tag_census_rank(tags, 12);
        check_text("census of no tags stays empty", tag_order(tags), "");
    }

    // 1000000 bytes, 250000 free: 750000 used. The census attributes 1000020 bytes over 10 allocations in five tags,
    // more than the allocator reports used, so unattributed goes negative instead of wrapping. Only the top four are
    // printed; the totals still count the fifth.
    zone_tag_census census;
    census.capacity     = 1000000;
    census.available    = 250000;
    census.largest_free = 100000;
    census.tags         = census_tags();
    check_text("short reserve line names the zone's figures and its top tags",
               ggml_sycl::moe_stream_ring_weight_zone_line(0, census, 3, 256, 4),
               "[MOE-STREAM-RING] WEIGHT zone at the short reserve: device=0 ring_slots_held=3 slot_bytes=256 "
               "size=1000000 free=250000 largest_free=100000 used=750000 attributed=1000020 unattributed=-250020 "
               "allocations=10 tags=5; top 4 by bytes: exact_zone_allocate_owner/weight count=5 bytes=400000, "
               "kv_cache/kv count=2 bytes=300000, scratch_pool/compute count=1 bytes=300000, a/other count=1 "
               "bytes=10");
    check_text("the line ranks a copy: the census keeps its order", tag_order(census.tags),
               "b/other,scratch_pool/compute,exact_zone_allocate_owner/weight,a/other,kv_cache/kv");

    // A zone that reports more free than its size uses 0, not a wrapped size; a census that found nothing leaves all
    // of the used bytes unattributed.
    zone_tag_census empty;
    empty.available = 4096;
    check_text("short reserve line with no tags", ggml_sycl::moe_stream_ring_weight_zone_line(1, empty, 0, 512, 12),
               "[MOE-STREAM-RING] WEIGHT zone at the short reserve: device=1 ring_slots_held=0 slot_bytes=512 size=0 "
               "free=4096 largest_free=0 used=0 attributed=0 unattributed=0 allocations=0 tags=0; top 0 by bytes: "
               "none");
    zone_tag_census unseen;
    unseen.capacity  = 8192;
    unseen.available = 1024;
    check_text("short reserve line shows what the census did not attribute",
               ggml_sycl::moe_stream_ring_weight_zone_line(0, unseen, 2, 512, 12),
               "[MOE-STREAM-RING] WEIGHT zone at the short reserve: device=0 ring_slots_held=2 slot_bytes=512 "
               "size=8192 free=1024 largest_free=0 used=7168 attributed=0 unattributed=7168 allocations=0 tags=0; top "
               "0 by bytes: none");
}

int main() {
    constexpr size_t KiB = 1024;
    constexpr size_t MiB = 1024 * KiB;
    constexpr size_t GiB = 1024 * MiB;

    // Qwen3.8 on the B70: 410 MiB of host experts per layer in triplets of
    // 1741 KiB is 241 triplets, 723 matrices. One matrix is 1741 KiB / 3 =
    // 594261.3 bytes, rounded up to 594262 and then to the 256-byte slot
    // alignment, 594432. A region is 1446 slots, the ring 2892 slots =
    // 1719097344 bytes (1639 MiB), under 22 GiB / 11 (2147483648 bytes), so
    // it fits.
    const uint32_t qwen_matrices = 3 * static_cast<uint32_t>((410 * MiB) / (1741 * KiB));
    const size_t   qwen_matrix   = (1741 * KiB + 2) / 3;
    check_geometry("Qwen/B70 fits", moe_stream_ring_size(2 * qwen_matrices, qwen_matrix, 22 * GiB), 594432, 2892,
                   1719097344ull, 1446);
    // The same layer staged AOS-only, one slot per matrix: half the ring.
    check_geometry("Qwen/B70 AOS-only fits", moe_stream_ring_size(qwen_matrices, qwen_matrix, 22 * GiB), 594432, 1446,
                   859548672ull, 723);

    // GPT-OSS 20B on the B50 at default context: 712 MB of host experts per
    // layer in triplets of 37.5 MB is 19 triplets, 57 matrices of 12.5 MB. A
    // region is 114 slots of 12500224 bytes (12.5 MB rounded up to 256), the
    // ring 228 slots = 2850051072 bytes, over ~12 GB / 11
    // (1.09 GB). No ring until the loan grows the budget: RING_UNAVAILABLE, with
    // every field 0.
    check_geometry("GPT-OSS/B50 does not fit", moe_stream_ring_size(2 * 57, 12500000, 12000000000ull), 0, 0, 0, 0);

    // 10 matrices of 1024 bytes (already aligned) staged as SOA: regions of 20
    // slots, 40 slots, 40960 bytes; the cap holds them at 11 x 40960 = 450560.
    check_geometry("exactly at the post-loan cap", moe_stream_ring_size(20, 1024, 450560), 1024, 40, 40960, 20);
    check_geometry("one slot over the cap", moe_stream_ring_size(20, 1024, 439296), 0, 0, 0, 0);
    check_geometry("one byte over the cap", moe_stream_ring_size(20, 1024, 450559), 0, 0, 0, 0);

    // An odd region, which only AOS-only matrices make: 3 matrices at one
    // slot each, and a mixed layer of 2 SOA matrices and 3 AOS-only ones (7
    // slots). Still two equal regions (ruling D).
    check_geometry("AOS-only region of 3 slots", moe_stream_ring_size(3, 1024, 67584), 1024, 6, 6144, 3);
    check_geometry("AOS-only region one byte over the cap", moe_stream_ring_size(3, 1024, 67583), 0, 0, 0, 0);
    check_geometry("mixed region of 7 slots", moe_stream_ring_size(7, 1000, 157696), 1024, 14, 14336, 7);

    // An unaligned matrix is rounded up to the slot alignment, and the cap counts
    // the padding: 1000 bytes becomes 1024, so 40000 bytes of resident / 11 no
    // longer holds the ring that 1000-byte slots would have fitted.
    check_geometry("unaligned matrix rounds up", moe_stream_ring_size(20, 1000, 450560), 1024, 40, 40960, 20);
    check_geometry("padding counts against the cap", moe_stream_ring_size(20, 1000, 440000), 0, 0, 0, 0);
    check_geometry("already aligned matrix is unchanged", moe_stream_ring_size(20, 256, 112640), 256, 40, 10240, 20);

    // Degenerate inputs give no ring rather than a zero-sized or overflowed one.
    // Every no-ring answer is the default geometry: every field is 0.
    check_geometry("no host matrices", moe_stream_ring_size(0, 1000, 400000), 0, 0, 0, 0);
    check_geometry("matrix_bytes_max 0", moe_stream_ring_size(20, 0, 400000), 0, 0, 0, 0);
    // Two regions of UINT32_MAX slots, and of one slot past half of it.
    check_geometry("slot count overflows uint32", moe_stream_ring_size(UINT32_MAX, 1, SIZE_MAX), 0, 0, 0, 0);
    check_geometry("slot count just overflows uint32", moe_stream_ring_size((UINT32_MAX / 2) + 1, 1, SIZE_MAX), 0, 0, 0,
                   0);
    // 2^30 slots fit in uint32, but 2^30 slots of 2^34 + 256 bytes overflow size_t.
    check_geometry("slot bytes overflow size_t", moe_stream_ring_size(1u << 29, (size_t(1) << 34) + 1, SIZE_MAX), 0, 0,
                   0, 0);
    check_geometry("rounding up overflows size_t", moe_stream_ring_size(1, SIZE_MAX, SIZE_MAX), 0, 0, 0, 0);

    // Task 7 (2026-10-24 plan, llama.cpp-7999): the cap leaves 10 % of what remains once the ring's own bytes are
    // taken out, b <= (R - b) / 10, which is b <= R / 11.
    {
        const size_t slot = 256;
        // 2 regions x 4 slots x 256 = 2048 bytes. Under / 10: resident >= 20480. Under / 11: resident >= 22528.
        check_geometry("eleventh: just inside the cap", moe_stream_ring_size(4, slot, 22528), slot, 8, 2048, 4);
        check_geometry("eleventh: inside a tenth, outside an eleventh", moe_stream_ring_size(4, slot, 22527), 0, 0, 0,
                       0);
    }

    check_weight_zone_census();

    if (g_failures != 0) {
        std::printf("FAILED: %d case(s)\n", g_failures);
        return 1;
    }
    std::printf("ALL PASSED\n");
    return 0;
}
