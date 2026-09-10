// Host-only gate for llama.cpp-ibj0: the pure functions behind re-planning
// the PP MoE oneDNN scratch ring for the RUNTIME n_ubatch instead of the
// load-time default the loader always plans for (src/llama-model.cpp hardcodes
// inventory.n_ubatch = 512).
//
// THE REPRO SHAPE THIS IS FIT TO (scratchpad ibj0-repro-ub1024-v.log, B50,
// GPT-OSS 20B MXFP4, master e1253b403): `llama-bench -m gpt-oss-20b-mxfp4.gguf
// -p 2048 -n 0 -ub 1024 -r 1 -v`. The load-time plan (n_ubatch=512) recorded
// `[SYCL-PLAN] PP MoE oneDNN scratch ring: depth=1 total=404.5 MB
// weight_slot=134.5 MB activation_slot=90.0 MB output_slot=180.0 MB`; the
// first -ub 1024 prefill then refused with `reason=activation-cap
// required=[141008896,153354240,306708480] planned_cap=[141008896,94371840,
// 188743680]` -- the 512-row plan (94371840 B = 90.0 MB activation,
// 188743680 B = 180.0 MB output) is too small for the 1024-row executor
// demand (153354240 B, 306708480 B), and llama_decode returned -3 with no
// diagnostic at default verbosity. GPT-OSS 20B MXFP4: n_expert=32,
// max_k=max_n=2880, weight_slot_bytes=141008896 (134.5 MB, constant --
// weight sizing does not scale with n_ubatch).
//
// No SYCL device is needed -- every function under test is a pure function of
// integers (atomics keyed by device id, no queue/context touched). Modeled on
// tests/test-sycl-nonfa-attn-scratch-demand.cpp (the `#if !defined(GGML_USE_SYCL)`
// skip, `g_failures`, `check()`).

#include "ggml-sycl/unified-cache.hpp"
#include "test-skip.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#if !defined(GGML_USE_SYCL)
int main() {
    std::fprintf(stderr, "SKIP: GGML_USE_SYCL not enabled; no gate was evaluated.\n");
    return LLAMA_TEST_EXIT_SKIP;
}
#else

using ggml_sycl::unified_cache_get_planned_pp_moe_onednn_activation_bytes_per_row;
using ggml_sycl::unified_cache_get_planned_pp_moe_onednn_n_ubatch;
using ggml_sycl::unified_cache_get_planned_pp_moe_onednn_output_bytes_per_row;
using ggml_sycl::unified_cache_largest_fitting_n_ubatch_for_pp_moe_onednn;
using ggml_sycl::unified_cache_pp_moe_onednn_slots_for_ubatch;
using ggml_sycl::unified_cache_set_planned_pp_moe_onednn_n_ubatch;
using ggml_sycl::unified_cache_set_planned_pp_moe_onednn_row_bytes;

namespace {

int g_failures = 0;

void check(bool ok, const char * what) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) {
        g_failures++;
    }
}

constexpr int    kDevice          = 0;
// GPT-OSS 20B MXFP4 shape, from the repro log cited in the file header.
constexpr size_t kActPerRow       = 32ull * 2880ull * 2ull;  // n_expert * max_k * sizeof(f16)
constexpr size_t kOutPerRow       = 32ull * 2880ull * 4ull;  // n_expert * max_n * sizeof(float)
// weight_slot_bytes is CONSTANT across n_ubatch (per-expert weight size does
// not scale with the micro-batch) -- the repro log's own figure, 134.5 MB.
constexpr size_t kWeightSlotBytes = 141008896ull;

void test_slots_reproduce_loader_formula() {
    printf("PP MoE oneDNN ring slot sizing (unified_cache_pp_moe_onednn_slots_for_ubatch):\n");

    unified_cache_set_planned_pp_moe_onednn_row_bytes(kDevice, kActPerRow, kOutPerRow);
    check(unified_cache_get_planned_pp_moe_onednn_activation_bytes_per_row(kDevice) == kActPerRow,
          "activation per-row bytes round-trip through the setter/getter");
    check(unified_cache_get_planned_pp_moe_onednn_output_bytes_per_row(kDevice) == kOutPerRow,
          "output per-row bytes round-trip through the setter/getter");

    size_t act = 0, out = 0;
    check(unified_cache_pp_moe_onednn_slots_for_ubatch(kDevice, 512, &act, &out), "slots computed at n_ubatch=512");
    check(act == 94371840, "activation slot at 512 = 90.0 MB (matches the loader's 512-row plan)");
    check(out == 188743680, "output slot at 512 = 180.0 MB (matches the loader's 512-row plan)");

    check(
        unified_cache_pp_moe_onednn_slots_for_ubatch(kDevice, 1024, &act, &out) && act == 188743680 && out == 377487360,
        "slots at 1024 double (align256(1024*32*2880*2)=180.0 MB, align256(1024*32*2880*4)=360.0 MB)");

    check(
        unified_cache_pp_moe_onednn_slots_for_ubatch(kDevice, 2048, &act, &out) && act == 377487360 && out == 754974720,
        "slots at 2048 double again (360.0 MB activation, 720.0 MB output)");

    check(unified_cache_pp_moe_onednn_slots_for_ubatch(kDevice, 0, &act, &out) && act == 0 && out == 0,
          "n_ubatch=0 -> both slots exactly 0 (0 rows, not a refusal: the per-row bytes are still planned)");
}

void test_slots_shrink_correctly_after_a_grow() {
    printf("Shrinking after a grow (llama.cpp-nphx c-wgxn: the ladder trial can settle smaller):\n");

    // Direction independence at the SIZE-COMPUTATION layer: querying 2048
    // then 1024 then 512 (descending) must produce the exact same numbers as
    // querying them ascending above -- unified_cache_pp_moe_onednn_slots_for_ubatch()
    // is a pure function of (device's planned per-row bytes, n_ubatch), with
    // no memory of prior calls. The ACTUAL ring release+reserve sequence
    // (ggml_sycl_replan_pp_moe_onednn_ring() in ggml-sycl.cpp, calling
    // unified_cache::release_pp_moe_onednn_scratch_ring() then
    // reserve_pp_moe_onednn_scratch()) touches a live SYCL device queue and
    // cannot be exercised host-only; this test covers the sizing half of the
    // shrink path, the half that IS a pure function.
    size_t act = 0, out = 0;
    check(
        unified_cache_pp_moe_onednn_slots_for_ubatch(kDevice, 2048, &act, &out) && act == 377487360 && out == 754974720,
        "descending: 2048 first (360.0 MB activation, 720.0 MB output)");
    check(
        unified_cache_pp_moe_onednn_slots_for_ubatch(kDevice, 1024, &act, &out) && act == 188743680 && out == 377487360,
        "descending: then 1024 -- shrinks to exactly what an ascending 512->1024 call produced above, not a "
        "stale/latched larger value");
    check(unified_cache_pp_moe_onednn_slots_for_ubatch(kDevice, 512, &act, &out) && act == 94371840 && out == 188743680,
          "descending: then 512 -- shrinks all the way back to the loader's own 90.0/180.0 MB plan");
}

void test_dense_model_returns_false() {
    printf("Dense model (per-row bytes 0):\n");

    unified_cache_set_planned_pp_moe_onednn_row_bytes(kDevice, 0, 0);
    size_t act = 1, out = 1;  // poisoned, must stay untouched on a false return
    check(!unified_cache_pp_moe_onednn_slots_for_ubatch(kDevice, 512, &act, &out),
          "0 per-row bytes (dense model, no MoE PP ring planned) -> false");
    check(act == 1 && out == 1, "outputs are left untouched on a false return");

    // Restore the planned shape for the tests below.
    unified_cache_set_planned_pp_moe_onednn_row_bytes(kDevice, kActPerRow, kOutPerRow);
}

void test_slots_overflow_returns_false() {
    printf("Overflowing per-row bytes:\n");

    // A per-row byte count that overflows size_t when multiplied by a large
    // n_ubatch -- SIZE_MAX/2 * 4 cannot be represented.
    unified_cache_set_planned_pp_moe_onednn_row_bytes(kDevice, std::numeric_limits<size_t>::max() / 2, kOutPerRow);
    size_t act = 0, out = 0;
    check(!unified_cache_pp_moe_onednn_slots_for_ubatch(kDevice, 4, &act, &out),
          "n_ubatch * per_row overflow -> false rather than a wrapped, small-looking size");

    unified_cache_set_planned_pp_moe_onednn_row_bytes(kDevice, kActPerRow, kOutPerRow);
}

void test_n_ubatch_round_trips_through_setter_getter() {
    printf("Planned ring n_ubatch bookkeeping:\n");

    check(unified_cache_get_planned_pp_moe_onednn_n_ubatch(999999) == 0,
          "an out-of-range device id reads back 0, not garbage");
    unified_cache_set_planned_pp_moe_onednn_n_ubatch(kDevice, 1024);
    check(unified_cache_get_planned_pp_moe_onednn_n_ubatch(kDevice) == 1024,
          "planned n_ubatch round-trips through the setter/getter");
    unified_cache_set_planned_pp_moe_onednn_n_ubatch(kDevice, 512);
    check(unified_cache_get_planned_pp_moe_onednn_n_ubatch(kDevice) == 512, "and can be set again (not latched)");
}

// capacity(ub) built the same way the ring itself sizes its total: weight
// slot(s) (constant, does not scale with ub) plus ring_depth copies of the
// per-ub activation+output slots. act_per_row/out_per_row are already
// multiples of 256 for the GPT-OSS shape (184320 and 368640), so
// n_ubatch * per_row needs no additional alignment rounding for this
// round-trip to be exact.
size_t capacity_for_ubatch(uint32_t ub, uint32_t ring_depth) {
    return static_cast<size_t>(ring_depth) * (kWeightSlotBytes + static_cast<size_t>(ub) * (kActPerRow + kOutPerRow));
}

void test_largest_fitting_inverts_the_capacity_formula() {
    printf("Largest-fitting n_ubatch (unified_cache_largest_fitting_n_ubatch_for_pp_moe_onednn):\n");

    for (uint32_t ub : { 512u, 1024u, 2048u }) {
        const size_t   capacity = capacity_for_ubatch(ub, /*ring_depth=*/1);
        const uint32_t largest  = unified_cache_largest_fitting_n_ubatch_for_pp_moe_onednn(
            capacity, kWeightSlotBytes, kActPerRow, kOutPerRow, /*ring_depth=*/1);
        char what[160];
        std::snprintf(what, sizeof(what), "largest(capacity(%u)) == %u (round-trips exactly)", ub, ub);
        check(largest == ub, what);
    }

    check(unified_cache_largest_fitting_n_ubatch_for_pp_moe_onednn(kWeightSlotBytes, kWeightSlotBytes, kActPerRow,
                                                                   kOutPerRow, 1) == 0,
          "capacity exactly at the weight slot (no room left for even one row) -> 0");
    check(unified_cache_largest_fitting_n_ubatch_for_pp_moe_onednn(kWeightSlotBytes - 1, kWeightSlotBytes, kActPerRow,
                                                                   kOutPerRow, 1) == 0,
          "capacity below the weight slot -> 0");
    check(unified_cache_largest_fitting_n_ubatch_for_pp_moe_onednn(1ull << 40, kWeightSlotBytes, 0, 0, 1) == 0,
          "degenerate per-row bytes (both 0) -> 0, not an unbounded n_ubatch");
    check(unified_cache_largest_fitting_n_ubatch_for_pp_moe_onednn(1ull << 40, kWeightSlotBytes, kActPerRow, kOutPerRow,
                                                                   0) == 0,
          "ring_depth=0 -> 0 (degenerate, mirrors the dense/no-ring case)");

    // Rounding: one byte short of the next 32-ubatch step must NOT round up.
    const size_t   capacity_512      = capacity_for_ubatch(512, 1);
    const uint32_t largest_short_one = unified_cache_largest_fitting_n_ubatch_for_pp_moe_onednn(
        capacity_512 - 1, kWeightSlotBytes, kActPerRow, kOutPerRow, 1);
    check(largest_short_one == 480,
          "one byte short of capacity(512) rounds DOWN to 480 (the next-lower multiple of 32), not up");
}

}  // namespace

int main() {
    test_slots_reproduce_loader_formula();
    test_slots_shrink_correctly_after_a_grow();
    test_dense_model_returns_false();
    test_slots_overflow_returns_false();
    test_n_ubatch_round_trips_through_setter_getter();
    test_largest_fitting_inverts_the_capacity_formula();

    printf("%s (%d failures)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}

#endif  // GGML_USE_SYCL
