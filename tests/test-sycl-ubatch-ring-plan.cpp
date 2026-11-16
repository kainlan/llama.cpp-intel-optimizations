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
// The slots are sized for the packed scratch now (pp_moe_onednn_slots_at): one
// scratch row of 5760 B activation and 11520 B output times the rows 32 experts
// with 4 used can need, so the figures below are smaller than the repro's.
//
// No SYCL device is needed -- every function under test is a pure function of
// integers (atomics keyed by device id, no queue/context touched). Modeled on
// tests/test-sycl-nonfa-attn-scratch-demand.cpp (the `#if !defined(GGML_USE_SYCL)`
// skip, `g_failures`, `check()`).

#include "ggml-sycl/unified-cache.hpp"
#include "test-skip.h"

#include <cstdio>
#include <limits>

#if !defined(GGML_USE_SYCL)
int main() {
    std::fprintf(stderr, "SKIP: GGML_USE_SYCL not enabled; no gate was evaluated.\n");
    return LLAMA_TEST_EXIT_SKIP;
}
#else

using ggml_sycl::unified_cache_get_planned_pp_moe_onednn_activation_bytes_per_row;
using ggml_sycl::unified_cache_get_planned_pp_moe_onednn_n_expert;
using ggml_sycl::unified_cache_get_planned_pp_moe_onednn_n_expert_used;
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

constexpr int      kDevice          = 0;
// GPT-OSS 20B MXFP4 shape, from the repro log cited in the file header. One
// scratch row is max_k half floats in and max_n floats out.
constexpr size_t   kActPerRow       = 2880ull * 2ull;
constexpr size_t   kOutPerRow       = 2880ull * 4ull;
constexpr uint32_t kNExpert         = 32;
constexpr uint32_t kNExpertUsed     = 4;
// weight_slot_bytes is CONSTANT across n_ubatch (per-expert weight size does
// not scale with the micro-batch) -- the repro log's own figure, 134.5 MB.
constexpr size_t   kWeightSlotBytes = 141008896ull;

void test_slots_follow_the_packed_rows() {
    printf("PP MoE oneDNN ring slot sizing (unified_cache_pp_moe_onednn_slots_for_ubatch):\n");

    unified_cache_set_planned_pp_moe_onednn_row_bytes(kDevice, kActPerRow, kOutPerRow, kNExpert, kNExpertUsed);
    check(unified_cache_get_planned_pp_moe_onednn_activation_bytes_per_row(kDevice) == kActPerRow,
          "activation per-row bytes round-trip through the setter/getter");
    check(unified_cache_get_planned_pp_moe_onednn_output_bytes_per_row(kDevice) == kOutPerRow,
          "output per-row bytes round-trip through the setter/getter");
    check(unified_cache_get_planned_pp_moe_onednn_n_expert(kDevice) == kNExpert,
          "the expert count round-trips through the setter/getter");
    check(unified_cache_get_planned_pp_moe_onednn_n_expert_used(kDevice) == kNExpertUsed,
          "the used-expert count round-trips through the setter/getter");
    check(unified_cache_get_planned_pp_moe_onednn_n_expert(999999) == 0 &&
              unified_cache_get_planned_pp_moe_onednn_n_expert_used(999999) == 0,
          "an out-of-range device id reads back 0 expert counts");

    size_t act = 0, out = 0;
    check(unified_cache_pp_moe_onednn_slots_for_ubatch(kDevice, 512, &act, &out), "slots computed at n_ubatch=512");
    check(act == 35205120, "activation slot at 512 = 33.6 MB (6112 packed rows of 5760 B)");
    check(out == 70410240, "output slot at 512 = 67.1 MB (6112 packed rows of 11520 B)");

    check(
        unified_cache_pp_moe_onednn_slots_for_ubatch(kDevice, 1024, &act, &out) && act == 58798080 && out == 117596160,
        "slots at 1024 = 56.1 MB activation, 112.1 MB output (10208 packed rows)");

    check(
        unified_cache_pp_moe_onednn_slots_for_ubatch(kDevice, 2048, &act, &out) && act == 105984000 && out == 211968000,
        "slots at 2048 = 101.1 MB activation, 202.1 MB output (18400 packed rows)");

    // llama.cpp-ibj0 spec round 1 F9: this PURE FUNCTION deliberately accepts
    // n_ubatch=0 (0 rows is not an overflow, so returning slots of exactly 0
    // bytes is the correct answer for the question this function actually
    // answers -- "what size do 0 rows need"). n_ubatch=0 is not itself a
    // valid RUNTIME micro-batch, but that is a caller-level policy decision,
    // not something this sizing function should encode: the actual guard
    // against ever calling this with n_ubatch=0 from a real runtime-context
    // transaction lives in ggml_sycl_replan_pp_moe_onednn_ring() (ggml-sycl.cpp),
    // which cannot be exercised host-only (it needs a live unified_cache and
    // SYCL device queue) -- pinned instead by
    // test_replan_guards_n_ubatch_zero() in test-sycl-ubatch-ring-replan-source.py.
    check(unified_cache_pp_moe_onednn_slots_for_ubatch(kDevice, 0, &act, &out) && act == 0 && out == 0,
          "n_ubatch=0 -> both slots exactly 0 (0 rows, not a refusal: the SIZING function's own correct "
          "answer -- the n_ubatch==0 GUARD itself is caller-level and source-gated, not tested here)");
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
        unified_cache_pp_moe_onednn_slots_for_ubatch(kDevice, 2048, &act, &out) && act == 105984000 && out == 211968000,
        "descending: 2048 first (101.1 MB activation, 202.1 MB output)");
    check(
        unified_cache_pp_moe_onednn_slots_for_ubatch(kDevice, 1024, &act, &out) && act == 58798080 && out == 117596160,
        "descending: then 1024 -- shrinks to exactly what an ascending 512->1024 call produced above, not a "
        "stale/latched larger value");
    check(unified_cache_pp_moe_onednn_slots_for_ubatch(kDevice, 512, &act, &out) && act == 35205120 && out == 70410240,
          "descending: then 512 -- shrinks all the way back to the 33.6/67.1 MB load-time ring");
}

void test_dense_model_returns_false() {
    printf("Dense model (per-row bytes 0):\n");

    unified_cache_set_planned_pp_moe_onednn_row_bytes(kDevice, 0, 0, 0, 0);
    size_t act = 1, out = 1;  // poisoned, must stay untouched on a false return
    check(!unified_cache_pp_moe_onednn_slots_for_ubatch(kDevice, 512, &act, &out),
          "0 per-row bytes (dense model, no MoE PP ring planned) -> false");
    check(act == 1 && out == 1, "outputs are left untouched on a false return");

    // The packed rows depend on the expert counts, so per-row bytes without them size nothing.
    unified_cache_set_planned_pp_moe_onednn_row_bytes(kDevice, kActPerRow, kOutPerRow, 0, 0);
    check(!unified_cache_pp_moe_onednn_slots_for_ubatch(kDevice, 512, &act, &out),
          "per-row bytes with no expert counts -> false");
    unified_cache_set_planned_pp_moe_onednn_row_bytes(kDevice, kActPerRow, kOutPerRow, kNExpert, 0);
    check(!unified_cache_pp_moe_onednn_slots_for_ubatch(kDevice, 512, &act, &out),
          "per-row bytes with no experts used -> false");
    check(act == 1 && out == 1, "and the outputs stay untouched");

    // Restore the planned shape for the tests below.
    unified_cache_set_planned_pp_moe_onednn_row_bytes(kDevice, kActPerRow, kOutPerRow, kNExpert, kNExpertUsed);
}

void test_slots_overflow_returns_false() {
    printf("Overflowing per-row bytes:\n");

    // A per-row byte count that overflows size_t when multiplied by the
    // packed rows -- 4 rows of 4 picks need 1024 scratch rows, and
    // SIZE_MAX/2 * 1024 cannot be represented.
    unified_cache_set_planned_pp_moe_onednn_row_bytes(kDevice, std::numeric_limits<size_t>::max() / 2, kOutPerRow,
                                                      kNExpert, kNExpertUsed);
    size_t act = 0, out = 0;
    check(!unified_cache_pp_moe_onednn_slots_for_ubatch(kDevice, 4, &act, &out),
          "packed rows * per_row overflow -> false rather than a wrapped, small-looking size");

    unified_cache_set_planned_pp_moe_onednn_row_bytes(kDevice, kActPerRow, kOutPerRow, kNExpert, kNExpertUsed);
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

// The ring's bytes at `ub` as the ring itself sizes them: ring_depth copies of
// the weight slot (constant, does not scale with ub) and of the activation and
// output slots pp_moe_onednn_slots_at gives. 0 when the slots cannot be sized.
size_t ring_bytes_for_ubatch(uint32_t ub, uint32_t ring_depth) {
    size_t act = 0;
    size_t out = 0;
    if (!ggml_sycl::pp_moe_onednn_slots_at(ub, kNExpert, kNExpertUsed, kActPerRow, kOutPerRow, &act, &out)) {
        return 0;
    }
    return static_cast<size_t>(ring_depth) * (kWeightSlotBytes + act + out);
}

void test_largest_fitting_inverts_the_ring_size() {
    printf("Largest-fitting n_ubatch (unified_cache_largest_fitting_n_ubatch_for_pp_moe_onednn):\n");

    for (uint32_t ub : { 512u, 1024u, 2048u }) {
        const size_t   capacity = ring_bytes_for_ubatch(ub, /*ring_depth=*/1);
        const uint32_t largest  = unified_cache_largest_fitting_n_ubatch_for_pp_moe_onednn(
            capacity, kWeightSlotBytes, kActPerRow, kOutPerRow, kNExpert, kNExpertUsed, /*ring_depth=*/1);
        char what[160];
        std::snprintf(what, sizeof(what), "largest(ring(%u)) == %u: ring(%u) fits and ring(%u) does not", ub, ub,
                      largest, largest + 32);
        check(largest == ub && capacity != 0 && ring_bytes_for_ubatch(largest, 1) <= capacity &&
                  ring_bytes_for_ubatch(largest + 32, 1) > capacity,
              what);
    }

    check(unified_cache_largest_fitting_n_ubatch_for_pp_moe_onednn(kWeightSlotBytes, kWeightSlotBytes, kActPerRow,
                                                                   kOutPerRow, kNExpert, kNExpertUsed, 1) == 0,
          "capacity exactly at the weight slot (no room left for even one row) -> 0");
    check(unified_cache_largest_fitting_n_ubatch_for_pp_moe_onednn(kWeightSlotBytes - 1, kWeightSlotBytes, kActPerRow,
                                                                   kOutPerRow, kNExpert, kNExpertUsed, 1) == 0,
          "capacity below the weight slot -> 0");
    check(unified_cache_largest_fitting_n_ubatch_for_pp_moe_onednn(1ull << 40, kWeightSlotBytes, 0, 0, kNExpert,
                                                                   kNExpertUsed, 1) == 0,
          "degenerate per-row bytes (both 0) -> 0, not an unbounded n_ubatch");
    check(unified_cache_largest_fitting_n_ubatch_for_pp_moe_onednn(1ull << 40, kWeightSlotBytes, kActPerRow, kOutPerRow,
                                                                   kNExpert, kNExpertUsed, 0) == 0,
          "ring_depth=0 -> 0 (degenerate, mirrors the dense/no-ring case)");
    check(unified_cache_largest_fitting_n_ubatch_for_pp_moe_onednn(1ull << 40, kWeightSlotBytes, kActPerRow, kOutPerRow,
                                                                   0, kNExpertUsed, 1) == 0,
          "n_expert=0 -> 0: no packed rows can be sized");
    check(unified_cache_largest_fitting_n_ubatch_for_pp_moe_onednn(1ull << 40, kWeightSlotBytes, kActPerRow, kOutPerRow,
                                                                   kNExpert, 0, 1) == 0,
          "n_expert_used=0 -> 0: no packed rows can be sized");

    // Rounding: one byte short of the next 32-ubatch step must NOT round up.
    const size_t   ring_512          = ring_bytes_for_ubatch(512, 1);
    const uint32_t largest_short_one = unified_cache_largest_fitting_n_ubatch_for_pp_moe_onednn(
        ring_512 - 1, kWeightSlotBytes, kActPerRow, kOutPerRow, kNExpert, kNExpertUsed, 1);
    check(largest_short_one == 480,
          "one byte short of ring(512) rounds DOWN to 480 (the next-lower multiple of 32), not up");

    // A capacity every micro-batch fits returns the largest multiple of 32 a uint32_t holds, never a value past it
    // or one that is not a multiple of 32. With weight 0, one activation byte per row, no output, one expert of one
    // used and depth 1, the largest candidate's 4294967264 rows pack into 2^32 scratch rows, and 2^62 bytes hold them.
    const uint32_t saturating = unified_cache_largest_fitting_n_ubatch_for_pp_moe_onednn(1ull << 62, 0, 1, 0, 1, 1, 1);
    check(saturating % 32 == 0, "a saturating capacity still returns a multiple of 32");
    // The modulo check alone would also pass a regression that returns 0 (0 % 32 == 0), so pin the value:
    // (UINT32_MAX / 32) * 32 = 4294967264.
    check(saturating == 4294967264u,
          "the saturating capacity's largest-fitting value is exactly 4294967264 "
          "((UINT32_MAX / 32) * 32), not 0 or some other multiple of 32");
}

// The inverse must agree with the slot function where alignment matters. A slot whose bytes are not a multiple of
// 256 rounds up, so a micro-batch fits only when its aligned slots do, not when its packed rows times the row bytes
// do. One expert, one used, 3 bytes a row, no output, weight 0: 32 and 64 rows both pack into 64 scratch rows, 192
// bytes in a 256-byte slot; 96 rows pack into 128, 384 bytes in a 512-byte slot.
void test_largest_fitting_honours_slot_alignment() {
    printf("Largest-fitting n_ubatch with an unaligned slot:\n");

    constexpr size_t kPerRow = 3;
    constexpr size_t kSlot64 = 256;  // align256(64 * 3)
    constexpr size_t kSlot96 = 512;  // align256(128 * 3)
    check(unified_cache_largest_fitting_n_ubatch_for_pp_moe_onednn(kSlot64, 0, kPerRow, 0, 1, 1, 1) == 64,
          "unaligned slot, depth 1: 256 B (the aligned 64-scratch-row slot) fits 64 rows");
    check(unified_cache_largest_fitting_n_ubatch_for_pp_moe_onednn(kSlot64 - 1, 0, kPerRow, 0, 1, 1, 1) == 0,
          "unaligned slot, depth 1: 255 B fits no micro-batch, though 64 scratch rows of 3 B are 192 B");
    check(unified_cache_largest_fitting_n_ubatch_for_pp_moe_onednn(2 * kSlot64, 0, kPerRow, 0, 1, 1, 2) == 64,
          "unaligned slot, depth 2: 512 B (two aligned slots) fits 64 rows");
    check(unified_cache_largest_fitting_n_ubatch_for_pp_moe_onednn(2 * kSlot64 - 1, 0, kPerRow, 0, 1, 1, 2) == 0,
          "unaligned slot, depth 2: 511 B fits no micro-batch, though two 192 B slots are 384 B");
    check(unified_cache_largest_fitting_n_ubatch_for_pp_moe_onednn(kSlot96 - 1, 0, kPerRow, 0, 1, 1, 1) == 64,
          "unaligned slot, depth 1: 511 B fits 64 rows, not 96 (96 rows pack into 384 B in a 512 B slot)");
}

}  // namespace

int main() {
    test_slots_follow_the_packed_rows();
    test_slots_shrink_correctly_after_a_grow();
    test_dense_model_returns_false();
    test_slots_overflow_returns_false();
    test_n_ubatch_round_trips_through_setter_getter();
    test_largest_fitting_inverts_the_ring_size();
    test_largest_fitting_honours_slot_alignment();

    printf("%s (%d failures)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}

#endif  // GGML_USE_SYCL
