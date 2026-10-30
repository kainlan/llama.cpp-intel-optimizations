// Host-only gate for the PP MoE oneDNN scratch ring's runtime admission
// (pp_moe_onednn_admit_ring(), moe-scratch-admission.hpp): where the ring for a
// runtime n_ubatch goes, and whether it is admitted, once the runtime-context
// transaction has already admitted KV.
//
// The rule under test: the weight slot stays in the RUNTIME zone; each
// ubatch-scaled slot kind (activation, output) goes to the RUNTIME zone if it
// still fits there, larger kind first, and otherwise to the shared KV/weight
// zone, where it may only use what is left after the admitted KV and the
// compute-buffer reserve. KV is admitted first, so the ring can never be the
// reason KV leaves VRAM.
//
// THE SHAPE THIS IS FIT TO (B50, GPT-OSS 20B MXFP4): weight slot 134.5 MB,
// per row 184320 B activation + 368640 B output, ring depth 1; with the old
// ring released the 512 MB RUNTIME zone has 505.3 MB available
// (llama.cpp-ibj0 comment c-f68c), 370.8 MB of it after the weight slot. So
// -ub 512 fits the RUNTIME zone outright, -ub 1024 puts its 180 MB activation
// slot in the KV zone, -ub 2048 its 720 MB output slot, -ub 4096 both.
//
// Host-only: moe-scratch-admission.cpp carries no SYCL or backend dependency,
// so this target compiles it directly, like test-moe-scratch-admission.

#include "moe-scratch-admission.hpp"

#include <cstdio>
#include <limits>
#include <string>

using ggml_sycl::pp_moe_onednn_admit_ring;
using ggml_sycl::pp_moe_onednn_explicit_ubatch_action;
using ggml_sycl::pp_moe_onednn_explicit_ubatch_action_name;
using ggml_sycl::pp_moe_onednn_explicit_ubatch_decide;
using ggml_sycl::pp_moe_onednn_explicit_ubatch_decision;
using ggml_sycl::pp_moe_onednn_explicit_ubatch_inputs;
using ggml_sycl::pp_moe_onednn_explicit_ubatch_plan_room;
using ggml_sycl::pp_moe_onednn_explicit_ubatch_room;
using ggml_sycl::pp_moe_onednn_ring_admission;
using ggml_sycl::pp_moe_onednn_ring_admission_inputs;

namespace {

int g_failures = 0;

void check(bool ok, const char * what) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) {
        g_failures++;
    }
}

constexpr size_t kMiB             = 1024ull * 1024ull;
constexpr size_t kMax             = std::numeric_limits<size_t>::max();
// GPT-OSS 20B MXFP4: n_expert=32, max_k=max_n=2880, WOQ weight slot 134.5 MB.
constexpr size_t kActPerRow       = 32ull * 2880ull * 2ull;
constexpr size_t kOutPerRow       = 32ull * 2880ull * 4ull;
constexpr size_t kWeightSlotBytes = 141008896ull;
// B50 RUNTIME zone available with the old ring released: 505.3 MB.
constexpr size_t kRuntimeAvail    = 529845452ull;
// KV the transaction admitted; any figure works, the tests pin the headroom.
constexpr size_t kKvAdmitted      = 1024ull * kMiB;

// The GPT-OSS per-row bytes are multiples of 256, so no alignment rounding
// enters these.
constexpr size_t act_slot(uint32_t n_ubatch) {
    return static_cast<size_t>(n_ubatch) * kActPerRow;
}

constexpr size_t out_slot(uint32_t n_ubatch) {
    return static_cast<size_t>(n_ubatch) * kOutPerRow;
}

// Inputs whose KV-zone headroom (available - admitted - reserve) is exactly
// `headroom` with no compute reserve.
pp_moe_onednn_ring_admission_inputs b50_inputs(uint32_t n_ubatch, size_t headroom) {
    pp_moe_onednn_ring_admission_inputs in;
    in.kv_zone_available_bytes       = kKvAdmitted + headroom;
    in.kv_admitted_bytes             = kKvAdmitted;
    in.compute_reserve_bytes_per_row = 0;
    in.runtime_available_bytes       = kRuntimeAvail;
    in.weight_slot_bytes             = kWeightSlotBytes;
    in.activation_bytes_per_row      = kActPerRow;
    in.output_bytes_per_row          = kOutPerRow;
    in.ring_depth                    = 1;
    in.n_ubatch                      = n_ubatch;
    return in;
}

void test_b50_512_fits_the_runtime_zone() {
    printf("B50 -ub 512 (today's ring):\n");
    const pp_moe_onednn_ring_admission r = pp_moe_onednn_admit_ring(b50_inputs(512, 0));
    check(r.admit, "admitted with zero KV-zone headroom");
    check(!r.activation_in_kv_zone && !r.output_in_kv_zone && r.kv_zone_bytes == 0,
          "the whole ring stays in the RUNTIME zone");
    check(r.activation_slot_bytes == act_slot(512) && r.output_slot_bytes == out_slot(512),
          "slots are 90 MB activation + 180 MB output");
    check(r.largest_fitting_n_ubatch == 512, "largest fitting is the request itself");
}

void test_b50_1024_threshold() {
    printf("B50 -ub 1024 (activation slot to the KV zone):\n");
    const size_t need = act_slot(1024);  // 180 MB

    const pp_moe_onednn_ring_admission at = pp_moe_onednn_admit_ring(b50_inputs(1024, need));
    check(at.admit, "admitted with KV-zone headroom exactly 180 MB");
    check(at.activation_in_kv_zone && !at.output_in_kv_zone,
          "the 360 MB output slot keeps the RUNTIME zone, the 180 MB activation slot goes to the KV zone");
    check(at.kv_zone_bytes == need, "KV zone is charged exactly the activation slot");
    check(at.kv_zone_headroom_bytes == need, "headroom reported as available - admitted - reserve");

    const pp_moe_onednn_ring_admission below = pp_moe_onednn_admit_ring(b50_inputs(1024, need - 1));
    check(!below.admit, "refused one byte short");
    check(below.largest_fitting_n_ubatch == 992, "and names -ub 992 (174.4 MB activation slot) as the largest fit");

    const pp_moe_onednn_ring_admission above = pp_moe_onednn_admit_ring(b50_inputs(1024, need + 1));
    check(above.admit && above.kv_zone_bytes == need, "admitted one byte over, charging the same 180 MB");
}

void test_b50_2048_and_4096() {
    printf("B50 -ub 2048 and 4096:\n");
    const size_t                       need_2048 = out_slot(2048);  // 720 MB
    const pp_moe_onednn_ring_admission r2048     = pp_moe_onednn_admit_ring(b50_inputs(2048, need_2048));
    check(r2048.admit && r2048.output_in_kv_zone && !r2048.activation_in_kv_zone,
          "2048: the 720 MB output slot no longer fits the RUNTIME zone, the 360 MB activation slot does");
    check(r2048.kv_zone_bytes == need_2048, "2048: KV zone charged 720 MB");
    check(!pp_moe_onednn_admit_ring(b50_inputs(2048, need_2048 - 1)).admit, "2048: refused one byte short");

    const size_t                       need_4096 = act_slot(4096) + out_slot(4096);  // 2160 MB
    const pp_moe_onednn_ring_admission r4096     = pp_moe_onednn_admit_ring(b50_inputs(4096, need_4096));
    check(r4096.admit && r4096.activation_in_kv_zone && r4096.output_in_kv_zone, "4096: both slots go to the KV zone");
    check(r4096.kv_zone_bytes == need_4096, "4096: KV zone charged 2160 MB");
    check(!pp_moe_onednn_admit_ring(b50_inputs(4096, need_4096 - 1)).admit, "4096: refused one byte short");
}

void test_b70_sized_headroom_admits_the_top_rung() {
    printf("B70-sized KV-zone headroom (16 GB):\n");
    const pp_moe_onednn_ring_admission r = pp_moe_onednn_admit_ring(b50_inputs(4096, 16384ull * kMiB));
    check(r.admit && r.largest_fitting_n_ubatch == 4096, "-ub 4096 admitted");
}

void test_compute_reserve_is_per_row_and_only_for_kv_zone_rings() {
    printf("Compute-buffer reserve (1 MiB per row):\n");
    const size_t                        need    = act_slot(1024);  // 180 MB
    const size_t                        reserve = 1024ull * kMiB;  // 1024 rows x 1 MiB
    pp_moe_onednn_ring_admission_inputs in      = b50_inputs(1024, need + reserve);
    in.compute_reserve_bytes_per_row            = kMiB;
    const pp_moe_onednn_ring_admission at       = pp_moe_onednn_admit_ring(in);
    check(at.admit && at.compute_reserve_bytes == reserve, "admitted with headroom exactly slot + 1024 MB reserve");

    in                                       = b50_inputs(1024, need + reserve - 1);
    in.compute_reserve_bytes_per_row         = kMiB;
    const pp_moe_onednn_ring_admission below = pp_moe_onednn_admit_ring(in);
    check(!below.admit, "one byte short refuses");
    // 992 rows: 174.4 MB activation slot + 992 MB reserve fit 1204 MB - 1 B.
    check(below.largest_fitting_n_ubatch == 992, "and the largest fit scales the reserve with the rung: 992");

    in                               = b50_inputs(512, 0);
    in.compute_reserve_bytes_per_row = kMiB;
    check(pp_moe_onednn_admit_ring(in).admit,
          "a ring that fits the RUNTIME zone ignores the reserve (-ub 512, zero KV-zone headroom)");
}

void test_overcommitted_zone_admits_only_what_fits_the_runtime_zone() {
    printf("Admitted KV already over what the KV zone can hold:\n");
    pp_moe_onednn_ring_admission_inputs in = b50_inputs(1024, 0);
    in.kv_admitted_bytes                   = in.kv_zone_available_bytes + 1;
    const pp_moe_onednn_ring_admission r   = pp_moe_onednn_admit_ring(in);
    check(!r.admit && r.kv_zone_headroom_bytes == 0, "1024 refused, headroom clamps at 0 instead of wrapping");
    // 672 rows: 354.4 MB of slots fit the 370.8 MB left in RUNTIME; 704 rows
    // (371.3 MB) do not, and there is no KV-zone headroom for the rest.
    check(r.largest_fitting_n_ubatch == 672, "largest fit is -ub 672, the whole ring inside the RUNTIME zone");
    in.n_ubatch = 672;
    check(pp_moe_onednn_admit_ring(in).admit, "672 admitted (control for the line above)");
    in.n_ubatch = 704;
    check(!pp_moe_onednn_admit_ring(in).admit, "704 refused");
}

void test_runtime_zone_thresholds() {
    printf("RUNTIME zone thresholds (KV-zone headroom 16 GB):\n");
    pp_moe_onednn_ring_admission_inputs in = b50_inputs(1024, 16384ull * kMiB);

    in.runtime_available_bytes     = kWeightSlotBytes + act_slot(1024) + out_slot(1024);
    pp_moe_onednn_ring_admission r = pp_moe_onednn_admit_ring(in);
    check(r.admit && r.kv_zone_bytes == 0, "RUNTIME exactly weight + activation + output: all in RUNTIME");
    in.runtime_available_bytes -= 1;
    r = pp_moe_onednn_admit_ring(in);
    check(r.admit && r.activation_in_kv_zone && !r.output_in_kv_zone && r.kv_zone_bytes == act_slot(1024),
          "one byte less: the smaller activation slot moves to the KV zone");

    in.runtime_available_bytes = kWeightSlotBytes + out_slot(1024);
    r                          = pp_moe_onednn_admit_ring(in);
    check(r.admit && !r.output_in_kv_zone && r.activation_in_kv_zone, "RUNTIME exactly weight + output: output stays");
    in.runtime_available_bytes -= 1;
    r = pp_moe_onednn_admit_ring(in);
    check(r.admit && r.output_in_kv_zone && !r.activation_in_kv_zone && r.kv_zone_bytes == out_slot(1024),
          "one byte less: output moves to the KV zone and activation takes the RUNTIME space instead");

    in.runtime_available_bytes = kWeightSlotBytes;
    r                          = pp_moe_onednn_admit_ring(in);
    check(r.admit && r.activation_in_kv_zone && r.output_in_kv_zone,
          "RUNTIME exactly the weight slot: both ubatch slots go to the KV zone");
    in.runtime_available_bytes -= 1;
    r = pp_moe_onednn_admit_ring(in);
    check(!r.admit && r.largest_fitting_n_ubatch == 0,
          "one byte less: the weight slot (never in the KV zone) does not fit, at any -ub");
}

void test_ring_depth_multiplies_every_slot() {
    printf("Ring depth 2:\n");
    pp_moe_onednn_ring_admission_inputs in = b50_inputs(512, 16384ull * kMiB);
    in.ring_depth                          = 2;
    const pp_moe_onednn_ring_admission r   = pp_moe_onednn_admit_ring(in);
    // 2 x 134.5 MB weights leave 236.3 MB: 2 x 180 MB output does not fit,
    // 2 x 90 MB activation does.
    check(r.admit && r.output_in_kv_zone && !r.activation_in_kv_zone, "two output slots go to the KV zone");
    check(r.kv_zone_bytes == 2 * out_slot(512), "KV zone charged 2 x 180 MB");
}

void test_kv_zone_part_must_fit_the_largest_free_block() {
    printf("KV-zone largest free block:\n");
    pp_moe_onednn_ring_admission_inputs in = b50_inputs(1024, 16384ull * kMiB);
    in.kv_zone_largest_block_bytes         = act_slot(1024);
    pp_moe_onednn_ring_admission r         = pp_moe_onednn_admit_ring(in);
    check(r.admit && r.activation_in_kv_zone, "-ub 1024: a block exactly the 180 MB activation slot admits it");
    in.kv_zone_largest_block_bytes -= 1;
    r = pp_moe_onednn_admit_ring(in);
    check(!r.admit, "one byte less: refused, though the zone has 16 GB free in pieces");
    check(r.largest_fitting_n_ubatch == 992, "and the -ub it names (992) has an activation slot that fits the block");
    in.kv_zone_largest_block_bytes = 0;
    r                              = pp_moe_onednn_admit_ring(in);
    check(!r.admit && r.largest_fitting_n_ubatch == 672,
          "no free block: only a ring that fits the RUNTIME zone outright, 672");

    pp_moe_onednn_ring_admission_inputs runtime_only = b50_inputs(512, 0);
    runtime_only.kv_zone_largest_block_bytes         = 0;
    check(pp_moe_onednn_admit_ring(runtime_only).admit, "-ub 512 fits the RUNTIME zone: the KV-zone block is moot");

    // Ring depth 2 puts two 180 MB output slots in the KV zone. A block that
    // holds one of them but not both is refused: the admission counts their sum.
    pp_moe_onednn_ring_admission_inputs two = b50_inputs(512, 16384ull * kMiB);
    two.ring_depth                          = 2;
    two.kv_zone_largest_block_bytes         = 2 * out_slot(512);
    r                                       = pp_moe_onednn_admit_ring(two);
    check(r.admit && r.output_in_kv_zone, "depth 2: a block holding both output slots admits them");
    two.kv_zone_largest_block_bytes = out_slot(512);
    r                               = pp_moe_onednn_admit_ring(two);
    check(!r.admit, "depth 2: a block holding only one output slot is refused");
    // At 320 rows both output slots fit the RUNTIME zone and the two 56.25 MB
    // activation slots, 112.5 MB together, go to the KV zone. Past 336 rows the
    // output slots go there, and two of them no longer fit the 180 MB block.
    check(r.largest_fitting_n_ubatch == 320,
          "and the -ub it names (320) puts two activation slots in the KV zone that fit the block together");
}

void test_kv_zone_part_is_charged_to_the_vram_budget() {
    printf("KV-zone part against the VRAM budget:\n");
    pp_moe_onednn_ring_admission_inputs in = b50_inputs(1024, 16384ull * kMiB);
    in.vram_budget_room_bytes              = act_slot(1024);
    pp_moe_onednn_ring_admission r         = pp_moe_onednn_admit_ring(in);
    check(r.admit && r.activation_in_kv_zone, "-ub 1024: budget room exactly the 180 MB activation slot admits it");
    in.vram_budget_room_bytes -= 1;
    r = pp_moe_onednn_admit_ring(in);
    check(!r.admit, "one byte less: refused, though the KV zone has 16 GB free");
    check(r.largest_fitting_n_ubatch == 992, "and the -ub it names (992) has an activation slot the budget holds");
    in.vram_budget_room_bytes = 0;
    r                         = pp_moe_onednn_admit_ring(in);
    check(!r.admit && r.largest_fitting_n_ubatch == 672,
          "no budget room: only a ring that fits the RUNTIME zone outright, 672");

    // The RUNTIME slots are not charged to the plan's VRAM, so a ring that fits
    // there needs no budget room.
    pp_moe_onednn_ring_admission_inputs runtime_only = b50_inputs(512, 0);
    runtime_only.vram_budget_room_bytes              = 0;
    check(pp_moe_onednn_admit_ring(runtime_only).admit, "-ub 512 fits the RUNTIME zone: no budget room needed");
}

void test_degenerate_and_overflow() {
    printf("Degenerate inputs and overflow:\n");
    pp_moe_onednn_ring_admission_inputs dense = b50_inputs(512, 16384ull * kMiB);
    dense.activation_bytes_per_row            = 0;
    dense.output_bytes_per_row                = 0;
    const pp_moe_onednn_ring_admission rd     = pp_moe_onednn_admit_ring(dense);
    check(!rd.admit && rd.largest_fitting_n_ubatch == 0, "no per-row bytes (dense model, no ring) -> not admitted");

    pp_moe_onednn_ring_admission_inputs no_depth = b50_inputs(512, 16384ull * kMiB);
    no_depth.ring_depth                          = 0;
    check(!pp_moe_onednn_admit_ring(no_depth).admit, "ring_depth=0 -> not admitted");

    pp_moe_onednn_ring_admission_inputs no_rows = b50_inputs(0, 16384ull * kMiB);
    check(!pp_moe_onednn_admit_ring(no_rows).admit, "n_ubatch=0 -> not admitted");

    // 256 rows of kMax / 256 bytes is kMax - 255, a multiple of 256: it fits a
    // KV zone of kMax. 512 rows overflow size_t and must be refused, not
    // wrapped into a small slot that gets admitted.
    pp_moe_onednn_ring_admission_inputs huge = b50_inputs(512, 0);
    huge.activation_bytes_per_row            = kMax / 256;
    huge.kv_zone_available_bytes             = kMax;
    huge.kv_admitted_bytes                   = 0;
    const pp_moe_onednn_ring_admission rh    = pp_moe_onednn_admit_ring(huge);
    check(!rh.admit, "a slot that overflows size_t is refused");
    check(rh.largest_fitting_n_ubatch == 256, "and the largest fit is the last rung before the overflow, 256");
}

// An explicit micro-batch (pp_moe_onednn_explicit_ubatch_plan_room/_decide). THE SHAPE THIS IS FIT TO (B70,
// Qwen3.8-Flash-Next IQ3_XXS, llama.cpp-86bl): weight slot 425 MiB, per row 2.5 MiB activation + 5 MiB output, ring
// depth 1, so the ring is 4265 MiB at the default 512 and 15785 MiB at -ub 2048. Dense scratch 6912 + 12288 B/token
// plus a 60 MiB weight copy; the rest of the RUNTIME requirement (compute and state terms, MoE control, host scatter)
// is about 2168 MiB. The experts were packed into about 18880 MiB of the WEIGHT zone. Every figure is a multiple of
// 256, so the zone's rounding enters only the alignment case.
constexpr size_t kQwenWeightSlot   = 425ull * kMiB;
constexpr size_t kQwenActPerRow    = 2621440ull;
constexpr size_t kQwenOutPerRow    = 5242880ull;
constexpr size_t kQwenFixed        = 2168ull * kMiB;
constexpr size_t kQwenExpertBudget = 18880ull * kMiB;

constexpr size_t qwen_dense(uint32_t n_ubatch) {
    return 6912ull * n_ubatch + 60ull * kMiB + 12288ull * n_ubatch;
}

constexpr size_t qwen_ring(uint32_t n_ubatch) {
    return kQwenWeightSlot + static_cast<size_t>(n_ubatch) * (kQwenActPerRow + kQwenOutPerRow);
}

constexpr size_t qwen_zone(uint32_t n_ubatch) {
    return kQwenFixed + qwen_dense(n_ubatch) + qwen_ring(n_ubatch);
}

pp_moe_onednn_explicit_ubatch_inputs qwen_inputs(uint32_t n_ubatch, size_t runtime_zone) {
    pp_moe_onednn_explicit_ubatch_inputs in;
    in.n_ubatch                    = n_ubatch;
    in.n_ubatch_default            = 512;
    in.weight_slot_bytes           = kQwenWeightSlot;
    in.activation_bytes_per_row    = kQwenActPerRow;
    in.output_bytes_per_row        = kQwenOutPerRow;
    in.ring_depth                  = 1;
    in.runtime_fixed_bytes         = kQwenFixed;
    in.dense_scratch_default_bytes = qwen_dense(512);
    in.dense_scratch_bytes         = qwen_dense(n_ubatch);
    in.runtime_floor_bytes         = 512ull * kMiB;
    in.runtime_zone_bytes          = runtime_zone;
    return in;
}

void test_explicit_ubatch_room_qwen_b70() {
    printf("explicit -ub 2048, Qwen3.8 on the B70:\n");
    check(qwen_ring(512) == 4265ull * kMiB && qwen_ring(2048) == 15785ull * kMiB,
          "the fixture reproduces the log's rings: 4265 MiB at 512, 15785 MiB at 2048");

    // The load's first stage: the zone is the one the default micro-batch planned.
    const pp_moe_onednn_explicit_ubatch_room room =
        pp_moe_onednn_explicit_ubatch_plan_room(qwen_inputs(2048, qwen_zone(512)));
    check(room.valid, "an explicit 2048 above the default 512 with a ring is a room to plan");
    check(room.ring_bytes == 15785ull * kMiB, "the ring at 2048 is 15785 MiB");
    check(room.zone_default_bytes == qwen_zone(512), "the default zone is fixed + dense(512) + ring(512)");
    check(room.zone_bytes == qwen_zone(2048), "the zone for 2048 is fixed + dense(2048) + ring(2048)");
    check(room.room_bytes == 11520ull * kMiB + (qwen_dense(2048) - qwen_dense(512)),
          "the room is the ring's 11520 MiB growth plus the dense scratch's");
    check(room.room_reserved_bytes == 0, "the default zone holds none of it");

    const pp_moe_onednn_explicit_ubatch_decision grow = pp_moe_onednn_explicit_ubatch_decide(room, kQwenExpertBudget);
    check(grow.action == pp_moe_onednn_explicit_ubatch_action::GROW_ZONE,
          "18880 MiB of device experts can give up the room: grow the zone, then re-plan");

    // The re-plan, with the zone grown by exactly the room: placed.
    const pp_moe_onednn_explicit_ubatch_room grown =
        pp_moe_onednn_explicit_ubatch_plan_room(qwen_inputs(2048, qwen_zone(2048)));
    check(grown.room_bytes == room.room_bytes && grown.room_reserved_bytes == room.room_bytes,
          "a zone grown to the 2048 size holds the whole room");
    const size_t left = kQwenExpertBudget - room.room_bytes;
    check(pp_moe_onednn_explicit_ubatch_decide(grown, left).action == pp_moe_onednn_explicit_ubatch_action::PLACE,
          "and the experts packed beside it are placed");
    check(pp_moe_onednn_explicit_ubatch_decide(grown, 0).action == pp_moe_onednn_explicit_ubatch_action::PLACE,
          "a zone that already holds the room is placed whatever the experts were packed into");

    // A zone a load stage grew for a smaller requirement: only the rest has to come from the experts.
    const size_t                             short_by = 2000ull * kMiB;
    const pp_moe_onednn_explicit_ubatch_room part =
        pp_moe_onednn_explicit_ubatch_plan_room(qwen_inputs(2048, qwen_zone(2048) - short_by));
    check(part.room_reserved_bytes == room.room_bytes - short_by, "a part-grown zone holds the room less the rest");
    check(
        pp_moe_onednn_explicit_ubatch_decide(part, short_by).action == pp_moe_onednn_explicit_ubatch_action::GROW_ZONE,
        "the experts can give the rest: grow");
    check(
        pp_moe_onednn_explicit_ubatch_decide(part, short_by - 1).action == pp_moe_onednn_explicit_ubatch_action::REFUSE,
        "one byte short of the rest: refuse");
}

void test_explicit_ubatch_refusal_names_the_largest_fit() {
    printf("explicit -ub the experts cannot make room for:\n");
    // -ub 4096: the ring is 31145 MiB, more than the default zone plus every device expert.
    const pp_moe_onednn_explicit_ubatch_room room =
        pp_moe_onednn_explicit_ubatch_plan_room(qwen_inputs(4096, qwen_zone(512)));
    check(room.ring_bytes == 31145ull * kMiB, "the ring at 4096 is 31145 MiB");
    const pp_moe_onednn_explicit_ubatch_decision d = pp_moe_onednn_explicit_ubatch_decide(room, kQwenExpertBudget);
    check(d.action == pp_moe_onednn_explicit_ubatch_action::REFUSE, "a room above the expert budget is refused");
    check(d.max_ring_runtime_bytes == qwen_zone(512) + kQwenExpertBudget - kQwenFixed - qwen_dense(4096),
          "the ring could have the default zone plus every device expert, less the fixed rest and the dense scratch");
    // 425 MiB + n * 7.5 MiB <= that: n <= 3020.6, so 3008 is the last multiple of 32.
    check(d.largest_fitting_n_ubatch == 3008, "the largest -ub that fits is 3008");

    // Nothing demotable: the figure is what the zone alone holds, the 480 the old refusal printed for this model.
    const pp_moe_onednn_explicit_ubatch_room room2048 =
        pp_moe_onednn_explicit_ubatch_plan_room(qwen_inputs(2048, qwen_zone(512)));
    const pp_moe_onednn_explicit_ubatch_decision none = pp_moe_onednn_explicit_ubatch_decide(room2048, 0);
    check(none.action == pp_moe_onednn_explicit_ubatch_action::REFUSE, "with no device expert to move, refuse");
    check(none.largest_fitting_n_ubatch == 480, "and the largest -ub that fits the default zone is 480");
}

void test_explicit_ubatch_none_and_floor() {
    printf("explicit -ub at or below what the default zone holds:\n");
    check(!pp_moe_onednn_explicit_ubatch_plan_room(qwen_inputs(512, qwen_zone(512))).valid,
          "the default micro-batch itself is no explicit room");
    check(!pp_moe_onednn_explicit_ubatch_plan_room(qwen_inputs(256, qwen_zone(512))).valid,
          "a smaller one is none either: the zone was planned for 512");
    pp_moe_onednn_explicit_ubatch_inputs dense_model = qwen_inputs(2048, qwen_zone(512));
    dense_model.weight_slot_bytes                    = 0;
    check(!pp_moe_onednn_explicit_ubatch_plan_room(dense_model).valid, "a model with no ring has no room");
    pp_moe_onednn_explicit_ubatch_inputs no_default = qwen_inputs(2048, qwen_zone(512));
    no_default.n_ubatch_default                     = 0;
    check(!pp_moe_onednn_explicit_ubatch_plan_room(no_default).valid, "nor a load with no default micro-batch");
    check(pp_moe_onednn_explicit_ubatch_decide(pp_moe_onednn_explicit_ubatch_room{}, kMax).action ==
              pp_moe_onednn_explicit_ubatch_action::NONE,
          "an invalid room decides nothing");

    // A small ring under the zone's floor: the floor holds rings up to its size, and only the excess is a room.
    pp_moe_onednn_explicit_ubatch_inputs small;
    small.n_ubatch_default                         = 512;
    small.weight_slot_bytes                        = 1ull * kMiB;
    small.activation_bytes_per_row                 = 64ull * 1024ull;
    small.output_bytes_per_row                     = 128ull * 1024ull;
    small.ring_depth                               = 1;
    small.runtime_floor_bytes                      = 512ull * kMiB;
    small.runtime_zone_bytes                       = 512ull * kMiB;
    small.n_ubatch                                 = 2048;  // ring 385 MiB, under the floor
    const pp_moe_onednn_explicit_ubatch_room under = pp_moe_onednn_explicit_ubatch_plan_room(small);
    check(under.valid && under.zone_default_bytes == 512ull * kMiB && under.zone_bytes == 512ull * kMiB &&
              under.room_bytes == 0,
          "a 385 MiB ring under the 512 MiB floor needs no room");
    check(pp_moe_onednn_explicit_ubatch_decide(under, kMax).action == pp_moe_onednn_explicit_ubatch_action::NONE,
          "so nothing is decided");
    small.n_ubatch                                = 4096;  // ring 769 MiB
    const pp_moe_onednn_explicit_ubatch_room over = pp_moe_onednn_explicit_ubatch_plan_room(small);
    check(over.zone_bytes == 769ull * kMiB && over.room_bytes == 257ull * kMiB,
          "a 769 MiB ring needs 257 MiB above the floor, not above the 97 MiB default ring");
}

void test_explicit_ubatch_alignment_and_overflow() {
    printf("explicit -ub rounding and overflow:\n");
    // Per-row bytes that are not multiples of 256: each slot rounds up as the ring allocates it, and the zone rounds
    // up as the arena reserves it.
    pp_moe_onednn_explicit_ubatch_inputs odd;
    odd.n_ubatch                               = 1000;
    odd.n_ubatch_default                       = 512;
    odd.weight_slot_bytes                      = 1000;
    odd.activation_bytes_per_row               = 3;
    odd.output_bytes_per_row                   = 5;
    odd.ring_depth                             = 2;
    odd.runtime_fixed_bytes                    = 7;
    odd.runtime_floor_bytes                    = 0;
    const pp_moe_onednn_explicit_ubatch_room r = pp_moe_onednn_explicit_ubatch_plan_room(odd);
    // slots: align256(3000) = 3072, align256(5000) = 5120; ring = 2 * (1000 + 3072 + 5120) = 18384.
    check(r.ring_bytes == 18384, "each ubatch-scaled slot rounds up to 256 and the ring multiplies by its depth");
    // zone = align256(7 + 0 + 18384) = 18432
    check(r.zone_bytes == 18432, "the zone rounds up to 256 like the arena's");

    pp_moe_onednn_explicit_ubatch_inputs huge   = qwen_inputs(4096, qwen_zone(512));
    huge.activation_bytes_per_row               = kMax / 1024;
    const pp_moe_onednn_explicit_ubatch_room hr = pp_moe_onednn_explicit_ubatch_plan_room(huge);
    check(hr.valid && hr.ring_bytes == kMax && hr.room_bytes == kMax,
          "a ring that overflows size_t is a room no budget holds, not a wrapped small one");
    check(pp_moe_onednn_explicit_ubatch_decide(hr, kQwenExpertBudget).action ==
              pp_moe_onednn_explicit_ubatch_action::REFUSE,
          "so it is refused");
}

void test_explicit_ubatch_action_names() {
    printf("explicit -ub action names:\n");
    check(
        std::string(pp_moe_onednn_explicit_ubatch_action_name(pp_moe_onednn_explicit_ubatch_action::NONE)) == "none" &&
            std::string(pp_moe_onednn_explicit_ubatch_action_name(pp_moe_onednn_explicit_ubatch_action::PLACE)) ==
                "place" &&
            std::string(pp_moe_onednn_explicit_ubatch_action_name(pp_moe_onednn_explicit_ubatch_action::GROW_ZONE)) ==
                "grow-zone" &&
            std::string(pp_moe_onednn_explicit_ubatch_action_name(pp_moe_onednn_explicit_ubatch_action::REFUSE)) ==
                "refuse",
        "the four actions print as none, place, grow-zone and refuse");
}

}  // namespace

int main() {
    test_b50_512_fits_the_runtime_zone();
    test_b50_1024_threshold();
    test_b50_2048_and_4096();
    test_b70_sized_headroom_admits_the_top_rung();
    test_compute_reserve_is_per_row_and_only_for_kv_zone_rings();
    test_overcommitted_zone_admits_only_what_fits_the_runtime_zone();
    test_runtime_zone_thresholds();
    test_ring_depth_multiplies_every_slot();
    test_kv_zone_part_must_fit_the_largest_free_block();
    test_kv_zone_part_is_charged_to_the_vram_budget();
    test_degenerate_and_overflow();
    test_explicit_ubatch_room_qwen_b70();
    test_explicit_ubatch_refusal_names_the_largest_fit();
    test_explicit_ubatch_none_and_floor();
    test_explicit_ubatch_alignment_and_overflow();
    test_explicit_ubatch_action_names();

    printf("%s (%d failures)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}
