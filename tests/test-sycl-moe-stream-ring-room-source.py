"""Source contract: the single-device planner holds the prompt phase's window ring room in the WEIGHT zone.

Host-only: reads sources, builds nothing, loads no model and touches no device.

The defect (B70, Qwen3.8 IQ3_XXS): the window ring reserves its slots from the WEIGHT zone, the routed experts fill
that zone at default placement, so the first reserve came up short and only an expert loan could fund the ring. The
loan counts whole slots per lent matrix, and every lendable matrix of that model is smaller than the ring's slot, so
the loan found nothing to lend and the ring never engaged: every prompt ran its expert layers on the CPU.

The fix this gate pins:
  - compute_placement_plan holds the ring's room as the last term before the expert pack, after the KV room and
    after every group's PP-alternate charge, and before the pack and its with-the-KV-room-added-back first fit;
  - the room is the ring's own size, from moe_stream_plan_ring_geometry (the function the phase and the auto-ubatch
    fit size the ring with) under the planned f16 weight slot, the executor's group cap and nothing lent, over the
    pack the planner predicts with the pack's own first-fit predicate, iterated to the ring of the plan it produces;
  - the room comes out of `remaining`, is recorded in the plan and counted in vram_bytes, and the runtime
    transaction's KV re-derivation keeps it there; none is held when the ring's cap refuses the ring under its room
    or the room exceeds what is left;
  - the backend fills the planner's inputs where it copies the inventory in: the staged-type mask and the group
    cap. The mask and the executor's staged claim ask one helper for the claim's type half, the only place its two
    terms are spelled. The claim's only other questions about the type are the type-only gate the helper implies and
    two that read the planned layout, which the plan decides after the mask;
  - the decision prints one [SYCL-PLAN] line, a WARN when the room displaced experts or did not settle;
  - the room needs no padding: where it is sized, the build refuses a slot alignment other than the zone
    allocator's grain, the one alignment at which each slot takes exactly its bytes;
  - a run that has not settled in max_rounds packs once more with the room it holds, so the line's cost is that
    room's and not the previous round's;
  - the line's slot figures are the ring the held room was taken from, and the round bound is written once.

Every claim is checked on COMMENT-STRIPPED, whitespace-normalized text (sycl_gate.norm, which also joins adjacent
string literals) and has a mutant that must make it fail.

Pytest-style (module-level test_* functions): register with llama_test_pytest.
"""

import re

from sycl_gate import body, norm, ordered, read

CACHE = norm(read("ggml/src/ggml-sycl/unified-cache.cpp"))
CACHE_HPP = norm(read("ggml/src/ggml-sycl/unified-cache.hpp"))
SYCL = norm(read("ggml/src/ggml-sycl/ggml-sycl.cpp"))


def _once(text: str, old: str, new: str) -> str:
    """`text` (already normalized) with the one occurrence of `old` replaced by `new`. Stricter than sycl_gate.mutate,
    which needs `old` present: an anchor that matches twice would let a mutant edit the wrong copy."""
    assert text.count(old) == 1, f"mutant anchor must match exactly once: {old!r} x{text.count(old)}"
    return text.replace(old, new, 1)


PLAN_SIG = "placement_plan compute_placement_plan(const std::vector<placement_tensor_info> & tensor_inventory,"
HELPER_SIG = "static void hold_moe_stream_ring_room(placement_plan & plan,"
DRY_SIG = "static size_t moe_stream_room_dry_pack(placement_plan & plan,"
REFRESH_SIG = "void refresh_kv_byte_totals() {"
POPULATE_SIG = ("static void populate_inventory_globals(ggml_backend_sycl_context * ctx, "
                "const ggml_sycl_tensor_inventory * inventory) {")
MASK_SIG = "static uint64_t ggml_sycl_moe_stream_staged_types() {"
CLAIM_SIG = ("static bool ggml_sycl_moe_pp_onednn_batched_claims_tensor(const ggml_sycl::placement_plan * plan, "
             "const ggml_tensor * weight, int device, bool host_staged) {")

KV_ROOM_CALL = "room = hold_kv_context_room(plan, kv_info, device_id, n_experts, room_expert_rate, remaining);"
PP_ALT_LINE = 'GGML_LOG_INFO("[PLACEMENT-MOE] PP executable alternate reserve: '
RING_ROOM_CALL = "hold_moe_stream_ring_room(plan, moe_groups, kv_info, device_id, remaining);"
PACK_STATS = "moe_triplet_pack_stats stats;"
WITHOUT_KV_ROOM = "size_t without_room = remaining + room.held;"
PACK_FIT = "const bool on_device = group.charge_bytes <= remaining;"
PACK_TAKE = "remaining -= group.charge_bytes;"


# ---- (a) the room is the last term before the pack --------------------------------------------------------------


def claim_room_is_the_last_term_before_the_pack(cache: str) -> bool:
    """After the KV room and the PP-alternate charges (so every group's charge is final), before the pack: once."""
    b = body(cache, PLAN_SIG)
    return (bool(b) and b.count(RING_ROOM_CALL) == 1
            and ordered(b, KV_ROOM_CALL, PP_ALT_LINE, RING_ROOM_CALL, PACK_STATS, WITHOUT_KV_ROOM, PACK_FIT))


def test_room_is_the_last_term_before_the_pack():
    assert claim_room_is_the_last_term_before_the_pack(CACHE)


def test_mutant_room_not_held_fails():
    assert not claim_room_is_the_last_term_before_the_pack(_once(CACHE, RING_ROOM_CALL, ""))


def test_mutant_room_held_after_the_pack_fails():
    """The experts would take the ring's bytes first, which is the defect."""
    pack_log = 'log_moe_triplet_pack_stats("PLACEMENT-MOE", stats, remaining);'
    mutant = _once(_once(CACHE, RING_ROOM_CALL, ""), pack_log, RING_ROOM_CALL + " " + pack_log)
    assert not claim_room_is_the_last_term_before_the_pack(mutant)


def test_mutant_room_held_twice_fails():
    assert not claim_room_is_the_last_term_before_the_pack(_once(CACHE, RING_ROOM_CALL, RING_ROOM_CALL * 2))


# ---- (b) the room is held, counted and guarded ------------------------------------------------------------------

HELD_WRITES = ("remaining -= room;", "plan.moe_stream_ring_reserve_bytes = room;", "plan.vram_bytes += room;")
CAP_GUARD = "if (settled && geo.bytes == 0) {"
LEFT_GUARD = "if (room > remaining) {"


def claim_room_is_held_counted_and_guarded(cache: str) -> bool:
    """None is held when the ring's cap refuses the ring under its room or the room exceeds what is left; otherwise
    the room comes out of `remaining`, is recorded, and counts in the plan's device bytes, each once."""
    h = body(cache, HELPER_SIG)
    return (bool(h) and ordered(h, CAP_GUARD, LEFT_GUARD, *HELD_WRITES)
            and all(h.count(w) == 1 for w in HELD_WRITES))


def test_room_is_held_counted_and_guarded():
    assert claim_room_is_held_counted_and_guarded(CACHE)


def test_mutant_room_not_counted_in_vram_bytes_fails():
    """S1-PRELOAD's WOQ guard keeps vram_bytes free: an uncounted room would be WOQ's to take."""
    assert not claim_room_is_held_counted_and_guarded(_once(CACHE, "plan.vram_bytes += room;", ""))


def test_mutant_room_not_taken_from_remaining_fails():
    """The pack and the post-pack alternates would fill the room."""
    assert not claim_room_is_held_counted_and_guarded(_once(CACHE, "remaining -= room;", ""))


def test_mutant_room_held_past_what_is_left_fails():
    assert not claim_room_is_held_counted_and_guarded(_once(CACHE, LEFT_GUARD, "if (false) {"))


def test_mutant_cap_guard_dropped_fails():
    """Without the cap's refusal the planner would hold room for a ring the phase's cap refuses once the room has
    displaced experts, and those experts would go to the host for nothing."""
    assert not claim_room_is_held_counted_and_guarded(_once(CACHE, CAP_GUARD, "if (false) {"))


# ---- (c) the room is the shared geometry, iterated over the predicted pack ---------------------------------------

HINTS = ("hints.weight_slot_bytes = plan.pp_moe_onednn_weight_slot_bytes;",
         "hints.group_cap = kv_info.moe_stream_group_cap;", "hints.lent_bytes = 0;")
GEOMETRY = "geo = moe_stream_plan_ring_geometry(plan, device_id, &claims, moe_stream_room_claims_by_type, hints);"
ROUND = ("while (rounds < max_rounds) {",
         "placed_with = moe_stream_room_dry_pack(plan, groups, device_id, avail, &groups_with);", GEOMETRY,
         "if (geo.bytes <= room) { settled = true; break; }", "room = geo.bytes;")


def claim_room_is_the_shared_geometry_iterated(cache: str) -> bool:
    """One call of the geometry function the phase and the fit use, with the planned weight slot, the group cap and
    nothing lent, in a loop that re-packs with the room held until the ring is no larger than the room."""
    h = body(cache, HELPER_SIG)
    return (bool(h) and h.count("moe_stream_plan_ring_geometry(") == 1 and "moe_stream_ring_size(" not in h
            and "moe_stream_plan_index_build(" not in h and all(x in h for x in HINTS) and ordered(h, *ROUND))


def test_room_is_the_shared_geometry_iterated():
    assert claim_room_is_the_shared_geometry_iterated(CACHE)


def test_mutant_room_counts_itself_lent_fails():
    """The phase reads the published plan with nothing lent: a cap that counted the room back would size a ring the
    phase refuses."""
    assert not claim_room_is_the_shared_geometry_iterated(_once(CACHE, "hints.lent_bytes = 0;",
                                                                "hints.lent_bytes = room;"))


def test_mutant_room_ignores_the_group_cap_fails():
    assert not claim_room_is_the_shared_geometry_iterated(_once(CACHE, "hints.group_cap = kv_info.moe_stream_group_cap;",
                                                                "hints.group_cap = 0;"))


def test_mutant_room_sized_by_a_second_formula_fails():
    mutant = _once(CACHE, GEOMETRY, "geo = moe_stream_ring_size(3, 1, SIZE_MAX);")
    assert not claim_room_is_the_shared_geometry_iterated(mutant)


def test_mutant_room_sized_in_one_round_fails():
    """Without the re-pack, the room is the ring of the plan without it, and holding it can widen the ring."""
    mutant = _once(CACHE, "if (geo.bytes <= room) { settled = true; break; }", "settled = true; break;")
    assert not claim_room_is_the_shared_geometry_iterated(mutant)


# ---- (d) the predicted pack is the pack's first fit -------------------------------------------------------------

DRY_FIT = "const bool fits = group.charge_bytes <= avail;"
DRY_TAKE = "avail -= group.charge_bytes;"


def claim_dry_pack_is_the_packs_first_fit(cache: str) -> bool:
    """The prediction and the pack decide a group by the same comparison, so one cannot change without the other."""
    d = body(cache, DRY_SIG)
    b = body(cache, PLAN_SIG)
    return bool(d) and bool(b) and ordered(d, DRY_FIT, DRY_TAKE) and ordered(b, PACK_FIT, PACK_TAKE)


def test_dry_pack_is_the_packs_first_fit():
    assert claim_dry_pack_is_the_packs_first_fit(CACHE)


def test_mutant_dry_pack_strict_fails():
    assert not claim_dry_pack_is_the_packs_first_fit(_once(CACHE, DRY_FIT,
                                                           "const bool fits = group.charge_bytes < avail;"))


def test_mutant_pack_predicate_changed_alone_fails():
    assert not claim_dry_pack_is_the_packs_first_fit(_once(CACHE, PACK_FIT,
                                                           "const bool on_device = group.charge_bytes < remaining;"))


# ---- (e) the runtime KV re-derivation keeps the room --------------------------------------------------------------

REFRESH_KEEP = "vram_bytes += moe_stream_ring_reserve_bytes;"


def claim_refresh_keeps_the_room(hpp: str) -> bool:
    """The ring is reserved again at every prompt, so its room outlives the load plan's KV room."""
    r = body(hpp, REFRESH_SIG)
    return (bool(r) and ordered(r, "vram_bytes = weight_vram_bytes + kv_vram_bytes;", REFRESH_KEEP)
            and "moe_stream_ring_reserve_bytes = 0" not in r)


def test_refresh_keeps_the_room():
    assert claim_refresh_keeps_the_room(CACHE_HPP)


def test_mutant_refresh_drops_the_room_fails():
    assert not claim_refresh_keeps_the_room(_once(CACHE_HPP, REFRESH_KEEP, ""))


def test_mutant_refresh_zeroes_the_room_fails():
    assert not claim_refresh_keeps_the_room(_once(CACHE_HPP, REFRESH_KEEP,
                                                  "moe_stream_ring_reserve_bytes = 0; " + REFRESH_KEEP))


# ---- (f) the backend fills the inputs, and the mask is the staged claim's type half ------------------------------

FILL_MASK = "g_placement_kv_info.moe_stream_staged_types = ggml_sycl_moe_stream_staged_types();"
FILL_CAP = "g_placement_kv_info.moe_stream_group_cap = ggml_sycl_moe_pp_f16_group_experts_max();"
TYPE_SIG = "static bool ggml_sycl_moe_pp_batched_claims_staged_type(ggml_type type) {"
TYPE_TERMS = ("moe_pp_batched_takes_staged_type(type) && "
              "ggml_sycl_moe_pp_batched_takes_layout(type, GGML_LAYOUT_AOS)")
TYPE_BODY = "{ return " + TYPE_TERMS + "; }"
MASK_ASKS = "if (ggml_sycl_moe_pp_batched_claims_staged_type(type)) {"
CLAIM_ASKS = ("planned = plan != nullptr && ggml_sycl_moe_pp_batched_claims_staged_type(weight->type) && "
              "ggml_sycl_moe_tensor_plan_staged_layout(*plan, weight, device, &planned_layout);")
CLAIM_INLINE = ("planned = plan != nullptr && moe_pp_batched_takes_staged_type(weight->type) && "
                "ggml_sycl_moe_pp_batched_takes_layout(weight->type, GGML_LAYOUT_AOS) && "
                "ggml_sycl_moe_tensor_plan_staged_layout(*plan, weight, device, &planned_layout);")
# Each term of the type half, wherever it is spelled: the staged-type question, and the layout question asked at AOS.
STAGED_TERM_RE = re.compile(r"\bmoe_pp_batched_takes_staged_type\(")
AOS_TERM_RE = re.compile(r"\bggml_sycl_moe_pp_batched_takes_layout\([^()]*, GGML_LAYOUT_AOS\)")
STAGED_BRANCH = "if (host_staged) {"
# Every question the claim asks about the weight's type. The first is the type-only gate, "any layout taken", which
# the helper's AOS term implies; the helper answers the type half; the last two read the planned layout, which the
# plan decides after the mask is taken. Any other use of the type is a question the mask does not ask.
CLAIM_TYPE_USES = ("!ggml_sycl_moe_pp_batched_takes_type(weight->type)",
                   "ggml_sycl_moe_pp_batched_claims_staged_type(weight->type)",
                   "moe_pp_batched_has_expert_entries(weight->type, planned_layout, device_buffer)",
                   "return ggml_sycl_moe_pp_batched_takes_layout(weight->type, planned_layout);")
EXTRA_TYPE_STATEMENT = "planned = planned && weight->type != GGML_TYPE_Q8_0;"
AFTER_STAGED_BRANCH = ("ggml_sycl_moe_tensor_plan_primary_layout(weight, device, &planned_layout); } "
                       "if (!planned) { return false; }")


def claim_backend_fills_the_inputs(sycl: str) -> bool:
    p = body(sycl, POPULATE_SIG)
    return bool(p) and p.count(FILL_MASK) == 1 and p.count(FILL_CAP) == 1


def claim_mask_is_the_staged_claims_type_half(sycl: str) -> bool:
    """The mask and the staged claim ask one helper for the type half, so the room is sized over the matrices the
    executor will claim. The helper is the only place either term is spelled in the file (a census, so a second copy
    anywhere goes red). The claim's staged branch is the helper's statement alone, and the claim asks nothing else
    about the type: a census of its uses of the weight's type admits only CLAIM_TYPE_USES, each once, so a type
    statement added inside the branch or after it goes red."""
    h = body(sycl, TYPE_SIG)
    m = body(sycl, MASK_SIG)
    c = body(sycl, CLAIM_SIG)
    if not (h and m and c):
        return False
    helper_only = (len(STAGED_TERM_RE.findall(sycl)) == 1 and len(STAGED_TERM_RE.findall(h)) == 1
                   and len(AOS_TERM_RE.findall(sycl)) == 1 and len(AOS_TERM_RE.findall(h)) == 1)
    branch_alone = body(c, STAGED_BRANCH) == "{ " + CLAIM_ASKS + " }"
    type_uses = (c.count("weight->type") == len(CLAIM_TYPE_USES)
                 and all(c.count(use) == 1 for use in CLAIM_TYPE_USES))
    return (h == TYPE_BODY and helper_only and m.count(MASK_ASKS) == 1 and c.count(CLAIM_ASKS) == 1
            and branch_alone and type_uses)


def test_backend_fills_the_inputs():
    assert claim_backend_fills_the_inputs(SYCL)


def test_mask_is_the_staged_claims_type_half():
    assert claim_mask_is_the_staged_claims_type_half(SYCL)


def test_mutant_mask_not_filled_fails():
    assert not claim_backend_fills_the_inputs(_once(SYCL, FILL_MASK, ""))


def test_mutant_group_cap_not_filled_fails():
    assert not claim_backend_fills_the_inputs(_once(SYCL, FILL_CAP, ""))


def test_mutant_helper_drops_the_layout_term_fails():
    assert not claim_mask_is_the_staged_claims_type_half(
        _once(SYCL, TYPE_BODY, "{ return moe_pp_batched_takes_staged_type(type); }"))


def test_mutant_claim_inlines_its_own_terms_again_fails():
    assert not claim_mask_is_the_staged_claims_type_half(_once(SYCL, CLAIM_ASKS, CLAIM_INLINE))


def test_mutant_mask_inlines_its_own_terms_again_fails():
    assert not claim_mask_is_the_staged_claims_type_half(
        _once(SYCL, MASK_ASKS, "if (" + TYPE_TERMS + ") {"))


def test_mutant_claim_adds_a_type_term_the_mask_lacks_fails():
    assert not claim_mask_is_the_staged_claims_type_half(
        _once(SYCL, CLAIM_ASKS, CLAIM_ASKS.replace("plan != nullptr && ",
                                                   "plan != nullptr && weight->type != GGML_TYPE_Q8_0 && ")))


def test_mutant_mask_stops_asking_the_helper_fails():
    assert not claim_mask_is_the_staged_claims_type_half(
        _once(SYCL, MASK_ASKS, "if (moe_pp_batched_takes_staged_type(type)) {"))


def test_mutant_staged_branch_asks_the_type_again_fails():
    assert not claim_mask_is_the_staged_claims_type_half(
        _once(SYCL, CLAIM_ASKS, CLAIM_ASKS + " " + EXTRA_TYPE_STATEMENT))


def test_mutant_claim_asks_the_type_after_the_staged_branch_fails():
    assert not claim_mask_is_the_staged_claims_type_half(
        _once(SYCL, AFTER_STAGED_BRANCH, AFTER_STAGED_BRANCH.replace(
            "} if (!planned) {", "} " + EXTRA_TYPE_STATEMENT + " if (!planned) {")))


# ---- (g) the decision line -----------------------------------------------------------------------------------------

HELD_LINE = '"[SYCL-PLAN] MoE window ring room on device %d: held %.1f MB in the WEIGHT zone before the routed experts'
WARN_WHEN = "const bool warn = displaced > 0 || !settled;"
LEVEL = "ggml_log_internal(warn ? GGML_LOG_LEVEL_WARN : GGML_LOG_LEVEL_INFO,"


def claim_decision_line_is_visible(cache: str) -> bool:
    """One [SYCL-PLAN] line for a held room, after the plan counts it, as a WARN when the room displaced experts or
    did not settle, so a default run shows it."""
    h = body(cache, HELPER_SIG)
    return (bool(h) and h.count(HELD_LINE) == 1 and WARN_WHEN in h
            and ordered(h, "plan.vram_bytes += room;", WARN_WHEN, LEVEL, HELD_LINE))


def test_decision_line_is_visible():
    assert claim_decision_line_is_visible(CACHE)


def test_mutant_decision_line_never_warns_fails():
    assert not claim_decision_line_is_visible(_once(CACHE, WARN_WHEN, "const bool warn = false;"))


def test_mutant_decision_line_renamed_fails():
    assert not claim_decision_line_is_visible(_once(CACHE, HELD_LINE, HELD_LINE.replace("[SYCL-PLAN]", "[PLACEMENT]")))


# ---- (h) the room needs no padding ---------------------------------------------------------------------------------

ALIGN_ASSERT = "static_assert(MOE_STREAM_RING_SLOT_ALIGN == tlsf_allocator::block_grain,"


def claim_room_needs_no_padding(cache: str) -> bool:
    """A WEIGHT-zone block takes the request rounded to the larger of its alignment and the grain, so n_slots x
    slot_bytes funds every slot only while the slot alignment is the grain: the build checks it once, beside the room."""
    at = cache.find(ALIGN_ASSERT)
    return cache.count(ALIGN_ASSERT) == 1 and 0 <= at < cache.find(HELPER_SIG)


def test_room_needs_no_padding():
    assert claim_room_needs_no_padding(CACHE)


def test_mutant_alignment_assert_dropped_fails():
    assert not claim_room_needs_no_padding(_once(CACHE, ALIGN_ASSERT, "static_assert(true,"))


def test_mutant_alignment_assert_weakened_fails():
    """At an alignment below the grain a slot still takes a whole number of grains, more than slot_bytes."""
    weakened = ALIGN_ASSERT.replace(" == ", " <= ")
    assert not claim_room_needs_no_padding(_once(CACHE, ALIGN_ASSERT, weakened))


# ---- (i) an unsettled run's cost is the held room's ------------------------------------------------------------------

UNSETTLED_REPACK = ("if (!settled) { const size_t held_avail = remaining > room ? remaining - room : 0; "
                    "placed_with = moe_stream_room_dry_pack(plan, groups, device_id, held_avail, &groups_with); }")
RESTORE = "plan.entries[idx].on_device = saved[i].first;"


def claim_unsettled_cost_is_the_held_rooms(cache: str) -> bool:
    """The loop's last round packs with the previous round's room, so a run that stops unsettled packs once more with
    the room it holds, after the loop and before the entries are restored and the line is printed."""
    h = body(cache, HELPER_SIG)
    return (bool(h) and h.count(UNSETTLED_REPACK) == 1
            and ordered(h, "room = geo.bytes;", UNSETTLED_REPACK, RESTORE, HELD_LINE))


def test_unsettled_cost_is_the_held_rooms():
    assert claim_unsettled_cost_is_the_held_rooms(CACHE)


def test_mutant_unsettled_repack_dropped_fails():
    assert not claim_unsettled_cost_is_the_held_rooms(_once(CACHE, UNSETTLED_REPACK, ""))


def test_mutant_unsettled_repack_ignores_the_room_fails():
    mutant = _once(CACHE, UNSETTLED_REPACK, UNSETTLED_REPACK.replace("device_id, held_avail,", "device_id, remaining,"))
    assert not claim_unsettled_cost_is_the_held_rooms(mutant)


def test_mutant_unsettled_repack_after_the_restore_fails():
    """Packing after the restore would leave the entries as the dry pack wrote them, not as the plan had them."""
    restore_loop_end = RESTORE + " plan.entries[idx].target_device = saved[i].second; ++i; } } }"
    mutant = _once(_once(CACHE, UNSETTLED_REPACK, ""), restore_loop_end, restore_loop_end + " " + UNSETTLED_REPACK)
    assert not claim_unsettled_cost_is_the_held_rooms(mutant)


# ---- (j) the line names the ring the room was taken from, and the round bound has one source ---------------------

HELD_GEO_TAKEN = "room = geo.bytes; held_geo = geo;"
HELD_GEO_WRITE_RE = re.compile(r"\bheld_geo(?:\.\w+)? (?:=(?!=)|\+=|-=)")
LINE_SLOTS = "held_geo.n_slots, held_geo.slot_bytes / mb, held_geo.region_slots, rounds,"
ROUND_BOUND = "constexpr int max_rounds = 8;"
UNSETTLED_SUFFIX = '" (not settled: the phase\'s ring may be larger, and an expert loan funds the rest)"'
ROUND_COUNT_LITERAL_RE = re.compile(r'"[^"]*\b\d+ rounds?\b[^"]*"')


def claim_line_names_the_held_ring(cache: str) -> bool:
    """The held ring is taken with the room, and nowhere else, so a ring that shrinks below the room without reaching
    0 leaves the line naming the slots that make up the room; the last round's geometry never reaches the line's slot
    figures."""
    h = body(cache, HELPER_SIG)
    return (bool(h) and h.count(HELD_GEO_TAKEN) == 1 and len(HELD_GEO_WRITE_RE.findall(h)) == 1
            and h.count(LINE_SLOTS) == 1
            and not any(f"geo.{f}" in h.replace("held_geo.", "") for f in ("n_slots", "slot_bytes", "region_slots")))


def claim_round_bound_has_one_source(cache: str) -> bool:
    """The bound is max_rounds alone: the line prints the rounds it took and the unsettled suffix names no count."""
    h = body(cache, HELPER_SIG)
    return (bool(h) and h.count(ROUND_BOUND) == 1 and h.count(UNSETTLED_SUFFIX) == 1
            and not ROUND_COUNT_LITERAL_RE.search(h))


def test_line_names_the_held_ring():
    assert claim_line_names_the_held_ring(CACHE)


def test_mutant_line_prints_the_last_rounds_ring_fails():
    assert not claim_line_names_the_held_ring(
        _once(CACHE, LINE_SLOTS, "geo.n_slots, geo.slot_bytes / mb, geo.region_slots, rounds,"))


def test_mutant_held_ring_not_taken_with_the_room_fails():
    assert not claim_line_names_the_held_ring(_once(CACHE, HELD_GEO_TAKEN, "room = geo.bytes;"))


def test_mutant_held_ring_taken_again_after_the_loop_fails():
    """A second take after the loop would hand the line the last round's ring again, the smaller one."""
    assert not claim_line_names_the_held_ring(
        _once(CACHE, HELD_GEO_TAKEN + " } if (!settled) {", HELD_GEO_TAKEN + " } held_geo = geo; if (!settled) {"))


def test_round_bound_has_one_source():
    assert claim_round_bound_has_one_source(CACHE)


def test_mutant_suffix_spells_the_bound_again_fails():
    assert not claim_round_bound_has_one_source(
        _once(CACHE, UNSETTLED_SUFFIX, UNSETTLED_SUFFIX.replace("(not settled:", "(not settled after 8 rounds:")))


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
