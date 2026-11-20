"""Source contract for the SCRATCH zone's MoE batched MMVQ scratch bound.

Each MUL_MAT_ID op that mmvq_moe_batched_dispatch runs takes a Q8_1 copy of its activations and an ids table from the
device SCRATCH zone and hands both to retain_handles_until_event. A detached drain thread releases them once the op's
completion event fires; no later op's allocation waits for that release, and the zone's allocator reclaims nothing on
its own. So nothing bounds how many ops hold their scratch at once, and the zone is sized for every op's buffers
together: zone_moe_mmvq_scratch_bytes's graph figure, at the micro-batch the context runs. The pure rules
(zone_scratch_zone_plan, zone_scratch_kept_bytes, zone_scratch_warn_due, and the arena reserve's checks with
zone_arena_scratch_room) are checked by test-zone-sizing Cases 47 to 49. These checks pin how the cache and the backend
use them:

The planner computes the bound at the context's micro-batch (the load's planning micro-batch until a context names
one), stores the graph figure in the plan, and publishes it to every device the plan places. Its line names the plan's
tensor count, the micro-batch and the depth, and states no zone size: the ensure holds the bound to two limits the
planner does not see, so only the ensure's line states the zone. An MXFP4 operand names its expert count, which sizes
its grouped dispatch's per-op tables, and the tables' rows a chunk stay pinned to the dispatch's constant.

The ensure sizes the SCRATCH zone from zone_scratch_zone_plan over the arena's default, the non-FA attention raise and
the published bound, held to the quarter of the available budget the non-FA raise is held to (one figure, read once)
and to the room the arena's reserve accepts, computed after the ONEDNN and RUNTIME zones from the arguments the reserve
is called with. A clamp logs a latched WARN that names the limit and the miss path. arena_reserve makes its tail checks
through the same zone-sizing functions the room is built from, so the bound never makes the reserve refuse a zone it
would have reserved without it.

A rebuild that something live blocks keeps a SCRATCH zone short of the bound alone when the ONEDNN and RUNTIME zones
hold their plan, binds the compute arena and keeps the ONEDNN plan, says so through the latched WARN and returns true
before any teardown of the arena; it is never a new refusal. A ring that alone holds the arena defers that keep with a
REBUILD_HELD refusal that names the holders, and the load's entry ensure recovers by releasing the ring or asking again.

The runtime context transaction re-sizes the bound at the context's micro-batch after the oneDNN Graph check, never
below the plan's figure, raises the published bound (never stores it plainly) only outside a probe, and the raise
never lowers it.

Host-only text assertions, no SYCL device. llama_test_pytest hands this file to pytest.main(), so the checks live
inside test_*() functions. Checks read COMMENT-STRIPPED, whitespace-normalised text. Each check has a mutation witness
that applies the regression it guards and must make the check fail.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"
GS = (SYCL / "ggml-sycl.cpp").read_text()
UC = (SYCL / "unified-cache.cpp").read_text()
ZS = (SYCL / "zone-sizing.cpp").read_text()

_LEXEME_RE = re.compile(
    r'"(?:\\.|[^"\\\n])*"'  # string literal (kept)
    r"|'(?:\\.|[^'\\\n])*'"  # char literal (kept)
    r"|//[^\n]*"  # line comment (dropped)
    r"|/\*.*?\*/",  # block comment (dropped)
    flags=re.DOTALL,
)


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def code(src: str) -> str:
    """Comment-stripped text with every run of whitespace made one space. Idempotent."""
    return re.sub(r"\s+", " ", strip_comments(src))


def block_at(text: str, start: int) -> str:
    """The brace block that opens at the first '{' at or after `start`, skipping string and char literals."""
    o = text.find("{", start) if start >= 0 else -1
    if o == -1:
        return ""
    depth, i = 0, o
    while i < len(text):
        c = text[i]
        if c in "\"'":
            j = i + 1
            while j < len(text) and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            i = j + 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[o : i + 1]
        i += 1
    return ""


def fn_body(text: str, sig: str) -> str:
    s = text.find(sig)
    return block_at(text, s + len(sig)) if s != -1 else ""


def ordered(text: str, *parts: str) -> bool:
    pos = 0
    for p in parts:
        k = text.find(p, pos)
        if k == -1:
            return False
        pos = k + len(p)
    return bool(text)


def mutate_in(src: str, sig: str, old: str, new: str) -> str:
    """`src` as code() with the first `old` inside the function `sig` replaced by `new`."""
    c = code(src)
    body = fn_body(c, sig)
    assert body and old in body, f"witness anchor {old!r} is not in {sig!r}"
    s = c.find(body, c.find(sig))
    return c[:s] + body.replace(old, new, 1) + c[s + len(body) :]


# ---- the planner sizes the bound at the context's micro-batch and publishes it ----

POPULATE_SIG = "static void populate_host_zone_sizing("
PLAN_UBATCH = ("plan.moe_mmvq_scratch_n_ubatch = plan.context_n_ubatch != 0 ? plan.context_n_ubatch : "
               "placement_load_n_ubatch(plan.planner_n_ubatch);")
PLAN_BOUND = ("if (!zone_moe_mmvq_scratch_bytes(plan.moe_mmvq_scratch_ops, plan.moe_mmvq_scratch_n_ubatch, "
              "n_expert_used, &mmvq_scratch)) {")
PLAN_GRAPH = "plan.moe_mmvq_scratch_bytes = mmvq_scratch.graph;"
PLAN_LINE = "plan of %zu tensors: %zu expert op(s) at n_ubatch=%u, depth %zu"
PLAN_LINE_ARGS = "tensor_inventory.size(), n_ops, plan.moe_mmvq_scratch_n_ubatch, n_ops,"
PUBLISH_ONE = "unified_cache_set_planned_moe_mmvq_scratch_bytes(plan.device_id, plan.moe_mmvq_scratch_bytes);"
PUBLISH_EACH = ("for (int device : plan.devices) { "
                "unified_cache_set_planned_moe_mmvq_scratch_bytes(device, plan.moe_mmvq_scratch_bytes); }")


def claim_plan_publishes_the_bound(uc: str) -> bool:
    body = fn_body(code(uc), POPULATE_SIG)
    return (ordered(body, PLAN_UBATCH, PLAN_BOUND, PLAN_GRAPH, PLAN_LINE, PLAN_LINE_ARGS, PUBLISH_ONE, PUBLISH_EACH)
            and body.count("plan.moe_mmvq_scratch_bytes =") == 2
            and "zone_scratch_zone_plan(" not in body)


def test_the_plan_publishes_the_bound_at_the_context_micro_batch():
    assert claim_plan_publishes_the_bound(UC), (
        "populate_host_zone_sizing does not size the MoE batched MMVQ scratch at the context's micro-batch, store the "
        "every-op (graph) figure, name its tensor count, micro-batch and depth, and publish it to every device, or "
        "its line states a zone size the ensure may clamp")


def test_witness_plan_line_states_a_zone_size():
    m = mutate_in(UC, POPULATE_SIG, PLAN_GRAPH,
                  PLAN_GRAPH + " (void) zone_scratch_zone_plan(0, 0, mmvq_scratch.graph, SIZE_MAX, SIZE_MAX);")
    assert not claim_plan_publishes_the_bound(m)


def test_witness_plan_sizes_one_op():
    m = mutate_in(UC, POPULATE_SIG, PLAN_GRAPH, "plan.moe_mmvq_scratch_bytes = mmvq_scratch.per_op;")
    assert not claim_plan_publishes_the_bound(m)


def test_witness_plan_sizes_at_the_planning_micro_batch():
    m = mutate_in(UC, POPULATE_SIG, PLAN_UBATCH,
                  "plan.moe_mmvq_scratch_n_ubatch = placement_load_n_ubatch(plan.planner_n_ubatch);")
    assert not claim_plan_publishes_the_bound(m)


def test_witness_plan_line_drops_the_tensor_count():
    m = mutate_in(UC, POPULATE_SIG, PLAN_LINE_ARGS, "n_ops, plan.moe_mmvq_scratch_n_ubatch, n_ops,")
    assert not claim_plan_publishes_the_bound(m)


def test_witness_multi_device_plan_publishes_nothing():
    m = mutate_in(UC, POPULATE_SIG, PUBLISH_EACH, "for (int device : plan.devices) { (void) device; }")
    assert not claim_plan_publishes_the_bound(m)


def test_witness_plan_stores_a_second_figure():
    m = mutate_in(UC, POPULATE_SIG, PUBLISH_ONE, "plan.moe_mmvq_scratch_bytes = 0; " + PUBLISH_ONE)
    assert not claim_plan_publishes_the_bound(m)


OPERANDS_SIG = "static std::vector<zone_moe_mmvq_operand> moe_mmvq_scratch_operands("
OPERAND_EXPERTS = "item.type == GGML_TYPE_MXFP4 ? item.ne[2] : 0"
OPERAND_PUSH = ("ops.push_back({ item.ne[0], role == expert_tensor_role::DOWN || role == expert_tensor_role::CHUNK_DOWN, "
                + OPERAND_EXPERTS + " });")
GROUP_ROWS_ASSERT = ("static_assert(static_cast<size_t>(k_zone_mxfp4_moe_group_rows) == GGML_SYCL_MXFP4_MOE_XMX_N, "
                     "\"zone-sizing.hpp MXFP4 group rows drifted\");")


def claim_mxfp4_operands_name_their_experts(uc: str) -> bool:
    c = code(uc)
    body = fn_body(c, OPERANDS_SIG)
    return OPERAND_PUSH in body and body.count("ops.push_back(") == 1 and GROUP_ROWS_ASSERT in c


def test_mxfp4_operands_name_their_expert_count():
    assert claim_mxfp4_operands_name_their_experts(UC), (
        "moe_mmvq_scratch_operands must give an MXFP4 stack's operand its expert count (ne2), which sizes the grouped "
        "dispatch's per-op tables, and no other type's; and the tables' rows a chunk must stay pinned to "
        "GGML_SYCL_MXFP4_MOE_XMX_N")


def test_witness_operands_drop_the_expert_count():
    m = mutate_in(UC, OPERANDS_SIG, OPERAND_EXPERTS, "0")
    assert not claim_mxfp4_operands_name_their_experts(m)


def test_witness_every_type_names_its_experts():
    m = mutate_in(UC, OPERANDS_SIG, OPERAND_EXPERTS, "item.ne[2]")
    assert not claim_mxfp4_operands_name_their_experts(m)


def test_witness_group_rows_unpinned():
    c = code(UC)
    m = c.replace(GROUP_ROWS_ASSERT, "", 1)
    assert m != c and not claim_mxfp4_operands_name_their_experts(m)


# ---- the ensure sizes the SCRATCH zone from the bound, under the budget quarter and the reserve's room ----

ENSURE_SIG = ("bool unified_cache::ensure_planned_arena_zones(zone_arena_refusal * refusal, "
              "const zone_arena_ensure_options & options)")
QUARTER = "const size_t scratch_budget_cap = available_budget() / 4;"
NONFA_CLAMP = "if (planned_nonfa_attn_scratch > scratch_budget_cap) {"
NONFA_CLAMPED = "planned_nonfa_attn_scratch = scratch_budget_cap;"
NONFA_TERM = "scratch_nonfa = planned_nonfa_attn_scratch;"
ONEDNN_ZONE = "size_t onednn_zone ="
RUNTIME_ZONE = "!zone_runtime_reserved_bytes(runtime_floor, planned_runtime_scratch, &runtime_zone)"
REQUIRED = "const size_t scratch_required = zone_scratch_required_bytes(scratch_zone, scratch_nonfa);"
OTHER_TAIL = ("if (zone_runtime_reserved_bytes(0, runtime_zone, &scratch_other_tail) && "
              "scratch_other_tail <= SIZE_MAX - zone_arena_zone_bytes(onednn_zone)) {")
OTHER_ONEDNN = "scratch_other_tail += zone_arena_zone_bytes(onednn_zone);"
ROOM_SHAPE = "zone_arena_reserve_shape_of(budget_, device_total_vram_, max_alloc, safe_max_alloc)"
ROOM = f"scratch_room = zone_arena_scratch_room({ROOM_SHAPE}, scratch_other_tail, scratch_required);"
BOUND = "const size_t moe_mmvq_bound = unified_cache_get_planned_moe_mmvq_scratch_bytes(dev_id);"
PLAN_ARGS = "moe_mmvq_bound, scratch_budget_cap, scratch_room);"
ZONE_PLAN = ("const zone_scratch_plan scratch_plan = zone_scratch_zone_plan(scratch_zone, scratch_nonfa, "
             + PLAN_ARGS)
CLAMP_WARN = ("if (scratch_plan.clamp != zone_scratch_clamp::NONE) { scratch_zone_clamp_warn(dev_id, scratch_plan, "
              "moe_mmvq_bound, scratch_budget_cap, scratch_room); }")
ZONE_WANTED = "scratch_zone = scratch_plan.wanted;"
RESERVE_CALL = ("if (!arena_reserve(queue_, budget_, max_alloc, safe_max_alloc, scratch_zone, onednn_zone, "
                "runtime_zone, device_total_vram_)) {")


def claim_zone_sized_from_the_bound(uc: str) -> bool:
    """The room is computed after the ONEDNN and RUNTIME zones, on what the zone must hold, from the reserve's own
    arguments, and both limits reach the rule."""
    body = fn_body(code(uc), ENSURE_SIG)
    return ordered(body, NONFA_TERM, ONEDNN_ZONE, RUNTIME_ZONE, REQUIRED, OTHER_TAIL, OTHER_ONEDNN, ROOM, BOUND,
                   ZONE_PLAN, ZONE_WANTED, RESERVE_CALL)


def claim_ensure_passes_the_nonfa_cap(uc: str) -> bool:
    """One quarter of the available budget, read once, holds the non-FA raise and the MoE bound."""
    body = fn_body(code(uc), ENSURE_SIG)
    return (body.count(QUARTER) == 1 and body.count("scratch_budget_cap =") == 1
            and ordered(body, QUARTER, NONFA_CLAMP, NONFA_CLAMPED, ZONE_PLAN))


def test_the_zone_is_sized_from_the_bound():
    assert claim_zone_sized_from_the_bound(UC), (
        "the ensure does not size the SCRATCH zone as zone_scratch_zone_plan's wanted figure over the default, the "
        "non-FA attention raise and the published MoE batched MMVQ bound, held to the room the arena's reserve "
        "accepts beside the ONEDNN and RUNTIME zones, computed from the arguments the reserve is called with")


def test_the_ensure_passes_the_nonfa_cap():
    assert claim_ensure_passes_the_nonfa_cap(UC), (
        "the ensure must hold the MoE batched MMVQ bound to the same budget quarter the non-FA raise is clamped to")


def test_witness_zone_ignores_the_bound():
    m = mutate_in(UC, ENSURE_SIG, ZONE_WANTED, "scratch_zone = scratch_plan.required;")
    assert not claim_zone_sized_from_the_bound(m)


def test_witness_zone_plan_reads_no_bound():
    m = mutate_in(UC, ENSURE_SIG, PLAN_ARGS, "0, scratch_budget_cap, scratch_room);")
    assert not claim_zone_sized_from_the_bound(m)


def test_witness_zone_plan_drops_the_nonfa_raise():
    m = mutate_in(UC, ENSURE_SIG, NONFA_TERM, "")
    assert not claim_zone_sized_from_the_bound(m)


def test_witness_zone_plan_has_no_room_limit():
    m = mutate_in(UC, ENSURE_SIG, PLAN_ARGS, "moe_mmvq_bound, scratch_budget_cap, SIZE_MAX);")
    assert not claim_zone_sized_from_the_bound(m)


def test_witness_room_from_the_live_budget():
    m = mutate_in(UC, ENSURE_SIG, ROOM_SHAPE, ROOM_SHAPE.replace("budget_,", "available_budget(),"))
    assert not claim_zone_sized_from_the_bound(m)


def test_witness_room_ignores_the_onednn_zone():
    m = mutate_in(UC, ENSURE_SIG, OTHER_ONEDNN, "")
    assert not claim_zone_sized_from_the_bound(m)


def test_witness_room_on_the_default_alone():
    m = mutate_in(UC, ENSURE_SIG, "scratch_other_tail, scratch_required);", "scratch_other_tail, scratch_zone);")
    assert not claim_zone_sized_from_the_bound(m)


def test_witness_room_before_the_runtime_zone():
    c = code(UC)
    m = c.replace(" " + REQUIRED, "", 1).replace(NONFA_TERM, NONFA_TERM + " " + REQUIRED, 1)
    assert m != c and not claim_zone_sized_from_the_bound(m)


def test_witness_zone_plan_without_the_quarter():
    m = mutate_in(UC, ENSURE_SIG, PLAN_ARGS, "moe_mmvq_bound, SIZE_MAX, scratch_room);")
    assert not claim_ensure_passes_the_nonfa_cap(m)


def test_witness_zone_plan_with_a_second_fraction():
    m = mutate_in(UC, ENSURE_SIG, PLAN_ARGS, "moe_mmvq_bound, available_budget() / 2, scratch_room);")
    assert not claim_ensure_passes_the_nonfa_cap(m)


def test_witness_nonfa_with_its_own_fraction():
    m = mutate_in(UC, ENSURE_SIG, NONFA_CLAMP, "if (planned_nonfa_attn_scratch > available_budget() / 2) {")
    assert not claim_ensure_passes_the_nonfa_cap(m)


# ---- a clamp of the bound says so once, naming the limit and the miss path ----

CLAMP_SIG = "void unified_cache::scratch_zone_clamp_warn("
CLAMP_GATE = "if (!scratch_zone_warn_latch(scratch_zone_clamp_warn_bytes_, bound_bytes)) { return; }"
CLAMP_LIMIT = ('room ? "the room the arena\'s reserve accepts holds it" : '
               '"a quarter of the available budget holds it"')
CLAMP_FIGURES = ('"bound: %s (%.1f MB; a quarter of the available budget is %.1f MB, the arena\'s reserve accepts a '
                 '%.1f MB "')
MISS_PATH = "by name (q8-scratch-alloc) and the op takes its fallback"
LATCH_SIG = "static bool scratch_zone_warn_latch(std::atomic<size_t> & latch, size_t bytes)"
LATCH_DUE = "if (!zone_scratch_warn_due(latched, bytes)) { return false; }"
LATCH_CAS = "} while (!latch.compare_exchange_weak(latched, bytes, std::memory_order_relaxed)); return true;"


def claim_clamp_warns_once(uc: str) -> bool:
    c = code(uc)
    body = fn_body(c, ENSURE_SIG)
    warn = fn_body(c, CLAMP_SIG)
    return (ordered(body, ZONE_PLAN, CLAMP_WARN, ZONE_WANTED)
            and ordered(warn, CLAMP_GATE, "GGML_LOG_WARN(", CLAMP_FIGURES, "the miss path", MISS_PATH, CLAMP_LIMIT)
            and ordered(fn_body(c, LATCH_SIG), LATCH_DUE, LATCH_CAS))


def test_a_clamp_warns_once_naming_the_limit_and_the_miss_path():
    assert claim_clamp_warns_once(UC), (
        "a plan whose budget quarter or reserve room holds the SCRATCH zone below the MoE batched MMVQ bound must log "
        "one latched WARN naming the limit, its figures and the miss path")


def test_witness_clamp_is_silent():
    m = mutate_in(UC, ENSURE_SIG, CLAMP_WARN, "")
    assert not claim_clamp_warns_once(m)


def test_witness_clamp_warns_every_plan():
    m = mutate_in(UC, CLAMP_SIG, CLAMP_GATE, "")
    assert not claim_clamp_warns_once(m)


def test_witness_clamp_names_one_limit():
    m = mutate_in(UC, CLAMP_SIG, CLAMP_LIMIT, '"a quarter of the available budget holds it"')
    assert not claim_clamp_warns_once(m)


def test_witness_clamp_drops_the_miss_path():
    m = mutate_in(UC, CLAMP_SIG, MISS_PATH, "and the op is served")
    assert not claim_clamp_warns_once(m)


def test_witness_latch_never_due():
    m = mutate_in(UC, LATCH_SIG, LATCH_DUE, "")
    assert not claim_clamp_warns_once(m)


# ---- the reserve makes its tail checks through the functions the room is built from ----

RESERVE_SIG = "bool unified_cache::arena_reserve("
R_ZONES = ("scratch_bytes = zone_arena_zone_bytes(scratch_bytes); onednn_bytes = zone_arena_zone_bytes(onednn_bytes); "
           "if (!zone_runtime_reserved_bytes(0, runtime_bytes, &runtime_bytes)) {")
R_CAP = "const size_t per_chunk_cap = zone_arena_per_chunk_cap(max_alloc_size, safe_max_alloc_size);"
R_CLAMP = "if (device_total_vram > 0 && alloc_size > device_total_vram) {"
R_CLAMPED = "alloc_size = device_total_vram;"
R_ROUND = "alloc_size = zone_arena_round_alloc_bytes(alloc_size);"
R_SINGLE = "const bool try_single_chunk = zone_arena_single_chunk_tried(alloc_size, max_alloc_size, per_chunk_cap);"
R_TAIL = "const size_t tail_bytes = onednn_bytes + runtime_bytes + scratch_bytes;"
R_SINGLE_CHECK = "if (!zone_arena_chunk_holds_tail(alloc_size, tail_bytes)) {"
R_CAP_CHECK = "if (!zone_arena_chunk_holds_tail(per_chunk_cap, tail_bytes)) {"
R_KV = "const size_t tail_kv_reserve = zone_arena_tail_kv_reserve(alloc_size);"
R_LAST = ("const size_t last_chunk_size = zone_arena_tail_chunk_bytes(alloc_size, per_chunk_cap, tail_bytes, "
          "tail_kv_reserve);")
R_LAST_CHECK = "if (!zone_arena_chunk_holds_tail(last_chunk_size, tail_bytes)) {"
R_WEIGHT_CHECK = "if (!zone_arena_weight_chunks_fit(alloc_size, last_chunk_size)) {"
SHAPE_SIG = "zone_arena_reserve_shape zone_arena_reserve_shape_of("
SHAPE_ARENA = ("const size_t arena = device_total_vram > 0 && budget_bytes > device_total_vram ? device_total_vram : "
               "budget_bytes;")
SHAPE = (SHAPE_ARENA + " shape.alloc_bytes = zone_arena_round_alloc_bytes(arena); shape.per_chunk_cap = "
         "zone_arena_per_chunk_cap(max_alloc_size, safe_max_alloc_size); shape.single_chunk = "
         "zone_arena_single_chunk_tried(shape.alloc_bytes, max_alloc_size, shape.per_chunk_cap); "
         "shape.tail_kv_reserve = zone_arena_tail_kv_reserve(shape.alloc_bytes);")
SINGLE_SIG = "bool zone_arena_single_chunk_accepts(const zone_arena_reserve_shape & shape, size_t tail_bytes)"
SINGLE_ACCEPTS = "return shape.single_chunk && zone_arena_chunk_holds_tail(shape.alloc_bytes, tail_bytes);"
N_CHUNK_SIG = "bool zone_arena_n_chunk_accepts(const zone_arena_reserve_shape & shape, size_t tail_bytes)"
N_CHUNK_WEIGHT = " && zone_arena_weight_chunks_fit(shape.alloc_bytes, tail_chunk);"
N_CHUNK_ACCEPTS = ("const size_t tail_chunk = zone_arena_tail_chunk_bytes(shape.alloc_bytes, shape.per_chunk_cap, "
                   "tail_bytes, shape.tail_kv_reserve); return zone_arena_chunk_holds_tail(tail_chunk, tail_bytes)"
                   + N_CHUNK_WEIGHT)


def claim_reserve_checks_are_the_rooms(uc: str, zs: str) -> bool:
    """Every size and tail check arena_reserve makes is a call of the zone-sizing function the room is built from, in
    the reserve's order, and the shape the room reads derives the arena as the reserve does."""
    reserve = fn_body(code(uc), RESERVE_SIG)
    z = code(zs)
    return (ordered(reserve, R_ZONES, R_CAP, R_CLAMP, R_CLAMPED, R_ROUND, R_SINGLE, R_TAIL, R_SINGLE_CHECK, R_TAIL,
                    R_CAP_CHECK, R_KV, R_LAST, R_LAST_CHECK, R_WEIGHT_CHECK)
            and reserve.count(R_TAIL) == 2
            and "k_min_shared_bytes) {" not in reserve
            and SHAPE in fn_body(z, SHAPE_SIG)
            and SINGLE_ACCEPTS in fn_body(z, SINGLE_SIG)
            and N_CHUNK_ACCEPTS in fn_body(z, N_CHUNK_SIG))


def test_the_reserve_checks_are_the_rooms():
    assert claim_reserve_checks_are_the_rooms(UC, ZS), (
        "arena_reserve must size its zones and its chunks and make each tail check through the zone-sizing functions "
        "zone_arena_scratch_room is built from, and zone_arena_reserve_shape_of must derive the arena as it does")


RESERVE_WITNESSES = {
    "single-chunk check inline": (R_SINGLE_CHECK, "if (alloc_size < tail_bytes + k_min_shared_bytes) {"),
    "per-chunk cap check inline": (R_CAP_CHECK, "if (per_chunk_cap < tail_bytes + k_min_shared_bytes) {"),
    "tail chunk check inline": (R_LAST_CHECK, "if (last_chunk_size < tail_bytes + k_min_shared_bytes) {"),
    "weight check inline": (R_WEIGHT_CHECK, "if (weight_total < k_min_shared_bytes) {"),
    "single chunk tried inline": (R_SINGLE,
                                  "const bool try_single_chunk = max_alloc_size == 0 || alloc_size <= per_chunk_cap;"),
    "per-chunk cap inline": (R_CAP, "const size_t per_chunk_cap = max_alloc_size;"),
    "tail chunk ignores the arena": (R_LAST, "const size_t last_chunk_size = std::min(tail_bytes + tail_kv_reserve, "
                                             "per_chunk_cap);"),
    "KV share fixed": (R_KV, "const size_t tail_kv_reserve = 128ull * 1024ull * 1024ull;"),
    "arena not rounded": (R_ROUND, ""),
    "zones not rounded": ("scratch_bytes = zone_arena_zone_bytes(scratch_bytes);", ""),
}


def test_witnesses_reserve_condition_inline():
    for name, (old, new) in RESERVE_WITNESSES.items():
        m = mutate_in(UC, RESERVE_SIG, old, new)
        assert not claim_reserve_checks_are_the_rooms(m, ZS), name


def test_witness_shape_skips_the_device_clamp():
    m = mutate_in(ZS, SHAPE_SIG, SHAPE_ARENA, "const size_t arena = budget_bytes;")
    assert not claim_reserve_checks_are_the_rooms(UC, m)


def test_witness_n_chunk_skips_the_weight_chunks():
    m = mutate_in(ZS, N_CHUNK_SIG, N_CHUNK_WEIGHT, ";")
    assert not claim_reserve_checks_are_the_rooms(UC, m)


# ---- a blocked rebuild keeps the zone and says so ----

KEEP_BINDS = "{ bind_compute_arena(); } #if GGML_SYCL_DNNL onednn_graph_direct_cap_snapshot_keep(); #endif };"
KEEP_DEF = ("const auto keep_arena_as_built = [&]() { if (!compute_arena_ptr_ || !vram_owns(compute_arena_ptr_)) "
            + KEEP_BINDS)
KEPT = "const size_t scratch_kept = zone_scratch_kept_bytes(scratch_plan, arena_zones.scratch);"
BLOCKED = "if (scratch_kept < scratch_zone && !zone_arena_rebuild_allowed(holders)) {"
OTHERS_HOLD = "const bool others_hold = arena_zones.onednn >= onednn_zone && arena_zones.runtime >= runtime_zone;"
KEEP_AT = "scratch_zone = scratch_kept;"
KEEP_BIND = "if (others_hold) { keep_arena_as_built(); onednn_zone_plan_keep_and_store(live_plan);"
KEEP_RETURN = "scratch_zone_kept_warn(dev_id, scratch_plan, scratch_kept); return true;"
HELD = "if (!zone_arena_rebuild_allowed(holders)) {"
SHORT_KEEP_RETURN = ("if (scratch_zone < scratch_plan.wanted) { scratch_zone_kept_warn(dev_id, scratch_plan, "
                     "scratch_zone); } return true;")
DESTROY = "if (!arena_destroy()) {"
WARN_SIG = "void unified_cache::scratch_zone_kept_warn(int dev_id, const zone_scratch_plan & plan, size_t kept_bytes)"
WARN_GATE = "if (!scratch_zone_warn_latch(scratch_zone_kept_warn_bytes_, plan.wanted)) { return; }"


def claim_blocked_rebuild_keeps_the_zone(uc: str) -> bool:
    c = code(uc)
    body = fn_body(c, ENSURE_SIG)
    warn = fn_body(c, WARN_SIG)
    return (ordered(body, KEEP_DEF, KEPT, BLOCKED, OTHERS_HOLD, KEEP_AT, KEEP_BIND, KEEP_RETURN, HELD,
                    SHORT_KEEP_RETURN, DESTROY)
            and ordered(warn, WARN_GATE, "GGML_LOG_WARN(", "the miss path", MISS_PATH)
            and ordered(fn_body(c, LATCH_SIG), LATCH_DUE, LATCH_CAS))


def test_a_blocked_rebuild_keeps_the_zone_and_says_so_once():
    assert claim_blocked_rebuild_keeps_the_zone(UC), (
        "a rebuild that live allocations block must keep a SCRATCH zone short of the MoE batched MMVQ bound alone when "
        "the ONEDNN and RUNTIME zones hold their plan, bind the compute arena, keep the ONEDNN plan, warn once per "
        "wanted size naming the miss path, and return true before any arena teardown")


def test_witness_blocked_keep_refuses():
    m = mutate_in(UC, ENSURE_SIG, KEEP_RETURN, "scratch_zone_kept_warn(dev_id, scratch_plan, scratch_kept); "
                  "return false;")
    assert not claim_blocked_rebuild_keeps_the_zone(m)


def test_witness_blocked_keep_still_asks_for_the_bound():
    m = mutate_in(UC, ENSURE_SIG, KEEP_AT, "")
    assert not claim_blocked_rebuild_keeps_the_zone(m)


def test_witness_short_onednn_keep_drops_the_scratch_warn():
    m = mutate_in(UC, ENSURE_SIG, SHORT_KEEP_RETURN, "return true;")
    assert not claim_blocked_rebuild_keeps_the_zone(m)


def test_witness_warn_every_keep():
    m = mutate_in(UC, WARN_SIG, WARN_GATE, "")
    assert not claim_blocked_rebuild_keeps_the_zone(m)


def test_witness_scratch_keep_leaves_the_compute_arena_unbound():
    m = mutate_in(UC, ENSURE_SIG, KEEP_BIND, "if (others_hold) { onednn_zone_plan_keep_and_store(live_plan);")
    assert not claim_blocked_rebuild_keeps_the_zone(m)


def test_witness_kept_arena_never_binds():
    m = mutate_in(UC, ENSURE_SIG, KEEP_BINDS, KEEP_BINDS.replace("bind_compute_arena(); ", "", 1))
    assert not claim_blocked_rebuild_keeps_the_zone(m)


def test_witness_scratch_keep_stores_the_smaller_onednn_plan():
    m = mutate_in(UC, ENSURE_SIG, KEEP_BIND, KEEP_BIND.replace("_keep_and_store(", "_store(", 1))
    assert not claim_blocked_rebuild_keeps_the_zone(m)


def test_witness_scratch_keep_ignores_a_short_runtime_zone():
    m = mutate_in(UC, ENSURE_SIG, OTHERS_HOLD, "const bool others_hold = arena_zones.onednn >= onednn_zone;")
    assert not claim_blocked_rebuild_keeps_the_zone(m)


# ---- a ring that alone holds the arena defers the SCRATCH keep, and the load's entry recovers ----

SCRATCH_DEFER = ("if (others_hold && options.ring_release_first && zone_arena_only_ring_holds(true, holders)) { if "
                 "(refusal != nullptr) { refusal->reason = zone_arena_refusal_reason::REBUILD_HELD; refusal->device = "
                 "dev_id; refusal->need = { scratch_zone, onednn_zone, runtime_zone }; refusal->have = arena_zones; "
                 "refusal->holders = holders; refusal->keep_deferred = true; } return false; }")
ENTRY_SIG = "static bool ggml_sycl_ensure_load_arena_zones(int device, ggml_sycl::zone_arena_refusal & refusal)"
ENTRY_CALL = "ggml_sycl::unified_cache_ensure_planned_arena_zones(device, &refusal, options)"
ENTRY_STEPS = (
    "options.ring_release_first = true;",
    "if (" + ENTRY_CALL + ") { return true; }",
    "if (!ggml_sycl::zone_arena_only_ring_holds(refusal.reason == ggml_sycl::zone_arena_refusal_reason::REBUILD_HELD, "
    "refusal.holders)) { return false; }",
    "options.ring_release_first = false;",
    "if (cache == nullptr || !cache->release_pp_moe_onednn_scratch_ring()) {",
    "if (!refusal.keep_deferred) { return false; } refusal = {}; return " + ENTRY_CALL + "; }",
    "refusal = {}; return " + ENTRY_CALL + ";",
)


def claim_entry_recovers_a_deferred_scratch_keep(uc: str, gs: str) -> bool:
    """The SCRATCH keep's deferred refusal carries what the entry tests: REBUILD_HELD, the holders, keep_deferred.
    The entry then releases the ring or, when the ring stays, asks again without ring_release_first."""
    body = fn_body(code(uc), ENSURE_SIG)
    return (ordered(body, OTHERS_HOLD, SCRATCH_DEFER, KEEP_AT)
            and ordered(fn_body(code(gs), ENTRY_SIG), *ENTRY_STEPS))


def test_the_entry_recovers_a_deferred_scratch_keep():
    assert claim_entry_recovers_a_deferred_scratch_keep(UC, GS), (
        "a SCRATCH keep a lone ring defers must refuse with REBUILD_HELD, the holders and keep_deferred, and the "
        "load's entry ensure must release that ring or ask again without ring_release_first")


def test_witness_deferred_scratch_keep_names_no_holders():
    m = mutate_in(UC, ENSURE_SIG, "refusal->holders = holders; refusal->keep_deferred = true; } return false; }",
                  "refusal->keep_deferred = true; } return false; }")
    assert not claim_entry_recovers_a_deferred_scratch_keep(m, GS)


# ---- the context re-sizes the bound and never lowers it ----

TXN_SIG = "static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction("
GRAPH_CHECK = "if (!ggml_sycl_check_onednn_graph_scratch(ctx->device, next_plan, flash_attn_enabled, probe_mode,"
REPLAN_CALL = ("if (!ggml_sycl_replan_moe_mmvq_scratch(ctx->device, *current->plan, next_plan, probe_mode)) { "
               "return refuse(")
REPLAN_SIG = "static bool ggml_sycl_replan_moe_mmvq_scratch("
REPLAN_UBATCH = "const uint32_t n_ubatch = next_plan.context_n_ubatch;"
REPLAN_BOUND = "if (!ggml_sycl::zone_moe_mmvq_scratch_bytes(next_plan.moe_mmvq_scratch_ops, n_ubatch,"
REPLAN_GRAPH = "next_plan.moe_mmvq_scratch_bytes = scratch.graph;"
NEVER_BELOW = ("if (current.moe_mmvq_scratch_bytes >= next_plan.moe_mmvq_scratch_bytes) { "
               "next_plan.moe_mmvq_scratch_bytes = current.moe_mmvq_scratch_bytes;")
PROBE_RETURN = "if (probe_mode) { return true; }"
RAISE = "const size_t published = ggml_sycl::unified_cache_raise_planned_moe_mmvq_scratch_bytes(device, scratch.graph);"
ENSURE_AGAIN = "(void) ggml_sycl::unified_cache_ensure_planned_arena_zones(device, &arena_refusal);"
SETTER = "unified_cache_set_planned_moe_mmvq_scratch_bytes("
RAISE_SIG = "size_t unified_cache_raise_planned_moe_mmvq_scratch_bytes(int device_id, size_t bytes)"
RAISE_LOOP = "while (held < bytes && !term.compare_exchange_weak(held, bytes, std::memory_order_acq_rel)) { }"
RAISE_RESULT = "return std::max(held, bytes);"


def claim_context_raises_the_bound(gs: str, uc: str) -> bool:
    c = code(gs)
    txn = fn_body(c, TXN_SIG)
    replan = fn_body(c, REPLAN_SIG)
    raise_body = fn_body(code(uc), RAISE_SIG)
    return (ordered(txn, GRAPH_CHECK, REPLAN_CALL)
            and ordered(replan, REPLAN_UBATCH, REPLAN_BOUND, REPLAN_GRAPH, NEVER_BELOW, PROBE_RETURN, RAISE,
                        ENSURE_AGAIN)
            and SETTER not in replan
            and ordered(raise_body, RAISE_LOOP, RAISE_RESULT) and ".store(" not in raise_body)


def test_the_context_raises_the_bound_and_never_lowers_it():
    assert claim_context_raises_the_bound(GS, UC), (
        "the runtime context transaction must re-size the MoE batched MMVQ bound at the context's micro-batch after "
        "the oneDNN Graph check, keep the plan's figure when it is larger, raise the published bound only outside a "
        "probe and ask for the zones again; the raise must never lower the bound")


def test_witness_context_never_replans():
    m = mutate_in(GS, TXN_SIG, REPLAN_CALL, "if (false) { return refuse(")
    assert not claim_context_raises_the_bound(m, UC)


def test_witness_context_lowers_the_bound():
    m = mutate_in(GS, REPLAN_SIG, NEVER_BELOW, "if (false) { next_plan.moe_mmvq_scratch_bytes = "
                  "current.moe_mmvq_scratch_bytes;")
    assert not claim_context_raises_the_bound(m, UC)


def test_witness_probe_publishes():
    m = mutate_in(GS, REPLAN_SIG, PROBE_RETURN, "")
    assert not claim_context_raises_the_bound(m, UC)


def test_witness_context_stores_the_bound_plainly():
    m = mutate_in(GS, REPLAN_SIG, RAISE, "ggml_sycl::unified_cache_set_planned_moe_mmvq_scratch_bytes(device, "
                  "scratch.graph); " + RAISE)
    assert not claim_context_raises_the_bound(m, UC)


def test_witness_context_sizes_at_the_planning_micro_batch():
    m = mutate_in(GS, REPLAN_SIG, REPLAN_UBATCH, "const uint32_t n_ubatch = next_plan.planner_n_ubatch;")
    assert not claim_context_raises_the_bound(m, UC)


def test_witness_raise_lowers():
    m = mutate_in(UC, RAISE_SIG, RAISE_LOOP, "term.store(bytes, std::memory_order_release);")
    assert not claim_context_raises_the_bound(GS, m)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
