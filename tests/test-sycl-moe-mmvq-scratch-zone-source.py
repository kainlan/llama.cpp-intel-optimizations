"""Source contract for the SCRATCH zone's MoE batched MMVQ scratch bound.

Each MUL_MAT_ID op that mmvq_moe_batched_dispatch runs takes a Q8_1 copy of its activations and an ids table from the
device SCRATCH zone and hands both to retain_handles_until_event. A detached drain thread releases them once the op's
completion event fires; no later op's allocation waits for that release, and the zone's allocator reclaims nothing on
its own. So nothing bounds how many ops hold their scratch at once, and the zone is sized for every op's buffers
together: zone_moe_mmvq_scratch_bytes's graph figure, at the micro-batch the context runs. The pure rules
(zone_scratch_zone_plan, zone_scratch_kept_bytes, zone_kept_scratch_warn_due) are checked by test-zone-sizing Case 47.
These checks pin how the cache and the backend use them:

The planner computes the bound at the context's micro-batch (the load's planning micro-batch until a context names
one), stores the graph figure in the plan, and publishes it to every device the plan places. Its line names the plan's
tensor count, the micro-batch and the depth.

The ensure sizes the SCRATCH zone from zone_scratch_zone_plan over the arena's default, the non-FA attention raise and
the published bound.

A rebuild that something live blocks keeps a SCRATCH zone short of the bound alone, says so through the latched WARN
and returns true before any teardown of the arena; it is never a new refusal.

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
            and body.count("plan.moe_mmvq_scratch_bytes =") == 2)


def test_the_plan_publishes_the_bound_at_the_context_micro_batch():
    assert claim_plan_publishes_the_bound(UC), (
        "populate_host_zone_sizing does not size the MoE batched MMVQ scratch at the context's micro-batch, store the "
        "every-op (graph) figure, name its tensor count, micro-batch and depth, and publish it to every device")


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


# ---- the ensure sizes the SCRATCH zone from the bound ----

ENSURE_SIG = ("bool unified_cache::ensure_planned_arena_zones(zone_arena_refusal * refusal, "
              "const zone_arena_ensure_options & options)")
NONFA_TERM = "scratch_nonfa = planned_nonfa_attn_scratch;"
ZONE_PLAN = ("const zone_scratch_plan scratch_plan = zone_scratch_zone_plan(scratch_zone, scratch_nonfa, "
             "unified_cache_get_planned_moe_mmvq_scratch_bytes(dev_id));")
ZONE_WANTED = "scratch_zone = scratch_plan.wanted;"
ONEDNN_ZONE = "size_t onednn_zone ="


def claim_zone_sized_from_the_bound(uc: str) -> bool:
    body = fn_body(code(uc), ENSURE_SIG)
    return ordered(body, NONFA_TERM, ZONE_PLAN, ZONE_WANTED, ONEDNN_ZONE)


def test_the_zone_is_sized_from_the_bound():
    assert claim_zone_sized_from_the_bound(UC), (
        "the ensure does not size the SCRATCH zone as zone_scratch_zone_plan's wanted figure over the default, the "
        "non-FA attention raise and the published MoE batched MMVQ bound")


def test_witness_zone_ignores_the_bound():
    m = mutate_in(UC, ENSURE_SIG, ZONE_WANTED, "scratch_zone = scratch_plan.required;")
    assert not claim_zone_sized_from_the_bound(m)


def test_witness_zone_plan_reads_no_bound():
    m = mutate_in(UC, ENSURE_SIG, "unified_cache_get_planned_moe_mmvq_scratch_bytes(dev_id));", "0);")
    assert not claim_zone_sized_from_the_bound(m)


def test_witness_zone_plan_drops_the_nonfa_raise():
    m = mutate_in(UC, ENSURE_SIG, NONFA_TERM, "")
    assert not claim_zone_sized_from_the_bound(m)


# ---- a blocked rebuild keeps the zone and says so ----

KEPT = "const size_t scratch_kept = zone_scratch_kept_bytes(scratch_plan, arena_zones.scratch);"
BLOCKED = "if (scratch_kept < scratch_zone && !zone_arena_rebuild_allowed(holders)) {"
KEEP_AT = "scratch_zone = scratch_kept;"
KEEP_RETURN = "scratch_zone_kept_warn(dev_id, scratch_plan, scratch_kept); return true;"
HELD = "if (!zone_arena_rebuild_allowed(holders)) {"
SHORT_KEEP_RETURN = ("if (scratch_zone < scratch_plan.wanted) { scratch_zone_kept_warn(dev_id, scratch_plan, "
                     "scratch_zone); } return true;")
DESTROY = "if (!arena_destroy()) {"
WARN_SIG = "void unified_cache::scratch_zone_kept_warn(int dev_id, const zone_scratch_plan & plan, size_t kept_bytes)"
WARN_DUE = "if (!zone_kept_scratch_warn_due(latched, plan.wanted)) { return; }"
WARN_LATCH = "scratch_zone_kept_warn_bytes_.compare_exchange_weak(latched, plan.wanted,"


def claim_blocked_rebuild_keeps_the_zone(uc: str) -> bool:
    c = code(uc)
    body = fn_body(c, ENSURE_SIG)
    warn = fn_body(c, WARN_SIG)
    return (ordered(body, KEPT, BLOCKED, KEEP_AT, KEEP_RETURN, HELD, SHORT_KEEP_RETURN, DESTROY)
            and ordered(warn, WARN_DUE, WARN_LATCH, "GGML_LOG_WARN("))


def test_a_blocked_rebuild_keeps_the_zone_and_says_so_once():
    assert claim_blocked_rebuild_keeps_the_zone(UC), (
        "a rebuild that live allocations block must keep a SCRATCH zone short of the MoE batched MMVQ bound alone, "
        "warn once per wanted size, and return true before any arena teardown")


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
    m = mutate_in(UC, WARN_SIG, WARN_DUE, "")
    assert not claim_blocked_rebuild_keeps_the_zone(m)


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
