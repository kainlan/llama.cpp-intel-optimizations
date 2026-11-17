"""Source contract for a placement plan that is tried and discarded: it publishes nothing.

A load that names a micro-batch above its planning shape first plans with the room that micro-batch's PP MoE oneDNN
ring needs. When the arena cannot grow for that room, the load falls back to its planning shape: it publishes the ring
and the dense scratch at that shape again, sizes the zones, and plans without the room. The plan tried with the room is
discarded, but while it was computed the planner published terms of its own (the MoE host scatter scratch for the
experts it moved to host, the MoE CONTROL requirement, the oneDNN Graph and non-FA attention scratch shapes). The
fallback used to size the zones with those terms still published, so it asked a zone that could not grow for more than
the zone held, and the load aborted.

These checks pin the fix:

The RUNTIME zone's requirement is the sum of its terms as published, computed by the pure zone_runtime_requirement
(test-zone-sizing Case 37 checks the arithmetic, the run's 92160 bytes included), which the cache's getter feeds.

One helper, ggml_sycl::planner_published_terms (declared in unified-cache.hpp, defined in unified-cache.cpp beside the
getters and setters it calls), snapshots and restores every term the planner publishes: every
unified_cache_set_planned_* call in the planner's own functions has its getter in the snapshot and its setter in the
restore, fed from the field the snapshot filled. The backend keeps no copy of its own. The plan dry run, which also
discards the plans it makes, restores through the same helper and keeps no list of its own. test-sycl-runtime-alloc's
planner-terms case checks the restore against the cache's own state.

The scan for the planner's terms reads a fixed list of function bodies in unified-cache.cpp (PLANNER_SIGS):
populate_host_zone_sizing, publish_moe_host_scatter_term, record_host_ledger_plan, placement_plan_size_host_side,
compute_placement_plan and compute_multi_device_plan. It does not follow the calls those bodies make, so a setter
reached only through a function outside the list is not seen: a planner function that starts to publish a term joins
the list. PLANNER_TERMS_TODAY keeps a scan that reads the wrong bodies from passing by finding nothing.

The stage takes the snapshot after the entry ensure and before the plan with the room, and its fallback restores it
after it publishes the planning shape again and before it sizes the zones.

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
UCH = (SYCL / "unified-cache.hpp").read_text()
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


def definition_bodies(text: str, sig: str) -> list:
    """The body of every definition that starts with `sig`; a declaration (a ';' before any '{') is skipped."""
    bodies, pos = [], 0
    while True:
        s = text.find(sig, pos)
        if s == -1:
            return bodies
        pos = s + len(sig)
        semi, brace = text.find(";", pos), text.find("{", pos)
        if brace != -1 and (semi == -1 or brace < semi):
            bodies.append(block_at(text, pos))


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


def mutate(src: str, old: str, new: str) -> str:
    """`src` as code() with the first `old` replaced by `new`."""
    c = code(src)
    assert old in c, f"witness anchor {old!r} is not in the source"
    return c.replace(old, new, 1)


def call_args(text: str, start: int) -> list:
    """The top-level arguments of the call whose '(' is the first one at or after `start`."""
    o = text.find("(", start)
    depth, i, args, cur = 0, o, [], []
    while i < len(text):
        c = text[i]
        if c in "([{":
            depth += 1
            if depth == 1:
                i += 1
                continue
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                args.append("".join(cur).strip())
                return [a for a in args if a]
        elif c == "," and depth == 1:
            args.append("".join(cur).strip())
            cur = []
            i += 1
            continue
        cur.append(c)
        i += 1
    return []


# ---- the RUNTIME requirement is the pure sum of its published terms ----

REQ_SIG = "bool unified_cache_get_planned_runtime_zone_requirement(int device_id, size_t * out)"
PURE_SIG = "bool zone_runtime_requirement(const zone_runtime_terms & terms, size_t * out)"
# Each term the getter reads, as the field it hands the pure sum.
REQ_FIELDS = (
    "terms.base = base;",
    "terms.moe_host_scatter = moe_host_scatter;",
    "terms.mmq_src1 = mmq_src1;",
    "terms.dequant_f16 = dequant_f16;",
    "terms.compute = compute;",
    "terms.state = state;",
)
REQ_RETURN = "return zone_runtime_requirement(terms, out);"
PURE_ADDED = (
    "const size_t added[] = { terms.moe_host_scatter, terms.mmq_src1, terms.dequant_f16, terms.compute, terms.state };"
)
PURE_LOOP = (
    "size_t total = terms.base;",
    "for (const size_t term : added) { if (term > SIZE_MAX - total) { return false; } total += term; }",
    "*out = total; return true;",
)


def claim_requirement_is_the_pure_sum(uc: str, zs: str) -> bool:
    """The cache's getter reads every RUNTIME term into zone_runtime_terms and returns the pure helper's answer; it
    adds nothing itself. The helper starts from the base and adds every other term, each addition checked."""
    req = fn_body(code(uc), REQ_SIG)
    pure = fn_body(code(zs), PURE_SIG)
    return (
        bool(req)
        and bool(pure)
        and ordered(req, "zone_runtime_terms terms;", *REQ_FIELDS, REQ_RETURN)
        and "*out" not in req
        and ordered(pure, "if (out == nullptr) { return false; }", PURE_ADDED, *PURE_LOOP)
    )


# ---- one helper snapshots and restores every term the planner publishes ----

# A struct's signature stops before its brace: fn_body() opens at the first brace after the signature.
HELPER_SIG = "struct planner_published_terms"
HELPER_DECLS = (
    "static planner_published_terms snapshot(int device);",
    "void restore(int device) const;",
)
SNAPSHOT_SIG = "planner_published_terms planner_published_terms::snapshot(int device)"
RESTORE_SIG = "void planner_published_terms::restore(int device) const"
# The planner functions whose bodies the scan reads: the ones that publish a planned term today when
# compute_placement_plan or compute_multi_device_plan runs. The scan reads these bodies only and follows no call.
PLANNER_SIGS = (
    "static void populate_host_zone_sizing(",
    "static void publish_moe_host_scatter_term(",
    "static void record_host_ledger_plan(",
    "static void placement_plan_size_host_side(",
    "placement_plan compute_placement_plan(",
    "placement_plan compute_multi_device_plan(",
)
SETTER_RE = re.compile(r"\bunified_cache_set_planned_(\w+)\(")
GETTER_INTO_RE = re.compile(r"\bterms\.(\w+) = (?:ggml_sycl::)?unified_cache_get_planned_(\w+)\(device\);")
# The terms the planner publishes today; the set the scan finds must hold them, so a scan that read the wrong bodies
# cannot pass by finding nothing.
PLANNER_TERMS_TODAY = {
    "moe_host_scatter_scratch_bytes",
    "moe_control_requirement",
    "onednn_graph_scratch_shape",
    "nonfa_attn_scratch_shape",
}


def planner_published(uc: str) -> set:
    c = code(uc)
    names = set()
    for sig in PLANNER_SIGS:
        for body in definition_bodies(c, sig):
            names.update(SETTER_RE.findall(body))
    return names


def claim_helper_restores_every_planner_term(uc: str, uch: str, gs: str) -> bool:
    """Every term the planner publishes is read by the helper's snapshot into a field and written back by its restore
    from that field, through the term's own getter and setter. The helper is declared in the cache's header and defined
    in the cache, and the backend defines no helper of its own."""
    published = planner_published(uc)
    helper = fn_body(code(uch), HELPER_SIG)
    c = code(uc)
    snap = fn_body(c, SNAPSHOT_SIG)
    restore = fn_body(c, RESTORE_SIG)
    if not (published >= PLANNER_TERMS_TODAY and snap and restore and all(d in helper for d in HELPER_DECLS)):
        return False
    if re.search(r"\bstruct \w*planner_published_terms\b", code(gs)):
        return False
    field_of = {term: field for field, term in GETTER_INTO_RE.findall(snap)}
    for term in published:
        field = field_of.get(term)
        call = f"unified_cache_set_planned_{term}("
        at = restore.find(call)
        if field is None or at == -1:
            return False
        args = call_args(restore, at)
        if not args or args[0] != "device" or not all(a == field or a.startswith(field + ".") for a in args[1:]):
            return False
    return True


# ---- the stage snapshots before the plan with the room and restores in the fallback before the ensure ----

COMPUTE_SIG = "static bool compute_and_store_plan_for_inventory(ggml_backend_sycl_context * ctx, size_t vram_budget,"
ENTRY_ENSURE = "!ggml_sycl::unified_cache_ensure_planned_arena_zones(ctx->device, &refusal)) {"
SNAPSHOT = (
    "const ggml_sycl::planner_published_terms published_before = "
    "ggml_sycl::planner_published_terms::snapshot(ctx->device);"
)
FIRST_PLAN = "plan_candidate = plan_single_device(true);"
FALLBACK = "if (!placed) {"
REPUBLISH = "(void) ggml_sycl_publish_load_ubatch_scratch(ctx->device, g_placement_kv_info.n_ubatch);"
RESTORE = "published_before.restore(ctx->device);"
FALLBACK_ENSURE = "if (!ggml_sycl::unified_cache_ensure_planned_arena_zones(ctx->device, &refusal)) {"
FALLBACK_PLAN = "plan_candidate = plan_single_device(false);"


def claim_snapshot_before_the_plan_with_room(gs: str) -> bool:
    """The snapshot is taken once, after the entry ensure proved the zones hold the terms, and before the first plan
    with the room publishes any."""
    body = fn_body(code(gs), COMPUTE_SIG)
    snap_at = body.find(SNAPSHOT)
    return (
        body.count(SNAPSHOT) == 1
        and ordered(body, ENTRY_ENSURE, SNAPSHOT, FIRST_PLAN)
        and body.find(FIRST_PLAN) > snap_at
        and "plan_single_device(true)" not in body[:snap_at]
    )


def claim_fallback_restores_before_the_ensure(gs: str) -> bool:
    """The fallback publishes the planning shape again, restores the snapshot, and only then sizes the zones and plans
    without the room, which publishes its own terms."""
    body = fn_body(code(gs), COMPUTE_SIG)
    fallback = block_at(body, body.find(FALLBACK))
    return bool(fallback) and ordered(fallback, REPUBLISH, RESTORE, FALLBACK_ENSURE, FALLBACK_PLAN)


# ---- the plan dry run restores through the same helper ----

DRY_SIG = "struct ggml_sycl_plan_dry_run_saved_state"
DRY_FIELD = "ggml_sycl::planner_published_terms planner;"
DRY_SNAPSHOT = "s.planner = ggml_sycl::planner_published_terms::snapshot(d);"
DRY_RESTORE = "s.planner.restore(d);"


def claim_dry_run_restores_through_the_helper(uc: str, gs: str) -> bool:
    """The dry run keeps the planner's terms in the helper and writes them back through it, and calls none of the
    planner's setters itself, so the list of terms lives in one place."""
    dry = fn_body(code(gs), DRY_SIG)
    published = planner_published(uc)
    return (
        bool(dry)
        and DRY_FIELD in dry
        and DRY_SNAPSHOT in dry
        and DRY_RESTORE in dry
        and bool(published)
        and not any(f"unified_cache_set_planned_{term}(" in dry for term in published)
    )


# ---- tests ----


def test_the_runtime_requirement_is_the_pure_sum():
    assert claim_requirement_is_the_pure_sum(UC, ZS)


def test_one_helper_restores_every_term_the_planner_publishes():
    assert claim_helper_restores_every_planner_term(UC, UCH, GS)


def test_the_snapshot_is_taken_before_the_plan_with_the_room():
    assert claim_snapshot_before_the_plan_with_room(GS)


def test_the_fallback_restores_before_it_sizes_the_zones():
    assert claim_fallback_restores_before_the_ensure(GS)


def test_the_dry_run_restores_through_the_helper():
    assert claim_dry_run_restores_through_the_helper(UC, GS)


# ---- mutation witnesses: each applies the regression a check guards and must make it fail ----


def test_witness_getter_sums_its_own_terms():
    m = mutate_in(UC, REQ_SIG, REQ_RETURN, "*out = base + mmq_src1 + dequant_f16 + compute + state; return true;")
    assert not claim_requirement_is_the_pure_sum(m, ZS)


def test_witness_getter_leaves_out_the_host_scatter_term():
    m = mutate_in(UC, REQ_SIG, "terms.moe_host_scatter = moe_host_scatter;", "")
    assert not claim_requirement_is_the_pure_sum(m, ZS)


def test_witness_pure_sum_leaves_out_the_host_scatter_term():
    m = mutate_in(ZS, PURE_SIG, "{ terms.moe_host_scatter, ", "{ ")
    assert not claim_requirement_is_the_pure_sum(UC, m)


def test_witness_pure_sum_adds_unchecked():
    m = mutate_in(ZS, PURE_SIG, "if (term > SIZE_MAX - total) { return false; } ", "")
    assert not claim_requirement_is_the_pure_sum(UC, m)


def test_witness_restore_leaves_the_host_scatter_term_published():
    m = mutate_in(UC, RESTORE_SIG, "unified_cache_set_planned_moe_host_scatter_scratch_bytes(device, scatter);", "")
    assert not claim_helper_restores_every_planner_term(m, UCH, GS)


def test_witness_snapshot_never_reads_the_host_scatter_term():
    m = mutate_in(
        UC,
        SNAPSHOT_SIG,
        "terms.scatter = unified_cache_get_planned_moe_host_scatter_scratch_bytes(device);",
        "terms.scatter = 0;",
    )
    assert not claim_helper_restores_every_planner_term(m, UCH, GS)


def test_witness_restore_writes_a_constant():
    m = mutate_in(
        UC,
        RESTORE_SIG,
        "unified_cache_set_planned_moe_host_scatter_scratch_bytes(device, scatter);",
        "unified_cache_set_planned_moe_host_scatter_scratch_bytes(device, 0);",
    )
    assert not claim_helper_restores_every_planner_term(m, UCH, GS)


def test_witness_helper_no_longer_declared_in_the_cache_header():
    m = mutate_in(UCH, HELPER_SIG, HELPER_DECLS[1], "")
    assert not claim_helper_restores_every_planner_term(UC, m, GS)


def test_witness_backend_keeps_a_helper_of_its_own():
    m = code(GS) + " struct ggml_sycl_planner_published_terms { size_t scatter = 0; };"
    assert not claim_helper_restores_every_planner_term(UC, UCH, m)


def test_witness_planner_publishes_a_term_the_helper_does_not_know():
    sig = "static void populate_host_zone_sizing("
    c = code(UC)
    body = fn_body(c, sig)
    assert body, "witness anchor populate_host_zone_sizing is not in the source"
    s = c.find(body, c.find(sig))
    m = c[:s] + "{ unified_cache_set_planned_new_side_effect_bytes(plan.device_id, 1); " + c[s + 1 :]
    assert not claim_helper_restores_every_planner_term(m, UCH, GS)


def test_witness_snapshot_after_the_plan_with_the_room():
    m = mutate(mutate(GS, SNAPSHOT, ""), FIRST_PLAN, FIRST_PLAN + " " + SNAPSHOT)
    assert not claim_snapshot_before_the_plan_with_room(m)


def test_witness_snapshot_before_the_entry_ensure():
    body = fn_body(code(GS), COMPUTE_SIG)
    assert ENTRY_ENSURE in body, "witness anchor: the entry ensure is not in the stage"
    m = mutate(mutate(GS, SNAPSHOT, ""), "if (ggml_sycl::get_unified_cache_for_device(ctx->device) && " + ENTRY_ENSURE,
               SNAPSHOT + " if (ggml_sycl::get_unified_cache_for_device(ctx->device) && " + ENTRY_ENSURE)
    assert not claim_snapshot_before_the_plan_with_room(m)


def test_witness_fallback_without_the_restore():
    m = mutate_in(GS, COMPUTE_SIG, RESTORE, "")
    assert not claim_fallback_restores_before_the_ensure(m)


def test_witness_fallback_restores_after_the_ensure():
    m = mutate_in(GS, COMPUTE_SIG, RESTORE + " " + FALLBACK_ENSURE, FALLBACK_ENSURE + " " + RESTORE)
    assert not claim_fallback_restores_before_the_ensure(m)


def test_witness_fallback_restores_a_snapshot_of_its_own():
    m = mutate_in(
        GS, COMPUTE_SIG, RESTORE, "ggml_sycl::planner_published_terms::snapshot(ctx->device).restore(ctx->device);"
    )
    assert not claim_fallback_restores_before_the_ensure(m)


def test_witness_dry_run_keeps_its_own_write_back():
    m = mutate_in(GS, DRY_SIG, DRY_RESTORE, "ggml_sycl::unified_cache_set_planned_moe_host_scatter_scratch_bytes(d, 0);")
    assert not claim_dry_run_restores_through_the_helper(UC, m)


def test_witness_dry_run_never_restores():
    m = mutate_in(GS, DRY_SIG, DRY_RESTORE, "")
    assert not claim_dry_run_restores_through_the_helper(UC, m)


# ---- the witnesses' own anchors: a scan that reads nothing must not pass ----


def test_the_planner_scan_finds_the_terms_published_today():
    assert planner_published(UC) >= PLANNER_TERMS_TODAY


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main(["-q", __file__]))
