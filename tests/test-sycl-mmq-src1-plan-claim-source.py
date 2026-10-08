"""Source contract for llama.cpp-g6yk: the planned dense Q8_1 src1 buffer is CLAIMED when its plan is published.

Host-only: reads sources, builds nothing, loads no model and touches no device.

The defect (B70, Qwen3.8 IQ3_XXS, llama-bench -p 512 -ub 512): the RUNTIME zone was sized for the PP MoE oneDNN ring
plus both dense scratch plans (4265 + 3.4 + 66 MB), and the ring re-plan was admitted against that zone less the dense
plans. Every byte was counted. But the planned Q8_1 src1 buffer was never materialized before the first graph: the
graph-entry walk did not count the nodes that drew it, so the first op found "the planned buffer holds 0" and grew it
to its OWN need (1.4 MB), and a later op grew it again (3.4 MB) while the first backing was still retained behind its
queue marker. The two backings together exceed the plan, the zone had 2.6 MB left, and the 479i plan-breach abort
fired. A planned buffer that only exists once an op asks for it is counted, not reserved.

The fix this gate pins: the runtime-context transaction, on its publish path, claims the buffer at its whole plan
right after the plan is committed and before the hold is recomputed, so it holds its planned bytes before any graph
and an op within n_ubatch never grows it. The claim goes through the same unified-cache allocator as every other
planned RUNTIME scratch (RUNTIME zone, spill forbidden). A claim that cannot be met is reported, not repaired: the
hold still keeps the bytes off spill-capable allocations and the graph-entry walk still refuses by name.

It also pins the accounting the ticket suspected was missing and is not: the RUNTIME zone requirement folds in both
dense plans, and the ring re-plan is admitted against the zone's free bytes less the dense plans (pending demand).

Every claim is checked on COMMENT-STRIPPED, whitespace-normalized text and has a mutant that must make it fail.

Pytest-style (module-level test_* functions): register with llama_test_pytest.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_LEXEME_RE = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'",
    re.DOTALL,
)


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def norm(text: str) -> str:
    """Comment-stripped, whitespace-canonical text, so a clang-format re-wrap cannot turn a claim red."""
    t = re.sub(r"\s+", " ", strip_comments(text))
    t = re.sub(r"\( ", "(", t)
    t = re.sub(r" \)", ")", t)
    t = re.sub(r" ,", ",", t)
    t = re.sub(r",(?! )", ", ", t)
    return t


def read(rel: str) -> str:
    return (ROOT / rel).read_text()


SYCL = read("ggml/src/ggml-sycl/ggml-sycl.cpp")
COMMON = read("ggml/src/ggml-sycl/common.hpp")
CACHE = read("ggml/src/ggml-sycl/unified-cache.cpp")


def body(text: str, signature: str) -> str:
    """The brace-balanced body of the first definition whose normalized text starts with `signature` (string
    literals are skipped while balancing). Empty when there is none."""
    at = text.find(signature)
    if at < 0:
        return ""
    open_at = text.find("{", at + len(signature))
    if open_at < 0:
        return ""
    depth = 0
    i = open_at
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "\"'":
            j = i + 1
            while j < n and text[j] != ch:
                j += 2 if text[j] == "\\" else 1
            i = j + 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_at:i + 1]
        i += 1
    return ""


TXN_SIG = "static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction("
CLAIM_SIG = "static void ggml_sycl_mmq_src1_claim_plan(ggml_backend_sycl_context & ctx)"
RING_SIG = "static ggml_sycl_ring_replan_result ggml_sycl_replan_pp_moe_onednn_ring("
ENSURE_SIG = "inline void * ggml_sycl_runtime_scratch_ensure("
REQ_SIG = "bool unified_cache_get_planned_runtime_zone_requirement(int device_id, size_t * out)"


# ---- (a) the plan-time claim ------------------------------------------------------------------------------------


def claim_transaction_claims_after_commit_before_hold(sycl: str) -> bool:
    """The publish path claims the buffer after the plan is committed and the ring is re-planned, and before the
    hold is recomputed (so the hold sees the buffer at its plan, not short of it)."""
    txn = body(norm(sycl), TXN_SIG)
    if not txn:
        return False
    ring = txn.find("ggml_sycl_replan_pp_moe_onednn_ring(ctx->device, next_kv_info.n_ubatch, probe_mode, &ring_kv_zone)")
    commit = txn.find("dense_guard.commit();")
    claim = txn.find("ggml_sycl_mmq_src1_claim_plan(*ctx);")
    if min(ring, commit, claim) < 0:
        return False
    hold = txn.find("ggml_sycl_planned_scratch_hold_refresh(*ctx);", claim)
    # The call sits at the commit's own statement level: nothing between the two may open a block or a condition,
    # or the claim could be made conditional (on probe_mode, say) while still following the commit in the text.
    between = txn[commit + len("dense_guard.commit();"):claim]
    # Exactly one claim, on the publish path only (after the commit, which a probe never reaches).
    return (ring < commit < claim < hold and "{" not in between and "if (" not in between
            and txn.count("ggml_sycl_mmq_src1_claim_plan(") == 1)


def claim_helper_claims_the_whole_plan(sycl: str) -> bool:
    """The helper sizes the claim from the published plan (not from any op's demand), through the context's own
    Q8_1 buffer, and reports a claim it cannot meet instead of aborting or refusing after the publish."""
    helper = body(norm(sycl), CLAIM_SIG)
    if not helper:
        return False
    m = re.search(r"const size_t (\w+) = ggml_sycl::unified_cache_get_planned_mmq_src1_scratch_bytes\((\w+)\);",
                  helper)
    if not m:
        return False
    planned, dev = m.group(1), m.group(2)
    skip = f"if ({planned} == 0 || ctx.mmvq_q8_activation_cache.capacity({dev}) >= {planned}) {{ return; }}"
    ensure = f"ctx.mmvq_q8_activation_cache.ensure_buffer({planned}, {dev}, *ctx.stream({dev}, 0))"
    read_at = m.start()
    skip_at = helper.find(skip, m.end())
    ensure_at = helper.find(ensure, skip_at + len(skip)) if skip_at >= 0 else -1
    if min(skip_at, ensure_at) < 0:
        return False
    # In order: the plan read, the skip for a buffer already at plan, the claim. The skip's own `return` is the only
    # one allowed before the claim, so no early exit can stand before the read or between the skip and the claim.
    before_read = helper[:read_at]
    skip_to_claim = helper[skip_at + len(skip):ensure_at]
    return ("return" not in before_read and "return" not in skip_to_claim and "GGML_LOG_WARN(" in helper
            and "GGML_ABORT" not in helper and "malloc" not in helper)


def claim_runtime_scratch_stays_in_the_runtime_zone(common: str) -> bool:
    """The allocator the claim goes through is the unified cache's, RUNTIME zone, spill forbidden."""
    ensure = body(norm(common), ENSURE_SIG)
    return (bool(ensure) and "ggml_sycl::unified_allocate_owner(req)" in ensure
            and "req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::RUNTIME;" in ensure
            and "req.intent.constraints.forbid_vram_zone_spill = true;" in ensure and "malloc" not in ensure)


# ---- (b) the RUNTIME zone accounting the ring is admitted against -------------------------------------------------


def claim_ring_replan_is_charged_the_dense_plans(sycl: str) -> bool:
    """The transaction measures the dense plans at the runtime n_ubatch and hands them to the ring re-plan as
    pending RUNTIME demand, before the ring is re-planned."""
    txn = body(norm(sycl), TXN_SIG)
    fit = txn.find("ggml_sycl::unified_cache_dense_scratch_runtime_fit(ctx->device, next_kv_info.n_ubatch, &needed,")
    taken = txn.find("dense_scratch_runtime_bytes = needed;", fit)
    charged = txn.find("ring_kv_zone.runtime_pending_bytes += dense_scratch_runtime_bytes;", taken)
    ring = txn.find("ggml_sycl_replan_pp_moe_onednn_ring(ctx->device, next_kv_info.n_ubatch, probe_mode, &ring_kv_zone)",
                    charged)
    return min(fit, taken, charged, ring) >= 0


def claim_ring_admission_subtracts_the_pending_demand(sycl: str) -> bool:
    """The ring is admitted against the RUNTIME zone's free bytes LESS the pending demand, not the raw free bytes."""
    ring = body(norm(sycl), RING_SIG)
    return (bool(ring) and "const size_t runtime_pending = arena && kv_zone ? kv_zone->runtime_pending_bytes : 0;" in ring
            and "const size_t runtime_net_bytes = capacity_bytes > runtime_pending ? capacity_bytes - runtime_pending : 0;"
            in ring and "admit_in.runtime_available_bytes = runtime_net_bytes;" in ring)


def claim_zone_requirement_folds_in_the_dense_plans(cache: str) -> bool:
    """The RUNTIME zone is sized for both dense plans on top of the ring, pipeline and control pool, and the
    zone sizing reads that requirement."""
    t = norm(cache)
    req = body(t, REQ_SIG)
    return (bool(req) and "const size_t mmq_src1 = unified_cache_get_planned_mmq_src1_scratch_bytes(device_id);" in req
            and "const size_t dequant_f16 = unified_cache_get_planned_dequant_f16_scratch_bytes(device_id);" in req
            and "*out = base + mmq_src1 + dequant_f16;" in req
            and re.search(r"unified_cache_get_planned_runtime_zone_requirement\(dev_id, &planned_runtime_scratch\).*?"
                          r"runtime_zone = planned_runtime_scratch;", t) is not None)


def test_the_publish_claims_the_buffer_after_the_commit_and_before_the_hold():
    assert claim_transaction_claims_after_commit_before_hold(SYCL)


def test_the_claim_is_the_whole_plan_and_a_short_claim_is_reported():
    assert claim_helper_claims_the_whole_plan(SYCL)


def test_the_claim_allocates_in_the_runtime_zone_with_spill_forbidden():
    assert claim_runtime_scratch_stays_in_the_runtime_zone(COMMON)


def test_the_ring_replan_is_charged_the_dense_plans():
    assert claim_ring_replan_is_charged_the_dense_plans(SYCL)


def test_the_ring_admission_subtracts_the_pending_demand():
    assert claim_ring_admission_subtracts_the_pending_demand(SYCL)


def test_the_runtime_zone_requirement_folds_in_the_dense_plans():
    assert claim_zone_requirement_folds_in_the_dense_plans(CACHE)


# ---- mutants: every claim must fail against the thing it forbids --------------------------------------------------


def _once(raw: str, old: str, new: str) -> str:
    assert raw.count(old) >= 1, f"mutant anchor not found: {old!r}"
    return raw.replace(old, new, 1)


def test_mutant_no_claim_fails():
    assert not claim_transaction_claims_after_commit_before_hold(
        _once(SYCL, "ggml_sycl_mmq_src1_claim_plan(*ctx);", ";"))


def test_mutant_claim_after_the_hold_refresh_fails():
    raw = _once(SYCL, "ggml_sycl_mmq_src1_claim_plan(*ctx);\n", "")
    raw = _once(raw, "ggml_sycl_planned_scratch_hold_refresh(*ctx);\n    // This plan's own reserve",
                "ggml_sycl_planned_scratch_hold_refresh(*ctx);\n    ggml_sycl_mmq_src1_claim_plan(*ctx);\n"
                "    // This plan's own reserve")
    assert not claim_transaction_claims_after_commit_before_hold(raw)


def test_mutant_claim_before_the_commit_fails():
    raw = _once(SYCL, "ggml_sycl_mmq_src1_claim_plan(*ctx);\n", "")
    raw = _once(raw, "dense_guard.commit();", "ggml_sycl_mmq_src1_claim_plan(*ctx);\n    dense_guard.commit();")
    assert not claim_transaction_claims_after_commit_before_hold(raw)


def test_mutant_claim_only_on_the_probe_path_fails():
    assert not claim_transaction_claims_after_commit_before_hold(
        _once(SYCL, "    ggml_sycl_mmq_src1_claim_plan(*ctx);\n",
              "    if (probe_mode) {\n        ggml_sycl_mmq_src1_claim_plan(*ctx);\n    }\n"))


def test_mutant_claim_helper_returns_first_fails():
    assert not claim_helper_claims_the_whole_plan(
        _once(SYCL, "static void ggml_sycl_mmq_src1_claim_plan(ggml_backend_sycl_context & ctx) {\n",
              "static void ggml_sycl_mmq_src1_claim_plan(ggml_backend_sycl_context & ctx) {\n    return;\n"))


def test_mutant_claim_helper_returns_after_the_skip_fails():
    assert not claim_helper_claims_the_whole_plan(
        _once(SYCL, "        return;\n    }\n    if (ctx.mmvq_q8_activation_cache.ensure_buffer(",
              "        return;\n    }\n    if (true) {\n        return;\n    }\n"
              "    if (ctx.mmvq_q8_activation_cache.ensure_buffer("))


def test_mutant_claim_sized_by_an_op_not_the_plan_fails():
    helper_raw = SYCL[SYCL.find("static void ggml_sycl_mmq_src1_claim_plan("):]
    m = re.search(r"ensure_buffer\((\w+),", helper_raw)
    assert m, "claim helper has no ensure_buffer call to mutate"
    assert not claim_helper_claims_the_whole_plan(_once(SYCL, m.group(0), "ensure_buffer(required_bytes,"))


def test_mutant_claim_that_aborts_fails():
    assert not claim_helper_claims_the_whole_plan(
        _once(SYCL, "GGML_LOG_WARN(\n        \"[MMQ-SRC1] device %d: the planned Q8_1 src1 buffer",
              "GGML_ABORT(\n        \"[MMQ-SRC1] device %d: the planned Q8_1 src1 buffer"))


def test_mutant_runtime_scratch_spill_allowed_fails():
    assert not claim_runtime_scratch_stays_in_the_runtime_zone(
        _once(COMMON, "req.intent.constraints.forbid_vram_zone_spill = true;\n    ggml_sycl::allocation_result",
              "req.intent.constraints.forbid_vram_zone_spill = false;\n    ggml_sycl::allocation_result"))


def test_mutant_ring_not_charged_the_dense_plans_fails():
    assert not claim_ring_replan_is_charged_the_dense_plans(
        _once(SYCL, "ring_kv_zone.runtime_pending_bytes += dense_scratch_runtime_bytes;", ";"))


def test_mutant_ring_admitted_against_the_raw_free_bytes_fails():
    assert not claim_ring_admission_subtracts_the_pending_demand(
        _once(SYCL, "admit_in.runtime_available_bytes       = runtime_net_bytes;",
              "admit_in.runtime_available_bytes       = capacity_bytes;"))


def test_mutant_zone_requirement_without_the_q8_plan_fails():
    assert not claim_zone_requirement_folds_in_the_dense_plans(
        _once(CACHE, "*out = base + mmq_src1 + dequant_f16;", "*out = base + dequant_f16;"))


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
