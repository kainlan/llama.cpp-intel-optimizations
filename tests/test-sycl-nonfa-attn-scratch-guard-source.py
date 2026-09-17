"""Source contract for llama.cpp-pvjr: the runtime-context-update guard that
refuses a context whose non-flash-attention batched mul_mat scratch demand
exceeds this device's LIVE outside-arena headroom, instead of letting
ggml_sycl_mul_mat_batched_sycl() abort mid-prefill (the "batched F16 mul_mat
failed -- no recovery path available" GGML_ABORT in ggml-sycl.cpp).

This predicate is EMPIRICAL, not a modeled worst case, and has already been
revised TWICE on hardware evidence -- read this before "improving" it again.
Round 1 (llama.cpp-oyfl) modeled everything that can spill outside the fixed
SYCL arena (compute-buffer regrowth, SCRATCH overflow, oneDNN scratchpad
fragmentation) and compared it against live free VRAM; hardware falsified it
-- a non-FA prefill was measured consuming several GB more outside the arena
than any term the model accounted for, on both discrete cards, and no zone
size or headroom override closed the gap (llama.cpp-k1ev, filed for that
still-unexplained consumer). Round 2 (also llama.cpp-oyfl) retreated to "does
the demand exceed the SCRATCH zone the arena already reserved" -- simpler,
and it refused every case that actually aborted, but a 2026-09-09 bracketing
sweep on f594574bf (task llama.cpp-pvjr, o3a0 + oyfl + rqak merged) falsified
that one too: several shapes exceeding the 512 MiB SCRATCH zone ran clean on
both cards, so the zone's own capacity is not the resource that actually runs
out. That sweep is the "new evidence that k1ev's consumer is understood and
bounded" round 1's own text asked for -- it re-derived a live-free-memory
predicate (demand + an EMPIRICAL 928 MiB reserve vs. live free memory), this
time bracketed from measured hardware points on both cards rather than
modeled from first principles, so it is what this file now gates. Do not
retreat to the SCRATCH-zone-capacity predicate (round 2) or a
from-first-principles live-free model (round 1) without new hardware
evidence that falsifies this one the same way.

Host-only, pure text assertions -- no SYCL device required, matching
test-sycl-onednn-graph-allocator-source.py's pytest-collectible pattern
(llama_test_pytest hands this file to pytest.main(), so checks must live
inside a test_*() function or pytest's "no tests collected" (exit 5) is
scored as a failure by design, not a skip).

Checks run against COMMENT-STRIPPED text (see strip_comments()) so a
positive structural check cannot be fooled by prose that quotes a call the
code does not actually make -- a comment can describe a call site
accurately without the code actually containing it, and a substring check
against raw (comment-bearing) text cannot tell the difference; stripping
comments first closes that gap. test-sycl-onednn-graph-allocator-source.py
applies the same hardening for the same reason.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GGML_SYCL_H = (ROOT / "ggml/include/ggml-sycl.h").read_text()
CACHE_HPP = (ROOT / "ggml/src/ggml-sycl/unified-cache.hpp").read_text()
CACHE_CPP = (ROOT / "ggml/src/ggml-sycl/unified-cache.cpp").read_text()
LLAMA_CONTEXT_CPP = (ROOT / "src/llama-context.cpp").read_text()
# llama.cpp-rqak: llama_model_sycl_populate_inventory() lives here -- see
# test_all_layers_max_updates_before_eligibility_continue() below.
LLAMA_MODEL_CPP = (ROOT / "src/llama-model.cpp").read_text()
# llama.cpp-oyfl: plain file I/O, not the codescout index -- CLAUDE.md
# documents that index (and search_text's live scan) as blind/oversized for
# this specific ~60k-line file, so a tool-assisted search here would
# silently miss real occurrences.
GGML_SYCL_CPP = (ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()
# llama.cpp-pvjr: the design doc's "A non-tensor consumer" subsection and the
# GGML_SYCL_NONFA_ATTN_SCRATCH_MB env-var row -- see
# test_docs_reflect_the_headroom_predicate() below. Prose, not source, so it
# is read raw (not comment-stripped).
SYCL_MEMORY_DESIGN_MD = (ROOT / "docs/backend/sycl-memory-design.md").read_text()
SYCL_ENV_VARS_MD = (ROOT / "docs/backend/sycl-env-vars.md").read_text()


# Single left-to-right alternation, not two sequential passes -- see
# test-sycl-onednn-graph-allocator-source.py's comment on why a block-comment
# pass run first can swallow a `/*` that appears inside a `//` comment's own
# prose (e.g. a glob like `weight-reclaim/*`). Order matters here for the
# same reason even though none of these four files are known to hit it today.
_LEXEME_RE = re.compile(
    r'"(?:\\.|[^"\\\n])*"'  # string literal (kept)
    r"|'(?:\\.|[^'\\\n])*'"  # char literal (kept)
    r"|//[^\n]*"  # line comment (dropped)
    r"|/\*.*?\*/",  # block comment (dropped)
    flags=re.DOTALL,
)


def strip_comments(src: str) -> str:
    """Remove // and /* */ comments; keep string/char literals verbatim."""

    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


GGML_SYCL_H_CODE = strip_comments(GGML_SYCL_H)
CACHE_HPP_CODE = strip_comments(CACHE_HPP)
CACHE_CPP_CODE = strip_comments(CACHE_CPP)
LLAMA_CONTEXT_CPP_CODE = strip_comments(LLAMA_CONTEXT_CPP)
LLAMA_MODEL_CPP_CODE = strip_comments(LLAMA_MODEL_CPP)
GGML_SYCL_CPP_CODE = strip_comments(GGML_SYCL_CPP)


def _normalize_ws(text: str) -> str:
    """Collapse all whitespace runs to a single space, so a call or
    declaration re-wrapped by clang-format across lines still matches a
    single-line pattern."""
    return re.sub(r"\s+", " ", text)


def test_api_carries_flash_attn_enabled():
    """Both public entry points must declare the new parameter -- without it
    ggml_backend_sycl_set_runtime_context() has no way to know whether flash
    attention is on for this context, and the guard below cannot be gated on
    the real per-context state."""
    header_norm = _normalize_ws(GGML_SYCL_H_CODE)
    assert "ggml_backend_sycl_set_runtime_context(ggml_backend_t backend" in header_norm
    assert re.search(
        r"ggml_backend_sycl_set_runtime_context\(ggml_backend_t backend,"
        r".*?bool\s+flash_attn_enabled\);",
        header_norm,
    ), "ggml_backend_sycl_set_runtime_context() declaration must carry a bool flash_attn_enabled parameter"
    assert re.search(
        r"ggml_backend_sycl_set_runtime_context_for_model\(.*?bool\s+flash_attn_enabled\);",
        header_norm,
    ), "ggml_backend_sycl_set_runtime_context_for_model() declaration must carry a bool flash_attn_enabled parameter"


def test_llama_context_threads_real_flash_attn_state():
    """llama-context.cpp must pass the RESOLVED cparams.flash_attn (not the
    raw llama_flash_attn_type, and not a hardcoded true/false) into the
    runtime-context call -- otherwise the guard below is gated on nothing
    real for the one caller that matters."""
    ctx_norm = _normalize_ws(LLAMA_CONTEXT_CPP_CODE)
    # llama.cpp-3aos: tolerate additional cparams.* arguments inserted
    # between n_seq_max and flash_attn (e.g. cparams.kv_unified) -- this
    # check's intent is "flash_attn is the real cparams field, passed all
    # the way to the call", not "these two arguments are adjacent".
    matches = re.findall(
        r"runtime_context_fn\([^;]*?cparams\.n_seq_max,(?:\s*cparams\.\w+,)*\s*cparams\.flash_attn\)", ctx_norm
    )
    assert len(matches) >= 2, (
        "expected at least two runtime_context_fn(...) call sites in llama-context.cpp "
        "(the initial call and the BUSY-retry loop) to pass cparams.flash_attn -- found "
        f"{len(matches)}"
    )


def test_guard_consults_the_headroom_predicate():
    """ggml_sycl_check_nonfa_attn_scratch() (ggml-sycl.cpp) -- the shared
    helper both ggml_backend_sycl_set_runtime_context() and the narrow
    ggml_backend_sycl_recheck_runtime_context_flash_attn() re-check funnel
    into -- must actually call the exported demand formula, read LIVE
    device free memory, and gate the refusal on outside-arena headroom
    (demand + reserve vs free), not the SCRATCH zone's own capacity
    (llama.cpp-pvjr: the zone-capacity predicate over-refused the B70)."""
    func_start = GGML_SYCL_CPP_CODE.find("static bool ggml_sycl_check_nonfa_attn_scratch(")
    assert func_start != -1, "ggml_sycl_check_nonfa_attn_scratch() definition not found in ggml-sycl.cpp"
    # Bound the search to this function's body: from the definition to the
    # next top-level function definition after it
    # (ggml_backend_sycl_set_runtime_context, the first of its two callers),
    # so a match cannot come from some unrelated later call site.
    next_func = GGML_SYCL_CPP_CODE.find(
        "void ggml_backend_sycl_set_runtime_context(", func_start + 1
    )
    assert next_func != -1, "could not bound ggml_sycl_check_nonfa_attn_scratch()'s body"
    body = GGML_SYCL_CPP_CODE[func_start:next_func]
    body_norm = _normalize_ws(body)

    assert "if (flash_attn_enabled) { return true; }" in body_norm, (
        "ggml_sycl_check_nonfa_attn_scratch() must gate the non-FA scratch guard on "
        "flash_attn_enabled being false"
    )
    assert "unified_cache_nonfa_attn_scratch_demand_bytes(" in body_norm, (
        "ggml_sycl_check_nonfa_attn_scratch() must call "
        "unified_cache_nonfa_attn_scratch_demand_bytes() to size the guard"
    )
    assert "ggml_backend_sycl_get_device_memory(" in body_norm, (
        "the guard must read LIVE device free memory (ggml_backend_sycl_get_device_memory()) at guard "
        "time -- a stored/cached headroom figure would go stale against a budget-pct override or "
        "another tenant on the card"
    )
    assert "unified_cache_nonfa_attn_scratch_fits_headroom(" in body_norm, (
        "the guard must decide fit/refuse through the shared "
        "unified_cache_nonfa_attn_scratch_fits_headroom() helper, not a re-derived comparison"
    )
    assert "unified_cache_nonfa_attn_outside_arena_reserve_bytes(" in body_norm, (
        "the guard must read the empirical outside-arena reserve "
        "(unified_cache_nonfa_attn_outside_arena_reserve_bytes(), 928 MiB) rather than hardcoding it"
    )
    assert "unified_cache_nonfa_attn_scratch_headroom_capacity_bytes(" in body_norm, (
        "the refusal's largest-fitting capacity must come from "
        "unified_cache_nonfa_attn_scratch_headroom_capacity_bytes(), the inverse of fits_headroom()'s "
        "own comparison -- not a hand-written free_mem - reserve subtraction, which can drift from "
        "the predicate it is supposed to invert"
    )
    assert not re.search(r"free_mem\s*>\s*reserve\s*\?", body_norm), (
        "the hand-written headroom-capacity subtraction must not return -- call the helper instead"
    )
    assert not re.search(
        r"nonfa_demand\s*<=\s*scratch_capacity|scratch_capacity\s*>=\s*nonfa_demand", body_norm
    ), (
        "the refusal must no longer be decided by comparing demand against the SCRATCH zone's own "
        "capacity (llama.cpp-pvjr) -- that predicate over-refused the B70; the SCRATCH-zone re-plan may "
        "still run for its own INFO logging, but must not decide fit/refuse"
    )
    assert "unified_cache_largest_fitting_n_ctx_for_nonfa_attn_scratch(" in body_norm, (
        "a refusal must report the largest fitting n_ctx (same style as the KV budget refusal's "
        "ggml_sycl_largest_fitting_n_ctx()), not just an error with no remediation"
    )
    assert "runtime context update rejected" in body_norm, (
        "this guard's own refusals must say \"runtime context update rejected\", matching the "
        "pre-existing KV budget refusal's \"runtime KV update rejected\" family of wording (both are "
        "runtime-context-update refusals, just for different checks)"
    )
    assert "GGML_SYCL_NONFA_ATTN_SCRATCH_MB" not in body_norm, (
        "the runtime refusal must NOT advertise GGML_SYCL_NONFA_ATTN_SCRATCH_MB as a remediation -- "
        "it only replaces the demand term d, not the reserve or the headroom comparison, so it is an "
        "experimentation knob (llama.cpp-k1ev), not a user-facing fix; -fa 1/auto or a smaller -c are "
        "the only remediations with hardware support"
    )
    assert "headroom-limited" in body_norm, (
        "the largest-fitting-n_ctx remediation must be labeled \"headroom-limited\" -- it is bounded by "
        "this device's live outside-arena headroom, not a whole-device or zone-only guarantee"
    )
    assert "scratch-limited" not in body_norm, (
        "\"scratch-limited\" is the retired zone-capacity predicate's own label (llama.cpp-oyfl) -- the "
        "headroom-based remediation must use \"headroom-limited\" instead, not both"
    )
    disabled_match = re.search(
        r"unified_cache_nonfa_attn_scratch_guard_disabled\s*\(\s*\)"
        r"|nonfa_attn_scratch_mb_override\s*\(\s*\)\s*==\s*0",
        body_norm,
    )
    fits_idx = body_norm.find("unified_cache_nonfa_attn_scratch_fits_headroom(")
    assert disabled_match is not None, (
        "an explicit GGML_SYCL_NONFA_ATTN_SCRATCH_MB=0 must skip the guard entirely -- expected either "
        "unified_cache_nonfa_attn_scratch_guard_disabled() or the override accessor compared against 0"
    )
    assert fits_idx != -1 and disabled_match.start() < fits_idx, (
        "the explicit-0 skip must appear BEFORE the headroom fit/refuse comparison -- 0 + reserve "
        "compared against free would otherwise refuse any card with less than the reserve free, which "
        "is not what \"disable\" means"
    )
    assert "unified_cache_ensure_planned_arena_zones(" in body_norm, (
        "the guard must make the opportunistic re-plan attempt (same call reserve_onednn_scratch() "
        "already documents as succeeding only while the arena is still unused) before checking"
    )
    free_idx = body_norm.find("ggml_backend_sycl_get_device_memory(")
    replan_idx = body_norm.find("unified_cache_ensure_planned_arena_zones(")
    assert free_idx != -1 and replan_idx != -1 and free_idx < replan_idx, (
        "the live free-memory read must precede the opportunistic SCRATCH-zone re-plan -- reading "
        "after it would double count a committed zone growth as both the growth and the headroom "
        "this guard then requires on top of it"
    )
    assert "unified_cache_set_planned_nonfa_attn_scratch_shape(" in body_norm, (
        "the opportunistic re-plan must record the REAL runtime shape first, not the load-time one"
    )
    assert "scratch_capacity > scratch_capacity_before" in body_norm, (
        "the re-plan's own log line must be gated on the zone actually having grown -- logging "
        "\"raised\" when unified_cache_ensure_planned_arena_zones() could not rebuild (live weight "
        "leases) would misreport a no-op as a success"
    )


def test_both_callers_wire_into_the_shared_guard():
    """Both ggml_sycl_run_runtime_context_transaction() (the shared body
    behind the full transaction -- llama.cpp-tsfl split
    ggml_backend_sycl_set_runtime_context() into a thin wrapper that
    forwards into this shared body, which the new probe entry point also
    calls) and ggml_backend_sycl_recheck_runtime_context_flash_attn() (the
    narrow re-check) must actually call the shared
    ggml_sycl_check_nonfa_attn_scratch() helper -- a helper that exists and
    is correct but is never called by one of its two intended entry points
    would leave that path's contexts unguarded."""
    full_start = GGML_SYCL_CPP_CODE.find(
        "static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction("
    )
    assert full_start != -1, "ggml_sycl_run_runtime_context_transaction() definition not found"
    full_next = GGML_SYCL_CPP_CODE.find(
        "void ggml_backend_sycl_set_runtime_context(", full_start + 1
    )
    assert full_next != -1, "could not bound ggml_sycl_run_runtime_context_transaction()'s body"
    assert full_start < full_next, "ggml_sycl_run_runtime_context_transaction() must precede its wrapper"
    full_body_norm = _normalize_ws(GGML_SYCL_CPP_CODE[full_start:full_next])
    assert "ggml_sycl_check_nonfa_attn_scratch(" in full_body_norm, (
        "ggml_sycl_run_runtime_context_transaction() (the shared body) must call the shared guard helper"
    )

    recheck_start = GGML_SYCL_CPP_CODE.find(
        "ggml_sycl_lifecycle_result ggml_backend_sycl_recheck_runtime_context_flash_attn("
    )
    assert recheck_start != -1, "ggml_backend_sycl_recheck_runtime_context_flash_attn() definition not found"
    recheck_next = GGML_SYCL_CPP_CODE.find(
        "void ggml_backend_sycl_set_runtime_n_ctx(", recheck_start + 1
    )
    assert recheck_next != -1, "could not bound ggml_backend_sycl_recheck_runtime_context_flash_attn()'s body"
    recheck_body_norm = _normalize_ws(GGML_SYCL_CPP_CODE[recheck_start:recheck_next])
    assert "ggml_sycl_check_nonfa_attn_scratch(" in recheck_body_norm, (
        "ggml_backend_sycl_recheck_runtime_context_flash_attn() must call the shared guard helper -- "
        "this is the entry point resolve_fused_ops() uses for an AUTO context resolving to non-FA, "
        "so a missing call here would leave that specific context unguarded"
    )

    # The header declaration for the narrow re-check entry point must also
    # exist, so llama-context.cpp compiles against a real exported symbol.
    header_norm = _normalize_ws(GGML_SYCL_H_CODE)
    assert (
        "ggml_backend_sycl_recheck_runtime_context_flash_attn(" in header_norm
    ), "ggml_backend_sycl_recheck_runtime_context_flash_attn() must be declared in ggml-sycl.h"


def test_callers_pass_the_all_layers_head_count():
    """llama.cpp-rqak: both guard call sites -- the shared body's
    (ggml_sycl_run_runtime_context_transaction(), behind the full
    transaction) next_plan and the narrow re-check's current->plan -- must
    pass the ALL-LAYERS head-count field into ggml_sycl_check_nonfa_attn_scratch().
    The non-FA attention path runs on EVERY attention layer, so its demand
    model (max(16 MiB, n_head x n_ubatch x n_ctx x 2 B x 3)) needs the
    maximum query-head count over all layers -- not llama.cpp-o3a0's
    n_head_ctx_max/n_head_swa_max, which are maxima over oneDNN-ELIGIBLE
    layers only and both read 0 for a model where every layer is
    ineligible (e.g. a DeepSeek-V3-class model, D=576), which would push
    this guard into its "could not evaluate" WARN path and leave that
    exact shape unguarded. And never the pre-o3a0 planner_n_head name,
    which o3a0 removed -- git's clean auto-merge of o3a0 (per-class
    maxima) with llama.cpp-oyfl (this guard, developed against the
    pre-o3a0 single field) left the guard referencing a member that no
    longer exists."""
    full_start = GGML_SYCL_CPP_CODE.find(
        "static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction("
    )
    assert full_start != -1
    full_next = GGML_SYCL_CPP_CODE.find(
        "void ggml_backend_sycl_set_runtime_context(", full_start + 1
    )
    assert full_next != -1
    assert full_start < full_next
    full_body_norm = _normalize_ws(GGML_SYCL_CPP_CODE[full_start:full_next])

    recheck_start = GGML_SYCL_CPP_CODE.find(
        "ggml_sycl_lifecycle_result ggml_backend_sycl_recheck_runtime_context_flash_attn("
    )
    assert recheck_start != -1
    recheck_next = GGML_SYCL_CPP_CODE.find("void ggml_backend_sycl_set_runtime_n_ctx(", recheck_start + 1)
    assert recheck_next != -1
    recheck_body_norm = _normalize_ws(GGML_SYCL_CPP_CODE[recheck_start:recheck_next])

    for caller_name, body in (
        ("the shared body (ggml_sycl_run_runtime_context_transaction)", full_body_norm),
        ("the narrow re-check (ggml_backend_sycl_recheck_runtime_context_flash_attn)", recheck_body_norm),
    ):
        assert "planner_n_head_all_max" in body, (
            f"{caller_name} must pass the all-layers head-count field (planner_n_head_all_max) into "
            "ggml_sycl_check_nonfa_attn_scratch() -- see llama.cpp-rqak"
        )
        # \b requires a non-word char on both sides, and "_" is a word char,
        # so this cannot match inside planner_n_head_all_max/planner_n_head_ctx_max/
        # planner_n_head_swa_max -- it only matches the bare, pre-o3a0 name.
        assert not re.search(r"\bplanner_n_head\b", body), (
            f"{caller_name} must not reference the removed planner_n_head field (llama.cpp-o3a0 replaced "
            "it with the per-class n_head_ctx_max/n_head_swa_max maxima; this guard needs its own "
            "all-layers field, planner_n_head_all_max, not either of those)"
        )
        assert "n_head_ctx_max" not in body and "n_head_swa_max" not in body, (
            f"{caller_name} must not pass the o3a0 per-class oneDNN-eligible-only maxima "
            "(n_head_ctx_max / n_head_swa_max) into the non-FA attention guard -- those exclude "
            "oneDNN-ineligible layers and can both be 0 for a model where every layer is ineligible, "
            "which would silently leave that model's non-FA scratch demand unguarded"
        )


def test_all_layers_max_updates_before_eligibility_continue():
    """llama.cpp-rqak: n_head_all_max is correct only because its update in
    llama_model_sycl_populate_inventory() (src/llama-model.cpp) executes
    BEFORE the per-layer eligibility check's `continue` -- moving the update
    below that `continue` would leave every OTHER assertion in this file
    green (they only check the field's NAME reaches the guard, not that the
    loop computing it walked every layer) while silently making
    n_head_all_max eligible-only, i.e. identical to n_head_ctx_max/
    n_head_swa_max's union -- exactly the DeepSeek-V3-class (every layer
    ineligible, D=576) failure mode this field exists to guard against,
    with a value of 0 that this test suite would then have no way to catch
    from the guard side alone.

    Checked structurally (index-of-substring ordering) rather than by name
    match, because both statements name n_head_all_max -- a substring check
    for "n_head_all_max = std::max(" existing somewhere in the function body
    would still pass with the update moved below the `continue`."""
    func_start = LLAMA_MODEL_CPP_CODE.find("static void llama_model_sycl_populate_inventory(")
    assert func_start != -1, "llama_model_sycl_populate_inventory() definition not found in llama-model.cpp"
    # Bound to this function's own body: the next top-level function
    # definition after it (llama_model_sycl_apply_inventory), so a match
    # cannot come from some unrelated later call site.
    next_func = LLAMA_MODEL_CPP_CODE.find(
        "static void llama_model_sycl_apply_inventory(", func_start + 1
    )
    assert next_func != -1, "could not bound llama_model_sycl_populate_inventory()'s body"
    body = LLAMA_MODEL_CPP_CODE[func_start:next_func]

    update_idx = body.find("n_head_all_max = std::max(")
    eligible_idx = body.find("llama_model_sycl_onednn_head_dim_eligible(")
    assert update_idx != -1, "n_head_all_max's std::max() update not found in populate_inventory()'s body"
    assert eligible_idx != -1, (
        "llama_model_sycl_onednn_head_dim_eligible() call (the eligibility gate before the `continue`) "
        "not found in populate_inventory()'s body"
    )
    assert update_idx < eligible_idx, (
        "n_head_all_max's std::max() update must execute BEFORE the "
        "llama_model_sycl_onednn_head_dim_eligible() eligibility check's `continue` -- the non-FA "
        "attention guard this field feeds runs on every attention layer, not just oneDNN-eligible "
        "ones, so computing it after the `continue` would silently make it eligible-only (0 for a "
        "model where every layer is ineligible, e.g. DeepSeek-V3-class, D=576) -- see llama.cpp-rqak"
    )


def test_narrow_recheck_forbids_replan_and_takes_the_lock():
    """The narrow re-check must call the shared guard with allow_replan=false
    (never recording, re-planning, or restoring the plan-time SCRATCH-zone
    shape it does not own), and must take the same module-admission guard
    and tensor-inventory lock the full transaction serializes its own
    mutating work under, confirming under that lock that the plan snapshot
    read before the lock is still the live one.

    llama.cpp-tsfl: the publish path's allow_replan=true is no longer a
    literal at the call site -- ggml_sycl_run_runtime_context_transaction()
    (the shared body) now derives it from its own probe_mode parameter
    (allow_replan=!probe_mode, so replanning is allowed on the publish path
    and forbidden for a probe, matching the probe's own no-side-effects
    contract) and its thin wrapper, ggml_backend_sycl_set_runtime_context(),
    is what pins probe_mode=false for the publish path. Both halves are
    checked, on their own separately-bounded bodies, so a mutation to
    either the shared body's derivation or the wrapper's own argument is
    caught."""
    full_start = GGML_SYCL_CPP_CODE.find(
        "static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction("
    )
    assert full_start != -1
    full_next = GGML_SYCL_CPP_CODE.find(
        "void ggml_backend_sycl_set_runtime_context(", full_start + 1
    )
    assert full_next != -1
    assert full_start < full_next
    full_body_norm = _normalize_ws(GGML_SYCL_CPP_CODE[full_start:full_next])
    # allow_replan is passed positionally as the call's 6th argument -- matched
    # by position, not by the inline `/*allow_replan=*/` comment naming it,
    # because that comment is exactly the kind of text strip_comments()
    # (deliberately) removes before this check ever sees the source. The
    # optional `/*allow_replan=*/` group tolerates a future caller of this
    # helper reading from non-comment-stripped text; it never matches here.
    # No trailing `)` anchor -- the shared body's call also passes the
    # (7th, defaulted) probe_mode argument after allow_replan, unlike the
    # narrow re-check's call below, which relies on that parameter's default.
    assert re.search(
        r"ggml_sycl_check_nonfa_attn_scratch\([^()]*flash_attn_enabled,\s*(/\*allow_replan=\*/\s*)?!probe_mode\b",
        full_body_norm,
    ), (
        "the shared body must call the shared guard with allow_replan=!probe_mode (the 6th positional "
        "argument, after flash_attn_enabled) -- true for the publish path (probe_mode=false), false for "
        "a probe"
    )

    wrapper_start = full_next
    wrapper_next = GGML_SYCL_CPP_CODE.find(
        "ggml_backend_sycl_probe_runtime_context_for_model(", wrapper_start + 1
    )
    assert wrapper_next != -1, "could not bound ggml_backend_sycl_set_runtime_context()'s (wrapper) body"
    assert wrapper_start < wrapper_next
    wrapper_body_norm = _normalize_ws(GGML_SYCL_CPP_CODE[wrapper_start:wrapper_next])
    assert re.search(
        r"ggml_sycl_run_runtime_context_transaction\([^()]*flash_attn_enabled,\s*(/\*probe_mode=\*/\s*)?false",
        wrapper_body_norm,
    ), (
        "ggml_backend_sycl_set_runtime_context() (the wrapper) must pass probe_mode=false into the "
        "shared body, so allow_replan=!probe_mode is still true on the publish path"
    )

    recheck_start = GGML_SYCL_CPP_CODE.find(
        "ggml_sycl_lifecycle_result ggml_backend_sycl_recheck_runtime_context_flash_attn("
    )
    assert recheck_start != -1
    recheck_next = GGML_SYCL_CPP_CODE.find("void ggml_backend_sycl_set_runtime_n_ctx(", recheck_start + 1)
    assert recheck_next != -1
    recheck_body_norm = _normalize_ws(GGML_SYCL_CPP_CODE[recheck_start:recheck_next])

    assert re.search(r"ggml_sycl_check_nonfa_attn_scratch\([^()]*flash_attn_enabled,\s*false\)", recheck_body_norm), (
        "the narrow re-check must call the shared guard with allow_replan=false (the final positional "
        "argument, after flash_attn_enabled) -- it must not record, re-plan, or restore the plan-time "
        "SCRATCH-zone shape"
    )
    assert "sycl_module_mutation_guard module_guard;" in recheck_body_norm, (
        "the narrow re-check must take sycl_module_mutation_guard, the same admission guard the full "
        "transaction's callers rely on, so it cannot run past a module shutdown"
    )
    assert "std::lock_guard<std::mutex> lock(g_tensor_inventory_mutex);" in recheck_body_norm, (
        "the narrow re-check must take g_tensor_inventory_mutex, the same mutex the full transaction "
        "serializes its own mutating work under, before confirming the plan snapshot is still live -- "
        "not because some specific accessor this then reads requires the lock (zone_capacity() is an "
        "unsynchronized array read with no lock contract of its own)"
    )
    assert "ggml_sycl_global_plan_snapshot().get() != current.get()" in recheck_body_norm, (
        "the narrow re-check must confirm, under the lock, that the plan snapshot read before the lock "
        "is still the live one -- a read-then-confirm-under-lock construction of its own, guarding "
        "against acting on a plan a concurrent model load/unload or runtime-context call has already "
        "superseded"
    )


def test_for_model_forwards_flash_attn_enabled():
    """ggml_backend_sycl_set_runtime_context_for_model() must forward its new
    parameter into the inner call rather than dropping it on the floor."""
    func_start = GGML_SYCL_CPP_CODE.find("ggml_sycl_lifecycle_result ggml_backend_sycl_set_runtime_context_for_model(")
    assert func_start != -1, "ggml_backend_sycl_set_runtime_context_for_model() definition not found"
    body_norm = _normalize_ws(GGML_SYCL_CPP_CODE[func_start : func_start + 8000])
    # llama.cpp-3aos: tolerate additional arguments inserted between
    # n_seq_max and flash_attn_enabled (e.g. kv_unified) -- the intent is
    # "flash_attn_enabled reaches the inner call", not "these two
    # parameters are adjacent".
    assert re.search(
        r"ggml_backend_sycl_set_runtime_context\(backend, n_ctx, n_ubatch, n_seq_max,(?:\s*\w+,)*\s*flash_attn_enabled\)",
        body_norm,
    ), "ggml_backend_sycl_set_runtime_context_for_model() must forward flash_attn_enabled to the inner call"


def test_formula_and_inverse_are_declared_and_defined():
    """llama.cpp-pvjr: covers the demand formula and its inverse, plus the
    headroom predicate's own exported surface (the reserve, fits_headroom,
    guard_disabled, and headroom_capacity_bytes). The presence assertions
    below (declared in the header, defined in the implementation) check
    only the original two names; the not-file-static loop further down
    covers all six -- each must be exported (not file-static), unlike the
    oneDNN Graph-scratch floor sibling, because ggml-sycl.cpp (a different
    translation unit) must call them directly."""
    hpp_norm = _normalize_ws(CACHE_HPP_CODE)
    cpp_norm = _normalize_ws(CACHE_CPP_CODE)

    assert "size_t unified_cache_nonfa_attn_scratch_demand_bytes(uint32_t n_head, uint32_t n_ubatch, uint32_t n_ctx);" in hpp_norm
    assert "size_t unified_cache_nonfa_attn_scratch_demand_bytes(uint32_t n_head, uint32_t n_ubatch, uint32_t n_ctx) {" in cpp_norm

    assert "unified_cache_largest_fitting_n_ctx_for_nonfa_attn_scratch(" in hpp_norm
    assert "unified_cache_largest_fitting_n_ctx_for_nonfa_attn_scratch(" in cpp_norm

    # Neither may be declared `static` -- in the .cpp that would make them
    # unreachable from ggml-sycl.cpp, silently turning the guard above into
    # a compile error this text-only test cannot otherwise catch (a build is
    # a stronger check, but this test runs without one); in the .hpp a
    # `static` on the DECLARATION would give each translation unit that
    # includes the header its own internal-linkage copy, which is just as
    # wrong even though it happens to still compile. Checked in both
    # files for every name in checked_names below (two files x six
    # names), with a "static ... name(" bridge that does not name
    # a return type at all -- `\bstatic\b[^;{}]*?\bname\(` -- so
    # it is not fooled by a return-type spelling change ("static
    # std::size_t", say) or a storage-class keyword or attribute
    # stacked between `static` and the type ("static inline", "static
    # __attribute__((used)) size_t"); bounded to `[^;{}]` so it cannot
    # cross a statement or scope boundary and match some unrelated
    # earlier `static` against this name's own later, unrelated
    # appearance. Each match is reduced to a bool BEFORE the assert
    # -- asserting directly on a `re.search()` result (or on the huge
    # normalized-source string itself) would make a failing pytest
    # try to render that whole multi-hundred-KB string as part of
    # the diff.
    def _has_static_before(text: str, name: str) -> bool:
        return bool(re.search(r"\bstatic\b[^;{}]*?\b" + re.escape(name) + r"\s*\(", text))

    # llama.cpp-pvjr: extended from the original (demand_name, inverse_name)
    # pair to also cover the headroom predicate's own exported surface --
    # each must be reachable from ggml-sycl.cpp the same way the original
    # two are, and a file-static regression on any of them would be exactly
    # as silent (a compile error this text-only test would otherwise miss).
    checked_names = (
        "unified_cache_nonfa_attn_scratch_demand_bytes",
        "unified_cache_largest_fitting_n_ctx_for_nonfa_attn_scratch",
        "unified_cache_nonfa_attn_outside_arena_reserve_bytes",
        "unified_cache_nonfa_attn_scratch_fits_headroom",
        "unified_cache_nonfa_attn_scratch_guard_disabled",
        "unified_cache_nonfa_attn_scratch_headroom_capacity_bytes",
    )
    for name in checked_names:
        assert not _has_static_before(cpp_norm, name), f"{name}() must not be file-static (.cpp)"
        assert not _has_static_before(hpp_norm, name), f"{name}() must not be declared static (.hpp)"


def test_auto_flash_attn_resolution_rechecks_the_guard():
    """cparams.flash_attn defaults true for AUTO before resolve_fused_ops()
    resolves it (llama-context.cpp's cparams init runs before the
    constructor's runtime-context call), so an AUTO context that resolves
    to OFF must re-trigger a SYCL guard re-check with the now-resolved
    value, or the guard above never sees it for that context.

    Two DIFFERENT methods are involved, not one shared between both call
    sites: sycl_resync_runtime_context_flash_attn() (the FULL
    runtime-context transaction: KV replan, MoE MMID reaccount/materialize,
    plan republish) is the constructor's OWN initial call, made before any
    AUTO resolution; sycl_recheck_runtime_context_flash_attn() (a NARROW
    re-check of only the non-FA attention scratch guard) is what
    resolve_fused_ops() calls once AUTO actually resolves. Re-running the
    full transaction from resolve_fused_ops() would needlessly redo KV/MMID
    work that has no reason to change just because flash_attn_enabled did."""
    ctx_norm = _normalize_ws(LLAMA_CONTEXT_CPP_CODE)
    assert "void llama_context::sycl_resync_runtime_context_flash_attn()" in ctx_norm, (
        "expected the constructor's full-transaction helper to still exist"
    )
    assert "void llama_context::sycl_recheck_runtime_context_flash_attn()" in ctx_norm, (
        "expected a separate narrow re-check helper for resolve_fused_ops() to call"
    )

    # The constructor calls the FULL-transaction helper, not the narrow one.
    # Bounded to the destructor that immediately follows it (not all the way
    # to resolve_fused_ops(), which would also swallow the two helper
    # methods' own definitions sitting in between and make this check
    # imprecise about what "the constructor" actually means).
    ctor_start = ctx_norm.find("llama_context::llama_context(")
    assert ctor_start != -1, "llama_context::llama_context() definition not found"
    dtor_start = ctx_norm.find("llama_context::~llama_context(", ctor_start + 1)
    assert dtor_start != -1, "~llama_context() definition not found"
    resolve_start = ctx_norm.find("void llama_context::resolve_fused_ops(")
    assert resolve_start != -1, "resolve_fused_ops() definition not found"
    assert ctor_start < dtor_start < resolve_start, (
        "expected the constructor, then the destructor, then resolve_fused_ops(), in that order"
    )
    ctor_body = ctx_norm[ctor_start:dtor_start]
    assert "sycl_resync_runtime_context_flash_attn();" in ctor_body, (
        "the constructor must call the FULL-transaction sycl_resync_runtime_context_flash_attn() -- "
        "it is where n_ctx/n_ubatch are established for the first time"
    )
    assert "sycl_recheck_runtime_context_flash_attn();" not in ctor_body, (
        "the constructor must NOT call the narrow re-check -- that would skip the KV replan/MMID "
        "reaccount a first-time call needs"
    )

    # resolve_fused_ops() calls the NARROW re-check, inside the same
    # if (cparams.auto_fa) block that resolves flash_attn, so it fires
    # exactly once, right when the real value becomes known -- and must NOT
    # call the full-transaction helper (that would needlessly re-run KV/MMID
    # work resolve_fused_ops() has no reason to touch).
    resolve_body = ctx_norm[resolve_start : resolve_start + 4000]
    assert re.search(r"if \(cparams\.auto_fa\) \{[^}]*resolve\([^;]*flash_attn[^;]*;[^}]*"
                      r"sycl_recheck_runtime_context_flash_attn\(\);", resolve_body), (
        "resolve_fused_ops() must call sycl_recheck_runtime_context_flash_attn() inside the same "
        "if (cparams.auto_fa) block that resolves flash_attn, so it fires exactly once, right when "
        "the real value becomes known"
    )
    assert "sycl_resync_runtime_context_flash_attn();" not in resolve_body, (
        "resolve_fused_ops() must NOT call the full-transaction helper -- only the narrow re-check"
    )

    # Both helpers' own header declarations exist too, so both callers
    # compile against real class members, not undeclared symbols this test
    # alone would not catch.
    ctx_h = (ROOT / "src/llama-context.h").read_text()
    ctx_h_norm = _normalize_ws(strip_comments(ctx_h))
    assert "void sycl_resync_runtime_context_flash_attn();" in ctx_h_norm
    assert "void sycl_recheck_runtime_context_flash_attn();" in ctx_h_norm


_PREPROC_OPEN_RE = re.compile(r"^\s*#\s*(if|ifdef|ifndef)\b")
_PREPROC_ENDIF_RE = re.compile(r"^\s*#\s*endif\b")
_PREPROC_DNNL_IF_RE = re.compile(r"^\s*#\s*if\s+GGML_SYCL_DNNL\b")


def _is_inside_if_dnnl(raw: str, position: int) -> bool:
    """True if `position` falls lexically inside an #if GGML_SYCL_DNNL /
    #endif block, tracking full nesting depth rather than the nearest
    preceding #if/#endif tokens -- a last-match scan gives the wrong answer
    as soon as an unrelated nested #if/#endif (or #ifdef/#ifndef) sits
    between the #if GGML_SYCL_DNNL and the position being checked, because
    the innermost #endif it finds may close that unrelated nested block
    rather than the GGML_SYCL_DNNL one."""
    depth = 0
    dnnl_depth = None
    for line in raw[:position].splitlines():
        stripped = line.strip()
        if _PREPROC_OPEN_RE.match(stripped):
            depth += 1
            if dnnl_depth is None and _PREPROC_DNNL_IF_RE.match(stripped):
                dnnl_depth = depth
        elif _PREPROC_ENDIF_RE.match(stripped):
            if dnnl_depth is not None and depth == dnnl_depth:
                dnnl_depth = None
            depth = max(0, depth - 1)
    return dnnl_depth is not None


def test_plan_time_shape_is_recorded_unconditionally():
    """populate_host_zone_sizing() must record the non-FA shape
    UNCONDITIONALLY (not gated behind #if GGML_SYCL_DNNL like the oneDNN
    sibling) -- the path this sizes for is independent of oneDNN."""
    cpp_norm = _normalize_ws(CACHE_CPP_CODE)
    assert "unified_cache_set_planned_nonfa_attn_scratch_shape(plan.device_id, plan.planner_n_head_all_max, plan.planner_n_ubatch, plan.planner_n_ctx)" in cpp_norm

    # Comment-stripped text drops "#if GGML_SYCL_DNNL" lines too (they are
    # preprocessor directives, not comments -- re-check against the RAW,
    # comment-bearing source): the call site must not sit inside that guard.
    # Tracked by nesting depth (see _is_inside_if_dnnl above), not by a
    # nearest-preceding-token scan, which can misjudge across an unrelated
    # nested #if/#endif sitting between the guard and the call site.
    raw = CACHE_CPP
    call_idx = raw.find("unified_cache_set_planned_nonfa_attn_scratch_shape(plan.device_id")
    assert call_idx != -1
    assert not _is_inside_if_dnnl(raw, call_idx), (
        "unified_cache_set_planned_nonfa_attn_scratch_shape() call site must not be inside "
        "an #if GGML_SYCL_DNNL block"
    )


def test_docs_reflect_the_headroom_predicate():
    """llama.cpp-pvjr: the design doc's "A non-tensor consumer" subsection and
    the GGML_SYCL_NONFA_ATTN_SCRATCH_MB env-var row must describe the
    headroom predicate that replaced the zone-capacity one, not the retired
    claims that motivated it -- both of which the pvjr sweep falsified: the
    B70 (and B50) ran several shapes clean that exceeded the 512 MiB SCRATCH
    zone, so "the zone-only check is not more conservative than reality" and
    "nothing measured runs with a demand above the zone" are no longer true."""
    for name, text in (
        ("sycl-memory-design.md", SYCL_MEMORY_DESIGN_MD),
        ("sycl-env-vars.md", SYCL_ENV_VARS_MD),
    ):
        assert "both cards abort at the same shape" not in text, (
            f"{name} must not claim both cards abort at the same shape -- the pvjr sweep showed the "
            "B70 running several shapes the B50 could not"
        )
        assert "nothing measured runs with a demand above the zone" not in text, (
            f"{name} must not claim nothing measured runs with a demand above the SCRATCH zone -- the "
            "pvjr sweep ran p6144/p7168 (and more) clean on both cards despite exceeding the 512 MiB "
            "zone"
        )
        assert "928 MiB" in text, f"{name} must document the decided reserve, R = 928 MiB"
        assert "headroom-limited" in text, f"{name} must use the new remediation label \"headroom-limited\""


# Shared between the positive check and its mutation witness below, so the
# two cannot drift apart -- a fragment edited in one place is edited in
# both, rather than the witness carrying its own, independently-typed copy
# that could go stale (or start passing vacuously) if the positive check's
# wording ever changes.
_WARN_GUARD_OFF_FRAGMENT = "turns the runtime-context non-FA attention scratch guard off"
_WARN_NAMES_VAR_FRAGMENT = "GGML_SYCL_NONFA_ATTN_SCRATCH_MB"


def _env_mb_override_body_norm(code: str = CACHE_CPP_CODE) -> str:
    """Bound env_mb_override()'s own body (comment-stripped, whitespace-
    normalized) -- from its definition to the next file-scope function
    after it (nonfa_attn_scratch_mb_override(), the first of its two
    memoizing wrappers), so a match cannot come from some unrelated later
    WARN in the file. `code` defaults to the real, already comment-stripped
    unified-cache.cpp; the mutation witness below passes a mutated,
    already comment-stripped copy through this same accessor instead of
    re-deriving its own find/slice/bound logic, so the two cannot drift
    apart."""
    func_start = code.find("static long env_mb_override(const char * name) {")
    assert func_start != -1, "env_mb_override() definition not found in unified-cache.cpp"
    next_func = code.find("static long nonfa_attn_scratch_mb_override(", func_start + 1)
    assert next_func != -1, "could not bound env_mb_override()'s body"
    return _normalize_ws(code[func_start:next_func])


def test_explicit_zero_warn_names_the_nonfa_guard():
    """llama.cpp-wkrx: env_mb_override()'s explicit-0 WARN string is shared
    between GGML_SYCL_NONFA_ATTN_SCRATCH_MB and
    GGML_SYCL_ONEDNN_GRAPH_ZONE_MB (both route through this one formula),
    but its old parenthetical -- "this disables the consumer's own floor" --
    understated the effect for the non-FA guard: an explicit 0 there does
    not just drop its 16 MiB floor, it turns the whole runtime-context
    non-FA attention scratch refusal off. The WARN text must name that guard
    and say so, while staying accurate for the oneDNN Graph-scratch sibling
    too (it still only loses ITS floor)."""
    body_norm = _env_mb_override_body_norm()
    assert _WARN_GUARD_OFF_FRAGMENT in body_norm, (
        "env_mb_override()'s explicit-0 WARN must say that GGML_SYCL_NONFA_ATTN_SCRATCH_MB=0 turns "
        "the runtime-context non-FA attention scratch guard off entirely, not just its 16 MiB floor -- "
        "the shared WARN text previously said only \"this disables the consumer's own floor\", which "
        "understated the effect for this specific consumer (llama.cpp-wkrx)"
    )
    assert _WARN_NAMES_VAR_FRAGMENT in body_norm, (
        "the WARN's explicit-0 clarification must name GGML_SYCL_NONFA_ATTN_SCRATCH_MB specifically, "
        "not just describe \"the consumer\" in the abstract"
    )


def test_explicit_zero_warn_has_a_mutation_witness() -> None:
    """Mutation witness for the check above: proves it would actually catch
    a reversion to the old, understating WARN text -- "this disables the
    consumer's own floor", with no mention of the guard it turns off --
    rather than only ever passing on the current, correct source.

    Mutates the RAW source (the exact three C string literal lines) and
    re-derives body_norm through the same _env_mb_override_body_norm()
    accessor the positive check above uses, rather than hand-building the
    already-normalized text or re-deriving its own find/slice/bound copy --
    both would be fragile to (respectively) exactly how
    strip_comments()/_normalize_ws() join adjacent string literals across a
    line boundary, and to the two implementations drifting apart."""
    current_block = (
        '            "[UNIFIED-CACHE] %s=0 -- using an explicit 0 MB override (not \\"unset\\"; the '
        'consumer treats 0 "\n'
        '            "as an explicit disable -- for GGML_SYCL_NONFA_ATTN_SCRATCH_MB that "\n'
        '            "turns the runtime-context non-FA attention scratch guard off entirely, not just '
        'its 16 MiB floor)\\n",\n'
    )
    assert current_block in CACHE_CPP, "mutation target string not found -- update this witness"
    old_block = (
        '            "[UNIFIED-CACHE] %s=0 -- using an explicit 0 MB override (not \\"unset\\"; this '
        'disables the "\n'
        '            "consumer\'s own floor)\\n",\n'
    )
    mutated_raw = CACHE_CPP.replace(current_block, old_block, 1)
    assert mutated_raw != CACHE_CPP

    mutated_body_norm = _env_mb_override_body_norm(strip_comments(mutated_raw))

    assert _WARN_GUARD_OFF_FRAGMENT not in mutated_body_norm, (
        "mutation witness is broken: the reverted (pre-fix) text still trips the positive check above"
    )
    assert _WARN_NAMES_VAR_FRAGMENT not in mutated_body_norm, (
        "mutation witness is broken: the reverted (pre-fix) text still trips the check's second "
        "assertion (that the WARN names GGML_SYCL_NONFA_ATTN_SCRATCH_MB specifically)"
    )
