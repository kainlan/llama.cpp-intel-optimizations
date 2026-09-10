"""Source contract for llama.cpp-ibj0: the PP MoE oneDNN scratch ring must be
re-planned for the RUNTIME n_ubatch inside ggml_backend_sycl_set_runtime_context()
-- the same transaction that re-plans KV and the non-FA attention scratch --
not left fixed at the loader's own load-time n_ubatch (src/llama-model.cpp
hardcodes inventory.n_ubatch = 512). Without this, the first prefill at a
larger runtime n_ubatch hits the ring's own admission refusal
(moe-scratch-admission.hpp's "activation-cap"/"output-cap") and llama_decode
returns -3 with nothing printed at default verbosity (see this ticket's repro,
scratchpad ibj0-repro-ub1024-v.log).

Host-only, pure text assertions -- no SYCL device required, matching
test-sycl-nonfa-attn-scratch-guard-source.py's pytest-collectible pattern
(llama_test_pytest hands this file to pytest.main(), so checks must live
inside a test_*() function or pytest's "no tests collected" (exit 5) is
scored as a failure by design, not a skip).

Checks run against COMMENT-STRIPPED text (see strip_comments(), copied
verbatim from test-sycl-nonfa-attn-scratch-guard-source.py) so a positive
structural check cannot be fooled by prose that quotes a call the code does
not actually make.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GGML_SYCL_H = (ROOT / "ggml/include/ggml-sycl.h").read_text()
CACHE_HPP = (ROOT / "ggml/src/ggml-sycl/unified-cache.hpp").read_text()
CACHE_CPP = (ROOT / "ggml/src/ggml-sycl/unified-cache.cpp").read_text()
# llama.cpp-oyfl: plain file I/O, not the codescout index -- CLAUDE.md
# documents that index (and search_text's live scan) as blind/oversized for
# this specific ~100k-line file, so a tool-assisted search here would
# silently miss real occurrences.
GGML_SYCL_CPP = (ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()
LLAMA_MODEL_CPP = (ROOT / "src/llama-model.cpp").read_text()


# Single left-to-right alternation, not two sequential passes -- see
# test-sycl-onednn-graph-allocator-source.py's comment on why a block-comment
# pass run first can swallow a `/*` that appears inside a `//` comment's own
# prose (e.g. a glob like `weight-reclaim/*`).
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
GGML_SYCL_CPP_CODE = strip_comments(GGML_SYCL_CPP)
LLAMA_MODEL_CPP_CODE = strip_comments(LLAMA_MODEL_CPP)


def _normalize_ws(text: str) -> str:
    """Collapse all whitespace runs to a single space, so a call or
    declaration re-wrapped by clang-format across lines still matches a
    single-line pattern."""
    return re.sub(r"\s+", " ", text)


def _bounded_body(code: str, start_marker: str, end_marker: str, *, after: int = 0) -> str:
    """Slice `code` from `start_marker` to the next `end_marker` after it,
    raising a clear assertion if either is not found -- shared by every check
    below so a match cannot silently come from some unrelated, later call
    site elsewhere in the (large) file."""
    start = code.find(start_marker, after)
    assert start != -1, f"{start_marker!r} not found"
    end = code.find(end_marker, start + 1)
    assert end != -1, f"could not bound {start_marker!r}'s body (looked for {end_marker!r})"
    return code[start:end]


def _runtime_context_body() -> str:
    return _bounded_body(
        GGML_SYCL_CPP_CODE,
        "void ggml_backend_sycl_set_runtime_context(",
        "ggml_backend_sycl_set_runtime_context_for_model(",
    )


def _replan_ring_fn_body() -> str:
    return _bounded_body(
        GGML_SYCL_CPP_CODE,
        "static bool ggml_sycl_replan_pp_moe_onednn_ring(",
        "void ggml_backend_sycl_set_runtime_context(",
    )


def test_transaction_calls_replan_after_nonfa_and_before_mmid_materialize():
    """ggml_backend_sycl_set_runtime_context() must call the ring re-plan
    AFTER ggml_sycl_check_nonfa_attn_scratch() (so it sees the post-nonfa-
    guard plan) and BEFORE the MMID workspace materialization call that
    finalizes the plan for publication -- inserting it later would mean the
    plan could be published (or the transaction could return early for an
    unrelated reason) without the ring ever having been checked against the
    real runtime n_ubatch."""
    body_norm = _normalize_ws(_runtime_context_body())

    nonfa_idx = body_norm.find("ggml_sycl_check_nonfa_attn_scratch(")
    replan_idx = body_norm.find("ggml_sycl_replan_pp_moe_onednn_ring(")
    mmid_materialize_idx = body_norm.find("ggml_sycl_materialize_published_mmid_workspaces(")

    assert nonfa_idx != -1, "ggml_backend_sycl_set_runtime_context() must call ggml_sycl_check_nonfa_attn_scratch()"
    assert replan_idx != -1, (
        "ggml_backend_sycl_set_runtime_context() must call ggml_sycl_replan_pp_moe_onednn_ring() -- "
        "the PP MoE oneDNN scratch ring is never re-planned for the runtime n_ubatch otherwise"
    )
    assert mmid_materialize_idx != -1, (
        "ggml_sycl_materialize_published_mmid_workspaces() call not found -- could not bound the check"
    )

    assert nonfa_idx < replan_idx < mmid_materialize_idx, (
        "the ring re-plan must run AFTER ggml_sycl_check_nonfa_attn_scratch() and BEFORE the MMID "
        "workspace materialization step that finalizes the plan for publication -- found nonfa at "
        f"{nonfa_idx}, replan at {replan_idx}, mmid-materialize at {mmid_materialize_idx}"
    )


def test_transaction_refuses_when_replan_fails():
    """A failed re-plan must abort the transaction (return, not merely log)
    -- llama_context sees this as a non-OK lifecycle result and throws at
    context construction, rather than silently publishing a plan whose ring
    the first prefill will refuse anyway."""
    body_norm = _normalize_ws(_runtime_context_body())
    assert re.search(
        r"if\s*\(\s*!\s*ggml_sycl_replan_pp_moe_onednn_ring\([^)]*\)\s*\)\s*\{\s*return\s*;",
        body_norm,
    ), "a failed ggml_sycl_replan_pp_moe_onednn_ring() call must return from the transaction immediately"


def test_replan_call_has_a_mutation_witness():
    """Mutation witness for the two checks above: proves they would actually
    catch the re-plan call being removed, rather than only ever passing on
    the current, correct source."""
    raw = GGML_SYCL_CPP
    call_block = (
        "    if (!ggml_sycl_replan_pp_moe_onednn_ring(ctx->device, next_kv_info.n_ubatch)) {\n"
        "        return;  // refusal already logged with the largest fitting -ub\n"
        "    }\n\n"
    )
    assert call_block in raw, "mutation target block not found -- update this witness to match the real source"
    mutated_raw = raw.replace(call_block, "", 1)
    assert mutated_raw != raw

    mutated_code = strip_comments(mutated_raw)
    mutated_body_norm = _normalize_ws(
        _bounded_body(
            mutated_code,
            "void ggml_backend_sycl_set_runtime_context(",
            "ggml_backend_sycl_set_runtime_context_for_model(",
        )
    )
    assert "ggml_sycl_replan_pp_moe_onednn_ring(" not in mutated_body_norm, (
        "mutation witness is broken: deleting the call site left a reference to "
        "ggml_sycl_replan_pp_moe_onednn_ring() behind"
    )


def test_replan_reads_before_writing_and_restores_on_failure():
    """ggml_sycl_replan_pp_moe_onednn_ring() must record the OLD planned slot
    sizes (and n_ubatch) BEFORE overwriting them, and restore them on a
    failed reserve -- reserve_pp_moe_onednn_scratch()'s own admission
    preflight compares its REQUESTED shape against the CURRENTLY PLANNED
    ceiling and refuses anything larger, so growing the ring means raising
    the ceiling first (unified_cache_set_planned_pp_moe_onednn_scratch())
    and only then reserving; a refused grow that left the ceiling raised
    would claim a ring larger than what is physically allocated."""
    body_norm = _normalize_ws(_replan_ring_fn_body())

    assert "unified_cache_get_planned_pp_moe_onednn_activation_slot_bytes(device)" in body_norm, (
        "must read the OLD planned activation slot bytes before overwriting them"
    )
    assert "unified_cache_get_planned_pp_moe_onednn_output_slot_bytes(device)" in body_norm, (
        "must read the OLD planned output slot bytes before overwriting them"
    )

    set_indices = [
        m.start() for m in re.finditer(r"unified_cache_set_planned_pp_moe_onednn_scratch\(", body_norm)
    ]
    assert len(set_indices) >= 2, (
        "expected at least two unified_cache_set_planned_pp_moe_onednn_scratch() calls -- one to raise "
        "the ceiling before reserving, one to restore it on a failed reserve"
    )

    reserve_idx = body_norm.find("reserve_pp_moe_onednn_scratch(")
    assert reserve_idx != -1, "must call reserve_pp_moe_onednn_scratch()"
    assert set_indices[0] < reserve_idx, (
        "the planned ceiling must be raised BEFORE calling reserve_pp_moe_onednn_scratch() -- its own "
        "admission preflight compares REQUESTED against the CURRENTLY PLANNED shape and refuses a "
        "request larger than a ceiling that has not yet been raised"
    )
    assert any(idx > reserve_idx for idx in set_indices), (
        "must call unified_cache_set_planned_pp_moe_onednn_scratch() again AFTER the reserve attempt, "
        "to restore the old ceiling on failure"
    )


def test_refusal_message_contains_the_required_fragments():
    """The refusal must name the failing n_ubatch and, when a fit exists,
    the largest -ub that does -- the same 'named remediation' shape as the
    KV budget refusal's 'the largest context that fits is about -c %u'."""
    body_norm = _normalize_ws(_replan_ring_fn_body())
    assert "runtime context update rejected" in body_norm, (
        "the refusal must say \"runtime context update rejected\", matching the pre-existing KV/non-FA "
        "refusal family's wording"
    )
    assert "n_ubatch=%u" in body_norm, "the refusal must name the failing n_ubatch"
    assert "the largest -ub that fits is about %u" in body_norm, (
        "the refusal must report the largest -ub that fits, when one exists"
    )
    assert "unified_cache_largest_fitting_n_ubatch_for_pp_moe_onednn(" in body_norm, (
        "the largest-fitting figure must come from the shared inverse function, not a hand-derived one"
    )
    assert "GGML_LOG_ERROR(" in body_norm, "the refusal must log at ERROR (INFO is dropped at default verbosity)"


def test_success_warn_message_contains_the_required_fragments():
    """A successful re-plan must be visible at default verbosity too -- WARN,
    not INFO, exactly once, naming the new n_ubatch and the resulting sizes."""
    body_norm = _normalize_ws(_replan_ring_fn_body())
    assert "GGML_LOG_WARN(" in body_norm, "a successful re-plan must log at WARN"
    assert "PP MoE oneDNN scratch ring re-planned for n_ubatch=%u" in body_norm, (
        "the success WARN must name the fragment \"PP MoE oneDNN scratch ring re-planned for n_ubatch=%u\""
    )


def test_replan_is_idempotent_and_skips_dense_models():
    """A dense model (weight_slot_bytes == 0, no MoE PP ring ever planned)
    and a repeated call at the SAME n_ubatch must both return true without
    calling reserve -- the first because there is nothing to re-plan, the
    second so a transaction re-run at an unchanged n_ubatch (e.g. the narrow
    flash-attn re-check path) is a no-op rather than repeating the
    ceiling-raise/reserve dance every time."""
    body_norm = _normalize_ws(_replan_ring_fn_body())
    assert re.search(
        r"if\s*\(\s*weight_slot_bytes\s*==\s*0\s*\)\s*\{\s*return\s+true\s*;", body_norm
    ), "weight_slot_bytes == 0 (dense model) must return true immediately, before any reserve attempt"
    assert re.search(
        r"if\s*\(\s*n_ubatch\s*==\s*ggml_sycl::unified_cache_get_planned_pp_moe_onednn_n_ubatch\(device\)\s*\)\s*"
        r"\{\s*return\s+true\s*;",
        body_norm,
    ), (
        "n_ubatch already equal to the planned n_ubatch must return true immediately (idempotent re-plan)"
    )


def test_replan_is_bidirectional_and_releases_the_ring_on_shrink():
    """llama.cpp-nphx Task 3 spike (comment c-wgxn): a SHRINK -- the runtime
    n_ubatch smaller than the ring's currently-planned one -- must release
    the physical ring BEFORE reserving the smaller size, or
    reserve_pp_moe_onednn_scratch()'s own 'already sufficient, reuse without
    reallocating' fast path silently keeps the larger, discarded ring (and
    the RUNTIME zone's charge for it) forever. This matters because Task 4b's
    ascending-ladder trial can settle on a smaller n_ubatch than the largest
    one it tried."""
    body_norm = _normalize_ws(_replan_ring_fn_body())

    assert "release_pp_moe_onednn_scratch_ring(" in body_norm, (
        "the re-plan must call release_pp_moe_onednn_scratch_ring() somewhere -- the shrink path has no "
        "other way to force reserve_pp_moe_onednn_scratch() to actually shrink the physical ring"
    )
    release_match = re.search(
        r"if\s*\(\s*n_ubatch\s*<\s*old_n_ubatch\s*&&\s*!\s*cache->release_pp_moe_onednn_scratch_ring\(\)\s*\)",
        body_norm,
    )
    assert release_match is not None, (
        "the release call must be gated on n_ubatch < old_n_ubatch (a SHRINK only -- a grow does not need "
        "it, reserve_pp_moe_onednn_scratch()'s own allocate-new-then-retire-old path already frees a "
        "smaller predecessor correctly)"
    )

    set_first_idx = body_norm.find("unified_cache_set_planned_pp_moe_onednn_scratch(")
    reserve_idx = body_norm.find("reserve_pp_moe_onednn_scratch(")
    assert set_first_idx != -1 and reserve_idx != -1
    assert release_match.start() < set_first_idx < reserve_idx, (
        "the release call must precede BOTH the ceiling raise/lower and the reserve attempt -- releasing "
        "after either would either shrink a ring the new (still-old) ceiling claims is bigger, or race "
        "reserve_pp_moe_onednn_scratch()'s own fast path"
    )


def test_release_ring_function_is_exported_and_refuses_when_busy():
    """unified_cache::release_pp_moe_onednn_scratch_ring() (and its
    free-function wrapper) must exist, be exported (not file-static), and
    refuse (return false, ring untouched) when any slot is still claimed --
    mirroring shutdown_resources()'s own identical safety for the same
    resource."""
    hpp_norm = _normalize_ws(CACHE_HPP_CODE)
    cpp_norm = _normalize_ws(CACHE_CPP_CODE)

    assert "bool release_pp_moe_onednn_scratch_ring();" in hpp_norm, (
        "unified_cache::release_pp_moe_onednn_scratch_ring() must be declared in the class"
    )
    assert "bool unified_cache_release_pp_moe_onednn_scratch_ring(int device_id);" in hpp_norm, (
        "the free-function wrapper must be declared in unified-cache.hpp"
    )

    def _has_static_before(text: str, name: str) -> bool:
        return bool(re.search(r"\bstatic\b[^;{}]*?\b" + re.escape(name) + r"\s*\(", text))

    assert not _has_static_before(cpp_norm, "unified_cache_release_pp_moe_onednn_scratch_ring"), (
        "the free-function wrapper must not be file-static"
    )

    body = _bounded_body(
        CACHE_CPP_CODE, "bool unified_cache::release_pp_moe_onednn_scratch_ring()", "bool unified_cache::claim_pp_moe_onednn_scratch_slot("
    )
    body_norm = _normalize_ws(body)
    assert re.search(r"if\s*\(\s*slot\.refcount\s*!=\s*0\s*\)\s*\{\s*return\s+false\s*;", body_norm), (
        "must refuse (return false) when a currently-held slot is still claimed (refcount != 0), before "
        "touching the ring"
    )


def test_stale_batched_refusal_is_warn_not_info():
    """llama.cpp-ibj0 acceptance criterion: the once-per-process refusal at
    the batched PP MoE oneDNN executor's own admission check must be
    GGML_LOG_WARN, so if the admission ever refuses again (e.g. this task's
    re-plan itself has a bug) the reason is visible at default verbosity,
    not silently dropped as INFO."""
    idx = GGML_SYCL_CPP_CODE.find('"[SYCL] refusing unplanned PP MoE oneDNN batched scratch')
    assert idx != -1, "the batched PP MoE oneDNN refusal string not found"
    # The macro call opens on an earlier line than the string literal
    # (clang-format wraps the format string onto its own line); search
    # backwards from the string for the nearest GGML_LOG_* macro name.
    preceding = GGML_SYCL_CPP_CODE[max(0, idx - 200):idx]
    macro_match = list(re.finditer(r"GGML_LOG_(WARN|INFO|ERROR)\s*\(", preceding))
    assert macro_match, "could not find the logging macro preceding the batched refusal string"
    assert macro_match[-1].group(1) == "WARN", (
        "the batched PP MoE oneDNN refusal must be GGML_LOG_WARN, not "
        f"GGML_LOG_{macro_match[-1].group(1)} -- GGML_LOG_INFO is dropped at default verbosity"
    )


def test_loader_stores_both_per_row_fields():
    """The loader (src/llama-model.cpp) must store BOTH new per-row fields
    next to the existing slot formulas, tied together with an assertion so
    the two cannot silently drift apart."""
    body = _bounded_body(
        LLAMA_MODEL_CPP_CODE,
        "static void llama_model_sycl_populate_inventory(",
        "static void llama_model_sycl_apply_inventory(",
    )
    assert "inventory.pp_moe_onednn_activation_bytes_per_row =" in body, (
        "the loader must store inventory.pp_moe_onednn_activation_bytes_per_row"
    )
    assert "inventory.pp_moe_onednn_output_bytes_per_row =" in body, (
        "the loader must store inventory.pp_moe_onednn_output_bytes_per_row"
    )
    body_norm = _normalize_ws(body)
    assert re.search(
        r"GGML_ASSERT\(\s*inventory\.pp_moe_onednn_activation_slot_bytes\s*==.*?"
        r"pp_moe_onednn_activation_bytes_per_row",
        body_norm,
    ), (
        "the activation slot formula and the new per-row field must be tied together with a GGML_ASSERT, "
        "so the two cannot drift apart silently"
    )
    assert re.search(
        r"GGML_ASSERT\(\s*inventory\.pp_moe_onednn_output_slot_bytes\s*==.*?"
        r"pp_moe_onednn_output_bytes_per_row",
        body_norm,
    ), (
        "the output slot formula and the new per-row field must be tied together with a GGML_ASSERT, "
        "so the two cannot drift apart silently"
    )


def test_inventory_struct_declares_both_per_row_fields():
    """ggml_sycl_tensor_inventory (ggml-sycl.h) must declare both new fields
    -- without them the loader above cannot compile against a real struct
    member."""
    assert "pp_moe_onednn_activation_bytes_per_row;" in GGML_SYCL_H_CODE
    assert "pp_moe_onednn_output_bytes_per_row;" in GGML_SYCL_H_CODE


def test_populate_inventory_globals_wires_the_row_bytes():
    """populate_inventory_globals() (ggml-sycl.cpp) must carry the loader's
    per-row fields into the unified cache -- declaring them on the struct and
    setting them in the pure functions is not enough if nothing at model-load
    time actually publishes them for a real model."""
    body = _bounded_body(GGML_SYCL_CPP_CODE, "static void populate_inventory_globals(", "g_model_n_layer")
    body_norm = _normalize_ws(body)
    assert "unified_cache_set_planned_pp_moe_onednn_row_bytes(" in body_norm, (
        "populate_inventory_globals() must call unified_cache_set_planned_pp_moe_onednn_row_bytes() "
        "so a real model load actually publishes the per-row bytes"
    )
    assert "inventory->pp_moe_onednn_activation_bytes_per_row" in body_norm
    assert "inventory->pp_moe_onednn_output_bytes_per_row" in body_norm
    assert "unified_cache_set_planned_pp_moe_onednn_n_ubatch(" in body_norm, (
        "populate_inventory_globals() must record the load-time plan's own n_ubatch as the ring's "
        "initially-planned n_ubatch, or the re-plan's idempotence check starts every fresh model load "
        "believing nothing was ever planned"
    )


def test_new_functions_are_not_file_static_in_the_shared_header():
    """The new unified-cache functions must be exported (declared, and not
    `static`, in both the header and the implementation) -- ggml-sycl.cpp is
    a different translation unit and must be able to call them directly."""
    hpp_norm = _normalize_ws(CACHE_HPP_CODE)
    cpp_norm = _normalize_ws(CACHE_CPP_CODE)

    def _has_static_before(text: str, name: str) -> bool:
        return bool(re.search(r"\bstatic\b[^;{}]*?\b" + re.escape(name) + r"\s*\(", text))

    checked_names = (
        "unified_cache_set_planned_pp_moe_onednn_row_bytes",
        "unified_cache_get_planned_pp_moe_onednn_activation_bytes_per_row",
        "unified_cache_get_planned_pp_moe_onednn_output_bytes_per_row",
        "unified_cache_set_planned_pp_moe_onednn_n_ubatch",
        "unified_cache_get_planned_pp_moe_onednn_n_ubatch",
        "unified_cache_pp_moe_onednn_slots_for_ubatch",
        "unified_cache_largest_fitting_n_ubatch_for_pp_moe_onednn",
    )
    for name in checked_names:
        assert f"{name}(" in hpp_norm, f"{name}() must be declared in unified-cache.hpp"
        assert f"{name}(" in cpp_norm, f"{name}() must be defined in unified-cache.cpp"
        assert not _has_static_before(cpp_norm, name), f"{name}() must not be file-static (.cpp)"
        assert not _has_static_before(hpp_norm, name), f"{name}() must not be declared static (.hpp)"
