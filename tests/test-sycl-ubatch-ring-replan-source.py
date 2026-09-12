"""Source contract for llama.cpp-ibj0: the PP MoE oneDNN scratch ring must be
re-planned for the RUNTIME n_ubatch inside the runtime-context transaction
(llama.cpp-tsfl: as of that task, the actual admission logic --
including this ring re-plan -- lives in
ggml_sycl_run_runtime_context_transaction(); ggml_backend_sycl_set_runtime_
context() is now a thin wrapper around it) -- the same transaction that
re-plans KV and the non-FA attention scratch -- not left fixed at the
loader's own load-time n_ubatch (src/llama-model.cpp hardcodes
inventory.n_ubatch = 512). Without this, the first prefill at a larger
runtime n_ubatch hits the ring's own admission refusal
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


def _has_static_before(text: str, name: str) -> bool:
    """True if `name(` is preceded, within the same statement/scope
    ([^;{}] bounded), by a `static` keyword -- i.e. `name` would have
    internal linkage rather than being exported. llama.cpp-ibj0 quality
    round 1 Q4: hoisted here from two identical local copies (one per
    check that used it) so the two cannot drift apart."""
    return bool(re.search(r"\bstatic\b[^;{}]*?\b" + re.escape(name) + r"\s*\(", text))


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


def _body_of(raw: str, start_marker: str, end_marker: str) -> str:
    """Comment-strip `raw`, bound the result from `start_marker` to the next
    `end_marker`, and whitespace-normalize -- the strip+bound+normalize
    pipeline every mutation witness needs to re-derive a checkable body from
    its freshly mutated RAW string. llama.cpp-ibj0 quality round 4 Q6.6:
    hoisted here so each witness's own copy of this three-call chain (with
    its own copy of the boundary literals) cannot drift from the others; the
    module-level *_CODE globals are already comment-stripped once at import
    time, so a plain (non-witness) check uses `_bounded_body` on those
    directly instead of this function."""
    return _normalize_ws(_bounded_body(strip_comments(raw), start_marker, end_marker))


# llama.cpp-ibj0 quality round 4 Q6.6: boundary literals named once, used by
# both the plain body-extraction helpers below and every mutation witness's
# _body_of() call, so a boundary string is spelled in exactly one place.
# llama.cpp-jumy: return type changed from bool to the internal
# ggml_sycl_ring_replan_result enum (OK / RELEASE_REFUSED / DOES_NOT_FIT) --
# the marker must match the real signature or every downstream
# _bounded_body() call fails loudly with "'<marker>' not found".
_REPLAN_START = "static ggml_sycl_ring_replan_result ggml_sycl_replan_pp_moe_onednn_ring("
# llama.cpp-tsfl: the transaction body this file's checks pin was extracted
# out of ggml_backend_sycl_set_runtime_context() into a shared static
# function, ggml_sycl_run_runtime_context_transaction() (also called, in
# probe mode, by the new ggml_backend_sycl_probe_runtime_context_for_model())
# -- ggml_backend_sycl_set_runtime_context() is now a thin wrapper around it.
# This constant's NAME is kept (it is used throughout this file as both the
# admission-logic start marker and the ring function's own end marker) but
# its VALUE now points at the real logic.
# llama.cpp-tsfl round 1 F6: the function's return type changed from bool
# to the internal ggml_sycl_txn_result enum -- the marker must match the
# real signature or every downstream _bounded_body() call fails loudly with
# "'<marker>' not found".
_RUNTIME_CONTEXT_START = "static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction("
# llama.cpp-tsfl: the transaction body's own end marker is the
# now-thin ggml_backend_sycl_set_runtime_context() wrapper's start, not
# ggml_backend_sycl_set_runtime_context_for_model() -- ".find()" would
# otherwise silently span FOUR functions (the real transaction body, the
# wrapper, ggml_backend_sycl_probe_runtime_context_for_model(), and
# ggml_backend_sycl_auto_ubatch_enabled()), because none of the latter three
# contain "ggml_backend_sycl_set_runtime_context_for_model(" either, so
# .find() just kept going until it hit the real thing.
_SET_RUNTIME_CONTEXT_WRAPPER_START = "void ggml_backend_sycl_set_runtime_context("
_RESERVE_PP_MOE_START = "bool unified_cache::reserve_pp_moe_onednn_scratch("
_RELEASE_RING_START = "bool unified_cache::release_pp_moe_onednn_scratch_ring("
_UNIFIED_ALLOC_START = "bool unified_alloc(const alloc_request & req_in, alloc_handle * out) {"
_ACQUIRE_OFFLOAD_BUFFER_START = "bool acquire_offload_buffer("


def _runtime_context_body() -> str:
    # llama.cpp-tsfl: assert both bounds explicitly and that the
    # slice is strictly forward -- _bounded_body() already raises a clear
    # AssertionError if either marker is entirely missing, so this is a
    # second, explicit check that the slice actually landed where intended.
    start = GGML_SYCL_CPP_CODE.find(_RUNTIME_CONTEXT_START)
    assert start != -1, f"{_RUNTIME_CONTEXT_START!r} not found"
    end = GGML_SYCL_CPP_CODE.find(_SET_RUNTIME_CONTEXT_WRAPPER_START, start + 1)
    assert end != -1, f"{_SET_RUNTIME_CONTEXT_WRAPPER_START!r} not found after the transaction body"
    assert start < end, "the transaction body's start marker must precede its end marker"
    return _bounded_body(GGML_SYCL_CPP_CODE, _RUNTIME_CONTEXT_START, _SET_RUNTIME_CONTEXT_WRAPPER_START)


def _replan_ring_fn_body() -> str:
    return _bounded_body(GGML_SYCL_CPP_CODE, _REPLAN_START, _RUNTIME_CONTEXT_START)


def test_transaction_calls_replan_after_nonfa_and_before_mmid_materialize():
    """ggml_sycl_run_runtime_context_transaction() (llama.cpp-tsfl: the
    shared body ggml_backend_sycl_set_runtime_context() now merely
    wraps) must call the ring re-plan AFTER ggml_sycl_check_nonfa_attn_
    scratch() (so it sees the post-nonfa-guard plan) and BEFORE the MMID
    workspace materialization call that finalizes the plan for publication
    -- inserting it later would mean the plan could be published (or the
    transaction could return early for an unrelated reason) without the
    ring ever having been checked against the real runtime n_ubatch."""
    body_norm = _normalize_ws(_runtime_context_body())

    nonfa_idx = body_norm.find("ggml_sycl_check_nonfa_attn_scratch(")
    replan_idx = body_norm.find("ggml_sycl_replan_pp_moe_onednn_ring(")
    mmid_materialize_idx = body_norm.find("ggml_sycl_materialize_published_mmid_workspaces(")

    assert nonfa_idx != -1, (
        "ggml_sycl_run_runtime_context_transaction() must call ggml_sycl_check_nonfa_attn_scratch()"
    )
    assert replan_idx != -1, (
        "ggml_sycl_run_runtime_context_transaction() must call ggml_sycl_replan_pp_moe_onednn_ring() -- "
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
    # llama.cpp-jumy: the call site now captures the ring re-plan's own
    # ggml_sycl_ring_replan_result (RELEASE_REFUSED classified BUSY, every
    # other non-OK value REFUSED) rather than testing a bare `!` on the
    # call directly -- match the DOES_NOT_FIT-shaped branch, which is the
    # one this test's docstring cares about ("a failed re-plan must abort
    # the transaction"). test_ring_release_refused_is_classified_busy (in
    # tests/test-sycl-compute-buffer-fallback-source.py) pins the BUSY
    # branch on its own.
    assert re.search(
        r"const\s+ggml_sycl_ring_replan_result\s+ring_replan_result\s*=\s*"
        r"ggml_sycl_replan_pp_moe_onednn_ring\([^;]*;\s*"
        r"if\s*\(\s*ring_replan_result\s*==\s*ggml_sycl_ring_replan_result::RELEASE_REFUSED\s*\)\s*\{\s*"
        r"return\s+busy\(",
        body_norm,
    ), "the ring release-refused shape must return busy(), not refuse() -- transient, a caller may retry"
    assert re.search(
        r"if\s*\(\s*ring_replan_result\s*!=\s*ggml_sycl_ring_replan_result::OK\s*\)\s*\{\s*return\s+refuse\(",
        body_norm,
    ), "a failed ggml_sycl_replan_pp_moe_onednn_ring() call must return from the transaction immediately"


def test_replan_call_has_a_mutation_witness():
    """Mutation witness for the two checks above: proves they would actually
    catch the re-plan call being removed, rather than only ever passing on
    the current, correct source.

    Counts occurrences rather than asserting total absence: since spec round
    1 F8 and round 2 F10, plus llama.cpp-tsfl's own probe-mode rollback,
    ggml_sycl_replan_pp_moe_onednn_ring() is legitimately called FIVE times
    in this function (the original re-plan, the probe_mode branch's own
    rollback, plus three later publish-path rollback call sites) --
    deleting the original call site must drop the count by exactly one, to
    four, not zero."""
    raw = GGML_SYCL_CPP
    call_block = (
        "    const ggml_sycl_ring_replan_result ring_replan_result =\n"
        "        ggml_sycl_replan_pp_moe_onednn_ring(ctx->device, next_kv_info.n_ubatch, probe_mode);\n"
        "    if (ring_replan_result == ggml_sycl_ring_replan_result::RELEASE_REFUSED) {\n"
        "        // refusal already logged (ERROR normally, INFO in probe mode)\n"
        '        return busy("busy (PP MoE oneDNN scratch ring claimed by an in-flight dispatch)");\n'
        "    }\n"
        "    if (ring_replan_result != ggml_sycl_ring_replan_result::OK) {\n"
        "        // refusal already logged (ERROR normally, INFO in probe mode) with\n"
        "        // the largest fitting -ub\n"
        '        return refuse("PP MoE oneDNN scratch ring does not fit");\n'
        "    }\n\n"
    )
    assert call_block in raw, "mutation target block not found -- update this witness to match the real source"
    mutated_raw = raw.replace(call_block, "", 1)
    assert mutated_raw != raw

    def _call_count(raw_source: str) -> int:
        body_norm = _body_of(raw_source, _RUNTIME_CONTEXT_START, _SET_RUNTIME_CONTEXT_WRAPPER_START)
        return len(re.findall(r"ggml_sycl_replan_pp_moe_onednn_ring\(", body_norm))

    original_count = _call_count(raw)
    mutated_count = _call_count(mutated_raw)
    assert original_count == 5, (
        f"expected exactly five calls in the unmutated source (the re-plan itself, the probe_mode branch's "
        f"own rollback, plus three publish-path rollback call sites) -- found {original_count}; update this "
        "witness to match the real source"
    )
    assert mutated_count == original_count - 1, (
        "mutation witness is broken: deleting the call site must drop the count by exactly one -- found "
        f"{mutated_count} (expected {original_count - 1})"
    )


def test_replan_reads_before_writing_and_restores_on_failure():
    """ggml_sycl_replan_pp_moe_onednn_ring() must record the OLD planned slot
    sizes (and n_ubatch) BEFORE overwriting them, and restore them on a
    failed reserve -- reserve_pp_moe_onednn_scratch()'s own admission
    preflight compares its REQUESTED shape against the CURRENTLY PLANNED
    ceiling and refuses anything larger, so growing the ring means raising
    the ceiling first (unified_cache_set_planned_pp_moe_onednn_scratch())
    and only then reserving; a refused grow that left the ceiling raised
    would claim a ring larger than what is physically allocated.

    llama.cpp-ibj0 spec round 5 F15: identifies the Step 2 (ceiling raise
    for the NEW size) and Step 3 (reserve for the NEW size) calls by the
    slot-byte variables they pass, NOT by first-occurrence position. The
    round-5a refuse_and_restore() closure (F13) DEFINES a set/reserve pair
    using the OLD sizes textually BEFORE Step 2/3 even though it only RUNS
    on failure -- so `.find()`'s first match, and `set_indices[0]`, point
    at the closure's own internal restore pair, not at Step 2/3 at all.
    That fail-open let a mutant moving the real Step 2 ceiling-raise to
    AFTER the real Step 3 reserve -- the exact ordering mistake this check
    exists to catch -- pass 27/27 unmodified (see the mutation witness
    immediately below, which reproduces exactly that mutant)."""
    body_norm = _normalize_ws(_replan_ring_fn_body())

    assert "unified_cache_get_planned_pp_moe_onednn_activation_slot_bytes(device)" in body_norm, (
        "must read the OLD planned activation slot bytes before overwriting them"
    )
    assert "unified_cache_get_planned_pp_moe_onednn_output_slot_bytes(device)" in body_norm, (
        "must read the OLD planned output slot bytes before overwriting them"
    )

    step2_set_match = re.search(
        r"unified_cache_set_planned_pp_moe_onednn_scratch\(\s*device,\s*weight_slot_bytes,\s*"
        r"new_activation_slot_bytes",
        body_norm,
    )
    step3_reserve_match = re.search(
        r"reserve_pp_moe_onednn_scratch\(\s*weight_slot_bytes,\s*new_activation_slot_bytes", body_norm
    )
    assert step2_set_match is not None, "could not find the Step 2 ceiling-raise call for the NEW size"
    assert step3_reserve_match is not None, "could not find the Step 3 reserve call for the NEW size"
    assert step2_set_match.start() < step3_reserve_match.start(), (
        "the planned ceiling must be raised for the NEW size BEFORE reserve_pp_moe_onednn_scratch() is "
        "called for the NEW size -- its own admission preflight compares REQUESTED against the CURRENTLY "
        "PLANNED shape and refuses a request larger than a ceiling that has not yet been raised"
    )

    assert re.search(
        r"unified_cache_set_planned_pp_moe_onednn_scratch\(\s*device,\s*weight_slot_bytes,\s*"
        r"old_activation_slot_bytes",
        body_norm,
    ), (
        "must call unified_cache_set_planned_pp_moe_onednn_scratch() with the OLD sizes somewhere, to "
        "restore the ceiling on failure (the restore pair's exact placement and content is covered by "
        "test_replan_reserves_the_old_ring_on_a_failed_new_reserve)"
    )

    # llama.cpp-ibj0 spec round 6 Q6.1: pin that the FAILED Step 3 path
    # actually calls refuse_and_restore() -- presence checks for the OLD-size
    # restore pair (above) are satisfied by refuse_and_restore()'s own body
    # existing ANYWHERE, even if nothing on the Step-3-failure path ever
    # calls it. There are exactly two legitimate call sites: the F13
    # pre-check (before Step 3 ever runs) and this post-Step-3-failure
    # fallthrough; a mutant changing either to a bare `return false;` (ring
    # torn down, ceiling raised with no physical backing, nothing restored,
    # nothing logged) must change that count away from 2.
    refuse_calls = [m.start() for m in re.finditer(r"return\s+refuse_and_restore\(\)\s*;", body_norm)]
    assert len(refuse_calls) == 2, (
        "expected exactly two `return refuse_and_restore();` call sites (the F13 pre-check and the "
        f"post-Step-3-failure fallthrough) -- found {len(refuse_calls)}"
    )
    assert any(idx > step3_reserve_match.start() for idx in refuse_calls), (
        "the post-Step-3-failure path must call refuse_and_restore() -- a failed "
        "reserve_pp_moe_onednn_scratch() for the NEW size must restore the OLD ring and log the refusal, "
        "not silently return false and leave the ceiling raised with nothing physically backing it"
    )


def test_step3_failure_calls_refuse_and_restore_has_a_mutation_witness():
    """Mutation witness for llama.cpp-ibj0 spec round 6 Q6.1: proves the
    checks above would actually catch the post-Step-3-failure path silently
    returning false instead of restoring the old ring and logging the
    refusal -- reproduces the exact mutant the review found passing 29/29
    unmodified (the final `return refuse_and_restore();` changed to
    `return false;`)."""
    raw = GGML_SYCL_CPP
    # llama.cpp-tsfl: no longer anchored on the following function's name
    # (ggml_backend_sycl_set_runtime_context() is now a thin wrapper defined
    # much further below, after ggml_sycl_run_runtime_context_transaction())
    # -- "return refuse_and_restore();\n}\n" alone (function-closing brace
    # immediately after) is already unique in the file.
    original_tail = "    return refuse_and_restore();\n}\n"
    assert raw.count(original_tail) == 1, (
        f"mutation target block not found or not unique (count={raw.count(original_tail)}) -- update this "
        "witness to match the real source"
    )
    mutated_tail = "    return false;\n}\n"
    mutated_raw = raw.replace(original_tail, mutated_tail, 1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _REPLAN_START, _RUNTIME_CONTEXT_START)
    mutated_refuse_calls = [
        m.start() for m in re.finditer(r"return\s+refuse_and_restore\(\)\s*;", mutated_body_norm)
    ]
    assert len(mutated_refuse_calls) != 2, (
        "mutation witness is broken: the mutated source (Step-3-failure path no longer calling "
        f"refuse_and_restore()) should have a DIFFERENT count than 2 -- found {len(mutated_refuse_calls)}"
    )


def test_step2_before_step3_has_a_mutation_witness():
    """Mutation witness for the check above (llama.cpp-ibj0 spec round 5
    F15): proves it would actually catch the round-1 defect it exists to
    catch -- the Step 2 ceiling-raise for the NEW size moved to AFTER the
    Step 3 reserve for the NEW size succeeds, instead of before it. This
    is exactly the ordering mistake reserve_pp_moe_onednn_scratch()'s own
    admission preflight is supposed to prevent shipping (requesting the
    new size while the ceiling still says the old one)."""
    raw = GGML_SYCL_CPP
    original_block = (
        "    // Step 2: raise (or lower) the ceiling BEFORE reserving (see the\n"
        "    // function comment for why this order is required).\n"
        "    ggml_sycl::unified_cache_set_planned_pp_moe_onednn_scratch(device, weight_slot_bytes, "
        "new_activation_slot_bytes,\n"
        "                                                               new_output_slot_bytes, ring_depth);\n"
        "\n"
        "    // Step 3: reserve the NEW size, now that nothing of this ring's own is\n"
        "    // outstanding.\n"
        "    if (cache->reserve_pp_moe_onednn_scratch(weight_slot_bytes, new_activation_slot_bytes, "
        "new_output_slot_bytes,\n"
        "                                             ring_depth)) {\n"
    )
    assert original_block in raw, "mutation target block not found -- update this witness to match the real source"
    mutated_block = (
        "    // Step 3: reserve the NEW size, now that nothing of this ring's own is\n"
        "    // outstanding.\n"
        "    if (cache->reserve_pp_moe_onednn_scratch(weight_slot_bytes, new_activation_slot_bytes, "
        "new_output_slot_bytes,\n"
        "                                             ring_depth)) {\n"
        "        // Step 2 (MUTATED, moved here): raise the ceiling AFTER reserving.\n"
        "        ggml_sycl::unified_cache_set_planned_pp_moe_onednn_scratch(device, weight_slot_bytes, "
        "new_activation_slot_bytes,\n"
        "                                                                   new_output_slot_bytes, ring_depth);\n"
    )
    mutated_raw = raw.replace(original_block, mutated_block, 1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _REPLAN_START, _RUNTIME_CONTEXT_START)
    step2_set_match = re.search(
        r"unified_cache_set_planned_pp_moe_onednn_scratch\(\s*device,\s*weight_slot_bytes,\s*"
        r"new_activation_slot_bytes",
        mutated_body_norm,
    )
    step3_reserve_match = re.search(
        r"reserve_pp_moe_onednn_scratch\(\s*weight_slot_bytes,\s*new_activation_slot_bytes", mutated_body_norm
    )
    assert step2_set_match is not None and step3_reserve_match is not None, (
        "mutation witness is broken: could not find both calls in the mutated source"
    )
    assert not (step2_set_match.start() < step3_reserve_match.start()), (
        "mutation witness is broken: the mutated (ceiling-after-reserve) source still satisfies "
        "'Step 2 before Step 3' -- it should not"
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
    and a repeated call at the SAME n_ubatch must both return OK without
    calling reserve -- the first because there is nothing to re-plan, the
    second so a transaction re-run at an unchanged n_ubatch (e.g. the narrow
    flash-attn re-check path) is a no-op rather than repeating the
    ceiling-raise/reserve dance every time."""
    body_norm = _normalize_ws(_replan_ring_fn_body())
    ok_return = r"return\s+ggml_sycl_ring_replan_result::OK\s*;"
    assert re.search(
        r"if\s*\(\s*weight_slot_bytes\s*==\s*0\s*\)\s*\{\s*" + ok_return, body_norm
    ), "weight_slot_bytes == 0 (dense model) must return OK immediately, before any reserve attempt"
    assert re.search(
        r"if\s*\(\s*n_ubatch\s*==\s*ggml_sycl::unified_cache_get_planned_pp_moe_onednn_n_ubatch\(device\)\s*\)\s*"
        r"\{\s*" + ok_return,
        body_norm,
    ), (
        "n_ubatch already equal to the planned n_ubatch must return OK immediately (idempotent re-plan)"
    )


def test_replan_always_releases_the_ring_before_reserving():
    """llama.cpp-ibj0 spec round 1 F1/F2/F4 (supersedes the pre-round-1
    shrink-only release): the re-plan must release the CURRENT physical ring
    UNCONDITIONALLY -- both a grow and a shrink -- before reserving the new
    size, and that release must precede both the ceiling write and the
    reserve attempt.

    Why unconditional, not shrink-only (llama.cpp-nphx Task 3 spike comment
    c-wgxn, and the round-1 review that found the shrink-only version's own
    capacity math was wrong for grow): reserve_pp_moe_onednn_scratch()'s
    'already sufficient, reuse without reallocating' fast path would
    otherwise silently keep a larger, discarded ring (and the RUNTIME zone's
    charge for it) forever after a settle-smaller; and on a GROW,
    reserve_pp_moe_onednn_scratch()'s own allocate-new-then-retire-old path
    holds the OLD ring's bytes outstanding while attempting the NEW one, so
    a refusal's capacity figure that assumed those bytes were already free
    named an -ub that would also fail. Releasing first makes every re-plan
    attempt, either direction, measure the exact same thing: real capacity
    with nothing from this ring outstanding."""
    body_norm = _normalize_ws(_replan_ring_fn_body())

    assert "release_pp_moe_onednn_scratch_ring(" in body_norm, (
        "the re-plan must call release_pp_moe_onednn_scratch_ring()"
    )
    release_match = re.search(r"if\s*\(\s*!\s*cache->release_pp_moe_onednn_scratch_ring\(\)\s*\)", body_norm)
    assert release_match is not None, (
        "the release call must be UNCONDITIONAL -- gated on neither n_ubatch < old_n_ubatch nor any other "
        "direction check (see F1/F4)"
    )

    # llama.cpp-ibj0 spec round 5 F15: anchor on the NEW-size Step 2/Step 3
    # calls specifically (identified by the slot-byte variables they pass),
    # not on the first occurrence of either function name -- the
    # refuse_and_restore() closure (F13) defines its OWN internal
    # old-size set/reserve pair, in that same relative order, textually
    # between the release call and the real Step 2/3 calls, which would
    # satisfy a position-only "first set precedes first reserve" check
    # regardless of where the REAL Step 2/3 calls end up.
    step2_set_match = re.search(
        r"unified_cache_set_planned_pp_moe_onednn_scratch\(\s*device,\s*weight_slot_bytes,\s*"
        r"new_activation_slot_bytes",
        body_norm,
    )
    step3_reserve_match = re.search(
        r"reserve_pp_moe_onednn_scratch\(\s*weight_slot_bytes,\s*new_activation_slot_bytes", body_norm
    )
    assert step2_set_match is not None and step3_reserve_match is not None
    assert release_match.start() < step2_set_match.start() < step3_reserve_match.start(), (
        "the release call must precede BOTH the ceiling write and the reserve attempt for the NEW size -- "
        "releasing after either would either publish a ceiling the physical ring does not yet back, or race "
        "reserve_pp_moe_onednn_scratch()'s own fast path"
    )


def test_refusal_capacity_measures_the_path_that_actually_reserves():
    """llama.cpp-ibj0 spec round 1 F1/F2: the capacity figure in the refusal
    must come from the SAME accessor reserve_pp_moe_onednn_scratch() itself
    draws from on the CURRENT route -- zone_available(RUNTIME) when
    arena-backed, available() (budget_ minus used_) on the direct-device
    path -- branching on the identical arena_active() predicate
    reserve_pp_moe_onednn_scratch()'s own 'reserved from %s' log already uses
    to name the two routes. Must NOT add back any of the ring's own bytes:
    a grow's allocator holds the OLD ring outstanding while attempting the
    NEW one (transiently two full rings), so those bytes are never actually
    available to that attempt -- the pre-round-1 add-back named a
    largest-fitting -ub that would also fail."""
    body_norm = _normalize_ws(_replan_ring_fn_body())

    assert "cache->arena_active()" in body_norm, (
        "must branch on the SAME arena_active() predicate reserve_pp_moe_onednn_scratch() uses to pick its "
        "own capacity source"
    )
    assert "cache->zone_available(ggml_sycl::vram_zone_id::RUNTIME)" in body_norm, (
        "the arena-backed route must read zone_available(RUNTIME)"
    )
    assert "cache->available()" in body_norm, "the direct-device route must read available() (budget_ - used_)"

    assert not re.search(r"capacity_bytes\s*=[^;]*old_activation_slot_bytes", body_norm), (
        "the capacity figure must not add back the old ring's bytes -- see this test's docstring"
    )
    assert '"but the RUNTIME zone has' not in body_norm, (
        "the refusal must not hardcode \"RUNTIME\" in the format string -- name the real route the "
        "reservation actually decided against (the dynamic zone_name variable)"
    )
    assert "zone_name" in body_norm, "the refusal must print the dynamic zone_name variable, not a fixed string"

    reserve_indices = [m.start() for m in re.finditer(r"cache->reserve_pp_moe_onednn_scratch\(", body_norm)]
    assert len(reserve_indices) == 2, (
        "expected exactly two reserve_pp_moe_onednn_scratch() calls (the NEW-size attempt, and the OLD-size "
        f"restore attempt used on any failure) -- found {len(reserve_indices)}"
    )
    # llama.cpp-ibj0 spec round 5 F13 refactor: capacity_bytes is now
    # computed ONCE, right after the ring release and BEFORE either the
    # restore-reserve helper closure is even DEFINED or the NEW-size Step 3
    # reserve is attempted -- valid for both the F13 pre-check (which never
    # reaches either reserve call when it refuses) and the post-Step-3
    # failure path. The restore closure's own body (with the OLD-size
    # reserve call inside it) is DEFINED textually before Step 3 even
    # though it only runs later, so reserve_indices[0]/[1] no longer
    # reliably distinguish "attempted first" from "attempted second" --
    # identify each call by which slot-byte variables it requests instead.
    capacity_idx = body_norm.find("capacity_bytes =")
    assert capacity_idx != -1 and capacity_idx < min(reserve_indices), (
        "capacity_bytes must be measured BEFORE either reserve_pp_moe_onednn_scratch() call is attempted -- "
        "it must describe the state right after release, which is unaffected by either subsequent attempt "
        "(a failed reserve cleans up fully; the restore attempt runs only after capacity is already reported)"
    )


def test_replan_reserves_the_old_ring_on_a_failed_new_reserve():
    """llama.cpp-ibj0 spec round 1 F1: on a failed NEW-size reserve, the OLD
    ring must be RE-RESERVED, not merely have its ceiling relabeled -- the
    unconditional release earlier in this function already tore the old ring
    down physically, so restoring only the ceiling would claim a ring that
    no longer exists. A failure of the restore attempt itself must be
    reported, not silently swallowed."""
    body_norm = _normalize_ws(_replan_ring_fn_body())
    reserve_indices = [m.start() for m in re.finditer(r"cache->reserve_pp_moe_onednn_scratch\(", body_norm)]
    assert len(reserve_indices) == 2
    # Identify the restore-reserve call by the OLD-size variables it
    # requests, rather than by position -- see the ordering note in the
    # test above for why position alone is no longer a reliable anchor.
    restore_call = next(
        (idx for idx in reserve_indices if "old_activation_slot_bytes" in body_norm[idx : idx + 120]), None
    )
    assert restore_call is not None, (
        "expected one of the two reserve_pp_moe_onednn_scratch() calls to request old_activation_slot_bytes "
        "-- actually restoring the physical ring, not just relabeling the ceiling"
    )
    assert "old_output_slot_bytes" in body_norm[restore_call : restore_call + 120], (
        "the restore-reserve call must request old_output_slot_bytes too"
    )
    assert "restore FAILED" in body_norm, "a failed restore attempt must be logged at ERROR, not silently ignored"


def test_refusal_is_a_single_combined_line_not_two_calls():
    """llama.cpp-ibj0 spec round 1 F3: the refusal must be ONE GGML_LOG_ERROR
    call, with the largest-fitting clause appended to the SAME line (omitted
    when the figure is < 32) -- not a second, separate GGML_LOG_ERROR call
    just for the remediation, which was the pre-round-1 shape."""
    body_norm = _normalize_ws(_replan_ring_fn_body())
    assert 'GGML_LOG_ERROR("[SYCL-PLAN] the largest -ub that fits is about %u\\n"' not in body_norm, (
        "the largest-fitting remediation must not be its own separate GGML_LOG_ERROR call"
    )
    assert 'available%s\\n"' in body_norm, (
        "the main refusal's format string must end \"...available%s\\n\" -- a %s slot for the "
        "conditionally-omitted largest-fitting clause, on the SAME line as the rest of the message"
    )


def test_replan_guards_n_ubatch_zero():
    """llama.cpp-ibj0 spec round 1 F9: n_ubatch == 0 is not a valid runtime
    micro-batch (unified_cache_pp_moe_onednn_slots_for_ubatch() itself
    accepts it and computes a valid-looking zero-sized ring -- 0 rows is not
    an overflow -- so the SIZING function's leniency cannot be relied on as
    the guard). The re-plan must guard it explicitly with a WARN and never
    publish a zero ceiling."""
    body_norm = _normalize_ws(_replan_ring_fn_body())
    guard_match = re.search(r"if\s*\(\s*n_ubatch\s*==\s*0\s*\)\s*\{", body_norm)
    assert guard_match is not None, "the re-plan must explicitly guard n_ubatch == 0"

    idempotence_idx = body_norm.find(
        "if (n_ubatch == ggml_sycl::unified_cache_get_planned_pp_moe_onednn_n_ubatch(device))", guard_match.start()
    )
    assert idempotence_idx != -1 and idempotence_idx > guard_match.start(), (
        "could not bound the n_ubatch==0 guard's own if-block (looked for the idempotence check just after it)"
    )
    guard_block = body_norm[guard_match.start() : idempotence_idx]
    assert "GGML_LOG_WARN(" in guard_block, "the n_ubatch==0 guard must WARN, not silently return"
    assert "return ggml_sycl_ring_replan_result::OK" in guard_block, (
        "the n_ubatch==0 guard must skip the re-plan (return OK), not proceed"
    )
    assert "unified_cache_set_planned_pp_moe_onednn_scratch(" not in guard_block, (
        "the n_ubatch==0 guard must never publish a zero ceiling"
    )


def test_transaction_rolls_back_the_ring_on_a_later_failure():
    """llama.cpp-ibj0 spec round 1 F8 + round 2 F10: a successful ring
    re-plan can still be undone by a LATER, unrelated transaction failure --
    publication-ID exhaustion, MMID workspace materialization, or the
    lifecycle_replace_placement_plan CAS losing to a concurrent transaction
    (F10: the winning plan may describe a different n_ubatch, so the
    leftover ring is not provably harmless) -- all THREE later
    `return refuse(...);` sites (llama.cpp-tsfl round 1 F6: refuse() now
    returns ggml_sycl_txn_result::REFUSED rather than bare `false`) must
    roll the ring back to the pre-transaction n_ubatch (by re-invoking the
    same, direction-symmetric re-plan function), or a refused transaction
    leaves a changed ring behind even though nothing about the ring itself
    was ever refused."""
    body_norm = _normalize_ws(_runtime_context_body())
    assert "pre_replan_pp_moe_ring_n_ubatch" in body_norm, (
        "the pre-transaction ring n_ubatch must be captured BEFORE the re-plan call, so later failure paths "
        "can roll back to it"
    )
    rollback_calls = [
        m.start()
        for m in re.finditer(
            r"ggml_sycl_replan_pp_moe_onednn_ring\(ctx->device,\s*pre_replan_pp_moe_ring_n_ubatch\)", body_norm
        )
    ]
    assert len(rollback_calls) == 3, (
        "expected exactly three rollback calls (publication-ID exhaustion, MMID materialization failure, "
        f"the CAS failure) -- found {len(rollback_calls)}"
    )

    publication_idx = body_norm.find("publication ID exhausted")
    mmid_fail_idx = body_norm.find("MMID workspace materialization failed")
    cas_idx = body_norm.find("lifecycle_replace_placement_plan(current, immutable)")
    # llama.cpp-ibj0 spec round 3 F12: bound the CAS-path rollback ABOVE by
    # the success-path publish call, symmetric with how the first two
    # anchors are already bounded on both sides. Without this upper bound, a
    # mutant that moves the CAS-path rollback line out of its
    # `if (!lifecycle_replace_placement_plan(...))` block to one line AFTER
    # ggml_sycl_publish_prepared_plan_locked(...) -- so the CAS FAILURE path
    # no longer rolls back and the SUCCESS path wrongly undoes the ring
    # instead -- still passed `any(idx > cas_idx for idx in rollback_calls)`
    # (a rollback call sitting just after the publish satisfies "idx >
    # cas_idx" just as well as one sitting inside the CAS failure block).
    # ggml_sycl_publish_prepared_plan_locked(prepared_publication) occurs
    # exactly once inside this bounded body (a second, unrelated call site
    # exists elsewhere in the file, in a different function, outside this
    # slice), so this is unambiguous.
    publish_idx = body_norm.find("ggml_sycl_publish_prepared_plan_locked(prepared_publication)")
    assert publication_idx != -1 and mmid_fail_idx != -1 and cas_idx != -1 and publish_idx != -1
    assert publication_idx < mmid_fail_idx < cas_idx < publish_idx, (
        "could not establish the expected source order of the three failure sites and the success-path publish"
    )
    assert any(publication_idx < idx < mmid_fail_idx for idx in rollback_calls), (
        "the publication-ID-exhaustion failure path must roll back before its own return"
    )
    assert any(mmid_fail_idx < idx < cas_idx for idx in rollback_calls), (
        "the MMID-materialization-failure path must roll back before its own return, and before the CAS "
        "call site (not attributable to the third rollback)"
    )
    assert any(cas_idx < idx < publish_idx for idx in rollback_calls), (
        "the lifecycle_replace_placement_plan CAS failure path must roll back before its own return, and "
        "BEFORE the success-path publish call -- a rollback sitting after the publish would actually be "
        "undoing a SUCCESSFUL transaction's ring, not the CAS failure's (llama.cpp-ibj0 spec round 2 F10, "
        "bound tightened round 3 F12)"
    )


def test_cas_rollback_upper_bound_has_a_mutation_witness():
    """Mutation witness for llama.cpp-ibj0 spec round 3 F12: proves the
    upper-bound check above (the CAS-path rollback must precede the
    success-path publish call) would actually catch the reviewer's mutant --
    moving the CAS-path rollback line out of its
    `if (!lifecycle_replace_placement_plan(...))` block to one line AFTER
    ggml_sycl_publish_prepared_plan_locked(...), so the CAS FAILURE path no
    longer rolls back and the SUCCESS path wrongly undoes the ring instead.

    Before F12 this exact mutant PASSED the (then only lower-bounded)
    rollback-count check: `any(idx > cas_idx for idx in rollback_calls)` is
    still true for a rollback call sitting just after the publish, so the
    unbounded-above form could not tell a moved rollback from a correctly
    placed one."""
    raw = GGML_SYCL_CPP
    # Unique block: the CAS-failure comment's last line, its rollback call,
    # the return/close-brace, and the publish call immediately after --
    # unique because the identical rollback-call text appears twice more
    # elsewhere in this function (the other two failure paths), but only
    # THIS occurrence is immediately preceded by "as the two earlier failure
    # paths above." and immediately followed by the publish call.
    # llama.cpp-tsfl: the CAS-failure path's own `return;` is now
    # `return refuse("concurrent transaction won the CAS");` (the shared
    # body's local refuse() helper, which also fills the probe's `out` when
    # non-NULL) -- text updated to match, structure unchanged.
    old_block = (
        "        // as the two earlier failure paths above.\n"
        "        (void) ggml_sycl_replan_pp_moe_onednn_ring(ctx->device, pre_replan_pp_moe_ring_n_ubatch);\n"
        '        return refuse("concurrent transaction won the CAS");\n'
        "    }\n"
        "    ggml_sycl_publish_prepared_plan_locked(prepared_publication);\n"
    )
    assert old_block in raw, "mutation target block not found -- update this witness to match the real source"
    new_block = (
        "        // as the two earlier failure paths above.\n"
        '        return refuse("concurrent transaction won the CAS");\n'
        "    }\n"
        "    ggml_sycl_publish_prepared_plan_locked(prepared_publication);\n"
        "    (void) ggml_sycl_replan_pp_moe_onednn_ring(ctx->device, pre_replan_pp_moe_ring_n_ubatch);\n"
    )
    mutated_raw = raw.replace(old_block, new_block, 1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _RUNTIME_CONTEXT_START, _SET_RUNTIME_CONTEXT_WRAPPER_START)
    mutated_rollback_calls = [
        m.start()
        for m in re.finditer(
            r"ggml_sycl_replan_pp_moe_onednn_ring\(ctx->device,\s*pre_replan_pp_moe_ring_n_ubatch\)",
            mutated_body_norm,
        )
    ]
    mutated_cas_idx = mutated_body_norm.find("lifecycle_replace_placement_plan(current, immutable)")
    mutated_publish_idx = mutated_body_norm.find("ggml_sycl_publish_prepared_plan_locked(prepared_publication)")
    assert mutated_cas_idx != -1 and mutated_publish_idx != -1

    # The OLD (round-2) predicate the review found wrongly passing on this
    # mutant: still true, since the rollback call still sits somewhere after
    # cas_idx (just after the publish, not inside the CAS-failure block).
    assert any(idx > mutated_cas_idx for idx in mutated_rollback_calls), (
        "mutation witness is broken: the pre-F12 (unbounded-above) predicate should still wrongly pass on "
        "this mutant, or this is not reproducing the reviewer's failure mode"
    )
    # The F12 (fixed, upper-bounded) predicate: must now correctly FAIL,
    # proving the bound actually catches the mutant.
    assert not any(mutated_cas_idx < idx < mutated_publish_idx for idx in mutated_rollback_calls), (
        "mutation witness is broken: the F12 upper-bounded predicate should FAIL on this mutant (no rollback "
        "call between the CAS check and the publish), but it did not"
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

    assert not _has_static_before(cpp_norm, "unified_cache_release_pp_moe_onednn_scratch_ring"), (
        "the free-function wrapper must not be file-static"
    )

    body_norm = _normalize_ws(
        _bounded_body(CACHE_CPP_CODE, _RELEASE_RING_START, "bool unified_cache::claim_pp_moe_onednn_scratch_slot(")
    )
    assert re.search(r"if\s*\(\s*slot\.refcount\s*!=\s*0\s*\)\s*\{\s*return\s+false\s*;", body_norm), (
        "must refuse (return false) when a currently-held slot is still claimed (refcount != 0), before "
        "touching the ring"
    )


def test_replan_refuses_before_reserving_when_ring_exceeds_runtime_zone():
    """llama.cpp-ibj0 spec round 5 F13: on the arena route, the re-plan must
    refuse BEFORE ever attempting Step 3 (reserve) when the new ring's total
    would exceed zone_available(RUNTIME) -- a merge-gate B50 GPT-OSS sweep
    found a 674.5 MB ring "succeed" against a 512 MB RUNTIME zone by
    silently spilling into raw device memory outside the arena, consuming
    outside-arena headroom another path (the oneMath gemm scratch) then
    crashed for lack of (UR_RESULT_ERROR_OUT_OF_RESOURCES, not this
    function's own refusal). The gate is a source-level proxy for a GPU
    behavior no host-only check can exercise directly: it pins that the
    checking code exists and runs in the right place, not that the
    allocator actually refuses on real hardware (that is the lead's GPU
    acceptance)."""
    body_norm = _normalize_ws(_replan_ring_fn_body())

    guard_match = re.search(r"if\s*\(\s*arena\s*&&\s*needed_total\s*>\s*capacity_bytes\s*\)\s*\{", body_norm)
    assert guard_match is not None, (
        "the re-plan must have a guard refusing when the new ring's needed_total exceeds capacity_bytes on "
        "the arena route (`if (arena && needed_total > capacity_bytes)`)"
    )

    release_idx = body_norm.find("cache->release_pp_moe_onednn_scratch_ring()")
    # The ceiling write for the NEW size (Step 2) is the marker for
    # "the pre-check did NOT run before this point" -- it is immediately
    # followed by the reserve call for the NEW size (Step 3), so finding
    # its SECOND occurrence in the whole function (the first is inside the
    # refuse_and_restore() closure, restoring the OLD size) would be a
    # different, more fragile anchor; instead anchor on the reserve call
    # for the NEW activation/output bytes specifically.
    step3_reserve_idx = body_norm.find(
        "cache->reserve_pp_moe_onednn_scratch(weight_slot_bytes, new_activation_slot_bytes, new_output_slot_bytes"
    )
    assert release_idx != -1 and step3_reserve_idx != -1
    assert release_idx < guard_match.start() < step3_reserve_idx, (
        "the F13 guard must run AFTER the ring release and BEFORE the Step 3 reserve attempt for the new "
        "size -- checking after Step 3 has already run would waste the attempt this gate exists to skip"
    )


def test_replan_guard_has_a_mutation_witness():
    """Mutation witness for the F13 guard above: proves it would actually
    catch the guard being deleted, rather than only ever passing on the
    current, correct source."""
    raw = GGML_SYCL_CPP
    guard_block = (
        "    if (arena && needed_total > capacity_bytes) {\n"
        "        return refuse_and_restore();\n"
        "    }\n\n"
    )
    assert guard_block in raw, "mutation target block not found -- update this witness to match the real source"
    mutated_raw = raw.replace(guard_block, "", 1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _REPLAN_START, _RUNTIME_CONTEXT_START)
    assert not re.search(r"if\s*\(\s*arena\s*&&\s*needed_total\s*>\s*capacity_bytes\s*\)", mutated_body_norm), (
        "mutation witness is broken: deleting the guard left a reference to it behind"
    )


def test_reserve_pp_moe_onednn_scratch_forbids_vram_zone_spill():
    """llama.cpp-ibj0 spec round 5 F13: the PP MoE oneDNN scratch ring's own
    allocation requests (reserve_pp_moe_onednn_scratch()'s allocate_buffer
    lambda) must set forbid_vram_zone_spill, so that a request too big for
    the preferred zone FAILS the allocation instead of unified_alloc()
    silently falling through to a raw sycl::malloc_device outside the
    arena."""
    body_norm = _normalize_ws(_bounded_body(CACHE_CPP_CODE, _RESERVE_PP_MOE_START, _RELEASE_RING_START))
    assert "req.intent.constraints.prefer_vram_zone = vram_zone_id::RUNTIME;" in body_norm, (
        "expected the existing prefer_vram_zone = RUNTIME line -- update this test if that call site moved"
    )
    assert "req.intent.constraints.forbid_vram_zone_spill = true;" in body_norm, (
        "reserve_pp_moe_onednn_scratch()'s own allocate_buffer lambda must set "
        "req.intent.constraints.forbid_vram_zone_spill = true"
    )


def test_reserve_pp_moe_onednn_scratch_spill_flag_has_a_mutation_witness():
    """Mutation witness for the check above: proves it would actually catch
    the forbid_vram_zone_spill assignment being removed."""
    raw = CACHE_CPP
    line = "        req.intent.constraints.forbid_vram_zone_spill = true;\n"
    assert raw.count(line) == 1, f"expected exactly one occurrence, found {raw.count(line)}"
    mutated_raw = raw.replace(line, "", 1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _RESERVE_PP_MOE_START, _RELEASE_RING_START)
    assert "req.intent.constraints.forbid_vram_zone_spill = true;" not in mutated_body_norm, (
        "mutation witness is broken: deleting the assignment left a reference to it behind"
    )


def test_unified_alloc_enforces_forbid_vram_zone_spill():
    """llama.cpp-ibj0 spec round 5 F13: unified_alloc()'s own fallthrough --
    'if the preferred zone is full, fall through to a raw device malloc' --
    must check forbid_vram_zone_spill and fail the allocation instead, for
    ANY caller that sets it (not just the PP MoE oneDNN ring), so the
    constraint is a real, enforced contract rather than a flag callers set
    that nothing reads."""
    body_norm = _normalize_ws(_bounded_body(CACHE_CPP_CODE, _UNIFIED_ALLOC_START, _ACQUIRE_OFFLOAD_BUFFER_START))
    guard_match = re.search(
        r"if\s*\(\s*!\s*ptr\s*&&\s*req\.intent\.constraints\.forbid_vram_zone_spill\s*\)\s*\{\s*return\s+false\s*;",
        body_norm,
    )
    assert guard_match is not None, (
        "unified_alloc() must refuse (return false) when the preferred-zone allocation failed (!ptr) and "
        "the caller set forbid_vram_zone_spill"
    )

    zone_attempt_idx = body_norm.find("prefer_vram_zone != vram_zone_id::COUNT")
    raw_malloc_idx = body_norm.find('unified_cache_malloc_device_tracked(alloc_size, *req.queue, "unified_alloc:device")')
    assert zone_attempt_idx != -1 and raw_malloc_idx != -1
    assert zone_attempt_idx < guard_match.start() < raw_malloc_idx, (
        "the forbid_vram_zone_spill check must run AFTER the preferred-zone allocation attempt and BEFORE "
        "the raw device malloc fallthrough it exists to intercept"
    )

    # llama.cpp-ibj0 spec round 5 F14: the guard must be NESTED INSIDE the
    # `cache && cache->arena_active()` block, not merely textually between
    # the zone-attempt and the raw-malloc fallthrough -- `ptr` is
    # nullptr-initialized and nothing else assigns it whenever no zone
    # attempt actually ran (the arena disabled via GGML_SYCL_VRAM_ARENA=0,
    # or no cache yet), so a guard sitting immediately AFTER this whole
    # block closes (as the round-5a version did) would satisfy a pure
    # ordering check ("between zone-attempt and raw-malloc") while still
    # firing on "ptr still null for an unrelated reason", refusing every
    # direct-device-route allocation from a caller that sets the flag, not
    # just an actual zone-full case. A pure text-order check cannot tell
    # "inside" from "immediately after" apart -- both place the guard
    # between the same two anchors -- so this counts closing braces
    # instead: the guard's own if-block adds one, and being nested inside
    # BOTH `if (prefer_vram_zone != COUNT && vram_arena_enabled())` and
    # `if (cache && cache->arena_active())` requires two more before the
    # KV-role fallback that follows -- three total. The round-5a version
    # (guard outside both) had only one (its own).
    kv_fallback_idx = body_norm.find("!ptr && req.intent.role == alloc_role::KV && vram_arena_enabled()")
    assert kv_fallback_idx != -1, "could not find the KV-role fallback condition to bound the prefer_vram_zone block"
    assert guard_match.start() < kv_fallback_idx, (
        "the forbid_vram_zone_spill guard must run before the KV-role fallback that follows the "
        "prefer_vram_zone block"
    )
    closing_braces_after_guard = body_norm[guard_match.start() : kv_fallback_idx].count("}")
    assert closing_braces_after_guard == 3, (
        "the forbid_vram_zone_spill guard must be nested inside BOTH the outer prefer_vram_zone check and "
        "the `cache && cache->arena_active()` check (three closing braces between the guard and the "
        f"KV-role fallback: the guard's own, then arena_active()'s, then prefer_vram_zone's) -- found "
        f"{closing_braces_after_guard}"
    )


def test_unified_alloc_spill_guard_has_a_mutation_witness():
    """Mutation witness for the check above: proves it would actually catch
    unified_alloc()'s forbid_vram_zone_spill enforcement being deleted."""
    raw = CACHE_CPP
    guard_block = (
        "                if (!ptr && req.intent.constraints.forbid_vram_zone_spill) {\n"
        "                    return false;\n"
        "                }\n"
    )
    assert guard_block in raw, "mutation target block not found -- update this witness to match the real source"
    mutated_raw = raw.replace(guard_block, "", 1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _UNIFIED_ALLOC_START, _ACQUIRE_OFFLOAD_BUFFER_START)
    assert not re.search(r"forbid_vram_zone_spill\s*\)\s*\{\s*return\s+false\s*;", mutated_body_norm), (
        "mutation witness is broken: deleting the enforcement left a reference to it behind"
    )


def test_unified_alloc_spill_guard_nesting_has_a_mutation_witness():
    """Mutation witness for llama.cpp-ibj0 spec round 5 F14 itself: proves
    the brace-count nesting check in test_unified_alloc_enforces_forbid_vram_zone_spill()
    would actually catch the round-5a regression -- the guard moved OUT of
    the `cache && cache->arena_active()` block to sit immediately after it
    closes, changing nothing about its text ORDER relative to the
    zone-attempt and the KV-role fallback (both position-only checks would
    still pass) but changing what it actually guards: `ptr` reaching that
    point still null no longer means the zone attempt failed, it can also
    mean no zone attempt was ever made."""
    raw = CACHE_CPP
    nested_guard = (
        "                if (!ptr && req.intent.constraints.forbid_vram_zone_spill) {\n"
        "                    return false;\n"
        "                }\n"
        "            }\n"
        "        }\n"
    )
    assert nested_guard in raw, "mutation target block not found -- update this witness to match the real source"
    flattened_guard = (
        "            }\n"
        "        }\n"
        "        if (!ptr && req.intent.constraints.forbid_vram_zone_spill) {\n"
        "            return false;\n"
        "        }\n"
    )
    mutated_raw = raw.replace(nested_guard, flattened_guard, 1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _UNIFIED_ALLOC_START, _ACQUIRE_OFFLOAD_BUFFER_START)
    guard_match = re.search(
        r"if\s*\(\s*!\s*ptr\s*&&\s*req\.intent\.constraints\.forbid_vram_zone_spill\s*\)\s*\{\s*return\s+false\s*;",
        mutated_body_norm,
    )
    kv_fallback_idx = mutated_body_norm.find("!ptr && req.intent.role == alloc_role::KV && vram_arena_enabled()")
    assert guard_match is not None and kv_fallback_idx != -1, (
        "mutation witness is broken: could not find the guard or the KV-role fallback in the mutated source"
    )
    closing_braces_after_guard = mutated_body_norm[guard_match.start() : kv_fallback_idx].count("}")
    assert closing_braces_after_guard != 3, (
        "mutation witness is broken: the flattened (round-5a-style) guard should have a DIFFERENT "
        f"closing-brace count than the correctly-nested one (found 3, same as nested, expected != 3)"
    )


def test_alloc_constraints_declares_forbid_vram_zone_spill():
    """The new constraint field must actually be declared in alloc_constraints
    (unified-cache.hpp) -- without it, none of the call/check sites above
    would compile."""
    hpp_norm = _normalize_ws(CACHE_HPP_CODE)
    assert re.search(r"struct\s+alloc_constraints\s*\{[^}]*\bforbid_vram_zone_spill\b[^}]*\}", hpp_norm), (
        "alloc_constraints must declare forbid_vram_zone_spill"
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
