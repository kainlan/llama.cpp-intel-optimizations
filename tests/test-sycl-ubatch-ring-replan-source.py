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
    the current, correct source.

    Counts occurrences rather than asserting total absence: since spec round
    1 F8 and round 2 F10, ggml_sycl_replan_pp_moe_onednn_ring() is
    legitimately called FOUR times in this function (the original re-plan,
    plus three later rollback call sites) -- deleting the original call site
    must drop the count by exactly one, to three, not zero."""
    raw = GGML_SYCL_CPP
    call_block = (
        "    if (!ggml_sycl_replan_pp_moe_onednn_ring(ctx->device, next_kv_info.n_ubatch)) {\n"
        "        return;  // refusal already logged with the largest fitting -ub\n"
        "    }\n\n"
    )
    assert call_block in raw, "mutation target block not found -- update this witness to match the real source"
    mutated_raw = raw.replace(call_block, "", 1)
    assert mutated_raw != raw

    def _call_count(code: str) -> int:
        body_norm = _normalize_ws(
            _bounded_body(
                code,
                "void ggml_backend_sycl_set_runtime_context(",
                "ggml_backend_sycl_set_runtime_context_for_model(",
            )
        )
        return len(re.findall(r"ggml_sycl_replan_pp_moe_onednn_ring\(", body_norm))

    original_count = _call_count(strip_comments(raw))
    mutated_count = _call_count(strip_comments(mutated_raw))
    assert original_count == 4, (
        f"expected exactly four calls in the unmutated source (the re-plan itself plus three rollback call "
        f"sites) -- found {original_count}; update this witness to match the real source"
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

    set_first_idx = body_norm.find("unified_cache_set_planned_pp_moe_onednn_scratch(")
    reserve_idx = body_norm.find("reserve_pp_moe_onednn_scratch(")
    assert set_first_idx != -1 and reserve_idx != -1
    assert release_match.start() < set_first_idx < reserve_idx, (
        "the release call must precede BOTH the ceiling write and the reserve attempt -- releasing after "
        "either would either publish a ceiling the physical ring does not yet back, or race "
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
        "expected exactly two reserve_pp_moe_onednn_scratch() calls (the failed NEW-size attempt, and the "
        f"OLD-size restore attempt on failure) -- found {len(reserve_indices)}"
    )
    capacity_idx = body_norm.find("capacity_bytes =")
    assert capacity_idx != -1 and reserve_indices[0] < capacity_idx < reserve_indices[1], (
        "capacity_bytes must be measured AFTER the failed NEW-size reserve attempt but BEFORE the OLD-size "
        "restore reserve attempt -- restoring first would consume back some of the capacity being reported, "
        "making the printed figure describe a state the failed attempt never actually saw"
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
    restore_call_and_after = body_norm[reserve_indices[1] : reserve_indices[1] + 200]
    assert "old_activation_slot_bytes" in restore_call_and_after and "old_output_slot_bytes" in restore_call_and_after, (
        "the second reserve_pp_moe_onednn_scratch() call must request the OLD sizes -- actually restoring "
        "the physical ring, not just relabeling the ceiling"
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
    assert "return true" in guard_block, "the n_ubatch==0 guard must skip the re-plan (return true), not proceed"
    assert "unified_cache_set_planned_pp_moe_onednn_scratch(" not in guard_block, (
        "the n_ubatch==0 guard must never publish a zero ceiling"
    )


def test_transaction_rolls_back_the_ring_on_a_later_failure():
    """llama.cpp-ibj0 spec round 1 F8 + round 2 F10: a successful ring
    re-plan can still be undone by a LATER, unrelated transaction failure --
    publication-ID exhaustion, MMID workspace materialization, or the
    lifecycle_replace_placement_plan CAS losing to a concurrent transaction
    (F10: the winning plan may describe a different n_ubatch, so the
    leftover ring is not provably harmless) -- all THREE later `return;`
    sites must roll the ring back to the pre-transaction n_ubatch (by
    re-invoking the same, direction-symmetric re-plan function), or a
    refused transaction leaves a changed ring behind even though nothing
    about the ring itself was ever refused."""
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
    old_block = (
        "        // as the two earlier failure paths above.\n"
        "        (void) ggml_sycl_replan_pp_moe_onednn_ring(ctx->device, pre_replan_pp_moe_ring_n_ubatch);\n"
        "        return;\n"
        "    }\n"
        "    ggml_sycl_publish_prepared_plan_locked(prepared_publication);\n"
    )
    assert old_block in raw, "mutation target block not found -- update this witness to match the real source"
    new_block = (
        "        // as the two earlier failure paths above.\n"
        "        return;\n"
        "    }\n"
        "    ggml_sycl_publish_prepared_plan_locked(prepared_publication);\n"
        "    (void) ggml_sycl_replan_pp_moe_onednn_ring(ctx->device, pre_replan_pp_moe_ring_n_ubatch);\n"
    )
    mutated_raw = raw.replace(old_block, new_block, 1)
    assert mutated_raw != raw

    mutated_body_norm = _normalize_ws(
        _bounded_body(
            strip_comments(mutated_raw),
            "void ggml_backend_sycl_set_runtime_context(",
            "ggml_backend_sycl_set_runtime_context_for_model(",
        )
    )
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
