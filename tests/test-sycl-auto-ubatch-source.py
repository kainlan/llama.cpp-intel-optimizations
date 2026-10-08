"""Source contract for llama.cpp-xojq (nphx Task 4b, track B; depends on
Task 1 = llama.cpp-ibj0, Task 2 = llama.cpp-tsfl, the Task 3 spike =
llama.cpp-24ox comment c-wgxn, and Task 4a = llama.cpp-y8xv):

The SYCL auto micro-batch selection trial, `llama_context::
sycl_select_auto_ubatch()` (src/llama-context.cpp), and its one call site in
the `llama_context` constructor (replacing the unconditional `sched_reserve()`
call). Per c-wgxn and the task's own acceptance criteria:

- The constructor's call site gates the trial on FOUR conditions:
  `params.n_ubatch_auto`, at least one SYCL backend, `GGML_SYCL_AUTO_UBATCH`
  (`ggml_backend_sycl_auto_ubatch_enabled()`), and `cparams.causal_attn` (a
  non-causal model keeps its `n_ubatch == n_batch` semantics -- comment
  c-dcct). Any condition false falls through to today's single
  `sched_reserve()` call, unchanged.
- The trial tries the ascending ladder {512, 1024, 2048, 4096}, each
  candidate capped by both `n_batch` and `n_ctx`, and additionally, for a
  MoE model, by the GPU MoE routing ceiling
  (`ggml_backend_sycl_moe_gpu_ubatch_max()`, comment c-s747 / llama.cpp-ohkx).
- Per candidate: Task 2's non-publishing probe
  (`ggml_backend_sycl_probe_runtime_context_for_model`) is consulted first
  (a BUSY answer loses the candidate at once: the retry loop it once had is
  gone, design gate 22b); STALE_IDENTITY is
  branched distinctly ("not the published model", c-rkye item 2); a refused
  or demoting candidate stops the ladder. Only a probe-accepted candidate is
  PUBLISHED (`sycl_resync_runtime_context_flash_attn()`) and given a full
  `sched_reserve()` cycle -- publish strictly BEFORE reserve, every time.
  The host-pinned compute-buffer fallback counter
  (`ggml_backend_sycl_compute_buffer_host_fallbacks`) is consulted AFTER that
  reserve, never before.
- The settle step and the trial's own stop-reason vocabulary; exactly one
  `[SYCL-PLAN] auto n_ubatch=` WARN reports the outcome.

Host-only, pure text assertions -- no SYCL device required, matching
test-sycl-compute-buffer-fallback-source.py's pytest-collectible pattern
(llama_test_pytest hands this file to pytest.main(), so checks must live
inside a test_*() function or pytest's "no tests collected" (exit 5) is
scored as a failure by design, not a skip).

Checks run against COMMENT-STRIPPED text so a positive structural check
cannot be fooled by prose that quotes a call the code does not actually make.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LLAMA_CONTEXT_CPP = (ROOT / "src/llama-context.cpp").read_text()
GGML_SYCL_H = (ROOT / "ggml/include/ggml-sycl.h").read_text()
# llama.cpp-oyfl / CLAUDE.md: plain file I/O, not the codescout index --
# ggml-sycl.cpp is ~100k lines and that index (and search_text's live scan)
# is documented blind/oversized for this specific file, so a tool-assisted
# search here would silently miss real occurrences.
GGML_SYCL_CPP = (ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()
# llama.cpp-7n6n (wires nphx Task 5): the persisted auto n_ubatch tuning
# cache's own TU -- small enough not to need the ggml-sycl.cpp caveat above.
UBATCH_TUNING_CACHE_CPP = (ROOT / "ggml/src/ggml-sycl/ubatch-tuning-cache.cpp").read_text()
# llama.cpp-7n6n: the host-only unit test file itself, read here only to pin
# the two 64-bit boundary literals a live GPU defect was found against (see
# test_boundary_literals_are_pinned_in_the_unit_test below) -- not otherwise
# parsed by this file's checks.
TEST_TUNING_CACHE_IO_CPP = (ROOT / "tests/test-tuning-cache-io.cpp").read_text()
# llama.cpp-pyu4: the env-var catalog, whose GGML_SYCL_AUTO_UBATCH row is
# pinned below.
SYCL_ENV_VARS_MD = (ROOT / "docs/backend/sycl-env-vars.md").read_text()
# llama.cpp-pyu4: the pure helper behind the early exit before the ladder
# loop, the host test that executes it, and that test's registration.
LLAMA_AUTO_UBATCH_H = (ROOT / "src/llama-auto-ubatch.h").read_text()
TEST_AUTO_UBATCH_LADDER_CPP = (ROOT / "tests/test-auto-ubatch-ladder.cpp").read_text()
TESTS_CMAKELISTS = (ROOT / "tests/CMakeLists.txt").read_text()


# ---------------------------------------------------------------------------
# Small generic helpers, copied verbatim (not imported) from
# test-sycl-compute-buffer-fallback-source.py, which itself copied them from
# test-sycl-ubatch-ring-replan-source.py / test-sycl-nonfa-attn-scratch-guard-
# source.py -- the established convention in this file family, stated
# explicitly in those files' own docstrings. Each source-gate test file is
# registered and collected standalone by pytest (llama_test_pytest /
# CMakeLists.txt), with no shared conftest.py or importable helper module in
# tests/, so copying with attribution is the working, already-proven pattern.
# ---------------------------------------------------------------------------

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


LLAMA_CONTEXT_CPP_CODE = strip_comments(LLAMA_CONTEXT_CPP)
GGML_SYCL_H_CODE = strip_comments(GGML_SYCL_H)
GGML_SYCL_CPP_CODE = strip_comments(GGML_SYCL_CPP)
UBATCH_TUNING_CACHE_CPP_CODE = strip_comments(UBATCH_TUNING_CACHE_CPP)


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


def _body_of(raw: str, start_marker: str, end_marker: str) -> str:
    """Comment-strip `raw`, bound the result from `start_marker` to the next
    `end_marker`, and whitespace-normalize -- the pipeline every mutation
    witness needs to re-derive a checkable body from its freshly mutated raw
    string."""
    return _normalize_ws(_bounded_body(strip_comments(raw), start_marker, end_marker))


# Boundary literals named once, used by both the plain body-extraction
# helpers below and every mutation witness's _body_of() call.
_CALL_SITE_START = "bool sycl_auto_ubatch_trial = false;"
_CALL_SITE_END = "if (!cparams.flash_attn) {"
# llama.cpp-7n6n (quality round 1, Q4): the signature grew two parameters
# (type_k/type_v, forwarded from the constructor's own llama_context_params
# -- see test_the_hoisted_block_takes_type_k_and_type_v below) --
# updated here since every other check in this file depends on this exact
# string via _trial_body().
# llama.cpp-7gno: the trial is two functions now. The hoisted block (sycl_auto_ubatch_prepare: the SYCL backends and
# procs, the cap, the cache key and lookup, the rung set) sits right before sycl_select_auto_ubatch (the per-candidate
# validator, the ladder, the settle and the outcome lines), so one slice from the first to the end marker is the trial.
_TRIAL_START = "void llama_context::sycl_auto_ubatch_prepare(ggml_type type_k, ggml_type type_v) {"
# The trial is followed by upstream's llama_graph_n_input_tensors() helper,
# not by sched_reserve() itself; ending at sched_reserve() would pull that
# helper's LLAMA_LOG_WARN into the trial body.
_TRIAL_END = "static int llama_graph_n_input_tensors(ggml_cgraph * gf, bool log) {"
# llama.cpp-7n6n: the shared per-candidate validator --
# probe with busy backoff, publish in a try/catch, reserve, host-fallback
# check -- extracted into one lambda used by BOTH the cache-hit revalidation
# and the ladder loop (previously two independently-maintained copies of the
# same sequence). Bounds a smaller region than the old loop-anchored checks
# used to; several per-candidate tests below are anchored here instead of to
# "for (uint32_t c : rung_ladder)" now that the sequence itself lives here, not
# in the loop body.
_TRY_CANDIDATE_START = "auto try_candidate = [&](uint32_t c) -> const char * {"
_TRY_CANDIDATE_END = "auto cache_enabled_fn ="
# The raw text of try_candidate()'s own publish and its catch, shared by the
# publish-order check and the catch's mutation witnesses below.
_CANDIDATE_PUBLISH_BLOCK = (
    "        try {\n"
    "            sycl_resync_runtime_context_flash_attn();\n"
    "        } catch (const std::exception &) {\n"
    "            sched_matches_last_good = false;\n"
    "            publish_dirty           = true;\n"
    '            return "transaction refused";\n'
    "        }\n"
)


def _call_site_body() -> str:
    return _bounded_body(LLAMA_CONTEXT_CPP_CODE, _CALL_SITE_START, _CALL_SITE_END)


def _trial_body() -> str:
    return _bounded_body(LLAMA_CONTEXT_CPP_CODE, _TRIAL_START, _TRIAL_END)


def _try_candidate_body() -> str:
    return _bounded_body(LLAMA_CONTEXT_CPP_CODE, _TRY_CANDIDATE_START, _TRY_CANDIDATE_END)


# ---------------------------------------------------------------------------
# The constructor's call-site gate
# ---------------------------------------------------------------------------


def test_call_site_gates_on_all_four_conditions():
    """The constructor must gate the trial on all four documented
    conditions -- n_ubatch_auto, a SYCL backend, GGML_SYCL_AUTO_UBATCH
    (auto_ubatch_enabled), and causal_attn -- through one
    llama_auto_ubatch_trial_runs() call fed the four evaluated conditions
    (tests/test-sycl-planned-ladder-source.py pins its argument spelling), so
    a single condition being false cannot leave the others live."""
    body_norm = _normalize_ws(_call_site_body())
    assert re.search(
        r"sycl_auto_ubatch_trial\s*=\s*llama_auto_ubatch_trial_runs\(\s*params\.n_ubatch_auto\s*,\s*"
        r"cparams\.causal_attn\s*,\s*sycl_backend_present\s*,\s*sycl_auto_ubatch_enabled\s*\)\s*;",
        body_norm,
    ), (
        "the call site must decide the trial with llama_auto_ubatch_trial_runs(params.n_ubatch_auto, "
        "cparams.causal_attn, sycl_backend_present, sycl_auto_ubatch_enabled)"
    )
    assert re.search(
        r"const\s+bool\s+sycl_backend_present\s*=\s*llama_context_has_sycl_backend\(\s*backends\s*\)\s*;", body_norm
    ), "sycl_backend_present must be llama_context_has_sycl_backend(backends), evaluated"

    # auto_ubatch_enabled() must be consulted INSIDE the cheap-condition gate
    # (either the direct GGML_USE_SYCL symbol or the GGML_BACKEND_DL
    # proc-address lookup), not unconditionally before or after it.
    gate_match = re.search(
        r"if\s*\(\s*params\.n_ubatch_auto\s*&&\s*cparams\.causal_attn\s*&&\s*sycl_backend_present\s*\)\s*\{", body_norm
    )
    assert gate_match is not None, (
        "the accessor must sit behind `if (params.n_ubatch_auto && cparams.causal_attn && sycl_backend_present)`"
    )
    assert "ggml_backend_sycl_auto_ubatch_enabled" in body_norm, (
        "the call site must consult ggml_backend_sycl_auto_ubatch_enabled() (directly or via its "
        "proc-address lookup)"
    )
    enabled_idx = body_norm.find("ggml_backend_sycl_auto_ubatch_enabled", gate_match.end())
    assert enabled_idx != -1, (
        "ggml_backend_sycl_auto_ubatch_enabled must be consulted AFTER the cheap-condition gate opens, "
        "not before it (it should not run when the other conditions already ruled the trial out)"
    )

    # And the trial itself must only run when sycl_auto_ubatch_trial ends up
    # true; otherwise today's single sched_reserve() call is unchanged.
    # quality round 1, Q4 / llama.cpp-7gno: the KV types reach the cache key through the hoisted block's call (see
    # test_the_hoisted_block_takes_type_k_and_type_v); the ladder half takes no arguments.
    assert re.search(
        r"if\s*\(\s*sycl_auto_ubatch_trial\s*\)\s*\{\s*sycl_select_auto_ubatch\s*\(\s*\)\s*;",
        body_norm,
    ), (
        "the call site must call sycl_select_auto_ubatch() only when "
        "sycl_auto_ubatch_trial is true"
    )
    assert re.search(r"\}\s*else\s*\{\s*sched_reserve\s*\(\s*\)\s*;\s*\}", body_norm), (
        "the call site must fall through to today's unconditional sched_reserve() otherwise"
    )


def test_trial_runs_helper_is_the_four_way_conjunction():
    """The helper behind the call-site decision must be exactly the four-way
    conjunction; the mutants drop one condition each."""
    code = strip_comments(LLAMA_AUTO_UBATCH_H)
    conj = "return n_ubatch_auto && causal_attn && has_sycl_backend && auto_ubatch_enabled;"
    assert conj in code, "llama_auto_ubatch_trial_runs must return the four-way conjunction"
    for dropped in ("n_ubatch_auto && ", "causal_attn && ", "has_sycl_backend && "):
        mutated = code.replace(conj, conj.replace(dropped, "", 1), 1)
        assert conj not in mutated


# ---------------------------------------------------------------------------
# The trial's ladder and caps
# ---------------------------------------------------------------------------


_LADDER_LITERAL_RE = (
    r"static\s+const\s+uint32_t\s+llama_auto_ubatch_ladder\s*\[\s*\]\s*=\s*"
    r"\{\s*512\s*,\s*1024\s*,\s*2048\s*,\s*4096\s*\}\s*;"
)


def test_ladder_literal_is_512_1024_2048_4096():
    """The candidate ladder must be exactly {512, 1024, 2048, 4096},
    ascending, per the task spec and the Task 3 spike. It lives in
    src/llama-auto-ubatch.h so the trial and tests/test-auto-ubatch-
    ladder.cpp read the same array; the trial binds it by reference."""
    assert re.search(_LADDER_LITERAL_RE, _normalize_ws(strip_comments(LLAMA_AUTO_UBATCH_H))), (
        "src/llama-auto-ubatch.h's llama_auto_ubatch_ladder must be exactly { 512, 1024, 2048, 4096 }"
    )
    assert re.search(r"const\s+auto\s*&\s*ladder\s*=\s*llama_auto_ubatch_ladder\s*;", _trial_body()), (
        "the trial must bind `ladder` to llama_auto_ubatch_ladder, not keep a literal of its own"
    )
    assert not re.search(r"uint32_t\s+ladder\s*\[", _trial_body()), "the trial must not declare its own ladder array"


def test_ladder_literal_has_a_mutation_witness():
    """Mutation witness for the ladder check above: proves it would
    actually catch a rung being changed (4096 -> 8192)."""
    raw = LLAMA_AUTO_UBATCH_H
    ladder_line = "static const uint32_t llama_auto_ubatch_ladder[] = { 512, 1024, 2048, 4096 };\n"
    assert raw.count(ladder_line) == 1, f"mutation target not unique -- found {raw.count(ladder_line)}"
    mutated_raw = raw.replace(
        ladder_line, "static const uint32_t llama_auto_ubatch_ladder[] = { 512, 1024, 2048, 8192 };\n", 1
    )
    assert mutated_raw != raw
    assert not re.search(_LADDER_LITERAL_RE, _normalize_ws(strip_comments(mutated_raw))), (
        "mutation witness is broken: changing the last rung should make the literal check fail"
    )


def test_candidate_cap_uses_n_batch_and_n_ctx():
    """The ladder cap must be min(n_batch, n_ctx) -- both, not either
    alone -- computed by llama_auto_ubatch_cap() (written once, pinned by
    tests/test-sycl-planned-ladder-source.py) after `ladder` is bound and
    before the loop, and the helper itself must take that minimum."""
    body_norm = _normalize_ws(_trial_body())
    assert re.search(
        r"const\s+uint32_t\s+cap\s*=\s*llama_auto_ubatch_cap\(\s*cparams\.n_batch\s*,\s*cparams\.n_ctx\s*,", body_norm
    ), "cap must be computed by llama_auto_ubatch_cap(cparams.n_batch, cparams.n_ctx, ...)"

    cap_idx = body_norm.find("const uint32_t cap = llama_auto_ubatch_cap(")
    ladder_idx = body_norm.find("const auto & ladder = llama_auto_ubatch_ladder;")
    loop_idx = body_norm.find("for (uint32_t c : rung_ladder)")
    assert cap_idx != -1 and ladder_idx != -1 and loop_idx != -1
    assert ladder_idx < cap_idx < loop_idx, "cap must be computed after `ladder` is bound to llama_auto_ubatch_ladder and before the loop"

    header = _normalize_ws(strip_comments(LLAMA_AUTO_UBATCH_H))
    assert "uint32_t cap = n_batch < n_ctx ? n_batch : n_ctx;" in header, (
        "llama_auto_ubatch_cap must start from min(n_batch, n_ctx)"
    )
    for mutant in ("uint32_t cap = n_batch;", "uint32_t cap = n_ctx;", "uint32_t cap = n_batch > n_ctx ? n_batch : n_ctx;"):
        assert mutant != "uint32_t cap = n_batch < n_ctx ? n_batch : n_ctx;"
        assert mutant not in header


# ---------------------------------------------------------------------------
# Early exits before the loop, quality round 1 -- both take the pre-trial
# path (a bare sched_reserve()) with NO WARN, since the trial never got a
# chance to try a single candidate.
# ---------------------------------------------------------------------------


def _empty_prep_exit_precedes_the_loop() -> None:
    """llama.cpp-7gno: every single-reserve exit of the hoisted block leaves the prep empty, and the trial's one answer to
    an empty prep is the pre-trial path: sched_reserve() then return, before the ladder loop and with no WARN."""
    body_norm = _normalize_ws(_trial_body())
    exit_idx = body_norm.find("if (!auto_ubatch_prep) { sched_reserve(); return; }")
    loop_idx = body_norm.find("for (uint32_t c : rung_ladder)")
    warn_idx = body_norm.find("[SYCL-PLAN] auto n_ubatch=")
    assert exit_idx != -1, "an empty prep must take the single reserve: `if (!auto_ubatch_prep) { sched_reserve(); return; }`"
    assert exit_idx < loop_idx and exit_idx < warn_idx, "the empty-prep exit must precede the ladder loop and the WARN"



def test_cap_below_first_rung_exits_before_the_loop_with_no_warn():
    """Q2a: when the fully-narrowed cap is below the ladder's first rung
    (any -c below 512, or llama-bench's pp128/tg128 rows), the function
    must take the pre-trial path -- sched_reserve() then return -- BEFORE
    ever entering the ladder loop, and must NOT log the [SYCL-PLAN] WARN
    (an empty `tried` list against a ladder that never ran)."""
    body_norm = _normalize_ws(_trial_body())
    cap_check_idx = body_norm.find("if (cap < ladder[0]) {")
    loop_idx = body_norm.find("for (uint32_t c : rung_ladder)")
    warn_idx = body_norm.find("[SYCL-PLAN] auto n_ubatch=")
    assert cap_check_idx != -1, "the cap < ladder[0] early exit must exist"
    assert loop_idx != -1 and warn_idx != -1
    assert cap_check_idx < loop_idx, "the cap < ladder[0] check must precede the ladder loop"
    assert cap_check_idx < warn_idx, "the WARN literal must not appear before the cap < ladder[0] check"

    early_exit_block = body_norm[cap_check_idx : body_norm.find("}", cap_check_idx) + 1]
    assert re.search(r"\{\s*return\s*;\s*\}", early_exit_block), (
        "the early exit must return with the prep empty, with no publish and no WARN in between"
    )
    _empty_prep_exit_precedes_the_loop()


def test_cap_below_first_rung_early_exit_has_a_mutation_witness():
    """Mutation witness for the check above: proves it would actually
    catch the early exit being deleted (falling through into the loop
    with an empty ladder that would silently exhaust with an empty
    `tried` list instead of exiting cleanly beforehand)."""
    raw = LLAMA_CONTEXT_CPP
    early_exit_block = (
        "    if (cap < ladder[0]) {\n"
        "        return;\n"
        "    }\n"
    )
    assert early_exit_block in raw, "mutation target not found -- update this witness to match the real source"
    mutated_raw = raw.replace(early_exit_block, "", 1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _TRIAL_START, _TRIAL_END)
    assert "if (cap < ladder[0]) {" not in mutated_body_norm, (
        "mutation witness is broken: deleting the block should remove the early-exit check"
    )


def test_zero_sycl_token_exits_before_the_loop_with_no_warn():
    """Q7: the same zero-token guard sycl_resync_runtime_context_flash_
    attn()/sycl_recheck_runtime_context_flash_attn() apply per backend
    (owner.model_id == 0 || owner.load_txn_id == 0) must be applied here
    too, BEFORE the token is used to enter the loop -- and, like the cap
    check above, take the pre-trial path with no WARN rather than treating
    a model with no SYCL token as "not the published model"."""
    body_norm = _normalize_ws(_trial_body())
    guard_idx = body_norm.find("if (owner.model_id == 0 || owner.load_txn_id == 0) {")
    token_idx = body_norm.find("const ggml_sycl_model_token token")  # the select half builds it after the hoisted guard
    loop_idx = body_norm.find("for (uint32_t c : rung_ladder)")
    warn_idx = body_norm.find("[SYCL-PLAN] auto n_ubatch=")
    assert guard_idx != -1, "the zero-token guard must exist"
    assert token_idx != -1 and loop_idx != -1 and warn_idx != -1
    assert guard_idx < token_idx < loop_idx, (
        "the zero-token guard must precede both the token construction and the ladder loop"
    )
    assert guard_idx < warn_idx, "the WARN literal must not appear before the zero-token guard"

    guard_block = body_norm[guard_idx : body_norm.find("}", guard_idx) + 1]
    assert re.search(r"\{\s*return\s*;\s*\}", guard_block), (
        "the zero-token guard must return with the prep empty, with no publish and no WARN in between"
    )
    _empty_prep_exit_precedes_the_loop()


def test_zero_sycl_token_guard_has_a_mutation_witness():
    """Mutation witness for the check above: proves it would actually
    catch the zero-token guard being deleted."""
    raw = LLAMA_CONTEXT_CPP
    guard_block = (
        "    if (owner.model_id == 0 || owner.load_txn_id == 0) {\n"
        "        return;\n"
        "    }\n"
    )
    assert guard_block in raw, "mutation target not found -- update this witness to match the real source"
    mutated_raw = raw.replace(guard_block, "", 1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _TRIAL_START, _TRIAL_END)
    assert "if (owner.model_id == 0 || owner.load_txn_id == 0) {" not in mutated_body_norm, (
        "mutation witness is broken: deleting the block should remove the zero-token guard"
    )


_MOE_NARROW = "if (n_expert > 0 && moe_cap_available && moe_cap <= cap) {"


def _cap_helper_body() -> str:
    header = _normalize_ws(strip_comments(LLAMA_AUTO_UBATCH_H))
    start = header.find("inline uint32_t llama_auto_ubatch_cap(")
    assert start != -1, "llama_auto_ubatch_cap not found in src/llama-auto-ubatch.h"
    return header[start : header.find("return cap;", start)]


def test_moe_model_cap_binds_whenever_moe_cap_does_not_exceed_the_batch_ctx_cap():
    """For a MoE model (hparams.n_expert > 0), the cap must be additionally
    narrowed to ggml_backend_sycl_moe_gpu_ubatch_max() -- comment c-s747 /
    llama.cpp-ohkx -- and the "MoE GPU routing ceiling" stop reason reported
    whenever that ceiling BINDS, i.e. moe_cap <= cap, not only when it is
    STRICTLY smaller (a MoE context whose batch/ctx cap already equals
    moe_cap, e.g. cap == 512, is bound by the ceiling exactly as much as one
    where moe_cap is smaller, so "ladder exhausted" would misreport why the
    ladder stopped). The 512 constant itself must NOT be
    hardcoded in llama-context.cpp; it must come from the accessor.

    The narrowing now lives in llama_auto_ubatch_cap(), gated on n_expert > 0
    and on moe_cap_available -- the DL-without-SYCL branch reports the
    ceiling unavailable when the accessor is absent, which would otherwise
    report the ceiling reason for every such MoE model although no ceiling
    was ever consulted. The trial passes the accessor's value, and reads the
    accessor only for a MoE model."""
    helper = _cap_helper_body()
    assert _MOE_NARROW in helper, (
        "llama_auto_ubatch_cap must narrow `cap` only when n_expert > 0 && moe_cap_available && "
        "moe_cap <= cap -- not on moe_cap <= cap alone"
    )
    assert "*moe_bound = true;" in helper and "*moe_bound = false;" in helper

    body_norm = _normalize_ws(_trial_body())
    assert "ggml_backend_sycl_moe_gpu_ubatch_max" in body_norm, (
        "the MoE cap must be read from ggml_backend_sycl_moe_gpu_ubatch_max(), not a local constant"
    )
    assert re.search(r"const\s+bool\s+moe_cap_available\s*=\s*true\s*;", body_norm), (
        "the direct GGML_USE_SYCL branch must declare moe_cap_available = true (the accessor is always real there)"
    )
    assert re.search(r"const\s+bool\s+moe_cap_available\s*=\s*moe_cap_fn\s*!=\s*nullptr\s*;", body_norm), (
        "the GGML_BACKEND_DL-without-SYCL branch must declare moe_cap_available = (moe_cap_fn != nullptr)"
    )
    assert re.search(
        r"moe_cap\s*=\s*model\.hparams\.n_expert\s*>\s*0\s*\?\s*ggml_backend_sycl_moe_gpu_ubatch_max\(\s*\)\s*:\s*0\s*;", body_norm
    ), "the direct branch must read the accessor only for a MoE model"
    assert re.search(
        r"moe_cap\s*=\s*\(\s*model\.hparams\.n_expert\s*>\s*0\s*&&\s*moe_cap_fn\s*\)\s*\?\s*moe_cap_fn\(\s*\)\s*:\s*0\s*;", body_norm
    ), "the DL branch must call the looked-up accessor only for a MoE model, and only when it exists"
    assert re.search(r'stop\s*=\s*moe_bound\s*\?\s*"MoE GPU routing ceiling"\s*:\s*"ladder exhausted"\s*;', body_norm), (
        "the stop reason must be the MoE ceiling exactly when the helper reports it bound"
    )
    assert not re.search(r"\bcap\s*=\s*512\b", body_norm), (
        "the MoE ceiling must not be hardcoded as a bare 512 in llama-context.cpp"
    )


def test_moe_model_cap_binds_at_equal_has_a_mutation_witness():
    """Mutation witness for the check above: proves it would actually
    catch the condition reverting to strict `<` (which silently drops the
    "MoE GPU routing ceiling" reason whenever moe_cap == cap, e.g. a MoE
    model whose batch/ctx cap is already exactly 512)."""
    header = strip_comments(LLAMA_AUTO_UBATCH_H)
    old = "moe_cap_available && moe_cap <= cap"
    assert _normalize_ws(header).count(old) == 1, "mutation target not unique"
    mutated = _normalize_ws(header).replace(old, "moe_cap_available && moe_cap < cap", 1)
    assert _MOE_NARROW not in mutated, (
        "mutation witness is broken: reverting to strict < should make the <= check fail"
    )


def test_moe_model_cap_accessor_gate_has_a_mutation_witness():
    """Mutation witness for the moe_cap_available gate itself: proves it
    would actually catch the gate being dropped (which would report "MoE
    GPU routing ceiling" for every MoE model on a GGML_BACKEND_DL build
    whose SYCL DSO lacks the accessor, even though moe_cap was never
    narrowed by anything)."""
    header = strip_comments(LLAMA_AUTO_UBATCH_H)
    old = "n_expert > 0 && moe_cap_available && moe_cap <= cap"
    assert _normalize_ws(header).count(old) == 1, "mutation target not unique"
    mutated = _normalize_ws(header).replace(old, "n_expert > 0 && moe_cap <= cap", 1)
    assert _MOE_NARROW not in mutated, (
        "mutation witness is broken: dropping the moe_cap_available gate should make the gated-condition check fail"
    )
    mutated = _normalize_ws(header).replace(old, "moe_cap_available && moe_cap <= cap", 1)
    assert _MOE_NARROW not in mutated, "dropping the n_expert gate must also fail the check"


def test_header_declares_the_moe_gpu_ubatch_max_accessor():
    """ggml_backend_sycl_moe_gpu_ubatch_max() must be declared in
    ggml-sycl.h and defined (not file-static) in ggml-sycl.cpp, returning
    the real MOE_GPU_UBATCH_MAX constant, not a re-declared literal."""
    assert re.search(
        r"GGML_BACKEND_API\s+uint32_t\s+ggml_backend_sycl_moe_gpu_ubatch_max\s*\(\s*void\s*\)\s*;", GGML_SYCL_H_CODE
    ), "ggml_backend_sycl_moe_gpu_ubatch_max(void) must be declared in ggml-sycl.h"

    assert re.search(
        r"uint32_t\s+ggml_backend_sycl_moe_gpu_ubatch_max\s*\(\s*\)\s*\{\s*return\s+ggml_sycl::MOE_GPU_UBATCH_MAX\s*;"
        r"\s*\}",
        GGML_SYCL_CPP_CODE,
    ), "ggml_backend_sycl_moe_gpu_ubatch_max() must be defined in ggml-sycl.cpp, returning ggml_sycl::MOE_GPU_UBATCH_MAX"
    assert not re.search(r"static\s+uint32_t\s+ggml_backend_sycl_moe_gpu_ubatch_max\(", GGML_SYCL_CPP_CODE), (
        "the accessor must not be file-static -- llama-context.cpp (a different translation unit) calls it"
    )


def test_proc_address_registers_moe_gpu_ubatch_max_and_auto_ubatch_enabled():
    """llama.cpp-1mwi: a GGML_BACKEND_DL build's llama-context lookup needs
    both ggml_backend_sycl_auto_ubatch_enabled (Task 4a defined it but
    never registered it here) and this task's own new
    ggml_backend_sycl_moe_gpu_ubatch_max registered in the strcmp
    proc-address chain, or the lookup silently returns nullptr."""
    assert re.search(
        r'strcmp\(\s*name\s*,\s*"ggml_backend_sycl_auto_ubatch_enabled"\s*\)\s*==\s*0\s*\)\s*\{\s*'
        r"return\s*\(\s*void\s*\*\s*\)\s*ggml_backend_sycl_auto_ubatch_enabled\s*;",
        GGML_SYCL_CPP_CODE,
    ), "ggml_backend_sycl_auto_ubatch_enabled must be registered in the proc-address strcmp chain"
    assert re.search(
        r'strcmp\(\s*name\s*,\s*"ggml_backend_sycl_moe_gpu_ubatch_max"\s*\)\s*==\s*0\s*\)\s*\{\s*'
        r"return\s*\(\s*void\s*\*\s*\)\s*ggml_backend_sycl_moe_gpu_ubatch_max\s*;",
        GGML_SYCL_CPP_CODE,
    ), "ggml_backend_sycl_moe_gpu_ubatch_max must be registered in the proc-address strcmp chain"


# ---------------------------------------------------------------------------
# The per-candidate probe: BUSY backoff, STALE_IDENTITY, would_demote_kv
# ---------------------------------------------------------------------------


def test_probe_is_called_with_the_candidate_shape():
    """Each candidate must be checked with the non-publishing probe before
    anything is published."""
    body_norm = _normalize_ws(_trial_body())
    # llama.cpp-3aos: tolerate additional cparams.* arguments inserted
    # between n_seq_max and flash_attn (e.g. cparams.kv_unified) -- the
    # intent is "flash_attn is the real cparams field, passed to the probe",
    # not "these two parameters are adjacent".
    assert re.search(
        r"probe_fn\(\s*sb\.backend\s*,\s*token\s*,\s*cparams\.n_ctx\s*,\s*c\s*,\s*cparams\.n_seq_max\s*,"
        r"(?:\s*cparams\.\w+\s*,)*\s*cparams\.flash_attn\s*,\s*&probe\s*\)",
        body_norm,
    ), "the probe must be called with (backend, token, n_ctx, the CANDIDATE c, n_seq_max, flash_attn, &probe)"


def test_probe_busy_is_not_retried():
    """A GGML_SYCL_LIFECYCLE_BUSY probe result is the module-admission
    refusal, not lock contention: the candidate loses with "transaction
    busy" at once. llama-context.cpp has no retry loop over a lifecycle
    result, no sleep_for, and no llama_context_sycl_max_busy_waits (design
    gate 22b; tests/test-sycl-publish-status-source.py pins the same for the
    whole file and the publish)."""
    assert "llama_context_sycl_max_busy_waits" not in LLAMA_CONTEXT_CPP_CODE, (
        "the shared BUSY-retry bound is gone along with both retry loops"
    )
    assert "sleep_for" not in LLAMA_CONTEXT_CPP_CODE
    body_norm = _normalize_ws(_try_candidate_body())
    assert not re.search(r"for\s*\(\s*int\s+wait\b", body_norm), "the probe is not retried"
    assert re.search(
        r'if\s*\(\s*rc\s*==\s*GGML_SYCL_LIFECYCLE_BUSY\s*\)\s*\{\s*return\s*"transaction busy"\s*;\s*\}', body_norm
    ), 'a BUSY probe must return "transaction busy" directly'


def test_stale_identity_branches_to_not_the_published_model():
    """GGML_SYCL_LIFECYCLE_STALE_IDENTITY must be branched distinctly to
    the "not the published model" stop reason (c-rkye item 2), not folded
    into the generic "transaction refused" branch.

    llama.cpp-7n6n: try_candidate() now RETURNS the
    reason instead of assigning `stop` directly (the two pre-refactor call
    sites -- the cache-hit revalidation and the ladder loop -- used to each
    assign this into the SAME `stop` variable inline, which is exactly the
    bug this caused: a failed cache revalidation could leave a stale reason
    in `stop` for a ladder that went on to finish cleanly). Only the ladder
    loop's own call site writes the returned reason into `stop`; this test
    is scoped to try_candidate()'s body, where the string literal itself
    lives, not to the assignment (which is now generic and reason-agnostic:
    `stop = reason;`)."""
    body_norm = _normalize_ws(_try_candidate_body())
    assert re.search(
        r'rc\s*==\s*GGML_SYCL_LIFECYCLE_STALE_IDENTITY\s*\)\s*\{\s*return\s*"not the published model"\s*;\s*\}',
        body_norm,
    ), 'GGML_SYCL_LIFECYCLE_STALE_IDENTITY must return "not the published model"'


def test_would_demote_kv_is_consulted_and_stops_the_ladder():
    """probe.would_demote_kv must be consulted and, when true, stop the
    ladder with the "KV would be demoted" reason -- never published.
    Scoped to try_candidate() -- see the note on the sibling check above
    for why this is now a `return`, not a `stop = ...` assignment."""
    body_norm = _normalize_ws(_try_candidate_body())
    assert re.search(
        r'probe\.would_demote_kv\s*\)\s*\{\s*return\s*"KV would be demoted"\s*;\s*\}', body_norm
    ), 'probe.would_demote_kv must be consulted and return "KV would be demoted"'


def test_probe_reasons_never_assign_stop_directly():
    """None of the four probe-outcome branches inside try_candidate() may
    assign `stop` directly (the pre-refactor shape, and the bug that shape
    caused) -- each must RETURN its reason so the two call sites (cache-hit
    revalidation, ladder loop) can each decide for themselves where it
    belongs. A direct `stop = "...";` inside try_candidate() would silently
    reintroduce that bug: a losing cache revalidation would again leave its
    reason in `stop` for a ladder that subsequently finishes cleanly."""
    body_norm = _normalize_ws(_try_candidate_body())
    for reason in ("transaction busy", "not the published model", "transaction refused", "KV would be demoted"):
        assert not re.search(rf'stop\s*=\s*"{re.escape(reason)}"', body_norm), (
            f"try_candidate() must not assign stop directly for {reason!r} -- it must return the reason instead"
        )
        # The positive half: every reason IS returned (not silently dropped).
        assert re.search(rf'return\s*"{re.escape(reason)}"\s*;', body_norm), (
            f"try_candidate() must return {reason!r} from somewhere in its body"
        )


# ---------------------------------------------------------------------------
# Publish-before-reserve order, and the host-fallback query after reserve --
# both now checked WITHIN try_candidate() (llama.cpp-7n6n): the sequence
# used to live twice (once inline in the ladder loop, once inline in the
# cache-hit revalidation); it now lives once, in the shared lambda, called
# from both places (pinned by test_both_call_sites_use_try_candidate below).
# ---------------------------------------------------------------------------


def test_publish_happens_before_reserve_in_try_candidate():
    """sycl_resync_runtime_context_flash_attn() (publish) must be called
    strictly BEFORE sched_reserve() inside try_candidate() -- the narrow
    flash-attn re-check inside sched_reserve() re-evaluates against the
    PUBLISHED plan's planner_n_ubatch, so publishing after reserving would
    let sched_reserve() see the PREVIOUS candidate's plan."""
    body_norm = _normalize_ws(_try_candidate_body())
    publish_idx = body_norm.find("sycl_resync_runtime_context_flash_attn();")
    reserve_idx = body_norm.find("sched_reserve();")
    assert publish_idx != -1 and reserve_idx != -1, "both the publish and the reserve call must exist"
    assert publish_idx < reserve_idx, "sycl_resync_runtime_context_flash_attn() must precede sched_reserve()"


def test_publish_reserve_order_has_a_mutation_witness():
    """Mutation witness for the ordering check above: proves it would
    actually catch the publish's try/catch block being moved below the
    reserve. Anchored on code lines only, so rewording a comment between
    the two cannot disarm it."""
    raw = LLAMA_CONTEXT_CPP
    publish_block = _CANDIDATE_PUBLISH_BLOCK
    reserve_end = '            return "compute buffers did not fit";\n        }\n'
    assert raw.count(publish_block) == 1, "mutation target not found -- update this witness to match the real source"
    assert raw.count(reserve_end) == 1, "mutation anchor not found -- update this witness to match the real source"
    mutated_raw = raw.replace(publish_block, "", 1)
    mutated_raw = mutated_raw.replace(reserve_end, reserve_end + publish_block, 1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _TRY_CANDIDATE_START, _TRY_CANDIDATE_END)
    mutated_publish_idx = mutated_body_norm.find("sycl_resync_runtime_context_flash_attn();")
    mutated_reserve_idx = mutated_body_norm.find("sched_reserve();")
    assert mutated_publish_idx != -1 and mutated_reserve_idx != -1
    assert not (mutated_publish_idx < mutated_reserve_idx), (
        "mutation witness is broken: moving the publish below the reserve should make the ordering check fail"
    )


def test_host_fallback_query_is_consulted_after_reserve():
    """ggml_backend_sycl_compute_buffer_host_fallbacks() must be consulted
    AFTER the reserve call inside try_candidate() (it reports fallbacks
    since the last successful publish, which sched_reserve() just
    exercised), and BEFORE the lambda's own success return -- a candidate
    whose reserve fell back to host must not be reported as a pass."""
    body_norm = _normalize_ws(_try_candidate_body())
    reserve_idx = body_norm.find("sched_reserve();")
    fallback_idx = body_norm.find("fallback_fn(", reserve_idx)
    success_return_idx = body_norm.find("return nullptr;", reserve_idx)
    assert reserve_idx != -1 and fallback_idx != -1 and success_return_idx != -1
    assert reserve_idx < fallback_idx < success_return_idx, (
        "the host-fallback query must run strictly between the reserve() call and the lambda's own success "
        "return"
    )


def test_ladder_updates_last_good_only_after_try_candidate_passes():
    """In the LADDER LOOP specifically (not try_candidate() itself), the
    call to try_candidate(c) must precede the `last_good = c;` assignment
    -- a losing candidate (non-null reason) must not win."""
    body_norm = _normalize_ws(_trial_body())
    loop_idx = body_norm.find("for (uint32_t c : rung_ladder)")
    assert loop_idx != -1
    call_idx = body_norm.find("try_candidate(c)", loop_idx)
    last_good_idx = body_norm.find("last_good = c;", loop_idx)
    assert call_idx != -1 and last_good_idx != -1
    assert call_idx < last_good_idx, "the ladder loop must call try_candidate(c) before assigning last_good = c"


def test_fallback_fn_is_bound_to_the_real_accessor():
    """fallback_fn must be bound to ggml_backend_sycl_compute_buffer_host_
    fallbacks in BOTH branches -- the direct GGML_USE_SYCL address-of and
    the GGML_BACKEND_DL-without-SYCL proc-address lookup -- not merely
    called as fallback_fn(...) with the binding itself unpinned (a rename
    or a swap to the wrong accessor would otherwise still satisfy every
    other check in this file, since they all match on the local name
    fallback_fn, not the real symbol it resolves to)."""
    body_norm = _normalize_ws(_trial_body())
    assert re.search(
        r"auto\s+fallback_fn\s*=\s*&ggml_backend_sycl_compute_buffer_host_fallbacks\s*;", body_norm
    ), "the direct GGML_USE_SYCL branch must bind fallback_fn = &ggml_backend_sycl_compute_buffer_host_fallbacks"
    assert re.search(
        r"auto\s+fallback_fn\s*=\s*llama_context_sycl_fallbacks_proc\s*\(\s*first_dev\s*\)\s*;", body_norm
    ), (
        "the GGML_BACKEND_DL-without-SYCL branch must bind fallback_fn via "
        "llama_context_sycl_fallbacks_proc(first_dev)"
    )


def test_host_fallback_query_has_a_mutation_witness():
    """Mutation witness for the check above: proves it would actually
    catch the host-fallback query being deleted (a losing candidate would
    then silently win).

    llama.cpp-7n6n: unlike the pre-refactor version of
    this witness, no loop-scoping workaround is needed any more -- since
    try_candidate() is now the ONLY place fallback_fn( is called at all
    (both the cache-hit revalidation and the ladder loop call the same
    lambda instead of each carrying their own copy), deleting this block
    removes every fallback_fn( occurrence in the whole trial body, not just
    the loop's own copy."""
    raw = LLAMA_CONTEXT_CPP
    # The block restores cparams.pipeline_parallel before returning the
    # loss; it is included verbatim so this witness matches the real source
    # exactly.
    fallback_block = (
        "        for (auto & sb : sycl_backends) {\n"
        "            if (fallback_fn(sb.dev_index) > 0) {\n"
        "                cparams.pipeline_parallel = pipeline_parallel_before_reserve;\n"
        "                sched_matches_last_good   = false;\n"
        '                return "compute buffer fell back to host";\n'
        "            }\n"
        "        }\n"
    )
    assert fallback_block in raw, "mutation target not found -- update this witness to match the real source"
    mutated_raw = raw.replace(fallback_block, "", 1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _TRIAL_START, _TRIAL_END)
    assert "fallback_fn(" not in mutated_body_norm, (
        "mutation witness is broken: deleting the block should remove every fallback_fn( call from the body"
    )


# ---------------------------------------------------------------------------
# llama.cpp-pyu4: the in-loop candidate's own sched_reserve() call
# must be guarded exactly like the publish beside it (a candidate the probe
# accepted but whose reserve throws must lose like any other candidate, not
# abort context creation), and a losing candidate must not leave
# cparams.pipeline_parallel permanently disabled for last_good.
# ---------------------------------------------------------------------------


_IN_LOOP_RESERVE_WRAP_RE = (
    r"try\s*\{\s*sched_reserve\s*\(\s*\)\s*;\s*\}\s*catch\s*\(\s*const\s+std::exception\s*&\s*e\s*\)\s*\{\s*"
    r"LLAMA_LOG_INFO\s*\([^;]*\be\.what\s*\(\s*\)\s*\)\s*;\s*"
    r"cparams\.pipeline_parallel\s*=\s*pipeline_parallel_before_reserve\s*;\s*sched_matches_last_good\s*=\s*"
    r'false\s*;\s*rung_fit_refused\s*=\s*dynamic_cast\s*<\s*const\s+llama_auto_ubatch_fit_refusal\s*\*\s*>\s*\(\s*&\s*e\s*\)\s*!=\s*nullptr\s*;'
    r'\s*return\s*"compute buffers did not fit"\s*;\s*\}'
)


def test_in_loop_reserve_is_wrapped_in_try_catch():
    """llama.cpp-pyu4: the candidate's own sched_reserve() call -- distinct from the
    settle step's own reserve, which must stay unguarded (see
    test_settle_reserve_is_not_wrapped_in_try_catch below) -- must be
    wrapped in try/catch exactly like the publish beside it, catching on
    "compute buffers did not fit" and marking sched_matches_last_good
    false. sched_reserve() throws "failed to allocate compute pp/tg
    buffers" when graph_reserve() fails even after its own host-pinned
    retry, and can also throw from inside resolve_fused_ops()'s call to
    sycl_recheck_runtime_context_flash_attn() or from memory-module
    initialization -- any of these must make this candidate lose cleanly,
    not escape as an uncaught exception. Because one stop reason covers
    all of them, the catch must log e.what() (at INFO) before losing."""
    body_norm = _normalize_ws(_try_candidate_body())
    assert re.search(_IN_LOOP_RESERVE_WRAP_RE, body_norm), (
        "the in-loop candidate reserve must be wrapped in try { sched_reserve(); } catch (const std::exception "
        '& e) { LLAMA_LOG_INFO(..., e.what()); ...; sched_matches_last_good = false; '
        'return "compute buffers did not fit"; }'
    )


@pytest.mark.parametrize("mutation", ["unwrapped", "cause-not-logged"])
def test_in_loop_reserve_try_catch_has_a_mutation_witness(mutation):
    """Mutation witness for the check above: proves it would actually
    catch the try/catch being deleted (a bare sched_reserve() that lets a
    reserve failure escape), and the cause no longer being logged."""
    raw = LLAMA_CONTEXT_CPP
    wrapped = re.compile(
        r"        try \{\n            sched_reserve\(\);\n        \} catch \(const std::exception & e\) \{\n"
        r'.*?            return "compute buffers did not fit";\n        \}\n',
        re.DOTALL,
    )
    assert len(wrapped.findall(raw)) == 1, "mutation target not found -- update this witness to match the real source"
    if mutation == "unwrapped":
        mutated_raw = wrapped.sub("        sched_reserve();\n", raw, count=1)
    else:
        log_call = re.compile(r"            LLAMA_LOG_INFO\(\"\[SYCL-PLAN\] auto n_ubatch candidate [^;]*;\n")
        assert len(log_call.findall(raw)) == 1, "mutation target not found -- update this witness"
        mutated_raw = log_call.sub("", raw, count=1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _TRY_CANDIDATE_START, _TRY_CANDIDATE_END)
    assert not re.search(_IN_LOOP_RESERVE_WRAP_RE, mutated_body_norm), (
        "mutation witness is broken: the mutant should make the wrapped-reserve check fail"
    )


def _settle_block(body_norm: str) -> str:
    """The settle step: from its own gate up to the auto n_ubatch WARN."""
    settle_start = body_norm.find("if (!sched_matches_last_good")
    assert settle_start != -1, "could not find the settle step's own gate"
    settle_end = body_norm.find('LLAMA_LOG_WARN("[SYCL-PLAN] auto n_ubatch=', settle_start)
    assert settle_end != -1, "could not find the auto n_ubatch WARN after the settle step"
    return body_norm[settle_start:settle_end]


def _settle_reserve_is_unguarded(body_norm: str) -> bool:
    # llama.cpp-kpjw: the settle PUBLISH is recorded rather than thrown (see _settle_refusal_is_never_swallowed), so
    # the reserve is the `else` of that refusal branch and must hold no try/catch of its own.
    settle = _settle_block(body_norm)
    return re.search(r"\} else \{ sched_need_reserve = true; sched_reserve\(\); \}", settle) is not None


def test_settle_reserve_is_not_wrapped_in_try_catch():
    """llama.cpp-pyu4: unlike the in-loop candidate reserve above, the SETTLE step's
    own sched_reserve() call must stay unguarded -- a refusal there must
    propagate, matching today's behaviour for a context that does not fit
    at all (the same asymmetry test_settle_publish_is_not_wrapped_in_try_
    catch already pins for the settle's publish)."""
    assert _settle_reserve_is_unguarded(_normalize_ws(_trial_body())), (
        "the settle reserve must NOT be wrapped in try/catch -- its failure must propagate"
    )


def test_settle_reserve_unguarded_check_has_a_mutation_witness():
    """Mutation witness for the check above: wrapping the settle's reserve
    in try/catch must make it fail."""
    raw = LLAMA_CONTEXT_CPP
    settle_reserve = "        } else {\n            sched_need_reserve = true;\n            sched_reserve();\n        }\n    }\n"
    # opt_init() carries the same three lines, so count and mutate only
    # inside the trial.
    start = raw.find(_TRIAL_START)
    end = raw.find(_TRIAL_END, start + 1)
    assert start != -1 and end != -1, "could not bound the trial -- update this witness to match the real source"
    assert raw.count(settle_reserve, start, end) == 1, (
        "mutation target not found -- update this witness to match the real source"
    )
    mutated_raw = raw[:start] + raw[start:end].replace(
        settle_reserve,
        "        } else {\n"
        "            sched_need_reserve = true;\n"
        "            try {\n"
        "                sched_reserve();\n"
        "            } catch (const std::exception &) {\n"
        "            }\n"
        "        }\n"
        "    }\n",
        1,
    ) + raw[end:]
    assert mutated_raw != raw
    assert not _settle_reserve_is_unguarded(_body_of(mutated_raw, _TRIAL_START, _TRIAL_END)), (
        "mutation witness is broken: wrapping the settle reserve should make the unguarded check fail"
    )


def test_pipeline_parallel_is_saved_before_the_in_loop_reserve():
    """llama.cpp-pyu4: sched_reserve()'s own pipeline-parallel fallback sets
    cparams.pipeline_parallel = false PERMANENTLY the moment a reserve
    needs it -- save the pre-reserve value immediately before the call
    that can flip it, so a losing candidate's fallback can be undone."""
    body_norm = _normalize_ws(_try_candidate_body())
    save_idx = body_norm.find("const bool pipeline_parallel_before_reserve = cparams.pipeline_parallel;")
    reserve_idx = body_norm.find("sched_reserve();", save_idx)
    assert save_idx != -1, "pipeline_parallel_before_reserve must be saved somewhere in try_candidate()"
    assert reserve_idx != -1 and save_idx < reserve_idx, (
        "pipeline_parallel_before_reserve must be saved BEFORE the in-loop sched_reserve() call"
    )


def test_pipeline_parallel_is_restored_on_every_losing_path_after_the_reserve():
    """llama.cpp-pyu4: cparams.pipeline_parallel must be restored to its pre-reserve
    value on BOTH paths where the candidate goes on to lose after the
    reserve call succeeds or throws -- the reserve's own catch, and the
    host-fallback loss right below it -- so a losing candidate never
    leaves pipeline parallelism disabled for last_good, which may never
    have needed the fallback."""
    body_norm = _normalize_ws(_try_candidate_body())
    reserve_idx = body_norm.find("sched_reserve();")
    assert reserve_idx != -1
    restores = [
        m.start()
        for m in re.finditer(r"cparams\.pipeline_parallel\s*=\s*pipeline_parallel_before_reserve\s*;", body_norm)
    ]
    assert len(restores) == 3, (
        f"expected exactly 3 restores of cparams.pipeline_parallel (reserve-throws, host-fallback and "
        f"hold-spill paths) -- found {len(restores)}"
    )
    assert all(idx > reserve_idx for idx in restores), (
        "all restores must appear after the in-loop sched_reserve() call"
    )
    fallback_return_idx = body_norm.find('return "compute buffer fell back to host";')
    assert fallback_return_idx != -1
    assert restores[1] < fallback_return_idx, (
        "the host-fallback loss must restore cparams.pipeline_parallel before returning its reason"
    )
    hold_spill_return_idx = body_norm.find('return "hold spill left no headroom";')
    assert hold_spill_return_idx != -1
    assert restores[1] < restores[2] < hold_spill_return_idx, (
        "the hold-spill loss (llama.cpp-kpjw) must restore cparams.pipeline_parallel before returning its reason"
    )


def test_pipeline_parallel_restore_has_a_mutation_witness():
    """Mutation witness for the two checks above: proves they would
    actually catch the host-fallback loss's own restore being deleted
    (leaving pipeline parallelism disabled for last_good after an
    unrelated LATER candidate's host-pinned fallback)."""
    raw = LLAMA_CONTEXT_CPP
    fallback_block = (
        "        for (auto & sb : sycl_backends) {\n"
        "            if (fallback_fn(sb.dev_index) > 0) {\n"
        "                cparams.pipeline_parallel = pipeline_parallel_before_reserve;\n"
        "                sched_matches_last_good   = false;\n"
        '                return "compute buffer fell back to host";\n'
        "            }\n"
        "        }\n"
    )
    assert fallback_block in raw, "mutation target not found -- update this witness to match the real source"
    mutated_raw = raw.replace(
        fallback_block,
        "        for (auto & sb : sycl_backends) {\n"
        "            if (fallback_fn(sb.dev_index) > 0) {\n"
        "                sched_matches_last_good   = false;\n"
        '                return "compute buffer fell back to host";\n'
        "            }\n"
        "        }\n",
        1,
    )
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _TRY_CANDIDATE_START, _TRY_CANDIDATE_END)
    restores = re.findall(
        r"cparams\.pipeline_parallel\s*=\s*pipeline_parallel_before_reserve\s*;", mutated_body_norm
    )
    assert len(restores) != 3, (
        "mutation witness is broken: deleting the host-fallback restore should make the restore-count check fail"
    )


# ---------------------------------------------------------------------------
# llama.cpp-pyu4: the ladder (and a cache hit) must never choose
# something SMALLER than the caller's own explicit n_ubatch
# (fallback_ubatch) -- a raw-API caller can set llama_context_params.
# n_ubatch above the ladder's first rung together with n_ubatch_auto=true.
# ---------------------------------------------------------------------------


# The floor is now enforced by the rung set the loop iterates: the trial
# passes fallback_ubatch and cap to llama_auto_ubatch_rung_set(), which admits
# a ladder rung only inside [fallback_ubatch, cap], and the loop carries no
# bound checks of its own (tests/test-sycl-planned-ladder-source.py pins that
# deletion). A rung the ladder never tries can never become last_good.
_RUNG_IN_BOUNDS = "if (ladder[i] >= fallback_ubatch && ladder[i] <= cap) {"
_CACHED_VALID_RETURN = (
    "return n_ladder > 0 && cached_ubatch >= ladder[0] && cached_ubatch <= cap && cached_ubatch >= fallback_ubatch;"
)


def _header_norm() -> str:
    return _normalize_ws(strip_comments(LLAMA_AUTO_UBATCH_H))


def test_ladder_skips_rungs_below_fallback_ubatch():
    """llama.cpp-pyu4: the ladder must never try a rung strictly below
    fallback_ubatch -- a rung the ladder never tries can never become
    last_good, so this is what actually enforces the "never silently shrink"
    contract for a raw-API caller's explicit n_ubatch. The rung set admits a
    ladder rung only when fallback_ubatch <= rung <= cap, and the trial hands
    it both bounds."""
    assert _RUNG_IN_BOUNDS in _header_norm(), (
        "llama_auto_ubatch_rung_set must admit a ladder rung only when fallback_ubatch <= rung <= cap"
    )
    assert re.search(
        r"llama_auto_ubatch_rung_set\(\s*ladder\s*,\s*llama_auto_ubatch_ladder_size\s*,\s*fallback_ubatch\s*,\s*cap\s*,",
        _normalize_ws(_trial_body()),
    ), "the trial must build its rung set from fallback_ubatch and cap"


@pytest.mark.parametrize("mutation", ["no-floor-bound", "no-cap-bound"])
def test_ladder_floor_skip_has_a_mutation_witness(mutation):
    """Mutation witness for the check above: dropping either bound of the
    rung set's admission test must make it fail."""
    header = _header_norm()
    assert header.count(_RUNG_IN_BOUNDS) == 1, "mutation target not unique -- update this witness to match the real source"
    replacement = {
        "no-floor-bound": "if (ladder[i] <= cap) {",
        "no-cap-bound": "if (ladder[i] >= fallback_ubatch) {",
    }[mutation]
    assert _RUNG_IN_BOUNDS not in header.replace(_RUNG_IN_BOUNDS, replacement, 1), (
        "mutation witness is broken: the mutant should make the bound check fail"
    )


def test_cache_hit_below_fallback_ubatch_is_treated_as_a_miss():
    """llama.cpp-pyu4: a cached value below fallback_ubatch must not win either --
    the trial's cache_usable must come from llama_auto_ubatch_cached_valid,
    which also checks cached_ubatch >= fallback_ubatch alongside the
    ladder[0]/cap bounds, and a value it rejects is a miss."""
    assert _CACHED_VALID_RETURN in _header_norm(), (
        "llama_auto_ubatch_cached_valid must reject cached_ubatch < fallback_ubatch"
    )
    body_norm = _normalize_ws(_trial_body())
    assert re.search(
        r"cache_usable\s*=\s*cache_found\s*&&\s*llama_auto_ubatch_cached_valid\(\s*ladder\s*,\s*"
        r"llama_auto_ubatch_ladder_size\s*,\s*cached_ubatch\s*,\s*fallback_ubatch\s*,\s*cap\s*\)\s*;",
        body_norm,
    ), "cache_usable must be cache_found && llama_auto_ubatch_cached_valid(..., cached_ubatch, fallback_ubatch, cap)"
    assert re.search(r"if\s*\(\s*!\s*cache_usable\s*\)\s*\{\s*cache_state\s*=\s*\"miss\"\s*;", body_norm), (
        "an unusable cached value must be reported as a miss"
    )


def test_cache_hit_floor_has_a_mutation_witness():
    """Mutation witness for the check above: proves it would actually
    catch the `cached_ubatch >= fallback_ubatch` clause being dropped
    (which would let a stale cached value below the caller's explicit
    n_ubatch win without ever revalidating against the floor)."""
    header = _header_norm()
    assert header.count(_CACHED_VALID_RETURN) == 1, "mutation target not unique -- update this witness to match the real source"
    mutated = header.replace(_CACHED_VALID_RETURN, _CACHED_VALID_RETURN.replace(" && cached_ubatch >= fallback_ubatch", ""), 1)
    assert _CACHED_VALID_RETURN not in mutated, (
        "mutation witness is broken: dropping the clause should remove it from the helper"
    )


_HAS_CANDIDATE_BOTH_BOUNDS_RE = (
    r"inline\s+bool\s+llama_auto_ubatch_ladder_has_candidate\s*\(\s*const\s+uint32_t\s*\*\s*ladder\s*,\s*"
    r"size_t\s+n_ladder\s*,\s*uint32_t\s+ubatch_floor\s*,\s*uint32_t\s+ubatch_cap\s*\)\s*\{\s*"
    r"for\s*\(\s*size_t\s+i\s*=\s*0\s*;\s*i\s*<\s*n_ladder\s*;\s*\+\+i\s*\)\s*\{\s*"
    r"if\s*\(\s*ladder\[i\]\s*>=\s*ubatch_floor\s*&&\s*ladder\[i\]\s*<=\s*ubatch_cap\s*\)\s*\{\s*"
    r"return\s+true\s*;\s*\}\s*\}\s*return\s+false\s*;\s*\}"
)

_EARLY_EXIT_RE = (
    r"if\s*\(\s*tried\.empty\s*\(\s*\)\s*&&\s*!\s*llama_auto_ubatch_ladder_has_candidate\s*\(\s*ladder\s*,\s*"
    r"llama_auto_ubatch_ladder_size\s*,\s*fallback_ubatch\s*,\s*cap\s*\)\s*"
    r"\)\s*\{"
)


def test_ladder_has_candidate_checks_both_bounds():
    """llama.cpp-pyu4: llama_auto_ubatch_ladder_has_candidate()
    (src/llama-auto-ubatch.h) must count a rung only when floor <= rung <=
    cap. The ladder loop skips rungs under the floor AND breaks at the
    first rung over the cap, so dropping either half readmits a shape in
    which the loop tries nothing. tests/test-auto-ubatch-ladder.cpp
    executes the helper on those shapes; this pins the text it runs."""
    assert re.search(_HAS_CANDIDATE_BOTH_BOUNDS_RE, _normalize_ws(strip_comments(LLAMA_AUTO_UBATCH_H))), (
        "llama_auto_ubatch_ladder_has_candidate must return true only when some rung satisfies "
        "ubatch_floor <= rung <= ubatch_cap"
    )


@pytest.mark.parametrize(
    "mutant",
    [
        "        if (ladder[i] >= ubatch_floor) {\n",
        "        if (ladder[i] <= ubatch_cap) {\n",
    ],
    ids=["no-cap-bound", "no-floor-bound"],
)
def test_ladder_has_candidate_one_bound_mutants_have_a_witness(mutant):
    """Mutation witness for the check above: dropping either bound must
    make it fail."""
    raw = LLAMA_AUTO_UBATCH_H
    line = "        if (ladder[i] >= ubatch_floor && ladder[i] <= ubatch_cap) {\n"
    assert raw.count(line) == 1, f"mutation target not unique -- found {raw.count(line)}"
    mutated_raw = raw.replace(line, mutant, 1)
    assert mutated_raw != raw
    assert not re.search(_HAS_CANDIDATE_BOTH_BOUNDS_RE, _normalize_ws(strip_comments(mutated_raw))), (
        "mutation witness is broken: dropping a bound should make the both-bounds check fail"
    )


def test_ladder_has_candidate_host_test_is_registered_and_uses_the_trial_ladder():
    """The executable witness must actually run under ctest, and its cases
    are only meaningful against the trial's own rungs (the 600/1000 case
    needs 512 and 1024 to be adjacent rungs), so it must read the header's
    ladder rather than carry a copy."""
    assert re.search(r"^\s*llama_build_and_test\(\s*test-auto-ubatch-ladder\.cpp\b", TESTS_CMAKELISTS, re.M), (
        "tests/CMakeLists.txt must register test-auto-ubatch-ladder.cpp with llama_build_and_test"
    )
    code = strip_comments(TEST_AUTO_UBATCH_LADDER_CPP)
    assert '#include "../src/llama-auto-ubatch.h"' in code
    assert re.search(r"=\s*llama_auto_ubatch_ladder\s*;", code), "the host test must use llama_auto_ubatch_ladder"
    assert re.search(r"=\s*llama_auto_ubatch_ladder_size\s*;", code), (
        "the host test must use llama_auto_ubatch_ladder_size"
    )
    assert not re.search(r"uint32_t\s+\w*ladder\w*\s*\[\s*\]\s*=", code), (
        "the host test must not carry its own copy of the ladder"
    )


def test_no_candidate_rung_exits_before_the_loop_with_no_warn():
    """llama.cpp-pyu4: the trial must exit before the ladder loop
    when no rung lies in [fallback_ubatch, cap] and no cache candidate was
    tried this trial -- gated on tried.empty() &&
    !llama_auto_ubatch_ladder_has_candidate(ladder, ..., fallback_ubatch,
    cap). Not only when fallback_ubatch exceeds the largest rung: with
    n_batch=1000 and n_ubatch=600 the floor is 600 and the cap 1000, so 512
    is skipped and 1024 breaks the loop; and a MoE routing ceiling can
    narrow cap below an explicit n_ubatch. Either would otherwise reach the
    [SYCL-PLAN] auto n_ubatch= WARN with an empty `tried` list and persist
    a terminal cache entry for a ladder that never ran -- the shape
    test_cap_below_first_rung_exits_before_the_loop_with_no_warn forbids at
    the other edge. The exit comes AFTER the tuning-cache lookup (a
    persisted value at or above the floor can still be revalidated) and
    BEFORE the loop."""
    body_norm = _normalize_ws(_trial_body())
    cache_warn_idx = body_norm.find("[SYCL-PLAN] tuning cache %s: n_ubatch=%u (%s)")
    loop_idx = body_norm.find("for (uint32_t c : rung_ladder)")
    warn_idx = body_norm.find("[SYCL-PLAN] auto n_ubatch=")
    assert cache_warn_idx != -1 and loop_idx != -1 and warn_idx != -1

    exit_match = re.search(_EARLY_EXIT_RE, body_norm)
    assert exit_match is not None, (
        "there must be an early exit gated on tried.empty() && "
        "!llama_auto_ubatch_ladder_has_candidate(ladder, llama_auto_ubatch_ladder_size, fallback_ubatch, cap)"
    )
    assert cache_warn_idx < exit_match.start() < loop_idx, (
        "the no-candidate early exit must come after the tuning-cache WARN and before the ladder loop"
    )
    assert exit_match.start() < warn_idx, "the WARN literal must not appear before this early exit"

    exit_block = body_norm[exit_match.start() : body_norm.find("}", exit_match.start()) + 1]
    assert re.search(r"sched_reserve\(\s*\)\s*;\s*return\s*;", exit_block), (
        "the early exit must call sched_reserve() then return, with no publish, no store, and no WARN in between"
    )


_EARLY_EXIT_BLOCK = (
    "    if (tried.empty() &&\n"
    "        !llama_auto_ubatch_ladder_has_candidate(ladder, llama_auto_ubatch_ladder_size, fallback_ubatch, cap)) {\n"
    "        sched_reserve();\n"
    "        return;\n"
    "    }\n"
)


@pytest.mark.parametrize(
    "replacement",
    [
        # deleted outright: the loop runs with nothing to try
        "",
        # the narrower last-rung form, which misses the 600/1000 and MoE shapes
        "    if (tried.empty() && fallback_ubatch > ladder[llama_auto_ubatch_ladder_size - 1]) {\n"
        "        sched_reserve();\n"
        "        return;\n"
        "    }\n",
        # the cap argument swapped for the largest rung
        "    if (tried.empty() &&\n"
        "        !llama_auto_ubatch_ladder_has_candidate(ladder, llama_auto_ubatch_ladder_size, fallback_ubatch, 4096)) {\n"
        "        sched_reserve();\n"
        "        return;\n"
        "    }\n",
    ],
    ids=["deleted", "last-rung-form", "cap-replaced-by-4096"],
)
def test_no_candidate_early_exit_mutants_have_a_witness(replacement):
    """Mutation witness for the check above."""
    raw = LLAMA_CONTEXT_CPP
    assert raw.count(_EARLY_EXIT_BLOCK) == 1, (
        "mutation target not found -- update this witness to match the real source"
    )
    mutated_raw = raw.replace(_EARLY_EXIT_BLOCK, replacement, 1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _TRIAL_START, _TRIAL_END)
    assert not re.search(_EARLY_EXIT_RE, mutated_body_norm), (
        "mutation witness is broken: the mutant should make the early-exit check fail"
    )


# ---------------------------------------------------------------------------
# Residue-freedom comes from the sched.reset() inside sched_reserve(), not
# ggml-alloc.c's realloc-on-shrink-no-op, and the trial's comment must say so.
# ---------------------------------------------------------------------------


def test_residue_freedom_comment_credits_sched_reset_not_ggml_alloc():
    """The trial's own docstring must credit residue-freedom (a losing
    candidate's oversized buffers not being left behind) to
    sched_reserve()'s sched.reset(ggml_backend_sched_new(...)) call, which
    is the actual mechanism -- not to a nonexistent "ggml-alloc.c's
    realloc-on-shrink-no-op"."""
    assert "ggml-alloc.c's realloc-on-shrink-no-op" not in LLAMA_CONTEXT_CPP, (
        "the inaccurate ggml-alloc.c attribution must be gone from the comment"
    )
    assert "sched.reset(ggml_backend_sched_new(...)) destroys the whole scheduler" in LLAMA_CONTEXT_CPP, (
        "the comment must credit the real mechanism: sched_reserve()'s own sched.reset(...) call"
    )


def test_both_call_sites_use_try_candidate():
    """llama.cpp-7n6n: both the cache-hit
    revalidation and the ladder loop must call the SAME try_candidate()
    lambda -- not each carry their own inline copy of the per-candidate
    sequence, which is what previously let a losing cache revalidation's
    reason leak into the ladder's own outcome (see test_all_eight_stop_
    reasons_are_present's docstring). A regression that reintroduced
    an inline copy at either call site (rather than a call to the shared
    lambda) would satisfy every other check in this section (they all
    look inside try_candidate()'s own body) while silently duplicating the
    logic again."""
    body_norm = _normalize_ws(_trial_body())
    assert re.search(r"try_candidate\s*\(\s*cached_ubatch\s*\)", body_norm), (
        "the cache-hit revalidation must call try_candidate(cached_ubatch)"
    )
    assert re.search(r"try_candidate\s*\(\s*c\s*\)", body_norm), "the ladder loop must call try_candidate(c)"
    assert len(re.findall(r"auto\s+try_candidate\s*=\s*\[&\]", body_norm)) == 1, (
        "try_candidate must be defined exactly once"
    )


# ---------------------------------------------------------------------------
# The WARN line and the stop-reason vocabulary
# ---------------------------------------------------------------------------


def test_exactly_one_sycl_plan_auto_warn_in_the_body():
    """Exactly one [SYCL-PLAN] auto n_ubatch= WARN must appear in the
    trial's body -- the task spec's "keep the trial's own log to ONE WARN"
    gotcha. (A second, separate WARN family -- "[SYCL-PLAN] tuning cache
    ..." -- reports the persisted-cache outcome, llama.cpp-7n6n Task 5; it
    is a different literal string and does not count against this one.)"""
    body_norm = _normalize_ws(_trial_body())
    count = len(re.findall(r"\[SYCL-PLAN\] auto n_ubatch=", body_norm))
    assert count == 1, f"expected exactly one '[SYCL-PLAN] auto n_ubatch=' WARN -- found {count}"
    assert re.search(r'LLAMA_LOG_WARN\(\s*"\[SYCL-PLAN\] auto n_ubatch=', body_norm), (
        "the line must be logged at GGML_LOG_WARN (LLAMA_LOG_WARN), not INFO"
    )


def test_all_ten_stop_reasons_are_present():
    """The trial's stop-reason vocabulary must be EXACTLY the ten strings
    the task spec names -- the original seven (llama.cpp-xojq Task 4b),
    "cached" (llama.cpp-7n6n, Task 5: a persisted-cache hit that
    revalidates cleanly skips the ladder with this stop reason), and
    "compute buffers did not fit" (llama.cpp-pyu4: the in-loop
    candidate's own sched_reserve() threw).

    llama.cpp-7n6n: three of these are still
    literal `stop = "...";` assignments ("ladder exhausted"'s initial
    declaration, "MoE GPU routing ceiling", and "cached"); the other six --
    including this task's new one -- are `return "...";` statements inside
    try_candidate(), moved there specifically so a losing cache
    revalidation cannot leave its reason behind in `stop`. Both forms are
    drawn from and must match this one set, with none missing and none
    extra."""
    body_norm = _normalize_ws(_trial_body())
    ten = {
        "ladder exhausted",
        "MoE GPU routing ceiling",
        "transaction refused",
        "transaction busy",
        "not the published model",
        "KV would be demoted",
        "compute buffer fell back to host",
        "hold spill left no headroom",
        "compute buffers did not fit",
        "cached",
    }
    for reason in ten:
        assert f'"{reason}"' in body_norm, f"missing stop reason literal: {reason!r}"

    # Presence alone (the loop above) would pass even if a tenth string had
    # silently slipped in as a `stop = "..."` or `return "...";` somewhere --
    # this closes that gap by requiring the extracted SET to match exactly.
    found = set(re.findall(r'stop\s*=\s*"([^"]*)"', body_norm)) | set(
        re.findall(r'return\s*"([^"]*)"\s*;', body_norm)
    )
    # The initial stop reason is the helper's binding verdict: the MoE ceiling
    # when llama_auto_ubatch_cap reports it bound, else the ladder running out.
    for pair in re.findall(r'stop\s*=\s*moe_bound\s*\?\s*"([^"]*)"\s*:\s*"([^"]*)"\s*;', body_norm):
        found |= set(pair)
    assert found == ten, f"stop-reason literal set does not match exactly -- found {found}"


# ---------------------------------------------------------------------------
# The candidate publish can still refuse after an accepted probe (Task 2
# final review addendum, 2026-09-11: the probe's own exit branch returns
# BEFORE the publication-ID check, MMID materialization, and the CAS, all of
# which still run for a real publish and can still refuse).
# ---------------------------------------------------------------------------


_CANDIDATE_PUBLISH_CATCH_RE = (
    r"try\s*\{\s*sycl_resync_runtime_context_flash_attn\(\s*\)\s*;\s*\}\s*catch\s*\(\s*const\s+std::exception\s*"
    r"&\s*\)\s*\{\s*sched_matches_last_good\s*=\s*false\s*;\s*publish_dirty\s*=\s*true\s*;\s*"
    r'return\s*"transaction refused"\s*;\s*\}'
)


def test_candidate_publish_is_wrapped_in_try_catch():
    """The publish inside try_candidate() must be wrapped in
    try/catch(const std::exception&) -- sycl_resync_runtime_context_flash_
    attn() throws on a refusal, and an accepted probe does not guarantee
    the publish itself still succeeds. The catch must also mark
    sched_matches_last_good false (spec round 1 F1) and return the
    "transaction refused" reason immediately -- try_candidate() RETURNING
    from inside the catch block is itself the guarantee that a refused
    publish can never fall through to the reserve/host-fallback stages
    below it. This replaces the pre-refactor `candidate_lost = true;`
    flag-and-check-immediately-after pattern with a stronger, structural
    one -- a `return` cannot be "forgotten" the way a flag check
    theoretically could be."""
    body_norm = _normalize_ws(_try_candidate_body())
    assert re.search(
        _CANDIDATE_PUBLISH_CATCH_RE,
        body_norm,
    ), (
        "the publish must be wrapped in try { ... } catch (const std::exception &) { sched_matches_last_good = "
        'false; publish_dirty = true; return "transaction refused"; } -- a publish that threw may have landed on '
        "some devices, so the settle must republish"
    )


@pytest.mark.parametrize("mutation", ["unwrapped", "dirty-flag-dropped"])
def test_candidate_publish_try_catch_has_a_mutation_witness(mutation):
    """Mutation witness for the check above: proves it would actually
    catch the try/catch being deleted (leaving a bare, unprotected publish
    call that would let a refusal escape as an uncaught exception instead
    of a clean "transaction refused" stop), and the catch no longer
    marking the publish dirty."""
    raw = LLAMA_CONTEXT_CPP
    assert _CANDIDATE_PUBLISH_BLOCK in raw, "mutation target not found -- update this witness to match the real source"
    if mutation == "unwrapped":
        replacement = "        sycl_resync_runtime_context_flash_attn();\n"
    else:
        replacement = _CANDIDATE_PUBLISH_BLOCK.replace("            publish_dirty           = true;\n", "")
    mutated_raw = raw.replace(_CANDIDATE_PUBLISH_BLOCK, replacement, 1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _TRY_CANDIDATE_START, _TRY_CANDIDATE_END)
    assert not re.search(
        _CANDIDATE_PUBLISH_CATCH_RE,
        mutated_body_norm,
    ), "mutation witness is broken: the mutant should make the wrapped-publish check fail"


def _balanced_braces(text: str, open_at: int) -> str:
    """The `{ ... }` block whose opening brace is text[open_at]."""
    assert text[open_at] == "{"
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_at : i + 1]
    raise AssertionError("unbalanced braces")


def test_settle_republishes_only_when_needed():
    """The settle step must re-publish and re-reserve only when the
    published plan/sched do not already describe last_good -- not
    unconditionally (which would waste a redundant transaction+reserve on
    the common case where the ladder's last accepted candidate already
    won)."""
    body_norm = _normalize_ws(_trial_body())
    assert re.search(
        r"if\s*\(\s*!sched_matches_last_good\s*\|\|\s*cparams\.n_ubatch\s*!=\s*last_good\s*\)\s*\{", body_norm
    ), "the settle step must be gated on !sched_matches_last_good || cparams.n_ubatch != last_good"


def test_publish_flags_are_declared_false_and_published_any_set_after_the_publish_catch():
    """published_any and publish_dirty must both start false. published_any
    must become true right after try_candidate()'s own publish succeeds
    (the try did not throw, so its catch block's early return was not
    taken) -- tracking whether ANY candidate's publish actually took effect
    this trial run, independent of whether that candidate goes on to lose
    the host-fallback check. publish_dirty is set by that catch instead (see
    the try/catch check above): a publish that threw may still have landed
    on some devices."""
    body_norm = _normalize_ws(_trial_body())
    assert re.search(r"bool\s+published_any\s*=\s*false\s*;", body_norm), (
        "published_any must be declared, initialized false"
    )
    assert re.search(r"bool\s+publish_dirty\s*=\s*false\s*;", body_norm), (
        "publish_dirty must be declared, initialized false"
    )

    # llama.cpp-7n6n: the old anchor
    # (`"sched_matches_last_good = false; } if (candidate_lost) { break; }"`)
    # named a flag-and-check pattern that no longer exists -- the catch
    # block now returns immediately instead. The new anchor is the catch
    # clause's own opening, which is still unique in the body.
    catch_idx = body_norm.find("catch (const std::exception &) { sched_matches_last_good = false;")
    assert catch_idx != -1, "could not find the publish's try/catch clause"
    published_any_idx = body_norm.find("published_any = true;", catch_idx)
    reserve_idx = body_norm.find("sched_need_reserve = true;", catch_idx)
    assert published_any_idx != -1 and reserve_idx != -1
    assert catch_idx < published_any_idx < reserve_idx, (
        "published_any must be set true strictly between the publish's own try/catch and the reserve() call"
    )


def _assert_settle_publish_call_is_inside_the_guard(settle_block: str, publish_guard_idx: int) -> None:
    """Shared by the real check below and its mutation witness (quality
    round 2 R3): the positional ordering check alone (need_publish <
    assign < guard < call < reserve) passes even when the guard's body is
    EMPTIED and the publish call moved to just after it -- ordering never
    notices, since the call still textually follows the guard. Bounding
    the guard's own `{ ... }` body and requiring the call INSIDE it closes
    that gap."""
    guard_body_end = settle_block.find("}", publish_guard_idx)
    assert guard_body_end != -1, "could not bound the publish guard's body"
    guard_body = settle_block[publish_guard_idx : guard_body_end + 1]
    assert "sycl_resync_runtime_context_flash_attn();" in guard_body, (
        "the publish call must be INSIDE the need_publish guard's body, not merely appear somewhere after it"
    )


_SETTLE_NEED_PUBLISH = (
    "const bool need_publish = llama_auto_ubatch_settle_needs_publish(published_any, publish_dirty, "
    "cparams.n_ubatch, fallback_ubatch);"
)


def _assert_settle_publish_gate(body_norm: str) -> None:
    """The settle step's PUBLISH (not its reserve) must be gated on
    llama_auto_ubatch_settle_needs_publish(published_any, publish_dirty,
    cparams.n_ubatch, fallback_ubatch). A candidate publish that took
    effect (published_any) or threw after possibly landing on some devices
    (publish_dirty) forces a republish of last_good on every device; when
    neither happened the constructor's own earlier publish still describes
    fallback_ubatch, and a settle republish would be a redundant
    runtime-context transaction with no state change (e.g. the loop
    stopping at once on a first-candidate probe refusal). The RESERVE must
    still run unconditionally inside the settle gate -- a context that
    never calls sched_reserve() anywhere has no compute buffers at all.
    Shared by the real check and its mutation witness."""
    settle_idx = body_norm.find("if (!sched_matches_last_good")
    assert settle_idx != -1
    settle_block = body_norm[settle_idx:]

    need_publish_idx = settle_block.find(_SETTLE_NEED_PUBLISH)
    assign_idx = settle_block.find("cparams.n_ubatch = last_good;")
    publish_guard_idx = settle_block.find("if (need_publish) {")
    publish_call_idx = settle_block.find("sycl_resync_runtime_context_flash_attn();")
    reserve_idx = settle_block.find("sched_reserve();")
    assert -1 not in (need_publish_idx, assign_idx, publish_guard_idx, publish_call_idx, reserve_idx), (
        "the settle block must compute need_publish, reassign cparams.n_ubatch, gate the publish on it, and "
        "still reserve"
    )
    assert need_publish_idx < assign_idx < publish_guard_idx < publish_call_idx < reserve_idx, (
        "the settle block's need_publish computation must precede the cparams.n_ubatch reassignment, which "
        "must precede the publish gate, which must precede the publish call, which must precede the reserve"
    )
    _assert_settle_publish_call_is_inside_the_guard(settle_block, publish_guard_idx)


def test_settle_publish_is_gated_on_settle_needs_publish():
    """See _assert_settle_publish_gate()."""
    _assert_settle_publish_gate(_normalize_ws(_trial_body()))


def test_settle_publish_gate_has_a_mutation_witness():
    """Mutation witness for the two checks above: proves they would
    actually catch the settle publish reverting to unconditional (always
    publishing, even when nothing changed), AND that the guard-body check
    specifically would catch the guard being emptied with the call moved
    below it -- a mutant the ordering assertion alone cannot see (ordering
    is satisfied textually either way)."""
    raw = LLAMA_CONTEXT_CPP
    # The settle block as it stands (its publish carries the kpjw catch, which is the part of the text this witness
    # does not care about), cut out of the source rather than copied so the witness follows the real shape.
    settle_open = "    if (!sched_matches_last_good || cparams.n_ubatch != last_good) {\n"
    settle_close = "        } else {\n            sched_need_reserve = true;\n            sched_reserve();\n        }\n    }\n"
    trial_at = raw.find(_TRIAL_START)
    open_at = raw.find(settle_open, trial_at)
    close_at = raw.find(settle_close, open_at)
    assert -1 not in (trial_at, open_at, close_at), (
        "mutation target not found -- update this witness to match the real source"
    )
    original_settle = raw[open_at : close_at + len(settle_close)]
    mutated_settle = (
        "    if (!sched_matches_last_good || cparams.n_ubatch != last_good) {\n"
        "        cparams.n_ubatch = last_good;\n"
        "        sycl_resync_runtime_context_flash_attn();\n"
        "        sched_need_reserve = true;\n"
        "        sched_reserve();\n"
        "    }\n"
    )
    mutated_raw = raw.replace(original_settle, mutated_settle, 1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _TRIAL_START, _TRIAL_END)
    with pytest.raises(AssertionError, match="the settle block must compute need_publish"):
        _assert_settle_publish_gate(mutated_body_norm)

    # Second mutant: keeps need_publish and its
    # ordering intact, but EMPTIES the guard's body and moves the publish
    # call to just after it -- passes the positional ordering assertion
    # above (need_publish < assign < guard < call < reserve still holds
    # textually) but must fail the REAL guard-body check, not a
    # re-implementation of it.
    emptied_guard_settle = (
        "    if (!sched_matches_last_good || cparams.n_ubatch != last_good) {\n"
        "        const bool need_publish =\n"
        "            llama_auto_ubatch_settle_needs_publish(published_any, publish_dirty, cparams.n_ubatch, "
        "fallback_ubatch);\n"
        "        cparams.n_ubatch = last_good;\n"
        "        if (need_publish) {\n"
        "        }\n"
        "        sycl_resync_runtime_context_flash_attn();\n"
        "        sched_need_reserve = true;\n"
        "        sched_reserve();\n"
        "    }\n"
    )
    mutated_raw_2 = raw.replace(original_settle, emptied_guard_settle, 1)
    assert mutated_raw_2 != raw

    mutated_body_norm_2 = _body_of(mutated_raw_2, _TRIAL_START, _TRIAL_END)
    mutated_settle_idx_2 = mutated_body_norm_2.find("if (!sched_matches_last_good")
    assert mutated_settle_idx_2 != -1
    mutated_settle_block_2 = mutated_body_norm_2[mutated_settle_idx_2:]
    mutated_publish_guard_idx_2 = mutated_settle_block_2.find("if (need_publish) {")
    assert mutated_publish_guard_idx_2 != -1, "mutation witness is broken: could not re-find the emptied guard"

    with pytest.raises(AssertionError, match="must be INSIDE the need_publish guard"):
        _assert_settle_publish_call_is_inside_the_guard(mutated_settle_block_2, mutated_publish_guard_idx_2)


_SETTLE_NEEDS_PUBLISH_RE = (
    r"inline\s+bool\s+llama_auto_ubatch_settle_needs_publish\s*\(\s*bool\s+published_any\s*,\s*"
    r"bool\s+publish_dirty\s*,\s*uint32_t\s+n_ubatch\s*,\s*uint32_t\s+fallback_ubatch\s*\)\s*\{\s*"
    r"return\s+published_any\s*\|\|\s*publish_dirty\s*\|\|\s*n_ubatch\s*!=\s*fallback_ubatch\s*;\s*\}"
)


def test_settle_needs_publish_reads_publish_dirty():
    """llama.cpp-pyu4: a candidate publish that threw may have landed on
    some devices before one refused, and it never sets published_any. The
    settle's cparams.n_ubatch only records the LAST candidate, so a later
    candidate at fallback_ubatch would erase the evidence of an earlier
    partial publish above it (cached 1024 half-published, then rung 512's
    publish also throws: need_publish would be false and a device would
    keep the 1024 plan). llama_auto_ubatch_settle_needs_publish()
    (src/llama-auto-ubatch.h) must therefore also return true for
    publish_dirty; tests/test-auto-ubatch-ladder.cpp executes it on that
    trace."""
    assert re.search(_SETTLE_NEEDS_PUBLISH_RE, _normalize_ws(strip_comments(LLAMA_AUTO_UBATCH_H))), (
        "llama_auto_ubatch_settle_needs_publish must return published_any || publish_dirty || "
        "n_ubatch != fallback_ubatch"
    )


def test_settle_needs_publish_dirty_term_has_a_mutation_witness():
    """Mutation witness for the check above: dropping the publish_dirty
    term from the helper must make it fail."""
    raw = LLAMA_AUTO_UBATCH_H
    line = "    return published_any || publish_dirty || n_ubatch != fallback_ubatch;\n"
    assert raw.count(line) == 1, f"mutation target not unique -- found {raw.count(line)}"
    mutated_raw = raw.replace(line, "    return published_any || n_ubatch != fallback_ubatch;\n", 1)
    assert not re.search(_SETTLE_NEEDS_PUBLISH_RE, _normalize_ws(strip_comments(mutated_raw))), (
        "mutation witness is broken: dropping publish_dirty should make the helper check fail"
    )


def test_settle_need_publish_dirty_argument_has_a_mutation_witness():
    """Mutation witness for test_settle_publish_is_gated_on_settle_needs_
    publish: passing `false` instead of publish_dirty at the settle must
    make the real check fail."""
    raw = LLAMA_CONTEXT_CPP
    call = "llama_auto_ubatch_settle_needs_publish(published_any, publish_dirty, cparams.n_ubatch, fallback_ubatch)"
    assert raw.count(call) == 1, f"mutation target not unique -- found {raw.count(call)}"
    mutated_raw = raw.replace(
        call, "llama_auto_ubatch_settle_needs_publish(published_any, false, cparams.n_ubatch, fallback_ubatch)", 1
    )
    mutated_body_norm = _body_of(mutated_raw, _TRIAL_START, _TRIAL_END)
    with pytest.raises(AssertionError, match="the settle block must compute need_publish"):
        _assert_settle_publish_gate(mutated_body_norm)


def test_warn_and_settle_have_a_mutation_witness():
    """Mutation witness: proves the WARN-count check would actually catch
    a second [SYCL-PLAN] auto n_ubatch= WARN being introduced."""
    raw = LLAMA_CONTEXT_CPP
    warn_line = (
        'LLAMA_LOG_WARN("[SYCL-PLAN] auto n_ubatch=%u for n_ctx=%u n_batch=%u (tried %s; %s); '
        'pass -ub N to override\\n",'
    )
    assert warn_line in raw, "mutation target not found -- update this witness to match the real source"
    duplicated = raw.replace(warn_line, warn_line + "\n    " + warn_line, 1)
    assert duplicated != raw

    mutated_body_norm = _body_of(duplicated, _TRIAL_START, _TRIAL_END)
    count = len(re.findall(r"\[SYCL-PLAN\] auto n_ubatch=", mutated_body_norm))
    assert count == 2, (
        f"mutation witness is broken: duplicating the WARN line should make the count-exactly-one check see 2, "
        f"saw {count}"
    )


# ---------------------------------------------------------------------------
# llama.cpp-7n6n (wires nphx Task 5): the persisted auto n_ubatch tuning
# cache -- a lookup tried BEFORE the ladder (a clean hit skips it entirely,
# with the new "cached" stop reason) and a store that follows any ladder run
# (never after a hit, which is already the persisted value). GGML_SYCL_
# TUNING_CACHE=0 disables both; the cache is advisory throughout, so every
# check here is about ORDERING and PRESENCE, never about changing the
# ladder's own pre-existing behaviour.
# ---------------------------------------------------------------------------


def test_header_declares_the_ubatch_cache_key_and_four_accessors():
    """ggml-sycl.h must declare the ggml_sycl_ubatch_cache_key struct and
    all four entry points the trial and ubatch-tuning-cache.cpp share --
    enabled/path (diagnostics) and lookup/store (the two calls the trial
    makes)."""
    assert re.search(r"struct\s+ggml_sycl_ubatch_cache_key\s*\{", GGML_SYCL_H_CODE), (
        "ggml_sycl_ubatch_cache_key must be declared in ggml-sycl.h"
    )
    for member in ("const\\s+int\\s*\\*\\s*devices\\s*;", "uint32_t\\s+n_devices\\s*;",
                   "const\\s+char\\s*\\*\\s*model_name\\s*;", "uint64_t\\s+model_size\\s*;",
                   "uint64_t\\s+model_hash\\s*;", "uint32_t\\s+n_ctx\\s*;", "uint32_t\\s+n_batch\\s*;",
                   "bool\\s+flash_attn\\s*;",
                   # quality round 1, Q4: the four fields added this round.
                   "uint32_t\\s+n_seq_max\\s*;", "int32_t\\s+type_k\\s*;", "int32_t\\s+type_v\\s*;",
                   # llama.cpp-3aos: kv_unified, added once KV sizing depended on it.
                   "bool\\s+kv_unified\\s*;",
                   # llama.cpp-uajm: swa_full, same reason (SWA layers sized as FULL under it).
                   "bool\\s+swa_full\\s*;"):
        assert re.search(member, GGML_SYCL_H_CODE), f"ggml_sycl_ubatch_cache_key is missing a member matching {member!r}"

    assert re.search(r"GGML_BACKEND_API\s+bool\s+ggml_backend_sycl_ubatch_cache_enabled\s*\(\s*void\s*\)\s*;",
                      GGML_SYCL_H_CODE), "ggml_backend_sycl_ubatch_cache_enabled(void) must be declared"
    assert re.search(
        r"GGML_BACKEND_API\s+bool\s+ggml_backend_sycl_ubatch_cache_path\s*\(\s*int\s+device\s*,\s*char\s*\*\s*buf\s*,"
        r"\s*size_t\s+buf_size\s*\)\s*;",
        GGML_SYCL_H_CODE,
    ), "ggml_backend_sycl_ubatch_cache_path(int, char*, size_t) must be declared"
    # quality round 1, Q2: lookup gained a reason_buf/reason_buf_size pair so
    # the caller can tell a TERMINAL cached outcome from a transient one.
    assert re.search(
        r"GGML_BACKEND_API\s+bool\s+ggml_backend_sycl_ubatch_cache_lookup_layout1\s*\(\s*const\s+struct\s+"
        r"ggml_sycl_ubatch_cache_key\s*\*\s*key\s*,\s*uint32_t\s*\*\s*n_ubatch\s*,\s*char\s*\*\s*reason_buf\s*,"
        r"\s*size_t\s+reason_buf_size\s*\)\s*;",
        GGML_SYCL_H_CODE,
    ), (
        "ggml_backend_sycl_ubatch_cache_lookup_layout1(const ggml_sycl_ubatch_cache_key*, uint32_t*, char*, size_t) must "
        "be declared"
    )
    assert re.search(
        r"GGML_BACKEND_API\s+bool\s+ggml_backend_sycl_ubatch_cache_store_layout1\s*\(\s*const\s+struct\s+"
        r"ggml_sycl_ubatch_cache_key\s*\*\s*key\s*,\s*uint32_t\s+n_ubatch\s*,\s*const\s+char\s*\*\s*reason\s*\)\s*;",
        GGML_SYCL_H_CODE,
    ), (
        "ggml_backend_sycl_ubatch_cache_store_layout1(const ggml_sycl_ubatch_cache_key*, uint32_t, const char*) must "
        "be declared"
    )


def test_ubatch_cache_key_layout_is_pinned_next_to_the_accessors():
    """The _layoutN suffix only protects a GGML_BACKEND_DL pair if a layout
    change forces a rename, so the backend pins the struct's size and every
    field's offset; ggml-sycl.h's comment points at that check."""
    code = _normalize_ws(UBATCH_TUNING_CACHE_CPP_CODE)
    assert re.search(r"static_assert\s*\([^;]*sizeof\s*\(\s*ggml_sycl_ubatch_cache_key\s*\)\s*==\s*72", code), (
        "ubatch-tuning-cache.cpp must static_assert sizeof(ggml_sycl_ubatch_cache_key)"
    )
    for field in ("devices", "n_devices", "model_name", "model_size", "model_hash", "n_ctx", "n_batch",
                  "flash_attn", "n_seq_max", "type_k", "type_v", "kv_unified", "swa_full"):
        assert re.search(rf"UBATCH_CACHE_KEY_FIELD_AT\s*\(\s*{field}\s*,\s*\d+\s*\)\s*;", code), (
            f"ubatch-tuning-cache.cpp must pin the offset of ggml_sycl_ubatch_cache_key::{field}"
        )
    assert re.search(
        r"#define UBATCH_CACHE_KEY_FIELD_AT\s*\(\s*field\s*,\s*offset\s*\)\s*\\?\s*static_assert\s*\([^;]*"
        r"offsetof\s*\(\s*ggml_sycl_ubatch_cache_key\s*,\s*field\s*\)\s*==\s*\(\s*offset\s*\)",
        UBATCH_TUNING_CACHE_CPP,
    ), "UBATCH_CACHE_KEY_FIELD_AT must static_assert offsetof(ggml_sycl_ubatch_cache_key, field) == offset"
    assert "ubatch-tuning-cache.cpp" in GGML_SYCL_H and "static_assert" in GGML_SYCL_H, (
        "ggml-sycl.h's comment must point at the layout check"
    )


def test_proc_address_registers_the_four_ubatch_cache_accessors():
    """A GGML_BACKEND_DL build's llama-context lookup needs all four
    entry points registered in the strcmp proc-address chain, mirroring
    the auto_ubatch_enabled/moe_gpu_ubatch_max precedent just above."""
    for symbol in (
        "ggml_backend_sycl_ubatch_cache_enabled",
        "ggml_backend_sycl_ubatch_cache_path",
        "ggml_backend_sycl_ubatch_cache_lookup_layout1",
        "ggml_backend_sycl_ubatch_cache_store_layout1",
    ):
        assert re.search(
            rf'strcmp\(\s*name\s*,\s*"{symbol}"\s*\)\s*==\s*0\s*\)\s*\{{\s*'
            rf"return\s*\(\s*void\s*\*\s*\)\s*{symbol}\s*;",
            GGML_SYCL_CPP_CODE,
        ), f"{symbol} must be registered in the proc-address strcmp chain"


def test_ubatch_cache_lookup_precedes_the_ladder():
    """The tuning-cache lookup must be tried BEFORE the ladder loop --
    the whole point of Task 5 is to skip the ladder on a clean hit."""
    body_norm = _normalize_ws(_trial_body())
    lookup_idx = body_norm.find("cache_lookup_fn(&cache_key,")
    loop_idx = body_norm.find("for (uint32_t c : rung_ladder)")
    assert lookup_idx != -1, "could not find the cache_lookup_fn(&cache_key, ...) call"
    assert loop_idx != -1
    assert lookup_idx < loop_idx, "the tuning-cache lookup must precede the ladder loop"


def test_ubatch_cache_store_follows_the_ladder():
    """The tuning-cache store must run AFTER the ladder loop has finished
    (and after the last_good == 0 fallback correction, so it never
    persists 0) -- never before it, and never inside it."""
    body_norm = _normalize_ws(_trial_body())
    loop_idx = body_norm.find("for (uint32_t c : rung_ladder)")
    fallback_fixup_idx = body_norm.find("if (last_good == 0) {")
    store_idx = body_norm.find("cache_store_fn(&cache_key,")
    assert loop_idx != -1 and fallback_fixup_idx != -1
    assert store_idx != -1, "could not find the cache_store_fn(&cache_key, ...) call"
    assert loop_idx < fallback_fixup_idx < store_idx, (
        "the tuning-cache store must run after both the ladder loop and the last_good == 0 fallback fixup"
    )


def test_cache_hit_gates_the_ladder_and_sets_the_cached_stop_reason():
    """A validated cache hit must set stop = "cached", flip ladder_needed
    to false, and the ladder loop's very first statement must check
    ladder_needed -- otherwise a hit would still (uselessly) walk the
    ladder's cap/probe/publish machinery for nothing."""
    body_norm = _normalize_ws(_trial_body())
    assert re.search(r'stop\s*=\s*"cached"\s*;', body_norm), 'a cache hit must set stop = "cached";'
    assert re.search(r"ladder_needed\s*=\s*false\s*;", body_norm), "a cache hit must set ladder_needed = false;"

    loop_idx = body_norm.find("for (uint32_t c : rung_ladder) {")
    assert loop_idx != -1
    after_loop_open = body_norm[loop_idx + len("for (uint32_t c : rung_ladder) {") :]
    assert re.match(r"\s*if\s*\(\s*!ladder_needed\s*\)\s*\{\s*break\s*;\s*\}", after_loop_open), (
        "the ladder loop's first statement must be `if (!ladder_needed) { break; }`, so a cache hit skips every "
        "rung without trying any of them"
    )


def test_ubatch_cache_disabled_by_env_var_zero():
    """GGML_SYCL_TUNING_CACHE=0 must disable both lookup and store --
    ubatch-tuning-cache.cpp's memoized accessor must return false for
    exactly "0", mirroring unified_cache_auto_ubatch_enabled()'s own
    convention (unified-cache.cpp)."""
    assert re.search(
        r'std::strcmp\(\s*env\s*,\s*"0"\s*\)\s*==\s*0\s*\)\s*\{\s*return\s+false\s*;\s*\}',
        UBATCH_TUNING_CACHE_CPP_CODE,
    ), 'GGML_SYCL_TUNING_CACHE="0" must return false (disabled) from the memoized accessor'
    assert re.search(
        r"ggml_backend_sycl_ubatch_cache_lookup_layout1[\s\S]{0,400}?ubatch_tuning_cache_env_enabled\s*\(\s*\)",
        UBATCH_TUNING_CACHE_CPP_CODE,
    ), "ggml_backend_sycl_ubatch_cache_lookup_layout1 must consult the enabled accessor before doing anything else"
    assert re.search(
        r"ggml_backend_sycl_ubatch_cache_store_layout1[\s\S]{0,400}?ubatch_tuning_cache_env_enabled\s*\(\s*\)",
        UBATCH_TUNING_CACHE_CPP_CODE,
    ), "ggml_backend_sycl_ubatch_cache_store_layout1 must consult the enabled accessor before doing anything else"


def test_ubatch_cache_lookup_and_store_have_mutation_witnesses():
    """Mutation witnesses for the two position checks above: prove they
    would actually catch the lookup/store being moved to the wrong side
    of the ladder loop."""
    raw = LLAMA_CONTEXT_CPP

    lookup_line = "    uint32_t   cached_ubatch         = 0;\n"
    assert lookup_line in raw, "mutation target not found -- update this witness to match the real source"
    loop_line = "    for (uint32_t c : rung_ladder) {\n"
    assert loop_line in raw, "mutation target not found -- update this witness to match the real source"

    # Swap the two markers' relative order by moving the loop's opening
    # line to just before the cache-availability check -- crude, but
    # sufficient to prove the position assertions would notice.
    mutated_raw = raw.replace(loop_line, "", 1).replace(lookup_line, loop_line + lookup_line, 1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _TRIAL_START, _TRIAL_END)
    mutated_lookup_idx = mutated_body_norm.find("cache_lookup_fn(&cache_key,")
    mutated_loop_idx = mutated_body_norm.find("for (uint32_t c : rung_ladder)")
    assert mutated_lookup_idx != -1 and mutated_loop_idx != -1
    assert not (mutated_lookup_idx < mutated_loop_idx), (
        "mutation witness is broken: moving the loop ahead of the cache check should make the lookup-precedes-"
        "the-ladder check fail"
    )


def test_ubatch_cache_store_follows_the_ladder_has_a_mutation_witness():
    """llama.cpp-7n6n: a SEPARATE mutation witness for
    the store-follows-the-ladder position check -- the combined witness
    above (test_ubatch_cache_lookup_and_store_have_mutation_witnesses) only
    exercises the lookup-precedes-the-ladder half; moving the loop earlier
    proves that half fails but says nothing about whether the STORE's own
    position check would catch the store being moved too early."""
    raw = LLAMA_CONTEXT_CPP
    # quality round 1, Q2/Q3: the whole store gate (not just the call) --
    # moved as one block, since its two preceding stop_is_pure_race/
    # resumed_outcome_unchanged declarations are not part of what this
    # witness is testing (this mutation is not meant to compile).
    store_block = (
        "    if (ladder_needed && !stop_is_pure_race && store_outcome && !resumed_outcome_unchanged && have_cache_accessors &&\n"
        "        cache_enabled_fn()) {\n"
        "        if (!cache_store_fn(&cache_key, last_good, stop)) {\n"
        '            LLAMA_LOG_WARN("[SYCL-PLAN] tuning cache store failed: %s\\n", cache_path_buf);\n'
        "        }\n"
        "    }\n"
    )
    assert store_block in raw, "mutation target not found -- update this witness to match the real source"
    loop_line = "    for (uint32_t c : rung_ladder) {\n"
    assert loop_line in raw, "mutation target not found -- update this witness to match the real source"

    # Move the store block to just BEFORE the ladder loop itself (not merely
    # before the last_good == 0 fixup, which is already after the loop and
    # so would not actually exercise the loop-vs-store ordering check).
    mutated_raw = raw.replace(store_block, "", 1).replace(loop_line, store_block + loop_line, 1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _TRIAL_START, _TRIAL_END)
    mutated_loop_idx = mutated_body_norm.find("for (uint32_t c : rung_ladder)")
    mutated_store_idx = mutated_body_norm.find("cache_store_fn(&cache_key,")
    assert mutated_loop_idx != -1 and mutated_store_idx != -1
    assert not (mutated_loop_idx < mutated_store_idx), (
        "mutation witness is broken: moving the store ahead of the ladder loop should make the "
        "store-follows-the-ladder check fail"
    )


def test_ubatch_cache_store_is_gated_on_ladder_needed():
    """llama.cpp-7n6n: the store must be gated on
    `ladder_needed` (never re-storing right after a TERMINAL cache hit
    skipped the ladder entirely), alongside the newer Q2/Q3 conditions
    (quality round 1: !stop_is_pure_race, !resumed_outcome_unchanged) and
    the pre-existing have_cache_accessors/cache_enabled_fn() gate."""
    body_norm = _normalize_ws(_trial_body())
    assert re.search(
        r"if\s*\(\s*ladder_needed\s*&&\s*!stop_is_pure_race\s*&&\s*store_outcome\s*&&\s*"
        r"!resumed_outcome_unchanged\s*&&\s*"
        r"have_cache_accessors\s*&&\s*cache_enabled_fn\s*\(\s*\)\s*\)\s*\{",
        body_norm,
    ), (
        "the store must be gated on ladder_needed && !stop_is_pure_race && store_outcome && "
        "!resumed_outcome_unchanged && "
        "have_cache_accessors && cache_enabled_fn()"
    )


def test_ubatch_cache_store_ladder_needed_gate_has_a_mutation_witness():
    """Mutation witness for the check above: proves it would actually
    catch the `ladder_needed &&` clause being dropped from the store's
    gate (which would re-store an identical entry after every TERMINAL
    cache hit, not just after a real ladder run)."""
    raw = LLAMA_CONTEXT_CPP
    guard_line = (
        "    if (ladder_needed && !stop_is_pure_race && store_outcome && !resumed_outcome_unchanged && have_cache_accessors &&\n"
    )
    assert guard_line in raw, "mutation target not found -- update this witness to match the real source"
    mutated_guard_line = (
        "    if (!stop_is_pure_race && store_outcome && !resumed_outcome_unchanged && have_cache_accessors &&\n"
    )
    mutated_raw = raw.replace(guard_line, mutated_guard_line, 1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _TRIAL_START, _TRIAL_END)
    assert not re.search(
        r"if\s*\(\s*ladder_needed\s*&&\s*!stop_is_pure_race\s*&&\s*store_outcome\s*&&\s*"
        r"!resumed_outcome_unchanged\s*&&\s*"
        r"have_cache_accessors\s*&&\s*cache_enabled_fn\s*\(\s*\)\s*\)\s*\{",
        mutated_body_norm,
    ), "mutation witness is broken: dropping ladder_needed from the guard should make the gate check fail"


# ---------------------------------------------------------------------------
# llama.cpp-7n6n quality round 1: Q2 (a stale hit could pin a value
# forever), Q3 (a silent store failure), Q4 (three key omissions).
# ---------------------------------------------------------------------------


def test_the_hoisted_block_takes_type_k_and_type_v():
    """quality round 1, Q4 / llama.cpp-7gno: type_k/type_v are constructor-local llama_context_params fields that the
    tuning-cache key needs, and the key is built by the hoisted block (a separate member function), so they are passed
    in as parameters and forwarded from the one call site. The ladder half reads the key from the prep and takes no
    parameters: a dead one would hide a second source of the key."""
    assert re.search(
        r"void\s+llama_context::sycl_auto_ubatch_prepare\s*\(\s*ggml_type\s+type_k\s*,\s*ggml_type\s+type_v\s*\)\s*\{",
        LLAMA_CONTEXT_CPP_CODE,
    ), "sycl_auto_ubatch_prepare must take (ggml_type type_k, ggml_type type_v)"
    assert re.search(
        r"sycl_auto_ubatch_prepare\s*\(\s*params\.type_k\s*,\s*params\.type_v\s*\)\s*;", LLAMA_CONTEXT_CPP_CODE
    ), "the call site must forward params.type_k, params.type_v"
    assert re.search(r"void\s+llama_context::sycl_select_auto_ubatch\s*\(\s*\)\s*\{", LLAMA_CONTEXT_CPP_CODE), (
        "sycl_select_auto_ubatch must take no parameters"
    )
    assert re.search(r"sycl_select_auto_ubatch\s*\(\s*\)\s*;", LLAMA_CONTEXT_CPP_CODE)
    select = LLAMA_CONTEXT_CPP_CODE[LLAMA_CONTEXT_CPP_CODE.index("void llama_context::sycl_select_auto_ubatch()"):]
    select = select[: select.index("static int llama_graph_n_input_tensors")]
    assert not re.search(r"\btype_[kv]\b", select), "the ladder half must not mention type_k/type_v"


def test_cache_key_populates_the_four_new_fields():
    """n_seq_max/type_k/type_v and the device list must all be assigned into
    cache_key -- declaring the struct fields (covered elsewhere) is not
    enough if nothing ever fills them in."""
    body_norm = _normalize_ws(_trial_body())
    for assignment in (
        r"cache_key\.n_seq_max\s*=\s*cparams\.n_seq_max\s*;",
        r"cache_key\.type_k\s*=\s*static_cast<int32_t>\s*\(\s*type_k\s*\)\s*;",
        r"cache_key\.type_v\s*=\s*static_cast<int32_t>\s*\(\s*type_v\s*\)\s*;",
        r"cache_key\.devices\s*=\s*cache_devices\.data\s*\(\s*\)\s*;",
        r"cache_key\.n_devices\s*=\s*static_cast<uint32_t>\s*\(\s*cache_devices\.size\s*\(\s*\)\s*\)\s*;",
    ):
        assert re.search(assignment, body_norm), f"missing cache_key field assignment matching {assignment!r}"


def test_cache_key_populates_kv_unified():
    """llama.cpp-3aos: cache_key.kv_unified must be assigned from
    cparams.kv_unified -- declaring the struct field (covered elsewhere) is
    not enough if nothing ever fills it in, and a stale (always-false)
    field would silently let two different-kv_unified contexts collide on
    one cache entry."""
    body_norm = _normalize_ws(_trial_body())
    assert re.search(r"cache_key\.kv_unified\s*=\s*cparams\.kv_unified\s*;", body_norm), (
        "missing cache_key.kv_unified = cparams.kv_unified; assignment"
    )


def test_cache_key_populates_swa_full():
    """llama.cpp-uajm: cache_key.swa_full must be assigned from
    cparams.swa_full -- same shape as kv_unified above: a CLI run
    (swa_full=false) and a raw-API context (llama_context_default_params()'s
    swa_full=true) size every SWA layer differently, so a stale
    (always-false) field would let the two collide on one cache entry."""
    body_norm = _normalize_ws(_trial_body())
    assert re.search(r"cache_key\.swa_full\s*=\s*cparams\.swa_full\s*;", body_norm), (
        "missing cache_key.swa_full = cparams.swa_full; assignment"
    )


def test_cache_devices_list_every_sycl_backend_in_order():
    """cache_devices must hold EVERY entry of sycl_backends, in order, and be
    filled before it is handed to cache_key -- a list of only the first
    backend would name one card for a scheduler-visible split."""
    body_norm = _normalize_ws(_trial_body())
    decl_idx = body_norm.find("std::vector<int> & cache_devices = prep->cache_devices;")
    assert decl_idx != -1, "could not find the cache_devices declaration"
    loop_idx = body_norm.find("for (auto & sb : sycl_backends)", decl_idx)
    assert loop_idx != -1, "cache_devices must be filled via a loop over sycl_backends"
    cache_key_idx = body_norm.find("cache_key.devices", loop_idx)
    assert cache_key_idx != -1, "cache_devices must be filled before it is assigned into cache_key"
    loop_body = body_norm[loop_idx:cache_key_idx]
    assert re.search(r"cache_devices\.push_back\s*\(\s*sb\.dev_index\s*\)\s*;", loop_body), (
        "the loop must append each backend's dev_index"
    )


def test_device_set_hash_is_gone():
    """The index-only device_set_hash hashed the scheduler-visible backends
    alone, which is [0] for both a collapsed level_zero:0,1 split and
    level_zero:0 alone. It must not come back beside the device list."""
    for name, code in (("ggml-sycl.h", GGML_SYCL_H_CODE), ("llama-context.cpp", LLAMA_CONTEXT_CPP_CODE),
                       ("ubatch-tuning-cache.cpp", UBATCH_TUNING_CACHE_CPP_CODE)):
        assert "device_set_hash" not in code, f"{name} still carries device_set_hash"


def test_backend_extends_the_device_set_with_hidden_planner_gpus():
    """ubatch-tuning-cache.cpp must compose the key over the PARTICIPATING
    set: the context's devices plus, under the multi-device plan, every other
    physical GPU, gated on the same ggml_backend_sycl_moe_multi_gpu_requested()
    the multi-device plan uses, with each device's budget from its budget
    authority."""
    code = _normalize_ws(UBATCH_TUNING_CACHE_CPP_CODE)
    for pattern, what in (
        (r"topo\.total_gpu_count\s*=\s*std::min\s*\(\s*info\.total_gpu_count\s*,", "the physical GPU count"),
        (r"topo\.multi_device_plan\s*=\s*ggml_backend_sycl_moe_multi_gpu_requested\s*\(\s*\)\s*;",
         "the planner's own multi-GPU gate"),
        (r"for\s*\(\s*int\s+device\s*:\s*ubatch_participating_devices\s*\(\s*topo\s*\)\s*\)",
         "a walk over the participating devices"),
        (r"ggml_sycl::ggml_sycl_device_budget_authority_existing\s*\(",
         "each device's budget authority, read without constructing a cache"),
        (r"id\.budget_pct\s*=\s*budget\.budget_pct\s*;", "the budget percentage"),
        (r"id\.external_headroom\s*=\s*budget\.external_headroom\s*;", "the external headroom"),
        (r"out_device_key\s*=\s*ubatch_device_set_key\s*\(\s*topo\s*,\s*identities\s*\)\s*;",
         "the composed device-set key"),
    ):
        assert re.search(pattern, code), f"resolve_device_set_key() must use {what} ({pattern!r})"
    # A key read must never create a cache for a hidden GPU as a side effect.
    assert not re.search(r"ggml_sycl::ggml_sycl_device_budget_authority\s*\(", code), (
        "resolve_device_set_key() must use the non-creating budget authority"
    )
    # The split itself is keyed: mode and ratio change each card's share.
    for env in ("GGML_SYCL_MULTI_GPU_MODE", "GGML_SYCL_SPLIT_RATIO", "GGML_SYCL_TENSOR_SPLIT"):
        assert f'"{env}"' in code, f"the placement config in the key must include {env}"
    assert re.search(
        r"topo\.placement_config\s*=\s*ubatch_placement_config\s*\(\s*placement_env\s*,", code
    ), "the placement config must be composed into topo by ubatch_placement_config()"
    # lookup and store both key through it; the path accessor needs only the file name.
    assert len(re.findall(r"resolve_device_set_key\s*\(\s*\*\s*key\s*,", code)) == 2, (
        "both lookup and store must resolve their key through resolve_device_set_key()"
    )
    # The gate above is only right while the planner uses the same one.
    _assert_planner_branches_on_the_gate(_normalize_ws(GGML_SYCL_CPP_CODE))


def _brace_block(code: str, open_idx: int) -> str:
    depth = 0
    for i in range(open_idx, len(code)):
        depth += {"{": 1, "}": -1}.get(code[i], 0)
        if depth == 0:
            return code[open_idx + 1:i]
    raise AssertionError("unbalanced braces")


_PLANNER_FN = "static void compute_and_store_plan_for_inventory("
_PLANNER_GATE = "if (info.total_gpu_count >= 2 && ggml_backend_sycl_moe_multi_gpu_requested()) {"


def _assert_planner_branches_on_the_gate(code: str) -> None:
    """compute_and_store_plan_for_inventory() must reach
    compute_multi_device_plan() only inside the branch taken on
    ggml_backend_sycl_moe_multi_gpu_requested() -- the predicate the key
    reads to decide |plan=multi."""
    fn = code.find(_PLANNER_FN)
    assert fn != -1, "could not find compute_and_store_plan_for_inventory()"
    fn_body = _brace_block(code, code.index("{", fn))
    gate = fn_body.find(_PLANNER_GATE)
    assert gate != -1, (
        "compute_and_store_plan_for_inventory() must branch on "
        "`info.total_gpu_count >= 2 && ggml_backend_sycl_moe_multi_gpu_requested()`"
    )
    gate_block = _brace_block(fn_body, gate + len(_PLANNER_GATE) - 1)
    assert "compute_multi_device_plan(" in gate_block, (
        "compute_multi_device_plan() must be called inside the multi-device gate's branch"
    )
    assert fn_body.count("compute_multi_device_plan(") == gate_block.count("compute_multi_device_plan("), (
        "compute_multi_device_plan() must be reachable only inside the multi-device gate's branch"
    )


def test_planner_gate_check_has_a_mutation_witness():
    """Mutation witness for the planner-gate check: dropping the
    ggml_backend_sycl_moe_multi_gpu_requested() call from the branch
    condition must fail it, since the key would then describe a plan the
    planner no longer gates the same way."""
    raw = GGML_SYCL_CPP
    fn = raw.find(_PLANNER_FN)
    assert fn != -1, "mutation target not found -- update this witness to match the real source"
    target = "    if (info.total_gpu_count >= 2 && ggml_backend_sycl_moe_multi_gpu_requested()) {\n"
    at = raw.find(target, fn)
    assert at != -1, "mutation target not found -- update this witness to match the real source"
    mutated_raw = raw[:at] + "    if (info.total_gpu_count >= 2) {\n" + raw[at + len(target):]
    with pytest.raises(AssertionError, match="must branch on"):
        _assert_planner_branches_on_the_gate(_normalize_ws(strip_comments(mutated_raw)))


def test_planner_gate_check_else_branch_has_a_mutation_witness():
    """Mutation witness for the planner-gate check: a
    compute_multi_device_plan() call in the single-device else branch,
    AFTER the gate's block, must fail it too."""
    code = GGML_SYCL_CPP_CODE
    fn = code.find(_PLANNER_FN)
    assert fn != -1, "mutation target not found -- update this witness to match the real source"
    target = "plan_candidate = ggml_sycl::compute_placement_plan("
    at = code.find(target, fn)
    assert at != -1, "mutation target not found -- update this witness to match the real source"
    gate_at = code.find(_PLANNER_GATE, fn)
    assert gate_at != -1, "mutation target not found -- update this witness to match the real source"
    gate_open = gate_at + len(_PLANNER_GATE) - 1
    gate_end = gate_open + 1 + len(_brace_block(code, gate_open))
    assert at > gate_end, "the else-branch target must come after the gate's block"
    mutated = code[:at] + "plan_candidate = ggml_sycl::compute_multi_device_plan(" + code[at + len(target):]
    with pytest.raises(AssertionError, match="reachable only inside"):
        _assert_planner_branches_on_the_gate(_normalize_ws(mutated))


def test_hidden_gpu_gate_holds_for_a_dense_model():
    """The hidden-GPU gate is named for MoE but is not MoE-specific: with
    GGML_SYCL_MOE_MULTI_GPU unset it is just total_gpu_count >= 2, which is
    why a dense Mistral level_zero:0,1 run takes the multi-device plan and
    places a layer block on the B50. If it ever starts consulting the model
    (an expert count, is_moe), a dense split would silently key as its
    first card alone again."""
    _assert_hidden_gpu_gate_model_independent(_normalize_ws(GGML_SYCL_CPP_CODE))


def _hidden_gpu_gate_body(code: str) -> str:
    start = code.find("bool ggml_backend_sycl_moe_multi_gpu_requested() {")
    assert start != -1, "could not find ggml_backend_sycl_moe_multi_gpu_requested()'s definition"
    open_idx = code.index("{", start)
    depth, end = 0, -1
    for i in range(open_idx, len(code)):
        depth += {"{": 1, "}": -1}.get(code[i], 0)
        if depth == 0:
            end = i
            break
    return code[open_idx + 1:end].strip()


# The whole gate, whitespace-normalized: the GGML_SYCL_MOE_MULTI_GPU override,
# then total_gpu_count >= 2 alone. An exact match rather than a forbid-list of
# model terms, so ANY new input -- an expert count, g_model_n_layer, a global
# nobody has named yet -- fails it.
_HIDDEN_GPU_GATE_BODY = _normalize_ws(
    'const char * env = std::getenv("GGML_SYCL_MOE_MULTI_GPU"); '
    "if (env) { return std::atoi(env) != 0 && ggml_sycl_info().total_gpu_count >= 2; } "
    "return ggml_sycl_info().total_gpu_count >= 2;"
)


def _assert_hidden_gpu_gate_model_independent(code: str) -> None:
    body = _hidden_gpu_gate_body(code)
    assert body == _HIDDEN_GPU_GATE_BODY, (
        "the hidden-GPU gate must be exactly the env override plus total_gpu_count >= 2 -- it must not depend "
        f"on the model; got: {body!r}"
    )


_HIDDEN_GPU_GATE_RETURN = "\n    return ggml_sycl_info().total_gpu_count >= 2;\n}"


def _hidden_gpu_gate_mutant(insert_before_return: str = "", replace_return: str = "") -> str:
    """Mutate the gate's final return, located from the function's own
    signature so the anchor does not depend on what follows it."""
    raw = GGML_SYCL_CPP
    fn = raw.find("bool ggml_backend_sycl_moe_multi_gpu_requested() {")
    assert fn != -1, "mutation target not found -- update this witness to match the real source"
    ret = raw.find(_HIDDEN_GPU_GATE_RETURN, fn)
    assert ret != -1, "mutation target not found -- update this witness to match the real source"
    original = _HIDDEN_GPU_GATE_RETURN
    replacement = insert_before_return + (replace_return or original)
    mutated_raw = raw[:ret] + replacement + raw[ret + len(original):]
    assert mutated_raw != raw
    return _normalize_ws(strip_comments(mutated_raw))


def test_hidden_gpu_gate_holds_for_a_dense_model_has_a_mutation_witness():
    """Mutation witness for the check above: an early return on the expert
    count leaves the trailing `return ... >= 2;` intact -- the change that
    would make a dense level_zero:0,1 split key as its first card alone."""
    mutated = _hidden_gpu_gate_mutant(
        insert_before_return="\n    if (g_moe_n_experts_total == 0) {\n        return false;\n    }"
    )
    with pytest.raises(AssertionError, match="must not depend on the model"):
        _assert_hidden_gpu_gate_model_independent(mutated)


def test_hidden_gpu_gate_unnamed_model_input_has_a_mutation_witness():
    """Mutation witness that the exact-shape check is not a forbid-list: a
    model input no list names (g_model_n_layer) folded into the return is
    caught too."""
    mutated = _hidden_gpu_gate_mutant(
        replace_return="\n    return ggml_sycl_info().total_gpu_count >= 2 && g_model_n_layer > 0;\n}"
    )
    with pytest.raises(AssertionError, match="must not depend on the model"):
        _assert_hidden_gpu_gate_model_independent(mutated)


def test_cache_hit_reads_the_stored_reason():
    """quality round 1, Q2: the lookup must read back the REASON the entry
    was stored with (not just n_ubatch) -- that reason is what tells a
    TERMINAL hit from one that should resume the ladder."""
    body_norm = _normalize_ws(_trial_body())
    assert re.search(
        r"cache_lookup_fn\s*\(\s*&cache_key\s*,\s*&cached_ubatch\s*,\s*cached_reason_buf\s*,\s*"
        r"sizeof\s*\(\s*cached_reason_buf\s*\)\s*\)", body_norm
    ), "the lookup call must pass cached_reason_buf/sizeof(cached_reason_buf) alongside &cached_ubatch"


def test_terminal_reasons_are_exactly_two_strings():
    """quality round 1, Q2(c): the TERMINAL reason set -- the ones that
    still skip the ladder outright on a hit -- must be EXACTLY "ladder
    exhausted" and "MoE GPU routing ceiling", no more and no fewer. Any
    OTHER stored reason must resume the ladder instead of trusting the
    cached value forever."""
    body_norm = _normalize_ws(_trial_body())
    terminal_idx = body_norm.find("const bool terminal =")
    assert terminal_idx != -1, "could not find the `terminal` reason check"
    # Bound to the statement itself (up to its terminating `;`) so this
    # cannot accidentally pick up an unrelated later `strcmp` call.
    terminal_stmt_end = body_norm.find(";", terminal_idx)
    assert terminal_stmt_end != -1
    terminal_stmt = body_norm[terminal_idx : terminal_stmt_end + 1]
    found = set(re.findall(r'std::strcmp\s*\(\s*cached_reason_buf\s*,\s*"([^"]*)"\s*\)\s*==\s*0', terminal_stmt))
    assert found == {"ladder exhausted", "MoE GPU routing ceiling"}, (
        f"the terminal-reason set must be exactly {{'ladder exhausted', 'MoE GPU routing ceiling'}} -- found {found}"
    )


def test_terminal_reasons_have_a_mutation_witness():
    """Mutation witness for the check above: proves it would actually
    catch a third reason being silently added to the terminal set (which
    would wrongly let that reason skip the ladder on a hit, the exact
    sticky behaviour this finding fixed)."""
    raw = LLAMA_CONTEXT_CPP
    terminal_stmt = (
        '                const bool terminal = std::strcmp(cached_reason_buf, "ladder exhausted") == 0 ||\n'
        '                                      std::strcmp(cached_reason_buf, "MoE GPU routing ceiling") == 0;\n'
    )
    assert terminal_stmt in raw, "mutation target not found -- update this witness to match the real source"
    mutated_stmt = terminal_stmt.replace(
        '== 0;\n', '== 0 || std::strcmp(cached_reason_buf, "transaction refused") == 0;\n', 1
    )
    mutated_raw = raw.replace(terminal_stmt, mutated_stmt, 1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _TRIAL_START, _TRIAL_END)
    terminal_idx = mutated_body_norm.find("const bool terminal =")
    assert terminal_idx != -1
    terminal_stmt_end = mutated_body_norm.find(";", terminal_idx)
    mutated_terminal_stmt = mutated_body_norm[terminal_idx : terminal_stmt_end + 1]
    found = set(re.findall(r'std::strcmp\s*\(\s*cached_reason_buf\s*,\s*"([^"]*)"\s*\)\s*==\s*0', mutated_terminal_stmt))
    assert found != {"ladder exhausted", "MoE GPU routing ceiling"}, (
        "mutation witness is broken: adding a third reason should make the exact-set check fail"
    )


def test_non_terminal_hit_resumes_the_ladder_above_the_cached_value():
    """quality round 1, Q2: a non-terminal hit must set cache_resume_above
    to the cached (already-validated) value BEFORE the ladder loop, and the
    loop must skip every rung at or below it -- otherwise the ladder would
    needlessly re-try a value already known to pass."""
    body_norm = _normalize_ws(_trial_body())
    resume_assign_idx = body_norm.find("cache_resume_above = cached_ubatch;")
    loop_idx = body_norm.find("for (uint32_t c : rung_ladder)")
    assert resume_assign_idx != -1, "could not find `cache_resume_above = cached_ubatch;`"
    assert loop_idx != -1
    assert resume_assign_idx < loop_idx, "cache_resume_above must be set before the ladder loop runs"

    skip_idx = body_norm.find("if (c <= cache_resume_above) { continue; }", loop_idx)
    assert skip_idx != -1, "the ladder loop must skip rungs at or below cache_resume_above"


def test_resume_skip_has_a_mutation_witness():
    """Mutation witness for the check above: proves it would actually
    catch the resume-skip being deleted (which would make the ladder
    re-try the already-validated cached rung, harmless but wasteful, and --
    more importantly -- proves the position/presence check is load-bearing
    rather than vacuous)."""
    raw = LLAMA_CONTEXT_CPP
    skip_block = (
        "        if (c <= cache_resume_above) {\n"
        "            continue;\n"
        "        }\n"
    )
    assert skip_block in raw, "mutation target not found -- update this witness to match the real source"
    mutated_raw = raw.replace(skip_block, "", 1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _TRIAL_START, _TRIAL_END)
    assert "if (c <= cache_resume_above)" not in mutated_body_norm, (
        "mutation witness is broken: deleting the block should remove the resume-skip check"
    )


def test_store_writes_the_real_stop_reason_not_a_literal():
    """quality round 1, Q2(a): the store must persist `stop` (the trial's
    OWN actual outcome) -- not a fixed "ladder" literal, which is what let
    a lost cache revalidation's reason go unrecorded in the first place."""
    body_norm = _normalize_ws(_trial_body())
    assert re.search(r"cache_store_fn\s*\(\s*&cache_key\s*,\s*last_good\s*,\s*stop\s*\)", body_norm), (
        'the store must be called as cache_store_fn(&cache_key, last_good, stop) -- not a "ladder" literal'
    )
    assert '"ladder"' not in body_norm, 'the literal "ladder" must not appear anywhere in the trial body any more'


def test_store_call_has_a_mutation_witness():
    """Mutation witness for the check above: proves it would actually
    catch the store reverting to a fixed "ladder" literal."""
    raw = LLAMA_CONTEXT_CPP
    call = "cache_store_fn(&cache_key, last_good, stop)"
    assert call in raw, "mutation target not found -- update this witness to match the real source"
    mutated_raw = raw.replace(call, 'cache_store_fn(&cache_key, last_good, "ladder")', 1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _TRIAL_START, _TRIAL_END)
    assert not re.search(r"cache_store_fn\s*\(\s*&cache_key\s*,\s*last_good\s*,\s*stop\s*\)", mutated_body_norm), (
        "mutation witness is broken: reverting to a literal should make the variable-reason check fail"
    )


def test_store_skips_exactly_the_two_pure_race_reasons():
    """quality round 1, Q2(b): the store must be skipped for EXACTLY
    "transaction busy" and "not the published model" (pure races) -- no
    more, no fewer. Persisting either as if it were a real shape limit
    would reintroduce a sticky-hit bug one level up."""
    body_norm = _normalize_ws(_trial_body())
    race_idx = body_norm.find("const bool stop_is_pure_race =")
    assert race_idx != -1, "could not find the stop_is_pure_race computation"
    race_stmt_end = body_norm.find(";", race_idx)
    assert race_stmt_end != -1
    race_stmt = body_norm[race_idx : race_stmt_end + 1]
    found = set(re.findall(r'std::strcmp\s*\(\s*stop\s*,\s*"([^"]*)"\s*\)\s*==\s*0', race_stmt))
    assert found == {"transaction busy", "not the published model"}, (
        f"stop_is_pure_race must check exactly {{'transaction busy', 'not the published model'}} -- found {found}"
    )


def test_store_pure_race_skip_has_a_mutation_witness():
    """Mutation witness for the check above: proves it would actually
    catch a third reason being silently added to the pure-race set."""
    raw = LLAMA_CONTEXT_CPP
    race_stmt = (
        "    const bool stop_is_pure_race =\n"
        '        std::strcmp(stop, "transaction busy") == 0 || std::strcmp(stop, "not the published model") == 0;\n'
    )
    assert race_stmt in raw, "mutation target not found -- update this witness to match the real source"
    mutated_stmt = race_stmt.replace(
        "== 0;\n", '== 0 || std::strcmp(stop, "KV would be demoted") == 0;\n', 1
    )
    mutated_raw = raw.replace(race_stmt, mutated_stmt, 1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _TRIAL_START, _TRIAL_END)
    race_idx = mutated_body_norm.find("const bool stop_is_pure_race =")
    assert race_idx != -1
    race_stmt_end = mutated_body_norm.find(";", race_idx)
    mutated_race_stmt = mutated_body_norm[race_idx : race_stmt_end + 1]
    found = set(re.findall(r'std::strcmp\s*\(\s*stop\s*,\s*"([^"]*)"\s*\)\s*==\s*0', mutated_race_stmt))
    assert found != {"transaction busy", "not the published model"}, (
        "mutation witness is broken: adding a third reason should make the exact-set check fail"
    )


def test_store_skips_an_unchanged_resumed_outcome():
    """quality round 1, Q2(c): a RESUMED hit whose ladder run reproduced
    the exact same (last_good, reason) it started from must not re-store
    -- an identical outcome is not worth a rewrite."""
    body_norm = _normalize_ws(_trial_body())
    assert re.search(
        r"resumed_outcome_unchanged\s*=\s*cache_resumed\s*&&\s*last_good\s*==\s*cache_resume_ubatch\s*&&\s*"
        r"cache_resume_reason\s*==\s*stop\s*;",
        body_norm,
    ), "resumed_outcome_unchanged must compare last_good and the reason against the resumed-from hit"
    store_gate_idx = body_norm.find(
        "if (ladder_needed && !stop_is_pure_race && store_outcome && !resumed_outcome_unchanged"
    )
    assert store_gate_idx != -1, "the store gate must check !resumed_outcome_unchanged"


def test_store_failure_logs_exactly_one_warn():
    """quality round 1, Q3: a failed store must log exactly one
    "[SYCL-PLAN] tuning cache store failed: ..." WARN naming the path --
    the store's own return value used to be discarded silently."""
    body_norm = _normalize_ws(_trial_body())
    assert re.search(
        r'if\s*\(\s*!cache_store_fn\s*\(\s*&cache_key\s*,\s*last_good\s*,\s*stop\s*\)\s*\)\s*\{\s*'
        r'LLAMA_LOG_WARN\s*\(\s*"\[SYCL-PLAN\] tuning cache store failed: %s\\n"\s*,\s*cache_path_buf\s*\)\s*;\s*\}',
        body_norm,
    ), "a failed store must log exactly one [SYCL-PLAN] tuning cache store failed: %s WARN naming cache_path_buf"


def test_store_failure_warn_has_a_mutation_witness():
    """Mutation witness for the check above: proves it would actually
    catch the store-failure WARN being deleted (a read-only HOME would then
    silently re-run the ladder on every start with no diagnostic)."""
    raw = LLAMA_CONTEXT_CPP
    block = (
        "        if (!cache_store_fn(&cache_key, last_good, stop)) {\n"
        '            LLAMA_LOG_WARN("[SYCL-PLAN] tuning cache store failed: %s\\n", cache_path_buf);\n'
        "        }\n"
    )
    assert block in raw, "mutation target not found -- update this witness to match the real source"
    mutated_raw = raw.replace(block, "        cache_store_fn(&cache_key, last_good, stop);\n", 1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _TRIAL_START, _TRIAL_END)
    assert "tuning cache store failed" not in mutated_body_norm, (
        "mutation witness is broken: deleting the block should remove the store-failure WARN"
    )


def test_at_most_four_llama_log_warn_call_sites_in_the_body():
    """quality round 1, Q3: this function's own docstring says AT MOST FOUR
    GGML_LOG_WARN lines report the outcome (the tuning-cache lookup
    outcome, an optional store-failure WARN, llama.cpp-kpjw's "lowered
    from" WARN, and the pre-existing auto n_ubatch outcome) -- pin the
    literal call-site count in the source, not just the docstring's prose."""
    body_norm = _normalize_ws(_trial_body())
    count = len(re.findall(r"LLAMA_LOG_WARN\(", body_norm))
    assert count == 4, f"expected exactly 4 LLAMA_LOG_WARN( call sites in the trial body -- found {count}"


def test_cache_path_return_is_checked_and_substituted():
    """quality round 1, Q9: ggml_backend_sycl_ubatch_cache_path()'s return
    must be checked -- a call that fails (an out-of-range device, or a
    path too long for the buffer) must not leave a caller trusting a
    silently-truncated or stale path."""
    body_norm = _normalize_ws(_trial_body())
    assert re.search(
        r"if\s*\(\s*have_cache_accessors\s*&&\s*!cache_path_fn\s*\(\s*cache_devices\.front\s*\(\s*\)\s*,\s*"
        r"cache_path_buf\s*,\s*"
        r"sizeof\s*\(\s*cache_path_buf\s*\)\s*\)\s*\)\s*\{", body_norm
    ), "the cache_path_fn(...) call must be negated and checked"
    assert '"(cache path unavailable)"' in body_norm, (
        'a failed cache_path_fn(...) call must substitute "(cache path unavailable)"'
    )


def test_ubatch_cache_path_source_returns_false_on_oversized_path():
    """quality round 1, Q9: ggml_backend_sycl_ubatch_cache_path() itself
    (ubatch-tuning-cache.cpp) must refuse rather than silently truncate
    when the resolved path does not fit in the caller's buffer."""
    assert re.search(
        r"if\s*\(\s*path\.size\s*\(\s*\)\s*>=\s*buf_size\s*\)\s*\{\s*return\s+false\s*;\s*\}",
        _normalize_ws(UBATCH_TUNING_CACHE_CPP_CODE),
    ), "ggml_backend_sycl_ubatch_cache_path must return false when path.size() >= buf_size"


def test_boundary_literals_are_pinned_in_the_unit_test():
    """llama.cpp-7n6n round 2: a live GPU run found a Mistral model_hash of
    12629460749384247297 (20 digits) silently mismatching after a save/load
    round trip, because parse_u64()'s old fixed 19-digit cap truncated it on
    read. Pin the two boundary literals -- the exact value observed live,
    and UINT64_MAX itself -- so a future edit cannot quietly narrow
    parse_u64_rejects_overflow / ubatch_cache_u64_hash_boundary_roundtrip
    (tests/test-tuning-cache-io.cpp) back down to values that never exercise
    the 20-digit boundary."""
    text = TEST_TUNING_CACHE_IO_CPP
    assert "12629460749384247297" in text, (
        "the exact model_hash observed on live GPU hardware must stay covered by a unit test"
    )
    assert "18446744073709551615" in text, "UINT64_MAX itself must stay covered by a unit test"
    assert "parse_u64_rejects_overflow" in text
    assert "ubatch_cache_u64_hash_boundary_roundtrip" in text


# ---------------------------------------------------------------------------
# llama.cpp-pyu4: the GGML_SYCL_AUTO_UBATCH doc row must explain that
# every MoE model is pinned at 512 by the GPU MoE routing ceiling.
# ---------------------------------------------------------------------------


def test_env_vars_doc_explains_the_moe_512_pin():
    """llama.cpp-pyu4: the GGML_SYCL_AUTO_UBATCH row must state that every MoE model
    is pinned at 512 today by the GPU MoE routing ceiling (llama.cpp-ohkx),
    and name why GPT-OSS (MoE) and Mistral (dense) report different
    outcomes -- otherwise a reader sees "MoE GPU routing ceiling" reported
    for every MoE model and has no explanation for why it is always 512."""
    row_start = SYCL_ENV_VARS_MD.find("| `GGML_SYCL_AUTO_UBATCH=0` |")
    assert row_start != -1, "could not find the GGML_SYCL_AUTO_UBATCH row"
    row_end = SYCL_ENV_VARS_MD.find("\n", row_start)
    assert row_end != -1
    row = SYCL_ENV_VARS_MD[row_start:row_end]

    assert "llama.cpp-ohkx" in row, "the row must cite llama.cpp-ohkx for the GPU MoE routing ceiling"
    assert "every MoE model is pinned at 512" in row, (
        "the row must state that every MoE model is pinned at 512 by the ceiling"
    )
    assert "GPT-OSS" in row and "Mistral" in row, (
        "the row must name GPT-OSS (MoE, pinned) and Mistral (dense, not pinned) as the contrasting example"
    )


# ---------------------------------------------------------------------------
# llama.cpp-kpjw (kpjw-g7, one fact one source): the trial has no settle descent, a refused settle is a named error
# whose -ub comes from the one hold-spill fit function, the downward continuation runs only after a real fit refusal,
# a refused cached value is not paid for twice, and a scheduler compute buffer is identified by an explicit scope.
# ---------------------------------------------------------------------------


def _trial_norm() -> str:
    return _normalize_ws(_trial_body())


def _try_candidate_norm() -> str:
    return _normalize_ws(_try_candidate_body())


def _ws_tolerant(old: str) -> "re.Pattern":
    """`old` as a pattern that ignores every whitespace difference, so a clang-format run cannot make a mutation
    target stop matching (the mutant is then a vacuous 'target not found' failure, or worse a silent skip)."""
    return re.compile(r"\s*".join(re.escape(ch) for ch in old if not ch.isspace()))


def _trial_mutant(old: str, new: str) -> str:
    """The normalized trial body of the source with `old` (raw text, whitespace-insensitive, once) replaced by `new`."""
    pat = _ws_tolerant(old)
    found = len(pat.findall(LLAMA_CONTEXT_CPP))
    assert found == 1, f"mutation target not unique -- found {found}: {old!r}"
    return _body_of(pat.sub(lambda _m: new, LLAMA_CONTEXT_CPP, count=1), _TRIAL_START, _TRIAL_END)


def _settle_refused_branch(body_norm: str) -> str:
    settle = body_norm[body_norm.find("if (!sched_matches_last_good") :]
    at = settle.find("if (settle_error) {")
    assert at != -1, "the settle's refusal branch is missing"
    return _balanced_braces(settle, at + len("if (settle_error) "))


def test_there_is_no_settle_refusal_descent():
    """A settle refusal is a named error, not a second walk down the ladder: the helper, its call and the `won` it
    adopted are gone from the header and the trial."""
    assert "settle_refusal_descend" not in LLAMA_AUTO_UBATCH_H
    body = _trial_norm()
    assert "settle_refusal_descend" not in body
    refused = _settle_refused_branch(body)
    assert "try_candidate(" not in refused and "won" not in refused, "the refusal branch must not try rungs"


def _settle_refusal_names_the_fit_function(body_norm: str) -> bool:
    refused = _settle_refused_branch(body_norm)
    fits_at = refused.find("llama_context_sycl_hold_spill_fits(backends, last_good, &largest_ub)")
    rethrow_at = refused.find("std::rethrow_exception(settle_error);")
    advice_at = refused.find("llama_auto_ubatch_advice(largest_ub, lowest_refused)")
    throw_at = refused.find("throw std::runtime_error(")
    return (
        -1 not in (fits_at, rethrow_at, advice_at, throw_at)
        and fits_at < rethrow_at < advice_at < throw_at
        and refused.count("throw std::runtime_error(") == 1
        and "refusal_largest_ub" not in body_norm
        and "last_stop" in refused[throw_at:]
    )


def test_a_refused_settle_is_a_named_error_with_the_fit_functions_n():
    """The settle publish's refusal asks the ONE fit function (the entry the realized check uses): when it accepts the
    rung the refusal is no fit refusal (a race) and leaves as it came; when it refuses, the context fails by name with
    the -ub that function accepts, capped under every rung this start already lost, and the LAST rung's stop reason."""
    assert _settle_refusal_names_the_fit_function(_trial_norm())


@pytest.mark.parametrize(
    "old,new",
    [
        ("            if (llama_context_sycl_hold_spill_fits(backends, last_good, &largest_ub)) {\n",
         "            if (false) {\n"),
        ("llama_auto_ubatch_advice(largest_ub, lowest_refused)", "largest_ub"),
        ("            throw std::runtime_error(\n                format(\"auto n_ubatch: %s (tried",
         "            (void) (\n                format(\"auto n_ubatch: %s (tried"),
    ],
)
def test_settle_named_error_has_a_mutation_witness(old, new):
    assert not _settle_refusal_names_the_fit_function(_trial_mutant(old, new)), "the mutant must make the pin fail"


def test_the_settle_publish_catch_only_records():
    body = _trial_norm()
    settle = body[body.find("if (!sched_matches_last_good") :]
    catch_at = settle.find("} catch (const std::exception & e) {")
    assert catch_at != -1 and settle.count("catch") == 1
    catch_body = _balanced_braces(settle, catch_at + len("} catch (const std::exception & e) "))
    assert catch_body[1:-1].strip() == "settle_error = std::current_exception(); settle_refusal = e.what();"
    refused = _settle_refused_branch(body)
    assert "return" not in refused, "a refused settle must throw, never return"
    tail = settle[settle.find("if (settle_error) {") :]
    assert "} else { sched_need_reserve = true; sched_reserve(); }" in tail


_DESCENT_GATE = "if (last_good == 0 && ladder_needed && fallback_tried && rung_fit_refused) {"


def test_the_descent_runs_only_after_a_real_fit_refusal():
    """`stop_is_pure_race` was the only thing keeping a probe/publish anomaly (a lifecycle failure, a CAS race, a
    demoted KV, a host fallback) from lowering -ub. The descent now needs the rung that ended the ladder to have been a
    fit refusal: the probe's own refusal, a compute buffer that did not fit, or the hold spill."""
    body = _trial_norm()
    assert _DESCENT_GATE in body
    assert "stop_is_pure_race" in body  # still the store gate's race term


def test_the_descent_stops_at_a_loss_that_is_no_fit_refusal():
    body = _trial_norm()
    at = body.find(_DESCENT_GATE)
    block = _balanced_braces(body, at + len(_DESCENT_GATE) - 1)
    assert "descent_ran = true;" in block
    assert "publish_dirty = true;" in block, "a below-default rung ran: the ring must be republished by the settle"
    assert re.search(r"descent_aborted\s*=\s*!rung_fit_refused", block), "the walk must stop at a non-fit loss"
    assert "note_loss(c, rung_reason);" in block
    assert re.search(r"if \(ended != 0 && !descent_aborted\)", block)


def _fit_flag_is_set_only_on_fit_refusals(tc: str) -> bool:
    if not re.search(r"\[&\]\(uint32_t c\) -> const char \* \{ rung_fit_refused = false;", tc):
        return False
    fit = [
        'rung_fit_refused = true; return "transaction refused";',
        'rung_fit_refused = true; return "hold spill left no headroom";',
    ]
    nonfit = ["KV would be demoted", "compute buffer fell back to host", "not the published model", "transaction busy"]
    if not all(f in tc for f in fit) or tc.count("rung_fit_refused = true;") != 2:
        return False
    # the reserve's catch is a fit verdict only for the dedicated exception type, never for any std::exception
    if not re.search(
        r"rung_fit_refused = dynamic_cast<const llama_auto_ubatch_fit_refusal \*>\(&e\) != nullptr; return \"compute buffers did not fit\";",
        tc,
    ):
        return False
    # the probe's own refusal is a fit refusal only when the probe ran and said no
    if not re.search(r"if \(!probe\.accepted\) \{ rung_fit_refused = true; return \"transaction refused\"; \}", tc):
        return False
    for reason in nonfit:
        at = tc.find(f'return "{reason}";')
        if at == -1 or "rung_fit_refused = true;" in tc[max(0, at - 60) : at]:
            return False
    # the lifecycle failure and the publish throw keep their stop reason and stay non-fit
    return 'rc != GGML_SYCL_LIFECYCLE_OK' in tc and 'publish_dirty = true; return "transaction refused";' in tc


def test_the_fit_flag_is_set_only_by_real_fit_refusals():
    assert _fit_flag_is_set_only_on_fit_refusals(_try_candidate_norm())


@pytest.mark.parametrize(
    "old,new",
    [
        ('            return "KV would be demoted";', '            rung_fit_refused = true;\n            return "KV would be demoted";'),
        ('                return "compute buffer fell back to host";',
         '                rung_fit_refused = true;\n                return "compute buffer fell back to host";'),
        ('            return "transaction busy";', '            rung_fit_refused = true;\n            return "transaction busy";'),
        ('            publish_dirty           = true;\n            return "transaction refused";',
         '            publish_dirty           = true;\n            rung_fit_refused = true;\n            return "transaction refused";'),
        ('            return "hold spill left no headroom" ;', ''),
        ('rung_fit_refused = dynamic_cast<const llama_auto_ubatch_fit_refusal *>(&e) != nullptr;', 'rung_fit_refused = true;'),
        ('rung_fit_refused = dynamic_cast<const llama_auto_ubatch_fit_refusal *>(&e) != nullptr;', 'rung_fit_refused = false;'),
    ],
)
def test_the_fit_flag_has_a_mutation_witness(old, new):
    raw = LLAMA_CONTEXT_CPP
    pat = _ws_tolerant(old)
    found = len(pat.findall(raw))
    assert found == 1, f"mutation target not unique/found ({found}) -- update this witness: {old!r}"
    mutated = _body_of(pat.sub(lambda _m: new, raw, count=1), _TRY_CANDIDATE_START, _TRY_CANDIDATE_END)
    assert not _fit_flag_is_set_only_on_fit_refusals(mutated)


@pytest.mark.parametrize(
    "decl",
    [
        "bool           descent_ran             = false;",
        "bool           fallback_tried          = false;",
        "uint32_t       lowered_from            = 0;",
        "bool           rung_fit_refused        = false;",
        "uint32_t       lowest_refused          = 0;",
    ],
)
def test_the_trial_state_is_initialised_to_nothing_happened(decl):
    """Each of these starts false/zero: `descent_ran` true would run the walk for a default nobody refused,
    `fallback_tried` true would lower a default that was never asked about, `lowered_from` non-zero would suppress the
    tuning-cache store and print a lowering that did not happen."""
    pattern = r"\s+".join(re.escape(tok) for tok in decl.split())
    assert re.search(pattern, LLAMA_CONTEXT_CPP_CODE), decl


def test_the_trial_never_writes_the_context_size():
    """A smaller -ub is not a smaller context: no part of the trial or its settle may assign cparams.n_ctx."""
    body = _trial_norm()
    assert not re.search(r"cparams\.n_ctx\s*(?:[-+*/]?=(?!=)|\+\+|--)", body)
    mutant = _trial_mutant("        cparams.n_ubatch = last_good;\n        // The settle's publish", "        cparams.n_ctx = last_good;\n        // The settle's publish") \
        if "        cparams.n_ubatch = last_good;\n        // The settle's publish" in LLAMA_CONTEXT_CPP else None
    if mutant is not None:
        assert re.search(r"cparams\.n_ctx\s*(?:[-+*/]?=(?!=)|\+\+|--)", mutant)


def test_every_rung_asked_is_named_in_tried():
    body = _trial_norm()
    assert 'tried += (tried.empty() ? "" : ",") + std::to_string(cached_ubatch);' in body
    assert 'tried += (tried.empty() ? "" : ",") + std::to_string(c);' in body
    assert 'tried.append(tried.empty() ? "" : ",").append(std::to_string(c));' in body


def test_a_refused_cached_value_is_not_paid_for_twice_and_is_evicted():
    """A cached rung that fails its revalidation is a loss this start already knows: the ladder stops AT it instead of
    asking again, and a result that ends up lowered (which is never cached) still overwrites the refused entry."""
    body = _trial_norm()
    assert re.search(
        r"if \(rung_fit_refused\) \{ cache_refused_ub = cached_ubatch; cache_refused_reason = cache_reason; "
        r"fallback_tried = fallback_tried \|\| cached_ubatch == fallback_ubatch; \}",
        body,
    )
    assert re.search(r"if \(cache_refused_ub != 0 && c >= cache_refused_ub\) \{ stop = cache_refused_reason; "
        r"last_stop = cache_refused_reason; break; \}", body)
    assert "const bool store_outcome = lowered_from == 0 ? !descent_ran : cache_refused_ub != 0;" in body
    assert re.search(r"if \(ladder_needed && !stop_is_pure_race && store_outcome && !resumed_outcome_unchanged &&", body)


def test_the_header_names_the_advice_and_rounds_the_descent_to_64():
    code = _normalize_ws(strip_comments(LLAMA_AUTO_UBATCH_H))
    assert "inline uint32_t llama_auto_ubatch_advice(uint32_t largest_fit, uint32_t lowest_refused)" in code
    assert "return half >= llama_auto_ubatch_descent_floor ? half - half % 64 : 0;" in code


def test_the_scheduler_scope_is_opened_around_every_compute_buffer_allocation():
    """A buffer the backend places is a scheduler compute buffer only inside this scope; the absence of a model load
    (which also covers a recurrent-state buffer) no longer says so."""
    cpp = _normalize_ws(LLAMA_CONTEXT_CPP_CODE)
    assert "struct sycl_compute_scope_guard" in cpp
    assert re.search(
        r"bool llama_context::sched_alloc_graph\(ggml_cgraph \* gf\) \{ sycl_compute_scope_guard \w+\(sycl_compute_scope_fn\(\)\); "
        r"return ggml_backend_sched_alloc_graph\(sched\.get\(\), gf\); \}",
        cpp,
    )
    assert re.search(
        r"bool llama_context::sched_reserve_graph\(ggml_cgraph \* gf\) \{ sycl_compute_scope_guard \w+\(sycl_compute_scope_fn\(\)\); "
        r"return ggml_backend_sched_reserve\(sched\.get\(\), gf\); \}",
        cpp,
    )
    assert '"ggml_backend_sycl_compute_alloc_scope"' in cpp
    header = _normalize_ws(strip_comments((ROOT / "src/llama-context.h").read_text()))
    assert "sycl_compute_scope_fn()" in header


def test_the_design_doc_describes_the_named_settle_error_not_a_descent():
    doc = (ROOT / "docs/backend/sycl-memory-design.md").read_text()
    assert "settle_refusal_descend" not in doc
    assert "llama_auto_ubatch_advice" in doc


def _settle_catch_only_records(body_norm: str) -> bool:
    settle = body_norm[body_norm.find("if (!sched_matches_last_good") :]
    catch_at = settle.find("} catch (const std::exception & e) {")
    if catch_at == -1 or settle.count("catch") != 1:
        return False
    catch_body = _balanced_braces(settle, catch_at + len("} catch (const std::exception & e) "))
    return (
        catch_body[1:-1].strip() == "settle_error = std::current_exception(); settle_refusal = e.what();"
        and "return" not in _settle_refused_branch(body_norm)
    )


@pytest.mark.parametrize("mutation", ["catch-records-nothing", "return-in-refusal"])
def test_the_settle_refusal_record_has_a_mutation_witness(mutation):
    assert _settle_catch_only_records(_trial_norm())
    if mutation == "catch-records-nothing":
        old = "                settle_error   = std::current_exception();\n"
        new = ""
    else:
        old = "            uint32_t largest_ub = 0;\n            if (llama_context_sycl_hold_spill_fits(backends, last_good, &largest_ub)) {"
        new = "            return;\n" + old
    assert not _settle_catch_only_records(_trial_mutant(old, new))


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
