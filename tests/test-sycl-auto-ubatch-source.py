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
  (`ggml_backend_sycl_probe_runtime_context_for_model`) is consulted first,
  with a bounded exponential BUSY backoff (comment c-rkye item 3, mirroring
  `sycl_resync_runtime_context_flash_attn()`'s own retry); STALE_IDENTITY is
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
# -- see test_sycl_select_auto_ubatch_takes_type_k_and_type_v below) --
# updated here since every other check in this file depends on this exact
# string via _trial_body().
_TRIAL_START = "void llama_context::sycl_select_auto_ubatch(ggml_type type_k, ggml_type type_v) {"
_TRIAL_END = "void llama_context::sched_reserve() {"
# llama.cpp-7n6n: the shared per-candidate validator --
# probe with busy backoff, publish in a try/catch, reserve, host-fallback
# check -- extracted into one lambda used by BOTH the cache-hit revalidation
# and the ladder loop (previously two independently-maintained copies of the
# same sequence). Bounds a smaller region than the old loop-anchored checks
# used to; several per-candidate tests below are anchored here instead of to
# "for (uint32_t c : ladder)" now that the sequence itself lives here, not
# in the loop body.
_TRY_CANDIDATE_START = "auto try_candidate = [&](uint32_t c) -> const char * {"
_TRY_CANDIDATE_END = "auto cache_enabled_fn ="


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
    (auto_ubatch_enabled), and causal_attn -- combined in one `if` so a
    single condition being false cannot leave the others live."""
    body_norm = _normalize_ws(_call_site_body())
    gate_match = re.search(
        r"if\s*\(\s*params\.n_ubatch_auto\s*&&\s*cparams\.causal_attn\s*&&\s*"
        r"llama_context_has_sycl_backend\(\s*backends\s*\)\s*\)\s*\{",
        body_norm,
    )
    assert gate_match is not None, (
        "the call site must gate on params.n_ubatch_auto && cparams.causal_attn && "
        "llama_context_has_sycl_backend(backends) as one combined condition"
    )

    # auto_ubatch_enabled() must be consulted INSIDE that gate (either the
    # direct GGML_USE_SYCL symbol or the GGML_BACKEND_DL proc-address
    # lookup), not unconditionally before or after it.
    assert "ggml_backend_sycl_auto_ubatch_enabled" in body_norm, (
        "the call site must consult ggml_backend_sycl_auto_ubatch_enabled() (directly or via its "
        "proc-address lookup)"
    )
    enabled_idx = body_norm.find("ggml_backend_sycl_auto_ubatch_enabled", gate_match.end())
    assert enabled_idx != -1, (
        "ggml_backend_sycl_auto_ubatch_enabled must be consulted AFTER the four-condition gate opens, "
        "not before it (it should not run when the other conditions already ruled the trial out)"
    )

    # And the trial itself must only run when sycl_auto_ubatch_trial ends up
    # true; otherwise today's single sched_reserve() call is unchanged.
    # quality round 1, Q4: the call now forwards params.type_k/type_v (see
    # test_sycl_select_auto_ubatch_takes_type_k_and_type_v).
    assert re.search(
        r"if\s*\(\s*sycl_auto_ubatch_trial\s*\)\s*\{\s*sycl_select_auto_ubatch\s*\(\s*params\.type_k\s*,\s*"
        r"params\.type_v\s*\)\s*;",
        body_norm,
    ), (
        "the call site must call sycl_select_auto_ubatch(params.type_k, params.type_v) only when "
        "sycl_auto_ubatch_trial is true"
    )
    assert re.search(r"\}\s*else\s*\{\s*sched_reserve\s*\(\s*\)\s*;\s*\}", body_norm), (
        "the call site must fall through to today's unconditional sched_reserve() otherwise"
    )


def test_call_site_gate_has_a_mutation_witness():
    """Mutation witness for the gate check above: proves it would actually
    catch one of the four conditions (causal_attn) being dropped from the
    gate -- the exact regression c-dcct exists to prevent (a non-causal
    model's n_ubatch shrunk by the trial)."""
    raw = LLAMA_CONTEXT_CPP
    gate_line = (
        "        if (params.n_ubatch_auto && cparams.causal_attn && "
        "llama_context_has_sycl_backend(backends)) {\n"
    )
    assert gate_line in raw, "mutation target not found -- update this witness to match the real source"
    mutated_raw = raw.replace(
        gate_line, "        if (params.n_ubatch_auto && llama_context_has_sycl_backend(backends)) {\n", 1
    )
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _CALL_SITE_START, _CALL_SITE_END)
    assert not re.search(
        r"if\s*\(\s*params\.n_ubatch_auto\s*&&\s*cparams\.causal_attn\s*&&\s*"
        r"llama_context_has_sycl_backend\(\s*backends\s*\)\s*\)\s*\{",
        mutated_body_norm,
    ), "mutation witness is broken: dropping causal_attn from the gate should make the four-condition check fail"


# ---------------------------------------------------------------------------
# The trial's ladder and caps
# ---------------------------------------------------------------------------


def test_ladder_literal_is_512_1024_2048_4096():
    """The candidate ladder must be exactly {512, 1024, 2048, 4096},
    ascending, per the task spec and the Task 3 spike."""
    body_norm = _normalize_ws(_trial_body())
    assert re.search(
        r"static\s+const\s+uint32_t\s+ladder\s*\[\s*\]\s*=\s*\{\s*512\s*,\s*1024\s*,\s*2048\s*,\s*4096\s*\}\s*;",
        body_norm,
    ), "the ladder literal must be exactly { 512, 1024, 2048, 4096 }"


def test_ladder_literal_has_a_mutation_witness():
    """Mutation witness for the ladder check above: proves it would
    actually catch a rung being changed (4096 -> 8192)."""
    raw = LLAMA_CONTEXT_CPP
    ladder_line = "    static const uint32_t ladder[] = { 512, 1024, 2048, 4096 };\n"
    assert raw.count(ladder_line) == 1, f"mutation target not unique -- found {raw.count(ladder_line)}"
    mutated_raw = raw.replace(ladder_line, "    static const uint32_t ladder[] = { 512, 1024, 2048, 8192 };\n", 1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _TRIAL_START, _TRIAL_END)
    assert not re.search(
        r"static\s+const\s+uint32_t\s+ladder\s*\[\s*\]\s*=\s*\{\s*512\s*,\s*1024\s*,\s*2048\s*,\s*4096\s*\}\s*;",
        mutated_body_norm,
    ), "mutation witness is broken: changing the last rung should make the literal check fail"


def test_candidate_cap_uses_n_batch_and_n_ctx():
    """The ladder cap must be min(n_batch, n_ctx) -- both, not either
    alone -- computed before the loop, matching the Task 3 spike."""
    body_norm = _normalize_ws(_trial_body())
    assert re.search(
        r"uint32_t\s+cap\s*=\s*std::min\(\s*cparams\.n_batch\s*,\s*cparams\.n_ctx\s*\)\s*;", body_norm
    ), "cap must be computed as std::min(cparams.n_batch, cparams.n_ctx)"

    cap_idx = body_norm.find("uint32_t cap = std::min(cparams.n_batch, cparams.n_ctx);")
    ladder_idx = body_norm.find("static const uint32_t ladder[]")
    loop_idx = body_norm.find("for (uint32_t c : ladder)")
    assert cap_idx != -1 and ladder_idx != -1 and loop_idx != -1
    assert ladder_idx < cap_idx < loop_idx, "cap must be computed after the ladder literal and before the loop"

    assert re.search(r"if\s*\(\s*c\s*>\s*cap\s*\)\s*\{\s*break\s*;\s*\}", body_norm), (
        "the loop must break on the first candidate exceeding cap"
    )


# ---------------------------------------------------------------------------
# Early exits before the loop, quality round 1 -- both take the pre-trial
# path (a bare sched_reserve()) with NO WARN, since the trial never got a
# chance to try a single candidate.
# ---------------------------------------------------------------------------


def test_cap_below_first_rung_exits_before_the_loop_with_no_warn():
    """Q2a: when the fully-narrowed cap is below the ladder's first rung
    (any -c below 512, or llama-bench's pp128/tg128 rows), the function
    must take the pre-trial path -- sched_reserve() then return -- BEFORE
    ever entering the ladder loop, and must NOT log the [SYCL-PLAN] WARN
    (an empty `tried` list against a ladder that never ran)."""
    body_norm = _normalize_ws(_trial_body())
    cap_check_idx = body_norm.find("if (cap < ladder[0]) {")
    loop_idx = body_norm.find("for (uint32_t c : ladder)")
    warn_idx = body_norm.find("[SYCL-PLAN] auto n_ubatch=")
    assert cap_check_idx != -1, "the cap < ladder[0] early exit must exist"
    assert loop_idx != -1 and warn_idx != -1
    assert cap_check_idx < loop_idx, "the cap < ladder[0] check must precede the ladder loop"
    assert cap_check_idx < warn_idx, "the WARN literal must not appear before the cap < ladder[0] check"

    early_exit_block = body_norm[cap_check_idx : body_norm.find("}", cap_check_idx) + 1]
    assert re.search(r"sched_reserve\(\s*\)\s*;\s*return\s*;", early_exit_block), (
        "the early exit must call sched_reserve() then return, with no publish and no WARN in between"
    )


def test_cap_below_first_rung_early_exit_has_a_mutation_witness():
    """Mutation witness for the check above: proves it would actually
    catch the early exit being deleted (falling through into the loop
    with an empty ladder that would silently exhaust with an empty
    `tried` list instead of exiting cleanly beforehand)."""
    raw = LLAMA_CONTEXT_CPP
    early_exit_block = (
        "    if (cap < ladder[0]) {\n"
        "        sched_reserve();\n"
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
    token_idx = body_norm.find("const ggml_sycl_model_token token")
    loop_idx = body_norm.find("for (uint32_t c : ladder)")
    warn_idx = body_norm.find("[SYCL-PLAN] auto n_ubatch=")
    assert guard_idx != -1, "the zero-token guard must exist"
    assert token_idx != -1 and loop_idx != -1 and warn_idx != -1
    assert guard_idx < token_idx < loop_idx, (
        "the zero-token guard must precede both the token construction and the ladder loop"
    )
    assert guard_idx < warn_idx, "the WARN literal must not appear before the zero-token guard"

    guard_block = body_norm[guard_idx : body_norm.find("}", guard_idx) + 1]
    assert re.search(r"sched_reserve\(\s*\)\s*;\s*return\s*;", guard_block), (
        "the zero-token guard must call sched_reserve() then return, with no publish and no WARN in between"
    )


def test_zero_sycl_token_guard_has_a_mutation_witness():
    """Mutation witness for the check above: proves it would actually
    catch the zero-token guard being deleted."""
    raw = LLAMA_CONTEXT_CPP
    guard_block = (
        "    if (owner.model_id == 0 || owner.load_txn_id == 0) {\n"
        "        sched_reserve();\n"
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


def test_moe_model_cap_uses_the_gpu_moe_ubatch_ceiling():
    """For a MoE model (hparams.n_expert > 0), the cap must be additionally
    narrowed to ggml_backend_sycl_moe_gpu_ubatch_max() when that ceiling is
    smaller than the n_batch/n_ctx cap -- comment c-s747 / llama.cpp-ohkx.
    The 512 constant itself must NOT be hardcoded in llama-context.cpp; it
    must come from the accessor."""
    body_norm = _normalize_ws(_trial_body())
    assert re.search(r"if\s*\(\s*model\.hparams\.n_expert\s*>\s*0\s*\)\s*\{", body_norm), (
        "the MoE cap must be gated on model.hparams.n_expert > 0"
    )
    assert "ggml_backend_sycl_moe_gpu_ubatch_max" in body_norm, (
        "the MoE cap must be read from ggml_backend_sycl_moe_gpu_ubatch_max(), not a local constant"
    )
    assert re.search(r"if\s*\(\s*moe_cap\s*<\s*cap\s*\)\s*\{", body_norm), (
        "the MoE cap must only narrow `cap` when it is actually smaller"
    )
    assert not re.search(r"\bcap\s*=\s*512\b", body_norm), (
        "the MoE ceiling must not be hardcoded as a bare 512 in llama-context.cpp"
    )


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


def test_probe_busy_retries_with_bounded_exponential_backoff():
    """A GGML_SYCL_LIFECYCLE_BUSY probe result must retry with the same
    bounded exponential backoff sycl_resync_runtime_context_flash_attn()
    uses (max 7 waits, 1<<wait ms), not spin immediately or retry
    unbounded. Quality round 1 Q5: both retries share ONE named file-scope
    constant, llama_context_sycl_max_busy_waits -- neither loop may
    re-declare its own local max_busy_waits."""
    assert re.search(
        r"static\s+constexpr\s+int\s+llama_context_sycl_max_busy_waits\s*=\s*7\s*;", LLAMA_CONTEXT_CPP_CODE
    ), "the shared BUSY-retry bound must be declared once as llama_context_sycl_max_busy_waits = 7"
    assert not re.search(r"constexpr\s+int\s+max_busy_waits\s*=\s*7\s*;", LLAMA_CONTEXT_CPP_CODE), (
        "neither retry loop may re-declare its own local max_busy_waits -- both must use the shared constant"
    )

    body_norm = _normalize_ws(_trial_body())
    assert re.search(
        r"for\s*\(\s*int\s+wait\s*=\s*0\s*;\s*rc\s*==\s*GGML_SYCL_LIFECYCLE_BUSY\s*&&\s*wait\s*<\s*"
        r"llama_context_sycl_max_busy_waits\s*;\s*\+\+wait\s*\)\s*\{",
        body_norm,
    ), (
        "the BUSY retry loop must be gated on rc == GGML_SYCL_LIFECYCLE_BUSY && wait < "
        "llama_context_sycl_max_busy_waits"
    )
    assert "std::this_thread::sleep_for(std::chrono::milliseconds(1u << wait))" in body_norm, (
        "the BUSY retry must sleep 1u << wait milliseconds, matching the exponential backoff shape"
    )


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
    actually catch the publish's try/catch block and the reserve calls
    being swapped."""
    raw = LLAMA_CONTEXT_CPP
    try_catch_block = (
        "        try {\n"
        "            sycl_resync_runtime_context_flash_attn();\n"
        "        } catch (const std::exception &) {\n"
        "            sched_matches_last_good = false;\n"
        '            return "transaction refused";\n'
        "        }\n"
    )
    reserve_block = (
        "        // llama.cpp-xojq (quality round 1 Q2b): this publish just took\n"
        "        // effect on every SYCL backend (the try above did not throw), so\n"
        "        // device state may now differ from fallback_ubatch even if this\n"
        "        // candidate goes on to lose the host-fallback check below -- the\n"
        "        // settle step's own publish gate reads this flag to know whether\n"
        "        // it must correct that state back.\n"
        "        published_any      = true;\n"
        "        sched_need_reserve = true;\n"
        "        sched_reserve();\n"
    )
    original_block = try_catch_block + reserve_block
    assert original_block in raw, "mutation target not found -- update this witness to match the real source"
    mutated_block = reserve_block + try_catch_block
    mutated_raw = raw.replace(original_block, mutated_block, 1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _TRY_CANDIDATE_START, _TRY_CANDIDATE_END)
    mutated_publish_idx = mutated_body_norm.find("sycl_resync_runtime_context_flash_attn();")
    mutated_reserve_idx = mutated_body_norm.find("sched_reserve();")
    assert mutated_publish_idx != -1 and mutated_reserve_idx != -1
    assert not (mutated_publish_idx < mutated_reserve_idx), (
        "mutation witness is broken: swapping the two blocks should make the ordering check fail"
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
    loop_idx = body_norm.find("for (uint32_t c : ladder)")
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
    fallback_block = (
        "        for (auto & sb : sycl_backends) {\n"
        "            if (fallback_fn(sb.dev_index) > 0) {\n"
        "                sched_matches_last_good = false;\n"
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


def test_all_eight_stop_reasons_are_present():
    """The trial's stop-reason vocabulary must be EXACTLY the eight
    strings the task spec names -- the original seven (llama.cpp-xojq
    Task 4b) plus "cached" (llama.cpp-7n6n, Task 5: a persisted-cache hit
    that revalidates cleanly skips the ladder with this stop reason).

    llama.cpp-7n6n: three of the eight are still
    literal `stop = "...";` assignments ("ladder exhausted"'s initial
    declaration, "MoE GPU routing ceiling", and "cached"); the other five
    are `return "...";` statements inside try_candidate(), moved there
    specifically so a losing cache revalidation cannot leave its reason
    behind in `stop`. Both forms are drawn from and must match this
    one set, with none missing and none extra."""
    body_norm = _normalize_ws(_trial_body())
    eight = {
        "ladder exhausted",
        "MoE GPU routing ceiling",
        "transaction refused",
        "transaction busy",
        "not the published model",
        "KV would be demoted",
        "compute buffer fell back to host",
        "cached",
    }
    for reason in eight:
        assert f'"{reason}"' in body_norm, f"missing stop reason literal: {reason!r}"

    # Presence alone (the loop above) would pass even if a ninth string had
    # silently slipped in as a `stop = "..."` or `return "...";` somewhere --
    # this closes that gap by requiring the extracted SET to match exactly.
    found = set(re.findall(r'stop\s*=\s*"([^"]*)"', body_norm)) | set(
        re.findall(r'return\s*"([^"]*)"\s*;', body_norm)
    )
    assert found == eight, f"stop-reason literal set does not match exactly -- found {found}"


# ---------------------------------------------------------------------------
# The candidate publish can still refuse after an accepted probe (Task 2
# final review addendum, 2026-09-11: the probe's own exit branch returns
# BEFORE the publication-ID check, MMID materialization, and the CAS, all of
# which still run for a real publish and can still refuse).
# ---------------------------------------------------------------------------


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
        r"try\s*\{\s*sycl_resync_runtime_context_flash_attn\(\s*\)\s*;\s*\}\s*catch\s*\(\s*const\s+std::exception\s*"
        r'&\s*\)\s*\{\s*sched_matches_last_good\s*=\s*false\s*;\s*return\s*"transaction refused"\s*;\s*\}',
        body_norm,
    ), (
        "the publish must be wrapped in try { ... } catch (const std::exception &) { sched_matches_last_good = "
        'false; return "transaction refused"; }'
    )


def test_candidate_publish_try_catch_has_a_mutation_witness():
    """Mutation witness for the check above: proves it would actually
    catch the try/catch being deleted (leaving a bare, unprotected publish
    call that would let a refusal escape as an uncaught exception instead
    of a clean "transaction refused" stop)."""
    raw = LLAMA_CONTEXT_CPP
    wrapped_block = (
        "        try {\n"
        "            sycl_resync_runtime_context_flash_attn();\n"
        "        } catch (const std::exception &) {\n"
        "            sched_matches_last_good = false;\n"
        '            return "transaction refused";\n'
        "        }\n"
    )
    assert wrapped_block in raw, "mutation target not found -- update this witness to match the real source"
    mutated_raw = raw.replace(wrapped_block, "        sycl_resync_runtime_context_flash_attn();\n", 1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _TRY_CANDIDATE_START, _TRY_CANDIDATE_END)
    assert not re.search(
        r"try\s*\{\s*sycl_resync_runtime_context_flash_attn\(\s*\)\s*;\s*\}\s*catch\s*\(\s*const\s+std::exception\s*"
        r'&\s*\)\s*\{\s*sched_matches_last_good\s*=\s*false\s*;\s*return\s*"transaction refused"\s*;\s*\}',
        mutated_body_norm,
    ), "mutation witness is broken: deleting the try/catch should make the wrapped-publish check fail"


def test_settle_publish_is_not_wrapped_in_try_catch():
    """Unlike the candidate publish inside try_candidate(), the SETTLE
    publish must let a refusal propagate -- today's behaviour for a
    context that does not fit at all (Task 2 final review addendum item 1:
    only the candidate publish gets the try/catch)."""
    body_norm = _normalize_ws(_trial_body())
    settle_start = body_norm.find("if (!sched_matches_last_good")
    assert settle_start != -1, "could not find the settle step's own gate"
    settle_block = body_norm[settle_start:]
    settle_publish_idx = settle_block.find("sycl_resync_runtime_context_flash_attn();")
    assert settle_publish_idx != -1, "could not find the settle step's own publish call"
    assert "try {" not in settle_block[:settle_publish_idx + 40], (
        "the settle publish must NOT be wrapped in try/catch -- its refusal must propagate, matching today's "
        "behaviour for a context that does not fit"
    )


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


def test_published_any_is_declared_false_and_set_true_after_the_publish_catch():
    """Q2b: published_any must start false and become true right after
    try_candidate()'s own publish succeeds (the try did not throw, so its
    catch block's early return was not taken) -- tracking whether ANY
    candidate's publish actually took effect this trial run, independent
    of whether that candidate goes on to lose the host-fallback check."""
    body_norm = _normalize_ws(_trial_body())
    assert re.search(r"bool\s+published_any\s*=\s*false\s*;", body_norm), (
        "published_any must be declared, initialized false"
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


def test_settle_publish_is_gated_on_published_any_or_changed_value():
    """Q2b: the settle step's PUBLISH (not its reserve) must be gated on
    `published_any || cparams.n_ubatch != fallback_ubatch` -- when nothing
    was ever published this trial and the resolved value is the same one
    already published by the constructor's own earlier publish, a settle
    republish would be a redundant runtime-context transaction with no
    state change (the exact scenario this finding reported: the loop
    breaking at once on a first-candidate refusal). The RESERVE must still
    run unconditionally inside the settle gate -- a context that never
    calls sched_reserve() anywhere has no compute buffers at all."""
    body_norm = _normalize_ws(_trial_body())
    settle_idx = body_norm.find("if (!sched_matches_last_good")
    assert settle_idx != -1
    settle_block = body_norm[settle_idx:]

    need_publish_idx = settle_block.find(
        "const bool need_publish = published_any || cparams.n_ubatch != fallback_ubatch;"
    )
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


def test_settle_publish_gate_has_a_mutation_witness():
    """Mutation witness for the two checks above: proves they would
    actually catch the settle publish reverting to unconditional (always
    publishing, even when nothing changed), AND (quality round 2 R3) that
    the guard-body check specifically would catch the guard being emptied
    with the call moved below it -- a mutant the ordering assertion alone
    cannot see (ordering is satisfied textually either way)."""
    raw = LLAMA_CONTEXT_CPP
    original_settle = (
        "    if (!sched_matches_last_good || cparams.n_ubatch != last_good) {\n"
        "        const bool need_publish = published_any || cparams.n_ubatch != fallback_ubatch;\n"
        "        cparams.n_ubatch        = last_good;\n"
        "        if (need_publish) {\n"
        "            sycl_resync_runtime_context_flash_attn();\n"
        "        }\n"
        "        sched_need_reserve = true;\n"
        "        sched_reserve();\n"
        "    }\n"
    )
    assert original_settle in raw, "mutation target not found -- update this witness to match the real source"
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
    mutated_settle_idx = mutated_body_norm.find("if (!sched_matches_last_good")
    assert mutated_settle_idx != -1
    assert "need_publish" not in mutated_body_norm[mutated_settle_idx:], (
        "mutation witness is broken: reverting to an unconditional publish should remove need_publish entirely"
    )

    # Second mutant (quality round 2 R3): keeps need_publish and its
    # ordering intact, but EMPTIES the guard's body and moves the publish
    # call to just after it -- passes the positional ordering assertion
    # above (need_publish < assign < guard < call < reserve still holds
    # textually) but must fail the REAL guard-body check, not a
    # re-implementation of it.
    emptied_guard_settle = (
        "    if (!sched_matches_last_good || cparams.n_ubatch != last_good) {\n"
        "        const bool need_publish = published_any || cparams.n_ubatch != fallback_ubatch;\n"
        "        cparams.n_ubatch        = last_good;\n"
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
    for member in ("int\\s+device\\s*;", "const\\s+char\\s*\\*\\s*model_name\\s*;", "uint64_t\\s+model_size\\s*;",
                   "uint64_t\\s+model_hash\\s*;", "uint32_t\\s+n_ctx\\s*;", "uint32_t\\s+n_batch\\s*;",
                   "bool\\s+flash_attn\\s*;",
                   # quality round 1, Q4: the four fields added this round.
                   "uint32_t\\s+n_seq_max\\s*;", "int32_t\\s+type_k\\s*;", "int32_t\\s+type_v\\s*;",
                   "uint32_t\\s+device_set_hash\\s*;"):
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
        r"GGML_BACKEND_API\s+bool\s+ggml_backend_sycl_ubatch_cache_lookup\s*\(\s*const\s+struct\s+"
        r"ggml_sycl_ubatch_cache_key\s*\*\s*key\s*,\s*uint32_t\s*\*\s*n_ubatch\s*,\s*char\s*\*\s*reason_buf\s*,"
        r"\s*size_t\s+reason_buf_size\s*\)\s*;",
        GGML_SYCL_H_CODE,
    ), (
        "ggml_backend_sycl_ubatch_cache_lookup(const ggml_sycl_ubatch_cache_key*, uint32_t*, char*, size_t) must "
        "be declared"
    )
    assert re.search(
        r"GGML_BACKEND_API\s+bool\s+ggml_backend_sycl_ubatch_cache_store\s*\(\s*const\s+struct\s+"
        r"ggml_sycl_ubatch_cache_key\s*\*\s*key\s*,\s*uint32_t\s+n_ubatch\s*,\s*const\s+char\s*\*\s*reason\s*\)\s*;",
        GGML_SYCL_H_CODE,
    ), "ggml_backend_sycl_ubatch_cache_store(const ggml_sycl_ubatch_cache_key*, uint32_t, const char*) must be declared"


def test_proc_address_registers_the_four_ubatch_cache_accessors():
    """A GGML_BACKEND_DL build's llama-context lookup needs all four
    entry points registered in the strcmp proc-address chain, mirroring
    the auto_ubatch_enabled/moe_gpu_ubatch_max precedent just above."""
    for symbol in (
        "ggml_backend_sycl_ubatch_cache_enabled",
        "ggml_backend_sycl_ubatch_cache_path",
        "ggml_backend_sycl_ubatch_cache_lookup",
        "ggml_backend_sycl_ubatch_cache_store",
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
    loop_idx = body_norm.find("for (uint32_t c : ladder)")
    assert lookup_idx != -1, "could not find the cache_lookup_fn(&cache_key, ...) call"
    assert loop_idx != -1
    assert lookup_idx < loop_idx, "the tuning-cache lookup must precede the ladder loop"


def test_ubatch_cache_store_follows_the_ladder():
    """The tuning-cache store must run AFTER the ladder loop has finished
    (and after the last_good == 0 fallback correction, so it never
    persists 0) -- never before it, and never inside it."""
    body_norm = _normalize_ws(_trial_body())
    loop_idx = body_norm.find("for (uint32_t c : ladder)")
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

    loop_idx = body_norm.find("for (uint32_t c : ladder) {")
    assert loop_idx != -1
    after_loop_open = body_norm[loop_idx + len("for (uint32_t c : ladder) {") :]
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
        r"ggml_backend_sycl_ubatch_cache_lookup[\s\S]{0,400}?ubatch_tuning_cache_env_enabled\s*\(\s*\)",
        UBATCH_TUNING_CACHE_CPP_CODE,
    ), "ggml_backend_sycl_ubatch_cache_lookup must consult the enabled accessor before doing anything else"
    assert re.search(
        r"ggml_backend_sycl_ubatch_cache_store[\s\S]{0,400}?ubatch_tuning_cache_env_enabled\s*\(\s*\)",
        UBATCH_TUNING_CACHE_CPP_CODE,
    ), "ggml_backend_sycl_ubatch_cache_store must consult the enabled accessor before doing anything else"


def test_ubatch_cache_lookup_and_store_have_mutation_witnesses():
    """Mutation witnesses for the two position checks above: prove they
    would actually catch the lookup/store being moved to the wrong side
    of the ladder loop."""
    raw = LLAMA_CONTEXT_CPP

    lookup_line = "    if (cache_available) {\n"
    assert lookup_line in raw, "mutation target not found -- update this witness to match the real source"
    loop_line = "    for (uint32_t c : ladder) {\n"
    assert loop_line in raw, "mutation target not found -- update this witness to match the real source"

    # Swap the two markers' relative order by moving the loop's opening
    # line to just before the cache-availability check -- crude, but
    # sufficient to prove the position assertions would notice.
    mutated_raw = raw.replace(loop_line, "", 1).replace(lookup_line, loop_line + lookup_line, 1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _TRIAL_START, _TRIAL_END)
    mutated_lookup_idx = mutated_body_norm.find("cache_lookup_fn(&cache_key,")
    mutated_loop_idx = mutated_body_norm.find("for (uint32_t c : ladder)")
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
        "    if (ladder_needed && !stop_is_pure_race && !resumed_outcome_unchanged && have_cache_accessors &&\n"
        "        cache_enabled_fn()) {\n"
        "        if (!cache_store_fn(&cache_key, last_good, stop)) {\n"
        '            LLAMA_LOG_WARN("[SYCL-PLAN] tuning cache store failed: %s\\n", cache_path_buf);\n'
        "        }\n"
        "    }\n"
    )
    assert store_block in raw, "mutation target not found -- update this witness to match the real source"
    loop_line = "    for (uint32_t c : ladder) {\n"
    assert loop_line in raw, "mutation target not found -- update this witness to match the real source"

    # Move the store block to just BEFORE the ladder loop itself (not merely
    # before the last_good == 0 fixup, which is already after the loop and
    # so would not actually exercise the loop-vs-store ordering check).
    mutated_raw = raw.replace(store_block, "", 1).replace(loop_line, store_block + loop_line, 1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _TRIAL_START, _TRIAL_END)
    mutated_loop_idx = mutated_body_norm.find("for (uint32_t c : ladder)")
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
        r"if\s*\(\s*ladder_needed\s*&&\s*!stop_is_pure_race\s*&&\s*!resumed_outcome_unchanged\s*&&\s*"
        r"have_cache_accessors\s*&&\s*cache_enabled_fn\s*\(\s*\)\s*\)\s*\{",
        body_norm,
    ), (
        "the store must be gated on ladder_needed && !stop_is_pure_race && !resumed_outcome_unchanged && "
        "have_cache_accessors && cache_enabled_fn()"
    )


def test_ubatch_cache_store_ladder_needed_gate_has_a_mutation_witness():
    """Mutation witness for the check above: proves it would actually
    catch the `ladder_needed &&` clause being dropped from the store's
    gate (which would re-store an identical entry after every TERMINAL
    cache hit, not just after a real ladder run)."""
    raw = LLAMA_CONTEXT_CPP
    guard_line = (
        "    if (ladder_needed && !stop_is_pure_race && !resumed_outcome_unchanged && have_cache_accessors &&\n"
    )
    assert guard_line in raw, "mutation target not found -- update this witness to match the real source"
    mutated_guard_line = (
        "    if (!stop_is_pure_race && !resumed_outcome_unchanged && have_cache_accessors &&\n"
    )
    mutated_raw = raw.replace(guard_line, mutated_guard_line, 1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _TRIAL_START, _TRIAL_END)
    assert not re.search(
        r"if\s*\(\s*ladder_needed\s*&&\s*!stop_is_pure_race\s*&&\s*!resumed_outcome_unchanged\s*&&\s*"
        r"have_cache_accessors\s*&&\s*cache_enabled_fn\s*\(\s*\)\s*\)\s*\{",
        mutated_body_norm,
    ), "mutation witness is broken: dropping ladder_needed from the guard should make the gate check fail"


# ---------------------------------------------------------------------------
# llama.cpp-7n6n quality round 1: Q2 (a stale hit could pin a value
# forever), Q3 (a silent store failure), Q4 (three key omissions).
# ---------------------------------------------------------------------------


def test_sycl_select_auto_ubatch_takes_type_k_and_type_v():
    """quality round 1, Q4: type_k/type_v are constructor-local
    llama_context_params fields sycl_select_auto_ubatch() cannot otherwise
    see (it is a separate member function, not inline in the constructor),
    so they are passed in as parameters and forwarded from the one call
    site."""
    assert re.search(
        r"void\s+llama_context::sycl_select_auto_ubatch\s*\(\s*ggml_type\s+type_k\s*,\s*ggml_type\s+type_v\s*\)\s*\{",
        LLAMA_CONTEXT_CPP_CODE,
    ), "sycl_select_auto_ubatch must take (ggml_type type_k, ggml_type type_v)"
    assert re.search(
        r"sycl_select_auto_ubatch\s*\(\s*params\.type_k\s*,\s*params\.type_v\s*\)\s*;", LLAMA_CONTEXT_CPP_CODE
    ), "the call site must forward params.type_k, params.type_v"


def test_cache_key_populates_the_four_new_fields():
    """quality round 1, Q4: n_seq_max/type_k/type_v/device_set_hash must
    all be assigned into cache_key -- declaring the struct fields (covered
    elsewhere) is not enough if nothing ever fills them in."""
    body_norm = _normalize_ws(_trial_body())
    for assignment in (
        r"cache_key\.n_seq_max\s*=\s*cparams\.n_seq_max\s*;",
        r"cache_key\.type_k\s*=\s*static_cast<int32_t>\s*\(\s*type_k\s*\)\s*;",
        r"cache_key\.type_v\s*=\s*static_cast<int32_t>\s*\(\s*type_v\s*\)\s*;",
        r"cache_key\.device_set_hash\s*=\s*device_set_hash\s*;",
    ):
        assert re.search(assignment, body_norm), f"missing cache_key field assignment matching {assignment!r}"


def test_device_set_hash_is_computed_over_every_sycl_backend():
    """quality round 1, Q4: device_set_hash must be an FNV-1a accumulation
    over EVERY entry of sycl_backends (not just the first, which is what
    cache_key.device itself already names) -- level_zero:0 and
    level_zero:0,1 both start with device 0, so a hash over only the first
    entry would not tell the two selector shapes apart."""
    body_norm = _normalize_ws(_trial_body())
    hash_idx = body_norm.find("uint32_t device_set_hash")
    assert hash_idx != -1, "could not find the device_set_hash declaration"
    loop_idx = body_norm.find("for (auto & sb : sycl_backends)", hash_idx)
    assert loop_idx != -1, "device_set_hash must be computed via a loop over sycl_backends"
    cache_key_idx = body_norm.find("cache_key.device_set_hash", loop_idx)
    assert cache_key_idx != -1, "device_set_hash must be computed before it is assigned into cache_key"
    loop_body = body_norm[loop_idx:cache_key_idx]
    assert re.search(r"device_set_hash\s*\^=\s*static_cast<uint32_t>\s*\(\s*sb\.dev_index\s*\)\s*;", loop_body), (
        "the loop must XOR in each backend's dev_index"
    )
    assert re.search(r"device_set_hash\s*\*=\s*0x01000193u\s*;", loop_body), (
        "the loop must multiply by the FNV-1a 32-bit prime after each XOR"
    )


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
    loop_idx = body_norm.find("for (uint32_t c : ladder)")
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
    store_gate_idx = body_norm.find("if (ladder_needed && !stop_is_pure_race && !resumed_outcome_unchanged")
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


def test_at_most_three_llama_log_warn_call_sites_in_the_body():
    """quality round 1, Q3: this function's own docstring was updated to
    say AT MOST THREE GGML_LOG_WARN lines report the outcome (the
    tuning-cache lookup outcome, an optional store-failure WARN, and the
    pre-existing auto n_ubatch outcome) -- pin the literal call-site count
    in the source, not just the docstring's prose."""
    body_norm = _normalize_ws(_trial_body())
    count = len(re.findall(r"LLAMA_LOG_WARN\(", body_norm))
    assert count == 3, f"expected exactly 3 LLAMA_LOG_WARN( call sites in the trial body -- found {count}"


def test_cache_path_return_is_checked_and_substituted():
    """quality round 1, Q9: ggml_backend_sycl_ubatch_cache_path()'s return
    must be checked -- a call that fails (an out-of-range device, or a
    path too long for the buffer) must not leave a caller trusting a
    silently-truncated or stale path."""
    body_norm = _normalize_ws(_trial_body())
    assert re.search(
        r"if\s*\(\s*have_cache_accessors\s*&&\s*!cache_path_fn\s*\(\s*cache_key\.device\s*,\s*cache_path_buf\s*,\s*"
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
