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
_TRIAL_START = "void llama_context::sycl_select_auto_ubatch() {"
_TRIAL_END = "void llama_context::sched_reserve() {"


def _call_site_body() -> str:
    return _bounded_body(LLAMA_CONTEXT_CPP_CODE, _CALL_SITE_START, _CALL_SITE_END)


def _trial_body() -> str:
    return _bounded_body(LLAMA_CONTEXT_CPP_CODE, _TRIAL_START, _TRIAL_END)


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
    assert re.search(r"if\s*\(\s*sycl_auto_ubatch_trial\s*\)\s*\{\s*sycl_select_auto_ubatch\s*\(\s*\)\s*;", body_norm), (
        "the call site must call sycl_select_auto_ubatch() only when sycl_auto_ubatch_trial is true"
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
    into the generic "transaction refused" branch."""
    body_norm = _normalize_ws(_trial_body())
    assert re.search(
        r'rc\s*==\s*GGML_SYCL_LIFECYCLE_STALE_IDENTITY\s*\)\s*\{\s*stop\s*=\s*"not the published model"\s*;',
        body_norm,
    ), 'GGML_SYCL_LIFECYCLE_STALE_IDENTITY must set stop = "not the published model"'


def test_would_demote_kv_is_consulted_and_stops_the_ladder():
    """probe.would_demote_kv must be consulted and, when true, stop the
    ladder with the "KV would be demoted" reason -- never published."""
    body_norm = _normalize_ws(_trial_body())
    assert re.search(
        r'probe\.would_demote_kv\s*\)\s*\{\s*stop\s*=\s*"KV would be demoted"\s*;', body_norm
    ), 'probe.would_demote_kv must be consulted and set stop = "KV would be demoted"'


# ---------------------------------------------------------------------------
# Publish-before-reserve order, and the host-fallback query after reserve
# ---------------------------------------------------------------------------


def test_publish_happens_before_reserve_inside_the_loop():
    """sycl_resync_runtime_context_flash_attn() (publish) must be called
    strictly BEFORE sched_reserve() inside the candidate loop -- the narrow
    flash-attn re-check inside sched_reserve() re-evaluates against the
    PUBLISHED plan's planner_n_ubatch, so publishing after reserving would
    let sched_reserve() see the PREVIOUS candidate's plan."""
    body_norm = _normalize_ws(_trial_body())
    loop_idx = body_norm.find("for (uint32_t c : ladder)")
    assert loop_idx != -1
    publish_idx = body_norm.find("sycl_resync_runtime_context_flash_attn();", loop_idx)
    reserve_idx = body_norm.find("sched_reserve();", loop_idx)
    assert publish_idx != -1 and reserve_idx != -1, "both the publish and the in-loop reserve call must exist"
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
        '            stop                    = "transaction refused";\n'
        "            candidate_lost          = true;\n"
        "            sched_matches_last_good = false;\n"
        "        }\n"
        "        if (candidate_lost) {\n"
        "            break;\n"
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

    mutated_body_norm = _body_of(mutated_raw, _TRIAL_START, _TRIAL_END)
    mutated_loop_idx = mutated_body_norm.find("for (uint32_t c : ladder)")
    mutated_publish_idx = mutated_body_norm.find("sycl_resync_runtime_context_flash_attn();", mutated_loop_idx)
    mutated_reserve_idx = mutated_body_norm.find("sched_reserve();", mutated_loop_idx)
    assert mutated_publish_idx != -1 and mutated_reserve_idx != -1
    assert not (mutated_publish_idx < mutated_reserve_idx), (
        "mutation witness is broken: swapping the two blocks should make the ordering check fail"
    )


def test_host_fallback_query_is_consulted_after_reserve():
    """ggml_backend_sycl_compute_buffer_host_fallbacks() must be consulted
    AFTER the in-loop sched_reserve() call (it reports fallbacks since the
    last successful publish, which sched_reserve() just exercised), and
    BEFORE last_good is updated to this candidate -- a candidate whose
    reserve fell back to host must not win."""
    body_norm = _normalize_ws(_trial_body())
    loop_idx = body_norm.find("for (uint32_t c : ladder)")
    reserve_idx = body_norm.find("sched_reserve();", loop_idx)
    fallback_idx = body_norm.find("fallback_fn(", reserve_idx)
    last_good_idx = body_norm.find("last_good = c;", reserve_idx)
    assert reserve_idx != -1 and fallback_idx != -1 and last_good_idx != -1
    assert reserve_idx < fallback_idx < last_good_idx, (
        "the host-fallback query must run strictly between the in-loop sched_reserve() call and the "
        "last_good = c assignment"
    )


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
    then silently win)."""
    raw = LLAMA_CONTEXT_CPP
    fallback_block = (
        "        bool host_fallback = false;\n"
        "        for (auto & sb : sycl_backends) {\n"
        "            if (fallback_fn(sb.dev_index) > 0) {\n"
        "                host_fallback = true;\n"
        "                break;\n"
        "            }\n"
        "        }\n"
        "        if (host_fallback) {\n"
        '            stop                    = "compute buffer fell back to host";\n'
        "            sched_matches_last_good = false;\n"
        "            break;\n"
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
# The WARN line and the stop-reason vocabulary
# ---------------------------------------------------------------------------


def test_exactly_one_sycl_plan_auto_warn_in_the_body():
    """Exactly one [SYCL-PLAN] auto n_ubatch= WARN must appear in the
    trial's body -- the task spec's "keep the trial's own log to ONE WARN"
    gotcha."""
    body_norm = _normalize_ws(_trial_body())
    count = len(re.findall(r"\[SYCL-PLAN\] auto n_ubatch=", body_norm))
    assert count == 1, f"expected exactly one '[SYCL-PLAN] auto n_ubatch=' WARN -- found {count}"
    assert re.search(r'LLAMA_LOG_WARN\(\s*"\[SYCL-PLAN\] auto n_ubatch=', body_norm), (
        "the line must be logged at GGML_LOG_WARN (LLAMA_LOG_WARN), not INFO"
    )


def test_all_seven_stop_reasons_are_present():
    """The trial's stop-reason vocabulary must be EXACTLY the seven
    strings the task spec names -- every `stop = "..."` literal in the
    body (including the initial `const char * stop = "...";`
    declaration) must be drawn from this set, with none missing and none
    extra."""
    body_norm = _normalize_ws(_trial_body())
    seven = {
        "ladder exhausted",
        "MoE GPU routing ceiling",
        "transaction refused",
        "transaction busy",
        "not the published model",
        "KV would be demoted",
        "compute buffer fell back to host",
    }
    for reason in seven:
        assert f'"{reason}"' in body_norm, f"missing stop reason literal: {reason!r}"

    # Presence alone (the loop above) would pass even if an eighth string
    # had silently slipped in as a `stop = "..."` assignment somewhere --
    # this closes that gap by requiring the extracted SET to match exactly.
    found = set(re.findall(r'stop\s*=\s*"([^"]*)"', body_norm))
    assert found == seven, f"stop-reason literal set does not match exactly -- found {found}"


# ---------------------------------------------------------------------------
# The candidate publish can still refuse after an accepted probe (Task 2
# final review addendum, 2026-09-11: the probe's own exit branch returns
# BEFORE the publication-ID check, MMID materialization, and the CAS, all of
# which still run for a real publish and can still refuse).
# ---------------------------------------------------------------------------


def test_candidate_publish_is_wrapped_in_try_catch():
    """The CANDIDATE publish inside the loop must be wrapped in
    try/catch(const std::exception&) -- sycl_resync_runtime_context_flash_
    attn() throws on a refusal, and an accepted probe does not guarantee
    the publish itself still succeeds. The catch must also mark
    sched_matches_last_good false (spec round 1 F1): the publish loop can
    throw partway through the SYCL backends it walks, so the published
    plan can no longer be trusted to describe last_good on any device,
    and only sched_matches_last_good=false forces the settle step to
    unconditionally re-publish last_good everywhere."""
    body_norm = _normalize_ws(_trial_body())
    assert re.search(
        r"try\s*\{\s*sycl_resync_runtime_context_flash_attn\(\s*\)\s*;\s*\}\s*catch\s*\(\s*const\s+std::exception\s*"
        r'&\s*\)\s*\{\s*stop\s*=\s*"transaction refused"\s*;\s*candidate_lost\s*=\s*true\s*;\s*'
        r"sched_matches_last_good\s*=\s*false\s*;\s*\}",
        body_norm,
    ), (
        "the candidate publish must be wrapped in try { ... } catch (const std::exception &) { "
        'stop = "transaction refused"; candidate_lost = true; sched_matches_last_good = false; }'
    )


def test_candidate_lost_is_checked_immediately_after_the_try_catch():
    """The candidate_lost check for the publish's own try/catch must run
    BEFORE sched_need_reserve/sched_reserve() -- a refused publish must
    never reach a reserve for the plan it failed to publish."""
    body_norm = _normalize_ws(_trial_body())
    try_idx = body_norm.find("try { sycl_resync_runtime_context_flash_attn(); }")
    assert try_idx != -1, "could not find the candidate publish's try block"
    after = body_norm[try_idx:]
    break_idx = after.find("if (candidate_lost) { break; }")
    reserve_idx = after.find("sched_need_reserve = true;")
    assert break_idx != -1 and reserve_idx != -1
    assert break_idx < reserve_idx, (
        "the candidate_lost check for the publish's try/catch must precede sched_need_reserve/sched_reserve()"
    )


def test_candidate_publish_try_catch_has_a_mutation_witness():
    """Mutation witness for the two checks above: proves they would
    actually catch the try/catch being deleted (leaving a bare,
    unprotected publish call that would let a refusal escape as an
    uncaught exception instead of a clean "transaction refused" stop)."""
    raw = LLAMA_CONTEXT_CPP
    wrapped_block = (
        "        try {\n"
        "            sycl_resync_runtime_context_flash_attn();\n"
        "        } catch (const std::exception &) {\n"
        '            stop                    = "transaction refused";\n'
        "            candidate_lost          = true;\n"
        "            sched_matches_last_good = false;\n"
        "        }\n"
        "        if (candidate_lost) {\n"
        "            break;\n"
        "        }\n"
    )
    assert wrapped_block in raw, "mutation target not found -- update this witness to match the real source"
    mutated_raw = raw.replace(wrapped_block, "        sycl_resync_runtime_context_flash_attn();\n", 1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _TRIAL_START, _TRIAL_END)
    assert not re.search(
        r"try\s*\{\s*sycl_resync_runtime_context_flash_attn\(\s*\)\s*;\s*\}\s*catch\s*\(\s*const\s+std::exception\s*"
        r'&\s*\)\s*\{\s*stop\s*=\s*"transaction refused"\s*;\s*candidate_lost\s*=\s*true\s*;\s*'
        r"sched_matches_last_good\s*=\s*false\s*;\s*\}",
        mutated_body_norm,
    ), "mutation witness is broken: deleting the try/catch should make the wrapped-publish check fail"


def test_settle_publish_is_not_wrapped_in_try_catch():
    """Unlike the in-loop candidate publish, the SETTLE publish must let a
    refusal propagate -- today's behaviour for a context that does not fit
    at all (Task 2 final review addendum item 1: only the candidate publish
    gets the new try/catch)."""
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


def test_published_any_is_declared_false_and_set_true_after_the_in_loop_publish():
    """Q2b: published_any must start false and become true right after the
    in-loop publish succeeds (the try did not throw, so the candidate_lost
    check just above already returned false) -- tracking whether ANY
    candidate's publish actually took effect this trial run, independent
    of whether that candidate goes on to lose the host-fallback check."""
    body_norm = _normalize_ws(_trial_body())
    assert re.search(r"bool\s+published_any\s*=\s*false\s*;", body_norm), (
        "published_any must be declared, initialized false"
    )

    catch_idx = body_norm.find("sched_matches_last_good = false; } if (candidate_lost) { break; }")
    assert catch_idx != -1, "could not find the try/catch's closing candidate_lost check"
    published_any_idx = body_norm.find("published_any = true;", catch_idx)
    reserve_idx = body_norm.find("sched_need_reserve = true;", catch_idx)
    assert published_any_idx != -1 and reserve_idx != -1
    assert catch_idx < published_any_idx < reserve_idx, (
        "published_any must be set true strictly between the publish's own candidate_lost check and the "
        "in-loop sched_reserve() call"
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
