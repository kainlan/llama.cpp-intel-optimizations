"""Source contract for llama.cpp-o65k (plan docs/plans/2026-09-10-auto-ubatch.md,
Task 6): ggml/src/ggml-sycl/dispatch-tuning.cpp's dispatch-tuning JSON loader
must be opt-in via GGML_SYCL_DISPATCH_TUNING_JSON, not a hardcoded default path
tried -- and WARNed about -- on every model load.

Before this fix, `tuning_path()` fell back to the literal
"/tmp/onednn_unified_bench.json" whenever the env var was unset, and
`ensure_model_loaded()` always called `load_dispatch_tuning_from_file()`
against whatever `tuning_path()` returned. On a machine without that file
(essentially every run -- the WARN appears throughout the tracked bench logs
under `artifacts/perf-6ae16115c-longprompt/*.log`, self-verifying with
`grep -c 'dispatch tuning: failed to load' artifacts/perf-6ae16115c-longprompt/*.log`),
this meant a per-model-load filesystem touch plus a scary-looking WARN that
nobody could act on -- the mechanism this file wants
(`/tmp/onednn_unified_bench.json`, produced by `sycl-kernel-bench
--emit-json`) has no device or driver identity in its key anyway, so
applying it unconditionally across cards was never safe.

After this fix: unset/empty GGML_SYCL_DISPATCH_TUNING_JSON means "nothing to
load" -- no path returned, no file opened, no log line. A path that is set
keeps today's load behaviour unchanged, except a successful load now logs at
GGML_LOG_WARN (was GGML_LOG_INFO, which CLAUDE.md documents as dropped at
default verbosity in every tool -- an opt-in the caller explicitly asked for
should stay visible when it succeeds).

Host-only, pure text assertions -- no SYCL device required, matching
test-sycl-onednn-graph-allocator-source.py's pytest-collectible pattern
(llama_test_pytest hands this file to pytest.main(), so checks must live
inside a test_*() function or pytest's "no tests collected" (exit 5) is
scored as a failure by design, not a skip).

Checks run against COMMENT-STRIPPED text (see strip_comments(), copied from
test-sycl-nonfa-attn-scratch-guard-source.py) so a positive structural check
cannot be fooled by a comment that quotes a call the code does not actually
make.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DISPATCH_TUNING_CPP_PATH = ROOT / "ggml/src/ggml-sycl/dispatch-tuning.cpp"
DISPATCH_TUNING_CPP = DISPATCH_TUNING_CPP_PATH.read_text()
SYCL_ENV_VARS_MD = (ROOT / "docs/backend/sycl-env-vars.md").read_text()


# Copied verbatim from test-sycl-nonfa-attn-scratch-guard-source.py: single
# left-to-right alternation (not two sequential passes), so a `/*` inside a
# `//` comment's own prose cannot be mishandled -- order matters even though
# this specific file is not known to hit that case today.
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


def _normalize_ws(text: str) -> str:
    """Collapse all whitespace runs to a single space, so a call or
    declaration re-wrapped by clang-format across lines still matches a
    single-line pattern."""
    return re.sub(r"\s+", " ", text)


DISPATCH_TUNING_CPP_CODE = strip_comments(DISPATCH_TUNING_CPP)

# The default path literal this task removes. Held as one constant so the
# positive check and its own explanation cannot drift, and so this file
# itself never has to spell the literal a second time.
_DEFAULT_PATH_LITERAL = "/tmp/onednn_unified_bench.json"


def _bounded_body(code: str, start_marker: str, end_marker: str, label: str) -> str:
    """Slice `code` from `start_marker` to the next occurrence of
    `end_marker` after it, comment-stripped and whitespace-normalized --
    matching test-sycl-nonfa-attn-scratch-guard-source.py's bounded-body
    convention so a match cannot come from some unrelated later call site,
    and so a mutation witness can re-derive the same slice from a mutated
    copy of the raw source rather than hand-rolling its own bound logic."""
    start = code.find(start_marker)
    assert start != -1, f"{label}: start marker {start_marker!r} not found"
    end = code.find(end_marker, start + 1)
    assert end != -1, f"{label}: end marker {end_marker!r} not found after start"
    return _normalize_ws(code[start:end])


def _tuning_path_body(code: str = DISPATCH_TUNING_CPP_CODE) -> str:
    return _bounded_body(
        code,
        "std::string tuning_path() {",
        "struct ModelCache {",
        "tuning_path()",
    )


def _ensure_model_loaded_body(code: str = DISPATCH_TUNING_CPP_CODE) -> str:
    return _bounded_body(
        code,
        "void ensure_model_loaded(uint64_t model_id) {",
        "std::optional<ggml_sycl_mul_mat_kernel> lookup_kernel(",
        "ensure_model_loaded()",
    )


def test_no_default_path_literal_anywhere_in_the_file():
    """The default-path constant must be removed outright -- not just made
    conditional -- so nothing in this file can fall back to it, including a
    stray copy left in a comment or a log format string."""
    assert _DEFAULT_PATH_LITERAL not in DISPATCH_TUNING_CPP, (
        f"{_DEFAULT_PATH_LITERAL!r} must not appear anywhere in dispatch-tuning.cpp -- "
        "the opt-in fix removes the hardcoded default path entirely, it does not just "
        "gate reaching it"
    )


def test_tuning_path_returns_empty_when_unset():
    """tuning_path() must return an empty string (not the removed default
    literal, and not any other non-empty fallback) when
    GGML_SYCL_DISPATCH_TUNING_JSON is unset or empty -- that empty return is
    exactly the signal ensure_model_loaded() below must key its early return
    on."""
    body = _tuning_path_body()
    assert "GGML_SYCL_DISPATCH_TUNING_JSON" in body, (
        "tuning_path() must still read GGML_SYCL_DISPATCH_TUNING_JSON"
    )
    # llama.cpp-o65k quality round 1 (rev-o65k-qual-1, Q1): the previous
    # two-armed check's second arm (body.rstrip().endswith('return "";}'))
    # could never fire -- _normalize_ws() collapses the whitespace run
    # between `;` and `}` down to a single space, so a genuine `return "";`
    # implementation normalizes to `return ""; }` (single space before the
    # brace), not the no-space `"";}` the old arm looked for, and would fail
    # this gate despite being an accepted spelling the docstring promises.
    # One regex that actually accepts both spellings, whitespace-tolerant
    # throughout.
    assert re.search(r'return\s+(?:std::string\(\s*\)|"")\s*;\s*}\s*$', body), (
        "tuning_path()'s fallback return (env unset or empty) must be an empty string "
        "(std::string() or \"\"), not the removed default path"
    )


def test_ensure_model_loaded_early_returns_on_empty_path_before_load():
    """ensure_model_loaded() must check tuning_path() for emptiness and
    return BEFORE calling load_dispatch_tuning_from_file() -- an empty path
    must never reach the loader (no filesystem touch) and must never reach
    either GGML_LOG_WARN failure branch (no log line)."""
    body = _ensure_model_loaded_body()

    loaded_latch_idx = body.find("entry.loaded = true;")
    path_idx = body.find("tuning_path()")
    load_call_idx = body.find("load_dispatch_tuning_from_file(")
    assert loaded_latch_idx != -1, "ensure_model_loaded() must still set entry.loaded = true"
    assert path_idx != -1, "ensure_model_loaded() must still call tuning_path()"
    assert load_call_idx != -1, "ensure_model_loaded() must still call load_dispatch_tuning_from_file()"

    # The `loaded` latch must be set before the path is even read, so a
    # failed-to-load-because-unset model_id is not retried on the next call
    # (the acceptance criterion's "keep the `loaded` latch").
    assert loaded_latch_idx < path_idx, (
        "entry.loaded = true must be set before tuning_path() is consulted, so an "
        "empty-path model is not retried on a later call"
    )

    # An explicit emptiness check on the path, positioned between reading the
    # path and calling the loader.
    empty_check = re.search(r"if\s*\(\s*path\s*\.\s*empty\s*\(\s*\)\s*\)\s*{\s*return\s*;\s*}", body)
    assert empty_check is not None, (
        "ensure_model_loaded() must contain an explicit `if (path.empty()) { return; }` "
        "guard between reading tuning_path() and calling load_dispatch_tuning_from_file()"
    )
    assert path_idx < empty_check.start() < load_call_idx, (
        "the path.empty() guard must sit AFTER tuning_path() is called and BEFORE "
        "load_dispatch_tuning_from_file() -- a guard anywhere else cannot actually "
        "prevent the filesystem touch and the log line the acceptance criterion forbids"
    )

    # llama.cpp-i3x2 R3: the checks above pin the path.empty() guard's own
    # position but never establish that the !tuning_enabled() master-switch
    # check (top of the function) runs BEFORE tuning_path() is even called.
    # A reorder that read the path first and checked tuning_enabled() second
    # would satisfy every assertion above while still touching the
    # filesystem with the master switch off -- falsifying the
    # sycl-env-vars.md sentence pinned by test_docs_have_both_env_var_rows
    # below ("The path is only consulted while `GGML_SYCL_DISPATCH_TUNING`
    # is on; with the master switch off no load is attempted and neither
    # WARN appears.").
    tuning_enabled_idx = body.find("tuning_enabled()")
    assert tuning_enabled_idx != -1, "ensure_model_loaded() must still consult tuning_enabled()"
    assert tuning_enabled_idx < path_idx, (
        "the !tuning_enabled() master-switch check must be consulted BEFORE tuning_path() "
        "is read -- otherwise the master switch does not prevent the filesystem touch the "
        "sycl-env-vars.md doc promises it does"
    )


def test_early_return_guard_has_a_mutation_witness():
    """Mutation witness for the check above: proves it would actually catch
    a reversion that dropped the path.empty() guard, rather than only ever
    passing on the current, correct source. Mutates the RAW source (deleting
    the exact guard statement) and re-derives the bounded body through the
    same _ensure_model_loaded_body() accessor the positive check uses,
    rather than hand-building the already-normalized text."""
    guard_statement = "if (path.empty()) {\n        return;\n    }\n\n"
    assert guard_statement in DISPATCH_TUNING_CPP, (
        "mutation target not found verbatim -- update this witness to match the current "
        "guard's exact formatting"
    )
    mutated_raw = DISPATCH_TUNING_CPP.replace(guard_statement, "", 1)
    assert mutated_raw != DISPATCH_TUNING_CPP

    mutated_body = _ensure_model_loaded_body(strip_comments(mutated_raw))
    empty_check = re.search(
        r"if\s*\(\s*path\s*\.\s*empty\s*\(\s*\)\s*\)\s*{\s*return\s*;\s*}", mutated_body
    )
    assert empty_check is None, (
        "mutation witness is broken: the reverted (guard-deleted) source still trips the "
        "positive check above"
    )


def test_master_switch_ordering_has_a_mutation_witness():
    """Mutation witness for the tuning_enabled()-before-tuning_path() check
    above: proves it would actually catch a reorder that moved the
    master-switch check to AFTER the path read, rather than only ever
    passing on the current, correct source. Mutates the RAW source (moving
    the exact guard statement from the top of the function to immediately
    after the path read) and re-derives the bounded body through the same
    _ensure_model_loaded_body() accessor the positive check uses, rather
    than hand-building the already-normalized text."""
    guard_statement = "    if (!tuning_enabled() || model_id == 0) {\n        return;\n    }\n\n"
    path_read_statement = "    const std::string path = tuning_path();\n"
    assert guard_statement in DISPATCH_TUNING_CPP, (
        "mutation target not found verbatim -- update this witness to match the current "
        "guard's exact formatting"
    )
    assert path_read_statement in DISPATCH_TUNING_CPP, (
        "mutation target not found verbatim -- update this witness to match the current "
        "path-read statement's exact formatting"
    )

    mutated_raw = DISPATCH_TUNING_CPP.replace(guard_statement, "", 1)
    assert mutated_raw != DISPATCH_TUNING_CPP
    mutated_raw = mutated_raw.replace(
        path_read_statement, path_read_statement + guard_statement, 1
    )
    assert mutated_raw != DISPATCH_TUNING_CPP

    mutated_body = _ensure_model_loaded_body(strip_comments(mutated_raw))
    mutated_tuning_enabled_idx = mutated_body.find("tuning_enabled()")
    mutated_path_idx = mutated_body.find("tuning_path()")
    assert mutated_tuning_enabled_idx != -1, (
        "mutation witness is broken: ensure_model_loaded() no longer consults "
        "tuning_enabled() after the mutation"
    )
    assert mutated_path_idx != -1, (
        "mutation witness is broken: ensure_model_loaded() no longer calls tuning_path() "
        "after the mutation"
    )
    assert mutated_tuning_enabled_idx > mutated_path_idx, (
        "mutation witness is broken: moving the guard after the path read did not actually "
        "reorder tuning_enabled() after tuning_path() in the bounded body -- the positive "
        "check above would not catch this reversion"
    )


def test_success_log_is_warn_not_info():
    """The successful-load log line must be GGML_LOG_WARN, not
    GGML_LOG_INFO -- CLAUDE.md documents GGML_LOG_INFO as dropped at default
    verbosity in every tool, and a successful load is now an explicit opt-in
    the caller asked for, so it must stay visible."""
    body = _ensure_model_loaded_body()
    success_idx = body.find('"[SYCL] dispatch tuning: loaded %zu entries from %s for model=%llu')
    assert success_idx != -1, "the success log's format string was not found in ensure_model_loaded()"
    # The macro name is the nearest preceding identifier ending in
    # `GGML_LOG_...` before the format string's opening quote.
    preceding = body[:success_idx]
    macro_match = re.search(r"(GGML_LOG_\w+)\s*\(\s*$", preceding)
    assert macro_match is not None, "could not find the logging macro immediately preceding the success format string"
    assert macro_match.group(1) == "GGML_LOG_WARN", (
        f"the successful-load log must be GGML_LOG_WARN, found {macro_match.group(1)!r}"
    )

    # The two pre-existing failure branches must remain GGML_LOG_WARN
    # (unchanged) -- this task only touches the success line's level.
    assert (
        'GGML_LOG_WARN("[SYCL] dispatch tuning: no entries loaded from %s'
        in body
    ), "the empty-cache failure WARN must remain unchanged"
    assert (
        'GGML_LOG_WARN("[SYCL] dispatch tuning: failed to load %s (%s)'
        in body
    ), "the I/O-failure WARN must remain unchanged"


def test_dispatch_tuning_master_switch_is_unchanged():
    """GGML_SYCL_DISPATCH_TUNING (the tuning_enabled() master switch,
    default on) is explicitly out of scope for this task -- pin that its
    parse is untouched, so a future edit here notices if it drifts."""
    body = _bounded_body(
        DISPATCH_TUNING_CPP_CODE,
        "bool tuning_enabled() {",
        "std::string tuning_path() {",
        "tuning_enabled()",
    )
    assert "GGML_SYCL_DISPATCH_TUNING" in body
    assert "std::atoi(env) != 0" in body, (
        "tuning_enabled()'s default-on atoi(env) != 0 parse must be unchanged"
    )


def test_docs_have_both_env_var_rows():
    """docs/backend/sycl-env-vars.md must carry one row each for
    GGML_SYCL_DISPATCH_TUNING and GGML_SYCL_DISPATCH_TUNING_JSON, and the
    JSON-path row must state the caveat that its lookup key carries no
    device or driver identity -- a JSON measured on one card is applied to
    every card."""
    # llama.cpp-o65k round 1 (rev-o65k-spec-1, finding F1): the bare prefix
    # "`GGML_SYCL_DISPATCH_TUNING" is also a substring of the JSON row's own
    # name ("`GGML_SYCL_DISPATCH_TUNING_JSON=<path>`"), so that needle alone
    # is satisfied by the JSON row even with the master-switch row entirely
    # deleted -- pin the leading "| `...=0`" table-row prefix instead, which
    # only the master-switch row can produce.
    assert "| `GGML_SYCL_DISPATCH_TUNING=0`" in SYCL_ENV_VARS_MD, (
        "docs/backend/sycl-env-vars.md must document GGML_SYCL_DISPATCH_TUNING as its own "
        "table row"
    )
    assert "`GGML_SYCL_DISPATCH_TUNING_JSON" in SYCL_ENV_VARS_MD, (
        "docs/backend/sycl-env-vars.md must document GGML_SYCL_DISPATCH_TUNING_JSON"
    )
    assert "a JSON measured on one card is applied to every card" in SYCL_ENV_VARS_MD, (
        "the GGML_SYCL_DISPATCH_TUNING_JSON row must state the no-device/driver-identity "
        "caveat in these exact words, so the doc cannot silently drop it on a later edit"
    )
    # llama.cpp-o65k round 1 (rev-o65k-spec-1, finding F2): the JSON-path
    # row must also document that a whitespace-only value is NOT treated as
    # unset -- it is a real path that fails to open, WARNing once per model
    # (dispatch-tuning.cpp's `if (env && env[0])` guard sees a lone " " as
    # non-empty).
    assert (
        "An empty value is unset; a whitespace-only value is treated as a path and fails "
        "to open with one WARN per model." in SYCL_ENV_VARS_MD
    ), (
        "the GGML_SYCL_DISPATCH_TUNING_JSON row must state the whitespace-only-value "
        "behaviour in these words, so the doc cannot silently drop it on a later edit"
    )
    # llama.cpp-o65k quality round 1 (rev-o65k-qual-1, Q4): the row's "one
    # WARN per model" claims are only true while GGML_SYCL_DISPATCH_TUNING
    # is on -- with the master switch off, ensure_model_loaded() returns
    # before tuning_path() is even consulted, so neither WARN appears and no
    # load is attempted. Pin that qualifier so the doc cannot silently drop
    # it on a later edit.
    assert (
        "The path is only consulted while `GGML_SYCL_DISPATCH_TUNING` is on; with the "
        "master switch off no load is attempted and neither WARN appears." in SYCL_ENV_VARS_MD
    ), (
        "the GGML_SYCL_DISPATCH_TUNING_JSON row must state that the path is only "
        "consulted while GGML_SYCL_DISPATCH_TUNING is on"
    )


def test_readme_no_longer_advertises_the_bare_tmp_path_as_sufficient():
    """tools/sycl-kernel-bench/README.md's --emit-json example must tell the
    reader that the resulting file now has to be passed via
    GGML_SYCL_DISPATCH_TUNING_JSON -- writing to the old default path is no
    longer enough on its own to have it picked up."""
    readme = (ROOT / "tools/sycl-kernel-bench/README.md").read_text()
    assert "GGML_SYCL_DISPATCH_TUNING_JSON" in readme, (
        "tools/sycl-kernel-bench/README.md must mention GGML_SYCL_DISPATCH_TUNING_JSON "
        "now that the loader no longer tries a default path automatically"
    )
