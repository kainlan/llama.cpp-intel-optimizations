"""Source contract for llama.cpp-o65k (plan docs/plans/2026-09-10-auto-ubatch.md,
Task 6): ggml/src/ggml-sycl/dispatch-tuning.cpp's dispatch-tuning JSON loader
must be opt-in via GGML_SYCL_DISPATCH_TUNING_JSON, not a hardcoded default path
tried -- and WARNed about -- on every model load.

Before this fix, `tuning_path()` fell back to the literal
"/tmp/onednn_unified_bench.json" whenever the env var was unset, and
`ensure_model_loaded()` always called `load_dispatch_tuning_from_file()`
against whatever `tuning_path()` returned. On a machine without that file
(essentially every run: `plan-research/tuning-cache.md` §3 counted 197
occurrences of the resulting "failed to load ... (unable to open file)"
GGML_LOG_WARN across committed logs), this meant a per-model-load filesystem
touch plus a scary-looking WARN that nobody could act on -- the mechanism
this file wants (`/tmp/onednn_unified_bench.json`, produced by
`sycl-kernel-bench --emit-json`) has no device or driver identity in its key
anyway, so applying it unconditionally across cards was never safe.

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
    assert re.search(r'return\s+std::string\(\s*\)\s*;\s*}\s*$', body) or body.rstrip().endswith(
        'return "";}'
    ), (
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
    assert "`GGML_SYCL_DISPATCH_TUNING" in SYCL_ENV_VARS_MD, (
        "docs/backend/sycl-env-vars.md must document GGML_SYCL_DISPATCH_TUNING"
    )
    assert "`GGML_SYCL_DISPATCH_TUNING_JSON" in SYCL_ENV_VARS_MD, (
        "docs/backend/sycl-env-vars.md must document GGML_SYCL_DISPATCH_TUNING_JSON"
    )
    assert "a JSON measured on one card is applied to every card" in SYCL_ENV_VARS_MD, (
        "the GGML_SYCL_DISPATCH_TUNING_JSON row must state the no-device/driver-identity "
        "caveat in these exact words, so the doc cannot silently drop it on a later edit"
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
