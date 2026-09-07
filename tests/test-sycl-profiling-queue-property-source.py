#!/usr/bin/env python3
"""Source gate for llama.cpp-yke2. Full mechanism: see
ggml/src/ggml-sycl/dpct/helper.hpp's create_queue_impl() comment (the
canonical explanation of why sycl::property::queue::enable_profiling() must
be unconditional, not `#ifdef DPCT_PROFILING_ENABLED`) and
tests/test-sycl-profiling-queue-property.cpp's header comment (why that
GPU test has two phases). This gate checks two fixes at the SOURCE TEXT
level only, without touching a SYCL device: (1) both create_queue_impl()
overloads in dpct/helper.hpp add the property unconditionally, with the
macro itself no longer defined for the ggml-sycl target; (2)
unified-cache.cpp's create_cache_for_device() calls
ensure_single_device_context_queue() unconditionally, not gated on
`total_gpus > 1`.

Runs under pytest (llama_test_pytest registration) and as a plain script.
Point it at alternate copies (to exercise the RED path against a deliberately
unmodified tree) via GGML_SYCL_YKE2_UNIFIED_CACHE_SOURCE,
GGML_SYCL_YKE2_DPCT_HELPER_SOURCE and GGML_SYCL_YKE2_GGML_SYCL_CMAKE_SOURCE;
the defaults are the in-tree files.
"""

import os
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(
    os.environ.get(
        "GGML_SYCL_YKE2_UNIFIED_CACHE_SOURCE",
        str(ROOT / "ggml/src/ggml-sycl/unified-cache.cpp"),
    )
)
HELPER_SOURCE = Path(
    os.environ.get(
        "GGML_SYCL_YKE2_DPCT_HELPER_SOURCE",
        str(ROOT / "ggml/src/ggml-sycl/dpct/helper.hpp"),
    )
)
SYCL_CMAKE_SOURCE = Path(
    os.environ.get(
        "GGML_SYCL_YKE2_GGML_SYCL_CMAKE_SOURCE",
        str(ROOT / "ggml/src/ggml-sycl/CMakeLists.txt"),
    )
)

source = SOURCE.read_text(encoding="utf-8")
helper_source = HELPER_SOURCE.read_text(encoding="utf-8")
sycl_cmake_source = SYCL_CMAKE_SOURCE.read_text(encoding="utf-8")

FUNC_SIG = "static unified_cache * create_cache_for_device(int"
ENSURE_CALL = "ensure_single_device_context_queue(device_id)"
# The exact buggy pattern this gate must never see reappear (llama.cpp-yke2):
# gating the single-device-context queue behind more than one visible GPU
# left dpct's raw, interposition-vulnerable default_queue() as the single-GPU
# cache-owner queue.
BUGGY_GATE_RE = re.compile(r"!\s*cache_queue\s*&&\s*total_gpus\s*>\s*1")

CREATE_QUEUE_IMPL_SIG = "sycl::queue create_queue_impl("
ENABLE_PROFILING_CALL = "sycl::property::queue::enable_profiling()"
DPCT_PROFILING_MACRO = "DPCT_PROFILING_ENABLED"

# The commit this branch (task/S7) started from -- confirmed pre-fix for all
# three files these checks read. Used only by the automated positive control
# below (test_positive_control_gate_can_fail_on_the_known_pre_fix_revision),
# so the RED half of RED->GREEN is re-verified on every run rather than
# resting on a one-time manual check against a hand-saved scratch copy
# (lead review requirement, llama.cpp-yke2).
KNOWN_PRE_FIX_REVISION = "c4f25b31c"


def matching_brace(text, open_idx):
    """Comment/string-aware brace match (self-contained copy, per this
    fork's one-file-per-gate convention -- see
    tests/test-sycl-extra-leak-probe-source.py)."""
    assert text[open_idx] == "{"
    depth = 0
    state = "code"
    i = open_idx
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if state == "code":
            if ch == "/" and nxt == "/":
                state = "line"
                i += 2
                continue
            if ch == "/" and nxt == "*":
                state = "block"
                i += 2
                continue
            if ch == '"':
                state = "str"
            elif ch == "'":
                prev = text[i - 1] if i > 0 else ""
                if not (prev.isalnum() or prev == "_"):
                    state = "chr"
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i
        elif state == "line":
            if ch == "\n":
                state = "code"
        elif state == "block":
            if ch == "*" and nxt == "/":
                state = "code"
                i += 2
                continue
        elif state == "str":
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                state = "code"
        elif state == "chr":
            if ch == "\\":
                i += 2
                continue
            if ch == "'":
                state = "code"
        i += 1
    raise AssertionError("unbalanced braces")


def function_body(text, signature, start=0):
    idx = text.find(signature, start)
    assert idx >= 0, f"missing definition: {signature}"
    open_idx = text.find("{", idx)
    assert open_idx >= 0, f"no opening brace found after {signature}"
    return text[open_idx : matching_brace(text, open_idx) + 1], idx


def test_create_cache_for_device_defined_exactly_once():
    assert source.count(FUNC_SIG) == 1, "create_cache_for_device must be defined exactly once"


def matching_paren(text, open_idx):
    """Comment/string-aware matching ')' for the '(' at open_idx
    (self-contained copy, per this fork's one-file-per-gate convention --
    see tests/test-sycl-extra-leak-probe-source.py's sibling helper). Needed
    because an if-condition is not guaranteed to be paren-free, and a naive
    `text.find(")", ...)` would stop at the first ')' inside a nested
    expression."""
    assert text[open_idx] == "("
    depth = 0
    state = "code"
    i = open_idx
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if state == "code":
            if ch == "/" and nxt == "/":
                state = "line"
                i += 2
                continue
            if ch == "/" and nxt == "*":
                state = "block"
                i += 2
                continue
            if ch == '"':
                state = "str"
            elif ch == "'":
                prev = text[i - 1] if i > 0 else ""
                if not (prev.isalnum() or prev == "_"):
                    state = "chr"
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return i
        elif state == "line":
            if ch == "\n":
                state = "code"
        elif state == "block":
            if ch == "*" and nxt == "/":
                state = "code"
                i += 2
                continue
        elif state == "str":
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                state = "code"
        elif state == "chr":
            if ch == "\\":
                i += 2
                continue
            if ch == "'":
                state = "code"
        i += 1
    raise AssertionError("unbalanced parens")


# Word-boundary aware (`(?<![A-Za-z0-9_])`) so this cannot match `if (` as a
# substring of a longer identifier immediately followed by ` (` (e.g. a
# hypothetical `motif (...)` call).
_IF_RE = re.compile(r"(?<![A-Za-z0-9_])if\s*\(")


def _find_enclosing_if_condition(body, call_idx):
    """The whitespace-stripped condition of the SMALLEST `if ( ... )` in
    `body` whose body (a `{ ... }` block, or a single-statement body up to
    the next top-level `;`) actually CONTAINS call_idx, or None if no such
    enclosing if exists (a genuinely unconditional call).

    Deliberately NOT "the nearest preceding `if (`" (quality review c-69sf,
    nit 1): that textual-proximity approach reports a call as guarded by an
    UNRELATED `if` earlier in the same function, as long as that if's own
    body ends before the call -- a genuinely unconditional call preceded by
    an unrelated `if (something_else) { ... }` would be misreported as
    guarded by `something_else` and this gate would false-fail. This scans
    EVERY `if (` in the body and only accepts one whose body span actually
    contains the call, picking the innermost (latest-starting) match among
    any that do (nested ifs)."""
    best = None  # (body_start, condition) for the tightest enclosing if found so far
    for m in _IF_RE.finditer(body):
        if_start = m.start()
        if if_start >= call_idx:
            continue  # this if starts at or after the call -- cannot enclose it
        cond_open = m.end() - 1
        cond_close = matching_paren(body, cond_open)
        condition = body[cond_open + 1 : cond_close]
        rest = body[cond_close + 1 :]
        stripped_rest = rest.lstrip()
        after_ws = cond_close + 1 + (len(rest) - len(stripped_rest))
        if body[after_ws : after_ws + 1] == "{":
            body_start = after_ws
            body_end = matching_brace(body, body_start)
        else:
            # Single-statement body (no braces): up to the next top-level ';'.
            semi_idx = body.find(";", after_ws)
            if semi_idx < 0:
                continue
            body_start, body_end = after_ws, semi_idx
        if body_start < call_idx <= body_end and (best is None or body_start > best[0]):
            best = (body_start, condition)
    if best is None:
        return None
    return re.sub(r"\s+", "", best[1])


def _ensure_call_guard_condition(body):
    """The whitespace-stripped condition of the smallest `if ( ... )`
    actually enclosing ENSURE_CALL in `body`, or None if the call is
    genuinely unconditional. Thin wrapper over
    _find_enclosing_if_condition() that locates the call itself. Shared by
    the GREEN check below and the positive control's complement check on
    the pre-fix revision."""
    idx = body.find(ENSURE_CALL)
    assert idx >= 0, f"expected {ENSURE_CALL} to be called"
    return _find_enclosing_if_condition(body, idx)


def test_guard_condition_extraction_is_structural_not_textual_proximity():
    # Regression test for the extraction LOGIC itself (quality review
    # c-69sf, nit 1), independent of whatever unified-cache.cpp currently
    # contains -- durable the same way the git-show positive control below
    # is durable, in the "positive control" style: synthetic minimal
    # function bodies standing in for the three mutants the reviewer
    # actually tried against a naive nearest-preceding-`if` implementation.
    unrelated_if_before_unconditional_call = """{
        if (something_else) {
            do_other_stuff();
        }
        ensure_single_device_context_queue(device_id);
    }"""
    assert _ensure_call_guard_condition(unrelated_if_before_unconditional_call) is None, (
        "an unrelated `if` earlier in the function, whose body does not contain the call, must not be "
        "reported as guarding it -- the call here is genuinely unconditional"
    )

    regated_on_something_else = """{
        if (!cache_queue && dev_count > 1) {
            cache_queue = ensure_single_device_context_queue(device_id);
        }
    }"""
    assert _ensure_call_guard_condition(regated_on_something_else) == "!cache_queue&&dev_count>1", (
        "expected the full condition to be extracted, not just the absence of a specific substring"
    )

    correctly_fixed = """{
        if (!cache_queue) {
            cache_queue = ensure_single_device_context_queue(device_id);
        }
    }"""
    assert _ensure_call_guard_condition(correctly_fixed) == "!cache_queue", (
        "expected the shipped tree's guard pattern to extract exactly `!cache_queue`"
    )


def test_ensure_single_device_context_queue_is_called_unconditionally():
    # Exact-equality check, not merely "total_gpus is absent": a re-gate
    # spelled with a different name (e.g. `dev_count > 1` or
    # `g_total_gpu_count > 1`) would satisfy an absence-of-"total_gpus"
    # check while reintroducing exactly the bug this test exists to catch
    # (spec review c-a1xt, should-fix 1). `!cache_queue` (queue_override not
    # already supplied) is the one condition allowed to remain, because it
    # is not a GPU-count gate; no enclosing `if` at all is equally
    # acceptable (genuinely unconditional).
    body, _ = function_body(source, FUNC_SIG)
    condition = _ensure_call_guard_condition(body)
    if condition is None:
        return
    assert condition == "!cache_queue", (
        f"the {ENSURE_CALL} call must be guarded by exactly `if (!cache_queue)` (queue_override not already "
        f"supplied) or by no enclosing if at all -- found condition `if ({condition})`, which re-gates the "
        "call on something else (e.g. a GPU-count check) and would leave a single visible GPU falling back "
        "to dpct's interposition-vulnerable default_queue() again"
    )


def test_buggy_multi_gpu_gate_pattern_is_absent():
    # Regression guard for the exact pre-fix pattern, independent of the
    # structural check above: `!cache_queue && total_gpus > 1` must not
    # reappear anywhere in this function (e.g. reintroduced via a rebase or
    # a copy-paste from the historical multi-GPU-only code path).
    body, _ = function_body(source, FUNC_SIG)
    assert not BUGGY_GATE_RE.search(body), (
        "found the pre-fix gate pattern `!cache_queue && total_gpus > 1` -- "
        "ensure_single_device_context_queue() must be called unconditionally (llama.cpp-yke2)"
    )


def test_fallback_to_raw_default_queue_still_exists_for_construction_failure():
    # The fix removes the total_gpus > 1 GATE, not the fallback itself:
    # ensure_single_device_context_queue() can still legitimately return
    # nullptr if queue construction throws, and create_cache_for_device must
    # still degrade to the raw default_queue() in that case rather than
    # crashing on a null dereference.
    body, _ = function_body(source, FUNC_SIG)
    assert "ggml_sycl_get_device(device_id).default_queue()" in body, (
        "create_cache_for_device must keep the default_queue() fallback for when "
        "ensure_single_device_context_queue() fails to construct a queue"
    )


def _create_queue_impl_bodies():
    bodies = []
    start = 0
    for _ in range(2):
        body, idx = function_body(helper_source, CREATE_QUEUE_IMPL_SIG, start)
        bodies.append(body)
        start = idx + len(CREATE_QUEUE_IMPL_SIG)
    return bodies


def test_create_queue_impl_defined_exactly_twice():
    # The two dpct::device_ext::create_queue_impl() overloads (with and
    # without an explicit sycl::device parameter) -- both must be fixed, not
    # just one, since dpct::device_ext::default_queue() and other call sites
    # can reach either.
    assert helper_source.count(CREATE_QUEUE_IMPL_SIG) == 2, (
        "expected exactly two create_queue_impl(...) overloads in dpct/helper.hpp"
    )


def test_create_queue_impl_enables_profiling_unconditionally():
    for body in _create_queue_impl_bodies():
        assert ENABLE_PROFILING_CALL in body, (
            f"create_queue_impl must unconditionally add {ENABLE_PROFILING_CALL} to its property_list "
            "(llama.cpp-yke2) -- a macro-gated `#ifdef DPCT_PROFILING_ENABLED` cannot be interposition-safe "
            "in a header-only inline"
        )
        assert DPCT_PROFILING_MACRO not in body, (
            f"create_queue_impl must not read {DPCT_PROFILING_MACRO} at all -- the property must be "
            "unconditional source code, not gated on a macro that depends on which TU's copy of this "
            "header-only inline the linker resolves"
        )


# Anchored to the START of a (stripped) line, so this cannot match the macro
# name spelled out inside a `//` explanatory comment (e.g. this very fix's own
# comment, which legitimately quotes the historical `#ifdef ...` directive it
# removed) -- only an actual conditional-compilation directive line counts as
# the live, interposition-vulnerable mechanism this gate must catch. Handled
# in two steps rather than one regex (spec review c-a1xt, nit 5): first find
# every #ifdef/#ifndef/#if/#elif line at all, then check WITHIN that line
# whether it actually reads DPCT_PROFILING_MACRO -- a single anchored regex
# for `#if defined(MACRO)` alone missed `#if !defined(MACRO)` (no bare
# `defined MACRO` sequence -- the `!` sits between them) and
# `#if defined(X) && defined(MACRO)` (MACRO is not the FIRST defined(...) on
# the line).
_PREPROC_CONDITIONAL_LINE_RE = re.compile(r"^[ \t]*#[ \t]*(ifdef|ifndef|if|elif)\b(.*)$", re.MULTILINE)
_DEFINED_MACRO_RE = re.compile(r"defined\s*\(?\s*" + re.escape(DPCT_PROFILING_MACRO) + r"\b")


def _directive_reads_macro(directive, rest_of_line):
    if directive in ("ifdef", "ifndef"):
        # #ifdef DPCT_PROFILING_ENABLED / #ifndef DPCT_PROFILING_ENABLED --
        # the macro name follows the directive keyword directly, bare, no
        # defined(...) wrapper.
        return re.match(r"^\s*" + re.escape(DPCT_PROFILING_MACRO) + r"\b", rest_of_line) is not None
    # #if / #elif -- the macro may appear anywhere on the line inside a
    # defined(...) or `defined NAME` construct: `#if defined(MACRO)`,
    # `#if !defined(MACRO)`, `#if defined(X) && defined(MACRO)` all match.
    return _DEFINED_MACRO_RE.search(rest_of_line) is not None


def _find_live_macro_directive(text):
    """The re.Match for the first #ifdef/#ifndef/#if/#elif line that reads
    DPCT_PROFILING_MACRO, or None if no such live directive exists."""
    for m in _PREPROC_CONDITIONAL_LINE_RE.finditer(text):
        if _directive_reads_macro(m.group(1), m.group(2)):
            return m
    return None


def test_dpct_profiling_enabled_macro_is_gone_from_helper_hpp():
    # A preprocessor-directive check, not a blanket string search: the macro
    # NAME may legitimately still appear in an explanatory comment (e.g.
    # describing the historical bug this fix removes), which is not the
    # live, interposition-vulnerable mechanism this gate must catch --
    # only an actual #ifdef/#ifndef/#if/#elif directive reading it is.
    match = _find_live_macro_directive(helper_source)
    assert not match, (
        f"found a live preprocessor directive reading {DPCT_PROFILING_MACRO} in dpct/helper.hpp "
        f"({match.group(0)!r}) -- the property must be unconditional source code (llama.cpp-yke2)"
    )


def test_ggml_sycl_cmake_no_longer_defines_dpct_profiling_enabled():
    assert "target_compile_definitions(ggml-sycl PRIVATE DPCT_PROFILING_ENABLED)" not in sycl_cmake_source, (
        f"ggml-sycl/CMakeLists.txt must no longer define {DPCT_PROFILING_MACRO} for the ggml-sycl target -- "
        "it is dead once create_queue_impl() no longer reads it, and leaving it would misleadingly imply "
        "the property is still opt-in/compile-time-gated (llama.cpp-yke2)"
    )


# _git_show raises unittest.SkipTest (not a bespoke exception) when there is
# no .git directory, so this test is skipped, not failed, under EITHER
# runner without importing pytest directly: pytest natively converts a
# raised unittest.SkipTest into a skip outcome (its documented interop point
# for non-pytest test code), and the plain-script runner at the bottom of
# this file catches it explicitly as SKIP.
def _git_show(path):
    import subprocess

    if not (ROOT / ".git").exists():
        raise unittest.SkipTest(
            f"no .git directory at {ROOT} -- cannot fetch {KNOWN_PRE_FIX_REVISION}:{path} for the positive control"
        )
    proc = subprocess.run(
        ["git", "show", f"{KNOWN_PRE_FIX_REVISION}:{path}"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"git show {KNOWN_PRE_FIX_REVISION}:{path} failed (rc={proc.returncode}): {proc.stderr}"
    )
    return proc.stdout


def test_positive_control_gate_can_fail_on_the_known_pre_fix_revision():
    # RED counterpart to every GREEN check above, re-run automatically on
    # every invocation of this file rather than resting on a one-time manual
    # check against a hand-saved scratch copy (lead review requirement,
    # llama.cpp-yke2): fetches the three pre-fix blobs via `git show
    # <KNOWN_PRE_FIX_REVISION>:<path>` and re-applies the same structural
    # assertions this file uses for GREEN, confirming each is actually
    # capable of catching its own bug -- an assertion that can never fail is
    # not a check. Skipped, not failed, when there is no .git directory to
    # fetch the pre-fix revision from (nit from spec review c-wbm2).
    pre_unified_cache = _git_show("ggml/src/ggml-sycl/unified-cache.cpp")
    pre_helper = _git_show("ggml/src/ggml-sycl/dpct/helper.hpp")
    pre_sycl_cmake = _git_show("ggml/src/ggml-sycl/CMakeLists.txt")

    # 1. unified-cache.cpp: create_cache_for_device must still carry the
    #    total_gpus > 1 gate at the pre-fix revision.
    pre_cache_body, _ = function_body(pre_unified_cache, FUNC_SIG)
    assert BUGGY_GATE_RE.search(pre_cache_body), (
        f"expected the pre-fix gate pattern in {KNOWN_PRE_FIX_REVISION}'s create_cache_for_device -- "
        "if this no longer fails, the positive control itself is void (llama.cpp-yke2)"
    )
    # Complement of the EXACT-EQUALITY structural check
    # (_ensure_call_guard_condition / test_ensure_single_device_context_queue_
    # is_called_unconditionally): on the pre-fix revision the guarding
    # condition must NOT already be exactly `!cache_queue` -- otherwise that
    # GREEN check's own positive control would be void too (spec review
    # c-a1xt, nit 6).
    pre_condition = _ensure_call_guard_condition(pre_cache_body)
    assert pre_condition != "!cache_queue", (
        f"expected {KNOWN_PRE_FIX_REVISION}'s guarding condition to NOT already be exactly `!cache_queue` "
        f"(found {pre_condition!r}) -- if it is, the exact-equality structural check's positive control is void"
    )

    # 2. helper.hpp: both create_queue_impl overloads must still read the
    #    macro, and a live directive must still be present, at the pre-fix
    #    revision.
    assert pre_helper.count(CREATE_QUEUE_IMPL_SIG) == 2, (
        f"expected exactly two create_queue_impl(...) overloads in {KNOWN_PRE_FIX_REVISION}'s dpct/helper.hpp"
    )
    start = 0
    for _ in range(2):
        pre_impl_body, idx = function_body(pre_helper, CREATE_QUEUE_IMPL_SIG, start)
        assert DPCT_PROFILING_MACRO in pre_impl_body, (
            f"expected {DPCT_PROFILING_MACRO} still gating create_queue_impl in {KNOWN_PRE_FIX_REVISION}'s "
            "dpct/helper.hpp -- if this no longer fails, the positive control itself is void"
        )
        start = idx + len(CREATE_QUEUE_IMPL_SIG)
    assert _find_live_macro_directive(pre_helper), (
        f"expected a live #ifdef {DPCT_PROFILING_MACRO} directive in {KNOWN_PRE_FIX_REVISION}'s dpct/helper.hpp"
    )

    # 3. ggml-sycl/CMakeLists.txt: the compile definition must still be
    #    present at the pre-fix revision.
    assert "target_compile_definitions(ggml-sycl PRIVATE DPCT_PROFILING_ENABLED)" in pre_sycl_cmake, (
        f"expected the DPCT_PROFILING_ENABLED compile definition in {KNOWN_PRE_FIX_REVISION}'s "
        "ggml-sycl/CMakeLists.txt -- if this no longer fails, the positive control itself is void"
    )


if __name__ == "__main__":
    import sys

    failures = 0
    for fn_name, fn in sorted(list(globals().items())):
        if fn_name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {fn_name}")
            except unittest.SkipTest as exc:
                # Not a failure: see _git_show's no-.git-directory case.
                print(f"SKIP {fn_name}: {exc}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {fn_name}: {exc}")
    if failures:
        print(f"{failures} test(s) failed")
        sys.exit(1)
    print("all tests passed")
