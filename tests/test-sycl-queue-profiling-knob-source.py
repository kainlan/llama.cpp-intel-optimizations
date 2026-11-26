#!/usr/bin/env python3
"""Source gate for the GGML_SYCL_QUEUE_PROFILING knob (three states, default off).

Every queue of the SYCL backend is built WITHOUT sycl::property::queue::enable_profiling by default. On the Level Zero
v2 adapter a profiling queue reads a device timestamp for every command, and only diagnostics read the result. The knob
GGML_SYCL_QUEUE_PROFILING has three states: unset is off unless a profiling diagnostic is requested in the environment
(then on, with one WARN naming the variable); "1" is on; "0" is off and a requested diagnostic prints its refusal WARN
and no figure; any other value is one WARN and then counts as unset. The answer is resolved once and never changes.
This gate reads the source text only (no device, no build) and pins:

  P  the resolver's shape: the pure decision function in common.cpp touches no environment, static, log or SYCL; the
     knob's environment variable is read exactly once in the backend; the predicate is declared out of line in
     queue-profiling.hpp (never an inline, never a macro: every translation unit would compile its own copy). What the
     resolver answers, how often it reads the environment and which WARN lines it prints are behaviour, run by the
     forked arms of test-sycl-queue-profiling-resolver instead of read from the source here;
  L  the one list of diagnostic variable names in queue-profiling.hpp: every warn_unless_readable() call names exactly
     one entry of it, and the resolver has a parse branch for every entry;
  E  each diagnostic's own parse of its own variable is the one the resolver uses for it (a diagnostic that considers
     itself enabled while the resolver does not count it as requested would run with queues that carry no profiling);
  Q  the three places that add enable_profiling (both create_queue_impl overloads in dpct/helper.hpp and
     default_queue_properties() in common.hpp) go through the predicate, keep today's list and order when it is true,
     and leave the property out when it is false; no other backend code names enable_profiling;
  R  every get_profiling_info call in non-test backend code sits in a function that first consults the predicate (a
     query on an event of a queue without profiling throws, and a diagnostic must report no figure instead);
  D  the diagnostics whose enabling predicate gates every figure they print close when the queues carry no profiling;
  docs  the two docs rows say default off and name every diagnostic.

Runs under pytest (llama_test_pytest registration) and as a plain script. Each pin has a mutant that must fail it;
the mutants run inside the tests, on in-memory copies of the sources.
"""
from __future__ import annotations

import bisect
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BACKEND = ROOT / "ggml" / "src" / "ggml-sycl"

PREDICATE_HEADER = "ggml/src/ggml-sycl/queue-profiling.hpp"
HELPER = "ggml/src/ggml-sycl/dpct/helper.hpp"
COMMON_HPP = "ggml/src/ggml-sycl/common.hpp"
COMMON_CPP = "ggml/src/ggml-sycl/common.cpp"
MMVQ = "ggml/src/ggml-sycl/mmvq.cpp"
GGML_SYCL = "ggml/src/ggml-sycl/ggml-sycl.cpp"
PROFILER = "ggml/src/ggml-sycl/sycl-kernel-profiler.cpp"
PLAN = "ggml/src/ggml-sycl/moe-stream-plan.cpp"

PREDICATE = "ggml_sycl_queue_profiling_enabled"
READABLE = "ggml_sycl_queue_profiling_warn_unless_readable"
WARN = "ggml_sycl_queue_profiling_warn_unavailable"
DECIDE = "ggml_sycl_queue_profiling_decide"
REQUESTED = "ggml_sycl_queue_profiling_requested_diagnostic"
DIAG_REQUESTED = "ggml_sycl_queue_profiling_diagnostic_requested"
LIST = "ggml_sycl_queue_profiling_diagnostics"
GUARD_RE = re.compile(r"\b(?:%s|%s)\s*\(" % (PREDICATE, READABLE))
ENV_NAME = "GGML_SYCL_QUEUE_PROFILING"


# The parse each diagnostic applies to its own variable. The resolver's branch and every consumer use the same helper,
# so the two cannot disagree. GGML_SYCL_MOE_EXEC_TRACE is parsed by moe_exec_config_parse (moe-exec-policy.hpp), and
# only level 2 reads event profiling (the ring windows' timing), so the resolver asks for level 2.
HELPER_FOR = {
    "GGML_SYCL_KERNEL_PROFILE":    "ggml_sycl_profile_env_flag",
    "GGML_SYCL_MXFP4_TG_PROFILE":  "ggml_sycl_profile_env_nonzero",
    "GGML_SYCL_MXFP4_PP_PROFILE":  "ggml_sycl_profile_env_nonzero",
    "GGML_SYCL_ATTN_SYNC_PROFILE": "ggml_sycl_profile_env_set",
}
CONSUMERS = {
    "GGML_SYCL_KERNEL_PROFILE":    (PROFILER,),
    "GGML_SYCL_MXFP4_TG_PROFILE":  (MMVQ, GGML_SYCL),
    "GGML_SYCL_MXFP4_PP_PROFILE":  (MMVQ, GGML_SYCL),
    "GGML_SYCL_ATTN_SYNC_PROFILE": (GGML_SYCL,),
}
MOE_TRACE = "GGML_SYCL_MOE_EXEC_TRACE"
PLAN_HPP_POLICY = "ggml/src/ggml-sycl/moe-exec-policy.hpp"


# ---- reading and masking -------------------------------------------------------------------------------------------

def is_test_path(rel: str) -> bool:
    return "/tests/" in rel or rel.startswith("ggml/src/ggml-sycl/tests/")


def load_sources() -> dict[str, str]:
    """Every non-test backend source file, by repo-relative path, plus the docs the knob is catalogued in."""
    out: dict[str, str] = {}
    for path in sorted(BACKEND.rglob("*")):
        if path.suffix not in (".cpp", ".hpp", ".h", ".cu", ".txt") or not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if is_test_path(rel):
            continue
        out[rel] = path.read_text(encoding="utf-8", errors="replace")
    for rel in ("docs/backend/sycl-env-vars.md", ".claude/rules/sycl-backend.md"):
        out[rel] = (ROOT / rel).read_text(encoding="utf-8")
    return out


_MASK_RE = re.compile(
    r"//[^\n]*"                                   # line comment
    r"|/\*.*?\*/"                                 # block comment
    r"|\"(?:\\.|[^\"\\\n])*\""                    # string literal
    r"|(?<![A-Za-z0-9_])'(?:\\.|[^'\\\n])*'",     # char literal (a ' after an alnum is a digit separator)
    re.S,
)
_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"", re.S)


def _blank(m: re.Match) -> str:
    text = m.group(0)
    return " " * len(text) if "\n" not in text else re.sub(r"[^\n]", " ", text)


def mask(text: str) -> str:
    """Same length, same offsets: comments, string literals and char literals become spaces."""
    return _MASK_RE.sub(_blank, text)


def mask_comments(text: str) -> str:
    """Same length, same offsets: comments become spaces, string literals stay (a getenv name is one)."""

    def keep_strings(m: re.Match) -> str:
        return m.group(0) if m.group(0).startswith('"') else _blank(m)

    return _COMMENT_RE.sub(keep_strings, text)


def brace_pairs(masked: str) -> dict[int, int]:
    pairs: dict[int, int] = {}
    stack: list[int] = []
    for m in re.finditer(r"[{}]", masked):
        if m.group(0) == "{":
            stack.append(m.start())
        elif stack:
            pairs[stack.pop()] = m.start()
    return pairs


def body_of(masked: str, signature_re: str, pairs: dict[int, int] | None = None) -> tuple[int, int] | None:
    """(open, close) offsets of the first function body whose header matches signature_re."""
    pairs = pairs if pairs is not None else brace_pairs(masked)
    for m in re.finditer(signature_re, masked):
        i = m.end()
        # skip to the opening brace of the body: a ';' first means a declaration only
        while i < len(masked) and masked[i] not in "{;":
            i += 1
        if i < len(masked) and masked[i] == "{" and i in pairs:
            return i, pairs[i]
    return None


# ---- P: the resolver ------------------------------------------------------------------------------------------------

def header_list(src: dict[str, str]) -> list[str] | None:
    """The names in the header's one list of diagnostic variables, in order; None when the list is missing."""
    code = mask_comments(src.get(PREDICATE_HEADER, ""))
    m = re.search(r"inline\s+constexpr\s+const\s+char\s*\*\s*%s\s*\[\s*\]\s*=\s*\{([^}]*)\}\s*;" % LIST, code)
    if m is None:
        return None
    return re.findall(r"\"([^\"]*)\"", m.group(1))


def check_predicate(src: dict[str, str]) -> list[str]:
    bad: list[str] = []
    header = src.get(PREDICATE_HEADER)
    if header is None:
        return [f"{PREDICATE_HEADER} does not exist: there is no predicate for {ENV_NAME}"]
    code = mask_comments(header)
    # the predicate is declared out of line: one definition in libggml-sycl, not a copy per translation unit
    if re.search(r"inline\s+bool\s+%s\s*\(" % PREDICATE, code):
        bad.append(f"{PREDICATE} is an inline in {PREDICATE_HEADER}: every translation unit would resolve the knob "
                   "itself")
    if not re.search(r"(?<!inline )\bbool\s+%s\s*\(\s*\)\s*;" % PREDICATE, code):
        bad.append(f"{PREDICATE_HEADER} does not declare `bool {PREDICATE}();` out of line")
    if re.search(r"\bgetenv\s*\(", code):
        bad.append(f"{PREDICATE_HEADER} reads the environment: only the memoised wrapper in {COMMON_CPP} may")
    if re.search(r"^[ \t]*#[ \t]*(?:if|ifdef|ifndef|elif)\b", code, re.M):
        bad.append(f"{PREDICATE_HEADER} has a preprocessor conditional: the answer must come from the environment")
    for rel, text in src.items():
        if rel.endswith((".md", ".txt")) or ENV_NAME not in text:
            continue
        if re.search(r"^[ \t]*#[ \t]*(?:if|ifdef|ifndef|elif)\b[^\n]*QUEUE_PROFILING", mask_comments(text), re.M):
            bad.append(f"{rel} reads a QUEUE_PROFILING macro in a preprocessor conditional")
    helper = src.get(HELPER, "")
    if not re.search(r"#[ \t]*include\s+\"\.\./queue-profiling\.hpp\"", mask_comments(helper)):
        bad.append(f"{HELPER} does not include queue-profiling.hpp, so common.hpp and the dpct queues cannot share one "
                   "predicate")
    # the reader's guard asks the predicate and warns once; the warning is a WARN that names the knob
    body = body_of(code, r"inline\s+bool\s+%s\s*\([^)]*\)" % READABLE)
    if body is None:
        bad.append(f"{PREDICATE_HEADER} defines no `inline bool {READABLE}(...)` for the readers")
    else:
        text = code[body[0]:body[1]]
        if PREDICATE + "(" not in text or WARN + "(" not in text:
            bad.append(f"{READABLE} does not ask {PREDICATE} and warn through {WARN}")
    cpp = mask_comments(src.get(COMMON_CPP, ""))
    # the knob's variable is read exactly once in the whole backend, by the resolver in common.cpp; that it is read once
    # per process, and what the WARN lines say, is behaviour that test-sycl-queue-profiling-resolver runs
    knob_read = re.compile(r"getenv\s*\(\s*\"%s\"\s*\)" % ENV_NAME)
    reads = [rel for rel, text in src.items()
             if not rel.endswith((".md", ".txt")) for _ in knob_read.finditer(mask_comments(text))]
    if reads != [COMMON_CPP]:
        bad.append(f"{ENV_NAME} is read with getenv {len(reads)} times ({sorted(set(reads))}), expected exactly once "
                   f"in {COMMON_CPP}: the answer is resolved once and never re-read")
    # the pure decision: no environment, no static, no logging; the three states and the unknown value
    body = body_of(cpp, r"ggml_sycl_queue_profiling_decision\s+%s\s*\([^)]*\)" % DECIDE)
    if body is None:
        bad.append(f"{COMMON_CPP} defines no pure {DECIDE}()")
    else:
        text = cpp[body[0]:body[1]]
        if re.search(r"getenv|\bstatic\b|GGML_LOG|sycl::", text):
            bad.append(f"{DECIDE} is not pure: it touches the environment, a static, a log or SYCL")
    return bad


# ---- L: the one list of diagnostics, and E: each diagnostic's own parse ----------------------------------------------

def check_list(src: dict[str, str]) -> list[str]:
    bad: list[str] = []
    names = header_list(src)
    if names is None:
        return [f"{PREDICATE_HEADER} has no `inline constexpr const char * {LIST}[]` list of diagnostic variables"]
    if len(set(names)) != len(names):
        bad.append(f"{LIST} names a variable twice")
    for n in names:
        if not re.fullmatch(r"GGML_SYCL_[A-Z0-9_]+", n):
            bad.append(f"{LIST} entry {n!r} is not a GGML_SYCL_ variable name")
    # every readable() call names exactly one entry of the list
    for rel, text in src.items():
        if rel.endswith((".md", ".txt")) or READABLE not in text:
            continue
        for m in re.finditer(r"\b%s\s*\(\s*([^)]*?)\s*\)" % READABLE, mask_comments(text)):
            arg = m.group(1)
            if arg.startswith("const char *"):
                continue  # the definition in the header
            lit = re.fullmatch(r"\"([^\"]*)\"", arg)
            if lit is None:
                bad.append(f"{rel}: {READABLE}({arg}) does not pass exactly one string literal")
            elif lit.group(1) not in names:
                bad.append(f"{rel}: {READABLE}(\"{lit.group(1)}\") names a variable that is not in {LIST}")
    # the resolver has a parse branch for every entry, and it is the parse the diagnostic itself uses
    cpp = mask_comments(src.get(COMMON_CPP, ""))
    body = body_of(cpp, r"static\s+bool\s+%s\s*\([^)]*\)" % DIAG_REQUESTED)
    if body is None:
        return bad + [f"{COMMON_CPP} defines no {DIAG_REQUESTED}()"]
    text = cpp[body[0]:body[1]]
    for n in names:
        if n == MOE_TRACE:
            want = r"moe_exec_config_parse\s*\(\s*getenv_fn\s*\)\s*\.trace\s*>=\s*2"
        elif n in HELPER_FOR:
            want = r"%s\s*\(\s*value\b" % HELPER_FOR[n]
        else:
            bad.append(f"{LIST} entry {n} has no row in this test's HELPER_FOR table: add its parse here on purpose")
            continue
        m = re.search(r"strcmp\s*\(\s*name\s*,\s*\"%s\"\s*\)\s*==\s*0[^{]*\)\s*\{\s*return\s+([^;]*);" % n, text)
        if m is None:
            bad.append(f"{COMMON_CPP}: {DIAG_REQUESTED} has no branch for {n}: it would count as requested only by the "
                       "fallback rule")
        elif not re.search(want, m.group(1)):
            bad.append(f"{COMMON_CPP}: {DIAG_REQUESTED} parses {n} with something other than the diagnostic's own "
                       "parse")
    fallback = (r"return\s+value\s*!=\s*nullptr\s*&&\s*value\[0\]\s*!=\s*'\\0'\s*&&\s*(?:std::)?strcmp\s*\("
                r"\s*value\s*,\s*\"0\"\s*\)\s*!=\s*0\s*;\s*$")
    if not re.search(fallback, text.rstrip()):
        bad.append(f"{COMMON_CPP}: {DIAG_REQUESTED} has no fallback that counts a set, non-empty value other than "
                   "\"0\" as requested (a diagnostic added to the list without a parse must not run unrequested)")
    return bad


def check_consumers(src: dict[str, str]) -> list[str]:
    """E: every read of a diagnostic's variable goes through the helper the resolver uses for it."""
    bad: list[str] = []
    for n, helper in HELPER_FOR.items():
        total = 0
        for rel, text in src.items():
            if rel.endswith((".md", ".txt")):
                continue
            code = mask_comments(text)
            reads = len(re.findall(r"getenv\s*\(\s*\"%s\"\s*\)" % n, code))
            via = len(re.findall(r"\b%s\s*\(\s*(?:std::)?getenv\s*\(\s*\"%s\"\s*\)" % (helper, n), code))
            total += via
            if reads != via:
                bad.append(f"{rel}: reads {n} {reads - via} time(s) without {helper}(), the parse the resolver uses "
                           "for it")
        for rel in CONSUMERS[n]:
            through = r"\b%s\s*\(\s*(?:std::)?getenv\s*\(\s*\"%s\"\s*\)" % (helper, n)
            if not re.search(through, mask_comments(src.get(rel, ""))):
                bad.append(f"{rel} no longer reads {n} through {helper}()")
    # GGML_SYCL_MOE_EXEC_TRACE: the ring windows are timed from device events only at level 2
    if not re.search(r"\btrace\s*>=\s*2\b", mask_comments(src.get(PLAN, ""))):
        bad.append(f"{PLAN} no longer gates the window timing on trace level 2, which is what the resolver asks for")
    # a string literal counts, a comment that quotes the name does not: the policy header must still read the variable
    if f"\"{MOE_TRACE}\"" not in mask_comments(src.get(PLAN_HPP_POLICY, "")):
        bad.append(f"{PLAN_HPP_POLICY} no longer names {MOE_TRACE}")
    return bad


# ---- Q: the property sites -----------------------------------------------------------------------------------------

ENABLE_RE = re.compile(r"\benable_profiling\b")


def check_property_sites(src: dict[str, str]) -> list[str]:
    bad: list[str] = []
    helper = mask(src.get(HELPER, ""))
    sigs = [m for m in re.finditer(r"sycl::queue\s+create_queue_impl\s*\(", helper)]
    pairs = brace_pairs(helper)
    allowed: dict[str, list[tuple[int, int]]] = {HELPER: [], COMMON_HPP: []}
    if len(sigs) != 2:
        bad.append(f"{HELPER} defines {len(sigs)} create_queue_impl overloads, expected 2")
    for m in sigs:
        i = m.end()
        while i < len(helper) and helper[i] != "{":
            i += 1
        if i not in pairs:
            bad.append(f"{HELPER}: a create_queue_impl overload has no body")
            continue
        lo, hi = i, pairs[i]
        allowed[HELPER].append((lo, hi))
        text = helper[lo:hi]
        # the original text decides the order: property list built as (enable_profiling, properties...)
        on = re.search(r"%s\s*\(\s*\)\s*\?\s*sycl::property_list\s*\(\s*"
                       r"sycl::property::queue::enable_profiling\s*\(\s*\)"
                       r"\s*,\s*properties\.\.\.\s*\)\s*:\s*"
                       r"sycl::property_list\s*\(\s*properties\.\.\.\s*\)" % PREDICATE,
                       text)
        if not on:
            bad.append(f"{HELPER}: a create_queue_impl overload does not build `{PREDICATE}() ? property_list("
                       "enable_profiling(), properties...) : property_list(properties...)`")
    common = mask(src.get(COMMON_HPP, ""))
    cpairs = brace_pairs(common)
    body = body_of(common, r"inline\s+sycl::property_list\s+default_queue_properties\s*\(\s*\)", cpairs)
    if body is None:
        bad.append(f"{COMMON_HPP} has no default_queue_properties()")
    else:
        allowed[COMMON_HPP].append(body)
        text = common[body[0]:body[1]]
        on = re.search(r"if\s*\(\s*%s\s*\(\s*\)\s*\)\s*\{\s*return\s+sycl::property_list\s*\{\s*"
                       r"sycl::property::queue::in_order\s*\{\s*\}\s*,\s*"
                       r"sycl::property::queue::enable_profiling\s*\{\s*\}"
                       r"\s*\}\s*;\s*\}" % PREDICATE, text)
        off = re.search(r"return\s+sycl::property_list\s*\{\s*sycl::property::queue::in_order\s*\{\s*\}\s*\}\s*;\s*$",
                        text.rstrip().rstrip("}").rstrip() + "")
        if not on:
            bad.append(f"{COMMON_HPP}: default_queue_properties() does not return (in_order, enable_profiling) exactly "
                       f"when {PREDICATE}() is true")
        if not off:
            bad.append(f"{COMMON_HPP}: default_queue_properties() does not fall back to in_order alone when the knob "
                       "is off")
    # no other backend code names the property: every site that does goes through the predicate
    for rel, text in src.items():
        if rel.endswith((".md", ".txt")) or "enable_profiling" not in text:
            continue
        masked = mask(text)
        for m in ENABLE_RE.finditer(masked):
            if any(lo <= m.start() < hi for lo, hi in allowed.get(rel, [])):
                continue
            line = masked.count("\n", 0, m.start()) + 1
            bad.append(f"{rel}:{line} names enable_profiling outside the three predicate-gated property sites")
    return bad


# ---- R: every reader is guarded ------------------------------------------------------------------------------------

_NOT_A_FUNCTION = re.compile(
    r"^\s*(?:template\s*<.*?>\s*)?(?:namespace|struct|class|union|enum|extern)\b"
    r"|^\s*(?:if|else|for|while|switch|do|try|catch)\b",
    re.S,
)


def reader_sites(src: dict[str, str], only: str | None = None) -> list[tuple[str, int, int, int]]:
    """(file, line, function-open offset, call offset) for every get_profiling_info call in non-test backend code."""
    sites: list[tuple[str, int, int, int]] = []
    for rel, text in src.items():
        if rel.endswith((".md", ".txt")) or "get_profiling_info" not in text or (only and rel != only):
            continue
        masked = mask(text)
        stack: list[tuple[int, bool]] = []  # (offset, is a function-like block)
        for m in re.finditer(r"[{}]|\bget_profiling_info\b", masked):
            tok = m.group(0)
            if tok == "{":
                cut = max(masked.rfind(c, 0, m.start()) for c in ";{}")
                header = "\n".join(ln for ln in masked[cut + 1:m.start()].split("\n")
                                   if not ln.lstrip().startswith("#"))
                stack.append((m.start(), "(" in header and not _NOT_A_FUNCTION.search(header)))
            elif tok == "}":
                if stack:
                    stack.pop()
            else:
                fn = next((off for off, is_fn in stack if is_fn), None)
                line = masked.count("\n", 0, m.start()) + 1
                sites.append((rel, line, -1 if fn is None else fn, m.start()))
    return sites


def guard_dominates(masked: str, pairs: dict[int, int], opens: list[int], fn: int, pos: int) -> bool:
    """A guard before `pos` whose own innermost block still encloses `pos`: a guard in a closed sibling block (or in a
    branch that has ended) does not decide the read."""
    for g in GUARD_RE.finditer(masked, fn, pos):
        i = bisect.bisect_left(opens, g.start()) - 1
        while i >= 0 and pairs[opens[i]] < g.start():
            i -= 1
        if (pairs[opens[i]] if i >= 0 else len(masked)) > pos:
            return True
    return False


def check_readers(src: dict[str, str], only: str | None = None) -> list[str]:
    """`only` limits the scan to one file, for the mutants, which change one file at a time."""
    bad: list[str] = []
    cache: dict[str, tuple[str, dict[int, int], list[int]]] = {}
    sites = reader_sites(src, only)
    if not sites:
        return ["no get_profiling_info call found in the backend: the scan is void"]
    for rel, line, fn, pos in sites:
        if rel not in cache:
            masked = mask(src[rel])
            pairs = brace_pairs(masked)
            cache[rel] = (masked, pairs, sorted(pairs))
        masked, pairs, opens = cache[rel]
        if fn < 0:
            bad.append(f"{rel}:{line} get_profiling_info outside any function")
        elif not guard_dominates(masked, pairs, opens, fn, pos):
            bad.append(f"{rel}:{line} get_profiling_info is not preceded, in a block that contains it, by "
                       f"{PREDICATE}/{READABLE}")
    return bad


READABLE_NAME_RE = re.compile(r"\b%s\s*\(\s*(\"[^\"]*\")" % READABLE)


def check_names(src: dict[str, str]) -> list[str]:
    """One variable name per readable() call: the warning names a diagnostic by the variable that asks for it."""
    bad: list[str] = []
    for rel, text in src.items():
        if rel.endswith((".md", ".txt")) or READABLE not in text:
            continue
        for m in READABLE_NAME_RE.finditer(mask_comments(text)):
            if not re.fullmatch(r"\"GGML_SYCL_[A-Z0-9_]+\"", m.group(1)):
                bad.append(f"{rel}: {READABLE}({m.group(1)}) does not name exactly one GGML_SYCL_ variable")
    return bad


# ---- D: the diagnostics' own gates ---------------------------------------------------------------------------------

DIAGNOSTIC_GATES = (
    (MMVQ, r"static\s+bool\s+mmvq_moe_tg_profile_enabled\s*\(\s*\)"),
    (MMVQ, r"static\s+bool\s+mmvq_moe_pp_profile_enabled\s*\(\s*\)"),
    (GGML_SYCL, r"bool\s+mxfp4_pp_batched_profile_enabled\s*\(\s*\)"),
    (PROFILER, r"ggml_sycl_kernel_profile_config\s+ggml_sycl_kernel_profile_config_from_env\s*\(\s*\)"),
)


def check_gates(src: dict[str, str]) -> list[str]:
    bad: list[str] = []
    for rel, sig in DIAGNOSTIC_GATES:
        masked = mask(src.get(rel, ""))
        body = body_of(masked, sig)
        if body is None:
            bad.append(f"{rel}: gate {sig} not found")
            continue
        g = GUARD_RE.search(masked, body[0], body[1])
        if g is None:
            bad.append(f"{rel}: gate {sig} stays open when {ENV_NAME}=0")
            continue
        start = max(masked.rfind(c, body[0], g.start()) for c in ";{}")
        stmt = masked[start + 1:g.start()]
        # the diagnostic's own enabled flag decides first, the guard last: with nothing requested the WARN path is never
        # reached on a per-op call and no WARN names a diagnostic nobody asked for. The guard's first call is also what
        # resolves the knob (one resolution per process, from the environment at that moment), so a guard asked ahead of
        # the flag would resolve it with the variable still unset: tests/test-sycl-kernel-profiler.cpp reads the
        # profiler config once with KERNEL_PROFILE unset, then sets it, and passes only because that first read stops
        # at the flag
        if not (re.search(r"(?:\breturn\b|[^=!<>]=)\s*(?:\w+\.)?enabled\s*&&", stmt) and stmt.rstrip().endswith("&&")):
            bad.append(f"{rel}: gate {sig} asks the guard before the diagnostic's own enabled flag")
    return bad


# ---- docs ------------------------------------------------------------------------------------------------------------

def check_docs(src: dict[str, str]) -> list[str]:
    bad: list[str] = []
    names = header_list(src) or []
    for rel in ("docs/backend/sycl-env-vars.md", ".claude/rules/sycl-backend.md"):
        rows = [ln for ln in src.get(rel, "").split("\n") if ln.lstrip().startswith("|") and f"`{ENV_NAME}" in ln]
        if not rows:
            bad.append(f"{rel} has no table row for {ENV_NAME}")
            continue
        row = " ".join(rows)
        if "OFF" not in row:
            bad.append(f"{rel}: the {ENV_NAME} row does not say the default is OFF")
        for n in names:
            if n not in row:
                bad.append(f"{rel}: the {ENV_NAME} row does not name the diagnostic {n}")
    for rel, text in src.items():
        if re.search(r"default stays on|activate counter-based events on L0|~\s*15\s*%\s*TG\s+(?:speedup|improvement)"
                     r"|profiling[^.]{0,40}enabled unconditionally", text):
            bad.append(f"{rel} still says the default stays on, that profiling is unconditional, or quotes the "
                       "unmeasured ~15% TG speedup")
    return bad


# ---- mutants ---------------------------------------------------------------------------------------------------------

def mutate_re(src: dict[str, str], rel: str, pattern: str, new: str, count: int = 1) -> dict[str, str]:
    assert re.search(pattern, src[rel], re.S), f"mutant precondition: /{pattern}/ matches in {rel}"
    out = dict(src)
    out[rel] = re.sub(pattern, lambda _m: new, src[rel], count=count, flags=re.S)
    return out


def flags(problems: list[str], needle: str) -> bool:
    """True when the pin reports a problem about `needle`: a mutant must fail its own pin, not any pin."""
    return any(needle in p for p in problems)


def mutate(src: dict[str, str], rel: str, old: str, new: str, count: int = 1) -> dict[str, str]:
    assert old in src[rel], f"mutant precondition: {old!r} is in {rel}"
    out = dict(src)
    out[rel] = src[rel].replace(old, new, count)
    return out


SOURCES = load_sources()


def test_unmutated_tree_passes_every_pin():
    for name, check in (("P", check_predicate), ("L", check_list), ("E", check_consumers), ("Q", check_property_sites),
                        ("R", check_readers), ("D", check_gates), ("names", check_names), ("docs", check_docs)):
        problems = check(SOURCES)
        assert not problems, f"pin {name}:\n  " + "\n  ".join(problems)


def test_predicate_mutants():
    # no predicate header at all
    gone = {k: v for k, v in SOURCES.items() if k != PREDICATE_HEADER}
    assert flags(check_predicate(gone), "does not exist")
    # the predicate is an inline again: a copy per translation unit
    assert flags(check_predicate(mutate(SOURCES, PREDICATE_HEADER, "bool                             %s();" % PREDICATE,
                                        "inline bool %s() { return false; }" % PREDICATE)),
                 "is an inline")
    # the header reads the environment itself
    env_set = "inline bool ggml_sycl_profile_env_set(const char * value) {"
    reading = env_set + "\n    (void) std::getenv(\"X\");"
    assert flags(check_predicate(mutate(SOURCES, PREDICATE_HEADER, env_set, reading)), "reads the environment")
    # a preprocessor conditional inside the header
    reason_enum = "enum class ggml_sycl_queue_profiling_reason"
    assert flags(check_predicate(mutate(SOURCES, PREDICATE_HEADER, reason_enum,
                                        "#ifdef GGML_SYCL_QUEUE_PROFILING_MACRO\n" + reason_enum)),
                 "preprocessor conditional")
    # the dpct header cannot see it
    assert flags(check_predicate(mutate(SOURCES, HELPER, "queue-profiling.hpp", "async-failure.hpp")),
                 "does not include queue-profiling.hpp")
    # the decision reads the environment itself
    requested_line = "    const bool requested = requested_diagnostic != nullptr;\n"
    assert flags(check_predicate(mutate(SOURCES, COMMON_CPP, requested_line + "    if (knob_value != nullptr) {",
                                        requested_line + "    if (std::getenv(\"X\") != nullptr) {}\n"
                                        "    if (knob_value != nullptr) {")),
                 "is not pure")
    # a second read of the knob in the resolver's file
    enabled_head = "bool ggml_sycl_queue_profiling_enabled() {\n"
    second_read = enabled_head + "    if (std::getenv(\"%s\") != nullptr) { return true; }\n" % ENV_NAME
    assert flags(check_predicate(mutate(SOURCES, COMMON_CPP, enabled_head, second_read)), "read with getenv")


def test_list_mutants():
    # one diagnostic dropped from the list: its readable() call names a variable outside it, and the resolver has a
    # branch for a name the list no longer holds
    dropped = mutate(SOURCES, PREDICATE_HEADER, '"GGML_SYCL_ATTN_SYNC_PROFILE", ', "")
    assert flags(check_list(dropped), "is not in %s" % LIST)
    # a readable() call naming a variable that is not a diagnostic
    mutant = mutate(SOURCES, MMVQ, 'warn_unless_readable("GGML_SYCL_MXFP4_TG_PROFILE")',
                    'warn_unless_readable("GGML_SYCL_NOT_A_DIAGNOSTIC")')
    assert flags(check_list(mutant), "GGML_SYCL_NOT_A_DIAGNOSTIC")
    # a readable() call that does not pass a literal
    mutant = mutate(SOURCES, MMVQ, 'warn_unless_readable("GGML_SYCL_MXFP4_TG_PROFILE")', "warn_unless_readable(kName)")
    assert flags(check_list(mutant), "exactly one string literal")
    # a diagnostic added to the list without a parse in the resolver and without a row in this table
    added = mutate(SOURCES, PREDICATE_HEADER, '"GGML_SYCL_MOE_EXEC_TRACE",\n',
                   '"GGML_SYCL_MOE_EXEC_TRACE",\n    "GGML_SYCL_NEW_PROFILE",\n')
    assert flags(check_list(added), "GGML_SYCL_NEW_PROFILE has no row")
    # the list is gone
    assert flags(check_list(mutate(SOURCES, PREDICATE_HEADER, "ggml_sycl_queue_profiling_diagnostics[] = {",
                                   "other[] = {")),
                 "has no `inline constexpr")
    # the resolver parses a diagnostic differently from the diagnostic: ATTN_SYNC is on for any set value, "0" included
    mutant = mutate(SOURCES, COMMON_CPP, "return ggml_sycl_profile_env_set(value);",
                    "return ggml_sycl_profile_env_nonzero(value);")
    assert flags(check_list(mutant), "GGML_SYCL_ATTN_SYNC_PROFILE with something other than the diagnostic's own parse")
    # no branch for a diagnostic
    mutant = mutate(SOURCES, COMMON_CPP, 'std::strcmp(name, "GGML_SYCL_ATTN_SYNC_PROFILE") == 0',
                    'std::strcmp(name, "GGML_SYCL_OTHER") == 0')
    assert flags(check_list(mutant), "has no branch for GGML_SYCL_ATTN_SYNC_PROFILE")
    # MoE trace: level 1 would count as requested
    mutant = mutate(SOURCES, COMMON_CPP, ".trace >= 2;", ".trace >= 1;")
    assert flags(check_list(mutant), "GGML_SYCL_MOE_EXEC_TRACE with something other")
    # the fallback for a name without a parse is gone
    mutant = mutate(SOURCES, COMMON_CPP,
                    "return value != nullptr && value[0] != '\\0' && std::strcmp(value, \"0\") != 0;", "return false;")
    assert flags(check_list(mutant), "no fallback")


def test_consumer_parse_mutants():
    # a diagnostic parses its own variable with something else than the resolver's helper
    for rel, name, helper in ((MMVQ, "GGML_SYCL_MXFP4_TG_PROFILE", "ggml_sycl_profile_env_nonzero"),
                              (MMVQ, "GGML_SYCL_MXFP4_PP_PROFILE", "ggml_sycl_profile_env_nonzero"),
                              (GGML_SYCL, "GGML_SYCL_ATTN_SYNC_PROFILE", "ggml_sycl_profile_env_set"),
                              (PROFILER, "GGML_SYCL_KERNEL_PROFILE", "ggml_sycl_profile_env_flag")):
        old = re.search(r"%s\s*\(\s*std::getenv\(\"%s\"\)" % (helper, name), SOURCES[rel])
        assert old, f"mutant precondition: {rel} reads {name} through {helper}"
        mutant = mutate(SOURCES, rel, old.group(0),
                        'std::getenv("%s") != nullptr || ggml_sycl_unrelated(std::getenv("%s")' % (name, name))
        problems = check_consumers(mutant)
        assert flags(problems, name) and flags(problems, "without " + helper), f"{rel} {name}: {problems}"
    # a second, unshared read of a diagnostic variable anywhere in the backend
    extra = mutate(SOURCES, COMMON_CPP, '#include "common.hpp"',
                   '#include "common.hpp"\n'
                   'static const bool g_mutant = std::getenv("GGML_SYCL_MXFP4_PP_PROFILE") != nullptr;', 1)
    assert flags(check_consumers(extra), "GGML_SYCL_MXFP4_PP_PROFILE")
    # the window timing no longer waits for trace level 2
    assert flags(check_consumers(mutate(SOURCES, PLAN, "op.log->trace >= 2", "op.log->trace >= 1")), "level 2")
    # the policy header stops naming the variable in code and only a comment still quotes it
    name_literal = '"%s"' % MOE_TRACE
    assert name_literal in mask_comments(SOURCES[PLAN_HPP_POLICY]), "mutant precondition: the policy header names it"
    assert not flags(check_consumers(SOURCES), PLAN_HPP_POLICY)
    commented = mutate(SOURCES, PLAN_HPP_POLICY, name_literal, "kRenamed /* %s */" % name_literal, count=1000)
    assert name_literal not in mask_comments(commented[PLAN_HPP_POLICY])
    assert flags(check_consumers(commented), "%s no longer names" % PLAN_HPP_POLICY)


def test_property_site_mutants():
    src = SOURCES
    on = (r"%s\(\)\s*\?\s*sycl::property_list\(sycl::property::queue::enable_profiling\(\),\s*properties\.\.\.\)"
          r"\s*:\s*sycl::property_list\(properties\.\.\.\)" % PREDICATE)
    # a dpct overload adds the property unconditionally again
    assert flags(check_property_sites(mutate_re(src, HELPER, on,
                                                "sycl::property_list(sycl::property::queue::enable_profiling(), "
                                                "properties...)")),
                 "create_queue_impl overload does not build")
    # a dpct overload loses the property for every value of the knob
    assert flags(check_property_sites(mutate_re(src, HELPER, on, "sycl::property_list(properties...)")),
                 "create_queue_impl overload does not build")
    # default_queue_properties() drops the predicate
    assert flags(check_property_sites(mutate(src, COMMON_HPP, "if (%s()) {" % PREDICATE, "if (true) {")),
                 "default_queue_properties() does not return")
    # default_queue_properties() reorders the list
    assert flags(check_property_sites(mutate(src, COMMON_HPP,
                                             "sycl::property::queue::in_order{}, "
                                             "sycl::property::queue::enable_profiling{} };",
                                             "sycl::property::queue::enable_profiling{}, "
                                             "sycl::property::queue::in_order{} };")),
                 "default_queue_properties() does not return")
    # default_queue_properties() has no off path
    assert flags(check_property_sites(mutate(src, COMMON_HPP,
                                             "return sycl::property_list{ sycl::property::queue::in_order{} };",
                                             "return sycl::property_list{ sycl::property::queue::in_order{}, "
                                             "sycl::property::queue::enable_profiling{} };")),
                 "does not fall back to in_order alone")
    # a queue built elsewhere names the property itself
    assert flags(check_property_sites(mutate(src, "ggml/src/ggml-sycl/expert-prefetch.cpp", "sycl::property_list{}",
                                             "sycl::property_list{ sycl::property::queue::enable_profiling{} }")),
                 "names enable_profiling outside")


def test_reader_mutants():
    src = SOURCES
    sites = reader_sites(src)
    assert sites, "the reader scan found no get_profiling_info call"
    assert {s[0] for s in sites} >= {MMVQ, GGML_SYCL, PROFILER, PLAN}, "the scan misses a known reader file"
    # every site, one at a time: neuter the guards of its own function and the pin names that site
    seen: set[tuple[str, int]] = set()
    for rel, line, fn, pos in sites:
        if (rel, fn) in seen:
            continue
        seen.add((rel, fn))
        region = src[rel][fn:pos]
        neutered = re.sub(r"\b(?:%s|%s)\b" % (PREDICATE, READABLE), "ggml_sycl_unrelated_call", region)
        assert neutered != region, f"mutant precondition: {rel}:{line} has a guard to remove"
        mutant = dict(src)
        mutant[rel] = src[rel][:fn] + neutered + src[rel][pos:]
        problems = check_readers(mutant, rel)
        assert any(f"{rel}:{line} " in p and "not preceded" in p for p in problems), \
            f"removing the guard before {rel}:{line} was not caught"
    # a new unguarded reader anywhere in the backend
    extra = mutate(src, "ggml/src/ggml-sycl/common.cpp", "#include \"common.hpp\"",
                   "#include \"common.hpp\"\n"
                   "static int ggml_sycl_mutant_reader(sycl::event e) { return (int) e.get_profiling_info<"
                   "sycl::info::event_profiling::command_end>(); }\n")
    assert any("common.cpp" in p and "not preceded" in p for p in check_readers(extra, "ggml/src/ggml-sycl/common.cpp"))


def test_reader_guard_in_a_closed_sibling_block_does_not_count():
    old = r"if\s*\(\s*!\s*ggml_sycl_queue_profiling_enabled\s*\(\s*\)\s*\)\s*\{\s*return\s+-1\.0;\s*\}"
    mutant = mutate_re(SOURCES, MMVQ, old, '{ ggml_sycl_queue_profiling_warn_unless_readable("GGML_SYCL_X"); }')
    problems = check_readers(mutant, MMVQ)
    assert flags(problems, f"{MMVQ}:") and flags(problems, "not preceded"), "a guard in a closed block was accepted"


def test_reader_names_mutants():
    mutant = mutate(SOURCES, MMVQ, 'readable("GGML_SYCL_MXFP4_TG_PROFILE")', 'readable("GGML_SYCL_A / GGML_SYCL_B")')
    assert flags(check_names(mutant), "exactly one GGML_SYCL_ variable")


def test_gate_operand_order_mutant():
    # The guard asked ahead of the diagnostic's own flag resolves the knob at that first call, from the environment as
    # it is then, and prints a WARN for a diagnostic nobody asked for. tests/test-sycl-kernel-profiler.cpp relies on the
    # flag-first order too: its first config read runs with the variable unset and must not resolve the knob, or the
    # read after it sets the variable and finds the knob already off.
    tg = "GGML_SYCL_MXFP4_TG_PROFILE"
    old = (r'return enabled && !ggml_sycl_graph_recording_active\(\) &&\s*'
           r'ggml_sycl_queue_profiling_warn_unless_readable\("%s"\);' % tg)
    new = ('return ggml_sycl_queue_profiling_warn_unless_readable("%s") && enabled && '
           '!ggml_sycl_graph_recording_active();' % tg)
    assert flags(check_gates(mutate_re(SOURCES, MMVQ, old, new)), "before the diagnostic's own enabled flag")
    old = (r'cfg\.enabled = cfg\.enabled &&\s*'
           r'ggml_sycl_queue_profiling_warn_unless_readable\("GGML_SYCL_KERNEL_PROFILE"\);')
    new = 'cfg.enabled = ggml_sycl_queue_profiling_warn_unless_readable("GGML_SYCL_KERNEL_PROFILE") && cfg.enabled;'
    assert flags(check_gates(mutate_re(SOURCES, PROFILER, old, new)), "before the diagnostic's own enabled flag")


def test_gate_mutants():
    src = SOURCES
    for rel, sig in DIAGNOSTIC_GATES:
        masked = mask(src[rel])
        body = body_of(masked, sig)
        region = src[rel][body[0]:body[1]]
        neutered = re.sub(r"\b(?:%s|%s)\b" % (PREDICATE, READABLE), "ggml_sycl_unrelated_call", region)
        assert neutered != region
        mutant = dict(src)
        mutant[rel] = src[rel][:body[0]] + neutered + src[rel][body[1]:]
        assert flags(check_gates(mutant), f"{rel}: gate"), f"opening the gate in {rel} {sig} was not caught"


def test_docs_mutants():
    src = SOURCES
    stripped = dict(src)
    stripped["docs/backend/sycl-env-vars.md"] = src["docs/backend/sycl-env-vars.md"].replace(ENV_NAME, "GGML_SYCL_X")
    assert flags(check_docs(stripped), "no table row")
    # a row that says the default is ON again, and the old sentence back
    row = [ln for ln in src[".claude/rules/sycl-backend.md"].split("\n") if f"`{ENV_NAME}" in ln][0]
    on = dict(src)
    on[".claude/rules/sycl-backend.md"] = src[".claude/rules/sycl-backend.md"].replace(row, row.replace("OFF", "ON"))
    assert flags(check_docs(on), "does not say the default is OFF")
    stale = dict(src)
    env_doc = "docs/backend/sycl-env-vars.md"
    stale[env_doc] = src[env_doc] + "\nThe default stays on while that is measured.\n"
    assert flags(check_docs(stale), "still says the default stays on")
    # the stale claim back in a CMakeLists.txt: the gate scans build files too
    cmake = "ggml/src/ggml-sycl/CMakeLists.txt"
    assert cmake in src, f"{cmake} is not among the scanned sources"
    for stale_text in ("# Result: ~15% TG improvement on Arc B580.\n",
                       "# SYCL queue profiling is enabled unconditionally here.\n"):
        cmake_stale = dict(src)
        cmake_stale[cmake] = src[cmake] + "\n" + stale_text
        assert flags(check_docs(cmake_stale), cmake + " still says"), stale_text
    unnamed = dict(src)
    unnamed[".claude/rules/sycl-backend.md"] = src[".claude/rules/sycl-backend.md"].replace(
        row, row.replace("GGML_SYCL_ATTN_SYNC_PROFILE", "X"))
    assert flags(check_docs(unnamed), "does not name the diagnostic GGML_SYCL_ATTN_SYNC_PROFILE")


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failed else 0)
