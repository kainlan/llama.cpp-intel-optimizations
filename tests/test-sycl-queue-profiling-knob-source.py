#!/usr/bin/env python3
"""Source gate for the GGML_SYCL_QUEUE_PROFILING knob (llama.cpp-atiz).

Every queue of the SYCL backend is built with sycl::property::queue::enable_profiling by default. On the Level Zero
v2 adapter that makes every submitted command pay one device timestamp query, which only diagnostics read. The knob
GGML_SYCL_QUEUE_PROFILING=0 removes the property from every queue the backend builds; unset, or any other value,
keeps today's property lists. This gate reads the source text only (no device, no build) and pins four things:

  P  the predicate: one inline function in queue-profiling.hpp that reads the environment with getenv (never a
     macro: every translation unit compiles its own copy of the header-only inlines, so a macro read would let the
     linker pick one answer for all of them), whose "0" is the only value that turns profiling off, and which is
     reachable from both dpct/helper.hpp and common.hpp;
  Q  the three places that add enable_profiling (both create_queue_impl overloads in dpct/helper.hpp and
     default_queue_properties() in common.hpp) go through the predicate, keep today's list and order when it is
     true, and leave the property out when it is false; no other backend code names enable_profiling;
  R  every get_profiling_info call in non-test backend code sits in a function that first consults the predicate (a
     query on an event of a queue without profiling throws, and a diagnostic must report no figure instead);
  D  the diagnostics whose enabling predicate gates every figure they print close when the knob is off.

Runs under pytest (llama_test_pytest registration) and as a plain script. Each pin has a mutant that must fail it;
the mutants run inside the tests, on in-memory copies of the sources.
"""
from __future__ import annotations

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
READABLE = "ggml_sycl_queue_profiling_readable"
WARN = "ggml_sycl_queue_profiling_warn_unavailable"
GUARD_RE = re.compile(r"\b(?:%s|%s)\s*\(" % (PREDICATE, READABLE))
ENV_NAME = "GGML_SYCL_QUEUE_PROFILING"


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


# ---- P: the predicate ------------------------------------------------------------------------------------------------

def check_predicate(src: dict[str, str]) -> list[str]:
    bad: list[str] = []
    header = src.get(PREDICATE_HEADER)
    if header is None:
        return [f"{PREDICATE_HEADER} does not exist: there is no predicate for {ENV_NAME}"]
    code = mask_comments(header)
    body = body_of(code, r"inline\s+bool\s+%s\s*\(\s*\)" % PREDICATE)
    if body is None:
        bad.append(f"{PREDICATE_HEADER} defines no `inline bool {PREDICATE}()`")
    else:
        text = code[body[0]:body[1]]
        if not re.search(r"getenv\s*\(\s*\"%s\"\s*\)" % ENV_NAME, text):
            bad.append(f"{PREDICATE} does not read {ENV_NAME} with getenv")
        if not re.search(r"return\s+\w+\s*==\s*nullptr\s*\|\|\s*(?:std::)?strcmp\s*\(\s*\w+\s*,\s*\"0\"\s*\)\s*!=\s*0\s*;",
                         text):
            bad.append(f"{PREDICATE} is not true for an unset value and for every value but exactly \"0\"")
    if re.search(r"^[ \t]*#[ \t]*(?:if|ifdef|ifndef|elif)\b", code, re.M):
        bad.append(f"{PREDICATE_HEADER} has a preprocessor conditional: the answer must come from the environment")
    for rel, text in src.items():
        if rel.endswith((".md", ".txt")) or ENV_NAME not in text:
            continue
        if re.search(r"^[ \t]*#[ \t]*(?:if|ifdef|ifndef|elif)\b[^\n]*QUEUE_PROFILING", mask_comments(text), re.M):
            bad.append(f"{rel} reads a QUEUE_PROFILING macro in a preprocessor conditional")
    helper = src.get(HELPER, "")
    if not re.search(r"#[ \t]*include\s+\"\.\./queue-profiling\.hpp\"", mask_comments(helper)):
        bad.append(f"{HELPER} does not include queue-profiling.hpp, so common.hpp and the dpct queues cannot share one predicate")
    # the reader's guard asks the predicate and warns once; the warning is a WARN that names the knob
    body = body_of(code, r"inline\s+bool\s+%s\s*\([^)]*\)" % READABLE)
    if body is None:
        bad.append(f"{PREDICATE_HEADER} defines no `inline bool {READABLE}(...)` for the readers")
    else:
        text = code[body[0]:body[1]]
        if PREDICATE + "(" not in text or WARN + "(" not in text:
            bad.append(f"{READABLE} does not ask {PREDICATE} and warn through {WARN}")
    cpp = mask_comments(src.get(COMMON_CPP, ""))
    body = body_of(cpp, r"void\s+%s\s*\([^)]*\)" % WARN)
    if body is None:
        bad.append(f"{COMMON_CPP} does not define {WARN}")
    else:
        text = cpp[body[0]:body[1]]
        if "GGML_LOG_WARN" not in text:
            bad.append(f"{WARN} does not log at WARN (INFO is dropped at the default verbosity)")
        if f"{ENV_NAME}=1" not in text:
            bad.append(f"{WARN} does not tell the reader to set {ENV_NAME}=1")
        if not re.search(r"\.insert\s*\(", text) or ".second" not in text:
            bad.append(f"{WARN} does not warn once per diagnostic")
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
        on = re.search(r"%s\s*\(\s*\)\s*\?\s*sycl::property_list\s*\(\s*sycl::property::queue::enable_profiling\s*\(\s*\)"
                       r"\s*,\s*properties\.\.\.\s*\)\s*:\s*sycl::property_list\s*\(\s*properties\.\.\.\s*\)" % PREDICATE,
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
                       r"sycl::property::queue::in_order\s*\{\s*\}\s*,\s*sycl::property::queue::enable_profiling\s*\{\s*\}"
                       r"\s*\}\s*;\s*\}" % PREDICATE, text)
        off = re.search(r"return\s+sycl::property_list\s*\{\s*sycl::property::queue::in_order\s*\{\s*\}\s*\}\s*;\s*$",
                        text.rstrip().rstrip("}").rstrip() + "")
        if not on:
            bad.append(f"{COMMON_HPP}: default_queue_properties() does not return (in_order, enable_profiling) exactly "
                       f"when {PREDICATE}() is true")
        if not off:
            bad.append(f"{COMMON_HPP}: default_queue_properties() does not fall back to in_order alone when the knob is off")
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
                cut = max(masked.rfind(";", 0, m.start()), masked.rfind("{", 0, m.start()), masked.rfind("}", 0, m.start()))
                header = "\n".join(ln for ln in masked[cut + 1:m.start()].split("\n") if not ln.lstrip().startswith("#"))
                stack.append((m.start(), "(" in header and not _NOT_A_FUNCTION.search(header)))
            elif tok == "}":
                if stack:
                    stack.pop()
            else:
                fn = next((off for off, is_fn in stack if is_fn), None)
                line = masked.count("\n", 0, m.start()) + 1
                sites.append((rel, line, -1 if fn is None else fn, m.start()))
    return sites


def check_readers(src: dict[str, str], only: str | None = None) -> list[str]:
    """`only` limits the scan to one file, for the mutants, which change one file at a time."""
    bad: list[str] = []
    masked_cache: dict[str, str] = {}
    sites = reader_sites(src, only)
    if not sites:
        return ["no get_profiling_info call found in the backend: the scan is void"]
    for rel, line, fn, pos in sites:
        masked = masked_cache.setdefault(rel, mask(src[rel]))
        if fn < 0:
            bad.append(f"{rel}:{line} get_profiling_info outside any function")
        elif not GUARD_RE.search(masked, fn, pos):
            bad.append(f"{rel}:{line} get_profiling_info is not preceded, in its function, by {PREDICATE}/{READABLE}")
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
        elif not GUARD_RE.search(masked, body[0], body[1]):
            bad.append(f"{rel}: gate {sig} stays open when {ENV_NAME}=0")
    return bad


# ---- docs ------------------------------------------------------------------------------------------------------------

def check_docs(src: dict[str, str]) -> list[str]:
    bad: list[str] = []
    for rel in ("docs/backend/sycl-env-vars.md", ".claude/rules/sycl-backend.md"):
        rows = [ln for ln in src.get(rel, "").split("\n") if ln.lstrip().startswith("|") and ENV_NAME in ln]
        if not rows:
            bad.append(f"{rel} has no table row for {ENV_NAME}")
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
    for name, check in (("P", check_predicate), ("Q", check_property_sites), ("R", check_readers),
                        ("D", check_gates), ("docs", check_docs)):
        problems = check(SOURCES)
        assert not problems, f"pin {name}:\n  " + "\n  ".join(problems)


def test_predicate_mutants():
    # no predicate header at all
    gone = {k: v for k, v in SOURCES.items() if k != PREDICATE_HEADER}
    assert flags(check_predicate(gone), "does not exist")
    # a macro read instead of the environment
    assert flags(check_predicate(mutate(SOURCES, PREDICATE_HEADER, "std::getenv(\"%s\")" % ENV_NAME, "std::getenv(\"OTHER\")")),
                 "does not read")
    # any value but "0" turns it off: the default flips
    assert flags(check_predicate(mutate(SOURCES, PREDICATE_HEADER, "== nullptr ||", "!= nullptr &&")),
                 'exactly "0"')
    # a preprocessor conditional inside the header
    assert flags(check_predicate(mutate(SOURCES, PREDICATE_HEADER, "inline bool %s" % PREDICATE,
                                        "#ifdef GGML_SYCL_QUEUE_PROFILING_MACRO\ninline bool %s" % PREDICATE)),
                 "preprocessor conditional")
    # the dpct header cannot see it
    assert flags(check_predicate(mutate(SOURCES, HELPER, "queue-profiling.hpp", "async-failure.hpp")),
                 "does not include queue-profiling.hpp")
    # the warning is not a WARN
    body = body_of(mask_comments(SOURCES[COMMON_CPP]), r"void\s+%s\s*\([^)]*\)" % WARN)
    assert body is not None
    text = SOURCES[COMMON_CPP]
    mutant = dict(SOURCES)
    mutant[COMMON_CPP] = text[:body[0]] + text[body[0]:body[1]].replace("GGML_LOG_WARN", "GGML_LOG_INFO") + text[body[1]:]
    assert flags(check_predicate(mutant), "does not log at WARN")
    # the warning is not once per diagnostic
    mutant[COMMON_CPP] = text[:body[0]] + text[body[0]:body[1]].replace(".second", ".first") + text[body[1]:]
    assert flags(check_predicate(mutant), "once per diagnostic")


def test_property_site_mutants():
    src = SOURCES
    on = (r"%s\(\)\s*\?\s*sycl::property_list\(sycl::property::queue::enable_profiling\(\),\s*properties\.\.\.\)"
          r"\s*:\s*sycl::property_list\(properties\.\.\.\)" % PREDICATE)
    # a dpct overload adds the property unconditionally again
    assert flags(check_property_sites(mutate_re(src, HELPER, on,
                                                "sycl::property_list(sycl::property::queue::enable_profiling(), properties...)")),
                 "create_queue_impl overload does not build")
    # a dpct overload loses the property for every value of the knob
    assert flags(check_property_sites(mutate_re(src, HELPER, on, "sycl::property_list(properties...)")),
                 "create_queue_impl overload does not build")
    # default_queue_properties() drops the predicate
    assert flags(check_property_sites(mutate(src, COMMON_HPP, "if (%s()) {" % PREDICATE, "if (true) {")),
                 "default_queue_properties() does not return")
    # default_queue_properties() reorders the list
    assert flags(check_property_sites(mutate(src, COMMON_HPP,
                                             "sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{} };",
                                             "sycl::property::queue::enable_profiling{}, sycl::property::queue::in_order{} };")),
                 "default_queue_properties() does not return")
    # default_queue_properties() has no off path
    assert flags(check_property_sites(mutate(src, COMMON_HPP, "return sycl::property_list{ sycl::property::queue::in_order{} };",
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
                   "#include \"common.hpp\"\nstatic int ggml_sycl_mutant_reader(sycl::event e) { return (int) e.get_profiling_info<"
                   "sycl::info::event_profiling::command_end>(); }\n")
    assert any("common.cpp" in p and "not preceded" in p for p in check_readers(extra, "ggml/src/ggml-sycl/common.cpp"))


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
