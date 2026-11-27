#!/usr/bin/env python3
"""GGML_SYCL_UNIFIED_CACHE_MODE is read once per process: source contract.

THE COST.  get_effective_mode() asks get_unified_cache_mode() for the topology on every unified-cache lookup
(get_existing_unified_cache_for_device and its siblings), and the decode path makes such lookups per op. When the
accessor called std::getenv and strcmp on each call, getenv was 5.6 % of the submitting thread of a two-card decode
profile. The value cannot usefully change once the cache exists, and docs/backend/sycl-env-vars.md states read-once
as the convention for the backend's variables.

THE CONTRACT.  With comments blanked, in ggml/src/ggml-sycl/unified-cache.cpp:
  O. The one std::getenv("GGML_SYCL_UNIFIED_CACHE_MODE") of the backend sits in the initializer of a function-local
     `static const` inside get_unified_cache_mode(), so a second call reads the held value and not the environment.
     No other backend source calls getenv on the variable.
  S. Only the override is held: get_unified_cache_mode() still returns g_cache_mode, outside that initializer, when
     the variable names no mode, so set_unified_cache_mode() keeps working until the cache locks the mode.
  L. set_unified_cache_mode() still refuses a change once g_cache_mode_locked is set, and otherwise assigns
     g_cache_mode.

NOT VACUOUS.  --self-test runs the gate on in-memory mutants of the real source and requires each to fail the named
pin, then requires the unmodified tree to pass.

RED on the tree before the change: pin O fails (the getenv ran on every call).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SYCL = ROOT / "ggml" / "src" / "ggml-sycl"
CACHE = SYCL / "unified-cache.cpp"
OTHER_SOURCES = (SYCL / "ggml-sycl.cpp", SYCL / "unified-cache.hpp", SYCL / "common.hpp")

ENV_NAME = "GGML_SYCL_UNIFIED_CACHE_MODE"
GETENV = re.compile(r'std::getenv\(\s*"' + ENV_NAME + r'"\s*\)|(?<![\w:])getenv\(\s*"' + ENV_NAME + r'"\s*\)')

# Comments blank to spaces, keeping offsets; strings and char literals are matched first so a comment marker inside
# one is not taken for a comment.
_TOKEN = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n]){1,4}\'', re.S)


def blank_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        t = m.group(0)
        if t.startswith("//") or t.startswith("/*"):
            return "".join("\n" if c == "\n" else " " for c in t)
        return t

    return _TOKEN.sub(repl, src)


def body_span(code: str, signature: str) -> tuple[int, int] | None:
    """The [start, end) offsets of the brace-balanced body of the first definition matching `signature` (a regex
    ending at its opening brace), or None. Braces inside string and char literals do not count, so blank comments
    first: an apostrophe in a comment would otherwise open a char literal."""
    m = re.search(signature, code)
    if not m:
        return None
    i = code.find("{", m.end() - 1)
    if i < 0:
        return None
    depth = 0
    j = i
    while j < len(code):
        c = code[j]
        if c in "\"'":
            k = j + 1
            while k < len(code) and code[k] != c:
                k += 2 if code[k] == "\\" else 1
            j = k + 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i, j + 1
        j += 1
    return None


def body_of(code: str, signature: str) -> str | None:
    """The body body_span() finds, as text."""
    span = body_span(code, signature)
    return None if span is None else code[span[0]:span[1]]


def static_initializer(body: str, at: int) -> bool:
    """True when offset `at` of a function body lies inside the initializer of a function-local `static const`
    declaration: from its `=` to the `;` that ends the declaration at the declaration's own nesting depth."""
    for m in re.finditer(r"\bstatic\s+const\b[^=;{}]*=", body):
        start = m.end()
        depth = 0
        j = start
        while j < len(body):
            c = body[j]
            if c in "([{":
                depth += 1
            elif c in ")]}":
                depth -= 1
            elif c == ";" and depth == 0:
                break
            j += 1
        if start <= at < j:
            return True
    return False


SIG_GET = r"\bunified_cache_mode\s+get_unified_cache_mode\(\s*\)\s*\{"
SIG_SET = r"\bvoid\s+set_unified_cache_mode\(\s*unified_cache_mode\s+mode\s*\)\s*\{"


def check(cache_src: str, others: dict[str, str]) -> list[str]:
    """The failed pins, each as 'PIN: reason'. Empty means the contract holds."""
    fails: list[str] = []
    code = blank_comments(cache_src)

    get_body = body_of(code, SIG_GET)
    set_body = body_of(code, SIG_SET)
    if get_body is None:
        fails.append("O: get_unified_cache_mode() definition not found")
        fails.append("S: get_unified_cache_mode() definition not found")
        get_body = ""
    if set_body is None:
        fails.append("L: set_unified_cache_mode() definition not found")
        set_body = ""

    # O: one getenv in the file, inside a static const initializer of the accessor; none in the other sources.
    in_file = list(GETENV.finditer(code))
    in_get = list(GETENV.finditer(get_body))
    if len(in_file) != 1:
        fails.append(f"O: unified-cache.cpp calls getenv on {ENV_NAME} {len(in_file)} times, want exactly 1")
    if len(in_get) != 1:
        fails.append(f"O: get_unified_cache_mode() calls getenv on {ENV_NAME} {len(in_get)} times, want exactly 1")
    for m in in_get:
        if not static_initializer(get_body, m.start()):
            fails.append(f"O: get_unified_cache_mode() reads {ENV_NAME} on every call, outside a static const "
                         "initializer")
    for name, src in others.items():
        n = len(GETENV.findall(blank_comments(src)))
        if n:
            fails.append(f"O: {name} calls getenv on {ENV_NAME} {n} times, want 0")

    # S: the setter's value is still returned, read on each call (outside any static initializer).
    live_reads = [m for m in re.finditer(r"\bg_cache_mode\b", get_body) if not static_initializer(get_body, m.start())]
    if not any(re.match(r"\s*;", get_body[m.end():]) and re.search(r"\breturn\s*$", get_body[:m.start()])
               for m in live_reads):
        fails.append("S: get_unified_cache_mode() no longer returns g_cache_mode on each call, so "
                     "set_unified_cache_mode() would stop working after the first read")

    # L: the setter refuses once locked, then assigns.
    lock = re.search(r"if\s*\(\s*g_cache_mode_locked\s*\)\s*\{[^{}]*\breturn\s*;[^{}]*\}", set_body)
    assign = re.search(r"\bg_cache_mode\s*=\s*mode\s*;", set_body)
    if not lock or not assign or assign.start() < lock.end():
        fails.append("L: set_unified_cache_mode() must return while g_cache_mode_locked is set, then assign "
                     "g_cache_mode = mode")
    return fails


def read_sources() -> tuple[str, dict[str, str]]:
    return CACHE.read_text(), {p.name: p.read_text() for p in OTHER_SOURCES}


def substitute(src: str, pattern: str, repl: str) -> str:
    out, n = re.subn(pattern, repl, src, count=1, flags=re.S)
    if n != 1:
        raise SystemExit(f"self-test: mutant anchor missing: {pattern!r}")
    return out


def mutants(cache_src: str, others: dict[str, str]):
    """(name, pin it must fail, cache source, other sources) for each mutant."""
    # Offsets from the comment-blanked text (same length), spliced into the real source.
    span = body_span(blank_comments(cache_src), SIG_GET)
    if span is None:
        raise SystemExit("self-test: get_unified_cache_mode() not found")

    def with_body(body: str) -> str:
        return cache_src[:span[0]] + body + cache_src[span[1]:]

    # The shape before the change: the accessor reads and parses the variable on every call.
    per_call = ("{\n    const char * env = std::getenv(\"" + ENV_NAME + "\");\n"
                "    if (env && strcmp(env, \"global\") == 0) {\n        return unified_cache_mode::GLOBAL;\n    }\n"
                "    return g_cache_mode;\n}")
    yield "per-call getenv", "O", with_body(per_call), others

    # Memoizing the whole answer, setter value included, freezes set_unified_cache_mode() after the first read.
    whole = ("{\n    static const unified_cache_mode mode = [] {\n"
             "        const char * env = std::getenv(\"" + ENV_NAME + "\");\n"
             "        if (env && strcmp(env, \"global\") == 0) {\n            return unified_cache_mode::GLOBAL;\n"
             "        }\n        return g_cache_mode;\n    }();\n    return mode;\n}")
    yield "memoized setter value", "S", with_body(whole), others

    unlocked = substitute(cache_src, r"(void set_unified_cache_mode\(unified_cache_mode mode\) \{\s*)"
                                     r"if \(g_cache_mode_locked\)", r"\1if (false)")
    yield "setter ignores the lock", "L", unlocked, others

    second = "\nstatic bool g_cache_mode_probe = std::getenv(\"" + ENV_NAME + "\") != nullptr;\n"
    yield "second getenv in unified-cache.cpp", "O", cache_src + second, others

    other = dict(others)
    other["ggml-sycl.cpp"] = others["ggml-sycl.cpp"] + second
    yield "getenv in ggml-sycl.cpp", "O", cache_src, other


def main() -> int:
    cache_src, others = read_sources()
    if "--self-test" in sys.argv[1:]:
        bad = []
        n = 0
        for name, pin, src, oth in mutants(cache_src, others):
            n += 1
            fired = any(f.startswith(pin + ":") for f in check(src, oth))
            print(("PASS" if fired else "FAIL") + f": mutant '{name}' fails pin {pin}")
            if not fired:
                bad.append(name)
        if bad:
            print(f"self-test: {len(bad)} of {n} mutants not caught: {bad}")
            return 1
        print(f"self-test: all {n} mutants caught")

    fails = check(cache_src, others)
    for f in fails:
        print("FAIL " + f)
    if fails:
        print(f"{ENV_NAME} read-once contract: {len(fails)} pin(s) failed")
        return 1
    print(f"{ENV_NAME} read-once contract: all pins hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
