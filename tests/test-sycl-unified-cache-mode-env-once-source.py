#!/usr/bin/env python3
"""GGML_SYCL_UNIFIED_CACHE_MODE is read once per process: source contract.

THE COST.  get_effective_mode() asks get_unified_cache_mode() for the topology on every unified-cache lookup
(get_existing_unified_cache_for_device and its siblings), and the decode path makes such lookups per op. When the
accessor called std::getenv and strcmp on each call, getenv was 5.6 % of the submitting thread of a two-card decode
profile. The value cannot usefully change once the cache exists, and docs/backend/sycl-env-vars.md states read-once
as the convention for the backend's variables.

THE CONTRACT.  With comments blanked:
  O. The one getenv("GGML_SYCL_UNIFIED_CACHE_MODE") of the backend (any spelling: getenv, ::getenv, std::getenv)
     sits in the initializer of a function-local static (`static`, `static const` or `const static`; not constexpr)
     inside get_unified_cache_mode() in unified-cache.cpp, so a second call reads the held value and not the
     environment. No other ggml/src/ggml-sycl source calls getenv on the variable.
  P. That initializer parses once: the getenv is the argument of unified_cache_mode_from_env(), so what is held is
     the parsed mode, never the raw string, and get_unified_cache_mode() itself calls no strcmp.
  M. unified_cache_mode_from_env() maps "global", "per_device" and "auto" each to its own enum value.
  R. The held override is used: a return statement outside the initializer returns the held value (`*v`,
     `v.value()`, `v.value_or(...)`).
  S. Only the override is held: a return statement outside the initializer reads g_cache_mode, so with no valid
     value set_unified_cache_mode() keeps working until the cache locks the mode.
  L. set_unified_cache_mode() still refuses a change once g_cache_mode_locked is set, and otherwise assigns
     g_cache_mode.

NOT VACUOUS.  --self-test runs the gate on in-memory mutants of the real source and requires each to fail the named
pin, requires the correct alternative spellings to pass every pin, then requires the unmodified tree to pass.
--root <dir> checks another tree (a copy of the sources before the change).

RED on the tree before the change: pins O, P, M and R fail (the getenv and the parse ran on every call, and the
parsing helper did not exist).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ENV_NAME = "GGML_SYCL_UNIFIED_CACHE_MODE"
# Every spelling of the call ends in the bare name; the word boundary keeps secure_getenv and the like out.
GETENV = re.compile(r'\bgetenv\s*\(\s*"' + ENV_NAME + r'"\s*\)')
FROM_ENV = "unified_cache_mode_from_env"

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


# A function-local static with an initializer: `static`, `static const` or `const static`, then the type and the
# declared name. constexpr is excluded both before and after `static`.
_STATIC_DECL = re.compile(r"(?<![\w])(?:const\s+)?static\b(?:\s+const\b)?(?!\s+constexpr\b)([^=;{}()]*?)\b(\w+)\s*=")


def static_initializers(body: str) -> list[tuple[str, int, int]]:
    """(declared name, start, end) of each function-local static initializer in `body`: from after its `=` to the `;`
    that ends the declaration at the declaration's own nesting depth."""
    out = []
    for m in _STATIC_DECL.finditer(body):
        if re.search(r"\bconstexpr\s*$", body[:m.start()]):
            continue
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
        out.append((m.group(2), start, j))
    return out


def live_returns(body: str, inits: list[tuple[str, int, int]]) -> list[str]:
    """The expression of each return statement in `body` that lies outside every static initializer."""
    out = []
    for m in re.finditer(r"\breturn\b([^;]*);", body):
        if not any(s <= m.start() < e for _, s, e in inits):
            out.append(m.group(1))
    return out


SIG_GET = r"\bunified_cache_mode\s+get_unified_cache_mode\(\s*\)\s*\{"
SIG_SET = r"\bvoid\s+set_unified_cache_mode\(\s*unified_cache_mode\s+mode\s*\)\s*\{"
SIG_FROM_ENV = r"\b" + FROM_ENV + r"\(\s*const\s+char\s*\*\s*\w+\s*\)\s*\{"
MAPPING = {"global": "GLOBAL", "per_device": "PER_DEVICE", "auto": "AUTO"}


def check(cache_src: str, others: dict[str, str]) -> list[str]:
    """The failed pins, each as 'PIN: reason'. Empty means the contract holds."""
    fails: list[str] = []
    code = blank_comments(cache_src)

    get_body = body_of(code, SIG_GET)
    set_body = body_of(code, SIG_SET)
    if get_body is None:
        fails += [f"{p}: get_unified_cache_mode() definition not found" for p in "OPRS"]
        get_body = ""
    if set_body is None:
        fails.append("L: set_unified_cache_mode() definition not found")
        set_body = ""
    inits = static_initializers(get_body)

    # O: one getenv in the file, inside a static initializer of the accessor; none in the other sources.
    in_file = list(GETENV.finditer(code))
    in_get = list(GETENV.finditer(get_body))
    if len(in_file) != 1:
        fails.append(f"O: unified-cache.cpp calls getenv on {ENV_NAME} {len(in_file)} times, want exactly 1")
    if len(in_get) != 1:
        fails.append(f"O: get_unified_cache_mode() calls getenv on {ENV_NAME} {len(in_get)} times, want exactly 1")
    held = [(n, s, e) for n, s, e in inits if any(s <= m.start() < e for m in in_get)]
    if in_get and not held:
        fails.append(f"O: get_unified_cache_mode() reads {ENV_NAME} on every call, outside a static initializer")
    for name, src in others.items():
        n = len(GETENV.findall(blank_comments(src)))
        if n:
            fails.append(f"O: {name} calls getenv on {ENV_NAME} {n} times, want 0")

    # P: the held initializer parses the variable through the helper; the accessor compares no strings.
    parse = re.compile(r"\b" + FROM_ENV + r"\s*\(\s*(?:std\s*)?(?:::\s*)?getenv\s*\(\s*\"" + ENV_NAME + r"\"\s*\)\s*\)")
    if not any(parse.search(get_body[s:e]) for _, s, e in held):
        fails.append(f"P: the held initializer is not {FROM_ENV}(getenv(\"{ENV_NAME}\")): a raw string would be held "
                     "and parsed on every call")
    if re.search(r"\bstrcmp\s*\(", get_body):
        fails.append("P: get_unified_cache_mode() calls strcmp, so it parses on every call")

    # M: each value maps to its own mode.
    from_env = body_of(code, SIG_FROM_ENV)
    if from_env is None:
        fails.append(f"M: {FROM_ENV}(const char *) definition not found")
    else:
        for text, mode in MAPPING.items():
            pat = (r"strcmp\s*\(\s*\w+\s*,\s*\"" + text + r"\"\s*\)\s*==\s*0\s*\)\s*\{?\s*return\s+"
                   r"(?:unified_cache_mode::)?" + mode + r"\s*;")
            if not re.search(pat, from_env):
                fails.append(f"M: {FROM_ENV}() does not map \"{text}\" to unified_cache_mode::{mode}")

    returns = live_returns(get_body, inits)

    # R: the held override is returned.
    names = [n for n, _, _ in held]
    if not any(re.search(r"\*\s*" + n + r"\b|\b" + n + r"\s*\.\s*value(?:_or)?\s*\(", r) for r in returns
               for n in names):
        fails.append("R: get_unified_cache_mode() never returns the held override")

    # S: the setter's value is still returned, read on each call.
    if not any(re.search(r"\bg_cache_mode\b", r) for r in returns):
        fails.append("S: get_unified_cache_mode() no longer returns g_cache_mode on each call, so "
                     "set_unified_cache_mode() would stop working after the first read")

    # L: the setter refuses once locked, then assigns.
    lock = re.search(r"if\s*\(\s*g_cache_mode_locked\s*\)\s*\{[^{}]*\breturn\s*;[^{}]*\}", set_body)
    assign = re.search(r"\bg_cache_mode\s*=\s*mode\s*;", set_body)
    if not lock or not assign or assign.start() < lock.end():
        fails.append("L: set_unified_cache_mode() must return while g_cache_mode_locked is set, then assign "
                     "g_cache_mode = mode")
    return fails


ROOT = Path(__file__).resolve().parent.parent


def read_sources(root: Path = ROOT) -> tuple[str, dict[str, str]]:
    sycl = root / "ggml" / "src" / "ggml-sycl"
    cache = sycl / "unified-cache.cpp"
    others = {}
    for p in sorted(sycl.rglob("*")):
        if p.suffix in (".cpp", ".hpp") and p.is_file() and p != cache:
            others[str(p.relative_to(sycl))] = p.read_text(errors="replace")
    return cache.read_text(), others


def substitute(src: str, pattern: str, repl: str) -> str:
    out, n = re.subn(pattern, repl, src, count=1, flags=re.S)
    if n != 1:
        raise SystemExit(f"self-test: mutant anchor missing: {pattern!r}")
    return out


E = '"' + ENV_NAME + '"'


def held_body(decl: str, ret: str) -> str:
    return ("{\n    " + decl + " env_mode =\n        " + FROM_ENV + "(std::getenv(" + E + "));\n    " + ret + "\n}")


IF_RETURN = "if (env_mode) {\n        return *env_mode;\n    }\n    return g_cache_mode;"

# Correct alternative spellings: each must pass every pin.
CORRECT = {
    "value_or": held_body("static const std::optional<unified_cache_mode>", "return env_mode.value_or(g_cache_mode);"),
    "ternary": held_body("static const std::optional<unified_cache_mode>",
                         "return env_mode ? *env_mode : g_cache_mode;"),
    "static non-const": held_body("static std::optional<unified_cache_mode>", IF_RETURN),
    "const static": held_body("const static std::optional<unified_cache_mode>", IF_RETURN),
    "::getenv": ("{\n    static const std::optional<unified_cache_mode> env_mode = " + FROM_ENV + "(::getenv(" + E +
                 "));\n    " + IF_RETURN + "\n}"),
}


def mutants(cache_src: str, others: dict[str, str]):
    """(name, pin it must fail, cache source, other sources) for each mutant."""
    # Offsets from the comment-blanked text (same length), spliced into the real source.
    span = body_span(blank_comments(cache_src), SIG_GET)
    if span is None:
        raise SystemExit("self-test: get_unified_cache_mode() not found")

    def with_body(body: str) -> str:
        return cache_src[:span[0]] + body + cache_src[span[1]:]

    # The shape before the change: the accessor reads and parses the variable on every call.
    per_call = ("{\n    const char * env = std::getenv(" + E + ");\n"
                "    if (env && strcmp(env, \"global\") == 0) {\n        return unified_cache_mode::GLOBAL;\n    }\n"
                "    return g_cache_mode;\n}")
    yield "per-call getenv", "O", with_body(per_call), others

    yield ("constexpr static", "O",
           with_body(held_body("static constexpr std::optional<unified_cache_mode>", IF_RETURN)), others)

    # Memoizing the whole answer, setter value included, freezes set_unified_cache_mode() after the first read.
    whole = ("{\n    static const unified_cache_mode mode = [] {\n"
             "        const char * env = std::getenv(" + E + ");\n"
             "        if (env && strcmp(env, \"global\") == 0) {\n            return unified_cache_mode::GLOBAL;\n"
             "        }\n        return g_cache_mode;\n    }();\n    return mode;\n}")
    yield "memoized setter value", "S", with_body(whole), others

    # A held raw string still costs a strcmp per call.
    raw = ("{\n    static const char * env = std::getenv(" + E + ");\n"
           "    if (env && strcmp(env, \"global\") == 0) {\n        return unified_cache_mode::GLOBAL;\n    }\n"
           "    return g_cache_mode;\n}")
    yield "raw string held, parsed per call", "P", with_body(raw), others

    yield "override ignored", "R", with_body(held_body("static const std::optional<unified_cache_mode>",
                                                       "(void) env_mode;\n    return g_cache_mode;")), others

    swapped = substitute(cache_src, r"(\"global\"\) == 0\) \{\s*return unified_cache_mode::)GLOBAL;",
                         r"\1PER_DEVICE;")
    yield "parser maps global to PER_DEVICE", "M", swapped, others

    unlocked = substitute(cache_src, r"(void set_unified_cache_mode\(unified_cache_mode mode\) \{\s*)"
                                     r"if \(g_cache_mode_locked\)", r"\1if (false)")
    yield "setter ignores the lock", "L", unlocked, others

    second = "\nstatic bool g_cache_mode_probe = std::getenv(" + E + ") != nullptr;\n"
    yield "second getenv in unified-cache.cpp", "O", cache_src + second, others

    for name, text in (("std::getenv in ggml-sycl.cpp", second),
                       ("::getenv in common.hpp", "\nstatic bool g_probe = ::getenv(" + E + ") != nullptr;\n")):
        other = dict(others)
        key = "ggml-sycl.cpp" if "ggml-sycl.cpp" in name else "common.hpp"
        other[key] = others[key] + text
        yield name, "O", cache_src, other


def main() -> int:
    args = sys.argv[1:]
    root = ROOT
    if "--root" in args:
        root = Path(args[args.index("--root") + 1]).resolve()
    cache_src, others = read_sources(root)
    if "--self-test" in args:
        bad = []
        n = 0
        for name, pin, src, oth in mutants(cache_src, others):
            n += 1
            fired = any(f.startswith(pin + ":") for f in check(src, oth))
            print(("PASS" if fired else "FAIL") + f": mutant '{name}' fails pin {pin}")
            if not fired:
                bad.append(name)
        span = body_span(blank_comments(cache_src), SIG_GET)
        if span is None:
            raise SystemExit("self-test: get_unified_cache_mode() not found")
        for name, body in CORRECT.items():
            n += 1
            got = check(cache_src[:span[0]] + body + cache_src[span[1]:], others)
            print(("PASS" if not got else "FAIL") + f": correct form '{name}' passes every pin" +
                  ("" if not got else f" ({got})"))
            if got:
                bad.append(name)
        if bad:
            print(f"self-test: {len(bad)} of {n} cases wrong: {bad}")
            return 1
        print(f"self-test: all {n} cases right")

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
