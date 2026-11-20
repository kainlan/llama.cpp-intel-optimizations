#!/usr/bin/env python3
"""Source contract: ggml/src/ggml-sycl/moe-swap-policy.hpp is plain host code with no state of its own.

Host-only: reads the one header, builds nothing, loads no model and touches no device.

The swap policy (tests/test-sycl-moe-swap-policy.cpp) decides at a graph boundary which resident expert leaves and
which host expert enters. The unit test shows the decisions are right. It cannot show what the header does not do:
reach a device, allocate through the cache, log, read the environment or keep state outside the struct its caller
owns. Two policies in one process, or one policy called from two places, would then share something the caller never
passed in. This gate holds that by text:

  S1  the header includes the standard library only, from a short list of headers a pure computation needs: no quoted
      include, no ggml, no SYCL, nothing outside the list;
  S2  nothing mutable lives outside a state struct the caller owns: no `static`, `thread_local`, `extern`, `mutable`
      or `volatile` anywhere (a namespace constant is `constexpr`), no non-const `inline` variable (the object's
      constness is read from where `const` binds: `inline const int * p` is a mutable pointer, `inline int * const p`
      is not), and no atomic or lock, so the policy has nothing to synchronise because it shares nothing;
  S3  no device, allocation-through-the-cache, logging or environment access: no `sycl`, `malloc`/`free`, the unified
      cache's names, `GGML_LOG`, `printf`, `std::cout`/`cerr` or `getenv`.

Checks run on comment-blanked text (the header's comments say what it does not do, in those very words).
NOT VACUOUS: --self-test runs the gate on in-memory mutants of the real header. Each mutant names the pin that must
catch it, and is reported if it survives, if a different pin catches it, or if its anchor no longer matches exactly
once (NOT APPLIED). The self-test also proves that last report fires. ctest registers it with --self-test.

WHAT THIS DOES NOT PROVE. It reads text. An `inline` variable with a parenthesised initializer that is made only of
identifiers, such as `inline int g(x);`, cannot be told from a function declaration by text and is not caught; a
literal argument (`inline int g(5);`) is. A mutable object reached through a typedef or `using` alias that hides a
`const` is judged by the spelling in front of it. That the policy decides correctly and stays inside its time bound
is the unit test's job; that a caller keeps one state per model and passes it by reference is the caller's.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "ggml/src/ggml-sycl/moe-swap-policy.hpp"

# The standard headers a pure computation over vectors and integers needs. Adding one is a decision, so it is listed.
ALLOWED_INCLUDES = {"algorithm", "cstddef", "cstdint", "vector"}
STATE_KEYWORDS = re.compile(r"\b(?:static|thread_local|extern|mutable|volatile)\b")
# A C++17 `inline` variable is a global with external linkage and needs none of the words above.
INLINE = re.compile(r"\binline\b")
TEMPLATE_ARGS = re.compile(r"<[^<>]*>")
# An argument list that holds a literal argument cannot be a parameter list: `inline int g(5);` is a variable.
LITERAL_ARG = re.compile(r"^(?:[+-]?\d[\w.']*|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|nullptr|true|false)$")


def _matching_paren(text: str, open_at: int) -> int:
    depth = 0
    for k in range(open_at, len(text)):
        if text[k] == "(":
            depth += 1
        elif text[k] == ")":
            depth -= 1
            if depth == 0:
                return k
    return -1


def _top_level_args(text: str) -> list[str]:
    args, depth, cur = [], 0, ""
    for c in text:
        if c in "([{<":
            depth += 1
        elif c in ")]}>":
            depth -= 1
        if c == "," and depth == 0:
            args.append(cur.strip())
            cur = ""
        else:
            cur += c
    if cur.strip():
        args.append(cur.strip())
    return args


def _is_const_object(head: str) -> bool:
    """Is the object a declaration head declares immutable? `constexpr` makes it so; otherwise a pointer is const only
    if `const` follows its last `*` (`int * const p`), and anything else only needs a `const`. Template arguments are
    cut first, so `std::vector<const int>` is a mutable vector."""
    while TEMPLATE_ARGS.search(head):
        head = TEMPLATE_ARGS.sub(" ", head)
    if re.search(r"\bconstexpr\b", head):
        return True
    if "*" in head:
        head = head[head.rfind("*") + 1 :]
    return re.search(r"\bconst\b", head) is not None


def find_inline_variable(code: str) -> int | None:
    """The offset of the first non-const `inline` variable in `code`, or None. `inline` starts a function when a `(`
    comes before any `=`, `;` or `{`; that is a variable only if its parentheses hold a literal argument."""
    for m in INLINE.finditer(code):
        rest = code[m.end() :]
        t = re.search(r"[;={(]", rest)
        if not t:
            continue
        head = rest[: t.start()]
        if re.search(r"\boperator\b", head) or len(head.split()) < 2:
            continue
        if t.group() == "(":
            close = _matching_paren(rest, t.start())
            if close < 0 or not any(LITERAL_ARG.match(a) for a in _top_level_args(rest[t.start() + 1 : close])):
                continue
        if not _is_const_object(head):
            return m.start()
    return None
SYNC_NAMES = re.compile(
    r"\bstd::(?:atomic\w*|mutex|shared_mutex|recursive_mutex|lock_guard|unique_lock|call_once|once_flag)\b"
)
FORBIDDEN_CALLS = re.compile(
    r"\bsycl\b|\b(?:malloc|calloc|realloc|free|aligned_alloc)\s*\(|\bunified_\w+|\bmem_handle\b|\balloc_owner\b|"
    r"\bGGML_LOG\w*|\b(?:f|s|sn)?printf\s*\(|\bstd::(?:cout|cerr|clog)\b|\bgetenv\s*\("
)
INCLUDE = re.compile(r"^[ \t]*#[ \t]*include[ \t]*(<[^>\n]*>|\"[^\"\n]*\")", re.MULTILINE)

_COMMENT_OR_LITERAL = re.compile(r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'", re.DOTALL)


class ContractError(AssertionError):
    pass


class MutantNotApplied(Exception):
    pass


def blank(text: str) -> str:
    """Comments blanked, string and character literals kept, so `#include "x"` still reads as an include."""

    def repl(m: re.Match) -> str:
        t = m.group(0)
        if t.startswith("//") or t.startswith("/*"):
            return "".join("\n" if c == "\n" else " " for c in t)
        return t

    return _COMMENT_OR_LITERAL.sub(repl, text)


def check(raw: str) -> None:
    code = blank(raw)

    # S1 ---------------------------------------------------------------------------------------------------------------
    includes = INCLUDE.findall(code)
    if not includes:
        raise ContractError("FAIL [pin S1]: the header includes nothing, so the pin read the wrong text")
    for inc in includes:
        if inc.startswith('"'):
            raise ContractError(f"FAIL [pin S1]: the header includes a project header {inc}")
        name = inc[1:-1]
        if name not in ALLOWED_INCLUDES:
            raise ContractError(f"FAIL [pin S1]: the header includes <{name}>, which is not a listed standard header")

    # S2 ---------------------------------------------------------------------------------------------------------------
    # static_cast and static_assert are compile-time words, not storage: cut them out before looking for `static`.
    stateless = re.sub(r"\bstatic_(?:cast|assert)\b", "  ", code)
    m = STATE_KEYWORDS.search(stateless)
    if m:
        line = stateless.count("\n", 0, m.start()) + 1
        raise ContractError(f"FAIL [pin S2]: `{m.group(0)}` at line {line}: state outside the caller's struct")
    at = find_inline_variable(stateless)
    if at is not None:
        line = stateless.count("\n", 0, at) + 1
        raise ContractError(f"FAIL [pin S2]: an inline variable at line {line}: global state outside the caller")
    m = SYNC_NAMES.search(code)
    if m:
        raise ContractError(f"FAIL [pin S2]: `{m.group(0)}`: the policy shares nothing and needs no atomic or lock")

    # S3 ---------------------------------------------------------------------------------------------------------------
    m = FORBIDDEN_CALLS.search(code)
    if m:
        line = code.count("\n", 0, m.start()) + 1
        raise ContractError(f"FAIL [pin S3]: `{m.group(0).strip()}` at line {line}: device, allocation, log or env use")


def mutate(raw: str, edits: list[tuple[str, str]]) -> str:
    for old, new in edits:
        n = raw.count(old)
        if n != 1:
            raise MutantNotApplied(f"NOT APPLIED: anchor appears {n} times, not once: {old[:70]!r}")
        raw = raw.replace(old, new)
    return raw


def self_test(raw: str) -> int:
    failures: list[str] = []
    n_mutants = 0
    n_negatives = 0

    def expect_pass(name: str, edits: list[tuple[str, str]]) -> None:
        """A mutant that is legal for this header and must NOT be caught: the false-positive side of the pins."""
        nonlocal n_negatives
        n_negatives += 1
        try:
            check(mutate(raw, edits))
        except MutantNotApplied as e:
            failures.append(name)
            print(f"  self-test: {name}: {e}")
        except ContractError as e:
            failures.append(name)
            print(f"  self-test: {name}: a legal declaration was caught: {e}")

    def expect_fail(name: str, pin: str, edits: list[tuple[str, str]]) -> None:
        nonlocal n_mutants
        n_mutants += 1
        try:
            mutant = mutate(raw, edits)
        except MutantNotApplied as e:
            failures.append(name)
            print(f"  self-test: {name}: {e}")
            return
        try:
            check(mutant)
        except ContractError as e:
            if f"[pin {pin}]" in str(e):
                return
            failures.append(name)
            print(f"  self-test: {name}: caught by a different pin than {pin}: {e}")
            return
        failures.append(name)
        print(f"  self-test: {name}: SURVIVED (pin {pin} did not catch it)")

    anchor = "#include <vector>"
    ns = "namespace ggml_sycl {\n"
    expect_fail("includes-a-project-header", "S1", [(anchor, '#include "common.hpp"\n' + anchor)])
    expect_fail("includes-ggml", "S1", [(anchor, '#include "ggml.h"\n' + anchor)])
    expect_fail("includes-sycl", "S1", [(anchor, "#include <sycl/sycl.hpp>\n" + anchor)])
    expect_fail("includes-an-unlisted-std-header", "S1", [(anchor, "#include <mutex>\n" + anchor)])
    expect_fail("includes-nothing-read", "S1", [("#include <algorithm>\n", ""), ("#include <cstddef>\n", ""),
                                                ("#include <cstdint>\n", ""), (anchor + "\n", "")])
    expect_fail("static-counter", "S2",
                [(ns, ns + "inline int next_id() {\n    static int n = 0;\n    return ++n;\n}\n")])
    expect_fail("inline-global-vector", "S2", [(ns, ns + "inline std::vector<int> g_scratch;\n")])
    expect_fail("inline-global-counter", "S2", [(ns, ns + "inline int g_n = 0;\n")])
    expect_fail("inline-global-braced", "S2", [(ns, ns + "inline int g_k{5};\n")])
    expect_fail("inline-global-array", "S2", [(ns, ns + "inline int g_a[4];\n")])
    expect_fail("inline-global-array-initialised", "S2", [(ns, ns + "inline int g_b[4] = {};\n")])
    expect_fail("inline-global-paren-literal", "S2", [(ns, ns + "inline int g_p(5);\n")])
    expect_fail("inline-global-paren-string", "S2", [(ns, ns + 'inline std::string g_s("x");\n')])
    expect_fail("inline-pointer-to-const-is-mutable", "S2", [(ns, ns + "inline const int * g_q = nullptr;\n")])
    expect_fail("inline-pointer", "S2", [(ns, ns + "inline int * g_r = nullptr;\n")])
    expect_fail("inline-vector-of-const-is-mutable", "S2", [(ns, ns + "inline std::vector<const int> g_v;\n")])
    expect_fail("inline-lambda-variable", "S2", [(ns, ns + "inline auto g_f = [](int a) { return a; };\n")])
    expect_fail("thread-local-scratch", "S2", [(ns, ns + "thread_local std::vector<int> scratch;\n")])
    expect_fail("extern-state", "S2", [(ns, ns + "extern int g_swaps;\n")])
    member_anchor = "    // The usage of one cell as of the last absorb;"
    expect_fail("mutable-member", "S2", [(member_anchor, "    mutable int scratch_n = 0;\n" + member_anchor)])
    expect_fail("atomic-counter", "S2", [(ns, ns + "inline int bump(std::atomic<int> & a) { return ++a; }\n")])
    expect_fail("mutex-member", "S2", [(member_anchor, "    std::mutex lock;\n" + member_anchor)])
    expect_fail("device-use", "S3", [(ns, ns + "inline void * dev(sycl::queue & q) { return nullptr; }\n")])
    expect_fail("malloc-use", "S3", [(ns, ns + "inline void * grab() { return malloc(8); }\n")])
    expect_fail("unified-cache-use", "S3", [(ns, ns + "inline void * grab() { return unified_alloc(8); }\n")])
    expect_fail("logs", "S3", [(ns, ns + "inline void say() { GGML_LOG_INFO(\"x\"); }\n")])
    expect_fail("prints", "S3", [(ns, ns + "inline void say() { printf(\"x\"); }\n")])
    expect_fail("reads-the-environment", "S3", [(ns, ns + "inline bool on() { return getenv(\"X\") != nullptr; }\n")])

    # The false-positive side: legal declarations the pin must let through.
    expect_pass("legal-inline-function", [(ns, ns + "inline int twice(int a) { return a * 2; }\n")])
    expect_pass("legal-inline-function-declaration", [(ns, ns + "inline int twice(int a);\n")])
    expect_pass("legal-inline-function-no-parameters", [(ns, ns + "inline int zero();\n")])
    expect_pass("legal-inline-function-default-argument", [(ns, ns + "inline void put(int a = 5);\n")])
    expect_pass("legal-inline-function-templated-return",
                [(ns, ns + "inline std::vector<int> make(int n) { return std::vector<int>(n); }\n")])
    expect_pass("legal-inline-operator",
                [(ns, ns + "inline bool operator==(const int & a, const int & b) { return a == b; }\n")])
    expect_pass("legal-inline-constexpr", [(ns, ns + "inline constexpr int k_three = 3;\n")])
    expect_pass("legal-inline-const", [(ns, ns + "inline const int k_four = 4;\n")])
    expect_pass("legal-inline-const-pointer", [(ns, ns + "inline int * const k_ptr = nullptr;\n")])
    expect_pass("legal-inline-const-array", [(ns, ns + "inline const int k_arr[2] = { 1, 2 };\n")])
    expect_pass("legal-inline-constexpr-pointer", [(ns, ns + "inline constexpr const char * k_name = \"x\";\n")])

    # The NOT APPLIED report must fire on an anchor the header does not hold, or a stale mutant would pass unseen.
    try:
        mutate(raw, [("this anchor is in no source", "x")])
        failures.append("stale-anchor-not-reported")
        print("  self-test: a stale mutant anchor is NOT reported")
    except MutantNotApplied:
        print("  self-test: a stale mutant anchor is reported")

    try:
        check(raw)
        print("  self-test: unmodified header passes")
    except ContractError as e:
        print(f"  self-test: unmodified header FAILS: {e}")
        failures.append("unmodified")

    if failures:
        print(f"SELF-TEST FAIL: {', '.join(failures)}")
        return 1
    print(
        f"SELF-TEST PASS: {n_mutants} mutants caught by their pins, {n_negatives} legal declarations let through, "
        "unmodified header passes"
    )
    return 0


def main(argv: list[str]) -> int:
    raw = HEADER.read_text()
    if "--self-test" in argv:
        return self_test(raw)
    try:
        check(raw)
    except ContractError as e:
        print(e)
        return 1
    print("PASS: moe-swap-policy.hpp includes the standard library only and keeps no state outside its caller's struct")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
