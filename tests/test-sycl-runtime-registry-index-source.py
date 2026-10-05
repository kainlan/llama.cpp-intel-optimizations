"""Source contract for llama.cpp-ii25: the runtime allocation registry's containment index cannot drift from the registry.

unified_lookup_runtime_allocation() answers "which registered allocation contains this pointer" through
g_runtime_alloc_index (range-index.hpp) instead of scanning every row of g_runtime_alloc_registry (61.65% of Qwen3.8
decode CPU with a 512-expert MoE). The index is correct only while every insert and erase of a registry row also updates
it, and while a row's extent changes only through the helpers that update the index with it. What this file enforces:

(A) g_runtime_alloc_registry is mutated (emplace/erase/insert/clear/extract/swap/operator[]/assignment) ONLY inside the
    runtime_registry_emplace_locked / runtime_registry_erase_locked helpers, and every other mention of the registry is
    a read member call, a const range-for or the declaration (so no alias, pointer, or reference argument, and no at()).
(B) A registered row is never WRITTEN THROUGH a plain access path outside those helpers:
    - an assignment, compound assignment (+=, |=, <<= ...) or ++/-- of `second.handle`, `second.handle.ptr` or
      `second.handle.size`, anywhere in the file;
    - a whole-row assignment (`it->second = rec;`, `registry.find(p)->second = rec;`) anywhere in the file, except the
      three non-registry sites in ROW_WRITE_ALLOWLIST (each checked by hand; an entry that stops matching fails the gate)
      and the one in runtime_registry_assign_locked, which resizes or re-keys the index before it moves a row over the old;
    - in a function that touches the registry (names it, or takes a runtime_registry_iterator / the map's own iterator
      type): a non-const reference or pointer bound to a row (`auto & r = it->second;`, `auto q = &it->second;`,
      `auto * q = &it->second.handle;`, `alloc_metadata & h = ...second.handle;`, `decltype(auto) r = it->second;`), a
      non-const structured binding (`auto & [k, v] = *it;`), and a lambda or function parameter that is a non-const
      reference to a row or its metadata (`[&](runtime_alloc_record & r)`, `[](auto & r)`).
(C) Neither registry nor index is touched from another translation unit (both are file-static in unified-cache.cpp).
(D) The lookup consults the index and does not iterate the registry.

NOT COVERED, stated so nobody mistakes this for a proof. The gate reads text, not the AST:
    - writes by CALL, even directly on a row: `std::swap(it->second, x)`, `std::exchange(it->second.handle.size, 0)`,
      `memcpy(&it->second, ...)`, `memset(&it->second.handle, ...)`;
    - `const_cast` of a const view of a row;
    - a helper that RETURNS `runtime_alloc_record &` / `alloc_metadata &` (the call site never names the registry), and a
      non-const reference parameter in a function that does not itself touch the registry;
    - `#if 0` blocks and raw string literals, which the comment stripper does not understand;
    - functions are split at a closing brace in column 0, which is how this file is formatted.
Those are bounded only by review; (A) and the alias/parameter rules above keep any such write on code that already looks
wrong.

Host-only, pure text assertions. llama_test_pytest hands this file to pytest.main(), so the checks live inside test_*()
functions. Checks run against COMMENT-STRIPPED text, and each has a mutation witness so it is known to fail on the
regression it guards.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SYCL_DIR = ROOT / "ggml/src/ggml-sycl"
UNIFIED_CACHE_CPP = (SYCL_DIR / "unified-cache.cpp").read_text()

_LEXEME_RE = re.compile(
    r'"(?:\\.|[^"\\\n])*"'  # string literal (kept)
    r"|'(?:\\.|[^'\\\n])*'"  # char literal (kept)
    r"|//[^\n]*"  # line comment (dropped)
    r"|/\*.*?\*/",  # block comment (dropped)
    flags=re.DOTALL,
)


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


MUTATION_RE = re.compile(
    r"\bg_runtime_alloc_registry\s*(?:\.\s*(?:emplace|erase|insert|insert_or_assign|try_emplace|clear|extract|swap|merge"
    r"|rehash|reserve)\b|\[|=[^=])"
    r"|\bswap\s*\(\s*g_runtime_alloc_registry\b"
)
# Signatures as regexes: clang-format realigns the parameter list, so the whitespace between tokens must not matter.
HELPERS = (
    r"runtime_registry_emplace_locked\(\s*void\s*\*\s*ptr\s*,\s*runtime_alloc_record\s+rec\s*\)\s*\{",
    r"runtime_registry_erase_locked\(\s*runtime_registry_iterator\s+it\s*\)\s*noexcept\s*\{",
)
# Members that only read the registry. Any other use of the registry outside the helpers (an alias, a pointer, a
# reference handed to a function, a non-const range-for) could mutate it behind the index's back.
# `at` is deliberately absent: it returns a mutable reference and a read can always be spelled find.
READ_MEMBER_RE = re.compile(
    r"\s*\.\s*(?:find|end|begin|cbegin|cend|crbegin|crend|size|max_size|empty|count|contains|equal_range|bucket_count"
    r"|load_factor|max_load_factor)\b"
)
CONST_RANGE_FOR_RE = re.compile(r"const\s+auto\s*&\s*\w+\s*:\s*$")
# A non-const reference or pointer bound to a registry row, through which a row's geometry could be rewritten. Only looked
# for in functions that touch the registry.
ROW_ALIAS_RE = re.compile(
    # auto & r = it->second;  auto * q = &it->second.handle;  alloc_metadata & h = it->second.handle;
    r"(?<!const )(?<!const\t)\b(?:auto|alloc_metadata|runtime_alloc_record)\s*(?:&&?|\*)\s*\w+\s*(?:=|\{|\()[^;{}]*\bsecond\b"
    # auto q = &it->second;   (a pointer deduced from an address-of; `const auto q` is still a pointer to a mutable row)
    r"|\bauto\s+\w+\s*=\s*&[^;{}]*\bsecond\b"
    # decltype(auto) r = it->second;
    r"|\bdecltype\s*\(\s*auto\s*\)\s*\w+\s*=[^;{}]*\bsecond\b"
)
# A non-const structured binding: `auto & [k, v] = *it;`, `auto && [k, v] = ...;`, `for (auto & [k, v] : map)`.
STRUCTURED_BINDING_RE = re.compile(r"(?<!const )(?<!const\t)\bauto\s*&{1,2}\s*\[[^\]]*\]\s*(?:=|:|\{|\()")
# A function or lambda parameter that is a non-const reference to a row or its metadata, or a generic one.
ROW_REF_PARAM_RE = re.compile(
    r"[(,]\s*(?:runtime_alloc_record|alloc_metadata)\s*&&?\s*\w*\s*(?=[,)])|[(,]\s*auto\s*&&?\s*\w+\s*(?=[,)])"
)
# Functions that deal in registry rows even when they never name the registry (an iterator parameter, say).
REGISTRY_FUNCTION_MARKERS = (
    "g_runtime_alloc_registry",
    "runtime_registry_iterator",
    "unordered_map<void *, runtime_alloc_record>",
)
# The `second` writes in the file that are NOT registry rows, each checked by hand. Keyed by the stripped source line;
# an entry that no longer appears in the file fails test_row_write_allowlist_is_not_stale.
ROW_WRITE_ALLOWLIST = {
    "current->second = replacement;": "the MoE mmid plan registry's entry",
    "entry.second += bytes;": "offload host-alloc stats, keyed by tag",
    "mapped->second = replacement;": "the id_to_key_ map",
}
_ASSIGN = r"(?:(?:[-+*/%|&^]|<<|>>)?=(?!=)|\+\+|--)"  # =, op=, ++, -- ; not ==, <=, >=, !=
# A row's identity or geometry written through an iterator/reference into the registry: `second.handle = x`,
# `second.handle.size += n`, `second.handle.ptr++`, `++second.handle.size`.
GEOMETRY_WRITE_RE = re.compile(
    r"(?:->|\.)second\s*\.\s*handle\s*(?:\.\s*(?:ptr|size)\s*)?" + _ASSIGN
    + r"|(?:\+\+|--)\s*[\w.>()\-]*?(?:->|\.)second\s*\.\s*handle\s*\.\s*(?:ptr|size)\b"
)
# A whole row written through an iterator/reference: `it->second = rec;`, `registry.find(p)->second = rec;`. Scanned over
# the whole file (a helper that takes the iterator never names the registry), minus ROW_WRITE_ALLOWLIST.
ROW_WRITE_RE = re.compile(r"(?:->|\.)second\s*" + _ASSIGN)


def function_span(code: str, signature: str):
    m = re.search(signature, code)
    assert m is not None, f"{signature!r} is not defined"
    start = m.start()
    end = code.find("\n}\n", start)
    assert end != -1, f"could not bound {signature!r}"
    return start, end


def registry_mutations_outside_helpers(code: str):
    spans = [function_span(code, h) for h in HELPERS]
    bad = []
    for m in MUTATION_RE.finditer(code):
        if not any(s <= m.start() < e for s, e in spans):
            line = code.count("\n", 0, m.start()) + 1
            bad.append((line, code[m.start() : m.start() + 60].replace("\n", " ")))
    return bad


def registry_uses_that_are_not_reads(code: str):
    """Every mention of the registry outside the helpers must be a read member call, a const range-for, or its declaration."""
    spans = [function_span(code, h) for h in HELPERS]
    bad = []
    for m in re.finditer(r"\bg_runtime_alloc_registry\b", code):
        if any(s <= m.start() < e for s, e in spans):
            continue
        tail = code[m.end() : m.end() + 80]
        pre = code[max(0, m.start() - 80) : m.start()]
        if READ_MEMBER_RE.match(tail) or CONST_RANGE_FOR_RE.search(pre):
            continue
        if re.search(r">\s*$", pre) and tail.lstrip().startswith(";"):
            continue  # the declaration
        line = code.count("\n", 0, m.start()) + 1
        bad.append((line, (pre[-30:] + "|" + tail[:30]).replace("\n", " ")))
    return bad


def touches_registry(body: str) -> bool:
    return any(marker in body for marker in REGISTRY_FUNCTION_MARKERS)


def registry_function_scan(code: str, pattern):
    """Matches of `pattern` inside functions that touch the registry; (line, text)."""
    bad = []
    pos = 0
    for end in re.finditer(r"\n}\n", code):
        body = code[pos : end.end()]
        if touches_registry(body):
            for m in pattern.finditer(body):
                bad.append((code.count("\n", 0, pos + m.start()) + 1, m.group(0)[:70]))
        pos = end.end()
    return bad


def row_aliases_in_registry_functions(code: str):
    """A non-const reference/pointer to a registry row, in a function that touches the registry."""
    return registry_function_scan(code, ROW_ALIAS_RE)


def non_const_bindings_in_registry_functions(code: str):
    """Non-const structured bindings and non-const row/generic reference parameters, in functions that touch the registry."""
    return registry_function_scan(code, STRUCTURED_BINDING_RE) + registry_function_scan(code, ROW_REF_PARAM_RE)


# The one helper that replaces a row in place: it resizes or re-keys the index first, then moves the new row over the old.
ASSIGN_HELPER = r"runtime_registry_assign_locked\(\s*void\s*\*\s*ptr\s*,\s*const\s+runtime_alloc_record\s*&\s*rec\s*\)\s*\{"


def row_writes_outside_the_allowlist(code: str):
    bad = []
    lines = code.split("\n")
    start, end = function_span(code, ASSIGN_HELPER)
    for m in ROW_WRITE_RE.finditer(code):
        line = code.count("\n", 0, m.start()) + 1
        if start <= m.start() < end:
            continue
        if lines[line - 1].strip() not in ROW_WRITE_ALLOWLIST:
            bad.append((line, lines[line - 1].strip()[:70]))
    return bad


def geometry_writes(code: str):
    return [(code.count("\n", 0, m.start()) + 1) for m in GEOMETRY_WRITE_RE.finditer(code)]


def lookup_body(code: str) -> str:
    sig = r"bool\s+unified_lookup_runtime_allocation\(const void \* ptr, alloc_metadata \* out, sycl::queue \*\* queue_out\)\s*\{"
    s, e = function_span(code, sig)
    return code[s:e]


CODE = strip_comments(UNIFIED_CACHE_CPP)


def test_registry_is_mutated_only_through_the_index_keeping_helpers():
    assert registry_mutations_outside_helpers(CODE) == []


def test_the_helpers_really_mutate_the_registry():
    # Positive control: the (A) scan finds the helpers' own emplace and erase, so a zero above is not a dead pattern.
    spans = [function_span(CODE, h) for h in HELPERS]
    inside = [m for m in MUTATION_RE.finditer(CODE) if any(s <= m.start() < e for s, e in spans)]
    assert len(inside) >= 3


@pytest.mark.parametrize(
    "line",
    [
        "g_runtime_alloc_registry.emplace(ptr, rec);",
        "g_runtime_alloc_registry.erase(it);",
        "g_runtime_alloc_registry.erase(ptr);",
        "g_runtime_alloc_registry[ptr] = rec;",
        "g_runtime_alloc_registry.clear();",
        "g_runtime_alloc_registry.insert({ptr, rec});",
        "g_runtime_alloc_registry.extract(ptr);",
        "std::swap(g_runtime_alloc_registry, other);",
        "g_runtime_alloc_registry = other;",
    ],
)
def test_mutation_gate_has_a_witness(line):
    assert registry_mutations_outside_helpers(CODE + "\nvoid f() {\n    " + line + "\n}\n"), line


@pytest.mark.parametrize(
    "line",
    [
        "auto it = g_runtime_alloc_registry.find(ptr);",
        "return g_runtime_alloc_registry.size();",
        "for (const auto & kv : g_runtime_alloc_registry) { use(kv); }",
        "return g_runtime_alloc_registry.end();",
        "if (a == g_runtime_alloc_registry.end()) return;",
    ],
)
def test_mutation_gate_allows_reads(line):
    assert registry_mutations_outside_helpers(CODE + "\nvoid f() {\n    " + line + "\n}\n") == [], line


def test_every_registry_use_outside_the_helpers_is_a_read():
    assert registry_uses_that_are_not_reads(CODE) == []


@pytest.mark.parametrize(
    "line",
    [
        "auto & reg = g_runtime_alloc_registry; reg.erase(it);",
        "auto * reg = &g_runtime_alloc_registry; reg->clear();",
        "auto && reg = g_runtime_alloc_registry;",
        "std::unordered_map<void *, runtime_alloc_record> & reg(g_runtime_alloc_registry);",
        "drop_rows(g_runtime_alloc_registry);",
        "for (auto & kv : g_runtime_alloc_registry) { kv.second.handle.size = 0; }",
        "g_runtime_alloc_registry.erase(it);",
        "g_runtime_alloc_registry.at(p) = rec;",
    ],
)
def test_use_gate_has_a_witness(line):
    assert registry_uses_that_are_not_reads(CODE + "\nvoid f() {\n    " + line + "\n}\n"), line


@pytest.mark.parametrize(
    "line",
    [
        "auto it = g_runtime_alloc_registry.find(ptr);",
        "for (const auto & kv : g_runtime_alloc_registry) { use(kv); }",
        "if (a == g_runtime_alloc_registry.end()) return;",
        "return g_runtime_alloc_registry.size() + g_runtime_alloc_registry . count(p);",
        "auto r = g_runtime_alloc_registry.equal_range(p); return g_runtime_alloc_registry.bucket_count();",
        "return g_runtime_alloc_registry.load_factor() + g_runtime_alloc_registry.max_size();",
        "auto it = g_runtime_alloc_registry.crbegin(); auto jt = g_runtime_alloc_registry.crend();",
    ],
)
def test_use_gate_allows_reads(line):
    assert registry_uses_that_are_not_reads(CODE + "\nvoid f() {\n    " + line + "\n}\n") == [], line


def test_row_aliases_are_absent_from_registry_functions():
    assert row_aliases_in_registry_functions(CODE) == []


@pytest.mark.parametrize(
    "line",
    [
        "auto & rec = it->second; rec.handle.size = 0;",
        "runtime_alloc_record & rec = it->second;",
        "auto * row = &it->second;",
        "alloc_metadata & h = it->second.handle;",
        "auto && h = kv.second.handle;",
    ],
)
def test_alias_gate_has_a_witness(line):
    planted = CODE + "\nvoid f() {\n    auto it = g_runtime_alloc_registry.find(p);\n    " + line + "\n}\n"
    assert row_aliases_in_registry_functions(planted), line


@pytest.mark.parametrize(
    "line",
    [
        "const auto & rec = it->second;",
        "const alloc_metadata & h = it->second.handle;",
        "const runtime_alloc_record * row = &it->second;",
    ],
)
def test_alias_gate_allows_const_views(line):
    planted = CODE + "\nvoid f() {\n    auto it = g_runtime_alloc_registry.find(p);\n    " + line + "\n}\n"
    assert row_aliases_in_registry_functions(planted) == [], line


def test_alias_gate_ignores_functions_that_do_not_touch_the_registry():
    assert row_aliases_in_registry_functions(CODE + "\nvoid f() {\n    auto & x = other_map.second;\n}\n") == []


def test_helper_signatures_tolerate_reformatting():
    realigned = "static std::pair<A, bool> runtime_registry_emplace_locked(void *               ptr,\n  runtime_alloc_record rec) {\n body\n}\n"
    assert function_span(realigned, HELPERS[0])


def test_registered_row_geometry_is_never_rewritten():
    assert geometry_writes(CODE) == []


@pytest.mark.parametrize(
    "line",
    [
        "it->second.handle.size = 4;",
        "it->second.handle.ptr = nullptr;",
        "found->second.handle = other;",
        "it->second.handle.size += n;",
        "it->second.handle.size -= n;",
        "it->second.handle.size |= 1;",
        "it->second.handle.size <<= 1;",
        "it->second.handle.size >>= 1;",
        "it->second.handle.ptr &= ~0xfull;",
        "it->second.handle.size *= 2;",
        "it->second.handle.size /= 2;",
        "it->second.handle.size %= 7;",
        "it->second.handle.size ^= 1;",
        "it->second.handle.size++;",
        "it->second.handle.size--;",
        "++it->second.handle.size;",
        "--it->second.handle.ptr;",
        "++g_runtime_alloc_registry.find(p)->second.handle.size;",
        "g_runtime_alloc_registry.find(p)->second.handle.size = 0;",
    ],
)
def test_geometry_gate_has_a_witness(line):
    assert geometry_writes(CODE + "\nvoid f() {\n    " + line + "\n}\n"), line


@pytest.mark.parametrize(
    "line",
    [
        "const auto & h = it->second.handle;",
        "it->second.state = X;",
        "it->second.handle.ptr == other;",
        "it->second.handle.size != 0;",
        "it->second.handle.size <= n;",
        "it->second.handle.size >= n;",
        "it->second.handle.size < n && it->second.handle.size > m;",
        "it->second.handle.size == it->second.handle.size;",
        "auto n = it->second.handle.size + 1;",
        "it->second.release_generation += 1;",
        "it->second.handle.alloc_id++;",
    ],
)
def test_geometry_gate_allows_reads_and_other_fields(line):
    assert geometry_writes(CODE + "\nvoid f() {\n    " + line + "\n}\n") == [], line


def test_no_row_is_written_whole_outside_the_allowlist():
    assert row_writes_outside_the_allowlist(CODE) == []


def test_the_assign_helper_really_writes_a_row():
    # Positive control for the exemption above: the pattern matches inside the helper, so excusing it is not vacuous.
    start, end = function_span(CODE, ASSIGN_HELPER)
    assert [m for m in ROW_WRITE_RE.finditer(CODE) if start <= m.start() < end]


def test_row_write_allowlist_is_not_stale():
    stripped = {line.strip() for line in CODE.split("\n")}
    assert [entry for entry in ROW_WRITE_ALLOWLIST if entry not in stripped] == []
    # And each entry really is a match of the pattern, so the allowlist is exercising what it excuses.
    matched = {CODE.split("\n")[CODE.count("\n", 0, m.start())].strip() for m in ROW_WRITE_RE.finditer(CODE)}
    assert set(ROW_WRITE_ALLOWLIST) <= matched


@pytest.mark.parametrize(
    "text",
    [
        "static void setrow(runtime_registry_iterator it, const runtime_alloc_record & rec) { it->second = rec; }",
        "static void setrow(runtime_registry_iterator it, runtime_alloc_record rec) {\n    it->second = std::move(rec);\n}",
        "void f() {\n    auto it = g_runtime_alloc_registry.find(p);\n    it->second = rec;\n}",
        "void f() {\n    g_runtime_alloc_registry.find(p)->second = rec;\n}",
        "void f() {\n    g_runtime_alloc_registry.begin()->second = rec;\n}",
        "void f() {\n    it->second |= flags;\n}",
        "void f() {\n    it->second++;\n}",
        "void f(std::pair<void *, runtime_alloc_record> & kv) {\n    kv.second = rec;\n}",
    ],
)
def test_row_write_gate_has_a_witness(text):
    assert row_writes_outside_the_allowlist(CODE + "\n" + text + "\n"), text


@pytest.mark.parametrize(
    "text",
    [
        "if (it->second == rec) {}",
        "bool b = it->second != rec;",
        "auto copy = it->second;",
        "use(it->second.state);",
    ],
)
def test_row_write_gate_allows_reads(text):
    assert row_writes_outside_the_allowlist(CODE + "\nvoid f() {\n    " + text + "\n}\n") == []


@pytest.mark.parametrize(
    "line",
    [
        "auto q = &it->second; q->handle.size = 0;",
        "auto q = &it->second.handle;",
        "const auto q = &it->second;",
        "decltype(auto) r = it->second;",
        "auto * q = &it->second;",
        "runtime_alloc_record * q = &it->second;",
    ],
)
def test_more_alias_forms_have_a_witness(line):
    planted = CODE + "\nvoid f() {\n    auto it = g_runtime_alloc_registry.find(p);\n    " + line + "\n}\n"
    assert row_aliases_in_registry_functions(planted), line


def test_alias_gate_follows_a_function_that_only_takes_the_iterator():
    planted = CODE + "\nstatic void g(runtime_registry_iterator it) {\n    auto & r = it->second;\n}\n"
    assert row_aliases_in_registry_functions(planted)
    planted = CODE + "\nstatic void g(std::unordered_map<void *, runtime_alloc_record>::iterator it) {\n    auto & r = it->second;\n}\n"
    assert row_aliases_in_registry_functions(planted)


@pytest.mark.parametrize(
    "line",
    [
        "auto & [k, v] = *it; v = rec; v.handle.size = 0;",
        "auto && [k, v] = *it;",
        "for (auto & [k, v] : g_runtime_alloc_registry) { v.handle.size = 0; }",
    ],
)
def test_non_const_structured_binding_has_a_witness(line):
    planted = CODE + "\nvoid f() {\n    auto it = g_runtime_alloc_registry.find(p);\n    " + line + "\n}\n"
    assert non_const_bindings_in_registry_functions(planted), line


@pytest.mark.parametrize(
    "line",
    [
        "[&](runtime_alloc_record & r) { r.handle.size = 0; }(it->second);",
        "[](auto & r) { r.handle.size = 0; }(it->second);",
        "[](auto && r) { r.handle.size = 0; }(it->second);",
        "[&](alloc_metadata & h) { h.size = 0; }(it->second.handle);",
        "fn([&](int a, runtime_alloc_record & r) {});",
    ],
)
def test_non_const_row_parameter_has_a_witness(line):
    planted = CODE + "\nvoid f() {\n    auto it = g_runtime_alloc_registry.find(p);\n    " + line + "\n}\n"
    assert non_const_bindings_in_registry_functions(planted), line


@pytest.mark.parametrize(
    "line",
    [
        "const auto & [k, v] = *it;",
        "auto [k, v] = *it;",
        "[&](const runtime_alloc_record & r) { use(r); }(it->second);",
        "[&](const alloc_metadata & h) { use(h); }(it->second.handle);",
        "[](int a, int b) { return a + b; }(1, 2);",
        "for (const auto & kv : g_runtime_alloc_registry) { use(kv); }",
    ],
)
def test_non_const_gate_allows_const_and_unrelated_forms(line):
    planted = CODE + "\nvoid f() {\n    auto it = g_runtime_alloc_registry.find(p);\n    " + line + "\n}\n"
    assert non_const_bindings_in_registry_functions(planted) == [], line


def test_the_rules_ignore_functions_that_do_not_touch_the_registry():
    other = "\nvoid f(std::map<int, int> & m) {\n    for (auto & [k, v] : m) { v = 1; }\n    auto & x = m.begin()->second;\n}\n"
    assert non_const_bindings_in_registry_functions(CODE + other) == []
    assert row_aliases_in_registry_functions(CODE + other) == []


def test_non_const_bindings_are_absent_from_registry_functions():
    assert non_const_bindings_in_registry_functions(CODE) == []


def test_registry_and_index_stay_file_static():
    offenders = []
    for path in sorted(SYCL_DIR.rglob("*")):
        if path.suffix not in (".cpp", ".hpp", ".h", ".cu", ".cuh") or path.name == "unified-cache.cpp":
            continue
        text = strip_comments(path.read_text(errors="replace"))
        if "g_runtime_alloc_registry" in text or "g_runtime_alloc_index" in text:
            offenders.append(path.name)
    assert offenders == []


def lookup_violations(body: str):
    out = []
    if "g_runtime_alloc_index.find_innermost(" not in body:
        out.append("does not consult the index")
    if re.search(r"\b(?:for|while)\s*\([^)]*g_runtime_alloc_registry", body) or "g_runtime_alloc_registry.begin()" in body:
        out.append("iterates the registry")
    return out


def test_lookup_consults_the_index_and_does_not_scan():
    assert lookup_violations(lookup_body(CODE)) == []


@pytest.mark.parametrize(
    "body",
    [
        "for (const auto & kv : g_runtime_alloc_registry) { use(kv); }",
        "g_runtime_alloc_index.find_innermost(a, &h); for (auto it = g_runtime_alloc_registry.begin(); it != e; ++it) {}",
        "g_runtime_alloc_index.find_innermost(a, &h); while (x != g_runtime_alloc_registry.end()) {}",
        "auto it = g_runtime_alloc_registry.begin();",
        "for (int i = 0; i < n; i++) {}",
    ],
)
def test_lookup_gate_has_a_witness(body):
    assert lookup_violations(body), body


def test_lookup_gate_allows_an_unrelated_loop_beside_the_index_query():
    body = "g_runtime_alloc_index.find_innermost(a, &h); for (int i = 0; i < 3; i++) { pick(i); }"
    assert lookup_violations(body) == []


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
