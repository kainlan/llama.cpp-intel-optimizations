"""Source contract for llama.cpp-ii25: the runtime allocation registry's containment index cannot drift from the registry.

unified_lookup_runtime_allocation() answers "which registered allocation contains this pointer" through
g_runtime_alloc_index (range-index.hpp) instead of scanning every row of g_runtime_alloc_registry (61.65% of Qwen3.8
decode CPU with a 512-expert MoE). The index is correct only while every insert and erase of a registry row also updates
it, and while a row's extent never changes after it is registered.

TWO LAYERS, and only one of them is a proof:
  - RUNTIME BACKSTOP (the guarantee): the lookup GGML_ASSERTs that the row it found still has the base and end the
    index holds for it (runtime_registry_row_matches_index_entry). A row rewritten behind the index's back therefore
    aborts the process at the first lookup that reaches it instead of answering from a stale extent;
    test-runtime-registry-containment drives exactly that, through a PRIVATE_TESTING seam, in a forked child. The assert
    sees only what a lookup reaches, so it is a backstop, not a substitute for the rules below.
  - THIS FILE (a tripwire): comment-stripped regexes that keep the obvious ways of writing a row, and the index, out of
    the code. It reads text, not the AST, so it is an honest tripwire and not a proof: see NOT COVERED.

What this file enforces:

(A) g_runtime_alloc_registry is mutated (emplace/erase/insert/clear/extract/swap/operator[]/assignment) ONLY inside the
    runtime_registry_emplace_locked / runtime_registry_erase_locked helpers, and every other mention of the registry is
    a read member call, a const range-for or the declaration (so no alias, pointer, or reference argument, and no at()).
(B) A registered row is never WRITTEN THROUGH a plain access path outside those helpers:
    - an assignment, compound assignment (+=, |=, <<= ...) or ++/-- of `second.handle`, `second.handle.ptr` or
      `second.handle.size`, anywhere in the file, with any parenthesisation (`(it->second).handle.size = 0;`), except
      the entries of GEOMETRY_WRITE_ALLOWLIST (the PRIVATE_TESTING seam that proves the backstop, and nothing else);
    - the address of `second.handle`, `second.handle.ptr` or `second.handle.size` taken anywhere in the file, whatever
      the pointer's declared type (`size_t * s = &it->second.handle.size;`, `*(&it->second.handle.size) = 0;`);
    - a whole-row assignment (`it->second = rec;`, `registry.find(p)->second = rec;`) anywhere in the file, except the
      entries of ROW_WRITE_ALLOWLIST: three non-registry sites and the one row move in runtime_registry_assign_locked,
      which resizes or re-keys the index before it moves a row over the old. Entries are keyed by (enclosing function,
      line text) with an exact occurrence count, so a second assignment in an allowed function, a copy of an allowed
      line in another function, and an entry that stops matching all fail;
    - in a function that touches the registry (names it, or takes a runtime_registry_iterator or the map's own iterator
      type): a non-const reference or pointer bound to a whole row or to `second.handle[.ptr|.size]`
      (`auto & r = it->second;`, `auto q = &it->second;`, `auto * q = &it->second.handle;`, `decltype(auto) r = ...;`),
      a non-const structured binding bound to the registry, a registry iterator or a row (`auto & [k, v] = *it;`), and a
      non-const reference parameter of a lambda or function (`[&](runtime_alloc_record & r)`, `[](auto & r)`) when that
      function passes a row (`it->second`, `*it`, `.second`) as an argument. An out-parameter that is only written FROM a
      row is not flagged; a body that also hands a row on is flagged conservatively, even if the callee only reads it.
(C) Neither registry nor index is touched from another translation unit (both are file-static in unified-cache.cpp).
(D) The lookup consults the index, does not iterate the registry, and carries the backstop assertion.
(E) g_runtime_alloc_index is MUTATED (insert/erase/resize/clear/assignment, an alias, a pointer) only inside
    runtime_registry_emplace_locked, runtime_registry_erase_locked and runtime_registry_assign_locked. Everywhere else
    it is a read member call (find_innermost/find_exact/find_first_base_in/size/check_invariants) or its declaration, which covers the
    PRIVATE_TESTING consistency audit.

NOT COVERED, stated so nobody mistakes this for a proof. The gate reads text, not the AST:
    - writes by CALL, even directly on a row: `std::swap(it->second, x)`, `std::exchange(it->second.handle.size, 0)`,
      `memcpy(&it->second, ...)`, `std::addressof(it->second.handle.size)`, `std::tie(a, b) = ...`;
    - a macro that wraps a write (`SET_FIELD(it->second.handle.size, 0)`), since macros are not expanded;
    - the address of a WHOLE row passed straight to a function that never names the registry (`zap(&it->second)`), and a
      non-const row-reference parameter on a function that does not itself touch the registry (`void zap(
      runtime_alloc_record & r)`, called as `zap(it->second)`);
    - `auto it = ...` and template-iterator parameters (`template <class It> void f(It it)`): the function reaches rows
      through a type the gate cannot see, so it is not recognised as touching the registry unless it also names the
      registry or one of the two iterator types; copies of an iterator (`auto jt = it;`) are likewise not followed;
    - `const_cast` of a const view of a row;
    - a helper that RETURNS `runtime_alloc_record &` / `alloc_metadata &` (the call site never names the registry);
    - `#if 0` blocks and raw string literals, which the comment stripper does not understand;
    - functions are split at a closing brace in column 0, which is how this file is formatted, and the enclosing function
      of an allowlisted line is the last column-0 line that contains a `(`.
Those are bounded by review and by the runtime backstop; (A) and the alias/parameter rules above keep any such write on
code that already looks wrong.

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
# The one helper that replaces a row in place: it resizes or re-keys the index first, then moves the new row over the old.
ASSIGN_HELPER = (
    r"runtime_registry_assign_locked\(\s*void\s*\*\s*ptr\s*,\s*const\s+runtime_alloc_record\s*&\s*rec\s*\)\s*\{"
)
# Every function allowed to touch the index: the three that keep it in step with the registry.
INDEX_HELPERS = HELPERS + (ASSIGN_HELPER,)
# Members that only read the registry. Any other use of the registry outside the helpers (an alias, a pointer, a
# reference handed to a function, a non-const range-for) could mutate it behind the index's back.
# `at` is deliberately absent: it returns a mutable reference and a read can always be spelled find.
READ_MEMBER_RE = re.compile(
    r"\s*\.\s*(?:find|end|begin|cbegin|cend|crbegin|crend|size|max_size|empty|count|contains|equal_range|bucket_count"
    r"|load_factor|max_load_factor)\b"
)
# The index's const members. Every other mention of it outside the three helpers is a mutation or an alias.
INDEX_READ_MEMBER_RE = re.compile(r"\s*\.\s*(?:find_innermost|find_exact|find_first_base_in|size|check_invariants)\b")
CONST_RANGE_FOR_RE = re.compile(r"const\s+auto\s*&\s*\w+\s*:\s*$")
# What follows `second` in an alias target. `second` itself, `second.handle`, `second.handle.ptr` and `second.handle.size`
# are the row and its identity/geometry; `second.owned_segments` or `second.handle.alloc_id` is another field's reference.
_ROW_OR_GEOMETRY = r"\bsecond\b(?!\s*\.\s*(?!handle\b)\w)(?!\s*\.\s*handle\s*\.\s*(?!ptr\b|size\b)\w)"
# A non-const reference or pointer bound to a registry row, through which a row's geometry could be rewritten. Only looked
# for in functions that touch the registry.
ROW_ALIAS_RE = re.compile(
    # auto & r = it->second;  auto * q = &it->second.handle;  alloc_metadata & h = it->second.handle;
    r"(?<!const )(?<!const\t)\b(?:auto|alloc_metadata|runtime_alloc_record)\s*(?:&&?|\*)\s*\w+\s*(?:=|\{|\()[^;{}]*"
    + _ROW_OR_GEOMETRY
    # auto q = &it->second;   (a pointer deduced from an address-of; `const auto q` is still a pointer to a mutable row)
    + r"|\bauto\s+\w+\s*=\s*&[^;{}]*"
    + _ROW_OR_GEOMETRY
    # decltype(auto) r = it->second;
    + r"|\bdecltype\s*\(\s*auto\s*\)\s*\w+\s*=[^;{}]*"
    + _ROW_OR_GEOMETRY
)
# A non-const structured binding: `auto & [k, v] = *it;`, `auto && [k, v] = ...;`, `for (auto & [k, v] : map)`. Whether it
# is bound to the registry is decided from its initialiser (see non_const_bindings_in_registry_functions).
STRUCTURED_BINDING_RE = re.compile(r"(?<!const )(?<!const\t)\bauto\s*&{1,2}\s*\[[^\]]*\]\s*(?:=|:|\{|\()")
# A function or lambda parameter that is a non-const reference to a row or its metadata, or a generic one.
ROW_REF_PARAM_RE = re.compile(
    r"[(,]\s*(?:runtime_alloc_record|alloc_metadata)\s*&&?\s*\w*\s*(?=[,)])|[(,]\s*auto\s*&&?\s*\w+\s*(?=[,)])"
)
# Functions that deal in registry rows even when they never name the registry (an iterator parameter, say). Regexes, so
# `unordered_map<void*,runtime_alloc_record>` is recognised however it is spaced.
REGISTRY_FUNCTION_MARKER_RES = (
    re.compile(r"\bg_runtime_alloc_registry\b"),
    re.compile(r"\bruntime_registry_iterator\b"),
    re.compile(r"\bunordered_map\s*<\s*void\s*\*\s*,\s*runtime_alloc_record\s*>"),
)
# Names that hold a registry iterator inside a function: `auto it = g_runtime_alloc_registry.find(p);`, a
# runtime_registry_iterator, or the map's own iterator type.
ITERATOR_NAME_RES = (
    re.compile(
        r"\b(\w+)\s*=\s*g_runtime_alloc_registry\s*\.\s*(?:find|begin|end|cbegin|cend|lower_bound|upper_bound)\b"
    ),
    re.compile(r"\bruntime_registry_iterator\s*&?\s*(\w+)\b"),
    re.compile(r"\bunordered_map\s*<[^;>]*>\s*::\s*(?:const_)?iterator\s*&?\s*(\w+)\b"),
)
_ASSIGN = r"(?:(?:[-+*/%|&^]|<<|>>)?=(?!=)|\+\+|--)"  # =, op=, ++, -- ; not ==, <=, >=, !=
# A row's identity or geometry written through an iterator/reference into the registry: `second.handle = x`,
# `second.handle.size += n`, `second.handle.ptr++`, `++second.handle.size`, `(it->second).handle.size = 0` (closing
# parentheses may sit between the parts).
GEOMETRY_WRITE_RE = re.compile(
    r"(?:->|\.)\s*second\s*\)*\s*\.\s*handle\s*\)*\s*(?:\.\s*(?:ptr|size)\s*\)*\s*)?" + _ASSIGN
    + r"|(?:\+\+|--)\s*[\w.>()\-]*?(?:->|\.)\s*second\s*\)*\s*\.\s*handle\s*\)*\s*\.\s*(?:ptr|size)\b"
)
# The address of a row's identity or geometry, whatever the pointer is declared as: `size_t * s = &it->second.handle.size;`,
# `void ** s = &it->second.handle.ptr;`, `*(&it->second.handle.size) = 0;`. `&it->second.handle.alloc_id` (another field)
# and `&it->second.handle.key()` (a call's result) do not match. Whether the `&` is unary is decided by the caller.
ADDRESS_OF_GEOMETRY_RE = re.compile(
    r"&\s*[\w.>()\-\[\]*\s]*?(?:->|\.)\s*second\s*\)*\s*\.\s*handle\s*\)*(?:\s*\.\s*(?:ptr|size)\b)?(?!\s*\.\s*\w)"
)
# A whole row written through an iterator/reference: `it->second = rec;`, `registry.find(p)->second = rec;`. Scanned over
# the whole file (a helper that takes the iterator never names the registry), minus ROW_WRITE_ALLOWLIST.
ROW_WRITE_RE = re.compile(r"(?:->|\.)second\s*" + _ASSIGN)
# The `second` writes in the file that are rows of the registry or are NOT registry rows, each checked by hand. Keyed by
# (enclosing function, stripped source line) with the exact number of times that line appears in that function: a second
# copy in the same function, the same line in another function, and an entry that no longer matches all fail.
ROW_WRITE_ALLOWLIST = {
    ("lifecycle_replace_placement_plan", "current->second = replacement;"): (1, "the MoE mmid plan registry's entry"),
    ("offload_stats_note_host_alloc", "entry.second += bytes;"): (1, "offload host-alloc stats, keyed by tag"),
    ("remap_or_erase_id_mapping_locked", "mapped->second = replacement;"): (1, "the id_to_key_ map"),
    ("runtime_registry_assign_locked", "it->second = std::move(fresh);"): (
        1,
        "the registry's own row move, after the index was resized or re-keyed to the new geometry",
    ),
}
# Geometry writes that are allowed, same keying, and each must sit inside `#if defined(GGML_SYCL_PRIVATE_TESTING)`.
GEOMETRY_WRITE_ALLOWLIST = {
    ("allocation_registry_test_corrupt_row_size", "it->second.handle.size = bytes;"): (
        1,
        "the seam that rewrites a row behind the index so the lookup backstop can be proven to fire",
    ),
}


def function_span(code: str, signature: str):
    m = re.search(signature, code)
    assert m is not None, f"{signature!r} is not defined"
    start = m.start()
    end = code.find("\n}\n", start)
    assert end != -1, f"could not bound {signature!r}"
    return start, end


def line_of(code: str, pos: int) -> int:
    return code.count("\n", 0, pos) + 1


def enclosing_function(code: str, pos: int):
    """The name on the last column-0 line before `pos` that holds a `(`; this file's functions start in column 0."""
    name = None
    for m in re.finditer(r"^[A-Za-z_][^\n]*$", code[:pos], flags=re.M):
        line = m.group(0)
        call = re.search(r"(\w+)\s*\(", line)
        if call and not line.startswith("static_assert"):
            name = call.group(1)
    return name


def preprocessor_guards(code: str, pos: int):
    """The conditions of the #if blocks that enclose `pos`, outermost first; an #else/#elif branch reads as '<else>'."""
    stack = []
    for m in re.finditer(r"^[ \t]*#[ \t]*(if|ifdef|ifndef|elif|else|endif)\b([^\n]*)", code[:pos], flags=re.M):
        kind, rest = m.group(1), m.group(2).strip()
        if kind in ("if", "ifdef", "ifndef"):
            stack.append(rest if kind == "if" else kind + " " + rest)
        elif kind in ("elif", "else"):
            if stack:
                stack[-1] = "<else>"
        elif stack:
            stack.pop()
    return stack


def allowlist_violations(code: str, pattern, allow, private_testing_only=False):
    """Matches of `pattern` that `allow` does not excuse, and allow entries whose occurrence count is not exact.

    `allow` maps (enclosing function, stripped line) -> (count, why). Returned: ('unlisted', line, function, text) for
    each match with no entry, ('count', function, text, expected, found) for each entry that does not match exactly, and,
    with `private_testing_only`, ('unguarded', line, function, text) for an allowed match outside a PRIVATE_TESTING block.
    """
    lines = code.split("\n")
    found = {}
    out = []
    for m in pattern.finditer(code):
        line_no = line_of(code, m.start())
        key = (enclosing_function(code, m.start()), lines[line_no - 1].strip())
        found[key] = found.get(key, 0) + 1
        if key not in allow:
            out.append(("unlisted", line_no, key[0], key[1][:70]))
        elif private_testing_only and "defined(GGML_SYCL_PRIVATE_TESTING)" not in preprocessor_guards(code, m.start()):
            out.append(("unguarded", line_no, key[0], key[1][:70]))
    for key, (count, _why) in allow.items():
        if found.get(key, 0) != count:
            out.append(("count", key[0], key[1][:70], count, found.get(key, 0)))
    return out


def registry_mutations_outside_helpers(code: str):
    spans = [function_span(code, h) for h in HELPERS]
    bad = []
    for m in MUTATION_RE.finditer(code):
        if not any(s <= m.start() < e for s, e in spans):
            bad.append((line_of(code, m.start()), code[m.start() : m.start() + 60].replace("\n", " ")))
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
        bad.append((line_of(code, m.start()), (pre[-30:] + "|" + tail[:30]).replace("\n", " ")))
    return bad


def index_uses_that_are_not_reads(code: str):
    """Every mention of the index outside its three helpers must be a const member call or its declaration."""
    spans = [function_span(code, h) for h in INDEX_HELPERS]
    bad = []
    for m in re.finditer(r"\bg_runtime_alloc_index\b", code):
        if any(s <= m.start() < e for s, e in spans):
            continue
        tail = code[m.end() : m.end() + 80]
        pre = code[max(0, m.start() - 80) : m.start()]
        if INDEX_READ_MEMBER_RE.match(tail):
            continue
        if re.search(r"\baddress_range_index\s*$", pre) and tail.lstrip().startswith(";"):
            continue  # the declaration
        bad.append((line_of(code, m.start()), (pre[-30:] + "|" + tail[:30]).replace("\n", " ")))
    return bad


def touches_registry(body: str) -> bool:
    return any(marker.search(body) for marker in REGISTRY_FUNCTION_MARKER_RES)


def registry_functions(code: str):
    """(offset, body) of each column-0-delimited chunk that touches the registry."""
    pos = 0
    for end in re.finditer(r"\n}\n", code):
        body = code[pos : end.end()]
        if touches_registry(body):
            yield pos, body
        pos = end.end()


def iterator_names(body: str):
    return {m.group(1) for rx in ITERATOR_NAME_RES for m in rx.finditer(body)}


def registry_function_scan(code: str, pattern):
    """Matches of `pattern` inside functions that touch the registry; (line, text)."""
    bad = []
    for pos, body in registry_functions(code):
        for m in pattern.finditer(body):
            bad.append((line_of(code, pos + m.start()), m.group(0)[:70]))
    return bad


def row_aliases_in_registry_functions(code: str):
    """A non-const reference/pointer to a registry row or its geometry, in a function that touches the registry."""
    return registry_function_scan(code, ROW_ALIAS_RE)


def initialiser_text(body: str, start: int) -> str:
    """The text of a binding's initialiser or range expression: up to a `;`, a `{` or an unbalanced `)`."""
    depth = 0
    for i in range(start, len(body)):
        c = body[i]
        if c == "(":
            depth += 1
        elif c == ")":
            if depth == 0:
                return body[start:i]
            depth -= 1
        elif c in ";{" and depth == 0:
            return body[start:i]
    return body[start:]


def binds_a_registry_row(init: str, names) -> bool:
    if re.search(r"\bg_runtime_alloc_registry\b|\bsecond\b", init):
        return True
    return any(re.search(r"\*\s*\(?\s*" + re.escape(n) + r"\b|\b" + re.escape(n) + r"\s*->", init) for n in names)


def row_argument_re(names):
    star = r"|\*\s*\(?\s*(?:" + "|".join(re.escape(n) for n in sorted(names)) + r")\b\)?" if names else ""
    return re.compile(r"[(,]\s*&?\s*(?:[\w.>()\-\[\]*]*?(?:->|\.)second(?:\s*\.\s*handle)?" + star + r")\s*[,)]")


def non_const_bindings_in_registry_functions(code: str):
    """Non-const structured bindings bound to a registry iterator or row, and non-const row/generic reference parameters
    of a function that also passes a row on, in functions that touch the registry."""
    bad = []
    for pos, body in registry_functions(code):
        names = iterator_names(body)
        for m in STRUCTURED_BINDING_RE.finditer(body):
            if binds_a_registry_row(initialiser_text(body, m.end()), names):
                bad.append((line_of(code, pos + m.start()), m.group(0)[:70]))
        if row_argument_re(names).search(body):
            for m in ROW_REF_PARAM_RE.finditer(body):
                bad.append((line_of(code, pos + m.start()), m.group(0)[:70]))
    return bad


def row_write_violations(code: str):
    return allowlist_violations(code, ROW_WRITE_RE, ROW_WRITE_ALLOWLIST)


def geometry_write_violations(code: str):
    return allowlist_violations(code, GEOMETRY_WRITE_RE, GEOMETRY_WRITE_ALLOWLIST, private_testing_only=True)


def geometry_addresses_taken(code: str):
    """The unary `&` of a row's `second.handle[.ptr|.size]`, anywhere; (line, text). A binary `&` or a `&&` is not one."""
    bad = []
    for m in ADDRESS_OF_GEOMETRY_RE.finditer(code):
        before = code[: m.start()].rstrip()
        if before.endswith("&"):
            continue  # `&&`
        if before and (before[-1].isalnum() or before[-1] in "_)]") and not re.search(r"\breturn$", before):
            continue  # a binary &
        bad.append((line_of(code, m.start()), m.group(0)[:70].replace("\n", " ")))
    return bad


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


# (E) the index ---------------------------------------------------------------------------------------------------------


def test_the_index_is_touched_only_by_the_three_helpers_and_const_reads():
    assert index_uses_that_are_not_reads(CODE) == []


def test_the_three_helpers_really_mutate_the_index():
    # Positive control: each helper's own insert/erase/resize is found inside its span, so the zero above is not vacuous.
    mutate = re.compile(r"\bg_runtime_alloc_index\s*\.\s*(?:insert|erase|resize|clear)\b")
    for helper in INDEX_HELPERS:
        s, e = function_span(CODE, helper)
        assert mutate.search(CODE, s, e), helper


@pytest.mark.parametrize(
    "line",
    [
        "g_runtime_alloc_index.insert(a, n, k);",
        "g_runtime_alloc_index.erase(a, k);",
        "g_runtime_alloc_index.resize(a, k, n);",
        "g_runtime_alloc_index.clear();",
        "g_runtime_alloc_index = address_range_index();",
        "auto & ix = g_runtime_alloc_index;",
        "auto * ix = &g_runtime_alloc_index;",
        "drop(g_runtime_alloc_index);",
        "g_runtime_alloc_index . insert (a, n, k);",
    ],
)
def test_index_gate_has_a_witness(line):
    assert index_uses_that_are_not_reads(CODE + "\nvoid f() {\n    " + line + "\n}\n"), line


@pytest.mark.parametrize(
    "line",
    [
        "address_range_index::entry e; return g_runtime_alloc_index.find_innermost(a, &e);",
        "return g_runtime_alloc_index.find_exact(a, &e) && g_runtime_alloc_index.size() == 0;",
        "return g_runtime_alloc_index.check_invariants();",
    ],
)
def test_index_gate_allows_const_reads(line):
    assert index_uses_that_are_not_reads(CODE + "\nvoid f() {\n    " + line + "\n}\n") == [], line


def test_the_index_gate_follows_a_reformatted_helper_signature():
    realigned = "static void runtime_registry_assign_locked(void *                        ptr,\n  const runtime_alloc_record & rec) {\n body\n}\n"
    assert function_span(realigned, ASSIGN_HELPER)


# (B) rows -------------------------------------------------------------------------------------------------------------


def test_row_aliases_are_absent_from_registry_functions():
    assert row_aliases_in_registry_functions(CODE) == []


def planted_in_registry_function(line: str) -> str:
    return CODE + "\nvoid f() {\n    auto it = g_runtime_alloc_registry.find(p);\n    " + line + "\n}\n"


@pytest.mark.parametrize(
    "line",
    [
        "auto & rec = it->second; rec.handle.size = 0;",
        "runtime_alloc_record & rec = it->second;",
        "auto * row = &it->second;",
        "alloc_metadata & h = it->second.handle;",
        "auto && h = kv.second.handle;",
        "auto & p = it->second.handle.ptr;",
        "auto & n = it->second.handle.size;",
    ],
)
def test_alias_gate_has_a_witness(line):
    assert row_aliases_in_registry_functions(planted_in_registry_function(line)), line


@pytest.mark.parametrize(
    "line",
    [
        "const auto & rec = it->second;",
        "const alloc_metadata & h = it->second.handle;",
        "const runtime_alloc_record * row = &it->second;",
        # a reference to another FIELD of a row cannot move the row's extent
        "auto & segs = it->second.owned_segments;",
        "auto & cohort = it->second.cohort_id;",
        "auto & id = it->second.handle.alloc_id;",
        "auto * q = &it->second.queue;",
    ],
)
def test_alias_gate_allows_const_views_and_other_fields(line):
    assert row_aliases_in_registry_functions(planted_in_registry_function(line)) == [], line


def test_alias_gate_ignores_functions_that_do_not_touch_the_registry():
    assert row_aliases_in_registry_functions(CODE + "\nvoid f() {\n    auto & x = other_map.second;\n}\n") == []


def test_helper_signatures_tolerate_reformatting():
    realigned = "static std::pair<A, bool> runtime_registry_emplace_locked(void *               ptr,\n  runtime_alloc_record rec) {\n body\n}\n"
    assert function_span(realigned, HELPERS[0])


def test_registered_row_geometry_is_never_rewritten_outside_the_allowlist():
    assert geometry_write_violations(CODE) == []


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
        # parenthesised access paths
        "(it->second).handle.size = 0;",
        "((*it).second).handle.size = 0;",
        "(it->second.handle).size = 0;",
        "(it->second).handle.ptr = nullptr;",
        "(it->second).handle = other;",
        "((it->second).handle).size += 1;",
        "++(it->second).handle.size;",
        "it->second . handle . size = 0;",
        "it->second.handle.size\n        = 0;",
    ],
)
def test_geometry_gate_has_a_witness(line):
    assert geometry_write_violations(CODE + "\nvoid f() {\n    " + line + "\n}\n"), line


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
        "(it->second).handle.size == 0;",
        "((*it).second).handle.ptr != nullptr;",
        "(it->second).state = X;",
    ],
)
def test_geometry_gate_allows_reads_and_other_fields(line):
    assert geometry_write_violations(CODE + "\nvoid f() {\n    " + line + "\n}\n") == [], line


@pytest.mark.parametrize(
    "line",
    [
        "size_t * s = &it->second.handle.size; *s = 0;",
        "void ** s = &it->second.handle.ptr; *s = nullptr;",
        "*(&it->second.handle.size) = 0;",
        "auto * s = &it->second.handle.size;",
        "const auto s = &it->second.handle.ptr;",
        "uintptr_t * s = reinterpret_cast<uintptr_t *>(&it->second.handle.ptr);",
        "alloc_metadata * h = &it->second.handle;",
        "auto * h = &(it->second).handle;",
        "size_t * s = &(it->second.handle.size);",
        "size_t * s = &g_runtime_alloc_registry.find(p)->second.handle.size;",
        "size_t * s = &((*it).second).handle.size;",
        "return &it->second.handle.size;",
        "zap(&it->second.handle.size);",
        "zap(1, &it->second.handle);",
    ],
)
def test_address_gate_has_a_witness(line):
    assert geometry_addresses_taken(CODE + "\nvoid f() {\n    " + line + "\n}\n"), line


@pytest.mark.parametrize(
    "line",
    [
        "return a & it->second.handle.size;",
        "return flags & it->second.handle.size && ok;",
        "bool b = x && it->second.handle.size;",
        "bool b = (x) & it->second.handle.size;",
        "auto m = mask[1] & it->second.handle.ptr;",
        "auto * k = &it->second.handle.alloc_id;",
        "auto k = &it->second.handle.key();",
        "const auto * q = &it->second.queue;",
        "auto && h = it->second.handle;",
    ],
)
def test_address_gate_allows_binary_and_and_other_fields(line):
    assert geometry_addresses_taken(CODE + "\nvoid f() {\n    " + line + "\n}\n") == [], line


def test_no_address_of_row_geometry_is_taken():
    assert geometry_addresses_taken(CODE) == []


# the allowlists --------------------------------------------------------------------------------------------------------


def test_no_row_is_written_whole_outside_the_allowlist():
    # Also fails if an allowlist entry has stopped matching, or matches a different number of times than recorded.
    assert row_write_violations(CODE) == []


def test_the_allowlist_names_the_enclosing_function_of_each_entry():
    assert enclosing_function(CODE, CODE.index("it->second = std::move(fresh);")) == "runtime_registry_assign_locked"
    assert enclosing_function(CODE, CODE.index("entry.second += bytes;")) == "offload_stats_note_host_alloc"


def mutate(code: str, old: str, new: str) -> str:
    assert code.count(old) == 1, f"{old!r} must appear exactly once to be mutated"
    return code.replace(old, new)


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
        # an allowed line copied into a function that is not the allowed one
        "void f() {\n    current->second = replacement;\n}",
        "void f() {\n    it->second = std::move(fresh);\n}",
    ],
)
def test_row_write_gate_has_a_witness(text):
    assert row_write_violations(CODE + "\n" + text + "\n"), text


@pytest.mark.parametrize(
    "mutation",
    [
        # a second whole-row write in the assign helper, different text and identical text
        ("it->second = std::move(fresh);", "it->second = std::move(fresh);\n    it->second = rec;"),
        ("it->second = std::move(fresh);", "it->second = std::move(fresh);\n    it->second = std::move(fresh);"),
        ("it->second = std::move(fresh);", "it->second = rec;\n    it->second = std::move(fresh);"),
        # a second copy of an allowed line in its own function
        ("entry.second += bytes;", "entry.second += bytes;\n    entry.second += bytes;"),
        # an allowed line that stopped being a write: the entry is stale
        ("entry.second += bytes;", "entry.second2 += bytes;"),
        ("it->second = std::move(fresh);", "it->second.state = fresh.state;"),
    ],
)
def test_allowlist_is_keyed_by_function_line_and_exact_count(mutation):
    assert row_write_violations(mutate(CODE, *mutation)), mutation


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
    assert row_write_violations(CODE + "\nvoid f() {\n    " + text + "\n}\n") == []


def test_the_geometry_allowlist_is_exact_and_confined_to_private_testing():
    assert geometry_write_violations(CODE) == []
    seam = "it->second.handle.size = bytes;"
    # a second copy in the seam, the same line in another function, a stale entry: all fail
    assert geometry_write_violations(mutate(CODE, seam, seam + "\n    " + seam))
    assert geometry_write_violations(CODE + "\nvoid f() {\n    " + seam + "\n}\n")
    assert geometry_write_violations(mutate(CODE, seam, "it->second.handle.size == bytes;"))
    # outside `#if defined(GGML_SYCL_PRIVATE_TESTING)` the same allowed line is refused
    allow = {("g", "x.second.handle.size = 1;"): (1, "t")}
    body = "void g() {\n    x.second.handle.size = 1;\n}\n"
    assert allowlist_violations(body, GEOMETRY_WRITE_RE, allow, private_testing_only=True) == [
        ("unguarded", 2, "g", "x.second.handle.size = 1;")
    ]
    guarded = "#if defined(GGML_SYCL_PRIVATE_TESTING)\n" + body + "#endif\n"
    assert allowlist_violations(guarded, GEOMETRY_WRITE_RE, allow, private_testing_only=True) == []
    after_endif = guarded + body
    assert allowlist_violations(after_endif, GEOMETRY_WRITE_RE, allow, private_testing_only=True)
    in_else = "#if defined(GGML_SYCL_PRIVATE_TESTING)\n#else\n" + body + "#endif\n"
    assert allowlist_violations(in_else, GEOMETRY_WRITE_RE, allow, private_testing_only=True)
    nested = "#if defined(GGML_SYCL_PRIVATE_TESTING)\n#if X\n#endif\n" + body + "#endif\n"
    assert allowlist_violations(nested, GEOMETRY_WRITE_RE, allow, private_testing_only=True) == []


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
    assert row_aliases_in_registry_functions(planted_in_registry_function(line)), line


def test_alias_gate_follows_a_function_that_only_takes_the_iterator():
    planted = CODE + "\nstatic void g(runtime_registry_iterator it) {\n    auto & r = it->second;\n}\n"
    assert row_aliases_in_registry_functions(planted)
    planted = CODE + "\nstatic void g(std::unordered_map<void *, runtime_alloc_record>::iterator it) {\n    auto & r = it->second;\n}\n"
    assert row_aliases_in_registry_functions(planted)


@pytest.mark.parametrize(
    "decl",
    [
        "std::unordered_map<void *, runtime_alloc_record>::iterator it",
        "std::unordered_map<void*,runtime_alloc_record>::iterator it",
        "std::unordered_map< void * , runtime_alloc_record >::const_iterator it",
    ],
)
def test_the_map_type_marker_tolerates_spacing(decl):
    planted = CODE + "\nstatic void g(" + decl + ") {\n    auto & r = it->second;\n}\n"
    assert row_aliases_in_registry_functions(planted), decl


@pytest.mark.parametrize(
    "line",
    [
        "auto & [k, v] = *it; v = rec; v.handle.size = 0;",
        "auto && [k, v] = *it;",
        "for (auto & [k, v] : g_runtime_alloc_registry) { v.handle.size = 0; }",
        "auto & [k, v] = *g_runtime_alloc_registry.find(p);",
        "auto & [a, b] = it->second;",
        "auto & [k, v] = (*it);",
    ],
)
def test_non_const_structured_binding_has_a_witness(line):
    assert non_const_bindings_in_registry_functions(planted_in_registry_function(line)), line


@pytest.mark.parametrize(
    "line",
    [
        "[&](runtime_alloc_record & r) { r.handle.size = 0; }(it->second);",
        "[](auto & r) { r.handle.size = 0; }(it->second);",
        "[](auto && r) { r.handle.size = 0; }(it->second);",
        "[&](alloc_metadata & h) { h.size = 0; }(it->second.handle);",
        "auto f = [&](int a, runtime_alloc_record & r) {}; f(1, it->second);",
        "auto f = [](auto & r) {}; f(*it);",
        "auto f = [](auto & r) {}; f(g_runtime_alloc_registry.find(p)->second);",
        "auto f = [&](alloc_metadata & h) {}; f(&it->second.handle);",
    ],
)
def test_non_const_row_parameter_has_a_witness(line):
    assert non_const_bindings_in_registry_functions(planted_in_registry_function(line)), line


@pytest.mark.parametrize(
    "line",
    [
        "const auto & [k, v] = *it;",
        "auto [k, v] = *it;",
        "[&](const runtime_alloc_record & r) { use(r); }(it->second);",
        "[&](const alloc_metadata & h) { use(h); }(it->second.handle);",
        "[](int a, int b) { return a + b; }(1, 2);",
        "for (const auto & kv : g_runtime_alloc_registry) { use(kv); }",
        # a binding of something that is not the registry, inside a function that touches the registry
        "for (auto & [id, cache] : g_device_caches) { use(id, cache); }",
        "auto & [a, b] = other_pair;",
        "for (auto & [k, v] : some_other_map) { v = 0; }",
        # an out-parameter written FROM a row never hands the row on
        "auto fill = [](alloc_metadata & out, int n) { out = {}; out.size = n; }; alloc_metadata m; fill(m, 3);",
        "auto grab = [&](alloc_metadata & out) { out = it->second.handle; }; alloc_metadata m; grab(m);",
        "auto g = [](auto & x) { x = 0; }; int n; g(n);",
    ],
)
def test_non_const_gate_allows_const_and_unrelated_forms(line):
    assert non_const_bindings_in_registry_functions(planted_in_registry_function(line)) == [], line


def test_an_out_parameter_beside_a_row_argument_is_flagged_conservatively():
    # Documented limit: once the function passes a row on as an argument, every non-const row reference parameter in it
    # is suspect, even one that is only an out-parameter.
    line = "auto grab = [&](alloc_metadata & out) { out = it->second.handle; }; alloc_metadata m; grab(m); use(it->second);"
    assert non_const_bindings_in_registry_functions(planted_in_registry_function(line))


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


# (D) the lookup --------------------------------------------------------------------------------------------------------


def lookup_violations(body: str):
    out = []
    if "g_runtime_alloc_index.find_innermost(" not in body:
        out.append("does not consult the index")
    if re.search(r"\b(?:for|while)\s*\([^)]*g_runtime_alloc_registry", body) or "g_runtime_alloc_registry.begin()" in body:
        out.append("iterates the registry")
    if not re.search(r"GGML_ASSERT\(\s*runtime_registry_row_matches_index_entry\(", body):
        out.append("does not assert the row still matches the index entry")
    return out


BACKSTOP = "GGML_ASSERT(runtime_registry_row_matches_index_entry(it->second.handle, hit)"


def test_lookup_consults_the_index_and_does_not_scan():
    assert lookup_violations(lookup_body(CODE)) == []


@pytest.mark.parametrize(
    "body",
    [
        "for (const auto & kv : g_runtime_alloc_registry) { use(kv); } " + BACKSTOP,
        "g_runtime_alloc_index.find_innermost(a, &h); for (auto it = g_runtime_alloc_registry.begin(); it != e; ++it) {} " + BACKSTOP,
        "g_runtime_alloc_index.find_innermost(a, &h); while (x != g_runtime_alloc_registry.end()) {} " + BACKSTOP,
        "auto it = g_runtime_alloc_registry.begin();",
        "for (int i = 0; i < n; i++) {}",
        # the lookup without its backstop
        "g_runtime_alloc_index.find_innermost(a, &h); return true;",
    ],
)
def test_lookup_gate_has_a_witness(body):
    assert lookup_violations(body), body


def test_lookup_gate_allows_an_unrelated_loop_beside_the_index_query():
    body = "g_runtime_alloc_index.find_innermost(a, &h); " + BACKSTOP + "; for (int i = 0; i < 3; i++) { pick(i); }"
    assert lookup_violations(body) == []


def test_the_backstop_compares_both_ends_of_the_extent():
    s, e = function_span(CODE, r"static bool runtime_registry_row_matches_index_entry\(")
    helper = CODE[s:e]
    assert re.search(r"reinterpret_cast<uintptr_t>\(h\.ptr\)\s*==\s*e\.base", helper)
    assert re.search(r"address_range_index::end_of\(e\.base,\s*h\.size\)\s*==\s*e\.end", helper)
    # and the assertion in the lookup is not dropped or weakened to a log
    assert "GGML_ASSERT(" in lookup_body(CODE)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
