#!/usr/bin/env python3
"""Source pin: the standalone cache test targets name none of the IO pool's test-only symbols (th32 §11 T5).

Several test targets in ggml/src/ggml-sycl/CMakeLists.txt compile unified-cache.cpp and its neighbours from source,
most of them under GGML_SYCL_PRIVATE_TESTING, but do not compile lazy-io-pool.cpp: they take the pool from the
ordinary backend library, which is built without that macro. A symbol the pool declares only under the macro
(lazy_ring_mutant_is, the ggml_sycl_test_lazy_* seams and the rest) is then in no object those targets link, so a
reference to it from one of their translation units is an undefined reference at link time. A syntax-only check cannot
see that, and a mutant check in unified-cache.cpp has broken two builds this way.

The pin derives both sides rather than listing them:
- the standalone targets are the add_executable targets whose literal source list names unified-cache.cpp and not
  lazy-io-pool.cpp, and the forbidden units are every source those targets list;
- the test-only symbols are the names lazy-io-pool.hpp declares at namespace scope inside a GGML_SYCL_PRIVATE_TESTING
  block and nowhere outside one: functions, types and aliases.
It fails naming each forbidden unit, line and symbol where a reference appears, comments and literals excluded.

Engagement: the derived units include the four cache sources the targets share (unified-cache.cpp, model-lifecycle.cpp,
pinned-pool.cpp, mem-handle.cpp), each derived unit exists, and lazy_ring_mutant_is is among the derived symbols.
Positive controls, on scratch copies held in memory: a reference planted in unified-cache.cpp fails the pin; the same
name planted in a comment and in a string literal does not; a standalone target planted in the CMakeLists that compiles
ggml-sycl.cpp brings that unit's references into the forbidden set; a planted target that also compiles lazy-io-pool.cpp
is not standalone.

Scope: the units the targets compile, not the headers those units include, and no variables among the symbols. A header
naming a test-only pool symbol in an inline function a standalone unit calls would escape the pin.
"""
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sycl_gate import ROOT, finish, gate, views  # noqa: E402

CMAKE_FILES = ("ggml/src/ggml-sycl/CMakeLists.txt", "tests/CMakeLists.txt")
POOL_HEADER = "ggml/src/ggml-sycl/lazy-io-pool.hpp"
POOL_SOURCE = "ggml/src/ggml-sycl/lazy-io-pool.cpp"
CACHE_SOURCE = "ggml/src/ggml-sycl/unified-cache.cpp"
SHARED_UNITS = ("unified-cache.cpp", "model-lifecycle.cpp", "pinned-pool.cpp", "mem-handle.cpp")
NOT_NAMES = {
    "alignas", "alignof", "decltype", "defined", "noexcept", "operator", "return", "sizeof", "static_assert",
    "void", "bool", "char", "int", "long", "short", "unsigned", "signed", "float", "double", "auto", "const",
}


def read(rel: str) -> str:
    path = ROOT / rel
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def strip_cmake_comments(text: str) -> str:
    """CMake text with each `#` comment blanked to its line end; a `#` inside a quoted string is kept."""
    out = []
    for line in text.split("\n"):
        quoted, cut = False, len(line)
        for i, ch in enumerate(line):
            if ch == '"' and (i == 0 or line[i - 1] != "\\"):
                quoted = not quoted
            elif ch == "#" and not quoted:
                cut = i
                break
        out.append(line[:cut])
    return "\n".join(out)


def target_sources(cmake: dict) -> dict:
    """{target: [repo-relative source]} for every add_executable with a literal source list, from {path: text}."""
    found = {}
    for rel, text in cmake.items():
        here = str(Path(rel).parent)
        code = strip_cmake_comments(text)
        for m in re.finditer(r"\badd_executable\(\s*([^\s)]+)([^)]*)\)", code):
            units = []
            for token in m.group(2).split():
                token = token.replace("${CMAKE_CURRENT_SOURCE_DIR}", here)
                token = token.replace("${PROJECT_SOURCE_DIR}", ".").replace("${CMAKE_SOURCE_DIR}", ".")
                if "$" in token or not re.search(r"\.(c|cc|cpp|cxx)$", token):
                    continue
                path = token if token.startswith(here + "/") or token.startswith("./") else here + "/" + token
                units.append(os.path.normpath(path).replace("\\", "/"))
            found.setdefault(m.group(1), []).extend(units)
    return found


def standalone_units(cmake: dict):
    """(standalone targets, forbidden units): the targets that compile the cache but not the pool, and their sources."""
    targets = {}
    for name, units in target_sources(cmake).items():
        if any(u.endswith("ggml-sycl/unified-cache.cpp") for u in units) and not any(
                u.endswith("ggml-sycl/lazy-io-pool.cpp") for u in units):
            targets[name] = units
    forbidden = sorted({u for units in targets.values() for u in units})
    return sorted(targets), forbidden


def namespace_scope_mask(bare: str) -> list:
    """Per offset of `bare`, whether it sits at namespace scope: every open brace is a namespace's, no paren is open."""
    mask, braces, parens = [False] * len(bare), [], 0
    for i, ch in enumerate(bare):
        mask[i] = parens == 0 and all(kind == "ns" for kind in braces)
        if ch == "(":
            parens += 1
        elif ch == ")":
            parens = max(0, parens - 1)
        elif ch == "{":
            head = bare[max(0, i - 200):i]
            head = re.split(r"[;{}]", head)[-1]
            scope = re.search(r"(\bnamespace\b[\w\s:]*|\bextern\s*\"[^\"]*\"\s*)$", head)
            braces.append("ns" if scope else "body")
        elif ch == "}" and braces:
            braces.pop()
    return mask


def private_testing_lines(bare: str) -> set:
    """The 1-based lines of `bare` inside a GGML_SYCL_PRIVATE_TESTING block (its #else side excluded)."""
    inside, stack = set(), []
    for number, line in enumerate(bare.split("\n"), 1):
        directive = line.strip()
        if re.match(r"#\s*if", directive):
            pt = bool(re.match(r"#\s*(ifdef\s+GGML_SYCL_PRIVATE_TESTING\b|if\s+(defined\s*\(\s*)?"
                               r"GGML_SYCL_PRIVATE_TESTING\b)", directive))
            stack.append(pt)
            continue
        if re.match(r"#\s*(else|elif)", directive) and stack:
            stack[-1] = False
            continue
        if re.match(r"#\s*endif", directive) and stack:
            stack.pop()
            continue
        if any(stack):
            inside.add(number)
    return inside


def private_testing_symbols(header: str) -> set:
    """The names `header` declares at namespace scope only inside GGML_SYCL_PRIVATE_TESTING blocks."""
    _, bare = views(header)
    mask = namespace_scope_mask(bare)
    pt_lines = private_testing_lines(bare)
    starts = [0]
    for i, ch in enumerate(bare):
        if ch == "\n":
            starts.append(i + 1)

    def line_of(offset: int) -> int:
        lo, hi = 0, len(starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if starts[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    testing, plain = set(), set()
    patterns = (r"\b([A-Za-z_]\w*)\s*\(", r"\benum\s+class\s+([A-Za-z_]\w*)",
                r"\b(?:class|struct|union)\s+([A-Za-z_]\w*)", r"\busing\s+([A-Za-z_]\w*)\s*=")
    for pattern in patterns:
        for m in re.finditer(pattern, bare):
            name = m.group(1)
            if not mask[m.start(1)] or name in NOT_NAMES or name.isupper():
                continue
            (testing if line_of(m.start(1)) in pt_lines else plain).add(name)
    return testing - plain


def violations(forbidden: list, symbols: set, source) -> list:
    """`unit:line: symbol` for each reference to `symbols` in a forbidden unit, read through `source(rel)`."""
    if not symbols:
        return []
    found = []
    pattern = re.compile(r"\b(" + "|".join(sorted(map(re.escape, symbols))) + r")\b")
    for unit in forbidden:
        _, bare = views(source(unit))
        for number, line in enumerate(bare.split("\n"), 1):
            for m in pattern.finditer(line):
                found.append("%s:%d: %s" % (unit, number, m.group(1)))
    return found


CMAKE = {rel: read(rel) for rel in CMAKE_FILES}
TARGETS, FORBIDDEN = standalone_units(CMAKE)
SYMBOLS = private_testing_symbols(read(POOL_HEADER))

print("standalone targets: %s" % ", ".join(TARGETS))
print("forbidden units: %d, test-only pool symbols: %d" % (len(FORBIDDEN), len(SYMBOLS)))

with gate("the standalone targets are derived from CMake and compile the four shared cache sources"):
    assert TARGETS, "no add_executable compiles unified-cache.cpp without lazy-io-pool.cpp"
    missing = [u for u in SHARED_UNITS if not any(f.endswith("ggml-sycl/" + u) for f in FORBIDDEN)]
    assert not missing, "derived units lack %s" % ", ".join(missing)

with gate("every derived unit exists in the tree"):
    absent = [u for u in FORBIDDEN if not (ROOT / u).exists()]
    assert not absent, "derived units not in the tree: %s" % ", ".join(absent)

with gate("the test-only pool symbols are derived from lazy-io-pool.hpp and include lazy_ring_mutant_is"):
    for name in ("lazy_ring_mutant_is", "ggml_sycl_test_lazy_ring_mutant_select", "lazy_ring_mutant"):
        assert name in SYMBOLS, "%s is not among the derived symbols" % name
    assert "lazy_io_pool_shutdown" not in SYMBOLS, "a production declaration was taken as test-only"

with gate("no unit a standalone target compiles names a test-only pool symbol"):
    found = violations(FORBIDDEN, SYMBOLS, read)
    assert not found, "; ".join(found[:20])

# The planting controls compare with the unplanted scan, so they score what the planting added on any tree.
BASELINE = violations(FORBIDDEN, SYMBOLS, read)
PLANT_LINE = read(CACHE_SOURCE).count("\n") + 4

with gate("positive control: a reference planted in unified-cache.cpp fails the pin"):
    planted = read(CACHE_SOURCE) + "\nnamespace ggml_sycl {\nbool planted_check() {\n" \
                                   "    return lazy_ring_mutant_is(lazy_ring_mutant::NONE);\n}\n}\n"
    found = violations(FORBIDDEN, SYMBOLS, lambda rel: planted if rel == CACHE_SOURCE else read(rel))
    want = "%s:%d: lazy_ring_mutant_is" % (CACHE_SOURCE, PLANT_LINE)
    assert want in found and want not in BASELINE, [f for f in found if f not in BASELINE][:5]

with gate("negative control: the name in a comment or a string literal is not a reference"):
    planted = read(CACHE_SOURCE) + "\n// lazy_ring_mutant_is(lazy_ring_mutant::NONE)\n" \
                                   "static const char * planted_text = \"lazy_ring_mutant_is\";\n"
    found = violations(FORBIDDEN, SYMBOLS, lambda rel: planted if rel == CACHE_SOURCE else read(rel))
    assert found == BASELINE, [f for f in found if f not in BASELINE][:5]

with gate("positive control: a planted standalone target brings ggml-sycl.cpp into the forbidden set"):
    cmake = dict(CMAKE)
    cmake[CMAKE_FILES[0]] += "\nadd_executable(test-planted-standalone\n    unified-cache.cpp\n    ggml-sycl.cpp)\n"
    _, forbidden = standalone_units(cmake)
    found = violations(forbidden, SYMBOLS, read)
    assert any(f.startswith("ggml/src/ggml-sycl/ggml-sycl.cpp:") for f in found), "ggml-sycl.cpp was not scanned"

with gate("control: a planted target that also compiles lazy-io-pool.cpp is not standalone"):
    cmake = dict(CMAKE)
    cmake[CMAKE_FILES[0]] += ("\nadd_executable(test-planted-carrier\n    unified-cache.cpp\n    ggml-sycl.cpp\n"
                              "    lazy-io-pool.cpp)\n")
    targets, forbidden = standalone_units(cmake)
    assert "test-planted-carrier" not in targets and POOL_SOURCE not in forbidden, targets

finish("lazy ring mutant selector closure: PASS", min_checks=8)
