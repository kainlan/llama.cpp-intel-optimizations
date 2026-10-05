"""Source contract for llama.cpp-ii25: the runtime allocation registry's containment index cannot drift from the registry.

unified_lookup_runtime_allocation() answers "which registered allocation contains this pointer" through
g_runtime_alloc_index (range-index.hpp) instead of scanning every row of g_runtime_alloc_registry (61.65% of Qwen3.8
decode CPU with a 512-expert MoE). The index is correct only while every insert and erase of a registry row also updates
it, so:

(A) g_runtime_alloc_registry is mutated (emplace/erase/insert/clear/extract/swap/operator[]/assignment) ONLY inside the
    runtime_registry_emplace_locked / runtime_registry_erase_locked helpers, which keep the index in step.
(B) A registered row's handle.ptr and handle.size are never assigned through the registry (the index keeps the geometry it
    was given at insertion).
(C) Neither registry nor index is touched from another translation unit (both are file-static in unified-cache.cpp).
(D) The lookup consults the index and does not iterate the registry.

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
HELPERS = (
    "runtime_registry_emplace_locked(void *               ptr,",
    "runtime_registry_erase_locked(runtime_registry_iterator it) noexcept {",
)
# A row's identity or geometry assigned through an iterator/reference into the registry.
GEOMETRY_WRITE_RE = re.compile(r"(?:->|\.)second\s*\.\s*handle\s*(?:\.\s*(?:ptr|size)\s*)?=[^=]")


def function_span(code: str, signature: str):
    start = code.find(signature)
    assert start != -1, f"{signature!r} is not defined"
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


def geometry_writes(code: str):
    return [(code.count("\n", 0, m.start()) + 1) for m in GEOMETRY_WRITE_RE.finditer(code)]


def lookup_body(code: str) -> str:
    sig = "bool unified_lookup_runtime_allocation(const void * ptr, alloc_metadata * out, sycl::queue ** queue_out) {"
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


def test_registered_row_geometry_is_never_rewritten():
    assert geometry_writes(CODE) == []


@pytest.mark.parametrize(
    "line",
    [
        "it->second.handle.size = 4;",
        "it->second.handle.ptr = nullptr;",
        "found->second.handle = other;",
    ],
)
def test_geometry_gate_has_a_witness(line):
    assert geometry_writes(CODE + "\nvoid f() {\n    " + line + "\n}\n"), line


def test_geometry_gate_allows_reads_and_other_fields():
    ok = "const auto & h = it->second.handle; it->second.state = X; it->second.handle.ptr == other;"
    assert geometry_writes(CODE + "\nvoid f() {\n    " + ok + "\n}\n") == []


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
    if re.search(r"\bfor\s*\(", body) or "g_runtime_alloc_registry.begin()" in body:
        out.append("iterates the registry")
    return out


def test_lookup_consults_the_index_and_does_not_scan():
    assert lookup_violations(lookup_body(CODE)) == []


@pytest.mark.parametrize(
    "body",
    [
        "for (const auto & kv : g_runtime_alloc_registry) { use(kv); }",
        "g_runtime_alloc_index.find_innermost(a, &h); for (int i = 0; i < n; i++) {}",
        "auto it = g_runtime_alloc_registry.begin();",
    ],
)
def test_lookup_gate_has_a_witness(body):
    assert lookup_violations(body), body


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
