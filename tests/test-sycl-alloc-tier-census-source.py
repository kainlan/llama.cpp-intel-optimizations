"""Source contract for llama.cpp-5efe C2: no name in the tree says "mmap" without meaning a FILE_MAPPED owner.

The tier-switch census (docs/design/sycl-mmap-host-tier.md §4.8) renames the legacy mapping vocabulary, because
"host memory that is not USM" would equally describe a file mapping and invite registering one under it:

  alloc_tier::MMAP_TRACKED         -> alloc_tier::UNREGISTERED (not an allocation: the query result for a pointer the
                                      registry does not know), plus the real alloc_tier::FILE_MAPPED
  alloc_type::MMAP                 -> split by release semantics: HOST_MALLOC (released with ::free) and
                                      HOST_OWNED_ELSEWHERE (registered so it can be identified, never freed)
  memory_tier::MMAP                -> memory_tier::UNRESOLVED (the initial value and "not found" result of a lookup)
  HOST_NON_USM                     -> never introduced

This gate fails while any of the four legacy names remains anywhere in the tree's C and C++ sources, tests/ included,
and requires the names they became. It reads comment-stripped text, so prose that mentions a legacy name does not
trip it; it also reads the three enum bodies, because an enumerator's own definition is unqualified.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL_DIR = ROOT / "ggml/src/ggml-sycl"

SCAN_DIRS = ["ggml", "src", "tests", "common", "tools", "examples"]
SUFFIXES = {".cpp", ".hpp", ".h", ".c", ".cu", ".cuh"}

LEGACY = {
    "MMAP_TRACKED": re.compile(r"\bMMAP_TRACKED\b"),
    "alloc_type::MMAP": re.compile(r"\balloc_type\s*::\s*MMAP\b"),
    "memory_tier::MMAP": re.compile(r"\bmemory_tier\s*::\s*MMAP\b"),
    "HOST_NON_USM": re.compile(r"\bHOST_NON_USM\b"),
}

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


def tree_sources() -> dict:
    files = {}
    for d in SCAN_DIRS:
        for path in sorted((ROOT / d).rglob("*")):
            if path.suffix in SUFFIXES and path.is_file():
                files[path.relative_to(ROOT).as_posix()] = strip_comments(path.read_text(errors="replace"))
    return files


def enum_body(code: str, name: str) -> str:
    m = re.search(r"\benum\s+class\s+" + name + r"\b[^{;]*\{(.*?)\}", code, flags=re.DOTALL)
    assert m, f"enum class {name} not found"
    return m.group(1)


def enumerators(body: str) -> list:
    return re.findall(r"^\s*([A-Z][A-Z0-9_]*)\b", body, flags=re.MULTILINE)


def test_no_legacy_name_remains_in_the_tree():
    left = []
    for rel, code in tree_sources().items():
        for label, rx in LEGACY.items():
            for m in rx.finditer(code):
                left.append(f"{rel}:{code.count(chr(10), 0, m.start()) + 1} {label}")
    assert not left, f"{len(left)} legacy mapping names remain: " + ", ".join(left)


def test_the_enums_hold_no_mmap_enumerator():
    for rel, enum in (
        ("alloc-registry.hpp", "alloc_type"),
        ("tensor-types.hpp", "memory_tier"),
        ("unified-cache.hpp", "alloc_tier"),
    ):
        names = enumerators(enum_body(strip_comments((SYCL_DIR / rel).read_text()), enum))
        assert not [n for n in names if n.startswith("MMAP")], f"{enum} still has an MMAP enumerator: {names}"


def test_the_renamed_names_exist():
    reg = enumerators(enum_body(strip_comments((SYCL_DIR / "alloc-registry.hpp").read_text()), "alloc_type"))
    for n in ("HOST_MALLOC", "HOST_OWNED_ELSEWHERE"):
        assert n in reg, f"alloc_type has no {n}: {reg}"
    assert "FILE_MAPPED" in reg, f"alloc_type has no FILE_MAPPED: {reg}"
    tiers = enumerators(enum_body(strip_comments((SYCL_DIR / "tensor-types.hpp").read_text()), "memory_tier"))
    assert "UNRESOLVED" in tiers, f"memory_tier has no UNRESOLVED: {tiers}"
    cache = enumerators(enum_body(strip_comments((SYCL_DIR / "unified-cache.hpp").read_text()), "alloc_tier"))
    assert "UNREGISTERED" in cache, f"alloc_tier has no UNREGISTERED: {cache}"
    assert "FILE_MAPPED" in cache, f"alloc_tier has no FILE_MAPPED: {cache}"


def test_the_gate_can_fail():
    """A planted legacy name in comment-stripped text is found; the same name in a comment is not."""
    code = strip_comments("// MMAP_TRACKED in prose\nauto t = alloc_tier::MMAP_TRACKED;\n")
    hits = [label for label, rx in LEGACY.items() if rx.search(code)]
    assert hits == ["MMAP_TRACKED"], hits
    assert not any(rx.search(strip_comments("/* alloc_type::MMAP */")) for rx in LEGACY.values())


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
