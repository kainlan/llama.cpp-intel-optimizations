"""Source contract for llama.cpp-5efe C2: the raw mapping primitives have an allowlist.

`mmap`, `munmap`, `mlock` and `madvise` are the calls that make, lock and release address space the unified cache
must own. The FILE_MAPPED tier (docs/design/sycl-mmap-host-tier.md §4.1) keeps them in one device-free translation
unit, ggml/src/ggml-sycl/file-mapping.cpp, behind four wrappers (`unified_cache_raw_map_file`,
`unified_cache_raw_lock_file_segments`, `unified_cache_raw_advise_file_segments`,
`unified_cache_raw_unmap_file`). An allowlist entry in the canonical contract does not enforce itself, so this gate
does:

1. it tokenizes first: comments, string literals and character literals are stripped, then `\\b(mmap|munmap|mlock|
   madvise)\\s*\\(` is matched. A comment such as "host/mmap (conservative)" or a literal "mmap(" is not a call, and a
   real `mmap (` with a space is. A fixture of those forms is part of this test;
2. every call in ggml/src/ggml-sycl/ must sit in a listed site: the file-mapping TU (pinned by call count too: a
   second `munmap` there is a second release, and the tier's munmap counter would not see it), or one of today's eight
   sites, pinned by file and by call count (a ninth call in a listed file is as much a new site as one in a new file);
3. the canonical contract's §3 allowlist names the four wrappers.

th32 T1's HOST_CPU wrapper (ggml/src/ggml-sycl/host-cpu-mapping.cpp, `unified_cache_raw_alloc_host_cpu` and
`unified_cache_raw_free_host_cpu`) is the second wrapper TU: anonymous mmap, MADV_HUGEPAGE and a full mlock, in one place.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL_DIR = ROOT / "ggml/src/ggml-sycl"
CONTRACT = ROOT / "docs/design/sycl-canonical-memory-architecture.md"

# the wrapper translation units: any call in host-cpu-mapping.cpp; file-mapping.cpp is pinned to the calls of its four
# wrappers, so a second release (or a second map) cannot be added unseen
WRAPPER_TUS = {"host-cpu-mapping.cpp"}
PINNED_WRAPPER_TUS = {
    # reserve + segment map, the one munmap wrapper, the lock, the advise
    "file-mapping.cpp": {"mmap": 2, "munmap": 1, "mlock": 1, "madvise": 1},
}

# today's sites, by file relative to ggml/src/ggml-sycl: {call: count}
ALLOWED = {
    # the DPCT memory manager's address-space reservation, and its release
    "dpct/helper.hpp": {"mmap": 1, "munmap": 1},
    # the weight-cache allocator's guard-page mapping (one mapping, released at two sites)
    "ggml-sycl.cpp": {"mmap": 1, "munmap": 2},
    # the cache container allocator's guard-page mapping (one mapping, released at two sites)
    "unified-cache.hpp": {"mmap": 1, "munmap": 2},
}

CALL_RE = re.compile(r"\b(mmap|munmap|mlock|madvise)\s*\(")

_LEXEME_RE = re.compile(
    r'"(?:\\.|[^"\\\n])*"'  # string literal
    r"|'(?:\\.|[^'\\\n])*'"  # char literal
    r"|//[^\n]*"  # line comment
    r"|/\*.*?\*/",  # block comment
    flags=re.DOTALL,
)


def strip_comments_and_literals(src: str) -> str:
    """Drop comments and the contents of string and character literals, keeping newlines."""

    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] == '"':
            return '""'
        if tok[0] == "'":
            return "''"
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def calls_in(src: str) -> dict:
    counts: dict = {}
    for m in CALL_RE.finditer(strip_comments_and_literals(src)):
        counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    return counts


def violations(files: dict) -> list:
    """files: path relative to ggml/src/ggml-sycl -> source text."""
    out = []
    for rel, src in sorted(files.items()):
        counts = calls_in(src)
        if not counts or rel in WRAPPER_TUS:
            continue
        allowed = ALLOWED.get(rel, PINNED_WRAPPER_TUS.get(rel))
        if allowed is None:
            out.append(f"{rel}: {counts} (not a listed site)")
        elif counts != allowed:
            out.append(f"{rel}: {counts} (listed for {allowed})")
    return out


def tree_files() -> dict:
    files = {}
    for path in sorted(SYCL_DIR.rglob("*")):
        if path.suffix in {".cpp", ".hpp", ".h", ".c", ".cu", ".cuh"} and path.is_file():
            files[path.relative_to(SYCL_DIR).as_posix()] = path.read_text()
    return files


# ---- the tokenizer, against a fixture ----


def test_tokenizer_ignores_comments_and_strings_and_finds_a_spaced_call():
    fixture = (
        "// host/mmap (conservative)\n"
        "/* tensor-types: mmap (slowest, fallback) */\n"
        'const char * a = "mmap(";\n'
        "const char c = 'm';\n"
        "void f() { munmap (p, n); }\n"
    )
    assert calls_in(fixture) == {"munmap": 1}, calls_in(fixture)
    assert calls_in("void g() { mmap (nullptr, 4096, 0, 0, -1, 0); }") == {"mmap": 1}


def test_a_call_in_a_non_listed_file_is_a_violation():
    files = tree_files()
    files["some-new-file.cpp"] = "void h() { mmap (nullptr, 4096, 0, 0, -1, 0); }\n"
    found = violations(files)
    assert any(v.startswith("some-new-file.cpp") for v in found), found


def test_a_second_munmap_in_the_file_mapping_tu_is_a_violation():
    files = tree_files()
    files["file-mapping.cpp"] += "\nvoid second_release(void * p) { munmap(p, 4096); }\n"
    assert any(v.startswith("file-mapping.cpp") for v in violations(files)), violations(files)


def test_an_extra_call_in_a_listed_file_is_a_violation():
    files = tree_files()
    files["ggml-sycl.cpp"] += "\nvoid extra() { madvise(nullptr, 0, 0); }\n"
    assert any(v.startswith("ggml-sycl.cpp") for v in violations(files)), violations(files)


# ---- the tree ----


def test_every_mapping_call_in_the_backend_sits_in_a_listed_site():
    found = violations(tree_files())
    assert not found, "mmap/munmap/mlock/madvise outside the allowlist: " + "; ".join(found)


def test_the_wrapper_tu_exists_and_includes_no_sycl_header():
    tu = SYCL_DIR / "file-mapping.cpp"
    hdr = SYCL_DIR / "file-mapping.hpp"
    assert tu.is_file() and hdr.is_file(), "file-mapping.cpp / file-mapping.hpp missing"
    for path in (tu, hdr):
        text = strip_comments_and_literals(path.read_text())
        assert not re.search(r"#\s*include\s*[<\"](sycl|dpct)", text), f"{path.name} includes a SYCL header"


def test_the_host_cpu_wrapper_tu_exists_and_includes_no_sycl_header():
    for name in ("host-cpu-mapping.cpp", "host-cpu-mapping.hpp"):
        path = SYCL_DIR / name
        assert path.is_file(), f"{name} missing"
        text = strip_comments_and_literals(path.read_text())
        assert not re.search(r"#\s*include\s*[<\"](sycl|dpct)", text), f"{name} includes a SYCL header"
    assert calls_in((SYCL_DIR / "host-cpu-mapping.cpp").read_text()).keys() >= {"mmap", "munmap", "mlock", "madvise"}, (
        "the HOST_CPU wrapper does not make, advise, lock and release the mapping itself"
    )


def test_the_canonical_contract_allowlists_the_four_wrappers():
    text = CONTRACT.read_text()
    section = text[text.index("## 3. Allocator Entry Points (Allowlist)") : text.index("## 4. Pointer-Resolution")]
    for name in (
        "unified_cache_raw_map_file",
        "unified_cache_raw_lock_file_segments",
        "unified_cache_raw_unmap_file",
        "unified_cache_raw_advise_file_segments",
        "unified_cache_raw_alloc_host_cpu",
        "unified_cache_raw_free_host_cpu",
    ):
        assert name in section, f"the canonical contract §3 allowlist does not name {name}"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
