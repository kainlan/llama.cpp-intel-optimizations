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

llama.cpp-th32 T1b adds the HOST_CPU tier to the same census (docs/design/sycl-storage-tier-lazy-tables.md §3.1, "The
tier-switch census T1 must audit"). A tier that is added to an enum is not added to the tree until every switch over
that enum has learned it, so the second half of this file reads the switches: every switch that names FILE_MAPPED must
name HOST_CPU of the same enum, and the sites that hand a pointer to a device must not name it at all, because a
HOST_CPU allocation is read by the CPU only and is never device-mappable.
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


# ---------------------------------------------------------------------------------------------------------------------
# llama.cpp-th32 T1b: the HOST_CPU tier and its census
# ---------------------------------------------------------------------------------------------------------------------


def sycl_sources() -> dict:
    """The backend's own non-test sources, comment-stripped, keyed by path relative to ggml/src/ggml-sycl."""
    files = {}
    for path in sorted(SYCL_DIR.rglob("*")):
        rel = path.relative_to(SYCL_DIR).as_posix()
        if path.suffix in SUFFIXES and path.is_file() and not rel.startswith("tests/"):
            files[rel] = strip_comments(path.read_text(errors="replace"))
    return files


def matching_brace(code: str, open_at: int) -> int:
    """Index just past the brace that closes the one at `open_at`."""
    depth = 0
    for i in range(open_at, len(code)):
        if code[i] == "{":
            depth += 1
        elif code[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    raise AssertionError("unbalanced braces")


def switch_bodies(code: str) -> list:
    """(line, body) of every `switch (...) { ... }` in the text, nested ones included."""
    out = []
    for m in re.finditer(r"\bswitch\s*\(", code):
        open_at = code.find("{", m.end())
        assert open_at >= 0
        out.append((code.count("\n", 0, m.start()) + 1, code[open_at : matching_brace(code, open_at)]))
    return out


def function_body(code: str, header_re: str) -> str:
    """Body of the one function whose header matches `header_re` (the match must be unique, and be a definition)."""
    hits = [m for m in re.finditer(header_re, code)]
    assert len(hits) == 1, f"{header_re!r} matched {len(hits)} times"
    open_at = code.find("{", hits[0].end())
    assert open_at >= 0
    return code[open_at : matching_brace(code, open_at)]


def test_the_new_names_exist():
    hpp = strip_comments((SYCL_DIR / "unified-cache.hpp").read_text())
    reg = strip_comments((SYCL_DIR / "alloc-registry.hpp").read_text())
    tier = enum_body(hpp, "alloc_tier")
    assert re.search(r"\bHOST_CPU\s*=\s*4\b", tier), f"alloc_tier has no HOST_CPU = 4: {enumerators(tier)}"
    assert "HOST_CPU" in enumerators(enum_body(reg, "alloc_type"))
    constraints = re.search(r"struct\s+alloc_constraints\s*\{(.*?)\n\};", hpp, flags=re.DOTALL)
    assert constraints, "struct alloc_constraints not found"
    assert re.search(r"\bbool\s+must_host_cpu\s*=\s*false\s*;", constraints.group(1)), "alloc_constraints.must_host_cpu"
    assert enumerators(enum_body(hpp, "lazy_residency")) == ["HOST", "STORAGE"]
    assert "STORAGE" in enumerators(enum_body(hpp, "cache_location"))
    assert "STORAGE" in enumerators(enum_body(hpp, "expert_resolve_tier"))
    roles = enum_body(hpp, "alloc_role")
    for name, value in (("LAZY_ROW_CACHE", 9), ("LAZY_STAGING", 10), ("LAZY_OUTPUT", 11)):
        assert re.search(rf"\b{name}\s*=\s*{value}\b", roles), f"alloc_role has no {name} = {value}"
    assert re.search(r"\bOTHER\s*=\s*8\b", roles), "OTHER moved: the new roles are appended after it"


def test_every_switch_that_names_file_mapped_names_host_cpu():
    """The tier-switch census: FILE_MAPPED and HOST_CPU are both host memory with no USM identity, so a switch that
    says what a mapping is must say what a locked anonymous mapping is, in the same enum."""
    seen = 0
    missing = []
    for rel, code in sycl_sources().items():
        for line, body in switch_bodies(code):
            for enum in ("alloc_tier", "alloc_type"):
                if re.search(rf"\b{enum}\s*::\s*FILE_MAPPED\b", body):
                    seen += 1
                    if not re.search(rf"\b{enum}\s*::\s*HOST_CPU\b", body):
                        missing.append(f"{rel}:{line} {enum}")
    assert seen >= 9, f"the census found only {seen} switches that name FILE_MAPPED, expected at least 9"
    assert not missing, "switches that name FILE_MAPPED and not HOST_CPU: " + ", ".join(missing)


def test_a_host_cpu_pointer_is_never_resolved_for_a_device():
    code = sycl_sources()["mem-handle.cpp"]
    body = function_body(code, r"resolved_ptr\s+mem_handle::resolve\s*\(\s*int\s+device_id\s*\)\s*const")
    assert re.search(r"\balloc_tier\s*::\s*HOST_CPU\b", body), "resolve(device_id) hands out a HOST_CPU pointer"


def test_host_cpu_is_named_only_where_it_is_not_handed_to_a_device():
    """The sites that name HOST_PINNED because they hand a pointer to a device must fall to their else arm for a
    HOST_CPU pointer. Naming HOST_CPU there is how a locked CPU-only mapping would reach a kernel."""
    sources = sycl_sources()
    for rel in ("common.hpp", "binbcast.cpp", "dmmv.cpp", "element_wise.cpp", "fattn.cpp", "mem-ops.cpp",
                "set_rows.cpp"):
        assert not re.search(r"\balloc_tier\s*::\s*HOST_CPU\b", sources[rel]), f"{rel} names alloc_tier::HOST_CPU"
    cache = sources["unified-cache.cpp"]
    for header in (
        r"static\s+mem_handle\s+make_copy_handle_for_raw_ptr\s*\(",
        r"mem_handle\s+alloc_handle::as_mem_handle\s*\(\s*\)\s*const",
    ):
        assert not re.search(r"\bHOST_CPU\b", function_body(cache, header)), f"{header} names HOST_CPU"
    backend = sources["ggml-sycl.cpp"]
    named = [m.start() for m in re.finditer(r"\balloc_tier\s*::\s*HOST_CPU\b", backend)]
    start = re.search(r"static\s+const\s+char\s*\*\s*ggml_sycl_alloc_tier_name\s*\(", backend)
    assert start, "ggml_sycl_alloc_tier_name not found"
    open_at = backend.find("{", start.end())
    inside = [p for p in named if open_at <= p < matching_brace(backend, open_at)]
    assert len(named) == len(inside) == 1, (
        f"ggml-sycl.cpp names alloc_tier::HOST_CPU {len(named)}x, {len(inside)} in the name function"
    )


def test_a_storage_location_is_reported_and_never_resolved():
    cache = sources_text("unified-cache.cpp")
    body = function_body(cache, r"static\s+expert_resolve_tier\s+expert_tier_from_location\s*\(")
    assert re.search(r"cache_location\s*::\s*STORAGE\b", body), "a STORAGE location has no resolve tier"
    assert re.search(r"expert_resolve_tier\s*::\s*STORAGE\b", body)
    allowed = function_body(
        cache, r"static\s+bool\s+expert_resolution_allowed\s*\(\s*const\s+expert_resolve_request\s*&\s+req,"
    )
    assert not re.search(r"cache_location\s*::\s*STORAGE\b", allowed), "a STORAGE entry has no pointer to resolve"
    backend = sources_text("ggml-sycl.cpp")
    names = function_body(backend, r"static\s+const\s+char\s*\*\s*ggml_sycl_expert_tier_name\s*\(")
    assert re.search(r"expert_resolve_tier\s*::\s*STORAGE\b", names)


def sources_text(rel: str) -> str:
    return strip_comments((SYCL_DIR / rel).read_text(errors="replace"))


def test_the_census_can_fail():
    """A switch that names FILE_MAPPED and not HOST_CPU is found; the same switch with both is not."""
    bad = strip_comments(
        "switch (t) { case alloc_tier::UNREGISTERED: return 1; case alloc_tier::FILE_MAPPED: return 2; }"
    )
    good = strip_comments(
        "switch (t) { case alloc_tier::FILE_MAPPED: return 2; case alloc_tier::HOST_CPU: return 3; }"
    )
    flag = [bool(re.search(r"alloc_tier\s*::\s*HOST_CPU", body)) for _, body in switch_bodies(bad + good)]
    assert flag == [False, True], flag


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
