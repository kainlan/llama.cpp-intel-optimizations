"""Source contract for llama.cpp-5efe C2: the FILE_MAPPED tier is a tier, not a flag on a host allocation.

The runtime test (test-sycl-file-mapped-tier) discriminates on behaviour; this gate pins the properties it cannot see
from outside, or that a rebase restoring upstream's shape would silently undo:

1. unified_free_record() handles FILE_MAPPED before every other release branch: before the UNREGISTERED no-free
   case, the zone_managed branch, the `::free` branch and the sycl::free fallthrough. A mapping handed to any of them
   is freed as something it is not;
2. the branch releases through unified_cache_raw_unmap_file and unregisters the pointer-type registry row, and
   names neither `::free` nor `sycl::free`;
3. the front door mints the owner control (EXTERNAL_EXACT) before the mapping exists: the create() call precedes
   unified_cache_raw_map_file;
4. memory_location::device_mappable() does not name FILE_MAPPED (a device is never handed a file mapping), and
   mem_handle::resolve(int) refuses a FILE_MAPPED owner;
5. unified_alloc() (the legacy raw-pointer entry) refuses FILE_MAPPED and UNREGISTERED;
6. alloc_registry's type and the tier both have FILE_MAPPED, and the front door registers the reservation under
   alloc_type::FILE_MAPPED, not HOST_MALLOC (which would send it to ::free);
7. the validator's host-ledger sum does not add the file-mapped bytes to the host total.

Plain file I/O, comment-stripped text.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"

_LEXEME_RE = re.compile(
    r'"(?:\\.|[^"\\\n])*"'
    r"|'(?:\\.|[^'\\\n])*'"
    r"|//[^\n]*"
    r"|/\*.*?\*/",
    flags=re.DOTALL,
)


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def code(rel: str) -> str:
    return strip_comments((SYCL / rel).read_text())


def balanced(text: str, open_idx: int, open_ch: str = "{", close_ch: str = "}") -> str:
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == open_ch:
            depth += 1
        elif text[i] == close_ch:
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i]
    raise AssertionError("unbalanced brackets")


def function_body(text: str, signature_re: str) -> str:
    m = re.search(signature_re, text)
    assert m, f"function not found: {signature_re}"
    return balanced(text, text.index("{", m.end()))


def free_record_body() -> str:
    return function_body(code("unified-cache.cpp"), r"static\s+bool\s+unified_free_record\s*\(const runtime_alloc_record\s*&\s*rec\)")


def front_door_body() -> str:
    return function_body(
        code("unified-cache.cpp"),
        r"allocation_result\s+unified_allocate_file_mapping\s*\(const file_mapping_request\s*&\s*req\)\s*noexcept",
    )


# ---- 1, 2. the release branch ----


def test_file_mapped_branch_precedes_every_other_release_branch():
    body = free_record_body()
    branch = re.search(r"if\s*\(\s*rec\.handle\.tier\s*==\s*alloc_tier::FILE_MAPPED\s*\)\s*\{", body)
    assert branch, "unified_free_record has no FILE_MAPPED branch"
    for label, rx in (
        ("the UNREGISTERED case", r"alloc_tier::UNREGISTERED"),
        ("the zone_managed branch", r"rec\.handle\.zone_managed"),
        ("the ::free branch", r"(?<![\w:])::free\s*\(|\bstd::free\s*\("),
        ("the sycl::free fallthrough", r"\bsycl::free\s*\("),
    ):
        m = re.search(rx, body)
        assert m, f"unified_free_record has no {label} (the gate lost its anchor: {rx})"
        assert branch.start() < m.start(), f"the FILE_MAPPED branch comes after {label}"


def test_file_mapped_branch_unmaps_once_and_never_frees():
    body = free_record_body()
    start = re.search(r"if\s*\(\s*rec\.handle\.tier\s*==\s*alloc_tier::FILE_MAPPED\s*\)\s*\{", body)
    assert start, "unified_free_record has no FILE_MAPPED branch"
    branch = balanced(body, start.end() - 1)
    assert "unified_cache_raw_unmap_file" in branch, "the FILE_MAPPED branch does not release through the wrapper"
    assert "unregister_alloc" in branch, "the FILE_MAPPED branch leaves the pointer-type registry row"
    assert not re.search(r"::free\s*\(|\bfree\s*\(|sycl::free", branch), "the FILE_MAPPED branch frees the mapping"
    assert "return" in branch, "the FILE_MAPPED branch falls through to the later release branches"


# ---- 3. owner first ----


def test_front_door_mints_the_owner_before_mapping():
    body = front_door_body()
    create = re.search(r"allocation_owner_internal_access::create\s*\(", body)
    mapped = re.search(r"unified_cache_raw_map_file\s*\(", body)
    assert create and mapped, "the front door lacks create() or the map call"
    assert create.start() < mapped.start(), "the mapping exists before its owner control"
    args = balanced(body, create.end() - 1, "(", ")")
    assert "EXTERNAL_EXACT" in args, "the control is not minted EXTERNAL_EXACT"
    assert "CACHE_BACKING" not in body, "a file mapping must never mint CACHE_BACKING"


# ---- 6. the two registries ----


def test_the_reservation_is_registered_as_file_mapped_not_host_malloc():
    body = front_door_body()
    reg = re.search(r"register_alloc\s*\(", body)
    assert reg, "the front door does not register the reservation in alloc_registry"
    args = balanced(body, reg.end() - 1, "(", ")")
    assert "alloc_type::FILE_MAPPED" in args, f"alloc_registry row is not FILE_MAPPED: {args}"
    assert "HOST_MALLOC" not in args
    assert re.search(r"rec\.handle\.tier\s*=\s*alloc_tier::FILE_MAPPED", body), "the record's tier is not FILE_MAPPED"


def test_the_enums_name_file_mapped():
    def enum_body(rel: str, name: str) -> str:
        m = re.search(r"\benum\s+class\s+" + name + r"\b[^{;]*\{(.*?)\}", code(rel), flags=re.DOTALL)
        assert m, f"enum class {name} not found in {rel}"
        return m.group(1)

    assert re.search(r"\bFILE_MAPPED\b", enum_body("unified-cache.hpp", "alloc_tier")), "alloc_tier has no FILE_MAPPED"
    assert re.search(r"\bFILE_MAPPED\b", enum_body("alloc-registry.hpp", "alloc_type")), "alloc_type has no FILE_MAPPED"


# ---- 4. a device is never handed a mapping ----


def test_device_mappable_excludes_file_mapped():
    text = code("unified-cache.hpp")
    m = re.search(r"bool\s+device_mappable\s*\(\s*\)\s*const", text)
    assert m, "memory_location::device_mappable() not found"
    body = function_body(text, r"bool\s+device_mappable\s*\(\s*\)\s*const")
    assert "FILE_MAPPED" not in body, "device_mappable() admits FILE_MAPPED"
    assert "DEVICE_VRAM" in body and "HOST_PINNED" in body, f"device_mappable() lost its tiers: {body}"


def test_resolve_for_a_device_refuses_a_file_mapped_owner():
    body = function_body(code("mem-handle.cpp"), r"resolved_ptr\s+mem_handle::resolve\s*\(\s*int\s+device_id\s*\)\s*const")
    assert "alloc_tier::FILE_MAPPED" in body, "mem_handle::resolve(device) does not refuse a FILE_MAPPED owner"
    assert re.search(r"FILE_MAPPED\s*\)\s*\{\s*return\s*\{\s*\}\s*;", body), "the refusal does not return an empty pointer"


# ---- 5. the legacy raw entry ----


def test_unified_alloc_refuses_a_file_mapping():
    body = function_body(code("unified-cache.cpp"), r"bool\s+unified_alloc\s*\(const alloc_request\s*&\s*req_in,\s*alloc_handle\s*\*\s*out\)")
    m = re.search(r"if\s*\(\s*tier\s*==\s*alloc_tier::UNREGISTERED\s*\|\|\s*tier\s*==\s*alloc_tier::FILE_MAPPED\s*\)\s*\{(.*?)return\s+false\s*;", body, flags=re.DOTALL)
    assert m, "unified_alloc() does not refuse FILE_MAPPED together with UNREGISTERED"


# ---- 7. the host ledger ----


def test_the_registry_validator_does_not_charge_file_mapped_bytes_to_the_host_total():
    body = function_body(code("unified-cache.cpp"), r"unified_alloc_validate_registry\s*\([^)]*\)")
    assert "registry_file_mapped" in body, "the validator does not keep file-mapped bytes apart"
    assert not re.search(r"registry_host\w*\s*\+=\s*[^;]*FILE_MAPPED", body)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
