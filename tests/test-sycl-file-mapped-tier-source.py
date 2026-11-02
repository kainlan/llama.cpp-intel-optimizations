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
7. the registry's tier sums come from one helper: the file-mapped arm adds to its own sum, no arm that adds to the
   host sum names FILE_MAPPED, and the validator reads the helper rather than summing the registry itself (the
   runtime arm of test-sycl-file-mapped-tier reads the same helper through a private accessor);
8. file-mapping.cpp compiles everywhere: its POSIX headers and primitives sit under `#if !defined(_WIN32)`, the
   `#else` arm holds refusing stubs, the segment plan is outside the guard, and the host tests that need POSIX are
   registered only under `if (CMAKE_SYSTEM_NAME STREQUAL "Linux")`;
9. no raw accessor hands out the owner's fd by address, and the ctest name of the runtime test is its binary's;
10. the WILLNEED advice is chunked at 128 KiB with a floor of one page, and the loop steps by the chunk (the kernel
   clamps one call to the backing device's io_pages, so a larger chunk or one call prefetches only part of a
   segment; the residency test's arm 7 measures it where a host can show it);
11. the front door adds its row through runtime_registry_emplace_locked, after the pointer claim, so the row gets its
   containment-index entry and its settle-counter share. A row the map holds without its index entry aborts its own
   release, because runtime_registry_erase_locked asserts the entry was there. Registry-wide coverage (every write of
   g_runtime_alloc_registry kept inside its helpers) is rule (A) of test-sycl-runtime-registry-index-source.py; this
   clause pins only the front door. The direct emplace this front door once used fails it; the control below asserts
   that.

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


FRONT_DOOR_SIG = r"allocation_result\s+unified_allocate_file_mapping\s*\(const file_mapping_request\s*&\s*req\)\s*noexcept"


def front_door_body() -> str:
    return function_body(code("unified-cache.cpp"), FRONT_DOOR_SIG)


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
    # alone, or beside the other CPU-only tier in one condition (llama.cpp-th32 T1b)
    assert re.search(
        r"FILE_MAPPED\s*(?:\|\|\s*\w+\s*==\s*alloc_tier::HOST_CPU\s*)?\)\s*\{\s*return\s*\{\s*\}\s*;", body
    ), "the refusal does not return an empty pointer"


# ---- 5. the legacy raw entry ----


def test_unified_alloc_refuses_a_file_mapping():
    body = function_body(code("unified-cache.cpp"), r"bool\s+unified_alloc\s*\(const alloc_request\s*&\s*req_in,\s*alloc_handle\s*\*\s*out\)")
    m = re.search(r"if\s*\(\s*tier\s*==\s*alloc_tier::UNREGISTERED\s*\|\|\s*tier\s*==\s*alloc_tier::FILE_MAPPED\s*\)\s*\{(.*?)return\s+false\s*;", body, flags=re.DOTALL)
    assert m, "unified_alloc() does not refuse FILE_MAPPED together with UNREGISTERED"


# ---- 7. the host ledger ----


def sums_body() -> str:
    return function_body(code("unified-cache.cpp"), r"static\s+registry_tier_sums\s+unified_alloc_registry_tier_sums\s*\(\s*\)")


def test_the_file_mapped_arm_adds_to_its_own_sum_and_never_to_the_host_sum():
    body = sums_body()
    arms = list(re.finditer(r"(?:if|else\s+if)\s*\(([^{]*?)\)\s*\{([^}]*)\}", body, flags=re.DOTALL))
    assert arms, "the sums helper has no tier arms"
    file_arm = [a for a in arms if "FILE_MAPPED" in a.group(1)]
    assert len(file_arm) == 1, f"expected one FILE_MAPPED arm, got {len(file_arm)}"
    assert re.search(r"sums\s*\.\s*file_mapped\s*\+=", file_arm[0].group(2)), "the FILE_MAPPED arm does not add to sums.file_mapped"
    assert "sums.host" not in file_arm[0].group(2), "the FILE_MAPPED arm adds to the host sum"
    for a in arms:
        if re.search(r"sums\s*\.\s*host\s*\+=", a.group(2)):
            assert "FILE_MAPPED" not in a.group(1), f"an arm that adds to the host sum names FILE_MAPPED: {a.group(1)}"


def test_the_validator_reads_the_helper_and_reports_the_file_mapped_sum():
    body = function_body(code("unified-cache.cpp"), r"unified_alloc_validate_registry\s*\([^)]*\)")
    assert "unified_alloc_registry_tier_sums()" in body, "the validator does not read the one helper"
    assert "g_runtime_alloc_registry" not in body, "the validator sums the registry itself, not through the helper"
    assert "registry_file_mapped" in body, "the validator does not keep file-mapped bytes apart"
    assert not re.search(r"registry_host\w*\s*\+=", body), "the validator adds to the host sum itself"


def test_the_sums_accessor_is_the_helper():
    body = function_body(code("unified-cache.cpp"), r"void\s+unified_alloc_registry_sums_for_test\s*\([^)]*\)\s*noexcept")
    assert "unified_alloc_registry_tier_sums()" in body, "the test accessor does not read the helper the validator reads"


# ---- 8. Windows ----


def sycl_text(rel: str) -> str:
    return (SYCL / rel).read_text()


def guard_regions(text: str) -> tuple:
    """(posix, other): the text of the `#if !defined(_WIN32)` arm(s) and of everything else, at the top nesting level."""
    posix, other, depth, in_posix = [], [], 0, False
    for line in text.splitlines(keepends=True):
        stripped = re.sub(r"^\s*#\s*", "#", line.strip())
        if depth == 0 and re.match(r"#if\s+!defined\s*\(\s*_WIN32\s*\)", stripped):
            in_posix, depth = True, 1
            continue
        if depth >= 1 and stripped.startswith("#if"):
            depth += 1
        elif depth >= 1 and stripped.startswith("#endif"):
            depth -= 1
            if depth == 0:
                in_posix = False
                continue
        elif depth == 1 and stripped.startswith("#else"):
            in_posix = False
            continue
        (posix if in_posix else other).append(line)
    return "".join(posix), "".join(other)


POSIX_HEADERS = r"#\s*include\s*<(sys/\w+\.h|unistd\.h)>"
POSIX_CALLS = r"\b(mmap|munmap|mlock|madvise|fstat|dup|close|sysconf)\s*\("


def test_file_mapping_cpp_guards_its_posix_body_and_stubs_the_rest():
    text = sycl_text("file-mapping.cpp")
    posix, other = guard_regions(text)
    assert posix, "file-mapping.cpp has no `#if !defined(_WIN32)` body"
    assert re.search(POSIX_HEADERS, posix), "the POSIX headers are not inside the guard"
    other_code = strip_comments(other)
    assert not re.search(POSIX_HEADERS, other_code), "a POSIX header is included outside the guard"
    assert not re.search(POSIX_CALLS, other_code), f"a POSIX call sits outside the guard: {re.search(POSIX_CALLS, other_code).group(0)}"
    stubs = other_code[other_code.index("unified_cache_raw_map_file") :] if "unified_cache_raw_map_file" in other_code else ""
    for name in ("unified_cache_raw_map_file", "unified_cache_raw_lock_file_segments", "unified_cache_raw_advise_file_segments", "unified_cache_raw_unmap_file"):
        assert name in other_code, f"the non-POSIX arm has no stub for {name}"
    assert stubs.count("file_map_status::UNSUPPORTED") >= 3, "the stubs do not refuse as UNSUPPORTED"


def test_the_segment_plan_is_outside_the_guard():
    posix, other = guard_regions(sycl_text("file-mapping.cpp"))
    assert "file_mapping_segments(" in strip_comments(other), "the pure segment plan sits inside the POSIX guard, so Windows loses it"
    assert "file_mapping_segments(" not in strip_comments(posix)


def test_the_capability_query_is_false_on_windows():
    hpp = strip_comments(sycl_text("file-mapping.hpp"))
    m = re.search(r"constexpr\s+bool\s+file_mapping_supported\s*\(\s*\)\s*\{(.*?)\n\}", hpp, flags=re.DOTALL)
    assert m, "file-mapping.hpp has no file_mapping_supported()"
    assert re.search(r"#\s*if\s+defined\s*\(\s*_WIN32\s*\)\s*return\s+false\s*;\s*#\s*else\s*return\s+true\s*;", m.group(1)), (
        "file_mapping_supported() is not false on Windows and true elsewhere"
    )


def cmake_blocks() -> dict:
    """target/test name -> True when registered inside `if (CMAKE_SYSTEM_NAME STREQUAL "Linux")`."""
    out, stack = {}, []
    for line in (SYCL / "CMakeLists.txt").read_text().splitlines():
        t = line.strip()
        if re.match(r"if\s*\(", t):
            stack.append(bool(re.search(r'CMAKE_SYSTEM_NAME\s+STREQUAL\s+"Linux"', t)))
        elif re.match(r"endif\s*\(", t) and stack:
            stack.pop()
        for m in re.finditer(r"add_(?:executable|test)\s*\(\s*(?:NAME\s+)?([\w-]+)", t):
            out[m.group(1)] = any(stack)
    return out


def test_the_posix_only_tests_are_registered_only_on_linux():
    reg = cmake_blocks()
    for name in ("test-sycl-file-mapping-host", "test-sycl-file-mapping-residency", "test-sycl-file-mapped-tier"):
        assert name in reg, f"{name} is not registered"
        assert reg[name], f"{name} is registered on every platform, but needs POSIX"
    for name in ("test-sycl-file-mapping-segments", "test-sycl-file-mapping-unsupported"):
        assert name in reg and not reg[name], f"{name} is portable and should not be Linux-only"


def test_the_unsupported_test_compiles_the_stub():
    text = (SYCL / "CMakeLists.txt").read_text()
    m = re.search(r"target_compile_definitions\(\s*test-sycl-file-mapping-unsupported\s+PRIVATE\s+([^)]*)\)", text)
    assert m and "_WIN32" in m.group(1), "test-sycl-file-mapping-unsupported is not compiled with _WIN32 defined"


# ---- 9. no raw fd accessor, and the ctest name ----


def test_no_raw_accessor_hands_out_the_owners_fd_by_address():
    for rel in ("unified-cache.hpp", "unified-cache.cpp"):
        assert "unified_lookup_file_mapping_fd" not in code(rel), (
            f"{rel} brought back unified_lookup_file_mapping_fd: an fd keyed by a raw base pointer outlives its owner"
        )


def test_the_ctest_name_is_the_binary_name():
    text = (SYCL / "CMakeLists.txt").read_text()
    assert re.search(r"add_test\(\s*NAME\s+test-sycl-file-mapped-tier\s+COMMAND\s+test-sycl-file-mapped-tier\s*\)", text), (
        "the runtime test is not registered under its binary's name (`ctest -R '^test-sycl-file-mapped-tier$'` would select nothing)"
    )
    assert not re.search(r"add_test\(\s*NAME\s+sycl-file-mapped-tier\b", text)


# ---- 10. the WILLNEED chunk ----


def test_the_willneed_chunk_is_128_kib_with_a_page_floor_and_the_loop_steps_by_it():
    body = function_body(
        code("file-mapping.cpp"),
        r"file_map_status\s+unified_cache_raw_advise_file_segments\s*\([^)]*\)",
    )
    chunk_re = (
        r"const\s+uint64_t\s+chunk\s*=\s*advice\s*==\s*file_advice::WILLNEED\s*\?\s*"
        r"std::max\(\s*page\s*,\s*128\s*\*\s*1024\s*/\s*page\s*\*\s*page\s*\)\s*:\s*~0ull\s*;"
    )
    assert re.search(chunk_re, body), "the WILLNEED chunk is not max(page, 128 KiB rounded to pages)"
    assert re.search(r"done\s*\+=\s*chunk", body), "the advice loop does not step by the chunk"
    assert re.search(r"std::min\(\s*chunk\s*,\s*seg\.length\s*-\s*done\s*\)", body), (
        "a chunk may reach past its segment"
    )


# ---- 11. the front door adds its row through the helper ----

# Registry-wide, rule (A) of test-sycl-runtime-registry-index-source.py keeps every write of g_runtime_alloc_registry
# inside the helpers that keep its containment index and settle counters in step. This clause pins only the front door.


def front_door_adds_its_row_through_the_helper(body: str) -> bool:
    claim = re.search(r"\bruntime_registry_claim_ptr_locked\s*\(", body)
    emplace = re.search(r"\bruntime_registry_emplace_locked\s*\(\s*mapping\.base\s*,", body)
    return bool(claim and emplace and claim.start() < emplace.start())


def test_the_front_door_adds_its_row_through_the_emplace_helper():
    assert front_door_adds_its_row_through_the_helper(front_door_body()), (
        "unified_allocate_file_mapping does not add its row through runtime_registry_emplace_locked after the claim"
    )


FIXED_ROW = "registered = runtime_registry_emplace_locked(mapping.base, rec).second;"
DIRECT_ROW = "registered = g_runtime_alloc_registry.emplace(mapping.base, rec).second;"


def test_the_direct_emplace_fails_the_front_door_clause():
    # Positive control: the map holds the row and the index never sees it.
    text = code("unified-cache.cpp")
    assert text.count(FIXED_ROW) == 1, "the front door's row line moved (the control lost its anchor)"
    mutant = text.replace(FIXED_ROW, DIRECT_ROW)
    assert not front_door_adds_its_row_through_the_helper(function_body(mutant, FRONT_DOOR_SIG))


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))

