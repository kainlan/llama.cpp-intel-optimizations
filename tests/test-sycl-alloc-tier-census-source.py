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
    assert seen == 10, f"the census found {seen} switches that name FILE_MAPPED, expected exactly 10"
    assert not missing, "switches that name FILE_MAPPED and not HOST_CPU: " + ", ".join(missing)


def test_a_host_cpu_pointer_is_never_resolved_for_a_device():
    code = sycl_sources()["mem-handle.cpp"]
    body = function_body(code, r"resolved_ptr\s+mem_handle::resolve\s*\(\s*int\s+device_id\s*\)\s*const")
    # the two CPU-only tiers share one refusal, not two copies of it
    assert re.search(
        r"alloc_tier\s*::\s*FILE_MAPPED\s*\|\|\s*\w+\s*==\s*alloc_tier\s*::\s*HOST_CPU\s*\)"
        r"\s*\{\s*return\s*\{\s*\}\s*;",
        body,
    ), "resolve(device_id) hands out a HOST_CPU pointer, or refuses it apart from FILE_MAPPED"


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


# ---------------------------------------------------------------------------------------------------------------------
# llama.cpp-th32 T1b: the HOST_CPU front door and its release
# ---------------------------------------------------------------------------------------------------------------------


def test_select_tier_maps_must_host_cpu_and_a_cohort_cannot_inherit_it():
    body = function_body(sources_text("unified-cache.cpp"), r"alloc_tier\s+unified_select_tier\s*\(")
    assert re.search(r"must_host_cpu\s*\)\s*\{\s*return\s+alloc_tier\s*::\s*HOST_CPU\s*;", body), (
        "must_host_cpu is not mapped to HOST_CPU"
    )
    cohort = body[body.index("prefer_same_tier_as_cohort") :]
    assert re.search(r"alloc_tier\s*::\s*HOST_CPU\b", cohort), "the cohort branch does not skip a HOST_CPU entry"
    # nothing falls INTO the tier: the size fallback and the default stay where they were
    named = body.count("alloc_tier::HOST_CPU")
    assert named == 2, f"{named} names of HOST_CPU in selection"


def test_unified_alloc_serves_host_cpu_through_the_one_wrapper():
    code = sources_text("unified-cache.cpp")
    body = function_body(code, r"bool\s+unified_alloc\s*\(\s*const\s+alloc_request\s*&\s+req_in,")
    assert len(re.findall(r"\bunified_cache_raw_alloc_host_cpu\s*\(", body)) == 1
    assert re.search(r"register_alloc\([^;]*alloc_type\s*::\s*HOST_CPU\s*\)", body), "no pointer-type registration"
    assert re.search(r"must_host_cpu\s*&&\s*\([^)]*must_device[^)]*must_host_pinned", body), (
        "unified_alloc does not refuse must_host_cpu beside a routing constraint"
    )
    # the front door is the only caller of the wrapper outside its own TU
    callers = []
    for rel, text in sycl_sources().items():
        if rel in ("host-cpu-mapping.cpp", "host-cpu-mapping.hpp"):
            continue
        for name in ("unified_cache_raw_alloc_host_cpu", "unified_cache_raw_free_host_cpu"):
            callers += [f"{rel}:{name}" for _ in re.finditer(rf"\b{name}\s*\(", text)]
    assert sorted(callers) == [
        "unified-cache.cpp:unified_cache_raw_alloc_host_cpu",
        "unified-cache.cpp:unified_cache_raw_free_host_cpu",
    ], callers


# ---- M1: no lock across the mapping ----


def blank_literals(code: str) -> str:
    """The text with string and character literal contents blanked, so a brace inside one is not a scope."""
    return re.sub(
        r"\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'",
        lambda m: m.group(0)[0] + " " * (len(m.group(0)) - 2) + m.group(0)[0],
        code,
    )


LOCK_DECL = re.compile(
    r"\b(?:std\s*::\s*)?(?:lock_guard|unique_lock|scoped_lock|shared_lock)\s*(?:<[^>]*>)?\s*\w+\s*[({]"
    r"|\bmem_handle_lock_guard\s+\w+\s*[({]"
    r"|\.\s*lock\s*\(\s*\)"
)


def locks_open_at(code: str, pos: int) -> list:
    """The lock declarations whose enclosing scope is still open at `pos` (a lock in a block that has closed is not)."""
    scopes = [[]]
    decls = sorted(m.start() for m in LOCK_DECL.finditer(code[:pos]))
    nxt = 0
    for i in range(pos):
        while nxt < len(decls) and decls[nxt] == i:
            scopes[-1].append(code[i : i + 60].split("\n")[0])
            nxt += 1
        if code[i] == "{":
            scopes.append([])
        elif code[i] == "}" and len(scopes) > 1:
            scopes.pop()
    return [d for scope in scopes for d in scope]


def test_no_lock_is_held_across_the_host_cpu_mapping():
    body = blank_literals(
        function_body(
            sources_text("unified-cache.cpp"), r"bool\s+unified_alloc\s*\(\s*const\s+alloc_request\s*&\s+req_in,"
        )
    )
    call = [m.start() for m in re.finditer(r"\bunified_cache_raw_alloc_host_cpu\s*\(", body)]
    assert len(call) == 1
    held = locks_open_at(body, call[0])
    assert not held, f"a lock is held across the mmap/mlock of a HOST_CPU request: {held}"


def test_the_lock_scope_reader_sees_an_open_lock_and_ignores_a_closed_one():
    closed = "void f() { { std::lock_guard<std::mutex> l(m); x(); } work(); }"
    assert locks_open_at(closed, closed.index("work")) == []
    open_ = "void f() { std::lock_guard<std::mutex> l(m); { y(); } work(); }"
    assert len(locks_open_at(open_, open_.index("work"))) == 1
    nested = "void f() { if (a) { std::unique_lock<std::mutex> l(m); work(); } }"
    assert len(locks_open_at(nested, nested.index("work"))) == 1
    manual = "void f() { m.lock(); work(); }"
    assert len(locks_open_at(manual, manual.index("work"))) == 1


# ---- M2: a HOST_CPU request cannot ask for an alignment the mapping does not give ----


def test_host_cpu_refuses_an_alignment_beyond_a_page_at_both_entries():
    hpp = strip_comments((SYCL_DIR / "host-cpu-mapping.hpp").read_text())
    assert re.search(r"HOST_CPU_MAX_ALIGNMENT\s*=\s*4096\b", hpp), "no HOST_CPU alignment cap of one 4 KiB page"
    # the check stands alone as a condition, so nothing short-circuits it
    sites = (
        (
            "unified_alloc",
            r"bool\s+unified_alloc\s*\(\s*const\s+alloc_request\s*&\s+req_in,",
            r"\bif\s*\(\s*[\w.]*must_host_cpu\s*&&\s*[\w.]*\balignment\s*>\s*HOST_CPU_MAX_ALIGNMENT\s*\)",
        ),
        (
            "the owner entry",
            r"static\s+allocation_result\s+unified_allocate_owner_impl\s*\(",
            r"\(\s*[\w.]*must_host_cpu\s*&&\s*\([^;{]*?\|\|\s*[\w.]*\balignment\s*>\s*"
            r"HOST_CPU_MAX_ALIGNMENT\s*\)\s*\)",
        ),
    )
    cache = sources_text("unified-cache.cpp")
    for name, header, pattern in sites:
        assert re.search(pattern, function_body(cache, header)), (
            f"{name} does not refuse a HOST_CPU alignment beyond the cap"
        )


def test_the_owner_entry_refuses_the_combination_and_keeps_the_class_exact():
    body = function_body(
        sources_text("unified-cache.cpp"), r"static\s+allocation_result\s+unified_allocate_owner_impl\s*\("
    )
    assert re.search(r"must_host_cpu\s*&&\s*\(\s*[\w.]*must_device\s*\|\|\s*[\w.]*must_host_pinned", body), (
        "must_host_cpu is not refused beside a routing constraint"
    )
    suballoc = re.search(r"const\s+bool\s+cache_suballocation\s*=(.*?);", body, flags=re.DOTALL)
    assert suballoc and re.search(r"!\s*req\.intent\.constraints\.must_host_cpu", suballoc.group(1)), (
        "a role or constraint can still make a HOST_CPU owner a cache suballocation"
    )


def test_the_free_branch_releases_a_host_cpu_record_once_and_first():
    body = function_body(sources_text("unified-cache.cpp"), r"static\s+bool\s+unified_free_record\s*\(")
    branch = re.search(r"if\s*\(\s*rec\.handle\.tier\s*==\s*alloc_tier\s*::\s*HOST_CPU\s*\)\s*\{", body)
    assert branch, "unified_free_record has no HOST_CPU branch"
    unregistered = body.index("alloc_tier::UNREGISTERED")
    assert branch.start() < unregistered, "the HOST_CPU branch is not ahead of the UNREGISTERED no-free case"
    inner = body[branch.end() : matching_brace(body, branch.end() - 1)]
    assert len(re.findall(r"\bunified_cache_raw_free_host_cpu\s*\(", inner)) == 1
    assert re.search(r"host_cpu_mapped_size\s*\(\s*rec\.handle\.size\s*\)", inner), "the release is sized another way"
    assert re.search(r"unregister_alloc\s*\(\s*rec\.handle\.ptr\s*\)", inner), "the pointer-type record survives"
    assert "return true" in inner


def test_the_registry_sums_report_host_cpu_apart_from_the_host_sum():
    body = function_body(
        sources_text("unified-cache.cpp"), r"static\s+registry_tier_sums\s+unified_alloc_registry_tier_sums\s*\("
    )
    host = re.search(r"tier\s*==\s*alloc_tier\s*::\s*HOST_PINNED[^{]*\{\s*sums\.host\s*\+=", body)
    assert host and "HOST_CPU" not in host.group(0), "HOST_CPU is summed into the host-pinned ledger comparison"
    assert re.search(r"alloc_tier\s*::\s*HOST_CPU\s*\)\s*\{\s*sums\.host_cpu\s*\+=\s*h\.size\s*;\s*\}", body), (
        "the registry does not sum HOST_CPU bytes on their own"
    )


# ---------------------------------------------------------------------------------------------------------------------
# llama.cpp-th32 T1b: the unconditional role-bytes read accessor and its export
# ---------------------------------------------------------------------------------------------------------------------


def under_private_testing(code: str, pos: int) -> bool:
    """Is `pos` inside a preprocessor conditional whose opening line names GGML_SYCL_PRIVATE_TESTING? An #else or
    #elif arm counts as inside, because it is compiled only for one value of the same macro."""
    stack = []
    for m in re.finditer(r"^[ \t]*#[ \t]*(ifdef|ifndef|if|endif)\b([^\n]*)", code[:pos], flags=re.MULTILINE):
        if m.group(1) == "endif":
            stack.pop()
        else:
            stack.append("GGML_SYCL_PRIVATE_TESTING" in m.group(2))
    return any(stack)


def declaration_at(code: str, regex: str) -> int:
    hits = [m.start() for m in re.finditer(regex, code)]
    assert len(hits) == 1, f"{regex!r} matched {len(hits)} times"
    return hits[0]


ROLE_BYTES_CXX = (
    r"\bbool\s+unified_cache_role_bytes_for_testing\s*\(\s*int\s+device,\s*alloc_role\s+role,"
    r"\s*uint64_t\s*\*\s*out\s*\)\s*noexcept\s*"
)
ROLE_BYTES_C = (
    r"\bbool\s+ggml_backend_sycl_role_bytes_for_testing\s*\(\s*int\s+device,\s*int\s+role,"
    r"\s*uint64_t\s*\*\s*out\s*\)\s*"
)


def test_the_role_bytes_accessor_is_unconditional_and_declared_beside_the_census_accessors():
    hpp = sources_text("unified-cache.hpp")
    decl = declaration_at(hpp, ROLE_BYTES_CXX + ";")
    assert not under_private_testing(hpp, decl), "the accessor is declared only under GGML_SYCL_PRIVATE_TESTING"
    census = declaration_at(hpp, r"\bbool\s+unified_cache_shutdown_allocation_control_census_for_test\s*\(")
    assert 0 < decl - census < 2000, "the accessor is not declared beside the shutdown census accessors"
    cpp = sources_text("unified-cache.cpp")
    define = declaration_at(cpp, ROLE_BYTES_CXX + r"\{")
    assert not under_private_testing(cpp, define), "the accessor is defined only under GGML_SYCL_PRIVATE_TESTING"
    body = function_body(cpp, r"\bbool\s+unified_cache_role_bytes_for_testing\s*\(")
    assert re.search(r"g_device_caches\.end\(\)[^{]*\{\s*return\s+false\s*;", body), (
        "the reading is not VOID for a device with no cache"
    )
    assert "g_runtime_alloc_registry" in body, "the accessor does not read the live registry"
    assert re.search(r"handle\.role\s*==\s*role|\.role\s*==\s*role", body), "the sum is not by role"
    assert re.search(r"handle\.device\s*==\s*device|\.device\s*==\s*device", body), "the sum is not by device"


def test_the_role_bytes_export_resolves_through_the_proc_address_table():
    header = sources_text("../../include/ggml-sycl.h")
    decl = declaration_at(header, r"GGML_BACKEND_API\s+" + ROLE_BYTES_C + ";")
    assert not under_private_testing(header, decl), "the export is declared only under GGML_SYCL_PRIVATE_TESTING"
    backend = sources_text("ggml-sycl.cpp")
    define = declaration_at(backend, r"extern\s+\"C\"\s+" + ROLE_BYTES_C + r"\{")
    assert not under_private_testing(backend, define), "the export is defined only under GGML_SYCL_PRIVATE_TESTING"
    body = function_body(backend, r"extern\s+\"C\"\s+bool\s+ggml_backend_sycl_role_bytes_for_testing\s*\(")
    assert "unified_cache_role_bytes_for_testing" in body and "alloc_role::LAZY_OUTPUT" in body, (
        "the wrapper does not range-check the role before it casts it"
    )
    entry = declaration_at(backend, r'strcmp\(name,\s*"ggml_backend_sycl_role_bytes_for_testing"\)\s*==\s*0')
    assert not under_private_testing(backend, entry), "the proc-address entry is only under GGML_SYCL_PRIVATE_TESTING"
    assert "return (void *) ggml_backend_sycl_role_bytes_for_testing;" in backend[entry : entry + 200]


def test_the_private_testing_reader_sees_a_conditional_and_an_unconditional_site():
    code = strip_comments(
        "int a;\n#if defined(GGML_SYCL_PRIVATE_TESTING)\nint b;\n#else\nint c;\n#endif\nint d;\n#if 1\nint e;\n#endif\n"
    )
    seen = {n: under_private_testing(code, code.index(f"int {n};")) for n in "abcde"}
    assert seen == {"a": False, "b": True, "c": True, "d": False, "e": False}, seen


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
