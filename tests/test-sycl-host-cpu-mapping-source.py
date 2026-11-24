"""Source contract for llama.cpp-th32 T1a: the HOST_CPU raw wrapper compiles everywhere and releases once.

The runtime tests (test-sycl-host-cpu-mapping on Linux, test-sycl-host-cpu-unsupported with _WIN32 defined) discriminate
on behaviour; this gate pins what they cannot see from outside:

1. host-cpu-mapping.cpp compiles on every platform (the backend globs its sources): its POSIX headers and calls sit
   under `#if !defined(_WIN32)`, the `#else` arm holds refusing stubs for both wrappers, and the sizing function and
   the refusal text are outside the guard (the plan's HOST terms use the sizing function on every platform);
2. host_cpu_supported() is false on Windows and true elsewhere, as a constexpr;
3. the POSIX test is registered only under `if (CMAKE_SYSTEM_NAME STREQUAL "Linux")`, and the unsupported test is
   registered everywhere and compiled with _WIN32 defined;
4. the mapping is released through ONE function, which both the lock-failure unwind and the public release call, and
   that function holds the one munmap and the test counter (a second release added anywhere is a gate failure here and
   in the raw-mapping allowlist gate);
5. mlock is called without MLOCK_ONFAULT and over the whole mapped range (resident == planned from the first moment).

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


def sycl_text(rel: str) -> str:
    return (SYCL / rel).read_text()


def balanced(text: str, open_idx: int) -> str:
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i]
    raise AssertionError("unbalanced braces")


def function_body(text: str, signature_re: str) -> str:
    m = re.search(signature_re, text)
    assert m, f"function not found: {signature_re}"
    return balanced(text, text.index("{", m.end()))


def guard_regions(text: str) -> tuple:
    """(posix, other): the text of the `#if !defined(_WIN32)` arm and of everything else, at the top nesting level."""
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
POSIX_CALLS = r"\b(mmap|munmap|mlock|madvise|getrlimit)\s*\("

CPP = "host-cpu-mapping.cpp"


# ---- 1. the guard ----


def test_the_posix_body_is_guarded_and_the_other_arm_stubs_both_wrappers():
    posix, other = guard_regions(sycl_text(CPP))
    assert posix, f"{CPP} has no `#if !defined(_WIN32)` body"
    assert re.search(POSIX_HEADERS, posix), "the POSIX headers are not inside the guard"
    other_code = strip_comments(other)
    assert not re.search(POSIX_HEADERS, other_code), "a POSIX header is included outside the guard"
    hit = re.search(POSIX_CALLS, other_code)
    assert not hit, f"a POSIX call sits outside the guard: {hit.group(0) if hit else ''}"
    for name in ("unified_cache_raw_alloc_host_cpu", "unified_cache_raw_free_host_cpu"):
        assert name in strip_comments(other), f"the non-POSIX arm has no stub for {name}"
    stub = function_body(
        strip_comments(other), r"host_cpu_status\s+unified_cache_raw_alloc_host_cpu\s*\([^)]*\)"
    )
    assert re.search(r"return\s+host_cpu_status::UNSUPPORTED\s*;", stub), "the stub does not refuse as UNSUPPORTED"
    assert not re.search(r"\bmmap\s*\(", stub), "the stub maps"


def test_the_sizing_function_and_the_refusal_text_are_outside_the_guard():
    posix, other = guard_regions(sycl_text(CPP))
    for ret, name in (
        (r"size_t", "host_cpu_mapped_size"),
        (r"std::string", "host_cpu_refusal_text"),
        (r"const\s+char\s*\*", "host_cpu_status_name"),
    ):
        definition = rf"^{ret}\s+{name}\s*\("
        assert re.search(definition, strip_comments(other), flags=re.MULTILINE), (
            f"{name} is not defined outside the POSIX guard, so Windows loses it"
        )
        assert not re.search(definition, strip_comments(posix), flags=re.MULTILINE), (
            f"{name} is defined inside the POSIX guard"
        )


# ---- 2. the capability query ----


def test_the_capability_query_is_false_on_windows():
    hpp = strip_comments(sycl_text("host-cpu-mapping.hpp"))
    m = re.search(r"constexpr\s+bool\s+host_cpu_supported\s*\(\s*\)\s*\{(.*?)\n\}", hpp, flags=re.DOTALL)
    assert m, "host-cpu-mapping.hpp has no host_cpu_supported()"
    assert re.search(
        r"#\s*if\s+defined\s*\(\s*_WIN32\s*\)\s*return\s+false\s*;\s*#\s*else\s*return\s+true\s*;", m.group(1)
    ), "host_cpu_supported() is not false on Windows and true elsewhere"


# ---- 3. registration ----


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


def test_the_posix_test_is_linux_only_and_the_unsupported_test_is_everywhere():
    reg = cmake_blocks()
    assert "test-sycl-host-cpu-mapping" in reg, "test-sycl-host-cpu-mapping is not registered"
    assert reg["test-sycl-host-cpu-mapping"], "test-sycl-host-cpu-mapping needs POSIX but is registered everywhere"
    assert "test-sycl-host-cpu-unsupported" in reg and not reg["test-sycl-host-cpu-unsupported"], (
        "test-sycl-host-cpu-unsupported is portable and should not be Linux-only"
    )


def test_the_unsupported_test_compiles_the_stub():
    text = (SYCL / "CMakeLists.txt").read_text()
    m = re.search(r"target_compile_definitions\(\s*test-sycl-host-cpu-unsupported\s+PRIVATE\s+([^)]*)\)", text)
    assert m and "_WIN32" in m.group(1), "test-sycl-host-cpu-unsupported is not compiled with _WIN32 defined"


# ---- 4. one release ----


def test_one_function_holds_the_munmap_and_the_counter_and_both_releases_call_it():
    code = strip_comments(sycl_text(CPP))
    posix, _ = guard_regions(sycl_text(CPP))
    posix = strip_comments(posix)
    assert len(re.findall(r"\bmunmap\s*\(", code)) == 1, "munmap is called other than once in the wrapper TU"
    body = function_body(posix, r"static\s+void\s+host_cpu_munmap\s*\(\s*void\s*\*\s*\w+\s*,\s*size_t\s+\w+\s*\)")
    assert re.search(r"\bmunmap\s*\(", body), "the munmap is not inside host_cpu_munmap"
    assert "g_munmap_count" in body, "the test counter does not sit at the munmap"
    free_body = function_body(posix, r"void\s+unified_cache_raw_free_host_cpu\s*\([^)]*\)")
    assert re.search(r"\bhost_cpu_munmap\s*\(", free_body), "the public release does not go through host_cpu_munmap"
    alloc_body = function_body(posix, r"host_cpu_status\s+unified_cache_raw_alloc_host_cpu\s*\([^)]*\)")
    assert re.search(r"\bhost_cpu_munmap\s*\(", alloc_body), (
        "the lock-failure unwind does not go through host_cpu_munmap"
    )


# ---- 5. the lock ----


def test_the_lock_covers_the_whole_mapping_and_is_not_on_fault():
    code = strip_comments(sycl_text(CPP))
    assert "MLOCK_ONFAULT" not in code and "mlock2" not in code, "the lock is lazy: resident would not equal planned"
    posix, _ = guard_regions(sycl_text(CPP))
    alloc_body = function_body(strip_comments(posix), r"host_cpu_status\s+unified_cache_raw_alloc_host_cpu\s*\([^)]*\)")
    assert re.search(r"\bmlock\s*\(\s*p\s*,\s*mapped\s*\)", alloc_body), "mlock is not over the whole mapped range"


def test_the_huge_page_advice_precedes_the_lock_and_is_guarded():
    posix, _ = guard_regions(sycl_text(CPP))
    alloc_body = function_body(strip_comments(posix), r"host_cpu_status\s+unified_cache_raw_alloc_host_cpu\s*\([^)]*\)")
    advise = re.search(r"\bmadvise\s*\(\s*p\s*,\s*mapped\s*,\s*MADV_HUGEPAGE\s*\)", alloc_body)
    lock = re.search(r"\bmlock\s*\(\s*p\s*,\s*mapped\s*\)", alloc_body)
    assert advise and lock, "the wrapper does not advise and lock the whole mapping"
    # mlock faults every page in: advice given after it records VM_HUGEPAGE but leaves small pages
    assert advise.start() < lock.start(), "MADV_HUGEPAGE comes after the mlock that faults the pages in"
    # MADV_HUGEPAGE is Linux-only while the arm is every POSIX: the call is under its own feature test
    before = alloc_body[: advise.start()]
    assert re.search(r"#\s*if\s+defined\s*\(\s*MADV_HUGEPAGE\s*\)\s*(?:\w+\s*->\s*\w+\s*=\s*)?\Z", before) and (
        re.search(r"#\s*endif", alloc_body[advise.end() : lock.start()])
    ), "madvise(MADV_HUGEPAGE) is not under `#if defined(MADV_HUGEPAGE)`"


def test_the_munmap_counter_counts_only_a_munmap_that_succeeded():
    posix, _ = guard_regions(sycl_text(CPP))
    body = function_body(
        strip_comments(posix),
        r"static\s+(?:void|bool)\s+host_cpu_munmap\s*\(\s*void\s*\*\s*\w+\s*,\s*size_t\s+\w+\s*\)",
    )
    m = re.search(r"if\s*\(\s*munmap\s*\([^)]*\)\s*==\s*0\s*\)\s*\{", body)
    assert m, "the counter is not inside `if (munmap(...) == 0)`"
    assert body.index("g_munmap_count") > m.end(), "the counter is bumped outside the success branch of the munmap"


def test_the_refusal_text_prints_errno_only_when_there_is_one():
    code = strip_comments(sycl_text(CPP))
    body = function_body(code, r"std::string\s+host_cpu_refusal_text\s*\(")
    non_lock = body[body.index("else") :]
    assert re.search(r"if\s*\(\s*mapping\.err\s*!=\s*0\s*\)", non_lock), "the non-lock text does not test errno"
    assert "no errno" not in non_lock, "the non-lock text still prints a placeholder for a missing errno"


def test_the_header_states_the_release_contract_and_the_thp_alignment_assumption():
    hpp = sycl_text("host-cpu-mapping.hpp")
    assert re.search(r"exactly once", hpp), "the header does not say the release is exactly once"
    assert re.search(r"6\.7", hpp), "the header does not name the kernel's THP alignment assumption"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
