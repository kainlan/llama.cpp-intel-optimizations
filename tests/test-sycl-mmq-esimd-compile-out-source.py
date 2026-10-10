"""Source contract for llama.cpp-xtvl: the opt-in ESIMD Q4_0 mmq kernels are compiled out by default.

Since the 2026-10-10 compute-runtime upgrade (ocloc 26.35.39758.11 / IGC 2.41.9) ocloc segfaults on the
device image holding mmq_esimd_q4_0_kernel and mmq_esimd_q4_0_v3_kernel, which fails every ggml-sycl device
link. The kernels are opt-in (GGML_SYCL_MMQ_ESIMD=<n>, AOS only), so they are built only when the CMake
option GGML_SYCL_MMQ_ESIMD is ON, which defines GGML_SYCL_MMQ_ESIMD_KERNELS. A build without them says so
once at backend init (ggml_check_sycl), because the mmq.cpp dispatch site is not reached by every vehicle.

Host-only: reads sources, runs no build and touches no device. Each check runs against comment-stripped
text and returns a reason string (None = the claim holds). Each mutant test asserts the SPECIFIC reason its
mutation must produce, so an unrelated failure cannot satisfy it.
"""

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"
HEADER = (SYCL / "mmq-esimd.hpp").read_text()
CMAKE = (SYCL / "CMakeLists.txt").read_text()
MMQ = (SYCL / "mmq.cpp").read_text()
BACKEND = (SYCL / "ggml-sycl.cpp").read_text()

KERNEL_LAUNCHES = ("launch_mmq_q4_0_esimd", "launch_mmq_q4_0_esimd_v2", "launch_mmq_q4_0_esimd_v3")
HELPERS = ("mmq_esimd_enabled", "mmq_esimd_available", "mmq_esimd_version", "mmq_esimd_supported")

EXPECTED_IF = "#if __has_include(<sycl/ext/intel/esimd.hpp>) && defined(GGML_SYCL_MMQ_ESIMD_KERNELS)"

# Files outside the ggml-sycl build that mention the header. The kernel-bench tier2 file is dead code: no
# CMake file names it and no file(GLOB) in an enclosing directory could pick it up (checked below), so it
# is not a compiled TU of any target.
KNOWN_EXTERNAL_INCLUDERS = {"tools/sycl-kernel-bench/kernels/tier2/mmvq_esimd_v2.cpp"}

IF_KINDS = ("if", "ifdef", "ifndef")


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def norm(text):
    return re.sub(r"\s+", " ", text).strip()


def directives(code):
    """(start, kind, normalised line) of every preprocessor directive, in order."""
    out = []
    for m in re.finditer(r"^[ \t]*#[ \t]*(\w+)[^\n]*$", code, flags=re.M):
        line = re.sub(r"^#\s*(\w+)", r"#\1", norm(m.group(0)))
        out.append((m.start(), m.group(1), line))
    return out


def frames_at(code, pos):
    """The stack of open conditionals at `pos`: [directive line, 'if' | 'else'] per frame, outermost first."""
    stack = []
    for start, kind, line in directives(code):
        if start >= pos:
            break
        if kind in IF_KINDS:
            stack.append([line, "if"])
        elif kind in ("else", "elif") and stack:
            stack[-1][1] = "else"
        elif kind == "endif" and stack:
            stack.pop()
    return stack


def split_regions(header):
    """(guarded, stub) code of the header: the availability #if to its matching #else, and that #else to
    its matching #endif. Nested conditionals inside either branch are skipped by depth."""
    code = strip_comments(header)
    ds = directives(code)
    first = next(i for i, d in enumerate(ds) if d[1] == "if" and "esimd.hpp" in d[2])
    start, depth, else_at = ds[first][0], 1, None
    for pos, kind, _ in ds[first + 1 :]:
        if kind in IF_KINDS:
            depth += 1
        elif kind in ("else", "elif") and depth == 1 and else_at is None:
            else_at = pos
        elif kind == "endif":
            depth -= 1
            if depth == 0:
                assert else_at is not None, "no #else for the availability #if"
                return code[start:else_at], code[else_at:pos]
    raise AssertionError("the availability #if is never closed")


def signature(region, name):
    """Normalised 'ret name(params)' of the inline definition of `name` in region, or None."""
    m = re.search(r"\binline\s+(\w+)\s+" + re.escape(name) + r"\s*\(([^)]*)\)\s*\{", region)
    if not m:
        return None
    params = re.sub(r"\s*,\s*", ", ", norm(m.group(2)))
    return "%s %s(%s)" % (m.group(1), name, params)


def match_brace(code, open_idx):
    """Index just past the '}' matching the '{' at open_idx (string and char literals skipped)."""
    depth, i, in_str = 1, open_idx + 1, None
    while i < len(code) and depth:
        ch = code[i]
        if in_str:
            if ch == "\\":
                i += 1
            elif ch == in_str:
                in_str = None
        elif ch in "\"'":
            in_str = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    return i


def function_body(code, name):
    """(body, offset) of the 'static ... name(...) {' definition in comment-stripped code."""
    m = re.search(r"^static\s+[^\n;]*\b" + re.escape(name) + r"\s*\([^)]*\)[^;{]*\{", code, flags=re.M)
    if not m:
        return None
    return code[m.end() : match_brace(code, m.end() - 1) - 1]


# ---- checks (each returns a reason string, or None when the claim holds) ---------------------------------

GUARD_REASON = "the define is not under exactly the esimd.hpp include AND the cmake define"


def check_available_define_guarded(header):
    """(a) SYCL_ESIMD_MMQ_AVAILABLE is defined in the TRUE branch of exactly EXPECTED_IF, with no #else or
    #endif between that condition and the define."""
    code = strip_comments(header)
    defs = [m.start() for m in re.finditer(r"^[ \t]*#[ \t]*define\s+SYCL_ESIMD_MMQ_AVAILABLE\b", code, flags=re.M)]
    if len(defs) != 1:
        return "expected exactly one #define SYCL_ESIMD_MMQ_AVAILABLE, found %d" % len(defs)
    frames = frames_at(code, defs[0])
    if not frames or frames[-1][0] != EXPECTED_IF:
        return "%s: innermost conditional is %r" % (GUARD_REASON, frames[-1][0] if frames else None)
    if frames[-1][1] != "if":
        return "SYCL_ESIMD_MMQ_AVAILABLE is defined in the #else branch of the guard, not its true branch"
    return None


def check_kernels_inside_guard(header):
    """(b) every parallel_for sits inside the guarded region, none in the stub region."""
    guarded, stub = split_regions(header)
    total = len(re.findall(r"parallel_for\s*<", strip_comments(header)))
    if total == 0:
        return "no parallel_for<...> launches found at all (header restructured?)"
    if re.search(r"parallel_for\s*<", stub):
        return "a parallel_for launch is present in the stub region"
    if len(re.findall(r"parallel_for\s*<", guarded)) != total:
        return "some parallel_for launches are outside the guarded region"
    return None


def check_stub_matches_real(header, mmq):
    """(c) the stub region declares every esimd symbol mmq.cpp calls, with the SAME signature as the real one."""
    guarded, stub = split_regions(header)
    called = sorted(set(re.findall(r"\b(launch_mmq_q4_0_esimd\w*|mmq_esimd_\w+)\s*\(", strip_comments(mmq))))
    missing = [n for n in called if signature(stub, n) is None]
    if missing:
        return "stub region does not declare: " + ", ".join(missing)
    for n in KERNEL_LAUNCHES + HELPERS:
        real, stubbed = signature(guarded, n), signature(stub, n)
        if real is None:
            return "guarded region does not define %s" % n
        if stubbed is None:
            return "stub region does not declare %s" % n
        if real != stubbed:
            return "stub signature differs from the real one for %s: [%s] vs [%s]" % (n, stubbed, real)
    for n in KERNEL_LAUNCHES:
        if not re.search(r"\binline\s+bool\s+" + n + r"\s*\([^)]*\)\s*\{[^}]*return\s+false\s*;", stub, flags=re.S):
            return "stub %s does not return false" % n
    en = re.search(r"inline\s+bool\s+mmq_esimd_enabled\s*\(\s*\)\s*\{(.*?)\n\}", stub, flags=re.S)
    if not en or "return false;" not in en.group(1):
        return "stub mmq_esimd_enabled does not return false"
    if "GGML_LOG" in en.group(1) or "static" in en.group(1):
        return "stub mmq_esimd_enabled must not log or hold static state (the warning lives at init)"
    return None


def check_cmake_option(cmake):
    """(d) the option exists, defaults OFF, and defines GGML_SYCL_MMQ_ESIMD_KERNELS on ggml-sycl only when ON."""
    code = re.sub(r"#[^\n]*", "", cmake)
    m = re.search(r"option\s*\(\s*GGML_SYCL_MMQ_ESIMD\s+\"[^\"]*\"\s+(\w+)\s*\)", code)
    if not m:
        return "no option(GGML_SYCL_MMQ_ESIMD) with a default found"
    if m.group(1) != "OFF":
        return "GGML_SYCL_MMQ_ESIMD defaults to %s, must be OFF" % m.group(1)
    block = r"if\s*\(\s*GGML_SYCL_MMQ_ESIMD\s*\)(?P<body>(?:(?!endif\s*\().)*)endif\s*\(\s*\)"
    b = re.search(block, code, flags=re.S)
    if not b or not re.search(
        r"target_compile_definitions\s*\(\s*ggml-sycl\s+PRIVATE\s+GGML_SYCL_MMQ_ESIMD_KERNELS\s*\)", b.group("body")
    ):
        return "the kernels macro is not defined on ggml-sycl under if (GGML_SYCL_MMQ_ESIMD)"
    if "GGML_SYCL_MMQ_ESIMD_KERNELS" in re.sub(block, "", code, flags=re.S):
        return "the kernels macro is also defined outside the option's if-block"
    return None


def check_init_warn(backend):
    """(e) ggml_check_sycl warns once per process, in a build WITHOUT the kernels, only when the value of
    GGML_SYCL_MMQ_ESIMD (the getenv result) parses > 0, and the message names the option and says the kernels
    are compiled out."""
    code = strip_comments(backend)
    body = function_body(code, "ggml_check_sycl")
    if body is None:
        return "ggml_check_sycl definition not found"
    ifn = re.search(r"#ifndef\s+GGML_SYCL_MMQ_ESIMD_KERNELS\b(.*?)#endif", body, flags=re.S)
    if len(re.findall(r"#ifndef\s+GGML_SYCL_MMQ_ESIMD_KERNELS\b", body)) != 1 or not ifn:
        return "expected exactly one #ifndef GGML_SYCL_MMQ_ESIMD_KERNELS block in ggml_check_sycl, found %d" % len(
            re.findall(r"#ifndef\s+GGML_SYCL_MMQ_ESIMD_KERNELS\b", body)
        )
    init = re.search(r"\bif\s*\(\s*!\s*initialized\s*\)\s*\{", body)
    if not init:
        return "ggml_check_sycl has no if (!initialized) body"
    init_end = match_brace(body, init.end() - 1)
    if not (init.end() <= ifn.start() < init_end):
        return "the block is not inside the if (!initialized) body, so it would not warn once per process"
    blk = ifn.group(1)
    g = re.search(r'(\w+)\s*=\s*getenv\("GGML_SYCL_MMQ_ESIMD"\)', blk)
    if not g:
        return "the block does not read getenv(GGML_SYCL_MMQ_ESIMD) into a variable"
    var = g.group(1)
    c = re.search(r"\bif\s*\(\s*%s\s*!=\s*nullptr\s*&&\s*atoi\s*\(\s*%s\s*\)\s*>\s*0\s*\)\s*\{" % (var, var), blk)
    if not c:
        return "the warning condition is not `%s != nullptr && atoi(%s) > 0` on the getenv result" % (var, var)
    cbody = blk[c.end() : match_brace(blk, c.end() - 1) - 1]
    if len(re.findall(r"GGML_LOG_WARN\s*\(", blk)) != 1 or not re.search(r"GGML_LOG_WARN\s*\(", cbody):
        return "the single GGML_LOG_WARN is not inside the body of the value condition"
    w = re.search(r"GGML_LOG_WARN\s*\((?P<args>.*?)\)\s*;", cbody, flags=re.S)
    message = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', w.group("args"))) if w else ""
    if "GGML_SYCL_MMQ_ESIMD" not in message:
        return "the warning message does not name the GGML_SYCL_MMQ_ESIMD option"
    if "compiled out" not in message:
        return "the warning message does not say the kernels are compiled out"
    if len(re.findall(r"ESIMD mmq kernels are compiled out", code)) != 1:
        return "the compiled-out warning must appear exactly once in the backend"
    return None


def repo_includers():
    found = set()
    for top in ("ggml", "src", "common", "tools", "examples", "tests", "include"):
        for f in (ROOT / top).rglob("*"):
            if f.suffix in (".cpp", ".hpp", ".h", ".cu", ".cuh") and f.name != "mmq-esimd.hpp" and f.is_file():
                if "mmq-esimd.hpp" in f.read_text(errors="replace"):
                    found.add(str(f.relative_to(ROOT)))
    return found


def check_includers(found, cmake_files):
    """(f) inside the ggml-sycl build only mmq.cpp includes the header. Any other includer in the repo is a
    known dead file: no CMake file names it, and no file(GLOB) in an enclosing directory could pick it up.
    cmake_files is a list of (repo-relative path, text)."""
    in_sycl = sorted(f for f in found if f.startswith("ggml/src/ggml-sycl/"))
    if in_sycl != ["ggml/src/ggml-sycl/mmq.cpp"]:
        return "includers inside ggml-sycl changed: %r" % in_sycl
    unknown = sorted(f for f in found if not f.startswith("ggml/src/ggml-sycl/") and f not in KNOWN_EXTERNAL_INCLUDERS)
    if unknown:
        return "new repo includer(s) of mmq-esimd.hpp: %r" % unknown
    for dead in sorted(found & KNOWN_EXTERNAL_INCLUDERS):
        stem, dead_dir = Path(dead).name, Path(dead).parent
        for rel, text in cmake_files:
            code = re.sub(r"#[^\n]*", "", text)
            if stem in code:
                return "%s is named by %s, so it may be a compiled TU" % (dead, rel)
            enclosing = Path(rel).parent == dead_dir or Path(rel).parent in dead_dir.parents
            if enclosing and re.search(r"\bfile\s*\(\s*GLOB", code, flags=re.I):
                return "%s uses file(GLOB) over a directory containing %s, so it may be a compiled TU" % (rel, dead)
    return None


def all_cmake_files():
    """(repo-relative path, text) of every CMakeLists.txt and *.cmake, build*/ and .git pruned during the walk."""
    out = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if not d.startswith("build") and d not in (".git", "node_modules")]
        for name in filenames:
            if name == "CMakeLists.txt" or name.endswith(".cmake"):
                p = Path(dirpath) / name
                out.append((str(p.relative_to(ROOT)), p.read_text(errors="replace")))
    return out


# ---- the real checks -------------------------------------------------------------------------------


def test_available_define_requires_the_cmake_define():
    assert check_available_define_guarded(HEADER) is None, check_available_define_guarded(HEADER)


def test_every_kernel_launch_is_inside_the_guard():
    assert check_kernels_inside_guard(HEADER) is None, check_kernels_inside_guard(HEADER)


def test_stub_matches_the_real_signatures_and_every_mmq_cpp_call():
    assert check_stub_matches_real(HEADER, MMQ) is None, check_stub_matches_real(HEADER, MMQ)


def test_cmake_option_defaults_off_and_defines_the_kernels_macro():
    assert check_cmake_option(CMAKE) is None, check_cmake_option(CMAKE)


def test_includers_of_the_header():
    r = check_includers(repo_includers(), all_cmake_files())
    assert r is None, r


def test_init_path_warns_when_the_kernels_are_compiled_out():
    assert check_init_warn(BACKEND) is None, check_init_warn(BACKEND)


# ---- mutants: each check must fire, with the reason its mutation produces ------------------------------


def mutate(text, old, new, count=1):
    out = text.replace(old, new, count)
    assert out != text, "mutation did not apply: %r" % old
    return out


def expect(reason, phrase):
    assert reason is not None, "the check returned None; expected a reason containing %r" % phrase
    assert phrase in reason, "reason %r does not contain %r" % (reason, phrase)


GUARD = " && defined(GGML_SYCL_MMQ_ESIMD_KERNELS)"
DEFINE = "#define SYCL_ESIMD_MMQ_AVAILABLE 1\n"
STUB_ELSE = re.search(r"^#else[^\n]*!SYCL_ESIMD_MMQ_AVAILABLE[^\n]*\n", HEADER, flags=re.M).group(0)
NESTED = "#if 0\nint nested_a;\n#else\nint nested_b;\n#endif\n"


def test_mutant_guard_dropped():
    expect(check_available_define_guarded(mutate(HEADER, GUARD, "")), GUARD_REASON)


def test_mutant_guard_weakened_by_or_true():
    expect(check_available_define_guarded(mutate(HEADER, GUARD + "\n", GUARD + " || 1\n")), GUARD_REASON)


def test_mutant_guard_negated():
    expect(check_available_define_guarded(mutate(HEADER, "&& defined(", "&& !defined(")), GUARD_REASON)


def test_mutant_guard_or_instead_of_and():
    expect(check_available_define_guarded(mutate(HEADER, "&& defined(", "|| defined(")), GUARD_REASON)


def test_mutant_define_moved_into_the_else_branch():
    m = mutate(HEADER, DEFINE, "")
    m = mutate(m, STUB_ELSE, STUB_ELSE + DEFINE)
    expect(check_available_define_guarded(m), "defined in the #else branch")


def test_mutant_endif_between_the_guard_and_the_define():
    m = mutate(HEADER, EXPECTED_IF + "\n", EXPECTED_IF + "\n#endif\n")
    expect(check_available_define_guarded(m), GUARD_REASON)


def test_mutant_define_duplicated():
    expect(check_available_define_guarded(mutate(HEADER, DEFINE, DEFINE + DEFINE)), "found 2")


def test_nested_conditionals_inside_either_branch_keep_the_regions_and_the_checks():
    for anchor in ("namespace esimd = sycl::ext::intel::esimd;\n", STUB_ELSE):
        m = mutate(HEADER, anchor, anchor + NESTED)
        guarded, stub = split_regions(m)
        in_guarded = "nested_b" in guarded
        assert in_guarded == (anchor != STUB_ELSE) and ("nested_b" in stub) == (anchor == STUB_ELSE), anchor
        assert "launch_mmq_q4_0_esimd_v3" in guarded and "launch_mmq_q4_0_esimd_v3" in stub
        assert check_available_define_guarded(m) is None
        assert check_kernels_inside_guard(m) is None
        assert check_stub_matches_real(m, MMQ) is None


def test_mutant_kernel_launch_in_stub():
    m = mutate(HEADER, "return false;  // ESIMD kernels compiled out, use fallback", "cgh.parallel_for<x>(r, k); return false;")
    expect(check_kernels_inside_guard(m), "present in the stub region")


def test_mutant_kernel_launch_in_a_nested_stub_branch_is_still_the_stub():
    m = mutate(HEADER, STUB_ELSE, STUB_ELSE + "#if 0\nvoid f() { cgh.parallel_for<x>(r, k); }\n#endif\n")
    expect(check_kernels_inside_guard(m), "present in the stub region")


def test_mutant_missing_stub_symbol():
    for name in ("launch_mmq_q4_0_esimd_v2", "launch_mmq_q4_0_esimd_v3", "mmq_esimd_version"):
        idx = HEADER.rindex("inline ", 0, HEADER.rindex(name + "("))
        m = HEADER[:idx] + HEADER[idx:].replace(name + "(", name + "_renamed(", 1)
        r = check_stub_matches_real(m, MMQ)
        expect(r, "stub region does not declare")
        expect(r, name)


def test_mutant_stub_signature_drift():
    idx = HEADER.rindex("const int64_t nrows_dst")  # a parameter of the last stub, launch_..._v3
    m = HEADER[:idx] + HEADER[idx:].replace("const int64_t nrows_dst", "const int nrows_dst", 1)
    expect(check_stub_matches_real(m, MMQ), "stub signature differs from the real one for launch_mmq_q4_0_esimd_v3")


def test_mutant_new_mmq_call_without_stub():
    expect(
        check_stub_matches_real(HEADER, MMQ + "\nvoid f() { mmq_esimd_brand_new_entry(); }\n"),
        "stub region does not declare: mmq_esimd_brand_new_entry",
    )


def test_mutant_stub_enabled_warns():
    marker = "inline bool mmq_esimd_enabled() {"
    idx = HEADER.rindex(marker) + len(marker)  # the stub definition is the last one
    r = check_stub_matches_real(HEADER[:idx] + '\n    GGML_LOG_WARN("x");' + HEADER[idx:], MMQ)
    expect(r, "must not log or hold static state")


def test_mutant_stub_enabled_holds_static_state():
    marker = "inline bool mmq_esimd_enabled() {"
    idx = HEADER.rindex(marker) + len(marker)
    r = check_stub_matches_real(HEADER[:idx] + "\n    static int calls = 0;" + HEADER[idx:], MMQ)
    expect(r, "must not log or hold static state")


def test_mutant_option_default_on():
    m = re.sub(r'(option\(GGML_SYCL_MMQ_ESIMD "[^"]*" )OFF\)', r"\1ON)", CMAKE, count=1)
    assert m != CMAKE
    expect(check_cmake_option(m), "defaults to ON")


def test_mutant_unconditional_define():
    m = CMAKE + "\ntarget_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_MMQ_ESIMD_KERNELS)\n"
    expect(check_cmake_option(m), "also defined outside the option's if-block")


def test_mutant_missing_option():
    m = mutate(CMAKE, "option(GGML_SYCL_MMQ_ESIMD ", "option(GGML_SYCL_MMQ_ESIMD_RENAMED ")
    expect(check_cmake_option(m), "no option(GGML_SYCL_MMQ_ESIMD)")


DEAD = "tools/sycl-kernel-bench/kernels/tier2/mmvq_esimd_v2.cpp"


def test_mutant_new_includer_inside_ggml_sycl():
    expect(check_includers(repo_includers() | {"ggml/src/ggml-sycl/fattn.cpp"}, []), "includers inside ggml-sycl changed")


def test_mutant_new_includer_outside_ggml_sycl():
    expect(check_includers(repo_includers() | {"tools/other/new.cpp"}, []), "new repo includer(s)")


def test_mutant_dead_includer_named_by_a_cmakelists():
    files = [("tools/sycl-kernel-bench/CMakeLists.txt", "add_executable(x kernels/tier2/mmvq_esimd_v2.cpp)")]
    expect(check_includers(repo_includers(), files), "tools/sycl-kernel-bench/CMakeLists.txt, so it may be a compiled TU")


def test_mutant_dead_includer_named_by_a_cmake_module():
    files = [("tools/sycl-kernel-bench/sources.cmake", "set(SRCS kernels/tier2/mmvq_esimd_v2.cpp)")]
    expect(check_includers(repo_includers(), files), "tools/sycl-kernel-bench/sources.cmake, so it may be a compiled TU")


def test_mutant_enclosing_directory_globs_the_dead_includer():
    for rel in ("tools/sycl-kernel-bench/CMakeLists.txt", "tools/sycl-kernel-bench/kernels/tier2/CMakeLists.txt"):
        files = [(rel, "file(GLOB K CONFIGURE_DEPENDS kernels/tier2/*.cpp)")]
        expect(check_includers(repo_includers(), files), "uses file(GLOB) over a directory containing " + DEAD)


def test_a_glob_in_an_unrelated_directory_does_not_trip_the_dead_includer_check():
    files = [("tools/ui/CMakeLists.txt", "file(GLOB_RECURSE UI_DIST_FILES dist/*)")]
    assert check_includers(repo_includers(), files) is None


def test_cmake_walk_prunes_build_directories():
    for rel, _ in all_cmake_files():
        assert not any(part.startswith("build") for part in Path(rel).parts[:-1]), rel


INIT_IF = "#ifndef GGML_SYCL_MMQ_ESIMD_KERNELS\n"
INIT_DECL = "    static bool initialized = false;\n"
COND = "atoi(mmq_esimd_env) > 0) {"


def init_block(code=BACKEND):
    i = code.index(INIT_IF)
    return i, code.index("#endif", i) + len("#endif")


def test_mutant_init_warn_unconditional():
    expect(check_init_warn(mutate(BACKEND, INIT_IF, "#if 1\n")), "expected exactly one #ifndef GGML_SYCL_MMQ_ESIMD_KERNELS block")


def test_mutant_init_warn_reworded_to_drop_the_option_name():
    r = check_init_warn(mutate(BACKEND, "GGML_SYCL_MMQ_ESIMD requested but", "the opt-in requested but"))
    expect(r, "does not name the GGML_SYCL_MMQ_ESIMD option")


def test_mutant_init_warn_reworded_to_drop_compiled_out():
    r = check_init_warn(mutate(BACKEND, "ESIMD mmq kernels are compiled out", "ESIMD mmq kernels are gone"))
    expect(r, "does not say the kernels are compiled out")


def test_mutant_init_warn_weakened_positive_value_condition():
    expect(check_init_warn(mutate(BACKEND, "atoi(mmq_esimd_env) > 0", "atoi(mmq_esimd_env) >= 0")), "is not `mmq_esimd_env != nullptr")


def test_mutant_init_warn_condition_on_a_different_variable():
    expect(check_init_warn(mutate(BACKEND, "atoi(mmq_esimd_env) > 0", "atoi(debug_env) > 0")), "on the getenv result")


def test_mutant_init_warn_outside_the_value_condition():
    # the condition guards an empty body and the WARN sits in a bare block that always runs
    expect(check_init_warn(mutate(BACKEND, COND, COND + "} {")), "not inside the body of the value condition")


def test_mutant_init_warn_ignores_the_env_var():
    r = check_init_warn(mutate(BACKEND, 'getenv("GGML_SYCL_MMQ_ESIMD")', 'getenv("GGML_SYCL_MMQ_OTHER")'))
    expect(r, "does not read getenv(GGML_SYCL_MMQ_ESIMD)")


def test_mutant_init_warn_hoisted_above_the_once_guard():
    i, j = init_block()
    block, rest = BACKEND[i:j], BACKEND[:i] + BACKEND[j:]
    m = mutate(rest, INIT_DECL, block + "\n" + INIT_DECL)
    expect(check_init_warn(m), "not inside the if (!initialized) body")


def test_mutant_init_warn_moved_out_of_ggml_check_sycl():
    i, j = init_block()
    block, moved = BACKEND[i:j], BACKEND[:i] + BACKEND[j:]
    start = moved.index("static void ggml_check_sycl()")
    nxt = moved.index("\nstatic ", start + 10)
    k = moved.index("{", nxt) + 1
    m = moved[:k] + "\n" + block + "\n" + moved[k:]
    expect(check_init_warn(m), "expected exactly one #ifndef GGML_SYCL_MMQ_ESIMD_KERNELS block in ggml_check_sycl, found 0")
