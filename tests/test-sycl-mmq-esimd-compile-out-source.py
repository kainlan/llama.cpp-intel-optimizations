"""Source contract for llama.cpp-xtvl: the opt-in ESIMD Q4_0 mmq kernels are compiled out by default.

Since the 2026-10-10 compute-runtime upgrade (ocloc 26.35.39758.11 / IGC 2.41.9) ocloc segfaults on the
device image holding mmq_esimd_q4_0_kernel and mmq_esimd_q4_0_v3_kernel, which fails every ggml-sycl device
link. The kernels are opt-in (GGML_SYCL_MMQ_ESIMD=<n>, AOS only), so they are built only when the CMake
option GGML_SYCL_MMQ_ESIMD is ON, which defines GGML_SYCL_MMQ_ESIMD_KERNELS. A build without them says so
once at backend init (ggml_check_sycl), because the mmq.cpp dispatch site is not reached by every vehicle.

Host-only: reads sources, runs no build and touches no device. Each check runs against comment-stripped
text and returns a reason string (None = holds). Each mutant test asserts the SPECIFIC reason its mutation
must produce, so an unrelated failure cannot satisfy it.
"""

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
# add_executable / source list names it (checked below), so it is not a compiled TU of any target.
KNOWN_EXTERNAL_INCLUDERS = {"tools/sycl-kernel-bench/kernels/tier2/mmvq_esimd_v2.cpp"}


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def norm(text):
    return re.sub(r"\s+", " ", text).strip()


def split_regions(header):
    """(guarded, stub) code of the header: the availability #if to its #else, and the #else to its #endif."""
    code = strip_comments(header)
    m = re.search(r"^#if[^\n]*esimd\.hpp[^\n]*$", code, flags=re.M)
    assert m, "no #if line testing for <sycl/ext/intel/esimd.hpp> found"
    start = m.start()
    e = re.search(r"^#else[^\n]*$", code[start:], flags=re.M)
    assert e, "no #else after the availability #if"
    else_at = start + e.start()
    endif = code.rindex("#endif", 0, code.rindex("#endif"))  # the #endif closing the availability #if
    return code[start:else_at], code[else_at:endif]


def signature(region, name):
    """Normalised 'ret name(params)' of the inline definition of `name` in region, or None."""
    m = re.search(r"\binline\s+(\w+)\s+" + re.escape(name) + r"\s*\(([^)]*)\)\s*\{", region)
    if not m:
        return None
    params = re.sub(r"\s*,\s*", ", ", norm(m.group(2)))
    return "%s %s(%s)" % (m.group(1), name, params)


# ---- checks (each returns a reason string, or None when the claim holds) ---------------------------------


def check_available_define_guarded(header):
    """(a) SYCL_ESIMD_MMQ_AVAILABLE is defined only under exactly the esimd-include AND the cmake define."""
    code = strip_comments(header)
    defs = [m.start() for m in re.finditer(r"^\s*#\s*define\s+SYCL_ESIMD_MMQ_AVAILABLE\b", code, flags=re.M)]
    if len(defs) != 1:
        return "expected exactly one #define SYCL_ESIMD_MMQ_AVAILABLE, found %d" % len(defs)
    ifs = list(re.finditer(r"^\s*#\s*if[^\n]*$", code[: defs[0]], flags=re.M))
    if not ifs:
        return "SYCL_ESIMD_MMQ_AVAILABLE defined outside any #if"
    cond = norm(ifs[-1].group(0))
    cond = re.sub(r"^#\s*if", "#if", cond)
    if cond != EXPECTED_IF:
        return "guard condition is not exactly the esimd.hpp include AND the cmake define: " + cond
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


def function_body(code, name):
    """Brace-matched body of the 'static ... name(...) {' definition in comment-stripped code."""
    m = re.search(r"^static\s+[^\n;]*\b" + re.escape(name) + r"\s*\([^)]*\)[^;{]*\{", code, flags=re.M)
    if not m:
        return None
    depth, i, in_str = 1, m.end(), None
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
    return code[m.end() : i]


def check_init_warn(backend):
    """(e) ggml_check_sycl warns once, in a build WITHOUT the kernels, when the env var parses > 0, and the
    message itself names the option and says the kernels are compiled out."""
    code = strip_comments(backend)
    body = function_body(code, "ggml_check_sycl")
    if body is None:
        return "ggml_check_sycl definition not found"
    blocks = re.findall(r"#ifndef\s+GGML_SYCL_MMQ_ESIMD_KERNELS\b(.*?)#endif", body, flags=re.S)
    if len(blocks) != 1:
        return "expected exactly one #ifndef GGML_SYCL_MMQ_ESIMD_KERNELS block in ggml_check_sycl, found %d" % len(blocks)
    blk = blocks[0]
    if 'getenv("GGML_SYCL_MMQ_ESIMD")' not in blk:
        return "the block does not read getenv(GGML_SYCL_MMQ_ESIMD)"
    if not re.search(r"atoi\s*\(\s*\w+\s*\)\s*>\s*0\b", blk):
        return "the block does not require the parsed value to be > 0"
    w = re.search(r"GGML_LOG_WARN\s*\((?P<args>.*?)\)\s*;", blk, flags=re.S)
    if not w:
        return "the block does not GGML_LOG_WARN"
    message = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', w.group("args")))
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


def check_includers(found, cmake_texts):
    """(f) inside the ggml-sycl build only mmq.cpp includes the header; any other includer in the repo is
    a known dead file that no CMake source list names."""
    in_sycl = sorted(f for f in found if f.startswith("ggml/src/ggml-sycl/"))
    if in_sycl != ["ggml/src/ggml-sycl/mmq.cpp"]:
        return "includers inside ggml-sycl changed: %r" % in_sycl
    unknown = sorted(f for f in found if not f.startswith("ggml/src/ggml-sycl/") and f not in KNOWN_EXTERNAL_INCLUDERS)
    if unknown:
        return "new repo includer(s) of mmq-esimd.hpp: %r" % unknown
    for f in sorted(found & KNOWN_EXTERNAL_INCLUDERS):
        stem = Path(f).name
        if any(stem in t for t in cmake_texts):
            return "%s is now named by a CMake file, so it may be a compiled TU" % f
    return None


def all_cmake_texts():
    out = []
    for f in ROOT.rglob("CMakeLists.txt"):
        if "build" not in f.relative_to(ROOT).parts[0]:
            out.append(f.read_text(errors="replace"))
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
    found = repo_includers()
    assert check_includers(found, all_cmake_texts()) is None, check_includers(found, all_cmake_texts())


def test_init_path_warns_when_the_kernels_are_compiled_out():
    assert check_init_warn(BACKEND) is None, check_init_warn(BACKEND)


# ---- mutants: each check must fire, with the reason its mutation produces ------------------------------


def mutate(text, old, new, count=1):
    out = text.replace(old, new, count)
    assert out != text, "mutation did not apply: %r" % old
    return out


GUARD = " && defined(GGML_SYCL_MMQ_ESIMD_KERNELS)"


def test_mutant_guard_dropped():
    assert "exactly the esimd.hpp include AND the cmake define" in check_available_define_guarded(mutate(HEADER, GUARD, ""))


def test_mutant_guard_weakened_by_or_true():
    r = check_available_define_guarded(mutate(HEADER, GUARD + "\n", GUARD + " || 1\n"))
    assert r and "exactly the esimd.hpp include AND the cmake define" in r


def test_mutant_guard_negated():
    r = check_available_define_guarded(mutate(HEADER, "&& defined(", "&& !defined("))
    assert r and "exactly the esimd.hpp include AND the cmake define" in r


def test_mutant_guard_or_instead_of_and():
    r = check_available_define_guarded(mutate(HEADER, "&& defined(", "|| defined("))
    assert r and "exactly the esimd.hpp include AND the cmake define" in r


def test_mutant_kernel_launch_in_stub():
    m = mutate(HEADER, "return false;  // ESIMD kernels compiled out, use fallback", "cgh.parallel_for<x>(r, k); return false;")
    assert "present in the stub region" in check_kernels_inside_guard(m)


def test_mutant_missing_stub_symbol():
    for name in ("launch_mmq_q4_0_esimd_v2", "launch_mmq_q4_0_esimd_v3", "mmq_esimd_version"):
        idx = HEADER.rindex("inline ", 0, HEADER.rindex(name + "("))
        m = HEADER[:idx] + HEADER[idx:].replace(name + "(", name + "_renamed(", 1)
        r = check_stub_matches_real(m, MMQ)
        assert r and "stub region does not declare" in r and name in r, (name, r)


def test_mutant_stub_signature_drift():
    # change a parameter type of the last stub definition (launch_..._v3) only
    idx = HEADER.rindex("const int64_t nrows_dst")
    m = HEADER[:idx] + HEADER[idx:].replace("const int64_t nrows_dst", "const int nrows_dst", 1)
    r = check_stub_matches_real(m, MMQ)
    assert r and "stub signature differs from the real one for launch_mmq_q4_0_esimd_v3" in r, r


def test_mutant_new_mmq_call_without_stub():
    r = check_stub_matches_real(HEADER, MMQ + "\nvoid f() { mmq_esimd_brand_new_entry(); }\n")
    assert r and "stub region does not declare: mmq_esimd_brand_new_entry" in r


def test_mutant_stub_enabled_warns():
    marker = "inline bool mmq_esimd_enabled() {"
    idx = HEADER.rindex(marker) + len(marker)  # the stub definition is the last one
    r = check_stub_matches_real(HEADER[:idx] + '\n    GGML_LOG_WARN("x");' + HEADER[idx:], MMQ)
    assert r and "must not log or hold static state" in r, r


def test_mutant_option_default_on():
    m = re.sub(r'(option\(GGML_SYCL_MMQ_ESIMD "[^"]*" )OFF\)', r"\1ON)", CMAKE, count=1)
    assert m != CMAKE
    assert "defaults to ON" in check_cmake_option(m)


def test_mutant_unconditional_define():
    m = CMAKE + "\ntarget_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_MMQ_ESIMD_KERNELS)\n"
    assert "also defined outside the option's if-block" in check_cmake_option(m)


def test_mutant_missing_option():
    m = mutate(CMAKE, "option(GGML_SYCL_MMQ_ESIMD ", "option(GGML_SYCL_MMQ_ESIMD_RENAMED ")
    assert "no option(GGML_SYCL_MMQ_ESIMD)" in check_cmake_option(m)


def test_mutant_new_includer_inside_ggml_sycl():
    found = repo_includers() | {"ggml/src/ggml-sycl/fattn.cpp"}
    assert "includers inside ggml-sycl changed" in check_includers(found, [])


def test_mutant_new_includer_outside_ggml_sycl():
    found = repo_includers() | {"tools/other/new.cpp"}
    assert "new repo includer(s)" in check_includers(found, [])


def test_mutant_dead_includer_becomes_a_target_source():
    r = check_includers(repo_includers(), ["add_executable(x mmvq_esimd_v2.cpp)"])
    assert r and "may be a compiled TU" in r


INIT_IF = "#ifndef GGML_SYCL_MMQ_ESIMD_KERNELS\n"


def test_mutant_init_warn_unconditional():
    r = check_init_warn(mutate(BACKEND, INIT_IF, "#if 1\n"))
    assert r and "expected exactly one #ifndef GGML_SYCL_MMQ_ESIMD_KERNELS block" in r


def test_mutant_init_warn_reworded_to_drop_the_option_name():
    r = check_init_warn(mutate(BACKEND, "GGML_SYCL_MMQ_ESIMD requested but", "the opt-in requested but"))
    assert r and "does not name the GGML_SYCL_MMQ_ESIMD option" in r, r


def test_mutant_init_warn_reworded_to_drop_compiled_out():
    r = check_init_warn(mutate(BACKEND, "ESIMD mmq kernels are compiled out", "ESIMD mmq kernels are gone"))
    assert r and "does not say the kernels are compiled out" in r, r


def test_mutant_init_warn_drops_the_positive_value_condition():
    r = check_init_warn(mutate(BACKEND, "atoi(mmq_esimd_env) > 0", "atoi(mmq_esimd_env) >= 0"))
    assert r and "> 0" in r, r


def test_mutant_init_warn_ignores_the_env_var():
    r = check_init_warn(mutate(BACKEND, 'getenv("GGML_SYCL_MMQ_ESIMD")', 'getenv("GGML_SYCL_MMQ_OTHER")'))
    assert r and "does not read getenv(GGML_SYCL_MMQ_ESIMD)" in r, r


def test_mutant_init_warn_moved_out_of_ggml_check_sycl():
    i = BACKEND.index(INIT_IF)
    j = BACKEND.index("#endif", i) + len("#endif")
    block, moved = BACKEND[i:j], BACKEND[:i] + BACKEND[j:]
    # re-insert it at the top of the next function after ggml_check_sycl's definition
    start = moved.index("static void ggml_check_sycl()")
    nxt = moved.index("\nstatic ", start + 10)
    k = moved.index("{", nxt) + 1
    m = moved[:k] + "\n" + block + "\n" + moved[k:]
    r = check_init_warn(m)
    assert r and "expected exactly one #ifndef GGML_SYCL_MMQ_ESIMD_KERNELS block in ggml_check_sycl, found 0" in r, r
