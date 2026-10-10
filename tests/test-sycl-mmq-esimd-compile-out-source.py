"""Source contract for llama.cpp-xtvl: the opt-in ESIMD Q4_0 mmq kernels are compiled out by default.

Since the 2026-10-10 compute-runtime upgrade (ocloc 26.35.39758.11 / IGC 2.41.9) ocloc segfaults on the
device image holding mmq_esimd_q4_0_kernel and mmq_esimd_q4_0_v3_kernel, which fails every ggml-sycl device
link. The kernels are opt-in (GGML_SYCL_MMQ_ESIMD=<n>, AOS only), so they are built only when the CMake
option GGML_SYCL_MMQ_ESIMD is ON, which defines GGML_SYCL_MMQ_ESIMD_KERNELS.

Host-only: reads sources, runs no build and touches no device. Each check runs against comment-stripped
text, and each has a mutant test proving the check fires on the thing it forbids.
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


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def split_regions(header):
    """Return (guarded, stub) code of the header: text between '#if ... SYCL_ESIMD_MMQ_AVAILABLE'
    machinery. The guarded region runs from the availability #if to the matching '#else'."""
    code = strip_comments(header)
    m = re.search(r"^#if[^\n]*esimd\.hpp[^\n]*$", code, flags=re.M)
    assert m, "no #if line testing for <sycl/ext/intel/esimd.hpp> found"
    start = m.start()
    e = re.search(r"^#else[^\n]*$", code[start:], flags=re.M)
    assert e, "no #else after the availability #if"
    else_at = start + e.start()
    endif = code.rindex("#endif", 0, code.rindex("#endif"))  # the #endif closing the availability #if
    return code[start:else_at], code[else_at:endif], code[start:else_at].split("\n", 1)[0]


def check_available_define_guarded(header):
    """(a) SYCL_ESIMD_MMQ_AVAILABLE is defined only under GGML_SYCL_MMQ_ESIMD_KERNELS (and the header)."""
    code = strip_comments(header)
    defs = [m.start() for m in re.finditer(r"^\s*#\s*define\s+SYCL_ESIMD_MMQ_AVAILABLE\b", code, flags=re.M)]
    if len(defs) != 1:
        return "expected exactly one #define SYCL_ESIMD_MMQ_AVAILABLE, found %d" % len(defs)
    # the nearest preceding #if line is the condition guarding the define
    ifs = list(re.finditer(r"^#\s*if[^\n]*$", code[: defs[0]], flags=re.M))
    if not ifs:
        return "SYCL_ESIMD_MMQ_AVAILABLE defined outside any #if"
    cond = ifs[-1].group(0)
    if "defined(GGML_SYCL_MMQ_ESIMD_KERNELS)" not in cond:
        return "the #if guarding SYCL_ESIMD_MMQ_AVAILABLE lacks defined(GGML_SYCL_MMQ_ESIMD_KERNELS): " + cond
    if "__has_include(<sycl/ext/intel/esimd.hpp>)" not in cond:
        return "the #if guarding SYCL_ESIMD_MMQ_AVAILABLE lost the esimd.hpp __has_include: " + cond
    return None


def check_kernels_inside_guard(header):
    """(b) every parallel_for<mmq_esimd_*> (and the v2 kernel_class alias) sits inside the guarded region."""
    guarded, stub, _ = split_regions(header)
    code = strip_comments(header)
    total = len(re.findall(r"parallel_for\s*<", code))
    inside = len(re.findall(r"parallel_for\s*<", guarded))
    if total == 0:
        return "no parallel_for<...> launches found at all (header restructured?)"
    if inside != total:
        return "%d of %d parallel_for launches are outside the guarded region" % (total - inside, total)
    if re.search(r"parallel_for\s*<", stub):
        return "a parallel_for launch is present in the stub region"
    return None


def declared(region, name):
    return re.search(r"\binline\s+\w+\s+" + re.escape(name) + r"\s*\(", region) is not None


def check_stub_declares_everything(header, mmq):
    """(c) the stub region declares every esimd symbol mmq.cpp calls, and no real kernel leaks into it."""
    guarded, stub, _ = split_regions(header)
    called = sorted(set(re.findall(r"\b(launch_mmq_q4_0_esimd\w*|mmq_esimd_\w+)\s*\(", strip_comments(mmq))))
    missing = [n for n in called if not declared(stub, n)]
    if missing:
        return "stub region does not declare: " + ", ".join(missing)
    for n in KERNEL_LAUNCHES + HELPERS:
        if not declared(guarded, n):
            return "guarded region does not define %s (stub/real signature pair broken)" % n
    for n in KERNEL_LAUNCHES:
        if not re.search(r"\binline\s+bool\s+" + n + r"\s*\([^)]*\)\s*\{[^}]*return\s+false\s*;", stub, flags=re.S):
            return "stub %s does not return false" % n
    return None


def check_cmake_option(cmake):
    """(d) the option exists, defaults OFF, and defines GGML_SYCL_MMQ_ESIMD_KERNELS on ggml-sycl only when ON."""
    code = re.sub(r"#[^\n]*", "", cmake)
    m = re.search(r"option\s*\(\s*GGML_SYCL_MMQ_ESIMD\s+\"[^\"]*\"\s+(\w+)\s*\)", code)
    if not m:
        return "no option(GGML_SYCL_MMQ_ESIMD \"...\" <default>) found"
    if m.group(1) != "OFF":
        return "GGML_SYCL_MMQ_ESIMD defaults to %s, must be OFF" % m.group(1)
    b = re.search(
        r"if\s*\(\s*GGML_SYCL_MMQ_ESIMD\s*\)(?P<body>(?:(?!endif\s*\().)*)endif\s*\(", code, flags=re.S
    )
    if not b or not re.search(
        r"target_compile_definitions\s*\(\s*ggml-sycl\s+PRIVATE\s+GGML_SYCL_MMQ_ESIMD_KERNELS\s*\)", b.group("body")
    ):
        return "GGML_SYCL_MMQ_ESIMD_KERNELS is not defined on ggml-sycl under if (GGML_SYCL_MMQ_ESIMD)"
    unconditional = re.sub(
        r"if\s*\(\s*GGML_SYCL_MMQ_ESIMD\s*\)(?:(?!endif\s*\().)*endif\s*\(\s*\)", "", code, flags=re.S
    )
    if "GGML_SYCL_MMQ_ESIMD_KERNELS" in unconditional:
        return "GGML_SYCL_MMQ_ESIMD_KERNELS is also defined outside the option's if-block"
    return None


def check_only_mmq_includes_header():
    """The header is included only by mmq.cpp, the one TU the option must reach."""
    users = []
    for f in sorted(SYCL.glob("**/*")):
        if f.suffix in (".cpp", ".hpp", ".h") and f.name != "mmq-esimd.hpp":
            if "mmq-esimd.hpp" in f.read_text(errors="replace"):
                users.append(f.name)
    return None if users == ["mmq.cpp"] else "mmq-esimd.hpp includers changed: %r" % users


def function_body(code, name):
    """Brace-matched body of the (single) definition 'static ... name(...) ... {' in comment-stripped code."""
    m = re.search(r"^static\s+[^\n;]*\b" + re.escape(name) + r"\s*\([^)]*\)[^;{]*\{", code, flags=re.M)
    if not m:
        return None
    depth, i = 1, m.end()
    in_str = None
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
    """The 'requested but compiled out' WARN is emitted once at backend init (ggml_check_sycl), only in a build
    without GGML_SYCL_MMQ_ESIMD_KERNELS, only when the env var parses > 0."""
    code = strip_comments(backend)
    body = function_body(code, "ggml_check_sycl")
    if body is None:
        return "ggml_check_sycl definition not found"
    blocks = re.findall(r"#ifndef\s+GGML_SYCL_MMQ_ESIMD_KERNELS\b(.*?)#endif", body, flags=re.S)
    if len(blocks) != 1:
        return "expected exactly one #ifndef GGML_SYCL_MMQ_ESIMD_KERNELS block in ggml_check_sycl, found %d" % len(blocks)
    blk = blocks[0]
    if 'getenv("GGML_SYCL_MMQ_ESIMD")' not in blk or "atoi" not in blk or "> 0" not in blk:
        return "the block does not gate on getenv(GGML_SYCL_MMQ_ESIMD) parsing > 0"
    if not re.search(r"GGML_LOG_WARN\s*\(", blk) or "compiled out" not in blk:
        return "the block does not GGML_LOG_WARN that the kernels are compiled out"
    if body.count("compiled out") != blk.count("compiled out") or len(re.findall(r"ESIMD mmq kernels are compiled out", code)) != 1:
        return "the compiled-out warning also appears outside the guarded block"
    return None


# ---- the real checks -------------------------------------------------------------------------------


def test_available_define_requires_the_cmake_define():
    assert check_available_define_guarded(HEADER) is None, check_available_define_guarded(HEADER)


def test_every_kernel_launch_is_inside_the_guard():
    assert check_kernels_inside_guard(HEADER) is None, check_kernels_inside_guard(HEADER)


def test_stub_declares_every_symbol_mmq_cpp_calls():
    assert check_stub_declares_everything(HEADER, MMQ) is None, check_stub_declares_everything(HEADER, MMQ)


def test_cmake_option_defaults_off_and_defines_the_kernels_macro():
    assert check_cmake_option(CMAKE) is None, check_cmake_option(CMAKE)


def test_only_mmq_cpp_includes_the_header():
    assert check_only_mmq_includes_header() is None, check_only_mmq_includes_header()


def test_stub_enabled_is_false_and_warns_nowhere():
    _, stub, _ = split_regions(HEADER)
    enabled = re.search(r"inline\s+bool\s+mmq_esimd_enabled\s*\(\s*\)\s*\{.*?\n\}", stub, flags=re.S)
    assert enabled, "stub mmq_esimd_enabled not found"
    assert "return false;" in enabled.group(0)
    # mmq.cpp's dispatch site is not reached by every vehicle: the contract warning lives at init.
    assert "GGML_LOG" not in enabled.group(0)


def test_init_path_warns_when_the_kernels_are_compiled_out():
    assert check_init_warn(BACKEND) is None, check_init_warn(BACKEND)


# ---- mutants: each check must fire on the thing it forbids -------------------------------------------


def test_mutant_define_without_cmake_guard_is_caught():
    mutant = HEADER.replace(" && defined(GGML_SYCL_MMQ_ESIMD_KERNELS)", "", 1)
    assert mutant != HEADER
    assert check_available_define_guarded(mutant) is not None


def test_mutant_kernel_launch_in_stub_is_caught():
    mutant = HEADER.replace("return false;  // ESIMD kernels compiled out, use fallback", "cgh.parallel_for<x>(r, k); return false;", 1)
    assert mutant != HEADER
    assert check_kernels_inside_guard(mutant) is not None


def test_mutant_missing_stub_symbol_is_caught():
    for name in ("launch_mmq_q4_0_esimd_v2", "launch_mmq_q4_0_esimd_v3", "mmq_esimd_version"):
        # rename only the stub's definition (the last one in the file)
        idx = HEADER.rindex("inline ", 0, HEADER.rindex(name + "("))
        mutant = HEADER[:idx] + HEADER[idx:].replace(name + "(", name + "_renamed(", 1)
        assert mutant != HEADER
        assert check_stub_declares_everything(mutant, MMQ) is not None, name


def test_mutant_new_mmq_call_without_stub_is_caught():
    mutant_mmq = MMQ + "\nvoid f() { mmq_esimd_brand_new_entry(); }\n"
    assert check_stub_declares_everything(HEADER, mutant_mmq) is not None


def test_mutant_option_default_on_is_caught():
    mutant = re.sub(r'(option\(GGML_SYCL_MMQ_ESIMD "[^"]*" )OFF\)', r"\1ON)", CMAKE, count=1)
    assert mutant != CMAKE
    assert check_cmake_option(mutant) is not None


def test_mutant_unconditional_define_is_caught():
    mutant = CMAKE + "\ntarget_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_MMQ_ESIMD_KERNELS)\n"
    assert check_cmake_option(mutant) is not None


def test_mutant_missing_option_is_caught():
    mutant = CMAKE.replace("option(GGML_SYCL_MMQ_ESIMD ", "option(GGML_SYCL_MMQ_ESIMD_RENAMED ", 1)
    assert mutant != CMAKE
    assert check_cmake_option(mutant) is not None


def test_mutant_init_warn_unconditional_is_caught():
    mutant = BACKEND.replace("#ifndef GGML_SYCL_MMQ_ESIMD_KERNELS\n", "#if 1\n", 1)
    assert mutant != BACKEND
    assert check_init_warn(mutant) is not None


def test_mutant_init_warn_removed_is_caught():
    mutant = BACKEND.replace("ESIMD mmq kernels are compiled out", "ESIMD mmq kernels are gone", 1)
    assert mutant != BACKEND
    assert check_init_warn(mutant) is not None


def test_mutant_init_warn_moved_out_of_init_is_caught():
    code = BACKEND
    i = code.index("#ifndef GGML_SYCL_MMQ_ESIMD_KERNELS")
    j = code.index("#endif", i) + len("#endif")
    block = code[i:j]
    moved = code[:i] + code[j:]
    # re-insert the block into a different function: right after the first function that is not ggml_check_sycl
    k = moved.index("static void ggml_sycl_pipeline_cpu_warn_retired") if "static void ggml_sycl_pipeline_cpu_warn_retired" in moved else -1
    if k < 0:
        k = moved.index("\nstatic ", moved.index("static void ggml_check_sycl()") + 10)
    k = moved.index("{", k) + 1
    mutant = moved[:k] + "\n" + block + "\n" + moved[k:]
    assert mutant != BACKEND
    assert check_init_warn(mutant) is not None


def test_mutant_init_warn_without_env_gate_is_caught():
    mutant = BACKEND.replace('getenv("GGML_SYCL_MMQ_ESIMD")', 'getenv("GGML_SYCL_MMQ_OTHER")', 1)
    assert mutant != BACKEND
    assert check_init_warn(mutant) is not None
