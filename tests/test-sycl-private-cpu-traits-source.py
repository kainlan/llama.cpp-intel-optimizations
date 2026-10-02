#!/usr/bin/env python3
"""Source/module contract for the SYCL-private CPU traits provider."""
from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parents[1]
sycl_dir = root / "ggml/src/ggml-sycl"
provider = (sycl_dir / "cpu-traits-support.cpp").read_text()
header = (sycl_dir / "cpu-traits-support.hpp").read_text()
sycl_cpp = (sycl_dir / "ggml-sycl.cpp").read_text()
dispatch = (sycl_dir / "cpu-dispatch.cpp").read_text()
cpu_cpp = (root / "ggml/src/ggml-cpu/ggml-cpu.cpp").read_text()
root_cmake = (root / "ggml/src/CMakeLists.txt").read_text()
sycl_cmake = (sycl_dir / "CMakeLists.txt").read_text()



def strip_comments(source):
    return re.sub(r"//[^\n]*|/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), source, flags=re.S)


def dl_view(source):
    """The text a GGML_BACKEND_DL build compiles: comments gone, and every `#ifndef GGML_BACKEND_DL` branch
    dropped (its `#else` kept), every `#ifdef GGML_BACKEND_DL` branch kept (its `#else` dropped). Other
    conditionals are left alone, both branches kept. The TKV-13 host attention dispatch (c099dbc1c) names
    ggml_backend_graph_compute behind exactly such a guard, so a flat text scan of the file cannot tell a
    DL-reachable reference from one the DL module never compiles."""
    out, stack = [], []  # stack entries: [is_dl_conditional, currently_emitting_parent, active_branch_is_dl_visible]
    emitting = True
    for line in strip_comments(source).split("\n"):
        directive = re.match(r"\s*#\s*(ifndef|ifdef|if|else|elif|endif)\b\s*(.*)", line)
        if directive:
            kind, rest = directive.group(1), directive.group(2).strip()
            if kind in ("ifndef", "ifdef", "if"):
                condition = rest.split("//")[0].strip()
                # The three spellings of "is GGML_BACKEND_DL defined": `#ifdef X`, `#ifndef X` and
                # `#if [!]defined(X)` / `#if [!]defined X`. A compound condition is not decided here.
                defined_form = re.fullmatch(r"(!\s*)?defined\s*(?:\(\s*GGML_BACKEND_DL\s*\)|\s+GGML_BACKEND_DL)",
                                            condition) if kind == "if" else None
                dl = condition == "GGML_BACKEND_DL" and kind in ("ifndef", "ifdef") or defined_form is not None
                if kind == "if" and defined_form is not None:
                    visible = defined_form.group(1) is None
                else:
                    visible = (kind == "ifdef") if dl else True
                stack.append([dl, emitting, visible])
                emitting = emitting and visible
            elif kind in ("else", "elif") and stack:
                dl, parent, visible = stack[-1]
                if dl:
                    stack[-1][2] = not visible
                    emitting = parent and stack[-1][2]
            elif kind == "endif" and stack:
                _, parent, _ = stack.pop()
                emitting = parent
            out.append("")
            continue
        out.append(line if emitting else "")
    return "\n".join(out)


def dl_view_spelling_control():
    """Positive control, independent of the tree: the same non-DL-only reference, guarded each of the ways a
    conditional can say "not a DL build", must be gone from the DL view, and the DL-only branch must stay."""
    for open_guard in ("#ifndef GGML_BACKEND_DL", "#if !defined(GGML_BACKEND_DL)", "#if !defined GGML_BACKEND_DL",
                       "#if ! defined( GGML_BACKEND_DL )"):
        view = dl_view(open_guard + "\n  non_dl_only();\n#else\n  dl_only();\n#endif\n")
        if "non_dl_only" in view or "dl_only" not in view:
            return False
    for open_guard in ("#ifdef GGML_BACKEND_DL", "#if defined(GGML_BACKEND_DL)", "#if defined GGML_BACKEND_DL"):
        view = dl_view(open_guard + "\n  dl_only();\n#else\n  non_dl_only();\n#endif\n")
        if "non_dl_only" in view or "dl_only" not in view:
            return False
    # An unrelated conditional keeps both branches.
    view = dl_view("#if defined(OTHER)\n  a();\n#else\n  b();\n#endif\n")
    return "a();" in view and "b();" in view


sycl_dir_sources = [
    p.read_text(errors="replace") for p in sycl_dir.rglob("*")
    if p.suffix in (".cpp", ".hpp") and "tests" not in p.parts
]
module_sources = "\n".join(dl_view(text) for text in sycl_dir_sources)
module_sources_flat = "\n".join(strip_comments(text) for text in sycl_dir_sources)
PROVIDER_FILES = ("cpu-traits-support.cpp", "cpu-traits-support.hpp")
# Comment-blind call-site inventory of the private provider's wrapper, per consumer file. A new consumer, or
# a consumer growing a call, is a deliberate edit here rather than a drifting literal total.
EXPECTED_PRIVATE_TRAIT_CALLS = {"cpu-dispatch.cpp": 24, "ggml-sycl.cpp": 7, "unified-cache.cpp": 1}
private_trait_calls, raw_trait_calls = {}, {}
for p in sorted(sycl_dir.rglob("*")):
    if p.suffix not in (".cpp", ".hpp") or "tests" in p.parts or p.name in PROVIDER_FILES:
        continue
    text = strip_comments(p.read_text(errors="replace"))
    n_private = len(re.findall(r"\bggml_sycl_get_type_traits_cpu\(", text))
    n_raw = len(re.findall(r"\bggml_get_type_traits_cpu\(", text))
    if n_private:
        private_trait_calls[p.name] = n_private
    if n_raw:
        raw_trait_calls[p.name] = n_raw
table_types = re.findall(r"^\s*TRAIT\(([A-Z0-9_]+),", provider, re.M)
expected = "F32 F16 Q1_0 Q2_0 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q8_1 MXFP4 NVFP4 Q2_K Q3_K Q4_K Q5_K Q6_K IQ2_XXS IQ2_XS IQ3_XXS IQ3_S IQ2_S IQ1_S IQ1_M IQ4_NL IQ4_XS BF16 TQ1_0 TQ2_0".split()
forbidden_registry_symbols = (
    "ggml_backend_reg_by_name", "ggml_backend_reg_get_proc_address", "ggml_backend_dev_init",
    "ggml_backend_graph_compute",
)

checks = {
    "SYCL trait call sites are the expected private-provider inventory": private_trait_calls == EXPECTED_PRIVATE_TRAIT_CALLS,
    "no SYCL consumer calls the raw ggml-cpu trait getter": not raw_trait_calls,
    "exact portable baseline types": table_types == expected and "table[GGML_TYPE_Q8_K]" in provider,
    "bounds checked": "index >= 0 && index < GGML_TYPE_COUNT" in provider,
    "DL local/static optimized split": "#ifndef GGML_BACKEND_DL" in provider
        and "ggml_get_type_traits_cpu(type)" in provider
        and "#else\n    return ggml_sycl_get_baseline_type_traits_cpu(type);" in provider,
    "no CPU registry proc export": "ggml_backend_cpu_get_type_traits" not in cpu_cpp,
    "allocation-free packed vec dot": "std::array<float, k_vec_dot_tile>" in provider
        and "packed_value<Y>" in provider
        and all(x not in provider for x in ("std::vector", "thread_local", "malloc(", "new ")),
    "traits resolved before row loop": provider.index("ggml_get_type_traits(X)") < provider.index("for (int row = 0;"),
    "canonical Q8_K metadata": "{ from_float_q8_k, nullptr, static_cast<ggml_type>(0), 0 }" in provider,
    "module has no registry references": all(x not in module_sources for x in forbidden_registry_symbols),
    # Positive control for the DL view above: ggml_backend_graph_compute IS still named in the non-DL
    # code, so a clean DL view proves the guards are doing the excluding. Without it, an all-comments
    # or empty view would pass the check above for the wrong reason.
    # MAINTENANCE: this control reads the real tree. If a cleanup deletes the last non-DL-guarded
    # ggml_backend_graph_compute reference (the TKV-13 host attention dispatch), `module_sources_flat` stops
    # containing it and this check FAILS with nothing wrong in the DL view; retarget it at another
    # guarded reference then. The synthetic check below does not depend on the tree and keeps covering the
    # spellings of the guard itself.
    "the DL view really excludes the guarded non-DL references":
        "ggml_backend_graph_compute" in module_sources_flat and "ggml_backend_graph_compute" not in module_sources,
    "dl_view treats every spelling of the DL guard alike":
        dl_view_spelling_control(),
    "DL fallback propagates recoverable status": "throw ggml_sycl_fallback_error(reason)" in sycl_cpp
        and "catch (const ggml_sycl_fallback_error & error)" in sycl_cpp
        and "return GGML_STATUS_FAILED" in sycl_cpp,
    "CPU remains a DL module": "add_library(${backend} MODULE ${ARGN})" in root_cmake
        and 'backend STREQUAL "ggml-cpu"' not in root_cmake,
    "CPU link is non-DL only": re.search(r"if \(NOT GGML_BACKEND_DL\).*?target_link_libraries\(ggml-sycl PRIVATE ggml-cpu\).*?endif", sycl_cmake, re.S)
        and "BUILD_RPATH" not in sycl_cmake and "INSTALL_RPATH" not in sycl_cmake,
    "Windows runtime install destination": root_cmake.count("RUNTIME DESTINATION") >= 2,
    "static/DL test guards": "if (NOT GGML_BACKEND_DL)" in sycl_cmake and "if (GGML_BACKEND_DL)" in sycl_cmake,
    "private header only": "cpu-traits-support" not in " ".join(str(p) for p in (root / "ggml/include").glob("*")),
}

failed = [name for name, ok in checks.items() if not ok]
for name, ok in checks.items():
    print(("PASS" if ok else "FAIL") + ": " + name)
if failed:
    print(f"{len(failed)} source contract(s) failed", file=sys.stderr)
    raise SystemExit(1)
