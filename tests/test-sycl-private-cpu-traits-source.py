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

module_sources = "\n".join(
    p.read_text(errors="replace") for p in sycl_dir.rglob("*")
    if p.suffix in (".cpp", ".hpp") and "tests" not in p.parts
)
calls = len(re.findall(r"ggml_sycl_get_type_traits_cpu\(", sycl_cpp + dispatch))
table_types = re.findall(r"^\s*TRAIT\(([A-Z0-9_]+),", provider, re.M)
expected = "F32 F16 Q1_0 Q2_0 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q8_1 MXFP4 NVFP4 Q2_K Q3_K Q4_K Q5_K Q6_K IQ2_XXS IQ2_XS IQ3_XXS IQ3_S IQ2_S IQ1_S IQ1_M IQ4_NL IQ4_XS BF16 TQ1_0 TQ2_0".split()
forbidden_module_symbols = (
    "ggml_backend_reg_by_name", "ggml_backend_reg_get_proc_address", "ggml_backend_dev_init",
    "ggml_backend_graph_compute", "ggml_get_type_traits_cpu", "ggml_threadpool_new",
    "ggml_threadpool_free", "ggml_graph_plan", "ggml_graph_compute(",
)

checks = {
    "all original SYCL trait sites use private provider": calls == 27 and "ggml_get_type_traits_cpu(" not in sycl_cpp + dispatch,
    "exact portable baseline types": table_types == expected and "table[GGML_TYPE_Q8_K]" in provider,
    "bounds checked": "index >= 0 && index < GGML_TYPE_COUNT" in provider,
    "sole local source": "return ggml_sycl_get_baseline_type_traits_cpu(type)" in provider
        and "ggml_backend_" not in provider,
    "no CPU registry proc export": "ggml_backend_cpu_get_type_traits" not in cpu_cpp,
    "allocation-free vec dot": "std::array<float, k_vec_dot_tile>" in provider
        and all(x not in provider for x in ("std::vector", "thread_local", "malloc(", "new ")),
    "traits resolved before row loop": provider.index("ggml_get_type_traits(X)") < provider.index("for (int row = 0;")
        and provider.index("ggml_get_type_traits(Y)") < provider.index("for (int row = 0;"),
    "canonical Q8_K metadata": "{ from_float_q8_k, nullptr, static_cast<ggml_type>(0), 0 }" in provider,
    "module has no registry or CPU compute references": all(x not in module_sources for x in forbidden_module_symbols),
    "fallback returns to scheduler": "Let the scheduler route unsupported work to CPU instead" in sycl_cpp
        and re.search(r"ggml_sycl_cpu_fallback_graph\(.*?return false;\n\}", sycl_cpp, re.S),
    "CPU remains a DL module": "add_library(${backend} MODULE ${ARGN})" in root_cmake
        and 'backend STREQUAL "ggml-cpu"' not in root_cmake,
    "no CPU link or RPATH": "target_link_libraries(ggml-sycl PRIVATE ggml-cpu)" not in sycl_cmake
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
