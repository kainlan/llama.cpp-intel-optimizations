#!/usr/bin/env python3
"""Source/module contract for the SYCL-private CPU traits provider."""
from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parents[1]
provider = (root / "ggml/src/ggml-sycl/cpu-traits-support.cpp").read_text()
header = (root / "ggml/src/ggml-sycl/cpu-traits-support.hpp").read_text()
sycl_cpp = (root / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()
dispatch = (root / "ggml/src/ggml-sycl/cpu-dispatch.cpp").read_text()
cpu_cpp = (root / "ggml/src/ggml-cpu/ggml-cpu.cpp").read_text()
root_cmake = (root / "ggml/src/CMakeLists.txt").read_text()
sycl_cmake = (root / "ggml/src/ggml-sycl/CMakeLists.txt").read_text()

calls = len(re.findall(r"ggml_sycl_get_type_traits_cpu\(", sycl_cpp + dispatch))
table_types = re.findall(r"^\s*TRAIT\(([A-Z0-9_]+),", provider, re.M)
expected = "F32 F16 Q1_0 Q2_0 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q8_1 MXFP4 NVFP4 Q2_K Q3_K Q4_K Q5_K Q6_K IQ2_XXS IQ2_XS IQ3_XXS IQ3_S IQ2_S IQ1_S IQ1_M IQ4_NL IQ4_XS Q8_K BF16 TQ1_0 TQ2_0".split()

checks = {
    "all 28 SYCL trait call sites replaced": calls == 28 and "ggml_get_type_traits_cpu(" not in sycl_cpp + dispatch,
    "exact 30-type portable baseline": table_types == expected,
    "bounds checked": "index < 0 || index >= GGML_TYPE_COUNT" in provider and "index >= 0 && index < GGML_TYPE_COUNT" in provider,
    "fresh optional CPU delegate": 'ggml_backend_reg_by_name("CPU")' in provider
        and 'ggml_backend_reg_get_proc_address(cpu, "ggml_backend_cpu_get_type_traits")' in provider
        and "static traits_getter" not in provider,
    "private CPU registry proc": 'strcmp(name, "ggml_backend_cpu_get_type_traits")' in cpu_cpp,
    "CPU remains a DL module": "add_library(${backend} MODULE ${ARGN})" in root_cmake
        and 'backend STREQUAL "ggml-cpu"' not in root_cmake,
    "no CPU link or RPATH": "target_link_libraries(ggml-sycl PRIVATE ggml-cpu)" not in sycl_cmake
        and "BUILD_RPATH" not in sycl_cmake and "INSTALL_RPATH" not in sycl_cmake,
    "Windows runtime install destination": root_cmake.count("RUNTIME DESTINATION") >= 2,
    "generic CPU graph fallback": "ggml_backend_graph_compute(cpu_backend, graph)" in sycl_cpp
        and 'ggml_backend_reg_by_name("CPU")' in sycl_cpp
        and '"ggml_backend_set_n_threads"' in sycl_cpp,
    "no direct CPU graph symbols": all(x not in sycl_cpp for x in (
        "ggml_threadpool_new(", "ggml_threadpool_free(", "ggml_graph_plan(", "ggml_graph_compute(graph,")),
    "CPU-off clean failure": "CPU fallback unavailable" in sycl_cpp and "if (!cpu_backend)" in sycl_cpp,
    "static/DL test guards": "if (NOT GGML_BACKEND_DL)" in sycl_cmake and "if (GGML_BACKEND_DL)" in sycl_cmake,
    "private header only": "cpu-traits-support" not in " ".join(str(p) for p in (root / "ggml/include").glob("*")),
}

failed = [name for name, ok in checks.items() if not ok]
for name, ok in checks.items():
    print(("PASS" if ok else "FAIL") + ": " + name)
if failed:
    print(f"{len(failed)} source contract(s) failed", file=sys.stderr)
    raise SystemExit(1)
