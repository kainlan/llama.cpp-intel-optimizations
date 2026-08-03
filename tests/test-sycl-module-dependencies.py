#!/usr/bin/env python3
"""Assert that a built SYCL module has no CPU dependency/direct CPU compute imports."""
import pathlib
import shutil
import subprocess
import sys

module = pathlib.Path(sys.argv[1])
if not module.is_file():
    raise SystemExit(f"missing SYCL module: {module}")

if sys.platform == "darwin":
    deps = subprocess.check_output(["otool", "-L", str(module)], text=True)
    symbols = subprocess.check_output(["nm", "-u", str(module)], text=True)
elif sys.platform == "win32":
    dumpbin = shutil.which("dumpbin")
    if not dumpbin:
        raise SystemExit("dumpbin is required for the Windows module dependency test")
    deps = subprocess.check_output([dumpbin, "/dependents", str(module)], text=True, errors="replace")
    symbols = subprocess.check_output([dumpbin, "/imports", str(module)], text=True, errors="replace")
else:
    readelf = shutil.which("readelf")
    nm = shutil.which("nm")
    if not readelf or not nm:
        raise SystemExit("readelf and nm are required for the ELF module dependency test")
    deps = subprocess.check_output([readelf, "-d", str(module)], text=True)
    symbols = subprocess.check_output([nm, "-D", "--undefined-only", str(module)], text=True)

if "ggml-cpu" in deps.lower() or "ggml_cpu" in deps.lower():
    raise SystemExit("SYCL module has a CPU DT_NEEDED/import dependency:\n" + deps)
for forbidden in ("ggml_backend_reg_by_name", "ggml_backend_reg_get_proc_address", "ggml_backend_dev_init",
                  "ggml_backend_graph_compute", "ggml_get_type_traits_cpu", "ggml_graph_plan",
                  "ggml_graph_compute", "ggml_threadpool_new", "ggml_threadpool_free"):
    if forbidden in symbols:
        raise SystemExit(f"SYCL module directly imports forbidden CPU symbol {forbidden}")
print("SYCL module dependency contract: PASS")
