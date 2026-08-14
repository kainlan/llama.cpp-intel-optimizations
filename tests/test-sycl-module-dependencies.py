#!/usr/bin/env python3
"""RTLD_NOW-load and audit the ordinary SYCL DSO's production boundary."""
import ctypes
import os
import pathlib
import re
import shutil
import subprocess
import sys

module = pathlib.Path(sys.argv[1])
if not module.is_file():
    raise SystemExit(f"missing SYCL module: {module}")

if sys.platform == "darwin":
    deps = subprocess.check_output(["otool", "-L", str(module)], text=True)
    symbols = subprocess.check_output(["nm", "-u", "-C", str(module)], text=True)
    exports = subprocess.check_output(["nm", "-gU", "-C", str(module)], text=True)
elif sys.platform == "win32":
    dumpbin = shutil.which("dumpbin")
    if not dumpbin:
        raise SystemExit("dumpbin is required for the Windows module dependency test")
    deps = subprocess.check_output([dumpbin, "/dependents", str(module)], text=True, errors="replace")
    symbols = subprocess.check_output([dumpbin, "/imports", str(module)], text=True, errors="replace")
    exports = subprocess.check_output([dumpbin, "/exports", str(module)], text=True, errors="replace")
else:
    readelf = shutil.which("readelf")
    nm = shutil.which("nm")
    strings_tool = shutil.which("strings")
    if not readelf or not nm or not strings_tool:
        raise SystemExit("readelf, nm, and strings are required for the ELF module dependency test")
    deps = subprocess.check_output([readelf, "-d", str(module)], text=True)
    symbols = subprocess.check_output([nm, "-D", "--undefined-only", "-C", str(module)], text=True)
    exports = subprocess.check_output([nm, "-D", "--defined-only", "-C", str(module)], text=True)
    payload_symbols = subprocess.check_output([nm, "-C", str(module)], text=True)
    payload_strings = subprocess.check_output([strings_tool, str(module)], text=True)

if "ggml-cpu" in deps.lower() or "ggml_cpu" in deps.lower():
    raise SystemExit("SYCL module has a CPU DT_NEEDED/import dependency:\n" + deps)
for forbidden in ("ggml_backend_reg_by_name", "ggml_backend_reg_get_proc_address", "ggml_backend_dev_init",
                  "ggml_backend_graph_compute", "ggml_get_type_traits_cpu", "ggml_compute_forward_get_rows", "ggml_graph_plan",
                  "ggml_graph_compute", "ggml_threadpool_new", "ggml_threadpool_free"):
    if forbidden in symbols:
        raise SystemExit(f"SYCL module directly imports forbidden CPU symbol {forbidden}")

# Private fixture calls must never leak into the ordinary DSO's dynamic
# undefined set. RTLD_LAZY can hide these until first use, hence both this audit
# and the eager loader check below.
undefined_seams = [line for line in symbols.splitlines()
                   if re.search(r"(?:^|[^A-Za-z0-9])(?:test_|fail_next_|failpoint)", line)]
if undefined_seams:
    raise SystemExit("SYCL module imports private test/failure seams:\n" + "\n".join(undefined_seams))

# These mutable/failure-injection seams belong only to the private direct-source
# fixture. Pure production policy helpers whose historical names start test_
# are intentionally not covered by this exact denylist.
private_exports = (
    "test_set_layout_override", "test_clear_layout_override", "test_get_layout_override",
    "test_set_moe_planned_layout_probe_override", "test_clear_moe_planned_layout_probe_overrides",
    "test_reset_orchestrator_call_count", "test_get_orchestrator_call_count",
    "test_plan_publication_prepare_failure_is_caught", "test_provisional_placement_id_exhaustion_is_caught",
    "test_set_kv_placement_plan", "test_clear_kv_placement_plan",
    "test_set_sycl_info_override", "test_clear_sycl_info_override", "test_sycl_info_override_active",
    "ggml_backend_sycl_test_fail_", "ggml_backend_sycl_test_block_", "ggml_backend_sycl_test_wait_",
    "ggml_backend_sycl_test_release_", "ggml_backend_sycl_test_hold_", "ggml_backend_sycl_test_seed_",
    "ggml_backend_sycl_test_allocate_predictor_scores", "ggml_backend_sycl_test_pop_kv_layer_mask",
    "ggml_backend_sycl_test_pending_kv_layer_mask_count", "ggml_backend_sycl_test_kv_layer_mask_identity",
)
leaked_exports = [line for line in exports.splitlines() if any(name in line for name in private_exports)]
if leaked_exports:
    raise SystemExit("SYCL module exports private test/failure seams:\n" + "\n".join(leaked_exports))

# The reload seed must not survive merely as a local symbol or string either.
# It belongs to the separately built lifecycle carrier's registry; checking only
# the dynamic export table would miss a hidden-but-still-mutable ordinary seam.
if sys.platform not in ("darwin", "win32"):
    reload_seed = "ggml_backend_sycl_test_seed_moe_module_state"
    if reload_seed in payload_symbols or reload_seed in payload_strings:
        raise SystemExit(f"ordinary SYCL module contains private reload seed payload {reload_seed}")

try:
    if sys.platform == "win32":
        ctypes.WinDLL(str(module))
    else:
        # Build-tree modules keep ggml-base beside the backend, but an install
        # RPATH configuration need not make that sibling discoverable. Preload
        # it by absolute path without weakening eager resolution of the module.
        base_candidates = sorted(module.parent.glob("libggml-base.*"))
        if base_candidates:
            ctypes.CDLL(str(base_candidates[0]), mode=os.RTLD_NOW | os.RTLD_GLOBAL)
        # IntelLLVM normally adds SVML to an executable's link line. Python is
        # not such an executable, so make that compiler runtime global when it
        # is available before applying RTLD_NOW to the backend.
        for directory in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
            svml = pathlib.Path(directory) / "libsvml.so"
            if svml.is_file():
                ctypes.CDLL(str(svml), mode=os.RTLD_NOW | os.RTLD_GLOBAL)
                break
        ctypes.CDLL(str(module), mode=os.RTLD_NOW | os.RTLD_LOCAL)
except OSError as exc:
    raise SystemExit(f"eager dynamic load failed for {module}: {exc}") from exc

print("SYCL module RTLD_NOW/dependency/seam contract: PASS")
