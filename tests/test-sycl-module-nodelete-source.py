#!/usr/bin/env python3
"""Contracts for versioned backend lifetime policy and logical SYCL reload."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
registry = (root / "ggml/src/ggml-backend-reg.cpp").read_text()
loader = (root / "ggml/src/ggml-backend-dl.cpp").read_text()
header = (root / "ggml/src/ggml-backend-dl.h").read_text()
wrapper = (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text()
probe = (root / "tests/test-sycl-module-dlopen.cpp").read_text()
backend = (root / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()

# Policy, not filename, decides process lifetime. This covers renamed DSOs.
assert "ggml_backend_lifetime_policy_v1" in header
assert "ggml_backend_lifetime_policy_v1" in loader
assert "ggml_backend_lifetime_policy_v1" in backend
assert 'filename.find("ggml-sycl")' not in registry
assert "dl_pin_library(handle.get(), path)" in registry  # initialized-registry fallback
assert "RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE" in loader
assert "GET_MODULE_HANDLE_EX_FLAG_PIN" in loader
assert "LoadLibraryW" in loader  # positive Windows reference precedes pin

# The dependency probe keeps RTLD_NOW validation, verifies the policy, then
# logically unloads while retaining the process pin.
assert "RTLD_NOW | RTLD_LOCAL" in probe
assert "ggml_backend_lifetime_policy_v1" in probe
assert "RTLD_NODELETE" in probe
assert "GET_MODULE_HANDLE_EX_FLAG_PIN" in probe

# NODELETE reload must reset/rebuild module state rather than reuse shutdown
# worker, split queue/config, or watchdog authority.
assert "split_config_shutdown_reset();" in backend
assert "g_split_config               = {};" in backend
assert "g_split_secondary_queue_owner.reset();" in backend
assert "g_cpu_worker.e_src1   = sycl::event{};" in backend
assert "ggml_sycl_watchdog_start();" in backend
assert "watchdog_once" not in backend
assert "prepare_unified_cache_for_module_use()" in backend
assert 'ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_shutdown")' in wrapper
assert wrapper.count("ggml_backend_unload(reg)") == 1
assert wrapper.count("ggml_backend_unload_checked(reg)") >= 3
assert "GGML_BACKEND_UNLOAD_BUSY" in wrapper
assert 'phase("complete")' in wrapper
print("SYCL module lifetime-policy source contract: PASS")
