#!/usr/bin/env python3
"""Contract for SYCL-only logical DSO unload on platforms with RTLD_NODELETE."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
registry = (root / "ggml/src/ggml-backend-reg.cpp").read_text()
loader = (root / "ggml/src/ggml-backend-dl.cpp").read_text()
wrapper = (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text()
backend = (root / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()

assert "backend_requires_logical_unload(path)" in registry
assert "backend_requires_logical_unload(entry.path())" in registry
assert 'filename.find("ggml-sycl")' in registry
assert 'filename.find("ggml_sycl")' in registry
assert "flags |= RTLD_NODELETE" in loader
assert "RTLD_NOW | RTLD_LOCAL" in loader
assert "prepare_unified_cache_for_module_use()" in backend
assert 'ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_shutdown")' in wrapper
assert wrapper.count("ggml_backend_unload(reg)") == 2
assert 'phase("complete")' in wrapper
print("SYCL module NODELETE source contract: PASS")
