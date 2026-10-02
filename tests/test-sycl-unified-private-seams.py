#!/usr/bin/env python3
"""Audit mutable SYCL cache/MMID/streaming seams and ordinary artifact payload."""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"

checks = {
    "residency-plan.hpp": (
        "#if defined(GGML_SYCL_PRIVATE_TESTING)",
        "residency_diagnostics_reset_for_test",
    ),
    "moe-mmid-workspace.hpp": (
        "#if defined(GGML_SYCL_PRIVATE_TESTING)",
        "set_generation_for_test",
    ),
    "layer-streaming.hpp": (
        "#if defined(GGML_SYCL_PRIVATE_TESTING)",
        "test_install_loaded_buffers",
    ),
    "mem-ops.cpp": (
        "GGML_SYCL_MEM_FILL_TEST_CHECK",
        "mem_fill_set_profile_error_after_submit_for_test",
    ),
    # llama.cpp-23mk S3-3: the scratchpad decline seam. The hook and the three test accessors are defined behind the guard
    # and the hook compiles to a constexpr false without it; the guard is what keeps them out of the ordinary artifact.
    "common.cpp": (
        "#if defined(GGML_SYCL_PRIVATE_TESTING)\nbool ggml_sycl_scratchpad_site_hook(",
        "ggml_sycl_test_inject_scratchpad_decline",
    ),
    "common.hpp": (
        "#if defined(GGML_SYCL_PRIVATE_TESTING)\nbool ggml_sycl_scratchpad_site_hook(",
        "constexpr bool ggml_sycl_scratchpad_site_hook(",
    ),
}
for name, needles in checks.items():
    text = (SYCL / name).read_text(encoding="utf-8")
    missing = [needle for needle in needles if needle not in text]
    if missing:
        raise SystemExit(f"{name}: missing private seam contract: {missing}")

# The needles above say the guard exists somewhere in the file. These say each seam definition or declaration sits inside one:
# the nearest `#if defined(GGML_SYCL_PRIVATE_TESTING)` before it opens a region with no other preprocessor line before the
# target, so a seam moved outside its guard (or a guard closed early) fails here.
GUARD = "#if defined(GGML_SYCL_PRIVATE_TESTING)"
guarded = {
    SYCL / "common.cpp": (
        "bool ggml_sycl_scratchpad_site_hook(",
        "GGML_BACKEND_API bool ggml_sycl_test_inject_scratchpad_decline",
        "GGML_BACKEND_API void ggml_sycl_test_scratchpad_sites_reset",
        "GGML_BACKEND_API bool ggml_sycl_test_scratchpad_site_counts",
    ),
    SYCL / "common.hpp": ("bool ggml_sycl_scratchpad_site_hook(ggml_sycl_scratchpad_site site);",),
    ROOT / "ggml/include/ggml-sycl.h": (
        "GGML_BACKEND_API bool ggml_sycl_test_inject_scratchpad_decline",
        "GGML_BACKEND_API void ggml_sycl_test_scratchpad_sites_reset",
        "GGML_BACKEND_API bool ggml_sycl_test_scratchpad_site_counts",
    ),
}
for path, targets in guarded.items():
    text = path.read_text(encoding="utf-8")
    for target in targets:
        at = text.find(target)
        if at < 0:
            raise SystemExit(f"{path.name}: seam `{target}` not found")
        opened = text.rfind(GUARD, 0, at)
        if opened < 0 or re.search(r"^\s*#\s*(?:if|ifdef|ifndef|elif|else|endif)\b", text[opened + len(GUARD):at], re.M):
            raise SystemExit(f"{path.name}: seam `{target}` is not directly inside a `{GUARD}` region")

if len(sys.argv) > 1:
    artifact = Path(sys.argv[1])
    nm = subprocess.run(["nm", "-C", str(artifact)], text=True, capture_output=True, check=True).stdout
    strings = subprocess.run(["strings", str(artifact)], text=True, capture_output=True, check=True).stdout
    forbidden_symbols = (
        "evaluate_residency_request_for_test",
        "residency_diagnostics_reset_for_test",
        "test_cache_replacement_allowed_for_test",
        "lifecycle_set_next_plan_publication_id_for_test",
        "set_generation_for_test",
        "test_install_loaded_buffers",
        "unified_cache_set_expert_publication_test_hook",
        "unified_cache_fail_next_expert_phase_for_test",
        "unified_cache_fail_expert_allocation_after_for_test",
        "mem_fill_set_profile_error_after_submit_for_test",
        "ggml_backend_sycl_test_allocate_predictor_scores",
        "ggml_sycl_test_inject_scratchpad_decline",
        "ggml_sycl_test_scratchpad_sites_reset",
        "ggml_sycl_test_scratchpad_site_counts",
        "ggml_sycl_scratchpad_site_hook",
    )
    leaked = [name for name in forbidden_symbols if name in nm]
    if leaked:
        raise SystemExit(f"ordinary artifact exposes private mutable seams: {leaked}")
    forbidden_strings = (
        "GGML_SYCL_TEST_MEM_FILL_PROFILE_ERROR_AFTER_SUBMIT",
        "GGML_SYCL_TEST_DMA_FAIL",
    )
    leaked_strings = [name for name in forbidden_strings if name in strings]
    if leaked_strings:
        raise SystemExit(f"ordinary artifact contains private fail-control strings: {leaked_strings}")

print("SYCL unified private seam source/artifact contract: PASS")
