#!/usr/bin/env python3
"""Audit mutable SYCL cache/MMID/streaming seams and ordinary artifact payload."""
from pathlib import Path
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
}
for name, needles in checks.items():
    text = (SYCL / name).read_text(encoding="utf-8")
    missing = [needle for needle in needles if needle not in text]
    if missing:
        raise SystemExit(f"{name}: missing private seam contract: {missing}")

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
