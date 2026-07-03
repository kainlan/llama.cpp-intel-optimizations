#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sycl-gptoss-down-variant-profile-matrix.sh"


VARIANTS = [
    "baseline",
    "row2",
    "row4",
    "atomic",
    "cached-vector-qs",
    "cached-cache-y",
    "cached-vector-qs-cache-y",
]

BASELINE_ENVS = [
    "GGML_SYCL_KERNEL_PROFILE=1",
    "GGML_SYCL_KERNEL_PROFILE_FORMAT=both",
    "GGML_SYCL_KERNEL_PROFILE_TOP_N=80",
    "GGML_SYCL_MOE_PHASE_MATERIALIZE=1",
    "GGML_SYCL_MOE_PHASE_BULK_XMX=1",
    "GGML_SYCL_MOE_DOWN_SUM_DIRECT=1",
    "GGML_SYCL_MXFP4_TG_PROFILE=1",
    "ONEAPI_DEVICE_SELECTOR=level_zero:1",
]


def _script_text() -> str:
    return SCRIPT.read_text()


def _run_script(*args: str, out_dir: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["SYCL_DOWN_VARIANT_PROFILE_OUT"] = str(out_dir)
    return subprocess.run(
        [str(SCRIPT), *args],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _combined_output(completed: subprocess.CompletedProcess[str]) -> str:
    return completed.stdout + completed.stderr


def _section(output: str, name: str) -> str:
    marker = f"# case: {name}"
    start = output.index(marker)
    next_markers = [output.find("# case: ", start + len(marker))]
    next_markers = [pos for pos in next_markers if pos != -1]
    end = min(next_markers) if next_markers else len(output)
    return output[start:end]


def test_down_variant_profile_script_defaults_to_dry_run(tmp_path: Path) -> None:
    completed = _run_script(out_dir=tmp_path / "dry-run")
    out = _combined_output(completed)
    assert completed.returncode == 0, out
    assert "DRY RUN" in out
    assert "--execute" in out
    assert "llama-bench" in out
    assert "gpt-oss-20b-mxfp4.gguf" in out


def test_down_variant_profile_script_contains_required_variants_and_profile_envs() -> None:
    text = _script_text()
    for name in VARIANTS:
        assert name in text
    for env in BASELINE_ENVS:
        assert env in text
    assert "-fa 1" in text


def test_down_variant_profile_script_avoids_unsafe_probes() -> None:
    text = _script_text().lower()
    forbidden = ["sycl-ls", "/dev/dri", "lsof", "vtune"]
    for token in forbidden:
        assert token not in text


def test_dry_run_has_no_side_effects_and_does_not_source_oneapi(tmp_path: Path) -> None:
    out_dir = tmp_path / "dry-run-output"
    completed = _run_script("--dry-run", out_dir=out_dir)
    out = _combined_output(completed)
    assert completed.returncode == 0, out
    assert not out_dir.exists()
    assert "setvars.sh" not in out
    assert "source /opt/intel/oneapi" not in out


def test_dry_run_lists_all_variants_in_order(tmp_path: Path) -> None:
    completed = _run_script("--dry-run", out_dir=tmp_path / "dry-run")
    out = _combined_output(completed)
    assert completed.returncode == 0, out
    last = -1
    for name in VARIANTS:
        pos = out.find(f"# case: {name}")
        assert pos != -1, f"missing variant {name}:\n{out}"
        assert pos > last, f"variant {name} out of order:\n{out}"
        last = pos


def test_baseline_row_group_and_atomic_rows_include_fa_on_baseline_envs(tmp_path: Path) -> None:
    completed = _run_script("--dry-run", out_dir=tmp_path / "dry-run")
    out = _combined_output(completed)
    assert completed.returncode == 0, out
    for name in ["baseline", "row2", "row4", "atomic"]:
        section = _section(out, name)
        assert "env " in section
        assert " -fa 1 " in section
        for env in BASELINE_ENVS:
            assert env in section


def test_cached_rows_explicitly_override_direct_sum_and_use_per_command_env(tmp_path: Path) -> None:
    completed = _run_script("--dry-run", out_dir=tmp_path / "dry-run")
    out = _combined_output(completed)
    assert completed.returncode == 0, out
    expected_variant_envs = {
        "cached-vector-qs": "GGML_SYCL_MOE_DOWN_CACHED_Q8_SOA_TG_VARIANT=vector-qs",
        "cached-cache-y": "GGML_SYCL_MOE_DOWN_CACHED_Q8_SOA_TG_VARIANT=cache-y",
        "cached-vector-qs-cache-y": "GGML_SYCL_MOE_DOWN_CACHED_Q8_SOA_TG_VARIANT=vector-qs-cache-y",
    }
    for name, env in expected_variant_envs.items():
        section = _section(out, name)
        assert "env " in section
        assert env in section
        assert "GGML_SYCL_MOE_DOWN_SUM_DIRECT=1" in section
        assert "GGML_SYCL_MOE_DOWN_SUM_DIRECT=0" in section
        assert section.rindex("GGML_SYCL_MOE_DOWN_SUM_DIRECT=0") > section.index("GGML_SYCL_MOE_DOWN_SUM_DIRECT=1")


def test_row_and_atomic_variant_envs_do_not_leak_between_rows(tmp_path: Path) -> None:
    completed = _run_script("--dry-run", out_dir=tmp_path / "dry-run")
    out = _combined_output(completed)
    assert completed.returncode == 0, out
    baseline = _section(out, "baseline")
    assert "GGML_SYCL_MOE_DOWN_SUM_Q8_SOA_TG_VARIANT" not in baseline
    assert "GGML_SYCL_MOE_DOWN_SUM_DIRECT_ATOMIC=1" not in baseline
    assert "GGML_SYCL_MOE_DOWN_CACHED_Q8_SOA_TG_VARIANT" not in baseline
    assert "GGML_SYCL_MOE_DOWN_SUM_Q8_SOA_TG_VARIANT=row2" in _section(out, "row2")
    assert "GGML_SYCL_MOE_DOWN_SUM_Q8_SOA_TG_VARIANT=row4" in _section(out, "row4")
    assert "GGML_SYCL_MOE_DOWN_SUM_DIRECT_ATOMIC=1" in _section(out, "atomic")
