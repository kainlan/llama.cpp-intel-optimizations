from __future__ import annotations

import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sycl-intel-pc-sampling-capability.sh"


def script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_pc_sampling_probe_is_dry_run_by_default(tmp_path: pathlib.Path) -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--out-root", str(tmp_path / "out")],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert "DRY RUN" in result.stdout
    assert "vtune -help collect gpu-hotspots" in result.stdout
    assert "vtune -help report gpu-source-line" in result.stdout
    assert "command -v gtpin64 || command -v gtpin" in result.stdout
    assert "GTPin/Profilers/Bin/gtpin" in result.stdout
    assert "gtpin-memorytrace-path.txt" in result.stdout
    assert "find /opt/intel/oneapi" in result.stdout
    assert "zetMetricGroupGet" in result.stdout
    assert "zetMetricGetProperties" in result.stdout
    assert "ZET_ENABLE_METRICS" in result.stdout
    assert "ip_metric_count" in result.stdout
    assert "level zero metric" in result.stdout.lower()
    assert "pc_sampling.status" in result.stdout
    assert not (tmp_path / "out" / "level-zero-metric-groups.txt").exists()


def test_pc_sampling_probe_never_synthesizes_pc_sample_csv() -> None:
    text = script_text()
    assert "pc-samples.csv" in text
    assert ">\"${OUT_ROOT}/pc-samples.csv\"" not in text
    assert "> ${OUT_ROOT}/pc-samples.csv" not in text


def test_pc_sampling_probe_distinguishes_ip_metric_capability_from_samples() -> None:
    text = script_text()
    assert "ZET_METRIC_TYPE_IP" in text
    assert "no_level_zero_ip_metric_type_exposed" in text
    assert "level_zero_ip_metrics_found_but_no_pc_sample_producer" in text
    assert "level_zero_metrics_are_not_pc_samples" in text
    assert "gtpin_memorytrace_available_but_not_sampled_pc" in text
    assert "gtpin_framework_present_but_no_profiler_plugin" in text
    assert "status=\"available\"" in text


def test_pc_sampling_probe_refuses_execute_without_ack(tmp_path: pathlib.Path) -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--execute", "--out-root", str(tmp_path / "out")],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 2
    assert "--i-understand-this-probes-intel-gpu-pc-sampling" in result.stdout
