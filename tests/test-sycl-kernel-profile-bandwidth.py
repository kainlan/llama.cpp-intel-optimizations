#!/usr/bin/env python3
"""Roofline arithmetic for the SYCL named-kernel profile parser.

Pure stdlib, no GPU: the geometry preset and the bandwidth arithmetic are
checked against the recorded B50 GPT-OSS 20B TG capture so a regression in
either inverts visibly instead of silently.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
from importlib.machinery import SourceFileLoader

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARSER = ROOT / "scripts" / "parse-sycl-kernel-profile.py"

mod = SourceFileLoader("kprof", str(PARSER)).load_module()

CSV_HEADER = (
    "name,category,metadata,device,queue_kind,count,total_ns,mean_ns,"
    "min_ns,p50_ns,p95_ns,max_ns,bytes,failed_timestamps,graph_recorded"
)

# Reference B50 GPT-OSS 20B TG capture, rounded to the per-call means quoted in
# the plan: gate/up 235 us per call, down (SOA batched matvec) 264 us per call.
GATEUP_CALLS = 1560
GATEUP_NS = GATEUP_CALLS * 235_000
DOWN_CALLS = 1235
DOWN_NS = DOWN_CALLS * 264_000

REFERENCE_CSV = "\n".join(
    [
        CSV_HEADER,
        f"mxfp4.gateup.xmx_tiled_dpas_m2,mmvq,path=packed-q8-m2;role=gateup,0,compute,"
        f"{GATEUP_CALLS},{GATEUP_NS},235000,221041,235000,248646,408854,0,0,0",
        f"mxfp4.soa.batched,mmvq,path=soa;role=matvec,0,compute,"
        f"{DOWN_CALLS},{DOWN_NS},264000,246146,264000,282604,297500,0,0,0",
        "sycl.rope,rope,role=rope,0,compute,3120,9622085,3084,2083,3437,3959,21562,28753920,0,0",
        "",
    ]
)


def write_csv(tmp_path: pathlib.Path, text: str = REFERENCE_CSV) -> pathlib.Path:
    path = tmp_path / "kprof.csv"
    path.write_text(text, encoding="utf-8")
    return path


def run_parser(path: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PARSER), *args, str(path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def metrics(stdout: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in stdout.splitlines():
        key, _, value = line.partition(" ")
        if key:
            out[key] = value
    return out


# --- arithmetic -----------------------------------------------------------


def test_mxfp4_expert_matrix_bytes():
    # 2880x2880 MXFP4 = 8_294_400 elems / 32 * 17
    assert mod.mxfp4_bytes(2880, 2880) == 8_294_400 // 32 * 17


def test_achieved_bandwidth_gbs():
    # 16.8 MiB in 264 us  ~= 66.7 GB/s
    gbs = mod.achieved_gbs(bytes_moved=16.8 * 2**20, seconds=264e-6)
    assert 60.0 < gbs < 72.0


def test_percent_of_peak():
    assert abs(mod.percent_of_peak(150.0, 224.0) - 66.96) < 0.1


def test_zero_duration_and_zero_peak_do_not_divide_by_zero():
    assert mod.achieved_gbs(bytes_moved=1024.0, seconds=0.0) == 0.0
    assert mod.percent_of_peak(150.0, 0.0) == 0.0


def test_mxfp4_bytes_rounds_partial_block_up():
    # a trailing partial block still occupies a whole 17-byte MXFP4 block
    assert mod.mxfp4_bytes(33, 1) == 2 * 17


# --- geometry preset ------------------------------------------------------


def test_gpt_oss_20b_preset_shape():
    preset = mod.GEOMETRY_PRESETS["gpt-oss-20b"]
    assert preset["layers"] == 24
    assert preset["experts"] == 32
    assert preset["experts_active"] == 4
    assert preset["expert_ncols"] == 2880
    assert preset["expert_nrows"] == 2880


def test_gateup_moves_two_matrices_down_moves_one():
    # The load-bearing asymmetry: gate/up streams gate AND up (2 matrices x 4
    # experts = 33.6 MiB); down streams a single matrix per expert (16.8 MiB).
    # Swapping these inverts the roofline verdict.
    matrix = mod.mxfp4_bytes(2880, 2880)
    gateup = mod.geometry_kernel_bytes("gpt-oss-20b", "mxfp4.gateup.xmx_tiled_dpas_m2")
    down = mod.geometry_kernel_bytes("gpt-oss-20b", "mxfp4.soa.batched")
    assert gateup == 2 * 4 * matrix
    assert down == 1 * 4 * matrix
    assert gateup == 2 * down
    assert abs(gateup / 2**20 - 33.6) < 0.1
    assert abs(down / 2**20 - 16.8) < 0.1


def test_unmapped_kernel_has_no_geometry_bytes():
    assert mod.geometry_kernel_bytes("gpt-oss-20b", "sycl.rope") is None


def test_unknown_preset_raises_value_error():
    # argparse `choices` guards the CLI, but the lookup is a library entry
    # point too, so it must reject an unknown name on its own.
    with pytest.raises(ValueError, match="unknown geometry preset"):
        mod.geometry_preset("bogus")


def test_role_rule_without_role_matrices_entry_raises_value_error():
    # Adding a preset is a pure data edit; a role key typo must not reach
    # `geometry_kernel_bytes` and die there on a bare KeyError.
    broken = {
        "layers": 1,
        "experts": 2,
        "experts_active": 1,
        "expert_ncols": 32,
        "expert_nrows": 32,
        "role_matrices": {"gateup": 2},
        "role_rules": [("gateup", "gateup"), ("down", "dwon")],
    }
    with pytest.raises(ValueError, match="role_matrices"):
        mod.validate_geometry_preset("broken", broken)

    mod.GEOMETRY_PRESETS["broken"] = broken
    try:
        with pytest.raises(ValueError, match="role_matrices"):
            mod.geometry_kernel_bytes("broken", "some.down.kernel")
    finally:
        del mod.GEOMETRY_PRESETS["broken"]


def test_shipped_presets_are_self_consistent():
    for name, preset in mod.GEOMETRY_PRESETS.items():
        mod.validate_geometry_preset(name, preset)


# --- CLI ------------------------------------------------------------------


def test_cli_reproduces_recorded_bandwidths(tmp_path):
    path = write_csv(tmp_path)
    proc = run_parser(path, "--geometry", "gpt-oss-20b", "--peak-gbs", "224")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    found = metrics(proc.stdout)

    gateup = int(found["kernel.mxfp4.gateup.xmx_tiled_dpas_m2.achieved_gbps_x1000"])
    down = int(found["kernel.mxfp4.soa.batched.achieved_gbps_x1000"])
    assert 145_000 <= gateup <= 155_000, gateup
    assert 62_000 <= down <= 72_000, down

    # percent of the 224 GB/s peak
    assert 65_000 <= int(found["kernel.mxfp4.gateup.xmx_tiled_dpas_m2.pct_of_peak_x1000"]) <= 69_000
    assert 27_000 <= int(found["kernel.mxfp4.soa.batched.pct_of_peak_x1000"]) <= 32_000


def test_cli_reports_geometry_bytes_per_call(tmp_path):
    path = write_csv(tmp_path)
    proc = run_parser(path, "--geometry", "gpt-oss-20b")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    found = metrics(proc.stdout)
    assert found["geometry.gpt-oss-20b.layers"] == "24"
    assert found["geometry.gpt-oss-20b.experts"] == "32"
    assert found["geometry.gpt-oss-20b.experts_active"] == "4"
    assert found["geometry.gpt-oss-20b.expert_matrix_bytes"] == str(8_294_400 // 32 * 17)
    assert found["kernel.mxfp4.gateup.xmx_tiled_dpas_m2.geometry_bytes_per_call"] == "35251200"
    assert found["kernel.mxfp4.soa.batched.geometry_bytes_per_call"] == "17625600"
    # no peak supplied -> no percentage rows
    assert "kernel.mxfp4.soa.batched.pct_of_peak_x1000" not in found


def test_cli_rejects_unknown_geometry(tmp_path):
    path = write_csv(tmp_path)
    proc = run_parser(path, "--geometry", "not-a-model")
    assert proc.returncode != 0


def test_explicit_kernel_bytes_wins_over_geometry(tmp_path):
    path = write_csv(tmp_path)
    proc = run_parser(
        path,
        "--geometry",
        "gpt-oss-20b",
        "--kernel-bytes",
        "mxfp4.soa.batched=17625600000",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    found = metrics(proc.stdout)
    # 1000x the geometry bytes -> 1000x the geometry bandwidth
    assert int(found["kernel.mxfp4.soa.batched.achieved_gbps_x1000"]) > 60_000_000


def test_peak_applies_to_explicit_kernel_bytes_without_geometry(tmp_path):
    path = write_csv(tmp_path)
    proc = run_parser(
        path,
        "--kernel-bytes",
        "mxfp4.soa.batched=17625600",
        "--peak-gbs",
        "224",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    found = metrics(proc.stdout)
    assert 27_000 <= int(found["kernel.mxfp4.soa.batched.pct_of_peak_x1000"]) <= 32_000


def test_existing_cli_is_unchanged_without_new_flags(tmp_path):
    path = write_csv(tmp_path)
    proc = run_parser(path, "--top-kernels", "2", "--wall-ms", "500", "--require-kernel", "sycl.rope")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    found = metrics(proc.stdout)
    assert found["cost.top1_kernel"] == "mxfp4.gateup.xmx_tiled_dpas_m2 366600"
    assert found["cost.kernel.rank.2.name"] == "mxfp4.soa.batched"
    assert found["kernel.sycl.rope.count"] == "3120"
    assert "profile.decode_wall_ms_x1000" in found
    # geometry annotations stay off unless asked for
    assert "kernel.mxfp4.soa.batched.geometry_bytes_per_call" not in found
    assert "kernel.mxfp4.soa.batched.achieved_gbps_x1000" not in found
