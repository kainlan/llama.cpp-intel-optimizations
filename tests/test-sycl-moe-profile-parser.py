#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARSER = ROOT / "scripts" / "parse-sycl-moe-profile.py"
HARNESS = ROOT / "scripts" / "sycl-b50-gptoss-moe-gates.sh"


def run_parser(path: pathlib.Path, *extra_args: str) -> str:
    return subprocess.check_output([sys.executable, str(PARSER), "--no-lines", *extra_args, str(path)], text=True)


def run_parser_result(path: pathlib.Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PARSER), "--no-lines", *extra_args, str(path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def run_harness_dry_run(mode: str) -> str:
    with tempfile.TemporaryDirectory() as tmp_raw:
        logdir = pathlib.Path(tmp_raw) / "logs"
        result = subprocess.run(
            [
                "bash",
                str(HARNESS),
                "--mode",
                mode,
                "--dry-run",
                "--logdir",
                str(logdir),
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert result.returncode == 0, result.stdout
        return result.stdout


def test_parser_counts_tg_sequence_submit_calls_and_exact_answer() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "count.stdout").write_text(
            "> Count from 1 to 5. Answer with only: 1, 2, 3, 4, 5\n\n1, 2, 3, 4, 5\n[ Prompt: 10.0 t/s | Generation: 50.0 t/s ]\n"
        )
        (tmp / "count.stderr").write_text(
            "[GRAPH-DIAG] phase=TG sequence_graphlet_replay=48 sequence_graphlet_submit_calls=48 sequence_direct_gap=0.125ms\n"
            "[SYCL-MOE-SEQUENCE-GRAPHLET] replayed node=42\n"
            "[MOE-GLU-Q8-DIAG] action=fused_store path=direct_final fused_store=3 cached_allow=2\n"
        )
        out = run_parser(tmp)
        assert "generated.count_exact.true 1" in out
        assert "counter.sequence_graphlet_replay 48" in out
        assert "counter.sequence_graphlet_submit_calls 48" in out
        assert "counter.phase.TG.sequence_graphlet_submit_calls 48" in out
        assert "counter.sequence_direct_gap 1" in out
        assert "counter.phase.TG.sequence_direct_gap 1" in out
        assert "diag.action.fused_store 1" in out
        assert "diag.path.direct_final 1" in out
        assert "counter.fused_store 3" in out
        assert "counter.cached_allow 2" in out


def test_parser_extracts_grouped_decode_device_packed_path_evidence() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "diag.stderr").write_text(
            "[MOE-GLU-Q8-DIAG] action=fused-store layer=0 device=1 rows=4 ne0=2880 event=1 "
            "path=grouped-packed-q8-m2-device fused_candidate=0 fused_reject=disabled saved_launches=0\n"
        )
        out = run_parser(tmp, "--require-diag-path", "grouped-packed-q8-m2-device")
        assert "diag.action.fused-store 1" in out
        assert "diag.path.grouped-packed-q8-m2-device 1" in out
        assert "diag.fused_reject.disabled 1" in out




def test_parser_extracts_clean_xmx_original_validator() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "diag.stderr").write_text(
            "[MOE-XMX-OUTPUT-ORIGINAL-VALIDATE] tensor=blk.0.ffn_gate_exps.weight path=aggressive-partial-fused-tg "
            "batches=4 rows=8 ncols=2880 checked=32 mismatches=0 max_abs=0 mean_abs=0 "
            "first_group=-1 first_row=-1 first_expert=-1 actual=0 expected=0 gate=0 up=0\n"
        )
        out = run_parser(tmp, "--require-xmx-original-clean")
        assert "validator.xmx_original.checked 32" in out
        assert "validator.xmx_original.mismatches 0" in out
        assert "validator.xmx_original.clean.true 1" in out


def test_parser_rejects_dirty_xmx_original_validator() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "diag.stderr").write_text(
            "[MOE-XMX-OUTPUT-ORIGINAL-VALIDATE] tensor=blk.0.ffn_gate_exps.weight path=aggressive-partial-fused-tg "
            "batches=4 rows=8 ncols=2880 checked=32 mismatches=3 max_abs=1.25 mean_abs=0.2 "
            "first_group=0 first_row=2 first_expert=7 actual=1 expected=2 gate=3 up=4\n"
        )
        result = run_parser_result(tmp, "--require-xmx-original-clean")
        assert result.returncode != 0
        assert "validator.xmx_original.checked 32" in result.stdout
        assert "validator.xmx_original.mismatches 3" in result.stdout
        assert "validator.xmx_original.clean.false 1" in result.stdout
        assert "error: XMX original-layout validator reported mismatches: 3" in result.stdout


def test_parser_rejects_missing_xmx_original_validator_when_required() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "diag.stderr").write_text(
            "[MOE-GLU-Q8-DIAG] action=fused-store layer=0 device=1 rows=4 ne0=2880 event=1 "
            "path=aggressive-partial-fused-tg fused_candidate=1 fused_reject=none saved_launches=1\n"
        )
        result = run_parser_result(tmp, "--require-xmx-original-clean")
        assert result.returncode != 0
        assert "error: XMX original-layout validator did not run" in result.stdout


def test_parser_require_diag_path_rejects_stale_host_packed_label_only() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "diag.stderr").write_text(
            "[MOE-GLU-Q8-DIAG] action=publish layer=0 device=1 rows=4 ne0=2880 event=1 "
            "path=packed-q8-m2 fused_candidate=0 fused_reject=no-kernel-q8 saved_launches=0\n"
        )
        result = run_parser_result(tmp, "--require-diag-path", "grouped-packed-q8-m2-device")
        assert result.returncode != 0
        assert "diag.path.packed-q8-m2 1" in result.stdout
        assert "diag.path.grouped-packed-q8-m2-device" not in result.stdout
        assert "error: required diagnostic path missing: grouped-packed-q8-m2-device" in result.stdout


def test_parser_rejects_missing_input_when_gate_supplied() -> None:
    missing = pathlib.Path("/tmp/llama-nonexistent-review-log")
    result = run_parser_result(missing, "--require-no-fatal-markers")
    assert result.returncode != 0
    assert "missing" in result.stdout
    assert "error: missing log inputs:" in result.stdout


def test_parser_rejects_gate_without_inputs() -> None:
    result = subprocess.run(
        [sys.executable, str(PARSER), "--no-lines", "--require-down-dpas-direct-final"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode != 0
    assert "error: no log inputs supplied for gate mode" in result.stdout


def test_parser_rejects_empty_directory_when_gate_supplied() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        result = run_parser_result(pathlib.Path(tmp_raw), "--require-no-fatal-markers")
    assert result.returncode != 0
    assert "error: no log inputs matched for gate mode" in result.stdout


def test_parser_forbid_diag_path_accepts_safe_fallback_path() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "diag.stderr").write_text(
            "[MOE-GLU-Q8-DIAG] action=publish layer=0 device=1 rows=4 ne0=2880 event=1 "
            "path=packed-q8-m2 fused_candidate=0 fused_reject=no-kernel-q8 saved_launches=0\n"
        )
        out = run_parser(
            tmp,
            "--forbid-diag-path",
            "grouped-packed-q8-m2-device",
            "--require-no-fatal-markers",
        )
        assert "diag.path.packed-q8-m2 1" in out
        assert "fatal.total 0" in out


def test_parser_forbid_diag_path_rejects_catastrophic_partial_device_path() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "diag.stderr").write_text(
            "[MOE-GLU-Q8-DIAG] action=publish layer=0 device=1 rows=4 ne0=2880 event=1 "
            "path=grouped-packed-q8-m2-device fused_candidate=0 fused_reject=no-kernel-q8 saved_launches=0\n"
        )
        result = run_parser_result(tmp, "--forbid-diag-path", "grouped-packed-q8-m2-device")
        assert result.returncode != 0
        assert "diag.path.grouped-packed-q8-m2-device 1" in result.stdout
        assert "error: forbidden diagnostic path present: grouped-packed-q8-m2-device count=1" in result.stdout


def test_parser_requires_grouped_diag_tg_floor_before_full_perf() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "diag.stdout").write_text(
            "| model                          |       size |     params | backend    |  ngl | fa |            test |                  t/s |\n"
            "| ------------------------------ | ---------: | ---------: | ---------- | ---: | -: | --------------: | -------------------: |\n"
            "| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | SYCL       |   99 |  1 |            tg32 |          0.21 ± 0.00 |\n"
        )
        result = run_parser_result(tmp / "diag.stdout", "--require-bench-min", "tg32", "5")
        assert result.returncode != 0
        assert "bench.tg32.count 1" in result.stdout
        assert "bench.tg32.tps_x100 21" in result.stdout
        assert "error: bench tg32 below minimum: actual=0.21 required=5.00" in result.stdout


def test_parser_accepts_grouped_diag_tg_floor_when_non_catastrophic() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "diag.stdout").write_text(
            "| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | SYCL       |   99 |  1 |            tg32 |         31.30 ± 0.22 |\n"
        )
        out = run_parser(tmp / "diag.stdout", "--require-bench-min", "tg32", "5")
        assert "bench.tg32.count 1" in out
        assert "bench.tg32.tps_x100 3130" in out


def test_parser_bench_min_rejects_missing_benchmark_even_for_small_threshold() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "empty.stdout").write_text("")
        result = run_parser_result(tmp, "--require-bench-min", "tg32", "0.01")
        assert result.returncode == 8
        assert "error: required bench test missing: tg32" in result.stdout


def test_parser_bench_min_rejects_non_positive_threshold() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "empty.stdout").write_text("")
        result = run_parser_result(tmp, "--require-bench-min", "tg32", "0")
        assert result.returncode != 0
        assert "TPS threshold must be finite and positive" in result.stdout


def test_parser_accepts_generic_model_name_in_bench_rows() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "bench.stdout").write_text(
            "| custom model name             |   1.00 GiB |     1.00 B | SYCL       |   99 |  1 |            tg32 |         12.50 ± 0.10 |\n"
        )
        out = run_parser(tmp / "bench.stdout", "--require-bench-min", "tg32", "5")
        assert "bench.tg32.count 1" in out
        assert "bench.tg32.tps_x100 1250" in out


def test_parser_bench_min_uses_best_throughput_not_sum_across_logs() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        bench_line = (
            "| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | SYCL       |   99 |  1 |"
            "            tg32 |          3.00 ± 0.01 |\n"
        )
        (tmp / "diag_a.stdout").write_text(bench_line)
        (tmp / "diag_b.stdout").write_text(bench_line)
        result = run_parser_result(tmp, "--require-bench-min", "tg32", "5")
        assert result.returncode != 0
        assert "bench.tg32.count 2" in result.stdout
        assert "bench.tg32.tps_x100 300" in result.stdout
        assert "bench.tg32.tps_x100 600" not in result.stdout
        assert "error: bench tg32 below minimum: actual=3.00 required=5.00" in result.stdout


def test_parser_requires_tg128_presence_for_full_grouped_perf() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "perf.stdout").write_text(
            "| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | SYCL       |   99 |  1 |           pp512 |       1036.72 ± 7.25 |\n"
        )
        result = run_parser_result(tmp / "perf.stdout", "--require-bench-test", "tg128")
        assert result.returncode != 0
        assert "bench.pp512.count 1" in result.stdout
        assert "error: required bench test missing: tg128" in result.stdout


def test_parser_accepts_tg128_presence_for_full_grouped_perf() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "perf.stdout").write_text(
            "| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | SYCL       |   99 |  1 |           pp512 |       1036.72 ± 7.25 |\n"
            "| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | SYCL       |   99 |  1 |           tg128 |         31.30 ± 0.22 |\n"
        )
        out = run_parser(tmp / "perf.stdout", "--require-bench-test", "tg128")
        assert "bench.tg128.count 1" in out


def test_parser_require_any_diag_path_accepts_any_aggressive_label() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "diag.stderr").write_text(
            "[MOE-GLU-Q8-DIAG] action=publish layer=0 device=1 rows=4 ne0=2880 event=1 "
            "path=aggressive-partial-fused-tg fused_candidate=1 fused_reject=none saved_launches=1\n"
        )
        out = run_parser(
            tmp,
            "--require-any-diag-path",
            "aggressive-partial,aggressive-partial-packed-q8-m4-artifact,aggressive-partial-fused-tg,aggressive-partial-soa-packed-q8-m4-artifact",
        )
        assert "diag.path.aggressive-partial-fused-tg 1" in out


def test_parser_accepts_aggressive_soa_m4_label_as_saved_launch_evidence() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "diag.stderr").write_text(
            "[MOE-GLU-Q8-DIAG] action=fused-store layer=0 device=1 rows=4 ne0=2880 event=1 "
            "path=aggressive-partial-soa-packed-q8-m4-artifact fused_candidate=1 fused_reject=none saved_launches=1\n"
        )
        out = run_parser(
            tmp,
            "--require-any-diag-path",
            "aggressive-partial,aggressive-partial-packed-q8-m4-artifact,aggressive-partial-fused-tg,aggressive-partial-soa-packed-q8-m4-artifact",
            "--require-aggressive-optimized-substrate",
        )
        assert "optimized.aggressive_substrate.true 1" in out
        assert "diag.aggressive_fused_saved_launches 1" in out
        assert "diag.path.aggressive-partial-soa-packed-q8-m4-artifact 1" in out


def test_parser_require_any_diag_path_rejects_unknown_non_aggressive_suffix() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "diag.stderr").write_text(
            "[MOE-GLU-Q8-DIAG] action=publish layer=0 device=1 rows=4 ne0=2880 event=1 "
            "path=direct-xmx-disabled fused_candidate=0 fused_reject=disabled saved_launches=0\n"
        )
        result = run_parser_result(tmp, "--require-any-diag-path", "direct-xmx")
        assert result.returncode == 12
        assert "diag.path.direct-xmx-disabled 1" in result.stdout
        assert "error: required diagnostic paths missing: direct-xmx" in result.stdout


def test_parser_require_any_diag_path_rejects_missing_aggressive_label() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "diag.stderr").write_text(
            "[MOE-GLU-Q8-DIAG] action=publish layer=0 device=1 rows=4 ne0=2880 event=1 "
            "path=partial-packed-q8-m2-device fused_candidate=1 fused_reject=none saved_launches=0\n"
        )
        result = run_parser_result(
            tmp,
            "--require-any-diag-path",
            "aggressive-partial,aggressive-partial-packed-q8-m4-artifact,aggressive-partial-fused-tg",
        )
        assert result.returncode == 12
        assert "error: required diagnostic paths missing" in result.stdout


def test_parser_require_bench_within_pct_passes_when_candidate_within_limit() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        candidate = tmp / "candidate.stdout"
        baseline = tmp / "safe.stdout"
        candidate.write_text(
            "| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | SYCL       |   99 |  1 |           pp512 |       1193.08 ± 1.00 |\n"
            "| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | SYCL       |   99 |  1 |           tg128 |         45.25 ± 0.20 |\n"
        )
        baseline.write_text(
            "| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | SYCL       |   99 |  1 |           pp512 |       1207.62 ± 1.00 |\n"
            "| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | SYCL       |   99 |  1 |           tg128 |         31.49 ± 0.20 |\n"
        )
        out = run_parser(
            candidate,
            "--require-bench-within-pct",
            "pp512",
            str(candidate),
            str(baseline),
            "5",
        )
        assert "bench.pp512.tps_x100 119308" in out


def test_parser_require_bench_within_pct_fails_when_candidate_regresses_too_far() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        candidate = tmp / "candidate.stdout"
        baseline = tmp / "safe.stdout"
        candidate.write_text(
            "| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | SYCL       |   99 |  1 |           pp512 |       1100.00 ± 1.00 |\n"
            "| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | SYCL       |   99 |  1 |           tg128 |         45.25 ± 0.20 |\n"
        )
        baseline.write_text(
            "| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | SYCL       |   99 |  1 |           pp512 |       1207.62 ± 1.00 |\n"
        )
        result = run_parser_result(
            candidate,
            "--require-bench-within-pct",
            "pp512",
            str(candidate),
            str(baseline),
            "5",
        )
        assert result.returncode == 10
        assert "bench pp512 regression too large" in result.stdout


def test_parser_require_bench_within_pct_rejects_missing_candidate_even_with_large_threshold() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        candidate = tmp / "missing-candidate.stdout"
        baseline = tmp / "safe.stdout"
        baseline.write_text(
            "| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | SYCL       |   99 |  1 |           pp512 |       1207.62 ± 1.00 |\n"
        )
        result = run_parser_result(
            baseline,
            "--require-bench-within-pct",
            "pp512",
            str(candidate),
            str(baseline),
            "99",
        )
        assert result.returncode == 10
        assert "error: candidate bench pp512 missing for regression check" in result.stdout


def test_parser_require_bench_within_pct_rejects_invalid_percent() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        candidate = tmp / "candidate.stdout"
        baseline = tmp / "safe.stdout"
        candidate.write_text(
            "| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | SYCL       |   99 |  1 |           pp512 |       1200.00 ± 1.00 |\n"
        )
        baseline.write_text(
            "| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | SYCL       |   99 |  1 |           pp512 |       1207.62 ± 1.00 |\n"
        )
        result = run_parser_result(
            candidate,
            "--require-bench-within-pct",
            "pp512",
            str(candidate),
            str(baseline),
            "100",
        )
        assert result.returncode == 10
        assert "error: max regression percent must be finite and in [0, 100): 100" in result.stdout


def test_parser_require_bench_within_pct_uses_best_not_sum_for_candidate_directory() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        candidate_dir = tmp / "candidate"
        baseline_dir = tmp / "baseline"
        candidate_dir.mkdir()
        baseline_dir.mkdir()
        candidate_line = (
            "| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | SYCL       |   99 |  1 |"
            "            tg32 |          3.00 ± 0.01 |\n"
        )
        (candidate_dir / "candidate_a.stdout").write_text(candidate_line)
        (candidate_dir / "candidate_b.stdout").write_text(candidate_line)
        (baseline_dir / "baseline.stdout").write_text(
            "| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | SYCL       |   99 |  1 |"
            "            tg32 |          5.00 ± 0.01 |\n"
        )
        result = run_parser_result(
            candidate_dir,
            "--require-bench-within-pct",
            "tg32",
            str(candidate_dir),
            str(baseline_dir),
            "20",
        )
        assert result.returncode == 10
        assert "bench.tg32.count 2" in result.stdout
        assert "bench.tg32.tps_x100 300" in result.stdout
        assert "bench.tg32.tps_x100 600" not in result.stdout
        assert "candidate=3.00 baseline=5.00 max_pct=20.00 floor=4.00" in result.stdout


def test_aggressive_suite_dry_run_contains_speed_and_portability_gates() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        logdir = pathlib.Path(tmp_raw) / "logs"
        result = subprocess.run(
            [
                "bash",
                str(HARNESS),
                "--mode",
                "aggressive-suite",
                "--dry-run",
                "--logdir",
                str(logdir),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        out = result.stdout
        assert "GGML_SYCL_MOE_AGGRESSIVE_TG=1" in out
        assert "GGML_SYCL_MOE_PARTIAL_DEVICE_GROUPING=1" in out
        assert "--require-bench-min tg128 45" in out
        assert "--require-bench-within-pct pp512" in out
        assert "--require-bench-within-pct tg128" in out
        assert "--require-any-diag-path" in out
        assert "--require-aggressive-optimized-substrate" in out
        assert "--forbid-diag-path split-sg16" in out
        assert "--forbid-diag-path grouped-packed-q8-m2-device" in out
        assert "b50_aggressive_count_output_check" in out
        assert "--require-generated-count-exact" in out
        assert "b580_aggressive_mistral_count" in out
        assert "b580_aggressive_mistral_count_output_check" in out
        assert "--require-mistral-count-prefix" in out
        assert "b580_aggressive_mistral_perf_check" in out
        perf_line = next(line for line in out.splitlines() if line.startswith("[b50_aggressive_perf]"))
        assert "GGML_SYCL_MOE_PARTIAL_DEVICE_GROUPING=1" in perf_line
        assert "GGML_SYCL_MOE_GLU_Q8_PUBLISH_DIAG=1" not in perf_line
        assert "GGML_SYCL_GRAPH_DIAG=1" not in perf_line


def test_b50_pp_materialize_tg_safe_dry_run_has_no_direct_final_envs() -> None:
    out = run_harness_dry_run("b50-pp-materialize-tg-safe")
    assert "GGML_SYCL_SELECTED_DPAS_MATERIALIZE=1" in out
    assert "GGML_SYCL_MOE_DOWN_SUM_DPAS_DIRECT_FINAL" not in out
    assert "--require-generated-count-exact" in out
    assert "--require-mxfp4-profile-evidence" in out
    assert "--forbid-down-dpas-direct-final" in out
    assert "--forbid-diag-path split-sg16" in out
    assert "--forbid-diag-path grouped-packed-q8-m2-device" in out
    assert "--require-bench-min tg128 45" in out or "--require-bench-min tg128=45" in out
    assert "--require-bench-min pp512 1100" in out or "--require-bench-min pp512=1100" in out




def test_no_arg_dry_run_defaults_to_safe_default_without_sequence_graphlet_unsafe_envs() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        logdir = pathlib.Path(tmp_raw) / "logs"
        result = subprocess.run(
            ["bash", str(HARNESS), "--dry-run", "--logdir", str(logdir)],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert result.returncode == 0, result.stdout
        out = result.stdout
        assert "[default_count]" in out
        assert "[default_bench]" in out
        assert "[sequence_graphlet_count]" not in out
        assert "GGML_SYCL_MOE_SEQUENCE_GRAPHLETS_UNSAFE_REPLAY" not in out
        assert "GGML_SYCL_MOE_SEQUENCE_GRAPHLETS_ALLOW_UNSAFE_RECORD" not in out


def test_explicit_sequence_graphlet_dry_run_contains_unsafe_envs() -> None:
    out = run_harness_dry_run("sequence-graphlet")
    assert "[sequence_graphlet_count]" in out
    assert "GGML_SYCL_MOE_SEQUENCE_GRAPHLETS_UNSAFE_REPLAY=1" in out
    assert "GGML_SYCL_MOE_SEQUENCE_GRAPHLETS_ALLOW_UNSAFE_RECORD=1" in out


def test_down_dpas_direct_final_dry_run_contains_envs_and_forbidden_paths() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        logdir = pathlib.Path(tmp_raw) / "logs"
        result = subprocess.run(
            [
                "bash",
                str(HARNESS),
                "--case",
                "down-dpas-direct-final",
                "--dry-run",
                "--logdir",
                str(logdir),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        out = result.stdout
        assert "GGML_SYCL_MOE_DOWN_SUM_DIRECT=1" in out
        assert "GGML_SYCL_MOE_DOWN_SUM_DPAS_DIRECT_FINAL=1" in out
        assert "GGML_SYCL_MOE_DOWN_SUM_DPAS_DIRECT_FINAL_I8=1" in out
        assert "GGML_SYCL_MOE_DOWN_SUM_DPAS_DIRECT_FINAL_DPAS=1" in out
        assert "GGML_SYCL_MOE_DECODE_DOWN_I8_SELECTED=1" in out
        assert "GGML_SYCL_SELECTED_DPAS_MATERIALIZE=1" in out
        assert "GGML_SYCL_MXFP4_TG_PROFILE=1" in out
        assert "--require-down-dpas-direct-final" in out
        assert "--forbid-diag-path split-sg16" in out
        assert "--forbid-diag-path grouped-packed-q8-m2-device" in out


def test_harness_dry_run_prints_kernel_runtime_metadata_capture_without_executing() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        logdir = pathlib.Path(tmp_raw) / "logs"
        result = subprocess.run(
            [
                "bash",
                str(HARNESS),
                "--case",
                "down-dpas-direct-final",
                "--dry-run",
                "--logdir",
                str(logdir),
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert result.returncode == 0, result.stdout
        out = result.stdout
        assert "[kernel_runtime_metadata]" in out
        assert "capture_kernel_runtime_metadata" in out
        assert "kernel_runtime_metadata.kernel-runtime.log" in out
        assert not (logdir / "kernel_runtime_metadata.kernel-runtime.log").exists()


def test_down_dpas_concurrency_dry_run_contains_envs_and_forbidden_paths() -> None:
    variants = [
        (
            "down-dpas-rank-parallel-atomic",
            "GGML_SYCL_MOE_DOWN_SUM_DPAS_DIRECT_FINAL_RANK_PARALLEL_ATOMIC=1",
            (
                "GGML_SYCL_MOE_DOWN_SUM_DPAS_DIRECT_FINAL_SCRATCH_REDUCE=1",
                "GGML_SYCL_MOE_DOWN_SUM_DPAS_DIRECT_FINAL_SAME_EXPERT_GROUPED=1",
            ),
        ),
        (
            "down-dpas-scratch-reduce",
            "GGML_SYCL_MOE_DOWN_SUM_DPAS_DIRECT_FINAL_SCRATCH_REDUCE=1",
            (
                "GGML_SYCL_MOE_DOWN_SUM_DPAS_DIRECT_FINAL_RANK_PARALLEL_ATOMIC=1",
                "GGML_SYCL_MOE_DOWN_SUM_DPAS_DIRECT_FINAL_SAME_EXPERT_GROUPED=1",
            ),
        ),
        (
            "down-dpas-same-expert-grouped",
            "GGML_SYCL_MOE_DOWN_SUM_DPAS_DIRECT_FINAL_SAME_EXPERT_GROUPED=1",
            (
                "GGML_SYCL_MOE_DOWN_SUM_DPAS_DIRECT_FINAL_RANK_PARALLEL_ATOMIC=1",
                "GGML_SYCL_MOE_DOWN_SUM_DPAS_DIRECT_FINAL_SCRATCH_REDUCE=1",
            ),
        ),
    ]
    for mode, required_env, forbidden_envs in variants:
        with tempfile.TemporaryDirectory() as tmp_raw:
            logdir = pathlib.Path(tmp_raw) / "logs"
            result = subprocess.run(
                [
                    "bash",
                    str(HARNESS),
                    "--mode",
                    mode,
                    "--dry-run",
                    "--logdir",
                    str(logdir),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            assert result.returncode == 0, result.stdout
            out = result.stdout
            assert "GGML_SYCL_MOE_DOWN_SUM_DPAS_DIRECT_FINAL=1" in out
            assert required_env in out
            for forbidden_env in forbidden_envs:
                assert forbidden_env not in out
            assert "--require-down-dpas-direct-final" in out
            assert "--forbid-diag-path split-sg16" in out
            assert "--forbid-diag-path grouped-packed-q8-m2-device" in out


def test_b50_profile_matrix_dry_run_orders_grouped_protection_before_full_grouped_perf() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        result = subprocess.run(
            [
                "bash",
                str(HARNESS),
                "--mode",
                "b50-profile-matrix",
                "--dry-run",
                "--logdir",
                tmp_raw,
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert result.returncode == 0, result.stdout
        out = result.stdout
        ordered_markers = [
            "[b50_grouped_decode_binary_label_check]",
            "[b50_grouped_decode_diag]",
            "[b50_grouped_decode_diag_partial_path_check]",
            "[b50_grouped_decode_diag_tg_floor_check]",
            "[b50_grouped_decode_perf]",
            "[b50_grouped_decode_perf_completion_check]",
        ]
        last_pos = -1
        for marker in ordered_markers:
            pos = out.find(marker)
            assert pos != -1, f"missing dry-run command {marker}:\n{out}"
            assert pos > last_pos, f"dry-run command out of order: {marker}:\n{out}"
            last_pos = pos
        assert out.find("[b50_grouped_decode_perf]") < out.find("[b50_default_perf]")
        assert "--require-bench-min tg32 5" in out

        perf_line = next(line for line in out.splitlines() if line.startswith("[b50_grouped_decode_perf]"))
        assert "timeout --kill-after=30s" in perf_line
        assert "GGML_SYCL_MOE_GROUPED_DECODE=1" in perf_line
        for forbidden in (
            "GGML_SYCL_GRAPH_DIAG",
            "GGML_SYCL_MOE_GLU_Q8_PUBLISH_DIAG",
            "GGML_SYCL_MOE_PROFILE",
            "GGML_SYCL_MXFP4_TG_PROFILE",
            "GGML_SYCL_MXFP4_PP_PROFILE",
        ):
            assert forbidden not in perf_line
        assert "[HARNESS-TIMEOUT] name=%s seconds=%s rc=%s" in HARNESS.read_text()


def test_parser_requires_generated_count_exact_output() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "count.stdout").write_text(
            "> Count from 1 to 5. Answer with only: 1, 2, 3, 4, 5\n\n: 1, 2, 3, 4, 5\n"
        )
        out = run_parser(tmp, "--require-generated-count-exact", "--require-no-fatal-markers")
        assert "generated.count_exact.true 1" in out
        assert "fatal.total 0" in out


def test_parser_accepts_promptless_generated_count_exact_output() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "count.stdout").write_text(": 1, 2, 3, 4, 5\n[ Prompt: 37.9 t/s | Generation: 5.5 t/s ]\n")
        out = run_parser(tmp, "--require-generated-count-exact", "--require-no-fatal-markers")
        assert "generated.count_exact.true 1" in out
        assert "fatal.total 0" in out


def test_parser_rejects_missing_generated_count_exact_output() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "count.stdout").write_text(
            "> Count from 1 to 5. Answer with only: 1, 2, 3, 4, 5\n\n: 1, 2, 3, 4, nope\n"
        )
        result = run_parser_result(tmp, "--require-generated-count-exact")
        assert result.returncode == 13
        assert "generated.count_exact.false 1" in result.stdout
        assert "error: generated count exact output mismatch present" in result.stdout


def test_parser_rejects_mixed_promptless_generated_count_exact_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "good.stdout").write_text(": 1, 2, 3, 4, 5\n")
        (tmp / "bad.stdout").write_text(": 1, 2, 3, 4, nope\n")
        result = run_parser_result(tmp, "--require-generated-count-exact")
        assert result.returncode == 13
        assert "generated.count_exact.true 1" in result.stdout
        assert "generated.count_exact.false 1" in result.stdout
        assert "error: generated count exact output mismatch present" in result.stdout


def test_parser_rejects_mixed_generated_count_exact_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "good.stdout").write_text(
            "> Count from 1 to 5. Answer with only: 1, 2, 3, 4, 5\n\n: 1, 2, 3, 4, 5\n"
        )
        (tmp / "bad.stdout").write_text(
            "> Count from 1 to 5. Answer with only: 1, 2, 3, 4, 5\n\n: 1, 2, 3, 4, nope\n"
        )
        result = run_parser_result(tmp, "--require-generated-count-exact")
        assert result.returncode == 13
        assert "generated.count_exact.true 1" in result.stdout
        assert "generated.count_exact.false 1" in result.stdout
        assert "error: generated count exact output mismatch present" in result.stdout


def test_parser_requires_mistral_count_prefix_output() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "mistral.stdout").write_text(
            "1, 2, 3, 4, 5,\n6, 7, 8, 9, 10, 11, 12\n[ Prompt: 1200.0 t/s | Generation: 86.0 t/s ]\n"
        )
        out = run_parser(tmp, "--require-mistral-count-prefix", "--require-no-fatal-markers")
        assert "generated.mistral_count_prefix.true 1" in out
        assert "fatal.total 0" in out


def test_parser_rejects_missing_mistral_count_prefix_output() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "mistral_bad.stdout").write_text(
            "1, 2, 3, 4, 5,\n99, 100\n[ Prompt: 1200.0 t/s | Generation: 86.0 t/s ]\n"
        )
        result = run_parser_result(tmp, "--require-mistral-count-prefix")
        assert result.returncode == 14
        assert "generated.mistral_count_prefix.false 1" in result.stdout
        assert "error: Mistral count prefix output mismatch present" in result.stdout


def test_parser_rejects_mixed_mistral_count_prefix_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "mistral_good.stdout").write_text(
            "1, 2, 3, 4, 5,\n6, 7, 8, 9, 10, 11, 12\n[ Prompt: 1200.0 t/s | Generation: 86.0 t/s ]\n"
        )
        (tmp / "mistral_bad.stdout").write_text(
            "1, 2, 3, 4, 5,\n99, 100\n[ Prompt: 1200.0 t/s | Generation: 86.0 t/s ]\n"
        )
        result = run_parser_result(tmp, "--require-mistral-count-prefix")
        assert result.returncode == 14
        assert "generated.mistral_count_prefix.true 1" in result.stdout
        assert "generated.mistral_count_prefix.false 1" in result.stdout
        assert "error: Mistral count prefix output mismatch present" in result.stdout


def test_parser_counts_b580_mistral_prefix_after_prompt_echo() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "mistral.stdout").write_text(
            "1, 2, 3, 4, 5,\n6, 7, 8, 9, 10, 11, 12\n[ Prompt: 1200.0 t/s | Generation: 86.0 t/s ]\n"
        )
        out = run_parser(tmp)
        assert "generated.mistral_count_prefix.true 1" in out


def test_parser_rejects_b580_mistral_prompt_echo_only_or_wrong_continuation() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "mistral_bad.stdout").write_text(
            "1, 2, 3, 4, 5,\n99, 100\n[ Prompt: 1200.0 t/s | Generation: 86.0 t/s ]\n"
        )
        out = run_parser(tmp)
        assert "generated.mistral_count_prefix.false 1" in out
        assert "generated.mistral_count_prefix.true" not in out


def test_parser_reports_fatal_markers() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "bench.stderr").write_text(
            "UR_RESULT_ERROR_DEVICE_LOST\n"
            "live allocation still retained\n"
            "reset-with-live-handle\n"
            "graph replay exception\n"
            "replay-exception\n"
            "fusion-exception\n"
            "[SYCL-WATCHDOG] No GPU progress for 30000 ms (timeout 30000 ms, 1 devices known).\n"
        )
        out = run_parser(tmp)
        assert "fatal.device_lost 1" in out
        assert "fatal.live_allocation 1" in out
        assert "fatal.reset_with_live_handle 1" in out
        assert "fatal.graph_replay_exception 1" in out
        assert "fatal.replay_exception 1" in out
        assert "fatal.fusion_exception 1" in out
        assert "fatal.watchdog 1" in out
        assert "fatal.total 7" in out


def test_parser_reports_process_crashes_as_fatal_markers() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "crash.stderr").write_text(
            "Segmentation fault (core dumped)\n"
            "Aborted (core dumped)\n"
            "SIGSEGV\n"
            "SIGABRT\n"
        )
        result = run_parser_result(tmp, "--require-no-fatal-markers")
        assert result.returncode != 0
        assert "fatal.segmentation_fault 2" in result.stdout
        assert "fatal.abort 2" in result.stdout
        assert "fatal.core_dump 2" in result.stdout
        assert "fatal.total 6" in result.stdout
        assert "error: fatal markers present" in result.stdout


def test_parser_reports_harness_timeout_as_fatal_marker() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "diag.stderr").write_text(
            "[HARNESS-TIMEOUT] name=b50_grouped_decode_diag seconds=900 rc=124\n"
        )
        result = run_parser_result(tmp, "--require-no-fatal-markers")
        assert result.returncode != 0
        assert "fatal.harness_timeout 1" in result.stdout
        assert "fatal.total 1" in result.stdout
        assert "error: fatal markers present" in result.stdout


def test_parser_reports_generic_timeout_text_as_fatal_marker() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "diag.stderr").write_text(
            "timeout: the monitored command dumped core\n"
            "Command timed out after 3600000 milliseconds\n"
        )
        result = run_parser_result(tmp, "--require-no-fatal-markers")
        assert result.returncode != 0
        assert "fatal.command_timeout 2" in result.stdout
        assert "fatal.total 2" in result.stdout
        assert "error: fatal markers present" in result.stdout


def test_parser_does_not_treat_reject_watchdog_reason_as_fatal() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "diag.stderr").write_text(
            "[XMX-MOE-REJECT] tensor=blk.0.ffn_gate_exps.weight type=mxfp4 "
            "reason=mxfp4-pp-submit-watchdog src0=[2880,2880,32,1]\n"
        )
        out = run_parser(tmp, "--require-no-fatal-markers")
        assert "fatal.total 0" in out
        assert "fatal.watchdog" not in out
        assert "fatal.harness_timeout" not in out
        assert "fatal.command_timeout" not in out
        assert "reject.reason.mxfp4-pp-submit-watchdog 1" in out


def test_parser_require_no_fatal_markers_fails_even_when_optimized() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "diag.stderr").write_text(
            "[GRAPH-DIAG] phase=TG sequence_graphlet_segmented_replay_calls=1\n"
            "UR_RESULT_ERROR_DEVICE_LOST\n"
            "replay-exception\n"
            "fusion-exception\n"
            "reset-with-live-handle\n"
        )
        result = run_parser_result(
            tmp,
            "--require-default-fast-path-optimized",
            "--require-no-fatal-markers",
        )
        assert result.returncode != 0
        assert "optimized.default_fast_path.true 1" in result.stdout
        assert "fatal.device_lost 1" in result.stdout
        assert "fatal.replay_exception 1" in result.stdout
        assert "fatal.fusion_exception 1" in result.stdout
        assert "fatal.reset_with_live_handle 1" in result.stdout
        assert "error: fatal markers present" in result.stdout


def test_parser_does_not_treat_headroom_as_oom() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "diag.stderr").write_text(
            "[SYCL-SEG-MOE-POLICY] free_vram=397.9MB headroom_ok=0 moe_rerecord=1\n"
        )
        out = run_parser(tmp)
        assert "fatal.total 0" in out
        assert "fatal.out_of_device_memory" not in out


def test_parser_directory_skips_activation_check_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "bench.stderr").write_text(
            "[SYCL-SEG-MOE-POLICY] free_vram=397.9MB headroom_ok=0 moe_rerecord=1\n"
        )
        (tmp / "b50_default_candidate_activation_check.stdout").write_text(
            "fatal.out_of_device_memory 99\nerror: fatal markers present\n"
        )
        out = run_parser(tmp)
        assert "activation_check.stdout" not in out
        assert "fatal.total 0" in out
        assert "fatal.out_of_device_memory" not in out


def test_parser_extracts_mxfp4_tg_profile_counters() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "profile.stderr").write_text(
            "[MXFP4-MOE-TG-PROFILE] calls=72 soa=66 coalesced=0 aos=0 dpas=0 i8=6 "
            "entries=288 batches=288 total=6.773 ms quant=0.180 ms artifact=0.000 ms "
            "batch_ids=0.000 ms kernel=6.594 ms gateup_glu=0.336 ms/48 down=6.438 ms/24 "
            "other=0.000 ms/0 per_call total=94.036 us quant=2.395 us "
            "batch_ids=0.000 us kernel=91.641 us per_entry kernel=22.910 us\n"
        )
        out = run_parser(tmp)
        assert "profile.mxfp4_tg.calls 72" in out
        assert "profile.mxfp4_tg.layout.soa 66" in out
        assert "profile.mxfp4_tg.layout.dpas 0" in out
        assert "profile.mxfp4_tg.total_ms_x1000 6773" in out
        assert "profile.mxfp4_tg.kernel_ms_x1000 6594" in out
        assert "profile.mxfp4_tg.gateup_glu_ms_x1000 336" in out
        assert "profile.mxfp4_tg.down_ms_x1000 6438" in out


def test_parser_extracts_mxfp4_tg_pack_profile_counter() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "profile.stderr").write_text(
            "[MXFP4-MOE-TG-PROFILE] calls=72 soa=0 coalesced=0 aos=0 dpas=48 i8=6 "
            "entries=288 batches=288 total=7.100 ms quant=0.200 ms artifact=0.100 ms "
            "batch_ids=0.000 ms pack=1.250 ms kernel=5.550 ms gateup_glu=6.800 ms/48 down=0.700 ms/24 "
            "other=0.000 ms/0 per_call total=98.611 us quant=2.778 us "
            "batch_ids=0.000 us kernel=77.083 us per_entry kernel=19.271 us last_path=packed-q8-m2\n"
        )
        out = run_parser(tmp)
        assert "profile.mxfp4_tg.pack_ms_x1000 1250" in out
        assert "profile.mxfp4_tg.path.packed-q8-m2 1" in out


def test_parser_extracts_mxfp4_tg_down_pack_profile_counter() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "profile.stderr").write_text(
            "[MXFP4-MOE-TG-PROFILE] calls=24 soa=0 coalesced=0 aos=0 dpas=0 i8=24 "
            "entries=96 batches=96 total=1.100 ms quant=0.000 ms artifact=0.000 ms "
            "batch_ids=0.000 ms pack=0.250 ms kernel=0.850 ms gateup_glu=0.000 ms/0 down=1.100 ms/24 "
            "other=0.000 ms/0 per_call total=45.833 us quant=0.000 us "
            "batch_ids=0.000 us kernel=35.417 us per_entry kernel=8.854 us last_path=down-grouped-packed-q8\n"
        )
        out = run_parser(tmp)
        assert "profile.mxfp4_tg.pack_ms_x1000 250" in out
        assert "profile.mxfp4_tg.path.down-grouped-packed-q8 1" in out


def test_tg_profile_source_records_down_pack_timing() -> None:
    mmvq = (ROOT / "ggml" / "src" / "ggml-sycl" / "mmvq.cpp").read_text()
    down_begin = mmvq.index("bool mmvq_moe_batched_dispatch_down_from_cached_q8_mxfp4")
    down_end = mmvq.index("class mxfp4_down_sum_q8_soa_kernel", down_begin)
    down_dispatch = mmvq[down_begin:down_end]
    route_begin = down_dispatch.index("if (!chunked_row_limit && mxfp4_i8_grouped_pack_q8_enabled(n_tokens))")
    route_end = down_dispatch.index('profile_path = "down-grouped-packed-q8";', route_begin)
    down_pack_route = down_dispatch[route_begin:route_end]
    assert "if (detail_profile)" in down_pack_route
    assert "profile_pack_event     = pack_event" in down_pack_route
    assert "if (pp_profile)" not in down_pack_route
    record_begin = down_dispatch.index("if (mmvq_moe_tg_profile_enabled())")
    record_end = down_dispatch.index("if (pp_profile)", record_begin)
    down_tg_record = down_dispatch[record_begin:record_end]
    assert "profile_path, pack_us" in down_tg_record


def test_down_dpas_direct_final_labels_are_counted() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "profile.stderr").write_text(
            "[MXFP4-MOE-TG-PROFILE] kind=down layout=mxfp4_i8 "
            "path=down-dpas-direct-final-i8 entries=4 total_ms=3.0\n"
            "[MXFP4-MOE-TG-PROFILE] kind=down layout=mxfp4_dpas "
            "path=down-dpas-direct-final-dpas entries=4 total_ms=2.5\n"
        )
        out = run_parser(tmp)
        assert "diag.path.down-dpas-direct-final-i8 1" in out
        assert "diag.path.down-dpas-direct-final-dpas 1" in out


def test_parser_require_mxfp4_profile_evidence_rejects_empty_log() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        (tmp / "empty.stderr").write_text("")
        result = run_parser_result(tmp, "--require-mxfp4-profile-evidence")
        assert result.returncode == 16
        assert "error: MXFP4 profile evidence missing" in result.stdout


def test_parser_require_mxfp4_profile_evidence_rejects_diag_path_only_log() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        (tmp / "diag.stderr").write_text(
            "[MOE-GLU-Q8-DIAG] action=publish layer=0 device=1 rows=4 ne0=2880 event=1 "
            "path=packed-q8-m2 fused_candidate=0 fused_reject=no-kernel-q8 saved_launches=0\n"
        )
        result = run_parser_result(
            tmp,
            "--require-mxfp4-profile-evidence",
            "--require-diag-path",
            "packed-q8-m2",
        )
        assert result.returncode == 16
        assert "diag.path.packed-q8-m2 1" in result.stdout
        assert "error: MXFP4 profile evidence missing" in result.stdout


def test_parser_require_mxfp4_profile_evidence_accepts_pp_profile_only() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        (tmp / "profile.stderr").write_text(
            "[MXFP4-MOE-PP-PROFILE] calls=72 gateup=48 down=24 entries=288 batches=288 "
            "paths packed=48 direct=24 chunked=0 other=0 total=5.250 ms "
            "quant=0.100 ms group_host=0.050 ms group_copy=0.040 ms pack=0.200 ms "
            "kernel_event=4.700 ms kernel_wall=4.900 ms artifact=0.160 ms "
            "gateup_total=3.800 ms down_total=1.450 ms per_call_total=72.917 us "
            "per_entry_kernel=16.319 us last_path=packed-q8-m2\n"
        )
        out = run_parser(tmp, "--require-mxfp4-profile-evidence")
        assert "profile.mxfp4_pp.calls 72" in out
        assert "profile.mxfp4_pp.path.packed-q8-m2 1" in out


def test_parser_forbid_down_dpas_direct_final_rejects_any_variant() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        (tmp / "profile.stderr").write_text(
            "[MXFP4-MOE-TG-PROFILE] calls=1 soa=1 coalesced=0 aos=0 dpas=0 i8=0 "
            "total=1.0 ms quant=0.1 ms artifact=0.1 ms batch_ids=0.1 ms kernel=0.7 ms "
            "gateup_glu=0.2 ms down=0.5 ms entries=4 batches=4\n"
            "[MXFP4-MOE-TG-PROFILE] path=down-dpas-direct-final-rank-parallel-scratch entries=4 batches=4 layout=mxfp4_i8\n"
        )
        result = run_parser_result(tmp, "--require-mxfp4-profile-evidence", "--forbid-down-dpas-direct-final")
        assert result.returncode == 17
        assert "error: forbidden down-dpas-direct-final path present" in result.stdout


def test_parser_split_profile_evidence_accepts_safe_tg_profile() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        (tmp / "profile.stderr").write_text(
            "[MXFP4-MOE-TG-PROFILE] calls=72 soa=66 coalesced=0 aos=0 dpas=0 i8=6 "
            "total=6.773 ms quant=0.122 ms artifact=0.000 ms batch_ids=0.057 ms kernel=6.594 ms "
            "gateup_glu=0.336 ms down=6.438 ms entries=288 batches=288\n"
        )
        out = run_parser(tmp, "--require-mxfp4-profile-evidence", "--forbid-down-dpas-direct-final")
        assert "profile.mxfp4_tg.calls 72" in out
        assert "diag.down_dpas_direct_final.present.false 1" in out


def test_parser_forbid_down_dpas_direct_final_rejects_unknown_suffixed_label() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        (tmp / "profile.stderr").write_text(
            "[MXFP4-MOE-TG-PROFILE] path=down-dpas-direct-final-experimental entries=4 total_ms=1.0\n"
        )
        result = run_parser_result(tmp, "--forbid-down-dpas-direct-final")
        assert result.returncode == 17
        assert "diag.path.down-dpas-direct-final-experimental 1" in result.stdout
        assert "diag.down_dpas_direct_final.present.true 1" in result.stdout
        assert "error: forbidden down-dpas-direct-final path present: count=1" in result.stdout


def test_require_down_dpas_direct_final_fails_when_missing() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "profile.stderr").write_text(
            "[MXFP4-MOE-TG-PROFILE] kind=down layout=soa path=packed-q8-m2 entries=4 total_ms=6.0\n"
        )
        result = run_parser_result(tmp, "--require-down-dpas-direct-final")
        assert result.returncode != 0
        assert "diag.path.packed-q8-m2 1" in result.stdout
        assert "error: required down-dpas-direct-final path was not observed" in result.stdout


def test_require_down_dpas_direct_final_rejects_unknown_suffixed_label() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "profile.stderr").write_text(
            "[MXFP4-MOE-TG-PROFILE] path=down-dpas-direct-final-disabled entries=4 total_ms=1.0\n"
        )
        result = run_parser_result(tmp, "--require-down-dpas-direct-final")
        assert result.returncode == 12
        assert "diag.path.down-dpas-direct-final-disabled 1" in result.stdout
        assert "diag.down_dpas_direct_final.present.true 1" in result.stdout
        assert "diag.down_dpas_direct_final.success.false 1" in result.stdout
        assert "error: required down-dpas-direct-final path was not observed" in result.stdout


def test_require_down_dpas_direct_final_accepts_i8_or_dpas() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "profile.stderr").write_text(
            "[MXFP4-MOE-TG-PROFILE] kind=down layout=mxfp4_i8 "
            "path=down-dpas-direct-final-i8 entries=4 total_ms=3.0\n"
        )
        out = run_parser(tmp, "--require-down-dpas-direct-final")
        assert "diag.path.down-dpas-direct-final-i8 1" in out


def test_require_down_dpas_direct_final_accepts_concurrency_variants() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "diag.stderr").write_text(
            "[MXFP4-MOE-TG-PROFILE] path=down-dpas-direct-final-rank-parallel-atomic entries=4 total_ms=1.0\n"
            "[MXFP4-MOE-TG-PROFILE] path=down-dpas-direct-final-rank-parallel-scratch entries=4 total_ms=1.1\n"
            "[MXFP4-MOE-TG-PROFILE] path=down-dpas-direct-final-same-expert-grouped entries=4 total_ms=1.2\n"
        )
        out = run_parser(tmp, "--require-down-dpas-direct-final")
        assert "diag.path.down-dpas-direct-final-rank-parallel-atomic 1" in out
        assert "diag.path.down-dpas-direct-final-rank-parallel-scratch 1" in out
        assert "diag.path.down-dpas-direct-final-same-expert-grouped 1" in out


def test_parser_counts_unified_cache_stats() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "cache.stderr").write_text(
            "[UNIFIED-CACHE-STATS] tag=teardown device=1 raw_device_alloc_calls=2 raw_device_alloc_bytes=12288 "
            "host_fallback_attempts=1 host_fallback_bytes=16384\n"
        )
        out = run_parser(tmp)
        assert "cache.raw_device_alloc_calls 2" in out
        assert "cache.raw_device_alloc_bytes 12288" in out
        assert "cache.host_fallback_attempts 1" in out
        assert "cache.host_fallback_bytes 16384" in out


def test_parser_counts_kernel_runtime_metadata() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "metadata.log").write_text(
            "[SYCL-KERNEL-RUNTIME] uname=Linux kainlan-server 7.1.2-070102-generic\n"
            "[SYCL-KERNEL-RUNTIME] cmdline=iommu=pt xe.probe_display=0\n"
            "[SYCL-KERNEL-RUNTIME] modinfo.xe filename: /lib/modules/7.1.2/kernel/drivers/gpu/drm/xe/xe.ko.zst\n"
            "[SYCL-KERNEL-RUNTIME] config.CONFIG_DRM_XE=m\n"
            "[SYCL-KERNEL-RUNTIME] config.CONFIG_DRM_XE_GPUSVM=y\n"
            "[SYCL-KERNEL-RUNTIME] config.CONFIG_DRM_XE_PAGEMAP=y\n"
            "[SYCL-KERNEL-RUNTIME] config.CONFIG_HMM_MIRROR=y\n"
            "[SYCL-KERNEL-RUNTIME] config.CONFIG_DEVICE_PRIVATE=y\n"
        )
        out = run_parser(tmp)
        assert "kernel_runtime.lines 8" in out
        assert "kernel_runtime.uname 1" in out
        assert "kernel_runtime.cmdline 1" in out
        assert "kernel_runtime.modinfo.xe 1" in out
        assert "kernel_runtime.config.CONFIG_DRM_XE 1" in out
        assert "kernel_runtime.config.CONFIG_DRM_XE_GPUSVM 1" in out
        assert "kernel_runtime.config.CONFIG_DRM_XE_PAGEMAP 1" in out
        assert "kernel_runtime.config.CONFIG_HMM_MIRROR 1" in out
        assert "kernel_runtime.config.CONFIG_DEVICE_PRIVATE 1" in out


def test_parser_aggressive_optimized_substrate_rejects_direct_only() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "diag.stderr").write_text(
            "[GRAPH-DIAG] phase=TG sequence_graphlet_direct_replay_calls=3816 "
            "sequence_graphlet_segmented_replay_calls=0 sequence_graphlet_replay=3816\n"
        )
        out = run_parser(tmp)
        assert "optimized.aggressive_substrate.false 1" in out
        result = run_parser_result(tmp, "--require-aggressive-optimized-substrate")
        assert result.returncode == 11
        assert "error: aggressive optimized substrate missing" in result.stdout


def test_parser_aggressive_optimized_substrate_accepts_segmented_replay() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "diag.stderr").write_text(
            "[GRAPH-DIAG] phase=TG sequence_graphlet_direct_replay_calls=12 "
            "sequence_graphlet_segmented_replay_calls=48 sequence_graphlet_replay=60\n"
        )
        out = run_parser(tmp, "--require-aggressive-optimized-substrate")
        assert "optimized.aggressive_substrate.true 1" in out
        assert "counter.sequence_graphlet_segmented_replay_calls 48" in out


def test_parser_aggressive_optimized_substrate_accepts_aggressive_fused_saved_launch() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "diag.stderr").write_text(
            "[GRAPH-DIAG] phase=TG sequence_graphlet_direct_replay_calls=48 "
            "sequence_graphlet_segmented_replay_calls=0 sequence_graphlet_replay=48\n"
            "[MOE-GLU-Q8-DIAG] action=fused-store layer=0 device=1 rows=4 ne0=2880 event=1 "
            "path=aggressive-partial-fused-tg fused_candidate=1 fused_reject=none saved_launches=1\n"
        )
        out = run_parser(tmp, "--require-aggressive-optimized-substrate")
        assert "optimized.aggressive_substrate.true 1" in out
        assert "diag.aggressive_fused_saved_launches 1" in out
        assert "diag.path.aggressive-partial-fused-tg 1" in out


def test_parser_aggressive_optimized_substrate_accepts_direct_xmx_path() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "diag.stderr").write_text(
            "[MOE-GLU-Q8-DIAG] action=publish layer=0 device=1 rows=4 ne0=2880 event=1 "
            "path=direct-xmx fused_candidate=1 fused_reject=not-used saved_launches=0 publish_us=2.5\n"
        )
        out = run_parser(tmp, "--require-aggressive-optimized-substrate")
        assert "optimized.aggressive_substrate.true 1" in out
        assert "diag.path.direct-xmx 1" in out


def test_parser_forbids_mixed_direct_xmx_and_split_sg16_when_requested() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "diag.stderr").write_text(
            "[MOE-GLU-Q8-DIAG] action=publish layer=0 device=1 rows=4 ne0=2880 event=1 "
            "path=direct-xmx fused_candidate=1 fused_reject=not-used saved_launches=0 publish_us=2.5\n"
            "[MOE-GLU-Q8-DIAG] action=fused-store layer=1 device=1 rows=4 ne0=2880 event=1 "
            "path=split-sg16 fused_candidate=1 fused_reject=none saved_launches=1\n"
        )
        result = run_parser_result(
            tmp,
            "--require-aggressive-optimized-substrate",
            "--require-any-diag-path",
            "direct-xmx",
            "--forbid-diag-path",
            "split-sg16",
        )
        assert result.returncode == 7
        assert "error: forbidden diagnostic path present: split-sg16" in result.stdout


def test_parser_aggressive_optimized_substrate_rejects_non_aggressive_saved_launch() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "diag.stderr").write_text(
            "[GRAPH-DIAG] phase=TG sequence_graphlet_direct_replay_calls=48 "
            "sequence_graphlet_segmented_replay_calls=0 sequence_graphlet_replay=48\n"
            "[MOE-GLU-Q8-DIAG] action=fused-store layer=0 device=1 rows=4 ne0=2880 event=1 "
            "path=partial-packed-q8-m2-device fused_candidate=1 fused_reject=none saved_launches=1\n"
        )
        result = run_parser_result(tmp, "--require-aggressive-optimized-substrate")
        assert result.returncode == 11
        assert "diag.path.partial-packed-q8-m2-device 1" in result.stdout
        assert "diag.aggressive_fused_saved_launches" not in result.stdout


def test_default_fast_path_optimized_rejects_direct_only_sequence_replay() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "diag.stderr").write_text(
            "[GRAPH-DIAG] phase=TG sequence_graphlet_replay=10 "
            "sequence_graphlet_submit_calls=10 sequence_graphlet_direct_replay_calls=10 "
            "sequence_graphlet_segmented_replay_calls=0 block_graphlet_replay=0 "
            "block_replay=0 attempt_fusion=0\n"
        )
        out = run_parser(tmp)
        assert "optimized.default_fast_path.false 1" in out
        result = run_parser_result(tmp, "--require-default-fast-path-optimized")
        assert result.returncode != 0
        assert "direct sequence replay alone is insufficient" in result.stdout


def test_default_fast_path_optimized_accepts_segmented_sequence_replay() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "diag.stderr").write_text(
            "[GRAPH-DIAG] phase=TG sequence_graphlet_replay=10 sequence_graphlet_submit_calls=4 "
            "sequence_graphlet_direct_replay_calls=6 sequence_graphlet_segmented_replay_calls=4\n"
        )
        out = run_parser(tmp, "--require-default-fast-path-optimized")
        assert "optimized.default_fast_path.true 1" in out


def test_default_fast_path_optimized_accepts_block_replay() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "diag.stderr").write_text(
            "[GRAPH-DIAG] phase=TG sequence_graphlet_replay=10 sequence_graphlet_submit_calls=10 "
            "sequence_graphlet_direct_replay_calls=10 block_graphlet_replay=2\n"
        )
        out = run_parser(tmp, "--require-default-fast-path-optimized")
        assert "optimized.default_fast_path.true 1" in out


def test_default_fast_path_optimized_accepts_proven_direct_final_saved_submit() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "diag.stderr").write_text(
            "[GRAPH-DIAG] phase=TG sequence_graphlet_replay=10 sequence_graphlet_submit_calls=10 "
            "sequence_graphlet_direct_replay_calls=10 direct_final_saved_submits=3\n"
        )
        out = run_parser(tmp, "--require-default-fast-path-optimized")
        assert "optimized.default_fast_path.true 1" in out



def test_parser_requires_single_xmx_gateup_evidence() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "profile.stderr").write_text(
            "[PLACEMENT-MOE] single_xmx_gateup=1 tensor=blk.0.ffn_gate_exps.weight layout=xmx_tiled soa_alternate=0\n"
            "[MOE-PHASE-LAYOUT] tensor=blk.0.ffn_gate_exps.weight target=xmx_tiled single_xmx_gateup=1 materialized=32/32 complete=1\n"
            "[MXFP4-MOE-TG-PROFILE] calls=72 soa=0 coalesced=0 aos=0 dpas=48 i8=0 total=6.000 ms quant=0.100 ms "
            "artifact=0.100 ms batch_ids=0.000 ms kernel=5.500 ms gateup_glu=4.100 ms/48 down=0.700 ms/24 "
            "last_path=xmx-tiled-single-gateup\n"
            "[MXFP4-MOE-PP-PROFILE] calls=24 gateup=48 down=24 entries=8192 batches=8192 last_path=xmx-tiled-single-gateup\n"
        )
        out = run_parser(tmp, "--require-single-xmx-gateup")
        assert "placement.single_xmx_gateup 1" in out
        assert "phase.single_xmx_gateup.complete 1" in out
        assert "profile.mxfp4_tg.path.xmx-tiled-single-gateup 1" in out
        assert "profile.mxfp4_pp.path.xmx-tiled-single-gateup 1" in out


def test_parser_rejects_missing_single_xmx_gateup_when_required() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "profile.stderr").write_text(
            "[MXFP4-MOE-TG-PROFILE] calls=72 soa=18 coalesced=0 aos=0 dpas=48 i8=0 total=6.000 ms "
            "quant=0.100 ms artifact=0.100 ms batch_ids=0.000 ms kernel=5.500 ms gateup_glu=5.700 ms/48 "
            "down=0.700 ms/24 last_path=packed-q8-m2\n"
        )
        result = run_parser_result(tmp, "--require-single-xmx-gateup")
        assert result.returncode != 0
        assert "error: single XMX_TILED gate/up evidence missing" in result.stdout


def test_parser_rejects_single_xmx_without_direct_pp_and_tg_paths() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "profile.stderr").write_text(
            "[PLACEMENT-MOE] single_xmx_gateup=1 tensor=blk.0.ffn_gate_exps.weight layout=xmx_tiled soa_alternate=0\n"
            "[MOE-PHASE-LAYOUT] tensor=blk.0.ffn_gate_exps.weight target=soa single_xmx_gateup=0 materialized=32/32 complete=1\n"
            "[MXFP4-MOE-TG-PROFILE] calls=72 soa=18 coalesced=0 aos=0 dpas=48 i8=0 total=6.000 ms "
            "quant=0.100 ms artifact=0.100 ms batch_ids=0.000 ms kernel=5.500 ms gateup_glu=5.700 ms/48 "
            "down=0.700 ms/24 last_path=packed-q8-m2\n"
        )
        result = run_parser_result(tmp, "--require-single-xmx-gateup")
        assert result.returncode != 0
        assert "error: single XMX_TILED gate/up profile path evidence missing" in result.stdout


def test_parser_forbids_gateup_soa_fallback_in_single_xmx_mode() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "profile.stderr").write_text(
            "[PLACEMENT-MOE] PP primary gate/up layouts prepared before packing: device=1 promoted_soa=64 no_layout_bytes=0\n"
            "[MXFP4-MOE-TG-PROFILE] calls=72 soa=18 coalesced=0 aos=0 dpas=48 i8=0 total=6.000 ms "
            "quant=0.100 ms artifact=0.100 ms batch_ids=0.000 ms kernel=5.500 ms gateup_glu=5.700 ms/48 "
            "down=0.700 ms/24 last_path=packed-q8-m2\n"
        )
        result = run_parser_result(tmp, "--forbid-gateup-soa-fallback")
        assert result.returncode != 0
        assert "error: gate/up SOA fallback present in single-layout proof mode" in result.stdout


def test_harness_dry_run_single_xmx_gateup_mode_includes_required_gates() -> None:
    out = run_harness_dry_run("single-xmx-gateup")
    assert "GGML_SYCL_MOE_GATEUP_SINGLE_XMX=1" in out
    assert "--require-single-xmx-gateup" in out
    assert "--forbid-gateup-soa-fallback" in out


def test_parser_forbids_explicit_single_xmx_gateup_zero() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        (tmp / "profile.stderr").write_text(
            "[PLACEMENT-MOE] single_xmx_gateup=0 tensor=blk.0.ffn_gate_exps.weight layout=soa soa_alternate=1\n"
        )
        result = run_parser_result(tmp, "--forbid-gateup-soa-fallback")
        assert result.returncode != 0
        assert "placement.single_xmx_gateup.0 1" in result.stdout
        assert "error: gate/up SOA fallback present in single-layout proof mode" in result.stdout


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
