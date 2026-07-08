#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check-sycl-dpas-capabilities.sh"

DPAS_CONTRACT = (
    'static_assert(RepeatCount >= 1 && RepeatCount <= 8, "Repeat count must be within 1 to 8 range");\n'
    'static_assert(SystolicDepth == 8, "Systolic depth must be equal to 8");\n'
    'static_assert(ExecutionSize == 8 || (!IsDPASW && ExecutionSize == 16), "Execution size must be 8 or 16 for DPAS and 8 for DPASW.");\n'
)


def parse_output(output: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        key, sep, value = line.partition("=")
        assert sep == "=", f"line is not key=value: {line!r}"
        assert key not in values, f"duplicate key in checker output: {key}"
        values[key] = value
    return values


def run(header: pathlib.Path) -> dict[str, str]:
    output = subprocess.check_output(["bash", str(SCRIPT), str(header)], text=True, cwd=ROOT)
    return parse_output(output)


def run_result(header: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), str(header)],
        text=True,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def run_relative(cwd: pathlib.Path, header: str) -> dict[str, str]:
    output = subprocess.check_output(["bash", str(SCRIPT), header], text=True, cwd=cwd)
    return parse_output(output)


def test_capability_checker_reports_current_dpas_contract() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        header = pathlib.Path(tmp_raw) / "dpas.hpp"
        header.write_text(DPAS_CONTRACT)
        values = run(header)
        assert values == {
            "dpas.header": str(header.resolve()),
            "dpas.common_header": "missing",
            "dpas.repeat_count.max": "8",
            "dpas.systolic_depth": "8",
            "dpas.exec_size.allowed": "8,16",
            "dpas.bdpas.present": "0",
            "dpas.fp4_e2m1.present": "0",
        }


def test_capability_checker_reports_bdpas_when_header_mentions_it() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        header = pathlib.Path(tmp_raw) / "dpas.hpp"
        header.write_text(
            DPAS_CONTRACT
            + 'template <typename T> auto bdpas(T x) { return x; }\n'
            + 'enum class dpas_argument_type { e2m1 };\n'
        )
        values = run(header)
        assert values["dpas.repeat_count.max"] == "8"
        assert values["dpas.systolic_depth"] == "8"
        assert values["dpas.exec_size.allowed"] == "8,16"
        assert values["dpas.bdpas.present"] == "1"
        assert values["dpas.fp4_e2m1.present"] == "1"


def test_capability_checker_rejects_missing_header() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        missing = pathlib.Path(tmp_raw) / "missing-dpas.hpp"
        result = run_result(missing)
        assert result.returncode != 0
        assert f"error: missing dpas header: {missing}" in result.stdout


def test_capability_checker_prints_resolved_absolute_header_for_relative_argument() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        header = tmp / "dpas.hpp"
        header.write_text(DPAS_CONTRACT)
        values = run_relative(tmp, "dpas.hpp")
        assert values["dpas.header"] == str(header.resolve())


def test_capability_checker_ignores_comment_only_core_contract_markers() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        header = pathlib.Path(tmp_raw) / "dpas.hpp"
        header.write_text(
            '// static_assert(RepeatCount >= 1 && RepeatCount <= 8, "Repeat count must be within 1 to 8 range");\n'
            '/* static_assert(SystolicDepth == 8, "Systolic depth must be equal to 8"); */\n'
            '// static_assert(ExecutionSize == 8 || (!IsDPASW && ExecutionSize == 16), "Execution size must be 8 or 16 for DPAS and 8 for DPASW.");\n'
        )
        values = run(header)
        assert values["dpas.repeat_count.max"] == "unknown"
        assert values["dpas.systolic_depth"] == "unknown"
        assert values["dpas.exec_size.allowed"] == "unknown"


def test_capability_checker_ignores_comment_only_bdpas_and_e2m1_markers() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        header = pathlib.Path(tmp_raw) / "dpas.hpp"
        header.write_text(
            DPAS_CONTRACT
            + '// template <typename T> auto bdpas(T x) { return x; }\n'
            + '/* enum class dpas_argument_type { e2m1 }; */\n'
        )
        values = run(header)
        assert values["dpas.bdpas.present"] == "0"
        assert values["dpas.fp4_e2m1.present"] == "0"


def test_capability_checker_ignores_string_literal_only_bdpas_and_e2m1_markers() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        header = pathlib.Path(tmp_raw) / "dpas.hpp"
        header.write_text(
            DPAS_CONTRACT
            + 'const char * bdpas_diag = "template <typename T> auto bdpas(T x) { return x; }";\n'
            + 'const char * fp4_diag = "enum class dpas_argument_type { e2m1 };";\n'
            + "const char quote_diag = 'e';\n"
        )
        values = run(header)
        assert values["dpas.bdpas.present"] == "0"
        assert values["dpas.fp4_e2m1.present"] == "0"


def test_capability_checker_scans_sibling_common_header_for_e2m1() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        root = pathlib.Path(tmp_raw)
        header = root / "dpas.hpp"
        common = root / "common.hpp"
        header.write_text(DPAS_CONTRACT)
        common.write_text("enum class dpas_argument_type { s8, e2m1 };\n")
        values = run(header)
        assert values["dpas.common_header"] == str(common.resolve())
        assert values["dpas.bdpas.present"] == "0"
        assert values["dpas.fp4_e2m1.present"] == "1"
