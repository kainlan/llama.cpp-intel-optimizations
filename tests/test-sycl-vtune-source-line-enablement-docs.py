from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/backend/SYCL.md"
SOURCE_LINE_HEADING = "VTune GPU source-line attribution"
NEXT_HEADING = "### Full layered SYCL profiling closure"


def source_line_section() -> str:
    text = DOC.read_text(encoding="utf-8")
    start = text.index(SOURCE_LINE_HEADING)
    end = text.index(NEXT_HEADING, start)
    return text[start:end]


def test_source_line_enablement_docs_define_three_evidence_layers() -> None:
    section = source_line_section()
    required = (
        "ZEBin DWARF line table",
        "`.debug_line` is necessary but not sufficient",
        "VTune computing-task selection",
        "VTune `gpu-source-line`",
        "source_line.status pass",
    )
    for phrase in required:
        assert phrase in section

    assert section.index("ZEBin DWARF line table") < section.index("VTune computing-task selection")
    assert section.index("VTune computing-task selection") < section.index("VTune `gpu-source-line`")


def test_probe_matrix_must_pass_before_mxfp4_feasibility() -> None:
    section = source_line_section()
    assert "sycl-source-line-probe" in section
    assert "scripts/sycl-vtune-source-line-feasibility.sh" in section
    assert "matrix must pass before MXFP4 exact-line feasibility runs" in section
    assert section.index("sycl-source-line-probe") < section.index("scripts/sycl-vtune-source-line-feasibility.sh")
    assert "--require-matrix-pass /tmp/sycl_source_line_matrix_20260705_120000/build-matrix/debug_full/source-line-feasibility.parse" in section


def test_lead_only_acknowledgements_and_worker_ban_are_documented() -> None:
    section = source_line_section()
    for phrase in (
        "scripts/sycl-source-line-debug-matrix.sh",
        "--dry-run",
        "set +u",
        "source /opt/intel/oneapi/setvars.sh --force",
        "set -u",
        "--execute",
        "--i-understand-this-runs-gpu-source-probe",
        "--i-understand-this-runs-gpu-microbenchmarks",
        "Workers must not run",
        "--vtune-target-gpu 0:7:0.0",
        "Do not hard-code target GPU values across hosts or boots",
    ):
        assert phrase in section

    dry_run_pos = section.index("Safe dry-run command")
    execute_pos = section.index("Lead-only execute command")
    worker_ban_pos = section.index("Workers must not run")
    assert dry_run_pos < execute_pos < worker_ban_pos


def test_sycl_docs_describe_asm_line_static_cost_status() -> None:
    section = source_line_section()
    for phrase in (
        "asm-line-static-cost",
        "asm-line-static",
        "not sampled VTune exact",
        "dwarf-line-table-only",
    ):
        assert phrase in section
