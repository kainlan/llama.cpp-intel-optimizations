from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/backend/SYCL.md"


def read_doc() -> str:
    return DOC.read_text(encoding="utf-8")


def test_staged_profiling_docs_describe_strict_merge_contract() -> None:
    text = read_doc()
    for required in (
        "Staged layered SYCL profiling closure",
        "sycl-gptoss-staged-attribution-profile.sh",
        "stage-manifest.json",
        "merge-sycl-staged-ledger.py",
        "coverage.layer_status ok",
        "layer.unknown_wall_ms_x1000",
        "source_region_plus_ablation",
        "source_attribution.ablation_delta_ms_x1000",
        "build/model/device/FA/MoE/prompt/gen/repeat",
        "all stage manifests metadata-match",
        "set +u",
        "source /opt/intel/oneapi/setvars.sh --force",
        "workers must not run",
    ):
        assert required in text


def test_staged_profiling_docs_define_success_and_fallback_requirements() -> None:
    text = read_doc()
    for required in (
        "Exact source-line attribution",
        "source_line.status pass",
        "source_attribution.status exact_source_line",
        "source_line.status fail",
        "source_line.blocker",
        "recorded `source_line.blocker`",
        "Source-region fallback",
        "plain source-region mapping is triage evidence, not closure",
        "metadata_mismatch",
        "source_attribution_incomplete",
    ):
        assert required in text


def test_staged_profiling_docs_call_out_dry_run_execute_ack_and_safe_knobs() -> None:
    text = read_doc()
    for required in (
        "dry-run by default",
        "--dry-run",
        "--execute \\",
        "--i-understand-this-runs-staged-gpu-profiling",
        "-fa 1",
        "GGML_SYCL_MOE_PHASE_MATERIALIZE=1",
        "GGML_SYCL_MOE_PHASE_BULK_XMX=1",
        "GGML_SYCL_MOE_DOWN_SUM_DIRECT=1",
    ):
        assert required in text


def test_staged_profiling_docs_list_stage_artifacts() -> None:
    text = read_doc()
    for required in (
        "`base/`",
        "`l0/`",
        "`ur/`",
        "`vtune-source/`",
        "`ablation/`",
        "`merged/`",
    ):
        assert required in text
