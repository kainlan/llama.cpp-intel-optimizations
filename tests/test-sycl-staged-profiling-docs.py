import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/backend/SYCL.md"
STAGED_HEADING = "### Staged layered SYCL profiling closure"


def read_doc() -> str:
    return DOC.read_text(encoding="utf-8")


def staged_profiling_section() -> str:
    text = read_doc()
    start = text.index(STAGED_HEADING)
    next_heading = re.search(r"^#{1,3} ", text[start + len(STAGED_HEADING) :], re.MULTILINE)
    assert next_heading is not None
    return text[start : start + len(STAGED_HEADING) + next_heading.start()]


def command_block_after(section: str, label: str) -> str:
    label_pos = section.index(label)
    block_start = section.index("```bash", label_pos)
    content_start = section.index("\n", block_start) + 1
    block_end = section.index("```", content_start)
    return section[content_start:block_end]


def test_staged_profiling_docs_describe_strict_merge_contract() -> None:
    section = staged_profiling_section()
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
        assert required in section


def test_staged_profiling_docs_define_success_and_fallback_requirements() -> None:
    section = staged_profiling_section()
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
        assert required in section


def test_staged_profiling_docs_scope_dry_run_and_execute_safety_to_commands() -> None:
    section = staged_profiling_section()
    dry_run_command = command_block_after(section, "Dry-run command:")
    real_command = command_block_after(section, "Lead-only real command.")

    assert "dry-run by default" in section
    assert "real execution requires both `--execute --i-understand-this-runs-staged-gpu-profiling`" in section

    assert "bash scripts/sycl-gptoss-staged-attribution-profile.sh \\" in dry_run_command
    assert "  --dry-run \\" in dry_run_command
    assert "  --out-root /tmp/sycl_gptoss_staged_attribution_dryrun" in dry_run_command
    assert "--execute" not in dry_run_command
    assert "--i-understand-this-runs-staged-gpu-profiling" not in dry_run_command

    assert "set +u" in real_command
    assert "source /opt/intel/oneapi/setvars.sh --force" in real_command
    assert "set -u" in real_command
    assert "bash scripts/sycl-gptoss-staged-attribution-profile.sh \\" in real_command
    assert "  --execute \\" in real_command
    assert "  --i-understand-this-runs-staged-gpu-profiling \\" in real_command
    assert "  --out-root /tmp/sycl_gptoss_staged_attribution_$(date +%Y%m%d_%H%M%S)" in real_command
    assert "--dry-run" not in real_command


def test_staged_profiling_docs_call_out_safe_knobs_in_staged_section() -> None:
    section = staged_profiling_section()
    for required in (
        "-fa 1",
        "GGML_SYCL_MOE_PHASE_MATERIALIZE=1",
        "GGML_SYCL_MOE_PHASE_BULK_XMX=1",
        "GGML_SYCL_MOE_DOWN_SUM_DIRECT=1",
    ):
        assert required in section


def test_staged_profiling_docs_list_stage_artifacts() -> None:
    section = staged_profiling_section()
    for required in (
        "`base/`",
        "`l0/`",
        "`ur/`",
        "`vtune-source/`",
        "`ablation/`",
        "`merged/`",
    ):
        assert required in section
