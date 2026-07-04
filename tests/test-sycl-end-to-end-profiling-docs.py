from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/backend/SYCL.md"
RESEARCH = ROOT / "research.md"


def test_full_attribution_docs_describe_layered_closure_contract() -> None:
    text = DOC.read_text(encoding="utf-8")
    for required in (
        "sycl-gptoss-full-attribution-profile.sh",
        "parsed/layer-ledger.parse",
        "coverage.layer_status",
        "source_attribution.status",
        ".debug_line",
        "source_region_plus_ablation",
        "set +u",
        "source /opt/intel/oneapi/setvars.sh --force",
        "ZE_ENABLE_TRACING_LAYER=1",
        "SYCL_UR_TRACE=2",
        "fails closed",
        "workers must not run",
    ):
        assert required in text


def test_research_artifact_is_present() -> None:
    text = RESEARCH.read_text(encoding="utf-8")
    assert "Complete end-to-end profiling" in text
    assert "Level Zero" in text
    assert "Unified Runtime" in text
    assert "VTune" in text
