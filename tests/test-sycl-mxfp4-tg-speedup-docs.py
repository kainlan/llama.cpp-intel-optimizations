from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SYCL_DOC = ROOT / "docs" / "backend" / "SYCL.md"
FINAL = ROOT / "activation" / "sycl-mxfp4-tg-speedup-final-review-20260707.md"


def _doc_row(text: str, key: str) -> str:
    for line in text.splitlines():
        if key in line:
            return line
    raise AssertionError(f"missing documentation row for {key}")


def test_speedup_envs_are_documented_as_default_off() -> None:
    text = SYCL_DOC.read_text(encoding="utf-8")
    row = _doc_row(text, "GGML_SYCL_MOE_DOWN_Q8_DPAS_TILE")
    assert "GGML_SYCL_MOE_DOWN_Q8_DPAS_TILE" in row
    assert "tile2" in row
    assert "tile4" in row
    assert "default is OFF" in row
    assert "Experimental/default-off" in row
    assert "rejected default-on promotion" in row
    assert "met end-to-end TG or named-kernel performance criteria" in row
    assert "recommend" not in row.lower()


def test_final_review_records_rejection_and_copied_validation_evidence() -> None:
    text = FINAL.read_text(encoding="utf-8")
    lower = text.lower()
    assert "promotion decision: reject default-on, keep opt-in" in lower
    assert "default-off" in lower
    assert "GGML_SYCL_MOE_DOWN_Q8_DPAS_TILE" in text
    assert "GGML_SYCL_MOE_DOWN_Q8_DPAS_TILE=tile2" in text
    assert "GGML_SYCL_MOE_DOWN_Q8_DPAS_TILE=tile4" in text
    assert "Task 6" in text and "skipped" in lower
    assert "TG128 >=45 tok/s" in text and "was not reached" in lower

    for expected in [
        "PP512",
        "TG128",
        "mxfp4.gateup.xmx_tiled_dpas_m2",
        "mxfp4.down.q8_soa",
        "mxfp4.down.q8_dpas_tile2",
        "mxfp4.down.q8_dpas_tile4",
        "34.37",  # baseline TG128 from Task 7 matrix
        "33.12",  # tile2 TG128 from Task 7 matrix
        "34.21",  # tile4 TG128 from Task 7 matrix
        "620.982",  # baseline down ms
        "755.553",  # tile2 down ms
        "635.430",  # tile4 down ms
    ]:
        assert expected in text


def test_final_review_does_not_present_tiles_as_default_on_or_recommended() -> None:
    text = FINAL.read_text(encoding="utf-8")
    lower = text.lower()
    decision_line = next(line for line in text.splitlines() if line.startswith("Promotion decision:"))
    assert decision_line == "Promotion decision: reject default-on, keep opt-in"
    assert "do not promote" in lower
    assert "keep both values available only as experimental/default-off diagnostics" in lower
    assert "recommend" not in lower
    for forbidden in [
        "promotion decision: promote",
        "promotion decision: accept",
        "promotion decision: default-on",
        "enable by default",
        "make default-on",
        "make default",
        "default-on candidate accepted",
        "default-on is approved",
        "tile2 is approved",
        "tile4 is approved",
    ]:
        assert forbidden not in lower
