from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SYCL_DOC = ROOT / "docs" / "backend" / "SYCL.md"
FINAL = ROOT / "activation" / "sycl-mxfp4-tg-speedup-final-review-20260707.md"


def test_speedup_envs_are_documented_as_default_off() -> None:
    text = SYCL_DOC.read_text(encoding="utf-8")
    assert "GGML_SYCL_MOE_DOWN_Q8_DPAS_TILE" in text
    assert "default is OFF" in text[text.index("GGML_SYCL_MOE_DOWN_Q8_DPAS_TILE") : text.index("GGML_SYCL_MOE_DOWN_Q8_DPAS_TILE") + 240]


def test_final_review_records_promotion_decision() -> None:
    text = FINAL.read_text(encoding="utf-8")
    assert "Promotion decision:" in text
    assert "PP512" in text
    assert "TG128" in text
    assert "mxfp4.gateup.xmx_tiled_dpas_m2" in text
    assert "mxfp4.down" in text
