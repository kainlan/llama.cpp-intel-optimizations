#!/usr/bin/env python3
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
HEADER = ROOT / "ggml" / "src" / "ggml-sycl" / "sycl-kernel-profiler.hpp"
CPP = ROOT / "ggml" / "src" / "ggml-sycl" / "sycl-kernel-profiler.cpp"
COMMON = ROOT / "ggml" / "src" / "ggml-sycl" / "common.hpp"
MMVQ = ROOT / "ggml" / "src" / "ggml-sycl" / "mmvq.cpp"
FATTN = ROOT / "ggml" / "src" / "ggml-sycl" / "fattn.cpp"


def slice_between(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    finish = text.index(end, begin + len(start))
    return text[begin:finish]


def test_fattn_major_submits_have_named_profile_labels() -> None:
    fattn = FATTN.read_text(encoding="utf-8")
    assert "#include \"sycl-kernel-profiler.hpp\"" in fattn
    assert "fattn.xmx_pack_k_set_rows" in fattn
    assert "fattn.pack" in fattn
    set_rows = slice_between(
        fattn,
        "static sycl::event ggml_sycl_fattn_xmx_submit_set_rows_update",
        "}  // namespace",
    )
    assert "ggml_sycl_profile_submit(*stream" in set_rows
    assert "cgh.depends_on(set_rows_event)" in set_rows
    assert ".wait(" not in set_rows
