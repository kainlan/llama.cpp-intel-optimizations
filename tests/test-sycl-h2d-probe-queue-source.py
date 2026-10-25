#!/usr/bin/env python3
"""The load-time H2D probe and the expert stream ring copy on the same cache queue (2026-10-24 plan, W5 / Task 5)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml" / "src" / "ggml-sycl"
GGML_SYCL = (SYCL / "ggml-sycl.cpp").read_text()
PLAN = (SYCL / "moe-stream-plan.cpp").read_text()


def probe_body() -> str:
    # The definition, not the forward declaration earlier in the file.
    start = GGML_SYCL.index("static void ggml_sycl_measure_h2d_pinned(ggml_sycl_device_info & info) {")
    end = GGML_SYCL.index("\n}\n", start)
    return GGML_SYCL[start:end]


def test_probe_times_the_bcs_queue():
    body = probe_body()
    # clang-format column-aligns the declaration with its neighbours, so the spacing is not pinned.
    assert re.search(r"sycl::queue \*\s+q = &cache->get_bcs_queue\(\);", body)
    assert "&cache->get_queue();" not in body


def test_ring_copies_on_the_bcs_queue():
    assert PLAN.count("cache.get_bcs_queue()") == 1
