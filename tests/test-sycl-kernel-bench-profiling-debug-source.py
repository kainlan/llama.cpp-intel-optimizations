#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CMAKE = ROOT / "tools" / "sycl-kernel-bench" / "CMakeLists.txt"


def test_kernel_bench_honors_sycl_profiling_debug_flag() -> None:
    text = CMAKE.read_text(encoding="utf-8")
    assert "if (GGML_SYCL_PROFILING_DEBUG)" in text
    assert '"-g"' in text
    assert '"-gline-tables-only"' in text
    assert '"-fdebug-info-for-profiling"' in text
    assert '"-fsycl-instrument-device-code"' in text
    assert 'target_link_options(${TARGET} PRIVATE "-fsycl-instrument-device-code")' in text
