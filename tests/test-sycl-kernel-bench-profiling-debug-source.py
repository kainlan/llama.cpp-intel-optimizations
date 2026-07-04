#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CMAKE = ROOT / "tools" / "sycl-kernel-bench" / "CMakeLists.txt"


def test_kernel_bench_honors_sycl_profiling_debug_flag() -> None:
    text = CMAKE.read_text(encoding="utf-8")
    match = re.search(r"^    if \(GGML_SYCL_PROFILING_DEBUG\)\n(?P<block>.*?)^    endif\(\)$", text, re.MULTILINE | re.DOTALL)
    assert match is not None
    debug_block = match.group("block")
    assert '"-g"' in debug_block
    assert '"-gline-tables-only"' in debug_block
    assert '"-fdebug-info-for-profiling"' in debug_block
    assert '"-fsycl-instrument-device-code"' in debug_block
    assert 'target_link_options(${TARGET} PRIVATE "-fsycl-instrument-device-code")' in debug_block
