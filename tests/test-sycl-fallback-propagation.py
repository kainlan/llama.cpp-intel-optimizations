#!/usr/bin/env python3
"""Caller-level contract for injected DL mmap/staging fallback failures."""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
sycl = root / "ggml/src/ggml-sycl"
files = {name: (sycl / name).read_text() for name in ("ggml-sycl.cpp", "getrows.cpp", "mmq.cpp", "mmvq.cpp", "dmmv.cpp")}
combined = "\n".join(files.values())

# Every former runtime CPU fallback caller remains wired to the single injected
# failure point. In DL that point throws before the immediately-following abort;
# graph_compute catches it and converts it to a recoverable status.
expected_reasons = (
    "mul_mat streaming", "get_rows seq staging", "get_rows seq host staging",
    "get_rows streaming", "get_rows streaming exception", "get_rows no stream",
    "mmq streaming", "mmvq streaming", "dmmv streaming",
)
for reason in expected_reasons:
    assert f'ggml_sycl_cpu_fallback_graph(ctx, dst, "{reason}")' in combined, reason
main = files["ggml-sycl.cpp"]
assert re.search(r"#ifdef GGML_BACKEND_DL.*?throw ggml_sycl_fallback_error\(reason\);.*?#else", main, re.S)
assert "catch (const ggml_sycl_fallback_error & error)" in main
assert "return GGML_STATUS_FAILED;" in main
print("SYCL injected mmap/staging fallback propagation: PASS")
