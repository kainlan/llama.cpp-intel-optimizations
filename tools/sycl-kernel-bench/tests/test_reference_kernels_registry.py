import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REGISTRY = os.path.join(ROOT, "tools", "sycl-kernel-bench", "kernel_registry.hpp")


def registered_kernel_names():
    with open(REGISTRY, "r", encoding="utf-8") as f:
        text = f.read()
    return set(re.findall(r'\{\s*"([^"]+)"\s*,', text))


def test_reference_kernels_registered():
    with open(REGISTRY, "r", encoding="utf-8") as f:
        text = f.read()
    for name in [
        "onednn_fp16_gemm",
        "onednn_int8_gemm",
        "onednn_woq_gemm",
        "memory_bandwidth",
        "roofline_compute",
    ]:
        assert name in text, f"{name} not registered in kernel_registry.hpp"


def test_mxfp4_pair_glu_loadv2_variants_registered():
    names = registered_kernel_names()
    for name in [
        "mxfp4_pair_glu_xmx_tiled_packed_r8_m2_loadv2",
        "mxfp4_pair_glu_xmx_tiled_packed_r8_m2_loadv2_bias",
    ]:
        assert name in names, f"{name} not registered as an exact kernel_registry.hpp entry"
