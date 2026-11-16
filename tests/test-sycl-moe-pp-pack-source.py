"""The batched MoE prompt executor takes every activation and output offset from one helper each.

The executor lambda try_pp_onednn_f16_batched (ggml/src/ggml-sycl/ggml-sycl.cpp) lays its activation and output
scratch out per GEMM group, from moe_pp_pack_gemm_groups (moe-pp-pack.hpp). The gather, both GEMM arms and the scatter
take every offset into that scratch from act_base and out_base, so they cannot disagree about where an expert's rows
are, and none of the old rectangle's names remains.

Host-only text assertions. llama_test_pytest hands this file to pytest, so every check lives inside a test_*()
function: a file with none is "no tests collected" (exit 5), which ctest scores as a failure. Checks run on
comment-stripped text, so prose that quotes a name or a call cannot satisfy or trip them.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Plain file I/O: the codescout index is blind inside ggml-sycl.cpp.
GGML_SYCL_CPP = (ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text(encoding="utf-8")


def _strip_comments(text):
    text = re.sub(r"/\*.*?\*/", lambda m: " " * len(m.group(0)), text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def _executor_body():
    """The executor lambda's text. A lambda that is not found fails the calling test rather than skipping it."""
    text = _strip_comments(GGML_SYCL_CPP)
    begin = text.find("auto try_pp_onednn_f16_batched = [&]() -> bool {")
    end = text.find("if (try_pp_onednn_f16_batched()) {")
    assert begin >= 0 and end >= begin, "the executor lambda try_pp_onednn_f16_batched was not found"
    return text[begin:end]


def test_the_rectangle_names_are_gone():
    body = _executor_body()
    # Whole words only: the header's own type, moe_pp_gemm_group, contains the retired pp_gemm_group, so a substring
    # test would fire on the correct executor.
    failures = [
        f"{name}: the rectangle's name is still in the executor"
        for name in ("max_rows_padded", "act_slot_elems", "out_slot_elems", "pp_gemm_group")
        if re.search(rf"\b{re.escape(name)}\b", body)
    ]
    assert not failures, "\n".join(failures)


def test_the_groups_come_from_the_header():
    body = _executor_body()
    assert "moe_pp_pack_gemm_groups(" in body, "the executor does not take its groups from moe_pp_pack_gemm_groups"


@pytest.mark.parametrize("pointer,helper,uses", [("batched_acts", "act_base", 3), ("batched_out", "out_base", 3)])
def test_every_offset_comes_from_one_helper(pointer, helper, uses):
    body = _executor_body()
    failures = []
    offsets = len(re.findall(rf"\b{pointer}\s*\+", body))
    if offsets != 1:
        failures.append(f"{pointer}: {offsets} pointer offsets, want exactly one (inside {helper})")
    calls = len(re.findall(rf"\b{helper}\(", body))
    if calls < uses:
        failures.append(f"{helper}: {calls} calls, want at least {uses} (gather or scatter, and both GEMM arms)")
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
