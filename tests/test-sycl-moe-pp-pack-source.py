#!/usr/bin/env python3
"""The batched MoE prompt executor takes every activation and output offset from one helper each."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
text = (ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text(encoding="utf-8")
text = re.sub(r"/\*.*?\*/", lambda m: " " * len(m.group(0)), text, flags=re.S)
text = re.sub(r"//[^\n]*", "", text)

begin = text.find("auto try_pp_onednn_f16_batched = [&]() -> bool {")
end = text.find("if (try_pp_onednn_f16_batched()) {")
failures = []
if begin < 0 or end < begin:
    failures.append("the executor lambda try_pp_onednn_f16_batched was not found")
body = text[begin:end] if not failures else ""

# Whole words only: the header's own type, moe_pp_gemm_group, contains the retired pp_gemm_group, so a substring test
# would fire on the correct executor.
for name in ("max_rows_padded", "act_slot_elems", "out_slot_elems", "pp_gemm_group"):
    if re.search(rf"\b{re.escape(name)}\b", body):
        failures.append(f"{name}: the rectangle's name is still in the executor")
if "moe_pp_pack_gemm_groups(" not in body:
    failures.append("the executor does not take its groups from moe_pp_pack_gemm_groups")
for pointer, helper, uses in (("batched_acts", "act_base", 3), ("batched_out", "out_base", 3)):
    offsets = len(re.findall(rf"\b{pointer}\s*\+", body))
    if offsets != 1:
        failures.append(f"{pointer}: {offsets} pointer offsets, want exactly one (inside {helper})")
    calls = len(re.findall(rf"\b{helper}\(", body))
    if calls < uses:
        failures.append(f"{helper}: {calls} calls, want at least {uses} (gather or scatter, and both GEMM arms)")

if failures:
    print("\n".join(failures))
    sys.exit(1)
print("ok")
