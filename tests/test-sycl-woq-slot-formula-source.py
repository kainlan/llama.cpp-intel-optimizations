#!/usr/bin/env python3
"""Host-only source contract: the WOQ scratch-slot byte formula must match,
byte-for-byte, between the two independent sites that compute it --
llama_model_sycl_populate_inventory (plan-time sizing, src/llama-model.cpp)
and try_pp_mxfp4_soa_onednn_f16_batched (per-dispatch sizing,
ggml-sycl.cpp). The two sites cannot share a helper (llama-model.cpp does
not link the SYCL backend's device headers), so this test is the only
thing that keeps them from silently drifting apart -- an under-sized plan
makes admission fail-closed refuse every real dispatch (f2bdfbffe).

Shape both sites must express: nibble plane = align256(N * (K/2)); scale
plane = align256((K/32) * N); K/32 is QK_MXFP4 in ggml-sycl.cpp and the
locally-named kMxfp4BlockSize in llama-model.cpp (which does not include
ggml-common.h). llama.cpp-sr83, quality review coverage gap (a).
"""
import argparse
from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--model", default=str(root / "src/llama-model.cpp"))
parser.add_argument("--backend", default=str(root / "ggml/src/ggml-sycl/ggml-sycl.cpp"))
args = parser.parse_args()

model = Path(args.model).read_text()
backend = Path(args.backend).read_text()

checks = {
    "llama-model.cpp: nibble plane is N * (K/2), 256-aligned":
        re.search(r"align_up\(\s*n\s*\*\s*\(\s*k\s*/\s*2\s*\)\s*,\s*256\s*\)", model) is not None,
    "llama-model.cpp: scale plane is (K/blocksize) * N, 256-aligned":
        re.search(r"align_up\(\s*\(\s*k\s*/\s*kMxfp4BlockSize\s*\)\s*\*\s*n\s*,\s*256\s*\)", model) is not None,
    "llama-model.cpp: block size is 32 (matches QK_MXFP4)":
        re.search(r"kMxfp4BlockSize\s*=\s*32\s*;", model) is not None,
    "ggml-sycl.cpp: nibble plane divides K by 2 before the N multiply":
        re.search(r"ne00\)\s*/\s*2\s*,\s*nibble_bytes_per_expert\s*\)", backend) is not None,
    "ggml-sycl.cpp: scale plane multiplies blocks_per_row (K/QK_MXFP4) by N":
        re.search(r"blocks_per_row\).*?,\s*\n?\s*.*?scale_bytes_per_expert\s*\)", backend, re.S) is not None,
    "ggml-sycl.cpp: blocks_per_row is ne00 / QK_MXFP4 (block size 32)":
        "blocks_per_row = static_cast<int>(ne00 / QK_MXFP4)" in backend,
    "ggml-sycl.cpp: both planes align to 256 via the same helper":
        backend.count("align_up_256(nibble_bytes_per_expert)") == 1
        and backend.count("align_up_256(scale_bytes_per_expert)") == 1
        and "(v + 255) & ~static_cast<size_t>(255)" in backend,
    "ggml-sycl.cpp: weight slot is nibble plane + scale plane, nothing else":
        "weight_slot_bytes     = woq_nibble_slot_bytes + woq_scale_slot_bytes;" in backend
        or re.search(r"weight_slot_bytes\s*=\s*woq_nibble_slot_bytes\s*\+\s*woq_scale_slot_bytes;", backend)
        is not None,
}

failed = sorted(name for name, ok in checks.items() if not ok)
for name in failed:
    print("FAIL: " + name, file=sys.stderr)
if failed:
    raise SystemExit(1)
print(f"WOQ slot-formula source contract: PASS ({len(checks)} checks)")
