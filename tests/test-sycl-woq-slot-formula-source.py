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

Backend checks are anchored to the sizing region only (between the
pp_woq_enabled declaration and the weight_slot_bytes assignment), not the
whole 60k-line file: ggml-sycl.cpp reuses the exact expression
"blocks_per_row = static_cast<int>(ne00 / QK_MXFP4)" (with different
whitespace) at an unrelated call site further down (the f16 arm's repack
call), so a whole-file substring/regex search on that text passes
vacuously even when the sizing region's own copy is mutated -- caught by
re-review (llama.cpp-sr83 fix cycle 4, findings B1/B2). Slicing to the
sizing region first, and requiring the N multiplicand
(static_cast<size_t>(ne01)) inside each plane's checked_mul call, closes
both gaps without needing an unbounded re.S wildcard.
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
backend_full = Path(args.backend).read_text()

REGION_START = "pp_woq_enabled    = ggml_sycl_moe_pp_woq_enabled();"
REGION_END = "weight_slot_bytes     = woq_nibble_slot_bytes + woq_scale_slot_bytes;"
region_start_pos = backend_full.find(REGION_START)
region_end_pos = backend_full.find(REGION_END)
if region_start_pos < 0 or region_end_pos < 0 or region_end_pos < region_start_pos:
    print("FAIL: could not locate the WOQ sizing region anchors in ggml-sycl.cpp "
          f"(start={region_start_pos}, end={region_end_pos})", file=sys.stderr)
    raise SystemExit(1)
backend = backend_full[region_start_pos:region_end_pos + len(REGION_END)]

checks = {
    "llama-model.cpp: nibble plane is N * (K/2), 256-aligned":
        re.search(r"align_up\(\s*n\s*\*\s*\(\s*k\s*/\s*2\s*\)\s*,\s*256\s*\)", model) is not None,
    "llama-model.cpp: scale plane is (K/blocksize) * N, 256-aligned":
        re.search(r"align_up\(\s*\(\s*k\s*/\s*kMxfp4BlockSize\s*\)\s*\*\s*n\s*,\s*256\s*\)", model) is not None,
    "llama-model.cpp: block size is 32 (matches QK_MXFP4)":
        re.search(r"kMxfp4BlockSize\s*=\s*32\s*;", model) is not None,
    "ggml-sycl.cpp (sizing region): blocks_per_row is ne00 / QK_MXFP4 (block size 32)":
        re.search(r"const\s+int\s+blocks_per_row\s*=\s*static_cast<int>\(\s*ne00\s*/\s*QK_MXFP4\s*\)", backend)
        is not None,
    "ggml-sycl.cpp (sizing region): nibble plane is N * (K/2)":
        re.search(
            r"checked_mul\(\s*static_cast<size_t>\(ne01\)\s*,\s*static_cast<size_t>\(ne00\)\s*/\s*2\s*,\s*"
            r"nibble_bytes_per_expert\s*\)", backend) is not None,
    "ggml-sycl.cpp (sizing region): scale plane is (K/blocks_per_row) * N":
        re.search(
            r"checked_mul\(\s*static_cast<size_t>\(blocks_per_row\)\s*,\s*static_cast<size_t>\(ne01\)\s*,\s*"
            r"scale_bytes_per_expert\s*\)", backend) is not None,
    "ggml-sycl.cpp (sizing region): both planes align to 256 via the same helper":
        backend.count("align_up_256(nibble_bytes_per_expert)") == 1
        and backend.count("align_up_256(scale_bytes_per_expert)") == 1
        and "(v + 255) & ~static_cast<size_t>(255)" in backend,
    "ggml-sycl.cpp (sizing region): weight slot is nibble plane + scale plane, nothing else":
        re.search(r"weight_slot_bytes\s*=\s*woq_nibble_slot_bytes\s*\+\s*woq_scale_slot_bytes;", backend)
        is not None,
}

failed = sorted(name for name, ok in checks.items() if not ok)
for name in failed:
    print("FAIL: " + name, file=sys.stderr)
if failed:
    raise SystemExit(1)
print(f"WOQ slot-formula source contract: PASS ({len(checks)} checks)")
