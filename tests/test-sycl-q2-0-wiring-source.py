#!/usr/bin/env python3
"""Source gate for llama.cpp-s36q phase 4: Q2_0 has every SYCL piece it is admitted on.

Until phase 4 the type was refused (yitq/phbr): the backend had no Q2_0 dequant, no MMVQ vec_dot and no
_id arm, and an op admitted on it reached an executor with nothing to run. The type is now a first-class
weight type, and these facts have to hold together; dropping any one reopens that gap while the rest of
the tree still reads as covered:

  * dequantize_q2_0 exists (dequantize.hpp) and both to_fp16 and to_fp32 getters return it through
    dequantize_block_sycl<QK2_0, QR2_0, ...> -- the converter behind the PP dequant arm, the per-expert
    dense fallback every n > 1 MUL_MAT_ID takes, and dense n > 1;
  * vec_dot_q2_0_q8_1 exists (vecdotq.hpp) with the signature-generic vec_dot_q_sycl_t shape, reads qs
    with the 2-byte-aligned reader (block_q2_0 is 18 bytes, qs is 2 mod 4 on every other block) and
    unpacks signed (code - 1) lanes, so it needs no activation-sum term;
  * the dense MMVQ dispatch has a Q2_0 case into a launcher that instantiates mul_mat_vec_q with the
    (QK2_0, QI2_0, block_q2_0, VDR_Q2_0_Q8_1_MMVQ, vec_dot_q2_0_q8_1) tuple;
  * mmvq_submit_quant_aos_id has a Q2_0 arm instantiating mmvq_submit_aos_id_impl with the same tuple
    (the consumer switches' coverage is gated by test-sycl-moe-mmvq-consumer-coverage.py);
  * ggml_sycl_mul_mat_type_supported admits Q2_0, and GET_ROWS still does NOT: no arm exists, and no
    Q2_0 tensor is an embedding table (Qwen3.8's token_embd is IQ3_S).

Runs under pytest and as a plain script. GGML_SYCL_Q2_0_ROOT points the gate at another tree (the
pre-phase-4 export, for the RED run). No device or build is touched.
"""

import os
import re
import sys
from pathlib import Path

ROOT = Path(os.environ.get("GGML_SYCL_Q2_0_ROOT", str(Path(__file__).resolve().parents[1])))
SYCL = ROOT / "ggml/src/ggml-sycl"


def read(name: str) -> str:
    return (SYCL / name).read_text(encoding="utf-8")


def function_body(text: str, header: str) -> str:
    """The braced body after `header`, or "" when the function does not exist (a check then fails by name)."""
    start = text.find(header)
    if start < 0:
        return ""
    opening = text.index("{", start + len(header) - 1)
    depth = 0
    for position in range(opening, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : position]
    raise ValueError(f"unclosed body: {header}")


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def check_dequant(dequantize: str, convert: str) -> list:
    failures = []
    if "static __dpct_inline__ void dequantize_q2_0(" not in dequantize:
        failures.append("dequantize_q2_0 is missing")
    want = "caseGGML_TYPE_Q2_0:returndequantize_block_sycl<QK2_0,QR2_0,dequantize_q2_0>;"
    fp16 = squash(function_body(convert, "to_fp16_sycl_t ggml_get_to_fp16_sycl_for_layout("))
    fp32 = squash(function_body(convert, "to_fp32_sycl_t ggml_get_to_fp32_sycl("))
    if want not in fp16:
        failures.append("the layout-keyed to_fp16 getter has no Q2_0 -> dequantize_q2_0 case")
    if want not in fp32:
        failures.append("the to_fp32 getter has no Q2_0 -> dequantize_q2_0 case")
    return failures


def check_vec_dot(vecdotq: str) -> list:
    failures = []
    if "#define VDR_Q2_0_Q8_1_MMVQ 1" not in vecdotq:
        failures.append("VDR_Q2_0_Q8_1_MMVQ is not 1")
    match = re.search(
        r"vec_dot_q2_0_q8_1\(const void \*\s*__restrict__ vbq,\s*const block_q8_1 \*\s*__restrict__ bq8_1,\s*const int\s*&\s*iqs\)\s*\{",
        vecdotq,
    )
    if match is None:
        failures.append("vec_dot_q2_0_q8_1 is missing or lost the generic vec_dot_q_sycl_t signature")
        return failures
    body = function_body(vecdotq[match.start() :], "vec_dot_q2_0_q8_1(")
    if "get_int_from_uint8(bq2_0->qs" not in body:
        failures.append("vec_dot_q2_0_q8_1 must read qs with the 2-byte-aligned get_int_from_uint8")
    if "_aligned(bq2_0->qs" in body:
        failures.append("vec_dot_q2_0_q8_1 reads qs with a 4-byte-aligned reader; qs is 2 mod 4 on odd blocks")
    if ") - 1)" not in body:
        failures.append("vec_dot_q2_0_q8_1 does not apply the (code - 1) zero point while unpacking")
    if "ds[1]" in body or ".y()" in body:
        failures.append("vec_dot_q2_0_q8_1 uses the activation sum; signed unpacking needs none")
    return failures


def check_mmvq(mmvq: str) -> list:
    failures = []
    tuple_text = "QK2_0,QI2_0,block_q2_0,VDR_Q2_0_Q8_1_MMVQ"
    launcher = squash(function_body(mmvq, "static void mul_mat_vec_q2_0_q8_1_sycl("))
    if f"mul_mat_vec_q<{tuple_text},vec_dot_q2_0_q8_1>" not in launcher:
        failures.append("the dense launcher does not instantiate mul_mat_vec_q with the Q2_0 tuple")
    if not re.search(r"case GGML_TYPE_Q2_0:\s*GGML_SYCL_KTRACE\(\"mmvq_q2_0\"[^;]*;\s*mul_mat_vec_q2_0_q8_1_sycl\(", mmvq):
        failures.append("the dense MMVQ dispatch has no Q2_0 case calling the launcher")
    submit = squash(function_body(mmvq, "\nbool mmvq_submit_quant_aos_id("))
    if "caseGGML_TYPE_Q2_0:" not in submit or f"mmvq_submit_aos_id_impl<GGML_TYPE_Q2_0,{tuple_text},vec_dot_q2_0_q8_1>" not in submit:
        failures.append("mmvq_submit_quant_aos_id has no Q2_0 arm with the Q2_0 tuple")
    if "ncols%QK2_0!=0" not in submit:
        failures.append("the Q2_0 _id arm does not refuse an ncols that is not a multiple of QK2_0")
    return failures


def check_admission(ggml_sycl: str, get_rows: str) -> list:
    failures = []
    dense = function_body(ggml_sycl, "static bool ggml_sycl_mul_mat_type_supported(ggml_type type)")
    if "case GGML_TYPE_Q2_0:" not in dense:
        failures.append("ggml_sycl_mul_mat_type_supported does not admit Q2_0")
    rows = function_body(get_rows, "inline bool ggml_sycl_get_rows_type_supported(ggml_type type)")
    if "GGML_TYPE_Q2_0" in rows:
        failures.append("GET_ROWS admits Q2_0 but no ggml_sycl_op_get_rows arm exists for it")
    return failures


def all_failures() -> list:
    return (
        check_dequant(read("dequantize.hpp"), read("convert.cpp"))
        + check_vec_dot(read("vecdotq.hpp"))
        + check_mmvq(read("mmvq.cpp"))
        + check_admission(read("ggml-sycl.cpp"), read("get-rows-support.hpp"))
    )


def test_q2_0_is_wired_end_to_end() -> None:
    failures = all_failures()
    assert not failures, "\n".join(failures)


def test_gate_fails_on_each_missing_piece() -> None:
    """Every check above must be able to fail: strip each piece from the real source and expect red."""
    dequantize, convert = read("dequantize.hpp"), read("convert.cpp")
    vecdotq, mmvq = read("vecdotq.hpp"), read("mmvq.cpp")
    ggml_sycl, get_rows = read("ggml-sycl.cpp"), read("get-rows-support.hpp")
    mutants = {
        "converter case dropped": lambda: check_dequant(
            dequantize, convert.replace("case GGML_TYPE_Q2_0:\n            return dequantize_block_sycl<QK2_0, QR2_0, dequantize_q2_0>;", "", 1)),
        "vec_dot reads 4-byte aligned": lambda: check_vec_dot(
            vecdotq.replace("get_int_from_uint8(bq2_0->qs", "get_int_from_uint8_aligned(bq2_0->qs", 1)),
        "dense case dropped": lambda: check_mmvq(
            mmvq.replace('case GGML_TYPE_Q2_0:\n                GGML_SYCL_KTRACE("mmvq_q2_0"', 'case GGML_TYPE_TQ2_0:\n                GGML_SYCL_KTRACE("mmvq_q2_0"', 1)),
        "dense admission dropped": lambda: check_admission(
            ggml_sycl.replace("        case GGML_TYPE_Q2_0:\n        case GGML_TYPE_Q4_0:", "        case GGML_TYPE_Q4_0:", 1), get_rows),
        "get_rows admits Q2_0": lambda: check_admission(
            ggml_sycl, get_rows.replace("        case GGML_TYPE_Q4_K:\n        case GGML_TYPE_Q6_K:\n            return true;", "        case GGML_TYPE_Q4_K:\n        case GGML_TYPE_Q2_0:\n            return true;", 1)),
    }
    for label, run in mutants.items():
        assert run(), f"gate still passes under mutation: {label}"


if __name__ == "__main__":
    failed = 0
    for name, test in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
        try:
            test()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL: {name}: {exc}")
    if failed:
        sys.exit(1)
    print("test-sycl-q2-0-wiring-source: OK")
