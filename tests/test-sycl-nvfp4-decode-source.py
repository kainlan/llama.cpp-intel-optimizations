#!/usr/bin/env python3
"""Source gate: the NVFP4 dot product decodes its E2M1 codes arithmetically and builds its scale from bits.

vec_dot_nvfp4_q8_1 (ggml/src/ggml-sycl/vecdotq.hpp) serves every NVFP4 MMVQ launch: the dense mat-vec and
the expert (_id) launches of decode and of the prompt. It decoded its codes with
get_int_from_table_16(..., kvalues_mxfp4), six dpct::byte_level_permute calls per int, each emulated with
64-bit shifts on Intel GPUs; that decode is what made the MXFP4 mat-vec compute-bound before
vec_dot_mxfp4_q8_1 moved to get_int_from_mxfp4. NVFP4 uses the same code table, so the functor decodes its
codes with get_int_from_mxfp4 and never with the permute table.

Its UE4M3 scale, ggml_sycl_ue4m3_to_fp32 (common.hpp), comes once per 16 weights, twice MXFP4's rate. It was
three compares, a branch and a sycl::ldexp per scale; it is now a float read from bits put together from the
exponent and the mantissa, with selects and no branch, as MXFP4's E8M0 scale already is. The to-fp16
converter, the copy and the dequantize paths call the same helper.

This gate pins the form only. The values are tests/test-sycl-nvfp4-decode-host.cpp's job: it compares the two
code decodes, compares the scale bit for bit with a format reference for all 256 bytes and runs the functor
against an oracle, so a correct refactor of either function passes here and a wrong value fails there.

Runs under pytest and as a plain script. GGML_SYCL_NVFP4_DECODE_ROOT points the gate at another tree.
No device or build is touched.
"""

import os
import re
import sys
from pathlib import Path

ROOT = Path(os.environ.get("GGML_SYCL_NVFP4_DECODE_ROOT", str(Path(__file__).resolve().parents[1])))
SYCL = ROOT / "ggml/src/ggml-sycl"

FUNCTOR = "static __dpct_inline__ float vec_dot_nvfp4_q8_1("
SCALE = "static __dpct_inline__ float ggml_sycl_ue4m3_to_fp32(uint8_t x)"

# A float read from integer bits: memcpy into a float, or a bit_cast to float.
FROM_BITS = re.compile(r"memcpy\(&\w+,&\w+,sizeof\(float\)\)|bit_cast<float>\(")


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


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def code_only(text: str) -> str:
    """The text without comments and without whitespace, so a comment may name the old form."""
    return re.sub(r"\s+", "", strip_comments(text))


def check_codes(vecdotq: str) -> list:
    body = code_only(function_body(vecdotq, FUNCTOR))
    if not body:
        return ["vec_dot_nvfp4_q8_1 is missing from vecdotq.hpp"]
    failures = []
    if "get_int_from_table_16" in body:
        failures.append("vec_dot_nvfp4_q8_1 still decodes its codes with get_int_from_table_16 (emulated permutes)")
    if "get_int_from_mxfp4(" not in body:
        failures.append("vec_dot_nvfp4_q8_1 does not decode its codes with get_int_from_mxfp4")
    return failures


def check_scale(common: str) -> list:
    text = strip_comments(function_body(common, SCALE))
    body = re.sub(r"\s+", "", text)
    if not body:
        return ["ggml_sycl_ue4m3_to_fp32 is missing from common.hpp"]
    failures = []
    if "ldexp" in body:
        failures.append("ggml_sycl_ue4m3_to_fp32 still scales with ldexp; build the float from its bits")
    if re.search(r"\bif\b", text):
        failures.append("ggml_sycl_ue4m3_to_fp32 branches; a lane per scale wants selects")
    if not FROM_BITS.search(body):
        failures.append("ggml_sycl_ue4m3_to_fp32 does not read its normal value from bits (memcpy or bit_cast)")
    return failures


def test_nvfp4_functor_decodes_codes_arithmetically() -> None:
    failures = check_codes(read("vecdotq.hpp"))
    assert not failures, "\n".join(failures)


def test_ue4m3_scale_is_built_from_its_bits() -> None:
    failures = check_scale(read("common.hpp"))
    assert not failures, "\n".join(failures)


def test_gate_fails_on_each_mutation() -> None:
    """Every code check must be able to fail: change the real functor and expect red."""
    vecdotq = read("vecdotq.hpp")
    body = function_body(vecdotq, FUNCTOR)
    assert body, "vec_dot_nvfp4_q8_1 is missing from vecdotq.hpp"

    def with_body(new_body: str) -> list:
        return check_codes(vecdotq.replace(body, new_body, 1))

    def mutate(old: str, new: str) -> list:
        assert old in body, f"mutation anchor missing from vec_dot_nvfp4_q8_1: {old}"
        return with_body(body.replace(old, new, 1))

    mutants = {
        "first code int back on the table": lambda: mutate(
            "get_int_from_mxfp4(get_int_b4(bq4->qs, iqs0))",
            "get_int_from_table_16(get_int_b4(bq4->qs, iqs0), kvalues_mxfp4)"),
        "second code int back on the table": lambda: mutate(
            "get_int_from_mxfp4(get_int_b4(bq4->qs, iqs1))",
            "get_int_from_table_16(get_int_b4(bq4->qs, iqs1), kvalues_mxfp4)"),
    }
    for label, run in mutants.items():
        assert run(), f"gate still passes under mutation: {label}"
    assert not with_body(body + "\n    // was get_int_from_table_16\n    /* get_int_from_table_16 */\n"), \
        "a comment naming the old decode must not fail the gate"


def test_scale_gate_fails_on_each_mutation() -> None:
    """Every scale check must be able to fail: change the real helper and expect red."""
    common = read("common.hpp")
    body = function_body(common, SCALE)
    assert body, "ggml_sycl_ue4m3_to_fp32 is missing from common.hpp"

    def with_body(new_body: str) -> list:
        return check_scale(common.replace(body, new_body, 1))

    def mutate(old: str, new: str) -> list:
        assert old in body, f"mutation anchor missing from ggml_sycl_ue4m3_to_fp32: {old}"
        return with_body(body.replace(old, new, 1))

    mutants = {
        "subnormal through ldexp": lambda: mutate("(float) man * 0x1p-10f", "sycl::ldexp((float) man, -10)"),
        "an early return": lambda: mutate(
            "    return (x & 0x7fu) == 0x7fu ? 0.0f : value;",
            "    if ((x & 0x7fu) == 0x7fu) {\n        return 0.0f;\n    }\n    return value;"),
        "normal not read from its bits": lambda: mutate(
            "memcpy(&normal, &bits, sizeof(float));", "normal = (float) bits;"),
    }
    for label, run in mutants.items():
        assert run(), f"gate still passes under mutation: {label}"
    assert not with_body(body + "\n    // was: if (...) sycl::ldexp\n    /* if */\n"), \
        "a comment naming the old form must not fail the gate"
    assert not mutate("memcpy(&normal, &bits, sizeof(float));", "normal = sycl::bit_cast<float>(bits);"), \
        "reading the bits with bit_cast is the same form and must pass"


if __name__ == "__main__":
    failed = 0
    for name, function in list(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print(f"PASS {name}")
            except AssertionError as error:
                failed += 1
                print(f"FAIL {name}: {error}")
    sys.exit(1 if failed else 0)
