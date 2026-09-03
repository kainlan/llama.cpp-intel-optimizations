#!/usr/bin/env python3
"""Source gate for llama.cpp-nz1k (prefill L2b phase 1): the Q8_0 SOA oneDNN
WoQ-int8 PP arm must be reachable ONLY under BOTH opt-in env conditions --
GGML_SYCL_Q8_DENSE_LAYOUT=soa (the weight is materialized SOA) and
GGML_SYCL_ONEDNN_WOQ_Q8=1 (the arm itself) -- and both must default OFF.

Ruling 9 / nz1k c-c4jp: phase 1 stages the scale plane per call, which is an
opt-in interim only; the default is decided on llama.cpp-2zsc with the A/B
numbers. This file exists so a quiet default flip, or a new eligibility site
for ONEDNN_SOA that forgets the env gate, fails a test instead of shipping.

Point it at alternate copies (to exercise the RED path with a deliberately
broken tree) via GGML_SYCL_WOQ_Q8_BACKEND_SOURCE / GGML_SYCL_WOQ_Q8_COMMON_SOURCE;
defaults are the in-tree files. No SYCL device or build is touched.
"""

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = Path(os.environ.get("GGML_SYCL_WOQ_Q8_BACKEND_SOURCE", str(ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp")))
COMMON = Path(os.environ.get("GGML_SYCL_WOQ_Q8_COMMON_SOURCE", str(ROOT / "ggml/src/ggml-sycl/common.hpp")))

backend = BACKEND.read_text()
common = COMMON.read_text()

failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)


def matching_brace(text, open_idx):
    """Comment/string-aware brace match (self-contained copy, per this fork's
    one-file-per-gate convention)."""
    assert text[open_idx] == "{"
    depth = 0
    state = "code"
    i = open_idx
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if state == "code":
            if ch == "/" and nxt == "/":
                state = "line"
                i += 2
                continue
            if ch == "/" and nxt == "*":
                state = "block"
                i += 2
                continue
            if ch == '"':
                state = "str"
            elif ch == "'":
                state = "chr"
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i
        elif state == "line":
            if ch == "\n":
                state = "code"
        elif state == "block":
            if ch == "*" and nxt == "/":
                state = "code"
                i += 2
                continue
        elif state == "str":
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                state = "code"
        elif state == "chr":
            if ch == "\\":
                i += 2
                continue
            if ch == "'":
                state = "code"
        i += 1
    raise AssertionError("unbalanced braces")


def function_body(text, signature):
    idx = text.find(signature)
    check(idx >= 0, f"missing definition: {signature}")
    if idx < 0:
        return ""
    open_idx = text.find("{", idx)
    return text[open_idx : matching_brace(text, open_idx) + 1]


# 1. The arm env accessor exists once, reads GGML_SYCL_ONEDNN_WOQ_Q8, defaults OFF.
ARM_SIG = "static bool ggml_sycl_onednn_woq_q8_enabled() {"
check(backend.count(ARM_SIG) == 1, "ggml_sycl_onednn_woq_q8_enabled must be defined exactly once")
arm_body = function_body(backend, ARM_SIG)
check('std::getenv("GGML_SYCL_ONEDNN_WOQ_Q8")' in arm_body, "arm accessor must read GGML_SYCL_ONEDNN_WOQ_Q8")
check(
    "(env != nullptr && std::atoi(env) != 0) ? 1 : 0" in arm_body,
    "GGML_SYCL_ONEDNN_WOQ_Q8 must default OFF (unset => 0); ruling 9 / nz1k c-c4jp",
)

# 2. The layout env: read once in common.hpp, default coalesced (only the exact
#    string "soa" selects SOA), and the SOA return is used in the dense blocks.
check(
    'std::getenv("GGML_SYCL_Q8_DENSE_LAYOUT")' in common,
    "common.hpp must read GGML_SYCL_Q8_DENSE_LAYOUT",
)
check(
    'q8_dense_soa_cached = (env && std::strcmp(env, "soa") == 0) ? 1 : 0;' in common,
    "GGML_SYCL_Q8_DENSE_LAYOUT must default to coalesced (only \"soa\" selects SOA)",
)
soa_returns = common.count("if (qtype == GGML_TYPE_Q8_0 && q8_dense_soa_cached) {\n                return GGML_LAYOUT_SOA;")
check(soa_returns == 2, f"expected the SOA layout return in the ATTENTION/FFN and OUTPUT blocks (2), found {soa_returns}")
get_optimal = function_body(common, "static layout_mode get_optimal(ggml_type qtype, tensor_usage usage, int device_id = -1) {")
emb_idx = get_optimal.find("if (usage == tensor_usage::EMBEDDING) {")
check(emb_idx > 0, "get_optimal EMBEDDING block not found")
if emb_idx > 0:
    emb_block = get_optimal[emb_idx : matching_brace(get_optimal, get_optimal.find("{", emb_idx)) + 1]
    check("q8_dense_soa_cached" not in emb_block, "GGML_SYCL_Q8_DENSE_LAYOUT must not touch EMBEDDING layouts (nz1k scope)")

# 3. Every ONEDNN_SOA eligibility site is gated on the arm env. The sites are
#    the two `case ggml_sycl_mul_mat_kernel::ONEDNN_SOA:` switch arms (override
#    check + preferred-kernel selection) and the pick_kernel_for_layout SOA arm.
case_sites = [m.start() for m in re.finditer(r"case ggml_sycl_mul_mat_kernel::ONEDNN_SOA:", backend)]
# Four sites: kernel-name table, override check, preferred-kernel selection,
# dispatch. The two SELECTION sites must consult the env gate; the name table
# and the dispatch arm (which executes an already-selected kernel) do not.
check(
    len(case_sites) == 4,
    f"expected 4 `case ONEDNN_SOA:` sites (name table, override check, preferred selection, dispatch), found {len(case_sites)}",
)
kinds = {"gated": 0, "dispatch": 0, "name": 0}
for start in case_sites:
    # the case body runs to the next `case`/`default` at the same switch level
    end = re.search(r"\n\s*(case |default:)", backend[start + 10 :])
    body = backend[start : start + 10 + (end.start() if end else 800)]
    if "ggml_sycl_onednn_woq_q8_enabled()" in body:
        kinds["gated"] += 1
    elif "ggml_sycl_op_mul_mat<no_quantize_q8_1>" in body:
        kinds["dispatch"] += 1
    elif 'return "ONEDNN_SOA";' in body:
        kinds["name"] += 1
    else:
        failures.append("an ONEDNN_SOA case body is neither env-gated, the dispatch arm, nor the name table")
check(kinds["gated"] == 2, f"both ONEDNN_SOA selection sites must consult the env gate (found {kinds['gated']})")
check(kinds["dispatch"] == 1 and kinds["name"] == 1, f"unexpected ONEDNN_SOA case census: {kinds}")

pick_idx = backend.find("auto pick_kernel_for_layout = [&](layout_mode layout) -> std::optional<ggml_sycl_mul_mat_kernel> {")
check(pick_idx > 0, "pick_kernel_for_layout lambda not found")
if pick_idx > 0:
    pick = backend[pick_idx : matching_brace(backend, backend.find("{", pick_idx)) + 1]
    soa_case = pick[pick.find("case GGML_LAYOUT_SOA:") : pick.find("case GGML_LAYOUT_AOS:")]
    assign = soa_case.find("layout_kernel = ggml_sycl_mul_mat_kernel::ONEDNN_SOA;")
    check(assign > 0, "pick_kernel_for_layout SOA case must be able to select ONEDNN_SOA")
    if assign > 0:
        guard = soa_case[:assign]
        check(
            "src0->type == GGML_TYPE_Q8_0" in guard and "ggml_sycl_onednn_woq_q8_enabled()" in guard,
            "pick_kernel_for_layout's ONEDNN_SOA choice must be guarded by Q8_0 && the env gate",
        )
    check(soa_case.count("ONEDNN_SOA") == 1, "ONEDNN_SOA must be selected from exactly one branch of the SOA case")

# 4. The GEMM call site: exactly one, inside ggml_sycl_op_mul_mat_sycl, guarded
#    by Q8_0 && the env gate, and consuming the SOA plane (not the AOS pointer).
check(backend.count("DnnlGemmWrapper::woq_gemm_q8_0(") == 1, "woq_gemm_q8_0 must be called from exactly one site")
op_body = function_body(backend, "inline void ggml_sycl_op_mul_mat_sycl(ggml_backend_sycl_context & ctx,")
call = op_body.find("DnnlGemmWrapper::woq_gemm_q8_0(")
check(call > 0, "the woq_gemm_q8_0 call must live in ggml_sycl_op_mul_mat_sycl")
if call > 0:
    before = op_body[:call]
    guard_idx = before.rfind("if (!used_woq && src0->type == GGML_TYPE_Q8_0 && ggml_sycl_onednn_woq_q8_enabled()")
    check(guard_idx > 0, "the arm must be guarded by `!used_woq && Q8_0 && ggml_sycl_onednn_woq_q8_enabled()`")
    if guard_idx > 0:
        arm = before[guard_idx:]
        check(
            "ggml_sycl_get_weight_layout_ptr(src0, ctx.device, GGML_LAYOUT_SOA)" in arm,
            "the arm must consume the SOA-materialized plane (GGML_LAYOUT_SOA lookup)",
        )
        check('decline = "soa_plane_not_resident"' in arm, "a missing SOA plane must decline, not proceed")
        check('decline = "no_pp_scratch"' in arm, "scales must be staged only into the existing PP scratch (ruling 1)")
    after = op_body[call:]
    check(
        "dequantize_row_q8_0_soa_to_fp16_rowmajor(q8_0_soa_ptr" in after,
        "the decline path must dequantize from the SOA plane, never the AOS dispatcher",
    )

# 5. No other SOA-scale binding: the [N][K/32] order must never reach oneDNN
#    directly (probe V5 accepted-but-garbage). The only scale-staging producer
#    is the transpose kernel.
check(
    backend.count("q8_0_soa_scale_plane_to_kbn_sycl(") == 1,
    "exactly one scale-staging call (the transpose into PP scratch) is expected",
)

if failures:
    for f in failures:
        print("FAIL:", f)
    sys.exit(1)
print("PASS: llama.cpp-nz1k oneDNN WoQ-Q8 arm is opt-in on both axes and reachable only through the SOA plane")
