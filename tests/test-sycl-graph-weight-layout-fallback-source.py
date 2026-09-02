#!/usr/bin/env python3
"""MUL_MAT's weight-layout resolution must not silently drop a dispatch under
SYCL graph recording, and its generic BLAS fallback must not silently skip
computing a tensor.

llama.cpp-dyi3 round 6 root cause: ggml_sycl_get_weight_layout_ptr() had a
"device-resident source, no host preference needed" fast path (an already-
device-resident, AOS-target pointer is directly usable) placed AFTER a
recording-only early-return that consults nothing but the unified cache's
name-keyed view lookup. gemma4's kmeq BF16->F32 materialization synthesizes
an alias tensor (name "<name>.bf16_materialized_f32", extra=nullptr, data
pointing at the already-materialized F32 device buffer) under a cache key
that is DESIGNED to miss that name-keyed lookup. Under recording the early
return therefore returned nullptr for that tensor -- never reaching the fast
path that would have returned its data pointer correctly -- and the caller's
generic BLAS fallback (the last route in the dispatch chain) then also
declined without its return value being checked, so the failed MUL_MAT
silently left its destination unwritten. gemma4's per-layer-embedding
projection went missing on every recorded/replayed decode token,
deterministically, from the first recorded token onward.

Both defects are ordering/return-value properties of the source, not
something a device test can exercise without a live SYCL device -- this
follows the fork's source-gate convention (see
test-sycl-buffer-base-alignment-source.py for the established pattern this
file borrows its function-extraction and mutation-witness helpers from).
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"

WEIGHT_LAYOUT_FN = "void * ggml_sycl_get_weight_layout_ptr(const ggml_tensor * tensor, int device, layout_mode target, bool prefer_host)"
DEVICE_RESIDENT_MARKER = "if (src_is_device && !request_prefer_host) {"
RECORDING_GATE_MARKER = "if (ggml_sycl_graph_recording_active()) {"

# ggml_sycl_mul_mat has a forward declaration (~line 48137, terminated by
# ";") whose first line is byte-identical to the real definition's (~line
# 59952, terminated by "= nullptr) {"). function() locates its target via
# text.index(signature) -- the FIRST match -- so a prefix ending after just
# "& ctx," resolves to the declaration, and the next "{" after that point
# belongs to a wholly different function (ggml_sycl_take_owned_alloc_handle)
# that happens to follow it, silently handing back a ~768-char bogus "body".
# The anchor must therefore run through the full multi-line parameter list
# to the definition's distinguishing "= nullptr) {" tail, which the
# semicolon-terminated declaration can never match.
MUL_MAT_DISPATCH_FN = (
    "static void ggml_sycl_mul_mat(ggml_backend_sycl_context & ctx,\n"
    "                              const ggml_tensor *         src0,\n"
    "                              const ggml_tensor *         src1,\n"
    "                              ggml_tensor *               dst,\n"
    "                              const layout_mode *         forced_layout = nullptr) {"
)
FALLBACK_CALL = "ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_sycl,"


def function(text: str, signature: str) -> str:
    start = text.index(signature)
    brace = text.index("{", start)
    depth = 0
    state = "code"
    i = brace
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if state == "code":
            if ch == "/" and nxt == "/":
                state = "line"
                i += 1
            elif ch == "/" and nxt == "*":
                state = "block"
                i += 1
            elif ch == '"':
                state = "string"
            elif ch == "'":
                state = "char"
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        elif state == "line":
            if ch == "\n":
                state = "code"
        elif state == "block":
            if ch == "*" and nxt == "/":
                state = "code"
                i += 1
        else:
            quote = '"' if state == "string" else "'"
            if ch == "\\":
                i += 1
            elif ch == quote:
                state = "code"
        i += 1
    raise AssertionError(f"unclosed function {signature}")


def function_or_none(text: str, signature: str) -> str | None:
    # A missing function is a violation to report, not an exception to raise:
    # the checker must stay usable against a tree that predates the guard.
    if signature not in text:
        return None
    return function(text, signature)


def weight_layout_violations(source: str) -> list[str]:
    """The device-resident fast path must be reachable under recording too."""
    found: list[str] = []

    body = function_or_none(source, WEIGHT_LAYOUT_FN)
    if body is None:
        return [f"{WEIGHT_LAYOUT_FN} is missing"]

    device_idx = body.find(DEVICE_RESIDENT_MARKER)
    recording_idx = body.find(RECORDING_GATE_MARKER)
    if device_idx == -1:
        found.append("the device-resident fast path (src_is_device && !request_prefer_host) is missing")
    if recording_idx == -1:
        found.append("the graph-recording early return is missing")
    if device_idx != -1 and recording_idx != -1 and device_idx > recording_idx:
        found.append(
            "the device-resident fast path comes AFTER the recording gate -- a tensor whose "
            "cache-keyed lookup misses by design (e.g. a materialized-alias name) resolves to "
            "nullptr under recording instead of reaching the fast path's direct pointer return"
        )

    # The fast path must still return the raw src pointer for a null-extra
    # alias tensor (extra->layout hit is only a cache-freshness shortcut, not
    # a gate on the return) -- this is what makes it correct for kmeq's
    # extra=nullptr materialized-weight alias specifically.
    if device_idx != -1:
        aos_return = re.search(
            r"if \(resolved == GGML_LAYOUT_AOS\) \{.*?return const_cast<void \*>\(src_ptr\);\s*\}",
            body[device_idx:], re.S,
        )
        if not aos_return:
            found.append("the device-resident fast path no longer returns src_ptr unconditionally for AOS")

    return found


def fallback_return_violations(source: str) -> list[str]:
    """The generic BLAS fallback -- the last route in the dispatch chain --
    must check ggml_sycl_op_mul_mat's return value rather than discard it.
    A silent false here means the destination tensor is never written."""
    found: list[str] = []

    body = function_or_none(source, MUL_MAT_DISPATCH_FN)
    if body is None:
        return [f"{MUL_MAT_DISPATCH_FN} is missing"]

    if FALLBACK_CALL not in body:
        return found + [f"the generic BLAS fallback call ({FALLBACK_CALL} ...) is missing"]

    # The call must appear inside an `if (!...)` (or equivalent) whose body
    # reports a failure -- not as a bare statement whose result is thrown
    # away. Token-anchored: look for the call preceded by "if (!" within a
    # short window (allows a line-wrapped condition) and followed by an
    # abort/error path before the enclosing block's own `return;`.
    call_idx = body.index(FALLBACK_CALL)
    window_before = body[max(0, call_idx - 80):call_idx]
    if "if (!" not in window_before:
        found.append(
            "the generic BLAS fallback call's return value is discarded (not guarded by `if (!...)`) "
            "-- this is the exact defect that let gemma4's per-layer-embedding projection silently "
            "fail to compute under SYCL graph recording"
        )
    else:
        window_after = body[call_idx:call_idx + 600]
        if "GGML_ABORT" not in window_after:
            found.append(
                "the generic BLAS fallback's failure branch does not GGML_ABORT -- a silently skipped "
                "MUL_MAT produced deterministic wrong output that survived replay, rerecord, and every "
                "other instrument; a crash is the correct failure mode here, not a quiet decline"
            )

    return found


def test_weight_layout_device_resident_path_precedes_recording_gate() -> None:
    assert weight_layout_violations(SOURCE.read_text()) == []


def test_generic_blas_fallback_checks_its_return_value() -> None:
    assert fallback_return_violations(SOURCE.read_text()) == []


def test_mutations_are_witnessed() -> None:
    source = SOURCE.read_text()

    # Reordering mutation: put the recording gate back in front of the
    # device-resident fast path (the llama.cpp-dyi3 round-6 defect).
    weight_fn_before = function(source, WEIGHT_LAYOUT_FN)
    device_block_match = re.search(
        r"if \(src_is_device && !request_prefer_host\) \{.*?\n    \}\n",
        weight_fn_before, re.S,
    )
    assert device_block_match, "could not isolate the device-resident fast-path block for the reorder mutation"
    device_block = device_block_match.group(0)
    swapped_fn = weight_fn_before.replace(device_block, "", 1)
    # Re-insert it immediately after the recording gate's closing brace so the
    # mutation is a pure reorder, not a deletion (the deletion case is caught
    # separately by the "is missing" branch above).
    recording_gate_match = re.search(
        r"if \(ggml_sycl_graph_recording_active\(\)\) \{.*?\n    \}\n", swapped_fn, re.S,
    )
    assert recording_gate_match, "could not isolate the recording gate for the reorder mutation"
    swapped_fn = swapped_fn.replace(
        recording_gate_match.group(0), recording_gate_match.group(0) + device_block, 1
    )
    reordered_source = source.replace(weight_fn_before, swapped_fn, 1)
    assert reordered_source != source, "reorder mutation did not change the source"
    assert weight_layout_violations(reordered_source), "reorder mutation was not witnessed"

    # Discard-the-return-value mutation on the generic BLAS fallback (the
    # exact form the code had before this round's fix).
    discarded = source.replace(
        "if (!ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_sycl,\n"
        "                                                        GGML_LAYOUT_AOS)) {\n"
        "                GGML_ABORT(\n"
        "                    \"[MUL_MAT] generic BLAS fallback did not compute %s (type=%d, recording=%d) -- dst was left \"\n"
        "                    \"unwritten instead of silently succeeding; see llama.cpp-dyi3\",\n"
        "                    src0->name ? src0->name : \"?\", (int) src0->type, ggml_sycl_graph_recording_active() ? 1 : 0);\n"
        "            }",
        "ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_sycl, GGML_LAYOUT_AOS);",
        1,
    )
    assert discarded != source, "discard-return-value mutation did not change the source"
    assert fallback_return_violations(discarded), "discard-return-value mutation was not witnessed"
