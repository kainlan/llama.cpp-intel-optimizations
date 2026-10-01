#!/usr/bin/env python3
"""SYCL GET_ROWS runs Q4_K on the device, from one shared support predicate (llama.cpp-qhfp).

Host-only: reads sources, runs no build, loads no model and touches no device. It pins the source
facts below; whether the kernel is numerically right is the host test's job (test-get-rows-q4-k.cpp)
and whether a graph replays is the GPU witness's.

The defect: ggml_backend_sycl_device_supports_op admitted GET_ROWS only for F16, F32, Q4_0, Q4_1,
Q5_0, Q5_1, Q8_0 and Q6_K. qwen35 (Qwen3.6-27B Q4_K_M) has a q4_K token_embd.weight that the
placement plan puts on the device, so the embedding GET_ROWS was declined, ggml_backend_sched ran it
on the CPU, and the SYCL split read its output out of pinned host memory. That is a zero-copy host
read, and the in-place allocator reuse also made the layer-0 residual ADD write the same host
buffer. Neither can be recorded into a command graph, so qwen35 decode never replayed. Mistral's
Q4_0 embedding is admitted, has no CPU split, and replays.

A missing kernel for the loaded (type, layout) is a support gap to close, so the fix is the kernel:
a Q4_K arm in ggml_sycl_op_get_rows. Embedding tables are materialised AoS (the layout policy's
EMBEDDING rule), and the arm refuses any other layout instead of reading it as AoS.

Pinned here:
  * supports_op asks the shared predicates (get-rows-support.hpp) and keeps no private type list;
  * the predicate admits Q4_K, and every type it admits has an arm in ggml_sycl_op_get_rows
    (admitting a type nothing computes is the worse failure);
  * (type, layout) pairs: Q4_K is admitted for the AoS layout only, and supports_op asks it with the
    layout the placement plan actually materialises the weight in, so a pair no kernel covers is
    declined before placement routes it. The dispatch arm asks the same predicate; its abort is a
    backstop for a pair supports_op already refused, never the way a pair is declined;
  * the streamed-slice dispatcher has the arm too;
  * the kernel decodes through the K-quant element function the host test checks against ggml, which
    unpacks scales through the one shared scale/min helper (Q5_K reuses it).

Run with --self-test to prove every check fires against a mutant of the thing it forbids; a check
that cannot fail is decoration.
"""
import argparse
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sycl = root / "ggml/src/ggml-sycl"

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default=str(sycl / "ggml-sycl.cpp"))
parser.add_argument("--getrows", default=str(sycl / "getrows.cpp"))
parser.add_argument("--support", default=str(sycl / "get-rows-support.hpp"))
parser.add_argument("--kquant", default=str(sycl / "get-rows-kquant.hpp"))
parser.add_argument("--self-test", action="store_true")
args = parser.parse_args()


def strip_comments(source):
    """Remove C/C++ comments, keeping string literals and the line count."""
    out = []
    i = 0
    n = len(source)
    while i < n:
        ch = source[i]
        if ch in ('"', "'"):
            quote = ch
            out.append(ch)
            i += 1
            while i < n:
                out.append(source[i])
                if source[i] == "\\":
                    if i + 1 < n:
                        out.append(source[i + 1])
                        i += 2
                        continue
                elif source[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        if source.startswith("//", i):
            while i < n and source[i] != "\n":
                i += 1
            continue
        if source.startswith("/*", i):
            end = source.find("*/", i + 2)
            end = n if end < 0 else end + 2
            out.append("\n" * source.count("\n", i, end))
            i = end
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def read(path):
    p = Path(path)
    return strip_comments(p.read_text()) if p.exists() else ""


def function_body(source, signature_regex):
    """Return the brace-balanced body of the first function matching the regex."""
    m = re.search(signature_regex, source)
    if not m:
        return None
    start = source.find("{", m.end() - 1 if source[m.end() - 1] == "{" else m.end())
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start:i + 1]
    return None


def case_arm(body, type_name):
    """Text of `case GGML_TYPE_<type_name>:` up to the next case/default label of the same switch level."""
    m = re.search(r"case\s+GGML_TYPE_%s\s*:" % type_name, body)
    if not m:
        return None
    nxt = re.search(r"\n\s*(case\s+GGML_TYPE_\w+\s*:|default\s*:)", body[m.end():])
    return body[m.start():m.end() + (nxt.start() if nxt else len(body))]


SUPPORT_SIG = r"inline bool ggml_sycl_get_rows_type_supported\([^)]*\)\s*\{"
LAYOUT_SIG = r"inline bool ggml_sycl_get_rows_layout_supported\([^)]*\)\s*\{"
SCALE_SIG = r"inline void ggml_sycl_kquant_scale_min_k4\([^)]*\)\s*\{"
ELEM_SIG = r"inline float ggml_sycl_get_rows_q4_k_elem\([^)]*\)\s*\{"
SUPPORTS_OP_SIG = r"static bool ggml_backend_sycl_device_supports_op\([^)]*\)\s*\{"
OP_SIG = r"void ggml_sycl_op_get_rows\([^)]*\)\s*\{"
SLICE_SIG = r"static void ggml_sycl_get_rows_dispatch_slice\([^)]*\)\s*\{"
KERNEL_SIG = r"static void k_get_rows_q4_k_aos\([^)]*\)\s*\{"


def evaluate(backend, getrows, support, kquant):
    results = {}

    pred = function_body(support, SUPPORT_SIG)
    supports_op = function_body(backend, SUPPORTS_OP_SIG)
    op = function_body(getrows, OP_SIG)
    slice_ = function_body(getrows, SLICE_SIG)
    results["anchor: the shared support predicate exists"] = pred is not None
    layout_pred = function_body(support, LAYOUT_SIG)
    results["anchor: the (type, layout) predicate exists"] = layout_pred is not None
    results["anchor: supports_op exists"] = supports_op is not None
    results["anchor: ggml_sycl_op_get_rows exists"] = op is not None
    results["anchor: the streamed-slice dispatcher exists"] = slice_ is not None
    if None in (pred, layout_pred, supports_op, op, slice_):
        return results

    # --- supports_op has one source for the GET_ROWS type list ----------------
    m = re.search(r"case\s+GGML_OP_GET_ROWS\s*:", supports_op)
    arm = supports_op[m.start():m.start() + 1200] if m else ""
    results["supports_op GET_ROWS asks the shared predicate"] = "ggml_sycl_get_rows_type_supported(" in arm
    results["supports_op GET_ROWS asks the (type, layout) predicate with the planned layout"] = \
        "ggml_sycl_get_rows_layout_supported(" in arm and "ggml_sycl_get_planned_weight_layout(" in arm
    results["supports_op keeps no private GET_ROWS type list"] = \
        m is not None and re.search(r"GGML_TYPE_Q4_0", arm) is None

    # --- the predicate admits Q4_K, and everything it admits has an arm -------
    admitted = re.findall(r"case\s+GGML_TYPE_(\w+)\s*:", pred)
    results["the predicate admits Q4_K"] = "Q4_K" in admitted
    missing = [t for t in admitted if case_arm(op, t) is None]
    results["every admitted type has an arm in ggml_sycl_op_get_rows"] = bool(admitted) and not missing

    # --- (type, layout): Q4_K is admitted for AoS only -------------------------
    q4k_layout = case_arm(layout_pred, "Q4_K")
    results["the layout predicate admits Q4_K for AoS only"] = \
        q4k_layout is not None and "GGML_LAYOUT_AOS" in q4k_layout and \
        re.search(r"GGML_LAYOUT_(SOA|COALESCED)", q4k_layout) is None

    # --- the Q4_K arm uses the predicate as a backstop --------------------------
    q4k = case_arm(op, "Q4_K")
    results["op: a Q4_K arm exists"] = q4k is not None
    if q4k is not None:
        refuse = re.search(r"ggml_sycl_get_rows_layout_supported\(", q4k)
        launch = re.search(r"get_rows_q4_k_aos_sycl\(", q4k)
        abort = re.search(r"GGML_ABORT", q4k)
        results["op: the Q4_K arm asks the layout predicate before it launches"] = \
            bool(refuse and launch and abort) and refuse.start() < launch.start()
    sq4k = case_arm(slice_, "Q4_K")
    results["slice: the streamed-slice dispatcher has a Q4_K arm"] = \
        sq4k is not None and "get_rows_q4_k_aos_sycl(" in sq4k

    # --- the kernel decodes through the function the host test checks ---------
    kernel = function_body(getrows, KERNEL_SIG)
    results["the Q4_K kernel exists"] = kernel is not None
    if kernel is not None:
        results["the Q4_K kernel decodes through the shared element function"] = \
            "ggml_sycl_get_rows_q4_k_elem(" in kernel

    scale = function_body(kquant, SCALE_SIG)
    elem = function_body(kquant, ELEM_SIG)
    results["the K-quant header has the shared scale/min helper"] = scale is not None
    results["the Q4_K element decode unpacks scales through the shared helper"] = \
        elem is not None and "ggml_sycl_kquant_scale_min_k4(" in elem
    return results


def run(label, sources, expect_fail=None):
    results = evaluate(*sources)
    failed = sorted(k for k, v in results.items() if not v)
    if expect_fail is None:
        for k in sorted(results):
            print(("PASS: " if results[k] else "FAIL: ") + k)
        return failed
    fired = expect_fail in failed
    print(("PASS" if fired else "FAIL") + f": mutant '{label}' fires '{expect_fail}'")
    return [] if fired else [label]


backend, getrows, support, kquant = read(args.backend), read(args.getrows), read(args.support), read(args.kquant)
failed = run("tree", (backend, getrows, support, kquant))

if args.self_test:

    def mutate(src, sig_regex, old, new):
        m = re.search(sig_regex, src)
        if not m or src.find(old, m.end()) < 0:
            print(f"FAIL: self-test anchor missing: {sig_regex!r} .. {old!r}")
            failed.append("self-test anchor " + old)
            return src
        k = src.find(old, m.end())
        return src[:k] + new + src[k + len(old):]

    def with_(**kw):
        return (kw.get("backend", backend), kw.get("getrows", getrows), kw.get("support", support),
                kw.get("kquant", kquant))

    mutants = [
        ("Q4_K dropped from the predicate", "the predicate admits Q4_K",
         with_(support=mutate(support, SUPPORT_SIG, "case GGML_TYPE_Q4_K:", "case GGML_TYPE_Q3_K:"))),
        ("private type list in supports_op", "supports_op keeps no private GET_ROWS type list",
         with_(backend=mutate(backend, SUPPORTS_OP_SIG, "case GGML_OP_GET_ROWS:",
                              "case GGML_OP_GET_ROWS: if (op->src[0]->type == GGML_TYPE_Q4_0) { return true; }"))),
        ("supports_op ignores the predicate", "supports_op GET_ROWS asks the shared predicate",
         with_(backend=mutate(backend, SUPPORTS_OP_SIG, "ggml_sycl_get_rows_type_supported(",
                              "ggml_sycl_get_rows_type_supportedX("))),
        ("supports_op ignores the planned layout",
         "supports_op GET_ROWS asks the (type, layout) predicate with the planned layout",
         with_(backend=mutate(backend, SUPPORTS_OP_SIG, "ggml_sycl_get_planned_weight_layout(",
                              "ggml_sycl_get_planned_weight_layoutX("))),
        ("admitted type without an arm", "every admitted type has an arm in ggml_sycl_op_get_rows",
         with_(getrows=mutate(getrows, OP_SIG, "case GGML_TYPE_Q5_1:", "case GGML_TYPE_IQ4_NL:"))),
        ("Q4_K admitted for SoA", "the layout predicate admits Q4_K for AoS only",
         with_(support=mutate(support, LAYOUT_SIG, "GGML_LAYOUT_AOS", "GGML_LAYOUT_SOA"))),
        ("op arm skips the layout predicate", "op: the Q4_K arm asks the layout predicate before it launches",
         with_(getrows=mutate(getrows, OP_SIG, "ggml_sycl_get_rows_layout_supported(",
                              "ggml_sycl_get_rows_layout_supportedX("))),
        ("streamed arm dropped", "slice: the streamed-slice dispatcher has a Q4_K arm",
         with_(getrows=mutate(getrows, SLICE_SIG, "case GGML_TYPE_Q4_K:", "case GGML_TYPE_IQ4_NL:"))),
        ("kernel stops using the shared decode", "the Q4_K kernel decodes through the shared element function",
         with_(getrows=mutate(getrows, KERNEL_SIG, "ggml_sycl_get_rows_q4_k_elem(",
                              "ggml_sycl_get_rows_q4_k_elemX("))),
        ("Q4_K decode keeps a private scale unpack",
         "the Q4_K element decode unpacks scales through the shared helper",
         with_(kquant=mutate(kquant, ELEM_SIG, "ggml_sycl_kquant_scale_min_k4(",
                             "ggml_sycl_private_scale_min("))),
    ]
    for label, expect, sources in mutants:
        failed += run(label, sources, expect)

if failed:
    print("\nFAILED: " + ", ".join(failed))
    sys.exit(1)
print("\nALL CHECKS PASSED")
