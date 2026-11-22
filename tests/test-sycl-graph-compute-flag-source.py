#!/usr/bin/env python3
"""The SYCL backend runs only the nodes a graph computes.

Host-only: reads sources, runs no build, loads no model and touches no device.

ggml_build_forward_select adds every branch it is given to the graph and marks only the selected one with
GGML_TENSOR_FLAG_COMPUTE. llama selects its input embeddings that way (tokens, raw embeddings, or a mixed batch), so
every graph of an architecture with mixed-batch support (the default in llm_arch_supports_mixed_batch) carries the
unselected branches: for a token batch, a GET_ROWS on inp_mixed_tokens and a SET_ROWS through inp_mixed_slots,
inputs llama does not write for that batch. The scheduler does not upload an input that only an uncomputed node
reads, and ggml-alloc frees an uncomputed dead end straight away, so its bytes belong to another tensor. ggml-cpu,
CUDA and upstream SYCL skip such a node, and the meta backend clears the flag to drop a node on one device. A backend
that runs it reads indices nobody wrote and writes through them.

Checks:
  * one rule: ggml_sycl_graph_node_is_computed (prestage-class.hpp) tests the flag, and ggml-sycl.cpp tests it
    nowhere else;
  * ggml_sycl_is_noop, the predicate every executor already asks, is true for a node the graph does not compute.
    The executors pinned here ask it: the node loop of graph_compute_impl, the segment dispatch
    (moe_graph_dispatch_direct_node), the MoE block graphs, the layer-block dispatch, and the Q8_1 src1 planner that
    sizes what those dispatches draw;
  * the four input readers (the pre-stage, its diagnostic source count, the replay refresh's discovery and the
    segment slot's input list) read a node's sources only when the graph computes that node, so an input that only
    an uncomputed node reads is neither staged, refreshed nor listed, and the four give one answer. A recorded kernel
    that looks up a staged input (bin-broadcast) runs only for a computed node, whose sources the pre-stage walked;
  * their leaf loops do not ask it: a leaf has no op and never carries the flag, so asking would drop every leaf.

--self-test runs each check against a mutant that must fail it.
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SYCL = ROOT / "ggml" / "src" / "ggml-sycl"

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--backend", default=str(SYCL / "ggml-sycl.cpp"))
parser.add_argument("--prestage-class", default=str(SYCL / "prestage-class.hpp"))
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


def balanced(source, start):
    """The brace-balanced block that opens at source[start] == '{'."""
    depth = 0
    for i in range(start, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start:i + 1]
    return None


def function_body(source, signature_regex):
    """The body of the first function whose signature matches, or None."""
    m = re.search(signature_regex, source)
    if not m:
        return None
    start = source.find("{", m.end() - 1 if source[m.end() - 1] == "{" else m.end())
    return balanced(source, start) if start >= 0 else None


def loop_body(body, bound):
    """The body of the first `for (int x = 0; x < cgraph->BOUND; ...)` loop in `body`, or None."""
    if body is None:
        return None
    m = re.search(r"for\s*\(\s*int\s+(\w+)\s*=\s*0\s*;\s*\1\s*<\s*cgraph->" + bound + r"\s*;[^)]*\)\s*\{", body)
    return balanced(body, m.end() - 1) if m else None


SKIP_UNCOMPUTED = re.compile(r"if\s*\(\s*!(\w+)\s*\|\|\s*!ggml_sycl_graph_node_is_computed\(\1\)\s*\)\s*\{\s*continue;\s*\}")


def skips_uncomputed_before_sources(loop):
    """The node loop leaves an uncomputed node before it reads any of that node's sources."""
    if loop is None:
        return False
    m = SKIP_UNCOMPUTED.search(loop)
    first_src = loop.find("src[")
    return m is not None and first_src > m.end()


WALKERS = {
    "the pre-stage": r"static bool graph_prestage_leaf_tensors\([^)]*\)\s*\{",
    "the pre-stage source count": r"static size_t ggml_sycl_graph_stage_source_count\([^)]*\)\s*\{",
    "the replay refresh discovery": r"static void graph_refresh_input_tensors\([^)]*\)\s*\{",
    "the segment slot's input list": r"static void moe_segment_slot_collect_inputs\([^)]*\)\s*\{",
}

# name -> (signature, the guard that asks ggml_sycl_is_noop, the same guard without it for the self-test)
EXECUTORS = {
    "the node loop of graph_compute_impl":
        (r"static void ggml_backend_sycl_graph_compute_impl\([^)]*\)\s*\{",
         r"if\s*\(\s*ggml_sycl_is_noop\(node\)\s*\)\s*\{\s*continue;\s*\}", ""),
    "the segment dispatch":
        (r"static bool moe_graph_dispatch_direct_node\([^)]*\)\s*\{",
         r"if\s*\(\s*!node\s*\|\|\s*ggml_sycl_is_noop\(node\)\s*\|\|", "if (!node ||"),
    "the MoE block-graph replay":
        (r"static void moe_graph_replay_block_graphs\([^)]*\)\s*\{",
         r"if\s*\(\s*!node\s*\|\|\s*ggml_sycl_is_noop\(node\)\s*\|\|", "if (!node ||"),
    "the MoE block-graph record":
        (r"static bool moe_graph_record_block_graphs\([^)]*\)\s*\{",
         r"if\s*\(\s*!node\s*\|\|\s*ggml_sycl_is_noop\(node\)\s*\|\|", "if (!node ||"),
    "the layer-block dispatch":
        (r"static bool ggml_sycl_block_exec_dispatch_node_on_device\([^)]*\)\s*\{",
         r"if\s*\(\s*!node\s*\|\|\s*ggml_sycl_is_noop\(node\)\s*\)", "if (!node)"),
    "the Q8_1 src1 planner":
        (r"static bool ggml_sycl_mmq_src1_ensure_for_graph\([^)]*\)\s*\{",
         r"node->op\s*!=\s*GGML_OP_MUL_MAT\s*\|\|\s*ggml_sycl_is_noop\(node\)", "node->op != GGML_OP_MUL_MAT"),
}


def evaluate(backend, class_hdr):
    results = {}

    helper = function_body(class_hdr, r"static inline bool ggml_sycl_graph_node_is_computed\(const ggml_tensor \* node\)\s*\{")
    results["prestage-class.hpp defines ggml_sycl_graph_node_is_computed"] = helper is not None
    results["the helper answers from GGML_TENSOR_FLAG_COMPUTE"] = helper is not None and re.fullmatch(
        r"\{\s*return\s*\(\s*node->flags\s*&\s*GGML_TENSOR_FLAG_COMPUTE\s*\)\s*!=\s*0\s*;\s*\}", helper) is not None
    results["ggml-sycl.cpp tests the flag only through the helper"] = "GGML_TENSOR_FLAG_COMPUTE" not in backend

    noop = function_body(backend, r"static inline bool ggml_sycl_is_noop\(const ggml_tensor \* node\)\s*\{")
    ret = re.search(r"return\b[^;]*;", noop).group(0) if noop and re.search(r"return\b[^;]*;", noop) else ""
    results["ggml_sycl_is_noop is true for a node the graph does not compute"] = \
        re.search(r"\|\|\s*!ggml_sycl_graph_node_is_computed\(node\)", ret) is not None or \
        re.search(r"return\s*!ggml_sycl_graph_node_is_computed\(node\)\s*\|\|", ret) is not None

    for name, (sig, pattern, _) in EXECUTORS.items():
        body = function_body(backend, sig)
        results[f"{name} asks ggml_sycl_is_noop"] = body is not None and re.search(pattern, body) is not None

    for name, sig in WALKERS.items():
        body = function_body(backend, sig)
        results[f"{name} reads only a computed node's sources"] = \
            skips_uncomputed_before_sources(loop_body(body, "n_nodes"))
        leafs = loop_body(body, "n_leafs")
        results[f"{name} keeps every leaf (a leaf never carries the flag)"] = \
            leafs is not None and "ggml_sycl_graph_node_is_computed" not in leafs

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


backend = strip_comments(Path(args.backend).read_text())
class_hdr = strip_comments(Path(args.prestage_class).read_text())
failed = run("tree", (backend, class_hdr))

if args.self_test:
    def mutate_in_func(src, sig_regex, pattern, repl):
        """Replace the first match of `pattern` after the first match of `sig_regex`."""
        m = re.search(sig_regex, src)
        pm = re.compile(pattern).search(src, m.end()) if m else None
        if not pm:
            print(f"FAIL: self-test anchor missing: {sig_regex!r} .. {pattern!r}")
            failed.append("self-test anchor " + pattern)
            return src
        return src[:pm.start()] + repl + src[pm.end():]

    def mutate_all_in_func(src, sig_regex, pattern, repl):
        """Replace every match of `pattern` inside the body of the first function matching `sig_regex`."""
        m = re.search(sig_regex, src)
        start = src.find("{", m.end() - 1) if m else -1
        body = balanced(src, start) if start >= 0 else None
        if body is None or not re.search(pattern, body):
            print(f"FAIL: self-test anchor missing: {sig_regex!r} .. {pattern!r}")
            failed.append("self-test anchor " + pattern)
            return src
        return src[:start] + re.sub(pattern, repl, body) + src[start + len(body):]

    noop_sig = r"static inline bool ggml_sycl_is_noop\(const ggml_tensor \* node\)\s*\{"
    helper_sig = r"static inline bool ggml_sycl_graph_node_is_computed\(const ggml_tensor \* node\)\s*\{"
    mutants = [
        ("the helper ignores the flag", "the helper answers from GGML_TENSOR_FLAG_COMPUTE",
         (backend, mutate_in_func(class_hdr, helper_sig, r"return[^;]*;", "return true;"))),
        ("the helper is gone", "prestage-class.hpp defines ggml_sycl_graph_node_is_computed",
         (backend, class_hdr.replace("ggml_sycl_graph_node_is_computed(", "ggml_sycl_graph_node_is_gone("))),
        ("the backend tests the flag inline", "ggml-sycl.cpp tests the flag only through the helper",
         (backend + "\nstatic bool f(const ggml_tensor * t) { return t->flags & GGML_TENSOR_FLAG_COMPUTE; }\n",
          class_hdr)),
        ("is_noop forgets uncomputed nodes", "ggml_sycl_is_noop is true for a node the graph does not compute",
         (mutate_in_func(backend, noop_sig, r"\s*\|\|\s*!ggml_sycl_graph_node_is_computed\(node\)", ""), class_hdr)),
        ("is_noop asks the rule outside its answer", "ggml_sycl_is_noop is true for a node the graph does not compute",
         (mutate_in_func(backend, noop_sig, r"\s*\|\|\s*!ggml_sycl_graph_node_is_computed\(node\)\s*;",
                         ";\n    (void) ggml_sycl_graph_node_is_computed(node);"), class_hdr)),
    ]
    for name, (sig, pattern, without) in EXECUTORS.items():
        mutants.append((f"{name} forgets ggml_sycl_is_noop", f"{name} asks ggml_sycl_is_noop",
                        (mutate_all_in_func(backend, sig, pattern, without), class_hdr)))
    for name, sig in WALKERS.items():
        mutants.append((f"{name} reads an uncomputed node's sources", f"{name} reads only a computed node's sources",
                        (mutate_in_func(backend, sig, SKIP_UNCOMPUTED.pattern, ""), class_hdr)))
        mutants.append((f"{name} drops leaves", f"{name} keeps every leaf (a leaf never carries the flag)",
                        (mutate_in_func(backend, sig,
                                        r"for\s*\(\s*int\s+(\w+)\s*=\s*0\s*;\s*\1\s*<\s*cgraph->n_leafs\s*;[^)]*\)\s*\{",
                                        "for (int i = 0; i < cgraph->n_leafs; ++i) {\n"
                                        "        if (!cgraph->leafs[i] || !ggml_sycl_graph_node_is_computed(cgraph->leafs[i])) { continue; }"),
                         class_hdr)))
    for label, expect, sources in mutants:
        failed += run(label, sources, expect)

if failed:
    print("\nFAILED: " + ", ".join(failed))
    sys.exit(1)
print("\nALL CHECKS PASSED")
