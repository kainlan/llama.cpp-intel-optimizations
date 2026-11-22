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
  * ggml_sycl_is_noop is true for a node the graph does not compute. The executors pinned here ask it: the node
    loop of graph_compute_impl, the segment dispatch (moe_graph_dispatch_direct_node), the MoE block graphs, the
    layer-block dispatch, and the Q8_1 src1 planner that sizes what those dispatches draw;
  * the walkers that choose, stage or plan what runs leave an uncomputed node before they read it: every node loop
    of the MoE topology scan (g_moe_gate_up_pairs) and of the MoE dispatch list (whose dispatch_indices the segment
    recorder, the block graphs, the graphlets and the decode analysis take); the layer-block pre-block copy, the
    layer-block plan's source scan and its alias publication; the persistent TG plan's fast path and full walk, the
    split plan's weight list, and the eligibility checks before them (decode, architecture, op support, MoE probe,
    policy summary), so an unselected branch's DUP no longer declines persistent TG;
  * the persistent TG answer, its plan and the plan's update recipe are keyed on the graph signature, which mixes each
    node's flags, as well as on n_nodes and the phase: a new key drops the cached plans (the drop a graph leaving the
    persistent path takes) before the graph is judged. A recipe replay takes each node through one lookup that
    refuses an index outside the graph or an uncomputed node, and such a node fails the replay into a full rebuild;
  * a fusion that also runs a node it looked ahead to asks the rule of that node: the TG router and MUL_MAT+ADD
    fusions, ADD+RMS_NORM, MUL+ADD, the FFN fusion, the MoE down-sum chain and its weighted MUL, and the persistent
    plan's RMS_NORM+MUL. The two CPU-path fusions ask ggml_sycl_is_noop of the next node. The RMS_NORM fusions that
    go through ggml_can_fuse_subgraph are covered by it: it refuses an uncomputed node;
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


def loops_in(body, header):
    """(start, end) of the body of every loop in `body` whose header matches `header`."""
    spans = []
    if body is None:
        return spans
    for m in re.finditer(header, body):
        start = body.find("{", m.start())
        block = balanced(body, start) if start >= 0 else None
        if block is not None:
            spans.append((start, start + len(block)))
    return spans


def leaves_uncomputed_first(loop):
    """The loop leaves an uncomputed node before it reads anything of that node."""
    m = SKIP_UNCOMPUTED.search(loop)
    if m is None:
        return False
    first_read = re.search(r"\b" + m.group(1) + r"->", loop)
    return first_read is None or first_read.start() > m.end()


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

ZERO_TO_N_NODES = r"for\s*\(\s*int\s+(\w+)\s*=\s*0\s*;\s*\1\s*<\s*cgraph->n_nodes\s*;[^)]*\)\s*\{"
PERSISTENT_PLAN = r"static bool extract_persistent_plan\([^)]*\)\s*\{"

# name -> (signature, the header of each loop that must leave an uncomputed node first, the least number of such loops)
NODE_WALKS = {
    "the MoE topology scan":
        (r"static void moe_layer_scan_graph_topology\([^)]*\)\s*\{", ZERO_TO_N_NODES, 4),
    "the MoE dispatch list":
        (r"static moe_graph_dispatch_plan moe_graph_collect_dispatch_indices\([^)]*\)\s*\{", ZERO_TO_N_NODES, 4),
    "the layer-block pre-block copy":
        (r"static bool ggml_sycl_block_exec_stage_preblock_inputs\([^)]*\)\s*\{",
         r"for\s*\(\s*int\s+i\s*=\s*exec_ctx\.block_first\[block_idx\]\s*;", 1),
    "the persistent plan's fast path":
        (PERSISTENT_PLAN, r"for\s*\(\s*int\s+i\s*=\s*0\s*;\s*i\s*<\s*cgraph->n_nodes\s*&&\s*fast_path_ok\s*;", 1),
    "the persistent plan's full walk":
        (PERSISTENT_PLAN,
         r"for\s*\(\s*int\s+i\s*=\s*0\s*;\s*i\s*<\s*cgraph->n_nodes\s*;\s*i\+\+\s*\)\s*\{\s*"
         r"ggml_tensor\s*\*\s*node\s*=\s*cgraph->nodes\[i\]\s*;\s*const\s+int\s+ops_before\b", 1),
    "the persistent split plan's weight list":
        (r"static bool extract_persistent_plan_split\([^)]*\)\s*\{", ZERO_TO_N_NODES, 1),
    "the persistent TG decode check":
        (r"static bool is_decode_phase_graph\([^)]*\)\s*\{", ZERO_TO_N_NODES, 1),
    "the persistent TG architecture check":
        (r"static bool has_supported_architecture\([^)]*\)\s*\{", ZERO_TO_N_NODES, 1),
    "the persistent TG op check":
        (r"static bool graph_has_only_persistent_ops\([^)]*\)\s*\{", ZERO_TO_N_NODES, 2),
    "the persistent TG MoE probe":
        (r"static persistent_moe_graph_probe persistent_tg_probe_moe_graph\([^)]*\)\s*\{", ZERO_TO_N_NODES, 1),
    "the persistent TG policy summary":
        (r"static persistent_tg_policy_summary persistent_tg_summarize_policy\([^)]*\)\s*\{", ZERO_TO_N_NODES, 1),
    "the layer-block plan's source scan":
        (r"static bool ggml_sycl_block_exec_prepare_context\([^)]*\)\s*\{",
         r"for\s*\(\s*int\s+i\s*=\s*0\s*;\s*i\s*<\s*cgraph->n_nodes\s*;\s*\+\+i\s*\)\s*\{\s*"
         r"const\s+ggml_tensor\s*\*\s*dst\s*=\s*cgraph->nodes\[i\]\s*;", 1),
    "the layer-block alias publication":
        (r"static bool ggml_sycl_block_exec_publish_aliases\([^)]*\)\s*\{", ZERO_TO_N_NODES, 1),
}

GRAPH_COMPUTE = r"static void ggml_backend_sycl_graph_compute_impl\([^)]*\)\s*\{"

# name -> (signature, the check of the look-ahead node, the same text without it for the self-test, least count)
LOOKAHEADS = {
    "the TG router fusion":
        (r"static bool ggml_sycl_try_fuse_tg_router_f32_add_argsort\([^)]*\)\s*\{",
         r"!ggml_sycl_graph_node_is_computed\(add\)\s*\|\|\s*!ggml_sycl_graph_node_is_computed\(sort\)", "false", 1),
    "the TG MUL_MAT+ADD fusion":
        (r"static bool ggml_sycl_try_fuse_tg_mul_mat_add\([^)]*\)\s*\{",
         r"!ggml_sycl_graph_node_is_computed\(add\)\s*\|\|\s*mul->op\b", "mul->op", 1),
    "the ADD+RMS_NORM fusion":
        (GRAPH_COMPUTE, r"ggml_sycl_graph_node_is_computed\(next\)\s*&&\s*next->op\s*==\s*GGML_OP_RMS_NORM\b",
         "next->op == GGML_OP_RMS_NORM", 1),
    "the MUL+ADD fusion":
        (GRAPH_COMPUTE, r"ggml_sycl_graph_node_is_computed\(next\)\s*&&\s*next->op\s*==\s*GGML_OP_ADD\b",
         "next->op == GGML_OP_ADD", 1),
    "the FFN fusion":
        (GRAPH_COMPUTE, r"glu_info\.glu\s*!=\s*nullptr\s*&&\s*ggml_sycl_graph_node_is_computed\(glu_info\.glu\)",
         "glu_info.glu != nullptr", 1),
    "each of the two CPU-path fusions":
        (GRAPH_COMPUTE, r"next\s*&&\s*!ggml_sycl_is_noop\(next\)", "next", 2),
    "the MoE down-sum chain":
        (r"static int ggml_sycl_match_moe_down_sum_chain\([^)]*\)\s*\{",
         r"!ggml_sycl_graph_node_is_computed\(add\)\s*\|\|\s*add->op\s*!=\s*GGML_OP_ADD\b", "add->op != GGML_OP_ADD", 1),
    "the MoE down-sum weighted MUL":
        (r"static bool ggml_sycl_try_fuse_moe_down_weighted_sum\([^)]*\)\s*\{",
         r"!ggml_sycl_graph_node_is_computed\(mul\)\s*\|\|\s*mul->op\s*!=\s*GGML_OP_MUL\b", "mul->op != GGML_OP_MUL", 1),
    "the persistent plan's RMS_NORM+MUL fusion":
        (PERSISTENT_PLAN, r"ggml_sycl_graph_node_is_computed\(next\)\s*&&\s*next->op\s*==\s*GGML_OP_MUL\b",
         "next->op == GGML_OP_MUL", 1),
}

GRAPH_COMPUTE_UNCHECKED = r"static ggml_status ggml_backend_sycl_graph_compute_unchecked\([^)]*\)\s*\{"
DROP_PLANS = r"static void persistent_tg_drop_cached_plans\([^)]*\)\s*\{"

# name -> (signature, the statement that must be in its body, the same text without it for the self-test)
PERSISTENT_KEY = {
    "the persistent eligibility key carries the graph signature":
        (GRAPH_COMPUTE_UNCHECKED,
         r"persistent_cache_sig\s*=\s*ggml_sycl::env_persistent_tg_enabled\(\)\s*\?\s*"
         r"ggml_sycl_graph_signature\(cgraph\)\s*:\s*0\s*;", "persistent_cache_sig = 0;"),
    "a persistent eligibility hit needs the same graph signature":
        (GRAPH_COMPUTE_UNCHECKED,
         r"persistent_cache_hit\s*=\s*sycl_ctx->cached_persistent_n_nodes\s*==\s*persistent_cache_key\s*&&\s*"
         r"sycl_ctx->cached_persistent_sig\s*==\s*persistent_cache_sig\s*;",
         "persistent_cache_hit = sycl_ctx->cached_persistent_n_nodes == persistent_cache_key;"),
    "a persistent eligibility miss drops the cached plans before it judges the graph":
        (GRAPH_COMPUTE_UNCHECKED,
         r"\}\s*else\s*\{\s*persistent_tg_drop_cached_plans\(\*sycl_ctx\)\s*;\s*"
         r"persistent_eligible\s*=\s*should_use_persistent_tg\(",
         "} else {\n        persistent_eligible = should_use_persistent_tg("),
    "a persistent eligibility miss stores the graph signature":
        (GRAPH_COMPUTE_UNCHECKED, r"sycl_ctx->cached_persistent_sig\s*=\s*persistent_cache_sig\s*;", ""),
    "leaving the persistent path drops the cached plans":
        (GRAPH_COMPUTE_UNCHECKED, r"normal_dispatch:\s*persistent_tg_drop_cached_plans\(\*sycl_ctx\)\s*;",
         "normal_dispatch:\n"),
    "the plan drop drops the primary kernel's plan":
        (DROP_PLANS, r"ctx\.unified_kernel->invalidate_plan_cache\(\)\s*;", ""),
    "the plan drop drops the split secondary kernel's plan":
        (DROP_PLANS, r"g_split_secondary_kernel->invalidate_plan_cache\(\)\s*;", ""),
    "the plan drop drops the split plan's secondary ops":
        (DROP_PLANS, r"g_split_plan_cached\s*=\s*false\s*;\s*g_cached_secondary_ops\.clear\(\)\s*;", ""),
}

RECIPE_LOOKUP = r"auto\s+recipe_node\s*=\s*\[cgraph\]\(\s*const\s+auto\s*&\s*entry\s*\)\s*->\s*ggml_tensor\s*\*\s*\{"
RECIPE_BOUNDS = r"entry\.graph_node_idx\s*<\s*0\s*\|\|\s*entry\.graph_node_idx\s*>=\s*cgraph->n_nodes"
RECIPE_COMPUTED = r"return\s+node\s*&&\s*ggml_sycl_graph_node_is_computed\(node\)\s*\?\s*node\s*:\s*nullptr\s*;"


def recipe_take(var):
    """A recipe loop that takes `var` through the lookup and fails the replay when it names no computed node."""
    return (r"ggml_tensor\s*\*\s*" + var + r"\s*=\s*recipe_node\(entry\)\s*;\s*"
            r"if\s*\(\s*!" + var + r"\s*\)\s*\{\s*recipe_ok\s*=\s*false\s*;\s*break\s*;\s*\}")


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

    for name, (sig, header, least) in NODE_WALKS.items():
        body = function_body(backend, sig)
        spans = loops_in(body, header)
        results[f"{name} leaves an uncomputed node before it reads it"] = \
            len(spans) >= least and all(leaves_uncomputed_first(body[a:b]) for a, b in spans)

    for name, (sig, pattern, _, least) in LOOKAHEADS.items():
        body = function_body(backend, sig)
        results[f"{name} asks the rule of the node it looks ahead to"] = \
            body is not None and len(re.findall(pattern, body)) >= least

    for name, (sig, pattern, _) in PERSISTENT_KEY.items():
        body = function_body(backend, sig)
        results[name] = body is not None and re.search(pattern, body) is not None

    plan = function_body(backend, PERSISTENT_PLAN) or ""
    lookup = re.search(RECIPE_LOOKUP, plan)
    lookup_body = balanced(plan, lookup.end() - 1) if lookup else None
    results["the recipe's node lookup refuses an index outside the graph"] = \
        lookup_body is not None and re.search(RECIPE_BOUNDS, lookup_body) is not None
    results["the recipe's node lookup refuses an uncomputed node"] = \
        lookup_body is not None and re.search(RECIPE_COMPUTED, lookup_body) is not None
    results["the recipe replay takes each node through the lookup and fails on none"] = \
        re.search(recipe_take("node"), plan) is not None
    results["the recipe's CPY entries take each node through the lookup and fail on none"] = \
        re.search(recipe_take("cpy_node"), plan) is not None
    results["the recipe reads a graph node only inside the lookup"] = \
        len(re.findall(r"cgraph->nodes\[entry\.graph_node_idx\]", plan)) == 1

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
    def mutate_loop(src, sig_regex, header, k):
        """Drop the uncomputed-node skip from the k-th loop matching `header` in the function matching `sig_regex`."""
        m = re.search(sig_regex, src)
        start = src.find("{", m.end() - 1) if m else -1
        body = balanced(src, start) if start >= 0 else None
        spans = loops_in(body, header)
        skip = SKIP_UNCOMPUTED.search(body, spans[k][0], spans[k][1]) if k < len(spans) else None
        if not skip:
            print(f"FAIL: self-test anchor missing: {sig_regex!r} loop {k}")
            failed.append(f"self-test anchor {sig_regex} loop {k}")
            return src
        return src[:start + skip.start()] + src[start + skip.end():]

    for name, (sig, header, least) in NODE_WALKS.items():
        for k in range(least):
            mutants.append((f"{name} reads an uncomputed node (loop {k + 1})",
                            f"{name} leaves an uncomputed node before it reads it",
                            (mutate_loop(backend, sig, header, k), class_hdr)))
    mutants.append(("the persistent full walk reads the node before the skip",
                    "the persistent plan's full walk leaves an uncomputed node before it reads it",
                    (mutate_in_func(backend, PERSISTENT_PLAN, r"const\s+int\s+ops_before\s*=\s*ops_added\s*;",
                                    "const int ops_before = ops_added + (node->op == GGML_OP_NONE);"), class_hdr)))
    for name, (sig, pattern, without, least) in LOOKAHEADS.items():
        mutants.append((f"{name} fuses an uncomputed look-ahead node",
                        f"{name} asks the rule of the node it looks ahead to",
                        (mutate_in_func(backend, sig, pattern, without), class_hdr)))
    for name, (sig, pattern, without) in PERSISTENT_KEY.items():
        mutants.append((f"not: {name}", name, (mutate_in_func(backend, sig, pattern, without), class_hdr)))
    direct_node = "ggml_tensor * {0} = cgraph->nodes[entry.graph_node_idx];"
    mutants += [
        ("the recipe's lookup takes any index", "the recipe's node lookup refuses an index outside the graph",
         (mutate_in_func(backend, RECIPE_LOOKUP, RECIPE_BOUNDS, "false"), class_hdr)),
        ("the recipe's lookup takes an uncomputed node", "the recipe's node lookup refuses an uncomputed node",
         (mutate_in_func(backend, RECIPE_LOOKUP, RECIPE_COMPUTED, "return node;"), class_hdr)),
        ("the recipe replay reads its node directly",
         "the recipe replay takes each node through the lookup and fails on none",
         (mutate_in_func(backend, PERSISTENT_PLAN, r"ggml_tensor\s*\*\s*node\s*=\s*recipe_node\(entry\)\s*;",
                         direct_node.format("node")), class_hdr)),
        ("the recipe replay runs on when the lookup names no node",
         "the recipe replay takes each node through the lookup and fails on none",
         (mutate_in_func(backend, PERSISTENT_PLAN, recipe_take("node"),
                         "ggml_tensor * node = recipe_node(entry);"), class_hdr)),
        ("the recipe's CPY entries read their node directly",
         "the recipe's CPY entries take each node through the lookup and fail on none",
         (mutate_in_func(backend, PERSISTENT_PLAN, r"ggml_tensor\s*\*\s*cpy_node\s*=\s*recipe_node\(entry\)\s*;",
                         direct_node.format("cpy_node")), class_hdr)),
        ("the recipe's CPY entries skip a node the lookup refuses",
         "the recipe's CPY entries take each node through the lookup and fail on none",
         (mutate_in_func(backend, PERSISTENT_PLAN, recipe_take("cpy_node"),
                         "ggml_tensor * cpy_node = recipe_node(entry);\n if (!cpy_node) { continue; }"), class_hdr)),
    ]

    for label, expect, sources in mutants:
        failed += run(label, sources, expect)

if failed:
    print("\nFAILED: " + ", ".join(failed))
    sys.exit(1)
print("\nALL CHECKS PASSED")
