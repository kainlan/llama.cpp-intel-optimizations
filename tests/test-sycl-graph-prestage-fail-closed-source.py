#!/usr/bin/env python3
"""Graph input pre-stage fails closed, and handles view inputs (llama.cpp-qhfp, review fold B1/B2).

Host-only: reads sources, runs no build, loads no model and touches no device.

B1. Once CONCAT stopped keeping qwen35 out of recording, `llama-bench` on Qwen3.6-27B aborted:

    [GRAPH-PRESTAGE] Failed to stage node_src tensor rs_s_copy (view) (data=..., 4 bytes)
    [GET_ROWS] graph recording needs pre-staged input indices for tensor=rs_s_copy (view) bytes=4
    [GET_ROWS] Falling back to CPU get_rows (device index staging failed)
    [SYCL] CPU direct get_rows failed (get_rows index staging): wait method cannot be used for an event
    associated with a command graph.

Two defects, one chain:
  (a) rs_s_copy is a recurrent-state copy index: `ggml_view_1d(s_copy, ...)` of a host-buffer INPUT.
      ggml_view_tensor() does not copy flags, so the view has no GGML_TENSOR_FLAG_INPUT; the pre-stage
      pass only stages INPUT tensors to a stable device buffer, and a non-weight non-INPUT host tensor
      then fails the unified-cache and staging-cache paths. The same chain feeds the replay refresh,
      which also keys on the flag, so a staged view would never be refreshed. A view of an INPUT is that
      input, and a zero-byte view (s_copy_extra with one sequence) has nothing to stage.
  (b) A pre-stage failure did not stop the recording. A consumer that finds no staged copy falls back to a
      host path whose wait is illegal inside a recording. A graph whose inputs cannot be staged must decline
      recording BEFORE it starts and run on the direct path.

B2 (diagnostic). The "keeping attention nodes out" line is logged once, on the FIRST decode call, when no
decode FA has dispatched yet, so every observed count reads zero and the line says nothing about the kernel
that is actually blocking engagement. It must not claim a block before anything was observed, and when it
does claim one it must name the kernel.

Run with --self-test to prove every check fires against a mutant of the thing it forbids.
"""
import argparse
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sycl = root / "ggml/src/ggml-sycl"

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default=str(sycl / "ggml-sycl.cpp"))
parser.add_argument("--common", default=str(sycl / "common.hpp"))
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
    return strip_comments(Path(path).read_text())


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


def evaluate(backend, common):
    results = {}

    prestage = function_body(backend, r"static bool graph_prestage_leaf_tensors\([^)]*\)\s*\{")
    refresh = function_body(backend, r"static void graph_refresh_input_tensors\([^)]*\)\s*\{")
    is_input = function_body(backend, r"static bool graph_tensor_is_input\([^)]*\)\s*\{")
    decline = function_body(backend, r"static bool graph_prestage_or_decline\([^)]*\)\s*\{")
    declined = function_body(backend, r"static bool graph_prestage_declined\([^)]*\)\s*\{")
    compute = function_body(backend, r"static ggml_status ggml_backend_sycl_graph_compute_unchecked\([^)]*\)\s*\{")
    obs = function_body(common, r"struct fa_decode_kernel_observation\s*\{")
    results["anchor: prestage returns bool"] = prestage is not None
    results["anchor: graph_refresh_input_tensors exists"] = refresh is not None
    results["anchor: graph_tensor_is_input exists"] = is_input is not None
    results["anchor: graph_prestage_or_decline exists"] = decline is not None
    results["anchor: graph_prestage_declined exists"] = declined is not None
    results["anchor: graph_compute_unchecked exists"] = compute is not None
    results["anchor: fa_decode_kernel_observation exists"] = obs is not None
    if None in (prestage, refresh, is_input, decline, declined, compute, obs):
        return results

    # --- (a) a view of an INPUT is that input -------------------------------------------------
    results["the INPUT test walks view_src"] = "view_src" in is_input and "GGML_TENSOR_FLAG_INPUT" in is_input
    results["pre-stage asks the view-aware INPUT test"] = "graph_tensor_is_input(" in prestage
    results["pre-stage has no flag-only INPUT test left"] = "flags & GGML_TENSOR_FLAG_INPUT" not in prestage
    results["refresh discovery asks the view-aware INPUT test"] = "graph_tensor_is_input(" in refresh
    results["refresh discovery has no flag-only INPUT test left"] = "flags & GGML_TENSOR_FLAG_INPUT" not in refresh
    # A zero-byte tensor has nothing to stage; reporting it as a failure is what made fail-closed unusable.
    failed_at = prestage.find("Failed to stage")
    zero_at = re.search(r"ggml_nbytes\(tensor\)\s*==\s*0", prestage)
    results["a zero-byte tensor is a no-op, not a staging failure"] = \
        zero_at is not None and 0 <= zero_at.start() < failed_at

    # --- (b) fail closed ----------------------------------------------------------------------
    results["a staging failure is reported to the caller"] = \
        re.search(r"all_staged\s*=\s*false", prestage) is not None and "return all_staged" in prestage
    results["the decline names itself at WARN"] = "GGML_LOG_WARN" in decline and "not recording" in decline
    results["the decline returns false on failure"] = "return false" in decline
    memo = re.search(r"static thread_local struct\s*\{[^}]*\}\s*g_graph_prestage_declined\s*;", backend)
    results["a declined graph is remembered, so the pass is not repeated per token"] = \
        memo is not None and "g_graph_prestage_declined.hash = graph_hash" in decline and \
        "g_graph_prestage_declined.hash == graph_hash" in declined
    sites = [m.start() for m in re.finditer(r"model_sycl_graph\.begin_recording\(", compute)]
    results["both full-graph recording sites exist"] = len(sites) == 2
    ok_sites = True
    for at in sites:
        before = compute[:at]
        call = before.rfind("graph_prestage_or_decline(")
        if call < 0:
            ok_sites = False
            continue
        between = before[call - 20:]
        # the decline must leave the function (direct path) before recording begins, and no void pre-stage
        # may sit between the check and the recording.
        if re.match(r"\s*if\s*\(\s*!graph_prestage_or_decline\(sycl_ctx, cgraph, graph_hash\)\s*\)\s*\{\s*"
                    r"compute_impl_unlocked\(\);\s*record_completion\(false\);\s*return GGML_STATUS_SUCCESS;",
                    between[between.find("if"):]) is None:
            ok_sites = False
        if "graph_prestage_leaf_tensors(" in between:
            ok_sites = False
    results["every full-graph recording is preceded by a pre-stage that can decline it"] = len(sites) == 2 and ok_sites
    results["an already-declined graph goes direct before any recording state is built"] = \
        "graph_prestage_declined(" in compute and \
        0 <= compute.find("graph_prestage_declined(") < compute.find("model_sycl_graph.begin_recording(")

    # --- B2: the FA gate's diagnostic ------------------------------------------------------------
    results["the observation can say whether anything was observed"] = "observed_any" in obs
    results["the observation remembers the blocking kernel's name"] = \
        "last_other_kernel" in obs and re.search(r"last_other_kernel\s*=\s*kernel", obs) is not None
    log_at = compute.find("keeping attention nodes out of SYCL command")
    guard_at = compute.rfind("observed_any()", 0, log_at) if log_at >= 0 else -1
    results["no block is claimed before a decode FA was observed"] = 0 <= guard_at < log_at
    results["the block line names the kernel"] = log_at >= 0 and "last_other=" in compute[log_at - 600:log_at + 1200]
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


backend, common = read(args.backend), read(args.common)
failed = run("tree", (backend, common))

if args.self_test:
    def mutate(src, old, new, count=1):
        if old not in src:
            print(f"FAIL: self-test anchor missing: {old!r}")
            failed.append("self-test anchor " + old)
            return src
        return src.replace(old, new, count)

    def mutate_in_func(src, sig_regex, old, new):
        m = re.search(sig_regex, src)
        if not m or src.find(old, m.end()) < 0:
            print(f"FAIL: self-test anchor missing: {sig_regex!r} .. {old!r}")
            failed.append("self-test anchor " + old)
            return src
        k = src.find(old, m.end())
        return src[:k] + new + src[k + len(old):]

    pre_sig = r"static bool graph_prestage_leaf_tensors\([^)]*\)\s*\{"
    ref_sig = r"static void graph_refresh_input_tensors\([^)]*\)\s*\{"
    dec_sig = r"static bool graph_prestage_or_decline\([^)]*\)\s*\{"
    cmp_sig = r"static ggml_status ggml_backend_sycl_graph_compute_unchecked\([^)]*\)\s*\{"
    mutants = [
        ("flag-only INPUT test in pre-stage", "pre-stage has no flag-only INPUT test left",
         (mutate_in_func(backend, pre_sig, "graph_tensor_is_input(tensor)",
                         "(tensor->flags & GGML_TENSOR_FLAG_INPUT)"), common)),
        ("flag-only INPUT test in refresh", "refresh discovery has no flag-only INPUT test left",
         (mutate_in_func(backend, ref_sig, "graph_tensor_is_input(tensor)",
                         "(tensor->flags & GGML_TENSOR_FLAG_INPUT)"), common)),
        ("view walk dropped", "the INPUT test walks view_src",
         (mutate_in_func(backend, r"static bool graph_tensor_is_input\([^)]*\)\s*\{", "view_src", "view_XXXX"),
          common)),
        ("zero-byte failure restored", "a zero-byte tensor is a no-op, not a staging failure",
         (mutate_in_func(backend, pre_sig, "ggml_nbytes(tensor) == 0", "ggml_nbytes(tensor) == 12345"), common)),
        ("failure swallowed", "a staging failure is reported to the caller",
         (mutate_in_func(backend, pre_sig, "return all_staged", "return true"), common)),
        ("decline is silent", "the decline names itself at WARN",
         (mutate_in_func(backend, dec_sig, "GGML_LOG_WARN", "GGML_SYCL_DEBUG"), common)),
        ("decline not remembered", "a declined graph is remembered, so the pass is not repeated per token",
         (mutate_in_func(backend, dec_sig, "g_graph_prestage_declined.hash = graph_hash",
                         "(void) graph_hash"), common)),
        ("memo made process-wide", "a declined graph is remembered, so the pass is not repeated per token",
         (mutate(backend, "static thread_local struct {\n    const ggml_backend_sycl_context * ctx",
                 "static struct {\n    const ggml_backend_sycl_context * ctx"), common)),
        ("recording site loses its decline", "every full-graph recording is preceded by a pre-stage that can decline it",
         (mutate_in_func(backend, cmp_sig, "graph_prestage_or_decline(sycl_ctx, cgraph, graph_hash)",
                         "graph_prestage_leaf_tensors(sycl_ctx, cgraph)"), common)),
        ("decline does not leave", "every full-graph recording is preceded by a pre-stage that can decline it",
         (mutate_in_func(backend, cmp_sig, "graph_prestage_or_decline(sycl_ctx, cgraph, graph_hash)",
                         "graph_prestage_or_decline(sycl_ctx, cgraph, graph_hash) || true"), common)),
        ("block claimed before observation", "no block is claimed before a decode FA was observed",
         (mutate_in_func(backend, cmp_sig, "observed_any()", "true"), common)),
        ("block names no kernel", "the block line names the kernel",
         (mutate_in_func(backend, cmp_sig, "last_other=", "last_XXXX="), common)),
        ("observation forgets the kernel", "the observation remembers the blocking kernel's name",
         (backend, mutate_in_func(common, r"struct fa_decode_kernel_observation\s*\{", "last_other_kernel = kernel",
                                  "last_other_kernel = nullptr"))),
    ]
    for label, expect, sources in mutants:
        failed += run(label, sources, expect)

if failed:
    print("\nFAILED: " + ", ".join(failed))
    sys.exit(1)
print("\nALL CHECKS PASSED")
