#!/usr/bin/env python3
"""resolve_fused_ops() tells a capability gap from a placement landing (llama.cpp-pmzl).

Host-only: reads src/llama-context.cpp, runs no build and loads no model.

The defect: a fused op (flash attention, gated delta net) that the scheduler put on the CPU was always reported as
"executes on CPU ... -- the executor follows data placement, not a capability gap". That is true when a host-demoted
layer's KV lives in host memory, and FALSE when the layer's device has no kernel for the op (the qwen4exp
DSV4_HC_* / LIGHTNING_INDEXER case before llama.cpp-pmzl). The log read as reassurance exactly where it should have
read as a defect.

Pinned here:
  * a classifier asks the layer's device whether it supports the node, and treats a persistent host-resident operand
    (a host buffer whose usage is not COMPUTE) as placement, because supports_op also declines for placement;
  * a landing the device could not have executed is logged with capability wording naming the device;
  * a genuine placement landing keeps its placement wording;
  * the enable/disable decision is untouched: a CPU landing never disables the op, a landing on a non-CPU device
    still does.

Run with --self-test to prove every check fires against a mutant of the thing it forbids.
"""
import argparse
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--context", default=str(root / "src/llama-context.cpp"))
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


def squash(text):
    return re.sub(r"\s+", " ", text or "")


def evaluate(src):
    classifier = function_body(
        src, r"static\s+bool\s+llama_fused_cpu_landing_is_placement\s*\([^)]*\)\s*\{") or ""
    resolve = function_body(src, r"void\s+llama_context::resolve_fused_ops\s*\([^)]*\)\s*\{") or ""
    flat_resolve = squash(resolve)
    flat_classifier = squash(classifier)

    checks = {}
    checks["classifier exists"] = bool(classifier)
    checks["classifier asks the layer device supports_op"] = bool(
        re.search(r"ggml_backend_dev_supports_op\s*\(\s*dev_layer\s*,\s*node\s*\)", classifier))
    checks["classifier treats a host operand as placement"] = bool(
        re.search(r"ggml_backend_buffer_is_host\s*\(\s*data->buffer\s*\)", classifier))
    checks["classifier looks through views"] = "src->view_src" in classifier
    checks["classifier ignores compute scratch buffers"] = bool(
        re.search(r"ggml_backend_buffer_get_usage\s*\(\s*data->buffer\s*\)\s*!=\s*GGML_BACKEND_BUFFER_USAGE_COMPUTE",
                  flat_classifier))
    checks["classifier answers capability gap with false"] = bool(re.search(r"return\s+false\s*;\s*\}\s*$", classifier))
    checks["resolve_fused_ops asks the classifier for each CPU landing"] = bool(
        re.search(r"llama_fused_cpu_landing_is_placement\s*\(\s*device_layer\s*,\s*node\.tensor\s*\)", resolve))
    checks["capability wording names the device that lacks support"] = bool(
        re.search(r"executes on CPU for %u layer\(s\).{0,60}because %s does not support it", flat_resolve))
    checks["capability wording is a warning"] = bool(
        re.search(r"if\s*\(\s*n_cpu_gaps\s*>\s*0\s*\)\s*\{\s*LLAMA_LOG_WARN", flat_resolve))
    checks["placement wording is kept"] = "the executor follows data placement, not a capability gap" in flat_resolve
    checks["placement wording is only for placement"] = bool(
        re.search(r"if\s*\(\s*n_cpu_landings\s*>\s*0\s*\)\s*\{\s*LLAMA_LOG_WARN", flat_resolve))
    checks["a non-CPU mismatch still disables the op"] = bool(
        re.search(r"device_mismatch\s*=\s*true", flat_resolve) and re.search(r"if\s*\(\s*device_mismatch\s*\)\s*\{\s*enabled\s*=\s*false", flat_resolve))
    checks["a CPU landing still leaves the op enabled"] = bool(
        re.search(r"\}\s*else\s*\{\s*enabled\s*=\s*true", flat_resolve))
    return checks


def run(label, text, expect_fail=None):
    results = evaluate(text)
    failed = sorted(k for k, v in results.items() if not v)
    if expect_fail is None:
        for k in results:
            print(("PASS: " if results[k] else "FAIL: ") + k)
        return failed
    fired = expect_fail in failed
    print(("PASS" if fired else "FAIL") + f": mutant '{label}' fires '{expect_fail}'")
    return [] if fired else [label]


path = Path(args.context)
source = strip_comments(path.read_text()) if path.exists() else ""
failed = run("tree", source)

if args.self_test:
    def mutate(text, old, new):
        k = text.find(old)
        if k < 0:
            print(f"FAIL: self-test anchor missing: {old!r}")
            failed.append("self-test anchor " + old)
            return text
        return text[:k] + new + text[k + len(old):]

    mutants = [
        ("supports_op not asked", "classifier asks the layer device supports_op",
         mutate(source, "ggml_backend_dev_supports_op(dev_layer, node)", "!dev_layer")),
        ("host operand ignored", "classifier treats a host operand as placement",
         mutate(source, "ggml_backend_buffer_is_host(data->buffer)", "false")),
        ("views not followed", "classifier looks through views",
         mutate(source, "src->view_src ? src->view_src : src", "src")),
        ("compute scratch counted as placement", "classifier ignores compute scratch buffers",
         mutate(source, "ggml_backend_buffer_get_usage(data->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE", "true")),
        ("classifier never says gap", "classifier answers capability gap with false",
         mutate(source, "    return false;\n}\n\nvoid llama_context::resolve_fused_ops",
                "    return true;\n}\n\nvoid llama_context::resolve_fused_ops")),
        ("classifier not consulted", "resolve_fused_ops asks the classifier for each CPU landing",
         mutate(source, "llama_fused_cpu_landing_is_placement(device_layer, node.tensor)", "true")),
        ("capability wording dropped", "capability wording names the device that lacks support",
         mutate(source, "because %s does not support it", "because of placement")),
        ("capability line demoted", "capability wording is a warning",
         mutate(source, "if (n_cpu_gaps > 0) {\n                LLAMA_LOG_WARN(", "if (n_cpu_gaps > 0) {\n                LLAMA_LOG_INFO(")),
        ("placement wording dropped", "placement wording is kept",
         mutate(source, "the executor follows data placement, not a capability gap", "the executor follows placement")),
        ("placement line unguarded", "placement wording is only for placement",
         mutate(source, "if (n_cpu_landings > 0) {", "if (n_cpu_landings > 0 || n_cpu_gaps > 0) {")),
        ("mismatch no longer disables", "a non-CPU mismatch still disables the op",
         mutate(source, "if (device_mismatch) {\n            enabled = false;", "if (device_mismatch) {\n            enabled = true;")),
        ("CPU landing disables the op", "a CPU landing still leaves the op enabled",
         mutate(source, "} else {\n            enabled = true;", "} else {\n            enabled = n_cpu_gaps == 0;")),
    ]
    for label, expect, text in mutants:
        failed += run(label, text, expect)

if failed:
    print("\nFAILED: " + ", ".join(failed))
    sys.exit(1)
print("\nALL CHECKS PASSED")
