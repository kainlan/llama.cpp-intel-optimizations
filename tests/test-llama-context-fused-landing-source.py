#!/usr/bin/env python3
"""resolve_fused_ops() tells a capability gap from a placement landing (llama.cpp-pmzl).

Host-only: reads src/llama-context.cpp, src/llama-fused-landing.h, ggml-sycl.cpp and ggml-sycl.h, runs no build and
loads no model. The classifier's behaviour is test-llama-fused-landing (fake devices and tensors); this gate pins the
wiring around it.

The defect: a fused op (flash attention, gated delta net) that the scheduler put on the CPU was always reported as
"executes on CPU ... -- the executor follows data placement, not a capability gap". That is true when a host-demoted
layer's KV lives in host memory, and FALSE when the layer's device has no kernel for the op (the qwen4exp
DSV4_HC_* / LIGHTNING_INDEXER case before llama.cpp-pmzl). The log read as reassurance exactly where it should have
read as a defect.

supports_op also declines for placement (host-demoted KV, planner-on-host weights), so "supports_op is false" cannot
tell a missing kernel from a placement, and the planner-on-host operand is a SYCL-owned buffer no host-buffer test
can see. The SYCL backend therefore exports a capability-only query and resolve_fused_ops asks it.

Pinned here:
  * the SYCL backend exports ggml_backend_sycl_supports_op_capability, registered as a reg proc and declared in
    ggml-sycl.h, and it runs the SAME supports_op body with the two placement predicates (KV-host buft, planner-on-
    host) switched off for the calling thread, so the two cannot drift apart;
  * the classifier asks that query when the backend has one, and otherwise falls back to the device supporting the
    node or a persistent host-resident operand (a host buffer whose usage is not COMPUTE);
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
parser.add_argument("--landing-header", default=str(root / "src/llama-fused-landing.h"))
parser.add_argument("--backend", default=str(root / "ggml/src/ggml-sycl/ggml-sycl.cpp"))
parser.add_argument("--sycl-header", default=str(root / "ggml/include/ggml-sycl.h"))
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


def evaluate(ctx, hdr, be, sh):
    classifier = function_body(
        hdr, r"static\s+inline\s+bool\s+llama_fused_cpu_landing_is_placement\s*\([^)]*\)\s*\{") or ""
    resolve = function_body(ctx, r"void\s+llama_context::resolve_fused_ops\s*\([^)]*\)\s*\{") or ""
    cap_proc = function_body(
        ctx, r"static\s+llama_fused_capability_fn\s+llama_context_sycl_capability_proc\s*\([^)]*\)\s*\{") or ""
    export = function_body(
        be, r"\bbool\s+ggml_backend_sycl_supports_op_capability\s*\([^)]*\)\s*\{") or ""
    kv_host = function_body(
        be, r"static\s+bool\s+ggml_sycl_tensor_is_in_kv_host_buft\s*\([^)]*\)\s*\{") or ""
    planned = function_body(
        be, r"static\s+bool\s+ggml_sycl_op_is_planned_on_host\s*\([^)]*\)\s*\{") or ""
    flat_resolve = squash(resolve)
    flat_classifier = squash(classifier)

    checks = {}
    # --- the backend side ---------------------------------------------------------------------------------------
    checks["backend: the capability query is defined"] = bool(export)
    checks["backend: the query sets the capability-only flag for the scope"] = bool(
        re.search(r"g_sycl_supports_op_capability_only\s*=\s*true", export))
    checks["backend: the flag is restored afterwards"] = bool(
        re.search(r"~capability_scope\s*\(\s*\)\s*\{\s*g_sycl_supports_op_capability_only\s*=\s*saved", squash(export)))
    checks["backend: the query runs the SAME supports_op body"] = bool(
        re.search(r"return\s+ggml_backend_sycl_device_supports_op\s*\(\s*dev\s*,\s*op\s*\)", export))
    checks["backend: the flag is thread-local"] = bool(
        re.search(r"static\s+thread_local\s+bool\s+g_sycl_supports_op_capability_only\s*=\s*false", be))
    checks["backend: the KV-host placement predicate honours the flag"] = bool(
        re.search(r"if\s*\(\s*!t\s*\|\|\s*g_sycl_supports_op_capability_only\s*\)\s*\{\s*return\s+false", squash(kv_host)))
    checks["backend: the planner-on-host predicate honours the flag"] = bool(
        re.search(r"g_sycl_supports_op_capability_only\s*\)\s*\{\s*return\s+false", squash(planned)))
    checks["backend: the query is registered as a reg proc"] = bool(
        re.search(r"strcmp\s*\(\s*name\s*,\s*\"ggml_backend_sycl_supports_op_capability\"\s*\)\s*==\s*0\s*\)\s*\{\s*return\s*\(void\s*\*\)\s*ggml_backend_sycl_supports_op_capability",
                  squash(be)))
    checks["backend: the query is declared in ggml-sycl.h"] = bool(
        re.search(r"GGML_BACKEND_API\s+bool\s+ggml_backend_sycl_supports_op_capability\s*\(", sh))

    # --- the classifier ----------------------------------------------------------------------------------------
    checks["classifier exists"] = bool(classifier)
    checks["classifier asks the capability query when the backend has one"] = bool(
        re.search(r"if\s*\(\s*capability\s*\)\s*\{\s*return\s+capability\s*\(\s*dev_layer\s*,\s*node\s*\)\s*;", flat_classifier))
    checks["classifier fallback asks the layer device supports_op"] = bool(
        re.search(r"ggml_backend_dev_supports_op\s*\(\s*dev_layer\s*,\s*node\s*\)", classifier))
    checks["classifier fallback treats a host operand as placement"] = bool(
        re.search(r"ggml_backend_buffer_is_host\s*\(\s*data->buffer\s*\)", classifier))
    checks["classifier looks through views"] = "src->view_src" in classifier
    checks["classifier ignores compute scratch buffers"] = bool(
        re.search(r"ggml_backend_buffer_get_usage\s*\(\s*data->buffer\s*\)\s*!=\s*GGML_BACKEND_BUFFER_USAGE_COMPUTE",
                  flat_classifier))
    checks["classifier answers capability gap with false"] = bool(re.search(r"return\s+false\s*;\s*\}\s*$", classifier))

    # --- resolve_fused_ops --------------------------------------------------------------------------------------
    checks["context includes the classifier header"] = bool(re.search(r'#include\s+"llama-fused-landing\.h"', ctx))
    checks["context looks the capability query up by its exported name"] = (
        bool(re.search(r"llama_context_dev_is_sycl\s*\(\s*dev\s*\)\s*\?\s*&ggml_backend_sycl_supports_op_capability", squash(cap_proc))) and
        '"ggml_backend_sycl_supports_op_capability"' in cap_proc)
    checks["resolve_fused_ops hands the classifier the query for the layer device"] = bool(
        re.search(r"llama_fused_cpu_landing_is_placement\s*\(\s*device_layer\s*,\s*node\.tensor\s*,\s*llama_context_sycl_capability_proc\s*\(\s*device_layer\s*\)\s*\)",
                  flat_resolve))
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


NAMES = ("ctx", "hdr", "be", "sh")


def run(label, srcs, expect_fail=None):
    results = evaluate(*srcs)
    failed = sorted(k for k, v in results.items() if not v)
    if expect_fail is None:
        for k in results:
            print(("PASS: " if results[k] else "FAIL: ") + k)
        return failed
    fired = expect_fail in failed
    print(("PASS" if fired else "FAIL") + f": mutant '{label}' fires '{expect_fail}'")
    return [] if fired else [label]


def load(path, strip=True):
    p = Path(path)
    if not p.exists():
        return ""
    return strip_comments(p.read_text()) if strip else p.read_text()


sources = {
    "ctx": load(args.context),
    "hdr": load(args.landing_header),
    "be": load(args.backend),
    "sh": load(args.sycl_header),
}
failed = run("tree", tuple(sources[k] for k in NAMES))

if args.self_test:
    def mutate(text, old, new):
        k = text.find(old)
        if k < 0:
            print(f"FAIL: self-test anchor missing: {old!r}")
            failed.append("self-test anchor " + old)
            return text
        return text[:k] + new + text[k + len(old):]

    def with_(**kw):
        return tuple(kw.get(k, sources[k]) for k in NAMES)

    mutants = [
        ("query never sets the flag", "backend: the query sets the capability-only flag for the scope",
         with_(be=mutate(sources["be"], "g_sycl_supports_op_capability_only = true;", "(void) 0;"))),
        ("flag not restored", "backend: the flag is restored afterwards",
         with_(be=mutate(sources["be"], "g_sycl_supports_op_capability_only = saved;", "(void) saved;"))),
        ("query answers from a private switch", "backend: the query runs the SAME supports_op body",
         with_(be=mutate(sources["be"], "return ggml_backend_sycl_device_supports_op(dev, op);", "return true;"))),
        ("flag is a plain global", "backend: the flag is thread-local",
         with_(be=mutate(sources["be"], "static thread_local bool g_sycl_supports_op_capability_only", "static bool g_sycl_supports_op_capability_only"))),
        ("KV-host predicate ignores the flag", "backend: the KV-host placement predicate honours the flag",
         with_(be=mutate(sources["be"], "if (!t || g_sycl_supports_op_capability_only) {", "if (!t) {"))),
        ("planner predicate ignores the flag", "backend: the planner-on-host predicate honours the flag",
         with_(be=mutate(sources["be"], "if (!op || device < 0 || g_sycl_supports_op_capability_only) {", "if (!op || device < 0) {"))),
        ("proc not registered", "backend: the query is registered as a reg proc",
         with_(be=mutate(sources["be"], 'strcmp(name, "ggml_backend_sycl_supports_op_capability") == 0', 'strcmp(name, "ggml_backend_sycl_supports_op_capability_x") == 0'))),
        ("query undeclared", "backend: the query is declared in ggml-sycl.h",
         with_(sh=mutate(sources["sh"], "ggml_backend_sycl_supports_op_capability(", "ggml_backend_sycl_supports_op_capability_x("))),
        ("capability query ignored by the classifier", "classifier asks the capability query when the backend has one",
         with_(hdr=mutate(sources["hdr"], "if (capability) {", "if (false) {"))),
        ("fallback loses supports_op", "classifier fallback asks the layer device supports_op",
         with_(hdr=mutate(sources["hdr"], "ggml_backend_dev_supports_op(dev_layer, node)", "false"))),
        ("fallback host operand ignored", "classifier fallback treats a host operand as placement",
         with_(hdr=mutate(sources["hdr"], "ggml_backend_buffer_is_host(data->buffer)", "false"))),
        ("views not followed", "classifier looks through views",
         with_(hdr=mutate(sources["hdr"], "src->view_src ? src->view_src : src", "src"))),
        ("compute scratch counted as placement", "classifier ignores compute scratch buffers",
         with_(hdr=mutate(sources["hdr"], "ggml_backend_buffer_get_usage(data->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE", "true"))),
        ("classifier never says gap", "classifier answers capability gap with false",
         with_(hdr=mutate(sources["hdr"], "    return false;\n}", "    return true;\n}"))),
        ("classifier header not included", "context includes the classifier header",
         with_(ctx=mutate(sources["ctx"], '#include "llama-fused-landing.h"', ""))),
        ("capability proc not looked up", "context looks the capability query up by its exported name",
         with_(ctx=mutate(sources["ctx"], "&ggml_backend_sycl_supports_op_capability", "nullptr"))),
        ("capability proc not passed", "resolve_fused_ops hands the classifier the query for the layer device",
         with_(ctx=mutate(sources["ctx"], "llama_context_sycl_capability_proc(device_layer)", "nullptr"))),
        ("capability wording dropped", "capability wording names the device that lacks support",
         with_(ctx=mutate(sources["ctx"], "because %s does not support it", "because of placement"))),
        ("capability line demoted", "capability wording is a warning",
         with_(ctx=mutate(sources["ctx"], "if (n_cpu_gaps > 0) {\n                LLAMA_LOG_WARN(", "if (n_cpu_gaps > 0) {\n                LLAMA_LOG_INFO("))),
        ("placement wording dropped", "placement wording is kept",
         with_(ctx=mutate(sources["ctx"], "the executor follows data placement, not a capability gap", "the executor follows placement"))),
        ("placement line unguarded", "placement wording is only for placement",
         with_(ctx=mutate(sources["ctx"], "if (n_cpu_landings > 0) {", "if (n_cpu_landings > 0 || n_cpu_gaps > 0) {"))),
        ("mismatch no longer disables", "a non-CPU mismatch still disables the op",
         with_(ctx=mutate(sources["ctx"], "if (device_mismatch) {\n            enabled = false;", "if (device_mismatch) {\n            enabled = true;"))),
        ("CPU landing disables the op", "a CPU landing still leaves the op enabled",
         with_(ctx=mutate(sources["ctx"], "} else {\n            enabled = true;", "} else {\n            enabled = n_cpu_gaps == 0;"))),
    ]
    for label, expect, srcs in mutants:
        failed += run(label, srcs, expect)

if failed:
    print("\nFAILED: " + ", ".join(failed))
    sys.exit(1)
print("\nALL CHECKS PASSED")
