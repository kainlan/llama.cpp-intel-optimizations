#!/usr/bin/env python3
"""The SYCL backend owns the DeepSeek-V4 hyper-connection ops and the lightning indexer (llama.cpp-pmzl).

Host-only: reads sources, runs no build, loads no model and touches no device. It proves the wiring facts
below; the numbers are test-sycl-dsv4-hc-kernels (host, opencl:cpu) and test-backend-ops -o DSV4_HC_* /
LIGHTNING_INDEXER (device, lead-run).

The defect: GGML_OP_DSV4_HC_PRE / _COMB / _POST and GGML_OP_LIGHTNING_INDEXER had no SYCL arm, so
ggml_backend_sched put every one of them on the CPU. Qwen3.8-Flash-Next emits a gated HC pre and an
identity-comb HC post around every mixer and every FFN (96 CPU landings), which fragmented the graph into about
273 CPU islands per token. A missing kernel is a support gap to close, not a routing problem.

Pinned here:
  * the executor switch and the supports_op switch each carry all four ops;
  * supports_op asks ONE predicate per op, and the executor asserts that same predicate, so supports_op
    cannot admit a (type, shape) the executor aborts on;
  * the new op sources allocate nothing and never block the host (graph-recordable);
  * the post predicate admits a null comb and the pre predicate reads the gated flag, because those are the
    two forms the Qwen graph emits and the first port of this commit (upstream 31558dbb7657) had neither.

Run with --self-test to prove every check fires against a mutant of the thing it forbids; a check that
cannot fail is decoration.
"""
import argparse
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sycl = root / "ggml/src/ggml-sycl"

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default=str(sycl / "ggml-sycl.cpp"))
parser.add_argument("--hc", default=str(sycl / "dsv4-hc.cpp"))
parser.add_argument("--hc-header", default=str(sycl / "dsv4-hc.hpp"))
parser.add_argument("--hc-kernels", default=str(sycl / "dsv4-hc-kernels.hpp"))
parser.add_argument("--lid", default=str(sycl / "lightning-indexer.cpp"))
parser.add_argument("--lid-header", default=str(sycl / "lightning-indexer.hpp"))
parser.add_argument("--lid-kernels", default=str(sycl / "lightning-indexer-kernel.hpp"))
parser.add_argument("--self-test", action="store_true")
args = parser.parse_args()

OPS = {
    "GGML_OP_DSV4_HC_PRE": ("ggml_sycl_op_dsv4_hc_pre", "ggml_sycl_dsv4_hc_pre_supported"),
    "GGML_OP_DSV4_HC_COMB": ("ggml_sycl_op_dsv4_hc_comb", "ggml_sycl_dsv4_hc_comb_supported"),
    "GGML_OP_DSV4_HC_POST": ("ggml_sycl_op_dsv4_hc_post", "ggml_sycl_dsv4_hc_post_supported"),
    "GGML_OP_LIGHTNING_INDEXER": ("ggml_sycl_op_lightning_indexer", "ggml_sycl_lightning_indexer_supported"),
}


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


def case_arm(body, op):
    """The text of `case <op>:` up to the next top-level `case`/`default` label of the same switch."""
    m = re.search(r"case\s+" + op + r"\s*:", body)
    if not m:
        return None
    n = re.search(r"\n\s*(case\s+GGML_OP_\w+\s*:|default\s*:)", body[m.end():])
    return body[m.end():m.end() + n.start()] if n else body[m.end():]


# Every spelling of a host-side block: a queue or event wait, a throwing wait, a direct synchronize.
HOST_WAIT = re.compile(r"(\.|->|::)\s*wait\s*\(|wait_and_throw|\bsynchronize\s*\(|\bhost_task\b")
ALLOC = re.compile(r"sycl::malloc|malloc_device|malloc_host|malloc_shared|unified_alloc|unified_allocate|\bnew\s+\w|\bmalloc\s*\(")


def evaluate(backend, hc, hc_header, hc_kernels, lid, lid_header, lid_kernels):
    results = {}

    fwd = function_body(backend, r"static bool ggml_sycl_compute_forward_impl\([^)]*\)\s*try\s*\{")
    sup = function_body(backend, r"static bool ggml_backend_sycl_device_supports_op\([^)]*\)\s*\{")
    results["anchor: ggml_sycl_compute_forward_impl exists"] = fwd is not None
    results["anchor: ggml_backend_sycl_device_supports_op exists"] = sup is not None
    if fwd is None or sup is None:
        return results

    for op, (executor, predicate) in OPS.items():
        # --- the executor switch carries the op and calls its executor -------------------------------------------
        arm = case_arm(fwd, op)
        results[f"compute_forward has {op}"] = arm is not None and re.search(executor + r"\s*\(", arm) is not None

        # --- supports_op carries the op and asks the one shared predicate ------------------------------------------
        sarm = case_arm(sup, op)
        results[f"supports_op has {op}"] = sarm is not None
        results[f"supports_op {op} asks {predicate}"] = sarm is not None and re.search(predicate + r"\s*\(\s*op\s*\)", sarm) is not None

    # --- the executors assert the SAME predicate, so an admitted op cannot reach an abort -----------------------
    for op, (executor, predicate) in OPS.items():
        src = lid if op == "GGML_OP_LIGHTNING_INDEXER" else hc
        body = function_body(src, r"\bvoid\s+" + executor + r"\s*\([^)]*\)\s*\{")
        results[f"anchor: {executor} exists"] = body is not None
        if body is not None:
            results[f"{executor} asserts {predicate}"] = \
                re.search(r"GGML_ASSERT\s*\(\s*" + predicate + r"\s*\(\s*dst(\.raw\(\))?\s*\)\s*\)", body) is not None
        header = lid_header if op == "GGML_OP_LIGHTNING_INDEXER" else hc_header
        results[f"{predicate} is declared"] = re.search(r"\bbool\s+" + predicate + r"\s*\(", header) is not None
        results[f"{predicate} is defined"] = \
            re.search(r"\bbool\s+" + predicate + r"\s*\([^)]*\)\s*\{", src) is not None

    # --- the post predicate admits a null comb; the pre predicate reads the gated flag ---------------------------
    post = function_body(hc, r"\bbool\s+ggml_sycl_dsv4_hc_post_supported\s*\([^)]*\)\s*\{")
    if post is not None:
        # the optional operand is guarded, not dereferenced unconditionally
        results["post predicate admits a null comb"] = re.search(r"if\s*\(\s*comb\s*!=\s*nullptr\s*\)", post) is not None
    else:
        results["post predicate admits a null comb"] = False
    pre = function_body(hc, r"\bbool\s+ggml_sycl_dsv4_hc_pre_supported\s*\([^)]*\)\s*\{")
    results["pre predicate reads the gated flag"] = pre is not None and "ggml_get_op_params_i32" in pre
    post_exec = function_body(hc, r"\bvoid\s+ggml_sycl_op_dsv4_hc_post\s*\([^)]*\)\s*\{")
    results["post executor forwards a possibly-null comb"] = post_exec is not None and "src[3]" in post_exec

    # --- graph-recordable: no allocation, no host wait ------------------------------------------------------------
    for label, text in (("dsv4-hc.cpp", hc), ("dsv4-hc-kernels.hpp", hc_kernels),
                        ("lightning-indexer.cpp", lid), ("lightning-indexer-kernel.hpp", lid_kernels)):
        results[f"{label} exists and is non-empty"] = len(text.strip()) > 0
        results[f"{label} has no host wait"] = HOST_WAIT.search(text) is None
        results[f"{label} allocates nothing"] = ALLOC.search(text) is None
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


names = ("backend", "hc", "hc_header", "hc_kernels", "lid", "lid_header", "lid_kernels")
sources = dict(zip(names, (read(args.backend), read(args.hc), read(args.hc_header), read(args.hc_kernels),
                           read(args.lid), read(args.lid_header), read(args.lid_kernels))))
failed = run("tree", tuple(sources[k] for k in names))

if args.self_test:
    def with_(**kw):
        return tuple(kw.get(k, sources[k]) for k in names)

    def mutate(text, old, new):
        k = text.find(old)
        if k < 0:
            print(f"FAIL: self-test anchor missing: {old!r}")
            failed.append("self-test anchor " + old)
            return text
        return text[:k] + new + text[k + len(old):]

    mutants = [
        ("executor arm dropped", "compute_forward has GGML_OP_DSV4_HC_PRE",
         with_(backend=mutate(sources["backend"], "case GGML_OP_DSV4_HC_PRE:", "case GGML_OP_NONE_PRE_XX:"))),
        ("supports_op arm dropped", "supports_op has GGML_OP_LIGHTNING_INDEXER",
         with_(backend=sources["backend"].replace("case GGML_OP_LIGHTNING_INDEXER:", "case GGML_OP_NONE_LID_XX:"))),
        ("supports_op ignores the predicate", "supports_op GGML_OP_DSV4_HC_COMB asks ggml_sycl_dsv4_hc_comb_supported",
         with_(backend=sources["backend"].replace("ggml_sycl_dsv4_hc_comb_supported(op)", "true"))),
        ("executor does not assert the predicate", "ggml_sycl_op_dsv4_hc_post asserts ggml_sycl_dsv4_hc_post_supported",
         with_(hc=mutate(sources["hc"], "GGML_ASSERT(ggml_sycl_dsv4_hc_post_supported(dst.raw()))", "(void) dst"))),
        ("null comb refused", "post predicate admits a null comb",
         with_(hc=mutate(sources["hc"], "if (comb != nullptr) {", "if (true) {"))),
        ("gated flag ignored", "pre predicate reads the gated flag",
         with_(hc=sources["hc"].replace("ggml_get_op_params_i32", "ggml_get_op_params_XX"))),
        ("host wait added", "dsv4-hc.cpp has no host wait",
         with_(hc=mutate(sources["hc"], "GGML_ASSERT(", "stream->wait(); GGML_ASSERT("))),
        ("allocation added", "dsv4-hc-kernels.hpp allocates nothing",
         with_(hc_kernels=sources["hc_kernels"] + "\nvoid * p = sycl::malloc_device(8, q);\n")),
        ("indexer kernel allocation added", "lightning-indexer-kernel.hpp allocates nothing",
         with_(lid_kernels=sources["lid_kernels"] + "\nvoid * p = sycl::malloc_host(8, q);\n")),
        ("indexer wait added", "lightning-indexer.cpp has no host wait",
         with_(lid=sources["lid"] + "\nvoid f(sycl::queue * s) { s->wait_and_throw(); }\n")),
    ]
    for label, expect, srcs in mutants:
        failed += run(label, srcs, expect)

if failed:
    print("\nFAILED: " + ", ".join(failed))
    sys.exit(1)
print("\nALL CHECKS PASSED")
