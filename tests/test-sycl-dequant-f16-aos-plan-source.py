#!/usr/bin/env python3
"""Dense f16-dequant plan covers the AOS-only quantized types (llama.cpp-gldu).

Host-only: reads sources, runs no build, loads no model and touches no device.

The defect: unified_cache_adapt_zone_inventory marked the planned dense f16 dequant buffers only for Q8_0 and for
the unified kernel's Q4_0 / MXFP4. A dense IQ4_XS / IQ3_S / IQ4_NL / Q5_K / Q4_K weight is materialized AOS (the layout
policy has no coalesced layout for it), and the router sends an AOS-layout quantized weight to the oneDNN dequant arm
at PP batch, so that arm is its normal route. With the RUNTIME zone full the first PP graph was refused
("dense f16 dequant src0 buffer needs 50.0 MB ... planned 0.0 MB").

The router walks its priority list, so only a quantized type that no MMQ kernel, no coalesced layout and no unified
kernel serves (the IQ family) has the dequant arm as its PP route; claiming an MMQ type (Q5_K) sized the plan from the
1.2 GB f16 copy of a Q5_K LM head on a device run.

The fix this gate pins: the planner marks those weights through ONE type predicate built from the same two lists the
router's eligibility terms read (unified-types.hpp: coalesced_capable_type, mmq_capable_type,
dense_mul_mat_type_supported), composed in ONE helper (aos_dequant_f16_plan_claims) that excludes expert stacks and
the LM head, with the router's own dequant-support function, and honouring the oneDNN PP scratch. The
arithmetic and the get_rows_only exclusion are proved by test-zone-sizing (Case 14k); this gate proves the production
adapter actually uses the predicate and that each type list has one source.

Run with --self-test to prove every check fires against a mutant of the thing it requires.
"""
import argparse
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sycl = root / "ggml/src/ggml-sycl"

parser = argparse.ArgumentParser()
parser.add_argument("--types", default=str(sycl / "unified-types.hpp"))
parser.add_argument("--common", default=str(sycl / "common.hpp"))
parser.add_argument("--cache", default=str(sycl / "unified-cache.cpp"))
parser.add_argument("--backend", default=str(sycl / "ggml-sycl.cpp"))
parser.add_argument("--self-test", action="store_true")
args = parser.parse_args()


def strip_comments(source):
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    return re.sub(r"//[^\n]*", "", source)


def squash(source):
    return " ".join(source.split())


def function_body(source, signature):
    start = source.find(signature)
    if start < 0:
        return ""
    brace = source.find("{", start)
    depth = 0
    for i in range(brace, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[brace:i + 1]
    return ""


def check(srcs):
    types_code = strip_comments(srcs["types"])
    common_code = strip_comments(srcs["common"])
    backend_code = strip_comments(srcs["backend"])
    adapter = squash(function_body(strip_comments(srcs["cache"]), "unified_cache_adapt_zone_inventory("))
    predicate = squash(function_body(types_code, "dense_pp_route_is_f16_dequant_arm("))
    claims = squash(function_body(types_code, "aos_dequant_f16_plan_claims("))
    mmq_fn = function_body(backend_code, "inline bool ggml_sycl_supports_mmq(enum ggml_type type)")
    dense_fn = function_body(backend_code, "static bool ggml_sycl_mul_mat_type_supported(ggml_type type)")
    coalesced_fn = function_body(common_code, "inline bool is_coalesced_supported(ggml_type type) {")
    results = {}
    results["the predicate lives in the shared header"] = bool(predicate)
    results["the predicate is the complement of the coalesced list"] = "!coalesced_capable_type(type)" in predicate
    results["the predicate is the complement of the MMQ list"] = "!mmq_capable_type(type)" in predicate
    results["the layout policy's coalesced list is the shared header's, not a second copy"] = (
        "coalesced_capable_type(" in coalesced_fn and "case GGML_TYPE" not in coalesced_fn)
    results["the router's MMQ eligibility is the shared header's, not a second copy"] = (
        "mmq_capable_type(" in mmq_fn and "case GGML_TYPE" not in mmq_fn)
    results["the router's dense MUL_MAT type list is the shared header's, not a second copy"] = (
        "dense_mul_mat_type_supported(" in dense_fn and "case GGML_TYPE" not in dense_fn)
    results["the claim composes the type lists, head and expert exclusion in the shared header"] = all(
        term in claims for term in (
            "quantized", "!is_head", "!is_expert", "dequant_supported",
            "dense_mul_mat_type_supported(type)", "dense_pp_route_is_f16_dequant_arm(type)"))
    results["the adapter asks the shared claim"] = "aos_dequant_f16_plan_claims(" in adapter
    results["the adapter asks the router's dequant-support function"] = (
        "onednn_woq::supports_dequant_fp16(item.type)" in adapter)
    results["the adapter classifies the head by usage and the tied-embedding classifier, and passes it"] = (
        re.search(r"lm_head = infer_tensor_usage\(item\.name\.c_str\(\)\) == tensor_usage::OUTPUT_WEIGHT \|\|"
                  r" ggml_sycl_is_canonical_tied_embedding_name\(item\.name\.c_str\(\)\);", adapter) is not None
        and re.search(r"aos_dequant_f16_plan_claims\([^;]*lm_head[^;]*\)", adapter) is not None)
    results["the adapter passes the expert flag to the claim"] = (
        re.search(r"aos_dequant_f16_plan_claims\([^;]*is_expert[^;]*\)", adapter) is not None)
    results["the unified kernel's mark excludes expert stacks"] = (
        "shaped && !is_expert && unified_kernel_serves_type(item.type)" in adapter
        and "is_expert = expert_tensor_role_from_tensor_name(item.name.c_str()) != expert_tensor_role::UNKNOWN"
        in adapter)
    results["the mark honours the oneDNN PP scratch (type enablement AND admission)"] = (
        "pp_scratch_type_enabled = onednn_pp_unified_scratch_enabled(item.type) &&"
        " ggml_sycl_onednn_pp_type_admitted(item.type)" in adapter)
    results["the stale 'any other type is not planned' claim is gone"] = (
        "is not planned: the graph-entry walk" not in srcs["cache"])
    return results


def ws_pattern(text):
    """A regex matching `text` with any run of whitespace between its tokens."""
    return r"\s*".join(re.escape(tok) for tok in text.split())


def mutate(text, old, new):
    """Replace the first match of `old` (whitespace-tolerant) by `new`, which is literal."""
    out, n = re.subn(ws_pattern(old), lambda _m: new, text, count=1)
    assert n == 1, f"mutation target missing: {old!r}"
    return out


sources = {
    "types": Path(args.types).read_text(),
    "common": Path(args.common).read_text(),
    "cache": Path(args.cache).read_text(),
    "backend": Path(args.backend).read_text(),
}
results = check(sources)

if args.self_test:
    # (check it must trip, source, text to replace, replacement)
    mutants = [
        ("the predicate lives in the shared header", "types",
         "inline bool dense_pp_route_is_f16_dequant_arm(", "inline bool dense_pp_route_renamed("),
        ("the predicate is the complement of the coalesced list", "types", "!coalesced_capable_type(type) &&", ""),
        ("the predicate is the complement of the MMQ list", "types", "&& !mmq_capable_type(type)", ""),
        ("the layout policy's coalesced list is the shared header's, not a second copy", "common",
         "return ggml_sycl::coalesced_capable_type(type);", "switch (type) { case GGML_TYPE_Q6_K: return true; }"),
        ("the router's MMQ eligibility is the shared header's, not a second copy", "backend",
         "return ggml_sycl::mmq_capable_type(type);", "switch (type) { case GGML_TYPE_Q5_K: return true; }"),
        ("the router's dense MUL_MAT type list is the shared header's, not a second copy", "backend",
         "return ggml_sycl::dense_mul_mat_type_supported(type);",
         "switch (type) { case GGML_TYPE_NVFP4: return true; }"),
        # one mutant per term of the claim: dropping any of them must trip the check
        ("the claim composes the type lists, head and expert exclusion in the shared header", "types",
         "return quantized &&", "return"),
        ("the claim composes the type lists, head and expert exclusion in the shared header", "types",
         "!is_head &&", ""),
        ("the claim composes the type lists, head and expert exclusion in the shared header", "types",
         "!is_expert &&", ""),
        ("the claim composes the type lists, head and expert exclusion in the shared header", "types",
         "dequant_supported &&", ""),
        ("the claim composes the type lists, head and expert exclusion in the shared header", "types",
         "dense_mul_mat_type_supported(type) &&", ""),
        ("the claim composes the type lists, head and expert exclusion in the shared header", "types",
         "&& dense_pp_route_is_f16_dequant_arm(type);", ";"),
        ("the adapter classifies the head by usage and the tied-embedding classifier, and passes it", "cache",
         "item.type, quantized, lm_head,", "item.type, quantized, false,"),
        ("the adapter asks the shared claim", "cache", "aos_dequant_f16_plan_claims(", "aos_claim_renamed("),
        ("the adapter asks the router's dequant-support function", "cache",
         "&& onednn_woq::supports_dequant_fp16(item.type)", ""),
        ("the adapter classifies the head by usage and the tied-embedding classifier, and passes it", "cache",
         "const bool lm_head = infer_tensor_usage(item.name.c_str()) == tensor_usage::OUTPUT_WEIGHT ||",
         "const bool lm_head = false; const bool lm_head_unused = infer_tensor_usage(item.name.c_str()) == "
         "tensor_usage::OUTPUT_WEIGHT ||"),
        ("the adapter passes the expert flag to the claim", "cache", "lm_head, is_expert,", "lm_head, false,"),
        ("the unified kernel's mark excludes expert stacks", "cache",
         "shaped && !is_expert && unified_kernel_serves_type(item.type)",
         "shaped && unified_kernel_serves_type(item.type)"),
        ("the mark honours the oneDNN PP scratch (type enablement AND admission)", "cache",
         "onednn_pp_unified_scratch_enabled(item.type) && ggml_sycl_onednn_pp_type_admitted(item.type);",
         "onednn_pp_unified_scratch_enabled(item.type);"),
        ("the stale 'any other type is not planned' claim is gone", "cache",
         "// llama.cpp-gldu: a quantized type",
         "// is not planned: the graph-entry walk\n        // llama.cpp-gldu: a quantized type"),
    ]
    covered = {name for name, *_ in mutants}
    dead = [name for name in results if name not in covered]
    for name, which, old, new in mutants:
        mutated = dict(sources)
        mutated[which] = mutate(mutated[which], old, new)
        if check(mutated)[name]:
            dead.append(name)
    for name in dead:
        print(f"SELF-TEST FAIL: no mutant trips: {name}")
    if dead:
        sys.exit(1)

failed = [name for name, ok in results.items() if not ok]
for name, ok in results.items():
    print(("PASS " if ok else "FAIL ") + name)
sys.exit(1 if failed else 0)
