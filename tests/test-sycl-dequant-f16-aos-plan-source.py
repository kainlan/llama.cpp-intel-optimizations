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
router's eligibility terms read (unified-types.hpp: coalesced_capable_type, mmq_capable_type), excluding expert
stacks and the LM head, asking the router's own dequant-support function, and honouring the oneDNN PP scratch. The
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
    results = {}
    results["the predicate lives in the shared header"] = bool(predicate)
    results["the predicate is the complement of the coalesced list"] = "!coalesced_capable_type(type)" in predicate
    results["the predicate is the complement of the MMQ list"] = "!mmq_capable_type(type)" in predicate
    results["the layout policy's coalesced list is the shared header's, not a second copy"] = (
        "coalesced_capable_type(" in common_code
        and "case GGML_TYPE_Q6_K" not in function_body(common_code, "inline bool is_coalesced_supported(ggml_type type)"))
    results["the router's MMQ eligibility is the shared header's, not a second copy"] = (
        "mmq_capable_type(" in function_body(backend_code, "inline bool ggml_sycl_supports_mmq(enum ggml_type type)")
        and "case GGML_TYPE_Q5_K"
        not in function_body(backend_code, "inline bool ggml_sycl_supports_mmq(enum ggml_type type)"))
    results["the adapter asks the shared predicate"] = "dense_pp_route_is_f16_dequant_arm(item.type)" in adapter
    results["the adapter asks the router's dequant-support function"] = (
        "onednn_woq::supports_dequant_fp16(item.type)" in adapter)
    results["the adapter leaves the LM head out"] = (
        "!lm_head" in adapter and "tensor_usage::OUTPUT_WEIGHT" in adapter
        and "ggml_sycl_is_canonical_tied_embedding_name(" in adapter)
    results["the mark excludes expert stacks"] = (
        "(unified_dequant_type || aos_dequant_type) && expert_tensor_role_from_tensor_name(item.name.c_str()) =="
        " expert_tensor_role::UNKNOWN" in adapter)
    results["the mark honours the oneDNN PP scratch (type enablement AND admission)"] = (
        "pp_scratch_type_enabled = onednn_pp_unified_scratch_enabled(item.type) &&"
        " ggml_sycl_onednn_pp_type_admitted(item.type)" in adapter)
    results["the stale 'any other type is not planned' claim is gone"] = (
        "is not planned: the graph-entry walk" not in srcs["cache"])
    return results


def mutate(text, old, new):
    assert old in text, f"mutation target missing: {old!r}"
    return text.replace(old, new, 1)


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
        ("the predicate is the complement of the coalesced list", "types",
         "!coalesced_capable_type(type) && ", ""),
        ("the predicate is the complement of the MMQ list", "types", " && !mmq_capable_type(type)", ""),
        ("the layout policy's coalesced list is the shared header's, not a second copy", "common",
         "return ggml_sycl::coalesced_capable_type(type);", "switch (type) { case GGML_TYPE_Q6_K: return true; }"),
        ("the router's MMQ eligibility is the shared header's, not a second copy", "backend",
         "return ggml_sycl::mmq_capable_type(type);", "switch (type) { case GGML_TYPE_Q5_K: return true; }"),
        ("the adapter asks the shared predicate", "cache",
         "dense_pp_route_is_f16_dequant_arm(item.type)", "ggml_is_quantized(item.type)"),
        ("the adapter asks the router's dequant-support function", "cache",
         " && onednn_woq::supports_dequant_fp16(item.type)", ""),
        ("the adapter leaves the LM head out", "cache", "!lm_head &&", ""),
        ("the mark excludes expert stacks", "cache",
         "expert_tensor_role_from_tensor_name(item.name.c_str()) == expert_tensor_role::UNKNOWN) {\n            size_t weight_bytes = 0;\n            size_t src1_bytes   = 0;\n            if (zone_dequant_f16_weight_bytes(item.ne[0], item.ne[1] > 0 ? item.ne[1] : 1, &weight_bytes) &&\n                zone_dequant_f16_src1_bytes_per_token(item.ne[0], item.ne[2] > 0 ? item.ne[2] : 1,\n                                                      item.ne[3] > 0 ? item.ne[3] : 1, &src1_bytes)) {\n                desc.dequant_f16_if_unsupplied_weight_bytes",
         "true) {\n            size_t weight_bytes = 0;\n            size_t src1_bytes   = 0;\n            if (zone_dequant_f16_weight_bytes(item.ne[0], item.ne[1] > 0 ? item.ne[1] : 1, &weight_bytes) &&\n                zone_dequant_f16_src1_bytes_per_token(item.ne[0], item.ne[2] > 0 ? item.ne[2] : 1,\n                                                      item.ne[3] > 0 ? item.ne[3] : 1, &src1_bytes)) {\n                desc.dequant_f16_if_unsupplied_weight_bytes"),
        ("the mark honours the oneDNN PP scratch (type enablement AND admission)", "cache",
         "onednn_pp_unified_scratch_enabled(item.type) && ggml_sycl_onednn_pp_type_admitted(item.type);",
         "onednn_pp_unified_scratch_enabled(item.type);"),
        ("the stale 'any other type is not planned' claim is gone", "cache",
         "// llama.cpp-gldu: a quantized type", "// is not planned: the graph-entry walk\n        // llama.cpp-gldu: a quantized type"),
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
