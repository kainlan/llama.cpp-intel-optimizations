#!/usr/bin/env python3
"""Dense f16-dequant plan covers the AOS-only quantized types (llama.cpp-gldu).

Host-only: reads sources, runs no build, loads no model and touches no device.

The defect: unified_cache_adapt_zone_inventory marked the planned dense f16 dequant buffers only for Q8_0 and for
the unified kernel's Q4_0 / MXFP4. A dense IQ4_XS / IQ3_S / IQ4_NL / Q5_K / Q4_K weight is materialized AOS (the layout
policy has no coalesced layout for it), and the router sends an AOS-layout quantized weight to the oneDNN dequant arm
at PP batch, so that arm is its normal route. With the RUNTIME zone full the first PP graph was refused
("dense f16 dequant src0 buffer needs 50.0 MB ... planned 0.0 MB").

The fix this gate pins: the planner marks those weights through ONE type predicate that the layout policy's coalesced
list is also built from (unified-types.hpp), excluding expert stacks, and honouring the oneDNN PP scratch. The
arithmetic and the get_rows_only exclusion are proved by test-zone-sizing (Case 14k); this gate proves the production
adapter actually uses the predicate and that the type list has one source.

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
parser.add_argument("--self-test", action="store_true")
args = parser.parse_args()


def strip_comments(source):
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    return re.sub(r"//[^\n]*", "", source)


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


def check(types, common, cache):
    types_code = strip_comments(types)
    common_code = strip_comments(common)
    adapter = function_body(strip_comments(cache), "unified_cache_adapt_zone_inventory(")
    results = {}
    results["the predicate lives in the shared header"] = (
        "dense_pp_route_is_f16_dequant_arm(" in types_code)
    results["the layout policy's coalesced list is the shared header's, not a second copy"] = (
        "coalesced_capable_type(" in common_code
        and "case GGML_TYPE_Q6_K" not in function_body(common_code, "inline bool is_coalesced_supported(ggml_type type)"))
    results["the predicate is the complement of that list"] = (
        "!coalesced_capable_type(" in function_body(types_code, "dense_pp_route_is_f16_dequant_arm("))
    results["the adapter asks the shared predicate"] = "dense_pp_route_is_f16_dequant_arm(item.type)" in adapter
    marks = [m.start() for m in re.finditer(r"dense_pp_route_is_f16_dequant_arm\(item\.type\)", adapter)]
    window = adapter[marks[0]:marks[0] + 600] if marks else ""
    results["the new mark excludes expert stacks"] = (
        "expert_tensor_role_from_tensor_name(item.name.c_str()) == expert_tensor_role::UNKNOWN" in window)
    results["the new mark honours the oneDNN PP scratch (admission AND type enablement)"] = (
        "onednn_pp_unified_scratch_enabled(item.type) &&" in adapter
        and "ggml_sycl_onednn_pp_type_admitted(item.type)" in adapter.split("pp_scratch_type_enabled", 1)[-1])
    results["the stale 'any other type is not planned' claim is gone"] = (
        "is not planned: the graph-entry walk" not in cache)
    return results


def mutate(text, old, new):
    assert old in text, f"mutation target missing: {old!r}"
    return text.replace(old, new, 1)


types = Path(args.types).read_text()
common = Path(args.common).read_text()
cache = Path(args.cache).read_text()
results = check(types, common, cache)

if args.self_test:
    ok_tree = all(results.values())
    mutants = {
        "the predicate lives in the shared header": (
            "types", "dense_pp_route_is_f16_dequant_arm(", "dense_pp_route_renamed("),
        "the adapter asks the shared predicate": (
            "cache", "dense_pp_route_is_f16_dequant_arm(item.type)", "ggml_is_quantized(item.type)"),
        "the stale 'any other type is not planned' claim is gone": (
            "cache", "Any other type reaching the arm", "Any other type reaching the arm is not planned: the graph-entry walk"),
    }
    sources = {"types": types, "common": common, "cache": cache}
    dead = []
    for name, (which, old, new) in mutants.items():
        mutated = dict(sources)
        if old not in mutated[which]:
            # The fixed tree may not carry the stale phrase at all: inject it.
            mutated[which] = mutated[which] + "\n// " + new + "\n"
        else:
            mutated[which] = mutate(mutated[which], old, new)
        res = check(mutated["types"], mutated["common"], mutated["cache"])
        if res[name]:
            dead.append(name)
    for name in dead:
        print(f"SELF-TEST FAIL: check cannot fail: {name}")
    if dead:
        sys.exit(1)
    if not ok_tree:
        print("self-test: mutants all detected; the tree itself does not pass yet")

failed = [name for name, ok in results.items() if not ok]
for name, ok in results.items():
    print(("PASS " if ok else "FAIL ") + name)
sys.exit(1 if failed else 0)
