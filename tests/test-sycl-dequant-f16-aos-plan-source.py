#!/usr/bin/env python3
"""Dense f16-dequant plan covers the quantized types whose PP route is the oneDNN f16 dequant arm (llama.cpp-gldu,
llama.cpp-h9c8).

Host-only: reads sources, runs no build, loads no model and touches no device.

The defect: unified_cache_adapt_zone_inventory marked the planned dense f16 dequant buffers only for Q8_0 and for
the unified kernel's Q4_0 / MXFP4. A dense IQ4_XS / IQ3_S / IQ4_NL weight is materialized AOS (the layout policy has
no coalesced layout for it), and the router sends an AOS-layout quantized weight to the oneDNN dequant arm at PP
batch, so that arm is its normal route. With the RUNTIME zone full the first PP graph was refused
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

llama.cpp-h9c8 moves eight MMQ types to the same arm at PP batch (Q4_1 Q5_0 Q5_1 Q2_K Q3_K Q4_K Q5_K Q6_K: the MMQ
kernel ran them ~35x below the oneDNN f16 GEMM on the same shapes). Planner and router must move together, so one
list and one claim (unified-types.hpp: dense_pp_route_prefers_f16_dequant_arm, mmq_dequant_f16_plan_claims) are asked
by the adapter AND by the router's one helper (ggml_sycl_dense_f16_arm_serves_pp), which the priority walk, the
override switch and pick_kernel_for_layout all ask. The head comes from one classifier both sides call
(ggml_sycl_weight_is_output_head). The arm converts from the layout the weight is loaded in and refuses by name a
layout it has no converter for; the converter table answers nullptr for a (type, layout) pair it cannot decode
(llama.cpp-pzu9), and the router asks it. The router also keeps on MMQ an op that asks for F32 accumulation and a
weight in a row-split buffer: either would take the arm's fp32 branch, whose f32 weight copy no plan reserves. A
weight in a tensor-parallel buffer is refused with the row-split one, as the BF16 route refuses both. The
selector's node parameter has no default, so a caller cannot drop the node without a compile error. Case 14l proves
the list, the claim and the plan arithmetic.

The router helper asks the type list before anything else, so a type it does not name (Q4_0, Q8_0) pays for no name
scan, and it admits through the planner's own function (ggml_sycl_onednn_pp_mmq_types_admitted). Each refusal of the
loaded-layout helper is checked on its own and must precede the conversion it protects; one check that some refusal
exists passes with any of them deleted. The SOA plane's lease is released by a barrier submitted after its conversion.
With GGML_SYCL_ONEDNN_PP_TRACE=1 the arm prints one WARN line per conversion
("[ONEDNN-PP-TRACE] f16-dequant-arm served"), the witness that a weight took the arm.

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
parser.add_argument("--convert", default=str(sycl / "convert.cpp"))
parser.add_argument("--catalog", default=str(root / "docs/backend/sycl-env-vars.md"))
parser.add_argument("--selection", default=str(sycl / "kernel-selection.hpp"))
parser.add_argument("--self-test", action="store_true")
args = parser.parse_args()

# The types llama.cpp-h9c8 measured MMQ losing to the oneDNN f16 arm on at PP batch.
MMQ_ARM_TYPES = ("Q4_1", "Q5_0", "Q5_1", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K")
# Of those, the types convert.cpp has an f16 converter for in AOS only.
AOS_ONLY_TYPES = ("Q4_1", "Q5_0", "Q5_1", "Q2_K", "Q3_K", "Q5_K")
SELECT_SIG = "std::optional<ggml_sycl_mul_mat_kernel> ggml_sycl_select_preferred_kernel("
PICK_SIG = "auto pick_kernel_for_layout = [&](layout_mode layout) -> std::optional<ggml_sycl_mul_mat_kernel> {"
OP_SIG = "inline void ggml_sycl_op_mul_mat_sycl(ggml_backend_sycl_context & ctx,"
SERVES_SIG = "static bool ggml_sycl_dense_f16_arm_serves_pp("
LOADED_SIG = "static void ggml_sycl_dequant_f16_from_loaded_layout("
KNOB_SIG = "static bool ggml_sycl_onednn_pp_mmq_types_enabled() {"
ADMITTED_SIG = "bool ggml_sycl_onednn_pp_mmq_types_admitted(ggml_type type) {"
TO_FP16_SIG = "to_fp16_sycl_t ggml_get_to_fp16_sycl_for_layout(ggml_type type, layout_mode layout) {"
KNOB_LITERAL = '"GGML_SYCL_ONEDNN_PP_MMQ_TYPES"'
# The four router sites that ask the helper, named by the function each lives in.
WALK_MMQ = "ggml_sycl_select_preferred_kernel's priority walk: MMQ_AOS declines an op the f16 arm serves from AOS"
WALK_SOA = ("ggml_sycl_select_preferred_kernel's priority walk: ONEDNN_SOA admits an op the f16 arm serves from SOA, "
            "beside the Q8_0 term")
OVERRIDE_SOA = "ggml_sycl_select_preferred_kernel's override switch: ONEDNN_SOA accepts the same op"
PICK_SOA = "UnifiedMatmulOrchestrator::select's pick_kernel_for_layout: the SOA case answers ONEDNN_SOA for the same op"
SPLIT = "the router helper refuses a weight in a row-split or a tensor-parallel buffer"
NODE_REQUIRED = "the selector's node parameter has no default, so a call that forgets it does not compile"
TYPE_FIRST = ("the router helper asks the type list first, so a type the list does not name pays for no name "
              "scan")
ADMISSION = ("the router helper admits through the planner's one admission function and takes the batch from the one "
             "PP admission, once each, and compares no batch itself")
# The loaded-layout helper's refusals, each with the conversion it must precede: a refusal deleted or moved past its
# conversion lets the arm decode bytes it cannot read (llama.cpp-pzu9).
AOS_CONVERT = "to_fp16(src0_dd_i, dst_f16, ne, stream);"
SOA_CONVERT = "to_fp16(plane, dst_f16, ne, stream);"
REFUSAL_AOS_NULL = 'if (to_fp16 == nullptr) { refuse("f16 dequant arm: no AOS f16 converter for this type"); }'
REFUSAL_LAYOUT = ('if (layout != GGML_LAYOUT_SOA) { '
                  'refuse("f16 dequant arm: no f16 converter for the layout this weight is loaded in"); }')
REFUSAL_ROWS = ('if (row_low != 0 || row_diff != ggml_nrows(src0)) { '
                'refuse("f16 dequant arm: an SOA weight converts only as a whole tensor"); }')
REFUSAL_SOA_NULL = 'if (to_fp16 == nullptr) { refuse("f16 dequant arm: no SOA f16 converter for this type"); }'
REFUSAL_PLANE = 'if (plane == nullptr) { refuse("f16 dequant arm: the SOA plane is not staged on this device"); }'
REFUSAL_RESIDENT = ('if (ptr_type != sycl::usm::alloc::device && ptr_type != sycl::usm::alloc::shared) { '
                    'refuse("f16 dequant arm: the SOA plane is not device-resident"); }')
REFUSALS = (
    ("an AOS weight of a type with no AOS converter", REFUSAL_AOS_NULL, AOS_CONVERT),
    ("a weight loaded in a layout other than AOS and SOA", REFUSAL_LAYOUT, SOA_CONVERT),
    ("an SOA weight handed over as a row slice", REFUSAL_ROWS, SOA_CONVERT),
    ("an SOA weight of a type with no SOA converter", REFUSAL_SOA_NULL, SOA_CONVERT),
    ("an SOA plane that is not staged on this device", REFUSAL_PLANE, SOA_CONVERT),
    ("an SOA plane that is not device-resident", REFUSAL_RESIDENT, SOA_CONVERT),
)
READ_DONE_EXPR = ("sycl::event read_done = g_ggml_sycl_graph_recording ? sycl::event{} : "
                  "stream->ext_oneapi_submit_barrier();")
RETAIN = "ggml_sycl::retain_handles_until_event({ std::move(owner) }, std::move(read_done));"
READ_DONE = ("the SOA plane's lease is released by a barrier submitted after its conversion (an empty event only while "
             "a graph records)")
WITNESS_CALL = ('if (ggml_sycl_onednn_pp_trace_enabled()) { GGML_LOG_WARN("[ONEDNN-PP-TRACE] f16-dequant-arm served '
                'tensor=%s type=%s layout=%s rows=%lld device=%d\\n",')
WITNESS = ("the f16 dequant arm prints one WARN witness line per conversion under GGML_SYCL_ONEDNN_PP_TRACE, after "
           "each conversion")
TRACE_ROW = "GGML_SYCL_ONEDNN_PP_TRACE has a catalog row that names the witness line"


def refusal_check(what):
    return f"the loaded-layout helper refuses {what}, before it converts"


def strip_comments(source):
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    return re.sub(r"//[^\n]*", "", source)


def squash(source):
    return " ".join(source.split())


def nows(source):
    """`source` with every whitespace character removed, so a check does not depend on where clang-format wraps."""
    return "".join(source.split())


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


def kernel_case(part, kernel):
    """The text of `case ggml_sycl_mul_mat_kernel::<kernel>:` in `part`, up to its first `break;`."""
    start = part.find(f"case ggml_sycl_mul_mat_kernel::{kernel}:")
    if start < 0:
        return ""
    end = part.find("break;", start)
    return part[start:end] if end > 0 else ""


def type_case(fn, type_name):
    """The text of `case GGML_TYPE_<type_name>:` in `fn`, up to the next `case GGML_TYPE_` label."""
    start = fn.find(f"case GGML_TYPE_{type_name}:")
    if start < 0:
        return ""
    end = fn.find("case GGML_TYPE_", start + 1)
    return fn[start:end] if end > 0 else fn[start:]


def conjuncts(body):
    """The `&&` terms of the one `return ...;` in `body`, whitespace removed."""
    m = re.search(r"return(.*?);", body, re.S)
    return sorted(nows(term) for term in m.group(1).split("&&")) if m else []


def check(srcs):
    types_code = strip_comments(srcs["types"])
    common_code = strip_comments(srcs["common"])
    backend_code = strip_comments(srcs["backend"])
    convert_code = strip_comments(srcs["convert"])
    adapter = squash(function_body(strip_comments(srcs["cache"]), "unified_cache_adapt_zone_inventory("))
    predicate = squash(function_body(types_code, "inline bool dense_pp_route_is_f16_dequant_arm("))
    claims = squash(function_body(types_code, "inline bool aos_dequant_f16_plan_claims("))
    mmq_list = function_body(types_code, "inline bool dense_pp_route_prefers_f16_dequant_arm(ggml_type type) {")
    mmq_claims = function_body(types_code, "inline bool mmq_dequant_f16_plan_claims(")
    head_fn = nows(function_body(common_code, "inline bool ggml_sycl_weight_is_output_head(const char * name) {"))
    mmq_fn = function_body(backend_code, "inline bool ggml_sycl_supports_mmq(enum ggml_type type)")
    dense_fn = function_body(backend_code, "static bool ggml_sycl_mul_mat_type_supported(ggml_type type)")
    coalesced_fn = function_body(common_code, "inline bool is_coalesced_supported(ggml_type type) {")
    serves = function_body(backend_code, SERVES_SIG)
    loaded = function_body(backend_code, LOADED_SIG)
    knob = function_body(backend_code, KNOB_SIG)
    admitted = nows(function_body(backend_code, ADMITTED_SIG))
    select = function_body(backend_code, SELECT_SIG)
    loop_at = select.find("for (const auto kernel : k_mul_mat_priority)")
    override_part = select[:loop_at] if loop_at > 0 else ""
    loop_part = select[loop_at:] if loop_at > 0 else ""
    pick = function_body(backend_code, PICK_SIG)
    pick_soa = pick[pick.find("case GGML_LAYOUT_SOA:"):pick.find("case GGML_LAYOUT_AOS:")]
    op_body = nows(function_body(backend_code, OP_SIG))
    to_fp16 = function_body(convert_code, TO_FP16_SIG)
    serves_aos = nows("ggml_sycl_dense_f16_arm_serves_pp(src0, src1, dst, device, GGML_LAYOUT_AOS)")
    serves_soa = nows("ggml_sycl_dense_f16_arm_serves_pp(src0, src1, dst, device, GGML_LAYOUT_SOA)")
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
    # The head is classified ONCE, in common.hpp, and both the adapter and the router helper call it, so the head the
    # plan leaves out is the head the router keeps on MMQ.
    results["the adapter classifies the head by usage and the tied-embedding classifier, and passes it"] = (
        nows("infer_tensor_usage(name) == tensor_usage::OUTPUT_WEIGHT") in head_fn
        and nows("ggml_sycl_is_canonical_tied_embedding_name(name)") in head_fn
        and nows("const bool lm_head = ggml_sycl_weight_is_output_head(item.name.c_str());") in nows(adapter)
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

    # --- llama.cpp-h9c8: the MMQ types the f16 dequant arm serves at PP batch ---
    listed = sorted(re.findall(r"case GGML_TYPE_(\w+):", mmq_list))
    results["the MMQ-type list lives in the shared header and names exactly the eight measured types"] = (
        listed == sorted(MMQ_ARM_TYPES))
    results["the MMQ claim composes exactly its eight terms in the shared header"] = conjuncts(mmq_claims) == sorted(
        nows(term) for term in (
            "quantized", "!is_head", "!is_expert", "is_matrix", "dequant_supported",
            "dense_mul_mat_type_supported(type)", "mmq_capable_type(type)",
            "dense_pp_route_prefers_f16_dequant_arm(type)"))
    results["the adapter asks the MMQ claim with the head, expert, matrix and dequant flags"] = nows(
        "mmq_dequant_f16_plan_claims(item.type, quantized, lm_head, is_expert, is_matrix, dequant_supported)"
    ) in nows(adapter)
    results["the adapter's matrix flag is a 2-D weight"] = (
        nows("const bool is_matrix = item.ne[2] <= 1 && item.ne[3] <= 1;") in nows(adapter))
    results["the MMQ claim is gated on its own PP admission and feeds the if-unsupplied block"] = (
        re.search(r"constboolmmq_dequant_type=mmq_dequant_f16_plan_claims\([^;]*\)&&"
                  r"ggml_sycl_onednn_pp_mmq_types_admitted\(item\.type\);", nows(adapter)) is not None
        and nows("if (unified_dequant_type || aos_dequant_type || mmq_dequant_type) {") in nows(adapter))
    results["the router helper asks the MMQ claim with the shared head classifier, the expert role, the 2-D shape "
            "and the dequant support"] = nows(
        "ggml_sycl::mmq_dequant_f16_plan_claims(src0->type, quantized, ggml_sycl_weight_is_output_head(src0->name), "
        "ggml_sycl::expert_tensor_role_from_tensor_name(src0->name) != ggml_sycl::expert_tensor_role::UNKNOWN, "
        "src0->ne[2] == 1 && src0->ne[3] == 1, "
        "quantized && ggml_sycl::onednn_woq::supports_dequant_fp16(src0->type))") in nows(serves)
    results["the router helper asks for a converter for the layout it is given"] = (
        nows("ggml_get_to_fp16_sycl_for_layout(src0->type, layout) == nullptr") in nows(serves))
    # The batch question has one source (llama.cpp-je3b): the helper hands it to the PP admission and compares none.
    results[ADMISSION] = (
        nows(serves).count(nows("ggml_sycl_onednn_pp_mmq_types_admitted(src0->type)")) == 1
        and "ggml_sycl_onednn_pp_mmq_types_enabled()" not in serves
        and nows(serves).count("ggml_sycl_onednn_pp_candidate(") == 1
        and nows("return ggml_sycl_onednn_pp_candidate(src0, src1, dst, device);") in nows(serves)
        and "ne[1]" not in serves and "min_batch" not in serves)
    # The arm's f16 branch runs only at the default accumulator precision (ggml_sycl_op_mul_mat_sycl): an op that
    # asks for F32 accumulation (ffn_down and attn_output of GLM4, GLM4_MOE and JAIS2; a quantized ffn_gate_inp
    # under SQRT_SOFTPLUS gating) takes the fp32 branch, whose weight copy is not planned (llama.cpp-4bu9).
    results["the router helper refuses an op that asks for F32 accumulation, and a call without the op"] = (
        nows("if (!dst || dst->op_params[0] != GGML_PREC_DEFAULT) {") in nows(serves))
    # The f16 branch also needs every row of the weight: a row-split weight hands each device a slice, which takes the
    # same fp32 branch. A tensor-parallel weight is refused with it, as the BF16 route refuses both. The split
    # predicate dereferences the buffer, so the term asks for one first.
    results[SPLIT] = nows(
        "if (src0->buffer && (ggml_backend_buffer_is_sycl_split(src0->buffer) || "
        "ggml_backend_buffer_is_sycl_tp(src0->buffer))) { return false; }") in nows(serves)
    results["the router helper exists only in a build with the f16 dequant arm"] = (
        nows(serves).startswith(nows("{ #if GGML_SYCL_DEQUANT_F16_ARM")))
    # The claim's arguments scan the weight's name twice (the head classifier, the expert role). The router asks the
    # helper for every PP MUL_MAT that reaches MMQ_AOS, Q4_0 and Q8_0 included, so the type list answers first.
    results[TYPE_FIRST] = nows(serves).startswith(nows(
        "{ #if GGML_SYCL_DEQUANT_F16_ARM if (!src0 || !src1 || "
        "!ggml_sycl::dense_pp_route_prefers_f16_dequant_arm(src0->type) ||"))
    results[WALK_MMQ] = (
        nows("if (use_mmq && !") + serves_aos + ")" in nows(kernel_case(loop_part, "MMQ_AOS")))
    results[WALK_SOA] = (
        "||" + serves_soa in nows(kernel_case(loop_part, "ONEDNN_SOA"))
        and "src0->type==GGML_TYPE_Q8_0" in nows(kernel_case(loop_part, "ONEDNN_SOA")))
    results[OVERRIDE_SOA] = (
        "&&!" + serves_soa in nows(kernel_case(override_part, "ONEDNN_SOA"))
        and "src0->type!=GGML_TYPE_Q8_0" in nows(kernel_case(override_part, "ONEDNN_SOA")))
    results[PICK_SOA] = (
        "||" + nows("ggml_sycl_dense_f16_arm_serves_pp(src0, src1, dst, ctx_.device, GGML_LAYOUT_SOA)")
        in nows(pick_soa))
    selector_calls = re.findall(r"ggml_sycl_select_preferred_kernel\(([^;{]*)\);", backend_code)
    results["the selector is given the op at every call"] = bool(selector_calls) and all(
        nows(call).endswith(",dst") for call in selector_calls)
    selection_code = strip_comments(srcs["selection"])
    decl_at = selection_code.find(SELECT_SIG)
    decl = selection_code[decl_at:selection_code.find(";", decl_at)] if decl_at >= 0 else ""
    results[NODE_REQUIRED] = (
        nows("std::optional<ggml_sycl_mul_mat_kernel> force_kernel, const ggml_tensor * dst)") in nows(decl)
        and "=" not in decl)
    results["set_decision labels ONEDNN_SOA WoQ only for Q8_0"] = nows(
        "decision.onednn_path = (src0->type == GGML_TYPE_Q8_0 && ggml_sycl_onednn_woq_q8_enabled()) ? "
        "OneDnnPath::WoQ : OneDnnPath::DequantFp16;") in nows(backend_code)
    branch = nows(
        "} else if (ggml_sycl::dense_pp_route_prefers_f16_dequant_arm(src0->type)) { "
        "ggml_sycl_dequant_f16_from_loaded_layout(ctx, src0, src0_dd_i, row_low, row_diff, dst_f16, stream); "
        "} else {")
    generic = op_body.find(nows("ggml_get_to_fp16_sycl(src0->type, dst, false)"))
    results["the arm's branch for the listed types converts through the loaded-layout helper, ahead of the generic "
            "AOS converter"] = 0 < op_body.find(branch) < generic
    results["the loaded-layout helper takes the layout from the resolver and refuses by ggml_sycl_fallback_error"] = (
        nows("const ggml_sycl::resolved_ptr loaded = ggml_sycl_resolve(src0, ctx.device);") in nows(loaded)
        and nows("const layout_mode layout = loaded ? loaded.layout : GGML_LAYOUT_AOS;") in nows(loaded)
        and nows("throw ggml_sycl_fallback_error(reason);") in nows(loaded))
    # src0_dd_i is the op's row slice. Only AOS rows are independent blocks, so it is read by the AOS converter only.
    results["the loaded-layout helper reads an AOS weight's row slice with the AOS converter only"] = (
        nows("if (layout == GGML_LAYOUT_AOS) { const to_fp16_sycl_t to_fp16 = "
             "ggml_get_to_fp16_sycl_for_layout(src0->type, GGML_LAYOUT_AOS);") in nows(loaded)
        and nows(loaded).count("src0_dd_i") == 1)
    # The SOA converters take the flat full-tensor k (their scale plane follows all the quants), so they need the
    # whole tensor, and the plane is read through a leased handle held until the conversion's reads complete.
    results["the loaded-layout helper converts an SOA weight as a whole tensor from a leased plane held until its "
            "reads complete"] = all(nows(term) in nows(loaded) for term in (
                "if (row_low != 0 || row_diff != ggml_nrows(src0)) {",
                "ggml_get_to_fp16_sycl_for_layout(src0->type, GGML_LAYOUT_SOA)",
                "ggml_sycl::mem_handle owner = ggml_sycl_acquire_weight_layout(src0, ctx.device, GGML_LAYOUT_SOA);",
                "ggml_sycl::retain_handles_until_event({ std::move(owner) }, std::move(read_done));"))
    for what, guard, conversion in REFUSALS:
        guard_at = nows(loaded).find(nows(guard))
        results[refusal_check(what)] = 0 <= guard_at < nows(loaded).find(nows(conversion))
    # A barrier submitted after the conversion covers its reads; an empty event outside recording would release the
    # lease while the kernel may still read the plane.
    convert_at = nows(loaded).find(nows(SOA_CONVERT))
    read_done_at = nows(loaded).find(nows(READ_DONE_EXPR))
    results[READ_DONE] = 0 <= convert_at < read_done_at < nows(loaded).find(nows(RETAIN))
    # The witness: WARN, because INFO is dropped at default verbosity, and after each of the two conversions.
    results[WITNESS] = (
        nows(WITNESS_CALL) in nows(loaded)
        and nows(loaded).count(nows("witness();")) == 2
        and nows(AOS_CONVERT + " witness(); return;") in nows(loaded)
        and nows(loaded).endswith(nows(RETAIN + " witness(); }")))
    trace_row = re.search(r"^\| `GGML_SYCL_ONEDNN_PP_TRACE=1` \|[^\n]*", srcs["catalog"], re.M)
    results[TRACE_ROW] = trace_row is not None and "[ONEDNN-PP-TRACE] f16-dequant-arm served" in trace_row.group(0)
    results["convert.cpp has no reordered-layout converter for the AOS-only listed types"] = all(
        nows(type_case(to_fp16, t)).startswith(nows(f"case GGML_TYPE_{t}: if (layout != GGML_LAYOUT_AOS) {{ "
                                                    f"return nullptr; }}"))
        for t in AOS_ONLY_TYPES)
    results["convert.cpp has no COALESCED converter for Q4_K and Q6_K"] = all(
        nows(type_case(to_fp16, t)).startswith(nows(f"case GGML_TYPE_{t}: if (use_coalesced) {{ return nullptr; }}"))
        for t in ("Q4_K", "Q6_K"))
    results["GGML_SYCL_ONEDNN_PP_MMQ_TYPES is read once, in one function, inside a static initializer"] = (
        backend_code.count(KNOB_LITERAL) == 1 and KNOB_LITERAL in knob
        and nows('static const bool mmq_types = []() { const bool on = get_sycl_env("GGML_SYCL_ONEDNN_PP_MMQ_TYPES", 1)'
                 ' != 0;') in nows(knob)
        and KNOB_LITERAL not in srcs["cache"] and KNOB_LITERAL not in srcs["common"])
    results["the MMQ types' plan-time admission is the knob AND the oneDNN PP type admission, declared for the "
            "planner"] = (
        nows("return ggml_sycl_onednn_pp_mmq_types_enabled() && ggml_sycl_onednn_pp_type_admitted(type);") in admitted
        and nows("bool ggml_sycl_onednn_pp_mmq_types_admitted(ggml_type type);") in nows(common_code))
    results["GGML_SYCL_ONEDNN_PP_MMQ_TYPES has a catalog row"] = (
        re.search(r"^\| `GGML_SYCL_ONEDNN_PP_MMQ_TYPES=0` \| ON \|", srcs["catalog"], re.M) is not None)
    return results


def ws_pattern(text):
    """A regex matching `text` with any run of whitespace between its tokens."""
    return r"\s*".join(re.escape(tok) for tok in text.split())


def mutate(text, old, new):
    """Replace the one match of `old` (whitespace-tolerant) by `new`, which is literal. The target must occur exactly
    once in its source: a second copy elsewhere would take the mutation and leave the checked site intact."""
    pattern = ws_pattern(old)
    n = len(re.findall(pattern, text))
    assert n == 1, f"mutation target must match once, matched {n}: {old!r}"
    return re.sub(pattern, lambda _m: new, text, count=1)


sources = {
    "types": Path(args.types).read_text(),
    "common": Path(args.common).read_text(),
    "cache": Path(args.cache).read_text(),
    "backend": Path(args.backend).read_text(),
    "convert": Path(args.convert).read_text(),
    "catalog": Path(args.catalog).read_text(),
    "selection": Path(args.selection).read_text(),
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
         "return quantized && !is_head && !is_expert && dequant_supported", "return !is_head && !is_expert && "
         "dequant_supported"),
        ("the claim composes the type lists, head and expert exclusion in the shared header", "types",
         "!is_head && !is_expert && dequant_supported", "!is_expert && dequant_supported"),
        ("the claim composes the type lists, head and expert exclusion in the shared header", "types",
         "!is_expert && dequant_supported", "dequant_supported"),
        ("the claim composes the type lists, head and expert exclusion in the shared header", "types",
         "!is_expert && dequant_supported &&", "!is_expert &&"),
        ("the claim composes the type lists, head and expert exclusion in the shared header", "types",
         "dense_mul_mat_type_supported(type) && dense_pp_route_is_f16_dequant_arm(type);",
         "dense_pp_route_is_f16_dequant_arm(type);"),
        ("the claim composes the type lists, head and expert exclusion in the shared header", "types",
         "&& dense_pp_route_is_f16_dequant_arm(type);", ";"),
        ("the adapter classifies the head by usage and the tied-embedding classifier, and passes it", "cache",
         "item.type, quantized, lm_head, is_expert, dequant_supported", "item.type, quantized, false, is_expert, "
         "dequant_supported"),
        ("the adapter asks the shared claim", "cache", "aos_dequant_f16_plan_claims(", "aos_claim_renamed("),
        ("the adapter asks the router's dequant-support function", "cache",
         "&& onednn_woq::supports_dequant_fp16(item.type)", ""),
        ("the adapter classifies the head by usage and the tied-embedding classifier, and passes it", "cache",
         "const bool lm_head = ggml_sycl_weight_is_output_head(item.name.c_str());", "const bool lm_head = false;"),
        ("the adapter classifies the head by usage and the tied-embedding classifier, and passes it", "common",
         "return infer_tensor_usage(name) == tensor_usage::OUTPUT_WEIGHT ||", "return"),
        ("the adapter passes the expert flag to the claim", "cache", "lm_head, is_expert, dequant_supported",
         "lm_head, false, dequant_supported"),
        ("the unified kernel's mark excludes expert stacks", "cache",
         "shaped && !is_expert && unified_kernel_serves_type(item.type)",
         "shaped && unified_kernel_serves_type(item.type)"),
        ("the mark honours the oneDNN PP scratch (type enablement AND admission)", "cache",
         "onednn_pp_unified_scratch_enabled(item.type) && ggml_sycl_onednn_pp_type_admitted(item.type);",
         "onednn_pp_unified_scratch_enabled(item.type);"),
        ("the stale 'any other type is not planned' claim is gone", "cache",
         "// llama.cpp-gldu: a quantized type",
         "// is not planned: the graph-entry walk\n        // llama.cpp-gldu: a quantized type"),
        # --- llama.cpp-h9c8 ---
        ("the MMQ-type list lives in the shared header and names exactly the eight measured types", "types",
         "case GGML_TYPE_Q5_1: case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K:",
         "case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K:"),
        # The list's own run of labels (the MMQ list has Q8_0 between Q5_1 and Q2_K, so it cannot match).
        ("the MMQ-type list lives in the shared header and names exactly the eight measured types", "types",
         "case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1: case GGML_TYPE_Q2_K:",
         "case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1: case GGML_TYPE_Q8_0: case GGML_TYPE_Q2_K:"),
        ("the MMQ claim composes exactly its eight terms in the shared header", "types",
         "return quantized && !is_head && !is_expert && is_matrix", "return !is_head && !is_expert && is_matrix"),
        ("the MMQ claim composes exactly its eight terms in the shared header", "types",
         "!is_head && !is_expert && is_matrix", "!is_expert && is_matrix"),
        ("the MMQ claim composes exactly its eight terms in the shared header", "types",
         "!is_expert && is_matrix", "is_matrix"),
        ("the MMQ claim composes exactly its eight terms in the shared header", "types",
         "is_matrix && dequant_supported &&", "dequant_supported &&"),
        ("the MMQ claim composes exactly its eight terms in the shared header", "types",
         "is_matrix && dequant_supported &&", "!is_matrix && dequant_supported &&"),
        ("the MMQ claim composes exactly its eight terms in the shared header", "types",
         "is_matrix && dequant_supported && dense_mul_mat_type_supported(type)",
         "is_matrix && dense_mul_mat_type_supported(type)"),
        ("the MMQ claim composes exactly its eight terms in the shared header", "types",
         "dense_mul_mat_type_supported(type) && mmq_capable_type(type)", "mmq_capable_type(type)"),
        ("the MMQ claim composes exactly its eight terms in the shared header", "types",
         "mmq_capable_type(type) && dense_pp_route_prefers_f16_dequant_arm(type);",
         "dense_pp_route_prefers_f16_dequant_arm(type);"),
        ("the MMQ claim composes exactly its eight terms in the shared header", "types",
         "&& dense_pp_route_prefers_f16_dequant_arm(type);", ";"),
        ("the adapter asks the MMQ claim with the head, expert, matrix and dequant flags", "cache",
         "item.type, quantized, lm_head, is_expert, is_matrix,", "item.type, quantized, false, is_expert, is_matrix,"),
        ("the adapter asks the MMQ claim with the head, expert, matrix and dequant flags", "cache",
         "lm_head, is_expert, is_matrix, dequant_supported", "lm_head, is_expert, true, dequant_supported"),
        ("the adapter's matrix flag is a 2-D weight", "cache",
         "const bool is_matrix = item.ne[2] <= 1 && item.ne[3] <= 1;", "const bool is_matrix = true;"),
        ("the MMQ claim is gated on its own PP admission and feeds the if-unsupplied block", "cache",
         "&& ggml_sycl_onednn_pp_mmq_types_admitted(item.type);", ";"),
        ("the MMQ claim is gated on its own PP admission and feeds the if-unsupplied block", "cache",
         "if (unified_dequant_type || aos_dequant_type || mmq_dequant_type) {",
         "if (unified_dequant_type || aos_dequant_type) {"),
        ("the router helper asks the MMQ claim with the shared head classifier, the expert role, the 2-D shape "
         "and the dequant support", "backend",
         "ggml_sycl_weight_is_output_head(src0->name),", "false,"),
        ("the router helper asks the MMQ claim with the shared head classifier, the expert role, the 2-D shape "
         "and the dequant support", "backend",
         "src0->ne[2] == 1 && src0->ne[3] == 1,", "true,"),
        ("the router helper asks for a converter for the layout it is given", "backend",
         "if (ggml_get_to_fp16_sycl_for_layout(src0->type, layout) == nullptr) { return false; }", ""),
        (ADMISSION, "backend",
         "return ggml_sycl_onednn_pp_candidate(src0, src1, dst, device); #else",
         "return src1->ne[1] >= 16; #else"),
        # The bare knob, as before the admission was one function: the type admission then comes only from the
        # candidate.
        (ADMISSION, "backend",
         "!ggml_sycl_onednn_pp_mmq_types_admitted(src0->type)", "!ggml_sycl_onednn_pp_mmq_types_enabled()"),
        (ADMISSION, "backend", "|| !ggml_sycl_onednn_pp_mmq_types_admitted(src0->type)", ""),
        (TYPE_FIRST, "backend", "!ggml_sycl::dense_pp_route_prefers_f16_dequant_arm(src0->type) ||", ""),
        # The same term asked after the claim's name scans.
        (TYPE_FIRST, "backend", [
            ("!ggml_sycl::dense_pp_route_prefers_f16_dequant_arm(src0->type) ||", ""),
            ("if (ggml_get_to_fp16_sycl_for_layout(src0->type, layout) == nullptr) {",
             "if (!ggml_sycl::dense_pp_route_prefers_f16_dequant_arm(src0->type) || "
             "ggml_get_to_fp16_sycl_for_layout(src0->type, layout) == nullptr) {")], None),
        ("the router helper refuses an op that asks for F32 accumulation, and a call without the op", "backend",
         "if (!dst || dst->op_params[0] != GGML_PREC_DEFAULT) {", "if (!dst) {"),
        (SPLIT, "backend",
         "(ggml_backend_buffer_is_sycl_split(src0->buffer) || ggml_backend_buffer_is_sycl_tp(src0->buffer))",
         "ggml_backend_buffer_is_sycl_tp(src0->buffer)"),
        (SPLIT, "backend",
         "(ggml_backend_buffer_is_sycl_split(src0->buffer) || ggml_backend_buffer_is_sycl_tp(src0->buffer))",
         "ggml_backend_buffer_is_sycl_split(src0->buffer)"),
        ("the router helper exists only in a build with the f16 dequant arm", "backend",
         "#if GGML_SYCL_DEQUANT_F16_ARM if (!src0 || !src1 ||", "if (!src0 || !src1 ||"),
        (WALK_MMQ, "backend",
         "if (use_mmq && !ggml_sycl_dense_f16_arm_serves_pp(src0, src1, dst, device, GGML_LAYOUT_AOS)) {",
         "if (use_mmq) {"),
        (WALK_SOA, "backend",
         "|| ggml_sycl_dense_f16_arm_serves_pp(src0, src1, dst, device, GGML_LAYOUT_SOA)) { return kernel;",
         ") { return kernel;"),
        (OVERRIDE_SOA, "backend",
         "&& !ggml_sycl_dense_f16_arm_serves_pp(src0, src1, dst, device, GGML_LAYOUT_SOA)) { override_ok = false;",
         ") { override_ok = false;"),
        (PICK_SOA, "backend",
         "|| ggml_sycl_dense_f16_arm_serves_pp(src0, src1, dst, ctx_.device, GGML_LAYOUT_SOA)", ""),
        ("the selector is given the op at every call", "backend",
         "ggml_sycl_select_preferred_kernel(src0, src1, ctx_.device, kernel_override, dst);",
         "ggml_sycl_select_preferred_kernel(src0, src1, ctx_.device, kernel_override);"),
        (NODE_REQUIRED, "selection",
         "std::optional<ggml_sycl_mul_mat_kernel> force_kernel, const ggml_tensor * dst);",
         "std::optional<ggml_sycl_mul_mat_kernel> force_kernel = std::nullopt, const ggml_tensor * dst = nullptr);"),
        ("set_decision labels ONEDNN_SOA WoQ only for Q8_0", "backend",
         "decision.onednn_path = (src0->type == GGML_TYPE_Q8_0 && ggml_sycl_onednn_woq_q8_enabled()) ?",
         "decision.onednn_path = ggml_sycl_onednn_woq_q8_enabled() ?"),
        ("the arm's branch for the listed types converts through the loaded-layout helper, ahead of the generic "
         "AOS converter", "backend",
         "} else if (ggml_sycl::dense_pp_route_prefers_f16_dequant_arm(src0->type)) { "
         "ggml_sycl_dequant_f16_from_loaded_layout(ctx, src0, src0_dd_i, row_low, row_diff, dst_f16, stream); }",
         "}"),
        ("the loaded-layout helper takes the layout from the resolver and refuses by ggml_sycl_fallback_error",
         "backend", "const layout_mode layout = loaded ? loaded.layout : GGML_LAYOUT_AOS;",
         "const layout_mode layout = GGML_LAYOUT_AOS;"),
        ("the loaded-layout helper takes the layout from the resolver and refuses by ggml_sycl_fallback_error",
         "backend", "ggml_sycl_layout_mode_name(layout), reason); throw ggml_sycl_fallback_error(reason);",
         "ggml_sycl_layout_mode_name(layout), reason); GGML_UNUSED(reason);"),
        ("the loaded-layout helper reads an AOS weight's row slice with the AOS converter only", "backend",
         "if (layout == GGML_LAYOUT_AOS) { const to_fp16_sycl_t to_fp16 =",
         "if (layout != GGML_LAYOUT_COALESCED) { const to_fp16_sycl_t to_fp16 ="),
        ("the loaded-layout helper converts an SOA weight as a whole tensor from a leased plane held until its "
         "reads complete", "backend", "if (row_low != 0 || row_diff != ggml_nrows(src0)) {", "if (false) {"),
        ("the loaded-layout helper converts an SOA weight as a whole tensor from a leased plane held until its "
         "reads complete", "backend",
         "ggml_sycl::retain_handles_until_event({ std::move(owner) }, std::move(read_done));",
         "GGML_UNUSED(read_done);"),
        # one mutant per refusal deletes that guard, and two move guards past their conversion
        (refusal_check(REFUSALS[0][0]), "backend", REFUSAL_AOS_NULL, ""),
        (refusal_check(REFUSALS[1][0]), "backend", REFUSAL_LAYOUT, ""),
        (refusal_check(REFUSALS[2][0]), "backend", REFUSAL_ROWS, ""),
        (refusal_check(REFUSALS[3][0]), "backend", REFUSAL_SOA_NULL, ""),
        (refusal_check(REFUSALS[4][0]), "backend", REFUSAL_PLANE, ""),
        (refusal_check(REFUSALS[5][0]), "backend", REFUSAL_RESIDENT, ""),
        (refusal_check(REFUSALS[0][0]), "backend", REFUSAL_AOS_NULL + " " + AOS_CONVERT,
         AOS_CONVERT + " " + REFUSAL_AOS_NULL),
        (refusal_check(REFUSALS[5][0]), "backend", REFUSAL_RESIDENT + " " + SOA_CONVERT,
         SOA_CONVERT + " " + REFUSAL_RESIDENT),
        # The expression has a twin at another site, so the conversion before it anchors the mutant.
        (READ_DONE, "backend", SOA_CONVERT + " " + READ_DONE_EXPR,
         SOA_CONVERT + " sycl::event read_done = sycl::event{};"),
        (READ_DONE, "backend", SOA_CONVERT + " " + READ_DONE_EXPR, READ_DONE_EXPR + " " + SOA_CONVERT),
        (WITNESS, "backend", 'GGML_LOG_WARN("[ONEDNN-PP-TRACE] f16-dequant-arm served',
         'GGML_LOG_INFO("[ONEDNN-PP-TRACE] f16-dequant-arm served'),
        (WITNESS, "backend", AOS_CONVERT + " witness(); return;", AOS_CONVERT + " return;"),
        (WITNESS, "backend", AOS_CONVERT + " witness(); return;", "witness(); " + AOS_CONVERT + " return;"),
        (WITNESS, "backend", RETAIN + " witness(); }", RETAIN + " }"),
        (WITNESS, "backend",
         'if (ggml_sycl_onednn_pp_trace_enabled()) { GGML_LOG_WARN("[ONEDNN-PP-TRACE] f16-dequant-arm',
         'if (true) { GGML_LOG_WARN("[ONEDNN-PP-TRACE] f16-dequant-arm'),
        (TRACE_ROW, "catalog", "| `GGML_SYCL_ONEDNN_PP_TRACE=1` |", "| `GGML_SYCL_ONEDNN_PP_TRACE_RENAMED=1` |"),
        (TRACE_ROW, "catalog", "`[ONEDNN-PP-TRACE] f16-dequant-arm served", "`[ONEDNN-PP-TRACE] served"),
        ("convert.cpp has no reordered-layout converter for the AOS-only listed types", "convert",
         "case GGML_TYPE_Q5_K: if (layout != GGML_LAYOUT_AOS) { return nullptr; }", "case GGML_TYPE_Q5_K:"),
        ("convert.cpp has no reordered-layout converter for the AOS-only listed types", "convert",
         "case GGML_TYPE_Q4_1: if (layout != GGML_LAYOUT_AOS) { return nullptr; }", "case GGML_TYPE_Q4_1:"),
        ("convert.cpp has no COALESCED converter for Q4_K and Q6_K", "convert",
         "case GGML_TYPE_Q6_K: if (use_coalesced) { return nullptr; } else if (use_reorder) {",
         "case GGML_TYPE_Q6_K: if (use_reorder) {"),
        ("GGML_SYCL_ONEDNN_PP_MMQ_TYPES is read once, in one function, inside a static initializer", "backend",
         "static const bool mmq_types = []() {", "const bool mmq_types = []() {"),
        ("GGML_SYCL_ONEDNN_PP_MMQ_TYPES is read once, in one function, inside a static initializer", "backend",
         "bool ggml_sycl_onednn_pp_mmq_types_admitted(ggml_type type) {",
         'bool ggml_sycl_onednn_pp_mmq_types_admitted(ggml_type type) { (void) std::getenv('
         '"GGML_SYCL_ONEDNN_PP_MMQ_TYPES");'),
        ("the MMQ types' plan-time admission is the knob AND the oneDNN PP type admission, declared for the "
         "planner", "backend",
         "return ggml_sycl_onednn_pp_mmq_types_enabled() && ggml_sycl_onednn_pp_type_admitted(type);",
         "return ggml_sycl_onednn_pp_mmq_types_enabled();"),
        ("GGML_SYCL_ONEDNN_PP_MMQ_TYPES has a catalog row", "catalog",
         "| `GGML_SYCL_ONEDNN_PP_MMQ_TYPES=0` |", "| `GGML_SYCL_ONEDNN_PP_MMQ_TYPES_RENAMED=0` |"),
    ]
    covered = {name for name, *_ in mutants}
    dead = [name for name in results if name not in covered]
    for name, which, old, new in mutants:
        mutated = dict(sources)
        for old_text, new_text in (old if isinstance(old, list) else [(old, new)]):
            mutated[which] = mutate(mutated[which], old_text, new_text)
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
