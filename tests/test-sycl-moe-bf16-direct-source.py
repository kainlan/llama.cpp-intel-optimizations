#!/usr/bin/env python3
"""Source contract for the BF16 expert device route (Task 8g of the Inkling 2 Mini plan).

A MUL_MAT_ID whose experts are BF16 runs on the device whose plan holds every expert at AOS, from the cached full-local
expert table and the device ids, with src1 and dst read and written in place. These claims pin that shape:

  (a) one predicate, ggml_sycl_moe_bf16_direct_route_available(op, device), defined once in ggml-sycl.cpp: a view
      of a weight is declined before any placement question (the route publishes and reads a root tensor's experts
      only); a known plan answers alone (every expert of the weight planned on this device at AOS, asked through
      ggml_sycl_moe_tensor_plan_primary_layout); with no plan, a weight nothing has placed yet counts (test-backend-ops
      asks supports_op before it allocates, then allocates on the backend it asked) and a placed weight counts only in
      this device's own buffer (an unsplit SYCL buffer of this device, in VRAM), asked through the one helper that
      holds that whole test; it reads no environment and has no row-multiple rule;
  (b) supports_op asks that predicate as the MUL_MAT_ID gate's third term, and the route asks the same predicate
      again at run time and refuses by name when it says no;
  (c) the route sits in ggml_sycl_mul_mat_id after the src0->extra ensure block and before the MMID_SCHED device
      schedule, the yx28 decode-direct request and ggml_sycl_ensure_weight_on_device(); its block is keyed on the BF16
      expert type, not on the predicate, and is one call to ggml_sycl_mul_mat_id_bf16_direct() and a return, so no
      BF16 op reaches the routes after it, none of which has a BF16 kernel;
  (d) ggml_sycl_mul_mat_id_bf16_direct() takes the AOS full-local table through one table helper (which publishes the
      weight's AOS slices on a miss only from this device's own buffer, through the same buffer test the predicate
      asks, and not again at a storage generation where the publish failed or an expert still did not resolve after
      it; a table upload that failed is retried at the next node, as this route did before the helper existed),
      src1 and dst only where they resolve to
      this device's own memory (the ruling forbids a GPU read of host memory), the device-resident ids, submits
      through moe_bf16_direct_submit() inside the kernel profiler's existing wrapper under the label
      mulmat.moe.bf16_direct (its metadata naming the launch count, from the one tokens-per-launch answer the route
      asks for), and retains a snapshot of the table's dispatch bundle (its active leases and the table's
      own allocation) until the kernel's event, as the production pointer-table dispatchers do; it never consumes the
      table's active leases (the next dispatch from the same cached table needs them), reads the ids on the host,
      waits, touches the MMID workspace or Q8_1, re-routes to the CPU or aborts; a refusal is a named WARN line and
      ggml_sycl_fallback_error;
  (e) under graph recording it does what the NVFP4 decode node does: a WARN line and a sycl::exception before any
      table or ids work, and on a direct run it sets moe_graphs_disabled_once like every other MoE route (its consumer
      selects segmented MoE replay, it does not turn replay off); check_graph_compatibility keeps a BF16 MUL_MAT_ID
      out of whole-graph capture, so it is a segment boundary like NVFP4;
  (f) the op-entry flush of another node's pending host scatter precedes the MUL_MAT_ID dispatch in
      ggml_sycl_compute_forward_impl, so the route reads src1 after it;
  (g) the kernel lives in moe-bf16-direct.{hpp,cpp}: BF16 decoded with bf16_to_f32, a 16-lane sub-group per output
      row reduced with reduce_over_group, id -1 a zero row and any other bad slot a quiet NaN; a dispatch whose global
      range exceeds the shape's launch limit (an int by default) runs in token chunks, each after the one before, and
      one that cannot be chunked is refused; the float4 activation load runs only on a 16-byte aligned src1 base and
      row strides, and the 8-byte weight load only on an 8-byte aligned weight row (opencl:cpu reads unaligned loads
      correctly, so the kernel test cannot see either guard go); the file allocates, frees and waits on nothing;
  (h) the MMVQ tables stay BF16-free and moe_type_needs_direct_kernel stays Q1_0/NVFP4.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SYCL_DIR = ROOT / "ggml/src/ggml-sycl"
FILES = {
    "sycl": SYCL_DIR / "ggml-sycl.cpp",
    "hpp": SYCL_DIR / "moe-bf16-direct.hpp",
    "cpp": SYCL_DIR / "moe-bf16-direct.cpp",
    "tables": SYCL_DIR / "moe-mmvq-tables.hpp",
    "resolved": SYCL_DIR / "moe-resolved-batch.hpp",
}

PREDICATE = "ggml_sycl_moe_bf16_direct_route_available"
PREDICATE_HEAD = f"static bool {PREDICATE}(const ggml_tensor * op, int device) {{"
ROUTE_FN = "ggml_sycl_mul_mat_id_bf16_direct"
ROUTE_FN_HEAD = f"static void {ROUTE_FN}(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {{"
ROUTE_IF = "if (src0->type == GGML_TYPE_BF16) {"
ROUTE_ASK = f"if (!{PREDICATE}(dst, ctx.device)) {{"
UNPLACED_CLAIM = "if(src0->buffer==nullptr){returntrue;}"
# One helper holds the whole "this device's own buffer" test: one device's unsplit SYCL buffer, of `device`, in VRAM.
OWN_BUFFER = "ggml_sycl_moe_direct_own_vram_buffer"
OWN_BUFFER_HEAD = f"static bool {OWN_BUFFER}(ggml_backend_buffer_t buffer, int device) {{"
OWN_BUFFER_BODY = (
    "if(buffer==nullptr||!ggml_backend_buffer_is_sycl(buffer)||ggml_backend_buffer_is_sycl_split(buffer)){returnfalse;}"
    "constauto*buf_ctx=static_cast<constggml_backend_sycl_buffer_context*>(buffer->context);"
    "returnbuf_ctx!=nullptr&&buf_ctx->device==device&&buf_ctx->managed_meta.tier==ggml_sycl::alloc_tier::DEVICE_VRAM;")
NO_PLAN_TAIL = f"{UNPLACED_CLAIM}return{OWN_BUFFER}(src0->buffer,device);"
# Spellings of the buffer test that belong to the helper alone.
BUFFER_TEST_PARTS = ("ggml_backend_buffer_is_sycl_split(", "ggml_backend_buffer_is_sycl(", "buf_ctx",
                     "alloc_tier::DEVICE_VRAM")
# The one table helper this route and the NVFP4 resident decode route ask: probe, then on a miss in this device's own
# buffer publish once and probe again, unless the publish failed or an expert did not resolve at this storage
# generation. A table upload that failed is not stamped, so this route retries it as it did before the helper.
ENSURE_HELPER = "ggml_sycl_moe_direct_ensure_aos_table"
ENSURE_HELPER_HEAD = (f"static const void * const * {ENSURE_HELPER}(ggml_backend_sycl_context & ctx, "
                      "const ggml_tensor * src0) {")
HELPER_EARLY = (f"if (table != nullptr || src0->extra == nullptr || !{OWN_BUFFER}(src0->buffer, ctx.device)) "
                "{ return table; }")
HELPER_PUBLISH = "if (ggml_sycl_publish_mmid_canonical_aos_experts(src0, ctx.device, GGML_LAYOUT_AOS,"
HELPER_TABLE_CALL = "table = moe_fusion_ensure_full_local_ptr_table(ctx, src0, layer, GGML_LAYOUT_AOS);"
HELPER_RETRY = ("table = moe_fusion_ensure_full_local_ptr_table(ctx, src0, layer, GGML_LAYOUT_AOS, "
                "&every_expert_resolved);")
HELPER_STAMP_SKIP = "if (failed == weight.moe_expert_storage_generation) { return nullptr; }"
HELPER_STAMP_STORE = ("if (table == nullptr && !every_expert_resolved) { "
                      "failed = weight.moe_expert_storage_generation; }")
# src1 and dst: only an on-device pointer, resolved without materializing.
OPERAND_HELPER = "ggml_sycl_moe_direct_device_operand"
OPERAND_HELPER_HEAD = f"static void * {OPERAND_HELPER}(const ggml_tensor * tensor, int device) {{"
OPERAND_HELPER_BODY = ("constggml_sycl::resolved_ptrresolved=ggml_sycl_resolve_no_materialize(tensor,device);"
                       "returnresolved.on_device?resolved.ptr:nullptr;")
OPERAND_CALLS = (f"{OPERAND_HELPER}(src1, ctx.device)", f"{OPERAND_HELPER}(dst, ctx.device)")
PROFILE_SUBMIT = "ggml_sycl_profile_submit("
PROFILE_NAME = '"mulmat.moe.bf16_direct"'
PROFILE_KEYS = ('"ncols="', '";nrows_per_expert="', '";total_batches="', '";launches="')
TOKENS_PER_LAUNCH_CALL = "ggml_sycl::moe_bf16_direct_tokens_per_launch(shape)"
# The kernel's two alignment guards. opencl:cpu reads unaligned float4 and 8-byte loads correctly, so the kernel test
# cannot see either guard go; these pins are what can.
X_ALIGN_GUARD = ("const bool xvec = vec4 && reinterpret_cast<uintptr_t>(src1) % 16 == 0 && shape.src1_nb1 % 16 == 0 && "
                 "shape.src1_nb2 % 16 == 0;")
W_ALIGN_GUARD = "if (VEC4 && (reinterpret_cast<uintptr_t>(w_row) & 7) == 0) {"
ROUTE_BODY = f"{ROUTE_FN}(ctx,dst);return;"
GATE = ("if (!moe_mmvq_admission_supports_type(indexed_a_type) && !ggml_sycl_moe_pp_batched_admits_op(op, device) && "
        f"!{PREDICATE}(op, device)) {{")
MUL_MAT_ID_HEAD = "static void ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx, ggml_tensor * dst) try {"
EXTRA_ENSURED = "s_mul_mat_id_extras.push_back(std::move(new_extra));"
MMID_SCHED = "(g_ggml_sycl_mmid_sched & GGML_SYCL_MMID_SCHED_DEVICE)"
DECODE_DIRECT = "ggml_sycl::moe_decode_direct_request direct_request;"
ENSURE_WEIGHT = "ggml_sycl_ensure_weight_on_device(src0, ctx.device);"
TABLE_CALL = f"{ENSURE_HELPER}(ctx, src0)"
IDS_CALL = "ggml_sycl_moe_ids_device_resident_ptr(ctx, ids"
SUBMIT_CALL = "ggml_sycl::moe_bf16_direct_submit("
SNAPSHOT_CALL = "ggml_sycl_snapshot_moe_ptr_table_dispatch_bundle("
RETAIN_CALL = "ggml_sycl::retain_handles_until_event("
RECORDING_IF = "if (g_ggml_sycl_graph_recording) {"
RECORDING_THROW = "throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),"
GRAPH_FLAG = "ctx.moe_graphs_disabled_once = true;"
REFUSAL_WARN = 'GGML_LOG_WARN("[MoE] BF16 direct route refused'
REFUSAL_THROW = "throw ggml_sycl_fallback_error("
ROUTE_LOG = "[MOE-ROUTE] bf16 direct"
FORWARD_HEAD = "static bool ggml_sycl_compute_forward_impl(ggml_backend_sycl_context & ctx, struct ggml_tensor * dst) try {"
OP_ENTRY_FLUSH = "flush_pending_cpu_scatter_if_consumed(dst, ctx.device);"
MUL_MAT_ID_DISPATCH = "case GGML_OP_MUL_MAT_ID:"
COMPAT_HEAD = "static ggml_sycl_graph_compat check_graph_compatibility(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph) {"
COMPAT_CASE = "case GGML_OP_MUL_MAT_ID:"
COMPAT_QUANT_GUARD = "if (ggml_is_quantized(src0->type)) {"
SUBMIT_DECL = "bool moe_bf16_direct_submit("
NEEDS_DIRECT_HEAD = "inline bool moe_type_needs_direct_kernel(ggml_type type) {"
NEEDS_DIRECT_BODY = "returntype==GGML_TYPE_Q1_0||type==GGML_TYPE_NVFP4;"
# A host wait is spelled .wait(, ->wait( or sycl::event::wait(: the three spellings are listed apart.
ROUTE_FORBIDDEN = (".wait(", "->wait(", "::wait(", "wait_and_throw", "ids_host", "ggml_sycl_copy_ids_to_host",
                   "mmid_workspace", "q8_1", "GGML_ABORT", "ggml_sycl_ensure_weight_on_device", "malloc_", "sycl::free(",
                   "getenv", "cpu", "ggml_sycl_retain_moe_ptr_table_leases_until_event",
                   "ggml_sycl_resolve_tensor_ptr", "moe_fusion_ensure_full_local_ptr_table",
                   "ggml_sycl_publish_mmid_canonical_aos_experts")
SYCL_INCLUDE = re.compile(r'#include "(?:ggml-sycl/)?moe-bf16-direct\.hpp"')
KERNEL_FORBIDDEN = (".wait(", "->wait(", "::wait(", "wait_and_throw", "malloc_", "sycl::free(", "unified_alloc",
                    "q8_1", "GGML_ABORT", "getenv")


def strip_comments(src):
    """Drop // and /* */ comments, leaving string and char literals intact (a quoted "//" is not a comment)."""
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c in "\"'":
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == "\\" else 1
            out.append(src[i : j + 1])
            i = j + 1
        elif src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            i = n if j < 0 else j + 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def squeeze(src):
    return re.sub(r"\s+", " ", src)


def load():
    """Squeezed, comment-stripped text per file; a file that does not exist yet reads as empty."""
    out = {}
    for key, path in FILES.items():
        out[key] = squeeze(strip_comments(path.read_text(encoding="utf-8"))) if path.exists() else ""
    return out


def tight(text):
    return re.sub(r"\s+", "", text)


def braced_body(text, head, start=0):
    """The text between the '{' that ends `head` and its matching '}', or None."""
    at = text.find(head, start)
    if at < 0:
        return None
    opening = at + len(head) - 1
    depth = 0
    for pos in range(opening, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : pos]
    return None


def function_body(text, name_head):
    """The braced body of the first definition whose text starts with `name_head` (a name and its open paren)."""
    at = text.find(name_head)
    while at >= 0:
        brace, semi = text.find("{", at), text.find(";", at)
        if brace >= 0 and (semi < 0 or brace < semi):
            return braced_body(text, text[at : brace + 1], at)
        at = text.find(name_head, at + 1)
    return None


def claim_a_predicate(s):
    fails = []
    sycl = s["sycl"]
    if sycl.count(PREDICATE_HEAD) != 1 or len(re.findall(rf"\b(?:static )?bool {PREDICATE}\(", sycl)) != 1:
        return [f"{PREDICATE}(op, device) is not defined exactly once as a static bool in ggml-sycl.cpp"]
    body = braced_body(sycl, PREDICATE_HEAD) or ""
    for needed in ("GGML_TYPE_BF16", "ggml_is_contiguous(", "ggml_sycl_moe_plan_owner_for_device(device)",
                   "ggml_sycl_moe_tensor_plan_primary_layout(", "GGML_LAYOUT_AOS", f"{OWN_BUFFER}("):
        if needed not in body:
            fails.append(f"the predicate does not ask {needed}")
    # One helper says what this device's own buffer is (unsplit SYCL, this device, VRAM); the predicate's no-plan arm
    # and the route's publish guard ask it, and neither spells any part of it.
    if sycl.count(OWN_BUFFER_HEAD) != 1 or tight(braced_body(sycl, OWN_BUFFER_HEAD) or "") != OWN_BUFFER_BODY:
        fails.append(f"{OWN_BUFFER} is not defined once as `an unsplit SYCL buffer of `device`, in DEVICE_VRAM`")
    elif sycl.find(OWN_BUFFER_HEAD) > sycl.find(PREDICATE_HEAD):
        fails.append(f"{OWN_BUFFER} is defined after the predicate that asks it")
    # Counted in this route's own bodies, so another route's use of the helper cannot move the count.
    route = route_fn_body(sycl) or ""
    helper = braced_body(sycl, ENSURE_HELPER_HEAD) or ""
    for where, text, want in (("predicate", body, 1), ("table helper", helper, 1), ("route", route, 0)):
        asks = len(re.findall(rf"\b{OWN_BUFFER}\(", text))
        if asks != want:
            fails.append(f"the {where} asks {OWN_BUFFER} {asks} times; want {want}")
    for where, text in (("predicate", body), ("table helper", helper), ("route", route)):
        spelled = [part for part in BUFFER_TEST_PARTS if part in text]
        if spelled:
            fails.append(f"the {where} spells the buffer test out ({', '.join(spelled)}) instead of asking {OWN_BUFFER}")
    for forbidden in ("getenv", "% 32", "%32", "WARP_SIZE", "ggml_sycl_moe_pp_batched", "GGML_TYPE_NVFP4"):
        if forbidden in body:
            fails.append(f"the predicate carries {forbidden!r}")
    view_at = body.find("src0->view_src != nullptr")
    plan_at = body.find("ggml_sycl_moe_plan_owner_for_device(device)")
    if view_at < 0 or (plan_at >= 0 and view_at > plan_at):
        fails.append("the predicate does not decline a view of the weight before it asks the plan")
    plan_arm = braced_body(body, "if (plan) {")
    if plan_arm is None:
        fails.append("the predicate has no `if (plan) {` arm")
    else:
        statements = [stmt for stmt in tight(plan_arm).split(";") if stmt]
        last = statements[-1] if statements else ""
        if not last.startswith("returnggml_sycl_moe_tensor_plan_primary_layout(") or "GGML_LAYOUT_AOS" not in last:
            fails.append("the plan arm does not end by returning the plan's AOS primary layout on this device")
        if f"{OWN_BUFFER}(" in plan_arm or "->buffer" in plan_arm:
            fails.append("the plan arm consults the weight's buffer: a known plan answers alone")
        tail = body[body.find("if (plan) {") + len("if (plan) {") + len(plan_arm) :]
        # The no-plan arm, whole: a weight nothing has placed yet counts, and a placed one only in this device's own
        # VRAM buffer.
        if not tight(tail).endswith(NO_PLAN_TAIL):
            fails.append("the no-plan arm is not `if (src0->buffer == nullptr) { return true; } "
                         f"return {OWN_BUFFER}(src0->buffer, device);`")
    return fails


def claim_b_one_predicate(s):
    fails = []
    sycl = s["sycl"]
    if sycl.count(GATE) != 1:
        fails.append("the supports_op MUL_MAT_ID gate does not ask the BF16 predicate as its third term")
    calls = len(re.findall(rf"\b{PREDICATE}\(", sycl))
    if calls != 3:
        fails.append(f"{PREDICATE} appears {calls} times; want its definition, the gate and the route")
    route = route_fn_body(sycl) or ""
    if sycl.count(ROUTE_ASK) != 1 or ROUTE_ASK not in route:
        fails.append(f"{ROUTE_FN} does not ask the predicate exactly once at run time")
    else:
        declined = braced_body(route, ROUTE_ASK) or ""
        if not tight(declined).startswith("refuse("):
            fails.append("a run-time `no` from the predicate is not a named refusal")
        if route.find(ROUTE_ASK) > route.find(TABLE_CALL) >= 0:
            fails.append("the run-time predicate is asked after the table is built")
    return fails


def claim_c_route_position(s):
    fails = []
    sycl = s["sycl"]
    head = sycl.find(MUL_MAT_ID_HEAD)
    if head < 0:
        return ["ggml_sycl_mul_mat_id was not found"]
    marks = {name: sycl.find(text, head) for name, text in (
        ("extra", EXTRA_ENSURED), ("route", ROUTE_IF), ("mmid_sched", MMID_SCHED), ("decode_direct", DECODE_DIRECT),
        ("ensure_weight", ENSURE_WEIGHT))}
    missing = [name for name, at in marks.items() if at < 0]
    if missing:
        return [f"ggml_sycl_mul_mat_id lacks {', '.join(missing)}"]
    if not marks["extra"] < marks["route"] < marks["mmid_sched"] < marks["decode_direct"] < marks["ensure_weight"]:
        fails.append("the route is not after the extra ensure block and ahead of MMID_SCHED, yx28 and the weight ensure")
    body = braced_body(sycl, ROUTE_IF, head)
    if body is None or tight(body) != ROUTE_BODY:
        fails.append(f"the route block is not exactly `{ROUTE_FN}(ctx, dst); return;`")
    return fails


def route_fn_body(sycl):
    if sycl.count(ROUTE_FN_HEAD) != 1:
        return None
    return braced_body(sycl, ROUTE_FN_HEAD)


def claim_d_route_body(s):
    body = route_fn_body(s["sycl"])
    if body is None:
        return [f"{ROUTE_FN} is not defined exactly once"]
    fails = []
    for needed in (TABLE_CALL, IDS_CALL, SNAPSHOT_CALL, SUBMIT_CALL, RETAIN_CALL, REFUSAL_WARN, REFUSAL_THROW,
                   ROUTE_LOG, "ggml_sycl::ggml_sycl_moe_route_log_enabled()", PROFILE_SUBMIT, PROFILE_NAME,
                   *PROFILE_KEYS, *OPERAND_CALLS):
        if needed not in body:
            fails.append(f"{ROUTE_FN} does not contain {needed}")
    sycl = s["sycl"]
    helper = braced_body(sycl, ENSURE_HELPER_HEAD) if sycl.count(ENSURE_HELPER_HEAD) == 1 else None
    if helper is None:
        fails.append(f"{ENSURE_HELPER} is not defined exactly once")
    else:
        marks = [helper.find(HELPER_EARLY), helper.find(HELPER_STAMP_SKIP), helper.find(HELPER_PUBLISH),
                 helper.find(HELPER_RETRY), helper.find(HELPER_STAMP_STORE)]
        if min(marks) < 0 or marks != sorted(marks) or \
                helper.count("ggml_sycl_publish_mmid_canonical_aos_experts(") != 1 or \
                helper.count(HELPER_TABLE_CALL) != 1 or helper.count(HELPER_RETRY) != 1 or \
                helper.count("failed = weight.moe_expert_storage_generation;") != 1:
            fails.append("the table helper does not publish once, only on a miss in this device's own buffer at a "
                         "generation not stamped, probe again after it, and stamp only a failed publish or an expert "
                         "that did not resolve")
    operand = braced_body(sycl, OPERAND_HELPER_HEAD) if sycl.count(OPERAND_HELPER_HEAD) == 1 else None
    if operand is None or tight(operand) != OPERAND_HELPER_BODY:
        fails.append(f"{OPERAND_HELPER} does not answer only an on-device pointer, resolved without materializing")
    # The launch count in the profiler row comes from the one tokens-per-launch answer the refusal already asked for.
    if body.count(TOKENS_PER_LAUNCH_CALL) != 1:
        fails.append(f"the route does not ask {TOKENS_PER_LAUNCH_CALL} exactly once")
    profiled = body.find(PROFILE_SUBMIT)
    if profiled >= 0 and not profiled < body.find(SUBMIT_CALL):
        fails.append("the kernel is not submitted inside the profiler's wrapper")
    lowered = body.lower()
    for forbidden in ROUTE_FORBIDDEN:
        if forbidden.lower() in lowered:
            fails.append(f"{ROUTE_FN} contains {forbidden!r}")
    order = [body.find(TABLE_CALL), body.find(SNAPSHOT_CALL), body.find(SUBMIT_CALL), body.find(RETAIN_CALL)]
    if min(order) >= 0 and not order[0] < order[1] < order[2] < order[3]:
        fails.append("the table is not built and its bundle snapshotted before the submit, or the bundle is not "
                     "retained after it")
    warn = body.find(REFUSAL_WARN)
    throw = body.find(REFUSAL_THROW)
    if warn >= 0 and throw >= 0 and not warn < throw:
        fails.append("the refusal throws before it names itself")
    return fails


def claim_e_graph_parity(s):
    sycl = s["sycl"]
    body = route_fn_body(sycl)
    if body is None:
        return [f"{ROUTE_FN} is not defined exactly once"]
    fails = []
    recording = braced_body(body, RECORDING_IF)
    if recording is None:
        fails.append(f"{ROUTE_FN} has no graph-recording arm")
    else:
        if RECORDING_THROW not in recording or "GGML_LOG_WARN(" not in recording:
            fails.append("the recording arm does not warn and throw the sycl::exception the NVFP4 decode node throws")
        at = body.find(RECORDING_IF)
        for later in (TABLE_CALL, IDS_CALL, SUBMIT_CALL):
            if 0 <= body.find(later) < at:
                fails.append(f"{later} runs before the recording arm")
    if GRAPH_FLAG not in body:
        fails.append(f"{ROUTE_FN} does not set moe_graphs_disabled_once as the other MoE routes do")
    compat = braced_body(sycl, COMPAT_HEAD)
    case_at = compat.find(COMPAT_CASE) if compat is not None else -1
    if case_at < 0:
        fails.append("check_graph_compatibility or its MUL_MAT_ID case was not found")
    else:
        case = compat[case_at : compat.find("case GGML_OP_MUL_MAT:", case_at)]
        if COMPAT_QUANT_GUARD not in case or "GGML_TYPE_BF16" in case:
            fails.append("check_graph_compatibility could admit a BF16 MUL_MAT_ID into whole-graph capture")
    return fails


def claim_f_flush_first(s):
    sycl = s["sycl"]
    head = sycl.find(FORWARD_HEAD)
    if head < 0:
        return ["ggml_sycl_compute_forward_impl was not found"]
    flush = sycl.find(OP_ENTRY_FLUSH, head)
    case = sycl.find(MUL_MAT_ID_DISPATCH, head)
    call = sycl.find("ggml_sycl_mul_mat_id(ctx, dst);", head)
    if min(flush, case, call) < 0 or not flush < case < call:
        return ["the op-entry pending-scatter flush does not precede the MUL_MAT_ID dispatch"]
    return []


def claim_g_kernel(s):
    hpp, cpp = s["hpp"], s["cpp"]
    if not hpp or not cpp:
        return ["moe-bf16-direct.hpp / moe-bf16-direct.cpp do not exist"]
    fails = []
    if SUBMIT_DECL not in hpp or "namespace ggml_sycl" not in hpp:
        fails.append("moe-bf16-direct.hpp does not declare ggml_sycl::moe_bf16_direct_submit")
    if SUBMIT_DECL not in cpp or '#include "moe-bf16-direct.hpp"' not in cpp:
        fails.append("moe-bf16-direct.cpp does not define moe_bf16_direct_submit")
    if not SYCL_INCLUDE.search(s["sycl"]):
        fails.append("ggml-sycl.cpp does not include moe-bf16-direct.hpp")
    for needed in ("bf16_to_f32(", "sycl::reduce_over_group(", "quiet_NaN()", "== -1", "reqd_sub_group_size(16)",
                   "sycl::nd_range<3>"):
        if needed not in cpp:
            fails.append(f"the kernel lacks {needed}")
    zero = cpp.find("== -1")
    nan = cpp.find("quiet_NaN()")
    if zero >= 0 and nan >= 0 and not zero < nan:
        fails.append("the id -1 zero row is not decided before the NaN arm")
    # The launch limit and the token chunks: an int-sized range by default, chunks chained so the last event covers
    # every launch, and a dispatch that cannot be chunked refused.
    if "max_launch_items = static_cast<size_t>(std::numeric_limits<int>::max());" not in hpp:
        fails.append("the shape's launch limit does not default to the largest int")
    submit = function_body(cpp, "bool moe_bf16_direct_submit(") or ""
    for needed in ("moe_bf16_direct_tokens_per_launch(shape)", "depends_on("):
        if needed not in cpp or (needed.startswith("moe_") and needed not in submit):
            fails.append(f"the submitter does not use {needed}")
    if "shape.max_launch_items" not in (function_body(cpp, "int64_t moe_bf16_direct_tokens_per_launch(") or ""):
        fails.append("moe_bf16_direct_tokens_per_launch does not read the shape's launch limit")
    # The alignment guards: the float4 activation load only on a 16-byte aligned src1 base and row strides, and the
    # 8-byte weight load only on an 8-byte aligned weight row.
    if X_ALIGN_GUARD not in submit:
        fails.append("the submitter does not take the float4 activation load only on 16-byte aligned src1 rows")
    if W_ALIGN_GUARD not in cpp:
        fails.append("the kernel does not take the 8-byte weight load only on an 8-byte aligned weight row")
    lowered = (hpp + cpp).lower()
    for forbidden in KERNEL_FORBIDDEN:
        if forbidden.lower() in lowered:
            fails.append(f"moe-bf16-direct carries {forbidden!r}")
    return fails


def claim_h_tables(s):
    fails = []
    if "GGML_TYPE_BF16" in s["tables"]:
        fails.append("moe-mmvq-tables.hpp names GGML_TYPE_BF16: the MMVQ tables gained a type they have no kernel for")
    body = braced_body(s["resolved"], NEEDS_DIRECT_HEAD)
    if body is None or tight(body) != NEEDS_DIRECT_BODY:
        fails.append("moe_type_needs_direct_kernel is no longer exactly Q1_0/NVFP4")
    return fails


CLAIMS = {
    "a": claim_a_predicate,
    "b": claim_b_one_predicate,
    "c": claim_c_route_position,
    "d": claim_d_route_body,
    "e": claim_e_graph_parity,
    "f": claim_f_flush_first,
    "g": claim_g_kernel,
    "h": claim_h_tables,
}


@pytest.fixture(scope="module")
def sources():
    return load()


@pytest.mark.parametrize("claim", sorted(CLAIMS))
def test_claim_holds(sources, claim):
    fails = CLAIMS[claim](sources)
    assert not fails, f"claim ({claim}) failed:\n  " + "\n  ".join(fails)


# (claim, file, region, old, new). `old` is matched in the squeezed, comment-stripped text, at its first occurrence
# at or after `region` (a head that occurs once; None is the whole file), and must occur there. Several anchors
# (the graph flag, the fallback throw, the table call) also occur in other routes, so they are scoped to this one.
MUTANTS = [
    ("a", "sycl", PREDICATE_HEAD, "ggml_sycl_moe_plan_owner_for_device(device)", "nullptr"),
    ("a", "sycl", None, PREDICATE_HEAD, PREDICATE_HEAD + ' if (getenv("X")) { return true; }'),
    ("a", "sycl", PREDICATE_HEAD, "return ggml_sycl_moe_tensor_plan_primary_layout(",
     "(void) ggml_sycl_moe_tensor_plan_primary_layout("),
    ("a", "sycl", OWN_BUFFER_HEAD, " || ggml_backend_buffer_is_sycl_split(buffer)", ""),
    # Another card's buffer passes the publish guard: this card's thread would write that device's expert records.
    ("a", "sycl", OWN_BUFFER_HEAD, "buf_ctx->device == device && ", ""),
    ("a", "sycl", OWN_BUFFER_HEAD, " && buf_ctx->managed_meta.tier == ggml_sycl::alloc_tier::DEVICE_VRAM", ""),
    ("a", "sycl", PREDICATE_HEAD, f"{OWN_BUFFER}(src0->buffer, device)",
     "ggml_backend_buffer_is_sycl(src0->buffer) && !ggml_backend_buffer_is_sycl_split(src0->buffer)"),
    ("a", "sycl", ENSURE_HELPER_HEAD, f"!{OWN_BUFFER}(src0->buffer, ctx.device)",
     "!(ggml_backend_buffer_is_sycl(src0->buffer) && !ggml_backend_buffer_is_sycl_split(src0->buffer))"),
    ("a", "sycl", ROUTE_FN_HEAD, TABLE_CALL, f"({OWN_BUFFER}(src0->buffer, ctx.device), {TABLE_CALL})"),
    ("a", "sycl", PREDICATE_HEAD, "src0->view_src != nullptr || ", ""),
    ("a", "sycl", PREDICATE_HEAD, "if (src0->buffer == nullptr) { return true; }", ""),
    ("a", "sycl", PREDICATE_HEAD, "if (src0->buffer == nullptr) { return true; }",
     "if (src0->buffer == nullptr) { return false; }"),
    ("b", "sycl", None, f" && !{PREDICATE}(op, device)) {{", ") {"),
    ("c", "sycl", None, ROUTE_IF, "if (true) {"),
    ("c", "sycl", None, ROUTE_IF + " " + ROUTE_FN + "(ctx, dst); return; }", ""),
    ("c", "sycl", ROUTE_IF, ROUTE_FN + "(ctx, dst); return; }", ROUTE_FN + "(ctx, dst); }"),
    # The fall-through this shape replaced: a block keyed on the predicate lets a run-time `no` reach MMID_SCHED.
    ("c", "sycl", None, ROUTE_IF, f"if ({PREDICATE}(dst, ctx.device)) {{"),
    ("b", "sycl", None, ROUTE_IF, f"if (src0->type == GGML_TYPE_BF16 && {PREDICATE}(dst, ctx.device)) {{"),
    ("b", "sycl", ROUTE_FN_HEAD, ROUTE_ASK, "if (false) {"),
    ("b", "sycl", ROUTE_ASK, "refuse(", "(void) ("),
    ("d", "sycl", ENSURE_HELPER_HEAD, f" || !{OWN_BUFFER}(src0->buffer, ctx.device)", ""),
    ("d", "sycl", ENSURE_HELPER_HEAD, HELPER_STAMP_SKIP, ""),
    ("d", "sycl", ENSURE_HELPER_HEAD, "GGML_LAYOUT_AOS);", "GGML_LAYOUT_SOA);"),
    # A table upload that failed stamped like an expert that did not resolve: this route would stop retrying it.
    ("d", "sycl", ENSURE_HELPER_HEAD, "if (table == nullptr && !every_expert_resolved) {", "if (table == nullptr) {"),
    ("d", "sycl", ROUTE_FN_HEAD, TABLE_CALL,
     "moe_fusion_ensure_full_local_ptr_table(ctx, src0, moe_cache_layer_id(src0->name), GGML_LAYOUT_AOS)"),
    # src1 or dst read through a pointer that may be host memory.
    ("d", "sycl", ROUTE_FN_HEAD, OPERAND_CALLS[0], "ggml_sycl_resolve_tensor_ptr(src1)"),
    ("d", "sycl", ROUTE_FN_HEAD, OPERAND_CALLS[1], "ggml_sycl_resolve_tensor_ptr(dst)"),
    ("d", "sycl", OPERAND_HELPER_HEAD, "return resolved.on_device ? resolved.ptr : nullptr;", "return resolved.ptr;"),
    ("d", "sycl", ROUTE_FN_HEAD, PROFILE_NAME, '"mulmat.moe.bf16"'),
    ("d", "sycl", ROUTE_FN_HEAD, PROFILE_SUBMIT, "ggml_sycl_profile_record_returned_event("),
    ("d", "sycl", ROUTE_FN_HEAD, '";launches="', '";chunks="'),
    ("d", "sycl", ROUTE_FN_HEAD, "std::to_string(shape.k)",
     f"std::to_string(shape.k) + std::to_string({TOKENS_PER_LAUNCH_CALL})"),
    ("g", "cpp", None, "depends_on(", "ignore_dependency("),
    ("g", "cpp", None, "reinterpret_cast<uintptr_t>(src1) % 16 == 0", "true"),
    ("g", "cpp", None, "shape.src1_nb1 % 16 == 0 && shape.src1_nb2 % 16 == 0", "true"),
    ("g", "cpp", None, "(reinterpret_cast<uintptr_t>(w_row) & 7) == 0", "true"),
    ("g", "hpp", None, "std::numeric_limits<int>::max()", "std::numeric_limits<size_t>::max()"),
    ("d", "sycl", ROUTE_FN_HEAD, RETAIN_CALL, "(void) ("),
    ("d", "sycl", ROUTE_FN_HEAD, SNAPSHOT_CALL, "(void) ("),
    ("d", "sycl", ROUTE_FN_HEAD, RETAIN_CALL, "ggml_sycl_retain_moe_ptr_table_leases_until_event(extra, ctx.device, "),
    ("g", "sycl", None, '#include "ggml-sycl/moe-bf16-direct.hpp"', ""),
    ("g", "cpp", None, "return true;", "event->wait(); return true;"),
    ("d", "sycl", ROUTE_FN_HEAD, REFUSAL_THROW, "GGML_ABORT("),
    ("d", "sycl", ROUTE_FN_HEAD, SUBMIT_CALL, "stream->wait(); " + SUBMIT_CALL),
    ("e", "sycl", ROUTE_FN_HEAD, RECORDING_THROW, "throw ggml_sycl_fallback_error("),
    ("e", "sycl", ROUTE_FN_HEAD, GRAPH_FLAG, ""),
    ("e", "sycl", COMPAT_HEAD + "|" + COMPAT_CASE, COMPAT_QUANT_GUARD,
     "if (ggml_is_quantized(src0->type) || src0->type == GGML_TYPE_BF16) {"),
    ("f", "sycl", FORWARD_HEAD, OP_ENTRY_FLUSH, ""),
    ("g", "cpp", None, "sycl::reduce_over_group(", "sycl::select_from_group("),
    ("g", "cpp", None, "quiet_NaN()", "infinity()"),
    ("g", "cpp", None, "reqd_sub_group_size(16)", "reqd_sub_group_size(32)"),
    ("h", "resolved", None, "return type == GGML_TYPE_Q1_0 || type == GGML_TYPE_NVFP4;",
     "return type == GGML_TYPE_Q1_0 || type == GGML_TYPE_NVFP4 || type == GGML_TYPE_BF16;"),
]


def region_start(text, region):
    """Offset of the region: each '|'-separated head is searched after the previous one; -1 when any is missing."""
    at = 0
    for head in (region.split("|") if region else ()):
        at = text.find(head, at)
        if at < 0:
            return -1
    return at


@pytest.mark.parametrize("claim,key,region,old,new", MUTANTS, ids=[f"{m[0]}-{i}" for i, m in enumerate(MUTANTS)])
def test_mutant_turns_claim_red(sources, claim, key, region, old, new):
    text = sources[key]
    start = region_start(text, region)
    at = text.find(old, start) if start >= 0 else -1
    assert at >= 0, f"mutant anchor missing from {FILES[key].name} after {region!r}: {old!r}"
    mutated = dict(sources)
    mutated[key] = text[:at] + new + text[at + len(old) :]
    assert CLAIMS[claim](mutated), f"claim ({claim}) stayed green under the mutant {old!r} -> {new!r}"


def test_tables_mutant_turns_claim_red(sources):
    """The tables file names no BF16 today, so its mutant appends one rather than replacing an anchor."""
    mutated = dict(sources)
    mutated["tables"] = sources["tables"] + " case GGML_TYPE_BF16: return true;"
    assert claim_h_tables(mutated)


def test_claims_are_not_vacuous(sources):
    """Positive controls: the probes find the real anchors, so a red claim is a missing shape, not a parse miss."""
    assert sources["sycl"].find(MUL_MAT_ID_HEAD) >= 0
    assert sources["sycl"].find(FORWARD_HEAD) >= 0
    assert braced_body(sources["sycl"], COMPAT_HEAD) is not None
    assert braced_body(sources["resolved"], NEEDS_DIRECT_HEAD) is not None
    assert sources["tables"]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
