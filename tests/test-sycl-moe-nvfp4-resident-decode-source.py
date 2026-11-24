#!/usr/bin/env python3
"""Source contract for the NVFP4 resident decode route (Task 18 of the Inkling 2 Mini plan).

An NVFP4 MUL_MAT_ID decode node whose experts the plan puts all on this card at AOS runs from the full-local expert
table and the ids where they are, with no host read of the ids, no MMID workspace and no host task: the input is
quantized in place into a Q8_1 scratch slice and the checked MMVQ-ID kernel writes dst, and the kernel's event is dst's
ready event. These claims pin that shape:

  (a) one predicate, ggml_sycl_moe_resident_decode_route_available(op, device, geometry, decline), defined once in
      ggml-sycl.cpp and asked only by the route: NVFP4 experts only (a comment at it says why Q1_0 stays on the
      admitted route), one token, a whole contiguous root weight, adjacent input floats and rows, adjacent output
      floats and rows, a shape the shared geometry helper takes, and, asked last, the plan's memoized answer; it reads
      no environment and asks no other decode decision;
  (b) the plan answer is memoized on the weight, keyed by the plan object, the replan epoch and the expert storage
      generation, all three compared on a hit and all three stored on a miss; the memo holds the key and the answer
      and nothing else, and its function reads none of the node's shapes, strides or ids;
  (c) the route sits in ggml_sycl_mul_mat_id after the BF16 route and ahead of MMID_SCHED, the yx28 decode-direct
      request and the weight ensure, keyed on the NVFP4 expert type; when it takes the node, the kernel's event becomes
      dst's ready event and the guard's queue-sync event is not submitted; a decline falls through unchanged;
  (d) the route takes the AOS full-local table (publishing AOS slices on a miss only from this card's own buffer,
      through the helper the BF16 route asks), the ids' own device storage and a Q8_1 slice from the SCRATCH zone that
      may not spill, and retains a snapshot of the table's dispatch bundle, the ids' storage and the scratch until the
      kernel's event; it never reads ids on the host, builds a resolved batch, leases the workspace, runs a host task,
      waits, copies, aborts or asks the admitted entry;
  (e) under graph recording it warns and throws the sycl::exception the BF16 route throws, before any table, ids or
      scratch work, and it sets moe_graphs_disabled_once there and after a direct run;
  (f) a launch past an int, and a kernel entry that refuses a shape the predicate took, are one named WARN line and
      ggml_sycl_fallback_error; every other no is a decline;
  (g) every decline goes through one function that names it in the route log at INFO, once per weight, card and
      reason (a bit per reason on the weight), behind GGML_SYCL_MOE_ROUTE_LOG; a taken node logs a capped INFO line;
  (h) mmvq: one geometry helper holds the checked size arithmetic and the ids view rule, and both the admitted and the
      resident entries ask it and spell none of it; the resident entry reads no host ids, allocates and waits on
      nothing, quantizes under a profiler row of its own and launches the checked kernel with the table's slot count;
  (i) the route's head comment states what bad ids do: a zero row below zero, a quiet NaN row past the table or at a
      null slot, and the graph succeeds.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SYCL_DIR = ROOT / "ggml/src/ggml-sycl"
FILES = {
    "sycl": SYCL_DIR / "ggml-sycl.cpp",
    "common": SYCL_DIR / "common.hpp",
    "mmvq_cpp": SYCL_DIR / "mmvq.cpp",
    "mmvq_hpp": SYCL_DIR / "mmvq.hpp",
}

PREDICATE = "ggml_sycl_moe_resident_decode_route_available"
PREDICATE_HEAD = (f"static bool {PREDICATE}(const ggml_tensor * op, int device, mmvq_q1_nvfp4_id_geometry * geometry, "
                  "moe_resident_decode_decline * decline) {")
MEMO = "ggml_sycl_moe_resident_decode_plan_all_aos"
MEMO_HEAD = f"static bool {MEMO}(const ggml_tensor * src0, int device) {{"
MEMO_ASK = f"if (!{MEMO}(src0, device)) {{"
MEMO_STRUCT_HEAD = "struct ggml_sycl_moe_resident_decode_plan_memo {"
MEMO_STRUCT_BODY = ("std::weak_ptr<constggml_sycl::placement_plan>plan;uint64_treplan_epoch=0;"
                    "uint64_tstorage_generation=0;boolvalid=false;boolall_aos=false;")
MEMO_FIELD = "ggml_sycl_moe_resident_decode_plan_memo moe_resident_decode_plan[GGML_SYCL_MAX_DEVICES];"
EPOCH_READ = "const uint64_t epoch = ggml_sycl::moe_route_table_current_replan_epoch();"
GENERATION_READ = "const uint64_t generation = weight.moe_expert_storage_generation;"
SAME_PLAN = "const bool same_plan = !memo.plan.owner_before(plan) && !plan.owner_before(memo.plan);"
MEMO_HIT = ("if (memo.valid && memo.replan_epoch == epoch && memo.storage_generation == generation && same_plan) {")
MEMO_ANSWER = ("memo.all_aos = ggml_sycl_moe_tensor_plan_primary_layout(*plan, src0, device, &layout) && "
               "layout == GGML_LAYOUT_AOS;")
MEMO_STORES = ("memo.plan = plan;", "memo.replan_epoch = epoch;", "memo.storage_generation = generation;",
               "memo.valid = true;")
ROUTE_FN = "ggml_sycl_mul_mat_id_nvfp4_resident_decode"
ROUTE_FN_HEAD = (f"static bool {ROUTE_FN}(ggml_backend_sycl_context & ctx, ggml_tensor * dst, "
                 "sycl::event * kernel_event) {")
ROUTE_ASK = f"if (!{PREDICATE}(dst, ctx.device, &geometry, &reason)) {{"
ROUTE_IF = "if (src0->type == GGML_TYPE_NVFP4) {"
CALLER_BLOCK = ("if(src0->type==GGML_TYPE_NVFP4){sycl::eventresident_event;"
                f"if({ROUTE_FN}(ctx,dst,&resident_event)){{"
                "ggml_sycl_set_tensor_ready_event(dst,ctx.device,resident_event,ctx.stream());"
                "ready_guard.published=true;return;}}")
BF16_IF = "if (src0->type == GGML_TYPE_BF16) {"
MUL_MAT_ID_HEAD = "static void ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx, ggml_tensor * dst) try {"
MMID_SCHED = "(g_ggml_sycl_mmid_sched & GGML_SYCL_MMID_SCHED_DEVICE)"
DECODE_DIRECT = "ggml_sycl::moe_decode_direct_request direct_request;"
ENSURE_WEIGHT = "ggml_sycl_ensure_weight_on_device(src0, ctx.device);"
OWN_BUFFER = "ggml_sycl_moe_direct_own_vram_buffer"
PUBLISH_GUARD = f"if ({OWN_BUFFER}(src0->buffer, ctx.device) && ggml_sycl_publish_mmid_canonical_aos_experts("
TABLE_CALL = "moe_fusion_ensure_full_local_ptr_table(ctx, src0, moe_cache_layer_id(src0->name), GGML_LAYOUT_AOS)"
IDS_CALL = "ggml_sycl_moe_ids_device_resident_ptr(ctx, ids, &ids_handle)"
SNAPSHOT_CALL = "ggml_sycl_snapshot_moe_ptr_table_dispatch_bundle("
RETAIN_IDS = "retained.push_back(ids_handle);"
RETAIN_Q8 = "retained.push_back(std::move(q8_handle));"
RESIDENT_ENTRY = "mmvq_submit_q1_nvfp4_aos_id_resident("
RETAIN_CALL = "ggml_sycl::retain_handles_until_event(std::move(retained), *kernel_event);"
SCRATCH_REQUEST = ("ggml_sycl::alloc_request req{};", "req.intent.constraints.must_device = true;",
                   "req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::SCRATCH;",
                   "req.intent.constraints.forbid_vram_zone_spill = true;",
                   "req.intent.constraints.cascade_step = true;", "ggml_sycl::unified_allocate_owner(req)",
                   "ggml_sycl::mem_handle::from_owned_alloc(")
RECORDING_IF = "if (g_ggml_sycl_graph_recording) {"
RECORDING_THROW = "throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),"
GRAPH_FLAG = "ctx.moe_graphs_disabled_once = true;"
REFUSAL_WARN = 'GGML_LOG_WARN("[MoE] NVFP4 resident decode route refused'
REFUSAL_THROW = "throw ggml_sycl_fallback_error("
LAUNCH_LIMIT = ("if (geometry.kernel_items > static_cast<size_t>(INT_MAX) || "
                "geometry.quantize_items > static_cast<size_t>(INT_MAX)) {")
SUBMIT_REFUSED = f"if (!{RESIDENT_ENTRY}"
DECLINE_FN = "ggml_sycl_moe_resident_decode_log_decline"
DECLINE_FN_HEAD = (f"static void {DECLINE_FN}(const ggml_tensor * src0, int device, "
                   "moe_resident_decode_decline decline) {")
DECLINE_LINE = '"[MOE-ROUTE] resident decode declined tensor=%s device=%d reason=%s\\n"'
DECLINE_BITS = "moe_resident_decode_declines_logged[device]"
DECLINE_ONCE = "if ((logged & bit) != 0) { return; } logged |= bit;"
DECLINE_FIELD = "uint32_t moe_resident_decode_declines_logged[GGML_SYCL_MAX_DEVICES] = {};"
DECLINE_LAMBDA = f"{DECLINE_FN}(src0, ctx.device, reason); return false;"
DECLINE_REASONS = ("PROMPT", "OPERANDS", "WEIGHT_VIEW", "INPUT_ROWS", "OUTPUT_ROWS", "GEOMETRY", "PLAN",
                   "IDS_NOT_RESIDENT", "NO_POINTER", "TABLE_INCOMPLETE", "SCRATCH")
ROUTE_LOG = '"[MOE-ROUTE] resident decode tensor='
ROUTE_LOG_GATE = "if (ggml_sycl::ggml_sycl_moe_route_log_enabled()) {"
GEOMETRY_FN = "bool mmvq_q1_nvfp4_id_geometry_of("
GEOMETRY_STRUCT = "struct mmvq_q1_nvfp4_id_geometry {"
GEOMETRY_ASK = ("mmvq_q1_nvfp4_id_geometry_of(weight_type, ncols, nrows_per_expert, top_k, n_tokens, ne11, ids_nb0, "
                "ids_nb1, &geometry)")
IDS_VIEW_RULE = ("if (ids_nb0 != static_cast<int64_t>(sizeof(int32_t)) || "
                 "ids_nb1 < static_cast<int64_t>(sk * sizeof(int32_t))) {")
ADMITTED_FN = "bool mmvq_submit_q1_nvfp4_aos_id_admitted("
RESIDENT_FN = "bool mmvq_submit_q1_nvfp4_aos_id_resident("
QUANTIZE_LABEL = '"mulmat.mmvq.id_quantize_q8_1_aos"'
CHECKED_CALL = ("static_cast<int64_t>(geometry.dst_nb2), n_experts, quant_event, event_out);")
# A host wait is spelled .wait(, ->wait( or sycl::event::wait(: the three spellings are listed apart.
ROUTE_FORBIDDEN = (".wait(", "->wait(", "::wait(", "wait_and_throw", "ids_host", "ggml_sycl_copy_ids_to_host",
                   "build_moe_resolved_batch", "mmid_workspace", "host_task", "GGML_ABORT", "std::exit",
                   "ggml_sycl_ensure_weight_on_device", "malloc_", "sycl::free(", "getenv",
                   "ggml_sycl_moe_decode_direct_decide", "mmvq_submit_q1_nvfp4_aos_id_admitted", "mem_copy",
                   "ggml_sycl_retain_moe_ptr_table_leases_until_event", "ggml_sycl_submit_queue_sync_event",
                   "GGML_LAYOUT_SOA", "return false;")
RESIDENT_FORBIDDEN = (".wait(", "->wait(", "::wait(", "wait_and_throw", "ids_host", "unified_alloc", "malloc",
                      "host_task", "GGML_ABORT", "checked_mul(")


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
    """Squeezed, comment-stripped text per file, and the squeezed raw ggml-sycl.cpp for the comment claims."""
    out = {}
    for key, path in FILES.items():
        out[key] = squeeze(strip_comments(path.read_text(encoding="utf-8"))) if path.exists() else ""
    out["sycl_raw"] = squeeze(FILES["sycl"].read_text(encoding="utf-8"))
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


def defined_once(text, head):
    return text.count(head) == 1


def comment_above(raw, head, width=1600):
    """The squeezed raw text just above `head`: the comment block a reader sees with the definition."""
    at = raw.find(head)
    return raw[max(0, at - width) : at] if at >= 0 else ""


def arm_after(body, anchor):
    """The braced arm of the first `if` whose condition starts with `anchor`, or None."""
    at = body.find(anchor)
    brace = body.find("{", at) if at >= 0 else -1
    return braced_body(body, body[at : brace + 1], at) if brace >= 0 else None


def claim_a_predicate(s):
    sycl = s["sycl"]
    if not defined_once(sycl, PREDICATE_HEAD) or len(re.findall(rf"\bbool {PREDICATE}\(", sycl)) != 1:
        return [f"{PREDICATE} is not defined exactly once as a static bool in ggml-sycl.cpp"]
    fails = []
    asks = len(re.findall(rf"\b{PREDICATE}\(", sycl))
    if asks != 2 or ROUTE_ASK not in (braced_body(sycl, ROUTE_FN_HEAD) or ""):
        fails.append(f"{PREDICATE} appears {asks} times; want its definition and the route's one ask")
    body = braced_body(sycl, PREDICATE_HEAD) or ""
    for needed in ("src0->type != GGML_TYPE_NVFP4", "ids->ne[1] != 1", "src0->view_src != nullptr",
                   "!ggml_is_contiguous(src0)", "src1->nb[0] != sizeof(float)",
                   "src1->nb[1] != src1->ne[0] * sizeof(float)", "op->nb[0] != sizeof(float)",
                   "op->nb[1] != op->ne[0] * sizeof(float)", "!mmvq_q1_nvfp4_id_geometry_of(src0->type,", MEMO_ASK):
        if needed not in body:
            fails.append(f"the predicate does not ask {needed}")
    for forbidden in ("GGML_TYPE_Q1_0", "GGML_TYPE_BF16", "getenv", "ggml_sycl_moe_decode_direct_decide",
                      "ggml_sycl_moe_tensor_plan_primary_layout", "ggml_sycl_moe_plan_owner_for_device",
                      "ggml_sycl_resolve", "unified_allocate"):
        if forbidden in body:
            fails.append(f"the predicate carries {forbidden!r}")
    # The plan's memoized answer is the last question, asked once: every shape and stride is asked of the node before it.
    if body.count(f"{MEMO}(") != 1:
        fails.append(f"the predicate asks {MEMO} other than once")
    memo_at = body.find(MEMO_ASK)
    for earlier in ("ids->ne[1] != 1", "src1->nb[1]", "op->nb[1]", "mmvq_q1_nvfp4_id_geometry_of("):
        if memo_at >= 0 and not 0 <= body.find(earlier) < memo_at:
            fails.append(f"the plan memo is asked before {earlier}")
    if not tight(body).endswith("returntrue;"):
        fails.append("the predicate does not end by taking the node after the plan's answer")
    note = comment_above(s["sycl_raw"], PREDICATE_HEAD)
    if not all(word in note for word in ("Q1_0", "admitted", "failure")):
        fails.append("no comment at the predicate says Q1_0 stays on the admitted route for its failure-path tests")
    return fails


def claim_b_memo(s):
    sycl, common = s["sycl"], s["common"]
    if not defined_once(sycl, MEMO_HEAD):
        return [f"{MEMO} is not defined exactly once"]
    fails = []
    asks = len(re.findall(rf"\b{MEMO}\(", sycl))
    if asks != 2:
        fails.append(f"{MEMO} appears {asks} times; want its definition and the predicate's one ask")
    body = braced_body(sycl, MEMO_HEAD) or ""
    for needed in (EPOCH_READ, GENERATION_READ, SAME_PLAN, MEMO_HIT, MEMO_ANSWER, *MEMO_STORES,
                   "ggml_sycl_moe_plan_owner_for_device(device)"):
        if needed not in body:
            fails.append(f"the memo does not contain {needed}")
    hit = braced_body(body, MEMO_HIT)
    if hit is None or tight(hit) != "returnmemo.all_aos;":
        fails.append("a memo hit does not answer with the memoized answer alone")
    hit_at = body.find(MEMO_HIT)
    for store in (MEMO_ANSWER, *MEMO_STORES):
        if hit_at >= 0 and not body.find(store) > hit_at:
            fails.append(f"{store} is not a store after the hit test")
    # The memo answers the plan question only: nothing about the node's input, ids, strides or launch.
    for forbidden in ("op->", "src1", "ids", "->nb[", "geometry", "->ne["):
        if forbidden in body:
            fails.append(f"the memo reads {forbidden!r}, which its key does not cover")
    if tight(braced_body(common, MEMO_STRUCT_HEAD) or "") != MEMO_STRUCT_BODY:
        fails.append("the memo struct is not exactly the plan identity, the replan epoch, the storage generation, the "
                     "valid bit and the answer")
    if common.count(MEMO_FIELD) != 1:
        fails.append("the weight does not carry one memo per device")
    return fails


def claim_c_route_position(s):
    sycl = s["sycl"]
    head = sycl.find(MUL_MAT_ID_HEAD)
    if head < 0:
        return ["ggml_sycl_mul_mat_id was not found"]
    marks = {name: sycl.find(text, head) for name, text in (
        ("bf16", BF16_IF), ("resident", f"{ROUTE_FN}(ctx, dst, &resident_event)"), ("mmid_sched", MMID_SCHED),
        ("decode_direct", DECODE_DIRECT), ("ensure_weight", ENSURE_WEIGHT))}
    missing = [name for name, at in marks.items() if at < 0]
    if missing:
        return [f"ggml_sycl_mul_mat_id lacks {', '.join(missing)}"]
    fails = []
    if not marks["bf16"] < marks["resident"] < marks["mmid_sched"] < marks["decode_direct"] < marks["ensure_weight"]:
        fails.append("the route is not after the BF16 route and ahead of MMID_SCHED, yx28 and the weight ensure")
    if_at = sycl.rfind(ROUTE_IF, marks["bf16"], marks["resident"])
    body = braced_body(sycl, ROUTE_IF, if_at) if if_at >= 0 else None
    if body is None or tight(ROUTE_IF + body + "}") != CALLER_BLOCK:
        fails.append("the caller is not exactly: on NVFP4 ask the route, and when it takes the node publish its kernel "
                     "event as dst's ready event, mark the guard published and return")
    calls = len(re.findall(rf"\b{ROUTE_FN}\(", sycl))
    if calls != 2:
        fails.append(f"{ROUTE_FN} appears {calls} times; want its definition and the one caller")
    return fails


def route_body(s):
    return braced_body(s["sycl"], ROUTE_FN_HEAD) if defined_once(s["sycl"], ROUTE_FN_HEAD) else None


def claim_d_route_body(s):
    body = route_body(s)
    if body is None:
        return [f"{ROUTE_FN} is not defined exactly once"]
    fails = []
    for needed in (ROUTE_ASK, TABLE_CALL, PUBLISH_GUARD, IDS_CALL, SNAPSHOT_CALL, RETAIN_IDS, RETAIN_Q8,
                   RESIDENT_ENTRY, RETAIN_CALL, *SCRATCH_REQUEST):
        if needed not in body:
            fails.append(f"{ROUTE_FN} does not contain {needed}")
    if body.count("ggml_sycl_publish_mmid_canonical_aos_experts(") != 1:
        fails.append("the route publishes other than once, behind the own-VRAM-buffer test")
    if body.count(RESIDENT_ENTRY) != 1 or len(re.findall(r"\bmmvq_submit_q1_nvfp4_aos_id_resident\(", s["sycl"])) != 1:
        fails.append("the resident entry is not submitted exactly once, from the route")
    # The decline lambda's own `return false;` is the one a decline goes through; any other is a silent decline.
    lowered = body.replace(DECLINE_LAMBDA, " ", 1).lower()
    for forbidden in ROUTE_FORBIDDEN:
        if forbidden.lower() in lowered:
            fails.append(f"{ROUTE_FN} contains {forbidden!r}")
    order = [body.find(ROUTE_ASK), body.find(TABLE_CALL), body.find(SNAPSHOT_CALL), body.find(RESIDENT_ENTRY),
             body.find(RETAIN_CALL)]
    if min(order) >= 0 and not order[0] < order[1] < order[2] < order[3] < order[4]:
        fails.append("the predicate, the table, the bundle snapshot, the submit and the retention are out of order")
    retains = [body.find(RETAIN_IDS), body.find(RETAIN_Q8)]
    if min(retains) >= 0 and min(order) >= 0 and not (order[2] < min(retains) and max(retains) < order[3]):
        fails.append("the ids' storage and the scratch do not join the snapshot before the submit")
    return fails


def claim_e_graph_parity(s):
    body = route_body(s)
    if body is None:
        return [f"{ROUTE_FN} is not defined exactly once"]
    fails = []
    recording = braced_body(body, RECORDING_IF)
    if recording is None:
        fails.append(f"{ROUTE_FN} has no graph-recording arm")
    else:
        for needed in (RECORDING_THROW, "GGML_LOG_WARN(", GRAPH_FLAG):
            if needed not in recording:
                fails.append(f"the recording arm lacks {needed}")
        at = body.find(RECORDING_IF)
        for later in (TABLE_CALL, IDS_CALL, "unified_allocate_owner(", RESIDENT_ENTRY):
            if 0 <= body.find(later) < at:
                fails.append(f"{later} runs before the recording arm")
    submit_at = body.find(RESIDENT_ENTRY)
    if submit_at < 0 or body.find(GRAPH_FLAG, submit_at) < 0:
        fails.append("a direct run does not set moe_graphs_disabled_once after the submit")
    return fails


def claim_f_refusals(s):
    body = route_body(s)
    if body is None:
        return [f"{ROUTE_FN} is not defined exactly once"]
    fails = []
    for anchor, what in ((LAUNCH_LIMIT, "a launch past an int"), (SUBMIT_REFUSED, "an entry refusal")):
        arm = arm_after(body, anchor)
        if arm is None or not tight(arm).startswith("refuse("):
            fails.append(f"{what} is not a named refusal")
    warn, throw = body.find(REFUSAL_WARN), body.find(REFUSAL_THROW)
    if warn < 0 or throw < 0 or not warn < throw:
        fails.append("the refusal does not name itself in a WARN line before it throws ggml_sycl_fallback_error")
    if body.count(REFUSAL_THROW) != 1:
        fails.append("the route throws ggml_sycl_fallback_error from more than the one refusal")
    limit_at, table_at = body.find(LAUNCH_LIMIT), body.find(TABLE_CALL)
    if limit_at >= 0 and table_at >= 0 and not limit_at < table_at:
        fails.append("the launch limit is checked after the table is built")
    return fails


def claim_g_logs(s):
    sycl, common = s["sycl"], s["common"]
    body = route_body(s)
    if body is None or not defined_once(sycl, DECLINE_FN_HEAD):
        return [f"{ROUTE_FN} or {DECLINE_FN} is not defined exactly once"]
    fails = []
    log_fn = braced_body(sycl, DECLINE_FN_HEAD) or ""
    for needed in (DECLINE_LINE, DECLINE_BITS, DECLINE_ONCE, "GGML_LOG_INFO(",
                   "!ggml_sycl::ggml_sycl_moe_route_log_enabled()"):
        if needed not in log_fn:
            fails.append(f"the decline log does not contain {needed}")
    if log_fn.find(DECLINE_ONCE) > log_fn.find("GGML_LOG_INFO(") >= 0:
        fails.append("the decline log prints before it tests the weight's bit for the reason")
    if common.count(DECLINE_FIELD) != 1:
        fails.append("the weight does not carry one decline bit set per device")
    if DECLINE_LAMBDA not in body or len(re.findall(rf"\b{DECLINE_FN}\(", sycl)) != 2:
        fails.append("the route does not name every decline through the one decline log")
    for reason in DECLINE_REASONS:
        if f"moe_resident_decode_decline::{reason}" not in sycl:
            fails.append(f"no decline is named {reason}")
    for reason in ("IDS_NOT_RESIDENT", "NO_POINTER", "TABLE_INCOMPLETE", "SCRATCH"):
        if f"return decline(moe_resident_decode_decline::{reason});" not in body:
            fails.append(f"the route does not decline {reason}")
    if "return decline(reason);" not in body:
        fails.append("the route does not decline with the predicate's reason")
    gate = braced_body(body, ROUTE_LOG_GATE)
    if gate is None or ROUTE_LOG not in gate or "GGML_LOG_INFO(" not in gate or "static std::atomic<int>" not in gate:
        fails.append("a taken node does not log a capped INFO route line behind GGML_SYCL_MOE_ROUTE_LOG")
    return fails


def claim_h_mmvq(s):
    cpp, hpp = s["mmvq_cpp"], s["mmvq_hpp"]
    fails = []
    if hpp.count(GEOMETRY_STRUCT) != 1 or GEOMETRY_FN not in hpp or RESIDENT_FN not in hpp:
        fails.append("mmvq.hpp does not declare the geometry, its helper and the resident entry")
    geometry = function_body(cpp, GEOMETRY_FN)
    admitted = function_body(cpp, ADMITTED_FN)
    resident = function_body(cpp, RESIDENT_FN)
    if geometry is None or admitted is None or resident is None:
        return fails + ["mmvq.cpp does not define the geometry helper, the admitted entry and the resident entry"]
    if cpp.count(GEOMETRY_FN) != 1:
        fails.append("the geometry helper is defined more than once")
    for needed in (IDS_VIEW_RULE, "checked_mul(", "out->kernel_items = kernel_items;",
                   "out->quantize_items = quantize_items;"):
        if needed not in geometry:
            fails.append(f"the geometry helper does not hold {needed}")
    for name, entry in (("admitted", admitted), ("resident", resident)):
        if GEOMETRY_ASK not in entry:
            fails.append(f"the {name} entry does not ask the geometry helper")
        if "checked_mul(" in entry or "ids_nb0 != static_cast<int64_t>(sizeof(int32_t))" in entry:
            fails.append(f"the {name} entry spells part of the geometry itself")
        if CHECKED_CALL not in entry:
            fails.append(f"the {name} entry does not launch the checked kernel with the table's slot count")
    for needed in ("activation_q8_bytes != geometry.q8_bytes", "dst_bytes != geometry.output_bytes",
                   "ggml_sycl_profile_submit(", QUANTIZE_LABEL, "mmvq_submit_quantize_q8_1_aos_after("):
        if needed not in resident:
            fails.append(f"the resident entry does not contain {needed}")
    lowered = resident.lower()
    for forbidden in RESIDENT_FORBIDDEN:
        if forbidden.lower() in lowered:
            fails.append(f"the resident entry contains {forbidden!r}")
    quantize, launch = resident.find("ggml_sycl_profile_submit("), resident.find(CHECKED_CALL)
    if quantize >= 0 and launch >= 0 and not quantize < launch:
        fails.append("the resident entry launches the kernel before it quantizes")
    return fails


def claim_i_bad_ids(s):
    note = comment_above(s["sycl_raw"], ROUTE_FN_HEAD)
    fails = []
    for needed in ("zero row", "quiet NaN", "graph succeeds"):
        if needed not in note:
            fails.append(f"the route's head comment does not say {needed!r}")
    return fails


CLAIMS = {
    "a": claim_a_predicate,
    "b": claim_b_memo,
    "c": claim_c_route_position,
    "d": claim_d_route_body,
    "e": claim_e_graph_parity,
    "f": claim_f_refusals,
    "g": claim_g_logs,
    "h": claim_h_mmvq,
    "i": claim_i_bad_ids,
}


@pytest.fixture(scope="module")
def sources():
    return load()


@pytest.mark.parametrize("claim", sorted(CLAIMS))
def test_claim_holds(sources, claim):
    fails = CLAIMS[claim](sources)
    assert not fails, f"claim ({claim}) failed:\n  " + "\n  ".join(fails)


# (claim, file, region, old, new). `old` is matched in the squeezed, comment-stripped text, at its first occurrence
# at or after `region` (a head that occurs once; None is the whole file), and must occur there. Several anchors (the
# graph flag, the fallback throw, the table call) also occur in other routes, so they are scoped to this one.
MUTANTS = [
    ("a", "sycl", PREDICATE_HEAD, "src0->type != GGML_TYPE_NVFP4",
     "(src0->type != GGML_TYPE_NVFP4 && src0->type != GGML_TYPE_Q1_0)"),
    ("a", "sycl", PREDICATE_HEAD, "ids->ne[1] != 1 || ", ""),
    ("a", "sycl", PREDICATE_HEAD, "src0->view_src != nullptr || ", ""),
    ("a", "sycl", PREDICATE_HEAD, "src1->nb[1] != src1->ne[0] * sizeof(float)", "false"),
    ("a", "sycl", PREDICATE_HEAD, "op->nb[1] != op->ne[0] * sizeof(float)", "false"),
    ("a", "sycl", PREDICATE_HEAD, MEMO_ASK, "if (false) {"),
    ("a", "sycl", None, PREDICATE_HEAD, PREDICATE_HEAD + ' if (getenv("X")) { return true; }'),
    # The memo asked first: a node whose shape the route cannot take would read the plan before its strides.
    ("a", "sycl", PREDICATE_HEAD, "const ggml_tensor * src0 = op->src[0];",
     f"const ggml_tensor * src0 = op->src[0]; (void) {MEMO}(src0, device);"),
    ("b", "sycl", MEMO_HEAD, " && memo.replan_epoch == epoch", ""),
    ("b", "sycl", MEMO_HEAD, " && memo.storage_generation == generation", ""),
    ("b", "sycl", MEMO_HEAD, " && same_plan", ""),
    ("b", "sycl", MEMO_HEAD, "memo.plan = plan;", ""),
    ("b", "sycl", MEMO_HEAD, "memo.replan_epoch = epoch;", ""),
    ("b", "sycl", MEMO_HEAD, "memo.storage_generation = generation;", ""),
    ("b", "sycl", MEMO_HEAD, "ggml_sycl::moe_route_table_current_replan_epoch()", "0"),
    ("b", "sycl", MEMO_HEAD, "&& layout == GGML_LAYOUT_AOS;", ";"),
    # The memo widened to a shape question its key does not cover.
    ("b", "sycl", MEMO_HEAD, "return memo.all_aos; }", "return memo.all_aos && src0->ne[1] > 0; }"),
    ("b", "common", None, "bool all_aos = false;", "bool all_aos = false; int64_t ne11 = 0;"),
    ("c", "sycl", MUL_MAT_ID_HEAD, "ready_guard.published = true; return; } }", "return; } }"),
    ("c", "sycl", MUL_MAT_ID_HEAD, "ggml_sycl_set_tensor_ready_event(dst, ctx.device, resident_event, ctx.stream());",
     ""),
    ("c", "sycl", MUL_MAT_ID_HEAD, "ggml_sycl_set_tensor_ready_event(dst, ctx.device, resident_event, ctx.stream());",
     "ggml_sycl_set_tensor_ready_event(dst, ctx.device, ggml_sycl_submit_queue_sync_event(*ctx.stream()), "
     "ctx.stream());"),
    ("c", "sycl", MUL_MAT_ID_HEAD, ROUTE_IF + " sycl::event resident_event;",
     "if (src0->type == GGML_TYPE_NVFP4 || src0->type == GGML_TYPE_Q1_0) { sycl::event resident_event;"),
    ("d", "sycl", ROUTE_FN_HEAD, TABLE_CALL, TABLE_CALL.replace("GGML_LAYOUT_AOS", "GGML_LAYOUT_SOA")),
    ("d", "sycl", ROUTE_FN_HEAD, PUBLISH_GUARD, "if (ggml_sycl_publish_mmid_canonical_aos_experts("),
    ("d", "sycl", ROUTE_FN_HEAD, RETAIN_CALL, "(void) retained;"),
    ("d", "sycl", ROUTE_FN_HEAD, RETAIN_Q8, ""),
    ("d", "sycl", ROUTE_FN_HEAD, RETAIN_IDS, ""),
    ("d", "sycl", ROUTE_FN_HEAD, SNAPSHOT_CALL, "(void) ("),
    ("d", "sycl", ROUTE_FN_HEAD, "req.intent.constraints.forbid_vram_zone_spill = true;", ""),
    ("d", "sycl", ROUTE_FN_HEAD, "ggml_sycl::vram_zone_id::SCRATCH;", "ggml_sycl::vram_zone_id::COUNT;"),
    ("d", "sycl", ROUTE_FN_HEAD, RESIDENT_ENTRY, "ctx.stream()->wait(); " + RESIDENT_ENTRY),
    ("d", "sycl", ROUTE_FN_HEAD, IDS_CALL, "ggml_sycl_copy_ids_to_host(ctx, ids), " + IDS_CALL),
    ("e", "sycl", ROUTE_FN_HEAD, RECORDING_THROW, "throw ggml_sycl_fallback_error("),
    ("e", "sycl", ROUTE_FN_HEAD + "|" + RECORDING_IF, GRAPH_FLAG, ""),
    ("e", "sycl", ROUTE_FN_HEAD + "|" + "ggml_sycl::retain_handles_until_event(", GRAPH_FLAG, ""),
    ("f", "sycl", ROUTE_FN_HEAD, LAUNCH_LIMIT + " refuse(", LAUNCH_LIMIT + " (void) ("),
    ("f", "sycl", ROUTE_FN_HEAD, REFUSAL_THROW, "GGML_ABORT("),
    ("g", "sycl", DECLINE_FN_HEAD, DECLINE_ONCE, ""),
    ("g", "sycl", DECLINE_FN_HEAD, "!ggml_sycl::ggml_sycl_moe_route_log_enabled() || ", ""),
    ("g", "sycl", ROUTE_FN_HEAD, "return decline(moe_resident_decode_decline::TABLE_INCOMPLETE);", "return false;"),
    ("g", "sycl", ROUTE_FN_HEAD, DECLINE_LAMBDA, "return false;"),
    ("h", "mmvq_cpp", ADMITTED_FN, GEOMETRY_ASK, "true"),
    ("h", "mmvq_cpp", GEOMETRY_FN, "ids_nb1 < static_cast<int64_t>(sk * sizeof(int32_t))",
     "ids_nb1 != static_cast<int64_t>(sk * sizeof(int32_t))"),
    ("h", "mmvq_cpp", RESIDENT_FN, QUANTIZE_LABEL, '"mulmat.mmvq.id_aos_nvfp4"'),
    ("h", "mmvq_cpp", RESIDENT_FN, "dst_bytes != geometry.output_bytes", "false"),
    ("h", "mmvq_cpp", RESIDENT_FN, CHECKED_CALL, "static_cast<int64_t>(geometry.dst_nb2), quant_event, event_out);"),
    ("h", "mmvq_cpp", RESIDENT_FN, "return mmvq_submit_q1_nvfp4_aos_id(",
     "q.wait(); return mmvq_submit_q1_nvfp4_aos_id("),
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


def test_comment_mutants_turn_claims_red(sources):
    """The comment claims read the raw text: a predicate or route head without its comment fails them."""
    for claim, head in (("a", PREDICATE_HEAD), ("i", ROUTE_FN_HEAD)):
        raw = sources["sycl_raw"]
        at = raw.find(head)
        assert at >= 0, f"{head!r} is missing from the raw text"
        mutated = dict(sources)
        mutated["sycl_raw"] = raw[: max(0, at - 1600)] + " " * min(at, 1600) + raw[at:]
        assert CLAIMS[claim](mutated), f"claim ({claim}) stayed green with no comment above {head!r}"


def test_claims_are_not_vacuous(sources):
    """Positive controls: the probes find the real anchors, so a red claim is a missing shape, not a parse miss."""
    assert sources["sycl"].find(MUL_MAT_ID_HEAD) >= 0
    assert sources["sycl"].find(BF16_IF) >= 0
    assert sources["sycl"].find(MMID_SCHED) >= 0
    assert function_body(sources["mmvq_cpp"], ADMITTED_FN) is not None
    assert sources["common"]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
