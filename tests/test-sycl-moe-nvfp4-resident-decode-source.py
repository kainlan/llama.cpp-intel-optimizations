#!/usr/bin/env python3
"""Source contract for the NVFP4 resident decode route.

An NVFP4 MUL_MAT_ID decode node whose experts the plan puts all on this card at AOS runs from the full-local expert
table and the ids where they are, with no host read of the ids, no MMID workspace and no host task: the input is
quantized in place into a Q8_1 scratch slice and the checked MMVQ-ID kernel writes dst, and the kernel's event is dst's
ready event. These claims pin that shape:

  (a) one predicate, ggml_sycl_moe_resident_decode_route_available(op, device, geometry, fit, decline), defined once
      in ggml-sycl.cpp and asked only by the route: NVFP4 experts only (a comment at it says why Q1_0 stays on the
      admitted route), one token, a whole contiguous root weight, adjacent input floats and rows, adjacent output
      floats and rows, a shape the shared geometry helper has a kernel for (a launch past an int is a yes with that
      fit), and, asked last, the plan's memoized answer; it reads no environment and asks no other decode decision;
  (b) the plan answer is memoized on the weight, keyed by the plan object, the replan epoch and the expert storage
      generation, all three compared on a hit and all three stored on a miss; the memo holds the key and the answer
      and nothing else, and its function reads none of the node's shapes, strides or ids;
  (c) the route sits in ggml_sycl_mul_mat_id after the BF16 route and ahead of MMID_SCHED, the yx28 decode-direct
      request and the weight ensure, keyed on the NVFP4 expert type; when it takes the node the caller only marks the
      guard published, so the guard's queue-sync event is not submitted; a decline falls through unchanged;
  (d) the route takes the AOS full-local table through the helper the BF16 route asks (which publishes AOS slices on
      a miss only from this card's own buffer, and not again at a storage generation where that already failed),
      the input and the output only where they resolve to this card's own memory, the ids' own device storage and a
      Q8_1 slice from the SCRATCH zone that may not spill, and keeps a snapshot of the table's dispatch bundle taken
      without the identity dedup, the ids' storage and the scratch until the kernel's event; it never reads ids on
      the host, builds a resolved batch, leases the workspace, runs a host task, waits, copies, aborts or asks the
      admitted entry;
  (e) under graph recording it warns and throws the sycl::exception the BF16 route throws, before any table, ids or
      scratch work, and it sets moe_graphs_disabled_once there and after a direct run;
  (f) a launch past an int, asked first after the recording arm, and a kernel entry that refuses a shape the
      predicate took, are one named WARN line and ggml_sycl_fallback_error; every other no is a decline;
  (g) every decline goes through one function that names it in the route log at INFO, once per weight, card and
      reason (a bit per reason on the weight), behind GGML_SYCL_MOE_ROUTE_LOG; a taken node logs a capped INFO line;
  (h) mmvq: one geometry helper holds the checked size arithmetic, the ids view rule and the int bound on both
      launches, and answers FITS, NO_KERNEL or LAUNCH_PAST_INT; the admitted and the resident entries both ask it,
      take only FITS and spell none of it; the resident entry reads no host ids, allocates and waits on nothing,
      quantizes under a profiler row of its own and launches the checked kernel with the table's slot count, whose
      event, and no other, it hands back;
  (i) the route's head comment states what bad ids do: a zero row below zero, a quiet NaN row past the table or at a
      null slot, and the graph succeeds;
  (j) no handle the submitted work reads is released before that work is done. The handles live in one retention
      whose destructor drains the stream while it is armed; the route arms it right before the entry (which submits
      the quantize, then the kernel, each possibly through the profiler), publishes the same vector with
      retain_handles_until_event_or_keep, and disarms only after that returns. So a submit or a profiler record that
      throws, a refusal after the quantize, and a publication that throws all drain before the handles go, and a
      taken node never drains. The publication leaves the vector as it was when it throws: the record enters the
      store with an empty vector and takes the handles by a swap, and the graph-lifetime path reserves before it
      moves anything;
  (k) the cheap per-node costs stay out: the entries read the card's half support from the capabilities taken at
      init, the plan owner is fetched once, the quantize's profiler label is built only while the profiler is on,
      the route asks ctx.stream() once, and the kernel's event is constructed only once the route has taken the node.
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
    "mem_cpp": SYCL_DIR / "mem-handle.cpp",
    "mem_hpp": SYCL_DIR / "mem-handle.hpp",
}

PREDICATE = "ggml_sycl_moe_resident_decode_route_available"
PREDICATE_HEAD = (f"static bool {PREDICATE}(const ggml_tensor * op, int device, mmvq_q1_nvfp4_id_geometry * geometry, "
                  "mmvq_q1_nvfp4_id_fit * fit, moe_resident_decode_decline * decline) {")
PREDICATE_FIT = ("*fit = mmvq_q1_nvfp4_id_geometry_of(src0->type,", 
                 "if (*fit == mmvq_q1_nvfp4_id_fit::NO_KERNEL) { return no(moe_resident_decode_decline::GEOMETRY); }")
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
ROUTE_FN_HEAD = f"static bool {ROUTE_FN}(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {{"
ROUTE_ASK = f"if (!{PREDICATE}(dst, ctx.device, &geometry, &fit, &reason)) {{"
ROUTE_IF = f"if (src0->type == GGML_TYPE_NVFP4 && {ROUTE_FN}(ctx, dst)) {{"
CALLER_BLOCK = f"if(src0->type==GGML_TYPE_NVFP4&&{ROUTE_FN}(ctx,dst)){{ready_guard.published=true;return;}}"
BF16_IF = "if (src0->type == GGML_TYPE_BF16) {"
MUL_MAT_ID_HEAD = "static void ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx, ggml_tensor * dst) try {"
MMID_SCHED = "(g_ggml_sycl_mmid_sched & GGML_SYCL_MMID_SCHED_DEVICE)"
DECODE_DIRECT = "ggml_sycl::moe_decode_direct_request direct_request;"
ENSURE_WEIGHT = "ggml_sycl_ensure_weight_on_device(src0, ctx.device);"
OWN_BUFFER = "ggml_sycl_moe_direct_own_vram_buffer"
ENSURE_HELPER = "ggml_sycl_moe_direct_ensure_aos_table"
ENSURE_HELPER_HEAD = (f"static const void * const * {ENSURE_HELPER}(ggml_backend_sycl_context & ctx, "
                      "const ggml_tensor * src0) {")
TABLE_CALL = f"{ENSURE_HELPER}(ctx, src0)"
HELPER_TABLE_CALL = "table = moe_fusion_ensure_full_local_ptr_table(ctx, src0, layer, GGML_LAYOUT_AOS);"
HELPER_EARLY = (f"if (table != nullptr || src0->extra == nullptr || !{OWN_BUFFER}(src0->buffer, ctx.device)) "
                "{ return table; }")
STAMP_READ = "uint64_t & failed = weight.moe_direct_publish_failed_generation[ctx.device];"
STAMP_SKIP = "if (failed == weight.moe_expert_storage_generation) { return nullptr; }"
STAMP_STORE = "if (table == nullptr) { failed = weight.moe_expert_storage_generation; }"
STAMP_FIELD = "uint64_t moe_direct_publish_failed_generation[GGML_SYCL_MAX_DEVICES] = {};"
HELPER_PUBLISH = "if (ggml_sycl_publish_mmid_canonical_aos_experts(src0, ctx.device, GGML_LAYOUT_AOS,"
OPERAND_HELPER = "ggml_sycl_moe_direct_device_operand"
OPERAND_HELPER_HEAD = f"static void * {OPERAND_HELPER}(const ggml_tensor * tensor, int device) {{"
OPERAND_HELPER_BODY = ("constggml_sycl::resolved_ptrresolved=ggml_sycl_resolve_no_materialize(tensor,device);"
                       "returnresolved.on_device?resolved.ptr:nullptr;")
OPERAND_CALLS = (f"{OPERAND_HELPER}(src1, ctx.device)", f"{OPERAND_HELPER}(dst, ctx.device)")
IDS_CALL = "ggml_sycl_moe_ids_device_resident_ptr(ctx, ids, &ids_handle)"
SNAPSHOT_CALL = "ggml_sycl_snapshot_moe_ptr_table_dispatch_bundle("
SNAPSHOT_NO_DEDUP = "ctx.device, false, false, moe_dispatch_bundle_dedup::NONE);"
SNAPSHOT_DEF = ("std::vector<ggml_sycl::mem_handle> ggml_sycl_snapshot_moe_ptr_table_dispatch_bundle("
                "const ggml_tensor_extra_gpu * extra, int device, bool include_compact, bool include_missing, "
                "moe_dispatch_bundle_dedup dedup) {")
SNAPSHOT_APPEND = ("constautoappend=[&](constggml_sycl::mem_handle&handle){"
                   "if(dedup==moe_dispatch_bundle_dedup::BY_IDENTITY){"
                   "ggml_sycl_append_moe_dispatch_handle(bundle,handle);}"
                   "elseif(ggml_sycl_mem_handle_has_identity(handle)){bundle.push_back(handle);}};")
RETAIN_IDS = "retention.handles.push_back(std::move(ids_handle));"
RETAIN_Q8 = "retention.handles.push_back(std::move(q8_handle));"
RESIDENT_ENTRY = "mmvq_submit_q1_nvfp4_aos_id_resident("
RETAIN_CALL = "ggml_sycl::retain_handles_until_event_or_keep(retention.handles, kernel_event);"
RETENTION = "ggml_sycl_moe_resident_decode_retention"
RETENTION_HEAD = f"struct {RETENTION} {{"
RETENTION_DTOR = ("~ggml_sycl_moe_resident_decode_retention(){if(submitted!=nullptr){try{submitted->wait_and_throw();}"
                  "catch(...){}}}")
RETENTION_NO_COPY = (f"{RETENTION}(const {RETENTION} &) = delete;",
                     f"{RETENTION} & operator=(const {RETENTION} &) = delete;")
ARM = "retention.submitted = stream;"
DISARM = "retention.submitted = nullptr;"
# The protocol, whitespace removed: everything the kernels read joins the retention, which is armed right before the
# entry and disarmed only once the publication has returned; dst's ready event is the event the entry filled.
RETENTION_BEFORE = (f"{RETENTION}retention;retention.handles=ggml_sycl_snapshot_moe_ptr_table_dispatch_bundle("
                    "static_cast<constggml_tensor_extra_gpu*>(src0->extra),ctx.device,false,false,"
                    "moe_dispatch_bundle_dedup::NONE);retention.handles.push_back(std::move(ids_handle));"
                    "retention.handles.push_back(std::move(q8_handle));sycl::eventkernel_event;"
                    "retention.submitted=stream;if(!mmvq_submit_q1_nvfp4_aos_id_resident(*stream,ctx.device,")
RETENTION_AFTER = ('&kernel_event)){refuse("thekernelentryrefusedashapethepredicatetook");}'
                   "ggml_sycl::retain_handles_until_event_or_keep(retention.handles,kernel_event);"
                   "retention.submitted=nullptr;ggml_sycl_set_tensor_ready_event(dst,ctx.device,kernel_event,stream);"
                   "ctx.moe_graphs_disabled_once=true;")
OR_KEEP_DECL = "void retain_handles_until_event_or_keep(std::vector<mem_handle> & handles, sycl::event event);"
OR_KEEP_HEAD = "void retain_handles_until_event_or_keep(std::vector<mem_handle> & handles, sycl::event event) {"
PUBLISH_HEAD = "static void publish_handles_until_event(std::vector<mem_handle> & handles, sycl::event event) {"
PUBLISH_PUSH = ("state.queue.push_back({{},std::move(event),std::move(ids),std::move(gates)});"
                "state.queue.back().handles.swap(handles);}")
PUBLISH_GRAPH = "retain_handles_for_current_graph(handles); return;"
GRAPH_HEAD = "void retain_handles_for_current_graph(std::vector<mem_handle> & handles) {"
GRAPH_SINK = ("g_graph_retained_handle_sink->reserve(g_graph_retained_handle_sink->size()+handles.size());"
              "g_graph_retained_handle_sink->insert(g_graph_retained_handle_sink->end(),"
              "std::make_move_iterator(handles.begin()),std::make_move_iterator(handles.end()));handles.clear();return;")
GRAPH_UNWAITABLE = ("retained_store_locklock(state);"
                    "state.graph_unwaitable.reserve(state.graph_unwaitable.size()+handles.size());"
                    "state.graph_unwaitable_ids.reserve(state.graph_unwaitable_ids.size()+ids.size());"
                    "state.graph_unwaitable.insert(state.graph_unwaitable.end(),std::make_move_iterator(handles.begin()),"
                    "std::make_move_iterator(handles.end()));"
                    "state.graph_unwaitable_ids.insert(state.graph_unwaitable_ids.end(),ids.begin(),ids.end());"
                    "handles.clear();")
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
LAUNCH_LIMIT = "if (fit == mmvq_q1_nvfp4_id_fit::LAUNCH_PAST_INT) {"
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
                   "IDS_NOT_RESIDENT", "OPERAND_NOT_ON_DEVICE", "TABLE_INCOMPLETE", "SCRATCH")
ROUTE_LOG = '"[MOE-ROUTE] resident decode tensor='
ROUTE_LOG_GATE = "if (ggml_sycl::ggml_sycl_moe_route_log_enabled()) {"
GEOMETRY_FN = "mmvq_q1_nvfp4_id_fit mmvq_q1_nvfp4_id_geometry_of("
FIT_ENUM = "enum class mmvq_q1_nvfp4_id_fit : uint8_t {"
FIT_VALUES = "FITS,NO_KERNEL,LAUNCH_PAST_INT,"
FIT_ONLY = " != mmvq_q1_nvfp4_id_fit::FITS"
LAUNCH_BOUND = ("out->quantize_items = quantize_items; if (kernel_items > static_cast<size_t>(INT_MAX) || "
                "quantize_items > static_cast<size_t>(INT_MAX)) { return mmvq_q1_nvfp4_id_fit::LAUNCH_PAST_INT; } "
                "return mmvq_q1_nvfp4_id_fit::FITS;")
GEOMETRY_STRUCT = "struct mmvq_q1_nvfp4_id_geometry {"
GEOMETRY_ASK = ("mmvq_q1_nvfp4_id_geometry_of(weight_type, ncols, nrows_per_expert, top_k, n_tokens, ne11, ids_nb0, "
                "ids_nb1, &geometry)")
IDS_VIEW_RULE = ("if (ids_nb0 != static_cast<int64_t>(sizeof(int32_t)) || "
                 "ids_nb1 < static_cast<int64_t>(sk * sizeof(int32_t))) {")
ADMITTED_FN = "bool mmvq_submit_q1_nvfp4_aos_id_admitted("
RESIDENT_FN = "bool mmvq_submit_q1_nvfp4_aos_id_resident("
QUANTIZE_LABEL = '"mulmat.mmvq.id_quantize_q8_1_aos"'
CHECKED_CALL = ("static_cast<int64_t>(geometry.dst_nb2), n_experts, quant_event, event_out);")
FP16_HELPER_HEAD = "static bool mmvq_q1_nvfp4_id_device_has_fp16(int device) {"
FP16_READ = "ggml_sycl_info().devices[device].xmx_caps.supports_fp16_type"
FP16_ASK = "!mmvq_q1_nvfp4_id_device_has_fp16(device)"
PROFILE_GATE = "const sycl::event quant_event = !ggml_sycl_kernel_profile_enabled() ? quantize(q) : [&]() {"
PLAN_OWNER_HEAD = ("static std::shared_ptr<const ggml_sycl::placement_plan> "
                   "ggml_sycl_moe_plan_owner_for_device(int device) {")
KERNEL_EVENT = "sycl::event kernel_event;"
# A host wait is spelled .wait(, ->wait( or sycl::event::wait(: the three spellings are listed apart.
ROUTE_FORBIDDEN = (".wait(", "->wait(", "::wait(", "wait_and_throw", "ids_host", "ggml_sycl_copy_ids_to_host",
                   "build_moe_resolved_batch", "mmid_workspace", "host_task", "GGML_ABORT", "std::exit",
                   "ggml_sycl_ensure_weight_on_device", "malloc_", "sycl::free(", "getenv",
                   "ggml_sycl_moe_decode_direct_decide", "mmvq_submit_q1_nvfp4_aos_id_admitted", "mem_copy",
                   "ggml_sycl_retain_moe_ptr_table_leases_until_event", "ggml_sycl_submit_queue_sync_event",
                   "GGML_LAYOUT_SOA", "return false;", "ggml_sycl_resolve_tensor_ptr",
                   "moe_fusion_ensure_full_local_ptr_table", "ggml_sycl_publish_mmid_canonical_aos_experts",
                   "retain_handles_until_event(", "terminal_retention_ticket", "std::move(retention")
RESIDENT_FORBIDDEN = (".wait(", "->wait(", "::wait(", "wait_and_throw", "ids_host", "unified_alloc", "malloc",
                      "host_task", "GGML_ABORT", "checked_mul(", "aspect::fp16", "get_device()",
                      "ggml_sycl_get_device_id_from_queue", "*event_out")


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


def comment_block(raw, head, width=1600):
    """The squeezed raw text between the last '}' above `head` and `head`: the comment block of a definition that
    follows another one."""
    at = raw.find(head)
    if at < 0:
        return ""
    return raw[max(raw.rfind("}", 0, at) + 1, at - width) : at]


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
                   "op->nb[1] != op->ne[0] * sizeof(float)", *PREDICATE_FIT, MEMO_ASK):
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
        ("bf16", BF16_IF), ("resident", ROUTE_IF), ("mmid_sched", MMID_SCHED), ("decode_direct", DECODE_DIRECT),
        ("ensure_weight", ENSURE_WEIGHT))}
    missing = [name for name, at in marks.items() if at < 0]
    if missing:
        return [f"ggml_sycl_mul_mat_id lacks {', '.join(missing)}"]
    fails = []
    if not marks["bf16"] < marks["resident"] < marks["mmid_sched"] < marks["decode_direct"] < marks["ensure_weight"]:
        fails.append("the route is not after the BF16 route and ahead of MMID_SCHED, yx28 and the weight ensure")
    body = braced_body(sycl, ROUTE_IF, marks["resident"])
    if body is None or tight(ROUTE_IF + body + "}") != CALLER_BLOCK:
        fails.append("the caller is not exactly: on NVFP4 ask the route, and when it takes the node (it has published "
                     "the kernel's event as dst's ready event) mark the guard published and return")
    calls = len(re.findall(rf"\b{ROUTE_FN}\(", sycl))
    if calls != 2:
        fails.append(f"{ROUTE_FN} appears {calls} times; want its definition and the one caller")
    return fails


def route_body(s):
    return braced_body(s["sycl"], ROUTE_FN_HEAD) if defined_once(s["sycl"], ROUTE_FN_HEAD) else None


def claim_d_route_body(s):
    sycl, common = s["sycl"], s["common"]
    body = route_body(s)
    if body is None:
        return [f"{ROUTE_FN} is not defined exactly once"]
    fails = []
    for needed in (ROUTE_ASK, TABLE_CALL, IDS_CALL, SNAPSHOT_CALL, SNAPSHOT_NO_DEDUP, RETAIN_IDS, RETAIN_Q8,
                   RESIDENT_ENTRY, RETAIN_CALL, *OPERAND_CALLS, *SCRATCH_REQUEST):
        if needed not in body:
            fails.append(f"{ROUTE_FN} does not contain {needed}")
    if body.count(RESIDENT_ENTRY) != 1 or len(re.findall(r"\bmmvq_submit_q1_nvfp4_aos_id_resident\(", sycl)) != 1:
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

    # The table: one helper, shared with the BF16 route, publishes on a miss only from this card's own buffer and not
    # again at a storage generation where the publish and its retry already failed.
    helper = braced_body(sycl, ENSURE_HELPER_HEAD) if defined_once(sycl, ENSURE_HELPER_HEAD) else None
    if helper is None:
        fails.append(f"{ENSURE_HELPER} is not defined exactly once")
    else:
        marks = [helper.find(HELPER_EARLY), helper.find(STAMP_READ), helper.find(STAMP_SKIP),
                 helper.find(HELPER_PUBLISH), helper.rfind(HELPER_TABLE_CALL), helper.find(STAMP_STORE)]
        if min(marks) < 0 or marks != sorted(marks):
            fails.append("the table helper does not, in order: return the probe's table unless it missed on this "
                         "card's own buffer, skip a generation whose publish already failed, publish once, probe again "
                         "and stamp the generation when the table still does not resolve")
        if helper.count("ggml_sycl_publish_mmid_canonical_aos_experts(") != 1 or helper.count(HELPER_TABLE_CALL) != 2:
            fails.append("the table helper publishes other than once or probes other than before and after it")
    asks = len(re.findall(rf"\b{ENSURE_HELPER}\(", sycl))
    if asks != 3:
        fails.append(f"{ENSURE_HELPER} appears {asks} times; want its definition, the BF16 route's ask and this one")
    if common.count(STAMP_FIELD) != 1:
        fails.append("the weight does not carry one failed-publish generation per device")

    # The operands: the input and the output only where they resolve to this card's own memory.
    operand = braced_body(sycl, OPERAND_HELPER_HEAD) if defined_once(sycl, OPERAND_HELPER_HEAD) else None
    if operand is None or tight(operand) != OPERAND_HELPER_BODY:
        fails.append(f"{OPERAND_HELPER} does not answer only an on-device pointer, resolved without materializing")

    # The bundle: taken without the identity dedup, which this table's one-lease-per-expert shape does not need.
    snapshot = braced_body(sycl, SNAPSHOT_DEF) if defined_once(sycl, SNAPSHOT_DEF) else None
    if snapshot is None or SNAPSHOT_APPEND not in tight(snapshot) or \
            snapshot.count("ggml_sycl_append_moe_dispatch_handle(") != 1:
        fails.append("the bundle snapshot does not skip the identity dedup when asked for none")
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
    limit_at = body.find(LAUNCH_LIMIT)
    for later in (IDS_CALL, OPERAND_CALLS[0], TABLE_CALL, "unified_allocate_owner("):
        if limit_at >= 0 and not limit_at < body.find(later):
            fails.append(f"the launch limit is checked after {later}")
    if not 0 <= body.find(RECORDING_IF) < limit_at:
        fails.append("the launch limit is not the first question after the recording arm")
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
    for reason in ("IDS_NOT_RESIDENT", "OPERAND_NOT_ON_DEVICE", "TABLE_INCOMPLETE", "SCRATCH"):
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
    if hpp.count(FIT_ENUM) != 1 or tight(braced_body(hpp, FIT_ENUM) or "") != FIT_VALUES:
        fails.append("mmvq.hpp does not declare the geometry's three answers: FITS, NO_KERNEL and LAUNCH_PAST_INT")
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
    # The int bound on both launches is the helper's last question, asked once the sizes are filled in.
    if not tight(geometry).endswith(tight(LAUNCH_BOUND)):
        fails.append("the geometry helper does not end by answering LAUNCH_PAST_INT for a launch past an int, else FITS")
    for name, entry in (("admitted", admitted), ("resident", resident)):
        if GEOMETRY_ASK + FIT_ONLY not in entry:
            fails.append(f"the {name} entry does not ask the geometry helper and take only FITS")
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
    # The event handed back is the checked kernel's: the quantize's event only orders the kernel after it.
    outs = [m.start() for m in re.finditer(r"\bevent_out\b", resident)]
    if len(outs) != 2 or not resident.rstrip().endswith(CHECKED_CALL):
        fails.append("the resident entry hands back an event other than its last act, the checked kernel's")
    if len(re.findall(r"\bquant_event\b", resident)) != 2:
        fails.append("the quantize's event is used other than as the checked kernel's dependency")
    return fails


def claim_i_bad_ids(s):
    note = comment_above(s["sycl_raw"], ROUTE_FN_HEAD)
    fails = []
    for needed in ("zero row", "quiet NaN", "graph succeeds"):
        if needed not in note:
            fails.append(f"the route's head comment does not say {needed!r}")
    return fails


def claim_j_retention(s):
    sycl, mem_cpp, mem_hpp = s["sycl"], s["mem_cpp"], s["mem_hpp"]
    body = route_body(s)
    if body is None or not defined_once(sycl, RETENTION_HEAD):
        return [f"{ROUTE_FN} or {RETENTION} is not defined exactly once"]
    fails = []
    struct = braced_body(sycl, RETENTION_HEAD) or ""
    if RETENTION_DTOR not in tight(struct) or struct.count("wait_and_throw") != 1:
        fails.append("the retention's destructor does not drain the armed stream, and only then, before the handles go")
    for needed in RETENTION_NO_COPY:
        if needed not in struct:
            fails.append(f"the retention can be copied: it lacks {needed}")
    tb = tight(body)
    if tb.count(RETENTION_BEFORE) != 1:
        fails.append("the route does not gather the bundle, the ids' storage and the scratch into the retention and "
                     "arm it right before the entry")
    if tb.count(RETENTION_AFTER) != 1:
        fails.append("the route does not publish the retention's own vector right after the entry, disarm only once "
                     "the publication returned, and then publish the kernel's event as dst's ready event")
    if body.count("retention.submitted") != 2 or body.count("retention.handles") != 4:
        fails.append("the retention is armed, disarmed, filled or published other than in the protocol")
    note = comment_block(s["sycl_raw"], RETENTION_HEAD)
    if not all(word in note for word in ("profiler", "refusal", "publication", "drains")):
        fails.append("no comment at the retention names the exits it drains for: a submit or profiler throw, a "
                     "refusal after the quantize, a publication throw")

    # The publication leaves the caller's vector as it was when it throws.
    if mem_hpp.count(OR_KEEP_DECL) != 1:
        fails.append("mem-handle.hpp does not declare retain_handles_until_event_or_keep on the caller's vector")
    if tight(braced_body(mem_cpp, OR_KEEP_HEAD) or "x") != "publish_handles_until_event(handles,std::move(event));":
        fails.append("retain_handles_until_event_or_keep does not publish the caller's own vector")
    publish = braced_body(mem_cpp, PUBLISH_HEAD) if defined_once(mem_cpp, PUBLISH_HEAD) else None
    if publish is None:
        fails.append("publish_handles_until_event does not take the caller's vector by reference")
    else:
        if PUBLISH_PUSH not in tight(publish):
            fails.append("the record does not enter the store empty and take the handles by a swap under the lock")
        if PUBLISH_GRAPH not in publish or "std::move(handles)" in publish:
            fails.append("the publication moves the handles out before it can no longer throw")
    graph = braced_body(mem_cpp, GRAPH_HEAD) if defined_once(mem_cpp, GRAPH_HEAD) else None
    if graph is None:
        fails.append("the graph-lifetime retention does not take the caller's vector by reference")
    else:
        tg = tight(graph)
        if GRAPH_SINK not in tg or GRAPH_UNWAITABLE not in tg:
            fails.append("the graph-lifetime retention moves a handle before every allocation it needs is made")
        if not 0 <= graph.find("owner_control_ids(handles)") < graph.find("retained_store_lock"):
            fails.append("the graph-lifetime retention reads the owner ids under the store lock")
    return fails


def claim_k_hot_path(s):
    cpp, sycl = s["mmvq_cpp"], s["sycl"]
    fails = []
    helper = braced_body(cpp, FP16_HELPER_HEAD)
    if helper is None or FP16_READ not in helper:
        fails.append("the entries' half-support question does not read the capabilities taken at init")
    for name, head in (("admitted", ADMITTED_FN), ("resident", RESIDENT_FN)):
        entry = function_body(cpp, head) or ""
        if FP16_ASK not in entry or "aspect::fp16" in entry or "get_device()" in entry:
            fails.append(f"the {name} entry asks the device for half support on every node")
    resident = function_body(cpp, RESIDENT_FN) or ""
    gate = resident.find(PROFILE_GATE)
    if gate < 0 or not resident.find(QUANTIZE_LABEL) > gate or "quantize_label.device = device;" not in resident:
        fails.append("the quantize's profiler label is built when the profiler is off, or from the queue")
    owner = braced_body(sycl, PLAN_OWNER_HEAD)
    if owner is None or owner.count("ggml_sycl_cache_plan_owner(") != 1:
        fails.append("the plan owner is fetched from the cache other than once")
    body = route_body(s) or ""
    if body.count("ctx.stream()") != 1:
        fails.append("the route asks ctx.stream() other than once")
    if body.count(KERNEL_EVENT) != 1 or not body.find(KERNEL_EVENT) > body.rfind("return decline("):
        fails.append("the kernel's event is constructed before the route has taken the node")
    head = sycl.find(MUL_MAT_ID_HEAD)
    caller = sycl[sycl.find(BF16_IF, head) : sycl.find(MMID_SCHED, head)] if head >= 0 else ""
    if not caller or "sycl::event" in caller:
        fails.append("the caller constructs an event for every NVFP4 node")
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
    "j": claim_j_retention,
    "k": claim_k_hot_path,
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
    # A launch past an int declined as a geometry: the route's named refusal would never be reached.
    ("a", "sycl", PREDICATE_HEAD, "if (*fit == mmvq_q1_nvfp4_id_fit::NO_KERNEL) {",
     "if (*fit != mmvq_q1_nvfp4_id_fit::FITS) {"),
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
    ("c", "sycl", MUL_MAT_ID_HEAD + "|" + ROUTE_IF, "ready_guard.published = true; return; }", "return; }"),
    ("c", "sycl", MUL_MAT_ID_HEAD + "|" + ROUTE_IF, "ready_guard.published = true;",
     "ggml_sycl_set_tensor_ready_event(dst, ctx.device, ggml_sycl_submit_queue_sync_event(*ctx.stream()), "
     "ctx.stream()); ready_guard.published = true;"),
    ("c", "sycl", MUL_MAT_ID_HEAD, ROUTE_IF,
     f"if ((src0->type == GGML_TYPE_NVFP4 || src0->type == GGML_TYPE_Q1_0) && {ROUTE_FN}(ctx, dst)) {{"),
    ("d", "sycl", ROUTE_FN_HEAD, TABLE_CALL,
     "moe_fusion_ensure_full_local_ptr_table(ctx, src0, moe_cache_layer_id(src0->name), GGML_LAYOUT_AOS)"),
    ("d", "sycl", ENSURE_HELPER_HEAD, f"!{OWN_BUFFER}(src0->buffer, ctx.device)", "false"),
    ("d", "sycl", ENSURE_HELPER_HEAD, STAMP_SKIP, ""),
    ("d", "sycl", ENSURE_HELPER_HEAD, STAMP_STORE, ""),
    ("d", "sycl", ENSURE_HELPER_HEAD, HELPER_PUBLISH, "if (true || ggml_sycl_publish_mmid_canonical_aos_experts(src0, "
     "ctx.device, GGML_LAYOUT_AOS,"),
    ("d", "sycl", ENSURE_HELPER_HEAD, "GGML_LAYOUT_AOS);", "GGML_LAYOUT_SOA);"),
    ("d", "sycl", OPERAND_HELPER_HEAD, "return resolved.on_device ? resolved.ptr : nullptr;", "return resolved.ptr;"),
    ("d", "sycl", ROUTE_FN_HEAD, OPERAND_CALLS[0], "ggml_sycl_resolve_tensor_ptr(src1)"),
    ("d", "sycl", ROUTE_FN_HEAD, OPERAND_CALLS[1], "ggml_sycl_resolve_tensor_ptr(dst)"),
    ("d", "sycl", ROUTE_FN_HEAD, "moe_dispatch_bundle_dedup::NONE", "moe_dispatch_bundle_dedup::BY_IDENTITY"),
    ("d", "sycl", SNAPSHOT_DEF, "if (dedup == moe_dispatch_bundle_dedup::BY_IDENTITY) {", "if (true) {"),
    ("d", "sycl", ROUTE_FN_HEAD, RETAIN_CALL, ""),
    ("d", "sycl", ROUTE_FN_HEAD, RETAIN_Q8, ""),
    ("d", "sycl", ROUTE_FN_HEAD, RETAIN_IDS, ""),
    ("d", "sycl", ROUTE_FN_HEAD, SNAPSHOT_CALL, "(void) ("),
    ("d", "sycl", ROUTE_FN_HEAD, "req.intent.constraints.forbid_vram_zone_spill = true;", ""),
    ("d", "sycl", ROUTE_FN_HEAD, "ggml_sycl::vram_zone_id::SCRATCH;", "ggml_sycl::vram_zone_id::COUNT;"),
    ("d", "sycl", ROUTE_FN_HEAD, RESIDENT_ENTRY, "ctx.stream()->wait(); " + RESIDENT_ENTRY),
    ("d", "sycl", ROUTE_FN_HEAD, IDS_CALL, "ggml_sycl_copy_ids_to_host(ctx, ids), " + IDS_CALL),
    ("e", "sycl", ROUTE_FN_HEAD, RECORDING_THROW, "throw ggml_sycl_fallback_error("),
    ("e", "sycl", ROUTE_FN_HEAD + "|" + RECORDING_IF, GRAPH_FLAG, ""),
    ("e", "sycl", ROUTE_FN_HEAD + "|" + RETAIN_CALL, GRAPH_FLAG, ""),
    ("f", "sycl", ROUTE_FN_HEAD, LAUNCH_LIMIT + " refuse(", LAUNCH_LIMIT + " (void) ("),
    ("f", "sycl", ROUTE_FN_HEAD, REFUSAL_THROW, "GGML_ABORT("),
    # The launch limit asked only after the scratch is taken: a refusal would then follow allocation work.
    ("f", "sycl", ROUTE_FN_HEAD, (LAUNCH_LIMIT + ' refuse("the launch exceeds an int"); }', "ggml_sycl::alloc_request req{};"),
     ("", "ggml_sycl::alloc_request req{}; " + LAUNCH_LIMIT + ' refuse("the launch exceeds an int"); }')),
    ("g", "sycl", DECLINE_FN_HEAD, DECLINE_ONCE, ""),
    ("g", "sycl", DECLINE_FN_HEAD, "!ggml_sycl::ggml_sycl_moe_route_log_enabled() || ", ""),
    ("g", "sycl", ROUTE_FN_HEAD, "return decline(moe_resident_decode_decline::TABLE_INCOMPLETE);", "return false;"),
    ("g", "sycl", ROUTE_FN_HEAD, "return decline(moe_resident_decode_decline::OPERAND_NOT_ON_DEVICE);",
     "return false;"),
    ("g", "sycl", ROUTE_FN_HEAD, DECLINE_LAMBDA, "return false;"),
    ("h", "mmvq_cpp", ADMITTED_FN, GEOMETRY_ASK, "true"),
    ("h", "mmvq_cpp", ADMITTED_FN, FIT_ONLY, " == mmvq_q1_nvfp4_id_fit::NO_KERNEL"),
    ("h", "mmvq_cpp", RESIDENT_FN, FIT_ONLY, " == mmvq_q1_nvfp4_id_fit::NO_KERNEL"),
    ("h", "mmvq_cpp", GEOMETRY_FN, "{ return mmvq_q1_nvfp4_id_fit::LAUNCH_PAST_INT; }", "{ }"),
    ("h", "mmvq_cpp", GEOMETRY_FN, "kernel_items > static_cast<size_t>(INT_MAX) || ", ""),
    ("h", "mmvq_hpp", None, "LAUNCH_PAST_INT,", ""),
    ("h", "mmvq_cpp", GEOMETRY_FN, "ids_nb1 < static_cast<int64_t>(sk * sizeof(int32_t))",
     "ids_nb1 != static_cast<int64_t>(sk * sizeof(int32_t))"),
    ("h", "mmvq_cpp", RESIDENT_FN, QUANTIZE_LABEL, '"mulmat.mmvq.id_aos_nvfp4"'),
    ("h", "mmvq_cpp", RESIDENT_FN, "dst_bytes != geometry.output_bytes", "false"),
    ("h", "mmvq_cpp", RESIDENT_FN, CHECKED_CALL, "static_cast<int64_t>(geometry.dst_nb2), quant_event, event_out);"),
    ("h", "mmvq_cpp", RESIDENT_FN, "return mmvq_submit_q1_nvfp4_aos_id(",
     "q.wait(); return mmvq_submit_q1_nvfp4_aos_id("),
    # The quantize's event handed back as dst's ready event in place of the kernel's.
    ("h", "mmvq_cpp", RESIDENT_FN, CHECKED_CALL,
     "static_cast<int64_t>(geometry.dst_nb2), n_experts, quant_event, nullptr) && ((*event_out = quant_event), true);"),
    # Every exit after the first submit that did not publish: a kernel or profiler throw, a refusal after the
    # quantize. The retention must be armed before the entry, which submits the quantize first.
    ("j", "sycl", ROUTE_FN_HEAD, ARM, ""),
    ("j", "sycl", ROUTE_FN_HEAD, (ARM, RETAIN_CALL), ("", ARM + " " + RETAIN_CALL)),
    ("j", "sycl", ROUTE_FN_HEAD, ARM, ARM + " " + DISARM),
    # A publication that throws: disarmed before it, its handles would go under a running kernel.
    ("j", "sycl", ROUTE_FN_HEAD, RETAIN_CALL + " " + DISARM, DISARM + " " + RETAIN_CALL),
    ("j", "sycl", ROUTE_FN_HEAD, RETAIN_CALL,
     "ggml_sycl::retain_handles_until_event(std::move(retention.handles), kernel_event);"),
    # A taken node that drains: never disarmed, or a wait after the disarm.
    ("j", "sycl", ROUTE_FN_HEAD, DISARM, ""),
    ("j", "sycl", ROUTE_FN_HEAD, DISARM, DISARM + " stream->wait_and_throw();"),
    # An event other than the kernel's published as dst's ready event.
    ("j", "sycl", ROUTE_FN_HEAD, "ggml_sycl_set_tensor_ready_event(dst, ctx.device, kernel_event, stream);",
     "ggml_sycl_set_tensor_ready_event(dst, ctx.device, ggml_sycl_submit_queue_sync_event(*stream), stream);"),
    ("j", "sycl", RETENTION_HEAD, "submitted->wait_and_throw();", ""),
    ("j", "sycl", RETENTION_HEAD, "if (submitted != nullptr) {", "if (true) {"),
    ("j", "sycl", RETENTION_HEAD, RETENTION_NO_COPY[0], ""),
    ("j", "mem_cpp", PUBLISH_HEAD, "state.queue.push_back({ {}, std::move(event)",
     "state.queue.push_back({ std::move(handles), std::move(event)"),
    ("j", "mem_cpp", PUBLISH_HEAD, "state.queue.back().handles.swap(handles);", ""),
    ("j", "mem_cpp", None, PUBLISH_HEAD, PUBLISH_HEAD.replace(" & handles", " handles")),
    ("j", "mem_cpp", GRAPH_HEAD,
     "g_graph_retained_handle_sink->reserve(g_graph_retained_handle_sink->size() + handles.size());", ""),
    ("j", "mem_cpp", GRAPH_HEAD, "state.graph_unwaitable.reserve(state.graph_unwaitable.size() + handles.size());",
     ""),
    ("j", "mem_hpp", None, OR_KEEP_DECL, OR_KEEP_DECL.replace(" & handles", " handles")),
    ("k", "mmvq_cpp", RESIDENT_FN, FP16_ASK, "!q.get_device().has(sycl::aspect::fp16)"),
    ("k", "mmvq_cpp", ADMITTED_FN, FP16_ASK, "!q.get_device().has(sycl::aspect::fp16)"),
    ("k", "mmvq_cpp", FP16_HELPER_HEAD, FP16_READ, "sycl::device().has(sycl::aspect::fp16)"),
    ("k", "sycl", PLAN_OWNER_HEAD, "if (!plan->entries.empty()) { return plan; }",
     "if (!plan->entries.empty()) { return ggml_sycl_cache_plan_owner(cache); }"),
    ("k", "mmvq_cpp", RESIDENT_FN, "!ggml_sycl_kernel_profile_enabled() ? quantize(q) : ", ""),
    ("k", "mmvq_cpp", RESIDENT_FN, "quantize_label.device = device;",
     "quantize_label.device = ggml_sycl_get_device_id_from_queue(q);"),
    ("k", "sycl", ROUTE_FN_HEAD, "kernel_event, stream);", "kernel_event, ctx.stream());"),
    ("k", "sycl", ROUTE_FN_HEAD, (KERNEL_EVENT + " " + ARM, "mmvq_q1_nvfp4_id_geometry geometry;"),
     (ARM, KERNEL_EVENT + " mmvq_q1_nvfp4_id_geometry geometry;")),
    ("k", "sycl", MUL_MAT_ID_HEAD, ROUTE_IF, "sycl::event resident_event; " + ROUTE_IF),
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
    """A tuple `old` is several replacements made in turn, each at or after the region: a statement moved."""
    text = sources[key]
    start = region_start(text, region)
    for one_old, one_new in zip(*((old, new) if isinstance(old, tuple) else ((old,), (new,)))):
        at = text.find(one_old, start) if start >= 0 else -1
        assert at >= 0, f"mutant anchor missing from {FILES[key].name} after {region!r}: {one_old!r}"
        text = text[:at] + one_new + text[at + len(one_old) :]
    mutated = dict(sources)
    mutated[key] = text
    assert CLAIMS[claim](mutated), f"claim ({claim}) stayed green under the mutant {old!r} -> {new!r}"


def test_comment_mutants_turn_claims_red(sources):
    """The comment claims read the raw text: a predicate or route head without its comment fails them."""
    for claim, head in (("a", PREDICATE_HEAD), ("i", ROUTE_FN_HEAD), ("j", RETENTION_HEAD)):
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
    assert sources["mem_cpp"].find("retained_store_lock lock(state);") >= 0
    assert sources["mem_hpp"].find("void retain_handles_until_event(std::vector<mem_handle> handles,") >= 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
