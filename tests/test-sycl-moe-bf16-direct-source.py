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
  (b) supports_op asks that predicate as the MUL_MAT_ID gate's third term, the route asks the same predicate again at
      run time and refuses by name when it says no, and the BF16 grouped prompt arm (Task 24) asks it too, before its
      table, and declines to the route when it says no;
  (c) the route sits in ggml_sycl_mul_mat_id after the src0->extra ensure block and before the MMID_SCHED device
      schedule, the yx28 decode-direct request and ggml_sycl_ensure_weight_on_device(); its block is keyed on the BF16
      expert type, not on the predicate, and is the grouped arm's try (dst's ready event published and a return when
      the arm ran the node), then one call to ggml_sycl_mul_mat_id_bf16_direct() and a return, so no BF16 op reaches
      the routes after it, none of which has a BF16 kernel;
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
  (h) the MMVQ tables stay BF16-free and moe_type_has_immutable_host_recipe (the Q1_0/NVFP4 type predicate, which the
      planner's MMID workspace pools and the host recipe chooser ask) stays Q1_0/NVFP4;
  (i) the BF16 grouped prompt arm, ggml_sycl_mul_mat_id_bf16_grouped() (Task 24), is a separate arm beside the route:
      called only from the route's block, and the NVFP4 batched prompt executor names no BF16. It asks the opt-out
      first (GGML_SYCL_MOE_BF16_GROUPED, read once into a function-local static), and every way it leaves without
      running the node is a counted decline, never a bare `return false`. Its host waits are the executor's alone: the
      graph-local ids D2H snapshot and the ring-slot claim, which a `// llama.cpp-uop2:` comment directly precedes;
      nothing in it or in moe-bf16-grouped.{hpp,cpp} waits. In order: the weightless admission against the planned
      ring for the routed rows unpadded (before the ids, so a node the ring cannot hold pays no ids wait), the ids, the
      plan, the table, the same admission for the packed rows, the reservation, the claim, one batched gather (profiled
      as mulmat.moe.bf16_grouped.gather), one oneDNN bf16 GEMM per expert after the gather's event (op context
      bf16_expert, no f32 down-conversion, a primitive that cannot be created declining), one batched scatter after the
      GEMMs' events (mulmat.moe.bf16_grouped.scatter), the row map joining the record leases, which are held until the
      scatter's event, the slot recorded on it and dst's ready event set to it. Only the scatter writes dst;
  (j) a failure after the arm's first submission lets nothing go that the device may still read: every submission, the
      row map's copy first, is inside one try, and nothing is submitted before it. A submission that declines (a
      kernel entry that refuses, a GEMM the execute could not create) holds the record leases, the row map and the slot
      until a marker submitted behind everything, then declines SUBMIT (dst is still unwritten, so the direct route
      runs). A submission that throws reaches a std::exception handler (after the rethrow clause the catch contract
      requires) that does the same hold first, logs a WARN naming the tensor, the device and the stage, and throws a
      ggml_sycl_fallback_error "BF16 grouped arm: <stage> failed after submission: <what>": it never returns, so a
      thrown submission never falls through to the direct kernel. The hold runs once. The private-testing failpoint
      that throws sits in the try after the gather and before the GEMMs.
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
    "arm_hpp": SYCL_DIR / "moe-bf16-grouped.hpp",
    "arm_cpp": SYCL_DIR / "moe-bf16-grouped.cpp",
}
# Keys whose text keeps its comments (squeezed only), for the claims that pin a comment.
RAW_FILES = {
    "sycl_raw": FILES["sycl"],
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
ARM_FN = "ggml_sycl_mul_mat_id_bf16_grouped"
ARM_FN_HEAD = f"static bool {ARM_FN}(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {{"
ARM_DECLINES_PREDICATE = "returndecline(moe_bf16_grouped_decline::PREDICATE);"
ARM_TRY = f"if ({ARM_FN}(ctx, dst)) {{ ready_guard.published = true; return; }}"
ROUTE_BODY = re.sub(r"\s+", "", ARM_TRY) + f"{ROUTE_FN}(ctx,dst);return;"
ARM_ENABLED = "ggml_sycl_moe_bf16_grouped_enabled()"
ARM_ENABLED_HEAD = f"static bool {ARM_ENABLED} {{"
ARM_ENABLED_PARTS = ("static const bool enabled = [", 'std::getenv("GGML_SYCL_MOE_BF16_GROUPED")', "return enabled;")
ARM_OPT_OUT_ENV = '"GGML_SYCL_MOE_BF16_GROUPED"'
ARM_DECLINE_HEAD = ("static bool ggml_sycl_moe_bf16_grouped_decline(const ggml_tensor * src0, int device, "
                    "moe_bf16_grouped_decline decline) {")
ARM_DECLINE_COUNT = "g_moe_bf16_grouped_declines[reason].fetch_add(1, std::memory_order_relaxed)"
ARM_IDS = "ggml_sycl_get_moe_ids_d2h_cache_entry(ctx, ids, "
ARM_IDS_REFRESH = "ggml_sycl_refresh_moe_ids_cache(ctx, ids, ids_entry)"
ARM_ADMIT = "ggml_sycl::pp_moe_onednn_admit_weightless_scratch(planned_shape, act_bytes, out_bytes)"
# The admission is one lambda asked twice: for the routed rows unpadded before the ids, for the packed rows after the plan.
ARM_ADMIT_LAMBDA = "const auto admit_planes = [&](size_t act_bytes, size_t out_bytes) {"
ARM_ADMIT_DECLINE = "return decline(moe_bf16_grouped_decline::SCRATCH_UNPLANNED); }"
ARM_ADMIT_ROUTED = ("if (!admit_planes(io.n_rows * static_cast<size_t>(k) * sizeof(uint16_t), "
                    f"io.n_rows * static_cast<size_t>(m) * sizeof(float))) {{ {ARM_ADMIT_DECLINE}")
ARM_ADMIT_PACKED = f"if (!admit_planes(act_bytes, out_bytes)) {{ {ARM_ADMIT_DECLINE}"
ARM_PLAN = "ggml_sycl::moe_bf16_grouped_build_plan("
ARM_RESERVE = "cache->reserve_pp_moe_onednn_scratch(planned_weight, planned_act, planned_out, ring_depth)"
ARM_CLAIM = "if (!pp_moe_onednn_claim_scratch_slot(ctx.device, ring_depth, scratch_slot)) {"
ARM_UOP2 = "// llama.cpp-uop2:"
ARM_GEMM = "DnnlGemmWrapper::gemm("
ARM_GEMM_ARGS = ('"bf16_expert"', "GGML_SYCL_DNNL_FPMATH_NO_DOWNCONVERT", "GGML_SYCL_DNNL_CREATE_FAILURE_DECLINE")
ARM_GEMM_RUN = "expert_gemm(i, false, { gather_event })"
ARM_GATHER = "ggml_sycl::moe_bf16_grouped_gather_submit(queue, rows_dev.get(), src1_dev, act, io, gather_deps, &event)"
ARM_SCATTER = "ggml_sycl::moe_bf16_grouped_scatter_submit(queue, rows_dev.get(), out, dst_dev, io, gemm_events, &event)"
ARM_GATHER_LABEL = 'profile_label.name = "mulmat.moe.bf16_grouped.gather";'
ARM_SCATTER_LABEL = 'profile_label.name = "mulmat.moe.bf16_grouped.scatter";'
ARM_ROWS_HELD = "leases.push_back(std::move(rows_dev.handle));"
ARM_RETAIN = "ggml_sycl::retain_handles_until_event(std::move(leases), scatter_event);"
ARM_RECORD = "claim.record(scatter_event);"
ARM_READY = "ggml_sycl_set_tensor_ready_event(dst, ctx.device, scatter_event, stream);"
ARM_ROUTE_LOG = "[MOE-ROUTE] bf16 grouped tensor="
# Claim (j): the hold to a marker, the decline after a submission, the one try and its handler.
ARM_HOLD_HEAD = "const auto hold_until_marker = [&]() {"
ARM_HOLD_BODY = ("if(held){return;}"
                 "constsycl::eventpassed=ggml_sycl_submit_marker<moe_bf16_grouped_abandon_marker>(*stream);"
                 "leases.push_back(std::move(rows_dev.handle));"
                 "ggml_sycl::retain_handles_until_event(std::move(leases),passed);claim.record(passed);held=true;")
ARM_ABANDON_HEAD = "const auto abandon = [&]() {"
ARM_ABANDON_BODY = "hold_until_marker();returndecline(moe_bf16_grouped_decline::SUBMIT);"
ARM_TRY_RE = re.compile(r"\btry\s*\{")
ARM_RETHROW = "catch (const ggml_sycl_fallback_error &) { throw; }"
ARM_HANDLER_HEAD = "catch (const std::exception & e) {"
ARM_HANDLER_HOLD = "hold_until_marker();"
ARM_FAILURE_WARN = ('GGML_LOG_WARN("[MoE] BF16 grouped arm failed after submission: tensor=%s device=%d stage=%s '
                    'error=%s')
ARM_FAILURE_TEXT = ('bf16_grouped_failure = std::string("BF16 grouped arm: ") + stage + " failed after submission: " + '
                    "e.what();")
ARM_FAILURE_THROW = "throw ggml_sycl_fallback_error(bf16_grouped_failure.c_str());"
ARM_STAGE_FIRST = 'const char * stage = "row map copy";'
ARM_STAGES = ('stage = "gather";', 'stage = "expert GEMM";', 'stage = "scatter";')
ARM_FIRST_SUBMIT = "ggml_sycl_safe_memcpy(*stream, rows_dev.get(), plan.rows.data(),"
# What submits work: none of it may come before the try. The marker inside the hold lambda is defined there, not run.
ARM_SUBMITTERS = ("ggml_sycl_safe_memcpy(", "ggml_sycl_profile_submit(", "expert_gemm(i, false,", "_gather_submit(",
                  "_scatter_submit(", ".submit(", "->submit(")
ARM_FAILPOINT = ("ggml_sycl_q1_nvfp4_test_consume_failure("
                 "GGML_SYCL_Q1_NVFP4_TEST_FAILURE_BF16_GROUPED_THROW_AFTER_GATHER)")
# The failpoint that declines: the expert GEMM lambda opens with it, private builds only, on a run and never on the
# query, and it returns no event, as a primitive oneDNN cannot create at execute would.
ARM_GEMM_HEAD = "const auto expert_gemm = [&](size_t i, bool query_only, const std::vector<sycl::event> & deps) {"
ARM_DECLINE_FAILURE = "GGML_SYCL_Q1_NVFP4_TEST_FAILURE_BF16_GROUPED_DECLINE_AFTER_GATHER"
ARM_DECLINE_FAILPOINT = ("#ifdefGGML_SYCL_Q1_NVFP4_ROUTE_TESTING"
                         f"if(!query_only&&ggml_sycl_q1_nvfp4_test_consume_failure({ARM_DECLINE_FAILURE})){{"
                         "returnstd::optional<sycl::event>();}#endif"
                         "constggml_sycl::moe_bf16_grouped_expert&expert=plan.experts[i];")
ARM_FORBIDDEN = (".wait(", "->wait(", "::wait(", "wait_and_throw", "getenv", "GGML_ABORT", "malloc_", "sycl::free(",
                 "return false;", "ggml_sycl_ensure_weight_on_device", "ids_host", "ggml_sycl_copy_ids_to_host")
EXECUTOR_HEAD = "auto try_pp_onednn_f16_batched = [&]() -> bool {"
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
NEEDS_DIRECT_HEAD = "inline bool moe_type_has_immutable_host_recipe(ggml_type type) {"
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
    for key, path in RAW_FILES.items():
        out[key] = squeeze(path.read_text(encoding="utf-8")) if path.exists() else ""
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


def braced_end(text, head, start=0):
    """The offset just past the '}' that closes the '{' ending `head` (its first occurrence at or after `start`), or
    -1."""
    at = text.find(head, start)
    body = braced_body(text, head, start) if at >= 0 else None
    return -1 if body is None else at + len(head) + len(body) + 1


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
    if calls != 4:
        fails.append(f"{PREDICATE} appears {calls} times; want its definition, the gate, the route and the grouped arm")
    route = route_fn_body(sycl) or ""
    arm = arm_fn_body(sycl) or ""
    if sycl.count(ROUTE_ASK) != 2 or route.count(ROUTE_ASK) != 1 or arm.count(ROUTE_ASK) != 1:
        fails.append(f"{ROUTE_FN} and {ARM_FN} do not each ask the predicate exactly once at run time")
    else:
        declined = braced_body(route, ROUTE_ASK) or ""
        if not tight(declined).startswith("refuse("):
            fails.append("a run-time `no` from the predicate is not a named refusal")
        if route.find(ROUTE_ASK) > route.find(TABLE_CALL) >= 0:
            fails.append("the run-time predicate is asked after the table is built")
        # The arm's `no` leaves the node to the route, which asks again and refuses by name.
        if tight(braced_body(arm, ROUTE_ASK) or "") != ARM_DECLINES_PREDICATE:
            fails.append("a run-time `no` from the predicate is not the grouped arm's counted PREDICATE decline")
        if arm.find(ROUTE_ASK) > arm.find(TABLE_CALL) >= 0 or arm.find(TABLE_CALL) < 0:
            fails.append("the grouped arm asks the predicate after its table, or builds no table")
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
        fails.append(f"the route block is not exactly `if ({ARM_FN}(ctx, dst)) {{ ready_guard.published = true; "
                     f"return; }} {ROUTE_FN}(ctx, dst); return;`")
    return fails


def route_fn_body(sycl):
    if sycl.count(ROUTE_FN_HEAD) != 1:
        return None
    return braced_body(sycl, ROUTE_FN_HEAD)


def arm_fn_body(sycl):
    if sycl.count(ARM_FN_HEAD) != 1:
        return None
    return braced_body(sycl, ARM_FN_HEAD)


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
    call = sycl.find("ggml_sycl_mul_mat_id_with_direct_reentry(ctx, dst);", head)
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
    body = braced_body(s["tables"], NEEDS_DIRECT_HEAD)
    if body is None or tight(body) != NEEDS_DIRECT_BODY:
        fails.append("moe_type_has_immutable_host_recipe is no longer exactly Q1_0/NVFP4")
    return fails


def claim_i_grouped_arm(s):
    sycl = s["sycl"]
    arm = arm_fn_body(sycl)
    if arm is None:
        return [f"{ARM_FN} is not defined exactly once"]
    fails = []
    # A separate arm: asked from the route's block alone, and the NVFP4 batched prompt executor names no BF16.
    calls = len(re.findall(rf"\b{ARM_FN}\(", sycl))
    if calls != 2:
        fails.append(f"{ARM_FN} appears {calls} times; want its definition and the route block's try")
    executor = braced_body(sycl, EXECUTOR_HEAD)
    if executor is None:
        fails.append("the NVFP4 batched prompt executor (try_pp_onednn_f16_batched) was not found")
    elif "bf16" in executor.lower():
        fails.append("the NVFP4 batched prompt executor names BF16: the grouped arm is meant to stand beside it")
    # The opt-out, read once, asked first.
    enabled = braced_body(sycl, ARM_ENABLED_HEAD) if sycl.count(ARM_ENABLED_HEAD) == 1 else None
    if enabled is None or any(part not in enabled for part in ARM_ENABLED_PARTS):
        fails.append(f"{ARM_ENABLED} does not read GGML_SYCL_MOE_BF16_GROUPED once into a function-local static")
    if sycl.count(ARM_OPT_OUT_ENV) != 1:
        fails.append("GGML_SYCL_MOE_BF16_GROUPED is read somewhere other than the arm's one helper")
    if arm.count(ARM_ENABLED) != 1 or not tight(arm).startswith(
            "constggml_tensor*src0=dst->src[0];constggml_tensor*src1=dst->src[1];constggml_tensor*ids=dst->src[2];"
            "constautodecline=[&](moe_bf16_grouped_declinereason){returnggml_sycl_moe_bf16_grouped_decline(src0,"
            f"ctx.device,reason);}};if(!{ARM_ENABLED}){{returndecline(moe_bf16_grouped_decline::OPT_OUT);}}"):
        fails.append("the grouped arm does not ask the opt-out first, as a counted decline")
    # Every way out short of running the node is a counted decline.
    decline = braced_body(sycl, ARM_DECLINE_HEAD) if sycl.count(ARM_DECLINE_HEAD) == 1 else None
    if decline is None or ARM_DECLINE_COUNT not in decline or not tight(decline).endswith("returnfalse;"):
        fails.append("the arm's decline helper does not count the reason and return false")
    for needed in (ARM_IDS, ARM_IDS_REFRESH, ARM_PLAN, TABLE_CALL, ARM_ADMIT, ARM_ADMIT_LAMBDA, ARM_ADMIT_ROUTED,
                   ARM_ADMIT_PACKED, ARM_RESERVE, ARM_CLAIM, ARM_GEMM, *ARM_GEMM_ARGS, ARM_GEMM_RUN, ARM_GATHER,
                   ARM_SCATTER, ARM_GATHER_LABEL, ARM_SCATTER_LABEL, ARM_ROWS_HELD, ARM_RETAIN, ARM_RECORD, ARM_READY,
                   ARM_ROUTE_LOG, *OPERAND_CALLS, GRAPH_FLAG):
        if needed not in arm:
            fails.append(f"{ARM_FN} does not contain {needed}")
    for once in (ARM_IDS, ARM_IDS_REFRESH, ARM_PLAN, ARM_ADMIT, ARM_ADMIT_LAMBDA, ARM_ADMIT_ROUTED, ARM_ADMIT_PACKED, ARM_RESERVE,
                 ARM_CLAIM, ARM_GEMM, ARM_GATHER, ARM_SCATTER, ARM_RETAIN, ARM_RECORD, ARM_READY,
                 "ggml_sycl::moe_bf16_grouped_gather_submit(", "ggml_sycl::moe_bf16_grouped_scatter_submit("):
        if arm.count(once) > 1:
            fails.append(f"{ARM_FN} contains {once} more than once")
    # The admission asked twice through its one lambda, which answers what the weightless admission answers.
    if arm.count("admit_planes(") != 2:
        fails.append("the arm asks its admission lambda other than twice (the routed rows, then the packed rows)")
    admit = braced_body(arm, ARM_ADMIT_LAMBDA)
    if admit is None or ARM_ADMIT not in admit or not tight(admit).endswith("returnadmission.allowed;"):
        fails.append("the admission lambda does not return the weightless admission's answer")
    # The row map joins the leases twice: in the hold to a marker (claim (j)) and on the path that ran the node.
    if arm.count(ARM_ROWS_HELD) != 2:
        fails.append("the row map joins the leases other than in the hold and on the path that ran the node")
    lowered = arm.lower()
    for forbidden in ARM_FORBIDDEN:
        if forbidden.lower() in lowered:
            fails.append(f"{ARM_FN} contains {forbidden!r}")
    # dst is written by the scatter alone: resolved, null-checked, handed to the scatter.
    if arm.count("dst_dev") != 3:
        fails.append("dst's device pointer is used somewhere other than its resolution, its check and the scatter")
    # The row map held on the run path is the last of its two spellings.
    order = [arm.find(text) for text in (ARM_ENABLED, ROUTE_ASK, ARM_ADMIT_LAMBDA, ARM_ADMIT_ROUTED, ARM_IDS,
                                         ARM_IDS_REFRESH, ARM_PLAN, TABLE_CALL, ARM_ADMIT_PACKED, ARM_RESERVE, ARM_CLAIM,
                                         ARM_GATHER_LABEL, ARM_GATHER, ARM_GEMM_RUN, ARM_SCATTER_LABEL, ARM_SCATTER)]
    order += [arm.rfind(ARM_ROWS_HELD)] + [arm.find(text) for text in (ARM_RETAIN, ARM_RECORD, ARM_READY)]
    if min(order) >= 0 and order != sorted(order):
        fails.append("the arm does not run opt-out, predicate, admission of the routed rows, ids, plan, table, "
                     "admission of the packed rows, reservation, claim, gather, GEMMs, scatter, then hold the row map "
                     "and leases to the scatter's event, record the slot on it and publish it as dst's ready event, in "
                     "that order")
    # The claim is the executor's wait, named for the task that removes it, by a comment directly before it.
    raw = s["sycl_raw"]
    head = raw.find(ARM_FN_HEAD)
    claim_at = raw.find(ARM_CLAIM, head) if head >= 0 else -1
    uop2_at = raw.rfind(ARM_UOP2, head, claim_at) if claim_at >= 0 else -1
    if uop2_at < 0 or ";" in raw[uop2_at:claim_at]:
        fails.append(f"the arm's ring-slot claim is not directly preceded by a `{ARM_UOP2}` comment")
    # The arm's kernels: one submit each, ordered by their deps, allocating, freeing and waiting on nothing.
    arm_hpp, arm_cpp = s["arm_hpp"], s["arm_cpp"]
    if not arm_hpp or not arm_cpp:
        fails.append("moe-bf16-grouped.hpp / moe-bf16-grouped.cpp do not exist")
    else:
        if arm_cpp.count("q.submit(") != 2 or arm_cpp.count("h.depends_on(deps);") != 2:
            fails.append("moe-bf16-grouped.cpp does not submit the gather and the scatter once each after their deps")
        lowered = (arm_hpp + arm_cpp).lower()
        for forbidden in KERNEL_FORBIDDEN:
            if forbidden.lower() in lowered:
                fails.append(f"moe-bf16-grouped carries {forbidden!r}")
    return fails


def claim_j_failure_after_submission(s):
    sycl = s["sycl"]
    arm = arm_fn_body(sycl)
    if arm is None:
        return [f"{ARM_FN} is not defined exactly once"]
    fails = []
    # The hold: once, a marker behind everything submitted, the row map joining the record leases, both held until
    # the marker, the slot recorded on it.
    hold = braced_body(arm, ARM_HOLD_HEAD) if arm.count(ARM_HOLD_HEAD) == 1 else None
    if hold is None or tight(hold) != ARM_HOLD_BODY:
        fails.append("the hold is not exactly: once, a marker, the row map joining the leases, both held until the "
                     "marker and the slot recorded on it")
    abandon = braced_body(arm, ARM_ABANDON_HEAD) if arm.count(ARM_ABANDON_HEAD) == 1 else None
    if abandon is None or tight(abandon) != ARM_ABANDON_BODY:
        fails.append("a submission that declines does not hold to the marker and then decline SUBMIT")
    # One try, holding every submission, the row map's copy first; nothing submitted before it.
    tries = list(ARM_TRY_RE.finditer(arm))
    if len(tries) != 1:
        return fails + [f"the arm has {len(tries)} try blocks; want the one around its submissions"]
    try_head = arm[tries[0].start() : tries[0].end()]
    body = braced_body(arm, try_head, tries[0].start())
    end = braced_end(arm, try_head, tries[0].start())
    if body is None or end < 0:
        return fails + ["the arm's try block does not close"]
    before = arm[: tries[0].start()]
    for submitter in ARM_SUBMITTERS:
        if submitter in before:
            fails.append(f"the arm submits work ({submitter!r}) before its try")
    if not tight(body).startswith(tight("gather_deps.push_back(" + ARM_FIRST_SUBMIT)):
        fails.append("the try does not begin with the arm's first submission, the row map's copy")
    for needed in (ARM_GATHER, ARM_GEMM_RUN, ARM_SCATTER, *ARM_STAGES):
        if needed not in body:
            fails.append(f"the try does not hold {needed}")
    if ARM_STAGE_FIRST not in before:
        fails.append("the stage the handler names is not the row map's copy until the gather")
    stages = [body.find(text) for text in (ARM_STAGES[0], ARM_GATHER, ARM_STAGES[1], ARM_GEMM_RUN, ARM_STAGES[2],
                                           ARM_SCATTER)]
    if min(stages) >= 0 and stages != sorted(stages):
        fails.append("the stage the handler names is not set before the gather, the GEMMs and the scatter")
    if body.count("return abandon();") != 3 or "return decline(" in body or "return true;" in body:
        fails.append("a submission inside the try leaves other than by the hold and a SUBMIT decline")
    # The failpoint that throws: once, inside the try, after the gather and before the GEMMs.
    point = body.find(ARM_FAILPOINT)
    if sycl.count(ARM_FAILPOINT) != 1 or not (body.find(ARM_GATHER) < point < body.find(ARM_GEMM_RUN)):
        fails.append("the private-testing failpoint is not the one in the try after the gather and before the GEMMs")
    # The failpoint that declines: once, opening the expert GEMM lambda, so a run after the gather reaches the
    # `return abandon();` a oneDNN decline reaches.
    gemm = braced_body(arm, ARM_GEMM_HEAD) if arm.count(ARM_GEMM_HEAD) == 1 else None
    if gemm is None or not tight(gemm).startswith(ARM_DECLINE_FAILPOINT) or sycl.count(ARM_DECLINE_FAILURE) != 1:
        fails.append("the expert GEMM lambda does not open with the private-testing failpoint that declines a run, "
                     "never the query, by returning no event")
    # The handlers: the rethrow clause the catch contract requires, then the std::exception handler: the hold first,
    # the WARN naming tensor, device and stage, the named fallback error thrown, and no way back to the direct route.
    after = arm[end:]
    if not tight(after).startswith(tight(ARM_RETHROW + ARM_HANDLER_HEAD)):
        return fails + ["the try is not followed by exactly the fallback-error rethrow and a std::exception handler"]
    if arm.count("catch (") != 2:
        fails.append("the arm catches other than the rethrow clause and its one std::exception handler")
    handler = braced_body(arm, ARM_HANDLER_HEAD, end)
    if handler is None:
        return fails + ["the std::exception handler does not close"]
    if not tight(handler).startswith(tight(ARM_HANDLER_HOLD)):
        fails.append("the handler does not hold to a marker before anything else")
    for needed in (ARM_FAILURE_WARN, "src0->name, ctx.device, stage, e.what()", "static thread_local std::string "
                   "bf16_grouped_failure;", ARM_FAILURE_TEXT):
        if needed not in handler:
            fails.append(f"the handler does not contain {needed}")
    if not tight(handler).endswith(tight(ARM_FAILURE_THROW)):
        fails.append("the handler does not end by throwing the named ggml_sycl_fallback_error")
    if "return" in handler or arm.count("throw ggml_sycl_fallback_error(") != 1:
        fails.append("a thrown submission can leave the arm other than by the named fallback error")
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
    "i": claim_i_grouped_arm,
    "j": claim_j_failure_after_submission,
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
    ("c", "sycl", None, f"{ROUTE_IF} {ARM_TRY} {ROUTE_FN}(ctx, dst); return; }}", ""),
    ("c", "sycl", ROUTE_IF, ROUTE_FN + "(ctx, dst); return; }", ROUTE_FN + "(ctx, dst); }"),
    # The arm never asked; the arm's node left without dst's ready event; the route run again after the arm ran.
    ("c", "sycl", ROUTE_IF, ARM_TRY + " ", ""),
    ("c", "sycl", ROUTE_IF, "ready_guard.published = true; return; }", "return; }"),
    ("c", "sycl", ROUTE_IF, "ready_guard.published = true; return; }", "ready_guard.published = true; }"),
    # The fall-through this shape replaced: a block keyed on the predicate lets a run-time `no` reach MMID_SCHED.
    ("c", "sycl", None, ROUTE_IF, f"if ({PREDICATE}(dst, ctx.device)) {{"),
    ("b", "sycl", None, ROUTE_IF, f"if (src0->type == GGML_TYPE_BF16 && {PREDICATE}(dst, ctx.device)) {{"),
    ("b", "sycl", ROUTE_FN_HEAD, ROUTE_ASK, "if (false) {"),
    ("b", "sycl", ROUTE_ASK, "refuse(", "(void) ("),
    ("b", "sycl", ARM_FN_HEAD, ROUTE_ASK, "if (false) {"),
    ("b", "sycl", f"{ARM_FN_HEAD}|{ROUTE_ASK}", "return decline(moe_bf16_grouped_decline::PREDICATE);", "(void) 0;"),
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
    ("h", "tables", None, "return type == GGML_TYPE_Q1_0 || type == GGML_TYPE_NVFP4;",
     "return type == GGML_TYPE_Q1_0 || type == GGML_TYPE_NVFP4 || type == GGML_TYPE_BF16;"),
    # The grouped arm asked from a second site; the NVFP4 executor taking BF16 itself.
    ("i", "sycl", None, ROUTE_FN_HEAD, f"{ROUTE_FN_HEAD} (void) {ARM_FN}(ctx, dst);"),
    ("i", "sycl", None, EXECUTOR_HEAD, f"{EXECUTOR_HEAD} if (src0->type == GGML_TYPE_BF16) {{ return false; }}"),
    # The opt-out read per call, read a second time, asked late, or leaving uncounted.
    ("i", "sycl", ARM_ENABLED_HEAD, "static const bool enabled = [", "const bool enabled = ["),
    ("i", "sycl", ARM_FN_HEAD, "const int64_t n_tokens",
     'if (getenv("GGML_SYCL_MOE_BF16_GROUPED")) { return decline(moe_bf16_grouped_decline::OPT_OUT); } '
     "const int64_t n_tokens"),
    ("i", "sycl", ARM_FN_HEAD, f"if (!{ARM_ENABLED}) {{ return decline(moe_bf16_grouped_decline::OPT_OUT); }}", ""),
    ("i", "sycl", ARM_FN_HEAD, "return decline(moe_bf16_grouped_decline::OPT_OUT);", "return false;"),
    ("i", "sycl", ARM_DECLINE_HEAD, ARM_DECLINE_COUNT, "0"),
    # A host wait in the chain, in the arm or in its kernels.
    ("i", "sycl", ARM_FN_HEAD, "bool scattered = false;", "stream->wait(); bool scattered = false;"),
    ("i", "sycl", ARM_FN_HEAD, "gemm_events.push_back(*gemm_event);",
     "gemm_event->wait(); gemm_events.push_back(*gemm_event);"),
    ("i", "arm_cpp", None, "return true;", "event->wait(); return true;"),
    ("i", "arm_cpp", None, "h.depends_on(deps);", ""),
    # The chain's edges: the GEMMs not after the gather, the scatter not after the GEMMs.
    ("i", "sycl", ARM_FN_HEAD, ARM_GEMM_RUN, "expert_gemm(i, false, {})"),
    ("i", "sycl", ARM_FN_HEAD, "io, gemm_events, &event)", "io, {}, &event)"),
    # A second gather or scatter, or the scatter's dst handed to the gather.
    ("i", "sycl", ARM_FN_HEAD, "gathered = ggml_sycl::moe_bf16_grouped_gather_submit(",
     "gathered = ggml_sycl::moe_bf16_grouped_gather_submit(queue, rows_dev.get(), src1_dev, act, io, {}, &event) && "
     "ggml_sycl::moe_bf16_grouped_gather_submit("),
    ("i", "sycl", ARM_FN_HEAD, "src1_dev, act, io, gather_deps", "src1_dev, reinterpret_cast<uint16_t *>(dst_dev), "
     "io, gather_deps"),
    # The leases, the row map and the slot let go before the scatter's event; dst's ready event left unset.
    ("i", "sycl", ARM_FN_HEAD, ARM_RETAIN, "(void) leases;"),
    ("i", "sycl", f"{ARM_FN_HEAD}|{ARM_FAILURE_THROW}", ARM_ROWS_HELD, ""),
    ("i", "sycl", ARM_FN_HEAD, ARM_RECORD, ""),
    ("i", "sycl", ARM_FN_HEAD, ARM_READY, ""),
    ("i", "sycl", ARM_FN_HEAD, ARM_RETAIN, "ggml_sycl::retain_handles_until_event(std::move(leases), gather_event);"),
    # The ring asked out of order, or with the weight-staging admission.
    ("i", "sycl", ARM_FN_HEAD, ARM_ADMIT, "ggml_sycl::pp_moe_onednn_admit_scratch(planned_shape, planned_shape)"),
    ("i", "sycl", ARM_FN_HEAD, "pp_moe_onednn_scratch_claim claim;",
     f"pp_moe_onednn_scratch_claim claim; {ARM_RESERVE};"),
    # The GEMM's contract: its profiler context, strict f32, and a primitive that cannot be created declining.
    ("i", "sycl", ARM_FN_HEAD, '"bf16_expert"', '"onednn_gemm"'),
    ("i", "sycl", ARM_FN_HEAD, "GGML_SYCL_DNNL_FPMATH_NO_DOWNCONVERT", "GGML_SYCL_DNNL_FPMATH_DEFAULT"),
    ("i", "sycl", ARM_FN_HEAD, "GGML_SYCL_DNNL_CREATE_FAILURE_DECLINE", "GGML_SYCL_DNNL_CREATE_FAILURE_ABORT"),
    ("i", "sycl", ARM_FN_HEAD, ARM_GATHER_LABEL, 'profile_label.name = "mulmat.moe.bf16_grouped";'),
    # The claim's wait unnamed, or a statement between the comment and the claim.
    ("i", "sycl_raw", ARM_FN_HEAD, ARM_UOP2, "//"),
    ("i", "sycl_raw", f"{ARM_FN_HEAD}|{ARM_UOP2}", ARM_CLAIM, f"(void) 0; {ARM_CLAIM}"),
    # The admission: asked after the ids only, its routed-rows answer ignored, or the lambda answering yes.
    ("i", "sycl", ARM_FN_HEAD, ARM_ADMIT_ROUTED, ""),
    ("i", "sycl", f"{ARM_FN_HEAD}|if (!admit_planes(io.n_rows", ARM_ADMIT_DECLINE, "(void) 0; }"),
    ("i", "sycl", ARM_FN_HEAD, ARM_ADMIT_ROUTED, f"(void) 0; {ARM_IDS_REFRESH}; {ARM_ADMIT_ROUTED}"),
    ("i", "sycl", ARM_ADMIT_LAMBDA, "return admission.allowed;", "return true;"),
    ("i", "sycl", ARM_FN_HEAD, ARM_ADMIT_PACKED, ""),
    # The hold: run twice, without the marker, letting the row map, the leases or the slot go before it.
    ("j", "sycl", ARM_HOLD_HEAD, "if (held) { return; }", ""),
    ("j", "sycl", ARM_HOLD_HEAD, "held = true;", ""),
    ("j", "sycl", ARM_HOLD_HEAD, "ggml_sycl_submit_marker<moe_bf16_grouped_abandon_marker>(*stream)", "sycl::event()"),
    ("j", "sycl", ARM_HOLD_HEAD, ARM_ROWS_HELD, ""),
    ("j", "sycl", ARM_HOLD_HEAD, "ggml_sycl::retain_handles_until_event(std::move(leases), passed);", "(void) leases;"),
    ("j", "sycl", ARM_HOLD_HEAD, "claim.record(passed);", ""),
    # A submission that declines without the hold, or not as SUBMIT.
    ("j", "sycl", ARM_ABANDON_HEAD, "hold_until_marker();", ""),
    ("j", "sycl", ARM_ABANDON_HEAD, "moe_bf16_grouped_decline::SUBMIT", "moe_bf16_grouped_decline::PRIMITIVE"),
    ("j", "sycl", f"{ARM_FN_HEAD}|if (!scattered) {{", "return abandon();", "return false;"),
    # No try; work submitted before it; the row map's copy outside it.
    ("j", "sycl", ARM_FN_HEAD, "try { gather_deps.push_back(", "{ gather_deps.push_back("),
    ("j", "sycl", ARM_FN_HEAD, "sycl::event scatter_event; try {",
     "sycl::event scatter_event; ggml_sycl_safe_memcpy(*stream, rows_dev.get(), plan.rows.data(), 0, {}); try {"),
    ("j", "sycl", ARM_FN_HEAD, "try { gather_deps.push_back(", "try { (void) 0; gather_deps.push_back("),
    # The stage the handler names left stale.
    ("j", "sycl", ARM_FN_HEAD, 'stage = "expert GEMM";', ""),
    ("j", "sycl", ARM_FN_HEAD, 'stage = "scatter";', ""),
    # The failpoint before the gather, or gone.
    ("j", "sycl", ARM_FN_HEAD, 'stage = "gather";', f'stage = "gather"; if ({ARM_FAILPOINT}) {{ throw 1; }}'),
    ("j", "sycl", ARM_FN_HEAD, f"if ({ARM_FAILPOINT})", "if (false)"),
    # The failpoint that declines: failing the query too, returning an event, consumed without declining, unguarded,
    # or gone.
    ("j", "sycl", ARM_GEMM_HEAD, "if (!query_only && ", "if ("),
    ("j", "sycl", ARM_GEMM_HEAD, "return std::optional<sycl::event>();",
     "return std::optional<sycl::event>(sycl::event());"),
    ("j", "sycl", ARM_GEMM_HEAD, "return std::optional<sycl::event>(); } # endif", "(void) 0; } # endif"),
    ("j", "sycl", ARM_GEMM_HEAD, "# ifdef GGML_SYCL_Q1_NVFP4_ROUTE_TESTING if (!query_only", "if (!query_only"),
    ("j", "sycl", ARM_GEMM_HEAD, ARM_DECLINE_FAILURE, "GGML_SYCL_Q1_NVFP4_TEST_FAILURE_NONE"),
    # The handler: missing, catching everything, without the hold, falling through to the direct route, rethrowing
    # the raw error, or throwing an error that names no stage.
    ("j", "sycl", ARM_FN_HEAD, ARM_RETHROW + " ", ""),
    ("j", "sycl", ARM_FN_HEAD, ARM_HANDLER_HEAD, "catch (...) {"),
    ("j", "sycl", ARM_FN_HEAD, f"{ARM_HANDLER_HEAD} {ARM_HANDLER_HOLD}", ARM_HANDLER_HEAD),
    ("j", "sycl", ARM_FN_HEAD, ARM_FAILURE_THROW, "return decline(moe_bf16_grouped_decline::SUBMIT);"),
    ("j", "sycl", ARM_FN_HEAD, ARM_FAILURE_THROW, "throw;"),
    ("j", "sycl", ARM_FN_HEAD, '+ stage + " failed after submission: "', '+ " failed: "'),
    ("j", "sycl", ARM_FN_HEAD, "src0->name, ctx.device, stage, e.what()", "src0->name, ctx.device, \"\", e.what()"),
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
    assert braced_body(sources["tables"], NEEDS_DIRECT_HEAD) is not None
    assert sources["tables"]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
