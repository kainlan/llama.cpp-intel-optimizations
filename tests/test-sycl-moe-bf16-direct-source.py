#!/usr/bin/env python3
"""Source contract for the BF16 expert device route (Task 8g of the Inkling 2 Mini plan).

A MUL_MAT_ID whose experts are BF16 runs on the device whose plan holds every expert at AOS, from the cached full-local
expert table and the device ids, with src1 and dst read and written in place. These claims pin that shape:

  (a) one predicate, ggml_sycl_moe_bf16_direct_route_available(op, device), defined once in ggml-sycl.cpp: a known
      plan answers alone (every expert of the weight planned on this device at AOS, asked through
      ggml_sycl_moe_tensor_plan_primary_layout), and with no plan a weight in a SYCL device buffer of this device
      counts; it reads no environment and has no row-multiple rule;
  (b) supports_op asks that predicate as the MUL_MAT_ID gate's third term, and the route asks the same predicate;
  (c) the route sits in ggml_sycl_mul_mat_id after the src0->extra ensure block and before the MMID_SCHED device
      schedule, the yx28 decode-direct request and ggml_sycl_ensure_weight_on_device(); its block is one call to
      ggml_sycl_mul_mat_id_bf16_direct() and a return, so nothing after it runs for a BF16 op the predicate takes;
  (d) ggml_sycl_mul_mat_id_bf16_direct() takes the AOS full-local table, the device-resident ids, submits through
      moe_bf16_direct_submit() and retains the table's leases until the kernel's event; it never reads the ids on the
      host, waits, touches the MMID workspace or Q8_1, re-routes to the CPU or aborts; a refusal is a named WARN line
      and ggml_sycl_fallback_error;
  (e) under graph recording it does what the NVFP4 decode node does: a WARN line and a sycl::exception before any
      table or ids work, and on a direct run it sets moe_graphs_disabled_once like every other MoE route (its consumer
      selects segmented MoE replay, it does not turn replay off); check_graph_compatibility keeps a BF16 MUL_MAT_ID
      out of whole-graph capture, so it is a segment boundary like NVFP4;
  (f) the op-entry flush of another node's pending host scatter precedes the MUL_MAT_ID dispatch in
      ggml_sycl_compute_forward_impl, so the route reads src1 after it;
  (g) the kernel lives in moe-bf16-direct.{hpp,cpp}: BF16 decoded with bf16_to_f32, a 16-lane sub-group per output
      row reduced with reduce_over_group, id -1 a zero row and any other bad slot a quiet NaN; the file allocates,
      frees and waits on nothing;
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
ROUTE_IF = f"if ({PREDICATE}(dst, ctx.device)) {{"
ROUTE_BODY = f"{ROUTE_FN}(ctx,dst);return;"
GATE = ("if (!moe_mmvq_admission_supports_type(indexed_a_type) && !ggml_sycl_moe_pp_batched_admits_op(op, device) && "
        f"!{PREDICATE}(op, device)) {{")
MUL_MAT_ID_HEAD = "static void ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx, ggml_tensor * dst) try {"
EXTRA_ENSURED = "s_mul_mat_id_extras.push_back(std::move(new_extra));"
MMID_SCHED = "(g_ggml_sycl_mmid_sched & GGML_SYCL_MMID_SCHED_DEVICE)"
DECODE_DIRECT = "ggml_sycl::moe_decode_direct_request direct_request;"
ENSURE_WEIGHT = "ggml_sycl_ensure_weight_on_device(src0, ctx.device);"
TABLE_CALL = "moe_fusion_ensure_full_local_ptr_table(ctx, src0, moe_cache_layer_id(src0->name), GGML_LAYOUT_AOS)"
IDS_CALL = "ggml_sycl_moe_ids_device_resident_ptr(ctx, ids"
SUBMIT_CALL = "ggml_sycl::moe_bf16_direct_submit("
RETAIN_CALL = "ggml_sycl_retain_moe_ptr_table_leases_until_event("
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
ROUTE_FORBIDDEN = (".wait(", "wait_and_throw", "ids_host", "ggml_sycl_copy_ids_to_host", "mmid_workspace", "q8_1",
                   "GGML_ABORT", "ggml_sycl_ensure_weight_on_device", "malloc_", "sycl::free(", "getenv", "cpu")
KERNEL_FORBIDDEN = (".wait(", "wait_and_throw", "malloc_", "sycl::free(", "unified_alloc", "q8_1", "GGML_ABORT",
                    "getenv")


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


def claim_a_predicate(s):
    fails = []
    sycl = s["sycl"]
    if sycl.count(PREDICATE_HEAD) != 1 or len(re.findall(rf"\b(?:static )?bool {PREDICATE}\(", sycl)) != 1:
        return [f"{PREDICATE}(op, device) is not defined exactly once as a static bool in ggml-sycl.cpp"]
    body = braced_body(sycl, PREDICATE_HEAD) or ""
    for needed in ("GGML_TYPE_BF16", "ggml_is_contiguous(", "ggml_sycl_moe_plan_owner_for_device(device)",
                   "ggml_sycl_moe_tensor_plan_primary_layout(", "GGML_LAYOUT_AOS", "ggml_backend_buffer_is_sycl("):
        if needed not in body:
            fails.append(f"the predicate does not ask {needed}")
    for forbidden in ("getenv", "% 32", "%32", "WARP_SIZE", "ggml_sycl_moe_pp_batched", "GGML_TYPE_NVFP4"):
        if forbidden in body:
            fails.append(f"the predicate carries {forbidden!r}")
    plan_arm = braced_body(body, "if (plan) {")
    if plan_arm is None:
        fails.append("the predicate has no `if (plan) {` arm")
    else:
        statements = [stmt for stmt in tight(plan_arm).split(";") if stmt]
        last = statements[-1] if statements else ""
        if not last.startswith("returnggml_sycl_moe_tensor_plan_primary_layout(") or "GGML_LAYOUT_AOS" not in last:
            fails.append("the plan arm does not end by returning the plan's AOS primary layout on this device")
        if "ggml_backend_buffer_is_sycl(" in plan_arm:
            fails.append("the plan arm consults the weight's buffer: a known plan answers alone")
        tail = body[body.find("if (plan) {") + len("if (plan) {") + len(plan_arm) :]
        for needed in ("ggml_backend_buffer_is_sycl(", "!ggml_backend_buffer_is_sycl_split("):
            if needed not in tail:
                fails.append(f"the no-plan arm does not ask {needed}")
    return fails


def claim_b_one_predicate(s):
    fails = []
    sycl = s["sycl"]
    if sycl.count(GATE) != 1:
        fails.append("the supports_op MUL_MAT_ID gate does not ask the BF16 predicate as its third term")
    calls = len(re.findall(rf"\b{PREDICATE}\(", sycl))
    if calls != 3:
        fails.append(f"{PREDICATE} appears {calls} times; want its definition, the gate and the route")
    if sycl.count(ROUTE_IF) != 1:
        fails.append("the route does not ask the predicate exactly once")
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
    for needed in (TABLE_CALL, IDS_CALL, SUBMIT_CALL, RETAIN_CALL, REFUSAL_WARN, REFUSAL_THROW, ROUTE_LOG,
                   "ggml_sycl::ggml_sycl_moe_route_log_enabled()"):
        if needed not in body:
            fails.append(f"{ROUTE_FN} does not contain {needed}")
    lowered = body.lower()
    for forbidden in ROUTE_FORBIDDEN:
        if forbidden.lower() in lowered:
            fails.append(f"{ROUTE_FN} contains {forbidden!r}")
    order = [body.find(TABLE_CALL), body.find(SUBMIT_CALL), body.find(RETAIN_CALL)]
    if min(order) >= 0 and not order[0] < order[1] < order[2]:
        fails.append("the table is not built before the submit, or the leases are not retained after it")
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
    if '#include "moe-bf16-direct.hpp"' not in s["sycl"]:
        fails.append("ggml-sycl.cpp does not include moe-bf16-direct.hpp")
    for needed in ("bf16_to_f32(", "sycl::reduce_over_group(", "quiet_NaN()", "== -1", "reqd_sub_group_size(16)",
                   "sycl::nd_range<3>"):
        if needed not in cpp:
            fails.append(f"the kernel lacks {needed}")
    zero = cpp.find("== -1")
    nan = cpp.find("quiet_NaN()")
    if zero >= 0 and nan >= 0 and not zero < nan:
        fails.append("the id -1 zero row is not decided before the NaN arm")
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
    ("a", "sycl", PREDICATE_HEAD, "!ggml_backend_buffer_is_sycl_split(", "true || !ggml_backend_sycl_unused("),
    ("b", "sycl", None, f" && !{PREDICATE}(op, device)) {{", ") {"),
    ("b", "sycl", None, ROUTE_IF, "if (true) {"),
    ("c", "sycl", None, ROUTE_IF + " " + ROUTE_FN + "(ctx, dst); return; }", ""),
    ("c", "sycl", ROUTE_IF, ROUTE_FN + "(ctx, dst); return; }", ROUTE_FN + "(ctx, dst); }"),
    ("d", "sycl", ROUTE_FN_HEAD, TABLE_CALL, TABLE_CALL.replace("GGML_LAYOUT_AOS", "GGML_LAYOUT_SOA")),
    ("d", "sycl", ROUTE_FN_HEAD, RETAIN_CALL, "(void) ("),
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
