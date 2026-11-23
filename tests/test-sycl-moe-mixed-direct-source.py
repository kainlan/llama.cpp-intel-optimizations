#!/usr/bin/env python3
"""A Q1_0/NVFP4 MUL_MAT_ID node that selects card experts and host experts runs both (Inkling Task 8c).

Placement decides the executor per occurrence (lead ruling 2026-11-22). The direct block in ggml_sycl_mul_mat_id is
the one place a Q1_0/NVFP4 node runs: it admits a batch whose operands are each PRIMARY on its card or HOST, with at
least one on the card. Card occurrences run in the admitted device kernel; host occurrences run on the CPU through the
host recipe (ggml_sycl_cpu_moe_host_aos_execute), and their rows reach dst after the kernel's output copy, on the same
in-order queue. One terminal covers both legs. These claims are read from the source; the device test
(test-q1-nvfp4-admitted-device.cpp, mixed_cases) runs the mixes.

The claims, each failure message starting with its name:
- kernel-*: the kernel computes no occurrence whose expert is not on the card, and reads through no null slot. An id
  past the table, or a null slot, is written as quiet NaN (the host scatter overwrites the host rows); a negative id is
  ggml's skip and is written as zero. The check is a compile-time property of the one Q1_0/NVFP4 instantiation the
  admitted adapter uses, so the other mul_mat_vec_q_id callers' kernel images do not change.
- table-*: a slot of the expert pointer table is null or a pointer the table's owner holds a lease for. A table is
  zeroed when it is created or its views are rebuilt, and a HOST operand's slot is nulled with its handle dropped. A
  refusal, or a throw, after the host payload may have changed marks the device table stale (table-stale). One upload
  builds every retained table: with skip_host, a mixed node's card table (table-card, direct-table); without it, a
  HOST operand refuses the batch as before.
- view-*, admission-*: the batch views and the workspace admission take the card occurrences only, each with its own
  occurrence index; the refusing local view stays for every other caller.
- host-*: the host leg refuses by name before admission (no input marker in play, a task the host recipe would refuse,
  no host workspace), never aborts, and lets a failure in its job reach the graph boundary through the join's host task.
  A workspace refusal prints the plan term only for the recipe workspace, the one buffer that has one, and planned=none
  for the activation and output buffers (host-workspace).
  The host recipe's executor runs on its check's findings. A leg whose catch could not drain the queue joins the
  bundle's quarantine through the terminal, unless the terminal task was submitted, which then owns its release
  (host-quarantine). A leg lets its job go before its future goes and as it is parked, and a job let go stops waiting
  for its input marker and runs no further task (host-let-go): the registry lets a quarantined slot go at process exit
  before the cache abandons the marker word, std::async's future waits its job, and a parked leg's marker may never be
  written.
- terminal-*: the node's terminal is armed on the last host row copy, after the device output copy.
- funnel-order: the per-op funnel flushes a pending CPU scatter on every op's path before MUL_MAT_ID is dispatched,
  which the direct block relies on to read src1 rows a deferred scatter may still owe.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SYCL_DIR = ROOT / "ggml/src/ggml-sycl"
FILES = {
    "sycl": SYCL_DIR / "ggml-sycl.cpp",
    "mmvq": SYCL_DIR / "mmvq.cpp",
    "batch": SYCL_DIR / "moe-resolved-batch.hpp",
    "workspace": SYCL_DIR / "moe-mmid-workspace.cpp",
    "workspace_hpp": SYCL_DIR / "moe-mmid-workspace.hpp",
    "plan_test": SYCL_DIR / "tests/test-moe-mmid-workspace-plan.cpp",
    "cpu": SYCL_DIR / "cpu-dispatch.cpp",
    "cpu_hpp": SYCL_DIR / "cpu-dispatch.hpp",
    "signal": SYCL_DIR / "cpu-job-signal.hpp",
}

DIRECT_GATE = "if ((q1_nvfp4_direct_b70_validated || q1_nvfp4_private_test_authorized) &&"
DIRECT_END = "// MoE hybrid GPU+CPU dispatch gate"
PRE_ADMISSION = "ggml_sycl_moe_mark_direct_authority_pre_admission(src0)"
DECLINE = "ggml_sycl_moe_direct_log_decline("
CHECKED = "mmvq_id_expert_slots_checked"
UNCHECKED = "mmvq_id_expert_slots_unchecked"
SKIP_HOST = "if (operand.residency() == ggml_sycl::moe_batch_residency::HOST) { continue; }"
HOST_HELPER = re.compile(r"\bstatic\s+[\w:<>,\s\*&]+?\b(ggml_sycl_moe_direct_host_\w+)\s*\(")
COMMENT_OR_LITERAL = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])\'', re.DOTALL)

_BLANKED: dict[int, tuple[str, str, str]] = {}


def read_sources() -> dict[str, str]:
    return {key: path.read_text() for key, path in FILES.items()}


def _blanked(text: str) -> tuple[str, str]:
    """(comments and literals blanked, comments blanked): the same text and length with those turned to spaces."""
    key = hash(text)
    cached = _BLANKED.get(key)
    if cached is None or cached[0] != text:
        def all_of(match: re.Match) -> str:
            return re.sub(r"[^\n]", " ", match.group(0))

        def comments(match: re.Match) -> str:
            return all_of(match) if match.group(0)[0] == "/" else match.group(0)

        cached = (text, COMMENT_OR_LITERAL.sub(all_of, text), COMMENT_OR_LITERAL.sub(comments, text))
        _BLANKED[key] = cached
    return cached[1], cached[2]


def code_of(text: str) -> str:
    return _blanked(text)[0]


def commentless(text: str) -> str:
    return _blanked(text)[1]


def matching(code: str, open_at: int) -> int:
    opener = code[open_at]
    closer = {"{": "}", "(": ")"}[opener]
    depth = 0
    for i in range(open_at, len(code)):
        if code[i] == opener:
            depth += 1
        elif code[i] == closer:
            depth -= 1
            if depth == 0:
                return i
    raise ValueError(f"unbalanced {opener} at {open_at}")


def spaced(text: str) -> str:
    """Whitespace runs as one space and none inside brackets' edges, so a literal survives re-wrapping."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    return re.sub(r"\s+([\)\]\}])", r"\1", text).strip()


def has(flat: str, needle: str) -> bool:
    """Whether the spaced text holds the needle, spaced the same way."""
    return spaced(needle) in flat


def find(flat: str, needle: str, start: int = 0) -> int:
    return flat.find(spaced(needle), start)


def body_after_params(code: str, open_paren: int) -> tuple[int, int] | None:
    """(open, close) of the body that follows the parameter list opening at open_paren, or None for a declaration."""
    close_paren = matching(code, open_paren)
    rest = re.match(r"[\s\w:&>\-]*?(\{|;)", code[close_paren + 1:])
    if rest is None or rest.group(1) != "{":
        return None
    brace = close_paren + 1 + rest.end() - 1
    return brace, matching(code, brace)


def definition(text: str, head: str) -> str | None:
    """The body, braces included, of the first definition whose text starts with `head` (a declaration is skipped)."""
    code = code_of(text)
    start = code.find(head)
    while start >= 0:
        open_paren = code.find("(", start)
        body = body_after_params(code, open_paren)
        if body is not None:
            return text[body[0]:body[1] + 1]
        start = code.find(head, matching(code, open_paren))
    return None


def template_head(text: str, head: str) -> str | None:
    """The `template <...>` that opens the definition starting with `head`."""
    at = text.find(head)
    start = text.rfind("template <", 0, at) if at >= 0 else -1
    return text[start:at] if start >= 0 else None


def direct_block(sycl: str) -> str | None:
    gate = sycl.find(DIRECT_GATE)
    end = sycl.find(DIRECT_END, gate)
    return sycl[gate:end] if gate >= 0 and end > gate else None


def host_helpers(sycl: str) -> dict[str, str]:
    """Every ggml_sycl_moe_direct_host_* definition, by name, comments blanked."""
    code = code_of(sycl)
    found = {}
    for m in HOST_HELPER.finditer(code):
        body = body_after_params(code, m.end() - 1)
        if body is not None:
            found[m.group(1)] = commentless(sycl)[body[0]:body[1] + 1]
    return found


def kernel_violations(src: dict[str, str]) -> list[str]:
    found: list[str] = []
    mmvq = commentless(src["mmvq"])
    if not re.search(r"struct\s+" + UNCHECKED + r"\s*\{\s*\}\s*;", mmvq) or \
            not re.search(r"struct\s+" + CHECKED + r"\s*\{\s*int\s+n_slots\s*;\s*\}\s*;", mmvq):
        found.append("kernel-tags: the checked and unchecked expert-slot tags are not defined as empty and "
                     "{ int n_slots; }")
    head = template_head(mmvq, "static void mul_mat_vec_q_id(")
    body = definition(mmvq, "static void mul_mat_vec_q_id(")
    if head is None or body is None:
        return found + ["kernel-template: mul_mat_vec_q_id was not found"]
    if not re.search(r"typename\s+slots_t\s*=\s*" + UNCHECKED, head):
        found.append("kernel-template: mul_mat_vec_q_id's slot check is not a template parameter that defaults to "
                     "unchecked")
    code = code_of(body)
    guard = re.search(r"if\s+constexpr\s*\(\s*std::is_same_v<\s*slots_t\s*,\s*" + CHECKED + r"\s*>\s*\)", code)
    read = code.find("x = static_cast<const block_q_t *>(expert_ptrs[expert_id]);")
    if guard is None or read < 0 or guard.start() > read:
        found.append("kernel-order: the checked instantiation does not test the slot before it reads through it")
    else:
        brace = code.index("{", guard.end())
        checked = spaced(code[brace:matching(code, brace) + 1])
        if not all(has(checked, part) for part in ("std::numeric_limits<float>::quiet_NaN()", "slots.n_slots",
                                                    "expert_ptrs[expert_id] == nullptr", "return;")):
            found.append("kernel-nan: the checked block does not write NaN and return for an id past the table or a "
                         "null slot")
        if not has(checked, "expert_id < 0") or not has(checked, "0.0f"):
            found.append("kernel-skip: the checked block does not write a negative id's row as zero")
        if "reduce_over_group" in checked:
            found.append("kernel-order: the checked block reduces before it decides")
    if not re.search(r"template\s*<\s*int\s+qtype\s*>\s*class\s+mmvq_id_checked_kernel_name\s*;", mmvq):
        found.append("kernel-name: the checked instantiation has no kernel name of its own")
    impl_head = template_head(mmvq, "static sycl::event mmvq_submit_aos_id_impl(")
    impl = definition(mmvq, "static sycl::event mmvq_submit_aos_id_impl(")
    if impl_head is None or impl is None or not re.search(r"typename\s+slots_t\s*=\s*" + UNCHECKED, impl_head) or \
            not has(spaced(impl), "parallel_for<mmvq_id_checked_kernel_name<qtype>>") or \
            not has(spaced(impl), "parallel_for<mmvq_id_kernel_name<qtype>>"):
        found.append("kernel-name: mmvq_submit_aos_id_impl does not choose its kernel name by the slot tag")
    # Only the admitted adapter's overload instantiates the check: its Q1_0 call and its NVFP4 call, nowhere else.
    code = code_of(src["mmvq"])
    dependency = re.search(r"bool\s+mmvq_submit_q1_nvfp4_aos_id\([^;{]*?const\s+sycl::event\s*&\s*dependency", code,
                           re.DOTALL)
    if dependency is None:
        return found + ["kernel-callers: the adapter's dependency overload was not found"]
    open_paren = code.index("(", dependency.start())
    body_range = body_after_params(code, open_paren)
    uses = [m.start() for m in re.finditer(r"(?<!struct )" + CHECKED + r"\s*\{", code)]
    if body_range is None or len(uses) != 2 or not all(body_range[0] < u < body_range[1] for u in uses):
        found.append(f"kernel-callers: {len(uses)} calls instantiate the checked kernel, not the adapter overload's "
                     "Q1_0 and NVFP4 calls alone")
    if not re.search(r"\bint\s+n_expert_slots\b", code[open_paren:matching(code, open_paren)]):
        found.append("kernel-callers: the adapter overload is not told the table's slot count")
    adapter = definition(src["mmvq"], "bool mmvq_submit_q1_nvfp4_aos_id_admitted(")
    flat = spaced(code_of(adapter)) if adapter else ""
    if not has(flat, "ids_host[i] < 0 || ids_host[i] >= n_experts"):
        found.append("kernel-callers: the adapter no longer refuses a snapshot id outside [0, n_experts) before "
                     "submit")
    elif not has(flat, "static_cast<int64_t>(dst_nb2), n_experts, quant_event, event_out)"):
        found.append("kernel-callers: the adapter does not pass n_experts as the table's slot count")
    return found


def table_violations(src: dict[str, str]) -> list[str]:
    found: list[str] = []
    sycl = src["sycl"]
    ensure = definition(sycl, "static bool ggml_sycl_ensure_moe_ptr_table(")
    if ensure is None:
        return ["table-ensure: ggml_sycl_ensure_moe_ptr_table was not found"]
    flat = spaced(code_of(ensure))
    prealloc_at = find(flat, "if (table_index >= 0) {")
    runtime_at = find(flat, "ggml_sycl::alloc_request req{};")
    if prealloc_at < 0 or runtime_at < prealloc_at:
        return found + ["table-ensure: the preallocated and runtime branches were not found"]
    existing, prealloc, runtime = flat[:prealloc_at], flat[prealloc_at:runtime_at], flat[runtime_at:]
    # The fills this task adds are queued, not waited: only commands behind them on the same in-order queue read the
    # table, and the submitting thread takes no new host wait. The runtime branch's own fill predates them.
    fill = "ggml_sycl::mem_fill_async(table_handle, 0, bytes, queue);"
    swap = "extra->weight().moe_expert_ptrs_handle[device] = std::move(table_handle);"
    if not has(prealloc, fill) or not has(prealloc, swap) or find(prealloc, fill) > find(prealloc, swap):
        found.append("table-zero: a preallocated table is not zeroed before it becomes the weight's table")
    if not has(runtime, "ggml_sycl::mem_fill(table_handle, 0, bytes, queue);") and not has(runtime, fill):
        found.append("table-zero: a runtime table is not zeroed when it is created")
    rebuild = find(existing, "ggml_sycl_build_moe_table_views(")
    if rebuild < 0 or not has(existing[rebuild:], "ggml_sycl::mem_fill_async(extra->weight().moe_expert_ptrs_handle["
                                                  "device], 0, bytes, queue);"):
        found.append("table-zero: an existing table whose views are rebuilt to null is not zeroed with them")

    upload = definition(sycl, "static const void * const * ggml_sycl_upload_moe_transient_ptr_table(")
    if upload is None:
        return found + ["table-host-null: the transient table upload was not found"]
    head_at = sycl.find("static const void * const * ggml_sycl_upload_moe_transient_ptr_table(")
    params = spaced(sycl[head_at:sycl.find("{", head_at)])
    flat = spaced(code_of(upload))
    if not has(params, "const std::vector<int32_t> * null_experts = nullptr"):
        found.append("table-host-null: the upload takes no list of slots to null")
    loop = find(flat, "for (const int32_t eid : *null_experts) {")
    if loop < 0 or not all(has(flat[loop:loop + 700], part) for part in (
            "expert_handles[slot] = ggml_sycl::mem_handle{};", "ptr_payload[slot] = nullptr;", "any_updated = true;")):
        found.append("table-host-null: a slot to null does not lose its handle and its pointer, or the change is not "
                     "uploaded")
    # A refusal once the payload may have changed, before its upload, marks the device table stale: otherwise the host
    # payload is ahead of the table, and the next upload skips the slots it thinks the table already holds.
    code = code_of(upload)
    lam = re.search(r"auto\s+refuse\s*=\s*\[&\]\(\)\s*->\s*const\s+void\s*\*\s*const\s*\*\s*\{", code)
    change = re.search(r"any_updated\s*=\s*true\s*;", code)
    if lam is None:
        found.append("table-stale: the transient upload has no refusal that marks the table stale")
    else:
        lam_end = matching(code, lam.end() - 1)
        if not has(spaced(code[lam.end() - 1:lam_end + 1]),
                   "if (any_updated) { extra->weight().mark_moe_device_table_stale(device); }"):
            found.append("table-stale: the upload's refusal does not mark the table stale once the payload changed")
        elif change is None or change.start() < lam.start() or re.search(r"return\s+nullptr\s*;", code[lam_end:]):
            found.append("table-stale: a refusal after the payload can change returns without marking the table "
                         "stale")
    # A throw once the payload may have changed (the payload's upload can throw) leaves it ahead of the device table
    # just as a refusal does, so a guard held from before the first change marks the table stale as any throw leaves
    # the upload. A destructor, not a catch: it swallows nothing, and it marks every throw, the fallback error included
    # that the rethrow-first rule makes a catch pass up untouched.
    guard = struct_body(code, "moe_ptr_table_stale_on_throw")
    marks = guard is not None and re.search(
        r"~moe_ptr_table_stale_on_throw\s*\(\s*\)\s*(?:noexcept\s*)?\{\s*if\s*\(\s*payload_changed\s*&&\s*"
        r"std::uncaught_exceptions\s*\(\s*\)\s*>\s*exceptions\s*\)\s*\{\s*weight\s*\.\s*"
        r"mark_moe_device_table_stale\s*\(\s*device\s*\)\s*;\s*\}\s*\}", guard) and re.search(
        r"\bconst\s+int\s+exceptions\s*=\s*std::uncaught_exceptions\s*\(\s*\)\s*;", guard)
    held = re.search(r"\bconst\s+moe_ptr_table_stale_on_throw\s+\w+\s*\{\s*extra\s*->\s*weight\s*\(\s*\)\s*,\s*"
                     r"device\s*,\s*any_updated\s*\}\s*;", code)
    declared = re.search(r"\bbool\s+any_updated\s*=\s*false\s*;", code)
    copy_at = code.find("ggml_sycl_copy_payload_to_handle_async(")
    if not marks:
        found.append("table-stale: no guard marks the table stale as a throw leaves the upload once the payload "
                     "changed")
    elif not held or not declared or change is None or copy_at < 0 or \
            not declared.end() <= held.start() < change.start() < copy_at:
        found.append("table-stale: the upload does not hold the stale-on-throw guard from before its payload can "
                     "change")
    # One upload builds a retained table: of an all-card batch (the hybrid decode, skip_host off, where a HOST operand
    # refuses the batch as the local view does) and of a mixed node's card part (skip_host on). The batch's view and
    # its null slots come from the shared upload from a batch, which takes skip_host too; the retain is the retained
    # upload's own.
    def head_of(head: str) -> tuple[str | None, str]:
        at = sycl.find(head)
        return definition(sycl, head), spaced(sycl[at:sycl.find("{", at)]) if at >= 0 else ""

    retained, retained_params = head_of(
        "static ggml_sycl::moe_retained_pointer_table ggml_sycl_upload_moe_retained_ptr_table_from_batch(")
    shared, shared_params = head_of("static const void * const * ggml_sycl_upload_moe_ptr_table_from_batch(")
    if retained is None or shared is None:
        return found + ["table-card: the retained upload or the shared upload from a batch is missing"]
    retained, shared = spaced(code_of(retained)), spaced(code_of(shared))
    if not has(retained_params, "bool skip_host = false)") or not has(shared_params, "bool skip_host = false)") or \
            not re.search(r"\bggml_sycl_upload_moe_ptr_table_from_batch\([^;]*, skip_host\);", retained) or \
            "ggml_sycl_upload_moe_card_ptr_table_from_batch" in code_of(sycl):
        found.append("table-card: the card table has an upload of its own, not the retained one with skip_host")
    if not has(shared, "ggml_sycl::make_moe_batch_view(batch, layout, skip_host)"):
        found.append("table-card: the table's view does not leave HOST operands out with skip_host")
    if not re.search(r"ggml_sycl_upload_moe_transient_ptr_table\([^;]*host_experts", shared):
        found.append("table-card: the card table does not null its HOST operands' slots")
    leases = find(retained, "result.role_leases.push_back(")
    if leases < 0 or not has(retained[:leases], SKIP_HOST):
        found.append("table-card: a HOST operand's lease enters the table owner's retain")
    return found


def view_violations(src: dict[str, str]) -> list[str]:
    found: list[str] = []
    batch = src["batch"]
    shared = definition(batch, "inline moe_batch_local_view make_moe_batch_view(")
    local = definition(batch, "inline moe_batch_local_view make_moe_batch_local_view(")
    if shared is None or local is None:
        return ["view-card: make_moe_batch_view and make_moe_batch_local_view are not both defined"]
    if not has(spaced(code_of(local)), "return make_moe_batch_view(batch, layout, false);"):
        found.append("view-local: the local view no longer refuses a HOST operand")
    flat = spaced(code_of(shared))
    skip = find(flat, "if (skip_host && operand.residency() == moe_batch_residency::HOST) { continue; }")
    if skip < 0 or skip > find(flat, "expert_slots.emplace("):
        found.append("view-card: a HOST operand is not skipped before the view counts its expert")
    return found


def admission_violations(src: dict[str, str]) -> list[str]:
    found: list[str] = []
    shape = re.search(r"struct workspace_admission_shape \{(.*?)\};", src["workspace_hpp"], re.DOTALL)
    if shape is None or not re.search(r"size_t\s+tokens\s*=\s*0\s*;", shape.group(1)):
        found.append("admission-tokens: the admission shape carries no token count")
    admit = definition(src["workspace"], "moe_mmid_admitted_result moe_mmid_workspace_registry::admit(\n"
                                         "    moe_mmid_authoritative_admission_request")
    if admit is None:
        return found + ["admission-tokens: the authoritative admit was not found"]
    flat = spaced(code_of(admit))
    if not has(flat, "const size_t capacity = request.shape.tokens;"):
        found.append("admission-tokens: the admitted capacity is not the request's token count")
    if has(flat, "bindings->size() % request.shape.top_k") or has(flat, "id.occurrence != i"):
        found.append("admission-occurrences: admission still requires every occurrence of the batch")
    if not has(flat, "id.occurrence >= occurrence_limit") or not has(flat, "id.occurrence <= previous_occurrence"):
        found.append("admission-occurrences: admission does not require strictly increasing occurrences below "
                     "tokens * top_k")
    direct = direct_block(src["sycl"])
    if direct is None or not re.search(r"request\.shape = \{[^}]*static_cast<size_t>\(ne12\)\s*\}",
                                       spaced(code_of(direct))):
        found.append("admission-tokens: the direct route does not tell admission its token count")
    if re.search(r"\{\s*2,\s*2,\s*64,\s*96,\s*1\s*\}", src["plan_test"]):
        found.append("admission-tokens: the workspace plan test still admits without a token count")
    return found


def direct_violations(src: dict[str, str]) -> list[str]:
    found: list[str] = []
    sycl = src["sycl"]
    raw = direct_block(sycl)
    if raw is None:
        return ["direct-block: the direct block was not found"]
    direct = spaced(commentless(raw))
    # Positions in the block's own text: a host refusal is pre-admission when it lies outside the body that marks the
    # authority and admits, which is where the issuer's else chain and everything before the issuer are.
    raw_code, raw_text = code_of(raw), commentless(raw)
    pre = raw_code.find(PRE_ADMISSION)
    if pre < 0 or find(direct, PRE_ADMISSION) < 0:
        return ["direct-block: the pre-admission mark was not found"]
    admitted_open = raw_code.rfind("{", 0, pre)
    admitted_close = matching(raw_code, admitted_open)

    def pre_admission(at: int) -> bool:
        return at >= 0 and not admitted_open < at < admitted_close

    precondition = re.search(r"const bool card_or_host = (.*?);(?= const bool context_id)", direct)
    if precondition is None:
        found.append("direct-precondition: the first conjunct is not card_or_host")
    elif not all(part in precondition.group(1) for part in (
            "moe_batch_residency::PRIMARY_DEVICE", "moe_batch_residency::HOST", "std::any_of(")):
        found.append("direct-precondition: card_or_host does not admit PRIMARY or HOST operands with at least one on "
                     "the card")
    tables = re.findall(r"ggml_sycl_upload_moe_retained_ptr_table_from_batch\([^;]*?(, true)?\);", direct)
    if tables != [", true"]:
        found.append("direct-table: the direct block does not upload the card table (the retained table with "
                     "skip_host)")
    for name, needle in (("binding", "canonical_allocation_integration::bind("),
                         ("queue capability", "if (!operand.recipe_queue_capability().valid() ||")):
        where = find(direct, needle)
        loop = direct.rfind("for (", 0, where) if where >= 0 else -1
        if loop < 0 or not has(direct[loop:where], SKIP_HOST):
            found.append(f"direct-card-loops: the {name} loop does not skip HOST operands")

    for step, needle in (("host_input_marker", "GGML_SYCL_MOE_CPU_JOB_DEVICE_SIGNAL"), ("host_task", ""),
                         ("host_workspace", "")):
        line = re.search(re.escape(DECLINE) + r'\s*src0,\s*ne12,\s*"' + step + '",', raw_text)
        if line is None or not pre_admission(line.start()):
            found.append(f"host-steps: no decline before admission names the step {step}")
        elif needle not in raw_text[line.start():line.start() + 400]:
            found.append(f"host-steps: the {step} decline does not name {needle}")
    in_play, may = raw_code.find("moe_cpu_job_input_marker_in_play()"), raw_code.find("moe_cpu_job_input_marker_may(")
    if not 0 <= in_play < admitted_open or not 0 <= may < admitted_open:
        found.append("host-steps: the input marker is not asked for before admission")
    helpers = host_helpers(sycl)
    prepare = helpers.get("ggml_sycl_moe_direct_host_prepare", "")
    if "ggml_sycl_cpu_moe_host_aos_check(" not in prepare:
        found.append("host-validate: the host leg's tasks are not checked by the host recipe's own check")
    elif not 0 <= raw_code.find("ggml_sycl_moe_direct_host_prepare(") < admitted_open:
        found.append("host-validate: the host leg is not prepared and checked before admission")
    # Only the recipe workspace has a plan term (moe_host_recipe_workspace_bytes); the activation and output buffers
    # have none, so their refusal carries none and the decline prints planned=none, never another buffer's budget.
    calls = re.findall(r'refuse_workspace\("(\w+)", \w+, ([^()]*)\)', spaced(prepare))
    terms = {"recipe_workspace": "plan.moe_host_recipe_workspace_bytes", "activation": "std::nullopt",
             "output": "std::nullopt"}
    prepared = struct_body(code_of(sycl), "moe_direct_host_prepared") or ""
    if {buffer for buffer, _ in calls} != set(terms) or any(term != terms[buffer] for buffer, term in calls) or \
            not re.search(r"\bstd::optional<size_t>\s+planned\s*;", prepared):
        found.append("host-workspace: a workspace refusal carries a plan term for a buffer that has none, or none "
                     "for the recipe workspace")
    planned = find(direct, "if (host.planned) {")
    number = find(direct, '"host_workspace", "buffer=%s bytes=%zu planned=%zu", host.buffer, host.bytes, '
                          '*host.planned);')
    none = find(direct, '"host_workspace", "buffer=%s bytes=%zu planned=none", host.buffer, host.bytes);')
    if not 0 <= planned < number < find(direct, "} else {", number) < none:
        found.append("host-workspace: the host_workspace decline does not print planned=%zu for the recipe "
                     "workspace alone and planned=none for the other buffers")
    for name, body in [("the direct block", raw)] + list(helpers.items()):
        if "GGML_ABORT" in code_of(body):
            found.append(f"host-no-abort: {name} aborts")
    launch = helpers.get("ggml_sycl_moe_direct_host_launch", "")
    if not all(part in launch for part in ("moe_cpu_job_wait_inputs(", "ggml_sycl_cpu_moe_host_aos_execute(", "throw ")):
        found.append("host-worker: the job does not wait its input marker, run the host recipe and throw on a "
                     "refusal")
    join = spaced(helpers.get("ggml_sycl_moe_direct_host_join", ""))
    # The join holds the job's future and nothing else: a functor copy the runtime keeps cannot hold the leg's buffers.
    if not re.search(r"host_task\(\[job = leg->job\]\s*\{job\.get\(\);\}\)", join) or \
            "unified_cache_note_host_task(" not in join:
        found.append("host-join: the job is not joined by a recorded host task whose get() lets its failure reach "
                     "the graph boundary")
    for name in ("ggml_sycl_moe_direct_host_gather", "ggml_sycl_moe_direct_host_scatter"):
        if "mem_copy_ptr_async(" not in helpers.get(name, ""):
            found.append(f"host-copies: {name} does not copy the host rows")

    gate_head = spaced("completion->bundle.mark_possible_submit()) {")
    gate = find(direct, gate_head)
    try_at = find(direct, "try {", gate) if gate >= 0 else -1
    if try_at < 0:
        return found + ["terminal-order: the submit try body was not found"]
    before_try = direct[gate + len(gate_head):try_at].strip()
    body = direct[try_at:]
    order = [find(body, needle) for needle in (
        "ggml_sycl_moe_direct_host_gather(", "mmvq_submit_q1_nvfp4_aos_id_admitted(",
        "ggml_sycl_moe_direct_host_launch(", "sycl::event final = ggml_sycl::mem_copy_ptr_async(dst_device_base",
        "ggml_sycl_moe_direct_host_join(", "ggml_sycl_moe_direct_host_scatter(", "terminal->arm(last);")]
    if any(at < 0 for at in order) or order != sorted(order):
        found.append("terminal-order: the try body is not gather, kernel, launch, output copy, join, scatter, then "
                     "the terminal armed on the last copy")
    if not has(body, "h.depends_on(last);") or not has(body, "completion->host_leg.reset();"):
        found.append("terminal-release: the terminal host task does not follow the last copy and release the host "
                     "leg")
    # A failed drain proves no quiescence: the host leg joins the bundle's quarantine through the terminal, so its
    # buffers are not freed under a queued copy and the submitting thread does not wait its job.
    # The branch parks the leg and does nothing else to it, but let its job go (host-let-go).
    drain = find(body, "exact_queue->wait_and_throw();")
    head = re.search(r"\} else if \((!terminal_submitted && )?completion->host_leg\) \{", body)
    park = head.start() if head else -1
    branch = body[head.end() - 1:matching(body, head.end() - 1) + 1] if head else ""
    parked = branch.replace(spaced("completion->host_leg->let_go_job();") + " ", "", 1)
    if drain < 0 or park < drain or find(body, "if (quiescent) {", drain) > park or \
            find(body, "if (quiescent) {", drain) < 0 or \
            parked != "{" + spaced("terminal->quarantine_host_leg(std::move(completion->host_leg));") + "}":
        found.append("host-quarantine: a host leg whose drain failed does not join the bundle's quarantine through "
                     "the terminal")
    # Once the terminal task is submitted it owns the leg's release: its body resets the leg on the host-task thread.
    # A flag cleared before the try is set as that submit returns, and the catch parks the leg only while it is clear;
    # otherwise a throw after the submit would move the leg here as that thread resets it.
    submit = find(body, "sycl::event terminal_task = exact_queue->submit(")
    submit_end = matching(body, submit + len(spaced("sycl::event terminal_task = exact_queue->submit(")) - 1) \
        if submit >= 0 else -1
    if before_try != "bool terminal_submitted = false;" or submit_end < 0 or \
            not body.startswith("; terminal_submitted = true;", submit_end + 1) or not head or not head.group(1):
        found.append("host-quarantine: the catch can park a host leg that the submitted terminal task releases on "
                     "the host-task thread")
    if not has(direct, "g_q1_nvfp4_test_host_submit.fetch_add(") or \
            not has(direct, "g_q1_nvfp4_test_host_occurrences.fetch_add("):
        found.append("host-counters: the route-testing build does not count the host leg's submits")
    return found


def cpu_violations(src: dict[str, str]) -> list[str]:
    found: list[str] = []
    if not re.search(r"bool\s+ggml_sycl_cpu_moe_host_aos_check\(\s*const\s+cpu_moe_host_aos_task\s*&\s*task,",
                     src["cpu_hpp"]):
        found.append("host-validate: the host recipe's check is not declared beside its executor")
    # The public check and the executor run one check: the check drops its findings, the executor runs on them.
    check = definition(src["cpu"], "bool ggml_sycl_cpu_moe_host_aos_check(")
    if check is None or spaced(code_of(check)) != ("{cpu_moe_host_aos_checked checked; return "
                                                   "cpu_moe_host_aos_check_into(task, reject, &checked);}"):
        found.append("host-validate: the public check is not the shared check alone")
    execute = definition(src["cpu"], "bool ggml_sycl_cpu_moe_host_aos_execute(")
    flat = spaced(code_of(execute)) if execute else ""
    if not flat.startswith("{using namespace ggml_sycl; cpu_moe_host_aos_checked checked; if "
                           "(!cpu_moe_host_aos_check_into(task, reject, &checked)) {return false;}"):
        found.append("host-validate: the executor does not run the same check first")
    elif "ggml_sycl_get_type_traits_cpu(" in flat or ".resolve(" in flat:
        found.append("host-validate: the executor looks up again what the check found")
    return found


def funnel_violations(src: dict[str, str]) -> list[str]:
    # The direct block reads src1 rows a deferred CPU scatter may still owe (G4); it relies on the per-op funnel to
    # flush that scatter first, at the top level of the dispatch function, before the MUL_MAT_ID case.
    forward = definition(src["sycl"], "static bool ggml_sycl_compute_forward_impl(")
    if forward is None:
        return ["funnel-order: ggml_sycl_compute_forward_impl was not found"]
    code = code_of(forward)
    flush = re.search(r"\bflush_pending_cpu_scatter_if_consumed\(\s*dst\s*,\s*ctx\.device\s*\)\s*;", code)
    dispatch = re.search(r"\bggml_sycl_mul_mat_id\(\s*ctx\s*,\s*dst\s*\)\s*;", code)
    if flush is None or dispatch is None or flush.start() > dispatch.start():
        return ["funnel-order: the pending CPU scatter is not flushed before MUL_MAT_ID is dispatched"]
    before = code[:flush.start()]
    if before.count("{") - before.count("}") != 1:
        return ["funnel-order: the pending CPU scatter flush is conditional, not on every op's path"]
    return []


def struct_body(code: str, name: str) -> str | None:
    """The body, braces included, of `struct name {`, in already blanked code."""
    m = re.search(r"\bstruct\s+" + name + r"\s*\{", code)
    return code[m.end() - 1:matching(code, m.end() - 1) + 1] if m else None


def let_go_violations(src: dict[str, str]) -> list[str]:
    # A leg parked in a quarantine is dropped only when its workspace slot is let go: once the terminal or a quiescence
    # proof is ready, or by the registry's teardown, and at process exit that registry (a function-local static) goes
    # before the caches, so before cpu_job_input_marker_end abandons the marker word its job may still wait on. The
    # leg's future comes from std::async, whose last copy waits the job, so the leg lets its job go first.
    found: list[str] = []
    code = code_of(src["sycl"])
    work = struct_body(code, "moe_direct_host_work")
    leg = struct_body(code, "moe_direct_host_leg")
    if work is None or not re.search(r"\bstd::atomic<bool>\s+let_go\s*\{\s*false\s*\}\s*;", work):
        found.append("host-let-go: the host work has no let_go flag, cleared at construction")
    lets_go = leg is not None and re.search(
        r"\bvoid\s+let_go_job\s*\(\s*\)\s*(?:const\s*)?(?:noexcept\s*)?\{\s*if\s*\(\s*work\s*\)\s*\{\s*work\s*->\s*"
        r"let_go\s*\.\s*store\s*\(\s*true\s*,\s*std::memory_order_release\s*\)\s*;\s*\}\s*\}", leg)
    if not lets_go or not re.search(r"~moe_direct_host_leg\s*\(\s*\)\s*(?:noexcept\s*)?\{\s*let_go_job\s*\(\s*\)\s*;"
                                    r"\s*\}", leg):
        found.append("host-let-go: the leg does not let its job go as it goes, before its future waits the job")
    launch = code_of(host_helpers(src["sycl"]).get("ggml_sycl_moe_direct_host_launch", ""))
    throw_if_let_go = (r"if\s*\(\s*held\s*->\s*let_go\s*\.\s*load\s*\(\s*std::memory_order_acquire\s*\)\s*\)\s*\{\s*"
                       r"throw\b")
    # The throw comes right after the marker wait: the input wait after it waits the marker again, without let_go.
    stop = re.search(r"\bcpu_job_input_marker_wait\s*\(\s*inputs\s*,\s*&\s*held\s*->\s*let_go\s*\)\s*;\s*" +
                     throw_if_let_go, launch)
    run = launch.find("ggml_sycl_cpu_moe_host_aos_execute(")
    if not stop or run < 0 or not stop.end() < run:
        found.append("host-let-go: the job does not wait its marker with the leg's let_go and throw, running nothing, "
                     "once let go")
    # A job let go while it runs stops at its next task: a dropped leg then waits for at most the task in hand, not for
    # every task left, under whatever lock or on whatever thread drops it.
    each = re.search(r"\bfor\s*\(\s*size_t\s+t\s*=\s*0\s*;\s*t\s*<\s*held\s*->\s*tasks\s*\.\s*size\s*\(\s*\)\s*;\s*"
                     r"\+\+\s*t\s*\)\s*\{\s*" + throw_if_let_go, launch)
    if not stop or not each or not stop.end() < each.start() or not each.end() < run:
        found.append("host-let-go: a job let go between its tasks runs the tasks left")
    # A parked leg's rows are never read (the graph has failed), and its marker may never be written: its job is let go
    # as it is parked, so it neither spins a core until exit nor runs tasks whose rows nothing reads.
    if not re.search(r"\}\s*else\s+if\s*\(\s*(?:!\s*terminal_submitted\s*&&\s*)?completion\s*->\s*host_leg\s*\)\s*\{"
                     r"\s*completion\s*->\s*host_leg\s*->\s*let_go_job\s*\(\s*\)\s*;\s*terminal\s*->\s*"
                     r"quarantine_host_leg\s*\(", code):
        found.append("host-let-go: a leg parked in the terminal keeps its job waiting for a marker that may never "
                     "come")
    header = code_of(src["signal"])
    m = re.search(r"inline\s+void\s+cpu_job_input_marker_wait\s*\(\s*const\s+cpu_job_input_marker\s*&\s*m\s*,\s*"
                  r"const\s+std::atomic<bool>\s*\*\s*let_go\s*=\s*nullptr\s*\)\s*\{", header)
    body = header[m.end() - 1:matching(header, m.end() - 1) + 1] if m else ""
    loop = re.search(r"\bwhile\s*\(\s*!\s*cpu_job_input_marker_seen\s*\(\s*m\s*\)\s*\)\s*\{", body)
    gone = re.search(r"\bif\s*\(\s*let_go\s*&&\s*let_go\s*->\s*load\s*\(\s*std::memory_order_acquire\s*\)\s*\)\s*\{\s*"
                     r"return\s*;\s*\}", body)
    if not loop or not gone or gone.start() < loop.end():
        found.append("host-let-go: cpu_job_input_marker_wait does not return, inside its wait, once let_go is set")
    return found


def violations(src: dict[str, str]) -> list[str]:
    return (kernel_violations(src) + table_violations(src) + view_violations(src) + admission_violations(src) +
            direct_violations(src) + cpu_violations(src) + funnel_violations(src) + let_go_violations(src))


def test_anchors_exist() -> None:
    # Methodology control, first: with the block or the kernel renamed, every claim below is vacuous.
    src = read_sources()
    assert direct_block(src["sycl"]) is not None
    assert definition(src["mmvq"], "static void mul_mat_vec_q_id(") is not None
    assert definition(src["sycl"], "static bool ggml_sycl_ensure_moe_ptr_table(") is not None
    assert PRE_ADMISSION in src["sycl"]


def test_mixed_direct_claims() -> None:
    failures = violations(read_sources())
    assert failures == [], "\n".join(failures)


def test_mutations_are_witnessed() -> None:
    src = read_sources()
    assert violations(src) == []
    mutations = [
        # The kernel reads through the slot before it checks it.
        ("mmvq", r"if constexpr \(std::is_same_v<slots_t, mmvq_id_expert_slots_checked>\)",
         "if constexpr (std::is_same_v<slots_t, mmvq_id_expert_slots_unchecked>)", "kernel-order"),
        # Another call instantiates the checked kernel, changing an image the others share.
        ("mmvq", r"(mmvq_id_expert_slots_checked\{\s*n_expert_slots\s*\}\);)",
         r"\1 (void) mmvq_id_expert_slots_checked{ 0 };", "kernel-callers"),
        # A null slot's row is left alone instead of NaN.
        ("mmvq", r"std::numeric_limits<float>::quiet_NaN\(\)", "123.0f", "kernel-nan"),
        # A preallocated table keeps whatever its slice held.
        ("sycl", r"ggml_sycl::mem_fill_async\(table_handle, 0, bytes, queue\);(\s*ggml_sycl_old_owner_retirement)",
         r"\1", "table-zero"),
        # A host operand's slot keeps its stale pointer.
        ("sycl", r"ptr_payload\[slot\]\s*=\s*nullptr;(\s*any_updated\s*=\s*true;\s*\}\s*\}\s*\})", r"\1",
         "table-host-null"),
        # The local view stops refusing a host operand.
        ("batch", r"make_moe_batch_view\(batch, layout, false\)", "make_moe_batch_view(batch, layout, true)",
         "view-local"),
        # Admission takes its capacity from the bindings again.
        ("workspace", r"const size_t capacity = request\.shape\.tokens;",
         "const size_t capacity = bindings->size() / request.shape.top_k;", "admission-tokens"),
        # The terminal is armed on the device output copy, before the host rows land.
        ("sycl", r"terminal->arm\(last\);", "terminal->arm(final);", "terminal-order"),
        # The activation buffer's refusal reports the recipe workspace's plan term as its own.
        ("sycl", r'refuse_workspace\("activation", act_size, std::nullopt\)',
         'refuse_workspace("activation", act_size, plan.moe_host_recipe_workspace_bytes)', "host-workspace"),
        # The recipe workspace's refusal drops its plan term.
        ("sycl", r'refuse_workspace\("recipe_workspace", recipe_bytes, plan\.moe_host_recipe_workspace_bytes\)',
         'refuse_workspace("recipe_workspace", recipe_bytes, std::nullopt)', "host-workspace"),
        # A buffer with no plan term prints a number again.
        ("sycl", r'"buffer=%s bytes=%zu planned=none", host\.buffer, host\.bytes\)',
         '"buffer=%s bytes=%zu planned=%zu", host.buffer, host.bytes, host.planned.value_or(0))', "host-workspace"),
        # The worker aborts on a refused task.
        ("sycl", r'throw std::runtime_error\(("MUL_MAT_ID direct route host task refused")\);', r"GGML_ABORT(\1);",
         "host-no-abort"),
        # The card table retains a host operand's lease.
        ("sycl", r"(ggml_sycl_upload_moe_retained_ptr_table_from_batch\(.*?)if \(operand\.residency\(\) == "
                 r"ggml_sycl::moe_batch_residency::HOST\) \{\s*continue;\s*\}(\s*result\.role_leases)", r"\1\2",
         "table-card"),
        # The retained upload drops skip_host on its way to the shared upload, so a mixed batch's table is refused.
        ("sycl", r"(&event_set),\s*skip_host\);", r"\1);", "table-card"),
        # The table's view refuses a HOST operand whatever skip_host says.
        ("sycl", r"make_moe_batch_view\(batch, layout, skip_host\)", "make_moe_batch_view(batch, layout, false)",
         "table-card"),
        # The direct block uploads the table without skip_host, so a mixed batch's table is refused.
        ("sycl", r"(ggml_sycl_upload_moe_retained_ptr_table_from_batch\(\s*ctx, src0, direct_batch,[^;]*?), "
                 r"/\*skip_host=\*/true\);", r"\1);", "direct-table"),
        # The null loop's id check refuses without marking the table stale.
        ("sycl", r"(for \(const int32_t eid : \*null_experts\) \{\s*if \(eid < 0 \|\| eid >= n_experts\) \{\s*)"
                 r"return refuse\(\);", r"\1return nullptr;", "table-stale"),
        # The upload's refusal stops marking the table stale.
        ("sycl", r"(auto refuse = \[&\]\(\) -> const void \* const \* \{\s*if \(any_updated\) \{\s*)"
                 r"extra->weight\(\)\.mark_moe_device_table_stale\(device\);", r"\1(void) device;", "table-stale"),
        # The stale-on-throw guard stops marking the table stale.
        ("sycl", r"(~moe_ptr_table_stale_on_throw\(\) \{\s*if \(payload_changed && std::uncaught_exceptions\(\) > "
                 r"exceptions\) \{\s*)weight\.mark_moe_device_table_stale\(device\);", r"\1(void) weight;",
         "table-stale"),
        # The guard marks the table stale on every exit, a successful return included.
        ("sycl", r"if \(payload_changed && std::uncaught_exceptions\(\) > exceptions\)", "if (payload_changed)",
         "table-stale"),
        # The upload holds no guard: a throw from the payload's copy leaves the payload ahead of a table still valid.
        ("sycl", r"const moe_ptr_table_stale_on_throw stale_on_throw\{[^;]*\};", "", "table-stale"),
        # The leg's let-go helper stores nothing: its future waits a job that may wait for its marker forever.
        ("sycl", r"work->let_go\.store\(true, std::memory_order_release\);", "(void) work;", "host-let-go"),
        # The leg goes without letting its job go.
        ("sycl", r"~moe_direct_host_leg\(\) \{\s*let_go_job\(\);\s*\}", "~moe_direct_host_leg() {}", "host-let-go"),
        # A job let go between its tasks runs the tasks left.
        ("sycl", r"(for \(size_t t = 0; t < held->tasks\.size\(\); \+\+t\) \{)\s*if \(held->let_go\.load\("
                 r"std::memory_order_acquire\)\) \{\s*throw std::runtime_error\([^;]*\);\s*\}", r"\1",
         "host-let-go"),
        # A parked leg's job keeps waiting for a marker that may never be written.
        ("sycl", r"completion->host_leg->let_go_job\(\);(\s*terminal->quarantine_host_leg)", r"\1", "host-let-go"),
        # The job waits its marker without the leg's let_go.
        ("sycl", r"cpu_job_input_marker_wait\(inputs, &held->let_go\);", "cpu_job_input_marker_wait(inputs);",
         "host-let-go"),
        # A job let go runs its tasks anyway.
        ("sycl", r'throw std::runtime_error\("MUL_MAT_ID direct route host job let go before its inputs landed"\);',
         "(void) 0;", "host-let-go"),
        # The marker wait ignores let_go.
        ("signal", r"if \(let_go && let_go->load\(std::memory_order_acquire\)\) \{\s*return;\s*\}", "",
         "host-let-go"),
        # A failed drain frees the host leg with the unwinding completion instead of parking it.
        ("sycl", r"terminal->quarantine_host_leg\(std::move\(completion->host_leg\)\);", "completion->host_leg.reset();",
         "host-quarantine"),
        # The catch parks the leg after the terminal task was submitted, as that task resets it on its own thread.
        ("sycl", r"\} else if \(!terminal_submitted && completion->host_leg\) \{", "} else if (completion->host_leg) {",
         "host-quarantine"),
        # The flag is never set, so the catch parks a leg the submitted terminal task owns.
        ("sycl", r"(\}\);\s*)terminal_submitted = true;", r"\1", "host-quarantine"),
        # The executor resolves the weight again instead of using what the check found.
        ("cpu", r"const resolved_ptr &\s*weight\s*=\s*checked\.weight;",
         "const resolved_ptr weight = task.weight_lease.resolve(mem_handle::HOST_DEVICE);", "host-validate"),
        # The funnel's flush is removed.
        ("sycl", r"flush_pending_cpu_scatter_if_consumed\(dst, ctx\.device\);(\s*flush_pending_secondary_scatter_if_"
                 r"consumed\(dst, ctx\.device\);)", r"\1", "funnel-order"),
        # The funnel's flush moves after the MUL_MAT_ID dispatch.
        ("sycl", r"flush_pending_cpu_scatter_if_consumed\(dst, ctx\.device\);(\s*flush_pending_secondary_scatter_if_"
                 r"consumed\(dst, ctx\.device\);.*?ggml_sycl_mul_mat_id\(ctx, dst\);)",
         r"\1 flush_pending_cpu_scatter_if_consumed(dst, ctx.device);", "funnel-order"),
        # The funnel's flush runs only for some ops.
        ("sycl", r"flush_pending_cpu_scatter_if_consumed\(dst, ctx\.device\);(\s*flush_pending_secondary_scatter_if_"
                 r"consumed\(dst, ctx\.device\);)",
         r"if (dst->op == GGML_OP_MUL_MAT) { flush_pending_cpu_scatter_if_consumed(dst, ctx.device); }\1",
         "funnel-order"),
    ]
    for key, pattern, replacement, claim in mutations:
        mutated = dict(src)
        mutated[key] = re.sub(pattern, replacement, src[key], count=1, flags=re.DOTALL)
        assert mutated[key] != src[key], f"mutation for {claim} did not change the source: {pattern[:60]}"
        failures = violations(mutated)
        assert any(f.startswith(claim + ":") for f in failures), f"mutation did not fail {claim}: {failures}"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
