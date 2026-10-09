"""Source contract for llama.cpp-cre6: the host-expert MoE scatter scratch is a PLANNED byte, CLAIMED at publish.

Host-only: reads sources, builds nothing, loads no model and touches no device.

The host-expert MUL_MAT_ID result scatter used to issue one H2D copy per run of rows with adjacent destinations
(~12 per MoE layer on Qwen3.8 decode). It now copies the CPU's compact result block into a device scratch and one
kernel places the rows. That scratch is the third planned RUNTIME-zone buffer and follows the g6yk shape exactly:

  * a planned term: populate_host_zone_sizing publishes it per device beside the MoE CONTROL requirement, from
    moe_host_scatter_scratch_bytes_for_device(), counting only tensors that keep experts on the host and only on
    the device whose layer runs them (an all-VRAM placement plans, claims and reports nothing); the RUNTIME zone
    requirement folds it in, and so does the dense-scratch fit check, among the consumers that do not follow
    n_ubatch;
  * a claim: the runtime-context transaction claims it at its whole plan on the publish path (after the commit and
    the Q8_1 claim, before the hold is recomputed), through the shared RUNTIME allocator (spill forbidden), and the
    ring re-plan before it is charged the part not yet held;
  * no spill, no estimate: the scatter never allocates. The producers bind the claimed scratch to the pending slot,
    the flush plans its chunks against the bytes the claim holds, and a flush larger than the plan goes through in
    chunks. When nothing was claimed the flush makes the per-run copies, which is slower and equally correct, so a
    failed claim is reported and never fatal, and a decline is reported once per reason;
  * no per-flush allocation: the flush builds its rows, plan and event lists in one thread-local workspace and
    retains its handles once.

Every claim is checked on COMMENT-STRIPPED, whitespace-normalized text and has a mutant that must make it fail.

Pytest-style (module-level test_* functions): register with llama_test_pytest.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_LEXEME_RE = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'",
    re.DOTALL,
)


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def norm(text: str) -> str:
    """Comment-stripped, whitespace-canonical text, so a clang-format re-wrap cannot turn a claim red."""
    t = re.sub(r"\s+", " ", strip_comments(text))
    t = re.sub(r"\( ", "(", t)
    t = re.sub(r" \)", ")", t)
    t = re.sub(r" ,", ",", t)
    t = re.sub(r",(?! )", ", ", t)
    return t


def read(rel: str) -> str:
    return (ROOT / rel).read_text()


SYCL = read("ggml/src/ggml-sycl/ggml-sycl.cpp")
COMMON = read("ggml/src/ggml-sycl/common.hpp")
CACHE = read("ggml/src/ggml-sycl/unified-cache.cpp")
HDR = read("ggml/src/ggml-sycl/moe-host-scatter.hpp")


def body(text: str, signature: str) -> str:
    """The brace-balanced body of the first definition whose normalized text starts with `signature` (string
    literals are skipped while balancing). Empty when there is none."""
    at = text.find(signature)
    if at < 0:
        return ""
    open_at = text.find("{", at + len(signature))
    if open_at < 0:
        return ""
    depth = 0
    i = open_at
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "\"'":
            j = i + 1
            while j < n and text[j] != ch:
                j += 2 if text[j] == "\\" else 1
            i = j + 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_at:i + 1]
        i += 1
    return ""


def code_only(text: str) -> str:
    """String literals blanked, so a WARN's text cannot satisfy or trip a check on code."""
    return re.sub(r"\"(?:\\.|[^\"\\])*\"", '""', text)


TXN_SIG = "static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction("
CLAIM_SIG = "static void ggml_sycl_moe_host_scatter_claim_plan(ggml_backend_sycl_context & ctx)"
ENSURE_SIG = "inline void * ggml_sycl_runtime_scratch_ensure("
REQ_SIG = "bool unified_cache_get_planned_runtime_zone_requirement(int device_id, size_t * out)"
FIT_SIG = "bool unified_cache_dense_scratch_runtime_fit(int device_id,"
SIZING_SIG = "static void populate_host_zone_sizing(placement_plan & plan,"
STRUCT_SIG = "struct moe_host_scatter_scratch_t"
COMPACT_SIG = "static bool moe_host_scatter_submit_compact(pending_cpu_scatter * const * slots,"
FLUSH_SIG = "static void flush_pending_cpu_scatter_slot(pending_cpu_scatter & slot)"
ROWS_SIG = "static void moe_host_scatter_rows_collect(pending_cpu_scatter * const * slots,"
FOR_DEVICE_SIG = "inline bool moe_host_scatter_scratch_bytes_for_device("
DECLINE_FIRST_SIG = "inline bool moe_scatter_decline_first(std::atomic<uint32_t> & seen, moe_scatter_decline why)"
BIND_SIG = "static void pending_cpu_scatter_bind_scratch(pending_cpu_scatter & slot, ggml_backend_sycl_context & ctx)"

# A by-value std::vector declaration (a container the function owns, so it allocates); references are not matched.
VECTOR_LOCAL_RE = re.compile(r"std::vector<[^;&]*?>\s+(\w+)\s*[;{=(]")
FATAL_RE = re.compile(r"\b(GGML_ABORT|GGML_ASSERT|abort|assert)\s*\(")
ALLOC_RE = re.compile(r"\b(ensure_buffer|unified_allocate\w*|unified_alloc|allocate_managed\w*|malloc\w*|"
                      r"ggml_sycl_runtime_scratch_ensure)\s*[<(]")
RAW_ALLOC_RE = re.compile(r"\b(unified_allocate\w*|unified_alloc|allocate_managed\w*|malloc\w*)\s*[<(]")


# ---- (a) the planned term -----------------------------------------------------------------------------------------


def claim_term_is_published_at_plan_time(cache: str, hdr: str = HDR) -> bool:
    """populate_host_zone_sizing sizes the scratch per device with moe_host_scatter_scratch_bytes_for_device() and
    publishes it for every device of the plan (or the plan's one device), the same devices the CONTROL requirement
    is published for."""
    sizing = body(norm(cache), SIZING_SIG)
    if not sizing:
        return False
    devices = sizing.find("std::vector<int> devices = plan.devices; if (devices.empty()) { devices.push_back(plan.device_id); }")
    sized = sizing.find("if (!moe_host_scatter_scratch_bytes_for_device(scattered, device, static_cast<size_t>(n_expert_used), "
                        "&moe_host_scatter_bytes)) {", devices)
    each = sizing.find("unified_cache_set_planned_moe_host_scatter_scratch_bytes(device, moe_host_scatter_bytes);", sized)
    return min(devices, sized, each) >= 0 and bool(body(norm(hdr), FOR_DEVICE_SIG))


def claim_term_counts_only_host_experts(cache: str, hdr: str = HDR) -> bool:
    """Only a tensor that keeps experts on the host contributes, and only on the device whose layer runs it: an
    all-VRAM placement plans 0, so nothing is claimed and nothing warns. An expert without an entry counts as
    host-resident, as at dispatch."""
    sizing = body(norm(cache), SIZING_SIG)
    for_device = body(norm(hdr), FOR_DEVICE_SIG)
    return (bool(sizing) and bool(for_device)
            and "if (!e.on_device) { r.any_host = true; } else if (e.expert_id < 0) { r.whole_on_device = true; } "
                "else { r.n_on_device++; }" in sizing
            and "t.has_host_experts = it == residency.end() || it->second.any_host || "
                "(!it->second.whole_on_device && it->second.n_on_device < n_experts);" in sizing
            and "t.device = plan.devices.empty() ? plan.device_id : plan.get_layer_device(layer);" in sizing
            and "if (!t.has_host_experts || (t.device >= 0 && t.device != device)) { continue; }" in for_device)


def claim_zone_requirement_folds_in_the_term(cache: str) -> bool:
    """The RUNTIME zone requirement adds the term (checked) before the dense plans, and the zone sizing reads it."""
    t = norm(cache)
    req = body(t, REQ_SIG)
    read_at = req.find("const size_t moe_host_scatter = unified_cache_get_planned_moe_host_scatter_scratch_bytes(device_id);")
    check_at = req.find("if (moe_host_scatter > SIZE_MAX - base) { return false; }", read_at)
    add_at = req.find("base += moe_host_scatter;", check_at)
    out_at = req.find("*out = base + mmq_src1 + dequant_f16;", add_at)
    return (min(read_at, check_at, add_at, out_at) >= 0
            and re.search(r"unified_cache_get_planned_runtime_zone_requirement\(dev_id, &planned_runtime_scratch\).*?"
                          r"runtime_zone = planned_runtime_scratch;", t) is not None)


def claim_fit_counts_the_term(cache: str) -> bool:
    """The dense-scratch fit check counts the term among the RUNTIME consumers that do not follow n_ubatch, so a
    rung is not admitted into room the scratch already owns."""
    fit = body(norm(cache), FIT_SIG)
    read_at = fit.find("const size_t moe_host_scatter = unified_cache_get_planned_moe_host_scatter_scratch_bytes(device_id);")
    add_at = fit.find("other = moe_host_scatter > SIZE_MAX - other ? SIZE_MAX : other + moe_host_scatter;", read_at)
    fits_at = fit.find("const bool fits = other <= zone_bytes && need <= zone_bytes - other;", add_at)
    return min(read_at, add_at, fits_at) >= 0


# ---- (b) the claim at publish -------------------------------------------------------------------------------------


def claim_transaction_claims_after_commit_before_hold(sycl: str) -> bool:
    """The publish path claims the scratch as the statement right after the Q8_1 claim (itself right after the
    commit) and before the hold is recomputed: once, unconditionally, never on a probe."""
    txn = body(norm(sycl), TXN_SIG)
    if not txn:
        return False
    commit = txn.find("dense_guard.commit();")
    q8 = txn.find("ggml_sycl_mmq_src1_claim_plan(*ctx);", commit)
    claim = txn.find("ggml_sycl_moe_host_scatter_claim_plan(*ctx);")
    if min(commit, q8, claim) < 0:
        return False
    hold = txn.find("ggml_sycl_planned_scratch_hold_refresh(*ctx);", claim)
    between = txn[q8 + len("ggml_sycl_mmq_src1_claim_plan(*ctx);"):claim]
    return (commit < q8 < claim < hold and between.strip() == ""
            and txn.count("ggml_sycl_moe_host_scatter_claim_plan(") == 1)


def claim_helper_claims_the_whole_plan(sycl: str) -> bool:
    """The helper sizes the claim from the published plan, skips a scratch already at plan, claims through the
    context's own scratch on its in-order stream, and reports a failed claim instead of aborting."""
    helper = body(norm(sycl), CLAIM_SIG)
    if not helper:
        return False
    m = re.search(r"const size_t (\w+) = ggml_sycl::unified_cache_get_planned_moe_host_scatter_scratch_bytes\((\w+)\);",
                  helper)
    if not m:
        return False
    planned, dev = m.group(1), m.group(2)
    skip = f"if ({planned} == 0 || ctx.moe_host_scatter_scratch.capacity({dev}) >= {planned}) {{ return; }}"
    ensure = f"if (ctx.moe_host_scatter_scratch.ensure_buffer({planned}, {dev}, *ctx.stream({dev}, 0)) != nullptr)"
    skip_at = helper.find(skip, m.end())
    ensure_at = helper.find(ensure, skip_at + len(skip)) if skip_at >= 0 else -1
    if min(skip_at, ensure_at) < 0:
        return False
    code = code_only(helper)
    warn_at = helper.find("GGML_LOG_WARN(")
    return ("return" not in helper[:m.start()] and "return" not in helper[skip_at + len(skip):ensure_at]
            and warn_at > ensure_at and FATAL_RE.search(code) is None and "malloc" not in code)


def claim_scratch_allocates_in_the_runtime_zone(common: str) -> bool:
    """The context's scratch allocates only through the shared planned-RUNTIME allocator, which is RUNTIME zone,
    spill forbidden, unified-cache owned."""
    t = norm(common)
    struct = body(t, STRUCT_SIG)
    ensure = body(t, ENSURE_SIG)
    return (bool(struct) and bool(ensure)
            and "ggml_sycl_runtime_scratch_ensure<ggml_sycl_moe_host_scatter_retire_marker_kernel>(s.backing_handle, "
                "s.backing_capacity, required_size, device, queue, \"moe-host-scatter\", &grew);" in struct
            and code_only(struct).count("ggml_sycl_runtime_scratch_ensure<") == 1
            and RAW_ALLOC_RE.search(code_only(struct)) is None
            and "ggml_sycl::unified_allocate_owner(req)" in ensure
            and "req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::RUNTIME;" in ensure
            and "req.intent.constraints.forbid_vram_zone_spill = true;" in ensure)


def claim_ring_is_charged_the_unheld_scratch(sycl: str) -> bool:
    """The ring re-plan, which runs before the claim, leaves the RUNTIME zone room for the part not yet held."""
    txn = body(norm(sycl), TXN_SIG)
    planned = txn.find("const size_t moe_host_scatter_planned = "
                       "ggml_sycl::unified_cache_get_planned_moe_host_scatter_scratch_bytes(ctx->device);")
    held = txn.find("const size_t moe_host_scatter_held = ctx->moe_host_scatter_scratch.capacity(ctx->device);", planned)
    charged = txn.find("ring_kv_zone.runtime_pending_bytes += moe_host_scatter_planned > moe_host_scatter_held ? "
                       "moe_host_scatter_planned - moe_host_scatter_held : 0;", held)
    ring = txn.find("ggml_sycl_replan_pp_moe_onednn_ring(ctx->device, next_kv_info.n_ubatch, probe_mode, &ring_kv_zone)",
                    charged)
    return min(planned, held, charged, ring) >= 0


# ---- (c) no spill, no estimate: the scatter draws only what was claimed ------------------------------------------


def claim_producers_bind_the_claimed_scratch(sycl: str) -> bool:
    """Both producers of a pending scatter bind the context's claimed scratch to the slot, and the binding reads the
    scratch's handle and the bytes it holds, nothing else."""
    t = norm(sycl)
    bind = body(t, BIND_SIG)
    return (bool(bind)
            and "slot.scatter_scratch = ctx.moe_host_scatter_scratch.handle(ctx.device);" in bind
            and "slot.scatter_scratch_bytes = ctx.moe_host_scatter_scratch.capacity(ctx.device);" in bind
            and ALLOC_RE.search(code_only(bind)) is None
            and t.count("pending_cpu_scatter_bind_scratch(g_pending_scatter, ctx);") == 2)


def claim_compact_scatter_never_allocates(sycl: str) -> bool:
    """The compact scatter plans against the bytes the claim holds (never an estimate of its own) and allocates
    nothing: no ensure, no unified allocation, no malloc."""
    compact = body(norm(sycl), COMPACT_SIG)
    return (bool(compact)
            and "!ggml_sycl::moe_scatter_plan_build(rows, lead.scatter_scratch_bytes, &ws.plan)" in compact
            and "const ggml_sycl::resolved_ptr scratch = lead.scatter_scratch.resolve(device);" in compact
            and "ggml_sycl::mem_copy_async(lead.scatter_scratch, copy.scratch_offset, srcs[copy.src], "
                "copy.src_offset, copy.bytes, queue)" in compact
            and ALLOC_RE.search(code_only(compact)) is None)


def claim_flush_takes_the_compact_path(sycl: str) -> bool:
    """The per-slot flush submits the compact scatter and makes the per-run copies only when it declines."""
    flush = body(norm(sycl), FLUSH_SIG)
    return (bool(flush)
            and "if (!moe_host_scatter_submit_compact(group, n_group, ws, &n_copies, &why)) {" in flush
            and ALLOC_RE.search(code_only(flush)) is None)


def claim_decline_reported_once_per_reason(sycl: str, hdr: str = HDR) -> bool:
    """The fallback WARN is latched per reason, not per process: a later decline for a different reason is still
    reported, and none is fatal."""
    flush = body(norm(sycl), FLUSH_SIG)
    first = body(norm(hdr), DECLINE_FIRST_SIG)
    latch = flush.find("static std::atomic<uint32_t> reported{ 0 }; "
                       "if (ggml_sycl::moe_scatter_decline_first(reported, why)) { GGML_LOG_WARN(")
    return (latch >= 0 and "ggml_sycl::moe_scatter_decline_name(why)" in flush[latch:]
            and "const uint32_t bit = 1u << static_cast<uint32_t>(why);" in first
            and "return (seen.fetch_or(bit, std::memory_order_relaxed) & bit) == 0;" in first)


def claim_flush_reuses_one_workspace(sycl: str) -> bool:
    """A decode flush allocates nothing of its own once the thread's workspace is sized: the rows, the plan, the
    handle lists, the copy events and the flush's events live in one thread-local workspace, the only container a
    flush builds is the retention list the retention API takes by value, and the flush retains once, after its last
    kernel, not once per chunk."""
    t = norm(sycl)
    compact = body(t, COMPACT_SIG)
    rows = body(t, ROWS_SIG)
    flush = body(t, FLUSH_SIG)
    if not (compact and rows and flush):
        return False
    owned = (VECTOR_LOCAL_RE.findall(code_only(compact)), VECTOR_LOCAL_RE.findall(code_only(rows)),
             VECTOR_LOCAL_RE.findall(code_only(flush)))
    last = compact.find("placed_last = placed; }")
    retain = compact.find("ggml_sycl::retain_handles_until_event(std::move(retained), placed_last);")
    return (owned == (["retained"], [], [])
            and "static thread_local moe_host_scatter_workspace g_moe_host_scatter_ws;" in t
            and "moe_host_scatter_workspace & ws = g_moe_host_scatter_ws; ws.events.clear();" in flush
            and "std::vector<sycl::event> & copied = ws.copied; copied.clear();" in compact
            and compact.count("retain_handles_until_event(") == 1 and 0 <= last < retain)


def test_the_term_is_published_at_plan_time():
    assert claim_term_is_published_at_plan_time(CACHE)


def test_the_term_counts_only_host_experts():
    assert claim_term_counts_only_host_experts(CACHE)


def test_a_decline_is_reported_once_per_reason():
    assert claim_decline_reported_once_per_reason(SYCL)


def test_the_flush_reuses_one_workspace():
    assert claim_flush_reuses_one_workspace(SYCL)


def test_the_runtime_zone_requirement_folds_in_the_term():
    assert claim_zone_requirement_folds_in_the_term(CACHE)


def test_the_dense_fit_counts_the_term():
    assert claim_fit_counts_the_term(CACHE)


def test_the_publish_claims_the_scratch_after_the_commit_and_before_the_hold():
    assert claim_transaction_claims_after_commit_before_hold(SYCL)


def test_the_claim_is_the_whole_plan_and_a_failed_claim_is_reported():
    assert claim_helper_claims_the_whole_plan(SYCL)


def test_the_scratch_allocates_in_the_runtime_zone_with_spill_forbidden():
    assert claim_scratch_allocates_in_the_runtime_zone(COMMON)


def test_the_ring_is_charged_the_unheld_scratch():
    assert claim_ring_is_charged_the_unheld_scratch(SYCL)


def test_the_producers_bind_the_claimed_scratch():
    assert claim_producers_bind_the_claimed_scratch(SYCL)


def test_the_compact_scatter_never_allocates():
    assert claim_compact_scatter_never_allocates(SYCL)


def test_the_flush_takes_the_compact_path():
    assert claim_flush_takes_the_compact_path(SYCL)


# ---- mutants: every claim must fail against the thing it forbids --------------------------------------------------


def _once(raw: str, old: str, new: str) -> str:
    assert raw.count(old) >= 1, f"mutant anchor not found: {old!r}"
    return raw.replace(old, new, 1)


_CLAIM_CALL = "    ggml_sycl_moe_host_scatter_claim_plan(*ctx);\n"
_HOLD_CALL = "    ggml_sycl_planned_scratch_hold_refresh(*ctx);\n    // This plan's own reserve"


def test_mutant_term_not_published_fails():
    assert not claim_term_is_published_at_plan_time(
        _once(CACHE, "unified_cache_set_planned_moe_host_scatter_scratch_bytes(device, moe_host_scatter_bytes);", ";"))


def test_mutant_term_not_sized_from_the_formula_fails():
    assert not claim_term_is_published_at_plan_time(
        _once(CACHE, "if (!moe_host_scatter_scratch_bytes_for_device(scattered, device,",
              "if (false && !moe_host_scatter_scratch_bytes_for_device(scattered, device,"))


def test_mutant_every_moe_tensor_counts_as_host_fails():
    assert not claim_term_counts_only_host_experts(
        _once(CACHE, "t.has_host_experts = it == residency.end() ||", "t.has_host_experts = true || it == residency.end() ||"))


def test_mutant_term_ignores_host_residency_fails():
    assert not claim_term_counts_only_host_experts(
        CACHE, _once(HDR, "if (!t.has_host_experts || (t.device >= 0 && t.device != device)) {",
                     "if (t.device >= 0 && t.device != device) {"))


def test_mutant_term_ignores_the_layer_device_fails():
    assert not claim_term_counts_only_host_experts(
        _once(CACHE, "plan.devices.empty() ? plan.device_id : plan.get_layer_device(layer);",
              "plan.devices.empty() ? plan.device_id : -1;"))


def test_mutant_claim_warns_when_nothing_is_planned_fails():
    assert not claim_helper_claims_the_whole_plan(
        _once(SYCL, CLAIM_SIG + " {\n", CLAIM_SIG + " {\n    GGML_LOG_WARN(\"nothing planned\\n\");\n"))


def test_mutant_decline_latched_once_per_process_fails():
    assert not claim_decline_reported_once_per_reason(
        _once(SYCL, "if (ggml_sycl::moe_scatter_decline_first(reported, why)) {", "if (!reported.exchange(1)) {"))


def test_mutant_decline_latch_ignores_the_reason_fails():
    assert not claim_decline_reported_once_per_reason(
        SYCL, _once(HDR, "return (seen.fetch_or(bit, std::memory_order_relaxed) & bit) == 0;",
                    "return seen.exchange(UINT32_MAX, std::memory_order_relaxed) == 0;"))


def test_mutant_flush_builds_its_own_event_list_fails():
    assert not claim_flush_reuses_one_workspace(
        _once(SYCL, "    moe_host_scatter_workspace & ws = g_moe_host_scatter_ws;\n",
              "    moe_host_scatter_workspace & ws = g_moe_host_scatter_ws;\n    std::vector<sycl::event> scatter_events;\n"))


def test_mutant_compact_builds_a_rows_vector_fails():
    assert not claim_flush_reuses_one_workspace(
        _once(SYCL, "    moe_host_scatter_rows_collect(slots, n_slots, ws);\n",
              "    std::vector<ggml_sycl::moe_scatter_row> spare;\n    moe_host_scatter_rows_collect(slots, n_slots, ws);\n"))


def test_mutant_retention_per_chunk_fails():
    assert not claim_flush_reuses_one_workspace(
        _once(SYCL, "        placed_last = placed;\n",
              "        placed_last = placed;\n        ggml_sycl::retain_handles_until_event({ lead.scatter_scratch }, placed);\n"))


def test_mutant_workspace_not_thread_local_fails():
    assert not claim_flush_reuses_one_workspace(
        _once(SYCL, "static thread_local moe_host_scatter_workspace g_moe_host_scatter_ws;",
              "static moe_host_scatter_workspace g_moe_host_scatter_ws;"))


def test_mutant_zone_requirement_without_the_term_fails():
    assert not claim_zone_requirement_folds_in_the_term(_once(CACHE, "base += moe_host_scatter;", ";"))


def test_mutant_zone_requirement_unchecked_add_fails():
    assert not claim_zone_requirement_folds_in_the_term(
        _once(CACHE, "if (moe_host_scatter > SIZE_MAX - base) {\n        return false;\n    }\n", ""))


def test_mutant_fit_without_the_term_fails():
    assert not claim_fit_counts_the_term(
        _once(CACHE, "other + moe_host_scatter;", "other;"))


def test_mutant_no_claim_fails():
    assert not claim_transaction_claims_after_commit_before_hold(_once(SYCL, _CLAIM_CALL, ""))


def test_mutant_claim_after_the_hold_refresh_fails():
    raw = _once(SYCL, _CLAIM_CALL, "")
    raw = _once(raw, _HOLD_CALL, "    ggml_sycl_planned_scratch_hold_refresh(*ctx);\n" + _CLAIM_CALL
                + "    // This plan's own reserve")
    assert not claim_transaction_claims_after_commit_before_hold(raw)


def test_mutant_claim_before_the_commit_fails():
    raw = _once(SYCL, _CLAIM_CALL, "")
    raw = _once(raw, "    dense_guard.commit();\n", _CLAIM_CALL + "    dense_guard.commit();\n")
    assert not claim_transaction_claims_after_commit_before_hold(raw)


def test_mutant_claim_only_on_the_probe_path_fails():
    assert not claim_transaction_claims_after_commit_before_hold(
        _once(SYCL, _CLAIM_CALL, "    if (probe_mode) {\n    " + _CLAIM_CALL + "    }\n"))


def test_mutant_claim_under_if_0_fails():
    assert not claim_transaction_claims_after_commit_before_hold(
        _once(SYCL, _CLAIM_CALL, "#if 0\n" + _CLAIM_CALL + "#endif\n"))


def test_mutant_claim_helper_returns_first_fails():
    assert not claim_helper_claims_the_whole_plan(
        _once(SYCL, CLAIM_SIG + " {\n", CLAIM_SIG + " {\n    return;\n"))


def test_mutant_claim_sized_by_an_estimate_fails():
    assert not claim_helper_claims_the_whole_plan(
        _once(SYCL, "ctx.moe_host_scatter_scratch.ensure_buffer(planned, d,",
              "ctx.moe_host_scatter_scratch.ensure_buffer(planned / 2, d,"))


def test_mutant_claim_that_aborts_fails():
    assert not claim_helper_claims_the_whole_plan(
        _once(SYCL, "    GGML_LOG_WARN(\n        \"[MOE-SCATTER] device %d: the planned host-expert scatter scratch",
              "    GGML_ABORT(\"claim failed\");\n    GGML_LOG_WARN(\n"
              "        \"[MOE-SCATTER] device %d: the planned host-expert scatter scratch"))


def test_mutant_scratch_spill_allowed_fails():
    assert not claim_scratch_allocates_in_the_runtime_zone(
        _once(COMMON, "req.intent.constraints.forbid_vram_zone_spill = true;\n    ggml_sycl::allocation_result",
              "req.intent.constraints.forbid_vram_zone_spill = false;\n    ggml_sycl::allocation_result"))


def test_mutant_scratch_with_its_own_allocation_fails():
    assert not claim_scratch_allocates_in_the_runtime_zone(
        _once(COMMON, "        void * ensure_buffer(size_t required_size, int device, sycl::queue & queue) {\n"
                      "            slot_t & s    = slot(device);\n            bool     grew = false;\n"
                      "            void *   ptr  = ggml_sycl_runtime_scratch_ensure<ggml_sycl_moe_host_scatter",
              "        void * ensure_buffer(size_t required_size, int device, sycl::queue & queue) {\n"
              "            slot_t & s    = slot(device);\n            bool     grew = false;\n"
              "            (void) sycl::malloc_device(required_size, queue);\n"
              "            void *   ptr  = ggml_sycl_runtime_scratch_ensure<ggml_sycl_moe_host_scatter"))


def test_mutant_ring_not_charged_fails():
    assert not claim_ring_is_charged_the_unheld_scratch(
        _once(SYCL, "        ring_kv_zone.runtime_pending_bytes +=\n            moe_host_scatter_planned >",
              "        (void)\n            moe_host_scatter_planned >"))


def test_mutant_one_producer_unbound_fails():
    assert not claim_producers_bind_the_claimed_scratch(
        _once(SYCL, "pending_cpu_scatter_bind_scratch(g_pending_scatter, ctx);", ";"))


def test_mutant_binding_reads_no_claim_fails():
    assert not claim_producers_bind_the_claimed_scratch(
        _once(SYCL, "slot.scatter_scratch_bytes = ctx.moe_host_scatter_scratch.capacity(ctx.device);",
              "slot.scatter_scratch_bytes = 1 << 20;"))


def test_mutant_compact_plans_against_an_estimate_fails():
    assert not claim_compact_scatter_never_allocates(
        _once(SYCL, "!ggml_sycl::moe_scatter_plan_build(rows, lead.scatter_scratch_bytes, &ws.plan)",
              "!ggml_sycl::moe_scatter_plan_build(rows, rows.size() * rows.front().bytes, &ws.plan)"))


def test_mutant_compact_grows_the_scratch_fails():
    assert not claim_compact_scatter_never_allocates(
        _once(SYCL, "    const ggml_sycl::resolved_ptr scratch = lead.scatter_scratch.resolve(device);\n",
              "    ggml_sycl::alloc_request grow{};\n    (void) ggml_sycl::unified_allocate_owner(grow);\n"
              "    const ggml_sycl::resolved_ptr scratch = lead.scatter_scratch.resolve(device);\n"))


def test_mutant_flush_bypasses_the_compact_path_fails():
    assert not claim_flush_takes_the_compact_path(
        _once(SYCL, "if (!moe_host_scatter_submit_compact(group, n_group, ws, &n_copies, &why)) {",
              "if (true) {"))


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
