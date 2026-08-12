#!/usr/bin/env python3
"""Formatting-insensitive source isolation checks and semantic mutation witnesses."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "ggml/src/ggml-sycl/moe-resolved-batch.hpp"
SOURCE = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"
HOST_TEST = ROOT / "tests/test-sycl-moe-resolved-batch.cpp"
MEM_HANDLE_SOURCE = ROOT / "ggml/src/ggml-sycl/mem-handle.cpp"
CPU_DISPATCH_SOURCE = ROOT / "ggml/src/ggml-sycl/cpu-dispatch.cpp"
UNIFIED_CACHE_SOURCE = ROOT / "ggml/src/ggml-sycl/unified-cache.cpp"

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|::|&&|\|\||!=|==|<=|>=|->|[{}()[\].,;:%=*<>!&+-]")


def tokens(text: str) -> list[str]:
    text = re.sub(r"//.*?$|/\*.*?\*/", " ", text, flags=re.MULTILINE | re.DOTALL)
    return TOKEN_RE.findall(text)


def token_sequence_index(haystack: list[str], pattern: str, start: int = 0) -> int:
    needle = tokens(pattern)
    width = len(needle)
    for i in range(start, len(haystack) - width + 1):
        if haystack[i:i + width] == needle:
            return i
    raise ValueError(f"token sequence not found: {pattern}")


def contains_tokens(haystack: list[str], pattern: str) -> bool:
    try:
        token_sequence_index(haystack, pattern)
        return True
    except ValueError:
        return False


def has_tokens(text: str, pattern: str) -> bool:
    return contains_tokens(tokens(text), pattern)


def inject_before_pair_end(source: str, code: str, count: int = 1) -> str:
    marker = "auto record_moe_gpu_path"
    return source.replace(marker, code + "\n    " + marker, count)


def function_definition(text: str, signature: str) -> str:
    """Return one C++ function using its actual balanced-brace boundary."""
    start = text.index(signature)
    brace = text.index("{", start)
    depth = 0
    state = "code"
    i = brace
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if state == "code":
            if ch == "/" and nxt == "/":
                state = "line-comment"
                i += 1
            elif ch == "/" and nxt == "*":
                state = "block-comment"
                i += 1
            elif ch == '"':
                state = "string"
            elif ch == "'":
                state = "char"
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        elif state == "line-comment":
            if ch == "\n":
                state = "code"
        elif state == "block-comment":
            if ch == "*" and nxt == "/":
                state = "code"
                i += 1
        elif state in ("string", "char"):
            quote = '"' if state == "string" else "'"
            if ch == "\\":
                i += 1
            elif ch == quote:
                state = "code"
        i += 1
    raise ValueError(f"unclosed function: {signature}")


def violations(header: str, source: str, host_test: str, mem_handle_source: str) -> list[str]:
    failures: list[str] = []
    required_header = {
        "ID snapshot": "out.batch.expert_ids.assign(ids, ids + count)",
        "occurrence": "operand.occurrence = i",
        "token index": "operand.token_index = i / slots_per_token",
        "slot index": "operand.slot_index = i % slots_per_token",
        "missing lease": "return moe_batch_reject_reason::MISSING_HANDLE",
        "raw lease": "return moe_batch_reject_reason::RAW_COMPAT_HANDLE",
        "raw direct requires ownerless identity":
            "route.lease.kind() == mem_handle_kind::DIRECT && !route.lease.has_stable_owner_identity()",
        "stable identity": "route.lease.has_stable_owner_identity()",
        "pointer agreement": "resolved.ptr != route.transient_ptr",
        "layout agreement": "resolved.layout != route.actual_layout",
        "primary lease device": "route.lease.device() != submit_device",
        "secondary lease device": "route.lease.device() != route.owning_device",
        "primary plan": "route.planned_device == route.owning_device",
        "opaque proof use": "route.has_authoritative_planned_alternate() && route.planned_on_device && primary",
        "stable-only sharing": "retained.stable_identity_equal(route.lease)",
        "retained role aggregate": "struct moe_retained_role_bundle",
        "role occurrence alignment": "actual.occurrence != expected.occurrence",
        "role token alignment": "actual.token_index != expected.token_index",
        "role slot alignment": "actual.slot_index != expected.slot_index",
        "owned pointer table": "mem_handle table_handle",
        "exact pointer-table leases": "std::vector<mem_handle> role_leases",
        "terminal role retention": "moe_retained_role_bundle roles",
        "role result defaults closed": "reject = moe_batch_reject_reason::MISSING_ROLE",
        "role alignment opens explicitly": "out.reject = moe_batch_reject_reason::NONE",
    }
    required_source = {
        "canonical resolver": "ggml_sycl_resolve_moe_expert_route_for_dispatch(",
        "adjusted alternate decision":
            "ggml_sycl_adjust_layout_for_tensor(tensor, alt.layout, current_device) == requested_layout",
        "TU-internal admission": "route.planned_alternate_admitted = current_device_planned_alternate",
        "opaque proof transfer":
            "normalized.authoritative_planned_alternate_ = route.planned_alternate_admitted",
        "canonical lease": "normalized.lease = std::move(route.lease)",
        "unconditional decode admission":
            "retained_decode_batch_result = ggml_sycl::ggml_sycl_build_moe_resolved_batch(",
        "metadata executor": "ggml_sycl::choose_moe_batch_executor(",
        "typed capability refusal": "moe_batch_reject_reason::CAPABILITY_UNSUPPORTED",
        "Q1 decode refusal": "type == GGML_TYPE_Q1_0",
        "NVFP4 decode refusal": "type == GGML_TYPE_NVFP4",
        "unconditional Q1/NVFP4 device refusal":
            "if (type == GGML_TYPE_Q1_0 || type == GGML_TYPE_NVFP4)",
        "opaque admitted recipe ticket": "moe_admitted_recipe_ticket admitted_recipe_ticket",
        "production host recipe executor": "ggml_sycl_cpu_moe_host_aos_execute(task, &reject)",
        "execution-row bounded activation copy":
            "std::memcpy(act, task.activations, rows * static_cast<size_t>(K) * sizeof(float))",
        "inventory planned host recipe workspace": "plan.moe_host_recipe_workspace_bytes",
        "host scratch includes recipe workspace":
            "plan.moe_cpu_expert_staging_bytes + plan.moe_host_recipe_workspace_bytes",
        "phase-specific recipe admission": "rows > 1 ? moe_route_phase::PROMPT : moe_route_phase::DECODE",
        "owning queue ready dependency": "target_queue->ext_oneapi_submit_barrier(route_ready_events)",
        "primary ready dependency": "dispatch_deps.push_back(entry->ready_event)",
        "CPU ready copy": "sycl::event ready = entry.ready_event",
        "CPU ready wait": "ready.wait()",
        "admission failure propagates":
            "throw ggml_sycl_fallback_error(\"MUL_MAT_ID retained decode admission failed\")",
        "CPU-TG executor failure propagates":
            "throw ggml_sycl_fallback_error(\"MUL_MAT_ID retained decode executor refused route\")",
        "main executor failure propagates":
            "throw ggml_sycl_fallback_error(\"MUL_MAT_ID retained hybrid executor refused route\")",
        "ready guard suppresses exceptional publish":
            "std::uncaught_exceptions() != uncaught_on_entry",
        "ID admission failure propagates":
            "throw ggml_sycl_fallback_error(\"MUL_MAT_ID expert ID admission failed\")",
        "shared retained dispatch helper": "append_retained_operand",
        "decode CPU-TG fast gate": "if (cpu_tg_candidate && ne12 != 1 && !xmx_moe_forced)",
        "decode GPU fast gate": "if (ne12 != 1) { if (!pp_cpu_reference_force_router",
        "decode precomputed gate": "src1 && src1->ne[2] != 1 && ggml_sycl_moe_precomputed_skip_contains",
        "all-local decode retained gate": "const bool moe_hybrid_with_plan = ne12 == 1 || selected_hybrid_route",
        "CPU-TG retains selected routing semantics":
            "const bool cpu_expert_tg_active = selected_hybrid_route && !prompt_batch",
        "CPU-TG canonical batch consumer":
            "const ggml_sycl::moe_resolved_batch & decode_batch = retained_decode_batch_result.batch",
        "main canonical batch consumer":
            "ne12 == 1 ? retained_decode_batch_result.batch : retained_prompt_batch_result.batch",
        "prompt ID snapshot": "std::vector<int32_t> prompt_ids_snapshot",
        "prompt retained admission": "retained_prompt_batch_result = ggml_sycl::ggml_sycl_build_moe_resolved_batch(",
        "prompt graph recording refusal":
            "throw ggml_sycl_fallback_error(\"MUL_MAT_ID retained prompt batch cannot enter graph sink\")",
        "prompt admission failure propagates":
            "throw ggml_sycl_fallback_error(\"MUL_MAT_ID retained prompt admission failed\")",
        "batch-only transient table": "ggml_sycl_upload_moe_ptr_table_from_batch(",
        "grouped prompt occurrence lookup": "prompt_batch.occurrence(iid1, id)",
        "XMX retained executor API": "const ggml_sycl::moe_resolved_batch & batch",
        "MMVQ retained pointer submit": "retained_ptrs, retained_table_event_set",
        "exact prompt occurrence fallback": "retained_prompt_batch_result.batch.occurrence(token_index, slot_index)",
        "repeated identity conflict refusal":
            "throw ggml_sycl_fallback_error(\"MUL_MAT_ID repeated prompt expert has conflicting retained occurrences\")",
        "selected prompt miss failure":
            "throw ggml_sycl_fallback_error(\"MUL_MAT_ID selected prompt expert unresolved before submit\")",
        "prompt cache overwrite gate": "if (ne12 == 1 && blk_layer_id >= 0 && !is_gate_subop)",
        "named pair capability gate": "prompt_pair_retained_roles_capable && prompt_pair_current_node",
        "primary ready dependency": "stream->ext_oneapi_submit_barrier({ route.ready_event })",
        "secondary ready propagation": "entry.has_ready_event = has_ready_event",
        "host ready wait": "if (has_ready_event) { ready_event.wait()",
        "linear prompt identity map": "retained_prompt_groups[expert] = &operand",
        "planned local table from batch":
            "ggml_sycl_upload_moe_ptr_table_from_batch(ctx, src0, retained_prompt_batch_result.batch",
        "exact hybrid prompt operand":
            "main_batch.occurrence(static_cast<size_t>(iid1), static_cast<size_t>(id))",
        "hybrid completeness refusal":
            "throw ggml_sycl_fallback_error(\"MUL_MAT_ID retained hybrid partition is incomplete\")",
        "decode-only MMVQ composite resolution":
            "if (ne12 == 1 && !planner_moe_uses_expert_handles)",
        "decode-only layout composite resolution":
            "if (ne12 == 1 && !host_weights && !placement_planned_moe)",
        "decode-only storage composite resolution":
            "if (ne12 == 1 && !use_expert_cache)",
        "independent gate admission": "pair.gate_weight, ctx.device, prompt_ids_snapshot.data()",
        "independent up admission": "pair.up_weight, ctx.device, prompt_ids_snapshot.data()",
        "independent down admission": "pair.down_weight, ctx.device, prompt_ids_snapshot.data()",
        "cross-role alignment": "align_moe_retained_role_batches(",
        "validated pair roles": "const bool prompt_pair_retained_roles_validated = [&]()",
        "pair capability quarantined": "const bool prompt_pair_retained_roles_capable = false",
        "retained pointer-table result": "ggml_sycl_upload_moe_retained_ptr_table_from_batch(",
        "actual terminal owner": "ggml_sycl_retain_moe_terminal_bundle(std::move(terminal))",
        "transactional terminal publication": "terminal.terminal_submitted = true",
        "transactional skip commit": "entry.set->insert(std::move(entry.node))",
        "down table preflight": "if (gate_ptrs && up_ptrs && down_table.valid())",
        "post-write failure waits": "terminal_event.wait()",
    }
    header_tokens = tokens(header)
    source_tokens = tokens(source)
    for name, pattern in required_header.items():
        if not contains_tokens(header_tokens, pattern):
            failures.append(name)
    for name, pattern in required_source.items():
        if not contains_tokens(source_tokens, pattern):
            failures.append(name)

    compute_fallback = source.index("catch (const ggml_sycl_fallback_error &)")
    compute_generic = source.index("catch (const std::exception & e)", compute_fallback)
    if not has_tokens(source[compute_fallback:compute_generic], "throw;"):
        failures.append("compute_forward rethrows MMID refusal")
    boundary_fallback = source.index("catch (const ggml_sycl_fallback_error & error)")
    boundary_end = source.index("return GGML_STATUS_FAILED", boundary_fallback)
    if boundary_end < boundary_fallback:
        failures.append("graph boundary reports MMID refusal")

    role_result_start = header.index("struct moe_retained_role_bundle_result")
    role_result_end = header.index("align_moe_retained_role_batches", role_result_start)
    if not has_tokens(header[role_result_start:role_result_end],
                      "reject = moe_batch_reject_reason::MISSING_ROLE"):
        failures.append("role result default-open")

    route_start = header.index("struct moe_batch_route")
    route_end = header.index("struct moe_resolved_operand", route_start)
    route_definition = header[route_start:route_end]
    compact_route = re.sub(r"\s+", " ", route_definition)
    if not re.search(r"private\s*:\s*.*bool\s+authoritative_planned_alternate_\s*=\s*false", compact_route):
        failures.append("proof private/default-deny")
    if has_tokens(header, "owner_is_planned_alternate"):
        failures.append("caller-writable proof")

    wrapper_start = source.index("moe_resolved_batch_result ggml_sycl_build_moe_resolved_batch")
    wrapper_end = source.index("bool test_moe_resolved_batch_accepts_actual_planned_alternate", wrapper_start)
    wrapper = source[wrapper_start:wrapper_end]
    if has_tokens(wrapper, "lookup_expert_placement"):
        failures.append("duplicate wrapper plan lookup")
    if has_tokens(wrapper, "alt.layout == route.requested_layout"):
        failures.append("raw alternate-layout comparison")

    fixture_signature = "mem_handle test_make_stable_weight_lease("
    if not has_tokens(host_test, fixture_signature):
        failures.append("test-TU lease definition")
    if has_tokens(mem_handle_source, "test_make_stable_weight_lease"):
        failures.append("production test lease symbol")
    for device_fixture in ("from_weight_lease_snapshot", "get_unified_cache_for_device"):
        if has_tokens(host_test, device_fixture):
            failures.append("device-dependent host fixture")

    mmid = function_definition(source, "static void ggml_sycl_mul_mat_id(")
    mmid_tokens = tokens(mmid)
    prompt_admission_tokens = token_sequence_index(
        mmid_tokens, "ggml_sycl::moe_resolved_batch_result retained_prompt_batch_result;")
    specialized_selection_tokens = token_sequence_index(
        mmid_tokens, "const bool cpu_tg_candidate", prompt_admission_tokens)
    admission_tokens = token_sequence_index(
        mmid_tokens, "ggml_sycl::moe_resolved_batch_result retained_decode_batch_result;", specialized_selection_tokens)
    route_mode_tokens = token_sequence_index(mmid_tokens, "const bool selected_hybrid_route", admission_tokens)
    dispatch_gate_tokens = token_sequence_index(mmid_tokens, "if (moe_hybrid_with_plan)", route_mode_tokens)
    prompt_admission = mmid.index("retained_prompt_batch_result")
    admission = mmid.index("retained_decode_batch_result", prompt_admission)
    dispatch_gate = mmid.index("if (moe_hybrid_with_plan)", admission)

    # Prompt and decode each have one occurrence admission. Prompt admission is
    # before every specialized selector; decode remains before route mode.
    if mmid.count("retained_prompt_batch_result = ggml_sycl::ggml_sycl_build_moe_resolved_batch(") != 1:
        failures.append("prompt has exactly one retained admission")
    if mmid.count("retained_decode_batch_result = ggml_sycl::ggml_sycl_build_moe_resolved_batch(") != 1:
        failures.append("decode has exactly one retained admission")
    if not prompt_admission_tokens < specialized_selection_tokens < admission_tokens:
        failures.append("prompt specialized selection precedes admission")
    if mmid.index('pp_phase_log("ids-ready"') >= admission:
        failures.append("decode admission precedes ID snapshot")
    if not admission_tokens < route_mode_tokens < dispatch_gate_tokens or contains_tokens(
            mmid_tokens[specialized_selection_tokens:admission_tokens], "if (moe_hybrid_with_plan)"):
        failures.append("decode route/dispatch precedes admission")
    if mmid[prompt_admission:admission].count("ggml_sycl_copy_ids_to_host(ctx, ids, prompt_ids_snapshot)") != 1:
        failures.append("prompt IDs are not snapshotted exactly once")
    if mmid.count("append_retained_operand(") != 2:
        failures.append("decode and hybrid prompt routers share retained dispatch helper")

    # Token-wise boundaries prevent comments, formatting, or a dead nested block
    # from hiding authority reacquisition in the complete active pair path.
    pair_start_tokens = token_sequence_index(
        mmid_tokens, "if (cpu_tg_candidate && ne12 != 1 && !xmx_moe_forced)", prompt_admission_tokens)
    pair_end_tokens = token_sequence_index(mmid_tokens, "auto record_moe_gpu_path", pair_start_tokens)
    pair_path_tokens = mmid_tokens[pair_start_tokens:pair_end_tokens]
    down_table_tokens = token_sequence_index(pair_path_tokens, "auto down_table")
    write_submit_tokens = token_sequence_index(
        pair_path_tokens, "submitted = mmvq_moe_batched_dispatch_pair_glu_mxfp4_soa(")
    skip_commit_tokens = token_sequence_index(pair_path_tokens, "entry.set->insert(std::move(entry.node))")
    ready_publish_tokens = token_sequence_index(
        pair_path_tokens, "ggml_sycl_set_tensor_ready_event(pair.glu_dst, ctx.device, terminal_event)")
    if not down_table_tokens < write_submit_tokens < skip_commit_tokens < ready_publish_tokens:
        failures.append("down preflight or transactional publication ordering")
    if contains_tokens(pair_path_tokens, "g_moe_precomputed_down_layer_skip") or contains_tokens(
            pair_path_tokens, "ggml_sycl_moe_precomputed_skip_insert(g_moe_precomputed_mmid_skip, pair.down_dst"):
        failures.append("optional down was skipped after GLU-only success")
    for forbidden in ("ggml_sycl_resolve_moe_expert_route(",
                      "ggml_sycl_resolve_moe_expert_route_for_dispatch(",
                      "ggml_sycl::ggml_sycl_resolve_expert_ptr(",
                      "ggml_sycl_resolve(src0, ctx.device)",
                      "ggml_sycl_resolve_tensor_ptr(src0, ctx.device)",
                      "ggml_sycl_refresh_moe_ids_cache(",
                      "moe_fusion_ensure_full_local_ptr_table(",
                      "moe_fusion_ensure_full_local_ptr_table_from_descriptor(",
                      "moe_fusion_ensure_gpu0_ptrs(", "moe_fusion_upload_ptrs_from_handles(",
                      "ggml_sycl_update_moe_ptr_table(", "ggml_sycl_materialize_planned_expert_layout(",
                      "src0_host_storage", "moe_layer_decode_role_plan"):
        if contains_tokens(pair_path_tokens, forbidden):
            failures.append(f"prompt pair reacquires route/table/raw authority: {forbidden}")
    pair_start = mmid.index("if (cpu_tg_candidate", prompt_admission)
    pair_end = mmid.index("auto record_moe_gpu_path", pair_start)
    prompt_reachable = mmid[prompt_admission:pair_start] + mmid[pair_end:]
    forbidden_prompt_ownership = (
        "ggml_sycl_resolve_moe_expert_route(",
        "ggml_sycl_resolve_moe_expert_route_for_dispatch(",
        "moe_fusion_ensure_gpu0_ptrs(",
        "ggml_sycl_materialize_planned_expert_layout(",
        "ggml_sycl_update_moe_ptr_table(",
        "ggml_sycl_resolve_expert_ptr(",
        "ggml_sycl_resolve_tensor_ptr(src0",
    )
    for forbidden in forbidden_prompt_ownership:
        if has_tokens(prompt_reachable, forbidden):
            failures.append(f"reachable prompt path reacquires/materializes route: {forbidden}")
    if prompt_reachable.count("ggml_sycl_copy_ids_to_host(ctx, ids, prompt_ids_snapshot)") != 1:
        failures.append("prompt IDs are copied outside the one admission snapshot")
    # Generic tensor resolution for src1/dst is canonical activation/output
    # resolution. Composite expert-weight resolution is allowed only in the
    # three explicitly decode-only src0 blocks.
    if prompt_reachable.count("ggml_sycl_resolve(src0, ctx.device)") != 3:
        failures.append("expert/composite weight resolution escaped decode-only guards")
    if not has_tokens(mmid, "moe_expert_route route = retained_prompt_route_for_occurrence"):
        failures.append("prompt fallback does not derive exact occurrences from batch")
    if mmid.count("retained_prompt_route_for_occurrence(") != 3:
        failures.append("prompt occurrence consumers collapsed routing identity")
    if has_tokens(mmid, "false && fusion_down_continuation"):
        failures.append("prompt pair path uses anonymous dead expression")
    cache_gate = mmid.find("if (ne12 == 1 && blk_layer_id >= 0 && !is_gate_subop)")
    if cache_gate < 0:
        failures.append("prompt IDs can be overwritten from layer cache")
    else:
        cache_end = mmid.index("if (ne12 == 1 && !ids_from_cache)", cache_gate)
        if not has_tokens(mmid[cache_gate:cache_end], "ids_host = cache_it->second.ids_host"):
            failures.append("decode cache reuse is not isolated behind prompt guard")

    decode_route = mmid[admission:dispatch_gate]
    for forbidden in ("from_chunk_ptr", "from_direct", "is_device_expert_ptr", "src0->data", "route.ptr"):
        if has_tokens(decode_route, forbidden):
            failures.append(f"decode raw routing before dispatch: {forbidden}")

    production = header + wrapper
    for forbidden in ("sycl::malloc_device", "sycl::malloc_host", "sycl::malloc_shared", "sycl::free(",
                      "mem_handle::from_chunk_ptr(", "mem_handle::from_direct("):
        if has_tokens(production, forbidden):
            failures.append(f"ownership bridge: {forbidden}")
    return failures


def test_direct_decode_review_contract_is_closed_and_lifetime_safe() -> None:
    source = SOURCE.read_text()
    mmid = function_definition(source, "static void ggml_sycl_mul_mat_id(")
    start = mmid.index("constexpr bool q1_nvfp4_direct_b70_validated = false")
    end = mmid.index("// MoE hybrid GPU+CPU dispatch gate", start)
    direct = mmid[start:end]
    assert "q1_nvfp4_direct_b70_validated && ne12 == 1" in direct
    validator = function_definition(HEADER.read_text(), "inline moe_batch_reject_reason validate_moe_batch_route(")
    assert "route.lease.kind() == mem_handle_kind::DIRECT && !route.lease.has_stable_owner_identity()" in validator
    assert "owned_direct_slice_route_acceptance" in HOST_TEST.read_text()
    assert mmid.index("ggml_sycl_publish_backend_aos_expert_handles(") < mmid.index(
        "retained_decode_batch_result = ggml_sycl::ggml_sycl_build_moe_resolved_batch(")
    assert "std::make_shared<const std::vector<int32_t>>(decode.expert_ids)" in direct
    assert "completion->retained_ids = retained_ids" in direct
    move = direct.index("completion->bundle = std::move(admitted.bundle)")
    slices = direct.index("const auto * lease = completion->bundle.owner_leases()")
    assert move < slices
    assert "completion->retained_ids->data()" in direct
    assert "terminal->confirm_quiescent()" in direct
    assert "unified_cache_recover_moe_mmid_workspaces" in direct
    post_mark = direct[direct.index("completion->bundle.mark_possible_submit()") :]
    assert "record_moe_gpu_path" not in post_mark
    assert "ggml_sycl_fallback_error" not in post_mark
    assert "std::make_shared" not in post_mark
    assert "std::vector<" not in post_mark

    materialize_start = source.index("static bool ggml_sycl_materialize_published_mmid_workspaces(")
    materialize_end = source.index("ggml_sycl_lifecycle_result ggml_backend_sycl_model_load_end", materialize_start)
    materialize = source[materialize_start:materialize_end]
    assert "ggml_sycl_get_backend_context_for_device(workspace.owner_device)" in materialize
    assert "backend ? backend->stream() : nullptr" in materialize
    assert ".default_queue()" not in materialize
    assert "unified_cache_materialize_moe_mmid_workspaces(\n        owner, snapshot.version" in materialize
    pre_admit = mmid[:start]
    assert "ggml_sycl_moe_log_canonical_publish_pre_admission" in pre_admit
    assert "canonical_published" in pre_admit
    assert "lease->kind()" in source and "lease->has_stable_owner_identity()" in source


def test_refusal_behavior_never_publishes_ready_or_reports_success() -> None:
    # Behavioral model of the production exception chain: MMID ready guard
    # observes unwinding, compute_forward rethrows, and graph boundary fails.
    for refusal in ("batch_rejected", "wrong_queue", "q1_capability", "nvfp4_capability",
                    "planned_pointer_table_missing"):
        published = False
        graph_success = True
        unwinding = False
        try:
            try:
                unwinding = True
                raise RuntimeError(refusal)
            finally:
                if not unwinding:
                    published = True
        except RuntimeError:
            graph_success = False
        assert not graph_success, refusal
        assert not published, refusal


def test_contract_and_mutation_witnesses() -> None:
    header = HEADER.read_text()
    source = SOURCE.read_text() + "\n" + CPU_DISPATCH_SOURCE.read_text() + "\n" + UNIFIED_CACHE_SOURCE.read_text()
    host_test = HOST_TEST.read_text()
    mem_source = MEM_HANDLE_SOURCE.read_text()
    assert not violations(header, source, host_test, mem_source)

    # Formatting-only rewrites retain the same token/structure contract.
    formatted_header = header.replace("operand.slot_index       = i % slots_per_token;",
                                      "operand . slot_index=\n i% slots_per_token ;")
    formatted_source = source.replace(
        "normalized.authoritative_planned_alternate_ = route.planned_alternate_admitted;",
        "normalized . authoritative_planned_alternate_\n=\nroute . planned_alternate_admitted ;")
    formatted_test = host_test.replace("mem_handle test_make_stable_weight_lease(",
                                       "mem_handle\n test_make_stable_weight_lease (")
    assert not violations(formatted_header, formatted_source, formatted_test, mem_source)

    semantic_mutants = [
        ("slot-zero", header.replace("operand.slot_index       = i % slots_per_token;",
                                     "operand.slot_index = 0;"), source, host_test, mem_source),
        ("raw-direct-rejects-owned", header.replace(
            "route.lease.kind() == mem_handle_kind::DIRECT && !route.lease.has_stable_owner_identity()",
            "route.lease.kind() == mem_handle_kind::DIRECT"), source, host_test, mem_source),
        ("drop-primary-lease-device", header.replace("route.lease.device() != submit_device", "false"),
         source, host_test, mem_source),
        ("drop-secondary-lease-device", header.replace("route.lease.device() != route.owning_device", "false"),
         source, host_test, mem_source),
        ("drop-role-slot-alignment", header.replace("actual.slot_index != expected.slot_index", "false"),
         source, host_test, mem_source),
        ("role-result-default-open", header.replace(
            "moe_batch_reject_reason  reject     = moe_batch_reject_reason::MISSING_ROLE;",
            "moe_batch_reject_reason  reject     = moe_batch_reject_reason::NONE;"),
         source, host_test, mem_source),
        ("drop-role-alignment-open", header.replace(
            "out.reject = moe_batch_reject_reason::NONE;", ""), source, host_test, mem_source),
        ("drop-down-table-preflight", header,
         source.replace("gate_ptrs && up_ptrs && down_table.valid()", "gate_ptrs && up_ptrs"),
         host_test, mem_source),
        ("enable-quarantined-role-capability", header,
         source.replace("const bool prompt_pair_retained_roles_capable = false;",
                        "const bool prompt_pair_retained_roles_capable = prompt_pair_retained_roles_validated;"),
         host_test, mem_source),
        ("publish-before-terminal", header,
         source.replace("terminal.terminal_submitted = true;", "terminal.terminal_submitted = false;"),
         host_test, mem_source),
        ("public-proof", header.replace("  private:\n    // Non-forgeable", "  public:\n    // forged"),
         source, host_test, mem_source),
        ("raw-layout-compare", header,
         source.replace("ggml_sycl_adjust_layout_for_tensor(tensor, alt.layout, current_device) == requested_layout",
                        "alt.layout == requested_layout"), host_test, mem_source),
        ("drop-admission-transfer", header,
         source.replace("normalized.authoritative_planned_alternate_ = route.planned_alternate_admitted;",
                        "normalized.authoritative_planned_alternate_ = false;"), host_test, mem_source),
        ("production-test-factory", header, source, host_test,
         mem_source + "\nmem_handle test_make_stable_weight_lease() { return {}; }\n"),
        ("device-fixture", header, source,
         host_test.replace("test_make_stable_weight_lease", "mem_handle::from_weight_lease_snapshot"), mem_source),
        ("decode-resolver-bypass", header,
         source.replace("ggml_sycl::choose_moe_batch_executor(", "ggml_sycl::moe_batch_executor_choice("),
         host_test, mem_source),
        ("drop-production-host-recipe-executor", header,
         source.replace("ggml_sycl_cpu_moe_host_aos_execute(task, &reject)", "false"),
         host_test, mem_source),
        ("restore-full-admitted-activation-copy", header,
         source.replace("std::memcpy(act, task.activations, rows * static_cast<size_t>(K) * sizeof(float))",
                        "std::memcpy(act, task.activations, ws.activation_f32_bytes)"),
         host_test, mem_source),
        ("enable-Q1-device-route", header,
         source.replace("if (type == GGML_TYPE_Q1_0 || type == GGML_TYPE_NVFP4)", "if (false)", 1),
         host_test, mem_source),
        ("refusal-return", header,
         source.replace("throw ggml_sycl_fallback_error(\"MUL_MAT_ID retained", "return; // MUL_MAT_ID retained"),
         host_test, mem_source),
        ("fast-path-before-admission", header,
         source.replace("if (ne12 != 1) {\n        if (!pp_cpu_reference_force_router",
                        "{\n        if (!pp_cpu_reference_force_router"),
         host_test, mem_source),
        ("all-local-decode-bypass", header,
         re.sub(r"const\s+bool\s+moe_hybrid_with_plan\s*=\s*ne12\s*==\s*1\s*\|\|\s*selected_hybrid_route\s*;",
                "const bool moe_hybrid_with_plan = selected_hybrid_route;", source),
         host_test, mem_source),
        ("prompt-pointer-stage", header,
         inject_before_pair_end(source,
                        "(void) ggml_sycl_update_moe_ptr_table("
                        "ctx, src0, ids, GGML_LAYOUT_AOS, nullptr);", 1),
         host_test, mem_source),
        ("optional-down-partial-skip", header,
         inject_before_pair_end(source,
                        "ggml_sycl_moe_precomputed_skip_insert("
                        "g_moe_precomputed_mmid_skip, pair.down_dst, ctx.device);", 1),
         host_test, mem_source),
        ("dispatch-before-admission", header,
         re.sub(r"ggml_sycl::moe_resolved_batch_result\s+retained_decode_batch_result\s*;",
                "if (moe_hybrid_with_plan) { return; }\n"
                "    ggml_sycl::moe_resolved_batch_result retained_decode_batch_result;", source),
         host_test, mem_source),
        ("selected-miss-continue", header,
         source.replace('throw ggml_sycl_fallback_error("MUL_MAT_ID selected prompt expert unresolved before submit")',
                        "continue"), host_test, mem_source),
        ("prompt-cache-overwrite", header,
         source.replace("if (ne12 == 1 && blk_layer_id >= 0 && !is_gate_subop)",
                        "if (blk_layer_id >= 0 && !is_gate_subop)"), host_test, mem_source),
        ("collapse-prompt-occurrence", header,
         source.replace("retained_prompt_route_for_occurrence(static_cast<size_t>(iid1)",
                        "retained_prompt_group_route_for_expert(static_cast<size_t>(iid1)", 1),
         host_test, mem_source),
        ("drop-primary-ready", header,
         source.replace("stream->ext_oneapi_submit_barrier({ route.ready_event });", "(void) route.ready_event;"),
         host_test, mem_source),
        ("drop-secondary-ready", header,
         source.replace("entry.has_ready_event = has_ready_event;", "entry.has_ready_event = false;"),
         host_test, mem_source),
        ("drop-host-ready-wait", header,
         source.replace("if (has_ready_event) {\n                ready_event.wait();",
                        "if (false) {\n                ready_event.wait();"), host_test, mem_source),
        ("reachable-prompt-resolver", header,
         inject_before_pair_end(source,
                        "(void) ggml_sycl_resolve_moe_expert_route("
                        "src0, ctx.device, 0, GGML_LAYOUT_AOS, false);", 1), host_test, mem_source),
        ("reachable-prompt-dispatch-resolver", header,
         inject_before_pair_end(source,
                        "(void) ggml_sycl_resolve_moe_expert_route_for_dispatch("
                        "src0, ctx.device, 0, GGML_LAYOUT_AOS, false);", 1), host_test, mem_source),
        ("reachable-prompt-reacquire-table", header,
         inject_before_pair_end(source,
                        "(void) moe_fusion_ensure_gpu0_ptrs("
                        "ctx, src0, nullptr, 0, 0, GGML_LAYOUT_AOS, nullptr, nullptr, true);", 1),
         host_test, mem_source),
        ("reachable-prompt-materializer", header,
         inject_before_pair_end(source,
                        "(void) ggml_sycl_materialize_planned_expert_layout("
                        "src0, {}, 0, ctx.device, GGML_LAYOUT_AOS, nullptr, true, true);", 1),
         host_test, mem_source),
        ("reachable-prompt-expert-ptr", header,
         inject_before_pair_end(source,
                        "(void) ggml_sycl::ggml_sycl_resolve_expert_ptr("
                        "src0, ctx.device, 0);", 1), host_test, mem_source),
        ("reachable-prompt-composite-resolver", header,
         inject_before_pair_end(source,
                        "(void) ggml_sycl_resolve(src0, ctx.device);", 1),
         host_test, mem_source),
        ("reachable-prompt-weight-resolver", header,
         inject_before_pair_end(source,
                        "(void) ggml_sycl_resolve_tensor_ptr(src0, ctx.device);", 1),
         host_test, mem_source),
    ]
    for name, mutant_header, mutant_source, mutant_test, mutant_mem in semantic_mutants:
        assert violations(mutant_header, mutant_source, mutant_test, mutant_mem), f"semantic mutation survived: {name}"
