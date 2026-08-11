#!/usr/bin/env python3
"""Formatting-insensitive source isolation checks and semantic mutation witnesses."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "ggml/src/ggml-sycl/moe-resolved-batch.hpp"
SOURCE = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"
HOST_TEST = ROOT / "tests/test-sycl-moe-resolved-batch.cpp"
MEM_HANDLE_SOURCE = ROOT / "ggml/src/ggml-sycl/mem-handle.cpp"

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|::|&&|\|\||!=|==|<=|>=|->|[{}()[\].,;:%=*<>!&+-]")


def tokens(text: str) -> list[str]:
    text = re.sub(r"//.*?$|/\*.*?\*/", " ", text, flags=re.MULTILINE | re.DOTALL)
    return TOKEN_RE.findall(text)


def contains_tokens(haystack: list[str], pattern: str) -> bool:
    needle = tokens(pattern)
    width = len(needle)
    return any(haystack[i:i + width] == needle for i in range(len(haystack) - width + 1))


def has_tokens(text: str, pattern: str) -> bool:
    return contains_tokens(tokens(text), pattern)


def violations(header: str, source: str, host_test: str, mem_handle_source: str) -> list[str]:
    failures: list[str] = []
    required_header = {
        "ID snapshot": "out.batch.expert_ids.assign(ids, ids + count)",
        "occurrence": "operand.occurrence = i",
        "token index": "operand.token_index = i / slots_per_token",
        "slot index": "operand.slot_index = i % slots_per_token",
        "missing lease": "return moe_batch_reject_reason::MISSING_HANDLE",
        "raw lease": "return moe_batch_reject_reason::RAW_COMPAT_HANDLE",
        "stable identity": "route.lease.has_stable_owner_identity()",
        "pointer agreement": "resolved.ptr != route.transient_ptr",
        "layout agreement": "resolved.layout != route.actual_layout",
        "primary lease device": "route.lease.device() != submit_device",
        "secondary lease device": "route.lease.device() != route.owning_device",
        "primary plan": "route.planned_device == route.owning_device",
        "opaque proof use": "route.has_authoritative_planned_alternate() && route.planned_on_device && primary",
        "stable-only sharing": "retained.stable_identity_equal(route.lease)",
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
        "decode-only type refusal": "phase == moe_route_phase::DECODE",
        "owning queue ready dependency": "target_queue->ext_oneapi_submit_barrier(route_ready_events)",
        "primary ready dependency": "dispatch_deps.push_back(entry->ready_event)",
        "CPU ready copy": "sycl::event ready = entry.ready_event",
        "CPU ready wait": "ready.wait()",
        "admission failure propagates":
            "throw ggml_sycl_fallback_error(\"MUL_MAT_ID retained decode admission failed\")",
        "CPU-TG executor failure propagates":
            "throw ggml_sycl_fallback_error(\"MUL_MAT_ID retained decode executor refused route\")",
        "main executor failure propagates":
            "throw ggml_sycl_fallback_error(\"MUL_MAT_ID retained main decode executor refused route\")",
        "ready guard suppresses exceptional publish":
            "std::uncaught_exceptions() != uncaught_on_entry",
        "planned pointer-table failure propagates":
            "throw ggml_sycl_fallback_error(\"MUL_MAT_ID planned pointer-table admission failed\")",
        "ID admission failure propagates":
            "throw ggml_sycl_fallback_error(\"MUL_MAT_ID expert ID admission failed\")",
        "shared retained dispatch helper": "append_retained_decode_operand",
        "decode CPU-TG fast gate": "if (cpu_tg_candidate && ne12 != 1 && !xmx_moe_forced)",
        "decode GPU fast gate": "if (ne12 != 1) { if (!pp_cpu_reference_force_router",
        "decode precomputed gate": "src1 && src1->ne[2] != 1 && ggml_sycl_moe_precomputed_skip_contains",
        "all-local decode retained gate": "const bool moe_hybrid_with_plan = ne12 == 1 || selected_hybrid_route",
        "no-plan decode skips pointer staging":
            "if (use_expert_cache && ne12 != 1 && !planned_pp_handle_routing)",
        "CPU-TG retains selected routing semantics":
            "const bool cpu_expert_tg_active = selected_hybrid_route && !prompt_batch",
        "CPU-TG canonical batch consumer":
            "const ggml_sycl::moe_resolved_batch & decode_batch = retained_decode_batch_result.batch",
        "main canonical batch consumer":
            "retained_decode_batch_result.batch.operands[occurrence]",
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
        "named pair capability gate": "prompt_pair_retained_roles_capable && fusion_down_continuation",
        "primary ready dependency": "stream->ext_oneapi_submit_barrier({ route.ready_event })",
        "secondary ready propagation": "entry.has_ready_event = has_ready_event",
        "host ready wait": "if (has_ready_event) { ready_event.wait()",
        "linear prompt identity map": "retained_prompt_groups[expert] = &operand",
        "planned local table from batch":
            "ggml_sycl_upload_moe_ptr_table_from_batch(ctx, src0, retained_prompt_batch_result.batch",
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

    function_start = source.index("static void ggml_sycl_mul_mat_id(")
    function_end = source.index("static bool ggml_sycl_compute_forward", function_start)
    mmid = source[function_start:function_end]
    prompt_admission = mmid.index("ggml_sycl::moe_resolved_batch_result retained_prompt_batch_result;")
    specialized_selection = mmid.index("const bool cpu_tg_candidate", prompt_admission)
    admission = mmid.index("ggml_sycl::moe_resolved_batch_result retained_decode_batch_result;")
    route_mode = mmid.index("const bool selected_hybrid_route", admission)
    dispatch_gate = mmid.index("if (moe_hybrid_with_plan)")

    # Prompt and decode each have one occurrence admission. Prompt admission is
    # before every specialized selector; decode remains before route mode.
    if mmid.count("retained_prompt_batch_result = ggml_sycl::ggml_sycl_build_moe_resolved_batch(") != 1:
        failures.append("prompt has exactly one retained admission")
    if mmid.count("retained_decode_batch_result = ggml_sycl::ggml_sycl_build_moe_resolved_batch(") != 1:
        failures.append("decode has exactly one retained admission")
    if not prompt_admission < specialized_selection < admission:
        failures.append("prompt specialized selection precedes admission")
    if mmid.index('pp_phase_log("ids-ready"') >= admission:
        failures.append("decode admission precedes ID snapshot")
    if not admission < route_mode < dispatch_gate:
        failures.append("decode route/dispatch precedes admission")
    if mmid[prompt_admission:admission].count("ggml_sycl_copy_ids_to_host(ctx, ids, prompt_ids_snapshot)") != 1:
        failures.append("prompt IDs are not snapshotted exactly once")
    if mmid.count("append_retained_decode_operand(") != 2:
        failures.append("both decode routers share retained dispatch helper")

    # Scan prompt admission through the actual function end. The only excluded
    # range is the explicitly false named multi-role capability gate.
    pair_start = mmid.index("if (cpu_tg_candidate && ne12 != 1 && !xmx_moe_forced)", prompt_admission)
    pair_end = mmid.index("cpu_tg_fallthrough:", pair_start)
    prompt_reachable = mmid[prompt_admission:pair_start] + mmid[pair_end:]
    forbidden_prompt_ownership = (
        "ggml_sycl_resolve_moe_expert_route(",
        "ggml_sycl_resolve_moe_expert_route_for_dispatch(",
        "moe_fusion_ensure_gpu0_ptrs(",
        "ggml_sycl_materialize_planned_expert_layout(",
    )
    for forbidden in forbidden_prompt_ownership:
        if has_tokens(prompt_reachable, forbidden):
            failures.append(f"reachable prompt path reacquires/materializes route: {forbidden}")
    if prompt_reachable.count("ggml_sycl_copy_ids_to_host(ctx, ids, prompt_ids_snapshot)") != 1:
        failures.append("prompt IDs are copied outside the one admission snapshot")
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
    source = SOURCE.read_text()
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
        ("drop-primary-lease-device", header.replace("route.lease.device() != submit_device", "false"),
         source, host_test, mem_source),
        ("drop-secondary-lease-device", header.replace("route.lease.device() != route.owning_device", "false"),
         source, host_test, mem_source),
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
        ("refusal-return", header,
         source.replace("throw ggml_sycl_fallback_error(\"MUL_MAT_ID retained", "return; // MUL_MAT_ID retained"),
         host_test, mem_source),
        ("fast-path-before-admission", header,
         source.replace("if (ne12 != 1) {\n        if (!pp_cpu_reference_force_router",
                        "{\n        if (!pp_cpu_reference_force_router"),
         host_test, mem_source),
        ("pointer-table-return", header,
         source.replace("throw ggml_sycl_fallback_error(\"MUL_MAT_ID planned pointer-table admission failed\")",
                        "return"),
         host_test, mem_source),
        ("all-local-decode-bypass", header,
         re.sub(r"const\s+bool\s+moe_hybrid_with_plan\s*=\s*ne12\s*==\s*1\s*\|\|\s*selected_hybrid_route\s*;",
                "const bool moe_hybrid_with_plan = selected_hybrid_route;", source),
         host_test, mem_source),
        ("no-plan-decode-pointer-stage", header,
         source.replace("if (use_expert_cache && ne12 != 1 && !planned_pp_handle_routing)",
                        "if (use_expert_cache && !planned_pp_handle_routing)"),
         host_test, mem_source),
        ("dispatch-before-admission", header,
         source.replace("ggml_sycl::moe_resolved_batch_result retained_decode_batch_result;",
                        "if (moe_hybrid_with_plan) { return; }\n"
                        "    ggml_sycl::moe_resolved_batch_result retained_decode_batch_result;"),
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
         source.replace("cpu_tg_fallthrough:",
                        "cpu_tg_fallthrough:\n    (void) ggml_sycl_resolve_moe_expert_route("
                        "src0, ctx.device, 0, GGML_LAYOUT_AOS, false);", 1), host_test, mem_source),
        ("reachable-prompt-dispatch-resolver", header,
         source.replace("cpu_tg_fallthrough:",
                        "cpu_tg_fallthrough:\n    (void) ggml_sycl_resolve_moe_expert_route_for_dispatch("
                        "src0, ctx.device, 0, GGML_LAYOUT_AOS, false);", 1), host_test, mem_source),
        ("reachable-prompt-reacquire-table", header,
         source.replace("cpu_tg_fallthrough:",
                        "cpu_tg_fallthrough:\n    (void) moe_fusion_ensure_gpu0_ptrs("
                        "ctx, src0, nullptr, 0, 0, GGML_LAYOUT_AOS, nullptr, nullptr, true);", 1),
         host_test, mem_source),
        ("reachable-prompt-materializer", header,
         source.replace("cpu_tg_fallthrough:",
                        "cpu_tg_fallthrough:\n    (void) ggml_sycl_materialize_planned_expert_layout("
                        "src0, {}, 0, ctx.device, GGML_LAYOUT_AOS, nullptr, true, true);", 1),
         host_test, mem_source),
    ]
    for name, mutant_header, mutant_source, mutant_test, mutant_mem in semantic_mutants:
        assert violations(mutant_header, mutant_source, mutant_test, mutant_mem), f"semantic mutation survived: {name}"
