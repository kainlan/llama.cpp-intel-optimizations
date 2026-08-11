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


def has_tokens(text: str, pattern: str) -> bool:
    haystack, needle = tokens(text), tokens(pattern)
    width = len(needle)
    return any(haystack[i:i + width] == needle for i in range(len(haystack) - width + 1))


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
        "decode batch seam": "auto decode_batch_result = ggml_sycl::ggml_sycl_build_moe_resolved_batch(",
        "metadata executor": "ggml_sycl::choose_moe_batch_executor(",
        "typed capability refusal": "moe_batch_reject_reason::CAPABILITY_UNSUPPORTED",
        "Q1 decode refusal": "type == GGML_TYPE_Q1_0",
        "NVFP4 decode refusal": "type == GGML_TYPE_NVFP4",
        "decode-only type refusal": "phase == moe_route_phase::DECODE",
        "owning queue ready dependency": "target_queue->ext_oneapi_submit_barrier(route_ready_events)",
        "primary ready dependency": "dispatch_deps.push_back(entry->ready_event)",
        "CPU ready copy": "sycl::event ready = entry.ready_event",
        "CPU ready wait": "ready.wait()",
    }
    for name, pattern in required_header.items():
        if not has_tokens(header, pattern):
            failures.append(name)
    for name, pattern in required_source.items():
        if not has_tokens(source, pattern):
            failures.append(name)

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

    if source.count("ggml_sycl::ggml_sycl_build_moe_resolved_batch(") < 2:  # CPU-TG + main decode
        failures.append("both decode routers use retained batch")

    decode_start = source.index("auto decode_batch_result = ggml_sycl::ggml_sycl_build_moe_resolved_batch(")
    decode_end = source.index("if (have_plan_hybrid) {", decode_start)
    decode_route = source[decode_start:decode_end]
    for forbidden in ("from_chunk_ptr", "from_direct", "is_device_expert_ptr", "src0->data", "route.ptr"):
        if has_tokens(decode_route, forbidden):
            failures.append(f"decode raw routing: {forbidden}")

    production = header + wrapper
    for forbidden in ("sycl::malloc_device", "sycl::malloc_host", "sycl::malloc_shared", "sycl::free(",
                      "mem_handle::from_chunk_ptr(", "mem_handle::from_direct("):
        if has_tokens(production, forbidden):
            failures.append(f"ownership bridge: {forbidden}")
    return failures


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
    ]
    for name, mutant_header, mutant_source, mutant_test, mutant_mem in semantic_mutants:
        assert violations(mutant_header, mutant_source, mutant_test, mutant_mem), f"semantic mutation survived: {name}"
