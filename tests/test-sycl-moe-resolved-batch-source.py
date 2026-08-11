#!/usr/bin/env python3
"""Source contract and positive-control mutations for retained MoE batches."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "ggml/src/ggml-sycl/moe-resolved-batch.hpp"
SOURCE = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"
HOST_TEST = ROOT / "tests/test-sycl-moe-resolved-batch.cpp"


def violations(header: str, source: str, host_test: str) -> list[str]:
    failures: list[str] = []
    required_header = {
        "copy IDs once": "out.batch.expert_ids.assign(ids, ids + count);",
        "occurrence preservation": "operand.occurrence       = i;",
        "token indexing": "operand.token_index      = i / slots_per_token;",
        "slot indexing": "operand.slot_index       = i % slots_per_token;",
        "missing lease refusal": "return moe_batch_reject_reason::MISSING_HANDLE;",
        "DIRECT/raw compatibility refusal": "return moe_batch_reject_reason::RAW_COMPAT_HANDLE;",
        "stable owner identity": "route.lease.has_stable_owner_identity()",
        "transient pointer agreement": "resolved.ptr != route.transient_ptr",
        "actual layout agreement": "resolved.layout != route.actual_layout",
        "submit/owner policy": "route.owning_device != submit_device",
        "primary lease device policy": "route.lease.device() != submit_device",
        "secondary lease device policy": "route.lease.device() != route.owning_device",
        "planner/owner policy": "route.planned_device == route.owning_device",
        "opaque alternate proof validation":
            "route.has_authoritative_planned_alternate() && route.planned_on_device && primary",
        "ready event propagation": "operand.ready_event = route.ready_event;",
        "typed rejection": "moe_batch_reject_reason  reject",
        "stable-only sharing": "retained.stable_identity_equal(route.lease)",
    }
    required_source = {
        "canonical dispatch resolver": "ggml_sycl_resolve_moe_expert_route_for_dispatch(",
        "canonical lease move": "normalized.lease         = std::move(route.lease);",
        "source reason propagation": "normalized.source_reason = static_cast<int>(route.reason);",
        "authoritative placement alternate lookup": "lookup_expert_placement(std::string(src0->name), expert_id)",
        "opaque planned alternate authorization": "normalized.authoritative_planned_alternate_ =",
    }
    for name, needle in required_header.items():
        if needle not in header:
            failures.append(name)
    for name, needle in required_source.items():
        if needle not in source:
            failures.append(name)

    route_start = header.index("struct moe_batch_route")
    route_end = header.index("struct moe_resolved_operand", route_start)
    route_definition = header[route_start:route_end]
    if "private:\n" not in route_definition or "bool authoritative_planned_alternate_ = false;" not in route_definition:
        failures.append("alternate proof is private and default-deny")
    if "owner_is_planned_alternate" in header:
        failures.append("public/caller-writable alternate authorization")
    if "from_weight_lease_snapshot" in host_test or "get_unified_cache_for_device" in host_test:
        failures.append("device-dependent host fixture")
    if "test_make_stable_weight_lease" not in host_test:
        failures.append("zero-device stable lease fixture")

    wrapper_start = source.index("moe_resolved_batch_result ggml_sycl_build_moe_resolved_batch")
    wrapper_end = source.index("static bool ggml_sycl_try_pp_local_moe_route", wrapper_start)
    production = header + source[wrapper_start:wrapper_end]
    for forbidden in ("sycl::malloc_device", "sycl::malloc_host", "sycl::malloc_shared",
                      "sycl::free(", "mem_handle::from_chunk_ptr(", "mem_handle::from_direct("):
        if forbidden in production:
            failures.append(f"new ownership/compat bridge: {forbidden}")
    return failures


def test_contract_and_mutation_witnesses() -> None:
    header = HEADER.read_text()
    source = SOURCE.read_text()
    host_test = HOST_TEST.read_text()
    assert not violations(header, source, host_test), violations(header, source, host_test)

    # Every load-bearing source invariant has an explicit RED witness.  Mutants
    # are in-memory only; the repository is never modified by this test.
    mutations = {
        "remove-id-snapshot": ("header", "out.batch.expert_ids.assign(ids, ids + count);"),
        "remove-occurrence": ("header", "operand.occurrence       = i;"),
        "remove-missing-handle": ("header", "return moe_batch_reject_reason::MISSING_HANDLE;"),
        "remove-direct-reject": ("header", "return moe_batch_reject_reason::RAW_COMPAT_HANDLE;"),
        "remove-stable-identity": ("header", "route.lease.has_stable_owner_identity()"),
        "remove-pointer-check": ("header", "resolved.ptr != route.transient_ptr"),
        "remove-layout-check": ("header", "resolved.layout != route.actual_layout"),
        "remove-submit-owner-check": ("header", "route.owning_device != submit_device"),
        "remove-primary-lease-device-check": ("header", "route.lease.device() != submit_device"),
        "remove-secondary-lease-device-check": ("header", "route.lease.device() != route.owning_device"),
        "remove-plan-owner-check": ("header", "route.planned_device == route.owning_device"),
        "remove-explicit-alternate-proof":
            ("header", "route.has_authoritative_planned_alternate() && route.planned_on_device && primary"),
        "remove-ready-event": ("header", "operand.ready_event = route.ready_event;"),
        "remove-stable-sharing": ("header", "retained.stable_identity_equal(route.lease)"),
        "bypass-canonical-resolver": ("source", "ggml_sycl_resolve_moe_expert_route_for_dispatch("),
        "drop-canonical-lease": ("source", "normalized.lease         = std::move(route.lease);"),
        "drop-planned-alternate-proof":
            ("source", "normalized.authoritative_planned_alternate_ ="),
    }
    for name, (which, needle) in mutations.items():
        assert (header if which == "header" else source).count(needle) >= 1, name
        mutant_header = header.replace(needle, "/* MUTATED */") if which == "header" else header
        mutant_source = source.replace(needle, "/* MUTATED */") if which == "source" else source
        assert violations(mutant_header, mutant_source, host_test), f"mutation survived: {name}"

    slot_mutant = header.replace("operand.slot_index       = i % slots_per_token;",
                                 "operand.slot_index       = 0;")
    assert violations(slot_mutant, source, host_test), "mutation survived: force-slot-zero"

    public_proof = header.replace("  private:\n    // Non-forgeable", "  public:\n    // FORGED")
    assert violations(public_proof, source, host_test), "mutation survived: make-proof-public"

    writable_proof = header.replace("bool authoritative_planned_alternate_ = false;",
                                    "bool owner_is_planned_alternate = false;")
    assert violations(writable_proof, source, host_test), "mutation survived: caller-writable-proof"

    device_fixture = host_test.replace("test_make_stable_weight_lease",
                                       "mem_handle::from_weight_lease_snapshot")
    assert violations(header, source, device_fixture), "mutation survived: device-dependent-fixture"
