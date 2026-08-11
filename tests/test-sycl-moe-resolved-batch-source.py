#!/usr/bin/env python3
"""Source contract and positive-control mutations for retained MoE batches."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "ggml/src/ggml-sycl/moe-resolved-batch.hpp"
SOURCE = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"


def violations(header: str, source: str) -> list[str]:
    failures: list[str] = []
    required_header = {
        "copy IDs once": "out.batch.expert_ids.assign(ids, ids + count);",
        "occurrence preservation": "operand.occurrence       = i;",
        "token/slot preservation": "operand.token_index      = i / slots_per_token;",
        "missing lease refusal": "return moe_batch_reject_reason::MISSING_HANDLE;",
        "DIRECT/raw compatibility refusal": "return moe_batch_reject_reason::RAW_COMPAT_HANDLE;",
        "stable owner identity": "route.lease.has_stable_owner_identity()",
        "transient pointer agreement": "resolved.ptr != route.transient_ptr",
        "actual layout agreement": "resolved.layout != route.actual_layout",
        "submit/owner policy": "route.owning_device != submit_device",
        "planner/owner policy": "route.planned_device == route.owning_device",
        "ready event propagation": "operand.ready_event = route.ready_event;",
        "typed rejection": "moe_batch_reject_reason  reject",
        "stable-only sharing": "retained.stable_identity_equal(route.lease)",
    }
    required_source = {
        "canonical dispatch resolver": "ggml_sycl_resolve_moe_expert_route_for_dispatch(",
        "canonical lease move": "normalized.lease         = std::move(route.lease);",
        "source reason propagation": "normalized.source_reason = static_cast<int>(route.reason);",
    }
    for name, needle in required_header.items():
        if needle not in header:
            failures.append(name)
    for name, needle in required_source.items():
        if needle not in source:
            failures.append(name)

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
    assert not violations(header, source), violations(header, source)

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
        "remove-plan-owner-check": ("header", "route.planned_device == route.owning_device"),
        "remove-ready-event": ("header", "operand.ready_event = route.ready_event;"),
        "remove-stable-sharing": ("header", "retained.stable_identity_equal(route.lease)"),
        "bypass-canonical-resolver": ("source", "ggml_sycl_resolve_moe_expert_route_for_dispatch("),
        "drop-canonical-lease": ("source", "normalized.lease         = std::move(route.lease);"),
    }
    for name, (which, needle) in mutations.items():
        assert (header if which == "header" else source).count(needle) >= 1, name
        mutant_header = header.replace(needle, "/* MUTATED */") if which == "header" else header
        mutant_source = source.replace(needle, "/* MUTATED */") if which == "source" else source
        assert violations(mutant_header, mutant_source), f"mutation survived: {name}"
