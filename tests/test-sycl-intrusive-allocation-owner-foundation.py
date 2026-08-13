#!/usr/bin/env python3
"""Host-only source/ABI gate for intrusive allocation-owner orders 1-3."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = (ROOT / "ggml/src/ggml-sycl/unified-cache.hpp").read_text()
SOURCE = (ROOT / "ggml/src/ggml-sycl/unified-cache.cpp").read_text()


def require(text: str, needle: str) -> None:
    assert needle in text, f"missing owner foundation contract: {needle}"


# Ownership shape and typed outcomes.
for token in (
    "class alloc_owner_control final",
    "class alloc_owner",
    "class shared_alloc_owner",
    "enum class allocation_error",
    "struct allocation_result",
    "enum class release_attempt_status",
    "struct release_attempt",
    "static_assert(!std::is_copy_constructible<alloc_owner>::value",
    "static_assert(std::is_copy_constructible<shared_alloc_owner>::value",
):
    require(HEADER, token)

# The unique-to-shared conversion transfers the same intrusive reference.
require(SOURCE, "shared_alloc_owner(std::exchange(control_, nullptr))")
# Control allocation precedes the only physical-allocation adapter call.
create = SOURCE.index("control = allocation_owner_internal_access::create(coordinator)")
physical = SOURCE.index("unified_alloc(req, &legacy)", create)
assert create < physical
# Registry metadata is published by the control before registry insertion.
publish = SOURCE.index("allocation_owner_internal_access::publish(owner_control, rec.handle)")
registry = SOURCE.index("g_runtime_alloc_registry.emplace(ptr, rec)", publish)
assert publish < registry
# Retry queue is embedded/intrusive and backend detach/shutdown are gated.
for token in (
    "alloc_owner_control * retry_next_ = nullptr",
    "control->retry_next_ = retry_head_",
    "bool unified_allocation_release_coordinator_detach",
    "if (coordinator && !coordinator->can_detach())",
    "shutdown refused: device=%d live allocation controls=%zu retries=%zu",
):
    require(HEADER + SOURCE, token)

print("intrusive allocation owner foundation source contracts: PASS")

# Legacy handle is composition-only and metadata cannot mint ownership.
for token in (
    "class alloc_handle final",
    "static_assert(!std::is_base_of<alloc_metadata, alloc_handle>::value",
    "static_assert(!std::is_constructible<alloc_handle, alloc_metadata>::value",
    "static_assert(!std::is_convertible<alloc_handle *, alloc_metadata *>::value",
    "registered_release_status release_registered_allocation",
    "runtime_alloc_state::RELEASING",
):
    require(HEADER + SOURCE, token)

# No lookup-to-owner reconstruction remains in production sources.
for forbidden in (
    "alloc_handle(tracked)",
    "alloc_handle(looked_up)",
    "alloc_handle(metadata)",
    "static_cast<alloc_metadata &>(*out)",
):
    assert forbidden not in HEADER + SOURCE, f"forbidden metadata owner reconstruction: {forbidden}"

print("owner mint and exact registry release source contracts: PASS")
