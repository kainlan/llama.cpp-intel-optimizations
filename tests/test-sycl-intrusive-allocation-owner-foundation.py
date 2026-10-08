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
# Every registry insertion goes through runtime_registry_emplace_locked, which
# keeps the range index in step (llama.cpp-ii25), so the call site pins that
# wrapper rather than the raw unordered_map emplace.
PUBLISH_ANCHOR = "allocation_owner_internal_access::publish(owner_control, rec.handle)"
REGISTRY_ANCHOR = "runtime_registry_emplace_locked(ptr, rec)"
RAW_EMPLACE = "g_runtime_alloc_registry.emplace("


def check_publish_before_registry(source: str) -> str:
    """Return "" when the publish-then-insert order holds, else the failure."""
    for anchor in (PUBLISH_ANCHOR, REGISTRY_ANCHOR):
        if anchor not in source:
            return f"anchor missing (renamed or moved?): {anchor}"
    if source.count(RAW_EMPLACE) != 1:
        return f"expected exactly one {RAW_EMPLACE} (inside runtime_registry_emplace_locked)"
    wrapper = source.index("runtime_registry_emplace_locked(void *")
    if not wrapper < source.index(RAW_EMPLACE) < source.index("\n}\n", wrapper):
        return f"{RAW_EMPLACE} is not inside runtime_registry_emplace_locked"
    publish = source.index(PUBLISH_ANCHOR)
    if REGISTRY_ANCHOR not in source[publish:]:
        return f"{REGISTRY_ANCHOR} does not follow {PUBLISH_ANCHOR}"
    return ""


def self_test(source: str) -> None:
    """Controls: a renamed anchor, a raw emplace beside the wrapper, and a
    publish moved after insertion must each FAIL, so a stale pin is visible."""
    publish = source.index(PUBLISH_ANCHOR)
    insert = source.index(REGISTRY_ANCHOR, publish)
    mutants = {
        "registry-anchor-renamed": source.replace(REGISTRY_ANCHOR, "runtime_registry_insert_locked(ptr, rec)"),
        "publish-anchor-renamed": source.replace(PUBLISH_ANCHOR, "allocation_owner_internal_access::publish_meta(owner_control, rec.handle)"),
        "raw-emplace-bypasses-wrapper": source + "\nvoid f() { g_runtime_alloc_registry.emplace(ptr, rec); }\n",
        "publish-after-insert": (source[:publish] + "/*moved*/" + source[publish + len(PUBLISH_ANCHOR):insert]
                                 + REGISTRY_ANCHOR + "; " + PUBLISH_ANCHOR
                                 + source[insert + len(REGISTRY_ANCHOR):]),
    }
    for name, mutant in mutants.items():
        assert mutant != source, f"self-test mutant {name} did not change the source (anchor stale)"
        assert check_publish_before_registry(mutant), f"self-test mutant {name} was NOT caught"


failure = check_publish_before_registry(SOURCE)
assert not failure, failure
self_test(SOURCE)
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
