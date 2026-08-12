#!/usr/bin/env python3
"""Canonical backend AoS expert capability source contract and mutations."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"
LIFECYCLE_TEST = ROOT / "tests/test-sycl-moe-handle-resolution.cpp"


def function(text: str, signature: str) -> str:
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
                state = "line"
                i += 1
            elif ch == "/" and nxt == "*":
                state = "block"
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
        elif state == "line":
            if ch == "\n":
                state = "code"
        elif state == "block":
            if ch == "*" and nxt == "/":
                state = "code"
                i += 1
        else:
            quote = '"' if state == "string" else "'"
            if ch == "\\":
                i += 1
            elif ch == quote:
                state = "code"
        i += 1
    raise AssertionError(f"unclosed function {signature}")


def violations(source: str) -> list[str]:
    failures: list[str] = []
    publish = function(source, "static bool ggml_sycl_publish_backend_aos_expert_handles")
    invalidate = function(source, "static void ggml_sycl_invalidate_backend_weight_mutation")
    invalidate_buffer = function(source, "static void ggml_sycl_invalidate_backend_buffer_weights")
    setter = function(source, "static void ggml_backend_sycl_buffer_set_tensor")
    memsetter = function(source, "static void ggml_backend_sycl_buffer_memset_tensor")
    copier = function(source, "static bool ggml_backend_sycl_buffer_cpy_tensor")
    clearer = function(source, "static void ggml_backend_sycl_buffer_clear")
    route = function(source, "static moe_expert_route ggml_sycl_resolve_moe_expert_route")
    owner = function(source, "void set_managed_owner")
    capability = function(source, "static moe_route_capability ggml_sycl_moe_query_route_capability")

    requirements = {
        "allocation-time owner": "mem_handle::from_owned_alloc(std::move(h), GGML_LAYOUT_AOS)" in owner,
        "device tier only": "managed_meta.tier != ggml_sycl::alloc_tier::DEVICE_VRAM" in publish,
        "ordinary expert eligibility": "tensor_usage::MOE_EXPERT_WEIGHT" in publish,
        "AoS only": "extra->layout.mode != GGML_LAYOUT_AOS" in publish,
        "checked allocation range": "tensor_span > ctx->managed_meta.size - tensor_offset" in publish,
        "per-expert retained slice": "ctx->managed_handle.slice(slice_offset, expert_size)" in publish,
        "exact device": "expert_handle.device() != ctx->device" in publish,
        "exact range": "expert_handle.size() != expert_size" in publish,
        "stable identity": "!expert_handle.has_stable_owner_identity()" in publish,
        "canonical registration": "remember_moe_storage_handle" in publish,
        "transaction prebuild": "replacement.reserve(expert_count)" in publish,
        "transaction rollback": "had_previous" in publish and "catch (...)" in publish,
        "mutation withdrawal": "forget_moe_storage_handle_on_device" in invalidate,
        "mutation generation": "moe_expert_storage_generation" in invalidate,
        "mutation cache invalidation": "ggml_sycl_drop_all_weight_cache_entries" in invalidate,
        "whole buffer inventory": "ctx->tensor_extras" in invalidate_buffer,
        "set invalidates": "ggml_sycl_invalidate_backend_weight_mutation(ctx, tensor)" in setter,
        "memset invalidates": "ggml_sycl_invalidate_backend_weight_mutation(ctx, tensor)" in memsetter,
        "copy invalidates": "ggml_sycl_invalidate_backend_weight_mutation(dst_ctx, dst)" in copier,
        "clear invalidates": "ggml_sycl_invalidate_backend_buffer_weights(ctx)" in clearer,
        "complete upload readiness": "is_moe_expert && offset == 0 && size == ggml_nbytes(tensor)" in setter,
        "withdraw before publish": setter.find("ggml_sycl_invalidate_backend_weight_mutation") < setter.find("ggml_sycl_publish_backend_aos_expert_handles"),
        "resolver checks registered handles first": route.index("ggml_sycl_try_moe_storage_handle_route") < route.index("cache->resolve_expert"),
        "resolver retains canonical lease": "route.lease" in route and "std::move(handle)" in source,
        "resolver propagates ready event": "route.has_ready_event = stored->has_ready_event" in source,
        "Q1/NVFP4 executor gate remains disabled": "q1_nvfp4_direct_b70_validated = false" in source,
        "Q1/NVFP4 recipe is decode-only": "phase == moe_route_phase::DECODE && rows == 1" in capability,
        "Q1/NVFP4 recipe is exact primary": "route_device == submit_device" in capability,
        "Q1/NVFP4 recipe requires stable owner": "direct_recipe_candidate->has_stable_owner_identity()" in capability,
    }
    failures.extend(name for name, ok in requirements.items() if not ok)

    forbidden = ("from_chunk_ptr", "from_direct", "query_registered_location", "unified_lookup", "register_ready")
    for token in forbidden:
        if token in publish:
            failures.append(f"raw fallback in publisher: {token}")
    registration_tail = setter[setter.index("A complete ordinary AoS upload is the registration boundary"):]
    for token in forbidden:
        if token in registration_tail:
            failures.append(f"raw/post-admission authority in registration tail: {token}")
    return failures


def test_canonical_backend_aos_expert_handle_contract() -> None:
    source = SOURCE.read_text()
    assert not violations(source), violations(source)


def test_real_sycl_owned_slice_lifecycle_is_registered() -> None:
    test = function(LIFECYCLE_TEST.read_text(), "static bool test_canonical_owned_aos_slice_lifecycle")
    for witness in (
        "unified_alloc(req, &allocation)",
        "mem_handle::from_owned_alloc",
        "owner.slice(expert_bytes, expert_bytes)",
        "expert.resolve(1)",
        "unified_lookup(base, &lookup)",
        "retained = {}",
        "replacement_slice.stable_identity_hash() != old_identity",
    ):
        assert witness in test, witness


def test_mutations_are_witnessed() -> None:
    source = SOURCE.read_text()
    mutations = [
        source.replace("ctx->managed_handle.slice(slice_offset, expert_size)",
                       "ggml_sycl::mem_handle::from_direct(reinterpret_cast<void *>(tensor_begin + expert * expert_size), GGML_LAYOUT_AOS, true, ctx->device)", 1),
        source.replace("expert_handle.device() != ctx->device ||", "false ||", 1),
        source.replace("expert_handle.size() != expert_size ||", "false ||", 1),
        source.replace("!expert_handle.has_stable_owner_identity()", "false", 1),
        source.replace("is_moe_expert && offset == 0 && size == ggml_nbytes(tensor)",
                       "is_moe_expert && size > 0", 1),
        source.replace("ggml_sycl_invalidate_backend_weight_mutation(ctx, tensor);", ""),
        source.replace("ggml_sycl_invalidate_backend_weight_mutation(dst_ctx, dst);", "", 1),
        source.replace("ggml_sycl_invalidate_backend_buffer_weights(ctx);", "", 1),
        source.replace("replacement.reserve(expert_count);", "", 1),
    ]
    assert all(violations(mutated) for mutated in mutations)
