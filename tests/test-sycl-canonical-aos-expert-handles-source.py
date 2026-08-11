#!/usr/bin/env python3
"""Canonical backend AoS expert capability source contract and mutations."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"


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
    forget = function(source, "static void ggml_sycl_forget_backend_aos_expert_handles")
    setter = function(source, "static void ggml_backend_sycl_buffer_set_tensor")
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
        "mutation withdrawal": "forget_moe_storage_handle_on_device" in forget,
        "complete upload readiness": "is_moe_expert && offset == 0 && size == ggml_nbytes(tensor)" in setter,
        "withdraw before publish": "ggml_sycl_forget_backend_aos_expert_handles" in setter and
                                   setter.find("ggml_sycl_forget_backend_aos_expert_handles") < setter.find("ggml_sycl_publish_backend_aos_expert_handles"),
        "resolver checks registered handles first": route.index("ggml_sycl_try_moe_storage_handle_route") < route.index("cache->resolve_expert"),
        "resolver retains canonical lease": "route.lease" in route and "std::move(handle)" in source,
        "resolver propagates ready event": "route.has_ready_event = stored->has_ready_event" in source,
        "Q1/NVFP4 remains disabled": "device-type-stage2-unimplemented" in capability,
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
        source.replace("ggml_sycl_forget_backend_aos_expert_handles(ctx, tensor);", "", 1),
    ]
    assert all(violations(mutated) for mutated in mutations)
