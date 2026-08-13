#!/usr/bin/env python3
"""Canonical backend AoS expert capability source contract and mutations."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"
COMMON = ROOT / "ggml/src/ggml-sycl/common.hpp"
UNIFIED_CACHE = ROOT / "ggml/src/ggml-sycl/unified-cache.cpp"
MMVQ = ROOT / "ggml/src/ggml-sycl/mmvq.cpp"
MMVQ_HEADER = ROOT / "ggml/src/ggml-sycl/mmvq.hpp"
PREFETCH_HEADER = ROOT / "ggml/src/ggml-sycl/expert-prefetch.hpp"
PREFETCH_SOURCE = ROOT / "ggml/src/ggml-sycl/expert-prefetch.cpp"
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
    storage_route = function(source, "static bool ggml_sycl_try_moe_storage_handle_route")
    common = COMMON.read_text()
    logical_resolver = function(common, "bool resolve_moe_storage_record")
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
        "exact canonical range": "expert_handle.size() != expert_size" in publish,
        "producer logical range": "record->logical_bytes != expected_bytes" in logical_resolver,
        "padded allocation bounded": "record->logical_bytes > record->handle.size() - record->logical_offset" in logical_resolver,
        "central logical resolver": "resolve_moe_storage_record" in storage_route and "logical.ptr" in storage_route,
        "bounded logical lease": "record->handle.slice(record->logical_offset, record->logical_bytes)" in logical_resolver,
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
        "resolver checks plan before registered handles": route.index("lookup_expert_placement") < route.index("ggml_sycl_try_moe_storage_handle_route"),
        "active plan rejects unnamed tensors": "if (!src0->name || src0->name[0] == '\\0')" in route and "route.plan_missing = true" in route,
        "resolver checks registered handles before requiring key": route.index("ggml_sycl_try_moe_storage_handle_route") < route.index("if (!base_key.valid)"),
        "resolver requires key only for cache fallback": route.index("if (!base_key.valid)") < route.index("cache->resolve_expert"),
        "resolver retains canonical lease": "route.lease" in route and "logical.logical_handle" in storage_route,
        "resolver propagates ready event": "route.has_ready_event = logical.has_ready_event" in storage_route,
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


def test_moe_metadata_registry_is_value_only_and_consumers_retain_sources() -> None:
    source = SOURCE.read_text()
    start = source.index("struct moe_expert_meta {")
    end = source.index("};", start)
    meta = source[start:end]
    assert "ggml_sycl_cache_id canonical_key" in meta
    for borrowed in ("ggml_tensor *", "ggml_tensor_extra_gpu *", "data_ptr"):
        assert borrowed not in meta, borrowed

    producer = function(source, "static void moe_hybrid_init_once")
    assert "meta.canonical_key = ggml_sycl_get_moe_expert_cache_key" in producer
    keys = function(source, "static std::vector<ggml_sycl_cache_id> ggml_sycl_get_canonical_moe_expert_keys")
    assert "const ggml_sycl_cache_id key = m.canonical_key" in keys
    assert "ggml_sycl_get_moe_expert_cache_key(m." not in keys

    resolver = function(source, "static moe_expert_source ggml_sycl_resolve_moe_meta_source")
    for witness in ("cache->resolve_expert(req)", "resolved.lifetime",
                    "has_stable_owner_identity()", "out.handle.resolve"):
        assert witness in resolver, witness
    prestage = function(source, "static void moe_prestage_popular_experts")
    assert "expert_meta = g_moe_expert_meta" in prestage
    assert "source_leases.push_back(std::move(source))" in prestage
    materialize = function(source, "static bool ggml_sycl_materialize_planned_expert_layout")
    assert "meta_value = m" in materialize
    assert "ggml_sycl_resolve_moe_meta_source(*meta, device)" in materialize

    # The legacy name lookup remains fail-closed. Prefetch uses a value-owned
    # descriptor whose source lease and readiness dependency survive lock exit.
    lookup = function(source, "bool ggml_sycl_lookup_moe_expert_source_by_name")
    stage = function(source, "bool moe_acquire_expert_stage_descriptor")
    assert "return false;" in lookup and "g_moe_expert_meta" not in lookup
    for witness in (
        "meta = candidate",
        "ggml_sycl_resolve_moe_meta_source(meta, device_id)",
        "out.source_owner = std::move(source.handle)",
        "out.deps.push_back(source.ready)",
        "out.valid = out.source_owner.valid()",
    ):
        assert witness in stage, witness


def test_expert_prefetch_api_never_returns_an_unleased_pointer() -> None:
    header = PREFETCH_HEADER.read_text()
    impl = PREFETCH_SOURCE.read_text()
    production = SOURCE.read_text()

    assert "owned_result await_owned" in header
    assert "bool await_ready" in header
    assert "bool is_cached" in header
    for legacy in ("void * await(", "void * get_cached_ptr(", "void * demand_load("):
        assert legacy not in header and legacy not in impl, legacy
    assert ".await(" not in production
    assert ".get_cached_ptr(" not in production
    assert "pf.await_ready(layer_id, eid)" in production
    assert "pf.is_cached(pt_lid_skip" in production

    owned = function(impl, "ExpertPrefetcher::owned_result ExpertPrefetcher::await_owned")
    cached = function(impl, "bool ExpertPrefetcher::is_cached")
    assert "result.owner = std::move(it->second.destination_owner)" in owned
    assert "result.owner = std::move(*cached.lifetime)" in owned
    assert "resolved.lifetime->valid()" in cached


def test_expert_prefetch_model_switch_and_speculative_gc_are_identity_safe() -> None:
    impl = PREFETCH_SOURCE.read_text()
    hint = function(impl, "bool ExpertPrefetcher::hint_locked")
    owned = function(impl, "ExpertPrefetcher::owned_result ExpertPrefetcher::await_owned")
    gc = function(impl, "void ExpertPrefetcher::gc_completed")

    assert "expert_prefetch_cache_key key{ info.cache_key, info.layout }" in hint
    assert "inflight_.count(key)" in hint
    assert "ambiguous across model identities: fail closed" in owned
    assert "inflight_.find(key)" in owned
    assert "command_execution_status" in gc
    assert "wait_and_throw()" in gc
    assert "inflight_.erase(it)" in gc


def test_expert_prefetch_cancel_and_shutdown_release_outside_lock() -> None:
    impl = PREFETCH_SOURCE.read_text()
    cancel = function(impl, "void ExpertPrefetcher::cancel_all")
    shutdown = function(impl, "void ExpertPrefetcher::shutdown")
    destructor = function(impl, "ExpertPrefetcher::~ExpertPrefetcher")

    swap = cancel.index("cancelled.swap(inflight_)")
    wait = cancel.index("req.event.wait_and_throw()")
    assert swap < wait
    assert cancel.rfind("}", 0, wait) > swap  # lock scope ended before wait
    assert "queue = dma_queue_" in cancel
    assert "dma_queue_.reset()" in shutdown
    assert shutdown.index("cancel_all()") < shutdown.index("dma_queue_.reset()")
    assert "ggml_sycl_is_shutting_down()" in destructor
    assert "late_state->swap(inflight_)" in destructor


def test_host_reorder_establishes_source_dependencies_before_cpu_read() -> None:
    source = SOURCE.read_text()
    signature = "static sycl::event ggml_sycl_fill_reordered_host"
    reorder = function(source[source.rindex(signature):], signature)
    wait = reorder.index("wait_and_throw()")
    cpu_read = reorder.index("ggml_sycl_reorder_weight_cpu")
    assert wait < cpu_read
    assert "catch (...)" not in reorder[reorder.index("for (const auto & ev : deps)"):wait]
    assert "deps_consumed = !deps.empty()" in reorder


def test_private_production_route_has_graph_churn_regression() -> None:
    test_source = (ROOT / "ggml/src/ggml-sycl/tests/test-q1-nvfp4-admitted-device.cpp").read_text()
    churn = function(test_source, "void graph_churn_regression")
    assert churn.count("successful_reuse_case(life, type, ne11)") == 2
    assert "ggml_free(churn)" in churn
    main = function(test_source, "int main()")
    for type_name in ("GGML_TYPE_Q1_0", "GGML_TYPE_NVFP4"):
        for ne11 in (1, 3):
            assert f"graph_churn_regression(life, {type_name}, {ne11})" in main


def test_real_sycl_owned_slice_lifecycle_is_registered() -> None:
    test = function(LIFECYCLE_TEST.read_text(), "static bool test_canonical_owned_aos_slice_lifecycle")
    for witness in (
        "unified_alloc(req, &allocation)",
        "mem_handle::from_owned_alloc",
        "owner.slice(expert_bytes, expert_bytes)",
        "expert.resolve(1)",
        "unified_lookup(base, &lookup)",
        "replacement_slice.stable_identity_hash() != old_identity",
    ):
        assert witness in test, witness

    # Pin the named lease reset, not one alignment-sensitive spelling of it or
    # an unrelated `handle = {}` elsewhere in the fixture. Reclamation must be
    # checked only after that final retained owner is released.
    retained_reset = re.search(r"\bretained\s*=\s*\{\s*\}\s*;", test)
    assert retained_reset, "retained lease reset"
    final_reclamation = test.index("TEST_ASSERT(!ggml_sycl::unified_lookup(base, &lookup)")
    assert retained_reset.end() < final_reclamation


def test_mandatory_ranges_reach_pointer_table_and_dpas_rollback() -> None:
    source = SOURCE.read_text()
    common = COMMON.read_text()
    registration = function(common, "bool remember_moe_storage_handle")
    signature = registration[:registration.index("{")]
    assert "logical_offset =" not in signature
    assert "logical_bytes =" not in signature
    assert signature.startswith("bool remember_moe_storage_handle")

    pointer_test = function(source, "bool test_moe_ptr_table_does_not_persist_pointer_cache")
    for witness in (
        "constexpr size_t logical_offset = 64",
        "logical_offset, data.size()",
        "expected_ptr = static_cast<uint8_t *>(padded_base.ptr) + logical_offset",
        "uploaded_ptr == expected_ptr",
    ):
        assert witness in pointer_test, witness

    rollback_start = source.index("action=stage-dpas-down")
    rollback_end = source.index("dpas_down_tensors++", rollback_start)
    rollback = source[rollback_start:rollback_end]
    assert "base_layout == GGML_LAYOUT_AOS" in rollback
    assert "static_cast<size_t>(tensor->nb[2])" in rollback
    assert "expert_base_bytes" in rollback
    assert "/*logical_offset=*/0" in rollback


def test_logical_consumers_and_publishers_fail_closed() -> None:
    source = SOURCE.read_text()
    common = COMMON.read_text()

    raw_payload = function(common, "bool build_moe_ptr_payload_from_handles")
    layout_payload = function(common, "bool build_moe_layout_ptr_payload_from_handles")
    assert "require_layout" not in raw_payload
    assert "expected_expert_bytes" not in raw_payload
    assert "size_t                expected_expert_bytes)" in layout_payload
    assert "expected_expert_bytes == 0" in layout_payload
    assert "resolve_moe_storage_record" in layout_payload
    assert "return count == 0 || populated != 0;" in raw_payload
    assert "return count == 0 || populated != 0;" in layout_payload

    wrapper = function(source, "bool ggml_sycl_build_moe_layout_ptr_payload")
    assert "ggml_sycl_moe_expert_layout_bytes(tensor, required_layout, device)" in wrapper
    assert "build_moe_layout_ptr_payload_from_handles" in wrapper

    mmvq = MMVQ.read_text()
    assert mmvq.count("ggml_sycl_build_moe_layout_ptr_payload(") == 5
    assert "build_moe_ptr_payload_from_handles" not in mmvq
    declaration = MMVQ_HEADER.read_text()
    declaration = declaration[declaration.index("bool ggml_sycl_build_moe_layout_ptr_payload"):]
    declaration = declaration[:declaration.index(";")]
    assert "required_layout" in declaration
    assert "=" not in declaration

    for signature in (
        "static void moe_layer_executor_abi_add_full_role_storage",
        "static bool persistent_tg_collect_full_role_descriptor",
        "static const void * const * moe_fusion_ensure_full_local_ptr_table(",
    ):
        consumer = function(source, signature)
        assert "resolve_moe_storage_record" in consumer, signature
        assert "find_moe_storage_handle" not in consumer, signature

    materialize = function(source, "static bool ggml_sycl_materialize_planned_expert_layout")
    assert "resolve_moe_storage_record" in materialize
    assert "logical.ptr" in materialize
    assert "materialize-publish-rejected" in materialize


def test_dpas_mid_loop_failure_is_transactional_failpoint_contract() -> None:
    source = SOURCE.read_text()
    start = source.index("// Transaction rollback state machine:")
    end = source.index("dpas_down_tensors++", start)
    rollback = source[start:end]
    ordered = (
        "forget_moe_storage_handle_on_device",
        "retire_expert_entry_exact",
        "dpas-rollback-drain-target",
        "stage-dpas-rollback",
        "dpas-rollback-drain-base",
        "resolve_moe_storage_record",
        "dpas-base-restore-failed",
        "GGML_ABORT",
    )
    positions = [rollback.index(token) for token in ordered]
    assert positions == sorted(positions)
    assert "local_experts" in rollback
    assert "target_keys" in rollback
    assert "SIZE_MAX / expert_dpas_bytes" in source
    assert "SIZE_MAX - expert_dpas_bytes" in source

    # Source-level failpoints: deleting logical withdrawal or exact cache
    # retirement must be witnessed by this contract.
    mutated = rollback.replace("extra->forget_moe_storage_handle_on_device", "/* failpoint: retained target */", 1)
    assert mutated.count("forget_moe_storage_handle_on_device") == rollback.count(
        "forget_moe_storage_handle_on_device") - 1
    assert "forget_moe_storage_handle_on_device" in rollback
    mutated = rollback.replace("cache->retire_expert_entry_exact", "/* failpoint: discoverable cache */", 1)
    assert mutated != rollback and "cache->retire_expert_entry_exact" in rollback


def test_rejected_publication_cleanup_and_retirement_ordering() -> None:
    source = SOURCE.read_text()
    cache = UNIFIED_CACHE.read_text()

    reject = function(source, "static void ggml_sycl_reject_expert_publication")
    assert "wait_and_throw()" in reject
    assert "catch (...)" in reject
    assert reject.index("catch (...)") < reject.index("retire_expert_entry_exact")
    assert reject.index("retire_expert_entry_exact") < reject.index("rethrow_exception")

    retire = function(cache, "expert_retire_status unified_cache::retire_expert_entry_exact")
    for witness in (
        "std::lock(direct_lock, cache_lock)",
        "pair.second.retired = true",
        "direct_expert_entries_.erase",
        "released_mirror_handles.clear()",
        "finalize_retired_entries_locked",
    ):
        assert witness in retire, witness
    assert "cache_generation_bump" not in retire

    start = source.index("// Transaction rollback state machine:")
    rollback = source[start:source.index("dpas_down_tensors++", start)]
    assert "catch (...)" in rollback
    assert rollback.count("s1_wait_event") >= 2
    assert rollback.index("dpas-base-restore-failed") < rollback.index("GGML_ABORT")


def test_mutations_are_witnessed() -> None:
    source = SOURCE.read_text()
    mutations = [
        source.replace("ctx->managed_handle.slice(slice_offset, expert_size)",
                       "ggml_sycl::mem_handle::from_direct(reinterpret_cast<void *>(tensor_begin + expert * expert_size), GGML_LAYOUT_AOS, true, ctx->device)", 1),
        source.replace("expert_handle.device() != ctx->device ||", "false ||", 1),
        source.replace("expert_handle.size() != expert_size ||", "false ||", 1),
        source.replace("resolve_moe_storage_record(expert_id, layout, owner, expected_size, &logical)",
                       "false", 1),
        source.replace("if (!src0->name || src0->name[0] == '\\0')", "if (false)", 1),
        source.replace("!expert_handle.has_stable_owner_identity()", "false", 1),
        source.replace("is_moe_expert && offset == 0 && size == ggml_nbytes(tensor)",
                       "is_moe_expert && size > 0", 1),
        source.replace("ggml_sycl_invalidate_backend_weight_mutation(ctx, tensor);", ""),
        source.replace("ggml_sycl_invalidate_backend_weight_mutation(dst_ctx, dst);", "", 1),
        source.replace("ggml_sycl_invalidate_backend_buffer_weights(ctx);", "", 1),
        source.replace("replacement.reserve(expert_count);", "", 1),
    ]
    assert all(violations(mutated) for mutated in mutations)
