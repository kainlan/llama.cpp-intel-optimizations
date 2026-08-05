#!/usr/bin/env python3
from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parents[1]
hpp = (root / "ggml/src/ggml-sycl/model-lifecycle.hpp").read_text()
cpp = (root / "ggml/src/ggml-sycl/model-lifecycle.cpp").read_text()
backend = (root / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()
registry_backend = (root / "ggml/src/ggml-backend-reg.cpp").read_text()
backend_base = (root / "ggml/src/ggml-backend.cpp").read_text()
backend_dl = (root / "ggml/src/ggml-backend-dl.cpp").read_text()
public = (root / "ggml/include/ggml-sycl.h").read_text()
llama = (root / "src/llama-model.cpp").read_text()
cache_hpp = (root / "ggml/src/ggml-sycl/unified-cache.hpp").read_text()
cache_cpp = (root / "ggml/src/ggml-sycl/unified-cache.cpp").read_text()
sycl_cmake = (root / "ggml/src/ggml-sycl/CMakeLists.txt").read_text()
sycl_bench_cmake = (root / "tools/sycl-kernel-bench/CMakeLists.txt").read_text()

def function_has_guard(source, name, guard):
    pattern = re.compile(r"(?:^|\n)[^\n;]*\b" + re.escape(name) + r"\s*\([^;]*?\)\s*(?:try\s*)?\{", re.S)
    return any(guard in source[m.end():m.end() + 1500] for m in pattern.finditer(source))

def function_has_module_guard(source, name):
    return function_has_guard(source, name, "sycl_module_mutation_guard")

saveable_mutating_procs = {
    "ggml_backend_split_buffer_type": "ggml_backend_sycl_split_buffer_type",
    "ggml_backend_tp_buffer_type": "ggml_backend_sycl_tp_buffer_type",
    "ggml_backend_sycl_kv_buffer_type_from_dev": "ggml_backend_sycl_kv_buffer_type_from_dev",
    "ggml_backend_sycl_push_kv_layer_mask_from_dev": "ggml_backend_sycl_push_kv_layer_mask_from_dev",
    "ggml_backend_sycl_host_compute_buffer_type": "ggml_backend_sycl_host_compute_buffer_type",
    "ggml_backend_sycl_cpu_offload_compute_buffer_type": "ggml_backend_sycl_cpu_offload_compute_buffer_type",
    "ggml_backend_sycl_host_buffer_type": "ggml_backend_sycl_host_buffer_type",
    "ggml_backend_sycl_host_buffer_type_for_device": "ggml_backend_sycl_host_buffer_type_for_device",
    "ggml_backend_sycl_register_host_weight_tensor": "ggml_backend_sycl_register_host_weight_tensor",
    "ggml_backend_sycl_register_weight_identity": "ggml_backend_sycl_register_weight_identity",
    "ggml_backend_sycl_register_weight_usage": "ggml_backend_sycl_register_weight_usage",
    "ggml_backend_sycl_try_register_weight_usage": "ggml_backend_sycl_try_register_weight_usage",
    "ggml_backend_sycl_stage_inventory_plan": "ggml_backend_sycl_stage_inventory_plan",
    "ggml_backend_sycl_model_load_begin": "ggml_backend_sycl_model_load_begin",
    "ggml_backend_sycl_model_load_enter_nested": "ggml_backend_sycl_model_load_enter_nested",
    "ggml_backend_sycl_model_load_end": "ggml_backend_sycl_model_load_end",
    "ggml_backend_sycl_activate_model_plan": "ggml_backend_sycl_activate_model_plan",
    "ggml_backend_sycl_set_runtime_context_for_model": "ggml_backend_sycl_set_runtime_context_for_model",
    "ggml_backend_sycl_model_unloaded_token": "ggml_backend_sycl_model_unloaded_token",
    "ggml_backend_sycl_model_quarantine_token": "ggml_backend_sycl_model_quarantine_token",
    "ggml_backend_sycl_test_seed_cpu_retained": "ggml_backend_sycl_test_seed_cpu_retained",
    "ggml_backend_sycl_test_seed_moe_module_state": "ggml_backend_sycl_test_seed_moe_module_state",
    "ggml_backend_sycl_test_allocate_predictor_scores": "ggml_backend_sycl_test_allocate_predictor_scores",
    "ggml_backend_sycl_test_hold_live_update": "ggml_backend_sycl_test_hold_live_update",
    "ggml_backend_sycl_test_release_live_update": "ggml_backend_sycl_test_release_live_update",
    "ggml_backend_sycl_test_fail_next_backend_publish": "ggml_backend_sycl_test_fail_next_backend_publish",
    "ggml_backend_sycl_test_fail_next_candidate_binding_allocation": "ggml_backend_sycl_test_fail_next_candidate_binding_allocation",
    "ggml_backend_sycl_test_block_next_candidate_binding_allocation": "ggml_backend_sycl_test_block_next_candidate_binding_allocation",
    "ggml_backend_sycl_test_release_candidate_binding_failure": "ggml_backend_sycl_test_release_candidate_binding_failure",
    "ggml_backend_sycl_test_fail_next_arena_free": "ggml_backend_sycl_test_fail_next_arena_free_guarded",
    "ggml_backend_sycl_test_fail_next_shutdown_clean": "ggml_backend_sycl_test_fail_next_shutdown_clean_guarded",
    "ggml_backend_sycl_test_fail_next_registry_stage": "ggml_backend_sycl_test_fail_next_registry_stage",
    "ggml_backend_sycl_test_block_next_kv_push": "ggml_backend_sycl_test_block_next_kv_push",
}
module_proc_inventory_ok = all(
    ('strcmp(name, "' + proc + '")') in backend and function_has_module_guard(backend, function)
    for proc, function in saveable_mutating_procs.items()
)
buft_callback_inventory = [
    "ggml_backend_buft_name", "ggml_backend_buft_alloc_buffer", "ggml_backend_buft_get_alignment",
    "ggml_backend_buft_get_max_size", "ggml_backend_buft_get_alloc_size", "ggml_backend_buft_is_host",
    "ggml_backend_buft_get_caps",
]
buft_callback_inventory_ok = all(function_has_guard(backend_base, name, "device_call_guard")
                                  for name in buft_callback_inventory)
buffer_callback_inventory = [
    "ggml_backend_buffer_free", "ggml_backend_buffer_get_base", "ggml_backend_buffer_init_tensor",
    "ggml_backend_buffer_clear", "ggml_backend_buffer_reset", "ggml_backend_buffer_get_caps",
    "ggml_backend_buffer_copy_tensor", "ggml_backend_tensor_set", "ggml_backend_tensor_get",
    "ggml_backend_tensor_set_2d", "ggml_backend_tensor_get_2d", "ggml_backend_tensor_memset",
]
buffer_callback_inventory_ok = all(function_has_guard(backend_base, name, "device_call_guard")
                                    for name in buffer_callback_inventory)
placement_paths = sorted(
    [root / "ggml/include/ggml-sycl.h"]
    + list((root / "ggml/src/ggml-sycl").rglob("*.cpp"))
    + list((root / "ggml/src/ggml-sycl").rglob("*.hpp"))
    + list((root / "ggml/src/ggml-sycl").rglob("*.h"))
    + list((root / "src").glob("*.cpp"))
    + list((root / "src").glob("*.h"))
)
placement_sources = {path.relative_to(root).as_posix(): path.read_text(errors="replace") for path in placement_paths}
placement_production = "\n".join(placement_sources.values())
legacy_global_re = re.compile(r"\bg_(?:has_)?placement_plan\b")
legacy_cache_re = re.compile(
    r"(?:->|\.)\s*(?:has_placement_plan|get_placement_plan|get_placement_plan_owner|"
    r"plan_on_device|plan_on_this_device|moe_tensor_has_host_experts)\s*\("
)
raw_snapshot_reader_re = re.compile(r"(?:->|\.)\s*get_placement_plan_snapshot\s*\(")
raw_snapshot_writer_re = re.compile(r"(?:->|\.)\s*set_placement_plan_snapshot\s*\(")
# Positive controls prove the census expressions still detect both prohibited
# reader families rather than passing because of an accidentally-empty regex.
census_fixture = (
    "if (g_has_placement_plan) use(g_placement_plan); cache->get_placement_plan();"
)
resolve_expert_body = re.search(
    r"expert_resolve_result unified_cache::resolve_expert\(.*?^}\n", cache_cpp, re.S | re.M
).group(0)
register_usage_body = re.search(
    r"bool ggml_backend_sycl_try_register_weight_usage\(.*?^}\n", backend, re.S | re.M
).group(0)
exact_runtime_body = re.search(
    r"ggml_sycl_lifecycle_result ggml_backend_sycl_set_runtime_context_for_model\(.*?^}\n",
    backend, re.S | re.M
).group(0)
checks = {
    "full slot token": re.search(r"struct SlotToken\s*\{\s*uint32_t\s+slot", hpp) is not None
    and "uint64_t generation" in hpp,
    "immutable publication": "std::shared_ptr<const ModelState>" in hpp,
    "two phase finish": all(
        x in hpp
        for x in (
            "COMMITTING",
            "ROLLING_BACK",
            "TEARING_DOWN",
            "prepare_end",
            "finalize_end",
            "prepare_teardown",
        )
    ),
    "checked IDs": "class CheckedCounter" in hpp and "ID_EXHAUSTED" in cpp,
    "explicit nested transaction": "enter_nested(LoadTxnId" in hpp,
    "abort default guard": "hooks.end(txn, false" in llama,
    "noncopyable guard": re.search(
        r"llama_model_sycl_loading_guard\(const llama_model_sycl_loading_guard\s*&\)\s*=\s*delete", llama
    ) is not None,
    "explicit outer success": "sycl_model_loading_guard.finish(true)" in llama,
    "no-alloc success": "if (ml.no_alloc)" in llama
    and "sycl_model_loading_guard.finish(true)" in llama,
    "generation-safe teardown": "hooks.unload(token)" in llama,
    "no legacy llama guard": "ggml_backend_sycl_set_model_loading(" not in llama,
    "public null-output result": "GGML_SYCL_LIFECYCLE_NULL_OUTPUT" in public
    and "model != nullptr" in backend,
    "C boundary catches": backend.count("catch (...)") >= 3,
    "full owner host rows": "ggml_sycl_release_host_weight_extras_for_owner" in backend,
    "mutation hooks": all(
        x in hpp for x in ("M1_SKIP_GENERATION", "M2_NESTED_COMMIT", "M3_CLEAR_POISON")
    ),
    "durable terminal identities": "tombstone_limit_" not in hpp
    and "txns_ is the durable terminal identity table" in cpp,
    "waiters refind by id": "current = txns_.find(id.value)" in cpp,
    "quarantine never live": cpp.count("model_phase::QUARANTINED") >= 3,
    "committing teardown busy": "BUSY, token" in cpp,
    "immutable full plan candidate": all(
        x in backend
        for x in (
            "lifecycle_stage_placement_plan",
            "lifecycle_publish_placement_plan",
            "lifecycle_abort_placement_plan",
        )
    ),
    "keyed transaction-local load candidate": "thread_local std::vector<candidate_binding> candidate_bindings" in cpp
    and "global_registry().bound_candidate()" in backend
    and "registry->bind_candidate(result.txn)" in backend
    and "registry_.unbind_candidate(txn_)" in backend
    and "g_load_candidate_txn_id" not in backend
    and "ggml_sycl_publish_plan_locked(have_plan ? plan_snapshot : nullptr)" not in backend
    and "ggml_sycl_reset_model_load_scratch_state(true)" in backend
    and "lifecycle_stage_no_placement_plan(g_sycl_plan_load_effect_txn" in backend
    and "if (!ggml_sycl::vram_arena_enabled())" in backend,
    "explicit exact activation": "ggml_backend_sycl_activate_model_plan" in public
    and "lifecycle_select_placement_plan" in backend
    and "ggml_backend_sycl_set_runtime_context_for_model" in public,
    "serialized coherent publication": all(
        x in backend
        for x in (
            "g_tensor_inventory_mutex",
            "ggml_sycl_publish_plan_locked",
            "set_placement_plan_snapshot(publication.participates[i] ? snapshot : nullptr)",
            "atomic_store_explicit(&g_placement_publication, snapshot",
        )
    ),
    "globally checked publication IDs": "lifecycle_next_plan_publication_id" in cache_hpp
    and "next == UINT64_MAX" in (root / "ggml/src/ggml-sycl/unified-cache.cpp").read_text()
    and "g_provisional_plan_version" not in backend,
    "single-load cross-TU coherent owners": all(
        name in cache_hpp
        for name in (
            "global_placement_plan_owner",
            "coherent_placement_plan_owner",
            "coherent_cache_placement_plan_owner",
        )
    )
    and "for (int attempt" not in backend
    and "global_second" not in backend,
    "no unsafe global placement readers": not legacy_global_re.search(
        placement_production
    ),
    "no unsafe cache placement readers": not legacy_cache_re.search(
        placement_production
    ),
    "global census positive control": len(legacy_global_re.findall(census_fixture))
    == 2,
    "cache census positive control": len(legacy_cache_re.findall(census_fixture)) == 1,
    "exact owning reader call census": {
        path: sum(
            text.count(name + "(")
            for name in (
                "global_placement_plan_owner",
                "coherent_placement_plan_owner",
                "coherent_cache_placement_plan_owner",
                "cache_placement_coherence",
            )
        )
        for path, text in placement_sources.items()
        if any(
            name + "(" in text
            for name in (
                "global_placement_plan_owner",
                "coherent_placement_plan_owner",
                "coherent_cache_placement_plan_owner",
                "cache_placement_coherence",
            )
        )
    }
    == {
        "ggml/src/ggml-sycl/common.hpp": 3,
        "ggml/src/ggml-sycl/expert-prefetch.cpp": 2,
        "ggml/src/ggml-sycl/ggml-sycl.cpp": 8,
        "ggml/src/ggml-sycl/mmvq.cpp": 1,
        "ggml/src/ggml-sycl/unified-cache.cpp": 14,
        "ggml/src/ggml-sycl/unified-cache.hpp": 4,
    },
    "exact internal wrapper census": {
        name: backend.count(name + "(")
        for name in (
            "ggml_sycl_cache_plan_owner",
            "ggml_sycl_global_plan_owner",
            "ggml_sycl_global_plan_snapshot",
            "ggml_sycl_has_global_plan",
        )
    }
    == {
        "ggml_sycl_cache_plan_owner": 127,
        "ggml_sycl_global_plan_owner": 16,
        "ggml_sycl_global_plan_snapshot": 8,
        "ggml_sycl_has_global_plan": 26,
    },
    "cache snapshot pointer identity validation": "lifecycle_plan_snapshot_matches(authority, cached)"
    in backend
    and "authority.get() == cache.get()" in cache_hpp
    and "authority->version == cache->version" not in cache_hpp,
    "raw snapshot reader exact allowlist": {
        path: len(raw_snapshot_reader_re.findall(text))
        for path, text in placement_sources.items()
        if raw_snapshot_reader_re.search(text)
    }
    == {"ggml/src/ggml-sycl/ggml-sycl.cpp": 1},
    "raw snapshot writer whitelist": len(raw_snapshot_writer_re.findall(placement_production)) == 1,
    "same snapshot cache publication": backend.count("set_placement_plan_snapshot") == 1,
    "no cache plan reference accessor": "get_placement_plan(" not in cache_hpp
    and "get_placement_plan_owner(" not in cache_hpp,
    "transactional model runtime update": "g_runtime_external_lease" in backend
    and "g_runtime_update_succeeded" in backend
    and "if (!backend || !backend->context || n_ctx == 0)" in backend,
    "runtime ownership CAS": "lifecycle_replace_placement_plan(current, immutable)" in backend
    and "auto next_kv_info = current->kv_info" in backend
    and "next->kv_info       = next_kv_info" in backend
    and "registry.acquire_live_update(current_token)" in backend
    and "live_update_guard::~live_update_guard()" in cpp,
    "provisional exhaustion is typed": "ggml_sycl_placement_publication_exhausted" in backend
    and "GGML_ABORT(\"[SYCL-PLAN] placement publication ID exhausted\")" not in backend,
    "global owner has no cache dependency": "return ggml_sycl::global_placement_plan_owner();" in backend,
    "hot global owner one atomic load": re.search(
        r"global_placement_plan_owner\(\) noexcept \{(?P<body>.*?)\n\}", backend, re.S
    ).group("body").count("atomic_load_explicit")
    == 1,
    "abort reset preserves authority": "ggml_sycl_reset_model_load_scratch_state(true)" in backend
    and "if (!preserve_placement_authority)" in backend,
    "production publisher primitive": "publish_cache_first_global_last(" in backend
    and "publish_cache_first_global_last(" in cache_hpp
    and "publish_cache_first_global_last(" in (root / "tests/test-sycl-tensor-placement.cpp").read_text(),
    "global cache aliases aggregated": "publication.caches" in backend
    and re.search(r"publication\.participates\[i\]\s*=\s*publication\.participates\[i\]\s*\|\|", backend),
    "KV allocation retains one lifecycle owner": (
        lambda body: body.count("ggml_sycl_global_plan_snapshot()") == 1
        and "g_placement_kv_info" not in body
        and "g_model_n_layer" not in body
        and "ggml_sycl_cache_plan_owner" not in body
    )(
        re.search(
            r"tiered_kv_buft_alloc_buffer\(.*?\n\}", backend, re.S
        ).group(0)
    ),
    "cache decisions retain one coherence read": all(
        (root / "ggml/src/ggml-sycl/unified-cache.cpp").read_text().count(token) >= count
        for token, count in (
            ("planned_materialization_allowed(\"direct_stage", 3),
            ("__func__, &placement", 3),
            ("const placement_cache_read * retained_read", 2),
        )
    ),
    "snapshot owns KV geometry": "placement_kv_info                     kv_info" in cache_hpp
    and "uint32_t                              model_n_layer" in cache_hpp
    and "auto next_kv_info = current->kv_info" in backend,
    "publish replacement before reclaim": re.search(
        r"ggml_sycl_teardown_owner_effects.*?ggml_sycl_publish_plan_locked\(restoration.snapshot\).*?"
        r"ggml_sycl_release_model_slot_resources\(owner\)", backend, re.S
    )
    is not None,
    "explicit cache mismatch state": all(
        name in cache_hpp for name in ("MATCH", "GENUINE_NO_PLAN", "TRANSIENT_MISMATCH")
    )
    and (root / "ggml/src/ggml-sycl/unified-cache.cpp").read_text().count(
        "placement_cache_coherence::TRANSIENT_MISMATCH"
    )
    >= 7,
    "atomic dying cache bit": "unified_cache_set_live_model_mask(live_after)"
    not in backend,
    "candidate-only publication accounting": "publication_from_plan" in backend
    and ("cache->" + "get_placement_plan().weight_host_bytes") not in backend,
    "latest live restoration": "global_registry().latest_live()" in backend
    and "explicit_no_plan"
    in (root / "ggml/src/ggml-sycl/unified-cache.hpp").read_text(),
    "plan deletion on teardown": "lifecycle_erase_placement_plan" in backend,
    "dead metadata preallocated": "dead_.emplace" in cpp
    and cpp.index("dead_.emplace")
    < cpp.index("model->second.phase = model_phase::TEARING_DOWN"),
    "rollback effect replay": "error::EFFECT_FAILED" in cpp
    and "poisoned_after_prepare" in cpp,
    "finalize poison authority": "poisoned_after_prepare" in cpp
    and "validate_end" in hpp,
    "serialized concurrent teardown": "item.second.phase == model_phase::TEARING_DOWN"
    in cpp,
    "retained cache coherence decisions": cache_cpp.count("cache_placement_coherence(this)") == 9
    and cache_cpp.count("planned_materialization_allowed(\"direct_stage_") == 3
    and cache_cpp.count("__func__, &placement") == 3
    and "unpin(key, layout, &placement)" in cache_cpp,
    "runtime update lease blocks teardown": "prepare_live_update" in hpp
    and "live_update_serials" in hpp
    and "current->second.live_update_count == 0" in cpp
    and "model_phase::DRAINING_UPDATES" in cpp,
    "update tickets exact one-shot": "next_live_update_serial == 0" in cpp
    and "entry.live_update_serials[index] != ticket.serial" in cpp
    and "--entry.live_update_count" in cpp,
    "fallible pre-finalize restoration": backend.index("teardown_owner_effects(owner)")
    < backend.index("finalize_teardown(ticket, true)"),
    "busy quarantine queues before defer": backend.index("if (!ggml_sycl_quarantine_enqueue(token))")
    < backend.index("registry.defer_quarantine(owner)"),
    "three phase commit publication before LIVE": backend.index("ggml_sycl_publish_prepared_plan_locked(prepared_publication)")
    < backend.index("registry->finalize_end(ticket, true, publication"),
    "DL early and late inventory planning parity": "ggml_backend_sycl_stage_inventory_plan" in public
    and "hooks.stage_inventory(&inventory, &envelope, early)" in llama
    and "defined(GGML_BACKEND_DL)" in llama
    and "llama_model_sycl_compute_early_plan(ml, hparams, __func__)" in llama
    and "ggml_backend_reg_get_proc_address(reg, \"ggml_backend_sycl_stage_inventory_plan\")" in llama
    and "strcmp(name, \"ggml_backend_sycl_stage_inventory_plan\")" in backend
    and "llama_model_sycl_set_late_inventory(ml, hparams, __func__)" in llama
    and "sycl_model_loading_guard.txn.id != 0 && has_sycl_weight_buft" in llama
    and "if (sycl_model_backend && ml.use_mmap)" in llama
    and "disabling mmap for SYCL weight layout upload" in llama
    and (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text().count(
        "ggml_backend_sycl_stage_inventory_plan)(&inventory, &envelope"
    ) >= 2,
    "DL allocation and metadata parity": all(
        f'LOAD_SYCL_PROC({field}, "{name}")' in (root / "src/llama-model-loader.cpp").read_text()
        for field, name in (
            ("weights_evictable", "ggml_backend_sycl_weights_evictable"),
            ("host_buffer_type_for_device", "ggml_backend_sycl_host_buffer_type_for_device"),
            ("register_host_weight", "ggml_backend_sycl_register_host_weight_tensor"),
            ("register_identity", "ggml_backend_sycl_register_weight_identity"),
            ("try_register_usage", "ggml_backend_sycl_try_register_weight_usage"),
        )
    )
    and "defined(GGML_BACKEND_DL)" in (root / "src/llama-model-loader.cpp").read_text()
    and "sycl_hooks.host_buffer_type_for_device(dev)" in (root / "src/llama-model-loader.cpp").read_text()
    and "sycl_hooks.register_host_weight" in (root / "src/llama-model-loader.cpp").read_text()
    and "allow_host_buft_with_mmap = true" in (root / "src/llama-model-loader.cpp").read_text()
    and "final_sycl_hooks.reg && final_sycl_hooks.weights_evictable" in
        (root / "src/llama-model-loader.cpp").read_text()
    and "sycl_hooks.register_identity" in (root / "src/llama-model-loader.cpp").read_text()
    and "sycl_hooks.try_register_usage" in (root / "src/llama-model-loader.cpp").read_text()
    and "ggml_backend_sycl_register_weight_usage" not in
        (root / "src/llama-model-loader.cpp").read_text()
    and "LOAD_SYCL(ggml_backend_sycl_register_weight_usage)" in
        (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text()
    and "LOAD_SYCL(ggml_backend_sycl_try_register_weight_usage)" in
        (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text()
    and all(f'strcmp(name, "{name}")' in backend for name in (
        "ggml_backend_sycl_weights_evictable",
        "ggml_backend_sycl_host_buffer_type",
        "ggml_backend_sycl_host_buffer_type_for_device",
        "ggml_backend_sycl_register_host_weight_tensor",
        "ggml_backend_sycl_register_weight_identity",
        "ggml_backend_sycl_register_weight_usage",
        "ggml_backend_sycl_try_register_weight_usage",
    )),
    "CPU speculative lifecycle cancellation": "sycl_model_loading_guard.cancel()" in llama
    and "if (has_sycl_weight_buffer)" in llama
    and "if (sycl_model_backend)" in llama
    and "rc == GGML_SYCL_LIFECYCLE_EFFECT_FAILED && rollback_token.model_id != 0" in llama
    and "for (int i = 0; i < 40; ++i)" in (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text()
    and "authority_before_cpu_cancels" in (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text(),
    "load effects drain before end": "struct load_effect_lease" in hpp
    and "test_fail_next_load_effect_allocation" in hpp
    and "load-effect-allocation-failure" in (root / "tests/test-sycl-lifecycle-load-txn.cpp").read_text()
    and "acquire_load_effect" in cpp
    and "load_effect_serials.empty()" in cpp
    and "load-effect-drain" in (root / "tests/test-sycl-lifecycle-load-txn.cpp").read_text()
    and "load-effect-abort-drain" in (root / "tests/test-sycl-lifecycle-load-txn.cpp").read_text()
    and "cache-effect-abort-drain" in (root / "tests/test-sycl-lifecycle-load-txn.cpp").read_text()
    and "planning-effect-abort-drain" in (root / "tests/test-sycl-lifecycle-load-txn.cpp").read_text()
    and "identity-effect-abort-drain" in (root / "tests/test-sycl-lifecycle-load-txn.cpp").read_text(),
    "planning and identity hold exact effects": backend.count("registry.acquire_load_effect(registry.bound_candidate())") >= 3
    and "const auto token = effect.owner" in backend
    and "plan_effect_scope" in backend
    and "g_sycl_plan_load_effect_txn" in backend,
    "parser-supported event catch boundary":
        "static void ggml_backend_sycl_event_record(ggml_backend_t backend, ggml_backend_event_t event) {" in backend
    and "static void ggml_backend_sycl_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {" in backend
    and "event_record(ggml_backend_t backend, ggml_backend_event_t event) try" not in backend,
    "host registration uses one leased owner": "load_effect_lease effect" in backend
    and "const auto owner = effect.owner" in backend
    and backend.count("sycl_host_weight_extra_entry new_entry{ tensor, extra, owner }") == 1,
    "exact model-scoped weight identity": "ggml_sycl_owner_name_key" in backend
    and "g_sycl_weight_identities_by_name.find(ggml_sycl_owner_name_key" in backend
    and "ggml_sycl_erase_weight_identities_for_owner" in backend
    and "register_identity(tensor" in (root / "src/llama-model-loader.cpp").read_text()
    and (root / "src/llama-model-loader.cpp").read_text().index("register_identity(tensor") <
        (root / "src/llama-model-loader.cpp").read_text().index("register_host_weight(selected_final_dev"),
    "CPU classified before lifecycle begin": "resolved_sycl_weight_owner" in llama
    and "this->n_gpu_layers() > 0" in llama
    and "params.tensor_buft_overrides" in llama
    and "sycl_tensor_override" in llama
    and "llama_model_sycl_loading_guard sycl_model_loading_guard(resolved_sycl_weight_owner" in llama
    and "if (sycl_model_loading_guard.active)" in llama,
    "final override device metadata": "selected_final_buft" in (root / "src/llama-model-loader.cpp").read_text()
    and "selected_final_dev" in (root / "src/llama-model-loader.cpp").read_text()
    and "llama_model_loader_sycl_hooks_for(selected_final_dev)" in
        (root / "src/llama-model-loader.cpp").read_text(),
    "host rows filtered before dereference": "ggml_sycl_host_row_authorized" in backend
    and "candidate->load_txn_id" in backend
    and "candidate.value" not in backend
    and backend.count("if (!ggml_sycl_host_row_authorized(entry.second.owner))") >= 2
    and "if (!ggml_sycl_host_row_authorized(it->second.owner))" in backend,
    "finisher effects authorize synchronous preload": "struct finisher_effect_scope" in hpp
    and "acquire_finisher_effect(ticket)" in backend
    and "bound_finisher_effect()" in cache_cpp
    and "finisher_effect_serials.empty()" in cpp
    and "finisher-effect-drain" in (root / "tests/test-sycl-lifecycle-load-txn.cpp").read_text(),
    "host cache registration uses exact guard":
        cache_cpp.count("bool unified_cache::register_host_") >= 2
    and cache_cpp.count("stamp_pending_owner(old->second, load_effect_guard)") >= 2
    and cache_cpp.count("stamp_pending_owner(cache_entry, load_effect_guard)") >= 2,
    "preload retains candidate identity": "const auto   preload_plan_snapshot = ggml_sycl_identity_plan_snapshot()" in backend
    and "ggml_sycl_scoped_alloc_rollback" in backend
    and "ggml_sycl_global_plan_snapshot()" not in re.search(
        r"static void ggml_sycl_preload_model_weights\(\).*?\n\}", backend, re.S).group(0)
    and "SYCL host expert registration failed during preload" in backend
    and "SYCL host weight registration failed during preload" in backend
    and "dense_alloc_guard" in backend
    and "host_alloc_guard" in backend,
    "cache reuse stamps exact bound transaction": "stage_expert_group" in cache_cpp
    and cache_cpp.count("load_effect_guard = acquire_bound_load_effect()") >= 12
    and "stamp_pending_owner(entry_it->second, load_effect_guard)" in cache_cpp
    and "stamp_pending_owner(it->second, load_effect_guard)" in cache_cpp
    and "load_effect_state::FAILED" in cache_cpp,
    "nested loading requires exact outer ownership":
        "llama_model_sycl_loading_guard sycl_loading_guard(sycl_model_backend, sycl_model_loading_guard.txn)" in llama
    and "const bool sycl_model_backend = sycl_model_loading_guard.txn.id != 0 && has_sycl_weight_buft" in llama
    and "if (sycl_model_backend)" in llama
    and "if (ml.no_alloc)" in llama
    and "defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)" in llama,
    "exact-device host allocation": "ggml_backend_sycl_host_buffer_type_for_device" in public
    and "req.device = exact_device" in backend
    and "get_current_device_id()" not in re.search(
        r"ggml_backend_sycl_host_buffer_type_alloc_buffer\(.*?\n\}", backend, re.S
    ).group(0),
    "DL KV and compute parity": all(f'strcmp(name, "{name}")' in backend for name in (
        "ggml_backend_sycl_kv_buffer_type_from_dev",
        "ggml_backend_sycl_push_kv_layer_mask_from_dev",
        "ggml_backend_sycl_host_compute_buffer_type",
        "ggml_backend_sycl_cpu_offload_compute_buffer_type",
        "ggml_backend_sycl_cpu_offload_available",
    ))
    and "llama_kv_cache_sycl_hooks_for" in (root / "src/llama-kv-cache.cpp").read_text()
    and "llama_recurrent_sycl_kv_buft" in (root / "src/llama-memory-recurrent.cpp").read_text()
    and "llama_context_sycl_compute_procs" in (root / "src/llama-context.cpp").read_text()
    and "hooks.device_index" in (root / "src/llama-context.cpp").read_text()
    and "hooks.has_active_plan" in (root / "src/llama-context.cpp").read_text(),
    "zero owner skips SYCL context activation": "if (owner.model_id == 0 || owner.load_txn_id == 0)" in
    (root / "src/llama-context.cpp").read_text(),
    "no global host registry clear on load": "Host-weight rows are exact-owner state" in backend
    and "ggml_sycl_release_host_weight_extras(ggml_sycl_host_weight_release_mode::release_registry_refs);" not in
        re.search(r"ggml_sycl_model_loading_effects\(.*?\n\}", backend, re.S).group(0),
    "bound transaction uses exact effect admission": "const auto txn      = registry.bound_candidate()" in cache_cpp
    and "registry.acquire_load_effect(txn)" in cache_cpp
    and "lifecycle_find_candidate_placement_plan(effect.owner.load.value)" in cache_cpp,
    "direct cache tests link lifecycle authority": sycl_cmake.count("model-lifecycle.cpp") >= 6,
    "DL excludes only incomplete direct-source cache fixtures": all(
        re.search(r"if \(NOT GGML_BACKEND_DL\)\s+add_executable\(" + re.escape(target), sycl_cmake)
        for target in (
            "test-unified-cache-fast-path",
            "test-mem-handle-eviction",
            "test-sycl-runtime-alloc",
        )
    )
    and all(name in sycl_cmake for name in (
        "test-sycl-lifecycle-load-txn",
        "test-sycl-lifecycle-runtime-host",
        "test-sycl-lifecycle-runtime-wrapper",
        "test-sycl-module-dlopen",
        "test-sycl-lifecycle-public-api",
    ))
    and re.search(r"if \(NOT GGML_BACKEND_DL\)\s+set\(TARGET sycl-kernel-bench\)", sycl_bench_cmake)
    and "sycl-mxfp4-source-line-probe" in sycl_bench_cmake,
    "DL unload drains module state before dlclose": "ggml_backend_shutdown" in
        (root / "ggml/src/ggml-backend-reg.cpp").read_text()
    and "ggml_backend_sycl_shutdown" in backend
    and "g_watchdog_thread.detach()" not in (root / "ggml/src/ggml-sycl/common.cpp").read_text()
    and "unlocked tombstone load/enumeration overlap" in
        (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text()
    and "ggml_backend_unload_checked(reg)" in
        (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text(),
    "DL SYCL module has private CPU traits and no CPU linkage":
        re.search(r"if \(NOT GGML_BACKEND_DL\).*?target_link_libraries\(ggml-sycl PRIVATE ggml-cpu\).*?endif",
                  (root / "ggml/src/ggml-sycl/CMakeLists.txt").read_text(), re.S)
    and 'if (backend STREQUAL "ggml-cpu")' not in (root / "ggml/src/CMakeLists.txt").read_text()
    and "add_library(${backend} MODULE ${ARGN})" in (root / "ggml/src/CMakeLists.txt").read_text()
    and "ggml_sycl_get_type_traits_cpu" in
        (root / "ggml/src/ggml-sycl/cpu-traits-support.cpp").read_text()
    and "RTLD_NOW | RTLD_LOCAL" in (root / "tests/test-sycl-module-dlopen.cpp").read_text(),
    "host registration insertion faults roll back": "fail_next_host_registration_insert_.exchange(false)" in cache_cpp
    and "test_host_registration_initial_insert_failures" in
        (root / "tests/test-sycl-reset-model-weight-lease-preserve.cpp").read_text(),
    "all direct mirrors carry transaction ownership": len(re.findall(
        r"pending_load_txn_id\s*=\s*load_effect_guard\.load_txn_id\(\)", cache_cpp)) >= 4,
    "expert resolution consumes lease snapshots": "from_weight_lease_locked" not in resolve_expert_body
    and resolve_expert_body.count("from_weight_lease_snapshot") == 2
    and "test_expert_abort_after_resolve_snapshot" in
        (root / "tests/test-sycl-reset-model-weight-lease-preserve.cpp").read_text(),
    "non-authoritative teardown preserves exact publication":
    "const bool owns_publication" in backend
    and "if (owns_publication)" in backend
    and "ggml_sycl_global_plan_snapshot()" in backend,
    "runtime context rejects foreign registry before mutation":
    "GGML_SYCL_LIFECYCLE_FOREIGN_BACKEND" in public
    and "ggml_backend_dev_backend_reg(backend->device) != ggml_backend_sycl_reg()" in exact_runtime_body
    and exact_runtime_body.index("return GGML_SYCL_LIFECYCLE_FOREIGN_BACKEND") <
        exact_runtime_body.index("registry.prepare_live_update(token)"),
    "weight usage ABI wrapper and status mutation hold exact load effect":
    "void ggml_backend_sycl_register_weight_usage" in backend
    and "bool ggml_backend_sycl_try_register_weight_usage" in backend
    and "(void) ggml_backend_sycl_try_register_weight_usage" in backend
    and "acquire_load_effect(registry.bound_candidate())" in register_usage_body
    and "catch (...)" in register_usage_body
    and "usage-effect-abort-drain" in (root / "tests/test-sycl-lifecycle-load-txn.cpp").read_text(),
    "lease result snapshots storage ownership": "std::shared_ptr<void> storage_owner" in cache_hpp
    and re.search(r"result\.storage_owner\s*=\s*entry\.storage_owner", cache_cpp)
    and "fresh.storage_owner = std::move(result.storage_owner)" in
        (root / "ggml/src/ggml-sycl/mem-handle.cpp").read_text()
    and "from_weight_lease_snapshot" in (root / "ggml/src/ggml-sycl/cpu-dispatch.cpp").read_text(),
    "external host registrations remain borrowed": "non_owning_external_host = !allocation_owner" in cache_cpp
    and "test_external_host_registration_never_freed" in
        (root / "tests/test-sycl-reset-model-weight-lease-preserve.cpp").read_text(),
    "no parser-breaking shared-owner defaults": "std::shared_ptr<void> allocation_owner = {}" not in cache_hpp,
    "cache-owned preload allocation lifetime": "allocation_released_via_owner" in cache_hpp
    and "ggml_sycl_transfer_alloc_owner" in backend
    and "leased_storage_owner_" in (root / "ggml/src/ggml-sycl/mem-handle.hpp").read_text()
    and "entry.storage_owner.reset()" in cache_cpp,
    "abort removes exact pending cache ownership": "direct_expert_entries_.erase(it)" in cache_cpp
    and "direct_weight_entries_.erase(it)" in cache_cpp
    and "entry.owner_mask != 0" in cache_cpp
    and "non_owning_external_host" in cache_cpp,
    "transaction-scoped cache ownership promotion": "pending_load_txn_id" in cache_hpp
    and "test_shared_entry_exact_two_owner_unload" in
        (root / "tests/test-sycl-reset-model-weight-lease-preserve.cpp").read_text()
    and "test_unrelated_thread_entry_not_claimed" in
        (root / "tests/test-sycl-reset-model-weight-lease-preserve.cpp").read_text()
    and "pair.second.pending_load_txn_id != load_txn_id" in cache_cpp
    and "acquire_bound_load_effect" in cache_cpp
    and "registry.acquire_load_effect(txn)" in cache_cpp
    and "g_pending_load_txn_id" not in cache_cpp
    and "unified_cache_note_model_load_abort" in backend
    and "unified_cache_note_model_load_end(ticket.token.owner.slot, ticket.token.load.value)" in backend,
    "DL context model-bound route": "llama_context_sycl_runtime_proc" in
    (root / "src/llama-context.cpp").read_text()
    and "ggml_backend_reg_get_proc_address" in (root / "src/llama-context.cpp").read_text()
    and "runtime_context_fn(backend.get(), token" in (root / "src/llama-context.cpp").read_text(),
    "durable quarantine reaper": all(
        x in backend
        for x in (
            "g_sycl_quarantine_queue",
            "g_sycl_quarantine_queue.take_all",
            "g_sycl_quarantine_queue.enqueue",
            "model_quarantine_token",
        )
    ),
    "late poison cleanup authority": "cleanup_required" in hpp
    and "finalize_cleanup" in cpp,
    "exact quarantine validation": "is_quarantined" in hpp
    and "tokens_[i] == token" in cpp
    and "operator==(ModelToken a, ModelToken b)" in hpp,
    "no all-device graph sweep": "release_graph_replay_leases_all_devices"
    not in backend,
    "bounded quarantine shutdown": "quarantine_drain_shutdown" in backend
    and "max_passes" in backend,
    "dynamic model lifecycle callbacks": "hooks.begin" in llama
    and "hooks.end" in llama
    and "hooks.unload" in llama
    and "static const llama_model_sycl_lifecycle_hooks" not in llama
    and "ggml_backend_reg_get_proc_address" in llama
    and "defined(GGML_BACKEND_DL)" in llama,
    "backend destructor admission tail": backend.index("delete backend;")
    < backend.rindex("global_registry().release_backend_context();")
    and "test_block_backend_context_release" in hpp
    and "checked unload crossed final backend destructor tail" in
        (root / "ggml/src/ggml-sycl/tests/test-model-lifecycle-runtime.cpp").read_text(),
    "runtime wrapper links aggregate registry in static and DL builds":
        sycl_cmake.count("target_link_libraries(test-sycl-lifecycle-runtime-wrapper PRIVATE ggml") == 2
        and "propagates the selected SYCL/CPU backend libraries" in sycl_cmake,
    "registry builtins initialize after object construction": "ggml_backend_registry() = default" in registry_backend
    and "void register_builtin_backends()" in registry_backend
    and "ggml_backend_reg_dev_count_unchecked(reg, &count)" in registry_backend
    and "ggml_backend_reg_dev_get_unchecked(reg, i, &device)" in registry_backend
    and "device->reg != reg" not in registry_backend
    and registry_backend.index("const bool is_new = existing == backends.end()")
        < registry_backend.index("backends.reserve(backends.size() + (is_new ? 1 : 0))")
    and "reactivation_rollback_guard" in registry_backend
    and "builtin_init_state::INITIALIZING && builtin_owner == std::this_thread::get_id()" in registry_backend
    and "reg.register_builtin_backends();" in registry_backend
    and "generic registry tombstone/failure fixture" in
        (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text()
    and "initial_reg_count < 2 || ggml_backend_reg_by_name(\"SYCL\") == nullptr" in
        (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text(),
    "commit failure retains device tombstone identity": "Keep staged device identities linked to their REMOVED" in registry_backend
    and "published_entry->state = ggml_backend_reg_state::REMOVED" in registry_backend,
    "pre-registry buffer adoption and durable events": "ggml_backend_refresh_buffer_lifecycle" in backend_base
    and "adoption_candidate" in backend_base
    and "g_live_events" in backend_base
    and "legacy_v2_event_layout" in (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text()
    and "sizeof(legacy_v2_event_layout) == sizeof(ggml_backend_event)" in
        (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text()
    and "pre_registry_buffer" in (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text()
    and "live backend event did not block checked unload" in
        (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text(),
    "registry name tombstone cache and controlled staging bypass": "ggml_backend_reg_name_unchecked" in backend_base
    and "g_registry_cached_name" in backend_base
    and "ggml_backend_registry_cached_name" in registry_backend
    and "std::string              cached_name" in registry_backend
    and "(*found)->cached_name.c_str()" in registry_backend
    and "Entry names are copied at staging and never mutated" in registry_backend
    and "const auto begin = g_registry_begin.load" in backend_base
    and "ggml_backend_reg_name_unchecked(reg)" in registry_backend
    and "!tombstoned_name" in (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text(),
    "base-safe lifecycle admission function table": "ggml_backend_set_registry_lifecycle" in backend_base
    and "g_registry_begin.load" in backend_base
    and "ggml_backend_registry_begin_call(reg)" not in backend_base
    and "ggml_backend_set_registry_lifecycle(&lifecycle_iface)" in registry_backend,
    "callbacks outside registry lock": "mutable std::mutex" in registry_backend
    and "Stage all plugin-owned metadata while mutation admission remains closed" in registry_backend
    and "Resolver is plugin code too" in registry_backend
    and "active_calls" in registry_backend
    and "cv.wait_for(lock, active_call_drain_timeout" in registry_backend
    and "mutable std::recursive_mutex" not in registry_backend
    and "module_operation_mutex" in registry_backend
    and "same-module load was not deferred or enumeration held the registry lock" in
        (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text(),
    "NODELETE MoE state reset": "ggml_sycl_reset_moe_module_state()" in backend
    and "g_moe_hybrid_init_success[d].store(false" in backend
    and "g_moe_expert_meta.clear()" in backend
    and "g_expert_groups.clear()" in backend
    and "g_expert_popularity.clear()" in backend
    and "g_adaptive_prestage.reset_after_stop()" in backend
    and "g_expert_predictors[d].reset()" in backend
    and "NODELETE reload retained model-bound MoE state" in
        (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text(),
    "process pinned raw handle lifetime": "Generic raw-handle policy: pin" in registry_backend
    and "dl_pin_library(handle.get(), path)" in registry_backend
    and "g_backend_raw_handle_leases" not in registry_backend
    and "g_dl_pinned_paths.count(key)" in backend_dl
    and backend_dl.count("g_dl_pinned_paths.insert(key)") == 2,
    "explicit registry tombstones": all(
        token in registry_backend for token in (
            "ggml_backend_reg_state::ACTIVE",
            "ggml_backend_reg_state::UNLOADING",
            "ggml_backend_reg_state::HIDDEN_FAILED",
            "ggml_backend_reg_state::REMOVED",
            "Never make\n            // this identity discoverable again",
        )
    ),
    "bounded stable enumeration and raced null": "g_backend_reg_enumeration.clear()" in registry_backend
    and "g_backend_dev_enumeration.clear()" in registry_backend
    and "state != ggml_backend_reg_state::ACTIVE" in registry_backend
    and "count snapshots captured before removal" in
        (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text(),
    "optional unload hooks cannot throw across C API": registry_backend.count("catch (...)") >= 10
    and "resolver_failed" in registry_backend
    and "cancel_noexcept" in registry_backend,
    "predictor ownership reset precedes cache shutdown": backend.index("ggml_sycl_reset_moe_module_state();\n    if (!ggml_sycl::shutdown_unified_cache())")
    > 0
    and "test_allocate_predictor_scores" in backend
    and "unified_alloc_validate_registry(-1, \"module-reload-clean\")" in backend,
    "reload validation does not create invalid caches": "active_devices = std::min(ggml_sycl_info().device_count" in cache_cpp
    and "tracked = arena_non_weight_used_locked(d)" in cache_cpp
    and "Validation is observational: never create a cache" in cache_cpp,
    "explicit arena shutdown releases registered chunks": "alloc_registry::instance().unregister_alloc(c.ptr);" in cache_cpp
    and "sycl::free(c.ptr, arena_queue_->get_context())" in cache_cpp
    and "pre-destroy queue drain failed" in cache_cpp
    and "logical unload retained device memory" in
        (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text(),
    "explicit cache ownership drain": "shutdown_shared_context_queues();" in cache_cpp
    and "g_live_arena_chunks.fetch_sub" in cache_cpp
    and "unified_cache_shutdown_state_clean()" in cache_cpp
    and "explicit shutdown postcondition failed after owner teardown" in cache_cpp
    and "arena chunk remained registered after unregister" in cache_cpp
    and "Destroyed %zu chunk(s), released %.1f MB" in cache_cpp
    and "unified_cache_shutdown_state_clean()" in backend,
    "backend construction publish rollback": "g_test_fail_next_backend_publish" in backend
    and "auto ctx = std::make_unique<ggml_backend_sycl_context>(device)" in backend
    and backend.index("auto sycl_backend = std::make_unique<ggml_backend>")
    < backend.index("g_backend_context_by_device[device] = ctx.get()")
    and "backend construction failure leaked publication or admission" in
        (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text()
    and "ggml_backend_t failed_backend = ggml_backend_sycl_init(0)" in
        (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text(),
    "atomic shutdown reservation": "reserve_shutdown()" in hpp
    and "shutdown_reserved_" in hpp
    and "backend_context_count_" in hpp
    and "global_registry().reserve_shutdown()" in backend
    and 'strcmp(name, "ggml_backend_sycl_can_unload")' in backend
    and 'strcmp(name, "ggml_backend_sycl_cancel_unload")' in backend
    and 'named_can_unload = ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_can_unload")'
        in (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text()
    and "ggml_backend_cancel_unload" in
        (root / "ggml/src/ggml-backend-reg.cpp").read_text()
    and "ggml_backend_complete_unload" in
        (root / "ggml/src/ggml-backend-reg.cpp").read_text()
    and "const auto settle_state" in registry_backend,
    "exhaustive saveable mutation and buffer callback admission": module_proc_inventory_ok
    and buft_callback_inventory_ok
    and buffer_callback_inventory_ok
    and "durable_owners" in registry_backend
    and "owner_lease" in backend_base
    and "ggml_backend_buffer_set_type" in backend_base
    and "ggml_backend_buffer_set_type(buffer, buft)" in backend
    and "consume_production_owner" in backend_base
    and backend_base.index("buffer_production_guard production(buft->device)")
        < backend_base.index("buft->iface.alloc_buffer(buft, size)")
    and "buffer_production_guard production(device)" in backend_base
    and "Underlying buffers already hold durable owners" in backend_base
    and "pre_registry_buffer" in (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text()
    and "live backend event did not block checked unload" in
        (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text()
    and "remaining host-compute buffer lost durable ownership" in
        (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text()
    and "sycl_module_admission_state::RETRY_CLOSED" in backend
    and "g_sycl_module_admission_cv.wait_for" in backend
    and backend.index("global_registry().reserve_shutdown()") < backend.index("g_sycl_module_admission_cv.wait_for")
    and "mutation drain timed out: calls=%zu txn=%llu models=%llu contexts=%llu" in backend
    and "g_sycl_module_shutdown_started = true" in backend
    and "if (!g_sycl_module_shutdown_started" in backend
    and "Reconcile the internal Registry gate" in backend
    and "CALL_SYCL(ggml_backend_sycl_model_load_end)(aborted, false, nullptr)" in
        (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text()
    and "if (!module_guard) return GGML_SYCL_LIFECYCLE_LOAD_BUSY" in backend
    and "ggml_backend_sycl_test_admission_snapshot" in backend
    and "ggml_backend_test_active_calls" in registry_backend
    and "shutdown reservation check failed: reserved=%d begin_rc=%d" in
        (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text()
    and (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text().count(
        "saved_model_load_begin(&stale_txn) != GGML_SYCL_LIFECYCLE_LOAD_BUSY") == 4
    and "ggml_backend_prepare_reactivate" in backend
    and "ggml_backend_commit_reactivate" in backend
    and "ggml_backend_rollback_reactivate" in backend,
    "completion, callback, arena retry and renamed DSO coverage": "shutdown_completed_" in hpp
    and "complete_shutdown()" in backend
    and "ggml_backend_commit_reactivate" in backend
    and "device_call_guard" in (root / "ggml/src/ggml-backend.cpp").read_text()
    and "failure-window saved model-load procedure" in (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text()
    and "saved buffer-type path allocated after completed unload" in
        (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text()
    and "reactivation staging failure published or reopened module" in
        (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text()
    and "measured_cache_drop" in (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text()
    and "GGML_SYCL_RENAMED_RUNTIME_MODULE" in (root / "ggml/src/ggml-sycl/CMakeLists.txt").read_text(),
    "generation tombstones retained": "current_generation" in registry_backend
    and "find_device_identity" in registry_backend
    and "Prefer" in registry_backend and "current committed row" in registry_backend
    and "Retain every prior generation row forever" in registry_backend
    and "devices.erase(std::remove_if" not in registry_backend,
    "synchronized buffer/event adoption": "live_owner_state::ADOPTING" in backend_base
    and "g_live_owner_cv.wait" in backend_base
    and "g_live_buffers.erase(found)" in backend_base
    and "g_live_events.erase(found)" in backend_base
    and "ggml_backend_test_owner_adoption_blocked" in
        (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text()
    and "overlapping_free.wait_for" in
        (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text()
    and "static auto * store = new live_owner_store" in backend_base
    and "Do not adopt here: first installation occurs inside get_reg()" in backend_base
    and "Built-in devices are now published; transactionally backfill buffers" in registry_backend
    and "buffer free timed out waiting for lifecycle adoption" in backend_base
    and "event free timed out waiting for lifecycle adoption" in backend_base
    and "generic fixture: await adoption barrier" in
        (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text()
    and "verify adopted event durable owner" in
        (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text()
    and "ggml_backend_test_durable_owners(reg) != 0" in
        (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text()
    and "await active callback entry" in
        (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text()
    and "callback-drain unload joined" in
        (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text(),
    "cache shutdown owner snapshot avoids rw-lock callbacks":
        "std::unordered_map<int, std::shared_ptr<unified_cache>> caches" in cache_cpp
        and cache_cpp.index("caches = g_device_caches")
            < cache_cpp.index("item.second->shutdown_resources()"),
    "bounded retained handle drain": "wait_for(lock, std::chrono::milliseconds(timeout_ms)" in
        (root / "ggml/src/ggml-sycl/mem-handle.cpp").read_text()
        and "SYCL retained-handle drain timed out" in backend,
    "retained cleanup precedes cache shutdown": backend.index("ggml_sycl_cpu_retained_cleanup()")
        < backend.index("ggml_sycl::shutdown_unified_cache()")
        and "offload_buffer_pool_shutdown()" in backend
        and backend.index("drain_retained_handles(true, 10000)", backend.index("void ggml_backend_sycl_shutdown"))
            < backend.index("ggml_sycl_cpu_retained_cleanup()", backend.index("void ggml_backend_sycl_shutdown"))
        and "!ggml_sycl_cpu_retained_active()" in backend
        and "g_runtime_alloc_registry.empty()" in cache_cpp
        and "g_offload_pool_slots.empty()" in cache_cpp
        and "g_offload_pool_slots.clear()" in cache_cpp
        and cache_cpp.index("caches.clear();", cache_cpp.index("bool shutdown_unified_cache()"))
            < cache_cpp.index("shutdown_shared_context_queues();", cache_cpp.index("bool shutdown_unified_cache()"))
            < cache_cpp.index("unified_cache_shutdown_state_clean();", cache_cpp.index("bool shutdown_unified_cache()")),
    "preflight, bounded generic drain, hidden reactivation and score pin":
        registry_backend.index("reserved = can_unload()") < registry_backend.index("cv.wait_for(lock, active_call_drain_timeout")
        and "ggml_backend_reg_state::REACTIVATING" in registry_backend
        and "COMMIT_REACTIVATE_THROW" in (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text()
        and "g_commit_reactivate_block" in (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text()
        and registry_backend.index("dl_pin_library(handle.get(), entry.path())")
            < registry_backend.index('dl_get_sym(handle.get(), "ggml_backend_score")',
                                     registry_backend.index("dl_pin_library(handle.get(), entry.path())"))
        and backend_dl.count("Pin before the first plugin-owned policy/score/probe callback") == 2
        and backend_dl.index("dl_pin_library(handle, path)") < backend_dl.index("(void) policy()"),
    "count/get census registered for every configuration": "add_test(NAME test-backend-count-get-null-census" in
        (root / "tests/CMakeLists.txt").read_text(),
    "dynamic runtime wrapper": "if (GGML_BACKEND_DL)"
    in (root / "ggml/src/ggml-sycl/CMakeLists.txt").read_text()
    and "GGML_SYCL_RUNTIME_MODULE" in (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text()
    and "ggml_backend_unload_checked(reg)" in (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text()
    and (root / "tests/test-sycl-lifecycle-runtime-wrapper.cpp").read_text().count("ggml_backend_load(") >= 2,
    "canonical G1 registration": "sycl-lifecycle-gpu-sequential"
    in (root / "tests/CMakeLists.txt").read_text()
    and "infer(o.a_shared, o, selected)"
    in (root / "tests/test-sycl-lifecycle-gpu-sequential.cpp").read_text()
    and "GGML_SYCL_G1_MODEL_A"
    not in (root / "tests/test-sycl-lifecycle-g1-aba.sh").read_text(),
}
failed = [name for name, ok in checks.items() if not ok]
if failed:
    print("lifecycle source contract failed: " + ", ".join(failed), file=sys.stderr)
    raise SystemExit(1)
print("lifecycle source contract: PASS")
