#!/usr/bin/env python3
from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parents[1]
hpp = (root / "ggml/src/ggml-sycl/model-lifecycle.hpp").read_text()
cpp = (root / "ggml/src/ggml-sycl/model-lifecycle.cpp").read_text()
backend = (root / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()
public = (root / "ggml/include/ggml-sycl.h").read_text()
llama = (root / "src/llama-model.cpp").read_text()
cache_hpp = (root / "ggml/src/ggml-sycl/unified-cache.hpp").read_text()
placement_production = "\n".join(
    (root / path).read_text()
    for path in (
        "ggml/src/ggml-sycl/ggml-sycl.cpp",
        "ggml/src/ggml-sycl/unified-cache.hpp",
        "ggml/src/ggml-sycl/unified-cache.cpp",
        "ggml/src/ggml-sycl/model-lifecycle.hpp",
        "ggml/src/ggml-sycl/model-lifecycle.cpp",
    )
)
legacy_global_re = re.compile(r"\bg_(?:has_)?placement_plan\b")
legacy_cache_re = re.compile(r"(?:->|\.)get_placement_plan\s*\(")
# Positive controls prove the census expressions still detect both prohibited
# reader families rather than passing because of an accidentally-empty regex.
census_fixture = (
    "if (g_has_placement_plan) use(g_placement_plan); cache->get_placement_plan();"
)
checks = {
    "full slot token": "struct SlotToken { uint32_t slot" in hpp
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
    "abort default guard": "ggml_backend_sycl_model_load_end(txn, false" in llama,
    "noncopyable guard": "llama_model_sycl_loading_guard(const llama_model_sycl_loading_guard &) = delete"
    in llama,
    "explicit outer success": "sycl_model_loading_guard.finish(true)" in llama,
    "no-alloc success": "if (ml.no_alloc)" in llama
    and "sycl_model_loading_guard.finish(true)" in llama,
    "generation-safe teardown": "ggml_backend_sycl_model_unloaded_token(token)"
    in llama,
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
    "versioned atomic plan publication": all(
        x in backend
        for x in (
            "g_placement_publication",
            "atomic_store_explicit",
            "set_placement_plan_snapshot",
        )
    )
    and all(
        x in (root / "ggml/src/ggml-sycl/unified-cache.hpp").read_text()
        for x in (
            "uint64_t                              version",
            "atomic_load_explicit",
            "shared_ptr<const lifecycle_plan_snapshot>",
        )
    ),
    "no unsafe global placement readers": not legacy_global_re.search(
        placement_production
    ),
    "no unsafe cache placement readers": not legacy_cache_re.search(
        placement_production
    ),
    "global census positive control": len(legacy_global_re.findall(census_fixture))
    == 2,
    "cache census positive control": len(legacy_cache_re.findall(census_fixture)) == 1,
    "all cache readers retain owners": backend.count("ggml_sycl_cache_plan_owner(")
    >= 49,
    "cache snapshot identity validation": "lifecycle_plan_snapshot_matches(global, cached)"
    in backend
    and all(
        x in cache_hpp
        for x in (
            "authority->version == cache->version",
            "authority->model_id == cache->model_id",
            "authority->load_txn_id == cache->load_txn_id",
            "authority->slot_generation == cache->slot_generation",
        )
    ),
    "same snapshot cache publication": backend.count("set_placement_plan_snapshot")
    >= 5,
    "no cache plan reference accessor": "const placement_plan & " + "get_placement_plan"
    not in cache_hpp,
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
    "fallible pre-finalize restoration": backend.index("teardown_owner_effects(owner)")
    < backend.index("finalize_teardown(ticket, true)"),
    "durable quarantine reaper": all(
        x in backend
        for x in (
            "g_sycl_quarantine_tokens",
            "ggml_sycl_quarantine_reap",
            "model_quarantine_token",
        )
    ),
    "late poison cleanup authority": "cleanup_required" in hpp
    and "finalize_cleanup" in cpp,
    "exact quarantine validation": "is_quarantined" in hpp
    and "slot_generation == token.slot_generation" in backend,
    "no all-device graph sweep": "release_graph_replay_leases_all_devices"
    not in backend,
    "bounded quarantine shutdown": "quarantine_drain_shutdown" in backend
    and "max_passes" in backend,
    "canonical G1 registration": "sycl-lifecycle-gpu-sequential"
    in (root / "tests/CMakeLists.txt").read_text()
    and "GGML_SYCL_G1_MODEL_A"
    not in (root / "tests/test-sycl-lifecycle-g1-aba.sh").read_text(),
}
failed = [name for name, ok in checks.items() if not ok]
if failed:
    print("lifecycle source contract failed: " + ", ".join(failed), file=sys.stderr)
    raise SystemExit(1)
print("lifecycle source contract: PASS")
