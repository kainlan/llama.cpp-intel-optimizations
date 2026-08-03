#!/usr/bin/env python3
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
hpp = (root / "ggml/src/ggml-sycl/model-lifecycle.hpp").read_text()
cpp = (root / "ggml/src/ggml-sycl/model-lifecycle.cpp").read_text()
backend = (root / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()
public = (root / "ggml/include/ggml-sycl.h").read_text()
llama = (root / "src/llama-model.cpp").read_text()
checks = {
    "full slot token": "struct SlotToken { uint32_t slot" in hpp and "uint64_t generation" in hpp,
    "immutable publication": "std::shared_ptr<const ModelState>" in hpp,
    "two phase finish": all(x in hpp for x in ("COMMITTING", "ROLLING_BACK", "TEARING_DOWN", "prepare_end", "finalize_end", "prepare_teardown")),
    "checked IDs": "class CheckedCounter" in hpp and "ID_EXHAUSTED" in cpp,
    "explicit nested transaction": "enter_nested(LoadTxnId" in hpp,
    "abort default guard": "ggml_backend_sycl_model_load_end(txn, false" in llama,
    "noncopyable guard": "llama_model_sycl_loading_guard(const llama_model_sycl_loading_guard &) = delete" in llama,
    "explicit outer success": "sycl_model_loading_guard.finish(true)" in llama,
    "no-alloc success": "if (ml.no_alloc)" in llama and "sycl_model_loading_guard.finish(true)" in llama,
    "generation-safe teardown": "ggml_backend_sycl_model_unloaded_token(token)" in llama,
    "no legacy llama guard": "ggml_backend_sycl_set_model_loading(" not in llama,
    "public null-output result": "GGML_SYCL_LIFECYCLE_NULL_OUTPUT" in public and "model != nullptr" in backend,
    "C boundary catches": backend.count("catch (...)") >= 3,
    "full owner host rows": "ggml_sycl_release_host_weight_extras_for_owner" in backend,
    "mutation hooks": all(x in hpp for x in ("M1_SKIP_GENERATION", "M2_NESTED_COMMIT", "M3_CLEAR_POISON")),
    "durable terminal identities": "tombstone_limit_" not in hpp and "txns_ is the durable terminal identity table" in cpp,
    "waiters refind by id": "current = txns_.find(id.value)" in cpp,
    "quarantine never live": cpp.count("model_phase::QUARANTINED") >= 3,
    "committing teardown busy": "BUSY, token" in cpp,
    "immutable full plan candidate": all(x in backend for x in ("lifecycle_stage_placement_plan", "lifecycle_publish_placement_plan", "lifecycle_abort_placement_plan")),
    "atomic dying cache bit": "unified_cache_set_live_model_mask(live_after)" not in backend,
}
failed = [name for name, ok in checks.items() if not ok]
if failed:
    print("lifecycle source contract failed: " + ", ".join(failed), file=sys.stderr)
    raise SystemExit(1)
print("lifecycle source contract: PASS")
