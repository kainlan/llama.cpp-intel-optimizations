#!/usr/bin/env python3
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
hpp = (root / "ggml/src/ggml-sycl/model-lifecycle.hpp").read_text()
cpp = (root / "ggml/src/ggml-sycl/model-lifecycle.cpp").read_text()
llama = (root / "src/llama-model.cpp").read_text()
checks = {
    "full slot token": "struct SlotToken { uint32_t slot" in hpp and "uint64_t generation" in hpp,
    "immutable publication": "std::shared_ptr<const ModelState>" in hpp,
    "checked IDs": "class CheckedCounter" in hpp and "ID_EXHAUSTED" in cpp,
    "explicit nested transaction": "enter_nested(LoadTxnId" in hpp,
    "abort default guard": "~llama_model_sycl_loading_guard() { finish(false); }" in llama,
    "explicit outer success": "sycl_model_loading_guard.finish(true)" in llama,
    "generation-safe teardown": "ggml_backend_sycl_model_unloaded_token(token)" in llama,
    "no legacy load guard": "ggml_backend_sycl_set_model_loading(" not in llama,
    "mutation hooks": all(x in hpp for x in ("M1_SKIP_GENERATION", "M2_NESTED_COMMIT", "M3_CLEAR_POISON")),
}
failed = [name for name, ok in checks.items() if not ok]
if failed:
    print("lifecycle source contract failed: " + ", ".join(failed), file=sys.stderr)
    raise SystemExit(1)
print("lifecycle source contract: PASS")
