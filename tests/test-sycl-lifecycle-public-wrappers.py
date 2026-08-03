#!/usr/bin/env python3
"""Host/source contract for the public C wrapper and llama RAII integration."""
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
api = (root / "ggml/include/ggml-sycl.h").read_text()
backend = (root / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()
llama = (root / "src/llama-model.cpp").read_text()
required = {
    "public full token": all(x in api for x in ("model_id;", "load_txn_id;", "slot_generation;")),
    "public typed failures": all(x in api for x in ("LIFECYCLE_NULL_OUTPUT", "LIFECYCLE_ALLOCATION_FAILED", "LIFECYCLE_EFFECT_FAILED", "LIFECYCLE_BUSY")),
    "wrapper two phase": all(x in backend for x in ("registry.prepare_end", "registry.finalize_end", "registry.prepare_teardown", "registry.finalize_teardown")),
    "wrapper terminal replay token": "ggml_sycl_export_token(ticket.replay.token, model)" in backend,
    "wrapper catches": backend.count("catch (...)") >= 5,
    "legacy abort default diagnostic": "deprecated bool model-load boundary called" in backend,
    "guard noncopyable": "operator=(const llama_model_sycl_loading_guard &) = delete" in llama,
    "guard abort default": "ggml_backend_sycl_model_load_end(txn, false" in llama,
    "guard explicit success": "sycl_model_loading_guard.finish(true)" in llama,
    "teardown failure logged": "SYCL model teardown failed" in llama,
}
failed = [k for k, v in required.items() if not v]
if failed:
    print("public wrapper/guard contract failed: " + ", ".join(failed), file=sys.stderr)
    raise SystemExit(1)
print("public wrapper/guard contract: PASS")
