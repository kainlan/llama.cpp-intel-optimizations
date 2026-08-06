#!/usr/bin/env python3
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
cache_hpp = (root / "ggml/src/ggml-sycl/unified-cache.hpp").read_text()
cache_cpp = (root / "ggml/src/ggml-sycl/unified-cache.cpp").read_text()
sycl_cpp = (root / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()

checks = {
    "logical token declared": "struct onednn_scratch_token" in cache_hpp,
    "result carries token": "onednn_scratch_token token{};" in cache_hpp,
    "release takes token": "void unified_cache_release_onednn_scratch(int device_id, onednn_scratch_token token);" in cache_hpp,
    "cross-thread unique_lock registry removed": "g_onednn_scratch_locks" not in cache_cpp and "std::unordered_map<int, std::unique_lock<std::mutex>>" not in cache_cpp,
    "reservation generation state present": "onednn_scratch_generation_" in cache_hpp and "onednn_scratch_refcount_" in cache_hpp and "onednn_scratch_cv_" in cache_hpp,
    "acquire waits on logical reservation": "acquire_onednn_scratch_reservation" in cache_cpp and "onednn_scratch_cv_.wait" in cache_cpp,
    "release validates generation": "onednn_scratch_generation_ != token.generation" in cache_cpp,
    "guard stores token not raw device lock": "onednn_scratch_token token{};" in sycl_cpp and "scratch_guard.arm(device_id, scratch.token);" in sycl_cpp,
}

failed = [name for name, ok in checks.items() if not ok]
if failed:
    print("onednn scratch source contract failed: " + ", ".join(failed), file=sys.stderr)
    raise SystemExit(1)
print("onednn scratch source contract: PASS")
