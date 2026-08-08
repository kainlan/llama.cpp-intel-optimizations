#!/usr/bin/env python3
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
cache_hpp = (root / "ggml/src/ggml-sycl/unified-cache.hpp").read_text()
cache_cpp = (root / "ggml/src/ggml-sycl/unified-cache.cpp").read_text()
sycl_cpp = (root / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()

checks = {
    "slot struct carries generation/refcount": "uint64_t   generation" in cache_hpp and "uint32_t   refcount" in cache_hpp,
    "claim api exists": "claim_pp_moe_onednn_scratch_slot" in cache_hpp and "claim_pp_moe_onednn_scratch_slot" in cache_cpp,
    "release api matches generation": "release_pp_moe_onednn_scratch_slot(uint32_t slot, uint64_t generation)" in cache_hpp and "candidate.generation != generation" in cache_cpp,
    "retired backing list exists": "pp_moe_onednn_retired_slots_" in cache_hpp,
    "reserve retires old slots instead of freeing under lock": "pp_moe_onednn_retired_slots_.push_back" in cache_cpp,
    "runtime slot state tracks generations": "std::vector<uint64_t>                       generations;" in sycl_cpp or "std::vector<uint64_t>    generations;" in sycl_cpp,
    "activate binds generation before record": "pp_moe_onednn_bind_scratch_slot_generation" in sycl_cpp,
    "record path matches generation": "state.generations[slot] != generation" in sycl_cpp,
    "claim rollback failpoint exists": "FAIL_AFTER_LOCAL_RESERVATION" in sycl_cpp and "pp_moe_onednn_rollback_unbound_scratch_slot" in sycl_cpp,
    "generation0 busy slots never wait": "if (state.generations[slot] == 0)" in sycl_cpp and "state.busy[slot]        = 0;" in sycl_cpp,
    "slot owners retained to done event": "std::vector<std::vector<ggml_sycl::mem_handle>> retained_owners;" in sycl_cpp and "pp_moe_onednn_scratch_claim_state.bind_owners(std::move(pp_moe_slot_owners));" in sycl_cpp,
    # llama.cpp-u2mz: the first clause used to be `sycl_cpp.count("arena_generation_bump") >= 0`,
    # which is true for every possible input -- and arena_generation_bump does not occur in
    # ggml-sycl.cpp at all. Only the unified-cache.cpp clause ever carried information.
    "arena generation bumped on reset": cache_cpp.count("arena_generation_bump();") >= 3,
}

failed = [name for name, ok in checks.items() if not ok]
if failed:
    print("pp-moe scratch source contract failed: " + ", ".join(failed), file=sys.stderr)
    raise SystemExit(1)
print("pp-moe scratch source contract: PASS")
