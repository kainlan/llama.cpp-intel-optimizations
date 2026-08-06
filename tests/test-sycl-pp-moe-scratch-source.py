#!/usr/bin/env python3
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
cache_hpp = (root / "ggml/src/ggml-sycl/unified-cache.hpp").read_text()
cache_cpp = (root / "ggml/src/ggml-sycl/unified-cache.cpp").read_text()
sycl_cpp = (root / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()
cpp_test = (root / "tests/test-sycl-pp-moe-scratch-lifecycle.cpp").read_text()

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
    "arena generation bumped on reset": sycl_cpp.count("arena_generation_bump") >= 0 and cache_cpp.count("arena_generation_bump();") >= 3,
    "cpu test has cross-thread completion case": "cross-thread-completion" in cpp_test,
    "cpu test has resize-in-flight case": "resize-in-flight" in cpp_test,
    "cpu test has stale-release case": "stale-release" in cpp_test,
    "cpu test has arena reset/destroy case": "arena-reset-destroy" in cpp_test,
    "cpu test has rollback failpoint case": "rollback-after-local-reservation" in cpp_test and "test_fail_after_local_reservation_rolls_back_busy_generation0" in cpp_test,
    "cpu test has ring-depth+1 pressure case": "ring-depth-plus-one-pressure" in cpp_test and "test_ring_depth_plus_one_pressure_does_not_deadlock_generation0_slots" in cpp_test,
    "cpu test has mutation case": "M1" in cpp_test and "mutation_stale_release_clears_new_generation" in cpp_test,
}

failed = [name for name, ok in checks.items() if not ok]
if failed:
    print("pp-moe scratch source contract failed: " + ", ".join(failed), file=sys.stderr)
    raise SystemExit(1)
print("pp-moe scratch source contract: PASS")
