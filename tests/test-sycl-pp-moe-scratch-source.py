#!/usr/bin/env python3
"""Host-only PP-MoE scratch ownership/lifecycle source contract.

The ring has two correlated identities: the runtime slot's generation and the
cache reservation's exact (slot, generation).  Reuse and teardown must release
that exact cache owner, and retired backing owners must be destroyed before raw
pointer metadata is cleared.  Arena lifetime is authority-driven; the removed
arena_generation_bump helper is not an acceptable substitute.
"""
import argparse
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--header", default=str(root / "ggml/src/ggml-sycl/unified-cache.hpp"))
parser.add_argument("--cache", default=str(root / "ggml/src/ggml-sycl/unified-cache.cpp"))
parser.add_argument("--backend", default=str(root / "ggml/src/ggml-sycl/ggml-sycl.cpp"))
parser.add_argument("--self-test", action="store_true")
args = parser.parse_args()


def function_body(source, signature):
    start = source.find(signature)
    if start < 0:
        return ""
    brace = source.find("{", start)
    if brace < 0:
        return ""
    depth = 0
    for pos in range(brace, len(source)):
        if source[pos] == "{":
            depth += 1
        elif source[pos] == "}":
            depth -= 1
            if depth == 0:
                return source[start:pos + 1]
    return ""


def precedes(text, first, second):
    a, b = text.find(first), text.find(second)
    return a >= 0 and b >= 0 and a < b


def enclosing_block_ends_before(text, needle, later):
    """Prove the lexical block containing needle closes before later."""
    pos, later_pos = text.find(needle), text.find(later)
    if pos < 0 or later_pos < 0:
        return False
    block = text.rfind("{", 0, pos)
    depth = 0
    for end in range(block, len(text)):
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
            if depth == 0:
                return end < later_pos
    return False


def evaluate(header, cache, backend):
    claim = function_body(backend, "static bool pp_moe_onednn_claim_scratch_slot(")
    drain = function_body(backend, "static void pp_moe_onednn_drain_scratch_slots(")
    record = function_body(backend, "static void pp_moe_onednn_record_scratch_slot_event(")
    cache_release = function_body(cache, "void unified_cache::release_pp_moe_onednn_scratch_slot(")
    anchors = {
        "runtime claim": claim,
        "runtime drain": drain,
        "runtime record": record,
        "cache exact release": cache_release,
    }
    checks = {
        "slot struct carries generation/refcount":
            "uint64_t   generation" in header and "uint32_t   refcount" in header,
        "claim and exact release APIs exist":
            "claim_pp_moe_onednn_scratch_slot" in header
            and "release_pp_moe_onednn_scratch_slot(uint32_t slot, uint64_t generation)" in header,
        "retired backing list preserves busy generations":
            "pp_moe_onednn_retired_slots_" in header
            and "pp_moe_onednn_retired_slots_.push_back(std::move(slot))" in cache,
        "runtime state carries generation and retained owners":
            "std::vector<uint64_t>" in backend and "generations;" in backend
            and "std::vector<std::vector<ggml_sycl::mem_handle>> retained_owners;" in backend,
        "record revalidates the exact generation before publication":
            "state.generations[slot] != generation" in record
            and precedes(record, "state.generations[slot] != generation", "state.done_events[slot]"),
        "completed reuse releases the exact cache slot generation":
            "completed_generation = state.generations[slot]" in claim
            and "release_pp_moe_onednn_scratch_slot(slot, completed_generation)" in claim,
        "drain revalidates epoch and releases the exact cache owner":
            "state.generations[wait_slot] != wait_generation" in drain
            and "release_pp_moe_onednn_scratch_slot(wait_slot, wait_generation)" in drain
            and precedes(drain, "state.generations[wait_slot] != wait_generation",
                         "release_pp_moe_onednn_scratch_slot(wait_slot, wait_generation)"),
        "cache release matches both slot and generation":
            "candidate.slot != slot || candidate.generation != generation" in cache_release,
        "retired owners are destroyed before pointer metadata is cleared":
            precedes(cache_release, "retired.weight_owner     = {};",
                     "retired.weight = retired.activation = retired.output = nullptr")
            and precedes(cache_release, "retired.activation_owner = {};",
                         "retired.weight = retired.activation = retired.output = nullptr")
            and precedes(cache_release, "retired.output_owner     = {};",
                         "retired.weight = retired.activation = retired.output = nullptr"),
        "owner destruction happens after the scratch mutex is released":
            enclosing_block_ends_before(cache_release,
                                        "std::lock_guard<std::mutex> lock(pp_moe_onednn_scratch_mutex_)",
                                        "release_releasable(releasable)"),
        "claim rollback clears generation-zero reservations without waiting":
            "if (state.generations[slot] == 0)" in claim
            and "state.busy[slot]        = 0;" in claim,
        "slot owners remain retained to the done event":
            "bind_owners(std::move(pp_moe_slot_owners))" in backend
            and "state.retained_owners[slot]  = std::move(owners)" in record,
        "legacy arena generation bump helper is absent":
            all("arena_generation_bump" not in source for source in (header, cache, backend)),
    }
    missing = sorted(name for name, body in anchors.items() if not body)
    failed = sorted(name for name, ok in checks.items() if not ok)
    return missing, failed, len(checks)


MUTANTS = {
    "record revalidates the exact generation before publication":
        ("backend", "if (state.generations[slot] != generation) {", "if (false) {"),
    "completed reuse releases the exact cache slot generation":
        ("backend", "cache->release_pp_moe_onednn_scratch_slot(slot, completed_generation);", ""),
    "drain revalidates epoch and releases the exact cache owner":
        ("backend", "if (ring_depth != state.ring_depth || wait_slot >= state.generations.size() ||\n                state.generations[wait_slot] != wait_generation) {",
         "if (false) {"),
    "cache release matches both slot and generation":
        ("cache", "candidate.slot != slot || candidate.generation != generation", "candidate.slot != slot"),
    "retired owners are destroyed before pointer metadata is cleared":
        ("cache", "retired.weight_owner     = {};", ""),
    "legacy arena generation bump helper is absent":
        ("cache", "void unified_cache::release_pp_moe_onednn_scratch_slot(uint32_t slot, uint64_t generation) {",
         "void unified_cache::release_pp_moe_onednn_scratch_slot(uint32_t slot, uint64_t generation) {\n    arena_generation_bump();"),
}


def self_test(header, cache, backend):
    problems = []
    originals = {"header": header, "cache": cache, "backend": backend}
    for check, (target, anchor, replacement) in MUTANTS.items():
        if anchor not in originals[target]:
            problems.append("mutation anchor missing: " + check)
            continue
        sources = dict(originals)
        sources[target] = sources[target].replace(anchor, replacement, 1)
        _, failed, _ = evaluate(sources["header"], sources["cache"], sources["backend"])
        if check not in failed:
            problems.append("check did not fire: " + check)
    return problems


header = Path(args.header).read_text()
cache = Path(args.cache).read_text()
backend = Path(args.backend).read_text()
missing, failed, count = evaluate(header, cache, backend)
for name in missing:
    print("MISSING ANCHOR: " + name, file=sys.stderr)
for name in failed:
    print("FAIL: " + name, file=sys.stderr)
if missing or failed:
    raise SystemExit(1)
if args.self_test:
    problems = self_test(header, cache, backend)
    for problem in problems:
        print("SELF-TEST FAIL: " + problem, file=sys.stderr)
    if problems:
        raise SystemExit(1)
    print(f"pp-moe scratch source contract: self-test PASS ({len(MUTANTS)} mutants)")
print(f"pp-moe scratch source contract: PASS ({count} checks)")
