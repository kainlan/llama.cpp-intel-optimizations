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
import re
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


def exact_generation_guard_dominates_refcount_mutation(cache_release):
    """All destructive refcount writes must follow the exact slot+generation reject guard."""
    release_match = function_body(cache_release, "auto release_match =")
    if not release_match:
        return False
    guard = re.search(
        r"if\s*\(\s*candidate\.slot\s*!=\s*slot\s*\|\|\s*"
        r"candidate\.generation\s*!=\s*generation\s*\)\s*\{\s*continue\s*;\s*\}",
        release_match, re.S)
    if not guard:
        return False
    mutations = list(re.finditer(
        r"(?:candidate\.refcount\s*(?:--|\-=|=(?!=))|--\s*candidate\.refcount)", release_match))
    return bool(mutations) and all(guard.end() < mutation.start() for mutation in mutations)


OWNERS_TAKEN = re.compile(r"batched_scratch_claim\.take_owners\(reserved\);")
LEGACY_CLAIM_DECL = re.compile(r"pp_moe_onednn_scratch_claim\s+pp_moe_onednn_scratch_claim_state;")
LEGACY_RESULT_DECL = re.compile(r"ggml_sycl::pp_moe_onednn_scratch_result\s+pp_moe_onednn_scratch;")
LEGACY_OWNER_MOVE = re.compile(r"pp_moe_onednn_scratch\.(\w+_owner)\s*=\s*std::move\(reserved\.\1\);")
LEGACY_OWNER_COPY = re.compile(r"pp_moe_onednn_scratch\.\w+_owner\s*=\s*reserved\.\w+_owner;")
LEGACY_REFUSED_RELEASE = re.compile(r"pp_moe_onednn_scratch = \{\};\s*"
                                    r"pp_moe_onednn_scratch_claim_state\.release_unused\(\);")


def evaluate(header, cache, backend):
    claim = function_body(backend, "static bool pp_moe_onednn_claim_scratch_slot(")
    drain = function_body(backend, "static void pp_moe_onednn_drain_scratch_slots(")
    record = function_body(backend, "static void pp_moe_onednn_record_scratch_slot_event(")
    cache_release = function_body(cache, "void unified_cache::release_pp_moe_onednn_scratch_slot(")
    release_unused = function_body(backend, "void release_unused() {")
    take_owners = function_body(backend, "void take_owners(")
    activation = backend.find("batched_scratch_claim.activate(")
    batched_from_activation = backend[activation:] if activation >= 0 else ""
    anchors = {
        "runtime claim": claim,
        "runtime drain": drain,
        "runtime record": record,
        "cache exact release": cache_release,
        "unused claim release": release_unused,
        "claim takes owners": take_owners,
        "batched claim activation": batched_from_activation,
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
        "exact generation guard dominates every refcount mutation":
            exact_generation_guard_dominates_refcount_mutation(cache_release),
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
        # A copy of a slot's owner keeps its allocation, and the arena under it, alive. Once the cache counts the slot
        # free, a last release may release the ring and empty the arena, so no copy may outlive the slot's claim.
        "an unused claim drops its owners before the cache slot release":
            precedes(release_unused, "owners.clear();", "release_pp_moe_onednn_scratch_slot(slot, generation)")
            and release_unused.count("owners.clear();") == 1,
        "the batched claim holds the slot's owners from its activation":
            OWNERS_TAKEN.search(batched_from_activation) is not None
            and OWNERS_TAKEN.search(batched_from_activation).start()
            < batched_from_activation.find("batched_scratch_claim.release_unused();")
            and re.search(r"owners\.push_back\(std::move\(\*owner\)\);", take_owners) is not None,
        "the legacy claim is declared before the scratch result it outlives":
            LEGACY_CLAIM_DECL.search(backend) is not None and LEGACY_RESULT_DECL.search(backend) is not None
            and LEGACY_CLAIM_DECL.search(backend).start() < LEGACY_RESULT_DECL.search(backend).start(),
        "the legacy result moves the slot's owners out of the cache's claim copy":
            len(LEGACY_OWNER_MOVE.findall(backend)) == 3 and LEGACY_OWNER_COPY.search(backend) is None,
        "a refused legacy result drops its owners before the claim's release":
            LEGACY_REFUSED_RELEASE.search(backend) is not None,
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
    "exact generation guard dominates every refcount mutation":
        ("cache", "auto & candidate = slots[i];",
         "auto & candidate = slots[i];\n                candidate.refcount--;"),
    "retired owners are destroyed before pointer metadata is cleared":
        ("cache", "retired.weight_owner     = {};", ""),
    "an unused claim drops its owners before the cache slot release":
        ("backend", "        owners.clear();\n        if (ggml_sycl::unified_cache * cache",
         "        if (ggml_sycl::unified_cache * cache"),
    "the batched claim holds the slot's owners from its activation":
        ("backend", "batched_scratch_claim.take_owners(reserved);", ""),
    # The claim declared below the result: on an exit the claim then releases its slot while the result still holds
    # the slot's owners.
    "the legacy claim is declared before the scratch result it outlives":
        ("backend",
         "        pp_moe_onednn_scratch_claim             pp_moe_onednn_scratch_claim_state;\n"
         "        ggml_sycl::pp_moe_onednn_scratch_result pp_moe_onednn_scratch;",
         "        ggml_sycl::pp_moe_onednn_scratch_result pp_moe_onednn_scratch;\n"
         "        pp_moe_onednn_scratch_claim             pp_moe_onednn_scratch_claim_state;"),
    "the legacy result moves the slot's owners out of the cache's claim copy":
        ("backend", "= std::move(reserved.activation_owner);", "= reserved.activation_owner;"),
    "a refused legacy result drops its owners before the claim's release":
        ("backend", "pp_moe_onednn_scratch = {};", ""),
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
