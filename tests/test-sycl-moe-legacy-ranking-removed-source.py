#!/usr/bin/env python3
"""Source contract: the legacy expert ranking is gone, and the planner's block-keyed rank API stays.

Host-only: reads the SYCL backend sources, builds nothing, loads no model and touches no device.

The backend once ranked experts four ways that no plan ever read: a warmup profile in a table clamped to 128 x 256
(layers x experts), a gate-norm placement that filled it at the first graph, a periodic rerank on the decode submitting
thread, and rank writes keyed by an FNV hash of the tensor name. The planner (build_moe_triplet_groups) asks for ranks
by block number, so none of those ranks reached a plan. Their only consumers were the runtime prestage (a no-op under a
placement plan), the opt-in adaptive prestage and the eviction frequency term, which read the profile's epoch counts.
This gate holds the removal and what must survive it:

  L1  no removed symbol, environment switch or log tag appears anywhere in the backend sources (comments included),
      including the code that existed only to serve the removed prestage;
  L2  the block-keyed rank API stays: get_expert_popularity_rank, set_expert_popularity_rank and
      is_expert_popularity_initialized are declared once in unified-cache.hpp and defined once in ggml-sycl.cpp, and
      build_moe_triplet_groups still reads them;
  L3  ranks have one writer: the one set_expert_popularity_rank call of the backend sits in ggml_sycl_update_moe_hotset,
      which is defined once and called once, and keyed by its layer_id parameter;
  L4  no rank is looked up under the hash layer id: the backend's get_expert_popularity_rank calls never take a
      moe_cache_layer_id() value;
  L5  eviction carries no frequency term: unified_cache::evict_one reads no per-expert frequency;
  L6  the down-I8 hot set ranks only by usage score and the block-keyed rank: an expert is pushed only when the tensor
      has scores or the expert has a block-keyed rank, the rank comes from that lookup alone, and a tensor with
      neither returns before any materialization (it keeps its loaded layout until scores exist).

Checks run on comment-blanked text except L1, which reads the raw text so a comment cannot keep a removed name alive.
NOT VACUOUS: --self-test runs the gate on in-memory mutants of the real sources. Each mutant names the pin that must
catch it, and is reported if it survives, if a different pin catches it, or if its anchor no longer matches exactly
once (NOT APPLIED). ctest registers it with --self-test.

WHAT THIS DOES NOT PROVE. It reads text. That the placement and the tokens are unchanged is the lead's run on a device.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL_DIR = ROOT / "ggml/src/ggml-sycl"
BACKEND = "ggml-sycl.cpp"
CACHE_CPP = "unified-cache.cpp"
CACHE_HPP = "unified-cache.hpp"

# Symbols, switches and log tags of the removed mechanisms. A word-bounded match in any backend source is a failure.
REMOVED_WORDS = (
    "moe_warmup_state",
    "g_moe_warmup",
    "MOE_MAX_LAYERS",
    "MOE_MAX_EXPERTS",
    "moe_apply_popularity_placement",
    "moe_compute_gate_norm_placement",
    "moe_periodic_rerank",
    "moe_prestage_popular_experts",
    "ggml_sycl_legacy_prestage_enabled",
    "ggml_sycl_runtime_prestage_enabled",
    "g_prestage_completed",
    "GGML_SYCL_LEGACY_PRESTAGE",
    "GGML_SYCL_MOE_WARMUP_TOKENS",
    "adaptive_prestage_state",
    "g_adaptive_prestage",
    "GGML_SYCL_ADAPTIVE_PRESTAGE",
    "GGML_SYCL_ADAPTIVE_PRESTAGE_INTERVAL",
    "GGML_SYCL_ADAPTIVE_PRESTAGE_THRESHOLD",
    "get_expert_frequency",
    # The MoE reorder temp buffer, reserved for the prestage's GPU reorder and read by nothing else.
    "reserve_reorder_temp",
    "release_reorder_temp",
    "get_reorder_temp_buffer",
    "get_reorder_temp_size",
    "reorder_temp_buffer_",
    "ggml_sycl_gpu_reorder_disabled",
    "GGML_SYCL_NO_GPU_REORDER",
)
REMOVED_TAGS = ("[MOE-WARMUP]", "[MOE-GATE-NORM]", "[MOE-RERANK]", "[ADAPTIVE-PRESTAGE]")

RANK_API = (
    "int get_expert_popularity_rank(int layer_id, int expert_id)",
    "void set_expert_popularity_rank(int layer_id, int expert_id, int rank)",
    "bool is_expert_popularity_initialized()",
)
HOTSET_DEF = "static void ggml_sycl_update_moe_hotset("
HOTSET_CALL = "ggml_sycl_update_moe_hotset(cache, src0, extra, ids_host,"
HOTSET_WRITE = "ggml_sycl::set_expert_popularity_rank(layer_id, indices[i], static_cast<int>(i));"
TRIPLET_DEF = "static std::vector<moe_triplet_group> build_moe_triplet_groups("
EVICT_ONE_DEF = "size_t unified_cache::evict_one(size_t"
DOWN_I8_HOTSET_DEF = "static size_t ggml_sycl_materialize_moe_down_i8_hotset_for_tensor("
DOWN_I8_RULE = (
    "const int block_rank = block_layer >= 0 ? ggml_sycl::get_expert_popularity_rank(block_layer, item.expert_id) : -1;",
    "if (block_rank >= 0) { item.rank = block_rank; }",
    "if (have_scores || item.rank != INT_MAX) { ranked.push_back(item); }",
    "if (ranked.empty()) { return 0; }",
)


class ContractError(AssertionError):
    pass


class MutantNotApplied(Exception):
    pass


_COMMENT_OR_LITERAL = re.compile(r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'", re.DOTALL)


def blank_comments(text: str) -> str:
    """Comments become spaces (newlines kept); string and character literals stay as they are."""

    def repl(m: re.Match) -> str:
        t = m.group(0)
        if t.startswith("//") or t.startswith("/*"):
            return "".join("\n" if c == "\n" else " " for c in t)
        return t

    return _COMMENT_OR_LITERAL.sub(repl, text)


def squash(text: str) -> str:
    t = re.sub(r"\s+", " ", text)
    t = re.sub(r"\( ", "(", t)
    return re.sub(r" \)", ")", t)


def body(code: str, marker: str, pin: str) -> str:
    """The squashed `{ ... }` body of the one definition that starts with `marker` (strings may hold braces)."""
    sq = squash(code)
    m = squash(marker)
    if sq.count(m) != 1:
        raise ContractError(f"FAIL [pin {pin}]: `{marker}` appears {sq.count(m)} times, not once")
    at = sq.find("{", sq.find(m) + len(m))
    depth = 0
    k = at
    while k < len(sq):
        c = sq[k]
        if c == '"':
            k += 1
            while k < len(sq) and sq[k] != '"':
                k += 2 if sq[k] == "\\" else 1
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return sq[at : k + 1]
        k += 1
    raise ContractError(f"FAIL [pin {pin}]: the body after `{marker}` is unbalanced")


def calls(code: str, name: str) -> list[str]:
    """Each call of `name` in squashed code, as its argument text up to the matching parenthesis."""
    sq = squash(code)
    out = []
    for m in re.finditer(r"(?<![\w.>:])(?:ggml_sycl::)?" + re.escape(name) + r"\(", sq):
        before = sq[max(0, m.start() - 8) : m.start()]
        if re.search(r"\b(?:int|void|bool)\s*$", before):
            continue  # a declaration or definition, not a call
        depth = 0
        for k in range(m.end() - 1, len(sq)):
            depth += sq[k] == "("
            depth -= sq[k] == ")"
            if depth == 0:
                out.append(sq[m.end() : k])
                break
    return out


def check(raws: dict[str, str]) -> None:
    code = {name: blank_comments(text) for name, text in raws.items()}

    # L1: nothing of the removed mechanisms survives, in code or in comments.
    for name, text in raws.items():
        for word in REMOVED_WORDS:
            m = re.search(r"\b" + re.escape(word) + r"\b", text)
            if m:
                line = text.count("\n", 0, m.start()) + 1
                raise ContractError(f"FAIL [pin L1]: removed name `{word}` is still in {name}:{line}")
        for tag in REMOVED_TAGS:
            at = text.find(tag)
            if at >= 0:
                line = text.count("\n", 0, at) + 1
                raise ContractError(f"FAIL [pin L1]: removed log tag `{tag}` is still in {name}:{line}")

    # L2: the rank API the planner reads is declared, defined and read.
    hpp = squash(code[CACHE_HPP])
    be = squash(code[BACKEND])
    for sig in RANK_API:
        decl = squash(sig) + ";"
        if hpp.count(decl) != 1:
            raise ContractError(f"FAIL [pin L2]: `{sig};` is declared {hpp.count(decl)} times in {CACHE_HPP}, not once")
        defn = squash(sig) + " {"
        if be.count(defn) != 1:
            raise ContractError(f"FAIL [pin L2]: `{sig}` is defined {be.count(defn)} times in {BACKEND}, not once")
    triplets = body(code[CACHE_CPP], TRIPLET_DEF, "L2")
    for needle in ("is_expert_popularity_initialized()", "get_expert_popularity_rank(group.layer_id, group.expert_id)"):
        if needle not in triplets:
            raise ContractError(f"FAIL [pin L2]: build_moe_triplet_groups no longer calls `{needle}`")

    # L3: one rank writer, the hotset, keyed by its layer_id parameter.
    writes = [c for name in raws for c in calls(code[name], "set_expert_popularity_rank")]
    if len(writes) != 1:
        raise ContractError(f"FAIL [pin L3]: the backend calls set_expert_popularity_rank {len(writes)} times, not once")
    hotset = body(code[BACKEND], HOTSET_DEF, "L3")
    if squash(HOTSET_WRITE) not in hotset:
        raise ContractError("FAIL [pin L3]: the one rank write is not ggml_sycl_update_moe_hotset's block-keyed write")
    n_calls = be.count(squash(HOTSET_CALL))
    if n_calls != 1:
        raise ContractError(f"FAIL [pin L3]: ggml_sycl_update_moe_hotset is called {n_calls} times, not once")

    # L4: no lookup under the hash layer id.
    for name in raws:
        for args in calls(code[name], "get_expert_popularity_rank"):
            if "moe_cache_layer_id" in args or "hash" in args:
                raise ContractError(f"FAIL [pin L4]: {name} looks a rank up under a hash layer id: `({args})`")
        sq = squash(code[name])
        for m in re.finditer(r"\b(\w+)\s*=\s*moe_cache_layer_id\(", sq):
            var = m.group(1)
            for args in calls(code[name], "get_expert_popularity_rank"):
                if re.search(r"\b" + re.escape(var) + r"\b", args):
                    raise ContractError(f"FAIL [pin L4]: {name} looks a rank up under `{var}`, a moe_cache_layer_id")

    # L5: eviction picks its victim by tier and age, with no per-expert frequency.
    evict = body(code[CACHE_CPP], EVICT_ONE_DEF, "L5")
    if re.search(r"\bfreq\b|frequency", evict):
        raise ContractError("FAIL [pin L5]: unified_cache::evict_one still reads a per-expert frequency")

    # L6: the down-I8 hot set has no rank source but the block-keyed lookup, and no hot set without score or rank.
    hot = body(code[BACKEND], DOWN_I8_HOTSET_DEF, "L6")
    for needle in DOWN_I8_RULE:
        if hot.count(needle) != 1:
            raise ContractError(f"FAIL [pin L6]: the down-I8 hot set no longer holds `{needle}` once")
    for needle, n in (("ranked.push_back(", 1), ("item.rank =", 1), ("get_expert_popularity_rank(", 1)):
        if hot.count(needle) != n:
            raise ContractError(f"FAIL [pin L6]: the down-I8 hot set has {hot.count(needle)} `{needle}`, not {n}")
    if hot.find(DOWN_I8_RULE[3]) > hot.find("std::sort(ranked.begin()"):
        raise ContractError("FAIL [pin L6]: the down-I8 hot set sorts before returning on an empty ranking")


def self_test(raws: dict[str, str]) -> int:
    failures: list[str] = []
    n_mutants = 0

    def mutate(file: str, old: str, new: str) -> dict[str, str]:
        out = dict(raws)
        if out[file].count(old) != 1:
            raise MutantNotApplied(f"anchor matches {out[file].count(old)} times in {file}: {old[:70]!r}")
        out[file] = out[file].replace(old, new, 1)
        return out

    def expect_fail(name: str, pin: str, file: str, old: str, new: str) -> None:
        nonlocal n_mutants
        n_mutants += 1
        try:
            mutated = mutate(file, old, new)
        except MutantNotApplied as e:
            failures.append(f"{name}: NOT APPLIED ({e})")
            print(f"  mutant {name}: NOT APPLIED ({e})")
            return
        try:
            check(mutated)
        except ContractError as e:
            if f"[pin {pin}]" not in str(e):
                failures.append(f"{name}: caught by the wrong pin ({e})")
                print(f"  mutant {name}: caught by the wrong pin, wanted {pin}: {e}")
            else:
                print(f"  mutant {name}: caught by {pin}")
            return
        failures.append(f"{name}: SURVIVED")
        print(f"  mutant {name}: SURVIVED (wanted {pin})")

    hotset_anchor = HOTSET_DEF
    ranks_anchor = "bool is_expert_popularity_initialized() {"
    evict_anchor = "tier      = eviction_tier(entry);\n"

    expect_fail("warmup-table-back", "L1", BACKEND, hotset_anchor,
                "static constexpr int MOE_MAX_LAYERS = 128;\n" + hotset_anchor)
    expect_fail("warmup-global-in-comment", "L1", BACKEND, hotset_anchor, "// g_moe_warmup.init()\n" + hotset_anchor)
    expect_fail("warmup-log-tag-back", "L1", BACKEND, hotset_anchor,
                'static void f() { GGML_LOG_INFO("[MOE-WARMUP] x\\n"); }\n' + hotset_anchor)
    expect_fail("rerank-back", "L1", BACKEND, hotset_anchor, "static void moe_periodic_rerank() {}\n" + hotset_anchor)
    expect_fail("legacy-prestage-switch-back", "L1", BACKEND, hotset_anchor,
                'static bool f() { return std::getenv("GGML_SYCL_LEGACY_PRESTAGE"); }\n' + hotset_anchor)
    expect_fail("adaptive-prestage-back", "L1", BACKEND, hotset_anchor,
                "static int g_adaptive_prestage;\n" + hotset_anchor)
    expect_fail("frequency-api-back", "L1", "expert-prefetch.hpp", "void * moe_expert_ensure_soa_cached(",
                "uint32_t get_expert_frequency(int layer_hash, int expert_id);\nvoid * moe_expert_ensure_soa_cached(")
    expect_fail("rank-api-undeclared", "L2", CACHE_HPP, "bool is_expert_popularity_initialized();",
                "bool expert_popularity_ready();")
    expect_fail("rank-api-undefined", "L2", BACKEND, ranks_anchor, "bool expert_popularity_ready() {")
    expect_fail("planner-stops-reading-ranks", "L2", CACHE_CPP,
                "group.popularity_rank = get_expert_popularity_rank(group.layer_id, group.expert_id);",
                "group.popularity_rank = -1;")
    expect_fail("hash-keyed-write-back", "L3", BACKEND, hotset_anchor,
                "static void f(const char * n) { ggml_sycl::set_expert_popularity_rank(moe_cache_layer_id(n), 0, 0); }\n"
                + hotset_anchor)
    expect_fail("hotset-write-rekeyed", "L3", BACKEND, HOTSET_WRITE,
                "ggml_sycl::set_expert_popularity_rank(0, indices[i], static_cast<int>(i));")
    expect_fail("hash-keyed-lookup-back", "L4", BACKEND, hotset_anchor,
                "static int f(const char * n) {\n    const int hash_layer = moe_cache_layer_id(n);\n"
                "    return ggml_sycl::get_expert_popularity_rank(hash_layer, 0);\n}\n" + hotset_anchor)
    expect_fail("eviction-frequency-back", "L5", CACHE_CPP, evict_anchor,
                evict_anchor + "        uint32_t freq = 0;\n")
    push_rule = "        if (have_scores || item.rank != INT_MAX) {\n            ranked.push_back(item);\n        }\n"
    expect_fail("down-i8-hotset-pushes-unranked", "L6", BACKEND, push_rule, "        ranked.push_back(item);\n")
    expect_fail("down-i8-hotset-no-empty-return", "L6", BACKEND,
                "    if (ranked.empty()) {\n        return 0;\n    }\n    std::sort(ranked.begin()",
                "    std::sort(ranked.begin()")
    expect_fail("down-i8-hotset-second-rank-source", "L6", BACKEND, push_rule,
                "        if (item.rank == INT_MAX) {\n            item.rank = static_cast<int>(e);\n        }\n" + push_rule)

    try:
        mutate(BACKEND, "this anchor is in no source", "x")
        failures.append("stale-anchor-not-reported")
        print("  self-test: a stale mutant anchor is NOT reported")
    except MutantNotApplied:
        print("  self-test: a stale mutant anchor is reported")

    try:
        check(raws)
        print("  self-test: unmodified tree passes")
    except ContractError as e:
        print(f"  self-test: unmodified tree FAILS: {e}")
        failures.append("unmodified")

    if failures:
        print(f"SELF-TEST FAIL: {', '.join(failures)}")
        return 1
    print(f"SELF-TEST PASS: {n_mutants} mutants caught by their pins, unmodified tree passes")
    return 0


def main(argv: list[str]) -> int:
    raws = {
        p.name: p.read_text()
        for p in sorted(SYCL_DIR.iterdir())
        if p.is_file() and p.suffix in (".cpp", ".hpp", ".h")
    }
    for need in (BACKEND, CACHE_CPP, CACHE_HPP, "expert-prefetch.hpp"):
        if need not in raws:
            print(f"FAIL: {need} not found under {SYCL_DIR}")
            return 1
    if "--self-test" in argv:
        return self_test(raws)
    try:
        check(raws)
    except ContractError as e:
        print(e)
        return 1
    print("PASS: the legacy expert ranking is gone and the planner's block-keyed rank API stays")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
