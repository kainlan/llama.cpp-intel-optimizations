#!/usr/bin/env python3
"""Source contract: the opt-in spread placement of routed experts (GGML_SYCL_MOE_SPREAD_PLACEMENT).

Host-only: reads the SYCL backend sources, builds nothing, loads no model and touches no device.

By default the planner walks the routed-expert triplets layer-first, so under pressure the early MoE blocks stay whole on
the device and the late ones go to the host. The switch walks them rank-major over every block instead, (rank, block,
expert) with a group's rank its popularity rank when it has one and its expert index otherwise, so every block gets an
equal share of the device slots. This gate holds how the switch is built:

  S1  one comparator orders the triplets: moe_triplet_group_before() is the only comparator over moe_triplet_group, it
      is called only from sort_moe_triplet_groups(), whose std::sort is the only sort of a triplet vector in the planner
      file, and the layer-first order (build_moe_triplet_groups) and the spread order (moe_spread_order_groups) are its
      only two callers; the spread rank is the popularity rank, else the expert index;
  S2  the switch is read in one place: its name is spelled once (MOE_SPREAD_PLACEMENT_ENV in unified-cache.hpp), the
      environment is read once (placement_moe_spread_enabled() caches it in a static), the reader is called only where
      the backend copies the planner inputs in (kv.moe_spread) and where ggml_check_sycl reports it (after its table
      row), and the planner file never calls the reader or reads the environment itself;
  S3  the planner never writes the rank store: unified-cache.cpp calls no set_expert_popularity_rank() and names no
      g_expert_popularity, and the order's functions read only the ranks build_moe_triplet_groups put on the groups;
  S4  the spread order is in place before anything predicts the pack: in the single-device builder after the PP
      alternate charges and before the window ring's room, and in the multi-device builder's per-group branch right
      after its groups are built and before the pack;
  S5  the swap headroom is a named term of the host WEIGHT zone (host_weight_zone_terms::swap_headroom, from the plan's
      moe_swap_host_headroom_bytes) and each builder holds it after its host side is sized.

Checks run on comment-blanked text with whitespace squashed. NOT VACUOUS: --self-test runs the gate on in-memory mutants
of the real sources. Each mutant names the pin that must catch it, and is reported if it survives, if a different pin
catches it, or if its anchor no longer matches exactly once (NOT APPLIED). ctest registers it with --self-test.

WHAT THIS DOES NOT PROVE. It reads text. The order each mode gives and the headroom's figures are
test-sycl-layout-choice's cases; that the tokens are unchanged is the lead's run on a device.
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

ENV_DECL = 'static constexpr const char * MOE_SPREAD_PLACEMENT_ENV = "GGML_SYCL_MOE_SPREAD_PLACEMENT";'
ENV_LITERAL = '"GGML_SYCL_MOE_SPREAD_PLACEMENT"'
ENV_CONST = "MOE_SPREAD_PLACEMENT_ENV"
READER = "placement_moe_spread_enabled"
READER_DEF = "bool placement_moe_spread_enabled() {"
INTAKE_DEF = "static void ggml_sycl_inventory_fill_kv_info("
INTAKE_READ = "kv.moe_spread = ggml_sycl::placement_moe_spread_enabled();"
REPORT_ROW = "{ ggml_sycl::MOE_SPREAD_PLACEMENT_ENV, &g_ggml_sycl_moe_spread_placement, 0 },"
REPORT_LOOP = "for (const sycl_env_setting & setting : sycl_env_settings) {"
REPORT_READ = "g_ggml_sycl_moe_spread_placement = ggml_sycl::placement_moe_spread_enabled() ? 1 : 0;"

COMPARATOR = "moe_triplet_group_before"
COMPARATOR_DEF = "static bool moe_triplet_group_before("
SORT = "sort_moe_triplet_groups"
SORT_DEF = "static void sort_moe_triplet_groups("
SPREAD_RANK_DEF = "static int moe_triplet_spread_rank("
SPREAD_RANK_BODY = "{ return group.popularity_rank >= 0 ? group.popularity_rank : group.expert_id; }"
BUILD_DEF = "static std::vector<moe_triplet_group> build_moe_triplet_groups("
BUILD_SORT = "sort_moe_triplet_groups(groups, moe_triplet_order::LAYER_FIRST);"
SPREAD_DEF = "static void moe_spread_order_groups("
SPREAD_SORT = "sort_moe_triplet_groups(groups, moe_triplet_order::SPREAD);"
SPREAD_SWITCH = "if (!kv_info.moe_spread) {"
ORDER_FUNCTIONS = (COMPARATOR_DEF, SORT_DEF, SPREAD_RANK_DEF, SPREAD_DEF)

SINGLE_DEF = "placement_plan compute_placement_plan(const std::vector<placement_tensor_info> & tensor_inventory,"
MULTI_DEF = (
    "placement_plan compute_multi_device_plan(const std::vector<device_budget> & device_budgets, "
    "const std::vector<placement_tensor_info> & tensor_inventory,"
)
PP_ALT_LINE = 'GGML_LOG_INFO("[PLACEMENT-MOE] PP executable alternate reserve: '
SINGLE_SPREAD = "moe_spread_order_groups(plan, moe_groups, kv_info, remaining, &hotness_source);"
RING_ROOM_CALL = "hold_moe_stream_ring_room(plan, moe_groups, kv_info, device_id, remaining, host_experts_pinned);"
MULTI_BUILD = "auto moe_groups = build_moe_triplet_groups(plan, moe_indices, &hotness_source);"
MULTI_SPREAD = (
    "moe_spread_order_groups(plan, moe_groups, kv_info, std::accumulate(remaining.begin(), remaining.end(), size_t(0)), "
    "&hotness_source);"
)
MULTI_PACK = "for (const auto & group : moe_groups) { if (moe_triplet_complete(group)) {"

TERMS_TOTAL = (
    "size_t total() const { return host_weights + host_compute_reserve + layout_copies + tlsf_bound + swap_headroom; }"
)
TERMS_DEF = "host_weight_zone_terms placement_plan_host_weight_zone_terms(const placement_plan & plan) {"
TERMS_SET = "terms.swap_headroom = plan.moe_swap_host_headroom_bytes;"
HOST_SIDE = "placement_plan_size_host_side("
SINGLE_HOLD = 'placement_plan_hold_moe_swap_host_headroom(plan, kv_info, "PLACEMENT-MOE");'
MULTI_HOLD = 'placement_plan_hold_moe_swap_host_headroom(plan, kv_info, "PLACEMENT-MOE-MULTI");'


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


def body(sq: str, marker: str, pin: str) -> str:
    """The `{ ... }` body of the one definition in squashed `sq` that starts with `marker` (strings may hold braces)."""
    m = squash(marker)
    if sq.count(m) != 1:
        raise ContractError(f"FAIL [pin {pin}]: `{marker}` appears {sq.count(m)} times, not once")
    at = sq.find("{", sq.find(m) + len(m) - 1)
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


def calls(sq: str, name: str) -> list[str]:
    """Each call of `name` in squashed code, as its argument text up to the matching parenthesis."""
    out = []
    for m in re.finditer(r"(?<![\w.>])(?:ggml_sycl::|std::)?" + re.escape(name) + r"\(", sq):
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


def ordered(sq: str, *needles: str) -> bool:
    at = -1
    for needle in needles:
        at = sq.find(squash(needle), at + 1)
        if at < 0:
            return False
    return True


def check(raws: dict[str, str]) -> None:
    code = {name: squash(blank_comments(text)) for name, text in raws.items()}
    uc = code[CACHE_CPP]
    be = code[BACKEND]

    # S1: one comparator, one sort, two orders.
    comparator = body(uc, COMPARATOR_DEF, "S1")
    sort_fn = body(uc, SORT_DEF, "S1")
    if len(calls(uc, COMPARATOR)) != 1 or len(calls(sort_fn, COMPARATOR)) != 1:
        raise ContractError(f"FAIL [pin S1]: {COMPARATOR} is called {len(calls(uc, COMPARATOR))} times in {CACHE_CPP}, "
                            f"not once, from {SORT}")
    if len(calls(sort_fn, "sort")) != 1:
        raise ContractError(f"FAIL [pin S1]: {SORT} holds {len(calls(sort_fn, 'sort'))} std::sort calls, not one")
    triplet_sorts = [a for a in calls(uc, "sort")
                     if "moe_triplet_group" in a or re.match(r"(?:moe_)?groups\.begin\(\)", a)]
    if len(triplet_sorts) != 1:
        raise ContractError(f"FAIL [pin S1]: {CACHE_CPP} sorts a triplet vector {len(triplet_sorts)} times, not once "
                            f"(in {SORT})")
    if uc.count("const moe_triplet_group & a, const moe_triplet_group & b") != 2:
        raise ContractError("FAIL [pin S1]: a second comparator over moe_triplet_group exists beside "
                            f"{COMPARATOR} and the one lambda that forwards to it")
    if [squash(a) for a in calls(uc, SORT)] != ["groups, moe_triplet_order::LAYER_FIRST",
                                                "groups, moe_triplet_order::SPREAD"]:
        raise ContractError(f"FAIL [pin S1]: {SORT} is not called exactly twice, layer-first then spread: "
                            f"{calls(uc, SORT)}")
    build = body(uc, BUILD_DEF, "S1")
    if build.count(squash(BUILD_SORT)) != 1 or calls(build, "sort"):
        raise ContractError("FAIL [pin S1]: build_moe_triplet_groups does not order its groups through "
                            f"`{BUILD_SORT}` alone")
    spread = body(uc, SPREAD_DEF, "S1")
    if spread.count(squash(SPREAD_SORT)) != 1 or calls(spread, "sort"):
        raise ContractError(f"FAIL [pin S1]: moe_spread_order_groups does not order its groups through `{SPREAD_SORT}`")
    if body(uc, SPREAD_RANK_DEF, "S1") != SPREAD_RANK_BODY:
        raise ContractError("FAIL [pin S1]: the spread rank is not the popularity rank, else the expert index")
    if len(calls(comparator, "moe_triplet_spread_rank")) != 2:
        raise ContractError(f"FAIL [pin S1]: {COMPARATOR} does not rank both groups by moe_triplet_spread_rank")

    # S2: the switch is read in one place.
    hpp = code[CACHE_HPP]
    if hpp.count(squash(ENV_DECL)) != 1:
        raise ContractError(f"FAIL [pin S2]: {CACHE_HPP} does not declare `{ENV_DECL}` once")
    spelled = sum(text.count(ENV_LITERAL) for text in code.values())
    if spelled != 1:
        raise ContractError(f"FAIL [pin S2]: the switch's name is spelled {spelled} times in the backend, not once")
    env_reads = [(name, a) for name, text in code.items() for a in calls(text, "getenv") if ENV_CONST in a]
    reader = body(uc, READER_DEF, "S2")
    if len(env_reads) != 1 or len([a for a in calls(reader, "getenv") if ENV_CONST in a]) != 1:
        raise ContractError(f"FAIL [pin S2]: the environment is read {len(env_reads)} times for the switch, not once "
                            f"in {READER}")
    if "static const bool" not in reader:
        raise ContractError(f"FAIL [pin S2]: {READER} reads the environment on every call instead of once")
    reads = [(name, a) for name, text in code.items() for a in calls(text, READER)]
    if len(reads) != 2 or any(name != BACKEND for name, _ in reads):
        raise ContractError(f"FAIL [pin S2]: {READER} is called {len(reads)} times ({[n for n, _ in reads]}), not "
                            f"twice in {BACKEND}: the intake and the report")
    if body(be, INTAKE_DEF, "S2").count(squash(INTAKE_READ)) != 1:
        raise ContractError(f"FAIL [pin S2]: the planner-input intake does not copy the switch in: `{INTAKE_READ}`")
    if not ordered(be, REPORT_ROW, REPORT_LOOP, REPORT_READ):
        raise ContractError("FAIL [pin S2]: the report has no table row for the switch, or does not set the row's "
                            "value from the reader after its loop")
    if SPREAD_SWITCH not in spread:
        raise ContractError("FAIL [pin S2]: the spread order is not decided from the planner input kv_info.moe_spread")

    # S3: the planner never writes the rank store.
    if calls(uc, "set_expert_popularity_rank") or re.search(r"\bg_expert_popularity\w*", uc):
        raise ContractError(f"FAIL [pin S3]: {CACHE_CPP} writes or names the popularity rank store")
    for marker in ORDER_FUNCTIONS:
        fn = body(uc, marker, "S3")
        if re.search(r"\b(?:get|set)_expert_popularity_rank\(|\bis_expert_popularity_initialized\(", fn):
            raise ContractError(f"FAIL [pin S3]: `{marker}` reaches the rank store instead of the groups' ranks")

    # S4: the spread order is in place before anything predicts the pack.
    single = body(uc, SINGLE_DEF, "S4")
    if single.count(squash(SINGLE_SPREAD)) != 1 or not ordered(single, PP_ALT_LINE, SINGLE_SPREAD, RING_ROOM_CALL):
        raise ContractError("FAIL [pin S4]: the single-device builder does not order its groups once, after the PP "
                            "alternate charges and before the window ring's room")
    multi = body(uc, MULTI_DEF, "S4")
    at = multi.find(squash(MULTI_SPREAD))
    before = multi[:at].rstrip()
    if (multi.count(squash(MULTI_SPREAD)) != 1 or not before.endswith(MULTI_BUILD)
            or before.endswith("const " + MULTI_BUILD) or not ordered(multi, MULTI_SPREAD, MULTI_PACK)):
        raise ContractError("FAIL [pin S4]: the multi-device builder does not order its per-group branch's groups once, "
                            "right after building them and before the pack")

    # S5: the swap headroom is a named WEIGHT zone term, held after the host side is sized.
    if hpp.count(squash(TERMS_TOTAL)) != 1 or body(uc, TERMS_DEF, "S5").count(squash(TERMS_SET)) != 1:
        raise ContractError("FAIL [pin S5]: the swap headroom is not a named term of the host WEIGHT zone")
    for fn, hold in ((single, SINGLE_HOLD), (multi, MULTI_HOLD)):
        if fn.count(squash(hold)) != 1 or not ordered(fn, HOST_SIDE, hold):
            raise ContractError(f"FAIL [pin S5]: `{hold}` is not called once after the builder sizes its host side")


def self_test(raws: dict[str, str]) -> int:
    failures: list[str] = []
    n_mutants = 0

    def mutate(file: str, old: str, new: str, base: dict[str, str] | None = None) -> dict[str, str]:
        out = dict(raws if base is None else base)
        if out[file].count(old) != 1:
            raise MutantNotApplied(f"anchor matches {out[file].count(old)} times in {file}: {old[:70]!r}")
        out[file] = out[file].replace(old, new, 1)
        return out

    def expect_fail(name: str, pin: str, file: str, old: str, new: str, *more: tuple[str, str]) -> None:
        """`more` holds further (old, new) edits of the same file, applied in order after the first."""
        nonlocal n_mutants
        n_mutants += 1
        try:
            mutated = mutate(file, old, new)
            for more_old, more_new in more:
                mutated = mutate(file, more_old, more_new, mutated)
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

    build_sort = "    sort_moe_triplet_groups(groups, moe_triplet_order::LAYER_FIRST);\n"
    spread_sort = "    sort_moe_triplet_groups(groups, moe_triplet_order::SPREAD);\n"
    own_sort = ("    std::sort(groups.begin(), groups.end(), [](const moe_triplet_group & a, const moe_triplet_group & b) "
                "{ return a.expert_id < b.expert_id; });\n")
    spread_rank = "    return group.popularity_rank >= 0 ? group.popularity_rank : group.expert_id;\n"
    reader_read = "        const char * env = std::getenv(MOE_SPREAD_PLACEMENT_ENV);\n"
    intake = "    kv.moe_spread              = ggml_sycl::placement_moe_spread_enabled();\n"
    report_row = "            { ggml_sycl::MOE_SPREAD_PLACEMENT_ENV, &g_ggml_sycl_moe_spread_placement, 0 },\n"
    report_read = "        g_ggml_sycl_moe_spread_placement         = ggml_sycl::placement_moe_spread_enabled() ? 1 : 0;\n"
    switch = "    if (!kv_info.moe_spread) {\n"
    single_spread = "        moe_spread_order_groups(plan, moe_groups, kv_info, remaining, &hotness_source);\n"
    ring = "        hold_moe_stream_ring_room(plan, moe_groups, kv_info, device_id, remaining, host_experts_pinned);\n"
    total = ("    size_t total() const { return host_weights + host_compute_reserve + layout_copies + tlsf_bound + "
             "swap_headroom; }")

    expect_fail("layer-first-sorts-with-its-own-lambda", "S1", CACHE_CPP, build_sort, own_sort)
    expect_fail("spread-sorts-with-its-own-lambda", "S1", CACHE_CPP, spread_sort, own_sort)
    expect_fail("spread-order-dropped", "S1", CACHE_CPP, spread_sort, "")
    expect_fail("spread-rank-unranked-last", "S1", CACHE_CPP, spread_rank,
                "    return group.popularity_rank >= 0 ? group.popularity_rank : std::numeric_limits<int>::max();\n")
    expect_fail("comparator-skips-spread-rank", "S1", CACHE_CPP, "        const int b_rank = moe_triplet_spread_rank(b);\n",
                "        const int b_rank = b.expert_id;\n")
    expect_fail("planner-reads-the-environment", "S2", CACHE_CPP, switch,
                "    if (std::getenv(MOE_SPREAD_PLACEMENT_ENV) == nullptr) {\n")
    expect_fail("planner-calls-the-reader", "S2", CACHE_CPP, switch, "    if (!placement_moe_spread_enabled()) {\n")
    expect_fail("reader-not-cached", "S2", CACHE_CPP, "    static const bool enabled = [] {\n" + reader_read,
                "    const bool enabled = [] {\n" + reader_read)
    expect_fail("name-spelled-twice", "S2", BACKEND, intake,
                intake + '    (void) std::getenv("GGML_SYCL_MOE_SPREAD_PLACEMENT");\n')
    expect_fail("intake-drops-the-switch", "S2", BACKEND, intake, "")
    expect_fail("report-row-dropped", "S2", BACKEND, report_row, "")
    expect_fail("report-reads-the-table-parse", "S2", BACKEND, report_read, "")
    expect_fail("reader-reads-another-name", "S2", CACHE_CPP, reader_read,
                '        const char * env = std::getenv("GGML_SYCL_MOE_SPREAD");\n')
    expect_fail("planner-writes-a-rank", "S3", CACHE_CPP, spread_sort,
                spread_sort + "    set_expert_popularity_rank(groups[0].layer_id, groups[0].expert_id, 0);\n")
    expect_fail("sort-looks-ranks-up", "S3", CACHE_CPP, "    std::sort(groups.begin(), groups.end(), [order, ranked]",
                "    (void) get_expert_popularity_rank(0, 0);\n    std::sort(groups.begin(), groups.end(), [order, ranked]")
    expect_fail("spread-after-the-ring-room", "S4", CACHE_CPP, single_spread, "", (ring, ring + single_spread))
    expect_fail("single-spread-dropped", "S4", CACHE_CPP, single_spread, "")
    expect_fail("multi-spread-dropped", "S4", CACHE_CPP,
                "            moe_spread_order_groups(plan, moe_groups, kv_info,\n", "            (void) (plan, kv_info,\n")
    expect_fail("headroom-not-in-the-zone-total", "S5", CACHE_HPP, total,
                "    size_t total() const { return host_weights + host_compute_reserve + layout_copies + tlsf_bound; }")
    expect_fail("headroom-term-unset", "S5", CACHE_CPP,
                "    terms.swap_headroom        = plan.moe_swap_host_headroom_bytes;\n", "")
    expect_fail("multi-holds-no-headroom", "S5", CACHE_CPP,
                '    placement_plan_hold_moe_swap_host_headroom(plan, kv_info, "PLACEMENT-MOE-MULTI");\n', "")

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
        p.name: p.read_text(errors="replace")
        for p in sorted(SYCL_DIR.iterdir())
        if p.is_file() and p.suffix in (".cpp", ".hpp", ".h")
    }
    for need in (BACKEND, CACHE_CPP, CACHE_HPP):
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
    print("PASS: one comparator orders the expert triplets, the spread switch is read once, the planner writes no rank")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
