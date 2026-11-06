#!/usr/bin/env python3
"""Source contract for the device RED test-sycl-scratch-trial-return (docs/design/sycl-mmap-host-tier.md §7 C4;
§4.4, §M75; r8 m-1, m-2; r9 m-2, m-3, n-6; r10 m-3, m-4, n-7; r11 m-4, m-5; r12 M-4).

The device test is lead-run. This pin holds what a host can read:
1. the registration: one executable linked against the private fixtures and never libllama (§M82 case (2)), labelled
   sycl;cache;mem-handle;gpu-serial, serial, skip code 77, the selector pinned to level_zero:1 and 64 MiB chunks, all
   inside the `BUILD_TESTING AND NOT GGML_BACKEND_DL` block; every planted fault the design names for the test is a
   registered arm with its cases and its expected text, and no other arm is registered;
2. the mutant census over the three device REDs: every scratch_test_mutant enumerator has a name in the selector's
   table and is scored by at least one registered arm, every name an arm uses is in the table, and every enumerator
   has a block in production code that the production build compiles out;
3. the test source: the trial is resolved by proc name, the fixture's notifies carry a plan term of 0, main lists the
   design's cases and each has its dispatch, every arm's cases are among them, every arm's expected text is printed by
   the test, and no case calls the owner's end;
4. production (§M75, r9 n-6, r10 n-7): the cache's host_scratch_trial handles the five steps, RETURN drops the
   context's demand, KEEP touches no marked set, both SCRATCH grow sites charge a grow to the transaction, and the
   witness line has the design's format.

Each clause has a positive control: a planted edit of the tree text that the clause must reject with its message. The
tree is read with plain file I/O (the codescout index is blind in ggml-sycl.cpp), comment-stripped, and with every
planted-fault block resolved as the production build compiles it wherever a clause scores production code.
"""

import re
import sys
from pathlib import Path

import pytest

# sycl_gate sits beside this file. `python3 -I` leaves the script's directory off sys.path, so it is appended here;
# appended rather than prepended, a file in tests/ never shadows a standard module.
sys.path.append(str(Path(__file__).resolve().parent))
from sycl_gate import (  # noqa: E402
    arms_of, body, cmake_strip, function_problems, if_stack_at, missing, norm, registration, rejects, resolve_mutants,
    strip_comments, tree_tests, tree_texts
)

SYCL = "ggml/src/ggml-sycl"
CMAKE = SYCL + "/CMakeLists.txt"
TEST = SYCL + "/tests/test-sycl-scratch-trial-return.cpp"
COMMON = SYCL + "/tests/scratch-return-test-common.hpp"
MUTANT_H = SYCL + "/scratch-test-mutant.hpp"
UC = SYCL + "/unified-cache.cpp"
# The production files the planted faults live in.
PRODUCTION = (SYCL + "/pinned-pool.cpp", SYCL + "/pinned-pool.hpp", UC, SYCL + "/ggml-sycl.cpp")
FILES = (CMAKE, TEST, COMMON, MUTANT_H) + PRODUCTION

STEM = "test-sycl-scratch-trial-return"
STEMS = (STEM, "test-sycl-configure-refusal", "test-sycl-scratch-owner-end")
FUNC = "ggml_sycl_scratch_return_test"
BLOCK = "if (BUILD_TESTING AND NOT GGML_BACKEND_DL)"

CASES = ("ladder", "second-context", "live-refusal", "window", "empty-return", "throw-kept", "retry-return",
         "retry-keep", "trial-retry-return", "trial-retry-keep", "dropped-carrier", "strand", "teardown",
         "registered-exact", "nine-strand", "status-order")

# Each planted fault the design names for this test: its cases, and the phrases of the design's failure text that the
# arm's expected text carries and the test prints (each phrase is one run of literal text in the test's source).
ARMS = {
    "return-noop": ("ladder,second-context", ("(winner's need)", "the capacity quotes the loser")),
    "return-releases-live": ("ladder", ("RETURN printed a SCRATCH return refused line for live chunk",)),
    "return-no-withdraw": ("window,ladder", ("after RETURN", "the capacity quotes the loser")),
    "return-no-file": ("window", ("after the ladder, expected", "no SCRATCH released line for chunk")),
    "double-decrement": ("ladder", ("exceeds the pool's allocated()",)),
    "return-no-immediate": ("empty-return", ("right after RETURN", " not released")),
    "footprint-before-free": ("empty-return", ("footprint decremented before the free",)),
    "release-no-erase": ("empty-return,window", ("entry(ies) after the release of chunk",)),
    "begin-names-any-owner": ("throw-kept", ("the second context's begin printed a SCRATCH return refused line",)),
    "rtm-left-in-txn": ("retry-return,retry-keep,trial-retry-return,trial-retry-keep",
                        ("after RETURN_TO_MARK, expected 0",)),
    "rtm-left-in-txn,withdraw-transition-blind": ("retry-return", ("exceeds the pool's allocated()",)),
    # The design's `chunk_count <n+2> after the begin` is unreachable: the un-withdrawn chunk is still marked, so the
    # begin's release_zone_chunk aborts on its withdrawn assert. The arm scores the check the fault fails first.
    "keep-walks-set": ("retry-keep,trial-retry-keep",
                       ("not withdrawn after KEEP", "KEEP reaches only the transaction")),
    "carrier-dtor-noop": ("dropped-carrier", ("footprint leaked by a dropped carrier",)),
    "moved-from-releases": ("dropped-carrier", ("footprint subtracted twice",)),
    "no-prune": ("strand", ("stranded footprint never reconciled",)),
    "prune-without-query": ("strand", ("at the first reconcile while the allocation is still registered",)),
    "subtract-at-refusal": ("strand", ("after a refused free", "footprint left while the allocation is still")),
    "dtor-no-report": ("teardown", ("no teardown line for the stranded entry",)),
    "registered-exact-ptr-only": ("registered-exact", ("registered_exact answered true for ptr", "the row holds id")),
    "stranded-reserve-8": ("nine-strand", ("stranded_ capacity", "exhausted at")),
    "lease-checked-first": ("status-order", ("which holds a live allocation",)),
}

# The arms whose expected text production prints, not the test: the pool's own abort.
PRINTED_BY = {"stranded-reserve-8": SYCL + "/pinned-pool.cpp"}


# ---- CMake ----------------------------------------------------------------------------------------------------------


def registration_problems(texts: dict) -> list:
    fails = missing(texts, CMAKE)
    if fails:
        return fails
    cmake = cmake_strip(texts[CMAKE])
    fails += function_problems(cmake, FUNC, BLOCK, CMAKE)
    reg = registration(cmake, FUNC, STEM)
    if reg is None:
        return fails + [f"{STEM} is not registered through {FUNC}"]
    at, parsed = reg
    if BLOCK not in if_stack_at(cmake, at):
        fails.append(f"{STEM} is registered outside `{BLOCK}`")
    arms = arms_of(parsed)
    if arms is None:
        return fails + [f"{STEM}'s MUTANTS list is not whole (names, cases, expected text) triples"]
    for name, (cases, phrases) in ARMS.items():
        if name not in arms:
            fails.append(f"arm {name} is not registered")
            continue
        if arms[name][0] != cases:
            fails.append(f"arm {name} runs cases {arms[name][0]!r}, expected {cases!r}")
        for phrase in phrases:
            if phrase not in arms[name][1].replace("\\\\", "\\").replace("\\", ""):
                fails.append(f"arm {name}'s expected text lacks the design's {phrase!r}")
        try:
            re.compile(arms[name][1].replace("\\\\", "\\"))
        except re.error as error:
            fails.append(f"arm {name}'s expected text is not a regular expression: {error}")
    for name in arms:
        if name not in ARMS:
            fails.append(f"arm {name} is registered but names no planted fault the design lists for {STEM}")
    return fails


# ---- the mutant census ----------------------------------------------------------------------------------------------

def enumerators(header: str) -> list:
    m = re.search(r"enum class scratch_test_mutant\s*:\s*int\s*\{(.*?)\};", strip_comments(header), re.S)
    if not m:
        return []
    return [e for e in re.findall(r"\b([A-Z][A-Z_0-9]*)\s*=\s*\d+\s*,", m.group(1)) if e != "COUNT"]


def name_table(common: str) -> dict:
    return dict(re.findall(r'\{\s*"([a-z0-9-]+)"\s*,\s*scratch_test_mutant::([A-Z_0-9]+)\s*\}', common))


def census_problems(texts: dict) -> list:
    fails = missing(texts, CMAKE, COMMON, MUTANT_H, *PRODUCTION)
    if fails:
        return fails
    enums = enumerators(texts[MUTANT_H])
    if not enums:
        return [f"{MUTANT_H}: no scratch_test_mutant enumerators"]
    table = name_table(strip_comments(texts[COMMON]))
    by_enum = {e: n for n, e in table.items()}
    cmake = cmake_strip(texts[CMAKE])
    scored = set()
    for stem in STEMS:
        reg = registration(cmake, FUNC, stem)
        arms = arms_of(reg[1]) if reg else None
        if arms is None:
            fails.append(f"{stem} has no whole registration to census")
            continue
        for names in arms:
            for name in names.split(","):
                if name not in table:
                    fails.append(f"{stem}'s arm {names} names {name!r}, which the selector's table lacks")
                scored.add(name)
    for e in enums:
        if e not in by_enum:
            fails.append(f"scratch_test_mutant::{e} has no name in the selector's table")
            continue
        if by_enum[e] not in scored:
            fails.append(f"scratch_test_mutant::{e} ({by_enum[e]}) is scored by no registered arm")
        call = f"scratch_test_mutant_on(scratch_test_mutant::{e})"
        sites = [rel for rel in PRODUCTION if call in strip_comments(texts[rel])]
        if not sites:
            fails.append(f"scratch_test_mutant::{e} has no planted block in production code")
        for rel in sites:
            if call in strip_comments(resolve_mutants(texts[rel])):
                fails.append(f"{rel}: scratch_test_mutant::{e} is tested outside a GGML_SYCL_SCRATCH_MUTANTS block, "
                             "so the production build compiles it")
    return fails


# ---- the test source ------------------------------------------------------------------------------------------------

def source_problems(texts: dict) -> list:
    fails = missing(texts, TEST, COMMON, *PRINTED_BY.values())
    if fails:
        return fails
    raw = texts[TEST]
    t = norm(raw)
    # What a case prints: its own text and the shared helpers'.
    printed = t + "\n" + norm(texts[COMMON])
    if '#include "scratch-return-test-common.hpp"' not in raw:
        fails.append(f"{TEST} does not include scratch-return-test-common.hpp")
    if 'ggml_backend_reg_get_proc_address(ggml_backend_sycl_reg(), "ggml_backend_sycl_host_scratch_trial")' not in t:
        fails.append("the test does not resolve ggml_backend_sycl_host_scratch_trial by proc name")
    if "r.ops.plan_scratch_bytes = 0;" not in t or "notify_context(" not in t:
        fails.append("the fixture's notifies do not carry the design's plan term of 0 through notify_context")
    if "host_scratch_owner_end" in t:
        fails.append("the test calls the owner's end")
    m = re.search(r"cases_main\(argc, argv, \{([^}]*)\}, run_case", t)
    listed = tuple(re.findall(r'"([a-z-]+)"', m.group(1))) if m else ()
    if listed != CASES:
        fails.append(f"main lists the cases {list(listed)}, expected {list(CASES)}")
    run = body(t, "int run_case(const child_args & a)")
    for case in CASES:
        if f'n == "{case}"' not in run:
            fails.append(f"run_case has no dispatch for case {case}")
    for name, (cases, phrases) in ARMS.items():
        for case in cases.split(","):
            if case not in CASES:
                fails.append(f"arm {name} runs case {case}, which the test does not list")
        source = norm(texts[PRINTED_BY[name]]) if name in PRINTED_BY else printed
        for phrase in phrases:
            if phrase not in source:
                fails.append(f"the test prints no {phrase!r}, which arm {name} expects")
    return fails


# ---- production -----------------------------------------------------------------------------------------------------

TRIAL_SIG = "size_t unified_cache::host_scratch_trial(host_scratch_trial_op op, uint64_t owner, uint32_t n_ubatch)"
STEPS = ("OPEN", "KEEP", "RETURN", "MARK", "RETURN_TO_MARK")
WITNESS_RE = re.compile(r'"\[HOST-ARENA\] SCRATCH %s dev=%d n_ubatch=%u (?:%s|kept|returned) %\.1f MB '
                        r'\(%zu chunk\(s\)\)\\n"')


def step_branches(trial: str) -> dict:
    """The text of each `case host_scratch_trial_op::X:` branch of the switch in `trial`."""
    labels = list(re.finditer(r"\bcase host_scratch_trial_op::([A-Z_]+):", trial))
    out = {}
    for i, lab in enumerate(labels):
        end = labels[i + 1].start() if i + 1 < len(labels) else len(trial)
        out[lab.group(1)] = trial[lab.end():end]
    return out


def production_problems(texts: dict) -> list:
    fails = missing(texts, UC)
    if fails:
        return fails
    uc = norm(resolve_mutants(texts[UC]))
    trial = body(uc, TRIAL_SIG)
    if not trial:
        return [f"{UC}: missing {TRIAL_SIG!r}"]
    branches = step_branches(trial)
    for step in STEPS:
        if step not in branches:
            fails.append(f"host_scratch_trial has no case for {step} (§M75, r9 n-6)")
    if "RETURN" in branches and "drop_context(" not in branches["RETURN"]:
        fails.append("host_scratch_trial's RETURN does not drop the context's demand")
    if "KEEP" in branches:
        touched = re.findall(r"\b(marked_set_|host_scratch_return_chunks|release_marked_chunk\w*|"
                             r"withdraw_zone_chunk)\b", branches["KEEP"])
        if touched:
            fails.append(f"host_scratch_trial's KEEP touches the marked set ({touched[0]}): KEEP reaches only the "
                         "transaction's own chunks (r12 M-4)")
    for sig, what in (("bool unified_cache::grow_scratch_zone(size_t request_bytes)", "the ledger's ensure"),
                      ("bool unified_cache::host_zone_grow(host_zone_id zone, size_t additional_bytes)",
                       "the miss path")):
        grow = body(uc, sig)
        if not grow:
            fails.append(f"{UC}: missing {sig!r}")
        elif "host_scratch_trial_note_grow(" not in grow:
            fails.append(f"{sig.split('(')[0]} ({what}) does not charge its grow to the open transaction")
    if not WITNESS_RE.search(uc):
        fails.append("no `[HOST-ARENA] SCRATCH <txn> dev=%d n_ubatch=%u kept|returned %.1f MB (%zu chunk(s))` "
                     "witness (r10 n-7)")
    return fails


CLAUSES = {
    "registration": registration_problems,
    "census": census_problems,
    "test-source": source_problems,
    "production": production_problems,
}


# ---- the tree tests -------------------------------------------------------------------------------------------------

(test_sources_exist, test_clause_holds_on_the_tree,
 test_missing_sources_fail_and_say_so) = tree_tests(FILES, CLAUSES, __file__)


# ---- positive controls ----------------------------------------------------------------------------------------------

def planted(rel: str, old: str, new: str) -> dict:
    texts = tree_texts(FILES)
    assert rel in texts, f"control precondition: {rel} not found"
    assert texts[rel].count(old) >= 1, f"control precondition: {old!r} is not in {rel}"
    texts[rel] = texts[rel].replace(old, new, 1)
    return texts


ARM_NOOP = "        return-noop ladder,second-context\n"


def test_registration_rejects_a_dropped_arm():
    texts = tree_texts(FILES)
    cmake = texts.get(CMAKE, "")
    at = cmake.find(ARM_NOOP)
    assert at >= 0, "control precondition: the return-noop arm is not where the control expects it"
    end = cmake.index("\n", cmake.index('"', at + len(ARM_NOOP)) + 1)
    texts[CMAKE] = cmake[:at] + cmake[end + 1:]
    rejects(CLAUSES, "registration", texts, "arm return-noop is not registered")


def test_registration_rejects_moved_cases_and_a_lost_phrase():
    rejects(CLAUSES, "registration", planted(CMAKE, ARM_NOOP, "        return-noop ladder\n"),
            "arm return-noop runs cases 'ladder'")
    rejects(CLAUSES, "registration", planted(CMAKE, "footprint decremented before the free", "footprint moved"),
            "arm footprint-before-free's expected text lacks the design's")


def test_registration_rejects_an_unlisted_arm():
    rejects(CLAUSES, "registration", planted(CMAKE, ARM_NOOP, ARM_NOOP.replace("return-noop", "return-noop,no-prune")),
            "arm return-noop,no-prune is registered but names no planted fault")


def test_registration_rejects_a_contract_change():
    rejects(CLAUSES, "registration", planted(CMAKE, "PRIVATE ggml-sycl-private-fixtures Threads::Threads)\n        "
                                                    "set_target_properties(${stem}", "PRIVATE llama Threads::Threads)\n"
                                                    "        set_target_properties(${stem}"),
            "no longer links the private fixtures")
    rejects(CLAUSES, "registration", planted(CMAKE, 'set(_env "ONEAPI_DEVICE_SELECTOR=level_zero:1"',
                                             'set(_env "ONEAPI_DEVICE_SELECTOR=level_zero:0,1"'),
            "pins the selector to level_zero:1")


def test_registration_rejects_a_registration_outside_the_block():
    texts = tree_texts(FILES)
    cmake = texts.get(CMAKE, "")
    m = re.search(r"^[ \t]*" + FUNC + r"\s*\(\s*" + STEM + r"\s", cmake, re.M)
    assert m, "control precondition: the registration is not found"
    texts[CMAKE] = cmake[:m.start()] + "endif()\nif (OTHER)\n" + cmake[m.start():]
    rejects(CLAUSES, "registration", texts, f"{STEM} is registered outside")


def test_census_rejects_an_unscored_or_unplanted_fault():
    rejects(CLAUSES, "census", planted(COMMON, '"return-noop"', '"return-nop"'),
            "names 'return-noop', which the selector's table lacks")
    rejects(CLAUSES, "census", planted(UC, "scratch_test_mutant_on(scratch_test_mutant::RETURN_NOOP)", "false"),
            "scratch_test_mutant::RETURN_NOOP has no planted block in production code")


def test_census_rejects_a_block_production_compiles():
    texts = tree_texts(FILES)
    uc = texts.get(UC, "")
    call = "scratch_test_mutant_on(scratch_test_mutant::RETURN_NOOP)"
    assert call in uc, "control precondition: the RETURN_NOOP block is not in unified-cache.cpp"
    texts[UC] = uc + "\nstatic bool planted_control() { return " + call + "; }\n"
    rejects(CLAUSES, "census", texts, "RETURN_NOOP is tested outside a GGML_SYCL_SCRATCH_MUTANTS block")


def test_test_source_rejects_a_lost_case_and_a_direct_trial():
    rejects(CLAUSES, "test-source", planted(TEST, 'n == "status-order"', 'n == "status_order"'),
            "run_case has no dispatch for case status-order")
    rejects(CLAUSES, "test-source", planted(TEST, '"ggml_backend_sycl_host_scratch_trial")', '"other")'),
            "does not resolve ggml_backend_sycl_host_scratch_trial by proc name")
    rejects(CLAUSES, "test-source", planted(TEST, "r.ops.plan_scratch_bytes = 0;", "r.ops.plan_scratch_bytes = 1;"),
            "plan term of 0")


def test_test_source_rejects_an_unprinted_phrase():
    rejects(CLAUSES, "test-source", planted(TEST, "footprint decremented before the free", "footprint moved"),
            "the test prints no 'footprint decremented before the free'")


def test_production_rejects_a_missing_step_and_a_keep_that_walks_the_set():
    rejects(CLAUSES, "production", planted(UC, "case host_scratch_trial_op::MARK:", "default:"),
            "host_scratch_trial has no case for MARK")
    texts = tree_texts(FILES)
    uc = texts.get(UC, "")
    at = uc.find("case host_scratch_trial_op::KEEP:")
    assert at >= 0, "control precondition: no KEEP case"
    cut = at + len("case host_scratch_trial_op::KEEP:")
    texts[UC] = uc[:cut] + " (void) marked_set_.size();" + uc[cut:]
    rejects(CLAUSES, "production", texts, "host_scratch_trial's KEEP touches the marked set (marked_set_)")


def test_production_rejects_an_uncharged_grow_and_a_lost_witness():
    texts = tree_texts(FILES)
    uc = texts.get(UC, "")
    sig = "bool unified_cache::grow_scratch_zone(size_t request_bytes)"
    grow = body(uc, sig)
    assert "host_scratch_trial_note_grow(" in grow, "control precondition: grow_scratch_zone charges no grow"
    texts[UC] = uc.replace(grow, grow.replace("host_scratch_trial_note_grow(", "(void) sizeof("), 1)
    rejects(CLAUSES, "production", texts, "grow_scratch_zone (the ledger's ensure) does not charge its grow")
    # Anchored on the witness itself: other lines of the file carry `n_ubatch=%u` too.
    rejects(CLAUSES, "production", planted(UC, '"[HOST-ARENA] SCRATCH %s dev=%d n_ubatch=%u',
                                           '"[HOST-ARENA] SCRATCH %s dev=%d n_ubatch=%d'), "witness")


def test_negative_control_comments_change_nothing():
    texts = tree_texts(FILES)
    for rel in FILES:
        if rel in texts:
            note = "\n# a comment\n" if rel.endswith(".txt") else "\n// a comment: case host_scratch_trial_op::KEEP:\n"
            texts[rel] += note
    for clause, check in CLAUSES.items():
        assert check(texts) == check(tree_texts(FILES)), f"{clause} changed under appended comments"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
