#!/usr/bin/env python3
"""Source contract: how the event-chained hybrid decode route keeps its per-tensor record current, and what it counts.

Host-only: reads ggml-sycl.cpp, moe-hybrid-record.hpp and docs/backend/sycl-env-vars.md, builds nothing, loads no model
and touches no device.

The route (ggml_sycl_moe_hybrid_decode_event_chained) looks its tensor's record up at every decode op
(ggml_sycl_moe_hybrid_decode_record_current). The record is cached under a stamp of (replan epoch, the weight's expert
storage generation, selected rows); a miss rebuilds it (ggml_sycl_moe_hybrid_decode_record_build), which resolves
every expert of the tensor. GGML_SYCL_MOE_HYBRID_RECORD_TRACE counts that work per token, so a device run can say how
many records a token rebuilds, why, and how many experts each rebuild resolves. The host test
(tests/test-sycl-moe-hybrid-record.cpp) proves the bookkeeping; this gate holds where the backend calls it:

  T1  the lookup classifies its stamp with moe_hybrid_record_miss_classify, once, and only a NONE miss returns the
      held decision; the stamp's own current test is not asked a second time beside it;
  T2  the trace level is read once, in a function-local static of ggml_sycl_moe_hybrid_record_trace_level(), from
      GGML_SYCL_MOE_HYBRID_RECORD_TRACE; the name appears nowhere else in the backend;
  T3  every lookup is counted, with its miss, before the held decision returns; the lookup that finds no published
      plan counts itself before it returns; each build's outcome is counted after the build;
  T4  the build counts the experts it resolves at its one resolve, right before the call that resolves them;
  T5  the per-token line and the per-build line are WARN, each printed from one site, the per-build one only at
      level 2 and above; the per-token line's buffer is the 512 bytes the host test proves the widest line fits;
  T6  the header is plain host code: standard includes and moe-decode-hostpath.hpp only, no SYCL;
  T7  the env-var row names the variable, both line tags, the read-once rule and the token rule.

Checks run on comment-blanked text (string literals kept). NOT VACUOUS: --self-test runs the gate on in-memory mutants
of the real sources. Each mutant names the pin that must catch it, and is reported if it survives, if another pin
catches it, or if its anchor no longer matches exactly once (NOT APPLIED). ctest registers it with --self-test.

WHAT THIS DOES NOT PROVE. It reads text. That the counts add up is the host test's job; what they read on Qwen3.8 is
the lead's device run.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"
HEADER = ROOT / "ggml/src/ggml-sycl/moe-hybrid-record.hpp"
DOCS = ROOT / "docs/backend/sycl-env-vars.md"

CURRENT_SIG = "static bool ggml_sycl_moe_hybrid_decode_record_current("
BUILD_SIG = "static ggml_sycl::moe_decode_direct_outcome ggml_sycl_moe_hybrid_decode_record_build("
LEVEL_SIG = "static int ggml_sycl_moe_hybrid_record_trace_level() {"
LOOKUP_SIG = "static void ggml_sycl_moe_hybrid_record_trace_lookup("
ENV_NAME = "GGML_SYCL_MOE_HYBRID_RECORD_TRACE"

CLASSIFY = ("const ggml_sycl::moe_hybrid_record_miss miss = ggml_sycl::moe_hybrid_record_miss_classify(stamp, "
            "replan_epoch, storage_generation, selected_rows);")
HELD_RETURN = ("if (miss == ggml_sycl::MOE_HYBRID_RECORD_MISS_NONE) { *out = weight.moe_hybrid_decode[device]; "
               "return stamp.eligible && *out; }")
TRACE_READ = "const int trace = ggml_sycl_moe_hybrid_record_trace_level();"
TRACE_LOOKUP = "if (trace > 0) { ggml_sycl_moe_hybrid_record_trace_lookup(src0, miss,"
NO_PLAN = ("if (!cache || ggml_sycl_cache_plan_owner(cache)->entries.empty()) { if (trace > 0) { "
           "g_moe_hybrid_record_trace.no_plan(); } return false; }")
BUILD_CALL = "ggml_sycl_moe_hybrid_decode_record_build(ctx, src0, &built, &refusal);"
OUTCOME_COUNT = "g_moe_hybrid_record_trace.build_outcome(outcome);"
LEVEL_BODY = ("{ static const int level = [] { const char * env = std::getenv(\"GGML_SYCL_MOE_HYBRID_RECORD_TRACE\"); "
              "return env ? std::atoi(env) : 0; }(); return level; }")
RESOLVED_COUNT = ("if (ggml_sycl_moe_hybrid_record_trace_level() > 0) { "
                  "g_moe_hybrid_record_trace.resolved(all_experts.size()); }")
RESOLVE_CALL = "const ggml_sycl::moe_resolved_batch_result resolved = ggml_sycl::ggml_sycl_build_moe_resolved_batch("
TOKEN_WARN = 'GGML_LOG_WARN("[MOE-HYBRID-RECORD] %s\\n", line);'
TOKEN_BUFFER = "char line[512];"
BUILD_WARN = 'GGML_LOG_WARN("[MOE-HYBRID-RECORD-BUILD] '
BUILD_GUARD = "if (trace >= 2) {"
WAIT_CALLS = re.compile(r"\.wait\(|wait_and_throw\(|get_info<")

DOC_NEEDLES = ("`GGML_SYCL_MOE_HYBRID_RECORD_TRACE=1\\|2`", "[MOE-HYBRID-RECORD]", "[MOE-HYBRID-RECORD-BUILD]",
               "Read once", "lower block")
HEADER_INCLUDE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]', re.M)
HEADER_ALLOWED = {"moe-decode-hostpath.hpp"}


class ContractError(AssertionError):
    pass


class MutantNotApplied(Exception):
    pass


_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'", re.DOTALL)


def blank_comments(text: str) -> str:
    """Comments become spaces (newlines kept); string and character literals are kept as written."""

    def repl(m: re.Match) -> str:
        t = m.group(0)
        if t.startswith("//") or t.startswith("/*"):
            return re.sub(r"[^\n]", " ", t)
        return t

    return _COMMENT.sub(repl, text)


def squash(text: str) -> str:
    t = re.sub(r"\s+", " ", text)
    t = re.sub(r"\( ", "(", t)
    return re.sub(r" \)", ")", t)


def body(code: str, sig: str, pin: str) -> str:
    """The squashed `{ ... }` body of the one definition that starts with `sig`."""
    n = code.count(sig)
    if n != 1:
        raise ContractError(f"FAIL [pin {pin}]: `{sig}` appears {n} times, not once")
    at = code.find(sig)
    open_at = code.find("{", at + len(sig) - 1 if sig.endswith("{") else at)
    depth = 0
    k = open_at
    while k < len(code):
        c = code[k]
        if c == '"':
            k += 1
            while k < len(code) and code[k] != '"':
                k += 2 if code[k] == "\\" else 1
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return squash(code[open_at : k + 1])
        k += 1
    raise ContractError(f"FAIL [pin {pin}]: the body of `{sig}` is unbalanced")


def once(text: str, needle: str, pin: str, what: str) -> int:
    n = text.count(needle)
    if n != 1:
        raise ContractError(f"FAIL [pin {pin}]: {what}: `{needle[:110]}` appears {n} times, not once")
    return text.find(needle)


def check(backend_raw: str, header_raw: str, docs_raw: str) -> None:
    code = blank_comments(backend_raw)
    current = body(code, CURRENT_SIG, "T1")
    build = body(code, BUILD_SIG, "T4")

    # T1: one classification, and only NONE returns the held decision.
    at_classify = once(current, CLASSIFY, "T1", "the lookup's stamp classification")
    at_held = once(current, HELD_RETURN, "T1", "the held decision's return under a NONE miss")
    if at_held < at_classify:
        raise ContractError("FAIL [pin T1]: the held decision returns before the stamp is classified")
    if "moe_decode_direct_stamp_current(" in current:
        raise ContractError("FAIL [pin T1]: the lookup asks the stamp's current test beside the classification, so "
                            "the two can disagree on what the held record answers for")
    if current.count("return stamp.eligible") != 1:
        raise ContractError("FAIL [pin T1]: the held decision returns from more than one place")

    # T2: the level, read once, from the one variable.
    level = body(code, LEVEL_SIG, "T2")
    if level != LEVEL_BODY:
        raise ContractError(f"FAIL [pin T2]: ggml_sycl_moe_hybrid_record_trace_level() is not the read-once static "
                            f"of {ENV_NAME}: `{level[:160]}`")
    if squash(code).count(ENV_NAME) != 1:
        raise ContractError(f"FAIL [pin T2]: {ENV_NAME} is read in more than one place")

    # T3: every lookup counted before the held return, the no-plan return counted, each build's outcome counted.
    at_read = once(current, TRACE_READ, "T3", "the lookup's one read of the trace level")
    at_lookup = once(current, TRACE_LOOKUP, "T3", "the lookup count")
    if not at_classify < at_read < at_lookup < at_held:
        raise ContractError("FAIL [pin T3]: the lookup is not counted between its classification and the held return")
    at_no_plan = once(current, NO_PLAN, "T3", "the no-plan return and its count")
    at_build = once(current, BUILD_CALL, "T3", "the record build")
    at_outcome = once(current, OUTCOME_COUNT, "T3", "the build's outcome count")
    if not at_held < at_no_plan < at_build < at_outcome:
        raise ContractError("FAIL [pin T3]: the no-plan count, the build and its outcome count are out of order")
    if current.count("return false;") != 1:
        raise ContractError("FAIL [pin T3]: the lookup has a `return false;` besides the no-plan return, which the "
                            "trace does not count")
    if current[at_build:at_outcome].count("return") != 0:
        raise ContractError("FAIL [pin T3]: the lookup returns between the build and its outcome count")

    # T4: the build counts what it resolves, right before resolving it.
    at_count = once(build, RESOLVED_COUNT, "T4", "the build's count of the experts it resolves")
    at_resolve = once(build, RESOLVE_CALL, "T4", "the build's resolve")
    if not at_count < at_resolve or build[at_count + len(RESOLVED_COUNT) : at_resolve].strip():
        raise ContractError("FAIL [pin T4]: the resolved-experts count is not the statement right before the resolve")

    # T5: the two lines are WARN, printed once each, the per-build one at level 2.
    lookup = body(code, LOOKUP_SIG, "T5")
    once(lookup, TOKEN_WARN, "T5", "the per-token WARN line")
    once(lookup, TOKEN_BUFFER, "T5", "the per-token line's 512-byte buffer")
    sq = squash(code)
    if sq.count("[MOE-HYBRID-RECORD]") != 1 or sq.count("[MOE-HYBRID-RECORD-BUILD]") != 1:
        raise ContractError("FAIL [pin T5]: a hybrid record trace line is printed from more than one site")
    at_guard = once(current, BUILD_GUARD, "T5", "the per-build line's level guard")
    at_warn = once(current, BUILD_WARN, "T5", "the per-build WARN line")
    if not at_outcome < at_guard < at_warn:
        raise ContractError("FAIL [pin T5]: the per-build line is not printed under its level-2 guard after the "
                            "outcome count")
    if WAIT_CALLS.search(lookup):
        raise ContractError("FAIL [pin T5]: the per-token count waits on the device")

    # T6: the header stays plain host code.
    header = blank_comments(header_raw)
    for m in HEADER_INCLUDE.finditer(header):
        kind, name = m.group(1), m.group(2)
        if kind == '"' and name not in HEADER_ALLOWED:
            raise ContractError(f"FAIL [pin T6]: moe-hybrid-record.hpp includes the project header `{name}`")
        if kind == "<" and ("sycl" in name or "/" in name):
            raise ContractError(f"FAIL [pin T6]: moe-hybrid-record.hpp includes `<{name}>`")
    if re.search(r"\bsycl::", header):
        raise ContractError("FAIL [pin T6]: moe-hybrid-record.hpp names a SYCL type")

    # T7: the env-var row.
    rows = [line for line in docs_raw.splitlines() if line.startswith(f"| `{ENV_NAME}")]
    if len(rows) != 1:
        raise ContractError(f"FAIL [pin T7]: docs/backend/sycl-env-vars.md has {len(rows)} rows for {ENV_NAME}, "
                            f"not one")
    for needle in DOC_NEEDLES:
        if needle not in rows[0]:
            raise ContractError(f"FAIL [pin T7]: the {ENV_NAME} row does not say `{needle}`")


def self_test(raws: dict[str, str]) -> int:
    failures: list[str] = []
    n_mutants = 0

    def mutate(file: str, edits: list) -> dict[str, str]:
        out = dict(raws)
        text = out[file]
        for old, new in edits:
            if text.count(old) != 1:
                raise MutantNotApplied(f"anchor matches {text.count(old)} times in {file}: {old[:70]!r}")
            text = text.replace(old, new, 1)
        out[file] = text
        return out

    def expect_fail(name: str, pin: str, file: str, edits: list) -> None:
        nonlocal n_mutants
        n_mutants += 1
        try:
            mutated = mutate(file, edits)
        except MutantNotApplied as e:
            failures.append(f"{name}: NOT APPLIED ({e})")
            print(f"  mutant {name}: NOT APPLIED ({e})")
            return
        try:
            check(mutated["backend"], mutated["header"], mutated["docs"])
        except ContractError as e:
            if f"[pin {pin}]" not in str(e):
                failures.append(f"{name}: caught, but not by pin {pin}: {str(e)[:140]}")
                print(f"  mutant {name}: caught by the WRONG pin ({str(e)[:120]}...)")
                return
            print(f"  mutant {name}: caught (pin {pin})")
            return
        failures.append(f"{name}: survived")
        print(f"  mutant {name}: SURVIVED")

    B, H, D = "backend", "header", "docs"
    held = "    if (miss == ggml_sycl::MOE_HYBRID_RECORD_MISS_NONE) {"
    expect_fail("held-on-generation-miss", "T1", B,
                [(held, "    if (miss == ggml_sycl::MOE_HYBRID_RECORD_MISS_NONE ||\n"
                        "        miss == ggml_sycl::MOE_HYBRID_RECORD_MISS_GENERATION) {")])
    expect_fail("held-on-old-test", "T1", B,
                [(held, "    if (ggml_sycl::moe_decode_direct_stamp_current(stamp, replan_epoch, storage_generation,\n"
                        "                                                   selected_rows)) {")])
    expect_fail("level-read-per-call", "T2", B,
                [("    static const int level = [] {\n        const char * env = std::getenv(\"" + ENV_NAME,
                  "    const int level = [] {\n        const char * env = std::getenv(\"" + ENV_NAME)])
    expect_fail("env-read-twice", "T2", B,
                [("    const int trace = ggml_sycl_moe_hybrid_record_trace_level();",
                  "    const int trace = std::getenv(\"GGML_SYCL_MOE_HYBRID_RECORD_TRACE\") ? 1 : 0;")])
    expect_fail("lookup-not-counted", "T3", B,
                [("        ggml_sycl_moe_hybrid_record_trace_lookup(src0, miss,",
                  "        (void) ggml_sycl_moe_hybrid_record_trace_lookup;\n        (void) (src0, miss,")])
    expect_fail("no-plan-not-counted", "T3", B,
                [("        if (trace > 0) {\n            g_moe_hybrid_record_trace.no_plan();\n        }\n", "")])
    expect_fail("outcome-not-counted", "T3", B,
                [("        g_moe_hybrid_record_trace.build_outcome(outcome);\n", "")])
    expect_fail("resolve-not-counted", "T4", B,
                [("        g_moe_hybrid_record_trace.resolved(all_experts.size());\n", "")])
    expect_fail("token-line-at-info", "T5", B,
                [('GGML_LOG_WARN("[MOE-HYBRID-RECORD] %s\\n", line);',
                  'GGML_LOG_INFO("[MOE-HYBRID-RECORD] %s\\n", line);')])
    expect_fail("token-buffer-shrunk", "T5", B,
                [("        char line[512];\n        ggml_sycl::moe_hybrid_record_line(",
                  "        char line[256];\n        ggml_sycl::moe_hybrid_record_line(")])
    expect_fail("build-line-at-info", "T5", B,
                [('GGML_LOG_WARN(\n                "[MOE-HYBRID-RECORD-BUILD] ',
                  'GGML_LOG_INFO(\n                "[MOE-HYBRID-RECORD-BUILD] ')])
    expect_fail("build-line-at-level-1", "T5", B, [("        if (trace >= 2) {", "        if (trace >= 1) {")])
    expect_fail("header-includes-sycl", "T6", H, [("#include <cstdio>", "#include <sycl/sycl.hpp>\n#include <cstdio>")])
    expect_fail("header-includes-common", "T6", H, [("#include <cstdio>", '#include "common.hpp"\n#include <cstdio>')])
    expect_fail("docs-row-dropped", "T7", D, [(f"| `{ENV_NAME}=1\\|2`", f"| `{ENV_NAME}_OLD=1\\|2`")])
    expect_fail("docs-row-lacks-build-line", "T7", D, [("`[MOE-HYBRID-RECORD-BUILD]", "`[MOE-HYBRID-BUILD]")])

    try:
        mutate(B, [("this anchor is in no source", "x")])
        failures.append("stale-anchor-not-reported")
        print("  self-test: a stale mutant anchor is NOT reported")
    except MutantNotApplied:
        print("  self-test: a stale mutant anchor is reported")

    try:
        check(raws["backend"], raws["header"], raws["docs"])
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
    raws = {}
    for key, path in (("backend", BACKEND), ("header", HEADER), ("docs", DOCS)):
        try:
            raws[key] = path.read_text()
        except FileNotFoundError:
            print(f"FAIL: {path.relative_to(ROOT)} does not exist")
            return 1
    if "--self-test" in argv:
        return self_test(raws)
    try:
        check(raws["backend"], raws["header"], raws["docs"])
    except ContractError as e:
        print(e)
        return 1
    print("PASS: the hybrid decode record lookup is classified once and counted at every exit, at WARN")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
