#!/usr/bin/env python3
"""Exact MoE CONTROL sizing source contract (llama.cpp-dimc, canonical §12.5).

Host-only: reads sources, runs no build, loads no model and touches no device.

The companion executable test (test-moe-control-plan) proves the sizing and the
reservation ledger against inputs it constructs. It cannot see whether the
PRODUCTION planner actually calls any of it, which is what this gate covers:
that populate_host_zone_sizing sizes the pointer tables from the descriptors
rather than from MAX_EXPERTS * n_moe_layers, that the coarse formula survives
only as the invalid-layout fallback, that the RUNTIME zone sums its terms with
the checked helper, and that the duplicated role enum is pinned by
static_asserts.

Every check below fails against the pre-dimc tree. Run it that way to confirm:

    ./tests/test-sycl-moe-control-plan-contract.py \\
        --cache <(git show 0bdef9ddd:ggml/src/ggml-sycl/unified-cache.cpp) \\
        --module ggml/src/ggml-sycl/moe-control-plan.cpp \\
        --header ggml/src/ggml-sycl/moe-control-plan.hpp

The `anchors` section proves every string the checks key off exists, so a
renamed symbol reports a missing anchor instead of passing vacuously. Run with
--self-test to prove the four absence-based checks fire against mutants.
"""
import argparse
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--cache", default=str(root / "ggml/src/ggml-sycl/unified-cache.cpp"))
parser.add_argument("--module", default=str(root / "ggml/src/ggml-sycl/moe-control-plan.cpp"))
parser.add_argument("--header", default=str(root / "ggml/src/ggml-sycl/moe-control-plan.hpp"))
parser.add_argument("--self-test", action="store_true",
                    help="prove the absence-based checks fire when the thing they forbid is present")
args = parser.parse_args()


def strip_comments(source):
    """Remove C and C++ comments, preserving string literals and line count.

    Every check below reads active code only. Without this a check like "the
    requirement keys on validity, not GPU eligibility" is spoofed by a COMMENT
    saying the word -- which is how this gate first failed against a tree that
    satisfied it. String literals are preserved because two checks read them.
    """
    out = []
    i = 0
    n = len(source)
    while i < n:
        ch = source[i]
        if ch in ('"', "'"):
            quote = ch
            out.append(ch)
            i += 1
            while i < n:
                out.append(source[i])
                if source[i] == "\\":
                    if i + 1 < n:
                        out.append(source[i + 1])
                        i += 2
                        continue
                elif source[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        if source.startswith("//", i):
            while i < n and source[i] != "\n":
                i += 1
            continue
        if source.startswith("/*", i):
            end = source.find("*/", i + 2)
            end = n if end < 0 else end + 2
            # Keep the newlines so line-oriented patterns still line up.
            out.append("\n" * source.count("\n", i, end))
            i = end
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def squeeze(source):
    """Collapse runs of spaces/tabs to one, preserving line structure.

    Every needle below is written with single spaces so that clang-format's
    column alignment cannot break a check. It already did once: aligning
    `plan.moe_expert_ptrs_bytes  =` to the field beside it turned a passing
    check red without changing a line of behaviour.
    """
    return re.sub(r"[ \t]+", " ", source)


def body_of(source, signature_prefix):
    """Text from a definition's signature to its first column-0 closing brace.

    `[^;]*?` before the opening brace is what skips a forward declaration: the
    declaration's terminating `;` cannot be crossed, so the match lands on the
    definition even when the prototype appears first.
    """
    match = re.search(re.escape(signature_prefix) + r"[^;]*?\)\s*(?:noexcept\s*)?\{.*?^\}\n",
                      source, re.S | re.M)
    return match.group(0) if match else ""


def ordered(text, *needles):
    """True when every needle appears, in the given order."""
    at = -1
    for needle in needles:
        found = text.find(needle, at + 1)
        if found < 0:
            return False
        at = found
    return True


def evaluate(cache, module, header):
    cache, module, header = squeeze(cache), squeeze(module), squeeze(header)
    sizing = body_of(cache, "static void populate_host_zone_sizing(")
    zone = body_of(cache, "bool unified_cache_get_planned_runtime_zone_requirement(")
    publish = body_of(cache, "void unified_cache_set_planned_moe_control_requirement(")
    requirement = body_of(module, "moe_control_requirement moe_control_requirement_from_layout(")
    rollback = body_of(module, "void moe_rollback_control_conversion(")
    begin = body_of(module, "moe_control_conversion_ticket moe_begin_control_conversion(")

    # Anchors: every region the checks read must have been found. A rename that
    # empties one of these reports a missing anchor rather than a silent pass.
    anchors = {
        "populate_host_zone_sizing body": sizing,
        "unified_cache_get_planned_runtime_zone_requirement body": zone,
        "unified_cache_set_planned_moe_control_requirement body": publish,
        "moe_control_requirement_from_layout body": requirement,
        "moe_rollback_control_conversion body": rollback,
        "moe_begin_control_conversion body": begin,
    }

    # The coarse assignment, as it stood before this port. Used by two checks:
    # one that the exact path exists, one that this is no longer the only path.
    coarse = "static_cast<size_t>(k_max_experts_prealloc) * sizeof(void *) * static_cast<size_t>(n_moe_layers)"
    coarse_at = sizing.find(coarse)
    fallback_at = sizing.find("if (control_layout.valid) {")

    checks = {
        # --- the point of the task: exact sizing, live in the planner ---
        "the planner builds the exact descriptor plan and control layout":
            ordered(sizing, "moe_build_ptr_table_plan(moe_descs, n_experts)",
                    "moe_build_context_control_layout(ptr_tables, n_expert_used, MOE_GPU_UBATCH_MAX)"),
        "the pointer-table budget comes from the layout, not from MAX_EXPERTS":
            "plan.moe_expert_ptrs_bytes = control_layout.tables_bytes;" in sizing,
        "the control pool total is recorded for the RUNTIME zone":
            "plan.moe_control_pool_bytes = control_layout.total_bytes;" in sizing,
        # ABSENCE: the coarse formula may exist only AFTER the validity branch,
        # i.e. as the fallback. An unconditional coarse assignment means the
        # exact sizing is dead code sitting next to the bug it replaces.
        "the coarse formula survives only as the invalid-layout fallback":
            coarse_at >= 0 and fallback_at >= 0 and coarse_at > fallback_at,
        "the plan publishes its per-device control requirement":
            "unified_cache_set_planned_moe_control_requirement(" in sizing
            and "moe_control_requirement_from_layout(plan.base_context_control_layout" in sizing,
        "a multi-device plan publishes for every participating device":
            ordered(sizing, "if (plan.devices.empty()) {",
                    "unified_cache_set_planned_moe_control_requirement(plan.device_id",
                    "for (int device : plan.devices) {"),

        # --- the RUNTIME zone arithmetic ---
        "the runtime zone requirement is summed with the checked helper":
            "moe_checked_runtime_zone_requirement(" in zone,
        "the runtime zone requirement includes the MoE control term":
            "unified_cache_get_planned_moe_control_requirement(device_id)" in zone,
        # ABSENCE: the unchecked sum this replaced. Reintroducing it lets a
        # poisoned term become a SMALLER zone instead of a refusal.
        "the zone sizing no longer adds its terms unchecked":
            "planned_pp_pipeline + planned_pp_moe_onednn" not in cache,
        "an unanswerable requirement keeps the default zone rather than shrinking it":
            ordered(cache, "if (!unified_cache_get_planned_runtime_zone_requirement(dev_id, &planned_runtime_scratch))",
                    "planned_runtime_scratch = 0;"),

        # --- fail-closed semantics that the executable test pins from below ---
        # ABSENCE: the flag must record INVALID, so a device nobody planned for
        # reads valid-zero. Storing validity the other way makes every dense
        # model poison its own zone sizing.
        "unplanned devices read as a valid zero requirement":
            "g_planned_moe_control_invalid" in publish and "g_planned_moe_control_valid" not in cache,
        # The bug the executable test caught: keying on gpu_eligible makes every
        # dense model unsizable, because a dense layout is valid but not eligible.
        "the requirement keys on layout validity, not GPU eligibility":
            "layout.valid" in requirement and "gpu_eligible" not in requirement,

        # --- the transactional conversion protocol ---
        "rollback cannot throw":
            "noexcept" in rollback,
        "rollback restores the reservation and clears its generation":
            ordered(rollback, "moe_control_reservation_state::UNCONSUMED",
                    "conversion_id = 0", "allocation_claimed = false"),
        "rollback reaps an entry whose last reference went away mid-conversion":
            ordered(rollback, "references == 0", "erase(plan_it)"),
        "each conversion attempt is stamped with a fresh generation":
            ordered(begin, "g_conversion_id.fetch_add(", "conversion_id = conversion_id"),
        "the guard rolls back unless commit was reached":
            ordered(header, "bool commit() {", "if (!committed) {", "moe_rollback_control_conversion(ticket);"),

        # --- the duplicated role enum is pinned ---
        "every duplicated role is pinned by a static_assert":
            all("static_cast<int>(moe_expert_role::{})".format(role) in cache
                for role in ("UNKNOWN", "GATE", "UP", "DOWN", "GATE_UP",
                             "CHUNK_GATE", "CHUNK_UP", "CHUNK_DOWN", "COUNT")),

        # --- the module's dependency-freedom, which the CMake target relies on ---
        # ABSENCE: one SYCL include here and the host test target stops linking.
        "the module includes no SYCL or backend header":
            not re.search(r'#\s*include\s*[<"](?:sycl/|CL/|.*unified-cache\.hpp|.*common\.hpp|ggml)', module + header),
        "the module owns the grovemoe chunked family":
            '"_chexps"' in module and "CHUNK_GATE" in module,
    }

    return sorted(name for name, text in anchors.items() if not text), \
           sorted(name for name, ok in checks.items() if not ok), len(checks)


# Positive controls for the absence-based checks. They pass vacuously on any
# tree that never had the forbidden construct, so each one is re-evaluated
# against a source that deliberately contains it.
ABSENCE_MUTANTS = {
    "the coarse formula survives only as the invalid-layout fallback": (
        "cache",
        "        const moe_ptr_table_plan ptr_tables = moe_build_ptr_table_plan(moe_descs, n_experts);",
        "        plan.moe_expert_ptrs_bytes =\n"
        "            static_cast<size_t>(k_max_experts_prealloc) * sizeof(void *) * static_cast<size_t>(n_moe_layers);\n"
        "        const moe_ptr_table_plan ptr_tables = moe_build_ptr_table_plan(moe_descs, n_experts);"),
    "the zone sizing no longer adds its terms unchecked": (
        "cache",
        "    size_t planned_runtime_scratch = 0;",
        "    size_t planned_runtime_scratch = planned_pp_pipeline + planned_pp_moe_onednn;"),
    "unplanned devices read as a valid zero requirement": (
        "cache",
        "static std::atomic<bool>   g_planned_moe_control_invalid[GGML_SYCL_MAX_DEVICES]{};",
        "static std::atomic<bool>   g_planned_moe_control_valid[GGML_SYCL_MAX_DEVICES]{};"),
    "the module includes no SYCL or backend header": (
        "module",
        '#include "moe-control-plan.hpp"',
        '#include "moe-control-plan.hpp"\n#include <sycl/sycl.hpp>'),
}


def self_test(cache, module, header):
    """Every absence check must fail once its forbidden construct is injected."""
    problems = []
    for name, (target, anchor, replacement) in ABSENCE_MUTANTS.items():
        sources = {"cache": cache, "module": module, "header": header}
        if anchor not in sources[target]:
            problems.append("mutation anchor missing for: " + name)
            continue
        sources[target] = sources[target].replace(anchor, replacement, 1)
        _, failed, _ = evaluate(sources["cache"], sources["module"], sources["header"])
        if name not in failed:
            problems.append("check did not fire on its mutant: " + name)
    return problems


cache = strip_comments(Path(args.cache).read_text())
module = strip_comments(Path(args.module).read_text()) if Path(args.module).exists() else ""
header = strip_comments(Path(args.header).read_text()) if Path(args.header).exists() else ""

missing_anchors, failed, n_checks = evaluate(cache, module, header)

for name in missing_anchors:
    print("MISSING ANCHOR: " + name)
for name in failed:
    print("FAIL: " + name)

if missing_anchors or failed:
    sys.exit(1)

if args.self_test:
    problems = self_test(cache, module, header)
    for problem in problems:
        print("SELF-TEST FAIL: " + problem)
    if problems:
        sys.exit(1)
    print("sycl moe control plan contract: self-test PASS ({} mutants)".format(len(ABSENCE_MUTANTS)))

print("sycl moe control plan contract: PASS ({} checks)".format(n_checks))
