#!/usr/bin/env python3
"""PP MoE oneDNN scratch hard-cap source contract (llama.cpp-ijla, canonical §1.1).

Host-only: reads sources, runs no build, loads no model and touches no device.

The companion executable test (test-moe-scratch-admission) proves the cap policy
against shapes it constructs, including that a refusal reaches no side effect.
It cannot see whether PRODUCTION asks the policy anything, which is what this
gate covers: that both live PP MoE executor call sites admit before they reserve
and pass the planned sizes verbatim rather than max(planned, required), that the
batched executor has no general temporary fallback left to make an inadmissible
batch fit, that unified_cache::reserve_pp_moe_onednn_scratch refuses before its
mutex and before its allocator, and that the generation/rollback protocol the
32dg lifecycle work established survives at both sites.

Every check below fails against the pre-ijla tree. Run it that way to confirm:

    ./tests/test-sycl-pp-moe-scratch-admission-contract.py \\
        --sycl <(git show 745062d4e:ggml/src/ggml-sycl/ggml-sycl.cpp) \\
        --cache <(git show 745062d4e:ggml/src/ggml-sycl/unified-cache.cpp)

The `anchors` section proves every region the checks read was actually found, so
a renamed lambda reports a missing anchor instead of passing vacuously. Run with
--self-test to prove the absence-based checks fire against mutants.
"""
import argparse
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--sycl", default=str(root / "ggml/src/ggml-sycl/ggml-sycl.cpp"))
parser.add_argument("--cache", default=str(root / "ggml/src/ggml-sycl/unified-cache.cpp"))
parser.add_argument("--module", default=str(root / "ggml/src/ggml-sycl/moe-scratch-admission.cpp"))
parser.add_argument("--header", default=str(root / "ggml/src/ggml-sycl/moe-scratch-admission.hpp"))
parser.add_argument("--doc", default=str(root / "docs/design/sycl-canonical-memory-architecture.md"))
parser.add_argument("--self-test", action="store_true",
                    help="prove the absence-based checks fire when the thing they forbid is present")
args = parser.parse_args()


def strip_comments(source):
    """Remove C and C++ comments, preserving string literals and line count.

    Every check below reads active code only. Without this, a check like "no
    general temporary fallback survives in the batched executor" is spoofed by
    the COMMENT that explains why there is none -- which this very file's
    production comment would do.
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
            out.append("\n" * source.count("\n", i, end))
            i = end
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def squeeze(source):
    """Collapse runs of spaces/tabs to one, preserving line structure.

    Every needle below is written with single spaces so clang-format's column
    alignment cannot break a check by padding an assignment.
    """
    return re.sub(r"[ \t]+", " ", source)


def between(source, start, end):
    """Text from the first `start` to the first `end` that follows it."""
    at = source.find(start)
    if at < 0:
        return ""
    stop = source.find(end, at + len(start))
    return source[at:stop + len(end)] if stop >= 0 else ""


def ordered(text, *needles):
    """True when every needle appears, in the given order."""
    at = -1
    for needle in needles:
        found = text.find(needle, at + 1)
        if found < 0:
            return False
        at = found
    return True


def evaluate(sycl, cache, module, header, doc=""):
    sycl, cache = squeeze(sycl), squeeze(cache)
    module, header = squeeze(module), squeeze(header)
    # The contract is prose: collapse every run of whitespace so a reflowed
    # paragraph does not break a phrase that spans a line break.
    doc = " ".join(doc.split())

    # The batched MXFP4 SoA oneDNN executor: the site that must run on planner
    # scratch or not at all.
    batched = between(sycl, "auto try_pp_mxfp4_soa_onednn_f16_batched = ",
                      "if (try_pp_mxfp4_soa_onednn_f16_batched())")
    # The non-batched PP MXFP4 SoA f16 staging: the terminal MoE PP route, which
    # keeps its transient staging but must not enlarge the planned ring.
    staging = between(sycl, "scoped_unified_queue_temp<sycl::half> pp_mxfp4_src0_f16;", "\n#endif\n")
    reserve = between(cache, "bool unified_cache::reserve_pp_moe_onednn_scratch(",
                      "bool unified_cache::claim_pp_moe_onednn_scratch_slot(")

    admit_at = reserve.find("pp_moe_onednn_preflight_scratch(")
    lock_at = reserve.find("std::lock_guard<std::mutex> lock(pp_moe_onednn_scratch_mutex_)")
    allocate_at = reserve.find("auto allocate_buffer")

    anchors = {
        "canonical memory architecture contract": doc,
        "batched PP MoE oneDNN executor body": batched,
        "non-batched PP MXFP4 f16 staging body": staging,
        "unified_cache::reserve_pp_moe_onednn_scratch body": reserve,
        "moe-scratch-admission module": module,
        "moe-scratch-admission header": header,
    }

    checks = {
        # --- the point of the task: the plan is a cap at both live call sites ---
        "the batched executor admits before it reserves, claims and submits":
            ordered(batched, "pp_moe_onednn_admit_scratch(planned_shape, required_shape)",
                    "reserve_pp_moe_onednn_scratch(planned_weight, planned_act, planned_out",
                    "pp_moe_onednn_claim_scratch_slot(",
                    "claim_pp_moe_onednn_scratch_slot(scratch_slot, reserved)",
                    "dequantize_row_mxfp4_soa_to_fp16_rowmajor(",
                    "DnnlGemmWrapper::gemm_batch_strided("),
        "the batched executor refuses with the typed reason name":
            "return reject_batched(ggml_sycl::pp_moe_onednn_scratch_admission_reason_name(admission.reason));"
            in batched,
        "the non-batched staging admits before it reserves":
            ordered(staging, "pp_moe_onednn_admit_scratch(planned_shape, required_shape)",
                    "reserve_pp_moe_onednn_scratch(planned_weight_slot, planned_activation_slot,"),
        # A refusal that nobody can see is indistinguishable from a route that
        # was never eligible, which is how the max() upsize survived unnoticed.
        "the non-batched staging reports its refusal reason":
            ordered(staging, "if (!admission.allowed && pp_oor_trace) {",
                    "pp_moe_onednn_scratch_admission_reason_name(admission.reason)"),
        "the non-batched staging skips the cache entirely when refused":
            "admission.allowed ? ggml_sycl::get_unified_cache(*ctx.stream()) : nullptr" in staging,
        # ABSENCE: the bug this task exists for. max(planned, required) turns a
        # budgeted zone into a high-water mark of every shape ever seen.
        "no PP MoE scratch reservation upsizes past the plan":
            "std::max(planned" not in sycl,
        # ABSENCE: with a general fallback present, refusing costs nothing and
        # the cap is decorative -- the batch simply allocates its own scratch.
        "the batched executor keeps no general temporary fallback":
            "scoped_unified_queue_temp" not in batched
            and not any(cohort in batched for cohort in ("moe_pp_batched_f16_weights",
                                                         "moe_pp_batched_f16_acts",
                                                         "moe_pp_batched_f32_out")),
        "the batched executor reports its scratch source as the planned ring":
            "scratch=planned-unified-cache" in batched and "scratch_from_unified_cache" not in sycl,

        # --- the cache enforces the same cap on its own entry ---
        "the reservation admits before its mutex and before its allocator":
            admit_at >= 0 and lock_at > admit_at and allocate_at > admit_at,
        "a refused reservation returns false and names the reason":
            ordered(reserve, "if (!admission.allowed) {",
                    "refusing unplanned PP MoE oneDNN scratch reservation",
                    "pp_moe_onednn_scratch_admission_reason_name(admission.reason)",
                    "return false;"),
        "the reservation allocates the admitted shape, not the raw request":
            ordered(reserve, "weight_slot_bytes = admitted.weight_slot_bytes;",
                    "activation_slot_bytes = admitted.activation_slot_bytes;",
                    "output_slot_bytes = admitted.output_slot_bytes;",
                    "ring_depth = admitted.ring_depth;"),
        # ABSENCE: the unchecked rounding this replaced. align_up wraps on an
        # unrepresentable size, and a wrapped size compares as small.
        "the reservation no longer rounds its slot sizes unchecked":
            "align_up(weight_slot_bytes, 256)" not in reserve,

        # --- the 32dg claim/release protocol must survive the port ---
        "the batched executor still binds the slot generation it claimed":
            "batched_scratch_claim.activate(ctx.device, ring_depth, scratch_slot, reserved.generation" in batched,
        "the batched executor still rolls back a slot it could not bind":
            "pp_moe_onednn_rollback_unbound_scratch_slot(ctx.device, ring_depth, scratch_slot);" in batched,
        "the batched executor still releases a claim it did not use":
            "batched_scratch_claim.release_unused();" in batched,
        "the non-batched staging still binds and rolls back by generation":
            ordered(staging, "pp_moe_onednn_claim_scratch_slot(ctx.device, planned_ring_depth, scratch_slot)",
                    "pp_moe_onednn_rollback_unbound_scratch_slot(ctx.device, planned_ring_depth, scratch_slot)"),

        # --- the module's dependency-freedom, which the host test target needs ---
        # ABSENCE: one SYCL include here and test-moe-scratch-admission stops
        # building, taking the only executable proof of the policy with it.
        "the admission module includes no SYCL or backend header":
            not re.search(r'#\s*include\s*[<"](?:sycl/|CL/|.*unified-cache\.hpp|.*common\.hpp|ggml)',
                          module + header),
        "every refusal reason has a name that reaches a log":
            all('"{}"'.format(name) in module
                for name in ("allowed", "missing-plan", "invalid-request", "weight-cap",
                             "activation-cap", "output-cap", "ring-cap")),
        "the cap is the aligned plan, so a plan admits itself":
            ordered(module, "pp_moe_onednn_scratch_shape cap = planned;",
                    "pp_moe_onednn_align_scratch_shape(planned, &cap)"),

        # --- the canonical contract, so a future executor cannot reintroduce
        #     the fallback by reading the architecture doc and finding no rule ---
        "the planner category table names PP MoE oneDNN scratch and its zone":
            "| PP MoE oneDNN scratch | `pp_moe_onednn_{weight,activation,output}_slot_bytes`, "
            "`pp_moe_onednn_ring_depth` → Zone: RUNTIME |" in doc,
        "the contract states planned scratch is a hard cap":
            "hard capacity ceilings, not runtime growth hints" in doc,
        "the contract requires failing closed before allocating inadmissible scratch":
            "must fail closed before allocating that scratch and select an already-planned safe route" in doc,
        "the contract preserves intentional host-pinned expert placement":
            "Host-pinned MoE expert weights are **intentional**" in doc,
        "the contract forbids forcing experts into VRAM to qualify for a batch":
            "must not force those experts into VRAM" in doc,
        "the contract records that admission holds no lock and precedes the scratch mutex":
            ordered(doc, "adds **no lock at all**", "pp_moe_onednn_scratch_mutex_"),
    }

    return sorted(name for name, text in anchors.items() if not text), \
           sorted(name for name, ok in checks.items() if not ok), len(checks)


# Positive controls for the absence-based checks. They pass vacuously on any
# tree that never had the forbidden construct -- which is every tree written
# after this port -- so each one is re-evaluated against a source that
# deliberately contains it.
ABSENCE_MUTANTS = {
    # The exact pre-ijla line, restored.
    "no PP MoE scratch reservation upsizes past the plan": (
        "sycl",
        "            if (!cache->reserve_pp_moe_onednn_scratch(planned_weight, planned_act, planned_out, ring_depth)) {",
        "            if (!cache->reserve_pp_moe_onednn_scratch(std::max(planned_weight, weight_bytes),\n"
        "                                                     std::max(planned_act, act_bytes),\n"
        "                                                     std::max(planned_out, out_bytes), ring_depth)) {"),
    "the batched executor keeps no general temporary fallback": (
        "sycl",
        "            pp_moe_onednn_scratch_claim batched_scratch_claim;\n\n"
        "            ggml_sycl::unified_cache * cache = ggml_sycl::get_unified_cache(*ctx.stream());",
        "            pp_moe_onednn_scratch_claim batched_scratch_claim;\n"
        "            scoped_unified_queue_temp<sycl::half> weight_pool;\n\n"
        "            ggml_sycl::unified_cache * cache = ggml_sycl::get_unified_cache(*ctx.stream());"),
    "the batched executor reports its scratch source as the planned ring": (
        "sycl",
        '                        "max_rows=%zu scratch=planned-unified-cache weight=%.1fMB',
        '                        "max_rows=%zu scratch=%s weight=%.1fMB'),
    # Admission after the lock is the failure this whole shape exists to
    # prevent: the refusal still happens, but only once the mutex is held and
    # the ring has been rebuilt.
    "the reservation admits before its mutex and before its allocator": (
        "cache",
        "    const int device_id = ggml_sycl_get_device_id_from_queue(queue_);\n"
        "    const pp_moe_onednn_scratch_shape planned = {",
        "    std::lock_guard<std::mutex> lock(pp_moe_onednn_scratch_mutex_);\n"
        "    const int device_id = ggml_sycl_get_device_id_from_queue(queue_);\n"
        "    const pp_moe_onednn_scratch_shape planned = {"),
    "the reservation no longer rounds its slot sizes unchecked": (
        "cache",
        "    weight_slot_bytes = admitted.weight_slot_bytes;",
        "    weight_slot_bytes = align_up(weight_slot_bytes, 256);"),
    "the admission module includes no SYCL or backend header": (
        "module",
        '#include "moe-scratch-admission.hpp"',
        '#include "moe-scratch-admission.hpp"\n#include <sycl/sycl.hpp>'),
}


def self_test(sycl, cache, module, header, doc):
    """Every absence check must fail once its forbidden construct is injected."""
    problems = []
    for name, (target, anchor, replacement) in ABSENCE_MUTANTS.items():
        sources = {"sycl": sycl, "cache": cache, "module": module, "header": header}
        anchor, replacement = squeeze(anchor), squeeze(replacement)
        if anchor not in squeeze(sources[target]):
            problems.append("mutation anchor missing for: " + name)
            continue
        sources[target] = squeeze(sources[target]).replace(anchor, replacement, 1)
        _, failed, _ = evaluate(sources["sycl"], sources["cache"], sources["module"], sources["header"], doc)
        if name not in failed:
            problems.append("check did not fire on its mutant: " + name)
    return problems


def read(path):
    return strip_comments(Path(path).read_text()) if Path(path).exists() else ""


sycl, cache = read(args.sycl), read(args.cache)
module, header = read(args.module), read(args.header)
# The contract is markdown: comment stripping would eat its code fences.
doc = Path(args.doc).read_text() if Path(args.doc).exists() else ""

missing_anchors, failed, n_checks = evaluate(sycl, cache, module, header, doc)

for name in missing_anchors:
    print("MISSING ANCHOR: " + name)
for name in failed:
    print("FAIL: " + name)

if missing_anchors or failed:
    sys.exit(1)

if args.self_test:
    problems = self_test(sycl, cache, module, header, doc)
    for problem in problems:
        print("SELF-TEST FAIL: " + problem)
    if problems:
        sys.exit(1)
    print("sycl pp-moe scratch admission contract: self-test PASS ({} mutants)".format(len(ABSENCE_MUTANTS)))

print("sycl pp-moe scratch admission contract: PASS ({} checks)".format(n_checks))
