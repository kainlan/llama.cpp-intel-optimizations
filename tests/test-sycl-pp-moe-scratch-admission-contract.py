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
mutex and before its allocator, that each refusal is reported at WARN
unconditionally and once per process rather than only under a debug flag, and
that the generation/rollback protocol the 32dg lifecycle work established
survives at both sites.

The checks that read ggml-sycl.cpp and unified-cache.cpp fail against the pre-ijla tree, and 18 of them do when run that
way (the others read the admission module, its header and the canonical contract, which that tree never had, so they
pass there by construction and are covered by --self-test's mutants instead):

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


def reserve_call_arguments(source):
    """Argument text of every `->reserve_pp_moe_onednn_scratch(...)` call in `source`."""
    calls = []
    for match in re.finditer(r"->\s*reserve_pp_moe_onednn_scratch\s*\(", source):
        depth, index = 1, match.end()
        while index < len(source) and depth:
            depth += (source[index] == "(") - (source[index] == ")")
            index += 1
        calls.append(source[match.end():index - 1])
    return calls


# The planned slot sizes are pinned POSITIVELY, not by a blacklist of the spellings of max(planned, required): a
# blacklist (std::max, ternary, GGML_MAX, sycl::max, `using std::max`, an alias) is always one spelling behind, and it
# collides with the legitimate maxima elsewhere in the file (479i's dense dequant buffers are sized to their own
# graph-entry demand). The property is that the ring the planner sized is what gets reserved, so, per executor body:
#   * the three slot sizes and the raw ring depth are each read ONCE, as `const T NAME = <planner getter>(ctx.device);`,
#     and the runtime ring depth is derived once from the raw one;
#   * nothing assigns to, increments, address-takes, casts around or redeclares those names afterwards, and no
#     preprocessor directive can redefine them;
#   * the admission shape is exactly `{slots..., raw ring depth}` and the executor's only reserve call passes exactly
#     `slots..., ring depth`, nothing computed from them;
# and file-wide: the number of reserve_pp_moe_onednn_scratch( and unified_cache_set_planned_pp_moe_onednn_scratch(
# calls is fixed, and the free unified_cache_reserve_ wrapper is not used, so a new reservation site (or one that
# raises the ceiling and reads it back) cannot appear without editing this gate.
RESERVE_CALLS_IN_FILE = 5   # plan, replan x2, the two executors
SET_PLANNED_CALLS_IN_FILE = 3   # plan, replan x2
PLANNED_SITES = (
    dict(slots=(("planned_weight", "weight"), ("planned_act", "activation"), ("planned_out", "output")),
         shape="planned_shape", raw="planned_ring_depth", ring="ring_depth"),
    dict(slots=(("planned_weight_slot", "weight"), ("planned_activation_slot", "activation"),
                ("planned_output_slot", "output")),
         shape="planned_shape", raw="planned_ring_depth_raw", ring="planned_ring_depth"),
)
_WRITES = (r"\b%(n)s\b\s*(?:[-+*/%%&|^]|<<|>>)?=(?!=)", r"(?:\+\+|--)\s*\b%(n)s\b", r"\b%(n)s\b\s*(?:\+\+|--)",
           r"&\s*%(n)s\b", r"\b%(n)s\b\s*\.\s*\w+\s*(?:[-+*/%%&|^]|<<|>>)?=(?!=)")
_UNSAFE_IN_REGION = (r"\bconst_cast\b", r"^\s*#\s*define\b")


def blank_strings(source):
    """String literals emptied: a log format's `planned_weight=%zu` is not a write to planned_weight."""
    return re.sub(r'"(?:[^"\\\n]|\\.)*"', '""', source)


def planned_slot_declaration(name, kind):
    return re.compile(r"\bconst\s+size_t\s+%s\s*=\s*ggml_sycl::unified_cache_get_planned_pp_moe_onednn_%s_slot_bytes"
                      r"\s*\(\s*ctx\s*\.\s*device\s*\)\s*;" % (name, kind))


def ring_raw_declaration(name):
    return re.compile(r"\bconst\s+uint32_t\s+%s\s*=\s*ggml_sycl::unified_cache_get_planned_pp_moe_onednn_ring_depth"
                      r"\s*\(\s*ctx\s*\.\s*device\s*\)\s*;" % name)


def ring_declaration(name, raw):
    return re.compile(r"\bconst\s+uint32_t\s+%s\s*=\s*pp_moe_onednn_runtime_ring_depth\s*\(\s*%s\s*\)\s*;" % (name, raw))


def shape_declaration(name, site):
    names = [slot for slot, _ in site["slots"]] + [site["raw"]]
    return re.compile(r"\bconst\s+ggml_sycl::pp_moe_onednn_scratch_shape\s+%s\s*=\s*\{\s*%s\s*,?\s*\}\s*;"
                      % (name, r"\s*,\s*".join(names)))


def pinned_once_and_never_written(region, name, declaration):
    """`name` is declared exactly once, by `declaration`, in this region, and never written after it."""
    region = blank_strings(region)
    if len(declaration.findall(region)) != 1:
        return False
    rest = declaration.sub("", region)
    if re.search(r"\b[\w:<>]+\s+%s\b" % name, rest):
        return False  # a second (shadowing) declaration of the same name
    return not any(re.search(pattern % {"n": name}, rest) for pattern in _WRITES)


def planned_slot_pinned(region, name, kind):
    return pinned_once_and_never_written(region, name, planned_slot_declaration(name, kind))


def region_is_free_of_escapes(region, pinned_names):
    """No const_cast (it writes a const name), no reinterpret_cast of a pinned name (the same through a type pun) and no
    #define (it redefines one) inside the executor body. reinterpret_cast of pointers is ordinary kernel code here."""
    region = blank_strings(region)
    if any(re.search(pattern, region, re.M) for pattern in _UNSAFE_IN_REGION):
        return False
    return not re.search(r"\breinterpret_cast\s*<[^>]*>\s*\(\s*[*&]?\s*(?:%s)\b" % "|".join(pinned_names), region)


def site_is_pinned(region, site):
    names = [slot for slot, _ in site["slots"]]
    pinned = names + [site["raw"], site["ring"], site["shape"]]
    return (region_is_free_of_escapes(region, pinned)
            and all(planned_slot_pinned(region, name, kind) for name, kind in site["slots"])
            and pinned_once_and_never_written(region, site["raw"], ring_raw_declaration(site["raw"]))
            and pinned_once_and_never_written(region, site["ring"], ring_declaration(site["ring"], site["raw"]))
            and pinned_once_and_never_written(region, site["shape"], shape_declaration(site["shape"], site))
            and len(names) == 3)


def top_level_arguments(text):
    out, depth, current = [], 0, []
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    out.append("".join(current).strip())
    return out


def reservation_passes_the_plan(region, site):
    """The executor's only reserve call passes the site's three planned names and its ring depth, verbatim, in order."""
    calls = [top_level_arguments(call) for call in reserve_call_arguments(region)]
    return calls == [[name for name, _ in site["slots"]] + [site["ring"]]]


def file_wide_call_counts_are_fixed(source):
    """The reserve / set-planned call counts of the whole file, and no use of the free reserve wrapper."""
    source = blank_strings(source)
    return (len(re.findall(r"(?<!unified_cache_)reserve_pp_moe_onednn_scratch\s*\(", source)) == RESERVE_CALLS_IN_FILE
            and len(re.findall(r"unified_cache_reserve_pp_moe_onednn_scratch\s*\(", source)) == 0
            and len(re.findall(r"unified_cache_set_planned_pp_moe_onednn_scratch\s*\(", source)) == SET_PLANNED_CALLS_IN_FILE)


def region_until_matching_endif(source, start):
    """From `start` to the #endif that closes the first #if after it, counting nested #if/#endif pairs. Ending at
    the first `#endif` after `start` would stop early at a nested one and hide everything after it."""
    at, depth = start, 0
    for line in source[start:].splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            directive = stripped[1:].lstrip()
            if directive.startswith("if"):
                depth += 1
            elif directive.startswith("endif"):
                depth -= 1
                if depth <= 0:
                    return source[start:at]
        at += len(line)
    return source[start:]


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


def latched_warn(region, latch_name):
    """True when the region reports its refusal at WARN, once, unconditionally.

    "Unconditional" is the whole point and it is the part a plain substring
    check cannot see: a WARN nested inside the debug-gated trace guard is
    invisible in a normal run, which is the state this check exists to end. So
    the check reads the WHOLE enclosing condition -- from the `if (` that opens
    the statement through to the warning -- and requires no `_trace` guard
    anywhere in it. Reading only forward from the report call would miss the
    obvious reintroduction, `if (trace && should_report(...))`, where the guard
    sits BEFORE the call it disables.

    The latch must also be a function-local `static`, or it is reconstructed on
    every dispatch and "once per process" silently becomes "every time".
    """
    latch_at = region.find("static ggml_sycl::pp_moe_onednn_scratch_refusal_latch " + latch_name)
    report_at = region.find("pp_moe_onednn_should_report_refusal(admission, " + latch_name + ")")
    warn_at = region.find("GGML_LOG_WARN(", report_at) if report_at >= 0 else -1
    if latch_at < 0 or report_at < latch_at or warn_at < 0:
        return False
    condition_at = region.rfind("if (", 0, report_at)
    if condition_at < 0:
        return False
    return "_trace" not in region[condition_at:warn_at]


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
    staging_start = sycl.find("scoped_unified_queue_temp<sycl::half> pp_mxfp4_src0_f16;")
    staging = region_until_matching_endif(sycl, staging_start) if staging_start >= 0 else ""
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
        # A refusal degrades routing silently, so it has to clear the DEFAULT
        # verbosity threshold -- GGML_LOG_INFO does not, and the debug-gated
        # traces beside these warnings do not either.
        "the batched executor warns unconditionally, once, on refusal":
            latched_warn(batched, "batched_scratch_refusal_latch"),
        "the non-batched staging warns unconditionally, once, on refusal":
            latched_warn(staging, "staging_scratch_refusal_latch"),
        "each refusal warning carries the reason and the requested-vs-cap bytes":
            all(ordered(region, "GGML_LOG_WARN(", "reported once per process",
                        "required=[%zu,%zu,%zu] planned_cap=[%zu,%zu,%zu]",
                        "pp_moe_onednn_scratch_admission_reason_name(admission.reason)")
                for region in (batched, staging)),
        # ABSENCE: the bug this task exists for. max(planned, required) turns a
        # budgeted zone into a high-water mark of every shape ever seen.
        # The reservation's own arguments may not carry a max ...
        "no PP MoE scratch reservation upsizes past the plan":
            reservation_passes_the_plan(batched, PLANNED_SITES[0]) and reservation_passes_the_plan(staging, PLANNED_SITES[1])
            and file_wide_call_counts_are_fixed(sycl),
        # ... and what it passes is the planner's own values, read once and never written (see PLANNED_SITES).
        "the planned slot sizes are read once, const, and never reassigned":
            site_is_pinned(batched, PLANNED_SITES[0]) and site_is_pinned(staging, PLANNED_SITES[1]),
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
        "the contract requires the refusal to be visible at default verbosity":
            ordered(doc, "reports it at `GGML_LOG_WARN`", "latched to fire once per process"),
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
    # The regression this guards is the state the code was IN before the review
    # ruling: the refusal was reported only under a debug flag, so a production
    # run that lost its fast path said nothing. The mutant tucks the warning
    # back inside the trace guard, which is the natural way to reintroduce it.
    "the batched executor warns unconditionally, once, on refusal": (
        "sycl",
        "if (ggml_sycl::pp_moe_onednn_should_report_refusal(admission, batched_scratch_refusal_latch)) {",
        "if (pp_mxfp4_soa_f16_batched_trace &&\n"
        " ggml_sycl::pp_moe_onednn_should_report_refusal(admission, batched_scratch_refusal_latch)) {"),
    # A latch that is not `static` is reconstructed on every dispatch, so
    # "once per process" silently becomes "every time" -- the spam this exists
    # to bound, and invisible to any check that only greps for the warning.
    "the non-batched staging warns unconditionally, once, on refusal": (
        "sycl",
        "static ggml_sycl::pp_moe_onednn_scratch_refusal_latch staging_scratch_refusal_latch;",
        "ggml_sycl::pp_moe_onednn_scratch_refusal_latch staging_scratch_refusal_latch;"),
}


# Ways to grow the planned slot size before the reservation, each of which must trip the positive pin. Every one is
# applied twice: with the declaration's `const` dropped (so it would compile) and with `const` kept (the write
# check, not the compiler, has to refuse it). The last two never write the planned name at all: the grown value is
# passed to the reservation under another name.
PLANNED_W = "ggml_sycl::unified_cache_get_planned_pp_moe_onednn_weight_slot_bytes(ctx.device);"
PLANNED_REWRITES = (
    ("if-assign", "if (weight_bytes > planned_weight) { planned_weight = weight_bytes; }"),
    ("paren-ternary", "planned_weight = (weight_bytes > planned_weight) ? weight_bytes : planned_weight;"),
    ("GGML_MAX", "planned_weight = GGML_MAX(planned_weight, weight_bytes);"),
    ("sycl-max", "planned_weight = sycl::max(planned_weight, weight_bytes);"),
    ("using-std-max", "using std::max; planned_weight = max(planned_weight, weight_bytes);"),
    ("alias-then-max", "{ const size_t w0 = planned_weight; planned_weight = std::max(w0, weight_bytes); }"),
    ("compound-assign", "planned_weight += weight_bytes;"),
    ("increment", "++planned_weight;"),
    ("address-taken", "grow_in_place(&planned_weight, weight_bytes);"),
    ("shadowing-redeclaration", "{ size_t planned_weight = weight_bytes; (void) planned_weight; }"),
    ("shadowing-brace-init", "{ size_t planned_weight{weight_bytes}; (void) planned_weight; }"),
)


# Growth that escapes the located executor bodies or the located statements. Each edit is (anchor, replacement) on the
# whitespace-squeezed source; ANY failed check counts as the gate catching it.
_CALL_BATCHED = "            if (!cache->reserve_pp_moe_onednn_scratch(planned_weight, planned_act, planned_out, ring_depth)) {"
_CALL_STAGING = ("                (void) cache->reserve_pp_moe_onednn_scratch(planned_weight_slot, planned_activation_slot,\n"
                 "                                                            planned_output_slot, planned_ring_depth);")
_ELSEWHERE = "static const char * ggml_sycl_planned_weight_residency_name("
_SHAPE = "            const ggml_sycl::pp_moe_onednn_scratch_shape planned_shape = {\n                planned_weight,\n                planned_act,"
_RING = "            const uint32_t ring_depth = pp_moe_onednn_runtime_ring_depth(planned_ring_depth);"
_DECL_W = "            const size_t planned_weight =\n                " + PLANNED_W
ESCAPES = (
    ("third-site-std-max-elsewhere", [(_ELSEWHERE, "static bool third_site(ggml_sycl::unified_cache * c, size_t pw, size_t wb, "
      "size_t pa, size_t po, uint32_t rd) {\n    return c->reserve_pp_moe_onednn_scratch(std::max(pw, wb), pa, po, rd);\n}\n" + _ELSEWHERE)]),
    ("third-site-free-wrapper-elsewhere", [(_ELSEWHERE, "static bool third_site(int dev, size_t pw, size_t wb, size_t pa, size_t po, "
      "uint32_t rd) {\n    return ggml_sycl::unified_cache_reserve_pp_moe_onednn_scratch(dev, std::max(pw, wb), pa, po, rd);\n}\n" + _ELSEWHERE)]),
    ("second-reserve-free-wrapper-in-batched", [(_CALL_BATCHED, "            (void) ggml_sycl::unified_cache_reserve_pp_moe_onednn_scratch(ctx.device, "
      "std::max(planned_weight, weight_bytes), planned_act, planned_out, ring_depth);\n" + _CALL_BATCHED)]),
    ("second-reserve-dot-form-in-batched", [(_CALL_BATCHED, "            (void) (*cache).reserve_pp_moe_onednn_scratch(std::max(planned_weight, "
      "weight_bytes), planned_act, planned_out, ring_depth);\n" + _CALL_BATCHED)]),
    ("second-reserve-arrow-in-batched", [(_CALL_BATCHED, "            (void) cache->reserve_pp_moe_onednn_scratch(std::max(planned_weight, "
      "weight_bytes), planned_act, planned_out, ring_depth);\n" + _CALL_BATCHED)]),
    ("second-reserve-arrow-in-staging", [(_CALL_STAGING, _CALL_STAGING + "\n                (void) cache->reserve_pp_moe_onednn_scratch("
      "std::max(planned_weight_slot, pp_mxfp4_src0_f16_bytes), planned_activation_slot, planned_output_slot, planned_ring_depth);")]),
    ("staging-args-swapped", [(_CALL_STAGING, _CALL_STAGING.replace("planned_weight_slot, planned_activation_slot,", "planned_activation_slot, planned_weight_slot,"))]),
    ("staging-nested-endif-truncates-the-region", [(_CALL_STAGING, _CALL_STAGING + "\n#if 1\n#endif\n                (void) cache->"
      "reserve_pp_moe_onednn_scratch(std::max(planned_weight_slot, pp_mxfp4_src0_f16_bytes), planned_activation_slot, "
      "planned_output_slot, planned_ring_depth);")]),
    ("staging-nested-endif-then-a-write", [(_CALL_STAGING, _CALL_STAGING + "\n#if 1\n#endif\n                "
      "const_cast<size_t &>(planned_weight_slot) += pp_mxfp4_src0_f16_bytes;")]),
    ("macro-alias-then-write", [(_CALL_BATCHED, "#define GROW(a) ((a) += weight_bytes)\n            GROW(planned_weight);\n" + _CALL_BATCHED)]),
    ("raw-ring-depth-written", [("            const uint32_t planned_ring_depth =\n                ggml_sycl::unified_cache_get_planned_pp_moe_onednn_ring_depth(ctx.device);",
      "            uint32_t planned_ring_depth =\n                ggml_sycl::unified_cache_get_planned_pp_moe_onednn_ring_depth(ctx.device);"),
      (_CALL_BATCHED, "            planned_ring_depth = 8;\n" + _CALL_BATCHED)]),
    ("admission-shape-member-written", [(_CALL_BATCHED, "            planned_shape.weight_slot_bytes = std::max(planned_shape.weight_slot_bytes, weight_bytes);\n" + _CALL_BATCHED)]),
    ("const-cast-write", [(_CALL_BATCHED, "            const_cast<size_t &>(planned_weight) = std::max(planned_weight, weight_bytes);\n" + _CALL_BATCHED)]),
    ("const-cast-pointer-write", [(_CALL_BATCHED, "            *const_cast<size_t *>(&planned_weight) = std::max(planned_weight, weight_bytes);\n" + _CALL_BATCHED)]),
    ("const-cast-reference-bind", [(_CALL_BATCHED, "            size_t & grow_w = const_cast<size_t &>(planned_weight);\n            grow_w = "
      "std::max(grow_w, weight_bytes);\n" + _CALL_BATCHED)]),
    ("reinterpret-cast-write", [(_CALL_BATCHED, "            reinterpret_cast<size_t &>(planned_weight) = std::max(planned_weight, weight_bytes);\n" + _CALL_BATCHED)]),
    ("const-ref-alias-then-const-cast", [(_CALL_BATCHED, "            const size_t & pw_ref = planned_weight;\n            const_cast<size_t &>(pw_ref) += weight_bytes;\n" + _CALL_BATCHED)]),
    ("macro-redefines-the-planned-name", [(_CALL_BATCHED, "#define planned_weight (planned_weight + 1)\n" + _CALL_BATCHED)]),
    ("grown-value-in-the-admission-shape", [(_SHAPE, "            const ggml_sycl::pp_moe_onednn_scratch_shape planned_shape = {\n                "
      "std::max(planned_weight, weight_bytes),\n                planned_act,")]),
    ("ring-depth-grown", [(_RING, "            const uint32_t ring_depth = std::max(pp_moe_onednn_runtime_ring_depth(planned_ring_depth), 4u);")]),
    ("ceiling-raised-before-the-read", [(_DECL_W, "            ggml_sycl::unified_cache_set_planned_pp_moe_onednn_scratch(ctx.device, std::max(weight_bytes, size_t(1)), "
      "act_bytes, out_bytes, 1);\n" + _DECL_W)]),
    # The whitelist refuses every other use of a pinned name, so none of these needs its own spelling banned.
    ("c-style-reference-bind", [(_CALL_BATCHED, "            size_t & g = (size_t &) planned_weight;\n            g = std::max(g, weight_bytes);\n" + _CALL_BATCHED)]),
    ("c-style-pointer-write-through-addressof", [(_CALL_BATCHED, "            *(size_t *) std::addressof(planned_weight) = std::max(planned_weight, weight_bytes);\n" + _CALL_BATCHED)]),
    ("memcpy-through-addressof", [(_CALL_BATCHED, "            { size_t g = std::max(planned_weight, weight_bytes); std::memcpy((void *) std::addressof(planned_weight), &g, sizeof(g)); }\n" + _CALL_BATCHED)]),
    ("decltype-shadow", [(_CALL_BATCHED, "            decltype(weight_bytes) planned_weight = std::max(weight_bytes, size_t(1));\n" + _CALL_BATCHED)]),
    ("structured-binding-shadow", [(_CALL_BATCHED, "            auto [planned_weight, junk_w] = std::pair<size_t, int>(std::max(weight_bytes, size_t(1)), 0);\n" + _CALL_BATCHED)]),
    ("lambda-parameter-shadows-the-ring-depth", [(_CALL_BATCHED, "            auto f = [&](uint32_t ring_depth) { return cache->reserve_pp_moe_onednn_scratch(planned_weight, planned_act, planned_out, ring_depth); };\n            if (!f(std::max(ring_depth, 4u))) {")]),
    ("admission-shape-written-through-a-pointer", [(_CALL_BATCHED, "            (&planned_shape)->weight_slot_bytes = std::max(planned_shape.weight_slot_bytes, weight_bytes);\n" + _CALL_BATCHED)]),
    ("planned-name-passed-to-a-by-reference-helper", [(_CALL_BATCHED, "            grow_in_place(planned_weight, weight_bytes);\n" + _CALL_BATCHED)]),
    ("setter-through-a-function-pointer", [(_DECL_W, "            auto setter = &ggml_sycl::unified_cache_set_planned_pp_moe_onednn_scratch;\n            setter(ctx.device, std::max(weight_bytes, size_t(1)), act_bytes, out_bytes, 1);\n" + _DECL_W)]),
    ("reserve-through-a-member-pointer", [(_CALL_BATCHED, "            auto rsv = &ggml_sycl::unified_cache::reserve_pp_moe_onednn_scratch;\n            (void) (cache->*rsv)(std::max(planned_weight, weight_bytes), planned_act, planned_out, ring_depth);\n" + _CALL_BATCHED)]),
    ("reserve-token-pasted-in-a-helper", [(_ELSEWHERE, "static bool third(ggml_sycl::unified_cache * c, size_t a, size_t b, size_t d, uint32_t r) { return c->reserve_pp_moe_onednn_ ## scratch(a, b, d, r); }\n" + _ELSEWHERE)]),
    ("reserve-token-pasted-in-a-macro", [(_ELSEWHERE, "#define RSV(c, ...) (c)->reserve_pp_moe_onednn_##scratch(__VA_ARGS__)\n" + _ELSEWHERE)]),
    ("reserve-with-a-line-break-before-the-paren", [(_ELSEWHERE, "static bool third(ggml_sycl::unified_cache * c, size_t a, size_t b, size_t d, uint32_t r) { return c->reserve_pp_moe_onednn_scratch\n(a, b, d, r); }\n" + _ELSEWHERE)]),
    ("helper-lambda-around-the-reserve", [(_CALL_BATCHED, "            auto do_reserve = [&](size_t w, size_t a, size_t o, uint32_t d) { return cache->reserve_pp_moe_onednn_"
      "scratch(w, a, o, d); };\n            if (!do_reserve(std::max(planned_weight, weight_bytes), planned_act, planned_out, ring_depth)) {")]),
)


def self_test(sycl, cache, module, header, doc):
    """Every absence check must fail once its forbidden construct is injected."""
    problems = []
    call_site = squeeze("            if (!cache->reserve_pp_moe_onednn_scratch(planned_weight, planned_act, planned_out, ring_depth)) {")
    check = "the planned slot sizes are read once, const, and never reassigned"
    declaration = squeeze("            const size_t planned_weight =\n                " + PLANNED_W)
    for label, statement in PLANNED_REWRITES:
        for unconst in (True, False):
            base = squeeze(sycl)
            if call_site not in base or declaration not in base:
                problems.append("mutation anchor missing for the planned-slot rewrites")
                break
            if unconst:
                base = base.replace(declaration, declaration.replace("const size_t", "size_t", 1), 1)
            mutated = base.replace(call_site, squeeze("            " + statement + "\n" + call_site), 1)
            _, failed, _ = evaluate(mutated, cache, module, header, doc)
            if check not in failed:
                problems.append("planned-slot pin did not fire on: %s (%s)" % (label, "const dropped" if unconst else "const kept"))
    # Dropping `const` alone writes nothing, but the declaration is no longer the pinned read-only one.
    base = squeeze(sycl)
    _, failed, _ = evaluate(base.replace(declaration, declaration.replace("const size_t", "size_t", 1), 1), cache, module,
                            header, doc)
    if check not in failed:
        problems.append("planned-slot pin did not fire on: const dropped, nothing written")
    # Computed under another name and passed to the reservation: the planned names are untouched, the call is not.
    for label, hoist in (("std-max-under-another-name", "const size_t grown_w = std::max(planned_weight, weight_bytes);"),
                         ("alias-then-std-max", "const size_t cap_w = planned_weight; const size_t grown_w = std::max(cap_w, weight_bytes);")):
        mutated = squeeze(sycl).replace(
            call_site, squeeze("            " + hoist + "\n" + call_site.replace("(planned_weight,", "(grown_w,")), 1)
        _, failed, _ = evaluate(mutated, cache, module, header, doc)
        if "no PP MoE scratch reservation upsizes past the plan" not in failed:
            problems.append("reservation pin did not fire on: " + label)
    nested = "a\n#if X\nb\n#if 1\n#endif\nc\n#endif\nd\n"
    if region_until_matching_endif(nested, 0) != "a\n#if X\nb\n#if 1\n#endif\nc\n":
        problems.append("region_until_matching_endif stops at a nested #endif")
    for label, edits in ESCAPES:
        mutated = squeeze(sycl)
        if not all(squeeze(anchor) in mutated for anchor, _ in edits):
            problems.append("mutation anchor missing for the escape: " + label)
            continue
        for anchor, replacement in edits:
            mutated = mutated.replace(squeeze(anchor), squeeze(replacement), 1)
        _, failed, _ = evaluate(mutated, cache, module, header, doc)
        if not failed:
            problems.append("no check fired on the escape: " + label)
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
