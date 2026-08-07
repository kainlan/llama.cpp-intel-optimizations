#!/usr/bin/env python3
"""Per-model tiering verdict source contract (llama.cpp-wmc2, canonical §12.7).

Host-only: reads sources, runs no build, loads no model and touches no device.

The executable test (test-tiered-dispatch) proves the verdict against three
inventories on a real device. It cannot see WHERE the answer came from, and
that is the whole defect this gate exists for: `g_tiered_enabled` is a
process-level "the unified cache is the pointer authority" flag that has been
unconditionally true for every SYCL model load, so any per-model question
routed through it can only ever answer "yes" -- and it answers plausibly,
which is why seven weeks passed with no failure signal.

So this gate pins the three things that keep the two questions apart:

  1. the per-model verdict is DERIVED on read from the identity snapshot,
     never latched into a process-global boolean (§12.7 names the latch
     non-compliant by construction);
  2. a STAGED candidate carries the same planned-host bytes its publication
     will, so the mid-load answer is this model's and not the previous one's;
  3. test-tiered-dispatch registers its inventory through the load bracket,
     because a direct set_tensor_inventory() call stages no plan and makes
     every planner-target assertion in it vacuous.

Check 3 is the one worth stating plainly: the test was passing its own
oracle nothing to read. Every check below fails against d529d889f. Run it
that way to confirm:

    ./tests/test-sycl-tiered-verdict-contract.py \\
        --backend <(git show d529d889f:ggml/src/ggml-sycl/ggml-sycl.cpp) \\
        --cache <(git show d529d889f:ggml/src/ggml-sycl/unified-cache.cpp) \\
        --test <(git show d529d889f:tests/test-tiered-dispatch.cpp)

The `anchors` section proves every region the checks key off was found, so a
renamed symbol reports a missing anchor instead of passing vacuously. Run with
--self-test to prove the absence-based checks fire against mutants.
"""
import argparse
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default=str(root / "ggml/src/ggml-sycl/ggml-sycl.cpp"))
parser.add_argument("--cache", default=str(root / "ggml/src/ggml-sycl/unified-cache.cpp"))
parser.add_argument("--test", default=str(root / "tests/test-tiered-dispatch.cpp"))
parser.add_argument("--self-test", action="store_true",
                    help="prove the absence-based checks fire when the thing they forbid is present")
args = parser.parse_args()


def strip_comments(source):
    """Remove C and C++ comments, preserving string literals and line count.

    Every check below reads active code only. Without this, a check like "no
    latched process-global verdict" is spoofed by a COMMENT naming the symbol
    it forbids -- and this file's own fix leaves exactly such a comment behind,
    explaining why the latch is gone.
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

    Every needle below is written with single spaces so clang-format's column
    alignment cannot break a check -- this file's declarations sit in a block
    that is vertically aligned to its widest member.
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


def evaluate(backend, cache, test):
    backend, cache, test = squeeze(backend), squeeze(cache), squeeze(test)
    derive = body_of(backend, "static bool ggml_sycl_current_model_planner_host_placement(")
    api = body_of(backend, "bool ggml_backend_sycl_is_tiered_enabled(")
    target = body_of(backend, "int ggml_backend_sycl_planned_target_device(")
    publish = body_of(backend, "static void ggml_sycl_publish_prepared_plan_locked(")
    stage = body_of(cache, "void lifecycle_stage_placement_plan(")
    case = body_of(test, "void run_placement_case(")

    # Anchors: every region the checks read must have been found. A rename that
    # empties one of these reports a missing anchor rather than a silent pass.
    anchors = {
        "ggml_sycl_current_model_planner_host_placement body": derive,
        "ggml_backend_sycl_is_tiered_enabled body": api,
        "ggml_backend_sycl_planned_target_device body": target,
        "ggml_sycl_publish_prepared_plan_locked body": publish,
        "lifecycle_stage_placement_plan body": stage,
        "run_placement_case body": case,
    }

    checks = {
        # --- 1. the verdict is derived, not latched ---
        "the per-model verdict is derived from the identity snapshot":
            ordered(derive, "ggml_sycl_identity_plan_snapshot()",
                    "snapshot->plan", "snapshot->planned_host_bytes > 0"),
        "the public tiering query answers from that derivation":
            "ggml_sycl_current_model_planner_host_placement()" in api,
        # ABSENCE: canonical §12.7 -- a process-global "current model has host
        # placement" boolean is not a compliant authority, because it inherits
        # the previous model's answer for the whole of the next model's load.
        "no latched process-global host-placement verdict survives":
            "g_current_model_planner_host_placement" not in backend,
        # ABSENCE: the conflation itself. g_tiered_enabled is unconditionally
        # true, so routing the per-model question through it answers "yes" for
        # every model ever loaded.
        "the per-model verdict is not routed through the process cache gate":
            "g_tiered_enabled" not in derive and "g_tiered_enabled" not in api,
        # The publication path may still publish the plan; it must not also
        # keep a second copy of the verdict beside it.
        "publication stores the plan snapshot and nothing derived from it":
            "g_placement_publication" in publish and "planned_host_bytes" not in publish,

        # --- 2. the verdict and the target query share one authority ---
        # Both must resolve through the bound-candidate-first authority, or the
        # two APIs disagree for the whole load window: one reads this model,
        # the other the last one committed.
        "the planned-target query resolves through the same plan authority":
            "ggml_sycl_global_plan_owner()" in target,
        "a staged candidate carries the planned-host bytes its publication will":
            "snapshot->planned_host_bytes = snapshot->plan->weight_host_bytes;" in stage,
        "publication derives planned-host bytes from the same plan field":
            "planned_host_bytes = published->plan->weight_host_bytes;" in cache,

        # --- 3. the executable test is non-vacuous ---
        "the executable test registers inventory through the load bracket":
            "ggml_backend_sycl_stage_inventory_plan(&fixture.inventory" in case,
        # ABSENCE: a direct call stages no plan onto the load-effect
        # transaction, so every planned-target assertion downstream reads
        # NO_PLAN and asserts against nothing.
        "the executable test does not stage inventory outside the bracket":
            "ggml_backend_sycl_set_tensor_inventory(" not in test,
        "the executable test asserts the verdict mid-load, before commit":
            ordered(case, "ggml_backend_sycl_stage_inventory_plan(",
                    "ggml_backend_sycl_is_tiered_enabled(backend)",
                    "ggml_backend_sycl_model_load_end("),
        "the executable test cross-checks the verdict against a target oracle":
            "planner_target_oracle(fixture, label)" in case,
    }

    return sorted(name for name, text in anchors.items() if not text), \
           sorted(name for name, ok in checks.items() if not ok), len(checks)


# Positive controls for the absence-based checks. They pass vacuously on any
# tree that never had the forbidden construct, so each one is re-evaluated
# against a source that deliberately contains it.
ABSENCE_MUTANTS = {
    # The exact pre-wmc2 shape: a process-global latch written at publication
    # and read by the API.
    "no latched process-global host-placement verdict survives": (
        "backend",
        "static std::atomic<bool>                                         g_fail_next_plan_publication_prepare{ false };",
        "static std::atomic<bool>                                         g_current_model_planner_host_placement{ false };\n"
        "static std::atomic<bool>                                         g_fail_next_plan_publication_prepare{ false };"),
    # The conflation this task closes: answer the per-model question with the
    # always-true process gate.
    "the per-model verdict is not routed through the process cache gate": (
        "backend",
        "    return ggml_sycl_current_model_planner_host_placement();\n}",
        "    return g_tiered_enabled.load(std::memory_order_relaxed);\n}"),
    # A second copy of the verdict beside the publication is how the latch
    # grew in the first place.
    "publication stores the plan snapshot and nothing derived from it": (
        "backend",
        "        [&] { std::atomic_store_explicit(&g_placement_publication, snapshot, std::memory_order_release); });",
        "        [&] {\n"
        "            const bool host = snapshot && snapshot->plan && snapshot->planned_host_bytes > 0;\n"
        "            (void) host;\n"
        "            std::atomic_store_explicit(&g_placement_publication, snapshot, std::memory_order_release);\n"
        "        });"),
    # The vacuous-test shape: register the inventory directly, so nothing is
    # staged and the oracle reads NO_PLAN for every tensor.
    "the executable test does not stage inventory outside the bracket": (
        "test",
        "    const bool oracle  = planner_target_oracle(fixture, label);",
        "    ggml_backend_sycl_set_tensor_inventory(backend, &fixture.inventory);\n"
        "    const bool oracle  = planner_target_oracle(fixture, label);"),
}


def self_test(backend, cache, test):
    """Every absence check must fail once its forbidden construct is injected.

    Sources are handed over unsqueezed: evaluate() squeezes for itself, and
    squeezing here too would leave every anchor written against the real
    (vertically aligned) source text unable to match.
    """
    problems = []
    for name, (target, anchor, replacement) in ABSENCE_MUTANTS.items():
        sources = {"backend": backend, "cache": cache, "test": test}
        if anchor not in sources[target]:
            problems.append("mutation anchor missing for: " + name)
            continue
        sources[target] = sources[target].replace(anchor, replacement, 1)
        _, failed, _ = evaluate(sources["backend"], sources["cache"], sources["test"])
        if name not in failed:
            problems.append("check did not fire on its mutant: " + name)
    return problems


backend = strip_comments(Path(args.backend).read_text())
cache = strip_comments(Path(args.cache).read_text())
test = strip_comments(Path(args.test).read_text())

missing_anchors, failed, n_checks = evaluate(backend, cache, test)

for name in missing_anchors:
    print("MISSING ANCHOR: " + name)
for name in failed:
    print("FAIL: " + name)

if missing_anchors or failed:
    sys.exit(1)

if args.self_test:
    problems = self_test(backend, cache, test)
    for problem in problems:
        print("SELF-TEST FAIL: " + problem)
    if problems:
        sys.exit(1)
    print("sycl tiered verdict contract: self-test PASS ({} mutants)".format(len(ABSENCE_MUTANTS)))

print("sycl tiered verdict contract: PASS ({} checks)".format(n_checks))
