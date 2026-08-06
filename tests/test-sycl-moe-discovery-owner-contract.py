#!/usr/bin/env python3
"""Owner-keyed MoE discovery state contract (llama.cpp-nn6z, canonical §12).

Host-only: reads sources, runs no build, loads no model and touches no device.

The companion executable test (test-moe-discovery-owner-state) proves the
registry's ownership rules against a stand-in live state. It cannot see whether
the real cluster is wired to that registry at all, which is what this gate
covers: the lifecycle hooks, their lock placement, and the absence of the
load-boundary global clear the k7b0 ruling rejects.

Every check below fails against the pre-nn6z tree. Run it that way to confirm:

    ./tests/test-sycl-moe-discovery-owner-contract.py \\
        --backend <(git show 606e252b0:ggml/src/ggml-sycl/ggml-sycl.cpp) \\
        --registry ggml/src/ggml-sycl/moe-discovery-state.hpp

The `positive controls` section proves every anchor the checks key off exists,
so a renamed symbol reports a missing anchor instead of passing vacuously.
"""
import argparse
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default=str(root / "ggml/src/ggml-sycl/ggml-sycl.cpp"))
parser.add_argument("--registry", default=str(root / "ggml/src/ggml-sycl/moe-discovery-state.hpp"))
parser.add_argument("--self-test", action="store_true",
                    help="prove the absence-based checks fire when the thing they forbid is present")
args = parser.parse_args()


def body_of(source, signature_prefix):
    """Text from a definition's signature to its first column-0 closing brace.

    `[^;]*?` before the opening brace is what skips a forward declaration: the
    declaration's terminating `;` cannot be crossed, so the match lands on the
    definition even when the prototype appears first.
    """
    match = re.search(re.escape(signature_prefix) + r"[^;]*?\)\s*(?:noexcept\s*)?(?:override\s*)?(?:try\s*)?\{.*?^\}\n",
                      source, re.S | re.M)
    return match.group(0) if match else ""


def member_body_of(source, signature_prefix):
    """Same, for a member function indented four spaces inside a class.

    `[^;{]*?` spans the qualifiers between the prefix and the body without
    crossing a `;`, so a pure declaration cannot be mistaken for a definition.
    """
    match = re.search(re.escape(signature_prefix) + r"[^;{]*?\{.*?^    \}\n", source, re.S | re.M)
    return match.group(0) if match else ""


def locked_regions(text):
    """Every span from a `lock_guard(mutex_)` to the end of its enclosing block.

    Returns an empty list when the guard is absent, so a rewrite that drops the
    lock reports a failure instead of an empty-region pass.
    """
    regions = []
    marker = "std::lock_guard<std::mutex> lock(mutex_);"
    at = text.find(marker)
    while at >= 0:
        end = text.find("\n        }", at)
        regions.append(text[at:end if end >= 0 else len(text)])
        at = text.find(marker, at + 1)
    return regions


def ordered(text, *needles):
    """True when every needle appears, in the given order."""
    at = -1
    for needle in needles:
        found = text.find(needle, at + 1)
        if found < 0:
            return False
        at = found
    return True


def evaluate(backend, registry):
    live_park       = member_body_of(backend, "    std::unique_ptr<ggml_sycl::moe_discovery_parked_state> park(")
    live_restore    = member_body_of(backend, "    void restore(std::unique_ptr<ggml_sycl::moe_discovery_parked_state> parked)")
    live_fresh      = member_body_of(backend, "    void install_fresh(")
    owner_key_impl  = body_of(backend, "static ggml_sycl::moe_owner_key ggml_sycl_moe_owner_key(")
    report_impl     = body_of(backend, "static void ggml_sycl_moe_discovery_report(")
    seed_impl       = body_of(backend, 'extern "C" void ggml_backend_sycl_test_seed_moe_module_state(')
    seed_helper     = body_of(backend, "static void ggml_sycl_moe_discovery_seed_owner(")
    teardown_impl   = body_of(backend, "static bool ggml_sycl_teardown_owner_effects(")
    load_end_impl   = body_of(backend, "ggml_sycl_lifecycle_result ggml_backend_sycl_model_load_end(")
    activate_impl   = body_of(backend, "ggml_sycl_lifecycle_result ggml_backend_sycl_activate_model_plan(")
    module_reset    = body_of(backend, "static void ggml_sycl_reset_moe_module_state(")
    registry_activate = re.search(r"    moe_discovery_result activate\(.*?^    \}\n", registry, re.S | re.M)
    registry_release  = re.search(r"    moe_discovery_result release\(.*?^    \}\n", registry, re.S | re.M)
    registry_activate = registry_activate.group(0) if registry_activate else ""
    registry_release  = registry_release.group(0) if registry_release else ""

    # Positive controls: an empty body means the anchor moved or was renamed, which
    # would make every check over it pass on absence.
    anchors = {
        "moe-discovery-state.hpp contents": registry,
        "moe_discovery_registry::activate body": registry_activate,
        "moe_discovery_registry::release body": registry_release,
        "live discovery park() body": live_park,
        "live discovery restore() body": live_restore,
        "live discovery install_fresh() body": live_fresh,
        "ggml_sycl_moe_owner_key body": owner_key_impl,
        "ggml_sycl_moe_discovery_report body": report_impl,
        "ggml_backend_sycl_test_seed_moe_module_state body": seed_impl,
        "ggml_sycl_moe_discovery_seed_owner body": seed_helper,
        "ggml_sycl_teardown_owner_effects body": teardown_impl,
        "ggml_backend_sycl_model_load_end body": load_end_impl,
        "ggml_backend_sycl_activate_model_plan body": activate_impl,
        "ggml_sycl_reset_moe_module_state body": module_reset,
    }
    missing_anchors = sorted(name for name, text in anchors.items() if not text)

    checks = {
        # The registry exists and is reachable from the backend.
        "the backend includes the owner-keyed discovery registry":
            '#include "ggml-sycl/moe-discovery-state.hpp"' in backend,
        "the backend owns exactly one live discovery working set":
            backend.count("static ggml_sycl_moe_live_discovery ") == 1
            and backend.count("static ggml_sycl::moe_discovery_registry ") == 1,

        # Owner identity is the full token, so a reused slot is a different owner.
        "the owner key carries the full model token":
            all(field in owner_key_impl for field in
                ("token.model.value", "token.load.value", "token.owner.slot", "token.owner.generation")),
        "an unattributed or none-slot key cannot become an owner":
            "key.model_id != 0" in registry and "key.slot != moe_owner_slot_none" in registry
            and "key.slot_generation != 0" in registry,
        "owner equality compares the generation, not just the slot":
            "a.slot_generation == b.slot_generation" in registry,

        # The repair: an owner switch makes the activating model re-discover.
        "installing a fresh working set clears the per-device hybrid-init guards":
            "g_moe_hybrid_init_success[d].store(false" in live_fresh
            and "g_moe_hybrid_init_done.store(false" in live_fresh,
        "installing a fresh working set drops the previous owner's expert registries":
            "g_moe_expert_meta.clear();" in live_fresh and "g_expert_groups.clear();" in live_fresh
            and "g_moe_layer_seq[d].clear();" in live_fresh,
        "installing a fresh working set drops the previous owner's popularity ranks":
            "g_expert_popularity.clear();" in live_fresh
            and "g_expert_popularity_initialized = false;" in live_fresh,
        "a restored owner still re-discovers its device-bound half":
            ordered(live_restore, "install_fresh();", "state->popularity"),
        "parking failure is reported as null rather than a partial snapshot":
            ordered(live_park, "catch (...)", "return nullptr;"),

        # The three lifecycle hooks, and where they sit relative to the locks.
        "a committed load takes MoE discovery ownership":
            ordered(load_end_impl, "if (result.committed) {",
                    "ggml_sycl_moe_owner_key(result.token)",
                    "moe_discovery_activate_retrying(g_moe_discovery_registry, moe_owner)"),
        "explicit activation hands the owner back its own discovery":
            "ggml_sycl_moe_owner_key(token)" in activate_impl
            and "moe_discovery_activate_retrying(g_moe_discovery_registry, moe_owner)" in activate_impl,
        "explicit activation transitions outside the publication lock":
            ordered(activate_impl, "std::lock_guard<std::mutex> lock(g_tensor_inventory_mutex);",
                    "ggml_sycl_publish_plan_locked(snapshot);", "}",
                    "moe_discovery_activate_retrying("),
        "model teardown releases exactly one owner, before its slot resources go":
            ordered(teardown_impl, "ggml_sycl_moe_owner_key(owner)",
                    "moe_discovery_release_retrying(g_moe_discovery_registry, moe_owner)",
                    "ggml_sycl_release_model_slot_resources(owner);"),
        "teardown releases by owner rather than clearing the cluster":
            "moe_discovery_release_retrying(g_moe_discovery_registry" in teardown_impl
            and "ggml_sycl_reset_moe_module_state" not in teardown_impl,

        # No isolated global clears on the load boundary (the k7b0 ruling).
        "the load boundary never clears the MoE cluster globally":
            "ggml_sycl_reset_moe_module_state" not in load_end_impl
            and "g_moe_discovery_registry.release_all_for_module_shutdown" not in load_end_impl,
        "the all-owner sweep exists only in module shutdown":
            backend.count("release_all_for_module_shutdown()") == 1
            and "g_moe_discovery_registry.release_all_for_module_shutdown();" in module_reset,

        # Registry transitions cannot run under the registry's own mutex, which is
        # what keeps allocation and destruction out of a locked region.
        "the live working set is never touched with the registry lock held":
            bool(locked_regions(registry_activate))
            and all(call not in region
                    for region in locked_regions(registry_activate)
                    for call in ("live_.park()", "live_.restore(", "live_.install_fresh()")),
        "a re-entrant transition is refused instead of interleaved":
            "MOE_DISCOVERY_RESULT_BUSY" in registry_activate and "MOE_DISCOVERY_RESULT_BUSY" in registry_release,
        "releasing a token that owns nothing is reported, not silently swept":
            "MOE_DISCOVERY_RESULT_NOT_FOUND" in registry_release,

        # A transition that did not take effect must be actionable, not discarded.
        "no integration site discards the transition result":
            all("(void) g_moe_discovery_registry." not in body
                for body in (teardown_impl, load_end_impl, activate_impl)),
        "the module-reload seed drives the registry through the same helper":
            "ggml_sycl_moe_discovery_seed_owner(" in seed_impl
            and "moe_discovery_activate_retrying(" in seed_helper
            and "ggml_sycl_moe_discovery_report(" in seed_helper,
        "every integration site retries a BUSY transition before giving up":
            "moe_discovery_release_retrying(" in teardown_impl
            and "moe_discovery_activate_retrying(" in load_end_impl
            and "moe_discovery_activate_retrying(" in activate_impl,
        "failed transitions are reported at WARN, which survives default verbosity":
            "GGML_LOG_WARN(" in report_impl and "GGML_LOG_INFO(" not in report_impl,
        "every integration site reports a transition that did not take effect":
            all("ggml_sycl_moe_discovery_report(" in body
                for body in (teardown_impl, load_end_impl, activate_impl)),
        "model teardown fails closed instead of releasing slot resources anyway":
            ordered(teardown_impl, "moe_discovery_release_retrying(",
                    "ggml_sycl_moe_discovery_report(", "return false;",
                    "ggml_sycl_release_model_slot_resources(owner);"),
        "teardown treats NOT_FOUND as the post-condition, not a failure":
            "MOE_DISCOVERY_RESULT_NOT_FOUND" in teardown_impl,
        "explicit activation reports BUSY rather than OK for a half-done switch":
            "return GGML_SYCL_LIFECYCLE_BUSY;" in activate_impl,

        # The clear list lives in one place.
        "module shutdown reuses the owner path's clear list":
            "g_moe_live_discovery.install_fresh();" in module_reset
            and "g_moe_hybrid_init_done.store(false" not in module_reset
            and "g_expert_popularity.clear();" not in module_reset,

        # The existing module-reload gate covers the new state too.
        "the module-reload clean gate checks the registry is empty":
            "g_moe_discovery_registry.parked_owner_count() == 0" in backend
            and "!g_moe_discovery_registry.has_active_owner()" in backend,
    }

    return sorted(name for name, text in anchors.items() if not text), \
           sorted(name for name, ok in checks.items() if not ok), len(checks)


# Positive controls for the two absence-based checks. They pass vacuously on any
# tree that never had the forbidden construct, so each one is re-evaluated
# against a backend that deliberately contains it.
ABSENCE_MUTANTS = {
    "the load boundary never clears the MoE cluster globally": (
        "        if (result.committed) {",
        "        ggml_sycl_reset_moe_module_state();\n        if (result.committed) {"),
    "the all-owner sweep exists only in module shutdown": (
        "    g_adaptive_prestage.reset_after_stop();",
        "    g_moe_discovery_registry.release_all_for_module_shutdown();\n    g_adaptive_prestage.reset_after_stop();"),
    "no integration site discards the transition result": (
        "        ggml_sycl_release_model_slot_resources(owner);",
        "        (void) g_moe_discovery_registry.activate(moe_owner);\n        ggml_sycl_release_model_slot_resources(owner);"),
    "module shutdown reuses the owner path's clear list": (
        "    g_adaptive_prestage.reset_after_stop();",
        "    g_moe_hybrid_init_done.store(false, std::memory_order_relaxed);\n    g_adaptive_prestage.reset_after_stop();"),
}


def self_test(backend, registry):
    """Every absence check must fail once its forbidden construct is injected."""
    problems = []
    for name, (anchor, replacement) in ABSENCE_MUTANTS.items():
        if anchor not in backend:
            problems.append("mutation anchor missing for: " + name)
            continue
        _, failed, _ = evaluate(backend.replace(anchor, replacement, 1), registry)
        if name not in failed:
            problems.append("check did not fire on its mutant: " + name)
    return problems


backend = Path(args.backend).read_text()
registry = Path(args.registry).read_text() if Path(args.registry).exists() else ""

missing_anchors, failed, n_checks = evaluate(backend, registry)

for name in missing_anchors:
    print("MISSING ANCHOR: " + name)
for name in failed:
    print("FAIL: " + name)

if missing_anchors or failed:
    sys.exit(1)

if args.self_test:
    problems = self_test(backend, registry)
    for problem in problems:
        print("SELF-TEST FAIL: " + problem)
    if problems:
        sys.exit(1)
    print("sycl moe discovery owner contract: self-test PASS ({} mutants)".format(len(ABSENCE_MUTANTS)))

print("sycl moe discovery owner contract: PASS ({} checks)".format(n_checks))
