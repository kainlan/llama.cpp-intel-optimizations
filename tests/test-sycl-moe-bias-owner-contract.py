#!/usr/bin/env python3
"""Owner-keyed MoE expert-bias / fused-activation contract (llama.cpp-nlww, canonical §12).

Host-only: reads sources, runs no build, loads no model and touches no device.

The companion executable test (test-moe-bias-owner-state) proves the state type
and the ownership rules against a stand-in live state. It cannot see whether the
REAL cluster in ggml-sycl.cpp is wired to that registry at all, which is what
this gate covers: the park/restore/clear wiring, the retired once-semantics, the
absence of the load-boundary global clear the k7b0 ruling rejects, and the two
API shapes that make the dangling-pointer mistakes unwritable.

Every check below fails against the pre-nlww tree. Run it that way to confirm:

    ./tests/test-sycl-moe-bias-owner-contract.py \\
        --backend <(git show 3ccf26264:ggml/src/ggml-sycl/ggml-sycl.cpp) \\
        --header  /dev/null

The `positive controls` section proves every anchor the checks key off exists,
so a renamed symbol reports a missing anchor instead of passing vacuously, and
`--self-test` proves the absence-based checks fire when the construct they
forbid is present.
"""
import argparse
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default=str(root / "ggml/src/ggml-sycl/ggml-sycl.cpp"))
parser.add_argument("--header", default=str(root / "ggml/src/ggml-sycl/moe-bias-state.hpp"))
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
    """Same, for a member function indented four spaces inside a class."""
    match = re.search(re.escape(signature_prefix) + r"[^;{]*?\{.*?^    \}\n", source, re.S | re.M)
    return match.group(0) if match else ""


def block_after(source, marker, closer="\n    }\n"):
    """The text from `marker` to the end of its enclosing four-space block.

    Returns "" when the marker is absent, so a rewrite that drops it reports a
    failure rather than an empty-region pass.
    """
    at = source.find(marker)
    if at < 0:
        return ""
    end = source.find(closer, at)
    return source[at:end if end >= 0 else len(source)]


# Every mutex acquisition spelled the way this file spells them. Used to prove
# the bias-state lock is strictly leaf: never held while another is taken.
LOCK_RE = re.compile(r"(?:lock_guard|unique_lock|shared_lock)<[^>]+>\s+\w+\(([^)]+)\)")


def header_includes(header):
    """Every `#include` target in the header, angle brackets or quotes intact."""
    return re.findall(r'^\s*#\s*include\s+(<[^>]+>|"[^"]+")', header, re.M)


def other_locks_taken(text):
    """Names of mutexes locked in `text` other than the bias-state mutex."""
    return sorted({name.strip() for name in LOCK_RE.findall(text)
                   if name.strip() != "g_moe_bias_state_mutex"})


def evaluate(backend, header):
    live_park    = member_body_of(backend, "    std::unique_ptr<ggml_sycl::moe_discovery_parked_state> park(")
    live_restore = member_body_of(backend, "    void restore(std::unique_ptr<ggml_sycl::moe_discovery_parked_state> parked)")
    live_fresh   = member_body_of(backend, "    void install_fresh(")
    lookup       = body_of(backend, "static bool ggml_sycl_moe_bias_lookup(")
    scanned      = body_of(backend, "static bool ggml_sycl_moe_bias_scanned(")
    publish_bias = body_of(backend, "static void ggml_sycl_moe_bias_publish(")
    act_detected = body_of(backend, "static bool ggml_sycl_moe_act_detected(")
    publish_act  = body_of(backend, "static void ggml_sycl_moe_act_publish(")
    take         = body_of(backend, "static ggml_sycl::moe_bias_activation_state ggml_sycl_moe_bias_state_take(")
    install      = body_of(backend, "static void ggml_sycl_moe_bias_state_install(")
    clear        = body_of(backend, "static void ggml_sycl_moe_bias_state_clear(")
    module_reset = body_of(backend, "static void ggml_sycl_reset_moe_module_state(")
    bias_scan    = block_after(backend, "if (!ggml_sycl_moe_bias_scanned()) {")
    act_scan     = block_after(backend, "if (!ggml_sycl_moe_act_detected()) {")

    helpers = {
        "ggml_sycl_moe_bias_lookup":       lookup,
        "ggml_sycl_moe_bias_scanned":      scanned,
        "ggml_sycl_moe_bias_publish":      publish_bias,
        "ggml_sycl_moe_act_detected":      act_detected,
        "ggml_sycl_moe_act_publish":       publish_act,
        "ggml_sycl_moe_bias_state_take":   take,
        "ggml_sycl_moe_bias_state_install": install,
        "ggml_sycl_moe_bias_state_clear":  clear,
    }

    # ---- positive controls: every anchor the checks below key off exists ----
    anchors = [
        ("backend defines the owned live state",     "static ggml_sycl::moe_bias_activation_state g_moe_bias_state;" in backend),
        ("backend defines the bias-state mutex",     "static std::shared_mutex                    g_moe_bias_state_mutex;" in backend),
        ("backend includes the state header",        'include "ggml-sycl/moe-bias-state.hpp"' in backend),
        ("parked discovery carries a bias member",   "ggml_sycl::moe_bias_activation_state bias;" in backend),
        ("live park() body found",                   bool(live_park)),
        ("live restore() body found",                bool(live_restore)),
        ("live install_fresh() body found",          bool(live_fresh)),
        ("module reset body found",                  bool(module_reset)),
        ("bias scan block found",                    bool(bias_scan)),
        ("activation scan block found",              bool(act_scan)),
        ("header defines the state type",            "struct moe_bias_activation_state {" in header),
        ("header defines layer_expert_biases",       "struct layer_expert_biases {" in header),
        ("header defines the undetected sentinel",   "moe_fused_act_undetected" in header),
    ] + [("helper %s() body found" % name, bool(text)) for name, text in helpers.items()]

    # ---- the wiring the executable test cannot see ----
    checks = [
        # park/restore/clear: an owner's biases move with the owner.
        ("park() detaches the bias working set into the parked state",
         "parked->bias = ggml_sycl_moe_bias_state_take();" in live_park),
        ("restore() moves the parked bias working set back",
         "ggml_sycl_moe_bias_state_install(std::move(state->bias));" in live_restore),
        ("restore() downcasts non-const, since a const parked state cannot be moved from",
         "ggml_sycl_moe_parked_discovery * state = static_cast<ggml_sycl_moe_parked_discovery *>(parked.get());"
         in live_restore),
        ("install_fresh() clears the bias/activation cluster",
         "ggml_sycl_moe_bias_state_clear();" in live_fresh),

        # The retired once-semantics. std::once_flag cannot be reset, so the
        # guards must be the owner-keyed ones.
        ("the bias scan is guarded by the owner-keyed flag",
         "if (!ggml_sycl_moe_bias_scanned()) {" in backend),
        ("the bias scan records that it looked, even when it found nothing",
         "scanned.biases_scanned = true;" in bias_scan),
        ("the activation scan is guarded by the owner-keyed sentinel",
         "if (!ggml_sycl_moe_act_detected()) {" in backend),
        ("the activation scan publishes through the owner-keyed setter",
         "ggml_sycl_moe_act_publish(variant, alpha, limit);" in act_scan),

        # k7b0 ruling: ownership transfer, not isolated global clears.
        ("the only bias clear is the owner path's install_fresh()",
         backend.count("ggml_sycl_moe_bias_state_clear();") == 1),
        ("module shutdown clears the cluster through the owner path, not directly",
         "g_moe_bias_state" not in module_reset),

        # Both publishes re-check under the lock, so a losing concurrent scan
        # discards its own duplicate instead of replacing live pointers.
        ("bias publish re-checks the guard under the lock",
         "if (g_moe_bias_state.biases_scanned) {" in publish_bias),
        ("activation publish re-checks the sentinel under the lock",
         "if (g_moe_bias_state.act_detected()) {" in publish_act),
        ("bias publish installs via adopt_biases, so it cannot roll back the activation half",
         "adopt_biases(std::move(scanned))" in publish_bias),

        # The consumer reads a copy through the helper, never the map directly.
        ("the fusion path looks biases up through the helper",
         "ggml_sycl_moe_bias_lookup(cur_layer_fast, &layer_bias)" in backend),
        ("the lookup helper copies the entry out rather than returning a pointer into the map",
         "*out = *entry;" in lookup),

        # §12.5: g_moe_bias_state_mutex is L3 strictly leaf. Both halves are
        # required -- "takes no other lock" is vacuously true of a helper that
        # takes no lock at all, which would be the worse defect of the two.
        ("every helper takes the bias-state lock",
         all("g_moe_bias_state_mutex" in text for text in helpers.values())),
        ("no helper holds the bias-state lock while taking another lock",
         all(not other_locks_taken(text) for text in helpers.values())),

        # The pre-nlww process-lifetime state is gone, not merely shadowed.
        ("the process-lifetime bias map is gone",       "g_moe_expert_biases" not in backend),
        ("the process-lifetime host copies are gone",   "g_moe_bias_host_copies" not in backend),
        ("the process-lifetime activation globals are gone",
         "g_moe_fused_act_variant" not in backend and "g_moe_fused_act_alpha" not in backend),
        ("no std::once_flag guards the bias scan",      "bias_detect_flag" not in backend),

        # The two API shapes that make the dangling-pointer mistakes unwritable.
        # Copying the state would leave the copy's map pointing into the
        # original's buffers; returning a reference to host_copies.back() hands
        # out the one thing growth relocates. Both were real, not theoretical:
        # the reference form segfaulted this task's own test.
        ("the state type forbids copying",
         "moe_bias_activation_state(const moe_bias_activation_state &)             = delete;" in header),
        ("the state type forbids copy assignment",
         "moe_bias_activation_state & operator=(const moe_bias_activation_state &) = delete;" in header),
        ("add_host_copy hands out a stable pointer, not a reference into the outer vector",
         "float * add_host_copy(size_t n_floats) {" in header),
        # Host-only is what lets the executable test link no backend library and
        # take no -fsycl options, so it is checked structurally: every include
        # must be a standard-library angle include, none a project header.
        ("the header includes only the standard library",
         header_includes(header) and all(inc.startswith("<") for inc in header_includes(header))),
    ]
    return anchors, checks


def report(title, results):
    failed = [name for name, ok in results if not ok]
    for name, ok in results:
        print("  %-4s %s" % ("PASS" if ok else "FAIL", name))
    return failed


backend_text = Path(args.backend).read_text()
header_text = Path(args.header).read_text()

print("positive controls (anchors the checks key off):")
anchor_results, check_results = evaluate(backend_text, header_text)
missing_anchors = report("anchors", anchor_results)
print("contract checks:")
failed_checks = report("checks", check_results)

self_test_failures = []
if args.self_test:
    # Re-run the absence-based checks against a backend that deliberately
    # contains what they forbid. Without this they pass on any tree that never
    # had the construct, which is every tree written after the migration.
    print("self-test (absence checks must fire when the forbidden construct is present):")
    #
    # The module-reset poison is a substitution rather than an append, because
    # that check reads one function's body: appending anywhere else would leave
    # it passing and the control would certify nothing.
    module_reset_anchor = "    g_moe_multi_gpu_active.store(false, std::memory_order_relaxed);"
    if module_reset_anchor not in backend_text:
        print("  FAIL missing self-test anchor: module-reset poison site")
        self_test_failures.append("module-reset poison site")
    poisoned = backend_text.replace(
        module_reset_anchor, "    g_moe_bias_state.clear();\n" + module_reset_anchor, 1) + (
        "\nstatic std::unordered_map<int, layer_expert_biases> g_moe_expert_biases;\n"
        "static std::vector<std::vector<float>> g_moe_bias_host_copies;\n"
        "static std::atomic<int> g_moe_fused_act_variant{ -1 };\n"
        "static float g_moe_fused_act_alpha = 0.0f;\n"
        "static std::once_flag bias_detect_flag;\n"
        "static void ggml_sycl_moe_bias_extra_clear() {\n"
        "    ggml_sycl_moe_bias_state_clear();\n"
        "}\n")
    must_fire = {
        "the process-lifetime bias map is gone",
        "the process-lifetime host copies are gone",
        "the process-lifetime activation globals are gone",
        "no std::once_flag guards the bias scan",
        "the only bias clear is the owner path's install_fresh()",
        "module shutdown clears the cluster through the owner path, not directly",
    }
    _, poisoned_checks = evaluate(poisoned, header_text)
    for name, ok in poisoned_checks:
        if name in must_fire:
            fired = not ok
            print("  %-4s %s" % ("PASS" if fired else "FAIL", "fires: " + name))
            if not fired:
                self_test_failures.append(name)
    absent = must_fire - {name for name, _ in poisoned_checks}
    for name in sorted(absent):
        print("  FAIL missing self-test target: %s" % name)
        self_test_failures.append(name)

if missing_anchors:
    print("\nMISSING ANCHORS -- the checks below them cannot have tested anything:")
    for name in missing_anchors:
        print("  - %s" % name)
if failed_checks:
    print("\nFAILED CHECKS:")
    for name in failed_checks:
        print("  - %s" % name)
if self_test_failures:
    print("\nSELF-TEST DID NOT FIRE (the check is decorative):")
    for name in self_test_failures:
        print("  - %s" % name)

if missing_anchors or failed_checks or self_test_failures:
    sys.exit(1)
print("\nmoe bias owner contract: PASS (%d anchors, %d checks)" % (len(anchor_results), len(check_results)))
