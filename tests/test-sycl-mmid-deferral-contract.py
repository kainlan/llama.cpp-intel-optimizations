#!/usr/bin/env python3
"""Source contract for the MoE MMID workspace deferral (llama.cpp-7lvk).

Three properties that no runtime test on this host can cheaply cover, because
reaching them needs a GPU and a loaded MoE model:

  1. the admission gate and the materialization gate share ONE constant, so
     they cannot drift into materializing a pool for a route that cannot run
     it, or withholding one from a route that can;
  2. load_end DEFERS instead of throwing -- the regression that made GPT-OSS
     unloadable was a bare `throw std::bad_alloc()` on this exact path;
  3. the context-bind hook carries the tripwire, so a deferral that never
     resolves cannot pass silently.

EVERY assertion here is checked against a POISONED copy of the source under
--self-test, and the gate fails if a poisoned variant still passes. A source
gate authored after the fix goes green on every tree that has the fix, so its
green means nothing without a demonstrated red.
"""

import argparse
import pathlib
import sys

SOURCE = pathlib.Path(__file__).resolve().parent.parent / "ggml" / "src" / "ggml-sycl" / "ggml-sycl.cpp"

ADMISSION_INIT = "constexpr bool q1_nvfp4_direct_b70_validated ="
SHARED_CONST = "k_moe_mmid_direct_route_validated"
LOAD_END_CALL = "ggml_sycl_materialize_published_mmid_workspaces(ticket.token"
TRIPWIRE_SITE = '"context-bind"'
REPORT_FN = "ggml_sycl_moe_mmid_report_refusal"


def check_shared_predicate(text):
    at = text.find(ADMISSION_INIT)
    if at < 0:
        return False, "admission gate initialiser not found (renamed?)"
    rhs = text[at + len(ADMISSION_INIT): text.find(";", at)]
    if SHARED_CONST not in rhs:
        return False, "admission gate no longer reads %s -- the two gates can drift" % SHARED_CONST
    return True, "admission gate reads the shared constant"


def check_load_end_defers(text):
    at = text.find(LOAD_END_CALL)
    if at < 0:
        return False, "load_end materialize call not found (renamed?)"
    window = text[at: at + 900]
    if "throw" in window:
        return False, "load_end still throws on materialization failure -- the deferral was reverted"
    return True, "load_end defers instead of throwing"


def check_tripwire_present(text):
    at = text.find(TRIPWIRE_SITE)
    if at < 0:
        return False, "context-bind tripwire site not found"
    if REPORT_FN not in text[max(0, at - 400): at]:
        return False, "context-bind site no longer reports through %s" % REPORT_FN
    return True, "context-bind tripwire reports refusals"


CHECKS = [
    ("shared-predicate", check_shared_predicate),
    ("load-end-defers", check_load_end_defers),
    ("tripwire-present", check_tripwire_present),
]

def poison_load_end(text):
    """Reinstate the exact regression: a throw inside load_end's failure branch.

    An earlier version of this poison prepended the throw BEFORE the call, which
    the check's forward window cannot see -- so the control passed and the check
    looked enforcing when it was not. The poison has to reproduce the real
    mutation, not merely mention the same token.
    """
    at = text.find(LOAD_END_CALL)
    if at < 0:
        return text
    brace = text.find("{", at)
    if brace < 0:
        return text
    return text[:brace + 1] + " throw std::bad_alloc();" + text[brace + 1:]


# Each poison must make its own check go red. If it does not, that check cannot
# observe the property it claims to enforce.
POISONS = {
    "shared-predicate":
        lambda t: t.replace(ADMISSION_INIT + " " + SHARED_CONST, ADMISSION_INIT + " false"),
    "load-end-defers": poison_load_end,
    "tripwire-present":
        lambda t: t.replace(TRIPWIRE_SITE, '"context-bind-REMOVED"', 1),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true",
                        help="prove each check can go red against a poisoned source")
    args = parser.parse_args()

    if not SOURCE.exists():
        print("SKIP: %s not found" % SOURCE)
        return 77
    text = SOURCE.read_text(encoding="utf-8", errors="replace")

    failures = 0
    for name, check in CHECKS:
        ok, detail = check(text)
        print("%-20s %s  %s" % (name, "PASS" if ok else "FAIL", detail))
        if not ok:
            failures += 1

    if args.self_test:
        print("--- poisoned controls (each MUST go red) ---")
        for name, check in CHECKS:
            poisoned = POISONS[name](text)
            if poisoned == text:
                print("%-20s VOID  poison did not modify the source" % name)
                failures += 1
                continue
            ok, _ = check(poisoned)
            print("%-20s %s  poisoned source" % (name, "VOID (still passed!)" if ok else "red as required"))
            if ok:
                failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
