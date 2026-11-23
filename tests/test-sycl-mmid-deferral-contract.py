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
     resolves cannot pass silently;
  4. the plan CAS (lifecycle_replace_placement_plan) takes the MMID context a
     prepared replacement displaces into a holder declared before the plan
     lock, so the pools' last owners are released after that lock (canonical
     12.5: no final release under a listed lock). The registry side is pinned
     by the host test's registry lock probe; this pins the caller;
  5. one predicate answers "same MMID workspace plan": the context bind's
     reconcile, the expert loan's assert and the plan CAS all call
     moe_mmid_same_workspace_plan (unified-cache.cpp), so no site can drift
     back to a subset of its fields;
  6. the follows question (ggml_sycl_mmid_workspaces_follow) asks the owner
     device's execution queue, not a backend context: a context pointer read
     after the lookup mutex is released can be freed under it;
  7. the bind's reconcile and exception reports name what refused
     (reconcile-refused, bind-exception), not registry-invalid: the registry
     accepted the bind's attempt on both paths.

EVERY assertion here is checked against a POISONED copy of the source under
--self-test, and the gate fails if a poisoned variant still passes. A source
gate authored after the fix goes green on every tree that has the fix, so its
green means nothing without a demonstrated red.
"""

import argparse
import pathlib
import re
import sys

from sycl_gate import body, run_without_source, strip_comments

SOURCE = pathlib.Path(__file__).resolve().parent.parent / "ggml" / "src" / "ggml-sycl" / "ggml-sycl.cpp"
CACHE_SOURCE = SOURCE.parent / "unified-cache.cpp"

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


BIND_FN = "static void ggml_sycl_bind_materialize_live_mmid_workspaces("
REPORT_CALL = REPORT_FN + "(" + TRIPWIRE_SITE

# The bind's three tripwire sites, each named by the argument only it passes: a refused materialization reports the
# registry's own detail, a refused reconcile the reconcile's refusal, and the catch-all EXCEPTION.
TRIPWIRE_SITES = [
    ("materialize-refused", ", detail,"),
    ("reconcile-refused", "refusal, token"),
    ("exception", "moe_mmid_materialize_refusal::EXCEPTION"),
]


def bind_body(text):
    at = text.find(BIND_FN)
    if at < 0:
        return None
    end = text.find("\n}\n", at)
    return text[at: end if end >= 0 else len(text)]


def tripwire_calls(body):
    """Every REPORT_FN call in the bind that names the tripwire, as (start, end) of the call statement."""
    calls = []
    at = body.find(REPORT_CALL)
    while at >= 0:
        end = body.find(");", at)
        calls.append((at, end + 2 if end >= 0 else len(body)))
        at = body.find(REPORT_CALL, at + 1)
    return calls


def check_tripwire_present(text):
    # Every named site must report the tripwire through REPORT_FN. The check names the missing site, so one site's
    # removal cannot hide behind the other two still reporting.
    body = bind_body(text)
    if body is None:
        return False, "context-bind function not found (renamed?)"
    calls = [body[s:e] for s, e in tripwire_calls(body)]
    missing = [name for name, marker in TRIPWIRE_SITES if not any(marker in call for call in calls)]
    if missing:
        return False, "context-bind tripwire missing at: %s" % ", ".join(missing)
    return True, "context-bind tripwire reports refusals at %d sites" % len(TRIPWIRE_SITES)


# (site, the reason its report must carry)
SITE_REASONS = [
    ("reconcile-refused", "moe_mmid_materialize_reason::RECONCILE_REFUSED"),
    ("exception", "moe_mmid_materialize_reason::BIND_EXCEPTION"),
]


def check_site_reasons(text):
    body = bind_body(text)
    if body is None:
        return False, "context-bind function not found (renamed?)"
    calls = [body[s:e] for s, e in tripwire_calls(body)]
    markers = dict(TRIPWIRE_SITES)
    wrong = [site for site, reason in SITE_REASONS
             if not any(markers[site] in call and reason in call for call in calls)]
    if wrong:
        return False, "context-bind report does not name what refused at: %s" % ", ".join(wrong)
    return True, "context-bind reconcile and exception reports name their own reason"


def poison_one_site(text, marker):
    """Delete the one tripwire report call that carries `marker`, leaving the other sites in place."""
    body = bind_body(text)
    if body is None:
        return text
    for s, e in tripwire_calls(body):
        if marker in body[s:e]:
            return text.replace(body, body[:s] + body[e:], 1)
    return text


CAS_FN = "bool lifecycle_replace_placement_plan("
CAS_HOLDER = "std::shared_ptr<void> displaced_mmid"
# g_lifecycle_plan_mutex is a witnessed mutex, so the guard deduces its type; any alignment of the name
CAS_LOCK_RE = re.compile(r"std::lock_guard(?:<[^>]*>)?\s+lock\(g_lifecycle_plan_mutex\);")
CAS_HAND_OUT = "stable_workspace, &displaced_mmid)"


def check_cas_releases_after_plan_lock(cache):
    fn = strip_comments(body(cache, CAS_FN))
    if not fn:
        return False, "plan CAS not found (renamed?)"
    lock_match = CAS_LOCK_RE.search(fn)
    holder, hand_out = fn.find(CAS_HOLDER), fn.find(CAS_HAND_OUT)
    lock = lock_match.start() if lock_match else -1
    if lock < 0:
        return False, "plan CAS no longer takes the plan lock with a lock_guard named lock (renamed?)"
    if hand_out < 0:
        return False, "plan CAS no longer hands replace_plan a holder: the displaced context dies in the registry"
    if holder < 0 or holder > lock:
        return False, "plan CAS declares its displaced-context holder after the plan lock: released under it"
    return True, "plan CAS releases the displaced MMID context after the plan lock"


SAME_PLAN = "moe_mmid_same_workspace_plan("
SAME_PLAN_SYCL_SITES = [
    ("bind", "ggml_sycl::" + SAME_PLAN + "*published->plan, *live->plan)"),
    ("loan-assert", "GGML_ASSERT(ggml_sycl::" + SAME_PLAN + "*current->plan, *next_plan))"),
]
SAME_PLAN_CAS_SITE = SAME_PLAN + "*expected->plan, *replacement->plan)"


def check_same_plan_sycl(text):
    code = strip_comments(text)
    missing = [name for name, call in SAME_PLAN_SYCL_SITES if call not in code]
    if missing:
        return False, "no longer asks %s at: %s" % (SAME_PLAN.rstrip("("), ", ".join(missing))
    return True, "bind and loan assert ask the shared same-plan predicate"


def check_same_plan_cas(cache):
    fn = strip_comments(body(cache, CAS_FN))
    if SAME_PLAN_CAS_SITE not in fn:
        return False, "plan CAS no longer asks %s for its in-place rebind" % SAME_PLAN.rstrip("(")
    return True, "plan CAS asks the shared same-plan predicate"


FOLLOWS_FN = "static bool ggml_sycl_mmid_workspaces_follow("
FOLLOWS_QUEUE = "ggml_sycl_execution_queue_for_device(workspace.owner_device)"
BACKEND_LOOKUP = "ggml_sycl_get_backend_context_for_device("


def check_follows_device_queue(text):
    fn = strip_comments(body(text, FOLLOWS_FN))
    if not fn:
        return False, "follows question not found (renamed?)"
    if BACKEND_LOOKUP in fn:
        return False, "follows question reads a backend context: freed under it once the lookup mutex is released"
    if FOLLOWS_QUEUE not in fn:
        return False, "follows question no longer asks the owner device's execution queue"
    return True, "follows question asks the device-global execution queue"


# (name, source the check reads, check)
CHECKS = [
    ("shared-predicate", "sycl", check_shared_predicate),
    ("load-end-defers", "sycl", check_load_end_defers),
    ("tripwire-present", "sycl", check_tripwire_present),
    ("cas-release-order", "cache", check_cas_releases_after_plan_lock),
    ("same-plan:sycl", "sycl", check_same_plan_sycl),
    ("same-plan:cas", "cache", check_same_plan_cas),
    ("follows-queue", "sycl", check_follows_device_queue),
    ("site-reasons", "sycl", check_site_reasons),
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


def poison_cas_holder_under_lock(cache):
    """The regression itself: the holder declared after the plan lock, so it is destroyed while the lock is held."""
    fn = body(cache, CAS_FN)
    at = fn.find(CAS_HOLDER)
    if at < 0:
        return cache
    line_end = fn.find("\n", at)
    holder_line = fn[at:line_end + 1]
    without = fn.replace(holder_line, "", 1)
    lock = CAS_LOCK_RE.search(without)
    if not lock:
        return cache
    moved = without[:lock.end()] + "\n        " + holder_line.strip() + without[lock.end():]
    return cache.replace(fn, moved, 1)


# Each poison must make its own check go red. If it does not, that check cannot
# observe the property it claims to enforce.
POISONS = {
    "shared-predicate":
        lambda t: t.replace(ADMISSION_INIT + " " + SHARED_CONST, ADMISSION_INIT + " false"),
    "load-end-defers": poison_load_end,
    "tripwire-present":
        lambda t: t.replace(TRIPWIRE_SITE, '"context-bind-REMOVED"'),
    "cas-release-order": poison_cas_holder_under_lock,
    # The regression: a site going back to a local predicate of its own.
    "same-plan:sycl": lambda t: t.replace(SAME_PLAN_SYCL_SITES[0][1],
                                          "ggml_sycl_same_mmid_workspace_plan(*published->plan, *live->plan)", 1),
    # The regression: the reconcile's report back on the registry's reason.
    "site-reasons": lambda t: t.replace(SITE_REASONS[0][1], "moe_mmid_materialize_reason::REGISTRY_INVALID", 1),
    "follows-queue": lambda t: t.replace(
        "ggml_sycl_execution_queue_for_device(workspace.owner_device) :",
        "ggml_sycl_get_backend_context_for_device(workspace.owner_device)->stream() :", 1),
    "same-plan:cas": lambda t: t.replace(SAME_PLAN_CAS_SITE, "same_workspace_locally(*expected->plan, *replacement->plan)", 1),
}

# A second control for the CAS: the holder kept but not handed to replace_plan.
CAS_DROP_HAND_OUT = ("cas:hand-out-dropped", lambda t: t.replace(CAS_HAND_OUT, "stable_workspace)", 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true",
                        help="prove each check can go red against a poisoned source")
    args = parser.parse_args()

    for source in (SOURCE, CACHE_SOURCE):
        if not source.exists():
            # Not 77: ctest reports that as a skip, which reads as green, and a moved source is what this gate is for.
            print("FAIL: %s not found" % source)
            return 1
    sources = {
        "sycl": SOURCE.read_text(encoding="utf-8", errors="replace"),
        "cache": CACHE_SOURCE.read_text(encoding="utf-8", errors="replace"),
    }
    text = sources["sycl"]

    failures = 0
    for name, key, check in CHECKS:
        ok, detail = check(sources[key])
        print("%-20s %s  %s" % (name, "PASS" if ok else "FAIL", detail))
        if not ok:
            failures += 1

    if args.self_test:
        print("--- poisoned controls (each MUST go red) ---")
        for name, key, check in CHECKS + [(CAS_DROP_HAND_OUT[0], "cache", check_cas_releases_after_plan_lock)]:
            poison = CAS_DROP_HAND_OUT[1] if name == CAS_DROP_HAND_OUT[0] else POISONS[name]
            poisoned = poison(sources[key])
            if poisoned == sources[key]:
                print("%-20s VOID  poison did not modify the source" % name)
                failures += 1
                continue
            ok, _ = check(poisoned)
            print("%-20s %s  poisoned source" % (name, "VOID (still passed!)" if ok else "red as required"))
            if ok:
                failures += 1

        # One control per tripwire site: that site's report deleted, the other two left. Red is required, and the
        # failure must name that site.
        for name, marker in TRIPWIRE_SITES:
            label = "tripwire:" + name
            poisoned = poison_one_site(text, marker)
            if poisoned == text:
                print("%-20s VOID  poison did not modify the source" % label)
                failures += 1
                continue
            ok, detail = check_tripwire_present(poisoned)
            named = not ok and name in detail and all(other not in detail for other, _ in TRIPWIRE_SITES
                                                      if other != name)
            print("%-20s %s  one site deleted" % (label, "red as required, named" if named else
                                                   "VOID (still passed!)" if ok else "VOID (red, site not named)"))
            if not named:
                failures += 1

        # A missing source must FAIL and say so (sycl_gate.run_without_source): 77 is ctest's skip, which reads as green.
        status, output = run_without_source(__file__)
        absent_ok = status not in (0, 77) and "not found" in output
        print("%-20s %s  exit status %d with the source absent"
              % ("missing source", "fails as required" if absent_ok else "VOID (passes, skips or dies elsewhere)", status))
        if not absent_ok:
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
