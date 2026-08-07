#!/usr/bin/env python3
"""Zone-reset escape audit source contract (llama.cpp-iiff Phase 0).

Host-only: reads sources, runs no build, loads no model and touches no device.

The audit is an env-gated instrument with no test of its own that can prove it is
WIRED. Its output is the Phase 0 deliverable -- the inventory that becomes the
Phase 1 work-list -- and its characteristic failure is silent: a reset site that
carries no hook contributes nothing to the capture, and a capture that is missing
a site looks exactly like a site with no escapes. That is the "absence of work
looks like success" shape, so the wiring is asserted here instead of inferred
from a green run.

What this gates:

  1. Every current reset/drain site carries the audit hook. Sites move; this
     fails loudly when one grows a new spelling or a new site appears untouched.
  2. The hook is declared BEFORE the site's allocator lock. The audit's mutex is
     canonical §12.5 isolated D, which may not be co-held with L1-L5; reverse
     destruction order is the entire mechanism that keeps it clear, and it is
     invisible at the call site, so it is easy to "tidy" away.
  3. The report reaches the sink. GGML_LOG_INFO is dropped at default verbosity
     in every tool, so an INFO-level audit line yields an empty capture --
     indistinguishable from "no escapes found".
  4. The audit frees nothing. Report-instead-of-reset only means anything if the
     allocations stay owned by whoever holds their handles.
  5. It is inert by default and honours the level-2 suppression at both resets.

Every check fails against the pre-port tree. Run it that way to confirm:

    ./tests/test-sycl-zone-reset-audit-source.py \\
        --cache <(git show a67339ce4:ggml/src/ggml-sycl/unified-cache.cpp) \\
        --backend <(git show a67339ce4:ggml/src/ggml-sycl/ggml-sycl.cpp) \\
        --common <(git show a67339ce4:ggml/src/ggml-sycl/common.cpp)

The `anchors` section proves every region the checks read was actually found, so
a renamed function reports a missing anchor instead of passing vacuously. Run
with --self-test to prove the order- and absence-based checks fire against
mutants.
"""
import argparse
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--cache", default=str(root / "ggml/src/ggml-sycl/unified-cache.cpp"))
parser.add_argument("--backend", default=str(root / "ggml/src/ggml-sycl/ggml-sycl.cpp"))
parser.add_argument("--common", default=str(root / "ggml/src/ggml-sycl/common.cpp"))
parser.add_argument("--header", default=str(root / "ggml/src/ggml-sycl/unified-cache.hpp"))
parser.add_argument("--self-test", action="store_true",
                    help="prove the order- and absence-based checks fire when the property is broken")
args = parser.parse_args()


def strip_comments(source):
    """Remove C and C++ comments, preserving string literals and line count.

    Every check below reads active code only. Without this, a check like "the
    emit path does not use GGML_LOG_INFO" is spoofed by a COMMENT naming it --
    and the audit's comments discuss GGML_LOG_INFO at length, precisely because
    avoiding it is the point. String literals are preserved because several
    checks read them.
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
        if ch == "/" and i + 1 < n and source[i + 1] == "/":
            while i < n and source[i] != "\n":
                i += 1
            continue
        if ch == "/" and i + 1 < n and source[i + 1] == "*":
            i += 2
            while i + 1 < n and not (source[i] == "*" and source[i + 1] == "/"):
                if source[i] == "\n":
                    out.append("\n")
                i += 1
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def squeeze(source):
    """Collapse runs of spaces and tabs to one space.

    Every needle below is written with single spaces so that clang-format's
    column alignment cannot break a check -- this file's declarations sit in
    vertically aligned blocks where adding one neighbour re-pads the column.
    """
    return re.sub(r"[ \t]+", " ", source)


def body_of(source, signature_prefix):
    """Text from a definition's signature to its first column-0 closing brace.

    `[^;]*?` before the opening brace is what skips a forward declaration: the
    declaration's terminating `;` cannot be crossed, so the match lands on the
    definition even when a prototype appears first.
    """
    match = re.search(re.escape(signature_prefix) + r"[^;]*?\)\s*(?:const\s*)?(?:noexcept\s*)?\{.*?^\}\n",
                      source, re.S | re.M)
    return match.group(0) if match else ""


def region(source, start_needle, end_needle):
    """Text from start_needle to the end of the definition end_needle opens."""
    start = source.find(start_needle)
    if start < 0:
        return ""
    tail = body_of(source[start:], end_needle)
    if not tail:
        return ""
    return source[start:start + source[start:].find(tail) + len(tail)]


def precedes(text, first, second):
    """True when both appear and the FIRST occurrence of `first` comes earlier.

    `ordered()` is not enough for the lock-order checks: it would accept a hook
    placed after one lock so long as some later lock followed it.
    """
    a = text.find(first)
    b = text.find(second)
    return a >= 0 and b >= 0 and a < b


def evaluate(cache, backend, common, header):
    cache, backend, common, header = squeeze(cache), squeeze(backend), squeeze(common), squeeze(header)

    zone_reset = body_of(cache, "void unified_cache::zone_reset(")
    host_reset = body_of(cache, "void unified_cache::host_zone_reset(")
    scratch = body_of(cache, "void unified_cache::reset_scratch_pool(")
    reclaim = body_of(cache, "size_t unified_cache::reclaim_weight_entries(")
    zone_alloc = body_of(cache, "void * unified_cache::zone_alloc(")
    zone_free = body_of(cache, "void unified_cache::zone_free(")
    host_alloc = body_of(cache, "void * unified_cache::host_zone_alloc(")
    host_free = body_of(cache, "void unified_cache::host_zone_free(")
    emit = body_of(cache, "void zone_audit_emit(")
    level = body_of(cache, "int zone_reset_audit_level(")
    sig_handler = body_of(cache, "void zone_audit_fatal_signal_handler(")
    sig_install = body_of(cache, "void zone_audit_install_handlers(")
    report = body_of(cache, "void zone_audit_report_locked(")
    audit_block = region(cache, "struct zone_audit_cohort_size_stats {", "void zone_reset_audit_report(")
    graph_compute = body_of(backend, "static void ggml_backend_sycl_graph_compute_impl(")
    watchdog = body_of(common, "static void watchdog_thread_fn(")

    # Anchors: every region the checks read must have been found. A rename that
    # empties one of these reports a missing anchor rather than a silent pass.
    anchors = {
        "unified_cache::zone_reset body": zone_reset,
        "unified_cache::host_zone_reset body": host_reset,
        "unified_cache::reset_scratch_pool body": scratch,
        "unified_cache::reclaim_weight_entries body": reclaim,
        "unified_cache::zone_alloc body": zone_alloc,
        "unified_cache::zone_free body": zone_free,
        "unified_cache::host_zone_alloc body": host_alloc,
        "unified_cache::host_zone_free body": host_free,
        "zone_audit_emit body": emit,
        "zone_reset_audit_level body": level,
        "zone_audit_fatal_signal_handler body": sig_handler,
        "zone_audit_install_handlers body": sig_install,
        "zone_audit_report_locked body": report,
        "audit implementation block": audit_block,
        "ggml_backend_sycl_graph_compute_impl body": graph_compute,
        "watchdog_thread_fn body": watchdog,
    }

    checks = {
        # 1. Every current reset/drain site carries the hook. A site with no hook
        #    contributes nothing to the capture, and a capture missing a site is
        #    indistinguishable from a site with no escapes.
        'the VRAM zone reset records a site visit':
            'zone_audit_site_visit audit("device-zone-reset"' in zone_reset,
        'the host zone reset records a site visit':
            'zone_audit_site_visit audit("host-zone-reset"' in host_reset,
        'the scratch pool reset records a site visit':
            'zone_audit_site_visit audit("scratch-pool-reset"' in scratch,
        'the weight reclaim boundary records a site visit':
            'zone_audit_site_visit audit("weight-reclaim"' in reclaim,

        # The weight-reclaim site name is the mode, so all three boundaries
        # (load-boundary / mid-load-replan / model-teardown) appear as distinct
        # rows rather than collapsing into one.
        'the weight reclaim site is named by mode, not collapsed':
            "weight_reclaim_mode_name(mode)" in reclaim.split("zone_audit_site_visit audit(")[-1][:200],

        # 2. Lock order (canonical §12.5): the audit mutex is isolated D and must
        #    not be co-held with L1-L5. Reverse destruction order is the whole
        #    mechanism, and it is invisible at the call site.
        'the VRAM reset hook is declared before the zone allocator lock':
            precedes(zone_reset, "zone_audit_site_visit audit(", "lock(z.alloc_mutex)"),
        'the VRAM reset hook is declared before the registry lock':
            precedes(zone_reset, "zone_audit_site_visit audit(", "lock(g_runtime_alloc_mutex)"),
        'the host reset hook is declared before the registry lock':
            precedes(host_reset, "zone_audit_site_visit audit(", "lock(g_runtime_alloc_mutex)"),
        'the weight reclaim hook is declared before the cache lock':
            precedes(reclaim, "zone_audit_site_visit audit(", "lock(rw_mutex_)"),

        # An early-return placed BEFORE the hook makes the site vanish from
        # the inventory entirely on that path, rather than reading as
        # "visited and clean" -- worse than a visits=0 reading, because an
        # absent row is indistinguishable from a wiring regression and voids
        # baseline comparisons (llama.cpp-2757's battery finding: the
        # unreserved-pool early-return moved above the hook during review
        # iteration, and self-test coverage did not exist to catch it).
        'the scratch pool reset hook is declared before the reservation early-return':
            precedes(scratch, 'zone_audit_site_visit audit("scratch-pool-reset"', "!scratch_pool_ptr_"),

        # 3. The report must reach the sink. GGML_LOG_INFO is dropped at default
        #    verbosity in EVERY tool, so an INFO-level line yields an empty
        #    capture -- indistinguishable from "no escapes found".
        'the emit path uses WARN, not INFO':
            "GGML_LOG_WARN(" in emit and "GGML_LOG_INFO" not in emit,
        'the emit path also writes a raw stderr copy':
            "fputs(" in emit and "stderr" in emit,

        # An unvisited site and a clean site must be distinguishable in the
        # output, or the inventory cannot be read at all.
        'the report distinguishes a clean site from an unvisited one':
            "visits_with_live=" in report and "visits=" in report,
        'the report says outright when no site was visited':
            "NO RESET SITE WAS VISITED" in report,

        # 4. Report-instead-of-reset means the allocations stay owned. The audit
        #    must not free, reset or reclaim anything itself.
        'the audit implementation frees nothing':
            not any(needle in audit_block for needle in
                    ("unified_free(", "sycl::free(", "->zone_free(", "->zone_reset(", "->reset()")),

        # 5. Inert by default, and level 2 is opt-in on top of level 1.
        'the audit is off unless the env var is set':
            'std::getenv("GGML_SYCL_ZONE_RESET_AUDIT")' in level and "return 0;" in level,
        'the VRAM reset honours level-2 suppression':
            "zone_reset_audit_suppresses_reset()" in zone_reset,
        'the host reset honours level-2 suppression':
            "zone_reset_audit_suppresses_reset()" in host_reset,
        'the scratch pool reset honours level-2 suppression':
            "zone_reset_audit_suppresses_reset()" in scratch,

        # A diagnostic must not disarm somebody else's crash handler. The planner
        # canaries (test-planner-canary-*.cpp) install SIGABRT/SIGSEGV spill
        # handlers whose output IS the test; restoring SIG_DFL unconditionally
        # would silently break them for any run with the audit env set.
        'the install saves the handlers it displaces':
            "g_zone_audit_prev_sigsegv.store(std::signal(SIGSEGV" in sig_install
            and "g_zone_audit_prev_sigabrt.store(std::signal(SIGABRT" in sig_install,
        'the fatal handler chains to the displaced handler, not SIG_DFL':
            "g_zone_audit_prev_sigsegv" in sig_handler and "std::signal(sig, SIG_DFL)" not in sig_handler,

        # Per-graph attribution, and the flush on the paths that skip atexit.
        'the graph boundary bumps the audit graph sequence':
            "zone_reset_audit_begin_graph(" in graph_compute,
        'the watchdog flushes the inventory before _Exit':
            precedes(watchdog, "zone_reset_audit_report(", "std::_Exit(1)"),

        # Timing instrumentation on all four zone alloc/free entry points -- the
        # measurement the pre-registered allocator decision rule (c-c1n3) needs.
        'zone_alloc is timed':
            "zone_audit_timer" in zone_alloc,
        'zone_free is timed':
            "zone_audit_timer" in zone_free,
        'host_zone_alloc is timed':
            "zone_audit_timer" in host_alloc,
        'host_zone_free is timed':
            "zone_audit_timer" in host_free,

        # The header must document the contract the captures are read against.
        'the header declares the audit API':
            all(needle in header for needle in
                ("bool zone_reset_audit_enabled();", "void zone_reset_audit_begin_graph(int device);",
                 "void zone_reset_audit_report(const char * where);")),
    }

    return sorted(name for name, text in anchors.items() if not text), \
           sorted(name for name, ok in checks.items() if not ok), len(checks)


# Positive controls. An order check and an absence check both pass vacuously on a
# tree that never had the defect, so each is re-evaluated against a source that
# deliberately contains it. Each value is (target, [(anchor, replacement), ...]);
# a list because reordering two statements that are not adjacent in the STRIPPED
# source -- a comment between them leaves a blank line -- takes two edits.
MUTANTS = {
    "the VRAM reset hook is declared before the zone allocator lock": (
        "cache",
        [('    zone_audit_site_visit audit("device-zone-reset", vram_zone_name(zone), '
          'ggml_sycl_get_device_id_from_queue(queue_));\n', ''),
         ('    std::lock_guard<std::mutex> lock(z.alloc_mutex);',
          '    std::lock_guard<std::mutex> lock(z.alloc_mutex);\n'
          '    zone_audit_site_visit audit("device-zone-reset", vram_zone_name(zone), '
          'ggml_sycl_get_device_id_from_queue(queue_));')]),
    "the emit path uses WARN, not INFO": (
        "cache",
        [('    GGML_LOG_WARN("%s", line.c_str());', '    GGML_LOG_INFO("%s", line.c_str());')]),
    "the audit implementation frees nothing": (
        "cache",
        [("void zone_reset_audit_report(const char * where) {",
          "void zone_reset_audit_report(const char * where) {\n    unified_free(nullptr);")]),
    "the graph boundary bumps the audit graph sequence": (
        "backend",
        [("    ggml_sycl::zone_reset_audit_begin_graph(sycl_ctx->device);\n", "")]),
    "the watchdog flushes the inventory before _Exit": (
        "common",
        [('            ggml_sycl::zone_reset_audit_report("sycl-watchdog-exit");\n', ''),
         ("            std::_Exit(1);",
          "            std::_Exit(1);\n"
          '            ggml_sycl::zone_reset_audit_report("sycl-watchdog-exit");')]),
    "the scratch pool reset records a site visit": (
        "cache",
        [('        zone_audit_site_visit audit("scratch-pool-reset", "bump", ',
          '        zone_audit_site_visit unused_audit("scratch-pool-reset-renamed", "bump", ')]),
    # Reproduces the exact llama.cpp-2757 battery regression: an early-return
    # inserted before the hook, so the site vanishes instead of reading
    # "visited and clean". A single insertion right after the signature is
    # enough to make "!scratch_pool_ptr_" precede the hook -- no need to
    # relocate the real (comment-bearing) early-return further down.
    "the scratch pool reset hook is declared before the reservation early-return": (
        "cache",
        [("void unified_cache::reset_scratch_pool() {",
          "void unified_cache::reset_scratch_pool() {\n    if (!scratch_pool_ptr_) { return; }")]),
    # The exact "tidy" this guards against: collapsing the chain back to the
    # unconditional restore the reference had.
    "the fatal handler chains to the displaced handler, not SIG_DFL": (
        "cache",
        [("    void (*prev)(int) =", "    void (*unused_prev)(int) ="),
         ("    std::signal(sig, prev);", "    std::signal(sig, SIG_DFL);")]),
}


def self_test(cache, backend, common, header):
    """Every mutable check must fail once its defect is injected."""
    problems = []
    for name, (target, edits) in MUTANTS.items():
        sources = {"cache": cache, "backend": backend, "common": common, "header": header}
        missing = [anchor for anchor, _ in edits if anchor not in sources[target]]
        if missing:
            problems.append("mutation anchor missing for: " + name)
            continue
        for anchor, replacement in edits:
            sources[target] = sources[target].replace(anchor, replacement, 1)
        _, failed, _ = evaluate(sources["cache"], sources["backend"], sources["common"], sources["header"])
        if name not in failed:
            problems.append("check did not fire on its mutant: " + name)
    return problems


cache = strip_comments(Path(args.cache).read_text())
backend = strip_comments(Path(args.backend).read_text())
common = strip_comments(Path(args.common).read_text())
header = strip_comments(Path(args.header).read_text())

missing_anchors, failed, n_checks = evaluate(cache, backend, common, header)

for name in missing_anchors:
    print("MISSING ANCHOR: " + name)
for name in failed:
    print("FAIL: " + name)

if missing_anchors or failed:
    sys.exit(1)

if args.self_test:
    problems = self_test(cache, backend, common, header)
    for problem in problems:
        print("SELF-TEST FAIL: " + problem)
    if problems:
        sys.exit(1)
    print("sycl zone reset audit source contract: self-test PASS ({} mutants)".format(len(MUTANTS)))

print("sycl zone reset audit source contract: PASS ({} checks)".format(n_checks))
