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


# NOTE: the precedes()-style ordering checks depend on this stripping: code
# comments legitimately name the anchors they discuss (e.g. the C3 hoist comment
# names ensure_planned_arena_zones() before the real call), and an unstripped
# source would false-fail them. Never bypass strip_comments for a subset of checks.
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


def guard_block_contains(text, guard_pattern, mutation_pattern):
    """Lightweight control-flow check: every mutation is lexically inside the guard block."""
    guard = re.search(guard_pattern + r"\s*\{", text, re.S)
    mutations = list(re.finditer(mutation_pattern, text))
    if not guard or not mutations:
        return False
    brace = guard.end() - 1
    depth = 0
    end = -1
    for pos in range(brace, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                end = pos
                break
    return end >= 0 and all(brace < mutation.start() < end for mutation in mutations)


def evaluate(cache, backend, common, header):
    cache, backend, common, header = squeeze(cache), squeeze(backend), squeeze(common), squeeze(header)

    # Variable names kept as zone_reset/host_reset/scratch (pre-llama.cpp-37ba
    # rename) -- they are local Python bindings, not asserted-against C++
    # identifiers; only the body_of() search strings below (the actual C++
    # signatures) needed to follow the rename.
    zone_reset = body_of(cache, "void unified_cache::zone_settle(")
    host_reset = body_of(cache, "void unified_cache::host_zone_settle(")
    scratch = body_of(cache, "void unified_cache::scratch_pool_epoch_boundary(")
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
    reserve_onednn = body_of(cache, "bool unified_cache::reserve_onednn_scratch(")
    arena_forget = body_of(cache, "void unified_cache::arena_forget_allocation_locked(")
    arena_destroy = body_of(cache, "bool unified_cache::arena_destroy(")

    # llama.cpp-37ba: the segment of host_zone_settle() between its level-2
    # suppression check and the real bulk host_arena_->zone_reset(zone) call.
    # "if (zone_reset_audit_suppresses_reset()) {" is unique within this
    # function (the level-2 check appears nowhere else in it), so slicing from
    # there to the reset call isolates exactly the epoch_tracked early-return
    # guard without depending on how many comment lines separate them --
    # robust to reformatting or a longer/shorter rationale comment.
    _host_reset_suppress_idx = host_reset.find("if (zone_reset_audit_suppresses_reset()) {")
    _host_reset_call_idx = host_reset.find("host_arena_->zone_reset(zone);")
    host_reset_epoch_guard_tail = (
        host_reset[_host_reset_suppress_idx:_host_reset_call_idx]
        if _host_reset_suppress_idx >= 0 and _host_reset_call_idx >= 0 else "")

    # Owner-ruled acceptance criterion 1, made a machine check (llama.cpp-37ba):
    # the three OLD dispatcher names -- as real call/definition syntax, name
    # immediately followed by '(' -- must have zero occurrences in production
    # backend code. Both the bare member names (zone_reset, host_zone_reset,
    # reset_scratch_pool) and their free-function wrappers
    # (unified_cache_zone_reset, unified_cache_host_zone_reset,
    # unified_cache_reset_scratch_pool) are checked explicitly, because a
    # word-boundary check for the bare name alone does NOT catch the wrapper
    # form: "unified_cache_" ends in '_', a word character, so there is no
    # boundary before "zone_reset" inside "unified_cache_zone_reset" and a
    # bare-name-only pattern would silently miss a surviving wrapper call.
    #
    # Comments never reach this point at all -- strip_comments() already ran
    # (module level) before evaluate() ever sees `cache`/`backend`/`common`/
    # `header`, so every "named X() before llama.cpp-37ba's rename" narrative
    # comment is gone before this scan runs; only real code and string
    # literals remain, and no renamed string literal was left pointing at an
    # old name (verified: every log-message string was updated alongside its
    # call site). The one deliberate real-code survivor is
    # host_arena_->zone_reset(...), the pinned_chunk_pool TLSF-layer primitive
    # the owner ruling explicitly exempts ("genuinely internal ... not
    # reachable as backend policy") -- excluded by lookbehind, not by skipping
    # its defining file, because it is CALLED from unified-cache.cpp, which
    # this script does read. pinned-pool.cpp/.hpp (where it and
    # scratch_pool_reset_regions -- also exempted -- are DEFINED) are outside
    # this script's four sources entirely, so no separate exemption is needed
    # for the definition side.
    _old_dispatcher_pattern = re.compile(
        r"(?<!host_arena_->)(?<!pinned_chunk_pool::)\b"
        r"(unified_cache_zone_reset|unified_cache_host_zone_reset|unified_cache_reset_scratch_pool|"
        r"zone_reset|host_zone_reset|reset_scratch_pool)\s*\(")
    old_dispatcher_name_survivors = [
        "{}:{}".format(_label, _m.group(1))
        for _label, _src in (("cache", cache), ("backend", backend), ("common", common), ("header", header))
        for _m in _old_dispatcher_pattern.finditer(_src)
    ]

    # Anchors: every region the checks read must have been found. A rename that
    # empties one of these reports a missing anchor rather than a silent pass.
    anchors = {
        "unified_cache::zone_settle body": zone_reset,
        "unified_cache::host_zone_settle body": host_reset,
        "unified_cache::scratch_pool_epoch_boundary body": scratch,
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
        "unified_cache::reserve_onednn_scratch body": reserve_onednn,
        "unified_cache::arena_forget_allocation_locked body": arena_forget,
        "unified_cache::arena_destroy body": arena_destroy,
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
        'the VRAM reset hook is declared before the allocator-group lock':
            precedes(zone_reset, "zone_audit_site_visit audit(", "lock(arena_allocator_group_mutex(zone))"),
        'the VRAM reset hook is declared before the registry lock':
            precedes(zone_reset, "zone_audit_site_visit audit(", "lock(g_runtime_alloc_mutex)"),
        'the host reset hook is declared before the registry lock':
            precedes(host_reset, "zone_audit_site_visit audit(", "lock(g_runtime_alloc_mutex)"),
        'the weight reclaim hook is declared before the cache lock':
            precedes(reclaim, "zone_audit_site_visit audit(", "lock(rw_mutex_)"),

        # The allocator lifecycle is a per-physical-group transaction, not a
        # logical-zone mutex plus a blind generation bump. RESETTING excludes
        # allocation/destroy while the terminal authority drain runs unlocked;
        # state and epoch must both be revalidated before reset/publication.
        'allocator groups carry mutex CV state and epoch':
            all(needle in header for needle in
                ("enum class allocator_group_state", "OPEN, RESETTING, CLOSING",
                 "std::mutex", "std::condition_variable lifecycle_cv;", "uint64_t lifecycle_epoch")),
        'zone settle claims OPEN to RESETTING under the group lock':
            "std::unique_lock<std::mutex> lock(arena_allocator_group_mutex(zone))" in zone_reset
            and precedes(zone_reset, "physical_group.state != allocator_group_state::OPEN",
                         "physical_group.state = allocator_group_state::RESETTING"),
        'zone settle revalidates exact group state and epoch after authority drain':
            "const uint64_t settle_epoch = ++physical_group.lifecycle_epoch" in zone_reset
            and "physical_group.state != allocator_group_state::RESETTING" in zone_reset
            and "physical_group.lifecycle_epoch != settle_epoch" in zone_reset
            and precedes(zone_reset, "wait_for_terminal_leases(", "lock.lock()")
            and precedes(zone_reset, "lock.lock()", "physical_group.lifecycle_epoch != settle_epoch"),
        'zone settle publishes before reopening and notifying the group':
            precedes(zone_reset, "arena_publish_prebuilt(", "physical_group.state = allocator_group_state::OPEN")
            and precedes(zone_reset, "physical_group.state = allocator_group_state::OPEN",
                         "physical_group.lifecycle_cv.notify_all()"),
        'exact allocation ownership is erased by allocation id':
            "found->second.allocation_id == allocation_id" in arena_forget
            and "runtime->second.handle.alloc_id == exact_id" in arena_forget
            and "group.allocations.erase(found)" in arena_forget
            and guard_block_contains(
                arena_forget,
                r"if\s*\(\s*runtime\s*!=\s*g_runtime_alloc_registry\.end\(\)\s*&&\s*"
                r"runtime->second\.handle\.alloc_id\s*==\s*exact_id\s*\)",
                r"g_runtime_alloc_registry\.erase\s*\([^)]*\)"),
        'arena destroy refuses owned groups before physical free and never raw-clears first':
            arena_destroy.count("if (!group.allocations.empty())") == 1
            and re.search(r"sycl::free\s*\(", arena_destroy) is not None
            and arena_destroy.find("if (!group.allocations.empty())") < re.search(r"sycl::free\s*\(", arena_destroy).start()
            and re.search(r"sycl::free\s*\(", arena_destroy).start() < arena_destroy.find("group.allocations.clear()"),
        'legacy arena generation bump helper is absent':
            "arena_generation_bump" not in cache and "arena_generation_bump" not in header,

        # An early-return placed BEFORE the hook makes the site vanish from
        # the inventory entirely on that path, rather than reading as
        # "visited and clean" -- worse than a visits=0 reading, because an
        # absent row is indistinguishable from a wiring regression and voids
        # baseline comparisons (llama.cpp-2757's battery finding: the
        # unreserved-pool early-return moved above the hook during review
        # iteration, and self-test coverage did not exist to catch it).
        'the scratch pool reset hook is declared before the reservation early-return':
            precedes(scratch, 'zone_audit_site_visit audit("scratch-pool-reset"', "!scratch_pool_ptr_"),

        # llama.cpp-1ntm: host_zone_settle()'s WEIGHT-zone guard and zone_settle()'s
        # WEIGHT/shared-KV-arena refusals predate 495343bb5 and had the identical
        # defect -- an early return ahead of the hook makes a refused reset
        # vanish from the inventory instead of reading as "visited and
        # refused". Same rationale as the scratch-pool check above.
        'the host reset hook is declared before the reservation early-return':
            precedes(host_reset, 'zone_audit_site_visit audit("host-zone-reset"', "!host_arena_"),
        'the VRAM reset hook is declared before the WEIGHT early-return':
            precedes(zone_reset, 'zone_audit_site_visit audit("device-zone-reset"', "zone == vram_zone_id::WEIGHT"),

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

        # 6. iiff Option C step 3 (llama.cpp-67c2): the oneDNN scratch point-release
        #    must free the OLD arena-owned reservation before anything consults zone
        #    capacity or attempts a re-plan. A spec-review finding on the first
        #    version of this step put the release inside the `total_needed <=
        #    zone_cap` branch, AFTER ensure_planned_arena_zones() -- so on the one
        #    path that most needed it (an existing reservation growing into a
        #    bigger one), the pointers were still live when the live-scratch
        #    refusal check ran, guaranteeing the re-plan refused itself, and
        #    control fell through to the shared direct-allocation cleanup whose
        #    arena-owned branch only nulled the fields: orphaned TLSF bytes,
        #    reclaimed only at whole-arena teardown. Same shape as the
        #    "hook declared before the early-return" checks above -- the ordering,
        #    not just the presence, is what a future edit can quietly break.
        #    The anchor moved from the bare zone_free() to the deferred-release
        #    helper when llama.cpp-ndn9 made the release event-gated; the
        #    ORDERING contract this check exists for is unchanged.
        'the oneDNN point-release precedes the growth-path re-plan':
            precedes(reserve_onednn, "defer_published_zone_release(onednn_weights_scratch_",
                    "ensure_planned_arena_zones()"),

        # 6b. llama.cpp-ndn9: releasing a PUBLISHED oneDNN scratch block must
        #    depend on the EVENT that consumes it, never on the reservation
        #    refcount and never on nothing at all.
        #
        #    onednn_pp_scratch_guard drops its token when its C++ scope ends,
        #    which is immediately after DnnlGemmWrapper::row_gemm() returns --
        #    and that ends in an asynchronous primitive.execute(). So
        #    onednn_scratch_refcount_ is already 0 while the primitive is still
        #    reading the block, and an immediate zone_free() here hands those
        #    bytes to the next zone_alloc() underneath live device work. The
        #    arena case is today shielded only by the backend streams happening
        #    to be in_order; the direct case (synchronous unified_free()) is not
        #    shielded by anything.
        #
        #    Presence of the deferred calls is checked together with ABSENCE of
        #    the immediate forms for the two published pointers: a future edit
        #    that "simplifies" one release back to zone_free() would otherwise
        #    leave the other deferred call still satisfying a presence-only
        #    check, and the regression would land silently.
        'the published oneDNN scratch releases are event-deferred':
            all(needle in reserve_onednn for needle in
                ("submit_barrier_all()",
                 "enqueue_deferred_zone_free(vram_zone_id::ONEDNN,",
                 "enqueue_deferred_free(ref, *ev)"))
            and not any(needle in reserve_onednn for needle in
                ("zone_free(vram_zone_id::ONEDNN, onednn_weights_scratch_)",
                 "zone_free(vram_zone_id::ONEDNN, onednn_activations_scratch_)")),

        # 7. llama.cpp-37ba (iiff Phase 3 / epic acceptance criterion 1, reconciled):
        #    host_zone_settle()'s SCRATCH/STAGING population no longer needs the bulk
        #    host_arena_->zone_reset(zone) call to reclaim anything -- C2 made every
        #    allocation in those zones free itself individually at release, so by the
        #    time this function's registry scan finds nothing live, there is nothing
        #    left to reclaim. The remaining call exists only for KV, which is
        #    deliberately out of the epic's scope (a context-lifetime zone with a
        #    genuine on-demand reclaim need at context-switch time -- see
        #    arena_reserve() and the KV buffer-type fallback in ggml-sycl.cpp).
        #    epoch_tracked's early return is what keeps SCRATCH/STAGING out of that
        #    real reclaim -- if a future edit weakens or removes it, the two zones
        #    silently regain a live dependency on a path that C2 already proved is
        #    unreachable for them, undoing this step's whole point. Dispatched
        #    through host_zone_boundary_check() (SCRATCH/STAGING) and
        #    host_zone_reclaim() (KV) since llama.cpp-37ba's naming split; both
        #    call this same shared implementation, so the property is checked
        #    once here rather than per-wrapper.
        'SCRATCH/STAGING return before host_zone_settle reaches the real bulk reclaim':
            "if (epoch_tracked) {" in host_reset_epoch_guard_tail and "return;" in host_reset_epoch_guard_tail,

        # 8. llama.cpp-37ba, owner-ruled rename (acceptance criterion 1 becomes a
        #    machine check): the three OLD dispatcher names must have zero call
        #    sites or definitions left in production backend code. Word-boundary
        #    regex so a compound identifier that happens to CONTAIN one of the
        #    old names as a substring (host_zone_settle, zone_boundary_check,
        #    scratch_pool_epoch_boundary, zone_reset_calls/us profiling fields,
        #    zone_reset_audit_* -- an entirely separate naming family) does not
        #    false-positive; see the exemption list below for what is
        #    deliberately allowed to remain.
        'the three old reset/boundary dispatcher names have zero production call sites':
            old_dispatcher_name_survivors == [],
    }

    return sorted(name for name, text in anchors.items() if not text), \
           sorted(name for name, ok in checks.items() if not ok), len(checks), old_dispatcher_name_survivors


# Positive controls. An order check and an absence check both pass vacuously on a
# tree that never had the defect, so each is re-evaluated against a source that
# deliberately contains it. Each value is (target, [(anchor, replacement), ...]);
# a list because reordering two statements that are not adjacent in the STRIPPED
# source -- a comment between them leaves a blank line -- takes two edits.
MUTANTS = {
    "the VRAM reset hook is declared before the allocator-group lock": (
        "cache",
        [('    zone_audit_site_visit audit("device-zone-reset", vram_zone_name(zone), '
          'ggml_sycl_get_device_id_from_queue(queue_));\n', ''),
         ('    std::unique_lock<std::mutex> lock(arena_allocator_group_mutex(zone));',
          '    std::unique_lock<std::mutex> lock(arena_allocator_group_mutex(zone));\n'
          '    zone_audit_site_visit audit("device-zone-reset", vram_zone_name(zone), '
          'ggml_sycl_get_device_id_from_queue(queue_));')]),
    "allocator groups carry mutex CV state and epoch": (
        "header", [("lifecycle_epoch = 0;", "unused_epoch = 0;")]),
    "zone settle claims OPEN to RESETTING under the group lock": (
        "cache", [("physical_group.state = allocator_group_state::RESETTING;",
                   "physical_group.state = allocator_group_state::OPEN;")]),
    "zone settle revalidates exact group state and epoch after authority drain": (
        "cache", [("physical_group.lifecycle_epoch != settle_epoch", "false")]),
    "zone settle publishes before reopening and notifying the group": (
        "cache", [("physical_group.state = allocator_group_state::OPEN;",
                   "physical_group.lifecycle_cv.notify_all();\n    physical_group.state = allocator_group_state::OPEN;")]),
    "exact allocation ownership is erased by allocation id": (
        "cache", [("void unified_cache::arena_forget_allocation_locked(vram_zone_id zone, void * ptr, uint64_t allocation_id) noexcept {",
                   "void unified_cache::arena_forget_allocation_locked(vram_zone_id zone, void * ptr, uint64_t allocation_id) noexcept {\n"
                   "    auto bypass = g_runtime_alloc_registry.find(ptr);\n"
                   "    if (bypass != g_runtime_alloc_registry.end()) g_runtime_alloc_registry.erase(bypass);")]),
    "arena destroy refuses owned groups before physical free and never raw-clears first": (
        "cache", [("bool unified_cache::arena_destroy() {",
                   "bool unified_cache::arena_destroy() {\n"
                   "    if (!arena_chunks_.empty()) sycl::free(arena_chunks_[0].ptr, arena_queue_->get_context());")]),
    "legacy arena generation bump helper is absent": (
        "cache", [("void unified_cache::zone_settle(vram_zone_id zone) {",
                   "void unified_cache::zone_settle(vram_zone_id zone) {\n    arena_generation_bump();")]),
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
        [("void unified_cache::scratch_pool_epoch_boundary() {",
          "void unified_cache::scratch_pool_epoch_boundary() {\n    if (!scratch_pool_ptr_) { return; }")]),
    # llama.cpp-1ntm: reproduces the same shape one level up -- an early
    # return inserted ahead of the hook in host_zone_settle()/zone_settle()
    # (named host_zone_reset()/zone_reset() before llama.cpp-37ba's rename).
    "the host reset hook is declared before the reservation early-return": (
        "cache",
        [("void unified_cache::host_zone_settle(host_zone_id zone) {",
          "void unified_cache::host_zone_settle(host_zone_id zone) {\n    if (!host_arena_) { return; }")]),
    "the VRAM reset hook is declared before the WEIGHT early-return": (
        "cache",
        [("void unified_cache::zone_settle(vram_zone_id zone) {",
          "void unified_cache::zone_settle(vram_zone_id zone) {\n    if (zone == vram_zone_id::WEIGHT) { return; }")]),
    # The exact "tidy" this guards against: collapsing the chain back to the
    # unconditional restore the reference had.
    "the fatal handler chains to the displaced handler, not SIG_DFL": (
        "cache",
        [("    void (*prev)(int) =", "    void (*unused_prev)(int) ="),
         ("    std::signal(sig, prev);", "    std::signal(sig, SIG_DFL);")]),
    # Reproduces the exact llama.cpp-67c2 spec-review regression: an earlier
    # occurrence of ensure_planned_arena_zones() ahead of the point-release
    # means precedes() no longer sees the release come first, the same
    # observable shape as if the release itself had been moved after it.
    # Inserting rather than relocating keeps this mutant a single-line,
    # whitespace-insensitive edit rather than a multi-line block move.
    "the oneDNN point-release precedes the growth-path re-plan": (
        "cache",
        [("        arena_attempt             = true;\n",
          "        arena_attempt             = true;\n"
          "        (void) ensure_planned_arena_zones();\n")]),
    # Reproduces the llama.cpp-ndn9 defect exactly: one published release
    # "simplified" back to the immediate zone_free() the deferred helper
    # replaced. Mutating only the weights half is deliberate -- it is the case a
    # presence-only check would miss, since the activations half would still
    # carry enqueue_deferred_zone_free().
    "the published oneDNN scratch releases are event-deferred": (
        "cache",
        [("defer_published_zone_release(onednn_weights_scratch_, onednn_weights_scratch_size_);",
          "zone_free(vram_zone_id::ONEDNN, onednn_weights_scratch_);\n"
          "            onednn_weights_scratch_ = nullptr;\n"
          "            onednn_weights_scratch_size_ = 0;")]),
    # Neuters the epoch_tracked guard's condition rather than deleting the
    # whole block: the anchor is short and comment-length-independent, so it
    # survives a longer or shorter rationale comment on the guard. The effect
    # is the same as deleting the guard -- SCRATCH/STAGING would fall through
    # to the real host_arena_->zone_reset(zone) call again.
    "SCRATCH/STAGING return before host_zone_settle reaches the real bulk reclaim": (
        "cache",
        [("if (zone_reset_audit_suppresses_reset()) {\n        return;\n    }\n\n    if (epoch_tracked) {",
          "if (zone_reset_audit_suppresses_reset()) {\n        return;\n    }\n\n    if (false && epoch_tracked) {")]),
    # Simulates exactly what llama.cpp-37ba's rename must never regress to: a
    # stray call to an old dispatcher name (here, the free-function wrapper
    # form specifically, since that is the one a bare-word-boundary check
    # would silently miss -- see the check's own comment).
    "the three old reset/boundary dispatcher names have zero production call sites": (
        "cache",
        [("void unified_cache::zone_free(vram_zone_id zone, void * ptr) {",
          "void unified_cache::zone_free(vram_zone_id zone, void * ptr) {\n"
          "    unified_cache_zone_reset(0, zone);")]),
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
        _, failed, _, _ = evaluate(sources["cache"], sources["backend"], sources["common"], sources["header"])
        if name not in failed:
            problems.append("check did not fire on its mutant: " + name)
    return problems


cache = strip_comments(Path(args.cache).read_text())
backend = strip_comments(Path(args.backend).read_text())
common = strip_comments(Path(args.common).read_text())
header = strip_comments(Path(args.header).read_text())

missing_anchors, failed, n_checks, old_dispatcher_name_survivors = evaluate(cache, backend, common, header)

for name in missing_anchors:
    print("MISSING ANCHOR: " + name)
for name in failed:
    print("FAIL: " + name)
    if name == 'the three old reset/boundary dispatcher names have zero production call sites':
        for survivor in old_dispatcher_name_survivors:
            print("  survivor: " + survivor)

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
