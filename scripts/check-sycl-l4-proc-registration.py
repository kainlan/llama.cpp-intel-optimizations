#!/usr/bin/env python3
"""L4 step 3: the three published-section procs are registered, and the state behind them has one writer.

A proc the header declares but the registration omits is a null from get_proc_address, which every
caller reads as UNSUPPORTED / GROWTH / NOT_RECORDED: the feature is silently off.  This gate keeps
that from happening, and pins the sites the behaviour hangs on, because the bodies need a backend to
run and cannot be host-tested.  Pure Python over comment-stripped source; it allocates nothing.

It checks, over ggml/include/ggml-sycl.h and ggml/src/ggml-sycl/ggml-sycl.cpp:

  * every `Proc name: "X"` in the header has an arm `strcmp(name, "X") == 0` in the registration
    that returns `(void *) X` itself, outside any build switch, and a definition of X;
  * the compute-term ledger has one writer: `.record(` is called only in
    ggml_sycl_load_record_compute_term, `.clear(` only in ggml_sycl_load_clear_compute_terms, and
    the ledger accessor is named only by those two, the late check and the test count hook;
  * the load's end call clears the ledger after the registry ended the transaction, on every path: a
    guard created after the finisher check clears on a normal exit and skips while unwinding, and the
    one catch (...) arms `after_end` as its first statement, sets its transaction before the end call
    on the finisher arm and on the recovery arm, and clears when the handler is left;
  * the backend context's destructor reads its context id once and erases its published section and then
    its held host reservation by that id before it resets its execution binding (which zeroes the registry
    key); the erase helpers take the id from their caller and never read a backend's;
  * the runtime-context transaction drops the published section on its success tail, and the
    descriptor publish stores a section only after the inner transaction succeeded;
  * the late check and the record read the open transaction (ggml_sycl_load_txn_is_open) only after the
    ledger's mutex is taken, and every function that names the ledger holds that mutex at each use of it
    (no block closes between the lock and the use); late_term_shrink_admitted has one producer;
  * the late check ends `return r.result;`; no try or catch sits between the transaction's
    runtime_kv_admitted store and its end, so a failed drop of the earlier section is not swallowed there;
    the C entry ggml_backend_sycl_set_runtime_context catches system_error, std::exception and
    everything, logs each at ERROR and rethrows none, ggml_backend_sycl_set_runtime_n_ctx delegates
    to it, and the descriptor publish calls the transaction itself (it answers EFFECT_FAILED for a
    failed drop, which the C entry swallows);
  * the fail-closed values: the late check answers NOT_RECORDED, the coverage query GROWTH and the clear 0
    on a closed module or an exception, a coverage query of an unbound context answers GROWTH, and a
    refusal logs at ERROR, a shrink and a not-open transaction at WARN (through ggml_sycl_load_ledger_log);
  * a publish for an unbound context is said at WARN, and an erase that fails says so at ERROR (the entry
    then stays until the process ends: its key is gone); the publish tail drops through the throwing set, not
    the erase;
  * (step 3c) a context's first publish reserves its host tier before L1 through
    unified_allocate_owner with the carve's request fields, installs the table once before the inner
    transaction (a rollback guard takes it back unless the section was stored), and the destructor drops
    it after the section and before the unbind; an ended execution context collects its devices under the
    binding mutex, resets the bindings, and then drops its entries BY THE ENDED ID outside the lock (no backend
    is pinned, so nothing can fail to pin and a publish after the reset finds no id to key an entry by); the
    SYCL_Host buffer type claims inside a claim scope before it reaches any allocation, and its free
    releases the claim without waiting on the host.  The pins also name each refusal the device test
    exercises: a refused or part-way reservation and a republish the held slots cannot carry are
    PLAN_REJECTED before anything is published, the table is installed (and refused by name) before anything
    is published and the section is stored last, a refused claim never falls through to the allocator, the claim in flight is owned
    by an RAII record, a table handle is cast only after its deleter is checked, and the claim scope's
    open answers a status for each way it can not open.

  * (the free-path contract) the SYCL_Host free_buffer releases the slot once, through
    tenant_claim_scope::release with event 0, and never waits on the host.  That is safe because every path
    that frees a claimed compute buffer runs a synchronize attempt of the scheduler's backends first, and
    every queue that can touch a slot is the device's one execution queue, which that synchronize drains.
    Pinned: the synchronize before each path (~llama_context first; sched_reserve_impl as a statement of
    its own on the ALLOC path, after the MEASURE branch; release_rung_buffers), the backend synchronize's
    contiguous sequence (stream, CPU-expert flush, deferred-event choice, drain), the aliasing that
    ggml_backend_sycl_context::stream(device, idx) answers the device's one execution queue (the TP queue
    or the unified cache's own) for every idx, the scheduler-level premises in the vendored ggml-backend.cpp
    and ggml-alloc.c (reserve and reserve_size synchronize first, the alloc_splits realloc synchronizes
    every backend before reserving, the automatic reserve in ggml_gallocr_alloc_graph is single-buffer only,
    the allocator has one buffer per backend) and in llama-context.cpp (the CPU backend is always appended, a
    measure-only context requires it last, the scheduler is built over every backend), set/get_tensor_async
    accepting only the device, host-compute and cpu-offload buffer types, the device not supporting the
    CpuActivation clone, and cpy_tensor_async staying unwired.  Each is pinned as a whole statement or
    sequence, so a statement under an `if`, behind a ternary arm or in another branch no longer matches.
    The contract text itself is the comment above ggml_backend_sycl_host_buffer_free_buffer.

  * (step 3d) the residency probe `ggml_backend_sycl_probe_residency` is registered like the other procs, and until
    step 1d wires the geometry its body answers only GGML_SYCL_RESIDENCY_PROBE_GEOMETRY_NOT_WIRED (or
    N_LAYER_CAP_TOO_SMALL), through named enumerators, writes `n_layer` and never `host_resident`, and says why at
    WARN.  The llama side resolves it by GGML_SYCL_PROC_PROBE_RESIDENCY into the one table, and calls it through one
    door, llama_sycl_l4_probe_residency in src/llama-context-tenant.h: no other file in src/ calls the pointer or the
    symbol.

Known limits (text-level pins; each is what the named test or review covers instead):
  * a lambda or macro that hides a release, a synchronize or a clear behind another name: the pins read
    the shapes named above, not the call graph (covered by the host-tenant-claim device test and review);
  * a tail-drop or late check inside a function the pin does not name (a statement added after the last
    pinned one): the pins anchor the first and the terminating statements, not the ones between (covered by
    the lifecycle device test's order witnesses);
  * the gate cannot tell a pinned statement that is reached from one that is dead behind a condition the
    pinned shape does not contain, such as a guard added in a caller or a function that never runs (a
    literal `if (false)`, a ternary arm and a branch move on the pinned statements ARE caught: the pin matches
    the statement's neighbours too, and these are in the mutation matrix); covered by the L6 wait_event
    consumption acceptance;
  * which scheduler a claim scope is opened on: the contract says never on the MEASURE scheduler, and
    nothing opens a scope in production yet, so there is nothing to pin (L6's constraint).

Usage:
  check-sycl-l4-proc-registration.py [--root DIR]     check the tree, then run the mutation matrix
  check-sycl-l4-proc-registration.py --no-mutations   check the tree only
  check-sycl-l4-proc-registration.py --mutations-only run the mutation matrix only
  check-sycl-l4-proc-registration.py --verbose        also name each mutant and the failure that caught it

The mutation matrix applies each RED to the source text in memory and requires the gate to fail with
that RED's message, so a check that can no longer fail is caught.
"""

import argparse
import os
import re
import sys

SIG_RECORD = r"\bbool\s+ggml_sycl_load_record_compute_term\s*\("
SIG_CLEAR = r"\bsize_t\s+ggml_sycl_load_clear_compute_terms\s*\("
SIG_LATE = r"\benum\s+ggml_sycl_late_check_result\s+ggml_backend_sycl_load_late_check\s*\("
SIG_COUNT = r'\bextern\s+"C"\s+size_t\s+ggml_backend_sycl_test_compute_term_count\s*\('
SIG_COVERAGE = r"\benum\s+ggml_sycl_tenant_coverage\s+ggml_backend_sycl_tenant_coverage\s*\("
SIG_PROBE = r"\benum\s+ggml_sycl_residency_probe_status\s+ggml_backend_sycl_probe_residency\s*\("
SIG_RECORD_EXPORT = r"\bbool\s+ggml_backend_sycl_load_record_compute_term\s*\("
SIG_RECORD_HOOK = r'\bextern\s+"C"\s+bool\s+ggml_backend_sycl_test_record_compute_term\s*\('

HEADER = "ggml/include/ggml-sycl.h"
SOURCE = "ggml/src/ggml-sycl/ggml-sycl.cpp"
CLAIM_HPP = "ggml/src/ggml-sycl/tenant-claim-scope.hpp"  # the claim record's destructor and the lowest-free walk
LLAMA = "src/llama-context.cpp"  # the scheduler's owner: read with SOURCE, as one text, for the free-path contract
COMMON_HPP = "ggml/src/ggml-sycl/common.hpp"  # stream(device, idx): the one execution queue per device
BACKEND_CPP = "ggml/src/ggml-backend.cpp"  # the scheduler's synchronize-before-free premises (vendored upstream)
ALLOC_C = "ggml/src/ggml-alloc.c"  # the allocator's automatic reserve (single-buffer only)
DOOR_H = "src/llama-context-tenant.h"  # the residency probe's one door (step 3d)
# One text: the free-path contract names all of them, and a mutation is one edit of it.
SOURCE_FILES = (SOURCE, LLAMA, CLAIM_HPP, COMMON_HPP, BACKEND_CPP, ALLOC_C, DOOR_H)
REG_FN = "ggml_backend_sycl_reg_get_proc_address"
IMPL_SIG = r"\bggml_sycl_set_runtime_context_for_model_impl\s*\("
CARRY_SIG = r"\bstatic\s+bool\s+ggml_sycl_host_tenants_carry\s*\("
INSTALL_SIG = r"\bstatic\s+bool\s+ggml_sycl_host_tenants_install\s*\("
RESERVE_SIG = r"\bstatic\s+bool\s+ggml_sycl_reserve_host_tenants\s*\("
SLOT_OF_SIG = r"\bstatic\s+ggml_sycl_host_tenant_slot\s*\*\s*ggml_sycl_host_tenant_slot_of\s*\("
SLOT_MAKE_SIG = r"\bstatic\s+ggml_sycl::kv_region_handle\s+ggml_sycl_host_tenant_slot_make\s*\("
ALLOC_SIG = r"\bstatic\s+ggml_backend_buffer_t\s+ggml_backend_sycl_host_buffer_type_alloc_buffer\s*\("
CLAIM_SLOT_SIG = r"\bstatic\s+bool\s+ggml_backend_sycl_host_buffer_claim_slot\s*\("
WRAP_SIG = r"\bstatic\s+ggml_backend_buffer_t\s+ggml_backend_sycl_host_buffer_wrap\s*\("
FREE_SIG = r"\bstatic\s+void\s+ggml_backend_sycl_host_buffer_free_buffer\s*\("
OPEN_SIG = r"\benum\s+ggml_sycl_claim_scope_status\s+ggml_backend_sycl_claim_scope_open\s*\("
CLAIMS_SIG = r"\bsize_t\s+ggml_backend_sycl_claim_scope_claims\s*\("
CLOSE_SIG = r"\bvoid\s+ggml_backend_sycl_claim_scope_close\s*\("
INSTALL_REFUSAL_TAIL = (
    "                        return GGML_SYCL_LIFECYCLE_PLAN_REJECTED;\n"
    "                    }\n"
    "                    host_tenants_installed.device = backend_ctx->device;\n")
DTOR_SIG = r"\bllama_context::~llama_context\s*\("
REIMPL_SIG = r"\bsched_reserve_result\s+llama_context::sched_reserve_impl\s*\("
SYNC_SIG = r"\bstatic\s+void\s+ggml_backend_sycl_synchronize\s*\("
SET_ASYNC_SIG = r"\bstatic\s+void\s+ggml_backend_sycl_set_tensor_async\s*\("
SUPPORTS_SIG = r"\bstatic\s+bool\s+ggml_backend_sycl_device_supports_buft\s*\("
CPY_ASYNC_DEF = "static bool ggml_backend_sycl_cpy_tensor_async("
GET_ASYNC_SIG = r"\bstatic\s+void\s+ggml_backend_sycl_get_tensor_async\s*\("
LOAD_END_SIG = r"\bggml_sycl_lifecycle_result\s+ggml_backend_sycl_model_load_end\s*\("
TXN_SIG = r"\bggml_sycl_txn_result\s+ggml_sycl_run_runtime_context_transaction\s*\("
DROP_SIG = r"\bvoid\s+drop\s*\(\s*\)\s*noexcept\s*\{"
CLAIM_WALK_SIG = r"\bstatic\s+tenant_claim_outcome\s+claim\s*\(const std::string & cohort"
CARRY_FN_SIG = r"\bstatic\s+bool\s+ggml_sycl_host_tenants_carry\s*\("
CLEAR_BIND_SIG = r"\bstatic\s+void\s+ggml_sycl_execution_clear_bindings_for_context\s*\("
DROP_ENTRIES_SIG = (r"\bstatic\s+void\s+ggml_sycl_execution_drop_context_registry_entries\s*\(uint64_t context_id,\s*"
                    r"const bool \(&devices\)\[GGML_SYCL_MAX_DEVICES\]\)\s*noexcept")
DTOR_BACKEND_SIG = r"ggml_backend_sycl_context::~ggml_backend_sycl_context\s*\("
ERASE_SECTION_SIG = r"\bstatic\s+void\s+ggml_sycl_published_section_erase\s*\("
ERASE_HOST_SIG = r"\bstatic\s+void\s+ggml_sycl_host_tenants_erase\s*\("
STREAM_SIG = r"\bqueue_ptr\s+stream\s*\(int device, int stream\)"
EXEC_QUEUE_SIG = r"\binline\s+sycl::queue \* ggml_sycl_execution_queue_for_device\s*\(int device\)"
SCHED_RESERVE_SIG = r"\bbool\s+ggml_backend_sched_reserve\s*\(ggml_backend_sched_t sched, struct ggml_cgraph \* measure_graph\)"
SCHED_RESERVE_SIZE_SIG = r"\bvoid\s+ggml_backend_sched_reserve_size\s*\(ggml_backend_sched_t sched,"
SCHED_ALLOC_SPLITS_SIG = r"\bstatic\s+bool\s+ggml_backend_sched_alloc_splits\s*\(ggml_backend_sched_t sched\)"
SCHED_SYNC_SIG = r"\bvoid\s+ggml_backend_sched_synchronize\s*\(ggml_backend_sched_t sched\)"
GALLOC_ALLOC_GRAPH_SIG = r"\bbool\s+ggml_gallocr_alloc_graph\s*\(ggml_gallocr_t galloc, struct ggml_cgraph \* graph\)"
CENTRY_SIG = r"\bvoid\s+ggml_backend_sycl_set_runtime_context\s*\("


def strip_comments(text):
    """Blank out comments, keeping every newline and the string/char literals."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            while i < n and text[i] != "\n":
                out.append(" ")
                i += 1
        elif c == "/" and nxt == "*":
            out.append("  ")
            i += 2
            while i < n and not (text[i] == "*" and i + 1 < n and text[i + 1] == "/"):
                out.append("\n" if text[i] == "\n" else " ")
                i += 1
            out.append("  ")
            i += 2
        elif c == '"' or c == "'":
            quote = c
            out.append(c)
            i += 1
            while i < n and text[i] != quote:
                if text[i] == "\\" and i + 1 < n:
                    out.append(text[i])
                    i += 1
                out.append(text[i])
                i += 1
            if i < n:
                out.append(text[i])
                i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


# The longest plain identifier in a signature is a literal every match contains, so instead of scanning the whole
# 6 MB source from a `\b` anchor (which defeats the regex engine's literal prefilter) the search visits the
# identifier's occurrences in order and searches a window around each: from _ANCHOR_REACH before it to _ANCHOR_SPAN
# after it.  The leftmost match contains some occurrence; every match that starts earlier would contain an earlier
# one, whose window was searched first, so the first window that matches answers the same match a full search
# would.  That holds while a match starts within _ANCHOR_REACH of its identifier and ends within _ANCHOR_SPAN of it
# (a declaration header, which is far shorter).  A signature with alternation, an optional or starred group, or no
# identifier of 8+ characters is searched from the start as before.  The gate runs its whole check once per mutant,
# and the full scan put it at its ctest TIMEOUT.
_ANCHOR_REACH = 4096
_ANCHOR_SPAN = 16384
_ANCHOR_CACHE = {}


def _signature_anchor(signature_re):
    if signature_re not in _ANCHOR_CACHE:
        anchor = None
        if "|" not in signature_re and not re.search(r"\)[*?]", signature_re):
            words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", re.sub(r"\\.", " ", signature_re))
            longest = max(words, key=len) if words else ""
            anchor = longest if len(longest) >= 8 else None
        _ANCHOR_CACHE[signature_re] = anchor
    return _ANCHOR_CACHE[signature_re]


def function_body(text, signature_re):
    """The brace-balanced body of the first definition whose header matches signature_re, or None."""
    pattern = re.compile(signature_re + r"[^;{]*\{")
    anchor = _signature_anchor(signature_re)
    if anchor is None:
        m = pattern.search(text)
    else:
        m = None
        at = text.find(anchor)
        while at >= 0 and m is None:
            m = pattern.search(text, max(0, at - _ANCHOR_REACH), at + len(anchor) + _ANCHOR_SPAN)
            at = text.find(anchor, at + 1)
    if not m:
        return None
    i = m.end() - 1
    depth = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == '"' or c == "'":
            quote = c
            i += 1
            while i < n and text[i] != quote:
                if text[i] == "\\":
                    i += 1
                i += 1
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[m.start():i + 1]
        i += 1
    return None


def min_depth(text):
    """The lowest running brace depth reached while reading `text` from depth 0: a `}` that closes a block the text
    did not open takes it below 0.  String and character literals are skipped."""
    depth = 0
    low = 0
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == '"' or c == "'":
            quote = c
            i += 1
            while i < n and text[i] != quote:
                if text[i] == "\\":
                    i += 1
                i += 1
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            low = min(low, depth)
        i += 1
    return low


def net_depth(text):
    """Opened minus closed braces in `text`; string and character literals are skipped."""
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == '"' or c == "'":
            quote = c
            i += 1
            while i < n and text[i] != quote:
                if text[i] == "\\":
                    i += 1
                i += 1
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    return depth


def header_proc_names(header_raw):
    return re.findall(r'Proc name:\s*"([A-Za-z0-9_]+)"', header_raw)


def impl_first_publish_wrong(source):
    """The reservation is made before L1 and the table is installed once, after it and before the inner
    transaction (so a refused install is a refusal with nothing published)."""
    impl = function_body(source, r"\bggml_sycl_set_runtime_context_for_model_impl\s*\(")
    if impl is None:
        return True
    r = impl.find("ggml_sycl_reserve_host_tenants(")
    l1 = impl.find("std::lock_guard<std::mutex> lock(g_tensor_inventory_mutex)")
    inner = impl.find("ggml_sycl_run_runtime_context_transaction(")
    inst = impl.find("ggml_sycl_host_tenants_install(")
    return not (0 <= r < inst < l1 and inst < inner and impl.count("ggml_sycl_host_tenants_install(") == 1)


def function_span(text, signature_re):
    """(start, end) of the first definition whose header matches signature_re, or None."""
    body = function_body(text, signature_re)
    if body is None:
        return None
    start = text.find(body)
    return (start, start + len(body))


def pin(fails, text, regex, message):
    """One required shape: `regex` must match `text` (None counts as no match)."""
    if text is None or not re.search(regex, text):
        fails.append(message)


def host_tier_pins(source, alloc, freeb, reserve, fails):
    """What the host tier's behaviour rests on, at the text level (the device test runs it; this is what keeps
    a refactor from quietly removing the shape the test's answers depend on).  Each pin names the refusal or
    ownership claim it guards."""
    impl = function_body(source, r"\bggml_sycl_set_runtime_context_for_model_impl\s*\(")
    # the reservation: a refused allocation, a wrong tier, memory that is not host memory, and an
    # exception are all refusals of the publish -- never an ignored failure and never a published section
    if reserve is not None:
        pin(fails, reserve, r"req\.size = \(size_t\) e\.slot_bytes;",
            "L4 host tier: the host carve is not the slot's full byte count")
        pin(fails, reserve, r"info == nullptr \|\| info->tier != GGML_SYCL_CONTEXT_COHORT_TIER_HOST_PINNED\) \{[^}]*return false;",
            "L4 host tier: the reservation does not refuse a cohort that is not host-pinned")
        pin(fails, reserve, r"if \(!allocation\) \{[^}]*return false;\s*\}",
            "L4 host tier: a refused carve does not refuse the reservation")
        pin(fails, reserve, r"if \(!resolved\.ptr \|\| resolved\.on_device\) \{[^}]*return false;\s*\}",
            "L4 host tier: the reservation accepts a carve that did not resolve to host memory")
        pin(fails, reserve, r"table->add\(info->name, e\.slot_index,\s",
            "L4 host tier: the reservation does not add its carves to the table")
        pin(fails, reserve, r"\n    out = std::move\(table\);\n    return true;\n\}$",
            "L4 host tier: the reservation does not hand the table out only on success")
        if re.search(r"\b(try|catch)\b", reserve):
            fails.append("L4 host tier: the reservation swallows or converts an exception itself (the publish's catch answers "
                         "EFFECT_FAILED, with the table dropped)")
    # the publish: reserve only when no table is held, carry check otherwise, both refusals are PLAN_REJECTED
    pin(fails, impl,
        r"if \(held == nullptr\) \{\s*if \(!ggml_sycl_reserve_host_tenants\(backend_ctx, \*section, host_tenants, refusal\)\) \{"
        r"[^}]*return GGML_SYCL_LIFECYCLE_PLAN_REJECTED;\s*\}\s*\} else if \(!ggml_sycl_host_tenants_carry\(\*held, \*section, refusal\)\) \{"
        r"[^}]*return GGML_SYCL_LIFECYCLE_PLAN_REJECTED;\s*\}",
        "L4 host tier: the publish does not reserve only without a held table, or does not refuse a refused reservation or "
        "a republish the held slots cannot carry")
    pin(fails, impl,
        r"if \(host_tenants\) \{\s*uint64_t installed_id = 0;\s*"
        r"if \(!ggml_sycl_host_tenants_install\(backend_ctx, host_tenants, section->tenant_key, installed_id\)\) \{"
        r"\s*GGML_LOG_ERROR\([\s\S]*?\);\s*return GGML_SYCL_LIFECYCLE_PLAN_REJECTED;\s*\}\s*"
        r"host_tenants_installed\.device = backend_ctx->device;\s*host_tenants_installed\.id = installed_id;\s*\}",
        "L4 host tier: the install result is not a refusal with nothing published, or the table it installed does not arm "
        "the rollback guard by the device and the id it was keyed with")
    pin(fails, impl, r"ggml_sycl_published_section_set\(backend_ctx, section\);\s*host_tenants_installed\.keep\(\);",
        "L4 host tier: the section is stored without keeping the installed table (or the table is kept before the store)")
    pin(fails, impl, r"std::shared_ptr<ggml_sycl::kv_tenant_slots> host_tenants;\s*ggml_sycl_host_tenants_install_guard host_tenants_installed;",
        "L4 host tier: the publish holds no rollback guard for the table it installs")
    guard_fn = function_body(source, r"\bstruct\s+ggml_sycl_host_tenants_install_guard\b")
    pin(fails, guard_fn, r"~ggml_sycl_host_tenants_install_guard\(\) \{\s*if \(id != 0\) \{\s*ggml_sycl_host_tenants_erase\(device, id\);\s*\}\s*\}"
                         r"\s*void keep\(\) \{ id = 0; \}",
        "L4 host tier: the install guard does not take the table back unless it was kept")
    install_fn = function_body(source, r"\bstatic\s+bool\s+ggml_sycl_host_tenants_install\s*\(")
    if install_fn is not None and install_fn.count("table.reset()") != 1:
        fails.append("L4 host tier: the table is dropped from the caller before the registry took it")
    pin(fails, install_fn,
        r"if \(id == 0 \|\| !ggml_sycl_kv_region_registry\(ctx->device\)\.install_tenant_slots\(id, table, tenant_key\)\) \{"
        r"\s*return false;\s*\}\s*table\.reset\(\);\s*installed_id = id;\s*return true;",
        "L4 host tier: the table is dropped from the caller before the registry took it")
    pin(fails, function_body(source, r"\bstatic\s+bool\s+ggml_sycl_host_tenants_carry\s*\("),
        r"held_bytes == 0\) \{[^}]*return false;\s*\}\s*if \(\(size_t\) e\.slot_bytes > held_bytes\) \{[^}]*return false;",
        "L4 host tier: a republish is carried by a missing or smaller held slot")
    if not re.search(r"\bstatic\s+void\s+ggml_sycl_host_tenants_erase\s*\(int device, uint64_t context_id\) noexcept \{", source):
        fails.append("L4 host tier: the drop of the host reservation may throw (not noexcept) or does not take its id")
    if not re.search(r"\bstatic\s+void\s+ggml_sycl_published_section_erase\s*\(int device, uint64_t context_id\) noexcept \{", source):
        fails.append("L4 section: the drop of the published section may throw (not noexcept) or does not take its id")
    # the slot's identity: a handle of any other type is refused, not reinterpreted
    pin(fails, function_body(source, r"\bstatic\s+ggml_sycl_host_tenant_slot\s*\*\s*ggml_sycl_host_tenant_slot_of\s*\("),
        r"std::get_deleter<ggml_sycl_host_tenant_slot_deleter>\(handle\) == nullptr\) \{\s*return nullptr;",
        "L4 host tier: a table handle is cast to a slot without checking that the slot's deleter made it")
    pin(fails, function_body(source, r"\bstatic\s+ggml_sycl::kv_region_handle\s+ggml_sycl_host_tenant_slot_make\s*\("),
        r"return ggml_sycl::kv_region_handle\(slot, ggml_sycl_host_tenant_slot_deleter\{\}\);",
        "L4 host tier: a slot is made without the deleter its reader checks")
    # the buffer type: a refused claim is a refusal, never a fall-through to the allocator; the claim is owned
    if alloc is not None:
        pin(fails, alloc,
            r"if \(ggml_sycl::tenant_claim_scope::active\(\)\) \{[^}]*if \(!ggml_backend_sycl_host_buffer_claim_slot\([^)]*\)\) \{"
            r"\s*return nullptr;\s*\}\s*return ggml_backend_sycl_host_buffer_wrap\([^;]*std::move\(claim\)\);\s*\}\s*auto \* exact_cache",
            "L4 host tier: a refused claim in a scope falls through to the legacy allocator (or the claim branch no longer "
            "ends in its own return)")
    claim_slot = function_body(source, r"\bstatic\s+bool\s+ggml_backend_sycl_host_buffer_claim_slot\s*\(")
    if claim_slot is None:
        fails.append("L4 host tier: ggml_backend_sycl_host_buffer_claim_slot not found")
    else:
        pin(fails, claim_slot, r"auto made = std::make_unique<ggml_sycl::tenant_claim>\(\);",
            "L4 host tier: the claim in flight is not owned by an RAII record (a refusal or an exception would leak the slot)")
        pin(fails, claim_slot, r"if \(!slot \|\| !resolved\.ptr \|\| resolved\.on_device\) \{[^}]*return false;\s*\}",
            "L4 host tier: a claim over memory that is not host memory is served")
        pin(fails, claim_slot, r"claim = std::move\(made\);\s*return true;\s*\}$",
            "L4 host tier: the claim is handed to the buffer other than as the last step before success")
        if re.search(r"\brelease\(", claim_slot):
            fails.append("L4 host tier: claim_slot releases a claim by hand (the RAII record does it)")
    wrap = function_body(source, r"\bstatic\s+ggml_backend_buffer_t\s+ggml_backend_sycl_host_buffer_wrap\s*\(")
    if wrap is None:
        fails.append("L4 host tier: ggml_backend_sycl_host_buffer_wrap not found")
    else:
        pin(fails, wrap, r"std::unique_ptr<ggml_sycl::tenant_claim> claim\) \{",
            "L4 host tier: wrap does not take its claim by value (an early return would leave it live)")
        pin(fails, wrap, r"new \(std::nothrow\) sycl_host_buf_ctx\{[^;]*\};\s*if \(!ctx\) \{\s*ggml_backend_buffer_free\(buffer\);\s*return nullptr;",
            "L4 host tier: a failed context allocation does not free the CPU buffer it built and answer null")
        pin(fails, wrap, r"if \(!ggml_backend_buffer_set_type\(buffer, buft\)\) \{\s*ggml_backend_buffer_free\(buffer\);\s*delete ctx;\s*return nullptr;",
            "L4 host tier: a refused set_type does not free the buffer and the context that owns the claim")
        if re.search(r"\brelease\(", wrap):
            fails.append("L4 host tier: wrap releases a claim by hand (the context's destructor does it)")
    # free: no host wait (event chain), and the claim goes back through the tenant release
    if freeb is not None:
        if re.search(r"wait_and_throw|\.wait\(|\bwait\(|ggml_backend_sycl_host_buffer_sync", freeb):
            fails.append("L4 host tier: the SYCL_Host free_buffer waits on the host (a queue wait is not the slot's last event)")
        pin(fails, freeb, r"if \(ctx->claim\) \{\s*\(void\) ggml_sycl::tenant_claim_scope::release\(\*ctx->claim, 0\);\s*ctx->claim\.reset\(\);\s*\}",
            "L4 host tier: the SYCL_Host free_buffer does not release then drop its claim as the first step")
    # a claim record's destructor says a failed release; the walk reports the real index and refuses at the top
    pin(fails, function_body(source, DROP_SIG), r"catch \(\.\.\.\) \{\s*std::fprintf\(stderr, \"WARN: \[CLAIM-SCOPE\]",
        "L4 claim scope: the claim record's destructor swallows a failed release silently (no WARN)")
    walk = function_body(source, CLAIM_WALK_SIG)
    pin(fails, walk, r"return \{ tenant_claim_status::NO_SLOT, !seen \? 0 : \(last == UINT32_MAX \? last : last \+ 1\), 0 \};",
        "L4 claim scope: an exhausted cohort does not name the index past its last slot (0 only for a cohort with none)")
    if walk is None or walk.count("index == UINT32_MAX") != 2:
        fails.append("L4 claim scope: the claim walk can wrap its index past UINT32_MAX instead of refusing")
    # the reservation and the carry look at the host tier only (device -1), the carve is a host-compute staging slot
    for fn, what in ((function_body(source, RESERVE_SIG), "reservation"), (function_body(source, CARRY_FN_SIG), "carry check")):
        pin(fails, fn, r"for \(const ggml_sycl::runtime_context_tenant & e : section\.tenants\) \{\s*if \(e\.device != -1\) \{\s*continue;\s*\}",
            "L4 host tier: the %s does not skip every element that is not host-tier (device -1)" % what)
    pin(fails, reserve, r"req\.intent\.role = ggml_sycl::alloc_role::STAGING;",
        "L4 host tier: the host carve is not a STAGING-role request")
    pin(fails, reserve, r"req\.device = ctx->device;",
        "L4 host tier: the host carve is not requested for the context's device")
    # the claim scope's C entries: every refusal is a named status, and a nested refusal is the first answer
    open_fn = function_body(source, r"\benum\s+ggml_sycl_claim_scope_status\s+ggml_backend_sycl_claim_scope_open\s*\(")
    if open_fn is None:
        fails.append("L4 claim scope: ggml_backend_sycl_claim_scope_open not found")
    else:
        pin(fails, open_fn, r"sycl_module_mutation_guard module_guard;\s*if \(!module_guard\) \{\s*return GGML_SYCL_CLAIM_SCOPE_FAILED;\s*\}",
            "L4 claim scope: open does not answer FAILED on a closed module")
        pin(fails, open_fn, r"if \(ctx->device < 0 \|\| ctx->device >= GGML_SYCL_MAX_DEVICES\) \{\s*return GGML_SYCL_CLAIM_SCOPE_INVALID_BACKEND;",
            "L4 claim scope: open does not refuse a context with no valid device as INVALID_BACKEND")
        pin(fails, open_fn, r"const uint64_t id = ggml_sycl_context_execution_id\(ctx\);\s*if \(id == 0\) \{\s*return GGML_SYCL_CLAIM_SCOPE_NO_RESERVATION;",
            "L4 claim scope: open does not answer NO_RESERVATION for a context not bound to an execution context")
        pin(fails, open_fn, r"^[^{]*\{\s*if \(scope == nullptr\) \{\s*return GGML_SYCL_CLAIM_SCOPE_FAILED;\s*\}\s*\*scope = nullptr;",
            "L4 claim scope: open does not refuse a null out pointer first and clear it before anything else")
        pin(fails, open_fn,
            r"ggml_backend_dev_backend_reg\(backend->device\) != ggml_backend_sycl_reg\(\)\) \{\s*return GGML_SYCL_CLAIM_SCOPE_INVALID_BACKEND;",
            "L4 claim scope: open does not refuse a foreign backend")
        pin(fails, open_fn,
            r"if \(ggml_sycl::tenant_claim_scope::active\(\)\) \{[^}]*return GGML_SYCL_CLAIM_SCOPE_NESTED;\s*\}\s*const uint64_t id =",
            "L4 claim scope: open does not answer NESTED before it reads the context's table")
        pin(fails, open_fn,
            r"open_status::OPENED:\s*\*scope = opened;\s*return GGML_SYCL_CLAIM_SCOPE_OPENED;\s*case \S*open_status::NO_TABLE:\s*"
            r"return GGML_SYCL_CLAIM_SCOPE_NO_RESERVATION;\s*case \S*open_status::NESTED:\s*return GGML_SYCL_CLAIM_SCOPE_NESTED;",
            "L4 claim scope: open does not map its three outcomes to OPENED, NO_RESERVATION and NESTED")
        pin(fails, open_fn, r"catch \(\.\.\.\) \{\s*return GGML_SYCL_CLAIM_SCOPE_FAILED;\s*\}\s*\}$",
            "L4 claim scope: open's catch does not answer FAILED")
    claims_fn = function_body(source, r"\bsize_t\s+ggml_backend_sycl_claim_scope_claims\s*\(")
    pin(fails, claims_fn,
        r"if \(scope == nullptr\) \{\s*return 0;\s*\}\s*size_t claims = 0;\s*if \(!ggml_sycl::tenant_claim_scope::claims_made\(",
        "L4 claim scope: claims reads the scope without the thread's open-scope check")
    close_fn = function_body(source, r"\bvoid\s+ggml_backend_sycl_claim_scope_close\s*\(")
    pin(fails, close_fn, r"if \(!ggml_sycl::tenant_claim_scope::close\(",
        "L4 claim scope: close does not check that the scope is the thread's open one")


def free_path_pins(source, freeb, fails):
    """The free-path contract of the SYCL_Host tenant buffer (llama.cpp-moua, the L4 free_buffer ruling): its free
    releases the slot with event 0 and never waits on the host, so the synchronize that makes that safe is
    pinned at each place a scheduler's compute buffer can be freed, and the queues that synchronize drains are
    pinned at the backend.  llama-context.cpp is read with the backend source as one text.  The pins check the
    shape the argument rests on; the argument itself is the comment above ggml_backend_sycl_host_buffer_free_buffer.

      * release goes only through tenant_claim_scope::release, once, with event 0
      * ~llama_context synchronizes (every backend) before its members, the scheduler among them, are destroyed
      * sched_reserve_impl synchronizes as a statement of its own on the ALLOC path (not under an `if`, not in the
        MEASURE branch) before it touches the scheduler, and so does release_rung_buffers
      * ggml_backend_sycl_synchronize's contiguous sequence: stream 0, the CPU-expert flush, the deferred-event
        choice, the drain
      * ggml_backend_sycl_context::stream(device, idx) answers the one execution queue for every idx
      * the scheduler's own premises (ggml-backend.cpp, ggml-alloc.c, llama-context.cpp), see check()
      * nothing but the SYCL device, host-compute and cpu-offload buffer types reaches set/get_tensor_async, the
        CpuActivation clone is not a buffer type the device supports, and cpy_tensor_async is not wired
    """
    if freeb is not None:
        if len(re.findall(r"\brelease\(", freeb)) != 1 or re.search(r"unified_free|zone_free|sycl::free|\bfree\(", freeb):
            fails.append("L4 free path: the SYCL_Host free_buffer releases other than once through tenant_claim_scope::release")
    dtor = function_body(source, DTOR_SIG)
    pin(fails, dtor, r"^[^{]*\{\s*synchronize\(\);",
        "L4 free path: ~llama_context does not synchronize() first, before its scheduler's compute buffers are freed")
    # sched_reserve_impl: the call is a statement of the function (not under an `if`, not in the MEASURE branch), it
    # follows the MEASURE branch and the progress line directly, and nothing touches the scheduler before it.
    reimpl = function_body(source, REIMPL_SIG)
    pin(fails, reimpl,
        r"\n    if \(mode == sched_reserve_mode::MEASURE\) \{\s*return sched_measure_impl\(state\);\s*\}\s*"
        r"(?:LLAMA_LOG_INFO\([^;]*\);\s*)?synchronize\(\);\s*const int64_t t_start_us",
        "L4 free path: sched_reserve_impl does not synchronize() unconditionally on the ALLOC path (a statement of its own, "
        "after the MEASURE branch and before anything else), where it replaces the scheduler")
    if reimpl is not None:
        sy = reimpl.find("synchronize();")
        first = min([i for i in (reimpl.find("sched.reset("), reimpl.find("ggml_backend_sched_")) if i >= 0] or [-1])
        if sy < 0 or first < 0 or sy > first:
            fails.append("L4 free path: sched_reserve_impl does not synchronize() before it first touches the scheduler")
    pin(fails, source, r"auto release_rung_buffers = \[&\]\(\) \{\s*synchronize\(\);\s*for \(auto & res : gf_res_prev\)",
        "L4 free path: release_rung_buffers does not synchronize() before it frees the rung's buffers")
    # the backend's synchronize: the one contiguous sequence is pinned whole, so no statement of it can sit under a
    # condition, in a ternary arm or behind another name and still match: the stream, the CPU-expert flush (unless
    # the test flag skips it), the deferred-event choice, the drain
    sync = function_body(source, SYNC_SIG)
    pin(fails, sync,
        r"const queue_ptr stream = sycl_ctx->stream\(sycl_ctx->device, 0\);\s*"
        r"if \(!ggml_sycl_cpu_tg_synchronize_flush_skipped_for_test\(\)\) \{\s*ggml_sycl_cpu_tg_flush_pending\(\);\s*\}\s*"
        r"const bool use_deferred_decode_event =\s*"
        r"sycl_ctx->last_graph_event\.has_value\(\) && sycl_ctx->last_graph_event_deferred_decode;\s*"
        r"auto err = use_deferred_decode_event \? CHECK_TRY_ERROR\(sycl_ctx->last_graph_event->wait_and_throw\(\)\) :\s*"
        r"CHECK_TRY_ERROR\(stream->wait_and_throw\(\)\);",
        "L4 free path: the backend synchronize no longer drains, in sequence and unconditionally, stream 0 or the deferred "
        "decode event after the CPU-expert flush")
    # one execution queue per device: stream(device, idx) answers it for EVERY idx, and it is the TP queue or the
    # unified cache's own queue.  This is what makes "the queues the synchronize does not drain" an empty set.
    stream_fn = function_body(source, STREAM_SIG)
    pin(fails, stream_fn,
        r"^[^{]*\{\s*if \(sycl::queue \* execution_queue = ggml_sycl_execution_queue_for_device\(device\)\) \{\s*"
        r"if \(qptrs\[device\]\[stream\] != execution_queue\) \{\s*qptrs\[device\]\[stream\] = execution_queue;\s*"
        r"GGML_SYCL_DEBUG\([^;]*\);\s*\}\s*return execution_queue;\s*\}",
        "L4 free path: ggml_backend_sycl_context::stream(device, idx) no longer answers the device's one execution queue "
        "for every idx (a second queue per device is one the synchronize does not drain)")
    pin(fails, function_body(source, EXEC_QUEUE_SIG),
        r"^[^{]*\{\s*if \(sycl::queue \* tp_queue = ggml_sycl_get_tp_queue\(device\)\) \{\s*return tp_queue;\s*\}\s*"
        r"if \(ggml_sycl::unified_cache \* cache = ggml_sycl::get_existing_unified_cache_for_device\(device\)\) \{\s*"
        r"return &cache->get_queue\(\);\s*\}\s*return nullptr;\s*\}$",
        "L4 free path: the device's execution queue is no longer the TP queue or the unified cache's own queue")
    # the upstream premises the synchronize-before-free argument rests on.  They are vendored files: a rebase that
    # changes one is the event this catches.
    pin(fails, function_body(source, SCHED_SYNC_SIG),
        r"GGML_ASSERT\(sched\);\s*for \(int i = 0; i < sched->n_backends; i\+\+\) \{\s*ggml_backend_synchronize\(sched->backends\[i\]\);\s*\}",
        "L4 free path: ggml_backend_sched_synchronize no longer synchronizes every backend of the scheduler")
    pin(fails, function_body(source, SCHED_RESERVE_SIG),
        r"GGML_ASSERT\(\(int\)sched->hash_set\.size >= measure_graph->n_nodes \+ measure_graph->n_leafs\);\s*"
        r"ggml_backend_sched_synchronize\(sched\);\s*ggml_backend_sched_split_graph\(sched, measure_graph\);\s*"
        r"if \(!ggml_gallocr_reserve_n\(sched->galloc,",
        "L4 free path: ggml_backend_sched_reserve no longer synchronizes the scheduler before ggml_gallocr_reserve_n frees "
        "the old buffers")
    pin(fails, function_body(source, SCHED_RESERVE_SIZE_SIG),
        r"ggml_backend_sched_reset\(sched\);\s*ggml_backend_sched_synchronize\(sched\);\s*"
        r"ggml_backend_sched_split_graph\(sched, measure_graph\);\s*ggml_gallocr_reserve_n_size\(",
        "L4 free path: ggml_backend_sched_reserve_size no longer synchronizes the scheduler before it reserves")
    splits = function_body(source, SCHED_ALLOC_SPLITS_SIG)
    pin(fails, splits,
        r"\}\s*for \(int i = 0; i < sched->n_backends; i\+\+\) \{\s*ggml_backend_synchronize\(sched->backends\[i\]\);\s*\}\s*"
        r"if \(!ggml_gallocr_reserve_n\(sched->galloc,",
        "L4 free path: the reallocation in ggml_backend_sched_alloc_splits no longer synchronizes every backend before "
        "ggml_gallocr_reserve_n")
    if splits is not None and splits.count("ggml_gallocr_reserve_n(") != 1:
        fails.append("L4 free path: ggml_backend_sched_alloc_splits reserves in more than one place")
    pin(fails, function_body(source, GALLOC_ALLOC_GRAPH_SIG),
        r"if \(ggml_gallocr_needs_realloc\(galloc, graph\)\) \{\s*if \(galloc->n_buffers == 1\) \{[^}]*?"
        r"if \(!ggml_gallocr_reserve\(galloc, graph\)\) \{\s*return false;\s*\}\s*\} else \{[^}]*?return false;\s*\}\s*\}",
        "L4 free path: the automatic reserve in ggml_gallocr_alloc_graph is no longer single-buffer only (it reallocates "
        "without a synchronize)")
    pin(fails, source, r"sched->galloc = ggml_gallocr_new_n\(sched->bufts, n_backends\);",
        "L4 free path: the scheduler's allocator is no longer built with one buffer per backend")
    # llama_context always appends the CPU backend, so a SYCL scheduler has two or more buffers and the single-buffer
    # automatic reserve above is unreachable for it: a normal context appends it, a measure-only one requires it last
    pin(fails, source,
        r"backend_cpu = ggml_backend_init_by_type\(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr\);\s*"
        r"if \(backend_cpu == nullptr\) \{\s*throw std::runtime_error\(\"failed to initialize CPU backend\"\);\s*\}\s*"
        r"backends\.emplace_back\(backend_cpu\);",
        "L4 free path: llama_context no longer appends the CPU backend to a normal context's backends")
    pin(fails, source,
        r"backend_cpu = backends\.back\(\)\.get\(\);\s*"
        r"if \(ggml_backend_dev_type\(ggml_backend_get_device\(backend_cpu\)\) != GGML_BACKEND_DEVICE_TYPE_CPU\) \{\s*"
        r"throw std::runtime_error\(\"the last backend of a measure-only context must be the CPU backend\"\);",
        "L4 free path: a measure-only context no longer requires the CPU backend last")
    if reimpl is not None:
        built = len(re.findall(r"ggml_backend_sched_new\(", reimpl))
        over_all = len(re.findall(
            r"state\.sched\.reset\(ggml_backend_sched_new\(backend_ptrs\.data\(\), backend_buft\.data\(\), backend_ptrs\.size\(\),\s*max_nodes,",
            reimpl))
        if built == 0 or built != over_all:
            fails.append("L4 free path: a scheduler sched_reserve_impl builds is no longer built over every backend of the "
                         "context (backend_ptrs)")
    for sig, what in ((SET_ASYNC_SIG, "set_tensor_async"), (GET_ASYNC_SIG, "get_tensor_async")):
        body = function_body(source, sig)
        pin(fails, body, r"GGML_ASSERT\(\(buf->buft == ggml_backend_sycl_buffer_type\(sycl_ctx->device\) \|\|\s*"
                         r"buf->buft == ggml_backend_sycl_host_compute_buffer_type\(sycl_ctx->device\) \|\|\s*"
                         r"buf->buft == ggml_backend_sycl_cpu_offload_compute_buffer_type\(sycl_ctx->device\)\) &&",
            "L4 free path: %s does not assert exactly the SYCL device, host-compute and cpu-offload buffer types" % what)
        if body is not None and "cpu_activation_buffer_type" in body:
            fails.append("L4 free path: %s accepts the CpuActivation buffer type" % what)
    supports = function_body(source, SUPPORTS_SIG)
    if supports is None or "cpu_activation_buffer_type" in supports:
        fails.append("L4 free path: the device supports the CpuActivation buffer type (a SYCL op could read a CPU activation)")
    if source.count("ggml_backend_sycl_cpy_tensor_async") != 1:
        fails.append("L4 free path: cpy_tensor_async is referenced beyond its definition (the backend's interface wires it)")


PROBE_STATUS_RETURN = re.compile(r"\breturn\s+([^;]*);")


def probe_pins(source, fails):
    """The residency probe (step 3d).  Until step 1d fills the zone geometry the proc cannot answer, and the one
    thing it must never do is look like it did: its body returns a named GEOMETRY_NOT_WIRED (or the cap refusal),
    writes the layer count and no host_resident byte, says why at WARN, and the llama side reaches it through one
    table entry and one door."""
    body = function_body(source, SIG_PROBE)
    if body is None:
        fails.append("L4 probe: ggml_backend_sycl_probe_residency not found")
        return
    if "host_resident" in body:
        fails.append("L4 probe: the proc touches host_resident before step 1d wires the geometry")
    allowed = {"GGML_SYCL_RESIDENCY_PROBE_GEOMETRY_NOT_WIRED", "GGML_SYCL_RESIDENCY_PROBE_N_LAYER_CAP_TOO_SMALL",
               "GGML_SYCL_RESIDENCY_PROBE_FOREIGN_BACKEND", "GGML_SYCL_RESIDENCY_PROBE_INVALID",
               "GGML_SYCL_RESIDENCY_PROBE_NOT_ANSWERED"}
    returns = [m.group(1).strip() for m in PROBE_STATUS_RETURN.finditer(body)]
    if not returns or any(r not in allowed for r in returns):
        fails.append("L4 probe: the proc returns something other than a named refusal enumerator (found %s)" % returns)
    if "GGML_SYCL_RESIDENCY_PROBE_OK" in body or "GGML_SYCL_LIFECYCLE_" in body:
        fails.append("L4 probe: the proc names OK or a lifecycle result before step 1d wires the geometry")
    if returns.count("GGML_SYCL_RESIDENCY_PROBE_GEOMETRY_NOT_WIRED") != 1 or not re.search(
            r"return GGML_SYCL_RESIDENCY_PROBE_GEOMETRY_NOT_WIRED;\s*\}$", body):
        fails.append("L4 probe: the proc does not end by answering GEOMETRY_NOT_WIRED")
    # the caller's struct is gated on what it declared, by the one helper the host test pins, before a byte of it is
    # written: nothing of `out` is touched ahead of the gate, and the gate's refusal is the next thing after it
    if not re.search(r"\{\s*\(void\)\s*model;\s*if \(!ggml_sycl::residency_probe_out_declared\(out\)\) \{[^{}]*"
                     r"return GGML_SYCL_RESIDENCY_PROBE_INVALID;\s*\}\s*out->n_layer = 0;", body):
        fails.append("L4 probe: the proc does not gate the caller's result struct on its declared size and version "
                     "(residency_probe_out_declared) before it writes any of it")
    # the proc's own refusal arms, each by name, in the order they run: a foreign backend, a zero shape, a malformed
    # descriptor, then a buffer smaller than the answer (the cap check needs the layer count the parse wrote)
    arms = [
        (r"if \(!backend \|\| !backend->context \|\| !ggml_backend_is_sycl\(backend\) \|\| !backend->device \|\|\s*"
         r"ggml_backend_dev_backend_reg\(backend->device\) != ggml_backend_sycl_reg\(\)\) \{[^{}]*"
         r"return GGML_SYCL_RESIDENCY_PROBE_FOREIGN_BACKEND;\s*\}", "does not refuse a foreign backend"),
        (r"if \(n_ctx == 0 \|\| n_ubatch == 0 \|\| n_seq_max == 0\) \{[^{}]*"
         r"return GGML_SYCL_RESIDENCY_PROBE_INVALID;\s*\}", "does not refuse a zero shape"),
        (r"if \(desc != nullptr\) \{[^}]*?ggml_sycl::parse_runtime_context_desc\(\s*desc,\s*geometry,[^;]*;\s*"
         r"if \(status != ggml_sycl::runtime_context_desc_status::OK\) \{[^{}]*"
         r"return GGML_SYCL_RESIDENCY_PROBE_INVALID;\s*\}", "does not refuse a malformed descriptor"),
        (r"if \(out->n_layer_cap < out->n_layer\) \{[^{}]*return GGML_SYCL_RESIDENCY_PROBE_N_LAYER_CAP_TOO_SMALL;\s*\}",
         "does not refuse a buffer smaller than the answer"),
    ]
    at = -1
    for pattern, what in arms:
        m = re.search(pattern, body)
        if not m:
            fails.append("L4 probe: the proc %s" % what)
        elif m.start() < at:
            fails.append("L4 probe: the proc %s before an earlier arm has run" % what)
        else:
            at = m.start()
    if "out->n_layer = (uint32_t) parsed.kv.layers.size();" not in body:
        fails.append("L4 probe: the proc does not write the layer count it was asked about")
    if not re.search(r'GGML_LOG_WARN\(\s*"\[RESIDENCY-PROBE\][^"]*"(?:[^;"]|"[^"]*")*;\s*return GGML_SYCL_RESIDENCY_PROBE_GEOMETRY_NOT_WIRED;', body):
        fails.append("L4 probe: the proc does not say why it did not answer, at WARN")
    # the llama side: one table entry through the reg, one door
    fill = ("procs.probe_residency = reinterpret_cast<decltype(procs.probe_residency)>("
            "llama_context_sycl_proc_addr(dev, GGML_SYCL_PROC_PROBE_RESIDENCY));")
    if re.sub(r"\s+", "", source).count(re.sub(r"\s+", "", fill)) != 1:
        fails.append("L4 probe: the llama table does not resolve the probe through the reg by its macro, once")
    calls = re.findall(r"(?:\.|->)\s*probe_residency\s*\)?\s*\(", source)
    if len(calls) != 1:
        fails.append("L4 probe: the proc pointer is called %d times in the sources the gate reads (the door has one call)"
                     % len(calls))
    door = function_body(source, r"\binline\s+ggml_sycl_residency_probe_status\s+llama_sycl_l4_probe_residency\s*\(")
    if door is None or "procs.probe_residency(" not in door:
        fails.append("L4 probe: the one call is not inside llama_sycl_l4_probe_residency")


DOOR_SCAN_DIR = "src"
DOOR_SYMBOL = re.compile(r"(?<!decltype\(&)\bggml_backend_sycl_probe_residency\b")
DOOR_CALL = re.compile(r"(?:\.|->)\s*probe_residency\s*\)?\s*\(")


def load_term_pins(header_raw, source, fails):
    """The load's compute-term record (llama.cpp-p6i0).  The record export is the production caller of the one
    ledger writer: it reaches the writer, under the module guard, and never the ledger itself; the writer is called
    only by it and by the private test hook, and is no longer marked unused."""
    names = header_proc_names(header_raw)
    for name in ("ggml_backend_sycl_load_record_compute_term",):
        if name not in names:
            fails.append("L4 load terms: the header does not name the proc %s" % name)
    if re.search(r"\[\[maybe_unused\]\]\s*static\s+bool\s+ggml_sycl_load_record_compute_term\s*\(", source):
        fails.append("L4 load terms: the ledger writer is still marked [[maybe_unused]]")
    rec = function_body(source, SIG_RECORD_EXPORT)
    if rec is None:
        fails.append("L4 load terms: the record export ggml_backend_sycl_load_record_compute_term is not defined")
    else:
        g = rec.find("sycl_module_mutation_guard module_guard;")
        chk = rec.find("if (!module_guard)")
        call = rec.find("return ggml_sycl_load_record_compute_term(txn.id, device, bytes, n_ctx);")
        if not (0 <= g < chk < call) or "ggml_sycl_load_ledger" in rec:
            fails.append("L4 load terms: the record export does not reach the writer under the module guard")
    spans = [function_span(source, SIG_RECORD_EXPORT), function_span(source, SIG_RECORD_HOOK)]
    spans = [sp for sp in spans if sp is not None]
    stray = 0
    for m in re.finditer(r"\bggml_sycl_load_record_compute_term\s*\(", source):
        if re.match(r"[^;{]*\)\s*\{", source[m.end():]) and re.search(r"\bbool\s+$", source[:m.start()]):
            continue  # the writer's own definition
        if not any(a <= m.start() < b for a, b in spans):
            stray += 1
    if stray:
        fails.append("L4 load terms: the ledger writer is called outside the record export and the test hook "
                     "(%d call(s))" % stray)


def door_texts(root):
    """Every source file under src/ except the door's own, comment-stripped: the files that must never reach the probe."""
    out = {}
    base = os.path.join(root, DOOR_SCAN_DIR)
    for name in sorted(os.listdir(base)):
        rel = os.path.join(DOOR_SCAN_DIR, name)
        if rel == DOOR_H or not name.endswith((".cpp", ".h", ".hpp")):
            continue
        with open(os.path.join(root, rel), encoding="utf-8") as f:
            out[rel] = strip_comments(f.read())
    return out


def door_check(texts):
    """The residency probe has one door (src/llama-context-tenant.h): no other file under src/ calls the proc pointer or
    names the symbol except as the table field's type."""
    fails = []
    for rel, text in texts.items():
        if DOOR_CALL.search(text):
            fails.append("L4 probe door: %s calls the probe proc pointer; only llama_sycl_l4_probe_residency may" % rel)
        if DOOR_SYMBOL.search(text):
            fails.append("L4 probe door: %s names ggml_backend_sycl_probe_residency; the table resolves it by its macro" % rel)
    if not texts:
        fails.append("L4 probe door: no source file under %s was read (the scan is void)" % DOOR_SCAN_DIR)
    return fails


def check(header_raw, source):
    """`source` is the comment-stripped, normalized text: a mutation is one edit of it, so the (slow) strip is
    done once, not once per mutant."""
    fails = []
    names = header_proc_names(header_raw)
    if len(names) < 3:
        fails.append("L4 proc: fewer than 3 `Proc name:` entries in %s (the scan found %d): the gate is void" % (HEADER, len(names)))
    for required in ("ggml_backend_sycl_set_runtime_context_desc", "ggml_backend_sycl_tenant_coverage",
                     "ggml_backend_sycl_load_late_check", "ggml_backend_sycl_probe_residency"):
        if required not in names:
            fails.append("L4 proc: %s lost its `Proc name:` line in %s" % (required, HEADER))

    reg = function_body(source, r"\b" + REG_FN + r"\s*\(")
    if reg is None:
        fails.append("L4 proc: %s not found in %s" % (REG_FN, SOURCE))
        reg = ""
    for name in names:
        arm = re.search(r'if\s*\(\s*strcmp\s*\(\s*name\s*,\s*"' + re.escape(name) + r'"\s*\)\s*==\s*0\s*\)\s*\{\s*'
                        r'return\s*\(void\s*\*\)\s*' + re.escape(name) + r'\s*;\s*\}', reg)
        if not arm:
            fails.append("L4 proc: %s has no arm returning itself in %s" % (name, REG_FN))
        else:
            before = reg[:arm.start()]
            opened = len(re.findall(r"^\s*#\s*if", before, re.M)) - len(re.findall(r"^\s*#\s*endif", before, re.M))
            if opened != 0:
                fails.append("L4 proc: the arm of %s sits under a build switch" % name)
        if not re.search(r"\n[A-Za-z_ *]*\b" + re.escape(name) + r"\s*\([^;{]*\)\s*\{", source):
            fails.append("L4 proc: %s is declared in the header and defined nowhere in %s" % (name, SOURCE))

    # one writer
    rec_fn = function_body(source, r"\bbool\s+ggml_sycl_load_record_compute_term\s*\(")
    clr_fn = function_body(source, r"\bsize_t\s+ggml_sycl_load_clear_compute_terms\s*\(")
    if rec_fn is None:
        fails.append("L4 ledger: ggml_sycl_load_record_compute_term not found")
        rec_fn = ""
    if clr_fn is None:
        fails.append("L4 ledger: ggml_sycl_load_clear_compute_terms not found")
        clr_fn = ""
    if source.count(".ledger.record(") != 1 or rec_fn.count(".ledger.record(") != 1:
        fails.append("L4 ledger: the ledger is written (record) outside ggml_sycl_load_record_compute_term")
    if source.count(".ledger.clear(") != 1 or clr_fn.count(".ledger.clear(") != 1:
        fails.append("L4 ledger: the ledger is cleared outside ggml_sycl_load_clear_compute_terms")
    ledger_fns = (
        ("ggml_sycl_load_record_compute_term", r"\bbool\s+ggml_sycl_load_record_compute_term\s*\("),
        ("ggml_sycl_load_clear_compute_terms", r"\bsize_t\s+ggml_sycl_load_clear_compute_terms\s*\("),
        ("ggml_backend_sycl_load_late_check",
         r"\benum\s+ggml_sycl_late_check_result\s+ggml_backend_sycl_load_late_check\s*\("),
        ("ggml_backend_sycl_test_compute_term_count",
         r'\bextern\s+"C"\s+size_t\s+ggml_backend_sycl_test_compute_term_count\s*\('),
    )
    spans = {}
    for name, sig in ledger_fns:
        span = function_span(source, sig)
        if span is None:
            if name != "ggml_backend_sycl_test_compute_term_count":  # the hook is a test build's
                fails.append("L4 ledger: %s not found" % name)
            continue
        spans[name] = span
    stray = set()
    for m in re.finditer(r"\bggml_sycl_load_ledger\s*\(\s*\)", source):
        if re.match(r"\s*\{", source[m.end():]):
            continue  # the accessor's own definition
        if not any(a <= m.start() < b for a, b in spans.values()):
            stray.add(m.start())
    if stray:
        fails.append("L4 ledger: the ledger accessor is named outside the four sanctioned functions (%d use(s))" % len(stray))
    for name, (a, b) in spans.items():
        if "ggml_sycl_load_ledger()" not in source[a:b]:
            fails.append("L4 ledger: %s no longer reaches the ledger through its accessor" % name)

    # the mutex: every function that uses the ledger takes state.mutex before its first `.ledger.` use, and the
    # two that decide on the open transaction read it only after the lock
    lock_re = r"std::lock_guard<std::mutex> lock\(state\.mutex\);"
    for name, (a, b) in spans.items():
        body = source[a:b]
        uses = [m.start() for m in re.finditer(r"\.ledger\.", body)]
        locks = [m for m in re.finditer(lock_re, body)]
        held = bool(uses)
        for u in uses:
            before = [m for m in locks if m.end() <= u]
            # held at the use: a lock_guard precedes it and no block that holds the lock closes between the two
            if not before or min_depth(body[before[-1].end():u]) < 0:
                held = False
            elif body[:before[-1].start()].rstrip()[-1:] not in (";", "{", "}"):
                held = False  # the guard is the substatement of an `if`/`for`/`while`/`else`: not unconditional
        if not held:
            fails.append("L4 ledger lock: %s uses the ledger without state.mutex held at the use" % name)
    for name in ("ggml_sycl_load_record_compute_term", "ggml_backend_sycl_load_late_check"):
        if name in spans:
            body = source[spans[name][0]:spans[name][1]]
            lk = re.search(lock_re, body)
            op = body.find("ggml_sycl_load_txn_is_open(")
            if lk is None or op < 0 or op < lk.start():
                fails.append("L4 ledger lock: %s reads the open transaction outside the ledger's lock" % name)
    is_open = function_body(source, r"\bstatic\s+bool\s+ggml_sycl_load_txn_is_open\s*\(")
    if is_open is None or not re.search(
            r"return txn != 0 && ggml_sycl::lifecycle::global_registry\(\)\.admission_diagnostics\(\)\.active_txn == txn;",
            is_open):
        fails.append("L4 ledger open: ggml_sycl_load_txn_is_open is not `txn != 0 && active_txn == txn`")
    if "record(txn, device, bytes, n_ctx, ggml_sycl_load_txn_is_open(txn))" not in source:
        fails.append("L4 ledger open: the record is not gated by the open transaction")
    if "check(txn.id, device, compute_bytes, ggml_sycl_load_txn_is_open(txn.id))" not in source:
        fails.append("L4 ledger open: the late check is not gated by the open transaction")

    # fail-closed values and levels
    late_body = function_body(source, r"\benum\s+ggml_sycl_late_check_result\s+ggml_backend_sycl_load_late_check\s*\(") or ""
    if len(re.findall(r"catch\s*\(\.\.\.\)", late_body)) != 1 or not re.search(
            r"catch\s*\(\.\.\.\)\s*\{\s*return GGML_SYCL_LATE_CHECK_NOT_RECORDED;\s*\}", late_body):
        fails.append("L4 fail-closed: the late check's catch does not answer NOT_RECORDED")
    if not re.search(r"sycl_module_mutation_guard module_guard;\s*if \(!module_guard\) \{\s*return GGML_SYCL_LATE_CHECK_NOT_RECORDED;",
                     late_body):
        fails.append("L4 fail-closed: the late check lost its module guard (a closed module answers NOT_RECORDED)")
    if not re.search(r"return r\.result;\s*\}\s*catch\s*\(\.\.\.\)", late_body):
        fails.append("L4 late: the late check does not answer the ledger's result (it must end `return r.result;`)")
    if "ggml_sycl_load_ledger_log(r);" not in late_body:
        fails.append("L4 log level: the late check does not log through the ledger's level")
    cov_body = function_body(source, r"\benum\s+ggml_sycl_tenant_coverage\s+ggml_backend_sycl_tenant_coverage\s*\(") or ""
    if len(re.findall(r"catch\s*\(\.\.\.\)", cov_body)) != 1 or not re.search(
            r"catch\s*\(\.\.\.\)\s*\{\s*return GGML_SYCL_TENANT_COVERAGE_GROWTH;\s*\}", cov_body):
        fails.append("L4 fail-closed: the coverage query's catch does not answer GROWTH")
    if not re.search(r"sycl_module_mutation_guard module_guard;\s*if \(!module_guard \|\|[^{]*\) \{\s*return GGML_SYCL_TENANT_COVERAGE_GROWTH;",
                     cov_body):
        fails.append("L4 fail-closed: the coverage query lost its module guard (a closed module answers GROWTH)")
    if not re.search(r"const uint64_t id = ggml_sycl_context_execution_id\(ctx\);\s*if \(id == 0\) \{\s*return GGML_SYCL_TENANT_COVERAGE_GROWTH;",
                     cov_body):
        fails.append("L4 fail-closed: the coverage query of an unbound context does not answer GROWTH")
    clr_fn2 = function_body(source, r"\bsize_t\s+ggml_sycl_load_clear_compute_terms\s*\(") or ""
    if not re.search(r"catch\s*\(\.\.\.\)\s*\{\s*return 0;\s*\}", clr_fn2):
        fails.append("L4 fail-closed: the ledger clear's catch does not answer 0")
    log_fn = function_body(source, r"\bstatic\s+void\s+ggml_sycl_load_ledger_log\s*\(") or ""
    for lvl, macro in (("ERROR", "GGML_LOG_ERROR"), ("WARN", "GGML_LOG_WARN"), ("INFO", "GGML_LOG_INFO")):
        if not re.search(r"case ggml_sycl::LOAD_LOG_LEVEL_" + lvl + r":\s*" + macro + r"\(", log_fn):
            fails.append("L4 log level: a ledger line of level %s is not logged through %s" % (lvl, macro))
    set_fn = function_body(source, r"\bstatic\s+void\s+ggml_sycl_published_section_set\s*\(") or ""
    if not re.search(r"if \(id == 0\) \{\s*if \(section\) \{\s*GGML_LOG_WARN\(", set_fn):
        fails.append("L4 section: a publish for an unbound context is silent (no WARN)")
    if not re.search(r"if \(id == 0\) \{\s*if \(section\) \{\s*GGML_LOG_WARN\([^;]*;\s*\}\s*return;\s*\}", set_fn):
        fails.append("L4 section: a publish for an unbound context goes on to the registry with key 0")
    # a failed drop is said at ERROR (the entry then stays until the process ends: its key is gone), and an erase
    # takes its id from the caller -- it never reads a backend's id itself, so one drop has one id
    herase_fn = function_body(source, ERASE_HOST_SIG) or ""
    if not re.search(r"catch\s*\(\.\.\.\)\s*\{\s*GGML_LOG_ERROR\(", herase_fn):
        fails.append("L4 host tier: the drop of the host reservation swallows a failure silently (no ERROR)")
    if "ggml_sycl_context_execution_id(" in herase_fn or "execution_context_id" in herase_fn:
        fails.append("L4 host tier: the host reservation's erase reads a context id itself (it must use the caller's)")
    pin(fails, herase_fn,
        r"^[^{]*\{\s*try \{\s*if \(context_id == 0 \|\| device < 0 \|\| device >= GGML_SYCL_MAX_DEVICES\) \{\s*return;\s*\}\s*"
        r"auto previous = ggml_sycl_kv_region_registry\(device\)\.take_tenant_slots\(context_id\);\s*\(void\) previous;\s*"
        r"\} catch \(\.\.\.\) \{\s*GGML_LOG_ERROR\(\s*(?:\"[^\"]*\"\s*)+\);\s*\}\s*\}$",
        "L4 host tier: the host reservation's erase does not take the table by the id it was given")
    erase_fn = function_body(source, ERASE_SECTION_SIG) or ""
    if not re.search(r"catch\s*\(\.\.\.\)\s*\{\s*GGML_LOG_ERROR\(", erase_fn):
        fails.append("L4 section: the erase swallows a failed drop silently (no ERROR)")
    if "ggml_sycl_context_execution_id(" in erase_fn or "execution_context_id" in erase_fn:
        fails.append("L4 section: the section's erase reads a context id itself (it must use the caller's)")
    pin(fails, erase_fn,
        r"^[^{]*\{\s*try \{\s*if \(context_id == 0 \|\| device < 0 \|\| device >= GGML_SYCL_MAX_DEVICES\) \{\s*return;\s*\}\s*"
        r"auto previous = ggml_sycl_kv_region_registry\(device\)\.drop_published_section\(context_id\);\s*\(void\) previous;\s*"
        r"\} catch \(\.\.\.\) \{\s*GGML_LOG_ERROR\(\s*(?:\"[^\"]*\"\s*)+\);\s*\}\s*\}$",
        "L4 section: the section's erase does not drop the section by the id it was given")

    # load_end: guard after the finisher check
    end = function_body(source, r"\bggml_sycl_lifecycle_result\s+ggml_backend_sycl_model_load_end\s*\(")
    if end is None:
        fails.append("L4 ledger: ggml_backend_sycl_model_load_end not found")
    else:
        g = end.find("ggml_sycl_load_ledger_clear_guard ledger_clear{ txn.id }")
        f = end.find("if (!ticket.finisher)")
        eff = end.find("acquire_finisher_effect")
        if g < 0:
            fails.append("L4 ledger: model_load_end does not clear the load's compute terms (no guard)")
        elif not (f >= 0 and f < g and (eff < 0 or g < eff)):
            fails.append("L4 ledger: the clear guard must come after the finisher check and before the effects")
        t = end.find("try {")
        if g >= 0 and t >= 0:
            between = end[t + len("try {"):g]
            if net_depth(between) != 0 or between.rstrip()[-1:] not in (";", "{", "}"):
                fails.append("L4 ledger: the clear guard is not directly in the try body (a nested block or an `if` ends it "
                             "before the registry's end)")
        tries = len(re.findall(r"\btry\b", end))
        catches = re.findall(r"\bcatch\s*\(([^)]*)\)", end)
        if tries != 1 or catches != ["..."]:
            fails.append("L4 ledger: model_load_end must have exactly one try and one catch (...) (found %d try, catches %s)"
                         % (tries, catches))
        h = re.search(r"\bcatch\s*\(\.\.\.\)\s*\{", end)
        handler = end[h.end():] if h else ""
        if re.search(r"\bthrow\s*;", handler):
            fails.append("L4 ledger: model_load_end's handler rethrows")
        if not re.match(r"\s*ggml_sycl_load_ledger_clear_after_end after_end;", handler):
            fails.append("L4 ledger: after_end is not the handler's first statement")
        if not re.match(r"\s*ggml_sycl_load_ledger_clear_after_end after_end;\s*if \(ticket\.finisher\) \{\s*after_end\.txn = txn\.id;\s*\}"
                        r"\s*g_sycl_abort_load_exit = false;\s*if \(placement_inserted\) \{", handler):
            fails.append("L4 ledger: the handler does not arm after_end for the finisher before anything that can throw "
                         "(or something sits between the declaration, the arming and the first work)")
        if handler.count("after_end.txn") != 2:
            fails.append("L4 ledger: after_end.txn is set other than once at the top and once in the recovery arm "
                         "(the finisher arm must not reset it)")
        if not re.search(r"if \(ticket\.finisher\) \{\s*\(void\) ggml_sycl_abort_owner_effects_noexcept\(ticket\.token, "
                         r"\"load_end/exception-rollback\"\);\s*const auto failed = registry->finalize_end\(ticket, false\);",
                         handler):
            fails.append("L4 ledger: the finisher arm does not abort the owner effects and then end the transaction")
        if not re.search(r"if \(!recovery\.finisher\) \{\s*return [^;]*;\s*\}\s*after_end\.txn = txn\.id;\s*"
                         r"ggml_sycl_finalize_binding_failure_abort\(\*registry, recovery\);", handler):
            fails.append("L4 ledger: the recovery arm does not set after_end.txn unconditionally before its abort")
        if "ggml_sycl_load_clear_compute_terms(" in end:
            fails.append("L4 ledger: model_load_end clears the ledger itself, not through the guard and after_end")
    guard = function_body(source, r"\bstruct\s+ggml_sycl_load_ledger_clear_guard") or ""
    if not re.search(r"if \(std::uncaught_exceptions\(\) > uncaught\) \{\s*return;\s*\}\s*"
                     r"\(void\) ggml_sycl_load_clear_compute_terms\(txn\);", guard):
        fails.append("L4 ledger: the try-scope guard clears during unwinding (no uncaught_exceptions skip)")
    if "uncaught(std::uncaught_exceptions())" not in guard:
        fails.append("L4 ledger: the try-scope guard does not snapshot uncaught_exceptions at construction")
    after = function_body(source, r"\bstruct\s+ggml_sycl_load_ledger_clear_after_end") or ""
    if not re.search(r"uint64_t txn = 0;", after) or not re.search(
            r"if \(txn != 0\) \{\s*\(void\) ggml_sycl_load_clear_compute_terms\(txn\);\s*\}", after):
        fails.append("L4 ledger: after_end does not clear only a transaction it was armed with")

    # the end of an execution context drops the entries its id keyed, BY THAT ID, after the binding's id is
    # zeroed: the backend's destructor cannot find them afterwards (finish_drain and close_if_idle reset the id
    # first), and a drop that re-read a backend's id would have to run before the reset, leaving a window in
    # which a publish re-creates an entry under the dead id.  The devices are collected under the binding mutex
    # (nothing to pin, no allocation that can fail); the drop runs outside it.
    clear_bind = function_body(source, CLEAR_BIND_SIG)
    if clear_bind is None:
        fails.append("L4 context end: ggml_sycl_execution_clear_bindings_for_context not found")
    else:
        call_text = "ggml_sycl_execution_drop_context_registry_entries(context_id, devices);"
        call = clear_bind.find(call_text)
        reset = clear_bind.find("ggml_sycl_execution_reset_backend_binding_state(")
        if call < 0 or reset < 0 or call < reset or clear_bind.count(call_text) != 1 or net_depth(clear_bind[:call]) != 1 or \
                clear_bind[:call].rstrip()[-1:] not in (";", "{", "}"):
            fails.append("L4 context end: the clear of an ended context's bindings does not drop its registry entries by the "
                         "ended id after it has reset the bindings, once, outside the lock (statement-initial)")
        pin(fails, clear_bind,
            r"\{\s*bool devices\[GGML_SYCL_MAX_DEVICES\] = \{\};\s*\{\s*std::lock_guard<std::mutex> lock\(g_execution_backend_binding_mutex\);"
            r"[\s\S]*?if \(it->first->device >= 0 && it->first->device < GGML_SYCL_MAX_DEVICES\) \{\s*"
            r"devices\[it->first->device\] = true;\s*\}\s*ggml_sycl_execution_reset_backend_binding_state\(it->first\);"
            r"[\s\S]*?\}\s*\}\s*ggml_sycl_execution_drop_context_registry_entries\(context_id, devices\);\s*\}$",
            "L4 context end: the clear of an ended context's bindings does not collect the bound devices under the "
            "lock, reset, and drop the entries by the ended id after the lock (devices collected before the reset)")
        pin(fails, clear_bind, r"for \(auto it = g_execution_backend_bindings\.begin\(\); it != g_execution_backend_bindings\.end\(\);\) \{\s*"
                               r"if \(it->second && it->second->context_id == context_id\) \{",
            "L4 context end: the collection loop does not select the bindings of the ended context by its id")
        if re.search(r"for_each_bound_backend|pin_bound_backends|\bpin_count\b", clear_bind):
            fails.append("L4 context end: the clear of an ended context's bindings pins backends again (an allocation that "
                         "can fail, and a failed pin leaves the entries orphaned)")
    drop_entries = function_body(source, DROP_ENTRIES_SIG)
    pin(fails, drop_entries,
        r"noexcept\s*\{\s*for \(int device = 0; device < GGML_SYCL_MAX_DEVICES; \+\+device\) \{\s*if \(devices\[device\]\) \{\s*"
        r"ggml_sycl_published_section_erase\(device, context_id\);\s*ggml_sycl_host_tenants_erase\(device, context_id\);"
        r"\s*\}\s*\}\s*\}$",
        "L4 context end: the drop of an ended context's registry entries does not erase the section then the host "
        "reservation, by the ended id, on each device the ended bindings were on")
    if drop_entries is not None and re.search(r"g_execution_backend_binding_mutex|\btry\b|\bcatch\b|ggml_sycl_context_execution_id", drop_entries):
        fails.append("L4 context end: the registry entries of an ended context are dropped under the binding mutex, or "
                     "through a try, or by a re-read id")

    # destructor order, and one read of the id for both drops
    dtor = function_body(source, DTOR_BACKEND_SIG)
    if dtor is None:
        fails.append("L4 section: ~ggml_backend_sycl_context not found")
    else:
        e = dtor.find("ggml_sycl_published_section_erase(device, context_id)")
        u = dtor.find("ggml_sycl_execution_unbind_backend(this)")
        if e < 0:
            fails.append("L4 section: the backend context's destructor does not erase its published section")
        elif u < 0 or e > u:
            fails.append("L4 section: the section is erased after the execution binding is reset (its key is gone)")
        h = dtor.find("ggml_sycl_host_tenants_erase(device, context_id)")
        if h < 0:
            fails.append("L4 host tier: the backend context's destructor does not drop its held host reservation")
        elif not (e >= 0 and e < h < u):
            fails.append("L4 host tier: the host reservation is dropped outside the section-erase to unbind window")
        pin(fails, dtor,
            r"\n    const uint64_t context_id = ggml_sycl_context_execution_id\(this\);\n"
            r"    ggml_sycl_published_section_erase\(device, context_id\);\n"
            r"    ggml_sycl_host_tenants_erase\(device, context_id\);\n"
            r"    ggml_sycl_execution_unbind_backend\(this\);",
            "L4 section: the destructor does not read its context id once, statement-initial, for both drops before the unbind")
        if dtor.count("ggml_sycl_context_execution_id(") != 1:
            fails.append("L4 section: the destructor reads its context id more than once (two sources for one drop)")

    # the host reservation: owner-first, the carve's request, one installer, one remover
    reserve = function_body(source, r"\bstatic\s+bool\s+ggml_sycl_reserve_host_tenants\s*\(")
    if reserve is None:
        fails.append("L4 host tier: ggml_sycl_reserve_host_tenants not found")
    else:
        if "ggml_sycl::unified_allocate_owner(req)" not in reserve:
            fails.append("L4 host tier: the host carve is not allocated through unified_allocate_owner")
        for field in ("req.intent.constraints.must_host_pinned = true;", "req.intent.constraints.use_pinned_pool = true;",
                      "req.intent.category = ggml_sycl::runtime_category::HOST_COMPUTE;",
                      "req.intent.cohort_id = info->name;"):
            if field not in reserve:
                fails.append("L4 host tier: the host carve's request lost `%s`" % field.strip())
        for bad in ("forbid_host_zone_growth", "require_host_usm_base"):
            if bad in reserve:
                fails.append("L4 host tier: the host carve's request sets %s (the first publish may grow the zone)" % bad)
        if "sycl::malloc_host" in reserve or "unified_alloc(" in reserve:
            fails.append("L4 host tier: the host carve bypasses the owner-first surface")
    if source.count("ggml_sycl_reserve_host_tenants(") != 2:
        fails.append("L4 host tier: ggml_sycl_reserve_host_tenants has more or fewer than one caller")
    if source.count(".install_tenant_slots(") != 1 or source.count(".take_tenant_slots(") != 1:
        fails.append("L4 host tier: the registry's slot table has more or fewer than one installer or remover")
    if impl_first_publish_wrong(source):
        fails.append("L4 host tier: the first publish does not reserve before L1, or does not install once before the inner transaction")

    # the buffer type: claim before any allocation, release on free
    alloc = function_body(source, r"\bstatic\s+ggml_backend_buffer_t\s+ggml_backend_sycl_host_buffer_type_alloc_buffer\s*\(")
    if alloc is None:
        fails.append("L4 host tier: the SYCL_Host alloc_buffer not found")
    else:
        c = alloc.find("ggml_sycl::tenant_claim_scope::active()")
        a = alloc.find("ggml_sycl::unified_alloc(")
        if c < 0 or a < 0 or c > a:
            fails.append("L4 host tier: the SYCL_Host alloc_buffer does not claim inside a scope before it allocates")
    freeb = function_body(source, r"\bstatic\s+void\s+ggml_backend_sycl_host_buffer_free_buffer\s*\(")
    if freeb is None or "ggml_sycl::tenant_claim_scope::release(*ctx->claim" not in freeb:
        fails.append("L4 host tier: the SYCL_Host free_buffer does not release its claim")
    host_tier_pins(source, alloc, freeb, reserve, fails)
    free_path_pins(source, freeb, fails)

    # transaction tail
    txn = function_body(source, r"\bggml_sycl_txn_result\s+ggml_sycl_run_runtime_context_transaction\s*\(")
    if txn is None:
        fails.append("L4 section: ggml_sycl_run_runtime_context_transaction not found")
    else:
        a = txn.find("ctx->runtime_kv_admitted = true;")
        b = txn.find("ggml_sycl_published_section_set(ctx, nullptr);")
        if a < 0 or b < 0 or b < a:
            fails.append("L4 section: a successful publish does not drop the context's earlier section")
        elif re.search(r"\btry\b|\bcatch\b", txn[a:]):
            fails.append("L4 section: a try or catch sits between the runtime_kv_admitted store and the end of the transaction "
                         "(a failed drop would be swallowed)")

    impl = function_body(source, r"\bggml_sycl_set_runtime_context_for_model_impl\s*\(")
    if impl is None:
        fails.append("L4 section: ggml_sycl_set_runtime_context_for_model_impl not found")
    else:
        inner = impl.find("const bool inner_ok = g_runtime_update_succeeded;")
        store = impl.find("ggml_sycl_published_section_set(backend_ctx, section)")
        if inner < 0 or store < 0 or store < inner or not re.search(r"if\s*\(\s*inner_ok\s*&&\s*section\s*\)", impl[inner:store]):
            fails.append("L4 section: the descriptor's section is stored without a successful inner transaction")
        if not re.search(r"parse_runtime_context_desc\s*\(", impl):
            fails.append("L4 section: the descriptor publish no longer reads the descriptor through parse_runtime_context_desc")

    cent = function_body(source, r"\bvoid\s+ggml_backend_sycl_set_runtime_context\s*\(")
    if cent is None:
        fails.append("L4 C entry: ggml_backend_sycl_set_runtime_context not found")
    else:
        arms = [m.group(1).strip() for m in re.finditer(r"\bcatch\s*\(([^)]*)\)", cent)]
        if arms != ["const std::system_error & e", "const std::exception & e", "..."]:
            fails.append("L4 C entry: ggml_backend_sycl_set_runtime_context must catch system_error, then std::exception, "
                         "then everything (found %s)" % arms)
        elif not re.search(r"try \{\s*\(void\) ggml_sycl_run_runtime_context_transaction\(", cent):
            fails.append("L4 C entry: the runtime context transaction is called outside the C entry's try")
        elif cent.count("GGML_LOG_ERROR(") != 3 or re.search(r"\bthrow\b", cent):
            fails.append("L4 C entry: every arm of the C entry must log at ERROR and none may rethrow")
    nctx = function_body(source, r"\bvoid\s+ggml_backend_sycl_set_runtime_n_ctx\s*\(")
    if nctx is None or "ggml_backend_sycl_set_runtime_context(backend," not in nctx or \
            "ggml_sycl_run_runtime_context_transaction" in nctx:
        fails.append("L4 C entry: ggml_backend_sycl_set_runtime_n_ctx does not delegate to the guarded C entry")

    impl_fn = function_body(source, r"\bggml_sycl_set_runtime_context_for_model_impl\s*\(") or ""
    if "ggml_backend_sycl_set_runtime_context(" in impl_fn or not re.search(
            r"try \{\s*\(void\) ggml_sycl_run_runtime_context_transaction\(backend,[^;]*;\s*\} catch \(\.\.\.\) \{\s*"
            r"return GGML_SYCL_LIFECYCLE_EFFECT_FAILED;\s*\}\s*const bool inner_ok", impl_fn):
        fails.append("L4 C entry: the descriptor publish must call the transaction in its own try, not the C entry that "
                     "swallows a failed drop it has to answer EFFECT_FAILED for, and that try's catch answers EFFECT_FAILED")

    late = function_body(source, r"\benum\s+ggml_sycl_late_check_result\s+ggml_backend_sycl_load_late_check\s*\(")
    if late is None:
        fails.append("L4 late: ggml_backend_sycl_load_late_check not found")
    else:
        if "late_term_shrink_admitted" not in late or "r.shrink_counted" not in late:
            fails.append("L4 late: the late check does not count an admitted shrink")
    if len(re.findall(r"dump_counter::late_term_shrink_admitted", source)) != 1:
        fails.append("L4 late: late_term_shrink_admitted has more or fewer than one producer")
    probe_pins(source, fails)
    load_term_pins(header_raw, source, fails)
    return fails


def mutations(header_raw, source):
    """Each entry: (label, expected message fragment, header text, source text)."""
    muts = []
    src = source
    for name in header_proc_names(header_raw):
        pat = re.compile(r'    if \(strcmp\(name, "' + re.escape(name) + r'"\) == 0\) \{\n        return \(void \*\) ' + re.escape(name) + r';\n    \}\n')
        if pat.search(src):
            muts.append(("drop the arm of " + name, "has no arm returning itself", header_raw, pat.sub("", src, count=1)))
        else:
            muts.append(("PATTERN NOT FOUND for arm " + name, "PATTERN", header_raw, src))
    pairs = [
        ("a second writer of the ledger", "written (record) outside",
         "static size_t ggml_sycl_load_clear_compute_terms(uint64_t txn) noexcept {",
         "static void h_second_writer() { ggml_sycl_load_ledger().ledger.record(1, 0, 1, 1); }\nstatic size_t ggml_sycl_load_clear_compute_terms(uint64_t txn) noexcept {"),
        ("a second clearer of the ledger", "cleared outside",
         "static size_t ggml_sycl_load_clear_compute_terms(uint64_t txn) noexcept {",
         "static void h_second_clearer() { ggml_sycl_load_ledger().ledger.clear(1); }\nstatic size_t ggml_sycl_load_clear_compute_terms(uint64_t txn) noexcept {"),
        ("the ledger read by a stray function", "outside the four sanctioned",
         "static size_t ggml_sycl_load_clear_compute_terms(uint64_t txn) noexcept {",
         "static size_t h_stray() { return ggml_sycl_load_ledger().ledger.size(); }\nstatic size_t ggml_sycl_load_clear_compute_terms(uint64_t txn) noexcept {"),
        ("the end call's clear guard dropped", "no guard",
         "        ggml_sycl_load_ledger_clear_guard ledger_clear{ txn.id };\n", ""),
        ("the clear guard before the finisher check", "after the finisher check",
         "        ggml_sycl_load_ledger_clear_guard ledger_clear{ txn.id };\n", ""),
        ("the carve through the legacy allocator", "not allocated through unified_allocate_owner",
         "ggml_sycl::allocation_result allocation = ggml_sycl::unified_allocate_owner(req);\n        if (!allocation) {\n            refusal = std::string(\"the host reservation of \")",
         "ggml_sycl::allocation_result allocation = ggml_sycl::detail::promote_legacy_alloc_owner({});\n        if (!allocation) {\n            refusal = std::string(\"the host reservation of \")"),
        ("the carve not pinned", "lost `req.intent.constraints.must_host_pinned = true;`",
         "        req.intent.constraints.must_host_pinned = true;\n        req.intent.constraints.use_pinned_pool = true;\n        ggml_sycl::allocation_result",
         "        req.intent.constraints.use_pinned_pool = true;\n        ggml_sycl::allocation_result"),
        ("the carve off the pool", "lost `req.intent.constraints.use_pinned_pool = true;`",
         "        req.intent.constraints.use_pinned_pool = true;\n        ggml_sycl::allocation_result",
         "        req.intent.constraints.use_pinned_pool = false;\n        ggml_sycl::allocation_result"),
        ("the carve in the wrong category", "lost `req.intent.category",
         "        req.intent.category = ggml_sycl::runtime_category::HOST_COMPUTE;\n        req.intent.cohort_id = info->name;",
         "        req.intent.category = ggml_sycl::runtime_category::COMPUTE;\n        req.intent.cohort_id = info->name;"),
        ("the carve's cohort retyped", "lost `req.intent.cohort_id",
         "        req.intent.cohort_id = info->name;", "        req.intent.cohort_id = \"context-compute-host\";"),
        ("the carve forbids zone growth", "sets forbid_host_zone_growth",
         "        req.intent.constraints.use_pinned_pool = true;\n        ggml_sycl::allocation_result",
         "        req.intent.constraints.use_pinned_pool = true;\n        req.intent.constraints.forbid_host_zone_growth = true;\n        ggml_sycl::allocation_result"),
        ("a second installer of the table", "more or fewer than one installer or remover",
         "static size_t ggml_sycl_load_clear_compute_terms(uint64_t txn) noexcept {",
         "static void h_second_installer() { ggml_sycl_kv_region_registry(0).install_tenant_slots(1, nullptr, 0); }\nstatic size_t ggml_sycl_load_clear_compute_terms(uint64_t txn) noexcept {"),
        ("a second remover of the table", "more or fewer than one installer or remover",
         "static size_t ggml_sycl_load_clear_compute_terms(uint64_t txn) noexcept {",
         "static void h_second_remover() { (void) ggml_sycl_kv_region_registry(0).take_tenant_slots(1); }\nstatic size_t ggml_sycl_load_clear_compute_terms(uint64_t txn) noexcept {"),
        ("the first publish reserves under L1", "does not reserve before L1",
         "                if (!ggml_sycl_reserve_host_tenants(backend_ctx, *section, host_tenants, refusal)) {",
         "                if (false) {"),
        ("the free forgets its claim", "does not release its claim",
         "        (void) ggml_sycl::tenant_claim_scope::release(*ctx->claim, 0);\n        ctx->claim.reset();", "        ctx->claim.reset();"),
        ("the transaction tail's drop removed", "does not drop the context's earlier section",
         "    ggml_sycl_published_section_set(ctx, nullptr);\n    return ggml_sycl_txn_result::ACCEPTED;",
         "    return ggml_sycl_txn_result::ACCEPTED;"),
        ("the section stored on any outcome", "without a successful inner transaction",
         "    if (inner_ok && section) {", "    if (section) {"),
        ("the shrink counter dropped", "does not count an admitted shrink",
         "        if (r.shrink_counted) {\n            ggml_sycl::unified_cache_dump_counter_add(ggml_sycl::dump_counter::late_term_shrink_admitted, device);\n        }\n", ""),
    ]
    # One edit inside one named function: (label, expected message, signature, old, new).  Several of the
    # lines these touch recur in the other ledger functions, so the edit is scoped to the function it names.
    scoped = [
        # ---- the one id of a drop: the erase helpers take it, the end of a context passes the ended one, the
        # ---- destructor reads its own once, the install guard carries the id the install used
        ("the destructor's erase dropped", "does not erase its published section", DTOR_BACKEND_SIG,
         "    ggml_sycl_published_section_erase(device, context_id);\n", ""),
        ("the destructor's erase after the unbind", "erased after the execution binding", DTOR_BACKEND_SIG,
         "    ggml_sycl_published_section_erase(device, context_id);\n    ggml_sycl_host_tenants_erase(device, context_id);\n    ggml_sycl_execution_unbind_backend(this);",
         "    ggml_sycl_host_tenants_erase(device, context_id);\n    ggml_sycl_execution_unbind_backend(this);\n    ggml_sycl_published_section_erase(device, context_id);"),
        ("the host reservation never dropped", "does not drop its held host reservation", DTOR_BACKEND_SIG,
         "    ggml_sycl_host_tenants_erase(device, context_id);\n", ""),
        ("the host reservation dropped after the unbind", "outside the section-erase to unbind window", DTOR_BACKEND_SIG,
         "    ggml_sycl_host_tenants_erase(device, context_id);\n    ggml_sycl_execution_unbind_backend(this);",
         "    ggml_sycl_execution_unbind_backend(this);\n    ggml_sycl_host_tenants_erase(device, context_id);"),
        ("the destructor reads its id twice", "does not read its context id once", DTOR_BACKEND_SIG,
         "    ggml_sycl_host_tenants_erase(device, context_id);\n",
         "    ggml_sycl_host_tenants_erase(device, ggml_sycl_context_execution_id(this));\n"),
        ("the destructor drops by another id", "does not read its context id once", DTOR_BACKEND_SIG,
         "    ggml_sycl_published_section_erase(device, context_id);\n", "    ggml_sycl_published_section_erase(device, 0);\n"),
        ("the host reservation's drop may throw", "may throw (not noexcept)", ERASE_HOST_SIG,
         "(int device, uint64_t context_id) noexcept {", "(int device, uint64_t context_id) {"),
        ("the section's drop may throw", "the drop of the published section may throw", ERASE_SECTION_SIG,
         "(int device, uint64_t context_id) noexcept {", "(int device, uint64_t context_id) {"),
        ("the host reservation's drop swallows silently", "drop of the host reservation swallows a failure silently",
         ERASE_HOST_SIG, "    } catch (...) {\n        GGML_LOG_ERROR(", "    } catch (...) {\n        (void) ("),
        ("the section's drop swallows silently", "swallows a failed drop silently", ERASE_SECTION_SIG,
         "    } catch (...) {\n        GGML_LOG_ERROR(", "    } catch (...) {\n        (void) ("),
        ("the host reservation's drop says a failure at WARN", "drop of the host reservation swallows a failure silently",
         ERASE_HOST_SIG, "GGML_LOG_ERROR(", "GGML_LOG_WARN("),
        ("the section's drop says a failure at WARN", "swallows a failed drop silently", ERASE_SECTION_SIG,
         "GGML_LOG_ERROR(", "GGML_LOG_WARN("),
        ("the host reservation's erase re-reads an id", "erase reads a context id itself", ERASE_HOST_SIG,
         "    try {\n", "    try {\n        (void) ggml_sycl_context_execution_id(nullptr);\n"),
        ("the section's erase re-reads an id", "erase reads a context id itself", ERASE_SECTION_SIG,
         "    try {\n", "    try {\n        (void) ggml_sycl_context_execution_id(nullptr);\n"),
        ("the host reservation's erase takes another table", "does not take the table by the id it was given", ERASE_HOST_SIG,
         ".take_tenant_slots(context_id)", ".take_tenant_slots(0)"),
        ("the section's erase drops another section", "does not drop the section by the id it was given", ERASE_SECTION_SIG,
         ".drop_published_section(context_id)", ".drop_published_section(0)"),
        ("a second install after the inner transaction", "does not install once before the inner transaction",
         IMPL_SIG, "    const bool inner_ok = g_runtime_update_succeeded;",
         "    uint64_t h_id = 0;\n    (void) ggml_sycl_host_tenants_install(backend_ctx, host_tenants, section->tenant_key, h_id);\n    const bool inner_ok = g_runtime_update_succeeded;"),
        ("the install result discarded", "the install result is not a refusal with nothing published", IMPL_SIG,
         "if (!ggml_sycl_host_tenants_install(backend_ctx, host_tenants, section->tenant_key, installed_id)) {",
         "if ((ggml_sycl_host_tenants_install(backend_ctx, host_tenants, section->tenant_key, installed_id), false)) {"),
        ("the installed table never arms the guard", "the install result is not a refusal with nothing published", IMPL_SIG,
         "                    host_tenants_installed.device = backend_ctx->device;\n                    host_tenants_installed.id = installed_id;\n", ""),
        ("the installed table arms the guard with a re-read id", "by the device and the id it was keyed with", IMPL_SIG,
         "host_tenants_installed.id = installed_id;", "host_tenants_installed.id = ggml_sycl_context_execution_id(backend_ctx);"),
        ("the install reports a re-read id", "the table is dropped from the caller before the registry took it", INSTALL_SIG,
         "    installed_id = id;\n", "    installed_id = ggml_sycl_context_execution_id(ctx);\n"),
        ("the install guard never takes the table back", "the install guard does not take the table back",
         r"\bstruct\s+ggml_sycl_host_tenants_install_guard\b", "            ggml_sycl_host_tenants_erase(device, id);\n", "            (void) device;\n"),
        ("the install guard takes the table back even when kept", "the install guard does not take the table back",
         r"\bstruct\s+ggml_sycl_host_tenants_install_guard\b", "    void keep() { id = 0; }", "    void keep() {}"),
        ("the install guard takes back another id", "the install guard does not take the table back",
         r"\bstruct\s+ggml_sycl_host_tenants_install_guard\b", "ggml_sycl_host_tenants_erase(device, id);", "ggml_sycl_host_tenants_erase(device, 0);"),
        # ---- the end of an execution context
        ("the ended context's entries are never dropped", "does not drop its registry entries by the ended id", CLEAR_BIND_SIG,
         "    ggml_sycl_execution_drop_context_registry_entries(context_id, devices);\n", ""),
        ("the ended context's entries are dropped before the reset", "does not drop its registry entries by the ended id", CLEAR_BIND_SIG,
         "    bool devices[GGML_SYCL_MAX_DEVICES] = {};\n",
         "    bool devices[GGML_SYCL_MAX_DEVICES] = {};\n    ggml_sycl_execution_drop_context_registry_entries(context_id, devices);\n"),
        ("the ended context's entries are dropped conditionally", "does not drop its registry entries by the ended id", CLEAR_BIND_SIG,
         "    ggml_sycl_execution_drop_context_registry_entries(context_id, devices);\n",
         "    if (context_id != 0) ggml_sycl_execution_drop_context_registry_entries(context_id, devices);\n"),
        ("the ended context's entries are dropped under the binding mutex", "does not drop its registry entries by the ended id", CLEAR_BIND_SIG,
         "    }\n    ggml_sycl_execution_drop_context_registry_entries(context_id, devices);\n}",
         "    ggml_sycl_execution_drop_context_registry_entries(context_id, devices);\n    }\n}"),
        ("the ended context's entries are dropped by another id", "does not drop its registry entries by the ended id", CLEAR_BIND_SIG,
         "drop_context_registry_entries(context_id, devices);", "drop_context_registry_entries(0, devices);"),
        ("the ended context's devices are not collected", "does not collect the bound devices under the lock", CLEAR_BIND_SIG,
         "                if (it->first->device >= 0 && it->first->device < GGML_SYCL_MAX_DEVICES) {\n                    devices[it->first->device] = true;\n                }\n", ""),
        ("the ended context's devices are collected after the reset", "does not collect the bound devices under the lock", CLEAR_BIND_SIG,
         "                if (it->first->device >= 0 && it->first->device < GGML_SYCL_MAX_DEVICES) {\n                    devices[it->first->device] = true;\n                }\n                ggml_sycl_execution_reset_backend_binding_state(it->first);\n",
         "                ggml_sycl_execution_reset_backend_binding_state(it->first);\n                if (it->first->device >= 0 && it->first->device < GGML_SYCL_MAX_DEVICES) {\n                    devices[it->first->device] = true;\n                }\n"),
        ("the ended context's drop pins backends again", "pins backends again", CLEAR_BIND_SIG,
         "    bool devices[GGML_SYCL_MAX_DEVICES] = {};\n",
         "    bool devices[GGML_SYCL_MAX_DEVICES] = {};\n    ggml_sycl_execution_for_each_bound_backend(context_id, [](ggml_backend_sycl_context *, const ggml_sycl_execution_backend_binding &) {});\n"),
        ("the ended context's entries are dropped inside the loop, under the lock", "does not drop its registry entries by the ended id", CLEAR_BIND_SIG,
         "                it = g_execution_backend_bindings.erase(it);\n",
         "                ggml_sycl_execution_drop_context_registry_entries(context_id, devices);\n                it = g_execution_backend_bindings.erase(it);\n"),
        ("the ended context's drop returns first", "does not erase the section then the host", DROP_ENTRIES_SIG,
         "    for (int device = 0; device < GGML_SYCL_MAX_DEVICES; ++device) {\n",
         "    return;\n    for (int device = 0; device < GGML_SYCL_MAX_DEVICES; ++device) {\n"),
        ("the ended context's drop erases another id as well", "does not erase the section then the host", DROP_ENTRIES_SIG,
         "    for (int device = 0; device < GGML_SYCL_MAX_DEVICES; ++device) {\n",
         "    ggml_sycl_published_section_erase(0, 1);\n    for (int device = 0; device < GGML_SYCL_MAX_DEVICES; ++device) {\n"),
        ("the ended context's host reservation is kept", "does not erase the section then the host", DROP_ENTRIES_SIG,
         "            ggml_sycl_host_tenants_erase(device, context_id);\n", ""),
        ("the ended context's section is kept", "does not erase the section then the host", DROP_ENTRIES_SIG,
         "            ggml_sycl_published_section_erase(device, context_id);\n", ""),
        ("the ended context's erases run in the other order", "does not erase the section then the host", DROP_ENTRIES_SIG,
         "            ggml_sycl_published_section_erase(device, context_id);\n            ggml_sycl_host_tenants_erase(device, context_id);\n",
         "            ggml_sycl_host_tenants_erase(device, context_id);\n            ggml_sycl_published_section_erase(device, context_id);\n"),
        ("the ended context's drop covers every device", "does not erase the section then the host", DROP_ENTRIES_SIG,
         "if (devices[device]) {", "if (true) {"),
        ("the ended context's entries are dropped under the binding mutex in the helper", "dropped under the binding mutex", DROP_ENTRIES_SIG,
         "    for (int device = 0; device < GGML_SYCL_MAX_DEVICES; ++device) {\n",
         "    std::lock_guard<std::mutex> h_lock(g_execution_backend_binding_mutex);\n    for (int device = 0; device < GGML_SYCL_MAX_DEVICES; ++device) {\n"),
        ("the ended context's drop gets a try", "dropped under the binding mutex", DROP_ENTRIES_SIG,
         "    for (int device = 0; device < GGML_SYCL_MAX_DEVICES; ++device) {\n",
         "    try {} catch (...) {}\n    for (int device = 0; device < GGML_SYCL_MAX_DEVICES; ++device) {\n"),
        # ---- the free path: each synchronize a statement of its own, unconditional
        ("sched_reserve_impl's synchronize conditional", "does not synchronize() unconditionally on the ALLOC path", REIMPL_SIG,
         "    synchronize();\n", "    if (false) synchronize();\n"),
        ("sched_reserve_impl's synchronize moved into the MEASURE branch", "does not synchronize() unconditionally on the ALLOC path", REIMPL_SIG,
         re.compile(r"(if \(mode == sched_reserve_mode::MEASURE\) \{\n)(\s*return sched_measure_impl\(state\);\s*\}\s*(?:LLAMA_LOG_INFO\([^;]*\);\s*))synchronize\(\);"),
         r"\1        synchronize();\n\2"),
        ("sched_reserve_impl's synchronize after the first statement of the ALLOC path", "does not synchronize() unconditionally on the ALLOC path", REIMPL_SIG,
         re.compile(r"synchronize\(\);(\s*)(const int64_t t_start_us = ggml_time_us\(\);)"), r"\2\1synchronize();"),
        ("sched_reserve_impl's synchronize behind a condition on a flag", "does not synchronize() unconditionally on the ALLOC path", REIMPL_SIG,
         "    synchronize();\n", "    if (state.cparams.n_ctx != 0) { synchronize(); }\n"),
        ("the backend synchronize skips stream 0", "the backend synchronize no longer drains, in sequence and unconditionally", SYNC_SIG,
         "CHECK_TRY_ERROR(stream->wait_and_throw())", "0"),
        ("the backend synchronize skips the stream's wait under a ternary", "the backend synchronize no longer drains, in sequence and unconditionally", SYNC_SIG,
         "CHECK_TRY_ERROR(stream->wait_and_throw())", "(false ? 0 : CHECK_TRY_ERROR(stream->wait_and_throw()))"),
        ("the backend synchronize skips the CPU-expert flush", "the backend synchronize no longer drains, in sequence and unconditionally", SYNC_SIG,
         "            ggml_sycl_cpu_tg_flush_pending();\n", ""),
        ("the backend synchronize's CPU-expert flush under if (false)", "the backend synchronize no longer drains, in sequence and unconditionally", SYNC_SIG,
         "            ggml_sycl_cpu_tg_flush_pending();\n", "            if (false) ggml_sycl_cpu_tg_flush_pending();\n"),
        ("the backend synchronize's CPU-expert flush behind a false condition", "the backend synchronize no longer drains, in sequence and unconditionally", SYNC_SIG,
         "if (!ggml_sycl_cpu_tg_synchronize_flush_skipped_for_test()) {", "if (false) {"),
        ("the backend synchronize skips the deferred decode event", "the backend synchronize no longer drains, in sequence and unconditionally", SYNC_SIG,
         "sycl_ctx->last_graph_event->wait_and_throw()", "0"),
        ("the backend synchronize's deferred-event arm is never chosen", "the backend synchronize no longer drains, in sequence and unconditionally", SYNC_SIG,
         "use_deferred_decode_event ? CHECK_TRY_ERROR(", "false ? CHECK_TRY_ERROR("),
        ("the backend synchronize returns before the drain", "the backend synchronize no longer drains, in sequence and unconditionally", SYNC_SIG,
         "        auto err = use_deferred_decode_event", "        if (sycl_ctx->has_pending_barrier) {\n            return;\n        }\n        auto err = use_deferred_decode_event"),
        # ---- the queue aliasing: one execution queue per device, answered for every idx
        ("stream() answers a private queue for idx > 0 ahead of the execution queue", "no longer answers the device's one execution queue for every idx", STREAM_SIG,
         "if (sycl::queue * execution_queue = ggml_sycl_execution_queue_for_device(device)) {",
         "if (stream != 0 && qptrs[device][stream] != nullptr) {\n            return qptrs[device][stream];\n        }\n        if (sycl::queue * execution_queue = ggml_sycl_execution_queue_for_device(device)) {"),
        ("the execution queue of device 1 is a private queue ahead of the TP check", "execution queue is no longer the TP queue or the unified cache's own queue", EXEC_QUEUE_SIG,
         "    if (sycl::queue * tp_queue = ggml_sycl_get_tp_queue(device)) {",
         "    if (device == 1) {\n        return &ggml_sycl_get_device(1).default_queue();\n    }\n    if (sycl::queue * tp_queue = ggml_sycl_get_tp_queue(device)) {"),
        ("the execution queue answers one more queue after the cache", "execution queue is no longer the TP queue or the unified cache's own queue", EXEC_QUEUE_SIG,
         "    return nullptr;\n}", "    return &ggml_sycl_get_device(device).default_queue();\n}"),
        ("the section's erase indexes device 0", "does not drop the section by the id it was given", ERASE_SECTION_SIG,
         "ggml_sycl_kv_region_registry(device)", "ggml_sycl_kv_region_registry(0)"),
        ("the host reservation's erase indexes device 0", "does not take the table by the id it was given", ERASE_HOST_SIG,
         "ggml_sycl_kv_region_registry(device)", "ggml_sycl_kv_region_registry(0)"),
        ("the section's erase drops its id==0 guard", "does not drop the section by the id it was given", ERASE_SECTION_SIG,
         "context_id == 0 || ", ""),
        ("the host reservation's erase drops its id==0 guard", "does not take the table by the id it was given", ERASE_HOST_SIG,
         "context_id == 0 || ", ""),
        ("the section's erase drops its device bounds", "does not drop the section by the id it was given", ERASE_SECTION_SIG,
         "context_id == 0 || device < 0 || device >= GGML_SYCL_MAX_DEVICES", "context_id == 0"),
        ("the host reservation's erase drops its device bounds", "does not take the table by the id it was given", ERASE_HOST_SIG,
         "context_id == 0 || device < 0 || device >= GGML_SYCL_MAX_DEVICES", "context_id == 0"),
        ("the stream block returns a second queue from inside the re-point", "no longer answers the device's one execution queue for every idx", STREAM_SIG,
         "qptrs[device][stream] = execution_queue;",
         "qptrs[device][stream] = execution_queue;\n                if (stream != 0) return &ggml_sycl_get_device(device).default_queue();"),
        ("the host reservation's erase rethrows after the log", "does not take the table by the id it was given", ERASE_HOST_SIG,
         "pinned until the process ends\\n\");", "pinned until the process ends\\n\");\n        throw;"),
        ("the section's erase rethrows after the log", "does not drop the section by the id it was given", ERASE_SECTION_SIG,
         "process ends\\n\");", "process ends\\n\");\n        throw;"),
        ("the section's erase drops again on device 0 after the catch", "does not drop the section by the id it was given", ERASE_SECTION_SIG,
         "process ends\\n\");\n    }", "process ends\\n\");\n    }\n    (void) ggml_sycl_kv_region_registry(0).drop_published_section(context_id);"),
        ("the host reservation's erase takes again on device 0 after the catch", "does not take the table by the id it was given", ERASE_HOST_SIG,
         "pinned until the process ends\\n\");\n    }", "pinned until the process ends\\n\");\n    }\n    (void) ggml_sycl_kv_region_registry(0).take_tenant_slots(context_id);"),
        ("the host reservation's erase keeps the table", "does not take the table by the id it was given", ERASE_HOST_SIG,
         "(void) previous;", "static std::shared_ptr<ggml_sycl::kv_tenant_slots> h_keep; h_keep = previous;"),
        ("the section's erase keeps the section", "does not drop the section by the id it was given", ERASE_SECTION_SIG,
         "(void) previous;", "static auto h_keep = previous; h_keep = previous;"),
        ("the collection loop selects the other contexts", "does not select the bindings of the ended context by its id", CLEAR_BIND_SIG,
         "it->second->context_id == context_id", "it->second->context_id != context_id"),
        ("stream(device, idx) answers another queue for idx > 0", "no longer answers the device's one execution queue for every idx", STREAM_SIG,
         "            return execution_queue;\n", "            return stream == 0 ? execution_queue : qptrs[device][stream];\n"),
        ("stream(device, idx) answers by idx under a condition", "no longer answers the device's one execution queue for every idx", STREAM_SIG,
         "if (sycl::queue * execution_queue = ggml_sycl_execution_queue_for_device(device)) {",
         "if (sycl::queue * execution_queue = stream == 0 ? ggml_sycl_execution_queue_for_device(device) : nullptr) {"),
        ("the execution queue is the device's default queue", "execution queue is no longer the TP queue or the unified cache's own queue", EXEC_QUEUE_SIG,
         "return &cache->get_queue();", "return &ggml_sycl_get_device(device).default_queue();"),
        ("the execution queue ignores the TP queue", "execution queue is no longer the TP queue or the unified cache's own queue", EXEC_QUEUE_SIG,
         "        return tp_queue;\n", "        return nullptr;\n"),
        # ---- the upstream premises: scheduler and allocator synchronize before they free
        ("the scheduler is built over one backend fewer", "no longer built over every backend of the context", REIMPL_SIG,
         re.compile(r"backend_ptrs\.size\(\), max_nodes,"), "backend_ptrs.size() - 1, max_nodes,"),
        ("ggml_backend_sched_reserve does not synchronize", "ggml_backend_sched_reserve no longer synchronizes", SCHED_RESERVE_SIG,
         "    ggml_backend_sched_synchronize(sched);\n", ""),
        ("ggml_backend_sched_reserve synchronizes under if (false)", "ggml_backend_sched_reserve no longer synchronizes", SCHED_RESERVE_SIG,
         "    ggml_backend_sched_synchronize(sched);\n", "    if (false) ggml_backend_sched_synchronize(sched);\n"),
        ("ggml_backend_sched_reserve_size does not synchronize", "ggml_backend_sched_reserve_size no longer synchronizes", SCHED_RESERVE_SIZE_SIG,
         "    ggml_backend_sched_synchronize(sched);\n", ""),
        ("the realloc in alloc_splits does not synchronize", "the reallocation in ggml_backend_sched_alloc_splits no longer synchronizes", SCHED_ALLOC_SPLITS_SIG,
         "            ggml_backend_synchronize(sched->backends[i]);\n", ""),
        ("the realloc in alloc_splits synchronizes only the first backend", "the reallocation in ggml_backend_sched_alloc_splits no longer synchronizes", SCHED_ALLOC_SPLITS_SIG,
         "for (int i = 0; i < sched->n_backends; i++) {\n            ggml_backend_synchronize",
         "for (int i = 0; i < 1; i++) {\n            ggml_backend_synchronize"),
        ("alloc_splits reserves a second time", "reserves in more than one place", SCHED_ALLOC_SPLITS_SIG,
         "        if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {\n            GGML_LOG_ERROR(\"%s: failed to allocate graph\\n\", __func__);",
         "        (void) ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids);\n        if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {\n            GGML_LOG_ERROR(\"%s: failed to allocate graph\\n\", __func__);"),
        ("ggml_backend_sched_synchronize skips a backend", "no longer synchronizes every backend", SCHED_SYNC_SIG,
         "        ggml_backend_synchronize(sched->backends[i]);\n", ""),
        ("ggml_backend_sched_synchronize stops at the first backend", "no longer synchronizes every backend", SCHED_SYNC_SIG,
         "for (int i = 0; i < sched->n_backends; i++) {", "for (int i = 0; i < 1; i++) {"),
        ("the allocator reallocates multi-buffer graphs automatically", "no longer single-buffer only", GALLOC_ALLOC_GRAPH_SIG,
         "if (galloc->n_buffers == 1) {", "if (galloc->n_buffers >= 1) {"),
        ("the allocator never refuses a multi-buffer realloc", "no longer single-buffer only", GALLOC_ALLOC_GRAPH_SIG,
         "        } else {\n", "        } else if (false) {\n"),
        ("the install refusal answers EFFECT_FAILED", "the install result is not a refusal with nothing published", IMPL_SIG,
         INSTALL_REFUSAL_TAIL, INSTALL_REFUSAL_TAIL.replace("PLAN_REJECTED", "EFFECT_FAILED")),
        ("the section stored without keeping the table", "is stored without keeping the installed table", IMPL_SIG,
         "            host_tenants_installed.keep();\n", ""),
        ("the table kept before the section store", "is stored without keeping the installed table", IMPL_SIG,
         "            ggml_sycl_published_section_set(backend_ctx, section);\n            host_tenants_installed.keep();\n",
         "            host_tenants_installed.keep();\n            ggml_sycl_published_section_set(backend_ctx, section);\n"),
        ("the publish holds no rollback guard", "holds no rollback guard", IMPL_SIG,
         "    ggml_sycl_host_tenants_install_guard host_tenants_installed;\n", ""),
        ("a refused reservation is ignored", "does not reserve only without a held table, or does not refuse", IMPL_SIG,
         "                        return GGML_SYCL_LIFECYCLE_PLAN_REJECTED;\n                    }\n                } else if",
         "                    }\n                } else if"),
        ("a republish allocates again", "does not reserve only without a held table, or does not refuse", IMPL_SIG,
         "if (held == nullptr) {", "if (true) {"),
        ("the republish carry check dropped", "does not reserve only without a held table, or does not refuse", IMPL_SIG,
         "} else if (!ggml_sycl_host_tenants_carry(*held, *section, refusal)) {", "} else if (false) {"),
        ("the carry accepts a larger slot", "a republish is carried by a missing or smaller held slot", CARRY_SIG,
         "if ((size_t) e.slot_bytes > held_bytes) {", "if (false) {"),
        ("the carry accepts a slot never held", "a republish is carried by a missing or smaller held slot", CARRY_SIG,
         "if (held_bytes == 0) {", "if (false) {"),
        ("the table reset before the registry took it", "the table is dropped from the caller before the registry took it", INSTALL_SIG,
         "    const uint64_t id = ggml_sycl_context_execution_id(ctx);\n", "    table.reset();\n    const uint64_t id = ggml_sycl_context_execution_id(ctx);\n"),
        ("the carve halved", "the host carve is not the slot's full byte count", RESERVE_SIG,
         "req.size = (size_t) e.slot_bytes;", "req.size = (size_t) e.slot_bytes / 2;"),
        ("a refused carve returns true", "a refused carve does not refuse the reservation", RESERVE_SIG,
         "\" failed\";\n            return false;", "\" failed\";\n            return true;"),
        ("the reservation swallows an exception", "swallows or converts an exception itself", RESERVE_SIG,
         "    out.reset();\n", "    out.reset();\n    try {\n    } catch (...) {\n    }\n"),
        ("the tier check dropped", "does not refuse a cohort that is not host-pinned", RESERVE_SIG,
         "info == nullptr || info->tier != GGML_SYCL_CONTEXT_COHORT_TIER_HOST_PINNED", "info == nullptr"),
        ("the on-device check dropped from the reservation", "accepts a carve that did not resolve to host memory", RESERVE_SIG,
         "if (!resolved.ptr || resolved.on_device) {", "if (!resolved.ptr) {"),
        ("the carve never added to the table", "does not add its carves to the table", RESERVE_SIG,
         "(void) table->add(info->name, e.slot_index,", "(void) (info->name, e.slot_index,"),
        ("the slot's deleter check dropped", "without checking that the slot's deleter made it", SLOT_OF_SIG,
         "std::get_deleter<ggml_sycl_host_tenant_slot_deleter>(handle) == nullptr", "false"),
        ("a slot made without its deleter", "without the deleter its reader checks", SLOT_MAKE_SIG,
         "ggml_sycl::kv_region_handle(slot, ggml_sycl_host_tenant_slot_deleter{})", "ggml_sycl::kv_region_handle(slot)"),
        ("a refused claim falls through to the allocator", "falls through to the legacy allocator", ALLOC_SIG,
         "        if (!ggml_backend_sycl_host_buffer_claim_slot(size, exact_device, claim, claimed_ptr, claimed_handle)) {\n            return nullptr;\n        }\n", ""),
        ("the claim branch no longer returns", "falls through to the legacy allocator", ALLOC_SIG,
         "        return ggml_backend_sycl_host_buffer_wrap(buft, claimed_ptr,", "        (void) ggml_backend_sycl_host_buffer_wrap(buft, claimed_ptr,"),
        ("the claim in flight owned by a raw pointer", "not owned by an RAII record", CLAIM_SLOT_SIG,
         "auto made = std::make_unique<ggml_sycl::tenant_claim>();", "ggml_sycl::tenant_claim * made = new ggml_sycl::tenant_claim();"),
        ("the on-device check dropped from the claim", "a claim over memory that is not host memory is served", CLAIM_SLOT_SIG,
         "if (!slot || !resolved.ptr || resolved.on_device) {", "if (!slot || !resolved.ptr) {"),
        ("the claim released by hand in claim_slot", "claim_slot releases a claim by hand", CLAIM_SLOT_SIG,
         "        return false;\n    }\n    const ggml_sycl_host_tenant_slot * slot",
         "        (void) ggml_sycl::tenant_claim_scope::release(*made, 0);\n        return false;\n    }\n    const ggml_sycl_host_tenant_slot * slot"),
        ("wrap takes the claim by reference", "wrap does not take its claim by value", WRAP_SIG,
         "std::unique_ptr<ggml_sycl::tenant_claim> claim) {", "std::unique_ptr<ggml_sycl::tenant_claim> & claim) {"),
        ("wrap's failed context allocation leaks the CPU buffer", "a failed context allocation does not free the CPU buffer", WRAP_SIG,
         "    if (!ctx) {\n        ggml_backend_buffer_free(buffer);\n        return nullptr;\n    }", "    if (!ctx) {\n        return nullptr;\n    }"),
        ("wrap's refused set_type leaks the context", "a refused set_type does not free the buffer", WRAP_SIG,
         "        delete ctx;", ""),
        ("the free waits on the host", "free_buffer waits on the host", FREE_SIG,
         "    if (ctx->claim) {\n", "    if (ctx->claim) {\n        ggml_backend_sycl_host_buffer_sync(ctx);\n"),
        ("the free waits on a queue", "free_buffer waits on the host", FREE_SIG,
         "    if (ctx->claim) {\n", "    if (ctx->claim) {\n        ggml_sycl_get_device(ctx->device).default_queue().wait_and_throw();\n"),
        ("open's nested check dropped", "does not answer NESTED before it reads", OPEN_SIG,
         "if (ggml_sycl::tenant_claim_scope::active()) {", "if (false) {"),
        ("open's foreign-backend check dropped", "does not refuse a foreign backend", OPEN_SIG,
         "        ggml_backend_dev_backend_reg(backend->device) != ggml_backend_sycl_reg()) {", "        false) {"),
        ("open answers FAILED for no reservation", "does not map its three outcomes", OPEN_SIG,
         "open_status::NO_TABLE:\n                return GGML_SYCL_CLAIM_SCOPE_NO_RESERVATION;",
         "open_status::NO_TABLE:\n                return GGML_SYCL_CLAIM_SCOPE_FAILED;"),
        ("open answers NO_RESERVATION for a nested open", "does not map its three outcomes", OPEN_SIG,
         "open_status::NESTED:\n                return GGML_SYCL_CLAIM_SCOPE_NESTED;",
         "open_status::NESTED:\n                return GGML_SYCL_CLAIM_SCOPE_NO_RESERVATION;"),
        ("open leaves the out pointer unwritten", "refuse a null out pointer first and clear it", OPEN_SIG,
         "    *scope = nullptr;\n", ""),
        ("open does not check its out pointer", "refuse a null out pointer first and clear it", OPEN_SIG,
         "    if (scope == nullptr) {\n        return GGML_SYCL_CLAIM_SCOPE_FAILED;\n    }\n", ""),
        ("open's catch answers NO_RESERVATION", "open's catch does not answer FAILED", OPEN_SIG,
         "} catch (...) {\n        return GGML_SYCL_CLAIM_SCOPE_FAILED;", "} catch (...) {\n        return GGML_SYCL_CLAIM_SCOPE_NO_RESERVATION;"),
        ("claims dereferences an unchecked scope", "claims reads the scope without", CLAIMS_SIG,
         "    if (scope == nullptr) {\n        return 0;\n    }\n", ""),
        ("close skips the open-scope check", "close does not check that the scope", CLOSE_SIG,
         "if (!ggml_sycl::tenant_claim_scope::close(", "if (false && !ggml_sycl::tenant_claim_scope::close("),
        ("the lock dropped from the record", "ggml_sycl_load_record_compute_term uses the ledger without state.mutex",
         SIG_RECORD, "    std::lock_guard<std::mutex> lock(state.mutex);\n", ""),
        ("the lock dropped from the clear", "ggml_sycl_load_clear_compute_terms uses the ledger without state.mutex",
         SIG_CLEAR, "        std::lock_guard<std::mutex> lock(state.mutex);\n", ""),
        ("the lock dropped from the late check", "ggml_backend_sycl_load_late_check uses the ledger without state.mutex",
         SIG_LATE, "            std::lock_guard<std::mutex> lock(state.mutex);\n", ""),
        ("the lock dropped from the count hook", "ggml_backend_sycl_test_compute_term_count uses the ledger without state.mutex",
         SIG_COUNT, "    std::lock_guard<std::mutex> lock(state.mutex);\n", ""),
        ("the open read moved before the lock in the late check", "ggml_backend_sycl_load_late_check reads the open transaction outside",
         SIG_LATE,
         "            std::lock_guard<std::mutex> lock(state.mutex);\n            r = state.ledger.check(txn.id, device, compute_bytes, ggml_sycl_load_txn_is_open(txn.id));",
         "            const bool open_early = ggml_sycl_load_txn_is_open(txn.id);\n            std::lock_guard<std::mutex> lock(state.mutex);\n            r = state.ledger.check(txn.id, device, compute_bytes, open_early);"),
        ("the record not gated by the open transaction", "the record is not gated by the open transaction",
         SIG_RECORD, "ggml_sycl_load_txn_is_open(txn)", "true"),
        ("the late check not gated by the open transaction", "the late check is not gated by the open transaction",
         SIG_LATE, "ggml_sycl_load_txn_is_open(txn.id)", "true"),
        ("the open test flipped to a mismatch", "is not `txn != 0 && active_txn == txn`",
         r"\bstatic\s+bool\s+ggml_sycl_load_txn_is_open\s*\(", "active_txn == txn", "active_txn != txn"),
        ("the open test accepts transaction 0", "is not `txn != 0 && active_txn == txn`",
         r"\bstatic\s+bool\s+ggml_sycl_load_txn_is_open\s*\(", "txn != 0 &&", "txn != 0 ||"),
        ("the late check's catch answers EQUAL", "the late check's catch does not answer NOT_RECORDED",
         SIG_LATE, "catch (...) {\n        return GGML_SYCL_LATE_CHECK_NOT_RECORDED;", "catch (...) {\n        return GGML_SYCL_LATE_CHECK_EQUAL;"),
        ("the late check's closed module answers EQUAL", "the late check lost its module guard",
         SIG_LATE, "    if (!module_guard) {\n        return GGML_SYCL_LATE_CHECK_NOT_RECORDED;", "    if (!module_guard) {\n        return GGML_SYCL_LATE_CHECK_EQUAL;"),
        ("the coverage catch answers COVERED", "the coverage query's catch does not answer GROWTH",
         SIG_COVERAGE, "catch (...) {\n        return GGML_SYCL_TENANT_COVERAGE_GROWTH;", "catch (...) {\n        return GGML_SYCL_TENANT_COVERAGE_COVERED;"),
        ("the coverage unbound-context guard dropped", "of an unbound context does not answer GROWTH",
         SIG_COVERAGE, "        if (id == 0) {\n            return GGML_SYCL_TENANT_COVERAGE_GROWTH;\n        }\n", ""),
        ("the coverage module guard dropped", "the coverage query lost its module guard",
         SIG_COVERAGE, "    if (!module_guard || ", "    if ("),
        ("the ledger clear's catch answers 1", "the ledger clear's catch does not answer 0",
         SIG_CLEAR, "    } catch (...) {\n        return 0;", "    } catch (...) {\n        return 1;"),
        ("the refusal logged at INFO", "a ledger line of level ERROR is not logged through GGML_LOG_ERROR",
         r"\bstatic\s+void\s+ggml_sycl_load_ledger_log\s*\(",
         "case ggml_sycl::LOAD_LOG_LEVEL_ERROR:\n            GGML_LOG_ERROR(", "case ggml_sycl::LOAD_LOG_LEVEL_ERROR:\n            GGML_LOG_INFO("),
        ("the not-open line logged at INFO", "a ledger line of level WARN is not logged through GGML_LOG_WARN",
         r"\bstatic\s+void\s+ggml_sycl_load_ledger_log\s*\(",
         "case ggml_sycl::LOAD_LOG_LEVEL_WARN:\n            GGML_LOG_WARN(", "case ggml_sycl::LOAD_LOG_LEVEL_WARN:\n            GGML_LOG_INFO("),
        ("the late check bypasses the ledger's level", "does not log through the ledger's level",
         SIG_LATE, "ggml_sycl_load_ledger_log(r);", "GGML_LOG_INFO(\"%s\\n\", r.line.c_str());"),
        ("a publish for an unbound context is silent", "a publish for an unbound context is silent",
         r"\bstatic\s+void\s+ggml_sycl_published_section_set\s*\(", "        if (section) {\n            GGML_LOG_WARN(", "        if (false) {\n            GGML_LOG_WARN("),
        ("the claim after the allocation", "does not claim inside a scope before it allocates",
         r"\bstatic\s+ggml_backend_buffer_t\s+ggml_backend_sycl_host_buffer_type_alloc_buffer\s*\(",
         "    if (ggml_sycl::tenant_claim_scope::active()) {",
         "    { ggml_sycl::alloc_handle early{}; (void) ggml_sycl::unified_alloc({}, &early); }\n    if (ggml_sycl::tenant_claim_scope::active()) {"),
        ("the unbound-context return dropped from the section set", "goes on to the registry with key 0",
         r"\bstatic\s+void\s+ggml_sycl_published_section_set\s*\(", "        }\n        return;\n    }\n", "        }\n    }\n"),
        ("the recovery arm no longer arms the clear", "the recovery arm does not set after_end.txn",
         LOAD_END_SIG, "            after_end.txn = txn.id;\n            ggml_sycl_finalize_binding_failure_abort", "            ggml_sycl_finalize_binding_failure_abort"),
        ("the recovery arm arms the clear after the abort", "the recovery arm does not set after_end.txn",
         LOAD_END_SIG, "            after_end.txn = txn.id;\n            ggml_sycl_finalize_binding_failure_abort(*registry, recovery);\n",
         "            ggml_sycl_finalize_binding_failure_abort(*registry, recovery);\n            after_end.txn = txn.id;\n"),
        ("the recovery arm's arm under if (false)", "the recovery arm does not set after_end.txn",
         LOAD_END_SIG, "            after_end.txn = txn.id;\n            ggml_sycl_finalize_binding_failure_abort",
         "            if (false) {\n                after_end.txn = txn.id;\n            }\n            ggml_sycl_finalize_binding_failure_abort"),
        ("the handler's after_end declaration dropped", "after_end is not the handler's first statement",
         LOAD_END_SIG, "        ggml_sycl_load_ledger_clear_after_end after_end;\n", ""),
        ("the handler rethrows without clearing", "model_load_end's handler rethrows",
         LOAD_END_SIG, "        ggml_sycl_load_ledger_clear_after_end after_end;\n",
         "        ggml_sycl_load_ledger_clear_after_end after_end;\n        throw;\n"),
        ("a second catch in the end call", "must have exactly one try and one catch",
         LOAD_END_SIG, "        return GGML_SYCL_LIFECYCLE_EFFECT_FAILED;\n    }\n}",
         "        return GGML_SYCL_LIFECYCLE_EFFECT_FAILED;\n    } catch (const std::bad_alloc &) {\n        return GGML_SYCL_LIFECYCLE_EFFECT_FAILED;\n    }\n}"),
        ("the end call clears the ledger itself", "model_load_end clears the ledger itself",
         LOAD_END_SIG, "        ggml_sycl_load_ledger_clear_after_end after_end;\n",
         "        ggml_sycl_load_ledger_clear_after_end after_end;\n        (void) ggml_sycl_load_clear_compute_terms(txn.id);\n"),
        ("the try-scope guard clears while unwinding", "the try-scope guard clears during unwinding",
         r"\bstruct\s+ggml_sycl_load_ledger_clear_guard", "        if (std::uncaught_exceptions() > uncaught) {\n            return;\n        }\n", ""),
        ("the try-scope guard's snapshot dropped", "does not snapshot uncaught_exceptions",
         r"\bstruct\s+ggml_sycl_load_ledger_clear_guard", "uncaught(std::uncaught_exceptions())", "uncaught(0)"),
        ("after_end clears a transaction it was never armed with", "after_end does not clear only a transaction it was armed with",
         r"\bstruct\s+ggml_sycl_load_ledger_clear_after_end", "        if (txn != 0) {\n            (void) ggml_sycl_load_clear_compute_terms(txn);\n        }\n",
         "        (void) ggml_sycl_load_clear_compute_terms(txn);\n"),
        ("the lock scoped away in the late check", "ggml_backend_sycl_load_late_check uses the ledger without state.mutex held at the use",
         SIG_LATE, "            std::lock_guard<std::mutex> lock(state.mutex);\n", "            { std::lock_guard<std::mutex> lock(state.mutex); }\n"),
        ("the lock scoped away in the record", "ggml_sycl_load_record_compute_term uses the ledger without state.mutex held at the use",
         SIG_RECORD, "    std::lock_guard<std::mutex> lock(state.mutex);\n", "    { std::lock_guard<std::mutex> lock(state.mutex); }\n"),
        ("the lock scoped away in the clear", "ggml_sycl_load_clear_compute_terms uses the ledger without state.mutex held at the use",
         SIG_CLEAR, "        std::lock_guard<std::mutex> lock(state.mutex);\n", "        { std::lock_guard<std::mutex> lock(state.mutex); }\n"),
        ("the lock scoped away in the count hook", "ggml_backend_sycl_test_compute_term_count uses the ledger without state.mutex held at the use",
         SIG_COUNT, "    std::lock_guard<std::mutex> lock(state.mutex);\n", "    { std::lock_guard<std::mutex> lock(state.mutex); }\n"),
        ("the late check answers EQUAL for every result", "the late check does not answer the ledger's result",
         SIG_LATE, "        return r.result;\n", "        return GGML_SYCL_LATE_CHECK_EQUAL;\n"),
        ("the publish tail's drop swallowed in place", "a try or catch sits between the runtime_kv_admitted store",
         TXN_SIG, "    ggml_sycl_published_section_set(ctx, nullptr);\n    return ggml_sycl_txn_result::ACCEPTED;",
         "    try {\n        ggml_sycl_published_section_set(ctx, nullptr);\n    } catch (...) {\n    }\n    return ggml_sycl_txn_result::ACCEPTED;"),
        ("the C entry lets a system_error out", "must catch system_error, then std::exception",
         CENTRY_SIG, "    } catch (const std::system_error & e) {", "    } catch (const std::logic_error & e) {"),
        ("the C entry lets a non-standard exception out", "must catch system_error, then std::exception",
         CENTRY_SIG, "    } catch (...) {", "    } catch (const std::bad_alloc &) {"),
        ("the C entry's arm rethrows", "every arm of the C entry must log at ERROR and none may rethrow",
         CENTRY_SIG, "    } catch (const std::exception & e) {", "    } catch (const std::exception & e) {\n        throw;"),
        ("the C entry's transaction called outside the try", "the runtime context transaction is called outside the C entry's try",
         CENTRY_SIG, "    try {\n        (void) ggml_sycl_run_runtime_context_transaction(", "    {\n        (void) ggml_sycl_run_runtime_context_transaction("),
        ("the descriptor publish calls the swallowing C entry", "the descriptor publish must call the transaction in its own try",
         r"\bggml_sycl_set_runtime_context_for_model_impl\s*\(",
         "        (void) ggml_sycl_run_runtime_context_transaction(backend, n_ctx, n_ubatch, n_seq_max, kv_unified, swa_full,",
         "        ggml_backend_sycl_set_runtime_context(backend, n_ctx, n_ubatch, n_seq_max, kv_unified, swa_full,"),
        ("the finisher is never armed at the top of the handler", "the handler does not arm after_end for the finisher",
         LOAD_END_SIG, "        if (ticket.finisher) {\n            after_end.txn = txn.id;", "        if (ticket.finisher) {\n            (void) 0;"),
        ("the finisher is armed under if (false)", "the handler does not arm after_end for the finisher",
         LOAD_END_SIG, "        if (ticket.finisher) {\n            after_end.txn = txn.id;", "        if (false) {\n            after_end.txn = txn.id;"),
        ("an early return before the handler arms", "the handler does not arm after_end for the finisher",
         LOAD_END_SIG, "        ggml_sycl_load_ledger_clear_after_end after_end;\n",
         "        ggml_sycl_load_ledger_clear_after_end after_end;\n        if (!registry) {\n            return GGML_SYCL_LIFECYCLE_EFFECT_FAILED;\n        }\n"),
        ("a goto before the handler arms", "the handler does not arm after_end for the finisher",
         LOAD_END_SIG, "        ggml_sycl_load_ledger_clear_after_end after_end;\n",
         "        ggml_sycl_load_ledger_clear_after_end after_end;\n        goto h_out;\n"),
        ("the finisher arm resets the arming", "after_end.txn is set other than once at the top",
         LOAD_END_SIG, "            const auto failed = registry->finalize_end(ticket, false);",
         "            after_end.txn = 0;\n            const auto failed = registry->finalize_end(ticket, false);"),
        ("the finisher arm skips the owner-effects abort", "the finisher arm does not abort the owner effects and then end",
         LOAD_END_SIG, "            (void) ggml_sycl_abort_owner_effects_noexcept(ticket.token, \"load_end/exception-rollback\");\n", ""),
        ("the guard is braced away from the end call", "the clear guard is not directly in the try body",
         LOAD_END_SIG, "        ggml_sycl_load_ledger_clear_guard ledger_clear{ txn.id };\n",
         "        { ggml_sycl_load_ledger_clear_guard ledger_clear{ txn.id }; }\n"),
        ("the guard is conditional", "the clear guard is not directly in the try body",
         LOAD_END_SIG, "        ggml_sycl_load_ledger_clear_guard ledger_clear{ txn.id };\n",
         "        if (txn.id != 0) ggml_sycl_load_ledger_clear_guard ledger_clear{ txn.id };\n"),
        ("the impl's catch answers OK", "that try's catch answers EFFECT_FAILED",
         IMPL_SIG, "return GGML_SYCL_LIFECYCLE_EFFECT_FAILED;\n    }\n    const bool inner_ok",
         "return GGML_SYCL_LIFECYCLE_OK;\n    }\n    const bool inner_ok"),
        ("the record's lock is a substatement of an if", "ggml_sycl_load_record_compute_term uses the ledger without state.mutex held at the use",
         SIG_RECORD, "    std::lock_guard<std::mutex> lock(state.mutex);\n", "    if (false) std::lock_guard<std::mutex> lock(state.mutex);\n"),
        ("the clear's lock is a substatement of an if", "ggml_sycl_load_clear_compute_terms uses the ledger without state.mutex held at the use",
         SIG_CLEAR, "        std::lock_guard<std::mutex> lock(state.mutex);\n", "        if (false) std::lock_guard<std::mutex> lock(state.mutex);\n"),
        ("the late check's lock is a substatement of an if", "ggml_backend_sycl_load_late_check uses the ledger without state.mutex held at the use",
         SIG_LATE, "            std::lock_guard<std::mutex> lock(state.mutex);\n", "            if (false) std::lock_guard<std::mutex> lock(state.mutex);\n"),
        ("the count hook's lock is a substatement of an if", "ggml_backend_sycl_test_compute_term_count uses the ledger without state.mutex held at the use",
         SIG_COUNT, "    std::lock_guard<std::mutex> lock(state.mutex);\n", "    if (false) std::lock_guard<std::mutex> lock(state.mutex);\n"),
        ("the free releases with an event", "does not release then drop its claim as the first step", FREE_SIG,
         "tenant_claim_scope::release(*ctx->claim, 0)", "tenant_claim_scope::release(*ctx->claim, ctx->last_event)"),
        ("the free drops the release", "does not release its claim", FREE_SIG,
         "        (void) ggml_sycl::tenant_claim_scope::release(*ctx->claim, 0);\n", ""),
        ("the free releases twice", "releases other than once through tenant_claim_scope::release", FREE_SIG,
         "        (void) ggml_sycl::tenant_claim_scope::release(*ctx->claim, 0);\n",
         "        (void) ggml_sycl::tenant_claim_scope::release(*ctx->claim, 0);\n        (void) ggml_sycl::tenant_claim_scope::release(*ctx->claim, 0);\n"),
        ("the free also frees through the allocator", "releases other than once through tenant_claim_scope::release", FREE_SIG,
         "    ctx->buffer_handle = {};\n", "    ggml_sycl::unified_free(ctx->buffer_handle);\n    ctx->buffer_handle = {};\n"),
        ("the destructor's first synchronize dropped", "~llama_context does not synchronize() first", DTOR_SIG,
         re.compile(r"(?<=\n)    synchronize\(\);\n"), ""),
        ("the destructor's first synchronize conditional", "~llama_context does not synchronize() first", DTOR_SIG,
         re.compile(r"(?<=\n)    synchronize\(\);\n"), "    if (false) synchronize();\n"),
        ("sched_reserve_impl's synchronize dropped", "sched_reserve_impl does not synchronize() before it first touches", REIMPL_SIG,
         "    synchronize();\n", ""),
        ("set_tensor_async takes the CpuActivation buffer type", "set_tensor_async accepts the CpuActivation", SET_ASYNC_SIG,
         "\"unsupported buffer type\"", "\"unsupported buffer type\" || buf->buft == ggml_backend_sycl_cpu_activation_buffer_type()"),
        ("get_tensor_async takes the CpuActivation buffer type", "accepts the CpuActivation buffer type", GET_ASYNC_SIG,
         "    GGML_ASSERT((buf->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) ||",
         "    GGML_ASSERT(buf->buft == ggml_backend_sycl_cpu_activation_buffer_type() || (buf->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) ||"),
        ("the device supports the CpuActivation buffer type", "the device supports the CpuActivation buffer type", SUPPORTS_SIG,
         "(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {\n",
         "(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {\n    if (buft == ggml_backend_sycl_cpu_activation_buffer_type()) {\n        return true;\n    }\n"),
        ("the claim record's destructor swallows silently", "swallows a failed release silently", DROP_SIG,
         re.compile(r'std::fprintf\(stderr, "WARN: \[CLAIM-SCOPE\][^;]*;'), "(void) 0;"),
        ("an exhausted one-slot cohort names no slot", "does not name the index past its last slot", CLAIM_WALK_SIG,
         "!seen ? 0 : (last == UINT32_MAX ? last : last + 1)", "last == 0 ? 0 : last + 1"),
        ("the walk wraps past the top index", "can wrap its index past UINT32_MAX", CLAIM_WALK_SIG,
         re.compile(r"if \(index == UINT32_MAX\) \{\s*break;\s*\}\s*"), ""),
        ("the carry check skips the host elements", "the carry check does not skip every element that is not host-tier", CARRY_FN_SIG,
         "if (e.device != -1) {", "if (e.device == -1) {"),
        ("the reservation skips the host elements", "the reservation does not skip every element that is not host-tier", RESERVE_SIG,
         "if (e.device != -1) {", "if (e.device == -1) {"),
        ("the carve is not a staging request", "is not a STAGING-role request", RESERVE_SIG,
         "req.intent.role = ggml_sycl::alloc_role::STAGING;", "req.intent.role = ggml_sycl::alloc_role::KV;"),
        ("the carve is for another device", "is not requested for the context's device", RESERVE_SIG,
         "req.device = ctx->device;", "req.device = 0;"),
        ("open answers INVALID_BACKEND on a closed module", "does not answer FAILED on a closed module", OPEN_SIG,
         "if (!module_guard) {\n        return GGML_SYCL_CLAIM_SCOPE_FAILED;", "if (!module_guard) {\n        return GGML_SYCL_CLAIM_SCOPE_INVALID_BACKEND;"),
        ("open answers NO_RESERVATION for an invalid device", "with no valid device as INVALID_BACKEND", OPEN_SIG,
         "ctx->device >= GGML_SYCL_MAX_DEVICES) {\n            return GGML_SYCL_CLAIM_SCOPE_INVALID_BACKEND;",
         "ctx->device >= GGML_SYCL_MAX_DEVICES) {\n            return GGML_SYCL_CLAIM_SCOPE_NO_RESERVATION;"),
        ("open answers FAILED for an unbound context", "not bound to an execution context", OPEN_SIG,
         "if (id == 0) {\n            return GGML_SYCL_CLAIM_SCOPE_NO_RESERVATION;", "if (id == 0) {\n            return GGML_SYCL_CLAIM_SCOPE_FAILED;"),
        ("the n_ctx entry stops delegating", "set_runtime_n_ctx does not delegate to the guarded C entry",
         r"\bvoid\s+ggml_backend_sycl_set_runtime_n_ctx\s*\(", "ggml_backend_sycl_set_runtime_context(backend,",
         "(void) ggml_sycl_run_runtime_context_transaction(backend,"),
    ]
    # (step 3d) the residency probe's body and its one door
    scoped.extend([
        ("the probe writes host_resident before step 1d", "touches host_resident", SIG_PROBE,
         "    out->n_layer = 0;\n    sycl_module_mutation_guard module_guard;",
         "    out->n_layer = 0;\n    out->host_resident[0] = 0;\n    sycl_module_mutation_guard module_guard;"),
        ("the probe answers OK where it answers not-wired", "returns something other than a named refusal enumerator", SIG_PROBE,
         "    return GGML_SYCL_RESIDENCY_PROBE_GEOMETRY_NOT_WIRED;\n}",
         "    return GGML_SYCL_RESIDENCY_PROBE_OK;\n}"),
        ("the probe's not-wired answer is a bare number", "returns something other than a named refusal enumerator", SIG_PROBE,
         "    return GGML_SYCL_RESIDENCY_PROBE_GEOMETRY_NOT_WIRED;\n}",
         "    return (ggml_sycl_residency_probe_status) 2;\n}"),
        ("the probe answers OK for a foreign backend", "returns something other than a named refusal enumerator", SIG_PROBE,
         "        return GGML_SYCL_RESIDENCY_PROBE_FOREIGN_BACKEND;",
         "        return GGML_SYCL_RESIDENCY_PROBE_OK;"),
        ("the probe answers a lifecycle result", "returns something other than a named refusal enumerator", SIG_PROBE,
         "        return GGML_SYCL_RESIDENCY_PROBE_FOREIGN_BACKEND;",
         "        return (ggml_sycl_residency_probe_status) GGML_SYCL_LIFECYCLE_OK;"),
        ("the probe's struct gate is a bare null test", "does not gate the caller's result struct", SIG_PROBE,
         "    if (!ggml_sycl::residency_probe_out_declared(out)) {", "    if (out == nullptr) {"),
        ("the probe's struct gate is an inline size test", "does not gate the caller's result struct", SIG_PROBE,
         "    if (!ggml_sycl::residency_probe_out_declared(out)) {",
         "    if (out == nullptr || out->struct_size < sizeof(*out)) {"),
        ("the probe writes the layer count ahead of its struct gate", "does not gate the caller's result struct", SIG_PROBE,
         "    (void) model;\n    if (!ggml_sycl::residency_probe_out_declared(out)) {",
         "    (void) model;\n    out->n_layer = 0;\n    if (!ggml_sycl::residency_probe_out_declared(out)) {"),
        ("the probe's struct gate no longer refuses", "does not gate the caller's result struct", SIG_PROBE,
         "        return GGML_SYCL_RESIDENCY_PROBE_INVALID;\n    }\n    out->n_layer = 0;",
         "        (void) 0;\n    }\n    out->n_layer = 0;"),
        ("the probe's cap check refuses an equal buffer", "does not refuse a buffer smaller than the answer",
         SIG_PROBE,
         "    if (out->n_layer_cap < out->n_layer) {", "    if (out->n_layer_cap <= out->n_layer) {"),
        ("the probe's cap check is removed", "does not refuse a buffer smaller than the answer", SIG_PROBE,
         "    if (out->n_layer_cap < out->n_layer) {", "    if (false) {"),
        ("the probe's zero-shape check is removed", "does not refuse a zero shape", SIG_PROBE,
         "    if (n_ctx == 0 || n_ubatch == 0 || n_seq_max == 0) {", "    if (false) {"),
        ("the probe skips the descriptor parse", "does not refuse a malformed descriptor", SIG_PROBE,
         "        if (desc != nullptr) {\n            ggml_sycl::runtime_context_geometry geometry;",
         "        if (desc != nullptr && false) {\n            ggml_sycl::runtime_context_geometry geometry;"),
        ("the probe's foreign-backend check is removed", "does not refuse a foreign backend", SIG_PROBE,
         "        ggml_backend_dev_backend_reg(backend->device) != ggml_backend_sycl_reg()) {\n"
         "        GGML_LOG_WARN(\"[RESIDENCY-PROBE]",
         "        false) {\n        GGML_LOG_WARN(\"[RESIDENCY-PROBE]"),
        # the arms must run in order: each of these moves one arm above one that has to run first
        ("the probe checks the shape before the backend", "before an earlier arm has run", SIG_PROBE,
         re.compile(r"(    if \(!backend \|\| !backend->context[^{]*\{\n[^}]*FOREIGN_BACKEND;\n    \}\n)"
                    r"(    if \(n_ctx == 0 \|\|[^{]*\{\n[^}]*\n    \}\n)"), r"\2\1"),
        ("the probe checks the cap before the descriptor parse", "before an earlier arm has run", SIG_PROBE,
         re.compile(r"(    try \{\n        if \(desc != nullptr\) \{[\s\S]*?\n    \}\n)"
                    r"(?=    if \(out->n_layer_cap < out->n_layer\) \{\n)"
                    r"(    if \(out->n_layer_cap < out->n_layer\) \{\n[^}]*\n    \}\n)"), r"\2\1"),
        ("the probe no longer writes the layer count", "does not write the layer count", SIG_PROBE,
         "            out->n_layer = (uint32_t) parsed.kv.layers.size();\n", ""),
        ("the probe's not-wired answer says nothing", "does not say why it did not answer",  SIG_PROBE,
         re.compile(r'    GGML_LOG_WARN\(\n\s*"\[RESIDENCY-PROBE\] no answer: the zone geometry[^"]*"(?:[^;"]|"[^"]*")*;\n'), ""),
        ("the probe's not-wired answer is not the last statement", "does not end by answering GEOMETRY_NOT_WIRED", SIG_PROBE,
         "    return GGML_SYCL_RESIDENCY_PROBE_GEOMETRY_NOT_WIRED;\n}",
         "    return GGML_SYCL_RESIDENCY_PROBE_GEOMETRY_NOT_WIRED;\n    return GGML_SYCL_RESIDENCY_PROBE_INVALID;\n}"),
        ("the probe names the OK enumerator in a dead branch", "names OK or a lifecycle result", SIG_PROBE,
         "    (void) model;\n", "    (void) model;\n    if (false) { (void) GGML_SYCL_RESIDENCY_PROBE_OK; }\n"),
    ])
    pairs.append(("the llama table resolves the probe by a string literal", "does not resolve the probe through the reg by its macro",
                  "llama_context_sycl_proc_addr(dev, GGML_SYCL_PROC_PROBE_RESIDENCY)",
                  "llama_context_sycl_proc_addr(dev, \"ggml_backend_sycl_probe_residency\")"))
    pairs.append(("the door calls the proc twice", "the proc pointer is called 2 times",
                  "    const ggml_sycl_residency_probe_status r = procs.probe_residency(",
                  "    (void) procs.probe_residency(nullptr, ggml_sycl_model_token{}, 0, 0, 0, false, false, false, nullptr, nullptr);\n"
                  "    const ggml_sycl_residency_probe_status r = procs.probe_residency("))
    pairs.append(("a second door in the context's own file", "the proc pointer is called 2 times",
                  "    procs.probe_residency = reinterpret_cast<decltype(procs.probe_residency)>(",
                  "    (void) procs.probe_residency(nullptr, ggml_sycl_model_token{}, 0, 0, 0, false, false, false, nullptr, nullptr);\n"
                  "    procs.probe_residency = reinterpret_cast<decltype(procs.probe_residency)>("))
    for label, msg, sig, old, new in scoped:
        span = function_span(src, sig)
        if isinstance(old, str):
            found = span is not None and src[span[0]:span[1]].count(old) == 1
        else:  # a compiled pattern: the first match inside the function (its first statement, say)
            found = span is not None and old.search(src[span[0]:span[1]]) is not None
        if not found:
            muts.append(("PATTERN NOT FOUND: " + label, "PATTERN", header_raw, src))
        else:
            a, b = span
            body = src[a:b].replace(old, new, 1) if isinstance(old, str) else old.sub(new, src[a:b], count=1)
            muts.append((label, msg, header_raw, src[:a] + body + src[b:]))
    pairs.append(("the scheduler's allocator is built with one buffer", "allocator is no longer built with one buffer per backend",
                  "sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);",
                  "sched->galloc = ggml_gallocr_new_n(sched->bufts, 1);"))
    pairs.append(("a normal context does not append the CPU backend", "no longer appends the CPU backend",
                  "        backends.emplace_back(backend_cpu);\n", ""))
    pairs.append(("a measure-only context no longer requires the CPU backend last", "no longer requires the CPU backend last",
                  "\"the last backend of a measure-only context must be the CPU backend\"", "\"x\""))
    pairs.append(("the transaction tail through the swallowing erase", "does not drop the context's earlier section",
                  "    ggml_sycl_published_section_set(ctx, nullptr);\n    return ggml_sycl_txn_result::ACCEPTED;",
                  "    ggml_sycl_published_section_erase(ctx->device, 1);\n    return ggml_sycl_txn_result::ACCEPTED;"))
    pairs.append(("the rung release's synchronize conditional", "release_rung_buffers does not synchronize()",
                  "auto release_rung_buffers = [&]() {\n        synchronize();\n", "auto release_rung_buffers = [&]() {\n        if (false) synchronize();\n"))
    pairs.append(("the rung release's synchronize dropped", "release_rung_buffers does not synchronize()",
                  "auto release_rung_buffers = [&]() {\n        synchronize();\n", "auto release_rung_buffers = [&]() {\n"))
    pairs.append(("the backend interface wires cpy_tensor_async", "cpy_tensor_async is referenced beyond its definition",
                  CPY_ASYNC_DEF, "static void h_wire_cpy() { (void) ggml_backend_sycl_cpy_tensor_async; }\n" + CPY_ASYNC_DEF))
    # llama.cpp-p6i0: the load's compute-term record
    pairs.append(("the writer marked unused again", "still marked [[maybe_unused]]",
                  "static bool ggml_sycl_load_record_compute_term(",
                  "[[maybe_unused]] static bool ggml_sycl_load_record_compute_term("))
    pairs.append(("a third caller of the ledger writer", "called outside the record export",
                  "static size_t ggml_sycl_load_clear_compute_terms(uint64_t txn) noexcept {",
                  "static void h_third_writer_caller() { (void) ggml_sycl_load_record_compute_term(1, 0, 1, 1); }\n"
                  "static size_t ggml_sycl_load_clear_compute_terms(uint64_t txn) noexcept {"))
    for label, msg, sig, old, new in (
            ("the record export skips the module guard", "does not reach the writer under the module guard",
             SIG_RECORD_EXPORT, "if (!module_guard)", "if (false)"),
            ("the record export writes the ledger itself", "does not reach the writer under the module guard",
             SIG_RECORD_EXPORT, "return ggml_sycl_load_record_compute_term(txn.id, device, bytes, n_ctx);",
             "return ggml_sycl_load_ledger().ledger.size() == 0;")):
        span = function_span(src, sig)
        if span is None or src[span[0]:span[1]].count(old) != 1:
            muts.append(("PATTERN NOT FOUND: " + label, "PATTERN", header_raw, src))
        else:
            a, b = span
            muts.append((label, msg, header_raw, src[:a] + src[a:b].replace(old, new, 1) + src[b:]))
    for name in ("ggml_backend_sycl_load_record_compute_term",):
        line = '// Proc name: "%s".' % name
        if header_raw.count(line) != 1:
            muts.append(("PATTERN NOT FOUND: the header's proc name " + name, "PATTERN", header_raw, src))
        else:
            muts.append(("the header forgets " + name, "the header does not name the proc " + name,
                         header_raw.replace(line, "", 1), src))
    for label, msg, old, new in pairs:
        if label == "the first publish reserves under L1":
            lock = ("        std::lock_guard<std::mutex> lock(g_tensor_inventory_mutex);\n"
                    "        const auto snapshot = ggml_sycl::lifecycle_select_placement_plan(model.model_id")
            if src.count(old) != 1 or src.count(lock) != 1:
                muts.append(("PATTERN NOT FOUND: " + label, "PATTERN", header_raw, src))
            else:
                moved = src.replace(old, new, 1).replace(
                    lock, "        std::lock_guard<std::mutex> lock(g_tensor_inventory_mutex);\n"
                          "        (void) ggml_sycl_reserve_host_tenants(backend_ctx, *section, host_tenants, refusal_under_l1);\n"
                          "        const auto snapshot = ggml_sycl::lifecycle_select_placement_plan(model.model_id", 1)
                muts.append((label, msg, header_raw, moved))
            continue
        if label == "the clear guard before the finisher check":
            moved = src.replace(old, "", 1).replace("        ticket = registry->prepare_end(", "        ggml_sycl_load_ledger_clear_guard ledger_clear{ txn.id };\n        ticket = registry->prepare_end(", 1)
            muts.append((label, msg, header_raw, moved))
            continue
        if src.count(old) != 1:
            muts.append(("PATTERN NOT FOUND: " + label, "PATTERN", header_raw, src))
        else:
            muts.append((label, msg, header_raw, src.replace(old, new, 1)))
    return muts


def door_mutations(texts):
    """Each entry: (label, expected message fragment, texts).  The door's files are separate from the pinned source text,
    so its mutants edit these."""
    muts = []
    victim = next(iter(texts))
    for label, msg, tail in (
            ("a file under src/ calls the probe pointer", "calls the probe proc pointer",
             "\nstatic void h_door1(llama_sycl_l4_procs & procs) { procs.probe_residency(nullptr); }\n"),
            ("a file under src/ calls it through a pointer", "calls the probe proc pointer",
             "\nstatic void h_door2(llama_sycl_l4_procs * p) { p->probe_residency(nullptr); }\n"),
            ("a file under src/ calls it through a parenthesised pointer", "calls the probe proc pointer",
             "\nstatic void h_door3(llama_sycl_l4_procs & procs) { (procs.probe_residency)(nullptr); }\n"),
            ("a file under src/ names the symbol", "names ggml_backend_sycl_probe_residency",
             "\nstatic void * h_door4 = (void *) ggml_backend_sycl_probe_residency;\n"),
            ("a file under src/ calls the symbol", "names ggml_backend_sycl_probe_residency",
             "\nstatic void h_door5() { ggml_backend_sycl_probe_residency(nullptr); }\n")):
        mutated = dict(texts)
        mutated[victim] = texts[victim] + tail
        muts.append((label, msg, mutated))
    muts.append(("the scan reads nothing", "the scan is void", {}))
    # the table field's type names the symbol and is allowed
    allowed = dict(texts)
    allowed[victim] = texts[victim] + "\nstatic decltype(&ggml_backend_sycl_probe_residency) h_ok = nullptr;\n"
    return muts, allowed


def normalize(text):
    """Collapse the runs of spaces inside each line (clang-format aligns declarations and assignments into
    columns, so the width of a gap is not part of what the source says); keep each line's indentation."""
    out = []
    for line in text.split("\n"):
        m = re.match(r"[ \t]*", line)
        out.append(m.group(0) + re.sub(r"[ \t]+", " ", line[m.end():]))
    return "\n".join(out)


def read(root, rel):
    with open(os.path.join(root, rel), encoding="utf-8") as f:
        text = f.read()
    return normalize(text) if rel in SOURCE_FILES else text


def load_source(root):
    """The comment-stripped, normalized text every pin and mutation reads: the files of SOURCE_FILES as one text."""
    return "\n".join(strip_comments(read(root, rel)) for rel in SOURCE_FILES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--no-mutations", action="store_true")
    ap.add_argument("--mutations-only", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="name each mutant and the failure that caught it")
    args = ap.parse_args()
    header_raw = read(args.root, HEADER)
    # once: a mutation edits this text
    source_raw = load_source(args.root)
    status = 0
    if not args.mutations_only:
        fails = check(header_raw, source_raw) + door_check(door_texts(args.root))
        for f in fails:
            print("FAIL: " + f)
        if fails:
            status = 1
        else:
            print("PASS: the L4 procs are registered and their state keeps its single writer")
    if not args.no_mutations:
        base = check(header_raw, source_raw)
        if args.mutations_only and base:
            for f in base:
                print("FAIL (control): " + f)
            return 1
        for label, msg, h, s in mutations(header_raw, source_raw):
            if msg == "PATTERN":
                print("FAIL: mutation harness: " + label)
                status = 1
                continue
            got = check(h, s)
            if not any(msg in g for g in got):
                print("FAIL: mutation survived: %s (wanted a failure containing %r, got %s)" % (label, msg, got[:2]))
                status = 1
            elif args.verbose:
                print("DIED: %s -> %s" % (label, next(g for g in got if msg in g)))
        door_muts, door_allowed = door_mutations(door_texts(args.root))
        if door_check(door_allowed):
            print("FAIL: the table field's decltype is refused by the door scan: %s" % door_check(door_allowed)[:1])
            status = 1
        for label, msg, texts in door_muts:
            got = door_check(texts)
            if not any(msg in g for g in got):
                print("FAIL: mutation survived: %s (wanted a failure containing %r, got %s)" % (label, msg, got[:2]))
                status = 1
            elif args.verbose:
                print("DIED: %s -> %s" % (label, next(g for g in got if msg in g)))
        if status == 0:
            print("PASS: every mutation is caught")
    return status


if __name__ == "__main__":
    sys.exit(main())
