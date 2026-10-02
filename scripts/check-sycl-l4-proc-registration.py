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
  * the backend context's destructor erases its published section before it resets its execution
    binding (which zeroes the registry key);
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
  * a publish for an unbound context is said at WARN, and the destructor's erase says a failed drop at
    WARN; the publish tail drops through the throwing set, not the erase;
  * (step 3c) a context's first publish reserves its host tier before L1 through
    unified_allocate_owner with the carve's request fields, installs the table once before the inner
    transaction (a rollback guard takes it back unless the section was stored), and the destructor drops
    it after the section and before the unbind; an ended execution context drops its entries first; the
    SYCL_Host buffer type claims inside a claim scope before it reaches any allocation, and its free
    releases the claim without waiting on the host.  The pins also name each refusal the device test
    exercises: a refused or part-way reservation and a republish the held slots cannot carry are
    PLAN_REJECTED before anything is published, the table is installed (and refused by name) before anything
    is published and the section is stored last, a refused claim never falls through to the allocator, the claim in flight is owned
    by an RAII record, a table handle is cast only after its deleter is checked, and the claim scope's
    open answers a status for each way it can not open.

  * (the free-path contract) the SYCL_Host free_buffer releases the slot once, through
    tenant_claim_scope::release with event 0, and never waits on the host; the synchronize that makes
    that safe is pinned where a compute buffer can be freed (~llama_context first, sched_reserve_impl
    before it touches the scheduler, release_rung_buffers), the backend's synchronize still drains stream
    0, the deferred decode event and the CPU-expert flush, set/get_tensor_async accept only the device,
    host-compute and cpu-offload buffer types, the device does not support the CpuActivation clone, and
    cpy_tensor_async stays unwired.  llama-context.cpp is read as one text with ggml-sycl.cpp.

Known limits (text-level pins; each is what the named test or review covers instead):
  * a lambda or macro that hides a release, a synchronize or a clear behind another name: the pins read
    the shapes named above, not the call graph (covered by the host-tenant-claim device test and review);
  * a tail-drop or late check inside a function the pin does not name (a statement added after the last
    pinned one): the pins anchor the first and the terminating statements, not the ones between (covered by
    the lifecycle device test's order witnesses);
  * the gate cannot tell a synchronize that is reached from one that is dead code behind a runtime
    condition other than a literal `if (false)` (covered by the L6 wait_event consumption acceptance).

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

HEADER = "ggml/include/ggml-sycl.h"
SOURCE = "ggml/src/ggml-sycl/ggml-sycl.cpp"
CLAIM_HPP = "ggml/src/ggml-sycl/tenant-claim-scope.hpp"  # the claim record's destructor and the lowest-free walk
LLAMA = "src/llama-context.cpp"  # the scheduler's owner: read with SOURCE, as one text, for the free-path contract
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
    "                    host_tenants_installed.ctx = backend_ctx;\n")
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
DROP_ENTRIES_SIG = r"\bstatic\s+void\s+ggml_sycl_execution_drop_context_registry_entries\s*\(uint64_t context_id\)\s*noexcept\s*\{"
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


def function_body(text, signature_re):
    """The brace-balanced body of the first definition whose header matches signature_re, or None."""
    m = re.search(signature_re + r"[^;{]*\{", text)
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
        r"if \(host_tenants\) \{\s*if \(!ggml_sycl_host_tenants_install\(backend_ctx, host_tenants, section->tenant_key\)\) \{"
        r"\s*GGML_LOG_ERROR\([\s\S]*?\);\s*return GGML_SYCL_LIFECYCLE_PLAN_REJECTED;\s*\}\s*"
        r"host_tenants_installed\.ctx = backend_ctx;\s*\}",
        "L4 host tier: the install result is not a refusal with nothing published, or the table it installed does not arm "
        "the rollback guard")
    pin(fails, impl, r"ggml_sycl_published_section_set\(backend_ctx, section\);\s*host_tenants_installed\.keep\(\);",
        "L4 host tier: the section is stored without keeping the installed table (or the table is kept before the store)")
    pin(fails, impl, r"std::shared_ptr<ggml_sycl::kv_tenant_slots> host_tenants;\s*ggml_sycl_host_tenants_install_guard host_tenants_installed;",
        "L4 host tier: the publish holds no rollback guard for the table it installs")
    guard_fn = function_body(source, r"\bstruct\s+ggml_sycl_host_tenants_install_guard\b")
    pin(fails, guard_fn, r"~ggml_sycl_host_tenants_install_guard\(\) \{\s*if \(ctx\) \{\s*ggml_sycl_host_tenants_erase\(ctx\);\s*\}\s*\}"
                         r"\s*void keep\(\) \{ ctx = nullptr; \}",
        "L4 host tier: the install guard does not take the table back unless it was kept")
    install_fn = function_body(source, r"\bstatic\s+bool\s+ggml_sycl_host_tenants_install\s*\(")
    if install_fn is not None and install_fn.count("table.reset()") != 1:
        fails.append("L4 host tier: the table is dropped from the caller before the registry took it")
    pin(fails, install_fn,
        r"if \(id == 0 \|\| !ggml_sycl_kv_region_registry\(ctx->device\)\.install_tenant_slots\(id, table, tenant_key\)\) \{"
        r"\s*return false;\s*\}\s*table\.reset\(\);\s*return true;",
        "L4 host tier: the table is dropped from the caller before the registry took it")
    pin(fails, function_body(source, r"\bstatic\s+bool\s+ggml_sycl_host_tenants_carry\s*\("),
        r"held_bytes == 0\) \{[^}]*return false;\s*\}\s*if \(\(size_t\) e\.slot_bytes > held_bytes\) \{[^}]*return false;",
        "L4 host tier: a republish is carried by a missing or smaller held slot")
    if not re.search(r"\bstatic\s+void\s+ggml_sycl_host_tenants_erase\s*\(const ggml_backend_sycl_context \* ctx\) noexcept \{", source):
        fails.append("L4 host tier: the destructor's drop of the host reservation may throw (not noexcept)")
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
      * sched_reserve_impl synchronizes before it first touches the scheduler, and so does release_rung_buffers
      * ggml_backend_sycl_synchronize drains stream 0 (or the deferred decode event) and the CPU-expert flush
      * nothing but the SYCL device, host-compute and cpu-offload buffer types reaches set/get_tensor_async, the
        CpuActivation clone is not a buffer type the device supports, and cpy_tensor_async is not wired
    """
    if freeb is not None:
        if len(re.findall(r"\brelease\(", freeb)) != 1 or re.search(r"unified_free|zone_free|sycl::free|\bfree\(", freeb):
            fails.append("L4 free path: the SYCL_Host free_buffer releases other than once through tenant_claim_scope::release")
    dtor = function_body(source, DTOR_SIG)
    pin(fails, dtor, r"^[^{]*\{\s*synchronize\(\);",
        "L4 free path: ~llama_context does not synchronize() first, before its scheduler's compute buffers are freed")
    reimpl = function_body(source, REIMPL_SIG)
    if reimpl is None:
        fails.append("L4 free path: llama_context::sched_reserve_impl not found")
    else:
        sy = reimpl.find("synchronize();")
        first = min([i for i in (reimpl.find("sched.reset("), reimpl.find("ggml_backend_sched_")) if i >= 0] or [-1])
        if sy < 0 or first < 0 or sy > first:
            fails.append("L4 free path: sched_reserve_impl does not synchronize() before it first touches the scheduler")
    pin(fails, source, r"auto release_rung_buffers = \[&\]\(\) \{\s*synchronize\(\);\s*for \(auto & res : gf_res_prev\)",
        "L4 free path: release_rung_buffers does not synchronize() before it frees the rung's buffers")
    sync = function_body(source, SYNC_SIG)
    pin(fails, sync, r"CHECK_TRY_ERROR\(stream->wait_and_throw\(\)\)",
        "L4 free path: the backend synchronize no longer drains stream 0")
    pin(fails, sync, r"ggml_sycl_cpu_tg_flush_pending\(\);",
        "L4 free path: the backend synchronize no longer flushes the CPU-expert work")
    pin(fails, sync, r"sycl_ctx->last_graph_event->wait_and_throw\(\)",
        "L4 free path: the backend synchronize no longer drains the deferred decode event")
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


def check(header_raw, source):
    """`source` is the comment-stripped, normalized text: a mutation is one edit of it, so the (slow) strip is
    done once, not once per mutant."""
    fails = []
    names = header_proc_names(header_raw)
    if len(names) < 3:
        fails.append("L4 proc: fewer than 3 `Proc name:` entries in %s (the scan found %d): the gate is void" % (HEADER, len(names)))
    for required in ("ggml_backend_sycl_set_runtime_context_desc", "ggml_backend_sycl_tenant_coverage",
                     "ggml_backend_sycl_load_late_check"):
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
    herase_fn = function_body(source, r"\bstatic\s+void\s+ggml_sycl_host_tenants_erase\s*\(") or ""
    if not re.search(r"catch\s*\(\.\.\.\)\s*\{\s*GGML_LOG_WARN\(", herase_fn):
        fails.append("L4 host tier: the destructor's drop of the host reservation swallows a failure silently (no WARN)")
    erase_fn = function_body(source, r"\bstatic\s+void\s+ggml_sycl_published_section_erase\s*\(") or ""
    if not re.search(r"catch\s*\(\.\.\.\)\s*\{\s*GGML_LOG_WARN\(", erase_fn):
        fails.append("L4 section: the destructor's erase swallows a failed drop silently (no WARN)")

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

    # the end of an execution context drops the entries its id keyed, before the id is zeroed: the backend's
    # destructor cannot find them afterwards (finish_drain and close_if_idle reset the id first)
    clear_bind = function_body(source, CLEAR_BIND_SIG)
    if clear_bind is None:
        fails.append("L4 context end: ggml_sycl_execution_clear_bindings_for_context not found")
    else:
        call = clear_bind.find("ggml_sycl_execution_drop_context_registry_entries(context_id);")
        reset = clear_bind.find("ggml_sycl_execution_reset_backend_binding_state(")
        locked = clear_bind.find("g_execution_backend_binding_mutex")
        if call < 0 or reset < 0 or call > reset or (locked >= 0 and call > locked) or net_depth(clear_bind[:call]) != 1 or \
                clear_bind[:call].rstrip()[-1:] not in (";", "{"):
            fails.append("L4 context end: the clear of an ended context's bindings does not first drop its registry entries "
                         "(the backend's id is zeroed before the destructor can drop them)")
    drop_entries = function_body(source, DROP_ENTRIES_SIG)
    pin(fails, drop_entries,
        r"ggml_sycl_execution_for_each_bound_backend\(\s*context_id, \[\]\(ggml_backend_sycl_context \* backend, "
        r"const ggml_sycl_execution_backend_binding &\) \{\s*ggml_sycl_published_section_erase\(backend\);\s*"
        r"ggml_sycl_host_tenants_erase\(backend\);\s*\}\);",
        "L4 context end: the drop of an ended context's registry entries does not erase the section then the host "
        "reservation of each bound backend")
    if drop_entries is not None and "g_execution_backend_binding_mutex" in drop_entries:
        fails.append("L4 context end: the registry entries of an ended context are dropped under the binding mutex")

    # destructor order
    dtor = function_body(source, r"ggml_backend_sycl_context::~ggml_backend_sycl_context\s*\(")
    if dtor is None:
        fails.append("L4 section: ~ggml_backend_sycl_context not found")
    else:
        e = dtor.find("ggml_sycl_published_section_erase(this)")
        u = dtor.find("ggml_sycl_execution_unbind_backend(this)")
        if e < 0:
            fails.append("L4 section: the backend context's destructor does not erase its published section")
        elif u < 0 or e > u:
            fails.append("L4 section: the section is erased after the execution binding is reset (its key is gone)")

    if dtor is not None:
        h = dtor.find("ggml_sycl_host_tenants_erase(this)")
        e = dtor.find("ggml_sycl_published_section_erase(this)")
        u = dtor.find("ggml_sycl_execution_unbind_backend(this)")
        if h < 0:
            fails.append("L4 host tier: the backend context's destructor does not drop its held host reservation")
        elif not (e >= 0 and e < h < u):
            fails.append("L4 host tier: the host reservation is dropped outside the section-erase to unbind window")

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
        ("the destructor's erase dropped", "does not erase its published section",
         "    ggml_sycl_published_section_erase(this);\n", ""),
        ("the destructor's erase after the unbind", "erased after the execution binding",
         "    ggml_sycl_published_section_erase(this);\n    ggml_sycl_host_tenants_erase(this);\n    ggml_sycl_execution_unbind_backend(this);",
         "    ggml_sycl_host_tenants_erase(this);\n    ggml_sycl_execution_unbind_backend(this);\n    ggml_sycl_published_section_erase(this);"),
        ("the host reservation never dropped", "does not drop its held host reservation",
         "    ggml_sycl_host_tenants_erase(this);\n", ""),
        ("the host reservation dropped after the unbind", "outside the section-erase to unbind window",
         "    ggml_sycl_host_tenants_erase(this);\n    ggml_sycl_execution_unbind_backend(this);",
         "    ggml_sycl_execution_unbind_backend(this);\n    ggml_sycl_host_tenants_erase(this);"),
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
        ("the host reservation's drop may throw", "may throw (not noexcept)",
         "ggml_sycl_host_tenants_erase(const ggml_backend_sycl_context * ctx) noexcept {",
         "ggml_sycl_host_tenants_erase(const ggml_backend_sycl_context * ctx) {"),
        ("the free forgets its claim", "does not release its claim",
         "        (void) ggml_sycl::tenant_claim_scope::release(*ctx->claim, 0);\n        ctx->claim.reset();", "        ctx->claim.reset();"),
        ("the transaction tail's drop removed", "does not drop the context's earlier section",
         "    ggml_sycl_published_section_set(ctx, nullptr);\n    return ggml_sycl_txn_result::ACCEPTED;",
         "    return ggml_sycl_txn_result::ACCEPTED;"),
        ("the transaction tail through the swallowing erase", "does not drop the context's earlier section",
         "    ggml_sycl_published_section_set(ctx, nullptr);\n    return ggml_sycl_txn_result::ACCEPTED;",
         "    ggml_sycl_published_section_erase(ctx);\n    return ggml_sycl_txn_result::ACCEPTED;"),
        ("the section stored on any outcome", "without a successful inner transaction",
         "    if (inner_ok && section) {", "    if (section) {"),
        ("the shrink counter dropped", "does not count an admitted shrink",
         "        if (r.shrink_counted) {\n            ggml_sycl::unified_cache_dump_counter_add(ggml_sycl::dump_counter::late_term_shrink_admitted, device);\n        }\n", ""),
    ]
    # One edit inside one named function: (label, expected message, signature, old, new).  Several of the
    # lines these touch recur in the other ledger functions, so the edit is scoped to the function it names.
    scoped = [
        ("a second install after the inner transaction", "does not install once before the inner transaction",
         IMPL_SIG, "    const bool inner_ok = g_runtime_update_succeeded;",
         "    (void) ggml_sycl_host_tenants_install(backend_ctx, host_tenants, section->tenant_key);\n    const bool inner_ok = g_runtime_update_succeeded;"),
        ("the install result discarded", "the install result is not a refusal with nothing published", IMPL_SIG,
         "if (!ggml_sycl_host_tenants_install(backend_ctx, host_tenants, section->tenant_key)) {",
         "if ((ggml_sycl_host_tenants_install(backend_ctx, host_tenants, section->tenant_key), false)) {"),
        ("the install refusal answers EFFECT_FAILED", "the install result is not a refusal with nothing published", IMPL_SIG,
         INSTALL_REFUSAL_TAIL, INSTALL_REFUSAL_TAIL.replace("PLAN_REJECTED", "EFFECT_FAILED")),
        ("the installed table never arms the guard", "the install result is not a refusal with nothing published", IMPL_SIG,
         "                    host_tenants_installed.ctx = backend_ctx;\n", ""),
        ("the section stored without keeping the table", "is stored without keeping the installed table", IMPL_SIG,
         "            host_tenants_installed.keep();\n", ""),
        ("the table kept before the section store", "is stored without keeping the installed table", IMPL_SIG,
         "            ggml_sycl_published_section_set(backend_ctx, section);\n            host_tenants_installed.keep();\n",
         "            host_tenants_installed.keep();\n            ggml_sycl_published_section_set(backend_ctx, section);\n"),
        ("the publish holds no rollback guard", "holds no rollback guard", IMPL_SIG,
         "    ggml_sycl_host_tenants_install_guard host_tenants_installed;\n", ""),
        ("the install guard never takes the table back", "the install guard does not take the table back", r"\bstruct\s+ggml_sycl_host_tenants_install_guard\b",
         "            ggml_sycl_host_tenants_erase(ctx);\n", "            (void) ctx;\n"),
        ("the install guard takes the table back even when kept", "the install guard does not take the table back", r"\bstruct\s+ggml_sycl_host_tenants_install_guard\b",
         "    void keep() { ctx = nullptr; }", "    void keep() {}"),
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
        ("the host reservation's drop swallows silently", "drop of the host reservation swallows a failure silently",
         r"\bstatic\s+void\s+ggml_sycl_host_tenants_erase\s*\(", "    } catch (...) {\n        GGML_LOG_WARN(", "    } catch (...) {\n        (void) ("),
        ("the destructor's erase swallows silently", "swallows a failed drop silently",
         r"\bstatic\s+void\s+ggml_sycl_published_section_erase\s*\(", "    } catch (...) {\n        GGML_LOG_WARN(", "    } catch (...) {\n        (void) ("),
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
        ("the backend synchronize skips stream 0", "the backend synchronize no longer drains stream 0", SYNC_SIG,
         "CHECK_TRY_ERROR(stream->wait_and_throw())", "0"),
        ("the backend synchronize skips the CPU-expert flush", "no longer flushes the CPU-expert work", SYNC_SIG,
         "            ggml_sycl_cpu_tg_flush_pending();\n", ""),
        ("the backend synchronize skips the deferred decode event", "no longer drains the deferred decode event", SYNC_SIG,
         "sycl_ctx->last_graph_event->wait_and_throw()", "0"),
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
        ("the ended context's entries are never dropped", "does not first drop its registry entries", CLEAR_BIND_SIG,
         "    ggml_sycl_execution_drop_context_registry_entries(context_id);\n", ""),
        ("the ended context's entries are dropped after the id is zeroed", "does not first drop its registry entries", CLEAR_BIND_SIG,
         "    ggml_sycl_execution_drop_context_registry_entries(context_id);\n    std::lock_guard<std::mutex> lock(g_execution_backend_binding_mutex);\n",
         "    std::lock_guard<std::mutex> lock(g_execution_backend_binding_mutex);\n    ggml_sycl_execution_drop_context_registry_entries(context_id);\n"),
        ("the ended context's entries are dropped conditionally", "does not first drop its registry entries", CLEAR_BIND_SIG,
         "    ggml_sycl_execution_drop_context_registry_entries(context_id);\n",
         "    if (context_id != 0) ggml_sycl_execution_drop_context_registry_entries(context_id);\n"),
        ("the ended context's host reservation is kept", "does not erase the section then the host", DROP_ENTRIES_SIG,
         "                ggml_sycl_host_tenants_erase(backend);\n", ""),
        ("the ended context's section is kept", "does not erase the section then the host", DROP_ENTRIES_SIG,
         "                ggml_sycl_published_section_erase(backend);\n", ""),
        ("the ended context's entries are dropped under the binding mutex", "dropped under the binding mutex", DROP_ENTRIES_SIG,
         "    try {\n", "    try {\n        std::lock_guard<std::mutex> h_lock(g_execution_backend_binding_mutex);\n"),
        ("the n_ctx entry stops delegating", "set_runtime_n_ctx does not delegate to the guarded C entry",
         r"\bvoid\s+ggml_backend_sycl_set_runtime_n_ctx\s*\(", "ggml_backend_sycl_set_runtime_context(backend,",
         "(void) ggml_sycl_run_runtime_context_transaction(backend,"),
    ]
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
    pairs.append(("the rung release's synchronize dropped", "release_rung_buffers does not synchronize()",
                  "auto release_rung_buffers = [&]() {\n        synchronize();\n", "auto release_rung_buffers = [&]() {\n"))
    pairs.append(("the backend interface wires cpy_tensor_async", "cpy_tensor_async is referenced beyond its definition",
                  CPY_ASYNC_DEF, "static void h_wire_cpy() { (void) ggml_backend_sycl_cpy_tensor_async; }\n" + CPY_ASYNC_DEF))
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
    return normalize(text) if rel in (SOURCE, LLAMA, CLAIM_HPP) else text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--no-mutations", action="store_true")
    ap.add_argument("--mutations-only", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="name each mutant and the failure that caught it")
    args = ap.parse_args()
    header_raw = read(args.root, HEADER)
    # once: a mutation edits this text.  The scheduler's owner is appended: the free-path contract names both.
    source_raw = strip_comments(read(args.root, SOURCE)) + "\n" + strip_comments(read(args.root, LLAMA)) + \
        "\n" + strip_comments(read(args.root, CLAIM_HPP))
    status = 0
    if not args.mutations_only:
        fails = check(header_raw, source_raw)
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
        if status == 0:
            print("PASS: every mutation is caught")
    return status


if __name__ == "__main__":
    sys.exit(main())
