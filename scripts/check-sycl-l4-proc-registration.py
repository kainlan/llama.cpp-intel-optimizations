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
  * the load's end call clears the ledger through a guard created after the finisher check, so
    commit, rollback and exception all clear;
  * the backend context's destructor erases its published section before it resets its execution
    binding (which zeroes the registry key);
  * the runtime-context transaction drops the published section on its success tail, and the
    descriptor publish stores a section only after the inner transaction succeeded;
  * the late check and the record read the open transaction (ggml_sycl_load_txn_is_open) only after the
    ledger's mutex is taken, and every function that names the ledger takes that mutex before its first
    use of it; late_term_shrink_admitted has one producer;
  * the fail-closed values: the late check answers NOT_RECORDED, the coverage query GROWTH and the clear 0
    on a closed module or an exception, a coverage query of an unbound context answers GROWTH, and a
    refusal logs at ERROR, a shrink and a not-open transaction at WARN (through ggml_sycl_load_ledger_log);
  * a publish for an unbound context is said at WARN, and the destructor's erase says a failed drop at
    WARN; the publish tail drops through the throwing set, not the erase;
  * the recovery path that ends a load without the clear guard clears the load's terms itself;
  * (step 3c) a context's first publish reserves its host tier before L1 through
    unified_allocate_owner with the carve's request fields, installs the table only after the inner
    transaction succeeded, and the destructor drops it after the section and before the unbind; the
    SYCL_Host buffer type claims inside a claim scope before it reaches any allocation, and its free
    releases the claim.

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
REG_FN = "ggml_backend_sycl_reg_get_proc_address"


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


def header_proc_names(header_raw):
    return re.findall(r'Proc name:\s*"([A-Za-z0-9_]+)"', header_raw)


def impl_first_publish_wrong(source):
    impl = function_body(source, r"\bggml_sycl_set_runtime_context_for_model_impl\s*\(")
    if impl is None:
        return True
    r = impl.find("ggml_sycl_reserve_host_tenants(")
    l1 = impl.find("std::lock_guard<std::mutex> lock(g_tensor_inventory_mutex)")
    inner = impl.find("const bool inner_ok = g_runtime_update_succeeded;")
    inst = impl.find("ggml_sycl_host_tenants_install(")
    return not (0 <= r < l1 and 0 <= inner < inst)


def function_span(text, signature_re):
    """(start, end) of the first definition whose header matches signature_re, or None."""
    body = function_body(text, signature_re)
    if body is None:
        return None
    start = text.find(body)
    return (start, start + len(body))


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
        use = body.find(".ledger.")
        lk = re.search(lock_re, body)
        if use < 0 or lk is None or lk.start() > use:
            fails.append("L4 ledger lock: %s uses the ledger without state.mutex held first" % name)
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
        if not re.search(r"case ggml_sycl::load_log_level::" + lvl + r":\s*" + macro + r"\(", log_fn):
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
        rc = end.find("(void) ggml_sycl_load_clear_compute_terms(txn.id);")
        fa = end.find("ggml_sycl_finalize_binding_failure_abort(*registry, recovery);")
        if fa < 0 or rc < 0 or rc > fa:
            fails.append("L4 ledger: the recovery path that ends a load without the clear guard does not clear its terms")

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
        fails.append("L4 host tier: the first publish does not reserve before L1, or installs before the inner transaction succeeded")

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

    # transaction tail
    txn = function_body(source, r"\bggml_sycl_txn_result\s+ggml_sycl_run_runtime_context_transaction\s*\(")
    if txn is None:
        fails.append("L4 section: ggml_sycl_run_runtime_context_transaction not found")
    else:
        a = txn.find("ctx->runtime_kv_admitted = true;")
        b = txn.find("ggml_sycl_published_section_set(ctx, nullptr);")
        if a < 0 or b < 0 or b < a:
            fails.append("L4 section: a successful publish does not drop the context's earlier section")

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
        ("the table installed on any outcome", "installs before the inner transaction",
         "            if (host_tenants) {\n                (void) ggml_sycl_host_tenants_install(backend_ctx, host_tenants, section->tenant_key);\n            }\n", ""),
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
         "case ggml_sycl::load_log_level::ERROR:\n            GGML_LOG_ERROR(", "case ggml_sycl::load_log_level::ERROR:\n            GGML_LOG_INFO("),
        ("the not-open line logged at INFO", "a ledger line of level WARN is not logged through GGML_LOG_WARN",
         r"\bstatic\s+void\s+ggml_sycl_load_ledger_log\s*\(",
         "case ggml_sycl::load_log_level::WARN:\n            GGML_LOG_WARN(", "case ggml_sycl::load_log_level::WARN:\n            GGML_LOG_INFO("),
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
        ("the recovery path no longer clears", "does not clear its terms",
         r"\bggml_sycl_lifecycle_result\s+ggml_backend_sycl_model_load_end\s*\(",
         "            (void) ggml_sycl_load_clear_compute_terms(txn.id);\n", ""),
    ]
    for label, msg, sig, old, new in scoped:
        span = function_span(src, sig)
        if span is None or src[span[0]:span[1]].count(old) != 1:
            muts.append(("PATTERN NOT FOUND: " + label, "PATTERN", header_raw, src))
        else:
            a, b = span
            muts.append((label, msg, header_raw, src[:a] + src[a:b].replace(old, new, 1) + src[b:]))
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
    return normalize(text) if rel == SOURCE else text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--no-mutations", action="store_true")
    ap.add_argument("--mutations-only", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="name each mutant and the failure that caught it")
    args = ap.parse_args()
    header_raw = read(args.root, HEADER)
    source_raw = strip_comments(read(args.root, SOURCE))  # once: a mutation edits this text
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
