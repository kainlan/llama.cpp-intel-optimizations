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
  * the late check reads the open transaction from the lifecycle registry, and
    late_term_shrink_admitted has that one producer;

Usage:
  check-sycl-l4-proc-registration.py [--root DIR]     check the tree, then run the mutation matrix
  check-sycl-l4-proc-registration.py --no-mutations   check the tree only
  check-sycl-l4-proc-registration.py --mutations-only run the mutation matrix only

The mutation matrix applies each RED to the source text in memory and requires the gate to fail with
that RED's message, so a check that can no longer fail is caught.
"""

import argparse
import os
import re
import sys

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


def check(header_raw, source_raw):
    fails = []
    source = strip_comments(source_raw)
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
    allowed = {"ggml_sycl_load_record_compute_term", "ggml_sycl_load_clear_compute_terms",
               "ggml_backend_sycl_load_late_check", "ggml_backend_sycl_test_compute_term_count"}
    users = set()
    for m in re.finditer(r"\bggml_sycl_load_ledger\s*\(\s*\)", source):
        if re.match(r"\s*\{", source[m.end():]):
            continue  # the accessor's own definition
        head = source[:m.start()]
        fn = [x for x in re.findall(r'\n(?:extern\s+"C"\s+)?(?:[A-Za-z_][\w:<>\*&, ]*\s+)?\*?&?\s*([A-Za-z_]\w*)\s*\([^;{}]*\)\s*(?:noexcept\s*)?\{', head)
              if x not in ("if", "for", "while", "switch", "catch")]
        users.add(fn[-1] if fn else "?")
    extra = users - allowed
    if extra:
        fails.append("L4 ledger: the ledger accessor is named outside the four sanctioned functions: %s" % sorted(extra))
    for a in sorted(allowed):
        if a not in users and a != "ggml_backend_sycl_test_compute_term_count":  # the hook is a test build's
            fails.append("L4 ledger: %s no longer reaches the ledger through its accessor" % a)

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

    # transaction tail
    txn = function_body(source, r"\bggml_sycl_txn_result\s+ggml_sycl_run_runtime_context_transaction\s*\(")
    if txn is None:
        fails.append("L4 section: ggml_sycl_run_runtime_context_transaction not found")
    else:
        a = txn.find("ctx->runtime_kv_admitted = true;")
        b = txn.find("ggml_sycl_published_section_erase(ctx);")
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
        if "admission_diagnostics().active_txn" not in late:
            fails.append("L4 late: the late check does not read the open load transaction from the lifecycle registry")
        if "late_term_shrink_admitted" not in late or "r.shrink_counted" not in late:
            fails.append("L4 late: the late check does not count an admitted shrink")
    if len(re.findall(r"dump_counter::late_term_shrink_admitted", source)) != 1:
        fails.append("L4 late: late_term_shrink_admitted has more or fewer than one producer")
    return fails


def mutations(header_raw, source_raw):
    """Each entry: (label, expected message fragment, header text, source text)."""
    muts = []
    src = source_raw
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
         "    ggml_sycl_published_section_erase(this);\n    ggml_sycl_execution_unbind_backend(this);",
         "    ggml_sycl_execution_unbind_backend(this);\n    ggml_sycl_published_section_erase(this);"),
        ("the transaction tail's drop removed", "does not drop the context's earlier section",
         "    ggml_sycl_published_section_erase(ctx);\n    return ggml_sycl_txn_result::ACCEPTED;",
         "    return ggml_sycl_txn_result::ACCEPTED;"),
        ("the section stored on any outcome", "without a successful inner transaction",
         "    if (inner_ok && section) {", "    if (section) {"),
        ("the late check without the open read", "open load transaction from the lifecycle registry",
         "admission_diagnostics().active_txn == txn.id", "true"),
        ("the shrink counter dropped", "does not count an admitted shrink",
         "        if (r.shrink_counted) {\n            ggml_sycl::unified_cache_dump_counter_add(ggml_sycl::dump_counter::late_term_shrink_admitted, device);\n        }\n", ""),
    ]
    for label, msg, old, new in pairs:
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
    args = ap.parse_args()
    header_raw = read(args.root, HEADER)
    source_raw = read(args.root, SOURCE)
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
        if status == 0:
            print("PASS: every mutation is caught")
    return status


if __name__ == "__main__":
    sys.exit(main())
