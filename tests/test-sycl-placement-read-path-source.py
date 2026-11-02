#!/usr/bin/env python3
"""Source gate for the placement plan's per-op read path (llama.cpp-5tdy).

The placement plan's answer to "does this MoE tensor have host experts?" and
"are all of this tensor's experts on this device?" is fixed once the plan is
published; it only changes on build_index() and update_expert_placement().
Before llama.cpp-5tdy both questions were answered by sweeping every expert
through the by-value lookup_expert_placement(tensor_name, e), which re-parses
the tensor name and copies a std::string plus a std::vector per expert. On
Qwen3.8 decode (512 experts) that was 8.8% of the saturated submitting thread.

Clauses (the walk is textual, over comment- and string-stripped source):
  1. The summary readers in unified-cache.hpp -- has_host_experts,
     all_experts_on_device and count_experts_on_device, every overload --
     contain no loop statement and no lookup_expert_placement call: they read
     the per-(layer, role) residency summary.
  2. In ggml_sycl_mul_mat_id (ggml-sycl.cpp), the two residency predicates
     planner_tensor_all_local_for_primary_fastpaths and
     planner_tensor_all_local_for_primary contain no loop statement and no
     lookup_expert_placement call, and each calls all_experts_on_device(.
  3. ggml_sycl_mul_mat_id contains no lookup_expert_placement call at all:
     its per-expert reads use the non-copying find_expert_entry view.
  4. The readers that may walk a strict prefix -- count_planned_experts,
     fallback_has_host_experts and count_fallback_experts_on_device -- answer
     the full range first: every loop statement in them is preceded by an
     `if` whose condition reads expert_span and whose block returns, and they
     call no lookup_expert_placement. Each must have such a shortcut.

The gate refuses to pass vacuously (every named body must be found and
non-empty) and proves itself on mutants of the real source: each mutant must
make it fail. A wrapper that hides a loop or a lookup is not seen.

argv: [checkout-root]
"""
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HPP = os.path.join("ggml", "src", "ggml-sycl", "unified-cache.hpp")
CPP = os.path.join("ggml", "src", "ggml-sycl", "ggml-sycl.cpp")

LOOP = re.compile(r"\b(for|while|do)\b")
LOOKUP = re.compile(r"\blookup_expert_placement\s*\(")
ALL_ON_DEVICE = re.compile(r"\ball_experts_on_device\s*\(")
PREDICATES = ("planner_tensor_all_local_for_primary_fastpaths", "planner_tensor_all_local_for_primary")
SUMMARY_READERS = ("has_host_experts", "all_experts_on_device", "count_experts_on_device")
PREFIX_READERS = ("count_planned_experts", "fallback_has_host_experts", "count_fallback_experts_on_device")
IF_STMT = re.compile(r"\bif\s*\(")


def scrub(text):
    """Blank comments and string/char literals, keeping every offset."""
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = " "
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                if text[k] != "\n":
                    out[k] = " "
            i = j
        elif c == '"' or c == "'":
            # A raw string literal R"(...)" never holds a quote we care about here.
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            for k in range(i + 1, min(j, n)):
                if text[k] != "\n":
                    out[k] = " "
            i = j + 1
        else:
            i += 1
    return "".join(out)


def close_of(c, open_pos, opener="{", closer="}"):
    depth = 0
    for i in range(open_pos, len(c)):
        if c[i] == opener:
            depth += 1
        elif c[i] == closer:
            depth -= 1
            if depth == 0:
                return i
    return -1


def function_bodies(c, name):
    """(open, close) of every DEFINITION of `name` (declarations are skipped)."""
    spans = []
    for m in re.finditer(r"\b" + re.escape(name) + r"\s*\(", c):
        paren = m.end() - 1
        end_paren = close_of(c, paren, "(", ")")
        if end_paren < 0:
            continue
        k = end_paren + 1
        # Qualifiers between ) and { : const, noexcept, override, try.
        tail = re.match(r"\s*(?:(?:const|noexcept|override|try)\b\s*)*", c[k:])
        k += tail.end()
        if k < len(c) and c[k] == "{":
            close = close_of(c, k)
            if close > k:
                spans.append((k, close))
    return spans


def full_range_shortcuts(c, lo, hi):
    """Offsets of each `if (... expert_span ...) { ... return ... }` in c[lo:hi]."""
    out = []
    for m in IF_STMT.finditer(c, lo, hi):
        paren = m.end() - 1
        end_paren = close_of(c, paren, "(", ")")
        if end_paren < 0 or end_paren > hi:
            continue
        if not re.search(r"\bexpert_span\b", c[paren:end_paren]):
            continue
        k = end_paren + 1
        while k < hi and c[k].isspace():
            k += 1
        if k >= hi or c[k] != "{":
            continue
        close = close_of(c, k)
        if close > k and re.search(r"\breturn\b", c[k:close]):
            out.append(m.start())
    return out


def lambda_body(c, lo, hi, name):
    m = re.compile(r"\b" + re.escape(name) + r"\s*=\s*\[[^\]]*\]\s*\([^)]*\)\s*(?:->\s*\w+\s*)?\{").search(c, lo, hi)
    if not m:
        return None
    open_pos = m.end() - 1
    close = close_of(c, open_pos)
    return (open_pos, close) if close > open_pos else None


def violations(hpp_text, cpp_text):
    bad = []
    h = scrub(hpp_text)
    c = scrub(cpp_text)

    for name in SUMMARY_READERS:
        spans = function_bodies(h, name)
        if not spans:
            bad.append("clause 1: no definition of %s found in unified-cache.hpp" % name)
        for lo, hi in spans:
            body = h[lo + 1:hi]
            if not body.strip():
                bad.append("clause 1: %s body at offset %d is empty" % (name, lo))
            if LOOP.search(body):
                bad.append("clause 1: %s has a loop statement (per-expert sweep)" % name)
            if LOOKUP.search(body):
                bad.append("clause 1: %s calls lookup_expert_placement" % name)

    for name in PREFIX_READERS:
        spans = function_bodies(h, name)
        if not spans:
            bad.append("clause 4: no definition of %s found in unified-cache.hpp" % name)
        n_shortcuts = 0
        for lo, hi in spans:
            shortcuts = full_range_shortcuts(h, lo, hi)
            n_shortcuts += len(shortcuts)
            if LOOKUP.search(h, lo, hi):
                bad.append("clause 4: %s calls lookup_expert_placement" % name)
            loop = LOOP.search(h, lo, hi)
            if loop and not any(pos < loop.start() for pos in shortcuts):
                bad.append("clause 4: %s walks before answering the full range from the counts" % name)
        if spans and n_shortcuts == 0:
            bad.append("clause 4: %s has no full-range shortcut on expert_span" % name)

    mmid = function_bodies(c, "ggml_sycl_mul_mat_id")
    if len(mmid) != 1:
        bad.append("clause 2: expected one definition of ggml_sycl_mul_mat_id, found %d" % len(mmid))
        return bad
    lo, hi = mmid[0]
    for name in PREDICATES:
        span = lambda_body(c, lo, hi, name)
        if not span:
            bad.append("clause 2: predicate %s not found in ggml_sycl_mul_mat_id" % name)
            continue
        body = c[span[0] + 1:span[1]]
        if LOOP.search(body):
            bad.append("clause 2: predicate %s has a loop statement (per-expert sweep)" % name)
        if LOOKUP.search(body):
            bad.append("clause 2: predicate %s calls lookup_expert_placement" % name)
        if not ALL_ON_DEVICE.search(body):
            bad.append("clause 2: predicate %s does not call all_experts_on_device" % name)
    for m in LOOKUP.finditer(c, lo, hi):
        line = c.count("\n", 0, m.start()) + 1
        bad.append("clause 3: ggml_sycl_mul_mat_id calls lookup_expert_placement at ggml-sycl.cpp:%d" % line)
    return bad


def insert_after_open(text, span, snippet):
    return text[:span[0] + 1] + snippet + text[span[0] + 1:]


def mutants(hpp_text, cpp_text):
    """(label, wanted-fragment, hpp, cpp); each must make the gate fail with that fragment."""
    h = scrub(hpp_text)
    c = scrub(cpp_text)
    out = []
    hhe = function_bodies(h, "has_host_experts")
    if hhe:
        span = hhe[-1]
        out.append(("loop in has_host_experts", "clause 1: has_host_experts has a loop",
                    insert_after_open(hpp_text, span, "\n        for (int64_t e = 0; e < 1; ++e) { (void) e; }\n"),
                    cpp_text))
        out.append(("while in has_host_experts", "clause 1: has_host_experts has a loop",
                    insert_after_open(hpp_text, span, "\n        while (false) {}\n"), cpp_text))
        out.append(("lookup in has_host_experts", "clause 1: has_host_experts calls lookup_expert_placement",
                    insert_after_open(hpp_text, span, "\n        (void) lookup_expert_placement(0, 0, expert_tensor_role::GATE);\n"),
                    cpp_text))
    for name in SUMMARY_READERS[1:]:
        spans = function_bodies(h, name)
        if spans:
            out.append(("loop in " + name, "clause 1: %s has a loop" % name,
                        insert_after_open(hpp_text, spans[0], "\n        for (int64_t e = 0; e < 1; ++e) { (void) e; }\n"),
                        cpp_text))
            out.append(("lookup in " + name, "clause 1: %s calls lookup_expert_placement" % name,
                        insert_after_open(hpp_text, spans[0],
                                          "\n        (void) lookup_expert_placement(0, 0, expert_tensor_role::GATE);\n"),
                        cpp_text))
    for name in PREFIX_READERS:
        spans = function_bodies(h, name)
        target = None
        for lo, hi in spans:
            shortcuts = full_range_shortcuts(h, lo, hi)
            if shortcuts and LOOP.search(h, lo, hi):
                target = (lo, hi, shortcuts[0])
                break
        if target is None:
            continue
        lo, hi, pos = target
        paren = h.index("(", pos)
        k = close_of(h, paren, "(", ")") + 1
        while h[k].isspace():
            k += 1
        block_end = close_of(h, k) + 1
        # Delete the shortcut: every query walks.
        out.append(("shortcut deleted in " + name, "clause 4: %s has no full-range shortcut" % name,
                    hpp_text[:pos] + hpp_text[block_end:], cpp_text))
        # Walk first, then answer from the counts.
        out.append(("walk before shortcut in " + name, "clause 4: %s walks before answering" % name,
                    insert_after_open(hpp_text, (lo, hi), "\n        for (int64_t w = 0; w < 1; ++w) { (void) w; }\n"),
                    cpp_text))
    mmid = function_bodies(c, "ggml_sycl_mul_mat_id")
    if len(mmid) == 1:
        lo, hi = mmid[0]
        for name in PREDICATES:
            span = lambda_body(c, lo, hi, name)
            if not span:
                continue
            out.append(("loop in " + name, "clause 2: predicate %s has a loop" % name, hpp_text,
                        insert_after_open(cpp_text, span, "\n        for (int64_t e = 0; e < 1; ++e) {}\n")))
            out.append(("lookup in " + name, "clause 2: predicate %s calls lookup_expert_placement" % name, hpp_text,
                        insert_after_open(cpp_text, span, "\n        (void) plan.lookup_expert_placement(tname, 0);\n")))
            m = ALL_ON_DEVICE.search(c, span[0], span[1])
            if m:
                renamed = cpp_text[:m.start()] + "any_experts_on_device(" + cpp_text[m.end():]
                out.append(("no all_experts_on_device in " + name,
                            "clause 2: predicate %s does not call all_experts_on_device" % name, hpp_text, renamed))
        out.append(("string lookup in mul_mat_id body", "clause 3: ggml_sycl_mul_mat_id calls lookup_expert_placement",
                    hpp_text, insert_after_open(cpp_text, (lo, hi), "\n    (void) plan_probe.lookup_expert_placement(name, 0);\n")))
    return out


def self_test():
    """The scrubber must hide comments and strings, and keep code visible."""
    sample = 'int a; // for (\n/* while */ const char * s = "do lookup_expert_placement(";\nfor (;;) {}\n'
    s = scrub(sample)
    assert len(s) == len(sample)
    assert "while" not in s and "lookup_expert_placement" not in s and "do" not in s.split("\n")[1]
    assert LOOP.search(s.split("\n")[2])
    body = function_bodies(scrub("bool f(int x) const { return x; }\nbool f(int x);\n"), "f")
    assert len(body) == 1


def main(argv):
    root = argv[1] if len(argv) > 1 else REPO
    with open(os.path.join(root, HPP), encoding="utf-8") as f:
        hpp_text = f.read()
    with open(os.path.join(root, CPP), encoding="utf-8") as f:
        cpp_text = f.read()
    self_test()

    real = violations(hpp_text, cpp_text)
    if real:
        for v in real:
            print("FAIL: " + v)
        print("FAIL: placement read-path gate: %d finding(s) on the real source" % len(real))
        return 1

    built = mutants(hpp_text, cpp_text)
    want_min = 3 + 2 * (len(SUMMARY_READERS) - 1) + 3 * len(PREDICATES) + 1 + 2 * len(PREFIX_READERS)
    if len(built) < want_min:
        print("FAIL: only %d of %d mutants could be built (anchors missing)" % (len(built), want_min))
        return 1
    for label, frag, h, c in built:
        res = violations(h, c)
        if not any(frag in r for r in res):
            print("FAIL: mutant went undetected: %s (wanted %r, got %r)" % (label, frag, res))
            return 1
    print("PASS: placement read-path gate holds on the real source; %d mutants caught" % len(built))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
