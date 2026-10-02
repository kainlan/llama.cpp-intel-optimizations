#!/usr/bin/env python3
"""Source gate for the oneDNN wrapper consumers of a declined scratchpad (llama.cpp-23mk S3-3, design 4.8).

`get_scratchpad_mem` can come back empty for a nonzero scratchpad: that is a decline, not an error. The three wrappers outside
ggml-sycl.cpp that ask for one (softmax, eltwise, binary_broadcast_row) used to throw std::runtime_error on it, which a
CHECK_TRY_ERROR-wrapped op turned into a process abort. They now return a [[nodiscard]] bool, false meaning declined, decide it
before their first write to the op's output, and their callers fall through to the path the default takes. The allocation-zone
gate's consumption clause (n) proves each result is used; this gate pins what that clause cannot see. In each of the three
wrappers (comments blanked first):

  - the signature is `[[nodiscard]] static bool NAME(`;
  - the decision is exactly `if (X.get(true) == nullptr && Y.get_size() > 0) { return false; }`, it is the only `return false`,
    the scratchpad request precedes it, and the wrapper ends by returning true on the executed path;
  - nothing before that decision writes: no parallel_for, memcpy, memset, fill, submit or `.execute(`, no assignment to dst or
    dst_f, and no dnnl::memory object bound to the output (a decline after a write would hand the fallback modified inputs;
    softmax's pre-scale pass runs in place, so a late decision would apply the scale twice);
  - no `throw` and no `catch` anywhere in the class: a decline is a return value, and a catch-all would swallow one.

Outside the wrappers:

  - the deleted names stay deleted: eltwise_inplace and its respellings, DnnlBinaryWrapper::binary, DnnlReductionWrapper and
    reduce_last_dim (all matched on word boundaries; the first two threw the old runtime_error and had no caller);
  - each caller's condition is exactly `if (WRAPPER::fn(...))` with a body that is exactly `return;`, so a decline reaches the
    fallback that follows and a success skips it. A caller count that differs from CALLERS fails; when a legitimate caller is
    added, add it there.

Limit: the caller check sees the if and its body, not what follows the if. An unconditional `return;` after it would swallow the
fallback and this gate cannot tell.

Every check is also run against mutants of the same text and each must fail there, so a regex that stopped matching fails the
gate instead of passing it. A mutant whose anchor text has moved is an assertion error, not a skip. Exit 0 on success, 1 on a
violation, 2 when a file cannot be read.
"""
import argparse
import re
import sys
from pathlib import Path

SYCL = "ggml/src/ggml-sycl"
HDR = SYCL + "/dnnl-ops.hpp"
WRAPPERS = (
    ("DnnlSoftmaxWrapper", "softmax"),
    ("DnnlEltwiseWrapper", "eltwise"),
    ("DnnlBinaryWrapper", "binary_broadcast_row"),
)
# file -> (the call, how many calls the file holds). Update the count here when a legitimate caller is added.
CALLERS = {"softmax.cpp": ("DnnlSoftmaxWrapper::softmax", 1), "element_wise.cpp": ("DnnlEltwiseWrapper::eltwise", 3),
           "binbcast.cpp": ("DnnlBinaryWrapper::binary_broadcast_row", 1)}

DECISION = re.compile(r"if\s*\(\s*\w+\.get\(true\)\s*==\s*nullptr\s*&&\s*\w+\.get_size\(\)\s*>\s*0\s*\)\s*\{\s*return\s+false\s*;\s*\}")
WRITE = re.compile(r"\bparallel_for\b|\bmemcpy\b|\bmemset\b|\bfill\b|\bsubmit\b|\.\s*execute\s*\(|\bdst(?:_f)?\s*(?:\[[^\]]*\]\s*)?=(?!=)"
                   r"|\bdnnl::memory\s*\(")
DEAD = (
    (re.compile(r"\beltwise_in_?place\b"), "eltwise_inplace (a forwarder with no caller)"),
    (re.compile(r"\bDnnlReductionWrapper\b"), "DnnlReductionWrapper (an emptied class, deleted by design 4.8)"),
    (re.compile(r"\breduce_last_dim\b"), "reduce_last_dim (it threw the old runtime_error and had no caller)"),
    (re.compile(r"\bDnnlBinaryWrapper\s*::\s*binary\s*\("), "DnnlBinaryWrapper::binary (it threw the old runtime_error and had no caller)"),
)
TOKENS = re.compile(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|/\*.*?\*/|//[^\n]*', re.S)


def strip_comments(text):
    """Blank every // and /* */ comment, keeping length and newlines; string and character literals are skipped over so a
    comment marker inside one is not taken for a comment."""
    def blank(m):
        s = m.group(0)
        return s if s[0] in "\"'" else re.sub(r"[^\n]", " ", s)
    return TOKENS.sub(blank, text)


def balanced(text, start, open_ch, close_ch):
    """Index just past the close that matches an already-consumed open at start - 1."""
    depth, i = 1, start
    while i < len(text) and depth:
        depth += {open_ch: 1, close_ch: -1}.get(text[i], 0)
        i += 1
    return i


def class_body(text, cls):
    m = re.search(r"\bclass\s+%s\s*\{" % cls, text)
    if m is None:
        return None
    return text[m.end():balanced(text, m.end(), "{", "}") - 1]


def function_body(cls_text, name):
    """(has_nodiscard_bool_signature, body) of the static member `name`, or (False, None) when it does not exist."""
    m = re.search(r"(\[\[nodiscard\]\]\s*)?static\s+(\w+)\s+%s\s*\(" % re.escape(name), cls_text)
    if m is None:
        return False, None
    open_brace = cls_text.index("{", m.end())
    return bool(m.group(1)) and m.group(2) == "bool", cls_text[open_brace + 1:balanced(cls_text, open_brace + 1, "{", "}") - 1]


def check_wrapper(cls, name, body_cls):
    errs = []
    where = "%s: %s::%s" % (HDR, cls, name)
    for kw in ("throw", "catch"):
        if re.search(r"\b%s\b" % kw, body_cls):
            errs.append("%s: %s appears in the class; a decline is a return value, never an exception or a swallowed one" % (HDR + ": " + cls, kw))
    shaped, body = function_body(body_cls, name)
    if body is None:
        return errs + ["%s not found" % where]
    if not shaped:
        errs.append("%s must be `[[nodiscard]] static bool`: false means the scratchpad request was declined" % where)
    decisions = list(DECISION.finditer(body))
    if len(decisions) != 1:
        errs.append("%s must carry exactly one decision `if (X.get(true) == nullptr && Y.get_size() > 0) { return false; }`, found %d"
                    % (where, len(decisions)))
    if len(re.findall(r"\breturn\s+false\b", body)) != 1:
        errs.append("%s must have exactly one `return false`, the scratchpad decision" % where)
    if not re.search(r"\breturn\s+true\s*;\s*$", body.rstrip()):
        errs.append("%s does not end by returning true on the executed path" % where)
    if decisions:
        before = body[:decisions[0].start()]
        if "get_scratchpad_mem" not in before:
            errs.append("%s decides before it asks for the scratchpad" % where)
        for m in WRITE.finditer(before):
            errs.append("%s writes or binds the output before the decline decision (`%s`)" % (where, m.group(0).strip()))
    return errs


def check_header(text):
    errs = []
    text = strip_comments(text)
    for cls, name in WRAPPERS:
        body_cls = class_body(text, cls)
        if body_cls is None:
            errs.append("%s: class %s not found" % (HDR, cls))
            continue
        errs += check_wrapper(cls, name, body_cls)
    for rx, what in DEAD:
        if rx.search(text):
            errs.append("%s: %s is back" % (HDR, what))
    binary_cls = class_body(text, "DnnlBinaryWrapper")
    if binary_cls is not None and function_body(binary_cls, "binary")[1] is not None:
        errs.append("%s: DnnlBinaryWrapper::binary is back (it threw the old runtime_error and had no caller)" % HDR)
    return errs


def check_caller(rel, text):
    errs = []
    text = strip_comments(text)
    sym, want = CALLERS[rel.rsplit("/", 1)[-1]]
    calls = list(re.finditer(re.escape(sym) + r"\s*\(", text))
    if len(calls) != want:
        errs.append("%s: expected %d call(s) of %s, found %d; a legitimate new caller must be added to CALLERS in this gate"
                    % (rel, want, sym, len(calls)))
    for m in calls:
        if not re.search(r"\bif\s*\(\s*$", text[:m.start()]):
            errs.append("%s: a call of %s is not directly the condition of an if (a decline must reach the fallback)" % (rel, sym))
            continue
        after = text[balanced(text, m.end(), "(", ")"):]
        branch = re.match(r"\s*\)\s*\{", after)
        if branch is None:
            errs.append("%s: the if condition around a call of %s is more than the call itself" % (rel, sym))
            continue
        body = after[branch.end():balanced(after, branch.end(), "{", "}") - 1]
        if body.strip() != "return;":
            errs.append("%s: a tested %s call's branch is not exactly `return;`, so the fallback could run after a success"
                        % (rel, sym))
    return errs


def run(files):
    errs = check_header(files[HDR])
    for rel in CALLERS:
        errs += check_caller(SYCL + "/" + rel, files[SYCL + "/" + rel])
    return errs


DECIDE = "        if (scratchpad_mem.get(true) == nullptr && scratchpad_md.get_size() > 0) {\n            return false;\n        }\n"


def mutants(files):
    """(label, mutated files) pairs, each of which a working gate must reject."""
    def edit(rel, old, new, count=1):
        text = files[rel]
        assert old in text, "mutant anchor missing: %r in %s" % (old, rel)
        return dict(files, **{rel: text.replace(old, new, count)})
    h = HDR
    ew, bb, sm = SYCL + "/element_wise.cpp", SYCL + "/binbcast.cpp", SYCL + "/softmax.cpp"
    assert files[h].count(DECIDE) == 3, "mutant anchor: expected the decision three times in " + h
    first = files[h].index(DECIDE)
    out = [
        ("softmax loses [[nodiscard]]", edit(h, "[[nodiscard]] static bool softmax(", "static bool softmax(")),
        ("eltwise returns void", edit(h, "[[nodiscard]] static bool eltwise(", "[[nodiscard]] static void eltwise(")),
        ("a wrapper throws again", edit(h, "return false;", "throw std::runtime_error(\"oneDNN scratchpad allocation failed\");", 3)),
        ("a wrapper throws something else", edit(h, "return false;", "throw 1;", 1)),
        ("a wrapper swallows with catch", edit(h, "        softmax_prim.execute(stream, args);",
                                               "        try { softmax_prim.execute(stream, args); } catch (...) {}")),
        ("eltwise_inplace comes back", edit(h, "// Supported element-wise operations", "static void eltwise_inplace() {}\n    // Supported element-wise operations")),
        ("eltwise_inplace comes back respelled", edit(h, "// Supported element-wise operations", "static void eltwise_in_place() {}\n    // Supported element-wise operations")),
        ("DnnlBinaryWrapper::binary comes back", edit(h, "    // Supported binary operations", "    static void binary(int) {}\n    // Supported binary operations")),
        ("DnnlReductionWrapper comes back", edit(h, "#endif // GGML_SYCL_DNNL\n", "class DnnlReductionWrapper {};\n#endif // GGML_SYCL_DNNL\n")),
        ("reduce_last_dim comes back", edit(h, "    // Supported binary operations", "    static void reduce_last_dim() {}\n    // Supported binary operations")),
        ("the decision returns true", edit(h, "            return false;", "            return true;", 1)),
        ("the decision drops its size guard", edit(h, "scratchpad_mem.get(true) == nullptr && scratchpad_md.get_size() > 0", "scratchpad_mem.get(true) == nullptr")),
        ("eltwise writes dst before the decision", edit(h, "        auto scratchpad_md = eltwise_pd.scratchpad_desc();",
                                                        "        q->memcpy(dst, src, nelements * sizeof(float));\n        auto scratchpad_md = eltwise_pd.scratchpad_desc();")),
        ("binary_broadcast_row writes dst before the decision", edit(h, "        auto scratchpad_md  = binary_pd.scratchpad_desc();",
                                                                     "        q->memset(dst, 0, batch * features * sizeof(float));\n        auto scratchpad_md  = binary_pd.scratchpad_desc();")),
        ("softmax copies into dst before the decision", edit(h, "        auto scratchpad_md = softmax_pd.scratchpad_desc();",
                                                             "        q->memcpy(dst, src, batch * features * sizeof(float));\n        auto scratchpad_md = softmax_pd.scratchpad_desc();")),
        ("softmax pre-scales before it asks", edit(h, "        auto scratchpad_md = softmax_pd.scratchpad_desc();",
                                                   "        q->parallel_for(sycl::range<1>(batch * features), [=](sycl::id<1> i) {});\n        auto scratchpad_md = softmax_pd.scratchpad_desc();")),
        ("softmax assigns dst before the decision", edit(h, "        auto scratchpad_md = softmax_pd.scratchpad_desc();",
                                                         "        ((float *) dst)[0] = 0.0f;\n        auto scratchpad_md = softmax_pd.scratchpad_desc();")),
        ("eltwise binds the output before the decision", edit(h, "        auto scratchpad_md = eltwise_pd.scratchpad_desc();",
                                                              "        auto dst_mem = dnnl::memory(md, eng, dst);\n        auto scratchpad_md = eltwise_pd.scratchpad_desc();")),
        ("an eltwise caller drops the test", edit(ew, "if (DnnlEltwiseWrapper::eltwise(", "(void) (DnnlEltwiseWrapper::eltwise(")),
        ("the binary caller drops the test", edit(bb, "if (DnnlBinaryWrapper::binary_broadcast_row(", "(void) (DnnlBinaryWrapper::binary_broadcast_row(")),
        ("the softmax caller drops the test", edit(sm, "if (DnnlSoftmaxWrapper::softmax(", "(void) (DnnlSoftmaxWrapper::softmax(")),
        ("the softmax caller swallows a decline with || true", edit(sm, "                stream)) {\n            return;", "                stream) || true) {\n            return;")),
        ("the softmax caller negates the test", edit(sm, "if (DnnlSoftmaxWrapper::softmax(", "if (!DnnlSoftmaxWrapper::softmax(")),
        ("the softmax caller's branch no longer returns", edit(sm, "                stream)) {\n            return;\n        }", "                stream)) {\n            (void) 0;\n        }")),
        ("an extra unchecked softmax caller", edit(sm, "    if (use_f16) {", "    DnnlSoftmaxWrapper::softmax(ctx, src0_d, dst_d, 1, 1, 1.0f, DnnlSoftmaxWrapper::to_dt<float>(), stream);\n    if (use_f16) {")),
    ]
    # the decision moved after the pre-scale: ask first, decide later (the double-application hazard)
    moved = files[h].replace(DECIDE, "", 1)
    anchor = "        auto src_mem = dnnl::memory(src_md, eng, const_cast<void *>(softmax_src));"
    assert anchor in moved, "mutant anchor missing: softmax src_mem in " + h
    assert first < files[h].index(anchor), "mutant anchor: the first decision is not softmax's"
    out.append(("softmax decides after the pre-scale pass", dict(files, **{h: moved.replace(anchor, DECIDE + anchor, 1)})))
    return out


def load(root):
    files = {}
    for rel in (HDR,) + tuple(SYCL + "/" + r for r in CALLERS):
        try:
            files[rel] = (Path(root) / rel).read_text()
        except OSError as exc:
            print("FAIL: cannot read %s (%s)" % (rel, exc))
            sys.exit(2)
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    a = ap.parse_args()
    files = load(a.root)
    errs = run(files)
    for e in errs:
        print("FAIL " + e)
    if errs:
        print("FAIL: %d violation(s)" % len(errs))
        return 1
    survived = [label for label, mutated in mutants(files) if not run(mutated)]
    for label in survived:
        print("FAIL mutant survived: %s (the gate would not notice)" % label)
    if survived:
        return 1
    print("PASS: the three wrappers return a [[nodiscard]] bool, decide the decline before any write, and every caller falls through")
    return 0


if __name__ == "__main__":
    sys.exit(main())
