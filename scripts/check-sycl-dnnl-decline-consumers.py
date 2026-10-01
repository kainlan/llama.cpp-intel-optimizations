#!/usr/bin/env python3
"""Source gate for the oneDNN wrapper consumers of a declined scratchpad (llama.cpp-23mk S3-3, design 4.8).

`get_scratchpad_mem` can come back empty for a nonzero scratchpad: that is a decline, not an error. The three wrappers outside
ggml-sycl.cpp that ask for one (softmax, eltwise, binary_broadcast_row) used to throw std::runtime_error on it, which a
CHECK_TRY_ERROR-wrapped op turned into a process abort. They now return a [[nodiscard]] bool, false meaning declined, decide it
before their first write to the op's output, and their callers fall through to the path the default takes. The allocation-zone
gate's consumption clause (n) proves each result is used; this gate pins what that clause cannot see:

  - the signature shape, `[[nodiscard]] static bool NAME(`, returning true only on the executed path;
  - no throw of the old runtime_error in the wrapper;
  - softmax asks for its scratchpad before the pre-scale pass writes into dst (an in-place decline after that pass would hand
    soft_max_f32_sycl an input scaled twice);
  - each caller tests the result in an `if (...)` that returns, so a decline reaches the fallback that follows;
  - eltwise_inplace, a forwarder with no caller, is gone.

Every check is also run against a mutant of the same text and must fail there, so a regex that stopped matching would fail the
gate instead of passing it. Exit 0 on success, 1 on a violation, 2 when a file cannot be read.
"""
import argparse
import re
import sys
from pathlib import Path

SYCL = "ggml/src/ggml-sycl"
HDR = SYCL + "/dnnl-ops.hpp"
WRAPPERS = (
    ("DnnlSoftmaxWrapper", "softmax", "soft_max", "softmax.cpp"),
    ("DnnlEltwiseWrapper", "eltwise", "eltwise", "element_wise.cpp"),
    ("DnnlBinaryWrapper", "binary_broadcast_row", "binary_broadcast_row", "binbcast.cpp"),
)
CALLERS = {"softmax.cpp": ("DnnlSoftmaxWrapper::softmax", 1), "element_wise.cpp": ("DnnlEltwiseWrapper::eltwise", 3),
           "binbcast.cpp": ("DnnlBinaryWrapper::binary_broadcast_row", 1)}
THROW = re.compile(r"throw\s+std::runtime_error\s*\(\s*\"oneDNN scratchpad allocation failed\"")


def strip_comments(text):
    """Blank // and /* */ comments and string contents' comment markers are not a concern for these files."""
    out = re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)
    return re.sub(r"//[^\n]*", lambda m: " " * len(m.group(0)), out)


def class_body(text, cls):
    m = re.search(r"\bclass\s+%s\s*\{" % cls, text)
    if m is None:
        return None
    depth, i = 1, m.end()
    while i < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return text[m.end():i - 1]


def function_body(cls_text, name):
    """(has_nodiscard_bool_signature, body) of the static member `name`, or (False, None) when it does not exist."""
    m = re.search(r"(\[\[nodiscard\]\]\s*)?static\s+(\w+)\s+%s\s*\(" % re.escape(name), cls_text)
    if m is None:
        return False, None
    open_brace = cls_text.index("{", m.end())
    depth, i = 1, open_brace + 1
    while i < len(cls_text) and depth:
        depth += {"{": 1, "}": -1}.get(cls_text[i], 0)
        i += 1
    return bool(m.group(1)) and m.group(2) == "bool", cls_text[open_brace + 1:i - 1]


def check_header(text):
    errs = []
    text = strip_comments(text)
    for cls, name, _tag, _file in WRAPPERS:
        body_cls = class_body(text, cls)
        if body_cls is None:
            errs.append("%s: class %s not found" % (HDR, cls))
            continue
        shaped, body = function_body(body_cls, name)
        if body is None:
            errs.append("%s: %s::%s not found" % (HDR, cls, name))
            continue
        if not shaped:
            errs.append("%s: %s::%s must be `[[nodiscard]] static bool`: false means the scratchpad request was declined"
                        % (HDR, cls, name))
        if THROW.search(body):
            errs.append("%s: %s::%s still throws the scratchpad runtime_error instead of returning the decline" % (HDR, cls, name))
        if not re.search(r"get_scratchpad_mem[\s\S]*?return\s+false\s*;", body):
            errs.append("%s: %s::%s does not return false after asking for the scratchpad" % (HDR, cls, name))
        if not re.search(r"return\s+true\s*;\s*$", body.rstrip()):
            errs.append("%s: %s::%s does not end by returning true on the executed path" % (HDR, cls, name))
        if name == "softmax":
            ask, write = body.find("get_scratchpad_mem"), body.find("parallel_for")
            if ask < 0 or write < 0 or ask > write:
                errs.append("%s: softmax must ask for the scratchpad before the pre-scale pass writes into dst (ask at %d, "
                            "write at %d)" % (HDR, ask, write))
    if "eltwise_inplace" in text:
        errs.append("%s: eltwise_inplace (a forwarder with no caller) is back" % HDR)
    return errs


def check_caller(rel, text):
    errs = []
    text = strip_comments(text)
    sym, want = CALLERS[rel.rsplit("/", 1)[-1]]
    total = len(re.findall(re.escape(sym) + r"\s*\(", text))
    tested = len(re.findall(r"if\s*\(\s*" + re.escape(sym) + r"\s*\(", text))
    if total != want:
        errs.append("%s: expected %d call(s) of %s, found %d" % (rel, want, sym, total))
    if tested != total:
        errs.append("%s: %d of %d call(s) of %s do not test the result in an if (a decline must reach the fallback)"
                    % (rel, total - tested, total, sym))
    for m in re.finditer(r"if\s*\(\s*" + re.escape(sym) + r"\s*\(", text):
        depth, i = 1, m.end()
        while i < len(text) and depth:
            depth += {"(": 1, ")": -1}.get(text[i], 0)
            i += 1
        j = text.index("{", i)
        k, depth = j + 1, 1
        while k < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[k], 0)
            k += 1
        if not re.search(r"\breturn\s*;", text[j:k]):
            errs.append("%s: a tested %s call's branch does not return, so the fallback would run after a success" % (rel, sym))
    return errs


def run(files):
    errs = check_header(files[HDR])
    for rel in CALLERS:
        errs += check_caller(SYCL + "/" + rel, files[SYCL + "/" + rel])
    return errs


def mutants(files):
    """(label, mutated files) pairs, each of which a working gate must reject."""
    def edit(rel, old, new, count=1):
        text = files[rel]
        assert old in text, "mutant anchor missing: %s in %s" % (old, rel)
        return dict(files, **{rel: text.replace(old, new, count)})
    h = HDR
    ew, bb, sm = SYCL + "/element_wise.cpp", SYCL + "/binbcast.cpp", SYCL + "/softmax.cpp"
    out = [
        ("softmax loses [[nodiscard]]", edit(h, "[[nodiscard]] static bool softmax(", "static bool softmax(")),
        ("eltwise returns void", edit(h, "[[nodiscard]] static bool eltwise(", "[[nodiscard]] static void eltwise(")),
        ("binary_broadcast_row throws again", edit(h, "return false;", "throw std::runtime_error(\"oneDNN scratchpad allocation failed\");", 3)),
        ("eltwise_inplace comes back", edit(h, "// Supported element-wise operations", "static void eltwise_inplace() {}\n    // Supported element-wise operations")),
        ("an eltwise caller drops the test", edit(ew, "if (DnnlEltwiseWrapper::eltwise(", "(void) (DnnlEltwiseWrapper::eltwise(")),
        ("the binary caller drops the test", edit(bb, "if (DnnlBinaryWrapper::binary_broadcast_row(", "(void) (DnnlBinaryWrapper::binary_broadcast_row(")),
        ("the softmax caller drops the test", edit(sm, "if (DnnlSoftmaxWrapper::softmax(", "(void) (DnnlSoftmaxWrapper::softmax(")),
    ]
    text = files[h]
    i, j = text.index("get_scratchpad_mem(scratchpad_md, eng, q);"), text.index("q->parallel_for")
    if i < j:  # the mutant moves the pre-scale ahead of the scratchpad query
        block_start = text.rindex("// Pre-scale", 0, j)
        block_end = text.index("softmax_src = dst;", j)
        block_end = text.index("}", block_end) + 1
        block = text[block_start:block_end]
        moved = text.replace(block, "", 1)
        k = moved.index("auto softmax_pd = dnnl::softmax_forward::primitive_desc(")
        out.append(("softmax pre-scales before it asks for the scratchpad", dict(files, **{h: moved[:k] + block + "\n        " + moved[k:]})))
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
    print("PASS: the three wrappers return a [[nodiscard]] bool, decide the decline before writing, and every caller falls through")
    return 0


if __name__ == "__main__":
    sys.exit(main())
