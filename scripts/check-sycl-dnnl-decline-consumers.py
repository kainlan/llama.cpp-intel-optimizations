#!/usr/bin/env python3
"""Source gate for the oneDNN wrapper consumers of a declined scratchpad (llama.cpp-23mk S3-3, design 4.8).

`get_scratchpad_mem` can come back empty for a nonzero scratchpad: that is a decline, not an error. The three wrappers outside
ggml-sycl.cpp that ask for one (softmax, eltwise, binary_broadcast_row) used to throw std::runtime_error on it, which a
CHECK_TRY_ERROR-wrapped op turned into a process abort. They now return a [[nodiscard]] bool, false meaning declined, decide it
before their first write to the op's output, and their callers fall through to the path the default takes. The allocation-zone
gate's consumption clause (n) proves each result is used; this gate pins what that clause cannot see. In each of the three
wrappers (comments blanked first):

  - the signature is `[[nodiscard]] static bool NAME(`;
  - the decision is exactly `if (ggml_sycl_scratchpad_declined(SITE, X, Y)) { return false; }` with the wrapper's own site tag,
    it is the only `return false`, the scratchpad request precedes it, and the wrapper ends by executing the primitive and
    returning true. The helper's body is pinned to the hook-first form below, so the test seam reaches every wrapper and the
    old condition (nullptr and a nonzero size) is the only other way to decline;
  - `ggml_sycl_dnnl_note_engaged(SITE)` is called once, unconditionally (the statement before it ends in `;` or `}`), after the
    decision and before the primitive is executed, so a run that reached the wrapper can say so; the decision's arguments are
    the literal `scratchpad_mem, scratchpad_md`;
  - nothing before that decision writes (a decline after a write would hand the fallback modified inputs; softmax's pre-scale
    pass runs in place, so a late decision would apply the scale twice): no parallel_for, single_task, memcpy, memmove, memset,
    fill, fill_n, copy, copy_n, transform, submit or `.execute(`, no mention of dst or dst_f at all (an alias would be a write
    the token list cannot see), no const_cast, static_cast or reinterpret_cast, no C-style cast of src, src0, src1 or dst (the
    input pointers are const only by declaration), and no dnnl::memory object bound to the output. The signature check pins
    the parameters those names stand for;
  - no `throw` and no `catch` anywhere in the class: a decline is a return value, and a catch-all would swallow one.

Outside the wrappers:

  - the deleted names stay deleted: eltwise_inplace and its respellings, DnnlBinaryWrapper::binary, DnnlReductionWrapper and
    reduce_last_dim (all matched on word boundaries; the first two threw the old runtime_error and had no caller);
  - each caller's condition is exactly `if (WRAPPER::fn(...))` with a body that is exactly `return;`, so a decline reaches the
    fallback that follows and a success skips it. A caller count that differs from CALLERS fails; when a legitimate caller is
    added, add it there.

Before the if, between the caller's `#if GGML_SYCL_DNNL` line and the if, nothing may leave (return, goto, throw, GGML_ABORT,
GGML_ASSERT, abort, exit, _Exit, quick_exit, terminate or longjmp): such a statement would skip the wrapper and the fallback
alike.

Between the if and the `#endif` / `#else` that ends the DNNL section only whitespace and the closing braces of its blocks may
appear: no statement, goto, throw, abort or else can follow it. When that line is an `#else`, its arm (up to the matching
`#endif`, nested #if blocks counted) may not contain any of the same leaving words, since the build without DNNL would leave
before the fallback. That is the shape of all five real callers. Limit: a `return` placed after the `#endif`, in the fallback
code itself, is not seen; the device test is the only catch for that.

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
    ("DnnlSoftmaxWrapper", "softmax", "DNNL_SOFTMAX"),
    ("DnnlEltwiseWrapper", "eltwise", "DNNL_ELTWISE"),
    ("DnnlBinaryWrapper", "binary_broadcast_row", "DNNL_BINARY_ROW"),
)
HELPER = "ggml_sycl_scratchpad_declined"
HELPER_BODY = ("if (ggml_sycl_scratchpad_site_hook(site)) { return true; } "
               "return scratchpad_mem.get(true) == nullptr && scratchpad_md.get_size() > 0;")
# file -> (the call, how many calls the file holds). Update the count here when a legitimate caller is added.
CALLERS = {"softmax.cpp": ("DnnlSoftmaxWrapper::softmax", 1), "element_wise.cpp": ("DnnlEltwiseWrapper::eltwise", 3),
           "binbcast.cpp": ("DnnlBinaryWrapper::binary_broadcast_row", 1)}

DECISION = re.compile(r"if\s*\(\s*" + HELPER + r"\s*\(\s*GGML_SYCL_SCRATCHPAD_SITE_(\w+)\s*,\s*scratchpad_mem\s*,\s*scratchpad_md\s*\)\s*\)"
                      r"\s*\{\s*return\s+false\s*;\s*\}")
NOTE = re.compile(r"\bggml_sycl_dnnl_note_engaged\s*\(\s*GGML_SYCL_SCRATCHPAD_SITE_(\w+)\s*\)\s*;")
WRITE = re.compile(r"\bparallel_for\b|\bsingle_task\b|\bmemcpy\b|\bmemmove\b|\bmemset\b|\bfill\b|\bfill_n\b|\bcopy\b|\bcopy_n\b"
                   r"|\btransform\b|\bsubmit\b|\.\s*execute\s*\(|\bdst(?:_f)?\b|\b(?:const|static|reinterpret)_cast\b"
                   r"|\(\s*[\w:\s]+\*\s*\)\s*(?:src\d?|dst)\b|\bdnnl::memory\s*\(")
# What may not appear in the #else arm of a caller's `#if GGML_SYCL_DNNL` section: any way to leave before the fallback runs.
LEAVE = re.compile(r"\breturn\b|\bgoto\b|\bthrow\b|\bGGML_ABORT\b|\bGGML_ASSERT\b|\babort\b|\bexit\b|\b_Exit\b|\bquick_exit\b"
                   r"|\bterminate\b|\blongjmp\b")
GEMM = SYCL + "/gemm.hpp"
COMMON = SYCL + "/common.hpp"
MAIN = SYCL + "/ggml-sycl.cpp"
OUTPROD = SYCL + "/outprod.cpp"
# DnnlGemmWrapper's consumers of get_scratchpad_mem (llama.cpp-23mk S3-4): function -> (site, declined result, the type it
# returns, how many scratchpad requests it makes). Every request is unconditional, decided by the helper before anything is
# submitted, and a decline is a return value.
GEMM_CONSUMERS = (
    ("gemm", "DNNL_GEMM", "std::nullopt", "std::optional<sycl::event>", 2),
    ("woq_gemm_q8_0", "DNNL_WOQ_Q8_0", "false", "bool", 1),
    ("woq_gemm_q4_0_impl", "DNNL_WOQ_Q4_0", "false", "bool", 1),
    ("gemm_batch_strided", "DNNL_GEMM_BATCH", "std::nullopt", "std::optional<sycl::event>", 2),
    ("woq_gemm_batch_mxfp4", "DNNL_WOQ_MXFP4_BATCH", "std::nullopt", "std::optional<sycl::event>", 2),
)
GEMM_FORWARDERS = (("row_gemm", "std::optional<sycl::event>"), ("woq_gemm_q4_0", "bool"))
GEMM_DEAD = ("woq_gemm_q4_0_packed", "gemm_batch_array", "row_gemm_batch")
GEMM_DECISION = re.compile(r"if\s*\(\s*" + HELPER + r"\s*\(\s*GGML_SYCL_SCRATCHPAD_SITE_(\w+)\s*,\s*scratchpad_mem\s*,\s*"
                           r"(?:scratchpad_md|cached\w*\s*->\s*scratchpad_md)\s*\)\s*\)\s*\{\s*return\s+([\w:]+)\s*;\s*\}")
# Statements in ggml-sycl.cpp and outprod.cpp that carry a declined gemm to its declared next path: text -> how many.
MAIN_PINS = (
    ('throw ggml_sycl_fallback_error("dnnl_gemm declined in mul_mat\'s f16 dense arm', 1),
    ('throw ggml_sycl_fallback_error("dnnl_gemm declined in mul_mat\'s f32 dense arm', 1),
    ('throw ggml_sycl_fallback_error("dnnl_decline_after_write:dnnl_gemm");', 3),
    ('throw ggml_sycl_fallback_error("dnnl_gemm declined and batched_f16_fallback failed");', 2),
    ("batched_declined && !ggml_sycl_mul_mat_batched_f16_fallback(", 2),
    ("[[nodiscard]] static bool ggml_sycl_mul_mat_batched_sycl(", 1),
)
OUTPROD_PIN = 'throw ggml_sycl_fallback_error("dnnl_gemm declined in out_prod'
OLD_THROW = 'std::runtime_error("oneDNN scratchpad allocation failed")'
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


PARAMS = {"softmax": ("const void * src", "void * dst"), "eltwise": ("const void * src", "void * dst"),
          "binary_broadcast_row": ("const void * src0", "const void * src1", "void * dst")}


def function_params(cls_text, name):
    m = re.search(r"static\s+\w+\s+%s\s*\(" % re.escape(name), cls_text)
    if m is None:
        return None
    return " ".join(re.sub(r"\s*\*\s*", " * ", cls_text[m.end():balanced(cls_text, m.end(), "(", ")") - 1]).split())


def check_wrapper(cls, name, site, body_cls):
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
    params = [p.strip() for p in (function_params(body_cls, name) or "").split(",")]
    for want in PARAMS[name]:
        if want not in params:
            errs.append("%s must keep the parameter `%s`: the no-write rule is stated over those names" % (where, want))
    decisions = list(DECISION.finditer(body))
    if len(decisions) != 1:
        errs.append("%s must carry exactly one decision `if (%s(SITE, X, Y)) { return false; }`, found %d"
                    % (where, HELPER, len(decisions)))
    if decisions and decisions[0].group(1) != site:
        errs.append("%s decides under site %s, not its own %s" % (where, decisions[0].group(1), site))
    if len(re.findall(r"\breturn\s+false\b", body)) != 1:
        errs.append("%s must have exactly one `return false`, the scratchpad decision" % where)
    if not re.search(r"\.\s*execute\s*\([^;]*\)\s*;\s*return\s+true\s*;\s*$", body.rstrip()):
        errs.append("%s does not end by submitting the primitive and returning true on the executed path" % where)
    if decisions:
        before = body[:decisions[0].start()]
        if "get_scratchpad_mem" not in before:
            errs.append("%s decides before it asks for the scratchpad" % where)
        for m in WRITE.finditer(before):
            errs.append("%s writes or binds the output before the decline decision (`%s`)" % (where, m.group(0).strip()))
        notes = list(NOTE.finditer(body))
        execute = re.search(r"\.\s*execute\s*\(", body)
        if len(notes) != 1 or notes[0].group(1) != site:
            errs.append("%s must call ggml_sycl_dnnl_note_engaged(GGML_SYCL_SCRATCHPAD_SITE_%s) exactly once, found %d"
                        % (where, site, len(notes)))
        elif not decisions[0].end() <= notes[0].start() < (execute.start() if execute else len(body)):
            errs.append("%s calls ggml_sycl_dnnl_note_engaged outside the span between the decision and the execute" % where)
        elif not re.search(r"[;}]\s*$", body[:notes[0].start()]):
            errs.append("%s must call ggml_sycl_dnnl_note_engaged unconditionally, as a statement of its own" % where)
    return errs


def check_header(text):
    errs = []
    text = strip_comments(text)
    helper = re.search(r"\binline\s+bool\s+%s\s*\(\s*ggml_sycl_scratchpad_site\s+site\s*,\s*const\s+dnnl::memory\s*&\s*scratchpad_mem\s*,"
                       r"\s*const\s+dnnl::memory::desc\s*&\s*scratchpad_md\s*\)\s*\{" % HELPER, text)
    if helper is None:
        errs.append("%s: %s(ggml_sycl_scratchpad_site site, const dnnl::memory & scratchpad_mem, const dnnl::memory::desc & "
                    "scratchpad_md) not found" % (HDR, HELPER))
    elif " ".join(text[helper.end():balanced(text, helper.end(), "{", "}") - 1].split()) != HELPER_BODY:
        errs.append("%s: the body of %s is not exactly `%s`" % (HDR, HELPER, HELPER_BODY))
    for cls, name, site in WRAPPERS:
        body_cls = class_body(text, cls)
        if body_cls is None:
            errs.append("%s: class %s not found" % (HDR, cls))
            continue
        errs += check_wrapper(cls, name, site, body_cls)
    for rx, what in DEAD:
        if rx.search(text):
            errs.append("%s: %s is back" % (HDR, what))
    binary_cls = class_body(text, "DnnlBinaryWrapper")
    if binary_cls is not None and function_body(binary_cls, "binary")[1] is not None:
        errs.append("%s: DnnlBinaryWrapper::binary is back (it threw the old runtime_error and had no caller)" % HDR)
    return errs


def member_body(text, name):
    """(declaration text before the name, body) of the static member `name` of DnnlGemmWrapper, or (None, None)."""
    m = re.search(r"((?:\[\[nodiscard\]\]\s*)?static\s+[\w:<>]+\s+)%s\s*\(" % re.escape(name), text)
    if m is None:
        return None, None
    params_end = balanced(text, m.end(), "(", ")")
    open_brace = text.index("{", params_end)
    return m.group(1), text[open_brace + 1:balanced(text, open_brace + 1, "{", "}") - 1]


def check_gemm(text):
    errs = []
    text = strip_comments(text)
    if OLD_THROW in text:
        errs.append("%s: a consumer throws the old oneDNN scratchpad runtime_error again; a decline is a return value" % GEMM)
    for name in GEMM_DEAD:
        if re.search(r"\b%s\b" % name, text):
            errs.append("%s: %s is back (a forwarder with no caller)" % (GEMM, name))
    total = 0
    for name, site, result, rtype, want in GEMM_CONSUMERS:
        decl, body = member_body(text, name)
        where = "%s: DnnlGemmWrapper::%s" % (GEMM, name)
        if body is None:
            errs.append("%s not found" % where)
            continue
        if not decl.startswith("[[nodiscard]]") or not re.search(r"static\s+%s\s*$" % re.escape(rtype), decl):
            errs.append("%s must be `[[nodiscard]] static %s`: a declined result may not be dropped" % (where, rtype))
        calls = list(re.finditer(r"\bget_scratchpad_mem\s*\(", body))
        total += len(calls)
        if len(calls) != want:
            errs.append("%s makes %d scratchpad request(s), expected %d" % (where, len(calls), want))
        decisions = list(GEMM_DECISION.finditer(body))
        if len(decisions) != want:
            errs.append("%s must decide each request with `if (%s(SITE, scratchpad_mem, <descriptor>)) { return %s; }`, found %d"
                        % (where, HELPER, result, len(decisions)))
        for d in decisions:
            if d.group(1) != site:
                errs.append("%s decides under site %s, not its own %s" % (where, d.group(1), site))
            if d.group(2) != result:
                errs.append("%s declines with `return %s`, expected `return %s`" % (where, d.group(2), result))
        for c in calls:
            stmt_start = max(body.rfind(";", 0, c.start()), body.rfind("{", 0, c.start()), body.rfind("}", 0, c.start()))
            if not re.fullmatch(r"\s*auto\s+scratchpad_mem\s*=\s*ctx\s*\.\s*", body[stmt_start + 1:c.start()]):
                errs.append("%s asks for the scratchpad other than as a plain `auto scratchpad_mem = ctx.get_scratchpad_mem(...)` "
                            "statement: the request must be unconditional, with the guard on the argument insert alone" % where)
            nxt = re.match(r"[^;]*;\s*", body[c.start():])
            after = body[c.start() + nxt.end():] if nxt else ""
            if not GEMM_DECISION.match(after):
                errs.append("%s: a scratchpad request is not followed directly by its decision" % where)
        notes = list(NOTE.finditer(body))
        if len(notes) != want or any(n.group(1) != site for n in notes):
            errs.append("%s must call ggml_sycl_dnnl_note_engaged(GGML_SYCL_SCRATCHPAD_SITE_%s) once per request, found %d"
                        % (where, site, len(notes)))
        for d, n in zip(decisions, notes):
            between = r"\s*if\s*\(\s*query_only\s*\)\s*\{\s*return\s+sycl::event\s*\{\s*\}\s*;\s*\}\s*" if name == "gemm" else r"\s*"
            if n.start() < d.end() or not re.fullmatch(between, body[d.end():n.start()]):
                errs.append("%s must note the engaged call right after its decision%s" % (
                    where, ", after query_only's exit (a pre-query is not an engaged call)" if name == "gemm" else ""))
        for m in re.finditer(r"\bthrow\b", body):
            if name in ("woq_gemm_q8_0", "woq_gemm_q4_0_impl", "gemm", "gemm_batch_strided") and \
                    re.match(r"\s+std::runtime_error\(\"oneDNN scratchpad", body[m.end() - 1:]):
                errs.append("%s throws on a declined scratchpad" % where)
    if total != sum(w for *_, w in GEMM_CONSUMERS) or len(re.findall(r"\bget_scratchpad_mem\s*\(", text)) != total:
        errs.append("%s: a get_scratchpad_mem request sits outside the listed consumers (found %d, listed %d)"
                    % (GEMM, len(re.findall(r"\bget_scratchpad_mem\s*\(", text)), total))
    # the 2-D fallback asks once, before its batch loop
    _, mx = member_body(text, "woq_gemm_batch_mxfp4")
    if mx is not None:
        loop = mx.find("for (int b = 0; b < batch_size; ++b)")
        late = [m.start() for m in re.finditer(r"\bget_scratchpad_mem\s*\(", mx) if loop >= 0 and m.start() > loop]
        if loop < 0 or late:
            errs.append("%s: woq_gemm_batch_mxfp4's 2-D fallback must ask for the scratchpad once before its batch loop" % GEMM)
    for name, rtype in GEMM_FORWARDERS:
        decl, body = member_body(text, name)
        if body is None:
            errs.append("%s: DnnlGemmWrapper::%s not found" % (GEMM, name))
        elif not decl.startswith("[[nodiscard]]") or not re.search(r"static\s+%s\s*$" % re.escape(rtype), decl):
            errs.append("%s: DnnlGemmWrapper::%s must be `[[nodiscard]] static %s`" % (GEMM, name, rtype))
    return errs


def check_common(text):
    errs = []
    text = strip_comments(text)
    m = re.search(r"(\[\[nodiscard\]\]\s*)?dnnl::memory\s+get_scratchpad_mem\s*\(", text)
    if m is None:
        return ["%s: get_scratchpad_mem not found" % COMMON]
    if not m.group(1):
        errs.append("%s: get_scratchpad_mem must be [[nodiscard]]" % COMMON)
    open_brace = text.index("{", balanced(text, m.end(), "(", ")"))
    body = text[open_brace + 1:balanced(text, open_brace + 1, "{", "}") - 1]
    # The whole prefix is pinned, not just the order: a 0 B descriptor returns the empty memory and takes no lock, any other
    # size reaches the lock, and nothing else sits between (a respelled test or a different empty value changes the answer
    # every unconditional caller decides on).
    prefix = re.compile(r"\s*size_t\s+scratchpad_size\s*=\s*scratchpad_md\s*\.\s*get_size\s*\(\s*\)\s*;"
                        r"\s*if\s*\(\s*scratchpad_size\s*==\s*0\s*\)\s*\{\s*return\s+dnnl::memory\s*\(\s*\)\s*;\s*\}"
                        r"\s*std::lock_guard<std::mutex>\s+lock\s*\(\s*dnnl_mutex\s*\)\s*;")
    if not prefix.match(body):
        errs.append("%s: get_scratchpad_mem must open with `size_t scratchpad_size = scratchpad_md.get_size(); if (scratchpad_size "
                    "== 0) { return dnnl::memory(); }` and only then take dnnl_mutex: the 0 B path returns the empty memory "
                    "without the lock, every other size takes it" % COMMON)
    if len(re.findall(r"\bdnnl_mutex\b", body)) != 1:
        errs.append("%s: get_scratchpad_mem must take dnnl_mutex exactly once" % COMMON)
    return errs


def ws_pattern(text):
    """A regex for `text` that matches it however a formatter wrapped it: no whitespace is significant."""
    return r"\s*".join(re.escape(c) for c in re.sub(r"\s+", "", text))


def check_main(main_text, outprod_text):
    errs = []
    main_text, outprod_text = strip_comments(main_text), strip_comments(outprod_text)
    if OLD_THROW in main_text:
        errs.append("%s: the old oneDNN scratchpad runtime_error is back" % MAIN)
    for text, want in MAIN_PINS:
        found = len(re.findall(ws_pattern(text), main_text))
        if found != want:
            errs.append("%s: expected %d of `%s`, found %d" % (MAIN, want, text, found))
    found = len(re.findall(ws_pattern(OUTPROD_PIN), outprod_text))
    if found != 1:
        errs.append("%s: a declined gemm in out_prod must fail by name (`%s`), found %d" % (OUTPROD, OUTPROD_PIN, found))
    return errs


def else_arm_of(text):
    """The text of the #else arm that begins `text` (after the closing braces), up to its matching #endif; None if there is none.
    Nested #if blocks inside the arm are counted so the scan stops at the right #endif."""
    m = re.match(r"[\s}]*#\s*(\w+)[^\n]*\n", text)
    if m is None or m.group(1) != "else":
        return None
    depth = 0
    pos = m.end()
    for line in re.finditer(r"[^\n]*\n|[^\n]+$", text[pos:]):
        d = re.match(r"\s*#\s*(\w+)", line.group(0))
        if d is not None:
            if d.group(1) in ("if", "ifdef", "ifndef"):
                depth += 1
            elif d.group(1) == "endif":
                if depth == 0:
                    return text[pos:pos + line.start()]
                depth -= 1
    return text[pos:]


def check_caller(rel, text):
    errs = []
    text = strip_comments(text)
    sym, want = CALLERS[rel.rsplit("/", 1)[-1]]
    calls = list(re.finditer(re.escape(sym) + r"\s*\(", text))
    if len(calls) != want:
        errs.append("%s: expected %d call(s) of %s, found %d; a legitimate new caller must be added to CALLERS in this gate"
                    % (rel, want, sym, len(calls)))
    for m in calls:
        section = [s for s in re.finditer(r"^[ \t]*#[ \t]*if[ \t]+GGML_SYCL_DNNL\b[^\n]*\n", text[:m.start()], re.M)]
        if not section:
            errs.append("%s: a call of %s is not inside an `#if GGML_SYCL_DNNL` section" % (rel, sym))
        elif LEAVE.search(text[section[-1].end():m.start()]):
            errs.append("%s: code between the `#if GGML_SYCL_DNNL` line and the if around %s can leave (return, goto, throw or "
                        "abort), skipping both the wrapper and its fallback" % (rel, sym))
        if not re.search(r"\bif\s*\(\s*$", text[:m.start()]):
            errs.append("%s: a call of %s is not directly the condition of an if (a decline must reach the fallback)" % (rel, sym))
            continue
        after = text[balanced(text, m.end(), "(", ")"):]
        branch = re.match(r"\s*\)\s*\{", after)
        if branch is None:
            errs.append("%s: the if condition around a call of %s is more than the call itself" % (rel, sym))
            continue
        end = balanced(after, branch.end(), "{", "}")
        body = after[branch.end():end - 1]
        if body.strip() != "return;":
            errs.append("%s: a tested %s call's branch is not exactly `return;`, so the fallback could run after a success"
                        % (rel, sym))
        # Only whitespace and the closers of the enclosing blocks may sit between the if and the preprocessor line that ends the
        # DNNL section (`#endif` or `#else`): no statement, jump, throw, abort or else can follow it.
        if not re.match(r"[\s}]*#\s*(?:endif|else)\b", after[end:]):
            errs.append("%s: something other than the closing braces of its blocks follows the if around %s before the end of "
                        "the DNNL section, so the fallback may never run" % (rel, sym))
        else_arm = else_arm_of(after[end:])
        if else_arm is not None and LEAVE.search(else_arm):
            errs.append("%s: the #else arm of the DNNL section around %s leaves (return, goto, throw or abort), so the fallback "
                        "would not run in the build without DNNL" % (rel, sym))
    return errs


def run(files):
    errs = check_header(files[HDR]) + check_gemm(files[GEMM]) + check_common(files[COMMON]) + \
        check_main(files[MAIN], files[OUTPROD])
    for rel in CALLERS:
        errs += check_caller(SYCL + "/" + rel, files[SYCL + "/" + rel])
    return errs


def decide(site):
    return ("        if (ggml_sycl_scratchpad_declined(GGML_SYCL_SCRATCHPAD_SITE_%s, scratchpad_mem, scratchpad_md)) {\n"
            "            return false;\n        }\n" % site)


def note(site):
    return "        ggml_sycl_dnnl_note_engaged(GGML_SYCL_SCRATCHPAD_SITE_%s);\n" % site


HELPER_COND = "scratchpad_mem.get(true) == nullptr && scratchpad_md.get_size() > 0"
HELPER_HOOK = "if (ggml_sycl_scratchpad_site_hook(site)) {\n        return true;\n    }\n"


SM_ANCHOR = "        auto scratchpad_md = softmax_pd.scratchpad_desc();"


def gemm_mutants(files, edit):
    """Mutants of the DnnlGemmWrapper consumers (llama.cpp-23mk S3-4), the ggml-sycl.cpp pins and the common.hpp getter."""
    g, c, m, o = GEMM, COMMON, MAIN, OUTPROD
    gt = files[g]
    out = []
    # decisions: each consumer's, with a wrong result, a wrong site, a negation, and a lost note
    for name, site, result, rtype, want in GEMM_CONSUMERS:
        decl, body = member_body(strip_comments(gt), name)
        first = GEMM_DECISION.search(body)
        assert first is not None, "mutant anchor: %s's decision" % name
        text = first.group(0)
        assert gt.count(text) >= 1, "mutant anchor: %s's decision text not in gemm.hpp as written" % name
        out.append(("%s's decision returns the other result" % name,
                    edit(g, text, text.replace("return " + result, "return " + ("true" if result == "false" else "false")))))
        out.append(("%s's decision names another site" % name,
                    edit(g, text, text.replace("SITE_" + site, "SITE_DNNL_SOFTMAX"))))
        out.append(("%s's decision is negated" % name, edit(g, text, text.replace("if (ggml_", "if (!ggml_"))))
        out.append(("%s's decision is gone" % name, edit(g, text, "")))
        out.append(("%s loses [[nodiscard]]" % name,
                    edit(g, "[[nodiscard]] static %s %s(" % (rtype, name), "static %s %s(" % (rtype, name))))
        out.append(("%s throws on a decline again" % name,
                    edit(g, text, text.replace("return " + result, 'throw std::runtime_error("oneDNN scratchpad allocation failed")'))))
    for name, rtype in GEMM_FORWARDERS:
        out.append(("%s loses [[nodiscard]]" % name, edit(g, "[[nodiscard]] static %s %s(" % (rtype, name), "static %s %s(" % (rtype, name))))
    note_gemm = "ggml_sycl_dnnl_note_engaged(GGML_SYCL_SCRATCHPAD_SITE_DNNL_GEMM);"
    out.append(("gemm loses an engaged note", edit(g, "            " + note_gemm + "\n", "")))
    out.append(("gemm's note moves before query_only's exit",
                edit(g, "            if (query_only) {\n                return sycl::event{};\n            }\n            " + note_gemm,
                     "            " + note_gemm + "\n            if (query_only) {\n                return sycl::event{};\n            }")))
    out.append(("a get_scratchpad_mem request sits under an if",
                edit(g, "        auto scratchpad_mem = ctx.get_scratchpad_mem(cached->scratchpad_md, eng, q);\n        if (ggml_sycl_scratchpad_declined(GGML_SYCL_SCRATCHPAD_SITE_DNNL_WOQ_Q8_0",
                     "        dnnl::memory scratchpad_mem;\n        if (cached->scratchpad_md.get_size() > 0) scratchpad_mem = ctx.get_scratchpad_mem(cached->scratchpad_md, eng, q);\n        if (ggml_sycl_scratchpad_declined(GGML_SYCL_SCRATCHPAD_SITE_DNNL_WOQ_Q8_0")))
    out.append(("a ninth scratchpad request appears",
                edit(g, "    [[nodiscard]] static bool woq_gemm_q4_0(", "    static void extra(ggml_backend_sycl_context & ctx, dnnl::memory::desc d, dnnl::engine e, sycl::queue q) {\n        auto x = ctx.get_scratchpad_mem(d, e, q);\n    }\n    [[nodiscard]] static bool woq_gemm_q4_0(")))
    out.append(("the mxfp4 2-D fallback asks inside its batch loop",
                edit(g, "        for (int b = 0; b < batch_size; ++b) {",
                     "        for (int b = 0; b < batch_size; ++b) {\n            auto again = ctx.get_scratchpad_mem(cached2d->scratchpad_md, eng, q);", 1)))
    for dead in GEMM_DEAD:
        out.append(("%s comes back" % dead, edit(g, "    [[nodiscard]] static bool woq_gemm_q4_0(", "    static void %s() {}\n    [[nodiscard]] static bool woq_gemm_q4_0(" % dead)))
    out.append(("gemm.hpp throws the old scratchpad error", edit(g, "    [[nodiscard]] static bool woq_gemm_q4_0(", "    static void bad() { throw std::runtime_error(\"oneDNN scratchpad allocation failed\"); }\n    [[nodiscard]] static bool woq_gemm_q4_0(")))
    # common.hpp: the getter
    out.append(("get_scratchpad_mem loses [[nodiscard]]", edit(c, "[[nodiscard]] dnnl::memory get_scratchpad_mem(", "dnnl::memory get_scratchpad_mem(")))
    ct = files[c]
    zero = "        if (scratchpad_size == 0) {\n            return dnnl::memory();\n        }\n"
    lock = "        std::lock_guard<std::mutex> lock(dnnl_mutex);\n"
    assert ct.count(zero + lock) == 1, "mutant anchor: the getter's zero-size return and lock in " + c
    out.append(("the getter's zero-size return is gone", edit(c, zero, "")))
    out.append(("the getter locks before its zero-size return", edit(c, zero + lock, lock + zero)))
    for label, new_zero in (("the getter returns early for a nonzero size", zero.replace("== 0", "!= 0")),
                            ("the getter returns early for every size", zero.replace("scratchpad_size == 0", "true")),
                            ("the getter's early return is for size > 0", zero.replace("== 0", "> 0")),
                            ("the getter's early return is for size <= 1", zero.replace("== 0", "<= 1")),
                            ("the getter's early return builds a memory", zero.replace("dnnl::memory()", "dnnl::memory(scratchpad_md, eng)"))):
        out.append((label, edit(c, zero, new_zero)))
    out.append(("the getter never takes the lock", edit(c, zero + lock, zero)))
    out.append(("the getter takes the lock twice", edit(c, zero + lock, zero + lock + lock.replace("lock(", "lock2("))))
    # ggml-sycl.cpp / outprod.cpp: the declared next paths
    def edit_pin(rel, pin, new, count=1):
        text = files[rel]
        assert len(re.findall(ws_pattern(pin), text)) >= 1, "mutant anchor missing: %r in %s" % (pin, rel)
        return dict(files, **{rel: re.sub(ws_pattern(pin), lambda _: new, text, count=count)})
    for text, want in MAIN_PINS:
        out.append(("ggml-sycl.cpp loses `%s`" % text[:48], edit_pin(m, text, "/* gone */", want)))
    out.append(("the dense f16 arm swallows the decline", edit_pin(m, MAIN_PINS[0][0], "return; (void) std::runtime_error(")))
    out.append(("the old scratchpad error is back in ggml-sycl.cpp",
                edit_pin(m, MAIN_PINS[0][0], 'throw std::runtime_error("oneDNN scratchpad allocation failed"); (void) std::runtime_error(')))
    out.append(("the dense f16 throw survives only in a comment", edit_pin(m, MAIN_PINS[0][0], "// " + MAIN_PINS[0][0])))
    out.append(("out_prod loses its named throw", edit_pin(o, OUTPROD_PIN, "throw 1; // ")))
    return out


def mutants(files):
    """(label, mutated files) pairs, each of which a working gate must reject."""
    def edit(rel, old, new, count=1):
        text = files[rel]
        assert old in text, "mutant anchor missing: %r in %s" % (old, rel)
        return dict(files, **{rel: text.replace(old, new, count)})
    h = HDR
    ew, bb, sm = SYCL + "/element_wise.cpp", SYCL + "/binbcast.cpp", SYCL + "/softmax.cpp"
    sd, ed = decide("DNNL_SOFTMAX"), decide("DNNL_ELTWISE")
    sm_tail = "                stream)) {\n            return;\n        }\n    }\n#else"
    assert files[sm].count(sm_tail) == 1, "mutant anchor: the softmax caller's tail in " + sm
    for text in (sd, ed, decide("DNNL_BINARY_ROW"), note("DNNL_SOFTMAX"), HELPER_HOOK, HELPER_COND):
        assert files[h].count(text) == 1, "mutant anchor: expected exactly one %r in %s" % (text, h)
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
        ("the decision returns true", edit(h, sd, sd.replace("return false", "return true"))),
        ("the decision negates the helper", edit(h, sd, sd.replace("if (ggml_", "if (!ggml_"))),
        ("the softmax decision names the eltwise site", edit(h, sd, sd.replace("DNNL_SOFTMAX", "DNNL_ELTWISE"))),
        ("the helper drops its size guard", edit(h, HELPER_COND, "scratchpad_mem.get(true) == nullptr")),
        ("the helper inverts its test", edit(h, HELPER_COND, "scratchpad_mem.get(true) != nullptr && scratchpad_md.get_size() > 0")),
        ("the helper loses the seam hook", edit(h, HELPER_HOOK, "")),
        ("the helper's hook returns false", edit(h, HELPER_HOOK, HELPER_HOOK.replace("return true", "return false"))),
        ("the softmax wrapper loses its engaged note", edit(h, note("DNNL_SOFTMAX"), "")),
        ("the softmax wrapper notes the wrong site", edit(h, note("DNNL_SOFTMAX"), note("DNNL_ELTWISE"))),
        ("the softmax wrapper notes before it decides", edit(h, sd + note("DNNL_SOFTMAX"), note("DNNL_SOFTMAX") + sd)),
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
        ("the decision tests the wrong memory", edit(h, decide("DNNL_SOFTMAX"), decide("DNNL_SOFTMAX").replace("scratchpad_mem, scratchpad_md", "scratchpad_mem, src_md"))),
        ("the decision swaps the descriptor", edit(h, ed, ed.replace("scratchpad_mem, scratchpad_md", "scratchpad_mem, md"))),
        ("the eltwise note sits under if (0)", edit(h, note("DNNL_ELTWISE"), "        if (0) ggml_sycl_dnnl_note_engaged(GGML_SYCL_SCRATCHPAD_SITE_DNNL_ELTWISE);\n")),
        ("the softmax note is conditional on the scale", edit(h, note("DNNL_SOFTMAX"), "        if (scale != 1.0f) ggml_sycl_dnnl_note_engaged(GGML_SYCL_SCRATCHPAD_SITE_DNNL_SOFTMAX);\n")),
        ("eltwise writes through single_task before the decision", edit(h, "        auto scratchpad_md = eltwise_pd.scratchpad_desc();",
                                                                       "        float * o = (float *) dst; q->single_task([=]() { o[0] = 0.0f; });\n        auto scratchpad_md = eltwise_pd.scratchpad_desc();")),
        ("eltwise writes through an alias before the decision", edit(h, "        auto scratchpad_md = eltwise_pd.scratchpad_desc();",
                                                                    "        float * o = (float *) dst; o[0] = 0.0f;\n        auto scratchpad_md = eltwise_pd.scratchpad_desc();")),
        ("binary_broadcast_row copies before the decision", edit(h, "        auto scratchpad_md  = binary_pd.scratchpad_desc();",
                                                                 "        q->copy((const float *) src0, (float *) dst, 1);\n        auto scratchpad_md  = binary_pd.scratchpad_desc();")),
        ("an eltwise caller returns unconditionally after the if", edit(ew, "                stream)) {\n            return;\n        }\n    }\n#endif\n    // Fallback",
                                                                       "                stream)) {\n            return;\n        }\n        return;\n    }\n#endif\n    // Fallback")),
        ("the softmax caller returns unconditionally after its block", edit(sm, "                stream)) {\n            return;\n        }\n    }\n#else",
                                                                            "                stream)) {\n            return;\n        }\n    }\n    return;\n#else")),
        ("the softmax block is followed by a goto", edit(sm, sm_tail, sm_tail.replace("    }\n#else", "    }\n    goto done;\n#else"))),
        ("the softmax block is followed by a statement and a return", edit(sm, sm_tail, sm_tail.replace("    }\n#else", "    }\n    (void) 0;\n    return;\n#else"))),
        ("the softmax block is followed by a throw", edit(sm, sm_tail, sm_tail.replace("    }\n#else", "    }\n    throw 1;\n#else"))),
        ("the softmax block is followed by an abort", edit(sm, sm_tail, sm_tail.replace("    }\n#else", "    }\n    GGML_ABORT(\"x\");\n#else"))),
        ("the softmax block has an enclosing else that returns", edit(sm, sm_tail, sm_tail.replace("    }\n#else", "    } else {\n        return;\n    }\n#else"))),
        ("the softmax if has an else that returns", edit(sm, sm_tail, sm_tail.replace("        }\n    }\n#else", "        } else {\n            return;\n        }\n    }\n#else"))),
        ("the softmax #else arm returns", edit(sm, "#else\n    (void) use_dnnl_softmax;", "#else\n    (void) use_dnnl_softmax;\n    return;")),
        ("the softmax #else arm returns first", edit(sm, "#else\n    (void) use_dnnl_softmax;", "#else\n    return;\n    (void) use_dnnl_softmax;")),
        ("the binary #else arm throws", edit(bb, "#else\n    (void) use_dnnl_mul;", "#else\n    (void) use_dnnl_mul;\n    throw 1;")),
        ("the softmax caller returns before its if", edit(sm, "    if (use_dnnl_softmax &&", "    if (nrows_x == 0) {\n        return;\n    }\n    if (use_dnnl_softmax &&")),
        ("the softmax caller jumps before its if", edit(sm, "    if (use_dnnl_softmax &&", "    goto done;\n    if (use_dnnl_softmax &&")),
        ("the softmax #else arm calls _Exit", edit(sm, "#else\n    (void) use_dnnl_softmax;", "#else\n    (void) use_dnnl_softmax;\n    _Exit(1);")),
        ("the softmax #else arm calls quick_exit", edit(sm, "#else\n    (void) use_dnnl_softmax;", "#else\n    (void) use_dnnl_softmax;\n    quick_exit(1);")),
        ("the softmax #else arm calls std::terminate", edit(sm, "#else\n    (void) use_dnnl_softmax;", "#else\n    (void) use_dnnl_softmax;\n    std::terminate();")),
        ("the softmax #else arm calls longjmp", edit(sm, "#else\n    (void) use_dnnl_softmax;", "#else\n    (void) use_dnnl_softmax;\n    longjmp(0, 1);")),
        ("softmax writes src through a C-style cast before the decision", edit(h, SM_ANCHOR, "        ((float *) src)[0] = 0.0f;\n" + SM_ANCHOR)),
        ("softmax copies with std::copy_n before the decision", edit(h, SM_ANCHOR, "        std::copy_n((const float *) src, 1, (float *) src);\n" + SM_ANCHOR)),
        ("softmax fills with std::fill_n before the decision", edit(h, SM_ANCHOR, "        std::fill_n((float *) src, 1, 0.0f);\n" + SM_ANCHOR)),
        ("softmax moves with std::memmove before the decision", edit(h, SM_ANCHOR, "        std::memmove((void *) src, src, 4);\n" + SM_ANCHOR)),
        ("softmax transforms in place before the decision", edit(h, SM_ANCHOR, "        std::transform((float *) src, (float *) src, (float *) src, [](float x) { return 0.0f; });\n" + SM_ANCHOR)),
        ("softmax writes src through a cast before the decision", edit(h, "        auto scratchpad_md = softmax_pd.scratchpad_desc();",
                                                                      "        const_cast<float *>((const float *) src)[0] = 0.0f;\n        auto scratchpad_md = softmax_pd.scratchpad_desc();")),
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
    # dst renamed inside softmax and written under the new name before the decision: the no-dst rule must not be dodgeable
    i = files[h].index("[[nodiscard]] static bool softmax(")
    j = files[h].index("softmax_prim.execute", i)
    seg = re.sub(r"\bdst\b", "out", files[h][i:j]).replace(
        "        auto scratchpad_md = softmax_pd.scratchpad_desc();",
        "        ((float *) out)[0] = 0.0f;\n        auto scratchpad_md = softmax_pd.scratchpad_desc();", 1)
    assert "((float *) out)[0]" in seg, "mutant anchor missing: softmax scratchpad_md in " + h
    out += gemm_mutants(files, edit)
    out.append(("softmax renames dst and writes through the new name", dict(files, **{h: files[h][:i] + seg + files[h][j:]})))
    # the decision moved after the pre-scale: ask first, decide later (the double-application hazard)
    moved = files[h].replace(sd + note("DNNL_SOFTMAX"), "", 1)
    anchor = "        auto src_mem = dnnl::memory(src_md, eng, const_cast<void *>(softmax_src));"
    assert anchor in moved, "mutant anchor missing: softmax src_mem in " + h
    out.append(("softmax decides after the pre-scale pass",
                dict(files, **{h: moved.replace(anchor, sd + note("DNNL_SOFTMAX") + anchor, 1)})))
    return out


def load(root):
    files = {}
    for rel in (HDR, GEMM, COMMON, MAIN, OUTPROD) + tuple(SYCL + "/" + r for r in CALLERS):
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
