#!/usr/bin/env python3
"""resolve_fused_ops() tells a capability gap from a placement landing (llama.cpp-pmzl).

Host-only: reads src/llama-context.cpp, src/llama-fused-landing.h, src/llama-fused-resolution.h, ggml-sycl.cpp and
ggml-sycl.h, runs no build and loads no model. The classifier's behaviour is test-llama-fused-landing (fake devices and tensors); this gate pins the
wiring around it.

The defect: a fused op (flash attention, gated delta net) that the scheduler put on the CPU was always reported as
"executes on CPU ... -- the executor follows data placement, not a capability gap". That is true when a host-demoted
layer's KV lives in host memory, and FALSE when the layer's device has no kernel for the op (the qwen4exp
DSV4_HC_* / LIGHTNING_INDEXER case before llama.cpp-pmzl). The log read as reassurance exactly where it should have
read as a defect.

supports_op also declines for placement (host-demoted KV, planner-on-host weights), so "supports_op is false" cannot
tell a missing kernel from a placement, and the planner-on-host operand is a SYCL-owned buffer no host-buffer test
can see. The SYCL backend therefore exports a capability-only query and resolve_fused_ops asks it.

Pinned here:
  * the SYCL backend exports ggml_backend_sycl_supports_op_capability, registered as a reg proc and declared in
    ggml-sycl.h. supports_op and the query are the same function, ggml_sycl_device_supports_op_impl, differing
    only in its `placement_declines` parameter: supports_op passes true, the query passes false, and each of the
    four placement-predicate call sites (host-demoted KV x3, planner-on-host x1) is gated on the parameter, so
    the query is supports_op minus placement and the two cannot drift apart. The predicates themselves stay pure;
  * the classifier asks that query when the backend has one, and otherwise falls back to the device supporting the
    node or a persistent host-resident operand (a host buffer whose usage is not COMPUTE);
  * a landing the device could not have executed is counted apart from a placement landing, and each gets its own
    warning: the gap one carries the gap count, the layer and the device name. resolve_fused_ops logs nothing (zhcn:
    it records into a fused_resolution_entry_data and fused_resolution_render prints the record at the publish), so
    the counters, layer and device are pinned where resolve_fused_ops writes the entry and the warnings where the
    renderer builds them (the English is not pinned, the WARN level, the counter and the arguments are);
  * the enable/disable decision is untouched: a CPU landing never disables the op, a landing on a non-CPU device
    still does.

Run with --self-test to prove every check fires against a mutant of the thing it forbids.
"""
import argparse
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--context", default=str(root / "src/llama-context.cpp"))
parser.add_argument("--landing-header", default=str(root / "src/llama-fused-landing.h"))
parser.add_argument("--resolution-header", default=str(root / "src/llama-fused-resolution.h"))
parser.add_argument("--backend", default=str(root / "ggml/src/ggml-sycl/ggml-sycl.cpp"))
parser.add_argument("--sycl-header", default=str(root / "ggml/include/ggml-sycl.h"))
parser.add_argument("--self-test", action="store_true")
args = parser.parse_args()


def is_digit_separator(source, i):
    """True when the `'` at `i` is a C++14 digit separator (`1'000`, `0xFF'FF`), not the start of a char literal."""
    if i + 1 >= len(source) or not (source[i + 1].isalnum() and source[i - 1].isalnum() if i > 0 else False):
        return False
    start = i
    while start > 0 and (source[start - 1].isalnum() or source[start - 1] in "_'."):
        start -= 1
    return source[start].isdigit()


def strip_comments(source, blank_literals=False, keep_length=False):
    """Remove C/C++ comments, keeping string literals and the line count.

    blank_literals: replace the contents of string and char literals with spaces (the quotes stay).
    keep_length:    replace comments with spaces instead of removing them, so offsets still mean something.
    Digit separators (`1'000`) are not char literals.
    """
    out = []
    i = 0
    n = len(source)
    while i < n:
        ch = source[i]
        if ch in ('"', "'") and not (ch == "'" and is_digit_separator(source, i)):
            quote = ch
            out.append(ch)
            i += 1
            while i < n:
                if source[i] == "\\" and i + 1 < n:
                    out.append("  " if blank_literals else source[i:i + 2])
                    i += 2
                    continue
                if source[i] == quote:
                    out.append(quote)
                    i += 1
                    break
                out.append(" " if blank_literals and source[i] != "\n" else source[i])
                i += 1
            continue
        if source.startswith("//", i):
            while i < n and source[i] != "\n":
                out.append(" " if keep_length else "")
                i += 1
            continue
        if source.startswith("/*", i):
            end = source.find("*/", i + 2)
            end = n if end < 0 else end + 2
            if keep_length:
                out.append("".join(c if c == "\n" else " " for c in source[i:end]))
            else:
                out.append("\n" * source.count("\n", i, end))
            i = end
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def function_body(source, signature_regex):
    """Return the brace-balanced body of the first function matching the regex."""
    m = re.search(signature_regex, source)
    if not m:
        return None
    start = source.find("{", m.end() - 1 if source[m.end() - 1] == "{" else m.end())
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start:i + 1]
    return None


def squash(text):
    return re.sub(r"\s+", " ", text or "")


def matching(text, open_at, open_ch, close_ch):
    """Index of the bracket closing the one at `open_at`, or -1."""
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == open_ch:
            depth += 1
        elif text[i] == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return -1


def gate_condition_ok(cond):
    """`placement_declines && ...` with no `||` at the top level, so nothing in it is reachable ungated."""
    if not re.match(r"\s*placement_declines\s*&&", cond):
        return False
    depth = 0
    for i, ch in enumerate(cond):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and cond.startswith("||", i):
            return False
    return True


def statement_end(clean, start):
    """Offset of the last character of the statement that begins at `start`, or -1.

    Understood: a braced block; `if`/`for`/`while`/`switch (...)` followed by a statement, an `if`'s `else` branch
    included (so `if (a) b; else { c; }` ends at the final `}`); `do ... while (...);`; and any other statement,
    which ends at the first `;` outside every parenthesis, bracket and brace (so a `;` inside a lambda body, bound
    by `=` or passed as an argument, does not end it).
    """
    m = re.compile(r"\s*").match(clean, start)
    i = m.end()
    if i >= len(clean):
        return -1
    if clean[i] == "{":
        return matching(clean, i, "{", "}")
    kw = re.compile(r"(if|for|while|switch)\s*\(").match(clean, i)
    if kw:
        close = matching(clean, kw.end() - 1, "(", ")")
        end = statement_end(clean, close + 1) if close >= 0 else -1
        if kw.group(1) == "if" and end >= 0:
            other = re.compile(r"\s*else\b").match(clean, end + 1)
            if other:
                return statement_end(clean, other.end())
        return end
    if re.compile(r"do\b").match(clean, i):
        end = statement_end(clean, i + 2)
        tail = re.compile(r"\s*while\s*\(").match(clean, end + 1) if end >= 0 else None
        if tail:
            close = matching(clean, tail.end() - 1, "(", ")")
            return clean.find(";", close) if close >= 0 else -1
        return end
    depth = 0
    while i < len(clean):
        c = clean[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == ";" and depth <= 0:
            return i
        i += 1
    return -1


def ungated_sites(body, site):
    """Offsets of `site` matches that no `if (placement_declines && ...)` covers.

    This is a source tripwire, not a C++ parser; the behavioural coverage of fused landing is
    test-llama-fused-landing. What it does:
    * comments and the contents of string and char literals are blanked first, and a digit separator (`1'000`) is
      not a char literal;
    * a site is covered when it sits in the condition of an `if` whose condition is `placement_declines && ...`
      with no top-level `||`, or in that if's body: a braced block, or a single statement as statement_end() reads
      it; a second `placement_declines &&` on a nested site would be dead code and is not required;
    * a site under no such if is reported.
    What it does not understand: raw string literals, `#if 0` regions (parsed as code), a comma or ternary at the top
    level of a condition (`placement_declines && a, true`, `c ? x : y`, read as gated), a `switch` body's labels, and
    the `else` branch of a gated BRACED `if` (reported, which fails closed), and a gated `try { } catch (...) { }`
    (the statement ends at the try block, so an ungated site after the catch is swallowed: fails open).
    """
    clean = strip_comments(body, blank_literals=True, keep_length=True)
    spans = []
    for m in re.finditer(r"\bif\s*\(", clean):
        open_paren = m.end() - 1
        close_paren = matching(clean, open_paren, "(", ")")
        if close_paren < 0 or not gate_condition_ok(clean[open_paren + 1:close_paren]):
            continue
        rest = clean[close_paren + 1:]
        body_start = close_paren + 1 + len(rest) - len(rest.lstrip())
        if clean.startswith("{", body_start):
            body_end = matching(clean, body_start, "{", "}")
        else:
            body_end = statement_end(clean, body_start)
        spans.append((open_paren, body_end if body_end >= 0 else len(clean)))
    return [m.start() for m in re.finditer(site, clean)
            if not any(lo <= m.start() <= hi for lo, hi in spans)]


def evaluate(ctx, hdr, be, sh, res):
    classifier = function_body(
        hdr, r"static\s+inline\s+bool\s+llama_fused_cpu_landing_is_placement\s*\([^)]*\)\s*\{") or ""
    resolve = function_body(ctx, r"void\s+llama_context::resolve_fused_ops\s*\([^)]*\)\s*\{") or ""
    cap_proc = function_body(
        ctx, r"static\s+llama_fused_capability_fn\s+llama_context_sycl_capability_proc\s*\([^)]*\)\s*\{") or ""
    export = function_body(
        be, r"\bbool\s+ggml_backend_sycl_supports_op_capability\s*\([^)]*\)\s*\{") or ""
    wrapper = function_body(
        be, r"static\s+bool\s+ggml_backend_sycl_device_supports_op\s*\([^)]*\)\s*\{") or ""
    impl = function_body(
        be, r"static\s+bool\s+ggml_sycl_device_supports_op_impl\s*\([^)]*\bbool\s+placement_declines\s*\)\s*\{") or ""
    kv_host = function_body(
        be, r"static\s+bool\s+ggml_sycl_tensor_is_in_kv_host_buft\s*\([^)]*\)\s*\{") or ""
    planned = function_body(
        be, r"static\s+bool\s+ggml_sycl_op_is_planned_on_host\s*\([^)]*\)\s*\{") or ""
    render = function_body(
        res, r"inline\s+std::vector<fused_resolution_line>\s+fused_resolution_render\s*\([^)]*\)\s*\{") or ""
    flat_resolve = squash(resolve)
    flat_classifier = squash(classifier)
    flat_render = squash(render)

    checks = {}
    # --- the backend side ---------------------------------------------------------------------------------------
    checks["backend: supports_op and the capability query share one impl taking placement_declines"] = bool(impl)
    checks["backend: supports_op keeps its placement declines"] = bool(
        re.search(r"return\s+ggml_sycl_device_supports_op_impl\s*\(\s*dev\s*,\s*op\s*,\s*true\s*\)", wrapper))
    checks["backend: the capability query drops them"] = bool(
        re.search(r"return\s+ggml_sycl_device_supports_op_impl\s*\(\s*dev\s*,\s*op\s*,\s*false\s*\)", export))
    kv_sites = re.findall(r"ggml_sycl_tensor_is_in_kv_host_buft\s*\(", impl)
    checks["backend: every KV-host placement site in the impl is gated on the parameter"] = \
        bool(kv_sites) and not ungated_sites(impl, r"ggml_sycl_tensor_is_in_kv_host_buft\s*\(")
    pl_sites = re.findall(r"ggml_sycl_op_is_planned_on_host\s*\(", impl)
    checks["backend: every planner-on-host site in the impl is gated on the parameter"] = \
        bool(pl_sites) and not ungated_sites(impl, r"ggml_sycl_op_is_planned_on_host\s*\(")
    checks["backend: the placement predicates take no flag (they stay pure)"] = bool(kv_host) and bool(planned) and not re.search(
        r"placement_declines|thread_local|capability_only", kv_host + planned)
    checks["backend: no thread-local capability state"] = not re.search(r"thread_local[^;]*capability", be)
    checks["backend: the query is registered as a reg proc"] = bool(
        re.search(r"strcmp\s*\(\s*name\s*,\s*\"ggml_backend_sycl_supports_op_capability\"\s*\)\s*==\s*0\s*\)\s*\{\s*return\s*\(void\s*\*\)\s*ggml_backend_sycl_supports_op_capability",
                  squash(be)))
    checks["backend: the query is declared in ggml-sycl.h"] = bool(
        re.search(r"GGML_BACKEND_API\s+bool\s+ggml_backend_sycl_supports_op_capability\s*\(", sh))

    # --- the classifier ----------------------------------------------------------------------------------------
    checks["classifier exists"] = bool(classifier)
    checks["classifier asks the capability query when the backend has one"] = bool(
        re.search(r"if\s*\(\s*capability\s*\)\s*\{\s*return\s+capability\s*\(\s*dev_layer\s*,\s*node\s*\)\s*;", flat_classifier))
    checks["classifier fallback asks the layer device supports_op"] = bool(
        re.search(r"ggml_backend_dev_supports_op\s*\(\s*dev_layer\s*,\s*node\s*\)", classifier))
    checks["classifier fallback treats a host operand as placement"] = bool(
        re.search(r"ggml_backend_buffer_is_host\s*\(\s*data->buffer\s*\)", classifier))
    checks["classifier looks through views"] = "src->view_src" in classifier
    checks["classifier ignores compute scratch buffers"] = bool(
        re.search(r"ggml_backend_buffer_get_usage\s*\(\s*data->buffer\s*\)\s*!=\s*GGML_BACKEND_BUFFER_USAGE_COMPUTE",
                  flat_classifier))
    checks["classifier answers capability gap with false"] = bool(re.search(r"return\s+false\s*;\s*\}\s*$", classifier))

    # --- resolve_fused_ops --------------------------------------------------------------------------------------
    checks["context includes the classifier header"] = bool(re.search(r'#include\s+"llama-fused-landing\.h"', ctx))
    checks["context looks the capability query up by its exported name"] = (
        bool(re.search(r"llama_context_dev_is_sycl\s*\(\s*dev\s*\)\s*\?\s*&ggml_backend_sycl_supports_op_capability", squash(cap_proc))) and
        '"ggml_backend_sycl_supports_op_capability"' in cap_proc)
    checks["resolve_fused_ops looks the query up once per layer device"] = bool(
        re.search(r"std::map\s*<\s*ggml_backend_dev_t\s*,\s*llama_fused_capability_fn\s*>\s*capability_procs\s*;.*?"
                  r"capability_procs\s*\.\s*emplace\s*\(\s*device_layer\s*,\s*llama_context_sycl_capability_proc\s*\(\s*device_layer\s*\)\s*\)",
                  flat_resolve))
    checks["resolve_fused_ops hands the classifier the cached query"] = bool(
        re.search(r"llama_fused_cpu_landing_is_placement\s*\(\s*device_layer\s*,\s*node\.tensor\s*,\s*capability\s*->\s*second\s*\)",
                  flat_resolve))
    # The English is not pinned, the shape is: resolve_fused_ops counts each kind of landing into its own entry field,
    # and the renderer gives each its own WARN line, counting only its own landings.
    split = re.search(
        r"if\s*\(\s*llama_fused_cpu_landing_is_placement\s*\([^;{}]*\)\s*\)\s*\{\s*"
        r"entry\.n_cpu_landings\s*\+\+\s*;\s*entry\.cpu_landing_layer\s*=\s*node\.il\s*;[^{}]*\}\s*"
        r"else\s*\{\s*entry\.n_cpu_gaps\s*\+\+\s*;\s*entry\.cpu_gap_layer\s*=\s*node\.il\s*;\s*"
        r"cpu_gap_dev\s*=\s*device_layer\s*;\s*\}", flat_resolve)
    gap_dev = re.search(
        r"entry\.cpu_gap_dev\s*=\s*cpu_gap_dev\s*\?\s*ggml_backend_dev_name\s*\(\s*cpu_gap_dev\s*\)", flat_resolve)
    gap_warn = re.search(
        r"if\s*\(\s*e\.n_cpu_gaps\s*>\s*0\s*\)\s*\{\s*lines\.push_back\s*\(\s*\{\s*FUSED_RESOLUTION_LEVEL_WARN\s*,"
        r"([^;]*)\}\s*\)\s*;\s*\}", flat_render)
    gap_call = gap_warn.group(1) if gap_warn else ""
    checks["capability warning counts the gaps, names the layer and the device"] = bool(
        split and gap_dev and gap_warn and
        re.search(r"std::to_string\s*\(\s*e\.n_cpu_gaps\s*\)", gap_call) and
        re.search(r"std::to_string\s*\(\s*e\.cpu_gap_layer\s*\)", gap_call) and
        re.search(r"\be\.cpu_gap_dev\b", gap_call) and re.search(r"\be\.probe_name\b", gap_call))
    place_warn = re.search(
        r"if\s*\(\s*e\.n_cpu_landings\s*>\s*0\s*\)\s*\{\s*lines\.push_back\s*\(\s*\{\s*FUSED_RESOLUTION_LEVEL_WARN\s*,"
        r"([^;]*)\}\s*\)\s*;\s*\}", flat_render)
    place_call = place_warn.group(1) if place_warn else ""
    checks["placement warning counts the placement landings and names the layer"] = bool(
        split and place_warn and
        re.search(r"std::to_string\s*\(\s*e\.n_cpu_landings\s*\)", place_call) and
        re.search(r"std::to_string\s*\(\s*e\.cpu_landing_layer\s*\)", place_call) and "n_cpu_gaps" not in place_call)
    checks["the two warnings do not share a counter"] = "n_cpu_gaps" not in place_call and "n_cpu_landings" not in gap_call
    disabled_at = re.search(r"if\s*\(\s*!\s*e\.enabled\s*\)\s*\{", flat_render)
    disabled_end = matching(flat_render, disabled_at.end() - 1, "{", "}") if disabled_at else -1
    disabled = flat_render[disabled_at.end():disabled_end] if disabled_end > 0 else ""
    checks["a non-CPU mismatch still disables the op"] = bool(
        re.search(r"entry\.mismatch\s*=\s*true", flat_resolve) and
        re.search(r"entry\.enabled\s*=\s*!\s*entry\.mismatch\s*;", flat_resolve) and
        "FUSED_RESOLUTION_LEVEL_WARN" in disabled and re.search(r"return\s+lines\s*;", disabled))
    checks["a CPU landing still leaves the op enabled"] = bool(
        re.search(r"entry\.enabled\s*=\s*!\s*entry\.mismatch\s*;", flat_resolve) and
        gap_warn and place_warn and "return" not in gap_warn.group(0) + place_warn.group(0))
    return checks


NAMES = ("ctx", "hdr", "be", "sh", "res")


def run(label, srcs, expect_fail=None):
    results = evaluate(*srcs)
    failed = sorted(k for k, v in results.items() if not v)
    if expect_fail is None:
        for k in results:
            print(("PASS: " if results[k] else "FAIL: ") + k)
        return failed
    fired = expect_fail in failed
    print(("PASS" if fired else "FAIL") + f": mutant '{label}' fires '{expect_fail}'")
    return [] if fired else [label]


def load(path, strip=True):
    p = Path(path)
    if not p.exists():
        return ""
    return strip_comments(p.read_text()) if strip else p.read_text()


sources = {
    "ctx": load(args.context),
    "hdr": load(args.landing_header),
    "be": load(args.backend),
    "sh": load(args.sycl_header),
    "res": load(args.resolution_header),
}
failed = run("tree", tuple(sources[k] for k in NAMES))

if args.self_test:
    def mutate(text, old, new):
        k = text.find(old)
        if k < 0:
            print(f"FAIL: self-test anchor missing: {old!r}")
            failed.append("self-test anchor " + old)
            return text
        return text[:k] + new + text[k + len(old):]

    def with_(**kw):
        return tuple(kw.get(k, sources[k]) for k in NAMES)

    mutants = [
        ("supports_op stops declining for placement", "backend: supports_op keeps its placement declines",
         with_(be=mutate(sources["be"], "return ggml_sycl_device_supports_op_impl(dev, op, true);", "return ggml_sycl_device_supports_op_impl(dev, op, false);"))),
        ("capability query keeps the declines", "backend: the capability query drops them",
         with_(be=mutate(sources["be"], "return ggml_sycl_device_supports_op_impl(dev, op, false);\n}", "return ggml_sycl_device_supports_op_impl(dev, op, true);\n}"))),
        ("impl loses the parameter", "backend: supports_op and the capability query share one impl taking placement_declines",
         with_(be=mutate(sources["be"], "const ggml_tensor * op, bool placement_declines) {", "const ggml_tensor * op, bool placement_declines_x) {"))),
        ("KV-host dst site ungated", "backend: every KV-host placement site in the impl is gated on the parameter",
         with_(be=mutate(sources["be"], "if (placement_declines && ggml_sycl_tensor_is_in_kv_host_buft(op)) {", "if (ggml_sycl_tensor_is_in_kv_host_buft(op)) {"))),
        ("KV-host site ungated", "backend: every KV-host placement site in the impl is gated on the parameter",
         with_(be=mutate(sources["be"], "if (placement_declines && ggml_sycl_tensor_is_in_kv_host_buft(op->src[i])) {", "if (ggml_sycl_tensor_is_in_kv_host_buft(op->src[i])) {"))),
        ("planner site ungated", "backend: every planner-on-host site in the impl is gated on the parameter",
         with_(be=mutate(sources["be"], "if (placement_declines && !is_multi_gpu_router_logits && ggml_sycl_op_is_planned_on_host(op, device)) {", "if (!is_multi_gpu_router_logits && ggml_sycl_op_is_planned_on_host(op, device)) {"))),
        ("KV-host predicate takes a flag", "backend: the placement predicates take no flag (they stay pure)",
         with_(be=mutate(sources["be"], "static bool ggml_sycl_tensor_is_in_kv_host_buft(const ggml_tensor * t) {\n    if (!t) {", "static bool ggml_sycl_tensor_is_in_kv_host_buft(const ggml_tensor * t) {\n    if (!t || placement_declines) {"))),
        ("thread-local capability state returns", "backend: no thread-local capability state",
         with_(be=sources["be"] + "\nstatic thread_local bool g_sycl_capability_flag = false;\n")),
        ("proc not registered", "backend: the query is registered as a reg proc",
         with_(be=mutate(sources["be"], 'strcmp(name, "ggml_backend_sycl_supports_op_capability") == 0', 'strcmp(name, "ggml_backend_sycl_supports_op_capability_x") == 0'))),
        ("query undeclared", "backend: the query is declared in ggml-sycl.h",
         with_(sh=mutate(sources["sh"], "ggml_backend_sycl_supports_op_capability(", "ggml_backend_sycl_supports_op_capability_x("))),
        ("capability query ignored by the classifier", "classifier asks the capability query when the backend has one",
         with_(hdr=mutate(sources["hdr"], "if (capability) {", "if (false) {"))),
        ("fallback loses supports_op", "classifier fallback asks the layer device supports_op",
         with_(hdr=mutate(sources["hdr"], "ggml_backend_dev_supports_op(dev_layer, node)", "false"))),
        ("fallback host operand ignored", "classifier fallback treats a host operand as placement",
         with_(hdr=mutate(sources["hdr"], "ggml_backend_buffer_is_host(data->buffer)", "false"))),
        ("views not followed", "classifier looks through views",
         with_(hdr=mutate(sources["hdr"], "src->view_src ? src->view_src : src", "src"))),
        ("compute scratch counted as placement", "classifier ignores compute scratch buffers",
         with_(hdr=mutate(sources["hdr"], "ggml_backend_buffer_get_usage(data->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE", "true"))),
        ("classifier never says gap", "classifier answers capability gap with false",
         with_(hdr=mutate(sources["hdr"], "    return false;\n}", "    return true;\n}"))),
        ("classifier header not included", "context includes the classifier header",
         with_(ctx=mutate(sources["ctx"], '#include "llama-fused-landing.h"', ""))),
        ("capability proc not looked up", "context looks the capability query up by its exported name",
         with_(ctx=mutate(sources["ctx"], "&ggml_backend_sycl_supports_op_capability", "nullptr"))),
        ("capability proc looked up per node", "resolve_fused_ops looks the query up once per layer device",
         with_(ctx=mutate(sources["ctx"], "capability_procs.emplace(device_layer, llama_context_sycl_capability_proc(device_layer))", "capability_procs.emplace(device_layer, llama_context_sycl_capability_proc(nullptr))"))),
        ("capability proc not passed", "resolve_fused_ops hands the classifier the cached query",
         with_(ctx=mutate(sources["ctx"], "node.tensor, capability->second)", "node.tensor, nullptr)"))),
        ("capability warning loses the device", "capability warning counts the gaps, names the layer and the device",
         with_(res=mutate(sources["res"], "e.cpu_gap_dev + \" does not support it\"", "\" does not support it\""))),
        ("capability warning counts placement landings", "capability warning counts the gaps, names the layer and the device",
         with_(res=mutate(sources["res"], "std::to_string(e.n_cpu_gaps)", "std::to_string(e.n_cpu_landings)"))),
        ("capability line demoted", "capability warning counts the gaps, names the layer and the device",
         with_(res=mutate(sources["res"], "if (e.n_cpu_gaps > 0) {\n        lines.push_back({ FUSED_RESOLUTION_LEVEL_WARN,", "if (e.n_cpu_gaps > 0) {\n        lines.push_back({ FUSED_RESOLUTION_LEVEL_INFO,"))),
        ("a gap counted as a placement landing", "capability warning counts the gaps, names the layer and the device",
         with_(ctx=mutate(sources["ctx"], "entry.n_cpu_gaps++;", "entry.n_cpu_landings++;"))),
        ("the gap device not recorded", "capability warning counts the gaps, names the layer and the device",
         with_(ctx=mutate(sources["ctx"], "entry.cpu_gap_dev     = cpu_gap_dev ? ggml_backend_dev_name(cpu_gap_dev)", "entry.cpu_gap_dev     = cpu_gap_dev ? \"?\""))),
        ("placement warning dropped", "placement warning counts the placement landings and names the layer",
         with_(res=mutate(sources["res"], "if (e.n_cpu_landings > 0) {", "if (e.n_cpu_landings > 1000000) {"))),
        ("placement warning counts the gaps", "the two warnings do not share a counter",
         with_(res=mutate(sources["res"], "std::to_string(e.n_cpu_landings)", "std::to_string(e.n_cpu_gaps)"))),
        ("mismatch no longer disables", "a non-CPU mismatch still disables the op",
         with_(ctx=mutate(sources["ctx"], "entry.enabled         = !entry.mismatch;", "entry.enabled         = true;"))),
        ("CPU landing disables the op", "a CPU landing still leaves the op enabled",
         with_(ctx=mutate(sources["ctx"], "entry.enabled         = !entry.mismatch;", "entry.enabled         = !entry.mismatch && entry.n_cpu_gaps == 0;"))),
        ("a gap ends the rendering before enabled", "a CPU landing still leaves the op enabled",
         with_(res=mutate(sources["res"], "e.cpu_gap_dev + \" does not support it\" });\n    }", "e.cpu_gap_dev + \" does not support it\" });\n        return lines;\n    }"))),
    ]
    for label, expect, srcs in mutants:
        failed += run(label, srcs, expect)

    # ungated_sites() is exercised on the extracted impl text itself, with brace-balanced inserts, so each case
    # reaches the parser instead of tripping the function-extraction anchors first.
    IMPL_SIGNATURE = (r"static\s+bool\s+ggml_sycl_device_supports_op_impl\s*"
                      r"\([^)]*\bbool\s+placement_declines\s*\)\s*\{")
    KV_SITE = r"ggml_sycl_tensor_is_in_kv_host_buft\s*\("
    KV_SRC_GATE = "if (placement_declines && ggml_sycl_tensor_is_in_kv_host_buft(op->src[i])) {"
    KV_SRC_UNGATED = "if (ggml_sycl_tensor_is_in_kv_host_buft(op->src[i])) {"
    impl_text = function_body(sources["be"], IMPL_SIGNATURE) or ""

    def parse_case(label, mutated_impl, flagged):
        sites = ungated_sites(mutated_impl, KV_SITE)
        ok = bool(sites) == flagged and bool(mutated_impl)
        verb = "flags" if flagged else "accepts"
        print(("PASS" if ok else "FAIL") + f": ungated_sites {verb} '{label}'")
        return [] if ok else ["ungated_sites " + label]

    def edited(old, new):
        return mutate(impl_text, old, new)

    failed += parse_case("the unmodified impl", impl_text, False)
    failed += parse_case("a gate removed from the src loop", edited(KV_SRC_GATE, KV_SRC_UNGATED), True)
    failed += parse_case("gate text in a comment over an ungated site",
                         edited(KV_SRC_GATE, "/* " + KV_SRC_GATE + " */ " + KV_SRC_UNGATED), True)
    failed += parse_case("gate text in a string over an ungated site",
                         edited(KV_SRC_GATE, 'const char * g = "' + KV_SRC_GATE + '"; ' + KV_SRC_UNGATED), True)
    failed += parse_case("a gate condition with a top-level ||",
                         edited(KV_SRC_GATE, "if (placement_declines && ggml_sycl_tensor_is_in_kv_host_buft(op->src[i])"
                                             " || g_ggml_sycl_debug) {"), True)
    failed += parse_case("a } in a comment ahead of the nested site",
                         edited(KV_SRC_GATE, KV_SRC_GATE + " /* } */"), False)
    failed += parse_case("a } in a string inside the gated body, ahead of the nested site",
                         edited(KV_SRC_GATE, KV_SRC_GATE + ' const char * s = "}";'), False)
    failed += parse_case("a digit separator inside the gated body",
                         edited(KV_SRC_GATE, KV_SRC_GATE + " int a = 1'000; int b = 2'000;"), False)

    # Snippets, for the shapes the tree does not contain. Each "flags" case has a site after the gated statement.
    SITE = "ggml_sycl_tensor_is_in_kv_host_buft(x)"
    failed += parse_case("a one-statement gated body holding a for (;;)",
                         "if (placement_declines && a) for (int i = 0; i < 3; ++i) use(" + SITE + ");", False)
    failed += parse_case("the same site under no gate", "for (int i = 0; i < 3; ++i) use(" + SITE + ");", True)
    failed += parse_case("a { in a string in a gated block, then an ungated site",
                         'if (placement_declines && a) { const char * s = "{"; } use(' + SITE + ");", True)
    failed += parse_case("an ungated site after a gated block, between two digit separators",
                         "if (placement_declines && a) { int a = 1'0; } use(" + SITE + "); int b = 2'0;", True)
    failed += parse_case("a gated for-block, then an ungated site",
                         "if (placement_declines && a) for (;;) { y(); } use(" + SITE + ");", True)
    failed += parse_case("a gated while-block, then an ungated site",
                         "if (placement_declines && a) while (b) { y(); } use(" + SITE + ");", True)
    failed += parse_case("a gated if/else, then an ungated site",
                         "if (placement_declines && a) if (b) { y(); } else { z(); } use(" + SITE + ");", True)
    failed += parse_case("a gated else-if chain, then an ungated site",
                         "if (placement_declines && a) if (b) y(); else if (c) { z(); } use(" + SITE + ");", True)
    failed += parse_case("a gated do/while, then an ungated site",
                         "if (placement_declines && a) do { y(); } while (b); use(" + SITE + ");", True)
    failed += parse_case("a site in the condition of a gated unbraced do/while",
                         "if (placement_declines && a) do y(); while (use(" + SITE + "));", False)
    failed += parse_case("a site in the else-if of a gated one-statement if",
                         "if (placement_declines && a) if (b) y(); else if (c) use(" + SITE + ");", False)
    failed += parse_case("a site inside a gated for-block",
                         "if (placement_declines && a) for (;;) { use(" + SITE + "); }", False)
    failed += parse_case("a site in the else of a gated one-statement if",
                         "if (placement_declines && a) if (b) y(); else use(" + SITE + ");", False)
    failed += parse_case("a site in a lambda bound in a gated statement (a `;` inside the braces)",
                         "if (placement_declines && a) f = [&] { y(); use(" + SITE + "); };", False)
    failed += parse_case("a site in a lambda passed in a gated statement (braces inside parentheses)",
                         "if (placement_declines && a) run([&] { y(); use(" + SITE + "); });", False)

if failed:
    print("\nFAILED: " + ", ".join(failed))
    sys.exit(1)
print("\nALL CHECKS PASSED")
