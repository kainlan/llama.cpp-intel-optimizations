#!/usr/bin/env python3
"""Source gate: a resolver call in cpu-dispatch.cpp runs inside a host-executor region.

The GW7 counter (resolver host returns of staged sources) counts only while a
thread is dispatching a device graph. cpu-dispatch.cpp functions run on the
host as executors inside that dispatch, and their resolver calls return host
pointers by design; they must open ggml_sycl_host_executor_region before the
first such call or the counter reads host execution as device reads.

The gate walks every top-level function of cpu-dispatch.cpp, and for each one
that calls a resolver requires the region declaration earlier in the same
function. It also checks the named host executors that live elsewhere
(EXTERNAL_EXECUTORS), located by their column-0 signature.

Limits, deliberately: the resolver names are a fixed list, so an unlisted
wrapper would pass; and a declaration in a closed inner scope still satisfies
the "earlier in the function" test. Extend RESOLVER_CALL when a resolver entry
point is added, and EXTERNAL_EXECUTORS when a host executor is.
argv: [cpu-dispatch.cpp [ggml-sycl.cpp]]
"""
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SOURCE = os.path.join(REPO, "ggml", "src", "ggml-sycl", "cpu-dispatch.cpp")

# The inline resolvers of common.hpp that can reach the counted host branches.
RESOLVER_CALL = re.compile(
    r"\b(ggml_sycl_get_data_ptr|ggml_sycl_resolve_tensor_ptr|ggml_sycl_resolve_or_host_tensor_ptr|"
    r"ggml_sycl_get_layout_ptr|ggml_sycl_resolve|ggml_sycl_resolve_no_materialize)\s*\(")
# Host executors outside cpu-dispatch.cpp: (file relative to the sycl dir, signature prefix).
EXTERNAL_EXECUTORS = (
    ("ggml-sycl.cpp", "bool ggml_sycl_cpu_fallback_graph("),
)
REGION_DECL = re.compile(r"^\s*ggml_sycl_host_executor_region\s+\w+\s*;")


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def top_level_functions(lines):
    """Yield (name_line_no, body_start, body_end) for each brace-balanced top-level block."""
    depth = 0
    start = None
    for i, line in enumerate(lines):
        for ch in line:
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    yield start, i
                    start = None


def violations(source_text):
    lines = strip_comments(source_text).split("\n")
    out = []
    for start, end in top_level_functions(lines):
        region_line = None
        for i in range(start, end + 1):
            if region_line is None and REGION_DECL.match(lines[i]):
                region_line = i
            m = RESOLVER_CALL.search(lines[i])
            if m and (region_line is None or region_line > i):
                out.append((i + 1, m.group(1), start + 1))
    return out


def named_function_violations(source_text, signature):
    """(violations, found) for the function whose column-0 line starts with `signature`."""
    lines = strip_comments(source_text).split("\n")
    start = next((i for i, l in enumerate(lines) if l.startswith(signature)), None)
    if start is None:
        return [], False
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("}")), len(lines) - 1)
    region_line = None
    out = []
    for i in range(start, end + 1):
        if region_line is None and REGION_DECL.match(lines[i]):
            region_line = i
        m = RESOLVER_CALL.search(lines[i])
        if m and (region_line is None or region_line > i):
            out.append((i + 1, m.group(1), start + 1))
    return out, True


def check_external_executors(sycl_dir, overrides):
    """Return an exit status contribution; prints its own FAIL lines."""
    status = 0
    for rel, signature in EXTERNAL_EXECUTORS:
        path = overrides.get(rel) or os.path.join(sycl_dir, rel)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        bad, found = named_function_violations(text, signature)
        if not found:
            print("FAIL: %s: host executor `%s` not found; the gate would pass vacuously" % (path, signature))
            status = 1
            continue
        for line_no, name, fn_line in bad:
            print("FAIL: %s:%d: %s called outside a ggml_sycl_host_executor_region in `%s`" %
                  (path, line_no, name, signature))
            status = 1
        # Mutant: without its region the executor must be reported.
        lines = text.split("\n")
        start = next(i for i, l in enumerate(lines) if l.startswith(signature))
        end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("}")), len(lines) - 1)
        decls = [i for i in range(start, end + 1) if REGION_DECL.match(strip_comments(lines[i]))]
        if not decls:
            print("FAIL: %s: `%s` carries no region declaration" % (path, signature))
            status = 1
            continue
        mutated = "\n".join(lines[:decls[0]] + lines[decls[0] + 1:])
        if not named_function_violations(mutated, signature)[0]:
            print("FAIL: removing the region from `%s` went undetected" % signature)
            status = 1
    return status


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SOURCE
    overrides = {"ggml-sycl.cpp": sys.argv[2]} if len(sys.argv) > 2 else {}
    with open(path, encoding="utf-8") as f:
        text = f.read()
    bad = violations(text)
    if not RESOLVER_CALL.search(strip_comments(text)):
        print("FAIL: no resolver call found in %s; the gate would pass vacuously" % path)
        return 1
    if bad:
        for line_no, name, fn_line in bad:
            print("FAIL: %s:%d: %s called outside a ggml_sycl_host_executor_region "
                  "(function block opens at line %d)" % (path, line_no, name, fn_line))
        return 1
    # Mutants: removing any one region declaration must be caught, or the
    # block walk is too coarse to attribute a call to its own function.
    src_lines = text.split("\n")
    decls = [i for i, l in enumerate(strip_comments(text).split("\n")) if REGION_DECL.match(l)]
    if not decls:
        print("FAIL: no ggml_sycl_host_executor_region declaration found in %s" % path)
        return 1
    missed = []
    for i in decls:
        mutated = "\n".join(src_lines[:i] + src_lines[i + 1:])
        if not violations(mutated):
            missed.append(i + 1)
    if missed:
        for line_no in missed:
            print("FAIL: removing the region at %s:%d went undetected (a resolver call is not "
                  "attributed to its own function)" % (path, line_no))
        return 1
    if check_external_executors(os.path.dirname(DEFAULT_SOURCE), overrides) != 0:
        return 1
    external = ", ".join(sig.split("(")[0].split()[-1] for _, sig in EXTERNAL_EXECUTORS)
    print("PASS: every resolver call in cpu-dispatch.cpp and in the external executors (%s) sits inside a "
          "host-executor region; %d cpu-dispatch region mutants caught" % (external, len(decls)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
