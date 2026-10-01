#!/usr/bin/env python3
"""Source gate: a resolver call in cpu-dispatch.cpp runs inside a host-executor region.

The GW7 counter (resolver host returns of staged sources) counts only while a
thread is dispatching a device graph. cpu-dispatch.cpp functions run on the
host as executors inside that dispatch, and their resolver calls return host
pointers by design; they must open ggml_sycl_host_executor_region before the
first such call or the counter reads host execution as device reads.

The gate walks every top-level function of the file, and for each one that
calls a resolver requires the region declaration earlier in the same function.
argv: [cpu-dispatch.cpp]
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


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SOURCE
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
    print("PASS: every resolver call in cpu-dispatch.cpp sits inside a host-executor region; %d region mutants caught" % len(decls))
    return 0


if __name__ == "__main__":
    sys.exit(main())
