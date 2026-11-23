#!/usr/bin/env python3
"""The CPU row kernels ask one bound before they quantize activations into src1_q.

Host-only: reads sources, runs no build, loads no model and touches no device.

Every CPU vec_dot kernel in cpu-dispatch.cpp quantizes its activation rows into the thread-local
g_cpu_dispatch_buffers.src1_q, which init() sizes to CPU_DISPATCH_SRC1_Q_BYTES. A row of the weight's vec_dot_type
can outgrow it: an F16 weight takes F32 rows, which pass the quantized-row bound at a K of a few thousand. One
predicate, cpu_dispatch_src1_q_fits (cpu-dispatch-buffers.hpp), states the bound.

Checks:
  B1  cpu_dispatch_src1_q_fits answers rows <= CPU_DISPATCH_SRC1_Q_BYTES / row_bytes and refuses a zero row size, and
      init() sizes src1_q to CPU_DISPATCH_SRC1_Q_BYTES;
  B2  cpu_mul_mat takes its vec_dot path only when use_vec_dot's initializer asked cpu_dispatch_src1_q_fits for its M
      rows of the vec_dot type's row size, so a K too large for the buffer takes the GEMM fallback instead;
  B3  cpu_dispatch_src1_q_require asks cpu_dispatch_src1_q_fits and that this thread's src1_q has that size, and
      aborts with a message on either miss (a void kernel has no fallback that leaves its output written);
  B4  the four single-row kernels (ggml_sycl_cpu_vec_dot_rows, ggml_sycl_cpu_expert_mul_mat, cpu_expert_mul_mat_int4,
      ggml_sycl_cpu_expert_fused_gate_up_silu) and cpu_mul_mat's vec_dot path call cpu_dispatch_src1_q_require before
      they take src1_q's data, and none of them compares a size with src1_q.size() itself.

Known gap: ggml_sycl_cpu_pp_gemm still asserts M * q_row_size <= src1_q.size() for a prompt's M rows; its fallback
would be its own dequant + sgemm path, which is a routing change, not a bound check.

--self-test runs each check against a mutant that must fail it, and requires the unmodified tree to pass.
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SYCL = ROOT / "ggml" / "src" / "ggml-sycl"

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--dispatch", default=str(SYCL / "cpu-dispatch.cpp"))
parser.add_argument("--buffers", default=str(SYCL / "cpu-dispatch-buffers.hpp"))
parser.add_argument("--self-test", action="store_true")
args = parser.parse_args()


def strip_comments(source):
    """Blank // and /* */ comments outside string literals, keeping the line count."""
    out, i, n = [], 0, len(source)
    while i < n:
        ch = source[i]
        if ch in ('"', "'"):
            j = i + 1
            while j < n and source[j] != ch:
                j += 2 if source[j] == "\\" else 1
            out.append(source[i:j + 1])
            i = j + 1
        elif source.startswith("//", i):
            while i < n and source[i] != "\n":
                i += 1
        elif source.startswith("/*", i):
            end = source.find("*/", i + 2)
            end = n if end < 0 else end + 2
            out.append("\n" * source.count("\n", i, end))
            i = end
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def function_body(source, signature_regex):
    """The brace-balanced body of the first definition whose signature matches, or None."""
    m = re.search(signature_regex, source)
    if not m:
        return None
    start = source.find("{", m.end() - 1)
    depth = 0
    for i in range(start, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start:i + 1]
    return None


FITS = r"inline\s+bool\s+cpu_dispatch_src1_q_fits\(\s*size_t\s+rows\s*,\s*size_t\s+row_bytes\s*\)\s*\{"
INIT = r"void\s+init\(\s*\)\s*\{"
REQUIRE = r"static\s+void\s+cpu_dispatch_src1_q_require\([^)]*\)\s*\{"
CPU_MUL_MAT = r"static\s+bool\s+cpu_mul_mat\(\s*ggml_backend_sycl_context\s*&\s*ctx\s*,\s*ggml_tensor\s*\*\s*dst\s*\)\s*\{"

# name -> (signature, the size variable of its activation row)
ROW_KERNELS = {
    "ggml_sycl_cpu_vec_dot_rows": (r"void\s+ggml_sycl_cpu_vec_dot_rows\([^)]*\)\s*\{", "q_row_size"),
    "ggml_sycl_cpu_expert_mul_mat": (r"void\s+ggml_sycl_cpu_expert_mul_mat\([^)]*\)\s*\{", "q_size"),
    "cpu_expert_mul_mat_int4": (r"static\s+void\s+cpu_expert_mul_mat_int4\([^)]*\)\s*\{", "q_size"),
    "ggml_sycl_cpu_expert_fused_gate_up_silu":
        (r"void\s+ggml_sycl_cpu_expert_fused_gate_up_silu\([^)]*\)\s*\{", "q_row_size"),
}

FITS_ANSWER = r"return\s+row_bytes\s*!=\s*0\s*&&\s*rows\s*<=\s*CPU_DISPATCH_SRC1_Q_BYTES\s*/\s*row_bytes\s*;"
INIT_SIZE = r"src1_q\.resize\(\s*CPU_DISPATCH_SRC1_Q_BYTES\s*\)\s*;"
USE_VEC_DOT_FITS = (r"const\s+bool\s+use_vec_dot\s*=[^;]*cpu_dispatch_src1_q_fits\(\s*static_cast<size_t>\(M\)\s*,\s*"
                    r"ggml_row_size\(\s*cpu_traits->vec_dot_type\s*,\s*K\s*\)\s*\)\s*;")
REQUIRE_FITS = (r"if\s*\(\s*!cpu_dispatch_src1_q_fits\(\s*rows\s*,\s*row_bytes\s*\)\s*\)\s*\{\s*GGML_ABORT\(")
REQUIRE_LIVE = (r"if\s*\(\s*g_cpu_dispatch_buffers\.src1_q\.size\(\)\s*<\s*CPU_DISPATCH_SRC1_Q_BYTES\s*\)\s*\{\s*"
                r"GGML_ABORT\(")
RAW_SIZE = r"src1_q\.size\(\)"


def require_call(rows, size_var):
    return r"cpu_dispatch_src1_q_require\(\s*" + rows + r"\s*,\s*" + size_var + r"\s*,"


def asks_before_data(body, rows, size_var):
    """`body` calls the require helper for its rows before it first takes src1_q's data."""
    call = re.search(require_call(rows, size_var), body)
    data = body.find("g_cpu_dispatch_buffers.src1_q.data()")
    return call is not None and data > call.end()


def evaluate(dispatch, buffers):
    results = {}
    fits = function_body(buffers, FITS) or ""
    init = function_body(buffers, INIT) or ""
    results["B1 cpu_dispatch_src1_q_fits states the bound and refuses a zero row size"] = \
        re.search(FITS_ANSWER, fits) is not None
    results["B1 init() sizes src1_q to the bound"] = re.search(INIT_SIZE, init) is not None

    mul_mat = function_body(dispatch, CPU_MUL_MAT) or ""
    results["B2 cpu_mul_mat's use_vec_dot asks cpu_dispatch_src1_q_fits for its M rows"] = \
        re.search(USE_VEC_DOT_FITS, mul_mat) is not None
    results["B4 cpu_mul_mat's vec_dot path asks the require helper before it takes src1_q"] = \
        asks_before_data(mul_mat, r"static_cast<size_t>\(M\)", "q_row_size")
    results["B4 cpu_mul_mat compares no size with src1_q.size() itself"] = re.search(RAW_SIZE, mul_mat) is None

    require = function_body(dispatch, REQUIRE) or ""
    results["B3 the require helper aborts when cpu_dispatch_src1_q_fits refuses"] = \
        re.search(REQUIRE_FITS, require) is not None
    results["B3 the require helper aborts when this thread's src1_q is smaller than the bound"] = \
        re.search(REQUIRE_LIVE, require) is not None

    for name, (sig, size_var) in ROW_KERNELS.items():
        body = function_body(dispatch, sig) or ""
        results[f"B4 {name} asks the require helper before it takes src1_q"] = \
            asks_before_data(body, "1", size_var)
        results[f"B4 {name} compares no size with src1_q.size() itself"] = \
            body != "" and re.search(RAW_SIZE, body) is None
    return results


def run(label, sources, expect_fail=None):
    results = evaluate(*sources)
    failed = sorted(k for k, v in results.items() if not v)
    if expect_fail is None:
        for k in sorted(results):
            print(("PASS: " if results[k] else "FAIL: ") + k)
        return failed
    fired = expect_fail in failed
    print(("PASS" if fired else "FAIL") + f": mutant '{label}' fires '{expect_fail}'")
    return [] if fired else [label]


dispatch = strip_comments(Path(args.dispatch).read_text())
buffers = strip_comments(Path(args.buffers).read_text())
failed = run("tree", (dispatch, buffers))

if args.self_test:
    def mutate_in_func(src, sig_regex, pattern, repl):
        """Replace the first match of `pattern` after the first match of `sig_regex`."""
        m = re.search(sig_regex, src)
        pm = re.compile(pattern).search(src, m.end()) if m else None
        if not pm:
            print(f"FAIL: self-test anchor missing: {sig_regex!r} .. {pattern!r}")
            failed.append("self-test anchor " + pattern)
            return src
        return src[:pm.start()] + repl + src[pm.end():]

    mutants = [
        ("the bound takes a zero row size", "B1 cpu_dispatch_src1_q_fits states the bound and refuses a zero row size",
         (dispatch, mutate_in_func(buffers, FITS, r"row_bytes\s*!=\s*0\s*&&\s*", ""))),
        ("init() sizes src1_q to one row", "B1 init() sizes src1_q to the bound",
         (dispatch, mutate_in_func(buffers, INIT, INIT_SIZE, "src1_q.resize(CPU_DISPATCH_MAX_Q_ROW_SIZE);"))),
        ("use_vec_dot no longer asks the bound", "B2 cpu_mul_mat's use_vec_dot asks cpu_dispatch_src1_q_fits for its M rows",
         (mutate_in_func(dispatch, CPU_MUL_MAT, r"&&\s*cpu_dispatch_src1_q_fits\([^;]*\)\s*\)\s*;", ";"), buffers)),
        ("cpu_mul_mat asserts the raw size again",
         "B4 cpu_mul_mat compares no size with src1_q.size() itself",
         (mutate_in_func(dispatch, CPU_MUL_MAT, require_call(r"static_cast<size_t>\(M\)", "q_row_size") + r"[^;]*;",
                         "GGML_ASSERT(static_cast<size_t>(M) * q_row_size <= g_cpu_dispatch_buffers.src1_q.size());"),
          buffers)),
        ("cpu_mul_mat takes src1_q before it asks", "B4 cpu_mul_mat's vec_dot path asks the require helper before it takes src1_q",
         (mutate_in_func(dispatch, CPU_MUL_MAT, require_call(r"static_cast<size_t>\(M\)", "q_row_size") + r"[^;]*;", ""),
          buffers)),
        ("the require helper skips the bound", "B3 the require helper aborts when cpu_dispatch_src1_q_fits refuses",
         (mutate_in_func(dispatch, REQUIRE, r"!cpu_dispatch_src1_q_fits\(\s*rows\s*,\s*row_bytes\s*\)", "false"), buffers)),
        ("the require helper trusts an unsized buffer",
         "B3 the require helper aborts when this thread's src1_q is smaller than the bound",
         (mutate_in_func(dispatch, REQUIRE, r"g_cpu_dispatch_buffers\.src1_q\.size\(\)\s*<\s*CPU_DISPATCH_SRC1_Q_BYTES",
                         "false"), buffers)),
    ]
    for name, (sig, size_var) in ROW_KERNELS.items():
        call = require_call("1", size_var) + r"[^;]*;"
        mutants.append((f"{name} asserts the raw size again", f"B4 {name} compares no size with src1_q.size() itself",
                        (mutate_in_func(dispatch, sig, call,
                                        f"GGML_ASSERT({size_var} <= g_cpu_dispatch_buffers.src1_q.size());"), buffers)))
        mutants.append((f"{name} takes src1_q without asking", f"B4 {name} asks the require helper before it takes src1_q",
                        (mutate_in_func(dispatch, sig, call, ""), buffers)))

    for label, expect, sources in mutants:
        failed += run(label, sources, expect)

if failed:
    print("\nFAILED: " + ", ".join(failed))
    sys.exit(1)
print("\nALL CHECKS PASSED")
