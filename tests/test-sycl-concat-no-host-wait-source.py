#!/usr/bin/env python3
"""CONCAT has no host wait and no graph-compatibility rejection (llama.cpp-qhfp).

Host-only: reads sources, runs no build, loads no model and touches no device. It proves the two
source facts below; it cannot prove that a graph replays (the GPU witness does that).

The defect: check_graph_compatibility() rejected every GGML_OP_CONCAT with the claim that
ggml_sycl_op_concat() "does a blocking host wait after memcpy operations". The only host wait in
concat.cpp was `stream->wait()` after the dim==3 contiguous memcpy pair, and it was already skipped
while recording. qwen35 (Qwen3.6-27B) builds a CONCAT into every gated-delta-net layer's conv-state
update, so the stale rejection disabled SYCL graph replay for the whole architecture's decode.

What actually enabled qwen35: removing the stale rejection. qwen35's CONCAT is dim 0 with a
non-contiguous transposed src1, so it takes concat_T_sycl_non_cont and never reached the deleted wait.
The wait is removed anyway: the compute stream is in-order, so the next op on it is ordered after the
two copies, and a host wait contradicts the fork rule "no host waits, event-chain everything".

Pinned here: no host wait anywhere in concat.cpp, no wait guarded by the recording flag, no
CONCAT rejection in check_graph_compatibility(), and the graph-safe memcpy helper is correct for
any alignment (the recording branch copies through the pure width helper, not an int32 split that
misaligns an odd byte offset).

Run with --self-test to prove every check fires against a mutant of the thing it
forbids; a check that cannot fail is decoration.
"""
import argparse
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sycl = root / "ggml/src/ggml-sycl"

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default=str(sycl / "ggml-sycl.cpp"))
parser.add_argument("--concat", default=str(sycl / "concat.cpp"))
parser.add_argument("--common", default=str(sycl / "common.hpp"))
parser.add_argument("--self-test", action="store_true")
args = parser.parse_args()


def strip_comments(source):
    """Remove C/C++ comments, keeping string literals and the line count."""
    out = []
    i = 0
    n = len(source)
    while i < n:
        ch = source[i]
        if ch in ('"', "'"):
            quote = ch
            out.append(ch)
            i += 1
            while i < n:
                out.append(source[i])
                if source[i] == "\\":
                    if i + 1 < n:
                        out.append(source[i + 1])
                        i += 2
                        continue
                elif source[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        if source.startswith("//", i):
            while i < n and source[i] != "\n":
                i += 1
            continue
        if source.startswith("/*", i):
            end = source.find("*/", i + 2)
            end = n if end < 0 else end + 2
            out.append("\n" * source.count("\n", i, end))
            i = end
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def read(path):
    return strip_comments(Path(path).read_text())


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


# Every spelling of a host-side block: a queue or event wait, a throwing wait, a
# direct synchronize. Any of these in the op path stalls the submitting thread.
HOST_WAIT = re.compile(r"(\.|->|::)\s*wait\s*\(|wait_and_throw|\bsynchronize\s*\(|\bhost_task\b")


def evaluate(backend, concat, common):
    results = {}

    compat = function_body(backend, r"static bool check_graph_compatibility\([^)]*\)\s*\{")
    impl = function_body(concat, r"void concat_impl_sycl\([^)]*\)\s*\{")
    op = function_body(concat, r"void ggml_sycl_op_concat\([^)]*\)\s*\{")
    results["anchor: check_graph_compatibility exists"] = compat is not None
    results["anchor: concat_impl_sycl exists"] = impl is not None
    results["anchor: ggml_sycl_op_concat exists"] = op is not None
    if None in (compat, impl, op):
        return results

    # --- the compatibility scan no longer rejects CONCAT ----------------------
    results["check_graph_compatibility has no CONCAT case"] = "GGML_OP_CONCAT" not in compat

    # --- no host wait anywhere in the op path --------------------------------
    results["concat.cpp has no host wait"] = HOST_WAIT.search(concat) is None
    results["concat_impl_sycl has no host wait"] = HOST_WAIT.search(impl) is None
    # A wait inside a recording branch is the exact shape of the original guard
    # (`if (!g_ggml_sycl_graph_recording) { stream->wait(); }`). Narrowed to that: the flag itself is fine
    # to read, a wait near it is not.
    results["no wait inside a recording branch"] = all(
        HOST_WAIT.search(concat[m.start():m.start() + 300]) is None
        for m in re.finditer(r"g_ggml_sycl_graph_recording", concat))

    # --- nothing else in the op can break recording or replay ----------------
    results["concat.cpp allocates nothing"] = \
        re.search(r"sycl::malloc|malloc_device|malloc_host|unified_alloc|unified_allocate", concat) is None
    # The copies go through the graph-safe helper, which emits a kernel (not a
    # memcpy node) while recording.
    results["contiguous dim==3 copies use the graph-safe memcpy"] = \
        impl.count("ggml_sycl_graph_safe_memcpy(") == 2 and re.search(r"\.memcpy\(", concat) is None
    # The removed wait is only sound on an in-order queue; the assert binds that premise in one expression.
    results["the op asserts an in-order queue"] = \
        re.search(r"GGML_ASSERT\([^;]*has_property<sycl::property::queue::in_order>", impl) is not None

    # --- the helper the op copies through is alignment-correct ------------------
    safe = function_body(common, r"inline sycl::event ggml_sycl_graph_safe_memcpy\([^)]*\)\s*\{")
    results["anchor: ggml_sycl_graph_safe_memcpy exists"] = safe is not None
    if safe is not None:
        recording = safe[:safe.find("const int             queue_device")] if "const int             queue_device" in safe \
            else safe
        # An int32 body followed by a byte tail misaligns dst whenever (dst | src | nbytes) is not a
        # multiple of 4 -- dim==3 CONCAT with an odd src0 byte count hands the second copy exactly that.
        results["recording copy never splits an int32 body from a byte tail"] = \
            re.search(r"nbytes\s*/\s*sizeof\(int32_t\)", recording) is None
        results["recording copy picks its width from the operands' alignment"] = \
            "ggml_sycl_graph_safe_memcpy_width(" in recording
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


backend, concat, common = read(args.backend), read(args.concat), read(args.common)
failed = run("tree", (backend, concat, common))

# A comment is not code, so the stripped sources cannot see it. The stale
# comment was a false claim a reader acted on.
raw_backend = re.sub(r"\s*\n\s*//\s*", " ", Path(args.backend).read_text())
if "does a blocking host wait after memcpy operations" in raw_backend:
    print("FAIL: the stale 'ggml_sycl_op_concat() does a blocking host wait' comment is still in ggml-sycl.cpp")
    failed.append("stale concat wait comment")
else:
    print("PASS: the stale concat wait comment is gone")

# Comments are not code either; two of them carry claims a reader acts on.
raw_common = Path(args.common).read_text()
raw_concat = re.sub(r"\s*\n\s*//\s*", " ", Path(args.concat).read_text())
if "same-device" in raw_common[raw_common.find("inline sycl::event ggml_sycl_graph_safe_memcpy"):][:4000]:
    print("PASS: the safe-memcpy comment scopes 'no host wait' to same-device USM operands")
else:
    print("FAIL: the safe-memcpy non-recording comment does not scope 'no host wait' to same-device USM operands")
    failed.append("safe-memcpy comment scope")
if "check_queue_order" in raw_concat and "precedent" in raw_concat:
    print("PASS: the in-order assert names its precedent")
else:
    print("FAIL: the in-order assert in concat.cpp does not name the check_queue_order / planned-scratch precedent")
    failed.append("assert precedent comment")

if args.self_test:
    def mutate_in_func(src, sig_regex, old, new):
        m = re.search(sig_regex, src)
        if not m or src.find(old, m.end()) < 0:
            print(f"FAIL: self-test anchor missing: {sig_regex!r} .. {old!r}")
            failed.append("self-test anchor " + old)
            return src
        k = src.find(old, m.end())
        return src[:k] + new + src[k + len(old):]

    compat_sig = r"static bool check_graph_compatibility\([^)]*\)\s*\{"
    impl_sig = r"void concat_impl_sycl\([^)]*\)\s*\{"
    safe_sig = r"inline sycl::event ggml_sycl_graph_safe_memcpy\([^)]*\)\s*\{"
    first_stmt = "const size_t size0 = src0.nbytes();"
    # Inject into the first node-op switch of the compat scan: the rejection as it used to be.
    reject = "case GGML_OP_CONCAT: GGML_LOG_INFO(\"%s: disabling SYCL graphs due to unsupported node type %s\\n\", " \
             "__func__, ggml_op_name(node_op)); return false;\n            "

    def with_concat(text):
        return (backend, text, common)

    mutants = [
        ("concat rejection restored", "check_graph_compatibility has no CONCAT case",
         (mutate_in_func(backend, compat_sig, "case GGML_OP_MUL_MAT_ID:", reject + "case GGML_OP_MUL_MAT_ID:"),
          concat, common)),
        ("stream wait restored", "concat.cpp has no host wait",
         with_concat(mutate_in_func(concat, impl_sig, "} else {\n        concat_T_sycl_non_cont",
                                    "stream->wait(); } else {\n        concat_T_sycl_non_cont"))),
        ("event wait restored", "concat_impl_sycl has no host wait",
         with_concat(mutate_in_func(concat, impl_sig, first_stmt, first_stmt + " sycl::event{}.wait();"))),
        ("throwing wait restored", "concat.cpp has no host wait",
         with_concat(mutate_in_func(concat, impl_sig, first_stmt, first_stmt + " stream->wait_and_throw();"))),
        ("scope-qualified wait", "concat.cpp has no host wait",
         with_concat(mutate_in_func(concat, impl_sig, first_stmt, first_stmt + " sycl::event::wait(evs);"))),
        ("host_task restored", "concat.cpp has no host wait",
         with_concat(mutate_in_func(concat, impl_sig, first_stmt,
                                    first_stmt + " stream->submit([&](sycl::handler & h) { h.host_task([] {}); });"))),
        ("recording-guarded wait", "no wait inside a recording branch",
         with_concat(mutate_in_func(concat, impl_sig, first_stmt,
                                    first_stmt + " if (!g_ggml_sycl_graph_recording) { stream->wait(); }"))),
        ("allocation added", "concat.cpp allocates nothing",
         with_concat(mutate_in_func(concat, impl_sig, first_stmt,
                                    first_stmt + " void * p = sycl::malloc_device(8, *stream);"))),
        ("raw memcpy node", "contiguous dim==3 copies use the graph-safe memcpy",
         with_concat(mutate_in_func(concat, impl_sig, "ggml_sycl_graph_safe_memcpy(*stream, dst_d, src0_d, size0)",
                                    "stream->memcpy(dst_d, src0_d, size0)"))),
        ("in-order assert dropped", "the op asserts an in-order queue",
         with_concat(mutate_in_func(concat, impl_sig, "has_property<sycl::property::queue::in_order>",
                                    "has_property<XXXX>"))),
        ("in-order assert detached", "the op asserts an in-order queue",
         with_concat(mutate_in_func(concat, impl_sig, "GGML_ASSERT(stream->has_property<sycl::property::queue::in_order>());",
                                    "bool ok = stream->has_property<sycl::property::queue::in_order>(); GGML_ASSERT(true);"))),
        ("int32 split restored", "recording copy never splits an int32 body from a byte tail",
         (backend, concat, mutate_in_func(common, safe_sig, "if (g_ggml_sycl_graph_recording) {",
                                          "if (g_ggml_sycl_graph_recording) { const size_t n_i32 = nbytes / sizeof(int32_t);"
                                          " (void) n_i32;"))),
        ("width helper unused", "recording copy picks its width from the operands' alignment",
         (backend, concat, mutate_in_func(common, safe_sig, "ggml_sycl_graph_safe_memcpy_width(",
                                          "ggml_sycl_graph_safe_memcpy_XXXX("))),
    ]
    for label, expect, sources in mutants:
        failed += run(label, sources, expect)

if failed:
    print("\nFAILED: " + ", ".join(failed))
    sys.exit(1)
print("\nALL CHECKS PASSED")
