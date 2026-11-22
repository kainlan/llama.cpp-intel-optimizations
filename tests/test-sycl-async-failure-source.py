#!/usr/bin/env python3
"""Asynchronous SYCL errors fail closed at the graph boundaries (llama.cpp-kjoc).

With comments blanked:
  Q. (unified-cache.cpp ensure_single_device_context_queue) the execution queue's context and queue are built with
     ggml_sycl_make_context(dev, device) and ggml_sycl_make_queue(ctx, dev, device, ...).
  S. (ggml-sycl.cpp ggml_backend_sycl_synchronize) no SYCL_CHECK(err) and no CHECK_TRY_ERROR; a drain that throws is
     recorded with ggml_sycl_async_failure_record(sycl_ctx->device, ...); the failure branch tests
     ggml_sycl_async_failure_since(sycl_ctx->async_failure_cursor), puts the drain's own error in the
     [SYCL-SYNC-DIAG] line and calls ggml_backend_sycl_graph_boundary_exception_cleanup(sycl_ctx, "synchronize", ...)
     before the tail releases the graph.
  C. (ggml-sycl.cpp) that cleanup is guarded by !sycl_ctx->async_failure_cleaned, and the flag is set immediately
     before the ggml_backend_sycl_graph_boundary_exception_cleanup call; neither graph_compute nor
     ggml_sycl_async_failure_report touches the flag, so a refusal that reports the failure first cannot skip the
     cleanup of the graph the context submitted before it.
  G. (ggml-sycl.cpp ggml_backend_sycl_graph_compute) ggml_sycl_async_failure_since( is tested and GGML_STATUS_FAILED
     returned before the first ggml_backend_sycl_graph_compute_unchecked( call.
  R. (ggml-sycl.cpp ggml_sycl_async_failure_report) the [SYCL-ASYNC-FAILURE] line says the outputs of the context's
     last graph are not valid.
  T. (ggml-sycl.cpp moe_direct_event_terminal) arm takes a ggml_sycl_async_failure_cursor; ready() tests
     ggml_sycl_async_failure_since( after event->wait_and_throw(); the route arms with a snapshot taken before its
     first submission.
  H. (common.cpp ggml_sycl_async_handler) the handler body has no throw statement, no GGML_LOG, no lock or mutex, no
     std::string and no new.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"


def blank_comments(text):
    # Line comments first: a "/*" inside one (unified-cache.cpp has `weight-reclaim/*`) would otherwise open a block
    # comment that swallows code up to the next "*/".
    text = re.sub(r"//[^\n]*", lambda m: " " * len(m.group(0)), text)
    return re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)


def body(text, signature):
    """The brace-balanced body after the first occurrence of `signature`, or '' if absent."""
    start = text.find(signature)
    if start < 0:
        return ""
    open_at = text.find("{", start)
    depth = 0
    for i in range(open_at, len(text)):
        depth += text[i] == "{"
        depth -= text[i] == "}"
        if depth == 0:
            return text[open_at:i + 1]
    return ""


def claims(cache, sycl, common):
    q = body(cache, "static sycl::queue * ensure_single_device_context_queue(int device)")
    s = body(sycl, "static void ggml_backend_sycl_synchronize(ggml_backend_t backend)")
    g = body(sycl, "static ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph)")
    r = body(sycl, "static void ggml_sycl_async_failure_report(ggml_backend_sycl_context * ctx, const char * stage)")
    t = body(sycl, "class moe_direct_event_terminal final")
    h = body(common, "sycl::async_handler ggml_sycl_async_handler(int slot)")
    g_check = g.find("ggml_sycl_async_failure_since(")
    g_first = g.find("ggml_backend_sycl_graph_compute_unchecked(")
    s_since = s.find("ggml_sycl_async_failure_since(sycl_ctx->async_failure_cursor)")
    s_clean = s.find('ggml_backend_sycl_graph_boundary_exception_cleanup(sycl_ctx, "synchronize"')
    s_release = s.find("(void) ggml_sycl_execution_release_graph(sycl_ctx);")
    return {
        "Q execution context has the handler": "ggml_sycl_make_context(dev, device)" in q,
        "Q execution queue has the handler": "ggml_sycl_make_queue(ctx, dev, device," in q,
        "S no abort on a throwing drain": s != "" and "SYCL_CHECK(err)" not in s and "CHECK_TRY_ERROR" not in s,
        "S a throwing drain is latched": "ggml_sycl_async_failure_record(sycl_ctx->device, drain_error.c_str());" in s,
        "S the diag line carries the drain's error": "[SYCL-SYNC-DIAG]" in s and "drain error: %s" in s,
        "S failure branch cleans up before the tail": 0 <= s_since < s_clean < s_release,
        "C cleanup guarded by the cleaned flag, set before the call": re.search(
            r"if \(!sycl_ctx->async_failure_cleaned\) \{\s*sycl_ctx->async_failure_cleaned = true;\s*"
            r"ggml_backend_sycl_graph_boundary_exception_cleanup\(sycl_ctx, \"synchronize\"", s) is not None,
        "C refusal and report leave the cleaned flag alone": g != "" and r != "" and
            "async_failure_cleaned" not in g and "async_failure_cleaned" not in r,
        "G refusal before compute": 0 <= g_check < g_first and "GGML_STATUS_FAILED" in g[g_check:g_first],
        "R the report says the last outputs are invalid": "are not valid" in r,
        "T arm takes a cursor": "void arm(sycl::event event, ggml_sycl_async_failure_cursor cursor)" in t,
        "T ready reads the latch after the wait": 0 <= t.find("event->wait_and_throw();") <
            t.find("ggml_sycl_async_failure_since(cursor)"),
        "T route arms with its cursor": "terminal->arm(final, failure_cursor);" in sycl and
            "const ggml_sycl_async_failure_cursor failure_cursor = ggml_sycl_async_failure_snapshot(ctx.device);" in sycl,
        "H handler is bounded": h != "" and not re.search(r"\bthrow\b|GGML_LOG|lock_guard|unique_lock|mutex|"
                                                      r"std::string|\bnew\b", h.replace("rethrow_exception", "")),
    }


def check(cache, sycl, common):
    return [name for name, ok in claims(cache, sycl, common).items() if not ok]


def main():
    cache = blank_comments((SYCL / "unified-cache.cpp").read_text())
    sycl = blank_comments((SYCL / "ggml-sycl.cpp").read_text())
    common = blank_comments((SYCL / "common.cpp").read_text())
    failed = check(cache, sycl, common)
    if failed:
        print("async failure source contract failed: " + ", ".join(failed), file=sys.stderr)
        return 1
    if "--self-test" in sys.argv:
        mutants = {
            "execution queue without handler": (cache.replace("ggml_sycl_make_queue(ctx, dev, device,",
                                                              "sycl::queue(ctx, dev,", 1), sycl, common),
            "synchronize aborts again": (cache, sycl.replace(
                "ggml_sycl_async_failure_record(sycl_ctx->device, drain_error.c_str());", "SYCL_CHECK(err);", 1),
                common),
            "cleanup guarded by first_report": (cache, sycl.replace("if (!sycl_ctx->async_failure_cleaned) {",
                                                                    "if (first_report) {", 1), common),
            "cleanup guarded by the reported flag": (cache, sycl.replace(
                "if (!sycl_ctx->async_failure_cleaned) {", "if (!sycl_ctx->async_failure_reported) {", 1), common),
            "cleaned flag set after the cleanup": (cache, re.sub(
                r"(sycl_ctx->async_failure_cleaned = true;)(\s*)(ggml_backend_sycl_graph_boundary_exception_cleanup"
                r"\(sycl_ctx, \"synchronize\",\s*\"[^\"]*\"\);)", r"\3\2\1", sycl, count=1), common),
            "refusal sets the cleaned flag": (cache, sycl.replace(
                "ggml_sycl_async_failure_report(failed_ctx, \"graph_compute\");",
                "failed_ctx->async_failure_cleaned = true; ggml_sycl_async_failure_report(failed_ctx, \"graph_compute\");",
                1), common),
            "report sets the cleaned flag": (cache, sycl.replace(
                "ctx->async_failure_reported = true;",
                "ctx->async_failure_reported = true; ctx->async_failure_cleaned = true;", 1), common),
            "graph_compute does not refuse": (cache, sycl.replace("ggml_sycl_async_failure_since(failed_ctx->",
                                                                  "false && (failed_ctx->", 1), common),
            "terminal ignores the latch": (cache, sycl.replace("terminal->arm(final, failure_cursor);",
                                                               "terminal->arm(final, {});", 1), common),
            "handler logs": (cache, sycl, common.replace("ggml_sycl_async_failure_record(bound, e.what());",
                                                         "GGML_LOG_ERROR(\"%s\", e.what());", 1)),
        }
        for name, args in mutants.items():
            if not check(*args):
                print(f"async failure source contract: mutant '{name}' was not caught", file=sys.stderr)
                return 1
    print("async failure source contract: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
