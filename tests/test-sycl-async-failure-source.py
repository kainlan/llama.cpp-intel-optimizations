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
  T. (ggml-sycl.cpp moe_direct_event_terminal) the terminal says when its work is finished and nothing else: ready()
     tests the command_execution_status, names neither ggml_sycl_async_failure_since nor wait_and_throw, and calls no
     wait at all, because the slot-release sites read it as "finished" under the registry and pool mutexes; the class
     has no observe_success and names no async-failure latch, arm takes the event alone, and the route arms it with
     its final event. Whether the work succeeded is the latch's to say, read only by a boundary that drained on its own
     thread: a wait on a host-task worker can return while another thread is still delivering the error.
  F. (ggml-sycl.cpp, the completion host task) the task releases a finished bundle whatever the latch says: the body
     computes recycled as !skip_release && completion->bundle.terminal_release(), and names no
     ggml_sycl_async_failure, no observe_success and no wait.
  B. (ggml-sycl.cpp, the completion host task) the body's last statement assigns an empty bundle to
     completion->bundle, after the terminal_release() expression. The bundle holds the invocation pin until
     terminal_release() runs, and a release that was refused or skipped never resets it; the task's event completes
     only after the body returns, so a boundary that waited on that event finds the pin released however long the
     runtime keeps the functor or copies of it.
  D. (ggml-sycl.cpp) a release refused during a drain is retried after the workspace recovery:
     ggml_sycl_execution_recover_and_release_graph waits the stream, then calls
     ggml_sycl_recover_exact_mmid_after_drain(ctx), then ggml_sycl_execution_release_graph(ctx); both drain twins
     call it before ggml_sycl_execution_abort_and_release_graph, with the same condition: a wait that threw
     (waited false) goes straight to the abort, else release, else recover and release; the backend destructor
     calls the recovery before its abort-and-release.
  W. (ggml-sycl.cpp) a drain wait that throws is recorded in the device latch, as synchronize records one: a
     synchronous throw (a lost device) bypasses the queue's handler. The catch of the drain retry's wait, and of the
     first wait of each drain twin, calls ggml_sycl_async_failure_record with that context's device.
  K. (ggml-sycl.cpp) the completion-hold test seam (a second holder of the direct route's completion object that
     outlives its host task, so the oracle can show that B, not the runtime's functor lifetime, releases the pin) is
     compiled only for route testing: every mention of ggml_sycl_q1_nvfp4_test_retain_completion,
     ggml_sycl_q1_nvfp4_test_hold_completion, ggml_sycl_q1_nvfp4_test_completion_held and the held pointer sits inside
     an #ifdef GGML_SYCL_Q1_NVFP4_ROUTE_TESTING branch, and the direct route in ggml_sycl_mul_mat_id hands its
     completion to the seam after mark_possible_submit() and before terminal->arm(final) and the completion host
     task, so the seam holds the completion of the graph it was armed for. The same holds
     for the completion failpoint GGML_SYCL_Q1_NVFP4_TEST_FAILURE_COMPLETION_NO_RELEASE (a completion task that has
     not released when the context is drained): the host task consumes it before its release, and the build without
     route testing defines skip_release as constexpr false.
  H. (common.cpp ggml_sycl_async_handler) the handler body has no throw statement, no GGML_LOG, no lock or mutex, no
     std::string and no new.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"
BUNDLE_DROP = "completion->bundle = ggml_sycl::moe_admitted_workspace_bundle{};"
TEST_SEAM = ("ggml_sycl_q1_nvfp4_test_retain_completion", "ggml_sycl_q1_nvfp4_test_hold_completion",
             "ggml_sycl_q1_nvfp4_test_completion_held", "g_q1_nvfp4_test_held_completion", "g_q1_nvfp4_test_hold_",
             "GGML_SYCL_Q1_NVFP4_TEST_FAILURE_COMPLETION_NO_RELEASE")
NO_RELEASE = "GGML_SYCL_Q1_NVFP4_TEST_FAILURE_COMPLETION_NO_RELEASE"
RELEASE = "const bool recycled = !skip_release && completion->bundle.terminal_release();"
RETAIN = "ggml_sycl_q1_nvfp4_test_retain_completion(completion);"
RETAIN_BLOCK = "#ifdef GGML_SYCL_Q1_NVFP4_ROUTE_TESTING\n                            " + RETAIN + "\n#endif\n"
# Both drain twins: a wait that threw goes straight to the abort, else release, else recover and release.
TWIN_CONDITION = (r"if \(!waited \|\|\s*\(!ggml_sycl_execution_release_graph\({ctx}\) &&\s*"
                  r"!ggml_sycl_execution_recover_and_release_graph\({ctx}\)\)\) \{{")
# (claim name, function body local, the record call its catch must make)
DRAIN_RECORDS = (
    ("W the drain retry records a throwing wait", "d",
     'ggml_sycl_async_failure_record(ctx->device, "drain retry wait threw");'),
    ("W the begin-path twin records a throwing wait", "d_own",
     'ggml_sycl_async_failure_record(ctx->device, "begin-path drain wait threw");'),
    ("W the context twin records a throwing wait", "d_ctx",
     'ggml_sycl_async_failure_record(backend_ctx->device, "context drain wait threw");'),
)


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


def route_testing_lines(text):
    """Indexes of the lines inside the taken branch of an #ifdef GGML_SYCL_Q1_NVFP4_ROUTE_TESTING."""
    inside, stack = set(), []
    for n, line in enumerate(text.split("\n")):
        stripped = line.strip()
        if stripped.startswith("#"):
            directive = " ".join(stripped[1:].split())
            if directive.startswith("if"):
                stack.append(directive in ("ifdef GGML_SYCL_Q1_NVFP4_ROUTE_TESTING",
                                           "if defined(GGML_SYCL_Q1_NVFP4_ROUTE_TESTING)"))
                continue
            if directive.startswith(("else", "elif")) and stack:
                stack[-1] = False
                continue
            if directive.startswith("endif") and stack:
                stack.pop()
                continue
        if any(stack):
            inside.add(n)
    return inside


def test_seam_guarded(sycl):
    inside = route_testing_lines(sycl)
    mentions = [n for n, line in enumerate(sycl.split("\n")) if any(name in line for name in TEST_SEAM)]
    return bool(mentions) and all(n in inside for n in mentions)


def claims(cache, sycl, common):
    q = body(cache, "static sycl::queue * ensure_single_device_context_queue(int device)")
    s = body(sycl, "static void ggml_backend_sycl_synchronize(ggml_backend_t backend)")
    g = body(sycl, "static ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph)")
    r = body(sycl, "static void ggml_sycl_async_failure_report(ggml_backend_sycl_context * ctx, const char * stage)")
    t = body(sycl, "class moe_direct_event_terminal final")
    t_ready = body(t, "bool ready() const noexcept override")
    h = body(common, "sycl::async_handler ggml_sycl_async_handler(int slot)")
    b = body(sycl, "h.host_task([completion")[1:-1].rstrip()
    d = body(sycl, "static bool ggml_sycl_execution_recover_and_release_graph(ggml_backend_sycl_context * ctx) noexcept")
    d_own = body(sycl, "static void ggml_sycl_execution_drain_own_terminal(ggml_backend_sycl_context * ctx) noexcept {")
    d_ctx = body(sycl, "static void ggml_sycl_execution_drain_context_terminal_events(uint64_t context_id) {")
    d_dtor = body(sycl, "ggml_backend_sycl_context::~ggml_backend_sycl_context() {")
    d_wait = d.find("stream->wait_and_throw();")
    d_recover = d.find("ggml_sycl_recover_exact_mmid_after_drain(ctx);")
    d_release = d.find("return ggml_sycl_execution_release_graph(ctx);")
    g_check = g.find("ggml_sycl_async_failure_since(")
    g_first = g.find("ggml_backend_sycl_graph_compute_unchecked(")
    s_since = s.find("ggml_sycl_async_failure_since(sycl_ctx->async_failure_cursor)")
    s_clean = s.find('ggml_backend_sycl_graph_boundary_exception_cleanup(sycl_ctx, "synchronize"')
    s_release = s.find("(void) ggml_sycl_execution_release_graph(sycl_ctx);")
    route = body(sycl, "static void ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx, ggml_tensor * dst) try {")
    bodies = {"d": d, "d_own": d_own, "d_ctx": d_ctx}
    records = {name: record in body(bodies[local], "catch (...)") for name, local, record in DRAIN_RECORDS}
    return records | {
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
        "T the terminal knows only that its work finished": t != "" and "observe_success" not in t and
            "ggml_sycl_async_failure" not in t and "void arm(sycl::event event) noexcept" in t,
        "T ready tests completion only": "command_execution_status" in t_ready and
            "ggml_sycl_async_failure_since" not in t_ready and "wait_and_throw" not in t_ready and
            re.search(r"\bwait\w*\s*\(", t_ready) is None,
        "F the completion task releases a finished bundle": RELEASE in b,
        "F the release asks neither the latch nor a wait": b != "" and "ggml_sycl_async_failure" not in b and
            "observe_success" not in b and re.search(r"\bwait\w*\s*\(", b) is None,
        "B the completion task drops the bundle before it returns": b.endswith(BUNDLE_DROP) and
            0 <= b.find("completion->bundle.terminal_release()") < b.rfind(BUNDLE_DROP),
        "D the helper waits, recovers, then releases": 0 <= d_wait < d_recover < d_release,
        "D the begin-path twin retries before it aborts": 0 <= d_own.find(
            "ggml_sycl_execution_recover_and_release_graph(ctx)") < d_own.find(
            "ggml_sycl_execution_abort_and_release_graph(ctx);"),
        "D the begin-path twin skips the release after a throwing wait": re.search(
            TWIN_CONDITION.format(ctx="ctx"), d_own) is not None,
        "D the context twin skips the release after a throwing wait": re.search(
            TWIN_CONDITION.format(ctx="backend_ctx"), d_ctx) is not None and "waited = false;" in body(
            d_ctx, "catch (...)"),
        "D the context twin retries before it aborts": 0 <= d_ctx.find(
            "ggml_sycl_execution_recover_and_release_graph(backend_ctx)") < d_ctx.find(
            "ggml_sycl_execution_abort_and_release_graph(backend_ctx);"),
        "D the destructor recovers before it releases": 0 <= d_dtor.find(
            "ggml_sycl_recover_exact_mmid_after_drain(this);") < d_dtor.find(
            "ggml_sycl_execution_abort_and_release_graph(this);"),
        "K the test seams are compiled only for route testing": test_seam_guarded(sycl),
        "K the completion failpoint is consumed before the release": 0 <= b.find(NO_RELEASE) < b.find(RELEASE) and
            "constexpr bool skip_release = false;" in b,
        "K the route hands its completion to the seam before it arms the terminal": 0 <= route.find(
            "completion->bundle.mark_possible_submit()") < route.find(RETAIN) < route.find(
            "terminal->arm(final);") < route.find("h.host_task([completion"),
        "T route arms with its final event": "terminal->arm(final);" in sycl,
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
            "terminal judges success again": (cache, sycl.replace(
                "void arm(sycl::event event) noexcept",
                "bool observe_success() noexcept { return !ggml_sycl_async_failure_since({}); }\n"
                "    void arm(sycl::event event) noexcept", 1), common),
            "ready reads the latch": (cache, re.sub(
                r"(bool ready\(\) const noexcept override \{)",
                r"\1 if (ggml_sycl_async_failure_since(cursor_)) return false;", sycl, count=1), common),
            "ready waits": (cache, re.sub(r"(bool ready\(\) const noexcept override \{)",
                                          r"\1 if (event_) event_->wait();", sycl, count=1), common),
            "completion task asks the latch": (cache, sycl.replace(
                RELEASE, RELEASE.replace("!skip_release && ", "!skip_release && !ggml_sycl_async_failure_since({}) && "),
                1), common),
            "completion task waits before it releases": (cache, sycl.replace(
                RELEASE, "sycl::event{}.wait(); " + RELEASE, 1), common),
            "completion failpoint outside the testing guard": (cache, sycl.replace(
                "constexpr bool skip_release = false;",
                "const bool skip_release = ggml_sycl_q1_nvfp4_test_consume_failure(" + NO_RELEASE + ");", 1), common),
            "completion task keeps the bundle": (cache, sycl.replace(BUNDLE_DROP, "", 1), common),
            "completion task drops the bundle first": (cache, sycl.replace(BUNDLE_DROP, "", 1).replace(
                RELEASE, BUNDLE_DROP + " " + RELEASE, 1), common),
            "helper releases without recovering": (cache, sycl.replace(
                "ggml_sycl_recover_exact_mmid_after_drain(ctx);\n    return ggml_sycl_execution_release_graph(ctx);",
                "return ggml_sycl_execution_release_graph(ctx);", 1), common),
            "context twin aborts without the retry": (cache, re.sub(
                r"\s*&&\s*!ggml_sycl_execution_recover_and_release_graph\(backend_ctx\)", "", sycl, count=1), common),
            "begin-path twin releases after a throwing wait": (cache, re.sub(
                r"if \(!waited \|\|\s*(\(!ggml_sycl_execution_release_graph\(ctx\))", r"if (\1", sycl, count=1),
                common),
            "route arms its terminal with no event": (cache, sycl.replace(
                "terminal->arm(final);", "terminal->arm(sycl::event{});", 1), common),
            "context twin releases after a throwing wait": (cache, re.sub(
                r"if \(!waited \|\|\s*(\(!ggml_sycl_execution_release_graph\(backend_ctx\))", r"if (\1", sycl,
                count=1), common),
            "context twin forgets the throwing wait": (cache, sycl.replace(
                'context drain wait threw");', 'context drain wait threw"); waited = true;', 1).replace(
                "                waited = false;\n            }", "            }", 1), common),
            "destructor releases without recovering": (cache, sycl.replace(
                "ggml_sycl_recover_exact_mmid_after_drain(this);", "", 1), common),
            "completion hold outside the testing guard": (cache, sycl.replace(
                "terminal->arm(final);",
                "ggml_sycl_q1_nvfp4_test_retain_completion(completion); terminal->arm(final);", 1),
                common),
            "completion hold under an else branch": (cache, sycl.replace(
                "#ifdef GGML_SYCL_Q1_NVFP4_ROUTE_TESTING\n                            "
                "ggml_sycl_q1_nvfp4_test_retain_completion(completion);",
                "#ifdef GGML_SYCL_Q1_NVFP4_ROUTE_TESTING\n#else\n                            "
                "ggml_sycl_q1_nvfp4_test_retain_completion(completion);", 1), common),
            "route hands over its completion after the submit": (cache, sycl.replace(RETAIN_BLOCK, "", 1).replace(
                BUNDLE_DROP + "\n                                });\n                            });\n",
                BUNDLE_DROP + "\n                                });\n                            });\n" + RETAIN_BLOCK, 1), common),
            "route does not hand over its completion": (cache, sycl.replace(
                "ggml_sycl_q1_nvfp4_test_retain_completion(completion);", "", 1), common),
            **{f"{name}: record removed": (cache, sycl.replace(record, "", 1), common)
               for name, _, record in DRAIN_RECORDS},
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
