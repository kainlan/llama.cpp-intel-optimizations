"""Source contract for the ring windows' timing line under GGML_SYCL_MOE_EXEC_TRACE=2 (llama.cpp-8ucb, Task 12 of
docs/plans/2026-10-24-prefill-window-ring.md; the timing source moved off the copies in llama.cpp-ivb6).

Host-only: reads sources, builds nothing, loads no model and touches no device. test-moe-stream-phase pins what the
timeline records and prints (it links ggml-sycl, so only the lead runs it); this gate pins how the backend feeds it.

Why the copies are not timed: with the copy (BCS) queue built with enable_profiling at trace level 2, its events were
counter-based timestamp events, and a traced Qwen3.8 prefill on the B70 stalled in its first micro-batch, at a window
that varied from run to run, until the watchdog ended the run, while the same run untraced finished. The stall is below
this backend, in the runtime or the driver, and tied to no copy count or layer. The window line now reads empty marker
kernels on a profiling queue of the phase's own, which wait on the copies from outside, and no copy event is read for
its device times or asked for its status.

What this gate pins, each claim with mutants that must turn it red (its positive controls). Every mutant edits an
in-memory copy of the normalized source, never a file:
  - the line's format literal (adjacent string literals joined) appears exactly once under ggml/src/ggml-sycl,
    outside tests/, inside moe_stream_window_timeline's flush;
  - the line is trace level 2's and nothing else's: begin_layer enables the timeline from the op's decision log at
    `trace >= 2` (the only assignment of `enabled`), each of the timeline's recorders returns at once when it is
    disabled, and flush_window_timing prints only at trace level 2, through the decision log's WARN sink (the trace
    sink is INFO, which llama-bench drops without -v);
  - the cache's copy queue has no profiling at any level: the BCS queue is built in order only, by one assignment,
    with no trace switch, no default_queue_properties() and no enable_profiling anywhere in unified-cache.cpp;
  - the marks: stage_window, at trace level 2 only, submits the window's free and start marks just before its first
    copy (after the read gate is chosen, before the gate joins the copy's dependencies) and its end mark after the
    loop, behind the last copy. The free mark waits on the copy's dependencies without the read gate plus the copy
    queued before it, the start mark on the same plus the gate. Every mark is an empty kernel
    (ggml_sycl_submit_marker) on timing_queue(), an in-order profiling queue built lazily from the compute queue's
    context and device, and dropped when prepare() takes another compute queue;
  - the timeline reads device times only from the marks and the read gate: flush asks end_of() of exactly the start
    mark, the end mark, the gate and the free mark, the record holds no copy event, and the event reader asks for
    command_start and command_end only;
  - copy_us is the end mark's end after the start mark's end; readers_wait_us is the gate's end after the free mark's
    end (clamped at 0), the wait FOR the gate; start_lag_us is the start mark's end after the gate's end, a separate
    field. Only a window's first copy is marked, copied() moves the queue's last copy on, and a flush forgets it;
  - the executor counts its launches: one per dequant kernel and one per GEMM group of the phase, and hands the count
    to window_launched after the phase's GEMM stage, on a windowed op only;
  - stage_window records each copy with its bytes, window_begin records the window with the host time it took, and
    begin_ubatch moves the micro-batch key on;
  - the synchronize boundary flushes the lines after the queue wait and before the expert-loan runner, its only
    caller in ggml-sycl.cpp, with no lock taken in synchronize ahead of the flush nor in flush_window_timing;
  - docs/backend/sycl-env-vars.md catalogs the literal; its trace row says the copy queue is never profiled, names the
    marks and the timing queue, records both changes of meaning (the wait counts the region's gate and the host
    entry's ready event; copy_us includes the timing queue's latency at both ends), and settles the status-poll
    caveat (the reaper's poll runs only at context admission); the window line's bullet describes the marks, not the
    copies' command_submit or command_start;
  - the trace knob's own comment in moe-exec-policy.hpp names the window line among level 2's output and says the
    copy queue is never profiled;
  - the cost below level 2 stays one branch per site: window_begin reads the clock only when the timeline is enabled,
    stage_window marks only when it is, flush_window_timing returns below level 2 before it prints, and the phase's
    destructor flushes what no synchronize did.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL_DIR = ROOT / "ggml" / "src" / "ggml-sycl"

_LEXEME_RE = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'",
    re.DOTALL,
)


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def norm(text: str) -> str:
    """Comment-stripped, whitespace-canonical text with adjacent string literals joined, so a clang-format re-wrap
    cannot turn a claim red. Idempotent, so a mutant built on normalized text reads the same as the source."""
    t = re.sub(r"\s+", " ", strip_comments(text))
    t = re.sub(r"\( ", "(", t)
    t = re.sub(r" \)", ")", t)
    t = re.sub(r" ,", ",", t)
    t = re.sub(r",(?! )", ", ", t)
    t = re.sub(r'(?<!\\)" "', "", t)
    return t


def read(rel: str) -> str:
    """The file's text, or empty when it does not exist, so a missing file fails the claims instead of erroring."""
    p = ROOT / rel
    return p.read_text(errors="replace") if p.exists() else ""


SYCL = read("ggml/src/ggml-sycl/ggml-sycl.cpp")
PLAN_CPP = read("ggml/src/ggml-sycl/moe-stream-plan.cpp")
PLAN_HPP = read("ggml/src/ggml-sycl/moe-stream-plan.hpp")
UC_CPP = read("ggml/src/ggml-sycl/unified-cache.cpp")
ENV_DOC = read("docs/backend/sycl-env-vars.md")
POLICY_HPP = read("ggml/src/ggml-sycl/moe-exec-policy.hpp")


def body(text: str, signature: str) -> str:
    """The brace-balanced body of the first definition whose text starts with `signature` (string literals skipped
    while balancing). A signature ending in "{" opens the body itself. Empty when there is none."""
    at = text.find(signature)
    if at < 0:
        return ""
    open_at = at + len(signature) - 1 if signature.endswith("{") else text.find("{", at + len(signature))
    if open_at < 0:
        return ""
    depth = 0
    i = open_at
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "\"'":
            j = i + 1
            while j < n and text[j] != ch:
                j += 2 if text[j] == "\\" else 1
            i = j + 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_at:i + 1]
        i += 1
    return ""


def mutate(text: str, old: str, new: str, count: int = 1) -> str:
    """`text` normalized, with `old` replaced; `old` must occur exactly `count` times, or the mutant proves nothing."""
    n = norm(text)
    assert n.count(old) == count, f"mutant precondition: {old!r} occurs {n.count(old)} times, not {count}"
    return n.replace(old, new)


# ---- the literal --------------------------------------------------------------------------------------------------

LINE = ('"[MOE-EXEC-TRACE] window device %d layer %d role %s phase %zu bytes=%zu exposed=%d submit_us=%.0f '
        'copy_us=%.0f readers_wait_us=%.0f start_lag_us=%.0f launches=%u"')
TIMELINE_SIG = "template <typename event_t> struct moe_stream_window_timeline {"
FLUSH_SIG = "template <typename end_fn, typename sink_fn> size_t flush(int device, end_fn end_of, sink_fn sink) {"


def sources():
    for p in sorted(SYCL_DIR.rglob("*")):
        if p.suffix in (".cpp", ".hpp", ".h") and "tests" not in p.relative_to(SYCL_DIR).parts:
            yield p


def claim_literal(texts: dict) -> bool:
    """The literal appears once over every backend source, and that once is the timeline's flush in the plan header."""
    total = sum(norm(t).count(LINE) for t in texts.values())
    hpp = norm(texts.get("moe-stream-plan.hpp", ""))
    flush = body(body(hpp, TIMELINE_SIG), FLUSH_SIG)
    return total == 1 and flush.count(LINE) == 1


def backend_texts() -> dict:
    return {p.name: p.read_text(errors="replace") for p in sources()}


def test_literal_once():
    assert claim_literal(backend_texts())


def test_literal_mutants():
    texts = backend_texts()
    # a second print of the line elsewhere
    doubled = dict(texts)
    doubled["ggml-sycl.cpp"] = norm(texts["ggml-sycl.cpp"]) + " void f() { std::printf(" + LINE + "); }"
    assert not claim_literal(doubled)
    # a field renamed, which the gate script and the catalog would no longer match
    renamed = dict(texts)
    renamed["moe-stream-plan.hpp"] = mutate(texts["moe-stream-plan.hpp"], "readers_wait_us=%.0f", "wait_us=%.0f")
    assert not claim_literal(renamed)
    # the lag field dropped, folding the two figures back into one
    dropped = dict(texts)
    dropped["moe-stream-plan.hpp"] = mutate(texts["moe-stream-plan.hpp"], " start_lag_us=%.0f", "")
    assert not claim_literal(dropped)


# ---- the trace-level gate -----------------------------------------------------------------------------------------

ENABLE = "window_timing_.enabled = op.log != nullptr && op.log->trace >= 2;"
EARLY = "if (!enabled) { return; }"
RECORDERS = (
    "void marked(uint64_t key, const event_t & free_mark, const event_t & start_mark, "
    "const std::optional<event_t> & gate) {",
    "void copied(uint64_t key, const event_t & copy, size_t bytes) {",
    "void ended(uint64_t key, const event_t & end_mark) {",
    "void begun(uint64_t key, int layer, expert_tensor_role role, size_t phase, bool exposed, double submit_us) {",
    "void launched(uint64_t key, uint32_t n) {",
)
FLUSH_GATE = "if (log_ == nullptr || log_->trace < 2 || log_->sink == nullptr) { window_timing_.clear(); return; }"
FLUSH_SINK = "[log](const char * line) { log->sink(log->sink_ctx, line); }"


def claim_trace_gate(plan_cpp: str, plan_hpp: str) -> bool:
    """Level 2 enables the timeline (its one assignment, in begin_layer); every recorder returns at once when disabled;
    the flush prints only at level 2, through the WARN sink."""
    cpp = norm(plan_cpp)
    hpp = norm(plan_hpp)
    begin_layer = body(cpp, "moe_exec_decision moe_stream_phase::begin_layer(")
    timeline = body(hpp, TIMELINE_SIG)
    flush = body(cpp, "void moe_stream_phase::flush_window_timing() {")
    recorders_gated = all(body(timeline, sig).startswith("{ " + EARLY) for sig in RECORDERS)
    return (begin_layer.count(ENABLE) == 1 and cpp.count("window_timing_.enabled =") == 1 and recorders_gated
            and flush.count(FLUSH_GATE) == 1 and flush.count(FLUSH_SINK) == 1 and "trace_sink" not in flush)


def test_trace_gate():
    assert claim_trace_gate(PLAN_CPP, PLAN_HPP)


def test_trace_gate_mutants():
    # the line at level 1, or always
    assert not claim_trace_gate(mutate(PLAN_CPP, ENABLE, ENABLE.replace(">= 2", ">= 1")), PLAN_HPP)
    assert not claim_trace_gate(mutate(PLAN_CPP, ENABLE, "window_timing_.enabled = true;"), PLAN_HPP)
    # a recorder that records while disabled, one mutant per recorder
    for sig in RECORDERS:
        gated = norm(sig) + " " + EARLY
        assert not claim_trace_gate(PLAN_CPP, mutate(PLAN_HPP, gated, norm(sig))), sig
    # the flush printing below level 2, or at INFO through the trace sink
    assert not claim_trace_gate(mutate(PLAN_CPP, "log_->trace < 2", "log_->trace < 1"), PLAN_HPP)
    assert not claim_trace_gate(mutate(PLAN_CPP, FLUSH_SINK, FLUSH_SINK.replace("log->sink(", "log->trace_sink(")),
                                PLAN_HPP)


# ---- the copy queue has no profiling ------------------------------------------------------------------------------

BCS_PLAIN = ("bcs_queue_ = std::make_unique<sycl::queue>(queue_.get_context(), queue_.get_device(), "
             "sycl::property::queue::in_order{});")
BCS_ASSIGN_RE = re.compile(r"\bbcs_queue_ =(?!=)")


def claim_bcs_plain(uc_cpp: str) -> bool:
    """The BCS queue is built in order only, by its one assignment: no trace level switches it, it is not built from
    default_queue_properties() (which carries enable_profiling), and no queue in the file takes enable_profiling."""
    n = norm(uc_cpp)
    return (n.count(BCS_PLAIN) == 1 and len(BCS_ASSIGN_RE.findall(n)) == 1 and "enable_profiling" not in n
            and "bcs_profiling" not in n and "moe_exec_config_parse(" not in n)


def test_bcs_plain():
    assert claim_bcs_plain(UC_CPP)


def test_bcs_plain_mutants():
    # the trace-level-2 arm this gate replaced: profiling when GGML_SYCL_MOE_EXEC_TRACE >= 2
    switch = ("const bool bcs_profiling = moe_exec_config_parse([](const char * name) -> const char * { return "
              "std::getenv(name); }).trace >= 2; ")
    traced = ("bcs_queue_ = bcs_profiling ? std::make_unique<sycl::queue>(queue_.get_context(), queue_.get_device(), "
              "sycl::property_list{ sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{} }) : "
              "std::make_unique<sycl::queue>(queue_.get_context(), queue_.get_device(), "
              "sycl::property::queue::in_order{});")
    assert not claim_bcs_plain(mutate(UC_CPP, BCS_PLAIN, switch + traced))
    # profiling at every level
    profiled = BCS_PLAIN.replace("sycl::property::queue::in_order{});", "sycl::property_list{ "
                                 "sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{} });")
    assert not claim_bcs_plain(mutate(UC_CPP, BCS_PLAIN, profiled))
    # the backend's standard properties, which carry enable_profiling out of sight
    assert not claim_bcs_plain(mutate(UC_CPP, BCS_PLAIN, BCS_PLAIN.replace("sycl::property::queue::in_order{}",
                                                                           "default_queue_properties()")))
    # a second assignment that could rebuild it another way
    assert not claim_bcs_plain(mutate(UC_CPP, BCS_PLAIN, BCS_PLAIN + " bcs_queue_ = nullptr;"))


# ---- the marks ----------------------------------------------------------------------------------------------------

STAGE_SIG = "moe_stream_phase::stage_counts moe_stream_phase::stage_window("
TIMING_KEY = "const uint64_t timing_key = moe_stream_window_timing_key(ubatch_seq_, window_id);"
REUSE_TAKE = ("if (refill_after_readers && !reuse_gate) { reuse_gate = region_reads_.read_gate[cur_region_] ? "
              "*region_reads_.read_gate[cur_region_] : ccs_->ext_oneapi_submit_barrier(); }")
MARK_AT = "if (window_timing_.enabled && n.staged == 0) { mark_window_start(timing_key, deps, reuse_gate); }"
REUSE_DEP = "if (reuse_gate) { deps.push_back(*reuse_gate); }"
COPY_DECL = "copied = mem_copy_async("
LOOP_END = "++n.staged; }"
END_AT = "if (window_timing_.enabled && n.staged != 0) { mark_window_end(timing_key, region_last_); }"
MARK_KERNEL = "class moe_stream_timing_mark;"
MARKER = "ggml_sycl_submit_marker<moe_stream_timing_mark>("
START_SIG = ("void moe_stream_phase::mark_window_start(uint64_t key, std::vector<sycl::event> deps, "
             "const std::optional<sycl::event> & gate) {")
START_BODY = ("{ if (window_timing_.queue_tail) { deps.push_back(*window_timing_.queue_tail); } "
              "sycl::queue & q = timing_queue(); "
              "const sycl::event free_mark = ggml_sycl_submit_marker<moe_stream_timing_mark>(q, deps); "
              "if (gate) { deps.push_back(*gate); } "
              "const sycl::event start_mark = ggml_sycl_submit_marker<moe_stream_timing_mark>(q, deps); "
              "window_timing_.marked(key, free_mark, start_mark, gate); }")
END_SIG = "void moe_stream_phase::mark_window_end(uint64_t key, const sycl::event & last_copy) {"
END_BODY = ("{ window_timing_.ended(key, ggml_sycl_submit_marker<moe_stream_timing_mark>(timing_queue(), "
            "{ last_copy })); }")
TQ_SIG = "sycl::queue & moe_stream_phase::timing_queue() {"
TQ_BODY = ("{ if (!timing_queue_) { timing_queue_.emplace(ccs_->get_context(), ccs_->get_device(), "
           "default_queue_properties()); } return *timing_queue_; }")
PREPARE_SIG = "void moe_stream_phase::prepare("
NEW_CCS = "if (!same_queue) { ccs_.emplace(ccs); timing_queue_.reset();"
TQ_MEMBER = "std::optional<sycl::queue> timing_queue_;"


def ordered(text: str, *parts: str) -> bool:
    """Each part occurs in `text`, each after the one before it."""
    at = -1
    for part in parts:
        at = text.find(part, at + 1)
        if at < 0:
            return False
    return True


def claim_marks(plan_cpp: str, plan_hpp: str) -> bool:
    """stage_window marks a window at trace level 2 only: its free and start marks just before its first copy, after
    the read gate is chosen and before the gate joins the copy's dependencies, so the free mark misses only the gate;
    its end mark after the loop, behind its last copy. The free mark adds the copy queued before (the copy queue is in
    order), the start mark the gate. Every mark is an empty kernel on timing_queue(): an in-order profiling queue on
    the compute queue's context and device, built at the first mark and dropped when prepare() takes a new compute
    queue. The two helpers are called once each, from stage_window, and timing_queue() only from them."""
    n = norm(plan_cpp)
    stage = body(n, STAGE_SIG)
    prepare = body(n, PREPARE_SIG)
    return (stage.count(TIMING_KEY) == 1 and stage.count(MARK_AT) == 1 and stage.count(END_AT) == 1
            and ordered(stage, TIMING_KEY, REUSE_TAKE, MARK_AT, REUSE_DEP, COPY_DECL, LOOP_END, END_AT, "return n;")
            and n.count("mark_window_start(") == 2 and n.count("mark_window_end(") == 2
            and body(n, START_SIG) == START_BODY and body(n, END_SIG) == END_BODY
            and n.count(MARK_KERNEL) == 1 and n.count(MARKER) == 3
            and body(n, TQ_SIG) == TQ_BODY and n.count("timing_queue_.emplace(") == 1 and n.count("timing_queue()") == 3
            and prepare.count(NEW_CCS) == 1 and n.count("timing_queue_.reset();") == 1
            and norm(plan_hpp).count(TQ_MEMBER) == 1)


def test_marks():
    assert claim_marks(PLAN_CPP, PLAN_HPP)


def test_marks_mutants():
    # the start marks after the copy, which ends them late by the window's own submit loop
    late = mutate(PLAN_CPP, " " + MARK_AT, "")
    copied = "window_timing_.copied(timing_key, copied, item.src_bytes);"
    assert late.count(copied) == 1, "mutant precondition: the copy is recorded once"
    assert not claim_marks(late.replace(copied, copied + " " + MARK_AT), PLAN_HPP)
    # the start marks after the gate joined the deps: the free mark would wait for the gate too
    gated = mutate(PLAN_CPP, MARK_AT + " " + REUSE_DEP, REUSE_DEP + " " + MARK_AT)
    assert not claim_marks(gated, PLAN_HPP)
    # the end mark inside the loop, behind every copy
    inside = mutate(PLAN_CPP, " " + END_AT, "")
    assert not claim_marks(inside.replace(copied, copied + " " + END_AT), PLAN_HPP)
    # the free mark carrying the gate, or missing the copy queued before
    start = norm(START_BODY)
    both = start.replace("if (gate) { deps.push_back(*gate); } ", "").replace(
        "sycl::queue & q = timing_queue(); ", "sycl::queue & q = timing_queue(); if (gate) { deps.push_back(*gate); } ")
    assert not claim_marks(mutate(PLAN_CPP, START_BODY, both), PLAN_HPP)
    no_tail = start.replace("if (window_timing_.queue_tail) { deps.push_back(*window_timing_.queue_tail); } ", "")
    assert not claim_marks(mutate(PLAN_CPP, START_BODY, no_tail), PLAN_HPP)
    # the marks on the copy queue, or on the compute queue (the readers would wait for the prefetch copies)
    assert not claim_marks(mutate(PLAN_CPP, "sycl::queue & q = timing_queue();", "sycl::queue & q = *ccs_;"),
                           PLAN_HPP)
    assert not claim_marks(mutate(PLAN_CPP, "(timing_queue(), { last_copy })", "(*ccs_, { last_copy })"), PLAN_HPP)
    # the timing queue without profiling (its marks would have no device times)
    assert not claim_marks(mutate(PLAN_CPP, "default_queue_properties()", "sycl::property::queue::in_order{}"),
                           PLAN_HPP)
    # a timing queue kept across a new compute queue, whose events it cannot wait on
    assert not claim_marks(mutate(PLAN_CPP, " timing_queue_.reset();", ""), PLAN_HPP)
    # a barrier instead of the marker kernel
    assert not claim_marks(mutate(PLAN_CPP, "ggml_sycl_submit_marker<moe_stream_timing_mark>(q, deps); if (gate)",
                                  "q.ext_oneapi_submit_barrier(deps); if (gate)"), PLAN_HPP)


# ---- what the timeline reads --------------------------------------------------------------------------------------

END_OF_RE = re.compile(r"\bend_of\(([^,()]+), &")
READS = ["r.start_mark", "r.end_mark", "*r.reuse_gate", "r.free_mark"]
EVENT_SIG = "bool moe_stream_event_end(const sycl::event & ev, uint64_t * end_ns) {"
FLUSH_CALL_PLAN = "window_timing_.flush(device_, moe_stream_event_end, "


def claim_reads_marks_only(plan_hpp: str, plan_cpp: str) -> bool:
    """The flush asks end_of() of the start mark, the end mark, the read gate and the free mark, in that order and
    nothing else; the record keeps no copy event; the phase's event reader asks for command_start and command_end, never
    command_submit, and is the flush's reader."""
    timeline = body(norm(plan_hpp), TIMELINE_SIG)
    flush = body(timeline, FLUSH_SIG)
    reader = body(norm(plan_cpp), EVENT_SIG)
    return (END_OF_RE.findall(flush) == READS and "times(" not in timeline
            and not any(k in timeline for k in ("first_copy", "last_copy", "prev_copy", "get_profiling_info"))
            and reader.count("command_start>()") == 1 and reader.count("command_end>()") == 1
            and "command_submit" not in reader and norm(plan_cpp).count(FLUSH_CALL_PLAN) == 1)


def test_reads_marks_only():
    assert claim_reads_marks_only(PLAN_HPP, PLAN_CPP)


def test_reads_marks_only_mutants():
    # the wait measured from the start mark, which already waited for the gate (the wait would always read 0)
    assert not claim_reads_marks_only(mutate(PLAN_HPP, "end_of(r.free_mark, &free_end)",
                                             "end_of(r.start_mark, &free_end)"), PLAN_CPP)
    # the copy span started at the free mark, so the gate's wait would count as copy time
    assert not claim_reads_marks_only(mutate(PLAN_HPP, "end_of(r.start_mark, &start_end)",
                                             "end_of(r.free_mark, &start_end)"), PLAN_CPP)
    # a copy's own event read again, through a field kept for it
    copy_field = mutate(PLAN_HPP, "event_t end_mark{};", "event_t end_mark{}; event_t last_copy{};")
    assert not claim_reads_marks_only(copy_field, PLAN_CPP)
    # the reader asking for the submit time again
    start = "start_ns = ev.get_profiling_info<sycl::info::event_profiling::command_start>();"
    assert not claim_reads_marks_only(PLAN_HPP, mutate(PLAN_CPP, start, start + " (void) ev.get_profiling_info<"
                                                       "sycl::info::event_profiling::command_submit>();"))


# ---- the figures --------------------------------------------------------------------------------------------------

START_TIMED = "const bool start_timed = r.n_copies != 0 && r.start_marked && end_of(r.start_mark, &start_end);"
COPY_EXPR = ("if (start_timed && r.end_marked && end_of(r.end_mark, &stop_end) && stop_end >= start_end) { "
             "copy_us = static_cast<double>(stop_end - start_end) / 1e3; }")
GATE_TIMED = "const bool gate_timed = start_timed && r.reuse_gate && end_of(*r.reuse_gate, &gate_end);"
WAIT_IF = "if (gate_timed && end_of(r.free_mark, &free_end)) {"
WAIT_EXPR = "readers_wait_us = gate_end > free_end ? static_cast<double>(gate_end - free_end) / 1e3 : 0.0;"
LAG_EXPR = "start_lag_us = start_end > gate_end ? static_cast<double>(start_end - gate_end) / 1e3 : 0.0;"
MARKED_BODY = ("{ if (!enabled) { return; } record & r = at(key); if (r.n_copies == 0 && !r.start_marked) { "
               "r.start_marked = true; r.free_mark = free_mark; r.start_mark = start_mark; r.reuse_gate = gate; } }")
COPIED_BODY = "{ if (!enabled) { return; } record & r = at(key); queue_tail = copy; ++r.n_copies; r.bytes += bytes; }"
ENDED_BODY = "{ if (!enabled) { return; } record & r = at(key); r.end_marked = true; r.end_mark = end_mark; }"
CLEAR = "void clear() { records.clear(); queue_tail.reset(); }"
FLUSH_END = "} clear(); return printed; }"


def claim_figures(plan_hpp: str) -> bool:
    """copy_us is the end mark's end after the start mark's end; readers_wait_us is the gate's end after the free mark's
    end, clamped at 0 (the wait FOR the gate), measured only when the gate and the free mark have times; start_lag_us
    is the start mark's end after the gate's end, a field of its own. Each figure is assigned once beyond its -1.
    Only a window's first copy is marked, copied() moves the queue's last copy on, a later stage's end mark replaces
    the earlier one, and the flush ends with the timeline's own clear(), which forgets the queue's last copy."""
    timeline = body(norm(plan_hpp), TIMELINE_SIG)
    flush = body(timeline, FLUSH_SIG)
    return (flush.count(START_TIMED) == 1 and flush.count(COPY_EXPR) == 1 and flush.count(GATE_TIMED) == 1
            and flush.count(WAIT_IF) == 1 and flush.count(WAIT_EXPR) == 1 and flush.count(LAG_EXPR) == 1
            and ordered(flush, START_TIMED, COPY_EXPR, GATE_TIMED, WAIT_IF, WAIT_EXPR, LAG_EXPR)
            and flush.count("copy_us =") == 2 and flush.count("readers_wait_us =") == 2
            and flush.count("start_lag_us =") == 2
            and body(timeline, RECORDERS[0]) == MARKED_BODY and body(timeline, RECORDERS[1]) == COPIED_BODY
            and body(timeline, RECORDERS[2]) == ENDED_BODY
            and timeline.count(CLEAR) == 1 and flush.endswith(FLUSH_END) and flush.count("clear();") == 1)


def test_figures():
    assert claim_figures(PLAN_HPP)


def test_figures_mutants():
    # the operands swapped: the free mark after the gate, which reads 0 whenever the gate held the copy
    swapped = WAIT_EXPR.replace("gate_end > free_end", "free_end > gate_end").replace("gate_end - free_end",
                                                                                      "free_end - gate_end")
    assert not claim_figures(mutate(PLAN_HPP, WAIT_EXPR, swapped))
    # readers_wait_us measured as the lag
    assert not claim_figures(mutate(PLAN_HPP, WAIT_EXPR, LAG_EXPR.replace("start_lag_us =", "readers_wait_us =")))
    # the lag and the copy span folded together, or the span reversed
    assert not claim_figures(mutate(PLAN_HPP, "stop_end - start_end", "stop_end - gate_end"))
    # a later copy's marks replacing the first copy's
    assert not claim_figures(mutate(PLAN_HPP, "if (r.n_copies == 0 && !r.start_marked) {", "if (true) {"))
    # the queue's last copy never moved on, or never forgotten at the flush
    assert not claim_figures(mutate(PLAN_HPP, " queue_tail = copy;", ""))
    assert not claim_figures(mutate(PLAN_HPP, CLEAR, "void clear() { records.clear(); }"))
    assert not claim_figures(mutate(PLAN_HPP, FLUSH_END, "} records.clear(); return printed; }"))
    # an end mark that never lands, so copy_us would always read -1
    assert not claim_figures(mutate(PLAN_HPP, " r.end_marked = true;", ""))


# ---- the executor's launch count ----------------------------------------------------------------------------------

PHASE_LOOP = "for (size_t p = 0; p < phases.size() && !gemm_declined; ++p) {"
DECL = "uint32_t phase_launches = 0;"
DEQUANT = "dequant_expert_f16(active_experts[i], phase_weights + (i - phase.begin) * weight_elems); ++phase_launches;"
GEMM = "half_readers[phase.buffer].push_back(*group_event); ++phase_launches;"
HANDOFF = ("} catch (const std::exception & e) { return gemm_stage_failed(e.what()); } if (pp_stream_windowed) { "
           "ctx.moe_stream->window_launched(pp_stream_layer, pp_stream_role, p, phase_launches); } }")


def claim_launches(sycl: str) -> bool:
    """The phase counts each dequant kernel and each GEMM group it submits, and hands the count to window_launched
    after its GEMM stage, on a windowed op, once in the file."""
    n = norm(sycl)
    loop = body(n, PHASE_LOOP)
    return (loop.count(DECL) == 1 and loop.count(DEQUANT) == 1 and loop.count(GEMM) == 1
            and loop.count("++phase_launches;") == 2 and loop.endswith(HANDOFF)
            and n.count("window_launched(") == 1)


def test_launches():
    assert claim_launches(SYCL)


def test_launches_mutants():
    # the count hard-coded
    assert not claim_launches(mutate(SYCL, "p, phase_launches);", "p, 2u);"))
    # a launch kind not counted
    assert not claim_launches(mutate(SYCL, DEQUANT, DEQUANT.replace(" ++phase_launches;", "")))
    assert not claim_launches(mutate(SYCL, GEMM, GEMM.replace(" ++phase_launches;", "")))
    # handed over before the GEMM stage counted its groups
    call = " if (pp_stream_windowed) { ctx.moe_stream->window_launched(pp_stream_layer, pp_stream_role, p, " \
           "phase_launches); }"
    moved = mutate(SYCL, call, "")
    done = "ctx.moe_stream->window_done(pp_stream_layer, pp_stream_role, p, readers_done); }"
    assert moved.count(done) == 1, "mutant precondition: window_done is called once"
    assert not claim_launches(moved.replace(done, done + call))


# ---- what the phase records ---------------------------------------------------------------------------------------

COPIED = "window_timing_.copied(timing_key, copied, item.src_bytes);"
BEGUN = ("if (timed) { const double submit_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::"
         "now() - t_begin).count(); window_timing_.begun(moe_stream_window_timing_key(ubatch_seq_, wid), req.layer, "
         "req.role, req.phase, exposed, submit_us); }")
CLOCK = "const auto t_begin = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();"


def claim_records(plan_cpp: str) -> bool:
    """stage_window records each copy after it is submitted, with its bytes; window_begin starts its clock before any
    staging and records the window last; begin_ubatch moves the micro-batch key on."""
    n = norm(plan_cpp)
    stage = body(n, STAGE_SIG)
    wb = body(n, "uint32_t moe_stream_phase::window_begin(")
    ub = body(n, "void moe_stream_phase::begin_ubatch(uint32_t n_tokens) {")
    copy_at = stage.find(COPY_DECL)
    clock_at = wb.find(CLOCK)
    return (stage.count(COPIED) == 1 and 0 <= copy_at < stage.find(COPIED) and wb.count(BEGUN) == 1
            and 0 <= clock_at < wb.find("stage_or_abort(items, req.role, req.phase)")
            and wb.find("log_->note_window(") < wb.find(BEGUN) and ub.count("++ubatch_seq_;") == 1)


def test_records():
    assert claim_records(PLAN_CPP)


def test_records_mutants():
    assert not claim_records(mutate(PLAN_CPP, COPIED, COPIED.replace("item.src_bytes", "0")))
    assert not claim_records(mutate(PLAN_CPP, COPIED, ""))
    assert not claim_records(mutate(PLAN_CPP, BEGUN, BEGUN.replace("exposed, submit_us", "false, submit_us")))
    assert not claim_records(mutate(PLAN_CPP, "++ubatch_seq_;", ""))


# ---- the flush site -----------------------------------------------------------------------------------------------

FLUSH_CALL = "if (err == 0 && sycl_ctx->moe_stream) { sycl_ctx->moe_stream->flush_window_timing(); }"
WAIT = "CHECK_TRY_ERROR(stream->wait_and_throw());"
RUNNER = "const bool loan_accepted = ggml_sycl_run_pending_expert_loan(backend);"
SYNC_SIG = "static void ggml_backend_sycl_synchronize(ggml_backend_t backend) {"
FLUSH_FN_SIG = "void moe_stream_phase::flush_window_timing() {"
LOCKS = ("lock_guard", "unique_lock", "scoped_lock", ".lock()")


def claim_flush_site(sycl: str, plan_cpp: str) -> bool:
    """The synchronize boundary flushes after its queue wait (every mark of a begun window then follows only finished
    work, so reading its device times does not wait on compute work) and before the expert-loan runner, nothing else in
    the file flushes, and no lock is taken on the way to the flush (in synchronize ahead of it) or inside it."""
    n = norm(sycl)
    sync = body(n, SYNC_SIG)
    at = sync.find(FLUSH_CALL)
    ahead = sync[:at]
    flush_fn = body(norm(plan_cpp), FLUSH_FN_SIG)
    return (sync.count(FLUSH_CALL) == 1 and 0 <= sync.find(WAIT) < at < sync.find(RUNNER)
            and n.count("flush_window_timing()") == 1
            and not any(k in ahead for k in LOCKS) and flush_fn != "" and not any(k in flush_fn for k in LOCKS))


def test_flush_site():
    assert claim_flush_site(SYCL, PLAN_CPP)


def test_flush_site_mutants():
    n = norm(SYCL)
    sync = body(n, SYNC_SIG)
    head = "const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);"
    assert sync.count(" " + FLUSH_CALL) == 1 and sync.count(head) == 1, "mutant precondition: one flush, one stream"
    assert sync.count(RUNNER) == 1, "mutant precondition: one loan runner"
    # before the wait, where reading a mark's device times would wait on the compute work
    moved = sync.replace(" " + FLUSH_CALL, "").replace(head, head + " " + FLUSH_CALL)
    assert not claim_flush_site(n.replace(sync, moved), PLAN_CPP)
    # after the loan runner
    after = sync.replace(" " + FLUSH_CALL, "").replace(RUNNER, RUNNER + " " + FLUSH_CALL)
    assert not claim_flush_site(n.replace(sync, after), PLAN_CPP)
    # a lock taken in synchronize ahead of the flush
    locked = sync.replace(head, head + " std::lock_guard<std::mutex> guard(g_sycl_graph_compute_mutex);")
    assert not claim_flush_site(n.replace(sync, locked), PLAN_CPP)
    # a lock taken inside the flush
    in_flush = mutate(PLAN_CPP, FLUSH_FN_SIG, FLUSH_FN_SIG + " std::lock_guard<std::mutex> g(m);")
    assert not claim_flush_site(SYCL, in_flush)
    # a second flush elsewhere, e.g. in the graph's compute path
    assert not claim_flush_site(norm(SYCL) + " void g(x * sycl_ctx) { sycl_ctx->moe_stream->flush_window_timing(); }",
                                PLAN_CPP)


# ---- the catalog --------------------------------------------------------------------------------------------------

TRACE_ROW_RE = re.compile(r"(?m)^\| `GGML_SYCL_MOE_EXEC_TRACE=[^`]*` \|[^\n]*$")
ROW_SAYS = ("never profiled", "marker kernel", "timing queue", "region's gate", "ready event",
            "latency at both ends", "`release_retained_referencing`", "context admission")
ROW_NEVER = ("reaper polls", "builds its copy (BCS) queue with `enable_profiling`")
BULLET_START = "- WARN `[MOE-EXEC-TRACE] window device"
BULLET_SAYS = ("free mark", "start mark", "end mark")
BULLET_NEVER = ("command_submit", "copy's `command_start`")


def bullet_span(doc: str) -> tuple:
    """Where the catalog's bullet for the window line starts and ends: to the next bullet or blank line."""
    at = doc.find(BULLET_START)
    if at < 0:
        return (-1, -1)
    m = re.compile(r"\n(?:- |\n)").search(doc, at + len(BULLET_START))
    return (at, m.start() if m else len(doc))


def window_bullet(doc: str) -> str:
    """The catalog's bullet for the window line, flattened; empty when there is none."""
    at, end = bullet_span(doc)
    return re.sub(r"\s+", " ", doc[at:end]) if at >= 0 else ""


def in_bullet(doc: str, old: str, new: str) -> str:
    """`doc` with every `old` in the window line's bullet replaced. `old` must be there, or the mutant proves
    nothing."""
    at, end = bullet_span(doc)
    assert at >= 0 and old in doc[at:end], f"mutant precondition: the bullet says {old!r}"
    return doc[:at] + doc[at:end].replace(old, new) + doc[end:]


def claim_catalog(doc: str) -> bool:
    """The catalog names the literal (its wrapped lines read as one). The trace row says the copy queue is never
    profiled, names the marker kernels and the timing queue, records both changes of meaning (the wait counts the
    region's gate and the host entry's ready event; copy_us includes the timing queue's latency at both ends), and
    settles the status-poll caveat: the reaper's poll (release_retained_referencing) runs only at context admission.
    The window line's bullet describes the three marks, not the copies' submit or start times."""
    flat = re.sub(r"\s+", " ", doc)
    row = TRACE_ROW_RE.search(doc)
    bullet = window_bullet(doc)
    return (LINE.strip('"') in flat and row is not None and all(k in row.group(0) for k in ROW_SAYS)
            and not any(k in row.group(0) for k in ROW_NEVER) and all(k in bullet for k in BULLET_SAYS)
            and not any(k in bullet for k in BULLET_NEVER))


def test_catalog():
    assert claim_catalog(ENV_DOC)


def test_catalog_mutants():
    assert not claim_catalog(ENV_DOC.replace("readers_wait_us=%.0f", "wait_us=%.0f"))
    row = TRACE_ROW_RE.search(ENV_DOC)
    assert row is not None, "mutant precondition: the trace row exists"
    for k in ROW_SAYS:
        assert k in row.group(0), f"mutant precondition: the row says {k!r}"
        assert not claim_catalog(ENV_DOC.replace(row.group(0), row.group(0).replace(k, "x"))), k
    # the old caveat back in the row
    assert not claim_catalog(ENV_DOC.replace(row.group(0), row.group(0) + " The retained-handle reaper polls the "
                                             "ring's copy events."))
    # a mark dropped from the window line's bullet, or the copies' submit time back in it
    for k in BULLET_SAYS:
        assert not claim_catalog(in_bullet(ENV_DOC, k, "first copy")), k
    assert not claim_catalog(in_bullet(ENV_DOC, "the free mark's", "the first copy's `command_submit`"))


# ---- the cost below level 2 ---------------------------------------------------------------------------------------

TIMED = "const bool timed = window_timing_.enabled;"
DTOR_SIG = "moe_stream_phase::~moe_stream_phase() {"


def claim_off_cost(plan_cpp: str) -> bool:
    """window_begin times a window only when the timeline is enabled; stage_window marks only then (the timing queue is
    built at the first mark, so below level 2 it never exists); flush_window_timing's level gate comes before its
    print; the phase's destructor flushes the windows no synchronize flushed."""
    cpp = norm(plan_cpp)
    begin = body(cpp, "uint32_t moe_stream_phase::window_begin(")
    stage = body(cpp, STAGE_SIG)
    flush = body(cpp, FLUSH_FN_SIG)
    dtor = body(cpp, DTOR_SIG)
    return (begin.count(TIMED) == 1 and cpp.count("const bool timed =") == 1
            and stage.count("if (window_timing_.enabled && n.staged == 0) { mark_window_start(") == 1
            and stage.count("if (window_timing_.enabled && n.staged != 0) { mark_window_end(") == 1
            and 0 <= flush.find(FLUSH_GATE) < flush.find(FLUSH_CALL_PLAN) and dtor.count("flush_window_timing();") == 1)


def test_off_cost():
    assert claim_off_cost(PLAN_CPP)


def test_off_cost_mutants():
    # the clock read on every window, traced or not
    assert not claim_off_cost(mutate(PLAN_CPP, TIMED, "const bool timed = true;"))
    # the marks (and the timing queue) at every level
    assert not claim_off_cost(mutate(PLAN_CPP, "if (window_timing_.enabled && n.staged == 0)", "if (n.staged == 0)"))
    assert not claim_off_cost(mutate(PLAN_CPP, "if (window_timing_.enabled && n.staged != 0)", "if (n.staged != 0)"))
    # the level gate after the print: lines below level 2
    cpp = norm(PLAN_CPP)
    flush = body(cpp, FLUSH_FN_SIG)
    assert flush.count(" " + FLUSH_GATE) == 1 and flush.endswith("); }"), "mutant precondition: one gate, one print"
    gate_after = flush.replace(" " + FLUSH_GATE, "")[:-1] + FLUSH_GATE + " }"
    assert not claim_off_cost(cpp.replace(flush, gate_after))
    # the destructor no longer flushing: the last prompt's windows would never print
    dtor = body(cpp, DTOR_SIG)
    assert dtor.count(" flush_window_timing();") == 1, "mutant precondition: the destructor flushes"
    assert not claim_off_cost(cpp.replace(dtor, dtor.replace(" flush_window_timing();", "")))


# ---- the knob's own comment ---------------------------------------------------------------------------------------

def claim_knob_comment(policy_hpp: str) -> bool:
    """The comment above `int trace` (the lines from "// 0 off;" to the member) says level 2 also prints the window
    timing line and that the copy queue is never profiled, so the comment, the catalog and the code name the same
    output and the same queue."""
    m = re.search(r"((?:[ \t]*//[^\n]*\n)+)[ \t]*int\s+trace\b", policy_hpp)
    if m is None:
        return False
    comment = " ".join(line.strip().lstrip("/").strip() for line in m.group(1).splitlines())
    level2 = comment.split("; 2 also", 1)
    return (comment.startswith("0 off;") and len(level2) == 2 and "[MOE-EXEC-TRACE] window" in level2[1]
            and "never profiled" in level2[1] and "with profiling" not in comment)


def test_knob_comment():
    assert claim_knob_comment(POLICY_HPP)


def test_knob_comment_mutants():
    assert "[MOE-EXEC-TRACE] window" in POLICY_HPP, "mutant precondition: the comment names the line"
    # the window line dropped from the comment
    assert not claim_knob_comment(POLICY_HPP.replace("[MOE-EXEC-TRACE] window", "a window line"))
    # the window line attributed to level 1 instead (level 2 lists only the probe)
    moved = POLICY_HPP.replace("; 2 also the load-time copy probe and", "; and")
    assert moved != POLICY_HPP, "mutant precondition: the level-2 clause exists"
    assert not claim_knob_comment(moved)
    # the copy queue's profiling back in the comment
    assert "never profiled" in POLICY_HPP, "mutant precondition: the comment says the copy queue is never profiled"
    assert not claim_knob_comment(POLICY_HPP.replace("never profiled", "built with profiling"))


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
