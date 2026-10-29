"""Source contract for the ring windows' timing line under GGML_SYCL_MOE_EXEC_TRACE=2 (llama.cpp-8ucb, Task 12 of
docs/plans/2026-10-24-prefill-window-ring.md).

Host-only: reads sources, builds nothing, loads no model and touches no device. test-moe-stream-phase pins what the
timeline records and prints (it links ggml-sycl, so only the lead runs it); this gate pins how the backend feeds it.

What this gate pins, each claim with mutants that must turn it red (its positive controls). Every mutant edits an
in-memory copy of the normalized source, never a file:
  - the line's format literal (adjacent string literals joined) appears exactly once under ggml/src/ggml-sycl,
    outside tests/, inside moe_stream_window_timeline's flush;
  - the line is trace level 2's and nothing else's: begin_layer enables the timeline from the op's decision log at
    `trace >= 2` (the only assignment of `enabled`), each of the timeline's recorders returns at once when it is
    disabled, and flush_window_timing prints only at trace level 2, through the decision log's WARN sink (the trace
    sink is INFO, which llama-bench drops without -v);
  - the cache's copy queue records device times only at the same level: the BCS queue takes enable_profiling in the
    `bcs_profiling` arm alone, and `bcs_profiling` is the level parsed by moe_exec_config_parse, `>= 2`;
  - the executor counts its launches: one per dequant kernel and one per GEMM group of the phase, and hands the count
    to window_launched after the phase's GEMM stage, on a windowed op only;
  - stage_window records each copy with its bytes and its read gate, window_begin records the window with the host
    time it took, and begin_ubatch moves the micro-batch key on;
  - the synchronize boundary flushes the lines after the queue wait, its only caller in ggml-sycl.cpp;
  - docs/backend/sycl-env-vars.md catalogs the literal and says the trace row's level 2 profiles the copy queue.
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
        'copy_us=%.0f readers_wait_us=%.0f launches=%u"')
TIMELINE_SIG = "template <typename event_t> struct moe_stream_window_timeline {"
FLUSH_SIG = "template <typename times_fn, typename sink_fn> size_t flush(int device, times_fn times, sink_fn sink) {"


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


# ---- the trace-level gate -----------------------------------------------------------------------------------------

ENABLE = "window_timing_.enabled = op.log != nullptr && op.log->trace >= 2;"
EARLY = "if (!enabled) { return; }"
RECORDERS = (
    "void copied(uint64_t key, const event_t & copy, size_t bytes, const std::optional<event_t> & gate) {",
    "void begun(uint64_t key, int layer, expert_tensor_role role, size_t phase, bool exposed, double submit_us) {",
    "void launched(uint64_t key, uint32_t n) {",
)
FLUSH_GATE = ("if (log_ == nullptr || log_->trace < 2 || log_->sink == nullptr) { window_timing_.records.clear(); "
              "return; }")
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
    # a recorder that records while disabled
    copied = norm(RECORDERS[0]) + " " + EARLY
    assert not claim_trace_gate(PLAN_CPP, mutate(PLAN_HPP, copied, norm(RECORDERS[0])))
    # the flush printing below level 2, or at INFO through the trace sink
    assert not claim_trace_gate(mutate(PLAN_CPP, "log_->trace < 2", "log_->trace < 1"), PLAN_HPP)
    assert not claim_trace_gate(mutate(PLAN_CPP, FLUSH_SINK, FLUSH_SINK.replace("log->sink(", "log->trace_sink(")),
                                PLAN_HPP)


# ---- the copy queue's profiling -----------------------------------------------------------------------------------

BCS_LEVEL = ("const bool bcs_profiling = moe_exec_config_parse([](const char * name) -> const char * { return "
             "std::getenv(name); }).trace >= 2;")
BCS_RE = re.compile(
    r"bcs_queue_ = bcs_profiling \? std::make_unique<sycl::queue>\([^;]*sycl::property::queue::enable_profiling"
    r"[^;]*\) : std::make_unique<sycl::queue>\((?:(?!enable_profiling)[^;])*\);")


def claim_bcs_profiling(uc_cpp: str) -> bool:
    """The BCS queue takes enable_profiling only in the bcs_profiling arm, and bcs_profiling is trace level 2 parsed
    through moe_exec_config_parse; no other queue in the file takes the property."""
    n = norm(uc_cpp)
    return n.count(BCS_LEVEL) == 1 and len(BCS_RE.findall(n)) == 1 and n.count("enable_profiling") == 1


def test_bcs_profiling():
    assert claim_bcs_profiling(UC_CPP)


def test_bcs_profiling_mutants():
    # another level, or always
    assert not claim_bcs_profiling(mutate(UC_CPP, BCS_LEVEL, BCS_LEVEL.replace(">= 2", ">= 1")))
    assert not claim_bcs_profiling(mutate(UC_CPP, "bcs_queue_ = bcs_profiling ?", "bcs_queue_ = true ?"))
    # the property made unconditional: both arms profile
    plain = ("std::make_unique<sycl::queue>(queue_.get_context(), queue_.get_device(), "
             "sycl::property::queue::in_order{});")
    profiled = ("std::make_unique<sycl::queue>(queue_.get_context(), queue_.get_device(), sycl::property_list{ "
                "sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{} });")
    n = norm(UC_CPP)
    at = n.find(": " + plain, n.find("bcs_queue_ = bcs_profiling ?"))
    assert at >= 0, "mutant precondition: the plain arm follows the profiling one"
    assert not claim_bcs_profiling(n[:at] + ": " + profiled + n[at + 2 + len(plain):])


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

COPIED = ("window_timing_.copied(moe_stream_window_timing_key(ubatch_seq_, window_id), copied, item.src_bytes, "
          "reuse_gate);")
BEGUN = ("if (timed) { const double submit_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::"
         "now() - t_begin).count(); window_timing_.begun(moe_stream_window_timing_key(ubatch_seq_, wid), req.layer, "
         "req.role, req.phase, exposed, submit_us); }")
CLOCK = "const auto t_begin = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();"


def claim_records(plan_cpp: str) -> bool:
    """stage_window records each copy after it is submitted, with its bytes and read gate; window_begin starts its
    clock before any staging and records the window last; begin_ubatch moves the micro-batch key on."""
    n = norm(plan_cpp)
    stage = body(n, "moe_stream_phase::stage_counts moe_stream_phase::stage_window(")
    wb = body(n, "uint32_t moe_stream_phase::window_begin(")
    ub = body(n, "void moe_stream_phase::begin_ubatch(uint32_t n_tokens) {")
    copy_at = stage.find("copied = mem_copy_async(")
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


def claim_flush_site(sycl: str) -> bool:
    """The synchronize boundary flushes after its queue wait, and nothing else in the file flushes."""
    n = norm(sycl)
    sync = body(n, "static void ggml_backend_sycl_synchronize(ggml_backend_t backend) {")
    return (sync.count(FLUSH_CALL) == 1 and 0 <= sync.find(WAIT) < sync.find(FLUSH_CALL)
            and n.count("flush_window_timing()") == 1)


def test_flush_site():
    assert claim_flush_site(SYCL)


def test_flush_site_mutants():
    n = norm(SYCL)
    sync = body(n, "static void ggml_backend_sycl_synchronize(ggml_backend_t backend) {")
    head = "const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);"
    assert sync.count(" " + FLUSH_CALL) == 1 and sync.count(head) == 1, "mutant precondition: one flush, one stream"
    # before the wait, where reading an event's device times would wait on the compute work
    moved = sync.replace(" " + FLUSH_CALL, "").replace(head, head + " " + FLUSH_CALL)
    assert not claim_flush_site(n.replace(sync, moved))
    # a second flush elsewhere, e.g. in the graph's compute path
    assert not claim_flush_site(norm(SYCL) + " void g(x * sycl_ctx) { sycl_ctx->moe_stream->flush_window_timing(); }")


# ---- the catalog --------------------------------------------------------------------------------------------------

def claim_catalog(doc: str) -> bool:
    """The catalog names the literal (its wrapped lines read as one) and the trace row says level 2 profiles the copy
    queue."""
    flat = re.sub(r"\s+", " ", doc)
    row = re.search(r"(?m)^\| `GGML_SYCL_MOE_EXEC_TRACE=[^`]*` \|[^\n]*$", doc)
    return LINE.strip('"') in flat and row is not None and "enable_profiling" in row.group(0)


def test_catalog():
    assert claim_catalog(ENV_DOC)


def test_catalog_mutants():
    assert not claim_catalog(ENV_DOC.replace("readers_wait_us=%.0f", "wait_us=%.0f"))
    row = re.search(r"(?m)^\| `GGML_SYCL_MOE_EXEC_TRACE=[^`]*` \|[^\n]*$", ENV_DOC)
    assert row is not None, "mutant precondition: the trace row exists"
    assert not claim_catalog(ENV_DOC.replace(row.group(0), row.group(0).replace("enable_profiling", "profiling")))


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
