#!/usr/bin/env python3
"""Every retained-handle record carries the gates of every CPU expert job a device command may still wait on
(llama.cpp-42zj).

Companion of test-sycl-moe-cpu-job-device-signal-source.py (the jobs' own device events, the gates, and the host
waits that sleep on them, among them the drain worker's sleep on a record's gates: its J5),
test-sycl-cpu-job-signal-book.cpp (the gate list, host-only) and test-sycl-cpu-job-signal-ring.cpp (the device
behaviour, lead-run, including a record published with no scope that carries a ring's gate).

THE DEADLOCK.  With the device waiting a CPU expert job's own host-signalled event, a host wait on an event behind
that job can take libze's lock-and-wait path: it locks a device-level mutex and holds it while it polls the device's
progress (libze_intel_gpu.so.1.17.39758: the lock at 0x9b6ace, the wait loop around 0x9b6ca4, no unlock between).
The job's prologue then checked its input events through the same driver path and needed that mutex before it could
run and signal (it now waits a device-written input marker; test-sycl-cpu-job-input-wait-source.py).  On the B70 the retained-handle drain worker held it while waiting an ungated record's event at the first
host-tiered MUL_MAT_ID of Qwen3.8's first 256-token prompt graph, so the job never started and nothing returned (4 of
5 plain runs on 97db9565e).  The drain worker did sleep on a record's gates first, but only the CPU-job scatter flush
gave records gates, through a scope of its own; every other publication gave the worker none.  One fact, two
sources: the host waits read the rings' gate lists, the drain worker read the publisher's scope.

THE CONTRACT.  With comments blanked:
  G1. One source, every record.  publish_handles_until_event declares its record's gates as
      `std::vector<std::shared_future<void>> gates = cpu_job_signal_gates_outstanding();`, once, at the top level of
      its body, after the graph-lifetime return and before the store lock, and names them once more, to move them
      into the one record it pushes under that lock: no other use, so nothing can assign, change or move them away
      before the push.  No other source exists: no ggml-sycl source
      names retained_handle_gate_scope or g_retained_handle_gates (comments included, since they would describe a
      source that is gone), the backend never names ws.gates, and moe_host_scatter_workspace has no gates member.
  G2. Every drained record is built there.  In mem-handle.cpp state.queue grows only by two push_back calls: the
      publication's, and release_retained_referencing's return of the records it kept, which keep their gates.
  G3. The snapshot.  cpu_job_signal_gates_outstanding() (unified-cache.cpp, declared in cpu-job-signal.hpp) declares
      its live-ring vector before a bare block that takes the registry's timed_mutex with one lock_guard (never a
      try lock or a deferred unique_lock, which could skip every ring) and does nothing else there but copy each live
      ring into that vector, so a last owner is dropped only after the lock is released (a ring's destructor prunes
      the registry under it).  After the block it appends every ring's gates with append_outstanding, one statement
      per ring and no break, and returns them.  Nothing in it waits, takes a gate's value, queries a status or calls
      the driver, and nothing in it makes a ring.
  G4. cpu_job_gate_list::append_outstanding reserves once and copies every gate's future under the list's lock, in
      push order, and does nothing else: nothing appended, and nothing allocated, for an empty list.
  G5. Completeness.  unified-cache.cpp asserts statically that a gate list holds at least as many gates as a ring
      has entries, and the two constants agree (max_gates >= max_events): one gate is pushed per wrapped entry and a
      recycle clears its entries' gates, so the list's drop of its oldest gate never runs.  The refuted claim that a
      gate decides only how a host thread waits, "never whether", is gone from cpu-job-signal.hpp.
  G6. Every ring is in the registry the snapshot reads.  cpu_job_signal_ring::create is the only place a ring is
      constructed (one `new cpu_job_signal_ring(` in ggml-sycl, no make_shared), its one successful return comes
      right after cpu_job_signal_registry_add(ring), and that function pushes the ring under the registry's lock. A
      ring left out of the registry would hold gates no record takes.

--self-test applies one mutant per row, requires each to fail on its own pin, and requires the unmodified tree to
pass.

RED on 97db9565e, before llama.cpp-42zj:
  FAIL [pin G1]: a publication does not take its record's gates from cpu_job_signal_gates_outstanding() in one
  declaration (llama.cpp-42zj)
"""

from __future__ import annotations

import functools
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SYCL = ROOT / "ggml" / "src" / "ggml-sycl"
BACKEND = SYCL / "ggml-sycl.cpp"
MEM_CPP = SYCL / "mem-handle.cpp"
MEM_HPP = SYCL / "mem-handle.hpp"
SIGNAL_HPP = SYCL / "cpu-job-signal.hpp"
CACHE_CPP = SYCL / "unified-cache.cpp"
TAG = "(llama.cpp-42zj)"


class ContractError(Exception):
    pass


def fail(pin: str, msg: str) -> None:
    raise ContractError(f"FAIL [pin {pin}]: {msg} {TAG}")


# One token at a time, leftmost first: a // or /* */ comment (an unterminated one runs to the end), or a string or
# character literal, which is kept so that a comment marker inside it stays. A quote right after a letter or digit is a
# digit separator, not a literal.
_BLANK_TOKEN = re.compile(r'//[^\n]*|/\*.*?(?:\*/|\Z)|"(?:\\.|[^"\\])*(?:"|\Z)|(?<![^\W_])\'(?:\\.|[^\'\\])*(?:\'|\Z)', re.S)
_NOT_NEWLINE = re.compile(r"[^\n]")


def _blank_token(m: re.Match) -> str:
    t = m.group(0)
    if t[0] != "/":
        return t
    return " " * len(t) if t[1] == "/" else _NOT_NEWLINE.sub(" ", t)


@functools.lru_cache(maxsize=16)
def blank_comments(src: str) -> str:
    """Blank // and /* */ comments, keeping string literals, line breaks and offsets."""
    return _BLANK_TOKEN.sub(_blank_token, src)


@functools.lru_cache(maxsize=16)
def strip_strings(src: str) -> str:
    return re.sub(r'"(?:\\.|[^"\\])*"', lambda m: '"' + " " * (len(m.group(0)) - 2) + '"', src)


def block_at(code: str, brace: int, pin: str) -> tuple[str, int]:
    """The text inside the `{` at `brace` and the offset of its first character."""
    flat_code = strip_strings(code)
    depth = 0
    for j in range(brace, len(flat_code)):
        if flat_code[j] == "{":
            depth += 1
        elif flat_code[j] == "}":
            depth -= 1
            if depth == 0:
                return code[brace + 1 : j], brace + 1
    fail(pin, "unbalanced braces")
    return "", 0


def body_of(code: str, signature: str, pin: str) -> tuple[str, int]:
    m = re.search(signature, code)
    if not m:
        fail(pin, f"no definition matching {signature!r}")
    return block_at(code, code.index("{", m.end() - 1), pin)


def flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def enclosing_brace(code: str, pos: int) -> int:
    """The offset of the innermost `{` still open at `pos`, or -1 at the top level of `code`."""
    flat_code = strip_strings(code)
    depth = 0
    for i in range(pos - 1, -1, -1):
        c = flat_code[i]
        if c == "}":
            depth += 1
        elif c == "{":
            if depth == 0:
                return i
            depth -= 1
    return -1


def header_of(code: str, brace: int) -> str:
    """The text between the statement before `brace` and the brace: an if, for or try header, or "" for a bare
    block."""
    flat_code = strip_strings(code)
    k = brace - 1
    while k >= 0 and flat_code[k] not in ";{}":
        k -= 1
    return re.sub(r"\s+", " ", code[k + 1 : brace]).strip()


PUBLISH_SIG = r"static\s+void\s+publish_handles_until_event\s*\([^)]*\)\s*\{"
RELEASE_SIG = (r"retained_reap_result\s+release_retained_referencing\s*\(\s*const\s+retained_reap_request\s*&\s*request"
               r"\s*\)\s*\{")
SNAP_SIG = r"std::vector<std::shared_future<void>>\s+cpu_job_signal_gates_outstanding\s*\(\s*\)\s*\{"
GATE_LIST_SIG = r"class\s+cpu_job_gate_list\s*\{"
APPEND_SIG = r"void\s+append_outstanding\s*\(\s*std::vector<std::shared_future<void>>\s*\*\s*out\s*\)\s*const\s*\{"
WORKSPACE_SIG = r"struct\s+moe_host_scatter_workspace\s*\{"
CREATE_SIG = (r"static\s+std::shared_ptr<cpu_job_signal_ring>\s+create\s*\(\s*const\s+sycl::queue\s*&\s*q\s*,"
              r"\s*const\s+char\s*\*\*\s*why\s*\)\s*\{")
REG_ADD_SIG = (r"static\s+void\s+cpu_job_signal_registry_add\s*\(\s*const\s+std::shared_ptr<cpu_job_signal_ring>\s*&"
               r"\s*ring\s*\)\s*\{")

SNAP_STMT = "std::vector<std::shared_future<void>> gates = cpu_job_signal_gates_outstanding();"
GRAPH_RETURN = "retain_handles_for_current_graph(std::move(handles)); return; }"
STORE_LOCK = "retained_store_lock lock(state);"
PUSH_PREFIX = "state.queue.push_back({ std::move(handles), std::move(event), std::move(ids), std::move(gates)"
KEPT_PUSH = "state.queue.push_back(std::move(record));"
SNAP_DECL = "std::vector<std::shared_future<void>> cpu_job_signal_gates_outstanding();"
LIVE_DECL = "std::vector<std::shared_ptr<cpu_job_signal_ring>> live;"
REG_LOCK = "std::lock_guard<std::timed_mutex> lock(r.mutex);"
REG_BLOCK = ("std::lock_guard<std::timed_mutex> lock(r.mutex); live.reserve(r.rings.size()); "
             "for (const std::weak_ptr<cpu_job_signal_ring> & w : r.rings) { "
             "if (std::shared_ptr<cpu_job_signal_ring> ring = w.lock()) { live.push_back(std::move(ring)); } }")
APPEND_LOOP = ("std::vector<std::shared_future<void>> gates; "
               "for (const std::shared_ptr<cpu_job_signal_ring> & ring : live) { "
               "ring->gates().append_outstanding(&gates); } return gates;")
APPEND_BODY = ("std::lock_guard<std::mutex> lock(mutex_); out->reserve(out->size() + gates_.size()); "
               "for (const gate & g : gates_) { out->push_back(g.future); }")
ASSERT = "static_assert(cpu_job_gate_list::max_gates >= cpu_job_signal_ring::max_events,"
REFUTED = "never whether"
SECOND_SOURCES = ("retained_handle_gate_scope", "g_retained_handle_gates")
NOT_IN_SNAPSHOT = ((r"\bwait(?:_and_throw|_for|_until)?\s*\(", "waits"),
                   (r"\.\s*get\s*\(\s*\)", "takes a gate's value"),
                   (r"\bget_info\s*<", "queries a status"),
                   (r"\bze[A-Z]\w*\s*\(", "calls the driver"),
                   (r"\bget_cpu_job_signal_ring\s*\(", "makes a ring"),
                   (r"\bcreate(?:_for_testing)?\s*\(", "makes a ring"))
TRY_LOCKS = r"\b(?:try_lock\w*|unique_lock|defer_lock)\b"


def check_publish(s: dict) -> None:
    mem = blank_comments(s["mem_cpp"])
    body, _ = body_of(mem, PUBLISH_SIG, "G1")
    f = flat(body)
    if f.count(SNAP_STMT) != 1:
        fail("G1", "a publication does not take its record's gates from cpu_job_signal_gates_outstanding() in one "
                   "declaration")
    if enclosing_brace(body, body.find("cpu_job_signal_gates_outstanding")) != -1:
        fail("G1", "the publication's gate snapshot is under a condition or inside a block")
    g, r, lk = f.find(SNAP_STMT), f.find(GRAPH_RETURN), f.find(STORE_LOCK)
    if r < 0 or lk < 0 or not (r < g < lk):
        fail("G1", "the gate snapshot is not taken after the graph-lifetime return and before the store lock")
    if len(re.findall(r"\bgates\b", strip_strings(body))) != 2:
        fail("G1", "the publication names its gates other than in the snapshot's declaration and the record's push, so "
                   "they can be assigned, changed or moved away before the push")
    if f.count(PUSH_PREFIX) != 1 or f.find(PUSH_PREFIX) < lk:
        fail("G1", "the record pushed under the store lock does not carry the snapshot's gates")


def check_one_source(s: dict) -> None:
    scan = dict(s["others"])
    scan.update({"ggml-sycl.cpp": s["backend"], "mem-handle.cpp": s["mem_cpp"], "mem-handle.hpp": s["mem_hpp"],
                 "cpu-job-signal.hpp": s["signal_hpp"], "unified-cache.cpp": s["cache_cpp"]})
    for name in sorted(scan):
        text = scan[name]
        for token in SECOND_SOURCES:
            # A substring test first: the boundary regex is slow over the backend's 6 MB and rarely needed.
            if token in text and re.search(r"(?<!\w)" + token + r"(?!\w)", text):
                fail("G1", f"{name} names {token}, a second source of a record's gates")
    backend = blank_comments(s["backend"])
    if re.search(r"\bws\s*\.\s*gates\b", backend):
        fail("G1", "the backend's flush keeps its own gates (ws.gates), a second source of a record's gates")
    wb, _ = body_of(backend, WORKSPACE_SIG, "G1")
    if re.search(r"\bgates\s*(?:;|=|\{)", wb):
        fail("G1", "moe_host_scatter_workspace keeps a gates member, a second source of a record's gates")


def check_queue_pushes(s: dict) -> None:
    mem = blank_comments(s["mem_cpp"])
    grows = re.findall(r"\bstate\s*\.\s*queue\s*\.\s*(push_back|push_front|emplace_back|emplace_front|emplace|insert)"
                       r"\s*\(", mem)
    if grows != ["push_back", "push_back"]:
        fail("G2", f"state.queue grows by {grows}, not exactly by the publication's push and the reap's return of the "
                   "records it kept")
    rb, _ = body_of(mem, RELEASE_SIG, "G2")
    pb, _ = body_of(mem, PUBLISH_SIG, "G2")
    if flat(rb).count(KEPT_PUSH) != 1 or flat(pb).count("state.queue.push_back(") != 1:
        fail("G2", "a record reaches the drain worker's queue other than through the publication or a kept record's "
                   "return")


def check_snapshot(s: dict) -> None:
    if SNAP_DECL not in flat(blank_comments(s["signal_hpp"])):
        fail("G3", "cpu-job-signal.hpp does not declare cpu_job_signal_gates_outstanding()")
    cc = blank_comments(s["cache_cpp"])
    body, _ = body_of(cc, SNAP_SIG, "G3")
    f = flat(body)
    bare = strip_strings(body)
    for pattern, what in NOT_IN_SNAPSHOT:
        if re.search(pattern, bare):
            fail("G3", f"the snapshot {what}")
    if re.search(TRY_LOCKS, bare):
        fail("G3", "the snapshot takes the registry lock with a try lock or a deferred unique_lock, which can skip "
                   "every ring")
    if f.count(REG_LOCK) != 1:
        fail("G3", "the snapshot does not take the registry lock with one lock_guard")
    lk = body.find("std::lock_guard<std::timed_mutex>")
    brace = enclosing_brace(body, lk)
    if brace < 0 or header_of(body, brace) != "" or enclosing_brace(body, brace) != -1:
        fail("G3", "the registry lock is not held in a bare block of its own at the top of the snapshot")
    block, boff = block_at(body, brace, "G3")
    if flat(block).strip() != REG_BLOCK:
        fail("G3", "the registry lock covers more than copying each live ring, or misses a ring")
    live = f.find(LIVE_DECL)
    if live < 0 or live > len(flat(body[:brace])):
        fail("G3", "the live rings are not declared before the registry lock, so a last owner could be dropped under it")
    if flat(body[boff + len(block) + 1 :]).strip() != APPEND_LOOP:
        fail("G3", "after the registry lock the snapshot does not append every live ring's gates and return them")


def check_append(s: dict) -> None:
    hpp = blank_comments(s["signal_hpp"])
    gl, _ = body_of(hpp, GATE_LIST_SIG, "G4")
    ab, _ = body_of(gl, APPEND_SIG, "G4")
    if flat(ab).strip() != APPEND_BODY:
        fail("G4", "cpu_job_gate_list::append_outstanding does more than copy every gate's future under the list's "
                   "lock")


def check_complete(s: dict) -> None:
    cc = flat(blank_comments(s["cache_cpp"]))
    if ASSERT not in cc:
        fail("G5", "unified-cache.cpp does not assert that a gate list holds a gate for every entry its ring can have "
                   "in use")
    mg = re.search(r"static constexpr size_t max_gates = (\d+);", flat(blank_comments(s["signal_hpp"])))
    me = re.search(r"static constexpr uint32_t max_events = (\d+);", cc)
    if not mg or not me or int(mg.group(1)) < int(me.group(1)):
        fail("G5", "a gate list can hold fewer gates than its ring has entries, so a push could drop a running job's "
                   "gate")
    if REFUTED in s["signal_hpp"]:
        fail("G5", "cpu-job-signal.hpp still says a gate decides only how a host thread waits, never whether")


def check_registered(s: dict) -> None:
    cc = blank_comments(s["cache_cpp"])
    body, _ = body_of(cc, CREATE_SIG, "G6")
    f = flat(body).strip()
    if not f.endswith("cpu_job_signal_registry_add(ring); return ring;") or f.count("return ring;") != 1:
        fail("G6", "cpu_job_signal_ring::create can return a ring it did not add to the registry")
    sites = []
    named = [(p.name, s[k]) for k, p in (("backend", BACKEND), ("mem_cpp", MEM_CPP), ("mem_hpp", MEM_HPP),
                                         ("signal_hpp", SIGNAL_HPP), ("cache_cpp", CACHE_CPP))]
    for name, text in sorted(list(s["others"].items()) + named):
        code = strip_strings(blank_comments(text))
        sites += [name] * len(re.findall(r"\bnew\s+cpu_job_signal_ring\s*\(", code))
        if re.search(r"\b(?:make_shared|make_unique|allocate_shared)\s*<\s*cpu_job_signal_ring\s*>", code):
            fail("G6", f"{name} makes a ring outside cpu_job_signal_ring::create")
    if sites != ["unified-cache.cpp"] or not re.search(r"\bnew\s+cpu_job_signal_ring\s*\(", body):
        fail("G6", f"a ring is constructed outside cpu_job_signal_ring::create ({sites})")
    ab, _ = body_of(cc, REG_ADD_SIG, "G6")
    af = flat(ab)
    lk, push = af.find("std::lock_guard<std::timed_mutex> lock(r.mutex);"), af.find("r.rings.push_back(ring);")
    if lk < 0 or push < lk:
        fail("G6", "cpu_job_signal_registry_add does not push the ring under the registry's lock")


def read_sources(root: Path) -> dict:
    s = {}
    for key, path in (("backend", BACKEND), ("mem_cpp", MEM_CPP), ("mem_hpp", MEM_HPP), ("signal_hpp", SIGNAL_HPP),
                      ("cache_cpp", CACHE_CPP)):
        p = root / path.relative_to(ROOT)
        s[key] = p.read_text(encoding="utf-8") if p.exists() else ""
    named = {p.name for p in (BACKEND, MEM_CPP, MEM_HPP, SIGNAL_HPP, CACHE_CPP)}
    others = {}
    for p in sorted((root / SYCL.relative_to(ROOT)).glob("*.[ch]pp")):
        if p.name not in named:
            others[p.name] = p.read_text(encoding="utf-8", errors="replace")
    s["others"] = others
    return s


def check(s: dict) -> None:
    check_publish(s)
    check_one_source(s)
    check_queue_pushes(s)
    check_snapshot(s)
    check_append(s)
    check_complete(s)
    check_registered(s)


def self_test(s: dict) -> int:
    def sub(pattern: str, repl: str, flags: int = 0):
        return lambda text: re.sub(pattern, repl, text, count=1, flags=flags)

    snap_line = r"\n\s*std::vector<std::shared_future<void>> gates = cpu_job_signal_gates_outstanding\(\);"
    mutants = [
        ("G1", "mem_cpp", "the snapshot is taken only under a condition",
         sub(r"std::vector<std::shared_future<void>> gates = cpu_job_signal_gates_outstanding\(\);",
             "std::vector<std::shared_future<void>> gates;\n    if (!ids.empty()) {\n"
             "        gates = cpu_job_signal_gates_outstanding();\n    }")),
        ("G1", "mem_cpp", "the snapshot is taken under the store lock",
         lambda t: re.sub(r"(retained_store_lock lock\(state\);\n)(\s*)(state\.queue\.push_back\(\{ std::move\(handles\))",
                          r"\1\2std::vector<std::shared_future<void>> gates = cpu_job_signal_gates_outstanding();\n\2\3",
                          re.sub(snap_line, "", t, count=1), count=1)),
        ("G1", "mem_cpp", "the record is pushed without the snapshot's gates",
         sub(r"std::move\(ids\), std::move\(gates\)", "std::move(ids), {}")),
        ("G1", "mem_cpp", "a thread's own gate pointer is back",
         sub(r"(thread_local graph_recording_sink_state g_graph_recording_sink;\n)",
             r"\1thread_local const std::vector<std::shared_future<void>> * g_retained_handle_gates = nullptr;\n")),
        ("G1", "mem_hpp", "the publisher's gate scope is back",
         sub(r"(\n// The drain worker's waits since the last take)", r"\nclass retained_handle_gate_scope;\n\1")),
        ("G1", "backend", "the flush's workspace keeps gates again",
         sub(r"(std::vector<sycl::event>\s+joins;  // each slot's CPU job join \(llama\.cpp-z4kd\))",
             r"\1\n    std::vector<std::shared_future<void>>   gates;")),
        ("G1", "mem_cpp", "the record ships an emptied vector",
         sub(r"(\n(\s*)std::vector<std::shared_future<void>> gates = cpu_job_signal_gates_outstanding\(\);)",
             r"\1\n\2auto stolen = std::move(gates);")),
        ("G6", "cache_cpp", "a ring is made but not registered",
         sub(r"\n\s*cpu_job_signal_registry_add\(ring\);\n(\s*return ring;)", r"\n\1")),
        ("G6", "cache_cpp", "the registry does not keep the ring",
         sub(r"\n\s*r\.rings\.push_back\(ring\);\n", "\n")),
        ("G6", "signal_hpp", "a header makes a ring",
         sub(r"(\nstd::vector<std::shared_future<void>> cpu_job_signal_gates_outstanding\(\);)",
             r"\1\ninline std::shared_ptr<cpu_job_signal_ring> cpu_job_signal_ring_make_unregistered() "
             r"{ return std::make_shared<cpu_job_signal_ring>(); }")),
        ("G6", "cache_cpp", "a ring is constructed outside create",
         sub(r"(\n(\s*)auto\s+r\s+= cpu_job_signal_ring::create\(q, &why\);)",
             r"\n\2(void) std::shared_ptr<cpu_job_signal_ring>(new cpu_job_signal_ring(q.get_context(), nullptr, "
             r"cpu_job_signal_l0_get()));\1")),
        ("G2", "mem_cpp", "a record is queued outside the publication",
         sub(r"(\nvoid retain_handles_until_event\(std::vector<mem_handle> handles, sycl::event event\) \{\n)",
             r"\1    if (handles.empty()) {\n        auto & state = *g_retained_handles_state;\n"
             r"        state.queue.push_back({ {}, event, {}, {} });\n    }\n")),
        ("G3", "cache_cpp", "the registry lock is held across the gate lists",
         sub(r"\n(\s*)\{\n(\s*)std::lock_guard<std::timed_mutex> lock\(r\.mutex\);\n(\s*)live\.reserve",
             r"\n\1std::lock_guard<std::timed_mutex> lock(r.mutex);\n\1{\n\3live.reserve")),
        ("G3", "cache_cpp", "the snapshot stops after the first ring",
         sub(r"(ring->gates\(\)\.append_outstanding\(&gates\);)", r"\1\n        break;")),
        ("G3", "cache_cpp", "the snapshot waits a gate",
         sub(r"(append_outstanding\(&gates\);\n(\s*)\}\n)(\s*)(return gates;)",
             r"\1\3for (const std::shared_future<void> & g : gates) {\n\3    g.wait();\n\3}\n\3\4")),
        ("G3", "cache_cpp", "the snapshot can skip the registry",
         sub(r"std::lock_guard<std::timed_mutex> lock\(r\.mutex\);\n(\s*)live\.reserve",
             r"std::unique_lock<std::timed_mutex> lock(r.mutex, std::defer_lock);\n"
             r"\1if (!lock.try_lock_for(std::chrono::milliseconds(200))) {\n\1    return {};\n\1}\n\1live.reserve")),
        ("G4", "signal_hpp", "append_outstanding polls a gate",
         sub(r"(\n(\s*)out->push_back\(g\.future\);)", r"\1\n\2(void) g.future.wait_for(std::chrono::seconds(0));")),
        ("G4", "signal_hpp", "append_outstanding copies only the newest gate",
         sub(r"for \(const gate & g : gates_\) \{\n(\s*)out->push_back\(g\.future\);\n(\s*)\}",
             r"if (!gates_.empty()) {\n\1out->push_back(gates_.back().future);\n\2}")),
        ("G5", "cache_cpp", "no completeness assert",
         sub(r"static_assert\(cpu_job_gate_list::max_gates >= cpu_job_signal_ring::max_events,\s*\"[^\"]*\"\);\n", "")),
        ("G5", "signal_hpp", "the list caps below the ring's entries",
         sub(r"static constexpr size_t max_gates = 1024;", "static constexpr size_t max_gates = 512;")),
        ("G5", "signal_hpp", "the refuted claim is back",
         sub(r"(\nclass cpu_job_gate_list \{)", r"\n// A gate decides only how a host thread waits, never whether.\1")),
    ]
    failures, caught = [], 0
    for pin, key, name, mutate in mutants:
        m = dict(s)
        m[key] = mutate(s[key])
        if m[key] == s[key]:
            failures.append(f"mutant '{name}' did not apply")
            print(f"  self-test: mutant '{name}' did not apply")
            continue
        try:
            check(m)
        except ContractError as e:
            if f"[pin {pin}]" in str(e):
                caught += 1
                print(f"  self-test: mutant '{name}' caught: {e}")
                continue
            failures.append(f"mutant '{name}' failed on the wrong pin: {e}")
            print(f"  self-test: mutant '{name}' failed on the wrong pin: {e}")
            continue
        failures.append(f"mutant '{name}' passed")
        print(f"  self-test: mutant '{name}' PASSED (not caught)")
    try:
        check(s)
        print("  self-test: unmodified tree passes")
    except ContractError as e:
        failures.append(f"unmodified tree fails: {e}")
        print(f"  self-test: unmodified tree FAILS: {e}")
    if failures:
        print(f"SELF-TEST FAIL: {', '.join(failures)}")
        return 1
    print(f"SELF-TEST PASS: {caught} mutants caught, unmodified tree passes")
    return 0


def main(argv: list[str]) -> int:
    root = ROOT
    if "--root" in argv:
        root = Path(argv[argv.index("--root") + 1]).resolve()
    s = read_sources(root)
    if "--self-test" in argv:
        rc = self_test(s)
        if rc != 0:
            return rc
    try:
        check(s)
    except ContractError as e:
        print(e)
        return 1
    print("PASS: every retained record takes the gates of every live ring's outstanding CPU expert jobs, from one "
          f"snapshot that waits on nothing, and the gate lists cannot drop a running job's gate {TAG}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
