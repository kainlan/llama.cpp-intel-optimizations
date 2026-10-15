#!/usr/bin/env python3
"""Same-queue ready events skipped by the per-op source chain (llama.cpp-wfsl).

Companion of test-sycl-moe-join-queue-source.py (the other wfsl fix).

THE COST.  Before every op, ggml_sycl_compute_forward_impl calls
ggml_sycl_chain_source_ready_events_if_needed(*ctx.stream(), dst, device), which
resolves each source and its view chain and, for every ready event it finds, asks
ggml_sycl_should_add_dependency, a get_info(command_execution_status) query.  The
backend's queues are profiling-enabled, and on driver 26.35 a status query on an
unfinished kernel of a profiling queue busy-waits on the kernel's timestamp packet
(libze_intel_gpu.so.1.17.39758, 0x389ae0).  During Qwen3.8 hybrid decode the ready
events come from producers on the same in-order ctx.stream() the consumer is
submitted to, so the dependency the query decides is already implied by in-order
execution: about 9.7 ms per token of busy-wait on the submitting thread.

THE FIX.  A producer that submits on an in-order queue records that queue's
identity beside the ready event (resolved_ptr::has_ready_queue / ready_queue, set
and cleared with ready_event by mem_handle).  The chain path, and only it, skips
an event whose recorded producer is the consuming in-order queue.  Any other
event, from another queue, a host task or an unknown producer, keeps today's query
and barrier.  GGML_SYCL_CHAIN_SAME_QUEUE_SKIP=0 restores today's path for all.

THE CONTRACT.  With comments blanked:
  K. (ggml-sycl.cpp chain_same_queue_skip_enabled) GGML_SYCL_CHAIN_SAME_QUEUE_SKIP is
     read once into a function-local static, default on; the only getenv of it.
  I. (mem-handle.hpp ready_queue_identity) false for an out-of-order queue, else
     std::hash<sycl::queue> of it.
  R. (mem-handle.hpp resolved_ptr) carries bool has_ready_queue = false and
     size_t ready_queue = 0, after the fields positional initialisers name.
  H. (mem-handle.cpp) set_ready_event(event) forgets the producer, set_ready_event(
     event, producer) records it through ready_queue_identity, and
     clear_ready_event() forgets it.
  S. (common.hpp ggml_sycl_chain_tensor_ready_event_if_needed) an event is skipped
     only when the caller passed an identity and the event's recorded producer
     equals it; every other event is counted as a query and goes through
     ggml_sycl_chain_ready_event_if_needed as before.
  G. (common.hpp ggml_sycl_chain_source_ready_events_if_needed) the identity is
     passed only when the knob is on, no command graph records, and the queue is
     in-order.
  O. (common.hpp ggml_sycl_chain_ready_event_if_needed) the shared single-event
     helper is unchanged: it does not consult the producer, so its other callers
     (binbcast, mmvq, ggml-sycl.cpp) keep today's behaviour.
  P. The producers of activation ready events on ctx.stream() record their queue:
     the MUL_MAT_ID ready guard's sync kernel, binbcast MUL, ADD_ID, the down
     weighted sum (final, down, down_biased) and the decode pair's GLU and down
     events.  The prompt-fusion publication stays unrecorded.
  T. (ggml-sycl.cpp) the [MOE-HOSTPATH-WAITS] line prints ready_queries= and
     ready_skipped=, the census holds and resets both, and the op's chain call
     passes the knob and a count that is added to the census.
  C. (docs/backend/sycl-env-vars.md) the GGML_SYCL_CHAIN_SAME_QUEUE_SKIP row states
     the mechanism and the oneAPI 2026.1 / driver 26.35 caveat.

NOT VACUOUS.  --self-test runs the gate on in-memory mutants of the real sources and
requires each to FAIL on its own pin, then requires the unmodified tree to pass.

RED on the tree before the change:
  FAIL [pin K]: no definition matching ... chain_same_queue_skip_enabled ... (llama.cpp-wfsl)
"""

from __future__ import annotations

import bisect
import functools
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SYCL = ROOT / "ggml" / "src" / "ggml-sycl"
BACKEND = SYCL / "ggml-sycl.cpp"
COMMON = SYCL / "common.hpp"
HANDLE_HPP = SYCL / "mem-handle.hpp"
HANDLE_CPP = SYCL / "mem-handle.cpp"
BINBCAST = SYCL / "binbcast.cpp"
ADD_ID = SYCL / "add-id.cpp"
DOC = ROOT / "docs" / "backend" / "sycl-env-vars.md"
TAG = "(llama.cpp-wfsl)"
KNOB = "GGML_SYCL_CHAIN_SAME_QUEUE_SKIP"


class ContractError(Exception):
    pass


def fail(pin: str, msg: str) -> None:
    raise ContractError(f"FAIL [pin {pin}]: {msg} {TAG}")


_TOKEN = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n]){1,4}\'', re.S)


@functools.lru_cache(maxsize=512)
def blank_comments(src: str) -> str:
    """Blank // and /* */ comments, keeping string literals and offsets."""
    def rep(m: re.Match) -> str:
        s = m.group(0)
        if s.startswith("//"):
            return " " * len(s)
        if s.startswith("/*"):
            return re.sub(r"[^\n]", " ", s)
        return s
    return _TOKEN.sub(rep, src)


@functools.lru_cache(maxsize=512)
def strip_strings(src: str) -> str:
    """Blank the contents of string literals so braces in them do not count."""
    return re.sub(r'"(?:\\.|[^"\\\n])*"', lambda m: '"' + " " * (len(m.group(0)) - 2) + '"', src)


def body_of(code: str, signature: str, pin: str) -> tuple[str, int]:
    """The body of the first definition matching `signature` (ending at its opening brace), and its offset."""
    m = re.search(signature, code)
    if not m:
        fail(pin, f"no definition matching {signature!r}")
    i = code.index("{", m.end() - 1)
    depth = 0
    flat = strip_strings(code)
    for j in range(i, len(flat)):
        if flat[j] == "{":
            depth += 1
        elif flat[j] == "}":
            depth -= 1
            if depth == 0:
                return code[i + 1 : j], i + 1
    fail(pin, f"unbalanced body for {signature!r}")
    return "", 0


@functools.lru_cache(maxsize=64)
def brace_positions(flat: str) -> tuple[list[int], list[str]]:
    pos, kind = [], []
    for m in re.finditer(r"[{}]", flat):
        pos.append(m.start())
        kind.append(m.group(0))
    return pos, kind


def enclosing_headers(code: str, pos: int) -> list[str]:
    """The text before each `{` that encloses `pos`, innermost first (an if's condition, a lambda head, ...)."""
    flat = strip_strings(code)
    positions, kinds = brace_positions(flat)
    heads = []
    depth = 0
    for idx in range(bisect.bisect_left(positions, pos) - 1, -1, -1):
        i = positions[idx]
        if kinds[idx] == "}":
            depth += 1
        elif depth == 0:
            k = max(flat.rfind(";", 0, i), flat.rfind("{", 0, i), flat.rfind("}", 0, i))
            heads.append(code[k + 1 : i].strip())
        else:
            depth -= 1
    return heads


def guarded(code: str, pos: int, guard: str) -> bool:
    return any(re.search(guard + r"\s*$", h) for h in enclosing_headers(code, pos))


def line_of(code: str, pos: int) -> int:
    return code.count("\n", 0, pos) + 1


KNOB_SIG = r"static\s+bool\s+chain_same_queue_skip_enabled\s*\(\s*\)\s*\{"
IDENTITY_SIG = r"inline\s+bool\s+ready_queue_identity\s*\(\s*const\s+sycl::queue\s*&\s*q\s*,\s*size_t\s*\*\s*out\s*\)\s*\{"
RESOLVED_SIG = r"struct\s+resolved_ptr\s*\{"
SET1_SIG = r"void\s+mem_handle::set_ready_event\s*\(\s*const\s+sycl::event\s*&\s*event\s*\)\s*\{"
SET2_SIG = (r"void\s+mem_handle::set_ready_event\s*\(\s*const\s+sycl::event\s*&\s*event\s*,\s*"
            r"const\s+sycl::queue\s*&\s*producer\s*\)\s*\{")
CLEAR_SIG = r"void\s+mem_handle::clear_ready_event\s*\(\s*\)\s*\{"
TENSOR_SIG = r"inline\s+void\s+ggml_sycl_chain_tensor_ready_event_if_needed\s*\([^)]*\)\s*\{"
SOURCE_SIG = r"inline\s+void\s+ggml_sycl_chain_source_ready_events_if_needed\s*\([^)]*\)\s*\{"
SINGLE_SIG = r"inline\s+void\s+ggml_sycl_chain_ready_event_if_needed\s*\([^)]*\)\s*\{"
SET_TENSOR_SIG = r"inline\s+void\s+ggml_sycl_set_tensor_ready_event\s*\([^)]*\)\s*\{"
CENSUS_SIG = r"struct\s+moe_hostpath_wait_census\s*\{"
READBACK_SIG = r"static\s+void\s+moe_hostpath_waits_readback\s*\([^)]*\)\s*\{"
READY_ADD_SIG = r"static\s+void\s+moe_hostpath_waits_ready\s*\(\s*const\s+ggml_sycl_chain_ready_counts\s*&\s*c\s*\)\s*\{"


def flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def check_knob(s: dict) -> None:
    code = blank_comments(s["backend"])
    body, _ = body_of(code, KNOB_SIG, "K")
    if not re.search(r"static\s+const\s+bool\s+enabled\s*=\s*\[\s*\]\s*\{", body):
        fail("K", "chain_same_queue_skip_enabled does not read the knob once into a function-local static")
    if f'std::getenv("{KNOB}")' not in body:
        fail("K", f"chain_same_queue_skip_enabled does not read {KNOB}")
    if not re.search(r"return\s+env\s*==\s*nullptr\s*\|\|\s*std::atoi\s*\(\s*env\s*\)\s*!=\s*0\s*;", body):
        fail("K", f"{KNOB} is not on by default (unset or non-zero)")
    reads = sum(blank_comments(t).count(f'getenv("{KNOB}")') for t in s["sources"].values())
    if reads != 1:
        fail("K", f"{KNOB} is read at {reads} sites; the one read is chain_same_queue_skip_enabled's")


def check_identity(s: dict) -> None:
    code = blank_comments(s["handle_hpp"])
    body, _ = body_of(code, IDENTITY_SIG, "I")
    f = flat(body)
    if not re.search(r"if \( ?!q\.has_property<sycl::property::queue::in_order>\(\) ?\) \{ return false; \}", f):
        fail("I", "ready_queue_identity gives an identity to an out-of-order queue")
    if not re.search(r"\*out = std::hash<sycl::queue>\{\}\(q\); return true;", f):
        fail("I", "ready_queue_identity is not std::hash<sycl::queue> of the queue")


def check_resolved(s: dict) -> None:
    code = blank_comments(s["handle_hpp"])
    body, _ = body_of(code, RESOLVED_SIG, "R")
    has = re.search(r"\bbool\s+has_ready_queue\s*=\s*false\s*;", body)
    idn = re.search(r"\bsize_t\s+ready_queue\s*=\s*0\s*;", body)
    ret = re.search(r"std::shared_ptr<mem_handle>\s+retention\s*;", body)
    if not has or not idn:
        fail("R", "resolved_ptr does not carry has_ready_queue = false and ready_queue = 0")
    if not ret or has.start() < ret.end() or idn.start() < ret.end():
        fail("R", "the producer fields are not after every field a positional initialiser names")


def check_handle(s: dict) -> None:
    code = blank_comments(s["handle_cpp"])
    body, _ = body_of(code, SET1_SIG, "H")
    f = flat(body)
    if "cached_.has_ready_event = true;" not in f or "cached_.has_ready_queue = false;" not in f:
        fail("H", "set_ready_event(event) does not forget the previous event's producer")
    body, _ = body_of(code, SET2_SIG, "H")
    f = flat(body)
    rec = re.search(r"const bool known = ready_queue_identity\(producer, &identity\);", f)
    lock = re.search(r"mem_handle_lock_guard g\(lock_\);", f)
    if not rec or "cached_.has_ready_queue = known;" not in f or "cached_.ready_queue = identity;" not in f:
        fail("H", "set_ready_event(event, producer) does not record the producer through ready_queue_identity")
    if not lock or "cached_.ready_event = event;" not in f[lock.end():]:
        fail("H", "set_ready_event(event, producer) does not set the event and its producer under the handle lock")
    body, _ = body_of(code, CLEAR_SIG, "H")
    f = flat(body)
    if "cached_.has_ready_event = false;" not in f or "cached_.has_ready_queue = false;" not in f:
        fail("H", "clear_ready_event() does not forget the producer with the event")


def check_skip(s: dict) -> None:
    code = blank_comments(s["common"])
    body, _ = body_of(code, TENSOR_SIG, "S")
    f = flat(body)
    skip = re.search(r"if \(same_queue && resolved\.has_ready_queue && resolved\.ready_queue == \*same_queue\) \{"
                     r" if \(counts\) \{ counts->skipped \+= 1; \} \} else \{ if \(counts\) \{ counts->queries \+= 1; \}"
                     r" ggml_sycl_chain_ready_event_if_needed\(q, resolved\); \}", f)
    if not skip:
        fail("S", "an event is skipped without its recorded producer equalling the consumer's identity, or the "
                  "other events are not counted and chained as before")


def check_gate(s: dict) -> None:
    code = blank_comments(s["common"])
    body, _ = body_of(code, SOURCE_SIG, "G")
    f = flat(body)
    same = re.search(r"const bool same = skip_same_queue && !ggml_sycl_graph_recording_active\(\) && "
                     r"ggml_sycl::ready_queue_identity\(q, &q_identity\);", f)
    if not same:
        fail("G", "the consumer's identity is passed even when the knob is off, a graph records, or the queue is "
                  "out of order")
    if not re.search(r"ggml_sycl_chain_tensor_ready_event_if_needed\(q, dst->src\[i\], device, same \? &q_identity : "
                     r"nullptr, counts\);", f):
        fail("G", "the source chain does not pass the identity only when skipping applies")


def check_single(s: dict) -> None:
    code = blank_comments(s["common"])
    body, _ = body_of(code, SINGLE_SIG, "O")
    if "ready_queue" in body:
        fail("O", "the shared single-event helper consults the producer, so its other callers change too")


PRODUCERS = (
    ("backend", "the MUL_MAT_ID ready guard's sync kernel",
     "ggml_sycl_set_tensor_ready_event(dst, ctx->device, ggml_sycl_submit_queue_sync_event(*stream), stream);"),
    ("binbcast", "binbcast MUL", "ggml_sycl_set_tensor_ready_event(dst, device, ensure_done_event(), main_stream);"),
    ("add_id", "ADD_ID", "ggml_sycl_set_tensor_ready_event(const_cast<ggml_tensor *>(dst.raw()), ctx.device, evt, &q);"),
    ("backend", "the down weighted sum's final", "ggml_sycl_set_tensor_ready_event(final, ctx.device, evt, ctx.stream());"),
    ("backend", "the down weighted sum's down", "ggml_sycl_set_tensor_ready_event(down, ctx.device, evt, ctx.stream());"),
    ("backend", "the down weighted sum's down_biased",
     "ggml_sycl_set_tensor_ready_event(down_biased, ctx.device, evt, ctx.stream());"),
    ("backend", "the decode pair's GLU",
     "ggml_sycl_set_tensor_ready_event(pair.glu_dst, ctx.device, glu_event, ctx.stream());"),
    ("backend", "the decode pair's down",
     "ggml_sycl_set_tensor_ready_event(pair.down_dst, ctx.device, down_event, ctx.stream());"),
)


def check_producers(s: dict) -> None:
    for key, what, call in PRODUCERS:
        if flat(call) not in flat(blank_comments(s[key])):
            fail("P", f"{what} does not record the queue its ready event was submitted on")
    code = blank_comments(s["common"])
    body, _ = body_of(code, SET_TENSOR_SIG, "P")
    f = flat(body)
    if not re.search(r"if \(producer\) \{ handle\.set_ready_event\(event, \*producer\); \} else \{ "
                     r"handle\.set_ready_event\(event\); \}", f):
        fail("P", "ggml_sycl_set_tensor_ready_event does not hand the producer to the handle")


def check_trace(s: dict) -> None:
    code = blank_comments(s["backend"])
    body, _ = body_of(code, CENSUS_SIG, "T")
    for field in ("ready_queries", "ready_skipped"):
        if not re.search(r"\buint64_t\s+" + field + r"\s*=\s*0\s*;", body):
            fail("T", f"moe_hostpath_wait_census does not hold {field}")
    body, _ = body_of(code, READBACK_SIG, "T")
    f = flat(body)
    # Adjacent string literals are one format string, as the compiler joins them.
    m = re.search(r'GGML_LOG_WARN\(\s*((?:"[^"]*"\s*)+)', f)
    fmt = "".join(re.findall(r'"([^"]*)"', m.group(1))) if m else ""
    if not fmt.startswith("[MOE-HOSTPATH-WAITS]") or "ready_queries=%llu ready_skipped=%llu" not in fmt:
        fail("T", "the [MOE-HOSTPATH-WAITS] line does not print ready_queries= and ready_skipped=")
    if "(unsigned long long) w.ready_queries, (unsigned long long) w.ready_skipped);" not in f:
        fail("T", "the [MOE-HOSTPATH-WAITS] line does not print the census's ready counts")
    if "w.ready_queries = 0;" not in f or "w.ready_skipped = 0;" not in f:
        fail("T", "the ready counts are not reset with the token's other counts")
    body, _ = body_of(code, READY_ADD_SIG, "T")
    f = flat(body)
    if not re.search(r"if \(!moe_hostpath_waits_enabled\(\)\) \{ return; \}", f) or \
            "g_moe_hostpath_waits.ready_queries += c.queries;" not in f or \
            "g_moe_hostpath_waits.ready_skipped += c.skipped;" not in f:
        fail("T", "moe_hostpath_waits_ready does not add an op's counts to the census")
    call = re.search(r"ggml_sycl_chain_ready_counts ready_counts; ggml_sycl_chain_source_ready_events_if_needed\("
                     r"\*ctx\.stream\(\), dst, ctx\.device, chain_same_queue_skip_enabled\(\), &ready_counts\); "
                     r"moe_hostpath_waits_ready\(ready_counts\);", flat(code))
    if not call:
        fail("T", "the op's source chain does not pass the knob and count into the census")


def check_doc(doc: str) -> None:
    row = None
    for line in (doc or "").splitlines():
        if line.startswith("| `" + KNOB):
            row = line
            break
    if row is None:
        fail("C", f"docs/backend/sycl-env-vars.md has no {KNOB} row")
    for needle, what in (("timestamp packet", "the driver's busy-wait"),
                         ("in-order", "why a same-queue dependency is already ordered"),
                         ("another queue", "why a cross-queue event is still queried"),
                         ("ready_queries=", "the trace field of the queries still made"),
                         ("ready_skipped=", "the trace field of the skips"),
                         ("oneAPI 2026.1", "the oneAPI version it was verified on"),
                         ("26.35", "the driver it was verified on"),
                         ("re-measure", "the re-measure caveat")):
        if needle not in row:
            fail("C", f"the {KNOB} row does not state {what} ({needle!r})")


def read_sources(root: Path) -> dict:
    out = {}
    for key, path in (("backend", BACKEND), ("common", COMMON), ("handle_hpp", HANDLE_HPP), ("handle_cpp", HANDLE_CPP),
                      ("binbcast", BINBCAST), ("add_id", ADD_ID), ("doc", DOC)):
        p = root / path.relative_to(ROOT)
        out[key] = p.read_text(encoding="utf-8") if p.exists() else ""
    sources = {}
    sycl = root / SYCL.relative_to(ROOT)
    for p in sorted(sycl.rglob("*.[ch]pp")):
        name = p.relative_to(sycl).as_posix()
        if name.startswith("tests/"):
            continue
        sources[name] = p.read_text(encoding="utf-8", errors="replace")
    out["sources"] = sources
    return out


def check(s: dict) -> None:
    check_knob(s)
    check_identity(s)
    check_resolved(s)
    check_handle(s)
    check_skip(s)
    check_gate(s)
    check_single(s)
    check_producers(s)
    check_trace(s)
    check_doc(s["doc"])


def self_test(s: dict) -> int:
    def sub(pattern: str, repl: str, flags: int = 0):
        return lambda text: re.sub(pattern, repl, text, count=1, flags=flags)

    def drop_queue(call: str):
        # Drops the call's last argument, the producer queue, however the source wraps the call.
        head, last = call[:-2].rsplit(", ", 1)
        pat = r"\s*".join(map(re.escape, head.split())) + r",\s*" + re.escape(last) + r"\s*\);"
        return lambda text: re.sub(pat, lambda _m: head + ");", text, count=1)

    mutants = [
        ("K", "backend", "knob read per call",
         sub(r"(static bool chain_same_queue_skip_enabled\(\) \{\n\s*)static const bool enabled", r"\1const bool enabled")),
        ("K", "backend", "knob off by default",
         sub(r'(GGML_SYCL_CHAIN_SAME_QUEUE_SKIP"\);\n\s*)return env == nullptr \|\| std::atoi\(env\) != 0;',
             r"\1return env != nullptr && std::atoi(env) != 0;")),
        ("I", "handle_hpp", "identity for an out-of-order queue",
         sub(r"if \(!q\.has_property<sycl::property::queue::in_order>\(\)\) \{\n\s*return false;\n\s*\}", "")),
        ("I", "handle_hpp", "identity is not the queue's hash",
         sub(r"\*out = std::hash<sycl::queue>\{\}\(q\);", "*out = 1;")),
        ("R", "handle_hpp", "producer flag missing", sub(r"bool(\s+)has_ready_queue(\s*)= false;", r"bool\1has_ready_q\2= false;")),
        ("R", "handle_hpp", "producer fields before retention",
         sub(r"(    std::shared_ptr<mem_handle> retention;\n)((?:.*\n)*?)(    bool +has_ready_queue += false;\n    size_t +ready_queue += 0;\n)",
             r"\3\1\2")),
        ("H", "handle_cpp", "one-argument set keeps the old producer",
         sub(r"(void mem_handle::set_ready_event\(const sycl::event & event\) \{(?:(?!\n\}).)*?)cached_\.has_ready_queue = false;",
             r"\1;", re.S)),
        ("H", "handle_cpp", "producer not recorded",
         sub(r"cached_\.has_ready_queue = known;", "cached_.has_ready_queue = false;")),
        ("H", "handle_cpp", "clear keeps the producer",
         sub(r"(void mem_handle::clear_ready_event\(\) \{(?:(?!\n\}).)*?)cached_\.has_ready_queue = false;", r"\1;", re.S)),
        ("S", "common", "skip without a recorded producer",
         sub(r"same_queue && resolved\.has_ready_queue && resolved\.ready_queue == \*same_queue",
             "same_queue && resolved.ready_queue == *same_queue")),
        ("S", "common", "skip whenever an identity is passed",
         sub(r"same_queue && resolved\.has_ready_queue && resolved\.ready_queue == \*same_queue", "same_queue")),
        ("S", "common", "queried events no longer chained",
         sub(r"(counts->queries \+= 1;\n\s*\}\n\s*)ggml_sycl_chain_ready_event_if_needed\(q, resolved\);", r"\1;")),
        ("G", "common", "skip while a graph records",
         sub(r"skip_same_queue && !ggml_sycl_graph_recording_active\(\) &&", "skip_same_queue &&")),
        ("G", "common", "knob ignored",
         sub(r"const bool\s+same\s*=\s*skip_same_queue && ", "const bool same = ")),
        ("O", "common", "shared helper skips too",
         sub(r"(inline void ggml_sycl_chain_ready_event_if_needed\(sycl::queue & q, const ggml_sycl::resolved_ptr & resolved\) \{\n)",
             r"\1    if (resolved.has_ready_queue) {\n        return;\n    }\n")),
        ("P", "common", "producer not handed to the handle",
         sub(r"handle\.set_ready_event\(event, \*producer\);", "handle.set_ready_event(event);")),
    ]
    for key, what, call in PRODUCERS:
        mutants.append(("P", key, f"{what} unrecorded", drop_queue(call)))
    mutants += [
        ("T", "backend", "census lacks the skip count",
         sub(r"uint64_t(\s+)ready_skipped(\s*)= 0;", r"uint64_t\1ready_skip_n\2= 0;")),
        ("T", "backend", "line drops the queries count", sub(r' ready_queries=%llu "', '"')),
        ("T", "backend", "counts not reset", sub(r"w\.ready_queries(\s*)= 0;", r"(void) w\1;")),
        ("T", "backend", "op counts not added", sub(r"g_moe_hostpath_waits\.ready_skipped \+= c\.skipped;", ";")),
        ("T", "backend", "chain call without the knob",
         sub(r"chain_same_queue_skip_enabled\(\),(\s*)&ready_counts\);", r"false,\1&ready_counts);")),
        ("C", "doc", "doc row missing", sub(r"\| `GGML_SYCL_CHAIN_SAME_QUEUE_SKIP", "| `GGML_SYCL_CHAIN_SAME_Q")),
        ("C", "doc", "doc row loses the version caveat",
         sub(r"(\| `GGML_SYCL_CHAIN_SAME_QUEUE_SKIP[^\n]*?)oneAPI 2026\.1", r"\1oneAPI")),
    ]
    failures = []
    caught = 0
    for pin, key, name, mutate in mutants:
        m = dict(s)
        m[key] = mutate(s[key] or "")
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
    print("PASS: the per-op source chain skips ready events its own in-order stream produced, behind "
          f"GGML_SYCL_CHAIN_SAME_QUEUE_SKIP, and queries every other one as before {TAG}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
