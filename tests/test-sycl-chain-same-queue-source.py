#!/usr/bin/env python3
"""Same-queue ready events skipped by the per-op source chain and the shared helper (llama.cpp-wfsl).

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
execution: about 9.7 ms per token of busy-wait on the submitting thread.  The same
query is made by the shared single-event helper, ggml_sycl_chain_ready_event_if_needed,
for binbcast's sources and the mmvq and split mul_mat weight checks.

THE FIX.  A producer that submits on an in-order queue records that queue's
identity beside the ready event (resolved_ptr::has_ready_queue / ready_queue, set
and cleared with ready_event by mem_handle).  The per-op chain, and the shared helper
for the callers that pass the identity of the queue they chain on, skip an event whose
recorded producer is that consuming in-order queue.  Any other event, from another
queue, a host task or an unknown producer, keeps today's query and barrier.
GGML_SYCL_CHAIN_SAME_QUEUE_SKIP=0 restores today's path for all.

THE TRACE.  Under GGML_SYCL_MOE_IDS_COPY_TRACE each status query is counted by the
site that asked it, and the wall time of the queries that found their event unfinished
is summed, so the [MOE-HOSTPATH-WAITS] line names the site that holds the submitting
thread (ready_queries_other=, ready_query_us=).

THE CONTRACT.  With comments blanked:
  K. (common.cpp ggml_sycl_chain_same_queue_skip_enabled) GGML_SYCL_CHAIN_SAME_QUEUE_SKIP
     is read once into a function-local static, default on; the only getenv of it.
  I. (mem-handle.hpp ready_queue_identity) false for an out-of-order queue, else
     std::hash<sycl::queue> of it.
  R. (mem-handle.hpp resolved_ptr) carries bool has_ready_queue = false and
     size_t ready_queue = 0, after the fields positional initialisers name.
  H. (mem-handle.cpp) set_ready_event(event) forgets the producer, set_ready_event(
     event, producer) records it through ready_queue_identity, and
     clear_ready_event() forgets it.
  S. (common.hpp) ggml_sycl_ready_event_from_queue is true only when the caller passed
     an identity and the event's recorded producer equals it; the per-op chain counts
     such an event as skipped, and every other one as a query that goes through the
     shared helper with no identity under the PER_OP_CHAIN site.
  G. (common.hpp ggml_sycl_chain_same_queue_identity) an identity is handed out only
     when the knob is on, no command graph records, and the queue is in-order; the
     per-op source chain takes its identity from it.
  O. (common.hpp ggml_sycl_chain_ready_event_if_needed) the shared helper skips an
     event only through ggml_sycl_ready_event_from_queue, and otherwise queries it under
     its caller's site and chains a barrier when it is unfinished.
  E. The shared helper is called only by the per-op chain, binbcast's two sources, the
     two mmvq weight checks and the split mul_mat weight check; each of the five passes
     the identity of the very queue it chains on, and each call names its own site.
  D. Every other ggml_sycl_should_add_dependency call names its own site.
  Q. (common.hpp ggml_sycl_should_add_dependency) only under
     ggml_sycl_cpu_expert_trace_enabled() it counts each query in the calling thread's
     census (common.cpp, thread_local) and, outside recording, adds the wall time of a
     query that found its event unfinished; the recording branch keeps every in-flight
     dependency as before.
  P. The producers of activation ready events record their queue (the MUL_MAT_ID ready
     guard's sync kernel, binbcast MUL, ADD_ID, the down weighted sum's final, down and
     down_biased, the decode pair's GLU and down; the prompt-fusion publication stays
     unrecorded), and each recorded event was submitted on that queue: binbcast's kernel
     and its event on main_stream, through the profiler's submit on the queue it is given;
     ADD_ID's kernel on q = *ctx.stream(); the down weighted sum on *ctx.stream(); the
     decode pair's GLU and down events from dispatches whose one queue is ctx.stream().
  T. (ggml-sycl.cpp) the [MOE-HOSTPATH-WAITS] line prints ready_queries=, ready_skipped=,
     ready_queries_other= (the non-zero sites other than the per-op chain, by name) and
     ready_query_us=; the census holds and resets the counts; the op's chain call adds its
     counts; every site has a name of its own.
  C. (docs/backend/sycl-env-vars.md) the GGML_SYCL_CHAIN_SAME_QUEUE_SKIP row states the
     mechanism, the shared helper, the trace fields, the atoi parse and the oneAPI
     2026.1 / driver 26.35 caveat.

NOT VACUOUS.  --self-test runs the gate on in-memory mutants of the real sources and
requires each to FAIL on its own pin, then requires the unmodified tree to pass.

RED on the tree before the shared helper took the identity:
  FAIL [pin K]: no definition matching ... ggml_sycl_chain_same_queue_skip_enabled ... (llama.cpp-wfsl)
"""

from __future__ import annotations

import bisect
import collections
import functools
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SYCL = ROOT / "ggml" / "src" / "ggml-sycl"
BACKEND = SYCL / "ggml-sycl.cpp"
COMMON = SYCL / "common.hpp"
COMMON_CPP = SYCL / "common.cpp"
HANDLE_HPP = SYCL / "mem-handle.hpp"
HANDLE_CPP = SYCL / "mem-handle.cpp"
BINBCAST = SYCL / "binbcast.cpp"
ADD_ID = SYCL / "add-id.cpp"
MMVQ = SYCL / "mmvq.cpp"
FATTN = SYCL / "fattn.cpp"
MOE_XMX = SYCL / "moe-xmx.hpp"
PROFILER = SYCL / "sycl-kernel-profiler.hpp"
DOC = ROOT / "docs" / "backend" / "sycl-env-vars.md"
TAG = "(llama.cpp-wfsl)"
KNOB = "GGML_SYCL_CHAIN_SAME_QUEUE_SKIP"
SITE = "GGML_SYCL_READY_QUERY_SITE_"

# The file each source key is, so the census pins see the mutated text.
FILES = (("backend", BACKEND), ("common", COMMON), ("common_cpp", COMMON_CPP), ("handle_hpp", HANDLE_HPP),
         ("handle_cpp", HANDLE_CPP), ("binbcast", BINBCAST), ("add_id", ADD_ID), ("mmvq", MMVQ), ("fattn", FATTN),
         ("moe_xmx", MOE_XMX), ("profiler", PROFILER))


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
    flat_code = strip_strings(code)
    for j in range(i, len(flat_code)):
        if flat_code[j] == "{":
            depth += 1
        elif flat_code[j] == "}":
            depth -= 1
            if depth == 0:
                return code[i + 1 : j], i + 1
    fail(pin, f"unbalanced body for {signature!r}")
    return "", 0


def line_of(code: str, pos: int) -> int:
    return code.count("\n", 0, pos) + 1


def flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def call_args(code: str, open_paren: int) -> list[str] | None:
    """The top-level arguments, whitespace-normalised, of the call whose '(' is at `open_paren` in `code`."""
    bare = strip_strings(code)
    depth = 0
    start = open_paren + 1
    args = []
    for j in range(open_paren, len(bare)):
        c = bare[j]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                args.append(flat(code[start:j]).strip())
                return [a for a in args if a]
        elif c == "," and depth == 1:
            args.append(flat(code[start:j]).strip())
            start = j + 1
    return None


def calls_of(code: str, name: str) -> list[tuple[int, list[str]]]:
    """Each call of `name` in `code` (comments blanked) with its arguments; its definition is not a call."""
    out = []
    for m in re.finditer(r"\b" + name + r"\s*\(", code):
        prefix = code[code.rfind("\n", 0, m.start()) + 1 : m.start()]
        if re.search(r"\b(?:inline\s+)?(?:bool|void)\s*$", prefix):
            continue
        args = call_args(code, m.end() - 1)
        if args is not None:
            out.append((m.start(), args))
    return out


# A parameter list that may hold one level of parentheses (default arguments such as __builtin_FILE()).
ARGS = r"\((?:[^(){}]|\([^()]*\))*\)"

KNOB_SIG = r"\bbool\s+ggml_sycl_chain_same_queue_skip_enabled\s*\(\s*\)\s*\{"
IDENTITY_SIG = r"inline\s+bool\s+ready_queue_identity\s*\(\s*const\s+sycl::queue\s*&\s*q\s*,\s*size_t\s*\*\s*out\s*\)\s*\{"
RESOLVED_SIG = r"struct\s+resolved_ptr\s*\{"
SET1_SIG = r"void\s+mem_handle::set_ready_event\s*\(\s*const\s+sycl::event\s*&\s*event\s*\)\s*\{"
SET2_SIG = (r"void\s+mem_handle::set_ready_event\s*\(\s*const\s+sycl::event\s*&\s*event\s*,\s*"
            r"const\s+sycl::queue\s*&\s*producer\s*\)\s*\{")
CLEAR_SIG = r"void\s+mem_handle::clear_ready_event\s*\(\s*\)\s*\{"
FROM_QUEUE_SIG = r"inline\s+bool\s+ggml_sycl_ready_event_from_queue\s*\([^)]*\)\s*\{"
SAME_ID_SIG = r"inline\s+const\s+size_t\s*\*\s*ggml_sycl_chain_same_queue_identity\s*\([^)]*\)\s*\{"
TENSOR_SIG = r"inline\s+void\s+ggml_sycl_chain_tensor_ready_event_if_needed\s*\([^)]*\)\s*\{"
SOURCE_SIG = r"inline\s+void\s+ggml_sycl_chain_source_ready_events_if_needed\s*\([^)]*\)\s*\{"
SINGLE_SIG = r"inline\s+void\s+ggml_sycl_chain_ready_event_if_needed\s*\([^)]*\)\s*\{"
QUERY_SIG = r"inline\s+bool\s+ggml_sycl_should_add_dependency\s*\([^)]*\)\s*\{"
CENSUS_STRUCT_SIG = r"struct\s+ggml_sycl_ready_query_census\s*\{"
CENSUS_FN_SIG = r"ggml_sycl_ready_query_census\s*&\s*ggml_sycl_ready_query_census_this_thread\s*\(\s*\)\s*\{"
SITE_ENUM_SIG = r"enum\s+ggml_sycl_ready_query_site\s*\{"
SITE_NAME_SIG = r"inline\s+const\s+char\s*\*\s*ggml_sycl_ready_query_site_name\s*\([^)]*\)\s*\{"
SET_TENSOR_SIG = r"inline\s+void\s+ggml_sycl_set_tensor_ready_event\s*\([^)]*\)\s*\{"
CENSUS_SIG = r"struct\s+moe_hostpath_wait_census\s*\{"
READBACK_SIG = r"static\s+void\s+moe_hostpath_waits_readback\s*\([^)]*\)\s*\{"
READY_ADD_SIG = r"static\s+void\s+moe_hostpath_waits_ready\s*\(\s*const\s+ggml_sycl_chain_ready_counts\s*&\s*c\s*\)\s*\{"
BIN_OP_SIG = r"inline\s+void\s+ggml_sycl_op_bin_bcast\s*" + ARGS + r"\s*\{"
BIN_STRUCT_SIG = r"struct\s+bin_bcast_sycl\s*\{"
BIN_CALL_SIG = r"sycl::event\s+operator\(\)\s*" + ARGS + r"\s*\{"
BIN_KERNEL_SIG = r"static\s+sycl::event\s+ggml_sycl_submit_binbcast_kernel\s*" + ARGS + r"\s*\{"
BIN_EVENT_SIG = r"static\s+sycl::event\s+ggml_sycl_submit_binbcast_event\s*" + ARGS + r"\s*\{"
PROFILE_IMPL_SIG = r"inline\s+sycl::event\s+ggml_sycl_profile_submit_impl\s*" + ARGS + r"\s*\{"
ADD_ID_SIG = r"void\s+ggml_sycl_add_id\s*\(\s*ggml_backend_sycl_context\s*&\s*ctx\s*,\s*ggml_sycl::sycl_tensor\s+dst\s*\)\s*\{"
DOWN_FUSE_SIG = r"static\s+bool\s+ggml_sycl_try_fuse_moe_down_weighted_sum\s*\([^)]*\)\s*\{"
DOWN_SUM_SIG = r"static\s+sycl::event\s+ggml_sycl_moe_down_weighted_sum_sycl\s*\([^)]*\)\s*\{"
GLU_SIG = r"\bbool\s+mmvq_moe_batched_dispatch_pair_glu_mxfp4_soa\s*" + ARGS + r"\s*\{"
DOWN_SIG = r"\bbool\s+mmvq_moe_batched_dispatch_down_from_cached_q8_mxfp4\s*" + ARGS + r"\s*\{"


def sources_of(s: dict) -> dict:
    """Every backend source by its path under ggml-sycl, with the keyed sources' (possibly mutated) text."""
    out = dict(s["sources"])
    for key, path in FILES:
        out[path.relative_to(SYCL).as_posix()] = s[key]
    return out


def check_knob(s: dict) -> None:
    code = blank_comments(s["common_cpp"])
    body, _ = body_of(code, KNOB_SIG, "K")
    if not re.search(r"static\s+const\s+bool\s+enabled\s*=\s*\[\s*\]\s*\{", body):
        fail("K", "ggml_sycl_chain_same_queue_skip_enabled does not read the knob once into a function-local static")
    if f'std::getenv("{KNOB}")' not in body:
        fail("K", f"ggml_sycl_chain_same_queue_skip_enabled does not read {KNOB}")
    if not re.search(r"return\s+env\s*==\s*nullptr\s*\|\|\s*std::atoi\s*\(\s*env\s*\)\s*!=\s*0\s*;", body):
        fail("K", f"{KNOB} is not on by default (unset or non-zero)")
    reads = sum(blank_comments(t).count(f'getenv("{KNOB}")') for t in sources_of(s).values())
    if reads != 1:
        fail("K", f"{KNOB} is read at {reads} sites; the one read is ggml_sycl_chain_same_queue_skip_enabled's")


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
    body, _ = body_of(code, FROM_QUEUE_SIG, "S")
    if not re.fullmatch(r" ?return same_queue && resolved\.has_ready_queue && resolved\.ready_queue == \*same_queue; ?",
                        flat(body)):
        fail("S", "ggml_sycl_ready_event_from_queue is true without the event's recorded producer equalling the "
                  "identity the caller passed")
    body, _ = body_of(code, TENSOR_SIG, "S")
    skip = re.search(r"if \(ggml_sycl_ready_event_from_queue\(resolved, same_queue\)\) \{ if \(counts\) \{ "
                     r"counts->skipped \+= 1; \} \} else \{ if \(counts\) \{ counts->queries \+= 1; \} "
                     r"ggml_sycl_chain_ready_event_if_needed\(q, resolved, nullptr, " + SITE + r"PER_OP_CHAIN\); \}",
                     flat(body))
    if not skip:
        fail("S", "the per-op chain skips an event its queue did not produce, or the other events are not counted "
                  "and queried through the shared helper under the PER_OP_CHAIN site")


def check_gate(s: dict) -> None:
    code = blank_comments(s["common"])
    body, _ = body_of(code, SAME_ID_SIG, "G")
    if not re.fullmatch(r" ?if \(ggml_sycl_chain_same_queue_skip_enabled\(\) && !ggml_sycl_graph_recording_active\(\) "
                        r"&& ggml_sycl::ready_queue_identity\(q, identity\)\) \{ return identity; \} return nullptr; ?",
                        flat(body)):
        fail("G", "an identity is handed out even when the knob is off, a graph records, or the queue is out of order")
    body, _ = body_of(code, SOURCE_SIG, "G")
    f = flat(body)
    if not re.search(r"size_t q_identity = 0; const size_t \* same = ggml_sycl_chain_same_queue_identity\(q, "
                     r"&q_identity\);", f):
        fail("G", "the per-op source chain does not take its identity from ggml_sycl_chain_same_queue_identity")
    if not re.search(r"ggml_sycl_chain_tensor_ready_event_if_needed\(q, dst->src\[i\], device, same, counts\);", f):
        fail("G", "the per-op source chain does not pass that identity to each source")


def check_single(s: dict) -> None:
    code = blank_comments(s["common"])
    m = re.search(SINGLE_SIG, code)
    if not m or not re.search(r"const size_t \* same_queue = nullptr, ggml_sycl_ready_query_site site = " + SITE +
                              r"OTHER\)", flat(m.group(0))):
        fail("O", "the shared helper does not take an optional identity and its caller's site")
    body, _ = body_of(code, SINGLE_SIG, "O")
    if not re.fullmatch(r" ?if \(!resolved\.has_ready_event \|\| ggml_sycl_ready_event_from_queue\(resolved, "
                        r"same_queue\)\) \{ return; \} if \(ggml_sycl_should_add_dependency\(resolved\.ready_event, "
                        r"site\)\) \{ q\.ext_oneapi_submit_barrier\(\{ resolved\.ready_event \}\); \} ?", flat(body)):
        fail("O", "the shared helper skips an event its queue did not produce, or does not query the others under "
                  "its caller's site and chain the unfinished ones")


# The shared helper's callers: (file, site). The per-op chain passes no identity; the others pass their queue's.
HELPER_CALLS = (
    ("common.hpp", SITE + "PER_OP_CHAIN"),
    ("binbcast.cpp", SITE + "BINBCAST_SRC0"),
    ("binbcast.cpp", SITE + "BINBCAST_SRC1"),
    ("mmvq.cpp", SITE + "MMVQ_MOE_WEIGHT"),
    ("mmvq.cpp", SITE + "MMVQ_WEIGHT"),
    ("ggml-sycl.cpp", SITE + "MUL_MAT_WEIGHT"),
)


def check_helper_calls(s: dict) -> None:
    seen = []
    for name, text in sorted(sources_of(s).items()):
        if "ggml_sycl_chain_ready_event_if_needed" not in text:
            continue
        code = blank_comments(text)
        for pos, args in calls_of(code, "ggml_sycl_chain_ready_event_if_needed"):
            where = f"{name}:{line_of(code, pos)}"
            if len(args) != 4 or not args[3].startswith(SITE):
                fail("E", f"the shared helper's call at {where} names no site of its own")
            q, _, identity, site = args
            if site == SITE + "PER_OP_CHAIN":
                if identity != "nullptr":
                    fail("E", f"the per-op chain's call at {where} hands the helper an identity")
            else:
                m = re.fullmatch(r"ggml_sycl_chain_same_queue_identity\((.+), &\w+\)", identity)
                if not m or m.group(1).strip() != q:
                    fail("E", f"the shared helper's call at {where} does not pass the identity of {q}, the queue it "
                              "chains on")
            seen.append((name, site))
    if sorted(seen) != sorted(HELPER_CALLS):
        fail("E", f"the shared helper's calls are {sorted(seen)}, not the six this gate knows")


# Every other status query: (file, event, site).
QUERY_CALLS = (
    (("common.hpp", "resolved.ready_event", "site"),
     ("add-id.cpp", "resolved.ready_event", SITE + "ADD_ID"),
     ("fattn.cpp", "set_rows_event", SITE + "FATTN_SET_ROWS"),
     ("fattn.cpp", "zero_event", SITE + "FATTN_PACKED_ZERO"),
     ("fattn.cpp", "packed.ready_event", SITE + "FATTN_PACKED_PREV"),
     ("fattn.cpp", "previous_use", SITE + "FATTN_PREV_USE"),
     ("fattn-xmx-f16-v2.hpp", "packed_ready_event", SITE + "FATTN_XMX_PACKED_K"),
     ("moe-xmx.hpp", "dep_event", SITE + "MOE_XMX"))
    + (("moe-sort.hpp", "dep_event", SITE + "MOE_SORT"),) * 6
    + (("cpu-dispatch.cpp", "evt", SITE + "CPU_DISPATCH"),) * 3
)


def check_query_calls(s: dict) -> None:
    seen = []
    for name, text in sorted(sources_of(s).items()):
        if "ggml_sycl_should_add_dependency" not in text:
            continue
        code = blank_comments(text)
        for pos, args in calls_of(code, "ggml_sycl_should_add_dependency"):
            if len(args) != 2:
                fail("D", f"the status query at {name}:{line_of(code, pos)} names no site, so the trace counts it "
                          "as other")
            seen.append((name, args[0], args[1]))
    if sorted(seen) != sorted(QUERY_CALLS):
        missing = collections.Counter(QUERY_CALLS) - collections.Counter(seen)
        extra = collections.Counter(seen) - collections.Counter(QUERY_CALLS)
        fail("D", f"the status queries differ from the sites this gate knows: missing {sorted(missing)}, "
                  f"extra {sorted(extra)}")


def check_query(s: dict) -> None:
    code = blank_comments(s["common"])
    m = re.search(QUERY_SIG, code)
    if not m or not re.search(r"ggml_sycl_ready_query_site site = " + SITE + r"OTHER\)", flat(m.group(0))):
        fail("Q", "ggml_sycl_should_add_dependency does not take its caller's site")
    body, _ = body_of(code, QUERY_SIG, "Q")
    f = flat(body)
    status = (r"dep_event\.get_info<sycl::info::event::command_execution_status>\(\) != "
              r"sycl::info::event_command_status::complete")
    want = (r" ?const bool traced = ggml_sycl_cpu_expert_trace_enabled\(\); "
            r"if \(traced\) \{ ggml_sycl_ready_query_census_this_thread\(\)\.count\[site\] \+= 1; \} "
            r"if \(ggml_sycl_graph_recording_active\(\)\) \{ try \{ auto status = "
            r"dep_event\.get_info<sycl::info::event::command_execution_status>\(\); if \(status == "
            r"sycl::info::event_command_status::complete\) \{ return false; \} \} catch \(\.\.\.\) \{ \} return true; \} "
            r"if \(!traced\) \{ return " + status + r"; \} "
            r"const auto t0 = std::chrono::steady_clock::now\(\); const bool pending = " + status + r"; "
            r"if \(pending\) \{ ggml_sycl_ready_query_census_this_thread\(\)\.pending_us \+= "
            r"std::chrono::duration<double, std::micro>\(std::chrono::steady_clock::now\(\) - t0\)\.count\(\); \} "
            r"return pending; ?")
    if not re.fullmatch(want, f):
        fail("Q", "ggml_sycl_should_add_dependency does not count each query under the trace only, add the time of "
                  "the unfinished ones, and keep the recording branch's dependencies")
    if not re.search(r"\bbool\s+ggml_sycl_cpu_expert_trace_enabled\s*\(\s*\)\s*;", code):
        fail("Q", "common.hpp does not declare ggml_sycl_cpu_expert_trace_enabled, the trace's one reader")
    body, _ = body_of(code, CENSUS_STRUCT_SIG, "Q")
    if not re.search(r"uint64_t\s+count\[" + SITE + r"COUNT\]\s*=\s*\{\s*\}\s*;", body) or \
            not re.search(r"double\s+pending_us\s*=\s*0\.0\s*;", body):
        fail("Q", "the query census does not hold a count per site and the unfinished queries' time")
    body, _ = body_of(blank_comments(s["common_cpp"]), CENSUS_FN_SIG, "Q")
    if not re.fullmatch(r" ?static thread_local ggml_sycl_ready_query_census census; return census; ?", flat(body)):
        fail("Q", "the query census is not the calling thread's own")


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


def assignments(body: str, var: str) -> list[str]:
    """The right-hand sides of every `var = ...;` in `body`, whitespace-normalised."""
    return [flat(m.group(1)).strip() for m in re.finditer(r"(?<![\w.>])" + re.escape(var) + r"\s*=(?!=)\s*([^;]*);",
                                                            body)]


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


def check_origins(s: dict) -> None:
    # binbcast: the recorded event is the kernel's or a marker's, both submitted on main_stream.
    code = blank_comments(s["binbcast"])
    body, _ = body_of(code, BIN_OP_SIG, "P")
    if not re.search(r"dpct::queue_ptr\s+main_stream\s*=\s*ctx\.stream\(\)\s*;", body):
        fail("P", "binbcast's main_stream, the queue its MUL records, is not ctx.stream()")
    if sorted(assignments(body, "done_event")) != sorted(
            ["kernel_event", "ggml_sycl_submit_binbcast_event(*main_stream, mode)"]):
        fail("P", "binbcast's recorded event is not its kernel's or a marker submitted on *main_stream")
    calls = [m for m in re.finditer(r"\bkernel_event\s*=\s*op\(\)\s*\(", body)]
    if not calls or len(calls) != len(assignments(body, "kernel_event")):
        fail("P", "binbcast's kernel event does not come from op() alone")
    for m in calls:
        args = call_args(body, m.end() - 1) or []
        if args[-4:] != ["main_stream", "file", "line", "function"]:
            fail("P", f"binbcast hands op() a queue other than main_stream ({args[-4:-3]})")
    struct, off = body_of(code, BIN_STRUCT_SIG, "P")
    op_body, _ = body_of(struct, BIN_CALL_SIG, "P")
    subs = list(re.finditer(r"\bkernel_event\s*=\s*ggml_sycl_submit_binbcast_kernel\s*\(", op_body))
    if not subs or len(subs) != len(assignments(op_body, "kernel_event")):
        fail("P", "op()'s kernel event does not come from ggml_sycl_submit_binbcast_kernel alone")
    for m in subs:
        args = call_args(op_body, m.end() - 1) or []
        if not args or args[0] != "*stream" or "profiled_queue.parallel_for(" not in args[3]:
            fail("P", "op() submits its kernel on a queue other than the stream it was handed")
    body, _ = body_of(code, BIN_KERNEL_SIG, "P")
    if not re.search(r"return ggml_sycl_profile_submit\(q, label, static_cast<SubmitFn &&>\(submit_fn\), file, line, "
                     r"function\);", flat(body)):
        fail("P", "ggml_sycl_submit_binbcast_kernel does not submit on the queue it is given")
    body, _ = body_of(code, BIN_EVENT_SIG, "P")
    f = flat(body)
    if f.count("ggml_sycl_profile_submit(") != 2 or len(re.findall(r"ggml_sycl_profile_submit\( ?q, label,", f)) != 2 or \
            "return profiled_queue.ext_oneapi_submit_barrier();" not in f or "return profiled_queue.submit(" not in f:
        fail("P", "ggml_sycl_submit_binbcast_event does not submit its marker on the queue it is given")
    body, _ = body_of(blank_comments(s["profiler"]), PROFILE_IMPL_SIG, "P")
    if "sycl::event event = submit_fn(q);" not in flat(body):
        fail("P", "the profiler's submit does not hand the submission the queue it is given")
    # ADD_ID: its kernel goes on q = *ctx.stream(), the queue it records.
    body, _ = body_of(blank_comments(s["add_id"]), ADD_ID_SIG, "P")
    if not re.search(r"sycl::queue\s*&\s*q\s*=\s*\*ctx\.stream\(\)\s*;", body):
        fail("P", "ADD_ID's q, the queue it records, is not *ctx.stream()")
    rhs = assignments(body, "sycl::event evt")
    if len(rhs) != 1 or not rhs[0].startswith("q.submit(") or len(assignments(body, "evt")) != 1:
        fail("P", "ADD_ID's recorded event is not its one submission on q")
    # The down weighted sum: one submission on the queue the fusion passes, *ctx.stream().
    backend = blank_comments(s["backend"])
    body, _ = body_of(backend, DOWN_FUSE_SIG, "P")
    m = re.search(r"sycl::event\s+evt\s*=\s*ggml_sycl_moe_down_weighted_sum_sycl\s*\(", body)
    args = call_args(body, m.end() - 1) if m else None
    if not args or args[0] != "*ctx.stream()" or len(assignments(body, "evt")) != 1:
        fail("P", "the down weighted sum's recorded event is not its submission on *ctx.stream()")
    body, _ = body_of(backend, DOWN_SUM_SIG, "P")
    f = flat(body)
    if f.count(".submit(") != 1 or "return queue.submit(" not in f:
        fail("P", "ggml_sycl_moe_down_weighted_sum_sycl does not submit on the queue it is given")
    # The decode pair: each dispatch's one queue is ctx.stream(), and its completion event is a kernel of it.
    mmvq = blank_comments(s["mmvq"])
    for sig, what, done in ((GLU_SIG, "GLU", "down_q8_event_set ? down_q8_event : kernel_event"),
                            (DOWN_SIG, "down", "down_event")):
        body, _ = body_of(mmvq, sig, "P")
        queues = re.findall(r"\b(?:queue_ptr|dpct::queue_ptr|sycl::queue\s*[*&])\s*\w+\s*=[^;]*;", body)
        if [flat(q) for q in queues] != ["queue_ptr stream = ctx.stream();"] or body.count("ctx.stream(") != 1:
            fail("P", f"the decode pair's {what} dispatch submits on a queue other than ctx.stream()")
        if assignments(body, "*completion_event") != [done]:
            fail("P", f"the decode pair's {what} completion event is not the event of its own dispatch ({done})")


def check_names(s: dict) -> None:
    code = blank_comments(s["common"])
    body, _ = body_of(code, SITE_ENUM_SIG, "T")
    sites = re.findall(r"\b(" + SITE + r"\w+)", body)
    if not sites or sites[0] != SITE + "OTHER" or sites[-1] != SITE + "COUNT":
        fail("T", "the site enum does not run from OTHER to COUNT")
    body, _ = body_of(code, SITE_NAME_SIG, "T")
    names = dict(re.findall(r"case\s+(" + SITE + r"\w+)\s*:\s*return\s+\"([a-z0-9_]+)\"\s*;", body))
    want = [x for x in sites if x not in (SITE + "OTHER", SITE + "COUNT")]
    if sorted(names) != sorted(want) or len(set(names.values())) != len(names) or "other" in names.values():
        fail("T", "a site has no name of its own on the trace line")


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
    if not fmt.startswith("[MOE-HOSTPATH-WAITS]") or \
            "ready_queries=%llu ready_skipped=%llu ready_queries_other=%s ready_query_us=%.0f" not in fmt:
        fail("T", "the [MOE-HOSTPATH-WAITS] line does not print ready_queries=, ready_skipped=, ready_queries_other= "
                  "and ready_query_us=")
    if "(unsigned long long) w.ready_queries, (unsigned long long) w.ready_skipped, other_sites, rq.pending_us" not in f:
        fail("T", "the [MOE-HOSTPATH-WAITS] line does not print the census's ready counts")
    if "ggml_sycl_ready_query_census & rq = ggml_sycl_ready_query_census_this_thread();" not in f:
        fail("T", "the [MOE-HOSTPATH-WAITS] line does not read the submitting thread's query census")
    loop = re.search(r"for \(int site = 0; site < " + SITE + r"COUNT; \+\+site\) \{ if \(site == " + SITE +
                     r"PER_OP_CHAIN \|\| rq\.count\[site\] == 0 \|\| other_at >= \(int\) sizeof\(other_sites\)\) \{ "
                     r"continue; \}", f)
    if not loop or "ggml_sycl_ready_query_site_name((ggml_sycl_ready_query_site) site), (unsigned long long) " \
                   "rq.count[site]" not in f:
        fail("T", "ready_queries_other= is not the non-zero sites other than the per-op chain, by name")
    if "w.ready_queries = 0;" not in f or "w.ready_skipped = 0;" not in f or \
            "rq = ggml_sycl_ready_query_census{};" not in f:
        fail("T", "the ready counts are not reset with the token's other counts")
    body, _ = body_of(code, READY_ADD_SIG, "T")
    f = flat(body)
    if not re.search(r"if \(!moe_hostpath_waits_enabled\(\)\) \{ return; \}", f) or \
            "g_moe_hostpath_waits.ready_queries += c.queries;" not in f or \
            "g_moe_hostpath_waits.ready_skipped += c.skipped;" not in f:
        fail("T", "moe_hostpath_waits_ready does not add an op's counts to the census")
    call = re.search(r"ggml_sycl_chain_ready_counts ready_counts; ggml_sycl_chain_source_ready_events_if_needed\("
                     r"\*ctx\.stream\(\), dst, ctx\.device, &ready_counts\); moe_hostpath_waits_ready\(ready_counts\);",
                     flat(code))
    if not call:
        fail("T", "the op's source chain does not pass its counts into the census")
    check_names(s)


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
                         ("shared single-event helper", "the shared helper's skip"),
                         ("ready_queries=", "the trace field of the queries still made"),
                         ("ready_skipped=", "the trace field of the skips"),
                         ("ready_queries_other=", "the trace field of the other sites' queries"),
                         ("ready_query_us=", "the trace field of the unfinished queries' time"),
                         ("atoi", "how the value is parsed"),
                         ("oneAPI 2026.1", "the oneAPI version it was verified on"),
                         ("26.35", "the driver it was verified on"),
                         ("re-measure", "the re-measure caveat")):
        if needle not in row:
            fail("C", f"the {KNOB} row does not state {what} ({needle!r})")


def read_sources(root: Path) -> dict:
    out = {}
    for key, path in FILES + (("doc", DOC),):
        p = root / path.relative_to(ROOT)
        out[key] = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
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
    check_helper_calls(s)
    check_query_calls(s)
    check_query(s)
    check_producers(s)
    check_origins(s)
    check_trace(s)
    check_doc(s["doc"])


def self_test(s: dict) -> int:
    def sub(pattern: str, repl: str, flags: int = 0):
        return lambda text: re.sub(pattern, repl, text, count=1, flags=flags)

    def within(sig_text: str, pattern: str, repl: str):
        # Applies `pattern` once inside the definition that starts with `sig_text`, wherever it is in the file.
        def mutate(text: str) -> str:
            at = text.find(sig_text)
            if at < 0:
                return text
            return text[:at] + re.sub(pattern, repl, text[at:], count=1)
        return mutate

    def drop_queue(call: str):
        # Drops the call's last argument, the producer queue, however the source wraps the call.
        head, last = call[:-2].rsplit(", ", 1)
        pat = r"\s*".join(map(re.escape, head.split())) + r",\s*" + re.escape(last) + r"\s*\);"
        return lambda text: re.sub(pat, lambda _m: head + ");", text, count=1)

    def drop_identity(site: str):
        # Replaces the identity argument of the shared helper's call for `site` with nullptr.
        pat = r"ggml_sycl_chain_same_queue_identity\([^()]*\),(\s*)" + SITE + site + r"\)"
        return lambda text: re.sub(pat, r"nullptr,\1" + SITE + site + ")", text, count=1)

    def retag(site: str, new: str):
        return sub(SITE + site + r"\)", SITE + new + ")")

    OTHER_STREAM = "ctx.stream(ctx.device, 1)"
    mutants = [
        ("K", "common_cpp", "knob read per call",
         sub(r"(bool ggml_sycl_chain_same_queue_skip_enabled\(\) \{\n\s*)static const bool enabled",
             r"\1const bool enabled")),
        ("K", "common_cpp", "knob off by default",
         sub(r'(GGML_SYCL_CHAIN_SAME_QUEUE_SKIP"\);\n\s*)return env == nullptr \|\| std::atoi\(env\) != 0;',
             r"\1return env != nullptr && std::atoi(env) != 0;")),
        ("K", "backend", "a second read of the knob",
         sub(r"(static bool moe_hostpath_waits_enabled\(\) \{)",
             r'\1\n    (void) std::getenv("GGML_SYCL_CHAIN_SAME_QUEUE_SKIP");')),
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
         sub(r"(counts->queries \+= 1;\n\s*\}\n\s*)ggml_sycl_chain_ready_event_if_needed\(", r"\1(void) (")),
        ("S", "common", "per-op chain counted as another site",
         sub(r"ggml_sycl_chain_ready_event_if_needed\(q, resolved, nullptr, " + SITE + r"PER_OP_CHAIN\);",
             "ggml_sycl_chain_ready_event_if_needed(q, resolved, nullptr, " + SITE + "OTHER);")),
        ("G", "common", "identity while a graph records",
         sub(r"ggml_sycl_chain_same_queue_skip_enabled\(\) && !ggml_sycl_graph_recording_active\(\) &&",
             "ggml_sycl_chain_same_queue_skip_enabled() &&")),
        ("G", "common", "identity with the knob off",
         sub(r"if \(ggml_sycl_chain_same_queue_skip_enabled\(\) && ", "if (")),
        ("G", "common", "identity for an out-of-order queue",
         sub(r"ggml_sycl::ready_queue_identity\(q, identity\)\)", "(*identity = std::hash<sycl::queue>{}(q), true))")),
        ("G", "common", "per-op chain ignores the identity helper",
         sub(r"(const size_t \* same\s*= )ggml_sycl_chain_same_queue_identity\(q, &q_identity\);", r"\1&q_identity;")),
        ("O", "common", "shared helper never skips",
         sub(r"if \(!resolved\.has_ready_event \|\| ggml_sycl_ready_event_from_queue\(resolved, same_queue\)\)",
             "if (!resolved.has_ready_event)")),
        ("O", "common", "shared helper skips whenever an identity is passed",
         sub(r"\|\| ggml_sycl_ready_event_from_queue\(resolved, same_queue\)\)", "|| same_queue)")),
        ("O", "common", "shared helper queries without the site",
         sub(r"ggml_sycl_should_add_dependency\(resolved\.ready_event, site\)",
             "ggml_sycl_should_add_dependency(resolved.ready_event)")),
        ("E", "binbcast", "binbcast src0 drops the identity", drop_identity("BINBCAST_SRC0")),
        ("E", "binbcast", "binbcast src1 drops the identity", drop_identity("BINBCAST_SRC1")),
        ("E", "mmvq", "mmvq MoE weight drops the identity", drop_identity("MMVQ_MOE_WEIGHT")),
        ("E", "mmvq", "mmvq weight drops the identity", drop_identity("MMVQ_WEIGHT")),
        ("E", "backend", "split mul_mat weight drops the identity", drop_identity("MUL_MAT_WEIGHT")),
        ("E", "backend", "split mul_mat identity of another device's stream",
         sub(r"ggml_sycl_chain_same_queue_identity\(\*stream,(\s*&\w+\),\s*)" + SITE + r"MUL_MAT_WEIGHT\)",
             r"ggml_sycl_chain_same_queue_identity(*ctx.stream(),\1" + SITE + "MUL_MAT_WEIGHT)")),
        ("E", "binbcast", "binbcast src1 tagged as another site", retag("BINBCAST_SRC1", "OTHER")),
        ("E", "add_id", "an added helper call with no site",
         sub(r"(  collect_ready_deps\(src0\);\n)", r"\1  ggml_sycl_chain_ready_event_if_needed(q, ggml_sycl_resolve(src0.raw(), ctx.device));\n")),
        ("D", "moe_xmx", "an untagged status query", sub(r",(\s*)" + SITE + r"MOE_XMX\)", ")")),
        ("D", "fattn", "a fattn query under another site's tag", retag("FATTN_SET_ROWS", "FATTN_PACKED_ZERO")),
        ("D", "add_id", "a status query tagged other", retag("ADD_ID", "OTHER")),
        ("Q", "common", "queries not counted",
         sub(r"ggml_sycl_ready_query_census_this_thread\(\)\.count\[site\] \+= 1;", ";")),
        ("Q", "common", "queries counted with the trace off",
         sub(r"if \(traced\) \{\n(\s*)ggml_sycl_ready_query_census_this_thread\(\)\.count\[site\] \+= 1;",
             r"{\n\1ggml_sycl_ready_query_census_this_thread().count[site] += 1;")),
        ("Q", "common", "unfinished queries' time not added",
         sub(r"ggml_sycl_ready_query_census_this_thread\(\)\.pending_us \+=", "(void)")),
        ("Q", "common", "finished queries' time added too", sub(r"if \(pending\) \{", "{")),
        ("Q", "common", "recording branch drops in-flight dependencies",
         sub(r"(catch \(\.\.\.\) \{\n(?:.*\n)*?\s*\}\n\s*)return true;", r"\1return false;")),
        ("Q", "common_cpp", "census shared by every thread",
         sub(r"static thread_local ggml_sycl_ready_query_census census;", "static ggml_sycl_ready_query_census census;")),
        ("P", "common", "producer not handed to the handle",
         sub(r"handle\.set_ready_event\(event, \*producer\);", "handle.set_ready_event(event);")),
    ]
    for key, what, call in PRODUCERS:
        mutants.append(("P", key, f"{what} unrecorded", drop_queue(call)))
    mutants += [
        ("P", "binbcast", "binbcast's main_stream is another stream",
         sub(r"(dpct::queue_ptr main_stream = )ctx\.stream\(\);", r"\1" + OTHER_STREAM + ";")),
        ("P", "binbcast", "binbcast's marker on another queue",
         sub(r"ggml_sycl_submit_binbcast_event\(\*main_stream, mode\)",
             "ggml_sycl_submit_binbcast_event(*" + OTHER_STREAM + ", mode)")),
        ("P", "binbcast", "binbcast hands op() another queue",
         sub(r"main_stream, file, line, function\);", OTHER_STREAM + ", file, line, function);")),
        ("P", "binbcast", "op() submits on another queue",
         sub(r"(kernel_event = ggml_sycl_submit_binbcast_kernel\(\s*)\*stream,", r"\1*ggml_sycl_get_cpu_queue(),")),
        ("P", "profiler", "profiler submits on another queue",
         sub(r"sycl::event event = submit_fn\(q\);", "sycl::event event = submit_fn(*ggml_sycl_get_cpu_queue());")),
        ("P", "add_id", "ADD_ID's q is another stream",
         sub(r"(sycl::queue& q = )\*ctx\.stream\(\);", r"\1*" + OTHER_STREAM + ";")),
        ("P", "add_id", "ADD_ID's kernel on another queue",
         sub(r"sycl::event evt = q\.submit\(", "sycl::event evt = " + OTHER_STREAM + "->submit(")),
        ("P", "backend", "down weighted sum on another stream",
         sub(r"(sycl::event evt = ggml_sycl_moe_down_weighted_sum_sycl\(\s*)\*ctx\.stream\(\),",
             r"\1*" + OTHER_STREAM + ",")),
        ("P", "backend", "down weighted sum submits on another queue",
         sub(r"return queue\.submit\(\[&\]\(sycl::handler & h\) \{\n(\s*)h\.parallel_for<moe_down_weighted_sum_kernel>",
             r"return ggml_sycl_get_cpu_queue()->submit([&](sycl::handler & h) {\n\1h.parallel_for<moe_down_weighted_sum_kernel>")),
        ("P", "mmvq", "decode pair GLU on another stream",
         within("bool mmvq_moe_batched_dispatch_pair_glu_mxfp4_soa(ggml_backend_sycl_context &",
                r"(const queue_ptr stream\s*= )ctx\.stream\(\);", r"\1" + OTHER_STREAM + ";")),
        ("P", "mmvq", "decode pair down on another stream",
         within("bool mmvq_moe_batched_dispatch_down_from_cached_q8_mxfp4(ggml_backend_sycl_context &",
                r"(const queue_ptr stream\s*= )ctx\.stream\(\);", r"\1" + OTHER_STREAM + ";")),
        ("P", "mmvq", "decode pair GLU completes on another event",
         sub(r"\*completion_event = down_q8_event_set \? down_q8_event : kernel_event;",
             "*completion_event = ggml_sycl_submit_queue_sync_event(*ggml_sycl_get_cpu_queue());")),
        ("P", "mmvq", "decode pair down completes on another event",
         sub(r"\*completion_event = down_event;", "*completion_event = ggml_sycl_submit_queue_sync_event(*ggml_sycl_get_cpu_queue());")),
        ("T", "backend", "census lacks the skip count",
         sub(r"uint64_t(\s+)ready_skipped(\s*)= 0;", r"uint64_t\1ready_skip_n\2= 0;")),
        ("T", "backend", "line drops the queries count", sub(r' ready_queries=%llu "', '"')),
        ("T", "backend", "line drops the other sites", sub(r"ready_queries_other=%s ", "")),
        ("T", "backend", "line drops the unfinished queries' time", sub(r"ready_query_us=%\.0f ?", "")),
        ("T", "backend", "counts not reset", sub(r"w\.ready_queries(\s*)= 0;", r"(void) w\1;")),
        ("T", "backend", "query census not reset", sub(r"rq\s*= ggml_sycl_ready_query_census\{\};", ";")),
        ("T", "backend", "zero sites printed", sub(r" \|\| rq\.count\[site\] == 0", "")),
        ("T", "backend", "per-op chain printed as another site",
         sub(r"site == " + SITE + r"PER_OP_CHAIN \|\| ", "")),
        ("T", "backend", "op counts not added", sub(r"g_moe_hostpath_waits\.ready_skipped \+= c\.skipped;", ";")),
        ("T", "common", "two sites share a name", sub(r'return "binbcast_src1";', 'return "binbcast_src0";')),
        ("C", "doc", "doc row missing", sub(r"\| `GGML_SYCL_CHAIN_SAME_QUEUE_SKIP", "| `GGML_SYCL_CHAIN_SAME_Q")),
        ("C", "doc", "doc row loses the version caveat",
         sub(r"(\| `GGML_SYCL_CHAIN_SAME_QUEUE_SKIP[^\n]*?)oneAPI 2026\.1", r"\1oneAPI")),
        ("C", "doc", "doc row loses the shared helper",
         sub(r"(\| `GGML_SYCL_CHAIN_SAME_QUEUE_SKIP[^\n]*?)shared single-event helper", r"\1helper")),
        ("C", "doc", "doc row loses the other sites' field",
         sub(r"(\| `GGML_SYCL_CHAIN_SAME_QUEUE_SKIP[^\n]*?)ready_queries_other=", r"\1ready_queries_o")),
        ("C", "doc", "doc row loses the parse", sub(r"(\| `GGML_SYCL_CHAIN_SAME_QUEUE_SKIP[^\n]*?)atoi", r"\1a_to_i")),
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
    print("PASS: the per-op source chain and the shared helper skip ready events their own in-order stream produced, "
          f"behind GGML_SYCL_CHAIN_SAME_QUEUE_SKIP, and query every other one as before, under its site {TAG}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
