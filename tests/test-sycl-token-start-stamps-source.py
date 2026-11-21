#!/usr/bin/env python3
"""Token-start stamps of a replayed MoE decode token, under GGML_SYCL_MOE_IDS_COPY_TRACE.

THE GAP.  The per-job stamps of the trace start at the token's first job submit.  On Qwen3.8 decode the device idles
for about 10 ms before it, while the host prepares the token (llama's inputs, including the PLE storage gather),
runs the backend's prologue (about twenty walks of a 6783-node graph, two of them full hashes) and enqueues the
blocks that come before the first host expert.  Eight stamps, kept raw and printed relative to that first job submit,
split the time.  They add no work to the untraced path and change nothing the traced path submits or waits.

THE CONTRACT.  With comments blanked, in ggml-sycl.cpp:
  G. Every stamp site is inside the trace flag's branch, and the flag is the one the trace already caches:
       a. ggml_backend_sycl_graph_compute_unchecked: `if (moe_hostpath_waits_enabled()) { moe_token_stamps_enter(); }`
          as the first statement (the backend entry);
       b. `if (moe_hostpath_waits_enabled()) { moe_token_stamps_note_replay(); }` right before
          moe_graph_replay_segment_slot(sycl_ctx, cgraph, *slot) (the replay begin);
       b2. `if (moe_hostpath_waits_enabled()) { moe_token_stamps_note_replay_end(); }` right after that call (the
          replay end, so replay_end - replay_begin is the host's enqueue time);
       b3. `if (moe_hostpath_waits_enabled()) { moe_token_stamps_note_return(); }` right before the compute's final
          `return GGML_STATUS_SUCCESS;`, which the replay path takes (gc_return);
       c. moe_graph_replay_segment_slot: two moe_token_stamps_note_enqueue calls, each inside `if (enqueue_stamps)`,
          the segment one right before graph_exec_submit; enqueue_stamps comes from
          moe_token_stamps_enqueue_target(), which returns null unless the flag is on and the level is 2 or more;
       d. ggml_backend_sycl_lazy_storage_gather: the clock reads and moe_token_stamps_note_ple, on `traced`, which is
          `moe_hostpath_waits_enabled()` (the PLE gather, stamped on the backend side so llama code reads no
          GGML_SYCL variable);
       e. ggml_backend_sycl_get_tensor_async: moe_token_stamps_note_d2h, on `traced`, after the copy returns (the
          staged logits wait has returned by then).
  N. No stamp helper waits, queries an event, allocates or locks: the bodies of the stamp helpers hold no wait, no
     event query, no allocator, no container growth, no lock, and no sycl:: call.
  B. The per-block buffer is a fixed-size array in a record that lives in static thread-local storage, and the
     writer checks its bound.
  K. One clock: the stamps read ggml_sycl::cpu_expert_pool_trace_now_ns(), the job stamps' clock, and no other.
  V. One switch: GGML_SYCL_MOE_IDS_COPY_TRACE is read at the three sites it was read at (the cached flag, the cached
     level, and the lazy tracer elsewhere), and no new getenv appears in the stamp code.
  L. The lines.  [MOE-TOKEN-START] is printed once, from the readback's token-line block, and
     [MOE-TOKEN-ENQUEUE] once, inside `if (enqueue_lines)`, which the readback sets to
     `moe_hostpath_trace_level() >= 2`.  Their field lists parse: a synthetic
     line made from each format string is parsed back by this pin's parser with the fields it names, signed times in
     microseconds, "-" for a stamp not taken.  prologue is replay_begin - gc_enter, ple_misses is rows - hits.
  D. docs/backend/sycl-env-vars.md names both lines and every field of the first.

NOT VACUOUS.  --self-test runs the gate on in-memory mutants of the real sources and requires each to FAIL, then
requires the unmodified tree to pass.

RED on the tree before the change:
  FAIL [pin G]: the backend entry's first statement is not `if (moe_hostpath_waits_enabled()) { moe_token_stamps_enter(); }`
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "ggml" / "src" / "ggml-sycl" / "ggml-sycl.cpp"
DOC = ROOT / "docs" / "backend" / "sycl-env-vars.md"

ENV_NAME = "GGML_SYCL_MOE_IDS_COPY_TRACE"
# How many times the backend reads the variable: the cached flag, the cached level, and the lazy tracer's own read.
ENV_READS = 3

TOKEN_FIELDS = ("token", "t0", "gc_enter", "replay_begin", "replay_end", "gc_return", "prologue", "enqueue",
                "ple_begin", "ple_end", "ple_busy", "ple_calls", "ple_rows", "ple_hits", "ple_misses", "ple_reads",
                "logits_ret", "logits_staged", "logits_calls", "events", "events_dropped")
ENQUEUE_FIELDS = ("token", "i", "kind", "node", "layer", "t")

# The helpers whose bodies must hold no wait, no event query, no allocation and no lock.
HELPERS = ("moe_token_stamps_current", "moe_token_stamps_enter", "moe_token_stamps_note_replay",
           "moe_token_stamps_note_replay_end", "moe_token_stamps_note_return", "moe_token_stamps_enqueue_target",
           "moe_token_stamps_layer_of", "moe_token_stamps_note_enqueue",
           "moe_token_stamps_note_ple", "moe_token_stamps_note_d2h", "moe_token_stamps_rel",
           "moe_token_stamps_print")
FORBIDDEN = (
    (r"\.wait\s*\(", "a wait"),
    (r"wait_and_throw", "a wait"),
    (r"\bwait_deps\b", "a wait"),
    (r"\bwait_all\b", "a wait"),
    (r"\bsycl::", "a sycl call"),
    (r"get_info|ext_oneapi_get_state|\.get_status", "an event status query"),
    (r"\bmalloc|\bcalloc|\brealloc|\bnew\b|make_shared|make_unique|unified_alloc", "an allocation"),
    (r"std::vector|std::string|std::deque|std::map|push_back|emplace_back|\.resize\s*\(|\.reserve\s*\(",
     "container growth"),
    (r"lock_guard|unique_lock|scoped_lock|std::mutex|\.lock\s*\(", "a lock"),
    (r"\bgetenv\b", "an environment read"),
)


class ContractError(Exception):
    pass


def fail(pin: str, msg: str) -> None:
    raise ContractError(f"FAIL [pin {pin}]: {msg}")


# Comments blank to spaces, keeping offsets; strings and char literals are matched first so a comment marker
# inside one is not taken for a comment.
_TOKEN = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n]){1,4}\'', re.S)


def blank_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        t = m.group(0)
        if t.startswith("//") or t.startswith("/*"):
            return "".join("\n" if c == "\n" else " " for c in t)
        return t

    return _TOKEN.sub(repl, src)


def strip_strings(src: str) -> str:
    return re.sub(r'"(?:\\.|[^"\\\n])*"', lambda m: '"' + " " * (len(m.group(0)) - 2) + '"', src)


def body_of(code: str, signature: str, pin: str) -> str:
    """The body of the first definition matching `signature` (which ends at its opening brace)."""
    m = re.search(signature, code)
    if not m:
        fail(pin, f"no definition matching {signature!r}")
    i = code.index("{", m.end() - 1)
    depth = 0
    # Only this function's extent is scanned, so a huge file costs nothing here.
    j = i
    while j < len(code):
        c = code[j]
        if c == '"':
            j += 1
            while j < len(code) and code[j] != '"':
                if code[j] == "\\":
                    j += 1
                j += 1
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return code[i + 1 : j]
        j += 1
    fail(pin, f"unbalanced body for {signature!r}")
    return ""


def def_sig(name: str) -> str:
    return r"\bstatic\s+[\w:<>\*& ]+?\b" + re.escape(name) + r"\s*\([^)]*\)\s*(?:noexcept\s*)?\{"


def call_sites(code: str, name: str) -> list[re.Match]:
    """Every call of `name`, definitions excluded."""
    out = []
    for m in re.finditer(r"(?<![\w:])" + re.escape(name) + r"\s*\(", code):
        line_start = code.rfind("\n", 0, m.start()) + 1
        if re.match(r"\s*static\s", code[line_start : m.start()]):
            continue
        out.append(m)
    return out


def guarded_call(code: str, name: str, guard: str, pin: str, what: str, expect: int = 1) -> list[re.Match]:
    """Every call site of `name` is the sole statement of an `if (guard) {` block, and there are `expect` of them."""
    sites = call_sites(code, name)
    if len(sites) != expect:
        fail(pin, f"{what}: {len(sites)} call site(s) of {name}, want {expect}")
    for m in sites:
        before = code[max(0, m.start() - 160) : m.start()]
        if not re.search(r"if\s*\(\s*" + guard + r"\s*\)\s*\{\s*$", before):
            fail(pin, f"{what}: the call of {name} is not inside `if ({guard.replace(chr(92), '')}) {{`")
        end = code.index(";", m.end())
        after = code[end + 1 : end + 40]
        if not re.match(r"\s*\}", after):
            fail(pin, f"{what}: the call of {name} is not the sole statement of its guarded block")
    return sites


FLAG_GUARD = r"moe_hostpath_waits_enabled\s*\(\s*\)"


def check_sites(code: str) -> None:
    # a. the backend entry: the first statement of the compute function.
    entry = body_of(code, r"\bstatic\s+ggml_status\s+ggml_backend_sycl_graph_compute_unchecked\s*\([^)]*\)\s*\{", "G")
    lead = entry.lstrip()
    if not re.match(r"if\s*\(\s*" + FLAG_GUARD + r"\s*\)\s*\{\s*moe_token_stamps_enter\s*\(\s*\)\s*;\s*\}", lead):
        fail("G", "the backend entry's first statement is not `if (moe_hostpath_waits_enabled()) "
                  "{ moe_token_stamps_enter(); }`")
    guarded_call(code, "moe_token_stamps_enter", FLAG_GUARD, "G", "backend entry")

    # b. the replay begin, right before the slot replay call.
    sites = guarded_call(code, "moe_token_stamps_note_replay", FLAG_GUARD, "G", "replay begin")
    tail = code[sites[0].end() : sites[0].end() + 160]
    if not re.match(r"[^;]*;\s*\}\s*moe_graph_replay_segment_slot\s*\(\s*sycl_ctx\s*,\s*cgraph\s*,\s*\*slot\s*\)\s*;",
                    tail):
        fail("G", "the replay-begin stamp is not right before moe_graph_replay_segment_slot(sycl_ctx, cgraph, *slot)")

    # b2. the replay end, right after the slot replay call.
    sites = guarded_call(code, "moe_token_stamps_note_replay_end", FLAG_GUARD, "G", "replay end")
    head = code[max(0, sites[0].start() - 260) : sites[0].start()]
    if not re.search(r"moe_graph_replay_segment_slot\s*\(\s*sycl_ctx\s*,\s*cgraph\s*,\s*\*slot\s*\)\s*;\s*"
                     r"if\s*\(\s*" + FLAG_GUARD + r"\s*\)\s*\{\s*$", head):
        fail("G", "the replay-end stamp is not right after moe_graph_replay_segment_slot(sycl_ctx, cgraph, *slot)")

    # b3. the final return of the compute, which the replay path takes.
    sites = guarded_call(code, "moe_token_stamps_note_return", FLAG_GUARD, "G", "compute return")
    if not re.match(r"[^;]*;\s*\}\s*return\s+GGML_STATUS_SUCCESS\s*;\s*\}", code[sites[0].end() : sites[0].end() + 120]):
        fail("G", "the return stamp is not right before the compute's final `return GGML_STATUS_SUCCESS;`")
    if sites[0].start() < code.index("static ggml_status ggml_backend_sycl_graph_compute_unchecked"):
        fail("G", "the return stamp is outside ggml_backend_sycl_graph_compute_unchecked")

    # c. the per-block enqueue stamps, in the replay loop.
    replay = body_of(code, r"\bstatic\s+void\s+moe_graph_replay_segment_slot\s*\([^)]*\)\s*\{", "G")
    if not re.search(r"moe_token_stamps\s*\*\s*const\s+enqueue_stamps\s*=\s*moe_token_stamps_enqueue_target\s*\(\s*\)\s*;",
                     replay):
        fail("G", "the replay loop does not take enqueue_stamps from moe_token_stamps_enqueue_target()")
    notes = guarded_call(replay, "moe_token_stamps_note_enqueue", r"enqueue_stamps", "G", "replay loop", expect=2)
    kinds = [re.match(r"[^,]*,\s*(\w+)", replay[m.end() :]).group(1) for m in notes]
    if kinds != ["MOE_TOKEN_ENQUEUE_SEGMENT", "MOE_TOKEN_ENQUEUE_BOUNDARY"]:
        fail("G", f"the replay loop stamps {kinds}, want its recorded-segment submit and then its boundary dispatch")
    after_seg = replay[notes[0].end() :]
    if not re.match(r"[^;]*;\s*\}\s*ggml_sycl::graph_exec_submit\s*\(", after_seg):
        fail("G", "the segment stamp is not right before graph_exec_submit")
    target = body_of(code, def_sig("moe_token_stamps_enqueue_target"), "G")
    if not re.match(
        r"\s*if\s*\(\s*!\s*" + FLAG_GUARD + r"\s*\|\|\s*moe_hostpath_trace_level\s*\(\s*\)\s*<\s*2\s*\)\s*\{\s*"
        r"return\s+nullptr\s*;\s*\}",
        target,
    ):
        fail("G", "moe_token_stamps_enqueue_target does not return null unless the flag is on and the level is 2+")

    # d. the PLE gather, on the backend side.
    gather = body_of(code, r"\bint\s+ggml_backend_sycl_lazy_storage_gather\s*\([^)]*\)\s*\{", "G")
    if not re.search(r"const\s+bool\s+traced\s*=\s*moe_hostpath_waits_enabled\s*\(\s*\)\s*;", gather):
        fail("G", "the storage gather's `traced` is not moe_hostpath_waits_enabled()")
    guarded_call(gather, "moe_token_stamps_note_ple", r"traced", "G", "storage gather")
    clocks = [m.start() for m in re.finditer(r"cpu_expert_pool_trace_now_ns\s*\(", gather)]
    if len(clocks) != 2:
        fail("G", f"the storage gather reads the clock {len(clocks)} times, want 2 (begin, end)")
    if not re.search(r"traced\s*\?\s*ggml_sycl::cpu_expert_pool_trace_now_ns\s*\(\s*\)\s*:\s*0", gather):
        fail("G", "the storage gather's begin stamp is not read only when traced")

    # e. the logits copy's staged wait has returned when mem_copy_async does.
    get_tensor = body_of(code, r"\bstatic\s+void\s+ggml_backend_sycl_get_tensor_async\s*\([^)]*\)\s*try\s*\{", "G")
    if not re.search(r"const\s+bool\s+traced\s*=\s*moe_hostpath_waits_enabled\s*\(\s*\)\s*;", get_tensor):
        fail("G", "get_tensor_async's `traced` is not moe_hostpath_waits_enabled()")
    sites = guarded_call(get_tensor, "moe_token_stamps_note_d2h", r"traced", "G", "get_tensor_async")
    if "mem_copy_async" not in get_tensor[: sites[0].start()]:
        fail("G", "the readback stamp is not after the copy")
    if not re.search(r"traced\s*\?\s*ggml_sycl::mem_copy_staged_wait_census_this_thread\s*\(\s*\)\.staged_d2h\s*:\s*0",
                     get_tensor):
        fail("G", "get_tensor_async reads the staged-copy count only when traced")


def check_helpers(code: str) -> None:
    for name in HELPERS:
        body = body_of(code, def_sig(name), "N")
        for pat, what in FORBIDDEN:
            if re.search(pat, body):
                fail("N", f"{name} holds {what} (/{pat}/)")


def check_buffer(code: str) -> None:
    m = re.search(r"constexpr\s+int\s+MOE_TOKEN_ENQUEUE_EVENTS\s*=\s*(\d+)\s*;", code)
    if not m or int(m.group(1)) < 64:
        fail("B", "no fixed MOE_TOKEN_ENQUEUE_EVENTS of at least 64")
    rec = body_of(code, r"\bstruct\s+moe_token_stamps\s*\{", "B")
    if not re.search(r"moe_token_enqueue_event\s+events\s*\[\s*MOE_TOKEN_ENQUEUE_EVENTS\s*\]\s*;", rec):
        fail("B", "the per-block events are not a fixed-size array member")
    if re.search(r"std::|\*\s*events", rec):
        fail("B", "the stamp record holds a container or a pointer")
    if not re.search(r"static\s+thread_local\s+moe_token_stamp_ring\s+g_moe_token_stamp_ring\s*;", code):
        fail("B", "the stamp ring is not static thread-local storage")
    ring = body_of(code, r"\bstruct\s+moe_token_stamp_ring\s*\{", "B")
    if not re.search(r"moe_token_stamps\s+slot\s*\[\s*2\s*\]\s*;", ring):
        fail("B", "the ring is not two fixed records")
    note = body_of(code, def_sig("moe_token_stamps_note_enqueue"), "B")
    if not re.search(r"if\s*\(\s*rec\.n_events\s*>=\s*MOE_TOKEN_ENQUEUE_EVENTS\s*\)\s*\{\s*\+\+rec\.events_dropped\s*;"
                     r"\s*return\s*;\s*\}", note):
        fail("B", "moe_token_stamps_note_enqueue does not check its bound before it writes")


def check_clock(code: str) -> None:
    for name in HELPERS:
        body = body_of(code, def_sig(name), "K")
        if re.search(r"steady_clock|high_resolution_clock|system_clock|clock_gettime|rdtsc", body):
            fail("K", f"{name} reads a clock of its own")
    for name in ("moe_token_stamps_enter", "moe_token_stamps_note_replay", "moe_token_stamps_note_replay_end",
                 "moe_token_stamps_note_return", "moe_token_stamps_note_enqueue", "moe_token_stamps_note_d2h"):
        if "ggml_sycl::cpu_expert_pool_trace_now_ns()" not in body_of(code, def_sig(name), "K"):
            fail("K", f"{name} does not read the job stamps' clock, ggml_sycl::cpu_expert_pool_trace_now_ns()")


def check_switch(code: str) -> None:
    n = len(re.findall(r'getenv\s*\(\s*"' + ENV_NAME + r'"\s*\)', code))
    if n != ENV_READS:
        fail("V", f"{ENV_NAME} is read {n} times in ggml-sycl.cpp, want {ENV_READS}: the stamps add no reader")
    stamp_region = re.search(r"constexpr\s+int\s+MOE_TOKEN_ENQUEUE_EVENTS", code)
    if not stamp_region:
        fail("V", "no stamp block")
    block = code[stamp_region.start() : stamp_region.start() + 30000]
    end = block.find("// Adds one op's source ready-event counts")
    block = block[: end if end > 0 else len(block)]
    if re.search(r"\bgetenv\b", block):
        fail("V", "the stamp block reads an environment variable")


def formats_of(code: str, tag: str) -> list[str]:
    """The joined format string of every GGML_LOG_WARN that prints `tag`."""
    sites = [m for m in re.finditer(r'GGML_LOG_WARN\(\s*((?:"(?:\\.|[^"\\])*"\s*)+)', code)
             if m.group(1).lstrip().startswith('"' + tag)]
    return ["".join(re.findall(r'"((?:\\.|[^"\\])*)"', m.group(1))) for m in sites]


def format_of(code: str, tag: str) -> str:
    """The one format string that prints `tag`."""
    got = formats_of(code, tag)
    if len(got) != 1:
        fail("L", f"{tag} is printed at {len(got)} site(s), want 1")
    return got[0]


def parse_us(value: str):
    if value == "-":
        return None
    m = re.fullmatch(r"(-?\d+)us", value)
    if not m:
        raise ValueError(f"not a time in us: {value!r}")
    return int(m.group(1))


def parse_line(line: str, tag: str, fields: tuple[str, ...], times: tuple[str, ...], ints: tuple[str, ...]) -> dict:
    """Parses one synthetic line; every field named in `fields`, in that order, signed times in us."""
    m = re.match(r"\[" + re.escape(tag) + r"\] (.*)$", line.strip())
    if not m:
        raise ValueError(f"not a {tag} line: {line!r}")
    pairs = re.findall(r"(\w+)=(\S+)", m.group(1))
    if tuple(k for k, _ in pairs) != fields:
        raise ValueError(f"fields {[k for k, _ in pairs]} differ from {list(fields)}")
    out = {}
    for k, v in pairs:
        if k in times:
            out[k] = parse_us(v)
        elif k in ints:
            out[k] = int(v)
        else:
            out[k] = v
    return out


SAMPLES = {
    "token": "7", "t0": "job", "gc_enter": "-10532us", "replay_begin": "-9400us", "replay_end": "-2000us",
    "gc_return": "36000us", "prologue": "1132us", "enqueue": "7400us",
    "ple_begin": "-12000us", "ple_end": "-11000us", "ple_busy": "1000", "ple_calls": "1", "ple_rows": "16",
    "ple_hits": "2", "ple_misses": "14", "ple_reads": "14", "logits_ret": "38000us", "logits_staged": "1",
    "logits_calls": "1", "events": "3", "events_dropped": "0",
    "i": "2", "kind": "seg", "node": "120", "layer": "3", "t": "-8123",
}


def synthesize(fmt: str) -> str:
    """The format string with each `name=%conv` filled by SAMPLES[name]; units after the conversion stay."""
    def cb(m: re.Match) -> str:
        name = m.group(1)
        if name not in SAMPLES:
            fail("L", f"no sample value for field {name!r}")
        return f"{name}={SAMPLES[name]}"

    line = re.sub(r"(\w+)=%(?:\.\d+f|ll[ud]|[dsu])", cb, fmt)
    line = line.replace("\\n", "")
    if "%" in line:
        fail("L", f"a conversion is left unfilled in {line!r}")
    return line


def check_lines(code: str) -> None:
    joined = formats_of(code, "[MOE-TOKEN-START]")
    full = [f for f in joined if "gc_enter=" in f]
    none = [f for f in joined if "stamps=none" in f]
    if len(full) != 1 or len(none) != 1 or len(joined) != 2:
        fail("L", f"[MOE-TOKEN-START] is printed at {len(joined)} site(s), want its fields once and stamps=none once")
    line = synthesize(full[0])
    try:
        got = parse_line(line, "MOE-TOKEN-START", TOKEN_FIELDS,
                         ("gc_enter", "replay_begin", "replay_end", "gc_return", "prologue", "enqueue",
                          "ple_begin", "ple_end", "ple_busy", "logits_ret"),
                         ("token", "ple_calls", "ple_rows", "ple_hits", "ple_misses", "ple_reads", "logits_staged",
                          "logits_calls", "events", "events_dropped"))
    except ValueError as exc:
        fail("L", f"the [MOE-TOKEN-START] format does not parse: {exc}")
    if got["prologue"] != got["replay_begin"] - got["gc_enter"]:
        fail("L", "the synthetic prologue is not replay_begin - gc_enter")
    if got["enqueue"] != got["replay_end"] - got["replay_begin"]:
        fail("L", "the synthetic enqueue is not replay_end - replay_begin")
    # A stamp not taken parses as None.
    nt = parse_line(line.replace("replay_begin=-9400us", "replay_begin=-").replace("prologue=1132us", "prologue=-")
                    .replace("replay_end=-2000us", "replay_end=-").replace("enqueue=7400us", "enqueue=-"),
                    "MOE-TOKEN-START", TOKEN_FIELDS, ("replay_begin", "replay_end", "prologue", "enqueue"), ())
    if any(nt[k] is not None for k in ("replay_begin", "replay_end", "prologue", "enqueue")):
        fail("L", "a stamp not taken does not parse as absent")

    eq_fmt = format_of(code, "[MOE-TOKEN-ENQUEUE]")
    try:
        parse_line(synthesize(eq_fmt), "MOE-TOKEN-ENQUEUE", ENQUEUE_FIELDS, (), ())
    except ValueError as exc:
        fail("L", f"the [MOE-TOKEN-ENQUEUE] format does not parse: {exc}")
    if not re.search(r"\bt=%\.0fus", eq_fmt):
        fail("L", "the [MOE-TOKEN-ENQUEUE] time is not a signed %.0fus")

    printer = body_of(code, def_sig("moe_token_stamps_print"), "L")
    if not re.search(r"rec\.t_replay_ns\s*-\s*rec\.t_gc_enter_ns", printer):
        fail("L", "prologue is not computed from t_replay_ns - t_gc_enter_ns")
    if not re.search(r"rec\.t_replay_end_ns\s*-\s*rec\.t_replay_ns", printer):
        fail("L", "enqueue is not computed from t_replay_end_ns - t_replay_ns")
    if not re.search(r"rec\.ple_rows\s*>\s*rec\.ple_hits\s*\?\s*rec\.ple_rows\s*-\s*rec\.ple_hits", printer):
        fail("L", "ple_misses is not rows - hits")
    enq = [m.start() for m in re.finditer(r"\[MOE-TOKEN-ENQUEUE\]", printer)]
    if len(enq) != 1:
        fail("L", "the per-block line is not printed at exactly one site")
    heads = re.findall(r"if\s*\(\s*enqueue_lines\s*\)\s*\{", printer[: enq[0]])
    if not heads or printer[: enq[0]].rfind("{") < printer[: enq[0]].rfind(heads[-1]):
        fail("L", "the [MOE-TOKEN-ENQUEUE] line is not inside `if (enqueue_lines)`")
    # The token's line is printed once, from the readback's token-line block, anchored on the job stamps' t0.
    readback = body_of(code, r"\bstatic\s+void\s+moe_hostpath_waits_readback\s*\([^)]*\)\s*\{", "L")
    calls = re.findall(r"moe_token_stamps_print\s*\(([^;]*)\)\s*;", readback)
    if len(calls) != 1 or "pt.job_stamps.front().t_submit_ns" not in calls[0]:
        fail("L", "the readback does not print the stamps once, relative to the first job submit")
    if not re.search(r",\s*moe_hostpath_trace_level\s*\(\s*\)\s*>=\s*2\s*$", calls[0]):
        fail("L", "the readback does not allow the per-block lines only at trace level 2 and above")
    if len(call_sites(code, "moe_token_stamps_print")) != 1:
        fail("L", "moe_token_stamps_print is called outside the readback")
    if not re.search(r"slot\s*\[\s*g_moe_token_stamp_ring\.cur\s*\^\s*1\s*\]", readback):
        fail("L", "the readback does not print the record before the newest")
    if not re.search(r"slot\s*\[\s*g_moe_token_stamp_ring\.cur\s*\]\.readbacks_seen\s*=\s*true", readback):
        fail("L", "the readback does not mark its compute's record")


def check_doc(doc: str) -> None:
    row = next((ln for ln in doc.splitlines() if ln.startswith("| `" + ENV_NAME)), "")
    if not row:
        fail("D", "no env var row")
    for needle in ("[MOE-TOKEN-START]", "[MOE-TOKEN-ENQUEUE]") + tuple(f + "=" for f in TOKEN_FIELDS[1:]) + (
        "enqueue",):
        if needle not in row:
            fail("D", f"the {ENV_NAME} row does not mention {needle}")


def gate(code: str, doc: str) -> None:
    check_sites(code)
    check_helpers(code)
    check_buffer(code)
    check_clock(code)
    check_switch(code)
    check_lines(code)
    check_doc(doc)


def load() -> tuple[str, str]:
    return blank_comments(BACKEND.read_text()), DOC.read_text()


def sub1(text: str, old: str, new: str) -> str:
    """Replaces the first `old` (a regex) and fails the self-test if there is none to replace."""
    out, n = re.subn(old, lambda _m: new, text, count=1, flags=re.S)
    if n != 1:
        raise RuntimeError(f"mutant anchor not found: {old!r}")
    return out


def mutants(code: str, doc: str):
    """(name, the pin that must catch it, mutated code, mutated doc)."""
    G = FLAG_GUARD
    yield "entry stamp unguarded", "G", sub1(
        code, r"if\s*\(\s*" + G + r"\s*\)\s*\{\s*moe_token_stamps_enter\(\);\s*\}", "moe_token_stamps_enter();"), doc
    yield "replay-begin stamp unguarded", "G", sub1(
        code, r"if\s*\(\s*" + G + r"\s*\)\s*\{\s*moe_token_stamps_note_replay\(\);\s*\}",
        "moe_token_stamps_note_replay();"), doc
    yield "replay-end stamp unguarded", "G", sub1(
        code, r"if\s*\(\s*" + G + r"\s*\)\s*\{\s*moe_token_stamps_note_replay_end\(\);\s*\}",
        "moe_token_stamps_note_replay_end();"), doc
    yield "replay-end stamp before the replay", "G", sub1(
        code, r"(moe_graph_replay_segment_slot\(sycl_ctx, cgraph, \*slot\);)(\s*if \(" + G + r"\) \{\s*"
        r"moe_token_stamps_note_replay_end\(\);\s*\})", r"\2 \1"), doc
    yield "return stamp unguarded", "G", sub1(
        code, r"if\s*\(\s*" + G + r"\s*\)\s*\{\s*moe_token_stamps_note_return\(\);\s*\}",
        "moe_token_stamps_note_return();"), doc
    yield "return stamp away from the return", "G", sub1(
        code, r"(if \(" + G + r"\) \{\s*moe_token_stamps_note_return\(\);\s*\})(\s*return GGML_STATUS_SUCCESS;)",
        r"\2 \1 int g_pad = 0;"), doc
    yield "a wait in the return stamp", "N", sub1(
        code, r"static void moe_token_stamps_note_return\(\) \{",
        "static void moe_token_stamps_note_return() {\n    sycl::queue().wait();"), doc
    yield "enqueue not replay_end - replay_begin", "L", sub1(
        code, r"rec\.t_replay_end_ns - rec\.t_replay_ns", "rec.t_replay_ns - rec.t_replay_end_ns"), doc
    yield "the enqueue field dropped", "L", sub1(code, r"enqueue=%s ", ""), doc
    yield "segment stamp unguarded", "G", sub1(
        code, r"if \(enqueue_stamps\) \{\s*(moe_token_stamps_note_enqueue\([^;]*MOE_TOKEN_ENQUEUE_SEGMENT[^;]*;)\s*\}",
        "moe_token_stamps_note_enqueue(*enqueue_stamps, MOE_TOKEN_ENQUEUE_SEGMENT, 0, nullptr);"), doc
    yield "segment stamp after the submit", "G", sub1(
        code, r"(if \(enqueue_stamps\) \{\s*moe_token_stamps_note_enqueue\([^;]*MOE_TOKEN_ENQUEUE_SEGMENT[^;]*;\s*\})"
        r"(\s*ggml_sycl::graph_exec_submit\([^;]*;)", "ggml_sycl::graph_exec_submit(*stream, *seg.exec_graph);"), doc
    yield "enqueue target ignores the level", "G", sub1(
        code, r"\|\|\s*moe_hostpath_trace_level\(\)\s*<\s*2", ""), doc
    yield "ple stamp unguarded", "G", sub1(
        code, r"if \(traced\) \{\s*moe_token_stamps_note_ple\(([^;]*;)\s*\}", r"moe_token_stamps_note_ple(\1"), doc
    yield "d2h stamp unguarded", "G", sub1(
        code, r"if \(traced\) \{\s*moe_token_stamps_note_d2h\(([^;]*;)\s*\}", r"moe_token_stamps_note_d2h(\1"), doc
    yield "ple begin read untraced", "G", sub1(
        code, r"traced \? ggml_sycl::cpu_expert_pool_trace_now_ns\(\) : 0", "ggml_sycl::cpu_expert_pool_trace_now_ns()"), doc
    yield "staged count read untraced", "G", sub1(
        code, r"traced \? ggml_sycl::mem_copy_staged_wait_census_this_thread\(\)\.staged_d2h : 0",
        "ggml_sycl::mem_copy_staged_wait_census_this_thread().staged_d2h"), doc
    yield "a wait in a stamp helper", "N", sub1(
        code, r"static void moe_token_stamps_note_replay\(\) \{",
        "static void moe_token_stamps_note_replay() {\n    sycl::queue().wait();"), doc
    yield "an event query in a stamp helper", "N", sub1(
        code, r"static void moe_token_stamps_note_replay\(\) \{",
        "static void moe_token_stamps_note_replay() {\n"
        "    (void) g_ev.get_info<sycl::info::event::command_execution_status>();"), doc
    yield "an allocation in a stamp helper", "N", sub1(
        code, r"static void moe_token_stamps_note_d2h\(uint64_t staged_before\) \{",
        "static void moe_token_stamps_note_d2h(uint64_t staged_before) {\n    int * leak = new int[4];"), doc
    yield "a lock in a stamp helper", "N", sub1(
        code, r"static void moe_token_stamps_note_ple\(([^)]*)\) \{",
        r"static void moe_token_stamps_note_ple(\1) {\n    std::lock_guard<std::mutex> hold(g_x);"), doc
    yield "events become a vector", "B", sub1(
        code, r"moe_token_enqueue_event\s+events\s*\[\s*MOE_TOKEN_ENQUEUE_EVENTS\s*\]\s*;",
        "std::vector<moe_token_enqueue_event> events;"), doc
    yield "bound check removed", "B", sub1(
        code, r"if\s*\(\s*rec\.n_events\s*>=\s*MOE_TOKEN_ENQUEUE_EVENTS\s*\)\s*\{\s*\+\+rec\.events_dropped;"
        r"\s*return;\s*\}", ""), doc
    yield "ring not thread-local static", "B", sub1(
        code, r"static thread_local moe_token_stamp_ring g_moe_token_stamp_ring;",
        "static moe_token_stamp_ring g_moe_token_stamp_ring;"), doc
    yield "another clock", "K", sub1(
        code, r"static void moe_token_stamps_note_replay\(\) \{",
        "static void moe_token_stamps_note_replay() {\n    const auto bad = std::chrono::steady_clock::now();"), doc
    yield "the entry reads another clock", "K", sub1(
        code, r"const int64_t          now_ns = ggml_sycl::cpu_expert_pool_trace_now_ns\(\);",
        "const int64_t          now_ns = 0;"), doc
    extra_read = '\nstatic int g_x = std::getenv("' + ENV_NAME + '") != nullptr;\n'
    yield "a second reader of the switch", "V", code + extra_read, doc
    yield "a field dropped from the start line", "L", sub1(code, r"ple_reads=%llu ", ""), doc
    yield "the enqueue line outside its flag", "L", sub1(
        code, r"if \(enqueue_lines\) \{(\s*for \(int i = 0; i < rec\.n_events)", r"{\1"), doc
    yield "the enqueue lines at level 1", "L", sub1(
        code, r"(front\(\)\.t_submit_ns,\s*moe_hostpath_trace_level\(\) >= )2", r"\g<1>1"), doc
    yield "the line not relative to the first job", "L", sub1(
        code, r"pt\.job_stamps\.empty\(\) \? 0 : pt\.job_stamps\.front\(\)\.t_submit_ns,"
        r"(\s*moe_hostpath_trace_level\(\) >= 2\);)", r"0,\1"), doc
    yield "the record not marked by a readback", "L", sub1(
        code, r"g_moe_token_stamp_ring\.slot\[g_moe_token_stamp_ring\.cur\]\.readbacks_seen = true;", ""), doc
    yield "doc silent on the enqueue line", "D", code, doc.replace("[MOE-TOKEN-ENQUEUE]", "[x]")
    yield "doc silent on a field", "D", code, doc.replace("ple_misses=", "ple_m=")


def self_test() -> int:
    code, doc = load()
    try:
        gate(code, doc)
    except ContractError as exc:
        print(f"FAIL: the unmodified tree does not pass: {exc}")
        return 1
    bad = 0
    for name, pin, c, d in mutants(code, doc):
        try:
            gate(c, d)
        except ContractError as exc:
            if f"[pin {pin}]" not in str(exc):
                print(f"FAIL: mutant {name!r} was caught by another pin: {exc}")
                bad += 1
            continue
        print(f"FAIL: mutant not caught: {name}")
        bad += 1
    if bad:
        return 1
    print("test-sycl-token-start-stamps-source: self-test ok")
    return 0


def main() -> int:
    if "--self-test" in sys.argv:
        return self_test()
    code, doc = load()
    try:
        gate(code, doc)
    except ContractError as exc:
        print(exc)
        return 1
    print("test-sycl-token-start-stamps-source: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
