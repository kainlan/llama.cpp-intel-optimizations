#!/usr/bin/env python3
"""The census of mem_copy_submit's synchronous waits (llama.cpp-qwn5).

Companion of test-sycl-chain-same-queue-source.py (the ready-query census on the
same line) and test-sycl-moe-join-queue-source.py (joinq=, the line's last field).

THE COST.  mem_copy_submit (mem-ops.cpp) copies through a fresh pinned stage when
the host endpoint of a host<->device copy is not known to be host USM, and that
staged path holds the calling thread: the host->device copy waits for its deps on
the host before it fills the stage, waits for each chunk before it refills the stage
and, when the caller does not retain the copy, for the last one; the device->host
copy waits for every chunk before it copies it out.  The host->host copy waits for
its deps, and the cross-device copy waits on the host between its two halves.  On
the hybrid MoE decode route every copy to or from the pinned buffer pool took the
staged path (288 staging allocations per Qwen3.8 decode token, measured with
GGML_SYCL_STAGE_TRACE), and none of those waits was counted anywhere.

THE TRACE.  Under GGML_SYCL_MOE_IDS_COPY_TRACE each of those waits is counted by
kind and by the copy's caller (file, line), with its wall time, on the calling
thread, and so is each copy that took a staged branch, by direction.  The submitting
thread prints its census on the [MOE-HOSTPATH-WAITS] line beside the ready-query
census and resets it with the token's other counts.  Off, a wait costs one cached
flag test and reads no clock.

THE CONTRACT.  With comments blanked:
  K. (mem-ops.hpp, mem-ops.cpp) mem_copy_staged_wait_kind is exactly H2D_DEPS,
     H2D_CHUNK, H2D_FINAL, D2H, H2H_DEPS, CROSS, then COUNT, and each kind prints
     under its own name: h2d_deps, h2d_chunk, h2d_final, d2h, h2h_deps, cross.
  S. (mem-ops.hpp, mem-ops.cpp) the census holds per-kind counts and wall time, a
     16-slot site table, the overflow count and time, and the staged copies by
     direction; mem_copy_staged_wait_census_this_thread() returns one function-local
     thread_local census, and it and the formatter are declared in mem-ops.hpp.
  G. (mem-ops.cpp) the event and deps wrappers and the copy note test
     ggml_sycl_cpu_expert_trace_enabled() before anything else (the deps wrapper
     after its empty-list return) and, off, wait or return without reading a clock
     or touching the census; mem-ops.cpp reads no GGML_SYCL_MOE_IDS_COPY_TRACE of its
     own.
  R. (mem-ops.cpp staged_wait_record) a wait adds to its kind's count and time and to
     its site's row, matched on line, kind and file text; a free slot is claimed for
     a new site; a full table counts into overflow.
  W. (mem-ops.cpp mem_copy_submit, mem_copy_direct_submit) every synchronous wait in
     both goes through a wrapper under its own kind with the copy's caller's file and
     line: no bare wait_and_throw() or wait_deps() remains; the staged host->device
     branch waits deps (H2D_DEPS), the previous chunk inside `if (copied != 0)`
     (H2D_CHUNK) and, unretained, the last chunk (H2D_FINAL); the staged
     device->host branch waits each chunk before copying it out (D2H); the
     host->host branch waits deps before its memcpy (H2H_DEPS); the cross-device path
     makes three CROSS waits.
  C. (mem-ops.cpp) each staged branch notes its copy first, host->device as false and
     device->host as true, and nothing else notes one.
  F. (mem-ops.cpp mem_copy_staged_wait_census_format) copies print as
     d2h:N,h2d:N; waits as kind:N/USus for each kind that waited; sites as
     file:line:kind=N/USus for each used row, by its short file name, then
     overflow=N/USus when non-zero; waits and sites read none when empty.
  P. (ggml-sycl.cpp moe_hostpath_waits_readback) the [MOE-HOSTPATH-WAITS] line
     prints staged_copies=, staged_waits= and staged_sites= from the submitting
     thread's census, after ready_query_us= and before devsig= and joinq= (the
     device-signal count, llama.cpp-qwn5), and the census is reset with the token's
     other counts.
  D. (docs/backend/sycl-env-vars.md) the GGML_SYCL_MOE_IDS_COPY_TRACE row names the
     three fields, the six kinds, the 16-slot table and its overflow, and none.

NOT VACUOUS.  --self-test runs the gate on in-memory mutants of the real sources and
requires each to FAIL on its own pin, then requires the unmodified tree to pass.

RED on the tree before the change:
  FAIL [pin K]: mem-ops.hpp has no enum mem_copy_staged_wait_kind (llama.cpp-qwn5)
"""

from __future__ import annotations

import functools
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SYCL = ROOT / "ggml" / "src" / "ggml-sycl"
FILES = (
    ("ops_hpp", SYCL / "mem-ops.hpp"),
    ("ops_cpp", SYCL / "mem-ops.cpp"),
    ("backend", SYCL / "ggml-sycl.cpp"),
)
DOC = ROOT / "docs" / "backend" / "sycl-env-vars.md"
TAG = "(llama.cpp-qwn5)"
KIND = "MEM_COPY_STAGED_WAIT_KIND_"
KINDS = (("H2D_DEPS", "h2d_deps"), ("H2D_CHUNK", "h2d_chunk"), ("H2D_FINAL", "h2d_final"), ("D2H", "d2h"),
         ("H2H_DEPS", "h2h_deps"), ("CROSS", "cross"))
TRACE_GUARD = "ggml_sycl_cpu_expert_trace_enabled()"

SUBMIT_SIG = r"static sycl::event mem_copy_submit\(const mem_handle &\s+dst,\s+size_t\s+dst_offset,[^)]*\)\s*\{"
DIRECT_SIG = r"static sycl::event mem_copy_direct_submit\([^)]*\)\s*\{"
EVENT_SIG = r"static void staged_wait_event\([^)]*\)\s*\{"
DEPS_SIG = r"static void staged_wait_deps\([^)]*\)\s*\{"
NOTE_SIG = r"static void staged_copy_note\(bool device_to_host\)\s*\{"
RECORD_SIG = r"static void staged_wait_record\([^)]*\)\s*\{"
NAME_SIG = r"static const char \* mem_copy_staged_wait_kind_name\(int kind\)\s*\{"
FORMAT_SIG = (r"void mem_copy_staged_wait_census_format\(const mem_copy_staged_wait_census & census, "
              r"mem_copy_staged_wait_text & out\)\s*\{")
THIS_THREAD_SIG = r"mem_copy_staged_wait_census & mem_copy_staged_wait_census_this_thread\(\)\s*\{"
READBACK_SIG = r"static void moe_hostpath_waits_readback\(int layer, bool async = false\)\s*\{"


class ContractError(Exception):
    pass


def fail(pin: str, msg: str) -> None:
    raise ContractError(f"FAIL [pin {pin}]: {msg} {TAG}")


_TOKEN = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n]){1,4}\'', re.S)


@functools.lru_cache(maxsize=256)
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


def strip_strings(src: str) -> str:
    """Blank the contents of string literals so braces in them do not count."""
    return re.sub(r'"(?:\\.|[^"\\\n])*"', lambda m: '"' + " " * (len(m.group(0)) - 2) + '"', src)


def span_of(code: str, signature: str, pin: str) -> tuple[int, int, int]:
    """(start of the signature, start of the body, end of the body) of the first definition matching `signature`,
    which ends at its opening brace."""
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
                return m.start(), i + 1, j
    fail(pin, f"unbalanced body for {signature!r}")
    return 0, 0, 0


def body_of(code: str, signature: str, pin: str) -> str:
    """The body of the first definition matching `signature` (ending at its opening brace)."""
    _, b, e = span_of(code, signature, pin)
    return code[b:e]


def flat(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def flat_body(code: str, signature: str, pin: str) -> str:
    return flat(body_of(code, signature, pin))


# Comments in the code survive blank_comments as runs of spaces, so `/*device_to_host=*/true` is matched as the
# bare argument.
NOTE_H2D = "staged_copy_note( false);"
NOTE_D2H = "staged_copy_note( true);"


def check_kinds(s: dict) -> None:
    hpp = blank_comments(s["ops_hpp"])
    m = re.search(r"enum mem_copy_staged_wait_kind\s*\{([^}]*)\}", hpp)
    if not m:
        fail("K", "mem-ops.hpp has no enum mem_copy_staged_wait_kind")
    got = re.findall(r"\b" + KIND + r"(\w+)", m.group(1))
    want = [k for k, _ in KINDS] + ["COUNT"]
    if got != want:
        fail("K", f"the staged wait kinds are {got}, not {want}")
    if not re.search(KIND + r"H2D_DEPS\s*=\s*0\s*,", m.group(1)):
        fail("K", "the first kind is not 0, so the census arrays do not index from it")
    body = flat_body(blank_comments(s["ops_cpp"]), NAME_SIG, "K")
    names = dict(re.findall(r"case " + KIND + r"(\w+): return \"([a-z0-9_]+)\";", body))
    if names != dict(KINDS):
        fail("K", f"the kinds' names are {names}, not {dict(KINDS)}")


def check_storage(s: dict) -> None:
    hpp = blank_comments(s["ops_hpp"])
    fields = flat(body_of(hpp, r"struct mem_copy_staged_wait_census\s*\{", "S"))
    for want in ("uint64_t count[" + KIND + "COUNT] = {};",
                 "double us[" + KIND + "COUNT] = {};",
                 "mem_copy_staged_wait_site sites[k_mem_copy_staged_wait_site_slots] = {};",
                 "uint64_t overflow_count = 0;",
                 "double overflow_us = 0.0;",
                 "uint64_t staged_d2h = 0;",
                 "uint64_t staged_h2d = 0;"):
        if want not in fields:
            fail("S", f"the census does not hold {want!r}")
    if not re.search(r"\bconstexpr size_t k_mem_copy_staged_wait_site_slots = 16;", hpp):
        fail("S", "the site table is not 16 slots")
    m = re.search(r"struct mem_copy_staged_wait_site\s*\{([^}]*)\}", hpp)
    site = flat(m.group(1)) if m else ""
    for want in ("const char * file = nullptr;", "int line = 0;", "uint64_t count = 0;", "double us = 0.0;"):
        if want not in site:
            fail("S", f"a site row does not hold {want!r}")
    for decl in (r"mem_copy_staged_wait_census & mem_copy_staged_wait_census_this_thread\(\);",
                 r"void mem_copy_staged_wait_census_format\(const mem_copy_staged_wait_census & census,\s*"
                 r"mem_copy_staged_wait_text & out\);"):
        if not re.search(decl, hpp):
            fail("S", f"mem-ops.hpp does not declare {decl!r}")
    body = flat_body(blank_comments(s["ops_cpp"]), THIS_THREAD_SIG, "S")
    if body != "static thread_local mem_copy_staged_wait_census census; return census;":
        fail("S", "mem_copy_staged_wait_census_this_thread does not return one function-local thread_local census")


def check_guard(s: dict) -> None:
    code = blank_comments(s["ops_cpp"])
    body = flat_body(code, EVENT_SIG, "G")
    if not body.startswith("if (!" + TRACE_GUARD + ") { event.wait_and_throw(); return; } "
                           "const auto t0 = std::chrono::steady_clock::now(); event.wait_and_throw(); "
                           "staged_wait_record(kind, file, line, staged_wait_us_since(t0));"):
        fail("G", "staged_wait_event does not test the trace first and wait without a clock when it is off")
    body = flat_body(code, DEPS_SIG, "G")
    if not body.startswith("if (deps.empty()) { return; } if (!" + TRACE_GUARD + ") { wait_deps(deps); return; } "
                           "const auto t0 = std::chrono::steady_clock::now(); wait_deps(deps); "
                           "staged_wait_record(kind, file, line, staged_wait_us_since(t0));"):
        fail("G", "staged_wait_deps does not skip an empty list, then test the trace and wait without a clock off")
    body = flat_body(code, NOTE_SIG, "G")
    if not body.startswith("if (!" + TRACE_GUARD + ") { return; }"):
        fail("G", "staged_copy_note does not test the trace before it counts")
    if "GGML_SYCL_MOE_IDS_COPY_TRACE" in code:
        fail("G", "mem-ops.cpp reads GGML_SYCL_MOE_IDS_COPY_TRACE itself instead of the trace's one reader")
    # The census is written only by the guarded wrappers, by the record they call and by its own accessor.
    allowed = []
    for sig in (EVENT_SIG, DEPS_SIG, NOTE_SIG, RECORD_SIG, THIS_THREAD_SIG):
        a, _, b = span_of(code, sig, "G")
        allowed.append((a, b))
    for fn in ("staged_wait_record(", "mem_copy_staged_wait_census_this_thread()"):
        for m in re.finditer(re.escape(fn), code):
            if not any(a <= m.start() <= b for a, b in allowed):
                fail("G", f"{fn} is reached outside the trace-guarded wrappers")


def check_record(s: dict) -> None:
    body = flat_body(blank_comments(s["ops_cpp"]), RECORD_SIG, "R")
    want = ("mem_copy_staged_wait_census & c = mem_copy_staged_wait_census_this_thread(); c.count[kind] += 1; "
            "c.us[kind] += us; file = file != nullptr ? file : \"?\"; for (mem_copy_staged_wait_site & s : c.sites) "
            "{ if (s.file == nullptr) { s.file = file; s.line = line; s.kind = kind; } if (s.line == line && s.kind "
            "== kind && (s.file == file || std::strcmp(s.file, file) == 0)) { s.count += 1; s.us += us; return; } } "
            "c.overflow_count += 1; c.overflow_us += us;")
    if body != want:
        fail("R", "staged_wait_record does not add to the kind and to the site matched on line, kind and file text, "
                  "claiming a free slot and counting overflow")


def check_waits(s: dict) -> None:
    code = blank_comments(s["ops_cpp"])
    submit = flat_body(code, SUBMIT_SIG, "W")
    direct = flat_body(code, DIRECT_SIG, "W")
    for name, body in (("mem_copy_submit", submit), ("mem_copy_direct_submit", direct)):
        if re.search(r"\.wait_and_throw\(\)|(?<![\w])wait_deps\(|\.wait\(\)", body):
            fail("W", f"{name} still waits outside the census wrappers")
        for call in re.findall(r"staged_wait_(?:event|deps)\(([^;]*)\);", body):
            if not call.endswith(", file, line"):
                fail("W", f"a census wait in {name} does not name the copy's caller: {call!r}")
    seqs = (
        ("the staged host->device branch",
         ["staged_wait_deps(deps, " + KIND + "H2D_DEPS, file, line); resolved_ptr stage_ptr = stage.resolve();",
          "if (copied != 0) { staged_wait_event(event, " + KIND + "H2D_CHUNK, file, line); }",
          "std::memcpy(stage_ptr.ptr,",
          "} else { g_stage_waited.fetch_add(1, std::memory_order_relaxed); staged_wait_event(event, " + KIND +
          "H2D_FINAL, file, line); } return event;"]),
        ("the staged device->host branch",
         ["staged_wait_event(event, " + KIND + "D2H, file, line); "
          "std::memcpy(static_cast<char *>(d.ptr) + dst_offset + copied, stage_ptr.ptr, cur);"]),
        ("the cross-device path",
         ["} else { staged_wait_event(h2d, " + KIND + "CROSS, file, line); } return h2d; } "
          "staged_wait_event(d2h, " + KIND + "CROSS, file, line);",
          "} else { staged_wait_event(h2d, " + KIND + "CROSS, file, line); } return h2d; }"]),
    )
    for what, parts in seqs:
        at = 0
        for part in parts:
            i = submit.find(part, at)
            if i < 0:
                fail("W", f"{what} does not wait as pinned at {part!r}")
            at = i + len(part)
    counts = {k: len(re.findall(KIND + k + r"\b", submit)) for k, _ in KINDS}
    want = {"H2D_DEPS": 1, "H2D_CHUNK": 1, "H2D_FINAL": 1, "D2H": 1, "H2H_DEPS": 0, "CROSS": 3}
    if counts != want:
        fail("W", f"mem_copy_submit's waits by kind are {counts}, not {want}")
    if "staged_wait_deps(deps, " + KIND + "H2H_DEPS, file, line); std::memcpy(dst_ptr, src_ptr, size);" not in direct:
        fail("W", "the host->host copy does not wait its deps as H2H_DEPS before its memcpy")
    if len(re.findall(KIND + r"\w+", direct)) != 1:
        fail("W", "mem_copy_direct_submit makes a census wait other than its host->host one")


def check_copies(s: dict) -> None:
    code = flat(blank_comments(s["ops_cpp"]))
    notes = re.findall(r"staged_copy_note\(\s*(\w+)\s*\);", code)
    if sorted(notes) != ["false", "true"]:
        fail("C", f"the staged copies are noted {notes}, not once per direction")
    submit = flat_body(blank_comments(s["ops_cpp"]), SUBMIT_SIG, "C")
    h2d = re.search(r"if \(d\.on_device && !s\.on_device && ([^{]*)\) \{ " + re.escape(NOTE_H2D), submit)
    d2h = re.search(r"if \(!d\.on_device && s\.on_device && ([^{]*)\) \{ " + re.escape(NOTE_D2H), submit)
    if not h2d or not d2h:
        fail("C", "a staged branch does not note its copy, by its own direction, first")
    body = flat_body(blank_comments(s["ops_cpp"]), NOTE_SIG, "C")
    if not body.endswith("if (device_to_host) { c.staged_d2h += 1; } else { c.staged_h2d += 1; }"):
        fail("C", "staged_copy_note does not count the direction it is given")


def check_format(s: dict) -> None:
    body = flat_body(blank_comments(s["ops_cpp"]), FORMAT_SIG, "F")
    pins = (
        ("std::snprintf(out.copies, sizeof(out.copies), \"d2h:%\" PRIu64 \",h2d:%\" PRIu64, census.staged_d2h, "
         "census.staged_h2d);", "copies as d2h:N,h2d:N"),
        ("if (census.count[kind] == 0 || at >= (int) sizeof(out.waits)) { continue; }",
         "the kinds that waited only"),
        ("\"%s%s:%\" PRIu64 \"/%.0fus\", at > 0 ? \",\" : \"\", mem_copy_staged_wait_kind_name(kind), "
         "census.count[kind], census.us[kind]);", "each kind as kind:N/USus"),
        ("if (at == 0) { std::snprintf(out.waits, sizeof(out.waits), \"none\"); }", "waits as none when empty"),
        ("if (s.file == nullptr || s.count == 0 || at >= (int) sizeof(out.sites)) { continue; }",
         "the used rows only"),
        ("\"%s%s:%d:%s=%\" PRIu64 \"/%.0fus\", at > 0 ? \",\" : \"\", mem_copy_site_short_file(s.file), s.line, "
         "mem_copy_staged_wait_kind_name(s.kind), s.count, s.us);", "each site as file:line:kind=N/USus"),
        ("if (census.overflow_count != 0 && at < (int) sizeof(out.sites)) {", "the overflow when non-zero"),
        ("\"%soverflow=%\" PRIu64 \"/%.0fus\", at > 0 ? \",\" : \"\", census.overflow_count, census.overflow_us);",
         "the overflow as overflow=N/USus"),
        ("if (at == 0) { std::snprintf(out.sites, sizeof(out.sites), \"none\"); }", "sites as none when empty"),
    )
    for want, what in pins:
        if want not in body:
            fail("F", f"the formatter does not print {what}")


def check_print(s: dict) -> None:
    body = flat_body(blank_comments(s["backend"]), READBACK_SIG, "P")
    m = re.search(r'GGML_LOG_WARN\( ((?:"[^"]*" ?)+)', body)
    fmt = "".join(re.findall(r'"([^"]*)"', m.group(1))) if m else ""
    if not fmt.startswith("[MOE-HOSTPATH-WAITS]") or \
            "ready_query_us=%.0f staged_copies=%s staged_waits=%s staged_sites=%s devsig=%llu joinq=%d" not in fmt:
        fail("P", "the [MOE-HOSTPATH-WAITS] line does not print staged_copies=, staged_waits= and staged_sites= "
                  "between ready_query_us= and devsig=")
    for want, what in (
            ("ggml_sycl::mem_copy_staged_wait_census & sw = ggml_sycl::mem_copy_staged_wait_census_this_thread(); "
             "ggml_sycl::mem_copy_staged_wait_text st; ggml_sycl::mem_copy_staged_wait_census_format(sw, st);",
             "formats the submitting thread's census"),
            ("other_sites, rq.pending_us, st.copies, st.waits, st.sites, (unsigned long long) w.joins_device_signal, "
             "joinq);", "prints the three fields in order"),
            ("sw = ggml_sycl::mem_copy_staged_wait_census{};", "resets the census with the token's counts")):
        if want not in body:
            fail("P", f"the [MOE-HOSTPATH-WAITS] line does not {what}")
    reset = body.find("sw = ggml_sycl::mem_copy_staged_wait_census{};")
    if reset < body.find("GGML_LOG_WARN( \"[MOE-HOSTPATH-WAITS]") or reset > body.find("++w.token;"):
        fail("P", "the census is not reset after the line prints and with the token's other counts")


def check_doc(doc: str) -> None:
    rows = [ln for ln in (doc or "").splitlines() if ln.startswith("| `GGML_SYCL_MOE_IDS_COPY_TRACE")]
    if len(rows) != 1:
        fail("D", f"expected one GGML_SYCL_MOE_IDS_COPY_TRACE catalog row, found {len(rows)}")
    row = rows[0]
    needles = [("`staged_copies=d2h:N,h2d:N`", "the staged copies field"), ("`staged_waits=`", "the waits field"),
               ("`staged_sites=`", "the sites field"), ("16-slot", "the table's size"),
               ("`overflow=N/USus`", "the overflow"), ("`none`", "the empty value"),
               ("llama.cpp-qwn5", "the ticket")]
    needles += [(f"`{name}`", f"the {name} kind") for _, name in KINDS]
    for needle, what in needles:
        if needle not in row:
            fail("D", f"the GGML_SYCL_MOE_IDS_COPY_TRACE row does not state {what} ({needle!r})")


def check(s: dict) -> None:
    check_kinds(s)
    check_storage(s)
    check_guard(s)
    check_record(s)
    check_waits(s)
    check_copies(s)
    check_format(s)
    check_print(s)
    check_doc(s["doc"])


def read_sources(root: Path) -> dict:
    out = {}
    for key, path in FILES + (("doc", DOC),):
        p = root / path.relative_to(ROOT)
        out[key] = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
    return out


def sub(pattern: str, repl: str, flags: int = 0, count: int = 1):
    def mutate(text: str) -> str:
        return re.sub(pattern, repl, text, count=count, flags=flags)
    return mutate


def self_test(s: dict) -> int:
    K = KIND
    mutants = [
        ("K", "ops_hpp", "a kind dropped", sub(r"\n\s*" + K + r"CROSS,", "")),
        ("K", "ops_hpp", "two kinds reordered",
         sub(r"(" + K + r"H2D_FINAL,)(\s*)(" + K + r"D2H,)", r"\3\2\1")),
        ("K", "ops_cpp", "a kind loses its name", sub(r'return "d2h";', 'return "d2h_chunk";')),
        ("K", "ops_cpp", "two kinds share a name", sub(r'return "h2d_final";', 'return "h2d_chunk";')),
        ("S", "ops_cpp", "census not per thread",
         sub(r"static thread_local mem_copy_staged_wait_census census;", "static mem_copy_staged_wait_census census;")),
        ("S", "ops_hpp", "site table resized", sub(r"k_mem_copy_staged_wait_site_slots = 16;",
                                                   "k_mem_copy_staged_wait_site_slots = 8;")),
        ("S", "ops_hpp", "staged h2d count dropped", sub(r"\n\s*uint64_t\s+staged_h2d\s*=\s*0;", "")),
        ("S", "ops_hpp", "census accessor undeclared",
         sub(r"mem_copy_staged_wait_census & mem_copy_staged_wait_census_this_thread\(\);", "")),
        ("G", "ops_cpp", "event wrapper reads the clock with the trace off",
         sub(r"(static void staged_wait_event\([^)]*\) \{\n)(\s*)(if \(!ggml_sycl_cpu_expert_trace_enabled\(\)\))",
             r"\1\2const auto t_off = std::chrono::steady_clock::now();\n\2(void) t_off;\n\2\3")),
        ("G", "ops_cpp", "deps wrapper waits an empty list under the trace",
         sub(r"if \(deps\.empty\(\)\) \{\n\s*return;\n\s*\}\n", "")),
        ("G", "ops_cpp", "copy note counts with the trace off",
         sub(r"(static void staged_copy_note\(bool device_to_host\) \{\n)\s*if \(!ggml_sycl_cpu_expert_trace_enabled"
             r"\(\)\) \{\n\s*return;\n\s*\}\n", r"\1")),
        ("G", "ops_cpp", "a second reader of the trace knob",
         sub(r"(static bool mem_copy_trace_enabled\(\) \{)",
             r'static bool staged_trace_local() { return std::getenv("GGML_SYCL_MOE_IDS_COPY_TRACE") != nullptr; }\n\n\1')),
        ("G", "ops_cpp", "census touched outside the guarded wrappers",
         sub(r"(sycl::event mem_copy_ptr_async\([^)]*\) \{)",
             r"\1\n    mem_copy_staged_wait_census_this_thread().staged_h2d += 1;")),
        ("R", "ops_cpp", "kind totals not counted", sub(r"c\.count\[kind\] \+= 1;", ";")),
        ("R", "ops_cpp", "site matched without its kind", sub(r"s\.line == line && s\.kind == kind &&",
                                                              "s.line == line &&")),
        ("R", "ops_cpp", "site matched on the pointer only",
         sub(r"\(s\.file == file \|\| std::strcmp\(s\.file, file\) == 0\)", "s.file == file")),
        ("R", "ops_cpp", "overflow not counted", sub(r"c\.overflow_count \+= 1;", ";")),
        ("R", "ops_cpp", "site time not added", sub(r"s\.us \+= us;", ";")),
        ("W", "ops_cpp", "host->device deps wait bare",
         sub(r"staged_wait_deps\(deps, " + K + r"H2D_DEPS, file, line\);", "wait_deps(deps);")),
        ("W", "ops_cpp", "chunk wait bare",
         sub(r"staged_wait_event\(event, " + K + r"H2D_CHUNK, file, line\);", "event.wait_and_throw();")),
        ("W", "ops_cpp", "final wait bare",
         sub(r"staged_wait_event\(event, " + K + r"H2D_FINAL, file, line\);", "event.wait_and_throw();")),
        ("W", "ops_cpp", "device->host wait bare",
         sub(r"staged_wait_event\(event, " + K + r"D2H, file, line\);", "event.wait_and_throw();")),
        ("W", "ops_cpp", "host->host wait bare",
         sub(r"staged_wait_deps\(deps, " + K + r"H2H_DEPS, file, line\);", "wait_deps(deps);")),
        ("W", "ops_cpp", "cross wait bare",
         sub(r"staged_wait_event\(d2h, " + K + r"CROSS, file, line\);", "d2h.wait_and_throw();")),
        ("W", "ops_cpp", "chunk and final kinds swapped",
         sub(r"(staged_wait_event\(event, " + K + r")H2D_CHUNK(, file, line\);)((?:.|\n)*?)(staged_wait_event\(event, " +
             K + r")H2D_FINAL", r"\1H2D_FINAL\2\3\4H2D_CHUNK")),
        ("W", "ops_cpp", "a wait names mem-ops.cpp instead of the caller",
         sub(r"staged_wait_event\(event, " + K + r"D2H, file, line\);",
             "staged_wait_event(event, " + K + "D2H, __FILE__, __LINE__);")),
        ("W", "ops_cpp", "a cross wait counted as device->host",
         sub(r"staged_wait_event\(d2h, " + K + r"CROSS, file, line\);",
             "staged_wait_event(d2h, " + K + "D2H, file, line);")),
        ("C", "ops_cpp", "staged host->device copy not noted", sub(r"staged_copy_note\(/\*device_to_host=\*/false\);", ";")),
        ("C", "ops_cpp", "directions swapped",
         sub(r"c\.staged_d2h \+= 1;\n(\s*)\} else \{\n(\s*)c\.staged_h2d \+= 1;",
             r"c.staged_h2d += 1;\n\1} else {\n\2c.staged_d2h += 1;")),
        ("F", "ops_cpp", "kinds that never waited printed",
         sub(r"census\.count\[kind\] == 0 \|\| at >= \(int\) sizeof\(out\.waits\)", "at >= (int) sizeof(out.waits)")),
        ("F", "ops_cpp", "no none for waits",
         sub(r"std::snprintf\(out\.waits, sizeof\(out\.waits\), \"none\"\);", 'out.waits[0] = 0;')),
        ("F", "ops_cpp", "site printed by its full path",
         sub(r"mem_copy_site_short_file\(s\.file\), s\.line", "s.file, s.line")),
        ("F", "ops_cpp", "overflow not printed", sub(r"if \(census\.overflow_count != 0 && ", "if (false && ")),
        ("P", "backend", "line drops staged_sites", sub(r"staged_sites=%s ", "")),
        ("P", "backend", "census not reset", sub(r"sw(\s*)= ggml_sycl::mem_copy_staged_wait_census\{\};", r"(void) sw;")),
        ("P", "backend", "line formats a fresh census",
         sub(r"ggml_sycl::mem_copy_staged_wait_census_format\(sw, st\);",
             "ggml_sycl::mem_copy_staged_wait_census_format(ggml_sycl::mem_copy_staged_wait_census{}, st);")),
        ("P", "backend", "fields printed out of order", sub(r"st\.copies, st\.waits,", "st.waits, st.copies,")),
        ("D", "doc", "row loses staged_sites",
         sub(r"(\| `GGML_SYCL_MOE_IDS_COPY_TRACE[^\n]*?)`staged_sites=`", r"\1`staged_site=`")),
        ("D", "doc", "row loses a kind", sub(r"(\| `GGML_SYCL_MOE_IDS_COPY_TRACE[^\n]*?)`h2h_deps`", r"\1`h2h`")),
        ("D", "doc", "row loses the overflow",
         sub(r"(\| `GGML_SYCL_MOE_IDS_COPY_TRACE[^\n]*?)`overflow=N/USus`", r"\1`overflow`")),
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
    print("PASS: mem_copy_submit's synchronous waits are counted by kind and by caller under the MoE trace and "
          f"printed on the [MOE-HOSTPATH-WAITS] line {TAG}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
