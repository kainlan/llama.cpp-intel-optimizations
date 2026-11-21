#!/usr/bin/env python3
"""Source contract: the decode-boundary expert swap driver runs once per boundary, never waits and never holds a lock
across its policy.

Host-only: reads ggml-sycl.cpp, moe-swap-driver.hpp, moe-stream-plan.hpp and unified-cache.cpp, builds nothing, loads
no model and touches no device.

The driver (GGML_SYCL_MOE_SWAP=1) asks the swap policy at each synchronize boundary which resident triplet leaves and
which host triplet enters, and sends the answer through the named move. tests/test-sycl-moe-swap-driver.cpp shows the
decisions and the bookkeeping are right; the loan test's device case shows an evict into a full host zone is BUSY. What
a unit test cannot show is where the driver runs and what it does not do there. This gate holds that by text:

  D1  one call site: ggml_backend_sycl_synchronize() calls the driver once, after the refill, and nothing else does;
  D2  the policy step runs once, inside the driver;
  D3  no host wait and no event status query in the driver, its helpers (every ggml_sycl_moe_swap_* function), the
      driver header or the plan-view builder: no wait() or wait_and_throw(), no retained-handle drain, no event
      get_info or command status, no synchronize, no sleep, no condition variable;
  D4  no lock is held across the policy step: every lock the driver takes is scoped to a block that ends before the
      step, or opens after it, and none is taken by a bare lock() call;
  D5  the driver is gated before the step: the switch, the multithreaded-graph latch, the compute flag, the graphs in
      flight, and the claim of the device's instance, in that order;
  D6  the switch is read once: GGML_SYCL_MOE_SWAP is spelled once, as the header's MOE_SWAP_ENV; its one getenv is in
      the accessor's static initializer; the settings table names it once and its row is overwritten from the
      accessor after the parse, so the non-default report says what is in effect;
  D7  each summary line reaches the log at WARN: every summary call is an argument of GGML_LOG_WARN, and both exist;
      the second line names the accepted transactions, evicts and admits apart, beside the decode graphs, and the
      step counts each direction's accepted transaction once (each one can cost a context what it recorded);
  D8  the driver header is plain host code: it includes the policy header and a short list of standard headers, and
      keeps no state outside the caller's struct (no static, thread_local, extern, mutable or volatile, no inline
      variable, no atomic or lock), and reaches no device, cache, log or environment;
  D9  the plan-view builder keeps no state and takes no lock, logs nothing and reads no environment;
  D10 a named evict's host copy never grows the pinned host zone: the to-host branch mints an evict's copy through
      ggml_sycl_host_expert_alloc_no_growth, while a loan's copy keeps the growing mint. There is one host mint:
      ggml_sycl_host_expert_alloc takes grow_zone, and with it false its request forbids host-zone growth and keeps
      the miss out of the error log; the no-growth name only calls it with false;
  D11 the route counters are copied into the driver's own snapshot arrays: the step calls the counts callback once,
      with d.decode.data() and d.prompt.data(), the arrays the policy's block inputs point at;
  D12 a final refusal reaches the log at WARN once per reason and direction: the caller's transaction callback calls
      ggml_sycl_moe_swap_warn_refusal for every move; the helper passes over every outcome but REFUSED, keeps the
      reasons it logged in the instance's per-direction sets (touched nowhere else), and logs only a reason it had
      not logged;
  D13 the refusal that latches a block has one spelling: the plan header defines MOE_EXPERT_MOVE_SHARED_STORAGE as
      the only "shared storage" literal in the four sources; the cache's loan assigns it by name; the caller marks a
      move as latching only through ggml_sycl_moe_swap_refusal_latches_block, which compares the reason to that
      constant for a refused evict; and the step latches a block only when the move says so;
  D14 a device where no driver ran still reports: ggml_sycl_moe_swap_report, with no driver, takes both counters
      kept apart and prints them in one WARN line when either is nonzero;
  D15 the evict batch's room and the evict mint's cap are one figure: the driver's room callback returns
      unified_cache::host_zone_no_growth_room for the WEIGHT zone, the mint refuses a no-growth request past that
      same accessor, and the accessor is the zone's capacity capped at its reserved figure, less what it holds,
      never below zero.

Checks run on comment-stripped text, string literals kept. NOT VACUOUS: --self-test runs the gate on in-memory mutants
of the real sources. Each mutant names the pin that must catch it, and is reported if it survives, if a different pin
catches it, or if its anchor no longer matches exactly once (NOT APPLIED). The self-test also proves that last report
fires. ctest registers it with --self-test.

WHAT THIS DOES NOT PROVE. It reads text. A wait reached through a function the driver calls by another name is not
seen (the named move is the admit and loan pins' business, and it waits on nothing); a lock taken inside a callee is
not seen. That the decisions are right is the unit test's job; that an evict into a full zone answers BUSY on a device
is the loan test's.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "sycl": ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp",
    "driver": ROOT / "ggml/src/ggml-sycl/moe-swap-driver.hpp",
    "plan": ROOT / "ggml/src/ggml-sycl/moe-stream-plan.hpp",
    "cache": ROOT / "ggml/src/ggml-sycl/unified-cache.cpp",
}

SYNC_SIG = "static void ggml_backend_sycl_synchronize(ggml_backend_t backend) {"
CALLER_SIG = "static void ggml_sycl_run_expert_swap_at_boundary("
CALLER_NAME = "ggml_sycl_run_expert_swap_at_boundary("
CALLER_CALL = "ggml_sycl_run_expert_swap_at_boundary(backend);"
REFILL_CALL = "ggml_sycl_run_expert_refill_at_boundary(backend);"
STEP_CALL = "moe_swap_driver_step("
BUILDER_SIG = "static inline void moe_swap_plan_view_build("
ACCESSOR_SIG = "static bool ggml_sycl_moe_swap_enabled("
PREP_SIG = "static const char * ggml_sycl_expert_prepare_triplet("
NO_GROWTH_SIG = "static ggml_sycl::mem_handle ggml_sycl_host_expert_alloc_no_growth("
MINT_SIG = "static ggml_sycl::mem_handle ggml_sycl_host_expert_alloc(int device, size_t size, bool grow_zone) {"
STEP_SIG = "moe_swap_driver_status moe_swap_driver_step("
WARN_SIG = "static void ggml_sycl_moe_swap_warn_refusal("
LATCH_SIG = "static bool ggml_sycl_moe_swap_refusal_latches_block("
REPORT_SIG = "static void ggml_sycl_moe_swap_report("
ROOM_SIG = "static size_t ggml_sycl_moe_swap_host_room("
LOAN_SIG = "bool unified_cache::loan_expert_triplet("
NO_GROWTH_ROOM_SIG = "size_t unified_cache::host_zone_no_growth_room("
HELPER_DEF = re.compile(r"^static\s+[\w:<>,\s\*&]*?\b(ggml_sycl_moe_swap_\w+)\s*\(", re.MULTILINE)

GATES = ("ggml_sycl_moe_swap_enabled()", "g_sycl_graph_multithreaded", "unified_cache_is_graph_compute_active()",
         "ggml_sycl_graph_inflight_count()", "compare_exchange_strong(")
TABLE_ROW = re.compile(r"\{\s*ggml_sycl::MOE_SWAP_ENV\s*,\s*&g_ggml_sycl_moe_swap\s*,\s*0\s*\}")
ROW_OVERWRITE = re.compile(r"g_ggml_sycl_moe_swap\s*=\s*ggml_sycl_moe_swap_enabled\(\)\s*\?\s*1\s*:\s*0\s*;")
PARSE_LOOP = "*setting.dest = get_sycl_env(setting.name, setting.default_value);"
ENV_SPELLING = re.compile(r"MOE_SWAP_ENV\[\]\s*=\s*\"GGML_SYCL_MOE_SWAP\"\s*;")
SUMMARIES = ("moe_swap_driver_summary_moves(", "moe_swap_driver_summary_faults(")
FAULTS_SIG = "inline std::string moe_swap_driver_summary_faults("
FAULTS_FIELDS = ("accepted transactions %llu (evict %llu, admit %llu) over %llu decode graphs",
                 "(ull) (st.transactions_evict + st.transactions_admit), (ull) st.transactions_evict,",
                 "(ull) st.transactions_admit, (ull) st.decode_graphs,")
TXN_COUNTS = ("++st.transactions_evict;", "++st.transactions_admit;")
EVICT_MINT = re.compile(r"if\s*\(\s*out\.kind\s*==\s*ggml_sycl_expert_move_kind::EVICT\s*\)\s*\{\s*"
                        r"storage\s*=\s*ggml_sycl_host_expert_alloc_no_growth\(\s*device\s*,\s*target\.logical_bytes\s*\)"
                        r"\s*;\s*\}\s*else\s*\{\s*storage\s*=\s*ggml_sycl_host_expert_alloc\(\s*device\s*,\s*"
                        r"target\.logical_bytes\s*\)\s*;\s*\}")
MINT_FIELDS = ("req.intent.role = ggml_sycl::alloc_role::WEIGHT;", "req.intent.constraints.must_host_pinned = true;",
               "req.intent.constraints.use_pinned_pool = true;",
               "req.intent.constraints.forbid_host_zone_growth = !grow_zone;", "req.suppress_failure_log = !grow_zone;",
               "unified_allocate_owner(req)")
NO_GROWTH_CALL = "return ggml_sycl_host_expert_alloc(device, size, false);"
COUNTS_CALL = "counts(d.decode.data(), d.prompt.data());"
WARN_CALL = "ggml_sycl_moe_swap_warn_refusal(inst, device, m);"
WARN_FIELDS = ("if (m.outcome != ggml_sycl::MOE_SWAP_MOVE_REFUSED) { return; }",
               "evict ? inst.refusals_warned_evict : inst.refusals_warned_admit;",
               "if (!warned.insert(reason).second) { return; }",
               'GGML_LOG_WARN("[MOE-SWAP] device %d: %s refused: %s (logged once per reason)\\n", device, '
               'evict ? "evict" : "admit", reason);')
SHARED_LITERAL = '"shared storage"'
SHARED_DEF = 'constexpr const char MOE_EXPERT_MOVE_SHARED_STORAGE[] = "shared storage";'
SHARED_ASSIGN = "refusal = MOE_EXPERT_MOVE_SHARED_STORAGE;"
LATCH_FIELDS = ("m.kind == ggml_sycl::MOE_SWAP_MOVE_EVICT", "m.outcome == ggml_sycl::MOE_SWAP_MOVE_REFUSED",
                "std::strcmp(m.reason, ggml_sycl::MOE_EXPERT_MOVE_SHARED_STORAGE) == 0")
LATCH_SET = "m.latches_block = ggml_sycl_moe_swap_refusal_latches_block(m);"
STEP_LATCH = "if (m.latches_block) { d.latched[static_cast<size_t>(q.block)] = 1; }"
NO_DRIVER_FIELDS = ("const uint64_t off = inst.off_multithreaded.exchange(0, std::memory_order_relaxed);",
                    "const uint64_t skipped = inst.skipped.exchange(0, std::memory_order_relaxed);",
                    "if (off != 0 || skipped != 0) { GGML_LOG_WARN(\"[MOE-SWAP] device %d: no driver ran: ")
ROOM_RETURN = "return cache->host_zone_no_growth_room(ggml_sycl::host_zone_id::WEIGHT);"
MINT_CAP = ("if (req.intent.constraints.forbid_host_zone_growth && alloc_size > ucache->host_zone_no_growth_room(zone)) "
            "{ return nullptr; }")
NO_GROWTH_ROOM_FIELDS = ("const size_t limit = reserved > 0 ? std::min(capacity, reserved) : capacity;",
                         "return limit > used ? limit - used : 0;")

WAITS = re.compile(
    r"(?:\.|->)\s*wait\s*\(|\bwait_and_throw\b|\bdrain_retained_handles\b|\bwait_all\b|\bcommand_execution_status\b|"
    r"\bget_info\s*<|\bevent_status\b|\bsynchronize\s*\(|\bsleep_for\b|\bsleep_until\b|\bcondition_variable\b"
)
LOCK_DECL = re.compile(r"\bstd::(?:lock_guard|unique_lock|scoped_lock)\b")
BARE_LOCK = re.compile(r"\.\s*(?:lock|try_lock)\s*\(\s*\)")

ALLOWED_INCLUDES = {"algorithm", "cstddef", "cstdint", "cstdio", "string", "vector"}
INCLUDE = re.compile(r"^[ \t]*#[ \t]*include[ \t]*(<[^>\n]*>|\"[^\"\n]*\")", re.MULTILINE)
STATE_KEYWORDS = re.compile(r"\b(?:static|thread_local|extern|mutable|volatile)\b")
INLINE_VAR = re.compile(r"\binline\s+(?!constexpr\b)(?:const\s+)?[\w:<>,\s\*&]+?\b\w+\s*(?:=|\{|\[|;)")
SYNC_NAMES = re.compile(r"\bstd::(?:atomic\w*|mutex|shared_mutex|recursive_mutex|lock_guard|unique_lock|scoped_lock|"
                        r"call_once|once_flag)\b")
FORBIDDEN = re.compile(
    r"\bsycl\b|\b(?:malloc|calloc|realloc|free|aligned_alloc)\s*\(|\bunified_\w+|\bmem_handle\b|\balloc_owner\b|"
    r"\bGGML_LOG\w*|(?<![\w])f?printf\s*\(|\bstd::(?:cout|cerr|clog)\b|\bgetenv\s*\("
)
BUILDER_FORBIDDEN = re.compile(
    r"\b(?:static|thread_local)\b|\bstd::(?:lock_guard|unique_lock|scoped_lock|mutex)\b|\bGGML_LOG\w*|\bgetenv\s*\("
)

_LEXEME_RE = re.compile(r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'", re.DOTALL)


class ContractError(AssertionError):
    pass


class MutantNotApplied(Exception):
    pass


def fail(pin: str, what: str) -> None:
    raise ContractError(f"FAIL [pin {pin}]: {what}")


def strip_comments(src: str) -> str:
    """Comments blanked to spaces with their newlines kept, literals kept. The length is kept too, so an offset in the
    stripped text is the same offset in the source: a mutant finds its body in the stripped text, where an apostrophe in
    a comment cannot open a character literal, and edits the source there."""

    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return re.sub(r"[^\n]", " ", tok)

    return _LEXEME_RE.sub(repl, src)


def norm(text: str) -> str:
    """Whitespace runs collapsed to one space, and none just inside parentheses, so a re-wrap cannot turn a pin red."""
    t = re.sub(r"\s+", " ", text)
    t = re.sub(r"\( ", "(", t)
    return re.sub(r" \)", ")", t)


def block_span(text: str, open_at: int) -> tuple[int, int]:
    """[start, end) of the brace-balanced block opening at `open_at`, literals skipped; (-1, -1) when there is none."""
    if open_at < 0:
        return -1, -1
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
                return open_at, i + 1
        i += 1
    return -1, -1


def body_span(text: str, signature: str) -> tuple[int, int]:
    """[start, end) of the body of the first definition starting with `signature`; a declaration (its parameter list
    closes on a semicolon) is passed over. (-1, -1) when there is none."""
    at = text.find(signature)
    while at >= 0:
        if signature.endswith("("):
            depth = 0
            close = -1
            for i in range(at + len(signature) - 1, len(text)):
                depth += {"(": 1, ")": -1}.get(text[i], 0)
                if depth == 0:
                    close = i
                    break
            if close >= 0 and text[close + 1:close + 200].lstrip().startswith(";"):
                at = text.find(signature, close)
                continue
        return block_span(text, text.find("{", at + len(signature) - 1))
    return -1, -1


def body(text: str, signature: str) -> str:
    start, end = body_span(text, signature)
    return text[start:end] if start >= 0 else ""


def ordered(text: str, *needles: str) -> bool:
    at = 0
    for needle in needles:
        at = text.find(needle, at)
        if at < 0:
            return False
        at += len(needle)
    return True


def enclosing_block_end(text: str, at: int) -> int:
    """The offset just past the `}` that closes the innermost block open at `at`, literals skipped; len(text) if none."""
    depth = 0
    i = at
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
            if depth == 0:
                return i + 1
            depth -= 1
        i += 1
    return n


def call_arguments(text: str, open_paren: int) -> tuple[int, int]:
    """[start, end) of the parenthesised argument list opening at `open_paren`, literals skipped."""
    depth = 0
    i = open_paren
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "\"'":
            j = i + 1
            while j < n and text[j] != ch:
                j += 2 if text[j] == "\\" else 1
            i = j + 1
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return open_paren, i + 1
        i += 1
    return -1, -1


def check(raw: dict[str, str]) -> None:
    sycl = strip_comments(raw["sycl"])
    driver = strip_comments(raw["driver"])
    plan = strip_comments(raw["plan"])
    cache = strip_comments(raw["cache"])
    sync = body(sycl, SYNC_SIG)
    caller = body(sycl, CALLER_SIG)
    builder = body(plan, BUILDER_SIG)
    if not sync:
        fail("D1", "ggml_backend_sycl_synchronize has no body the pin can read")
    if not caller:
        fail("D1", "the swap driver ggml_sycl_run_expert_swap_at_boundary has no definition")

    # D1 ---------------------------------------------------------------------------------------------------------------
    if sync.count(CALLER_CALL) != 1 or not ordered(sync, REFILL_CALL, CALLER_CALL):
        fail("D1", "synchronize does not call the swap driver exactly once, after the refill")
    if sycl.count(CALLER_NAME) != 2:
        fail("D1", f"the swap driver is named {sycl.count(CALLER_NAME)} times, not at its definition and one call")

    # D2 ---------------------------------------------------------------------------------------------------------------
    if sycl.count(STEP_CALL) != 1 or caller.count(STEP_CALL) != 1:
        fail("D2", "the policy step does not run exactly once, inside the swap driver")

    # D3 ---------------------------------------------------------------------------------------------------------------
    helpers = sorted(set(HELPER_DEF.findall(sycl)))
    scopes = [("the swap driver", caller), ("the driver header", driver), ("the plan-view builder", builder)]
    for name in helpers:
        scopes.append((name, _helper_body(sycl, name)))
    for where, text in scopes:
        m = WAITS.search(text)
        if m:
            fail("D3", f"{where} waits or asks an event's status: `{m.group(0).strip()}`")

    # D4 ---------------------------------------------------------------------------------------------------------------
    step_at = caller.find(STEP_CALL)
    m = BARE_LOCK.search(caller)
    if m:
        fail("D4", f"the swap driver takes a lock by a bare call: `{m.group(0)}`")
    for lk in LOCK_DECL.finditer(caller):
        if lk.start() < step_at < enclosing_block_end(caller, lk.start()):
            fail("D4", "the swap driver holds a lock across the policy step")

    # D5 ---------------------------------------------------------------------------------------------------------------
    if not ordered(caller[:step_at], *GATES):
        fail("D5", "the swap driver is not gated, in order, by the switch, the multithreaded latch, the compute flag, "
                   "the graphs in flight and the claim of its instance before the step")

    # D6 ---------------------------------------------------------------------------------------------------------------
    accessor = body(sycl, ACCESSOR_SIG)
    getenvs = [m.start() for m in re.finditer(r"\bgetenv\s*\(\s*ggml_sycl::MOE_SWAP_ENV\s*\)", sycl)]
    if len(ENV_SPELLING.findall(driver)) != 1 or '"GGML_SYCL_MOE_SWAP"' in sycl:
        fail("D6", "GGML_SYCL_MOE_SWAP is not spelled once, as the header's MOE_SWAP_ENV")
    if (len(getenvs) != 1 or not accessor or "getenv(ggml_sycl::MOE_SWAP_ENV)" not in norm(accessor)
            or not re.search(r"\bstatic\s+const\s+bool\b", accessor)):
        fail("D6", "the switch is not read once, by getenv in the accessor's static initializer")
    rows = TABLE_ROW.findall(sycl)
    loop_at = sycl.find(PARSE_LOOP)
    over = ROW_OVERWRITE.search(sycl)
    if len(rows) != 1 or loop_at < 0 or not over or over.start() < loop_at:
        fail("D6", "the settings table does not name the switch once with its row overwritten from the accessor after "
                   "the parse")

    # D7 ---------------------------------------------------------------------------------------------------------------
    warn_spans = []
    for w in re.finditer(r"\bGGML_LOG_WARN\s*\(", sycl):
        warn_spans.append(call_arguments(sycl, w.end() - 1))
    for summary in SUMMARIES:
        calls = [m.start() for m in re.finditer(re.escape(summary), sycl)]
        if not calls:
            fail("D7", f"no `{summary[:-1]}` line is logged")
        for at in calls:
            if not any(s <= at < e for s, e in warn_spans):
                line = sycl.count("\n", 0, at) + 1
                fail("D7", f"`{summary[:-1]}` at line {line} is not an argument of GGML_LOG_WARN")
    faults = norm(body(driver, FAULTS_SIG))
    if not faults or not all(f in faults for f in FAULTS_FIELDS):
        fail("D7", "the second summary line does not name the accepted transactions, evicts and admits apart, "
                   "beside the decode graphs")
    if any(driver.count(c) != 1 for c in TXN_COUNTS):
        fail("D7", "the step does not count each direction's accepted transaction exactly once")

    # D8 ---------------------------------------------------------------------------------------------------------------
    includes = INCLUDE.findall(driver)
    if '"moe-swap-policy.hpp"' not in includes:
        fail("D8", "the driver header does not include the policy header")
    for inc in includes:
        if inc != '"moe-swap-policy.hpp"' and (inc.startswith('"') or inc[1:-1] not in ALLOWED_INCLUDES):
            fail("D8", f"the driver header includes {inc}, which is not the policy header or a listed standard header")
    stateless = re.sub(r"\bstatic_(?:cast|assert)\b", "  ", driver)
    m = STATE_KEYWORDS.search(stateless)
    if m:
        fail("D8", f"`{m.group(0)}` at line {stateless.count(chr(10), 0, m.start()) + 1}: state outside the caller")
    for m in INLINE_VAR.finditer(stateless):
        head = m.group(0)
        if "(" not in head and not re.search(r"\bconst\b", head.split("*")[-1]):
            fail("D8", f"an inline variable at line {stateless.count(chr(10), 0, m.start()) + 1}: global state")
    m = SYNC_NAMES.search(driver)
    if m:
        fail("D8", f"`{m.group(0)}`: the driver header shares nothing and needs no atomic or lock")
    m = FORBIDDEN.search(driver)
    if m:
        fail("D8", f"`{m.group(0).strip()}` at line {driver.count(chr(10), 0, m.start()) + 1}: device, cache, log or "
                   "environment use in the driver header")

    # D9 ---------------------------------------------------------------------------------------------------------------
    if not builder:
        fail("D9", "moe_swap_plan_view_build has no definition in moe-stream-plan.hpp")
    m = BUILDER_FORBIDDEN.search(re.sub(r"\bstatic_(?:cast|assert)\b", "  ", builder))
    if m:
        fail("D9", f"the plan-view builder keeps state, locks, logs or reads the environment: `{m.group(0).strip()}`")

    # D10 --------------------------------------------------------------------------------------------------------------
    prep = body(sycl, PREP_SIG)
    mint = norm(body(sycl, MINT_SIG))
    no_growth = norm(body(sycl, NO_GROWTH_SIG))
    if len(EVICT_MINT.findall(prep)) != 1:
        fail("D10", "the to-host branch does not mint an evict's copy without growth and a loan's with the growing mint")
    if not mint or not all(f in mint for f in MINT_FIELDS):
        fail("D10", "the host mint does not forbid host-zone growth, with its miss kept out of the error log, exactly "
                    "when grow_zone is false, in a pinned WEIGHT request")
    if not no_growth or NO_GROWTH_CALL not in no_growth or "unified_allocate_owner" in no_growth:
        fail("D10", "the no-growth mint is not one call of the host mint with grow_zone false")

    # D11 --------------------------------------------------------------------------------------------------------------
    step = norm(body(driver, STEP_SIG))
    if not step or step.count(COUNTS_CALL) != 1 or len(re.findall(r"\bcounts\(", step)) != 1:
        fail("D11", "the step does not call the counts callback once, with the driver's own decode and prompt arrays")

    # D12 --------------------------------------------------------------------------------------------------------------
    warn = norm(body(sycl, WARN_SIG))
    callback = caller[step_at:]
    if not warn or not all(f in warn for f in WARN_FIELDS):
        fail("D12", "ggml_sycl_moe_swap_warn_refusal does not log each REFUSED reason once per direction at WARN")
    if norm(callback).count(WARN_CALL) != 1 or sycl.count("ggml_sycl_moe_swap_warn_refusal(") != 2:
        fail("D12", "the transaction callback does not hand every move to ggml_sycl_moe_swap_warn_refusal, once")
    if sycl.count("refusals_warned_") != warn.count("refusals_warned_") + 2:
        fail("D12", "the logged-reason sets are touched outside their declaration and the warn helper")

    # D13 --------------------------------------------------------------------------------------------------------------
    latch = norm(body(sycl, LATCH_SIG))
    if norm(plan).count(SHARED_DEF) != 1 or plan.count(SHARED_LITERAL) != 1:
        fail("D13", "the plan header does not define MOE_EXPERT_MOVE_SHARED_STORAGE as the one \"shared storage\"")
    for name, text in (("ggml-sycl.cpp", sycl), ("the driver header", driver), ("unified-cache.cpp", cache)):
        if SHARED_LITERAL in text:
            fail("D13", f"{name} spells \"shared storage\" instead of naming MOE_EXPERT_MOVE_SHARED_STORAGE")
    if norm(body(cache, LOAN_SIG)).count(SHARED_ASSIGN) != 1:
        fail("D13", "the cache's loan does not give the shared-storage refusal by its name")
    if not latch or not all(f in latch for f in LATCH_FIELDS):
        fail("D13", "the block latch is not read from a refused evict whose reason is MOE_EXPERT_MOVE_SHARED_STORAGE")
    if norm(callback).count(LATCH_SET) != 1 or "latches_block" in norm(caller[:step_at]):
        fail("D13", "the transaction callback does not set each move's latch from ggml_sycl_moe_swap_refusal_latches_block")
    if step.count(STEP_LATCH) != 1 or step.count("d.latched[static_cast<size_t>(q.block)]") != 1:
        fail("D13", "the step latches an evict's block other than when the move says the refusal latches it")

    # D14 --------------------------------------------------------------------------------------------------------------
    report = body(sycl, REPORT_SIG)
    lead = report.find("if (!inst.driver) {")
    s0, e0 = block_span(report, report.find("{", lead)) if lead >= 0 else (-1, -1)
    no_driver = norm(report[s0:e0]) if s0 >= 0 else ""
    if not no_driver or not all(f in no_driver for f in NO_DRIVER_FIELDS):
        fail("D14", "with no driver, the report does not print the boundaries turned off and skipped in a WARN line")

    # D15 --------------------------------------------------------------------------------------------------------------
    room = norm(body(sycl, ROOM_SIG))
    if not room or ROOM_RETURN not in room or "host_zone_capacity" in room or "host_zone_used" in room:
        fail("D15", "the evict batch's room is not the WEIGHT zone's no-growth room")
    if norm(cache).count(MINT_CAP) != 1:
        fail("D15", "the evict mint's no-growth cap is not the cache's no-growth room")
    accessor_room = norm(body(cache, NO_GROWTH_ROOM_SIG))
    if not accessor_room or not all(f in accessor_room for f in NO_GROWTH_ROOM_FIELDS):
        fail("D15", "the no-growth room is not capacity capped at the reserved figure, less what is used, never below 0")


def _helper_body(sycl: str, name: str) -> str:
    for m in HELPER_DEF.finditer(sycl):
        if m.group(1) == name:
            start, end = body_span(sycl, sycl[m.start():m.end()])
            return sycl[start:end] if start >= 0 else ""
    return ""


def mutate(raw: dict[str, str], edits: list[tuple[str, str, str, str]]) -> dict[str, str]:
    """Apply (file, signature, old, new) edits. With a signature the anchor must appear exactly once inside that body;
    with an empty one, once in the whole file. The body is found in the comment-stripped text, whose offsets are the
    source's."""
    out = dict(raw)
    for key, sig, old, new in edits:
        text = out[key]
        if sig:
            start, end = body_span(strip_comments(text), sig)
            if start < 0:
                raise MutantNotApplied(f"NOT APPLIED: no body for {sig!r}")
        else:
            start, end = 0, len(text)
        n = text[start:end].count(old)
        if n != 1:
            raise MutantNotApplied(f"NOT APPLIED: anchor appears {n} times, not once: {old[:70]!r}")
        at = text.index(old, start, end)
        out[key] = text[:at] + new + text[at + len(old):]
    return out


def self_test(raw: dict[str, str]) -> int:
    failures: list[str] = []
    n_mutants = 0

    def expect_fail(name: str, pin: str, edits: list[tuple[str, str, str, str]]) -> None:
        nonlocal n_mutants
        n_mutants += 1
        try:
            mutant = mutate(raw, edits)
        except MutantNotApplied as e:
            failures.append(name)
            print(f"  self-test: {name}: {e}")
            return
        try:
            check(mutant)
        except ContractError as e:
            if f"[pin {pin}]" in str(e):
                return
            failures.append(name)
            print(f"  self-test: {name}: caught by a different pin than {pin}: {e}")
            return
        failures.append(name)
        print(f"  self-test: {name}: SURVIVED (pin {pin} did not catch it)")

    ns = "namespace ggml_sycl {\n"
    expect_fail("driver-not-called", "D1", [("sycl", SYNC_SIG, CALLER_CALL, "")])
    expect_fail("driver-before-the-refill", "D1",
                [("sycl", SYNC_SIG, CALLER_CALL, ""), ("sycl", SYNC_SIG, REFILL_CALL, CALLER_CALL + "\n" + REFILL_CALL)])
    expect_fail("driver-called-twice", "D1", [("sycl", SYNC_SIG, CALLER_CALL, CALLER_CALL + "\n" + CALLER_CALL)])
    expect_fail("driver-called-from-elsewhere", "D1",
                [("sycl", "", SYNC_SIG, "static void ggml_sycl_swap_twice(ggml_backend_t backend) {\n    "
                  + CALLER_CALL + "\n}\n\n" + SYNC_SIG)])
    expect_fail("step-run-twice", "D2", [("sycl", CALLER_SIG, STEP_CALL, STEP_CALL + "d, inst.view, call, c, r, t);\n"
                                          "    (void) ggml_sycl::" + STEP_CALL)])
    expect_fail("driver-waits-on-the-queue", "D3",
                [("sycl", CALLER_SIG, "if (!ggml_sycl_moe_swap_enabled()) {",
                  "ctx->stream()->wait();\n    if (!ggml_sycl_moe_swap_enabled()) {")])
    expect_fail("driver-drains-retained-handles", "D3",
                [("sycl", CALLER_SIG, "if (!ggml_sycl_moe_swap_enabled()) {",
                  "(void) ggml_sycl::drain_retained_handles(true, 100);\n    if (!ggml_sycl_moe_swap_enabled()) {")])
    expect_fail("driver-asks-an-event", "D3",
                [("sycl", CALLER_SIG, "if (!ggml_sycl_moe_swap_enabled()) {",
                  "(void) ev.get_info<sycl::info::event::command_execution_status>();\n"
                  "    if (!ggml_sycl_moe_swap_enabled()) {")])
    expect_fail("header-sleeps", "D3", [("driver", "", ns, ns + "inline void nap() { std::this_thread::sleep_for(x); }\n")])
    expect_fail("builder-waits", "D3", [("plan", BUILDER_SIG, "out.n_blocks", "q.wait();\n    out.n_blocks")])
    expect_fail("lock-across-the-step", "D4",
                [("sycl", CALLER_SIG, "if (!ggml_sycl_moe_swap_enabled()) {",
                  "std::lock_guard held(g_tensor_inventory_mutex);\n    if (!ggml_sycl_moe_swap_enabled()) {")])
    expect_fail("bare-lock", "D4",
                [("sycl", CALLER_SIG, "if (!ggml_sycl_moe_swap_enabled()) {",
                  "g_tensor_inventory_mutex.lock();\n    if (!ggml_sycl_moe_swap_enabled()) {")])
    expect_fail("switch-not-read", "D5",
                [("sycl", CALLER_SIG, "if (!ggml_sycl_moe_swap_enabled()) {", "if (false) {")])
    expect_fail("graphs-in-flight-ignored", "D5",
                [("sycl", CALLER_SIG, "ggml_sycl_graph_inflight_count()", "0")])
    expect_fail("instance-not-claimed", "D5",
                [("sycl", CALLER_SIG, "compare_exchange_strong(", "compare_exchange_weak(")])
    expect_fail("claim-deleted", "D5",
                [("sycl", CALLER_SIG,
                  "    bool expected = false;\n"
                  "    if (!inst.claimed.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {\n"
                  "        inst.skipped.fetch_add(1, std::memory_order_relaxed);\n"
                  "        return;\n"
                  "    }\n", "")])
    expect_fail("env-spelled-in-the-backend", "D6",
                [("sycl", "", "ggml_sycl::MOE_SWAP_ENV)", "\"GGML_SYCL_MOE_SWAP\")")])
    expect_fail("env-read-at-every-boundary", "D6",
                [("sycl", ACCESSOR_SIG, "static const bool", "const bool")])
    expect_fail("env-row-not-overwritten", "D6",
                [("sycl", "", "g_ggml_sycl_moe_swap = ggml_sycl_moe_swap_enabled() ? 1 : 0;", "")])
    expect_fail("summary-at-info", "D7",
                [("sycl", "", "GGML_LOG_WARN(\"%s\\n\", ggml_sycl::moe_swap_driver_summary_moves(",
                  "GGML_LOG_INFO(\"%s\\n\", ggml_sycl::moe_swap_driver_summary_moves(")])
    expect_fail("summary-drops-the-admits", "D7",
                [("driver", FAULTS_SIG, "(ull) st.transactions_admit, (ull) st.decode_graphs,", "(ull) st.decode_graphs,")])
    expect_fail("step-forgets-the-evict-transaction", "D7", [("driver", "", "++st.transactions_evict;", "")])
    expect_fail("header-includes-the-cache", "D8",
                [("driver", "", "#include \"moe-swap-policy.hpp\"",
                  "#include \"moe-swap-policy.hpp\"\n#include \"unified-cache.hpp\"")])
    expect_fail("header-static-counter", "D8",
                [("driver", "", ns, ns + "inline int next_id() {\n    static int n = 0;\n    return ++n;\n}\n")])
    expect_fail("header-inline-global", "D8", [("driver", "", ns, ns + "inline int g_swaps = 0;\n")])
    expect_fail("header-atomic", "D8", [("driver", "", ns, ns + "inline int bump(std::atomic<int> & a) { return ++a; }\n")])
    expect_fail("header-logs", "D8", [("driver", "", ns, ns + "inline void say() { GGML_LOG_WARN(\"x\"); }\n")])
    expect_fail("header-reads-the-environment", "D8",
                [("driver", "", ns, ns + "inline bool on() { return getenv(\"X\") != nullptr; }\n")])
    expect_fail("builder-static-scratch", "D9",
                [("plan", BUILDER_SIG, "out.n_blocks", "static std::vector<int> scratch;\n    out.n_blocks")])
    expect_fail("builder-logs", "D9",
                [("plan", BUILDER_SIG, "out.n_blocks", "GGML_LOG_WARN(\"view\\n\");\n    out.n_blocks")])
    expect_fail("evict-mints-with-growth", "D10",
                [("sycl", PREP_SIG, "ggml_sycl_host_expert_alloc_no_growth(device, target.logical_bytes)",
                  "ggml_sycl_host_expert_alloc(device, target.logical_bytes)")])
    expect_fail("no-growth-mint-grows", "D10",
                [("sycl", NO_GROWTH_SIG, "/*grow_zone=*/false", "/*grow_zone=*/true")])
    expect_fail("mint-ignores-the-flag", "D10",
                [("sycl", MINT_SIG, "forbid_host_zone_growth = !grow_zone;", "forbid_host_zone_growth = false;")])
    expect_fail("no-growth-miss-logged", "D10", [("sycl", MINT_SIG, "req.suppress_failure_log", "(void) req.size")])
    expect_fail("no-growth-mints-its-own", "D10",
                [("sycl", NO_GROWTH_SIG, "return ggml_sycl_host_expert_alloc(",
                  "(void) ggml_sycl::unified_allocate_owner(ggml_sycl::alloc_request{});\n"
                  "    return ggml_sycl_host_expert_alloc(")])
    expect_fail("counts-into-the-wrong-arrays", "D11",
                [("driver", STEP_SIG, "counts(d.decode.data(), d.prompt.data());",
                  "counts(d.prompt.data(), d.decode.data());")])
    expect_fail("counts-read-twice", "D11",
                [("driver", STEP_SIG, "counts(d.decode.data(), d.prompt.data());",
                  "counts(d.decode.data(), d.prompt.data());\n        counts(d.decode.data(), d.prompt.data());")])
    expect_fail("refusal-logged-every-time", "D12",
                [("sycl", WARN_SIG, "if (!warned.insert(reason).second) {\n        return;\n    }",
                  "(void) warned.insert(reason);")])
    expect_fail("refusal-logged-for-busy", "D12",
                [("sycl", WARN_SIG, "if (m.outcome != ggml_sycl::MOE_SWAP_MOVE_REFUSED) {", "if (false) {")])
    expect_fail("refusal-at-info", "D12",
                [("sycl", WARN_SIG, "GGML_LOG_WARN(", "GGML_LOG_INFO(")])
    expect_fail("refusal-not-logged", "D12", [("sycl", CALLER_SIG, "ggml_sycl_moe_swap_warn_refusal(inst, device, m);", "")])
    expect_fail("warned-set-cleared-elsewhere", "D12",
                [("sycl", "", "    ggml_sycl_moe_swap_fold(inst);\n    if (inst.driver->stats.boundaries",
                  "    inst.refusals_warned_evict.clear();\n    ggml_sycl_moe_swap_fold(inst);\n"
                  "    if (inst.driver->stats.boundaries")])
    expect_fail("cache-spells-the-refusal", "D13",
                [("cache", LOAN_SIG, "refusal = MOE_EXPERT_MOVE_SHARED_STORAGE;", 'refusal = "shared storage";')])
    expect_fail("latch-compares-a-literal", "D13",
                [("sycl", LATCH_SIG, "ggml_sycl::MOE_EXPERT_MOVE_SHARED_STORAGE", '"shared storage"')])
    expect_fail("latch-on-any-refused-move", "D13",
                [("sycl", LATCH_SIG, "m.kind == ggml_sycl::MOE_SWAP_MOVE_EVICT && ", "")])
    expect_fail("latch-not-set", "D13",
                [("sycl", CALLER_SIG, "m.latches_block = ggml_sycl_moe_swap_refusal_latches_block(m);", "")])
    expect_fail("step-latches-every-refusal", "D13", [("driver", STEP_SIG, "if (m.latches_block) {", "if (true) {")])
    expect_fail("no-driver-line-drops-skipped", "D14",
                [("sycl", REPORT_SIG, "if (off != 0 || skipped != 0) {", "if (off != 0) {")])
    expect_fail("no-driver-line-at-debug", "D14", [("sycl", REPORT_SIG, "GGML_LOG_WARN(\n", "GGML_LOG_DEBUG(\n")])
    expect_fail("room-is-capacity-less-used", "D15",
                [("sycl", ROOM_SIG, "return cache->host_zone_no_growth_room(ggml_sycl::host_zone_id::WEIGHT);",
                  "return cache->host_zone_capacity(ggml_sycl::host_zone_id::WEIGHT);")])
    expect_fail("mint-caps-at-reserved", "D15",
                [("cache", "", "alloc_size > ucache->host_zone_no_growth_room(zone)",
                  "ucache->host_zone_used(zone) + alloc_size > ucache->host_zone_reserved_capacity(zone)")])
    expect_fail("no-growth-room-negative", "D15",
                [("cache", NO_GROWTH_ROOM_SIG, "return limit > used ? limit - used : 0;", "return limit - used;")])
    expect_fail("no-growth-room-ignores-capacity", "D15",
                [("cache", NO_GROWTH_ROOM_SIG, "std::min(capacity, reserved)", "reserved")])

    try:
        mutate(raw, [("sycl", "", "this anchor is in no source", "x")])
        failures.append("stale-anchor-not-reported")
        print("  self-test: a stale mutant anchor is NOT reported")
    except MutantNotApplied:
        print("  self-test: a stale mutant anchor is reported")

    try:
        check(raw)
        print("  self-test: unmodified sources pass")
    except ContractError as e:
        print(f"  self-test: unmodified sources FAIL: {e}")
        failures.append("unmodified")

    if failures:
        print(f"SELF-TEST FAIL: {', '.join(failures)}")
        return 1
    print(f"SELF-TEST PASS: {n_mutants} mutants caught by their pins, unmodified sources pass")
    return 0


def main(argv: list[str]) -> int:
    missing = [str(p) for p in FILES.values() if not p.exists()]
    if missing:
        print(f"FAIL: missing source(s): {', '.join(missing)}")
        return 1
    raw = {k: p.read_text() for k, p in FILES.items()}
    if "--self-test" in argv:
        return self_test(raw)
    try:
        check(raw)
    except ContractError as e:
        print(e)
        return 1
    print("PASS: the swap driver runs once per synchronize boundary after the refill, gated, without a wait or a lock "
          "across its policy, its evict copies never grow the host zone, and its refusals, latch and room are named")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
