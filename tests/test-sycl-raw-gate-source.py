"""Source contract: a raw device allocation is gated on the device's physical bytes.

A raw allocation takes device memory the VRAM arena did not reserve. The arena reservation is committed whole when it
is made (one chunk or N), so the figure a raw request must fit in is the device total less the arena reservation, less
the raw rows the cache holds outside the arena, less the bytes in use outside this process's accounting when the
device's first cache was created (zone-raw-gate.hpp; test-zone-sizing Case 35 checks the arithmetic). Zone counters say
how full the arena is, not how much of the device is taken: a gate that read them admitted raw requests past an arena
whose zones were empty but whose bytes were all reserved. These checks pin where the two raw gates sit and what they
read:

- unified_alloc() has one raw device exit and the gate sits in it, before the tracked malloc; nothing gates or evicts
  before a zone is tried, so a request an arena zone serves never meets the gate;
- the figure is physical: the device total, arena_total_size(), the raw rows of the registry and the recorded term,
  never a zone counter, the cache's used bytes, its budget or a live driver free reading;
- with an arena active a refusal evicts nothing (evicting cached weights frees nothing outside the reservation); only
  with no arena, where cached weights are raw rows, does the gate evict and look again; a hold-induced spill is
  refused at ERROR without evicting;
- both device-pool chunk sites use the same gate;
- the term outside this process's accounting is written at one site, once per device, from the reading that sizes the
  first arena reservation, and read only by the gate's figure; only the pure helpers and the two log lines read it;
- a refusal names the request, the arena, the raw rows, the term, the device total and what fits; the figure is
  printed once per device at INFO.

Host-only text assertions, no SYCL device. llama_test_pytest hands this file to pytest.main(), so the checks live
inside test_*() functions. Checks read COMMENT-STRIPPED, whitespace-normalised text. Each check has a mutation witness
that applies the regression it guards inside the one function it names and must make the check fail.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"
UC = (SYCL / "unified-cache.cpp").read_text()
UH = (SYCL / "unified-cache.hpp").read_text()
DP = (SYCL / "device-pool.hpp").read_text()
BACKEND = (SYCL / "ggml-sycl.cpp").read_text()

_LEXEME_RE = re.compile(
    r'"(?:\\.|[^"\\\n])*"'  # string literal (kept)
    r"|'(?:\\.|[^'\\\n])*'"  # char literal (kept)
    r"|//[^\n]*"  # line comment (dropped)
    r"|/\*.*?\*/",  # block comment (dropped)
    flags=re.DOTALL,
)


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def code(src: str) -> str:
    """Comment-stripped text with every run of whitespace made one space. Idempotent."""
    return re.sub(r"\s+", " ", strip_comments(src))


def block_at(text: str, start: int) -> str:
    """The brace block that opens at the first '{' at or after `start`, skipping string and char literals."""
    o = text.find("{", start) if start >= 0 else -1
    if o == -1:
        return ""
    depth, i = 0, o
    while i < len(text):
        c = text[i]
        if c in "\"'":
            j = i + 1
            while j < len(text) and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            i = j + 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[o : i + 1]
        i += 1
    return ""


def fn_body(text: str, sig: str) -> str:
    s = text.find(sig)
    return block_at(text, s + len(sig)) if s != -1 else ""


def mutate_in_func(src: str, sig: str, old: str, new: str, nth: int = 0) -> str:
    """`src` as code() with the nth `old` inside the function `sig` replaced by `new`; never text outside it."""
    c = code(src)
    s = c.find(sig)
    body = block_at(c, s + len(sig)) if s != -1 else ""
    assert body and body.count(old) > nth, f"witness anchor {old!r} (occurrence {nth}) is not in {sig!r}"
    k = -1
    for _ in range(nth + 1):
        k = body.find(old, k + 1)
    b = c.find(body, s)
    return c[:b] + body[:k] + new + body[k + len(old) :] + c[b + len(body) :]


UA_SIG = "bool unified_alloc(const alloc_request & req_in, alloc_handle * out)"
GATE_SIG = "static bool unified_alloc_raw_gate(const alloc_request & req, size_t alloc_size, bool hold_spill)"
ADMITS_SIG = "bool unified_cache_raw_gate_admits(int device, sycl::queue & queue, size_t bytes, const char * what)"
INPUTS_SIG = "zone_raw_gate_inputs unified_cache_raw_gate_inputs(int device, sycl::queue & queue)"
NOTE_SIG = "static void unified_cache_note_raw_gate_outside_accounting(int device, size_t total, size_t free_mem)"
FIGURE_SIG = (
    "static void unified_cache_raw_gate_log_figure_once(int device, const zone_raw_gate_inputs & in, size_t request)"
)
TEXT_SIG = (
    "static std::string unified_cache_raw_gate_refusal_text(int device, const char * what, "
    "const zone_raw_gate_inputs & in, size_t request)"
)
CREATE_SIG = "static unified_cache * create_cache_for_device("
POOL_ALLOC_SIG = "alloc_result allocate(size_t size, size_t align = 256)"
POOL_GROW_SIG = "size_t grow_one_chunk()"

RAW_EXIT = "if (!ptr && !kv_spill_to_host) {"
GATE_CALL = "if (!unified_alloc_raw_gate(req, alloc_size, hold_spill)) { return false; }"
TRACKED_MALLOC = "unified_cache_malloc_device_tracked("
TERM_RE = re.compile(r"\bg_raw_gate_outside_accounting\b")
FIELD_RE = re.compile(r"\boutside_accounting\b")
NOTE_CALL = "unified_cache_note_raw_gate_outside_accounting(device_id, detected_total_mem, detected_free_mem);"
POOL_GATE_RE = re.compile(
    r'if \(device_id_ >= 0 && !unified_cache_raw_gate_admits\(device_id_, queue_, backing_size, "[^"]+"\)\) '
    r"\{ return [^;]*; \}"
)
# What a physical figure never reads: a zone counter, the cache's own used bytes or budget, or a live driver reading.
NOT_PHYSICAL = (
    "arena_non_weight_used",
    "used()",
    "zone_used(",
    "zone_available(",
    "weight_bytes(",
    "budget(",
    "total_available_bytes",
    "total_committed_bytes",
    "get_memory_info",
    "free_memory",
    "query_device_memory_no_info",
)


def claim_gate_sits_at_the_raw_exit(uc: str) -> bool:
    ua = fn_body(code(uc), UA_SIG)
    exit_block = block_at(ua, ua.find(RAW_EXIT)) if RAW_EXIT in ua else ""
    gate_at = exit_block.find(GATE_CALL)
    return (
        bool(exit_block)
        # one raw device exit, gated before its malloc
        and ua.count(TRACKED_MALLOC) == 1
        and 0 <= gate_at < exit_block.find(TRACKED_MALLOC)
        and ua.count("unified_alloc_raw_gate(") == 1
        # nothing before a zone is tried gates on the device total or evicts
        and "evict_and_flush(" not in ua
        and "unified_alloc_total_vram(" not in ua
    )


def claim_figure_is_physical(uc: str, uh: str) -> bool:
    c = code(uc)
    inputs, gate, admits = fn_body(c, INPUTS_SIG), fn_body(c, GATE_SIG), fn_body(c, ADMITS_SIG)
    if not (inputs and gate and admits):
        return False
    reads = (
        "in.device_total = unified_alloc_total_vram(device, queue);",
        "cache->arena_total_size()",
        "in.raw_held = unified_cache_raw_device_held_bytes(device, nullptr);",
        "in.outside_accounting = g_raw_gate_outside_accounting[device].load(",
    )
    return (
        all(r in inputs for r in reads)
        and all(g.count("unified_cache_raw_gate_inputs(") >= 1 and "zone_raw_gate_fits(" in g for g in (gate, admits))
        and not any(bad in body for bad in NOT_PHYSICAL for body in (inputs, gate, admits))
        and '#include "zone-raw-gate.hpp"' in uh
    )


def claim_arena_trip_evicts_nothing(uc: str) -> bool:
    c = code(uc)
    gate, admits = fn_body(c, GATE_SIG), fn_body(c, ADMITS_SIG)
    if not (gate and admits) or gate.count("evict_and_flush(") != 1:
        return False
    arena_off = "if (cache && !cache->arena_active()) {"
    off_block = block_at(gate, gate.find(arena_off)) if arena_off in gate else ""
    hold = re.search(r"if \(hold_spill\) \{[^{}]*GGML_LOG_ERROR\([^{}]*return false; \}", gate)
    return (
        "evict_and_flush(" in off_block
        and hold is not None
        and hold.start() < gate.find("evict_and_flush(")
        and "evict" not in admits
    )


def claim_pool_sites_use_the_gate(dp: str) -> bool:
    d = code(dp)
    if "unified_cache_total_available_bytes" in d:
        return False
    if "bool unified_cache_raw_gate_admits(int device, sycl::queue & queue, size_t bytes, const char * what);" not in d:
        return False
    for sig in (POOL_ALLOC_SIG, POOL_GROW_SIG):
        body = fn_body(d, sig)
        gates = list(POOL_GATE_RE.finditer(body))
        if len(gates) != 1 or not 0 <= gates[0].start() < body.find("device_pool_alloc_chunk("):
            return False
    return True


def claim_term_has_one_writer(uc: str, dp: str, backend: str) -> bool:
    c = code(uc)
    note, inputs = fn_body(c, NOTE_SIG), fn_body(c, INPUTS_SIG)
    create = fn_body(c, CREATE_SIG)
    figure, text = fn_body(c, FIGURE_SIG), fn_body(c, TEXT_SIG)
    if not (note and inputs and create and figure and text):
        return False
    # the table: declared, written once in the note, read once in the inputs, nothing else
    if len(TERM_RE.findall(c)) != 3:
        return False
    if len(TERM_RE.findall(note)) != 1 or ".store(" not in note[note.find("g_raw_gate_outside_accounting[") :]:
        return False
    if len(TERM_RE.findall(inputs)) != 1:
        return False
    # once per device, from the reading the first reservation already took: no driver query of its own
    if "g_raw_gate_outside_accounting_noted[device].exchange(true)" not in note:
        return False
    if any(q in note for q in ("query_device_memory_no_info", "get_memory_info", "get_info<", "free_memory")):
        return False
    # the one call: in the first cache creation, before the cache (and its arena) is made
    if c.count("unified_cache_note_raw_gate_outside_accounting(") != 2 or create.count(NOTE_CALL) != 1:
        return False
    if not create.find(NOTE_CALL) < create.find("std::make_unique<unified_cache>("):
        return False
    # the field: assigned by the inputs, read only by the two log lines (the pure helpers live in the header)
    in_three = sum(len(FIELD_RE.findall(b)) for b in (inputs, figure, text))
    return (
        len(FIELD_RE.findall(c)) == in_three
        and len(FIELD_RE.findall(inputs)) == 1
        and FIELD_RE.search(figure) is not None
        and FIELD_RE.search(text) is not None
        and FIELD_RE.search(code(dp)) is None
        and FIELD_RE.search(backend) is None
    )


def claim_refusal_names_every_figure(uc: str) -> bool:
    c = code(uc)
    gate, admits, text = fn_body(c, GATE_SIG), fn_body(c, ADMITS_SIG), fn_body(c, TEXT_SIG)
    if not (gate and admits and text):
        return False
    args = text[text.find("snprintf(") :]
    named = (
        "request / MB",
        "zone_raw_gate_free_outside_arena(in) / MB",
        "in.device_total / MB",
        "in.arena_bytes / MB",
        "in.raw_held / MB",
        "in.outside_accounting / MB",
    )
    words = ("arena", "raw outside the arena", "outside this process's accounting", "device")
    general = re.search(
        r"if \(req\.suppress_failure_log\) \{ GGML_SYCL_DEBUG\([^;]*text[^;]*\); \} "
        r"else \{ GGML_LOG_WARN\([^;]*text[^;]*\); \}",
        gate,
    )
    return (
        all(a in args for a in named)
        and all(w in args for w in words)
        and general is not None
        and "unified_cache_raw_gate_refusal_text(" in gate
        and re.search(r"GGML_LOG_WARN\([^;]*unified_cache_raw_gate_refusal_text\(", admits) is not None
        and not any("GGML_ABORT" in b or "GGML_ASSERT" in b for b in (gate, admits, text))
    )


def claim_figure_is_logged_once_at_info(uc: str) -> bool:
    c = code(uc)
    figure, gate, admits = fn_body(c, FIGURE_SIG), fn_body(c, GATE_SIG), fn_body(c, ADMITS_SIG)
    if not (figure and gate and admits):
        return False
    log_first = all(
        0 <= g.find("unified_cache_raw_gate_log_figure_once(") < g.find("zone_raw_gate_fits(") for g in (gate, admits)
    )
    return (
        re.search(r'GGML_LOG_INFO\( ?"\[RAW-GATE\]', figure) is not None
        and "raw_gate_figure_logged[device].exchange(true)" in figure
        and log_first
    )


def test_gate_sits_at_the_raw_exit():
    assert claim_gate_sits_at_the_raw_exit(UC), (
        "unified_alloc() does not gate its one raw device exit before the tracked malloc, or something before a zone "
        "is tried still gates on the device total or evicts"
    )


def test_figure_is_physical():
    assert claim_figure_is_physical(UC, UH), (
        "a raw gate's figure is not the device total less the arena reservation, the raw rows held and the term "
        "outside this process's accounting, or it reads a zone counter, the cache's used bytes or budget, or a live "
        "driver reading"
    )


def test_arena_trip_evicts_nothing():
    assert claim_arena_trip_evicts_nothing(UC), (
        "a raw gate evicts cached weights with an arena active (they sit in the reservation, so evicting frees nothing "
        "outside it), or a hold-induced spill is not refused at ERROR before any eviction"
    )


def test_pool_sites_use_the_gate():
    assert claim_pool_sites_use_the_gate(DP), (
        "a device-pool chunk site does not ask unified_cache_raw_gate_admits() for its whole backing size before it "
        "allocates, or still reads the cache budget"
    )


def test_term_has_one_writer():
    assert claim_term_has_one_writer(UC, DP, BACKEND), (
        "the term outside this process's accounting is not written at one site, once per device, in the first cache "
        "creation before the arena, or is read anywhere but the gate's figure and its two log lines"
    )


def test_refusal_names_every_figure():
    assert claim_refusal_names_every_figure(UC), (
        "a raw-gate refusal does not name the request, what fits, the device total, the arena, the raw rows and the "
        "term, or is not WARN (DEBUG when the caller suppresses it), or can abort"
    )


def test_figure_is_logged_once_at_info():
    assert claim_figure_is_logged_once_at_info(UC), "the raw gate's figure is not printed once per device at INFO"


HOLD_IF = "if (hold_spill) {"
ARENA_OFF = "if (cache && !cache->arena_active()) {"
ADMITS_FITS = "if (zone_raw_gate_fits(in, bytes)) {"
INPUTS_TERM = "in.outside_accounting = g_raw_gate_outside_accounting[device].load(std::memory_order_relaxed);"
TEXT_TERM = "in.outside_accounting / MB"


def test_witness_raw_exit_ungated():
    assert not claim_gate_sits_at_the_raw_exit(mutate_in_func(UC, UA_SIG, GATE_CALL, ""))


def test_witness_evicts_before_the_zones():
    assert not claim_gate_sits_at_the_raw_exit(
        mutate_in_func(
            UC,
            UA_SIG,
            "bool kv_placed = false;",
            "bool kv_placed = false; "
            "if (auto * c = get_unified_cache_for_device(req.device)) { c->evict_and_flush(1); }",
        )
    )


def test_witness_figure_reads_used_bytes():
    assert not claim_figure_is_physical(
        mutate_in_func(UC, INPUTS_SIG, "cache->arena_total_size()", "cache->used()"), UH
    )


def test_witness_figure_reads_zone_counters():
    assert not claim_figure_is_physical(
        mutate_in_func(
            UC,
            INPUTS_SIG,
            "unified_cache_raw_device_held_bytes(device, nullptr)",
            "unified_cache_arena_non_weight_used(device)",
        ),
        UH,
    )


def test_witness_gate_admits_on_zone_room():
    assert not claim_figure_is_physical(
        mutate_in_func(
            UC,
            GATE_SIG,
            "if (zone_raw_gate_fits(in, alloc_size)) {",
            "if (zone_raw_gate_fits(in, alloc_size) || cache->zone_available(vram_zone_id::RUNTIME) >= alloc_size) {",
        ),
        UH,
    )


def test_witness_arena_trip_evicts():
    assert not claim_arena_trip_evicts_nothing(mutate_in_func(UC, GATE_SIG, ARENA_OFF, "if (cache) {"))


def test_witness_hold_spill_evicts():
    assert not claim_arena_trip_evicts_nothing(mutate_in_func(UC, GATE_SIG, HOLD_IF, "if (false) {"))


def test_witness_pool_gate_evicts():
    assert not claim_arena_trip_evicts_nothing(
        mutate_in_func(
            UC,
            ADMITS_SIG,
            ADMITS_FITS,
            "if (auto * c = get_existing_cache_for_device(device)) { c->evict_and_flush(bytes); } " + ADMITS_FITS,
        )
    )


def test_witness_pool_chunk_ungated():
    assert not claim_pool_sites_use_the_gate(
        mutate_in_func(
            DP, POOL_ALLOC_SIG, "!unified_cache_raw_gate_admits(", "false && !unified_cache_raw_gate_admits("
        )
    )


def test_witness_pool_grow_gates_a_smaller_size():
    assert not claim_pool_sites_use_the_gate(
        mutate_in_func(DP, POOL_GROW_SIG, "queue_, backing_size,", "queue_, chunk_size,")
    )


def test_witness_term_written_by_the_gate():
    assert not claim_term_has_one_writer(
        mutate_in_func(
            UC,
            GATE_SIG,
            HOLD_IF,
            "g_raw_gate_outside_accounting[req.device].store(0, std::memory_order_relaxed); " + HOLD_IF,
        ),
        DP,
        BACKEND,
    )


def test_witness_term_re_read_at_every_gate():
    assert not claim_term_has_one_writer(
        mutate_in_func(
            UC,
            INPUTS_SIG,
            INPUTS_TERM,
            "unified_cache_note_raw_gate_outside_accounting(device, in.device_total, 0); " + INPUTS_TERM,
        ),
        DP,
        BACKEND,
    )


def test_witness_term_noted_every_creation():
    assert not claim_term_has_one_writer(
        mutate_in_func(UC, NOTE_SIG, "g_raw_gate_outside_accounting_noted[device].exchange(true)", "false"), DP, BACKEND
    )


def test_witness_term_read_by_the_gate():
    assert not claim_term_has_one_writer(
        mutate_in_func(UC, GATE_SIG, HOLD_IF, "if (hold_spill || in.outside_accounting == 0) {"), DP, BACKEND
    )


def test_witness_refusal_drops_the_term():
    assert not claim_refusal_names_every_figure(mutate_in_func(UC, TEXT_SIG, TEXT_TERM, "0.0"))


def test_witness_refusal_always_warns():
    assert not claim_refusal_names_every_figure(
        mutate_in_func(UC, GATE_SIG, "if (req.suppress_failure_log) {", "if (false) {")
    )


def test_witness_refusal_aborts():
    assert not claim_refusal_names_every_figure(
        mutate_in_func(UC, ADMITS_SIG, "return false;", 'GGML_ABORT("raw gate refused");')
    )


def test_witness_figure_at_debug():
    assert not claim_figure_is_logged_once_at_info(mutate_in_func(UC, FIGURE_SIG, "GGML_LOG_INFO(", "GGML_SYCL_DEBUG("))


def test_witness_figure_every_request():
    assert not claim_figure_is_logged_once_at_info(
        mutate_in_func(UC, FIGURE_SIG, "raw_gate_figure_logged[device].exchange(true)", "false")
    )


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
