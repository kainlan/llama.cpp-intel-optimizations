"""Source contract for llama.cpp-mmi1: a context whose scheduler compute buffer fits no tier of a SYCL device is
refused BY NAME, with what the card could have held.

Qwen3.8-Flash-Next IQ3_XXS on the B70, -ub 512, no -c (n_ctx 262144): the planner filled the arena, uize demoted all 48
KV layers to the host, and the 2035 MiB compute buffer found no room. The context failed with "failed to allocate compute
pp buffers" and nothing else. The wiring that must say more:

(A) The SYCL allocator records the scheduler compute buffer (an allocation made inside the compute scope) it could not
    place, per device, and clears the record where the runtime-context transaction publishes.
(B) ggml_backend_sycl_compute_refusal_advice() (ggml-sycl.h, reached by llama.cpp through the backend proc table)
    gathers the tiers' room, the kpjw hold-spill fit's -ub, and the budget authority's figures, and answers with
    compute_refusal_message() (compute-refusal-advice.hpp, host-tested by test-compute-refusal-advice).
(C) llama_context::sched_reserve() appends that text to each of its three "failed to allocate compute ... buffers"
    refusals.
(D) The host-pinned fallback's refusal says what it is: the pinned pool's base is not aligned, and a host-pinned buffer
    is no home for a device compute buffer; it is not an "under-reserve" of a device buffer.

Host-only, pure text assertions -- no SYCL device required. llama_test_pytest hands this file to pytest.main(), so the
checks live inside test_*() functions. Checks run against COMMENT-STRIPPED text, and each has a mutation witness so it is
known to fail on the regression it guards.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GGML_SYCL_H = (ROOT / "ggml/include/ggml-sycl.h").read_text()
# CLAUDE.md: plain file I/O, not the codescout index -- ggml-sycl.cpp is ~100k lines and that index is blind there.
GGML_SYCL_CPP = (ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()
LLAMA_CONTEXT_CPP = (ROOT / "src/llama-context.cpp").read_text()

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


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def bounded(code: str, start: str, end: str) -> str:
    s = code.find(start)
    assert s != -1, f"{start!r} not found"
    e = code.find(end, s + 1)
    assert e != -1, f"could not bound {start!r} (looked for {end!r})"
    return code[s:e]


ADVICE_FN = "ggml_backend_sycl_compute_refusal_advice"
ALLOC_START = "static ggml_backend_buffer_t ggml_backend_sycl_buffer_type_alloc_buffer("
ALLOC_END = "} catch (const sycl::exception & exc) {"
PUBLISH_START = "static ggml_backend_buffer_t ggml_backend_sycl_buffer_publish("
PUBLISH_END = "static const char * ggml_backend_sycl_buffer_type_get_name("
TXN_START = "static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction("
TXN_END = "void ggml_backend_sycl_set_runtime_context("
SCHED_START = "void llama_context::sched_reserve() {"


def advice_body(cpp: str) -> str:
    code = strip_comments(cpp)
    start = code.find("size_t " + ADVICE_FN + "(")
    assert start != -1, f"{ADVICE_FN} is not defined in ggml-sycl.cpp"
    end = code.find("\n}\n", start)
    assert end != -1
    return norm(code[start:end])


def sched_reserve_body(cpp: str) -> str:
    code = strip_comments(cpp)
    start = code.find(SCHED_START)
    assert start != -1, "llama_context::sched_reserve() not found"
    end = code.find("\n}\n", start)
    return norm(code[start:end])


# ---------------------------------------------------------------------------
# (A) record
# ---------------------------------------------------------------------------


def violations_record(cpp: str) -> list[str]:
    found: list[str] = []
    code = strip_comments(cpp)
    if not re.search(r"static\s+std::atomic<size_t>\s+g_compute_placement_refused_bytes\[GGML_SYCL_MAX_DEVICES\]", code):
        found.append("the per-device refused-request record g_compute_placement_refused_bytes is missing")
    alloc = norm(bounded(code, ALLOC_START, ALLOC_END))
    notes = re.findall(r"ggml_sycl_note_compute_placement_refusal\(", alloc)
    if len(notes) < 2:
        found.append(
            "the allocator must record a refused compute buffer at both ways out: the allocation that fails and the "
            f"publish that refuses (found {len(notes)} note calls)"
        )
    if "ggml_sycl_compute_alloc_scope_active()" not in alloc:
        found.append("the record must be gated on the compute scope (a weight or state buffer is not a compute buffer)")
    if "ggml_sycl_note_compute_placement_host_pinned_refusal(" not in alloc:
        found.append("the allocator does not record that the refused buffer was a host-pinned fallback")
    txn = bounded(code, TXN_START, TXN_END)
    if not re.search(r"g_compute_placement_refused_bytes\[ctx->device\]\.store\(0", txn):
        found.append("the runtime-context transaction does not clear the refused-request record where it publishes")
    if not re.search(r"static std::atomic<uint64_t>\s+g_compute_placement_refused_writer\[GGML_SYCL_MAX_DEVICES\]", code):
        found.append("each refused-request record must carry the token of the thread that wrote it")
    note = norm(bounded(code, "static void ggml_sycl_note_compute_placement_refusal(", "\n}\n"))
    if "g_compute_placement_refused_writer[device].store(" not in note or "ggml_sycl_this_thread_token()" not in note:
        found.append("the note must stamp the record with the writing thread's token")
    clear = norm(bounded(code, "static void ggml_sycl_clear_compute_placement_refusals() {", "\n}\n"))
    if "ggml_sycl_this_thread_token()" not in clear or "compare_exchange" not in clear:
        found.append(
            "clearing at scope entry must touch only the records THIS thread wrote: the scope opens on every decode "
            "alloc too, and another thread's pending advice on another device must survive it"
        )
    scope = norm(bounded(code, "void ggml_backend_sycl_compute_alloc_scope(bool enter) {", "\n}\n"))
    if "ggml_sycl_clear_compute_placement_refusals(" not in scope or "compute_alloc_scope_active()" not in scope:
        found.append(
            "entering the OUTERMOST compute scope must clear the refused-request records, so a retry that succeeded "
            "cannot leak its text into a later unrelated refusal"
        )
    return found


def test_the_allocator_records_a_refused_compute_buffer():
    assert violations_record(GGML_SYCL_CPP) == []


@pytest.mark.parametrize(
    "mutation",
    ["no-record", "ungated", "no-reset", "single-note", "clear-all", "unstamped"],
)
def test_record_has_a_mutation_witness(mutation):
    src = GGML_SYCL_CPP
    if mutation == "no-record":
        mutated = src.replace("g_compute_placement_refused_bytes", "g_other_record")
    elif mutation == "ungated":
        mutated = src.replace("ggml_sycl_compute_alloc_scope_active()", "true")
    elif mutation == "no-reset":
        mutated = re.sub(r"g_compute_placement_refused_bytes\[ctx->device\]\.store\(0[^;]*;", "", src)
    elif mutation == "clear-all":
        mutated = src.replace("compare_exchange_strong", "exchange_all")
    elif mutation == "unstamped":
        mutated = src.replace("g_compute_placement_refused_writer[device].store(", "g_compute_placement_refused_x[device].store(")
    else:
        mutated = src.replace("ggml_sycl_note_compute_placement_refusal(buft_ctx->device, size);", "", 1)
    assert mutated != src, f"mutation {mutation} did not change the source"
    assert violations_record(mutated), f"mutation {mutation} was not witnessed"


# ---------------------------------------------------------------------------
# (B) the advice entry
# ---------------------------------------------------------------------------


def violations_advice(h: str, cpp: str) -> list[str]:
    found: list[str] = []
    if not re.search(r"GGML_BACKEND_API\s+size_t\s+" + ADVICE_FN + r"\(", strip_comments(h)):
        found.append(f"{ADVICE_FN} is not declared GGML_BACKEND_API in ggml-sycl.h")
    code = strip_comments(cpp)
    if '#include "ggml-sycl/compute-refusal-advice.hpp"' not in cpp:
        found.append("ggml-sycl.cpp does not include compute-refusal-advice.hpp")
    if not re.search(r'strcmp\(name, "' + ADVICE_FN + r'"\) == 0\) \{\s*return \(void \*\) ' + ADVICE_FN, code):
        found.append(f"{ADVICE_FN} is not registered in the backend proc table")
    try:
        body = advice_body(cpp)
    except AssertionError as e:
        return found + [str(e)]
    for needle, why in (
        ("ggml_sycl_hold_spill_fit(", "the kpjw hold-spill fit (the -ub the realized check would also accept)"),
        ("hold_answer.largest_ub", "the kpjw largest-ub machinery (the answer the hold-spill fit gives when it refuses a rung)"),
        ("g_compute_placement_refused_bytes", "the recorded refused request"),
        ("zone_largest_free(ggml_sycl::vram_zone_id::RUNTIME)", "the RUNTIME zone's room"),
        ("ggml_sycl_hold_kv_room(", "the KV zone's room, net of the KV still to place"),
        ("ggml_sycl_device_budget_authority(", "the budget authority's own figures, not a re-parse of the env var"),
        ("unified_cache_hold_free_before(", "the ledger's free memory outside the arena"),
        ("g_compute_placement_host_pinned_refused", "whether the host-pinned fallback was tried and refused"),
        ("ctx->device < 0 || ctx->device >= GGML_SYCL_MAX_DEVICES", "the device range guard its siblings carry"),
        ("compute_refusal_advise(", "the advice arithmetic"),
        ("compute_refusal_message(", "the message text"),
    ):
        if needle not in body:
            found.append(f"{ADVICE_FN} does not use {needle} ({why})")
    if "getenv" in body:
        found.append(f"{ADVICE_FN} re-parses an environment variable instead of reading the budget authority")
    return found


def test_the_advice_entry_gathers_every_tier_and_the_authority():
    assert violations_advice(GGML_SYCL_H, GGML_SYCL_CPP) == []


@pytest.mark.parametrize(
    "mutation",
    ["undeclared", "unregistered", "no-hold-fit", "no-authority", "env-reparse", "no-message", "no-runtime-room"],
)
def test_advice_has_a_mutation_witness(mutation):
    h, cpp = GGML_SYCL_H, GGML_SYCL_CPP
    if mutation == "undeclared":
        h = h.replace("GGML_BACKEND_API size_t " + ADVICE_FN, "size_t " + ADVICE_FN)
    elif mutation == "unregistered":
        cpp = cpp.replace('strcmp(name, "' + ADVICE_FN + '") == 0', 'strcmp(name, "x") == 0')
    elif mutation == "no-hold-fit":
        cpp = cpp.replace("hold_answer.largest_ub", "hold_answer.x")
    elif mutation == "no-authority":
        cpp = cpp.replace("ggml_sycl_device_budget_authority(device, total", "ggml_sycl_device_budget_x(device, total", 1)
    elif mutation == "env-reparse":
        cpp = cpp.replace("compute_refusal_advise(in);", 'compute_refusal_advise(in); (void) getenv("X");', 1)
    elif mutation == "no-message":
        cpp = cpp.replace("compute_refusal_message(", "compute_refusal_msg(", 1)
    else:
        cpp = cpp.replace("zone_largest_free(ggml_sycl::vram_zone_id::RUNTIME)", "zone_available(ggml_sycl::vram_zone_id::RUNTIME)", 1)
    assert (h, cpp) != (GGML_SYCL_H, GGML_SYCL_CPP), f"mutation {mutation} did not change the source"
    assert violations_advice(h, cpp), f"mutation {mutation} was not witnessed"


# ---------------------------------------------------------------------------
# (C) llama.cpp appends it to the three reserve refusals
# ---------------------------------------------------------------------------


def violations_llama(src: str) -> list[str]:
    found: list[str] = []
    code = strip_comments(src)
    if "llama_context_sycl_compute_refusal_proc" not in code:
        found.append("no proc-address lookup for " + ADVICE_FN)
    elif ADVICE_FN not in code:
        found.append(ADVICE_FN + " is never named by llama-context.cpp")
    if not re.search(r"static std::string llama_context_sycl_compute_refusal_text\(", code):
        found.append("the text helper llama_context_sycl_compute_refusal_text is missing")
    body = sched_reserve_body(src)
    refusals = re.findall(r'throw llama_auto_ubatch_fit_refusal\(\s*"failed to allocate compute (pp|tg) buffers"([^;]*)\)\s*;', body)
    if len(refusals) != 3:
        found.append(f"sched_reserve() should throw its three compute-buffer refusals through one shape (found {len(refusals)})")
    for kind, rest in refusals:
        if "llama_context_sycl_compute_refusal_text(" not in rest:
            found.append(f"the {kind} refusal does not append the by-name text")
        if "cparams.n_ubatch" in rest:
            found.append(
                f"the {kind} refusal passes cparams.n_ubatch: that buffer was not necessarily shaped at -ub N "
                "(the tg reserve is shaped by n_seqs, and a context under -ub is shaped by n_ctx); pass the shape it ran at"
            )
    tg = [rest for kind, rest in refusals if kind == "tg"]
    if tg and not all(re.search(r",\s*0\s*\)", r) for r in tg):
        found.append("the tg refusal must pass 0: a token-generation buffer is not shaped by -ub")
    return found


def test_sched_reserve_refusals_carry_the_text():
    assert violations_llama(LLAMA_CONTEXT_CPP) == []


@pytest.mark.parametrize("mutation", ["no-proc", "no-helper", "pp-bare", "tg-bare"])
def test_llama_has_a_mutation_witness(mutation):
    src = LLAMA_CONTEXT_CPP
    if mutation == "no-proc":
        mutated = src.replace("llama_context_sycl_compute_refusal_proc", "llama_context_sycl_other_proc")
    elif mutation == "no-helper":
        mutated = src.replace("static std::string llama_context_sycl_compute_refusal_text(", "static std::string llama_x(")
    elif mutation == "pp-bare":
        mutated = re.sub(
            r'("failed to allocate compute pp buffers")\s*\+\s*llama_context_sycl_compute_refusal_text\([^)]*\)\)', r"\1)", src, count=1
        )
    else:
        mutated = re.sub(
            r'("failed to allocate compute tg buffers")\s*\+\s*llama_context_sycl_compute_refusal_text\([^)]*\)\)', r"\1)", src, count=1
        )
    assert mutated != src, f"mutation {mutation} did not change the source"
    assert violations_llama(mutated), f"mutation {mutation} was not witnessed"


# ---------------------------------------------------------------------------
# (D) the host-pinned fallback's refusal says what it is
# ---------------------------------------------------------------------------


def violations_publish(cpp: str) -> list[str]:
    found: list[str] = []
    code = strip_comments(cpp)
    body = norm(bounded(code, PUBLISH_START, PUBLISH_END))
    if "ggml_sycl::alloc_tier::HOST_PINNED" not in body:
        found.append("the publish guard does not tell a host-pinned fallback from a device buffer")
    if "host-pinned fallback" not in body:
        found.append("the publish guard does not say the buffer is a host-pinned fallback")
    if "not aligned" not in body and "is aligned only to" not in body:
        found.append("the publish guard does not say the pinned pool's base is the cause")
    if "did not fit any device tier" in body:
        found.append("the publish guard claims no device tier held the buffer; it does not know why it fell back")
    if "under-reserve" not in body:
        found.append("the device-buffer wording (under-reserve) was dropped for the case it is true of")
    return found


def test_the_host_pinned_refusal_says_what_it_is():
    assert violations_publish(GGML_SYCL_CPP) == []


@pytest.mark.parametrize("mutation", ["no-tier", "no-wording", "no-cause"])
def test_publish_has_a_mutation_witness(mutation):
    src = GGML_SYCL_CPP
    start = src.find(PUBLISH_START)
    end = src.find(PUBLISH_END, start)
    seg = src[start:end]
    if mutation == "no-tier":
        seg2 = seg.replace("ggml_sycl::alloc_tier::HOST_PINNED", "ggml_sycl::alloc_tier::DEVICE_VRAM")
    elif mutation == "no-wording":
        seg2 = seg.replace("host-pinned fallback", "fallback")
    else:
        seg2 = seg.replace("not aligned", "wrong").replace("is aligned only to", "is")
    assert seg2 != seg, f"mutation {mutation} did not change the segment"
    mutated = src[:start] + seg2 + src[end:]
    assert violations_publish(mutated), f"mutation {mutation} was not witnessed"


# ---------------------------------------------------------------------------
# (E) no SYCL refusal text suggests a smaller context
# ---------------------------------------------------------------------------

# Owner ruling: never shrink the context; place KV. A refusal names -ub, the VRAM budget or KV placement. A FACT about
# the largest context that fits ("Largest all-VRAM context is about -c N") is not a remedy and stays; a REMEDY that
# asks for a smaller -c / n_ctx / context does not. "a smaller -ub is not a smaller context" is the negation.
_STRING_RE = re.compile(r'"(?:\\.|[^"\\\n])*"')
_SMALLER_CONTEXT_RE = re.compile(
    r"(?<!not a )\b(?:smaller|reduce|reducing|lower|lowering|shrink|decrease)\s+(?:the\s+)?(?:-c\b|n_ctx|context)"
    r"|-ub\s+or\s+-c\b|\bor\s+reduce\s+-c",
    flags=re.IGNORECASE,
)


def violations_smaller_context(name: str, src: str) -> list[str]:
    found: list[str] = []
    for lit in _STRING_RE.findall(strip_comments(src)):
        m = _SMALLER_CONTEXT_RE.search(lit)
        if m:
            found.append(f"{name}: a refusal text suggests a smaller context ({m.group(0)!r}) in {lit[:100]}")
    return found


def test_no_sycl_refusal_text_suggests_a_smaller_context():
    assert violations_smaller_context("ggml-sycl.cpp", GGML_SYCL_CPP) == []
    assert violations_smaller_context("llama-context.cpp", LLAMA_CONTEXT_CPP) == []


@pytest.mark.parametrize(
    "literal",
    [
        '"or free VRAM on this card (another process, or a smaller -c) before loading\\n"',
        '"a smaller -ub or -c keeps those buffers in the zone\\n"',
        '"pass -fa 1/auto to use flash attention, or reduce -c/-p%s\\n"',
        '"no -ub is known to fit: free VRAM on the card, or pass a smaller -c"',
        '"lower n_ctx"',
        '"use a smaller context"',
    ],
)
def test_smaller_context_gate_has_a_witness(literal):
    assert violations_smaller_context("x", "const char * m = " + literal + ";"), literal


@pytest.mark.parametrize(
    "literal",
    [
        '"Largest all-VRAM context is about -c 34304."',
        '"the largest context that fits is about -c %u\\n"',
        '"a smaller -ub is not a smaller context"',
        '"a smaller -ub keeps those buffers in the zone\\n"',
    ],
)
def test_smaller_context_gate_allows_facts_and_negations(literal):
    assert violations_smaller_context("x", "const char * m = " + literal + ";") == [], literal


def test_the_nonfa_remedy_comment_names_flash_attention_only():
    assert "Flash attention or a smaller context" not in GGML_SYCL_CPP
    nonfa = (ROOT / "tests/test-sycl-nonfa-attn-scratch-guard-source.py").read_text()
    assert "or a smaller -c are" not in nonfa


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
