"""Source gate for the planned auto-ubatch ladder (zhcn design, gate 34 part).

The trial `llama_context::sycl_select_auto_ubatch()` and the constructor's
trial decision must take their inputs from the pure helpers in
src/llama-auto-ubatch.h, so the constructor, the ladder and anything that
plans host room for the ladder's rungs cannot compute two different answers:

- the constructor decides the trial with llama_auto_ubatch_trial_runs(), fed
  the four evaluated conditions, never constants;
- the trial's `cap` is written exactly once, by its initializer
  `llama_auto_ubatch_cap(...)`, with no inline std::min left beside it;
- the ladder loop iterates only the ladder members of
  llama_auto_ubatch_rung_set(), so a rung is a candidate only if it is in
  the set; it no longer carries its own `c > cap` / `c < fallback_ubatch`
  checks, because the set excludes those rungs by construction;
- the tuning cache is looked up once, and its value feeds the set;
- the observable log text is pinned byte for byte, since the hardware rows
  grep it.

Checks run on comment-stripped text. Every check has a mutant that must fail
it. Host-only; collected by pytest (llama_test_pytest).
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTEXT_CPP = (ROOT / "src/llama-context.cpp").read_text()
AUTO_UBATCH_H = (ROOT / "src/llama-auto-ubatch.h").read_text()

_LEXEME_RE = re.compile(
    r'"(?:\\.|[^"\\\n])*"'
    r"|'(?:\\.|[^'\\\n])*'"
    r"|//[^\n]*"
    r"|/\*.*?\*/",
    flags=re.DOTALL,
)


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def z(text: str) -> str:
    """Canonical spelling: whitespace runs collapse, then every space that is
    not between two identifier characters goes, so a call re-wrapped by
    clang-format matches however it is broken across lines while `const cap`
    stays two words."""
    return re.sub(r"(?<!\w) | (?!\w)", "", re.sub(r"\s+", " ", text))


_TRIAL_START = z("void llama_context::sycl_select_auto_ubatch(ggml_type type_k, ggml_type type_v) {")
_TRIAL_END = z("static int llama_graph_n_input_tensors(ggml_cgraph * gf, bool log) {")
_CTOR_START = z("bool sycl_auto_ubatch_trial = false;")
_CTOR_END = z("if (!cparams.flash_attn) {")


def bounded(raw: str, start: str, end: str) -> str:
    code = z(strip_comments(raw))
    a = code.find(start)
    assert a != -1, f"{start!r} not found"
    b = code.find(end, a + 1)
    assert b != -1, f"{end!r} not found after {start!r}"
    return code[a:b]


def trial(raw: str = CONTEXT_CPP) -> str:
    return bounded(raw, _TRIAL_START, _TRIAL_END)


def ctor(raw: str = CONTEXT_CPP) -> str:
    return bounded(raw, _CTOR_START, _CTOR_END)


def mutate(body: str, old: str, new: str) -> str:
    """Mutate a whitespace-free body; the anchor must occur exactly once."""
    zo = z(old)
    assert body.count(zo) == 1, f"mutant anchor must match exactly once: {old!r} x{body.count(zo)}"
    return body.replace(zo, z(new), 1)


# --- the constructor decision ------------------------------------------------


def ctor_ok(b: str) -> bool:
    # The call: four arguments, the last two evaluated names, not constants.
    if z("sycl_auto_ubatch_trial = llama_auto_ubatch_trial_runs(params.n_ubatch_auto, cparams.causal_attn, "
         "sycl_backend_present, sycl_auto_ubatch_enabled);") not in b:
        return False
    if z("const bool sycl_backend_present = llama_context_has_sycl_backend(backends);") not in b:
        return False
    if z("bool sycl_auto_ubatch_enabled = false;") not in b:
        return False
    # The accessor is consulted only once the cheap conditions hold.
    gate = z("if (params.n_ubatch_auto && cparams.causal_attn && sycl_backend_present) {")
    at = b.find(gate)
    if at == -1:
        return False
    return "ggml_backend_sycl_auto_ubatch_enabled" in b[at:] and "sycl_auto_ubatch_enabled=" in b[at:]


def test_constructor_decides_with_trial_runs():
    assert ctor_ok(ctor()), (
        "the constructor must decide the trial with llama_auto_ubatch_trial_runs(params.n_ubatch_auto, "
        "cparams.causal_attn, sycl_backend_present, sycl_auto_ubatch_enabled), the last two evaluated"
    )


def test_constructor_decision_mutants():
    b = ctor()
    mutants = {
        "constant for the backend input": ("sycl_backend_present, sycl_auto_ubatch_enabled);", "true, sycl_auto_ubatch_enabled);"),
        "constant for the enabled input": ("cparams.causal_attn, sycl_backend_present, sycl_auto_ubatch_enabled);", "cparams.causal_attn, sycl_backend_present, true);"),
        "constant for the causal input": ("params.n_ubatch_auto, cparams.causal_attn, sycl_backend_present", "params.n_ubatch_auto, true, sycl_backend_present"),
        "the backend input not evaluated": ("const bool sycl_backend_present = llama_context_has_sycl_backend(backends);", "const bool sycl_backend_present = true;"),
        "the enabled accessor consulted before the cheap gate": ("if (params.n_ubatch_auto && cparams.causal_attn && sycl_backend_present) {", "if (params.n_ubatch_auto) {"),
        "the call replaced by a bare conjunction": ("sycl_auto_ubatch_trial = llama_auto_ubatch_trial_runs(", "sycl_auto_ubatch_trial = sycl_backend_present && ("),
    }
    for name, (old, new) in mutants.items():
        assert not ctor_ok(mutate(b, old, new)), f"mutant {name!r} slipped through the constructor-decision gate"


# --- cap ---------------------------------------------------------------------

_CAP_WRITE_RE = re.compile(r"(?<![\w.>])cap(?:=(?!=)|[-+*/|&^]=|<<=|>>=|\+\+|--)|(?:\+\+|--)cap\b")


def cap_ok(b: str) -> bool:
    init = z("const uint32_t cap = llama_auto_ubatch_cap(cparams.n_batch, cparams.n_ctx, model.hparams.n_expert, "
             "moe_cap, moe_cap_available, &moe_bound);")
    if init not in b:
        return False
    # exactly one write to `cap`, and it is the initializer
    if len(_CAP_WRITE_RE.findall(b)) != 1:
        return False
    # no inline min left beside the helper
    return "std::min(cparams.n_batch" not in b


def test_cap_is_written_once_by_the_helper():
    assert cap_ok(trial()), (
        "sycl_select_auto_ubatch's cap must be written exactly once, by its initializer "
        "`= llama_auto_ubatch_cap(cparams.n_batch, cparams.n_ctx, model.hparams.n_expert, moe_cap, "
        "moe_cap_available, &moe_bound)`, with no inline std::min"
    )


def test_cap_mutants():
    b = trial()
    mutants = {
        "second write": ("if (cap < ladder[0]) {", "cap = std::min(cap, 4096u); if (cap < ladder[0]) {"),
        "second write by compound assignment": ("if (cap < ladder[0]) {", "cap -= 1; if (cap < ladder[0]) {"),
        "second write by increment": ("if (cap < ladder[0]) {", "++cap; if (cap < ladder[0]) {"),
        "reordered cap arguments": ("llama_auto_ubatch_cap(cparams.n_batch, cparams.n_ctx,", "llama_auto_ubatch_cap(cparams.n_ctx, cparams.n_batch,"),
        "inline cap beside the helper": ("const uint32_t cap = llama_auto_ubatch_cap(", "const uint32_t cap_unused = std::min(cparams.n_batch, cparams.n_ctx); const uint32_t cap = llama_auto_ubatch_cap("),
        "plain min instead of the helper": ("const uint32_t cap = llama_auto_ubatch_cap(cparams.n_batch, cparams.n_ctx, model.hparams.n_expert, moe_cap, moe_cap_available, &moe_bound);", "const uint32_t cap = std::min(cparams.n_batch, cparams.n_ctx);"),
        "ceiling availability dropped": ("moe_cap, moe_cap_available, &moe_bound);", "moe_cap, true, &moe_bound);"),
    }
    for name, (old, new) in mutants.items():
        assert not cap_ok(mutate(b, old, new)), f"mutant {name!r} slipped through the cap gate"


# --- the ladder loop ---------------------------------------------------------

_SET_CALL = z("llama_auto_ubatch_rung_set(ladder, llama_auto_ubatch_ladder_size, fallback_ubatch, cap, "
              "cache_set_value, rung_set, llama_auto_ubatch_rung_set_capacity)")
_MEMBERS_CALL = z("llama_auto_ubatch_ladder_members(rung_set, n_rung_set, ladder, llama_auto_ubatch_ladder_size, "
                  "rungs, llama_auto_ubatch_rung_set_capacity)")
_LOOP_HEAD = z("for (uint32_t c : rung_ladder) {")


def loop_of(b: str) -> str:
    a = b.find(_LOOP_HEAD)
    if a == -1:
        return ""
    e = b.find(z("if (last_good == 0) {"), a)
    return b[a:e] if e != -1 else ""


def ladder_ok(b: str) -> bool:
    if _SET_CALL not in b or _MEMBERS_CALL not in b:
        return False
    if z("std::vector<uint32_t> rung_ladder(rungs, rungs + n_rungs);") not in b:
        return False
    loop = loop_of(b)
    if not loop:
        return False
    # no iteration of the raw ladder anywhere in the trial
    if re.search(r"for\((?:const)?(?:auto|uint32_t)&?\w+:ladder\)", b):
        return False
    # the per-rung bounds stay deleted: the set excludes those rungs
    if z("c > cap") in loop or z("c < fallback_ubatch") in loop:
        return False
    # the resume skip and the stop-on-refusal break stay
    if z("if (c <= cache_resume_above) { continue; }") not in loop:
        return False
    return z("if (reason != nullptr) { stop = reason; break; }") in loop


def test_ladder_iterates_only_the_rung_set_members():
    assert ladder_ok(trial()), (
        "the ladder loop must iterate `rung_ladder`, the ladder members of llama_auto_ubatch_rung_set(), "
        "keep the resume skip and the break on refusal, and carry no `c > cap` / `c < fallback_ubatch`"
    )


def test_ladder_mutants():
    b = trial()
    mutants = {
        "direct iteration of the raw ladder": ("for (uint32_t c : rung_ladder) {", "for (uint32_t c : ladder) {"),
        "direct iteration by reference": ("for (uint32_t c : rung_ladder) {", "for (const auto & c : ladder) {"),
        "break on refusal becomes continue": ("if (reason != nullptr) { stop = reason; break; }", "if (reason != nullptr) { stop = reason; continue; }"),
        "per-rung cap break restored": ("for (uint32_t c : rung_ladder) {", "for (uint32_t c : rung_ladder) { if (c > cap) { break; }"),
        "per-rung floor skip restored": ("for (uint32_t c : rung_ladder) {", "for (uint32_t c : rung_ladder) { if (c < fallback_ubatch) { continue; }"),
        "resume skip dropped": ("if (c <= cache_resume_above) { continue; }", ""),
        "set built over a wrong cap": ("fallback_ubatch, cap, cache_set_value,", "fallback_ubatch, cap + 1, cache_set_value,"),
        "set built without the cached value": ("fallback_ubatch, cap, cache_set_value,", "fallback_ubatch, cap, 0,"),
        "members call dropped": ("llama_auto_ubatch_ladder_members(rung_set,", "llama_auto_ubatch_unrelated(rung_set,"),
        "inline rung copy instead of the helper's members": ("std::vector<uint32_t> rung_ladder(rungs, rungs + n_rungs);", "std::vector<uint32_t> rung_ladder(rung_set, rung_set + n_rung_set);"),
    }
    for name, (old, new) in mutants.items():
        assert not ladder_ok(mutate(b, old, new)), f"mutant {name!r} slipped through the ladder gate"


# --- one cache lookup --------------------------------------------------------


def cache_ok(b: str) -> bool:
    if b.count("cache_lookup_fn(") != 1:
        return False
    return (
        z("const uint32_t cache_set_value = cache_usable ? cached_ubatch : 0;") in b
        and "llama_auto_ubatch_cached_valid(" in b
    )


def test_cache_is_looked_up_once_and_validated_before_the_set():
    assert cache_ok(trial()), (
        "the tuning cache must be looked up exactly once, and its value must pass "
        "llama_auto_ubatch_cached_valid before it feeds the rung set"
    )


def test_cache_mutants():
    b = trial()
    mutants = {
        "second lookup": ("if (cache_available) {", "cache_lookup_fn(&cache_key, &cached_ubatch, cached_reason_buf, sizeof(cached_reason_buf)); if (cache_available) {"),
        "unvalidated value feeds the set": ("const uint32_t cache_set_value = cache_usable ? cached_ubatch : 0;", "const uint32_t cache_set_value = cached_ubatch;"),
        "validation dropped": ("llama_auto_ubatch_cached_valid(", "llama_auto_ubatch_unrelated("),
    }
    for name, (old, new) in mutants.items():
        assert not cache_ok(mutate(b, old, new)), f"mutant {name!r} slipped through the cache gate"


# --- pinned log text ---------------------------------------------------------

_LITERALS = [
    '"[SYCL-PLAN] tuning cache %s: n_ubatch=%u (%s)\\n"',
    '"[SYCL-PLAN] auto n_ubatch=%u for n_ctx=%u n_batch=%u (tried %s; %s); pass -ub N to override\\n"',
    '"%s: n_ubatch = %u (auto, was %u)\\n"',
]


def literals_ok(code: str) -> bool:
    return all(code.count(lit) == 1 for lit in _LITERALS) and all(
        f'"{r}"' in code for r in ("ladder exhausted", "MoE GPU routing ceiling")
    )


def test_log_literals_are_pinned():
    assert literals_ok(strip_comments(CONTEXT_CPP)), (
        "a [SYCL-PLAN] literal or stop reason the hardware rows grep is missing, reworded or duplicated"
    )


def test_log_literal_mutants():
    code = strip_comments(CONTEXT_CPP)
    for lit in _LITERALS:
        assert not literals_ok(code.replace(lit, lit.replace("n_ubatch", "ubatch", 1), 1)), lit
    assert not literals_ok(code.replace('"MoE GPU routing ceiling"', '"MoE ceiling"'))


# --- the helpers' own contract ------------------------------------------------

_HELPERS = (
    "llama_auto_ubatch_trial_runs",
    "llama_auto_ubatch_cap",
    "llama_auto_ubatch_rung_set",
    "llama_auto_ubatch_ladder_members",
    "llama_auto_ubatch_cached_valid",
)


def header_ok(src: str) -> bool:
    code = strip_comments(src)
    for name in _HELPERS:
        if not re.search(rf"inline\s+\w[\w\s*]*\b{name}\s*\(", code):
            return False
    # a helper that touched context or backend state would stop being host-testable
    return "llama_context" not in code and "ggml_backend" not in code


def test_helpers_are_pure_header_inline():
    assert header_ok(AUTO_UBATCH_H)


def test_header_mutants():
    for name in _HELPERS:
        renamed = re.sub(rf"(inline\s+\w[\w\s*]*\b){name}(\s*\()", rf"\1{name}_gone\2", AUTO_UBATCH_H, count=1)
        assert renamed != AUTO_UBATCH_H, name
        assert not header_ok(renamed), name
    assert not header_ok(AUTO_UBATCH_H + "\ninline void f(llama_context *) {}\n")
