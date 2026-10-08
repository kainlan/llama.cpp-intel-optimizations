"""Source gate for the auto n_ubatch trial's hoisted block (llama.cpp-7gno, zhcn design 2.7 step 1).

The planned residency fixpoint needs the ladder's rung set before the memory module exists, so the trial's decisions
move out of `sycl_select_auto_ubatch` into `sycl_auto_ubatch_prepare`, which the constructor runs before
`model.create_memory(`. The move is behaviour-neutral, and this gate pins, on comment-stripped text, what keeps it so:

- the constructor decides the trial, runs `sycl_auto_ubatch_prepare` only when the trial runs, before the memory module,
  and drops the prep after the reserve it was made for;
- the hoisted block owns the cap (the one `llama_auto_ubatch_cap` call), the one tuning-cache lookup, the rung set and its
  hand-over to the host hold; the ladder half carries none of them and reads them from the prep;
- the hoisted block prints nothing, so every `[SYCL-PLAN]` line is still emitted by the ladder half at the site and in the
  text it had (the literals are pinned there, one site each);
- every condition that made the trial take its single reserve leaves the prep empty, and an empty prep takes exactly that
  single reserve, before the ladder and before any WARN;
- the prep is stored once, by the last statement of the hoisted block.

Every clause has a mutant that must fail it. Host-only; collected by pytest.
"""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CTX_CPP = (ROOT / "src/llama-context.cpp").read_text()
CTX_H = (ROOT / "src/llama-context.h").read_text()

_spec = importlib.util.spec_from_file_location("reserve_state_gate", ROOT / "tests/test-sycl-reserve-state-source.py")
_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gate)

z = _gate.z
code_of = _gate.code_of
mutate = _gate.mutate
function_body = _gate.function_body

_CTOR_HEAD = (
    "llama_context::llama_context(const llama_model & model, llama_context_params params, "
    "llama_measure_context_args * measure) :"
)
_CTOR_END = "llama_context::~llama_context()"
_PREPARE = "void llama_context::sycl_auto_ubatch_prepare(ggml_type type_k, ggml_type type_v)"
_SELECT = "void llama_context::sycl_select_auto_ubatch()"

_SYCL_PLAN_LITERALS = [
    '"[SYCL-PLAN] tuning cache %s: n_ubatch=%u (%s)\\n"',
    '"[SYCL-PLAN] auto n_ubatch=%u for n_ctx=%u n_batch=%u (tried %s; %s); pass -ub N to override\\n"',
    '"%s: n_ubatch = %u (auto, was %u)\\n"',
    '"[SYCL-PLAN] tuning cache store failed: %s\\n"',
]


def ctor_of(code: str) -> str:
    start = code.index(z(_CTOR_HEAD))
    return code[start : code.index(z(_CTOR_END), start)]


def order_ok(code: str) -> bool:
    ctor = ctor_of(code)
    needles = [
        "llama_auto_ubatch_trial_runs(",
        "sycl_auto_ubatch_prepare(params.type_k, params.type_v);",
        "model.create_memory(",
        "sycl_select_auto_ubatch();",
        "auto_ubatch_prep.reset();",
    ]
    pos = []
    for n in needles:
        if ctor.count(z(n)) != 1:
            return False
        pos.append(ctor.index(z(n)))
    if pos != sorted(pos):
        return False
    # the hoisted block runs only when the trial does
    return z("if (sycl_auto_ubatch_trial) { sycl_auto_ubatch_prepare(params.type_k, params.type_v); }") in ctor


def ownership_ok(code: str) -> bool:
    prep = function_body(code, _PREPARE)
    sel = function_body(code, _SELECT)
    owned = [
        "llama_auto_ubatch_cap(",
        "cache_lookup_fn(",
        "llama_auto_ubatch_rung_set(",
        "llama_auto_ubatch_ladder_members(",
        "tenant_rung_set.assign(",
        "llama_auto_ubatch_cached_valid(",
    ]
    for n in owned:
        if prep.count(z(n)) != 1 or sel.count(z(n)) != 0:
            return False
    # the floor of the rung set is the prep's too: the ladder reads it back, it does not read cparams.n_ubatch a second time
    # (the residency fixpoint runs between the two)
    if z("fallback_ubatch = cparams.n_ubatch") in sel or prep.count(z("prep->fallback_ubatch = fallback_ubatch;")) != 1:
        return False
    # the ladder half reads them from the prep
    return all(
        z(read) in sel
        for read in (
            "const sycl_auto_ubatch_prep & prep = *auto_ubatch_prep;",
            "const uint32_t cap = prep.cap;",
            "const uint32_t fallback_ubatch = prep.fallback_ubatch;",
            "const bool moe_bound = prep.moe_bound;",
            "const bool cache_usable = prep.cache_usable;",
            "const std::vector<uint32_t> & rung_ladder = prep.rung_ladder;",
        )
    )


def silent_ok(code: str) -> bool:
    prep = function_body(code, _PREPARE)
    if re.search(r"LLAMA_LOG_|GGML_LOG_|fprintf|printf\(", prep):
        return False
    sel = function_body(code, _SELECT)
    return all(code.count(z(lit)) == 1 and z(lit) in sel for lit in _SYCL_PLAN_LITERALS)


def exits_ok(code: str) -> bool:
    prep = function_body(code, _PREPARE)
    sel = function_body(code, _SELECT)
    # the four single-reserve conditions return with the prep empty; none of them reserves
    if "sched_reserve(" in prep:
        return False
    for cond in (
        "if (sycl_backends.empty()) {",
        "if (!probe_fn || !fallback_fn) {",
        "if (cap < ladder[0]) {",
        "if (owner.model_id == 0 || owner.load_txn_id == 0) {",
    ):
        at = prep.find(z(cond))
        if at == -1 or not prep[at + len(z(cond)) :].startswith("return;}"):
            return False
    # an empty prep takes the one reserve, first thing after the validated-ub reset
    head = sel[sel.index("{") + 1 :]
    return head.startswith(
        z("sycl_hold_spill_validated_ub = 0;")
        + "#ifdefined(GGML_USE_SYCL)||defined(GGML_BACKEND_DL)"
        + z("if (!auto_ubatch_prep) { sched_reserve(); return; }")
    ) or z("if (!auto_ubatch_prep) { sched_reserve(); return; }") in head[:400]


_SINGLE_RESERVE_EXITS = (
    "if (sycl_backends.empty()) {",
    "if (!probe_fn || !fallback_fn) {",
    "if (cap < ladder[0]) {",
    "if (owner.model_id == 0 || owner.load_txn_id == 0) {",
)


def lookup_after_exits_ok(code: str) -> bool:
    """The tuning-cache lookup is made nowhere on a single-reserve exit: every exit condition precedes it. And the
    hoisted block reads no memory-module state, because the memory module does not exist yet when it runs."""
    prep = function_body(code, _PREPARE)
    lookup = prep.find(z("cache_lookup_fn(&cache_key,"))
    if lookup == -1:
        return False
    for cond in _SINGLE_RESERVE_EXITS:
        at = prep.find(z(cond))
        if at == -1 or at > lookup:
            return False
    return re.search(r"\bmemory\b", prep) is None


# `cache_devices` is written by the hoisted block and read by nobody in the ladder half: the cache key points into it
_PREP_KEEPALIVE = {"cache_devices"}


def prep_members_ok(code: str, header: str) -> bool:
    """Every member the hoisted block stores is read by the ladder half (bar the one the key points into), the struct
    declares no member the block never stores, and it is not copyable: the key points into its own cache_devices."""
    prep = function_body(code, _PREPARE)
    sel = function_body(code, _SELECT)
    written = set(re.findall(r"prep->(\w+)", prep))
    read = set(re.findall(r"prep\.(\w+)", sel))
    if written - read != _PREP_KEEPALIVE or read - written:
        return False
    # `header` is code_of(llama-context.h); the struct runs to the unique_ptr member declared right after it
    struct = header[header.index(z("struct sycl_auto_ubatch_prep {")) :]
    body = struct[: struct.index(z("std::unique_ptr<sycl_auto_ubatch_prep> auto_ubatch_prep;"))]
    # every declared member, by name: the statements that are not the struct's own constructor and copy operations,
    # cut at the initialiser, the brace or the array bound, name last
    declared = set()
    for stmt in body[body.index("{") + 1 :].split(";"):
        if stmt in ("", "}") or stmt.startswith("sycl_auto_ubatch_prep"):
            continue
        stmt = re.split(r"[=\[{]", stmt, maxsplit=1)[0]
        name = re.search(r"(\w+)$", stmt)
        if name is None:
            return False
        declared.add(name.group(1))
    # each is stored by the hoisted block, so none is declared and never filled
    if declared != written:
        return False
    return (
        z("sycl_auto_ubatch_prep(const sycl_auto_ubatch_prep &) = delete;") in body
        and z("sycl_auto_ubatch_prep & operator=(const sycl_auto_ubatch_prep &) = delete;") in body
    )


def stored_once_ok(code: str) -> bool:
    prep = function_body(code, _PREPARE)
    store = z("auto_ubatch_prep = std::move(prep);")
    if prep.count(store) != 1:
        return False
    # the last statement of the SYCL arm: nothing but the closing of the arm follows it
    tail = prep[prep.index(store) + len(store) :]
    if not re.fullmatch(r"#else\(void\)type_k;\(void\)type_v;#endif\}|#else.*#endif\}", tail.replace("GGML_UNUSED(type_k);", "(void)type_k;").replace("GGML_UNUSED(type_v);", "(void)type_v;")):
        return False
    return len(re.findall(r"(?<![\w.>])auto_ubatch_prep\s*=\s*std::move", code)) == 1


def test_constructor_runs_the_block_before_the_memory():
    assert order_ok(code_of(CTX_CPP))


def test_the_block_owns_the_decisions():
    assert ownership_ok(code_of(CTX_CPP))


def test_the_block_prints_nothing_and_the_plan_lines_stay_in_the_ladder():
    assert silent_ok(code_of(CTX_CPP))


def test_every_single_reserve_condition_leaves_the_prep_empty():
    assert exits_ok(code_of(CTX_CPP))


def test_the_lookup_is_after_every_single_reserve_exit():
    assert lookup_after_exits_ok(code_of(CTX_CPP))


def test_the_prep_has_no_dead_member_and_is_not_copyable():
    assert prep_members_ok(code_of(CTX_CPP), code_of(CTX_H))


def test_the_prep_is_stored_once_last():
    assert stored_once_ok(code_of(CTX_CPP))


def test_mutants():
    code = code_of(CTX_CPP)
    ctor = ctor_of(code)
    prep = function_body(code, _PREPARE)
    sel = function_body(code, _SELECT)

    def with_ctor(new: str) -> str:
        return code.replace(ctor, new, 1)

    def with_prep(new: str) -> str:
        return code.replace(prep, new, 1)

    def with_sel(new: str) -> str:
        return code.replace(sel, new, 1)

    # order
    assert not order_ok(with_ctor(mutate(ctor, "if (sycl_auto_ubatch_trial) { sycl_auto_ubatch_prepare(params.type_k, params.type_v); }", "sycl_auto_ubatch_prepare(params.type_k, params.type_v);"))), \
        "mutant 'the block runs when the trial does not' slipped through"
    late = mutate(ctor, "if (sycl_auto_ubatch_trial) { sycl_auto_ubatch_prepare(params.type_k, params.type_v); }", "")
    late = late.replace(z("if (sycl_auto_ubatch_trial) { sycl_select_auto_ubatch();"),
                        z("if (sycl_auto_ubatch_trial) { sycl_auto_ubatch_prepare(params.type_k, params.type_v); sycl_select_auto_ubatch();"), 1)
    assert not order_ok(with_ctor(late)), "mutant 'the block runs after the memory module' slipped through"
    assert not order_ok(with_ctor(mutate(ctor, "auto_ubatch_prep.reset();", ""))), "mutant 'the prep is kept' slipped through"

    # ownership: a second lookup / a second cap in the ladder half, the set not handed over
    assert not ownership_ok(with_sel(sel.replace(z("const uint32_t cap = prep.cap;"),
                                                 z("const uint32_t cap = llama_auto_ubatch_cap(cparams.n_batch, cparams.n_ctx, model.hparams.n_expert, 0, false, nullptr);"), 1))), \
        "mutant 'the ladder computes its own cap' slipped through"
    assert not ownership_ok(with_sel(sel.replace(z("if (cache_available) {"), z("cache_lookup_fn(&cache_key, nullptr, nullptr, 0); if (cache_available) {"), 1))), \
        "mutant 'a second cache lookup in the ladder' slipped through"
    assert not ownership_ok(with_prep(mutate(prep, "tenant_rung_set.assign(rung_set, rung_set + n_rung_set);", ""))), \
        "mutant 'the rung set is not handed to the hold' slipped through"
    assert not ownership_ok(with_sel(mutate(sel, "const std::vector<uint32_t> & rung_ladder = prep.rung_ladder;", "std::vector<uint32_t> rung_ladder;"))), \
        "mutant 'the ladder has no rungs' slipped through"

    # silence
    assert not silent_ok(with_prep(prep.replace(z("auto_ubatch_prep.reset();"), z('auto_ubatch_prep.reset(); LLAMA_LOG_INFO("x");'), 1))), \
        "mutant 'the hoisted block logs' slipped through"
    lit = z(_SYCL_PLAN_LITERALS[0])
    assert not silent_ok(code.replace(lit, lit.replace("n_ubatch=", "ubatch=", 1), 1)), \
        "mutant 'the tuning-cache line is reworded' slipped through"
    assert not silent_ok(with_prep(prep.replace(z("auto_ubatch_prep.reset();"), z("auto_ubatch_prep.reset();") + lit, 1))), \
        "mutant 'a plan line moved into the hoisted block' slipped through"

    # exits
    assert not exits_ok(with_prep(mutate(prep, "if (cap < ladder[0]) {", "if (cap < ladder[0]) { sched_reserve();"))), \
        "mutant 'the cap exit reserves from the hoisted block' slipped through"
    assert not exits_ok(with_prep(prep.replace(z("if (owner.model_id == 0 || owner.load_txn_id == 0) { return; }"), "", 1))), \
        "mutant 'the zero-token exit is gone' slipped through"
    assert not exits_ok(with_sel(sel.replace(z("if (!auto_ubatch_prep) { sched_reserve(); return; }"), "", 1))), \
        "mutant 'an empty prep falls into the ladder' slipped through"

    # stored once, last
    assert not stored_once_ok(with_prep(prep.replace(z("auto_ubatch_prep = std::move(prep);"), "", 1))), \
        "mutant 'the prep is never stored' slipped through"
    assert not stored_once_ok(with_prep(prep.replace(z("#else"), z("auto_ubatch_prep = std::move(prep); #else"), 1))), \
        "mutant 'the prep is stored twice' slipped through"

    # the lookup comes after every single-reserve exit, and the block reads no memory state
    def after_lookup(guard: str) -> str:
        g = z(guard)
        moved = prep.replace(g, "", 1)
        assert moved != prep, guard
        marker = z("const uint32_t cache_set_value")
        assert moved.count(marker) == 1
        return moved.replace(marker, g + marker, 1)

    assert not lookup_after_exits_ok(with_prep(after_lookup("if (owner.model_id == 0 || owner.load_txn_id == 0) { return; }"))), \
        "mutant 'the zero-token exit moved after the lookup' slipped through"
    assert not lookup_after_exits_ok(with_prep(after_lookup("if (cap < ladder[0]) { return; }"))), \
        "mutant 'the cap exit moved after the lookup' slipped through"
    assert not lookup_after_exits_ok(with_prep(after_lookup("if (sycl_backends.empty()) { return; }"))), \
        "mutant 'the no-backend exit moved after the lookup' slipped through"
    assert not lookup_after_exits_ok(with_prep(mutate(prep, "const bool cache_available = ", "const bool cache_available = memory != nullptr && "))), \
        "mutant 'the hoisted block reads the memory module' slipped through"

    # the floor is read once, from the prep
    assert not ownership_ok(with_sel(sel.replace(z("const uint32_t fallback_ubatch = prep.fallback_ubatch;"),
                                                 z("const uint32_t fallback_ubatch = cparams.n_ubatch;"), 1))), \
        "mutant 'the ladder re-reads cparams.n_ubatch' slipped through"

    # no dead member, no copy
    header = code_of(CTX_H)
    assert not prep_members_ok(with_prep(mutate(prep, "prep->cap = cap;", "prep->cap = cap; prep->cache_lookup_fn = cache_lookup_fn;")), header), \
        "mutant 'a member the ladder never reads is stored' slipped through"
    assert not prep_members_ok(code, header.replace(
        z("sycl_auto_ubatch_prep(const sycl_auto_ubatch_prep &) = delete;"), "", 1)), \
        "mutant 'the prep is copy-constructible' slipped through"
    assert not prep_members_ok(code, header.replace(
        z("sycl_auto_ubatch_prep & operator=(const sycl_auto_ubatch_prep &) = delete;"), "", 1)), \
        "mutant 'the prep is copy-assignable' slipped through"
    assert not prep_members_ok(code, header.replace(z("cache_store_fn = nullptr;"), z("cache_store_fn = nullptr; void * cache_lookup_fn = nullptr;"), 1)), \
        "mutant 'the prep declares the dead lookup member again' slipped through"

    # a declared member the block never stores, of any name, and one it stores that nothing reads
    assert not prep_members_ok(code, header.replace(z("cache_store_fn = nullptr;"), z("cache_store_fn = nullptr; bool dead_member = false;"), 1)), \
        "mutant 'a member is declared and never stored or read' slipped through"
    assert not prep_members_ok(
        with_prep(mutate(prep, "prep->cap = cap;", "prep->cap = cap; prep->dead_member = true;")),
        header.replace(z("cache_store_fn = nullptr;"), z("cache_store_fn = nullptr; bool dead_member = false;"), 1),
    ), "mutant 'a member is stored and never read' slipped through"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
