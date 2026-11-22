"""Source gate for the setter classification (zhcn design gate 16, section 2.3).

Every `llama_context::set_*` definition in src/llama-context.cpp is classified in
exactly one row of the table below, and the row's contract is pinned on the
setter's comment-stripped body, so a new setter cannot slip in unclassified and a
classified one cannot drift out of its row:

- RESERVING: the setter changes graph shape and sets `sched_need_reserve`, so the
  next decode re-measures (a candidate that fits the published slots is reused in
  place). Three of them (`set_embeddings_layer_inp_dev`, `set_embeddings_nextn_dev`,
  `set_ctx_state_src`, from upstream) reserve only on a change: the comparison with
  the old value guards the flag and precedes the write. Until the 19bc41dbb merge
  the fork-local `set_warmup` carried that on-change pin; upstream removed warmup
  (llama_set_warmup and cparams.warmup), so the pin moved to these three.
- MEASURED: the setter changes graph shape but both of its values are in the
  measured set, so a planned context never reserves for it (reserving per toggle
  would republish per batch). Since the 19bc41dbb merge an unplanned context does:
  upstream re-reserves on a change because pooling NONE crashed ggml_gallocr on the
  first /completion. So `sched_need_reserve` may appear only once, inside an `if`
  whose condition ends `&& !plan_caps` at top level, before the first cparams write.
  The measure must write the fields it covers from the graph set: embeddings and,
  under `n_layer_nextn > 0`, the nextn flags and offset.
- NONE: the setter does not change graph shape and touches no cparams graph
  field.

Every clause has a mutant that must fail it. Host-only; collected by pytest.
"""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTEXT_CPP = (ROOT / "src/llama-context.cpp").read_text()

_spec = importlib.util.spec_from_file_location("reserve_state_gate", ROOT / "tests/test-sycl-reserve-state-source.py")
_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gate)

z = _gate.z
code_of = _gate.code_of
mutate = _gate.mutate
strip_comments = _gate.strip_comments

RESERVING = "reserving"
MEASURED = "measured"
NONE = "none"

CLASSIFICATION = {
    "set_causal_attn": RESERVING,
    "set_sampler": RESERVING,
    "set_adapters_lora": RESERVING,
    "set_adapter_cvec": RESERVING,
    "set_embeddings_layer_inp": RESERVING,
    "set_embeddings_layer_inp_dev": RESERVING,
    "set_embeddings_nextn_dev": RESERVING,
    "set_ctx_state_src": RESERVING,
    "set_embeddings": MEASURED,
    "set_embeddings_nextn": MEASURED,
    "set_nextn_layer_offset": MEASURED,
    "set_n_threads": NONE,
    "set_abort_callback": NONE,
    "set_h_nextn_rows": NONE,
}

# the RESERVING setters that flag only on a change: setter -> the cparams field it compares and writes
ON_CHANGE = {
    "set_embeddings_layer_inp_dev": ("embeddings_layer_inp_dev", "value"),
    "set_embeddings_nextn_dev": ("embeddings_nextn_dev", "value"),
    "set_ctx_state_src": ("ctx_state_src", "ctx_src"),
}

_DEF_RE = re.compile(r"(?m)^[A-Za-z_][\w:<>\s\*&]*?\bllama_context::(set_\w+)\s*\(")

RESERVE_FLAG = z("sched_need_reserve = true;")


def setters(raw: str) -> list:
    return _DEF_RE.findall(strip_comments(raw))


def body_of(raw: str, name: str) -> str:
    code = strip_comments(raw)
    m = re.search(rf"(?m)^[A-Za-z_][\w:<>\s\*&]*?\bllama_context::{name}\s*\(", code)
    assert m, f"{name} not defined"
    i = code.index("{", m.end())
    depth = 0
    for j in range(i, len(code)):
        if code[j] == "{":
            depth += 1
        elif code[j] == "}":
            depth -= 1
            if depth == 0:
                return z(code[i : j + 1])
    raise AssertionError(f"unbalanced braces in {name}")


def classification_ok(raw: str, table=CLASSIFICATION) -> bool:
    found = setters(raw)
    return sorted(found) == sorted(table) and len(set(found)) == len(found)


_UNPLANNED_RESERVE_RE = re.compile(r"if\(([^{}]*)&&!plan_caps\)\{sched_need_reserve=true;\}")
_CPARAMS_WRITE_RE = re.compile(r"cparams\.\w+=(?!=)")


def top_level_or(cond: str) -> bool:
    depth = 0
    for i, c in enumerate(cond):
        depth += c == "("
        depth -= c == ")"
        if depth == 0 and cond.startswith("||", i):
            return True
    return False


def measured_ok(b: str) -> bool:
    """A MEASURED setter reserves, if at all, only for an unplanned context: one flag, inside the `!plan_caps`
    condition (no top-level `||` that would let a planned context through), before the first cparams write."""
    if "sched_need_reserve" not in b:
        return True
    guarded = _UNPLANNED_RESERVE_RE.findall(b)
    if b.count("sched_need_reserve") != 1 or len(guarded) != 1 or top_level_or(guarded[0]):
        return False
    write = _CPARAMS_WRITE_RE.search(b)
    return write is not None and _UNPLANNED_RESERVE_RE.search(b).start() < write.start()


def contract_ok(raw: str, table=CLASSIFICATION) -> bool:
    for name, kind in table.items():
        b = body_of(raw, name)
        if kind == RESERVING:
            if RESERVE_FLAG not in b:
                return False
        elif kind == MEASURED:
            if not measured_ok(b):
                return False
        else:
            if "sched_need_reserve" in b:
                return False
    return True


def on_change_ok(raw: str, name: str) -> bool:
    """Reserving only on a change, in both directions: the flag sits inside the comparison with the old value,
    which precedes the write (a comparison after the write would never see a change)."""
    field, arg = ON_CHANGE[name]
    b = body_of(raw, name)
    guarded = z(f"if (cparams.{field} != {arg}) {{ sched_need_reserve = true; }}")
    write = z(f"cparams.{field} = {arg};")
    if b.count(guarded) != 1 or b.count(write) != 1 or b.count(RESERVE_FLAG) != 1:
        return False
    return b.index(guarded) < b.index(write)


def measure_covers_ok(raw: str) -> bool:
    """The measure writes, per graph of the set, the fields the MEASURED setters own."""
    code = code_of(raw)
    start = code.find(z("sched_reserve_result llama_context::sched_measure_impl(sched_reserve_state & state)") + "{")
    assert start != -1
    b = code[start:]
    needed = [
        "state.cparams.embeddings = g.embeddings;",
        "if (model.hparams.n_layer_nextn > 0) { state.cparams.embeddings_nextn = g.nextn; "
        "state.cparams.embeddings_nextn_masked = g.nextn_masked; state.cparams.nextn_layer_offset = g.nextn_offset; }",
    ]
    return all(z(n) in b for n in needed)


def test_every_setter_is_classified():
    assert classification_ok(CONTEXT_CPP), (
        "llama_context::set_* definitions must equal the classification table exactly: "
        f"found {sorted(setters(CONTEXT_CPP))}"
    )


def test_classification_has_fourteen_setters():
    assert len(CLASSIFICATION) == 14


def test_each_row_holds_its_contract():
    assert contract_ok(CONTEXT_CPP)


def test_on_change_setters_reserve_in_both_directions_on_change_only():
    for name in ON_CHANGE:
        assert on_change_ok(CONTEXT_CPP, name), name


def test_measure_covers_the_measured_setters():
    assert measure_covers_ok(CONTEXT_CPP)


def test_classification_mutants():
    # a new setter nobody classified
    added = CONTEXT_CPP + "\nvoid llama_context::set_new_thing(bool value) {\n    cparams.embeddings = value;\n}\n"
    assert not classification_ok(added), "mutant 'an unclassified setter' slipped through"
    # a classified setter removed from the table is the same hole from the other side
    short = dict(CLASSIFICATION)
    del short["set_ctx_state_src"]
    assert not classification_ok(CONTEXT_CPP, short), "mutant 'a setter missing from the table' slipped through"
    # a table row for a setter that does not exist
    extra = dict(CLASSIFICATION)
    extra["set_ghost"] = NONE
    assert not classification_ok(CONTEXT_CPP, extra), "mutant 'a ghost row' slipped through"


def test_contract_mutants():
    for name in ("set_causal_attn", "set_adapter_cvec", "set_embeddings_layer_inp", *ON_CHANGE):
        b = body_of(CONTEXT_CPP, name)
        assert RESERVE_FLAG in b, f"{name} must reserve"
    # a reserving setter that stopped flagging
    code = strip_comments(CONTEXT_CPP)
    dropped = code.replace("    sched_need_reserve = true;\n\n    return res;", "    return res;", 1)
    assert dropped != code
    assert not contract_ok(dropped), "mutant 'cvec stops reserving' slipped through"
    # a measured setter that starts reserving per toggle
    reserving = code.replace(
        "    cparams.embeddings = value;\n", "    cparams.embeddings = value;\n    sched_need_reserve = true;\n", 1
    )
    assert reserving != code
    assert not contract_ok(reserving), "mutant 'embeddings reserves per toggle' slipped through"
    # a no-op setter that starts reserving
    thread = code.replace(
        "    cparams.n_threads_batch = n_threads_batch;\n",
        "    cparams.n_threads_batch = n_threads_batch;\n    sched_need_reserve = true;\n",
        1,
    )
    assert thread != code
    assert not contract_ok(thread), "mutant 'n_threads reserves' slipped through"


def test_measured_mutants():
    code = strip_comments(CONTEXT_CPP)
    assert contract_ok(code)
    for name, old, new in (
        ("embeddings reserves in a planned context", "if (cparams.embeddings != value && !plan_caps) {", "if (cparams.embeddings != value) {"),
        ("nextn reserves in a planned context",
         "if ((cparams.embeddings_nextn != value || cparams.embeddings_nextn_masked != masked) && !plan_caps) {",
         "if ((cparams.embeddings_nextn != value || cparams.embeddings_nextn_masked != masked)) {"),
        ("the nextn guard loses its parentheses",
         "if ((cparams.embeddings_nextn != value || cparams.embeddings_nextn_masked != masked) && !plan_caps) {",
         "if (cparams.embeddings_nextn != value || cparams.embeddings_nextn_masked != masked && !plan_caps) {"),
        ("embeddings reserves unconditionally",
         "    if (cparams.embeddings != value && !plan_caps) {\n        sched_need_reserve = true;\n    }\n",
         "    sched_need_reserve = true;\n"),
        ("nextn reserves unconditionally",
         "    if ((cparams.embeddings_nextn != value || cparams.embeddings_nextn_masked != masked) && !plan_caps) {\n        sched_need_reserve = true;\n    }\n",
         "    sched_need_reserve = true;\n"),
        ("embeddings flags after its write",
         "    if (cparams.embeddings != value && !plan_caps) {\n        sched_need_reserve = true;\n    }\n\n    cparams.embeddings = value;\n",
         "    cparams.embeddings = value;\n    if (cparams.embeddings != value && !plan_caps) {\n        sched_need_reserve = true;\n    }\n"),
    ):
        assert code.count(old) == 1, f"mutant anchor for {name!r} must match once"
        assert not contract_ok(code.replace(old, new, 1)), f"mutant {name!r} slipped through"


def test_on_change_mutants():
    code = strip_comments(CONTEXT_CPP)
    for name, (field, arg) in ON_CHANGE.items():
        assert on_change_ok(code, name), name
        # every mutant edits this setter's own definition only
        m = re.search(rf"(?m)^[^\n]*\bllama_context::{name}\s*\(", code)
        start, end = m.start(), code.index("\n}\n", m.start())
        seg = code[start:end]
        guarded = re.compile(rf"if \(cparams\.{field} != {arg}\) \{{\s*sched_need_reserve = true;\s*\}}")
        write = f"cparams.{field} = {arg};"
        assert len(guarded.findall(seg)) == 1 and seg.count(write) == 1, f"mutant anchors for {name} must match once"

        def splice(new_seg):
            assert new_seg != seg
            return code[:start] + new_seg + code[end:]

        # the flag dropped
        no_flag = splice(guarded.sub(f"if (cparams.{field} != {arg}) {{ }}", seg, count=1))
        assert not on_change_ok(no_flag, name), f"mutant '{name} does not reserve' slipped through"
        # reserving on every call, not only on a change
        every = splice(guarded.sub("sched_need_reserve = true;", seg, count=1))
        assert not on_change_ok(every, name), f"mutant '{name} reserves on every call' slipped through"
        # the write moved before the comparison, which then never sees a change
        moved_seg = seg.replace(write, "", 1)
        moved_seg = guarded.sub(lambda g: write + "\n    " + g.group(0), moved_seg, count=1)
        assert not on_change_ok(splice(moved_seg), name), f"mutant '{name} writes before it compares' slipped through"


def test_measure_cover_mutants():
    code = CONTEXT_CPP
    for old in (
        "state.cparams.embeddings = g.embeddings;",
        "state.cparams.nextn_layer_offset      = g.nextn_offset;",
    ):
        assert old in code, old
        assert not measure_covers_ok(code.replace(old, "", 1)), f"mutant 'measure drops {old}' slipped through"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
