"""Source gate for the setter classification (zhcn design gate 16, section 2.3).

Every `llama_context::set_*` definition in src/llama-context.cpp is classified in
exactly one row of the table below, and the row's contract is pinned on the
setter's comment-stripped body, so a new setter cannot slip in unclassified and a
classified one cannot drift out of its row:

- RESERVING: the setter changes graph shape and sets `sched_need_reserve`, so the
  next decode re-measures (a candidate that fits the published slots is reused in
  place). `set_warmup` is one too, fork-local and in both directions, but only on
  a change: an unchanged value returns before it writes or flags anything.
- MEASURED: the setter changes graph shape but both of its values are in the
  measured set, so it does not reserve (reserving per toggle would republish per
  batch). The measure must write the fields it covers from the graph set:
  embeddings, warmup and, under `n_layer_nextn > 0`, the nextn flags and offset.
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
    "set_warmup": RESERVING,
    "set_embeddings": MEASURED,
    "set_embeddings_nextn": MEASURED,
    "set_nextn_layer_offset": MEASURED,
    "set_n_threads": NONE,
    "set_abort_callback": NONE,
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


def contract_ok(raw: str, table=CLASSIFICATION) -> bool:
    for name, kind in table.items():
        b = body_of(raw, name)
        if kind == RESERVING:
            if RESERVE_FLAG not in b:
                return False
        else:
            if "sched_need_reserve" in b:
                return False
    return True


def warmup_ok(raw: str) -> bool:
    """Reserving only on a change, in both directions: the early return on an unchanged value precedes
    the write, and the flag follows it."""
    b = body_of(raw, "set_warmup")
    early = z("if (cparams.warmup == value) { return; }")
    write = z("cparams.warmup = value;")
    if b.count(early) != 1 or b.count(write) != 1 or b.count(RESERVE_FLAG) != 1:
        return False
    return b.index(early) < b.index(write) < b.index(RESERVE_FLAG)


def measure_covers_ok(raw: str) -> bool:
    """The measure writes, per graph of the set, the fields the MEASURED setters own."""
    code = code_of(raw)
    start = code.find(z("sched_reserve_result llama_context::sched_measure_impl(sched_reserve_state & state)") + "{")
    assert start != -1
    b = code[start:]
    needed = [
        "state.cparams.embeddings = g.embeddings;",
        "state.cparams.warmup = g.warmup;",
        "if (model.hparams.n_layer_nextn > 0) { state.cparams.embeddings_nextn = g.nextn; "
        "state.cparams.embeddings_nextn_masked = g.nextn_masked; state.cparams.nextn_layer_offset = g.nextn_offset; }",
    ]
    return all(z(n) in b for n in needed)


def test_every_setter_is_classified():
    assert classification_ok(CONTEXT_CPP), (
        "llama_context::set_* definitions must equal the classification table exactly: "
        f"found {sorted(setters(CONTEXT_CPP))}"
    )


def test_classification_has_eleven_setters():
    assert len(CLASSIFICATION) == 11


def test_each_row_holds_its_contract():
    assert contract_ok(CONTEXT_CPP)


def test_warmup_reserves_in_both_directions_on_change_only():
    assert warmup_ok(CONTEXT_CPP)


def test_measure_covers_the_measured_setters():
    assert measure_covers_ok(CONTEXT_CPP)


def test_classification_mutants():
    # a new setter nobody classified
    added = CONTEXT_CPP + "\nvoid llama_context::set_new_thing(bool value) {\n    cparams.embeddings = value;\n}\n"
    assert not classification_ok(added), "mutant 'an unclassified setter' slipped through"
    # a classified setter removed from the table is the same hole from the other side
    short = dict(CLASSIFICATION)
    del short["set_warmup"]
    assert not classification_ok(CONTEXT_CPP, short), "mutant 'a setter missing from the table' slipped through"
    # a table row for a setter that does not exist
    extra = dict(CLASSIFICATION)
    extra["set_ghost"] = NONE
    assert not classification_ok(CONTEXT_CPP, extra), "mutant 'a ghost row' slipped through"


def test_contract_mutants():
    for name in ("set_causal_attn", "set_adapter_cvec", "set_embeddings_layer_inp", "set_warmup"):
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


def test_warmup_mutants():
    code = strip_comments(CONTEXT_CPP)
    assert warmup_ok(code)
    # the flag dropped
    no_flag = re.sub(r"(void llama_context::set_warmup.*?)\n\s*sched_need_reserve = true;", r"\1", code, count=1, flags=re.DOTALL)
    assert no_flag != code and not warmup_ok(no_flag), "mutant 'warmup does not reserve' slipped through"
    # reserving on every call, not only on a change
    every = code.replace("    if (cparams.warmup == value) {\n        return;\n    }\n\n", "", 1)
    assert every != code and not warmup_ok(every), "mutant 'warmup reserves on every call' slipped through"
    # the flag set before the write
    early_flag = code.replace(
        "    cparams.warmup = value;\n", "    sched_need_reserve = true;\n    cparams.warmup = value;\n", 1
    )
    early_flag = re.sub(r"(cparams\.warmup = value;)\n\s*sched_need_reserve = true;", r"\1", early_flag, count=1)
    assert early_flag != code and not warmup_ok(early_flag), "mutant 'warmup flags before the write' slipped through"


def test_measure_cover_mutants():
    code = CONTEXT_CPP
    for old in (
        "state.cparams.embeddings = g.embeddings;",
        "state.cparams.warmup     = g.warmup;",
        "state.cparams.nextn_layer_offset      = g.nextn_offset;",
    ):
        assert old in code, old
        assert not measure_covers_ok(code.replace(old, "", 1)), f"mutant 'measure drops {old}' slipped through"
