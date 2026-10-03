#!/usr/bin/env python3
"""Arithmetic gate for scripts/moe-cache-sim.py (llama.cpp-05mh).

The simulator predicts what an adaptive VRAM expert cache would be worth for
this fork from a routing trace (examples/moe-trace), before one is built. A
simulator whose arithmetic drifts yields a confident wrong recommendation, so
every number here is worked by hand on a tiny synthetic trace and pinned:

  * hit rate for a hand-built trace, per (token, expert) pair;
  * the static profile is trained on one trace and scored on another, never on
    the trace it is scored against (the in-sample number is the 'oracle');
  * first-touch fills the first S distinct experts routed and never evicts;
  * adaptive swap follows Strata's rule (generate.cpp adapt()): every R rounds,
    candidates with usage >= min_count, paired best-candidate with
    coldest-victim per layer while gain >= margin, the global top N by gain,
    the victim evicted AT ONCE, the incoming expert resident only after
    land_delay more rounds, usage decayed after each adaptation;
  * slots come from bytes per layer divided by that layer's expert bytes;
  * CPU GiB per token is miss bytes over tokens.

Pytest-style (module-level test_* functions, no __main__ guard); register with
llama_test_pytest. Standard library only, except the GGUF-size test which skips
without numpy/gguf-py, and the C++ cross-check which skips without the binary.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "moe-cache-sim.py"


def load():
    spec = importlib.util.spec_from_file_location("moe_cache_sim", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["moe_cache_sim"] = mod   # dataclasses resolve annotations through it
    spec.loader.exec_module(mod)
    return mod


sim = load()

GIB = 1 << 30


def decode_trace(rows_by_step, layer=0, set_name="t", n_expert=4):
    """rows_by_step: list of one-token rows (list of ids) -> a decode-phase trace."""
    return sim.Trace(
        header={"set": set_name, "id": set_name, "n_expert": n_expert},
        steps=[
            sim.Step(index=i, phase=1, n_tokens=1, layers={layer: [row]})
            for i, row in enumerate(rows_by_step)
        ],
    )


def sizes(layer_bytes, n_expert=4):
    return {"n_expert": n_expert, "layer_bytes": dict(layer_bytes)}


# ---------------------------------------------------------------------------
# hit rate
# ---------------------------------------------------------------------------

def test_static_hit_rate_hand_built():
    t = decode_trace([[0, 1], [0, 2], [0, 1], [3, 1]])
    res = sim.simulate_static(
        resident={0: {0, 1}}, test=[t], layer_bytes={0: GIB}, phase="decode")
    # hits: 2 + 1 + 2 + 1 = 6 of 8 pairs
    assert (res.hits, res.total) == (6, 8)
    assert res.hit_rate == 0.75
    # 2 misses, 1 GiB each, over 4 tokens
    assert res.miss_bytes == 2 * GIB
    assert res.tokens == 4
    assert res.cpu_gib_per_token == 0.5
    assert res.full_gib_per_token == 2.0


def test_profile_trained_on_train_scored_on_test_not_in_sample():
    train = decode_trace([[2, 3], [2, 3], [2, 1]], set_name="train")
    test = decode_trace([[0, 1], [2, 3], [2, 1]], set_name="test")
    profile = sim.build_profile([train], phase="decode")
    assert profile[0][2] == 3 and profile[0][3] == 2 and profile[0][1] == 1
    resident = sim.resident_from_profile(profile, {0: 2})
    assert resident == {0: {2, 3}}
    held_out = sim.simulate_static(resident, [test], {0: GIB}, "decode")
    assert (held_out.hits, held_out.total) == (3, 6)
    # the oracle trains on the test trace itself: counts 1:2, 2:2 -> {1, 2}
    oracle_profile = sim.build_profile([test], phase="decode")
    oracle = sim.simulate_static(
        sim.resident_from_profile(oracle_profile, {0: 2}), [test], {0: GIB}, "decode")
    assert (oracle.hits, oracle.total) == (4, 6)


def test_profile_ties_break_to_lower_expert_id():
    t = decode_trace([[3, 2], [1, 0]])
    profile = sim.build_profile([t], phase="decode")
    assert sim.resident_from_profile(profile, {0: 2}) == {0: {0, 1}}


def test_first_touch_fills_with_first_distinct_and_never_evicts():
    t = decode_trace([[0, 1], [2, 3], [2, 1]])
    res = sim.simulate_first_touch({0: 2}, [t], {0: GIB}, "decode")
    # step 0: both miss and fill {0,1}; step 1: both miss, cache full;
    # step 2: 2 misses, 1 hits
    assert (res.hits, res.total) == (1, 6)


def test_zero_slots_is_all_misses():
    t = decode_trace([[0, 1], [0, 1]])
    res = sim.simulate_static({0: set()}, [t], {0: GIB}, "decode")
    assert (res.hits, res.total) == (0, 4)
    ft = sim.simulate_first_touch({0: 0}, [t], {0: GIB}, "decode")
    assert (ft.hits, ft.total) == (0, 4)


def test_phase_filter_decode_vs_all():
    prefill = sim.Step(index=0, phase=0, n_tokens=2, layers={0: [[0, 1], [0, 1]]})
    decode = sim.Step(index=1, phase=1, n_tokens=1, layers={0: [[2, 3]]})
    t = sim.Trace(header={"set": "t"}, steps=[prefill, decode])
    r_dec = sim.simulate_static({0: {0, 1}}, [t], {0: GIB}, "decode")
    assert (r_dec.hits, r_dec.total, r_dec.tokens) == (0, 2, 1)
    r_all = sim.simulate_static({0: {0, 1}}, [t], {0: GIB}, "all")
    assert (r_all.hits, r_all.total, r_all.tokens) == (4, 6, 3)
    r_pre = sim.simulate_static({0: {0, 1}}, [t], {0: GIB}, "prefill")
    assert (r_pre.hits, r_pre.total, r_pre.tokens) == (4, 4, 2)


def test_tokens_per_step_is_max_over_layers():
    # the last layer of a prefill ubatch can carry only the output rows
    step = sim.Step(index=0, phase=0, n_tokens=3,
                    layers={0: [[0], [0], [0]], 1: [[1]]})
    t = sim.Trace(header={}, steps=[step])
    res = sim.simulate_static({0: {0}, 1: {1}}, [t], {0: GIB, 1: GIB}, "all")
    assert res.tokens == 3
    assert (res.hits, res.total) == (4, 4)


def test_layer_missing_from_sizes_is_refused():
    t = decode_trace([[0]], layer=5)
    with pytest.raises(ValueError, match="layer 5"):
        sim.simulate_static({5: {0}}, [t], {0: GIB}, "decode")


# ---------------------------------------------------------------------------
# adaptive swap
# ---------------------------------------------------------------------------

ADAPT = dict(swap_n=1, every=2, decay=0.5, min_count=2.0, margin=1.5)


def test_adaptive_evicts_at_once_and_admits_after_land_delay():
    t = decode_trace([[1], [1], [1], [1], [1]])
    # after round 1 usage is {1: 2}: swap 1 in, 0 out. 0 leaves at round 2;
    # 1 lands land_delay rounds later.
    delayed = sim.simulate_adaptive({0: {0}}, {0: 1}, [t], {0: GIB}, "decode",
                                    land_delay=1, **ADAPT)
    assert (delayed.hits, delayed.total) == (2, 5)      # rounds 3, 4
    immediate = sim.simulate_adaptive({0: {0}}, {0: 1}, [t], {0: GIB}, "decode",
                                      land_delay=0, **ADAPT)
    assert (immediate.hits, immediate.total) == (3, 5)  # rounds 2, 3, 4
    static = sim.simulate_static({0: {0}}, [t], {0: GIB}, "decode")
    assert static.hits == 0


def test_adaptive_evicted_expert_misses_while_incoming_in_flight():
    t = decode_trace([[1], [1], [0], [1], [1]])
    res = sim.simulate_adaptive({0: {0}}, {0: 1}, [t], {0: GIB}, "decode",
                                land_delay=1, **ADAPT)
    # round 2 asks for 0, evicted since the swap: miss. rounds 3, 4 hit.
    assert (res.hits, res.total) == (2, 5)
    static = sim.simulate_static({0: {0}}, [t], {0: GIB}, "decode")
    assert static.hits == 1


def test_adaptive_min_count_gate():
    # usage of the candidate is 1 when adapting: below min_count 2.0
    t = decode_trace([[1], [0], [1], [1]])
    res = sim.simulate_adaptive({0: {0}}, {0: 1}, [t], {0: GIB}, "decode",
                                land_delay=0, **ADAPT)
    # adapt after round 1 sees u0=1,u1=1: no candidate >= 2.0, no swap.
    # decay -> 0.5 each; round 2: u1=1.5; round 3: u1=2.5 ... adapt after
    # round 3: cand 1 (2.5) vs victim 0 (0.5): 2.5 >= 0.5+1.5 swaps, too late.
    assert (res.hits, res.total) == (1, 4)


def test_adaptive_margin_gate():
    # every=4, adapt after round 3: u0=1, u1=2, u2=1. The best candidate is e1
    # (u=2 >= min_count) against the coldest victim e0 (u=1): gain 1 < 1.5.
    t = decode_trace([[0], [1], [1], [2], [1]])
    kw = dict(ADAPT, every=4)
    res = sim.simulate_adaptive({0: {0}}, {0: 1}, [t], {0: GIB}, "decode",
                                land_delay=0, **dict(kw, margin=1.5))
    assert (res.hits, res.total, res.swaps) == (1, 5, 0)   # only round 0 hits
    # the same trace with a margin of 0.5 swaps e1 in and hits it at round 4
    res = sim.simulate_adaptive({0: {0}}, {0: 1}, [t], {0: GIB}, "decode",
                                land_delay=0, **dict(kw, margin=0.5))
    assert (res.hits, res.total, res.swaps) == (2, 5, 1)


def test_adaptive_decay_forgets_old_heat():
    # r0 [1], r1 [3]; adapt after r1 sees u1 = u3 = 1: below min_count, no
    # swap, then usage *= decay. r2 [1], r3 [3]: u1 = decay + 1. Adapting after
    # r3, e1 qualifies (>= 2.0) only when nothing decayed.
    t = decode_trace([[1], [3], [1], [3], [1]])
    keep = sim.simulate_adaptive({0: {0}}, {0: 1}, [t], {0: GIB}, "decode",
                                 land_delay=0, **dict(ADAPT, decay=1.0))
    assert (keep.hits, keep.swaps) == (1, 1)    # e1 in; round 4 hits
    fade = sim.simulate_adaptive({0: {0}}, {0: 1}, [t], {0: GIB}, "decode",
                                 land_delay=0, **dict(ADAPT, decay=0.5))
    assert (fade.hits, fade.swaps) == (0, 0)


def test_adaptive_swap_cap_is_global_and_ranked_by_gain():
    # adapt once, after round 3. layer 0: cand e1 usage 4 vs victim e0 usage 0,
    # gain 4. layer 1: cand e1 usage 2 vs victim e0 usage 0, gain 2.
    l0 = [[1], [1], [1], [1]]
    l1 = [[1], [1], [2], [3]]
    steps = [sim.Step(index=i, phase=1, n_tokens=1, layers={0: [l0[i]], 1: [l1[i]]})
             for i in range(4)]
    t = sim.Trace(header={}, steps=steps)
    kw = dict(every=4, decay=1.0, min_count=2.0, margin=1.5, land_delay=0)
    args = ({0: {0}, 1: {0}}, {0: 1, 1: 1}, [t], {0: GIB, 1: GIB}, "decode")
    one = sim.simulate_adaptive(*args, swap_n=1, **kw)
    assert one.swap_log == [(3, 0, 1, 0)]       # (round, layer, in, out)
    two = sim.simulate_adaptive(*args, swap_n=2, **kw)
    assert two.swap_log == [(3, 0, 1, 0), (3, 1, 1, 0)]


def test_adaptive_pending_blocks_next_adaptation():
    # two slots {0, 2}; e1 and e3 both hot. Swap A after round 1 lands only at
    # round 5 (land_delay 3). The adaptation after round 3 is skipped while A
    # is in flight (no swap and no decay, as in Strata's early return); without
    # that block it would swap e1 in a second time.
    t = decode_trace([[1, 3]] * 8)
    res = sim.simulate_adaptive({0: {0, 2}}, {0: 2}, [t], {0: GIB}, "decode",
                                swap_n=1, every=2, decay=0.5, min_count=2.0,
                                margin=1.5, land_delay=3)
    assert res.swap_log == [(1, 0, 1, 0), (5, 0, 3, 2)]


def test_adaptive_boundaries_are_inclusive_as_in_strata():
    # Strata: candidates need `usage >= 2.0`, and a pair swaps unless
    # `cand < victim + margin`, so gain == margin and usage == min_count swap.
    t = decode_trace([[1]] * 4)
    kw = dict(swap_n=1, every=2, decay=0.5, land_delay=0)
    # after round 1: u1 = 2.0 == min_count, victim e0 u = 0.0, gain 2.0 == margin
    at = sim.simulate_adaptive({0: {0}}, {0: 1}, [t], {0: GIB}, "decode",
                               min_count=2.0, margin=2.0, **kw)
    assert (at.swaps, at.hits) == (1, 2)
    # two rounds only, so no second adaptation can find the usage again
    t = decode_trace([[1]] * 2)
    over_margin = sim.simulate_adaptive({0: {0}}, {0: 1}, [t], {0: GIB}, "decode",
                                        min_count=2.0, margin=2.0001, **kw)
    assert over_margin.swaps == 0
    over_min = sim.simulate_adaptive({0: {0}}, {0: 1}, [t], {0: GIB}, "decode",
                                     min_count=2.0001, margin=1.5, **kw)
    assert over_min.swaps == 0


def test_adaptive_ties_go_to_the_lower_expert_id():
    kw = dict(swap_n=1, every=2, decay=0.5, min_count=2.0, margin=1.5, land_delay=0)
    # candidates e1 and e3 tie at usage 2; the one victim slot takes e1
    t = decode_trace([[1, 3], [1, 3]])
    res = sim.simulate_adaptive({0: {0}}, {0: 1}, [t], {0: GIB}, "decode", **kw)
    assert res.swap_log == [(1, 0, 1, 0)]
    # victims e0 and e2 tie at usage 0; e0 is evicted first
    t = decode_trace([[1], [1]])
    res = sim.simulate_adaptive({0: {0, 2}}, {0: 2}, [t], {0: GIB}, "decode", **kw)
    assert res.swap_log == [(1, 0, 1, 0)]


def test_cli_train_test_path(tmp_path, capsys):
    tr, te = tmp_path / "train.moetrace", tmp_path / "test.moetrace"
    sim.write_trace(tr, {"set": "code", "id": "code-0", "n_expert": 4},
                    [(i, 0, 1, [[2, 3]]) for i in range(4)])
    sim.write_trace(te, {"set": "chat", "id": "chat-0", "n_expert": 4},
                    [(i, 0, 1, [[2, 1]]) for i in range(4)])
    rc = sim.main(["--train", str(tr), "--test", str(te),
                   "--uniform-expert-bytes", f"f={GIB}:1:4",
                   "--budget-mib-per-layer", "2048"])
    assert rc == 0
    rows = [l.split(",") for l in capsys.readouterr().out.splitlines()[1:]]
    by = {r[2]: r for r in rows}
    assert set(by) == {"static", "oracle", "first-touch", "adaptive"}
    assert by["static"][1] == "chat"
    assert float(by["static"][7]) == 0.5       # trained {2,3}: e2 hits, e1 misses
    assert float(by["oracle"][7]) == 1.0       # profiled on the test itself


def test_cli_budget_gib_total_splits_evenly_over_the_layers(tmp_path, capsys):
    tr, te = tmp_path / "a.moetrace", tmp_path / "b.moetrace"
    sim.write_trace(tr, {"set": "a", "id": "a", "n_expert": 4}, [(0, 0, 1, [[0, 1]])])
    sim.write_trace(te, {"set": "b", "id": "b", "n_expert": 4}, [(0, 0, 1, [[0, 1]])])
    sim.main(["--train", str(tr), "--test", str(te),
              "--uniform-expert-bytes", f"f={GIB}:2:4",
              "--budget-gib-total", "4"])
    rows = [l.split(",") for l in capsys.readouterr().out.splitlines()[1:]]
    # 4 GiB over 2 layers = 2048 MiB each = 2 one-GiB slots per layer
    assert {r[3] for r in rows} == {"2048"}
    assert {r[4] for r in rows} == {"4"}
    assert {float(r[6]) for r in rows} == {4.0}


def test_same_trace_in_train_and_test_is_refused(tmp_path):
    p = tmp_path / "a.moetrace"
    sim.write_trace(p, {"set": "a", "id": "a", "n_expert": 4}, [(0, 0, 1, [[0, 1]])])
    argv = ["--train", str(p), "--test", str(p),
            "--uniform-expert-bytes", f"f={GIB}:1:4", "--budget-mib-per-layer", "1024"]
    with pytest.raises(SystemExit):
        sim.main(argv)
    # by id, with the files differing: the same capture id is still the same trace
    q = tmp_path / "copy.moetrace"
    sim.write_trace(q, {"set": "a", "id": "a", "n_expert": 4}, [(0, 0, 1, [[2, 3]])])
    with pytest.raises(ValueError, match="both train and test"):
        sim.check_disjoint([sim.read_trace(p)], [sim.read_trace(q)])


def test_leave_one_out_refuses_missing_unset_and_duplicate_keys():
    args = dict(sizes_by_format={"f": sizes({0: GIB})},
                budgets_bytes_per_layer=[GIB], phase="decode",
                adapt=dict(swap_n=96, every=4, decay=0.7, min_count=2.0,
                           margin=1.5, land_delay=1))
    a = decode_trace([[0, 1]], set_name="x")
    b = decode_trace([[2, 3]], set_name="y")
    del a.header["id"]
    with pytest.raises(ValueError, match="no header 'id'"):
        sim.sweep_loo(traces=[a, b], by="id", **args)
    a.header["id"] = b.header["id"] = "unset"
    with pytest.raises(ValueError, match="no header 'id'"):
        sim.sweep_loo(traces=[a, b], by="id", **args)
    a.header["id"] = b.header["id"] = "same"
    with pytest.raises(ValueError, match="duplicate trace ids"):
        sim.sweep_loo(traces=[a, b], by="id", **args)
    a.header["set"] = b.header["set"] = "unset"
    with pytest.raises(ValueError, match="no header 'set'"):
        sim.sweep_loo(traces=[a, b], by="set", **args)


def test_trace_must_match_the_expert_sizes():
    kw = dict(budgets_bytes_per_layer=[GIB], phase="decode",
              adapt=dict(swap_n=96, every=4, decay=0.7, min_count=2.0,
                         margin=1.5, land_delay=1))
    ok = decode_trace([[0, 1]], set_name="a")
    other = decode_trace([[2, 3]], set_name="b")
    # header n_expert differs from the sizes
    wrong = decode_trace([[0, 1]], set_name="b", n_expert=8)
    with pytest.raises(ValueError, match="n_expert 8"):
        sim.sweep({"f": sizes({0: GIB})}, [ok], [wrong], **kw)
    # an expert id outside n_expert (header silent about it)
    bad = decode_trace([[0, 4]], set_name="b")
    del bad.header["n_expert"]
    with pytest.raises(ValueError, match="expert 4"):
        sim.sweep({"f": sizes({0: GIB})}, [ok], [bad], **kw)
    # two formats that disagree with each other
    with pytest.raises(ValueError, match="disagree on n_expert"):
        sim.sweep({"f": sizes({0: GIB}), "g": sizes({0: GIB}, n_expert=8)},
                  [ok], [other], **kw)


# ---------------------------------------------------------------------------
# bytes -> slots, CSV, sizes
# ---------------------------------------------------------------------------

def test_slots_for_budget_bytes_per_layer():
    layer_bytes = {0: 100, 1: 300}
    assert sim.slots_for_budget(layer_bytes, 650, n_expert=4) == {0: 4, 1: 2}
    assert sim.slots_for_budget(layer_bytes, 99, n_expert=4) == {0: 0, 1: 0}
    assert sim.slots_for_budget(layer_bytes, 300, n_expert=4) == {0: 3, 1: 1}


def test_sweep_csv_rows_and_header():
    train = decode_trace([[2, 3], [2, 3], [2, 1]], set_name="a")
    test = decode_trace([[0, 1], [2, 3], [2, 1]], set_name="b")
    rows = sim.sweep(
        sizes_by_format={"fmtA": sizes({0: GIB})},
        train=[train], test=[test],
        budgets_bytes_per_layer=[2 * GIB], phase="decode",
        adapt=dict(swap_n=96, every=4, decay=0.7, min_count=2.0, margin=1.5,
                   land_delay=1))
    by_policy = {r["policy"]: r for r in rows}
    assert set(by_policy) == {"static", "oracle", "first-touch", "adaptive"}
    assert by_policy["static"]["hit_rate"] == pytest.approx(0.5)
    assert by_policy["oracle"]["hit_rate"] == pytest.approx(4 / 6)
    assert by_policy["first-touch"]["hit_rate"] == pytest.approx(1 / 6)
    assert by_policy["static"]["resident_slots_total"] == 2
    assert by_policy["static"]["resident_fraction"] == pytest.approx(0.5)
    assert by_policy["static"]["format"] == "fmtA"
    assert by_policy["static"]["budget_mib_per_layer"] == 2048
    # 3 misses of 6 pairs, 1 GiB each, over 3 tokens
    assert by_policy["static"]["cpu_gib_per_token"] == pytest.approx(1.0)
    text = sim.rows_to_csv(rows)
    first = text.splitlines()[0]
    assert first == ("format,test_set,policy,budget_mib_per_layer,"
                     "resident_slots_total,resident_fraction,resident_gib,"
                     "hit_rate,cpu_gib_per_token,full_gib_per_token,tokens,swaps")


def test_leave_one_set_out_splits_by_header_set():
    a = decode_trace([[0, 1], [0, 1]], set_name="code")
    b = decode_trace([[2, 3], [2, 3]], set_name="chat")
    rows = sim.sweep_loo(
        sizes_by_format={"f": sizes({0: GIB})}, traces=[a, b],
        budgets_bytes_per_layer=[2 * GIB], phase="decode",
        adapt=dict(swap_n=96, every=4, decay=0.7, min_count=2.0, margin=1.5,
                   land_delay=1))
    static = {r["test_set"]: r for r in rows if r["policy"] == "static"}
    # trained on the other set only: disjoint experts -> 0 hits
    assert static["code"]["hit_rate"] == 0.0
    assert static["chat"]["hit_rate"] == 0.0
    oracle = {r["test_set"]: r for r in rows if r["policy"] == "oracle"}
    assert oracle["code"]["hit_rate"] == 1.0


def test_trace_summary_counts_and_prefill_coverage():
    steps = [
        sim.Step(index=0, phase=0, n_tokens=2,
                 layers={0: [[0, 1], [1, 2]], 1: [[0, 1], [0, 1]]}),
        sim.Step(index=1, phase=1, n_tokens=1, layers={0: [[3, 0]]}),
    ]
    summary = sim.trace_summary(sim.Trace({"set": "s"}, steps), n_expert=4)
    assert summary["prefill_tokens"] == 2 and summary["decode_tokens"] == 1
    assert summary["layers"] == 2
    # layer 0 touches {0,1,2} = 3/4, layer 1 {0,1} = 2/4
    assert summary["prefill_expert_coverage"] == pytest.approx(0.625)


def test_cli_loo_writes_csv(tmp_path, capsys):
    a, b = tmp_path / "a.moetrace", tmp_path / "b.moetrace"
    sim.write_trace(a, {"set": "code"}, [(i, 0, 1, [[0, 1]]) for i in range(8)])
    sim.write_trace(b, {"set": "chat"}, [(i, 0, 1, [[2, 3]]) for i in range(8)])
    rc = sim.main(["--loo", str(a), str(b),
                   "--uniform-expert-bytes", f"f={GIB}:1:4",
                   "--budget-mib-per-layer", "1024,2048"])
    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("format,test_set,policy,budget_mib_per_layer")
    # 2 held-out sets x 2 budgets x 4 policies
    assert len(lines) == 1 + 2 * 2 * 4


def test_leave_one_id_out_keeps_the_same_set_in_training():
    a = decode_trace([[0, 1], [0, 1]], set_name="code")
    a.header["id"] = "code-0"
    b = decode_trace([[0, 1]] * 4, set_name="code")
    b.header["id"] = "code-1"
    c = decode_trace([[2, 3], [2, 3]], set_name="chat")
    c.header["id"] = "chat-0"
    kw = dict(sizes_by_format={"f": sizes({0: GIB})}, traces=[a, b, c],
              budgets_bytes_per_layer=[2 * GIB], phase="decode",
              adapt=dict(swap_n=96, every=4, decay=0.7, min_count=2.0,
                         margin=1.5, land_delay=1))
    by_id = {r["test_set"]: r for r in sim.sweep_loo(**kw, by="id")
             if r["policy"] == "static"}
    assert by_id["code-0"]["hit_rate"] == 1.0   # counts: e0,e1 = 4 (code-1) beat e2,e3 = 2
    by_set = {r["test_set"]: r for r in sim.sweep_loo(**kw, by="set")
              if r["policy"] == "static"}
    assert by_set["code"]["hit_rate"] == 0.0    # trained on chat alone


# ---------------------------------------------------------------------------
# trace file format
# ---------------------------------------------------------------------------

def test_trace_roundtrip(tmp_path):
    p = tmp_path / "x.moetrace"
    header = {"set": "code", "id": "p0", "n_expert": 256}
    records = [
        (0, 0, 0, [[1, 2], [3, 4]]),
        (0, 1, 0, [[5, 6], [7, 8]]),
        (1, 0, 1, [[9, 10]]),
    ]
    sim.write_trace(p, header, records)
    t = sim.read_trace(p)
    assert t.header["set"] == "code" and t.header["n_expert"] == 256
    assert [s.phase for s in t.steps] == [0, 1]
    assert t.steps[0].n_tokens == 2
    assert t.steps[0].layers == {0: [[1, 2], [3, 4]], 1: [[5, 6], [7, 8]]}
    assert t.steps[1].layers == {0: [[9, 10]]}


def test_trace_bad_magic_and_truncation_are_refused(tmp_path):
    bad = tmp_path / "bad.moetrace"
    bad.write_bytes(b"NOTATRAC" + b"\0" * 32)
    with pytest.raises(ValueError, match="magic"):
        sim.read_trace(bad)
    good = tmp_path / "good.moetrace"
    sim.write_trace(good, {"set": "s"}, [(0, 0, 1, [[1, 2]])])
    raw = good.read_bytes()
    cut = tmp_path / "cut.moetrace"
    cut.write_bytes(raw[:-1])
    with pytest.raises(ValueError, match="truncated"):
        sim.read_trace(cut)


def test_cxx_tool_selftest_writes_a_trace_the_simulator_reads(tmp_path):
    binary = os.environ.get("LLAMA_MOE_TRACE_BIN")
    candidates = [pathlib.Path(binary)] if binary else [
        ROOT / "build" / "bin" / "llama-moe-trace",
        ROOT / "build-cpu" / "bin" / "llama-moe-trace",
    ]
    exe = next((c for c in candidates if c.is_file()), None)
    if exe is None:
        pytest.skip("llama-moe-trace is not built")
    out = tmp_path / "self.moetrace"
    r = subprocess.run([str(exe), "--selftest-write", str(out)],
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    t = sim.read_trace(out)
    # selftest: layer 7 is a [8, 3] strided view of a [256, 3] I32 parent
    # (value r * 11 + c); layer 9 a contiguous [4, 2] tensor (value r * 5 + c);
    # then layer 7 again, [8, 1], which must start a second step; layer 11 a
    # strided [6, 2] view of a [64, 2] tensor in a real ggml buffer.
    assert t.header["selftest"] is True
    assert t.steps[0].layers[7] == [[r * 11 + c for c in range(8)] for r in range(3)]
    assert t.steps[0].layers[9] == [[r * 5 + c for c in range(4)] for r in range(2)]
    # layer index not increasing starts a new step
    assert len(t.steps) == 2
    assert t.steps[1].layers[7] == [[0, 1, 2, 3, 4, 5, 6, 7]]
    # layer 11: a strided view read through ggml_backend_tensor_get
    assert t.steps[1].layers[11] == [[r * 13 + c for c in range(6)] for r in range(2)]


# ---------------------------------------------------------------------------
# expert bytes from a GGUF (gguf-py)
# ---------------------------------------------------------------------------

def test_expert_sizes_from_gguf_shards(tmp_path):
    np = pytest.importorskip("numpy")
    sys_path_gguf = ROOT / "gguf-py"
    sys.path.insert(0, str(sys_path_gguf))
    try:
        gguf = pytest.importorskip("gguf")
    finally:
        sys.path.remove(str(sys_path_gguf))
    n_expert, n_ff, n_embd = 4, 8, 16

    def write(path, tensors):
        w = gguf.GGUFWriter(str(path), "qwen4exp")
        for name, arr in tensors:
            w.add_tensor(name, arr)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()

    f32 = lambda *s: np.zeros(s, dtype=np.float32)
    f16 = lambda *s: np.zeros(s, dtype=np.float16)
    shard1 = tmp_path / "m-00001-of-00002.gguf"
    shard2 = tmp_path / "m-00002-of-00002.gguf"
    write(shard1, [
        ("blk.0.ffn_gate_exps.weight", f32(n_expert, n_ff, n_embd)),
        ("blk.0.ffn_up_exps.weight", f32(n_expert, n_ff, n_embd)),
        ("blk.0.ffn_gate_inp.weight", f32(n_expert, n_embd)),
        ("blk.0.ffn_gate_shexp.weight", f32(n_ff, n_embd)),
    ])
    write(shard2, [
        ("blk.0.ffn_down_exps.weight", f32(n_expert, n_embd, n_ff)),
        ("blk.1.ffn_gate_up_exps.weight", f16(n_expert, 2 * n_ff, n_embd)),
        ("blk.1.ffn_down_exps.weight", f16(n_expert, n_embd, n_ff)),
    ])
    # given the first shard only, the rest are discovered by name
    got = sim.load_expert_sizes_gguf([str(shard1)])
    assert got["n_expert"] == n_expert
    assert got["layer_bytes"] == {
        0: 3 * n_ff * n_embd * 4,
        1: (2 * n_ff * n_embd + n_ff * n_embd) * 2,
    }
