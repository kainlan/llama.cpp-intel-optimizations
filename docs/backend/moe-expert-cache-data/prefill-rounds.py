#!/usr/bin/env python3
"""Round-by-round replay of one held-out trace, to see why the adaptive policy
does badly when the rounds include prompt prefill (phase=all). Writes two CSVs.

Re-implements the three policy loops of scripts/moe-cache-sim.py using its own
building blocks (_score_round, _adapt, profiles, slots), records one row per
round, and asserts that its totals equal the simulator's own policy functions on
the same inputs (a replay that disagreed with the simulator would be measuring
something else).

  PYTHONPATH=gguf-py python3 prefill-rounds.py TEST.moetrace TRAIN.moetrace... \
      [--budget-mib-per-layer 149.333] --out-rounds rounds.csv --out-windows windows.csv

The committed long-0-*.csv files came from:
  prefill-rounds.py long-0.moetrace chat-{0..3}.moetrace code-{0..3}.moetrace \
      --out-rounds long-0-rounds.csv --out-windows long-0-windows.csv
(phase all, 149.333 MiB per layer = 7 GiB nominal, IQ3_XXS sizes).

rounds.csv   one row per round: phase, n_tokens, the mean fraction of a layer's
             experts that round touches, and the hit rate of each policy IN THAT
             ROUND (adaptive, static, first-touch as the simulator models it, and
             first-touch with admission at the END of the round).
windows.csv  one row per adaptation: how many (layer, expert) usage counters are
             at or above min_count, how many candidate swaps passed the margin,
             how many were taken (the swap_n cap), and the cumulative swaps.
"""
import argparse, csv, importlib.util, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "moe_cache_sim", os.path.join(HERE, "..", "..", "..", "scripts", "moe-cache-sim.py"))
sim = importlib.util.module_from_spec(spec)
sys.modules["moe_cache_sim"] = sim
spec.loader.exec_module(sim)

IQ3 = "/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"


def per_round(res, before):
    h, t = res.hits - before[0], res.total - before[1]
    return h / t if t else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test")
    ap.add_argument("train", nargs="+")
    ap.add_argument("--budget-mib-per-layer", type=float, default=149.333)
    ap.add_argument("--phase", default="all")
    ap.add_argument("--out-rounds", required=True)
    ap.add_argument("--out-windows", required=True)
    a = ap.parse_args()

    sizes = sim.load_expert_sizes_gguf([IQ3])
    n_expert, layer_bytes = sizes["n_expert"], sizes["layer_bytes"]
    test = sim.read_trace(a.test)
    train = [sim.read_trace(p) for p in a.train]
    sim.check_disjoint(train, [test])
    slots = sim.slots_for_budget(layer_bytes, a.budget_mib_per_layer * sim.MIB, n_expert)
    init = sim.resident_from_profile(sim.build_profile(train, a.phase), slots, n_expert)
    rounds = sim._rounds(test, a.phase)
    p = dict(swap_n=96, every=4, decay=0.7, min_count=2.0, margin=1.5, land_delay=1)

    # adaptive, instrumented copy of simulate_adaptive
    res_a = sim.Result()
    resident = {l: set(v) for l, v in init.items()}
    usage, pending = {}, []
    windows, rows = [], []
    per = {"adaptive": [], "static": [], "first_touch": [], "first_touch_end_of_round": []}
    for r, s in enumerate(rounds):
        landed = [x for x in pending if x[0] <= r]
        pending = [x for x in pending if x[0] > r]
        for _due, layer, e in landed:
            resident[layer].add(e)
        before = (res_a.hits, res_a.total)
        sim._score_round(s, resident, layer_bytes, res_a, usage=usage)
        per["adaptive"].append(per_round(res_a, before))
        if (r + 1) % p["every"] == 0:
            n_before = res_a.swaps
            skipped = bool(pending)
            above = sum(1 for u in usage.values() for v in u.values() if v >= p["min_count"])
            # candidates that pass the margin: count what _adapt would pair, before the cap
            cand = 0
            for layer in sorted(resident):
                u = usage.get(layer, {})
                have = resident[layer]
                c = sorted(((-uu, e) for e, uu in u.items() if e not in have and uu >= p["min_count"]))
                v = sorted((u.get(e, 0.0), e) for e in have)
                for (ncu, _ce), (vu, _ve) in zip(c, v):
                    if -ncu < vu + p["margin"]:
                        break
                    cand += 1
            sim._adapt(r, resident, usage, pending, p["swap_n"], p["decay"], p["min_count"],
                       p["margin"], p["land_delay"], res_a)
            windows.append({"round": r, "phase": s.phase, "n_tokens": s.n_tokens,
                            "skipped_pending": int(skipped),
                            "counters_ge_min_count": above,
                            "counters_total": len(layer_bytes) * n_expert,
                            "candidate_swaps_over_margin": 0 if skipped else cand,
                            "swaps_taken": res_a.swaps - n_before, "swaps_cum": res_a.swaps})

    # static
    res_s = sim.Result()
    cache = {l: set(v) for l, v in init.items()}
    for s in rounds:
        before = (res_s.hits, res_s.total)
        sim._score_round(s, cache, layer_bytes, res_s)
        per["static"].append(per_round(res_s, before))

    # first-touch as the simulator models it: a miss admits at once, so later tokens of
    # the SAME round already hit (for a 512-token ubatch that is most of its tokens)
    res_f = sim.Result()
    cache = {}
    def admit(layer, e):
        have = cache.setdefault(layer, set())
        if len(have) < slots.get(layer, 0):
            have.add(e)
    for s in rounds:
        before = (res_f.hits, res_f.total)
        sim._score_round(s, cache, layer_bytes, res_f, on_miss=admit)
        per["first_touch"].append(per_round(res_f, before))

    # first-touch with admission at the end of the round: the copy cannot land
    # mid-ubatch, so every token of the round sees the cache as it was at the start
    res_g = sim.Result()
    cache = {}
    for s in rounds:
        before = (res_g.hits, res_g.total)
        misses = []
        sim._score_round(s, cache, layer_bytes, res_g, on_miss=lambda l, e: misses.append((l, e)))
        for l, e in misses:
            have = cache.setdefault(l, set())
            if len(have) < slots.get(l, 0):
                have.add(e)
        per["first_touch_end_of_round"].append(per_round(res_g, before))

    # the replay must equal the simulator's own policies on the same inputs
    chk_a = sim.simulate_adaptive(init, slots, [test], layer_bytes, a.phase, **p)
    chk_s = sim.simulate_static(init, [test], layer_bytes, a.phase)
    chk_f = sim.simulate_first_touch(slots, [test], layer_bytes, a.phase)
    for name, mine, theirs in (("adaptive", res_a, chk_a), ("static", res_s, chk_s),
                               ("first_touch", res_f, chk_f)):
        assert (mine.hits, mine.total, mine.swaps) == (theirs.hits, theirs.total, theirs.swaps), name

    with open(a.out_rounds, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["round", "phase", "n_tokens", "mean_layer_expert_fraction_touched",
                    "hit_adaptive", "hit_static", "hit_first_touch", "hit_first_touch_end_of_round"])
        for i, s in enumerate(rounds):
            frac = [len({e for row in rows_ for e in row}) / n_expert
                    for rows_ in s.layers.values() if rows_]
            w.writerow([i, s.phase, s.n_tokens, f"{sum(frac) / len(frac):.4f}" if frac else "",
                        f"{per['adaptive'][i]:.4f}", f"{per['static'][i]:.4f}",
                        f"{per['first_touch'][i]:.4f}", f"{per['first_touch_end_of_round'][i]:.4f}"])
    with open(a.out_windows, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(windows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(windows)
    tot = lambda r: f"{r.hits / r.total:.4f}"
    print(f"slots_total={sum(slots.values())} rounds={len(rounds)} adaptive={tot(res_a)} static={tot(res_s)} "
          f"first_touch={tot(res_f)} first_touch_end_of_round={tot(res_g)} swaps={res_a.swaps}")


if __name__ == "__main__":
    main()
