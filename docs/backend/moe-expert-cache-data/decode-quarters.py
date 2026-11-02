#!/usr/bin/env python3
"""Hit rate by quarter of the 256 decode rounds, per held-out set and per trace, IQ3_XXS sizes,
7 GiB nominal (149.333 MiB per layer), phase decode, leave-one-set-out. Shows whether the adaptive
policy keeps improving through a short trace (warm-up) without any prefill in the picture.

  PYTHONPATH=gguf-py python3 decode-quarters.py TRACE.moetrace... --out decode-quarters.csv

Replays the adaptive, static and first-touch policies round by round with the simulator's own
building blocks and asserts that the totals of every held-out set equal the simulator's policy
functions on the same inputs. Rows: held_out is a set name or a trace id; quarter 1 = decode rounds
0-63. A set's row is the mean over its traces' rounds (each round is 480 routed pairs, so this is
also the pair-weighted rate).
"""
import argparse, csv, importlib.util, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "moe_cache_sim", os.path.join(HERE, "..", "..", "..", "scripts", "moe-cache-sim.py"))
sim = importlib.util.module_from_spec(spec)
sys.modules["moe_cache_sim"] = sim
spec.loader.exec_module(sim)

IQ3 = "/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
P = dict(swap_n=96, every=4, decay=0.7, min_count=2.0, margin=1.5, land_delay=1)


def per_round_hits(policy, init, slots, trace, layer_bytes):
    """hit rate of each decode round of one trace under one policy"""
    res = sim.Result()
    out = []
    rounds = sim._rounds(trace, "decode")
    if policy == "static":
        cache = {l: set(v) for l, v in init.items()}
        for s in rounds:
            b = (res.hits, res.total)
            sim._score_round(s, cache, layer_bytes, res)
            out.append((res.hits - b[0]) / (res.total - b[1]))
    elif policy == "first-touch":
        cache = {}

        def admit(layer, e):
            have = cache.setdefault(layer, set())
            if len(have) < slots.get(layer, 0):
                have.add(e)
        for s in rounds:
            b = (res.hits, res.total)
            sim._score_round(s, cache, layer_bytes, res, on_miss=admit)
            out.append((res.hits - b[0]) / (res.total - b[1]))
    else:
        resident = {l: set(v) for l, v in init.items()}
        usage, pending = {}, []
        for r, s in enumerate(rounds):
            for _due, layer, e in [x for x in pending if x[0] <= r]:
                resident[layer].add(e)
            pending = [x for x in pending if x[0] > r]
            b = (res.hits, res.total)
            sim._score_round(s, resident, layer_bytes, res, usage=usage)
            out.append((res.hits - b[0]) / (res.total - b[1]))
            if (r + 1) % P["every"] == 0:
                sim._adapt(r, resident, usage, pending, P["swap_n"], P["decay"], P["min_count"],
                           P["margin"], P["land_delay"], res)
    return out, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--budget-mib-per-layer", type=float, default=149.333)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sizes = sim.load_expert_sizes_gguf([IQ3])
    n_expert, layer_bytes = sizes["n_expert"], sizes["layer_bytes"]
    slots = sim.slots_for_budget(layer_bytes, a.budget_mib_per_layer * sim.MIB, n_expert)
    traces = [sim.read_trace(p) for p in a.traces]
    rows = []
    for held in sorted({sim.trace_set(t) for t in traces}):
        test = [t for t in traces if sim.trace_set(t) == held]
        train = [t for t in traces if sim.trace_set(t) != held]
        init = sim.resident_from_profile(sim.build_profile(train, "decode"), slots, n_expert)
        totals = {}
        curves = {}
        for pol in ("adaptive", "static", "first-touch"):
            tot = sim.Result()
            for t in test:
                c, res = per_round_hits(pol, init, slots, t, layer_bytes)
                curves[(pol, sim.trace_id(t))] = c
                tot.hits += res.hits
                tot.total += res.total
                tot.swaps += res.swaps
            totals[pol] = tot
        chk = {"adaptive": sim.simulate_adaptive(init, slots, test, layer_bytes, "decode", **P),
               "static": sim.simulate_static(init, test, layer_bytes, "decode"),
               "first-touch": sim.simulate_first_touch(slots, test, layer_bytes, "decode")}
        for pol in totals:
            assert (totals[pol].hits, totals[pol].total) == (chk[pol].hits, chk[pol].total), (held, pol)
        for pol in ("adaptive", "static", "first-touch"):
            ids = [sim.trace_id(t) for t in test]
            for who, members in [(held, ids)] + ([(i, [i]) for i in ids] if len(ids) > 1 else []):
                n = len(curves[(pol, members[0])])
                for q in range(4):
                    lo, hi = q * n // 4, (q + 1) * n // 4
                    vals = [x for m in members for x in curves[(pol, m)][lo:hi]]
                    rows.append([who, pol, q + 1, f"{lo}-{hi - 1}", f"{sum(vals) / len(vals):.4f}", len(vals)])
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["held_out", "policy", "quarter", "decode_rounds", "hit_rate", "n_round_samples"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
