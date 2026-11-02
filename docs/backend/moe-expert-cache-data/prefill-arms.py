#!/usr/bin/env python3
"""The adaptive policy with prefill swaps on, off and seed (moe-cache-sim.py --prefill-swaps),
split into the prefill rounds and the decode rounds that follow, for each held-out long trace.

  PYTHONPATH=gguf-py python3 prefill-arms.py TRACE.moetrace... --out prefill-arms.csv

Leave-one-set-out, phase all, budgets 4, 7 and 14 GiB nominal, IQ3_XXS and Q8_0 sizes. The decode
share of an arm is its all-rounds result minus the result on the same trace cut to its prefill
rounds (rounds do not depend on later rounds, so the prefill part of a cut run is identical).
Static and first-touch are given as references. Rows: held_out is a set, or a trace id for the
long traces, so a single prompt's numbers are visible.
"""
import argparse, csv, importlib.util, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "moe_cache_sim", os.path.join(HERE, "..", "..", "..", "scripts", "moe-cache-sim.py"))
sim = importlib.util.module_from_spec(spec)
sys.modules["moe_cache_sim"] = sim
spec.loader.exec_module(sim)

GGUF = {
    "iq3": "/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
    "q8": "/models/Qwen3.8-Flash-Next-GGUF/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf",
}
BUDGETS = {4: 85.333, 7: 149.333, 14: 298.667}
P = dict(swap_n=96, every=4, decay=0.7, min_count=2.0, margin=1.5, land_delay=1)


def prefill_only(t):
    return sim.Trace(t.header, [s for s in t.steps if s.phase == 0], t.path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    traces = [sim.read_trace(p) for p in a.traces]
    rows = []
    for fmt, path in GGUF.items():
        sizes = sim.load_expert_sizes_gguf([path])
        n_expert, lb = sizes["n_expert"], sizes["layer_bytes"]
        for gib, mib in BUDGETS.items():
            slots = sim.slots_for_budget(lb, mib * sim.MIB, n_expert)
            for held in sorted({sim.trace_set(t) for t in traces}):
                test = [t for t in traces if sim.trace_set(t) == held]
                train = [t for t in traces if sim.trace_set(t) != held]
                init = sim.resident_from_profile(sim.build_profile(train, "all"), slots, n_expert)
                groups = [(held, test)] + ([(sim.trace_id(t), [t]) for t in test] if held == "long" else [])
                for who, ts in groups:
                    pre = [prefill_only(t) for t in ts]
                    arms = {}
                    for mode in sim.PREFILL_SWAPS:
                        arms["adaptive-" + mode] = (
                            sim.simulate_adaptive(init, slots, ts, lb, "all", prefill_swaps=mode, **P),
                            sim.simulate_adaptive(init, slots, pre, lb, "all", prefill_swaps=mode, **P))
                    arms["static"] = (sim.simulate_static(init, ts, lb, "all"),
                                      sim.simulate_static(init, pre, lb, "all"))
                    arms["first-touch"] = (sim.simulate_first_touch(slots, ts, lb, "all"),
                                           sim.simulate_first_touch(slots, pre, lb, "all"))
                    for arm, (full, cut) in arms.items():
                        dh, dt = full.hits - cut.hits, full.total - cut.total
                        rows.append([fmt, gib, who, arm,
                                     f"{cut.hits / cut.total:.4f}", cut.total,
                                     f"{dh / dt:.4f}" if dt else "", dt,
                                     f"{full.hits / full.total:.4f}", full.swaps - cut.swaps, full.swaps])
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["format", "budget_gib", "held_out", "arm", "prefill_hit_rate", "prefill_pairs",
                    "decode_hit_rate", "decode_pairs", "all_rounds_hit_rate",
                    "decode_swaps", "total_swaps"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
