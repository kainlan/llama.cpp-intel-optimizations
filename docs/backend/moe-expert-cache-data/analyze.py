#!/usr/bin/env python3
"""Derives every summary table of docs/backend/moe-expert-cache-findings.md from the
committed raw CSVs in this directory. Standard library only:

    python3 analyze.py

Reads   curves-{decode,all}-by-{set,id}.csv, curves-all-by-set-prefill-{off,seed}.csv,
        ablations/*.csv, expert-sizes.csv, decode-quarters.csv,
        long-0-rounds.csv
Writes  summary-curves.csv, summary-ablations.csv, summary-prefill-arms.csv, swap-traffic.csv,
        cpu-bound-estimate.csv, long-0-blocks.csv

Token-weighted means: sum(value * tokens) / sum(tokens) over the held-out test
groups of one (phase, loo_by, format, policy, budget) cell. `tokens` is the number
of rounds' tokens the simulator scored (decode: 256 per trace; phase=all adds the
prompt tokens). Each token routes 10 experts in each of 48 layers, so this equals
the pair-weighted rate except where a zero-row last layer drops pairs (prefill).
"""
import csv, glob, os, re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
GIB = 1 << 30
MIB = 1 << 20
LAYERS = 48
# nominal budget in GiB for each budget_mib_per_layer used by run-curves.sh
NOMINAL = {42.667: 2, 85.333: 4, 149.333: 7, 213.333: 10, 298.667: 14, 384.0: 18, 512.0: 24}
POLICIES = ("adaptive", "first-touch", "static", "oracle")


def rows_of(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def nominal(mib):
    mib = round(float(mib), 3)
    if mib in NOMINAL:
        return NOMINAL[mib]
    return round(mib * LAYERS / 1024)   # ablations: --budget-gib-total 4,7,14


def tw(rows, key):
    t = sum(float(r["tokens"]) for r in rows)
    return sum(float(r[key]) * float(r["tokens"]) for r in rows) / t, t


def write(name, header, rows):
    with open(os.path.join(HERE, name), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def main():
    # ---- summary-curves.csv
    out = []
    for path in sorted(glob.glob(os.path.join(HERE, "curves-*-by-*.csv"))):
        if "prefill-" in os.path.basename(path):
            continue   # the prefill arms are summarised in summary-prefill-arms.csv
        m = re.match(r"curves-(\w+)-by-(\w+)\.csv", os.path.basename(path))
        phase, by = m.groups()
        cells = defaultdict(list)
        for r in rows_of(path):
            cells[(r["format"], r["policy"], nominal(r["budget_mib_per_layer"]))].append(r)
        for (fmt, pol, gib), rs in sorted(cells.items(), key=lambda kv: (kv[0][0], POLICIES.index(kv[0][1]), kv[0][2])):
            hit, tokens = tw(rs, "hit_rate")
            cpu, _ = tw(rs, "cpu_gib_per_token")
            hits = [float(r["hit_rate"]) for r in rs]
            swaps = sum(int(r["swaps"]) for r in rs)
            out.append([phase, by, fmt, pol, gib,
                        f"{sum(float(r['resident_gib']) for r in rs) / len(rs):.3f}",
                        f"{sum(float(r['resident_fraction']) for r in rs) / len(rs):.4f}",
                        f"{hit:.4f}", f"{cpu:.4f}", f"{swaps / tokens:.3f}", int(tokens), len(rs),
                        f"{min(hits):.4f}", f"{max(hits):.4f}"])
    write("summary-curves.csv",
          ["phase", "loo_by", "format", "policy", "budget_gib_nominal", "resident_gib_mean",
           "resident_fraction_of_experts", "hit_rate_token_weighted", "cpu_gib_per_token_token_weighted", "swaps_per_token",
           "tokens", "n_test_groups", "hit_rate_min_group", "hit_rate_max_group"], out)
    curves = {(r[0], r[1], r[2], r[3], r[4]): r for r in out}

    # ---- summary-ablations.csv (adaptive, decode, leave-one-set-out)
    default = {}
    abl = []
    for path in sorted(glob.glob(os.path.join(HERE, "ablations", "*.csv"))):
        name = os.path.basename(path)[:-4]
        m = re.match(r"^([a-z]+)([0-9.]+)$", name)
        param, value = (m.group(1), m.group(2)) if m else ("default", "")
        cells = defaultdict(list)
        for r in rows_of(path):
            if r["policy"] == "adaptive":
                cells[(r["format"], nominal(r["budget_mib_per_layer"]))].append(r)
        for (fmt, gib), rs in cells.items():
            hit, tokens = tw(rs, "hit_rate")
            cpu, _ = tw(rs, "cpu_gib_per_token")
            swaps = sum(int(r["swaps"]) for r in rs) / tokens
            abl.append([param, value, fmt, gib, hit, cpu, swaps])
            if param == "default":
                default[(fmt, gib)] = hit
    order = {"default": 0, "every": 1, "swapn": 2, "land": 3, "landnoskip": 4, "decay": 5, "margin": 6,
             "mincount": 7}
    abl.sort(key=lambda r: (order[r[0]], float(r[1] or 0), r[2], r[3]))
    write("summary-ablations.csv",
          ["param", "value", "format", "budget_gib", "hit_rate_token_weighted", "delta_vs_default",
           "cpu_gib_per_token_token_weighted", "swaps_per_token"],
          [[p, v, f, g, f"{h:.4f}", f"{h - default[(f, g)]:+.4f}", f"{c:.4f}", f"{s:.2f}"]
           for p, v, f, g, h, c, s in abl])

    # ---- expert sizes
    sizes = defaultdict(list)
    for r in rows_of(os.path.join(HERE, "expert-sizes.csv")):
        sizes[r["format"]].append(int(r["expert_bytes"]))
    stat = {f: (min(v), sum(v) / len(v), max(v)) for f, v in sizes.items()}

    # ---- swap-traffic.csv: adaptive, decode, leave-one-set-out
    tr = []
    for fmt in ("iq3", "q8"):
        lo, mean, hi = stat[fmt]
        for gib in (2, 4, 7, 10, 14, 18, 24):
            row = curves[("decode", "set", fmt, "adaptive", gib)]
            spt = float(row[9])
            stat_row = curves[("decode", "set", fmt, "static", gib)]
            adapt_row = row
            cpu_saved = (float(stat_row[8]) - float(adapt_row[8])) * 1024   # MiB/token vs static
            extra_hits = (float(adapt_row[7]) - float(stat_row[7])) * LAYERS * 10 / spt
            mib = lambda b: spt * b / MIB
            tr.append([fmt, gib, f"{spt:.2f}", f"{mean / MIB:.3f}", f"{mib(lo):.1f}", f"{mib(mean):.1f}",
                       f"{mib(hi):.1f}"] +
                      [f"{mib(mean) * MIB * tps / 1e9:.2f}" for tps in (20, 40)] +
                      [f"{mib(lo) * MIB * 40 / 1e9:.2f}", f"{mib(hi) * MIB * 40 / 1e9:.2f}",
                       f"{cpu_saved:.0f}", f"{cpu_saved / mib(mean):.1f}", f"{extra_hits:.1f}"])
    write("swap-traffic.csv",
          ["format", "budget_gib", "swaps_per_token", "mean_expert_mib", "mib_per_token_if_min_layer",
           "mib_per_token_mean_layer", "mib_per_token_if_max_layer", "gb_per_s_at_20_tps_mean",
           "gb_per_s_at_40_tps_mean", "gb_per_s_at_40_tps_min_layer", "gb_per_s_at_40_tps_max_layer",
           "cpu_mib_per_token_saved_vs_static", "cpu_mib_saved_per_mib_swapped",
           "extra_hits_per_swap_vs_static"], tr)

    # ---- cpu-bound-estimate.csv: upper bound on the CPU-limited decode rate
    est = []
    for fmt in ("iq3", "q8"):
        for pol in ("adaptive", "first-touch", "static", "oracle"):
            for gib in (4, 7, 14):
                row = curves[("decode", "set", fmt, pol, gib)]
                cpu_gb = float(row[8]) * GIB / 1e9
                est.append([fmt, pol, gib, row[7], f"{float(row[8]):.4f}", f"{cpu_gb:.3f}"] +
                           [f"{bw / cpu_gb:.1f}" for bw in (14, 21, 30, 47)])
    # no cache at all: every routed expert is read by the CPU (full_gib_per_token)
    for fmt in ("iq3", "q8"):
        rs = [r for r in rows_of(os.path.join(HERE, "curves-decode-by-set.csv"))
              if r["format"] == fmt and r["policy"] == "adaptive" and nominal(r["budget_mib_per_layer"]) == 7]
        full, _ = tw(rs, "full_gib_per_token")
        gb = full * GIB / 1e9
        est.append([fmt, "none", 0, "0.0000", f"{full:.4f}", f"{gb:.3f}"] +
                   [f"{bw / gb:.1f}" for bw in (14, 21, 30, 47)])
    write("cpu-bound-estimate.csv",
          ["format", "policy", "budget_gib", "hit_rate_token_weighted", "cpu_gib_per_token",
           "cpu_gb_per_token", "tps_bound_at_14_gbps", "tps_bound_at_21_gbps",
           "tps_bound_at_30_gbps", "tps_bound_at_47_gbps"], est)

    # ---- summary-prefill-arms.csv: adaptive with prefill swaps on / off / seed (phase all, by set)
    arms = []
    for arm, name in (("on", "curves-all-by-set.csv"), ("off", "curves-all-by-set-prefill-off.csv"),
                      ("seed", "curves-all-by-set-prefill-seed.csv")):
        cells = defaultdict(list)
        for r in rows_of(os.path.join(HERE, name)):
            if r["policy"] == "adaptive":
                cells[(r["format"], nominal(r["budget_mib_per_layer"]))].append(r)
        for (fmt, gib), rs in sorted(cells.items()):
            hit, tokens = tw(rs, "hit_rate")
            cpu, _ = tw(rs, "cpu_gib_per_token")
            arms.append([fmt, gib, "adaptive-" + arm, f"{hit:.4f}", f"{cpu:.4f}",
                         f"{sum(int(r['swaps']) for r in rs) / tokens:.3f}", int(tokens)])
    write("summary-prefill-arms.csv",
          ["format", "budget_gib", "arm", "hit_rate_token_weighted_all_rounds",
           "cpu_gib_per_token_token_weighted", "swaps_per_token", "tokens"], arms)

    # ---- long-0-blocks.csv: where the phase=all collapse happens
    rounds = rows_of(os.path.join(HERE, "long-0-rounds.csv"))
    blocks = []
    def block(label, rs):
        t = sum(int(r["n_tokens"]) for r in rs)
        blocks.append([label, len(rs), t] + [f"{sum(float(r[k]) * int(r['n_tokens']) for r in rs) / t:.4f}"
                       for k in ("hit_adaptive", "hit_static", "hit_first_touch", "hit_first_touch_end_of_round")])
    block("prefill (all 20 rounds)", [r for r in rounds if r["phase"] == "0"])
    dec = [r for r in rounds if r["phase"] == "1"]
    for a in range(0, len(dec), 32):
        block(f"decode rounds {a}-{min(a + 32, len(dec)) - 1}", dec[a:a + 32])
    block("decode (all 256 rounds)", dec)
    block("prefill + decode", rounds)
    # control: a fresh decode-only run of long-0 (chat+code decode profile), same budget
    q = [r for r in rows_of(os.path.join(HERE, "decode-quarters.csv")) if r["held_out"] == "long-0"]
    mean = lambda pol: sum(float(r["hit_rate"]) for r in q if r["policy"] == pol) / 4   # equal quarters
    blocks.append(["decode-only control (decode-quarters.csv)", 256, 256, f"{mean('adaptive'):.4f}",
                   f"{mean('static'):.4f}", f"{mean('first-touch'):.4f}", ""])
    write("long-0-blocks.csv",
          ["rounds", "n_rounds", "n_tokens", "hit_adaptive", "hit_static", "hit_first_touch",
           "hit_first_touch_end_of_round"], blocks)


if __name__ == "__main__":
    main()
