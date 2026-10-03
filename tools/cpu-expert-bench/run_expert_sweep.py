#!/usr/bin/env python3
"""Thread sweep for bench-cpu-expert-matvec (llama.cpp-cego).

The pool's TBB arena size is fixed once per process by GGML_SYCL_CPU_THREADS, so
a sweep is one process per thread count. Rounds are interleaved: every round
visits all thread counts in a fresh random order. Inside each process the
variants (including the pinned arms, `prod+pin` etc.) are interleaved burst by
burst, so a pinned and an unpinned arm see the same host state (see the bench
header).

    source /opt/intel/oneapi/setvars.sh --force
    ./run_expert_sweep.py --bin build/bin/bench-cpu-expert-matvec --rounds 3 \
        --out expert-sweep.csv -- --types q8_0,mxfp4 --shapes qwen38 \
        --variants prod,prod+pin,read,read+pin

Writes every raw call to --out, every stderr line of the bench (oracle results,
pin counts, last-run CPU samples) to <out>.stderr, and prints a summary CSV.
A bench exit code other than 0 stops the sweep: exit 3 is an oracle FAIL.

Summary columns, per (shape,mat,type,threads,variant), pooled over all rounds:
  best_gbps p90_gbps median_gbps p10_gbps min_gbps   effective weight GB/s per call
  burst_gbps    median over (round,burst) of bytes/time summed over the burst's calls
  ratio_read    median over (round,burst) of the variant's burst median divided by the
                `read` variant's burst median with the SAME pin suffix in the SAME
                burst (+pinE/+pinS fall back to read+pin, then read; the base is the
                first that was run, so keep it constant when comparing arms). Absolute GB/s float
                with ambient load; this ratio and the burst medians are the
                comparable numbers. Do not compare `best_gbps` of sub-millisecond
                calls with a STREAM best taken over 20-40 ms trials.
"""
import argparse
import csv
import os
import random
import statistics
import subprocess
import sys

HDR = "round,shape,mat,type,N,K,k,threads,variant,burst,call,us,bytes,gbps,cpu_us".split(",")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--threads", default="1,2,4,8,12,16,20,24")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--from-csv", help="re-summarise an existing --out CSV instead of running")
    ap.add_argument("--pairs", help="comma list of A:B variant pairs; prints a second CSV with the median over "
                    "(round,burst) of A's burst median divided by B's burst median in the same burst, "
                    "plus the 25th/75th percentile of that ratio, e.g. prod+pin:prod,prod+pin:vecdot+pin")
    ap.add_argument("extra", nargs="*", help="args after -- go to the bench binary")
    return ap.parse_args()


def run_sweep(a):
    rng = random.Random(a.seed)
    threads = [int(t) for t in a.threads.split(",")]
    rows = []
    with open(a.out + ".stderr", "w") as errf:
        for r in range(a.rounds):
            order = threads[:]
            rng.shuffle(order)
            for t in order:
                env = dict(os.environ, GGML_SYCL_CPU_THREADS=str(t))
                p = subprocess.run([a.bin] + a.extra, env=env, capture_output=True, text=True)
                errf.write(f"## round {r} threads {t} rc={p.returncode}\n{p.stderr}")
                if p.returncode != 0:
                    sys.exit(f"bench failed (threads={t}, rc={p.returncode}): {p.stderr[-800:]}")
                for line in p.stdout.splitlines()[1:]:
                    rows.append([str(r)] + line.split(","))
                print(f"round {r} threads {t} done", file=sys.stderr)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HDR)
        w.writerows(rows)
    return rows


def pct(sorted_vals, q):
    return sorted_vals[int(q * (len(sorted_vals) - 1))]


def summarise(rows):
    recs = [dict(zip(HDR, r)) for r in rows]
    by_key = {}
    for x in recs:
        by_key.setdefault((x["shape"], x["mat"], x["type"], int(x["threads"]), x["variant"]), []).append(x)

    # per (round,burst): median per-call GB/s, and bytes/time over the burst's calls
    burst_med, burst_agg = {}, {}
    for key, xs in by_key.items():
        g, tot = {}, {}
        for x in xs:
            b = (x["round"], x["burst"])
            g.setdefault(b, []).append(float(x["gbps"]))
            by, us = tot.get(b, (0.0, 0.0))
            tot[b] = (by + float(x["bytes"]), us + float(x["us"]))
        burst_med[key] = {b: statistics.median(v) for b, v in g.items()}
        burst_agg[key] = {b: by / (us * 1e3) for b, (by, us) in tot.items()}

    print("shape,mat,type,threads,variant,best_gbps,p90_gbps,median_gbps,p10_gbps,min_gbps,burst_gbps,"
          "ratio_read,median_us,n")
    for key in sorted(by_key):
        xs = by_key[key]
        g = sorted(float(x["gbps"]) for x in xs)
        us = statistics.median(float(x["us"]) for x in xs)
        suffix = key[4][key[4].index("+"):] if "+" in key[4] else ""
        ratio = ""
        rk = None
        for cand in ("read" + suffix, "read+pin", "read"):  # the pinned controls fall back to the pinned read
            if key[:4] + (cand,) in burst_med:
                rk = key[:4] + (cand,)
                break
        if rk and key != rk:
            rs = [burst_med[key][b] / burst_med[rk][b] for b in burst_med[key] if b in burst_med[rk]]
            ratio = f"{statistics.median(rs):.2f}"
        bg = statistics.median(burst_agg[key].values())
        print(f"{key[0]},{key[1]},{key[2]},{key[3]},{key[4]},{g[-1]:.1f},{pct(g, 0.9):.1f},"
              f"{statistics.median(g):.1f},{pct(g, 0.1):.1f},{g[0]:.1f},{bg:.1f},{ratio},{us:.0f},{len(g)}")


def pair_ratios(rows, pairs):
    recs = [dict(zip(HDR, r)) for r in rows]
    cell = {}  # (shape,mat,type,threads,variant) -> {(round,burst): [gbps]}
    for x in recs:
        k = (x["shape"], x["mat"], x["type"], int(x["threads"]), x["variant"])
        cell.setdefault(k, {}).setdefault((x["round"], x["burst"]), []).append(float(x["gbps"]))
    med = {k: {b: statistics.median(v) for b, v in d.items()} for k, d in cell.items()}
    print("shape,mat,type,threads,pair,ratio_median,ratio_p25,ratio_p75,n_bursts")
    for k in sorted(med):
        for pr in pairs:
            va, vb = pr.split(":")
            if k[4] != va:
                continue
            kb = k[:4] + (vb,)
            if kb not in med:
                continue
            rs = sorted(med[k][b] / med[kb][b] for b in med[k] if b in med[kb])
            if rs:
                print(f"{k[0]},{k[1]},{k[2]},{k[3]},{pr},{statistics.median(rs):.2f},{pct(rs, 0.25):.2f},"
                      f"{pct(rs, 0.75):.2f},{len(rs)}")


def main():
    a = parse_args()
    if a.from_csv:
        with open(a.from_csv, newline="") as fh:
            rows = list(csv.reader(fh))[1:]
    else:
        rows = run_sweep(a)
    summarise(rows)
    if a.pairs:
        print()
        pair_ratios(rows, a.pairs.split(","))


if __name__ == "__main__":
    main()
