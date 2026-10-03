#!/usr/bin/env python3
"""Thread sweep for bench-cpu-expert-matvec (llama.cpp-cego).

The pool's TBB arena size is fixed once per process by GGML_SYCL_CPU_THREADS, so
a sweep is one process per thread count. Rounds are interleaved: every round
visits all thread counts in a fresh random order. Inside each process the
variants are already interleaved burst by burst (see the bench header).

    source /opt/intel/oneapi/setvars.sh --force
    ./run-expert-sweep.py --bin build/bin/bench-cpu-expert-matvec --rounds 3 \
        --out expert-sweep.csv -- --types q8_0,mxfp4 --shapes qwen38

Writes every raw call to --out and prints, per (shape,mat,type,threads,variant):
best / p90 / median / min GB/s of effective weight bytes per second pooled over
all rounds, and `ratio_read`: the median over (round,burst) of the variant's
burst-median GB/s divided by the `read` variant's burst-median in the SAME
burst. The ratio is the drift-proof number; absolute GB/s float with ambient load.
"""
import argparse, csv, os, random, statistics, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("--bin", required=True)
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--threads", default="1,2,4,8,12,16,20,24")
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--from-csv", help="re-summarise an existing --out CSV instead of running")
ap.add_argument("--pin-arm", action="store_true",
                help="also run every (round,threads) with --pin-tbb, as a second arm "
                     "(rows get variant suffix +pin), order randomised per round")
ap.add_argument("extra", nargs="*", help="args after -- go to the bench binary")
a = ap.parse_args()

threads = [int(t) for t in a.threads.split(",")]
rng = random.Random(a.seed)
rows = []
HDR = "round,shape,mat,type,N,K,k,threads,variant,burst,call,us,bytes,gbps,cpu_us".split(",")
for r in range(0 if a.from_csv else a.rounds):
    order = threads[:]
    rng.shuffle(order)
    for t in order:
        arms = [("", [])] + ([("+pin", ["--pin-tbb"])] if a.pin_arm else [])
        rng.shuffle(arms)
        for suffix, flags in arms:
            env = dict(os.environ, GGML_SYCL_CPU_THREADS=str(t))
            p = subprocess.run([a.bin] + a.extra + flags, env=env, capture_output=True, text=True)
            if p.returncode != 0:
                sys.exit(f"bench failed (threads={t}): {p.stderr[-500:]}")
            for line in p.stdout.splitlines()[1:]:
                f = line.split(",")
                f[7] += suffix  # variant column
                rows.append([str(r)] + f)
        print(f"round {r} threads {t} done", file=sys.stderr)

if a.from_csv:
    with open(a.from_csv, newline="") as fh:
        rows = list(csv.reader(fh))[1:]
else:
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HDR)
        w.writerows(rows)

recs = [dict(zip(HDR, r)) for r in rows]
by_key = {}
for x in recs:
    by_key.setdefault((x["shape"], x["mat"], x["type"], int(x["threads"]), x["variant"]), []).append(x)
# per (round,burst) median GB/s per variant, for the paired ratio
burst_med = {}
for key, xs in by_key.items():
    g = {}
    for x in xs:
        g.setdefault((x["round"], x["burst"]), []).append(float(x["gbps"]))
    burst_med[key] = {b: statistics.median(v) for b, v in g.items()}

print("shape,mat,type,threads,variant,best_gbps,p90_gbps,median_gbps,min_gbps,ratio_read,median_us,n")
for key in sorted(by_key):
    xs = by_key[key]
    g = sorted(float(x["gbps"]) for x in xs)
    us = statistics.median(float(x["us"]) for x in xs)
    rk = key[:4] + ("read" + ("+pin" if key[4].endswith("+pin") else ""),)
    ratio = ""
    if rk in burst_med:
        rs = [burst_med[key][b] / burst_med[rk][b] for b in burst_med[key] if b in burst_med[rk]]
        ratio = f"{statistics.median(rs):.2f}"
    p90 = g[int(0.9 * (len(g) - 1))]
    print(f"{key[0]},{key[1]},{key[2]},{key[3]},{key[4]},{g[-1]:.1f},{p90:.1f},{statistics.median(g):.1f},{g[0]:.1f},{ratio},{us:.0f},{len(g)}")
