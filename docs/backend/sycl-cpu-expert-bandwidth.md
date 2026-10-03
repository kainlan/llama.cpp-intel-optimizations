# CpuExpertPool bandwidth versus a STREAM baseline

Tracker: `llama.cpp-cego` (Strata lesson 9). Host: Core Ultra 7 270K Plus, 24
CPUs (P-cores = CPUs 0-7, E-cores = 8-23), AVX2 + AVX-VNNI + AVX-VNNI-INT8, **no
AVX-512**, 2 DDR5 channels. `dmidecode` reports 4 x 64 GB Micron DDR5, configured
speed 5600 MT/s (2 DIMMs per channel), so the theoretical peak is
5600 MT/s x 8 B x 2 channels = 89.6 GB/s. L2 40 MB, L3 36 MB. Measured
2026-10-03 under the permanent ambient load: 1-minute load average 45-70 for the
STREAM and 1-4 thread runs (read from `uptime` at the time; **no snapshot of
those runs was saved**), 32-34 for the 8-24 thread runs (five snapshots in the
data directory).

"Placement decides the executor" puts host-resident experts on the CPU, so the
CPU expert matvec is the decode path for every expert that does not fit in VRAM.
This document asks how close that path is to what the memory system can deliver
and what it would cost to close the gap without AVX-512.

**Status of the numbers.** Every table says how it was measured. Per-core
(1-thread) kernel numbers and 8-24 thread numbers are both measured; where a
statement is an inference or a hypothesis it says so. Nothing here is an
end-to-end token rate.

## Tools (host-only, built on demand, no GPU)

| tool | what it does |
|------|--------------|
| `tools/cpu-expert-bench/bench-host-stream.cpp` | STREAM-style copy/scale/triad plus a pure-read kernel, 3 GiB total (three 1 GiB arrays, far above the 36 MB L3), threads 1..24, optional pinning, best and median of N 20-40 ms trials. No SYCL or ggml dependency; target `bench-host-stream` in any x86-64 configuration (AVX2 kernels), or plain `g++ -O3 -mavx2 -std=c++17 -pthread`. |
| `tools/cpu-expert-bench/bench-cpu-expert-matvec.cpp` | `prod` calls the real `ggml_sycl_cpu_expert_mul_mat_batched()` (the function a CpuExpertPool worker runs) on expert-shaped weights at batch 1. Other variants run the same cold weights through an in-process thread team: `read` (loads only), `vecdot` (ggml-cpu row `vec_dot`), `r8` (ggml's 8x8 repack gemv), `mx8`/`mx16`, `q2`, `q8_4row[_pf]` (prototypes). Each variant takes a pin suffix (`+pin`, `+pinE`, `+pinS`). Opens no SYCL queue. |
| `tools/cpu-expert-bench/run_expert_sweep.py` | One process per thread count (`GGML_SYCL_CPU_THREADS` is read once), rounds shuffled, summary with best/p90/median/p10/min, burst aggregate and paired `ratio_read`; `--pairs A:B,...` adds per-burst paired ratios of any two variants (median and interquartile range), `--from-csv` re-summarises a saved run; saves each run's stderr (oracle, pin and CPU samples). |

`bench-cpu-expert-matvec` is defined in `ggml/src/ggml-sycl/CMakeLists.txt`, next
to `test-sycl-cpu-dispatch` (it needs that scope's `GGML_SYCL_DNNL` and links the
static `ggml-sycl`), so it exists only with `-DGGML_SYCL=ON
-DGGML_BACKEND_DL=OFF -DLLAMA_BUILD_TESTS=ON` (it sits inside `BUILD_TESTING`).
`bench-host-stream` is in `tools/cpu-expert-bench/CMakeLists.txt` and needs none
of that beyond an x86-64 target. The matvec target is compiled with `-mavx2 -mfma -mf16c -mavxvnni
-mavxvnniint8`, as `cpu-dispatch.cpp` is (minus F16C); at start it checks cpuid
for all five and exits 77 ("SKIP") instead of faulting on a CPU without them.

```bash
source /opt/intel/oneapi/setvars.sh --force
./scripts/sycl-build.sh bench-cpu-expert-matvec bench-host-stream
# STREAM, pinned (thread i on CPU i), dynamic scheduling:
build/bin/bench-host-stream --sched dynamic --rounds 15 --threads 1,2,4,8,12,16,20,22,24 \
    --pin $(seq -s, 0 23) --pin-label pinned
# expert matvec; oracle first (non-zero exit = FAIL; --oracle-selftest is the positive control):
GGML_SYCL_CPU_THREADS=2 build/bin/bench-cpu-expert-matvec --oracle-only --oracle-selftest \
    --types q8_0,mxfp4,iq4_nl,iq3_xxs,q2_0 --shapes qwen38,gptoss --variants prod,vecdot,mx8,mx16,r8,q2,q8_4row
# the per-core table below (1/2/4 threads):
python3 tools/cpu-expert-bench/run_expert_sweep.py --bin build/bin/bench-cpu-expert-matvec \
    --rounds 3 --threads 1,2,4 --out low-qwen.csv -- \
    --types q8_0,mxfp4,iq4_nl,iq3_xxs,q2_0 --shapes qwen38 --mats gate,down \
    --variants prod,prod+pin,vecdot,vecdot+pin,read,read+pin,r8,r8+pin,mx8,mx8+pin,mx16,q2,q2+pin,q8_4row,q8_4row_pf \
    --bursts 8 --burst-calls 3 --pool-mb 256
```

The 8-24 thread runs, the paired ratios and the top-k runs (one process at a time;
`V` is the variant list, `S`/`B` the script and binary):

```bash
V=prod,prod+pin,prod+pinE,prod+pinS,vecdot,vecdot+pin,read,read+pin,r8,r8+pin,mx8,mx8+pin,mx16,mx16+pin,q2,q2+pin,q8_4row,q8_4row_pf
S=tools/cpu-expert-bench/run_expert_sweep.py; B=build/bin/bench-cpu-expert-matvec
python3 $S --bin $B --rounds 3 --threads 8,12,16,20,22,24 --out hi-qwen.csv -- \
    --types q8_0,mxfp4,iq4_nl,iq3_xxs,q2_0 --shapes qwen38 --mats gate,down --variants $V \
    --bursts 8 --burst-calls 3 --pool-mb 256 > hi-qwen.summary 2> hi-qwen.log
python3 $S --bin $B --rounds 3 --threads 8,12,16,20,22,24 --out hi-gptoss.csv -- \
    --types q8_0,mxfp4 --shapes gptoss --mats gate,down --variants $V \
    --bursts 8 --burst-calls 3 --pool-mb 256 > hi-gptoss.summary 2> hi-gptoss.log
for k in 1 2 3; do
  python3 $S --bin $B --rounds 3 --threads 22 --out topk$k.csv -- \
      --types q8_0,mxfp4,q2_0 --shapes qwen38 --mats gate --variants prod,prod+pin,read,read+pin \
      --bursts 16 --burst-calls 4 --pool-mb 256 --topk $k > topk$k.summary 2> topk$k.log
done
# paired ratios (pairs-8-24-threads.csv) from a saved raw CSV, once per CSV (hi-qwen.csv and hi-gptoss.csv);
# the pair names are those in that file:
for c in hi-qwen hi-gptoss; do python3 $S --from-csv $c.csv --pairs prod+pin:prod,prod+pinE:prod,prod+pinS:prod,prod:vecdot,prod+pin:vecdot+pin,prod+pin:read,prod+pin:read+pin,vecdot+pin:vecdot,read+pin:read,r8+pin:prod+pin,mx8+pin:prod+pin,q2+pin:prod+pin,mx8:vecdot,q2:prod; done
```

The raw call CSVs are not committed (large); the summaries, pairs, pin-sample totals
and a digest of the stderr are (see Data), so the committed files reproduce every
number in this document except the per-call distributions.

Shapes (experts, batch 1, one MUL_MAT_ID worth of rows per call): GPT-OSS 20B =
2880x2880 gate/up/down, MXFP4, top-4; Qwen3.8-Flash-Next = hidden 2560,
intermediate 640 (gate/up N=640 K=2560, down N=2560 K=640), top-10. The
GSQ-RCO IQ3_XXS Qwen file is a mixed allocation, so IQ3_XXS is measured on
gate/up only (K=640 is not a multiple of 256, which IQ3_XXS needs), and Q2_0 /
IQ4_NL stand in for the `down` tensors. The weight pool is 256 MB per type,
rotated so every call reads cold data. The pool's expert slots are copies of one
quantised expert.

## Method, and why the absolute numbers are soft

Absolute GB/s float by about 2x with ambient load, between and within runs, and
the per-call distribution has a heavy low tail (a Qwen gate call is ~1 ms and a
scheduler stall of a few ms moves it a long way). So:

* variants, including the pinned and unpinned arms of the same code, are
  interleaved burst by burst in random order inside one process: a pin arm and
  an unpin arm see the same host state;
* every call reads cold weights; timing is on the call, not the first thread;
* the paired `ratio_read` (variant burst median / `read` burst median in the same
  burst) and the burst-aggregate GB/s are the comparable numbers; tables give
  **median, best and p10** and say which;
* **medians are compared with medians.** A best of sub-millisecond calls and a
  best of 20-40 ms STREAM trials sample different windows of the load and are
  not comparable; `prod` bests above the STREAM best say nothing about the
  ceiling.

`prod` per call also quantises the activation to the vec_dot type and looks it
up in an activation map, a few microseconds against calls of 100 us to several
ms; the other variants take a pre-quantised activation.

**Oracle.** `prod` could silently skip work (`cpu-dispatch.cpp:996-1010` leaves a
task with no activation entry unquantised, and :1202 skips a task whose type has
no `cpu_traits`), which would read as an absurdly high GB/s. Every
config therefore runs `prod` and each producing variant once over NaN-filled
outputs and compares every row of every expert to ggml-cpu `vec_dot` on the same
weights and activation (tolerance 1e-3 of the largest reference value); a NaN or
a mismatch fails the run with exit 3. `--oracle-selftest` corrupts one run (one
row NaN, one row off by 1.0) and requires the oracle to reject it; it did, for
all five types, so the check can fail. All 13 configs of the 1-4 thread sweep
passed.

## STREAM baseline (GB/s, best / median over 15 trials)

Cells are best / median: *best* is the fastest single trial of either of the 2
runs (the maximum of the 2 runs' bests), *median* is the mean of the 2 runs'
medians over their 15 trials. STREAM convention for copy/scale/triad (bytes read
+ written). `read` = weight bytes read, which is what a matvec moves. Pinned = thread i on CPU i (P-cores
first). Raw rows: `sycl-cpu-expert-bandwidth-data/stream.csv`.

Pinned:

| kernel | 1 | 2 | 4 | 8 | 12 | 16 | 20 | 22 | 24 threads |
|---|---|---|---|---|---|---|---|---|---|
| copy  | 21/12 | 32/18 | 30/26 | 40/32 | 37/33 | 41/35 | 48/37 | 47/37 | 44/37 |
| scale | 24/12 | 33/18 | 29/26 | 38/31 | 41/32 | 42/35 | 46/37 | 47/38 | 49/38 |
| triad | 26/14 | 32/22 | 34/30 | 44/37 | 46/39 | 52/41 | 55/43 | 52/43 | 56/43 |
| read  | 26/13 | 25/23 | 40/33 | 52/40 | 56/42 | 62/45 | 68/45 | 69/46 | 54/48 |

Unpinned (scheduler free to migrate):

| kernel | 1 | 2 | 4 | 8 | 12 | 16 | 20 | 22 | 24 threads |
|---|---|---|---|---|---|---|---|---|---|
| copy  | 17/12 | 22/17 | 35/22 | 36/27 | 42/29 | 44/31 | 46/32 | 48/34 | 47/34 |
| scale | 13/12 | 25/16 | 31/22 | 38/26 | 43/31 | 43/30 | 47/32 | 47/33 | 44/34 |
| triad | 14/13 | 33/18 | 42/25 | 44/32 | 48/35 | 52/37 | 51/39 | 54/39 | 53/39 |
| read  | 14/13 | 33/21 | 38/26 | 52/32 | 58/37 | 62/37 | 69/40 | 70/42 | 68/42 |

The read median is **45-48 GB/s at 16-24 pinned threads** (about half the 89.6
GB/s peak), and flat from ~16 threads (the median rises 40 -> 42 -> 45 -> 46 ->
48 from 8 to 24 threads; the read best rises 52 -> 69 from 8 to 22 threads and
is 54 at 24, a window of 20-40 ms that does not compare with sub-millisecond
calls, see Method). One core pulls ~13 GB/s
median. Pinning is worth +10-20 % in STREAM already (pinned read median 45-48 vs
37-42 unpinned at 16-24 threads), which matches the +9-20 % the own-team `read`
shows below. Whether the expert matvec is flat past ~16 threads is a separate
question, answered by the next section (roughly yes, within noise: see there).

## The expert matvec (`prod`) against that baseline

### Per-core view (1 thread)

Effective weight GB/s, Qwen3.8 gate shape (N=640 K=2560, k=10 experts per call),
median (best) over 3 rounds x 8 bursts x 3 calls of the interleaved 1-thread run.
No arena scheduling and no pinning effect worth separating at one thread
(`prod+pin` and `prod` agree except where a load tail intervenes: Qwen IQ4_NL
gate `prod` 3.0 against `prod+pin` 6.8 median, both with best 9.6, and GPT-OSS
Q8_0 down `prod` 7.4 against `prod+pin` 5.7, best 11 for both, are such tail
artifacts): this isolates the kernel. A single core pulls 14-16 GB/s median. The
p10 of every cell is 2-5 GB/s (load tail); ratios are the better comparison.
Raw: `sycl-cpu-expert-bandwidth-data/expert-1-2-4-threads.summary.csv`.

| type | `read` | `prod` | ggml `vecdot` | ggml 8x8 repack (`r8`) | `mx8` | `mx16` | prototype `q2` | `q8_4row` / `_pf` | `prod` / `read` (paired `ratio_read`, not a ratio of the medians) |
|---|---|---|---|---|---|---|---|---|---|
| Q8_0   | 15.5 (19.5) | 9.1 (12.9) | 9.0 (12.7) | -- | -- | -- | -- | 5.8 (11.4) / 9.6 (12.5) | 0.58 |
| MXFP4  | 14.5 (18.3) | 2.3 (4.8)  | 7.6 (9.9)  | 8.6 (12.5) | 8.0 (9.8) | 1.8 (4.9) | -- | -- | 0.17 |
| IQ4_NL | 13.8 (19.5) | 3.0 (9.6)  | 5.2 (9.5)  | 8.3 (12.7) | -- | -- | -- | -- | 0.29 |
| IQ3_XXS| 13.9 (16.2) | 5.4 (6.0)  | 5.3 (6.2)  | -- | -- | -- | -- | -- | 0.39 |
| Q2_0   | 14.1 (17.5) | 3.5 (3.6)  | 3.5 (3.6)  | -- | -- | -- | 6.1 (7.2) | -- | 0.24 |

GPT-OSS gate (N=K=2880): MXFP4 `prod` 3.7 (5.1), `vecdot` 7.5 (9.8), `mx8` 7.5
(9.8), `mx16` 3.2 (4.1), `r8` 8.4 (11.0), `read` 12.7 (15.7); Q8_0 `prod` 8.8
(10.9), `vecdot` 8.9 (10.3), `read` 14.1 (16.4).

### `prod` against thread count

Median (best) GB/s of weight bytes. `prod` = production as built (arena workers
free to migrate); `prod+pin` = the same code with the arena workers pinned (main
on CPU 0, active workers on CPUs 1.., so P-cores first). 1/2/4 threads: the sweep
above (3 rounds x 8 bursts x 3 calls); 8-24 threads: the high-thread sweep (3
rounds x 8 bursts x 3 calls, one process per thread count, variants interleaved
burst by burst). Under the ambient load of the snapshots below, the p10 of every
cell is 2-10 GB/s and the best is a ceiling-touching outlier; compare medians.
All 13 configs of the high-thread sweep passed the oracle (exit 0; IQ3_XXS
`down` is skipped, K=640 is not a multiple of 256).

| config | variant | 1 | 2 | 4 | 8 | 12 | 16 | 20 | 22 | 24 threads |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen Q8_0 gate | `prod` | 9.1 (13) | 5.5 (18) | 9.2 (16) | 20 (35) | 21 (46) | 25 (43) | 22 (58) | 28 (58) | 32 (66) |
| Qwen Q8_0 gate | `prod+pin` | 8.9 (12) | 4.6 (17) | 6.0 (19) | 22 (37) | 25 (48) | 23 (50) | 23 (62) | 30 (67) | 28 (71) |
| Qwen Q8_0 down | `prod` | 9.1 (14) | 9.2 (24) | 9.2 (24) | 19 (30) | 24 (41) | 20 (46) | 33 (58) | 43 (71) | 23 (58) |
| Qwen Q8_0 down | `prod+pin` | 8.9 (13) | 11 (23) | 10 (22) | 18 (44) | 19 (47) | 21 (53) | 30 (63) | 47 (75) | 33 (74) |
| GPT-OSS Q8_0 gate | `prod` | 8.8 (11) | 11 (19) | 12 (26) | 23 (34) | 30 (48) | 30 (52) | 34 (49) | 29 (51) | 34 (63) |
| GPT-OSS Q8_0 gate | `prod+pin` | 9.0 (11) | 14 (20) | 15 (32) | 25 (37) | 31 (47) | 27 (47) | 28 (62) | 25 (69) | 34 (66) |
| GPT-OSS Q8_0 down | `prod` | 7.4 (11) | 11 (21) | 11 (22) | 27 (36) | 31 (74) | 28 (49) | 34 (60) | 31 (59) | 26 (55) |
| GPT-OSS Q8_0 down | `prod+pin` | 5.7 (11) | 14 (21) | 12 (28) | 30 (39) | 32 (72) | 29 (48) | 36 (55) | 34 (61) | 32 (64) |
| Qwen MXFP4 gate | `prod` | 2.3 (4.8) | 3.0 (6.5) | 3.2 (7.4) | 8.9 (20) | 12 (30) | 12 (37) | 15 (44) | 18 (47) | 21 (41) |
| Qwen MXFP4 gate | `prod+pin` | 2.2 (4.8) | 3.1 (6.5) | 3.5 (8.8) | 7.6 (19) | 10 (30) | 8.8 (39) | 18 (45) | 20 (48) | 25 (67) |
| Qwen MXFP4 down | `prod` | 3.8 (5.3) | 4.2 (13) | 4.1 (9.5) | 10 (21) | 13 (36) | 11 (33) | 20 (45) | 30 (63) | 20 (58) |
| Qwen MXFP4 down | `prod+pin` | 3.9 (5.6) | 4.7 (14) | 4.4 (12) | 10 (21) | 14 (36) | 13 (44) | 21 (56) | 30 (69) | 18 (53) |
| GPT-OSS MXFP4 gate | `prod` | 3.7 (5.1) | 4.3 (8.3) | 4.6 (10) | 13 (21) | 20 (54) | 17 (36) | 20 (48) | 22 (44) | 20 (47) |
| GPT-OSS MXFP4 gate | `prod+pin` | 3.7 (5.5) | 4.6 (8.4) | 5.2 (12) | 14 (22) | 22 (48) | 16 (36) | 19 (54) | 21 (62) | 19 (65) |
| GPT-OSS MXFP4 down | `prod` | 3.9 (6.0) | 3.2 (7.1) | 5.0 (14) | 13 (26) | 24 (52) | 16 (32) | 16 (45) | 16 (44) | 14 (40) |
| GPT-OSS MXFP4 down | `prod+pin` | 3.8 (6.8) | 3.6 (8.0) | 6.4 (11) | 13 (25) | 21 (54) | 13 (33) | 22 (57) | 14 (54) | 20 (62) |
| Qwen IQ4_NL gate | `prod` | 3.0 (9.6) | 6.6 (14) | 7.6 (16) | 16 (29) | 21 (43) | 18 (45) | 24 (54) | 28 (57) | 20 (67) |
| Qwen IQ4_NL gate | `prod+pin` | 6.8 (9.6) | 3.7 (14) | 7.8 (17) | 16 (40) | 21 (47) | 18 (54) | 21 (57) | 25 (69) | 25 (63) |
| Qwen IQ4_NL down | `prod` | 6.6 (9.6) | 8.6 (15) | 7.3 (16) | 16 (30) | 18 (43) | 20 (44) | 18 (50) | 22 (60) | 19 (49) |
| Qwen IQ4_NL down | `prod+pin` | 6.8 (9.5) | 9.6 (17) | 7.4 (17) | 18 (40) | 24 (50) | 20 (51) | 20 (59) | 32 (67) | 30 (72) |
| Qwen IQ3_XXS gate | `prod` | 5.4 (6.0) | 5.2 (11) | 5.2 (10) | 11 (30) | 16 (39) | 19 (42) | 15 (56) | 25 (52) | 20 (54) |
| Qwen IQ3_XXS gate | `prod+pin` | 5.3 (6.1) | 5.4 (11) | 4.7 (10) | 9.7 (24) | 16 (37) | 17 (42) | 14 (53) | 21 (54) | 17 (62) |
| Qwen Q2_0 gate | `prod` | 3.5 (3.6) | 3.3 (6.2) | 1.7 (6.6) | 8.5 (17) | 12 (27) | 10 (27) | 14 (40) | 16 (38) | 13 (48) |
| Qwen Q2_0 gate | `prod+pin` | 3.5 (3.6) | 3.4 (6.5) | 3.3 (9.3) | 8.8 (25) | 14 (26) | 7.9 (35) | 13 (40) | 16 (50) | 13 (52) |
| Qwen Q2_0 down | `prod` | 3.2 (3.6) | 3.3 (6.2) | 3.4 (8.4) | 8.6 (18) | 11 (26) | 9.9 (35) | 13 (33) | 17 (51) | 11 (39) |
| Qwen Q2_0 down | `prod+pin` | 3.2 (3.6) | 3.4 (7.0) | 3.3 (11) | 7.8 (25) | 9.5 (27) | 9.8 (35) | 18 (41) | 22 (51) | 11 (50) |
| pinned STREAM `read` median | -- | 13 | 23 | 33 | 40 | 42 | 45 | 45 | 46 | 48 |

Every config and thread count is in the data directory
(`expert-8-24-threads-qwen.summary.csv`, `expert-8-24-threads-gptoss.summary.csv`).
`prod+pin` at 22 threads is 14.4-46.7 GB/s median over the 13 configs, i.e.
**31-102 % of the pinned STREAM read median** (46): Qwen Q8_0 down is the one
config above the 37 GB/s Strata-equivalent target (46.7); GPT-OSS Q8_0 down
(33.6), Qwen IQ4_NL down (31.6), Qwen Q8_0 gate (30.5) and Qwen MXFP4 down
(30.1) are next, and the lowest are MXFP4 (14-30, GPT-OSS down 14.4) and Q2_0
(16-22). `prod` unpinned at 22 threads is 16-43.

**Scaling.** The pooled median over the 13 configs of `prod` is 13.3 / 19.9 /
18.2 / 19.9 / 24.9 / 19.8 GB/s at 8 / 12 / 16 / 20 / 22 / 24 threads
(`prod+pin`: 13.8 / 21.0 / 17.0 / 21.2 / 25.0 / 24.8). `prod` rises 1.4-1.9x
from 8 to 12-22 threads (pooled medians) and is flat within noise from 12 on; 24 threads
is above 22 in only 3 of the 13 `prod` configs. Each thread count is a separate
process at a different moment of the ambient load, so single cells jump by up to
2x between neighbours (Qwen Q8_0 down `prod`: 43 at 22 threads, 23 at 24);
neither a plateau nor a rise is established beyond that. Neither is `prod` close
to STREAM at any thread count: the ratio to STREAM is what the next sections
explain.

**Conditions.** The 8-24 thread runs and the pair ratios below ran under the
host's ambient load, not a quiet host. Five snapshots (`uptime` and `top -b -n1 |
head -15`, taken before each block): 1-minute load average 32.0-34.3 (5-minute
35.9-37.8, 15-minute 39.7-40.9); CPU 4-10 % user, 1-8 % system, **84-91 % nice,
0-5 % idle**; a codescout re-index at 1950-2217 % CPU, plus Emby, the ws2022ci VM
and Frigate. The files are `load-qwen.txt`, `load-gptoss.txt`,
`load-topk{1,2,3}.txt`. The model capture that would have corrupted the sweep
had finished. Absolute numbers are depressed and noisy; the pairs are not. The
STREAM table and the 1-4 thread runs ran earlier at a 1-minute load of 45-70
(no snapshot was saved).

#### Correction: the earlier "pinning is worth 1.4-3.5x" is not reproduced

The earlier measurement behind "pinning the arena workers one per core, P-cores
first, is worth 1.4-1.7x for Q8_0 and 2-3.5x for MXFP4 at 20-24 threads"
(`prod` 10-23 GB/s unpinned vs 19-37 pinned) used a separate process per arm
(the arms did not share host state), found the workers by thread-id order,
compared against an unpinned own-team `read`, and ran at load ~70. In the burst-paired, in-process
measurement at load ~33 the effect is **not there for `prod`**. Median over the 13
configs of the per-config median of (arm / `prod`) in the same burst, 24 bursts
per config per thread count:

| paired ratio (arm : base) | 8 | 12 | 16 | 20 | 22 | 24 threads |
|---|---|---|---|---|---|---|
| `prod+pin` : `prod` (P-first, 1 worker per CPU) | 1.04 | 1.03 | 0.96 | 1.04 | 1.01 | 1.11 |
| same, excluding the 8 over-counted config-runs (below) | 1.12 | 1.03 | 0.95 | 1.04 | 1.01 | 1.11 |
| `prod+pinS` : `prod` (fixed shuffled order, all CPUs) | 1.00 | 0.99 | 1.02 | 1.03 | 1.00 | 1.04 |
| `prod+pinE` : `prod` (E-cores only, control) | 0.97 | 0.97 | 1.00 | 1.08 | 0.90 | 0.86 |
| `vecdot+pin` : `vecdot` (own team) | 1.34 | 1.28 | 1.38 | 1.22 | 1.09 | 1.09 |
| `read+pin` : `read` (own team) | 1.20 | 1.16 | 1.18 | 1.13 | 1.09 | 1.10 |

The pooled `prod+pin : prod` ratio is **0.95-1.12** (1.12 at 8 and 1.11 at 24
threads after the exclusion below, 0.95-1.04 elsewhere, i.e. within about 10 % of
1.0), a shuffled pin does as well as
the P-first pin, and the E-core-only pin matches up to 16 threads (0.97-1.00;
see the next section for 20-24). By contrast the
same pin applied to the in-process own team is worth 1.1-1.4x. So the pin itself
works (below) and helps a persistent team, and for `prod` it is not what limits
throughput. Per type the `prod+pin : prod` median ratio at 22 threads is 0.99
(Q8_0), 1.06 (MXFP4), 1.11 (IQ4_NL), 0.99 (Q2_0), 0.99 (IQ3_XXS); the
interquartile range of the per-burst `prod+pin : prod` ratio contains 1.0 in all
78 cases (13 configs x 6 thread counts, 24 bursts each; the `--pairs` tables are
in the data directory). I cannot say pinning never helps:
the old observation was made at a heavier load (~70) and this one at ~33; whether
the benefit appears at heavier load was not tested.

### Pinning: mechanism evidence

All in-process, paired per burst. `prod+pin`, `prod+pinE` and `prod+pinS` act on
the arena workers, identified by on-CPU time (schedstat ns, not by
thread-id order); `vecdot+pin`/`read+pin` act on the own team. The pinned-worker
count and the last-run CPU of every pinned worker are printed (`cpus ...:
last-run CPU on P-cores=.. E-cores=..`; "last-run" because a worker that has not
run since the affinity change still reports its old CPU). Summed over all 39
config-runs of the two high-thread sweeps (`*.csv.stderr`):

| threads | `prod+pin` P / E samples | `prod+pinE` P / E | `prod+pinS` P / E |
|---|---|---|---|
| 8 (see the over-count below) | 2462 / 250  | 164 / 2548 | 1073 / 1639 |
| 16 | 2462 / 2594 | 33 / 5023  | 1598 / 3458 |
| 22 | 2471 / 4393 | 41 / 6823  | 2468 / 4396 |
| 24 | 2482 / 5006 | 46 / 7442  | 2487 / 5001 |

P-first pin puts main and the first 7 workers on the 8 P-cores (CPUs 0-7) and the
rest on E-cores as intended (the P count is flat at ~2470 because only 8 P-cores
exist); the E-only pin keeps the samples on E-cores (the few P samples are, as
an assumption not tested without a new run, workers sampled before their
affinity took effect); the shuffled pin lands in between. The E-only order is
the 16 E-core CPUs 8-23 with worker i on slot (i+1) % 16, so at 20, 22 and 24
threads it wraps and puts two threads on some E-cores: its 1.08 / 0.90 / 0.86
at those counts is an oversubscribed control, not a like-for-like one (at 8-16
threads no CPU is shared). At 22 and
24 threads P-first and shuffled pin occupy (almost) the same set of CPUs and
differ only in which worker is on which core, and they measure the same.
The pin is in effect (with the over-count caveat below); it just does not move
`prod`. At 22 threads for the large
calls 21 active arena workers were found and pinned (IQ3_XXS: 21 found, 23 other
idle threads left alone); in the `--topk 1` runs the number found varies by
round and type from 4 to 21 (`run-digest-8-24-threads.txt`; a 640-row call has
at most 10 chunks at the 64-row grain floor).

**Worker over-count in 8 of the 234 config-runs (found after the sweep).** The
discovery counts every thread of the process that burned CPU during 12 `prod`
calls, but the arena has at most `threads - 1` workers besides the calling
thread. In 8 config-runs it found more: Qwen Q8_0 down 8 threads round 0 (14
found), Qwen MXFP4 down 12 threads round 1 (17), GPT-OSS Q8_0 gate 8 threads
round 0 (14), GPT-OSS MXFP4 down 8 threads rounds 0 and 2 (14, 11), GPT-OSS
MXFP4 gate 8 threads round 1 (9) and 16 threads round 1 (23), GPT-OSS Q8_0 down
12 threads round 1 (15). Those counts include non-arena threads or workers
replaced during the window, so `prod+pin` there pinned extra threads onto shared
CPUs. That explains the 250 E-core samples in the 8-thread `prod+pin` cell above:
220 of them come from the five over-counted 8-thread runs; the other 34 runs
have 2146 P / 30 E. Recomputing the pooled paired ratios without those 8
config-runs (the rounds are dropped from those configs only; recomputed from
the raw per-call CSVs, which are not committed; the result is
`pin-discovery-overcount.csv`) changes two `prod+pin : prod` cells: 8 threads
1.04 -> 1.12 and 16 threads 0.96 -> 0.95; 12, 20, 22 and 24 threads are
unchanged, and so are the E-only control, `prod : vecdot` and `prod+pin :
vecdot+pin`, except the shuffled control at 8 threads (`prod+pinS : prod` 1.00 ->
1.01), all to two decimals. None of the excluded values (1.12, 0.95, the
per-run values) nor the 220-of-250 and "34 runs, 2146 P / 30 E" sample split can
be reproduced from any other committed file: the pairs file is pooled over
rounds, the digest has no per-run samples and `pin-samples-8-24-threads.csv`
has only totals. They come from the raw per-call CSVs and stderr (not
committed); the committed record is `pin-discovery-overcount.csv`. The conclusion stands (no robust benefit of pinning
`prod` above 8 threads; at most ~1.1x anywhere), with the 8-thread figure
higher than first reported. The bench now caps discovery at `threads - 1`
(busiest first) and logs when it does; the committed data predate that.

### Why `prod` loses to the same kernel in a persistent team

The paired `prod : vecdot` ratio runs the production dispatch against ggml-cpu's
row `vec_dot` in the in-process dynamic team. For Q8_0, IQ4_NL, IQ3_XXS and Q2_0
production calls the same ggml-cpu kernel (Q2_0's is the scalar generic one), so
the ratio is **dispatch path only**. Median over configs, per type, 8 / 12 / 16 /
20 / 22 / 24 threads:

| type (kernel identical) | `prod : vecdot` | `prod+pin : vecdot+pin` |
|---|---|---|
| Q8_0   (n=4 configs) | 0.95 / 0.86 / 0.81 / 0.87 / 0.76 / 0.74 | 0.74 / 0.71 / 0.56 / 0.65 / 0.73 / 0.59 |
| IQ4_NL (n=2)         | 0.96 / 0.97 / 0.81 / 0.80 / 0.83 / 0.66 | 0.73 / 0.84 / 0.68 / 0.71 / 0.89 / 0.99 |
| Q2_0   (n=2)         | 0.83 / 0.90 / 0.86 / 0.93 / 0.84 / 0.84 | 0.62 / 0.74 / 0.65 / 1.06 / 0.99 / 0.70 |
| IQ3_XXS (n=1)        | 0.70 / 0.85 / 1.00 / 0.50 / 0.87 / 0.67 | 0.42 / 0.69 / 0.72 / 0.65 / 0.72 / 0.63 |
| MXFP4 (n=4; kernel differs: 16-row tile vs ggml VNNI row) | 0.66 / 0.69 / 0.58 / 0.54 / 0.62 / 0.56 | 0.43 / 0.56 / 0.39 / 0.54 / 0.59 / 0.57 |

So for the same instructions, production reaches 0.50-1.00 of a dynamic
team unpinned (0.66-0.97 for Q8_0, IQ4_NL and Q2_0; IQ3_XXS, one config, gives
both extremes, 0.50 at 20 and 1.00 at 16 threads) and 0.42-1.06 of a *pinned*
dynamic team (Q8_0 0.56-0.74, IQ4_NL 0.68-0.99, Q2_0 0.62-1.06 with 1.06 at 20
threads, IQ3_XXS 0.42-0.72). That also revises the
earlier Q8_0 statement that its headroom is "per-call overhead": it is, but
the overhead is path-level, up to 25-45 % of throughput at 8-24 threads, not a
few microseconds of activation quantisation (which the bench `prod` call also
includes).

What differs, as facts (`cpu-dispatch.cpp:1166-1169`, `bench-cpu-expert-matvec.cpp`
`team`): production runs a TBB `parallel_for` over `total_rows` with grain
`max(64, total_rows / (2 * threads))`, so roughly two chunks per thread, 145 rows
for a Qwen gate call at 22 threads (~0.4 MB of weights per chunk) and never fewer
than 64 rows; the in-process team pulls **16-row** chunks off an atomic counter,
~9x finer. With P- and E-cores (E-cores are slower per core under the same
load) a 2-chunks-per-thread split ends with the P-cores idle while one slow chunk
finishes; a 16-row dynamic split does not. **This is a hypothesis for the gap, not
a measured cause**: I did not change production grain, and TBB wake-up latency
and the activation-quantise/map lookup are other differences I did not isolate.
The experiment that would decide it is a grain sweep of the production call
(`--chunk` already exists in my team), not a kernel change.

The same applies per kernel: at 22 threads `mx16+pin` (the production 16-row tile
shape in the pinned team) reaches 29/41/40/35 GB/s median on Qwen gate / Qwen
down / GPT-OSS gate / GPT-OSS down against `prod+pin` 20/30/21/14, i.e. 1.4-2.5x
with the **same tile shape**. (`mx16` is not byte-identical to production; see
the kernel section.)


## Bottleneck from the kernel inner loops

* **Q8_0** (`ggml_vec_dot_q8_0_q8_0`, x86 arch file): with a `GGML_NATIVE` build
  that defines `__AVXVNNIINT8__` (this host; **not** a `GGML_CPU_ALL_VARIANTS`
  build) it uses `vpdpbssd` and the fp16 table; one 34-byte block is two 32-byte
  loads, one dot, one FMA. 58-61 % of one-core read at 1T. A hand 4-row version
  with independent accumulators is no better (`q8_4row` 55 %), and a 512 B
  prefetch only matches `vecdot` (56-61 %). The kernel is at the ceiling its
  access pattern allows; the remaining headroom is **dispatch**, not the dot
  loop: at 22 threads production Q8_0 is 0.76 of the same `vec_dot` in a dynamic
  team (see "Why `prod` loses ..." above), and the pinned team's `vecdot+pin` is
  44.9 GB/s median against 50.5 for the pure read (Qwen Q8_0 gate).
* **MXFP4, production 16-row kernel** (`simd_mxfp4_q8_0_16row`): objdump of
  `cpu-dispatch.cpp.o` (icpx -O3, Release) shows the `for r<16` loop is **not
  unrolled** (3 `vpdpbssd` in the function, not 48): the 16 accumulators are a
  **stack array**, each row paying a `vfmadd213ps (%rsp,%r11,4)` load and a
  `vmovups %ymm,(%rsp,%r11,4)` store per 2-block step, the 16 row pointers are
  reloaded from the stack (`mov 0x200(%rsp,%r11,1),%rbx`), and the e8m0 scale is
  ~12 scalar instructions (shift, cmp, cmov) per row per block. "Accumulators in
  memory" is the accurate wording, not "register spill". My out-of-line
  `mxfp4_rows<16>` compiles the same way (1 `vpdpbssd`, accumulators on the
  stack) and measures the same as `prod` (1T Qwen gate, paired `ratio_read` 0.16
  vs 0.17; medians 1.8 vs 2.3 GB/s), while `mxfp4_rows<8>` unrolls (8
  `vpdpbssd`, accumulators in registers) and measures like ggml's one-row kernel
  (`ratio_read` 0.53 vs 0.53; medians 8.0 vs 7.6). So the A/B supports the tile
  shape, not only the inference: at one thread the 8-row tile is 1.6-3.5x the
  16-row production kernel by medians (8.0 / 5.9 / 7.5 against 2.3 / 3.8 / 3.7
  GB/s on Qwen gate, Qwen down, GPT-OSS gate). (My `mx16` is not byte-identical to the production kernel: no 2-block
  unroll and F16C for the activation scale instead of the table.)
* **Q2_0** (64 weights / 18 bytes): ggml-cpu has only the scalar
  `ggml_vec_dot_q2_0_q8_0_generic` on x86, 3.2-3.5 GB/s median per core. My `q2` prototype
  is **single-row with no row tiling**: per 64 weights and one row it does one
  16 B weight load, 3 shifts, 4 ands, 4 `vpdpbusd`, and ~6 activation loads (4
  planes, the ysum, the scale). Those activation loads are identical for every
  row, so a multi-row tile (4 rows share one set of plane loads) would cut loads
  per row from ~7 to ~2.5. It reaches 6.1-6.6 GB/s median per core (1.7-2.1x the scalar median, paired
  `ratio_read` 0.41-0.63; the bests are 7.2-8.1) and is numerically equal to the scalar reference (max diff
  4e-5 on |ref| up to 182).
* **IQ3_XXS / IQ4_NL**: generic AVX2 grid-lookup kernels, arithmetic-bound per
  core (IQ3_XXS 39 % of one-core read at 1T, IQ4_NL ~50 %). `r8` (ggml 8x8
  repack gemv) lifts IQ4_NL over ggml's row `vec_dot` by 1.3-1.6x per core by
  medians (8.3 / 5.2 on gate, 8.8 / 6.8 on down; paired `ratio_read` 0.61 / 0.47
  and 0.93 / 0.74).

### Strata's kernel changes on AVX2 / AVX-VNNI-INT8 (no AVX-512)

As summarised in the task brief, Strata's CPU kernels do one unpack plus one VNNI
dot per 64 weights, share the weight decode across tokens, and reach 42 of 52
GB/s (81 %) of its host's bandwidth.

* **One unpack + one dot per 64 weights.** AVX-512 VNNI holds 64 u8/s8 lanes in
  one register: one `vpdpbusd zmm` per 64 weights. AVX2/AVX-VNNI holds 32, so the
  same work is two `vpdpbusd ymm` (or, for Q2_0, four 16 B `xmm` dots as in my
  prototype). The unpack (shift/and, or `vpshufb` LUT for 4-bit codebooks) is
  the part that does not shrink, so the per-64-weight instruction count is ~2x
  Strata's; on a 50 GB/s part that is affordable only if the loop is not already
  front-end bound (the production MXFP4 16-row loop is: see above). The practical
  AVX2 equivalent is a register-resident multi-row tile (8 rows x one activation
  block), which `mx8`/`r8` already demonstrate at 1T for MXFP4.
* **Decode shared across tokens.** That applies to the prompt-processing and
  Phase 1.5 multi-activation path (several tokens against the same expert, e.g.
  `simd_mxfp4_1row_4act_tile` at `cpu-dispatch.cpp:1093`), where one weight
  decode feeds several activation dots. It does **not** apply to TG batch 1: one
  token, one activation, nothing to share. The batch-1 analogue is sharing the
  decode across the **rows' shared activation loads** (the multi-row tile), which
  is what the Q2_0 note above proposes.
* **Target.** Strata's 81 % of peak is against a ceiling it measured on its own
  host; the equivalent here is 81 % of the pinned STREAM read median at 22
  threads, ~37 GB/s. Production `prod+pin` is at 14.4-46.7 GB/s median over the 13
  configs (31-102 % of STREAM); one config, Qwen Q8_0 down (46.7), is above the
  target already. The others are at 14-34, lowest for MXFP4 (14-30) and Q2_0
  (16-22). A pinned dynamic team reaches 38-46 GB/s median there with an 8-row
  MXFP4 tile or ggml's 8x8 repack gemv (and 45-54 with ggml's `vec_dot` on Q8_0
  gate shapes), so the target is reachable on this host **in the bench's team**;
  where `prod+pin` is below it, the data point to the dispatch (and, for MXFP4
  and Q2_0, the kernel), not to one cause for all types.

## Headroom estimate and proposed changes

Everything in this section that is a number was measured at 8-24 threads under the
load described above, in a **pinned in-process dynamic team** (the bench's own
scheduler, 16-row chunks), not in production's TBB arena. Moving a kernel into
production is a separate step whose gain is **not measured**; where an item
relies on a transfer from the team to the arena, it says so.

Median GB/s at 22 threads (the 24-thread rows are in the data directory; each
thread count is a separate process, so single cells jump by up to 2x between
neighbouring thread counts: Qwen Q8_0 down `prod` is 43 at 22 threads and 23 at
24):

| config | `prod` | `prod+pin` | `vecdot+pin` | `mx16+pin` | `mx8+pin` | `r8+pin` | `q2+pin` | `read+pin` (ceiling) |
|---|---|---|---|---|---|---|---|---|
| Qwen MXFP4 gate | 18 | 20 | 38 | 29 | 40 | 38 | -- | 37 |
| Qwen MXFP4 down | 30 | 30 | 33 | 41 | 42 | 46 | -- | 44 |
| GPT-OSS MXFP4 gate | 22 | 21 | 44 | 40 | 43 | 43 | -- | 55 |
| GPT-OSS MXFP4 down | 16 | 14 | 40 | 35 | 45 | 42 | -- | 52 |
| Qwen Q2_0 gate | 16 | 16 | 25 | -- | -- | -- | 34 | 46 |
| Qwen Q2_0 down | 17 | 22 | 15 | -- | -- | -- | 29 | 33 |
| Qwen IQ4_NL gate | 28 | 25 | 34 | -- | -- | 41 | -- | 43 |
| Qwen IQ4_NL down | 22 | 32 | 34 | -- | -- | 43 | -- | 46 |
| Qwen IQ3_XXS gate | 25 | 21 | 34 | -- | -- | -- | -- | 46 |
| Qwen Q8_0 gate | 28 | 30 | 45 | -- | -- | -- | -- | 50 |
| Qwen Q8_0 down | 43 | 47 | 51 | -- | -- | -- | -- | 55 |
| GPT-OSS Q8_0 gate | 29 | 25 | 54 | -- | -- | -- | -- | 56 |
| GPT-OSS Q8_0 down | 31 | 34 | 51 | -- | -- | -- | -- | 54 |

Burst-paired ratios of candidate kernels against `prod+pin` (same burst, median
over the configs of that type, 22 / 24 threads): `mx8+pin` 1.65 / 2.02 and
`r8+pin` 1.67 / 1.99 on MXFP4 (4 configs), `q2+pin` 1.33 / 2.20 on Q2_0 (2
configs), `r8+pin` 1.22 / 1.19 on IQ4_NL (2 configs). At 8 threads the same MXFP4
ratios are 2.4 / 2.5, the same order as the 1-thread gap (1.6-3.5x by medians);
they shrink as the read stream saturates.

**What those ratios consist of.** For MXFP4, `mx16+pin` is the production tile
shape run in the same pinned team: it already reaches 1.4-2.5x `prod+pin` by
medians (29/41/40/35 vs 20/30/21/14), so **most of the 22-thread gap is the
dispatch path, and the tile change adds a further 1.0-1.4x** (`mx8+pin` over
`mx16+pin`: 40/29, 42/41, 43/40, 45/35). The 1.6-3.5x tile result at one thread
does not transfer to 22 threads, where the 16-row loop is no longer the limit.

1. **Dispatch granularity of the production TBB call (new first item;
   hypothesis).** Same kernel, same weights: production is 0.66-0.84 of a dynamic
   team at 22-24 threads for Q8_0/IQ4_NL/Q2_0, and 0.59-0.99 of a pinned one. If
   the cause is the ~2 chunks per thread (`grain = max(64, rows/(2*threads))`,
   `cpu-dispatch.cpp:1167`; the same line is at :740 and :1840), a finer grain or a dynamic row counter is worth about
   1.2-1.5x (unpinned) to 1.0-1.7x (with a pin, see item 2) on those
   types, for no kernel or layout change (MXFP4 needs item 3 as well). **The cause is not isolated.** The
   decisive experiment is to run production with a grain sweep (e.g. 16/32/64/128
   rows) at 22 threads using the same paired protocol; the bench team's `--chunk`
   already sweeps the other side. The 64-row grain floor also caps a small call at
   `ceil(rows/64)` chunks (10 for a Qwen gate at top-1; see item 7).
2. **Pinning the arena workers one per core, P-cores first: downgraded.** The
   earlier 1.4-3.5x is not reproduced (see "Correction" above): paired,
   in-process, at load ~33 the pooled `prod+pin : prod` ratio is 0.95-1.12
   (within a few percent of 1.0 except 1.12 at 8 and 1.11 at 24 threads) and a
   shuffled pin does as well. The same pin is worth 1.1-1.4x to the finer-grained own team, and
   STREAM gains 10-20 %, so it is plausible that pinning pays once the dispatch is
   fixed (item 1); it should be re-measured then, with the same E-only and
   shuffled controls, and at a heavier load than 33 as well. It needs a
   `tbb::task_scheduler_observer` (or `task_arena` constraints); the bench pins
   from outside only to measure the effect, which is not the production mechanism.
   Open: threads versus other host work (VM, Frigate), interaction with the GPU
   submit threads, the 22-thread default.
3. **MXFP4: the 16-row tile.** Per core (1T, measured) the production kernel is
   2.3-3.8 GB/s median against 7.6 for ggml's one-row VNNI kernel and 8.0 for an
   8-row register tile (1.6-3.5x over the production kernel; Qwen gate, Qwen
   down, GPT-OSS gate). At 22 threads the tile accounts for 1.0-1.4x (above); the
   remaining gap is item 1. The change needs no layout: use the 8-row kernel (or
   ggml's one-row kernel) for the **batch-1 decode path**, and keep accumulators
   in registers (explicit unroll) rather than a stack array if the 16-row tile
   stays. Scope: `simd_mxfp4_q8_0_16row` is also called by the tensor-split path
   (`ggml_sycl_cpu_vec_dot_rows`, `cpu-dispatch.cpp:586-592`) and the PP GEMM
   loop (`ggml_sycl_cpu_pp_gemm`, `cpu-dispatch.cpp:6501`), and the Q4_0 16-row
   tile has the same shape (`simd_mul_mat_q4_0_q8_0_16row`, :551, :1230, :6462).
   None of those was measured here; wide tiles may win in the PP loop where
   activations are amortised, and the Q4_0 tile is unmeasured. The proposal is
   scoped to the batched-experts call site at :1296.
4. **MXFP4 repack: not justified by these numbers.** The 8x8 repack
   (block_mxfp4x8 = 8 e8m0 + 128 qs bytes, the size of the unpacked tensor) was
   +12-13 % per core over ggml's one-row kernel at 1T (medians 8.6 / 7.6, 8.3 / 7.4,
   8.4 / 7.5; paired `ratio_read` 0.61 / 0.53, 0.86 / 0.76, 0.67 / 0.58). At 22
   threads it is **equal to the 8-row tile** (`r8+pin` : `mx8+pin` by medians
   0.95 / 1.10 / 0.99 / 0.93 on the four configs). The cost is real, so the repack
   is not worth proposing for MXFP4 batch 1:
   * *Consumers of host-resident MXFP4 AOS rows* that would have to read the new
     layout or keep an AOS copy: the batched path
     (`ggml_sycl_cpu_expert_mul_mat_batched`, :931, 16-row at :1296 and the
     Phase 1.5 multi-activation path :1093), the single-task and adaptive paths
     (`ggml_sycl_cpu_expert_mul_mat` :818, `_adaptive` :1550), the tensor-split
     path (`ggml_sycl_cpu_vec_dot_rows` :492, :586), `ggml_sycl_cpu_vec_dot_batched`
     :650 and the PP GEMM `ggml_sycl_cpu_pp_gemm` :6396. `cpu_moe_host_aos_execute`
     (:861) is **not** one: it rejects everything but Q1_0/NVFP4.
   * *Tier moves.* Promotion to VRAM and eviction back to host need a de-repack
     or a conversion to the device's SOA layout; the planner must treat a
     repacked host copy as a distinct layout of the weight. There is no
     CPU-repacked value in `ggml_layout_mode` today (AOS, SOA, COALESCED,
     MXFP4_I8, XMX_*, ONEDNN_*, MXFP4_DPAS), so it would be a new layout.
   * *The mmap / file-mapped tier.* A repacked copy cannot live in the mapped
     file; it needs resident (pinned) memory, which is a host-budget change and
     counts against the planner's placement, not a free transform.
   * *Batch > 1.* `ggml_gemv_mxfp4_8x8_q8_0` is nr=1 only; the PP/multi-activation
     paths would need a gemm kernel for the repacked layout or must keep AOS.
   * *Advertising.* Per the fork rules a route advertises only (type, layout)
     pairs whose kernels exist: MXFP4 x repacked-8x8 would be advertised for
     batch-1 host-resident experts only.
   IQ4_NL is the same question with a smaller measured gain: `r8+pin` is 1.2x
   `prod+pin` (2 configs) and 1.2x the same-team `vecdot+pin` (41/34, 43/34), for
   the same layout costs.
5. **Q2_0: add an AVX-VNNI kernel.** Prototype `q2` (per core 1.7-2.1x the scalar
   kernel, packed weights, no repack). In the pinned team at 22 threads `q2+pin`
   is 34/29 against `vecdot+pin` (the same scalar kernel) 25/15 and `prod+pin`
   16/22, paired 1.33x (22 threads) / 2.2x (24) over `prod+pin` on 2 configs. The
   spread is wide (separate processes, 2 configs); the kernel is single-row
   without tiling and a multi-row tile would remove most activation loads
   (untested).
6. **Q8_0: the kernel is at the ceiling in the team; the gap is item 1.**
   `vecdot+pin` is 45-54 against `read+pin` 50-56 at 22 threads on the gate
   shapes (0.9-1.0), and 4-row tiles and prefetch do nothing (`q8_4row` 28 at 22
   threads, equal to `prod`).
7. **Small calls (`--topk 1,2,3`, 22 threads, Qwen gate, 3 rounds x 16 bursts x 4
   calls, load 32.5).** Median per-call time and GB/s of weight bytes, `prod` /
   `prod+pin` / pure read in the pinned team:

   | type | k | `prod` us (GB/s) | `prod+pin` us (GB/s) | `read+pin` us (GB/s) |
   |---|---|---|---|---|
   | Q8_0  | 1 | 125 (13.9) | 116 (15.0) | 51 (34.4) |
   | Q8_0  | 2 | 157 (22.2) | 171 (20.4) | 77 (45.0) |
   | Q8_0  | 3 | 215 (24.3) | 224 (23.3) | 123 (42.3) |
   | MXFP4 | 1 | 128 (6.8)  | 96 (9.1)   | 33 (26.4) |
   | MXFP4 | 2 | 157 (11.1) | 116 (14.9) | 51 (33.9) |
   | MXFP4 | 3 | 161 (16.2) | 144 (18.2) | 68 (38.2) |
   | Q2_0  | 1 | 74 (6.2)   | 67 (6.9)   | 19 (24.5) |
   | Q2_0  | 2 | 120 (7.7)  | 86 (10.7)  | 39 (23.7) |
   | Q2_0  | 3 | 136 (10.2) | 125 (11.0) | 41 (33.6) |

   A top-1 call moves 0.45-1.7 MB and takes 67-128 us in production, 2.3-3.9x a
   pure read in the team (19-51 us); the throughput is 6.2-13.9 GB/s: Q8_0 (13.9) is at
   the bottom of the large-call range (14-47), MXFP4 and Q2_0 (6-7) well below it. Per-call overhead, not bandwidth, bounds small calls: for
   Q8_0 the production-over-team excess is ~65-75 us at k=1 and ~90-100 us at k=3
   (the earlier single observation of ~50-70 us was the right order). Pinning
   shortens the median call time by 1.09-1.40x for MXFP4 and Q2_0 (MXFP4 128 ->
   96, 157 -> 116, 161 -> 144 us; Q2_0 74 -> 67, 120 -> 86, 136 -> 125 us); for
   Q8_0 it is 7 % faster at k=1 and 9 % and 4 % **slower** at k=2 and 3 (medians
   over the same interleaved bursts, no paired interval, so weak evidence). The statement that production batches only 1-3
   CPU-routed experts (~30 % of top-k at a 0.7 VRAM hit rate) is an **assumption
   from the placement design, not measured here**, and no number in this document
   relies on it.

## Not established

* **Why `prod` is slower than the same kernel in a dynamic team.** Item 1's
  grain explanation is a hypothesis with no experiment behind it; TBB wake-up
  latency, activation quantisation and the activation map are not separated.
* **Whether pinning helps `prod` at heavier load or after the dispatch change.**
  Measured only at load ~33 (idle 0-5 %, nice work 84-91 %). The old
  1.4-3.5x claim was made at load ~70 with an unpaired protocol and cannot be
  compared with these runs.
* **The kernel gains inside production's arena.** Every candidate number is from
  the bench's pinned team. The own-team `vecdot`/`read` arms also pre-quantise
  the activation and skip the activation map; the oracle covers values, not
  that. A production-path prototype is the next measurement.
* **One process per thread-count column**, so cells carry run-to-run load
  differences (a 22 -> 24 thread drop is visible in some rows); only in-process
  pairs are comparable.
* THP (`--thp`) had no consistent effect in the earlier runs.
* End-to-end tokens/s with a real model, and interference with the GPU submit
  threads under pinning: not measured (no GPU or model loads in this task).
* The ceiling moves with ambient load: re-measure `bench-host-stream`
  immediately before any percent-of-ceiling claim, and use paired in-burst
  ratios for comparisons. The STREAM table here predates the 8-24 thread runs
  and ran at heavier load.
* The pool's expert slots are identical copies, so the oracle detects a missing
  or wrong row but not a wrong-slot read.

## Data

`docs/backend/sycl-cpu-expert-bandwidth-data/`:

* `stream.csv`: every STREAM run behind the tables.
* `expert-1-2-4-threads.summary.csv`: the 1/2/4-thread sweep's summary rows
  (shape, mat, type, threads, variant, best/p90/median/p10/min GB/s, burst
  aggregate, `ratio_read`, median us, n).
* `expert-8-24-threads-qwen.summary.csv`, `expert-8-24-threads-gptoss.summary.csv`:
  the same columns for 8/12/16/20/22/24 threads, all variants including the
  `+pinE`/`+pinS` controls.
* `pairs-8-24-threads.csv`: the burst-paired ratio tables (`run_expert_sweep.py
  --pairs`: median and interquartile range of arm/base per burst, 24 bursts per
  config) behind every ratio quoted above.
* `pin-samples-8-24-threads.csv`: the last-run-CPU sample totals per arm.
* `pin-discovery-overcount.csv`: the pooled paired ratios (columns `t8`..`t24`)
  with and without the 8 over-counted config-runs, and those 8 runs (threads,
  round, workers found). In the per-run rows `cell_all_rounds` and
  `cell_excluding_overcounted_rounds` are cell-level values of `prod+pin : prod`
  for that config and thread count, repeated on each of its over-counted runs:
  the second drops every over-counted round of the cell, so the two GPT-OSS
  MXFP4 down 8-thread rows (rounds 0 and 2) both carry 1.12. It is the only
  committed record of the exclusion (see the over-count paragraph).
* `topk-{1,2,3}.summary.csv`: the small-call runs (item 7).
* `run-digest-8-24-threads.txt`: from each run's stderr, the oracle verdict
  counts (all `ok`: 1404 + 720 for the 8-24 thread sweeps, 18 per top-k run),
  the skipped config, the process return codes (all 0) and every `pin:` line
  (arena workers found per config and round; the numbers quoted for 21 workers
  and the top-1 range of 4-21 come from it).
* `load-*.txt`: `date`, `uptime` and `top -b -n1 | head -15` before each block.

The raw per-call CSVs and stderr logs are large and live only in the author's
scratchpad; regenerate with the commands above. The oracle printed `ok` for all
2124 (config x variant) checks of the 8-24 thread sweeps and all 54 of the
small-call runs (counts in the digest); the 1-4 thread sweep and the positive control are described in
Method.
