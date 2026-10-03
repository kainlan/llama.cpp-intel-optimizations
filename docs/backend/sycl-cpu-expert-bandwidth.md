# CpuExpertPool bandwidth versus a STREAM baseline

Tracker: `llama.cpp-cego` (Strata lesson 9). Host: Core Ultra 7 270K Plus, 24
CPUs (P-cores = CPUs 0-7, E-cores = 8-23), AVX2 + AVX-VNNI + AVX-VNNI-INT8, **no
AVX-512**, 2 DDR5 channels (theoretical 89.6 GB/s), L2 40 MB, L3 36 MB.
Measured 2026-10-03 under the permanent ambient load (load average 45-55).

"Placement decides the executor" puts host-resident experts on the CPU, so the
CPU expert matvec is the decode path for every expert that does not fit in VRAM.
This document asks how close that path is to what the memory system can deliver,
and what it would cost to close the gap without AVX-512.

## Tools (host-only, built on demand, no GPU)

| tool | what it does |
|------|--------------|
| `tools/cpu-expert-bench/bench-host-stream.cpp` | STREAM-style copy/scale/triad plus a pure-read kernel, 3 GiB total (arrays well above the 36 MB L3), threads 1..24, optional pinning, best and median of N rounds. Plain `g++ -O3 -mavx2 -pthread`; no dependencies. |
| `tools/cpu-expert-bench/bench-cpu-expert-matvec.cpp` | Target `bench-cpu-expert-matvec` (`EXCLUDE_FROM_ALL`, `GGML_SYCL=ON`, `GGML_BACKEND_DL=OFF`). `prod` calls the real `ggml_sycl_cpu_expert_mul_mat_batched()` -- the function a CpuExpertPool worker runs -- on expert-shaped weights at batch 1. Other variants run the same cold weights through an in-process team: `read` (loads only, the ceiling for this access pattern), `vecdot` (ggml-cpu's row `vec_dot`), and the kernel prototypes below. Opens no SYCL queue. |
| `tools/cpu-expert-bench/run-expert-sweep.py` | One process per thread count (`GGML_SYCL_CPU_THREADS` is read once), rounds shuffled, optional `--pin-arm` second arm, per-burst paired ratio to `read`. |

Build and run:

```bash
source /opt/intel/oneapi/setvars.sh --force
./scripts/sycl-build.sh bench-cpu-expert-matvec
g++ -O3 -mavx2 -std=c++17 -pthread tools/cpu-expert-bench/bench-host-stream.cpp -o bench-host-stream
./bench-host-stream --sched dynamic --rounds 15 --threads 1,2,4,8,12,16,20,24 --pin 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 --pin-label pinned
python3 tools/cpu-expert-bench/run-expert-sweep.py --bin build/bin/bench-cpu-expert-matvec \
    --rounds 4 --pin-arm --threads 1,2,4,8,12,16,20,22,24 --out sweep.csv -- \
    --types q8_0,mxfp4,iq4_nl,iq3_xxs,q2_0 --shapes qwen38 --mats gate,down \
    --variants prod,read --bursts 16 --burst-calls 4 --pool-mb 256
```

Shapes (experts, batch 1, one MUL_MAT_ID worth of rows per call):
GPT-OSS 20B = 2880x2880 gate/up/down, MXFP4, top-4; Qwen3.8-Flash-Next =
hidden 2560, intermediate 640 (gate/up N=640 K=2560, down N=2560 K=640), top-10.
The GSQ-RCO IQ3_XXS Qwen file is a mixed allocation, so IQ3_XXS is measured on
gate/up only (K=640 is not a multiple of 256, which IQ3_XXS needs), and Q2_0 /
IQ4_NL stand in for the `down` tensors. Weight pool is 256 MB per type, rotated
so every call reads cold data.

## Method (and why the absolute numbers are soft)

Absolute GB/s here float by about 2x with ambient load, between and within runs.
So: variants run interleaved burst by burst in random order inside one process;
the weight pool is cold for every call; threads are timed on the workers (the
main thread's timing jitter once produced an impossible 107 GB/s read figure);
helper threads sleep rather than spin; and the drift-proof number is the paired
`ratio` of two variants inside the same burst. Tables show **median** (typical
under load) and **best** (what a quiet window delivers). Early sweeps that let
idle team workers yield-spin contaminated `prod` and were discarded.

One caveat on the ceiling: the in-process `read` variant uses its own thread
team and degrades under load at 16+ threads more than the STREAM tool does, so
**the STREAM table below is the ceiling to quote**, and `prod` is compared
against it directly.

## STREAM baseline (GB/s, best / median, 2 runs each, dynamic chunk scheduling)

STREAM convention for copy/scale/triad (bytes read + written). `read` = weight
bytes read, which is what a matvec moves. Pinned = thread i on CPU i (P-cores
first).

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

The read ceiling is about **45-48 GB/s typical, 55-70 GB/s in quiet windows**,
i.e. roughly half of the 89.6 GB/s theoretical peak typical and 60-78 % best.
One core cannot saturate it (13 GB/s median, ~25 best); 8 P-cores reach ~40 and
the curve is flat past ~16 threads. Pinning is worth +10-25 % in STREAM already.

## The expert matvec (`prod`) against that ceiling

Effective weight-bytes/s, Qwen3.8 shapes, median (best), batch 1, 22 threads
unless stated. `prod` = production as built (TBB arena workers unpinned);
`prod+pin` = same code with the arena workers pinned to CPUs 0.. from outside
(`--pin-tbb`); 4 rounds x 16 bursts x 4 calls per cell.

| type, tensor | prod 1T best | prod 22T | prod+pin 22T | prod+pin 24T | pinned STREAM read 22T |
|---|---|---|---|---|---|
| Q8_0 gate  | 9.3  | 23 (41) | 33 (69) | 35 (73) | 46 (69) |
| Q8_0 down  | 10.3 | 22 (39) | 37 (72) | 33 (76) | 46 (69) |
| MXFP4 gate | 4.4  | 10 (24) | 25 (61) | 27 (66) | 46 (69) |
| MXFP4 down | 4.6  | 11 (23) | 24 (66) | 26 (67) | 46 (69) |
| IQ4_NL gate | 7.9 | 11 (26) | 30 (69) | 31 (72) | 46 (69) |
| IQ3_XXS gate | 5.8 | 13 (26) | 21 (59) | 29 (68) | 46 (69) |
| Q2_0 gate  | 3.5  | 9 (19)  | 19 (51) | 17 (51) | 46 (69) |
| Q2_0 down  | 3.5  | 6 (17)  | 19 (50) | 18 (52) | 46 (69) |

GPT-OSS 20B (2880x2880x4, median (best), 22T): Q8_0 gate `prod` 12 (35) ->
`prod+pin` 27 (68); MXFP4 gate 5.7 (24) -> 20 (66); MXFP4 down 8.3 (26) -> 18
(67). At 24T `prod+pin` is 28 / 22 (Q8_0 / MXFP4 gate).

Reading it:

* **Pinning the CpuExpertPool workers is the biggest single lever, and a
  scheduling one.** At 20-24 threads `prod+pin` / `prod` medians are 1.4-1.7x
  for Q8_0, 2.0-2.7x for Qwen MXFP4 and IQ4_NL (3.5x for GPT-OSS MXFP4 at 22T),
  ~2x for Q2_0 and IQ3_XXS. The
  production arena (default 22 threads, hw-2) is free to migrate and to land
  workers on E-cores while P-cores idle; under the ambient load that costs more
  than any kernel difference. Below ~8 threads pinning matters little.
* **Q8_0 is at the memory system.** Pinned, it delivers 33-37 GB/s median
  (70-80 % of the pinned STREAM read median) and its best windows (69-76) match
  the best STREAM read. The kernel is not the problem.
* **MXFP4, Q2_0 and (at low thread counts) IQ3_XXS/IQ4_NL are kernel-limited.**
  Pinned MXFP4 sits at 25 of 46 GB/s typical (~55 %), GPT-OSS MXFP4 at ~20
  (~43 %), Q2_0 at 19 (~40 %). The per-core table shows why.
* Best-window numbers differ as much as medians do: unpinned `prod` never
  exceeds 23-41 GB/s at 22T even in its best window, pinned reaches 50-77 -- as
  high as the STREAM read best. So the pinned code *can* reach the ceiling when
  the host is quiet; the typical-case gap that remains is decode cost.

### Per-core view (1 thread, best of 4 rounds, GB/s of weight bytes)

One thread is the cleanest read on the kernel because there is no arena
scheduling and little contention. `read` = 14-16 GB/s is what one core can pull.

| type | `prod` | ggml `vecdot` | ggml 8x8 repack gemv (`r8`) | prototype | `read` |
|---|---|---|---|---|---|
| Q8_0 | 10.5 | 10.3 | -- | `q8_4row` 9.4, `q8_4row_pf` 10.4 | 15.8 |
| MXFP4 | 4.5 | 7.9 | 8.8 | `mx8` 5.9, `mx16` 4.8 | 14.0 |
| IQ4_NL | 8.6 | 8.4 | 10.4 | -- | 15.4 |
| IQ3_XXS | 5.8 | 5.8 | -- | -- | 14.4 |
| Q2_0 | 3.5 | 3.5 | -- | `q2` (AVX-VNNI) 6.6 | 14.2 |

## Bottleneck from the kernel inner loops

* **Q8_0** (`ggml_vec_dot_q8_0_q8_0`, x86 arch file): already `vpdpbssd`
  (AVX-VNNI-INT8) with the fp16 scale table; one 34-byte block is two 32-byte
  loads, one dot, one FMA. 65-70 % of one-core read at 1T, and a
  hand 4-row version with independent accumulators and a prefetch is within
  +/-5 %. It is memory-bound at 16+ threads. Nothing to do.
* **MXFP4** (hand 16-row kernel in `cpu-dispatch.cpp`): `acc[16]` of ymm
  accumulators needs 16 registers plus the shuffle-LUT, scale and activation
  registers on a 16-register AVX2 file, so it spills in the hot loop. It is
  **slower per core than the generic ggml row kernel** (4.5 vs 7.9 GB/s at 1T)
  and slower than an 8-row variant of its own shape (5.9). The bottleneck is
  register pressure and the nibble unpack, not memory. ggml's interleaved
  8x8 repack gemv (`ggml_gemv_mxfp4_8x8_q8_0`, exported from ggml-cpu) reaches
  8.8 GB/s per core (2x the hand kernel) because it unpacks 8 rows per load.
* **Q2_0** (64 weights / 18 bytes): ggml-cpu has only the scalar
  `ggml_vec_dot_q2_0_q8_0_generic` on x86, ~3.5 GB/s per core, FMA/decode
  bound. A plain AVX-VNNI kernel is straightforward (below).
* **IQ3_XXS / IQ4_NL**: generic AVX2 grid-lookup kernels, arithmetic-bound per
  core (5.8 / 8.4 GB/s at 1T, 40 % / 55 % of one-core read) but 16+ cores
  overtake the memory ceiling, so these only lose at low occupancy.

## Headroom estimate and proposed changes (no AVX-512)

Estimates are for the 22-24 thread, pinned, loaded-host regime (medians),
against a typical ceiling of ~46 GB/s. Memory-bound types cannot gain beyond the
ceiling; the cap is stated.

1. **Pin CpuExpertPool's workers one per core, P-cores first (scheduling).**
   Measured 1.4-1.7x (Q8_0), 2.0-3.5x (MXFP4/IQ4_NL), ~2x (Q2_0/IQ3_XXS) at
   20-24 threads on this loaded host; zero kernel change. Cheapest gain and
   it multiplies the others. Needs a `tbb::task_scheduler_observer` (or
   `task_arena` constraints) -- the bench pins from outside only to measure the
   effect, which is not the production mechanism. Open question for the
   change: threads versus other host work (the VM, Frigate) and interaction
   with the GPU submit threads; the default of 22 threads should be re-checked
   with pinning in place (24 was not worse here).
2. **MXFP4: drop the 16-row hand kernel.** 1.35-1.75x per core at the cheapest
   (generic ggml row kernel 7.9, 8-row prototype 5.9), 2x with ggml's 8x8
   repack gemv (8.8). At 22 threads the gain is capped by the ceiling:
   pinned MXFP4 25 -> ~35-40 GB/s (1.4-1.6x) for Qwen, GPT-OSS 20 -> ~35. The
   repack form is a CPU-optimal materialized layout for host-pinned experts and
   so fits "layout follows residency": repack once at materialization
   (block_mxfp4x8 = 8 e8m0 + 128 qs bytes, same size as the unpacked tensor),
   advertise it only for host-resident experts.
3. **Q2_0: add an AVX-VNNI kernel.** Prototype in the bench (`q2`): keep the
   weights packed (no repack), dot the four 2-bit planes with `vpdpbusd`
   against four pre-split activation planes, subtract the activation sums
   (the {0,1,2,3} -> {-1,0,1,2} offset). Numerically equal to the scalar
   reference (max diff 4e-5 on |ref| up to 182). 1.8x per core (6.6 vs 3.5 GB/s
   at 1T); at 22 threads it should move Q2_0 from ~19 toward the ceiling,
   ~2x. Needs the activation planes prepared once per call set (cheap, K/64
   blocks).
4. **IQ4_NL: repack 8x8** (`ggml_gemv_iq4_nl_8x8_q8_0`): only ~1.2x per core,
   ~0 once memory-bound. Low priority. **IQ3_XXS:** no cheap fix; a VNNI grid
   kernel is a real project for a gain that only appears below ~12 threads.
5. **Q8_0: nothing.** VNNI is already used. 4-row accumulators and prefetch
   distance 512 B give +/-5 %, inside noise. (Q8_0 is the only type where
   pinning alone reaches the ceiling.)
6. **Small calls (per-call latency).** At top-k = 1 the 22-thread `prod+pin`
   call takes ~120-150 us for 1.7 MB (about 13 GB/s), against ~75 us for the
   pure read; a fixed ~50-70 us of arena wake/join dominates. Production
   batches only the CPU-routed experts (about 30 % of top-k at a 0.7 VRAM hit
   rate, i.e. 1-3 experts), so many real calls are in this regime and the
   bandwidth tables above overstate them. Fewer threads for small calls (the
   gain is below the noise of this measurement; not measured as a change) and
   fusing gate+up+down submissions are the levers.

Combined, for MXFP4 experts (GPT-OSS, 22T medians): ~6-9 GB/s unpinned
production -> ~20 pinned -> ~35 with a repack kernel is the path to 4-5x, about
75 % of the typical 46 GB/s ceiling.

## Not established

* THP (`--thp`) had no consistent effect in these runs.
* End-to-end tokens/s with a real model, and interference with the GPU submit
  threads under pinning: not measured (no GPU or model loads in this task).
* The ceiling is a moving target under ambient load: re-measure
  `bench-host-stream` immediately before any claim about percent-of-ceiling, and
  use paired in-burst ratios for comparisons.
* The sweeps in this document use the 256 MB weight pool per type; per-expert
  call counts follow `--topk` (default = shape's top-k).
