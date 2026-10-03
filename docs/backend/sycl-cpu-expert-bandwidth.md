# CpuExpertPool bandwidth versus a STREAM baseline

Tracker: `llama.cpp-cego` (Strata lesson 9). Host: Core Ultra 7 270K Plus, 24
CPUs (P-cores = CPUs 0-7, E-cores = 8-23), AVX2 + AVX-VNNI + AVX-VNNI-INT8, **no
AVX-512**, 2 DDR5 channels. `dmidecode` reports 4 x 64 GB Micron DDR5, configured
speed 5600 MT/s (2 DIMMs per channel), so the theoretical peak is
5600 MT/s x 8 B x 2 channels = 89.6 GB/s. L2 40 MB, L3 36 MB. Measured
2026-10-03 under the permanent ambient load (load average 45-70).

"Placement decides the executor" puts host-resident experts on the CPU, so the
CPU expert matvec is the decode path for every expert that does not fit in VRAM.
This document asks how close that path is to what the memory system can deliver
and what it would cost to close the gap without AVX-512.

**Status of the numbers.** Every table says how it was measured. Anything marked
*per-core extrapolation* is derived from 1-4 thread runs and was not measured at
22 threads with the candidate kernel; a measured value replaces it only when the
table says so. Tables marked **PENDING** are filled in by the 22-thread runs
that are scheduled behind a lead go (they would distort a concurrent model
capture).

## Tools (host-only, built on demand, no GPU)

| tool | what it does |
|------|--------------|
| `tools/cpu-expert-bench/bench-host-stream.cpp` | STREAM-style copy/scale/triad plus a pure-read kernel, 3 GiB total (three 1 GiB arrays, far above the 36 MB L3), threads 1..24, optional pinning, best and median of N 20-40 ms trials. No SYCL or ggml dependency; target `bench-host-stream` in any configuration, or plain `g++ -O3 -mavx2 -std=c++17 -pthread`. |
| `tools/cpu-expert-bench/bench-cpu-expert-matvec.cpp` | `prod` calls the real `ggml_sycl_cpu_expert_mul_mat_batched()` (the function a CpuExpertPool worker runs) on expert-shaped weights at batch 1. Other variants run the same cold weights through an in-process thread team: `read` (loads only), `vecdot` (ggml-cpu row `vec_dot`), `r8` (ggml's 8x8 repack gemv), `mx8`/`mx16`, `q2`, `q8_4row[_pf]` (prototypes). Each variant takes a pin suffix (`+pin`, `+pinE`, `+pinS`). Opens no SYCL queue. |
| `tools/cpu-expert-bench/run_expert_sweep.py` | One process per thread count (`GGML_SYCL_CPU_THREADS` is read once), rounds shuffled, summary with best/p90/median/p10/min, burst aggregate and paired `ratio_read`; saves each run's stderr (oracle, pin and CPU samples). |

`bench-cpu-expert-matvec` is defined in `ggml/src/ggml-sycl/CMakeLists.txt`, next
to `test-sycl-cpu-dispatch` (it needs that scope's `GGML_SYCL_DNNL` and links the
static `ggml-sycl`), so it exists only with `-DGGML_SYCL=ON
-DGGML_BACKEND_DL=OFF -DLLAMA_BUILD_TESTS=ON` (it sits inside `BUILD_TESTING`).
`bench-host-stream` is in `tools/cpu-expert-bench/CMakeLists.txt` and needs none
of that. The matvec target is compiled with `-mavx2 -mfma -mf16c -mavxvnni
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

## STREAM baseline (GB/s, best / median over 15 trials, mean of 2 runs' medians)

STREAM convention for copy/scale/triad (bytes read + written). `read` = weight
bytes read, which is what a matvec moves. Pinned = thread i on CPU i (P-cores
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
48 from 8 to 24 threads; the best barely moves above 8). One core pulls ~13 GB/s
median. Pinning is worth +10-20 % in STREAM already (pinned read median 45-48 vs
37-42 unpinned at 16-24 threads). Whether the expert matvec is flat past ~16
threads is a separate question, answered by the next section.

## The expert matvec (`prod`) against that baseline

### Per-core view (1 thread)

Effective weight GB/s, Qwen3.8 gate shape (N=640 K=2560, k=10 experts per call),
median (best) over 3 rounds x 8 bursts x 3 calls of the interleaved 1-thread run.
No arena scheduling and (as `prod+pin` equals `prod` here) no pinning effect:
this isolates the kernel. A single core pulls 14-16 GB/s median. The p10 of every
cell is 2-5 GB/s (load tail); ratios are the better comparison.
Raw: `sycl-cpu-expert-bandwidth-data/expert-1-2-4-threads.summary.csv`.

| type | `read` | `prod` | ggml `vecdot` | ggml 8x8 repack (`r8`) | `mx8` | `mx16` | prototype `q2` | `q8_4row` / `_pf` | `prod` / `read` |
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

Median (best) GB/s of weight bytes, `prod` = production as built (arena workers
free to migrate) and `prod+pin` = the same code with the arena workers pinned
(main on CPU 0, active workers on CPUs 1..). 1/2/4 threads come from the sweep
above (3 rounds x 8 bursts x 3 calls, burst-paired); the p10 of those cells is
1-8 GB/s.

| config | variant | 1 | 2 | 4 | 8 | 12 | 16 | 20 | 22 | 24 threads |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen Q8_0 gate  | `prod`     | 9.1 (12.9) | 5.5 (18.3) | 9.2 (15.6) | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Qwen Q8_0 gate  | `prod+pin` | 8.9 (12.4) | 4.6 (16.6) | 6.0 (19.1) | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Qwen MXFP4 gate | `prod`     | 2.3 (4.8)  | 3.0 (6.5)  | 3.2 (7.4)  | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Qwen MXFP4 gate | `prod+pin` | 2.2 (4.8)  | 3.1 (6.5)  | 3.5 (8.8)  | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| GPT-OSS MXFP4 gate | `prod`     | 3.7 (5.1) | 4.3 (8.3) | 4.6 (10.2) | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| GPT-OSS MXFP4 gate | `prod+pin` | 3.7 (5.5) | 4.6 (8.4) | 5.2 (12.4) | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| pinned STREAM `read` median | -- | 13 | 23 | 33 | 40 | 42 | 45 | 45 | 46 | 48 |

At 1-4 threads pinning changes little: the arena cannot be mis-scheduled much
with 1-3 workers. The 22-thread columns, where the production arena (default 22,
hw-2) oversubscribes a host that is permanently loaded, are PENDING. The claim
"flat past ~16 threads" is a STREAM statement (above) and is not asserted for
`prod` until those columns are filled.

#### Provisional 22-thread numbers (superseded protocol)

An earlier version of this document reported, for Qwen shapes at 22 threads,
`prod` medians of 10-23 GB/s and `prod`-with-arena-workers-pinned medians of
19-37 GB/s (Q8_0 gate 23 -> 33; MXFP4 gate 10 -> 25; Q2_0 19). Those runs used a
separate process per arm (so the arms did not share host state), found the arena
workers by thread-id order, and compared against an unpinned own-team `read`.
They are kept here only to say what the PENDING runs are expected to show
(pinning helps most at 20+ threads); they are not evidence of a magnitude.

### Pinning: mechanism evidence

PENDING at 22 threads. What is in place to produce it (all in-process, paired
per burst): `prod`/`prod+pin`/`prod+pinE` (arena workers on the E-cores only, a
negative control: if pinning merely stopped migration, E-only would also gain)
/`prod+pinS` (a fixed random permutation of all CPUs, a "pinned but not P-first"
control); arena workers identified by on-CPU time; the pinned-worker count and the
last-run CPU of every pinned worker are printed (`cpus ...: last-run CPU on
P-cores=.. E-cores=..`; "last-run" because a worker that has not run since the
affinity change still reports its old CPU). At 4 threads the mechanism checks
out: across the 1/2/4-thread sweep every sampled pinned thread had last run on
a P-core (`prod+pin` 2,208 samples, 0 on E-cores; the same for every pinned
own-team variant); the `+pinE` control was exercised only in a 4-thread smoke
run (all samples on E-cores).

## Bottleneck from the kernel inner loops

* **Q8_0** (`ggml_vec_dot_q8_0_q8_0`, x86 arch file): with a `GGML_NATIVE` build
  that defines `__AVXVNNIINT8__` (this host; **not** a `GGML_CPU_ALL_VARIANTS`
  build) it uses `vpdpbssd` and the fp16 table; one 34-byte block is two 32-byte
  loads, one dot, one FMA. 58-61 % of one-core read at 1T. A hand 4-row version
  with independent accumulators is no better (`q8_4row` 55 %), and a 512 B
  prefetch only matches `vecdot` (56-61 %). The kernel is at the ceiling its
  access pattern allows; the remaining headroom is **per-call overhead and
  scheduling**, not the dot loop (pinned Q8_0 is 72-80 % of the STREAM median in
  the earlier runs; to be re-measured above).
* **MXFP4, production 16-row kernel** (`simd_mxfp4_q8_0_16row`): objdump of
  `cpu-dispatch.cpp.o` (icpx -O3, Release) shows the `for r<16` loop is **not
  unrolled** (3 `vpdpbssd` in the function, not 48): the 16 accumulators are a
  **stack array**, each row paying a `vfmadd213ps (%rsp,%r11,4)` load and a
  `vmovups %ymm,(%rsp,%r11,4)` store per 2-block step, the 16 row pointers are
  reloaded from the stack (`mov 0x200(%rsp,%r11,1),%rbx`), and the e8m0 scale is
  ~12 scalar instructions (shift, cmp, cmov) per row per block. "Accumulators in
  memory" is the accurate wording, not "register spill". My out-of-line
  `mxfp4_rows<16>` compiles the same way (1 `vpdpbssd`, accumulators on the
  stack) and measures the same as `prod` (1T gate: 4.9 vs 4.8 GB/s best), while
  `mxfp4_rows<8>` unrolls (8 `vpdpbssd`, accumulators in registers) and measures
  like ggml's one-row kernel (9.8 vs 9.9). So the A/B supports the tile shape,
  not only the inference: 16 rows is ~2x slower than 8 rows or one row at one
  thread. (My `mx16` is not byte-identical to the production kernel: no 2-block
  unroll and F16C for the activation scale instead of the table.)
* **Q2_0** (64 weights / 18 bytes): ggml-cpu has only the scalar
  `ggml_vec_dot_q2_0_q8_0_generic` on x86, 3.5 GB/s per core. My `q2` prototype
  is **single-row with no row tiling**: per 64 weights and one row it does one
  16 B weight load, 3 shifts, 4 ands, 4 `vpdpbusd`, and ~6 activation loads (4
  planes, the ysum, the scale). Those activation loads are identical for every
  row, so a multi-row tile (4 rows share one set of plane loads) would cut loads
  per row from ~7 to ~2.5. It reaches 6.1-8.1 GB/s per core (1.7-2.3x scalar,
  41-46 % of read) and is numerically equal to the scalar reference (max diff
  4e-5 on |ref| up to 182).
* **IQ3_XXS / IQ4_NL**: generic AVX2 grid-lookup kernels, arithmetic-bound per
  core (IQ3_XXS 39 % of one-core read at 1T, IQ4_NL ~50 %). `r8` (ggml 8x8
  repack gemv) lifts IQ4_NL 1.2-1.3x per core (best vs best).

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
  host; the equivalent here is 81 % of the pinned STREAM read median, ~36-39
  GB/s at 16+ threads. The 22-thread pinned `prod` medians (PENDING) are the
  distance to it.

## Headroom estimate and proposed changes

Per-core extrapolations are flagged. Nothing below the PENDING tables has been
measured at 22 threads with a candidate kernel and pinning together; the
estimates below are the 1T ratios applied to the pinned STREAM median and are
upper bounds, not forecasts.

1. **Pin CpuExpertPool's workers one per core, P-cores first.** Prior-protocol
   measurement (see the provisional note): 1.4-1.7x Q8_0, 2-3.5x MXFP4 at 20-24
   threads. Zero kernel change and it multiplies the others. Needs a
   `tbb::task_scheduler_observer` (or `task_arena` constraints); the bench pins
   from outside only to measure the effect, which is not the production
   mechanism. Open questions: threads versus other host work (the VM, Frigate),
   interaction with the GPU submit threads, and re-checking the 22-thread
   default with pinning in place. To be confirmed by the paired PENDING runs.
2. **MXFP4: the 16-row tile.** Per-core (measured, 1T): the production kernel
   4.8 GB/s best, ggml's one-row VNNI kernel 9.9, an 8-row register tile 9.8,
   ggml's 8x8 repack gemv 12.5. The cheap step needs no layout change: use the
   8-row kernel (or ggml's one-row kernel) instead of the 16-row kernel in the
   **batch-1 decode path**, 2x per core. Scope: `simd_mxfp4_q8_0_16row` is also
   called by the tensor-split path (`ggml_sycl_cpu_vec_dot_rows`,
   `cpu-dispatch.cpp:586-592`) and by the PP GEMM loop (`ggml_sycl_cpu_pp_gemm`,
   `cpu-dispatch.cpp:6501`), and the Q4_0 16-row tile has the same shape
   (`simd_mul_mat_q4_0_q8_0_16row`, :551, :1230, :6462). None of those was
   measured here; wide tiles may win in the PP loop where activations are
   amortised, and the Q4_0 tile is unmeasured. The proposal is therefore scoped
   to the batched-experts call site at :1296 until those are measured.
   Compiler fix: an `#pragma unroll`/explicit unrolled body so the accumulators
   stay in registers would be the minimal alternative to dropping the tile.
3. **MXFP4 repack: what it would take.** The 8x8 repack (block_mxfp4x8 = 8 e8m0 +
   128 qs bytes, the same size as the unpacked tensor) is 12.5 vs 9.9 GB/s
   best at 1T (+26 %) over the cheap step; measured value at 22 threads is
   PENDING, and the gain cannot exceed the ceiling. It is not a free swap:
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
     batch-1 host-resident experts only, and every other (type, layout, batch)
     keeps the AOS kernel and its capability statement.
   The cheap step (item 2) needs none of this: ~2x per core (1T measured), no
   layout change.
4. **Q2_0: add an AVX-VNNI kernel.** Prototype `q2` (above): 1.7-2.3x per core,
   packed weights, no repack. At 22 threads the scalar kernel is far from the
   ceiling (the earlier runs: ~19 of 46 GB/s pinned) so the gain should survive;
   a multi-row tile would remove most activation loads. The measured 22-thread
   value of `q2+pin` is PENDING.
5. **IQ4_NL: repack 8x8** is 1.2-1.3x per core, ~0 once memory-bound. **IQ3_XXS:**
   no cheap fix; a VNNI grid kernel is a real project for a gain that shows only
   below ~12 threads.
6. **Q8_0: kernel at ceiling; remaining headroom is per-call overhead.** VNNI is
   used; 4-row tiles (0.55 of read, no better than `vecdot`) and prefetch are
   inside noise.
7. **Small calls.** `--topk 1,2,3` runs at 22 threads (PENDING) will replace the
   earlier single observation (a top-k 1 call of ~120-150 us for 1.7 MB against
   ~75 us for the pure read, i.e. ~50-70 us of fixed overhead). The statement
   that production batches only 1-3 CPU-routed experts (~30 % of top-k at a 0.7
   VRAM hit rate) is an **assumption from the placement design, not measured
   here**; it is not used in any number above.

## Not established

* Everything marked PENDING, and the combined estimate for MXFP4 (it needs the
  pinned `prod` and a candidate kernel together at 22 threads).
* THP (`--thp`) had no consistent effect in the earlier runs.
* End-to-end tokens/s with a real model, and interference with the GPU submit
  threads under pinning: not measured (no GPU or model loads in this task).
* The ceiling moves with ambient load: re-measure `bench-host-stream`
  immediately before any percent-of-ceiling claim, and use paired in-burst
  ratios for comparisons.
* The pool's expert slots are identical copies, so the oracle detects a missing
  or wrong row but not a wrong-slot read.

## Data

`docs/backend/sycl-cpu-expert-bandwidth-data/`: `stream.csv` (every STREAM run
behind the tables) and `expert-1-2-4-threads.summary.csv` (the sweep's summary
rows: shape, mat, type, threads, variant, best/p90/median/p10/min GB/s, burst
aggregate, `ratio_read`, median us, n). The raw per-call CSVs are large and live
only in the author's scratchpad; regenerate with the commands above.
