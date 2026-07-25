# B70 profiling and attribution findings

**Date:** 2026-07-24
**Task:** codescout `llama.cpp-fuo6` (B70 plan, T12)
**Build:** `f7d59875f` (branch `feature/sycl-b70-capability`), llama-bench reports `b12095-1cd23b843`
**Hardware:** Arc Pro B70 (Battlemage G31, `8086:e223`, 32.6 GB, 256 CU) = `level_zero:0`;
Arc Pro B50 (Battlemage G21, `8086:e212`, 16.3 GB, 128 CU) = `level_zero:1`
**Model:** `gpt-oss-20b-mxfp4.gguf` (11.27 GiB, 20.91 B) throughout
**Card state:** clean for every run — no process held VRAM on any card, and `journalctl -k`
showed no GT reset, GuC error, or engine fault before, during, or after the session.

---

## 1. Bottom line

**The B70's measured improvement is fully explained by VRAM headroom. The T3/T4 capability
fix contributed nothing measurable to it, and by mechanism it could not have.**

Two independent lines of evidence agree:

1. **Direct experiment.** Reproducing the confounded VRAM budget on the *current* binary
   (`GGML_SYCL_VRAM_BUDGET_PCT=42`) drops the B70 to **pp512 594.69 / tg128 15.65** — *below*
   the historical pre-fix confounded baseline of 803.93 / 19.79. The VRAM mechanism alone
   spans a larger range than the entire observed gain, leaving no residual for the
   capability fix to explain.

2. **Mechanism.** The capability fix changes `XMXConfig`, which is consumed only by the
   unified kernel. The unified kernel **never executes** for this model — it appears in
   neither the decode nor the prompt-processing kernel profile, despite being instrumented
   at 32 sites. The kernel that dominates decode lives in `mmvq.cpp`, which contains **zero**
   references to `XMXConfig`, `supports_esimd_dpas`, or GPU family.

The separate and more useful finding is **what actually limits the B70**: GPU kernel
execution is only ~15–19% of a decode token. The B70 completes its MoE kernel work
**2.32x faster** than the B50 yet delivers only **1.20x** the decode throughput, because
both cards carry a per-token cost of roughly 22 ms that is not kernel execution.
Kernel-level tuning — tile shapes, occupancy, the 256-CU hypothesis — is optimizing the
small end of the token.

---

## 2. Method corrections (read before reusing the documented knobs)

### 2.1 `GGML_SYCL_UNIFIED_FORCE_LEGACY` does not exist

It is documented in four places — `CLAUDE.md:432`, `docs/backend/sycl-env-vars.md:22`,
`docs/backend/sycl-perf-baselines.md:17`, `AGENTS.md:449` and `:514` — but **there is no
`getenv` site anywhere in the tree**. The only source occurrence is a comment in
`ggml/src/ggml-sycl/l144i-probe.hpp:12`. Setting it does nothing. The
`docs/backend/sycl-perf-baselines.md:17` row that attributes "PP512 (legacy) ~159" to it is
therefore unreproducible as written.

### 2.2 Its nearest real equivalent is also inert in production

`GGML_SYCL_UNIFIED_KERNEL` is read at exactly one place, `dispatch.hpp:109`, inside
`is_unified_kernel_enabled()`. That function's only caller is
`is_unified_kernel_enabled_for_device()` (`dispatch.hpp:210`), whose only callers in the
whole repository are two tests (`ggml/src/ggml-sycl/tests/test-esimd-dpas-gate.cpp`,
`tests/test-dispatch.cpp`). **The entire `dispatch.hpp` gate is dead code at runtime.**

This matters beyond the missing knob: the gate that T3 fixed — `cfg.supports_esimd_dpas`
reaching `is_unified_kernel_enabled_for_device()` — is not on the live dispatch path. The
framing in the task description and in `dispatch.hpp`'s own comment block ("we disable the
unified kernel entirely on hardware that doesn't support ESIMD dpas") does not describe
what the running code does.

The A/B was run anyway before this was established, and serves as a **control**:

| B70, clean card, 4 runs x 5 reps each | pp512 mean | tg128 mean |
|---|---:|---:|
| default | 1396.82 ± 41.46 | 37.69 ± 2.89 |
| `GGML_SYCL_UNIFIED_KERNEL=0` | 1310.66 ± 55.20 | 36.87 ± 2.86 |

The distributions overlap, as the code requires. A normalized diff of the full `-v` logs
shows zero structural difference between the two configurations — identical layout tallies,
identical 24/24 `down=mxfp4_i8`, identical cache and KV placement.

### 2.3 The knob that does work

`GGML_SYCL_VRAM_BUDGET_PCT` is live (`unified-cache.cpp:8083`, `:9747`, `:9903`) and prints
a verifiable `[UNIFIED-CACHE] Budget override via GGML_SYCL_VRAM_BUDGET_PCT=N%` line. All
budget results below were confirmed against that line plus the reported free-VRAM figure.

---

## 3. Clean-card B70 baseline (replicated)

Every run below reported `32602 MiB free` and `24 layout=mxfp4_i8` for `ffn_down_exps`.

Eight independent runs of 5 reps each (the two configurations of §2.2 pooled, since the
flag is provably inert):

| metric | mean | sd | min | max | run-to-run spread |
|---|---:|---:|---:|---:|---:|
| pp512 | **1353.74** | 64.52 | 1233.51 | 1426.20 | 14.2% |
| tg128 | **37.27** | 2.70 | 32.58 | 41.11 | 22.9% |

**Use 1353.74 / 37.27 as the clean B70 baseline, not any single run.** The prior
clean-card figure (1316.40 / 37.35) sits comfortably inside this distribution and was
sound. A single run of this benchmark is not a baseline: `llama-bench`'s own ± understates
the true spread, because it varies *within* one process and the dominant variation is
*between* processes. The first run taken in this session (1399.39 ± 45.29 / 41.11 ± 0.65)
was a favorable draw and should not be quoted on its own.

**Count gate: PASSES.** The GPT-OSS chat gate on the clean B70 returned exactly
`1, 2, 3, 4, 5`. Correctness holds without the 13.8 GB budget cap.

Comparison to the B50 on the same build: **pp512 895.47, tg128 31.02**.
B70/B50 = **1.51x** on pp512, **1.20x** on tg128.

---

## 4. Attribution: VRAM headroom vs the capability fix

### 4.1 The VRAM sweep

Same binary, same clean card, `GGML_SYCL_VRAM_BUDGET_PCT` swept. 5 reps per point.

| budget | pp512 | tg128 | weights spilled to host | `ffn_down_exps` on `mxfp4_i8` |
|---|---:|---:|---:|---:|
| 42% | 594.69 ± 26.23 | 15.65 ± 1.79 | **630.3 MB** | 0 / 24 |
| 45% | 1469.79 ± 67.35 | 32.38 ± 4.34 | 0 | 1 / 24 |
| 50% | 1471.27 ± 67.22 | 33.78 ± 5.57 | 0 | 7 / 24 |
| 60% | 1334.59 ± 111.71 | 39.15 ± 6.05 | 0 | 19 / 24 |
| 70% | 1292.97 ± 36.11 | 38.14 ± 2.72 | 0 | 24 / 24 |
| 85% | 1394.39 ± 58.51 | 35.88 ± 2.58 | 0 | 24 / 24 |
| 100% | 1353.74 (pooled) | 37.27 (pooled) | 0 | 24 / 24 |

This decomposes mechanism (b) into two distinct sub-mechanisms:

**(b1) The host-spill cliff — binary, and very large.** Between 42% and 45% the model stops
spilling to host, and throughput jumps **2.47x on pp512** and **2.07x on tg128**. This is
not a gradient; it is a threshold. The 42% run pushed 630.3 MB of weights to host memory and
`[MOE-LAYOUT] down-i8` reported `declined ALL 22 candidates ... VRAM headroom guard
(10.3 MB left, 64.0 MB reserve)` — reproducing the historical confounded run's
`0 of 22, remaining=12.2 MB` almost exactly.

**(b2) The down-i8 layout upgrade — gradual, and modest.** Across 45% -> 70% the upgrade
count climbs 1 -> 7 -> 19 -> 24 and tg128 climbs 32.38 -> 33.78 -> 39.15 -> 38.14, worth
roughly **+15% tg128** end to end. pp512 does not benefit and may be marginally worse. Given
the ±4–6 spreads, treat +15% as an estimate, not a measurement. §5.2 shows the mechanism
behind it, which is unambiguous even though the throughput delta is noisy.

### 4.2 Placing the historical numbers on the curve

| run | budget seen | pp512 | tg128 |
|---|---|---:|---:|
| historical pre-T3 ("baseline") | 13842 MB, confounded by ComfyUI | 803.93 | 19.79 |
| historical post-T3 (T3's A–E) | ~13.8 GB, confounded | 828.83 | 18.67 |
| **this session, `PCT=42`** | **13139 MB, clean card** | **594.69** | **15.65** |
| **this session, clean** | **32603 MB** | **1353.74** | **37.27** |

Both historical "baselines" sit below the host-spill cliff. Reproducing that condition on
the current build lands *below* them (the 42% override gives ~700 MB less budget than
ComfyUI happened to leave, and below the cliff every megabyte matters). The claimed
803.93 -> ~1354 improvement (+68% pp512, +88% tg128) is therefore **entirely a
crossing of the host-spill cliff**, with a modest additional contribution from (b2).

**There is no residual gain left to attribute to the capability fix.** Its contribution is
not distinguishable from zero by this experiment.

### 4.3 Why it could not have contributed — mechanism

The capability fix (`f03af88fc`, "derive GPU family from architecture in XMXConfig") changes
`XMXConfig::from_device()`. Two facts close the question:

- **`mmvq.cpp` — which owns the dominant decode kernel — has zero references to
  `XMXConfig`, `supports_esimd_dpas`, `family_from_device`, or `sycl_gpu_family`.** The MoE
  ESIMD kernels there select their path without consulting the family classification at all.
- **The unified kernel never runs.** It is instrumented at 32 `ggml_sycl_profile_label` /
  `ggml_sycl_profile_submit` sites in `unified-kernel.cpp`, yet **no unified-kernel entry
  appears in either the decode or the prompt-processing profile**. Its absence is a real
  observation, not an instrumentation gap.

Of the 10 commits in `f03af88fc^..HEAD`, only `f03af88fc` is functionally significant; the
others are refactors, diagnostics, and docs. So the code axis between the historical
baseline and this build is essentially just that one commit — which the above shows cannot
reach either hot path.

This confirms the lead's revised hypothesis in tracker comment `c-aagr`, and by a stronger
route than expected: the unified-kernel MUL_MAT share of decode is not ~10%, it is **0%**.

---

## 5. Per-kernel attribution

### 5.1 The B70's peak memory bandwidth

**Neither route to a published figure works on this host, so the figure below is measured.**

- SYCL's `ext_intel_memory_bus_width` returns **64 bits for both cards** — the documented
  placeholder default (`dpct/helper.hpp:630-638`), not a real value. `memory_clock_rate`
  returns 2800 MHz on the B70, which is exactly `tile0/gt0/freq0/rp0_freq`, i.e. the *core*
  clock. The derived 22.4 GB/s is nonsense.
- `xpu-smi` is broken on this host: `error while loading shared libraries: libmetee.so.5.0.0`.

Measured with a STREAM-style triad and copy over 256 MB buffers. **Buffers must be filled
with incompressible data** — this host's patched compute-runtime does USM memory
compression, and constant-filled buffers initially reported 1327 GB/s on the B70 and 713
GB/s on the B50, both above physical peak.

| card | triad best / median | copy best / median |
|---|---:|---:|
| B70 | 520.7 / 517.6 GB/s | **529.6** / 525.1 GB/s |
| B50 | 195.9 / 195.8 GB/s | **197.3** / 197.1 GB/s |

**Calibration:** the B50's published peak is 224 GB/s, so this benchmark achieves
**88.1%** of peak — a normal STREAM efficiency, which validates the method.

Applying the same efficiency to the B70 implies a peak of **~601 GB/s**. That is consistent
with a 256-bit GDDR6 bus at ~19 Gbps (608 GB/s), which would be the expected configuration
for a full G31 die with 32 GB. **Treat 601 GB/s as inferred, not cited** — no datasheet was
available. For roofline work, prefer the *measured achievable* figures (529.6 and 197.3
GB/s); percent-of-achievable is the more meaningful denominator and needs no inference.

**The B70 has 2.68x the B50's usable memory bandwidth and 2x its compute units, and returns
1.20x the decode throughput.**

### 5.2 Decode kernel profile (`GGML_SYCL_KERNEL_PROFILE=1`, `-p 0 -n 128 -r 1`)

Both cards ran identical geometry: 129 tokens, 24 layers.

| kernel | B70 | B50 |
|---|---:|---:|
| `mxfp4.gateup.xmx_tiled_dpas_m2` | 555.70 ms / 3096 calls | 701.40 ms / 3096 calls |
| `mxfp4.soa.batched` (down projection) | **absent** | 586.55 ms / 2322 calls |
| all captured kernels | 663.87 ms | 1404.81 ms |
| MoE share of captured time | 83.7% | 91.7% |

**The structural difference is the down-projection layout, and it is exact.** The B50
upgraded 6 of 24 `ffn_down_exps` to `mxfp4_i8` and left 18 on SOA; 18 layers x 129 tokens =
**2322** — precisely the `mxfp4.soa.batched` call count. The B70 upgraded 24 of 24 and
issues **zero** such calls: the down projection is absorbed into the fused
`mxfp4.gateup.xmx_tiled_dpas_m2` kernel, whose call count is unchanged at 3096.

That is the mechanism behind (b2), and it is why the full down-i8 upgrade is worth pursuing
on the B50 independently of the noisy throughput delta: it removes an entire kernel pass.

**Roofline** (`scripts/parse-sycl-kernel-profile.py --geometry gpt-oss-20b --peak-gbs`):

| kernel | µs/call | bytes/call | achieved | % of achievable peak |
|---|---:|---:|---:|---:|
| B50 `gateup` | 226.5 | 35.25 MB | 155.6 GB/s | **78.9%** of 197.3 |
| B50 `soa.batched` (down) | 252.6 | 17.63 MB | 69.8 GB/s | **35.4%** of 197.3 |
| B70 fused `gateup`+down | 179.5 | 52.88 MB | 294.6 GB/s | **55.6%** of 529.6 |

> The parser attributes only `gateup` bytes (35.25 MB) to that kernel and so reports 196.4
> GB/s / 37.1% for the B70. That is an undercount: on the B70 the kernel also carries the
> down traffic. The 52.88 MB row above is the corrected figure. **The `gpt-oss-20b` geometry
> preset's `role_rules` do not model the fused-down case** and should be extended, or the
> B70's roofline will read ~18 points low in every future capture.

So the B70's dominant kernel runs at **55.6%** of its achievable bandwidth where the B50's
comparable kernel reaches **78.9%**. There is real kernel-level headroom on the B70 — but
§5.3 shows how little it is worth.

### 5.3 The actual limiter: decode is not kernel-bound

| | B70 | B50 |
|---|---:|---:|
| captured kernel time per token | 5.15 ms | 10.89 ms |
| wall time per token (profiled) | 33.89 ms | 32.43 ms |
| **kernel coverage of wall time** | **15.2%** | **33.6%** |

**The B70 does its GPU kernel work 2.32x faster than the B50 (4.31 vs 9.98 ms/token of MoE)
and still takes the same wall-clock time per token.**

Subtracting kernel time from unprofiled throughput (B70 37.27 t/s = 26.8 ms/token; B50
31.02 t/s = 32.2 ms/token) leaves **~22.5 ms/token on the B70 and ~22.2 ms/token on the
B50** — essentially identical, and card-independent. That residual is **84% of a B70 decode
token** and 69% of a B50 one.

An Amdahl bound follows: driving the B70's kernel time to *zero* would move tg128 from 37.3
to about 44.4 t/s, **+19%**. Every kernel-level lever available — tile shape, occupancy,
the 256-CU hypothesis, closing the 55.6% -> 78.9% roofline gap — competes for a share of
that 19%. The other 81% is elsewhere.

Per token the profile records ~461 kernel launches (24 `gateup`, 72 `binbcast.mul`, 72
`binbcast.add`, 72 `binbcast.event`, 48 `set_rows`, 48 `rope`, 24 `softmax`, 24 `pack_q8`,
24 `quantize`, ~25 `memcpy`, ~25 `mem_fill`). 22 ms across ~461 launches is ~48 µs per
launch of non-kernel cost. That points at submission/synchronization overhead and host-side
graph work, not at the kernels.

### 5.4 Prompt processing

A `-p 512 -n 0` capture on the B70 (pp512 1276.53):

| kernel | time | share of captured |
|---|---:|---:|
| `fattn.compute.xmx_v2` | 44.70 ms | 69.6% |
| `sycl.binbcast.add` / `.mul` | 13.18 ms | 20.5% |
| `mxfp4.soa.batched` | 0.46 ms | 0.7% |
| all captured | 64.20 ms | — |

Prompt processing is dominated by **flash attention**, and again shows **no unified-kernel
entry**. Coverage here is low (64.2 ms captured against ~379 ms of pp512 wall time, ~17%):
the dense and MoE matmuls for prompt processing run through oneDNN, which this profiler
does not instrument. **Treat the pp breakdown as indicative only** — the absence of the
unified kernel is solid, but the positive shares are not a complete accounting.

---

## 6. Limitations

State these alongside any number quoted from this document.

1. **Profiler coverage is partial.** 15.2% of B70 decode wall time and ~17% of pp512 wall
   time land in captured kernels. The decode profile contains **no flash-attention kernel at
   all**, though prompt processing does — decode attention is either uninstrumented or takes
   an uncaptured path. The uncaptured remainder is a mix of host overhead and uninstrumented
   work, and this document does not separate them.
2. **Enabling the profiler perturbs the measurement** (B70 tg128 falls from ~37.3 to 29.51).
   The ~22 ms/token residual in §5.3 mixes *profiled* kernel times with *unprofiled*
   throughput and is therefore an estimate. Its robustness comes from the two cards agreeing
   to within 0.3 ms, not from the precision of either figure.
3. **The 42% budget override is not a bit-exact replay** of the ComfyUI confound. It caps
   the budget by percentage (13139 MB) where ComfyUI capped it by residency (13842 MB). Below
   the cliff this matters, which is why the reproduction lands lower than the historical run
   rather than on top of it.
4. **The B70 peak of ~601 GB/s is inferred** from the B50's calibrated efficiency, not cited.
   The measured achievable figure (529.6 GB/s) is the reliable number.
5. **The +15% attributed to (b2)** rests on points whose spreads are ±4–6; the *mechanism*
   (elimination of the `soa.batched` pass) is certain, the magnitude is not.
6. **Between-run spread is large** (pp512 14.2%, tg128 22.9%). Differences below ~15% on
   pp512 or ~25% on tg128 between single runs are not evidence of anything.

---

## 7. Recommendations

No code changes were made in this task. In priority order:

1. **Retire the 256-CU / tile-shape / occupancy line of investigation as the primary lever.**
   It targets at most 19% of a decode token, and the unified kernel it would tune does not
   execute for this model. §5.3 is the reason, not a preference.
2. **Investigate the ~22 ms/token non-kernel decode cost.** It is the single largest term in
   decode on both cards, it is card-independent, and nothing in this plan has examined it.
   Start by instrumenting the decode path's host side and by confirming whether SYCL graph
   replay is actually active during `llama-bench` decode — ~48 µs per kernel launch across
   ~461 launches is the shape of un-batched submission.
3. **Close the decode flash-attention instrumentation gap.** A profile that captures 15% of
   the token cannot answer "what is slow", and the missing attention kernel is a known hole.
4. **Fix the docs for `GGML_SYCL_UNIFIED_FORCE_LEGACY`** (four files, §2.1) and decide the
   fate of the dead `dispatch.hpp` gate (§2.2) — either wire
   `is_unified_kernel_enabled_for_device()` into dispatch or delete it. Right now a fix
   landed against a gate nothing calls, and two tests assert on behavior that never runs.
5. **Extend the `gpt-oss-20b` geometry preset** to model the fused-down case, or every future
   B70 roofline reads ~18 points low (§5.2).
6. **The B50 down-i8 double-charge fix remains worth doing** — §5.2 shows it removes an entire
   kernel pass (2322 calls/128 tokens), which is a cleaner justification than the throughput
   delta.
7. **Record free VRAM beside every future GPU measurement.** Every run in this document did
   so; it is what made the confound self-evident rather than a three-task detour.

---

## 8. Reproduction

```bash
source /opt/intel/oneapi/setvars.sh --force

# Clean-card baseline (repeat >=5 times; a single run is not a baseline)
ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_OP_TIMEOUT_MS=180000 \
  ./build/bin/llama-bench -m /Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf \
  -p 512 -n 128 -r 5 -v

# Reproduce the confounded budget on a clean card
ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_OP_TIMEOUT_MS=180000 \
  GGML_SYCL_VRAM_BUDGET_PCT=42 \
  ./build/bin/llama-bench -m /Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf \
  -p 512 -n 128 -r 5 -v

# Decode kernel profile
ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_OP_TIMEOUT_MS=180000 \
  GGML_SYCL_KERNEL_PROFILE=1 GGML_SYCL_KERNEL_PROFILE_FORMAT=json \
  GGML_SYCL_KERNEL_PROFILE_OUTPUT=/tmp/prof.json \
  ./build/bin/llama-bench -m /Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf -p 0 -n 128 -r 1 -v
python3 scripts/parse-sycl-kernel-profile.py /tmp/prof.json \
  --geometry gpt-oss-20b --peak-gbs 529.6 --top-kernels 12
```

Per-tensor down-layout tally (the `[MOE-LAYOUT] down-i8` line fires only on decline, so
silence is a pass, not a failure):

```bash
... -v 2>&1 | grep ffn_down_exps | grep -oE "layout=[a-z0-9_]+" | sort | uniq -c
```

`-v` is mandatory or ggml's log callback is muted. Key off the **selector**, never the
logged `device=N` — a B50-only and a B70-only run both print `device=0`.
