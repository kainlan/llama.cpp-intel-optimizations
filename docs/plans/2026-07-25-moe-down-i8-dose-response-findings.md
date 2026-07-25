# MoE down-I8 layout upgrade — dose–response findings

**Task:** `llama.cpp-nzu4` — is the MoE down-i8 layout upgrade a net win at the
margin on the B50?
**Date:** 2026-07-25 · **Branch:** `feature/sycl-b70-capability`
**Instrument:** `GGML_SYCL_MOE_DOWN_I8_MAX_TENSORS` (commit `0e9a48dc3`)

## Answer

**The premise is refuted at the low end and only partly holds at the margin.**

`llama.cpp-nzu4` proposed that the extra down-i8 layers Plan C's VRAM reclaim
bought cost ~1.0–1.5% pp512 for no tg gain, and that "grant every layer that
fits" might therefore be the wrong policy. Measured:

1. Granting **zero** costs **−33.7% pp512**. The upgrade is load-bearing for the
   documented ~894 pp512 baseline, not a marginal optimization. A cap anywhere
   near the low end would be catastrophic.
2. There **is** a tg gain, contradicting T8's "no measurable tg gain":
   **+1.24% tg128 going 2→5, p=0.0089**.
3. So the pp512 cost is real in sign and roughly right in size, but it is **not
   a loss** — it buys a comparable tg gain that T8's method could not resolve.

**Policy conclusion: "grant what fits" is defensible as written.** Future VRAM
reclaim on the B50 is not buying pp512 losses; it is buying a pp/tg trade that
leans slightly toward tg — the direction this fork usually wants, since decode
latency is the common complaint.

## Why a new instrument was needed

`c-ifvz` ruled out reproducing T8's method. T8 moved the granted count with
`GGML_SYCL_VRAM_ARENA_EXTERNAL_HEADROOM_MB`, which changes the arena size, the
zone layout and the resident tensor set **simultaneously** — a throughput
difference between two such runs cannot be attributed to the layout. It also
matched only the granted *count*, never verifying both arms upgraded the *same*
tensors.

`GGML_SYCL_MOE_DOWN_I8_MAX_TENSORS` moves the granted count and nothing else:
same budget, same arena, same zones, same candidate set.

### The instrument was verified before being used

| property | result |
|---|---|
| cap is exact | cap 0/2/4 → 0/2/4 granted; cap 6 → 5 (headroom binds first) |
| **prefix identity** | cap 2 → `blk.0–1`; cap 6 → `blk.0–5`; uncapped → `blk.0–7` |
| **single variable** | `gateup-i8 upgraded_layers=0` at *every* cap, including cap 0 with 2320.8 MB freed |

The prefix property is what makes equal counts mean equal **tensors** —
`c-ifvz`'s actual requirement. Candidates are sorted by `(layer_id, name)` and
carry uniform `extra_charge` on a uniform-shape MoE model, so capping at N
grants exactly the layers the headroom guard would have granted at that count.

The gate/up check is a confound `c-ifvz` does not mention and which would have
invalidated everything: `maybe_upgrade_moe_gate_up_layouts_to_i8` takes
`remaining` **by reference** and runs *after* the down pass, so capping down
hands it ~261 MB per skipped tensor. It declines for its own reasons, so the
arms differ by exactly one layout change. **Re-check this if either pass
changes.**

## Results

GPT-OSS 20B MXFP4, B50 (`level_zero:1`), `-p 512 -n 128 -fa 1 -r 1`.
6 rounds × 4 arms = 24 runs; every arm in every round; arm order rotated each
round so arm never correlates with position.

| granted | n | pp512 mean ± sd | tg128 mean ± sd |
|--------:|--:|----------------:|----------------:|
| 0 | 6 | 605.51 ± 0.92 | 34.144 ± 0.188 |
| 2 | 6 | 913.50 ± 2.03 | 35.126 ± 0.126 |
| 4 | 6 | 910.75 ± 2.18 | 35.488 ± 0.120 |
| 5 | 6 | 907.13 ± 2.89 | 35.562 ± 0.190 |

Paired by round:

| transition | pp512 | tg128 |
|---|---|---|
| 0 → 2 | **+50.86%**  t=528.6  p<0.0001 | **+2.88%**  t=11.45  p=0.0001 |
| 2 → 4 | −0.30%  t=−5.44  p=0.0028 | +1.03%  t=+4.42  p=0.0069 |
| 4 → 5 | −0.40%  t=−10.09  p=0.0002 | +0.21%  t=+0.94  p=0.39 (ns) |
| 2 → 5 | −0.70%  t=−7.99  p=0.0005 | **+1.24%**  t=+4.15  p=0.0089 |

The pp512 cost is a **gradient, not a threshold** — monotonic and small beyond
2 layers.

## Limits — read before extending this

1. **Only measured to 5 of 24.** A codescout daemon held 1430 MB of B50 VRAM
   for the entire matrix (`free=14818.8 MB`, identical in all 24 runs — see
   `llama.cpp-2rkc`), capping achievable grants at 5. Whether the pp cost stays
   linear out to 10 or 24, or turns over, is **untested**. Linear extrapolation
   to 24 predicts roughly −3% pp512 / +4% tg128; that is extrapolation, not
   measurement. Re-run on a free B50 with caps 0/6/12/18/24 to close it.
2. **The machine was NOT quiet.** load1 ranged 6.8–12.3 (Frigate ffmpeg) plus
   the VRAM tenant. `free_mb` was *identical* in all 24 runs, so the tenant was
   steady rather than drifting, and the rotating interleave means session drift
   hits every arm equally. **The paired deltas are sound; the absolute t/s are
   depressed and must not become baselines.**
3. **The 0-layer arm at 605 pp512 is a diagnostic configuration, not a
   regression.** Do not quote it as one.

## Raw data

`round,order_pos,cap,achieved,free_mb,pp512,tg128,load1`

```csv
1,1,0,0,14818.8,607.161008,34.397148,8.19
1,2,2,2,14818.8,916.848420,35.111094,8.75
1,3,4,4,14818.8,913.511270,35.349792,9.93
1,4,6,5,14818.8,909.854387,35.545681,9.44
2,1,2,2,14818.8,912.817348,35.086078,8.86
2,2,4,4,14818.8,911.653859,35.572758,9.58
2,3,6,5,14818.8,908.482316,35.640760,10.21
2,4,0,0,14818.8,604.338079,33.866456,9.54
3,1,4,4,14818.8,907.364571,35.408519,9.48
3,2,6,5,14818.8,902.265368,35.196854,9.41
3,3,0,0,14818.8,605.159886,34.043384,10.80
3,4,2,2,14818.8,910.879643,35.242105,11.51
4,1,6,5,14818.8,909.561197,35.591187,12.34
4,2,0,0,14818.8,605.427480,34.258552,11.21
4,3,2,2,14818.8,914.133620,35.080006,9.85
4,4,4,4,14818.8,912.164960,35.681815,9.93
5,1,0,0,14818.8,605.574485,34.227890,10.76
5,2,2,2,14818.8,913.985344,35.293648,9.92
5,3,4,4,14818.8,909.495384,35.441009,8.54
5,4,6,5,14818.8,905.460517,35.736805,7.89
6,1,2,2,14818.8,912.305766,34.943046,7.01
6,2,4,4,14818.8,910.302615,35.475025,6.84
6,3,6,5,14818.8,907.186071,35.660400,7.28
6,4,0,0,14818.8,605.419703,34.072020,6.99
```

## Reproducing

```bash
source /opt/intel/oneapi/setvars.sh --force
timeout 900 env ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  GGML_SYCL_MOE_DOWN_I8_MAX_TENSORS=$CAP GGML_SYCL_MOE_LAYOUT_DEBUG=1 \
  ./build/bin/llama-bench -m /Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf \
  -p 512 -n 128 -fa 1 -r 1 -v -o csv > run.csv 2> run.log
```

Two parsing traps that cost time:

- **`llama-bench -o csv` emits no `pp512` / `tg128` string.** Rows are
  identified by the `n_prompt` (col 34) / `n_gen` (col 35) pair; throughput is
  `avg_ts` at col 40. Grepping for the test name silently yields empty, which in
  a batch run is indistinguishable from a null result.
- **Record the ACHIEVED count, not the requested cap.** The cap is an upper
  bound; if headroom binds first the arm degrades silently. Parse
  `upgraded_tensors=` from the `[MOE-LAYOUT] down-i8` line.

## See also

- `llama.cpp-nzu4` comment `c-mrz6` — this result in the tracker
- `llama.cpp-r5ib` — the `[MOE-LAYOUT]` counters and the ~261 MB/tensor charge
- `llama.cpp-2rkc` — the VRAM tenant that bounded this matrix at 5 layers
- `docs/plans/2026-07-25-zone-sizing-findings.md` — Plan C, whose reclaim
  raised the granted count and prompted this question
