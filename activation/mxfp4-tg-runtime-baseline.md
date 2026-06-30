# MXFP4 TG Runtime Baseline Evidence

Worktree: `/Apps/llama.cpp-mxfp4-tg-runtime`
Branch: `feature/sycl-mxfp4-tg-runtime`
Baseline commit: `b08e732cd chore(sycl): isolate MXFP4 TG runtime baseline`

This worktree combines the pushed dry-run microbench suite with the previously dirty runtime baseline that contains `GGML_SYCL_MOE_DOWN_SUM_DIRECT`.

## Build

```bash
./scripts/sycl-build.sh llama-cli llama-bench sycl-mxfp4-moe-bench
```

Result: build passed after the isolated baseline import.

## Dry-Run Microbench Ranking

Command:

```bash
python3 scripts/run-sycl-mxfp4-tg-microbenches.py --dry-run --out-dir /tmp/mxfp4_tg_runtime_plan_dryrun
```

Summary:

```text
1. fused-layer saving_vs_baseline_ms 4.000000 records 3
2. prepack saving_vs_baseline_ms 2.266667 records 3
3. host-bounce saving_vs_baseline_ms 2.233333 records 3
4. row-parallel saving_vs_baseline_ms 1.933333 records 3
5. launch saving_vs_baseline_ms 1.066667 records 3
6. baseline saving_vs_baseline_ms 0.000000 records 1
```

Interpretation: fused-layer remains the strongest synthetic route, but existing runtime layer-executor evidence below shows the current implementation is not a winner.

## Current Safe Direct Route

Environment:

```bash
export ONEAPI_DEVICE_SELECTOR=level_zero:1
export GGML_SYCL_MOE_PHASE_MATERIALIZE=1
export GGML_SYCL_MOE_PHASE_BULK_XMX=1
export GGML_SYCL_MOE_DOWN_SUM_DIRECT=1
```

Count gate:

- Output block: `1, 2, 3, 4, 5`
- `UR_RESULT_ERROR_DEVICE_LOST`: false
- fatal/error/exception/assert markers: false
- Log dir: `/tmp/mxfp4_tg_runtime_b50_20260629_084130`

PP/TG:

```text
pp512 1234.06 ± 9.51 tok/s
tg128   37.05 ± 0.06 tok/s
```

Profile (`pp64/tg32`, FA on):

```text
gateup_glu ~5.55-5.72 ms / 48 calls
down       ~0.73 ms / 24 calls
```

Interpretation: PP remains healthy; TG is below the `>=45 tok/s` target. The direct down-sum path is no longer the bottleneck. Gate/up+GLU is the dominant remaining target.

## Existing Runtime Candidate Checks

### `GGML_SYCL_MOE_LAYER_EXECUTOR=1`

Additional env:

```bash
export GGML_SYCL_MOE_LAYER_EXECUTOR=1
```

Count gate:

- Output block starts exact: `1, 2, 3, 4, 5`
- fatal/error/device-lost markers: false
- Trace includes `path=pair_glu_down` ABI evidence, but many route rejects such as `requested=soa actual=aos reason=not_found`.

PP/TG:

```text
pp512 1230.60 ± 8.91 tok/s
tg128   34.90 ± 0.10 tok/s
```

Decision: reject current layer executor as a speed route. It is correctness-safe in this smoke but slower than the direct baseline.

### `GGML_SYCL_MOE_FUSED_GLU_Q8=1`

Additional env:

```bash
export GGML_SYCL_MOE_FUSED_GLU_Q8=1
```

Count gate:

- Output block starts exact: `1, 2, 3, 4, 5`
- fatal/error/device-lost markers: false
- Log includes `[MOE-FUSED-GLU-Q8]` evidence.

PP/TG:

```text
pp512 1229.47 ± 8.40 tok/s
tg128   36.87 ± 0.01 tok/s
```

Decision: safe but not faster; not a winner.

### `GGML_SYCL_MOE_DOWN_SUM_FUSION=1 + GGML_SYCL_MOE_DOWN_SUM_FROM_BIAS=1`

Checked on the pushed microbench branch before the runtime baseline was isolated:

```text
pp512 1259.70 ± 7.54 tok/s
tg128   38.52 ± 0.13 tok/s
```

But the canonical count output was incorrect (`1, 2, 3.`). Decision: rejected for correctness.

## Runtime Route Decision

Do not promote existing layer-executor, fused-GLU-Q8, from-bias down-sum fusion, or selected-expert prepack paths. Any new gate/up route remains explicit opt-in/default-off until lead-owned evidence shows exact count, fatal-free logs, required route evidence, `PP512 >= 1100`, and `TG128 >= 45`.

### Rejected selected-expert gate/up prepack route

`GGML_SYCL_MOE_GATEUP_PREPACK=1` was implemented as a selected-expert scratch route and rejected after lead validation. The route was count-correct and fatal-free, but performance regressed below baseline:

```text
Log dir: /tmp/mxfp4_gateup_prepack_b50_20260629_120612
PP512: 1230.05 ± 9.33 tok/s
TG128:   36.21 ± 0.42 tok/s
Baseline TG128: 37.05 ± 0.06 tok/s
```

Accepted-route profiling showed why it is not viable: the route copied `35,251,200` bytes per layer/token, adding about `1.07 ms/layer` and about `25.7 ms/token` across 24 layers, while gate/up compute did not improve. This route must remain rejected/default-off.

### Single persistent XMX_TILED gate/up proof route

The next approved route is `GGML_SYCL_MOE_GATEUP_SINGLE_XMX=1`: materialize gate/up experts once as one persistent `GGML_LAYOUT_XMX_TILED` VRAM layout, rebuild pointer tables from XMX_TILED `mem_handle` leases, and require PP/TG route evidence that no SOA gate/up fallback is used. This mode allows transient scratch but forbids a persistent SOA gate/up duplicate and forbids per-token gate/up prepack.

Required parser gates for non-dry-run lead validation are:

```bash
--require-single-xmx-gateup --forbid-gateup-soa-fallback
```

Workers must not run B50/B580/model gates for this route. Worker-owned checks are build/unit/parser/dry-run-only; lead owns B50/B580/model validation.

Latest lead validation remains blocked:

```text
Log dir: /tmp/sycl_single_xmx_gateup_b50_routelog_20260629_201925
Harness status: 18
Count: canonical `1, 2, 3, 4, 5`, fatal.total 0
Parser count evidence: diag.path.xmx-tiled-single-gateup 13, phase.single_xmx_gateup.complete 48
Diag: pp64 52.56 ± 0.24 tok/s, tg32 34.24 ± 0.11 tok/s
Full PP512/TG128 perf: not run because path proof failed first
Path check: error: single XMX_TILED gate/up profile path evidence missing
```

The evidence shows TG direct XMX_TILED materialization/labels, but not PP `xmx-tiled-single-gateup` profile evidence. `GGML_SYCL_MOE_GATEUP_SINGLE_XMX=1` must not silently bypass the existing prompt-XMX PP safety gates; PP direct XMX is still behind the existing explicit unsafe opt-ins `GGML_SYCL_XMX_MOE=1` plus `GGML_SYCL_XMX_MOE_ALLOW_UNSAFE_PP=1` / `GGML_SYCL_XMX_MOE_PP=1`. Do not add those unsafe PP opt-ins to promotion harnesses without a new reviewed task and fresh correctness evidence.

Promotion remains blocked until lead-owned evidence shows exact count, fatal-free logs, required route evidence, `PP512 >= 1100`, and `TG128 >= 45`. A near-target result is only non-promotional follow-up evidence when `42.0 <= TG128 < 45.0`, route evidence is clean, and the profile shows gate/up+GLU `<= 4.2 ms/token`; it does not authorize default-on promotion.

## 2026-06-29 packed-Q8 TG profiling pivot

After the single-XMX proof failed closed, lead-owned short B50 profiling pivoted back to the existing faster safe path:

```text
Safe packed-Q8 profile log: /tmp/mxfp4_safe_packed_q8_pack_profile_20260629_211552
pp64: 52.01 tok/s
tg32: 34.66 tok/s
fatal.total: 0
path: packed-q8-m2
calls=72, dpas=48, i8=6, soa=30
quant=0.604 ms, artifact=0.243 ms, pack=0.072 ms, kernel=6.443 ms
gateup_glu=5.964 ms, down=0.791 ms
```

`pack` timing was added to `[MXFP4-MOE-TG-PROFILE]` because the earlier rejected prepack route suggested copy/pack overhead might dominate. It does not: the packed-Q8 M2 activation pack is only about `0.05-0.07 ms/token`; gate/up+GLU remains the bottleneck at roughly `5.6-6.0 ms/token`. Do not spend the next optimization task on another per-token gate/up copy/prepack path unless it also reduces the DPAS gate/up kernel work.

Rejected/no-op follow-up probes:

- `GGML_SYCL_MOE_AGGRESSIVE_TG=1 GGML_SYCL_MOE_PARTIAL_DEVICE_GROUPING=1 GGML_SYCL_MOE_AGGRESSIVE_XMX_TILED=1`: `/tmp/mxfp4_aggressive_pack_profile_noval_20260629_212420`, fatal-free but `tg32 18.56 tok/s`, `aggressive-partial-fused-tg`, `gateup_glu=10.035 ms`, `down=6.076 ms`. Keep rejected.
- M2 ESIMD prefetch experiment `GGML_SYCL_XMX_TILED_PACK_Q8_M2_PREFETCH=1`: `/tmp/mxfp4_m2_prefetch_profile_20260629_213905`, fatal-free but `tg32 22.63 tok/s`, `gateup_glu=21.506 ms`. The code path was removed after the probe.
- `GGML_SYCL_MXFP4_I8_GROUPED_PACK_Q8=1`: `/tmp/mxfp4_down_packq8_tg_profile_20260629_215125`, fatal-free but effectively unchanged (`tg32 35.32 tok/s`, `down=0.781 ms`).
- Down direct-final harness env: `/tmp/mxfp4_down_direct_final_profile_20260629_215224`, fatal-free but `tg32 8.26 tok/s`, doubled profile calls and slower down/gateup accounting. Keep rejected for GPT-OSS TG promotion.
- `GGML_SYCL_MOE_PARTIAL_DEVICE_GROUPING=1` without aggressive env: `/tmp/mxfp4_partial_device_m2_profile_20260629_215333`, fatal-free but route stayed `packed-q8-m2` and performance was within noise/slightly lower (`tg32 34.53 tok/s`).

Next safe route should target launch/kernel count or a genuinely faster gate/up DPAS kernel for the existing `packed-q8-m2` route while keeping PP untouched and default-off. Graph diagnostics on `/tmp/mxfp4_safe_packed_q8_graph_diag_20260629_212522` showed no SYCL graph replay active for the short safe path (`use_graph=0`, graphlet counters zero), so a future graphlet/launch-consolidation task must be reviewed carefully against the known MoE graphlet regression before any promotion attempt.
