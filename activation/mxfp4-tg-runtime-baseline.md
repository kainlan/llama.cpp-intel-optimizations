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

Do not promote existing layer-executor, fused-GLU-Q8, or from-bias down-sum fusion paths. The next runtime implementation should target a new, explicit opt-in gate/up route that attacks the `~5.6 ms/token` gate/up+GLU bucket directly.

The plan is a selected-expert gate/up prepack route:

- prepack only the active experts for a decode token/layer;
- store prepacked scratch through unified-cache `mem_handle` ownership;
- feed a compact DPAS-friendly gate+up+GLU compute kernel;
- keep current direct down-sum path as the downstream path;
- default off behind a new env gate;
- fail closed to the current safe route if metadata, handles, events, or capacity are not valid.

Promotion remains blocked until lead-owned evidence shows exact count, fatal-free logs, required route evidence, `PP512 >= 1100`, and `TG128 >= 45` or an explicitly accepted near-target result.
