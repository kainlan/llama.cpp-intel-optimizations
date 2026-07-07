# SYCL MXFP4 TG Speedup Validation — 2026-07-07

## Summary

Promotion decision from Task 7: **reject default-on and keep `GGML_SYCL_MOE_DOWN_Q8_DPAS_TILE` experimental/default-off**.

The new down Q8 DPAS tile variants pass correctness and build gates. The model-free layer benchmark shows `tile2` can reduce isolated layer-glu/down latency, but full GPT-OSS B50 profiling shows no end-to-end TG improvement. In the named-kernel matrix, `tile2` increases `mxfp4.down.*` time substantially and lowers TG128. `tile4` is count-correct but does not clear the 5% improvement bar and slightly trails the same-build baseline.

## Artifacts

| Purpose | Path |
| --- | --- |
| model-free layer-glu/down bench | `/tmp/sycl_mxfp4_tg_speedup_bench_20260707_120003` |
| tile2 count gate | `/tmp/sycl_mxfp4_tg_speedup_count_20260707_120018` |
| tile4 count gate | `/tmp/sycl_mxfp4_tg_speedup_count_tile4_20260707_120046` |
| direct-context llama-bench sweep | `/tmp/sycl_mxfp4_tg_speedup_llamabench_direct_20260707_120221` |
| down variant named-kernel matrix | `/tmp/sycl_mxfp4_tg_speedup_down_matrix_20260707_120320` |

## Non-model gates

Commands:

```bash
python3 -m pytest \
  tests/test-sycl-kernel-profiler-source-mmvq.py \
  tests/test-sycl-moe-gateup-work-reduction-source.py \
  tests/test-sycl-down-variant-profile-script.py \
  tools/sycl-kernel-bench/tests/test_reference_kernels_registry.py \
  tools/sycl-kernel-bench/tests/test_mxfp4_layer_glu_down_variants.py -q

git diff --check

./scripts/sycl-build.sh llama-bench llama-cli test-sycl-moe-fused-down-sum-policy sycl-kernel-bench
ctest --test-dir build -R 'test-sycl-moe-fused-down-sum-policy|test-sycl-kernel-profiler' --output-on-failure
```

Results:

- pytest: `38 passed`
- `git diff --check`: pass
- build: pass
- CTest: `2/2` pass

## Model-free layer-glu/down bench

Command shape:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/sycl-kernel-bench \
  --kernel=<candidate> --quant=MXFP4 --dim_m=2880 --dim_n=4 --dim_k=2880 \
  --iterations=20 --warmup=5 --validate --output=json
```

| Candidate | Latency us | Bandwidth GB/s | Correctness |
| --- | ---: | ---: | --- |
| `mxfp4_layer_glu_down_soa_r4` | `1340.122` | `39.538` | pass |
| `mxfp4_layer_glu_down_q8_dpas_tile2` | `935.122` | `56.661` | pass |
| `mxfp4_layer_glu_down_q8_dpas_tile4` | `1161.368` | `45.623` | pass |

This isolated benchmark favored `tile2`, which justified running model correctness and named-kernel profiling. It did **not** predict full-model performance.

## Canonical count gates

Command shape used the GGUF/Jinja Harmony template through `llama-cli -cnv`, pinned `reasoning_effort=medium`, and used `--reasoning-format none --reasoning-budget 0`.

| Candidate | Output | Fatal/device-lost markers | Decision |
| --- | --- | --- | --- |
| `GGML_SYCL_MOE_DOWN_Q8_DPAS_TILE=tile2` | `1, 2, 3, 4, 5` | none | correctness pass |
| `GGML_SYCL_MOE_DOWN_Q8_DPAS_TILE=tile4` | `1, 2, 3, 4, 5` | none | correctness pass |

## Same-build direct-context PP/TG sweep

Command shape:

```bash
GGML_SYCL_MOE_DOWN_SUM_DIRECT=1 \
GGML_SYCL_MOE_DOWN_Q8_DPAS_TILE=<tile?> \
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf -p 512 -n 128 -ngl 99 -fa 1 -r 1
```

| Candidate | PP512 tok/s | TG128 tok/s | Decision |
| --- | ---: | ---: | --- |
| same-build baseline | `1212.86` | `28.04` | baseline |
| `tile2` | `1212.08` | `27.33` | reject: TG regression |
| `tile4` | `1211.97` | `28.02` | reject: no improvement |

PP remains above the guardrail, but neither tile improves TG.

## Down variant named-kernel matrix

Matrix command:

```bash
SYCL_DOWN_VARIANT_PROFILE_OUT=/tmp/sycl_mxfp4_tg_speedup_down_matrix_20260707_120320 \
  scripts/sycl-gptoss-down-variant-profile-matrix.sh --execute
```

| Case | PP512 tok/s | TG128 tok/s | Down label | Down ms | Gate/up ms | Total named ms |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| baseline | `1212.02` | `34.37` | `mxfp4.down.q8_soa` | `620.982` | `706.458` | `1567.962` |
| down-dpas-tile2 | `1213.76` | `33.12` | `mxfp4.down.q8_dpas_tile2` | `755.553` | `703.576` | `1700.305` |
| down-dpas-tile4 | `1198.11` | `34.21` | `mxfp4.down.q8_dpas_tile4` | `635.430` | `707.154` | `1585.966` |
| row2 | `1209.58` | `34.47` | `mxfp4.down.q8_soa_row_group` | `632.797` | `708.353` | `1581.294` |
| row4 | `1202.57` | `34.30` | `mxfp4.down.q8_soa_row_group` | `633.297` | `706.579` | `1581.174` |
| atomic | `1205.74` | `33.92` | `mxfp4.down.q8_soa_atomic` | `633.278` | `706.665` | `1583.053` |

The current same-build profiled baseline is `TG128 34.37 tok/s`. `tile2` regresses TG by `3.6%` and down-kernel time by `21.7%`. `tile4` is count-correct but does not improve TG and does not meet the 5% named-kernel reduction bar.

## Decision

- Keep `GGML_SYCL_MOE_DOWN_Q8_DPAS_TILE=tile2|tile4` **default-off**.
- Do not promote either tile candidate.
- Do not implement the conditional runtime gate/up loadv2 Task 6 in this pass; only bench-only prep exists and no favorable lead evidence justifies runtime dispatch.
- Next optimization should not continue this row-tile shape blindly. Use the runtime-count line attribution around gate/up scale/packed loads or investigate launch consolidation/graphlet overhead after preserving current correctness gates.
