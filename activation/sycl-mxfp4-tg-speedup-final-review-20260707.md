# SYCL MXFP4 TG Speedup Final Review

Promotion decision: reject default-on, keep opt-in

Validation source of truth: `activation/sycl-mxfp4-tg-speedup-validation-20260707.md`.

The target `TG128 >=45 tok/s` was not reached. `GGML_SYCL_MOE_DOWN_Q8_DPAS_TILE=tile2|tile4` remains experimental/default-off. The tile2 and tile4 count gates passed, but both failed the promotion criteria: no end-to-end TG improvement and no >=5% named-kernel reduction versus the same-build baseline. Task 6 gate/up runtime dispatch was skipped because no lead-approved gate/up loadv2 evidence justified adding a runtime branch.

## Artifacts

| Purpose | Path |
| --- | --- |
| model-free layer-glu/down bench | `/tmp/sycl_mxfp4_tg_speedup_bench_20260707_120003` |
| tile2 count gate | `/tmp/sycl_mxfp4_tg_speedup_count_20260707_120018` |
| tile4 count gate | `/tmp/sycl_mxfp4_tg_speedup_count_tile4_20260707_120046` |
| direct-context llama-bench sweep | `/tmp/sycl_mxfp4_tg_speedup_llamabench_direct_20260707_120221` |
| down variant named-kernel matrix | `/tmp/sycl_mxfp4_tg_speedup_down_matrix_20260707_120320` |

## Validation Summary

| Case | Count gate | PP512 tok/s | TG128 tok/s | Down label | Down ms | Gate/up ms | Total named ms | Decision |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| baseline | n/a | `1212.02` | `34.37` | `mxfp4.down.q8_soa` | `620.982` | `706.458` | `1567.962` | same-build baseline |
| `GGML_SYCL_MOE_DOWN_Q8_DPAS_TILE=tile2` | pass: `1, 2, 3, 4, 5`; no fatal/device-lost markers | `1213.76` | `33.12` | `mxfp4.down.q8_dpas_tile2` | `755.553` | `703.576` | `1700.305` | reject: TG regressed and down time increased |
| `GGML_SYCL_MOE_DOWN_Q8_DPAS_TILE=tile4` | pass: `1, 2, 3, 4, 5`; no fatal/device-lost markers | `1198.11` | `34.21` | `mxfp4.down.q8_dpas_tile4` | `635.430` | `707.154` | `1585.966` | reject: no TG gain and no 5% kernel reduction |

Same-build direct-context sweep also rejected the new candidates: baseline `PP512 1212.86` / `TG128 28.04`, tile2 `PP512 1212.08` / `TG128 27.33`, and tile4 `PP512 1211.97` / `TG128 28.02`.

Model-free layer-glu/down latency favored tile2 (`935.122 us`) over the current comparable baseline (`1340.122 us`) and tile4 (`1161.368 us`), all correctness-pass, but this did not predict full-model TG performance.

## Kernel Hotspots

| Case | `mxfp4.gateup.xmx_tiled_dpas_m2` ms | Down kernel | Down ms | Total named ms | Outcome |
| --- | ---: | --- | ---: | ---: | --- |
| baseline | `706.458` | `mxfp4.down.q8_soa` | `620.982` | `1567.962` | baseline |
| down-dpas-tile2 | `703.576` | `mxfp4.down.q8_dpas_tile2` | `755.553` | `1700.305` | reject: down time +21.7%, TG -3.6% |
| down-dpas-tile4 | `707.154` | `mxfp4.down.q8_dpas_tile4` | `635.430` | `1585.966` | reject: no 5% named-kernel reduction |
| row2 | `708.353` | `mxfp4.down.q8_soa_row_group` | `632.797` | `1581.294` | rejected prior row-group path |
| row4 | `706.579` | `mxfp4.down.q8_soa_row_group` | `633.297` | `1581.174` | rejected prior row-group path |
| atomic | `706.665` | `mxfp4.down.q8_soa_atomic` | `633.278` | `1583.053` | rejected prior atomic path |

## Final Decision and Follow-up

- Do not promote `GGML_SYCL_MOE_DOWN_Q8_DPAS_TILE=tile2` or `tile4` to default-on.
- Keep both values available only as experimental/default-off diagnostics.
- Do not document either opt-in as recommended for performance. If a follow-up needs a tile comparison, `tile4` is the least disruptive measured tile candidate, but it still trails baseline and failed promotion.
- Do not add the Task 6 `GGML_SYCL_MOE_GATEUP_M2_LOADV2` runtime dispatch in this pass; the lead skipped it because there was no approved loadv2 gate/up evidence.
- Next optimization target: gate/up load-path evidence around the runtime-count lines and/or launch consolidation/graphlet overhead, with the same canonical correctness, PP, TG, fatal-marker, and named-kernel gates.
- Tracker action: update `llama.cpp-ghuz` with this rejection decision and tested commands; no default-on follow-up is authorized by this validation.
