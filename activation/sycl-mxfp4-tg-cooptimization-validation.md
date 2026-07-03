# SYCL MXFP4 TG Co-Optimization Validation

## Build Under Test

- Branch: `feature/sycl-mxfp4-tg-runtime`
- Implementation HEAD before validation-record commit: `776223351`
- Validation updates through: `ccc3afbb7` build string in model runs after validation commits

## Automated Tests

- Initial broad pytest command with duplicate collection failed due a pytest-asyncio recursion in duplicate collection, not due a test assertion. The duplicate was confirmed by `ls tests/test-sycl-kernel-profile*.py tests/test-sycl-kernel-profiler-source-mmvq.py ...`, which listed `tests/test-sycl-kernel-profiler-source-mmvq.py` twice.
- Unique Python/source gate passed:
  - Command: `python3 -m pytest tests/test-sycl-kernel-profiler-source-mmvq.py tests/test-sycl-down-variant-profile-script.py tests/test-sycl-kernel-profile-coverage.py tests/test-sycl-kernel-profile-parser.py tests/test-sycl-kernel-profiler-source-copy.py tests/test-sycl-kernel-profiler-source-fattn.py tests/test-sycl-kernel-profiler-source.py -q`
  - Result: `33 passed in 0.42s`
  - Log: `/tmp/sycl_mxfp4_tg_pytests_unique_20260703_013604.log`
- SYCL build passed:
  - Command: `./scripts/sycl-build.sh test-sycl-kernel-profiler test-sycl-moe-fused-down-sum-policy llama-bench llama-cli`
  - Result: build succeeded with pre-existing warnings
  - Log: `/tmp/sycl_mxfp4_tg_build_20260703_013612.log`
- CTest passed:
  - Command: `ctest --test-dir build -R "test-sycl-kernel-profiler|test-sycl-moe-fused-down-sum-policy" -V`
  - Result: `100% tests passed, 0 tests failed out of 2`
  - Log: `/tmp/sycl_mxfp4_tg_ctest_20260703_014354.log`

## B50 GPT-OSS Correctness Gate

- Artifact: `/tmp/sycl_mxfp4_tg_correctness_baseline_20260703_014407`
- Command shape: B50 `ONEAPI_DEVICE_SELECTOR=level_zero:1`, FA-on phase envs, `llama-cli -cnv` with model-embedded chat template and `--chat-template-kwargs '{"reasoning_effort":"medium"}'`.
- Result: PASS. Final answer contained exact sequence `1, 2, 3, 4, 5`. The current CLI output did not include the historical leading colon, but the final-channel answer was the expected count-only answer.

## FA-on Bench/Profile Matrix

Down-variant matrix artifact root: `/tmp/sycl_down_variant_profile_20260703_014446`.

TG1-index artifact root: `/tmp/sycl_mxfp4_tg_tg1_profile_20260703_014736`.

Coverage is `profile.kernel_coverage_pct_x1000` from `scripts/parse-sycl-kernel-profile.py --wall-ms`, using `wall_ms = 128 / tg128 * 1000` from the same `llama-bench` row.

| Variant | Env delta | PP512 tok/s | TG128 tok/s | Kernel coverage | Gate/up kernel time | Down kernel time | Artifact dir | Decision |
|---------|-----------|-------------|-------------|-----------------|---------------------|------------------|--------------|----------|
| baseline | none | 1208.97 | 34.18 | 38.543% | `mxfp4.gateup.xmx_tiled_dpas_m2` 708.663 ms | `mxfp4.down.q8_soa` 623.930 ms | `/tmp/sycl_down_variant_profile_20260703_014446/baseline` | baseline |
| row2 | `GGML_SYCL_MOE_DOWN_SUM_Q8_SOA_TG_VARIANT=row2` | 1198.47 | 33.91 | 38.500% | `mxfp4.gateup.xmx_tiled_dpas_m2` 707.745 ms | `mxfp4.down.q8_soa_row_group` 634.371 ms | `/tmp/sycl_down_variant_profile_20260703_014446/row2` | reject: slower TG and down kernel |
| row4 | `GGML_SYCL_MOE_DOWN_SUM_Q8_SOA_TG_VARIANT=row4` | 1186.55 | 34.12 | 38.887% | `mxfp4.gateup.xmx_tiled_dpas_m2` 710.687 ms | `mxfp4.down.q8_soa_row_group` 637.518 ms | `/tmp/sycl_down_variant_profile_20260703_014446/row4` | reject: slower TG and down kernel |
| atomic | `GGML_SYCL_MOE_DOWN_SUM_DIRECT_ATOMIC=1` | 1197.48 | 34.25 | 39.156% | `mxfp4.gateup.xmx_tiled_dpas_m2` 710.070 ms | `mxfp4.down.q8_soa_atomic` 639.178 ms | `/tmp/sycl_down_variant_profile_20260703_014446/atomic` | reject: no meaningful TG gain, slower down kernel |
| tg1-index | `GGML_SYCL_MOE_GATEUP_M2_TG1_INDEX=1` | 1211.09 | 33.83 | 38.153% | `mxfp4.gateup.xmx_tiled_dpas_m2_tg1_index` 713.290 ms | `mxfp4.down.q8_soa` 619.200 ms | `/tmp/sycl_mxfp4_tg_tg1_profile_20260703_014736` | reject: slower TG and gate/up kernel |
| best-combined | none selected | n/a | n/a | n/a | n/a | n/a | n/a | not run: all individual proof knobs failed performance |

Diagnostic cached rows from the down-variant matrix were intentionally non-promotion rows because they set `GGML_SYCL_MOE_DOWN_SUM_DIRECT=0`:

| Variant | PP512 tok/s | TG128 tok/s | Artifact dir | Note |
|---------|-------------|-------------|--------------|------|
| cached-vector-qs | 1207.55 | 34.15 | `/tmp/sycl_down_variant_profile_20260703_014446/cached-vector-qs` | diagnostic only |
| cached-cache-y | 1208.27 | 33.75 | `/tmp/sycl_down_variant_profile_20260703_014446/cached-cache-y` | diagnostic only |
| cached-vector-qs-cache-y | 1195.72 | 33.84 | `/tmp/sycl_down_variant_profile_20260703_014446/cached-vector-qs-cache-y` | diagnostic only |

## Promotion Decision

No variant is promotable.

- Default remains unchanged.
- Row2/row4 direct q8-SOA down row groups are correctness-safe/default-off source paths but slower in this B50 GPT-OSS FA-on profile.
- TG1-index gate/up proof path is correctness-safe/default-off source path but slower in this B50 GPT-OSS FA-on profile.
- The new coverage metric shows only about 38-39% of decode wall time is inside the named kernel event sum for these runs. The remaining wall time is outside the two co-primary kernels, so reaching `TG128 >= 45 tok/s` is unlikely from row-group/TG1-index micro-optimizations alone.
- Next optimization direction should prioritize launch/gap/graph replay/fusion work from `docs/plans/2026-07-03-sycl-moe-decode-graph-replay-fusion-roadmap.md` rather than promoting these proof knobs.
