# SYCL End-to-End Profiling Closure Validation

## Safe gates

Status: **PASS** for safe parser/source/dry-run gates.

Commands run:

```bash
python3 -m pytest \
  tests/test-sycl-pti-l0-parser.py \
  tests/test-sycl-ur-trace-parser.py \
  tests/test-sycl-vtune-exports-parser.py \
  tests/test-sycl-layer-ledger-parser.py \
  tests/test-sycl-full-attribution-profile-script.py \
  tests/test-sycl-source-line-probe-source.py \
  tests/test-sycl-source-line-debug-matrix-script.py \
  tests/test-sycl-vtune-source-line-checker.py \
  tests/test-sycl-vtune-source-line-feasibility-script.py \
  tests/test-sycl-source-attribution-parser.py \
  tests/test-sycl-end-to-end-profiling-docs.py -q
bash scripts/sycl-gptoss-full-attribution-profile.sh
bash scripts/sycl-source-line-debug-matrix.sh
```

Result: `48 passed in 0.97s`; both dry-runs exited 0.

Build/CTest gates used the required oneAPI wrapper:

```bash
set +u
source /opt/intel/oneapi/setvars.sh --force
set -u
./scripts/sycl-build.sh test-sycl-timeline test-sycl-kernel-profiler llama-bench sycl-source-line-probe
ctest --test-dir build -R "test-sycl-timeline|test-sycl-kernel-profiler" -V
```

Results:

- Build: **PASS**.
- CTest: **PASS**, `test-sycl-timeline` and `test-sycl-kernel-profiler` passed.

## Full-attribution GPT-OSS run

Artifact root: `/tmp/sycl_e2e_closure_20260704_170749`

Command shape:

```bash
set +u
source /opt/intel/oneapi/setvars.sh --force
set -u
ONEAPI_DEVICE_SELECTOR=level_zero:1 bash scripts/sycl-gptoss-full-attribution-profile.sh \
  --execute \
  --i-understand-this-runs-gpu-models-and-profilers \
  --out-root /tmp/sycl_e2e_closure_20260704_170749
```

Status: **FAIL-CLOSED**.

Observed failure:

- `scripts/sycl-gptoss-full-attribution-profile.sh` exited `4` while VTune finalized the GPT-OSS run.
- `bench.stderr` recorded the SYCL watchdog at `GET_ROWS` after no GPU progress for 30025 ms.
- The failing UR call ended with `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`, reported as `UR_RESULT_ERROR_OUT_OF_RESOURCES` at the Level Zero backend boundary.
- VTune also reported a corrupted user API trace while finalizing: `Cannot load data file ... userapicollector-...trace (Data file is corrupted)`.

Artifacts produced before fail-closed exit:

- `bench.stdout`, `bench.stderr`, `command.txt`, `setvars.log`
- `raw/kernel/sycl-kernels.csv` and `.json` headers
- VTune result directory with dumped `.zebin` and trace fragments under `vtune/result/`

Artifacts not produced because the run failed before parser stage:

- `raw/timeline/sycl-timeline.json`
- `pti/level-zero-api.jsonl`
- `ur/sycl-ur-trace.log`
- `vtune/exported-kernels.csv`
- `vtune/exported-source-lines.csv`
- `parsed/layer-ledger.parse`

Required status rows for this validation:

```text
coverage.layer_status not_produced_full_run_failed
layer.unknown_wall_ms_x1000 not_produced
```

For continuity with the most recent prior completed profiling artifact (`/tmp/sycl_decode_profile_full_20260704_114044`), the legacy wall ledger still shows incomplete coverage:

```text
ledger.coverage_status coverage_mismatch
ledger.unknown_wall_residual_ms_x1000 2413983
```

## Source-line debug matrix

Artifact root: `/tmp/sycl_source_line_matrix_20260704_171936`

Command shape:

```bash
set +u
source /opt/intel/oneapi/setvars.sh --force
set -u
ONEAPI_DEVICE_SELECTOR=level_zero:1 bash scripts/sycl-source-line-debug-matrix.sh \
  --execute \
  --i-understand-this-runs-gpu-source-probe \
  --out-root /tmp/sycl_source_line_matrix_20260704_171936
```

Status: **COMPLETE, all cases FAIL-CLOSED**. VTune `gpu-source-line` report generation returned `0x40000023 (User input error)` with empty report output for every case, but the matrix now writes explicit checker summaries.

Per-case results:

```text
release_split:
  source_line.debug_line_present 0
  source_line.non_unknown_rows 0
  source_line.required_kernel sycl_source_line_probe
  source_line.blocker missing_debug_line
  source_line.status fail

debug_line_tables:
  source_line.debug_line_present 0
  source_line.non_unknown_rows 0
  source_line.required_kernel sycl_source_line_probe
  source_line.blocker missing_debug_line
  source_line.status fail

debug_full:
  source_line.debug_line_present 1
  source_line.non_unknown_rows 0
  source_line.required_kernel sycl_source_line_probe
  source_line.blocker vtune_unknown_source
  source_line.status fail

debug_no_inline:
  source_line.debug_line_present 1
  source_line.non_unknown_rows 0
  source_line.required_kernel sycl_source_line_probe
  source_line.blocker vtune_unknown_source
  source_line.status fail
```

## MXFP4 exact-line feasibility

Status: **SKIPPED by gate**.

No single-CU matrix case reported `source_line.status pass`, so the lead-only MXFP4 exact-line feasibility runner was not attempted with `--require-matrix-pass`. This preserves the fail-closed rule: exact MXFP4 source-line attempts require a passing single-CU source-line matrix first.

## Source attribution result

Fallback parser check used the prior completed cost ranking from `/tmp/sycl_decode_profile_full_20260704_114044/cost-ranking.parse` plus the current matrix `debug_full` source-line failure:

```bash
python3 scripts/parse-sycl-source-attribution.py \
  --cost-ranking /tmp/sycl_decode_profile_full_20260704_114044/cost-ranking.parse \
  --source-line /tmp/sycl_source_line_matrix_20260704_171936/build-matrix/debug_full/source-line-feasibility.parse \
  --region-map activation/sycl-source-region-map.json
```

Result:

```text
source_attribution.status source_region
source_attribution.kernel mxfp4.gateup.xmx_tiled_dpas_m2
source_attribution.file ggml/src/ggml-sycl/mmvq.cpp
source_attribution.line_start 9730
source_attribution.line_end 9955
source_attribution.label_line 9767
source_attribution.exact_line_blocker vtune_unknown_source
```

Interpretation: fallback source-region attribution is available for the known hottest kernel, but `source_region_plus_ablation` closure was not produced in this validation because no ablation JSON was supplied and exact source lines remained blocked.

## Performance result

The full VTune/trace run failed before `llama-bench` emitted PP/TG rows, so performance was measured with a lead-only FA-on GPT-OSS baseline without VTune/UR/PTI tracing.

Artifact root: `/tmp/sycl_e2e_perf_baseline_20260704_172114`

Command shape:

```bash
set +u
source /opt/intel/oneapi/setvars.sh --force
set -u
ONEAPI_DEVICE_SELECTOR=level_zero:1 \
GGML_SYCL_MOE_PHASE_MATERIALIZE=1 \
GGML_SYCL_MOE_PHASE_BULK_XMX=1 \
GGML_SYCL_MOE_DOWN_SUM_DIRECT=1 \
./build/bin/llama-bench \
  -m /Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf \
  -ngl 99 -fa 1 -p 512 -n 128 -r 1
```

Observed performance:

```text
PP512: 1174.63 tok/s
TG128:   32.27 tok/s
```

Interpretation: PP remains near the strong ~1200 tok/s target, but TG remains below the `>=45 tok/s` target.

## Residual unknowns and follow-ups

- Full layered closure is **not achieved** in this validation because the full GPT-OSS VTune/UR/PTI run failed closed before `parsed/layer-ledger.parse` was produced.
- The full run appears to be perturbed by heavy VTune + UR tracing: it failed at `GET_ROWS` with watchdog/no-progress and `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` / `UR_RESULT_ERROR_OUT_OF_RESOURCES`.
- Level Zero/PTI JSONL and Unified Runtime trace artifacts were not produced in the failed full run, so `coverage.layer_status` could not reach `ok`.
- Exact GPU source-line attribution remains blocked: single-CU debug builds can produce `.debug_line` in `debug_full` and `debug_no_inline`, but VTune `gpu-source-line` export still returns empty/unknown rows.
- Fallback source-region attribution is available for `mxfp4.gateup.xmx_tiled_dpas_m2`; closure-quality `source_region_plus_ablation` still needs an ablation JSON tied to the same top kernel.
- Follow-up: split full attribution into lower-overhead staged runs (bench timeline/kernel first, VTune export second, UR/L0 short trace third) or reduce tracing scope so the full runner can produce `parsed/layer-ledger.parse` without triggering watchdog/OOM.
