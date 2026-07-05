# SYCL Staged Profiling Closure Validation

## Safe gates

Status: passed for worker-safe gates on 2026-07-04.

Commands run (safe-only):

```bash
python3 -m pytest \
  tests/test-sycl-stage-manifest-parser.py \
  tests/test-sycl-ur-stderr-converter.py \
  tests/test-sycl-vtune-l0-host-task-converter.py \
  tests/test-sycl-staged-ledger-merger.py \
  tests/test-sycl-staged-attribution-profile-script.py \
  tests/test-sycl-ablation-delta-parser.py \
  tests/test-sycl-staged-profiling-docs.py -q
```

Result: `28 passed in 0.66s` for the worker scaffold run; lead rerun also passed with `28 passed in 0.64s`.

```bash
bash scripts/sycl-gptoss-staged-attribution-profile.sh
```

Result: passed dry-run default mode. No `--execute` was used. The dry-run printed the staged plan for output root `/tmp/sycl_gptoss_staged_20260704_201918`, including base, UR, L0, VTune/source-line, ablation, and merge stages.

Lead dry-run artifact note: `/tmp/sycl_staged_dryrun_task9.txt`.

## Build and CTest gates

Status: passed in lead-only validation.

Lead-only commands:

```bash
set +u
source /opt/intel/oneapi/setvars.sh --force
set -u
./scripts/sycl-build.sh test-sycl-timeline test-sycl-kernel-profiler llama-bench sycl-source-line-probe
ctest --test-dir build -R "test-sycl-timeline|test-sycl-kernel-profiler" -V
```

Result: `./scripts/sycl-build.sh test-sycl-timeline test-sycl-kernel-profiler llama-bench sycl-source-line-probe` passed; `ctest` for `test-sycl-timeline` and `test-sycl-kernel-profiler` was 100% passed.

Reference logs: `/home/kainlan/.claude/tmp/pi-bash-45e77fde1f9cefd8.log`; ablation prep build log `/home/kainlan/.claude/tmp/pi-bash-36c9049fa7314681.log`.

## Stage artifacts and manifests

Status: captured in lead-only staged validation.

Artifact root: `/tmp/sycl_staged_closure_20260704_203613`.

Stage roots:

- base: `/tmp/sycl_staged_closure_20260704_203613/base`
- L0: `/tmp/sycl_staged_closure_20260704_203613/l0`
- UR: `/tmp/sycl_staged_closure_20260704_203613/ur`
- VTune/source-line: `/tmp/sycl_staged_closure_20260704_203613/vtune-source`
- ablation: `/tmp/sycl_staged_closure_20260704_203613/ablation`
- merge: `/tmp/sycl_staged_closure_20260704_203613/merged`

Manifest status:

```text
manifest.status ok
manifest.count 5
manifest.schema_version 1
manifest.merge_key 4b696505273d8727
```

Manifest stage roots:

```text
manifest.stage.ablation.root /tmp/sycl_staged_closure_20260704_203613/ablation
manifest.stage.base.root /tmp/sycl_staged_closure_20260704_203613/base
manifest.stage.l0.root /tmp/sycl_staged_closure_20260704_203613/l0
manifest.stage.ur.root /tmp/sycl_staged_closure_20260704_203613/ur
manifest.stage.vtune-source.root /tmp/sycl_staged_closure_20260704_203613/vtune-source
```

Stage notes:

- UR trace was converted from stdout because the stderr converter path found no rows.
- L0 data came from a dedicated VTune `gpu-offload` / host-task summary model run.
- Source-line exact-line attribution was blocked by `vtune_unknown_source`.
- Ablation JSONL came from dry-run microbench prep at `/tmp/sycl_staged_closure_ablation_prep_20260704_203541`.

## Merged ledger result

Status: strict staged ledger coverage reported ok in lead-only validation.

Observed merged ledger fields:

```text
layer.wall_ms_x1000 4510118
layer.app_host_ms_x1000 0
layer.sycl_submit_host_ms_x1000 119397
layer.ur_api_ms_x1000 0
layer.level_zero_api_ms_x1000 6462000
layer.gpu_kernel_ms_x1000 1542400
layer.vtune_gpu_ms_x1000 0
layer.unknown_wall_ms_x1000 0
coverage.layer_status ok
```

Residual caveat: `layer.unknown_wall_ms_x1000` is clamped to `0` because the independent L0 aggregate host-task summary exceeds the base timeline wall. The strict staged ledger still reports `coverage.layer_status ok` with the explicit L0 layer value `layer.level_zero_api_ms_x1000 6462000`.

Merged artifact: `/tmp/sycl_staged_closure_20260704_203613/merged/staged-ledger.parse`.

## Source-line verdict

Status: exact source-line attribution failed closed with an explicit blocker.

Observed fields:

```text
source_line.debug_line_present 1
source_line.non_unknown_rows 0
source_line.required_kernel sycl_source_line_probe
source_line.blocker vtune_unknown_source
source_line.status fail
```

The exact source-line blocker was `vtune_unknown_source`; fallback source attribution reached `source_region_plus_ablation`.

## Ablation attribution

Status: fallback source attribution reached `source_region_plus_ablation` in lead-only validation.

Observed fields:

```text
source_attribution.status source_region_plus_ablation
source_attribution.kernel mxfp4.gateup.xmx_tiled_dpas_m2
source_attribution.file ggml/src/ggml-sycl/mmvq.cpp
source_attribution.line_start 9730
source_attribution.line_end 9955
source_attribution.label_line 9767
source_attribution.exact_line_blocker vtune_unknown_source
source_attribution.ablation_delta_ms_x1000 2267
```

Ablation artifact roots:

- staged ablation output: `/tmp/sycl_staged_closure_20260704_203613/ablation`
- dry-run microbench prep: `/tmp/sycl_staged_closure_ablation_prep_20260704_203541`

## PP/TG guardrails

Status: recorded from lead-only base benchmark.

Observed guardrails:

```text
| model                          |       size |     params | backend    |  ngl | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ---: | -: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | SYCL       |   99 |  1 |           pp512 |       1216.10 ± 0.00 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | SYCL       |   99 |  1 |           tg128 |         33.58 ± 0.00 |

build: ce459e7db (11437)
```

Summary:

- PP512: `1216.10 ± 0.00 t/s`
- TG128: `33.58 ± 0.00 t/s`

## Residual unknowns and follow-ups

- This report records lead-only validation facts; it does not claim or close any tracker.
- Exact source-line attribution remains blocked by `vtune_unknown_source`; the accepted fallback evidence is `source_region_plus_ablation` with ablation delta `2267` ms x1000.
- `layer.unknown_wall_ms_x1000` is clamped to `0` due to the independent L0 host-task aggregate exceeding base timeline wall; the ledger nevertheless reports `coverage.layer_status ok` with explicit layer values.
- This worker did not edit tracker files and did not run real GPU/model/profiler commands.
