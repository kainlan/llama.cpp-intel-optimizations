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

Result: `28 passed in 0.66s`.

```bash
bash scripts/sycl-gptoss-staged-attribution-profile.sh
```

Result: passed dry-run default mode. No `--execute` was used. The dry-run printed the staged plan for output root `/tmp/sycl_gptoss_staged_20260704_201918`, including base, UR, L0, VTune/source-line, ablation, and merge stages.

## Build and CTest gates

Status: lead-only placeholder.

Build and CTest gates require oneAPI setup and are reserved for the lead:

```bash
set +u
source /opt/intel/oneapi/setvars.sh --force
set -u
./scripts/sycl-build.sh test-sycl-timeline test-sycl-kernel-profiler llama-bench sycl-source-line-probe
ctest --test-dir build -R "test-sycl-timeline|test-sycl-kernel-profiler" -V
```

Result: pending lead-only execution.

## Stage artifacts and manifests

Status: lead-only placeholder.

Expected stages:

- base: pending lead-only staged run
- L0: pending lead-only staged run
- UR: pending lead-only staged run
- VTune/source-line: pending lead-only staged run
- ablation: pending lead-only staged run
- merge: pending lead-only staged run

Artifact root(s): pending lead-only execution. Safe dry-run preview root: `/tmp/sycl_gptoss_staged_20260704_201918`.

Manifest merge key: pending lead-only execution.

## Merged ledger result

Status: pending lead-only execution.

Required strict closure signals:

```text
coverage.layer_status ok
layer.unknown_wall_ms_x1000 <integer>
source_attribution.status source_region_plus_ablation
```

Observed merged ledger fields:

- `coverage.layer_status`: pending
- `layer.unknown_wall_ms_x1000`: pending

## Source-line verdict

Status: pending lead-only execution.

Observed fields:

- `source_line.status`: pending
- `source_line.blocker`: pending

Exact source-line attribution may fail closed when a blocker is recorded and fallback source attribution reaches `source_region_plus_ablation`.

## Ablation attribution

Status: pending lead-only execution.

Observed fields:

- `source_attribution.status`: pending
- `source_attribution.ablation_delta_ms_x1000`: pending

Strict closure must not accept plain `source_region`.

## PP/TG guardrails

Status: pending lead-only execution.

Observed guardrails:

- PP512: pending
- TG128: pending

Use the FA-on safe baseline command documented in `activation/sycl-end-to-end-profiling-closure-validation.md` for lead-only guardrail recording.

## Residual unknowns and follow-ups

- Lead-only build/CTest gates are pending.
- Lead-only staged run is pending.
- Strict closure is not established until the merged ledger contains `coverage.layer_status ok` and `layer.unknown_wall_ms_x1000 <integer>`.
- This worker did not edit tracker files and did not run real GPU/model/profiler commands.
