# SYCL VTune Source-Line Enablement Validation

Date: 2026-07-05
Branch: `feature/sycl-mxfp4-tg-runtime`
Lead task: `llama.cpp-9eo2`
Follow-up blocker: `llama.cpp-0a68`

## Safe gates

Command:

```bash
python3 -m pytest \
  tests/test-sycl-zebin-line-table-parser.py \
  tests/test-sycl-vtune-source-line-checker.py \
  tests/test-sycl-vtune-task-parser.py \
  tests/test-sycl-source-line-debug-matrix-script.py \
  tests/test-sycl-vtune-source-line-feasibility-script.py \
  tests/test-sycl-vtune-source-line-enablement-docs.py -q
bash -n scripts/sycl-source-line-debug-matrix.sh
bash -n scripts/sycl-vtune-source-line-feasibility.sh
bash scripts/sycl-source-line-debug-matrix.sh --dry-run >/tmp/sycl-source-line-debug-matrix-dryrun.txt
bash scripts/sycl-vtune-source-line-feasibility.sh --dry-run >/tmp/sycl-vtune-source-line-feasibility-dryrun.txt
```

Result:

```text
40 passed in 0.84s
bash -n passed for both scripts
both dry-runs returned 0
```

## Build gate

Command:

```bash
set +u
source /opt/intel/oneapi/setvars.sh --force
set -u
./scripts/sycl-build.sh sycl-source-line-probe
```

Result: PASS. `sycl-source-line-probe` built successfully in `build/bin/`.

## Probe matrix artifacts

Two lead-only probe matrices were run. No model, `/Storage`, `llama-bench`, `sycl-kernel-bench`, DRM, `sycl-ls`, `lsof`, or P2P probe was used.

### Matrix A: no explicit VTune target GPU

Root:

```text
/tmp/sycl_source_line_matrix_20260705_011045
```

Command:

```bash
set +u
source /opt/intel/oneapi/setvars.sh --force
set -u
bash scripts/sycl-source-line-debug-matrix.sh \
  --execute \
  --i-understand-this-runs-gpu-source-probe \
  --out-root /tmp/sycl_source_line_matrix_20260705_011045 \
  --device-selector level_zero:1
```

### Matrix B: explicit local B50 VTune target GPU

Root:

```text
/tmp/sycl_source_line_matrix_targetgpu_20260705_011211
```

Command:

```bash
set +u
source /opt/intel/oneapi/setvars.sh --force
set -u
bash scripts/sycl-source-line-debug-matrix.sh \
  --execute \
  --i-understand-this-runs-gpu-source-probe \
  --out-root /tmp/sycl_source_line_matrix_targetgpu_20260705_011211 \
  --device-selector level_zero:1 \
  --vtune-target-gpu 0:7:0.0
```

## ZEBin line-table diagnostics

The initial matrix run exposed an implementation issue: DPC++ emitted only `main.cpp` for the tiny probe source file in device DWARF, while the runner required `tools/sycl-source-line-probe/main.cpp`. Follow-up task `llama.cpp-lhh9` fixed the runner to require `main.cpp`, and both spec and quality reviews passed.

After that fix, both matrices showed useful DWARF line-table evidence for the debug builds:

| Matrix | Row | `debug_line_present` | `dwarf_status` | `dwarf_source_rows` | required path |
|---|---|---:|---|---:|---:|
| A | `debug_full` | 1 | ok | 17 | 1 |
| A | `debug_no_inline` | 1 | ok | 295 | 1 |
| B | `debug_full` | 1 | ok | 17 | 1 |
| B | `debug_no_inline` | 1 | ok | 295 | 1 |

`release_split` and `debug_line_tables` did not contain decoded source rows and failed closed with `failed to check source lines: no source rows found`.

## VTune task selection

VTune task export failed in every matrix row for both Matrix A and Matrix B:

```text
vtune: Error: 0x40000023 (User input error)
Empty request output.
failed to parse VTune tasks: missing VTune task header
```

The exported task CSVs were zero bytes:

```text
vtune-computing-tasks.csv size: 0 bytes for all rows in both matrices
```

Explicit `--vtune-target-gpu 0:7:0.0` did not change this outcome.

## Probe source-line verdict

No matrix row reached `source_line.status pass`.

Best rows after the probe path fix:

```text
source_line.debug_line_present 1
source_line.non_unknown_rows 0
source_line.required_kernel sycl_source_line_probe
source_line.dwarf_status ok
source_line.dwarf_source_rows 17
source_line.dwarf_required_path_present 1
source_line.blocker vtune_unknown_source
source_line.status fail
```

```text
source_line.debug_line_present 1
source_line.non_unknown_rows 0
source_line.required_kernel sycl_source_line_probe
source_line.dwarf_status ok
source_line.dwarf_source_rows 295
source_line.dwarf_required_path_present 1
source_line.blocker vtune_unknown_source
source_line.status fail
```

The VTune `gpu-source-line` CSVs were zero bytes for all rows in both matrices, so `source_line.non_unknown_rows` stayed `0`.

## MXFP4 feasibility verdict

MXFP4 exact-line feasibility was **not run**. The plan requires a passing `sycl-source-line-probe` matrix row (`source_line.status pass`) before attempting MXFP4 exact-line validation with `--require-matrix-pass`, and no probe row passed.

## Residual blockers and follow-ups

Exact VTune GPU source-line attribution remains blocked by `vtune_unknown_source`: decoded ZEBin DWARF line tables are present and useful in debug builds, but VTune report exports for computing tasks and GPU source lines are empty and return `0x40000023`.

Follow-up issue filed:

```text
llama.cpp-0a68 [SYCL-SOURCE-LINE] Resolve VTune empty source-line/task exports or build non-VTune resolver
```

Recommended next step: determine whether VTune can be made to emit GPU computing-task and `gpu-source-line` rows for this BMG/B50 toolchain. If not, bypass VTune source-line CSVs and build exact attribution from dumped ZEBin DWARF plus EU assembly/IGA/ocloc/PTI correlation.
