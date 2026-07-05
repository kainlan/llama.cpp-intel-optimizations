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

## Follow-up closure: DWARF line-table fallback

Follow-up task `llama.cpp-0a68` confirmed VTune still does not provide sampled GPU source-line rows on this BMG/B50 stack, but the dumped ZEBin DWARF line tables are sufficient for a truthful non-VTUNE fallback status.

Additional implementation commits added:

- `f4507b9e5` — convert decoded ZEBin line tables to checker-compatible source CSV rows.
- `70e0e0e40` — classify VTune/GTPin GPU trace absence explicitly.
- `55768b1bf` — wire the DWARF CSV fallback through matrix/MXFP4/full/staged runners.
- `4036d6812` — avoid false exact attribution when source-line kernel and hot kernel differ.
- `170dd57ee` — keep MXFP4 feasibility alive when VTune report exports are empty.
- `69b3a6a54` — match MXFP4 DPC++ basename `mmvq.cpp` in device DWARF.

Updated safe gate:

```text
bash -n scripts/sycl-source-line-debug-matrix.sh \
       scripts/sycl-vtune-source-line-feasibility.sh \
       scripts/sycl-gptoss-full-attribution-profile.sh \
       scripts/sycl-gptoss-staged-attribution-profile.sh
python3 -m pytest <source-line parser/checker/runner/merger suite> -q

81 passed in 1.61s
```

### Matrix C: DWARF fallback enabled

Root:

```text
/tmp/sycl_source_line_matrix_dwarf_20260705_020828
```

Command:

```bash
set +u
source /opt/intel/oneapi/setvars.sh --force
set -u
scripts/sycl-source-line-debug-matrix.sh \
  --execute \
  --i-understand-this-runs-gpu-source-probe \
  --out-root /tmp/sycl_source_line_matrix_dwarf_20260705_020828 \
  --device-selector level_zero:1 \
  --vtune-target-gpu 0:7:0.0
```

Passing probe fallback rows:

```text
# debug_full
source_line.debug_line_present 1
source_line.non_unknown_rows 0
source_line.vtune_sampled_non_unknown_rows 0
source_line.vtune_no_gpu_side_trace 1
source_line.required_kernel sycl_source_line_probe
source_line.dwarf_status ok
source_line.dwarf_source_rows 17
source_line.dwarf_required_path_present 1
source_line.dwarf_source_line_rows 17
source_line.allow_dwarf_line_table_only 1
source_line.source_attribution_mode dwarf-line-table
source_line.blocker none
source_line.status dwarf-line-table-only
```

```text
# debug_no_inline
source_line.debug_line_present 1
source_line.non_unknown_rows 0
source_line.vtune_sampled_non_unknown_rows 0
source_line.vtune_no_gpu_side_trace 1
source_line.gtpin_no_kernels 1
source_line.gtpin_register_pressure 1
source_line.required_kernel sycl_source_line_probe
source_line.dwarf_status ok
source_line.dwarf_source_rows 295
source_line.dwarf_required_path_present 1
source_line.dwarf_source_line_rows 295
source_line.allow_dwarf_line_table_only 1
source_line.source_attribution_mode dwarf-line-table
source_line.blocker none
source_line.status dwarf-line-table-only
```

`release_split` and `debug_line_tables` still fail closed with `failed to check source lines: no source rows found`, as expected.

### MXFP4 feasibility verdict

Root:

```text
/tmp/sycl_vtune_source_line_dwarf_final_20260705_023133
```

Build root:

```text
/tmp/sycl_vtune_source_line_build_final_20260705_023133
```

Command:

```bash
set +u
source /opt/intel/oneapi/setvars.sh --force
set -u
scripts/sycl-vtune-source-line-feasibility.sh \
  --execute \
  --i-understand-this-runs-gpu-microbenchmarks \
  --out-root /tmp/sycl_vtune_source_line_dwarf_final_20260705_023133 \
  --build-dir /tmp/sycl_vtune_source_line_build_final_20260705_023133 \
  --device-selector level_zero:1 \
  --vtune-target-gpu 0:7:0.0 \
  --require-matrix-pass /tmp/sycl_source_line_matrix_dwarf_20260705_020828/build-matrix/debug_full/source-line-feasibility.parse
```

Result:

```text
source_line.debug_line_present 1
source_line.non_unknown_rows 0
source_line.vtune_sampled_non_unknown_rows 0
source_line.vtune_no_gpu_side_trace 0
source_line.gtpin_no_kernels 0
source_line.gtpin_register_pressure 1
source_line.required_kernel mxfp4_pair_glu_xmx_tiled_packed_r8_m2_sparse32_bias
source_line.dwarf_status ok
source_line.dwarf_source_rows 923
source_line.dwarf_required_path_present 1
source_line.dwarf_source_line_rows 923
source_line.allow_dwarf_line_table_only 1
source_line.source_attribution_mode dwarf-line-table
source_line.blocker none
source_line.status dwarf-line-table-only
```

VTune did emit a computing-task report in this microbench run, but the `gpu-source-line` report still had no non-unknown sampled rows. The selected task was:

```text
vtune_task.status ok
vtune_task.match mxfp4_pair_glu_xmx_tiled
vtune_task.selected mxfp4_pair_glu_xmx_tiled_dpas_m2_kernel<(int)8, (int)3, (bool)0, (bool)0>
vtune_task.selected_time_ms_x1000 0
vtune_task.count 2
```

Downstream attribution parser verification:

```text
source_attribution.status dwarf_line_table_only
source_attribution.kernel mxfp4_pair_glu_xmx_tiled_packed_r8_m2_sparse32_bias
source_attribution.source_line_kernel mxfp4_pair_glu_xmx_tiled_packed_r8_m2_sparse32_bias
source_attribution.source_line_status dwarf-line-table-only
```

## Final status and residual limitation

The blocker `llama.cpp-0a68` is resolved by the non-VTUNE DWARF line-table fallback. The status is deliberately **not** `pass` / `exact_source_line`; it is `dwarf-line-table-only`, meaning the ZEBin contains concrete decoded source-line rows for the target kernel, but VTune did not provide sampled per-source-line attribution.

Residual limitation: exact sampled VTune GPU source-line attribution remains unavailable on this stack (`source_line.non_unknown_rows 0`). This is now classified and fail-closed instead of blocking source-line coverage work. Any future EU assembly / PC-sampling resolver should add a new status rather than relabeling `dwarf-line-table-only` as exact sampled VTune evidence.
