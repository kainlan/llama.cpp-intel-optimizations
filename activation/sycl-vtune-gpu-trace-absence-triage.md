# SYCL VTune GPU-Side Trace Absence Triage

Date: 2026-07-05

## Scope and safety

This triage was source/read-only plus source-only tests. I did not run VTune,
GPU/model/profiler commands, `sycl-source-line-probe`, `sycl-kernel-bench`,
`llama-bench`, `sycl-ls`, `/dev/dri`/DRM probes, `lsof`, P2P probes, `/Storage`
commands, or any `--execute` runner.

## Findings

1. **High: VTune/GTPin collected Level Zero host API tasks but no GPU-side trace tables.**
   Evidence:
   - `/tmp/sycl_source_line_matrix_20260705_011045/build-matrix/debug_no_inline/probe.stdout`
     contains Level Zero host tasks (`zeCommandListAppendLaunchKernel` count 100,
     `zeCommandListAppendMemoryCopy`, `zeModuleCreate`) and then the VTune warning:
     `"Trace GPU programming APIs" option was turned ON, but no GPU-side trace data was collected`.
   - All matrix exports are empty:
     `/tmp/sycl_source_line_matrix_20260705_011045/build-matrix/*/vtune-computing-tasks.csv`
     and `*/vtune-gpu-source-line.csv` are 0 bytes.
   - `/opt/intel/oneapi/vtune/2025.10/config/query_library/gpu_data_queries.cfg`
     emits that warning when neither `dma_packet_data` nor `gpu_scheduler_data`
     exists, i.e. the GPU-side trace/viewpoint database tables are absent.

2. **High: this is not primarily a missing-device-DWARF problem.**
   Evidence:
   - `/tmp/sycl_source_line_matrix_20260705_011045/build-matrix/debug_no_inline/source-line-feasibility.parse`
     reports `source_line.debug_line_present 1`, `source_line.dwarf_status ok`,
     `source_line.dwarf_source_rows 295`, and `source_line.dwarf_required_path_present 1`.
   - `debug_full` similarly has usable decoded DWARF rows per
     `activation/sycl-vtune-source-line-enablement-validation.md`.
   - The blocker appears after DWARF survives into dumped ZEBin: VTune never emits
     non-empty computing-task or `gpu-source-line` CSV rows.

3. **High: source-analysis `mem-latency` depends on VTune's GTPin path, and GTPin is a likely failing layer.**
   Evidence:
   - `/opt/intel/oneapi/vtune/2025.10/config/analysis_type/include/gpu_collection_settings.xsl`
     sets `collectGTPin=true` for `gpu-profiling-mode=source-analysis`.
   - `/opt/intel/oneapi/vtune/2025.10/config/analysis_type/gpu_hotspots_base.cfg`
     maps `source-analysis=mem-latency` to `gpuProfilingMode=<BDF>|memlatency`.
   - `/tmp/sycl_source_line_matrix_20260705_011045/build-matrix/debug_no_inline/vtune-source-line/data.0/gtpin.3430664.log`
     says `Not enough free registers while memory-mapped registers (SREGs) are disabled`
     and `GTPin didn't find any kernels... Exiting without doing anything.`

4. **Medium: target-GPU selection was tested but did not resolve the failure.**
   Evidence:
   - `activation/sycl-vtune-source-line-enablement-validation.md` records Matrix B
     with `--vtune-target-gpu 0:7:0.0` and says explicit target GPU did not change
     the empty export outcome.
   - The target-GPU result's `runss.options` restricted collection to
     `--type=level_zero:nostack|0:7:0.0` and `--gpu-profiling-type=0:7:0.0|memlatency`,
     so this is not explained solely by tracing all adapters.

5. **Medium: existing scripts were collecting the intended VTune mode, but the checker collapsed no-GPU-trace into generic `vtune_unknown_source`.**
   Evidence:
   - `scripts/sycl-source-line-debug-matrix.sh` and
     `scripts/sycl-vtune-source-line-feasibility.sh` collect with
     `vtune -collect gpu-hotspots -knob gpu-profiling-mode=source-analysis -knob source-analysis=mem-latency -knob dump-compute-task-binaries=true`.
   - The checker previously only saw the empty CSV and reported
     `source_line.blocker vtune_unknown_source`, hiding the stronger VTune warning.

## Root-cause hypothesis set

1. **Most likely:** VTune 2025.10 source-analysis/GTPin cannot instrument this
   Battlemage Level Zero SYCL kernel path in the current driver/compiler mode.
   The no-inline row proves one GTPin instrumentation failure mode (register
   pressure with SREGs disabled). Other rows still lack GPU-side tables, so the
   broader GTPin/source-analysis path may be unsupported or silently failing.

2. **Likely contributor:** VTune source-analysis requires successful GTPin
   instrumentation in addition to Level Zero API tracing. The run captured host
   Level Zero calls but did not create the GPU scheduler/compute task tables
   needed by `-group-by computing-task` and `gpu-source-line` reports.

3. **Less likely as sole cause:** wrong physical adapter. Explicit B50
   `target-gpu=0:7:0.0` still produced empty task/source exports, though VTune's
   summaries continue to label default adapter metadata as B580 on this host.

4. **Less likely:** missing line info. Debug rows have `.debug_line` and decoded
   source rows; source-line attribution is blocked after the debug-info stage.

## Implemented source-only improvements

- `scripts/check-sycl-vtune-source-lines.py` now accepts optional VTune stdout/stderr
  logs and reports:
  - `source_line.vtune_no_gpu_side_trace`
  - `source_line.gtpin_no_kernels`
  - `source_line.gtpin_register_pressure`
  - blocker `source_line.blocker vtune_no_gpu_side_trace` when the VTune warning is present.
- The debug matrix, MXFP4 feasibility, and full-attribution scripts now pass the
  collected VTune stdout/stderr logs into the checker.
- Staged attribution selection now prefers the explicit `vtune_no_gpu_side_trace`
  blocker before falling back to generic `vtune_unknown_source`.
- `docs/backend/SYCL.md` documents `vtune_no_gpu_side_trace` as a collection-layer
  blocker distinct from missing DWARF or unknown source rows.

## Lead-only validation recommendation

First isolate whether VTune can collect non-GTPin GPU-side task data on the same
probe by switching source-analysis off. This is lead-only and must not be run by
workers:

```bash
set +u
source /opt/intel/oneapi/setvars.sh --force
set -u
OUT=/tmp/sycl_vtune_api_trace_smoke_$(date +%Y%m%d_%H%M%S)
PROBE=/tmp/sycl_source_line_matrix_targetgpu_20260705_011211/build-matrix/debug_full/build/bin/sycl-source-line-probe
ONEAPI_DEVICE_SELECTOR=level_zero:1 vtune -collect gpu-hotspots \
  -knob target-gpu=0:7:0.0 \
  -knob gpu-profiling-mode=characterization \
  -knob characterization-mode=overview \
  -knob dump-compute-task-binaries=true \
  -result-dir "$OUT" -- "$PROBE" --iterations 100 --size 1048576 --json "$OUT/probe.json"
vtune -report hotspots -r "$OUT" -group-by computing-task -format csv > "$OUT/vtune-computing-tasks.csv"
```

Interpretation:
- If this emits computing-task rows, Level Zero API/GPU task tracing works and the
  source-line blocker is specifically the GTPin `source-analysis=mem-latency`
  path. Next try `source-analysis=bb-latency` or a smaller scalar probe, then file
  a VTune/GTPin issue with the register-pressure log.
- If this still reports no GPU-side trace, treat VTune GPU Compute/Media collection
  as unavailable for this host/toolchain and use the non-VTune fallback: dumped
  ZEBin DWARF + EU/ocloc/IGA disassembly + SYCL event/kernel-profile/PTI correlation.

## Residual risks

- This triage did not prove a VTune workaround live because workers are prohibited
  from running VTune/GPU/profiler commands.
- The exact Intel VTune/GTPin limitation may be version-, driver-, kernel-, or
  Battlemage-specific; lead-owned repros are needed before filing upstream.
- `vtune_no_gpu_side_trace` is diagnostic classification, not a fix for exact
  VTune source-line attribution.
