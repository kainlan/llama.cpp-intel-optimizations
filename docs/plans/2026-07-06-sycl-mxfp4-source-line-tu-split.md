# SYCL MXFP4 source-line TU split plan

Date: 2026-07-06
Parent tracker: `llama.cpp-gml4`
Research: `activation/sycl-mxfp4-debugline-blocker-web-research.md`

## Goal

Make the MXFP4 source-line feasibility path compile a device image small enough to preserve ZEBin `.debug_line`, without changing normal `sycl-kernel-bench` or runtime behavior.

Current state before this plan:

- Probe IGA path validates on the standalone source-line probe as `source_line.status asm-line-static-cost`.
- MXFP4 full and narrow executable attempts still fail with `source_line.debug_line_present 0`.
- Research found Intel Graphics Compiler VectorCompiler drops module debug info unless a device LLVM module has exactly one DWARF CU.
- `sycl-mxfp4-source-line-probe` is narrower at the executable level, but before this plan it still compiled the large multi-kernel `mxfp4_inline_dot.cpp` device image.

2026-07-06 result:

- `SYCL_MXFP4_SOURCE_LINE_PROBE_ONLY=1` now guards unrelated `mxfp4_inline_dot.cpp` benchmark families for the probe target only.
- The probe target builds and runs, but real validation still reports `source_line.debug_line_present 0`, `source_line.blocker missing_debug_line`, and `source_line.status fail`.
- The remaining blocker appears to be the linked SYCL device module containing `mmvq.cpp` / `ggml_sycl_mxfp4_pair_glu_bench_launch`, not just the tool-side wrapper TU.

## Non-goals

- Do not change default MXFP4 benchmark/runtime behavior.
- Do not run real GPU/profiler/model commands from teammates.
- Do not relabel fallback coverage as `pass` / exact source-line.
- Do not rely on `-gline-tables-only`; local `icpx` ignores it for `spir64_gen`.

## Implementation strategy

Prefer a low-risk source-line-only guard before a larger source split:

1. Add a source-line-probe-only compile definition, e.g. `SYCL_MXFP4_SOURCE_LINE_PROBE_ONLY`, to the `sycl-mxfp4-source-line-probe` target only.
2. In `tools/sycl-kernel-bench/kernels/reference/mxfp4_inline_dot.cpp`, guard unrelated benchmark families so that the probe target compiles only the helpers and `run_mxfp4_pair_glu` path needed by `mxfp4_pair_glu_xmx_tiled_packed_r8_m2_sparse32_bias`.
3. Preserve the full file for the normal `sycl-kernel-bench` target.
4. Keep IGA/VTune runner fallback section matching explicit for the default kernel.
5. Add source tests that enforce the probe target has the guard and that normal `sycl-kernel-bench` still builds from the unguarded source list.

If macro guarding becomes too invasive, stop and document that a cleaner `mxfp4_pair_glu_source_line.cpp` split is required.

## Task breakdown

### Task A: Source-only dependency map

Read `mxfp4_inline_dot.cpp` and identify the smallest contiguous regions that must remain for `run_mxfp4_pair_glu(... packed r8 m2 sparse32 bias ...)` to compile. Produce a written map in the tracker or report; do not edit source.

Key expected regions include:

- top includes and namespace;
- common MXFP4 layout helpers used by pair-GLU;
- `run_mxfp4_pair_glu` and its directly called helper launches;
- any validation/reference helpers needed when `--validate` is true.

### Task B: Probe-only guard implementation

Apply the smallest safe `SYCL_MXFP4_SOURCE_LINE_PROBE_ONLY` guard to `mxfp4_inline_dot.cpp` and CMake. Keep normal target behavior unchanged. Update tests.

Acceptance:

- source tests pass;
- `./scripts/sycl-build.sh sycl-mxfp4-source-line-probe` passes (lead may run if worker cannot);
- no real GPU/profiler/model commands from workers.

### Task C: Lead validation and closure

Lead runs the build and, if build passes, the real source-line feasibility command. Record whether the selected ZEBin has `.debug_line` and whether `asm-line-static-cost` validates. Update `activation/sycl-iga-pc-source-line-validation.md`, `activation/sycl-line-attribution-final.md`, and tracker status.

## Worker safety constraints

Workers/subagents must not run:

- real GPU/model/profiler commands;
- `/Storage` access;
- `llama-bench`, `sycl-kernel-bench`, `sycl-mxfp4-source-line-probe` execution;
- VTune execute paths;
- `sycl-ls`, `/dev/dri`/DRM, `lsof`, P2P probes.

Workers may run source/static checks such as Python tests, `bash -n`, `git diff --check`, and read-only code inspection.

## Lead validation command template

Lead-only real commands must source oneAPI as:

```bash
set +u
source /opt/intel/oneapi/setvars.sh --force
set -u
```

Build gate:

```bash
./scripts/sycl-build.sh sycl-mxfp4-source-line-probe
```

Real feasibility gate after build succeeds:

```bash
scripts/sycl-vtune-source-line-feasibility.sh \
  --execute \
  --i-understand-this-runs-gpu-microbenchmarks \
  --out-root /tmp/sycl_mxfp4_source_line_probe_split_$(date +%Y%m%d_%H%M%S) \
  --build-dir /tmp/sycl_mxfp4_source_line_probe_split_build_$(date +%Y%m%d_%H%M%S) \
  --device-selector level_zero:1 \
  --vtune-target-gpu 0:7:0.0
```

If a fresh standalone-probe matrix artifact exists, `--require-matrix-pass <path>` may be added as an optional preflight. Do not hard-code stale `/tmp` paths in reproducible commands; the 2026-07-06 rerun omitted this preflight because the old matrix artifact had already been cleaned up.

## Success criteria

Minimum success:

- source-line probe target still builds;
- selected ZEBin changes from `source_line.debug_line_present 0` to `1`.

Full success:

- `source_line.status asm-line-static-cost` for the MXFP4 target.

Fail-closed acceptable outcome:

- if build or validation proves macro guarding cannot reduce the device image safely, document why and leave `llama.cpp-gml4` open for a clean source split.
