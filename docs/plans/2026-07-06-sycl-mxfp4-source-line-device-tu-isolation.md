# SYCL MXFP4 source-line device TU isolation plan

Date: 2026-07-06
Parent tracker: `llama.cpp-f0z3`
Upstream blocker: `llama.cpp-gml4`, `llama.cpp-040b`
Prior plan: `docs/plans/2026-07-06-sycl-mxfp4-source-line-tu-split.md`
Prior validation root: `/tmp/sycl_mxfp4_source_line_probe_split_20260706_100913`

## Goal

Make `sycl-mxfp4-source-line-probe` avoid linking the full `ggml-sycl` / `mmvq.cpp` device image and instead compile only the default MXFP4 pair-GLU source-line kernel family needed by:

```text
mxfp4_pair_glu_xmx_tiled_packed_r8_m2_sparse32_bias
```

The desired outcome is that the selected ZEBin for:

```text
.text._ZTS39mxfp4_pair_glu_xmx_tiled_dpas_m2_kernelILi8ELi3ELb0ELb0EE
```

retains usable `.debug_line` rows so the IGA PC resolver can produce:

```text
source_line.status asm-line-static-cost
```

Minimum success is weaker but still useful:

```text
source_line.debug_line_present 1
```

## Problem statement

The previous probe-only guard narrowed `tools/sycl-kernel-bench/kernels/reference/mxfp4_inline_dot.cpp` for the source-line probe target and preserved normal `sycl-kernel-bench` behavior. That was safe and buildable, but real validation still failed:

```text
source_line.debug_line_present 0
source_line.dwarf_error no source rows found
source_line.blocker missing_debug_line
source_line.status fail
```

The build log still emitted:

```text
warning: VCDebugInfo: only modules with one CU are supported at the moment, the debug information for Module will be dropped out.
```

Conclusion: the remaining multi-CU source of debug-info drop is the linked SYCL device image from `ggml/src/ggml-sycl/mmvq.cpp` / `ggml_sycl_mxfp4_pair_glu_bench_launch`, not the tool-side wrapper TU.

## Preferred implementation direction

Create a source-line-probe-specific device path that is not linked through the full `ggml` target:

1. Add a minimal source-line-only device TU, preferably under `tools/sycl-kernel-bench/kernels/reference/` unless reviewer evidence shows it belongs under `ggml/src/ggml-sycl/`.
2. This TU defines `ggml_sycl::ggml_sycl_mxfp4_pair_glu_bench_launch` for the source-line probe target only.
3. It supports only the default source-line probe variants:
   - `xmx_tiled=true`
   - `xmx_tiled_pack_q8=true`
   - `xmx_tiled_m_tiles=2`
   - `rows_per_wg=8`
   - `xmx_tiles_n=1`
   - `xmx_tiled_prefetch=false`
   - `xmx_tiled_v2=false`
   - `xmx_tiled_bundle4=false`
   - `xmx_tiled_grouped=false`
   - `split_gate_up=false`
   - `predecoded_i8=false`
   - `vector_qs_load=false`
   - `scale_stride_blocks=0`
   - `subgroup_size=32`
   - `gate_bias` / `up_bias` may be non-null.
4. It includes the minimal pack-Q8 kernel plus the selected `mxfp4_pair_glu_xmx_tiled_dpas_m2_kernel<8, GGML_GLU_OP_SWIGLU_OAI, false, false>` kernel, with any required ESIMD load/GLU helpers.
5. Avoid pulling unrelated MMVQ, grouped, layer-glu-down, bundle4, v2, m4, single-column, multirhs, persistent/runtime model code into the probe target.
6. Change `sycl-mxfp4-source-line-probe` to link the minimum required host library, preferably `ggml-base` rather than `ggml`, plus the minimal source-line TU. Do not link full `llama-common` unless a reviewer proves it is necessary.
7. Under `SYCL_MXFP4_SOURCE_LINE_PROBE_ONLY`, avoid host helper dependencies that would require linking the full `ggml-sycl` target. In particular, `select_mxfp4_xmx_tiles_n()` may need a source-line-probe-only path that validates `requested` as `1|2|4` and returns the requested/default value without calling `ggml_sycl_info()`.
8. Preserve normal `sycl-kernel-bench` and llama.cpp runtime behavior. All changes must be target-gated to the source-line probe path.

## Validation constraints

Workers/subagents must not run:

- real GPU/model/profiler commands;
- `/Storage` access;
- `llama-bench`, `sycl-kernel-bench`, `sycl-mxfp4-source-line-probe` execution;
- VTune execute paths;
- `sycl-ls`, `/dev/dri`/DRM, `lsof`, P2P probes.

Workers may run source/static checks such as:

```bash
python3 -m pytest tests/test-sycl-mxfp4-source-line-probe-source.py tests/test-sycl-vtune-source-line-feasibility-script.py -q
git diff --check
bash -n scripts/sycl-vtune-source-line-feasibility.sh
```

Lead-only real commands must source oneAPI as:

```bash
set +u
source /opt/intel/oneapi/setvars.sh --force
set -u
```

Lead build gate:

```bash
./scripts/sycl-build.sh sycl-mxfp4-source-line-probe
```

Lead real feasibility gate:

```bash
scripts/sycl-vtune-source-line-feasibility.sh \
  --execute \
  --i-understand-this-runs-gpu-microbenchmarks \
  --out-root /tmp/sycl_mxfp4_source_line_device_tu_$(date +%Y%m%d_%H%M%S) \
  --build-dir /tmp/sycl_mxfp4_source_line_device_tu_build_$(date +%Y%m%d_%H%M%S) \
  --device-selector level_zero:1 \
  --vtune-target-gpu 0:7:0.0
```

## Task breakdown

### Task A: source-only dependency/link map

Read the relevant code and map the minimal source/link dependencies for a full source-line-probe-only path. Do not edit files.

Files to inspect:

- `tools/sycl-kernel-bench/CMakeLists.txt`
- `tools/sycl-kernel-bench/mxfp4_source_line_probe.cpp`
- `tools/sycl-kernel-bench/kernels/reference/mxfp4_inline_dot.cpp`
- `tools/sycl-kernel-bench/data_generators.hpp`
- `ggml/src/ggml-sycl/ggml-sycl-bench.hpp`
- `ggml/src/ggml-sycl/mmvq.cpp`
- `ggml/src/ggml-sycl/CMakeLists.txt`
- `ggml/src/CMakeLists.txt`

The report must identify:

- exact `mmvq.cpp` helper ranges required by default pack-Q8 + m2 pair-GLU;
- whether CPU validation can replace the non-XMX backend reference launch in source-line probe mode, or whether a minimal reference launch is still needed;
- whether the probe target can link `ggml-base` instead of `ggml` / `llama-common`;
- tests that should enforce the target does not link full `ggml-sycl`.

### Task B: implement source-line-only device path

After Task A, implement the source-line-probe-only device TU/link changes with the smallest safe diff. Add/extend source tests.

Acceptance:

- source/static tests pass;
- `git diff --check` passes;
- normal `sycl-kernel-bench` target remains unchanged by static tests;
- `sycl-mxfp4-source-line-probe` no longer links the full `ggml` target if feasible;
- workers do not run real GPU/profiler/model/harness commands.

### Task C: lead validation/report closure

Lead runs source tests, build gate, and real feasibility gate. Update:

- `activation/sycl-iga-pc-source-line-validation.md`
- `activation/sycl-line-attribution-final.md`
- tracker comments/status

Close `llama.cpp-f0z3` and then `llama.cpp-gml4` only if the target achieves at least `source_line.debug_line_present 1` or if the fail-closed outcome proves a different blocker with evidence and a new follow-up task.

## Review rules

Use fresh reviewers after each implementation commit:

- spec reviewer checks plan compliance and source-line evidence semantics;
- quality reviewer checks build/link risk, source-test coverage, target gating, and normal runtime isolation.

A reviewer failure to run is not a `VERDICT: FAIL`; spawn a fresh reviewer.

## Completion update

Implementation commit: `aec44f0e1 SYCL: isolate MXFP4 source-line probe device TU`.

Implemented source-line-only diagnostic TU:

```text
tools/sycl-kernel-bench/kernels/reference/mxfp4_pair_glu_source_line_device.cpp
```

The diagnostic target now links `ggml-base` plus the minimal device TU for
`sycl-mxfp4-source-line-probe`; normal `sycl-kernel-bench` and runtime backend
paths remain unmodified by this target-specific isolation.

Lead source/static gates:

```text
python3 -m pytest tests/test-sycl-mxfp4-source-line-probe-source.py tests/test-sycl-vtune-source-line-feasibility-script.py -q
15 passed

git diff --check
passed

bash -n scripts/sycl-vtune-source-line-feasibility.sh
passed
```

Lead build gate:

```text
./scripts/sycl-build.sh sycl-mxfp4-source-line-probe
Build succeeded
```

Lead-only real feasibility gate:

```text
OUT_ROOT=/tmp/sycl_mxfp4_source_line_device_tu_20260706_112515
BUILD_DIR=/tmp/sycl_mxfp4_source_line_device_tu_build_20260706_112515
```

Result:

```text
source_line.debug_line_present 1
source_line.dwarf_status ok
source_line.dwarf_source_rows 1194
source_line.dwarf_required_path_present 1
source_line.asm_source_line_rows 65
source_line.source_attribution_mode asm-line-static
source_line.blocker none
source_line.status asm-line-static-cost
```

The selected section remained the intended MXFP4 pair-GLU compute section:

```text
.text._ZTS39mxfp4_pair_glu_xmx_tiled_dpas_m2_kernelILi8ELi3ELb0ELb0EE
```

Conclusion: the plan achieved full success, not just the minimum
`source_line.debug_line_present 1` threshold. MXFP4 pair-GLU static source-line
attribution is validated as `asm-line-static-cost`; sampled VTune exact source
rows remain unavailable and retain separate `pass` semantics.
