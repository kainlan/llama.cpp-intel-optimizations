# SYCL IGA PC Source-Line Attribution Validation

Date: 2026-07-05
Tracker issue: `llama.cpp-040b`
Validation owner: lead

## Summary

Lead validation was executed on B50 (`ONEAPI_DEVICE_SELECTOR=level_zero:1`) with oneAPI sourced via:

```bash
set +u
source /opt/intel/oneapi/setvars.sh --force
set -u
```

Result: **IGA static attribution is now validated on the standalone probe, but still not validated for the MXFP4 target kernel.**

Required semantics remain:

```text
source_line.status asm-line-static-cost means exact static source-line cost from IGA PC rows, not sampled runtime timing.
source_line.status pass remains the only sampled VTune exact status.
```

## Fixes validated in this round

Two previously-open implementation gaps were closed:

1. `scripts/parse-sycl-iga-pc-disasm.py` now parses real IGA JSON v2 output with top-level `elems` and instruction rows marked `kind: "I"`.
2. `scripts/sycl-vtune-source-line-feasibility.sh` now selects the archived ZEBin containing the VTune-selected compute task section instead of using the first archived ZEBin.  Template task names such as:

```text
mxfp4_pair_glu_xmx_tiled_dpas_m2_kernel<(int)8, (int)3, (bool)0, (bool)0>
```

are converted to a mangled section substring:

```text
mxfp4_pair_glu_xmx_tiled_dpas_m2_kernelILi8ELi3ELb0ELb0E
```

The ZEBin lookup is pipefail-safe and records `iga-section-selection.parse`.

## Probe matrix validation: passed static attribution

Artifact root:

```text
/tmp/sycl_source_line_iga_matrix_fix_20260705_192345
```

Selected parse:

```text
/tmp/sycl_source_line_iga_matrix_fix_20260705_192345/build-matrix/debug_full/source-line-feasibility.parse
```

Selected parse contents:

```text
source_line.debug_line_present 1
source_line.non_unknown_rows 0
source_line.vtune_sampled_non_unknown_rows 0
source_line.vtune_no_gpu_side_trace 1
source_line.gtpin_no_kernels 0
source_line.gtpin_register_pressure 0
source_line.required_kernel sycl_source_line_probe
source_line.dwarf_status ok
source_line.dwarf_source_rows 17
source_line.dwarf_required_path_present 1
source_line.dwarf_source_line_rows 17
source_line.allow_dwarf_line_table_only 1
source_line.asm_source_line_rows 2
source_line.allow_asm_line_static_cost 1
source_line.asm_top_source_line main.cpp:148
source_line.asm_top_static_score 13
source_line.asm_top_instruction_count 13
source_line.source_attribution_mode asm-line-static
source_line.blocker none
source_line.status asm-line-static-cost
```

IGA manifest for selected row:

```json
{
  "extract.kernel_match": "sycl_source_line_probe",
  "extract.platform": "xe2",
  "extract.section": ".text._ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E29sycl_source_line_probe_kernel",
  "extract.section_addr": "0x0",
  "extract.status": "ok"
}
```

IGA resolver summary:

```text
asm_source.status ok
asm_source.mapped_instruction_count 18
asm_source.unmapped_instruction_count 29
asm_source.source_line_rows 2
asm_source.top_source_line main.cpp:148
asm_source.top_static_score 13
asm_source.top_instruction_count 13
```

Interpretation: the non-VTune IGA PC path is real for a narrow probe artifact. It maps numeric IGA PCs through DWARF line ranges and produces `source_line.status asm-line-static-cost` without sampled runtime timing.

## MXFP4 feasibility validation: still blocked

Latest artifact roots:

```text
/tmp/sycl_mxfp4_iga_source_line_fix4_20260705_210052
/tmp/sycl_mxfp4_iga_source_line_build_fix4_20260705_210052
```

Original run result before structured checker replay:

```text
failed to check source lines: no source rows found
```

Structured checker replay after the fail-closed parser improvement:

```text
source_line.debug_line_present 0
source_line.non_unknown_rows 0
source_line.vtune_sampled_non_unknown_rows 0
source_line.vtune_no_gpu_side_trace 0
source_line.gtpin_no_kernels 0
source_line.gtpin_register_pressure 1
source_line.required_kernel mxfp4_pair_glu_xmx_tiled_packed_r8_m2_sparse32_bias
source_line.dwarf_status error
source_line.dwarf_error no source rows found
source_line.dwarf_source_rows 0
source_line.dwarf_required_path_present 0
source_line.source_attribution_mode none
source_line.blocker missing_debug_line
source_line.status fail
```

The runner now selects the VTune task and matching compute ZEBin correctly:

```text
selected_task mxfp4_pair_glu_xmx_tiled_dpas_m2_kernel<(int)8, (int)3, (bool)0, (bool)0>
iga_section_match mxfp4_pair_glu_xmx_tiled_dpas_m2_kernelILi8ELi3ELb0ELb0E
selected_zebin /tmp/sycl_mxfp4_iga_source_line_fix4_20260705_210052/vtune-source-line/archive/binaries/6553b51253c6c2a5.zebin/de3f60dd42e4af96a15eaccab47b0a38/6553b51253c6c2a5.zebin
```

MXFP4 IGA manifest:

```json
{
  "extract.kernel_match": "mxfp4_pair_glu_xmx_tiled_dpas_m2_kernelILi8ELi3ELb0ELb0E",
  "extract.platform": "xe2",
  "extract.section": ".text._ZTS39mxfp4_pair_glu_xmx_tiled_dpas_m2_kernelILi8ELi3ELb0ELb0EE",
  "extract.section_addr": "0x0",
  "extract.status": "ok"
}
```

IGA parser output exists for the selected section:

```text
1583 /tmp/sycl_mxfp4_iga_source_line_fix4_20260705_210052/iga-pc-instructions.csv
```

However, the selected compute ZEBin has no `.debug_line` section / usable DWARF source rows:

```text
source_line.debug_line_present 0
failed to convert ZEBin line table: no source rows found
failed to resolve ZEBin ASM source lines: no source rows found
```

Build logs also show the relevant Intel device debug-info warning:

```text
warning: VCDebugInfo: only modules with one CU are supported at the moment, the debug information for Module will be dropped out.
```

The latest full-benchmark run had 20 such warnings. A global `-g -fdebug-info-for-profiling -fsycl-instrument-device-code` experiment against the whole target failed the compiler backend (`gen compiler command failed with exit code 254`), so the runner was left at the safer `RelWithDebInfo` target-debug configuration.

A narrower diagnostic executable, `sycl-mxfp4-source-line-probe`, was added and built successfully. It links only `mxfp4_source_line_probe.cpp` plus `kernels/reference/mxfp4_inline_dot.cpp`, uses `sycl::gpu_selector_v`, and reuses the same pair-GLU launcher. Validation root:

```text
/tmp/sycl_mxfp4_source_line_probe_gpu_20260705_222039
```

That attempt also failed to produce source-line DWARF:

```text
source_line.debug_line_present 0
source_line.vtune_no_gpu_side_trace 1
source_line.gtpin_register_pressure 1
source_line.dwarf_status error
source_line.dwarf_error no source rows found
source_line.blocker missing_debug_line
source_line.status fail
```

The narrow target still compiles the large `mxfp4_inline_dot.cpp` device image, so it archives all MXFP4 sections and does not by itself solve the missing `.debug_line` problem. The runner now has an explicit fallback section match for the default registry kernel so an empty VTune computing-task report does not degrade to an ambiguous broad section match.

Interpretation: the remaining MXFP4 blocker is no longer IGA parsing or task-to-section selection. It is that the selected compute ZEBin lacks `.debug_line` / usable source-line DWARF, so numeric IGA PCs cannot be mapped to `mmvq.cpp` line rows.

## Final status

- Probe static IGA line attribution: **validated** as `source_line.status asm-line-static-cost`.
- MXFP4 static IGA line attribution: **blocked** by missing `.debug_line` / ZEBin DWARF source rows for the selected compute task.
- Runtime sampled source-line attribution: still unavailable; VTune `pass` remains separate and future sampled PC CSV work remains separate.
- `llama.cpp-040b` should remain open for MXFP4-specific source-row enablement.

## Follow-up blocker

The next fix should make the MXFP4 selected compute ZEBin contain usable source-line DWARF without breaking the compiler backend. Candidate directions:

1. split or guard `mxfp4_inline_dot.cpp` so the source-line target compiles only the needed pair-GLU kernel family, not the whole multi-kernel device image;
2. investigate oneAPI device debug flags that preserve ZEBin line tables for this multi-kernel `sycl-kernel-bench` target;
3. keep `sycl-mxfp4-source-line-probe` as the narrow execution harness once the device image itself can be reduced or made to preserve `.debug_line`.

Label-only `ocloc` rows such as `L0:` are still not byte PC evidence and must not be promoted to `asm-line-static-cost`.

## 2026-07-06 probe-only TU guard validation: still blocked

Follow-up implementation for `llama.cpp-gml4` / `llama.cpp-qqux` added a target-only
`SYCL_MXFP4_SOURCE_LINE_PROBE_ONLY=1` definition to `sycl-mxfp4-source-line-probe`
and guarded unrelated MXFP4 benchmark families in
`tools/sycl-kernel-bench/kernels/reference/mxfp4_inline_dot.cpp`. The normal
`sycl-kernel-bench` target remains unguarded.

Source/build gates:

```text
python3 -m pytest tests/test-sycl-mxfp4-source-line-probe-source.py tests/test-sycl-vtune-source-line-feasibility-script.py -q
13 passed

./scripts/sycl-build.sh sycl-mxfp4-source-line-probe
Build succeeded
```

Lead-only real validation was rerun with oneAPI sourced via the required
`set +u` / `setvars.sh` / `set -u` sequence. The old probe-matrix preflight
artifact under `/tmp/sycl_source_line_iga_matrix_fix_20260705_192345` was no
longer present, so the run was repeated without that stale preflight gate.

Validation artifacts:

```text
OUT_ROOT=/tmp/sycl_mxfp4_source_line_probe_split_20260706_100913
BUILD_DIR=/tmp/sycl_mxfp4_source_line_probe_split_build_20260706_100913
```

The diagnostic executable itself ran successfully under VTune:

```json
{"kernel":"mxfp4_pair_glu_xmx_tiled_packed_r8_m2_sparse32_bias","ok":true,"latency_us":1408.996940,"throughput_tps":709.724749,"throughput_tops":0.000000,"bandwidth_gbps":25.119118,"error":""}
```

However, the selected compute ZEBin still has no usable source-line DWARF:

```text
source_line.debug_line_present 0
source_line.non_unknown_rows 0
source_line.vtune_sampled_non_unknown_rows 0
source_line.vtune_no_gpu_side_trace 1
source_line.gtpin_no_kernels 0
source_line.gtpin_register_pressure 1
source_line.required_kernel mxfp4_pair_glu_xmx_tiled_packed_r8_m2_sparse32_bias
source_line.dwarf_status error
source_line.dwarf_error no source rows found
source_line.dwarf_source_rows 0
source_line.dwarf_required_path_present 0
source_line.source_attribution_mode none
source_line.blocker missing_debug_line
source_line.status fail
```

The section-selection fallback still selected the intended kernel section:

```text
selected_task
iga_section_match mxfp4_pair_glu_xmx_tiled_dpas_m2_kernelILi8ELi3ELb0ELb0E
selected_zebin /tmp/sycl_mxfp4_source_line_probe_split_20260706_100913/vtune-source-line/archive/binaries/94dd41225bbcd53f.zebin/284d5e809b1039b5343acc7b68814267/94dd41225bbcd53f.zebin
```

IGA extraction still succeeds (`extract.status ok`) and `iga-pc-instructions.csv`
is present, but both DWARF conversion and IGA PC source-line resolution fail with
`no source rows found`. The profiling-debug build log continues to emit the
known IGC blocker:

```text
warning: VCDebugInfo: only modules with one CU are supported at the moment, the debug information for Module will be dropped out.
```

Interpretation: guarding unrelated `mxfp4_inline_dot.cpp` benchmark families is
safe and narrows the diagnostic target, but it is **not sufficient** to make the
selected MXFP4 compute ZEBin retain `.debug_line`. The remaining source-row
enablement likely has to reduce or isolate the linked SYCL device module that
contains `mmvq.cpp` / `ggml_sycl_mxfp4_pair_glu_bench_launch`, not only the tool
side host/benchmark wrapper. `llama.cpp-040b` remains open.
