# SYCL line-attribution capability final report

Date: 2026-07-05
Tracker issue: `llama.cpp-040b`
Validation owner: lead

## Static line attribution

Status: **validated for the standalone probe and for the MXFP4 pair-GLU target kernel after source-line-only device TU isolation.**

Latest lead validation artifacts:

- Fixed probe matrix root: `/tmp/sycl_source_line_iga_matrix_fix_20260705_192345`
- Selected matrix parse: `/tmp/sycl_source_line_iga_matrix_fix_20260705_192345/build-matrix/debug_full/source-line-feasibility.parse`
- Previous blocked MXFP4 root: `/tmp/sycl_mxfp4_iga_source_line_fix4_20260705_210052`
- Previous blocked MXFP4 build root: `/tmp/sycl_mxfp4_iga_source_line_build_fix4_20260705_210052`
- Passing source-line-only MXFP4 root: `/tmp/sycl_mxfp4_source_line_device_tu_20260706_112515`
- Passing source-line-only MXFP4 build root: `/tmp/sycl_mxfp4_source_line_device_tu_build_20260706_112515`

Results:

- Probe matrix selected row: `source_line.status asm-line-static-cost`
- Probe static mapping: `asm_source.mapped_instruction_count 18`, `asm_source.source_line_rows 2`, top line `main.cpp:148`
- MXFP4 task-to-ZEBin selection: fixed; selected task maps to `.text._ZTS39mxfp4_pair_glu_xmx_tiled_dpas_m2_kernelILi8ELi3ELb0ELb0EE`
- MXFP4 IGA extraction: fixed; `extract.status ok`
- MXFP4 IGA PC rows: present; `1583` CSV lines in `iga-pc-instructions.csv`
- MXFP4 previous blocker: selected compute ZEBin had no `.debug_line` / usable DWARF source rows (`source_line.debug_line_present 0`, `source_line.status fail` on structured replay and on the first narrow `sycl-mxfp4-source-line-probe` attempt)
- MXFP4 source-line-only device TU result: `source_line.debug_line_present 1`, `source_line.dwarf_source_rows 1194`, `source_line.asm_source_line_rows 65`, `source_line.status asm-line-static-cost`

Detailed report: `activation/sycl-iga-pc-source-line-validation.md`.

Existing implementation artifacts are present:

- `scripts/prepare-sycl-iga-disasm-inputs.py` selects a `.text.*` ZEBin section and writes an IGA command manifest.
- `scripts/parse-sycl-iga-pc-disasm.py` parses IGA JSON/text rows with explicit PCs, including real JSON v2 `elems` / `kind: "I"` output.
- `scripts/resolve-sycl-zebin-asm-source-lines.py --iga-instructions-csv --pc-base` maps kernel-matched IGA section-relative PCs through DWARF line ranges.
- Matrix and MXFP4 runners prefer `iga-pc-instructions.csv` and fall back to `ocloc`/DWARF without fabricating static-cost evidence.
- The MXFP4 runner now selects the ZEBin matching the VTune-selected compute task instead of trusting the first archived binary; if VTune task export is empty, it uses an explicit fallback section substring for the default MXFP4 registry kernel.
- `sycl-mxfp4-source-line-probe` now links a minimal source-line-only device TU for the default MXFP4 pair-GLU path, avoiding the full `mmvq.cpp` device image for this diagnostic target.

Result: `asm-line-static-cost` is a real validated static evidence level for both source-line probe artifacts and the MXFP4 pair-GLU source-line diagnostic target.

## Runtime sampled line attribution

Status: **schema implemented; runtime PC sampling not available**.

Lead PC-sampling capability probe roots:

```text
/tmp/sycl_intel_pc_sampling_capability_20260705_175719
/tmp/sycl_pc_sampling_followup_capability_20260706_115208
```

Latest result:

```text
pc_sampling.status metrics_only
pc_sampling.blocker no_public_pc_sample_api_confirmed
pc_sampling.blocker vtune_source_rows_empty
pc_sampling.blocker gtpin_not_found
pc_sampling.blocker pti_files_found_but_no_pc_sample_producer
pc_sampling.blocker level_zero_metrics_are_not_pc_samples
pc_sampling.blocker no_level_zero_ip_metric_type_exposed
```

Latest metric-property evidence:

```text
metric_property_group_count 58
metric_property_metric_count 2893
ip_metric_count 0
```

Detailed report: `activation/sycl-intel-pc-sampling-capability.md`.

Existing implementation artifacts are present:

- `scripts/resolve-sycl-pc-samples-to-source-lines.py` maps real `kernel,pc,sample_count,sample_kind` rows to DWARF source lines.
- `scripts/check-sycl-vtune-source-lines.py`, `scripts/parse-sycl-source-attribution.py`, and `scripts/merge-sycl-staged-ledger.py` keep `sampled-line-cost` / `sampled_line_cost` distinct from VTune exact source rows.
- `scripts/sycl-intel-pc-sampling-capability.sh` reports `available`, `metrics_only`, or `unavailable`, enumerates Level Zero metric properties including `ZET_METRIC_TYPE_IP`, and never synthesizes `pc-samples.csv`.

No true sampled runtime source-line attribution is available from public/local tooling on this host yet. The installed PTI/Level Zero stack exposes metric counters, but no instruction-pointer metric source (`ip_metric_count 0`) and no producer for a positive-count `kernel,pc,sample_count` CSV.

## Exact source-line attribution rule

`source_attribution.status exact_source_line` remains reserved for `source_line.status pass` from VTune GPU source rows. Runtime sampled PC rows may produce `sampled-line-cost`; IGA rows may produce `asm-line-static-cost`; DWARF coverage may produce `dwarf-line-table-only`. None of those non-`pass` paths should be promoted to `exact_source_line`.

## Evidence levels

| Evidence | Runtime sampled? | Source-line ranked? | Accepted status |
|---|---:|---:|---|
| VTune GPU source rows | yes | yes | `source_line.status pass` |
| PC sample CSV mapped through DWARF | yes | yes | `source_line.status sampled-line-cost` |
| IGA PC static instruction rows mapped through DWARF | no | yes | `source_line.status asm-line-static-cost` |
| DWARF line-table coverage only | no | no cost ranking | `source_line.status dwarf-line-table-only` |

## Next optimization use

For TG optimization today, use:

1. `pass` / `exact_source_line` if lead VTune source rows are available.
2. `sampled-line-cost` only after a real positive-count `pc-samples.csv` is produced and mapped.
3. `asm-line-static-cost` for artifacts whose IGA PC rows and DWARF line rows both validate; this is now true for the standalone probe and for the source-line-only MXFP4 pair-GLU diagnostic target.
4. `dwarf-line-table-only` as coverage/fallback when cost-ranked source rows are not available.

Current MXFP4 state: task-to-ZEBin selection, IGA PC extraction, and static PC-to-source-row resolution work for the source-line-only pair-GLU diagnostic target. The normal benchmark/backend path remains unchanged and should not be relabeled as sampled exact source-line attribution.

## 2026-07-06 MXFP4 probe-only TU guard update

A follow-up narrowing change for `sycl-mxfp4-source-line-probe` was implemented
and validated:

- target-only compile definition: `SYCL_MXFP4_SOURCE_LINE_PROBE_ONLY=1`;
- `mxfp4_inline_dot.cpp` now excludes unrelated standalone/selected/layer/down
  benchmark families for the source-line probe target while preserving the
  normal `sycl-kernel-bench` target;
- source tests now enforce that the guard applies only to the probe target.

Validation:

```text
python3 -m pytest tests/test-sycl-mxfp4-source-line-probe-source.py tests/test-sycl-vtune-source-line-feasibility-script.py -q
13 passed

./scripts/sycl-build.sh sycl-mxfp4-source-line-probe
Build succeeded
```

Real VTune/GPU feasibility rerun:

```text
OUT_ROOT=/tmp/sycl_mxfp4_source_line_probe_split_20260706_100913
BUILD_DIR=/tmp/sycl_mxfp4_source_line_probe_split_build_20260706_100913
```

Result remains fail-closed for MXFP4 cost-ranked lines:

```text
source_line.debug_line_present 0
source_line.dwarf_status error
source_line.dwarf_error no source rows found
source_line.blocker missing_debug_line
source_line.status fail
```

The executable ran successfully and the intended `.text._ZTS39mxfp4_pair_glu_xmx_tiled_dpas_m2_kernelILi8ELi3ELb0ELb0EE`
section was selected, but the selected ZEBin still lacks source rows. Therefore,
`asm-line-static-cost` remained validated only for the standalone probe matrix at
this stage; the follow-up source-line-only device TU below resolves the MXFP4
source-row blocker.

## 2026-07-06 MXFP4 source-line-only device TU update

Follow-up implementation for `llama.cpp-f0z3` added
`tools/sycl-kernel-bench/kernels/reference/mxfp4_pair_glu_source_line_device.cpp`
and links it only into `sycl-mxfp4-source-line-probe`. The normal backend and
normal `sycl-kernel-bench` target are unchanged.

Lead gates:

```text
python3 -m pytest tests/test-sycl-mxfp4-source-line-probe-source.py tests/test-sycl-vtune-source-line-feasibility-script.py -q
15 passed

git diff --check
passed

bash -n scripts/sycl-vtune-source-line-feasibility.sh
passed

./scripts/sycl-build.sh sycl-mxfp4-source-line-probe
Build succeeded
```

Real VTune/GPU feasibility rerun:

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
source_line.asm_top_source_line /tmp/sycl_mxfp4_source_line_device_tu_build_20260706_112515/ggml/src/ggml-sycl/mmvq.cpp:7114
source_line.asm_top_static_score 112
source_line.asm_top_instruction_count 112
source_line.source_attribution_mode asm-line-static
source_line.blocker none
source_line.status asm-line-static-cost
```

IGA extraction selected the intended pair-GLU section:

```text
.text._ZTS39mxfp4_pair_glu_xmx_tiled_dpas_m2_kernelILi8ELi3ELb0ELb0EE
```

Conclusion: MXFP4 pair-GLU static line attribution is now validated as
`asm-line-static-cost`. VTune sampled exact source-line rows are still not
available (`source_line.vtune_no_gpu_side_trace 1`), so `pass` /
`exact_source_line` remains reserved.
