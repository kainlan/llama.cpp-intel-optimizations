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

## Runtime line attribution

Status: **GTPin BBL runtime-count attribution is available; true sampled PC attribution remains unavailable**.

Lead PC-sampling capability probe roots:

```text
/tmp/sycl_intel_pc_sampling_capability_20260705_175719
/tmp/sycl_pc_sampling_followup_capability_20260706_115208
/tmp/sycl_pc_sampling_gtpin_discovery_20260706_144209
```

Latest result:

```text
pc_sampling.status metrics_only
pc_sampling.blocker no_public_pc_sample_api_confirmed
pc_sampling.blocker vtune_source_rows_empty
pc_sampling.blocker pti_files_found_but_no_pc_sample_producer
pc_sampling.blocker level_zero_metrics_are_not_pc_samples
pc_sampling.blocker no_level_zero_ip_metric_type_exposed
pc_sampling.blocker gtpin_memorytrace_available_but_not_sampled_pc
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

No true sampled runtime source-line attribution is available from Level Zero/PTI/VTune metric tooling on this host yet. The installed PTI/Level Zero stack exposes metric counters, but no instruction-pointer metric source (`ip_metric_count 0`). A follow-up targeted VTune `stall-sampling` probe initially failed to configure on both B50 (`target-gpu=0:7:0.0`) and B580 (`target-gpu=0:3:0.0`), while a targeted B50 `mem-latency` rerun still produced empty `gpu-source-line`/`computing-task` reports and zero sampled/source execution rows in `dicer.db`.

Latest negative sampled-PC / source-row probe artifacts:

```text
/tmp/sycl_cpne_stall_sampling_probe_20260706_163026
/tmp/sycl_cpne_stall_sampling_b50_20260706_163051
/tmp/sycl_cpne_stall_sampling_b580_20260706_163117
/tmp/sycl_cpne_memlatency_b50_targeted_20260706_163227
/tmp/sycl_cpne_stall_sampling_explicit_20260706_214423
/tmp/sycl_cpne_stall_sampling_gpu_mode_20260706_214458
/tmp/sycl_cpne_gpu_profiling_setup_audit_20260706_214609
/tmp/sycl_cpne_stall_sampling_sysctl0_20260706_214707
/tmp/sycl_cpne_stall_sampling_sysctl0_long_20260706_214936
```

The setup audit found `render`/`video` groups and Metrics Discovery installed,
but the documented profiling sysctls were restrictive:
`dev.xe.observation_paranoid=1` and `dev.i915.perf_stream_paranoid=1`. With only
those sysctls temporarily set to `0`, VTune `stall-sampling` collection starts
and loads `stallreasons_*.lzm` on both B50 and B580. However, the resulting
VTune databases still contain no compute-task, source, or assembly correlation:

```text
dd_sample > 0
gpu_sampling_data_agg_data > 0
dd_compute_task 0
dd_compute_sample 0
dd_gpu_execution_stats 0
dd_source_file 0
dd_assembly 0
```

The raw `dd_sample.ip` values are therefore not promoted to `sampled-line-cost`:
there is no validated kernel/base/source correlation and no generated
`kernel,pc,sample_count` evidence file.

A stricter follow-up also tried VTune's debugfs/GPU metrics helper, explicit
tracefs permission opening, and root execution:

```text
/tmp/sycl_cpne_try_both_strict_vtune_20260706_221909
/tmp/sycl_cpne_try_both_tracefs_vtune_20260706_222013
/tmp/sycl_cpne_try_both_root_vtune_20260706_222058
```

Even with sysctls at `0`, tracefs readable/writable, and root execution, VTune
still emitted empty `hotspots` source/task CSV exports and zero compute/source
correlation rows (`dd_compute_task`, `dd_compute_sample`, `dd_source_file`, and
`dd_assembly` all remained `0`). Strict `pass` / `exact_source_line` is therefore
still unavailable on this local B50/B580 stack.

Final current-stack capability sweep:

```text
/tmp/sycl_cpne_final_sampled_pc_sweep_20260706_223803
pc_sampling.status metrics_only
metric_property_group_count 58
metric_property_metric_count 2893
ip_metric_count 0
```

This confirms that the installed Level Zero/PTI/VTune/GTPin stack still exposes
capability evidence and runtime instrumentation paths, but no public positive
sampled-PC producer. The strict path remains blocked until a newer
hardware/driver/VTune stack exposes source-correlated sampled IP rows or a real
`kernel,pc,sample_count` producer.

A separate runtime-count path is now validated through VTune's embedded GTPin
`memorytrace.so` profiler. This produces positive runtime BBL execution counts
for the profiled trace scope and maps them to instruction PCs via the generated
GTPin ASM plus MAAT `app.report.json`, then to source through the existing DWARF
line resolver. It is deliberately labeled **`gtpin_bbl_runtime_cost`**, not
`sampled-line-cost`, because the producer is BBL execution counting rather than
PC sampling.

GTPin validation artifact root:

```text
/tmp/sycl_gtpin_bbl_runtime_line_20260706_143724
```

Runtime BBL/PC extraction:

```text
gtpin_bbl_pc.status ok
gtpin_bbl_pc.trace_send_descriptor_u32_count 20
gtpin_bbl_pc.profiled_thread_count 1
gtpin_bbl_pc.bbl_header_count 53
gtpin_bbl_pc.dynamic_bbl_records 231
gtpin_bbl_pc.dynamic_instruction_count 63703
gtpin_bbl_pc.pc_rows 1136
gtpin_bbl_pc.send_pc_rows 63
gtpin_bbl_pc.id_shift 1
gtpin_bbl_pc.id_shift_send_matches 63
gtpin_bbl_pc.sample_kind gtpin-bbl-instruction-exec-count
```

Runtime source-line resolution with `--require-source-path mmvq.cpp`:

```text
gtpin_bbl_source.status ok
gtpin_bbl_source.mapped_sample_count 2910
gtpin_bbl_source.unmapped_sample_count 60793
gtpin_bbl_source.source_line_rows 31
gtpin_bbl_source.top_source_line /tmp/sycl_mxfp4_source_line_device_tu_build_20260706_112515/ggml/src/ggml-sycl/mmvq.cpp:7233
gtpin_bbl_source.top_sample_count 720
gtpin_bbl_source.top_sample_kind gtpin-bbl-instruction-exec-count
```

Checker validation:

```text
/tmp/sycl_gtpin_bbl_runtime_line_20260706_143724/gtpin-bbl-source-line-feasibility.parse
source_line.gtpin_bbl_source_line_rows 31
source_line.gtpin_bbl_top_source_line /tmp/sycl_mxfp4_source_line_device_tu_build_20260706_112515/ggml/src/ggml-sycl/mmvq.cpp:7233
source_line.gtpin_bbl_top_sample_count 720
source_line.source_attribution_mode gtpin-bbl-line
source_line.blocker none
source_line.status gtpin-bbl-runtime-cost
```

A second runtime-count path was validated through Intel PTI `instcount` from the
upstream `intel/pti-gpu` source tree. This tool instruments GPU instructions and
reports dynamic instruction execution counts plus SIMD active lanes; it is not a
statistical PC sampler.

PTI `instcount` artifacts:

```text
/tmp/sycl_cpne_pti_instcount_20260706_215023
/tmp/sycl_cpne_pti_instcount_run2_20260706_215133
/tmp/sycl_cpne_try_both_pti_pipeline_20260706_222720
```

Pair-GLU kernel validation on both B50 and B580:

```text
runs 4
results_num 1562
positive_offsets 1562
total_instruction_count 200448000
```

The PTI path is now first-class in repo scripts:

```text
scripts/extract-sycl-pti-instcount-pc-counts.py
scripts/resolve-sycl-pc-samples-to-source-lines.py --attribution-mode pti-instcount-line --attribution-status pti_instcount_runtime_cost
scripts/check-sycl-vtune-source-lines.py --pti-instcount-source-lines-csv --allow-pti-instcount-runtime-cost
scripts/parse-sycl-source-attribution.py
scripts/merge-sycl-staged-ledger.py
```

The extractor converts PTI JSON to the shared `kernel,pc,sample_count,sample_kind`
schema with:

```text
sample_kind=pti-instcount-instruction-exec-count
```

and the checker/parser/merger preserve explicit non-sampled status labels:

```text
source_line.status pti-instcount-runtime-cost
source_attribution.status pti_instcount_runtime_cost
```

Real first-class pipeline result:

```text
pti_instcount.status ok
pti_instcount.pc_rows 1562
pti_instcount.total_instruction_count 200448000
pti_instcount_source.status ok
pti_instcount_source.mapped_sample_count 11571840
pti_instcount_source.unmapped_sample_count 188876160
pti_instcount_source.source_line_rows 65
pti_instcount_source.top_source_line /tmp/sycl_mxfp4_source_line_device_tu_build_20260706_112515/ggml/src/ggml-sycl/mmvq.cpp:7233
pti_instcount_source.top_sample_count 2073600
source_line.status pti-instcount-runtime-cost
source_attribution.status pti_instcount_runtime_cost
```

This is accepted only as `pti_instcount_runtime_cost`, separate from
`gtpin_bbl_runtime_cost` and separate from `sampled-line-cost`.

Implementation artifacts:

- `scripts/extract-sycl-gtpin-bbl-pc-counts.py` parses GTPin `memorytrace_compressed.bin` BBL traces, including the VTune 2025.10 20-u32 send descriptor ABI, and emits positive `kernel,pc,sample_count,sample_kind` rows with `sample_kind=gtpin-bbl-instruction-exec-count`.
- `scripts/extract-sycl-pti-instcount-pc-counts.py` converts PTI `instcount` JSON into the same resolver schema with `sample_kind=pti-instcount-instruction-exec-count`.
- `scripts/resolve-sycl-pc-samples-to-source-lines.py` accepts explicit attribution labels/prefixes so runtime-count paths can emit `gtpin-bbl-line` / `gtpin_bbl_runtime_cost` or `pti-instcount-line` / `pti_instcount_runtime_cost` instead of falsely labeling counts as `sampled_line_cost`.
- `scripts/check-sycl-vtune-source-lines.py` accepts `--gtpin-bbl-source-lines-csv --allow-gtpin-bbl-runtime-cost` and `--pti-instcount-source-lines-csv --allow-pti-instcount-runtime-cost` and emits the corresponding runtime-count `source_line.status`.
- `scripts/sycl-gptoss-staged-attribution-profile.sh` and `scripts/sycl-gptoss-full-attribution-profile.sh` accept `--gtpin-bbl-source-lines-csv`, `--pti-instcount-source-lines-csv`, and `--source-kernel` so full/staged E2E ledgers can consume checked runtime-count source-line CSVs as non-exact attribution layers.

## Exact source-line attribution rule

`source_attribution.status exact_source_line` remains reserved for `source_line.status pass` from VTune GPU source rows. Runtime sampled PC rows may produce `sampled-line-cost`; GTPin memorytrace BBL counts may produce `gtpin_bbl_runtime_cost`; IGA rows may produce `asm-line-static-cost`; DWARF coverage may produce `dwarf-line-table-only`. None of those non-`pass` paths should be promoted to `exact_source_line`.

## Evidence levels

| Evidence | Runtime sampled? | Source-line ranked? | Accepted status |
|---|---:|---:|---|
| VTune GPU source rows | yes | yes | `source_line.status pass` |
| PC sample CSV mapped through DWARF | yes | yes | `source_line.status sampled-line-cost` |
| GTPin memorytrace BBL execution counts mapped through DWARF | runtime counted, not sampled | yes | `source_line.status gtpin-bbl-runtime-cost` / row status `gtpin_bbl_runtime_cost` |
| PTI `instcount` instruction execution counts mapped through DWARF | runtime counted, not sampled | yes | row status `pti_instcount_runtime_cost` |
| IGA PC static instruction rows mapped through DWARF | no | yes | `source_line.status asm-line-static-cost` |
| DWARF line-table coverage only | no | no cost ranking | `source_line.status dwarf-line-table-only` |

## Next optimization use

For TG optimization today, use:

1. `pass` / `exact_source_line` if lead VTune source rows are available.
2. `sampled-line-cost` only after a real sampled positive-count `pc-samples.csv` is produced and mapped.
3. `gtpin_bbl_runtime_cost` for profiled trace-scope runtime BBL execution counts; this is now available for the MXFP4 pair-GLU source-line diagnostic target.
4. `pti_instcount_runtime_cost` for PTI/GTPin instrumented dynamic instruction execution counts; this is validated for the MXFP4 pair-GLU source-line diagnostic target but remains separate from sampled PC evidence.
5. `asm-line-static-cost` for artifacts whose IGA PC rows and DWARF line rows both validate; this is now true for the standalone probe and for the source-line-only MXFP4 pair-GLU diagnostic target.
6. `dwarf-line-table-only` as coverage/fallback when cost-ranked source rows are not available.

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
