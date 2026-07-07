# Intel GPU PC-sampling capability probe

Date: 2026-07-05
Updated: 2026-07-06
Validation owner: lead

## Summary

Lead execute validation was run with oneAPI sourced via:

```bash
set +u
source /opt/intel/oneapi/setvars.sh --force
set -u
```

Command artifact roots:

```text
/tmp/sycl_intel_pc_sampling_capability_20260705_175719
/tmp/sycl_pc_sampling_followup_capability_20260706_115208
/tmp/sycl_pc_sampling_gtpin_discovery_20260706_144209
```

Result: **`pc_sampling.status metrics_only`**.

No true sampled runtime source-line attribution is available from public/local tooling on this host yet. The 2026-07-06 follow-ups enhanced the probe to enumerate actual Level Zero metric properties, confirmed that the installed metric groups expose no `ZET_METRIC_TYPE_IP` metrics, and detected VTune's embedded GTPin `memorytrace.so` as runtime BBL-count capability rather than sampled-PC capability.

## Parse output

```text
pc_sampling.status metrics_only
pc_sampling.device_selector level_zero:1
pc_sampling.pc_samples_csv /tmp/sycl_intel_pc_sampling_capability_20260705_175719/pc-samples.csv
pc_sampling.note Level Zero metric groups are counters only, not PC samples.
pc_sampling.blocker no_public_pc_sample_api_confirmed
pc_sampling.blocker vtune_source_rows_empty
pc_sampling.blocker gtpin_not_found
pc_sampling.blocker pti_files_found_but_no_pc_sample_producer
```

## Artifact files

```text
gtpin-path.txt 0 bytes
level-zero-metric-groups.txt 306 bytes
pc-sampling-capability.parse 443 bytes
pti-files.txt 10892 bytes
vtune-gpu-hotspots-help.txt 4913 bytes
vtune-gpu-source-line-help.txt 1133 bytes
```

## Interpretation

The probe reports `pc_sampling.status available` only if a real `pc-samples.csv` with `kernel,pc,sample_count` and positive sample counts is present. This run did not produce that file. Level Zero metric groups, PTI files, VTune help output, and GTPin discovery are capability evidence only; they are not PC samples.

Therefore:

- `sampled-line-cost` is implemented as a schema and resolver path, but no runtime sampled PC evidence is available yet.
- `source_attribution.status exact_source_line` remains reserved for VTune `source_line.status pass`.
- Do not treat Level Zero OA metrics, PTI kernel timestamps, or VTune/GTPin failure rows as sampled source-line attribution.

## 2026-07-06 follow-up: Level Zero metric-property/IP enumeration

Follow-up tracker: `llama.cpp-cpne`

The capability script now enables metrics for the probe process with
`ZET_ENABLE_METRICS=${ZET_ENABLE_METRICS:-1}` and enumerates actual Level Zero
metric group and metric properties via:

```text
zetMetricGroupGet
zetMetricGroupGetProperties
zetMetricGet
zetMetricGetProperties
```

It specifically counts metrics whose type is `ZET_METRIC_TYPE_IP` /
`0x7ffffffe`. These are the only Level Zero metric properties that could be a
candidate for an instruction-pointer sample producer; ordinary OA/time/event
metric groups remain counter evidence only.

Lead-only execute validation was run with oneAPI sourced via the required
`set +u` / `setvars.sh` / `set -u` sequence.

Command artifact root:

```text
/tmp/sycl_pc_sampling_followup_capability_20260706_115208
```

Parse output:

```text
pc_sampling.status metrics_only
pc_sampling.device_selector level_zero:1
pc_sampling.pc_samples_csv /tmp/sycl_pc_sampling_followup_capability_20260706_115208/pc-samples.csv
pc_sampling.note Level Zero metric groups/properties are capability evidence only, not PC samples.
pc_sampling.blocker no_public_pc_sample_api_confirmed
pc_sampling.blocker vtune_source_rows_empty
pc_sampling.blocker gtpin_not_found
pc_sampling.blocker pti_files_found_but_no_pc_sample_producer
pc_sampling.blocker level_zero_metrics_are_not_pc_samples
pc_sampling.blocker no_level_zero_ip_metric_type_exposed
```

Metric-property summary:

```text
driver_count 2
metric_property_group_count 58
metric_property_metric_count 2893
ip_metric_count 0
metric_group_lines 58
metric_property_lines 2893
```

Local PTI headers define `PTI_METRIC_TYPE_IP`, and Level Zero headers define
`ZET_METRIC_TYPE_IP`, but the installed B580/B50 metric groups enumerate zero IP
metrics. VTune help exposes `gpu-sampling-interval` and source-analysis knobs,
but the `gpu-source-line` report is still unavailable from the CLI in this
stack, and no `pc-samples.csv` producer is installed.

Conclusion from the metric-property probe: Level Zero/PTI/VTune counters are not
sampled PC producers. `sampled-line-cost` remains reserved for a real sampled PC
producer.

## 2026-07-06 follow-up: GTPin memorytrace BBL runtime counts

Follow-up tracker: `llama.cpp-cpne`

The VTune install does include the embedded GTPin framework at:

```text
/opt/intel/oneapi/vtune/2025.10/bin64/gma/GTPin/Profilers/Bin/gtpin
```

and the bundled example profiler:

```text
/opt/intel/oneapi/vtune/2025.10/bin64/gma/GTPin/Profilers/Examples/intel64/memorytrace.so
```

The framework requires:

```text
LD_LIBRARY_PATH=/opt/intel/oneapi/vtune/2025.10/lib64/gtpin:$LD_LIBRARY_PATH
```

Using MAAT/GTPin `memorytrace.so` on `sycl-mxfp4-source-line-probe` produced a
real positive runtime BBL trace for the pair-GLU kernel. MAAT's built-in decoder
mis-parsed this VTune 2025.10 trace because the send descriptor has one extra
`u32` versus MAAT's old parser, and MAAT's own source mapper failed on the
embedded ELF debug info. A repo extractor now parses the raw trace directly:

```text
scripts/extract-sycl-gtpin-bbl-pc-counts.py
```

Updated capability probe artifact root (embedded GTPin discovery):

```text
/tmp/sycl_pc_sampling_gtpin_discovery_20260706_144209
```

It reports:

```text
pc_sampling.status metrics_only
pc_sampling.blocker gtpin_memorytrace_available_but_not_sampled_pc
metric_property_group_count 58
metric_property_metric_count 2893
ip_metric_count 0
```

GTPin BBL/source validation artifact root:

```text
/tmp/sycl_gtpin_bbl_runtime_line_20260706_143724
```

Runtime BBL/PC extraction summary:

```text
gtpin_bbl_pc.status ok
gtpin_bbl_pc.trace_send_descriptor_u32_count 20
gtpin_bbl_pc.register_size_bits 512
gtpin_bbl_pc.timestamp_included 0
gtpin_bbl_pc.tile_count 1
gtpin_bbl_pc.profiled_thread_count 1
gtpin_bbl_pc.bbl_header_count 53
gtpin_bbl_pc.dynamic_bbl_records 231
gtpin_bbl_pc.dynamic_instruction_count 63703
gtpin_bbl_pc.pc_rows 1136
gtpin_bbl_pc.send_pc_rows 63
gtpin_bbl_pc.id_shift 1
gtpin_bbl_pc.id_shift_send_matches 63
gtpin_bbl_pc.missing_bbl_record_count 0
gtpin_bbl_pc.missing_instruction_count 0
gtpin_bbl_pc.sample_kind gtpin-bbl-instruction-exec-count
```

Source-line resolution used the existing DWARF resolver with explicit truthful
labels:

```text
--attribution-mode gtpin-bbl-line
--attribution-status gtpin_bbl_runtime_cost
--summary-prefix gtpin_bbl_source
--require-source-path mmvq.cpp
```

Result:

```text
gtpin_bbl_source.status ok
gtpin_bbl_source.mapped_sample_count 2910
gtpin_bbl_source.unmapped_sample_count 60793
gtpin_bbl_source.source_line_rows 31
gtpin_bbl_source.top_source_line /tmp/sycl_mxfp4_source_line_device_tu_build_20260706_112515/ggml/src/ggml-sycl/mmvq.cpp:7233
gtpin_bbl_source.top_sample_count 720
gtpin_bbl_source.top_sample_kind gtpin-bbl-instruction-exec-count
```

Checker-level status:

```text
source_line.gtpin_bbl_source_line_rows 31
source_line.gtpin_bbl_top_source_line /tmp/sycl_mxfp4_source_line_device_tu_build_20260706_112515/ggml/src/ggml-sycl/mmvq.cpp:7233
source_line.gtpin_bbl_top_sample_count 720
source_line.source_attribution_mode gtpin-bbl-line
source_line.blocker none
source_line.status gtpin-bbl-runtime-cost
```

This is runtime evidence, but it is **not** VTune `pass` and it is **not**
renamed to `sampled-line-cost`: it is a GTPin memorytrace BBL execution-count
profile for the profiled trace scope. The accepted status for this evidence is
`gtpin_bbl_runtime_cost`; `sampled-line-cost` remains reserved for sampled PC
rows from a true sampling producer.

## 2026-07-06 follow-up: targeted VTune stall-sampling probe

Follow-up tracker: `llama.cpp-cpne`

After GTPin BBL runtime-count attribution was integrated, lead ran narrow
VTune source-analysis probes on the standalone
`sycl-mxfp4-source-line-probe` binary to check whether VTune's hardware
`stall-sampling` mode can provide a true sampled-PC producer on this host.
All runs used the required oneAPI sourcing sequence and did not access models or
`/Storage`.

Artifacts:

```text
/tmp/sycl_cpne_stall_sampling_probe_20260706_163026
/tmp/sycl_cpne_stall_sampling_b50_20260706_163051
/tmp/sycl_cpne_stall_sampling_b580_20260706_163117
/tmp/sycl_cpne_memlatency_b50_targeted_20260706_163227
```

Results:

```text
# default VTune target (iGPU)
vtune: Error: Unable to run GPU hardware sampling for the GPU adapter: '0:0:2.0'.

# explicit B50 target-gpu=0:7:0.0
vtune: Error: Cannot configure the collection for the requested devices.
vtune: Collection failed.
vtune: Internal Error

# explicit B580 target-gpu=0:3:0.0
vtune: Error: Cannot configure the collection for the requested devices.
vtune: Collection failed.
vtune: Internal Error
```

A targeted B50 `source-analysis=mem-latency` rerun did execute the probe, but it
still produced no sampled GPU source rows or sampled-PC-like database tables:

```text
gpu-source-line.csv size 0
computing-tasks.csv size 0
VTune summary: "no GPU-side trace data was collected"
dd_gpu_execution_stats 0
dd_compute_task_type 0
dd_compute_task 0
dd_compute_sample 0
dd_sample 0
```

Installed VTune grouping definitions confirm that the CLI `gpu-source-line`
grouping is backed by non-empty `gsim_stall_data`, `gpu_sampling_data`, or
`gpu_gtpin_data`; none of those tables appear in the tested result databases.
Therefore local VTune stall-sampling and source-analysis modes still do not
provide a `kernel,pc,sample_count` producer. `sampled-line-cost` remains
unvalidated.

## 2026-07-06 follow-up: explicit stall-sampling environment and Linux setup audit

Follow-up tracker: `llama.cpp-cpne`

Code Scout web research found Intel's documented CLI recipe for hardware-assisted
stall sampling:

```text
AMPLXE_EXPERIMENTAL=gpu-stall-sampling
vtune -collect gpu-hotspots -knob profiling-mode=source-analysis -knob source-analysis=stall-sampling -- <app>
```

VTune 2025.10 warns that `profiling-mode` is deprecated, so lead also tried the
current spelling:

```text
-knob gpu-profiling-mode=source-analysis
```

All probe commands used the standalone `sycl-mxfp4-source-line-probe`, did not
access models or `/Storage`, and used the required oneAPI sourcing sequence.

Artifacts without changing system profiling permissions:

```text
/tmp/sycl_cpne_stall_sampling_explicit_20260706_214423
/tmp/sycl_cpne_stall_sampling_gpu_mode_20260706_214458
```

Results with the explicit `AMPLXE_EXPERIMENTAL=gpu-stall-sampling` environment
were still configuration failures on both discrete GPUs while the kernel
profiling sysctls remained restrictive:

```text
# B50 target-gpu=0:7:0.0
vtune: Error: Cannot configure the collection for the requested devices.
vtune: Collection failed.
vtune: Internal Error

# B580 target-gpu=0:3:0.0
vtune: Error: Cannot configure the collection for the requested devices.
vtune: Collection failed.
vtune: Internal Error
```

Linux setup audit artifact:

```text
/tmp/sycl_cpne_gpu_profiling_setup_audit_20260706_214609
```

Audit summary:

```text
group.render present
group.video present
intel-metrics-discovery installed
prepare-gpu-hardware-metrics.sh present
dev.xe.observation_paranoid=1
dev.i915.perf_stream_paranoid=1
/sys/kernel/debug mounted but not readable/writable by the user
/sys/kernel/tracing not readable/writable by the user
CONFIG_DRM_I915_LOW_LEVEL_TRACEPOINTS is not set
```

The VTune helper documents that it sets both `dev.i915.perf_stream_paranoid` and
`dev.xe.observation_paranoid` to zero. Lead therefore temporarily set only those
two sysctls to `0`, reran the same standalone probes, and restored both values to
`1` afterward.

Artifacts with temporary sysctl opening:

```text
/tmp/sycl_cpne_stall_sampling_sysctl0_20260706_214707
/tmp/sycl_cpne_stall_sampling_sysctl0_long_20260706_214936
```

This changed the failure mode: VTune collection now starts, runs, finalizes, and
loads `stallreasons_*.lzm` for both B50 and B580. However, VTune still reports
that the GPU Compute/Media Hotspots viewpoint is unavailable and that no GPU-side
trace data was collected. The result databases contain aggregate stall/sample
rows but no compute-task, source, or assembly correlation:

```text
# short run, B50
dd_sample 7160
gpu_sampling_data_agg_band 716
gpu_sampling_data_agg_data 7160
dd_compute_task 0
dd_compute_task_type 0
dd_compute_sample 0
dd_gpu_execution_stats 0
dd_source_file 0
dd_assembly 0

# long run, B50
dd_sample 10380
dd_compute_task 0
dd_compute_task_type 0
dd_compute_sample 0
dd_gpu_execution_stats 0
dd_source_file 0
dd_assembly 0

# long run, B580
dd_sample 10230
dd_compute_task 0
dd_compute_task_type 0
dd_compute_sample 0
dd_gpu_execution_stats 0
dd_source_file 0
dd_assembly 0
```

The `dd_sample.ip` values are instruction-aligned and the event taxonomy includes
`%GPUActive`, `%GPUPipeStall`, `%GPUSend`, `%GPUSbid`, `%GPUSync`,
`%GPUInstructionFetch`, and other stall reason names. They are still not accepted
as `sampled-line-cost`: VTune does not provide kernel names, module/base mapping,
source rows, or assembly rows for these samples in the tested results, and many
candidate base offsets can map them into the dense IGA PC range. A truthful
`sampled-line-cost` producer still requires validated `kernel,pc,sample_count`
rows with a non-ambiguous kernel/base correlation.

Updated conclusion: the earlier hard configuration failure was caused at least in
part by Linux profiling permission knobs. Opening those knobs enables aggregate
VTune stall-sampling collection, but on this B50/B580 stack it still does not
produce usable sampled source-line attribution. `source_line.status pass` and
`source_attribution.status exact_source_line` remain unavailable, and
`sampled-line-cost` remains unvalidated.

## 2026-07-06 follow-up: PTI `instcount` dynamic instruction counts

Follow-up tracker: `llama.cpp-cpne`

Intel's PTI `instcount` tool is not installed by the local oneAPI PTI package,
but the upstream `intel/pti-gpu` source builds cleanly in `/tmp` and downloads
its matching GTPin package during CMake configure.

Artifacts:

```text
/tmp/sycl_cpne_pti_instcount_20260706_215023
/tmp/sycl_cpne_pti_instcount_run2_20260706_215133
```

Build result:

```text
cmake.exit 0
build.exit 0
built: build-instcount/instcount
built: build-instcount/libinstcount_tool.so
```

Narrow validation on `sycl-mxfp4-source-line-probe` succeeded on both B50 and
B580 with `--disable-simd --json-output`. The tool writes its JSON profile to
stderr and the application output to stdout. It instruments kernel instructions;
this is runtime instruction execution counting, not statistical PC sampling.

Pair-GLU kernel summary on both B50 and B580:

```text
kernel _ZTS39mxfp4_pair_glu_xmx_tiled_dpas_m2_kernelILi8ELi3ELb0ELb0EE
runs 4
results_num 1562
positive_offsets 1562
total_instruction_count 200448000
```

A CSV adapter converted the PTI offsets to `kernel,pc,sample_count,sample_kind`
rows with:

```text
sample_kind=pti-instcount-instruction-exec-count
```

and mapped them through the existing DWARF resolver using explicit non-sampled
labels:

```text
--attribution-mode pti-instcount-line
--attribution-status pti_instcount_runtime_cost
--summary-prefix pti_instcount_source
--require-source-path mmvq.cpp
```

Mapping result on both B50 and B580:

```text
pti_instcount_source.status ok
pti_instcount_source.mapped_sample_count 11571840
pti_instcount_source.unmapped_sample_count 188876160
pti_instcount_source.source_line_rows 65
pti_instcount_source.top_source_line /tmp/sycl_mxfp4_source_line_device_tu_build_20260706_112515/ggml/src/ggml-sycl/mmvq.cpp:7233
pti_instcount_source.top_sample_count 2073600
pti_instcount_source.top_sample_kind pti-instcount-instruction-exec-count
```

This provides a second validated runtime-count attribution path alongside GTPin
`memorytrace.so`, but it must stay labeled separately as
`pti_instcount_runtime_cost`. It is not VTune exact source-line attribution, and
it is not `sampled-line-cost` because the producer instruments/counts
instructions rather than sampling PCs.

## 2026-07-06 follow-up: try both strict VTune and practical PTI paths

Follow-up tracker: `llama.cpp-cpne`

Lead tried both paths requested after the initial web/setup conclusion.

### Strict VTune exact/source-row path

Additional strict VTune artifacts:

```text
/tmp/sycl_cpne_try_both_strict_vtune_20260706_221909
/tmp/sycl_cpne_try_both_tracefs_vtune_20260706_222013
/tmp/sycl_cpne_try_both_root_vtune_20260706_222058
```

The first run invoked VTune's debugfs/GPU metrics helpers and exported `hotspots`
reports grouped by `computing-task`, `source-line`, and
`computing-task,source-line`. `prepare-debugfs.sh` could not fully configure
this kernel's split tracefs/debugfs layout:

```text
chgrp: changing group of '/sys/kernel/debug/tracing': Operation not permitted
Failed to change group ownership
```

A second run temporarily opened `/sys/kernel/tracing` directly, verified that
`available_events` was readable and `tracing_on` was writable, and restored
tracefs permissions afterward. A third run repeated the probe as root with
sysctls and tracefs open, matching Intel's recommendation that GPU profiling is
typically run with administrator privileges.

All three runs still produced no VTune compute-task/source/assembly correlation:

```text
hotspots-computing-task.csv size 0
hotspots-source-line.csv size 0
hotspots-computing-source-line.csv size 0
dd_compute_task 0
dd_compute_sample 0
dd_gpu_execution_stats 0
dd_source_file 0
dd_assembly 0
dd_sample > 0
```

Conclusion for strict path: permissions explain the original hard
"Cannot configure" failure, but even with sysctls, tracefs, and root execution,
VTune on this B50 stack only provides aggregate stall-sampling rows. It still
does not provide exact VTune source rows or validated sampled `kernel,pc` rows.
Therefore `source_line.status pass`, `source_attribution.status
exact_source_line`, and `sampled-line-cost` remain unavailable.

### Practical PTI runtime-count path

The PTI `instcount` path is now first-class and reproducible in repo scripts.
New extractor:

```text
scripts/extract-sycl-pti-instcount-pc-counts.py
```

It converts PTI `instcount` JSON to the shared resolver schema:

```text
kernel,pc,sample_count,sample_kind
sample_kind=pti-instcount-instruction-exec-count
```

The checker/parser/merger now accept the explicit non-sampled runtime-count
status:

```text
source_line.status pti-instcount-runtime-cost
source_attribution.status pti_instcount_runtime_cost
```

Real validation artifact:

```text
/tmp/sycl_cpne_try_both_pti_pipeline_20260706_222720
```

Pipeline result from the real B50 `instcount` artifact:

```text
pti_instcount.status ok
pti_instcount.pc_rows 1562
pti_instcount.total_instruction_count 200448000
pti_instcount.top_pc 0x5a90
pti_instcount.top_sample_count 552960

pti_instcount_source.status ok
pti_instcount_source.mapped_sample_count 11571840
pti_instcount_source.unmapped_sample_count 188876160
pti_instcount_source.source_line_rows 65
pti_instcount_source.top_source_line /tmp/sycl_mxfp4_source_line_device_tu_build_20260706_112515/ggml/src/ggml-sycl/mmvq.cpp:7233
pti_instcount_source.top_sample_count 2073600

source_line.status pti-instcount-runtime-cost
source_attribution.status pti_instcount_runtime_cost
source_attribution.pti_instcount_top_source_line /tmp/sycl_mxfp4_source_line_device_tu_build_20260706_112515/ggml/src/ggml-sycl/mmvq.cpp:7233
source_attribution.pti_instcount_top_sample_count 2073600
```

This gives a usable line-number cost attribution route for targeted kernels and
for future staged ledgers, but it remains explicitly **runtime instruction-count
attribution**, not exact VTune sampled source-line attribution and not sampled PC
line cost.

## 2026-07-06 final current-stack sampled-PC sweep and E2E runtime integration

Final no-model current-stack capability sweep:

```text
/tmp/sycl_cpne_final_sampled_pc_sweep_20260706_223803
pc_sampling.status metrics_only
pc_sampling.blocker no_public_pc_sample_api_confirmed
pc_sampling.blocker vtune_source_rows_empty
pc_sampling.blocker gtpin_memorytrace_available_but_not_sampled_pc
pc_sampling.blocker pti_files_found_but_no_pc_sample_producer
pc_sampling.blocker level_zero_metrics_are_not_pc_samples
pc_sampling.blocker no_level_zero_ip_metric_type_exposed
metric_property_group_count 58
metric_property_metric_count 2893
ip_metric_count 0
```

This confirms the strict sampled-PC path is exhausted on the current B50/B580
software stack unless a newer VTune/driver/runtime or supported hardware exposes
source-correlated sampled IP rows. `sampled-line-cost`, `pass`, and
`exact_source_line` remain unclaimed.

The practical path is now wired into the E2E scripts. Both staged and full
profiling runners accept checker-compatible runtime source-line CSVs:

```text
scripts/sycl-gptoss-staged-attribution-profile.sh \
  --gtpin-bbl-source-lines-csv <gtpin-source-lines.csv> \
  --pti-instcount-source-lines-csv <pti-source-lines.csv> \
  --source-kernel mxfp4_pair_glu_xmx_tiled

scripts/sycl-gptoss-full-attribution-profile.sh \
  --gtpin-bbl-source-lines-csv <gtpin-source-lines.csv> \
  --pti-instcount-source-lines-csv <pti-source-lines.csv> \
  --source-kernel mxfp4_pair_glu_xmx_tiled
```

These flags call `check-sycl-vtune-source-lines.py` with explicit allow flags and
produce only the non-exact statuses:

```text
source_line.status gtpin-bbl-runtime-cost
source_line.status pti-instcount-runtime-cost
source_attribution.status gtpin_bbl_runtime_cost
source_attribution.status pti_instcount_runtime_cost
```

They do not promote runtime-count rows to `sampled-line-cost` or
`exact_source_line`.
