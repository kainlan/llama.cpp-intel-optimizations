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
