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

Command artifact root:

```text
/tmp/sycl_intel_pc_sampling_capability_20260705_175719
```

Result: **`pc_sampling.status metrics_only`**.

No true sampled runtime source-line attribution is available from public/local tooling on this host yet. The 2026-07-06 follow-up enhanced the probe to enumerate actual Level Zero metric properties and confirmed that the installed metric groups expose no `ZET_METRIC_TYPE_IP` metrics.

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

Conclusion: the current host has counter/metric capability, but no public/local
runtime PC-sample producer. `sampled-line-cost` remains schema-only until a real
positive-count `kernel,pc,sample_count` CSV is produced and mapped through
`scripts/resolve-sycl-pc-samples-to-source-lines.py`.
