# Intel GPU PC-sampling capability probe

Date: 2026-07-05
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

No true sampled runtime source-line attribution is available from public/local tooling on this host yet.

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
