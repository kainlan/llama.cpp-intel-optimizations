# Intel GPU PC-sampling capability probe

Task 7 added a lead-only capability probe at
`scripts/sycl-intel-pc-sampling-capability.sh`. This worker did not run the
execute path, GPU probes, VTune, GTPin, Level Zero enumeration, `/Storage`, or
model commands. Only source-only gates were run.

## How to run under lead ownership

```bash
OUT=/tmp/sycl_intel_pc_sampling_capability_$(date +%Y%m%d_%H%M%S)
set +u
source /opt/intel/oneapi/setvars.sh --force
set -u
scripts/sycl-intel-pc-sampling-capability.sh \
  --execute \
  --i-understand-this-probes-intel-gpu-pc-sampling \
  --out-root "$OUT" \
  --device-selector level_zero:1
cat "$OUT/pc-sampling-capability.parse"
```

## Interpretation

The probe reports `pc_sampling.status available` only if a real
`pc-samples.csv` with `kernel,pc,sample_count` and positive sample counts is
present. Level Zero metric groups, PTI files, VTune help output, and GTPin help
output are treated as capability evidence only; they are not PC samples.

No true sampled runtime source-line attribution is available from public/local
tooling on this host yet.

Current worker status: implementation and dry-run validation only. Lead execute
validation is still required to determine whether this host reports
`available`, `metrics_only`, or `unavailable`.
