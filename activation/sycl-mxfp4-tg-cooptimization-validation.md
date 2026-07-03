# SYCL MXFP4 TG Co-Optimization Validation

## Build Under Test

- Branch: `feature/sycl-mxfp4-tg-runtime`
- Implementation HEAD before validation-record commit: `776223351`
- Validation record commit: this commit and later updates in this file

## Automated Tests

Pending.

## B50 GPT-OSS Correctness Gate

Pending lead-only run.

## FA-on Bench/Profile Matrix

| Variant | Env delta | PP512 tok/s | TG128 tok/s | Top gate/up kernel time | Top down kernel time | Artifact dir | Decision |
|---------|-----------|-------------|-------------|--------------------------|----------------------|--------------|----------|
| baseline | none | | | | | | |
| row2 | `GGML_SYCL_MOE_DOWN_SUM_Q8_SOA_TG_VARIANT=row2` | | | | | | |
| row4 | `GGML_SYCL_MOE_DOWN_SUM_Q8_SOA_TG_VARIANT=row4` | | | | | | |
| tg1-index | `GGML_SYCL_MOE_GATEUP_M2_TG1_INDEX=1` | | | | | | |
| best-combined | record exact env | | | | | | |

## Promotion Decision

Pending. Default remains unchanged unless the best-combined row passes correctness, keeps `PP512 >= 1150 tok/s`, and reaches `TG128 >= 45 tok/s`.
