# SYCL MXFP4 Down Direct-Final Proof Report

## Active Evidence

- Valid baseline artifact: `/tmp/sycl_named_kernel_profile_gptoss_r1_20260702_215807/findings.md`.
- Valid baseline throughput: `PP512 1211.90 tok/s`, `TG128 36.97 tok/s`.
- Active down hotspot: `mxfp4.down.q8_soa`, `625.864 ms`, `2324` events, mean `269.304 us`, about `4.89 ms/token`.

## Existing Direct-Final Knobs

- `GGML_SYCL_MOE_DOWN_SUM_XMX_DIRECT_FINAL`
- `GGML_SYCL_MOE_DOWN_SUM_DPAS_DIRECT_FINAL`
- `GGML_SYCL_MOE_DOWN_SUM_DPAS_DIRECT_FINAL_I8`
- `GGML_SYCL_MOE_DOWN_SUM_DPAS_DIRECT_FINAL_DPAS`
- `GGML_SYCL_MOE_DOWN_SUM_DPAS_DIRECT_FINAL_RANK_PARALLEL_ATOMIC`
- `GGML_SYCL_MOE_DOWN_SUM_DPAS_DIRECT_FINAL_SCRATCH_REDUCE`
- `GGML_SYCL_MOE_DOWN_SUM_DPAS_DIRECT_FINAL_SAME_EXPERT_GROUPED`

## Fail-Closed Policy

Direct-final down is not enabled by `GGML_SYCL_MOE_DOWN_SUM_Q8_SOA_TG_VARIANT` or `GGML_SYCL_MOE_DOWN_CACHED_Q8_SOA_TG_VARIANT`. Row-group variants are low-risk experiments on the active direct q8-SOA route. Cached q8-SOA variants are diagnostic non-promotion rows because they require direct-sum disabled. Direct-final remains a separate proof track.

## Recommendation

Keep direct-final default-off in this plan. A future direct-final implementation requires a new approved plan with a model-free reference/validation path, profiler labels distinct from `mxfp4.down.q8_soa`, and lead-owned B50 GPT-OSS FA-on correctness/performance gates.
