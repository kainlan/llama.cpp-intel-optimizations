# SYCL Decode Hottest-Kernel Source Attribution — 2026-07-04

## Scope

This note summarizes the Task 8 full-attribution pass for GPT-OSS 20B MXFP4 decode on the B50 path.

Primary profile artifact root:

```text
/tmp/sycl_decode_profile_full_20260704_114044
```

Exact GPU source-line feasibility artifact root:

```text
/tmp/sycl_vtune_source_line_20260704_114136
```

## Cost ranking

`cost-ranking.parse` reports named kernel-profile total time of `1567.156 ms`. The top kernels are:

| Rank | Kernel | Count | Total ms | Share of named kernel time |
| ---: | --- | ---: | ---: | ---: |
| 1 | `mxfp4.gateup.xmx_tiled_dpas_m2` | 3096 | 706.354 | 45.1% |
| 2 | `mxfp4.down.q8_soa` | 2324 | 618.590 | 39.5% |
| 3 | `fattn.compute.xmx_v2` | 48 | 97.730 | 6.2% |

The first two MXFP4 MoE kernels account for `1324.944 ms`, about `84.5%` of named kernel-profile time.

## Source-region attribution

The profiler labels map to these source regions in `ggml/src/ggml-sycl/mmvq.cpp`:

| Kernel label | Source region | Label line | Notes |
| --- | --- | ---: | --- |
| `mxfp4.gateup.xmx_tiled_dpas_m2` | `mxfp4_pair_glu_xmx_tiled_dpas_m2_sycl`, `ggml/src/ggml-sycl/mmvq.cpp:9730-9955` | `9767` | ESIMD/XMX gate+up DPAS M2 kernel over packed Q8 activations and MXFP4 weights. |
| `mxfp4.down.q8_soa` | `mxfp4_down_sum_q8_soa_sycl`, `ggml/src/ggml-sycl/mmvq.cpp:19308-19392` | `19338` | Q8 SOA down-projection sum/reduction kernel over selected experts and route weights. |
| `fattn.compute.xmx_v2` | named flash-attention XMX v2 path | not re-expanded here | Third-ranked kernel and much smaller than the two MoE kernels. |

This is source-region attribution, not exact device source-line attribution. The exact GPU source-line feasibility spike failed closed, so no narrower line-level attribution is claimed.

## Wall ledger

`wall-ledger.parse` reports:

| Ledger row | Value |
| --- | ---: |
| `ledger.wall_ms_x1000` | `4671780` |
| `ledger.timeline_gpu_event_ms_x1000` | `143491` |
| `ledger.kernel_profile_total_ms_x1000` | `1567156` |
| `ledger.timeline_kernel_delta_ms_x1000` | `1423665` |
| `ledger.timeline_kernel_ratio_pct_x1000` | `9156` |
| `ledger.coverage_status` | `coverage_mismatch` |
| `ledger.gap_class.host_overlap_ms_x1000` | `1178705` |
| `ledger.gap_class.queue_serialization_ms_x1000` | `0` |
| `ledger.gap_class.runtime_idle_ms_x1000` | `935601` |
| `ledger.unknown_wall_residual_ms_x1000` | `2413983` |

Because `ledger.coverage_status` is `coverage_mismatch`, `sycl-kernels.csv/json` remains the authoritative named-GPU-cost source. The residual wall bucket is explicit unknown time, not proof of GPU idle.

## Exact GPU source-line feasibility result

`source-line-feasibility.parse` reports:

```text
source_line.debug_line_present 0
source_line.non_unknown_rows 0
source_line.status fail
```

Supporting evidence:

- `zebin-debug-sections.txt` has `.symtab`, `.ze_info`, and `.strtab`, but no `.debug_line`.
- `vtune-gpu-source-line.csv` reports `Source Line = [Unknown]` and no non-unknown rows for the target MXFP4 kernel.
- `profiling-debug-build.log` includes `VCDebugInfo: only modules with one CU are supported at the moment, the debug information for Module will be dropped out.`

Result: the B-spike failed closed. Full GPT-OSS exact GPU source-line profiling is still blocked until a microbench target produces non-unknown source-line rows and `.debug_line` in dumped device binaries.

## Bound-type verdict

Fail-closed verdict: `unknown_with_static_xmx_region`.

Rationale:

- The top gate/up region is an ESIMD/XMX DPAS kernel by source inspection (`mxfp4_pair_glu_xmx_tiled_dpas_m2_sycl`).
- The down region is a Q8 SOA reduction/dot-product kernel by source inspection (`mxfp4_down_sum_q8_soa_sycl`).
- The current Task 8 artifacts did not include successful ocloc/asm or VTune per-source metrics for these kernels.
- Therefore no definitive memory-bound vs compute-bound claim is made from this run. A future bound-type claim should be generated from `parse-sycl-vtune-kernel-asm.py --classify-bound` over a real assembly/instruction artifact or from VTune kernel metrics that resolve the target kernel.

## Throughput context

The full profile run reported:

| Test | tok/s |
| --- | ---: |
| PP512 | `1189.49 ± 0.00` |
| TG128 | `31.84 ± 0.00` |

PP remains near the preservation target but this particular run is below the prior `1214.85 tok/s` run and below the original ~1200 guardrail by ~0.9%. TG remains below the `>=45 tok/s` objective.
