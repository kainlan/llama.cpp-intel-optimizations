# Current SYCL GPT-OSS full profiling pass

Date: 2026-07-06
Tracker issue: `llama.cpp-rflo`
Owner: lead

## Scope

This pass uses the evidence layers that currently work on this host:

- GPT-OSS 20B MXFP4 `llama-bench` PP512/TG128 on B50 (`ONEAPI_DEVICE_SELECTOR=level_zero:1`)
- SYCL named-kernel profiler
- SYCL timeline profiler
- PTI `instcount` runtime-count source-line attribution for the top MXFP4 gate/up diagnostic path

It does **not** claim VTune exact source rows or sampled PC line cost. Those remain unavailable on the current B50/B580 stack.

## Artifacts

```text
/tmp/sycl_current_full_profile_20260706_230949
/tmp/sycl_current_full_profile_20260706_230949/runtime-line
```

The benchmark binary reported:

```text
build: b59cdbf8c (11530)
```

Core SYCL backend source has not changed since that build in the current script/doc-only commits; only source-line diagnostic/tooling files changed.

## Throughput

```text
PP512: 1217.90 tok/s
TG128:   34.83 tok/s
```

PP remains in the desired ~1200 tok/s band. TG remains below the `>=45 tok/s` target.

## Kernel hotspots

Named-kernel total: `1546.070 ms`.

| Rank | Kernel | Total ms | Share | Count |
|---:|---|---:|---:|---:|
| 1 | `mxfp4.gateup.xmx_tiled_dpas_m2` | 697.506 | 45.11% | 3096 |
| 2 | `mxfp4.down.q8_soa` | 610.131 | 39.46% | 2324 |
| 3 | `fattn.compute.xmx_v2` | 96.390 | 6.23% | 48 |
| 4 | `sycl.binbcast.mul` | 33.039 | 2.14% | 7108 |
| 5 | `sycl.rope` | 24.447 | 1.58% | 6288 |
| 6 | `sycl.set_rows.generic` | 23.994 | 1.55% | 6288 |
| 7 | `sycl.binbcast.add` | 20.490 | 1.33% | 2698 |
| 8 | `sycl.memcpy.mem_ops` | 9.161 | 0.59% | 3533 |
| 9 | `sycl.softmax.forward` | 8.791 | 0.57% | 3144 |
| 10 | `mxfp4.pack_q8.single_col` | 6.703 | 0.43% | 3096 |
| 11 | `mxfp4.quantize.activation_q8_soa` | 5.823 | 0.38% | 3098 |

Combined MXFP4 gate/up + down kernels account for `84.57%` of named kernel time.

## Runtime line attribution for top hotspot

PTI `instcount` diagnostic attribution was attached to the current top E2E kernel name as `pti_instcount_runtime_cost`:

```text
source_attribution.status pti_instcount_runtime_cost
source_attribution.kernel mxfp4.gateup.xmx_tiled_dpas_m2
source_attribution.source_line_status pti-instcount-runtime-cost
source_attribution.pti_instcount_top_source_line /tmp/sycl_mxfp4_source_line_device_tu_build_20260706_112515/ggml/src/ggml-sycl/mmvq.cpp:7233
source_attribution.pti_instcount_top_sample_count 2073600
source_attribution.file ggml/src/ggml-sycl/mmvq.cpp
source_attribution.line_start 9730
source_attribution.line_end 9955
source_attribution.label_line 9767
```

Top PTI runtime-count source lines:

| Rank | Source line | Count | Meaning |
|---:|---|---:|---|
| 1 | `ggml/src/ggml-sycl/mmvq.cpp:7233` | 2073600 | `block_load` of MXFP4 scale bytes in `mxfp4_xmx_tiled_load_a_vec_from_group` |
| 2 | `ggml/src/ggml-sycl/mmvq.cpp:9816` | 1296000 | gate/up group pointer setup for the second m-tile |
| 3 | `ggml/src/ggml-sycl/mmvq.cpp:9814` | 1071360 | gate group pointer setup for the first m-tile |
| 4 | `ggml/src/ggml-sycl/mmvq.cpp:7232` | 1036800 | load-address setup for packed MXFP4 group |
| 5 | `ggml/src/ggml-sycl/mmvq.cpp:7235` | 1036800 | E8M0 scale conversion path |
| 6 | `ggml/src/ggml-sycl/mmvq.cpp:7236` | 1036800 | unrolled MXFP4 nibble unpack loop |
| 7 | `ggml/src/ggml-sycl/mmvq.cpp:9815` | 518400 | up group pointer setup for first m-tile |
| 8 | `ggml/src/ggml-sycl/mmvq.cpp:7114` | 348480 | GLU fallback return expression |
| 9 | `ggml/src/ggml-sycl/mmvq.cpp:9841` | 259200 | B-vector prefetch |
| 10 | `ggml/src/ggml-sycl/mmvq.cpp:9866` | 259200 | DPAS result extraction for up path |

These are runtime instruction-count rows, not sampled PC rows.

## Source regions for the two main hotspots

- `mxfp4.gateup.xmx_tiled_dpas_m2`: `ggml/src/ggml-sycl/mmvq.cpp:9730-9955`, label line `9767`.
- `mxfp4.down.q8_soa`: `ggml/src/ggml-sycl/mmvq.cpp:19308-19392`, label line `19338`.

The second hotspot is the serial Q8 SOA down-projection reduction loop:

```text
mxfp4_down_sum_q8_soa_sycl
  per token x row
  for each routed expert id
    mxfp4_soa_q8_1_block_dot(...)
    reduce_over_group(...)
    lane 0 applies route weight/bias and stores output
```

## Timeline notes

The timeline artifact captured host and queue gaps, but its GPU-event total does not reconcile with the named-kernel profile total in this run:

```text
ledger.timeline_gpu_event_ms_x1000 140608
ledger.kernel_profile_total_ms_x1000 1546070
ledger.coverage_status coverage_mismatch
```

Therefore the named-kernel profiler is the authoritative hotspot ranking for this pass. Timeline gap data is still useful qualitatively; the largest transitions include:

- `sycl.softmax.forward -> sycl.binbcast.mul`: `405.462 ms` total gap
- `sycl.binbcast.event -> sycl.binbcast.add`: `159.994 ms`
- `fattn.compute.xmx_v2 -> sycl.binbcast.add`: `130.612 ms`
- `sycl.set_rows.generic -> fattn.compute.xmx_v2`: `116.399 ms`

## Interpretation

Current TG is dominated by two kernels:

1. Gate/up MXFP4 XMX tiled DPAS m2 (`45.1%`)
2. Down Q8 SOA serial reduction (`39.5%`)

The immediate optimization surface is still the MoE decode path, not attention or generic elementwise kernels. The top line-count evidence points at MXFP4 scale/packed loads, pointer setup, nibble unpack/scale conversion, and DPAS result extraction inside the gate/up kernel. The down kernel remains nearly as large and is structurally a per-token/per-row serial loop over routed experts with subgroup reduction.
