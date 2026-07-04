# SYCL Decode Profiling Report — GPT-OSS 20B MXFP4 on B50

Date: 2026-07-04
Branch: `feature/sycl-mxfp4-tg-runtime`
Profile artifact root: `/tmp/sycl_decode_profile_real_20260704_072110`
Binary-reported build: `5832bfc33 (11343)`

## Command

The run used the lead-only profile script with the required execute acknowledgement:

```bash
set +u
source /opt/intel/oneapi/setvars.sh --force
set -u
./scripts/sycl-gptoss-decode-timeline-profile.sh \
  --execute \
  --i-understand-this-runs-gpu-models \
  --out-root /tmp/sycl_decode_profile_real_20260704_072110 \
  --device-selector level_zero:1
```

Effective benchmark command from `command.txt`:

```text
env ONEAPI_DEVICE_SELECTOR=level_zero:1 GGML_SYCL_TIMELINE=timeline+events GGML_SYCL_TIMELINE_OUTPUT=/tmp/sycl_decode_profile_real_20260704_072110/sycl-timeline.json GGML_SYCL_TIMELINE_TOKEN_START=1 GGML_SYCL_KERNEL_PROFILE=1 GGML_SYCL_KERNEL_PROFILE_OUTPUT=/tmp/sycl_decode_profile_real_20260704_072110/sycl-kernels GGML_SYCL_KERNEL_PROFILE_FORMAT=both GGML_SYCL_KERNEL_PROFILE_RAW=1 GGML_SYCL_KERNEL_PROFILE_TOP_N=80 GGML_SYCL_KERNEL_PROFILE_FLUSH=window GGML_SYCL_MOE_PHASE_MATERIALIZE=1 GGML_SYCL_MOE_PHASE_BULK_XMX=1 GGML_SYCL_MOE_DOWN_SUM_DIRECT=1 ./build/bin/llama-bench -m /Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf -ngl 99 -fa 1 -p 512 -n 128 -r 1
```

## Throughput

| Test | tok/s |
| --- | ---: |
| PP512 | `1214.85 ± 0.00` |
| TG128 | `34.89 ± 0.00` |

Interpretation:
- PP512 remains above the ~1200 tok/s preservation target.
- TG128 is still below the >=45 tok/s objective.

## Artifacts

| Artifact | Path |
| --- | --- |
| benchmark stdout | `/tmp/sycl_decode_profile_real_20260704_072110/bench.stdout` |
| benchmark stderr | `/tmp/sycl_decode_profile_real_20260704_072110/bench.stderr` |
| timeline trace | `/tmp/sycl_decode_profile_real_20260704_072110/sycl-timeline.json` |
| timeline summary | `/tmp/sycl_decode_profile_real_20260704_072110/timeline.parse` |
| gap summary | `/tmp/sycl_decode_profile_real_20260704_072110/timeline.gaps.parse` |
| kernel CSV | `/tmp/sycl_decode_profile_real_20260704_072110/sycl-kernels.csv` |
| kernel JSON | `/tmp/sycl_decode_profile_real_20260704_072110/sycl-kernels.json` |
| kernel summary | `/tmp/sycl_decode_profile_real_20260704_072110/kernels.parse` |

## Named kernel cost ranking

Named kernel profile total: `1562.017 ms`.

| Rank | Kernel/category | Count | Total ms | Share |
| ---: | --- | ---: | ---: | ---: |
| 1 | `mxfp4.gateup.xmx_tiled_dpas_m2` | 3096 | 705.044 | 45.1% |
| 2 | `mxfp4.down.q8_soa` | 2324 | 615.968 | 39.4% |
| 3 | `fattn.compute.xmx_v2` | 48 | 97.255 | 6.2% |
| 4 | `sycl.binbcast.mul` | 7108 | 33.088 | 2.1% |
| 5 | `sycl.rope` | 6288 | 24.988 | 1.6% |
| 6 | `sycl.set_rows.generic` | 6288 | 24.242 | 1.6% |
| 7 | `sycl.binbcast.add` | 2698 | 20.852 | 1.3% |
| 8 | `sycl.softmax.forward` | 3144 | 9.354 | 0.6% |
| 9 | `sycl.memcpy.mem_ops` | 3533 | 9.225 | 0.6% |
| 10 | `mxfp4.pack_q8.single_col` | 3096 | 6.872 | 0.4% |
| 11 | `mxfp4.quantize.activation_q8_soa` | 3098 | 5.479 | 0.4% |
| 12 | `sycl.memcpy.mem_fill` | 3371 | 4.498 | 0.3% |
| 13 | `sycl.binbcast.event` | 7108 | 3.928 | 0.3% |
| 14 | `mxfp4.soa.batched` | 4 | 1.088 | 0.1% |
| 15 | `sycl.get_rows.marker` | 393 | 0.137 | <0.1% |

Category totals:

| Category | Total ms | Share of named kernel time |
| --- | ---: | ---: |
| `mmvq` | 1334.450 | 85.4% |
| `fattn` | 97.255 | 6.2% |
| `binbcast` | 57.867 | 3.7% |
| `memory` | 38.101 | 2.4% |
| `rope` | 24.988 | 1.6% |
| `softmax` | 9.354 | 0.6% |

Key point: the dominant measured device cost is still MXFP4 MoE gate/up and down, together ~`1321.012 ms` / `84.6%` of named kernel time.

## Timeline/gap summary

Timeline parser summary:

| Metric | Value |
| --- | ---: |
| Timeline wall | `4200.566 ms` |
| Timeline `sycl.event` total | `141.821 ms` |
| Timeline event coverage | `3.376%` |
| Timeline unattributed | `4058.745 ms` |

Important caveat: the named kernel CSV/JSON profile reports `1562.017 ms` of device event time, while the timeline trace reports only `141.821 ms` of `sycl.event` spans. Therefore, the current timeline JSON still does not carry all named kernel events into the Chrome trace. Use `sycl-kernels.csv/json` as the primary ranked device-cost source; use `timeline.gaps.parse` as a trace-gap diagnostic, not as proof of total GPU idle.

Gap classification from timeline trace:

| Queue | Gap count | Total gap ms | Host-overlap ms | Runtime-idle ms |
| --- | ---: | ---: | ---: | ---: |
| compute | 961 | 888.630 | 184.773 | 703.857 |
| copy | 382 | 1027.008 | 832.008 | 195.000 |

No `queue_serialization` gaps were classified in this run.

## Largest gap transitions

Top compute-queue transitions:

| Transition | Count | Total ms | Max ms |
| --- | ---: | ---: | ---: |
| `sycl.softmax.forward -> sycl.binbcast.mul` | 46 | 405.907 | 58.485 |
| `sycl.binbcast.event -> sycl.binbcast.add` | 138 | 148.635 | 22.387 |
| `fattn.compute.xmx_v2 -> sycl.binbcast.add` | 48 | 132.545 | 3.856 |
| `sycl.set_rows.generic -> fattn.compute.xmx_v2` | 48 | 74.300 | 67.705 |
| `sycl.rope -> sycl.binbcast.add` | 96 | 48.116 | 0.704 |
| `mxfp4.soa.batched -> mxfp4.down.q8_soa` | 2 | 39.539 | 39.512 |
| `sycl.get_rows.marker -> sycl.binbcast.add` | 2 | 16.718 | 14.007 |

Top copy-queue transitions:

| Transition | Count | Total ms | Max ms |
| --- | ---: | ---: | ---: |
| `sycl.memcpy.mem_ops -> sycl.memcpy.mem_ops` | 187 | 667.340 | 96.965 |
| `sycl.memcpy.mem_fill -> sycl.memcpy.mem_fill` | 96 | 169.107 | 13.383 |
| `sycl.memcpy.mem_ops -> sycl.memcpy.mem_fill` | 49 | 150.480 | 12.430 |
| `sycl.memcpy.mem_fill -> sycl.memcpy.mem_ops` | 50 | 40.087 | 2.806 |

## Largest host-overlap attributions

The top host-overlap rows show that many timeline gaps overlap host-side graph/op work:

| Overlap attribution | Host ms |
| --- | ---: |
| `sycl.binbcast.event -> sycl.binbcast.add` / `MUL_MAT` | 386.877 |
| `sycl.memcpy.mem_ops -> sycl.memcpy.mem_ops` / `ADD_ID` | 157.191 |
| `sycl.memcpy.mem_ops -> sycl.memcpy.mem_ops` / `MUL_MAT_ID` | 120.619 |
| `sycl.memcpy.mem_ops -> sycl.get_rows.marker` / `GET_ROWS` | 76.517 |
| `sycl.memcpy.mem_ops -> sycl.binbcast.mul` / `ADD_ID` | 57.924 |
| `sycl.memcpy.mem_ops -> sycl.binbcast.mul` / `MUL_MAT_ID` | 48.345 |
| `mxfp4.soa.batched -> sycl.memcpy.mem_ops` / `MUL_MAT_ID` | 39.813 |
| `fattn.compute.xmx_v2 -> sycl.binbcast.add` / `MUL_MAT` | 19.080 |
| `sycl.memcpy.mem_fill -> sycl.binbcast.add` / `MUL_MAT` | 14.940 |
| `sycl.binbcast.event -> sycl.memcpy.mem_fill` / `MUL_MAT` | 12.421 |

## What changed relative to earlier incomplete traces

The newly added named coverage is visible in the kernel profile:
- `sycl.binbcast.add`: `20.852 ms`
- `sycl.binbcast.mul`: `33.088 ms`
- `sycl.rope`: `24.988 ms`
- `sycl.set_rows.generic`: `24.242 ms`
- `sycl.softmax.forward`: `9.354 ms`
- `sycl.memcpy.mem_ops`: `9.225 ms`
- `sycl.memcpy.mem_fill`: `4.498 ms`
- `sycl.get_rows.marker`: `0.137 ms`

This confirms the broadened named profiler coverage is working. These newly classified ops are measurable but not the dominant device time.

## Conclusions

1. PP performance is preserved: PP512 is `1214.85 tok/s`.
2. TG performance is still below target: TG128 is `34.89 tok/s` vs target >=45.
3. Named kernel cost remains dominated by MXFP4 MoE kernels:
   - gate/up DPAS M2: `705.044 ms`
   - down Q8 SOA: `615.968 ms`
4. The new profiling coverage successfully classifies binbcast, rope, set_rows, softmax, memcopy, memfill, and get_rows marker work.
5. Pack and activation quantization remain small (`6.872 ms` and `5.479 ms`), so they are not primary optimization targets.
6. Timeline trace coverage is still incomplete relative to the kernel CSV/JSON (`141.821 ms` timeline event total vs `1562.017 ms` kernel-profile total). This should be treated as a remaining profiler-completeness issue before using timeline `unattributed` as definitive idle time.

## Recommended next steps

1. Use `sycl-kernels.csv/json` as the authoritative cost table for optimization triage.
2. Prioritize MXFP4 MoE gate/up and down paths; together they account for ~85% of named kernel time.
3. Investigate why the timeline JSON omits most named kernel events even though the kernel profile CSV/JSON contains them. A follow-up should make timeline event coverage agree with kernel profile totals or explicitly document/report the split.
4. If continuing runtime optimization, avoid spending effort on pack/activation quantization unless a later run shows a new regression.

---

## Task 8 full-attribution rerun — 2026-07-04 11:40 UTC

A later run after the full-attribution parser/script tasks used the same lead-only GPT-OSS profile harness and added `cost-ranking.parse` plus `wall-ledger.parse`.

Artifact root:

```text
/tmp/sycl_decode_profile_full_20260704_114044
```

Binary-reported build:

```text
ad978813f (11373)
```

### Throughput

| Test | tok/s |
| --- | ---: |
| PP512 | `1189.49 ± 0.00` |
| TG128 | `31.84 ± 0.00` |

Interpretation:

- TG remains below the `>=45 tok/s` objective.
- PP is near but slightly below the previous `1214.85 tok/s` run; do not treat this as a new PP baseline without repeated same-build measurements.

### Full wall-time ledger

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

Because coverage status is `coverage_mismatch`, the timeline still does not carry enough named event duration to replace the kernel profiler as the ranked GPU-cost source. `sycl-kernels.csv/json` and `cost-ranking.parse` remain authoritative for named device cost. The `unknown_wall_residual` row is intentionally explicit; it must not be relabeled as idle time without additional evidence.

### Cost ranking

`cost-ranking.parse` reports named kernel-profile total time of `1567.156 ms`.

| Rank | Kernel | Count | Total ms | Share |
| ---: | --- | ---: | ---: | ---: |
| 1 | `mxfp4.gateup.xmx_tiled_dpas_m2` | 3096 | 706.354 | 45.1% |
| 2 | `mxfp4.down.q8_soa` | 2324 | 618.590 | 39.5% |
| 3 | `fattn.compute.xmx_v2` | 48 | 97.730 | 6.2% |
| 4 | `sycl.binbcast.mul` | 7108 | 33.818 | 2.2% |
| 5 | `sycl.rope` | 6288 | 24.815 | 1.6% |
| 6 | `sycl.set_rows.generic` | 6288 | 24.389 | 1.6% |
| 7 | `sycl.binbcast.add` | 2698 | 21.148 | 1.3% |
| 8 | `sycl.memcpy.mem_ops` | 3533 | 9.377 | 0.6% |
| 9 | `sycl.softmax.forward` | 3144 | 8.551 | 0.5% |
| 10 | `mxfp4.pack_q8.single_col` | 3096 | 6.871 | 0.4% |

The two MXFP4 MoE kernels account for `1324.944 ms`, about `84.5%` of named kernel-profile time.

### Source-region and bound-type attribution

The top labels map to these source regions:

| Kernel label | Source region | Label line | Attribution status |
| --- | --- | ---: | --- |
| `mxfp4.gateup.xmx_tiled_dpas_m2` | `ggml/src/ggml-sycl/mmvq.cpp:9730-9955` (`mxfp4_pair_glu_xmx_tiled_dpas_m2_sycl`) | `9767` | Region attribution only; exact device line unavailable. |
| `mxfp4.down.q8_soa` | `ggml/src/ggml-sycl/mmvq.cpp:19308-19392` (`mxfp4_down_sum_q8_soa_sycl`) | `19338` | Region attribution only; exact device line unavailable. |

Bound-type verdict for this run is fail-closed: `unknown_with_static_xmx_region`.

The gate/up source region is visibly an ESIMD/XMX DPAS kernel and the down region is a Q8 SOA dot/reduction kernel, but this Task 8 artifact set does not contain successful ocloc/asm or resolved VTune per-source metrics for either top kernel. Therefore the report does not claim a definitive memory-bound vs compute-bound classification from this run.

### Exact GPU source-line feasibility spike

Microbench-only exact source-line artifact root:

```text
/tmp/sycl_vtune_source_line_20260704_114136
```

The script exited with code `2` because the fail-closed checker reported:

```text
source_line.debug_line_present 0
source_line.non_unknown_rows 0
source_line.status fail
```

Supporting evidence:

- `zebin-debug-sections.txt` contains `.symtab`, `.ze_info`, and `.strtab`, but no `.debug_line`.
- `vtune-gpu-source-line.csv` contains only `Source Line = [Unknown]` for the captured row.
- `profiling-debug-build.log` warns: `VCDebugInfo: only modules with one CU are supported at the moment, the debug information for Module will be dropped out.`

Result: the exact GPU source-line B-spike failed closed at the microbench stage. Full GPT-OSS exact-line profiling remains blocked until the microbench proof produces non-unknown GPU source-line rows and dumped `.zebin` files with `.debug_line`.

### Updated conclusion

The new ledger/reporting path is working and exposes the remaining coverage mismatch directly. Optimization triage should still focus on the MXFP4 MoE gate/up and down kernels, but exact device source lines and definitive bound-type classification are not yet available from the current toolchain artifacts.
