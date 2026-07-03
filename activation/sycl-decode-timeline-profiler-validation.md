# SYCL decode timeline profiler validation

Date: 2026-07-03
Branch: `feature/sycl-mxfp4-tg-runtime`
Scope: Task `llama.cpp-rwjy` automated integration gates plus lead-only Task `llama.cpp-0aqr` B50 GPT-OSS FA-on decode timeline validation.

## Python source/parser/script gates

Command:

```bash
python3 -m pytest \
  tests/test-sycl-timeline-parser.py \
  tests/test-sycl-timeline-compute-forward-source.py \
  tests/test-sycl-timeline-wait-graph-source.py \
  tests/test-sycl-timeline-cache-source.py \
  tests/test-sycl-decode-timeline-profile-script.py \
  tests/test-sycl-kernel-profiler-source.py \
  tests/test-sycl-kernel-profiler-source-mmvq.py \
  tests/test-sycl-kernel-profiler-source-copy.py \
  -q
```

Result: PASS — `34 passed in 0.31s`.

## SYCL build gates

Command:

```bash
set +u; source /opt/intel/oneapi/setvars.sh --force; set -u
./scripts/sycl-build.sh test-sycl-timeline test-sycl-kernel-profiler llama-bench llama-cli
```

Result: PASS — all requested targets built successfully.

## Selected CTest gates

Command:

```bash
ctest --test-dir build -R "test-sycl-timeline|test-sycl-kernel-profiler|test-sycl-e2e-profile" -V
```

Result: PASS — `100% tests passed, 0 tests failed out of 3`.

Passing tests:
- `test-sycl-e2e-profile`
- `test-sycl-timeline`
- `test-sycl-kernel-profiler`

## Dry-run profile script gate

Command:

```bash
bash scripts/sycl-gptoss-decode-timeline-profile.sh
```

Result: PASS — printed dry-run command and artifact paths only; no model execution.

Dry-run included:
- `GGML_SYCL_TIMELINE=timeline+events`
- `GGML_SYCL_TIMELINE_TOKEN_START=1`
- `GGML_SYCL_KERNEL_PROFILE=1`
- FA-on MoE phase envs
- `./build/bin/llama-bench ... -fa 1 -p 512 -n 128 -r 1`
- `sycl-timeline.json`, `sycl-kernels`, `bench.stdout`, `bench.stderr`, `timeline.parse`, `kernels.parse`

## Backend-free timeline flush integration fix

The first lead-only execution completed `llama-bench` and wrote kernel profile artifacts, but did not write
`sycl-timeline.json`. The run showed that this TG128 validation path did not satisfy the existing
`cached_is_decode`-gated decode-teardown flush, even though named decode kernels ran.

Fix applied before the successful run:

- Added a backend-free final `sycl_timeline_flush("backend-free")` after `ggml_sycl_kernel_profile_flush(true, "backend-free")`.
- Made the backend-free flush fallback-only with `!sycl_timeline_has_flushed_file()` so it cannot clobber an earlier decode-teardown trace.
- Made `sycl_timeline_flush()` a one-shot non-empty file write; duplicate, empty, pathless, failed, or disabled flushes leave the existing file untouched.
- Kept the existing decode-teardown flush unchanged.
- Added source and unit coverage to enforce final-flush ordering and non-clobber behavior, with no waits/barriers/`std::atexit`.

Validation for the fix:

```bash
python3 -m pytest tests/test-sycl-timeline-flush-source.py tests/test-sycl-timeline-compute-forward-source.py -q
# 7 passed in 0.13s

python3 -m pytest tests/test-sycl-timeline-flush-source.py tests/test-sycl-decode-timeline-profile-script.py -q
# 8 passed in 0.08s

set +u; source /opt/intel/oneapi/setvars.sh --force; set -u
./scripts/sycl-build.sh test-sycl-kernel-profiler llama-bench
ctest --test-dir build -R test-sycl-kernel-profiler -V
# 100% tests passed, 0 tests failed out of 1

./scripts/sycl-build.sh test-sycl-timeline
ctest --test-dir build -R test-sycl-timeline -V
# 100% tests passed, 0 tests failed out of 1
```

## Lead-only B50 GPT-OSS FA-on timeline validation

Command:

```bash
set +u; source /opt/intel/oneapi/setvars.sh --force; set -u
out=/tmp/sycl_decode_timeline_lead_20260703_150450
bash scripts/sycl-gptoss-decode-timeline-profile.sh \
  --execute --i-understand-this-runs-gpu-models \
  --out-root "$out"
python3 scripts/parse-sycl-timeline.py "$out/sycl-timeline.json" > "$out/timeline.parse"
python3 scripts/parse-sycl-kernel-profile.py "$out/sycl-kernels.csv" > "$out/kernels.parse"
```

Artifacts:

- `/tmp/sycl_decode_timeline_lead_20260703_150450/sycl-timeline.json` — 43 MiB
- `/tmp/sycl_decode_timeline_lead_20260703_150450/sycl-kernels.csv` — 904 B
- `/tmp/sycl_decode_timeline_lead_20260703_150450/timeline.parse` — 1.4 KiB
- `/tmp/sycl_decode_timeline_lead_20260703_150450/kernels.parse` — 985 B
- `/tmp/sycl_decode_timeline_lead_20260703_150450/bench.stdout`
- `/tmp/sycl_decode_timeline_lead_20260703_150450/bench.stderr`

Throughput result:

- `pp512`: `1207.56 ± 0.00 tok/s`
- `tg128`: `35.11 ± 0.00 tok/s`

The PP512 guardrail remains strong (~1200 tok/s). TG128 remains below the `>=45 tok/s` target.

### Timeline parser findings

`timeline.parse`:

```text
timeline.wall_ms_x1000 3723714
timeline.gpu_event_total_ms_x1000 1321896
timeline.gpu_event_coverage_pct_x1000 35499
timeline.unattributed_ms_x1000 2401818
gap.device0.compute.count 11609
gap.device0.compute.total_ms_x1000 2382951
category.ggml.graph.host_ms_x1000 3326483
category.ggml.op.host_ms_x1000 4355103
category.sycl.submit.host_ms_x1000 50611
category.sycl.wait.host_ms_x1000 4003
```

Top host callsites:

```text
callsite.ggml-sycl.cpp:78549:ggml_backend_sycl_graph_compute_impl 3326.483 ms
callsite.ggml-sycl.cpp:80163:ggml_backend_sycl_graph_compute_impl 2721.761 ms
callsite.ggml-sycl.cpp:71531:ggml_sycl_compute_forward 1633.342 ms
callsite.mmvq.cpp:9761:mxfp4_pair_glu_xmx_tiled_dpas_m2_sycl 25.257 ms
callsite.mmvq.cpp:6769:mxfp4_dpas_pack_q8_single_col_groups_sycl 13.560 ms
callsite.mmvq.cpp:19337:mxfp4_down_sum_q8_soa_sycl 11.794 ms
callsite.ggml-sycl.cpp:78299:ggml_sycl_trace_queue_wait 3.479 ms
```

Trace shape:

- 120,744 complete events total.
- 129 `ggml.graph/graph_compute_impl` spans.
- 50,052 `ggml.op/compute_forward` spans and 50,052 `ggml.op/compute_forward_node` spans.
- 8,514 named `sycl.submit` spans.
- 11,610 named `sycl.event` spans.
- Per-graph wall from trace: first graph `78.002 ms`; remaining 128 graphs total `3248.481 ms`, mean `25.379 ms`, median `24.946 ms`, trace-implied decode rate `39.4 tok/s` before benchmark-level overhead.

Interpretation:

- Named SYCL GPU events cover only ~35.5% of the host timeline envelope.
- The parser reports ~2.383 s of device-queue gaps between named compute events. These gaps are not all proven idle time because the timeline only includes named profiled events; they may include unprofiled kernels and/or host/queue scheduling windows. They are still the dominant missing attribution bucket.
- Host-side submit/wait spans are small (`~50.6 ms` submit, `~4.0 ms` waits), so the current wall gap is not explained by explicit host waits or submit wrapper overhead alone.

### Named kernel profile findings

`kernels.parse`:

```text
profile.kernel_sum_total_ms_x1000 1419906
kernel.mxfp4.gateup.xmx_tiled_dpas_m2.count 3096
kernel.mxfp4.gateup.xmx_tiled_dpas_m2.total_ms_x1000 697911
kernel.mxfp4.down.q8_soa.count 2324
kernel.mxfp4.down.q8_soa.total_ms_x1000 612235
kernel.fattn.compute.xmx_v2.count 48
kernel.fattn.compute.xmx_v2.total_ms_x1000 96241
kernel.mxfp4.pack_q8.single_col.count 3096
kernel.mxfp4.pack_q8.single_col.total_ms_x1000 6853
kernel.mxfp4.quantize.activation_q8_soa.count 3098
kernel.mxfp4.quantize.activation_q8_soa.total_ms_x1000 5577
kernel.mxfp4.soa.batched.count 4
kernel.mxfp4.soa.batched.total_ms_x1000 1089
category.fattn.total_ms_x1000 96241
category.mmvq.total_ms_x1000 1323665
```

Interpretation:

- GPU kernel time is dominated by MXFP4 MoE gate/up (`~698 ms`) and down (`~612 ms`).
- Pack and activation quantization remain small (`~6.9 ms` and `~5.6 ms` respectively), confirming they should not be the next optimization target.
- FAttn is visible (`~96 ms`) but not the primary TG bottleneck.

### Evidence-based next optimization target

To move `tg128` from ~35 tok/s to `>=45 tok/s`, the runtime needs roughly a 0.8–0.9 s wall-time reduction on this run. The strongest evidence points to:

1. **Reduce or explain the ~2.4 s named-event gap/unattributed bucket** by expanding named-event coverage for remaining SYCL work and/or reducing per-token/per-node dispatch gaps with correctness-safe graph replay/fusion.
2. **Target the MoE gate/up + down critical path**, especially any graphlet/fusion route that reduces the 3,096 gate/up launches and 2,324 down launches without regressing PP512 correctness/performance.
3. **Do not spend the next pass on pack/quant micro-optimizations**; their combined measured GPU time is only ~12 ms.

The current instrumentation is sufficient to direct the next performance pass: first close attribution gaps around unprofiled work/queue gaps, then use the same timeline artifacts to verify any MoE graphlet/fusion attempt reduces both wall time and named-event gaps while preserving PP512 ~1200 tok/s.

## Notes

- No `sycl-ls`, VTune, DRM/P2P probes, `/dev/dri` probes, or `lsof` were run.
- Lead-only model access was limited to the approved GPT-OSS B50 `llama-bench` validation through `scripts/sycl-gptoss-decode-timeline-profile.sh --execute --i-understand-this-runs-gpu-models`.
