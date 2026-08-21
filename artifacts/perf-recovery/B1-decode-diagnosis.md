# B1 — Decode overhead diagnosis (llama.cpp-fwhv)

Date: 2026-08-21. Host protocol: B50 (`level_zero:1`), cool-card verified (throttle/status==0, act_freq==0) before every run, no tenants, Shmem ~3 GB. bench-guard landed mid-task; runs used the manual protocol equivalent.

## Method

Two instruments, after a failed first attempt:

- **Rejected instrument:** `/usr/bin/time` user-CPU differencing (n64 vs n256). Δuser ≈ 28.3 ms/token — exactly the wall per token, i.e. a host thread spin-waits through generation (SYCL/L0 busy-wait), swamping any prologue signal. Negative finding recorded so nobody retries this.
- **Probe:** temporary chrono timer over the whole `ggml_sycl_mul_mat_id` body (all exits, via destructor), env-gated `GGML_SYCL_MOE_PROLOGUE_TIMING=1`. HEAD: commit `6f02a6681` (removed by B5). Baseline worktree: branch `b1-instr`, same probe (`/Apps/llama.cpp-baseline`, branch discarded at teardown; note its `build/bin/llama-bench` now contains the inert probe — rebuild from detached `79ae63559` if bit-exact binaries are ever needed).
- **perf:** flat symbol profile of a HEAD-policy decode run (`b1-perf.data`, 499 Hz, dwarf).

## Numbers (decode-only, `-p 0 -n 128 -r 1`, GPT-OSS 20B MXFP4)

| arm | tg128 | mul_mat_id host time/call (steady state) |
|---|---:|---:|
| baseline `79ae63559` (+probe) | 35.38 | **616 µs** |
| HEAD policy (`GGML_SYCL_MOE_PP_ONEDNN_F16_BATCHED=1`, +probe) | 25.92 | **890 µs** |

- Wall gap: 38.6 − 28.3 = **10.3 ms/token**.
- Host-time gap: 890 − 616 = **274 µs/call**; at the graph's ~36 MoE `mul_mat_id` calls/token ≈ **9.9 ms/token**.
- **The `mul_mat_id` host-time delta accounts for ~96% of the decode regression.**

## Verdict

**Prologue/host-dispatch-dominated → B3 (route table) proceeds as designed.** Not a kernel regression: the device kernels are not the gap; the host-side per-op dispatch cost is.

Shape of the cost (perf, HEAD decode phase): 51%+11% of samples inside `libze_intel_gpu.so` polling entry points — the overhead is **wait-shaped**: the host blocks on small per-op device round-trips, it does not burn CPU in data structures. Top cost sites for B3/B4 to attack, in order:

1. **Per-op ids D2H refresh/wait** — `ggml_sycl_refresh_moe_ids_cache` / the moe ids cache path invoked per `mul_mat_id` (decode admission), plus the PP-side blocking ids copy (~`ggml-sycl.cpp:64316` region, the owner's known-debt wait). Every op pays a host-blocking device read.
2. **Per-op route resolution/retained-batch machinery** — the `[MOE-RESOLVE]`/retained-route building that reruns per token per tensor; the route table (B3) makes it once-per-generation.
3. Residual per-op submission overhead (barriers, small H2D uploads for pointer tables) — expected to shrink once 1-2 land; re-profile after B4 rather than pre-optimizing.

Baseline's own 616 µs/call shows the pre-regression dispatch also waited per-op; the route-table design may recover some of that too, but the B-track acceptance bar (tg128 ≥ 34.5) only requires closing the delta.

## Artifacts

Session scratchpad `dboi/`: `b1-head-probe.log`, `b1-base-probe.log`, `b1-base-n64.log`, `b1-base-n256.log`, `b1-perf.data`, `b1-perf-run.log`. Load-bearing numbers are all reproduced in this doc and the tracker comment (scratchpad is not durable).
