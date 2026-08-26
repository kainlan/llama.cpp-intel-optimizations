# TKV-2 findings — CPU-attention floor calibration, GPT-OSS 20B B50 (llama.cpp-ectw)

Date: 2026-08-26 · Lead session · B50 (`level_zero:1`) · build at master `d8a9790fd` · raw cells: `tkv2-bench-cells.txt` (same dir)

Method: `llama-bench -m gpt-oss-20b-mxfp4.gguf -p 0 -n 32 -d <depth> -r 1 -nkvo <1|0>`, one invocation per cell, serial, Shmem sampled between cells (flat ≈3.6 GB throughout; GPU clean after — journalctl 0 hits). `-nkvo 1` = **all 24 layers'** KV in host memory, attention on CPU via the ggml scheduler — the exact executor the campaign's B1′ design routes *demoted* layers through, so these numbers are a **lower bound** for the landed feature (which demotes only overflow full-attention layers; SWA KV always stays on device).

## Measured cells

| fill depth | host-KV TG (nkvo=1) | device-KV TG (control) | ratio | cell load avg |
|---:|---:|---:|---:|---:|
| 0      | 10.62 | 12.58 | 0.84 | ~97 |
| 4,096  |  8.88 | 13.34 | 0.67 | ~115 |
| 16,384 |  5.83 | 13.34 | 0.44 | ~111 |
| 32,768 |  5.29 | 20.12 | 0.26 | ~111 |

⚠️ **Absolutes are load-contaminated and are NOT baselines** (owner ruling 2026-08-07): ambient load 95–115 during every cell (concurrent mega-TU builds by campaign implementers on this 20-core host); the healthy B50 GPT-OSS TG128 baseline is ~32. The control column itself wobbles 12.6→20.1 with load, so **the pairwise ratio at each depth is the trustworthy signal**, and even it is conservative (arms ran back-to-back, not interleaved within one process).

## Reading

- Host-KV TG decays bandwidth-bound with fill (10.6 → 5.3 from 0 → 32K), device-KV holds roughly flat — exactly the co-location physics the design docs measured (CPU reads host KV at DDR5 rates; device attention is unaffected by fill at these depths thanks to SWA + GQA).
- Even the worst case measured — **all** layers host, 32K fill, load ~111 — sustains **5.3 tok/s** with correct scheduler-driven execution. The landed feature at default GPT-OSS context demotes ≈8 of 12 full-attention layers (≈2/3 of full-attn KV) and zero SWA layers, so it strictly does less host work than `-nkvo` at every fill.

## Proposed perf floor (for owner ratification — the TKV-2 STOP-AND-ASK gate)

Acceptance floor for TKV-11, measured on the landed build, B50, default n_ctx (131072, no `-c` pin), ambient load as-is:

1. **Absolute bound:** decode TG ≥ **5.3 tok/s** at ≤32K actual fill — the landed partial-demotion path must not be slower than the all-host `-nkvo` bound measured here at the same fill.
2. **Ratio bound:** TG ≥ **50% of a same-session `-c 32768` device-KV control run** (back-to-back pair, same binary, same host state) at 32K fill. Ratio framing is robust to this host's permanent ambient load.
3. **In-VRAM invariant (already in the plan):** contexts that fit entirely in VRAM regress by ≤ noise band vs pre-campaign master (paired interleaved A/B per `docs/backend/sycl-perf-baselines.md`).

Not proposed: a floor at full 131K fill — no cell measured it (32K prefill already dominates cell wall-time; 131K prefill through CPU attention would blow the timeout), and the gate's chat runs never approach it. If the owner wants one, it extrapolates to ≈2–3 tok/s all-host / ≈4–5 tok/s partial-demotion, bandwidth-scaled.
