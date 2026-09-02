---
type: entity
title: llama.cpp
description: Inference framework whose isolated/host-bounce multi-device path (GGML_SYCL_MOE_MULTI_GPU) is constrained by the P2P restriction.
created: 2026-08-29
updated: 2026-09-02
sources:
  - id: SRC-2026-08-29-003
    resource: /sources/SRC-2026-08-29-003.md
  - id: SRC-2026-09-02-006
    resource: /sources/SRC-2026-09-02-006.md
  - id: SRC-2026-09-02-008
    resource: /sources/SRC-2026-09-02-008.md
---

# llama.cpp

Inference framework whose isolated/host-bounce multi-device path (GGML_SYCL_MOE_MULTI_GPU) is constrained by the P2P restriction.

## Overview

[Key facts]

## Upstream features relevant to this fork (2026-09-01 research)

- **Backend-agnostic tensor parallelism (`--split-mode tensor`)** merged
  upstream around April 2026 (PR [#19378](https://github.com/ggml-org/llama.cpp/pull/19378)),
  optimized for 2 GPUs; the SYCL implementation landed via PR
  [#24152](https://github.com/ggml-org/llama.cpp/pull/24152)
  (`comm_init`/`comm_free`/`comm_allreduce_tensor`, small tensors via FP32
  direct memcpy + per-device ADD kernels, large tensors via BF16-compressed
  cross-device memcpy). GPT-OSS is among the upstream-supported MoE
  architectures for this path. This is a candidate alternative/complement to
  the fork's own dual-device block-pipeline design
  ([second-card-earns-its-keep](/syntheses/sycl-fork-epic-taxonomy-2026-09.md));
  evaluating it against the fork's bar is an open question, tracked as
  `llama.cpp-k2y8`.
- **Upstream PR [#25874](https://github.com/ggml-org/llama.cpp/pull/25874)**
  ("Extended SYCL oneDNN SDPA to non-FP16 KV caches", merged 2026-08-04)
  measured 2.37x–3.21x prefill speedup on Intel Arc GPUs when quantized or
  F32 KV is dequantized to dense F16 before the oneDNN SDPA graph (gated to
  prefill K≥1024, batch≥32) — directly relevant to this fork's own
  `op-coverage-and-upstream-ports` port queue.
- The **b10630 merge** (2026-08-26, sha `1ebfa4e4a`) landed core-ggml
  prerequisites this fork had open tickets waiting on, including
  `ggml_ssm_scan()`'s trailing `int64_t K` parameter and
  `ggml_rope_set_offset()` — confirmed present in `ggml.h` at HEAD
  `fed0b58e2`.

## Links

- [SRC-2026-08-29-003](/sources/SRC-2026-08-29-003.md)
- [SRC-2026-09-02-006](/sources/SRC-2026-09-02-006.md) — `second-card-earns-its-keep` epic web findings (upstream tensor-parallel PRs)
- [SRC-2026-09-02-008](/sources/SRC-2026-09-02-008.md) — `op-coverage-and-upstream-ports` epic web findings (oneDNN SDPA PR, b10630 prerequisites)
- [SYCL fork epic taxonomy (2026-09-01 tracker triage)](/syntheses/sycl-fork-epic-taxonomy-2026-09.md)
