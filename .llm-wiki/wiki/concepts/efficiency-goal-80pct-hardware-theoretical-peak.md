---
type: concept
title: Efficiency goal is 80% of hardware theoretical peak
description: Owner ruling (llama.cpp-0oad, 2026-09-01) that the ≥80% efficiency target means 80% of the card's theoretical peak — memory-bandwidth utilization for decode, INT8 XMX compute utilization for prefill — never 80% of another model's measured tok/s.
created: 2026-09-02
updated: 2026-09-02
sources:
  - id: SRC-2026-09-02-004
    resource: /sources/SRC-2026-09-02-004.md
  - id: SRC-2026-09-02-001
    resource: /sources/SRC-2026-09-02-001.md
---

# Efficiency goal is 80% of hardware theoretical peak

Goal ticket `llama.cpp-0oad`'s "≥80% efficiency on both cards" means **80% of
the hardware's theoretical maximum** — memory-bandwidth utilization for
decode (tg), compute utilization for prefill (pp) — not 80% of any model's
measured throughput. Owner's words (2026-09-01): "that 80% line is meant to
be 80% of the theoretical maximum on the hardware, aka memory bandwidth
utilization and compute utilization."

## Why this needed a ruling

A prior pass had substituted "80% of Mistral Q8_0's measured tok/s,
byte-adjusted" as the target (`0oad` comment `c-9u5f`), because Mistral
appeared to beat a guessed roofline. That guess was built on two wrong
numbers: it used the B580's 456 GB/s figure for the Arc Pro B70 (the B70 is
actually 608 GB/s, corroborated by a kernel measured streaming 713 MB at
602 GB/s), and it took the model **file size** (7.62 GB) as the bytes
streamed per token — but the GET_ROWS-only embedding table (2.99 GB) is not
streamed on every token; only ≈5.18 GB/token actually is. Three "all axes
green" scoreboards were posted against that wrong anchor while the true
efficiency was B70 tg 33% / B50 tg 58% of bandwidth, and pp only 13–16% of
the FP16 XMX peak.

## How to score it

- **Decode (tg):** memory-bandwidth utilization. Card specs: B70 608 GB/s,
  B50 224 GB/s (Intel memory specs). Compute utilization caps come from the
  device caps the backend prints at startup (`[SYCL] Device N caps: CU=…` is
  the XVE count; a Xe2 core is 8 XVE at 256 FP32 flop/clk/core and
  ~2044 FP16-XMX flop/clk/core; max clock from
  `/sys/class/drm/cardN/device/tile0/gt0/freq0/rp0_freq`).
- **Streamed bytes** must be the sum of per-token MUL_MAT weight bytes from a
  **kernel-profile census** — never the model file size, which over-counts
  tensors (like the embedding table) that are not read every token.
- **Prefill (pp):** ruled 2026-09-01 to be scored against the **INT8 XMX
  peak** — B70 367 TOPS, B50 170 TOPS — not the FP16 XMX peak, because
  Q8_0×Q8_1 is an INT8 dot product and the kernel choice used must not
  redefine the ceiling. Score as `2 × weights × tok/s ÷ INT8_peak`. Example
  from gemma-4-E4B: B70 6.4%, B50 7.9% at 2530/1446 tok/s respectively — the
  attainable ceiling (~40%, per Xe-Forge measurements on a B70) is the
  working near-term milestone, not 80%, which remains the longer-range goal.
- **Never re-anchor on another model's measured throughput.** The anchor is
  always the hardware roofline, derived from device caps and vendor memory
  specs, not from what any one model achieves.

## Links

- [SRC-2026-09-02-004](/sources/SRC-2026-09-02-004.md) — design brief, hardware/environment truths section
- [SRC-2026-09-02-001](/sources/SRC-2026-09-02-001.md) — CLAUDE.md perf-baseline and driver context
- [PP512 / TG128](/concepts/pp512-tg128.md)
- [Never shrink context — place KV instead](/concepts/never-shrink-context-place-kv-instead.md) — the sibling owner ruling from the same review pass

**Tracked by:** `llama.cpp-30ak7`, `llama.cpp-xihy`, `llama.cpp-po3nd.2` — see [SYCL fork epic taxonomy (2026-09-01 tracker triage)](/syntheses/sycl-fork-epic-taxonomy-2026-09.md)
