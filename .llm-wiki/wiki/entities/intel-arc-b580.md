---
type: entity
title: Intel Arc B580
description: Older Battlemage card removed from this machine on 2026-07-24 (replaced by the Arc Pro B70); its baselines are SUPERSEDED and gate nothing.
created: 2026-08-29
updated: 2026-08-29
sources:
  - id: SRC-2026-08-29-001
    resource: /sources/SRC-2026-08-29-001.md
  - id: SRC-2026-08-29-004
    resource: /sources/SRC-2026-08-29-004.md
---

# Intel Arc B580

Older Battlemage card that occupied slot `0000:03:00.0` until it was
physically replaced by the [Arc Pro B70](/entities/intel-arc-pro-b70.md) on
2026-07-24. Every B580 figure in the perf-baseline doc is retained as history
only, marked **SUPERSEDED**, and gates nothing — measuring a B70 against a
B580 target makes a healthy run look like a catastrophic regression.

## Key facts

- Historical Mistral 7B Q4_0 baselines: ~1700 PP512 / ~81 TG128 (default
  no-FA path); a `5b206c499-dirty` record of 2173.92 PP512 / 88.42 TG128
  (FA-on) appears in `docs/backend/SYCL.md` — same status, B580 history.
- Historical GPT-OSS 20B MXFP4: ~66 PP512 / ~17 TG128 (small VRAM budget
  caused heavy memory pressure on this model).
- Historical measurement basis for the (stale) ESIMD small-block dequant
  "1.9x slower" figure used to justify a `GGML_SYCL_ESIMD_DEQUANT` default —
  measured on B580 + old oneAPI; flagged as a documented contradiction.
- The B580-era Mistral FORCE_LEGACY row (~159 PP512, a ~10x collapse) is
  directionally corroborating for the same mechanism measured ~3x on current
  hardware, but unverified in magnitude.

## ⚠️ Naming discrepancy (needs human review)

compute-runtime.md (SRC-003) calls the replaced card the **"Arc A580"**
(see [Intel Arc A580](/entities/intel-arc-a580.md)); the perf-baseline doc,
env-var doc, AGENTS.md and CLAUDE.md all say **"B580"**. The Battlemage
G21/G31 context of this machine's cards favours B580, so the A580 mention is
likely a typo — but until confirmed, the two pages are kept separate.

## Merged from

Consolidates former duplicate pages `arc-b580`, `intel-arc-b580`.

## Links

- [SRC-2026-08-29-001](/sources/SRC-2026-08-29-001.md) · [SRC-2026-08-29-004](/sources/SRC-2026-08-29-004.md)
- [Intel Arc A580](/entities/intel-arc-a580.md) — possible alternate name for this card
- [Intel Arc Pro B70](/entities/intel-arc-pro-b70.md) — its replacement
