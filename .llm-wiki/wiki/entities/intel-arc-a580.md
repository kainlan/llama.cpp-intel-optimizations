---
type: entity
title: Intel Arc A580
description: Discrete card removed from this machine on 2026-07-24; its P2P test results are historical and superseded by the B70.
created: 2026-08-29
updated: 2026-08-29
sources:
  - id: SRC-2026-08-29-003
    resource: /sources/SRC-2026-08-29-003.md
---

# Intel Arc A580

Discrete card removed from this machine on 2026-07-24 per compute-runtime.md (SRC-003); its P2P test results are historical and superseded by the B70.

## ⚠️ Naming discrepancy (needs human review)

compute-runtime.md calls the card replaced on 2026-07-24 the **A580**; the
perf-baseline doc, env-var doc, AGENTS.md and CLAUDE.md all call it the
**B580** (see [Intel Arc B580](/entities/intel-arc-b580.md)). If these are the
same physical card (likely, given the machine's Battlemage context), the two
pages should be merged and this page's source attribution folded in.

## Links

- [SRC-2026-08-29-003](/sources/SRC-2026-08-29-003.md)
- [Intel Arc B580](/entities/intel-arc-b580.md) — likely the same card under a different name
- [Intel Arc Pro B70](/entities/intel-arc-pro-b70.md) — its replacement
