---
type: source
title: "Wiki staleness audit: driver pin, weight-reclaim nuance, two missing owner rulings"
status: insight
category: architecture
created: 2026-09-02
updated: 2026-09-02
slug: 2026-09-02-wiki-staleness-audit-driver-and-rulings
---

# Wiki staleness audit: driver pin, weight-reclaim nuance, two missing owner rulings

Audited .llm-wiki/wiki/** against CLAUDE.md (captured as SRC-2026-09-02-001) and docs/backend/compute-runtime.md's loader-state correction (SRC-2026-09-02-002) and docs/plans/2026-08-27-tkv13-b2-addendum.md (SRC-2026-09-02-003).

Findings and fixes:
- The vault described the patched 26.22 Battlemage/BMG-only compute-runtime build as "the system-default Level Zero driver" in the present tense across intel-compute-runtime.md, entities/index.md, and cross-device-in-order-dependency-fixes.md. This was true only through 2026-08-17; a one-way PPA upgrade on 2026-08-18 moved the loaded driver to stock 26.31 (llama.cpp-09um, corrected in compute-runtime.md 2026-08-30). Whether 26.31 carries the patched build's three fixes (hung-i915 discovery, cross-device in-order deps, USM compression PR 930) is unverified. Fixed all three pages plus added a superseded-note to the SRC-2026-08-29-003 source reflection page. The P2P topology finding in that same source is unaffected — it's a PCI-slot fact, not driver-dependent.
- leases-inusecount.md overclaimed that in_use_count == 0 is sufficient for eviction. CLAUDE.md documents an ownership-aware reclaim predicate (weight_entry_reclaimable()) that also preserves live-model-owned entries at in_use_count==0 and MID_LOAD_REPLAN-suppressed ownerless entries — getting this backwards previously broke test-llama-archs/test-thread-safety via a premature GGML_ABORT. Added the caveat with the historical lesson.
- No page in the vault covered two dated owner rulings that already have real design docs behind them: "never shrink context, place KV instead" (llama.cpp-uize background + the tiered-KV placement work) and "no host waits — event-chain everything" (the concrete event-topology exceptions in the tkv13 B2 addendum: deferred input-side join, blocking-consumer flush, and teardown are the only allowed waits). Created both as new concept pages, cross-linked, and indexed.
- Checked but found NOT stale (already correctly framed as historical/superseded, or already flagged in existing Tensions/Caveats): B580 references (all past-tense/superseded already), the retired >=1100 B50 PP512 guardrail, GGML_SYCL_FA_ONEDNN_ALLOW (absent — was already removed and never reintroduced in the vault), GGML_SYCL_UNIFIED_CACHE as opt-out (absent), zone-reset-vs-mem_handle-refcounting contradiction (already flagged in SRC-2026-08-29-002's Tensions/Caveats and epoch-refcounted-transient-zones.md is appropriately scoped to reset-only zones), b10630 upstream merge (already correctly captured in SRC-2026-08-29-001's raw extract, dated after the 2026-08-26 merge).
- Coverage gaps noted but NOT filled (out of this audit's requested scope): the 80%-of-hardware-theoretical-peak efficiency goal (2026-09-01 ruling) and the iGPU VRAM-budget-must-not-count-host-RAM fact have zero wiki presence — nothing states them wrongly, they're just absent.

Lesson for future audits of this vault: staleness here is rarely "the page is wrong" — it's "the page was right on its capture date and a correction landed in the live docs afterward that never got re-ingested." Always diff a source's raw capture date against the live doc's most recent dated section headers before trusting an existing SRC-* citation.

*Category: architecture*

---
*Captured: 2026-09-02*

## Related

_Add links to related pages._
