---
type: source
title: "Observation: Controlled reproduction of stale process-snapshot reuse"
tags:
  - sycl
  - supervision
  - regression
  - stale-snapshot
  - esrch
status: observation
created: 2026-09-15
updated: 2026-09-15
slug: obs-2026-09-15-controlled-reproduction-of-stale-process-snapshot-reuse
relevance: high
observed_at: 2026-09-15T23:07:27.183Z
source_context: ncsz c-rjif/c-6tcy; actual causal RED on canonical9695
---

# ⭐ Observation: Controlled reproduction of stale process-snapshot reuse

ncsz attempt8 ran exactly once on unchanged canonical core9695 with fixture04b0e7a8/packagef3625700: intended CAUSAL_RED_STALE_ROW_PG, actual exit1, external foreground0.802s. Evidence /home/kainlan/glkg-causal-esrch-ncsz/evidence-attempt8; external /home/kainlan/glkg-causal-esrch-attempt8-owner-1To4Pihe; 28-file manifest SHA40cda3571c0aea1678982064e503b93e861ec108bfc7bfb8782265fc102b8fb3. Real leaf4189620/birth60588727 was genuinely admitted by both non-SUT observers, then captured in a real snapshot. Held parent4189619 actually reaped it with decoded exit0. Original pidfd_open then returned real ESRCH, the same parent's post-ticket remained valid, and the original observer incorrectly reused the same stale row for an unproven-PG classification. No fabricated rows/errors or ownership shortcuts. Both non-SUT latest post-close certificates qualified; the SUT certificate remained unqualified with its recorded ambiguity. All managed decoded exits0, stopsnull, no survivors; test assertions occurred after receipts. This confirms the controlled stale-row defect, not a repair or attribution of the separate FA native failure. Total campaign8; no attempt9 or core change authorized. The owner's run reported no provider flag; a separate user-reported flag paused further launches.

*Relevance: high*
*Context: ncsz c-rjif/c-6tcy; actual causal RED on canonical9695*
*Tags: sycl supervision regression stale-snapshot esrch*

---
*Observed: 2026-09-15T23:07:27.183Z*
