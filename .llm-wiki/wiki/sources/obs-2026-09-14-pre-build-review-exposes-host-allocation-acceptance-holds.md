---
type: source
title: "Observation: Pre-build review exposes host allocation acceptance holds"
tags:
  - glkg
  - review
  - memory
  - contracts
status: observation
created: 2026-09-14
updated: 2026-09-14
slug: obs-2026-09-14-pre-build-review-exposes-host-allocation-acceptance-holds
relevance: high
observed_at: 2026-09-14T17:14:56.553Z
source_context: Production candidate SPEC→QUALITY qualification
---

# ⭐ Observation: Pre-build review exposes host allocation acceptance holds

Astra-low SPEC f261c6b7 and QUALITY dca5b4a7 cleared candidate528397d038a5ac0b2ecf60972e52d6013f4498a0 only for a bounded build/unchanged ordering case, not acceptance. get_max_size is an optional capability upper bound (defaultSIZE_MAX), not an allocation reservation; its2GiB floor masks many small-pool differences. Query-current-device vs buft-device allocation differs in PER_DEVICE, whose caches own separate pinned pools; GLOBAL mapsall0. Positive HOST_RESERVE_MB currently overrides percentage auto-input; byte-budget sentinel has no writer. Explicit environment/API priority awaits user choice. Newholds xgr6: eager host provisioning remains under inventory mutex and pool physical-allocation/wait locks, violating architecture§12.5;16el: newlylive parser accepts junk/ignoresERANGE and unchecked MiB conversion. All-device host-noop/rawfallback/protected64MiB claims were corrected in comments-only528. No candidateGREEN yet.

**Disposition update, 17:23 UTC:** `16el` is a static robustness follow-up, not an ordinary executable FAIL or automatic expansion of the approved gate; its dependency on `glkg` was removed. `xgr6` remains a narrow, explicitly lead-approved safety hold under the existing §12.5 contract, without a demonstrated deadlock claim. The bounded ordering diagnostic remains allowed. See tracker comment `c-36iy`.

*Relevance: high*
*Context: Production candidate SPEC→QUALITY qualification*
*Tags: glkg review memory contracts*

---
*Observed: 2026-09-14T17:14:56.553Z*
