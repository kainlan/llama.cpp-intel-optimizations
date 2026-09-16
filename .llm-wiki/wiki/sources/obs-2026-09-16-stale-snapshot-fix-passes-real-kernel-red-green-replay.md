---
type: source
title: "Observation: Stale-snapshot fix passes real-kernel RED/GREEN replay"
tags:
  - sycl
  - supervisor
  - esrch
  - regression
  - kernel
  - evidence
status: observation
created: 2026-09-16
updated: 2026-09-16
slug: obs-2026-09-16-stale-snapshot-fix-passes-real-kernel-red-green-replay
relevance: high
observed_at: 2026-09-16T03:22:26.152Z
source_context: ncsz c-kplt/c-dlqo/c-tshk; pfp4 c-t372/c-5ibq/c-2esm
---

# ⭐ Observation: Stale-snapshot fix passes real-kernel RED/GREEN replay

The existing controlled ESRCH reproducer now has a minimal role2-only selector/nonvacuous verdict at921f1d516ba4c850e38ec224ca2eac9622bbe055 in /home/kainlan/glkg-stale-snapshot-regression-ncsz. Its sut() body remains byte-identical555544dc and all nonSUT roles use original9695. Actual attempt9 baseline: CAUSAL_RED_STALE_ROW_PG, exit1/.679s, nonSUT afterclose receipts qualified and SUT ambiguity retained. Attempt10 candidate04e7: CAUSAL_GREEN_STALE_ROW_EXCLUDED, exit0/.640s, all3 afterclose receipts qualified. Both witnessed actual snapshot→parentwait0→same dictionary→real pidfd ESRCH3→valid held-parent postticket; fixed candidate had no stale-PG witness. Evidence2bf2b5554ccc50523e6f7b8a1d1092ee4b6c4806,61-file manifestd47377b981bf77b1c0acf28427dd7a25aad3e04a2439cc504def17a4c722b43b; total ncsz10, no11/retries. Separately, the already-prepared live-metadata regression9e1e763c passed once0/.426s/6checks, evidence7022279f4c76159f0ef659fb4de0176db3be3654: real bound child stayed alive across injected metadata absence; candidate retained unqualified history while actual independent9695 outer retired cleanly. Five prior component guards35checks unchanged. This supports only the narrow monitor correction: no canonical deployment, ABC/native/application qualification, FA attribution, historical ambiguity clearing or GPU reservation release. Completed-change SPEC lh04 pending; allocation application8c8 remains source-approved but uncompiled.

*Relevance: high*
*Context: ncsz c-kplt/c-dlqo/c-tshk; pfp4 c-t372/c-5ibq/c-2esm*
*Tags: sycl supervisor esrch regression kernel evidence*

---
*Observed: 2026-09-16T03:22:26.152Z*
