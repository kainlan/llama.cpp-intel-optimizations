---
type: source
title: "Observation: Four-line stale-snapshot fix passes five component controls"
tags:
  - sycl
  - monitor
  - stale-snapshot
  - component-tests
status: observation
created: 2026-09-16
updated: 2026-09-16
slug: obs-2026-09-16-four-line-stale-snapshot-fix-passes-five-component-controls
relevance: high
observed_at: 2026-09-16T00:26:02.943Z
source_context: pfp4 c-mmch/c-nlxi/c-ry4j/c-1puc
---

# ⭐ Observation: Four-line stale-snapshot fix passes five component controls

Original3aos implemented a private shared-monitor candidate at /home/kainlan/3aos-stale-snapshot-fix-pfp4/safety.py, SHA04e7ab13f69d40cc7160f12ecaa4510ef0fe211461ff280623ecee0c41ddfce4, delta7740199a. Four added lines at three AST sites: reset an invalidation map for each inner census, copy an exact row only on tentative-open ESRCH, and exclude only the matching PID/birth/full row from that same snapshot's final PG audit. Original9695 remains unchanged; existing ambiguity, live handles, metadata/poll/nonESRCH handling are unchanged. Five approved component cases ran once each and passed35predicates: row-birth-mismatch(.248s), fresh-census-reset(.270), unrelated-live-pg(.268), non-esrch(.291), sticky-ambiguity(.263), all exit0. Real original9695 owning after-close certificates qualified with zero handles/survivors and rawwait0; synthetic SUT rows/handles are explicitly not kernel ownership evidence. Sourcefixtureabad765c/manifest75e49380 remained fixed; evidencecommitfab80ae59f99d9d9419c0ceb07aa37a01bebdcbe; ledger373190c8a0a0222384f55f1fdeb3bac2257b0da4cfc9ec643c9f3d266be6c473; 41-record manifestb1d7e6f2bf8359a1ae34db0dda493ce510bcae59d8c3855e97bba530c0e49046. No causal candidate replay or deployment yet. Originalglkg is preparing candidate-selectable replay; live-bound-metadata regression and completed-change reviews remain before broader adoption.

*Relevance: high*
*Context: pfp4 c-mmch/c-nlxi/c-ry4j/c-1puc*
*Tags: sycl monitor stale-snapshot component-tests*

---
*Observed: 2026-09-16T00:26:02.943Z*
