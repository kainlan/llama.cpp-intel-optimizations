---
type: source
title: "Observation: Clean host-ordering RED with explicit rollback receipt"
tags:
  - glkg
  - red
  - verified
  - baseline
status: observation
created: 2026-09-14
updated: 2026-09-14
slug: obs-2026-09-14-clean-host-ordering-red-with-explicit-rollback-receipt
relevance: high
observed_at: 2026-09-14T16:17:00.689Z
source_context: Clean historical host-zone ordering gate
---

# ⭐ Observation: Clean host-ordering RED with explicit rollback receipt

d7a30717910bbb8f96640bbe7723d2b3b08aa17c completed the historical allocator RED gate. Lead ran B50 once: /home/kainlan/glkg-d7-runtime-HMxsg3/run-1, rc1,1run/0pass,0.441s, exact HOST_ORDER_RED and exactly one PASS cleanup receipt (MISSING_SUCCESS10,candidate_absent1,active0,models0,baseline0). No other FAILED lines, watchdog, survivors, kernel faults; lock released. Build162 hashes verified, manifest436884958fec0b37d7e22500639c4b5e186387e6d9b53937beebf31a85a53fd2, binary005b6f03abfd17cc84d774e4667043b2dd611554340f8239ca14969cff5d3513. This clean baseline must not be redone absent fixture changes. Production candidate091afc4391cf6421420c36b5c5067977e95c02c4 preserves exact fixture and awaits source findings/QUALITY/GREEN; whole-feature and post-load reserve qualification remain incomplete.

*Relevance: high*
*Context: Clean historical host-zone ordering gate*
*Tags: glkg red verified baseline*

---
*Observed: 2026-09-14T16:17:00.689Z*
