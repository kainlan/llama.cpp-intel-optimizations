---
type: source
title: Kernel-live processes survive missing metadata
status: insight
category: supervision-safety
created: 2026-09-14
updated: 2026-09-14
slug: kernel-live-owned-process-metadata-loss
---

# Kernel-live processes survive missing metadata

A deterministic CPU regression exposed an actual unsafe release boundary in the shared SYCL job supervisor: an owned dummy remained alive according to its bound kernel pidfd, but injected snapshot/identity absence made frozen core `ae5ef1a...` report `root_exit=null`, no survivors, and `slot_released=true`. RED exited1 in0.4433s before production edits. This is a controlled metadata-loss experiment, not proof that a historical OS read failed.

The SAME `closure_probe.py` (SHA `13369dd56bad4ac12b7a400ae4808c8a171098bce5687093d44973ca4e0add8d`) passed against core `/home/kainlan/3aos-final-closure-r0qyah7y/safety.py`, SHA `9695abb2e86caac7e81b0875d0c1863e5e8b90f58c89cf7b22f9814625436a23`. GREEN exited0 in0.3305s and correctly held release while the dummy was kernel-live. Injection was restored before cleanup; both runs retired their owned dummies and preserved outer receipts before assertions. The negative GREEN's inner scope intentionally retained its metadata-ambiguity hold after physical retirement.

The candidate adds affirmative final owned-subtree retirement: exact managed wait statuses without fabricated ECHILD success, managed-before-broad-reap ordering, all-child ECHILD (not WNOHANG zero), final launch sealing, fresh census, and fail-closed metadata/poll errors. It does NOT waive unresolved PG-only holds or authenticate historical numeric PIDs. Existing nine core cases passed; one later seven-case inert-adapter invocation passed with14 affirmative inner/outer certificates. Real archive/config/build/model execution remains unqualified.

Evidence: `/home/kainlan/3aos-final-closure-r0qyah7y/{red,green,dummy-results-k6a_5as9}` and `/home/kainlan/glkg-final-closure-convergence-WBnSFAUg`; tracker `llama.cpp-36a3` comments `c-seu4`, `c-29mi`. See [[obs-2026-09-14-supervisor-lineage-green-but-adapter-run-remains-unqualified]] for the distinct, historically unidentified PG-only events.

*Category: supervision-safety*

---
*Captured: 2026-09-14*

## Related

_Add links to related pages._
