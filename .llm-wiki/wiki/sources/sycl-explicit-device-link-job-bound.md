---
type: source
title: Explicit SYCL device-link job bound
status: insight
category: build-safety
created: 2026-09-14
updated: 2026-09-14
slug: sycl-explicit-device-link-job-bound
---

# Explicit SYCL device-link job bound

For the pinned Intel oneAPI 2026.1 icpx toolchain, help output and the unchanged-config `-###` probe did not establish the default llvm-foreach job limit. Historical outer `--parallel 6` remains insufficient proof of nested compiler concurrency.

A separate, non-executing driver probe added `-fsycl-max-parallel-link-jobs=1` and emitted `--jobs=1` at all five printed llvm-foreach sites (three SPIR-V translations, two AOT actions). It used one existing object and the frozen SYCL shared-link flags; this was not a full production-link replay. Evidence: `/home/kainlan/glkg-inner-jobs-explicit-em0r_d2u/result.json`, SHA `c90f8d34c8b6ff3997c463087328903da5e485335708a76b8c3291204b45b887`; tracker `llama.cpp-glkg` comment `c-khjy`.

The lead accepted this as the proposed resource-only constraint for a future exact build grant: at most six outer compiler/AOT jobs with device post-link fanout one. This does NOT mean at most six arbitrary internal IGC/OS threads. No configured flags, historical builds, or archives were changed. Actual target-generated commands must still be verified, and supervisor/adapter clearance is separate. See [[shared-supervisor-parent-lineage-red]] and [[obs-2026-09-14-supervisor-lineage-green-but-adapter-run-remains-unqualified]].

*Category: build-safety*

---
*Captured: 2026-09-14*

## Related

_Add links to related pages._
