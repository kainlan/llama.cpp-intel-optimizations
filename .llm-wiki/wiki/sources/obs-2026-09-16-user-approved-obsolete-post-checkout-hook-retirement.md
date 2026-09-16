---
type: source
title: "Observation: User-approved obsolete post-checkout hook retirement"
tags:
  - git
  - hooks
  - beads
  - retirement
  - authorization
  - stage-b
status: observation
created: 2026-09-16
updated: 2026-09-16
slug: obs-2026-09-16-user-approved-obsolete-post-checkout-hook-retirement
relevance: high
observed_at: 2026-09-16T19:08:30.541Z
source_context: Explicit user yes to narrow shared-hook retirement
---

# ⭐ Observation: User-approved obsolete post-checkout hook retirement

User approved retiring ONLY /Apps/llama.cpp/.git/hooks/post-checkout. On2026-09-16T18:52:43Z lead used atomic RENAME_NOREPLACE to preserve it as /Apps/llama.cpp/.git/hooks/post-checkout.retired-bd-20260916T185243Z-6dc38e2b62ba4d01baa7b42cd1b258d2. Verified SHA256 dcd7dd313bec02d8ddda36043a98977bcca10eb5d33df26d4eae3aefbd7292b3,491bytes,0755 andsameinode/owner/group/mtime; canonicalhookabsent, otherhookmetadataunchanged, directoryfsynced. Neither hook nor bd executed; no permissions/config/source changes. Do not restore without authorization or overwrite a newly installed hook. w4ol/c-t0l1 records completion; w4ol staysopen for other hooks, untouched and outside approval. Removed uvgc's umbrella w4ol dependency because this specific blocker is cleared. Configuration remains unapproved pending CMake dependency/binding, nestedparallelism proof and evidence-budget disposition. No checkout/build/model/GPU action.

*Relevance: high*
*Context: Explicit user yes to narrow shared-hook retirement*
*Tags: git hooks beads retirement authorization stage-b*

---
*Observed: 2026-09-16T19:08:30.541Z*
