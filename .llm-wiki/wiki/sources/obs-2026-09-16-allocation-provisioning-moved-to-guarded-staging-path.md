---
type: source
title: "Observation: Allocation provisioning moved to guarded staging path"
tags:
  - sycl
  - allocation
  - lifecycle
  - inventory-lock
  - application-fix
status: observation
created: 2026-09-16
updated: 2026-09-16
slug: obs-2026-09-16-allocation-provisioning-moved-to-guarded-staging-path
relevance: high
observed_at: 2026-09-16T02:46:02.202Z
source_context: xgr6 c-zoes/c-9r1c/c-4of8/c-gymy; advisory69ab
---

# ⭐ Observation: Allocation provisioning moved to guarded staging path

Originalglkg committed production-source correction8c8a0afaea0865dbf8ca13ba505c32dea559564e over528397d03 in /Apps/llama.cpp-wt-glkg: ggml/src/ggml-sycl/ggml-sycl.cpp plus three comment lines in ggml/include/ggml-sycl.h. The public setter lacks the staging wrapper's lifetime guards, so blindly unlocking it was rejected. Owner PRE-091 history and independent69ab consultation supported moving only newly introduced eager provisioning into guarded late-stage_inventory_plan. A private setter captures the exact candidate snapshot under inventory lock; staged provisioning runs after unlock while module/load-effect/backend/rollback guards remain alive, with full effect-owner token and candidate-pointer checks before/after. Numeric candidate.version is not a generation (staged candidates default0). Direct setter deliberately regains PRE091 behavior, not528 eager behavior; exported signature and two legacy S1 fallback callers remain. Explicit-no-plan and optional fallback behavior retained; no atomic competing-provisioning or full shared-capacity rollback claim. Original runtimefixture bdfae5b28 and protected staged blobs unchanged. Diff checks passed, but no compile/runtime test or GREEN yet; independent source review usvq pending. This addresses a source-grounded §12.5 safety violation, not an observed deadlock; existing pool-lock waits and broader acceptance remain separate.

*Relevance: high*
*Context: xgr6 c-zoes/c-9r1c/c-4of8/c-gymy; advisory69ab*
*Tags: sycl allocation lifecycle inventory-lock application-fix*

---
*Observed: 2026-09-16T02:46:02.202Z*
