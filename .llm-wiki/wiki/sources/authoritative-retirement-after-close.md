---
type: source
title: Authoritative retirement after close-time observation
status: insight
category: supervision-safety
created: 2026-09-15
updated: 2026-09-15
slug: authoritative-retirement-after-close
---

# Authoritative retirement after close-time observation

Shared core `9695abb2e86caac7e81b0875d0c1863e5e8b90f58c89cf7b22f9814625436a23` performs another census inside `Owner.close()`. A caller that captures `terminal()` first and publishes that old certificate after close can miss newly recorded ambiguity, even if close returns without survivors. Integrated SPEC review identified this concrete source path in the real A/B/C adapter at `9ca161cd...`.

A deterministic exact-production-AST FINALIZER component regression observed intended `STALE_POSITIVE` RED before the behavioral fix. Production `5b31e91ba990e2697ecd9439f6877bce595223bd` closes first, then obtains the authoritative terminal certificate; no close/helper/hash follows that certificate. SAME regression GREEN retains latest ambiguity and refuses release. A simulated low-memory hash case records failure while still retiring owned work. This exercises the finalizer component, not main, validation, or real A/B/C payloads; activation pins remain None.

The first three test runs had their own pre-close outer-receipt limitation. Preserve them as limited evidence. An explicitly authorized three-run requalification used corrected test-only receipt ordering: old RED rc1, fixed GREEN rc0, low-memory rc0, all with actual post-close outer certificates, raw waits256/0/0, ECHILD, empty census, no errors/ambiguity/survivors/handles. Total attempts are THREE historical plus THREE requalification, not three. Evidence `/home/kainlan/glkg-finalizer-requalification-f0hqjkj9`, manifest `42aefc6e32eb96a30721d6cb012a91d5a4d5d852aea70f493e39c3366f2098d0`; external HEAD `b066bfbfeec0496c9caa832122b77675f4b238fb`.

Independent SPEC delta `f764c903-9d93-403b-9284-7722cd719279` verified the close-order fix and all48 manifest files, but leaves an explicit memory-policy gap: floor checks bracket up to30 seconds of cleanup rather than sampling throughout it. A transient dip/recovery there is untested and can be missed. Do not describe this as continuous1s whole-lifecycle monitoring or cleared real-stage execution. See [[kernel-live-owned-process-metadata-loss]] for the distinct core-level regression.

*Category: supervision-safety*

---
*Captured: 2026-09-15*

## Related

_Add links to related pages._
