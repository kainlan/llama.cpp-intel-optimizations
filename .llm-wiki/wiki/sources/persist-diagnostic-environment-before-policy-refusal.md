---
type: source
title: Environment persistence before policy refusal
status: insight
category: debugging
created: 2026-09-16
updated: 2026-09-16
slug: persist-diagnostic-environment-before-policy-refusal
---

# Environment persistence before policy refusal

The existing ABC worker compared its actual projected environment with the grant before writing `environment.json`. A mismatch therefore discarded the exact values needed to diagnose configuration. The minimal fix in `/home/kainlan/glkg-abc-prerequisites/build_worker.py:56–59` moves the existing bounded atomic write before the unchanged strict comparison; computation, keys/prefixes, writer arguments, downstream checks and failure handling remain unchanged. Exact byte/AST reversal restores the baseline.

Candidate `9fa434c9d7793b9afa59c09391790c31923dd759`, worker SHA `49931342a96650e2364606d8a79f943cf73c9b932240db087bdb01e45c65166a`. Oracle frozen at `6467524` before editing; retained qualified RED was exit1/2.506s with no environment file. Fresh GREEN was still exit1/mismatch in1.506s, but retained3296bytes/19keys and exact hashes in both manifests, with both post-close custody certificates qualified. Evidence `2ef7b278d42c11a0613dd5237b6227bb7efe2f96`; independent completed SPEC and QUALITY passed; mdja closed. See [[sources/failed-preflight-can-have-qualified-process-custody]].

GREEN here means diagnostic persistence, not successful Stage A or policy acceptance. Dynamic shape/value checks are partial and complemented by exact source/writer identity; full process environment was not independently reconstructed. Write/budget/receipt failures remain fail-closed but were not fault-injected. The captured map required separate static vendor provenance review and explicit CPU-A-only acceptance before the successful archive in [[sources/obs-2026-09-16-d7-runtime-archive-accepted-before-allocation-rebuild]]. It grants no B/C/R policy or execution authority.

*Category: debugging*

---
*Captured: 2026-09-16*

## Related

_Add links to related pages._
