---
type: source
title: Existing Gemma first-token logits discriminator
status: insight
category: debugging
created: 2026-09-16
updated: 2026-09-16
slug: gemma-first-token-existing-logits-trace
---

# Existing Gemma first-token logits discriminator

The Gemma four-separated-slot failure needs no new logging framework. Existing `common/sampling.cpp` synchronizes before sampling and supports `LLAMA_LOGITS_TRACE`, raw/post phases and top-N output; the archived ab4 runtime contains these facilities. No concrete missing-synchronization or routing defect was found in the latest source pass.

The trace counter is process-global, not per slot. Four fixture requests each permit48tokens, so limit16 can miss late first tokens. Limit145 (=3×48+1) covers every first token under ordinary non-speculative sampling; raw/post share a sample number. Correlate with chronological server slot n_gen/token logs. Wrong raw logits implicate computation or row association, not necessarily one distinguishable from the other; a raw/selected discrepancy implicates sampling, and selected/slot-token discrepancy routing.

Use the existing prepared Python environment and `tools/server/tests` cwd with the unchanged acceptance fixture. Exact proposed command is in task3aos/c-k6et. This diagnostic remains UNRUN; passing under tracing is inconclusive because logging changes scheduling. Archived binaryab4 is not sourceHEAD801, and fixture source_sha is not binary provenance. See [[sources/obs-2026-09-16-user-correction-prioritize-application-fixes-not-test-infras]].

*Category: debugging*

---
*Captured: 2026-09-16*

## Related

_Add links to related pages._
