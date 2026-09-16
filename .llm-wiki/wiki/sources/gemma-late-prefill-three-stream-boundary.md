---
type: source
title: Gemma late-prefill three-stream failure boundary
status: insight
category: debugging
created: 2026-09-16
updated: 2026-09-16
slug: gemma-late-prefill-three-stream-boundary
---

# Gemma late-prefill three-stream failure boundary

The single existing logits diagnostic at `/home/kainlan/3aos-first-token-logits-astra-1` admitted all four42-token prompts together. All four responses were exact,14tokens,HTTP200; every first raw distribution selected token236770("1"), and post-sampling/slot logs agreed. The pytest command nevertheless exited1 afterward: strictUTF8 log reading encountered a byte-token piece containing isolated0xC2. That ancillary issue is deferred as task2yp2; it is not the historical model corruption and this run is not a pytestPASS. See [[sources/gemma-first-token-existing-logits-trace]].

Retained failing schedule `/home/kainlan/3aos-ab4-no-debug-ZbodP3/run-1/server.log` differs: slot3 starts alone; late slots0–2 start while it has8cachedtokens. Geometry Q8/S1→Q8/S4→Q22/S1→Q4/S4→Q26/S3; early slot's first token is correct. Then late slots have38cachedtokens; Q1/S1→Q4/S3 yields `<unused32>` for all three late slots while earlyslot produces comma. Q1/S4 follows only afterward. Cached38 is each slot's own processed prompt, not evidence of cross-slot copying.

Normal source paths already fence inputs: synchronous SYCL tensor_set drains device queues; scheduler split-input copies wait events/backend; normal prefill graph exits wait the compute stream. Do not add speculative synchronization. Actual buffer/setter and early-return branch remain unidentified for the failing calls. Next application focus is final-prefill positions38–41, KV indices, masks and output-row mapping for the three late streams. No Gemma repair is established. This remains application investigation, not a reason to extend testing infrastructure.

*Category: debugging*

---
*Captured: 2026-09-16*

## Related

_Add links to related pages._
