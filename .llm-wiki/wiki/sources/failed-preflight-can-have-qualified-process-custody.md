---
type: source
title: Failed preflight with qualified process custody
status: insight
category: debugging
created: 2026-09-16
updated: 2026-09-16
slug: failed-preflight-can-have-qualified-process-custody
---

# Failed preflight with qualified process custody

A failed business operation can still have qualified process custody. In the second real CPU preflight for [[sources/native-pi-astra-low-campaign-orchestration]], the unchanged worker rejected an intentionally empty expected environment before publishing `environment.json`: exit1/2.506s, no stages/payload/archive. Both final post-close certificates independently established sealed managed wait, raw status256, ECHILD, fresh empty census, closed handles, no ambiguity/survivors/stop. The outer message `inner certificate failure` came from `abc_outer.py:90` requiring BOTH successful status and valid custody; the failed status alone caused rejection.

Evidence: `/home/kainlan/glkg-abc-prerequisites/preflight-70-diagnostic-1`, commit `d795051d400e959745cff41dc1e76fd18cd67e39`, manifest `07cc1a9b26b20399131784a20f4bb1d19922745e89e3b62d80cc028e32cd8fbd`; independent review `llama.cpp-3rvq` passed. This supports a qualified negative baseline for the persistence-order defect, not Stage A success. See [[sources/obs-2026-09-16-native-pi-diagnostic-retention-and-qualified-negative-prefli]].

The reviewed diagnostic tail retains127 ordinary records plus firstPG, not complete causality or a byte/RSS bound. Both firstPG fields were null in this run;1138 outer events were omitted. Non-reproduction does not explain or qualify the historical PG2869069 failure. Preserve both histories. Before changing the worker, freeze the oracle: persist exactly the unchanged computed projection using the existing bounded writer before the unchanged equality refusal; retain exit1, no payload/archive, fail-closed write failures, bounds and independently qualified custody. Captured environment values are diagnostic evidence, not an automatically accepted policy.

*Category: debugging*

---
*Captured: 2026-09-16*

## Related

_Add links to related pages._
