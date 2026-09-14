---
type: source
title: Shared supervisor parent-lineage regression
status: insight
category: orchestration
created: 2026-09-14
updated: 2026-09-14
slug: shared-supervisor-parent-lineage-red
---

# Shared supervisor parent-lineage regression

The clean allocation RED in [[sources/obs-2026-09-14-clean-host-ordering-red-with-explicit-rollback-receipt]] does not qualify supervisor cancellation or ancestry handling.

A separate, inert regression against shared core `4912ce30014f3993981de66ecff5c6d7052db611b3e27c4c051a80e2f52de58f` reproduced incorrect child promotion after parent identity changed during discovery. The old core promoted simulated PID200/birth20 and recorded a simulated signal, while retaining the previously authenticated PID300/birth30. The assertion failed as intended (exit1, 0.021862652s); no real signals or children occurred inside the probe. The outer helper was reaped, but its PID was not recorded. Evidence: `/home/kainlan/3aos-shared-lineage-mmui3p14/{lineage_probe.py,lineage-red.log,lineage-red-result.json,red-interpreter.json}`; probe SHA `276a1aeb867e1a589e885bcb7244db35c3e6679346f7ca2e2a2cd132804638d9`.

A child's own valid pidfd is insufficient proof of ancestry. Newly discovered handles need tentative retention and post-discovery validation of the original parent's birth and pidfd liveness before promotion; invalid tentative handles must be discarded without signaling, preserving already-authenticated children. This is a monitor-boundary regression, not an observed production kernel race. GREEN is not completed; task `llama.cpp-36a3` remains open, and no heavy job is authorized through this core.

*Category: orchestration*

---
*Captured: 2026-09-14*

## Related

_Add links to related pages._
