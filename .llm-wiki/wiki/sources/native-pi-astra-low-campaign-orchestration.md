---
type: source
title: Native Pi Astra-low campaign orchestration
status: insight
category: workflow
created: 2026-09-16
updated: 2026-09-16
slug: native-pi-astra-low-campaign-orchestration
---

# Native Pi Astra-low campaign orchestration

User requested native Pi agent delegation instead of Codescout team orchestration for the SYCL completion campaign. Use `delegate_task`, `wait_for_agents`, `agent_result` and `agent_message`; Codescout remains the required task tracker, not the worker transport.

Actual Pi oracle w14 ran gpt-6-astra/high despite model suffix `:low` and stopped at the identity gate. Pi explorer w15 verified gpt-6-astra/low and completed read-only source/evidence work. Built-in profile thinking overrides the suffix; fixer defaults medium, oracle/reviewer high. Project configuration `.pi/agent/agents-team.json` now explicitly sets all seven existing roles to `openai-codex/gpt-6-astra`, low. Explicit builtin tool/write lists and `prompt: default` are necessary: omitting access made the loaded fixer read-only. Installed schema4/scaffold3 loader validates final configuration with builtin capabilities and recursion/write-scope guards preserved. Project registration changes allowProjectProfiles/projectRoot as expected. Active JSON LSP check clean.

Installed extension loads configuration on session_start, not on each delegation. A user `/reload` is still required before claiming writer/reviewer settings active; verify actual identity on first subsequent worker. No old RPC session was migrated, no supervisor/application source changed, and no new execution occurred. Original logical ownership, worktrees and evidence remain intact. Tracker jthc records activation; s46z c-s7br holds the ready bounded diagnostic-retention handoff. See [[abc-cpu-integration-qualified-on-census-local-fix]].

*Category: workflow*

---
*Captured: 2026-09-16*

## Related

_Add links to related pages._
