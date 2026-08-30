---
type: source
title: "Observation: Wiki rebuild/lint results post async — verify file state before reacting to error toasts"
tags:
  - llm-wiki
  - tooling
  - async
status: observation
created: 2026-08-29
updated: 2026-08-29
slug: obs-2026-08-29-wiki-rebuild-lint-results-post-async-verify-file-state-befor
relevance: medium
observed_at: 2026-08-29T22:42:17.595Z
source_context: Reviewing curation pass results; user received stale frontmatter error notifications
---

# 🔍 Observation: Wiki rebuild/lint results post async — verify file state before reacting to error toasts

wiki_rebuild_meta and wiki_lint run in the background and their result toasts can land SEVERAL MINUTES after completion, out of order with later fixes. During the 2026-08-29 curation pass, two "rebuild had issues / frontmatter_parse_error" toasts (for mem-handle.md and bench-guard-validsuspect-verdict.md) arrived AFTER the descriptions had already been fixed, looking like a regression that did not exist. Distinguish by: (1) the toast's embedded error text shows the file content at scan time — if it shows pre-fix text it is stale; (2) re-run wiki_status or parse the frontmatter directly (yaml.safe_load) before re-fixing. A clean trailing toast ("metadata rebuilt — N pages indexed") supersedes earlier error toasts from the same job chain.

*Relevance: medium*
*Context: Reviewing curation pass results; user received stale frontmatter error notifications*
*Tags: llm-wiki tooling async*

---
*Observed: 2026-08-29T22:42:17.595Z*
