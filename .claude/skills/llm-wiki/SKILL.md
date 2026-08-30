---
name: llm-wiki
description: Consult and maintain the project's shared LLM Wiki vault (.llm-wiki/, OKF v0.2, Obsidian-compatible). Use when researching how a subsystem works, before writing new design notes, or to bank a durable insight after finishing a task.
---

# LLM Wiki (shared knowledge vault)

The project knowledge vault lives at `/Apps/llama.cpp/.llm-wiki/` (OKF v0.2). It is
shared across harnesses: pi and oh-my-pi load it via the `@zosmaai/pi-llm-wiki`
extension; Claude Code reaches the same vault through the `llm-wiki` MCP server
configured in `.mcp.json` (tools: `wiki_recall`, `wiki_search`, `wiki_status`,
`wiki_retro`, `wiki_capture_source`, `wiki_bootstrap`); the Obsidian UI is the
`llm-wiki-obsidian` docker container (port 3001). All three views are the same
files — edits from any harness are visible everywhere.

Read `.llm-wiki/WIKI_SCHEMA.md` before writing pages.

## Ownership model (four layers — do not violate)

- `raw/sources/SRC-*/` — **immutable source packets.** Never edit. New material
  enters through `wiki_capture_source`, never by hand.
- `meta/` — **generated indexes** (backlinks, registry, event logs). Never edit
  by hand; they are rebuilt by the extension's tooling.
- `wiki/` — canonical pages (concepts, entities, syntheses, analyses).
  Editable by user and LLM. This is the only layer you write prose into.
- Events are append-only.

## Behavioral conventions (mirror of the pi-bundled skill)

- Capture first, integrate second: a new external source becomes a `SRC-*`
  packet via `wiki_capture_source`, then its facts get integrated into
  canonical pages that cite it.
- **Search before creating**: `wiki_recall`/`wiki_search` before any new page —
  the vault was dedup-curated (38 near-duplicate merges); do not regrow dupes.
- Cite facts by stable source-page ID (`SRC-...`), not by URL or memory.
- Default to read-only query mode; write only when banking something durable.
- Mixed/conflicting evidence goes in a page's "Tensions/Caveats" and "Open
  Questions" sections — 15 documented contradictions are already flagged for
  human review; add to them rather than silently picking a side.
- After completing a significant task, bank atomic insights with `wiki_retro`.

## What belongs here vs. elsewhere

- Wiki: durable knowledge about the system (how subsystems work, measured
  facts, design rationale, contradictions between sources).
- codescout tracker: live task/bug state.
- CLAUDE.md: operational rules for agents on this machine.
- Claude auto-memory: cross-session lessons for Claude specifically.
When a fact matures from "live debugging state" to "settled knowledge about
the system", it graduates from the tracker into a wiki page.
