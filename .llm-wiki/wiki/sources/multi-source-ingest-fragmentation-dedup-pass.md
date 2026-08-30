---
type: source
title: Multi-source background ingest fragments entities; run a dedup pass after
status: insight
category: workflow
created: 2026-08-29
updated: 2026-08-29
slug: multi-source-ingest-fragmentation-dedup-pass
---

# Multi-source background ingest fragments entities; run a dedup pass after

Ingesting multiple sources through background synthesis mints entity/concept pages **independently per source**, so the same real-world object gets 2–5 differently-named pages (e.g. one GPU landed in `b50`, `intel-arc-b50`, `arc-pro-b50`, `intel-arc-pro-b50`; tracker ticket IDs became pseudo-entities). The vault's lint reports this only as "N orphans / Warning" — "Good / 0 gaps" does NOT catch it, and it misses uningested `status: skeleton` source pages entirely.

Cure (done 2026-08-29, 38 pages merged in this vault): (1) merge each fragmented group into one canonical kebab-case page with a **union of the `sources:` frontmatter**; (2) delete the variants and the ticket-ID pseudo-entities; (3) repair inbound links globally (exact `(/dir/slug.md)` + `[[dir/slug]]` string replacements are collision-safe); (4) **regenerate every source page's "Entities/Concepts Mentioned" from the surviving pages' `sources:` frontmatter attribution** — that single step clears all orphans, since every derived page then gets an inbound link from its source; (5) `wiki_rebuild_meta` and verify with a disk-level link scan (orphan + dangling), because a rebuild can race a still-running background ingest and report stale counts. Watch for YAML frontmatter breaks when writing descriptions containing `: ` (quote the value).

*Category: workflow*

---
*Captured: 2026-08-29*

## Related

_Add links to related pages._
