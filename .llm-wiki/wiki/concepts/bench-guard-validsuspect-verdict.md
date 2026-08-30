---
type: concept
title: bench-guard VALID/SUSPECT verdict
description: "First-line stamp on every archived benchmark log; a number whose log does not start with '# bench-guard: VALID' is not a citable baseline."
created: 2026-08-29
updated: 2026-08-29
sources:
  - id: SRC-2026-08-29-001
    resource: /sources/SRC-2026-08-29-001.md
---

# bench-guard VALID/SUSPECT verdict

`scripts/bench-guard.sh` stamps every archived benchmark log's first line with
a `VALID` / `SUSPECT` verdict (and refuses to run at all — exit 3 — on a
corrupted host).

**A number whose log does not start with `# bench-guard: VALID` is not a
baseline and must not be cited** — cite the SUSPECT reason instead, or
re-run.

## Merged from

Consolidates former duplicate page `validsuspect-verdict`.

## Links

- [SRC-2026-08-29-001](/sources/SRC-2026-08-29-001.md)
- [scripts/bench-guard.sh](/entities/scriptsbench-guardsh.md)
- [Bench-guard preflight refusal](/concepts/bench-guard-preflight-refusal.md)
- [Measurement masking (co-resident tenants)](/concepts/measurement-masking-co-resident-tenants.md)
