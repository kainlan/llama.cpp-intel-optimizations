# iiff criterion 5 long-soak — PRE-REGISTERED criteria (written before any soak ran)

**Criterion under test:** "`zone_largest_free` does not degrade over a long run."

## What the instrument reports
Each audit site row is followed by:
`[ZONE-RESET-AUDIT]   largest_free first=A MB last=B MB min=C MB`
(first visit / last visit / minimum across the run).

## The baseline being extended (capture 01, Mistral, 16 graphs, B50 pinned)
| site | first | last | min |
|---|---|---|---|
| device-zone-reset/ONEDNN:dev0 | 256.00 | 143.88 | 143.88 |
| device-zone-reset/SCRATCH:dev0 | 512.00 | 512.00 | 512.00 |
| host-zone-reset/SCRATCH | 309.54 | 309.54 | 309.54 |
| host-zone-reset/STAGING | 223.08 | 223.08 | 223.08 |
| weight-reclaim/load-boundary | 13586.00 | 13586.00 | 13586.00 |
| weight-reclaim/mid-load-replan | 13586.00 | 13586.00 | 13586.00 |
| weight-reclaim/model-teardown | 5218.10 | 5218.10 | 5218.10 |

**Why 16 graphs cannot settle the criterion:** the ONEDNN 256→143.88 step is
indistinguishable, at this length, between (a) a ONE-TIME occupancy step that then
holds flat forever, and (b) a slow monotonic decline that would keep eating the
zone over thousands of graphs. Only graph count separates them.

## Soaks (each a SINGLE process — never loop a model-loading binary; pinned selector; Shmem sampled before/after with settle)
- **A — apples-to-apples vs capture 01:** same binary/model/selector as capture 01,
  `-n 3000` instead of `-n 15` → ~3000 graphs (~190x the baseline).
- **B — allocation-churn-heavy:** `llama-bench -p 512 -n 128 -r 30 -v` (Mistral, B50)
  → ~30 x (pp + 128 tg) graphs with repeated per-rep buffer cycles.
- **C — MoE/host-zone shape:** `llama-bench -p 512 -n 128 -r 15 -v` (GPT-OSS, B50).

## Verdict rules (fixed in advance)
PASS (no degradation) requires ALL of:
1. No zone's `min` in a soak is materially below its capture-01 `min` — tolerance
   **5%** — i.e. the ONEDNN step stays ~143.88 MB rather than deepening.
2. No zone's `min` scales with graph count (compare soak A ~3000 graphs vs soak B/C:
   a decline tracking run length is the fragmentation signature).
3. No zone's `min` collapses toward zero while the zone is otherwise idle.
4. `last` returns to `first` for every zone that started flat (coalescing recovers).

FAIL (fragmentation) is any of: `min` monotonically tracking graph count; `min`
materially deeper than baseline; a previously-flat zone developing a gap.

**Ambiguity rule:** if a soak's `min` moves but by a fixed step that does NOT deepen
with more graphs, that is OCCUPANCY (an allocation that lives longer), not
fragmentation — report it as such with the evidence, do not call it a FAIL.

**Void rule:** a soak whose audit output lacks `largest_free` lines for the zones of
interest is VOID, not clean — re-run rather than report.
