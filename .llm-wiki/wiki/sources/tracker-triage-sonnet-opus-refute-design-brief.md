---
type: source
title: "Tracker triage method: sonnet triage + opus refute + design-brief rulings"
status: insight
category: process
created: 2026-09-02
updated: 2026-09-02
slug: tracker-triage-sonnet-opus-refute-design-brief
---

# Tracker triage method: sonnet triage + opus refute + design-brief rulings

On 2026-09-01 the codescout tracker (581 open tasks, ~4 months of accumulated work on the llama.cpp SYCL fork) was triaged in three stages: (1) a first pass proposed closures against a written design brief carrying the owner's numbered rulings (sole allocator, mem_handle-only ownership, zone-reset elimination, placement-decides-executor, layout-follows-residency, no-host-waits, never-shrink-context, correctness-before-throughput, device-selection-stays-with-oneAPI, plus a fixed verdict vocabulary: done/superseded/obsolete-hw/against-design/duplicate/unverifiable/keep/needs-gpu) and a hard evidence bar (a verdict of done/superseded/against-design had to cite a commit sha, file:line, doc section, or ruling number a skeptic could check); (2) an adversarial refute pass reviewed every proposed closure against that same bar; (3) 8 remaining undecided tickets were resolved by actually running GPU gates (not more reading).

Outcome: 250 closures were proposed, 82 were refuted and kept open — a 33% refutation rate. The evidence that held up under adversarial review was near-exclusively: a `git log -S`/`git log --grep` commit citation, a live `grep -c` against HEAD proving code presence/absence, or a doc section with a date that postdates the ticket. The evidence that got refuted was near-exclusively: a single GPU measurement treated as decisive on a value the ticket itself called marginal/noisy (see PL2-throttle and thermal variance precedent), a "premise is stale" claim that didn't check whether the *remaining* scope (not the original scope) was still live, and closures that answered the ticket's original framing rather than its current, superseded-by-later-work framing.

Why this matters beyond this one triage: a design-brief-with-rulings is what let a fast, cheap model pass (sonnet-tier) generate high-volume closure proposals without the tracker degrading into "keep everything, nothing ever closes" — the rulings gave it a citable ground truth to check claims against instead of judging code quality from scratch. The adversarial refute pass is what kept the false-closure rate low despite that speed: closing a ticket is a one-way, higher-cost mistake than leaving one open (a wrongly-closed ticket vanishes from view; a wrongly-kept one just sits there), so the two passes should have asymmetric bars — propose liberally, refute conservatively, and require the refuter to itself cite evidence rather than just express doubt. The GPU-gate-for-undecided-remainder pattern is the fallback for the genuinely small number of cases where no amount of static reading resolves the question and an actual measurement is cheaper than continued argument — 8 of 396 survivors needed it here, which is a useful ratio to expect: most ambiguity resolves on paper if the design brief is specific enough, and only true empirical unknowns need a gate.

See also: [[verified-for-A-asserted-for-B]] (the refuted "premise is stale" pattern is a cousin — checking the *current* remaining scope, not the original ticket text), [[a-review-findings-rationale-can-be-false]], [[a-single-point-divided-is-not-a-single-per-unit-rate]]-adjacent (the refuted single-GPU-measurement pattern is the same "one point is not a rate/verdict" trap applied to a closure decision instead of a throughput number).

*Category: process*

---
*Captured: 2026-09-02*

## Related

_Add links to related pages._
