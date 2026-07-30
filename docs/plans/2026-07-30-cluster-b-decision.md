# Cluster B decision — no fix plan; `reuse` stays opt-in

**Task:** `llama.cpp-79m7` (plan phase 2, Task 5) of
`docs/plans/2026-07-25-sycl-decode-launch-overhead-phase2.md`.
**Entry condition:** `llama.cpp-kjj9`'s verdict, recorded as Task 14 of
`docs/plans/2026-07-25-decode-host-overhead-findings.md` (commits `051a86ba6`,
`046fdbc5f`).
**Date:** 2026-07-30. **Decided at:** `39a171536`.

---

## The rule that fired

Plan phase 2 Task 5 fixed its decision table **in advance**, before any number
existed, precisely so that the outcome could not be renegotiated after seeing the
measurement:

| Task 4 shows | Action |
|---|---|
| Causal, gates pass, win reproduces interleaved | Make no-submission the default; keep an opt-out env var; update `docs/backend/sycl-env-vars.md` |
| Causal but a caller needs the event | Scope the narrower change that preserves the contract; do not force it |
| **Not causal** | **Leave `binbcast.cpp` alone. Re-attribute Cluster B from Task 7 and write no fix plan** |

`llama.cpp-kjj9` returned **NOT CAUSAL**. The third row fires.

## Decision

1. **`GGML_SYCL_BINBCAST_EVENT_MODE` keeps its default of `barrier`.** No default
   flip. (Note: on an in-order queue `BARRIER` is downgraded to `SAFE` at
   `binbcast.cpp:158-160`, so the executed default is the empty `single_task`
   marker. The configured default is `barrier`; the executed one is `safe`.)
2. **No fix plan is written for Cluster B.** `binbcast.cpp` is left alone on
   performance grounds.
3. **`reuse` remains available, opt-in, documented** in
   `docs/backend/sycl-env-vars.md`. It is correct, gated, and shipped — it is
   simply not a 5.5 ms/step lever, because that lever does not exist.
4. **Cluster B's ~5.3 ms/step is UNATTRIBUTED** and its re-attribution belongs to
   `llama.cpp-hzgc` (plan Task 7), now in progress.

## Why, in one line each

- Deleting all 72 markers/step recovered **0.293 ms/step — their own cost**. The
  three Cluster B transitions kept their gap counts **event-for-event**
  (99 / 2300 / 2388→2389) and merely re-labelled their predecessor from
  `binbcast.event` to `binbcast.mul`.
- Unprofiled interleaved A/B, 10 order-counterbalanced pairs: **+0.244 t/s
  (+0.51 %), 95 % CI −0.45…+0.94 — spans zero.** 9/10 pairs positive
  (sign test p = 0.022), so the direction is likely real while the magnitude is
  not resolvable against B70 tg noise (documented cv 3.3 %).
- Honest effect size: **0.1–0.3 ms/step, ~0.5–1.4 % of TG.**

## What would justify revisiting the default later

Not this measurement. A future case for `reuse` as the default would have to rest
on it being a **structural cleanup** rather than a throughput win:

- 7200 fewer submissions per 100 steps (72/step) on the decode path;
- it eliminates the only explicit `queue_serialization` detected anywhere in the
  capture window (95 gaps, 0.056 ms total — the `binbcast.mul --to--
  binbcast.event` transition itself);
- ~0.1–0.3 ms/step, real but at the edge of measurability.

**And it must not be flipped before `llama.cpp-g6iw` closes.** The weight-cache
lease consumer, `cache->unpin_on_event`, is **never reached in any configuration
measured** (`pin_count == 0` throughout), so the passing gates do **not**
demonstrate that `reuse` is safe for the lease release. That path is correct by
inspection of `ggml_sycl_binbcast_resolve_event_source` plus a host-only policy
assertion — not by execution. Its failure mode is an early unpin, which presents
as a **speedup**. Flipping a default on an untested path whose bug looks like a
win is the exact shape of the fake +19.6 % PP regression this fork already had to
revert.

## What was ruled out, and is not worth re-testing

| candidate | status | evidence |
|---|---|---|
| The `binbcast.event` marker | **Not causal** | Deleted it; the gaps survived, counts identical |
| Implicit in-order queue serialization | **Ruled out** | 0.000 ms/step both arms, at 100 % `device_submit_ns` coverage (46100/46100) |
| Explicit `queue_serialization` | **Negligible** | Only instance was the marker transition, 0.001 ms/step |
| Graph replay / recorded marker nodes | **Not applicable** | GPT-OSS decode uses the direct TG executor; `graph_recorded=0` on all 46100 events |
| `llama.cpp-fvcx` (raw vs canonical recording flag) | **Unproven, no known trigger** | The recording path is never taken on this executor; the capture is silent in both directions |

## Process note

This decision record was written by the lead directly rather than by a spawned
implementer with spec and quality reviewers. The reason is that the decision
table above was **fixed in advance** and the entry condition is a verdict already
verified by two independent reviewers: there is no design latitude left to
exercise, no code change to make, and the deliverable is the application of a
pre-agreed rule. Spawning three agents to write "the rule says do nothing" would
have been ceremony, not diligence.

The substantive half of this plan branch — actually re-attributing the 5.3 ms/step
the marker turned out not to explain — is `llama.cpp-hzgc`, and that one **does**
have an implementer and will get the full review treatment, because it is where
the remaining judgement lives.
