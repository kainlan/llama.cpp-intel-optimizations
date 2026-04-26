# Unified Memory Placement Readiness Review

Date: 2026-04-25

## Readiness

The plan is ready to execute as an implementation program, but not "clean."
The design-validation state is strong enough to start Track A and Track E work:
A3a/A3b, A7, and C2 have empirical support, and m09zb is resolved under patched
libze. The remaining issues are tracking hygiene and a few explicitly unbacked
claims, not fundamental design blockers.

## A. Consistency And Evidence

Mostly consistent with the 2026-04-24 outcomes, with these exceptions:

- The main design doc has stale task-row text: A3a/A3b still say Task 5 is not
  written / awaited, while the final summary says Task 5 passed and validates
  both.
- There is an ownership inconsistency: one section says `llama_model` owns
  `weight_plan`, while D10 says the weight plan is process-scoped on
  `unified_cache`.
- Still unbacked or partial: weight priority order benchmark is deferred;
  single-shape sizing is only validated on Mistral-dense + GPT-OSS-MoE, not
  SWA/state-space; multi-device CPY visibility remains untested, though C2's
  actual keying claim is covered by op_id stability.
- Patched-runtime perf baseline and true multi-context init/concurrency are not
  fully revalidated. Not fatal for first PRs, but do not claim them as done.

## B. Bead Coverage

The open beads cover the design's full implementation chain, but the parent
epic's direct dependency list is incomplete/stale.

Actual Track A chain exists:

```text
9q4rn -> 6l35a -> pb9jk + 6b1ly -> dyeyy -> tzg5w -> 5ksb6 -> qm6gf -> 1mdps -> wuozk
```

So A3a/A3b/A7/C2 are tracked. A4 is tracked as `5ksb6`. C2 is `oib0o`.
A7 is `wuozk`. The main cleanup needed is bead graph hygiene: add the missing
Track A beads as direct epic dependencies or make the hierarchy explicit, and
mark `8gz7y` as superseded by A3a+A4 when those land.

## C. Track E Sequence

The proposed sequence is sane with one adjustment in wording:

1. Fix `khcc0 + w2ptt` together because both are arena reservation/overcommit
   symptoms.
2. Then land `zhzbp` as KV-clear cleanup once arena reservation is sane.
3. Then run Task 7 priority benchmark.
4. Handle `jfj0v` independently unless A3a production implementation needs
   mmap-backed no-alloc immediately.

I would not put `jfj0v` strictly after Task 7 as a dependency. It is Track A
production polish/blocker for the mmap fast path, not part of the Track E
inference-unblock chain.

## D. Smallest First Real PR

Smallest useful first PR: `llama.cpp-9q4rn` A1.

It lands real code, is low behavioral risk, and unblocks the whole Track A
chain: add `n_ctx`, `n_ubatch`, `n_seq_max`, and `flash_attn_type` to
`llama_model_params`, then thread/populate them through in-tree callers without
consuming them yet. It closes `9q4rn`.

## End Goal / System Model

The end goal is: before inference allocates anything substantial, SYCL has one
authoritative placement plan for every byte: weights, KV, infrastructure,
compute buffers, scratch, and per-op routing. If the plan says a model/context
fits, allocation should become deterministic and should not later discover an
unplanned 16 GB buffer or route an op based on stale weight heuristics.

The system is meant to work in two stages. At model load, a weight plan uses
caller-declared max context shape plus a skeleton/mini-context
`graph_reserve(no_alloc=true)` pass to size zones and choose
weight/KV/infrastructure placement. Weights are then loaded directly into final
arena offsets. At context creation, a compute plan runs the real graph reserve
for that context, allocates from the pre-reserved runtime/scratch zones, and
assigns every graph op a `plan.ops[op_id]` execution decision. Legacy shadow
heuristics and streaming paths are deleted once `plan.ops` is complete.
