# Planner Pre-flight Canaries — Aggregated Summary

Run date: 2026-04-24
Branch: `feature/sycl-coalescing`
Epic: `llama.cpp-3h5gm` (Unified Memory Placement Planning)

## Per-canary results

| ID | Bead | Result | Gates | Notes |
|----|------|--------|-------|-------|
| D0.1 | `llama.cpp-wca8b` | INCONCLUSIVE | A3a (`llama.cpp-dyeyy`) | Canary wedges in `llama_model_load_from_file` → S1-PRELOAD → `staging_buffer_pool::acquire` (`common.hpp:1863`). Blocked on E1 (`llama.cpp-m09zb`). See [d0.1-skeleton-determinism.md](d0.1-skeleton-determinism.md). |
| D0.2 | `llama.cpp-ge7rc` | PASS | A3a (`llama.cpp-dyeyy`) | PP and TG produce identical op sets on both Mistral 7B (13 ops) and GPT-OSS 20B (17 ops, MoE adds `ADD_ID`, `ARGSORT`, `MUL_MAT_ID`, `SOFT_MAX`). Single-shape reserve suffices. See [d0.2-pp-tg-union.md](d0.2-pp-tg-union.md). |
| D0.3 | `llama.cpp-5binh` | INCONCLUSIVE | C2 (`llama.cpp-oib0o`) | Only 1 SYCL device visible on this host (B50 disabled per `feedback_disable_b50.md`). Cross-device CPY scenario cannot be exercised. Rerun required once ≥2 devices are safely usable AND E1 lands. See [d0.3-cpy-visibility.md](d0.3-cpy-visibility.md). |
| D0.4 | `llama.cpp-zpp9k` | INCONCLUSIVE | A7 (`llama.cpp-wuozk`) | Canary uses only `ggml-backend` APIs (no llama, no unified_cache, no staging pool). Runtime wedges on the first `ggml_backend_tensor_set` at `ggml-sycl.cpp:12685` with the same L0 hang signature as m09zb — proves m09zb is an L0 DirectSubmission non-flush, not a staging-pool-specific bug. Blocked on E1. See [d0.4-direct-load.md](d0.4-direct-load.md). |

## Design-doc updates required

- **D0.2 (PASS)**: validated. A3a can size zones from a single shape
  (ubatch=max OR ubatch=1; they produce identical op sets on every
  tested model). The design doc's §D16 "PP + TG graph union sizing"
  does not need to be re-derived for Mistral 7B or GPT-OSS 20B — but
  the canary's value is now as an ongoing invariant check if a future
  model adds shape-dependent ops. No design change required.

- **D0.1 / D0.4 (INCONCLUSIVE)**: blocked on the L0 DirectSubmission
  non-flush captured as bead `llama.cpp-m09zb` and tracked as plan-doc
  Track E task E1. Design changes made outside this task's commit
  scope but directly triggered by these canaries: (a) Track E added
  with task E1 as a hard prerequisite to all other Tracks; (b)
  Migration plan now starts with Phase 0 (E1) before the original
  Phase 1; (c) Known-issues section documents both callsites (Example
  A preload, Example B tensor_set) and the three candidate fix
  directions. Landed across commits `ebbaee052`, `46f4a225f`,
  `fd2b016b7`, `eac575ab7`, `d9a193e80`, and the D0.1 canary lean-out
  `c791d9e27`.

- **D0.3 (INCONCLUSIVE)**: no design change required. Multi-device
  policy (§D13) stands as authored; re-validation depends on host
  policy, not on the canary or the plan.

## Track A unlock decisions

- **A3a** (`llama.cpp-dyeyy`): **partially unlocked**. D0.2 confirmed
  single-shape sizing is sound; A3a can proceed on the sizing
  question. The mini-context/`graph_reserve` mechanics themselves (D0.1
  gate) remain unverified pending E1. Recommend treating A3a as
  "design-blocked-on-E1, design-unblocked-on-sizing" until E1 lands
  and D0.1 can be re-run.
- **C2** (`llama.cpp-oib0o`): **BLOCKED**. D0.3 INCONCLUSIVE —
  multi-device CPY behavior unverified on this host; rerun when a
  second safely-usable SYCL device is available AND E1 lands.
- **A7** (`llama.cpp-wuozk`): **BLOCKED**. D0.4 INCONCLUSIVE — direct
  `ggml_backend_tensor_set` path wedges today; must wait on E1.

## Open follow-ups

- **E1** (`llama.cpp-m09zb`, P0): restructure `staging_buffer_pool` +
  address broader L0 flush path. Acceptance includes both D0.1 AND
  D0.4 canaries completing cleanly. Blocking follow-up.
- **D0.3 host capacity**: once B50 can be re-enabled without tripping
  the PM-underflow cascade (`feedback_disable_b50.md`), rerun D0.3 as
  a 2-device scenario. Secondary follow-up.
- **D0.1 canary structure decision**: current source uses sequential
  create-destroy-recreate (avoiding the earlier-hypothesized
  multi-context crash). The `D0_1_PROBE_MULTICONTEXT=1` guarded probe
  should be re-exercised post-E1 to confirm whether a second real bug
  exists at that level, or whether E1 also resolves it.
- **Multi-shape observation**: D0.2's finding that PP and TG produce
  identical op sets suggests §D16's "double-reserve" scheme may be a
  future-proofing measure rather than an active requirement. If E1
  lands and no model ever exercises shape-differential ops, consider
  simplifying §D16 to single-shape sizing with a validator rather
  than double-reserve.
