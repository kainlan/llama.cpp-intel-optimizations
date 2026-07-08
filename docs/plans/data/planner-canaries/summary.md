# Planner Pre-flight Canaries — Aggregated Summary

Run date: 2026-04-24
Branch: `feature/sycl-coalescing`
Epic: `llama.cpp-3h5gm` (Unified Memory Placement Planning)

## Per-canary results

| ID | Bead | Result | Gates | Notes |
|----|------|--------|-------|-------|
| D0.1 | `llama.cpp-wca8b` + superseding `llama.cpp-2t09r` | PASS (2026-04-25 rerun) | A3a (`llama.cpp-dyeyy`) | The stale `blocked_on_m09zb` placeholder was replaced by a real orchestrator over `test-mini-context-prototype`. Mistral 7B dense and GPT-OSS 20B active SWA+MoE both returned exit 0 for real-A / real-B / mini reserve comparison. See [d0.1-skeleton-determinism.md](d0.1-skeleton-determinism.md). |
| D0.2 | `llama.cpp-ge7rc` | PASS for tested families; STATE-SPACE OPEN | A3a (`llama.cpp-dyeyy`) | PP and TG produce identical op sets on Mistral dense/full-attention and GPT-OSS active SWA+MoE. State-space remains open because no local fixture exists and SYCL lacks `GGML_OP_SSM_SCAN` support/routing even though Mamba/Plamo2 graphs emit `ggml_ssm_scan`. See [d0.2-pp-tg-union.md](d0.2-pp-tg-union.md) and [d0.2-architecture-coverage-2026-04-25.md](d0.2-architecture-coverage-2026-04-25.md). |
| D0.3 | `llama.cpp-5binh` + `llama.cpp-bkvc9` | SINGLE-DEVICE PASS / MULTI-DEVICE DESIGN CORRECTION | C2 (`llama.cpp-oib0o`) | CPU op-id stability is validated. D0.3a synthetic two-GPU scheduler proof records stable scheduler copy-edge tensors but `0` `GGML_OP_CPY` nodes, so C2 must plan scheduler split-input copy edges instead of synthetic CPY graph nodes. See [d0.3-cpy-visibility.md](d0.3-cpy-visibility.md) and [d0.3a-multidevice-cpy-visibility.md](d0.3a-multidevice-cpy-visibility.md). |
| D0.4 | `llama.cpp-zpp9k` | PASS (2026-04-24 patched runtime) | A7 (`llama.cpp-wuozk`) | Canary uses only `ggml-backend` APIs (no llama, no unified_cache, no staging pool). Under patched libze 1.14.37435 it completes with byte-identical readback in `tensor_set_us=282422`, validating direct-load mechanics. See [d0.4-direct-load.md](d0.4-direct-load.md). |

## Design-doc updates required

- **D0.2 (PASS)**: validated for the local dense/full-attention and
  active SWA + MoE families. A3a can size zones from a single shape
  (ubatch=max OR ubatch=1; they produce identical op sets on every
  tested model). The design doc's §D16 "PP + TG graph union sizing"
  does not need to be re-derived for Mistral 7B or GPT-OSS 20B — but
  the canary's value is now as an ongoing invariant check if a future
  model adds shape-dependent ops. State-space remains unvalidated until
  a Mamba/RWKV/SSM GGUF fixture exists locally.

- **D0.1 (PASS) / D0.4 (PASS under patched runtime)**: E1's original
  L0 DirectSubmission non-flush blocker is resolved by the patched
  compute-runtime. D0.1 was rerun on 2026-04-25 using the real/mini
  protocol and now backs A3a for the tested dense and active SWA+MoE
  fixtures. D0.4 completed under patched libze and backs A7 direct-load
  mechanics.

- **D0.3 (PARTIAL / CORRECTED)**: C2's single-device op-id keying is
  validated. D0.3a proves multi-device scheduler transfers are stable
  copy-edge tensors (`SYCL1#...#0`) with `GGML_OP_NONE`, not
  `GGML_OP_CPY` graph nodes. C2 must consume scheduler transfer-edge
  metadata.

## Track A unlock decisions

- **A3a** (`llama.cpp-dyeyy`): **unlocked for tested families**. D0.1
  now runs the real/mini reserve protocol, and D0.2 confirmed
  single-shape sizing is sound for Mistral dense/full-attention and
  GPT-OSS active SWA+MoE. State-space remains outside this unlock until
  `GGML_OP_SSM_SCAN` support and a fixture exist.
- **C2** (`llama.cpp-oib0o`): **PARTIAL / CONTRACT CHANGE**.
  Single-device op-id keying is valid. Multi-device transfers must be
  represented from scheduler copy-edge metadata, not CPY nodes; follow-up:
  `llama.cpp-bkvc9`.
- **A7** (`llama.cpp-wuozk`): **VALIDATED**. D0.4 PASS under patched
  runtime proves direct `ggml_backend_tensor_set` load mechanics.

## Open follow-ups

- **E1** (`llama.cpp-m09zb`, P0): resolved by patched compute-runtime;
  D0.1 and D0.4 now have current completion evidence.
- **D0.3a real multi-device canary** (`llama.cpp-bkvc9`): harness now
  proves a design correction: stable scheduler copy-edge tensors exist,
  but no `GGML_OP_CPY` nodes are emitted for this cross-backend transfer
  path. Update C2 and close this with the corrected contract.
- **Multi-shape observation**: D0.2's finding that PP and TG produce
  identical op sets suggests §D16's "double-reserve" scheme may be a
  future-proofing measure rather than an active requirement. If E1
  lands and no model ever exercises shape-differential ops, consider
  simplifying §D16 to single-shape sizing with a validator rather
  than double-reserve.
