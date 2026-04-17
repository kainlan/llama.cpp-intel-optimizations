# MXFP4 Parity Epic — T6 Final Summary

**Date:** 2026-04-17 · **Epic:** `llama.cpp-tlcjr` · **Branch:** `feature/sycl-coalescing`

## Summary

Delivered everything within SYCL-backend scope. Host-resident MXFP4 GPU wedge eliminated. One architectural hygiene fix added. End-to-end coherent GPT-OSS 20B output blocked on `llama.cpp-4oi3i`, which turns out to be a **ggml-core scheduler buffer-aliasing issue** outside this epic's scope.

## Epic deliverables

| Task | Commit | Summary |
|------|--------|---------|
| T1 | `e2a5f8482` → `033859b80` | Vectorized per-block MXFP4 dequant helper (Track A) |
| T2 | (Track A, separate thread) | MXFP4 block-vectorized SLM-load integration |
| T3 | `4ddfdbb6e` → `7500793f4` → `fdf0d9629` | MXFP4 CPU-dispatch audit |
| T4 | `40d4c7759` → `ec7f04ac4` | Dispatch gates routing host-resident MXFP4 to CPU |
| T5 | `512d7ee72` | GPT-OSS 20B wedge resolution doc |
| W1 (bonus) | `9a81d14d9` | SYCL `graph_compute` queue drain at exit |

## W1 bonus hygiene

`ggml_backend_sycl_graph_compute` previously returned `GGML_STATUS_SUCCESS` while the GPU queue was still running. The ggml backend scheduler's split-input sync at `ggml-backend.cpp:1750-1763` synchronizes the CONSUMER backend only — not the PRODUCER. This violated the scheduler's implicit contract. Commit `9a81d14d9` adds `ctx.stream()->wait()` at graph_compute exit. Mistral 7B Q4_0 perf: PP512 -0.05%, TG128 -0.20%. No regression, closes a latent class of async-ordering bugs.

## `llama.cpp-4oi3i` investigation trail

T5 Step 1 repro: GPT-OSS 20B via `llama-bench -p 64 -n 32 -r 1` at `VRAM_BUDGET_PCT=30` crashes at `ops.cpp:743` CPU ADD_ID assertion `GGML_ASSERT(i11 >= 0 && i11 < ne11)`.

### Attempts

**(X)** — Remove ADD_ID from `ggml_sycl_op_is_planned_on_host`. Tried; produced GPU CAT fault + `guc_id=2` process-attributed wedge. **REVERTED.** Graph fragmentation across backends violates the `must-match-MUL_MAT_ID-backend` invariant at `ggml-sycl.cpp:50894`.

**(Z) / (Z-prime)** — SYCL-side self-staging handler. **SKIPPED** after diagnostics ruled out scheduler-sync as the root cause; self-staging would duplicate existing sync.

**(W1)** — Drain SYCL queue at `graph_compute` exit. **COMMITTED** as a correct-but-orthogonal hygiene fix. Did NOT address the ADD_ID crash (since the scheduler's `ggml_backend_synchronize(input_backend)` call at `ggml-backend.cpp:1882` already drains the queue before cross-backend copy).

### Diagnostic progression

**Diag 2** — Logged SYCL `graph_compute` node inventory. `ffn_moe_argsort-0` (op=ARGSORT) and `ffn_moe_topk-0` (op=VIEW of argsort) ARE in SYCL's graph. The consumer ADD_ID is in a separate CPU split. Scheduler's `ggml_backend_synchronize(input_backend)` in the cross-backend copy fallback path (ggml-backend.cpp:1876-1891) already drains SYCL before copying. Ruled out "missing producer sync."

**Diag 3b/4** — Dumped `src2->data` at ADD_ID entry across 3 consecutive calls (ffn_moe_gate_biased-0, ffn_moe_up_biased-0, ffn_moe_down_biased-0) in layer 0. All three ADD_IDs referenced the SAME src2 pointer (e.g. `0x7d0835d237c0`). Calls 0 and 1 contained valid ids (e.g. `[6, 27, 3, 0]` for every row). **Call 2 contained garbage float bit-patterns** (e.g. `1008464189`, `-1134781554`) — `ne11=32`, garbage exceeded bounds. Confirmed **buffer aliasing between consumers** of the same staging slot.

**Diag 5 (BUF-WATCH)** — Seeded a watch window from call 0's src2 pointer+size and instrumented `ggml_compute_forward` in `ggml-cpu.c` to flag any CPU op whose dst overlaps the window. **Zero BUF-WATCH entries fired.** Conclusion: the corrupter is NOT a CPU op going through the main dispatcher.

**Diag 6 (ALIAS walker)** — Captured the current cgraph in `ggml_backend_cpu_graph_compute` entry and walked it at ADD_ID call 2 looking for any tensor `->data` in the watch window. Result: `n_nodes=1`. **The scheduler submits each CPU ADD_ID as its own 1-node graph** — so the walker never had visibility into the SYCL-side or other CPU ops that might have written to the aliased buffer.

### Final root-cause characterization

ggml-core `ggml_gallocr` reuses the `CPU#ffn_moe_topk-0#0` cross-backend staging buffer slot for a later tensor — most likely a SYCL-computed float intermediate in the MoE gate→up→down compute chain — because galloc's liveness analysis determined the ids tensor was no longer needed after ADD_ID call 1. The `tensor->data` pointer stayed the same (staging slot's storage), but the backing bytes were overwritten by a different tensor's GPU kernel output (SYCL_Host host-pinned USM is GPU-writable).

**Fix requires ONE of**:
1. ggml-core scheduler change: prevent galloc from aliasing cross-backend staging buffers with other tensors until all their original consumers have executed.
2. llama-graph change: mark the argsort/topk output as multi-consumer explicitly (e.g., via an extra use/reference), extending its liveness.
3. ggml-graph-builder change: emit a `ggml_cont` or explicit persistent-buffer declaration for the ids tensor.

All three paths touch ggml-core or llama.cpp graph construction. **None are in SYCL-backend scope.**

## T6 closeout

- **Wedge eliminated**: GPU no longer hangs on host-resident MXFP4 (T4 gate fixes + T5 documentation).
- **Kernel vectorized**: MXFP4 per-block dequant helper + SLM-load integration (T1 + T2).
- **Architectural hygiene**: `graph_compute` queue drain (W1) closes a latent async-ordering class.
- **CPU dispatch audit**: full MXFP4 coverage map with 4 gaps enumerated + fixed (T3 + T4).
- **Coherent E2E output on GPT-OSS 20B**: BLOCKED. Requires ggml-core or llama-graph fix.

The `llama.cpp-4oi3i` bead retains the full investigation trail (X, Z, W1, Diag 2-6) in its notes field for handoff to whoever picks up the scheduler-aliasing fix.

## Handoff recommendation

Close `llama.cpp-tlcjr` (MXFP4 parity epic) at its SYCL-scope deliverables. Re-externalize `llama.cpp-4oi3i` as a standalone ggml-core task with the investigation trail already captured. Future work on 4oi3i should start from `Diag 6` and investigate ggml-core galloc liveness analysis for cross-backend staging tensors, or llama-graph construction of MoE routing.

## Distinct-trace catalog carried forward

Per T5's memory-file update, future sessions have the `guc_id=6` vs `guc_id=0` trace distinction documented in `~/.claude/projects/-Apps-llama-cpp/memory/feedback_no_20b_host_moe.md`. This investigation produced zero new wedges — xe driver auto-recovery worked throughout.

## Files preserved during investigation

- `/tmp/diag{2,3,4,5,6,10,11,12}.err` — diagnostic run logs.
- `/tmp/dmesg-{pre,post}-*.log` — kernel state snapshots.
- W1 benchmarks: `/tmp/w1-bench.out`, `/tmp/w1-bench.err`.
- Canary results for post-attempt GPU health: `/tmp/recanary.out`.

---

**Next session handoff:** nothing in the working tree beyond committed work. W1 is HEAD. All diagnostic instrumentation reverted. GPU healthy. Mistral Q4_0 canonical gate passes.
