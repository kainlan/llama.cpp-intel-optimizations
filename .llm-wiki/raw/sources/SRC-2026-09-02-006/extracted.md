# The second card earns its keep -- dual-device PP and decode over a host bounce

- **Slug:** second-card-earns-its-keep
- **Title:** The second card earns its keep -- dual-device PP and decode over a host bounce
- **Epic ticket:** llama.cpp-po3nd.2
- **Priority:** 0
- **Description:** ## Goal

B70 (`level_zero:0`, 256 CU, 32.6 GB, 608 GB/s) plus B50 (`level_zero:1`, 128 CU, 16 GB, 224 GB/s) together, on GPT-OSS 20B, must beat the best single card on both prompt processing and decode — with margin larger than run noise, no correctness loss, and no code path that assumes PCIe peer DMA (there isn't any between these two cards). This also covers per-device VRAM budget isolation, cross-device KV/activation routing, and a standing regression matrix that gates any default-behavior change.

## Why now

The 2026-08-26 merge of upstream b10630 landed backend-agnostic tensor parallelism (`--split-mode tensor`, PR #19378, the "meta backend") with an upstream SYCL comm implementation described in PR #24152 (`comm_init/comm_free/comm_allreduce_tensor`). **Correction (triage 2026-09-01): that SYCL comm layer is NOT in this tree.** `comm_allreduce_tensor` appears only in `ggml/include/ggml-backend.h:226` and the generic meta backend (`ggml/src/ggml-backend-meta.cpp:1792-1823`); `ggml-sycl.cpp`'s `get_proc_address` (`:103801-103826`) exports no `ggml_backend_comm_init`/`comm_free`/`comm_allreduce_tensor`, so on this backend today the meta backend gets `comm_ctx=nullptr`. The `g_sycl_tp_config`/`world_size` scaffolding already live in `ggml-sycl.cpp` (e.g. lines 16764, 21476, 24068-24128, 34737-34836) is FORK-LOCAL code introduced by `1352fb4fc` (2025-12-07), predating the b10630 merge — a same-purpose fork mechanism, not upstream's PR #24152 SYCL comm implementation (confirmed via `git log -S g_sycl_tp_config`). This still leaves a second, upstream-maintained *framework* for "both GPUs busy on one request" (PR #19378's meta backend) that did not exist when most of these tickets were filed (2026-03 through 2026-06) and is designed around exactly this fork's constraint — no P2P, host-bounced all-reduce — but exercising it on this backend requires porting in the missing SYCL comm implementation first, not merely flipping a flag. Before more months go into the fork's hand-rolled per-expert secondary-device split (which a 2026-06-01 executor validation already called "proven unsafe" for per-op cross-device routing), this epic needs to weigh porting upstream's TP mechanism against the fork's own contiguous-layer-block design and stop paying twice for the same goal.

## Current state (file:line refs)

- **Contiguous layer-block placement is DONE.** `ggml/src/ggml-sycl/unified-cache.cpp:24998` `populate_multi_device_layer_blocks`, `:25192` `populate_no_p2p_candidate_layer_blocks`, `:25974` `use_cohesive_no_p2p_moe` / `[PLACEMENT-BLOCK]` logging — landed at `f4b97d608` ("sycl: record dense-fastest layer blocks"), the sha `llama.cpp-po3nd.2.45`'s own comments cite. Dense-fastest scoring (`dense_score`) and a hardware-derived multi-device cost model (`compute_multi_device_plan`, `device_budget`, `moe_device_scores`, `moe_transfer_penalty`, `unified-cache.cpp` ~22160-26700) both exist.
- **The pipelined block executor does NOT exist.** `ggml/src/ggml-sycl/ggml-sycl.cpp:85363` computes `stats.safe_for_block_executor` and dry-run readiness probes (boundary staging, output allocation, leaf materialization) but every log line ends `executor=inactive` (`:85461`). This is `llama.cpp-po3nd.2.46`'s open scope — it is the carrying task of this epic, exactly as the taxonomy's `ordering_hint` says.
- **The 2026-05/06 per-expert secondary-device split direction is largely abandoned in favor of the block design.** The 2026-06-01 notes on `po3nd.2.45`/`.46` record that per-op cross-device routing DEVICE_LOSTs and was reverted; "per-op context switching is proven unsafe." `try_mixed_moe_layer_pair`/`try_secondary_moe_layer_pair` (the lambdas `.33`/`.36`/`.37` were diagnosing) are gone from the tree (deleted in `abecb785d`, confirmed by `llama.cpp-mk4l`'s own investigation). What remains live is a narrower secondary-device MoE route for decode (`ggml_sycl_try_decode_secondary_moe_route_on_device`, `ggml-sycl.cpp:25526`, called from `:25656`) and a PP-specific secondary dispatcher (`dispatch_experts_secondary_gpu_impl`, `:65320`, called from `:71965` and `:76324`) — both still compiled and reachable, used for capacity-spill rather than load-sharing replication.
- **Telemetry for route ownership is already in place.** Per-layer/role counters (`planned_device_to_host_rows`, `planned_host_rows`, `true_host_rows`, `ggml-sycl.cpp` ~19266-19268) and `moe_layer_grouped_route_view` (`moe-layer-plan.hpp:647-661`, reporting `device_rows`/`host_rows`/`missing_rows`/`layout_mismatch_rows`/`n_groups`) satisfy what `.11` and `.22` asked for.
- **Upstream tensor parallelism's framework (PR #19378, meta backend) is present in-tree post-merge, but its SYCL comm implementation (PR #24152) is NOT** (see Why now, corrected 2026-09-01) — the SYCL backend exports no `comm_*` proc addresses, so `--split-mode tensor` on this backend today gets `comm_ctx=nullptr`. Not yet exercised or measured against GPT-OSS on this hardware; no member task currently owns evaluating it, and evaluating it now requires porting the missing SYCL comm layer first, not just passing the flag.
- `docs/backend/sycl-perf-baselines.md` has single-device B70/B50 rows and an explicit "SUPERSEDED — B580 removed" note, but **no dual-GPU section** — nothing to gate a "beats best single card" claim against today.
- `GGML_SYCL_MOE_MULTI_GPU` remains the opt-in gate on the fork's multi-device MoE dispatch (6-7 call sites, e.g. `ggml-sycl.cpp:7843, 15813, 22118, 101650`), unchanged from ruling 9's list of opt-in-only flags.

## Design constraints

- **Ruling (constraints, this epic):** "No PCIe P2P between 0000:03:00.0 and 0000:07:00.0 ... `ext_oneapi_enable_peer_access` returning OK is NOT a capability check... All cross-device traffic host-bounces." Both the block design and upstream TP's all-reduce already respect this; any candidate design that assumes fast shared/peer memory is disqualified on inspection, not by benchmark.
- **Ruling 4 (placement decides the executor):** the planner places data where it fits/runs best; ops run where the data already is. The block executor (`.46`) must dispatch each block on its owning device, not stage weights per-op.
- **Ruling 5 (one layout per weight per device):** a device-resident block's weights are materialized once in that device's optimal layout; a boundary tensor crossing devices is a deliberate host-bounce copy, not a second live layout of the same weight.
- **Ruling 6 (no host waits):** boundary staging and merge must be SYCL event chains (`depends_on`), not queue drains — this is exactly what `.36`'s event-DAG requirement is about, scoped now to the live decode secondary route rather than the deleted lambdas.
- **Ruling 9 (correctness before throughput):** `GGML_SYCL_MOE_MULTI_GPU` stays opt-in until the epic's bar passes; the Mistral and GPT-OSS gates (CLAUDE.md) run on every claimed win.
- **Ruling 10 (device selection / VRAM budget):** `ONEAPI_DEVICE_SELECTOR` only, no code parsing it; per-device budget must not treat the iGPU's host-unified memory as VRAM (already fixed elsewhere in the fork, `llama.cpp-403s`) — `.03k1`/`.fawh` re-verify this holds under two discrete cards.
- Every B580-era number in these tickets (`>1100 pp512`/`~50 tg128` B50 guardrail, `>2000`/`>85` B580 guardrail, `~15.7%`/`~26%` cache-coverage math, `8-12`/`>15 tok/s` 120B targets) is **obsolete-hw** per the design brief and must be re-derived from `docs/backend/sycl-perf-baselines.md`'s current B70/B50 rows, never quoted as a live gate.

## External / upstream status

- **Backend-agnostic tensor parallelism merged upstream, April 2026** (PR #19378, JohannesGaessler) — a "meta backend" enabling `--split-mode tensor`, requiring flash-attention (auto-enabled), currently tuned for 2 GPUs. *Impact: makes several of this epic's from-scratch cross-device mechanisms (device role auto-detection, boundary-tensor event chains, all-reduce) potentially available as upstream code to port/adapt instead of hand-build; does not by itself solve the layer-block PP/decode pipelining goal, which is a different mechanism (row/column split + all-reduce vs. contiguous block ownership).*
- **SYCL implementation of TP landed** (PR #24152, "Sycl --split-mode tensor") — `comm_init/comm_free/comm_allreduce_tensor`, small tensors (`nelem < 32768`) via FP32 direct memcpy + per-device ADD kernels, large tensors via BF16-compressed cross-device memcpy (halves PCIe bytes) for the all-reduce, mirroring `ggml-cuda.cu`'s pattern. *Impact (corrected 2026-09-01): `g_sycl_tp_config`/`world_size` (`ggml-sycl.cpp` ~16764-34836) is FORK-LOCAL scaffolding predating this PR (introduced `1352fb4fc`, 2025-12-07) — a same-purpose mechanism, not this PR's code; `git log -S g_sycl_tp_config` bottoms out before the b10630 merge. PR #24152's actual `comm_init/comm_free/comm_allreduce_tensor` SYCL implementation is NOT in this tree — `get_proc_address` exports no `ggml_backend_comm_*` symbols. `llama.cpp-dt06`'s bug (TP debug sync swallows a kernel failure, ~62171-62181) is real, but it is a defect in the fork's own pre-existing TP scaffolding, not in upstream's landed comm layer. Porting PR #24152's SYCL comm implementation in is the actual port-instead-of-build opportunity — it does not yet exist compiled in.*
- **Reported dual-B70 result on a comparable topology (corrected 2026-09-01 — this is NOT an independent/community benchmark):** Llama-3.3-70B, tensor-split mode pp512 = 377.08 tok/s vs layer-split mode 313.65 (+20.2%), tg128 = 17.40 vs 9.74 (+78.6%). These figures come from PR #24152's own PR description (`https://github.com/ggml-org/llama.cpp/pull/24152`), not from the spheron.network blog previously cited here — that page (`https://www.spheron.network/blog/deploy-llama-cpp-server-gpu-cloud/`) does not contain any dual Arc Pro B70 / Llama-3.3-70B tensor-split-vs-layer-split comparison (verified by fetch: no such benchmark or throughput metrics present). *Impact: this is the PR author's own self-reported number on the exact card pairing this fork runs, for a dense model — a directional reference point only, unverified until independently reproduced on this fork's build, for `.46`'s "beat all-on-one" bar; it is a 70B dense model, not GPT-OSS MXFP4 MoE.*
- **Upstream TP + MoE (GPT-OSS/Qwen3-MoE) support exists but is architecture-limited** — many MoE/hybrid architectures still hit "LLAMA_SPLIT_MODE_TENSOR not implemented for architecture"; GPT-OSS is one of the supported ones, and GPT-OSS-120B-MXFP4 with >2 GPUs no longer crashes (previously did). A 4×RTX4090 measurement reports 1.53x PP speedup for GPT-OSS 120B MXFP4 via the new meta-backend TP path. *Impact: GPT-OSS, this epic's target model, is in the supported set upstream — evaluating `--split-mode tensor` against this epic's bar (rather than assuming it must be hand-built) is now a cheap, high-value first experiment.*
- **Intel compute-runtime / Level Zero**: no new evidence found of P2P becoming available between non-bridged Battlemage cards (searches turned up general Level Zero P2P query API docs, nothing specific to unlocking this topology); the fork's no-P2P finding stands unchanged and un-contradicted.
- **Open Intel compute-runtime issue** (`intel/compute-runtime#922`): a multi-rank MPI + Level Zero abort on Xe2/BMG-G31 after a compute-runtime upgrade (26.05→26.14) on the xe driver — not this fork's exact driver version (26.31) but a signal that multi-device Level Zero on Battlemage is still an active bug surface upstream; worth a quick compatibility check before leaning harder on TP.

## Work breakdown (prerequisites first)

1. **llama.cpp-po3nd.2.47** — *(close, done)* B50 enumeration/reset-storm recovery — resolved by driver 26.31 + selector pinning; no longer a blocker.
2. **llama.cpp-po3nd.2.11 / .22** — *(close, done)* per-device route telemetry — already landed, available for every step below.
3. **llama.cpp-03k1 / llama.cpp-fawh** — per-device VRAM budget correctness under two discrete cards (ruling 10) — prerequisite for any placement decision.
4. **llama.cpp-kkxtv / llama.cpp-kkxtv.7** — cross-device KV and activation routing correctness (root KV on a non-default device resolves through smart handles) — prerequisite for both the block executor and TP.
5. **llama.cpp-l80a3** — device-role auto-detection from queried capability (dense_score/budget/XMX), feeding the block planner's device choice.
6. **NEW (see new_tasks)** — port and evaluate upstream `--split-mode tensor` on GPT-OSS 20B B70+B50 as a same-build alternative to the block design (its SYCL comm layer is not yet in this tree, see Why now); this determines whether `.46`/`.42`/`.44`/`.3`/`.36` proceed as fork-native work, a port of the upstream all-reduce, or both.
7. **llama.cpp-po3nd.2.46** — the pipelined block executor (PP microbatch pipelining + decode across `.45`'s already-placed blocks) — the epic's carrying task; `executor=inactive` today.
8. **llama.cpp-po3nd.2.3 / .42 / .44** — PP- and decode-side consumers of planner-owned secondary MoE work (capacity-spill via the still-live `dispatch_experts_secondary_gpu_impl`/`ggml_sycl_try_decode_secondary_moe_route_on_device`), re-measured against current B70+B50 baselines.
9. **llama.cpp-po3nd.2.36** — event-chain the live secondary decode MoE route per ruling 6 (re-targeted off the deleted lambdas onto the current function).
10. **llama.cpp-mk4l** — close out the investigation: confirm (with the evidence this design pass already gathered) that the deleted mixed/secondary lambdas add nothing beyond the current planner-owned residency + block design, or scope the real gap.
11. **llama.cpp-won9** — verify secondary-device planned expert placement end to end once `.46` lands.
12. **llama.cpp-t00wi** — sharded 120B GGUF loading across devices (needed before any 120B validation).
13. **llama.cpp-bci0** — planned secondary-device expert placement for MoE dispatch, generalized (not B50-specific), after the block/TP decision is made.
14. **llama.cpp-vudft** — 120B multi-GPU TG re-baseline on B70+B50 (absorbs `w2zf`'s intent).
15. **llama.cpp-d8m7** — production validation run, lead-only, clean reboot + reservation protocol (absorbs `xzeu`'s T5 verification scope).
16. **llama.cpp-po3nd.2.28** — standing regression/benchmark matrix (absorbs `po3nd.2.7`) — the closing gate every other item reports into.
17. **llama.cpp-dt06** — fix the TP debug-sync swallow bug in the now-live `g_sycl_tp_config` path (low priority in isolation, but directly touches whichever mechanism this epic ends up shipping if TP is adopted).
18. **llama.cpp-po3nd.2** — the pre-existing umbrella epic; keep open as the roll-up ticket, retarget its own acceptance criteria to this document rather than its dead B580/B50 numbers.

## Bar / closing gate

Same-build, interleaved paired A/B, GPT-OSS 20B MXFP4, FA-on, `-p 512 -n 128`, across three configurations:

```bash
source /opt/intel/oneapi/setvars.sh --force
ONEAPI_DEVICE_SELECTOR=level_zero:0   ./build/bin/llama-bench -m /models/gpt-oss-20b-mxfp4.gguf -p 512 -n 128   # B70-only
ONEAPI_DEVICE_SELECTOR=level_zero:1   ./build/bin/llama-bench -m /models/gpt-oss-20b-mxfp4.gguf -p 512 -n 128   # B50-only
ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/llama-bench -m /models/gpt-oss-20b-mxfp4.gguf -p 512 -n 128   # dual
```

Expected/required: dual pp512 > max(~1415, ~894) and dual tg128 > max(~44, ~32) (current B70/B50 baselines, `docs/backend/sycl-perf-baselines.md`) by a margin bigger than B70's measured tg noise (cv 3.3%, ±10% single-run). Then, without touching the binary:

```bash
# Mistral single-device guard, both cards, unregressed vs baseline table
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench -m /models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench -m /models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128

# Correctness gates, dual selector
ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/llama-completion -m /models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0   # expect: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10

ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/llama-cli -m /models/gpt-oss-20b-mxfp4.gguf -ngl 99 \
  -c 4096 -st --simple-io --no-display-prompt \
  --chat-template-kwargs '{"reasoning_effort":"medium"}' --reasoning-format none --reasoning-budget 0 \
  -p 'Count from 1 to 5. Answer with only: 1, 2, 3, 4, 5' -n 48 --seed 42 --temp 0   # expect: 1, 2, 3, 4, 5
```

Closing gate: `docs/backend/sycl-perf-baselines.md` carries a new committed dual-GPU section with these numbers (owned by `.28`), `GGML_SYCL_MOE_MULTI_GPU` (or whatever the winning mechanism's flag is) stays opt-in until this passes, and 120B multi-GPU TG (`.vudft`, `.d8m7`) is re-baselined on B70+B50.

## Out of scope

- Anything requiring PCIe P2P DMA between the two discrete cards — structurally unavailable (kernel refuses it), not a software gap.
- Flipping any multi-GPU flag to default-on before the bar above passes (ruling 9).
- B580-specific numbers, topology, or hardware — the card is gone; do not re-derive history, only current B70+B50.
- Single-GPU dispatch/kernel work (unified-kernel, mmq/mmvq internals) unless it is specifically the secondary-device or boundary-copy path this epic owns.
- Non-SYCL backends and the RPC/multi-machine backend — out of scope for this hardware.

## Dependencies on other epics

- **llama.cpp-iiff** (zone-reset elimination / Option-C epoch-refcounted transient zones) — any staging/boundary buffer this epic's block executor or TP path allocates must land on that model, not a zone-reset fallback.
- **llama.cpp-017kb** (unify MoE expert memory under unified-cache) — `t00wi`'s sharded-120B loading fix and the secondary MoE dispatch paths both sit downstream of that epic's unified ownership model.
- **llama.cpp-omp4** (post-merge Q1/NVFP4 device decode program, deferred) — `mk4l` flags a scope-adjacency question (its dependents' names look like this epic's mixed/secondary executors) that needs an explicit owner call before any restoration work is scoped as `omp4`'s.
- **llama.cpp-30ak7.19** (B50 GPT-OSS TG restoration, external) — `po3nd.2.44` is blocked on it directly.

## Tasks

### llama.cpp-03k1

  - **Id:** llama.cpp-03k1
  - **Action:** keep
  - **Priority:** 1
### Depends on

  - llama.cpp-fawh

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Keep the multi-GPU VRAM budget calculation correct now that the second device is a discrete B50, not the contaminating factor it was against the old B580.

**Design:**
- Re-verify `min(total*pct, free_at_init)` (ruling 10) holds independently for level_zero:0 and level_zero:1 when both are visible.
- Confirm the iGPU (231.7 GB host-unified 'VRAM') stays excluded from any multi-device budget sum, not just single-device.
- Re-baseline the historical 2.42 vs 6.03 tok/s regression figure on current B70+B50 hardware; the old numbers are B580-era and not usable as a target.
- Keep this as the umbrella epic ticket (T1-T5 children); do not add new acceptance criteria beyond re-verification unless a live contamination bug is found.

**Bar:** Per-device budget log (`unified_cache_log_all_devices()` or equivalent) shows independent, correct budgets for B70 and B50 simultaneously visible; no budget inflation observed.

**Constraints:** Ruling 10 (VRAM budget formula, iGPU exclusion). No device-name branches.

**Acceptance test (corrected 2026-09-01 -- named target string):** `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/llama-bench -m /models/gpt-oss-20b-mxfp4.gguf -p 16 -n 4 -v 2>&1 | grep -A9 'Budget summary for device'` -- the concrete emitted line is `[UNIFIED-CACHE] Budget summary for device %d:` from `unified_cache_log_budget_summary()` (`unified-cache.cpp:14089`), called once per process via a `static bool budget_logged` guard at `ggml-sycl.cpp:100810`. Caveat: that guard means only ONE device's summary is guaranteed to print on a dual-device run, so this grep alone cannot confirm BOTH cards' budgets simultaneously -- true per-device-simultaneous verification is blocked on `llama.cpp-fawh` landing `unified_cache_log_all_devices()` (not yet landed; confirmed no such symbol exists). Until then, inspect the printed budget/available/committed MB fields against `min(total*pct, free_at_init)` for whichever device logs, and treat an EMPTY grep after `-v` as a real negative (the diagnostic did not print), not as ambiguous.

**Depends on:** llama.cpp-fawh

### llama.cpp-bci0

  - **Id:** llama.cpp-bci0
  - **Action:** keep
  - **Priority:** 1
### Depends on

  - llama.cpp-03k1
  - llama.cpp-fawh

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Let the planner place hot experts on the secondary device when the cost model says it pays for itself, using queried capability rather than card identity.

**Design:**
- Consume `unified-cache.cpp`'s existing `compute_multi_device_plan`/`moe_device_scores`/`moe_transfer_penalty` (landed as part of `.45`) instead of writing a new cost model.
- Route logs must show planned expert handles on the secondary device with no env-var required for correctness.
- Explicitly account for host-bounce cost (ruling: no P2P) in the placement decision, not just VRAM capacity.
- Do not resurrect per-token dynamic replication (that direction — `.20`/`.26`/`.24`/`.25` — is closed as superseded by the block design); this task is about model-load-time planned placement only.

**Bar:** Default dual-device model load places some expert or dense work on the secondary device per plan output (visible in `[PLACEMENT-BLOCK]`/`[MOE-LAYOUT]` logs), without `GGML_SYCL_MOE_MULTI_GPU` or any other env var required.

**Constraints:** Ruling 4 (placement decides executor), ruling 10 (device selection via ONEAPI_DEVICE_SELECTOR only). No B50/B70 name branches.

**Acceptance test:** `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/llama-completion -m /models/gpt-oss-20b-mxfp4.gguf -p '...' -n 4 -v 2>&1 | grep PLACEMENT-BLOCK` — confirm secondary-device entries in the plan.

**Depends on:** llama.cpp-03k1, llama.cpp-fawh

### llama.cpp-po3nd.2

  - **Id:** llama.cpp-po3nd.2
  - **Action:** keep
  - **Priority:** 0
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Keep this ticket as the standing roll-up epic for dual-device B70+B50 execution, but retarget its acceptance criteria to this document instead of the dead B580/B50 numbers already flagged as unreachable by its own 2026-07-25 lead comment.

**Design:**
- Rewrite title/acceptance criteria to reference `docs/backend/sycl-perf-baselines.md` by value, matching how `CLAUDE.md` itself was fixed.
- Its `depends_on` list carries child ids from before this triage; reconcile against the disposition table this epic ships (several children close as done/superseded).
- Preserve the epic-depends-on-child roll-up direction the ticket already documents; do not make it block on every child.
- The actual technical bar now lives in this document's Bar section, not in the ticket body's old 'first-principles target' math (which used B580 numbers).

**Bar:** The ticket's acceptance criteria field cites this document (or `docs/backend/sycl-perf-baselines.md`) instead of any inline number; `existing_epic_id` in the taxonomy already maps this ticket to the new organization.

**Constraints:** Design brief verdict vocabulary: against-design numbers must be rewritten, not silently kept. Ruling references throughout this doc.

**Acceptance test:** Manual: read the ticket's acceptance_criteria field after edit and confirm no B580 or '>1100 pp512' string remains (`task_show llama.cpp-po3nd.2 | grep -iE 'b580|1100'` returns nothing).

**Depends on:** none within this epic

  - **New title:** [EPIC] B70+B50 dual-device GPT-OSS PP/TG beats best single card

### llama.cpp-po3nd.2.28

  - **Id:** llama.cpp-po3nd.2.28
  - **Action:** keep
  - **Priority:** 0
### Depends on

  - llama.cpp-po3nd.2.46

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Maintain the one standing benchmark/regression matrix that every dual-device change reports into, replacing the B580-era acceptance criteria and absorbing `.7`'s duplicate scope.

**Design:**
- Six-way matrix: GPT-OSS 20B B70-only/B50-only/dual, FA on/off, PP512+TG128; Mistral B70/B50 single-device sanity FA on/off; GPT-OSS + Mistral correctness gates.
- Drop this ticket's dependency on `.26` (closed as superseded — no split-expert decode executor exists to gate against); depend on `.46` (the pipeline executor) instead.
- Record every run as same-build, one benchmark at a time (never overlapping GPU work — see CLAUDE.md Hard-Won Rules), with command line and env vars in the ticket comments.
- Own writing the new 'dual-GPU' section into `docs/backend/sycl-perf-baselines.md` — this is the closing gate's home per this epic's Bar section.

**Bar:** Every major dual-device change under this epic has a same-build before/after row in `docs/backend/sycl-perf-baselines.md`'s dual-GPU section; regressions get filed as dependent tickets with exact command lines.

**Constraints:** Ruling 9 (correctness before throughput — gates run alongside every perf claim). GPU work stays in the lead session only (CLAUDE.md).

**Acceptance test:** The Bar section's three `llama-bench` invocations plus the Mistral/GPT-OSS correctness gates, run by the lead, recorded in this ticket.

**Depends on:** llama.cpp-po3nd.2.46

### llama.cpp-po3nd.2.3

  - **Id:** llama.cpp-po3nd.2.3
  - **Action:** keep
  - **Priority:** 1
### Depends on

  - llama.cpp-po3nd.2.46

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Give prompt-processing rows a real path onto the secondary device instead of falling back to host, using the PP dispatcher that already exists and is called in production (`dispatch_experts_secondary_gpu_impl`).

**Design:**
- Start from a current capability census: which (layout, phase) combos still hit `reason=secondary-unsupported` vs. route through `dispatch_experts_secondary_gpu_impl`'s PP call sites (`ggml-sycl.cpp:71965`, `:76324`) today — the 2026-05 census this ticket describes is stale.
- Group PP rows by device/layout/expert/role exactly as the single-device path does; keep raw pointers as transient submit-time ABI only.
- Re-check the ticket's own non-goals (no B50/B70 branches, no fits-in-VRAM boolean) against current code for drift.
- Coordinate with the block executor (`.46`): once whole layer blocks are pipelined, this task's scope narrows to capacity-spill within a block's owning device set, not general PP row splitting.

**Bar:** GPT-OSS 20B dual PP512 shows nonzero secondary-device selected rows in route telemetry (already-landed counters, `moe-layer-plan.hpp:647-661`) with zero unplanned host fallback.

**Constraints:** Ruling 4/5 (placement decides executor, one layout per weight). No card-name branches, no fits-in-VRAM boolean (ticket's own non-goals).

**Acceptance test:** `GGML_SYCL_MOE_PROFILE=1 ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/llama-bench -m /models/gpt-oss-20b-mxfp4.gguf -p 512 -n 1 -v 2>&1 | grep -E 'host_rows|device_rows'` — host_rows should drop toward zero for supported layouts.

**Depends on:** llama.cpp-po3nd.2.46

### llama.cpp-po3nd.2.36

  - **Id:** llama.cpp-po3nd.2.36
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Event-chain the secondary decode MoE route so cross-device correctness does not depend on incidental in-order queue behavior — re-targeted at the function that still exists, not the deleted lambdas the original ticket names.

**Design:**
- Target `ggml_sycl_try_decode_secondary_moe_route_on_device` (`ggml-sycl.cpp:25526`), not `try_mixed_moe_layer_pair`/`try_secondary_moe_layer_pair` (deleted in `abecb785d`).
- `mmvq_submit_mxfp4_soa_batched` already returns a `sycl::event` (`mmvq.cpp:6930`, `mmvq.hpp:684`) — confirm the secondary route actually threads that event through `depends_on` rather than discarding it.
- Check whether ruling 6 (no host waits, landed 2026-08-20, after this ticket was filed) already forced this rewrite as a side effect; if so this closes as done with evidence, not as new work.
- Build the explicit DAG only for the route that ships (decode secondary + PP `dispatch_experts_secondary_gpu_impl`), not a speculative future mixed executor.

**Bar:** Focused secondary-route trace shows activation → quantize → gate/up → GLU → down → D2H all connected by returned events, zero implicit in-order-queue dependencies for cross-device data.

**Constraints:** Ruling 6 (no host waits — event-chain everything) is the direct source of this task's requirement.

**Acceptance test:** GPT-OSS dual count-to-5 gate passes (see epic Bar) plus a code read confirming `depends_on` chains through the secondary route's kernel submissions (no bare `.wait()` outside teardown/debug).

**Depends on:** none within this epic

### llama.cpp-po3nd.2.42

  - **Id:** llama.cpp-po3nd.2.42
  - **Action:** keep
  - **Priority:** 1
### Depends on

  - llama.cpp-po3nd.2.3
  - llama.cpp-po3nd.2.46

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Get GPT-OSS prompt processing to route selected MoE work to the secondary device by default, re-measured on current hardware, and reconciled against the possibility that upstream `--split-mode tensor` already does this more generally.

**Design:**
- Re-run the ticket's stale B580-era PP matrix on B70+B50 before writing new acceptance numbers — no current same-build dual PP baseline exists yet (per this epic's Bar section, gap to fill).
- Consume planner output from `.45`'s block placement plus `.3`'s PP secondary dispatcher; do not invent a third mechanism.
- Explicitly compare against `--split-mode tensor` (see epic's External/upstream status) on the same build — if TP already beats the fork's planner-owned secondary PP path, this ticket's remaining scope shrinks to whatever TP doesn't cover for MXFP4 MoE.
- Plan diagnostics must explain secondary work chosen (device, bytes, expected benefit), not just report a number.

**Bar:** Dual GPT-OSS PP512 beats max(B70-only, B50-only) pp512 by more than run noise (epic Bar), with route logs showing planner-owned secondary MoE execution, not host fallback.

**Constraints:** Ruling 4 (placement decides executor), ruling 9 (opt-in until bar passes).

**Acceptance test:** Epic Bar section's three-way `llama-bench` PP512 comparison plus GPT-OSS correctness gate.

**Depends on:** llama.cpp-po3nd.2.3, llama.cpp-po3nd.2.46

### llama.cpp-po3nd.2.44

  - **Id:** llama.cpp-po3nd.2.44
  - **Action:** keep
  - **Priority:** 1
### Depends on

  - llama.cpp-po3nd.2.46

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Prove true dual-device GPT-OSS decode beats the best single-card TG128 once the B50 TG restoration work it depends on lands.

**Design:**
- Blocked on `llama.cpp-30ak7.19` (external, open, B50 GPT-OSS TG restoration epic) — do not start measurement until that baseline is current.
- Re-baseline against `docs/backend/sycl-perf-baselines.md`'s current figures (B70 ~44 tg128, B50 ~32 tg128 GPT-OSS), not the ticket's historical '49-51 tok/s class' framing.
- Consume whichever decode mechanism wins between the block executor (`.46`) and upstream TP (see epic's External/upstream status) — do not build a third bespoke decode split.
- Keep dense/attention/KV serial on one device until a block or TP boundary proves overlap is worth the host-bounce cost.

**Bar:** Dual GPT-OSS TG128 FA-on beats max(current B70 tg128, current B50 tg128) by a margin bigger than B70's measured noise (cv 3.3%). Measure via ONE lead-run `llama-bench -r 3` invocation (a single process doing 3 repeats, not a loop of 3 separate GPU-loading invocations -- CLAUDE.md's never-loop rule governs any repeated GPU model-loading run), sampling `Shmem`/`MemAvailable` before and ~5s after per CLAUDE.md.

**Constraints:** Ruling 6 (event chains, no waits), ruling 9 (correctness gates run alongside).

**Acceptance test:** Epic Bar section's dual-selector TG128 `llama-bench` invocation, run once with `-r 3` (ONEAPI_DEVICE_SELECTOR pinned per epic Bar, lead session only), sampling `Shmem`/`MemAvailable` before and ~5s after, plus GPT-OSS count-to-5 correctness gate.

**Depends on:** llama.cpp-po3nd.2.46, llama.cpp-30ak7.19 (external)

### llama.cpp-po3nd.2.46

  - **Id:** llama.cpp-po3nd.2.46
  - **Action:** keep
  - **Priority:** 0
### Depends on

  - llama.cpp-po3nd.2.45
  - llama.cpp-po3nd.2.47

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Turn the already-placed contiguous layer blocks (`.45`, done) into an active executor that pipelines PP microbatches and batched decode across devices instead of the current dry-run-only classification (`executor=inactive`).

**Design:**
- Consume `plan.layer_blocks` (`unified-cache.cpp:24998`) as-is; do not re-derive placement.
- The dry-run readiness probes already exist end to end (`ggml-sycl.cpp:85300-85461`: boundary staging, output allocation, leaf materialization, transfer plan) — the remaining work is making `safe_for_block_executor==true` actually dispatch instead of only logging `executor=inactive`.
- Boundary movement must be explicit staging handles with event ownership (device output → host-pinned bounce → next device input), using `depends_on`, never queue-wide waits (ruling 6).
- Single-request TG may legitimately stay all-on-one if the cost model shows no-P2P serial dependency dominates — the ticket already permits this; don't force pipelining where profiling says it loses.
- Before committing to this as the sole mechanism, run the epic's upstream-TP evaluation (new_tasks) — `--split-mode tensor` was reported by its own PR author at +20-79% on a comparable dual-B70 dense-model topology (not independently verified; see External/upstream status) and may cover the PP/decode-overlap goal more cheaply for at least the dense portion of GPT-OSS, once its SYCL comm layer is ported into this tree.

**Bar:** PP512 GPT-OSS dual run shows both GPUs with overlapped block compute in profiler/xpu-smi evidence, not just memory residency; dual pp512/tg128 clear the epic's Bar numbers.

**Constraints:** Ruling 4, 5, 6 all apply directly (placement/layout/event-chain). No device-name branches, no env var required for default correctness.

**Acceptance test:** Epic Bar section's full three-way llama-bench matrix plus both correctness gates.

**Depends on:** llama.cpp-po3nd.2.45 (done), llama.cpp-po3nd.2.47 (done)

### llama.cpp-d8m7

  - **Id:** llama.cpp-d8m7
  - **Action:** keep
  - **Priority:** 1
### Depends on

  - llama.cpp-po3nd.2.28
  - llama.cpp-03k1

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Run the lead-only, clean-reboot production validation of dual-device routing on the current B70+B50 pair, absorbing the older T5 verification scope from `xzeu`.

**Design:**
- Retitle from 'B50/B580' to 'B70/B50' before running — the B580 named in the title is physically gone.
- Follow the stated protocol: clean reboot, GPU reservation, zero source changes during the run (this is explicitly a measurement-integrity requirement, not bureaucracy).
- Absorb `xzeu`'s scope (Mistral regression check, 120B single/multi-GPU perf, coherent output validation, budget summary) since that ticket closes as a duplicate into this one.
- Re-derive any numeric target from the 80%-of-hardware-peak efficiency ruling and current `docs/backend/sycl-perf-baselines.md` rows, not the 2026-03 'multi-GPU TG >= 7.0 tok/s' figure (120B/B580-era).

**Bar:** Full same-build validation matrix (Mistral single-device guard, GPT-OSS 20B three-way, 120B dual where applicable) passes on a freshly rebooted host with a GPU reservation, logged verbatim in the ticket.

**Constraints:** CLAUDE.md GPU-serialization rule (lead session only, one benchmark at a time). Ruling 9 (correctness alongside every perf claim).

**Acceptance test:** Lead runs the epic Bar section's full command set plus a 120B `llama-bench` pass under the clean-reboot protocol; results appended verbatim to this ticket.

**Depends on:** llama.cpp-po3nd.2.28, llama.cpp-03k1

  - **New title:** [SYCL-120B] Validate production B70/B50 dual-device routing

### llama.cpp-kkxtv

  - **Id:** llama.cpp-kkxtv
  - **Action:** keep
  - **Priority:** 1
### Depends on

  - llama.cpp-po3nd.2.46

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Fix the general case where dense/KV layers are planned on a hidden physical GPU that the scheduler has collapsed away, so cross-device KV/activation routing resolves through smart handles instead of silently materializing on the wrong device.

**Design:**
- Confirm current status: scheduler-visible devices are still collapsed to one logical GPU by default to avoid DEVICE_LOST (`ggml-sycl.cpp:15679, 22557, 48198`) while the unified-cache planner can still place weights/experts on all physical GPUs — this mismatch is the epic's actual bug surface.
- Break into planner / KV-buffer-materialization / smart-handle-view-resolution / cross-device-attention-SET_ROWS-routing sub-scopes as the description already gestures at; this ticket itself carries no standalone acceptance test.
- Every cross-device KV move is a host-bounce copy (no P2P) — cost that must be visible in plan diagnostics, not hidden inside a materialization call.
- No B50-specific logic; decisions come from queried device capability and planner output only.

**Bar:** Hybrid-mode GPT-OSS 20B with KV planned on a non-primary device completes p16/n4 and pp512/tg128 without silent host fallback or DEVICE_LOST, verified via `kkxtv.7`'s gate list.

**Constraints:** Ruling 2 (mem_handle is the only ownership token — KV views must resolve through it), ruling 6 (event-chained cross-device copies).

**Acceptance test:** `kkxtv.7`'s five-point gate list, run once this epic breaks the work into concrete sub-tickets.

**Depends on:** llama.cpp-po3nd.2.46

### llama.cpp-kkxtv.7

  - **Id:** llama.cpp-kkxtv.7
  - **Action:** keep
  - **Priority:** 1
### Depends on

  - llama.cpp-kkxtv

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Add the standing validation gates for cross-device KV once `kkxtv`'s routing fix lands, re-scoped against the multi-GPU infrastructure that already exists rather than the epic's original 'device collapsed to one' framing.

**Design:**
- Current code already has `multi_gpu_mode` infra in `unified-cache.cpp` beyond what this ticket's original text assumes — re-run the ticket's exact gate list (hybrid p16/n4, pp512/tg128 on `level_zero:0,1`) first to see what already passes before writing new gates.
- The six required gates stand as written: synthetic smart-handle KV test, hybrid p16/n4 with no CPU fallback, hybrid pp512/tg128, default auto-mode pp512/tg128 unregressed, single-device B70/B50 Mistral unregressed, concise debug-gated logs.
- Preserve exact command lines and logs in ticket comments per the original acceptance criteria.

**Bar:** All six gates pass and are captured as committed commands/logs in the ticket; this becomes the standing cross-device-KV regression check other epic tasks can point to.

**Constraints:** Ruling 2, 6 (handle-owned KV, event-chained routing). No env vars required for default correctness.

**Acceptance test:** The six gates listed in the ticket body, run in sequence on B70+B50.

**Depends on:** llama.cpp-kkxtv

### llama.cpp-l80a3

  - **Id:** llama.cpp-l80a3
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - llama.cpp-03k1

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Derive multi-device roles (which device is dense-primary, which holds which experts) from queried hardware capability rather than any hard-coded secondary-GPU-as-cache assumption.

**Design:**
- `GGML_SYCL_MOE_MULTI_GPU` remains a manual opt-in env var (7 call sites, e.g. `ggml-sycl.cpp:7843, 15813, 22118, 101650`) with no companion auto-role-from-queried-caps logic found nearby today — this is the concrete gap.
- Query device memory, allocation limits, queue count, XMX/subgroup support, bandwidth; feed into the same `moe_device_scores`/`dense_score` machinery `.45` already built rather than a parallel one.
- Add a concrete, greppable acceptance artifact: a log line format `(device, queried mem/XMX/bandwidth) -> assigned role` that can be verified without reading source — the ticket's current text is a goal statement, not a testable bar, per keep.json's own gap note.
- No B50-specific role enum or env var required for correctness.

**Bar:** Planner emits one log line per device showing the queried facts it used and the role it assigned, for both B70 and B50, on a fresh model load with no env vars set.

**Constraints:** Ruling 4 (placement decides executor — role assignment is upstream of that).

**Acceptance test:** `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/llama-completion -m /models/gpt-oss-20b-mxfp4.gguf -n 4 -v 2>&1 | grep -i role` shows both devices with distinct queried-fact-based role lines.

**Depends on:** llama.cpp-03k1

### llama.cpp-mk4l

  - **Id:** llama.cpp-mk4l
  - **Action:** keep
  - **Priority:** 1
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Answer, with evidence, whether the deleted `try_mixed_moe_layer_pair`/`try_secondary_moe_layer_pair` lambdas need restoring or are already superseded by the planner-side residency model — this design pass already gathered most of the needed evidence.

**Design:**
- Confirmed in this triage: those two functions do not exist in current `ggml-sycl.cpp` (deleted in `abecb785d`). What remains live and reachable is `ggml_sycl_try_decode_secondary_moe_route_on_device` (`:25526`) for decode and `dispatch_experts_secondary_gpu_impl` (`:65320`, called at `:71965`/`:76324`) for PP — both planner/capability-driven, not per-op re-checked the way the deleted lambdas were.
- The 2026-08-16 placement-decides-executor ruling (constraint 4 in this epic) is exactly the argument for why per-op re-verification is now redundant: the planner decides residency once, up front.
- No direct P2P exists (re-verified 2026-07-31) — any restored row-splitting would host-bounce, same cost the current narrower routes already pay.
- The `omp4` scope-adjacency question (do omp4's 'primary decode'/'secondary decode' dependents mean to extend this MXFP4-serving family to Q1/NVFP4) still needs an explicit owner call before closing that door — this ticket's own text flags it and this design pass does not resolve it.

**Bar:** A close-as-superseded disposition with the file:line evidence above, OR a scoped follow-up describing the specific gap if one is found — not new restoration code.

**Constraints:** Ruling 4 (placement decides executor) is the direct architectural argument. No P2P (epic constraints).

**Acceptance test:** Code read: `grep -c 'try_mixed_moe_layer_pair\|try_secondary_moe_layer_pair' ggml/src/ggml-sycl/ggml-sycl.cpp` returns 0; if the ticket closes as superseded, cite this plus the two live replacement function names.

**Depends on:** llama.cpp-haqk (external), llama.cpp-twl6 (external)

### llama.cpp-vudft

  - **Id:** llama.cpp-vudft
  - **Action:** keep
  - **Priority:** 1
### Depends on

  - llama.cpp-t00wi
  - llama.cpp-won9

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Re-baseline 120B multi-GPU TG on current B70+B50 hardware, absorbing `w2zf`'s duplicate B580-era target.

**Design:**
- Both the '8-12 tok/s' (this ticket) and '>15 tok/s' (`w2zf`, now merged in) targets are 2026-03/04 B580-era numbers and must be re-derived, not reused.
- Re-derive from the 80%-of-hardware-peak efficiency ruling: TG is memory-bandwidth-bound, so the target is a function of combined effective bandwidth actually achievable under host-bounce cross-device MoE, not a flat historical tok/s.
- Measure expert cache hit rate and cross-device staging overhead per token as the original ticket specifies — these numbers, not just the headline tok/s, are what future block/TP work will optimize against.
- Requires `t00wi` (sharded GGUF loading) to work first — 120B is a 3-shard model.

**Bar:** Same-build single-GPU vs. dual-GPU 120B TG comparison, with a target re-derived from measured hardware bandwidth (not a carried-over B580 number), no DEVICE_LOST/hangs/correctness drift.

**Constraints:** Ruling 9 (correctness gates alongside), efficiency-goal ruling (80% of hardware peak, memory-bandwidth-anchored for decode).

**Acceptance test:** `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 timeout 300 ./build/bin/llama-bench -m /models/gpt-oss-120b-mxfp4-00001-of-00003.gguf -p 16 -n 16 -r 1` plus `llama-completion` correctness check on 120B, lead-run per CLAUDE.md GPU-serialization rule.

**Depends on:** llama.cpp-t00wi, llama.cpp-won9

### llama.cpp-won9

  - **Id:** llama.cpp-won9
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - llama.cpp-bci0

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Verify that the secondary GPU actually participates in planned expert placement and dispatch through the current unified-cache/mem_handle contract, re-scoped off the deleted ExpertPrefetcher framing.

**Design:**
- The dependency this ticket names (`mfuj`) committed the old `moe_prestage_popular_experts()`/`ExpertCache` design, which is gone — re-scope the checklist to current terminology: `unified_cache` zones, `mem_handle`, `[MOE-LAYOUT]`/`[MOE-ROUTE-STATS]` logs.
- Five checks as originally specified: device query reports all Level Zero GPUs; placement plan assigns some work to the secondary device when policy says so; unified cache materializes handles on the planned device; route logs show dispatch to the owning device; cross-device buffers are event-chained.
- No MoE-specific env var should be required for the checks to pass (ticket's own acceptance criterion) — this is a default-path verification, not an opt-in-flag demo.

**Bar:** All five checks pass on a fresh dual-device GPT-OSS 20B load and short inference run with no env vars set; per-device allocation logs match the plan.

**Constraints:** Ruling 1/2 (unified_cache sole allocator, mem_handle sole ownership token).

**Acceptance test:** `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/llama-completion -m /models/gpt-oss-20b-mxfp4.gguf -p 'Hi' -n 8 -v 2>&1 | grep -E 'MOE-LAYOUT|MOE-ROUTE-STATS'` — confirm secondary-device entries.

**Depends on:** llama.cpp-bci0

### llama.cpp-fawh

  - **Id:** llama.cpp-fawh
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - llama.cpp-03k1

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Harden per-device VRAM budget isolation and add the missing cross-device diagnostics, now scoped as hardening rather than a bug fix (root cause of the original regression was MoE auto-enable, not budget contamination, per this ticket's own note).

**Design:**
- `PER_DEVICE` cache mode and `PER_DEVICE_REACCOUNT_FAILED` refusal path already exist (**corrected 2026-09-01**: refusal at `unified-cache.cpp:23662` -- `return refuse(moe_mmid_runtime_reason::PER_DEVICE_REACCOUNT_FAILED);` -- with a comment reference at `ggml-sycl.cpp:16314`; the previously-cited line numbers ~210186/222634/284529/299938 do not exist, `ggml-sycl.cpp` is 104330 lines) — partial coverage; confirm they hold with B70+B50 rather than B580+B50.
- Missing: `unified_cache_log_all_devices()` diagnostic (never landed) and an explicit cross-device leak-detection warning — these are the concrete remaining deliverables.
- Lower priority than the epic's core executor work; this is a safety-net for when `GGML_SYCL_MOE_MULTI_GPU=1` is actually used, not a blocker for the block/TP decision.

**Bar:** `unified_cache_log_all_devices()` prints independent budget/usage for every visible device on one call; a deliberately-triggered cross-device leak produces a warning, not silence.

**Constraints:** Ruling 10 (VRAM budget formula, per-device isolation).

**Acceptance test:** Manual: call the new diagnostic during a dual-device load and confirm both devices report independently; force a leak (hold a handle past its owning model's teardown) and confirm the warning fires.

**Depends on:** llama.cpp-03k1

### llama.cpp-t00wi

  - **Id:** llama.cpp-t00wi
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Fix sharded 120B GGUF model loading across devices under a multi-device split, retargeted to B70+B50 and the current query-driven planner.

**Design:**
- Historical repro (`GGML_SYCL_SPLIT_RATIO='60,40'` segfault) is B580+B50-era; needs a fresh repro on `level_zero:0,1` with the current tree before any fix is scoped — the ticket's own gap note says no commit yet addresses this specific cross-file/cross-device segfault.
- Placement must be queried/planned before allocation; `unified_cache` owns the mapped/device/host representations, `mem_handle`s preserve shard/tensor identity across the 3-file split.
- Cross-device shard boundaries use planned transfers (event-chained, ruling 6), not loader-local placement guesses.
- Blocks `vudft`'s 120B re-baseline — this is a hard prerequisite, not parallelizable with it.

**Bar:** gpt-oss-120b (3-shard) loads without segfault with a multi-device split on B70+B50; placement plan diagnostics correctly show the layer distribution across both GPUs.

**Constraints:** Ruling 1/2 (unified_cache/mem_handle ownership through the load path). No P2P — shard boundary transfers host-bounce.

**Acceptance test:** `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 timeout 300 ./build/bin/llama-completion -m /models/gpt-oss-120b-mxfp4-00001-of-00003.gguf -p 'Hi' -n 4` completes without segfault; lead-run.

**Depends on:** llama.cpp-017kb (external epic)

### llama.cpp-dt06

  - **Id:** llama.cpp-dt06
  - **Action:** keep
  - **Priority:** 2
### Depends on

  - _(empty)_

  - **Addendum:** ## Design addendum (triage 2026-09-01)

**Intent:** Stop the TP debug sync catch from silently returning a corrupted-but-successful result when a tensor-parallel kernel submission actually fails on one device.

**Design:**
- Location confirmed unchanged: `ggml-sycl.cpp` (~62168-62179 in current numbering), inside `ggml_sycl_mul_mat`'s TP debug sync block, gated by `g_sycl_tp_config.enabled && g_sycl_tp_config.world_size > 1`.
- **Correction (triage 2026-09-01):** this code lives in the fork's own pre-existing TP scaffolding (`g_sycl_tp_config`/`world_size`, introduced `1352fb4fc`, 2025-12-07, predating the b10630 merge) — a same-purpose mechanism to, but distinct from, upstream's PR #24152 SYCL comm implementation, which is NOT currently in this tree (`get_proc_address` exports no `ggml_backend_comm_*` symbols). It still matters: this fork-local scaffolding is exactly what `new_tasks[0]`'s evaluation would build on, replace, or port over — a silent-failure bug here is directly relevant to that evaluation because it sits in the mechanism being assessed, not because it IS PR #24152's landed code.
- Decide disposition before fixing blind: rethrow, route through the existing `ggml_sycl_try_dispatch_resource_exhaustion_fallback` ladder (from `o3h1`, `756be1d6f`), or a TP-specific path that informs the following `ALL_REDUCE` that this rank failed — read the TP dispatch/ALL_REDUCE code first.
- No single-GPU repro exists (opt-in, multi-GPU-only feature); verification needs the dual-device gate this epic is building anyway.

**Bar:** The TP debug sync path either rethrows/aborts or routes a real kernel failure through a fallback ladder that the following ALL_REDUCE can see — never returns a corrupted result as if it succeeded.

**Constraints:** Ruling 9 (correctness before throughput) — this is precisely a silent-wrong-answer risk in a multi-GPU path.

**Acceptance test:** Manual code read confirming the catch no longer falls through silently, plus a forced-failure test if TP tooling permits injecting a kernel failure; otherwise a design review sign-off is the interim bar.

**Depends on:** none within this epic

### llama.cpp-beas

  - **Id:** llama.cpp-beas
  - **Action:** close
  - **Close evidence:** unified-cache.cpp:11280 init_shared_context_queues(), :11330 get_shared_context_queue(), :16216 shutdown_shared_context_queues() -- landed in 461733c32 'unify MoE multi-GPU into unified cache planner'.

### llama.cpp-po3nd.2.11

  - **Id:** llama.cpp-po3nd.2.11
  - **Action:** close
  - **Close evidence:** Per-layer/role route counters already exist: ggml-sycl.cpp ~19266-19268 planned_device_to_host_rows/planned_host_rows/true_host_rows, plus device_rows/host_rows/missing_rows/layout_mismatch_rows in moe-layer-plan.hpp:647-661. Requested telemetry is landed.

### llama.cpp-po3nd.2.20

  - **Id:** llama.cpp-po3nd.2.20
  - **Action:** close
  - **Close evidence:** Superseded by llama.cpp-po3nd.2.45's landed contiguous layer-block placement (f4b97d608). The 2026-06-01 executor-validation notes on .45/.46 record that per-op cross-device MoE-row routing DEVICE_LOSTs and was reverted ('per-op context switching is proven unsafe'); no per-token expert split+replicate executor exists or is planned. Duplicate pair with .26, both closing together.

### llama.cpp-po3nd.2.22

  - **Id:** llama.cpp-po3nd.2.22
  - **Action:** close
  - **Close evidence:** moe_layer_grouped_route_view (moe-layer-plan.hpp:647-661) already reports device_rows/host_rows/missing_rows/layout_mismatch_rows/n_groups; expert-prefetch.cpp/.hpp tracks total/interval host+missing rows. Requested lightweight route/idle telemetry is landed.

### llama.cpp-po3nd.2.24

  - **Id:** llama.cpp-po3nd.2.24
  - **Action:** close
  - **Close evidence:** No decode-time consumer of cross-device 'secondary alternates' exists (0 hits for secondary_alternate/SECONDARY_SOA in mmvq.cpp). The per-expert staging-handle direction this ticket describes was abandoned for the contiguous layer-block design (.45, landed f4b97d608); see 2026-06-01 notes on .45/.46 documenting the per-op approach as proven unsafe.

### llama.cpp-po3nd.2.25

  - **Id:** llama.cpp-po3nd.2.25
  - **Action:** close
  - **Close evidence:** A hardware-derived cost model landed (unified-cache.cpp compute_multi_device_plan/device_budget/moe_device_scores/moe_transfer_penalty, ~22160-26700) but as part of .45's contiguous layer-block placement, not the per-expert replication/placement model this ticket asked for. Superseded by the block design.

### llama.cpp-po3nd.2.26

  - **Id:** llama.cpp-po3nd.2.26
  - **Action:** close
  - **Close evidence:** No decode-time dual-device MoE executor exists that consumes route groups/secondary alternates (0 hits for secondary_alternate/SECONDARY_SOA in mmvq.cpp). Superseded by .45's landed contiguous layer-block placement (f4b97d608) plus the still-open block-pipelining executor (.46). Duplicate pair with .20, both closing together.

### llama.cpp-po3nd.2.27

  - **Id:** llama.cpp-po3nd.2.27
  - **Action:** close
  - **Close evidence:** Depends solely on .26, which closes as superseded -- there is no per-layer-split dual-device decode executor left to optimize the launch overhead of. If graph/replay launch-overhead work is needed, it belongs against the block executor (.46) once that lands, as a fresh ticket.

### llama.cpp-po3nd.2.32

  - **Id:** llama.cpp-po3nd.2.32
  - **Action:** close
  - **Close evidence:** Direction (profile-backed overlap of per-expert split MoE execution) superseded by the pipelined block executor (.46), which is the actual successor mechanism for 'both GPUs busy at once.' Blocked on external open ticket .35; all cited baselines are B580-era and stale.

### llama.cpp-po3nd.2.33

  - **Id:** llama.cpp-po3nd.2.33
  - **Action:** close
  - **Close evidence:** Diagnostic child of already-closed llama.cpp-po3nd.2.31. Target functions (try_mixed_moe_layer_pair) no longer exist in ggml-sycl.cpp (deleted in abecb785d, confirmed via llama.cpp-mk4l's investigation). All cited evidence is /tmp logs from a B580-era run, gone per the /tmp-does-not-survive-reboot rule.

### llama.cpp-po3nd.2.37

  - **Id:** llama.cpp-po3nd.2.37
  - **Action:** close
  - **Close evidence:** Depends on .25/.20/.26, all closing as superseded by the block-placement direction. The per-expert 'load-sharing replica' default-flip this ticket gates was abandoned; the narrower secondary decode route that remains (ggml_sycl_try_decode_secondary_moe_route_on_device, ggml-sycl.cpp:25526) is used for capacity-spill only and is covered by the standard correctness gates, not a separate load-sharing default-flip proof.

### llama.cpp-po3nd.2.45

  - **Id:** llama.cpp-po3nd.2.45
  - **Action:** close
  - **Close evidence:** Landed at f4b97d608 ('sycl: record dense-fastest layer blocks'), the exact sha the epic's own comments cite. Live in unified-cache.cpp: populate_multi_device_layer_blocks (:24998), populate_no_p2p_candidate_layer_blocks (:25192), use_cohesive_no_p2p_moe / [PLACEMENT-BLOCK] logging (:25974), dense-fastest scoring and hardware-derived cost model (compute_multi_device_plan/device_budget/moe_device_scores/moe_transfer_penalty, ~22160-26700).

### llama.cpp-po3nd.2.47

  - **Id:** llama.cpp-po3nd.2.47
  - **Action:** close
  - **Close evidence:** Dated 2026-06-05, pre-B70, pre-driver-26.31. Closing as MITIGATED, not resolved -- no code fix removes the underlying iGPU-enumeration hazard, only selector-pinning discipline does. Checkable root cause + fix: unpinned ONEAPI_DEVICE_SELECTOR enumerates the iGPU's 231.7 GB host-unified 'VRAM' (llama.cpp-403s), and the documented one-line mitigation (pin ONEAPI_DEVICE_SELECTOR to level_zero:0,1) is already baked into every command in this epic's Bar section and CLAUDE.md's B50 health-check guidance. Supporting but non-checkable-here: llama.cpp-sk67 comments record the B50 (level_zero:1) used routinely for perf work through 2026-08-20 with no reset-storm recurrence under the pinned selector -- cited for context, not as the sole basis for closure.

### llama.cpp-xzeu

  - **Id:** llama.cpp-xzeu
  - **Action:** merge_into
  - **Merge into:** llama.cpp-d8m7

### llama.cpp-po3nd.2.7

  - **Id:** llama.cpp-po3nd.2.7
  - **Action:** merge_into
  - **Merge into:** llama.cpp-po3nd.2.28

### llama.cpp-w2zf

  - **Id:** llama.cpp-w2zf
  - **Action:** merge_into
  - **Merge into:** llama.cpp-vudft

## New tasks

### [MULTIGPU] Evaluate upstream --split-mode tensor on GPT-OSS 20B B70+B50 as an alternative/complement to the fork's block-pipeline design

  - **Title:** [MULTIGPU] Evaluate upstream --split-mode tensor on GPT-OSS 20B B70+B50 as an alternative/complement to the fork's block-pipeline design
  - **Priority:** 0
### Depends on

  - _(empty)_

  - **Description:** ## Design addendum (triage 2026-09-01)

**Intent:** Determine, with a same-build measurement, whether PORTING upstream's now-merged backend-agnostic tensor parallelism (PR #19378, the meta backend) plus its SYCL comm implementation (PR #24152, comm_init/comm_free/comm_allreduce_tensor -- **corrected 2026-09-01: NOT currently in this tree**; ggml-sycl.cpp's get_proc_address exports no ggml_backend_comm_* symbols, so the meta backend gets comm_ctx=nullptr on this backend today) meets or beats this epic's Bar on GPT-OSS 20B before more fork-native block-executor work (.46) is built on the assumption that a bespoke mechanism is required. Note: g_sycl_tp_config/world_size (ggml-sycl.cpp ~16764-34836) is pre-existing FORK-LOCAL scaffolding (1352fb4fc, 2025-12-07, predates the b10630 merge) for a same-purpose mechanism -- it is NOT PR #24152's code; do not conflate the two when scoping this task.

**Design:**
- Build with the current tree; the meta-backend FRAMEWORK (PR #19378) is present, but this backend's `comm_*` proc addresses are unexported (see Intent), so a first attempt to run `--split-mode tensor` on GPT-OSS 20B MXFP4 across level_zero:0,1 will most likely error, refuse the split mode, or silently fall back to single-device rather than perform a real TP run -- confirm which of those actually happens before assuming any resulting tok/s number reflects two-device execution. Porting PR #24152's SYCL comm implementation in (comm_init/comm_free/comm_allreduce_tensor, exported from get_proc_address) may be this task's real first deliverable, ahead of any benchmark.
- Compare against this epic's three-way baseline (B70-only, B50-only, dual layer-split/block) using identical prompts/settings; run the GPT-OSS and Mistral correctness gates against the TP path too, since it is a different code path with its own correctness risk (see llama.cpp-dt06's live bug in this exact scaffolding).
- If TP wins or ties: the block-executor work (.46) and per-device secondary MoE work (.3/.42/.44) should be rescoped to either adopt/extend TP for the MoE-specific parts it does not yet cover, or be explicitly deprioritized in favor of it -- an owner call, not a unilateral one.
- If TP loses (e.g. all-reduce host-bounce cost dominates on this narrower-bandwidth B50, or MXFP4 MoE hits an unsupported path): record the profiler-backed reason so the fork's block design is justified with evidence, not assumption.
- Community reference point: dual Arc Pro B70, Llama-3.3-70B dense, tensor-split pp512 377.08 vs layer-split 313.65 (+20.2%), tg128 17.40 vs 9.74 (+78.6%) -- a different model class but the same card pairing, worth reproducing as a sanity check on the dense portion of GPT-OSS.

**Bar:** A same-build, interleaved paired comparison (TP vs. current best dual-device mechanism) on GPT-OSS 20B PP512/TG128, with both correctness gates green on the TP path, written up with a clear keep/drop recommendation for TP as this epic's primary mechanism.

**Constraints:** No PCIe P2P (epic constraints) -- TP's all-reduce already host-bounces by design, so this is not a new risk, just a new code path to validate. Ruling 9 (correctness gates run alongside every perf claim).

**Acceptance test (corrected 2026-09-01 -- positive control added):** FIRST, before trusting any tok/s number, confirm `--split-mode tensor` is actually exercising two SYCL backends via a real meta/TP path rather than degrading to single-device or a generic copy path (this fork intentionally collapses the scheduler-visible backend set to one logical GPU by default -- ggml-sycl.cpp:15679, 22557, 48198 -- and with no SYCL comm_* proc exported today, this control is EXPECTED TO FAIL until the port lands): (a) code read confirming `ggml-sycl.cpp`'s `get_proc_address` exports `ggml_backend_comm_init`/`comm_free`/`comm_allreduce_tensor` (currently it does not), and (b) a startup/verbose log showing two distinct SYCL backend instances registered with the meta backend and a non-null comm context for both. ONLY once that control passes: `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/llama-bench -m /models/gpt-oss-20b-mxfp4.gguf -p 512 -n 128 --split-mode tensor` compared against the epic's Bar section's dual command, plus both correctness gates run against `--split-mode tensor`. A tok/s number produced without the positive control passing must be reported as unverified (possible silent single-device fallback), not as a result.

**Depends on:** none -- this can and should run before committing further effort to llama.cpp-po3nd.2.46.

## Web findings

### Item 1

  - **Claim:** Backend-agnostic tensor parallelism ('meta backend', --split-mode tensor) merged into llama.cpp upstream around April 2026, PR #19378 (JohannesGaessler); optimized for 2 GPUs currently, other counts fall back to generic all-reduce.
  - **Url:** https://github.com/ggml-org/llama.cpp/pull/19378
  - **Impact:** Makes a from-scratch dual-device execution mechanism potentially unnecessary to build for the dense/attention portion of GPT-OSS; directly informs the epic's Work breakdown item 6 (new evaluation task).

### Item 2

  - **Claim:** SYCL implementation of --split-mode tensor landed via PR #24152: comm_init/comm_free/comm_allreduce_tensor, small tensors (nelem<32768) via FP32 direct memcpy + per-device ADD kernels, large tensors via BF16-compressed cross-device memcpy halving PCIe bytes for the all-reduce.
  - **Url:** https://github.com/ggml-org/llama.cpp/pull/24152
  - **Impact:** **Corrected 2026-09-01:** `g_sycl_tp_config`/`world_size` (`ggml-sycl.cpp` ~16764-34836) is FORK-LOCAL scaffolding introduced by `1352fb4fc` (2025-12-07), predating the b10630 merge -- `git log -S g_sycl_tp_config` bottoms out there. It is a same-purpose fork mechanism, NOT this PR's code. PR #24152's actual `comm_init/comm_free/comm_allreduce_tensor` SYCL implementation is NOT in this tree -- `comm_allreduce_tensor` appears only in `ggml/include/ggml-backend.h:226` and the generic meta backend (`ggml/src/ggml-backend-meta.cpp:1792-1823`); `ggml-sycl.cpp`'s `get_proc_address` (:103801-103826) exports no `ggml_backend_comm_init/comm_free/comm_allreduce_tensor`, so the meta backend gets `comm_ctx=nullptr` on this backend today. `llama.cpp-dt06`'s bug is real and at the cited spot (ggml-sycl.cpp ~62171-62181), but it is a defect in the fork's OWN pre-existing TP scaffolding, not in this PR's landed code. 'Port instead of build' remains the right framing, but the port has not happened yet -- it is a real gap, not already-compiled-in.

### Item 3

  - **Claim:** Dual Arc Pro B70, Llama-3.3-70B dense model: tensor-split mode pp512=377.08 tok/s vs layer-split mode 313.65 (+20.2%), tg128=17.40 vs 9.74 (+78.6%), DPC++ 2026.0.0. These are the PR author's own self-reported numbers in PR #24152's description, NOT an independently reproduced or community benchmark.
  - **Url:** https://github.com/ggml-org/llama.cpp/pull/24152
  - **Impact:** **Corrected 2026-09-01:** the previously-cited URL (spheron.network blog) does NOT contain this benchmark -- WebFetch confirms 'no benchmark comparing dual Arc Pro B70 ... no throughput metrics' on that page. The figures actually come from PR #24152's own description (url above), so this is the PR author's self-reported number, not independent/community evidence, on the exact card pairing this fork runs -- a directional reference point only, unverified until reproduced on this fork's build, for llama.cpp-po3nd.2.46's 'beat all-on-one' bar, and for a dense model, not GPT-OSS MXFP4 MoE.

### Item 4

  - **Claim:** Upstream --split-mode tensor MoE support currently covers only GPT-OSS and Qwen3-MoE among MoE/hybrid architectures; many others (DeepSeek2, Grok, Minimax-M2, etc.) still hit 'LLAMA_SPLIT_MODE_TENSOR not implemented for architecture'. GPT-OSS-120B-MXFP4 with >2 GPUs no longer crashes (previously did); a 4xRTX4090 measurement reports 1.53x PP speedup for GPT-OSS 120B MXFP4 via the new meta-backend TP path.
  - **Url:** https://github.com/ggml-org/llama.cpp/discussions/18049
  - **Impact:** GPT-OSS -- this epic's target model -- is in the upstream-supported set. Makes evaluating --split-mode tensor against this epic's bar a cheap, high-value first experiment rather than a dead end.

### Item 5

  - **Claim:** No new evidence of Level Zero peer-to-peer access becoming available between non-bridged Intel Battlemage discrete GPUs; general P2P query API documentation exists but nothing specific to unlocking cross-root-port P2P on this topology.
  - **Url:** https://oneapi.io/blog/level-zero-latest-developments/
  - **Impact:** The fork's no-P2P finding (re-verified 2026-07-31) stands un-contradicted; every design in this epic must continue to assume host-bounce for any cross-device transfer.

### Item 6

  - **Claim:** Open Intel compute-runtime issue: multi-rank MPI + Level Zero abort on Xe2/BMG-G31 (Battlemage) after a compute-runtime upgrade from 26.05 to 26.14 on the xe driver, resource_info.cpp:15.
  - **Url:** https://github.com/intel/compute-runtime/issues/922
  - **Impact:** Not this fork's exact driver version (26.31) but a live signal that multi-device Level Zero on Battlemage remains an active upstream bug surface -- worth a quick compatibility check on this specific driver before leaning further on multi-GPU Level Zero contexts, whether via TP or the fork's own path.