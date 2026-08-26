# TKV-1 spike findings — forced load-time demotion status quo (llama.cpp-vchh)

Date: 2026-08-26 · Lead session · B50 (`ONEAPI_DEVICE_SELECTOR=level_zero:1`) · build at master `5c0516f9e`
Host state: MemAvailable ≈226 GB, Shmem ≈3.6 GB before/after every run (flat); no llama-*/comfy tenants; load avg ~97 (ambient + build agents — correctness runs only, no absolute perf claims from this doc except as noted).

## Verdict

**host-demoted layers execute correctly: NO** (the fork's internal whole-layer host placement path emits garbage tokens) —
**but the campaign's actual executor (upstream sched + CPU attention over host KV) is separately CONFIRMED WORKING** by the `-nkvo` positive control below.

## Runs

### R0 — baseline control (default budget)
`llama-completion -m mistral-7b-v0.1.Q4_0.gguf -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0` → `1, 2, 3, 4, 5, 6, 7, 8, 9, 10`, rc=0. Healthy.

### R1/R2 — forced demotion (`GGML_SYCL_VRAM_BUDGET_PCT=20`)
Logs: `spike-forced-demotion-run1.log` (merged streams), `spike-forced-demotion-run2-{stdout.txt,stderr.log}` (separated).

- Placement did demote heavily and cleanly: `[PLACEMENT] Totals: weights=289.9 MB device + 3628.0 MB host, kv=2.0 MB device + 62.0 MB host`, `Total: 291.9 MB device + 3690.0 MB host (budget=1630.0 MB)`; 62 `kv_target=host` lines; `sched_reserve: graph splits = 187`.
- Run completed rc=0, TG 3.97 tok/s, PP 10.29 tok/s.
- **Output is GARBAGE**: stdout is the echoed prompt then `##` / blank repetition (run2-stdout.txt, verbatim):

```
 1, 2, 3, 4, 5, 

##

##

##

##
```

- Ancillary defects observed on the same path (recorded, not diagnosed): repeated `[STAGE-TRACE] ... alloc_fail=N ... alloc_err=4(metadata_publication_failed)` on `mem-copy-host-to-device` (11 failures by end of run), `[SYCL-ZERO-ALLOC-CHECK] runtime allocation detected during tg phase: +290.1 MB`, oneDNN zone fragmentation warnings.

Interpretation: the load-time whole-layer host demotion path (weights+KV co-located on host, executed through the fork's internal host machinery) is **structurally alive but numerically broken** — placement, accounting, and graph splitting all function; the produced tokens are wrong. This is a pre-existing defect (no campaign code has landed); tracked as its own bug ticket (see below). Plan consequence: **approach A (reusing whole-layer runtime demotion) is dead**, exactly as the spike-first decision anticipated.

### R3 — positive control for the campaign's landed mechanism: `-nkvo`
`llama-completion ... -nkvo` (all KV in host buffers; upstream ggml scheduler runs attention on the CPU backend — the exact executor the B1′ design routes demoted layers through):

- Output: `1, 2, 3, 4, 5, 6, 7, 8, 9, 10` — **byte-correct**, rc=0.
- TG 21.8 tok/s, PP 32.7 tok/s (vs ~47/~1188 all-VRAM baseline) with **all 32 layers'** attention on host KV at trivial fill. The landed feature demotes only overflow layers, so this is a hard lower bound for the mechanism at this fill.
- Logs: `spike-nkvo-mistral-{stdout.txt,stderr.log}`.

## Consequences for the plan (docs/plans/2026-08-26-tiered-kv-placement.md)

1. **TKV-5's gate is satisfied in its intent**: the executor assumption behind the landed design (scheduler-driven CPU attention over host-buft KV) is confirmed by R3. The NO verdict applies to the *internal* host-layer path, which the design deliberately does not use — demoted KV goes to a CPU-visible host buft and the SYCL backend structurally declines those ops (plan Tasks 6–8). TKV-5 may proceed once TKV-4/TKV-6 land.
2. The internal host-path garbage is filed as a separate bug (see tracker: forced whole-layer host demotion emits garbage) — NOT in this campaign's scope; the campaign must simply avoid routing anything through that path, which B1′ does by construction.
3. TKV-2 (floor calibration on GPT-OSS B50 at depth) proceeds as planned; R3 seeds the expectation that correctness holds and the cost is bandwidth-shaped.
