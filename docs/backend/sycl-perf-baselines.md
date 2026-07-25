# SYCL Performance Baselines (this fork)

Reference throughput for this fork on the local hardware. The decision-critical
**regression guardrails** stay in `CLAUDE.md`; the fuller context and the
measurement provenance live here.

Always verify correctness (Mistral / GPT-OSS completion gates in `CLAUDE.md`)
before trusting any `llama-bench` number — throughput alone never proves a path
is correct.

> **Hardware change, 2026-07-24.** The **Arc B580 has been physically replaced by
> an Arc Pro B70.** Every B580 figure in this document is retained as history for
> that card and marked **SUPERSEDED**; none of them gates anything. Measuring a
> B70 against a B580 target makes a healthy run look like a catastrophic
> regression.

---

## How these numbers were taken (read before adding any)

Each rule below cost a round of discarded measurements.

1. **One `llama-bench` run is not a baseline.** The `±` that `llama-bench`
   prints is the spread of repetitions *inside one process*; it is far tighter
   than the spread *between* processes and will make a noisy configuration look
   precise. Every figure below is **mean ± sd across ≥5 separate processes**,
   with the observed min–max range given alongside. Quote the across-run spread,
   never the single-line `±`.
2. **Record free VRAM on every run.** A background workload (ComfyUI) once held
   18.3 GiB on the B70, so llama.cpp saw 13.8 GB instead of 32.6 GB, and an
   entire round of measurements was invalidated. Expect **~32.6 GB free on the
   B70** and **~16.2 GB on the B50**. A B70 run reporting ~13.8 GB is
   confounded — discard it, do not rationalise it. `llama-bench -v` prints the
   `... - NNNNN MiB free` line; plain `llama-bench` does not. Independently,
   sweep for holders:

   ```bash
   # any process with non-zero drm-resident-vram0 on either card
   for f in /proc/*/fdinfo/*; do
       grep -l 'drm-pdev:.*0000:03:00.0' "$f" 2>/dev/null   # B70
       grep -l 'drm-pdev:.*0000:07:00.0' "$f" 2>/dev/null   # B50
   done
   ```

3. **Key off the `ONEAPI_DEVICE_SELECTOR`, never the logged `device=N`.** That
   index is assigned *after* selector filtering — a B50-only run and a B70-only
   run both print `device=0`.
4. **Check for GT resets before and after.** Numbers taken across a card fault
   are invalid; discard them and say so.
5. **Interleave A/B arms.** Throughput drifts within a session. A B50 block here
   trended 31.00 → 31.44 → 33.20 → 33.22 → 32.95 across five consecutive
   identical runs — a 7% climb *within one configuration*, comparable to the
   effect sizes people try to measure. Sequential blocks cannot separate a flag
   effect from drift; alternate the arms.
6. **B70 runs require `GGML_SYCL_OP_TIMEOUT_MS=180000`.** Its cold prestage
   exceeds the 30 s default watchdog. That is not a hang.

---

## Current baselines — GPT-OSS 20B MXFP4, FA-on

**Measured 2026-07-24/25.** Build `ab7e79cb4` (`b12099`), branch
`feature/sycl-b70-capability`, compute-runtime **26.27.39122.12**,
`GGML_SYCL_F16=ON`. Command:

```bash
ONEAPI_DEVICE_SELECTOR=<sel> GGML_SYCL_OP_TIMEOUT_MS=180000 \
  ./build/bin/llama-bench -m /Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf \
  -p 512 -n 128 -fa 1 -r 5 -v
```

All figures **MEASURED**, mean ± sd across independent processes:

| # | Device (selector) | Config | Runs | Free VRAM | PP512 tok/s | TG128 tok/s |
|---|---|---|---:|---|---:|---:|
| A | Arc Pro B70 (`level_zero:0`) | default | 11 | 32602 MiB | **1412.34 ± 14.63** <br>[1384.81–1434.14] | **43.40 ± 1.83** <br>[40.18–46.09] |
| B | Arc Pro B50 (`level_zero:1`) | default | 5 | 16250 MiB | **893.77 ± 2.32** <br>[891.65–897.06] | **32.06 ± 0.22** <br>[31.81–32.35] |
| C | Arc Pro B70 (`level_zero:0`) | `GGML_SYCL_UNIFIED_FORCE_LEGACY=1` | 11 | 32602 MiB | 1414.67 ± 17.07 <br>[1393.69–1450.30] | 43.25 ± 2.55 <br>[40.65–48.80] |
| D | Arc Pro B50 (`level_zero:1`) | `GGML_SYCL_UNIFIED_FORCE_LEGACY=1` | 5 | 16250 MiB | 893.43 ± 4.39 <br>[887.03–897.81] | 32.36 ± 1.06 <br>[31.00–33.22] |

Free VRAM was identical on all 32 runs (32602 MiB / 16250 MiB), 32 independent
holder sweeps were clean, and the kernel log showed **no GT resets or GPU hangs**
before, during, or after.

**Spread.** The B70's token generation is the noisy axis: cv 4.2% across 11 runs,
range 40.18–46.09. The B50 is inherently steady (cv 0.7% tg, 0.3% pp). Treat
B70 tg differences below ~10% between single runs as nothing.

### Correctness gates — all PASS on both devices

Same build, both selectors:

| Gate | B70 (`level_zero:0`) | B50 (`level_zero:1`) |
|---|---|---|
| Mistral 7B Q4_0 deterministic completion | PASS — `1, 2, 3, 4, 5, 6, 7, 8, 9, 10` | PASS — `1, 2, 3, 4, 5, 6, 7, 8, 9, 10` |
| GPT-OSS 20B MXFP4 count gate | PASS — `1, 2, 3, 4, 5` | PASS — `1, 2, 3, 4, 5` |

The Mistral gate output **starts** `1, 2, …, 10` and stops there on EOS. An older
note claimed it must *end* `…14, 15`; that string is unreachable at `-n 15` and
turned a passing gate into a false alarm. Corrected in `CLAUDE.md`.

Note `llama-cli` / `llama-completion` do **not** print the backend's free-VRAM
line, so the gates' VRAM state was verified by the independent holder sweep
rather than from their own logs.

---

## `GGML_SYCL_UNIFIED_FORCE_LEGACY` — measured, and the +21.8% claim does not replicate

An earlier task reported this flag worth **+21.8% tg128 on the B70** (37.27 ± 2.70
→ 45.41 ± 0.30, 3 runs vs 8, sequential blocks). **That does not reproduce here.**

**Interleaved A/B, 6 pairs, B70** (arms alternated so drift loads both equally):

| | PP512 | TG128 |
|---|---:|---:|
| default | 1415.13 ± 18.43 | 43.99 ± 2.19 |
| `FORCE_LEGACY` | 1426.89 ± 12.93 | 45.01 ± 2.19 |

Paired difference (legacy − default), tg128: **+1.02 ± 3.48 sd, sem 1.42,
t = 0.72 on 5 df — not significant** (|t| would need to exceed 2.57 at p = 0.05).
Per-pair: −0.31, +6.04, −3.18, +4.23, −1.36, +0.69. As a percentage: **+2.3%,
not +21.8%.** PP512: +0.8%, sem 9.33 — also nothing.

**The variance-collapse claim does not replicate either.** The earlier result
reported legacy sd 0.30 against default sd 2.70 and concluded "the B70's
run-to-run instability is caused by that code path." Interleaved, both arms have
**sd 2.19**; pooled over 11 runs each, legacy is the *noisier* arm (2.55 vs 1.83).

**On the B50 the flag does nothing** — 32.06 → 32.36 tg, inside noise. That half
of the earlier finding replicates.

### What the flag actually does here — verified, not assumed

With `GGML_SYCL_MUL_MAT_ROUTE_TRACE=1`, routing is **identical** in both arms:

```
backend=legacy kernel=MMQ_COALESCED   36     (both arms)
backend=legacy kernel=ONEDNN_AOS       6     (both arms)
entry 50 / graph 50 / selected 14 / dispatch-legacy 14 / dispatched-legacy 14
```

On GPT-OSS the MUL_MAT path **already selects `backend=legacy` by default**. The
flag removes only host-side preamble (`ggml_sycl_resolve` plus oneDNN
eligibility) ahead of the same fall-through. The proposed *mechanism* is sound;
the disagreement is purely about magnitude.

> **Trap for anyone repeating this.** The `[MUL-MAT-ROUTE] unified` line inside
> the `!force_legacy` block prints **zero times in both arms** — it sits in a
> oneDNN sub-branch MXFP4 never enters, so it is useless as a flag-liveness
> probe. Use the routing tallies above.

> **The flag is presence-tested, not value-tested.** At
> `ggml/src/ggml-sycl/ggml-sycl.cpp:51354`:
> ```cpp
> static bool force_legacy = (std::getenv("GGML_SYCL_UNIFIED_FORCE_LEGACY") != nullptr);
> ```
> **`GGML_SYCL_UNIFIED_FORCE_LEGACY=0` ENABLES the bypass.** To disable it, unset
> the variable (`env -u GGML_SYCL_UNIFIED_FORCE_LEGACY`). The `=1` spelling used
> throughout the docs implies a value check that does not exist. It is also
> `static`, so it latches on first evaluation — per-process only.

**Do not change any default on the strength of this.** Two independent sessions
disagree about the effect's sign and size, and the flag additionally bypasses the
oneDNN PP path other workloads rely on.

### The unresolved part — stated rather than smoothed over

The two sessions disagree on **both arms**, in opposite directions:

| | default tg128 | `FORCE_LEGACY` tg128 |
|---|---:|---:|
| earlier session (sequential blocks) | 37.27 ± 2.70 [32.58–41.11] | 45.41 ± 0.30 |
| this session (11 runs each) | 43.40 ± 1.83 [40.18–46.09] | 43.25 ± 2.55 |

This session's *default* arm lands near the earlier session's *FORCE_LEGACY* arm.
**Ruled out as the cause:** the build. The only code commit between the two
measurement sets (`f7d59875f`) is comment-only on every hot path; its sole
functional change is an early return for a **negative device id** in a
diagnostic, unreachable in normal operation. **Not yet explained:** why the
earlier default arm was both slower and three times noisier. Until that is
understood, treat B70 tg128 as a quantity that can shift ~15% between sessions
for reasons not yet identified, and do not build a tight guardrail on it.

---

## Guardrail status

**No numeric B70 guardrail is being set by this document**, and the reason is
the spread, not the effort:

- **PP512 is stable enough** (cv 1.0% over 11 runs, min 1384.81) — but the
  earlier session saw 1353.74 ± 64.52 with a min of **1233.51**. Across sessions
  the honest floor is ~1200, which is so far below the working mean that it
  would catch only catastrophic failures.
- **TG128 is not stable enough.** Two sessions disagree by 16% on the mean. Any
  floor tight enough to be useful (say >39) would have fired constantly during
  the earlier session's perfectly healthy runs.

Setting a guardrail now would encode one session's conditions as a correctness
criterion. The prerequisite is explaining the session-to-session tg shift above,
not collecting more runs.

For orientation only — **not gates** — this build on clean cards delivers
**B70 ≈ 1410 PP512 / ≈ 43 TG128** and **B50 ≈ 894 PP512 / ≈ 32 TG128** on
GPT-OSS 20B MXFP4 FA-on.

### Open regression: the B50 is far below its documented guardrail

`CLAUDE.md` records the B50 GPT-OSS 20B MXFP4 FA-on guardrail as **≥1100 PP512
and ~50+ TG128**, citing restored-fast-path evidence of ~1255 PP512 / 52 TG128,
and this document previously listed ~926 PP512 / ~48 TG128 as "current".

Measured on this build: **893.77 PP512 / 32.06 TG128.**

That is **3.5% below the previously documented ~926 PP512**, **18.7% below the
≥1100 PP512 guardrail** (28.8% below the 1255 restored-fast-path evidence), and
**36% below the ~50+ TG128 guardrail**. An earlier task
independently measured 895.47 PP512 / 31.02–33.22 TG128 on the same card, so this
is reproducible and not an artifact of this session.

**These numbers are reported as a regression, not adopted as the new B50
baseline.** Per `CLAUDE.md`'s regression rule, the documented guardrails stand.
The cause is unattributed — it may well predate this plan — and needs a bisect
against the build that produced 1255/52. Do not "fix" the discrepancy by
lowering the guardrail.

---

## SUPERSEDED — Arc B580 (card removed 2026-07-24)

> **Hardware no longer installed.** Retained as valid history for the B580 only.
> These figures gate nothing and must not be used as B70 targets.

Mistral 7B Q4_0, Arc B580:

| Metric | tok/s | Notes |
|--------|-------|-------|
| PP512 (Level 0, all VRAM) | ~1700 | default no-FA bench path |
| TG128 (Level 0, all VRAM) | ~81 | MMVQ fast-path, SOA layout, graph replay, SCRATCH TLSF zone |
| TG128 (no graph) | ~70 | MMVQ fast-path alone (graph adds ~13%) |
| PP512 (Level 3, 30% budget) | ~269 | 15/33 GPU layers, rest on CPU |
| TG128 (Level 3, 30% budget) | ~14 | CPU offload via fit_params |
| PP512 (legacy) | ~159 | `GGML_SYCL_UNIFIED_FORCE_LEGACY` set |
| TG128 3-device (B580+B50+CPU) | ~27 | `GGML_SYCL_SPLIT_RATIO="60,32,8"` tensor split |
| TG128 (persistent TG, phase) | ~30 | `GGML_SYCL_PERSISTENT_TG=1` (experimental) |
| TG128 (persistent TG, DAG) | ~19 | `GGML_SYCL_PERSISTENT_TG=1` + `PHASE=0 DAG=1` |

A B580 record of `5b206c499-dirty` at **2173.92 PP512 / 88.42 TG128** (Mistral
7B Q4_0, FA-on) appears in `docs/backend/SYCL.md`. Same status: B580 history.

GPT-OSS 20B MXFP4 on the B580: **~66 PP512 / ~17 TG128** — its small VRAM budget
caused heavy memory pressure on this model.

The three-device split figure above involves a card that no longer exists;
**B70↔B50 P2P has not been re-tested** (see `CLAUDE.md`).

---

## Historical — Arc Pro B50, ECC disabled

Mistral 7B Q4_0 (card still installed; these predate the current build):

| Metric | tok/s | Notes |
|--------|-------|-------|
| PP512 (Level 0, all VRAM) | ~1197 | default no-FA bench path, ECC disabled |
| TG128 (Level 0, all VRAM) | ~44 | Coalesced/SOA MMVQ, 70 W power cap |

Do not use `GGML_SYCL_FA_ONEDNN_ALLOW=1` to restore Mistral PP numbers. It can
raise PP throughput, but the deterministic completion gate produces incorrect
output with the current `nc != D` contiguity fast-path.

Multi-device (`level_zero:0,1`) GPT-OSS throughput is still **TBD**; direct P2P
is unavailable, so host-bounce transfer paths are required.
