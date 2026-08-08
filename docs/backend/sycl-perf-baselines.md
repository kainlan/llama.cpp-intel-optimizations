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

   **The sharpest evidence is a sign flip within a single session.** The same
   flag, same build, same card, same day, measured two ways:

   | design | B70 tg128 result |
   |---|---:|
   | sequential blocks (default block, then legacy block) | **−3.6%** |
   | interleaved, 6 pairs | **+2.3%** (t = 0.72, ns) |

   Blocked designs here do not merely inflate an effect — they can invert its
   sign. Neither number is the "real" one at 3–4%; the point is that a blocked
   design's output is not a measurement of the flag at all. Pair the arms and
   report the paired statistic.

6. **B70 runs require `GGML_SYCL_OP_TIMEOUT_MS=180000`.** Its cold prestage
   exceeds the 30 s default watchdog. That is not a hang.

---

## Current baselines — GPT-OSS 20B MXFP4, FA-on

**Measured 2026-07-24/25.** Build `ab7e79cb4` (`b12099`), branch
`feature/sycl-b70-capability`, compute-runtime **26.27.39122.12**,
`GGML_SYCL_F16=ON`. Command:

```bash
ONEAPI_DEVICE_SELECTOR=<sel> GGML_SYCL_OP_TIMEOUT_MS=180000 \
  ./build/bin/llama-bench -m /models/gpt-oss-20b-mxfp4.gguf \
  -p 512 -n 128 -fa 1 -r 5 -v
```

All figures **MEASURED**, mean ± sd across independent processes:

| # | Device (selector) | Config | Runs | Free VRAM | PP512 tok/s | TG128 tok/s |
|---|---|---|---:|---|---:|---:|
| A | Arc Pro B70 (`level_zero:0`) | default | **21** | 32602 MiB | **1414.62 ± 11.13** <br>[1384.81–1434.14] | **43.57 ± 1.46** <br>[40.18–46.27] |
| B | Arc Pro B50 (`level_zero:1`) | default | 5 | 16250 MiB | **893.77 ± 2.32** <br>[891.65–897.06] | **32.06 ± 0.22** <br>[31.81–32.35] |
| C | Arc Pro B70 (`level_zero:0`) | `GGML_SYCL_UNIFIED_FORCE_LEGACY=1` | 11 | 32602 MiB | 1414.67 ± 17.07 <br>[1393.69–1450.30] | 43.25 ± 2.55 <br>[40.65–48.80] |
| D | Arc Pro B50 (`level_zero:1`) | `GGML_SYCL_UNIFIED_FORCE_LEGACY=1` | 5 | 16250 MiB | 893.43 ± 4.39 <br>[887.03–897.81] | 32.36 ± 1.06 <br>[31.00–33.22] |

Free VRAM was identical on all 32 runs (32602 MiB / 16250 MiB), 32 independent
holder sweeps were clean, and the kernel log showed **no GT resets or GPU hangs**
before, during, or after.

**Why row A pools 21 runs across three designs.** Its runs come from a sequential
block (5), the interleaved A/B's default arm (6), and a thermal probe (10). All
21 are the *same configuration* on the same build with the same free VRAM, so
they are 21 samples of one quantity; the designs differ only in what other work
surrounded them, which is exactly the session-to-session variation a baseline
should span rather than exclude. The three designs agree closely — tg means
42.69 / 43.99 / 43.76, pp means 1409.00 / 1415.13 / 1417.13 — so pooling is not
hiding a disagreement. (An earlier revision quoted the 11-run subset as
1412.34 ± 14.63 / 43.40 ± 1.83; the pooled figure supersedes it and is tighter.)
Rows C and D remain unpooled at their stated run counts.

**Spread.** The B70's token generation is the noisy axis: cv 3.3% across 21 runs,
range 40.18–46.27. The B50 is inherently steady (cv 0.7% tg, 0.3% pp). Treat
B70 tg differences below ~10% between single runs as nothing.

## Current baselines — Mistral 7B Q4_0, FA-on

Same build and method, 5 runs per device (the default arm of the interleaved
Mistral A/B below). All **MEASURED**:

| Device (selector) | Runs | Free VRAM | PP512 tok/s | TG128 tok/s |
|---|---:|---|---:|---:|
| Arc Pro B70 (`level_zero:0`) | 5 | 32602 MiB | **2495.42 ± 62.88** <br>[2425.24–2594.58] | **107.66 ± 1.06** <br>[106.57–109.15] |
| Arc Pro B50 (`level_zero:1`) | 5 | 16250 MiB | **1187.83 ± 18.79** <br>[1165.35–1211.57] | **46.53 ± 0.19** <br>[46.27–46.78] |

Two things worth noting:

- **The B70 comfortably clears the retired B580 Mistral guardrail** (>2000 PP512,
  >85 TG128): it measures ~2495 / ~108 against the B580's ~2174 / ~88. So the
  hardware change was an upgrade on this workload, and the old guardrail — while
  still not the right gate for a different card — is not a level the B70
  struggles to reach.
- **The B50 Mistral figures reproduce their historical values** (~1197 PP512,
  ~44 TG128 → measured 1187.83 / 46.53). Unlike the B50's GPT-OSS numbers, there
  is **no regression here**, which localises that regression to the GPT-OSS path
  rather than to the card or the driver generally.

Mistral is also far steadier than GPT-OSS on the B70 (tg cv 1.0% vs 3.3%), so it
is the better workload for detecting small changes on that card.

### Correctness gates — all PASS on both devices

Same build, both selectors, **both configurations** — 8 of 8 pass:

| Gate | config | B70 (`level_zero:0`) | B50 (`level_zero:1`) |
|---|---|---|---|
| Mistral 7B Q4_0 deterministic completion | default | PASS — `1, 2, 3, 4, 5, 6, 7, 8, 9, 10` | PASS — `1, 2, 3, 4, 5, 6, 7, 8, 9, 10` |
| Mistral 7B Q4_0 deterministic completion | `FORCE_LEGACY` | PASS — `1, 2, 3, 4, 5, 6, 7, 8, 9, 10` | PASS — `1, 2, 3, 4, 5, 6, 7, 8, 9, 10` |
| GPT-OSS 20B MXFP4 count gate | default | PASS — `1, 2, 3, 4, 5` | PASS — `1, 2, 3, 4, 5` |
| GPT-OSS 20B MXFP4 count gate | `FORCE_LEGACY` | PASS — `1, 2, 3, 4, 5` | PASS — `1, 2, 3, 4, 5` |

Generated text is **identical** between the default and `FORCE_LEGACY` arms on
both devices for both models — so the flag's large Mistral PP cost documented
below is a pure throughput loss, not a correctness difference. Gate both arms of
any dispatch A/B: a routing change that produced correct tokens on one model and
garbage on another would otherwise be invisible.

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
**sd 2.19**; pooled over the matched 11-run subsets, legacy is the *noisier* arm
(2.55 vs 1.83), and against the full 21-run default sample the gap widens
(2.55 vs 1.46).

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
disagree about the effect's sign and size on GPT-OSS — and on Mistral the flag is
catastrophic, as the next section measures.

### On Mistral the same flag costs ~⅔ of prompt processing

The GPT-OSS null result is **workload-specific and must not be generalised.**
Interleaved, 5 pairs per device, Mistral 7B Q4_0 FA-on, same build:

| device | metric | default | `FORCE_LEGACY` | paired change |
|---|---|---:|---:|---:|
| B70 | PP512 | 2495.42 ± 62.88 | **835.67 ± 3.43** | **−66.5%** (t = −58.9) |
| B70 | TG128 | 107.66 ± 1.06 | 105.36 ± 1.50 | −2.1% (t = −2.6) |
| B50 | PP512 | 1187.83 ± 18.79 | **357.26 ± 0.66** | **−69.9%** (t = −100.1) |
| B50 | TG128 | 46.53 ± 0.19 | 46.43 ± 0.11 | −0.2% (t = −1.3, ns) |

The arms do not overlap at all. Token generation is untouched; the entire loss is
in prompt processing, on both cards.

**Why GPT-OSS showed nothing and Mistral collapses** — route tallies, not
inference:

```
Mistral, default :  794 backend=unified kernel=MMQ_AOS        (400 unified-block traces)
Mistral, legacy  : 1191 backend=legacy  kernel=MMQ_COALESCED  (  0 unified-block traces)

GPT-OSS, default :   36 backend=legacy  kernel=MMQ_COALESCED  (  0 unified-block traces)
GPT-OSS, legacy  :   36 backend=legacy  kernel=MMQ_COALESCED  (  0 unified-block traces)
```

On GPT-OSS MXFP4 the flag is a **no-op** — MUL_MAT already routes to legacy, so
there is no oneDNN path left to lose, which is exactly why its A/B came back
null. On Mistral Q4_0 the default genuinely routes `backend=unified`, and the
flag diverts it to `backend=legacy`, forfeiting the oneDNN prompt-processing
path. A null result on one model is not evidence the flag is inert.

**The whole caveat in one line:** the flag's cost depends entirely on *which
routing decision that model's MUL_MAT would otherwise have made*. Same flag,
opposite consequences.

> ### Do not read the GPT-OSS null as "the flag is harmless"
> Setting `GGML_SYCL_UNIFIED_FORCE_LEGACY` globally on the strength of a null
> GPT-OSS result destroys Mistral prompt processing by a factor of three.

**This was already on record and had been missed.** The superseded B580 table
further down this document has always carried:

```
| PP512 (Level 0, all VRAM) | ~1700 | default no-FA bench path |
| PP512 (legacy)            | ~159  | GGML_SYCL_UNIFIED_FORCE_LEGACY set |
```

A ~10x Mistral PP collapse under this flag, recorded on B580-era hardware with no
build SHA, date, or rep count. It is **not** a figure this session measured, and
its magnitude is unverified on current hardware — the measured collapse here is
~3x (B70) and ~3.3x (B50), not ~10x, so treat ~159 as directionally
corroborating and numerically historical. What matters is that the *mechanism*
was documented long before this session and the plan still spent a full task
treating the oneDNN-PP downside as speculative.

This is the measured basis for the recommendation above. Earlier revisions of
this document asserted the oneDNN-PP downside on plausibility; it is now
quantified.

### The unresolved part — stated rather than smoothed over

The two sessions disagree on **both arms**, in opposite directions:

| | default tg128 | `FORCE_LEGACY` tg128 |
|---|---:|---:|
| earlier session (sequential blocks) | 37.27 ± 2.70 [32.58–41.11] | 45.41 ± 0.30 |
| this session (default 21 runs, legacy 11) | 43.57 ± 1.46 [40.18–46.27] | 43.25 ± 2.55 |

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

- **PP512 is stable enough** (cv 0.8% over 21 runs, min 1384.81) — but the
  earlier session saw 1353.74 ± 64.52 with a min of **1233.51**. Across sessions
  the honest floor is ~1200, which is so far below the working mean that it
  would catch only catastrophic failures.
- **TG128 is not stable enough.** Two sessions disagree by 16% on the mean. Any
  floor tight enough to be useful (say >39) would have fired constantly during
  the earlier session's perfectly healthy runs.

Setting a guardrail now would encode one session's conditions as a correctness
criterion. The prerequisite is explaining the session-to-session tg shift above,
not collecting more runs.

**Mistral is the better guardrail candidate for the B70** — its spread is far
tighter (PP cv 2.5%, TG cv 1.0%, versus GPT-OSS's 3.3% TG and a 16% cross-session
disagreement), and its B50 values reproduce their historical figures, which
GPT-OSS's do not. What it still lacks is a **second independent session**: these
are 5 runs from one sitting, and the whole point of the GPT-OSS failure above is
that one sitting can mislead. Re-measure Mistral on a separate boot before
proposing a number.

For orientation only — **not gates** — this build on clean cards delivers:

| model | B70 | B50 |
|---|---|---|
| GPT-OSS 20B MXFP4 | ≈ 1410 PP512 / ≈ 43 TG128 | ≈ 894 PP512 / ≈ 32 TG128 |
| Mistral 7B Q4_0 | ≈ 2495 PP512 / ≈ 108 TG128 | ≈ 1188 PP512 / ≈ 47 TG128 |

### Open regression: the B50 GPT-OSS PP gap against its historical evidence

⚠️ **Read the amendment at the end of this section before quoting the ≥1100
figure. It is no longer a guardrail anywhere, and resurrecting it as one has
already caused three false-regression scares in a single session.**

`CLAUDE.md` **used to** record the B50 GPT-OSS 20B MXFP4 FA-on guardrail as
**≥1100 PP512 and ~50+ TG128**, citing restored-fast-path evidence of ~1255
PP512 / 52 TG128, and this document previously listed ~926 PP512 / ~48 TG128 as
"current".

Measured on this build: **893.77 PP512 / 32.06 TG128.**

That is **3.5% below the previously documented ~926 PP512**, **18.7% below the
≥1100 PP512 guardrail** (28.8% below the 1255 restored-fast-path evidence), and
**36% below the ~50+ TG128 guardrail**. An earlier task
independently measured 895.47 PP512 / 31.02–33.22 TG128 on the same card, so this
is reproducible and not an artifact of this session.

**The regression is specific to GPT-OSS, not to the card.** The same B50 on the
same build reproduces its historical Mistral figures (~1197 → 1187.83 PP512,
~44 → 46.53 TG128). So the card, the driver install and the general SYCL path
are all delivering expected throughput; whatever regressed lives on the GPT-OSS
/ MoE path.

**Split the shortfall into what is attributed and what is not**, so a bisect
starts in the right place:

| component | status | evidence |
|---|---|---|
| **TG**, ~50+ → ~32 | **partly attributed to the driver** | 26.27 costs ~8.6% TG vs 26.22 on this exact card and model (26.22: 899.40 PP / 32.81 TG → 26.27: 884–893 PP / 29.5–30.7 TG). Measured with the *same binary* either side of the driver change |
| **PP**, ≥1100 → ~894 | **UNATTRIBUTED** | the driver costs only ~1.4% PP, so it explains essentially none of this gap. Measured 893.77 sits squarely inside the 26.27 band |

So the driver is a real but **partial** contributor on the TG axis and a
non-explanation on the PP axis. **The PP gap is where a bisect should start** —
against the build that produced 1255/52 — not the TG one.

> **Driver rollback is not available via `apt`.** The PPA carries only the newest
> build and the upgrade deleted the prior `.so`. Snapshot the `.deb` before any
> future driver change; treat the driver as a pinned, tuned dependency rather
> than something to keep current.

**These numbers are reported as a regression, not adopted as the new B50
baseline.** The residual cause may predate this plan. Do not "fix" the
discrepancy by lowering a guardrail.

#### Amendment 2026-08-08: the ≥1100 / ~50+ figures are RETIRED as a gate

The sentence above originally read "the documented guardrails stand." That is no
longer accurate about the ≥1100 PP512 / ~50+ TG128 pair specifically, and the
correction matters in the opposite direction from the rest of this section.

`CLAUDE.md` **removed** that guardrail on 2026-07-25 on the grounds that it
predates the 26.27 driver: against it, a healthy B50 (~894–902 PP512) reads as an
~18% catastrophe, and it triggered three separate false-regression scares in one
session. Its current text is explicit — *"A `≥1100 PP512, ~50+ TG128` B50 GPT-OSS
guardrail appeared here until 2026-07-25 and was wrong… If you find it quoted
anywhere else, it is wrong there too."* This document was one of the places it
was quoted.

What survives the retirement, and what does not:

- **Not a gate:** ≥1100 PP512 and ~50+ TG128. Never block a merge, fail a run, or
  open a regression ticket on them. The same applies wherever they are still
  baked into tracker acceptance criteria — `llama.cpp-po3nd.2` and its children,
  `llama.cpp-ix58x`, `llama.cpp-aqzz3` — which a doc pass cannot reach and which
  therefore cannot be closed as written.
- **Still a gate:** the merge-certification floors in the section below, which
  are the numbers this branch is actually certified against.
- **Still genuinely open:** the *analysis*. The measured 893.77 PP512 sits ~29%
  below the 1255 PP512 restored-fast-path evidence, the driver explains only
  ~1.4% of it, and nothing has attributed the remainder. That is an unexplained
  historical gap on the GPT-OSS/MoE path, worth a bisect against the build that
  produced 1255/52 — but it is **not** a regression of this branch against a live
  guardrail, and merge certification does not turn on it.

The distinction is the whole point: a retired guardrail is not the same as a
resolved question. Retiring the number does not close the investigation, and the
investigation does not license reinstating the number.

---

## Merge-certification performance gate (plan Task 20 step 6)

The numeric conditions under which
`docs/plans/2026-08-02-sycl-merge-readiness.md` certifies this branch. They are
stated here so the gate is read from the baseline document rather than from a
task's acceptance criteria — several of which still carry the retired figures
above.

**Matrix.** Five separate `llama-bench` processes per arm, four arms
(B70/B50 × Mistral 7B Q4_0 / GPT-OSS 20B MXFP4), `-p 512 -n 128 -fa 1 -r 5 -v`,
selector pinned per card, each call through the plan's `run_gpu` wrapper so
memory settlement is enforced between runs. `-v` is mandatory: without it
`llama-bench` installs a null log callback and the free-VRAM line never appears.

**B50 — explicit floors.** Five-process mean must be at least:

| model | PP512 floor | TG128 floor |
|---|---:|---:|
| GPT-OSS 20B MXFP4 | 849 | 30.4 |
| Mistral 7B Q4_0 | 1128 | 44.6 |

**B70 — no numeric floor is authorized.** This document does not license one.
The test is band membership against the recorded historical min–max:

| model | PP512 band | TG128 band |
|---|---|---|
| GPT-OSS 20B MXFP4 | 1384.81–1434.14 | 40.18–46.27 |
| Mistral 7B Q4_0 | 2425.24–2594.58 | 106.57–109.15 |

A B70 mean outside its band **blocks merge and opens an owner-reviewed
baseline/regression task**. It does not authorize inventing a floor, and it does
not authorize lowering one.

**Free VRAM must be checked on every run**, from the `-v` log: about 32602 MiB
(B70) and 16250 MiB (B50). A B70 run reporting ~13.8 GB is confounded by another
tenant and must be discarded, not rationalized.

**No quiet-host precondition** (owner ruling, 2026-08-07). The host's ambient
load — roughly 60, from Emby, the `ws2022ci` qemu VM and Frigate's ffmpeg — is
its permanent operating state. Run the matrix under it; never defer waiting for
quiet. If an arm misses its floor or band, discriminate load from regression with
an interleaved paired A/B against a known-good comparator build on the same card,
model and host state. Paired ratio within noise means load-depressed (file the
owner-reviewed baseline task with the pairing attached); branch consistently
slower within pairs means a real regression and blocks merge. Absolute numbers
taken under load still must not become new baselines.

### Fail-closed parsing — required procedure, and its current gap

The gate above is only as good as the step that reads the logs, and reading them
by eye fails **open**: a missing arm, a truncated log, or a run that emitted no
`-fa 1` row all look the same as a clean pass. The plan therefore requires a
parser that:

1. extracts the `-fa 1` PP512 and TG128 rows from each `llama-bench -v` log;
2. asserts **exactly five samples per arm** — a short arm is an error, not a
   smaller sample;
3. asserts the free-VRAM line is present and within the expected band for that
   selector;
4. computes the per-arm mean and compares it against the floor or band above;
5. **exits non-zero on any missing, short, or unparseable sample** — never
   reports a verdict it could not compute.

Point 5 is the fail-closed property and the reason the parser exists. An empty
grep over an empty capture returns clean forever; the parser must be able to say
"I could not measure this", and that must be a failure.

### `scripts/parse-sycl-bench-matrix.py` — the implementation

Stdlib-only Python, invoked directly by the lead. It is deliberately **not** a
registered CTest target: it reads evidence, it is not a test.

```sh
# after the matrix has run
python3 scripts/parse-sycl-bench-matrix.py --dir artifacts/perf-final

# verify the parser itself before trusting its verdict (see below)
python3 scripts/parse-sycl-bench-matrix.py --self-test

# explicit arms, if the logs are not in the conventional layout
python3 scripts/parse-sycl-bench-matrix.py --arm b50-mistral=a.log,b.log,c.log,d.log,e.log ...
```

`--dir` expects `<arm>-<n>.log` (or `.txt`) for n in 1..5, with arms named
`b70-mistral`, `b70-gptoss`, `b50-mistral`, `b50-gptoss` — the layout plan step 6
already writes.

**Exit codes, and why there are three rather than two:**

| exit | meaning | what to do |
|---:|---|---|
| 0 | every arm present, parseable, and within its floor/band | proceed |
| 1 | **VERDICT FAIL** — everything parsed, and a mean missed its gate | discriminate load from regression with an interleaved paired A/B; a B70 miss opens an owner-reviewed baseline task |
| 2 | **INPUT/PARSE FAILURE** — no verdict could be computed | fix the inputs and re-run the matrix; **this is not a pass and not a fail** |

"The branch is slow" and "I could not measure the branch" are different facts,
and collapsing them into one non-zero code is how a missing arm gets triaged as
a performance problem.

**Stream routing is part of the contract**, so a caller that captures only
stdout cannot mistake an error for a result: stdout carries results (the
per-arm means table, and a `PASS` verdict), stderr carries every diagnostic
that accompanies a non-zero exit. **Exit 2 writes nothing at all to stdout** —
no verdict was computable, so there is no result to report. The `--self-test`
asserts this per case rather than taking it on trust; a mutation routing one
diagnostic back to stdout drops it from 10/10 to 4/10.

Every one of these is exit 2, not a smaller sample: a missing file, an empty
file, an arm with fewer than five logs, an arm entirely absent, a results
directory that does not exist, a directory that exists but is empty, an
unparseable `t/s` cell, a table with no `fa` column (i.e. not the `-fa 1`
matrix), a log with no `- NNNNN MiB free` line (i.e. run without `-v`), and a
run whose free VRAM is below the contamination floor.

**Verify the parser before trusting it.** `--self-test` runs it against the
committed fixtures in `artifacts/task18-parser-fixtures/` and must report
**10/10**. The cases exist to prove the parser returns *all three* exit codes —
including a below-floor fixture that must produce exit 1 — so that a `PASS` is a
measurement rather than the only answer it is capable of giving. A checker nobody
has seen fail is indistinguishable from a checker that cannot fail.

> ⚠️ **The free-VRAM floors are contamination detectors, not precision checks,
> and the documented B50 figure does not match reality.** This document states
> "~16.2 GB free on the B50", but the only two committed real B50 `-v` captures
> (`captures/16-soak-mistral-bench-r30.err`, `captures/17-soak-gptoss-bench-r15.err`)
> both report **14677 / 14679 MiB free on a healthy card**. A check pinned to the
> documented figure would fail every healthy B50 run — the same shape as the
> retired ≥1100 guardrail, a stale number manufacturing a false alarm. The
> discrepancy is unresolved (16250 is suspiciously close to the card's ~16304 MB
> *total*, so the documented value may be a total rather than an observed free),
> so the parser defaults to floors that catch gross contamination — 30000 MiB
> (B70) and 14000 MiB (B50) — and exposes `--min-free-mib b70=NNNNN`. **No
> committed B70 `-v` capture exists at all**; the B70 floor is derived from the
> ~32602 MiB figure here and should be tightened once a real capture lands.

> ⚠️ **The matrix itself has not been run.** `artifacts/perf-final/` does not
> exist, so there are no results to parse yet — the parser is verified against
> fixtures only. Until Task 20 runs the matrix, this gate is unmeasured, and an
> unmeasured gate must never be recorded as passed.

Historical note on what *not* to copy: `captures/soak_analyze.py` prints
`NO SITE ROWS -- capture is VOID, not clean` on empty input but **exits 0 on
every path**. It is fail-loud, not fail-closed — fine for a human reading its
output, useless in a gate that checks a status code.

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
| PP512 (legacy) | ~159 | `GGML_SYCL_UNIFIED_FORCE_LEGACY` set — a ~10x PP collapse; see the Mistral FORCE_LEGACY section, where the same mechanism is measured on current hardware |
| TG128 3-device (B580+B50+CPU) | ~27 | `GGML_SYCL_SPLIT_RATIO="60,32,8"` tensor split |
| TG128 (persistent TG, phase) | ~30 | `GGML_SYCL_PERSISTENT_TG=1` (experimental) |
| TG128 (persistent TG, DAG) | ~19 | `GGML_SYCL_PERSISTENT_TG=1` + `PHASE=0 DAG=1` |

A B580 record of `5b206c499-dirty` at **2173.92 PP512 / 88.42 TG128** (Mistral
7B Q4_0, FA-on) appears in `docs/backend/SYCL.md`. Same status: B580 history.

GPT-OSS 20B MXFP4 on the B580: **~66 PP512 / ~17 TG128** — its small VRAM budget
caused heavy memory pressure on this model.

The three-device split figure above involves a card that no longer exists.

⚠️ **The "B70↔B50 P2P has not been re-tested" hedge that stood here is RETIRED.**
It was re-verified on the B70 on 2026-07-31 and the answer is settled: **there is
no direct P2P between the two discrete cards**, and the cause is PCI topology
rather than a property of either card — different CPU root ports (`00:06.0` vs
`00:06.3`), no shared upstream bridge, so the kernel refuses peer DMA outright.
A 256 KiB direct device-to-device USM copy fails **both** directions with
`UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` on a card with 31.89 GiB free, and
`can_access_peer` returns false both directions. Note that
`ext_oneapi_enable_peer_access()` returns OK anyway — it is not a capability
check. Full account in `CLAUDE.md`, "Patched compute-runtime & P2P topology".
Consequently host-bounce (`level_zero:0,1`) remains the only multi-device
transfer path and `GGML_SYCL_MOE_MULTI_GPU` stays opt-in.

---

## Historical — Arc Pro B50, ECC disabled

Mistral 7B Q4_0 (card still installed; these predate the current build):

| Metric | tok/s | Notes |
|--------|-------|-------|
| PP512 (Level 0, all VRAM) | ~1197 | default no-FA bench path, ECC disabled |
| TG128 (Level 0, all VRAM) | ~44 | Coalesced/SOA MMVQ, 70 W power cap |

⚠️ **RETIRED 2026-08-08 — the warning that stood here was wrong in both
directions.** It read: *"Do not use `GGML_SYCL_FA_ONEDNN_ALLOW=1` to restore
Mistral PP numbers. It can raise PP throughput, but the deterministic completion
gate produces incorrect output with the current `nc != D` contiguity
fast-path."* Both clauses are false:

1. **The variable does not exist.** `3c8f296fd` (2026-05-15) removed the
   `getenv("GGML_SYCL_FA_ONEDNN_ALLOW")` bypass **and added that warning in the
   same commit** — it documented a footgun it had just deleted. Setting the name
   today is a no-op. Both sites this note first listed as stale prose have since
   been corrected and are now history rather than advice:
   `ggml/src/ggml-sycl/fattn.cpp:2803-2807` states outright that "That getenv was
   removed in `3c8f296fd` and the materializer landed in `16a241dd1`; the
   variable no longer exists anywhere, so do not reintroduce it as a 'safety'
   switch", and `docs/backend/SYCL.md:897` carries its own correction. The last
   live settings of the dead name — one test leg and two lines of
   `scripts/validate-kkxtv7-sycl.sh` — were removed under `llama.cpp-ivue`.
2. **oneDNN SDPA is on by default and Mistral uses it, with no correctness
   penalty.** `16a241dd1` (2026-05-30) added the `MATERIALIZE_REQUIRED` path, so
   nc≠D GQA is no longer rejected. Disabling it costs ~39% of PP512 (1.63x
   slower) on an interleaved paired B50 A/B.

The real variable is **`GGML_SYCL_FA_ONEDNN`** (default ON; `=0` disables).
Full derivation in `CLAUDE.md`, "Performance Expectations".

Multi-device (`level_zero:0,1`) GPT-OSS throughput is still **TBD**; direct P2P
is unavailable, so host-bounce transfer paths are required.
