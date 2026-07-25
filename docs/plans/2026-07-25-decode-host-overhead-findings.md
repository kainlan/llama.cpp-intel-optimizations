# SYCL Decode Host-Overhead Attribution — Findings

Companion to `docs/plans/2026-07-25-sycl-decode-host-overhead-attribution.md`.
Each task appends its own section; do not rewrite earlier ones.

---

## Task 4 — Observer-effect baseline

**Question.** `GGML_SYCL_TIMELINE=timeline+events` forces per-event profiling and
wraps every submit in a `steady_clock` pair. Is the ~22 ms/token non-kernel decode
cost the *clean* cost or the *profiled* cost? Every percentage Task 6 quotes is
meaningless until this is fixed.

### Configuration

| item | value |
|---|---|
| Card | Arc Pro B70, `level_zero:0`, PCI `0000:03:00.0` |
| Model | `/Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf` |
| Command | `llama-bench -p 0 -n 128 -r 3 -v -o csv` |
| Guards | `GGML_SYCL_OP_TIMEOUT_MS=180000`, `timeout 900` per run |
| Binary under test | **`38c6c52cc` (build 12135)** — identical for all 12 runs |
| Repo HEAD at capture | `ba6cec314` |
| Design | 6 pairs, clean/profiled alternating **within** each pair |

Run artifacts: `/tmp/hostoverhead/` (not committed; tmpfs, will not survive a reboot).

**Preflight.** Kernel log checked with `journalctl -k` (`dmesg` is privilege-denied
for this user — a silent `dmesg` failure is indistinguishable from a clean log).
The only GT reset in the window was at 18:19 on `0000:07:00.0`, the **B50** — a
`ccs` CAT error `guc_id=10` attributed to `codescout.real`, followed by a benign
`bcs` `guc_id=0` `in no process [-1]` reset that reported `reset done`. The B70
(`0000:03:00.0`) shows **no reset of any kind**; its only kernel-log appearances
are P2P-topology warnings, which are diagnostic. Free VRAM was 32600.7 MB on every
single run, matching the ~32.6 GB expectation — no ComfyUI-class tenancy.

### The twelve TG128 values

Free VRAM was **32600.7 MB on all 12 runs**. Load average is captured immediately
before each run.

| pair | arm | tg128 (tok/s) | free VRAM (MB) | load (1 min) | start |
|---:|---|---:|---:|---:|---|
| 1 | clean | 48.893676 | 32600.7 | 11.75 | 18:42:33 |
| 1 | profiled | 47.339936 | 32600.7 | 14.40 | 18:42:53 |
| 2 | clean | 48.091690 | 32600.7 | 13.18 | 18:43:12 |
| 2 | profiled | 47.846896 | 32600.7 | 12.25 | 18:43:32 |
| 3 | clean | 48.030176 | 32600.7 | 15.25 | 18:43:51 |
| 3 | profiled | 46.555275 | 32600.7 | 16.68 | 18:44:11 |
| 4 | clean | 48.539602 | 32600.7 | 22.46 | 18:44:31 |
| 4 | profiled | 45.493383 | 32600.7 | 24.48 | 18:44:50 |
| 5 | clean | 47.999983 | 32600.7 | 27.02 | 18:45:10 |
| 5 | profiled | 47.236794 | 32600.7 | 26.89 | 18:45:30 |
| 6 | clean | 47.975224 | 32600.7 | 25.19 | 18:45:50 |
| 6 | profiled | 46.806277 | 32600.7 | 27.07 | 18:46:09 |

### Summary statistics

Computed **across runs**. `llama-bench`'s own `±` is within-process and is
deliberately not reported here.

| arm | mean | sd | range | cv |
|---|---:|---:|---|---:|
| clean | **48.2551** | 0.3767 | [47.9752, 48.8937] | 0.78 % |
| profiled | **46.8798** | 0.8133 | [45.4934, 47.8469] | 1.73 % |

Paired differences (clean − profiled), one per pair:

| pair | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| Δ tok/s | 1.5537 | 0.2448 | 1.4749 | 3.0462 | 0.7632 | 1.1689 |
| overhead % | 3.178 | 0.509 | 3.071 | 6.276 | 1.590 | 2.437 |

- mean Δ = 1.3753 tok/s, sd = 0.9517, se = 0.3885
- **paired t = 3.540 on 5 df, two-sided p = 0.017 → significant at α = 0.05** (not at 0.01)
- profiling overhead = **2.84 % ± 1.96** (mean ± sd of the per-pair percentages), range [0.51 %, 6.28 %]

The sign is consistent: clean beat profiled in **6 of 6** pairs.

### Correctness gate

Run with profiling **ON**, so the gate covers the instrumented path.

Deviation from the plan: run on `level_zero:0` (**B70**) rather than the plan's
`level_zero:1`. The B70 is the card under measurement and the one Task 5 will
trace, and it is the card demonstrably free of the 18:19 reset. Verbatim:

```
> Count from 1 to 5. Answer with only: 1, 2, 3, 4, 5
1, 2, 3, 4, 5
```

**PASS.** Note the plan's step 4 states the expected output starts `: 1, 2, 3, 4, 5`.
That leading colon is the tail of the *echoed prompt*, captured before
`--no-display-prompt` existed; with that flag the echo lands on the `> ` line and
the answer is the next line on its own. Gate on the digit sequence — grepping for
the colon form makes a passing gate read as a failure.

### Caveats

**1. Load varied by ~130 % across the series — the absolute baseline is PROVISIONAL.**
Load ran 11.75 → 27.07 within the 12 runs (and reached 32.23 during the addendum),
far beyond the ~30 % band that would let the absolute number stand unqualified.
The cause is identified: Frigate ffmpeg plus a **concurrent CMake/ninja rebuild by
another implementer**, which relinked `build/bin/` at 18:46. The interleaved paired
design is exactly what protects against this, and it did: the *ratio* (2.84 %) is
sound, and the load spike at pair 4 shows up as that pair's outlying 6.28 % rather
than as a systematic bias. The *absolute* 48.26 tok/s should be treated as
provisional and re-confirmed on a quiet machine before being used as a durable
baseline.

**2. The binary changed mid-session, but not mid-series.** All 12 runs report
`build_commit=38c6c52cc (12135)`; `llama-bench` re-run after the 18:46 relink
reports `ba6cec314 (12138)`. The relink therefore completed after the last run's
`exec`, so the paired comparison is a clean within-binary contrast. To check
whether the two intervening SYCL commits (`01d4991c4`, `0e9a48dc3`) moved the
absolute number, three extra **clean** runs on the current binary:

| run | tg128 | build | free VRAM (MB) | load |
|---:|---:|---|---:|---:|
| 1 | 47.893843 | `ba6cec314` | 32600.7 | 32.23 |
| 2 | 48.225852 | `ba6cec314` | 32600.7 | 29.69 |
| 3 | 48.748621 | `ba6cec314` | 32600.7 | 28.98 |

mean 48.2894, sd 0.4309 — **+0.07 % vs the 38c6c52cc clean arm, i.e. indistinguishable.**
The baseline below therefore carries over to the current binary.

**3. This baseline is ABOVE the documented B70 row, and that is expected.**
`docs/backend/sycl-perf-baselines.md` row A gives B70 GPT-OSS 20B TG128 as
43.57 ± 1.46, range [40.18–46.27] over 21 runs. The 48.26 measured here sits above
that upper bound. The configurations differ: the baseline row is a `-p 512 -n 128`
run, while this plan mandates `-p 0 -n 128`. **Task 5 must derive `--wall-ms` from
the number in this document, not from the baselines doc** — same command, same
card, same session. This is not a regression (it is faster, not slower), and the
baselines doc is not amended on the strength of a load-contaminated series.

**4. ⚠ Blocking caveat for Tasks 5 and 6: the trace instruments ~1 of 386 decode
steps.** The pair-1 timeline JSON (`/tmp/hostoverhead/t_1.json`, 305 KB) contains
only **926 events** — not the ~461 launches/token × 384 tokens the plan's framing
assumes:

| cat | count |
|---|---:|
| `ggml.op` | 920 (460 × `compute_forward`, 460 × `compute_forward_node`) |
| `sycl.wait` | 5 |
| `ggml.graph` | **1** (`graph_compute_impl`, `dur` 442343 µs, `nodes=1374`) |

This is not a truncation artifact: the defaults are `token_count=0` (no window
restriction) and `max_events=200000` (`sycl-timeline.hpp:20-22`), neither of which
binds here. The cause is **SYCL graph replay** — the same run reports
`graphs reused = 384` out of 386 tokens. Only the single non-replayed
graph-recording step enters the instrumented per-op path; the 384 replay steps that
constitute the overwhelming majority of decode emit no spans at all.

Consequence: the trace samples the graph-**recording** step, not the graph-**replay**
steps. Task 5 must not assume `timeline.unattributed_ms` divided by token count is a
per-token cost of steady-state decode, and Task 6's attribution of "the ~22 ms/token"
must state which of the two populations it actually describes. Resolving this is
outside Task 4's scope, but proceeding without accounting for it would attribute the
wrong thing. Re-running under `GGML_SYCL_DISABLE_GRAPH=1` would instrument every
step at the cost of changing the very path being measured — a tradeoff for Task 5 to
decide, not a fix to apply blindly.

### Which baseline Task 6 quotes against

Task 5 captures its trace **with profiling on**, and `--wall-ms` tells the parser the
true wall time of *the run it is parsing*. That run is profiled. Feeding it the clean
wall time would misattribute the profiler's own overhead into the gap classes and
inflate the residual. Task 6 therefore quotes against the **profiled** baseline, and
converts to real-world impact by discounting 2.84 % — small enough that it cannot
change which gap class dominates under the plan's >50 % decision rule.

For Task 5's arithmetic: 128 tokens ÷ 46.88 tok/s × 1000 = **2730 ms**. Recompute
for the actual token count the profile script uses; do not reuse this number blindly.

```
VERDICT: Task 6 quotes against the profiled baseline of 46.88 tok/s
(profiling overhead measured at 2.8% ± 2.0, t=3.54 on 5 df, significant).
```
