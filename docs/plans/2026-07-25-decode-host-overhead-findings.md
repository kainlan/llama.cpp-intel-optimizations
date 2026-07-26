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
| **Clean arm env** | *(no timeline variable set)* |
| **Profiled arm env** | `GGML_SYCL_TIMELINE=timeline+events`, `GGML_SYCL_TIMELINE_OUTPUT=/tmp/hostoverhead/t_<pair>.json` |
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
deliberately not reported here. **All standard deviations in this document are
sample sd (n−1 denominator)**, as is the sd of the per-pair overhead percentages;
the figures do not reproduce under a population (n) denominator.

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

Deviation from the plan: invoked with `ONEAPI_DEVICE_SELECTOR=level_zero:0`
(**B70**) rather than the plan's `level_zero:1`. The B70 is the card under
measurement and the one Task 5 will trace, and it is the card demonstrably free of
the 18:19 reset.

**Caveat on that claim:** the gate transcript prints no device-identifying line, so
the card is *not* independently verifiable from the artifact alone — it rests on the
selector passed at invocation. `gate.json`'s `device=0` does not settle it either:
that index is assigned after selector filtering, so a B50-only and a B70-only run
both log `device=0` (CLAUDE.md, "SYCL Device Selection"). Verbatim:

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
than as a systematic bias.

**Both absolute means are provisional — the clean 48.26 tok/s and the profiled
46.88 tok/s alike.** They come from the same 12 runs, and 46.88 is the number the
VERDICT hands to Tasks 5 and 6, so it inherits this caveat in full. Re-confirm both
on a quiet machine before either is used as a durable baseline. What is *not*
provisional is the 2.84 % ratio between them.

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

The mechanism, traced to the code: `ggml_backend_sycl_graph_compute`
(`ggml/src/ggml-sycl/ggml-sycl.cpp:90561`) wraps the instrumented implementation in a
`compute_impl` lambda (`:90770-90779`), and the timeline scope sits unconditionally at
the top of that implementation (`:78660`). A replayed step never calls the lambda, so
neither the `ggml.graph` span nor the per-op `compute_forward` / per-node scopes
(`:71641`, `:80195`) are ever entered — which is exactly the event profile observed.

Consequence: the trace samples the graph-**recording** step, not the graph-**replay**
steps. Task 5 must not assume `timeline.unattributed_ms` divided by token count is a
per-token cost of steady-state decode, and Task 6's attribution of "the ~22 ms/token"
must state which of the two populations it actually describes. Resolving this is
outside Task 4's scope, but proceeding without accounting for it would attribute the
wrong thing. Re-running under `GGML_SYCL_DISABLE_GRAPH=1` would instrument every
step at the cost of changing the very path being measured.

> **Decided after this section was written.** That tradeoff is no longer open: the
> owner approved the re-capture, and **Task 5 will re-capture with graph replay off
> and re-measure its own baseline** on that path. Decision recorded on tracker task
> `llama.cpp-3rzr`. The text above stands as the reasoning that prompted it.

### Which baseline Task 6 quotes against

Task 5 captures its trace **with profiling on**, and `--wall-ms` tells the parser the
true wall time of *the run it is parsing*. That run is profiled. Feeding it the clean
wall time would misattribute the profiler's own overhead into the gap classes and
inflate the residual. Task 6 therefore quotes against the **profiled** baseline, and
converts to real-world impact by discounting 2.84 % — small enough that it cannot
change which gap class dominates under the plan's >50 % decision rule.

**46.88 tok/s carries the same provisional-absolute caveat as Caveat 1** — it comes
from the identical load-contaminated 12-run series (it is `48.2551 × (1 − 0.0284)`),
measured with load running 11.75 → 27.07. Re-confirm it alongside the clean baseline
before treating it as durable. The 2.84 % **ratio** is unaffected: that is what the
interleaved paired design protects, and it is the part of this result that is settled.

~~For Task 5's arithmetic: 128 tokens ÷ 46.88 tok/s × 1000 = **2730 ms**.~~
**SUPERSEDED — do not use.** Per the decision on tracker task `llama.cpp-3rzr`
(see Caveat 4), Task 5 re-captures with graph replay **off** and measures its own
baseline on that path. A `--wall-ms` derived from the graph-**on** baseline above and
applied to a graph-**off** trace is a population mismatch that would silently corrupt
every percentage downstream. The derivation is kept visible as history only.

```
VERDICT: Task 6 quotes against the profiled baseline of 46.88 tok/s
(profiling overhead measured at 2.8% ± 2.0, t=3.54 on 5 df, significant).
```

> **Read the VERDICT with two riders**, both above and neither optional:
> 1. **46.88 tok/s is a provisional absolute** (Caveat 1 — load varied ~130 % across
>    the series). The 2.8 % overhead ratio is *not* provisional.
> 2. **The graph-on `wall_ms` derivation is superseded** — Task 5 measures its own
>    baseline with replay off (`llama.cpp-3rzr`).

---

## Task 5 — Trace capture

**Outcome: the capture FAILED its span assertion, in both arms, for a reason
neither the plan nor the tracker overrides anticipated.** No attribution is
presented below, because none is supportable from these artifacts. What *is*
presented is the root cause, the measurements that survive, and a corrected
per-token budget derived from a different instrument that did work.

### Configuration

| item | value |
|---|---|
| Card | Arc Pro B70, `level_zero:0`, PCI `0000:03:00.0` |
| Model | `/Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf` |
| Binary under test | **`baaf652e1` (build 12145)** — identical for all runs |
| Repo HEAD at capture | `f8ec00b93` (binary is one commit behind; `f8ec00b93` touched only `scripts/parse-sycl-timeline.py` and a test, and the parser runs from the working tree, so it is current) |
| Free VRAM | **32600.7 MB** on every capture run and on the `baseline_*` round; **32598.5 MB** on the `dec_*` round — both match the ~32.6 GB expectation (see "Which artifact prefix is which" below) |
| Load (1 min) | 8.5 – 10.1 across the series (Frigate ffmpeg only; no ninja/icpx) |
| Guards | `GGML_SYCL_OP_TIMEOUT_MS=180000`, `timeout` on every GPU command |

Run artifacts: `/tmp/decode-attrib/` (not committed; tmpfs, will not survive a reboot).

**Kernel log, before and after.** `journalctl -k` (`dmesg` is privilege-denied).
Before: resets at 18:19 and 19:24, both on `0000:07:00.0` (the **B50**) —
`class=ccs guc_id=10` attributed to `codescout.real`, each followed by a benign
`bcs guc_id=0 in no process [-1]` that reported `reset done`. After (first GPU run
19:45, last 19:50): **no new reset of any kind, and none ever on `0000:03:00.0`**.
The B70 numbers are valid.

### The assertion, verbatim, on both captures

Run with the lead's script against `traceEvents` (Chrome format — a parse keyed on
`events` returns 0 and looks identical to an empty capture):

```
primary   (GGML_SYCL_DISABLE_GRAPH=1)     graph_compute_impl: 1 | compute_forward: 460 | total: 3417
secondary (GGML_SYCL_DISABLE_GRAPH unset) graph_compute_impl: 1 | compute_forward: 460 | total: 3417
```

Expected for the primary: ≈129 spans (128 decode steps + 1 warmup). Observed: 1.
**Per the tracker override, this is a failed capture, not a result.**

The two arms are **identical to the event** — same span count, same op count, same
total. `GGML_SYCL_DISABLE_GRAPH` changed the artifact by exactly nothing. The
startup log confirms the variable was applied and differed between them
(`GGML_SYCL_DISABLE_GRAPH: 1` vs `GGML_SYCL_DISABLE_GRAPH: 0`), and the trap in the
tracker was respected — the secondary ran under `env -u GGML_SYCL_DISABLE_GRAPH`,
never `=0`, so `fattn.cpp:3649`'s presence-based read stayed off.

### Root cause: the flush is one-shot, and it fires on the first decode step

Graph replay is **not** the limiting mechanism. `sycl_timeline_flush`
(`ggml/src/ggml-sycl/sycl-timeline.cpp:461-486`) writes the file at most once —
`if (... || state.successful_file_flushes > 0) { return; }` — and it is called at
`ggml/src/ggml-sycl/ggml-sycl.cpp:92600-92606`:

```cpp
if (cached_is_decode && !g_ggml_sycl_graph_recording) {
    ...
    if (ggml_sycl::sycl_timeline_enabled()) {
        ggml_sycl::sycl_timeline_flush("decode-teardown");
    }
}
```

So the file is written at the end of the **first non-recording decode step** and
never again. Spans keep accumulating in memory for every later step —
`sycl_timeline_record_span_for_step` has no flush guard — but nothing ever writes
them. The artifact is structurally incapable of holding more than one decode step,
under any value of `GGML_SYCL_DISABLE_GRAPH`.

The trace's own shape confirms this exactly. In `primary-decode`:

| window | contents |
|---|---|
| 5.550 s before the graph span | 1959 `sycl.submit`, 2 `sycl.wait` (weight staging during load) |
| the single graph span (`dur` 440633 µs, `nodes=1374`) | 1 `ggml.graph`, 920 `ggml.op`, 532 `sycl.submit`, 3 `sycl.wait` |
| after the graph span | **nothing — 0 events** |

The last event in the file ends **4 µs** *before* the graph span ends. Arithmetic,
from the raw JSON: the graph span is `ts=32585790537 dur=440633`, so it ends at
`32586231170`; the latest-ending event other than the span itself is a
`compute_forward` at `ts=32586231093 dur=73`, ending at `32586231166`;
`32586231170 − 32586231166 = 4`. The `secondary-decode` trace gives the same 4 µs
(last inner event `compute_forward_node`). Compare against the *end* (`ts+dur`) of
the last inner event, not its `ts` — comparing against `ts` gives 57 µs, which is
just that event's duration plus the gap and is not the quantity meant here. That
4 µs is the flush firing inside the step, just ahead of the scope destructor at
`:78660` — which
is also why the one captured step is the 440 ms first-token step (first-use weight
materialization), not a steady-state ~20 ms one.

**Correction to the tracker's diagnosis.** Comment `c-4or8` reasoned from
`graphs reused = 384` to "SYCL graph replay bypassed the instrumented path". That
counter is llama.cpp's own ggml-graph reuse, not SYCL command-graph replay: this
session's `-p 0 -n 128` run reports `graphs reused = 128` **with
`GGML_SYCL_DISABLE_GRAPH=1` set**. The reasoning in `c-duci` about replay bypassing
`compute_impl` is correct as code description, but it is not what caps these traces.
The de-risking step ("this approach will actually work") did not hold, and the
assertion the same comment mandated is what caught it.

### A second, independent failure: zero device events on the decode path

```
timeline.wall_ms_x1000              5990322
timeline.gpu_event_total_ms_x1000         0
timeline.gpu_event_coverage_pct_x1000     0
timeline.unattributed_ms_x1000      5990322
```

(`primary-decode`; `secondary-decode` is the same with wall 6019827.) The decode
traces contain **no `sycl.event` entries at all**, so the profiler explains 0.00 %
of the wall and `unattributed` is trivially 100 %. This is not the ~15.2 % coverage
the plan expected. Device events do appear on the prompt path — the `-p 512` run
below has 1353 of them, 64.198 ms of device busy — so the emitters exist but none of
them are on the decode path.

`--wall-ms` was set to each trace's **own host-clock envelope** (5990.322 ms /
6019.827 ms), which is the only value that describes the population actually
present: 5.5 s of load-time staging plus one 440 ms first-token step. It is
deliberately **not** derived from tok/s — the trace holds neither 128 tokens nor
any steady-state token, so `128 / R × 1000` would have described a population that
is not in the file. That is the same class of error the tracker override caught in
the plan, one level down.

**Also note for whoever fixes the parser:** the parser's fallback envelope
(`parse-sycl-timeline.py:532`, used when `--wall-ms` is omitted) spans *all* events
including `sycl.event`, whose timestamps are on the device clock, a different epoch.
On the `-p 512` trace that yields `timeline.wall_ms_x1000 32493029089` — a 9-hour
"wall time" for a 6-second run. Always pass `--wall-ms` explicitly. Track A is
closed, so this is recorded, not fixed.

### Why the script could not be used as written

`scripts/sycl-gptoss-decode-timeline-profile.sh` benches `-p 512 -n 128`. The
prompt test runs first, so the one-shot flush fires on the *first token of the tg
test* and the file ends there. The script run (`/tmp/decode-attrib/primary/`)
produced 2 `graph_compute_impl` spans, both from the pp512 phase, and 1534
`compute_forward` — a trace of prompt processing, filed as a decode trace.

The script's bench args are hardcoded and this task may not modify it, and its
hardcoded `GGML_SYCL_TIMELINE_TOKEN_START=1` cannot be overridden from the
environment (the script's own `env` assignment wins), so windowing to decode was
not available either. The decode captures were therefore run by invoking
`llama-bench` directly with the script's **exact** env block and `-p 0 -n 128 -r 1
-v` — the same command Task 4 measured. `GGML_SYCL_TIMELINE_MAX_EVENTS=4000000` was
added because the 200000 default would truncate a full 129-step decode; it turned
out not to bind, since the flush stops the file long before.

### Commands (exact)

Primary (`GGML_SYCL_DISABLE_GRAPH=1`); the secondary is identical with
`env -u GGML_SYCL_DISABLE_GRAPH` in place of the `=1`:

```bash
source /opt/intel/oneapi/setvars.sh --force
timeout 1800 env GGML_SYCL_OP_TIMEOUT_MS=180000 \
  ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  GGML_SYCL_DISABLE_GRAPH=1 \
  GGML_SYCL_TIMELINE=timeline+events \
  GGML_SYCL_TIMELINE_OUTPUT=/tmp/decode-attrib/primary-decode/sycl-timeline.json \
  GGML_SYCL_TIMELINE_TOKEN_START=1 \
  GGML_SYCL_TIMELINE_MAX_EVENTS=4000000 \
  GGML_SYCL_KERNEL_PROFILE=1 \
  GGML_SYCL_KERNEL_PROFILE_OUTPUT=/tmp/decode-attrib/primary-decode/sycl-kernels \
  GGML_SYCL_KERNEL_PROFILE_FORMAT=both GGML_SYCL_KERNEL_PROFILE_RAW=1 \
  GGML_SYCL_KERNEL_PROFILE_TOP_N=80 GGML_SYCL_KERNEL_PROFILE_FLUSH=window \
  GGML_SYCL_MOE_PHASE_MATERIALIZE=1 GGML_SYCL_MOE_PHASE_BULK_XMX=1 GGML_SYCL_MOE_DOWN_SUM_DIRECT=1 \
  ./build/bin/llama-bench -m /Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf \
    -ngl 99 -fa 1 -p 0 -n 128 -r 1 -v \
  >/tmp/decode-attrib/primary-decode/bench.stdout \
  2>/tmp/decode-attrib/primary-decode/bench.stderr

# parses (W = that trace's own host-clock envelope)
python3 scripts/parse-sycl-timeline.py --wall-ms 5990.322 \
  /tmp/decode-attrib/primary-decode/sycl-timeline.json \
  >/tmp/decode-attrib/primary-decode/timeline.parse
python3 scripts/parse-sycl-timeline.py --top-gaps 20 --top-host-gap-overlaps 40 \
  --wall-ms 5990.322 /tmp/decode-attrib/primary-decode/sycl-timeline.json \
  >/tmp/decode-attrib/primary-decode/timeline.gaps.parse
python3 scripts/parse-sycl-kernel-profile.py \
  /tmp/decode-attrib/primary-decode/sycl-kernels.csv \
  >/tmp/decode-attrib/primary-decode/kernels.parse
```

Artifacts: `/tmp/decode-attrib/{primary-decode,secondary-decode,primary}/` —
`sycl-timeline.json`, `timeline.parse`, `timeline.gaps.parse`, `sycl-kernels.csv`,
`kernels.parse`, `cost-ranking.parse`, `bench.std{out,err}`.

#### Which artifact prefix is which

`/tmp/decode-attrib/` also holds two flat rounds of throughput runs, both real and
both cited in this document. They are easy to confuse, because `llama-bench`'s CSV
writes the tg row as `n_prompt=0, n_gen=128` in **both** — so a `0,128` row does not
tell you which command produced it. Distinguish them by prefix:

| prefix | command | free VRAM | cited as |
|---|---|---:|---|
| `baseline_{graphon,nograph}.{csv,log}` | `-p 512 -n 128 -r 3` | 32600.7 MB | the `-p 512 -n 128 -r 3` figures (1434.73 / 49.18 and 1418.41 / 49.01) in "Re-measured baselines" |
| `dec_{graphon,nograph}.{csv,log}` | `-p 0 -n 128 -r 3` | 32598.5 MB | the **48.3799 / 48.8345** pair in the "Re-measured baselines" table — the numbers the budget below is built on |

Neither round is a discarded attempt. The `baseline_*` round was run first, at the
profile script's own bench args, while the script was still the intended capture
vehicle; it is retained because it is the only clean `-p 512 -n 128` measurement of
the two graph arms on this binary. The `dec_*` round matches the capture command
(`-p 0 -n 128`) and is therefore the pair the per-token budget uses. The 2.2 MB VRAM
difference between the rounds is noise on a 32.6 GB card.

### Re-measured baselines (B70, `-p 0 -n 128`, `-r 3`, no profiler)

Both arms re-measured on the capture binary. Task 4's 46.88 tok/s is **not** reused
anywhere below.

| arm | tg128 tok/s | sd | ms/token | free VRAM | artifact |
|---|---:|---:|---:|---:|---|
| graph replay ON (`env -u`, log `GGML_SYCL_DISABLE_GRAPH: 0`) | **48.3799** | 0.0655 | 20.670 | 32598.5 MB | `dec_graphon.{csv,log}` |
| graph replay OFF (`=1`, log `GGML_SYCL_DISABLE_GRAPH: 1`) | **48.8345** | 0.2426 | 20.477 | 32598.5 MB | `dec_nograph.{csv,log}` |

Disabling graph replay costs **−0.94 %** on TG here (it is nominally *faster*, well
inside the spread). At `-p 512 -n 128 -r 3`: 1434.73 / 49.18 with replay on,
1418.41 / 49.01 with it off. The graph-off path is not a materially different
performance regime on this model, which is consistent with the plan's original
32.48-vs-32.81 observation — but, as established above, it is also not a different
*observability* regime, which is what this task needed.

**Profiler overhead is far larger here than Task 4's 2.84 %**: the profiled capture
runs measured 43.30 (graph off) and 42.59 (graph on) tok/s, i.e. **11.3 % / 12.0 %**
below their clean arms. Task 4's profiled arm set `GGML_SYCL_TIMELINE` only; these
runs also carry the script's `GGML_SYCL_KERNEL_PROFILE=1` block, which forces
per-kernel device profiling. Task 4's 2.84 % figure does not cover this
configuration and must not be applied to it.

### Corrected per-token budget and Amdahl ceiling

The timeline could not supply this, but the **kernel profiler is not subject to the
one-shot flush** and did cover the whole decode: its counts are exactly
129 × per-step (e.g. `mxfp4.gateup.xmx_tiled_dpas_m2` count 3096 = 24 layers ×
129 steps). Summing `total_ns` over all 12 kernels:

| capture | device busy total | per step |
|---|---:|---:|
| primary-decode (graph off) | 662.587 ms | **5.136 ms** |
| secondary-decode (graph on) | 662.679 ms | **5.137 ms** |

Agreement to 0.02 % across the two arms — further evidence replay changes nothing
about the device-side work.

Against the clean **shipping** path (graph replay ON, 20.670 ms/token):

| quantity | plan's premise | **measured** |
|---|---|---|
| total token time | (implied ~26 ms) | **20.670 ms** |
| device-busy | — | **5.136 ms** |
| non-kernel | ~22 ms | **15.533 ms** |
| non-kernel share | ~84 % | **75.2 %** |

The plan's ~22 ms is not reachable: it exceeds the whole 20.7 ms token. As the
tracker override anticipated, the figure most likely came from a `-p 512 -n 128`
token (~22.9 ms at 43.6 tok/s) with the non-kernel share taken as the residual of
~15 % device coverage. Even on that basis it would be ~17.8 ms and ~78 %, not 22 ms
and 84 %.

**Restated Amdahl ceiling**, from the measured 5.136 ms device floor:

- Eliminating **all** host time: 194.7 tok/s = **4.02×**. Unreachable, and it is a
  bound on the whole class of work, not a target.
- Eliminating **half** the non-kernel budget: 77.5 tok/s = **1.60×**.
- The plan's **+19 %** target needs 3.300 ms/token removed — **21.2 %** of the
  non-kernel budget.

So the headroom is real and large, and +19 % is not an unreasonable ask of it. What
is *not* established is where inside the 15.533 ms it sits: that was this task's
question, and these artifacts cannot answer it.

⚠ **The 5.136 ms device figure comes from the profiled runs.** If per-kernel
profiling inflates kernel durations, kernel time is overstated and the non-kernel
budget is correspondingly *understated* — 75.2 % is a conservative floor, not a
ceiling. It is also a sum over per-kernel `total_ns` on a single in-order compute
queue; it would overcount if work were concurrent across queues.

### What Task 6 must not do

1. **Do not attribute anything from these traces.** The primary failed its span
   assertion, device coverage is 0 %, and the one step actually captured is the
   440 ms first-token step — the least representative step in the run.
2. **Do not carry `~22 ms` or `~84 %` forward.** Use 15.533 ms / 75.2 %, or restate
   from measurement.
3. **The `host_overlap` caveat from the tracker still stands, and is stronger than
   stated.** It was argued from graph replay skipping the per-op host path. That
   argument is wrong in its mechanism (see above), but the conclusion is unchanged
   for a different reason: there is no decode-path gap-class measurement here at
   all, in either arm.

**To make this measurable**, one of the following has to change first — all are
source changes, out of scope for Task 5 and for closed Track A:

- make `sycl_timeline_flush` re-entrant, or move the decode-teardown flush to
  process/context teardown so the buffer is written once at the end rather than
  once at the beginning;
- and emit `sycl.event` device events on the decode path, without which coverage is
  0 % however many steps are captured;
- optionally, honour `GGML_SYCL_TIMELINE_TOKEN_START`/`_COUNT` as a decode-step
  window so a steady-state slice can be isolated without buffering all 128 steps.

---

## Task 6 — Attribution

**Outcome: the attribution could not be performed, and no class share is stated
below.** Task 6 exists to split the per-token non-kernel cost across
`host_overlap` / `queue_serialization` / `runtime_idle` and name a dominant
class. Task 5's capture cannot support that split — not weakly, not
provisionally, not "indicatively". This section records the corrected budget
that *is* supported, states the verdict as *not determined*, and points at the
ticket that has to land before anyone tries again.

This task ran **no GPU work**: it is analysis of Task 5's artifacts in
`/tmp/decode-attrib/` plus the sections above.

### Why there is nothing to attribute

Three independent facts, each sufficient on its own.

**1. The decode traces contain no gap-class metrics at all.** Not small ones —
none. Both decode parses emit exactly four `timeline.*` lines and zero
`gap.*` / `gap_class.*` lines:

```
$ grep -c '^gap_class\.' /tmp/decode-attrib/primary-decode/timeline.gaps.parse    -> 0
$ grep -c '^gap_class\.' /tmp/decode-attrib/secondary-decode/timeline.gaps.parse  -> 0
```

The plan's Task 6 step 1 (`grep -E '^gap_class\.' ... | sort`) returns an empty
set on both arms. There is no queue total to take a percentage of, no three-way
split to check for exhaustiveness, and no callsite / gap-transition / category
ranking to produce. Steps 2 and 3 of the implementation guide have no input.

**2. Device-event coverage on the decode path is 0.00 %.** Verbatim from
`primary-decode` (`secondary-decode` identical but for wall `6019827`):

```
timeline.wall_ms_x1000              5990322
timeline.gpu_event_total_ms_x1000         0
timeline.gpu_event_coverage_pct_x1000     0
timeline.unattributed_ms_x1000      5990322
```

`unattributed` equals `wall` to the digit because the profiler explains nothing.
The gap classifier derives gaps *between device events*; with zero `sycl.event`
entries there are no gaps, which is precisely why fact 1 holds. Expressing
anything "as a percentage of `timeline.unattributed_ms_x1000`" would be dividing
into a quantity that is 100 % residual by construction.

**3. One decode step was captured, and it is the worst possible one.** Both
arms, to the event:

```
primary   (GGML_SYCL_DISABLE_GRAPH=1)     graph_compute_impl: 1 | compute_forward: 460 | total: 3417
secondary (GGML_SYCL_DISABLE_GRAPH unset) graph_compute_impl: 1 | compute_forward: 460 | total: 3417
```

Expected for the primary: ≈129 spans (128 decode steps + 1 warmup). Observed: 1.
Per the tracker override on `llama.cpp-3rzr`, a 1-span capture is a **failed
capture, not a result**. The one step present is the ~440 ms first-token step
(`dur` 440633 µs), which pays first-use weight materialization and is the least
representative step in a 20.670 ms/token run — roughly 21× a steady-state token.
Any share computed from it would describe first-token materialization while
being labelled steady-state decode.

**No class share has been computed from these traces, in any form.** Not for the
primary arm, not for the secondary, not "for orientation". Facts 1–3 are the
whole of what the decode artifacts support.

### Verdict

```
DOMINANT CLASS: NOT DETERMINED — no dominant class was identified and no class
share was computed. 0.0% of timeline.unattributed_ms_x1000 is attributed
(0.000 ms/token of the 15.533 ms/token non-kernel budget measured, out of
20.670 ms/token total). The decode capture emitted zero gap_class.* metrics on
zero queues at 0.00% device-event coverage, over 1 of ~129 decode steps. This
is a failed measurement, not a finding of "no dominant class in the data".
CONFIDENCE: LOW — and strictly weaker than LOW: there is no classification to
rate. The rounding_delta >5% rule could not be applied, because no
gap_class.*.rounding_delta_ms_x1000 line exists for the decode path. Task 7
must take its LOW CONFIDENCE branch (write no fix plan; re-run the Task 5
capture) on the strength of the missing measurement, not of a weak one.
```

The `LOW` token is kept deliberately so Task 7's decision table matches without
interpretation; read the second sentence for what it actually means.

### The budget that does survive, carried forward unchanged from Task 5

These figures are **not** recomputed or re-derived here — they are Task 5's, from
the kernel profiler, which is not subject to the one-shot flush and covered all
129 steps. Measured against the clean **shipping** path (graph replay ON,
48.3799 tok/s):

| quantity | value |
|---|---:|
| total token time | **20.670 ms** |
| device-busy | **5.136 ms** |
| non-kernel | **15.533 ms** |
| non-kernel share | **75.2 %** |

Restated Amdahl ceiling from the 5.136 ms device floor:

- Eliminating **all** host time: 194.7 tok/s = **4.02×** — a bound on the whole
  class of work, not a target, and unreachable.
- Eliminating **half** the non-kernel budget: 77.5 tok/s = **1.60×**.
- The plan's **+19 %** target needs 3.300 ms/token removed = **21.2 %** of the
  non-kernel budget.

Both caveats travel with the numbers and are not optional:

⚠ **75.2 % is a conservative floor, not a ceiling.** The 5.136 ms device figure
comes from the *profiled* runs. If per-kernel profiling inflates kernel
durations, kernel time is overstated and the non-kernel budget is correspondingly
*understated*.

⚠ **It is a sum over per-kernel `total_ns` on a single in-order compute queue.**
It would overcount device busy — and therefore further understate the non-kernel
share — if work were concurrent across queues.

So the headroom is real and large, and +19 % is not an unreasonable ask of it.
**Where inside the 15.533 ms it sits remains unknown**, and that is exactly the
question Task 6 was opened to answer.

### Retired: the plan's `~22 ms` / `~84 %` premise

**Do not carry `~22 ms/token` or `~84 %` forward. Both are dead.** `~22 ms`
exceeds a whole 20.670 ms token, so it cannot be the non-kernel *part* of one;
it is arithmetically impossible on this path, not merely imprecise. The
replacements are **15.533 ms** and **75.2 %** above.

As Task 5 established, the original figure most plausibly came from a
`-p 512 -n 128` token (~22.9 ms at 43.6 tok/s) with the non-kernel share taken
as the residual of ~15 % device coverage — and even on that basis it would be
~17.8 ms and ~78 %, not 22 ms and 84 %. Any downstream document, task, or
acceptance criterion still quoting `~22 ms` or `~84 %` is quoting a superseded
premise.

### The rounding-delta rule: enforceable, unexercised, and its scaling hazard

Per the lead's comment `c-xhk3` on `llama.cpp-rn3e`, Task 3 (commit `f8ec00b93`)
added `gap_class.device{N}.{queue}.rounding_delta_ms_x1000` — the
**pre-correction** magnitude of the time the three-way split failed to explain,
which the parser then folds into `runtime_idle` to force the classes to sum to
the queue total. `runtime_idle` as printed is therefore inflated by exactly that
delta, and `runtime_idle` is the class the plan treats as the smoking gun for the
L0/UR submit path. The plan's rule — *any `rounding_delta` above 5 % of its
queue's gap total ⇒ `LOW CONFIDENCE`* — is now enforceable, where before the
magnitude was not observable at all.

**It could not be exercised here.** There is no `rounding_delta` line on the
decode path, because there is no gap-class data on the decode path (fact 1). The
rule is ready for whoever re-runs this capture; it simply had nothing to be
applied to.

**Carry the scaling hazard `c-xhk3` flagged.** The delta accumulates one
sub-microsecond rounding *per gap*, so its size tracks gap count. On Task 3's
2-gap synthetic fixture it is 2 (`_x1000` units, i.e. 0.002 ms). The only real
trace in this session's artifacts that carries the metric is the **prompt-path**
`-p 512` trace (`/tmp/decode-attrib/primary/`), where it reads:

| queue | gap count | `rounding_delta_ms_x1000` | queue gap total `_x1000` | delta as % of queue total |
|---|---:|---:|---:|---:|
| `device0.compute` | 968 | 52 | 1518482 | 0.0034 % |
| `device0.copy` | 382 | 5 | 1579085 | 0.0003 % |

Read that table for one thing only: **the metric is live and its observed
magnitude on a ~1000-gap real trace is three orders of magnitude below the 5 %
threshold.** No class share is read from that trace here — it is a prompt trace,
not decode, and it is not an attribution. A full-length decode trace will carry
many more gaps than 968, so recompute the percentage per queue rather than
assuming it stays negligible; the point of the metric is that the assumption no
longer has to be made.

### What has to land first: `llama.cpp-sacs`

The two source-level defects that made this measurement impossible are tracked
together as **`llama.cpp-sacs`**. Task 5 described them; this is the tracker id
that owns the fix, which the Task 5 write-up did not name:

1. **The timeline flush is one-shot.** `sycl_timeline_flush`
   (`ggml/src/ggml-sycl/sycl-timeline.cpp:461`) returns early once
   `state.successful_file_flushes > 0`, and is called at
   `ggml/src/ggml-sycl/ggml-sycl.cpp:92600-92606` under
   `if (cached_is_decode && !g_ggml_sycl_graph_recording)` — the end of the
   **first** non-recording decode step. Spans keep accumulating
   (`sycl_timeline_record_span_for_step` has no flush guard) but nothing writes
   them. A decode artifact is structurally incapable of holding more than one
   step, under any value of `GGML_SYCL_DISABLE_GRAPH`.
2. **The decode path emits zero `sycl.event` device events.** Fixing (1) alone
   would still attribute nothing: coverage would remain 0.00 % over 129 steps
   instead of over 1.

Both are source changes, out of scope for closed Track A. Until `llama.cpp-sacs`
is resolved, **re-running the Task 5 capture unchanged will reproduce this same
non-result** — the failure is in the instrument, not in the run.

The re-run, when it happens, needs: the flush made re-entrant or moved to
context/process teardown; `sycl.event` emission on the decode path; and
optionally `GGML_SYCL_TIMELINE_TOKEN_START`/`_COUNT` honoured as a decode-step
window so a steady-state slice can be isolated without buffering 128 steps. It
should also skip the first-token step explicitly — that step is 21× a
steady-state token and would dominate any unwindowed average.
