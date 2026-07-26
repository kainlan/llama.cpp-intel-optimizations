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

---

## Task 7 — Phase-2 selection

**Outcome: no phase-2 plan was written, and none should be.** Task 7's design is
that the branch is chosen by a rule fixed *in advance* in the plan's decision
table, applied verbatim to Task 6's `DOMINANT CLASS:` line, so the choice cannot
be rationalized after seeing results. The rule is applied below. It selects the
no-fix-plan branch, and it does so twice over.

This task ran no GPU work, no build, and touched no source, script, or test.

### The selection

```
PHASE-2 SELECTION: none — Task 6 is marked CONFIDENCE: LOW and states DOMINANT
CLASS: NOT DETERMINED, so the decision table's LOW CONFIDENCE row applies and no
fix plan is written. The Task 5 *timeline* capture must be re-run first, which is
blocked on llama.cpp-sacs; a separate callsite-level host attribution from the
kernel profiler's raw_events is NOT blocked and can proceed now (see "The
raw_events path is already open"). Neither changes the selection.
RULE APPLIED: "| Task 6 marked `LOW CONFIDENCE` | **Write no fix plan.** Re-run
Task 5 capture first; record why. | — |" (decision table,
docs/plans/2026-07-25-sycl-decode-host-overhead-attribution.md, "### Task 7" step
1). The adjacent row "| No class > 50% | **Write no fix plan.** ... | — |" leads
to the same outcome, so the selection does not depend on which reading of the
verdict is taken.
SCOPE SOURCE: none — no fix scope was drawn, because the branch taken writes no
fix plan. The verdict itself was read from Task 6's "### Verdict" block
(DOMINANT CLASS / CONFIDENCE lines) in this document; the underlying artifacts
are the Task 5 decode parses under /tmp/decode-attrib/{primary,secondary}-decode/,
which emit zero gap_class.* metrics.
```

### Which row matched, and why the decision is robust

Task 6's verdict, quoted from its `### Verdict` block above:

```
DOMINANT CLASS: NOT DETERMINED — no dominant class was identified and no class
share was computed. 0.0% of timeline.unattributed_ms_x1000 is attributed
[...] This is a failed measurement, not a finding of "no dominant class in the
data".
CONFIDENCE: LOW — and strictly weaker than LOW: there is no classification to
rate.
```

Two rows of the decision table can be reached from that text, and **both point to
the same branch**:

| row, quoted verbatim from the plan | matches because |
|---|---|
| `Task 6 marked LOW CONFIDENCE` → **Write no fix plan.** Re-run Task 5 capture first; record why. | The literal token `CONFIDENCE: LOW` is present. Task 6 retained that token deliberately so this table matches without interpretation. |
| `No class > 50%` → **Write no fix plan.** Instead append a `## No dominant class` section recommending a finer capture (per-op `GGML_SYCL_KERNEL_PROFILE_RAW=1` host-submit spans) as the next attribution step. | No class exceeds 50 %; no class has any share at all. |

The three `> 50%` rows (`host_overlap`, `runtime_idle`, `queue_serialization`)
are each unreachable: every measured share is absent, not merely below threshold.

**No phase-2 plan document exists for any branch** — not
`...-phase2-per-op-memoization.md`, not `...-phase2-launch-count-reduction.md`,
not `...-phase2-dependency-overlap.md`, and not a provisional or skeleton form of
any of them. The plan's own gotcha is explicit that writing plans which then get
thrown away is the waste this scope boundary exists to prevent, and here there is
no gap-class measurement to scope one from in the first place. Note that the
`raw_events` data described below does **not** close this: each fix branch is
scoped from a different Task 6 output (the callsite table, the launch count, the
`gap_transition` table), and a submit-time-by-callsite ranking does not select
among the three.

The plan's step-1 branch for `No class > 50%` also asks for a `## No dominant
class` section recommending a finer capture — per-op
`GGML_SYCL_KERNEL_PROFILE_RAW=1` host-submit spans. **That recommendation is
adopted**, and it turns out to be stronger than "finer": see
"The `raw_events` path is already open" below. It is a *different instrument*,
not a finer setting on the broken one, and the data it produces is already on
disk.

### The distinction that matters most for whoever picks this up

**This is a failed measurement, not a finding that the non-kernel cost is
diffuse.** The two route to the same branch of the decision table and mean
opposite things:

| | what the data would say | what it licenses |
|---|---|---|
| *cost is diffuse* (a real finding) | every class was measured; none reached 50 % | broad optimization is justified — attack several classes, or find a finer decomposition |
| *measurement failed* (**this case**) | no class was measured; coverage was 0.00 % over 1 of ~129 steps | **nothing**. The cost may well be concentrated in a single class. We do not know. |

A reader who concludes "the cost is spread evenly, so optimize broadly" is acting
on a false premise. So is a reader who concludes any class *isn't* dominant.
Nothing was ruled in and nothing was ruled out. The three-way split remains
entirely open, and the 15.533 ms/token it would divide is real and large.

### The `raw_events` path is already open

**`llama.cpp-sacs` is not a hard prerequisite for all further attribution.** The
kernel profiler is a separate instrument from the timeline, and — as Task 5
established when it salvaged the 5.136 ms device figure — **it is not subject to
the one-shot flush**. It covered all 129 steps. Its raw per-event output was
captured alongside the failed timeline traces and is sitting in the same
directory, unread.

`/tmp/decode-attrib/primary-decode/sycl-kernels.json` has two top-level keys,
`kernels` and **`raw_events`**. ⚠ **`/tmp` is tmpfs and is not guaranteed to
survive a reboot** — if that directory is gone, re-run the capture commands under
"Commands (exact)" above, which reproduce it with no source change and no
`llama.cpp-sacs` fix. The `raw_events` figures below, verified directly from the
artifact:

| property | value |
|---|---|
| records | **61,500** = **476.7 per step** across 129 steps — the whole decode, not one step |
| host-submit span | 8.948 s — the whole run |
| usable timestamps | `timestamp_status: ok` on **59,541 of 61,500 = 96.8 %** |
| the 1,959 failures | all `sycl.memcpy.mem_ops` on the **copy** queue, all in the load-time staging window (host-submit begin 32580.2–32585.8 s, entirely before the first `ok` event at 32585.8 s) — i.e. weight staging, not decode |
| per-event fields | `host_submit_begin_us`, `host_submit_end_us`, `device_submit_ns`, `device_start_ns`, `device_end_ns`, `duration_ns`, `queue_kind`, `category`, `graph_recorded`, and **`file` / `line` / `function`** |
| total host-submit time | **312.429 ms = 2.422 ms/step**, resolved across **14 distinct callsites** |

The ranked callsites, per step:

| count | per step | callsite |
|---:|---:|---|
| 9288 | 72.0 | `binbcast.cpp:972 ggml_sycl_op_mul` |
| 9288 | 72.0 | `binbcast.cpp:888 operator()` |
| 9288 | 72.0 | `binbcast.cpp:962 ggml_sycl_op_add` |
| 6192 | 48.0 | `rope.cpp:640 ggml_sycl_op_rope` |
| 6192 | 48.0 | `set_rows.cpp:728 set_rows_sycl` |
| 3398 | 26.3 | `mem-ops.cpp:356 mem_copy_submit` |
| 3225 | 25.0 | `mem-ops.cpp:454 mem_fill_submit` |
| 3096 | 24.0 | `softmax.cpp:417 ggml_sycl_op_soft_max` |
| 3096 | 24.0 | `mmvq.cpp:17473 mmvq_moe_batched_dispatch_pair_glu_mxfp4_soa` |
| 3096 | 24.0 | `mmvq.cpp:6849 mxfp4_dpas_pack_q8_single_col_groups_sycl` |
| 3096 | 24.0 | `mmvq.cpp:9841 mxfp4_pair_glu_xmx_tiled_dpas_m2_sycl` |
| 1729 | 13.4 | `mem-ops.cpp:391 mem_copy_submit` |
| 387 | 3.0 | `getrows.cpp:2898 ggml_sycl_op_get_rows` |
| 129 | 1.0 | `mem-ops.cpp:382 mem_copy_submit` |

So a per-op, whole-decode, **source-attributed** host-side measurement exists
today, on artifacts already captured, without `llama.cpp-sacs` being fixed first.

**Three limits on it, none optional:**

1. **This is not Task 6's decomposition.** It does **not** reproduce the
   `host_overlap` / `queue_serialization` / `runtime_idle` three-way gap
   classification. It is a different cut — submit-call *duration* per op, with
   source location — and it does not satisfy Task 6 or change its verdict. The
   gap classification still requires the timeline, and the timeline still
   requires `llama.cpp-sacs`.
2. **These numbers come from a profiled run.** Profiler overhead in this capture
   configuration (`GGML_SYCL_TIMELINE` **and** `GGML_SYCL_KERNEL_PROFILE`) is
   **11.3–12.0 %** per Task 5 — *not* Task 4's 2.84 %, which covered
   `GGML_SYCL_TIMELINE` alone and must not be applied here.
3. **2.422 ms/step is host *submit* time specifically.** It is a component of the
   15.533 ms non-kernel budget, **not** the whole of it. **What fraction of the
   remaining ~13.1 ms it explains has not been established, and must not be
   claimed.** Submit time is not the same quantity as the gap between device
   events, and the two cannot be equated without the measurement that is missing.

### What must happen before this question can be re-asked

**Two paths. One is blocked on the instrument fix; one is open now.** They
answer different questions and can proceed in parallel.

**Path A — callsite-level host attribution (open now, no blocker).** Analyse
`raw_events` as described above. Requires no source change and no
`llama.cpp-sacs` fix. It bounds and localizes host *submit* work by source line.
It cannot produce the three-way gap classification. No new GPU run is needed
**while the artifact exists** — `/tmp` is tmpfs, so if it has been lost to a
reboot the capture must be re-run first (same commands, still no source change).

**Path B — the three-way gap classification (blocked).** This is the question
Task 6 was opened to answer, and it needs the timeline. In order; skipping any
step reproduces the same non-result.

1. **Resolve `llama.cpp-sacs` — both defects, not just the first.**
   - *Defect 1:* the timeline flush is one-shot. `sycl_timeline_flush`
     (`ggml/src/ggml-sycl/sycl-timeline.cpp:461`) returns early once
     `state.successful_file_flushes > 0`, called at
     `ggml/src/ggml-sycl/ggml-sycl.cpp:92600-92606` at the end of the **first**
     non-recording decode step. A decode artifact is structurally incapable of
     holding more than one step.
   - *Defect 2:* the decode path emits zero `sycl.event` device events.
     **Fixing only defect 1 still attributes nothing** — coverage stays at
     0.00 %, just over 129 steps instead of 1. The gap classifier derives gaps
     *between device events*; with none, `queue_gaps` is empty and the parser
     emits no `gap_class.*` lines at all. That causal chain is verified, not
     assumed.
   - Optional but wanted: honour `GGML_SYCL_TIMELINE_TOKEN_START`/`_COUNT` as a
     real decode-step window, so a steady-state slice can be isolated without
     buffering 128 steps. Note the profile script hardcodes
     `GGML_SYCL_TIMELINE_TOKEN_START=1` in its own `env` block, which the
     caller's environment cannot override.
2. **Re-run the Task 5 capture** on the repaired instrument, and **skip the
   first-token step explicitly** — it is ~440 ms, roughly 21× a steady-state
   token, and would dominate any unwindowed average.
3. **Re-run Task 6's attribution** against that capture, including the
   rounding-delta check (below), which was never exercisable on decode.

Until step 1 lands, re-running the *timeline* capture unchanged reproduces this
same non-result: **the failure is in the instrument, not in the run.** That is a
statement about path B only — it does not block path A.

### What phase 1 established anyway

The plan did not come away empty. Two things are settled and outlive it.

**1. The corrected per-token budget** (carried forward verbatim from Task 6,
which carried it verbatim from Task 5 — not recomputed here). Measured against
the clean **shipping** path, graph replay ON, 48.3799 tok/s, B70, GPT-OSS 20B
MXFP4, `-p 0 -n 128`:

| quantity | value |
|---|---:|
| total token time | **20.670 ms** |
| device-busy | **5.136 ms** |
| non-kernel | **15.533 ms** |
| non-kernel share | **75.2 %** |

Both caveats travel with these numbers and are not optional:

⚠ **75.2 % is a conservative floor, not a ceiling.** The 5.136 ms device figure
comes from the *profiled* runs. If per-kernel profiling inflates kernel
durations, kernel time is overstated and the non-kernel budget is correspondingly
*understated*.

⚠ **It is a sum over per-kernel `total_ns` on a single in-order compute queue.**
It would overcount device busy — and therefore further understate the non-kernel
share — if work were concurrent across queues.

**2. The restated Amdahl ceiling**, from the 5.136 ms device floor:

- Eliminating **all** host time: 194.7 tok/s = **4.02×** — a bound on the whole
  class of work, not a target, and unreachable.
- Eliminating **half** the non-kernel budget: 77.5 tok/s = **1.60×**.
- The plan's **+19 %** target needs 3.300 ms/token removed = **21.2 %** of the
  non-kernel budget.

So the headroom is real and large, and +19 % is not an unreasonable ask of it —
conditioned entirely on *where* inside the 15.533 ms the time sits, which is the
question that remains unanswered.

**3. The plan's `~22 ms` / `~84 %` premise is retired.** `~22 ms` exceeds a whole
20.670 ms token, so it cannot be the non-kernel *part* of one — arithmetically
impossible on this path, not merely imprecise. Any downstream document, task, or
acceptance criterion still quoting `~22 ms` or `~84 %` is quoting a superseded
premise; the replacements are **15.533 ms** and **75.2 %**.

### State of the instrumentation

Track A is closed and final. It delivered three working pieces that outlive this
plan; whoever re-runs the capture inherits all of them.

| piece | commit(s) | state |
|---|---|---|
| The decode-timeline script test, registered in ctest (`test-sycl-decode-timeline-profile-script`) | `9125985a1`, `3845dcd62`, `51d4dd489` | Live. Previously `ctest -N -R decode-timeline` reported `Total Tests: 0`; the assertions existed but nothing ran them. |
| `--wall-ms` forwarding from `scripts/sycl-gptoss-decode-timeline-profile.sh` to both `parse-sycl-timeline.py` invocations | `3f2dbb54f`, `baaf652e1` | Live, and load-bearing: the parser's fallback envelope mixes host- and device-clock epochs (`parse-sycl-timeline.py:532`), reporting a 9-hour wall for a 6-second run. **Always pass `--wall-ms` explicitly** until that is fixed (defect 3 of `llama.cpp-sacs`). |
| `gap_class.device{N}.{queue}.rounding_delta_ms_x1000` plus its conservation test (`test-sycl-timeline-gap-class-conservation`) | `f8ec00b93` | Live. Makes the plan's *"any `rounding_delta` above 5 % of its queue's gap total ⇒ `LOW CONFIDENCE`"* rule **enforceable** where the magnitude was previously not observable at all. |

**The rounding-delta rule was never exercised on decode** — there is no
`rounding_delta` line on the decode path, because there is no gap-class data on
the decode path. It is ready and waiting for the re-run. Carry its scaling
hazard: the delta accumulates one sub-microsecond rounding *per gap*, so its size
tracks gap count, and a full-length decode trace will carry far more gaps than
the 968 on the prompt-path trace where it was observed at 0.0034 % of the queue
total. Recompute the percentage per queue rather than assuming it stays
negligible — the point of the metric is that the assumption no longer has to be
made.

---

## Task 8 (`llama.cpp-sacs`) — the decode capture works; the blocker is cleared

Commits `4457a87c2` (fix) and `7d3374553` (test). Both defects the plan recorded
turned out to be **one defect with two faces**, as the tracker's root-cause
comment `c-hxfe` diagnosed: `sycl_timeline_flush()` is one-shot, and it was being
called from `ggml_backend_sycl_graph_compute()`, which runs **once per decode
step**. That made it first-step-wins. It also locked out the flush in
`ggml_backend_sycl_free()`, which is the only one ordered *after*
`ggml_sycl_kernel_profile_flush()` — the drain that creates the device
(`sycl.event`) spans in the first place. So the zero device events were not a
missing emitter; they were the guaranteed content of a flush that ran before the
drain. The fix removes the per-step call. `e2e_tg_profile_force_flush()` keeps its
per-step site: separate instrument, its own repeat-safe flush.

### Result of the re-run

B70 (`level_zero:0`), GPT-OSS 20B MXFP4, `-p 0 -n 128 -r 1 -v`, the "Commands
(exact)" env block above with the output prefix moved to `/tmp/sacs-decode/`.
Card had the full 32600.7 MB free; no GT reset on `0000:03:00.0` before or after.

| check | before | after |
|---|---:|---:|
| `graph_compute_impl` spans | 1 | **129** (= the decode step count) |
| `sycl.event` entries | 0 | **59,541** |
| `gap_class.*` lines in `timeline.gaps.parse` | 0 | **8** (the pp512 reference count) |
| trace size | ~40 MB | 100 MB |

Full category census after the fix: `ggml.op` 118,680 · `sycl.submit` 61,500 ·
`sycl.event` 59,541 · `sycl.wait` 645 · `ggml.graph` 129 — 240,495 events.

**Independent cross-check.** The trace's `timeline.gpu_event_total_ms_x1000
662688` (662.688 ms) reproduces the kernel profiler's separately measured
662.587 ms device-busy total to within 0.02 %. Two instruments, two flush paths,
same number — the device spans now in the timeline are the real ones.

### The gap-class split, first time on decode

```
gap_class.device0.compute.host_overlap.total_ms_x1000        1371574
gap_class.device0.compute.queue_serialization.total_ms_x1000      67
gap_class.device0.compute.runtime_idle.total_ms_x1000        1378763
gap_class.device0.compute.rounding_delta_ms_x1000                432
gap_class.device0.copy.host_overlap.total_ms_x1000           2497988
gap_class.device0.copy.queue_serialization.total_ms_x1000          0
gap_class.device0.copy.runtime_idle.total_ms_x1000            890327
gap_class.device0.copy.rounding_delta_ms_x1000                   -34
```

The rounding-delta rule the previous section left "ready and waiting" now has its
first decode measurement, and it **passes**: 432 µs against a 2750.4 ms compute
gap total is 0.016 %, and −34 µs against 3388.3 ms of copy gap is 0.001 %. Both
are far under the 5 % `LOW CONFIDENCE` threshold, so this split is usable. As
predicted, the delta grew with gap count (0.0034 % → 0.016 %) and still has ample
margin.

**Do not read attribution conclusions off these eight numbers yet.** They are
whole-capture totals over a 9202.215 ms host envelope that includes model load and
weight materialization, not a steady-state decode window — `wall_ms` was set from
the trace's own host-clock envelope because `--wall-ms` must still be passed
explicitly (defect 3, unfixed). Against the graph-span envelope alone (3456.345 ms)
the same 662.688 ms of device time is 19.2 % coverage rather than the 7.2 % the
parse prints. Isolating a steady-state slice is the next unit of work, not this
one.

### What was changed in the test file, and why

`tests/test-sycl-timeline-flush-source.py` contained a test asserting the buggy
call site **must exist** — added by `da9a7ceb5`, the commit that introduced the
bug. It was replaced by `test_decode_teardown_does_not_flush_the_timeline`, which
asserts the inverse while still requiring the `e2e_tg_profile_force_flush()` site.
The backend-free test was left exactly as it was, and still passes.

The actual regression guard is new:
`test_one_shot_timeline_flush_is_reachable_only_from_backend_teardown`. Both
original tests passed throughout the bug's lifetime because each call site is
individually correct — **the composition is what was broken, and nothing covered
it.** The new test brace-matches the two function bodies and asserts no
`sycl_timeline_flush(` inside `ggml_backend_sycl_graph_compute()` and every call
inside `ggml_backend_sycl_free()`. It also asserts the one-shot guard still exists
in `sycl-timeline.cpp`, so making the flush re-entrant fails loudly rather than
quietly invalidating the placement rule. Comments are blanked before scanning;
without that, the comment now documenting the removal reads as a call and the
guard goes vacuous in precisely the direction that hides a regression.

Verified RED before GREEN by checking out the pre-fix `ggml-sycl.cpp`: 2 failed,
1 passed. On the fix: 3 passed.

The file was also **unregistered in ctest**, which is why it never ran. It is now
registered via `llama_test_pytest()`; `ctest -N` goes 86 → 87.

### Gates

- `ctest -R 'decode-timeline|timeline-gap|timeline-flush'`: 3/3 Passed, none Skipped.
- Mistral completion gate: `1, 2, 3, 4, 5, 6, 7, 8, 9, 10`. Run on the **B70**, not
  the B50 — a `codescout.real` process (pid 3781131, the same one the 21:36 B50 GT
  reset was attributed to) held all three render nodes and left the B50 with
  320.0 MB free, so the run aborted in `[SYCL-PLAN] failed to size VRAM arena
  zones`. Environmental, reproduced twice, unrelated to this change.
- GPT-OSS `-cnv` gate: `1, 2, 3, 4, 5`.
- Throughput, B70 `-p 0 -n 128 -r 3`, graph replay ON, no profiler:
  **48.8096 ± 0.3743 tok/s** against the 48.3799 reference, +0.89 % — inside the
  B70's 3.3 % tg noise band. Machine was loaded (load average 8–14, Frigate
  ffmpeg), so treat this as a no-regression check, not a new baseline.

### Still open after this fix

Defect 3 of `llama.cpp-sacs` is **not fixed**: `parse-sycl-timeline.py:532` still
computes its fallback envelope across `sycl.event` timestamps, which are on the
device clock. Keep passing `--wall-ms` explicitly.

## Task 9 — Steady-state attribution (the question Plan A opened with)

**This is the attribution Plan A set out to produce.** It is possible now because
`llama.cpp-sacs` fixed the one-shot flush; the remaining obstacle — an envelope
polluted by model load — is removed here by windowing the capture to a
steady-state slice rather than by fixing parser defect 3.

### Method: window the capture, don't post-hoc filter it

`GGML_SYCL_TIMELINE_TOKEN_START=15 GGML_SYCL_TIMELINE_TOKEN_COUNT=100` on a
`-p 0 -n 128 -r 1 -v` B70 run. That skips warmup and the ~440 ms first-token
weight-materialization step, and captures 100 steady-state decode steps. This
was untestable before `sacs` — the trace never held more than one step, so a
window was meaningless.

The window bound exactly: **100** `graph_compute_impl` spans, per-step duration
min 18.43 / p50 19.84 / max 24.84 ms. No first-token outlier. `--wall-ms` is the
window's **own** envelope (2332.624 ms), which is the only figure describing the
population actually in the trace.

⚠️ First attempt captured 100 clean steps but **zero** `sycl.event` entries:
`GGML_SYCL_TIMELINE=timeline+events` alone is not sufficient — device events come
from the kernel profiler's drain, so `GGML_SYCL_KERNEL_PROFILE=1` and its block
are required. Re-captured with it.

### Configuration

| item | value |
|---|---|
| Card | Arc Pro B70, `level_zero:0`, PCI `0000:03:00.0` |
| Graph replay | **ON** — `GGML_SYCL_DISABLE_GRAPH: 0`, `graphs reused = 128`. This is the shipping path. |
| Binary | `850ca064b` (12153) |
| Window | steps 15–114, 100 steps, 2332.624 ms |
| Launches | 461 `sycl.event`/step — matches the plan's "~461 launches/token" grounding |

### Result

```
timeline.wall_ms_x1000                2332624
timeline.gpu_event_total_ms_x1000      512845     (22.0 % device coverage)
timeline.unattributed_ms_x1000        1819779
```

Compute queue, 41094 gaps totalling 1828.277 ms:

| class | ms/step | % of compute gap | % of unattributed |
|---|---:|---:|---:|
| **`runtime_idle`** | **10.242** | **56.0 %** | **56.3 %** |
| `host_overlap` | 8.040 | 44.0 % | 44.2 % |
| `queue_serialization` | 0.0006 | 0.003 % | 0.003 % |

`rounding_delta` 0.250 ms = **0.014 %** of the queue total — far under the 5 %
`LOW CONFIDENCE` threshold, so the classification is trustworthy. The metric
added by `llama.cpp-7v8i` is doing exactly the job it was built for.

Device busy 5.128 ms/step reproduces the independently measured 5.136 ms
(kernel profiler, separate code path) to **0.16 %**.

```
DOMINANT CLASS: runtime_idle at 56.3% of timeline.unattributed_ms_x1000
(10.242 ms/step of the 18.283 ms/step compute-queue gap, on a 23.326 ms/step
profiled token).
CONFIDENCE: HIGH — rounding_delta 0.014% of queue total, well under the 5% rule.
```

### What this means, and what it does not

Per Plan A's Task 7 decision table, `runtime_idle` dominant ⇒ **reduce launch
count (batching, graphlets); per-op caching buys nothing.**

Plan A's Task 6 gotcha anticipated this outcome and warned it would be misread:
a dominant `runtime_idle` "is the *least* actionable outcome for ggml-sycl code
and the most likely to be misread as 'our code is slow.' It means the opposite."
At 461 launches/step and 10.242 ms of `runtime_idle`, that is **~22 µs of gap per
launch** — time explained by neither our host work nor a dependency.

**`queue_serialization` at 0.003 % is a clean negative.** The dependency graph is
not the bottleneck. One of the three candidate causes is eliminated outright.

Largest single compute-queue gap transition:

| transition | ms (100 steps) | % of compute gap |
|---|---:|---:|
| `mxfp4.gateup.xmx_tiled_dpas_m2` → `sycl.binbcast.mul` | 504.150 | 27.6 % |
| `sycl.binbcast.event` → `sycl.get_rows.marker` | 232.370 | 12.7 % |
| `sycl.set_rows.generic` → `sycl.binbcast.mul` | 208.363 | 11.4 % |
| `sycl.softmax.forward` → `mxfp4.quantize.activation_q8_soa` | 206.997 | 11.3 % |

Top host overlap is the same leading transition, under `MUL_MAT_ID` (636.429 ms).
So the largest gap sits immediately after the MoE gate/up XMX matmul.

The copy queue's 2323.797 ms total is **not** decode work — in steady-state decode
that queue is essentially idle, and its gap total tracks the wall clock. Do not
read it as cost.

### Caveats

- **Profiled run.** 23.326 ms/step here vs ~20.67 ms/step clean — roughly 12.9 %
  observer effect in this configuration (`TIMELINE` + `KERNEL_PROFILE`), not the
  2.84 % Task 4 measured for `TIMELINE` alone. **The class *shares* are the robust
  output; the absolute per-step milliseconds are inflated.** Which class the
  overhead inflates more is not established — do not assume it cancels.
- Parser defect 3 (`--wall-ms` epoch mixing) remains **unfixed**. Passing
  `--wall-ms` explicitly is still mandatory; this capture derives it from the
  window's own envelope for that reason.
- Single capture, not a repeated-measures design. The class shares are large and
  the rounding delta is negligible, but a second capture would strengthen it.

## Task 10 — Class-first re-analysis: the verdict holds, the phase-2 target does not

Task 9 is a single capture, and its own last caveat asks for a second one. This
task strengthens it a different way: by **re-analysing the same trace** rather
than re-running it. That is deliberate. A confirming capture taken at load 18.68
would confound the very class under test — `runtime_idle` is exactly the
signature a descheduled submitting thread produces — whereas the questions below
are about whether the classifier can be *believed at all*, which the artifact on
disk already answers.

Reproduced with `scripts/parse-sycl-gap-causes.py`, which loads
`parse-sycl-timeline.py` by path and reuses its classifier verbatim, so the two
cannot drift. It reproduces Task 9's split exactly (`host_overlap` 803989,
`runtime_idle` 1023981 in `_ms_x1000`), which is what licenses everything below.

```
python3 scripts/parse-sycl-gap-causes.py /tmp/steady-slice2/sycl-timeline.json \
  --queue compute --steps 100 --top-transitions 6
```

### The verdict survives both of its structural failure modes

`runtime_idle` is the classifier's `else` branch (`parse-sycl-timeline.py:392`).
It absorbs genuine runtime latency *and* every gap the first two tests merely
failed to prove something about. A dominant `runtime_idle` is therefore not a
finding until the residual is separated from the signal. Splitting it:

| cause | ms/step | % of `runtime_idle` | n |
|---|---:|---:|---:|
| **`truly_idle`** | **9.602** | **93.8 %** | 18840 |
| `sum_covers_max_does_not` | 0.619 | 6.05 % | 3755 |
| `submit_pipelined_ahead` | 0.019 | 0.19 % | 264 |
| `no_submit_span` | **0.000** | **0.00 %** | **0** |

**The dangerous hypothesis is dead.** With graph replay ON, submits are recorded
at graph *record* time, so a replay window could plausibly contain no submit
spans at all — which would make `host_overlap` untestable and the verdict a
tautology. It does not: **all 46,100 device events resolve a submit span
(100.0 %)**, via `sycl.submit` records and the `host_submit_*_us` fallback at
`parse-sycl-timeline.py:312`. `no_submit_span` is zero, not small.

**The real defect is small.** `max_host_node_overlap_us` requires one *single*
host op to cover half the gap; a host busy with many short ops fails that bar
even when its spans together cover the window. That misfiles 0.619 ms/step.
Correcting it moves the split from 56.0 / 44.0 to roughly **52.6 / 47.4** —
`runtime_idle` still dominant, still above the decision table's 50 % threshold.
**The phase-2 branch is unchanged.** A `union_host_node_overlap_us` helper and a
`HOST_OVERLAP_COVERAGE` policy constant are staged in the parser for this; the
dispatch is deliberately not yet wired, because flipping it changes a published
number and that is a call to make explicitly rather than in passing.

⚠️ Note what `rounding_delta` does and does not certify. Task 9 cites
`rounding_delta 0.014 %` as grounds for `CONFIDENCE: HIGH`. That metric is an
*arithmetic* check that the three class totals sum to the queue total — and
`parse-sycl-timeline.py:547` force-balances them, as
`test-sycl-timeline-gap-class-conservation.py`'s own docstring says. It is
silent on whether each gap was classified *correctly*. The table above is the
first evidence for that, and `test-sycl-gap-causes.py` now gates it.

### The transition table in Task 9 pools all three classes — do not scope from it

This is the correction that matters. Task 9's largest-transition table, and the
handoff drawn from it, name `mxfp4.gateup.xmx_tiled_dpas_m2 → sycl.binbcast.mul`
at 27.6 % of compute gap as the leading phase-2 candidate. Split by class:

| class | ms (100 steps) | share of that transition |
|---|---:|---:|
| `host_overlap` | **502.660** | **99.7 %** |
| `runtime_idle` | 1.490 | 0.3 % |

**That transition is host-busy time, not idle time.** The host is doing ggml work
across it and overlap is working as intended. It is 0.16 % of the actionable
class. Reducing launch count around it buys nothing — the pooled ranking put a
well-overlapped transition at the top of a list used to choose a launch-count
fix, because `host_overlap` and `runtime_idle` demand opposite responses and were
being ranked together. **Rank within a class, never across classes.**

### Where the actionable 9.602 ms/step actually is

Two clusters, needing different fixes. Per-gap spread across all `truly_idle`
gaps: p50 29.6 µs, p90 81.2 µs, p99 258.8 µs, max 4823.5 µs.

**Cluster A — the inter-token bubble. 2.324 ms/step, 24.2 %, one gap per step.**

| transition | ms/step | n/step | µs/gap |
|---|---:|---:|---:|
| `sycl.binbcast.event → sycl.get_rows.marker` | 2.324 | 0.99 | 2347.2 |

Verified as the step boundary, not inferred: the trace holds 100 `ggml.graph`
spans and this gap occurs **n=99**, exactly the number of inter-graph
transitions. Host inter-graph gap is p50 1646 µs / mean 1718 µs, so the host is
provably between `graph_compute` calls for ~1.65 ms of the ~2.24 ms the device
spends idle. On a 20.670 ms clean token that is **~11 % of every token**.
Launch count inside the graph is irrelevant to it.

**Cluster B — per-layer attention-path stalls. ~6.37 ms/step, 66 %.** Every row
recurs 23–24 times per step and GPT-OSS 20B has 24 layers, so these are
per-layer:

| transition | ms/step | n/step | µs/gap |
|---|---:|---:|---:|
| `sycl.binbcast.event → sycl.rope` | 1.766 | 23.0 | 76.8 |
| `sycl.set_rows.generic → sycl.binbcast.mul` | 1.708 | 20.2 | 84.6 |
| `sycl.binbcast.event → sycl.softmax.forward` | 1.399 | 23.9 | 58.6 |
| `sycl.rope → sycl.set_rows.generic` | 0.863 | 24.0 | 36.0 |
| `sycl.rope → sycl.rope` | 0.632 | 24.0 | 26.3 |

This is the KV/attention path — rope, set_rows, softmax, binbcast — **not** the
MoE matmul path. `sycl.binbcast.event` is the source side of 57 % of the
actionable idle, which is the single strongest structural hint in the capture.

### What this does and does not establish

- **Does:** the `runtime_idle` verdict is not an instrument artifact; it survives
  its worst known defect with the same branch selected; and the phase-2 scope is
  two named clusters rather than the MoE boundary.
- **Does not:** run-to-run reproducibility. This is re-analysis of the Task 9
  capture, so a second capture is still owed. Nothing above depends on the
  absolute milliseconds — only on shares, counts and the n=99 coincidence — which
  are the outputs Task 9's own caveat identifies as robust to observer effect.
- **Does not:** rule out host load as a contributor to `truly_idle`. A capture on
  a quiet machine is the test, and it has not been run.

### Gates

`scripts/parse-sycl-gap-causes.py` (new) and `tests/test-sycl-gap-causes.py`
(new, registered in ctest as `test-sycl-gap-causes`, driven by a checked-in
synthetic trace). The test gates conservation — the four causes must sum to the
`runtime_idle` class in both time and count, which nothing force-balances — and
non-vacuity, requiring every cause to be observed at least once so a defect
cannot report as a zero. The fixture builds exactly one gap per cause. Both it
and `test-sycl-timeline-gap-class-conservation` pass.

**What these gates cannot catch:** they run on a synthetic trace, so they prove
the split is self-consistent and live, never that the *causes are the right
partition* of why a real gap is idle, and never that a real capture is free of
host-load contamination. A machine at load 18.68 would still produce
`truly_idle` that these gates would happily conserve.

### Amendment: what `sycl.binbcast.event` is, and why `queue_serialization ≈ 0` is not the clean negative Task 9 called it

`sycl.binbcast.event` is the source side of 57 % of the actionable idle, so what
it *is* decides how Cluster B is read. It is not a compute kernel.
`ggml_sycl_submit_binbcast_event` (`ggml/src/ggml-sycl/binbcast.cpp:93`) submits
either an `ext_oneapi_submit_barrier()` or, in `SAFE` mode, an **empty
`single_task`**. Mode defaults to `BARRIER`, but lines 101–103 downgrade
`BARRIER` to `SAFE` whenever the queue is in-order — and ggml-sycl GPU queues
always are: `default_queue_properties()` (`common.hpp:5954`) returns
`{in_order, enable_profiling}`, and `dpct`'s `default_queue()`
(`dpct/helper.hpp:776`) returns the in-order queue, with a comment keeping it
that way deliberately for an Arc L0 driver issue.

Confirmed against the trace rather than inferred from the code path: **all
14,400 `sycl.binbcast.event` records carry `mode=kernel;event_mode=safe`**, and
all are owned by `node_op=MUL` (`attn_norm`, `attn_post_norm`,
`ffn_moe_weighted` — three per layer × 24 layers).

| quantity | per step |
|---|---:|
| launches | **72.0** |
| device time | 0.0382 ms (p50 0.521 µs — it is a no-op) |
| host submit time | 0.2519 ms (p50 3.000 µs) |
| **`truly_idle` in the gaps that follow them** | **≈5.503 ms (57.3 % of the actionable 9.602)** |

So 72 empty kernels per token, costing almost nothing themselves, are followed by
5.5 ms/step of device idle — **on a queue whose in-order property already
guarantees the ordering they exist to provide.** The same reasoning is already
applied elsewhere in this backend: `unified-cache.cpp:7677` skips
`ext_oneapi_submit_barrier` with the comment "In-order queues already serialize
submissions".

⚠️ **This is correlation, not causation.** The gap *follows* the no-op; that does
not prove the no-op causes it. Establishing that is phase 2's first job, and
`GGML_SYCL_BINBCAST_EVENT_MODE` (`binbcast.cpp:77`) is not sufficient to test it
— both of its values still submit something.

**The correction this forces.** Task 9 reads `queue_serialization` at 0.003 % as
"the dependency graph is not the bottleneck — one of three candidate causes is
eliminated outright", and the handoff repeats it. That over-reads the metric.
`device_gap_has_dependency` (`parse-sycl-timeline.py:342`) detects a dependency
only through explicit `depends_on` event ids. **An in-order queue serializes
implicitly, declaring no such edge**, so in-order serialization is invisible to
that test and falls through to `runtime_idle` — the `else` branch — by
construction. On this backend, where every GPU queue is in-order,
`queue_serialization` is close to unreachable whatever the truth is.

`queue_serialization ≈ 0` is therefore a clean negative for *explicitly declared*
dependencies only. It is **not** evidence that serialization is absent, and the
dependency hypothesis is **not** eliminated. Some unknown share of `truly_idle`
may be exactly that, wearing the residual class's name.

### Coverage policy resolved: `HOST_OVERLAP_COVERAGE = "union"`, and both splits recorded

The decision left open above was taken on 2026-07-25: `device_gap_has_host_overlap`
now measures **merged** host-node coverage (`union_host_node_overlap_us`) rather
than the single longest covering span. Under the old `"max"` rule a 2 ms gap
covered by twenty 90 µs ops back to back — 1.8 ms of real host work, and the
shape of a decode step submitting ~461 launches — scored 90 µs and was filed as
idle.

Both splits, same capture, same `--wall-ms 2332.624`. **Neither is a re-run**;
this is the identical trace parsed under each rule:

| metric (`_ms_x1000`) | `"max"` (Task 9, published) | `"union"` (in force) |
|---|---:|---:|
| `host_overlap` | 803989 · 8.040 ms/step | **865888 · 8.659 ms/step** |
| `runtime_idle` | 1023981 · 10.240 ms/step | **962332 · 9.623 ms/step** |
| `queue_serialization` | 57 | 57 |
| `rounding_delta` | 250 (0.014 %) | 250 (0.014 %) |
| `runtime_idle` % of `unattributed` | 56.3 % | **52.9 %** |

```
DOMINANT CLASS: runtime_idle at 52.9% of timeline.unattributed_ms_x1000
CONFIDENCE: HIGH — rounding_delta 0.014% of queue total, well under the 5% rule.
```

**The verdict and the phase-2 branch are unchanged** — `runtime_idle` remains
dominant and remains above the decision table's 50 % bar. Task 9's 56.3 % is not
retracted; it was correct under the rule in force when it was written, and is
kept above for exactly that reason.

**The result that matters is what did *not* move.** `truly_idle` is
**identical** under both rules — 960153, 9.602 ms/step, n=18840, and the same
transition ranking to the digit. The reclassified gaps were precisely the
`sum_covers_max_does_not` ones, which were never part of `truly_idle` to begin
with. So **the actionable budget every phase-2 task is scoped from is invariant
to this decision**, which is why the switch was safe to make after the plan was
written rather than before.

What improves is the residual's purity:

| | `"max"` | `"union"` |
|---|---:|---:|
| `truly_idle` share of `runtime_idle` | 93.8 % | **99.8 %** |
| `submit_pipelined_ahead` | 0.19 % | 0.20 % |
| `sum_covers_max_does_not` | 6.05 % | **retired** |
| `no_submit_span` | 0.00 % | 0.00 % |

`sum_covers_max_does_not` is retired rather than reported as zero: it was a
diagnostic *of the `"max"` rule's defect*, and `gap_cause_names()` drops it from
the schema under `"union"` so a retired cause cannot be mistaken for a real
instrument defect that never fired. Note `union ≥ max` always, so widening can
only move gaps *into* `host_overlap`, never out.

`tests/test-sycl-gap-causes.py` now pins the policy: its fixture's `k2→k3` gap
(three 400 µs nodes, merged 1200 µs, longest 400 µs, against a 1000 µs bar)
classifies differently under each rule, so flipping `HOST_OVERLAP_COVERAGE`
without deliberately re-stating the published split fails the gate loudly.
`test-sycl-timeline-gap-class-conservation` is unaffected and still passes.
