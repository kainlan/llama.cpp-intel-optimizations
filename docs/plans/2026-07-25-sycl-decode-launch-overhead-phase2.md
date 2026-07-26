# SYCL Decode Launch-Overhead Reduction — Phase 2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use team-driven-development to implement this plan with agent teams.

**Goal:** Reduce the **9.602 ms/step** of `truly_idle` decode time on GPT-OSS 20B / Arc Pro B70 — device time explained by neither ggml-sycl host work nor a declared dependency — by attacking the two clusters it actually sits in, established by measurement rather than by intuition.

> **Status 2026-07-25:** Tasks 1, 2 and 6 are done. Task 2 **relocated Cluster A above this backend** (its 2.324 ms/step window is 0.5 % covered by ggml-sycl instrumentation), so Task 3 correctly wrote no fix plan. **The remaining in-scope target is Cluster B, ~6.37 ms/step**, via Tasks 4 → 5, with Task 7 for its residual.

**Selected by:** Plan A's decision table, applied to Task 9's `DOMINANT CLASS: runtime_idle at 56.3%` → *"reduce launch count (batching, graphlets); per-op caching buys nothing."* Task 10 then re-analysed the same capture class-first and narrowed *which* launches.

**Tech Stack:** C++17 / SYCL (Intel oneAPI DPC++ 2026.1), Python 3, CMake/CTest, Arc Pro B70 (`level_zero:0`).

**Baseline to beat:** 20.670 ms/token clean (5.136 ms device-busy, 15.533 ms non-kernel). B70 GPT-OSS 20B MXFP4 gates: ~1415 PP512 / ~44 TG128 per `docs/backend/sycl-perf-baselines.md`.

---

## What is measured, and what is not (read first)

Everything below comes from **one** capture (`/tmp/steady-slice2/`, B70, graph replay ON, 100 steady-state steps), analysed twice — by `parse-sycl-timeline.py` in Task 9 and class-first by `parse-sycl-gap-causes.py` in Task 10. Full detail in `docs/plans/2026-07-25-decode-host-overhead-findings.md`.

**Established:**

| fact | value |
|---|---|
| Actionable idle (`truly_idle`) | **9.602 ms/step**, 99.8 % of `runtime_idle`, **invariant** to the coverage policy |
| Instrument is sound | 100 % of 46,100 device events resolve a submit span; `no_submit_span` = 0 |
| Coverage policy | Resolved to `"union"`; split 52.9 / 47.6, **same branch selected**. Both splits recorded in the findings doc |
| Cluster A — inter-token bubble | 2.324 ms/step, **n=99** vs 99 inter-graph transitions. ⛔ **Relocated above the backend by Task 2** — the host window is 0.5 % covered by ggml-sycl instrumentation and `ggml.op` is exactly 0.00 %. Not a launch-count target |
| Cluster B — per-layer attention stalls | ~6.37 ms/step, 23–24×/step over 24 layers |
| `sycl.binbcast.event` is an **empty `single_task`** | 72/step, 0.0382 ms/step device, followed by 5.503 ms/step idle |

**NOT established — do not build on these without the named task closing them first:**

1. ~~**Run-to-run reproducibility.**~~ ✅ **CLOSED by Task 1** (findings doc, Task 11). Second capture on the same binary: gap count identical (41094), device busy agrees to 0.005 %, `runtime_idle` share 52.9 % → 52.5 %, every cluster transition within ±2.4 %, n=99 exact.
2. ~~**Host load is not a contributor.**~~ ✅ **CLOSED by Task 1.** Load fell 2.9× (18.68 → 6.50) with codescout off the GPUs entirely, and `truly_idle` went *up* 0.9 %. Descheduling is not driving it.
3. **Causation for the `binbcast.event` no-op.** The idle *follows* the marker; nothing yet shows it is *caused* by it. **Task 4.**
4. **What consumes the ~1.65 ms host inter-graph window.** Sampling is normally microseconds. Unattributed. **Task 2.**

**Still true regardless:** both captures are profiled runs (~12.9 % observer effect). The class **shares** are the robust output; absolute per-step milliseconds are inflated.

⚠️ **`queue_serialization ≈ 0` does NOT eliminate the dependency hypothesis.** `device_gap_has_dependency` sees only explicit `depends_on` edges; ggml-sycl GPU queues are all in-order (`common.hpp:5954`) and serialize *implicitly*, declaring no edge. That class is near-unreachable on this backend by construction, so an unknown share of `truly_idle` may be serialization wearing the residual class's name. Task 9's "one of three candidate causes is eliminated outright" is withdrawn.

---

## Scope Boundary

**Tasks 1, 2, 4 and 7 are measurement or mechanism tasks and are fully specified.** The two *fixes* they feed — Tasks 3 and 5 — are deliberately **not** decomposed here.

This is the same discipline Plan A's scope boundary applied, and for the same reason: the fix for Cluster A depends entirely on what Task 2 finds is consuming the host window (a serial data dependency in autoregressive decode cannot be overlapped away, but 1.65 ms of it is far too large to *be* that dependency), and the fix for Cluster B depends on whether Task 4 shows the marker is causal. Writing GREEN code for every branch means writing code that gets thrown away.

**Task 3 and Task 5 each begin by writing their own plan from their predecessor's measured result**, using the decision rules stated in those tasks.

---

## Team Topology

**Recommended implementers:** 2 concurrent (two independent tracks after Task 1)
**Reviewers:** spec + quality, spawned FRESH per review

### Parallel Tracks

| Track | Tasks | Description |
|-------|-------|-------------|
| — | 1 | Confirming capture on a quiet machine (gates *absolute* claims only; tracks A and B may start their measurement tasks in parallel) |
| A | 2, 3 | Inter-token bubble: attribute the host window, then fix |
| B | 4, 5 | `binbcast.event` no-op: establish causation, then remove |
| C | 6, 7 | Parser semantics + residual per-layer attribution (independent, no GPU) |

### Dependency Graph

```dot
digraph dependencies {
    rankdir=LR;
    1 [label="T1: confirming capture (quiet host)"];
    2 [label="T2: attribute host inter-graph window"];
    3 [label="T3: Cluster A fix (plan from T2)"];
    4 [label="T4: binbcast.event causation"];
    5 [label="T5: Cluster B fix (plan from T4)"];
    6 [label="T6: wire HOST_OVERLAP_COVERAGE"];
    7 [label="T7: attribute residual per-layer stalls"];
    2 -> 3;
    4 -> 5;
    1 -> 3 [style=dashed,label="absolute claims"];
    1 -> 5 [style=dashed,label="absolute claims"];
    6 -> 7 [style=dashed];
}
```

### File Ownership Map

| File/Directory | Tasks | Conflict Risk |
|----------------|-------|---------------|
| `ggml/src/ggml-sycl/binbcast.cpp` | 4, 5 | Sequential (same track) |
| `scripts/parse-sycl-timeline.py` | 6 | None |
| `scripts/parse-sycl-gap-causes.py` | 7 | None |
| `tests/test-sycl-gap-causes.py` | 6, 7 | Sequential |
| `docs/plans/2026-07-25-decode-host-overhead-findings.md` | 1, 2, 4, 7 | **Append-only, one task at a time** |
| `docs/plans/` (new Cluster A / B fix plans) | 3, 5 | None (different files) |

---

## Safety Constraints (apply to EVERY task)

Carried forward from Plan A unchanged — these have hung or OOM-killed this host.

- **Never run `test-backend-ops`** in a subagent or background task (TTM shmem grows to 50–224 GB → OOM kill).
- **Never run `sycl-ls`** — has hung this host in `xe_drm_ioctl` requiring a reboot.
- **`timeout` every GPU command.** B70 runs also set `GGML_SYCL_OP_TIMEOUT_MS=180000`.
- **`TMPDIR=/tmp` on every build** (root fs ~98 %; AOT link fails ENOSPC otherwise).
- **Verify the backend, not the tokens**, after any reconfigure: `grep -E '^GGML_SYCL:' build/CMakeCache.txt` (want `BOOL=ON`) and `ldd build/bin/llama-completion | grep -cE 'libggml-sycl|libsycl'` (want ≥2). A CPU fallback passes every digit gate at ~8 tok/s.
- **Check competing load before any throughput measurement** — `uptime`, `pgrep -af 'codescout|ninja|icpx|ffmpeg'`. Absolute numbers under load are not baselines. `llama.cpp-2rkc` (codescout holding GPU VRAM) is a standing confound.
- **Check free VRAM on the B70** before trusting a number (~32.6 GB expected).
- **`dmesg` is privilege-denied**: use `journalctl -k --since "1 hour ago" --no-pager | grep -iE 'GT reset|guc_id|GPU hang|xe.*reset'`.
- **Never `git revert`.** Fix forward.
- **Every perf claim needs the Mistral completion gate re-run** — `llama-bench` measures tok/s, never correctness.

---

## Tasks

### Task 1: Confirming capture — ✅ **DONE 2026-07-25** (findings doc, Task 11)

Reproduced on the same binary `850ca064b`, paired on load rather than waiting for quiet. Both open questions it gated are discharged: the finding reproduces, and the host-load confound is refuted. The method below is kept because Tasks 2, 4 and 7 all re-capture with it.

**Why:** Task 9 is a single capture and Task 10 is a re-analysis of it, so nothing yet establishes reproducibility, and nothing separates `truly_idle` from host descheduling. Both are prerequisites for any *absolute* claim, and for attributing a later speedup to a change rather than to a quieter machine.

**Do:** Re-run the Task 9 capture **verbatim** — same env block, same window, same card.

⚠️ **Load below ~4 is not reachable on this host** and waiting for it is a trap. Its floor is ~6–9, all of it Frigate `ffmpeg` (30 processes) on the **iGPU** (`renderD128` → `0000:00:02.0`) — a security system, not something to stop for a benchmark. Neither the B70 nor the B50 is a Frigate consumer.

Prefer a **paired design over a quiet one**: capture at a load *deliberately different* from Task 9's 18.68 and compare shares. Reproduction across a load ratio is direct evidence against the host-load confound; a single quiet run only shows the shares once and cannot separate the two hypotheses at all. Do stop `codescout` first (`codescout daemon --stop` — drains; never SIGKILL): it is a real GPU consumer on the benchmark cards (`llama.cpp-2rkc`), unlike Frigate.

⚠️ **The output-path variables are load-bearing.** Without `GGML_SYCL_TIMELINE_OUTPUT` and `GGML_SYCL_KERNEL_PROFILE_OUTPUT` the run completes, reports tok/s, and writes **no trace at all** — indistinguishable from a capture that produced nothing. Env block below is the one `scripts/sycl-gptoss-decode-timeline-profile.sh:98-112` uses, with the window vars added; note that script's own bench args are `-p 512`, so it is not a drop-in for this capture.

```bash
OUT=/tmp/steady-sliceN; mkdir -p "$OUT"; uptime > "$OUT/load.before"
source /opt/intel/oneapi/setvars.sh --force
env ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_OP_TIMEOUT_MS=180000 \
  GGML_SYCL_TIMELINE=timeline+events \
  GGML_SYCL_TIMELINE_OUTPUT="$OUT/sycl-timeline.json" \
  GGML_SYCL_TIMELINE_TOKEN_START=15 GGML_SYCL_TIMELINE_TOKEN_COUNT=100 \
  GGML_SYCL_TIMELINE_MAX_EVENTS=4000000 \
  GGML_SYCL_KERNEL_PROFILE=1 GGML_SYCL_KERNEL_PROFILE_OUTPUT="$OUT/sycl-kernels" \
  GGML_SYCL_KERNEL_PROFILE_FORMAT=both GGML_SYCL_KERNEL_PROFILE_RAW=1 \
  GGML_SYCL_KERNEL_PROFILE_TOP_N=80 GGML_SYCL_KERNEL_PROFILE_FLUSH=window \
  GGML_SYCL_MOE_PHASE_MATERIALIZE=1 GGML_SYCL_MOE_PHASE_BULK_XMX=1 \
  GGML_SYCL_MOE_DOWN_SUM_DIRECT=1 \
  timeout 900 ./build/bin/llama-bench \
    -m /Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf \
    -ngl 99 -fa 1 -p 0 -n 128 -r 1 -v \
  > "$OUT/bench.stdout" 2> "$OUT/bench.stderr"
uptime > "$OUT/load.after"
```

Parse with `--wall-ms` set to the window's **own** envelope (parser defect 3 is still unfixed — never let it derive one). Then run `parse-sycl-gap-causes.py --steps 100`.

**Gate:** Record load average before and after. Compare **shares and counts**, not absolute ms: `truly_idle` share of `runtime_idle`, the n=99 inter-graph coincidence, and the 72/step `binbcast.event` count must all reproduce. Append the comparison to the findings doc as Task 11.

**If the shares move materially**, host load was a confound in Task 9 and Tasks 3 and 5 must not proceed on the old numbers — say so plainly and stop rather than averaging the two.

⚠️ `GGML_SYCL_TIMELINE=timeline+events` alone yields **zero** `sycl.event` entries; the kernel profiler's drain is what emits device events, so `GGML_SYCL_KERNEL_PROFILE=1` is mandatory. Traces are Chrome format — events are under `traceEvents`, not `events`.

---

### Task 2: Attribute the host inter-graph window — ✅ **DONE 2026-07-25** (findings doc, Task 12)

Coverage **0.5 %** on both captures; `ggml.op` exactly 0.00 %. The window is empty of ggml-sycl work, which relocates Cluster A above the backend and fires Task 3's no-fix-plan row. Method below kept for the record.

### Task 2 (completed): method

**Why:** The device is idle ~2.24 ms between graphs while the host is between `graph_compute` calls for ~1.65 ms of it. Autoregressive decode has a genuine serial dependency there — logits → sampling → next embedding — but that is normally microseconds, not 1.65 ms. Until we know what the 1.65 ms *is*, no fix can be scoped.

**Do:** Attribute the window between consecutive `ggml.graph` spans by callsite. The kernel profiler's `raw_events` already carries `file`/`line`/`function` per record and is not subject to the timeline's constraints (findings doc, "The `raw_events` path is already open"). Prefer that over new instrumentation; add host-side spans only if it proves insufficient, and say so explicitly if you do.

**Gate:** A ranked table accounting for ≥80 % of the measured host inter-graph window by callsite, appended to the findings doc. Below 80 %, report the coverage honestly rather than presenting a partial table as complete.

**Do NOT** propose a fix in this task. Task 3 does that from the table.

---

### Task 3: Cluster A fix — ⛔ **NO FIX PLAN WRITTEN** (rule applied, 2026-07-25)

Task 2 measured the window at **0.5 % coverage**, so the rule row *"No callsite > 20 % and coverage < 80 % → Write no fix plan; the window is unattributed, say so"* fires. See findings doc Task 12.

The outcome is stronger than "unattributed": ggml-sycl activity in the window is **positively absent** (`ggml.op` coverage exactly 0.00 %, reproduced on both captures). **Cluster A is not a launch-count problem and no ggml-sycl change can address it** — the window is `llama_synchronize` + `llama_decode` host work, above the backend.

⚠️ It also cannot be scoped from `llama-bench` at all: its tg loop feeds `std::rand()` with no sampling and calls `llama_synchronize` every iteration, so part of the bubble is a harness property. A real fix needs host-side instrumentation on `llama-cli`/`llama-server`. **Cluster B is unaffected and is now the only actionable ggml-sycl target.**

The original rule table is kept below for the record.

### Task 3 (superseded): Cluster A fix — write the plan, then implement

**Entry:** Task 2's table. **Blocked on Task 1 for any absolute speedup claim.**

**Decision rule, fixed in advance:**

| Task 2 shows | Fix direction |
|---|---|
| One callsite > 50 % of the window | Optimize that callsite; scope from it directly |
| Work that does **not** depend on this token's logits | Overlap it with graph execution (it is on the critical path only by accident of ordering) |
| Work that genuinely depends on this token's logits | **Do not try to overlap it.** Reduce its cost, or accept and document the floor |
| No callsite > 20 % and coverage < 80 % | Write no fix plan; the window is unattributed, say so |

**Gate:** Mistral completion gate + B70 GPT-OSS gate pass; `llama-bench` shows no regression against `docs/backend/sycl-perf-baselines.md`; the claimed win is measured **interleaved**, never blocked A-then-B.

---

### Task 4: Establish whether the `binbcast.event` no-op is causal (Cluster B)

**Why:** 72 empty `single_task` submissions per token, 0.0382 ms/step of device time, followed by 5.503 ms/step of `truly_idle` — on an in-order queue that already guarantees the ordering the marker exists to provide. If causal, this is the single largest actionable item in the capture. If not, Cluster B's budget must be re-attributed and the marker left alone.

**Do:** `GGML_SYCL_BINBCAST_EVENT_MODE` (`binbcast.cpp:77`) is **not** sufficient — both `safe` and `barrier` still submit something. Add a third mode that returns **no submission at all** when the queue is in-order, reusing the existing reasoning at `unified-cache.cpp:7677` ("In-order queues already serialize submissions"). Keep it **opt-in** and default-off for this task.

Then determine what the returned `sycl::event` is actually used for. It is a real return value with real callers — if any caller needs a concrete event to wait on or chain, a no-submission mode must give it something valid or the change is a correctness bug, not an optimization. **Establish this before measuring, and record the callers.**

**Gate:**
- Mistral completion gate **and** the GPT-OSS B50 chat gate pass with the new mode on. Correctness first — a marker removal that corrupts ordering will still look fast.
- Interleaved A/B on the B70, ≥5 paired runs, reporting the spread.
- Re-capture and confirm via `parse-sycl-gap-causes.py` that `truly_idle` actually *fell* — not merely that tok/s moved.

**Report honestly if it is not causal.** A no-op removal that saves its own 0.29 ms/step and nothing more is a real but small result, and saying so is the outcome. Do not reach for the 5.5 ms.

---

### Task 5: Cluster B fix — make it the default, or re-attribute

**Entry:** Task 4's verdict.

| Task 4 shows | Action |
|---|---|
| Causal, gates pass, win reproduces interleaved | Make no-submission the default for in-order queues; keep an opt-out env var; update `docs/backend/sycl-env-vars.md` |
| Causal but a caller needs the event | Scope the narrower change that preserves the contract; do not force it |
| Not causal | Leave `binbcast.cpp` alone. Re-attribute Cluster B from Task 7 and write no fix plan |

---

### Task 6: Wire `HOST_OVERLAP_COVERAGE` — ✅ **DONE 2026-07-25**

Resolved to `"union"`. `device_gap_has_host_overlap` now dispatches through `host_node_coverage_us`, which rejects an unknown policy loudly rather than defaulting. Both splits are recorded side by side in the findings doc ("Coverage policy resolved"); Task 9's 56.3 % is **not** retracted, since it was correct under the rule then in force.

`truly_idle` — the budget every task below is scoped from — is **identical** under both rules (960153, 9.602 ms/step, n=18840, same transition ranking to the digit). The reclassified gaps were never part of it. Nothing in this plan changes as a result.

`tests/test-sycl-gap-causes.py` pins the policy via a fixture gap that classifies differently under each rule, so a future flip cannot silently restate a published number. Tracker: `llama.cpp-nceh`.

---

### Task 7: Attribute the residual per-layer stalls (no GPU)

**Why:** If Task 4 shows the marker is causal, ~0.87 ms/step of Cluster B still sits in transitions it does not touch — `rope → set_rows.generic` (0.863), `rope → rope` (0.632), `set_rows.generic → binbcast.mul` (1.708, partly). These are 24×/step, so they are per-layer, and they are the KV-cache write path.

**Do:** Re-analyse the Task 1 capture for these transitions specifically. Given the `queue_serialization` correction above, explicitly test whether in-order queue serialization — invisible to `device_gap_has_dependency` — explains them, before concluding launch overhead does.

**Gate:** A written verdict naming which mechanism dominates, appended to the findings doc, with the evidence that distinguishes the two. "Unclear" is an acceptable verdict; a guess is not.

---

## End-to-End Validation (on the user's machine) — MANDATORY

Before any task in this plan is called done:

```bash
source /opt/intel/oneapi/setvars.sh --force

# 0. Backend is actually present (a CPU fallback passes every gate below)
grep -E '^GGML_SYCL:' build/CMakeCache.txt
ldd build/bin/llama-completion | grep -cE 'libggml-sycl|libsycl'

# 1. Mistral completion gate — output must start "1, 2, 3, 4, 5, 6, 7, 8, 9, 10"
ONEAPI_DEVICE_SELECTOR=level_zero:1 timeout 300 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0

# 2. ctest
ctest --test-dir build --output-on-failure -j $(nproc)

# 3. B70 throughput vs docs/backend/sycl-perf-baselines.md (~1415 PP512 / ~44 TG128)
ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_OP_TIMEOUT_MS=180000 \
  timeout 900 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf -p 512 -n 128
```

**B70 tg128 is the noisy axis** (cv 3.3 %, range 40.18–46.27 over 21 runs) — ignore single-run differences below ~10 %. The B50 is steady (cv 0.7 % tg) and is the better card for detecting a small real move.
