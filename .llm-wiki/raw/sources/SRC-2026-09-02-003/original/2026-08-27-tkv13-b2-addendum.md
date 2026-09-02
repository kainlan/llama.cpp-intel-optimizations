# TKV-13 Stage 13a: B2 Design Addendum — Overlapped Host-Task Attention

**Status:** DESIGN ONLY. No code changes accompany this document. Owner review
required before 13b (implementation) starts.

**Scope:** Replace the serial `ggml_backend_sched` split execution of
demoted-layer attention (landed as B1′, `llama.cpp-uize`/`llama.cpp-h56y`) with
an overlapped, event/DAG-chained host-task path so CPU attention for demoted
layers runs concurrently with GPU work, closing the gap to the owner-ratified
perf floor (decode TG ≥ 80% of the overlapped memory-bandwidth roofline,
`docs/plans/data/tiered-kv/spike-floor-calibration.md` §OWNER RATIFICATION).

**Grounding:** every claim below is cited to a `file:line` read live on this
tree (master, post-B1′) on 2026-08-27, or to a landed doc/ticket. `ggml-sycl.cpp`
line numbers were re-verified today by pipe-grep per CLAUDE.md (codescout is
blind in that file); they drift, re-anchor before editing.

---

## 1. Recap: what B1′ actually does, and why it cannot reach the floor

`ggml_backend_sycl_set_runtime_context` (`ggml-sycl.cpp:15841` onward) runs
`plan_runtime_kv_demotion` (`kv-runtime-demotion.cpp:5-44`) when a runtime KV
update is over budget, moves the returned layers' KV to a dedicated pinned-host
buft (`ggml_backend_sycl_kv_host_buffer_type()`, `ggml-sycl.cpp:38771`), and the
SYCL backend then structurally **declines** any op with an operand resident in
that buft (`ggml_sycl_tensor_is_in_kv_host_buft` + the decline at the top of
`ggml_backend_sycl_device_supports_op`, `ggml-sycl.cpp:99085-99136`). Declining
is what hands the op to `ggml_backend_sched`, which routes it to the CPU
backend.

That handoff is the bottleneck. `ggml_backend_sched_compute_splits`
(`ggml-backend.cpp:2311-2507`) executes splits **strictly in program order on
one host thread**: for each split it calls
`ggml_backend_graph_compute_async(split_backend, &split->graph)`
(`ggml-backend.cpp:2460`) and only synchronizes with the *previous* backend
when the next split changes backend and has no inputs (`:2328-2334`). The CPU
backend's `graph_compute` is not asynchronous in any meaningful sense — it runs
the whole CPU-assigned subgraph on the calling thread using its own internal
thread pool and returns only when done. So while a demoted layer's
`FLASH_ATTN_EXT` (and its neighboring `CPY`/`SET_ROWS` checkpoint ops) execute
on CPU, **the host thread driving the scheduler is blocked inside that call**
and cannot submit any *new* SYCL work. GPU kernels already enqueued from a
*prior* split keep draining on the device queue (that's the partial overlap
B1′ gets for free), but anything the schedule would submit *after* the CPU
split — the next layer's dense projections, later MoE dispatch — cannot be
submitted until the CPU split returns. That is exactly the ceiling the
spike-floor-calibration doc measured (serial ≈8.4 tok/s vs overlapped ≈14 at
32K fill, bar ≈11.2).

B2's job is to stop letting `ggml_backend_sched`'s split model own this
crossing.

---

## 2. Question 1 — where the overlap happens

**Recommendation: inside the SYCL backend, CpuExpertPool-style. Do not attempt
sched-level pipelining.**

### The existing precedent, read precisely

`CpuExpertPool` (`cpu-expert-pool.hpp/.cpp`) is NOT a `sycl::queue::host_task`
mechanism — it is a **persistent `std::thread` pool** fed through a
`std::mutex`/`std::condition_variable` work queue (`worker_thread()`,
`cpu-expert-pool.cpp:112-131`), and `submit_batch()` returns a
`std::future<void>` (`:133-152`). This matters because `sycl::depends_on()`
only orders **SYCL commands**; a `std::thread`'s completion is not a SYCL event
and cannot be `depends_on()`-ed directly. `docs/backend/sycl-memory-design.md:74`
describes the CpuExpertPool overlap as "via `sycl::depends_on`
(~9.7 µs cross-device latency)" — that phrasing is imprecise for what the code
actually does (see §2.1 note below); B2's design is written against the real
mechanism, not the doc's summary.

The real mechanism, read from the dispatch site (`ggml-sycl.cpp:70375-70536`)
and the deferred-flush machinery (`ggml-sycl.cpp:20125-20210`, `:84019-84031`):

1. **Producer side (GPU → CPU input):** a GPU-computed input (activations, for
   MoE) is copied device→host async; the `sycl::event` is captured but its
   `.wait()` is **deferred** as late as possible — "moved here from the D2H
   submission site above to overlap the transfer with task struct building"
   (`ggml-sycl.cpp:70517-70520`). The wait is unavoidable at this exact point
   because a `std::thread` cannot itself observe a SYCL event; it is a bounded,
   single, host-side join at the hand-off, not a per-op drain.
2. **Compute:** the host thread pool executes the CPU kernel; `submit_batch()`
   returns a `future` stored in a **thread-local pending-result slot**
   (`g_pending_scatter` / `g_pending_cpu_pipeline`).
3. **Consumer side (CPU → GPU output), the actual overlap mechanism:** as the
   SYCL backend's own graph-compute loop walks subsequent nodes, it calls
   `flush_pending_cpu_scatter_if_consumed(node, device)`
   (`ggml-sycl.cpp:20182-20210`) on every node. That function walks the node's
   `ggml_tensor::src[]` (through `view_src`) to test, structurally, whether
   *this* node actually reads the pending CPU-computed tensor
   (`ggml_sycl_op_consumes_tensor` / `ggml_sycl_tensor_depends_on`,
   `:20150-20175`). If not, it either does nothing or does a **non-blocking**
   `future.wait_for(std::chrono::seconds(0))` poll (`try_flush_pending_cpu_scatter`,
   `:20126-20141`) — genuine overlap, no host wait. Only when a real consumer
   is reached does it perform the blocking wait + host→device scatter memcpy
   (`flush_pending_cpu_scatter()`), and this is legitimate because at that
   point the GPU kernel about to be submitted for that node *genuinely cannot
   proceed* without the data — it is the edge, expressed as a DAG check plus a
   deferred wait, not a wait masking a missing edge.
4. **Teardown:** any still-pending future is waited unconditionally at
   graph-boundary/zone-reset time (`ggml-sycl.cpp:84019-84031`) — squarely the
   "teardown" exception the no-host-waits ruling carves out
   ([[no-host-waits-event-chain-everything]]).

### 2.1 A discrepancy to surface to the owner, not silently resolve

The design doc's "`sycl::depends_on`" framing and the actual "DAG-consumer-
deferred future + bounded input join" mechanism are different things that
happen to satisfy the same ruling for different reasons (one expresses
ordering as SYCL events; the other expresses it as a host-side dependency
check plus a deferred, unavoidable join at real data boundaries, because one
side of the crossing is not a SYCL command stream at all). B2 should be
designed against the mechanism that actually exists and is proven in
production (MoE CPU dispatch, live), not against the doc's simplified
description. This addendum flags the mismatch rather than quietly building on
the wrong model; a follow-up doc correction to `sycl-memory-design.md:74` is
recommended but out of scope for 13a/13b.

### Why not sched-level pipelining

Making `ggml_backend_sched_compute_splits` itself pipeline — start submitting
the *next* split's GPU work before a CPU split returns — would mean changing
shared/upstream scheduler code that every other backend and every other model
architecture depends on, to solve a fork-local placement problem. It also
does not match how this backend already solves the identical shape of problem
(GPU-resident data, host-resident compute, need to overlap): CpuExpertPool
proves the in-backend approach works, is measured, and is documented as
supported architecture (`sycl-memory-design.md` "Placement decides the
executor"). Recommendation: **extend the in-backend pattern to demoted-layer
attention** rather than opening a second, riskier front in `ggml-backend.cpp`.

### The concrete mechanism for attention

The prerequisite for the in-backend approach is that `supports_op` must
**stop declining** these ops (so `ggml_backend_sched` never routes them to the
CPU backend and never creates a split boundary for them at all) and the SYCL
backend must instead **accept and dispatch them itself**:

- `ggml_sycl_tensor_is_in_kv_host_buft` keys off a specific set of ops today
  implicitly (whatever reaches `supports_op` with a KV-host-buft operand: per
  the Task 8 comment at `ggml-sycl.cpp:99075-99084`, that is `FLASH_ATTN_EXT`
  and the checkpoint/rollback `CPY`/`SET_ROWS` family). B2 narrows the decline:
  for exactly these op kinds, when the KV operand is KV-host-buft-resident,
  **accept** (`supports_op` returns `true`) and route to a new dispatch path;
  every *other* op with a KV-host-buft operand keeps declining unchanged
  (nothing else is designed to run this way, and the campaign's blast-radius
  promise is that ops the design didn't reason about keep today's behavior).
- Inside `ggml_sycl_graph_compute`'s node loop, when such an op is reached:
  unlike CpuExpertPool's input (which needs a D2H copy because activations are
  GPU-computed), the demoted layer's **K/V cache is already host-pinned** — the
  CPU thread reads it directly at DDR5 rates, zero copy, zero D2H wait. Only
  **Q** (GPU-computed via RoPE, small — one row per token at TG) needs a D2H
  copy, exactly mirroring the CpuExpertPool activation-copy step.
- The CPU thread pool computes attention using the **same CPU-backend
  compute kernel FLASH_ATTN_EXT already runs today under B1′'s sched
  delegation** — B2 changes *who calls it and when*, not the numerics (see §4).
  13b must resolve the exact call surface (`ggml-cpu`'s
  `ggml_compute_forward_flash_attn_ext`-family entry point) by reading
  `ggml/src/ggml-cpu/ops.cpp` at implementation time; this addendum does not
  guess a signature it has not read.
- **Output** (small — one KQV row per token at TG) is written into a pinned
  host buffer and joins the graph via the identical deferred-consumer-flush
  pattern already proven for MoE: the next SYCL-dispatched op that actually
  reads this layer's attention output (the output projection matmul) triggers
  the flush; every other concurrently-submitted GPU op (other layers' dense
  work, other MoE dispatch) is never blocked by it.
- **Memory ownership:** every new host buffer this path touches (Q staging,
  attention-output staging) must be `unified_allocate`d — reuse
  `runtime_category::HOST_COMPUTE` ("Host-pinned scratch for CPU offload
  layers", `unified-cache.hpp:3944`), which already exists for exactly this
  purpose, rather than minting a new category. No raw `sycl::malloc_host`.

---

## 3. Question 2 — event topology (no host waits)

Restated in the real primitives (§2), not an idealized `depends_on` chain:

**What the CPU attention depends on:**
- Host-pinned demoted-layer K/V: already resident, no dependency to express —
  reading it costs nothing but DDR5 latency.
- Q on host: depends on the GPU RoPE kernel that produced it. Expressed as a
  captured `sycl::event` from the async D2H copy, with the `.wait()` **deferred
  to immediately before the CPU task is enqueued** (mirrors
  `act_deferred_evt`/`act_deferred_pending`, `ggml-sycl.cpp:70378-70393,
  70517-70520) — never a wait at submission time, so GPU submission for other
  graph regions in between is never blocked by it.

**What depends on the CPU attention's output:**
- The output projection matmul (and any other consumer of this layer's
  attention output) — via the same DAG-scan
  (`ggml_sycl_op_consumes_tensor`/`ggml_sycl_tensor_depends_on`) B2 must extend
  (or directly reuse) to also recognize the new pending-attention slot(s), not
  just `g_pending_scatter`/`g_pending_cpu_pipeline`. A non-consumer op reached
  first triggers a non-blocking poll only; only the real consumer blocks.

**No host wait anywhere except:** the one bounded, unavoidable Q-readiness
join immediately before CPU dispatch (input-side, same shape as the existing
`act_deferred_evt.wait()`); the deferred-consumer blocking flush (which *is*
the edge, not a substitute for one); and teardown (draining any still-pending
attention future at graph-boundary/zone-reset time, alongside the existing
`g_pending_scatter`/`g_pending_cpu_pipeline` drains at `ggml-sycl.cpp:84019-84031`).
No `stream->wait()`/`wait_and_throw()` in the hot dispatch path, no per-op
drain, no `sycl::queue::host_task` (the codebase explicitly avoids
`host_task` for compute dispatch — "Level Zero driver has issues with it that
can corrupt event handles and cause crashes", `ggml-sycl.cpp:29621-29624` — a
`std::thread` pool sidesteps that risk entirely, another reason to mirror
CpuExpertPool rather than invent a `host_task`-based design).

---

## 4. Question 3 — correctness vs B1′ (same math, different scheduling)

B2 must not change a single floating-point operation: it changes **which code
submits the CPU attention compute and when the host thread blocks on it**, not
the compute itself. The CPU-backend `FLASH_ATTN_EXT`/checkpoint kernels stay
the only numerics involved.

**Proof obligation:** bit-identical logits (equivalently, bit-identical
generated token IDs at `--temp 0`, which is what the canonical gates already
check) between a B1′ build and a B2 build, at a fill that forces the same
demotion set on both.

**Concrete test:**
1. Force an identical, deterministic demotion set on both builds — reuse
   Task-1's forced-budget technique (`GGML_SYCL_VRAM_BUDGET_PCT=<N>`, verified
   env-var spelling per that spike) so both runs demote the same layer
   indices; confirm via the existing `[SYCL-PLAN] KV overflow re-placed...`
   WARN line naming the same layer count on both builds.
2. Run the canonical Mistral digit gate and the GPT-OSS chat gate
   (`CLAUDE.md` canonical commands) on the B1′ build and the B2 build,
   `--seed 42 --temp 0`, and diff the full stdout token stream — not just the
   pass/fail digit check, the *exact* string — between the two runs.
3. A stronger, lower-level check (host-only, no GPU, buildable by an
   implementer without lead GPU time): a small standalone harness that
   constructs one `FLASH_ATTN_EXT` node with Q/K/V/mask fixtures fed to (a) the
   `ggml-cpu` reference kernel called directly (today's B1′-equivalent path)
   and (b) the same kernel invoked through B2's dispatch wrapper (same
   arguments, minus the actual thread-pool indirection, which a unit fixture
   can stub to run synchronously), and asserts byte-identical output. This is
   the "identity case" a ported-policy change needs
   ([[ported-policy-needs-the-identity-case]]) — it proves the wrapper is a
   pure pass-through before any GPU run is spent proving the same thing at
   much higher cost.
4. In-VRAM invariant, unchanged from B1′/TKV-11: a run with **zero** demoted
   layers must take **zero** new code paths — verify the new `supports_op`
   accept-branch is provably unreached (a counter, gated behind
   `GGML_SYCL_DEBUG` exactly like Task 8's existing decline counter) rather
   than inferred from "no demotion occurred" (layout-assignment-is-not-
   dispatch-engagement — the same trap Task 8 already guards against, applied
   one level up).

---

## 5. Question 4 — roofline instrumentation

**Repo search result:** no existing STREAM-like host bandwidth microbenchmark
or dedicated VRAM-read-bandwidth test was found under `tests/` or
`ggml/src/ggml-sycl/tests/` (checked via pipe-grep and `find_file`/directory
listing, 2026-08-27) — the roofline gate needs new, small, standalone
instruments, not an existing one to reuse. `ggml/src/ggml-sycl/tests/` already
has the registration precedent for a tiny host-linkable TU
(`test-kv-runtime-demotion` clones `test-kv-slice-sizing`'s pattern,
`CMakeLists.txt:1557-1603`) — the two microbenches below should follow the
same shape: standalone `.cpp`, minimal deps, `ctest`-registered, `<1s` (or
explicitly excluded from the default sweep if it must run longer to get a
stable sustained-BW read).

### 5.1 Host DDR5 sustained read bandwidth (attention access pattern)

A tiny host-only program, no SYCL, no model: allocate a buffer sized like a
realistic demoted-layer KV region (tens to low-hundreds of MB, matching the
per-layer KV bytes at the gate's fill — read `kv_per_layer` off the same
`placement_kv_info` arithmetic B1′ already computes), and read it with the
**same stride/access pattern the CPU attention kernel actually uses** — not a
generic `memcpy` benchmark, since the whole point of the roofline is
attention-shaped reads (mostly sequential per K/V row, one pass per generated
token at TG, not fully random and not a single giant sequential scan either).
Read `ggml/src/ggml-cpu/ops.cpp`'s FA loop structure at 13b time to match the
access pattern precisely; do not guess the stride shape here.

```bash
# Sketch — 13b writes the actual program, e.g.
#   ggml/src/ggml-sycl/tests/bench-host-kv-bandwidth.cpp
# host-only, no oneAPI needed to build or run this piece:
g++ -O3 -march=native -o /tmp/bench-host-kv-bw bench-host-kv-bandwidth.cpp
/tmp/bench-host-kv-bw --bytes <kv_per_layer * n_demoted_layers> --pattern attn-row --iters 20
# report: GB/s sustained, min/median/max across iters (this host's ambient
# load is permanent per owner ruling 2026-08-07 -- report the spread, not a
# single number, and note uptime/pgrep load at capture time)
```

Use `std::chrono::steady_clock` around the read loop (this is pure host code,
so [[host-chrono-cannot-see-past-submission-backpressure]] does not apply —
that trap is specific to timing SYCL submission, not host-only compute).

### 5.2 B50 VRAM read bandwidth (device-resident path)

A minimal SYCL kernel that reads a large device buffer and reduces it (so the
compiler cannot elide the reads), timed with **SYCL device events**, not host
`chrono` around the submit call — per
[[host-chrono-cannot-see-past-submission-backpressure]], host-side timing of a
submission can be measuring queue backpressure, not device throughput.

```cpp
// Sketch for 13b:
sycl::event e = q.submit([&](sycl::handler &h){ h.parallel_for(..., [=](...){ /* strided reduce */ }); });
e.wait();
double ns = e.get_profiling_info<sycl::info::event_profiling::command_end>()
          - e.get_profiling_info<sycl::info::event_profiling::command_start>();
// bytes_read / (ns * 1e-9) = GB/s; queue must be constructed with
// sycl::property::queue::enable_profiling{} for the profiling info to be valid.
```

Run standalone (lead session, single invocation, `level_zero:1` pinned per the
selector-pinning mitigation — CLAUDE.md's iGPU-Shmem hazard applies to any new
SYCL binary the same as an existing one), report GB/s.

### 5.3 Roofline arithmetic template (owner's formula, filled in)

```
host_bytes_per_token   = kv_per_layer * n_demoted_full_attn_layers   (bytes read from DDR5 per generated token)
device_bytes_per_token = <bytes read from VRAM per token by everything that stays device-resident at this fill>
host_time    = host_bytes_per_token   / host_BW_measured    (§5.1, GB/s -> s)
device_time  = device_bytes_per_token / device_BW_measured  (§5.2, GB/s -> s)
roofline_TG  = 1 / max(host_time, device_time)               (tokens/sec, OVERLAPPED)
floor        = 0.80 * roofline_TG
```

`n_demoted_full_attn_layers` and `kv_per_layer` are read from the landed
plan's own fields at the gate's fill (`plan.kv_per_layer`,
`plan.kv_layer_count()`/demoted count — same fields Task 5's WARN line already
prints) — never assumed from the design doc's back-of-envelope "~8 of 12"
figure, which was provisional. `device_bytes_per_token` needs a 13b-time
inventory of what stays device-resident at the gate's fill (SWA KV, non-KV
weights/activations reads) — this addendum does not compute it, since it
depends on the actual landed plan at the gate's fill, not a design-time
estimate.

---

## 6. Question 5 — blast radius

**Must stay byte-identical:**
- The in-VRAM path (zero demoted layers): zero new `supports_op` branches
  taken, zero new dispatch code reached — provable via the counter in §4.4,
  not inferred.
- Every op kind other than `FLASH_ATTN_EXT` and the checkpoint/rollback
  `CPY`/`SET_ROWS` family with a KV-host-buft operand: keeps declining to
  `ggml_backend_sched` exactly as B1′ does today. B2 narrows a decline to
  specific op kinds; it does not touch the general
  `ggml_sycl_tensor_is_in_kv_host_buft` predicate or the fallback for anything
  else.
- The pinned-staging consumer at `ggml-sycl.cpp:10012` (the reason Task 8's
  decline is keyed strictly on buft identity, not `is_host`) — B2 must
  preserve that same keying discipline; nothing about B2 should touch
  `ggml_backend_sycl_host_buffer_type()` or its consumers.
- CpuExpertPool itself: B2 should mirror its pattern (possibly sharing
  infrastructure — a second persistent thread pool, or a shared one keyed by
  work kind) but must not perturb MoE dispatch's own pending-slot bookkeeping
  (`g_pending_scatter`/`g_pending_cpu_pipeline`) — a new, separate pending slot
  for attention output, extending the same DAG-scan helper rather than
  reusing MoE's slot, is the safer shape (two independent producers sharing
  one slot is a correctness hazard: a second `submit_batch`-shaped write
  before the first is flushed would silently drop the first result).

**Touches, by design:**
- `ggml_backend_sycl_device_supports_op` (narrows one decline into an accept +
  new dispatch route for two specific op-kind cases).
- `ggml_sycl_graph_compute`'s node loop (new dispatch branch + a new
  deferred-flush call alongside the existing MoE ones).
- A new persistent host thread pool (or an extension of `CpuExpertPool` to a
  second work-kind), its own ring/staging buffers via `unified_allocate`
  (`runtime_category::HOST_COMPUTE`).
- Teardown (`ggml_backend_sycl_free`-adjacent drain, alongside the existing
  MoE future drains at `ggml-sycl.cpp:84019-84031`).

**Cross-reference to the carried-forward TKV-7 review comment
(`llama.cpp-sbky` comment `c-t5b3`):** the KV-host buft is pinned to device 0's
unified cache and `tiered_kv_buft_get_name` is device-agnostic — both are
fine for the single-device B50 gate this campaign targets and are explicitly
out of scope for B2; B2 must not attempt to fix or work around either, and
should say so in its own commit/ticket trail rather than silently depending on
single-device behavior without naming it.

---

## 7. Question 6 — RED/GREEN plan

**Host-only, implementer-testable (no GPU, no lead time):**
- The DAG-scan extension (recognizing a new pending-attention slot) — pure
  logic over `ggml_tensor` graphs, testable with synthetic fixture tensors the
  same way `ggml_sycl_op_consumes_tensor`'s existing behavior could be (no
  existing unit test covers it today — check before assuming one exists).
- The identity-case harness from §4.3 (CPU kernel via direct call vs via B2's
  wrapper, byte-identical) — this is the acceptance-critical RED-first test:
  write it BEFORE the dispatch wrapper exists, confirm it fails to link/build
  (RED), then implement the wrapper (GREEN).
- The two roofline microbenches (§5.1 host-only; §5.2 needs a device but is a
  single tiny standalone kernel, not a model load — still "GPU work" per
  CLAUDE.md's never-loop property (does it load a model onto a GPU? No — a
  microbench allocating one buffer is not the hazard class those rules target,
  but it still counts as GPU access and stays lead-run per the "GPU/model-
  loading work is SERIALISED THROUGH THE LEAD SESSION" rule's broader intent).
- Any new host-buffer allocation path through `unified_allocate` — testable
  against existing mem-handle/allocation unit tests' patterns
  (`test-mem-handle-*.cpp` precedents) without a device.

**Lead-run GPU gates only:**
- The B1′-vs-B2 bit-identical logit diff (§4.2) — needs both builds run on
  real hardware.
- The full roofline gate (§5.3 arithmetic filled in from §5.1/§5.2 measured
  numbers) scored against the 80% floor.
- TKV-11's existing acceptance criteria, re-run on the B2 build: default-ctx
  gate, in-VRAM no-regression A/B, recurrent tests (unaffected by B2, but part
  of the standing gate), decline-engagement observable now split into "declined
  to CPU-via-sched" (should be zero for the two B2-owned op kinds once B2
  lands) vs "accepted into B2's own dispatch" (a new counter, same
  `GGML_SYCL_DEBUG`-gated shape as Task 8's).

---

## 8. Question 7 — estimated task decomposition for 13b

Smallest-shippable steps, ordered so each has an independent, cheap-to-verify
RED/GREEN before the next depends on it:

1. **New pending-attention slot + DAG-scan extension** (host-only). Pure
   logic, own files, unit-testable. No `supports_op` change yet — dead code
   until step 3 wires it in.
2. **Host thread pool for attention** (host-only where possible; the
   `sycl::queue` reference for pinned staging allocation is the only SYCL
   surface, mirroring `CpuExpertPool::init`'s `q` parameter). Reuse
   `CpuExpertPool`'s shape (own class or a generalized pool parameterized by
   task kind) rather than duplicating the worker/ring/staging plumbing
   wholesale.
3. **`supports_op` narrowing + dispatch wiring**: accept the two op kinds when
   KV-host-buft-resident, route to the new pool, wire the deferred flush into
   the node loop. This is the task where the in-VRAM byte-identical invariant
   (§6, counter-provable) becomes acceptance-critical.
4. **Identity-case correctness harness** (§4.3) — written before step 3 lands
   ideally (RED first), scored green once step 3 is in.
5. **Roofline microbenches** (§5.1, §5.2) — independent of 1-4, can run in
   parallel; own files, own registrations.
6. **Lead-run integration**: B1′-vs-B2 logit diff, roofline gate, full TKV-11
   re-run. Blocks TKV-11's floor criterion and TKV-12 closure.

Suggested tracker shape: one task per numbered step above (6 children of
13b), step 3 depending on 1+2, step 4 depending on 3 (or run RED before,
GREEN after), step 6 depending on 3+4+5. This mirrors the file-ownership/track
discipline the parent campaign plan already uses
(`docs/plans/2026-08-26-tiered-kv-placement.md` "File Ownership Map").

---

## Open questions for the owner (not this addendum's to resolve)

- Whether to correct `sycl-memory-design.md:74`'s "`sycl::depends_on`"
  characterization of CpuExpertPool now, or fold it into TKV-12's docs pass.
- Whether B2 gets its own thread pool or a generalized/shared one with
  CpuExpertPool (this addendum recommends a separate pending slot regardless
  of pool-sharing decision, per §6's blast-radius reasoning).
- Whether `device_bytes_per_token` in the roofline template should be measured
  empirically (a real gate run, instrumented) or estimated structurally from
  the plan — this addendum leaves it open pending the 13b implementer's
  inventory.
