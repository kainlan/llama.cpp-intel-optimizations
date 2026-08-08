# oze0 — graph-boundary drain evidence

Tracker `llama.cpp-oze0` (closed 2026-08-01, merged as `582cf1dfe`): *"test-thread-safety
SEGFAULTs: graph launch resets the shared SCRATCH zone while another context legitimately
holds get_rows scratch."*

Everything here was produced during that ticket and is **historical evidence, preserved
unmodified**. Do not re-run these files as a gate and do not edit them to match later
findings — where a later measurement contradicts an earlier one, the contradiction is
recorded inside the file (see `repro-drain-barrier-output.txt`) rather than erased.

## The files

| file | added by | what it is |
|---|---|---|
| `repro-drain-barrier.cpp` | `d33fbc742` | v1 device-free reproducer: one owner thread holding an allocation across a boundary crossing, one thread calling `drain_retained_handles(true)` |
| `repro-drain-barrier-multi.cpp` | `a3c200565` | v2 of the same, parameterised on owner count, histogramming *how many* holders the boundary finds |
| `repro-drain-barrier-output.txt` | `d33fbc742`, extended by `a3c200565` and `f2804014f` | the measured output of both, plus the retraction of the attribution the first version supported |
| `mhyw-lead-gdb-backtrace.txt` | `a3c200565` | gdb capture of a run that aborted at `ggml-sycl.cpp:78815` `[GRAPH-COMPUTE] arena scratch unavailable before graph launch` |
| `fixture-staging-note.txt` | `049f9220b` | why `test-thread-safety`'s `stories15M-q4_0.gguf` fixture goes missing under direct binary invocation |

## What the reproducers establish

`ggml_sycl_graph_boundary_reset_arenas()` treats `drain_retained_handles(true)` as a barrier
before resetting the shared SCRATCH zone. It is not one: the drain waits on `state.queue` and
`state.active`, and a thread that owns an allocation but has **not yet** handed its handle to
`retain_handles_until_event()` appears in neither. The drain therefore reports "all clear"
while another context still owns scratch.

Measured device-free — no SYCL device, no oneAPI selector, so the result is a property of the
retained-handle machinery rather than of any card:

- v1, `kHoldWindow` 50 µs: **12 of 12 completed runs reproduce, at 79–99 % of boundary
  crossings** (3 further runs hit the 180 s harness limit; none ever reported zero).
- An earlier variant with a 200-cycle hold window — far shorter than a real `stream_dma`
  submission — reproduced only **1 in 5**. The window was widened because the short one
  flattered the barrier. Both figures are in the output file. **Do not shrink it.**
- v2, holders found per crossing: `owners=1` → 99.26 % find 1; `owners=2` → 86.57 % find 2;
  `owners=3` → 89.40 % find 3. **"0 live holders" — the only case in which the reset can
  succeed — is 0.06–0.74 % of crossings in every configuration.**

The retraction inside `repro-drain-barrier-output.txt` matters as much as the numbers. The
ticket's first attribution argued from *multiplicity* ("4 identical `get_rows:seq_device`
allocations = accumulation"); the lead then measured **one** live allocation on the merge
commit, and that argument did not survive. The corrected discriminator is
**stranding grows** (handles accumulate in `graph_unwaitable` until some other context
invalidates a graph) versus **TOCTOU is bounded and varies** (bounded by the number of
concurrent contexts). v1 could only ever observe 0 or 1 because it had exactly one owner
thread — reading a harness parameter as a prediction about the system. v2 exists to make the
holder count measured rather than assumed.

## What `mhyw-lead-gdb-backtrace.txt` is for

It is the **positive control** for abort probes: a run whose backtrace carries the
`ggml_abort` frame, so "this run aborted" is certainly true. Scored against it
(`llama.cpp-oze0` comment `c-5g42`):

| probe | count | verdict |
|---|---:|---|
| `GGML_ABORT` | 0 | **void** — zero on a run that provably aborted; the macro is consumed by the preprocessor and the literal never reaches a stream |
| `arena scratch unavailable` | 1 | valid |
| `ggml-sycl\.cpp:[0-9]+:` | 1 | valid |
| `ggml_abort` (lowercase) | 2 | **trap** — matches only because gdb symbolised the frame; returns 0 on a plain run |

That table is why criterion 5 of the ticket ("no new aborts") was closed as **NOT MET**
rather than green: the only evidence it had been met was a `GGML_ABORT` grep, a probe that
can never fire. The general rule — grep the string the code *prints*, not the identifier that
produces it, and confirm it against a run where the condition is known true — is now in
`CLAUDE.md`.

## What cites this directory

- **`artifacts/fbj5/README.md`** (plan Task 12c, tracker `llama.cpp-fbj5`) names
  `repro-drain-barrier*.cpp` and `repro-drain-barrier-output.txt` as its **historical
  evidence** and marks them untouched. `tests/test-sycl-retained-handoff-contended.cpp` is
  the productionised successor: it takes v2's finding that the holder count is a distribution
  bounded by the number of contexts and runs scenarios at 1, 2 and 3 owners, and it inherits
  the 50 µs `kHoldWindow` for the reason recorded above. The fbj5 gate's confidence that its
  mutation flips the test comes from the 79–99 %-per-crossing figure measured here.
- **`llama.cpp-oze0` comments `c-pppc`** (the retraction and the barrier measurement) and
  **`c-65f0`** (the acceptance-criteria verdict and the holder-count histogram).

## Scope

The escape — deterministic, unbounded stranding of live scratch in `graph_unwaitable` — was
fixed by `cf802823c` and is proven on its own CPU-only gate
(`tests/test-sycl-graph-retention-scope.cpp`). What these artifacts measure is the
**residual TOCTOU**, which the fix does not close and which `llama.cpp-iiff` Phase 2
dissolves by removing the reset. Two separate statements; conflating them is what produced
the confusion the retraction corrects.
