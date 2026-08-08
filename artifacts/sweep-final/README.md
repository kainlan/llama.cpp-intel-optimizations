# sweep-final — the `test-llama-archs` captures behind the gemma3n cross-model bug

Tracker `llama.cpp-8t4s` (closed 2026-08-06 as non-blocking), split out of `llama.cpp-81gx`.
These are **historical captures, preserved unmodified** — raw `test-llama-archs` stdout,
ANSI escapes and interleaved GGML log lines included. Do not regenerate them in place; a
later sweep belongs in its own directory.

## The files

| file | added by | what it captures |
|---|---|---|
| `full-sweep.txt` | `98e26fac2` (2026-08-02) | full 130-arch sweep, B70 pinned `level_zero:0`, seed 1946090279 — **gemma3n `FAIL (1.14e+00)`**, the only FAIL in 412 rows |
| `sweep-at-1107a53b0.txt` | `155d172e8` (2026-08-02) | full 130-arch sweep at `1107a53b0`, seed 1647373707 — **gemma3n `OK (7.57e-07)`**, 0 FAIL in 412 rows |
| `grok-default.txt` | `98e26fac2` | `-a grok` alone at the default 30 s watchdog: `OK (2.29e-12)` |
| `shmem-trace.txt` | `98e26fac2` | two `Shmem` samples in GB (`2`, `3`) taken across the pinned full sweep |

Counts above were re-derived from the files themselves, not copied. Note that `98e26fac2`'s
commit message says "131 archs / 413 rows"; comment `c-9427` on `llama.cpp-8t4s` corrects
that to **130 archs / 412 rows** — the table's `|-----|` separator row matched the arch
regex. Both sweeps ran the same 130 architectures.

## What each supports

**`full-sweep.txt` — the FAIL of record, and the source of the repro command.**
`llama.cpp-8t4s` splits gemma3n into two bugs: the `81gx` view-at-nonzero-offset fusion
defect (isolated, `FAIL 7.04e-01` → `OK 2.17e-06`, mutation-proven) and a *second* defect
that only the first was hiding, visible as `FAIL 1.03e+00` with one predecessor model loaded
and `FAIL 1.14e+00` with 130. This file is the 130-predecessor end of that. `c-9427` records
the pre-fix (`4407acc83`) and post-fix (`d293bf2b3`) sweeps as **bit-identical**, so this
capture is not distinguishable between the two — which was the whole point: the guard
demonstrably fixes the isolated case while the sweep does not move at all.

It is also **load-bearing as an input**, not only as a record. The ticket's 2-arch reproducer
derives its `-x` exclusion set by parsing the arch column out of this exact file (see the
`llama.cpp-8t4s` description, and `c-y2yo`, which re-derived the identical set for the
historical rerun at `d293bf2b3`). The `grep -E '[a-z0-9]'` in that pipeline is required —
without it the separator row parses as an arch name and the run dies at argument validation.

**`sweep-at-1107a53b0.txt` — the disappearance.** Same binary family, three behaviour-neutral
commits later, gemma3n passes and the sweep is clean. `c-9427` ruled out the diagnostic flag,
`e2df03733`, `77f7d7c8a`'s refactor, and an arch-count change, one at a time, and raised the
leading theory: **binary layout, not a fix** — env-gated diagnostic code still changes binary
size and shifts allocation addresses, and a defect sensitive to accumulated view offsets can
flip on that alone.

The decisive experiment that theory demanded was later run (`c-y2yo`): a fresh build at
**exactly `d293bf2b3`** re-ran the 2-arch repro and got `OK (6.23e-06)` — the recorded
`FAIL (1.03e+00)` does not reproduce at its own SHA. Per the ticket's pre-registered decision
rule, that reframes the failure as intermittent rather than deterministic. `c-prvu` then
supplied the mechanism: the underlying bit1 fused-path defect is **alive and latent** at HEAD
(guard bypassed ⇒ `FAIL 1.17e+00`, deterministic), and the historical failures match **guard
evasion** — `is_weight` flipping to 1 for gemma3n's graph-internal views once another model's
weights were resident. The intermittency lived in the trigger, not the kernel.

**`grok-default.txt` — a watchdog trip that was not a regression.** grok tripped the 30 s
default watchdog during sweep work under host load 64 (codescout at 1044 % CPU). Run alone it
passes `OK (2.29e-12)`, and the full sweep completes with 0 watchdog lines at
`GGML_SYCL_OP_TIMEOUT_MS=120000`. This file is that alone-run.

**`shmem-trace.txt` — the pinned-sweep memory figure.** A full sweep pinned to `level_zero:0`
peaks at **3 GB `Shmem`**, previously unmeasured (`CLAUDE.md` carried only the single-arch
figure). This is the corroborating measurement for the selector rule: the ~195–206 GB peaks
`CLAUDE.md` records for `test-llama-archs` come from the iGPU being enumerated, and pinning
to the discrete cards collapses them.

## Standing rule attached to these files

`llama.cpp-8t4s` closed with an explicit reopen condition: **any future gemma3n FAIL in any
sweep reopens the ticket immediately, and the failing build directory must be preserved
before anything else is run.** A bit-identical rerun of the original binary is already
impossible — its build directory is gone, and a rebuild at a different absolute path has a
different layout, which is the layout theory's own prediction rather than a flaw in the
experiment.
