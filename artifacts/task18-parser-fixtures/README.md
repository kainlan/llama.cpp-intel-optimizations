# Fixtures for `scripts/parse-sycl-bench-matrix.py --self-test`

The parser exists to stop the Task 20 performance gate passing vacuously. A
parser with that job needs its own positive control, or it repeats the trap it
was written to prevent: a checker nobody ever saw fail is indistinguishable from
a checker that cannot fail.

These fixtures are that control. Run:

```sh
python3 scripts/parse-sycl-bench-matrix.py --self-test
```

It must report **31/31** and exit 0. If it does not, the parser's verdicts are
not trustworthy and the gate must not be certified from them.

Ten cases cover the original **merge-cert** matrix (numeric floor/band, gates
merges); nine more (llama.cpp-z0wt, plan task L4) cover the **long-prompt**
matrix (report-only, no gate declared yet) and its `--table` markdown output;
five more (llama.cpp-z0wt, scope addition) cover `--partial-arm` as
`evaluate()` sees it -- the accept-fewer-than-`runs`-samples exception for a
declared arm that could not be fully measured; seven more (llama.cpp-z0wt,
review rounds 6-7) call `parse_partial_arms()` directly, since none of the
other cases ever exercise its own `ARM=REASON` splitting and validation --
they all hand `evaluate()` an already-parsed `{arm: reason}` dict.

## What the cases prove

The point is not that bad input is rejected — it is that the parser demonstrably
returns **all three** of its exit codes, so a `PASS` is a measurement rather than
the only answer it is capable of giving.

| exit | meaning | cases covering it |
|---:|---|---|
| 0 | every arm present and parseable (merge-cert: also within gate) | all-arms-good (both matrices, incl. `--table`) |
| 1 | VERDICT FAIL — parsed fine, a mean missed its floor/band | below-floor (merge-cert only — unreachable for long-prompt) |
| 2 | INPUT/PARSE FAILURE — no verdict could be computed | the rest |

The exit-1 case matters most and is the easiest to omit. Without it, a parser
that hard-codes "everything is fine" would still pass every other case here.
For long-prompt, exit 1 is impossible by construction (every arm is kind
REPORT, so `check_gate()` always returns `ok=True`) — the cases instead prove
exit 2 is still reachable (a missing arm, a missing `pp8192` row, the
`--table`-only n_ctx-disagreement check, a sample with no achieved n_ctx at
all — review round 1 found this silently defaulting to 8320 instead of
failing closed — and an achieved n_ctx below the arm's own prompt length,
i.e. the MAX line found in the file belonged to some other, smaller test —
review round 2) so a long-prompt PASS is still a measurement, not the only
answer available. A `--runs 1 --table` case also proves the sample-stdev
column renders `n/a` rather than a misleading `0.00`, in both the report and
the table, when there is no second sample to compute a spread from (review
rounds 1 and 2) — the case checks for both `"sd    n/a"` (the report line)
and `"± n/a"` (the table cell) in stdout, not just one of the two rendering
sites the claim covers (review round 3).

Five more cases (llama.cpp-z0wt, scope addition) cover `--partial-arm ARM=
REASON`: a declared arm with one real sample passes with the exact `PARTIAL
(n=1 of 5)` report text, the matching mention in the verdict line, and the
`mean (n=1, REASON)` table cell all checked in stdout (not merely exit 0 —
the same "an exit code alone can't tell a hardcoded value from a real one"
reasoning as the `--table` all-good case above); a declared arm with zero
samples, one with the full sample count, an undeclared arm name, and the
flag combined with `--matrix merge-cert` each drive exit 2. None of these
needed new fixtures — they reuse `b70-pp8192-good.txt` (long-prompt) and
`b50-mistral-good.txt` (merge-cert) at reduced or full sample counts.

Seven further cases call `parse_partial_arms()` itself, not `evaluate()`, and
prove something the five above cannot: that the function fails closed on its
own malformed input, independent of anything `evaluate()` later does with
the result. A `--partial-arm` string with no `=` at all, an empty arm name,
an empty reason, a reason containing `|` or a newline (either would split a
markdown table row — the newline case was confirmed live, not just reasoned
about), and a repeated arm name each raise `ParseError` rather than silently
producing a corrupted or ambiguous `{arm: reason}` dict; a well-formed
`a=b=c` case proves this isn't just blanket rejection — it still returns
`{"a": "b=c"}`, splitting on only the FIRST `=`. Without these, a regression
inside `parse_partial_arms()` itself could sit completely uncovered: every
case above hands `evaluate()` an already-parsed dict, so none of them ever
calls this function at all.

## Provenance — what is real and what is reconstructed

Stated explicitly, because a fixture that silently drifts from the real log
format turns a green self-test into a false all-clear.

**From committed real captures:**

- The free-VRAM line shape,
  `llama_prepare_model_devices: using device SYCL0 (...) - NNNNN MiB free`,
  is verbatim from `captures/16-soak-mistral-bench-r30.err` and
  `captures/17-soak-gptoss-bench-r15.err`.
- The B50 free value (14677 / 14679 MiB) is what those two captures actually
  measured on a healthy card.

**Reconstructed from `tools/llama-bench/llama-bench.cpp` (`markdown_printer`),
because no `llama-bench` results table is committed anywhere in this repo:**

- The header (`print_header`: `model`, `size`, `params`, `backend`, `ngl`, then
  `fa` only when `flash_attn` differs from its default, then `test`, `t/s`).
- The `test` cell spelling `pp512` / `tg128` (`print_test`).
- The `t/s` cell format `%.2f ± %.2f` (`print_test`) — and the fact that the `±`
  spread is *within one process*, which is exactly why the gate averages across
  five processes instead of reading it.
- The `fa` cell value `1`, since `flash_attn` is an INT field and
  `LLAMA_FLASH_ATTN_TYPE_ENABLED == 1` (`include/llama.h:193`).

**Reconstructed from documentation, with no committed capture to check against:**

- The B70 free-VRAM value 32602 MiB, from
  `docs/backend/sycl-perf-baselines.md`. **No committed B70 `-v` capture exists.**
  When one does, check it against this fixture and against the `MIN_FREE_MIB`
  floor in the parser.

**Throughput values** are the documented baselines from
`docs/backend/sycl-perf-baselines.md`, so the good fixtures sit inside their
arms' real gates rather than at invented numbers.

## Long-prompt fixture provenance (llama.cpp-z0wt, 2026-09-07)

The long-prompt matrix's own fixtures are trimmed from **real, guarded
captures**, not reconstructed from source like the merge-cert set above. The
lead archived 56 guarded logs (11 full cells plus a partial B50 Mistral pp8192
cell) and one `.FAULT` under
`/Apps/llama.cpp/artifacts/perf-6ae16115c-longprompt/` (each an
n=1..5 `llama-bench -p <pp> -n 128 -fa 1 -r 5 -v` process, one per
card/model/pp/n); this task's committed fixtures are one `n=1` sample per
(card, pp) — reused across all three models, since the parser doesn't care
which model a result table names, only whether the table and free-VRAM line
parse:

- `b70-pp2048-good.txt` — trimmed from `b70-mistral-pp2048-1.log` (VALID,
  captured 2026-09-07): the free-VRAM line, both `llama_context: n_ctx = N`
  lines (2048 for the `pp2048` test, 256 for `tg128`), one markdown
  header/separator, and the two `fa=1` rows.
- `b70-pp8192-good.txt` — same trim, from `b70-mistral-pp8192-1.log` (VALID,
  same capture run). Notably its pp-test `n_ctx` line reads **8192**, not the
  naive `n_prompt+n_gen=8320` — real evidence that the parser's "unless a log
  shows a different n_ctx" fallback is not a hypothetical.
- `b50-pp2048-good.txt`, `b50-pp8192-good.txt` — **real**, trimmed from
  `b50-gemma4-pp2048-1.log` / `b50-gemma4-pp8192-1.log` (VALID, captured
  2026-09-07, 18:10:05 / 18:11:22): free VRAM 14618 MiB (above the 14000 MiB
  contamination floor), pp-test `n_ctx` 2048 / 8192. These replace an earlier
  revision that derived them from the B70 fixtures with only the device line
  swapped, because no `b50-*-longprompt` log existed yet at the time; real
  B50 logs landed before review round 1 finished, so the derived versions
  were replaced rather than kept as a caveat.
- `pp8192-missing-pprow.txt` — synthetic: the free-VRAM line, one `n_ctx`
  line, and only the `tg128` row (no `pp8192` row) — drives the "missing
  wanted test" parse error for a long-prompt arm.
- `b70-pp8192-ctx-mismatch.txt` — `b70-pp8192-good.txt` with its pp-test
  `n_ctx` line changed from `8192` to `4096`; used as one of five samples
  for an arm so the five processes disagree on achieved `n_ctx`, driving the
  `--table`-only disagreement check (exit 2).
- `b70-pp8192-no-ctx.txt` — `b70-pp8192-good.txt` with both `llama_context:
  n_ctx = N` lines removed entirely; used as one of five samples so that
  sample's achieved n_ctx is `None`, driving the fail-closed
  no-achieved-n_ctx check (exit 2, never a silent `n_prompt+n_gen` default).
- `b70-pp8192-ctx-too-low.txt` — `b70-pp8192-good.txt` with only the pp-test's
  own `n_ctx = 8192` line removed, leaving just the tg-test's `n_ctx = 256`.
  All five samples in the arm agree on 256, so the disagreement check alone
  would pass it — this drives the *separate* sanity-floor check (review
  round 2): an achieved n_ctx (256) below the arm's own prompt length (8192)
  means the MAX line found in the file belongs to some other, smaller test,
  not the long-prompt one.

## Files

| file | role |
|---|---|
| `b50-mistral-good.txt`, `b50-gptoss-good.txt`, `b70-mistral-good.txt`, `b70-gptoss-good.txt` | merge-cert: one in-gate sample per arm |
| `b50-mistral-below-floor.txt` | merge-cert: 1000.00 / 40.00 — under the 1128 / 44.6 floor; drives exit 1 |
| `empty.txt` | zero bytes; an empty capture is VOID, not clean |
| `no-vram-line.txt` | what a run **without `-v`** looks like — the log callback is nulled and free VRAM is never printed |
| `unparseable-ts.txt` | `t/s` cell reads `N/A` |
| `no-fa-column.txt` | no `fa` column, i.e. not the `-fa 1` matrix |
| `low-free-vram.txt` | 13800 MiB free — the ~13.8 GB contamination case the perf doc names |
| `b70-pp2048-good.txt`, `b70-pp8192-good.txt` | long-prompt: real one-process trims (b70-mistral), reused across all 3 models on B70 |
| `b50-pp2048-good.txt`, `b50-pp8192-good.txt` | long-prompt: real one-process trims (b50-gemma4), reused across all 3 models on B50 |
| `pp8192-missing-pprow.txt` | long-prompt: no `pp8192` row — drives the missing-test parse error |
| `b70-pp8192-ctx-mismatch.txt` | long-prompt: pp-test `n_ctx` differs from its siblings — drives the `--table` n_ctx-disagreement check |
| `b70-pp8192-no-ctx.txt` | long-prompt: no `n_ctx` line at all — drives the fail-closed no-achieved-n_ctx check |
| `b70-pp8192-ctx-too-low.txt` | long-prompt: only n_ctx=256 present (tg-test's, not pp-test's) — drives the achieved-below-prompt-length sanity floor |

`.txt`, not `.log`: `.gitignore` line 17 is `*.log`, so fixtures committed under
that extension would be silently dropped and the self-test would run against
nothing.
