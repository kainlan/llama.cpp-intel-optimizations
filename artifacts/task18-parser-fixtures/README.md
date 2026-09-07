# Fixtures for `scripts/parse-sycl-bench-matrix.py --self-test`

The parser exists to stop the Task 20 performance gate passing vacuously. A
parser with that job needs its own positive control, or it repeats the trap it
was written to prevent: a checker nobody ever saw fail is indistinguishable from
a checker that cannot fail.

These fixtures are that control. Run:

```sh
python3 scripts/parse-sycl-bench-matrix.py --self-test
```

It must report **16/16** and exit 0. If it does not, the parser's verdicts are
not trustworthy and the gate must not be certified from them.

Ten cases cover the original **merge-cert** matrix (numeric floor/band, gates
merges); six more (llama.cpp-z0wt, plan task L4) cover the **long-prompt**
matrix (report-only, no gate declared yet) and its `--table` markdown output.

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
exit 2 is still reachable (a missing arm, a missing `pp8192` row, and the
`--table`-only n_ctx-disagreement check) so a long-prompt PASS is still a
measurement, not the only answer available.

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
lead archives 60 guarded logs under
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
- `b50-pp2048-good.txt`, `b50-pp8192-good.txt` — **derived, not real**: no
  `b50-*-longprompt` log existed yet when this task's implementation and
  self-test were otherwise done (the archive fills b70 first, then b50, per
  the task brief). Each is the matching `b70-pp*-good.txt` with only the
  device line replaced by the real B50 form seen on this boot,
  `Intel(R) Arc(TM) Pro B50 Graphics) (unknown id) - 14618 MiB free`
  (above the 14000 MiB contamination floor). **Swap these for real
  `b50-*-pp*-1.log` trims once the archive's B50 half lands.**
- `pp8192-missing-pprow.txt` — synthetic: the free-VRAM line, one `n_ctx`
  line, and only the `tg128` row (no `pp8192` row) — drives the "missing
  wanted test" parse error for a long-prompt arm.
- `b70-pp8192-ctx-mismatch.txt` — `b70-pp8192-good.txt` with its pp-test
  `n_ctx` line changed from `8192` to `4096`; used as one of five samples
  for an arm so the five processes disagree on achieved `n_ctx`, driving the
  `--table`-only disagreement check (exit 2).

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
| `b70-pp2048-good.txt`, `b70-pp8192-good.txt` | long-prompt: real one-process trims, reused across all 3 models on B70 |
| `b50-pp2048-good.txt`, `b50-pp8192-good.txt` | long-prompt: **derived** from the B70 fixtures (see provenance above) — swap for real trims once available |
| `pp8192-missing-pprow.txt` | long-prompt: no `pp8192` row — drives the missing-test parse error |
| `b70-pp8192-ctx-mismatch.txt` | long-prompt: pp-test `n_ctx` differs from its siblings — drives the `--table` n_ctx-disagreement check |

`.txt`, not `.log`: `.gitignore` line 17 is `*.log`, so fixtures committed under
that extension would be silently dropped and the self-test would run against
nothing.
