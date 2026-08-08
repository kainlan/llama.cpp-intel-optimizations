# Fixtures for `scripts/parse-sycl-bench-matrix.py --self-test`

The parser exists to stop the Task 20 performance gate passing vacuously. A
parser with that job needs its own positive control, or it repeats the trap it
was written to prevent: a checker nobody ever saw fail is indistinguishable from
a checker that cannot fail.

These fixtures are that control. Run:

```sh
python3 scripts/parse-sycl-bench-matrix.py --self-test
```

It must report **10/10** and exit 0. If it does not, the parser's verdicts are
not trustworthy and the gate must not be certified from them.

## What the cases prove

The point is not that bad input is rejected — it is that the parser demonstrably
returns **all three** of its exit codes, so a `PASS` is a measurement rather than
the only answer it is capable of giving.

| exit | meaning | cases covering it |
|---:|---|---|
| 0 | every arm present, parseable, within gate | all-arms-good |
| 1 | VERDICT FAIL — parsed fine, a mean missed its floor/band | below-floor |
| 2 | INPUT/PARSE FAILURE — no verdict could be computed | the other eight |

The exit-1 case matters most and is the easiest to omit. Without it, a parser
that hard-codes "everything is fine" would still pass every other case here.

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

## Files

| file | role |
|---|---|
| `b50-mistral-good.txt`, `b50-gptoss-good.txt`, `b70-mistral-good.txt`, `b70-gptoss-good.txt` | one in-gate sample per arm |
| `b50-mistral-below-floor.txt` | 1000.00 / 40.00 — under the 1128 / 44.6 floor; drives exit 1 |
| `empty.txt` | zero bytes; an empty capture is VOID, not clean |
| `no-vram-line.txt` | what a run **without `-v`** looks like — the log callback is nulled and free VRAM is never printed |
| `unparseable-ts.txt` | `t/s` cell reads `N/A` |
| `no-fa-column.txt` | no `fa` column, i.e. not the `-fa 1` matrix |
| `low-free-vram.txt` | 13800 MiB free — the ~13.8 GB contamination case the perf doc names |

`.txt`, not `.log`: `.gitignore` line 17 is `*.log`, so fixtures committed under
that extension would be silently dropped and the self-test would run against
nothing.
