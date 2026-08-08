# Task 19 — accepted restored-test lead gate, executed evidence

Plan Task 19 of `docs/plans/2026-08-02-sycl-merge-readiness.md`, tracker
`llama.cpp-8kyi` (closed 2026-08-06). This is the **only recorded runtime
execution of the accepted restored-test set**. The narrative reading of these
files lives in `docs/backend/sycl-test-inventory.md`, section
"Task 19: accepted-set lead gate — executed results".

Lifted here from a session scratchpad, which is not durable storage. An
uncommitted capture that a later step depends on is a capture that will not be
there — and an empty or missing evidence directory means *not verified*, never
"nothing observed".

## Provenance

| field | value |
|---|---|
| build SHA | `772798e91814340d07f20a4e9e3969427759ed2d` |
| build command | `run_build restored-build ./scripts/sycl-build.sh` |
| build rc | 0 |
| backend present | `GGML_SYCL:BOOL=ON`; `ldd build/bin/llama-completion \| grep -cE 'libggml-sycl\|libsycl'` → 2 |
| selector | `ONEAPI_DEVICE_SELECTOR=level_zero:0` (Arc Pro B70) |
| per-test command | `timeout 300 ctest --test-dir build -R "^<name>$" --output-on-failure` |
| concurrency | serial, one name per invocation, under `GPU.lock` |
| window | 2026-08-06 16:24:14 → 16:29:05 (−04:00) |

## Files

| file | what it is |
|---|---|
| `task19-run.sh` | the exact runner. The commands in it are the commands that ran — **a historical record, not a re-runnable script**: its `SCRATCH=` path points at the now-dead `d3b0c71c` session scratchpad and it `cd`s to `/Apps/llama.cpp`, so re-running it means editing both paths first |
| `task19-names.txt` | the 50 accepted CTest names fed to the runner |
| `task19-registered.txt` | `ctest -N` output reconciled against the accepted set before execution |
| `task19-results.tsv` | one row per name: verdict, non-vacuity control, `Shmem` before → after |
| `failing-logs/` | the full captured output of each of the 11 failing tests, one `.txt` per test |

The captures are `.txt`, not `.log`, because the repository's `.gitignore` line
17 is `*.log` — committing them under their original extension silently
committed nothing, leaving this README pointing at an empty directory. Same
convention as `artifacts/triage/`. Keep it.

## Result

38 pass (rc 0) · 1 skip (rc 77, the opt-in `test-mmvq-q8-0-streaming-bench`) ·
11 fail (ctest rc 8) · 50 total.

`Shmem` 5602656 kB → 5623064 kB across the whole sweep (5.60 → 5.62 GB); final
`MemAvailable` 207398448 kB. The runner would have aborted the sweep had `Shmem`
passed 100 GB after any test; it never approached it.

## Why the results are not vacuous

Every row carries `matched=1` — the count of `^\s*Start ` lines in that test's
log, i.e. how many tests the `-R` regex actually selected. A count of 0 writes a
`ZERO_MATCH` row instead. **No `ZERO_MATCH` row exists in the TSV**, so no
verdict here is a filter that selected nothing and reported success. Check it:

```sh
grep -c ZERO_MATCH task19-results.tsv      # 0
grep -c 'matched=1' task19-results.tsv     # 50
```

That control is the whole reason to trust the 38. A zero-match `ctest -R` exits
0, and so does a passing test.

## Standing caveat

Ten of the eleven failures have since had their RCA tickets closed at later
SHAs. **That is not the same as ten tests now passing** — no re-run of this set
has been recorded since `772798e91`, so the 38/1/11 split remains the current
executed result. `llama.cpp-szv8` (`test-dmmv-q4-0-coalesced`) is still open.
