# Tasks 19/21 re-verification at HEAD `da41f06e7` — 2026-08-08

Owner ruling (llama.cpp-ona8 comment `c-nxv0`, decision 2): rather than consuming the
`772798e91` recordings across 155 commits of drift, re-run the Task 19 accepted restored
set and the three Task 21 (llama.cpp-ktlb) gates at current HEAD. This directory is that
recording; it supersedes nothing in `../` (the 772798e91 evidence stands as history).

- Build: `./scripts/sycl-build.sh` rc=0 at `da41f06e7`, `GGML_SYCL:BOOL=ON`.
- Runner: each test individually via the plan's `run_gpu` wrapper (GPU.lock, settle-wait,
  post-run 5 s sample), selector `level_zero:0`, `timeout 900` per test, serial, `-j` never used.
- Registration check first: all 50 names from `../task19-names.txt` (union with the ktlb
  gates, which are members of the set) registered at HEAD — `MISSING=0`.

## Result: 48 PASS / 1 SKIP77 / 1 FAIL of 50

| vs `772798e91` (`../task19-results.tsv`) | then | now |
|---|---|---|
| pass | 38 | **48** |
| skip77 (`test-mmvq-q8-0-streaming-bench`, opt-in) | 1 | 1 |
| fail | 11 | **1** |

All ten failures whose RCA tickets closed since (`99ke`, `43uy`, `wmc2`, `n3pw`, `qvid`,
`sfe9`, `mequ` et al.) flipped green at HEAD, run-confirmed — the "closed ≠ green" caveat
in `../README.md` is now discharged for those ten. The three ktlb gates
(`test-sycl-moe-handle-resolution`, `test-sycl-cpu-dispatch`, `test-mul-mat-host-streaming`)
PASS, re-confirming llama.cpp-ktlb at HEAD.

The sole FAIL is `test-dmmv-q4-0-coalesced` (rc=8) — llama.cpp-szv8, open by design:
the owner ruled it certification-blocking the same day and its RCA/fix is in progress.
Memory trace: `Shmem` flat across the sweep (per-test post-run samples in the session log).
