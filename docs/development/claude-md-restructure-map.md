# Where each part of the old `CLAUDE.md` went

`CLAUDE.md` is the instruction file that Claude Code loads at the start of every session. On
2026-11-16 it was cut from 1463 lines to a short file of rules and exact commands. The history,
measurements and reasoning moved to the pages listed below. Nothing was deleted. This page is the
index from the old layout to the new one.

The old file is `CLAUDE.md` at commit `423fd3e78`. Read it with `git show 423fd3e78:CLAUDE.md`. All
line ranges below are lines of that version.

## How the move was done

- Each old line went to exactly one place, word for word, or is listed under "Lines not moved" with
  the reason. A script checked that all 1463 lines are accounted for.
- `CLAUDE.md` keeps each standing rule as a short sentence with its exact command. So a rule can
  appear twice: short in `CLAUDE.md`, and in full where the old text now lives.
- A "path-scoped rule file" is a file under `.claude/rules/` with a `paths:` list at its top. Claude
  Code loads it when a session reads or edits a file that matches the list. It is used here for
  rules that matter only when those files are open.
- Corrections made on 2026-11-16 are marked "Correction" in the destination pages. The moved text
  itself was not changed, with one exception listed under "Text changed in the move".
- `docs/backend/gpt-oss-testing.md` was not changed. `scripts/sycl-moe-exec-gate.sh --make-prompt`
  reads the bytes of that file to build a prompt, so adding text to it would change the prompt.

## New and changed files

| file | kind | loads |
|---|---|---|
| `CLAUDE.md` | rules and exact commands | every session |
| `.claude/rules/sycl-backend.md` | path-scoped rule file | with files under `ggml/src/ggml-sycl/` and the SYCL, unified-cache and mem tests |
| `.claude/rules/cpp-conventions.md` | path-scoped rule file | with any C or C++ file |
| `.claude/rules/ctest-registration.md` | path-scoped rule file | with `tests/CMakeLists.txt`, `ggml/src/ggml-sycl/CMakeLists.txt`, the guard script and `tests/merge-guards/` |
| `.claude/rules/arg-parser.md` | path-scoped rule file | with `common/arg.cpp`, `common/arg.h`, `tests/test-arg-parser.cpp` |
| `docs/backend/sycl-host-safety.md` | new page | when read |
| `docs/backend/sycl-verification-traps.md` | new page | when read |
| `docs/development/sycl-fork-guide.md` | new page | when read |
| `docs/backend/compute-runtime.md` | existing page, one section added at the end | when read |
| `docs/backend/sycl-perf-baselines.md` | existing page, two sections added at the end | when read |

## Map

One row per old heading, warning block or rule bullet. "Kept in `CLAUDE.md` as a rule" says whether
the short file still states the rule, and under which heading.

| old heading or first words | old lines | what it held | where it lives now (file, heading) | kept in `CLAUDE.md` as a rule |
|---|---|---|---|---|
| Heading: CLAUDE.md | 1-4 | Title and a one-line statement of purpose. | `CLAUDE.md`, "top of file" | no (not a rule; replaced by the ownership statement) |
| Heading: Build Commands (Intel SYCL) | 5-11 | Source oneAPI before running the binaries. | `docs/development/sycl-fork-guide.md`, "1. Build detail"; `CLAUDE.md`, "Build Commands (Intel SYCL)" | yes |
| Heading: Build | 12-39 | The build script, its four common forms, build time, what `-DGGML_SYCL_F16=ON` does. | `docs/development/sycl-fork-guide.md`, "1. Build detail"; `CLAUDE.md`, "Build Commands (Intel SYCL)" | yes (the four commands and a short note) |
| Heading: Ninja vs Make | 40-46 | Why the script uses Ninja, and `-c` for a build directory made by another generator. | `docs/development/sycl-fork-guide.md`, "1. Build detail"; `CLAUDE.md`, "Build Commands (Intel SYCL)" | yes (one sentence) |
| Heading: Running Tests | 47-48 | Section heading. | `CLAUDE.md`, "Running Tests" | yes (heading kept) |
| `-j $(nproc)` on the full suite OOM-kills this host. | 49-101 | `-j $(nproc)` on the full suite OOM-kills the host. The three allowed forms of `ctest` with exact commands, and the single-test form. | `docs/backend/sycl-host-safety.md`, "1. The full-suite sweep and `-j`"; `CLAUDE.md`, "Running Tests" | yes (all three forms, exact) |
| `-j N` IS A MEMORY MULTIPLIER, NOT A CPU THROTTLE. This file prescribed | 102-137 | `-j` multiplies memory. Why the full sweep is `-j 1` and not a better filter. | `docs/backend/sycl-host-safety.md`, "1. The full-suite sweep and `-j`"; `CLAUDE.md`, "Running Tests" | yes (`-j 1` is not negotiable) |
| `-LE 'residency\|mem-handle\|cache'` does NOT exclude `test-backend-ops` | 138-168 | The label filter does not exclude `test-backend-ops`, so form 2 also carries `-E`. Run the guard before a filtered sweep. Command to locate the registration. | `docs/backend/sycl-host-safety.md`, "2. Label filters fail open"; `CLAUDE.md`, "Running Tests" | yes (guard first, exact filter); the locate command is in the doc only |
| Since the 694ec2354 upstream merge (llama.cpp-n77l), `test-backend-ops` and | 169-216 | `test-backend-ops` and `test-llama-archs` are binaries, not ctests. New model loaders under the `main` label. How the guard derives the loader set. What to do when the guard names a test. | `docs/backend/sycl-host-safety.md`, "2. Label filters fail open"; `CLAUDE.md`, "Running Tests" | yes (binary commands, label rule, change file and script together) |
| `test-backend-ops` is not the only unlabelled member of that family. | 217-241 | `test-llama-archs` also carries only `main`. Looping it caused two global OOMs. Do not loop any GPU-allocating test. | `docs/backend/sycl-host-safety.md`, "3. Never repeat a model-loading test"; `CLAUDE.md`, "Running Tests" | yes (never repeat a model-loading test) |
| A post-run `free -g` sampled too early manufactures a FALSE OOM scare - let the | 242-273 | A `free -g` reading taken too early is a false alarm. Read `Shmem` after a 5 s settle. Pure-Python gates are safe. Check the GPU after an OOM. | `docs/backend/sycl-host-safety.md`, "4. Read `Shmem`, not `free`"; `docs/backend/sycl-host-safety.md`, "5. Tests that are always safe, and the check after an OOM"; `CLAUDE.md`, "Running Tests" | yes (both `Shmem` commands, the `journalctl` command, the pure-Python list) |
| GPU/model-loading work is SERIALISED THROUGH THE LEAD SESSION. Subagents | 274-304 | GPU and model-loading work runs only in the lead session. What subagents may do. | `docs/backend/sycl-host-safety.md`, "6. GPU work runs in the lead session"; `CLAUDE.md`, "Running Tests" | yes |
| Heading: Code Formatting | 305-306 | Section heading. | `CLAUDE.md`, "Code Formatting" | yes (heading kept) |
| `git clang-format` does not exist on this machine | 307-328 | `git clang-format` does not exist here. The working commands. | `docs/development/sycl-fork-guide.md`, "2. Code formatting"; `CLAUDE.md`, "Code Formatting" | yes |
| `clang-format-19 -i` on a whole file is actively destructive here, which is the | 329-351 | `clang-format-19 -i` rewrites about 180 unrelated lines. How to recover. | `docs/development/sycl-fork-guide.md`, "2. Code formatting"; `CLAUDE.md`, "Code Formatting" | yes (never `-i`) |
| `git-clang-format-19 --staged` operates on the WHOLE INDEX, not on your files | 352-378 | `git-clang-format-19 --staged` works on the whole index. Which command to use alone and with other agents live. | `docs/development/sycl-fork-guide.md`, "2. Code formatting"; `CLAUDE.md`, "Code Formatting" | yes |
| Heading: Project Architecture | 379-380 | Section heading. | `CLAUDE.md`, "Build Commands (Intel SYCL)" | no (a pointer to the fork guide) |
| Heading: Core Directories | 381-388 | Core directories. | `docs/development/sycl-fork-guide.md`, "3. Repository map" | no (reference, not a rule) |
| Heading: Key Binaries (in `build/bin/`) | 389-396 | Key binaries in `build/bin/`. | `docs/development/sycl-fork-guide.md`, "3. Repository map" | no (reference, not a rule) |
| Heading: Backend Structure (`ggml/src/`) | 397-403 | Backend directories under `ggml/src/`. | `docs/development/sycl-fork-guide.md`, "3. Repository map" | no (reference, not a rule) |
| Heading: SYCL Backend Structure (`ggml/src/ggml-sycl/`) | 404-417 | Key files of the SYCL backend. | `docs/development/sycl-fork-guide.md`, "3. Repository map" | no (reference, not a rule) |
| Heading: Inference Flow | 418-425 | Inference flow, six steps. | `docs/development/sycl-fork-guide.md`, "3. Repository map" | no (reference, not a rule) |
| Heading: Weight Caching (GPU Backends) | 426-431 | How GPU backends cache weights. | `docs/development/sycl-fork-guide.md`, "3. Repository map" | no (reference, not a rule) |
| Heading: SYCL Memory Ownership | 432-480 | The unified cache is the only allocator. The two ownership surfaces. Allocation provenance. `mem_handle` lifetime. No forced eviction. | `.claude/rules/sycl-backend.md`, "SYCL Memory Ownership"; `CLAUDE.md`, "SYCL Memory Ownership" | yes (short form; full text loads with backend files) |
| "At cleanup" is load-bearing, and was being read too broadly. | 481-538 | A live lease is a defect only when its owner is gone. History of the abort regression. `weight_entry_reclaimable()`. Leak diagnosis. Raw pointers are not ownership. Leases are ref-counted. | `.claude/rules/sycl-backend.md`, "SYCL Memory Ownership"; `CLAUDE.md`, "SYCL Memory Ownership" | yes (short form; full text loads with backend files) |
| Heading: ggml Conventions | 539-540 | Section heading. | `.claude/rules/cpp-conventions.md`, "C and C++ conventions"; `CLAUDE.md`, "ggml Conventions" | yes (heading kept) |
| Heading: Matrix Multiplication | 541-546 | `ggml_mul_mat` computes `C^T = A * B^T`. | `.claude/rules/cpp-conventions.md`, "C and C++ conventions"; `CLAUDE.md`, "ggml Conventions" | yes |
| Heading: Tensor Storage | 547-550 | Tensors are row-major. Meaning of dimensions 0, 1, 2. | `.claude/rules/cpp-conventions.md`, "C and C++ conventions"; `CLAUDE.md`, "ggml Conventions" | yes |
| Heading: Naming Patterns | 551-561 | Naming patterns with examples. | `.claude/rules/cpp-conventions.md`, "C and C++ conventions"; `CLAUDE.md`, "ggml Conventions" | yes (short form; examples load with C and C++ files) |
| Heading: Enum Values | 562-571 | Enum value naming with an example. | `.claude/rules/cpp-conventions.md`, "C and C++ conventions"; `CLAUDE.md`, "ggml Conventions" | yes (short form) |
| Heading: Struct Declarations | 572-579 | Struct declaration style with an example. | `.claude/rules/cpp-conventions.md`, "C and C++ conventions"; `CLAUDE.md`, "ggml Conventions" | yes (short form) |
| Heading: Coding Guidelines | 580-589 | Coding guidelines: dependencies, STL, alignment, formatting, API types, file naming. | `.claude/rules/cpp-conventions.md`, "C and C++ conventions"; `CLAUDE.md`, "ggml Conventions" | yes (short form; the public API types line and the cross-platform line load with C and C++ files only) |
| Heading: Hard-Won Rules (Workflow, Safety, Architecture) | 590-593 | Section heading and a one-line introduction. | `docs/development/sycl-fork-guide.md`, "4. Communication and workflow"; `CLAUDE.md`, "Hard-Won Rules (Workflow, Safety, Architecture)" | yes (heading kept) |
| Heading: Communication & Workflow | 594-594 | Sub-heading. | `docs/development/sycl-fork-guide.md`, "4. Communication and workflow" | no (sub-heading dropped) |
| The user reads Discord, not the terminal. | 595-595 | The user reads Discord, not the terminal. Send questions and status through the Discord reply tool. | `docs/development/sycl-fork-guide.md`, "4. Communication and workflow"; `CLAUDE.md`, "top of file" | yes, reworded: it now applies when the harness supplies a Discord channel id (flagged for the maintainer) |
| Work on the active feature branch | 596-596 | Work on the active feature branch. Bound a diff review by BASE_SHA/HEAD_SHA. | `docs/development/sycl-fork-guide.md`, "4. Communication and workflow"; `CLAUDE.md`, "Hard-Won Rules" | yes |
| Worktrees are allowed and are the right tool for build-heavy parallel work | 597-597 | Worktrees are allowed for build-heavy parallel work. Why the earlier advice was overstated. | `docs/development/sycl-fork-guide.md`, "4. Communication and workflow"; `CLAUDE.md`, "Hard-Won Rules" | yes (short form) |
| Since llama.cpp-7mqd, a first build at a new path hits the cache for host compiles. | 598-601 | Measurements of first-build cost at a new path since llama.cpp-7mqd. How to size a wave of agents. When to use a worktree. | `docs/development/sycl-fork-guide.md`, "4. Communication and workflow"; `CLAUDE.md`, "Hard-Won Rules" | yes (the sizing rule and the merge rule; measurements are in the doc) |
| Lock scope follows the build directory, not the act of building. | 602-602 | Lock scope follows the build directory. `GPU.lock` is always global. | `docs/development/sycl-fork-guide.md`, "4. Communication and workflow"; `CLAUDE.md`, "Hard-Won Rules" | yes |
| Delete an evidence worktree only after the finding it supports has been reviewed. | 603-603 | Delete an evidence worktree only after its finding has been reviewed. | `docs/development/sycl-fork-guide.md`, "4. Communication and workflow"; `CLAUDE.md`, "Hard-Won Rules" | yes |
| `git merge-base HEAD master` is usually the WRONG before-point for a measurement. | 604-604 | `git merge-base HEAD master` is usually the wrong before-point for a measurement. | `docs/development/sycl-fork-guide.md`, "4. Communication and workflow"; `CLAUDE.md`, "Hard-Won Rules" | yes |
| Fix-forward, never revert. | 605-605 | Fix forward, never revert. | `docs/development/sycl-fork-guide.md`, "4. Communication and workflow"; `CLAUDE.md`, "Hard-Won Rules" | yes |
| Verify correctness before claiming any perf win. | 606-607 | Verify correctness before claiming a performance win. | `docs/development/sycl-fork-guide.md`, "4. Communication and workflow"; `CLAUDE.md`, "Verification Commands & Correctness Gates" | yes |
| Heading: Safety (these have hung or exhausted memory on this host) | 608-608 | Sub-heading. | nowhere (see the kept column) | no (sub-heading dropped) |
| THE NEVER-LOOP RULE IS ABOUT A PROPERTY, NOT A LIST OF BINARIES. | 609-630 | The never-loop rule is about a property: any test that loads models onto a GPU. The 2026-08-01 near miss. What to ask before writing "run it N times". | `docs/backend/sycl-host-safety.md`, "3. Never repeat a model-loading test"; `CLAUDE.md`, "Running Tests" | yes |
| The rule is not about accumulation, and that misreading is why it gets broken. | 631-638 | A single `test-llama-archs` run peaks at 195 to 206 GB, so a second overlapping run does not fit. | `docs/backend/sycl-host-safety.md`, "3. Never repeat a model-loading test" | no (reasoning for the two rules above it) |
| `GPU.lock` (or any device lock) serialises ACCESS, not MEMORY RECOVERY. | 639-646 | `GPU.lock` serialises access, not memory recovery. Settle thresholds before touching the device. | `docs/backend/sycl-host-safety.md`, "3. Never repeat a model-loading test"; `CLAUDE.md`, "Running Tests" | yes (both thresholds and the 10 minute limit) |
| THE CAUSE IS NOW KNOWN, AND THERE IS A ONE-LINE MITIGATION | 647-675 | The cause is the iGPU. Pinning the selector holds `Shmem` flat. Measurements for three binaries. | `docs/backend/sycl-host-safety.md`, "3. Never repeat a model-loading test"; `CLAUDE.md`, "Running Tests" | yes (pin the selector, exact command); measurements are in the doc |
| The `test-llama-archs` half of that sentence leans on a table that is CORRUPTED | 676-695 | The `test-llama-archs` results table is corrupted by interleaved logs. Use the exit status and the `roundtrip mismatch` count. | `docs/backend/sycl-host-safety.md`, "3. Never repeat a model-loading test"; `CLAUDE.md`, "Running Tests" | yes |
| This does NOT retire the never-loop rule; it gives it one documented escape. | 696-714 | Pinning does not retire the never-loop rule. Known members of the family. Prefer single invocations. | `docs/backend/sycl-host-safety.md`, "3. Never repeat a model-loading test"; `CLAUDE.md`, "Running Tests" | yes (unpinned is the default; the member notes) |
| Always `timeout 60` GPT-OSS 20B test runs. | 715-715 | Always `timeout 60` GPT-OSS 20B test runs. How to tell the two hang signatures apart. | `docs/backend/sycl-host-safety.md`, "7. Timeouts, crashes, the kernel log and competing load"; `CLAUDE.md`, "Running Tests" | yes (the rule); the signatures are in the doc |
| Benchmark numbers are invalid after any crash or forced stop on that card | 716-716 | Benchmark numbers are invalid after a crash or forced stop on that card. | `docs/backend/sycl-host-safety.md`, "7. Timeouts, crashes, the kernel log and competing load"; `CLAUDE.md`, "Running Tests" | yes |
| `dmesg` is privilege-denied for this user | 717-721 | `dmesg` is denied. The `journalctl -k` command to use. | `docs/backend/sycl-host-safety.md`, "7. Timeouts, crashes, the kernel log and competing load"; `CLAUDE.md`, "Running Tests" | yes (exact command) |
| Check for competing load before any throughput measurement. | 722-722 | Check for competing load before a throughput measurement. | `docs/backend/sycl-host-safety.md`, "7. Timeouts, crashes, the kernel log and competing load"; `CLAUDE.md`, "Hard-Won Rules" | yes (exact command) |
| Owner ruling 2026-08-07: this host's ambient load (~60 - Emby, the ws2022ci qemu VM, Fr... | 723-724 | Ruling 2026-08-07: the host load is permanent. Never wait for a quiet host. | `docs/backend/sycl-host-safety.md`, "7. Timeouts, crashes, the kernel log and competing load"; `CLAUDE.md`, "Hard-Won Rules" | yes |
| Heading: Tooling | 725-725 | Sub-heading. | nowhere (see the kept column) | no (sub-heading dropped) |
| codescout's index is BLIND inside `ggml/src/ggml-sycl/ggml-sycl.cpp` | 726-726 | codescout's index misses matches inside `ggml-sycl.cpp`. Verify with `cat ... \| grep -n`. | `docs/development/sycl-fork-guide.md`, "5. Tooling"; `CLAUDE.md`, "Hard-Won Rules" | yes (exact command) |
| `/tmp` is tmpfs and does not survive a reboot. | 727-728 | `/tmp` does not survive a reboot. An empty artifact directory means not verified. | `docs/development/sycl-fork-guide.md`, "5. Tooling"; `CLAUDE.md`, "Hard-Won Rules" | yes |
| Heading: Architecture | 729-729 | Sub-heading. | nowhere (see the kept column) | no (sub-heading dropped) |
| The unified cache owns all GPU/host memory | 730-730 | The unified cache owns all GPU and host memory. | `.claude/rules/sycl-backend.md`, "Architecture"; `CLAUDE.md`, "SYCL Memory Ownership" | yes |
| Use smart handles, never hold a raw `void*` from the cache. | 731-731 | Use smart handles. Never hold a raw `void*` from the cache. | `.claude/rules/sycl-backend.md`, "Architecture"; `CLAUDE.md`, "SYCL Memory Ownership" | yes |
| PLACEMENT DECIDES THE EXECUTOR (owner ruling 2026-08-16; full statement in | 732-749 | Placement decides the executor. Layout follows residency. | `.claude/rules/sycl-backend.md`, "Architecture"; `CLAUDE.md`, "SYCL Memory Ownership" | yes (with the 2026-10-19 amendment from `docs/backend/sycl-memory-design.md`) |
| Dispatch-side sharpening (owner ruling 2026-09-18): THE LOADED LAYOUT IS THE | 750-771 | The loaded layout is the answer. A missing kernel is a gap to close. | `.claude/rules/sycl-backend.md`, "Architecture"; `CLAUDE.md`, "SYCL Memory Ownership" | yes |
| The VRAM budget calc is correct by design for DISCRETE cards | 772-772 | The VRAM budget calculation is correct by design for discrete cards. | `docs/backend/sycl-host-safety.md`, "8. The integrated GPU and the VRAM budget" | no (it is in `.claude/rules/sycl-backend.md`, which loads with backend files) |
| It is catastrophically wrong for an INTEGRATED GPU, and that is the cause of this host'... | 773-775 | The budget is wrong for an integrated GPU, which is the cause of the OOM history. Measurements. | `docs/backend/sycl-host-safety.md`, "8. The integrated GPU and the VRAM budget" | no (history; the rule it supports, pin the selector, is kept) |
| Small-block dequant (Q4_0/Q8_0/Q4_K) belongs on standard SYCL, not ESIMD. | 776-779 | Small-block dequant belongs on standard SYCL, not ESIMD. | `.claude/rules/sycl-backend.md`, "Architecture"; `docs/development/sycl-fork-guide.md`, "5. Tooling" | no (it is in `.claude/rules/sycl-backend.md`, which loads with backend files) |
| Heading: Development Workflow (Machine-Specific) | 780-781 | Section heading. | nowhere (see the kept column) | no (heading dropped) |
| Heading: Model Locations | 782-794 | Model locations under `/models/`. | `docs/backend/sycl-verification-traps.md`, "1. Model locations"; `CLAUDE.md`, "Verification Commands & Correctness Gates" | yes (the directory and the default benchmark model); the list is in the doc |
| Heading: Verification Commands & Correctness Gates | 795-796 | Section heading. | `CLAUDE.md`, "Verification Commands & Correctness Gates" | yes (heading kept) |
| The gates below check TOKENS, not which backend produced them. | 797-816 | The gates check tokens, not the backend. The backend check commands. Recovery from a CPU fallback. | `docs/backend/sycl-verification-traps.md`, "2. The gates check tokens, not the backend"; `CLAUDE.md`, "Verification Commands & Correctness Gates" | yes (exact commands) |
| The same blindness in test form: a SYCL test binary run WITHOUT sourcing | 817-842 | A SYCL test binary run without `setvars.sh` prints `SKIP` and exits 0. Exit 77 means skipped. | `docs/backend/sycl-verification-traps.md`, "3. A SYCL test run without oneAPI prints SKIP and exits 0"; `CLAUDE.md`, "Verification Commands & Correctness Gates" | yes |
| Generalise it: WHAT THE REGISTRATION PROVIDES, DIRECT INVOCATION DOES NOT. | 843-861 | What the registration provides, a direct run does not. Staging the model fixture. | `docs/backend/sycl-verification-traps.md`, "4. What the test registration provides, a direct run does not"; `CLAUDE.md`, "Verification Commands & Correctness Gates" | yes (exact fixture command); the `awk` command is in the doc only |
| GREP THE STRING THE CODE PRINTS, NOT THE NAME THAT PRODUCES IT. | 862-886 | Search for the string the code prints. `GGML_ABORT` never reaches a log. Probe table. | `docs/backend/sycl-verification-traps.md`, "5. Search for the string the code prints, not the name that produces it"; `CLAUDE.md`, "Verification Commands & Correctness Gates" | yes (the rule and the working probe) |
| Do not "fix" it by lowercasing to `ggml_abort`. | 887-906 | Do not repair the probe by lowercasing to `ggml_abort`. Why `GGML_ASSERT` differs. The positive-control test. | `docs/backend/sycl-verification-traps.md`, "5. Search for the string the code prints, not the name that produces it"; `CLAUDE.md`, "Verification Commands & Correctness Gates" | yes (the positive-control sentence); the rest is in the doc |
| `test-llama-archs -a <arch>` had the same shape and it is the one that | 907-1010 | `test-llama-archs -a <arch>` exits 77 when it measured nothing. Then the gate command block: Mistral gate, GPT-OSS gate with its scoring and history, benchmark, `test-backend-ops`. | `docs/backend/sycl-verification-traps.md`, "6. `test-llama-archs -a <arch>` and the exit status 77"; `docs/backend/sycl-verification-traps.md`, "7. The gate command block as it stood"; `CLAUDE.md`, "Verification Commands & Correctness Gates, Running Tests" | yes (exit 77; every command, exact; the scoring rule) |
| Heading: `llama-bench` traps (both cost real time) | 1011-1018 | `llama-bench` trap 1: `-v` is mandatory for log output. | `docs/backend/sycl-verification-traps.md`, "8. `llama-bench` traps and log verbosity"; `CLAUDE.md`, "Verification Commands & Correctness Gates" | yes |
| There is a SECOND, broader mechanism, and this section described only the | 1019-1069 | `GGML_LOG_INFO` is dropped at default verbosity in every tool. `-p 0 -n 4` does no prompt processing. Trap 2: several `-m` values load one after another. | `docs/backend/sycl-verification-traps.md`, "8. `llama-bench` traps and log verbosity"; `CLAUDE.md`, "Verification Commands & Correctness Gates" | yes (short form) |
| Heading: GPT-OSS Prompt Template Rule | 1070-1083 | GPT-OSS Prompt Template Rule. | `docs/backend/sycl-verification-traps.md`, "9. GPT-OSS Prompt Template Rule (as it stood)"; `CLAUDE.md`, "Verification Commands & Correctness Gates" | yes (bold lead-in, not a heading) |
| Heading: Patched compute-runtime & P2P topology | 1084-1085 | Section heading. | `CLAUDE.md`, "SYCL Device Selection" | yes (bold lead-in, not a heading) |
| THE LOADED DRIVER IS 26.40, NOT 26.31 (verified 2026-11-13). | 1086-1107 | The loaded driver was 26.40. Upgrade dates. Commands to read the version live. | `docs/backend/compute-runtime.md`, "The moved text"; `CLAUDE.md`, "SYCL Device Selection" | yes (exact commands; corrected to 26.44, loaded since 2026-11-14) |
| STALE-PIN CORRECTION (verified 2026-08-30, llama.cpp-09um) | 1108-1158 | Stale-pin correction and the patched 26.22 history. No direct P2P between the two discrete cards. Topology and kernel evidence. | `docs/backend/compute-runtime.md`, "The moved text"; `CLAUDE.md`, "SYCL Device Selection" | yes (no P2P; keep peer-copy and shared-context paths disabled) |
| `ext_oneapi_enable_peer_access()` returns OK on hardware that has no P2P. | 1159-1183 | `ext_oneapi_enable_peer_access()` returns OK without P2P. `can_access_peer` is the honest query. `GGML_SYCL_MOE_MULTI_GPU` stays opt-in. | `docs/backend/compute-runtime.md`, "The moved text"; `CLAUDE.md`, "SYCL Device Selection" | yes (short form) |
| Heading: SYCL Device Selection | 1184-1205 | Selector syntax. The three devices. The selector to PCI to render node table. | `docs/backend/sycl-host-safety.md`, "9. Device selection: the device map and its history"; `CLAUDE.md`, "SYCL Device Selection" | yes (table kept) |
| The PCI addresses MOVED at the 2026-09-05 19:03 boot | 1206-1219 | The PCI addresses moved at the 2026-09-05 boot. Scripts must derive the card live through `scripts/sycl-gpu-sysfs.sh`. | `docs/backend/sycl-host-safety.md`, "9. Device selection: the device map and its history"; `CLAUDE.md`, "SYCL Device Selection" | yes |
| This table had `renderD128` and `renderD129` SWAPPED until 2026-07-30 | 1220-1229 | The table had two render nodes swapped until 2026-07-30, and what that made people conclude. | `docs/backend/sycl-host-safety.md`, "9. Device selection: the device map and its history" | no (history; the corrected table is kept) |
| Derive it live rather than trusting the table | 1230-1240 | Loop that derives the render node to PCI map live. | `docs/backend/sycl-host-safety.md`, "9. Device selection: the device map and its history"; `CLAUDE.md`, "SYCL Device Selection" | yes, as a pointer to the doc (the loop itself is in the doc only) |
| DRM `cardN` numbering and `renderDN` numbering are independent - do not infer | 1241-1247 | `cardN` and `renderDN` numbering are independent. `device=N` in logs is the index after filtering. | `docs/backend/sycl-host-safety.md`, "9. Device selection: the device map and its history"; `CLAUDE.md`, "SYCL Device Selection" | yes |
| Before trusting any B70 benchmark, check the reported free VRAM in the startup | 1248-1258 | Check free VRAM before trusting a B70 benchmark. Three example selector commands. | `docs/backend/sycl-host-safety.md`, "9. Device selection: the device map and its history"; `CLAUDE.md`, "SYCL Device Selection" | yes (the check and the three selectors; the example commands are in the doc) |
| Rules | 1259-1274 | Device selection belongs to oneAPI. `GGML_SYCL_VISIBLE_DEVICES` does not work. `common_params_parse()` stays metadata-only. Hang hazards: `sycl-ls`, old comparison binaries, fdinfo. | `docs/backend/sycl-host-safety.md`, "9. Device selection: the device map and its history"; `CLAUDE.md`, "SYCL Device Selection" | yes; the parser rule is in `.claude/rules/arg-parser.md` |
| Current-boot B70/B50 P2P topology warnings are diagnostic only. Frigate | 1275-1284 | P2P topology warnings are diagnostic only. Frigate jobs on `renderD128` use the iGPU, not the B70, but cost CPU. | `docs/backend/sycl-host-safety.md`, "9. Device selection: the device map and its history"; `CLAUDE.md`, "SYCL Device Selection" | yes (the Frigate sentence); the P2P sentence is in the doc only |
| Heading: Performance Expectations | 1285-1308 | Orientation throughput table. Re-measurement of 2026-07-25. B70 TG is the noisy axis. B580 figures are history. | `docs/backend/sycl-perf-baselines.md`, "Performance Expectations (as it stood)"; `CLAUDE.md`, "Regression Baselines (hard guardrails)" | yes (the figures, with the B70 GPT-OSS TG corrected to ~39-40 as the newer passage says) |
| `GGML_SYCL_FA_ONEDNN_ALLOW` does not exist, and the warning that used to | 1309-1346 | `GGML_SYCL_FA_ONEDNN_ALLOW` does not exist. oneDNN attention is on by default. Turning it off costs about 39% of PP512. | `docs/backend/sycl-perf-baselines.md`, "Performance Expectations (as it stood)"; `CLAUDE.md`, "SYCL Environment Variables" | yes (leave `GGML_SYCL_FA_ONEDNN` on); the history is in the doc |
| Heading: Regression Baselines (hard guardrails) | 1347-1359 | Do not accept lower numbers as new baselines. Gate against the baselines doc. The four gate rows. | `docs/backend/sycl-perf-baselines.md`, "Regression Baselines (as it stood)"; `CLAUDE.md`, "Regression Baselines (hard guardrails)" | yes |
| This line carried `~44 TG128` until 2026-09-17 and it was stale in exactly | 1360-1370 | The B70 GPT-OSS line carried `~44 TG128` until 2026-09-17. That figure is driver 26.27. | `docs/backend/sycl-perf-baselines.md`, "Regression Baselines (as it stood)"; `CLAUDE.md`, "Regression Baselines (hard guardrails)" | yes (listed as a dead figure) |
| A `≥1100 PP512, ~50+ TG128` B50 GPT-OSS guardrail appeared here until | 1371-1386 | The `≥1100 PP512` B50 guardrail was wrong. The B580 gate is superseded. Variables that stay opt-in. | `docs/backend/sycl-perf-baselines.md`, "Regression Baselines (as it stood)"; `CLAUDE.md`, "Regression Baselines (hard guardrails)" | yes |
| The stale guardrail above is also baked into the tracker | 1387-1400 | The stale guardrail is also in tracker acceptance criteria. Use `task_search` with `semantic=false`. | `docs/backend/sycl-perf-baselines.md`, "Regression Baselines (as it stood)"; `CLAUDE.md`, "Regression Baselines (hard guardrails), Hard-Won Rules" | yes (short form) |
| Heading: SYCL Environment Variables | 1401-1420 | Pointer to the env-var catalog. Table of load-bearing opt-outs. Common diagnostics. | `.claude/rules/sycl-backend.md`, "SYCL Environment Variables (the short table)"; `CLAUDE.md`, "SYCL Environment Variables" | yes (as a list; the table loads with backend files) |
| Heading: CI and Validation | 1421-1422 | Section heading. | nowhere (see the kept column) | no (heading dropped; the sub-section is kept) |
| Heading: Before Submitting PRs | 1423-1443 | The six steps before a pull request. | `CLAUDE.md`, "Before Submitting PRs" | yes (step 3 carries the exact sweep filter) |
| Heading: Triggering Heavy CI | 1444-1446 | Add `ggml-ci` to a commit message to trigger the extended CI workloads. | `CLAUDE.md`, "Before Submitting PRs" | yes |
| Heading: Documentation | 1447-1463 | Documentation index. | `docs/development/sycl-fork-guide.md`, "6. Documentation index as it stood"; `CLAUDE.md`, "Documentation" | yes (rewritten to list the new pages) |

## Text changed in the move

- Old text: `([[git-add-is-not-scoping]] is the same fact seen from the commit side)`. New text: `('git add' does not scope a commit to your files; that is the same fact seen from the commit side)`. Reason: the double-bracket link pointed at a note outside the repository.

## Lines not moved

These old lines are not copied word for word anywhere. They are headings, blank lines between
blocks, or text that `CLAUDE.md` restates.

| old lines | reason |
|---|---|
| 1-6 | title and preamble; rewritten as the ownership statement in CLAUDE.md |
| 46-48 | heading 'Running Tests'; kept as a heading in CLAUDE.md |
| 137-137 | blank |
| 216-216 | blank |
| 241-241 | blank |
| 273-273 | blank |
| 304-306 | heading 'Code Formatting'; kept as a heading in CLAUDE.md |
| 378-380 | heading 'Project Architecture'; kept as a pointer in CLAUDE.md |
| 431-433 | heading 'SYCL Memory Ownership'; kept as a heading in CLAUDE.md and in the rules file |
| 538-538 | blank |
| 589-591 | heading 'Hard-Won Rules'; kept as a heading in CLAUDE.md |
| 607-608 | heading 'Safety' |
| 724-725 | heading 'Tooling' |
| 728-729 | heading 'Architecture'; kept in the rules file |
| 777-777 | blank |
| 779-782 | headings 'Development Workflow', 'Model Locations' |
| 794-796 | heading 'Verification Commands & Correctness Gates'; kept in CLAUDE.md |
| 816-816 | blank |
| 842-842 | blank |
| 861-861 | blank |
| 906-906 | blank |
| 921-921 | blank |
| 1010-1012 | heading '`llama-bench` traps'; kept in CLAUDE.md and the traps doc |
| 1069-1071 | heading 'GPT-OSS Prompt Template Rule'; kept in CLAUDE.md |
| 1083-1085 | heading 'Patched compute-runtime & P2P topology'; kept in CLAUDE.md |
| 1183-1185 | heading 'SYCL Device Selection'; kept in CLAUDE.md |
| 1284-1286 | heading 'Performance Expectations' |
| 1346-1348 | heading 'Regression Baselines (hard guardrails)'; kept in CLAUDE.md |
| 1400-1402 | heading 'SYCL Environment Variables'; kept in CLAUDE.md |
| 1420-1420 | blank |
| 1421-1446 | 'CI and Validation'; kept in CLAUDE.md |
| 1447-1448 | heading 'Documentation'; kept in CLAUDE.md |

## Coverage check

The check takes every fenced command line, every backticked environment variable, script path, test
name and commit id from the old file, and looks for each one in the new set of files (the ten files
in the table above). For the two existing pages, only the added sections count. The last line is a
wider check: every distinct backticked span of the old file. Result on 2026-11-16:

- Fenced blocks in the old file: 31. Found whole in the new text: 31. Not found whole: 0.
- Distinct non-blank lines inside those blocks: 181. Found: 181. Missing: 0. Also present in `CLAUDE.md` itself: 40.
- Backticked environment variables and other upper-case identifiers: 65. Found: 65. Missing: 0. Also present in `CLAUDE.md` itself: 25.
- Backticked paths and file names: 107. Found: 107. Missing: 0. Also present in `CLAUDE.md` itself: 46.
- Test names (`test-...`): 25. Found: 25. Missing: 0. Also present in `CLAUDE.md` itself: 18.
- Commit ids: 15. Found: 15. Missing: 0. Also present in `CLAUDE.md` itself: 1.
- Tracker ids (`llama.cpp-...`): 19. Found: 19. Missing: 0. Also present in `CLAUDE.md` itself: 0.
- All distinct backticked spans (superset check): 530. Found: 530. Missing: 0. Also present in `CLAUDE.md` itself: 218.

