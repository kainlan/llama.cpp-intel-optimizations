# CLAUDE.md

You are the owner and long-term maintainer of this codebase, and its design is up to you.

This is a fork of llama.cpp with an Intel SYCL backend, developed and measured on one machine with two Intel Arc
discrete cards and an integrated GPU (iGPU). You decide the design, write the code and keep the result
maintainable: fix root causes, keep tests and docs true, and record follow-up work in the codescout task tracker
(the `task_*` tools). A dated "ruling" or "owner ruling" records a decision the human made earlier. It is the
current design. Follow it. If you conclude it should change, change it on purpose, say why, and update every
place that states it.

Get the human's explicit yes before:
- a destructive git or file action on work that is not yours (other sessions share the main checkout);
- moving or downloading model files (`/models/`);
- changing system settings (drivers, packages, kernel, GPU or service configuration);
- anything outward-facing (a push, a pull request, a post, a message sent for the human).

When the harness supplies a Discord channel id, the human reads Discord, not the terminal: send every question,
confirmation, decision prompt and status update through the Discord reply tool. Otherwise write to the
conversation. This file holds what every session needs. History and reasoning are in the pages under
Documentation. Rules in `.claude/rules/` load when you open the files they name.

## Build Commands (Intel SYCL)
Source oneAPI before running any binary (the build script sources it for builds):
`source /opt/intel/oneapi/setvars.sh --force`
```bash
./scripts/sycl-build.sh                           # incremental build
./scripts/sycl-build.sh llama-bench               # build a single target
./scripts/sycl-build.sh -r                        # force CMake reconfigure
./scripts/sycl-build.sh -c                        # clean (rm build/) and rebuild
```
Output goes to `build/bin/`. A build takes ~10 minutes with ccache, ~25 without. The script always uses Ninja;
if `build/` came from another generator, run it with `-c`. It sets `-DGGML_SYCL_F16=ON` (16-bit float in dequant
and attention); declare that OFF only for precision-sensitive models. Build detail and the Project Architecture
map: `docs/development/sycl-fork-guide.md`.

### Running Tests
A test that loads a model onto a GPU allocates GPU buffers whose host backing is in no process's own figures.
An iGPU or `test-backend-ops` run shows it under `Shmem`; a discrete card's host-pinned memory shows only in the
`/proc/meminfo` remainder. "GPU host backing" below means `Shmem` plus that remainder. Such tests have OOM-killed
this 255 GB host several times.
History and measurements: `docs/backend/sycl-host-safety.md`.
- GPU and model-loading work runs only in the lead session, one run at a time: every `llama-bench`, `llama-cli`,
  `llama-completion`, `test-llama-archs`, `test-thread-safety`, `test-backend-ops` and every `ctest` sweep.
  Subagents read, analyse, edit, build and run non-GPU tests, and report what needs verifying.
- Never repeat a test that loads a model onto a GPU. Before you write "run it N times" into a script, gate or
  acceptance criterion, ask whether the binary loads a model onto a GPU. If it does, run it once.
- Pin the selector to the discrete cards for any local run that does not need the iGPU. Unpinned is the default
  and the dangerous form: `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ctest --test-dir build -R '<name>' --output-on-failure`
- Sample GPU host backing before each GPU run and about 5 s after it, and abort if it climbs past ~100 GiB:
  `source scripts/sycl-host-backing.sh; sycl_gpu_host_backing_gib`. Read it, not `Shmem` alone (flat for a
  discrete card) and not `free`. A reading taken in the same command as an exiting process is a false alarm:
  `sleep 5; source scripts/sycl-host-backing.sh; sycl_gpu_host_backing_gib`.
- `GPU.lock` serialises access, not memory recovery. After you take it and before you touch the device, wait for
  GPU host backing `< 30 GiB`. If that takes more than ~10 minutes, release it and escalate. Guard a run on that
  figure only: `MemAvailable` sits near 55 GB on purpose while a 122 GiB model is held in host memory (ruling
  2026-11-12, amended 2026-11-23: the instrument is `Shmem` plus the remainder, still no `MemAvailable` floor).
- Run every GPT-OSS test under `timeout`, so a hang cannot hold the card. `timeout 60` covers a short 20B run;
  a bench or a default-context run needs more (`scripts/sycl-canonical-gates.sh` uses 900).
- `dmesg` is denied for this user. Read the kernel log with
  `journalctl -k --since "1 hour ago" --no-pager | grep -iE 'GT reset|guc_id|GPU hang|xe.*reset'`
- After any OOM, crash or forced stop, benchmark numbers on that card are invalid until this is clean:
  `journalctl -k --since "1 hour ago" --no-pager | grep -iE 'GT reset|guc_id|CAT error'`
```bash
source /opt/intel/oneapi/setvars.sh --force
# Form 1 (preferred): run only what your change gates.
ctest --test-dir build -R <name-or-regex> --output-on-failure
# Form 2: full suite, rarely. Check `uptime` first. Run the guard first: it must print "safety net intact".
# If it names a test, do not run the sweep. `-j 1` is not negotiable: `-j` multiplies memory.
bash scripts/check-ctest-safety-net.sh --build-dir build
ctest --test-dir build --output-on-failure -j 1 \
      -LE 'residency|mem-handle|cache|^model$|^fatal-exit-device$' -E '^(test-backend-ops|test-generate-.*|test-llama-context.*|test-fp8-.*|test-speculative-rejection-sampling|test-thread-safety|test-sycl-model-lifecycle-hooks|test-state-restore-fragmented|test-eval-callback)$'
# Form 3: the excluded family, serially, with monitoring. Manually only, never in a subagent or background task.
ctest --test-dir build -N -L 'cache|mem-handle'      # must list >= 1 test (`-L residency` selects nothing)
ctest --test-dir build -L 'cache|mem-handle' --output-on-failure -j 1
# Run a single test by name
ctest --test-dir build -R <test-name> -V
# Not ctests (`ctest -R` on either selects nothing and "passes"). Run as binaries, pinned, one at a time.
ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/test-llama-archs -a llama; echo "rc=$?"
ONEAPI_DEVICE_SELECTOR=level_zero:0   ./build/bin/test-backend-ops
```
- `test-backend-ops`: by hand, with monitoring, never in a subagent or background task.
- `test-llama-archs`: `rc=0` is a pass, 77 means it measured nothing, `rc=1` is ambiguous. Do not search its
  table for `FAIL` (log lines corrupt the table); use `grep -c 'roundtrip not bit-exact' <log>`.
- `test-unified-cache-bugs` peaks at ~8.5 GB RSS: serial only. Pure-Python gates (`test-sycl-gap-causes`,
  `test-sycl-timeline-gap-class-conservation`, `test-jinja-py`) allocate nothing and are safe at any parallelism.
- `main` is the default label and means "nobody classified this", not "safe". A new model-loading test must
  carry `model`, `cache` or `mem-handle`. When the guard names a new test, change this file and the script's
  `SWEEP_E_DEFAULT` together. More: `.claude/rules/ctest-registration.md`.
- Form 2 also leaves out the label `fatal-exit-device`: the four device modes of `test-sycl-fatal-exit`, each of
  which ends a process through the backend's fast exit on a healthy card. Run them by label, alone, with the
  selector the registration pins: `ctest --test-dir build -L '^fatal-exit-device$' --output-on-failure -j 1`.
  Why: `docs/backend/sycl-host-safety.md`, section 1.

### Code Formatting
`git clang-format` does not exist here. Use `git-clang-format-19` (or `git clang-format-19`) and `clang-format-19`.
- Alone in the checkout: `git-clang-format-19` formats staged changes only. `git-clang-format-19 --diff --staged`
  checks without writing ("clang-format did not modify any files" = clean).
- Other agents live: `clang-format-19 --dry-run -Werror <your file>`. The staged form works on the whole shared
  index and would reformat their files.
- Never `clang-format-19 -i <file>`: it rewrites ~180 lines of unrelated drift. If you ran it, revert the file to
  HEAD and redo the edit by hand. Leave drift you did not introduce for its own commit.

## Verification Commands & Correctness Gates
Verify correctness before you claim any performance win, and before you commit a change to kernel dispatch,
weight staging, graph replay or allocation routing. `llama-bench` measures speed only. The gates check tokens,
not the backend, so a CPU-only build passes them. Single-digit tok/s on Mistral Q4_0 is a CPU fallback: delete
only `build/CMakeCache.txt` and rebuild.
```bash
source /opt/intel/oneapi/setvars.sh --force
# Backend check. Run it first, and again after anything touches build/CMakeCache.txt.
grep -E '^GGML_SYCL:' build/CMakeCache.txt                        # want BOOL=ON
ldd build/bin/llama-completion | grep -cE 'libggml-sycl|libsycl'  # want >= 2
# Mistral completion gate. Output must start "1, 2, 3, 4, 5, 6, 7, 8, 9, 10" and it stops at 10.
# Any other output (###..., repetition, <unk>) = broken path; fix before commit.
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-completion \
  -m /models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
# GPT-OSS B50 chat correctness gate. The answer is a line of its own: 1, 2, 3, 4, 5
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-cli \
  -m /models/gpt-oss-20b-mxfp4.gguf \
  -st --simple-io --no-display-prompt \
  --chat-template-kwargs '{"reasoning_effort":"medium"}' \
  --reasoning-format none --reasoning-budget 0 \
  -p 'Count from 1 to 5. Answer with only: 1, 2, 3, 4, 5' \
  -n 48 --seed 42 --temp 0
cat <log> | grep -cx '1, 2, 3, 4, 5'   # 1 = pass (the ANSWER line alone)
# Benchmark prompt processing (PP) and token generation (TG)
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
```
- Score the GPT-OSS gate with `grep -cx`, not `grep -c`: the echoed prompt ends with the same digits. The gate
  carries no `-c`. Never shrink the context to make it fit; the cache places the KV. Keep a `-c 4096` arm as
  the all-VRAM control when you A/B. No gate or script passes `-ngl`: the unified cache places every weight,
  and a layer-count flag would be a second source for that fact (ruling 2026-10-17).
- **GPT-OSS Prompt Template Rule.** Use `llama-cli` so the model's embedded chat template applies. Do not pass
  `-cnv` (invalid on `llama-cli`) or `--chat-template gpt-oss`, and do not hand-render a Harmony prompt. Always
  pin `reasoning_effort=medium` with `--chat-template-kwargs`. Rationale: `docs/backend/gpt-oss-testing.md`.
- A SYCL test binary run directly without `setvars.sh` prints `SKIP` and exits 0. That is not a pass; exit 77
  means skipped. A direct run also lacks the model fixture: stage it with
  `ctest --test-dir build -R '^test-download-model$'` (no GPU work, safe for any agent).
- Search a log for the string the code prints, not the name that produces it. `GGML_ABORT` never reaches a log;
  search for the message or `ggml-sycl\.cpp:[0-9]+:`. A probe with no positive control proves nothing.
- **`llama-bench` traps.** It discards every log line unless you pass `-v`, so confirm a "before" capture is
  non-empty. `GGML_LOG_INFO` is dropped at default verbosity in every tool: emit a must-see diagnostic at WARN.
  `-p 0 -n 4` does no prompt processing. Several `-m` values load models one after another, never together.
- Models are in `/models/`; the default benchmark model is `mistral-7b-v0.1.Q4_0.gguf`. Detail for this
  section: `docs/backend/sycl-verification-traps.md`. Automated runner: `scripts/sycl-canonical-gates.sh`.

## SYCL Device Selection
`ONEAPI_DEVICE_SELECTOR` takes `backend:devices`: `level_zero:0`, `level_zero:0,1`, `level_zero:gpu`. The
`level_zero:gpu:0` strings some tools print are display IDs, not selector values.

| selector | card | PCI (since 2026-09-05) | render node | DRM card |
|----------|------|-----|-------------|----------|
| `level_zero:0` | Arc Pro B70 (Battlemage G31, 256 CU, ~32.6 GB) | `0000:04:00.0` | `renderD129` | `card0` |
| `level_zero:1` | Arc Pro B50 (Battlemage G21, 128 CU, ~16 GB) | `0000:09:00.0` | `renderD130` | `card2` |
| device 2 | Arrow Lake-S iGPU | `0000:00:02.0` | `renderD128` | `card1` |

- The map can move across boots, and `cardN` and `renderDN` numbering are independent. Derive it live (the loop
  is in `docs/backend/sycl-host-safety.md`, section 9). Scripts use `scripts/sycl-gpu-sysfs.sh`, never a fixed
  PCI address. `device=N` in SYCL logs is the index after filtering: key off the selector, never the log id.
- `level_zero:1` is single-GPU B50 validation, `level_zero:0` the B70, `level_zero:0,1` both (host-bounce paths).
- Device selection belongs to oneAPI. Do not add code that parses, rewrites or refuses `ONEAPI_DEVICE_SELECTOR`.
  `GGML_SYCL_VISIBLE_DEVICES` does not work. Parser rule: `.claude/rules/arg-parser.md`.
- Never use `sycl-ls` as a health check: it has hung this host. Do not run old comparison binaries (for example
  the `60a8c042` build) with `--help`, `--version` or `lsof /dev/dri/*` on the discrete render nodes. Compare
  builds with the gate command after a fresh reboot. No DRM fdinfo checks while a SYCL process is hung.
- Before you trust a B70 benchmark, check free VRAM in the startup log: ~32.6 GB when the card is free. Frigate
  `ffmpeg` jobs on `renderD128` use the iGPU, not the B70, but cost CPU (~250 %). Any B580 figure is history.
- **Patched compute-runtime & P2P topology.** Read the driver version live before you attribute a throughput
  change to the tree: `readlink /usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1` and
  `dpkg -l libze-intel-gpu1 | tail -1`. Loaded on 2026-11-16: 26.44 (since 2026-11-14). The baseline figures
  were measured on 26.27 and 26.31 and have not been re-validated since. There is no direct peer-to-peer (P2P)
  path between the two discrete cards: keep direct peer-copy and shared-context paths disabled and
  `GGML_SYCL_MOE_MULTI_GPU` opt-in.
  `can_access_peer` is the honest query. Detail: `docs/backend/compute-runtime.md`.

## SYCL Memory Ownership
The unified cache is the memory allocator of the SYCL backend and is not optional. Read
`docs/backend/sycl-memory-design.md` and `docs/design/sycl-canonical-memory-architecture.md` before you touch
allocation, dispatch or eviction code. Full short form: `.claude/rules/sycl-backend.md`.
- Allocate only through the unified-cache APIs (`unified_alloc`, `unified_allocate`, cache materialization
  helpers, wrappers that return `mem_handle`). No direct `sycl::malloc_device`, `sycl::malloc_host`,
  `sycl::free`, raw TLSF allocation or side caches outside the unified-cache implementation.
- Ownership has two surfaces: a `mem_handle`, or an `alloc_owner` from `unified_allocate_owner()` (wrap it with
  `mem_handle::from_owned_alloc()`). A raw pointer is never one: it is a transient view for immediate use. Do
  not store it or use it as a cache key.
- Hold the `mem_handle` until the CPU thread, SYCL event, command graph or pointer table is done with the
  allocation. Its destructor is the sole release point.
- Never force eviction, reaping or a zone reset to reclaim memory that has a live handle. Evict only entries
  whose in-use count is zero. A live handle at cleanup whose owner is gone is a leak: fix the missing release.
  Another live model's lease is correct, not leaked. Weight reclaim goes through `weight_entry_reclaimable()`.
- Placement decides the executor (ruling 2026-08-16): an op runs where its data already is, device-resident on
  that GPU and host-pinned on the CPU. No GPU reads of host memory. No weight streaming, except that prompt
  processing may stage host-pinned experts to VRAM when the cost model says so (ruling 2026-10-19).
- Layout follows residency, and the loaded layout is the answer (rulings 2026-08-16 and 2026-09-18): dispatch
  picks the kernel from the layout the operand is in. A missing kernel for that layout is a gap to close by
  writing the kernel, not by re-routing, converting, refusing or widening a policy.

## ggml Conventions
- `C = ggml_mul_mat(ctx, A, B)` computes `C^T = A * B^T`, that is `C = B * A^T`. Tensors are row-major:
  dimension 0 = columns, 1 = rows, 2 = matrices.
- Names are `snake_case` with the longest common prefix first (`number_small`, `number_big`), in the form
  `<class>_<method>` where the method is `<action>_<noun>`. Enum values are UPPER_CASE with the enum name as
  prefix. Write `struct foo {}`, not a typedef. 4 spaces, brackets on the same line, `void * ptr`, `int & a`,
  vertical alignment. Few dependencies, simple STL. File names: C/C++ lowercase with dashes, Python lowercase
  with underscores. Full text and examples: `.claude/rules/cpp-conventions.md` and `CONTRIBUTING.md`.

## Hard-Won Rules (Workflow, Safety, Architecture)
- Work on the active feature branch (`git branch --show-current`). Bound a diff review by BASE_SHA/HEAD_SHA.
- Fix forward, never revert: no `git revert` or `git checkout --` to undo progress.
- Use a worktree for a track that builds repeatedly, and stay in the shared checkout for small work. Size a wave
  by its concurrent builds: three cold builds at once already slow everything. A merge into the active branch
  happens in the main checkout. Delete an evidence worktree only after its finding has been reviewed.
- Locks follow the build directory: building in the shared checkout takes `/Apps/llama.cpp/BUILD.lock`, your own
  worktree takes none. `GPU.lock` is always global.
- Before any throughput measurement: `uptime` plus `pgrep -af 'codescout|ninja|icpx|ffmpeg'`. The load (~60) is
  permanent (ruling 2026-08-07): never wait for a quiet host. Verdicts come from interleaved paired A/Bs.
  Absolute numbers taken under load must not become baselines. `git merge-base HEAD master` is usually the
  wrong before-point: use the commit just before your own first commit.
- `/tmp` does not survive a reboot. Results a later step needs go into a committed artifact. An empty or missing
  artifact directory means "not verified".
- codescout's index misses matches inside `ggml/src/ggml-sycl/ggml-sycl.cpp`. Verify there with
  `cat ggml/src/ggml-sycl/ggml-sycl.cpp | grep -n '<pattern>'`.
- Live debugging state lives in the task tracker (`task_ready`, `task_list`; `task_search` with
  `semantic=false`), not in this file.

### Regression Baselines (hard guardrails)
Gate against `docs/backend/sycl-perf-baselines.md`, never against a remembered number or a task's acceptance
criteria. Do not accept lower post-merge or post-debug numbers as new baselines. Orientation figures at
`-p 512 -n 128`, FA on, count gate passing, as PP512 / TG128:
- Arc Pro B70 (`level_zero:0`): GPT-OSS 20B MXFP4 ~1415 / ~39-40, Mistral 7B Q4_0 ~2495 / ~108.
- Arc Pro B50 (`level_zero:1`): GPT-OSS 20B MXFP4 ~894 / ~32, Mistral 7B Q4_0 ~1188 / ~47.
- B70 TG is noisy: ignore differences below ~10% between single runs. A B50 move of a few percent is real.
- Dead figures, wrong wherever quoted: B50 GPT-OSS `≥1100 PP512, ~50+ TG128`; B70 GPT-OSS `~44 TG128` (that was
  driver 26.27); every B580 figure.
- Keep opt-in until same-build B50 GPT-OSS and B70 Mistral gates pass on a clean boot:
  `GGML_SYCL_MOE_BLOCK_GRAPHLETS`, `GGML_SYCL_XMX_MOE_PP` / `GGML_SYCL_XMX_MOE_ALLOW_UNSAFE_PP`,
  `GGML_SYCL_PP_PIPELINE`.

### SYCL Environment Variables
Catalog: `docs/backend/sycl-env-vars.md`; short table: `.claude/rules/sycl-backend.md`. Opt-outs (ON by default):
`GGML_SYCL_UNIFIED_SOA=0`, `GGML_SYCL_TG_FAST=0`, `GGML_SYCL_ONEDNN_PP=0`, `GGML_SYCL_FA_ONEDNN=0` (leave it ON
unless you bisect attention; off costs ~39% of Mistral PP512). Opt-ins: `GGML_SYCL_DISABLE_GRAPH=1`,
`GGML_SYCL_UNIFIED_FORCE_LEGACY=1`. Diagnostics: `GGML_SYCL_DEBUG=1`, `GGML_SYCL_NAN_CHECK=1`,
`GGML_SYCL_SAFE_MODE=1`, `GGML_SYCL_OP_TIMEOUT_MS=<N>` (the last two can stall a card).

## Before Submitting PRs
1. Format: see Code Formatting. Never `clang-format-19 -i <file>`.
2. Build: `./scripts/sycl-build.sh`
3. Test: prefer `-R <what your change gates>`. Never a bare `ctest --test-dir build --output-on-failure`: it runs
   every model-loading test on every device, the iGPU included. For a full sweep, run
   `bash scripts/check-ctest-safety-net.sh --build-dir build` first, then form 2 verbatim. Do not raise `-j`:
   `-j 1 -LE 'residency|mem-handle|cache|^model$|^fatal-exit-device$' -E '^(test-backend-ops|test-generate-.*|test-llama-context.*|test-fp8-.*|test-speculative-rejection-sampling|test-thread-safety|test-sycl-model-lifecycle-hooks|test-state-restore-fragmented|test-eval-callback)$'`
4. For ggml changes, run `test-backend-ops` on multiple backends, by hand only. It is a binary, not a ctest
   (`ctest -R test-backend-ops` selects nothing): `ONEAPI_DEVICE_SELECTOR=level_zero:N ./build/bin/test-backend-ops`
5. Correctness: run the completion gates above. Tokens must be right, not just fast.
6. Performance: `llama-bench` and `llama-perplexity` should not regress.

Add `ggml-ci` to a commit message to trigger the extended CI workloads.

## Documentation
- `docs/backend/sycl-host-safety.md`: OOM and hang history, the reasoning behind the test and device rules
- `docs/backend/sycl-verification-traps.md`: gate detail, false passes, `llama-bench` and logging traps
- `docs/development/sycl-fork-guide.md`: build detail, repository map, formatting, worktrees, tooling
- `docs/backend/sycl-perf-baselines.md`: the numeric gate; prefer it over any figure in this file
- `docs/backend/sycl-memory-design.md`, `docs/design/sycl-canonical-memory-architecture.md`: memory design and
  the enforceable contract
- `docs/backend/sycl-env-vars.md`, `docs/backend/gpt-oss-testing.md`, `docs/backend/compute-runtime.md`,
  `docs/backend/SYCL.md`, `docs/build.md`, `docs/development/HOWTO-add-model.md`, `CONTRIBUTING.md`
- `.claude/rules/`: `sycl-backend.md`, `cpp-conventions.md`, `ctest-registration.md`, `arg-parser.md`
- `AGENTS.md`: upstream contribution policy, GPU address-space notes, profiling, session completion
- `.llm-wiki/`: shared knowledge vault. Conventions: `.claude/skills/llm-wiki/SKILL.md`
- `docs/development/claude-md-restructure-map.md`: where each part of the old version of this file went
