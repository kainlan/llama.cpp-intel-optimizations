# Host safety on the development machine: rules and incident history

This page is the history and the reasoning behind the host-safety rules in `CLAUDE.md`. The rules
themselves are short and live there, because every session needs them. Read this page when you
want to know why a rule exists, what was measured, or what went wrong the last time it was broken.

The machine has 255 GB of RAM, 20 cores, two Intel Arc discrete cards (Arc Pro B70 and Arc Pro B50)
and an Arrow Lake-S integrated GPU (iGPU). "TTM shmem" is the kernel's shared-memory backing for GPU
buffer objects. It is counted under `Shmem` in `/proc/meminfo` and does not show up in a process's
own memory figures. Section 10 (2026-11-23) adds that this holds for the iGPU and for `test-backend-ops`
only: a discrete card's host-pinned memory is in no `/proc/meminfo` field. "OOM" means the kernel's out-of-memory killer.

The text under each heading below was moved out of `CLAUDE.md` on 2026-11-16 and is unchanged,
apart from the corrections that are marked as such. In the moved text, "this file" means
`CLAUDE.md` as it stood before that date, and "above" or "below" refer to its old layout.
Line numbers quoted in the moved text were right on the date given beside them and have drifted
since. Find the code by the quoted text, not by the number.

The rules, in one place (the authoritative wording is in `CLAUDE.md`):

1. A full `ctest` sweep runs with `-j 1` and the exact `-LE ... -E ...` filter, after
   `scripts/check-ctest-safety-net.sh` prints "safety net intact".
2. Pin `ONEAPI_DEVICE_SELECTOR` to the discrete cards for any run that does not need the iGPU.
3. Never repeat a test that loads a model onto a GPU.
4. GPU and model-loading work runs only in the lead session, one run at a time.
5. Sample GPU host backing (`Shmem` plus the `/proc/meminfo` remainder, section 10) before a GPU run and
   about 5 seconds after it.
6. Read the kernel log with `journalctl -k`, because `dmesg` is denied for this user.
7. Run every GPT-OSS test under `timeout`. 60 seconds covers a short 20B run; benches and default-context
   runs need more.
8. Never use `sycl-ls` as a health check.

## 1. The full-suite sweep and `-j`

Old location: `CLAUDE.md`, "Running Tests", first half. It holds the two out-of-memory events of
2026-07-25, the three sweep forms as they were documented, and the 2026-08-01 event that showed
`-j` multiplies memory.

One clause here was superseded. The form 2 comment says "never on a loaded machine". The ruling of
2026-08-07 in section 7 says this host's load is permanent and that work must not wait for a quiet
host. `CLAUDE.md` now keeps "check `uptime` first" and drops "never on a loaded machine".

⚠️ **`-j $(nproc)` on the full suite OOM-kills this host.** This is not a
theoretical hazard — it happened twice on 2026-07-25, each time taking down the
Claude Code CLI along with `dbus-broker` and `xdg-document-portal`:

```
oom-kill: constraint=CONSTRAINT_NONE ... global_oom, task=test-llama-arch
Out of memory: Killed process 468242 (test-llama-arch)
```

Cause: tests **#1–#19** are the SYCL mem-handle / MoE-residency family, they all
allocate GPU buffer objects whose TTM shmem backing grows the same way
`test-backend-ops` does (see Hard-Won Rules), and they sit at the *front* of the
list — so `-j 20` on this 20-core box starts all 19 at once. `test-llama-arch`
was a victim of the global OOM (anon-rss 636 kB), not the cause. The specific
binary that consumed the ~230 GB was not isolated, and isolating it is not worth
another OOM.

The command that used to be documented here was `-j $(nproc)`, which is exactly
the hazard. Prefer, in order:

```bash
source /opt/intel/oneapi/setvars.sh --force

# 1. BEST: run only what your change actually gates. Almost always sufficient.
ctest --test-dir build -R <name-or-regex> --output-on-failure

# Since llama.cpp-9goq the SYCL test targets in
# ggml/src/ggml-sycl/CMakeLists.txt follow BUILD_TESTING (LLAMA_BUILD_TESTS):
# a -DLLAMA_BUILD_TESTS=OFF build (the Dockerfile's) configures none of them,
# and three host-only tests that link ggml-sycl directly are also disabled
# placeholders under GGML_BACKEND_DL=ON.

# 2. Full suite. `-j 1` is NOT a typo and NOT negotiable -- see below.
#    Check `uptime` first; never on a loaded machine.
#    Run the guard first: it derives the model loaders from the registration
#    (fixture users, fixture generators, tests labelled `model`) and fails
#    naming any that this command still selects.
bash scripts/check-ctest-safety-net.sh --build-dir build
ctest --test-dir build --output-on-failure -j 1 \
      -LE 'residency|mem-handle|cache|^model$|^fatal-exit-device$' -E '^(test-backend-ops|test-generate-.*|test-llama-context.*|test-fp8-.*|test-speculative-rejection-sampling|test-thread-safety|test-sycl-model-lifecycle-hooks|test-state-restore-fragmented|test-eval-callback)$'

# 3. The excluded family, serially, with monitoring. Manually only -- never in
#    a subagent or background task. NOTE the label is 'cache|mem-handle', NOT
#    'residency': after the llama.cpp-vrko relabel (2026-08-09) zero tests
#    carry 'residency', so `-L residency` selects nothing and passes vacuously.
#    Verify the selection is non-empty before trusting the run:
#    `ctest --test-dir build -N -L 'cache|mem-handle'` must list >= 1 test.
ctest --test-dir build -L 'cache|mem-handle' --output-on-failure -j 1

# Run a single test by name
ctest --test-dir build -R <test-name> -V
```

The form 2 `-LE` also leaves out the label `fatal-exit-device`. It is on the four
device modes of `test-sycl-fatal-exit` (`-mmid`, `-op`, `-sync` and `-graph`).
Each of them ends a process through the backend's fast exit on a healthy card,
which can leave copy-engine work in flight. About 2 exits in 40 leave a
copy-engine CAT error in the kernel log, and four of them one second apart once
reset the whole GT. The four host modes stay in the sweep. Run the device modes
by their label and alone, and leave `ONEAPI_DEVICE_SELECTOR` to the registration.
It pins `level_zero:1`, and each parent waits 12 s after its child:

```bash
ctest --test-dir build -L '^fatal-exit-device$' --output-on-failure -j 1
```

The guard fails with `SWEEP RUNS FAST-EXIT DEVICE TEST` if form 2 still selects
a test that carries the label.

⚠️ **`-j N` IS A MEMORY MULTIPLIER, NOT A CPU THROTTLE. This file prescribed
`-j 4` as the *safe* form until 2026-08-01, and it caused a global OOM.**

The reasoning that produced `-j 4` was "20 cores, so 4 is gentle" — sizing the
flag against the *abundant* resource. The scarce resource is TTM shmem, and `-j`
multiplies that just as readily. The arithmetic was already recorded elsewhere in
this very file and simply never applied here: **a single `test-llama-archs` run
peaks at 195–206 GB of 255 GB.** Two model-loading tests concurrently do not
fit — there is no gradual approach, the first overlap is fatal.

Verified 2026-08-01, and this is the whole proof — both survive the denylist
carrying only the default `main` label, so `-j 4` starts them together:

```bash
ctest --test-dir build -N -LE 'residency|mem-handle|cache' \
  | grep -E ' (test-llama-archs|test-thread-safety)$'   # both print. Both load models.
```

The blast radius was not just the test run: the kernel killed `pipewire`,
`dbus-broker`, both user `systemd` instances, `(sd-pam)`, two `ssh-agent`s,
`ffmpeg`, **and a 12 GB qemu VM**, then left `test-unified-cache-fast-path`
unkillable in D state on `drm_exec_lock_obj` holding ~208 GB. Only a reboot
cleared it.

**Why the answer is `-j 1` and not a better filter:** you cannot construct a
trustworthy filtered-concurrent sweep here, because *both* classifiers fail open.
Labels fail open (`main` means "nobody classified this", not "safe" — see below).
Names fail open too: matching `test-sycl-` against the suite returns ~88 tests,
almost all of which are pure-Python parser gates that allocate nothing, so a
name-based denylist is both too broad to use and too narrow to trust. `-j 1` needs
neither classifier to be correct — if a model-loading test slips through any
filter, it runs alone, which is the case this file already documents as safe.

Form 1 (`-R <what your change gates>`) remains the recommendation. Reach for the
full sweep rarely, and accept that it is slow.

## 2. Label filters fail open

Old location: `CLAUDE.md`, "Running Tests", middle. It explains why the sweep also needs a name
filter, which tests stopped being ctests after the 694ec2354 upstream merge, and how
`scripts/check-ctest-safety-net.sh` derives the set of model-loading tests.

Correction, checked 2026-11-16 at `423fd3e78`: the registration of `test-backend-ops` is at
`tests/CMakeLists.txt` line 3807, not 2846. The moved text already tells you to locate it with a
search and not to trust a number. That advice stands.

⚠️ **`-LE 'residency|mem-handle|cache'` does NOT exclude `test-backend-ops`** —
which is why form 2 above must also carry `-E '^test-backend-ops$'`. Until
2026-07-30 it did not, so the command this file prescribed as the *safe* one ran
the single binary this file separately forbids running unattended (see Hard-Won
Rules: 50–224 GB of TTM shmem, two OOM kills). At the `-j 4` this file then
prescribed, it would have started alongside three other tests.

The cause is structural, not a typo: `tests/CMakeLists.txt` registers it as
a bare `llama_build_and_test(test-backend-ops.cpp)` with **no labels**, so it
inherits only the default `main` and no label denylist can reach it. Locate the
registration rather than trusting a line number written down anywhere — this
file pinned `:2150` until a post-b10630 merge moved it ~700 lines (it is 2846
today), which is why it no longer pins one at all:

```bash
cat tests/CMakeLists.txt | grep -n 'test-backend-ops'
```

Verify rather than assume — a label filter is silently permissive toward
anything nobody remembered to tag, so it fails *open*:

```bash
# Always confirm what a filtered sweep will actually run before running it.
bash scripts/check-ctest-safety-net.sh --build-dir build
# ^ must print "safety net intact"; if it names a test, do not run the sweep.
```

Adding `LABELS "cache"` to that registration would also fix it, but it is
upstream code that a rebase would silently revert, and the failure mode of
losing the fix is an OOM. Excluding by name is the safer belt.

⚠️ **Since the 694ec2354 upstream merge (llama.cpp-n77l), `test-backend-ops` and
`test-llama-archs` are not ctests at all** — upstream registers both with
`llama_build` only. So `ctest -R '^test-backend-ops$'` and
`ctest -R '^test-llama-archs$'` select nothing and "pass" having run nothing, and
`-E '^test-backend-ops$'` excludes nothing. (It stays in form 2's `-E` as a belt
in case upstream registers the binary again.) Run both as binaries, pinned, one
run at a time:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/test-llama-archs -a llama; echo "rc=$?"
ONEAPI_DEVICE_SELECTOR=level_zero:0   ./build/bin/test-backend-ops
```

The same merge **added model loaders under the default `main` label**. Among them:

- `test-generate-models`, the `generate-models` fixture setup, which runs
  `test-llama-archs -o` to write every arch's model;
- `test-save-load-state`, which loads every generated arch in one process;
- the `test-recurrent-state-rollback*` family.

None of them pins a selector, so on this host they run on the iGPU too. The
older `-E '^test-backend-ops$'` form of the sweep selected all of them. It also
selected four loaders that need a downloaded model: `test-thread-safety`,
`test-sycl-model-lifecycle-hooks`, `test-state-restore-fragmented` and
`test-eval-callback`.

`scripts/check-ctest-safety-net.sh` now derives the loader set from the
registration. A loader is any of these:

- a test with `FIXTURES_REQUIRED`;
- a fixture setup that runs a binary rather than cmake;
- a test labelled `model`. That is upstream's label for tests that load
  `LLAMACPP_TEST_MODELFILE` (`test-model-load-cancel`, `test-autorelease`,
  `test-backend-sampler` and others). The fork also puts it on
  `sycl-lifecycle-gpu-sequential`, which loads the models its G1 fixture
  names. Form 2 leaves them out through `-LE '...|^model$'`.

The guard fails naming each loader that form 2 still selects. It also fails if
this file stops carrying form 2's exact `-LE ... -E ...`.

When the guard names a new test, change this file and the script together:
either label the test `model`, or add it to the `-E` here and to the script's
`SWEEP_E_DEFAULT`.

The guard cannot see a test that loads a model but has no fixture and no
`model` label. Give such a test a label that form 2 excludes (`model`, `cache`
or `mem-handle`); do not trust the guard to find it.

## 3. Never repeat a model-loading test

Old location: `CLAUDE.md`, "Running Tests" (the `test-llama-archs` loop) and "Hard-Won Rules",
"Safety" (the never-loop rule). The two passages describe the same hazard, so they sit together
here. The second passage also holds the measurements that show a pinned selector keeps `Shmem`
flat, the warning about the corrupted `test-llama-archs` table, and the note that a device lock
does not wait for memory to be released.

⚠️ **`test-backend-ops` is not the only unlabelled member of that family.**
`test-llama-archs` also carries only `main`, and **looping it exhausts host
memory the same way**. Measured 2026-07-30: two separate global OOMs
(12:00:25 and 12:49:57), each during a 6-run loop of

```bash
ctest --test-dir build -R '^test-llama-archs$' --output-on-failure   # ~36 s, fine ONCE
```

(That was the registration then. The command now selects nothing; see the
n77l note above, and run the binary instead.)

Both show the TTM-shmem signature this file documents for `test-backend-ops`:
`shmem:238266228` kB ≈ **227 GB** of 255 GB total, `inactive_anon` 227.8 GB (the
same pages — GPU BO shmem backing sits on the anon LRU, so reading only `anon`
makes it look like a runaway process instead of driver-backed allocations). The
kills took out the desktop session, Xwayland, `systemd --user`, and the 12 GB
`ws2022ci` qemu VM eight times over. No test binary was itself killed and the
GPU stayed clean (no GT reset / `guc_id` / CAT error), so results from such a
run are still valid — but the host is not.

A single run is fine. **Do not loop it unattended**, and treat any repeated
GPU-allocating test the same way. The general lesson: `main` is the *default*
label, so "only `main`" means "nobody classified this", not "this is safe".

- ⚠️ **THE NEVER-LOOP RULE IS ABOUT A PROPERTY, NOT A LIST OF BINARIES.** Read this before the
  examples below, because the examples are how it gets misapplied.

  **The property: any test that loads models onto a GPU allocates GPU buffer objects whose TTM
  shmem backing does not appear in ordinary RSS accounting.** Repeating such a test accumulates
  that backing faster than it is released. Nothing in the test's name, labels, or output says so.

  This rule used to be written purely through its two examples, so it read as a fact *about those
  two binaries*. On 2026-08-01 that framing cost a near-miss: a task's acceptance criterion said
  *"passes 3 consecutive times — it is a race; one green run is not evidence"* for
  **`test-thread-safety`**, which was not on the list. Executing it drove `Shmem` from 84.9 GB to
  **206 GB in twenty seconds** and `MemAvailable` to **16.9 GB** — killed roughly twenty seconds
  short of a global OOM. The reasoning behind the criterion was statistically sound; the missing
  step was asking what that binary *allocates*.

  **Before writing "run it N times" into any script, gate, or acceptance criterion, ask whether
  the binary loads a model onto a GPU.** If yes: run it ONCE, sample
  `grep -E '^(MemAvailable|Shmem):' /proc/meminfo` before and ~5 s after, and abort if `Shmem`
  climbs past ~100 GB. Prefer a narrower reproducer, or an instrumented single run, over
  repetition. Label filters do NOT protect you (`test-backend-ops` carries no labels), and a
  healthy-looking `free` reading does not either.

  ⚠️ **The rule is not about accumulation, and that misreading is why it gets broken.** Measured
  2026-08-01 across two independent events: **a single `test-llama-archs` run peaks at
  195–206 GB of 255 GB** — ~80 % of host RAM — then releases cleanly. So a second run starting
  before the first has released does not slowly accumulate toward a limit; **it simply does not
  fit.** The first overlap is fatal and there is no gradual approach to warn you.
  A correct single run leaves only ~20 GB of margin, which is why a memory watchdog fires on
  entirely healthy work here.

  ⚠️ **`GPU.lock` (or any device lock) serialises ACCESS, not MEMORY RECOVERY.** Lock release is
  synchronous; TTM shmem release is not. Observed 2026-08-01: one run sat at `Shmem` 204 GB /
  `MemAvailable` 25 GB while another agent polled `until mkdir GPU.lock` every 15 s, ready to
  launch from a 25 GB baseline the instant it freed — **both parties obeying the protocol
  exactly, and still an OOM.** After acquiring the lock and *before* touching the device, wait
  for `Shmem < 30 GB` **and** `MemAvailable > 150 GB`; if it has not settled in ~10 minutes,
  release the lock and escalate rather than proceeding or holding it while you wait.

  Correction, 2026-11-16: the `MemAvailable > 150 GB` half of that wait is retired. A ruling of 2026-11-12
  removed free-memory conditions from run guards, because `MemAvailable` cannot tell a runaway from a 122 GiB
  model held in host memory on purpose (55 to 60 GB available with `Shmem` flat). On this host `MemAvailable`
  also idles near 144 GB, so the condition could never be met. Wait on `Shmem < 30 GB` and guard on `Shmem`.

  Amendment, 2026-11-23: still no `MemAvailable` floor, but the instrument is GPU host backing, `Shmem` plus the
  `/proc/meminfo` remainder (section 10), because `Shmem` alone is flat for a discrete card's host-pinned memory.
  The `30 GB` and `100 GB` figures keep their numbers and apply to that sum.

  ⚠️ **THE CAUSE IS NOW KNOWN, AND THERE IS A ONE-LINE MITIGATION** (`llama.cpp-403s`,
  2026-08-01). It is not model size and not concurrency — it is the **iGPU**, which reports
  231.7 GB of "VRAM" that is actually system RAM, against a budget defaulting to 100 % (see
  the VRAM-budget entry under Architecture). Every binary in the list below enumerates all
  devices because none of them pins a selector.

  **Pinning the selector to the discrete cards holds `Shmem` flat.** Measured on a full model
  load: `level_zero:0` → 2.4 GB, `level_zero:0,1` → 2.4 GB, unset → 127.8 GB.

  ```bash
  # Prefer this for ANY local test run that does not specifically need the iGPU:
  ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ctest --test-dir build -R '<name>' --output-on-failure
  ```

  **Now measured on the two worst offenders, and both collapse to nothing:**

  | binary | unpinned | pinned `level_zero:0,1` |
  |--------|---------:|------------------------:|
  | `llama-completion` (19 MB model) | 127.8 GB | **2.4 GB** |
  | `test-llama-archs -a llama` | 195–206 GB | **2.2 GB** |
  | `test-thread-safety` | 156–180 GB | **2.2 GB** |

  Both pinned runs were confirmed to do **real work**, not skip: `test-llama-archs` emitted
  its results table with numerical error magnitudes (`3.13e-11`, `0.00e+00`), and
  `test-thread-safety` logged model load, `get_rows`, and KV-cache tensors. (It still
  crashes — that is `llama.cpp-oze0`, unrelated and unchanged by the selector. Post-fix it
  exits **134/SIGABRT** via `[GRAPH-COMPUTE] arena scratch unavailable`, not SIGSEGV;
  measured twice 2026-08-01, plainly and under gdb.)

  ⚠️ **The `test-llama-archs` half of that sentence leans on a table that is CORRUPTED**
  (`llama.cpp-to9m`). GGML log output interleaves *into* the row, so cells are routinely
  truncated mid-field — across four sweeps, 7–11 of ~51 rows were mangled and **every single
  `FAIL` cell was destroyed**, making `grep FAIL` return zero on runs that had 1–3 real
  failures. The use above happens to survive, because it needs cells to be *present* rather
  than *absent*, so the `403s` selector finding stands. But do not copy the pattern:

  ```bash
  # WRONG — the table is not evidence
  ./build/bin/test-llama-archs -a llama | grep FAIL          # returns nothing on a failing run

  # RIGHT — rc, plus the prose detail lines, which survive interleaving
  ./build/bin/test-llama-archs -a llama; echo "rc=$?"        # rc=0 trustworthy; 77 = skipped
  grep -c 'roundtrip mismatch' <log>                         # a real printed string
  ```

  `rc=0` is trustworthy — `all_ok` is computed from in-memory comparisons, and a separate
  `n_measured` counter returns **77** when a targeted `-a` run measured nothing. `rc=1` is
  **ambiguous**: a `[SYCL-WATCHDOG] _Exit(1)` and a genuine FAIL share it (`llama.cpp-4lnb`).

  ⚠️ **This does NOT retire the never-loop rule; it gives it one documented escape.** An
  *unpinned* run of either binary is exactly as dangerous as before, and unpinned is the
  **default** — nothing in either binary's ctest registration sets a selector. So the burden
  is entirely on whoever runs it. Two further limits: the full `test-llama-archs` sweep (no
  `-a`) has not been measured pinned, only a single arch; and the property-based rule at the
  top of this section still governs any binary nobody has measured.

  Treat "pin the selector" as a precondition you must actively satisfy, not as a fact about
  the tests.

  Known members (**not** an exhaustive list — apply the property):
  - **`test-backend-ops`** — never in a subagent or background task. Hundreds of GPU BOs, TTM
    shmem 50–224 GB, two hangs on 2026-04-06. Run manually, with monitoring, only.
  - **`test-llama-archs`** — one run (~36 s) is fine; a 6-run loop OOM-killed this host twice.
  - **`test-thread-safety`** — 3 models × 4 contexts; the 2026-08-01 near-miss above.
  - **`test-unified-cache-bugs`** — peaks ~8.5 GB RSS; serial only.

  For automated GPU testing prefer `llama-bench`, `llama-completion`, or a targeted
  `ctest -R <name>` — single invocations, not loops.

## 4. Read `Shmem`, not `free`

Old location: `CLAUDE.md`, "Running Tests". A memory reading taken while a process is still exiting
gives a false alarm.

⚠️ **A post-run `free -g` sampled too early manufactures a FALSE OOM scare — let the
process settle, and read `Shmem`, not `free`.** Measured 2026-07-31: immediately after a
`test-llama-archs` single-arch run, `free -g` reported **14 GB free**, down from 166 —
below the 100 GB floor above and a dead ringer for the ~227 GB TTM-shmem signature. Five
seconds later the same host read **182 GB free / 210 GB available, `Shmem` 3.9 GB**, with
no process alive. `free` had run in the same command as the exiting process, before its
page cache and GPU BO backing were released.

The reflex a low reading triggers is "stop, the host is dying", which is expensive when
wrong — the agent that hit this correctly halted a verification mid-sequence. So:

```bash
# wrong: samples the instant the process is still tearing down
./build/bin/test-llama-archs -a <arch>; free -g

# right: let it settle, and read the number that actually tracks the hazard
./build/bin/test-llama-archs -a <arch>
sleep 5; grep -E '^(Shmem|MemAvailable):' /proc/meminfo
```

`Shmem` is the figure that goes to ~227 GB in a real event; `free` conflates it with
reclaimable cache. This is the mirror image of the stale-guardrail trap recorded under
Regression Baselines — a false **alarm** rather than a false **all-clear** — and it has
the same root: a clean answer about the wrong instant.


## 5. Tests that are always safe, and the check after an OOM

Old location: `CLAUDE.md`, "Running Tests".

Pure-Python gates (`test-sycl-gap-causes`,
`test-sycl-timeline-gap-class-conservation`, `test-jinja-py`) allocate nothing
and are always safe at any parallelism.

After any OOM or forced stop, **check the GPU before trusting a benchmark** —
`journalctl -k --since "1 hour ago" --no-pager | grep -iE 'GT reset|guc_id|CAT error'`.

## 6. GPU work runs in the lead session

Old location: `CLAUDE.md`, "Running Tests", last block. The "lead session" is the one session that
talks to the human and coordinates the others. A "subagent" is a worker session that the lead
starts.

⚠️ **GPU/model-loading work is SERIALISED THROUGH THE LEAD SESSION. Subagents
write code; they do not run it on the GPU.**

This generalises the long-standing "never `test-backend-ops` in a subagent" rule
to the whole family, and it exists because lock-passing between agents provably
does not prevent overlap. On 2026-08-01 three GPU workloads ran concurrently and
OOM'd the host while every party believed it was following the protocol — one
agent held `GPU.lock` legitimately, a second polled for it correctly, and a third
was launched by a sweep whose own documentation called it safe. No single actor
misbehaved; the failure was in the seam between them.

Locks cannot close that seam, for a reason recorded above: **`GPU.lock`
serialises access, not memory recovery.** Lock release is synchronous and TTM
shmem release is not, so a correct handoff can still hand the next holder a 25 GB
baseline. Adding more protocol to a mechanism with that property does not
converge.

The workable division of labour:

- **Subagents:** read, analyse, edit, build, and run non-GPU tests (pure-Python
  gates, CPU-only unit tests). They report what they need verified.
- **The lead session:** runs every `llama-bench`, `llama-cli`/`llama-completion`
  gate, `test-llama-archs`, `test-thread-safety`, `test-backend-ops`, and any
  `ctest` sweep — one at a time, sampling GPU host backing (section 10) before and ~5 s
  after, never overlapping two.

This costs nothing in throughput. GPU verification is *already* serial — the
memory does not fit two model-loading tests concurrently — so making the serial
point explicit removes the coordination failure without removing parallelism that
ever existed. What parallelises is the code work, and that stays parallel.

## 7. Timeouts, crashes, the kernel log and competing load

Old location: `CLAUDE.md`, "Hard-Won Rules", "Safety", the items after the never-loop rule.

- **Always `timeout 60` GPT-OSS 20B test runs.** The historical host-MoE-routing hang (GuC `guc_id=6`, unrecoverable, requires reboot) was closed by commit `ec7f04ac4`, but keep the timeout as a guard. Distinguish the userspace hang (`guc_id=6`, attributed `in <llama-bench>`, unrecoverable) from the benign environmental XE timeout (`guc_id=0`, `in no process [-1]`, auto-recovers).
- **Benchmark numbers are invalid after any crash or forced stop on that card** (xe GT reset cascades) — check the kernel log first. `SAFE_MODE`/op-timing diagnostics can themselves stall cards.
- **`dmesg` is privilege-denied for this user** (`read kernel buffer failed: Operation not permitted`). Every "check dmesg" instruction in this repo's docs must be run as:
  ```bash
  journalctl -k --since "1 hour ago" --no-pager | grep -iE 'GT reset|guc_id|GPU hang|xe.*reset'
  ```
  which works unprivileged. A silent `dmesg` failure looks identical to a clean log.
- **Check for competing load before any throughput measurement.** `uptime` plus `pgrep -af 'codescout|ninja|icpx|ffmpeg'`. A codescout re-index has been observed holding 600–1430% CPU for hours; Frigate ffmpeg adds ~250%. Interleaved paired A/Bs survive sustained load, but **absolute** numbers taken under it are depressed and must not become baselines.
  ⚠️ **Owner ruling 2026-08-07: this host's ambient load (~60 — Emby, the ws2022ci qemu VM, Frigate ffmpeg) is permanent. Never defer or block a measurement waiting for a quiet host** — "re-run when quiet" is not an available disposition. The check above still runs (so you know what you measured under), verdicts come from interleaved paired A/Bs on the same host state, and absolute numbers taken under load still must not become baselines. Note the floors in `docs/backend/sycl-perf-baselines.md` were themselves validated under load (2026-07-25), so a floor miss is not automatically excused by load — discriminate with a pairing against a known-good build before blaming either.

## 8. The integrated GPU and the VRAM budget

Old location: `CLAUDE.md`, "Hard-Won Rules", "Architecture". This is the cause that section 3 points
to as "the VRAM-budget entry under Architecture".

Correction, checked 2026-11-16 at `423fd3e78`: the second paragraph says that neither
`ggml-sycl.cpp` nor `unified-cache.cpp` mentions `host_unified`, and that the information is "never
consulted". That is no longer true. `compute_vram_budget_authority()` in `unified-cache.cpp` takes a
`host_unified` argument, and `ggml_sycl_vram_budget_base_mem()` in `ggml-sycl.cpp` reduces the base
for an integrated device. See "The VRAM budget authority" in `docs/backend/sycl-memory-design.md`.
The line numbers `ggml-sycl.cpp:10043-10049` and `common.cpp:716` have also drifted
(`caps.global_mem_size` is assigned near `common.cpp` line 1074 today). The `Shmem` figures in the
third paragraph are measurements from 2026-08-01 and have not been taken again since the code
changed. Until someone measures an unpinned run again, the rule to pin the selector stands as
written.

- **The VRAM budget calc is correct by design for DISCRETE cards** (`min(total*pct, free_at_init)`). Low free VRAM is a system problem (other GPUs active, driver overhead), not an app bug to "fix" by ignoring free VRAM — fix the root cause at the system level.
  ⚠️ **It is catastrophically wrong for an INTEGRATED GPU, and that is the cause of this host's OOM history** (`llama.cpp-403s`, measured 2026-08-01). The Arrow Lake-S iGPU reports `global_mem_size` = **231.7 GB** — 94 % of the host's 246.9 GB — because for an integrated GPU "VRAM" *is* system RAM. `ggml-sycl.cpp:10043-10049` feeds that into the same budget path as a discrete card at a **default of 100 %**, and neither `ggml-sycl.cpp` nor `unified-cache.cpp` contains a single occurrence of `host_unified` or `is_integrated`. So the backend claims the machine.
  Isolated with one variable — same 19 MB model, same single-threaded `llama-completion`, only the selector changed: `level_zero:0` → peak `Shmem` **2.4 GB**; `level_zero:0,1` → **2.4 GB**; selector unset (adds the iGPU) → **127.8 GB**.
  For an integrated device, "free VRAM" and "free host RAM" are the same pool, so taking 100 % of it is not a conservative reading of available memory — it is claiming memory the host needs. `sycl::info::device::host_unified_memory` is queryable and `caps.global_mem_size` is already captured at `common.cpp:716`; the information is present and simply never consulted.

## 9. Device selection: the device map and its history

Old location: `CLAUDE.md`, "SYCL Device Selection". `CLAUDE.md` keeps the selector table, the
command that derives the map live, and the rules. This section holds the full text, with the
history of the two times the map was wrong.

Use `ONEAPI_DEVICE_SELECTOR` (syntax `backend:devices`): `level_zero:0` for one
device, `level_zero:0,1` for a numeric multi-device set, `level_zero:gpu` for all
GPUs. The `level_zero:gpu:0` strings some tools print are display IDs, not valid
selector values. This system (as of 2026-07-24): **Arc Pro B70** (device 0,
Battlemage G31, 256 CU, ~32.6 GB), **Arc Pro B50** (device 1, Battlemage G21,
128 CU, ~16 GB), iGPU (device 2, Arrow Lake-S). Single-GPU B50 validation runs
with `level_zero:1`.

The B580 that earlier notes reference has been **replaced by the B70** — it is
no longer in this machine. Treat any B580 figure as historical.

PCI / DRM mapping, because the numbering is not intuitive and the logs do not
disambiguate it:

| selector | card | PCI | render node | DRM card |
|----------|------|-----|-------------|----------|
| `level_zero:0` | Arc Pro B70 (G31) | `0000:04:00.0` (was `03:00.0` before the 2026-09-05 boot) | `renderD129` | `card0` |
| `level_zero:1` | Arc Pro B50 (G21) | `0000:09:00.0` (was `07:00.0` before the 2026-09-05 boot) | `renderD130` | `card2` |
| — | Arrow Lake-S iGPU | `0000:00:02.0` | `renderD128` | `card1` |

⚠️ **The PCI addresses MOVED at the 2026-09-05 19:03 boot** (verified 2026-09-07 with
`lspci` + the sysfs links below): B70 `03:00.0 → 04:00.0`, B50 `07:00.0 → 09:00.0`;
render nodes, DRM card numbers, and the `level_zero:N` order did not move (level_zero:0
still loads as "Arc Pro B70", level_zero:1 as "Arc Pro B50" in `llama-bench -v`). Any
script that hardcodes a PCI address is stale on this boot: `scripts/bench-guard.sh` did
and refused every run with `no DRM card for PCI 0000:03:00.0` (llama.cpp-imns; fixed to
derive the card live from the DRM/PCI enumeration). Pre-2026-09-05 notes quoting
`03:00.0`/`07:00.0` (the P2P section above included) are history, not the current map.
The live derivation itself lives in one shared file, `scripts/sycl-gpu-sysfs.sh`
(`derive_card_for_selector`/`is_top_level_card`/`find_card_by_pci`), sourced by all
three of its consumers -- `bench-guard.sh`, `sycl-decode-mode-capture.sh`, and
`sycl-gpu-preflight.sh` (llama.cpp-o4fs) -- so none of them can drift out of sync
with each other again.

⚠️ **This table had `renderD128` and `renderD129` SWAPPED until 2026-07-30** — it
listed the B70 as `renderD128`, which is actually the iGPU. Verified against the
live sysfs links (below) plus `lspci`: `0000:03:00.0` is Battlemage G31 (B70) and
`0000:00:02.0` is Arrow Lake-S. The swap is not cosmetic — Frigate's QSV jobs run
on `renderD128`, so with the old table any `pgrep -af ffmpeg` showing
`-qsv_device /dev/dri/renderD128` read as **"Frigate is contending on the B70"**,
which would make a healthy benchmark look invalid and send you hunting for
contention that isn't there. It is the reverse of the ~1100 PP512 stale-guardrail
trap: a false *invalidation* rather than a false regression.

**Derive it live rather than trusting the table** — this numbering can move across
boots, and the whole point of the entry below is that you cannot infer it:

```bash
for n in /sys/class/drm/renderD*; do
  printf '%s -> ' "$(basename $n)"
  readlink -f $n/device | grep -oE '[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]' | tail -1
done
# then name each PCI id:  lspci -s 04:00.0   ->  Battlemage G31 = B70   (03:00.0 before 2026-09-05)
```

DRM `cardN` numbering and `renderDN` numbering are **independent** — do not infer
one from the other (note `card0` is the B70 while `renderD128` is the iGPU: the
two sequences do not even agree on which device comes first). The `device=N`
printed in SYCL logs is the in-process index *after* `ONEAPI_DEVICE_SELECTOR`
filtering, not the physical card: a B50-only run and a B70-only run both print
`device=0`. Key off the selector, never the log id.

Before trusting any B70 benchmark, check the reported free VRAM in the startup
log. Other workloads (e.g. ComfyUI) can hold tens of GB on that card; a run that
sees ~13.8 GB free instead of ~32.6 GB is measuring under memory pressure and its
numbers are not comparable.

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench ...   # B70
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench ...   # B50
ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/llama-bench ... # host-bounce paths required
```

Rules:
- Device selection belongs to oneAPI/SYCL — do not add code that parses,
  rewrites, or refuses `ONEAPI_DEVICE_SELECTOR`, and don't let previous-boot
  evidence alter fresh-boot behavior. `GGML_SYCL_VISIBLE_DEVICES` does **not**
  work (filters at llama.cpp level; unified cache still sees all L0 devices).
- `common_params_parse()` must keep `--help`, `--version`, cache-list, and
  completion generation metadata-only even with `LLAMA_ARG_*` GPU env vars set;
  verify with `test-arg-parser` after parser changes.
- **Hang hazards (unrecoverable D-state):** `sycl-ls` is not a safe B50 health
  check (has hung this host in `xe_drm_ioctl` after a reset/oops). Do not run
  old comparison binaries (e.g. the `60a8c042` build) with `--help`/`--version`/
  `lsof /dev/dri/*` on the discrete render nodes — SYCL init alone left a process
  stuck in `xe_vm_destroy_ioctl` that FLR/GT reset could not clear. Use the
  canonical gated inference command for cross-build comparisons after a fresh
  reboot, and avoid DRM fdinfo checks while a SYCL process is hung.

Current-boot B70/B50 P2P topology warnings are diagnostic only. Frigate
QSV/OpenVINO jobs on the iGPU render node are not B70/B50 consumers — **re-verified
2026-07-30**: all 7 QSV-decoding `ffmpeg` processes pass
`-qsv_device /dev/dri/renderD128`, and `renderD128` is `0000:00:02.0`, the iGPU.
This conclusion was right while the table above was wrong, which is the worst
combination: the prose said "not a B70 consumer" and the table said `renderD128` was
the B70, so checking one against the other manufactured a contradiction. Frigate
does cost ~250 % CPU (30 `ffmpeg` processes here), so it still depresses *absolute*
throughput numbers via CPU contention — just not via the B70's execution units.

## 10. GPU host backing: `Shmem` plus the `/proc/meminfo` remainder (2026-11-23)

Every guard before this date watched `Shmem` only. That is blind to a discrete card's host memory.

**Measured 2026-11-23** (integ/th32 `9b5d7d183`, B70 pinned with `level_zero:0`, Qwen3.8 exps-IQ4_NL, default
load mode, `-ub 2048`, kernel 7.1.2, compute-runtime 26.44). The plan pinned 48133.3 MB of routed host
experts in SYCL host USM. While loaded, `MemAvailable` fell from 102.6 GiB to 41.8 GiB and `Shmem` stayed at
11.4 to 11.5 GiB. `Mlocked` was 2.0 GiB (the storage row cache only). The process's own `/proc/<pid>/status`
showed `RssShmem` 0 and `VmPin` 0. Its xe fdinfo showed `drm-total-gtt` = `drm-resident-gtt` = 52.1 GiB. The host
memory of such an allocation belongs to the xe driver. It lowers `MemAvailable` and is counted in no category
and in no process figure. About 45 sweep arms since 2026-11-15 printed a peak `Shmem` equal to the idle figure
of their day.

**The formula.** The unaccounted part of `/proc/meminfo` is

```
remainder = MemTotal - (MemFree + Buffers + Cached + SwapCached + AnonPages + Slab
                        + PageTables + KernelStack + Percpu + Hugetlb)
GPU host backing = Shmem + max(0, remainder)
```

The remainder was 54.0 GiB while the model was loaded and 2.7 GiB two minutes after exit. It is read from
`/proc/meminfo` alone, so it is safe while a SYCL process hangs (the fdinfo rule stays: no fdinfo checks then).
`scripts/sycl-host-backing.sh` computes it (`sycl_gpu_host_backing_kb`, `sycl_gpu_host_backing_gib`) from a
meminfo file, `/proc/meminfo` by default. A missing field counts as 0. A file that cannot be read, or that has
no `MemTotal`, returns non-zero and prints nothing, and every caller treats that as "do not run". Another
signature stays visible in the sum: the iGPU and the `test-backend-ops` class land in `Shmem`, which the sum
contains. That `Shmem` signature belongs to those two classes and not to a discrete card. One inference, not measured: the
2026-08 measurement above (a pinned selector holds `Shmem` at 2.4 GB on a full model load) probably shows the
iGPU's signature missing, not that nothing was held.

**Why the sum does not see the other tenant.** The other session holds a 122 GiB model in `AnonPages`
(126 to 130 GiB), which the formula subtracts. Measured with that tenant's model held (`AnonPages` 126.4 GiB):
remainder 1.9 GiB, sum 12.9 GiB. `MemAvailable` sits near 55 GB then, by design, which is why no
`MemAvailable` floor is used (ruling 2026-11-12). Recorded snapshots and the test:
`tests/sycl-meminfo-fixtures/` and `tests/test-sycl-host-backing.sh`.

**Thresholds.** They keep their numbers and apply to the sum.

- Idle sum on this host: 12.7 to 15 GiB (`Shmem` 11 to 12 GiB of ordinary tmpfs, remainder 2 to 3 GiB).
  Recorded idle snapshot with the tenant's model held: 12.7 GiB. Recorded Qwen3.8 default-mode load: 65.4 GiB.
- Settle floor, "below 30 GB": about 15 GB above idle, so a leftover of that size from the previous run blocks
  the next one.
- Abort ceiling, "past ~100 GB": the largest planned load is the planner's host budget (`MemAvailable` minus
  about 31 GB, about 76 GB pinned) plus idle, about 90 GB. The host is short at about 110 GiB: `MemTotal`
  247 GiB minus the tenant's ~127 GiB minus kernel memory. 100 GB sits between 90 and 110.

`scripts/bench-guard.sh` keeps its own, lower ceiling (10 GiB) and still nets ordinary tmpfs out of `Shmem`
before it adds the remainder.
