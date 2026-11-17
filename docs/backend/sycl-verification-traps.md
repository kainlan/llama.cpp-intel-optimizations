# Verification gates: detail and the ways a check can pass without proving anything

`CLAUDE.md` holds the correctness gates as exact commands with their scoring. This page holds the
detail behind them: where the models are, the cases where a gate or a test reported success without
doing the work, and the `llama-bench` and logging behaviour that hides output.

A "gate" is a command whose output must match a stated result before a change is accepted. "PP" is
prompt processing and "TG" is token generation; both are reported in tokens per second.

The text under each heading below was moved out of `CLAUDE.md` on 2026-11-16 and is unchanged,
apart from the corrections that are marked as such. In the moved text, "this file" means
`CLAUDE.md` as it stood before that date, and "above" or "below" refer to its old layout.
Line numbers quoted in the moved text were right on the date given beside them and have drifted
since. Find the code by the quoted text, not by the number.

The GPT-OSS rationale is in `docs/backend/gpt-oss-testing.md`. That file is not changed by this
move, because `scripts/sycl-moe-exec-gate.sh --make-prompt` reads its bytes to build a prompt, so
adding text there would change that prompt. The host-safety history is in
`docs/backend/sycl-host-safety.md`.

## 1. Model locations

Old location: `CLAUDE.md`, "Development Workflow (Machine-Specific)", "Model Locations". The list
is not complete: `/models/` also holds other model families today. List the directory to see them.

Models are stored on the local SSD in `/models/`:

**Mistral 7B variants** (standard benchmark model):
- `mistral-7b-v0.1.Q4_0.gguf` (3.9G) - **Default for benchmarks**
- `mistral-7b-v0.1.Q4_K_M.gguf` (4.1G) - Good quality/size balance
- `mistral-7b-v0.1.Q8_0.gguf` (7.2G) - Highest quality
- Other variants: Q2_K (2.9G), Q3_K_S/M/L, Q4_K_S, Q5_0/K_S/K_M, Q6_K

**GPT-OSS models** (large MoE, native MXFP4):
- `gpt-oss-20b-mxfp4.gguf` (12G) - Smaller variant
- `gpt-oss-120b-mxfp4-*.gguf` (60G total, 3-part split) - Full model

## 2. The gates check tokens, not the backend

Old location: `CLAUDE.md`, "Verification Commands & Correctness Gates", first warning.

⚠️ **The gates below check TOKENS, not which backend produced them.** A CPU-only
build passes every one of them — CPU inference emits the identical digits, just
~13x slower. This is not hypothetical: a reconfigure silently reset `GGML_SYCL`
to OFF on 2026-07-25, the build succeeded, and the Mistral gate passed green at
**8.38 tok/s** instead of ~108 with the SYCL backend absent from the binary. The
gates were designed to catch a broken kernel emitting garbage; a *missing*
backend is not a wrong answer, so they are blind to it.

**Run this first, before trusting any gate result or benchmark**, whenever
anything has touched `CMakeCache.txt` (a reconfigure, a `-D` probe, a cache edit):

```bash
grep -E '^GGML_SYCL:' build/CMakeCache.txt                        # want BOOL=ON
ldd build/bin/llama-completion | grep -cE 'libggml-sycl|libsycl'  # want >= 2
```

A run reporting single-digit tok/s on Mistral Q4_0 is a CPU fallback, not a
regression to investigate. Recovery is cheap: delete **only** `CMakeCache.txt`
and re-run `./scripts/sycl-build.sh` — ccache absorbs most of it, no `-c` needed.

## 3. A SYCL test run without oneAPI prints SKIP and exits 0

Old location: same section, second warning.

⚠️ **The same blindness in test form: a SYCL test binary run WITHOUT sourcing
`setvars.sh` prints `SKIP` and exits 0.** Verified 2026-08-01:

```
$ ./build/bin/test-mem-ops
SKIP: no SYCL GPU devices available
$ echo $?
0
```

**Exit 0. Green. It proves nothing** — the device test found no device and
reported success. Under `ctest` it passes for real, because the registration's
`ENVIRONMENT` property supplies `LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib`;
this bites only someone invoking a binary **directly**, which is exactly what
you do when iterating on one failing test. Nothing in the output distinguishes
"the device path ran and was correct" from "there was no device". So:

- **Source oneAPI first**, every time, even for a single binary.
- **Treat an implausibly short runtime as a signal.** This was caught only
  because a 0.4 s GPU test was not believed.
- **A `SKIP` line with status 0 is not a pass.** Where a test has been fixed it
  now exits **77** (ctest's `SKIP_RETURN_CODE`, already used by the
  `test-sycl-*-policy.sh` family), so ctest reports *skipped* and a bare shell
  run gets a non-zero status. The goal is to make a skip **visible as a skip**,
  not to forbid skipping — a CPU-only runner still legitimately skips.

## 4. What the test registration provides, a direct run does not

Old location: same section, third warning.

Correction, checked 2026-11-16 at `423fd3e78`: the download fixture is no longer at
`tests/CMakeLists.txt:312-329`. Find it with
`cat tests/CMakeLists.txt | grep -n 'test-download-model'` (it is near line 819 today).

⚠️ **Generalise it: WHAT THE REGISTRATION PROVIDES, DIRECT INVOCATION DOES NOT.**
`LD_LIBRARY_PATH` above is one instance; the **model fixture** is another, and it
bites the same way. `tests/CMakeLists.txt:312-329` downloads `stories15M-q4_0.gguf`
via a ctest fixture — `test-download-model` carries `FIXTURES_SETUP`,
`test-thread-safety` carries `FIXTURES_REQUIRED` — so `ctest -R '^test-thread-safety$'`
stages it automatically and `./build/bin/test-thread-safety -m …` does not. And
`build/` is gitignored (`/build*`), so a staged copy does **not** survive
`sycl-build.sh -c`.

Stage it without running any GPU work — safe for any agent, unlike recovering via
the gate itself:

```bash
ctest --test-dir build -R '^test-download-model$'
```

Before blaming a directly-invoked test, check what its **registration** supplies:
`awk '/add_test|set_tests_properties.*<name>/,/\)/' build/**/CTestTestfile.cmake`.

## 5. Search for the string the code prints, not the name that produces it

Old location: same section, fourth warning.

Correction, checked 2026-11-16 at `423fd3e78`: `ggml_abort` is defined at `ggml/src/ggml.c` line 253
today, not 252. The two macro lines in `ggml/include/ggml.h` are still at 300 and 301.

⚠️ **GREP THE STRING THE CODE PRINTS, NOT THE NAME THAT PRODUCES IT.** Measured
2026-08-01, and it reached a P1 ticket before anyone checked. `GGML_ABORT` count was
reported as **0** to show a change introduced no new aborts:

```c
ggml.h:300   #define GGML_ABORT(...) ggml_abort(__FILE__, __LINE__, __VA_ARGS__)
ggml.c:252   void ggml_abort(const char * file, int line, const char * fmt, ...)
```

The macro expands to a call printing `file:line: message`. **The literal string
`GGML_ABORT` never reaches a log, ever.** Two runs that demonstrably aborted
*through* `ggml_abort` (exit 134; the frame visible in the backtrace) both returned
`grep -c 'GGML_ABORT'` → **0**. That probe was guaranteed zero in every run ever
taken, so "no new aborts" was never tested.

Probes scored against a run that provably aborted (the committed gdb backtrace, in
which the `ggml_abort` frame appears):

| probe | count | verdict |
|---|---|---|
| `GGML_ABORT` | **0** | void — zero on a run that provably aborted |
| `arena scratch unavailable` | 1 | ✅ use this |
| `ggml-sycl\.cpp:[0-9]+:` | 1 | ✅ use this |
| `ggml_abort` (lowercase) | 2 | ⚠️ **trap** — matched only because *gdb symbolised the frame* |

⚠️ **Do not "fix" it by lowercasing to `ggml_abort`.** That is the obvious repair and
it is wrong in the same way one layer down: it passes on a gdb log and returns 0 on
every plain run.

**Why the mistake is seductive, and why the sibling macro is fine:**

```c
ggml.h:301   #define GGML_ASSERT(x) if (!(x)) GGML_ABORT("GGML_ASSERT(%s) failed", #x)
```

`GGML_ASSERT` bakes its own name into a **format string**, so it survives to the
output and *is* a valid probe. `GGML_ABORT` is consumed by the preprocessor. Two
adjacent, nearly identical identifiers, opposite answers — which is exactly why
symmetry is not a substitute for a positive control.

The general test, and it is one line of thought: **could this check have produced a
different result if the property were false?** If you cannot produce a positive
control — a run where the condition is known true and the probe fires — the probe is
decorative and its zero means nothing.

## 6. `test-llama-archs -a <arch>` and the exit status 77

Old location: same section, fifth warning.

⚠️ **`test-llama-archs -a <arch>` had the same shape and it is the one that
matters most**, because the whole point of `-a` is to answer "is this
architecture correct?". Several archs are excluded by the harness itself
(`gemma4`, `gemma4-assistant`, `eagle3`, `dflash` emit **no row at all**;
`gemma-embedding`, the BERT family and RWKV are `arch_supported() == false` and
emit an all-`SKIP` table). Every one of those used to print an empty or all-SKIP
table and **exit 0**, which reads as "verified". They now exit 77 with an
explicit "this run proves NOTHING about it". A full sweep (no `-a`) is
unaffected.

This is the third member of one family, and all three are clean answers about
the wrong thing: the CPU-fallback gate above (right tokens, wrong backend), the
`free -g` pre-run check (right number, wrong instant — see Running Tests), and
this (right status, no work done).

## 7. The gate command block as it stood

Old location: same section, the command block, with its full comments. The commands themselves
are also in `CLAUDE.md`. In the block, "the GGML_ABORT entry above" is section 5 of this page.

Corrections, checked 2026-11-16 at `423fd3e78`: the comment cites `AGENTS.md:232-236`. `AGENTS.md`
has 143 lines today and no longer carries the gate, so that citation points at nothing. The `-cnv`
registration is at `common/arg.cpp` lines 1916 to 1926 today, not 1908 to 1918. Find it with
`cat common/arg.cpp | grep -n '"-cnv"'`.

```bash
source /opt/intel/oneapi/setvars.sh --force

# Mistral completion gate — deterministic; output must start:
#   1, 2, 3, 4, 5, 6, 7, 8, 9, 10
# `-n 15` caps generation at 15 NEW tokens; reaching "15" would take ~20 under
# Mistral's tokenizer, so the run stops at 10 on EOS. (Matches AGENTS.md:232-236
# and every observed run. An older note here claimed the output ends
# "...11, 12, 13, 14, 15" — that string is unreachable at -n 15 and made a
# passing gate read as a failure.)
# Any other output (###..., repetition, <unk>) = broken path; fix before commit.
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-completion \
  -m /models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0

# GPT-OSS B50 chat correctness gate. With --no-display-prompt the prompt echo
# lands on the interactive "> " line and the model's ANSWER is the next line,
# on its own:
# ✅ RUNS AT DEFAULT CONTEXT (llama.cpp-uize closed 2026-10-03). llama-cli
# defaults to n_ctx_train=131072, which is ~3.2 GB of KV and more than the B50
# has left after the weights. The unified cache now re-places the overflow
# instead of refusing. Verified on master 77fad673a:
#   W [SYCL-PLAN] KV overflow re-placed to host tier: 9 layer(s) demoted, 0 SWA
#     (2304.0 MB host KV, layers 7..23) on device 0 for n_ctx=131072 ...
#     Largest all-VRAM context is about -c 34304.
# Result: rc=0, digit line count=1. Generation is 16.2 t/s, against 25.5 t/s at
# -c 4096, because the demoted layers' attention runs on the CPU. So the gate
# below carries NO -c. Keep a `-c 4096` arm as the all-VRAM control when you
# A/B; `-c` stays a normal user option (owner rulings, uize c-g612/c-xi9g:
# never shrink context, place KV). A "runtime KV update rejected:
# budget-exceeded" line now means even the host tiers could not hold the KV.
# Its arithmetic (the largest context that fits) is the fallback diagnostic.
# The old `-c 4096` workaround text is retired. The fxrg byte-contract defect
# that also broke this path is fixed (c1f4504c8 + ff9d10cd0).
# llama-bench GPT-OSS runs never went through this path.
#   > Count from 1 to 5. Answer with only: 1, 2, 3, 4, 5
#   1, 2, 3, 4, 5
# ⚠️ SCORE IT WITH `grep -cx`, NOT `grep -c` -- "count=1" above is only true of
# the exact-line form. The prompt echo on the "> " line ENDS with the same digit
# sequence, so a loose grep returns **2** on a passing gate (measured 2026-09-18);
# anyone scoring for 1 reads green as red. This is the same trap as the stale
# ": 1, 2, 3, 4, 5" colon form noted below -- the prompt contains the answer.
#   cat <log> | grep -cx '1, 2, 3, 4, 5'   # 1 = pass (the ANSWER line alone)
# Negative control, so the scorer is not taken on faith -- the echo line alone
# must score 0:
#   echo '> Count from 1 to 5. Answer with only: 1, 2, 3, 4, 5' \
#     | grep -cx '1, 2, 3, 4, 5'           # 0
# ⚠️ `-cnv` IS NOT VALID ON llama-cli AND HAS BEEN REMOVED FROM THE COMMAND BELOW
# (verified 2026-09-18, llama.cpp-9p6c). Passing it gives
# `error: invalid argument: -cnv`. This note previously said "`-cnv` still exists
# post-merge", which is true of the PARSER and false of this BINARY -- the exact
# trap the GGML_ABORT entry above warns about, one level up.
# `common/arg.cpp:1908-1918` registers {"-cnv","--conversation"} (and -no-cnv)
# with `.set_examples({LLAMA_EXAMPLE_COMPLETION})` -- COMPLETION only. Its
# neighbours show the filter is deliberate, not an oversight: `-sp` carries
# {COMPLETION, CLI, SERVER} and `-st` carries {COMPLETION, CLI}, which is why
# `-st` below still works. So grepping arg.cpp for "-cnv" returns a REAL hit for
# a flag `llama-cli` will still reject; `set_examples()` is a per-tool visibility
# filter applied after registration.
# The reason is the upstream main split: `llama-cli` IS the conversation tool
# (conversation is not a mode you enable there, it is what the binary does),
# while `llama-completion` does raw completion and needs `-cnv` to opt IN /
# `-no-cnv` to opt OUT -- which is how ci/run.sh and tools/quantize/tests.sh use
# it. Do not "restore" `-cnv` here; it would break the gate again.
# Ask the BINARY, never the parser source -- `--help` is generated from the
# filtered option set, so it cannot lie about this:
#   ./build/bin/llama-cli --help | grep -c cnv     # 0 = correctly absent
# The gate is the digit sequence. (An older note here said the output starts
# ": 1, 2, 3, 4, 5" — that colon was the tail of the echoed PROMPT, which itself
# ends "...only: 1, 2, 3, 4, 5", captured in a pre-`--no-display-prompt` form.
# Grepping for the colon form makes a passing gate read as a failure.)
# Use the GGUF tokenizer.chat_template metadata. Do not force
# `--chat-template gpt-oss`; that selects the older native formatter.
# No `-ngl`: the unified cache places every weight (ruling 2026-10-17). This command carried `-ngl 99`
# until 2026-11-16.
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-cli \
  -m /models/gpt-oss-20b-mxfp4.gguf \
  -st --simple-io --no-display-prompt \
  --chat-template-kwargs '{"reasoning_effort":"medium"}' \
  --reasoning-format none --reasoning-budget 0 \
  -p 'Count from 1 to 5. Answer with only: 1, 2, 3, 4, 5' \
  -n 48 --seed 42 --temp 0

# Benchmark prompt processing (PP) and token generation (TG)
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128

# Test backend operations (after modifying ggml operators)
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-backend-ops
```

## 8. `llama-bench` traps and log verbosity

Old location: `CLAUDE.md`, "`llama-bench` traps (both cost real time)".

Correction, checked 2026-11-16 at `423fd3e78`: the `common/log.cpp` line numbers have drifted. The
mapping is in `common_log_get_verbosity()` (near line 529), the threshold `common_log_verbosity_thold`
is defined near line 30, and the comparison `verbosity <= common_log_verbosity_thold` is in
`common_log_default_callback()` (near line 544). In `tools/llama-bench/llama-bench.cpp` the model is
freed where the code says "keep the same model between tests when possible" (near line 2396).

**1. `-v` is MANDATORY when you want log output.** `llama-bench` installs a null
log callback (`tools/llama-bench/llama-bench.cpp`, `if (!params.verbose)
{ llama_log_set(llama_null_log_callback, NULL); }`), so **every** `GGML_LOG_INFO`
— all `[SYCL-PLAN]`, `[UNIFIED-CACHE]`, `[VRAM-ARENA]`, `[HOST-ARENA]`,
`[MOE-LAYOUT]` lines — is silently discarded without it.

⚠️ **There is a SECOND, broader mechanism, and this section described only the
first until 2026-07-30.** `GGML_LOG_INFO` is dropped at default verbosity in
**every** tool, not just `llama-bench` — `llama-cli`, `llama-completion`, the
lot. Upstream `67b2b7f2f` ("logs : reduce", #23021) maps `GGML_LOG_LEVEL_INFO`
to `LOG_LEVEL_TRACE` (4) in `common_get_verbosity()` (`common/log.cpp:444`),
while `common_log_verbosity_thold` defaults to `LOG_DEFAULT_LLAMA` =
`LOG_LEVEL_INFO` (3) (`common/log.cpp:29`, `common/log.h:24-32`). The gate at
`common/log.cpp:456` is `verbosity <= thold`, so `4 <= 3` is false and the line
never reaches the sink.

Consequences worth knowing before you debug the wrong thing:

- **This is not a SYCL bug.** It silently affects every backend's init output.
  "My backend prints nothing at startup" on CUDA, Vulkan or Metal is the same
  mechanism.
- A default `llama-completion` run emits **no** `llama_model_loader:`, no
  `print_info:`, no `Found N SYCL devices:` — measured: ~58 s of model loading
  with zero library output.
- `llama-bench -v` fixes only the *first* mechanism (it suppresses that tool's
  null callback); it does **not** raise `common_log_verbosity_thold`. Two
  different levers.
- A line that *does* appear at default verbosity is either `GGML_LOG_WARN`/
  `ERROR` or a raw `fprintf(stderr)` bypassing the log system — e.g.
  `[SYCL] GGML_SYCL_F16 build:` (`fattn.cpp`), which is why it survives when the
  banner around it does not. A missing timestamp prefix is the tell.
- If you need a diagnostic visible in a normal run, emit it at **WARN**, not
  INFO. `ggml_check_sycl()`'s non-default-settings line does exactly this
  (`e7e2667bc`), and `tests/test-sycl-env-report.cpp` gates that it keeps
  reaching the callback. A grep then returns
nothing, which is indistinguishable from "my change had no effect" or "that zone
never got sized". **Always confirm a RED capture is non-empty before changing
code**; an empty baseline proves nothing and voids the before/after.

```bash
timeout 900 env ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench \
  -m /models/gpt-oss-20b-mxfp4.gguf -p 0 -n 4 -r 1 -v 2>&1 | grep ...
```

Note `-p 0 -n 4` triggers planning but does **no prompt processing**, so paths
reached only during PP will show zero activity. Use `-p 512` when you need them.

**2. Multiple `-m` values are sequential model switches.** `llama-bench` frees
the previous model before loading one with different model parameters
(`tools/llama-bench/llama-bench.cpp:2278-2284`). Thus a multi-`-m` run checks
teardown and the next load; it does not test simultaneously loaded models or
contexts. Multiple model/context objects may remain alive, and their live weight
ownership must be preserved as described under **SYCL Memory Ownership** above.
The process-global mutex does not make submission or device execution globally
serial; the direct/fallback overlap and separate same-device limitation in the
canonical contract §5 still apply.

## 9. GPT-OSS Prompt Template Rule (as it stood)

Old location: `CLAUDE.md`, "GPT-OSS Prompt Template Rule". `CLAUDE.md` keeps the rule in a few lines.

Use `llama-cli` (the GPT-OSS gate above) so the CLI applies the model's
embedded GGUF/Jinja chat template. `llama-cli` is conversational by default, so
no `-cnv` is needed — and passing it fails outright; see the gate block above for
why. Do **not** pass `--chat-template gpt-oss` (it
selects the older native formatter) or hand-render a raw Harmony prompt. Always
pin `reasoning_effort=medium` via `--chat-template-kwargs` so template metadata,
CLI defaults, or harness changes can't move the prompt across regression
comparisons; `--reasoning-format none` is not a substitute for pinning. GPT-OSS
was trained for OpenAI's Harmony format — wrong formatting causes cascading
generation failures, and `llama-bench` proves throughput only, never chat
correctness. Full rationale and sources: `docs/backend/gpt-oss-testing.md`.
