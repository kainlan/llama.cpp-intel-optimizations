# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands (Intel SYCL)

**IMPORTANT**: Always source oneAPI before running the binaries (the build script handles sourcing for builds):
```bash
source /opt/intel/oneapi/setvars.sh --force
```

### Build
```bash
./scripts/sycl-build.sh
```

The script sources oneAPI, runs `cmake -G Ninja` with all required flags
(`-DGGML_SYCL_F16=ON`, `-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON`,
`-DCMAKE_INSTALL_RPATH='$ORIGIN'`, ccache integration when available),
and invokes ninja. Output goes to `build/bin/`. The `$ORIGIN` install
RPATH lets the binaries find their colocated `lib*.so.0` without
setting `LD_LIBRARY_PATH`.

Common flags:
```bash
./scripts/sycl-build.sh                           # incremental build
./scripts/sycl-build.sh llama-bench               # build a single target
./scripts/sycl-build.sh -r                        # force CMake reconfigure
./scripts/sycl-build.sh -c                        # clean (rm build/) and rebuild
```

**Build time**: ~10 minutes with ccache, ~25 minutes without.

**`-DGGML_SYCL_F16=ON`** enables 16-bit float arithmetic throughout the SYCL backend:
- **dmmv dequant** (active today): the `dfloat`/`dfloat2` typedef in `ggml/src/ggml-sycl/common.hpp` pivots to `sycl::half`/`sycl::half2` under the flag.
- **Attention path**: gates the Q f16 cast and `afloat` accumulator typedef in `fattn-xmx-f16-v2.hpp`; unlocks oneDNN SDPA eligibility on Mistral.

Precision tradeoff: ~4 mantissa bits vs f32. Declare OFF only for precision-sensitive models (phi-2 per the `GGML_PREC_F32` comment in `ggml/include/ggml.h`, or similar).

### Ninja vs Make
The script always uses Ninja (`-G Ninja`). Reasons:
- **Correct header dependency tracking**: Changes to `.hpp` files reliably trigger recompilation
- **Faster no-op builds**: 1.5s vs 73s for Make on large projects

If a `build/` was created with a different generator, run `./scripts/sycl-build.sh -c` to wipe and reconfigure.

### Running Tests

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

# 2. Full suite. `-j 1` is NOT a typo and NOT negotiable -- see below.
#    Check `uptime` first; never on a loaded machine.
ctest --test-dir build --output-on-failure -j 1 \
      -LE 'residency|mem-handle|cache' -E '^test-backend-ops$'

# 3. The excluded family, serially, with monitoring. Manually only -- never in
#    a subagent or background task.
ctest --test-dir build -L residency --output-on-failure -j 1

# Run a single test by name
ctest --test-dir build -R <test-name> -V
```

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

⚠️ **`-LE 'residency|mem-handle|cache'` does NOT exclude `test-backend-ops`** —
which is why form 2 above must also carry `-E '^test-backend-ops$'`. Until
2026-07-30 it did not, so the command this file prescribed as the *safe* one ran
the single binary this file separately forbids running unattended (see Hard-Won
Rules: 50–224 GB of TTM shmem, two OOM kills). At the `-j 4` this file then
prescribed, it would have started alongside three other tests.

The cause is structural, not a typo: `tests/CMakeLists.txt:494` registers it as
a bare `llama_build_and_test(test-backend-ops.cpp)` with **no labels**, so it
inherits only the default `main` and no label denylist can reach it. Verify
rather than assume — a label filter is silently permissive toward anything
nobody remembered to tag, so it fails *open*:

```bash
# Always confirm what a filtered sweep will actually run before running it.
ctest --test-dir build -N -LE 'residency|mem-handle|cache' | grep backend-ops
# ^ must print NOTHING once -E is added; if it prints a line, do not run the sweep.
```

Adding `LABELS "cache"` to that registration would also fix it, but it is
upstream code that a rebase would silently revert, and the failure mode of
losing the fix is an OOM. Excluding by name is the safer belt.

⚠️ **`test-backend-ops` is not the only unlabelled member of that family.**
`test-llama-archs` also carries only `main`, and **looping it exhausts host
memory the same way**. Measured 2026-07-30: two separate global OOMs
(12:00:25 and 12:49:57), each during a 6-run loop of

```bash
ctest --test-dir build -R '^test-llama-archs$' --output-on-failure   # ~36 s, fine ONCE
```

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

Pure-Python gates (`test-sycl-gap-causes`,
`test-sycl-timeline-gap-class-conservation`, `test-jinja-py`) allocate nothing
and are always safe at any parallelism.

After any OOM or forced stop, **check the GPU before trusting a benchmark** —
`journalctl -k --since "1 hour ago" --no-pager | grep -iE 'GT reset|guc_id|CAT error'`.

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
  `ctest` sweep — one at a time, sampling `Shmem`/`MemAvailable` before and ~5 s
  after, never overlapping two.

This costs nothing in throughput. GPU verification is *already* serial — the
memory does not fit two model-loading tests concurrently — so making the serial
point explicit removes the coordination failure without removing parallelism that
ever existed. What parallelises is the code work, and that stays parallel.

### Code Formatting

⚠️ **`git clang-format` does not exist on this machine** — it exits 1. Git resolves
`git <sub>` by looking for `git-<sub>` in `PATH`, and only the *versioned*
`git-clang-format-19` is installed (`/usr/bin/git-clang-format-19`); there is no
unversioned `git-clang-format`. This file recommended the broken form as
**preferred** until 2026-07-30, so the one step labelled preferred was the one that
failed. Use either spelling below — both verified working:

```bash
# Preferred: format only staged changes (uses .clang-format)
git-clang-format-19          # direct
git clang-format-19          # identical; git dispatches to the same binary

# Check without writing (operates on the staged tree)
git-clang-format-19 --diff --staged   # "clang-format did not modify any files" = clean

# Format specific files
clang-format-19 -i <file.cpp>
clang-format-19 --dry-run -Werror <file.cpp>  # dry-run check
```

⚠️ **`clang-format-19 -i` on a whole file is actively destructive here, which is the
real reason the staged-only form is "preferred".** These files carry a lot of
pre-existing drift, so `-i` reformats code nobody touched: measured 2026-07-31, a
single `-i` on `ggml/src/ggml-sycl/common.hpp` rewrote **~180 lines** of unrelated
code — ternary realignment, lambda wrapping, struct-field alignment — none of it near
the actual edit.

Two consequences, and the second is worse:

- The diff stops being reviewable. 180 lines of noise around a 5-line change means a
  reviewer either reads all of it or trusts none of it.
- **In a shared checkout it is a collision generator.** Those 180 lines overlap
  whatever anyone else is editing in the same file, and the conflict does not present
  as a merge conflict — it presents as someone else's work silently reformatted
  underneath them.

If you have already run it, the recovery is what worked: revert the file to HEAD, redo
the edit by hand, and use `git-clang-format-19 --staged` (changed lines only). Do not
try to hand-prune a 180-line reformat.

`clang-format-19 --dry-run -Werror <file>` is safe and useful — it reports without
writing. Just do not act on its whole-file findings as part of an unrelated change.

## Project Architecture

### Core Directories
- **`src/`**: Main llama library (`llama.cpp`, `llama-*.cpp`)
- **`include/llama.h`**: Public C API header (~2000 lines)
- **`ggml/`**: Core tensor library (vendored ggml framework)
- **`common/`**: Shared utility code for examples
- **`examples/`** and **`tools/`**: 40+ CLI tools
- **`tests/`**: CTest integration

### Key Binaries (in `build/bin/`)
- **`llama-cli`**: Interactive chat/inference
- **`llama-completion`**: Non-interactive text completion (use for scripted tests)
- **`llama-server`**: OpenAI-compatible HTTP server
- **`llama-bench`**: Performance benchmarking
- **`llama-quantize`**: Model quantization
- **`llama-perplexity`**: Model evaluation (perplexity measurement)

### Backend Structure (`ggml/src/`)
- **`ggml-cpu/`**: CPU backend (AVX/NEON/RVV)
- **`ggml-cuda/`**: NVIDIA CUDA kernels
- **`ggml-metal/`**: Apple Metal shaders
- **`ggml-sycl/`**: Intel SYCL backend (see SYCL Backend Structure below)
- **`ggml-vulkan/`**: Vulkan compute shaders

### SYCL Backend Structure (`ggml/src/ggml-sycl/`)
Key files in the SYCL backend (largest first; use codescout `overview` for a
current outline rather than relying on line counts, which drift constantly):
- **`ggml-sycl.cpp`**: Main backend — graph_compute, mul_mat dispatch, buffer ops, graph replay (by far the largest file)
- **`mmvq.cpp`**: Matrix-vector quantized kernels (batch=1 TG fast-path)
- **`unified-cache.cpp/hpp`**: Tiered weight cache and SYCL memory allocator
- **`unified-kernel.cpp/hpp`**: Unified MUL_MAT kernel with XMX/ESIMD/MMVQ dispatch
- **`mmq.cpp`**: Matrix-matrix quantized kernels (persistent TG, streaming PP)
- **`common.hpp/cpp`**: Shared types, `extra_gpu` struct, layout_policy, device management
- **`fattn.cpp`**: Flash attention implementation
- **`mem-handle.cpp/hpp`**: Ref-counted allocation and cache-entry handles
- **`dispatch.hpp`**: Kernel dispatch policy and routing
- **`quants.hpp`**: SOA block offset calculations for quantized types

### Inference Flow
1. **Model loading** (`llama_model_load`): Reads GGUF file, maps weights to tensors
2. **Context creation** (`llama_init_from_model`): Allocates KV cache, scratch buffers
3. **Tokenization** (`llama_tokenize`): Text to token IDs
4. **Graph building** (`llama_build_graph`): Creates ggml computation graph per batch
5. **Graph execution** (`ggml_backend_graph_compute`): Dispatches to CPU/GPU backends
6. **Sampling** (`llama_sampler_sample`): Token selection from logits

### Weight Caching (GPU Backends)
GPU backends cache weights on-device for repeated inference:
- **CUDA**: `ggml_cuda_pool` with per-device allocation tracking
- **SYCL**: `unified_cache` with tiered memory (device VRAM, pinned host, mmap). Supports SOA (Structure-of-Arrays) layout for coalesced GPU memory access, oneDNN packed layouts, and LRU eviction.
- Weights are identified by tensor name hash + model ID for cache keys

### SYCL Memory Ownership

> **Full design & rationale:** `docs/backend/sycl-memory-design.md` (narrative —
> the three primitives, the single `unified_allocate` entry point, the
> allocation flow) and `docs/design/sycl-canonical-memory-architecture.md` (the
> in-force enforceable contract with allocator/pointer allowlists). This is the
> key design constraint of the fork — read those before touching allocation,
> dispatch, or eviction code. The rules below are the short form.

The unified cache is the memory allocator for the SYCL backend. All SYCL
backend GPU, host-pinned, staging, scratch, graph-temporary, KV, oneDNN, and
weight-layout allocations must flow through the unified-cache allocation APIs
(`unified_alloc`, `unified_allocate`, cache materialization helpers, or wrappers
that return `mem_handle`). Do not introduce direct `sycl::malloc_device`,
`sycl::malloc_host`, `sycl::free`, raw TLSF allocation, or side caches outside
the unified-cache implementation. Any low-level allocation implementation detail
must remain inside unified-cache code and surface to the rest of the backend as
a `mem_handle`.

`mem_handle` is the ownership and lifetime token. Code that uses an allocation
must hold a `mem_handle` (or an object that owns one) until the CPU thread,
SYCL event, command graph, or pointer table is finished with that allocation.
When the last handle/reference is released, the allocation is freed through the
unified cache (`unified_free`, `zone_free`, cache-entry lease release, etc.).
Do not add forced eviction, forced reap, or zone-reset logic to reclaim memory
that still has a live handle; a live allocation at cleanup means a leaked
reference or stale owner that must be fixed.

⚠️ **"At cleanup" is load-bearing, and was being read too broadly.** A live lease
is a defect only when its **owner is gone**. Several `llama_model` objects may be
loaded at once — that is supported public API, and `tests/test-thread-safety.cpp`
exists to exercise it (it loads one model per GPU plus a CPU copy, then runs them
concurrently). So **another live model's lease is correct, not leaked**, and a
*new model's load* is not a quiescent point for anybody else's weights.

Getting this backwards cost real time. `9a0670712` ("sycl: checkpoint unified
memory ownership work") replaced `reset_model_weight_entries`'s preserve-and-
continue with `GGML_ABORT`, plus the same in `host_zone_reset` and `zone_reset`.
That made `test-llama-archs` and `test-thread-safety` — both `main`-labelled —
fail deterministically, and each abort masked whatever came after it, so nobody
connected the failures back. Restored in `acdb192d4`; the zone-reset siblings are
tracked in `llama.cpp-fz2u`.

Two things this does **not** license:

- **It is a scoped exception, not a general loosening.** `mem_handle`'s destructor
  is still the sole release point, and reclaiming memory that still has a live
  handle is still forbidden. `acdb192d4` *refuses to reclaim* — the opposite of a
  forced reap.
- **It costs leak detection at that site, knowingly.** That scan can no longer
  distinguish "another live model owns this" from "something leaked it"; both now
  warn and preserve. A future real leak will surface as a growing
  `entries_preserved` count and eventual VRAM pressure, not a loud abort. Scoping
  the reset per model would have kept both properties but is structurally
  impossible: `unified-cache-key.hpp` deliberately excludes `model_id` from
  `cache_id_equal` for GGUF weights, and the primary call site
  (`ggml_backend_sycl_set_model_loading`) runs before any tensor of the incoming
  model exists. See `llama.cpp-ljb9`.

Raw pointers are not ownership tokens and must not model allocation state. They
are only transient ABI views resolved from `mem_handle` for immediate kernel
submission, oneDNN primitive calls, or tightly scoped CPU access. Do not store
raw pointers as the source of truth, use pointer addresses as cache keys, or
let pointers outlive their owning handle. Pointer tables and dispatch caches
must be derived from the stable identity/hash carried by `mem_handle`, not from
raw device addresses; if a table contains raw device pointers for a kernel ABI,
retain the corresponding handles for at least the lifetime of the queued work
or executable graph.

Weight cache entries are ref-counted through `mem_handle` leases. Eviction may
only remove entries whose in-use count is zero. If the cache cannot evict
because handles are still referenced, fix the missing release instead of
forcing eviction.

## ggml Conventions

### Matrix Multiplication
Matrix multiplication is **unconventional**: `C = ggml_mul_mat(ctx, A, B)` computes:
```
C^T = A * B^T  <=>  C = B * A^T
```

### Tensor Storage
- Tensors store data in **row-major order**
- Dimension 0 = columns, Dimension 1 = rows, Dimension 2 = matrices

### Naming Patterns
- Use `snake_case` for function, variable, and type names
- Optimize for **longest common prefix**: `number_small`, `number_big` (not `small_number`, `big_number`)
- General pattern: `<class>_<method>` with `<method>` being `<action>_<noun>`
  ```cpp
  llama_model_init();           // class: "llama_model", method: "init"
  llama_sampler_get_seed();     // class: "llama_sampler", method: "get_seed"
  ```
- The `get` action can be omitted; `_context` class suffix is optional
- Use `init`/`free` for constructor/destructor actions

### Enum Values
Enum values are always UPPER_CASE and prefixed with the enum name:
```cpp
enum llama_vocab_type {
    LLAMA_VOCAB_TYPE_NONE = 0,
    LLAMA_VOCAB_TYPE_SPM  = 1,
    LLAMA_VOCAB_TYPE_BPE  = 2,
};
```

### Struct Declarations
Use `struct foo {}` not `typedef struct foo {} foo`. Omit optional `struct`/`enum` keywords in C++ code:
```cpp
llama_context * ctx;              // OK
struct llama_context * ctx;       // not OK
const llama_rope_type rope_type;  // OK (no enum keyword)
```

## Coding Guidelines

- **Minimal dependencies**: Avoid adding third-party dependencies
- **Cross-platform**: Test on Linux, macOS, Windows when possible
- **Simple STL**: Avoid fancy modern STL, use basic `for` loops, minimize templates
- **Vertical alignment**: Makes code more readable and easier to batch edit
- **Formatting**: 4 spaces, brackets on same line, `void * ptr`, `int & a`
- **Public API types**: Use `int32_t` etc., `size_t` for allocation sizes
- **File naming**: C/C++ lowercase with dashes (e.g., `unified-kernel.cpp`), Python lowercase with underscores

## Hard-Won Rules (Workflow, Safety, Architecture)

Confirmed lessons from prior work on this fork. Treat them as defaults.

### Communication & Workflow
- **The user reads Discord, not the terminal.** CLI output is invisible to them. Any question, confirmation, decision prompt, or status update intended for the user MUST go through the Discord reply tool (the harness supplies the channel id each session). Terminal text is logging only — never "await a reply" there.
- **Work on the active feature branch** (`git branch --show-current` — do not trust a branch name written down here). When reviewing diffs, bound by BASE_SHA/HEAD_SHA, not "everything on the branch."
- **Worktrees are allowed and are the right tool for build-heavy parallel work** (owner decision, 2026-08-01). This entry previously said *"skip git worktrees — a worktree forces a fresh `build/` and loses the ~10-min ccache-warm hit rate."* The cost is real but was overstated: **ccache is global (`~/.ccache`), not per-tree**, so a worktree build still gets its hits; only the `build/` object tree and CMake cache are fresh.
  Measured 2026-08-01: with one shared `build/`, four agents serialised on a ~14-min build cycle and one track was starved **~2 hours**. Two worktree builds completed fine in the same session. Path-scoped commits prevent *file* conflicts; they do nothing about *build* contention, and that is what actually costs time.
  Use a worktree when a track will build repeatedly. Stay in the shared checkout when the work is small, or when it must operate on the checked-out branch itself (git refuses to check out one branch in two trees — so a **merge into** the active branch must happen in the main checkout, though only the build needs isolating).
  ⚠️ **Lock scope follows the build directory, not the act of building.** Shared checkout → take `/Apps/llama.cpp/BUILD.lock`. Own worktree → take nothing; nothing contends. `GPU.lock` is always global, from any tree, because the *devices* are shared.
  ⚠️ **Delete an evidence worktree only after the finding it supports has been reviewed.** ~1.4 GB is cheap next to a disagreement about what was measured that can no longer be settled (happened 2026-08-01).
- **`git merge-base HEAD master` is usually the WRONG before-point for a measurement.** On a long-lived feature branch it can sit dozens of commits behind the work you are attributing, silently crediting your change with everything in between. Use the commit immediately before your own first commit.
- **Fix-forward, never revert.** If a build or correctness test fails mid-implementation, diagnose and fix in a new commit. Don't `git revert` or `git checkout --` to undo progress.
- **Verify correctness before claiming any perf win.** `llama-bench` measures tok/s only — a change can boost throughput by silently skipping or mis-staging work and still emit garbage tokens. Before committing any change to kernel dispatch, weight staging, graph replay, or allocation routing, run the canonical Mistral completion gate (see "Verification Commands & Correctness Gates") and confirm the output. A fake +19.6% PP "win" shipped this way once and had to be reverted.

### Safety (these have hung or exhausted memory on this host)
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
  SIGSEGVs — that is `llama.cpp-oze0`, unrelated and unchanged by the selector.)

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
- **Always `timeout 60` GPT-OSS 20B test runs.** The historical host-MoE-routing hang (GuC `guc_id=6`, unrecoverable, requires reboot) was closed by commit `ec7f04ac4`, but keep the timeout as a guard. Distinguish the userspace hang (`guc_id=6`, attributed `in <llama-bench>`, unrecoverable) from the benign environmental XE timeout (`guc_id=0`, `in no process [-1]`, auto-recovers).
- **Benchmark numbers are invalid after any crash or forced stop on that card** (xe GT reset cascades) — check the kernel log first. `SAFE_MODE`/op-timing diagnostics can themselves stall cards.
- **`dmesg` is privilege-denied for this user** (`read kernel buffer failed: Operation not permitted`). Every "check dmesg" instruction in this repo's docs must be run as:
  ```bash
  journalctl -k --since "1 hour ago" --no-pager | grep -iE 'GT reset|guc_id|GPU hang|xe.*reset'
  ```
  which works unprivileged. A silent `dmesg` failure looks identical to a clean log.
- **Check for competing load before any throughput measurement.** `uptime` plus `pgrep -af 'codescout|ninja|icpx|ffmpeg'`. A codescout re-index has been observed holding 600–1430% CPU for hours; Frigate ffmpeg adds ~250%. Interleaved paired A/Bs survive sustained load, but **absolute** numbers taken under it are depressed and must not become baselines.

### Tooling
- **codescout's index is BLIND inside `ggml/src/ggml-sycl/ggml-sycl.cpp`** (~60k lines). `search_text` and `find_references` return matches tagged `source: index` and silently omit real occurrences in that file — despite `search_text` being documented as an exhaustive live grep. This is not theoretical: an entire implementation plan was written on the false premise that a field had no other writers, because the one call site that mattered was missed. It cost a task to discover empirically. **In that file, verify with `cat ggml/src/ggml-sycl/ggml-sycl.cpp | grep -n '<pattern>'`** — a downstream pipe grep is permitted by the search hook; only command-position in-repo search is redirected. Never conclude "no other uses" there from codescout alone.
- **`/tmp` is tmpfs and does not survive a reboot.** Anything a later step depends on belongs in the committed artifact (a findings doc, a commit message), not in `/tmp`. A mid-session reboot has already destroyed capture artifacts here. An empty or missing artifact directory means *not verified*, never "nothing observed" — a `grep` over it passes vacuously.

### Architecture
- **The unified cache owns all GPU/host memory** (decision Feb 9, 2026). Weight placement, eviction (device→pinned host→mmap), and budget tracking all flow through it.
- **Use smart handles, never hold a raw `void*` from the cache.** A raw VRAM pointer becomes dangling the moment the cache evicts to host → DEVICE_LOST errors or corrupted results. Handles must resolve location on dereference so the cache can move data between tiers transparently.
- **Host-resident weights → CPU dispatch, not GPU PCIe "zero-copy."** Measured CPU AOS = 18–30 GB/s vs GPU zero-copy = 11.3 GB/s (1.6–2.6x slower). Parallelize CPU work with GPU via `sycl::depends_on` (~9.7 µs cross-device latency). Never feed a host-pinned pointer to a GPU kernel as "zero-copy."
- **The VRAM budget calc is correct by design for DISCRETE cards** (`min(total*pct, free_at_init)`). Low free VRAM is a system problem (other GPUs active, driver overhead), not an app bug to "fix" by ignoring free VRAM — fix the root cause at the system level.
  ⚠️ **It is catastrophically wrong for an INTEGRATED GPU, and that is the cause of this host's OOM history** (`llama.cpp-403s`, measured 2026-08-01). The Arrow Lake-S iGPU reports `global_mem_size` = **231.7 GB** — 94 % of the host's 246.9 GB — because for an integrated GPU "VRAM" *is* system RAM. `ggml-sycl.cpp:10043-10049` feeds that into the same budget path as a discrete card at a **default of 100 %**, and neither `ggml-sycl.cpp` nor `unified-cache.cpp` contains a single occurrence of `host_unified` or `is_integrated`. So the backend claims the machine.
  Isolated with one variable — same 19 MB model, same single-threaded `llama-completion`, only the selector changed: `level_zero:0` → peak `Shmem` **2.4 GB**; `level_zero:0,1` → **2.4 GB**; selector unset (adds the iGPU) → **127.8 GB**.
  For an integrated device, "free VRAM" and "free host RAM" are the same pool, so taking 100 % of it is not a conservative reading of available memory — it is claiming memory the host needs. `sycl::info::device::host_unified_memory` is queryable and `caps.global_mem_size` is already captured at `common.cpp:716`; the information is present and simply never consulted.
- **Small-block dequant (Q4_0/Q8_0/Q4_K) belongs on standard SYCL, not ESIMD.** ESIMD measured 1.9x SLOWER on Arc B580 + oneAPI 2025.3 (block granularity too small to amortize LSC loads). The real dequant lever is structural — fuse dequant into the matmul. Opt-in retest hatch: `GGML_SYCL_ESIMD_DEQUANT=1`.

> Live debugging state (active bug investigations, bisect results, perf-regression hunts) lives in the **codescout task tracker** (`task_ready`, `task_list`), not here. This section is for settled rules only.

## Development Workflow (Machine-Specific)

### Model Locations
Models are stored in `/Storage/GenAI/models/`:

**Mistral 7B variants** (standard benchmark model):
- `mistral-7b-v0.1.Q4_0.gguf` (3.9G) - **Default for benchmarks**
- `mistral-7b-v0.1.Q4_K_M.gguf` (4.1G) - Good quality/size balance
- `mistral-7b-v0.1.Q8_0.gguf` (7.2G) - Highest quality
- Other variants: Q2_K (2.9G), Q3_K_S/M/L, Q4_K_S, Q5_0/K_S/K_M, Q6_K

**GPT-OSS models** (large MoE, native MXFP4):
- `gpt-oss-20b-mxfp4.gguf` (12G) - Smaller variant
- `gpt-oss-120b-mxfp4-*.gguf` (60G total, 3-part split) - Full model

### Verification Commands & Correctness Gates

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
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0

# GPT-OSS B50 chat correctness gate. With --no-display-prompt the prompt echo
# lands on the interactive "> " line and the model's ANSWER is the next line,
# on its own:
#   > Count from 1 to 5. Answer with only: 1, 2, 3, 4, 5
#   1, 2, 3, 4, 5
# The gate is the digit sequence. (An older note here said the output starts
# ": 1, 2, 3, 4, 5" — that colon was the tail of the echoed PROMPT, which itself
# ends "...only: 1, 2, 3, 4, 5", captured in a pre-`--no-display-prompt` form.
# Grepping for the colon form makes a passing gate read as a failure.)
# Use the GGUF tokenizer.chat_template metadata. Do not force
# `--chat-template gpt-oss`; that selects the older native formatter.
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-cli \
  -m /Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf -ngl 99 \
  -cnv -st --simple-io --no-display-prompt \
  --chat-template-kwargs '{"reasoning_effort":"medium"}' \
  --reasoning-format none --reasoning-budget 0 \
  -p 'Count from 1 to 5. Answer with only: 1, 2, 3, 4, 5' \
  -n 48 --seed 42 --temp 0

# Benchmark prompt processing (PP) and token generation (TG)
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128

# Test backend operations (after modifying ggml operators)
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-backend-ops
```

#### `llama-bench` traps (both cost real time)

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
  -m /Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf -p 0 -n 4 -r 1 -v 2>&1 | grep ...
```

Note `-p 0 -n 4` triggers planning but does **no prompt processing**, so paths
reached only during PP will show zero activity. Use `-p 512` when you need them.

**2. ONE model per process.** Passing two `-m` flags to a single `llama-bench`
aborts at the model switch on a leaked model-weight `mem_handle` lease
(`unified-cache.cpp`, `reset_model_weight_entries: leaked model-weight
mem_handle lease`). Loop the shell, don't loop the flag.

### GPT-OSS Prompt Template Rule

Use `llama-cli -cnv` (the GPT-OSS gate above) so the CLI applies the model's
embedded GGUF/Jinja chat template. Do **not** pass `--chat-template gpt-oss` (it
selects the older native formatter) or hand-render a raw Harmony prompt. Always
pin `reasoning_effort=medium` via `--chat-template-kwargs` so template metadata,
CLI defaults, or harness changes can't move the prompt across regression
comparisons; `--reasoning-format none` is not a substitute for pinning. GPT-OSS
was trained for OpenAI's Harmony format — wrong formatting causes cascading
generation failures, and `llama-bench` proves throughput only, never chat
correctness. Full rationale and sources: `docs/backend/gpt-oss-testing.md`.

### Patched compute-runtime & P2P topology

The system `libze_intel_gpu.so.1` is a patched 26.22/BMG-only build (from
`/Apps/compute-runtime-26.22-llama`, branch `llama/26.22-cross-device`) carrying
the hung-i915 discovery fix, cross-device in-order dependency fixes, and the
PR 930 USM compression fix. Stock `1.14.37020` is preserved alongside for
rollback. Reverting to stock without restoring the old allocation check can
reintroduce silent oversized-allocation hangs (the m09zb `event.wait()` hang).

**Durable rule — there is NO direct P2P between the two discrete cards.
Re-verified on the B70 on 2026-07-31; the earlier "not re-tested" hedge is retired.**

Keep direct peer-copy / shared-context paths disabled. Host-bounce
(`level_zero:0,1`) validation may continue.

The restriction is **PCI topology, not a property of either card**, which is why
swapping the B580 for the B70 in the same slot changed nothing:

```
$ lspci -tv
+-06.0-[01-04]--...--[03]----00.0  Battlemage G31  <- B70, 0000:03:00.0
+-06.3-[05-08]--...--[07]----00.0  Battlemage G21  <- B50, 0000:07:00.0
```

Different CPU root ports (`00:06.0` vs `00:06.3`), no shared PCIe switch — they meet
only at the root complex. The kernel says so outright:

```
xe 0000:07:00.0: cannot be used for peer-to-peer DMA as the client and provider
(0000:03:00.0) do not share an upstream bridge or whitelisted host bridge
```

Measured behaviour, identical on B580 (historical) and B70 (2026-07-31): a 256 KiB
direct device-to-device USM copy fails **both directions** with
`UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` (error 39), on a card with 31.89 GiB free.
`can_access_peer` returns **false** both directions for both `access_supported` and
`atomics_supported`.

⚠️ **`ext_oneapi_enable_peer_access()` returns OK on hardware that has no P2P.** No
throw, no warning, both directions. Code that treats a successful `enable_peer_access`
as proof P2P is available will conclude the exact opposite of the truth, then fail at
the first copy with a *memory* error that sends you hunting for a VRAM budget bug.
**`can_access_peer` is the honest query; `enable_peer_access` is not a capability
check.** This is why `OUT_OF_DEVICE_MEMORY` above is so misleading — it is a P2P
refusal wearing a memory error's name.

Probe and log: `p2p-probe.cpp` / `nguy-p2p-run1.log` (session scratchpad; not a
committed artifact — lift anything load-bearing into a ticket). Caveat stated by the
measurer: separate per-device contexts were used, mirroring the backend's isolated
per-device queues and the historical B580 test. The single-context variant was
deliberately **not** run, because that is the multi-GPU Level Zero context this file
records as triggering DEVICE_LOST on compute-runtime 26.x. The verdict does not rest
on context scoping — `can_access_peer` is a device-level query and the kernel refusal
is a fact about the two BDFs.

**Consequence for multi-GPU work:** `GGML_SYCL_MOE_MULTI_GPU` must stay opt-in. The
MoE multi-device path would be moving expert data between two cards that cannot DMA to
each other, so any traffic host-bounces — which is also why an earlier two-GPU run
halving throughput (32.11 → 15.81 tok/s) reads as expected rather than anomalous.

Install history, rollback commands, and loader-path notes:
`docs/backend/compute-runtime.md`.

### SYCL Device Selection

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
| `level_zero:0` | Arc Pro B70 (G31) | `0000:03:00.0` | `renderD129` | `card0` |
| `level_zero:1` | Arc Pro B50 (G21) | `0000:07:00.0` | `renderD130` | `card2` |
| — | Arrow Lake-S iGPU | `0000:00:02.0` | `renderD128` | `card1` |

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
# then name each PCI id:  lspci -s 03:00.0   ->  Battlemage G31 = B70
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

### Performance Expectations

Full tables, run counts and spreads live in `docs/backend/sycl-perf-baselines.md`.
**Gate against that document, not this one.** Orientation figures, `-p 512 -n 128`,
driver 26.27:

| card | model | PP512 | TG128 |
|------|-------|------:|------:|
| Arc Pro B70 (`level_zero:0`) | GPT-OSS 20B MXFP4 | ~1415 | ~44 |
| Arc Pro B70 (`level_zero:0`) | Mistral 7B Q4_0 | ~2495 | ~108 |
| Arc Pro B50 (`level_zero:1`) | GPT-OSS 20B MXFP4 | ~894 | ~32 |
| Arc Pro B50 (`level_zero:1`) | Mistral 7B Q4_0 | ~1188 | ~47 |

Re-measured 2026-07-25 at `79ae63559` (machine under load): B70 1397.55/47.79 and
2513.76/108.94; B50 902.26/36.55 and 1200.21/46.94. Seven of eight at or above
baseline, so the table above is current.

**B70 tg128 is the noisy axis** — cv 3.3% over 21 runs, range 40.18–46.27. Ignore
B70 tg differences below ~10% between single runs. The B50 is steady (cv 0.7% tg,
0.3% pp), so a B50 move of a few percent is real.

Any **B580** figure in older notes is history — that card was replaced by the B70.
Measuring a B70 against a B580 target makes a healthy run look catastrophic.

⚠️ **`GGML_SYCL_FA_ONEDNN_ALLOW` does not exist, and the warning that used to
stand here was wrong in both directions.** It read: *"Do not use
`GGML_SYCL_FA_ONEDNN_ALLOW=1` to restore Mistral PP numbers — it can raise PP
throughput, but the deterministic completion gate produces incorrect output with
the current nc!=D contiguity fast-path."* Corrected 2026-07-30. Every clause of
that is now false:

1. **No such variable.** Commit `3c8f296fd` (2026-05-15) *removed* the
   `getenv("GGML_SYCL_FA_ONEDNN_ALLOW")` bypass **and added that warning in the
   same commit** — it documented a footgun it had just deleted. Setting it today
   is a no-op; verified, the Mistral gate emits the identical correct
   `1, 2, 3, 4, 5, 6, 7, 8, 9, 10` with and without it. The name survives only in
   a stale comment at `fattn.cpp` and here.
2. **oneDNN SDPA is not retired for Mistral — it is ON by default and Mistral
   uses it.** `16a241dd1` (2026-05-30) added the `MATERIALIZE_REQUIRED` path, so
   nc≠D GQA is no longer rejected: the planner
   (`fattn-onednn.cpp:140`) asks a unified-cache-backed materializer for dense
   f16 K/V, and `fattn.cpp:2618` accepts that plan alongside `DIRECT`. Confirmed
   live with `GGML_SYCL_FA_DISPATCH_DEBUG=1` — 32 dispatches per generation:
   `oneDNN MATERIALIZED D=128 ne01=16 ne11=256 H_q=32 H_kv=8` (H_q≠H_kv **is**
   the GQA shape the old note claimed corrupts).
3. **There is no correctness penalty, and disabling it is expensive.**
   Interleaved paired A/B, Mistral Q4_0 pp512 on the B50:

   | pair | default (oneDNN on) | `GGML_SYCL_FA_ONEDNN=0` |
   |------|--------------------:|------------------------:|
   | 1 | 1142.76 | 695.31 |
   | 2 | 1145.14 | 697.12 |
   | 3 | 1113.92 | 697.23 |

   **Turning oneDNN FA off costs ~39% of PP512** (1.63x slower). Taken under
   load 38 — absolute values are depressed and are not baselines, but the
   interleaved pairing makes the *ratio* sound, and the spreads are tight
   (off: cv 0.15%).

The real variable is **`GGML_SYCL_FA_ONEDNN`** (default ON; `=0` disables).
Leave it alone unless you are deliberately bisecting the attention path.

### Regression Baselines (hard guardrails)

Do not accept lower post-merge/post-debug numbers as new baselines (codescout
tasks `llama.cpp-aqzz3.1`, `llama.cpp-po3nd.2.45/.46`, `llama.cpp-ix58x`):

**Gate against the rows in `docs/backend/sycl-perf-baselines.md`** (reproduced in
Performance Expectations above), never against a number remembered from an older
card or an older driver. Allow the stated spread: B70 tg is noisy (±10% between
single runs means nothing), the B50 is steady.

- **B50 GPT-OSS 20B MXFP4 FA-on:** ~894 PP512 / ~32 TG128, count gate passing.
- **B70 GPT-OSS 20B MXFP4 FA-on:** ~1415 PP512 / ~44 TG128, count gate passing.
- **B50 / B70 Mistral 7B Q4_0:** ~1188 / ~2495 PP512, ~47 / ~108 TG128.

⚠️ **A `≥1100 PP512, ~50+ TG128` B50 GPT-OSS guardrail appeared here until
2026-07-25 and was wrong** — it predates the 26.27 driver. Against it a healthy
B50 (~894–902 PP512) reads as an ~18% catastrophe. That stale figure triggered
three separate false-regression scares in one session. If you find it quoted
anywhere else, it is wrong there too.

- ~~**B580 Mistral 7B Q4_0 FA-on:** >2000 PP512, >85 TG128~~ — **SUPERSEDED, not
  a gate.** The B580 was replaced by the Arc Pro B70 and is no longer installed.
  The figure `docs/backend/SYCL.md` records (`5b206c499-dirty`, 2173.92 PP512 /
  88.42 TG128) remains valid history *for that card only*. **Do not gate on it.**

Keep these opt-in until same-build B50 GPT-OSS + B70 Mistral gates pass on a
clean boot: `GGML_SYCL_MOE_BLOCK_GRAPHLETS`, `GGML_SYCL_XMX_MOE_PP` /
`GGML_SYCL_XMX_MOE_ALLOW_UNSAFE_PP`, `GGML_SYCL_PP_PIPELINE` (the last has shown
GPT-OSS chat correctness failures).

⚠️ **The stale guardrail above is also baked into the tracker**, where the doc
pass could not reach it. `llama.cpp-po3nd.2` (epic, 42 dependents) carries both
dead figures in its **acceptance criteria**, and a June "gate audit" propagated
them into every open child (`.2.13`, `.2.21`, `.2.27`, `.2.30`, `.2.38`, `.2.45`,
`.2.46`) plus `llama.cpp-ix58x` and `llama.cpp-aqzz3`. Both conditions are now
unreachable — B50 ≥1100 PP512 exceeds a healthy card, and the B580 gate names
hardware that is no longer installed — so those tasks cannot be closed as
written. See the 2026-07-25 comment on `llama.cpp-po3nd.2`. Re-read the gate from
`docs/backend/sycl-perf-baselines.md`, never from a task's acceptance criteria.

Active regression-hunt state (commit deltas, suspect changes) lives in the
codescout task tracker. Find it with `task_search` (pass `semantic=false` — the
semantic path times out on this store); there is no single standing task id.

### SYCL Environment Variables

Full catalog (dispatch tuning, persistent-TG, memory hierarchy, cache,
debugging — 240+ vars) is in **`docs/backend/sycl-env-vars.md`**. The
load-bearing performance opt-outs (all default ON — flip to disable) that you
most need to know:

| Variable | Default | Effect |
|----------|---------|--------|
| `GGML_SYCL_UNIFIED_SOA=0` | ON | Disable SOA memory layout (AOS fallback, ~4x slower TG) |
| `GGML_SYCL_TG_FAST=0` | ON | Disable MMVQ fast-path (slower TG) |
| `GGML_SYCL_DISABLE_GRAPH=1` | OFF | Disable SYCL graph replay (~3% TG, mainly helps PP) |
| `GGML_SYCL_ONEDNN_PP=0` | ON | Disable oneDNN for prompt processing |
| `GGML_SYCL_UNIFIED_FORCE_LEGACY=1` | OFF | Force legacy kernel dispatch (skip unified kernel) |

Common diagnostics: `GGML_SYCL_DEBUG=1` (verbose dispatch), `GGML_SYCL_NAN_CHECK=1`,
`GGML_SYCL_SAFE_MODE=1` (drain queue per op to localize faults),
`GGML_SYCL_OP_TIMEOUT_MS=<N>` (abort before the xe GT-reset cascade). To find any
var not documented: search `getenv("GGML_SYCL` under `ggml/src/ggml-sycl/`
(codescout `search_text`, or `grep -r`).

## CI and Validation

### Before Submitting PRs
1. Format code: `git-clang-format-19` (preferred — **not** `git clang-format`, which does
   not exist here; see "Code Formatting") or `clang-format-19 -i <files>`
2. Build: `./scripts/sycl-build.sh`
3. Test: use a form from "Running Tests" above — **not** a bare
   `ctest --test-dir build --output-on-failure`, which runs `test-backend-ops`
   and so contradicts step 4. Prefer `-R <what your change gates>`; for a full
   sweep use form 2 verbatim — `-j 1` **and** `-E '^test-backend-ops$'`. Do not
   raise `-j` to save time; it is a memory multiplier and has OOM'd this host.
4. For ggml changes: Run `test-backend-ops` on multiple backends — **manually only, never in a subagent/background task (memory-exhaustion hazard, see Hard-Won Rules)**
5. Verify correctness: run the canonical completion gate (Hard-Won Rules) — tokens must be right, not just fast
6. Verify performance: `llama-bench` and `llama-perplexity` should not regress

### Triggering Heavy CI
Add `ggml-ci` to commit message to trigger extended CI workloads.

## Documentation

- **Build Details**: `docs/build.md`
- **Backend SYCL**: `docs/backend/SYCL.md`
- **SYCL memory design (unified cache + mem_handle)**: `docs/backend/sycl-memory-design.md` — includes **Path-scoped zone sizing**: how arena zones are sized from a structural `(type, ne)` classifier rather than one global max, the rule for adding a consumer, and the two separate oneDNN sizing sites
- **SYCL canonical memory contract (enforceable)**: `docs/design/sycl-canonical-memory-architecture.md`
- **SYCL env-var catalog (fork tuning)**: `docs/backend/sycl-env-vars.md`
- **SYCL perf baselines (fork)**: `docs/backend/sycl-perf-baselines.md` — **the numeric gate; prefer it over any figure in this file**
- **GPT-OSS testing rationale**: `docs/backend/gpt-oss-testing.md`
- **Patched compute-runtime & P2P**: `docs/backend/compute-runtime.md`
- **Add New Model**: `docs/development/HOWTO-add-model.md`
- **Contributing**: `CONTRIBUTING.md` (coding/naming guidelines, PR process)
