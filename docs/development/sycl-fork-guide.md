# Fork development guide: build detail, repository map, formatting, workflow, tooling

`CLAUDE.md` holds the commands and rules every session needs. This page holds the detail that used
to sit beside them: what the build script does, where things are in the repository, why the
formatting rules are what they are, the measurements behind the worktree and lock rules, and two
tooling limits.

The text under each heading below was moved out of `CLAUDE.md` on 2026-11-16 and is unchanged,
apart from the corrections that are marked as such. In the moved text, "this file" means
`CLAUDE.md` as it stood before that date, and "above" or "below" refer to its old layout.
Line numbers quoted in the moved text were right on the date given beside them and have drifted
since. Find the code by the quoted text, not by the number.

## 1. Build detail

Old location: `CLAUDE.md`, "Build Commands (Intel SYCL)".

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

**Python interpreter**: the script passes `-DPython3_EXECUTABLE=$(command -v python3)`, so `ctest` runs the Python contracts with the shell's `python3` (the alloc-zone contract needs its `tree_sitter_language_pack`). A build directory whose cache holds another interpreter is reconfigured automatically.

**`-DGGML_SYCL_F16=ON`** enables 16-bit float arithmetic throughout the SYCL backend:
- **dmmv dequant** (active today): the `dfloat`/`dfloat2` typedef in `ggml/src/ggml-sycl/common.hpp` pivots to `sycl::half`/`sycl::half2` under the flag.
- **Attention path**: gates the Q f16 cast and `afloat` accumulator typedef in `fattn-xmx-f16-v2.hpp`; unlocks oneDNN SDPA eligibility on Mistral.

Precision tradeoff: ~4 mantissa bits vs f32. Declare OFF only for precision-sensitive models (phi-2 per the `GGML_PREC_F32` comment in `ggml/include/ggml.h`, or similar).

### Ninja vs Make
The script always uses Ninja (`-G Ninja`). Reasons:
- **Correct header dependency tracking**: Changes to `.hpp` files reliably trigger recompilation
- **Faster no-op builds**: 1.5s vs 73s for Make on large projects

If a `build/` was created with a different generator, run `./scripts/sycl-build.sh -c` to wipe and reconfigure.

## 2. Code formatting

Old location: `CLAUDE.md`, "Code Formatting". The three rules are in `CLAUDE.md`. This is the
reasoning and the two incidents behind them.

One edit was made to the moved text: a link into a private memory directory
(`[[git-add-is-not-scoping]]`) was replaced by the fact it named, because a fresh clone does not
have that directory.

⚠️ **`git clang-format` does not exist on this machine** — it exits 1. Git resolves
`git <sub>` by looking for `git-<sub>` in `PATH`, and only the *versioned*
`git-clang-format-19` is installed (`/usr/bin/git-clang-format-19`); there is no
unversioned `git-clang-format`. This file recommended the broken form as
**preferred** until 2026-07-30, so the one step labelled preferred was the one that
failed. Use either spelling below — both verified working:

```bash
# Preferred WHEN YOU ARE ALONE in the checkout: format only staged changes
git-clang-format-19          # direct
git clang-format-19          # identical; git dispatches to the same binary

# Check without writing (operates on the staged tree — the WHOLE index, see below)
git-clang-format-19 --diff --staged   # "clang-format did not modify any files" = clean

# Preferred WHEN OTHER AGENTS ARE LIVE: your own file, never writing
clang-format-19 --dry-run -Werror <file.cpp>

# Format specific files
clang-format-19 -i <file.cpp>                 # ⚠️ see the -i warning below
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

⚠️ **`git-clang-format-19 --staged` operates on the WHOLE INDEX, not on your files —
so in a shared checkout with live agents it is the dangerous one, and the two hazards
on this page point in OPPOSITE directions.** Found 2026-08-02 by an implementer who
ran the `--diff` form and had another agent's unformatted `unified-cache.cpp:5588`
reported at it. It correctly did **not** run the write form, which would have
reformatted that file underneath its owner.

The index is shared mutable state (`git add` does not scope a commit to your files; that is the same fact seen
from the commit side). So there is no single "preferred" command — it depends on
whether anyone else is live:

| situation | use | why |
|---|---|---|
| alone in the checkout | `git-clang-format-19` (staged) | changed lines only; avoids the 180-line drift |
| **other agents live** | `clang-format-19 --dry-run -Werror <your file>` | scoped to your file, and cannot write |
| ever | ~~`clang-format-19 -i <file>`~~ | forbidden — 180 lines of unrelated drift |

**Both caveats must survive together, and fixing one naively reintroduces the other.**
"Just use `clang-format-19` instead of the staged form" walks into the `-i` drift
problem; "always use the staged form" walks into reformatting a teammate's file. The
resolution is that the *safe* substitute under concurrency is specifically
`--dry-run -Werror`, which is neither whole-file-writing nor index-wide.

If `--dry-run` reports drift in your own file that you did not introduce, leave it —
that is pre-existing drift, and fixing it belongs in its own commit, not smuggled into
an unrelated change.

## 3. Repository map

Old location: `CLAUDE.md`, "Project Architecture". The repository itself is the source for this.
The list is kept as an orientation aid and can lag the tree.

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

## 4. Communication and workflow

Old location: `CLAUDE.md`, "Hard-Won Rules", "Communication & Workflow".

The first item is superseded. It says the human reads Discord and never the terminal. The human
now also works in the desktop app, so `CLAUDE.md` states the rule as a condition: when the harness
supplies a Discord channel id, use the Discord reply tool; in any other session, write to the
conversation.

Confirmed lessons from prior work on this fork. Treat them as defaults.

### Communication & Workflow
- **The user reads Discord, not the terminal.** CLI output is invisible to them. Any question, confirmation, decision prompt, or status update intended for the user MUST go through the Discord reply tool (the harness supplies the channel id each session). Terminal text is logging only — never "await a reply" there.
- **Work on the active feature branch** (`git branch --show-current` — do not trust a branch name written down here). When reviewing diffs, bound by BASE_SHA/HEAD_SHA, not "everything on the branch."
- **Worktrees are allowed and are the right tool for build-heavy parallel work** (owner decision, 2026-08-01). This entry previously said *"skip git worktrees — a worktree forces a fresh `build/` and loses the ~10-min ccache-warm hit rate."* The cost is real but was overstated: **ccache is global (`~/.ccache`), not per-tree**, so a worktree build still gets its hits; only the `build/` object tree and CMake cache are fresh.
  ⚠️ **Since llama.cpp-7mqd, a first build at a new path hits the cache for host compiles.** `scripts/sycl-build.sh` runs ccache with `base_dir` at the tree (opt out with `GGML_SYCL_CCACHE_BASE_DIR=0`), so ccache rewrites the absolute paths it receives to paths relative to `build/` before hashing. Measured 2026-09-26 on `ggml-sycl` plus seven tests: 206 of the 213 cacheable compiles hit in a fresh worktree, and `ggml-sycl.cpp` took 0.4 s against 292 s cold. Before that change, a new path missed on every TU whose command line embedded absolute paths; on 2026-08-01 one first build spent ~50 min on `ggml-sycl.cpp` alone. **What still misses at a new path:** the 12 icpx `-march=native` `ggml-cpu` compiles (never cached by ccache), the `LLAMA_CPP_SOURCE_ROOT` test TUs, and `ggml.c` when the commit changes. The device links still run, but they hit the ocloc cache wherever a link on this toolchain already stored the same SPIR-V, because that cache is keyed by toolchain and SPIR-V rather than by path (`libggml-sycl` linked in 27 s in a fresh worktree against 643 s uncached; a toolchain upgrade or changed SPIR-V misses in every tree). **Changing the launcher (the first build after the switch, or any toggle of the variable) reruns every host compile once**, because the launcher is on every compile command. Those reruns hit wherever a matching entry already exists (a plain rebuild of a tree that built plain before hit 106 of 107 cacheable compiles), and are cold only where none does. A hit build is bound by device links and host load (99-219 s measured).
  Consequence for fan-out: size a wave by the number of concurrent **builds** it triggers, since each still pays its device links and the uncached `ggml-cpu` compiles. Reusing an existing worktree for a follow-up task remains cheaper than creating one, and three concurrent cold builds on this 20-core box is already past the point where adding agents slows everything down (measured 2026-08-01: load 96, ~1 wave/hour).
  Measured 2026-08-01: with one shared `build/`, four agents serialised on a ~14-min build cycle and one track was starved **~2 hours**. Two worktree builds completed fine in the same session. Path-scoped commits prevent *file* conflicts; they do nothing about *build* contention, and that is what actually costs time.
  Use a worktree when a track will build repeatedly. Stay in the shared checkout when the work is small, or when it must operate on the checked-out branch itself (git refuses to check out one branch in two trees — so a **merge into** the active branch must happen in the main checkout, though only the build needs isolating).
  ⚠️ **Lock scope follows the build directory, not the act of building.** Shared checkout → take `/Apps/llama.cpp/BUILD.lock`. Own worktree → take nothing; nothing contends. `GPU.lock` is always global, from any tree, because the *devices* are shared.
  ⚠️ **Delete an evidence worktree only after the finding it supports has been reviewed.** ~1.4 GB is cheap next to a disagreement about what was measured that can no longer be settled (happened 2026-08-01).
- **`git merge-base HEAD master` is usually the WRONG before-point for a measurement.** On a long-lived feature branch it can sit dozens of commits behind the work you are attributing, silently crediting your change with everything in between. Use the commit immediately before your own first commit.
- **Fix-forward, never revert.** If a build or correctness test fails mid-implementation, diagnose and fix in a new commit. Don't `git revert` or `git checkout --` to undo progress.
- **Verify correctness before claiming any perf win.** `llama-bench` measures tok/s only — a change can boost throughput by silently skipping or mis-staging work and still emit garbage tokens. Before committing any change to kernel dispatch, weight staging, graph replay, or allocation routing, run the canonical Mistral completion gate (see "Verification Commands & Correctness Gates") and confirm the output. A fake +19.6% PP "win" shipped this way once and had to be reverted.

## 5. Tooling

Old location: `CLAUDE.md`, "Hard-Won Rules", "Tooling", and the closing note of that chapter.

Correction, checked 2026-11-16 at `423fd3e78`: `ggml/src/ggml-sycl/ggml-sycl.cpp` has 125,883
lines, not about 60,000.

- **codescout's index is BLIND inside `ggml/src/ggml-sycl/ggml-sycl.cpp`** (~60k lines). `search_text` and `find_references` return matches tagged `source: index` and silently omit real occurrences in that file — despite `search_text` being documented as an exhaustive live grep. This is not theoretical: an entire implementation plan was written on the false premise that a field had no other writers, because the one call site that mattered was missed. It cost a task to discover empirically. **In that file, verify with `cat ggml/src/ggml-sycl/ggml-sycl.cpp | grep -n '<pattern>'`** — a downstream pipe grep is permitted by the search hook; only command-position in-repo search is redirected. Never conclude "no other uses" there from codescout alone.
- **`/tmp` is tmpfs and does not survive a reboot.** Anything a later step depends on belongs in the committed artifact (a findings doc, a commit message), not in `/tmp`. A mid-session reboot has already destroyed capture artifacts here. An empty or missing artifact directory means *not verified*, never "nothing observed" — a `grep` over it passes vacuously.

> Live debugging state (active bug investigations, bisect results, perf-regression hunts) lives in the **codescout task tracker** (`task_ready`, `task_list`), not here. This section is for settled rules only.

## 6. Documentation index as it stood

Old location: `CLAUDE.md`, "Documentation". `CLAUDE.md` keeps a shorter index. This is the old one,
with its notes on how the LLM Wiki is shared between tools.

- **LLM Wiki (shared knowledge vault)**: `.llm-wiki/` — OKF v0.2, Obsidian-compatible,
  shared across harnesses (pi/omp via `@zosmaai/pi-llm-wiki`; Claude Code via the
  `llm-wiki` MCP server in `.mcp.json`; Obsidian UI in the `llm-wiki-obsidian`
  container, port 3001). Conventions: `.claude/skills/llm-wiki/SKILL.md` — never edit
  `raw/` or `meta/` by hand, search before creating pages, cite `SRC-*` ids.
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
