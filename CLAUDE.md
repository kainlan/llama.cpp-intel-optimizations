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
- **dmmv dequant** (active today): `dfloat`/`dfloat2` typedef at `ggml/src/ggml-sycl/common.hpp:363` pivots to `sycl::half`/`sycl::half2` under the flag.
- **Attention path** (via bead `llama.cpp-mgx69` — gates Q f16 cast and `afloat` accumulator typedef in `fattn-xmx-f16-v2.hpp`; unlocks oneDNN SDPA eligibility on Mistral once the bead lands).

Precision tradeoff: ~4 mantissa bits vs f32. Declare OFF only for precision-sensitive models (phi-2 per `ggml/include/ggml.h:1294` comment, or similar).

### Ninja vs Make
The script always uses Ninja (`-G Ninja`). Reasons:
- **Correct header dependency tracking**: Changes to `.hpp` files reliably trigger recompilation
- **Faster no-op builds**: 1.5s vs 73s for Make on large projects

If a `build/` was created with a different generator, run `./scripts/sycl-build.sh -c` to wipe and reconfigure.

### Running Tests
```bash
source /opt/intel/oneapi/setvars.sh --force
ctest --test-dir build --output-on-failure -j $(nproc)

# Run a single test by name
ctest --test-dir build -R <test-name> -V
```

### Code Formatting
```bash
# Preferred: format only staged changes (uses .clang-format)
git clang-format

# Format specific files
clang-format-19 -i <file.cpp>
clang-format-19 --dry-run -Werror <file.cpp>  # dry-run check
```

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
Key files in the SYCL backend:
- **`ggml-sycl.cpp`** (~31K lines): Main backend — graph_compute, mul_mat dispatch, buffer ops, graph replay
- **`unified-kernel.cpp/hpp`** (~6.4K/~2.5K lines): Unified MUL_MAT kernel with XMX/ESIMD/MMVQ dispatch
- **`unified-cache.cpp/hpp`** (~5.3K/~960 lines): Tiered weight cache and SYCL memory allocator
- **`mem-handle.cpp/hpp`**: Ref-counted allocation and cache-entry handles
- **`common.hpp/cpp`** (~3K/~2.2K lines): Shared types, `extra_gpu` struct, layout_policy, device management
- **`mmvq.cpp`** (~4.2K lines): Matrix-vector quantized kernels (batch=1 TG fast-path)
- **`mmq.cpp`** (~7K lines): Matrix-matrix quantized kernels (persistent TG, streaming PP)
- **`fattn.cpp`** (~1.5K lines): Flash attention implementation
- **`dispatch.hpp`** (~350 lines): Kernel dispatch policy and routing
- **`quants.hpp`** (~190 lines): SOA block offset calculations for quantized types

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
- **The user reads Discord, not the terminal.** CLI output is invisible to them. Any question, confirmation, decision prompt, or status update intended for the user MUST go through the Discord reply tool (the harness injects the channel id each session). Terminal text is logging only — never "await a reply" there.
- **Work in-place on the active feature branch** (currently `feature/sycl-coalescing`); skip git worktrees. A worktree forces a fresh `build/` and loses the ~10-min ccache-warm hit rate. When reviewing diffs, bound by BASE_SHA/HEAD_SHA, not "everything on the branch."
- **Fix-forward, never revert.** If a build or correctness test fails mid-implementation, diagnose and fix in a new commit. Don't `git revert` or `git checkout --` to undo progress.
- **Verify correctness before claiming any perf win.** `llama-bench` measures tok/s only — a change can boost throughput by silently skipping or mis-staging work and still emit garbage tokens. Before committing any change to kernel dispatch, weight staging, graph replay, or allocation routing, run the canonical completion gate and confirm the output. A fake +19.6% PP "win" shipped this way once and had to be reverted.

  Canonical gate — output must end `6, 7, 8, 9, 10, 11, 12, 13, 14, 15`:
  ```bash
  ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
    -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
    -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
  ```
  Anything else (`###...`, `!!!...`, repetition, `<unk>`) means the path is broken — fix or revert before commit, no matter the throughput number.

### Safety (these have wedged or OOM-locked this host)
- **Never run `test-backend-ops` in a subagent or background task.** It allocates hundreds of GPU BOs whose TTM shmem backing grows to 50–224 GB and trips the OOM killer (two lockups on 2026-04-06). For automated GPU testing use only `llama-bench`, `llama-completion`, or a targeted `ctest -R <name>`. Run `test-backend-ops` manually, with monitoring, only.
- **Always `timeout 60` GPT-OSS 20B test runs.** The historical host-MoE-routing wedge (GuC `guc_id=6`, unrecoverable system death) was closed by commit `ec7f04ac4`, but keep the timeout as a guard. Distinguish the userspace wedge (`guc_id=6`, attributed `in <llama-bench>`, unrecoverable) from the benign environmental XE timeout (`guc_id=0`, `in no process [-1]`, auto-recovers).
- **Benchmark numbers are invalid after any crash/kill on that card** (xe GT-reset cascades) — check `dmesg` first. `SAFE_MODE`/op-timing diagnostics can themselves wedge cards.

### Architecture
- **The unified cache owns all GPU/host memory** (decision Feb 9, 2026). Weight placement, eviction (device→pinned host→mmap), and budget tracking all flow through it.
- **Use smart handles, never hold a raw `void*` from the cache.** A raw VRAM pointer becomes dangling the moment the cache evicts to host → DEVICE_LOST/corruption. Handles must resolve location on dereference so the cache can move data between tiers transparently.
- **Host-resident weights → CPU dispatch, not GPU PCIe "zero-copy."** Measured CPU AOS = 18–30 GB/s vs GPU zero-copy = 11.3 GB/s (1.6–2.6x slower). Parallelize CPU work with GPU via `sycl::depends_on` (~9.7 µs cross-device latency). Never feed a host-pinned pointer to a GPU kernel as "zero-copy."
- **The VRAM budget calc is correct by design** (`min(total*pct, free_at_init)`). Low free VRAM is a system problem (other GPUs active, driver overhead), not an app bug to "fix" by ignoring free VRAM — fix the root cause at the system level.
- **Small-block dequant (Q4_0/Q8_0/Q4_K) belongs on standard SYCL, not ESIMD.** ESIMD measured 1.9x SLOWER on Arc B580 + oneAPI 2025.3 (block granularity too small to amortize LSC loads). The real dequant lever is structural — fuse dequant into the matmul. Opt-in retest hatch: `GGML_SYCL_ESIMD_DEQUANT=1`.

> Live debugging state (active bug investigations, bisect results, perf-regression hunts) lives in **beads** (`bd ready`, `bd list`), not here. This section is for settled rules only.

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

### Verification Commands
```bash
source /opt/intel/oneapi/setvars.sh --force

# Non-interactive completion (deterministic output for testing)
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0

# GPT-OSS B50 chat correctness gate. Expected output starts:
# : 1, 2, 3, 4, 5
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

### GPT-OSS Prompt Template Rule

Use the exact gate below for GPT-OSS correctness tests. The prompt is:

```text
Count from 1 to 5. Answer with only: 1, 2, 3, 4, 5
```

Use `llama-cli -cnv` so the CLI applies the model's embedded GGUF/Jinja chat
template to that text as a user message. Do not hand-render a raw Harmony
prompt, and do not pass `--chat-template gpt-oss` or a custom template unless
the test is explicitly about that formatter. `llama-cli --help` reports Jinja
enabled by default and `--chat-template` as a custom override whose default is
the template from model metadata.

Web and local verification rechecked on 2026-06-19:

- GPT-OSS models were trained for OpenAI's Harmony response format and should
  not be run with raw text or a generic chat format.
- OpenAI's implementation-verification guide warns that inference providers
  must map inputs to Harmony correctly; wrong prompt formatting can cause
  cascading generation issues.
- OpenAI's GPT-OSS Transformers guide says prompts should be built with the
  tokenizer chat template or `openai-harmony`.
- The OpenAI Hugging Face model card says the Transformers chat template
  automatically applies Harmony and direct `model.generate` callers must apply
  Harmony manually.
- The `openai/gpt-oss-20b` Jinja template accepts `reasoning_effort`, defaults
  it to `medium`, renders `Reasoning: medium` in the Harmony system message,
  renders the user prompt as a Harmony user message, and appends
  `<|start|>assistant` as the generation prompt.
- The llama.cpp GPT-OSS guide says `--jinja` uses the Jinja chat template
  embedded in the GGUF and that the `ggml-org/gpt-oss` GGUFs have a built-in
  chat template used by default; manual template overrides are only for known
  template bugs or specialized experiments.
- Local `llama-cli --help` confirms `--jinja` defaults to enabled and the chat
  template defaults to the one taken from model metadata.

For cross-branch regression tests, pin the Harmony `reasoning_effort` template
argument to `medium` with `--chat-template-kwargs`. The known-good B50
GPT-OSS prompt rendered `Reasoning: medium`; pinning prevents accidental
changes in template metadata, CLI defaults, or test harness behavior from
moving the prompt while comparing backend performance. `--reasoning-format
none` controls how reasoning output is shown or hidden and is not a substitute
for pinning the template argument. The deterministic count gate deliberately
uses `--reasoning-budget 0` with hidden reasoning so the expected answer is a
short final-channel string; for normal GPT-OSS chat/server parser validation,
use llama.cpp's automatic reasoning handling instead of treating `none` as a
model-format requirement.

Sources checked 2026-06-19:

- `https://developers.openai.com/cookbook/articles/openai-harmony`
- `https://developers.openai.com/cookbook/articles/gpt-oss/verifying-implementations`
- `https://developers.openai.com/cookbook/articles/gpt-oss/run-transformers`
- `https://developers.openai.com/cookbook/articles/gpt-oss/handle-raw-cot`
- `https://huggingface.co/openai/gpt-oss-20b`
- `https://huggingface.co/openai/gpt-oss-20b/blob/main/chat_template.jinja`
- `https://github.com/ggml-org/llama.cpp/discussions/15396`

Canonical B50 GPT-OSS correctness gate:

```bash
source /opt/intel/oneapi/setvars.sh --force
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-cli \
  -m /Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf -ngl 99 \
  -cnv -st --simple-io --no-display-prompt \
  --chat-template-kwargs '{"reasoning_effort":"medium"}' \
  --reasoning-format none --reasoning-budget 0 \
  -p 'Count from 1 to 5. Answer with only: 1, 2, 3, 4, 5' \
  -n 48 --seed 42 --temp 0
```

Expected output starts with `: 1, 2, 3, 4, 5`. The leading colon is normal for
this CLI/Harmony rendering. `llama-bench` is valid for PP/TG throughput, but it
does not prove chat-template correctness; use the gate above before trusting
GPT-OSS performance numbers.

### Patched compute-runtime (system default as of 2026-05-30)

The system `libze_intel_gpu.so.1` is the patched 26.22/BMG-only build installed at
`/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.15.38646` from
`/Apps/compute-runtime-26.22-llama` branch `llama/26.22-cross-device`. The build
is based on `upstream/releases/26.22` and carries the local wedged-i915 discovery
fix, the cross-device in-order dependency fixes, and the upstream PR 930 USM
compression fix. It was configured with `SUPPORT_GEN_DEFAULT=FALSE`,
`SUPPORT_PLATFORM_DEFAULT=FALSE`, and `SUPPORT_BMG=TRUE` because the installed
IGC/ocloc does not recognize 26.22's future Xe3p/NVLP built-ins.

The install still uses the diverted system library path; stock `1.14.37020` is
preserved at `/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.14.37020.stock`.
The previous patched 26.09 files are also preserved. To roll back to the prior
patched runtime without removing the diversion:

```bash
sudo ln -sfn libze_intel_gpu.so.1.14.37435.pre-single-device-default-ctx /usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1
sudo ldconfig
```

As of 2026-06-15, unowned stale Level Zero loader/tracing/validation libraries
from `/usr/local/lib` were moved to
`/usr/local/lib/llama-backup-level-zero-20260615-100931` because they made new
processes resolve `libze_loader.so.1.27.0` ahead of the packaged
`/usr/lib/x86_64-linux-gnu` loader. Keep `libze_loader.so.1`,
`libze_tracing_layer.so.1`, and `libze_validation_layer.so.1` absent from
`/usr/local/lib`; `ldconfig -p` should resolve them from
`/usr/lib/x86_64-linux-gnu`.

Validation on 2026-05-30:
`sycl-ls` historically reported B580 and B50 Level Zero devices on driver
`1.15.38646`, and `ONEAPI_DEVICE_SELECTOR=level_zero:0,1` could run a full
GPT-OSS bench through llama.cpp's isolated/host-bounce path. Do not use
`sycl-ls` for B50 probing now; see the 2026-06-07 B50 safety note below. Raw
SYCL and Level Zero direct
device-to-device USM copy between B580 and B50 still fails
(`UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` / `ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`),
and importing a B580 device allocation on the B50 returns
`ZE_RESULT_ERROR_INVALID_ARGUMENT`. Kernel logs report:

```text
xe 0000:03:00.0: cannot be used for peer-to-peer DMA as the client and provider (0000:07:00.0) do not share an upstream bridge or whitelisted host bridge
```

This is a PCI P2PDMA/topology restriction, not just a compute-runtime selector
bug. Do not enable direct peer-copy or shared-context transfer paths by default
unless a runtime probe proves they are safe on the active hardware, kernel, and
driver.

The patched runtime still fixes the m09zb `event.wait()` post-init hang during
alloc-probe and cleanly enforces per-allocation hardware caps. Reverting to stock
without restoring the old allocation probe can reintroduce silent oversized
allocation hangs.

### SYCL Device Selection

On multi-GPU systems, use `ONEAPI_DEVICE_SELECTOR` to choose the visible Level
Zero devices. Selector syntax is `backend:devices`: use `level_zero:0` for one
device, `level_zero:0,1` for a numeric multi-device set, or `level_zero:gpu` for
all Level Zero GPU devices. The `level_zero:gpu:0` strings printed by some tools
are display IDs, not valid selector values.

This system has 3 GPUs: Arc B580 (device 0), Arc Pro B50 (device 1), iGPU (device 2).

As of 2026-06-07 after a fresh reboot and removal of repo-side selector guards,
single-GPU B50 validation can run with `ONEAPI_DEVICE_SELECTOR=level_zero:1`.
Do not add backend or harness code that parses, rewrites, or refuses
`ONEAPI_DEVICE_SELECTOR`; device selection belongs to oneAPI/SYCL. Previous-boot
evidence and P2P topology warnings must not alter fresh-boot SYCL selection
behavior. `sycl-ls` is still not a preferred B50 health probe because it has
previously wedged this host in `ttm_resource_manager_usage -> drm_ioctl ->
xe_drm_ioctl` after a reset/oops sequence.

Do not run old comparison binaries such as the `60a8c042` known-good build with
metadata-only helper commands (`--help`, `--version`, or `lsof /dev/dri/*`) on
the discrete render nodes. On 2026-06-19, `llama-cli --help` from that build
initialized SYCL and left a process stuck in `xe_vm_destroy_ioctl ->
drm_exec_lock_obj`; an `lsof` probe then stuck in `xe_bo_lock`. Sysfs PCI FLR
and debugfs GT reset returned successfully but did not clear the D-state tasks.
Use the canonical gated inference command for cross-build comparisons after a
fresh reboot, and avoid DRM fdinfo probes while a SYCL process is wedged.
Current `common_params_parse()` must keep `--help`, `--version`, cache-list, and
completion generation metadata-only even when `LLAMA_ARG_*` GPU env vars are
set; verify with `test-arg-parser` after parser changes.

```bash
# Select an explicit Level Zero device set.
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench ...
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench ...
ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/llama-bench ... # host-bounce paths required

# NOTE: GGML_SYCL_VISIBLE_DEVICES=0 does NOT work - it filters at llama.cpp level
# but unified cache still sees all Level Zero devices. Use ONEAPI_DEVICE_SELECTOR.
```

Current-boot B580/B50 P2P topology warnings are diagnostic only; direct
peer-copy paths must stay disabled unless probed safe, but host-bounce
validation may continue. Frigate QSV/OpenVINO jobs on the iGPU render node are
not B580/B50 consumers.

### Performance Expectations (Mistral 7B Q4_0, Arc B580)

| Metric | tok/s | Notes |
|--------|-------|-------|
| PP512 (Level 0, all VRAM) | ~1700 | default no-FA bench path |
| TG128 (Level 0, all VRAM) | ~81 | MMVQ fast-path with SOA layout + graph replay + SCRATCH TLSF zone |
| TG128 (no graph) | ~70 | MMVQ fast-path alone (graph adds ~13%) |
| PP512 (Level 3, 30% budget) | ~269 | 15/33 GPU layers, rest on CPU |
| TG128 (Level 3, 30% budget) | ~14 | CPU offload via fit_params |
| PP512 (legacy) | ~159 | `GGML_SYCL_UNIFIED_FORCE_LEGACY=1` |
| TG128 3-device (B580+B50+CPU) | ~27 | `GGML_SYCL_SPLIT_RATIO="60,32,8"` tensor split |
| TG128 (persistent TG, phase) | ~30 | `GGML_SYCL_PERSISTENT_TG=1` (experimental) |
| TG128 (persistent TG, DAG) | ~19 | `GGML_SYCL_PERSISTENT_TG=1` + `PHASE=0 DAG=1` |

### Historical Performance Expectations (Arc Pro B50, ECC Disabled)

Mistral 7B Q4_0:

| Metric | tok/s | Notes |
|--------|-------|-------|
| PP512 (Level 0, all VRAM) | ~1197 | default no-FA bench path, ECC disabled |
| TG128 (Level 0, all VRAM) | ~44 | Coalesced/SOA MMVQ, 70 W power cap |

Do not use `GGML_SYCL_FA_ONEDNN_ALLOW=1` to restore Mistral PP numbers. It can
raise PP throughput, but the deterministic completion gate produces incorrect
output with the current nc!=D contiguity bypass.

GPT-OSS 20B MXFP4:

| Device selector | PP512 tok/s | TG128 tok/s | Notes |
|-----------------|------------:|------------:|-------|
| `level_zero:1` B50 ECC-off | current ~926; target >1100 | current ~48; target ~50+ | Fresh-boot B50 smoke passes; PP target still unmet |
| `level_zero:0` B580 | ~66 | ~17 | Smaller VRAM budget causes more pressure |
| `level_zero:0,1` | TBD | TBD | Use isolated/host-bounce transfer paths; direct P2P is not available |

### Current Regression Baselines And Suspect Delta

Do not accept lower post-merge or post-debug numbers as new baselines. Beads
`llama.cpp-aqzz3.1`, `llama.cpp-po3nd.2.45`, `llama.cpp-po3nd.2.46`, and
`llama.cpp-ix58x` record the hard guardrails:

- B50 GPT-OSS20B MXFP4 FA-on should restore/maintain >1100 PP512 and about
  50+ TG128; older restored-fast-path evidence was about 1255 PP512 / 52
  TG128, with the canonical GGUF chat-template count gate passing.
- B580 Mistral 7B Q4_0 FA-on should restore/maintain >2000 PP512 and >85
  TG128. `docs/backend/SYCL.md` records build `5b206c499-dirty` at PP512
  `2173.92 +/- 10.01` and TG128 `88.42 +/- 0.47`, with the deterministic
  count gate correct.

For the latest regression hunt, use `581babb476b726665a03345feb1a9ebcabe630db`
as the close pre-regression comparison point. The first-parent delta to
`f7a332578` is:

- `06f8887a6` restore unified-cache prompt headroom
- `a42a4c9a3` harden unified-cache view ownership
- `129a04fcb` tighten gpu performance gates
- `feae906b4` restore default MoE block graphlets
- `f7a332578` prefer MoE block graphlets for decode

The most suspicious default-on change in that small delta is the MoE block
command-graphlet path. It must stay opt-in via
`GGML_SYCL_MOE_BLOCK_GRAPHLETS=1` until same-build B50 GPT-OSS and B580
Mistral correctness/performance gates pass on a clean boot. Prompt XMX MoE PP
is also unsafe as a default; keep `GGML_SYCL_XMX_MOE_ALLOW_UNSAFE_PP` /
`GGML_SYCL_XMX_MOE_PP` opt-in only. `GGML_SYCL_PP_PIPELINE=1` remains a
diagnostic proof knob, not a default, because it has shown GPT-OSS chat
correctness failures.

### SYCL Environment Variables

**Performance-critical (all default ON, opt-out)**:
| Variable | Default | Effect |
|----------|---------|--------|
| `GGML_SYCL_UNIFIED_SOA=0` | ON | Disable SOA memory layout (AOS fallback, ~4x slower TG) |
| `GGML_SYCL_TG_FAST=0` | ON | Disable MMVQ fast-path bypass (slower TG) |
| `GGML_SYCL_DISABLE_GRAPH=1` | OFF | Disable SYCL graph replay (minimal TG impact ~3%, mainly helps PP) |
| `GGML_SYCL_ONEDNN_PP=0` | ON | Disable oneDNN for prompt processing |
| `GGML_SYCL_UNIFIED_FORCE_LEGACY=1` | OFF | Force legacy kernel dispatch (bypass unified kernel) |

**Experimental (opt-in, off by default)**:
| Variable | Default | Effect |
|----------|---------|--------|
| `GGML_SYCL_PP_PIPELINE=1` | OFF | Enable double-buffered FP16 weight dequant prefetch. B50 GPT-OSS `llama-bench` PP improves to ~1030-1043 tok/s, but GPT-OSS chat correctness currently fails with repeated `isNaN`, so keep this opt-in until fixed. |

**Kernel dispatch tuning**:
| Variable | Effect |
|----------|--------|
| `GGML_SYCL_FORCE_MMVQ=1` | Force MMVQ kernels for all batch sizes |
| `GGML_SYCL_FORCE_ESIMD=1` | Force ESIMD kernels |
| `GGML_SYCL_FORCE_MMQ=1` | Force MMQ kernels |
| `GGML_SYCL_FORCE_DMMV=1` | Force DMMV kernels |
| `GGML_SYCL_ESIMD_MIN_BATCH=N` | Min batch size for ESIMD dispatch |
| `GGML_SYCL_ONEDNN_PP_MIN_BATCH=N` | Min batch for oneDNN PP path |
| `GGML_SYCL_ONEDNN_MUL=1` | Enable oneDNN for element-wise MUL (default OFF, SYCL kernel is 2.3x faster) |
| `GGML_SYCL_BATCH_EXPERTS=0` | Disable batched expert kernel launches (default ON) |

**Persistent TG kernel (experimental, opt-in)**:
| Variable | Default | Effect |
|----------|---------|--------|
| `GGML_SYCL_PERSISTENT_TG=1` | OFF | **Required** to enable persistent TG kernel. Without this, all other PERSISTENT_TG_* vars are ignored |
| `GGML_SYCL_PERSISTENT_TG_PHASE=0` | ON | Disable phase-based scheduling (falls back to DAG or legacy barrier) |
| `GGML_SYCL_PERSISTENT_TG_DAG=0` | ON | Disable DAG scheduling (falls back to legacy barrier) |
| `GGML_SYCL_PERSISTENT_TG_N_WGS=N` | auto | Override work-group count (auto: max_compute_units/4, clamped 4-64) |
| `GGML_SYCL_PERSISTENT_TG_LOG_POLICY=1` | OFF | Print kernel dispatch mode (phase/dag/split/n_wgs) on each launch |
| `GGML_SYCL_MOE_BLOCK_GRAPHLETS=1` | OFF | Enable experimental MoE block command graphlets |
| `GGML_SYCL_PERSISTENT_SPLIT=1` | OFF | Enable persistent kernel for multi-device row-split |

Testing persistent TG modes:
```bash
# Phase mode (default when persistent TG enabled)
GGML_SYCL_PERSISTENT_TG=1 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-bench -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -n 128

# DAG mode (disable phase, enable DAG)
GGML_SYCL_PERSISTENT_TG=1 GGML_SYCL_PERSISTENT_TG_PHASE=0 GGML_SYCL_PERSISTENT_TG_DAG=1 \
  ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-bench -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -n 128

# Correctness check (must output "6, 7, 8, 9, 10")
GGML_SYCL_PERSISTENT_TG=1 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```

**Memory budget and pressure hierarchy**:
| Variable | Default | Effect |
|----------|---------|--------|
| `GGML_SYCL_VRAM_BUDGET_PCT=N` | 90 | VRAM budget as % of total (triggers CPU offload when model exceeds) |
| `GGML_SYCL_KV_HOST=1` | OFF | Force KV cache to host pinned memory (Level 1 offload) |
| `GGML_SYCL_KV_HOT_LAYERS=N` | auto | Hot layer count for per-layer KV hot/cold tiering |
| `GGML_SYCL_KV_HOT_PCT=N` | auto | Hot window as % of total KV buffer |
| `GGML_SYCL_FORCE_STREAMING=1` | OFF | Enable GPU weight streaming (Level 5, last resort) |
| `GGML_SYCL_HOST_COMPUTE=1` | OFF | Use host-pinned compute buffers (eliminates staging for CPU-dispatched layers) |
| `GGML_SYCL_PIPELINE_MOE=1` | OFF | Pipeline multi-GPU MoE: overlap B50 expert compute with GPU0 attention via background scatter thread |
| `GGML_SYCL_PIPELINE_CPU=1` | OFF | Pipeline CPU expert compute with GPU attention across MoE layers: CPU experts from layer N run during layer N+1 attention |

**Cache and memory**:
| Variable | Effect |
|----------|--------|
| `GGML_SYCL_UNIFIED_CACHE_MODE=<mode>` | Cache topology: auto, global, per_device |
| `GGML_SYCL_NO_PINNED=1` | Disable pinned host memory |
| `GGML_SYCL_WEIGHTS_EVICTABLE=1` | Allow weight eviction under memory pressure |
| `GGML_SYCL_MEM_BUDGET=<MB>` | Set VRAM budget in MB |

**Debugging**:
| Variable | Effect |
|----------|--------|
| `GGML_SYCL_DEBUG=1` | Enable detailed kernel dispatch logging (MASSIVE output) |
| `GGML_SYCL_UNIFIED_DEBUG=1` | Debug unified kernel dispatch |
| `GGML_SYCL_NAN_CHECK=1` | Enable NaN detection in outputs |
| `GGML_SYCL_VALIDATE=1` | Enable A/B validation between kernel paths |
| `GGML_SYCL_GRAPH_RERECORD=1` | Use graph re-record instead of replay (very slow, diagnostic only) |
| `GGML_SYCL_OP_TIMEOUT_MS=<N>` | Abort with diagnostic if no inference progress for N ms (default 30000, set to 0 to disable). Fires before the xe driver's 10s GT-reset cascade. Effective detection latency is `timeout + ~500 ms`. |
| `GGML_SYCL_SAFE_MODE=1` | Drain the SYCL queue after every op submit so a fault surfaces at the op that caused it (2-3x slowdown, implies `GGML_SYCL_DISABLE_GRAPH=1`). Useful for CI canaries and correlating intermittent wedges 1:1 with their triggering op. |

**Note**: There are 100+ additional debug/tuning env vars (GGML_SYCL_*). Search with `grep -r 'getenv("GGML_SYCL' ggml/src/ggml-sycl/` to find them all.

## CI and Validation

### Before Submitting PRs
1. Format code: `git clang-format` (preferred) or `clang-format-19 -i <files>`
2. Build: `./scripts/sycl-build.sh`
3. Test: `ctest --test-dir build --output-on-failure`
4. For ggml changes: Run `test-backend-ops` on multiple backends — **manually only, never in a subagent/background task (OOM hazard, see Hard-Won Rules)**
5. Verify correctness: run the canonical completion gate (Hard-Won Rules) — tokens must be right, not just fast
6. Verify performance: `llama-bench` and `llama-perplexity` should not regress

### Triggering Heavy CI
Add `ggml-ci` to commit message to trigger extended CI workloads.

## Documentation

- **Build Details**: `docs/build.md`
- **Backend SYCL**: `docs/backend/SYCL.md`
- **Add New Model**: `docs/development/HOWTO-add-model.md`
- **Contributing**: `CONTRIBUTING.md` (coding/naming guidelines, PR process)
- **Copilot Instructions**: `.github/copilot-instructions.md` (cross-platform build/test patterns)
