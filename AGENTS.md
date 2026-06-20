# AGENTS.md

Agent-facing workstation guidance for this repository. This file mirrors the
current machine-specific facts in `CLAUDE.md`; when the two conflict,
`CLAUDE.md` is the source of truth and `AGENTS.md` should be updated.

## Build Commands (Intel SYCL)

Always source oneAPI before running SYCL binaries. The build script handles
sourcing for builds:

```bash
source /opt/intel/oneapi/setvars.sh --force
```

### Build

Use the build script for SYCL work:

```bash
./scripts/sycl-build.sh
```

The script sources oneAPI, configures Ninja, enables required SYCL flags
including `-DGGML_SYCL_F16=ON`, sets `$ORIGIN` RPATH so `build/bin` binaries can
find colocated shared libraries, and outputs to `build/bin/`.

Common flags:

```bash
./scripts/sycl-build.sh                           # incremental build
./scripts/sycl-build.sh llama-bench               # build a single target
./scripts/sycl-build.sh -r                        # force CMake reconfigure
./scripts/sycl-build.sh -c                        # clean rm build/ and rebuild
```

Build time is roughly 10 minutes with ccache and 25 minutes without.

`-DGGML_SYCL_F16=ON` enables 16-bit float arithmetic throughout the SYCL backend.
It is active for dmmv dequant today and is required for the current fast Mistral
attention path. Declare it off only for precision-sensitive models.

### Ninja

The script always uses Ninja. It gives correct header dependency tracking and
fast no-op builds. If `build/` was created with another generator, run:

```bash
./scripts/sycl-build.sh -c
```

### Running Tests

```bash
source /opt/intel/oneapi/setvars.sh --force
ctest --test-dir build --output-on-failure -j $(nproc)

# Run a single test by name
ctest --test-dir build -R <test-name> -V
```

### Code Formatting

```bash
# Preferred: format only staged changes
git clang-format

# Format specific files
clang-format-19 -i <file.cpp>
clang-format-19 --dry-run -Werror <file.cpp>
```

## Project Architecture

### Core Directories

- `src/`: main llama library
- `include/llama.h`: public C API header
- `ggml/`: core tensor library
- `common/`: shared utility code for examples
- `examples/` and `tools/`: CLI tools
- `tests/`: CTest integration

### Key Binaries

Binaries are in `build/bin/`:

- `llama-cli`: interactive chat/inference
- `llama-completion`: non-interactive text completion
- `llama-server`: OpenAI-compatible HTTP server
- `llama-bench`: performance benchmarking
- `llama-quantize`: model quantization
- `llama-perplexity`: model evaluation

### Backend Structure

- `ggml/src/ggml-cpu/`: CPU backend
- `ggml/src/ggml-cuda/`: NVIDIA CUDA backend
- `ggml/src/ggml-metal/`: Apple Metal backend
- `ggml/src/ggml-sycl/`: Intel SYCL backend
- `ggml/src/ggml-vulkan/`: Vulkan backend

Key SYCL files:

- `ggml-sycl.cpp`: main backend, graph compute, mul_mat dispatch, buffer ops
- `unified-kernel.cpp/hpp`: unified MUL_MAT kernel with XMX/ESIMD/MMVQ dispatch
- `unified-cache.cpp/hpp`: tiered weight cache and SYCL memory allocator
- `mem-handle.cpp/hpp`: ref-counted allocation and cache-entry handles
- `common.hpp/cpp`: shared types, `extra_gpu`, layout policy, device management
- `mmvq.cpp`: matrix-vector quantized kernels for batch=1 TG fast path
- `mmq.cpp`: matrix-matrix quantized kernels
- `fattn.cpp`: flash attention implementation
- `dispatch.hpp`: kernel dispatch policy
- `quants.hpp`: SOA block offset calculations

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
are only transient ABI views resolved from `mem_handle` for
immediate kernel submission, oneDNN primitive calls, or tightly scoped CPU
access. Do not store raw pointers as the source of truth, use pointer addresses
as cache keys, or let pointers outlive their owning handle. Pointer tables and
dispatch caches must be derived from the stable identity/hash carried by
`mem_handle`, not from raw device addresses; if a table contains raw device
pointers for a kernel ABI, retain the corresponding handles for at least the
lifetime of the queued work or executable graph.

Weight cache entries are ref-counted through `mem_handle` leases. Eviction may
only remove entries whose in-use count is zero. If the cache cannot evict
because handles are still referenced, fix the missing release instead of
forcing eviction.

## Intel Arc GPU Memory Architecture (Critical for Multi-GPU)

On Intel Arc discrete GPUs (Xe architecture), there are two GPU address
translation tables:

- **GGTT** (Global Graphics Translation Table): 32-bit, 4 GB address space.
  Reserved for **kernel/privileged** resources only (GuC firmware, display
  engine).  User-space USM allocations do **NOT** consume GGTT aperture.

- **PPGTT** (Per-Process GTT): 48-bit, 256 TB address space per process.
  This is where **all user-space allocations** (device, host, shared USM)
  are mapped.  The PPGTT is effectively unlimited for practical purposes.

**Implication**: USM host memory (`sycl::malloc_host` / `zeMemAllocHost`) is
mapped through the PPGTT, not the GGTT.  There is no GGTT aperture constraint
on pinned host memory, even in multi-GPU setups.  Do NOT cap pinned host
memory budgets based on GGTT size estimates — the previous code that did this
reduced pinned budgets from 128 GB to ~1.9 GB in multi-GPU, preventing large
model support for no valid reason.

The previous multi-GPU hang was caused by a `ggml_sycl_info()` re-entry
deadlock (calling a function with a C++ static-init guard from within the
function that fills that static), NOT by GGTT overflow.

Source: Intel GPU PRM Vol06 "Memory Views"; Level Zero Core Spec §Memory;
Xe kernel driver documentation (`Documentation/gpu/xe/xe_mm.rst`).

## ggml Conventions

Matrix multiplication is unconventional: `C = ggml_mul_mat(ctx, A, B)` computes:

```text
C^T = A * B^T  <=>  C = B * A^T
```

Tensor storage:

- Tensors store data in row-major order
- Dimension 0 is columns
- Dimension 1 is rows
- Dimension 2 is matrices

Naming and style:

- Use `snake_case`
- Optimize for longest common prefix
- Prefer `<class>_<method>` names
- Enum values are uppercase and prefixed with the enum name
- Use `struct foo {}` rather than `typedef struct foo {} foo`
- Avoid new third-party dependencies
- Keep STL usage simple
- Formatting is 4 spaces, brackets on same line, `void * ptr`, `int & a`
- File names are lowercase with dashes for C/C++ and underscores for Python

## Local Environment

### Models

Models are stored in `/Storage/GenAI/models/`.

Default benchmark model:

```text
/Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf
```

Other useful local models:

- `mistral-7b-v0.1.Q4_K_M.gguf`
- `mistral-7b-v0.1.Q8_0.gguf`
- `gpt-oss-20b-mxfp4.gguf`
- `gpt-oss-120b-mxfp4-*.gguf`

### Verification Commands

Use these as the default quick gates on this workstation:

```bash
source /opt/intel/oneapi/setvars.sh --force

# Deterministic correctness check. Expected output starts:
# 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
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

# PP512 and TG128 benchmark
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128

# Backend operations after modifying ggml operators
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

If binaries fail to load colocated libraries, use:

```bash
export LD_LIBRARY_PATH="/Apps/llama.cpp/build/bin:$LD_LIBRARY_PATH"
```

### SYCL Device Selection

On multi-GPU systems, use `ONEAPI_DEVICE_SELECTOR` to choose the visible Level
Zero devices. Selector syntax is `backend:devices`: use `level_zero:0` for one
device, `level_zero:0,1` for a numeric multi-device set, or `level_zero:gpu` for
all Level Zero GPU devices. The `level_zero:gpu:0` strings printed by some tools
are display IDs, not valid selector values.

This workstation has 3 GPUs: Arc B580 device 0, Arc Pro B50 device 1, and iGPU
device 2.

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
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench ...
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench ...
ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/llama-bench ... # host-bounce paths required
```

`GGML_SYCL_VISIBLE_DEVICES=0` is not sufficient for the unified cache. Use
`ONEAPI_DEVICE_SELECTOR`.

Current-boot B580/B50 P2P topology warnings are diagnostic only; direct
peer-copy paths must stay disabled unless probed safe, but host-bounce
validation may continue. Frigate QSV/OpenVINO jobs on the iGPU render node are
not B580/B50 consumers.

### Patched compute-runtime

The system `libze_intel_gpu.so.1` is the patched 26.22/BMG-only build installed
at:

```text
/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.15.38646
```

It was built from `/Apps/compute-runtime-26.22-llama` branch
`llama/26.22-cross-device`, based on `upstream/releases/26.22`, with the local
wedged-i915 discovery fix, cross-device in-order dependency fixes, and the
upstream PR 930 USM compression fix. The build is BMG-only because the installed
IGC/ocloc does not recognize 26.22's future Xe3p/NVLP built-ins.

It is installed through the diverted system library path; stock `1.14.37020` and
the previous patched 26.09 files are preserved. Do not casually revert this
runtime. To roll back to the prior patched 26.09 runtime without removing the
diversion:

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

Validation on 2026-05-30: `sycl-ls` historically reported the B580 and B50
Level Zero devices on driver `1.15.38646`, and a full GPT-OSS multi-GPU
llama.cpp bench ran through the isolated/host-bounce path. Do not use
`sycl-ls` for B50 probing now; see the 2026-06-07 B50 safety note above. Raw
SYCL and Level Zero direct
device-to-device USM copy between B580 and B50 still fails
(`UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` / `ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`),
and importing a B580 device allocation on the B50 returns
`ZE_RESULT_ERROR_INVALID_ARGUMENT`. Kernel logs report:

```text
xe 0000:03:00.0: cannot be used for peer-to-peer DMA as the client and provider (0000:07:00.0) do not share an upstream bridge or whitelisted host bridge
```

This is a PCI P2PDMA/topology restriction, not just a compute-runtime selector
bug. Do not enable direct peer-copy/shared-context transfer paths by default
unless a runtime probe proves they are safe on the active hardware, kernel, and
driver.

### Performance Expectations

Mistral 7B Q4_0 on Arc B580:

| Metric | Expected tok/s | Notes |
|--------|----------------|-------|
| PP512, all VRAM | ~1700 | default no-FA bench path |
| TG128, all VRAM | ~81 | MMVQ fast path, SOA layout, graph replay |
| TG128, no graph | ~70 | MMVQ fast path alone |
| PP512, Level 3 30% budget | ~269 | partial GPU offload |
| TG128, Level 3 30% budget | ~14 | CPU offload |
| PP512 legacy | ~159 | `GGML_SYCL_UNIFIED_FORCE_LEGACY=1` |
| TG128 3-device | ~27 | `GGML_SYCL_SPLIT_RATIO="60,32,8"` |

Historical Mistral 7B Q4_0 on Arc Pro B50 with ECC disabled:

| Metric | Expected tok/s | Notes |
|--------|----------------|-------|
| PP512, all VRAM | ~1197 | B50 ECC disabled, default no-FA bench path |
| TG128, all VRAM | ~44 | Coalesced/SOA MMVQ, 70 W power cap |

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

## SYCL Environment Variables

Performance-critical defaults:

| Variable | Default behavior |
|----------|------------------|
| `GGML_SYCL_UNIFIED_SOA=0` | Disable SOA layout; default is ON |
| `GGML_SYCL_TG_FAST=0` | Disable MMVQ fast path; default is ON |
| `GGML_SYCL_DISABLE_GRAPH=1` | Disable graph replay; default is OFF |
| `GGML_SYCL_ONEDNN_PP=0` | Disable oneDNN prompt processing; default is ON |
| `GGML_SYCL_UNIFIED_FORCE_LEGACY=1` | Force legacy dispatch; default is OFF |

Experimental:

| Variable | Default behavior |
|----------|------------------|
| `GGML_SYCL_PP_PIPELINE=1` | Enable double-buffered FP16 dequant prefetch; default is OFF until GPT-OSS chat correctness is fixed |
| `GGML_SYCL_PERSISTENT_TG=1` | Enable persistent TG kernel; default is OFF |
| `GGML_SYCL_PERSISTENT_TG_PHASE=0` | Disable phase scheduling |
| `GGML_SYCL_PERSISTENT_TG_DAG=0` | Disable DAG scheduling |
| `GGML_SYCL_PERSISTENT_SPLIT=1` | Enable persistent split kernel |
| `GGML_SYCL_MOE_BLOCK_GRAPHLETS=1` | Enable experimental MoE block command graphlets; default is OFF |

Kernel dispatch tuning:

| Variable | Effect |
|----------|--------|
| `GGML_SYCL_FORCE_MMVQ=1` | Force MMVQ kernels |
| `GGML_SYCL_FORCE_ESIMD=1` | Force ESIMD kernels |
| `GGML_SYCL_FORCE_MMQ=1` | Force MMQ kernels |
| `GGML_SYCL_FORCE_DMMV=1` | Force DMMV kernels |
| `GGML_SYCL_ESIMD_MIN_BATCH=N` | Minimum batch size for ESIMD |
| `GGML_SYCL_ONEDNN_PP_MIN_BATCH=N` | Minimum batch for oneDNN PP |
| `GGML_SYCL_BATCH_EXPERTS=0` | Disable batched expert launches |

Memory and pressure:

| Variable | Effect |
|----------|--------|
| `GGML_SYCL_VRAM_BUDGET_PCT=N` | VRAM budget percent |
| `GGML_SYCL_KV_HOST=1` | Force KV cache to pinned host memory |
| `GGML_SYCL_KV_HOT_LAYERS=N` | Hot KV layer count |
| `GGML_SYCL_KV_HOT_PCT=N` | Hot KV window percent |
| `GGML_SYCL_FORCE_STREAMING=1` | Enable weight streaming |
| `GGML_SYCL_HOST_COMPUTE=1` | Use host-pinned compute buffers |
| `GGML_SYCL_UNIFIED_CACHE_MODE=<mode>` | Cache topology: auto, global, per_device |
| `GGML_SYCL_NO_PINNED=1` | Disable pinned host memory |
| `GGML_SYCL_WEIGHTS_EVICTABLE=1` | Allow weight eviction |
| `GGML_SYCL_MEM_BUDGET=<MB>` | Set VRAM budget in MB |

Debugging:

| Variable | Effect |
|----------|--------|
| `GGML_SYCL_DEBUG=1` | Detailed kernel logging; very noisy |
| `GGML_SYCL_UNIFIED_DEBUG=1` | Debug unified kernel dispatch |
| `GGML_SYCL_NAN_CHECK=1` | Enable NaN detection |
| `GGML_SYCL_VALIDATE=1` | A/B validation between kernel paths |
| `GGML_SYCL_GRAPH_RERECORD=1` | Diagnostic graph re-record path |
| `GGML_SYCL_OP_TIMEOUT_MS=<N>` | Abort if no inference progress for N ms |
| `GGML_SYCL_SAFE_MODE=1` | Drain queue after every op; slow diagnostic mode |
| `GGML_SYCL_LAYOUT_OVERRIDE=<mode>` | Force `aos`, `soa`, `coalesced`, or `xmx_tiled` |

There are many additional `GGML_SYCL_*` variables. Search with:

```bash
rg 'getenv\("GGML_SYCL' ggml/src/ggml-sycl/
```

## Profiling

VTune GPU offload example:

```bash
source /opt/intel/oneapi/setvars.sh --force
GGML_SYCL_DEVICE=0 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
vtune -collect gpu-offload -knob enable-stack-collection=true \
  -result-dir /tmp/vtune_b580_llama \
  -- ./build/bin/llama-bench \
    -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 64 -n 8 --tg-batch 4 -ngl 99 -fa 1
```

If VTune only shows memcpy tasks, use PTI and UR tracers for kernel time and
launch shapes. See `CLAUDE.md` for the full tracer setup.

## CI and Validation

Before submitting PRs:

1. Format code with `git clang-format`
2. Build with `./scripts/sycl-build.sh`
3. Run `ctest --test-dir build --output-on-failure`
4. For ggml operator changes, run `test-backend-ops`
5. Verify `llama-bench` and `llama-perplexity` do not regress where relevant

Add `ggml-ci` to a commit message to trigger extended CI workloads.

## Professional Engineering Standards

Spinach Rule: when you detect a visible flaw the user may not see, correction is
mandatory. Do not optimize for agreement.

- Challenge wrong assumptions directly
- Question unclear requirements before implementing risky changes
- Identify performance and security trade-offs
- Never fake progress or certainty

## Documentation

- `docs/build.md`
- `docs/backend/SYCL.md`
- `docs/development/HOWTO-add-model.md`
- `CONTRIBUTING.md`
- `.github/copilot-instructions.md`

## Landing the Plane (Session Completion)

When ending a work session, complete all steps below. Work is not complete until
`git push` succeeds.

1. File issues for remaining work
2. Run quality gates if code changed
3. Update issue status
4. Push to remote:

```bash
git pull --rebase
bd sync
git push
git status
```

5. Clean up stale stashes or branches where appropriate
6. Verify all intended changes are committed and pushed
7. Hand off remaining context

Critical rules:

- Work is not complete until `git push` succeeds
- Do not stop with local-only committed work
- Do not say "ready to push when you are"; push the branch
- If push fails, resolve and retry
