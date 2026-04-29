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
- `unified-cache.cpp/hpp`: tiered weight cache
- `common.hpp/cpp`: shared types, `extra_gpu`, layout policy, device management
- `mmvq.cpp`: matrix-vector quantized kernels for batch=1 TG fast path
- `mmq.cpp`: matrix-matrix quantized kernels
- `fattn.cpp`: flash attention implementation
- `dispatch.hpp`: kernel dispatch policy
- `quants.hpp`: SOA block offset calculations

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

# PP512 and TG128 benchmark
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128

# Backend operations after modifying ggml operators
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-backend-ops
```

If binaries fail to load colocated libraries, use:

```bash
export LD_LIBRARY_PATH="/Apps/llama.cpp/build/bin:$LD_LIBRARY_PATH"
```

### SYCL Device Selection

On multi-GPU systems, explicitly select a single device. Without this, llama.cpp
may use all visible GPUs and the unified kernel can hang.

This workstation has 3 GPUs: Arc B580 device 0, Arc Pro B50 device 1, and iGPU
device 2.

```bash
sycl-ls
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench ...
```

`GGML_SYCL_VISIBLE_DEVICES=0` is not sufficient for the unified cache. Use
`ONEAPI_DEVICE_SELECTOR`.

### Patched compute-runtime

The system `libze_intel_gpu.so.1` is the patched build at:

```text
/Apps/compute-runtime/build-26.09/bin/libze_intel_gpu.so.1.14.37435
```

It is installed via `dpkg-divert`; stock `1.14.37020` is preserved. The patched
runtime fixes the m09zb `event.wait()` post-init hang during alloc-probe and
cleanly enforces per-allocation hardware caps on Arc B580.

Do not casually revert this runtime. Reverting to stock without restoring the old
allocation probe can reintroduce silent oversized allocation hangs.

### Performance Expectations

Mistral 7B Q4_0 on Arc B580:

| Metric | Expected tok/s | Notes |
|--------|----------------|-------|
| PP512, all VRAM | ~1700 | oneDNN SDPA graph / f16 attention path |
| TG128, all VRAM | ~81 | MMVQ fast path, SOA layout, graph replay |
| TG128, no graph | ~70 | MMVQ fast path alone |
| PP512, Level 3 30% budget | ~269 | partial GPU offload |
| TG128, Level 3 30% budget | ~14 | CPU offload |
| PP512 legacy | ~159 | `GGML_SYCL_UNIFIED_FORCE_LEGACY=1` |
| TG128 3-device | ~27 | `GGML_SYCL_SPLIT_RATIO="60,32,8"` |

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
| `GGML_SYCL_PP_PIPELINE=1` | Enable double-buffered FP16 dequant prefetch; default is OFF |
| `GGML_SYCL_PERSISTENT_TG=1` | Enable persistent TG kernel; default is OFF |
| `GGML_SYCL_PERSISTENT_TG_PHASE=0` | Disable phase scheduling |
| `GGML_SYCL_PERSISTENT_TG_DAG=0` | Disable DAG scheduling |
| `GGML_SYCL_PERSISTENT_SPLIT=1` | Enable persistent split kernel |

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
| `GGML_SYCL_UNIFIED_CACHE=0` | Disable unified cache |
| `GGML_SYCL_UNIFIED_CACHE_MODE=<mode>` | auto, device, host, mmap |
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
