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
- **`unified-cache.cpp/hpp`** (~5.3K/~960 lines): Tiered weight cache (device VRAM, pinned host, mmap)
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

# Benchmark prompt processing (PP) and token generation (TG)
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128

# Test backend operations (after modifying ggml operators)
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-backend-ops
```

### Patched compute-runtime (system default as of 2026-04-25)

The system `libze_intel_gpu.so.1` is the **patched** build at `/Apps/compute-runtime/build-26.09/bin/libze_intel_gpu.so.1.14.37435` (branch `fix/combined-26.09`). Installed via `dpkg-divert` so apt won't overwrite. The stock 1.14.37020 is preserved at `/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.14.37020.stock`.

The patched runtime fixes the **m09zb wedge** (`event.wait()` post-init hang during alloc-probe). Reverting to stock will reintroduce that wedge. To revert if needed:

```bash
sudo dpkg-divert --rename --remove /usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.14.37020
sudo ln -sf libze_intel_gpu.so.1.14.37020 /usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1
sudo ldconfig
```

The patched runtime correctly enforces per-allocation hardware caps (~1.5 GB on Arc B580). Historically the alloc probe in `ggml_sycl_init` did a binary-search of `sycl::malloc_device`/`sycl::free` to converge on `safe_max_alloc_size` (~1593 MB observed) — but on patched libze 1.14.37435 every freed probe chunk lingers in the L0 USM pool's internal cache, eating residual VRAM and surfacing later as `ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` at first kernel launch (mislabeled as `UR_RESULT_ERROR_OUT_OF_RESOURCES` / `40` in SYCL exception text). Commit `28ef6cea1` removed the probe and switched to `safe_max_alloc_size = floor(max_mem_alloc_size * 0.95)` directly — empirically validated against the bench gate (PP512 ≥ 1700, TG128 ≥ 80) on patched libze. **Reverting to stock libze 1.14.37020 without restoring the probe will reintroduce the original silent-oversize-alloc-wedge mode** that the probe was designed to defend against — stock historically accepted oversized allocs that hung at runtime, while patched libze 1.14.37435 fails them cleanly so the direct formula is safe on the supported config.

### SYCL Device Selection

On multi-GPU systems, use `ONEAPI_DEVICE_SELECTOR` to choose the visible Level
Zero devices. Selector syntax is `backend:devices`: use `level_zero:0` for one
device, `level_zero:0,1` for a numeric multi-device set, or `level_zero:gpu` for
all Level Zero GPU devices. The `level_zero:gpu:0` strings printed by some tools
are display IDs, not valid selector values.

This system has 3 GPUs: Arc B580 (device 0), Arc Pro B50 (device 1), iGPU (device 2).

As of 2026-05-10 after the ECC-off reboot, the B50 initializes with
`ONEAPI_DEVICE_SELECTOR=level_zero:1`, reports `Memory ECC: Current disabled /
Pending disabled`, and can run the Mistral 7B Q4_0 completion gate. ECC-off is
a workstation performance setting with a reliability tradeoff: it raises B50
reported VRAM from ~14.3 GiB to ~16.3 GiB and improves Mistral 7B Q4_0 from
~1053 PP512 / ~40 TG128 to ~1197 PP512 / ~44 TG128.

GPT-OSS 20B now fits and runs on the B50-only path. The all-GPU
`level_zero:0,1` / `level_zero:gpu` path still hits
`UR_RESULT_ERROR_DEVICE_LOST` in `MUL_MAT`, so use a single-device selector for
GPT-OSS quick gates.

```bash
# List available devices
sycl-ls

# REQUIRED: Select Arc B580 (device 0) for stable operation
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench ...

# Select all Level Zero GPUs
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./build/bin/llama-bench ...

# NOTE: GGML_SYCL_VISIBLE_DEVICES=0 does NOT work - it filters at llama.cpp level
# but unified cache still sees all Level Zero devices. Use ONEAPI_DEVICE_SELECTOR.
```

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

### Performance Expectations (Arc Pro B50, ECC Disabled)

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
| `level_zero:1` B50 ECC-off | ~169 | ~17 | Fits in 16.3 GiB reported VRAM |
| `level_zero:0` B580 | ~66 | ~17 | Smaller VRAM budget causes more pressure |
| `level_zero:0,1` | fails | fails | `UR_RESULT_ERROR_DEVICE_LOST` multi-device L0 regression |

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
| `GGML_SYCL_PP_PIPELINE=1` | OFF | Enable double-buffered FP16 weight dequant prefetch. Correct after the layout-mismatch fix, but provides no measured throughput gain on Arc B580 (dequant + GEMM compete for the same compute engine — overlap doesn't materialize). Left opt-in for future overlap research. |

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
| `GGML_SYCL_UNIFIED_CACHE=0` | Disable unified weight cache entirely |
| `GGML_SYCL_UNIFIED_CACHE_MODE=<mode>` | Cache mode (auto, device, host, mmap) |
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
4. For ggml changes: Run `test-backend-ops` on multiple backends
5. Verify performance: `llama-bench` and `llama-perplexity` should not regress

### Triggering Heavy CI
Add `ggml-ci` to commit message to trigger extended CI workloads.

## Documentation

- **Build Details**: `docs/build.md`
- **Backend SYCL**: `docs/backend/SYCL.md`
- **Add New Model**: `docs/development/HOWTO-add-model.md`
- **Contributing**: `CONTRIBUTING.md` (coding/naming guidelines, PR process)
- **Copilot Instructions**: `.github/copilot-instructions.md` (cross-platform build/test patterns)
