# Task 5 — Mini-context + FA prototype validation — PASS

**Date:** 2026-04-24
**Plan:** [`docs/plans/2026-04-24-planner-validation-phases.md`](../../2026-04-24-planner-validation-phases.md) §Task 5
**Canary:** `tests/mini-context-prototype.cpp` → `build/bin/test-mini-context-prototype`

## Question

Does a "throwaway mini-context" — built from metadata-only weights
(`llama_model_params::no_alloc = true`) — produce per-backend buffer
sizes and FA auto-detect decisions byte-identical to a real context
on the same model + cparams?

If yes, A3a's design (sizing the unified-cache zones from a throwaway
pre-load mini-context) is mechanistically validated, and A3b's FA
auto-detect can run inside the same throwaway pass.

## Test design

Three workers, fork+exec'd from the driver to isolate state:

| Worker  | `no_alloc` | `use_mmap` | Purpose                                   |
|---------|-----------:|-----------:|-------------------------------------------|
| real-A  | false      | true       | Real context, baseline                    |
| real-B  | false      | true       | Real context, second identical run (determinism check) |
| mini    | true       | false      | Mini-context, metadata-only weights       |

Each worker creates a fresh `llama_model` then a `llama_context` with
`n_ctx=1024, n_ubatch=512, flash_attn_type=AUTO` and exits. The driver
captures llama log output on each worker's stderr and parses two
signal classes:

1. Per-backend compute buffer size
   `<func>: <BUFT_NAME> compute buffer size = <N>.NN MiB`
2. Flash-attention auto-detect verdict
   `<func>: Flash Attention was auto, set to enabled` / `disabled`

PASS criteria:
- (1) `real-A == real-B` — graph_reserve is deterministic across two
  consecutive real contexts
- (2) `mini == real-A` — the mini-context's signals are byte-identical
  to a real context's signals

Backend: CPU (`ONEAPI_DEVICE_SELECTOR=opencl:cpu`). Task 5's question
is scheduler-level, not SYCL-specific. CPU sidesteps the
`tiered_kv_buffer_clear` wedge currently filed as `llama.cpp-zhzbp`
on GPU.

## Implementation note: `no_alloc=true` mmap-path assert

Initial run of the canary tripped `GGML_ASSERT(!ml.no_alloc)` at
`src/llama-model.cpp:8686`. The mmap-buffer-from-host-pointer path
asserts that the loader is not in metadata-only mode (because that
path actually maps tensor data into a backend buffer). Fix: in the
mini worker, set `mparams.use_mmap = false` so the loader takes the
non-mmap path which **does** support `no_alloc=true`. This is a
canary-side workaround — A3a's production implementation will need
to make the same use_mmap choice (or extend the mmap path to handle
metadata-only loads).

## Per-model results

### Mistral 7B v0.1 Q4_0

| Worker | FA verdict | CPU buffer size |
|--------|-----------:|----------------:|
| real-A | enabled    | 115.01 MiB      |
| real-B | enabled    | 115.01 MiB      |
| mini   | enabled    | 115.01 MiB      |

✅ `real-A == real-B` (determinism)
✅ `mini == real-A` (mini-context approach works)

### GPT-OSS 20B MXFP4

| Worker | FA verdict | CPU buffer size |
|--------|-----------:|----------------:|
| real-A | enabled    | 398.38 MiB      |
| real-B | enabled    | 398.38 MiB      |
| mini   | enabled    | 398.38 MiB      |

✅ `real-A == real-B` (determinism)
✅ `mini == real-A` (mini-context approach works)

Captured per-variant JSON snapshots:
- Mistral 7B Q4_0: [`tests/data/planner-canaries/mini-context.json`](../../../../tests/data/planner-canaries/mini-context.json)
- GPT-OSS 20B MXFP4: staged out-of-band during the validation session
  (the canary's single-output-file design overwrites between runs).

## Overall finding

**PASS.** Both A3a (mini-context sizing) and A3b (FA auto-detect in the
mini-context pass) are mechanistically sound on the architectures
tested. The throwaway-context approach yields signals byte-identical
to a real context, so the unified-cache zone planner can be sized off
a mini-context's `graph_reserve` output.

## Recommendations

1. **A3a can proceed.** Production implementation should construct the
   mini-context with `no_alloc=true, use_mmap=false`, run
   `llama_init_from_model`, capture `backend_buf_exp_size` (the
   internal vector populated by `graph_reserve`), and tear down. Add
   a thin public accessor `llama_context_buft_sizes()` (returns a
   `vector<pair<buft_name, size_bytes>>`) so the planner doesn't need
   private headers.

2. **A3b can run in the same pass.** The FA auto-detect logic at
   `src/llama-context.cpp:570-610` runs inside `llama_init_from_model`
   regardless of `no_alloc`. Capture the resolved
   `cparams.flash_attn` after init and store it on the model.

3. **Extend the mmap-path assert OR document the use_mmap=false
   constraint.** Currently `no_alloc=true` is incompatible with the
   mmap-from-host-pointer fast path. Either:
   - **Path A** — update the assert to a runtime check that takes a
     non-mmap fallback when no_alloc=true, OR
   - **Path B** — document that A3a's mini-context construction must
     pass `use_mmap=false` (canary's current approach).

   Path B is simpler and has zero perf impact (the mini-context isn't
   on a hot path).

4. **Architecture-coverage caveat (carried from Task 2).** Validated on
   dense (Mistral 7B v0.1) and MoE (GPT-OSS 20B). Sliding-window-attention
   (Gemma 2, Mixtral) and state-space (Mamba/RWKV) models are absent
   from `/Storage/GenAI/models/` and remain a documented blind spot
   for the single-shape sizing claim.

## Reproducer

```bash
source /opt/intel/oneapi/setvars.sh --force
ninja -C build test-mini-context-prototype

# Mistral 7B Q4_0 (default)
LD_LIBRARY_PATH="/Apps/llama.cpp/build/bin:$LD_LIBRARY_PATH" \
  ONEAPI_DEVICE_SELECTOR=opencl:cpu timeout 600 \
  ./build/bin/test-mini-context-prototype

# GPT-OSS 20B MXFP4
LD_LIBRARY_PATH="/Apps/llama.cpp/build/bin:$LD_LIBRARY_PATH" \
  MISTRAL_PATH="/Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf" \
  ONEAPI_DEVICE_SELECTOR=opencl:cpu timeout 900 \
  ./build/bin/test-mini-context-prototype
```

EXIT=0 on PASS, EXIT=1 on FAIL.

## Related canaries

- D0.1 (`docs/plans/data/planner-canaries/d0.1-skeleton-determinism.md`):
  closely related — tests that `graph_reserve` returns deterministic
  sizes across multiple context constructions on the same model.
  This canary's `real-A == real-B` check is essentially D0.1 with the
  added FA-verdict signal.
- D0.3b (Task 1): validated op-id stability across repeated
  `graph_reserve` calls on the same context. Together with this
  Task 5 finding, the determinism story is: op-id sequence stable
  within a context (D0.3b), buffer sizes + FA verdict stable across
  context recreations (this canary).
