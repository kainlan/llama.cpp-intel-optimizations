# `llama.cpp-fazjo`: what the `UR_RESULT_ERROR_OUT_OF_RESOURCES` actually is

This `OUT_OF_RESOURCES` is **not** the MoE placement planner failing.

It comes from the dense `mul_mat` streaming path for `blk.1.attn_output.weight`, not from expert placement, host arenas, or inference-time expert staging.

## What failed

The failing sequence is:

1. `mul_mat` tries to stream an mmap-backed dense weight through unified-cache DMA.
2. That enqueue fails with `UR_RESULT_ERROR_OUT_OF_RESOURCES`.
3. `mul_mat` catches that specific mmap-DMA failure and falls back to CPU graph execution.
4. The CPU fallback then tries to copy device-resident tensors back to host with `stream->memcpy(...).wait()`, and that enqueue also fails.

Relevant code:

- [ggml/src/ggml-sycl/unified-cache.cpp](/Apps/llama.cpp/ggml/src/ggml-sycl/unified-cache.cpp#L5662)
- [ggml/src/ggml-sycl/unified-cache.cpp](/Apps/llama.cpp/ggml/src/ggml-sycl/unified-cache.cpp#L5699)
- [ggml/src/ggml-sycl/unified-cache.cpp](/Apps/llama.cpp/ggml/src/ggml-sycl/unified-cache.cpp#L5709)
- [ggml/src/ggml-sycl/ggml-sycl.cpp](/Apps/llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp#L19848)
- [ggml/src/ggml-sycl/ggml-sycl.cpp](/Apps/llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp#L17843)

## Why the arenas did not prevent it

The important distinction is that the host-zone arenas only manage **host pinned allocations**.

This failing path allocates and uses temporary **device** DMA staging buffers:

- [ggml/src/ggml-sycl/unified-cache.cpp](/Apps/llama.cpp/ggml/src/ggml-sycl/unified-cache.cpp#L5576)

So "our unified cache + arenas + planning should automatically handle that" is only partially true:

- `host` arenas help with pinned host storage
- the failure here is on the `device` temporary streaming/submission path

That means the host arena logic never gets a chance to save this path.

## Why this is not the original MoE bug anymore

The MoE planning work is behaving correctly in this run:

- plan-host experts route through raw host AoS weights
- inference-time `direct_stage_expert()` is no longer being used
- the old MMVQ zero-copy miss path is bypassed under placement-plan mode

The log confirms that the later failure is on a regular dense tensor:

- `/tmp/fazjo-gptoss.log:6206` shows `blk.1.attn_output.weight`

That is a dense attention output weight, not an MoE expert tensor.

## What subsystem is actually incomplete

The remaining bug is that dense `HOST_MMAP` streaming is still an opportunistic runtime path outside the planner's full placement contract.

In practice:

- the planner currently fixes the MoE expert placement/staging problem
- dense mmap-backed weights can still hit the older on-demand DMA streaming path
- after that DMA OOR, CPU fallback still depends on the same SYCL queue being able to enqueue device-to-host copies
- that recovery path is not robust once the queue/backend is already in this failure state

## What likely needs to change

This is not mainly an "arena sizing" problem. The fix is likely one or more of:

1. Make dense `HOST_MMAP` weights participate in the same planned host/device placement model instead of ad hoc runtime streaming.
2. Make `stream_dma()` degrade earlier, before enqueue, when runtime/device reserve is too tight.
3. Make `ggml_sycl_cpu_fallback_graph()` avoid SYCL queue copies after a DMA OOR, or re-route through guaranteed host-resident sources first.

## Bottom line

Your expectation is directionally right, but the implementation is not complete yet.

Today:

- the planner covers the MoE expert path
- this `OUT_OF_RESOURCES` is coming from a separate dense streaming/fallback path
- that path is still outside the planner/arena guarantee
