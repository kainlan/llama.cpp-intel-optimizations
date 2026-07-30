# Plan: retire device-indexed handle storage in `ggml_tensor_extra_gpu`

**Filed** 2026-07-30. **Status:** audit complete, implementation not started.
**Tracker:** see "Task breakdown" below.

## The objection

`ggml_tensor_extra_gpu` (`ggml/src/ggml-sycl/common.hpp:2842`) stores three
parallel device-indexed arrays:

```cpp
void *                data_device[GGML_SYCL_MAX_DEVICES];   // "legacy -- use data_handle"
ggml_sycl::mem_handle data_handle[GGML_SYCL_MAX_DEVICES];   // "Smart handles (P12 migration)"
size_t                data_device_size[GGML_SYCL_MAX_DEVICES];
```

Storing a `mem_handle` in a `[device]`-indexed slot conflicts with what a
`mem_handle` is. From `mem-handle.hpp:177-180` and `:199`:

> Resolve for a dispatch device. Device-resident pointers must belong to that
> device; host-resident pointers are returned for any device. **The handle's
> `device_` is the allocator/cache owner** used for re-resolution.

So the handle already owns *tier* placement -- device VRAM / host-pinned / mmap
-- and resolves per dispatch device on dereference. That is precisely the
transparency CLAUDE.md requires ("Handles must resolve location on dereference
so the cache can move data between tiers transparently").

Two symptoms in the code confirm the conflation is real, not stylistic:

1. `data_device_ptr()` must guard with
   `if (handle.device() == dev || handle.device() == HOST_DEVICE)`
   (`common.hpp:2855`). That check exists **only** because a handle's own device
   can differ from the slot it is filed under.
2. Immediately below it (`common.hpp:2858-2865`) sits a live warning for
   `data_handle[dev]` and `data_device[dev]` disagreeing. The code already knows
   its two sources of truth diverge; it logs instead of resolving.

## Audit results

Scope: 323 call sites -- `data_device[` 85, `data_handle[` 110,
`data_device_ptr(` 62, `set_data_device(` 27, `data_device_size[` 39.
Concentrated in `ggml-sycl.cpp` (58) and `common.hpp` (17).

### Finding 1 -- row-split is LIVE; the array cannot simply collapse

`src/llama-model.cpp:1220`:

```cpp
if (split_mode == LLAMA_SPLIT_MODE_ROW) {
    auto fn = ggml_backend_reg_get_proc_address(reg, "ggml_backend_split_buffer_type");
    if (fn) { auto * buft = fn(dev_index, tensor_split); ... }
}
```

Wired from llama core, reachable via `--split-mode row` / `--tensor-split`.
`ggml_backend_sycl_split_buffer_init_tensor` (`ggml-sycl.cpp:28618`) loops over
devices calling `set_owned_data_device(i, ...)` with **different row ranges per
device for the same tensor** (`:28684-28716`).

**Collapsing `data_handle[]` to one `mem_handle` would make `--split-mode row`
silently produce wrong results** -- it would keep only the last device's shard.
Not a crash. This is the hard constraint on the whole plan.

### Finding 2 -- Tensor Parallelism is NOT reachable from llama core

The fork's TP (`g_sycl_tp_config`) is often cited as a second reason the array
must stay per-device. It is not currently reachable:

- `g_sycl_tp_config.enabled = true` is set only in `ggml_sycl_tp_init()`
  (`common.cpp:1136`),
- called only from `ggml_backend_sycl_tp_buffer_type()` (`ggml-sycl.cpp:29719`),
- exposed as proc address `"ggml_backend_tp_buffer_type"` (`ggml-sycl.cpp:93829`),
- and **`src/` contains 0 references to that name**, versus 4 for the split
  equivalent.

So llama core never requests the TP buffer type, and the ~30
`g_sycl_tp_config.enabled && world_size > 1` guards are all false in normal
inference. TP is public API (`ggml-sycl.h:49`) and an out-of-tree consumer could
call it, so it must not be deleted casually -- but it must **not** be used to
justify complexity in the hot path either.

Consequence: **one live constraint (row-split), not two.**

### Finding 3 -- the migration's only safety net is inert by default

The `data_handle`/`data_device` mismatch warning (`common.hpp:2858-2865`) and its
sibling in `install_direct_slice_storage` (`:2954`) are gated on
`g_ggml_sycl_debug`, which defaults to 0 (`ggml-sycl.cpp:320`). No canonical gate
in CLAUDE.md sets `GGML_SYCL_DEBUG=1`.

**Divergence could be occurring in every production run and nobody would know.**
Whether it actually occurs is unknown -- confirming needs a real run with debug
enabled, which the audit deliberately did not do.

### Finding 4 -- unchecked `data_device_ptr()` on the live split path

`data_device_ptr()` returns `nullptr` when the legacy fallback finds a `DEVICE`
allocation whose `device_id != dev` (`common.hpp:2874-2881`). Most of the 62 call
sites check. These do not:

| site | branch | reachable today | risk |
|------|--------|-----------------|------|
| `ggml-sycl.cpp:33394` | **split** buffer, `ggml_sycl_cpy_tensor_2d` | **YES** (`--split-mode row`) | `src_ptr` unchecked -> `x = src_ptr + i1_low*nb1 + ...` (`:33425`) -> `stream->memcpy` (`:33428`). Null-plus-offset inside a device memcpy. |
| `ggml-sycl.cpp:33404` | TP buffer, same function | no (Finding 2) | latent, identical shape |
| `ggml-sycl.cpp:34474` | TP branch of mul_mat dispatch | no (Finding 2) | latent. Note the **non-TP branch below it (`:34483-34537`) is markedly more careful** -- it resolves through `data_handle[i]` with device/layout validation and fallback tiers. The TP branch is a visible shortcut around that care. |
| `ggml-sycl.cpp:29089`, `:29248`, `:29334` | TP init | no (Finding 2) | latent; null `tensor->data` fails loudly |

Only `:33394` is live. Whether its nullptr branch is *reachable in practice*
(i.e. whether any caller legitimately requests a device other than where the
legacy pointer was registered) was **not** determined -- it may be latent too.

### Finding 5 -- what is already correct (do not "fix" these)

- **MoE expert pointer table** (`ggml-sycl.cpp:42896-43460`): stores raw
  `void*[]` because that is the kernel ABI, but pairs it with
  `moe_expert_handles[device]` + `moe_expert_ptrs_leases[device]`, retained via
  `ggml_sycl_retain_moe_ptr_table_leases_until_event()` until the event
  completes, and is guarded by a regression test
  (`test_moe_ptr_table_does_not_persist_pointer_cache`, `:10816`). This is the
  documented carve-out working as intended.
- **`g_data_ptr_cache`** (`ggml-sycl.cpp:15005-15060`): keyed by
  `{tensor, device}` and stores `mem_handle`, resolved fresh per read. Complies
  with "key on stable identity, not raw address".
- **Paired writes**: every write site found updates `data_device[dev]` and
  `data_handle[dev]` together. No write to one field without the other was
  found. The duplication is verbose, not divergent.

## Plan

Ordered so each stage is independently valuable and independently revertible.
**No stage after 0 should start before the one before it has landed and passed
gates**, because these are all in the memory core.

### Stage 0 -- make the invisible visible (cheap, do first)

0.1 **Add the missing null check at `ggml-sycl.cpp:33394`.** One `if`, live
    split path, converts a possible device-side fault into a clean error. Do
    `:33404` at the same time; identical shape, costs nothing.

0.2 **Make the mismatch warning observable without `GGML_SYCL_DEBUG=1`.**
    Rate-limited `GGML_LOG_WARN` (it already has a `stale_raw_warns` counter),
    or a dedicated `GGML_SYCL_HANDLE_STRICT=1` that aborts. Without this the
    whole migration proceeds blind.

0.3 **Run once with divergence detection on** and record the result on the
    tracker. This decides whether the migration is fixing a latent design smell
    or an active correctness bug. **Do not skip this** -- everything downstream
    is scoped by the answer.

Gate: Mistral completion gate + `-R unpin-event` + `-L profiling`. No
throughput claim needed; Stage 0 is not a perf change.

### Stage 1 -- separate the two populations at the API

Introduce an explicit split in the accessor surface so the *common* case stops
carrying the split case's shape:

- a scalar accessor for the single-device majority (~200+ sites), which asserts
  the tensor is not split-backed, and
- an explicit per-device accessor used **only** by the split path.

This is mechanical and behaviour-preserving: the storage stays as-is. The point
is to make every call site declare which population it belongs to, so Stage 2
can act on that declaration. Migrate call sites in tranches by file, gating each
tranche.

### Stage 2 -- collapse storage for the non-split population

Once Stage 1 has classified the call sites, move non-split tensors to a single
`mem_handle` and confine `[GGML_SYCL_MAX_DEVICES]` storage to split-backed
tensors -- e.g. a side allocation present only when the tensor's buffer type is
the split type. This is where the `handle.device() == dev` guard and the
mismatch warning become deletable, because the two axes are no longer conflated.

### Stage 3 -- retire `data_device[]` as a source of truth

Delete the legacy raw array, leaving `mem_handle` as the single source of truth
and `data_device_ptr()` as a thin `resolve()`. Only safe after Stages 1-2; until
then the raw pointer is load-bearing for the "inlined shim" readers
(`common.hpp:3530-3553`, `ggml-sycl.cpp:14990-14999`, `:15084-15093`,
`:27391-27399`, `:39312-39348`), which must be migrated first.

### Stage 4 -- decide TP's fate (separate decision, not a prerequisite)

TP is unreachable from core but is public API. Options: (a) keep and document as
out-of-tree-only, (b) wire it up properly, (c) deprecate and remove. **This is a
product decision, not an implementation detail** -- it should be made
explicitly rather than settled by accident during Stages 1-3. Note that if TP is
removed, findings 4's latent rows disappear with it.

## Explicitly out of scope

- Changing `mem_handle`'s own semantics. `device_` meaning "allocator/cache
  owner" and `resolve(dispatch_device)` handling tier placement is the *correct*
  design; this plan brings the storage in line with it, not the reverse.
- The MoE expert pointer table and `g_data_ptr_cache` (Finding 5) -- already
  correct.

## Method notes for whoever picks this up

- **codescout's index silently skips `ggml/src/ggml-sycl/ggml-sycl.cpp`**
  (oversize, ~60k lines): `search_text` returns
  `skipped: {reason: "oversize"}` and omits real matches. That file holds 58 of
  the 85 `data_device[` uses, and the mismatch-warning string was findable *only*
  via `cat ggml/src/ggml-sycl/ggml-sycl.cpp | grep -n`. A conclusion drawn from
  codescout alone in that file is a false negative.
- Both audits behind this plan were grep-driven, not end-to-end reads. Sites
  reaching these fields through an alias or reference rather than the literal
  text would be invisible. Treat the 323 count as a floor.
- ⚠️ **Do not loop `test-llama-archs`.** A single run is safe; looping it drove
  shmem to ~227 GB of 255 GB and caused two global OOMs on 2026-07-30. See
  CLAUDE.md (`b7886ed9c`).

## Task breakdown

| stage | scope | prerequisite |
|-------|-------|--------------|
| 0.1 | null checks at `:33394`, `:33404` | none |
| 0.2 | make divergence detection observable | none |
| 0.3 | run + record whether divergence occurs | 0.2 |
| 1 | split the accessor API, migrate call sites in tranches | 0.3 |
| 2 | collapse non-split storage to a scalar handle | 1 |
| 3 | delete `data_device[]` | 2 |
| 4 | decide TP: keep / wire / remove | independent |
