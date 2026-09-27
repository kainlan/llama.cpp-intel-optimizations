# SYCL file-mapped host tier (mmap as a unified-cache tier)

*Status: DESIGN rev 1, not implemented. Tracker: `llama.cpp-5efe`. Written
2026-09-27 against branch `task/5efe` at `69e70c715` (= master). Every cite
without a revision prefix is to that tree. `gs` is
`ggml/src/ggml-sycl/ggml-sycl.cpp`, `uc` is
`ggml/src/ggml-sycl/unified-cache.cpp`, `uch` is `unified-cache.hpp`, `lml`
is `src/llama-model-loader.cpp` and `lm` is `src/llama-model.cpp`. Load-phase
evidence is the lead's Qwen3.8 baseline `qwen-base/run2.log` (session
scratchpad, `fefb92980`, B70 alone, `-lm mmap -lzm on -c 4096 -v`), whose
load-path code is byte-identical to this tree in every region cited here.*

Sibling designs this one interfaces with, and does not restate:
`llama.cpp-th32` (STORAGE tier for lazy row tables, `docs/design/sycl-storage-tier-lazy-tables.md`
rev 2o at `2a2b85a4d`, abbreviated **th32**), `llama.cpp-7v5c` (host-only plan
dry-run), `llama.cpp-rnh3` (exact host WEIGHT bytes), `llama.cpp-sgc5` (one
zone set per plan), `llama.cpp-yl3v` (cold experts on STORAGE) and
`llama.cpp-qptd` (`-ngl 0` loses mmap).

---

## 0. Summary

The owner asked why SYCL disables mmap. There are **two** switches, not one:

1. **The caps slip** (the known root cause). Upstream `153d324bc` added
   `ggml_backend_dev_caps::mmap_support`. The fork's
   `ggml_backend_sycl_device_get_props` lists four initializers
   (`gs:107797-107802`), so the fifth is value-initialized to `false`, and
   `lm:2245-2265` then resolves `-lm auto` to `none` and `-lzm auto` to `off`
   on every SYCL load, discrete cards included.
2. **The fork's own override.** `lm:2693-2696` sets `ml.use_mmap = false` for
   every SYCL model *after* `create_tensor`, whatever `-lm` says ("disabling
   mmap for SYCL weight layout upload", added by `0f475d1aa`). The baseline
   shows it firing after the loader printed `load_mode = mmap`
   (`run2.log:314` then `:2902`).

Even with both removed, mmap would change nothing that matters, because of a
third fact: with the default `GGML_SYCL_WEIGHTS_EVICTABLE` (auto, `true`,
`gs:13591-13614`) the loader places **every** SYCL-layer weight in the
`SYCL_Host` buffer type, which is pinned USM from the unified cache
(`lml:1128-1135`, `:1400`; `gs:42627-42707`). mmap would only change where
the copy into that pinned buffer reads from. That is why the baseline pinned
110 GB and was killed at `MemAvailable` 14 GB.

This design therefore does three things:

- **Commit 1** restores `mmap_support = !host_unified_memory` (§3). By itself
  that moves one thing: `-lzm auto` engages again, so the 50.66 GiB Qwen PLE
  table leaves the pinned pool under default flags.
- **A new allocation tier, `FILE_MAPPED`** (§4): a read-only file mapping that
  the unified cache creates, owns through an `alloc_owner`, and releases with
  `munmap`. A new per-device buffer type, `SYCL_Mapped`, replaces `SYCL_Host`
  as the weight buffer when mmap is on. Host-resident weights that the CPU
  executes in file layout are served from the mapping. Device-resident
  weights are copied from it through a transient pinned window, so **the
  double residency of every VRAM weight (pinned host + VRAM) ends in mmap
  mode** (§6).
- **Planner and budget** (§4.5): the plan gets a host tier per entry
  (`PINNED` or `FILE_MAPPED`), mapped bytes are reported but not charged to
  the host ledger, `-lm mmap+mlock` makes them charged, the host-fit refusal
  is th32 T2's over 7v5c's computation (5efe supplies the terms), and the
  ledger's headroom becomes the P3 floor: `max(29 GiB, total/32)` instead of
  `max(8 GiB, total/32)`.

**Acceptance** (lead-run, §7): Qwen3.8-Flash-Next Q8_0 loads on the B70
(`level_zero:0`) under default load flags with `AnonPages` growth under
30 GiB, `MemAvailable` never under 25 GiB, most of the model in page cache,
and shard 3 (the PLE table, 54.4 GB) not resident after the load.

---

## 1. Scope and principles

- **P1, the cache owns the bytes.** Today the fork has no cache-owned file
  mapping. A weight in a `CPU_Mapped` buffer (upstream's mmap path, the lazy
  tables) sits outside the contract (canonical contract §8 items 2 and 4), and
  host experts are registered as `non_owning_external_host` slices the cache
  never frees (`uc:8150`, `:12351`). The tier fixes both for mapped weights.
- **P2, the handle owns the lifetime.** Every reader of a mapped byte holds a
  `mem_handle` whose last release unmaps (§4.2).
- **P3, placement decides the executor; no GPU zero-copy of host memory.** A
  mapped byte is read by the CPU only. The GPU receives mapped bytes only as a
  copy through a pinned window (§5.3).
- **P4, plan equals reality, and no idle reservation.** The plan says which
  bytes are pinned, which are mapped and which are neither, and the ledger
  charges exactly the resident ones (§4.5).
- **Out of scope here:** the lazy row tables themselves (th32 owns them; §4.6
  only guarantees the tier never maps, prefetches or locks them), the cold
  expert set (yl3v), the `-lm none` double residency (a ticket, §6), and
  `CPU_REPACK` buffers living outside the cache (an existing P1 gap, a ticket,
  §4.4).

---

## 2. The load path today (item 1)

### 2.1 How `-lm` and `-lzm` resolve

- `llama_model_loader` sets `use_mmap` for `mmap`, `mmap+mlock` and `auto`
  (`lml:599`).
- `load_tensors` then drops it under `auto` if any model device reports
  `mmap_support = false` (`lm:2245-2254`), and turns lazy `auto` into `off` on
  the same test (`lm:2256-2265`). The fork's SYCL devices all report `false`
  (§0), so the default command line loads with no mmap and no lazy tables.
- An explicit `-lm mmap` survives that check, and the loader prints
  `load_mode = mmap` (`lm:2268-2272`, `run2.log:314`).
- `create_tensor` runs next, with `use_mmap` still true. Then `lm:2693-2696`
  clears it for every SYCL model (`sycl_model_backend` is "a SYCL load
  transaction exists and some context uses a SYCL buft", `lm:2535-2542`), and
  `init_mappings` runs after that (`lm:2702`). So `load_all_data` never takes
  its mapping branch for an ordinary weight on SYCL, whatever `-lm` says.

### 2.2 Which buffer type every weight gets

| weight | buffer type chosen | cite |
|---|---|---|
| any tensor of a SYCL layer, input or output (default `weights_evictable() == true`) | `SYCL_Host` of that layer's device | `select_weight_buft(prefer_host_weights)`, `lml:1128-1135`; the flag, `lml:1396-1401`; `weight_buft_supported` returns true for it before any op check, `lml:978-985` |
| planner-host dense weight (`planned_target_device == -1`) | `CPU_REPACK`, or plain `CPU` under `GGML_SYCL_HOST_WEIGHT_LAYOUT=plain` (llama.cpp-ir18) | `lml:1473-1495` |
| `-ot <pat>=CPU` | `SYCL_Host` again, reported as not honoured (llama.cpp-lufn) | `lml:1406-1440` |
| `TENSOR_READ_LAZY` tensor chosen by `lazy.add` | the CPU device's default buft | `lml:1368-1370`, `:1232-1238`, `:1774-1777` |
| MoE `*_exps` | `SYCL_Host`; the plan has no name-indexed entry, so ir18 does not touch them | `lml:1469-1471` |

Upstream's "avoid a host buffer under mmap" demotion (`lml:1511-1517`) is
disabled for SYCL by `allow_host_buft_with_mmap` (`lml:1503-1511`), so even a
surviving `use_mmap` keeps the weights in `SYCL_Host`.

`make_cpu_buft_list` puts the first device's host buft at the front of the CPU
list unless `--no-host` (`lm:1765-1779`), which is `llama.cpp-qptd`'s finding:
at `-ngl 0` the weights still land in `SYCL_Host`.

### 2.3 Where the bytes land

**Host-resident weights land in pinned USM, copied, never in mapped pages.**
`SYCL_Host` allocates through `unified_alloc` with role `WEIGHT`, category
`HOST_COMPUTE`, `must_host_pinned` (`gs:42640-42665`), which routes to the host
WEIGHT zone (`uc:15361-15373`). `load_all_data` fills it with `read_raw`
straight into `cur->data` because the buffer is host (`lml:2134-2141`). With
mmap it would have been a `memcpy` from the mapping (`lml:2131`).

**Device-resident weights get their bytes from that pinned buffer, at load
end.** `ggml_backend_sycl_model_load_end` runs `ggml_sycl_preload_model_weights()`
(`gs:12476`, defined `:33515`), which takes `src_ptr = tensor->data`
(`gs:34763`, `:35662`) and calls `unified_cache::direct_stage_weight` (`uc:6036`)
to fill a fresh VRAM allocation. The "tiered" override that could have
replaced `src_ptr` is dead: `get_cached_tensor_ptr` always returns `nullptr`
(`gs:19236-19261`). Nothing releases the pinned copy afterwards, and the
published entry records `src_ptr` into it (`uc:6213`). **Every VRAM weight is resident twice** for the
model's lifetime (lead's trace report, point 1).

**Host experts are registered in place.** `expert_aos = src0->data +
e * expert_size` (`gs:33437`) is registered by `register_host_expert` with no
owner, so `non_owning_external_host = true` (`uc:8150`) and the cache never
frees it (`uc:12351-12353`). A second pinned allocation happens only when a
CPU reorder is needed (`register_host_aos_expert`, `gs:34142`, alloc at
`:34176`).

**`CpuExpertPool` reads that pinned slice.** Each task carries the host
lease resolved from the registered entry (`gs:76020-76028`) and the pool
executes it (`gs:76049-76067`). `ggml_sycl_host_data()` is `tensor->data`
(`common.hpp:4889-4891`), which is the SYCL_Host buffer.

### 2.4 `use_mmap_buffer` and `buffer_from_host_ptr` in this fork

`use_mmap_buffer` is the constant `true` (`lm:2241`). The mapped branch
requires `caps.buffer_from_host_ptr` **and** the buft being its device's
*default* buft (`lm:2740-2748`). The SYCL device reports `false`
(`gs:107801`) and its `buffer_from_host_ptr` hook is a stub returning
`nullptr` (`gs:107848-107856`); `SYCL_Host` is not a default buft anyway. The
only buffers built from a mapping are the CPU device's (`CPU_Mapped`,
`ggml-cpu.cpp:398-400`), i.e. plain-CPU weights under `use_mmap` and the lazy
tables. After the load `pimpl->mappings` keeps each `llama_mmap` alive for the
model's lifetime (`lm:2866-2870`), and the last `load_all_data` call trims the
unused head and tail of each mapping (`lml:2236-2247`).

### 2.5 `TENSOR_READ_LAZY` on SYCL (`-lzm on`, Qwen `per_layer_token_embd`)

- `qwen4exp` creates the table with `TENSOR_READ_LAZY` (`src/models/qwen4exp.cpp:189`).
  `lazy.add` accepts it under `on` at any size, and under `auto` only above
  4 GiB (`lml:1240-1268`); `run2.log:370` shows `lazy read enabled` for
  51880 MiB.
- Its buffer type is the CPU default buft (§2.2), so the context is a lazy
  CPU context. `init_mappings` maps every file whenever `lazy.any()`, even
  after `lm:2693` cleared `use_mmap` (`lml:1869-1872`), with prefetch 0
  (`lml:1888`). The mapping is advised `POSIX_MADV_RANDOM` over the lazy
  ranges and excluded from `WILLNEED` and `MADV_HUGEPAGE`
  (`src/llama-mmap.cpp:503-545`). `MAP_POPULATE` is gone (`:479-481`).
- `load_tensors` wraps it as `CPU_Mapped` (`is_lazy_mapped`, `lm:2745-2748`),
  and `load_all_data` points the tensor into the mapping without reading it,
  and never mlocks it (`lml:2099-2128`).
- It is not a SYCL inventory entry: the late inventory filters non-SYCL bufts,
  and the late plan's host total drops from 154263.6 to 102383.5 MB
  (`run2.log:308` vs `:2821`, a difference of exactly 51880.1 MB). Its
  `get_rows` runs on ggml-cpu, faulting rows in from the file at the storage
  rates th32 §5.1 measured.
- The table is the only tensor in shard 3 of 6 (offset 192, 54400261120 B,
  read from the GGUF headers). §7's residency gate uses that.

### 2.6 The baseline's second 104 GB

The lead asked whether `Host staging zone ... per_inflight=103762.3 MB`
(`run2.log:243`) is a second reservation. It is not. The STAGING zone is
`max(2 x max_tensor, max_staging_pair) + 2 MiB` per in-flight slot
(`uc:27674-27708`). The **early** plan (`lm:2230`, before `create_tensor`)
still counts the lazy table as `max_tensor`: 2 x 51880.1 + 2 = 103762.3 MB.
The late plan, which no longer sees it, prints 1702.0 MB (`run2.log:2756`),
and the zones actually configured carry `STAGING=1719.2 MB`
(`run2.log:2893`). So it is a plan-time figure that was never committed. Th32
§4.1 already requires lazy tables to be excluded from `max_tensor`; its T3
fixture should cover the early plan too.

### 2.7 The legacy "mmap" vocabulary

None of it refers to a `llama_mmap` page today:

- `alloc_tier::MMAP_TRACKED` (`uch:5593`) is what `query_location` returns for
  a pointer the registry does not know ("assume host/mmap", `uc:21649-21655`),
  and the default of `memory_location::tier` (`uch:5614`). `unified_alloc`
  refuses it (`uc:15125-15127`), and `unified_free_record` returns without
  freeing (`uc:14812-14814`).
- `cache_location::HOST_MMAP` (`uch:2170`) is derived from that
  classification when an entry is registered (`uc:8068-8081`, `:8235-8248`),
  and from the direct-stage last resort "host arena exhausted, raw mmap
  fallback" (`uc:6504-6545`).
- `alloc_type::MMAP` in the registry is used for the tiered KV buffer's
  synthetic, `std::vector`-backed base (`gs:39631-39639`), i.e. "not USM".

### 2.8 The host budget today

`[HOST-ARENA]` is `min(total x 90%, MemAvailable - max(8 GiB, total/32))`
(`uc:4200-4231`), the same function th32 §4.2 lifts into
`host_ledger_budget()`. The planner has only `placement_entry::on_device`, so
"host" means pinned, and `weight_host_bytes` is never compared with any
budget (lead's host-budget report; th32 §2.3). No load-time host-fit gate
exists: the baseline committed 139264 MB of pinned chunks (`run2.log:2900`)
and kept growing.

---

## 3. Commit 1: the caps restore (item 2)

```cpp
// gs:107797, ggml_backend_sycl_device_get_props
ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *) dev->context;
props->caps = {
    /* .async                 = */ async,
    /* .host_buffer           = */ host_buffer,
    /* .buffer_from_host_ptr  = */ false,
    /* .events                = */ events,
    /* .mmap_support          = */ !ggml_sycl_info().devices[ctx->device].host_unified_memory,
};
```

- **The iGPU detection already exists.** `ggml_sycl_device_is_host_unified()`
  queries `sycl::info::device::host_unified_memory` (`common.cpp:827-833`),
  and `ggml_sycl_info()` caches it per device (`gs:25195`, field at
  `common.hpp:1477`). `ggml_sycl_info()` is initialized before any SYCL device
  object exists, because registration iterates it (`gs:110187`), so
  the cached field is safe here. This matches CUDA's `type != IGPU`
  (`ggml-cuda.cu:5130`) and Vulkan's `!is_integrated_gpu`
  (`ggml-vulkan.cpp:15221`). Upstream SYCL sets a flat `true`
  (`694ec2354 ggml-sycl.cpp:6380`); the fork keeps the iGPU off because of its
  own history (llama.cpp-403s).
- **What changes under `-lm auto -lzm auto`**, for a device set of discrete
  cards only (`level_zero:0`, `:1`, `:0,1`):
  - `ml.use_mmap` stays true through `lm:2245-2254`, and the log reads
    `load_mode = mmap` instead of `none`.
  - Lazy `auto` stays `auto`, so every `TENSOR_READ_LAZY` tensor above 4 GiB
    becomes a lazy table (§2.5). On Qwen that takes the 50.66 GiB PLE table out
    of the pinned pool **under default flags**. This is the only material
    change of commit 1.
  - `create_tensor` sees `use_mmap = true`: the `-ot =CPU` mmap warning can
    print (`lml:1434-1439`), and the host-buft demotion stays disabled for SYCL
    (§2.2). Buffer types are otherwise identical.
  - `lm:2693` still clears `use_mmap`, so every non-lazy weight is still read
    into `SYCL_Host`. Commit 1 deliberately does **not** remove that line:
    without the tier, the only effect would be a `memcpy` from page cache into
    the same pinned buffer, doubling the transient footprint.
- **An unpinned selector** that includes the iGPU makes the iGPU a model device
  and both fallbacks fire exactly as today. That is the intended behaviour, and
  it is never run on this host (CLAUDE.md, llama.cpp-403s).
- **th32 drops its own flip** (lead ruling §T13, point 1). th32's T6 still
  resolves its lazy mode before the `auto` fallback; commit 1 does not depend
  on that.

---

## 4. The design: a cache-owned `FILE_MAPPED` tier (item 3)

### 4.1 The tier and its object

`alloc_tier` gains `FILE_MAPPED`, distinct from th32's `HOST_CPU`:

| tier | memory | reader | charged to the host ledger |
|---|---|---|---|
| `DEVICE_VRAM` | device USM | that device | no (VRAM budget) |
| `HOST_PINNED` | USM host, pinned pool | CPU; a device for DMA staging only | yes |
| `HOST_CPU` (th32) | anonymous, `mlock`ed | CPU only | yes |
| **`FILE_MAPPED`** (new) | `mmap(PROT_READ, MAP_SHARED)` of a GGUF byte range, made by the cache | **CPU only** | **no**, unless locked (§4.5) |
| `UNREGISTERED` (renamed from `MMAP_TRACKED`) | not an allocation: the query result for a pointer the registry does not know | nobody may allocate it | n/a |

- **The raw wrapper.** `unified_cache_raw_map_file(path, file_size, offset,
  length)` opens the file `O_RDONLY | O_CLOEXEC`, `fstat`s it and refuses if
  the size differs from the size the loader recorded
  (`register_file_identity`, `lml:1553-1556`), maps page-aligned, and closes
  the fd (the mapping keeps the file referenced).
  `unified_cache_raw_unmap_file(ptr, length)` is its only release. Both are
  added to the canonical contract §3 allowlist, next to
  `unified_cache_raw_malloc_host`, so no other SYCL file calls `mmap` for
  weights (P1).
- **The owner-first front door.** `unified_allocate_file_mapping(req)` mints
  the `alloc_owner_control` first, like `unified_allocate_owner`, then maps.
  The class is **`EXTERNAL_EXACT`**: exact, independently released, not a
  slice of cache backing (canonical contract §3.1). The two-site
  `CACHE_BACKING` allowlist is untouched.
- **Release.** `release_physical` reaches `unified_free_record` (`uc:983-993`,
  `:14806`), which gains a `FILE_MAPPED` case that calls the unmap wrapper.
  (It must not reuse `MMAP_TRACKED`, whose case at `uc:14812` returns without
  freeing.)
- **`memory_location::device_mappable()`** (th32 §3.1 adds it) is false for
  `FILE_MAPPED`. `host_accessible()` stays true (the CPU reads it).
- **Identity.** One `file_mapping` per (model, file index, device buffer),
  keyed by the file identity the loader already publishes (path, size,
  `lml:1553-1556`). Two models mapping the same file get two mappings over one
  page cache. Sharing them is not needed.

### 4.2 Ownership and teardown

- **Who holds references.** The `SYCL_Mapped` ggml buffer (§4.3) holds one for
  the model's lifetime. Every cache entry registered over mapped bytes holds
  one as its `storage_owner` (the field exists, `uch:2514`), which replaces
  today's `non_owning_external_host` registration for these entries. Every
  `CpuExpertPool` task holds one through its `weight_lease`, which already
  moves into the worker (`gs:76027`).
- **Order at unload.** `llama_model` frees its buffers, the lifecycle unload
  releases the model's cache entries (canonical contract §12), and in-flight
  CPU tasks drop their leases when they finish. The last reference unmaps.
  Nothing waits for anything, and nothing unmaps under a reader.
- **Independent of `llama_mmap`.** The cache makes its own mapping instead of
  adopting the loader's, for two reasons. Adoption would leave the release in
  llama's hands (P1). And `load_all_data`'s final call `munmap`s the head and
  tail of each loader mapping (`lml:2236-2247`), which would pull pages from
  under a cache entry. Two mappings of one file share the page cache, so the
  cost is address space only.
- **The loader's own mapping on the SYCL path.** `init_mappings` is still
  called for lazy tables (until th32 lands) and for CPU-device buffers. On a
  SYCL load it passes prefetch 0 for every file, because the tier owns the
  prefetch policy for its ranges (§4.6).
- **Shutdown.** A live `FILE_MAPPED` control at cache teardown is a leaked
  lease, reported by the pre-teardown census like any `EXTERNAL_EXACT` control
  (canonical contract §3.1). The `__cxa_finalize` abandon path (`uc:996-1007`)
  is safe: process exit unmaps.
- **`PER_DEVICE` caches.** A `SYCL_Mapped` buffer belongs to one device, like
  `SYCL_Host` today (`gs:42761-42785`). Its mapping is minted by that device's
  cache and covers only that buffer's tensors. Another device's cache may hold
  leases on it (handles are refcounted across caches).

### 4.3 The buffer type, and the loader change

`ggml_backend_sycl_mapped_buffer_type_for_device(dev)` returns a per-device
buft named `SYCL_Mapped`, with its own `get_name`, since
`supports_buft` keys on the name function (`gs:108779-108812`):

- `is_host` true, so the CPU backend can execute an op on it when SYCL
  declines (the planner-host decline, `ggml_sycl_op_is_planned_on_host`).
- `supports_buft` true for SYCL devices, exactly as for `SYCL_Host`
  (`gs:108810-108812`), so ops on device-materialized weights stay on SYCL and
  resolve the cache's VRAM copy by identity, as they do today.
- `set_tensor`, `memset_tensor` and `clear` abort naming the tensor: the
  mapping is read-only, and weights are never written after load.
  `init_tensor` registers the host weight the way `SYCL_Host` weights are
  registered today (`lml:1542-1561`).

The buffer is created by a new proc-address hook,
`ggml_backend_sycl_mapped_buffer_from_file(dev, file_idx, path, file_size,
first, last, lazy_ranges)`, which the loader calls **instead of**
`ggml_backend_alloc_ctx_tensors_from_buft` for each (`SYCL_Mapped`, file)
pair when `use_mmap` is on. It mirrors upstream's `buffer_from_host_ptr`
shape (`lm:2748-2767`): `[first, last)` is `get_mapping_range` over the
context's tensors (`lml:1908-1923`), and `load_all_data` points each tensor at
`base + (weight->offs - first)` without reading it (its mapping branch,
`lml:2099-2128`, keyed on the buffer instead of on `ml.mappings`).

`create_tensor` picks `SYCL_Mapped` wherever it picks `SYCL_Host` today
(§2.2), when `use_mmap` is on. `lm:2693-2696` is then deleted. With
`-lm none`, `SYCL_Host` remains, with today's behaviour.

### 4.4 Which weights stay mapped (the placement rule)

A byte stays mapped only if **the CPU executes it and its materialized layout
is the file layout** (P3 with layout-follows-residency). Everything else is
copied at load into a cache allocation.

| weight | executor | materialized layout | tier |
|---|---|---|---|
| device-planned dense or expert | GPU | per-device (SOA, XMX, WOQ...) | `DEVICE_VRAM`, staged from the mapping (§5.3); the mapped pages are page cache afterwards |
| host MoE expert, AOS | CPU (`CpuExpertPool`) | AOS = file | **`FILE_MAPPED`, served in place** |
| host MoE expert needing a CPU reorder (`register_host_aos_expert`'s SOA/XMX branch) | CPU | reordered | copied; into `HOST_CPU` once th32 T1 lands, pinned until then |
| planner-host dense, `GGML_SYCL_HOST_WEIGHT_LAYOUT=cpu` (default) | ggml-cpu | `CPU_REPACK` | copied by ggml-cpu, outside the cache (ticket below) |
| planner-host dense, `=plain` | ggml-cpu | file | `FILE_MAPPED` (`is_host`, so the CPU backend runs it) |
| `token_embd` (CPU `get_rows`) | CPU | file | `FILE_MAPPED` |
| `TENSOR_READ_LAZY` table | CPU / th32 gather | row-gathered | **never `FILE_MAPPED`**: never mapped, prefetched or locked by the tier (§4.6); th32's STORAGE or `HOST_CPU` |

Under P3 no host **weight** needs pinned memory: the GPU never reads one. So
in mmap mode the pinned pool holds only staging windows, KV, scratch and
compute buffers. `PINNED` for a weight survives only in `-lm none`.

**Ticket to file:** `CPU_REPACK` weights are allocated by ggml-cpu's
`ggml_backend_cpu_repack_buffer_type` (`ggml-cpu/repack.cpp:5238-5250`),
outside the unified cache and its ledger. That is a P1 gap independent of
mmap (related to `llama.cpp-qptd`). The natural fix is a `HOST_CPU`-backed
repack buffer after th32 T1.

**Planner representation** (the lead's point (a)). `placement_entry` gains
`host_tier { NONE, PINNED, FILE_MAPPED, HOST_CPU }`, set for every entry with
`on_device == false`: `FILE_MAPPED` when the load maps files (the resolved
`use_mmap`, passed through the inventory) and the table above says so,
`HOST_CPU` or `PINNED` otherwise. The plan gets `host_mapped_bytes` (the sum of
`FILE_MAPPED` entries) beside `weight_host_bytes`, which from then on sums
only charged host weights. `[PLACEMENT] Totals` prints both. Only the planner
sets the tier; the loader's buft choice reads it, never re-derives it.

### 4.5 The budget, the ledger and the host-fit gate

**What the ledger charges.** th32 §4.2 makes `host_ledger_budget()` the one
host budget and has every resident host byte charge it. `FILE_MAPPED` bytes
are clean page cache: the kernel can drop and re-read them, and
`MemAvailable` counts them as available. So:

- **`-lm mmap` / `auto` (unlocked):** mapped bytes are **not** charged. They
  appear in the plan as `host_mapped_bytes`, and the load prints them.
- **`-lm mmap+mlock` (locked):** the tier `mlock`s its host-executed ranges
  (never a lazy range, as `lml:2119-2123` already does for the loader), and
  those bytes **are** charged, at full size, because locked pages are no
  longer reclaimable. A lock failure refuses the load with `errno`, the bytes
  and `RLIMIT_MEMLOCK`, as th32's `host-cpu-lock-failed` does. No new
  environment variable: the policy is the existing load mode.

**The resident working-set check.** An unlocked mapping keeps its pages only
while memory allows. The plan predicts it, with P3's terms:

```
resident_room = MemAvailable_pre - expected_host - 29 GiB   // 24 GiB P3 floor + 5 GiB margin
mapped_ws     = host_mapped_bytes (host-executed ranges only)
```

If `mapped_ws > resident_room`, the load logs one WARN,
`[HOST-PLAN] mapped host weights exceed the page cache that can stay
resident`, with the three numbers, and proceeds. It is a speed hazard, not a
safety one (§5.2). For a model with a cold-expert class (DeepSeek), yl3v's
search should size `C` so that `mapped_ws <= resident_room` instead (open
question Q2, §8).

**The headroom formula** (the lead's point (c)). One change, in the one
function:

```
host_ledger_budget() = min(total x pct, MemAvailable - host_headroom)
host_headroom        = max(24 GiB + 5 GiB, total / 32)          // was max(8 GiB, total / 32)
```

Why: P3 (9g22's gate precondition, which th32 §4.3 makes bind every load)
requires `MemAvailable_pre - expected_host - 5 >= 24`. A ledger that leaves
only 8 GiB admits plans P3 refuses, which th32 §4.5 has to argue around ("the
planner's own fit would put the table HOST ... which is why P3, not the
planner's fit, decides"). With this headroom the ledger and P3 are one number.

On this host (`MemTotal` 246.9 GiB, so `total / 32` = 7.7 GiB and the
headroom is 29 GiB): at today's reading, `MemAvailable` 150.8 GiB, the budget
is **121.8 GiB** instead of 142.8. At the baseline's 154.8 GiB it is 125.8
instead of 146.8. Until th32 T2 lands, the same change applies to the
`[HOST-ARENA]` formula at `uc:4216` and `:4227-4231`, which T2 then lifts.

**The host-fit gate** (the lead's point (b)) is not duplicated here. The
refusal is th32 T2's `host-budget-exceeded` (phase (b) enforcing, phase (a)
WARN-only), and the number is 7v5c's dry-run. 5efe supplies their inputs:
`host_tier` per entry, `weight_host_bytes` restricted to charged weights,
`host_mapped_bytes`, and the locked/unlocked policy. 7v5c's printout gains
the mapped line and the working-set check. On Qwen the 154 GB overcommit that
the baseline hit becomes a fit instead of a refusal (§4.7), so no gate is
needed to protect the first Qwen run, but the gate remains the protection for
`-lm none` and for models whose charged bytes still overflow.

### 4.6 Prefetch, and the lazy ranges (lead rulings §T13a, §T13c)

- **5efe owns the exclusion, because it owns the mapping.** th32 stays off
  mmap entirely and does not register the table's range anywhere in the
  loader (§T13c), so nothing upstream excludes it for the tier. The loader
  passes every `TENSOR_READ_LAZY` range of the file to the tier through the
  hook's `lazy_ranges` argument (§4.3), taken from `lazy_read`'s own set,
  which `lazy.add` fills (`lml:1776`).
- **The tier maps only non-lazy tensor ranges.** `[first, last)` of a
  `SYCL_Mapped` buffer is computed over that buffer's tensors, and lazy
  tensors are never in it (they have their own CPU or th32 buffer). Where a
  lazy range still falls inside `[first, last)` (tensors from different
  buffers interleave in a file), the tier maps the complement segments as
  separate mappings under one owner, so a lazy byte is never mapped by the
  tier at all. The complement is the same computation as
  `ranges_complement` (`src/llama-mmap.cpp:444`), but that helper is a
  file-static in libllama and ggml-sycl cannot call it. So the tier has its
  own pure function, `file_mapping_segments`, which also rounds each segment
  *inward* at a lazy boundary, so no mapped page straddles a lazy byte. C3
  tests it.
- **Kernel readahead is the one residual path.** A fault near a segment edge
  can read ahead past the edge, because readahead is per file, not per
  mapping. The tier sets `MADV_RANDOM` on the last `read_ahead_kb` of each
  segment that borders a lazy range, which disables readahead there. The
  residency gate below measures what is left.
- **`WILLNEED`** is issued only over host-executed ranges (the `FILE_MAPPED`
  entries of §4.4), right after mapping, never over lazy ranges and never
  over device-only ranges (the staging copy reads those once).
- **No `MAP_POPULATE`, no `MADV_HUGEPAGE`** from the tier. Huge pages on a
  file mapping depend on the filesystem's large-folio support, which is not
  established for bcachefs here, and th32 §2.4 measured that bcachefs reads
  whole 256 KiB extents anyway.
- **After staging**, device-only ranges get `POSIX_FADV_DONTNEED`, so the page
  cache they occupied goes back to the host experts' working set. A later
  re-materialization after VRAM eviction simply re-reads them.
- **Until th32 lands**, lazy tables keep today's path (§2.5): the loader's
  lazy mapping, `MADV_RANDOM`, no prefetch, no lock. The tier changes none of
  it, and the loader's mapping gets prefetch 0 on SYCL (§4.2), which can only
  reduce what is prefetched. That loader mapping is upstream's, not the
  tier's. The tier never double-maps a lazy range, because it maps no lazy
  byte at all.
- **Gate** (commit 3, and §7's lead gate): after a load, no page of a lazy
  range is resident. Host-only form: a fixture file on a non-tmpfs directory,
  evicted with `POSIX_FADV_DONTNEED`, mapped through the tier with a lazy hole,
  then `mincore` over the hole must report 0 resident pages while the
  host-executed range reports resident after `WILLNEED` (the positive
  control). The test returns 77 if the directory is tmpfs, whose pages are
  always resident. A second case faults the byte just before the hole and
  expects the hole to stay non-resident; that is the readahead case. On Qwen
  the lead gate is simpler, because shard 3 holds only the table (§2.5):
  `fincore` on shard 3 after the load must still read about 0.

### 4.7 th32 interface, and the figures that move

- **Separate tiers, separate readers.** STORAGE reads through th32's own
  `O_DIRECT` fd and never maps (lead rulings §T13a, §T13c); `HOST_CPU` is anonymous.
  `FILE_MAPPED` never holds a lazy table. th32 §5.1's argument that "a HOST
  residency backed by mmap is not HOST at all" is consistent with this
  design: `FILE_MAPPED` is not charged as resident, and the plan says so.
- **T1's census** ("`MMAP_TRACKED`: legacy file mmap tracked for weights",
  th32 §3.1) changes: `MMAP_TRACKED` becomes `UNREGISTERED`, and
  `FILE_MAPPED` is a real tier. The census sites this design touches are listed
  in §4.8, so th32 T1 and 5efe C2 can land in either order against one list.
- **Figures in th32 §4.5 that move** once 5efe lands (th32 re-pins them):
  - "host experts, exact" (Qwen 98.7-101.7 GiB, DeepSeek 164.7-167.7 GiB
    today) moves out of step 2 into `host_mapped_bytes` under `-lm mmap`;
    under `-lm mmap+mlock` it stays.
  - "other host weights (`token_embd`)" moves the same way.
  - The WEIGHT zone term shrinks to rnh3's named terms plus the reordered
    experts.
  - Every "`MemAvailable_pre` needed" row drops by the moved bytes. For Qwen
    after (s), a HOST table would need about 180.9 - 99.3 = 81.6 GiB of
    `MemAvailable_pre` for step 2 plus the table, before the working-set check.
    With the mapped experts, the working-set check (§4.5) then decides, and
    the PLE table and the experts compete for one page cache: that trade is
    th32's to re-pin, through 7v5c.
  - §2.3's `mmap_support` bullet and T6's flip move to 5efe commit 1.
  - The ledger's headroom (§4.5) changes every "planner's fit" column; P3
    columns are unchanged, because P3 already carried the 29 GiB.

### 4.8 The tier-switch census (sites 5efe touches)

`MMAP_TRACKED` today, all renamed to `UNREGISTERED` or taught `FILE_MAPPED`:

| site | today | after |
|---|---|---|
| `uch:5593` enum | `MMAP_TRACKED = 2` | `UNREGISTERED = 2`, plus `FILE_MAPPED = 3` |
| `uch:5614` `memory_location::tier` default | `MMAP_TRACKED` | `UNREGISTERED` |
| `uc:2408` `alloc_tier_name` | "mmap_tracked" | "unregistered", "file_mapped" |
| `uc:2613` `offload_stats_note_alloc` | counts as shared | `UNREGISTERED` unreachable; `FILE_MAPPED` its own counter |
| `uc:14812` `unified_free_record` | returns without freeing | `UNREGISTERED`: unchanged; `FILE_MAPPED`: unmap |
| `uc:15125` `unified_alloc` refusal | refuses `MMAP_TRACKED` | refuses both (mappings come only from the file-mapping front door) |
| `uc:16825` registry host sum | pinned + mmap-tracked | pinned + file-mapped, reported separately |
| `uc:21637`, `:21654` `query_registered_location` / `query_location` | unknown → `MMAP_TRACKED` | unknown → `UNREGISTERED`; registry `FILE_MAPPED` → `FILE_MAPPED` |
| `gs:38167`, `:38202` `kv_layer_alloc::tier` default | `MMAP_TRACKED` | `UNREGISTERED` |
| `gs:81263-81269` `ggml_sycl_alloc_tier_name` | 3 cases | 4 cases |
| `gs:5731` (`expert_resolve_tier` switch) | `HOST_MMAP` | `HOST_MMAP` now means "served from a `FILE_MAPPED` owner" |
| registry `alloc_type` (`alloc-registry.hpp:26`) | `MMAP` = not-USM (KV synthetic base, `gs:39638`) | new `FILE_MAPPED`; the KV base keeps `MMAP`, renamed `HOST_UNREGISTERED` in the same commit |

`cache_location::HOST_MMAP` (39 sites) keeps its name, and its derivation
changes in one place each (`uc:8068-8081`, `:8235-8248`, `:6504-6545`): it
is `HOST_MMAP` only when `query_location` says `FILE_MAPPED`. An
`UNREGISTERED` pointer is refused by `register_host_expert` /
`register_host_weight` once every weight source is owned (commit 6), which
also retires the direct-stage "raw mmap fallback" as an unowned path.

---

## 5. Hazards (item 4)

### 5.1 `mlock`

- `mlock` without `MLOCK_ONFAULT` faults the whole range in synchronously. On
  Qwen that is ~100 GiB from the USB members at 0.29-1.78 GB/s (th32 §2.4),
  so a locked load takes one to six minutes. That is the price of the
  guarantee, and it is only paid under `-lm mmap+mlock`.
- Locked file pages are unevictable, so they are charged (§4.5); the working-set
  check does not apply to them.
- `RLIMIT_MEMLOCK` is unlimited on this host (th32 §3.1); elsewhere the refusal
  names it.
- The loader's existing `llama_mlock` of host buffers (`lm:2782-2787`) keeps
  applying to `SYCL_Host` buffers under `-lm mlock`; the tier's lock replaces
  it for `SYCL_Mapped` buffers.

### 5.2 Page-fault latency on CPU-executed experts

- Storage: bcachefs over USB SSDs, with NVMe as a promote cache; 1.78 GB/s
  sequential when served from the NVMe cache, 0.29-0.47 GB/s settled, and a
  4 KiB random read costs ~256 KiB of device reads (th32 §2.4).
- Qwen3.8's `ffn_*_exps` are 850 MiB for 256 experts, so one expert matrix is
  ~3.3 MiB and contiguous. A cold expert therefore faults as a short
  sequential run (the kernel's readahead), not as 4 KiB randoms: roughly
  2-12 ms per matrix at the rates above.
- A fully cold decode token touches 8 experts x 3 matrices x 48 layers x
  3.3 MiB ≈ 3.8 GiB, which is seconds per token. So the mapped experts must be
  warm: the tier `WILLNEED`s their ranges at load (§4.6), and the working-set
  check (§4.5) says whether they can stay warm.
- **Reclaim under other tenants.** This host's permanent tenants (the qemu VM,
  Emby, Frigate) can grow. The kernel then drops clean mapped pages first, and
  decode slows to storage speed without any error. `-lm mmap+mlock` is the
  opt-out.
- **Faults happen on `CpuExpertPool` workers**, inside `ggml_sycl_cpu_expert_mul_mat_batched`,
  so a stall holds a worker, not the submitting thread. The existing
  host waits on that path (th32 §2.3) are unchanged.
- **`SIGBUS`.** A mapped file that is truncated or replaced while mapped
  (a re-download into `/models`), or a bcachefs checksum failure on fault,
  raises `SIGBUS` in a worker instead of an `EIO` return. The tier's size
  check at map time (§4.1) catches a replaced file only at load. This is the
  same exposure upstream's mmap has; it is recorded, not mitigated.

### 5.3 A GPU reading mapped pages (forbidden)

Mapped pages are not USM. A device read of one is at best a slow
driver-staged copy and at worst a device loss (`uc:14044`'s own
comment). The ways a weight's host pointer can reach a device today, and what
each becomes:

| site | today | after |
|---|---|---|
| `mem_handle::resolve(device)` of a `FILE_MAPPED` handle | n/a | returns empty; `device_mappable()` is false |
| `ggml_sycl_resolve_or_host_tensor_ptr` (`common.hpp:4922-4945`) | returns the raw pointer for any non-device host pointer | refuses a `FILE_MAPPED` registration (returns null, and the caller's existing null path runs) |
| layout availability's "host buffer is acceptable" branch (`gs:62109-62111`) | `weights_evictable && is_host` → available | not for a `SYCL_Mapped` buffer: availability must come from a cache entry |
| `stream_dma`'s `HOST_MMAP` branch (`uc:14043-14046`) | `copy_to_device_async` straight from the source pointer, despite the comment | CPU `memcpy` into the pinned staging slot, then H2D, the same bounce `mem_copy` does |
| `direct_stage_weight`'s plain copy (`uc:6167-6175`) | `mem_copy` from a copy handle | unchanged: `mem_copy` already bounces a non-USM source through a 64 MiB pinned window (`mem-ops.cpp:700-729`) |
| reorder `fill_fn`s (`gs:34835-34840`) | read `src_ptr` on the CPU | unchanged (CPU reads are allowed) |
| direct-stage last resort (`uc:6504-6545`) | registers an unrecognized source in place as `HOST_MMAP` | registers the `FILE_MAPPED` slice with its owner; an `UNREGISTERED` source is refused |

Gates: a host-only test that `resolve(device)` of a `FILE_MAPPED` handle is
empty and `ggml_sycl_resolve_or_host_tensor_ptr` returns null for one; and a
`GGML_SYCL_PRIVATE_TESTING` counter of device-queue copies whose source is
`FILE_MAPPED`, which the lead's load gate reads as 0. The standing
`llama.cpp-v4jk` audit (no route stages a host operand to a device) covers
the dispatch side.

### 5.4 The staging bounce

- `mem_copy`'s bounce waits on each 64 MiB slice before refilling it
  (`mem-ops.cpp:716-719`), and on the last one unless the caller retains the
  handles (`:729`). That is a host wait, but only at load, where the
  preload already runs.
- The bounce buffer is a standalone pinned-pool allocation, role
  `EXPERT_STAGING`, released individually when its handle drops
  (`mem-ops.cpp:472-483`, allocated at `:704-705`). The transient pinned
  footprint is therefore 64 MiB per copy in flight, not a slice of the STAGING
  zone and not proportional to the model.
- It engages only if the copy handle of a mapped pointer reports a non-USM
  host source (`host_ptr_is_usm_accessible`, `mem-ops.cpp:700`). C4's
  host-only RED pins that, since a mis-classified mapped source would take the
  direct-copy path instead.

---

## 6. The double residency of VRAM weights

**Decision: fixed here for mmap mode, by construction; a ticket for
`-lm none`.**

- In mmap mode the persistent host source of a device weight is the
  `FILE_MAPPED` mapping. Its pages are clean page cache: reclaimable, and
  dropped after staging (§4.6). A VRAM weight is resident once.
- The entry's recorded `src_ptr` (`uc:6213`) stays valid: it points into a
  mapping that lives as long as the entry's `storage_owner`. A
  re-materialization after a VRAM eviction re-reads the file through page
  faults.
- **`-lm none`** keeps `SYCL_Host`, so it keeps the double residency. The fix
  there is different: a transient pinned window per preload slot, and a file
  re-read for re-materialization, which needs the loader's file handle to
  outlive the load. **Ticket to file:** "SYCL `-lm none`: device weights stay
  resident in `SYCL_Host` after preload (double residency)", depending on
  5efe.

---

## 7. Implementation plan (item 5)

Every commit is RED → GREEN, host-only where possible. GPU work is the
lead's; each commit names what the lead runs. Every GPU run below pins
`ONEAPI_DEVICE_SELECTOR`, samples `grep -E
'^(MemAvailable|Shmem|AnonPages|Cached|Mlocked):' /proc/meminfo` every 3 s
with a 25 GB `MemAvailable` watchdog, and runs once.

**C1. Restore `mmap_support`** (§3).
- RED: `tests/test-sycl-mmap-support-caps.py`, a source contract: the
  initializer in `ggml_backend_sycl_device_get_props` names `.mmap_support`,
  binds it to `host_unified_memory`, and is neither a literal `true` nor
  absent. Today it fails with the four-initializer list.
- Lead: Qwen on the B70 with **no** `-lm`/`-lzm` flags: the log shows
  `load_mode = mmap` and `per_layer_token_embd ... lazy read enabled`, and
  still `disabling mmap for SYCL weight layout upload`. The load is still
  expected to overcommit on the experts (it is C4 that fixes that), so the
  watchdog run is a witness, not a pass. Plus the Mistral gate on `:1` and
  `:0` (unchanged tokens, since Mistral has no lazy tensor).

**C2. The `FILE_MAPPED` tier and the census rename** (§4.1, §4.8).
- RED: host-only `test-sycl-file-mapped-tier` (`GGML_SYCL_PRIVATE_TESTING`, no
  device): maps a fixture file through `unified_allocate_file_mapping`, reads
  its bytes, checks `query_location` returns `FILE_MAPPED`,
  `device_mappable()` false and `resolve(device)` empty, drops the last
  handle, and checks the range is unmapped (`mincore` returns `ENOMEM`).
  Negative control: a size mismatch against the recorded file size is refused.
  Fails today: no such tier.
- Also: the canonical contract §3 allowlist gains the two wrappers, and the
  static-storage census and the allowlist gates are re-run.
- Lead: none (no load path uses the tier yet).

**C3. The mapping plan: lazy carve-out and prefetch policy** (§4.6).
- RED: a pure-function test of `file_mapping_segments(tensor ranges, lazy
  ranges, page size)` → map segments and `WILLNEED` segments, including an
  interleaved lazy range and page-boundary rounding that must never reach
  into a lazy page. Then the `mincore` fixture of §4.6, with its positive
  control and its readahead case; 77 on tmpfs.
- Lead: none.

**C4. `SYCL_Mapped` and the loader** (§4.3, §4.4, §5.3). This is the commit
that changes every SYCL load.
- RED, host-only: `register_host_expert` over a `FILE_MAPPED` slice stores the
  owner (`storage_owner` set, `non_owning_external_host` false), and the
  mapping outlives the buffer while an entry holds it; `stream_dma` over a
  `FILE_MAPPED` source makes zero direct device copies (the private counter);
  `mem_copy` from a copy handle over a mapped pointer takes the pinned bounce
  (`host_ptr_is_usm_accessible` false);
  `ggml_sycl_resolve_or_host_tensor_ptr` returns null for a mapped weight. The
  loader half (buft choice, `lm:2693` removal, `load_all_data` addressing) is
  covered by a source contract, since it needs a model load.
- Lead, in this order:
  1. The residency gate's baseline: `POSIX_FADV_DONTNEED` over shard 3, then
     `fincore` on it (expect ~0).
  2. **Qwen3.8 Q8_0 on `level_zero:0`, default load flags, `-c 4096`, the
     completion from the baseline.** Pass: the load completes and prints
     coherent text; peak `AnonPages` growth < 30 GiB over the pre-load
     reading; `MemAvailable` never under 25 GiB; `Cached` growth of at least
     90 GiB; buffer lines show `SYCL_Mapped` holding roughly the host experts
     and `SYCL_Host` holding no weights; `fincore` on shard 3 after the run
     still ~0; the private direct-copy counter is 0; `Shmem` flat.
  3. The Mistral gate on `:1` and `:0`, and the GPT-OSS gate on `:1` (every
     SYCL load now takes the mapped path).
  4. `llama-bench` Mistral Q4_0 and GPT-OSS 20B on both cards, interleaved
     against the parent commit, gated on `docs/backend/sycl-perf-baselines.md`.
     Load time is also reported: device weights now read from page cache
     through the bounce.

**C5. Planner host tier, the ledger split and the headroom** (§4.4, §4.5).
- RED, host-only planner fixture (Qwen-shaped inventory, B70 budget): host
  experts carry `host_tier = FILE_MAPPED` when the inventory says the load
  maps, they are absent from `weight_host_bytes` and present in
  `host_mapped_bytes`; with `mlock` they are charged. A budget unit test with
  injected `MemTotal`/`MemAvailable` (246.9 / 150.8 GiB) expects 121.8 GiB
  (the old formula gives 142.8, so the test fails today). A working-set test
  expects the WARN at `MemAvailable_pre` below the threshold and none above.
- Depends on th32 T2 for `host_ledger_budget()` if T2 is in; otherwise it
  edits `uc:4216`/`:4227-4231` and T2 lifts it.
- Lead: the C4 Qwen run again, reading the new `[PLACEMENT] Totals` line
  (mapped vs charged) against the `/proc/meminfo` samples.

**C6. Retire unowned host registration** (§4.8).
- RED: host-only, `register_host_expert` and `register_host_weight` with a
  `malloc`ed (`UNREGISTERED`) pointer are refused; today they succeed as
  `HOST_MMAP`, `non_owning_external_host`.
- Lead: the Mistral and GPT-OSS gates, and the C4 Qwen run (the refusal must
  never fire on a real load).

Tickets to file after review: `-lm none` double residency (§6);
`CPU_REPACK` outside the cache (§4.4); and, depending on Q2, a yl3v note.

---

## 8. Open questions for the lead

- **Q1. Unlocked mapped experts: WARN or refuse** when the working set exceeds
  the resident room (§4.5)? This design WARNs, because paging is a speed
  hazard, not a memory-safety one, and `-lm mmap+mlock` is the strict mode.
- **Q2. Cold experts.** For a model with a cold class, should the cold set be
  sized so the *mapped* warm experts fit the resident room (yl3v keeps its
  `O_DIRECT` path), or should the cold experts simply be unprefetched
  `FILE_MAPPED` pages? This design recommends the first: yl3v's reads are
  planned and prefetched at `set_input`, page faults are not.
- **Q3. `-lm none` on SYCL.** Should `none` remain a supported SYCL mode at
  all once mmap works? It is the only mode left with the double residency.
