# SYCL file-mapped host tier (mmap as a unified-cache tier)

*Status: DESIGN rev 3 (folds design-5efe-r1, including its addendum, under
lead rulings §F1, §F2 and §F2a, plus the th32 r15 m-4 mlock item and the §M49
note that the lazy-range record lands in C1), not implemented. Tracker: `llama.cpp-5efe`.
Written 2026-09-27 against branch `task/5efe` at `69e70c715` (= master). Every
cite without a revision prefix is to that tree. `gs` is
`ggml/src/ggml-sycl/ggml-sycl.cpp`, `uc` is
`ggml/src/ggml-sycl/unified-cache.cpp`, `uch` is `unified-cache.hpp`, `lml`
is `src/llama-model-loader.cpp` and `lm` is `src/llama-model.cpp`. Load-phase
evidence is the lead's Qwen3.8 baseline `qwen-base/run2.log` and `run2.mem`
(session scratchpad, `fefb92980`, B70 alone, `-lm mmap -lzm on -c 4096 -v`),
whose load-path code is byte-identical to this tree in every region cited here.
Appendix A disposes of every r1 finding.*

Sibling designs this one interfaces with, and does not restate:

- `llama.cpp-th32`: STORAGE tier for lazy row tables,
  `docs/design/sycl-storage-tier-lazy-tables.md`, abbreviated **th32**.
  Pinned at its latest reviewed head, **rev 2r `df6e83b3b`** (r17 found nits
  only). Rev 2s, `f364952d3`, folds those nits and §F2a I-9: its inventory
  reads byte ranges from this design's lazy-range record (§3.1). Section
  numbers cited here are the same in both.
- `llama.cpp-7v5c`: host-only plan dry-run.
- `llama.cpp-rnh3`: exact host WEIGHT bytes.
- `llama.cpp-sgc5`: one zone set per plan.
- `llama.cpp-yl3v`: cold experts on STORAGE.
- `llama.cpp-qptd`: `-ngl 0` loses mmap.

Lead rulings are cited by their labels in the session's rulings file. The
first two th32-interface rulings, first sent as §T10/§T10a, are **§T13/§T13a**.
§T13b is withdrawn by §T13c.

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
the copy into that pinned buffer reads from. The baseline shows what that
costs:

- The pre-size committed 139264 MB of pinned chunks (`run2.log:2900`).
- The pool reached 140.0 GB at chunk 67 (`:2967`).
- `MemAvailable` fell by 141 GiB, from 155.0 to 13.6 GiB (`run2.mem`),
  before the watchdog killed the run.

This design therefore does four things:

- **Commit 1 restores `mmap_support`**, as `host_unified_memory` known and
  false (§3). By itself that moves one thing: `-lzm auto` engages again, so
  the 50.66 GiB Qwen PLE table leaves the pinned pool under default flags.
  Commit 1 also lands the **per-file lazy-range record** (§3.1). It is the one
  source of every `TENSOR_READ_LAZY` byte range, and lazy_read, th32's
  inventory, the tier and llama_mmap all read it.
- **A new allocation tier, `FILE_MAPPED`** (§4.1-§4.3).
  - It is a read-only file mapping that the unified cache creates inside one
    VA reservation, owns through one `alloc_owner`, and releases with one
    `munmap`.
  - A new per-device buffer type, `SYCL_Mapped`, replaces `SYCL_Host` as the
    source buffer of every SYCL weight when the load maps files.
  - Host weights that the CPU executes in file layout are served from the
    mapping.
  - Device weights are copied from it once, at load or materialization,
    through a transient pinned window. So **the double residency of every
    VRAM weight (pinned host + VRAM) ends in mmap mode** (§6).
- **One published fact, resolved before the early plan** (§4.3, §4.4). The
  load mode is resolved before the planner runs, and the planner gives every
  weight a `host_backing`. Both the WEIGHT-zone sizing and the loader's
  buffer-type choice read that. The split lands in the same commit as the
  loader (C4), so the Qwen acceptance never runs against a WEIGHT zone sized
  for pinned experts (r1 C-1).
- **Budget** (§4.5).
  - Mapped bytes are reported but not charged, unless `-lm mmap+mlock` locks
    them.
  - The host budget becomes monotone in `MemAvailable`, with one named
    headroom, `max(total/32, min(29 GiB, total/4))`.
  - A mapped working set that cannot stay resident is a WARN (§F1 Q1).
  - The host-fit refusal is th32 T2's over 7v5c's computation; 5efe supplies
    its terms.

**Acceptance** (lead-run, §7 C4): Qwen3.8-Flash-Next Q8_0 loads on the B70
(`level_zero:0`) under default load flags, and every figure below was
pre-registered before the run:

- the WEIGHT zone is 64.0 MB;
- `AnonPages` grows by less than 30 GiB;
- `MemAvailable` never falls under 25 GiB;
- `SYCL_Mapped` holds about 127615 MiB;
- shard 3, the PLE table, stays under 256 MiB resident.

---

## 1. Scope and principles

- **P1, the cache owns the bytes.** Today the fork has no cache-owned file
  mapping. A weight in a `CPU_Mapped` buffer (upstream's mmap path, the lazy
  tables) sits outside the contract (canonical contract §8 items 2 and 4), and
  host experts are registered as `non_owning_external_host` slices the cache
  never frees (`uc:8150`, `:12351`). The tier fixes both for mapped weights.
- **P2, the handle owns the lifetime.** Every reader of a mapped byte holds a
  `mem_handle`, directly or through an owner that wraps one. The last release
  unmaps (§4.2).
- **P3, placement decides the executor.** There is no GPU zero-copy of host
  memory and no weight streaming.
  - A mapped byte is read by the CPU only.
  - The GPU receives mapped bytes only as a load-time or materialization copy
    through a pinned window.
  - A mapped weight reaching the dispatch-time streaming path is a typed
    refusal (§5.3).
- **P4, plan equals reality, and no idle reservation.** One published fact
  says whether the load maps files. The plan says which bytes are pinned,
  mapped or neither, and the ledger charges exactly the resident ones (§4.4,
  §4.5).
- **Out of scope here:**
  - The lazy row tables themselves. th32 owns them; §4.6 only guarantees that
    the tier never maps, prefetches or locks them.
  - The cold-expert policy. yl3v owns it; §4.5 states the interface.
  - The `-lm none` double residency. It gets a ticket (§6).
  - `CPU_REPACK` buffers living outside the cache. This is an existing P1 gap,
    also a ticket (§4.4).

---

## 2. The load path today (item 1)

### 2.1 How `-lm` and `-lzm` resolve

- `llama_model_loader` sets `use_mmap` for `mmap`, `mmap+mlock` and `auto`
  (`lml:599`).
- `load_tensors` computes the SYCL early placement plan first (`lm:2228-2231`).
  Only then does it drop `use_mmap` under `auto` when any model device reports
  `mmap_support = false` (`lm:2245-2254`), and turn lazy `auto` into `off` on
  the same test (`lm:2256-2265`). The fork's SYCL devices all report `false`
  (§0), so the default command line loads with no mmap and no lazy tables. The
  early plan therefore runs on an unresolved load mode, which §4.3 fixes
  (r1 I-5).
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

Upstream's "avoid a host buffer under mmap" demotion (`lml:1512-1518`) is
disabled for SYCL by `allow_host_buft_with_mmap` (`lml:1503-1511`), so even a
surviving `use_mmap` keeps the weights in `SYCL_Host`.

`make_cpu_buft_list` puts the first device's host buft at the front of the CPU
list unless `--no-host` (`lm:1765-1779`). That is where `token_embd` lands, and
it is `llama.cpp-qptd`'s finding: at `-ngl 0` the weights still land in
`SYCL_Host`.

### 2.3 Where the bytes land

**Host-resident weights land in pinned USM, copied, never in mapped pages.**
`SYCL_Host` allocates through `unified_alloc` with role `WEIGHT` (whenever
`weights_evictable`, even after the load, `gs:42645-42646`), category
`HOST_COMPUTE` and `must_host_pinned` (`gs:42640-42665`). That routes to the
host WEIGHT zone (`uc:15361-15372`). `load_all_data` fills it with `read_raw`
straight into `cur->data`, because the buffer is host (`lml:2134-2141`). With
mmap it would have been a `ggml_backend_tensor_set` from the mapping into the
same buffer (`lml:2129`).

**Device-resident weights get their bytes from that pinned buffer, at load
end.**
- `ggml_backend_sycl_model_load_end` runs `ggml_sycl_preload_model_weights()`
  (`gs:12476`, defined `:33515`).
- It takes `src_ptr = tensor->data` (`gs:34763`, `:35662`) and calls
  `unified_cache::direct_stage_weight` (`uc:6036`) to fill a fresh VRAM
  allocation.
- The "tiered" override that could have replaced `src_ptr` is dead:
  `get_cached_tensor_ptr` always returns `nullptr` (`gs:19236-19261`).
- Nothing releases the pinned copy afterwards, and the published entry records
  `src_ptr` into it (`uc:6213`).

**So every VRAM weight is resident twice** for the model's lifetime (lead's
trace report, point 1).

**Host experts are registered in place.** `expert_aos = src0->data +
e * expert_size` (`gs:33437`) is registered by `register_host_expert` with no
owner, so `non_owning_external_host = true` (`uc:8150`) and the cache never
frees it (`uc:12351-12353`). A second pinned allocation happens only when a
CPU reorder is needed (`register_host_aos_expert`, `gs:34142`, alloc at
`:34176`).

**`CpuExpertPool` reads that pinned slice.** Each task carries the host
lease resolved from the registered entry (`gs:76020-76028`), and the pool
executes it (`gs:76049-76067`). `ggml_sycl_host_data()` is `tensor->data`
(`common.hpp:4889-4891`), which is the `SYCL_Host` buffer.

### 2.4 `use_mmap_buffer` and `buffer_from_host_ptr` in this fork

- `use_mmap_buffer` is the constant `true` (`lm:2241`).
- The mapped branch requires `caps.buffer_from_host_ptr` **and** that the buft
  be its device's *default* buft (`lm:2740-2748`).
  - The SYCL device reports `false` (`gs:107801`), and its
    `buffer_from_host_ptr` hook is a stub returning `nullptr`
    (`gs:107848-107856`).
  - `SYCL_Host` is not a default buft anyway.
- So the only buffers built from a mapping are the CPU device's (`CPU_Mapped`,
  `ggml-cpu.cpp:398-400`): plain-CPU weights under `use_mmap`, and the lazy
  tables.
- After the load, `pimpl->mappings` keeps each `llama_mmap` alive for the
  model's lifetime (`lm:2866-2870`). The last `load_all_data` call trims the
  unused head and tail of each mapping (`lml:2236-2247`).

### 2.5 `TENSOR_READ_LAZY` on SYCL (`-lzm on`, Qwen `per_layer_token_embd`)

- `qwen4exp` creates the table with `TENSOR_READ_LAZY`
  (`src/models/qwen4exp.cpp:189`). `lazy.add` accepts it under `on` at any
  size, and under `auto` only above 4 GiB (`lml:1240-1268`). `run2.log:370`
  shows `lazy read enabled` for 51880 MiB.
- The byte range is kept in `lazy_read`'s private
  `std::map<uint32_t, llama_mmap::ranges> ranges` (`llama-model-loader.h:118`,
  filled at `lml:1258`). `llama_mmap` receives it through `lazy.for_file(idx)`
  (`lml:1890-1891`). §3.1 replaces that copy with the record.
- Its buffer type is the CPU default buft (§2.2), so the context is a lazy CPU
  context.
  - `init_mappings` maps every file whenever `lazy.any()`, even after
    `lm:2693` cleared `use_mmap` (`lml:1869-1872`), with prefetch 0
    (`lml:1888`).
  - The mapping is advised `POSIX_MADV_RANDOM` over the lazy ranges and
    excluded from `WILLNEED` and `MADV_HUGEPAGE`
    (`src/llama-mmap.cpp:503-545`). `MAP_POPULATE` is gone (`:479-481`).
- `load_tensors` wraps it as `CPU_Mapped` (`is_lazy_mapped`, `lm:2745-2748`).
  `load_all_data` points the tensor into the mapping without reading it
  (`lml:2099-2128`).
- **It can still be mlocked** (r1 addendum I-10). Under `-lm mmap+mlock`,
  `load_all_data` skips the growth call only for the lazy tensor itself
  (`!lazy.has(cur)`, `lml:2120`). A non-lazy mapped tensor that comes after the
  table in the same file calls `grow_to(weight->offs + n_size)` (`lml:2122`).
  `llama_mlock::grow_to` locks `[addr + size, target)` contiguously from the
  mapping's base (`src/llama-mmap.cpp:826-840`), so that call locks, and
  therefore faults in, every lazy byte before the tensor. §4.6 fixes it.
- It is not a SYCL inventory entry.
  - The late inventory filters out non-SYCL bufts, so the late plan's host
    total drops from 154263.6 to 102383.5 MB (`run2.log:308` vs `:2821`,
    exactly 51880.1 MB less).
  - Its `get_rows` runs on ggml-cpu, faulting rows in from the file at the
    storage rates th32 §5.1 measured.
- The table is the only tensor in shard 3 of 6 (offset 192, 54400261120 B,
  read from the GGUF headers). §7's residency gate uses that.

### 2.6 The baseline's second 104 GB

The lead asked whether `Host staging zone ... per_inflight=103762.3 MB`
(`run2.log:243`) is a second reservation. It is not.

- The STAGING zone is `max(2 x max_tensor, max_staging_pair) + 2 MiB` per
  in-flight slot (`uc:27674-27708`).
- The **early** plan (`lm:2230`, before `create_tensor`) still counts the lazy
  table as `max_tensor`: 2 x 51880.1 + 2 = 103762.3 MB.
- The late plan, which no longer sees the table, prints 1702.0 MB
  (`run2.log:2756`).
- Both configure lines carry `STAGING=1719.2 MB` (`run2.log:2893`, `:2899`).

So it is a plan-time figure that was never committed. th32 §4.1 already
requires lazy tables to be excluded from `max_tensor`, and its T3 fixture
should cover the early plan too.

### 2.7 The legacy "mmap" vocabulary

None of it refers to a `llama_mmap` page today:

- `alloc_tier::MMAP_TRACKED` (`uch:5593`) is what `query_location` returns for
  a pointer the registry does not know ("assume host/mmap", `uc:21649-21655`).
  It is also the default of `memory_location::tier` (`uch:5614`).
  `unified_alloc` refuses it (`uc:15125-15127`), and `unified_free_record`
  returns without freeing it (`uc:14812-14814`).
- `cache_location::HOST_MMAP` (`uch:2170`) is derived from that classification
  when an entry is registered (`uc:8068-8081`, `:8235-8248`). It is also set by
  the direct-stage last resort "host arena exhausted, raw mmap fallback"
  (`uc:6504-6545`).
- `alloc_type::MMAP` (`alloc-registry.hpp:26`, commented "mmap'd from GGUF")
  has two meanings in use, neither of them a GGUF mapping:
  - the tiered KV buffer's synthetic, non-USM base (`gs:39631-39639`);
  - "regular malloc (pinned cap overflow)", which `unified_free_record` frees
    with `::free` (`uc:14922-14929`).

### 2.8 The host budget today

- `[HOST-ARENA]` is `min(total x 90%, MemAvailable - max(8 GiB, total/32))`
  (`uc:4200-4231`). It is the same function th32 §4.2 lifts into
  `host_ledger_budget()`.
- It is not monotone: when `MemAvailable` does not exceed the reserve, the cap
  becomes `available/2` (`uc:4227-4230`).
- The planner has only `placement_entry::on_device`, so "host" means pinned,
  and `weight_host_bytes` is never compared with any budget (lead's
  host-budget report; th32 §2.3).
- No load-time host-fit gate exists.

---

## 3. Commit 1: the caps restore (item 2)

```cpp
// gs:107797, ggml_backend_sycl_device_get_props
ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *) dev->context;
const auto & info = ggml_sycl_info().devices[ctx->device];
props->caps = {
    /* .async                 = */ async,
    /* .host_buffer           = */ host_buffer,
    /* .buffer_from_host_ptr  = */ false,
    /* .events                = */ events,
    /* .mmap_support          = */ info.host_unified_known && !info.host_unified_memory,
};
```

- **The iGPU detection already exists.**
  - `ggml_sycl_device_is_host_unified()` queries
    `sycl::info::device::host_unified_memory` (`common.cpp:827-833`), and
    `ggml_sycl_info()` caches the result per device (`gs:25195`, field at
    `common.hpp:1477`).
  - `ggml_sycl_info()` is initialized before any SYCL device object exists,
    because registration iterates it (`gs:110187`). So the cached field is
    safe to read here.
  - This matches CUDA's `type != IGPU` (`ggml-cuda.cu:5130`) and Vulkan's
    `!is_integrated_gpu` (`ggml-vulkan.cpp:15221`).
  - Upstream SYCL sets a flat `true` (`694ec2354 ggml-sycl.cpp:6380`). The
    fork keeps the iGPU off because of its own history (llama.cpp-403s).
- **The query fails open today, so C1 closes it for this cap only** (r1 m-17).
  - `ggml_sycl_device_is_host_unified()` returns `false` when the query
    throws. An iGPU whose query throws would otherwise advertise
    `mmap_support = true`.
  - C1 adds `host_unified_known` beside the cached field, true only when the
    query returned.
  - `mmap_support` requires it. The other readers of `host_unified_memory`
    (the VRAM budget's iGPU cap, `common.cpp:889-936`) keep today's semantics.
- **What changes under `-lm auto -lzm auto`**, for a device set of discrete
  cards only (`level_zero:0`, `:1`, `:0,1`):
  - `ml.use_mmap` stays true through `lm:2245-2254`, and the log reads
    `load_mode = mmap` instead of `none`.
  - Lazy `auto` stays `auto`, so every `TENSOR_READ_LAZY` tensor above 4 GiB
    becomes a lazy table (§2.5). On Qwen that takes the 50.66 GiB PLE table out
    of the pinned pool **under default flags**. This is the only material
    change of commit 1.
  - `create_tensor` sees `use_mmap = true`. The `-ot =CPU` mmap warning can
    print (`lml:1434-1439`), and the host-buft demotion stays disabled for SYCL
    (§2.2). Buffer types are otherwise identical.
  - `lm:2693` still clears `use_mmap`, so every non-lazy weight is still read
    into `SYCL_Host`. Commit 1 deliberately does **not** remove that line:
    without the tier, the only effect would be a copy from page cache into
    the same pinned buffer, doubling the transient footprint.
- **An unpinned selector** that includes the iGPU makes the iGPU a model
  device, and both fallbacks fire exactly as today. That is the intended
  behaviour, and it is never run on this host (CLAUDE.md, llama.cpp-403s).
- **th32 drops its own flip** (lead ruling §T13, point 1). th32's T6 still
  resolves its lazy mode before the `auto` fallback; commit 1 does not depend
  on that.

### 3.1 Also in commit 1: the per-file lazy-range record (§F2a I-9, §M49)

**Why it exists.** Four consumers need the byte ranges of the
`TENSOR_READ_LAZY` tensors:

- `lazy_read`, for the CPU lazy path;
- th32's inventory, after T6;
- the tier's exclusion (§4.6);
- `llama_mmap`'s prefetch, `MADV` and lock exclusion (§4.6; §T13c permits
  feeding it from 5efe's code).

Today the ranges live only in `lazy_read`'s private map (§2.5). th32 T6
diverts SYCL tables *out* of `lazy_read`, so after T6 that map would hold no
SYCL table. The tier would then receive an empty exclusion set, and the
loader's mapping would `WILLNEED` the complement of that empty set, which is
every byte of every shard, the 54.4 GB table included (r1 addendum I-9). One
record, filled independently of who defers the read, removes that failure
mode.

**The record.** `llama_lazy_range_record`, one per loaded file, owned by 5efe
and held by `llama_model_loader`:

- **Rows.** One row per tensor created with `TENSOR_READ_LAZY`: (tensor name,
  file index, offset, size, `deferral`).
- **`deferral`** is `EAGER`, `LAZY_READ` or `STORAGE`.
  - `EAGER`: the flag is set, but the tensor is read normally, for example
    under `-lzm off`. It is an ordinary weight, and no consumer excludes it.
  - `LAZY_READ`: upstream's CPU lazy path.
  - `STORAGE`: th32's SYCL diversion, from T6.
- **Where it is filled.** `create_tensor` fills the row at the one place the
  flag is examined (`lml:1774-1777`), from the `llama_tensor_weight`, before
  and independently of any consumer's decision. The decision only sets
  `deferral`.
- **Its API.** `deferred_for_file(idx)` returns the sorted, merged ranges of
  every non-`EAGER` row of that file. `has(tensor)` and `deferral(tensor)` are
  the per-tensor queries.

**Consumers keep no copy.**
- `lazy_read` keeps its `mode` and its decision (`add` without storage).
  `any()`, `has()` and `for_file()` become queries on the record, filtered to
  `LAZY_READ`, and its private `ranges` and `tensors` members are deleted.
- `init_mappings` passes `record.deferred_for_file(idx)` to `llama_mmap`
  (`lml:1890-1891`).
- th32's inventory reads each table's (file, offset, size) from the record
  (th32 rev 2s §4.1).
- The tier hook's `lazy_ranges` argument (§4.3) is
  `record.deferred_for_file(idx)`.

**What it changes on its own.** Nothing observable in C1. The ranges and
decisions are the ones `lazy_read` computes today, and the edge now held by
th32 T6 on 5efe is a data edge on C1 alone (§M49).

---

## 4. The design: a cache-owned `FILE_MAPPED` tier (item 3)

### 4.1 The tier and its object

`alloc_tier` gains `FILE_MAPPED`, distinct from th32's `HOST_CPU`:

| tier | memory | reader | charged to the host ledger |
|---|---|---|---|
| `DEVICE_VRAM` | device USM | that device | no (VRAM budget) |
| `HOST_PINNED` | USM host, pinned pool | CPU; a device for DMA staging only | yes |
| `HOST_CPU` (th32) | anonymous, `mlock`ed | CPU only | yes |
| **`FILE_MAPPED`** (new) | one `PROT_NONE` VA reservation with `mmap(PROT_READ, MAP_SHARED \| MAP_FIXED)` file segments over it, made by the cache | **CPU only** | **no**, unless locked (§4.5) |
| `UNREGISTERED` (renamed from `MMAP_TRACKED`) | not an allocation: the query result for a pointer the registry does not know | nobody may allocate it | n/a |

**Enum value.** `FILE_MAPPED` takes the next unused `alloc_tier` value at the
time C2 lands. If th32 T1's `HOST_CPU` has landed, that is 4; otherwise it is
3, and T1 then takes the next one. Every census and test keys on names,
never on values (r1 m-10).

**The raw wrappers.** These live in one new, device-free translation unit,
`ggml/src/ggml-sycl/file-mapping.cpp`, which includes no SYCL headers. Host
tests link it alone (§7).

- `unified_cache_raw_map_file(fd, file_size, reservation, segments)`:
  - takes a **`dup` of the loader's own open fd** (`llama_file::file_id()`,
    `src/llama-mmap.cpp:412-423`), passed down through the hook (§4.3);
  - `fstat`s it and refuses if the size differs from the size the loader
    recorded (`register_file_identity`, `lml:1553-1556`);
  - reserves `[page_floor(first), page_ceil(last))` as `PROT_NONE`,
    `MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE`;
  - maps each segment of §4.6 over the reservation with `MAP_FIXED`.

  Reusing the loader's open file rather than reopening by path removes the
  same-size-replacement gap (r1 m-5). The owner keeps the fd, because
  `POSIX_FADV_DONTNEED` after staging needs it (§4.6), and closes it at
  release.
- `unified_cache_raw_lock_file_segments(base, lock_segments)` and
  `unified_cache_raw_unmap_file(reservation)` are the only lock and release.
  The release is **one `munmap` of the whole reservation** (r1 I-2).
- All three are added to the canonical contract §3 allowlist, next to
  `unified_cache_raw_malloc_host` and th32's `HOST_CPU` wrapper.
- A new source gate, `test-sycl-raw-mmap-allowlist-source`, makes that
  enforceable, which the allowlist entry alone is not (r1 m-11). It fails on
  any `mmap(`, `munmap(`, `mlock(` or `madvise(` call in `ggml/src/ggml-sycl/`
  outside a listed site.
  - Its list is today's sites (`uch:2066`, `:2071`, `:2107`; `gs:11195`,
    `:11200`, `:11261`; `dpct/helper.hpp:1362`, `:1389`), the new TU, and
    th32 T1's `HOST_CPU` wrapper when it lands.
  - The gate is shared with th32 T1.

**The owner-first front door.**
- `unified_allocate_file_mapping(req)` mints the `alloc_owner_control` first,
  as `unified_allocate_owner` does, then maps.
- The class is **`EXTERNAL_EXACT`**: exact, independently released, and not a
  slice of cache backing (canonical contract §3.1). The two-site
  `CACHE_BACKING` allowlist is untouched.
- There is **one owner and one registry record per reservation**:
  `ptr = reservation base`, `size = reservation length`. The segments are not
  separate records (r1 I-2(c)).

**The registry type.**
- `alloc_type` gains `FILE_MAPPED`. `unified_free_record` handles
  `alloc_tier::FILE_MAPPED` in its own branch, which calls the unmap wrapper.
- That branch sits before both the `MMAP_TRACKED`/`UNREGISTERED` no-free case
  (`uc:14812`) and the `alloc_type::MMAP` `::free` branch (`uc:14922-14929`),
  so a `FILE_MAPPED` record can never reach `::free` (r1 m-4).
- C2's test proves the ordering with a private `::free`-branch counter.

**`memory_location::device_mappable()`** (th32 §3.1 adds it) is false for
`FILE_MAPPED`. `host_accessible()` stays true, because the CPU reads it.

**Identity.** There is one reservation per (model, file index, device
buffer), keyed by the file identity the loader already publishes (path and
size, `lml:1553-1556`). Two models mapping the same file get two
reservations over one page cache; sharing them is not needed.

### 4.2 Ownership and teardown

- **Who holds references.**
  - The `SYCL_Mapped` ggml buffer (§4.3) holds the reservation's
    `mem_handle` for the model's lifetime.
  - Every cache entry registered over mapped bytes holds it through
    `storage_owner`. That field is a `std::shared_ptr<void>` (`uch:2514`), not
    a `mem_handle`, so the tier stores
    `std::make_shared<mem_handle>(reservation_handle)` in it. The owner's
    destructor drops the handle, and release still goes through the
    `alloc_owner` (r1 m-18).
  - This replaces today's `non_owning_external_host` registration for these
    entries.
  - Every `CpuExpertPool` task holds a reference through its `weight_lease`,
    which already moves into the worker (`gs:76027`).
- **Order at unload.**
  - `llama_model` frees its buffers.
  - The lifecycle unload releases the model's cache entries **on every
    device** (canonical contract §12).
  - In-flight CPU tasks drop their leases when they finish.
  - The last reference unmaps. Nothing waits for anything, and nothing unmaps
    under a reader.
- **Independent of `llama_mmap`.** The cache makes its own reservation instead
  of adopting the loader's mapping, for two reasons:
  - Adoption would leave the release in llama's hands (P1).
  - `load_all_data`'s final call `munmap`s the head and tail of each loader
    mapping (`lml:2236-2247`), which would pull pages from under a cache entry.

  Two mappings of one file share the page cache, so the cost is address space
  only. Lead ruling §T13c lets the tier do its own exclusion "from your own
  code".
- **The loader's own mapping on the SYCL path.**
  - `init_mappings` is still called, for lazy tables (until th32 lands) and
    for CPU-device buffers.
  - C4 changes its call at `lm:2702` so that a SYCL load passes prefetch 0.
    With `use_mmap` surviving, `init_mappings(true)` would otherwise set
    `prefetch_size = -1` (`lml:1888`), and `llama_mmap` would `WILLNEED` every
    non-lazy byte of every shard (`src/llama-mmap.cpp:535-539`) (r1 m-6).
  - The tier owns the prefetch policy for SYCL weights (§4.6).
- **Shutdown and `PER_DEVICE` caches** (r1 m-18).
  - A `SYCL_Mapped` buffer belongs to one device, like `SYCL_Host` today
    (`gs:42761-42785`). Its reservation is minted by that device's cache.
  - Another device's cache may hold leases on it, through `storage_owner` on
    entries that device registered over the mapped experts (the multi-device
    MoE split).
  - The required order is: **every model unloads, releasing its entries on
    every device, before any device's cache tears down.** The pre-teardown
    census of one cache therefore never sees a lease held by another cache's
    entries.
  - A live `FILE_MAPPED` control at cache teardown is a leaked lease. It is
    reported by the census like any `EXTERNAL_EXACT` control (canonical
    contract §3.1).
  - The `__cxa_finalize` abandon path (`uc:996-1007`) is safe: process exit
    unmaps.

### 4.3 One published fact, the buffer type, and the loader change

**The fact, resolved before the early plan** (r1 I-5, §F2).
- C4 moves the two caps loops (`lm:2245-2265`) above the early plan
  (`lm:2228-2231`). They depend only on `devices` and `params`.
- The resolved answer is published once, on the SYCL load transaction:
  `load_maps_files = ml.use_mmap after resolution`. It is false under
  `-lm none`, and false when an iGPU in the device set forced `none`.
- The planner reads it to give every inventory entry a `host_backing` (§4.4).
- The loader reads the planner's answer through a new hook,
  `planned_host_backing(name)`. Non-name-indexed tensors such as MoE experts
  get the transaction default, which the planner derived from the same fact.
- `create_tensor` never consults `use_mmap` for a SYCL weight.

**The buffer type.** `ggml_backend_sycl_mapped_buffer_type_for_device(dev)`
returns a per-device buft named `SYCL_Mapped`, with its own `get_name`,
because `supports_buft` keys on the name function (`gs:108779-108812`).

- `is_host` is true, so the CPU backend can execute an op on it when SYCL
  declines (the planner-host decline, `ggml_sycl_op_is_planned_on_host`).
- `supports_buft` is true for SYCL devices, exactly as for `SYCL_Host`
  (`gs:108810-108812`). Ops on device-materialized weights stay on SYCL and
  resolve the cache's VRAM copy by identity, as they do today.
- `set_tensor`, `memset_tensor` and `clear` abort, naming the tensor. The
  mapping is read-only, and weights are never written after load.
- `init_tensor` registers the host weight the way `SYCL_Host` weights are
  registered today (`lml:1542-1561`).

**The hook.** The buffer is created by a new proc-address hook:
`ggml_backend_sycl_mapped_buffer_from_file(dev, file_idx, dup_fd, file_size,
tensor_ranges, lazy_ranges)`, whose `lazy_ranges` is the record's
`deferred_for_file(file_idx)` (§3.1).
- The loader calls it **instead of** `ggml_backend_alloc_ctx_tensors_from_buft`
  for each (`SYCL_Mapped`, file) pair.
- It mirrors upstream's `buffer_from_host_ptr` shape (`lm:2748-2767`).
- `[first, last)` is `get_mapping_range` over the context's tensors
  (`lml:1908-1923`).
- `load_all_data` points each tensor at `base + (weight->offs - page_floor(first))`
  without reading it. It uses its mapping branch (`lml:2099-2128`), keyed on
  the buffer instead of on `ml.mappings`, and skips the loader's `lmlock`
  growth for this buffer (§5.1).
- The one contiguous reservation (§4.1) is what makes that single-base
  arithmetic valid (r1 I-2(b)).

**Contents.** `SYCL_Mapped` holds **every SYCL weight whose host backing is
`FILE_MAPPED`**: device-planned weights (the mapping is their staging source),
host experts, and host dense weights in file layout. On Qwen that is the whole
late inventory, 25231.2 + 102383.5 = 127614.7 MB (`run2.log:2821`), not only
the host experts (r1 I-8b).

**Loader sites C4 changes.** The source contract enumerates all of them
(r1 m-6):

| site | change |
|---|---|
| hook structs `lml:31-51`, `lm:76-117` | add the mapped-buft, `planned_host_backing` and `set_load_maps_files` hooks |
| caps loops `lm:2245-2265` | move above the early plan (`lm:2228-2231`); publish `load_maps_files` |
| `select_weight_buft` `lml:1128-1135`, the `prefer_host_weights` path | `SYCL_Mapped` when `planned_host_backing == FILE_MAPPED`, else `SYCL_Host` as today |
| `weight_buft_supported` `lml:978-985` | its `SYCL_Host` short-circuit also accepts `SYCL_Mapped` |
| `make_cpu_buft_list` `lm:1765-1779` | on a SYCL load that maps files, the front host buft is the device's `SYCL_Mapped`, so `token_embd` lands there |
| `allow_host_buft_with_mmap` `lml:1503-1518` | unchanged for `SYCL_Host`; `SYCL_Mapped` is never demoted |
| buffer creation `lm:2740-2791` | the `SYCL_Mapped` branch calls the hook; the `mlock_bufs` block (`lm:2782-2787`) skips `SYCL_Mapped` buffers, which lock through the tier (§5.1) |
| `lm:2693-2696` | deleted |
| `init_mappings` call `lm:2702` | prefetch 0 on a SYCL load |
| `load_all_data` mapping branch `lml:2099-2129` | addresses `SYCL_Mapped` tensors from the buffer base; no `lmlock` growth for them |

With `-lm none`, or any load where `load_maps_files` is false, `SYCL_Host`
remains, with today's behaviour.

### 4.4 Which weights stay mapped, and the planner's split

A byte stays mapped only if **the CPU executes it and its materialized layout
is the file layout** (P3 with layout-follows-residency). Everything else is
copied into a cache allocation at load.

| weight | executor | materialized layout | `host_backing` | tier of the executed bytes |
|---|---|---|---|---|
| device-planned dense or expert | GPU | per-device (SOA, XMX, WOQ...) | `FILE_MAPPED` (its staging source) | `DEVICE_VRAM`, staged once from the mapping (§5.3); the mapped pages are dropped after staging (§4.6) |
| host MoE expert, AOS | CPU (`CpuExpertPool`) | AOS = file | `FILE_MAPPED` | **`FILE_MAPPED`, served in place** |
| host MoE expert needing a CPU reorder (`register_host_aos_expert`'s SOA/XMX branch) | CPU | reordered | `FILE_MAPPED` source | copied: into `HOST_CPU` once th32 T1 lands, pinned until then |
| planner-host dense, `GGML_SYCL_HOST_WEIGHT_LAYOUT=cpu` (default) | ggml-cpu | `CPU_REPACK` | `CPU_OWNED` | copied by ggml-cpu, outside the cache (ticket below) |
| planner-host dense, `=plain` | ggml-cpu | file | `FILE_MAPPED` | `FILE_MAPPED` (`is_host`, so the CPU backend runs it) |
| `token_embd` (CPU `get_rows`) | CPU | file | `FILE_MAPPED` | `FILE_MAPPED` |
| `TENSOR_READ_LAZY` table | CPU / th32 gather | row-gathered | none (not a SYCL inventory entry) | **never mapped by the tier**, nor prefetched or locked (§4.6); th32's STORAGE or `HOST_CPU` |

Under P3 no host **weight** needs pinned memory, because the GPU never reads
one. So in mmap mode the pinned pool holds only staging windows, KV, scratch,
compute buffers, and the reordered experts until th32 T1 lands.
`PINNED` host backing survives only when `load_maps_files` is false.

**Ticket to file:** `CPU_REPACK` weights are allocated by ggml-cpu's
`ggml_backend_cpu_repack_buffer_type` (`ggml-cpu/repack.cpp:5238-5250`),
outside the unified cache and its ledger. That is a P1 gap independent of
mmap (related to `llama.cpp-qptd`). The natural fix is a `HOST_CPU`-backed
repack buffer after th32 T1.

**The planner split** (the lead's point (a); moved into C4 by r1 C-1 and
§F2). `placement_entry` gains `host_backing { PINNED, FILE_MAPPED, HOST_CPU,
CPU_OWNED }` for every entry. The planner derives it once from
`load_maps_files` and the table above, and the plan then carries:

| plan term | sums | charged to the ledger | sizes |
|---|---|---|---|
| `weight_host_bytes` | host-executed entries backed `PINNED` or `HOST_CPU`, **plus the reordered copies** | yes | the WEIGHT zone (`uc:27670-27672`) |
| `host_cpu_owned_bytes` | `CPU_OWNED` entries | yes (anonymous memory, P3 counts it) | nothing: not in the zone (today they inflate it) |
| `host_mapped_bytes` | host-executed `FILE_MAPPED` entries | only when locked (§4.5) | nothing |
| `host_mapped_source_bytes` | device-planned `FILE_MAPPED` sources | no | nothing (transient page cache during staging) |

Every accumulation site of `weight_host_bytes` in the planner
(`uc:25928`, `:25933`, `:28136`, `:28196`, `:28230`, `:28322`, `:29554`,
`:29798`) tests the entry's `host_backing` before adding. The C4 source
contract lists them. `[PLACEMENT] Totals` prints all four terms.

**Non-weight `SYCL_Host` buffers.**
- Every `SYCL_Host` buffer, including the context's input and output buffers
  created after the load, allocates with role `WEIGHT` while
  `weights_evictable` holds (`gs:42645-42646`). Its `HOST_COMPUTE` category
  routes it to the WEIGHT zone (`uc:15361-15372`).
- Today those buffers fit in the 20% headroom over 100 GB of pinned experts.
  After the split they must fit in the named reserve term,
  `k_host_runtime_reserve_bytes` = 64 MiB (`uc:27103`).
- For Qwen at `-c 4096` with one sequence that holds. The output buffer is
  `n_vocab x 4 B` per output: 248320 x 4 B = 0.95 MiB (`run2.log:174`).
- It does not hold for an all-logits run (`n_outputs = n_ubatch = 512` gives
  485 MiB). So C4 sizes this term explicitly at the runtime KV update, which
  knows `n_ctx` and `n_ubatch`, as `n_ubatch x (n_vocab + n_embd) x 4 B` plus
  the input tensors. If rnh3 (exact host WEIGHT terms) has landed, the term
  goes through rnh3.

**Pre-registered Qwen WEIGHT zone after C4** (r1 C-1):
- Host experts are all AOS, registered in place (`gs:33437-33457`), so
  `weight_host_bytes` = 0 and `host_cpu_owned_bytes` = 0. The host total is
  experts only: 61671 host tensors = 20557 groups x 3 (`run2.log:2824`).
- `host_zone_weight_bytes = 0 x 1.2 + 64 MiB + runtime term`, with a runtime
  term of at most 64 MiB at `-c 4096`. So `[HOST-ARENA] configured zones:
  WEIGHT=` reads **64.0-128.0 MB**, against 122924.2 MB in the baseline
  (`run2.log:2893`).
- The zone total is then about 64 + 64 + 1719.2 + 7775.9 = 9623.1 MB (KV,
  STAGING and SCRATCH are unchanged), and the pre-size commits **at most
  6 chunks, 12288 MB**, against 68 chunks, 139264 MB (`run2.log:2900`).
- Falsifier: a WEIGHT zone above 128 MB must equal 1.2 x the reordered-expert
  bytes, and the placement report must print that sum. Otherwise the split
  missed an accumulation site.

### 4.5 The budget, the ledger, the working set, and the host-fit gate

**What the ledger charges.** th32 §4.2 makes `host_ledger_budget()` the one
host budget, and every resident host byte charges it. `FILE_MAPPED` bytes are
clean page cache: the kernel can drop and re-read them, and `MemAvailable`
counts them as available. So:

- **`-lm mmap` / `auto` (unlocked):** mapped bytes are **not** charged. They
  appear in the plan as `host_mapped_bytes`, and the load prints them.
- **`-lm mmap+mlock` (locked):** the tier locks its host-executed lock
  segments (§4.6, §5.1), never a lazy range, and those bytes **are** charged
  at full size, because locked pages are no longer reclaimable.
  - A lock failure refuses the load, naming `errno`, the bytes and
    `RLIMIT_MEMLOCK`, as th32's `host-cpu-lock-failed` does.
  - No new environment variable is added: the policy is the existing load
    mode.

**The resident working-set check** (§F1 Q1: WARN, not refuse).
- An unlocked mapping keeps its pages only while memory allows.
- The plan computes:

  ```
  resident_room = MemAvailable_pre - expected_host - k_host_p3_floor
  mapped_ws     = host_mapped_bytes            (host-executed ranges only)
  ```

- If `mapped_ws > resident_room`, the load logs one WARN,
  `[HOST-PLAN] mapped host weights exceed the page cache that can stay
  resident: mapped_ws=... resident_room=... (MemAvailable_pre=...
  expected_host=... floor=...)`, and proceeds.
- The plan report (7v5c's dry-run and `[PLACEMENT] Totals`) carries the same
  three figures.
- A mapped overflow costs refaults, not an OOM (§5.2). Refusal stays for
  pinned and anonymous bytes, through th32 T2 over 7v5c.

**The cold-expert interface** (§F1 Q2, r1 m-9). 5efe defines the tier and its
terms and adds no cold-set logic. What yl3v and 7v5c consume is:

- per expert entry: `host_backing`, and whether it is host-executed;
- `host_mapped_bytes` and `resident_room` from the plan;
- permission to request `FILE_MAPPED` as the backing of a cold host expert,
  instead of an `O_DIRECT` STORAGE read, if yl3v chooses it.

yl3v's objective must read `mapped_ws` against `resident_room`, not only the
charged bytes. Once host experts are uncharged, a search over charged bytes
alone finds DeepSeek's cold set empty and leaves its experts faulting from USB
behind a single WARN. That policy is yl3v's to set; 5efe only guarantees the
terms exist and are published.

**The headroom and the monotone budget** (the lead's point (c); r1 I-4, §F2).
One named expression, in the one function:

```
k_host_p3_floor      = 29 GiB                       // this host's measurement:
                                                    // 9g22's 24 GiB P3 floor + 5 GiB margin (th32 §4.3)
host_headroom(total) = max(total / 32, min(k_host_p3_floor, total / 4))
host_ledger_budget() = min(total x pct, MemAvailable - host_headroom)   when MemAvailable > host_headroom
                     = refusal "host-headroom-exhausted"                 otherwise
```

- **Monotone.** The `available/2` branch at `uc:4227-4230` goes. The budget
  never rises when `MemAvailable` falls, and at zero headroom-free memory
  every pinned request is refused with a named reason that prints
  `MemAvailable`, `MemTotal` and the headroom.
- **Unknown total.** The `total_mem == 0` branch (`uc:4217-4219`) keeps its
  `min(8 GiB, available/4)` reserve, which is monotone once the `/2` cliff is
  gone.
- **Why the P3 constant.** P3 (9g22's gate precondition, which th32 §4.3 makes
  bind every load) requires `MemAvailable_pre - expected_host - 5 >= 24` on
  this host. A ledger that leaves only 8 GiB admits plans P3 refuses, which
  th32 §4.5 has to argue around.
  - With one named constant, the ledger and P3 read the same number.
  - th32 §4.3 is asked to cite `k_host_p3_floor` rather than the literal
    `24 + 5`.
  - The `min(..., total/4)` cap keeps this host's policy from becoming a
    universal allocation cap on smaller hosts.

| host | `MemTotal` | headroom | `MemAvailable` | budget (new) | budget (old) |
|---|---:|---:|---:|---:|---:|
| this host, now | 246.9 GiB | 29.0 | 150.8 | **121.8** | 142.8 |
| this host, baseline | 246.9 | 29.0 | 154.8 | 125.8 | 146.8 |
| this host, near the floor | 246.9 | 29.0 | 29.5 / 28.9 | 0.5 / refusal | 21.5 / 20.9 |
| 62 GiB NAS | 62.0 | 15.5 | 50.0 | 34.5 | 42.0 |
| 32 GiB host | 32.0 | 8.0 | 20.0 / 7.9 | 12.0 / refusal | 12.0 / 3.95 |

(On this host `total x 90%` = 222.2 GiB never binds.) Until th32 T2 lands,
the change applies to the `[HOST-ARENA]` formula at `uc:4200-4231`; T2 then
lifts it into `host_ledger_budget()`.

**The host-fit gate** (the lead's point (b)) is not duplicated here.
- The refusal is th32 T2's `host-budget-exceeded`: WARN-only in phase (a),
  enforcing in phase (b).
- The number is 7v5c's dry-run.
- 5efe supplies their inputs: `host_backing` per entry, the four terms of §4.4,
  the working-set figures, and the locked/unlocked policy. 7v5c's printout
  gains the mapped lines and the working-set check.

### 4.6 The mapping shape, prefetch, locking, and the lazy ranges (§T13a, §T13c, th32 r15 m-4)

- **5efe owns the exclusion, because it owns the mapping.** th32 stays off
  mmap entirely (§T13c), and after T6 its tables leave `lazy_read`. The
  exclusion set therefore comes from the **lazy-range record** (§3.1), never
  from `lazy_read`'s set. Two exclusion sites read the record:
  `deferred_for_file(idx)` goes to the tier through the hook's `lazy_ranges`
  (§4.3), and to `llama_mmap` through `init_mappings`. A table diverted to
  STORAGE is excluded exactly as a `LAZY_READ` one is.
- **Three segment sets, one pure function** (r1 I-2, §F2).
  `file_mapping_segments(tensor_ranges, lazy_ranges, page)` returns:

  | set | rounding | purpose | bound on lazy bytes |
  |---|---|---|---|
  | map segments | **outward** to pages, including at a lazy boundary | every non-lazy tensor byte is addressable (GGUF alignment is 32 B, so a lazy and a non-lazy tensor can share a page) | at most one page of lazy bytes mapped per lazy edge |
  | `WILLNEED` segments | **inward** | host-executed ranges only; never a device-only range, never a page holding a lazy byte | 0 |
  | lock segments | **inward** | host-executed ranges, under `-lm mmap+mlock` only | 0 |
  | `MADV_RANDOM` edge windows | **inward** from each lazy boundary into the map segment, plus the shared page | disables fault read-around next to a lazy range, on **both** sides of it | n/a (advice only) |

  (§F2a I-2(a): only map segments round outward. Every advice and lock set
  rounds inward, so no advice or lock ever reaches a lazy page other than the
  shared page's `MADV_RANDOM`.)

  - Pages strictly between map segments stay the reservation's `PROT_NONE`
    anonymous memory. They are never faulted, and a tier read of one is a
    `SIGSEGV`: a loud bug, not a silent read.
  - The complement is the same computation as `ranges_complement`
    (`src/llama-mmap.cpp:444`), but that helper is file-static in libllama,
    which ggml-sycl cannot call. `file_mapping_segments` lives in the
    device-free `file-mapping.cpp`.
- **Fault read-around** (r1 m-13, addendum m-19).
  - Linux's synchronous mmap read-around is **centred on the fault**:
    `start = pgoff - ra_pages/2`, `size = ra_pages`. It is per file, not per
    VMA. So a fault near either edge can read into a neighbouring lazy range:
    forward from a segment that ends at it, and **backward** from a segment
    that starts after it. That covers CPU execution faults and the staging
    bounce's `memcpy` faults alike.
  - The tier sets `MADV_RANDOM` on a window of `W = max(2 x read_ahead_kb,
    2 MiB)` at **both** edges of every lazy range: the last `W` of the map
    segment before it, the first `W` of the map segment after it, and the
    shared pages. That splits the VMAs and disables read-around there.
  - Device-only ranges are also staged through the mapping, so the same edge
    advice covers them.
- **The second residual path: the loader's own whole-file mapping** (addendum
  m-19).
  - `llama_mmap` still maps whole files during the load. `load_all_data` reads
    the tensors that are not in `SYCL_Mapped` through it: `CPU_REPACK` and
    plain CPU sources, and lazy tables on the CPU path before th32.
  - Today that mapping advises `MADV_RANDOM` only over the lazy range itself
    (`src/llama-mmap.cpp:541-543`), and it sets `POSIX_FADV_SEQUENTIAL` on the
    fd (`:475`), which enlarges the read-around window. A fault in the
    neighbouring VMA can therefore read around into the lazy range.
  - C4 applies the same both-edge window to `llama_mmap`: its lazy-range
    advice covers `[lazy_begin - W, lazy_end + W)`, from the record, not the
    lazy range alone.
  - Bound: with both mappings advised, the lazy bytes either mapping can
    fault in are the shared pages, at most one page per lazy edge per mapping,
    plus whatever bcachefs reads at extent granularity (256 KiB, th32 §2.4).
    The gate's 1 MiB guard band sits outside both.
  - The residency gate below measures what remains.
- **No `MAP_POPULATE`, no `MADV_HUGEPAGE`** from the tier. Huge pages on a
  file mapping depend on the filesystem's large-folio support, which is not
  established for bcachefs here. th32 §2.4 measured that bcachefs reads whole
  256 KiB extents anyway.
- **After staging**, device-only ranges get `POSIX_FADV_DONTNEED` through the
  owner's `dup` fd (§4.1). Their page cache goes back to the host experts'
  working set. A re-materialization after a VRAM eviction re-reads them.
- **Locking** (th32 r15 m-4, addendum I-10, §F2a).
  - **The tier.** It calls `mlock(2)` directly, once per inward-rounded,
    host-executed lock segment, in `unified_cache_raw_lock_file_segments`.
    Inward rounding matters because `mlock` rounds a mid-page start down to
    the page. The tier never uses `llama_mlock`, whose `grow_to` locks
    `[addr + size, target)` contiguously from the mapping base
    (`src/llama-mmap.cpp:826-840`).
  - **`mlock_bufs`.** `SYCL_Mapped` is excluded from the `mlock_bufs` block
    (`lm:2782-2787`). Without that, `use_mlock && is_host` would `grow_to` the
    whole buffer: every device-only source (about 25 GB on Qwen) and the
    reservation's `PROT_NONE` holes.
  - **The loader's prefix lock is fixed in 5efe**, because 5efe is what turns
    mmap on for SYCL (§F2a I-10). `lml:2120-2123` stops calling
    `grow_to(weight->offs + n_size)`. It calls a new
    `llama_mlock::lock_range(offs, size)` instead, which locks the tensor's own
    range: outward to pages, except inward at any end whose page holds a
    record-deferred byte. `llama_mlock` records the ranges it has locked, and
    `grow_to` remains only for whole-buffer locks (`mlock_bufs`).
  - Qwen is immune (its table is all of shard 3). DeepSeek's layouts are
    exposed (Q3_K_M shard 2; Q2_K shards 1 and 4), which is why both lock REDs
    put a lazy range before a locked range.
- **Until th32 lands**, lazy tables keep today's path (§2.5): the loader's lazy
  mapping, `MADV_RANDOM`, no prefetch, no lock.
  - The tier changes none of it, and the loader's mapping gets prefetch 0 on
    SYCL (§4.2), which can only reduce what is prefetched.
  - The tier never double-maps a lazy range beyond the one-page edge bound
    above.
  - The upstream prefix lock that §2.5 describes is fixed in C3 (above), so a
    plain `CPU_Mapped` tensor after a lazy range under `-lm mlock` no longer
    locks the range either.
- **The residency gate** (C3; r1 I-1, §F2). It measures the page-cache
  residency of the **file**, never `mincore` on the tier's hole.
  - **Fixture.** A 64 MiB file in the build directory: tensor A (16 MiB), a
    lazy range (24 MiB), then tensor B (16 MiB), with 32 B-aligned, deliberately
    non-page-aligned boundaries. The lazy range is larger than any readahead
    window.
  - **Precondition.** `POSIX_FADV_DONTNEED` over the whole file, then the probe
    must read about 0 resident. If it does not, the test exits 77: the
    directory cannot be evicted (tmpfs, or a filesystem that ignores the
    advice).
  - **Probe.** `cachestat(2)` on the file's fd (Linux 6.5+; this host runs
    7.1). The fallback is `mincore` over a **test-owned**, separate
    `PROT_READ` mapping of the file, which reports the page cache of those
    file pages whichever mapping faulted them.
  - **Arm 1 (the property).** Map through the tier with the lazy hole, apply
    the `WILLNEED` segments, and read every byte of A and B, including the
    bytes that share a page with the hole. That reads outward-rounded bytes
    without a `SIGSEGV`, which also tests I-2(a). Then the probe over the
    hole's interior, a 1 MiB guard band in from each edge, must read **0
    resident pages**.
  - **Arm 2 (the mlock case).** Arm 1 under the lock path, with the lazy range
    placed *before* a locked tensor. Same bound. It exits 77 if
    `RLIMIT_MEMLOCK` is below the fixture size.
  - **Arm 3 (negative control).** A `GGML_SYCL_PRIVATE_TESTING` mutant that
    passes the hole as non-lazy must read resident pages in the interior. This
    shows the gate can fail.
  - **Arm 4 (synchronous positive control).** `pread` 1 MiB inside the hole's
    interior, then the probe over that MiB must read resident. There is no
    polling and no `WILLNEED` race.
  - **Arms 5 and 6 (read-around, forward and backward; addendum m-20).**
    - Arm 5 faults the last mapped page before the hole, arm 6 the first
      mapped page after it. Both are the real faulting pages, not addresses
      the tier does not map.
    - With the edge `MADV_RANDOM` in place, the hole interior must read 0.
    - Each has a **control twin** with no edge advice, which must read
      resident pages inside the hole. This shows the arm could fail.
    - Precondition: `read_ahead_kb` of the fixture's backing device (the bdi
      of its `st_dev`) must be nonzero. Otherwise both arms exit 77, since
      with no read-around they would pass vacuously.
    - Measured through the probe above, as §F2 I-1 requires.
- **The loader-side gates** (addendum I-9, I-10). The same fixture and probe
  are driven through libllama, with no SYCL:
  - **`llama_mmap` prefetch exclusion.** Construct `llama_mmap` with
    `prefetch = -1`, the `init_mappings(true)` case, and lazy ranges from a
    record. Wait 10 s, then the hole interior must read 0. The mutant is an
    **empty** record (the T6 diversion with ranges taken from `lazy_read`); it
    must read resident within the same 10 s, polled.
  - **The loader's ranged lock.** Lock a tensor that sits after the lazy
    range through `llama_mlock::lock_range`; the hole interior must read 0. The
    mutant is `grow_to`, the whole-prefix lock, which must read resident. The
    test exits 77 if `RLIMIT_MEMLOCK` is below the fixture size.
  - **On Qwen** the lead gate is `fincore` on shard 3, which holds only the
    table (§2.5), against the bound pre-registered in §7 C4.

### 4.7 th32 interface, and the figures that move

- **Separate tiers, separate readers.** STORAGE reads through th32's own
  `O_DIRECT` fd and never maps (§T13a, §T13c); `HOST_CPU` is anonymous.
  `FILE_MAPPED` never holds a lazy table. th32 §5.1's argument that "a HOST
  residency backed by mmap is not HOST at all" is consistent with this design:
  `FILE_MAPPED` is not charged as resident, and the plan says so.
- **T1's census** ("`MMAP_TRACKED`: legacy file mmap tracked for weights",
  th32 §3.1) changes: `MMAP_TRACKED` becomes `UNREGISTERED`, and `FILE_MAPPED`
  is a real tier. The sites are listed in §4.8. th32 T1 and 5efe C2 can land
  in either order, because every site keys on names and the enum value is
  "next free" (§4.1).
- **Figures in th32 §4.5 that move** once 5efe lands. th32 re-pins them:
  - "Host experts, exact" (Qwen 98.7-101.7 GiB, DeepSeek 164.7-167.7 GiB
    today) moves out of step 2 into `host_mapped_bytes` under `-lm mmap`.
    Under `-lm mmap+mlock` it stays.
  - "Other host weights (`token_embd`)" moves the same way.
  - The WEIGHT zone term shrinks to 1.2 x the reordered experts, plus rnh3's
    named terms, plus the runtime term of §4.4. For Qwen that is about 64 MB
    (§4.4).
  - Every "`MemAvailable_pre` needed" row drops by the moved bytes.
    - For Qwen after (s), a HOST table would need about
      180.9 - 99.3 = 81.6 GiB of `MemAvailable_pre` for step 2 plus the table,
      before the working-set check.
    - With the experts mapped, the working-set check (§4.5) then decides, and
      the PLE table and the experts compete for one page cache. That trade is
      th32's to re-pin, through 7v5c.
  - §2.3's `mmap_support` bullet and T6's flip move to 5efe commit 1.
  - The ledger headroom (§4.5) is `max(total/32, min(29 GiB, total/4))`,
    which is 29 GiB here, as P3 already carried. On this host the planner's
    fit column moves to the P3 column; on smaller hosts it moves less.

### 4.8 The tier-switch census (sites 5efe touches)

`MMAP_TRACKED` today; every site is renamed to `UNREGISTERED` or taught
`FILE_MAPPED`:

| site | today | after |
|---|---|---|
| `uch:5593` enum | `MMAP_TRACKED = 2` | `UNREGISTERED = 2`, plus `FILE_MAPPED` at the next free value (§4.1) |
| `uch:5614` `memory_location::tier` default | `MMAP_TRACKED` | `UNREGISTERED` |
| `uc:2408` `alloc_tier_name` | "mmap_tracked" | "unregistered", "file_mapped" |
| `uc:2613` `offload_stats_note_alloc` | counts as shared | `UNREGISTERED` unreachable; `FILE_MAPPED` gets its own counter |
| `uc:14812` `unified_free_record` | returns without freeing | `UNREGISTERED`: unchanged; `FILE_MAPPED`: its own branch, before this one and before the `::free` branch (§4.1) |
| `uc:15125` `unified_alloc` refusal | refuses `MMAP_TRACKED` | refuses both (mappings come only from the file-mapping front door) |
| `uc:16825` registry host sum | pinned + mmap-tracked | pinned and file-mapped, reported separately |
| `uc:21637`, `:21654` `query_registered_location` / `query_location` | unknown → `MMAP_TRACKED` | unknown → `UNREGISTERED`; registry `FILE_MAPPED` → `FILE_MAPPED` |
| `gs:38167`, `:38202` `kv_layer_alloc::tier` default | `MMAP_TRACKED` | `UNREGISTERED` |
| `gs:81263-81269` `ggml_sycl_alloc_tier_name` | 3 cases | one case per tier |
| `gs:5731` (`expert_resolve_tier` name switch) | `HOST_MMAP` | `HOST_MMAP` now means "served from a `FILE_MAPPED` owner" |
| `tests/test-cpu-gpu-soa-interaction.cpp:178`, `:180`, `:182` | `alloc_tier::MMAP_TRACKED` x3 | `alloc_tier::UNREGISTERED` (r1 m-3; the rename breaks this build otherwise) |

The registry's `alloc_type::MMAP` (r1 m-4) is renamed **`HOST_NON_USM`**,
which names what the pointer is (registered host memory that is not USM)
rather than a GGUF mapping it never was. `alloc_type::FILE_MAPPED` is new. The
ten `MMAP` consumers, all in C2:

| consumer | meaning today | after |
|---|---|---|
| `alloc-registry.hpp:26`, `:143`, `:223` | enum, host-accessible set, commented usm mapping | `HOST_NON_USM`; `FILE_MAPPED` added to the host-accessible set |
| `common.hpp:2863` | maps to `usm::alloc::unknown` | both map to `unknown` |
| `dpct/helper.hpp:1327` | pointer-attribute classification | both |
| `gs:38730`, `:38755` | tiered-KV memcpy for the synthetic base | `HOST_NON_USM` only |
| `gs:39638` | registers the KV synthetic base | `HOST_NON_USM` |
| `gs:87745` | (classification) | both |
| `mem-ops.cpp:151` | non-USM host classification | both (a `FILE_MAPPED` source takes the bounce) |
| `set_rows.cpp:321` | non-USM host classification | both |
| `uc:14927` | malloc-overflow `::free` | `HOST_NON_USM` only; a `FILE_MAPPED` record never reaches it (§4.1) |
| `uc:21635` | `query_registered_location` | both, to their own tiers |

`cache_location::HOST_MMAP` (39 sites) keeps its name. Its derivation changes
in one place each (`uc:8068-8081`, `:8235-8248`, `:6504-6545`): it is
`HOST_MMAP` only when `query_location` says `FILE_MAPPED`. Once every weight
source is owned (C6), `register_host_expert` and `register_host_weight` refuse
an `UNREGISTERED` pointer. That also retires the direct-stage "raw mmap
fallback" as an unowned path.

---

## 5. Hazards (item 4)

### 5.1 `mlock`

- **Which lock and where.** The tier does its own locking and never uses
  `llama_mlock`. It calls `mlock` per inward-rounded lock segment (§4.6) in
  `unified_cache_raw_lock_file_segments` (`file-mapping.cpp`, an allowlisted
  site, §4.1), at `SYCL_Mapped` buffer creation, and only when the load mode
  locks and the segment's entry is host-executed.
- **The loader's locks.** `mlock_bufs` (`lm:2782-2787`) and the per-tensor
  lock in `load_all_data` (`lml:2120-2123`) skip `SYCL_Mapped` buffers.
  Otherwise `mlock_bufs` would lock the whole buffer, device sources included,
  because `is_host` is true (r1 m-6). They keep applying to `SYCL_Host`, and
  the per-tensor lock becomes the ranged `lock_range`, not the prefix
  `grow_to` (§4.6).
- **Lock time.** `mlock` without `MLOCK_ONFAULT` faults the whole segment in
  synchronously. On Qwen that is about 100 GiB from the USB members at
  0.29-1.78 GB/s (th32 §2.4), so a locked load takes one to six minutes. That
  is the price of the guarantee, and it is paid only under `-lm mmap+mlock`.
- **Charging.** Locked file pages are unevictable, so they are charged (§4.5).
  The working-set check does not apply to them.
- **`RLIMIT_MEMLOCK`** is unlimited on this host (th32 §3.1). Elsewhere the
  refusal names it.

### 5.2 Page-fault latency on CPU-executed experts

- **Storage.** bcachefs over USB SSDs, with NVMe as a promote cache: 1.78 GB/s
  sequential when served from the NVMe cache, 0.29-0.47 GB/s settled, and a
  4 KiB random read costs about 256 KiB of device reads (th32 §2.4).
- **Expert size.** Qwen3.8 has 512 experts and uses 10 per token
  (`run2.log:26-27`, `:156-157`). Each `ffn_*_exps` tensor is 850 MB for 512
  experts, so one expert matrix is about 1.7 MB and contiguous (r1 m-2). A
  cold expert therefore faults as a short sequential run under readahead, not
  as 4 KiB randoms: roughly 1-6 ms per matrix at the rates above.
- **A fully cold decode token** touches 10 experts x 3 matrices x 48 layers x
  1.7 MB ≈ 2.4 GB, which is seconds per token. So the mapped experts must be
  warm: the tier `WILLNEED`s their ranges at load (§4.6), and the working-set
  check (§4.5) says whether they can stay warm.
- **Reclaim under other tenants.** This host's permanent tenants (the qemu VM,
  Emby, Frigate) can grow. The kernel then drops clean mapped pages first, and
  decode slows to storage speed with no error. `-lm mmap+mlock` is the
  opt-out.
- **Faults happen on `CpuExpertPool` workers**, inside
  `ggml_sycl_cpu_expert_mul_mat_batched`, so a stall holds a worker, not the
  submitting thread. The existing host waits on that path (th32 §2.3) are
  unchanged.
- **`SIGBUS`.** A mapped file that is truncated while mapped, or a bcachefs
  checksum failure on fault, raises `SIGBUS` in a worker instead of an `EIO`
  return. Mapping the loader's own open file (§4.1) removes the
  replaced-by-path case, but not truncation in place. This is the same
  exposure upstream's mmap has; it is recorded, not mitigated.

### 5.3 A GPU reading mapped pages (forbidden)

Mapped pages are not USM. A device read of one is at best a slow
driver-staged copy and at worst a device loss (`uc:14044`'s own comment). The
ways a weight's host pointer can reach a device today, and what each becomes:

| site | today | after |
|---|---|---|
| `mem_handle::resolve(device)` of a `FILE_MAPPED` handle | n/a | returns empty; `device_mappable()` is false |
| `stream_dma` (`uc:13959`), the dispatch-time streaming path reached from `mmq.cpp:7639`, `mmvq.cpp:23880`, `dmmv.cpp:3817`, `getrows.cpp:2765` | a `HOST_MMAP` source is copied straight to the device (`uc:14043-14046`); a custom `copy_fn` is tested first (`:14041-14042`) | **typed, counted refusal** (r1 I-3, §F2), described below the table |
| `ggml_sycl_resolve_or_host_tensor_ptr` (`common.hpp:4922-4945`) | returns the raw pointer for any non-device host pointer | for a `FILE_MAPPED` registration, **aborts naming the tensor** (r1 m-7), described below the table |
| layout availability (`gs:62109-62114`) | `weights_evictable && is_host` → available, and AOS is available unconditionally | for a `SYCL_Mapped` buffer neither branch applies: availability for (tensor, device, layout) comes only from a cache entry (r1 m-7) |
| `direct_stage_weight`'s plain copy (`uc:6167-6175`) | `mem_copy` from a copy handle | unchanged: `mem_copy` bounces a non-USM source through a 64 MiB pinned window (`mem-ops.cpp:700-729`); this is load or materialization, not streaming |
| reorder `fill_fn`s (`gs:34835-34840`) | read `src_ptr` on the CPU | unchanged (CPU reads are allowed); see §5.4 for their staging |
| direct-stage last resort (`uc:6504-6545`) | registers an unrecognized source in place as `HOST_MMAP` | registers the `FILE_MAPPED` slice with its owner; an `UNREGISTERED` source is refused (C6) |

**The `stream_dma` refusal.**
- The check runs at entry, before the `copy_fn` test, so custom-copy callers
  are covered.
- `dma_stream_result` (`uch:3622`) gains `refused_file_mapped`, and a
  `stream_dma_file_mapped_refusals` counter increments.
- Callers route a refusal like `mmap_direct_failed` today: to the CPU executor
  (`ggml_sycl_cpu_fallback_graph`, `mmq.cpp:7653-7658`), because a host-tier
  weight executes on the CPU.
- Streaming mapped weights is forbidden by P3.

**The `resolve_or_host` abort.** Returning null would not be safe:
`binbcast.cpp:794` passes null to `needs_raw_host_staging`, which returns
false for null (`binbcast.cpp:394`), and then uses null as the kernel source.
- C4 audits all eight callers: `common.cpp:2537`, `:2538`, `binbcast.cpp:794`,
  `gs:44269`, `:51364`, `:51412`, `:64967`, `:90580`.
- A source gate pins that list.
- A caller that legitimately needs a CPU view of a weight uses
  `ggml_sycl_host_data()`.

**Gates.**
- The device-bound copy counter `file_mapped_device_copies_outside_materialization`
  counts **any** device-bound copy of `FILE_MAPPED` bytes, bounced or direct,
  outside a load transaction or a cache materialization
  (`direct_stage_weight` or a fill). The lead's load gate reads it as 0.
- `stream_dma_file_mapped_refusals` must also read 0 on a correct build: a
  refusal means dispatch routed a host-tier weight to a device kernel.
- The standing `llama.cpp-v4jk` audit (no route stages a host operand to a
  device) covers the rest of dispatch.

### 5.4 The staging bounce

- **Waits.** `mem_copy`'s bounce waits on each 64 MiB slice before refilling
  it (`mem-ops.cpp:716`), and on the last one unless the caller retains the
  handles (`:729`). That is a host wait, but only at load or materialization,
  where the preload already runs.
- **The bounce buffer** is a standalone pinned-pool allocation, role
  `EXPERT_STAGING`, released individually when its handle drops
  (`mem-ops.cpp:472-483`, allocated at `:704-705`). The transient pinned
  footprint of **`mem_copy`** is 64 MiB per copy in flight.
- **Other staging paths are unchanged by this design, and the 64 MiB claim
  does not cover them** (r1 m-16):
  - `ggml_sycl_fill_reordered_gpu` stages the whole `src_size` through
    `ggml_sycl_staging_pool` (`gs:27185-27195`, whose non-USM branch is
    commented "mmap'd").
  - The oneDNN WOQ and XMX fills make whole-source host copies
    (`gs:32185` onward, `:32432` onward).

  Their footprint is today's, bounded per call by the source tensor (850 MB
  on Qwen).
- **When it engages.** The bounce engages only if a mapped pointer is
  classified non-USM (`host_ptr_is_usm_accessible`, `mem-ops.cpp:700`). C4's
  device-labelled RED pins that, since a mis-classified mapped source would
  take the direct-copy path instead.

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
  faults, inside a cache materialization, which the §5.3 counter allows.
- **`-lm none`** stays supported on SYCL but is not the default (§F1 Q3). It
  keeps `SYCL_Host`, so it keeps the double residency. The fix there is
  different: a transient pinned window per preload slot, and a file re-read
  for re-materialization, which needs the loader's file handle to outlive the
  load. **Ticket to file after review:** "SYCL `-lm none`: device weights stay
  resident in `SYCL_Host` after preload (double residency)", depending on
  5efe.

---

## 7. Implementation plan (item 5)

Every commit is RED → GREEN. GPU work is the lead's, and each commit names
what the lead runs. Test rules (r1 I-6, §F2):

- A **host-only** test links no SYCL runtime and touches no queue, so it never
  reaches `stream_dma` or a device `mem_copy`. It carries `LABELS
  "sycl;host"`, plus `source-contract` for Python gates.
- A test that needs a device carries **`LABELS "sycl;cache;mem-handle"`** and
  `ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:1;LD_LIBRARY_PATH=..."`, and
  is **lead-run**. It is never run by an implementer or a subagent. Form 2 of
  CLAUDE.md excludes it through `-LE`.
- A RED must fail behaviourally, against a named mutant where needed, not only
  by not compiling (r1 m-12).

The lead's GPU runs share one protocol (r1 I-7):
- the selector is pinned;
- `grep -E '^(MemAvailable|Shmem|AnonPages|Cached|Mlocked):' /proc/meminfo`
  is sampled **every 1 s**;
- the watchdog floor is `MemAvailable < 25 GiB` = 26214400 kB, the same unit
  and value as the criteria;
- each run is done once;
- `journalctl -k` is checked for `GT reset|guc_id|CAT error` after it.

**C1. Restore `mmap_support`** (§3).
- RED `test-sycl-mmap-support-caps-source` (Python, `tests/`, `sycl;host;source-contract`).
  - It checks that the initializer in `ggml_backend_sycl_device_get_props`
    names `.mmap_support`, binds it to `host_unified_known &&
    !host_unified_memory`, and is neither a literal `true` nor absent.
  - Today it fails on the four-initializer list.
  - It also checks that `lm:2693-2696` is still present, since C1 must not
    remove it.
- **The lazy-range record (§3.1)** lands in C1 too, with two host-only tests
  (`host;lazy`, `tests/`, libllama only, no device):
  - `test-llama-lazy-range-record`:
    - builds `llama_tensor_weight`s from a test-written GGUF of four tensors
      in two files, two of them flagged `TENSOR_READ_LAZY`, and feeds them
      through the record's fill function, the same call `create_tensor` makes;
    - checks that the record's rows equal the flagged tensors' (file, offset,
      size) exactly, that `deferral` is `LAZY_READ` under `-lzm on` and
      `EAGER` under `off`, and that `lazy_read`'s `any()`, `has()` and
      `for_file()` and the ranges passed to `llama_mmap` agree with the record;
    - its RED is a mutant, compiled into the test binary only (libllama has
      no private-testing switch, so the test links a copy of the record and
      `lazy_read` sources built with a test-local define), in which
      `lazy_read::add` keeps its own copy and the record's fill is skipped
      for one tensor. The agreement check must fail naming that tensor.
  - `test-llama-lazy-range-record-source` (`host;lazy;source-contract`):
    - no consumer holds its own range container: `lazy_read` has no `ranges`
      or `tensors` member, and `init_mappings` passes the record, not
      `lazy.for_file`;
    - `create_tensor`'s `TENSOR_READ_LAZY` branch calls the fill;
    - today it fails on `llama-model-loader.h:118-119`.
- Lead witness (r1 I-7):
  - Run Qwen on the B70 with **no** `-lm`/`-lzm` flags.
  - Stop the process with `kill` as soon as the log shows both
    `load_mode = mmap` and `per_layer_token_embd ... lazy read enabled`. In
    run2 both appeared by t = 3.9 s (`run2.log:314`, `:370`), about 53 s
    before the pinned pre-size began (`:2893`).
  - This run is a witness, not a load; it never reaches the pre-size.
- Lead gates: the Mistral gate on `:1` and `:0`. Tokens are unchanged, since
  Mistral has no lazy tensor.

**C2. The `FILE_MAPPED` tier, the raw wrappers and the census** (§4.1, §4.8).
- RED `test-sycl-file-mapping-host` (C++, links `file-mapping.cpp` only,
  `sycl;host`). It checks:
  - reservation shape: one `PROT_NONE` reservation, `MAP_FIXED` segments at
    the right offsets, and the bytes read back equal to the file's;
  - `dup` fd ownership: the fd is closed at release;
  - the size-mismatch refusal;
  - the private `munmap` counter: exactly one `munmap`, of the whole
    reservation, at the last release.

  Mutants the test must fail: release skipped (counter 0); per-segment
  `munmap` (counter > 1).
- RED `test-sycl-file-mapped-tier` (links ggml-sycl, `sycl;cache;mem-handle`,
  `level_zero:1`, lead-run). It checks:
  - `query_location` returns `FILE_MAPPED`;
  - `device_mappable()` is false, and `resolve(device)` is empty;
  - the registry has one record for the reservation;
  - the `::free`-branch counter stays 0 at release.

  Mutant: the `FILE_MAPPED` branch placed after the `alloc_type` `::free`
  branch.
- RED `test-sycl-raw-mmap-allowlist-source` (`sycl;host;source-contract`),
  §4.1. Mutant: an `mmap(` call added to a non-listed file.
- RED `test-sycl-alloc-tier-census-source` (`sycl;host;source-contract`). No
  `MMAP_TRACKED` and no `alloc_type::MMAP` remain anywhere in the tree,
  `tests/` included. Today it fails on the 12 + 3 + 10 sites of §4.8.
- Also: the canonical contract §3 allowlist gains the three wrappers, and the
  static-storage census and the allowlist gates are re-run.

**C3. The segment plan and the residency gate** (§4.6).
- RED `test-sycl-file-mapping-segments` (`sycl;host`), a pure-function test.
  - Map segments round outward, and `WILLNEED` and lock segments round inward.
  - At most one lazy page is mapped per edge.
  - Cases: an interleaved lazy range, a lazy range sharing a page with a
    tensor at each end, back-to-back lazy ranges, and a lazy range at offset
    0 before a lock segment.
  - Mutants: inward map rounding (fails the shared-page read), outward
    `WILLNEED` rounding (fails the zero-lazy bound).
- RED `test-sycl-file-mapping-residency` (`sycl;host`), §4.6's six arms and
  their control twins.
  - The fixture lives under `${CMAKE_BINARY_DIR}`, which is not tmpfs.
  - The test exits 77 if the eviction precondition fails, arm 2 exits 77 if
    `RLIMIT_MEMLOCK` is below the fixture size, and arms 5 and 6 exit 77 if
    `read_ahead_kb` is 0.
  - Arm 3 and the arm 5 and 6 control twins are the must-fail cases.
  - Arms 5 and 6 need the both-edge window: a mutant that advises only the
    segment tail fails arm 6.
- The loader's ranged lock (§4.6) also lands here, in libllama, with RED
  `test-llama-mlock-ranged` (`host;lazy`). It is §4.6's loader lock gate: a
  lazy range before a locked range, the hole interior reading 0, and the
  whole-prefix `grow_to` mutant reading resident.
- Lead: none.

**C4. One published fact, the planner split, `SYCL_Mapped` and the loader**
(§4.3, §4.4, §5.3). This lands as **one commit**, because the fact and its only
consumers must land together (r1 C-1, I-5, §F2). It is the commit that changes
every SYCL load.
- **Prefetch 0 on SYCL is an explicit C4 item** (addendum I-9, m-6).
  `init_mappings` is called with prefetch 0 on a SYCL load (`lm:2702`), and
  `llama_mmap` excludes the record's deferred ranges with the both-edge
  window whatever the prefetch value.
- Host-only REDs:
  - `test-llama-mmap-lazy-exclusion` (`host;lazy`), §4.6's `llama_mmap`
    prefetch gate. Under `prefetch = -1` there is no `WILLNEED` over a lazy
    range and no read-around into one. The mutant is an empty record (lazy
    ranges from a diverted `lazy_read`), which must read resident.
  - `test-sycl-mapped-loader-source` (`sycl;host;source-contract`):
    - prefetch 0 at the SYCL `init_mappings` call;
    - every site of the §4.3 table;
    - the caps loops before `llama_model_sycl_compute_early_plan`;
    - no `use_mmap` read in `create_tensor`'s SYCL buft choice;
    - `lm:2693` gone;
    - the eight `weight_host_bytes` accumulation sites test `host_backing`;
    - the eight `resolve_or_host` callers are the pinned list (§5.3).
  - `test-sycl-host-backing-plan`, a Qwen-shaped inventory (1223 tensors,
    B70 budget):
    - with `load_maps_files` true, host experts are `FILE_MAPPED`,
      `weight_host_bytes` = 0, `host_mapped_bytes` = 102383.5 MB, and the
      WEIGHT zone is 64 MiB plus the runtime term;
    - with it false, today's 122924.2 MB zone;
    - with mlock, the mapped bytes are charged.

    It is host-only through 7v5c's dry-run entry if 7v5c has landed.
    Otherwise it is registered `sycl;cache;mem-handle`, `level_zero:1`, and is
    lead-run.
- Device REDs (`sycl;cache;mem-handle`, `level_zero:1`, lead-run):
  - `test-sycl-file-mapped-stream-refusal`: `stream_dma` over a `FILE_MAPPED`
    view refuses, with and without a `copy_fn`, and the counter counts.
    Mutant: the check placed after the `copy_fn` test.
  - `test-sycl-file-mapped-bounce`: `mem_copy` of a mapped source takes the
    bounce (`host_ptr_is_usm_accessible` false), and the outside-materialization
    counter stays 0 inside a materialization and counts outside one.
  - `test-sycl-file-mapped-expert-owner`: `register_host_expert` over a mapped
    slice stores the owner, and the mapping outlives the buffer while an entry
    holds it.
  - `test-sycl-file-mapped-resolve-abort`, a death test in its own process:
    `resolve_or_host` on a mapped weight aborts naming the tensor. It is
    scored on rc 134 plus the tensor name.
- Lead, in this order (r1 I-8, m-14, m-15):
  1. `POSIX_FADV_DONTNEED` over shard 3, then `fincore` on it. Pre-registered:
     ≤ 1 MiB resident.
  2. **Qwen3.8 Q8_0 on `level_zero:0`, default load flags, `-c 4096`, the
     baseline completion** (`-p 'The capital of France is' -n 32 --seed 42
     --temp 0`). Pre-registered pass criteria:
     - the load completes and prints coherent text;
     - `[HOST-ARENA] configured zones: WEIGHT=` is 64.0-128.0 MB, and the
       pre-size commits ≤ 12288 MB (§4.4);
     - `SYCL_Mapped` model buffer lines sum to 127615 MiB ± 1%;
     - `SYCL_Host` model buffer lines hold 0 weight bytes;
     - peak `AnonPages` minus the pre-load reading is < 30 GiB;
     - `MemAvailable` never falls under 25 GiB (the watchdog unit);
     - `Shmem` stays under 5 GiB;
     - `fincore` on shard 3 after the run is ≤ 256 MiB. The bound: the prompt
       plus 32 tokens is under 40 tokens, x 16 PLE rows per token (th32 §0 table),
       x a 256 KiB bcachefs extent per row fault = 160 MiB, rounded up;
     - `stream_dma_file_mapped_refusals` = 0, and
       `file_mapped_device_copies_outside_materialization` = 0;
     - no GPU fault lines.

     There is deliberately no page-cache-growth criterion, because it would
     depend on the other shards' prior cache state.
  3. The Mistral gate on `:1` and `:0`. The GPT-OSS gate on `:1` is **the
     canonical CLAUDE.md `llama-cli` command, including `-c 4096`, under
     `timeout 60`**, scored with `grep -cx '1, 2, 3, 4, 5'` = 1. Every SYCL
     load now takes the mapped path.
  4. `llama-bench` Mistral Q4_0 and GPT-OSS 20B on both cards, interleaved
     against the parent commit, gated on `docs/backend/sycl-perf-baselines.md`.
     Load time is also reported, since device weights now read from page cache
     through the bounce.

**C5. The monotone budget, the headroom and the working-set WARN** (§4.5).
- RED `test-sycl-host-budget` (`sycl;host`). The pure function lives in a
  device-free header, `host-budget.hpp`.
  - It checks every row of the §4.5 table: 121.8, 125.8, 0.5 and a refusal,
    34.5, 12.0 and a refusal.
  - It sweeps `MemAvailable` from 0 to `MemTotal` in 64 MiB steps for 32, 62
    and 246.9 GiB, and asserts the budget never decreases.
  - Today's formula fails both: the (246.9, 28.9) row gives 20.9, and the
    sweep drops at the `available/2` cliff.
- RED `test-sycl-host-working-set-warn` (`sycl;host`). The WARN is printed
  with its three figures at `resident_room < mapped_ws`, not above it, and
  the plan-report line carries the same figures.
- It depends on th32 T2 for `host_ledger_budget()` if T2 has landed.
  Otherwise it edits `uc:4200-4231`, and T2 lifts the result.
- Lead: the C4 Qwen run once more, reading `[PLACEMENT] Totals` (the four
  terms) and the working-set line against the `/proc/meminfo` samples.

**C6. Retire unowned host registration** (§4.8).
- RED `test-sycl-unregistered-host-refusal` (`sycl;cache;mem-handle`,
  `level_zero:1`, lead-run). `register_host_expert` and `register_host_weight`
  with a `malloc`ed (`UNREGISTERED`) pointer are refused. Today they succeed as
  `HOST_MMAP`, `non_owning_external_host`.
- Lead: the Mistral and GPT-OSS gates (GPT-OSS under `timeout 60`), and the C4
  Qwen run. The refusal must never fire on a real load.

Tickets to file after review: `-lm none` double residency (§6), and
`CPU_REPACK` outside the cache (§4.4).

---

## 8. Answers to the open questions (lead ruling §F1)

- **Q1. A mapped working-set overflow is a WARN, not a refusal.** It costs
  refaults, not an OOM. The WARN prints the arithmetic, and the plan report
  carries the same figures (§4.5). Refusal stays for pinned and anonymous
  bytes only, through th32 T2 over 7v5c.
- **Q2. 5efe defines the `FILE_MAPPED` tier and its terms; yl3v owns the
  cold-set policy** and may use `FILE_MAPPED` as the backing for cold host
  experts. §4.5 states the interface, including the terms yl3v must read so
  that DeepSeek's cold set is not implicitly empty. 5efe adds no cold-set
  logic.
- **Q3. `-lm none` stays supported on SYCL, and is not the default.** Its
  double residency gets its own ticket after review (§6).

---

## Appendix A. Disposition of design-5efe-r1 (69e70c715..de6d0d614 with addendum, 1C/10I/20m)

| finding | disposition | where |
|---|---|---|
| C-1 C4 acceptance ordered before the ledger split | Fixed per §F2: the split and the WEIGHT-zone sizing that reads it move into C4, one commit with the loader and no C4-only exclusion. Qwen's WEIGHT zone is pre-registered at 64.0-128.0 MB, the pre-size at ≤ 12288 MB, with a falsifier. The non-weight `SYCL_Host` runtime term is named. | §4.4, §7 C4 |
| I-1 mincore gate cannot measure the property | Fixed per §F2: `cachestat(2)` on the fd, or `mincore` on a test-owned mapping; never the tier's hole. Synchronous `pread` positive control, must-fail mutant arm, 24 MiB hole with a 1 MiB guard band, and an mlock arm. | §4.6, §7 C3 |
| I-2 multi-segment shape under-specified | Fixed per §F2: one `PROT_NONE` reservation with `MAP_FIXED` segments, one owner, one record, one `munmap`. Map segments round outward; `WILLNEED` and lock segments round inward. Bound: at most one lazy page mapped per edge. | §4.1, §4.3, §4.6 |
| I-3 stream_dma disposition legitimized streaming | Fixed per §F2: a typed, counted refusal checked before `copy_fn`, routed to the CPU executor. The counter counts any device-bound copy of mapped bytes outside a materialization. The RED asserts the refusal. | §5.3, §7 C4 |
| I-4 headroom cap universal and discontinuous | Fixed per §F2: `max(total/32, min(29 GiB, total/4))`, one named P3 constant cited as this host's measurement, the `available/2` cliff removed in favour of a named refusal, and 32 and 62 GiB cases plus a monotonicity sweep. | §4.5, §7 C5 |
| I-5 plan and loader derive "mapped" separately, and in the wrong order | Fixed per §F2: the caps are resolved before the early plan, and one `load_maps_files` fact is published. The planner's `host_backing` is read by the loader through a hook; `create_tensor` never reads `use_mmap` for SYCL weights. | §2.1, §4.3, §4.4 |
| I-6 host-only REDs need a device; no labels | Fixed per §F2: every test is named with its registration and label. Device tests are `sycl;cache;mem-handle`, `level_zero:1`, lead-run. Host tests link a device-free TU and never touch `stream_dma` or a device `mem_copy`. | §4.1, §7 |
| I-7 C1 witness re-runs the near-OOM | Fixed per §F2: the witness is killed on the two log lines (t ≈ 3.9 s, before the pre-size). Sampling every 1 s; floor and criteria both 25 GiB = 26214400 kB. | §7 |
| I-8 two C4 criteria fail on a correct build | Fixed per §F2: the page-cache growth criterion is dropped. Shard 3 gets a pre-registered bound (≤ 1 MiB before, ≤ 256 MiB after). `SYCL_Mapped` is expected at 127615 MiB ± 1%, all SYCL host-sourced weights. | §4.3, §7 C4 |
| m-1 "pinned 110 GB" | Fixed: 139264 MB committed, 140.0 GB reached, `MemAvailable` fell 141 GiB. | §0 |
| m-2 expert figures | Fixed: 512 experts, 10 used, about 1.7 MB per matrix, about 2.4 GB per cold token, 1-6 ms per matrix. | §5.2 |
| m-3 census misses tests | Fixed: `tests/test-cpu-gpu-soa-interaction.cpp:178/180/182` added, and the census gate covers `tests/`. | §4.8, §7 C2 |
| m-4 `alloc_type::MMAP` understated | Fixed: ten consumers listed, the `::free` meaning recorded, renamed `HOST_NON_USM`, and a `FILE_MAPPED` branch ordered before `::free` with a counter test. | §2.7, §4.1, §4.8 |
| m-5 fd lifetime and replacement gap | Fixed: `dup` of the loader's fd, kept by the owner for `FADV_DONTNEED`; no reopen by path. | §4.1, §4.6 |
| m-6 loader sites under-enumerated | Fixed: the §4.3 site table (hooks, `weight_buft_supported`, `make_cpu_buft_list`, `mlock_bufs`, the `init_mappings` prefetch) and the loader `lmlock` skip. | §4.2, §4.3, §5.1 |
| m-7 null path unverified; AOS branch | Fixed: `resolve_or_host` aborts naming the tensor, with the eight callers audited and pinned, and `binbcast.cpp:394/794` cited. Both layout-availability branches are excluded for `SYCL_Mapped`. | §5.3 |
| m-8 stale th32 pin | Fixed per §F2: pinned at the latest reviewed head, now rev 2r `df6e83b3b` (r17 nits only); 2s `f364952d3` is noted as the revision that reads the record. Rulings relabelled §T13/§T13a/§T13c. | header, §4.6 |
| m-9 F1 fold and cold set | Fixed per §F2: the yl3v/7v5c interface is stated, the recommendation removed, and the terms yl3v must read are named so that DeepSeek's cold set is not implicitly empty. | §4.5, §8 |
| m-10 enum value collision | Fixed per §F2: the next free value at landing; censuses key on names. | §4.1, §4.8 |
| m-11 allowlist unenforced | Fixed: `test-sycl-raw-mmap-allowlist-source` with today's eight sites, shared with th32 T1. | §4.1, §7 C2 |
| m-12 C2 RED weak | Fixed: named mutants plus a private `munmap` counter and a registry query; no `mincore`-ENOMEM check. | §7 C2 |
| m-13 staging faults can populate lazy pages | Fixed: `MADV_RANDOM` on segment edges bordering lazy ranges covers execution and bounce faults; the gate reads every byte of A and B, edges included. | §4.6 |
| m-14 GPT-OSS gate lacks `timeout 60` | Fixed per §F2: the canonical command with `-c 4096` under `timeout 60`, scored with `grep -cx`. | §7 C4, C6 |
| m-15 shard-3 "~0" not pre-registered | Fixed: ≤ 1 MiB before, ≤ 256 MiB after, with the arithmetic. | §7 C4 |
| m-16 64 MiB claim too broad | Fixed: scoped to `mem_copy`; the reorder, WOQ and XMX fill paths are listed as unchanged and bounded per source tensor. | §5.4 |
| m-17 host_unified fails open | Fixed: `host_unified_known` added, and `mmap_support` requires it. | §3 |
| m-18 storage_owner type; PER_DEVICE teardown | Fixed: `storage_owner` wraps the reservation's `mem_handle`; every model unloads on every device before any cache tears down. | §4.2 |
| addendum I-2(a) §4.6 text wrong for map segments | Fixed per §F2a: already corrected at `1ddbd9d56` (map segments outward); rev 3 states it at the table, adds the `MADV_RANDOM` edge windows as an inward set, and says that only map segments round outward. | §4.6 |
| addendum I-9 lazy_ranges source disappears after th32 T6 | Fixed per §F2a and §M49: one per-file lazy-range record, filled at `create_tensor` from the flag, owned by 5efe, landing in C1. `lazy_read`, th32's inventory, the tier and `llama_mmap` consume it and keep no copy. Prefetch 0 on SYCL is an explicit C4 item, with RED `test-llama-mmap-lazy-exclusion` (its mutant is an empty record). | §3.1, §4.6, §7 C1, C4 |
| addendum I-10 tier and loader inherit the prefix lock | Fixed per §F2a: the tier `mlock(2)`s inward-rounded host-executed segments directly; `SYCL_Mapped` is excluded from `mlock_bufs`; §2.5 is corrected; `lml:2122` becomes a ranged `lock_range` in 5efe (C3). Host-only RED `test-llama-mlock-ranged` with a lazy range before a locked range and a whole-prefix mutant; the tier's arm 2 covers the tier side. | §2.5, §4.6, §5.1, §7 C3 |
| addendum m-19 one-sided edge advice; residual undercounted | Fixed per §F2a: `MADV_RANDOM` windows on both edges, because read-around is centred on the fault. The loader's whole-file `llama_mmap` read-around is named as a second residual path, given the same record-fed window (C4), and bounded at one shared page per edge per mapping, plus bcachefs extent granularity inside the guard band. | §4.6 |
| addendum m-20 readahead case could pass vacuously | Fixed per §F2a: arms 5 (forward) and 6 (backward) fault real mapped pages, exit 77 when `read_ahead_kb` is 0, each has a no-advice control twin that must populate the hole, and both are measured with `cachestat` or a test-owned mapping. | §4.6, §7 C3 |
| §M49 note: record in C1 | Done: the record and its two host-only tests are C1 deliverables, so th32 T6 depends on C1 alone. | §3.1, §7 C1 |
| th32 r15 m-4 (relayed) mlock prefix | Fixed: the tier's own per-segment `mlock` (named, and where), no `llama_mlock` for `SYCL_Mapped`, a residency arm under mlock, and a RED with a lazy range before a locked tensor. | §4.6, §5.1, §7 C3 |
