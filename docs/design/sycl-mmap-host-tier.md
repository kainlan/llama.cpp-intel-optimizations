# SYCL file-mapped host tier (mmap as a unified-cache tier)

*Status: DESIGN rev 3d, not implemented. rev 3d is a new commit on rev 3c
(`fef860772`), which is not amended (addendum 10 (a)). It folds
design-5efe-r3 (0C/5I/7m/1n, and its contradictions X1-X7), addendum 10 (b)
and (c), §M56 (a), and th32 rev 2x's two C1 requirements; Appendix B's r3
rows say where each landed. rev 3c added §F2a I-9 addenda 4 to 6 and rulings
addenda 7 and 8 to rev 3b (`3b447ee70`). It folds:
- design-5efe-r1 and its addendum, under lead rulings §F1, §F2 and §F2a with
  §F2a I-9 addenda 2 to 6, and rulings addenda 7 and 8;
- design-5efe-r2 (0C/6I/19m), with the lead's rulings on its I-2 and I-5;
- the th32 r15 m-4 mlock item, and the §M49 note that the lazy-range record
  lands in C1;
- the owner's default-context ruling: every model gate set carries a no-`-c`
  arm, scored, with the vehicle clarification (a CLI with no `-c`, or a
  programmatic `n_ctx = 0`). Tracker: `llama.cpp-5efe`.
Written 2026-09-27 against branch `task/5efe` at `69e70c715` (= master). Every
cite without a revision prefix is to that tree. `gs` is
`ggml/src/ggml-sycl/ggml-sycl.cpp`, `uc` is
`ggml/src/ggml-sycl/unified-cache.cpp`, `uch` is `unified-cache.hpp`, `lml`
is `src/llama-model-loader.cpp` and `lm` is `src/llama-model.cpp`. Load-phase
evidence is the lead's Qwen3.8 baseline `qwen-base/run2.log` and `run2.mem`
(session scratchpad, `fefb92980`, B70 alone, `-lm mmap -lzm on -c 4096 -v`),
whose load-path code is byte-identical to this tree in every region cited here.
Appendix A disposes of every r1 finding, and Appendix B of every r2 finding
and later ruling.*

Sibling designs this one interfaces with, and does not restate:

- `llama.cpp-th32`: STORAGE tier for lazy row tables,
  `docs/design/sycl-storage-tier-lazy-tables.md`, abbreviated **th32**.
  Pinned at **rev 2x `90be8e260`** (it was 2w `3afc3b020` at rev 3c and 2u
  `0bd69e251` at rev 3b; r2 m-15, r3 X7). th32's later commits `90cb98cab`
  and `bde97cd15` only re-pin 5efe and move a tracker note. Its §4.1 consumes
  this design's pre-fill (§3.1), the `lazy_eligible` bit, the taker and
  `deferral`, and its T6a owns the eligibility predicate that §3.1 calls. Its
  §9.1 names this design's declaration and attribute function. Its T6 row
  owns the context allocation scope that §4.4 reads for host-buffer owners.
  Section numbers cited here are 2x's.
- zhcn's design, cited by row id: its **GDC3** row is the single source of
  the GPT-OSS B50 default-context demotion figures (§M56 (a)). This design
  carries no copy of them.
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

- the WEIGHT zone is 64.0 MB, and it never grows after the pre-size;
- the pre-size commits 5 chunks, 10240.0 MB (§4.4);
- `MemAvailable` never falls under 25 GiB;
- `SYCL_Mapped` holds about 127615 MiB;
- shard 3, the PLE table, stays under 256 MiB resident;
- a no-`-c` arm at the default context passes too, with the KV overflow
  re-placed to the host tier (§7 C4).

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
requires lazy tables to be excluded from `max_tensor`. The early plan cannot
see the table without the record, so C1's pre-fill is what makes that
exclusion reach it (§3.1), and th32's T3 fixture should cover the early plan
too.

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
- `memory_tier::MMAP` (`tensor-types.hpp:36-40`, "File-backed mmap (slowest,
  fallback)") is the fourth name (r2 m-11). Its six uses (`gs:14317`,
  `:19256`, `:19279`, `:34766`, `:35665`, `:98734`) are the initial value and
  the "not found" result of `get_cached_tensor_ptr`, which always returns
  `nullptr` (§2.3). So it means "unresolved", not a mapping.

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
- **th32 drops its own flip** (lead ruling §T13, point 1). The pre-fallback
  mode capture that th32's T6 used to own is now commit 1's (§F2a I-9
  addendum 4). Commit 1 captures it, with the taker, at the §3.1 pre-fill,
  before its moved fallback block.

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
and held by `llama_model_loader`.

- **Membership is by declaration** (§F2a I-9 addenda 3 and 5). There is one row
  per declared name found in `weights_map`, whatever any consumer decides about
  it. A row holds (tensor name, file index, offset, size, `lazy_eligible`,
  `deferral`, and the attributes `rows_per_token` and `consumer_layer`).
- **Two per-model facts are captured once, at the pre-fill** (addenda 4 and 5):
  - **`mode_pre`**, the lazy mode as requested (`ml.lazy.mode`), captured
    **before** C1's moved `AUTO`-to-`OFF` fallback block (`lm:2256-2265`
    today). This capture is a 5efe C1 deliverable, no longer a th32 T6 one.
  - **`taker`** = `sycl_model_loading_guard.active` (`lm:2228-2230`). It is
    true when the model's weights are SYCL-owned (`n_gpu_layers() > 0` with a
    SYCL model device, or a SYCL buffer-type override, `lm:2214-2227`) and the
    hooks resolve. It is the fact the early plan is already gated on
    (`lm:2229`). It is **not** `llama_model_sycl_hooks_enabled()`, which is
    process-level: a `-ngl 0` load in a SYCL build registers the hooks
    without taking, and must not get `STORAGE`. In a build with neither
    `GGML_USE_SYCL` nor `GGML_BACKEND_DL` (`lm:2213`), `taker` is false.
  - **`mode_post`** is `ml.lazy.mode` after the fallback block. It is not
    captured separately: it is what `lazy_read` holds.
- **`lazy_eligible`** is one bit, computed once at the pre-fill by **th32's
  T6a predicate**, the DLM (a) rule (th32 rev 2w `3afc3b020` §4.1 and its T6a
  row), called and not restated:

  ```
  lazy_eligible = T6a(taker ? mode_pre : mode_post,
                      size,
                      taker ? true : llama_mmap::SUPPORTED)
  ```

  - `OFF` is never eligible, `AUTO` only above 4 GiB, and `ON` always, each
    only when the third input is true.
  - **Why the mode differs by path** (addendum 4). The taker reads through
    its own `O_DIRECT` fd and never through a mapping, so the fallback, which
    exists because the CPU lazy path needs a mapping, must not disengage it,
    and a no-mmap platform does not either (addendum 6 (1)). Off the taker,
    `lazy_read` really does read with `mode_post`, so feeding it `mode_pre`
    would mark a table that is read eagerly as lazy, and the prefetch
    exclusion would then leave a table cold that is about to be read in full.
- **`deferral`** carries only the consumer kind (addendum 5):
  - `STORAGE` iff `lazy_eligible && taker`;
  - `LAZY_READ` iff `lazy_eligible && !taker`;
  - `EAGER` otherwise.

  So **`deferral != EAGER` iff `lazy_eligible`**: one fact, one source. rev
  3b's (`3b447ee70`) eligible-but-`EAGER` case is gone, because off the taker the bit
  already reads `mode_post` and `SUPPORTED`. `STORAGE` is written here at the
  pre-fill, like every other field, since rows are immutable afterwards; th32
  T6 never writes it. T6 takes a table iff `deferral == STORAGE`, in place of
  its `hooks_enabled` gate.
- Consumers read `deferral` or the bit. None re-evaluates the mode, the floor
  or `SUPPORTED`.
- **Where it is filled: before the early plan, not at `create_tensor`**
  (§F2a I-9 addendum 2).
  - **Why `create_tensor` is too late.** The SYCL early plan runs before
    `create_tensor` (`lm:2228-2231`) and builds its inventory from
    `ml.weights_map` (`lm:639-641`), so it cannot see a row filled there. It
    would place and stage the lazy table as an ordinary weight: run2's early
    plan counted the PLE in `max_tensor` (§2.6) and in the host total
    (154263.6 MB, `run2.log:308`).
  - **When the flag is knowable.** `TENSOR_READ_LAZY` is first knowable only
    at `create_tensor`. It is an argument at each arch's call site
    (`src/models/qwen4exp.cpp:189`, `src/models/gemma4.cpp:58`), and no table
    records it (`llm_tensor_info` is `{layer, op}`, `llama-arch.h:782-785`).
  - **The declaration** (r2 I-5; lead ruling: 5efe's, since the record is
    5efe's; th32 consumes it; keyed by arch per addendum 5; names only per
    addendum 7 (a), whose naming addendum 8 (a) amends back to this):

    ```cpp
    // src/llama-model.h, C1
    // The GGUF names, or name patterns with one "%d" standing for the block
    // index, that `arch` creates with TENSOR_READ_LAZY. Static and immutable:
    // names only, no hparams-valued field.
    const std::vector<std::string> & llama_model_lazy_tensor_names(llm_arch arch);

    struct llama_lazy_tensor_attrs {
        uint32_t rows_per_token;  // 0 = not known for this arch
        int32_t  consumer_layer;  // -1 = not known for this arch
    };
    // One per-arch function, next to the declaration, reading the loaded hparams.
    llama_lazy_tensor_attrs llama_model_lazy_tensor_attrs(llm_arch arch, const llama_hparams & hparams,
                                                          const std::string & name);
    ```

    - **Keyed by (arch, GGUF name or pattern)** (addendum 5, amending r2 I-5's
      "keyed by GGUF name"). The arch key is load-bearing: `gemma3n` creates
      the same `per_layer_token_embd.weight` **without** the flag
      (`src/models/gemma3n.cpp:39`). With a name-only key, every gemma3n load
      would get a row, and the drift check below would abort it.
    - **The declaration holds names only** (addendum 7 (a); names per
      addendum 8 (a), and th32 keeps them). rev 3b put `rows_per_token` and
      `consumer_layer` in a struct for T6 to fill in. A static table that T6
      writes is mutable global state shared by every loaded model, so two
      loaded models with different hparams would overwrite each other's
      values (P4).
    - The name list takes no hparams. Presence in the GGUF already bounds it,
      so the call sites' hparams conditions (`qwen4exp.cpp:171`
      `ple_n_heads > 0`, `gemma4.cpp:57` `n_embd_per_layer > 0`) are not
      stated a third time.
    - **The attributes live on the per-model record entry.** C1's pre-fill
      calls `llama_model_lazy_tensor_attrs` once per matched name, with the
      loaded hparams, and stores the result on that model's record entry.
      th32 T6 reads it from there through `record.attrs(name)`, and never
      from a static.
      - The function is 5efe's, and C1 lands it with th32 §4.1's values:
        `qwen4exp` returns {`hparams.ple_n_heads`, 1} (`qwen4exp.cpp:95`), and
        `gemma4` returns {1, 0}.
      - The **missing-attribute refusal** is T6's (addendum 5). It applies
        only to `deferral == STORAGE` rows, and runs before the early plan
        reads the row. A non-taker row is never refused for missing
        attributes.
    - Today's entries are `per_layer_token_embd.weight` for `qwen4exp` and for
      `gemma4`. deepseek41 is **not** declared: that arch does not exist in
      the tree, and llama.cpp-9g22 (paused) adds its own entry when it lands
      (addendum 4).
    - **Not an `LLM_TENSOR_INFOS` column.** That table is keyed by tensor kind
      across every arch (`llama-arch.cpp:721`). A kind-level column would
      declare gemma3n's table lazy, for the same gemma3n reason.
  - **The pre-fill.** In `load_tensors` (`lm:2212-2232`), C1 reorders the
    code to:
    1. `resolved_sycl_weight_owner` and the guard, unchanged (`lm:2214-2228`);
    2. capture `mode_pre` and `taker`;
    3. the caps loop (`lm:2245-2254`) and the lazy-mode fallback
       (`lm:2256-2265`), moved up. They read `devices`, `params.load_mode`,
       `ml.use_mmap` and `ml.lazy.mode`, and nothing the early plan writes
       (r2 verified);
    4. print **one per-model INFO line** (th32 rev 2x's requirement, its
       §4.1 and T6 row (k)), `lazy-range record: mode_pre=<mode>
       mode_post=<mode> taker=<0|1>`, with `mode_post` read after the
       moved fallback. It prints for every model, whether or not any name is
       declared, so a harness can read both modes and the taker without a
       lazy table. It is INFO, like the row lines below. A test that installs
       a callback through `llama_log_set` receives every level, since
       `llama_log_internal_v` hands each line to the callback unfiltered
       (`src/llama-impl.cpp:40-55`); only `common_log`'s default callback
       drops INFO, at `verbosity <= common_log_verbosity_thold`
       (`common/log.cpp:542-546`, with INFO mapped to TRACE at `:532`). A
       CLI reads it with `-v` or `-lv 4`;
    5. fill one row per declared name found in `weights_map`: (name, file
       index, offset, size) from the `llama_tensor_weight`, the bit, the
       deferral, and the attributes. Each row prints one INFO line,
       `lazy-range record: <name> (<size> MiB) eligible=<0|1>
       deferral=<EAGER|LAZY_READ|STORAGE> taker=<0|1> mode=<the mode the bit
       read>`. It replaces `add()`'s "lazy read enabled" line as the witness
       §7 C1's lead runs read;
    6. `llama_model_sycl_compute_early_plan`, if `taker` (`lm:2229-2231`).

    It runs the same way under `no_alloc` (r2 m-9), so the fit pass plans what
    the real load plans. See `lazy_read` below for why that maps nothing in
    the fit pass.
  - **One source, checked twice.**
    - **Statically:** `test-llama-lazy-declaration-source` (C1) maps every
      `TENSOR_READ_LAZY` `create_tensor` call in `src/models/*.cpp`, with its
      arch, to a declaration entry, and every entry back to a call site. An
      upstream merge that adds a flagged call, or a new upstream arch that
      arrives with one, fails it at build time, on every backend's tree,
      before any load.
    - **At runtime, mode-independent** (addendum 4; r2 I-5(3)).
      `create_tensor` aborts, naming the tensor and the arch, on a
      `TENSOR_READ_LAZY` call to a tensor with **no** row, and on a non-lazy
      call to a tensor that **has** one, whatever the row's eligibility,
      `deferral` and the mode. So drift is caught under `-lzm off`, and under
      `AUTO` at or below 4 GiB, too. rev 3a checked only non-`EAGER` rows,
      which let exactly those cases through.
  - **The early plan reads it.** Its loop over `weights_map` (`lm:639-641`)
    skips every row whose `deferral` is `STORAGE`, the only kind it can see,
    since the early plan runs only when `taker` is true. That keeps an
    eligible table out of placement, out of `max_tensor` and the staging
    maxima, and out of the host total, which is th32 §4.1's requirement
    applied to the early plan too. An ineligible row (`OFF`, or `AUTO` at or
    below 4 GiB) is placed as the ordinary weight it is (th32 r19 M-1; the
    witness is in §7 C1). The late inventory already excludes deferred tables
    (§2.5).
- **Its API.**
  - `deferred_for_file(idx)` returns the sorted, merged ranges of every
    non-`EAGER` row of that file.
  - `has(name)`, `eligible(name)`, `deferral(name)` and `attrs(name)` are the
    per-tensor queries.
  - `value(name)` returns a `llama_lazy_range_value {file_idx, offs, size}` **by
    value**. th32's backend `lazy_table` takes its (file, offset, size) from
    this once, at plan publish, through the inventory. It is a hand-off of the
    record's value, not a second source (§F2a I-9 addendum 2, M-1). What is
    forbidden is deriving those numbers independently, for example from
    `w->offs`.
- **Immutability.** Rows, the bit, `deferral` and the attributes are set
  once, by the pre-fill, and a later write aborts. So a by-value copy taken at
  plan publish stays equal to the record for the model's lifetime, and is
  safe for tickets that outlive the model.

**Consumers keep no copy of the ranges.**
- **`lazy_read`** keeps its `mode` and a **membership** set of names, as th32
  §4.1 requires (its r18 N-4).
  - **`add(name, t, w)` reads `deferral` and decides nothing** (addendum 5;
    addendum 6 (2): a C1 deliverable). **Its return value is
    `record.deferral(name) == LAZY_READ`, and nothing else** (r3 m1). Its
    own mode test, the 4 GiB floor and the `llama_mmap::SUPPORTED` test
    (`lml:1241-1256`) are deleted.
    - **Membership is a separate axis from the decision.** `add()` inserts
      the name into the membership set only when `w != nullptr`, that is,
      only when the real load will map the file. Under `no_alloc`,
      `create_tensor` passes `nullptr` (`lml:1776`), so `add()` still returns
      true for a `LAZY_READ` row and `is_lazy` is set, while membership stays
      empty and nothing is mapped. Upstream's comment at `lml:1775` requires
      exactly that: the decision "must not depend on the load mode, or the
      memory-fit pass (no_alloc, no mmap)" would size a lazy table as
      resident. rev 3c's `w != nullptr` term in the decision made `is_lazy`
      false in the fit pass, so the fit pass sized the 51880 MiB table that
      the real load leaves unread.
    - For a `STORAGE` row `add()` returns false, since `lazy_read` is not its
      consumer. Whether `create_tensor` treats that tensor as lazy is th32
      T6's SYCL branch, which keys on `deferral == STORAGE` (th32 2x T6 row).
    - That is a fork change to upstream code, so
      `test-llama-lazy-range-record-source` pins it (§7 C1). A rebase that
      restores `add()`'s own gates fails loudly.
    - Landing it in C1, where `deferral` is filled, leaves no window in which
      `add()` and the bit disagree.
    - `add()`'s "mmap is not available, so tensor %s ... is loaded into RAM in
      full" WARN (`lml:1252-1253`) moves to the pre-fill. It prints for a
      declared row whose bit is clear because `SUPPORTED` is false off the
      taker.
  - `any()` and `has()` answer from that membership. `for_file(idx)` returns
    the record's ranges for its members only.
  - Its private `ranges` map is deleted.
  - So under `taker`, where eligible rows are `STORAGE`, `any()` stays false,
    and `init_mappings` does not map every shard with `use_mmap` cleared
    (`lml:1871`, the hazard r2 I-6 names).
  - Under `no_alloc`, membership stays empty and the fit pass maps nothing,
    while a `LAZY_READ` table still reads as lazy to the fit pass (r2 m-9,
    r3 m1; the membership bullet above).
- `init_mappings` passes `record.deferred_for_file(idx)` to `llama_mmap`
  (`lml:1890-1891`).
- th32's inventory reads each table's (file, offset, size) from the record
  through `value(name)` (th32 §4.1).
- The tier hook's `lazy_ranges` argument (§4.3) is
  `record.deferred_for_file(idx)`.

**What it changes on its own.** The ranges and decisions are the ones
`lazy_read` computes today, on today's `AUTO`-to-`OFF` fallback. The one
observable change is that the early plan stops counting deferred tables. On
Qwen with lazy enabled:
- the early `[SYCL-PLAN] Host staging zone` reads `per_inflight` about
  1702 MB, like the late plan, instead of 103762.3 MB (`run2.log:243`);
- the early `[PLACEMENT] Totals` host figure reads 102383.5 MB instead of
  154263.6 MB (`run2.log:308`).

The arithmetic is exact. The early inventory's largest tensor is
`per_layer_token_embd.weight` at 51880.1 MB (`run2.log:235`), and
154263.6 - 51880.1 = 102383.5, the late host total (`:2821`). The early
`per_inflight` is 2 x 51880.1 + 2.1; the late one is 2 x 850.0 + 2.0, with
the 850.0 MB `ffn_*_exps` tensors next in line (`:235`, `:2756`).

**Data edges.** th32 T6 depends on C1 alone (§M49): C1 lands the record, the
two captures, the declaration, the attribute function and `add()`'s reading
of `deferral`. C1 depends on th32's T6a, whose predicate (mode, size,
`SUPPORTED`) the pre-fill calls, so T6a lands no later than C1 (§F2a I-9
addenda 3 and 6).

**The C1-to-T6 window (addendum 8 (b)).** C1 writes `STORAGE` for an eligible
row when `taker` is true, and th32 T6 is the only consumer of `STORAGE`. On a
tree holding C1 without T6, a taker load of Qwen under default flags would
therefore:
- exclude the table from the early plan, since its row is `STORAGE`;
- keep it out of `lazy_read`, since `add()` accepts only `LAZY_READ`;
- and then load it eagerly as an ordinary 51880 MiB pinned weight that the
  early plan's staging maxima never counted.

The lead's ruling:
- **C1 and T6 reach master in one merge.** T6 is built on top of C1
  (`task/th32` carries or merges `task/5efe`'s C1), and the merge to master
  happens once, after both are reviewed. The tracker blocks the merge of 5efe
  C1 on th32 T6.
- The interim-`EAGER` alternative is refused: it breaks addendum 4's iff and
  creates a second source for `deferral`.
- No lead-run gate of C1 reads the table (W1-W3 are killed first, and the
  Mistral gate has no lazy tensor), so C1 needs no Qwen load of its own.
- C4's Qwen arms are VOID without T6 and T7 (§7 C4), and that precondition
  was accepted.

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
  any `mmap`, `munmap`, `mlock` or `madvise` call in `ggml/src/ggml-sycl/`
  outside a listed site.
  - **It tokenizes first** (r2 m-12). It strips comments, string and
    character literals, then matches `\b(mmap|munmap|mlock|madvise)\s*\(`.
    Comments such as `uc:21652` "host/mmap (conservative)" and
    `tensor-types.hpp:40` "mmap (slowest" would otherwise be flagged by the
    `\s*` form, and a literal `mmap(` would miss a real `mmap (` call.
  - A fixture of those comment and string forms, plus one real `mmap (` call,
    is part of its test.
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
- That branch sits before the `MMAP_TRACKED`/`UNREGISTERED` no-free case
  (`uc:14812`), the `::free` branch (`uc:14922-14929`, `HOST_MALLOC` after
  §4.8), the `zone_managed` branch and the `sycl::free` fallthrough
  (`uc:14930-14940`). So a `FILE_MAPPED` record can never reach `::free` or
  `sycl::free` (r1 m-4).
- The `::free` branch keys on the **registry type**, which a `FILE_MAPPED`
  record never carries, so moving the `FILE_MAPPED` branch after it would
  change nothing (r2 m-6). C2's test therefore uses mutants that change
  behaviour:
  - the record registered as `HOST_MALLOC`, which reaches `::free`;
  - the branch moved after the `zone_managed` branch, into the `sycl::free`
    fallthrough.

  Private counters on the `::free` and `sycl::free` branches must both read 0
  at a correct release.

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
- C1 already moves the two caps loops (`lm:2245-2265`) above the early plan
  (`lm:2228-2231`), because the record's pre-fill needs the resolved lazy mode
  (§3.1). They depend only on `devices` and `params`. C4 adds the publication
  below.
- The resolved answer is published once, on the SYCL load transaction:
  `load_maps_files = ml.use_mmap after resolution && the file-mapping tier
  exists in this build`. It is false under `-lm none`, and false when an iGPU
  in the device set forced `none`.
- **Builds without the tier** (r2 m-19). `file-mapping.cpp` needs
  `PROT_NONE` reservations and `MAP_FIXED`, so it is compiled for POSIX only.
  On a non-POSIX build (Windows SYCL) the hook reports the tier absent, and
  `load_maps_files` is false. Those loads keep `SYCL_Host`, and C1's cap never
  routes them to a missing hook.
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
- `alloc_buffer` (r2 m-10). The `no_alloc` path calls
  `ggml_backend_buft_alloc_buffer(buft, 0)` for every buft (`lm:2770-2772`),
  so size 0 returns a dummy buffer with no storage. Any other size aborts,
  naming the caller, because a `SYCL_Mapped` buffer is made only by the hook
  below.
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
| `lm:2693-2696` | narrowed: it clears `use_mmap` for a SYCL load only when `load_maps_files` is false while `use_mmap` is true, which happens only on a build without the tier (m-19). There the loader keeps today's behaviour instead of copying from a mapping into `SYCL_Host`. C4 amends C1's caps source test, which asserts the line is present (r2 m-8) |
| preload's host-planned copy, `gs:34696-34730` (r2 m-7) | today it copies a weight that the late plan tiers to host, after `create_tensor` put it in a SYCL buffer (the `dev_layer` sync case, `lm:2680-2689`), into a pinned WEIGHT allocation (role `WEIGHT`, category `OTHER`, `memcpy` from `tensor->data`). For a `SYCL_Mapped` source it is served in place when the CPU executes the file layout (host backing `FILE_MAPPED`, registered as a mapped slice). When a reorder is needed, the copy is planned: host backing `PINNED`, or `HOST_CPU` after th32 T1, charged in `weight_host_bytes`. It never copies mapped bytes into memory the plan does not charge |
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

**Who reads `weight_host_bytes`** (r2 m-3). The C4 contract lists the writers
above, and these readers too. Each is assigned one meaning, "charged" or
"placed on host":

| reader | meaning after C4 | why |
|---|---|---|
| `planned_host_bytes` / `actual_host_bytes` on the plan snapshot (`uc:171`, `:228-229`) | **placed**: `weight_host_bytes + host_cpu_owned_bytes + host_mapped_bytes` | it feeds the verdict below |
| `ggml_sycl_current_model_planner_host_placement()` (`gs:2702`), which drives `ggml_backend_sycl_is_tiered_enabled` (`gs:19168`) and the "planner host placement: yes" log (`gs:16737`, `run2.log:2826`) | **placed** | otherwise Qwen would report "no" while 100 GB of host experts execute on the CPU |
| the lifecycle LIVE publication (`gs:13197`, `model-lifecycle.cpp:668`) | both: `ModelState` gains `charged_host_bytes` beside the placed figure | multi-model accounting needs the charge; the tiering verdict needs placement |
| the load summary (`gs:1135-1230`, `:12498`) | prints both, labelled | one number would be read as the other |

**Host-side non-weight requests: routed by role** (r2 I-2; lead ruling: 5efe
owns role-based routing of **host** non-weight requests).

*Today.*
- `select_zone` sends every host request with category `HOST_COMPUTE` or
  `EXPERT_CACHE` to the WEIGHT zone, whatever its role (`uc:15368-15370`).
- `SYCL_Host` buffers allocate with role `WEIGHT` whenever
  `weights_evictable` holds, even after the load (`gs:42645-42646`).
- Those tenants fitted, unseen, in the ~20 GB of 1.2x headroom over 100 GB of
  pinned experts. After the split the WEIGHT zone is 64 MiB. So they would
  grow zone 0 one ≥ 2048 MB chunk per miss (`uc:15418-15437`), which is what
  zone 0 did in run2 (`run2.log:2965`, `:2968`).
- The known tenants are:
  - llama-context's CPU-backend compute buffer, which uses the first
    device's host buft, i.e. `SYCL_Host` (`src/llama-context.cpp:889-895`);
  - the output buffer (`src/llama-context.cpp:3562-3564`);
  - `CpuExpertPool`'s process-lifetime ring, "role=COMPUTE +
    category=EXPERT_CACHE instead routes to the WEIGHT host zone"
    (`cpu-expert-pool.cpp:66-77`), and `PinnedBufferPool`'s ring
    (`pinned-buffer-pool.cpp:69`);
  - the other `HOST_COMPUTE` sites whose role is `STAGING`, `TP_TMP`,
    `COMPUTE`, `GRAPH_TMP` or `OTHER`.

*The sizing fact the reviewer relied on does not hold.*
`ggml_backend_sycl_notify_compute_buffer_sizes` (`gs:19103-19146`), which
would grow SCRATCH by the scheduler's compute-buffer sizes (`:19145`), **has no
caller in the tree**: `git grep notify_compute_buffer_sizes` finds only its
declaration (`ggml-sycl.h:629`) and definition. So the host compute buffer is
not "sized in SCRATCH, allocated in WEIGHT". It is sized nowhere, and
allocated in WEIGHT. The comment at `uc:27716` ("fed back via
`unified_cache_grow_host_scratch_zone()` after the ggml scheduler is
created") describes a path that does not run.

*C4's change.*
- **Route by role, in `select_zone`, the one place.**
  - Role `KV`, or `kv_spill_to_host`, goes to KV, as today.
  - Role `WEIGHT` goes to WEIGHT.
  - Role `EXPERT_STAGING` goes to SCRATCH, as today.
  - A request whose lifetime is the interim `UNOWNED` goes to WEIGHT, where
    it lands today. Only the 12 lines that addendum 10 (b) leaves to
    llama.cpp-30r1 carry it (the site table below).
  - Every other role whose category is `HOST_COMPUTE` or `EXPERT_CACHE` goes
    to **SCRATCH**, per the lead's ruling.
  - Anything else goes to STAGING, as today.
  - Category no longer selects WEIGHT. No call site changes its role,
    except the three host buffer types below; the `DEVICE` and `UNOWNED`
    sites change only their lifetime.
- **`SYCL_Host` takes role `WEIGHT` only inside a model load**
  (`in_model_load`). The `|| weights_evictable` term goes. A post-load
  `SYCL_Host` buffer (compute, output, inputs) takes role `COMPUTE`, and so
  SCRATCH.
- **`SYCL_KV_Host` gets its own `alloc_buffer`, with role `KV`** (r3 B; lead
  ruling on r3 B).
  - Today `ggml_backend_sycl_kv_host_buffer_type()` copies the `SYCL_Host`
    buft and overrides only `get_name` (`gs:42750-42758`). So uize-demoted
    host KV goes through `SYCL_Host`'s `alloc_buffer` (`gs:42627-42707`),
    with role `WEIGHT` or `STAGING` (`gs:42645-42646`), cohort
    `backend_host_buffer`, and lands in WEIGHT. Nothing sets
    `kv_spill_to_host` on that path; only the device-KV fallback does
    (`uc:15285-15303`). After the routing above alone, it would take role
    `COMPUTE`, zone SCRATCH and lifetime `GRAPH`: every `graph_compute`
    would trip the SCRATCH boundary WARN (`gs:92459`, `uc:20290-20298`), 6
    GiB would arrive in SCRATCH unannounced, and the KV zone would receive
    nothing.
  - C4 factors the body of `SYCL_Host`'s `alloc_buffer` into one helper that
    takes the intent, and gives `SYCL_KV_Host` its own `alloc_buffer` over
    that helper, with role `KV`, category `HOST_COMPUTE`, cohort
    `kv_host_buffer` and lifetime `CONTEXT`. `select_zone` sends role `KV` to
    the KV zone (zone 1) by its first rule. The buft keeps its own
    `get_name` (the residency decline keys on it,
    `ggml_sycl_tensor_is_in_kv_host_buft`), and every other iface slot stays
    shared.
  - **The qwen4exp indexer cache takes the same path.** `llama_memory_hybrid_idx`
    builds a second `llama_kv_cache`, prefix `idx_`, K only (the MLA trick,
    one head of `indexer_head_size`), over the layers `filter_idx` admits
    (`src/llama-memory-hybrid-idx.cpp:45-67`). qwen4exp's `filter_idx` is
    `!is_recr(il)` (`lm:3580-3585`), the same 12 layers that carry
    attention KV. The per-layer buft choice is the same hook call
    (`src/llama-kv-cache.cpp:290-306`), so a demoted layer's indexer keys go
    to `SYCL_KV_Host` too. At Qwen's default context that is 12 x 262144 x
    128 x 2 B = 768 MiB, which the printed host-KV figure must include (next
    item).
  - The recurrent state is **not** host KV: `llama_memory_recurrent`
    allocates it in the plain device buft (`src/llama-memory-recurrent.cpp:97-103`,
    `llama_recurrent_sycl_kv_buft`). rev 3c's "plus a small recurrent state"
    beside the host-KV figure is withdrawn.
- **The demotion counts the indexer** (lead ruling on r3 B: the indexer is
  in the printed figure).
  - Today the demotion's per-layer bytes come from
    `placement_plan::kv_size_for_layer(l)` (`uch:1033`), at both the demotion
    input (`gs:17089-17092`) and the residency re-fit (`gs:18037-18042`).
    Its per-layer truth comes from `llama_model_sycl_build_kv_layer_arrays`
    (`lm:350-365`) through the inventory's `kv_k_width_per_layer`,
    `kv_v_width_per_layer` and `kv_layer_kind` (`ggml-sycl.h:323-326`). No
    field carries the indexer, so the printed `X MB host KV` omits 768 MiB
    that the load then pins.
  - C4 adds **`uint32_t kv_idx_width`** to `ggml_sycl_tensor_inventory`,
    filled from `hparams.indexer_head_size` when the model's memory is
    `llama_memory_hybrid_idx` and 0 otherwise. Measured with a standalone
    compile against this tree's header: the struct is 176 bytes, 8-aligned,
    with `kv_layer_count` at offset 168, so a `uint32_t` after it lands at
    offset 172, in the tail padding, and `sizeof` stays 176. A stale
    libllama leaves the byte zero, which is today's behaviour (no indexer
    term), and never shifts another field (memory:
    extending-a-struct-that-crosses-the-backend-dlopen-boundary). Every
    in-tree construction is value-initialized (`lm:653`, `:705`,
    `tests/test-tiered-dispatch.cpp:35`,
    `tests/test-sycl-lifecycle-runtime-wrapper.cpp:1388`, `:1426`).
  - One accessor, `placement_plan::kv_demand_for_layer(l)` =
    `kv_size_for_layer(l)` + `kv_idx_width` x the layer's cells x the K
    element size, for every layer the plan counts as carrying KV. Both
    demotion sites read it, and so does `rec.host_bytes`, the printed
    figure. `kv_size_for_layer` itself, which sizes the device per-layer
    slabs, is unchanged: on the device the `idx_` tensors are not remapped
    into a layer slab (`[KV-REMAP]` matches only `cache_k_l`/`cache_v_l`,
    `gs:38570-38578`). How the device accounts for them is outside 5efe.
  - So the Qwen no-`-c` line prints 12 x (512 + 64) = **6912.0 MB** of host
    KV (§7 C4 step 3).
- **Every host buffer type's `get_max_size` asks `select_zone`** (found
  while folding r3 B; P4, one fact with one source).
  - ggml-alloc splits a host buffer into chunks of at most `get_max_size`,
    and `SYCL_Host`'s `get_max_size` answers from a zone it names itself:
    WEIGHT, hard-coded (`gs:42477`), and the `SYCL_HostCompute` buft's
    answers from STAGING (`gs:42977`). The glkg fix made the first one
    agree with `select_zone` as it stands today, and
    `tests/test-sycl-host-zone-config-source-contract.py` pins that
    agreement.
  - After C4's routing, both are wrong: a post-load `SYCL_Host` buffer goes
    to SCRATCH, a `SYCL_KV_Host` buffer to KV, and a `SYCL_HostCompute`
    buffer (role `COMPUTE`, `gs:42832`) to SCRATCH. Chunking against the
    wrong zone's largest free block is the llama.cpp-glkg defect again.
  - C4 gives each of the three bufts one intent builder, used by its
    `alloc_buffer` and by its `get_max_size`, which calls the extracted
    `select_zone` on that intent. The source contract is amended from
    "names WEIGHT" to "calls `select_zone` on the `alloc_buffer`'s intent".
    Mutant: the hard-coded WEIGHT restored.
  - The chunk cap (2 GiB, `GGML_SYCL_HOST_BUFFER_CHUNK_MB`, `gs:42421-42438`)
    is unchanged. So Qwen's 6144 MiB of demoted main KV reaches the KV zone
    as three 2048 MiB buffers (24 tensors of 256 MiB, 8 per buffer), and the
    768 MiB indexer cache as one buffer.
- **The notify path, before allocation** (r3 A; addendum 10 (c) accepts the
  entry point).
  - **The existing entry point stays uncalled.**
    `ggml_backend_sycl_notify_compute_buffer_sizes` (`gs:19103-19146`) has
    no caller. It reserves the **VRAM** compute arena and aborts if the
    reserve is short (`unified_cache_reserve_compute_arena`, `:19140-19142`),
    and then grows host SCRATCH by the sum over **every** backend's buffer,
    device buffers included (`:19145`). The lead sends its VRAM half to
    23mk's dead-code census (addendum 10 (c)).
  - **C4 adds a host-only entry point:**

    ```c
    // ggml/include/ggml-sycl.h, C4. Records one llama_context's host buffer
    // demand and grows the host pinned SCRATCH zone to cover it.
    GGML_BACKEND_API void ggml_backend_sycl_notify_host_buffer_sizes(ggml_backend_t backend,
                                                                     size_t         host_compute_bytes,
                                                                     size_t         host_output_bytes);
    ```

    - `ggml_backend_sycl_reg_get_proc_address` (`gs:109834`) exports it by
      name. A direct call does not link under `GGML_BACKEND_DL`.
    - **`backend` is checked, not cast blindly** (r3 A.4). The function
      returns with a WARN unless `ggml_backend_is_sycl(backend)`
      (`gs:107651`). `llama_context` passes exactly one backend: the SYCL
      backend of `model.devices[0]`, the device whose host buft its CPU
      backend uses (`src/llama-context.cpp:889-895`).
  - **The ABI shape is deliberate.** It takes scalars, so no struct crosses
    the dlopen boundary and no stride can drift between a fresh `libllama`
    and a stale `libggml-sycl` (memory:
    extending-a-struct-that-crosses-the-backend-dlopen-boundary).
    - It is a new name, not a new signature under the old name. So a stale
      backend returns `NULL` from the proc lookup, and never runs with
      misread arguments.
    - On `NULL`, `llama_context` WARNs `SYCL backend has no
      ggml_backend_sycl_notify_host_buffer_sizes; host SCRATCH is not
      grown`, and continues. The zone criteria below are then the witness.
    - The commit message states that both libraries rebuild together.
  - **When it is called: before each allocation it covers** (r3 A.2). An
    allocation made first fits or grows zone 3 through the miss path
    (`uc:15418-15437`); a notify after it would grow SCRATCH again for the
    same bytes, an idle reservation (P4).
    - `llama_context` keeps its two current terms, `host_compute_bytes` and
      `host_output_bytes`, and every call passes both.
    - **Output buffer.** `output_reserve` computes `new_size`
      (`src/llama-context.cpp:3535-3537`) and frees the old buffer (`:3550`)
      before it allocates (`:3566`). When the chosen buft is the SYCL host
      buft (`:3559-3565`), it notifies with the new
      `host_output_bytes` between those two steps. That covers the first
      reserve in the constructor (`:850`) and every regrowth (`:2896`,
      `:3231`, `:4974`). The term is `new_size`, the bytes the allocation
      asks for.
    - **Compute buffers.** In `sched_reserve`, after the scheduler is created
      (`:2019`) and before the first allocating `graph_reserve` (`:2055`),
      C4 runs the pp graph and the tg graph through `graph_reserve(...,
      /*split_only=*/true, sizes)`, which calls
      `ggml_backend_sched_reserve_size` (`:3908-3910`) and allocates nothing.
      `host_compute_bytes` is the larger of the two sizes at the index of
      the CPU backend, whose buft is the SYCL host buft (`:889-895`). It
      notifies, then runs the allocating reserves as today. The
      pipeline-parallel retry makes a new scheduler (`:2059-2064`) and
      repeats the measure and the notify. The `no_alloc` fit pass already
      takes the size-only path (`:2055-2056`) and does not notify.
    - **The witness.** After the allocating reserves, `llama_context` compares
      `ggml_backend_sched_get_buffer_size` on the CPU backend with the
      notified figure and WARNs `[HOST-ARENA] host compute buffer %.1f MB
      exceeds notified %.1f MB` if it is larger. Plan != reality is then
      printed, not inferred.
  - **What the SYCL side does with it** (r3 A.3, A.5).
    - One demand per `llama_context`, keyed by the notifying backend's
      `ggml_backend_sycl_context`. A new call **replaces** that context's
      demand; it does not add to it. The entry is dropped in that backend
      context's destructor (`gs:43872`, reached through
      `ggml_backend_sycl_free`, `gs:83938`), its owner boundary.
    - The zone is global: `unified_cache_grow_host_scratch_zone` resolves
      device 0 (`uc:21935`), and host zones are configured once, at load
      (`gs:5623`). So SCRATCH's target is the plan's `host_zone_scratch_bytes`
      plus the **sum of the live demands**, and it grows by
      `max(0, sum - grown_for_demands)`, where `grown_for_demands` is the
      most the demands have already grown it. Capacity is never returned, so
      a later context reuses what an earlier one grew. One call per context,
      and one `llama_context` never grows the zone once per SYCL backend.
    - The growth is one `grow_zone(SCRATCH, deficit + n x
      DEFAULT_ALIGNMENT)` call, where `n` is the number of buffers the demand
      splits into at the chunk cap, the same slack rule as the miss path
      (`uc:15432`). `grow_zone` commits **one chunk of
      `max(2048 MB, align_up(request))`** (`pinned-pool.cpp:613-712`). So
      the zone grows by at least the deficit, **rounded up to one chunk**,
      and the criteria score it that way.
    - It prints `[HOST-ARENA] SCRATCH demand ctx=%llu compute=%.1f MB
      output=%.1f MB total=%.1f MB grown=%.1f MB` on every call, and the
      existing `[HOST-ARENA] SCRATCH zone grown by %.1f MB for compute
      buffers` (`uc:20447-20448`) when it grows.
  - The "runtime term at the runtime KV update" of rev 3a stays withdrawn,
    since it had no path into the zone.
- **Lifetimes** (addendum 7 (b), addendum 10 (b)).
  - SCRATCH's graph-boundary check runs at every `graph_compute`
    (`gs:92459`), and WARNs on any live SCRATCH record ("refusing SCRATCH
    host_zone_boundary_check with N live registered allocations",
    `uc:20290-20298`). That is exactly why the `CpuExpertPool` ring was
    routed to WEIGHT (llama.cpp-cg8j).
  - C4 adds `alloc_intent::lifetime` with four values, plus the interim
    marker `UNOWNED` described under the site table, which llama.cpp-30r1
    deletes. There is **no `PROCESS`**: a process-lifetime allocation has no
    releasing owner (P2).
    - **`GRAPH`**, the default. The graph-boundary check counts only these
      records.
    - **`CONTEXT`**. The record carries its owner, and the owner's
      boundary is where it is checked:
      - a record held by a backend context (`readback_staging_handle`,
        `moe_ids_cache`) carries that `ggml_backend_sycl_context`'s id and
        is checked in its destructor (`gs:43872`, run from
        `ggml_backend_sycl_free`, `gs:83938`, at `llama_context` free);
      - a **host buffer** allocated through a buft (`SYCL_Host` after the
        load, `SYCL_KV_Host`, `SYCL_HostCompute`) carries the context serial
        of the `llama_context` that allocated it. The buft call has no
        context argument, so the serial comes from th32 T6's **context
        allocation scope**, `thread_local` and RAII-nested, which stamps every
        host compute buffer the context allocates with (context serial,
        allocation serial) (th32 2x T6 row, its §5.3 and *§D12h* I-3). T6
        opens it in `graph_reserve()` and at the scheduler's
        `alloc_graph` sites; **C4 opens the same scope around
        `output_reserve`'s allocation and around the memory module's
        construction** (`src/llama-context.cpp:870`), so the output buffer
        and the host KV carry the serial too. A post-load host-buffer
        allocation with no open scope WARNs `[HOST-ZONE] host buffer
        outside a context scope: <cohort> <bytes>` and aborts under
        `GGML_SYCL_STRICT_LEASES=1`. These records are checked at the
        device cache's teardown, not at the context's: the buffers are
        members of `llama_context` whose destruction order against its
        backends is not a boundary the backend can see. The WARN names the
        context serial.
    - **`MODEL`**. The record carries the model's `ModelId`
      (`model-lifecycle.hpp`), whose release is the model-free point.
    - **`DEVICE`** (addendum 10 (b)). The owner is that device's
      `unified_cache` instance, and the check runs at that cache's teardown.
      It is not the refused `PROCESS`: it has a named owner and a release
      point, and it is allowed only for a holder whose contents are
      device-scoped and shared by every context on the device. The module
      teardown releases every `DEVICE` holder first, through
      `ggml_sycl_shutdown_global_runtime_pinned_owners` (`gs:7575-7590`,
      asserted at `gs:84006`), which shuts down `g_cpu_expert_pools[d]`,
      `g_attn_host_pools[d]` and `g_pinned_buffer_pools[d]` and frees the
      attention host staging (`ggml_sycl_attn_host_cpu_backend_free`,
      `gs:82177`, which calls `ggml_sycl_attn_host_staging_free_all`,
      `gs:82097`), all while the cache is still alive.
  - **Each lifetime keeps a leak check at its owner's boundary.** A record
    still live at its owner's boundary prints `[HOST-ZONE] leaked
    <CONTEXT|MODEL|DEVICE> record: <cohort> <bytes> owner=<id>`. It is a
    WARN that aborts under `GGML_SYCL_STRICT_LEASES=1`, through the existing
    accessor (`strict_lease_checks_enabled`, `uc:12962-12968`).
  - Every non-`GRAPH` row of the census cites its owner.
- **Sizing: every SCRATCH tenant has a term** (r3 m4). The rerouted
  tenants used to fit only inside the VRAM-RUNTIME terms folded into host
  SCRATCH, which were never sized for them.
  - **Context host buffers** (compute, output): the notified demand above.
  - **Host KV** (`SYCL_KV_Host`): not SCRATCH at all. The KV zone grows at
    context init through the miss path (`uc:15418-15437`), one chunk per
    host-KV buffer that does not fit, and the demotion line prints the bytes
    first. That is outside the PP/TG phase gate (`pinned-pool.cpp:619-640`).
  - **`DEVICE` holders** register a demand keyed by (device, holder)
    through an internal call, `unified_cache_host_scratch_demand(owner,
    bytes)`, before they allocate, on the same ledger as the context
    demands. The two pools size their buffers at first `MUL_MAT_ID`
    initialization (`gs:9024-9037`) from `max_dispatch_count` (top-K times
    tokens, over the graph's `MUL_MAT_ID` nodes, `gs:8024-8030`), `max_K`
    and `max_N` (`gs:8018-8023`):
    - `PinnedBufferPool`: D x (K + N) x 4 B (`pinned-buffer-pool.cpp:38-39`);
    - `CpuExpertPool`: 4 ring slots x D x (K + N) x 4 B
      (`cpu-expert-pool.cpp:56-59`, `RING_SLOTS = 4`, `cpu-expert-pool.hpp:89`).

    The attention host staging registers the sum of its slot capacities
    before each grow, since it grows with `n_kv` (`gs:81970-82060`). So the
    demand is exact, and its value is printed, not pre-registered.
  - **`GRAPH` tenants** (the ten `GRAPH` rows of the site table): each is
    bounded by a named plan term that the census gate records per row:
    `max_tensor_bytes` + `k_scratch_headroom` for the single-tensor staging
    copies, `moe_cpu_expert_staging_bytes` + `moe_host_recipe_workspace_bytes`
    for the MoE staging, and `tp_vram_runtime_bytes` for `TP_TMP`. A `GRAPH`
    row the gate cannot map to a term is listed **`UNSIZED`** and ratcheted
    like `UNOWNED`.
  - **The VRAM-RUNTIME read that stays in host SCRATCH** (addendum 7 (c))
    then sizes no host tenant: the PP pipeline scratch and the PP MoE oneDNN
    ring are VRAM allocations (the ring's slots come from the VRAM RUNTIME
    and KV zones, `uc:18197-18199`). With the terms above it is an idle
    host reservation (P4). Removing it would amend addendum 7 (c), so it is
    **brought to the lead**, not changed here.
- **The site table** (addendum 7 (c)). 5efe owns each host site's role, zone,
  lifetime and sizing, and publishes them here. 23mk owns each site's
  conversion, and its class-H rows (56, 118 and the others) cite this table.
  C4's source gate (`test-sycl-host-role-routing-source`) regenerates the
  table from the tree and fails on any difference.
  - A first lexical cut finds 72 host-capable lines with category
    `HOST_COMPUTE` or `EXPERT_CACHE`, across 8 files. By role:
    - 23 `EXPERT_STAGING`: already SCRATCH, `GRAPH`, unchanged;
    - 14 role `WEIGHT`: 10 are device-tier cache entries that never reach
      `select_zone`'s host branch, and 4 stay WEIGHT. One of the four is a
      real weight, the direct-stage last resort (`uc:6522`). The other three
      are non-weight bytes carrying role `WEIGHT`: `cpu-dispatch.cpp:315`
      (the CPU-offload host weight copy), `gs:39954` (a device buffer's
      host-pinned fallback on low free VRAM) and `gs:63226` (the
      tensor-split weight cache). These are the "role-WEIGHT, unplanned"
      rows. None is reached on Qwen's default path, and if one is, the
      zone-0 criterion catches it. **`gs:39954` is also a P3 violation**: a
      device buffer's bytes in host-pinned memory are read by the GPU in
      place (zero-copy). That is llama.cpp-nl0f's, beside 23mk's row 56
      (r3 m6);
    - 35 that change zone, or choose it at the site, tabulated below, plus
      the new `SYCL_KV_Host` `alloc_buffer`.
  - Two of the 35 are not tenants: `pinned-pool.cpp:93` is the pool's own
    chunk backing (`require_host_usm_base`, never zone-selected), and
    `uc:16728` is `category_to_intent`, a mapping, not a request.
    `uc:15652` (`acquire_offload_buffer`) takes the lifetime its caller
    passes, so the gate follows its callers.

  | site | holder of the handle | role | zone after C4 | lifetime | owner |
  |---|---|---|---|---|---|
  | `common.cpp:551` | local `host_handle`, released in-function | STAGING | SCRATCH | GRAPH | - |
  | `common.cpp:1414` | local `host_handle` | TP_TMP | SCRATCH | GRAPH | - |
  | `gs:8549`, `gs:8711` | local `d2h_staging` in `moe_hybrid_init_once` | STAGING | SCRATCH | GRAPH | - |
  | `gs:36782` | local `fallback_handle` (`buffer_get_tensor`) | STAGING | SCRATCH | GRAPH | - |
  | `gs:36901` | local `host_stage_handle` | STAGING | SCRATCH | GRAPH | - |
  | `gs:44289` | graph-local `host_copies`, cleared at exit | GRAPH_TMP | SCRATCH | GRAPH | - |
  | `gs:48239` | `g_tp_*` job slot, consumed and cleared by the TP sync (`gs:48396`) | TP_TMP | SCRATCH | GRAPH | - |
  | `gs:48977` | `g_tp_async_ffn_jobs[layer]` slot, consumed within the graph | TP_TMP | SCRATCH | GRAPH | - |
  | `gs:82678` | local `vh`, waited in-function (`gs:82683`) | COMPUTE | SCRATCH | GRAPH | - |
  | `gs:42647` | a post-load `SYCL_Host` buffer (compute, output, inputs) | COMPUTE (WEIGHT only in `in_model_load`) | SCRATCH | CONTEXT | the allocating `llama_context`, by the context allocation scope's serial |
  | new, `SYCL_KV_Host`'s `alloc_buffer` (C4) | a demoted layer's host KV, and its `idx_` keys | KV | KV | CONTEXT | the allocating `llama_context`, by the context allocation scope's serial |
  | `gs:42832` | the `SYCL_HostCompute` buffer | COMPUTE | SCRATCH | CONTEXT | the allocating `llama_context`, by the context allocation scope's serial |
  | `gs:44097` | `ggml_backend_sycl_context::readback_staging_handle` | STAGING | SCRATCH | CONTEXT | its backend context (`gs:43872`) |
  | `gs:57647` | `ctx.moe_ids_cache[key]`, or a call-local entry | STAGING | SCRATCH | CONTEXT, or GRAPH for the call-local entry | its backend context |
  | `uc:15652` | the caller's `offload_buffer_lease` | caller's | SCRATCH | caller's | caller's |
  | `cpu-expert-pool.cpp:77` | `g_cpu_expert_pools[dev]` (`gs:2605`), the ring | COMPUTE | SCRATCH | **DEVICE** | that device's `unified_cache` (addendum 10 (b)) |
  | `pinned-buffer-pool.cpp:69` | `g_pinned_buffer_pools[dev]` (`gs:2604`) | COMPUTE | SCRATCH | **DEVICE** | that device's `unified_cache` (addendum 10 (b)) |
  | `gs:81996`, `:82016`, `:82027`, `:82045` | `g_attn_host_staging[dev]` (`gs:81948`) | COMPUTE | SCRATCH | **DEVICE** | that device's `unified_cache`; device-shared, see below |
  | `common.cpp:443` | `g_runtime_staging_cache` (file scope) | STAGING | WEIGHT until llama.cpp-30r1 | **UNOWNED**, target CONTEXT | `ggml_backend_sycl_context`, after llama.cpp-30r1 |
  | `common.cpp:1970` | `g_tp_host_staging` | TP_TMP | WEIGHT until llama.cpp-30r1 | **UNOWNED**, target CONTEXT | same |
  | `common.cpp:2083` | `g_tp_quant_comm_bufs` | TP_TMP | WEIGHT until llama.cpp-30r1 | **UNOWNED**, target CONTEXT | same |
  | `common.cpp:2279` | `g_tp_shared_reduce_handle` | TP_TMP | WEIGHT until llama.cpp-30r1 | **UNOWNED**, target CONTEXT | same |
  | `common.cpp:2328` | `g_tp_host_buf0/1_handle` | TP_TMP | WEIGHT until llama.cpp-30r1 | **UNOWNED**, target CONTEXT | same |
  | `cpu-dispatch.cpp:2160` | `g_retained_scratch_lease` | COMPUTE | WEIGHT until llama.cpp-30r1 | **UNOWNED**, target CONTEXT | same |
  | `gs:10882` | `g_routing_indices_cache` | STAGING | WEIGHT until llama.cpp-30r1 | **UNOWNED**, target CONTEXT | same |
  | `gs:21522` | `g_moe_fusions[layer].fused_handle` (`thread_local`, `gs:21664`) | COMPUTE | WEIGHT until llama.cpp-30r1 | **UNOWNED**, target CONTEXT | same |
  | `gs:62927`, `gs:62942` | `g_split_staging[dev]` (`gs:62915`) | STAGING | WEIGHT until llama.cpp-30r1 | **UNOWNED**, target CONTEXT | same |
  | `gs:63145` | `g_split_weight_staging` (`gs:63132`) | STAGING | WEIGHT until llama.cpp-30r1 | **UNOWNED**, target CONTEXT | same |
  | `gs:63548` | `s_second_out_ring_alloc` / `s_cpu_out_ring_alloc` (`gs:62645`) | STAGING | WEIGHT until llama.cpp-30r1 | **UNOWNED**, target CONTEXT | same |

  - **Addendum 10 (b) rules the 14 holders (18 lines) that rev 3c brought to
    the lead.** Each gets an owner; the refactor that moves them is
    **llama.cpp-30r1**, not 5efe's.
    - **`DEVICE`, landed in C4:** the `CpuExpertPool` ring and
      `PinnedBufferPool`, per the ruling, and the attention host staging,
      under the ruling's own clause ("if one of them turns out to be
      device-shared, it becomes DEVICE, with the reason stated"). The
      reason: `g_attn_host_staging[dev][kind]` is indexed by device, its
      slots keep a grow-only capacity that every context's host attention
      on that device reuses (`gs:81970-82060`), and it is released only by
      the module teardown above. All three already have their owner (the
      device's cache) and their release point, so C4 sets `lifetime =
      DEVICE` on their requests, and moving the holders is not needed. C4
      needs this now because the pools are on Qwen's default path, and the
      attention staging is on every no-`-c` arm's path: demoted layers run
      their attention on the CPU through it (`gs:82445-82450`,
      `:82640-82645`, `:82842-82844`). Left `GRAPH` in SCRATCH, each would
      trip the boundary WARN at every `graph_compute`.
    - **`CONTEXT`, pending llama.cpp-30r1:** the other 11 holders (12
      lines). Their handles live in file-scope statics or a `thread_local`,
      so no backend context holds them yet, and stamping them `CONTEXT`
      before 30r1 moves them would make the owner check fire on every
      context free. So until 30r1 lands they carry the interim value
      **`UNOWNED`**, which `select_zone` sends to WEIGHT, where each lands
      today, and which no boundary check counts. That is today's behaviour
      made explicit, not a new lifetime: 30r1 deletes the value. None of
      them is on a C4 lead arm's path, and if one is, the zone-0 criterion
      catches it.
    - **The ratchet** (accepted in addendum 10 (b)): C4's census gate fails
      on any request site carrying `UNOWNED` that is not one of these 12
      lines, and it cites llama.cpp-30r1 in the failure message.
- **Host SCRATCH reads RUNTIME through 23mk's function, not a copy**
  (addendum 7 (c)).
  - The fold at `uc:27727-27730` today reads plan fields:
    `moe_vram_runtime_bytes` (`= max(expert_ptrs, control_pool)`,
    `uc:27520`, the llama.cpp-s4ip defect), `pp_pipeline_scratch_bytes`
    and `pp_moe_onednn_scratch_bytes`.
  - C4 replaces those three terms with one read of
    `unified_cache_get_planned_runtime_zone_requirement(dev, &out)`
    (`uc:1568-1575`), the function the VRAM RUNTIME zone is sized through
    (`uc:4520`) and that 23mk routes through. That function already sums
    the PP pipeline scratch, the PP MoE oneDNN ring net of its KV-zone part,
    and the MoE control requirement. Keeping the two PP terms beside it would
    count them twice.
  - `tp_vram_runtime_bytes` is not in that function and stays. It is 0 on
    Qwen. `dma_staging_pool_bytes` is 0 in mmap mode (m-2).
  - **Order.** The function reads published values, so C4's source gate
    pins that the setters for `pp_pipeline`, `pp_moe_onednn` and
    `moe_control` run before the host SCRATCH sizing. Run2 prints the RUNTIME
    requirement at `run2.log:219`, before the host staging line at `:243`.
    If the getter returns false, sizing takes the existing reject path
    (`host_scratch_valid = false`).
  - **Qwen's figure.** Run2's RUNTIME requirement is 4265.6 MB (pipeline 0.0,
    moe_onednn 4265.0, moe_control 0.6; `run2.log:2732`), against the 0.0 +
    4265.0 + `moe_vram_runtime_bytes` those terms contributed before. So host
    SCRATCH changes by 0.6 MB minus `moe_vram_runtime_bytes`, which is at
    most +0.6 MB. The base pre-size stays at or below 7923.7 MB, which is
    still 4 chunks.
- **The runtime pre-allocate stops carrying the same two terms** (r3 m3, the
  m-2 pattern).
  - `ggml_sycl_configure_host_zones_for_plan` pre-allocates a second pinned
    pool of `onednn_scratchpad_bytes + dma_staging_pool_bytes +
    pp_pipeline_scratch_bytes + pp_moe_onednn_scratch_bytes`
    (`gs:5649-5653`). The PP pipeline scratch and the PP MoE oneDNN ring
    are therefore charged twice on the host: in host SCRATCH through the
    RUNTIME function, and here. 3c's replacement of the SCRATCH terms did
    not touch this site, so the double charge was still there.
  - **One owner: host SCRATCH**, because addendum 7 (c) put the RUNTIME
    read there. The runtime pool is not where either term is used: once host
    zones are configured, `host_pool_alloc` (the runtime chunks,
    `pinned-pool.cpp:209-212`) serves only the KV-zone miss fallback
    (`uc:15483-15495`); the other callers take it only when zones are not
    configured (`uc:15440`, `:15492`, `:10565`, `:20614`,
    `common.hpp:2982`). And both terms are VRAM allocations (above).
  - C4 drops them from `runtime_bytes`, which becomes
    `onednn_scratchpad_bytes + dma_staging_pool_bytes`.
    `reserve_pp_moe_onednn_scratch` (`gs:5654-5681`) is a VRAM reserve and
    is unchanged.
- **The criteria** (C4 lead run).
  - No `Pinned pool grow_zone: zone 0` line appears after the pre-size,
    through the end of the completion. Zone 0 is WEIGHT
    (`pinned-pool.hpp:62-67`). Its positive control is the
    `Pinned pool pre_allocate: grew` line of the same run: both are INFO,
    so the absence counts only in a run that printed the control (r3 m2).
  - **SCRATCH (zone 3)** grows only through the notify path: every zone-3
    `grow_zone` line follows a `[HOST-ARENA] SCRATCH demand` line with
    `grown` > 0, and is one chunk of `max(2048 MB, align_up(deficit + n x
    alignment))`. A zone-3 grow from a miss is a FAIL, and names the
    unplanned tenant by its cohort.
  - **KV (zone 1)** grows at context init, one chunk per host-KV buffer
    that does not fit, and the sum of its growth covers the printed host KV
    (§7 C4 step 3). On a run with no demotion it does not grow.
  - No `host compute buffer ... exceeds notified` WARN.
- **The boundary with llama.cpp-23mk** (`docs/design/sycl-alloc-zone-routing.md`,
  23mk's worktree; addendum 7 (c)). 23mk routes **device**-scope requests.
  For host sites, 5efe owns role, zone, lifetime and sizing, and publishes
  them in the site table above. 23mk owns each site's conversion. Its class-H
  rows cite that table: the nine host landings this change moves, including
  `cpu-expert-pool.cpp:62`, `pinned-buffer-pool.cpp:88` and the
  `ggml_sycl_attn_host_staging_get` rows (`gs:81783-81832`); row 56, beside
  `gs:39954`'s host-pinned fallback (and llama.cpp-nl0f for the zero-copy
  read); and row 118, the DMA staging pool (m-2 below). Host SCRATCH reads
  the RUNTIME function above, and does not copy it.

**The DMA staging pool in mmap mode** (r2 m-2).
- It is 1700 MB: 2 x `dma_streamed` (`run2.log:2752`), where `dma_streamed`
  is "largest tensor the host->device weight stream can carry"
  (`zone-sizing.hpp:80`, `uc:27537`).
- It is charged twice: in SCRATCH (`uc:27727-27730`) and in the runtime
  pre-allocate (`gs:5648-5651`).
- It exists for streaming, and §5.3 refuses streaming for every
  `FILE_MAPPED` source. So in mmap mode it is an idle reservation, which P4
  forbids.
- C4 sets `dma_streamed = 0` when no host weight is `PINNED`-backed. Both
  charges then vanish, and the log prints `DMA staging pool: 0.0 MB (no
  pinned host weights)`.
- `-lm none` keeps it, pending 23mk's row 118.

**Pre-registered Qwen zones and pre-size after C4** (r1 C-1, r2 I-1).
- Host experts are all AOS, registered in place (`gs:33437-33457`), so
  `weight_host_bytes` = 0 and `host_cpu_owned_bytes` = 0. The host total is
  experts only: 61671 host tensors = 20557 groups x 3 (`run2.log:2824`).
- `host_zone_weight_bytes = 0 x 1.2 + k_host_runtime_reserve_bytes (64 MiB,
  uc:27103)`, and nothing else reaches WEIGHT. So `[HOST-ARENA] configured
  zones: WEIGHT=` reads **64.0 MB**, against 122924.2 MB in the baseline
  (`run2.log:2893`).
- **The pre-size is two pools, and the "Committed" line counts both**
  (`gs:5615-5688`). Run2's 68 chunks were 65 + 3 (`run2.log:2892`, `:2897`,
  `:2900`).
  - **Base** (`pre_allocate_host_pool`, `gs:5643`): WEIGHT + KV + STAGING +
    SCRATCH. SCRATCH loses the 1700 MB DMA pool (m-2): 7775.9 - 1700.0 =
    6075.9. The base is 64.0 + 64.0 + 1719.2 + 6075.9 = 7923.1 MB, which is
    **4 chunks**, and the log reads `pre_allocate: grew 4 chunks`.
  - **Runtime** (`pre_allocate_runtime_chunks`, `gs:5649-5653`): today
    oneDNN scratchpad + DMA pool + PP pipeline + PP MoE oneDNN. Run2's need
    was 6056.9 MB (`:2897`): 1700.0 of DMA pool, 0.0 + 4265.0 of the two PP
    terms (`run2.log:2732`), so 91.9 MB of oneDNN scratchpad. C4 drops the
    DMA pool (m-2) and the two PP terms (r3 m3), so the need is **91.9 MB,
    1 chunk**: `pre_allocate_runtime: grew 1 chunks`.
  - So `[HOST-ARENA] Committed` reads **10240.0 MB in 5 chunk(s)**, against
    139264.0 MB in 68 (`run2.log:2900`). rev 3c's 7 chunks, 14336.0 MB, kept
    the double charge.
  - **The reading names what did not land:**

    | lands | base | runtime | Committed |
    |---|---|---|---|
    | m-2 and m3 (C4 as specified) | 4 (7923.1) | 1 (91.9) | 5 chunks, 10240.0 MB |
    | m-2 only | 4 (7923.1) | 3 (4356.9) | 7 chunks, 14336.0 MB |
    | m3 only | 5 (9623.1) | 1 (1791.9) | 6 chunks, 12288.0 MB |
    | neither | 5 (9623.1) | 3 (6056.9) | 8 chunks, 16384.0 MB |
- **Falsifiers.**
  - A WEIGHT zone above 64.0 MB must equal 64 MiB + 1.2 x the reordered-expert
    bytes, and the placement report must print that sum. Otherwise the split
    missed an accumulation site.
  - A base above 4 chunks names the zone that grew.
- **Re-checked against the early-plan fix** (§F2a I-9 addendum 2).
  - The configured zones come from the late plan: `configured zones`
    (`:2893`) follows the late `Totals` (`:2821`) and equals
    1.2 x 102383.5 + 64. The late plan already excluded the table.
  - What the record changes is the early plan (§3.1): the early STAGING
    `per_inflight` falls to about 1702 MB, and the early host total to
    102383.5 MB before the split.
  - After C4's split, the early and late `[PLACEMENT] Totals` both print
    charged host 0 and mapped 102383.5 MB. These witness lines are added to
    C4's run.

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
  resident_room = MemAvailable_pre - expected_host - host_headroom(MemTotal)
  mapped_ws     = host_mapped_bytes            (host-executed ranges only)
  ```

- If `mapped_ws > resident_room`, the load logs one WARN,
  `[HOST-PLAN] mapped host weights exceed the page cache that can stay
  resident: mapped_ws=... resident_room=... (MemAvailable_pre=...
  expected_host=... headroom=...)`, and proceeds.
- It uses `host_headroom(total)` below, not the P3 constant (r2 m-4). On a
  62 or 32 GiB host a 29 GiB floor would make the WARN fire on almost every
  load, against a headroom of 15.5 or 8 GiB.
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
  - The ledger and P3 read the same number **only where `total/4 >= 29 GiB`,
    that is `MemTotal >= 116 GiB`** (r2 m-4). On smaller hosts the headroom
    is `total/4` or `total/32`, and P3's measured floor does not describe
    them.
  - th32 §4.3 is asked to cite `host_headroom(total)` rather than the literal
    `24 + 5`.
  - The `min(..., total/4)` cap keeps this host's policy from becoming a
    universal allocation cap on smaller hosts.
- **`GGML_SYCL_HOST_RESERVE_MB` composes; it does not bypass** (r2 m-4).
  Today a valid value replaces the whole budget (`uc:4171-4181`). After C5 it
  is a cap: `budget = min(override, host_ledger_budget())`. It can lower the
  budget, and it can never take it past `MemAvailable - host_headroom`. The
  refusal still names the headroom when the override is the smaller term.

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
  from `lazy_read`'s set. Two exclusion sites read the record, both through
  `deferred_for_file(idx)`, which holds only non-`EAGER`, and so only
  eligible, rows (§3.1, §F2a I-9 addendum 3):
  - the tier, through the hook's `lazy_ranges` (§4.3);
  - `llama_mmap`, through `init_mappings`.

  A table diverted to STORAGE is excluded exactly as a `LAZY_READ` one is. An
  ineligible table is not in the set, and is mapped and prefetched as the
  ordinary weight it is.
- **Three segment sets, one pure function** (r1 I-2, §F2; defined exactly per
  r2 m-16). `file_mapping_segments(tensor_ranges, lazy_ranges, page)`, where
  `tensor_ranges` is the byte ranges of the tensors in this `SYCL_Mapped`
  buffer and `lazy_ranges` is `deferred_for_file(idx)`, returns:

  | set | definition | rounding | bound on lazy bytes |
  |---|---|---|---|
  | map segments | the union of `tensor_ranges`, each range rounded **outward** to pages, then merged. It is **not** a complement: bytes that belong to no tensor of this buffer are never mapped | outward, including at a lazy boundary (GGUF alignment is 32 B, so a lazy and a non-lazy tensor can share a page) | at most one page of lazy bytes mapped per lazy edge (the shared page) |
  | `WILLNEED` segments | the host-executed subset of `tensor_ranges`, each rounded **inward**, minus every page that holds a lazy byte | inward | 0 |
  | lock segments | as `WILLNEED`, under `-lm mmap+mlock` only | inward | 0 |
  | `MADV_RANDOM` edge windows | for each lazy range, `[lazy_begin - W, lazy_begin)` and `[lazy_end, lazy_end + W)` intersected with the map segments, plus the shared pages | inward from each lazy boundary | n/a (advice only) |

  (§F2a I-2(a): only map segments round outward. Every advice and lock set
  rounds inward, so no advice or lock ever reaches a lazy page other than the
  shared page's `MADV_RANDOM`.)

  - Lazy ranges are never in `tensor_ranges`, because deferred tables are not
    `SYCL_Mapped` tensors (§4.4). A lazy byte is mapped only through a shared
    page.
  - Pages strictly between map segments stay the reservation's `PROT_NONE`
    anonymous memory. They are never faulted, and a tier read of one is a
    `SIGSEGV`: a loud bug, not a silent read.
  - `file_mapping_segments` lives in the device-free `file-mapping.cpp`.
    libllama's `ranges_complement` (`src/llama-mmap.cpp:444`) is a different
    set and is not reused.
- **Fault read-around** (r1 m-13, addendum m-19, r2 I-3).
  - Linux's synchronous mmap read-around is **centred on the fault**:
    `start = pgoff - ra_pages/2`, `size = ra_pages`. It is per file, not per
    VMA. So a fault near either edge can read into a neighbouring lazy range:
    forward from a segment that ends at it, and **backward** from a segment
    that starts after it. That covers CPU execution faults and the staging
    bounce's `memcpy` faults alike.
  - **Its reach** on each side is `ra_pages/2`. `ra_pages` is the backing
    device's `read_ahead_kb`, doubled when the open file description carries
    `POSIX_FADV_SEQUENTIAL`. The tier's `dup` fd shares the loader's
    description, and `llama_mmap` sets that advice on it
    (`src/llama-mmap.cpp:475`). Every bdi on this host reads
    `read_ahead_kb = 128`, so the reach here is 64 KiB, or 128 KiB with the
    advice.
  - The tier sets `MADV_RANDOM` on a window of
    `W = max(2 x read_ahead_kb, 2 MiB)` at **both** edges of every lazy range,
    which is at least twice the reach. That splits the VMAs and disables
    read-around there.
  - Device-only ranges are also staged through the mapping, so the same edge
    advice covers them.
- **The second residual path: the loader's own whole-file mapping** (addendum
  m-19, r2 I-4).
  - `llama_mmap` still maps whole files during the load. `load_all_data` reads
    the tensors that are not in `SYCL_Mapped` through it: `CPU_REPACK` and
    plain CPU sources, and lazy tables on the CPU path before th32.
  - Today that mapping advises `MADV_RANDOM` only over the lazy range itself
    (`src/llama-mmap.cpp:541-543`), and it sets `POSIX_FADV_SEQUENTIAL` on the
    fd (`:475`), which doubles the read-around. A fault in the neighbouring
    VMA can therefore read into the lazy range.
  - **Worse, it advises `MADV_HUGEPAGE` over every non-lazy range** shrunk to
    pages (`src/llama-mmap.cpp:517-532`), and that includes the page runs
    next to each lazy range. Its own comment names the effect: "a fault in a
    VM_HUGEPAGE area reads a whole 2 MiB folio before POSIX_MADV_RANDOM is
    honoured" (`:518-522`).
    - In `do_sync_mmap_readahead` the `VM_HUGEPAGE` branch comes before the
      `VM_RAND_READ` return. It aligns the index down to 2 MiB and reads
      2 MiB, or 4 MiB without `VM_RAND_READ`.
    - THP is `[madvise]` on this host.
    - So a `MADV_RANDOM` window inside a `VM_HUGEPAGE` VMA does nothing. A
      fault in the page before a lazy range reads up to the next 2 MiB
      boundary into it. A fault after it reads back to the previous
      boundary.
  - **C4's `llama_mmap` change, both halves.**
    - The lazy-range advice covers `[lazy_begin - W, lazy_end + W)`, from the
      record, not the lazy range alone.
    - The same window, **widened outward to 2 MiB alignment**, is excluded
      from `MADV_HUGEPAGE`, and advised `MADV_NOHUGEPAGE`. The explicit
      `NOHUGEPAGE` also covers THP `[always]`, where a VMA is huge unless
      told otherwise.
  - The tier's own mapping never advises `MADV_HUGEPAGE` (below), so it needs
    only the `MADV_RANDOM` windows.
- **The bound, stated per mapping** (r2 I-4). This counts lazy bytes that a
  non-lazy fault can bring into the page cache:
  - **The tier's mapping:** the shared page at each lazy edge, and nothing
    else. There is no `HUGEPAGE`, and the `MADV_RANDOM` windows are at least
    twice the reach.
  - **`llama_mmap`'s mapping:** the same, once C4's window is both
    `MADV_RANDOM` and `NOHUGEPAGE`.
  - **Either mapping:** plus whatever bcachefs reads at extent granularity
    (256 KiB, th32 §2.4) around a page it does read.

  The gates below measure what remains. The C3 gate measures the band that
  read-around would reach. The C4 gate measures the band a 2 MiB folio would
  reach.
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
- **The residency gate** (C3; r1 I-1, §F2; hygiene per r2 m-14). It measures
  the page-cache residency of the **file**, never `mincore` on the tier's
  hole.
  - **Fixture.** A 64 MiB file in the build directory: tensor A (16 MiB), a
    lazy range (24 MiB), then tensor B (16 MiB), with 32 B-aligned,
    deliberately non-page-aligned boundaries. The test sets
    `POSIX_FADV_SEQUENTIAL` on its fd, as the loader does, so the reach
    matches production.
  - **Eviction, per arm.** Before **each** arm, every mapping of the fixture
    is unmapped (`FADV_DONTNEED` skips mapped pages), then
    `POSIX_FADV_DONTNEED` runs over the whole file, and the probe must read
    about 0 resident. If it does not, that arm exits 77: the directory
    cannot be evicted (tmpfs, or a filesystem that ignores the advice).
  - **Probe.** `cachestat(2)`, called as `syscall(__NR_cachestat, ...)`
    because glibc has no wrapper here (Linux 6.5+; this host runs 7.1). On
    `ENOSYS`, or where `__NR_cachestat` is undefined, the fallback is
    `mincore` over a **test-owned**, separate `PROT_READ` mapping of the
    file. That reports the page cache of those file pages whichever mapping
    faulted them. `mincore` reports every page resident for a file the
    caller neither owns nor can write, which does not apply to a fixture the
    test creates.
  - **The two bands.**
    - The **interior** is the hole minus a 1 MiB guard band in from each
      edge. The `WILLNEED`, lock and `pread` arms measure it.
    - The **edge band** at each lazy edge is `[shared page + 1 page,
      edge + reach)` inward from the edge, where the reach is computed from
      the fixture's bdi `read_ahead_kb` (the bdi of its `st_dev`), x2 for
      the fd's `FADV_SEQUENTIAL`, /2. The read-around arms measure it (r2
      I-3). If the reach does not extend at least 2 pages past the shared
      page, those arms exit 77, since an empty band passes vacuously.
  - **Arm 1 (the property).** Map through the tier with the lazy hole, apply
    the `WILLNEED` segments, and read every byte of A and B, including the
    bytes that share a page with the hole. That reads outward-rounded bytes
    without a `SIGSEGV`, which also tests I-2(a). Then the interior must read
    **0 resident pages**.
  - **Arm 2 (the mlock case).** Arm 1 under the lock path, with the lazy range
    placed *before* a locked tensor. Same bound. It exits 77 if
    `RLIMIT_MEMLOCK` is below the fixture size.
    - **It cannot fail by construction** (r2 m-14): the hole is `PROT_NONE`,
      not file-mapped, and edge read-around stays inside the guard band. So
      it gets its own mutant, which maps the hole and locks the prefix
      `[0, B_end)`. The mutant must read the interior resident.
  - **Arm 3 (negative control).** A `GGML_SYCL_PRIVATE_TESTING` mutant that
    passes the hole **as a tensor range**, that is, adds it to
    `tensor_ranges` and leaves it out of `lazy_ranges` (the one reading
    m-16's definition allows). It maps and `WILLNEED`s the hole, and must
    read resident pages in the interior. This shows the gate can fail.
  - **Arms 5 and 6 (read-around, forward and backward; addendum m-20, r2
    I-3).**
    - Arm 5 faults the mapped page adjacent to the hole's start, and arm 6
      the one adjacent to its end. Both are real faulting pages, not
      addresses the tier does not map.
    - With the edge `MADV_RANDOM` in place, the **edge band** must read 0.
    - Each has a **control twin** with no edge advice, which must read
      resident pages in the edge band. This shows the arm could fail.
    - A mutant that advises only the segment tail fails arm 6.
  - **Arm 4 (synchronous positive control) runs last**, on its own freshly
    evicted fixture. `pread` 1 MiB inside the interior, then the probe over
    that MiB must read resident. There is no polling and no `WILLNEED` race.
    It runs last because its `pread` populates the hole, plus readahead past
    the MiB.
- **The loader-side gates** (addendum I-9, I-10, r2 I-4). The same fixture and
  probe are driven through libllama, with no SYCL:
  - **`llama_mmap` prefetch exclusion.** Construct `llama_mmap` with
    `prefetch = -1`, the `init_mappings(true)` case, and lazy ranges from a
    record. Wait 10 s, then the interior must read 0. The mutant is an
    **empty** record (the T6 diversion with ranges taken from `lazy_read`); it
    must read resident within the same 10 s, polled.
  - **The `HUGEPAGE` arm** (r2 I-4).
    - A second fixture places the lazy boundaries at 2 MiB + 1 MiB offsets,
      not 2 MiB-aligned.
    - The arm faults the page adjacent to each lazy edge through
      `llama_mmap`'s mapping.
    - The **folio band** is the lazy bytes from the shared page + 1 page up
      to the edge's 2 MiB boundary, 1 MiB here. It must read 0.
    - The mutant leaves the window `MADV_HUGEPAGE`, and must read the folio
      band resident. If the mutant does not populate it (THP `[never]`, or
      a kernel whose file THP path ignores the advice), the arm exits 77
      naming that, rather than passing on a control that cannot fail.
  - **The loader's ranged lock.** Lock a tensor that sits after the lazy
    range through `llama_mlock::lock_range`; the interior must read 0. The
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
    named terms, plus the 64 MiB reserve. Non-weight host requests move to
    SCRATCH (§4.4). For Qwen the WEIGHT zone is 64.0 MB (§4.4).
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
| `tensor-types.hpp:36-40` `memory_tier::MMAP`, used at `gs:14317`, `:19256`, `:19279`, `:34766`, `:35665`, `:98734` (r2 m-11) | "File-backed mmap (slowest, fallback)"; in fact the initial value and "not found" result of the dead `get_cached_tensor_ptr` (§2.7) | `memory_tier::UNRESOLVED`, so no name in the tree says "mmap" without meaning a `FILE_MAPPED` owner |

The registry's `alloc_type::MMAP` (r1 m-4) is split and **named for its
release semantics** (r2 m-6), because "host memory that is not USM" would
also describe a file mapping, and invite registering one under it.
- **`HOST_MALLOC`**: registered host memory that `unified_free_record`
  releases with `::free`, the pinned-cap overflow (`uc:14922-14929`).
- **`HOST_OWNED_ELSEWHERE`**: registered only so the registry can identify
  it; `unified_free_record` never frees it. This is the tiered KV's synthetic
  base, a `std::vector` that its own context clears (`gs:39615-39619`,
  `:39636-39637`, released at `:38403-38410`).
- `alloc_type::FILE_MAPPED` is new.

The ten `MMAP` consumers, all in C2:

| consumer | meaning today | after |
|---|---|---|
| `alloc-registry.hpp:26`, `:143`, `:223` | enum, host-accessible set, commented usm mapping | `HOST_MALLOC` and `HOST_OWNED_ELSEWHERE`; `FILE_MAPPED` added to the host-accessible set |
| `common.hpp:2863` | maps to `usm::alloc::unknown` | all three map to `unknown` |
| `dpct/helper.hpp:1327` | pointer-attribute classification | all three |
| `gs:38730`, `:38755` | tiered-KV memcpy for the synthetic base | `HOST_OWNED_ELSEWHERE` only |
| `gs:39638` | registers the KV synthetic base | `HOST_OWNED_ELSEWHERE` |
| `gs:87745` | (classification) | all three |
| `mem-ops.cpp:151` | non-USM host classification | all three (a `FILE_MAPPED` source takes the bounce) |
| `set_rows.cpp:321` | non-USM host classification | all three |
| `uc:14927` | malloc-overflow `::free` | `HOST_MALLOC` only; a `FILE_MAPPED` or `HOST_OWNED_ELSEWHERE` record never reaches it (§4.1) |
| `uc:21635` | `query_registered_location` | all three, each to its own tier |

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
| preload's host-planned copy (`gs:34696-34730`, r2 m-7) | `memcpy`s a late-plan host weight into an uncharged pinned WEIGHT allocation | served in place from the mapping, or a planned, charged copy (§4.3 site table); a CPU copy, never a device read |

**The `stream_dma` refusal.**
- The check runs at entry, before the `copy_fn` test, so custom-copy callers
  are covered.
- `dma_stream_result` (`uch:3622`) gains `refused_file_mapped`, and a
  `stream_dma_file_mapped_refusals` counter increments.
- Callers route a refusal like `mmap_direct_failed` today: to the CPU executor
  (`ggml_sycl_cpu_fallback_graph`, `mmq.cpp:7653-7658`), because a host-tier
  weight executes on the CPU. C4 routes `refused_file_mapped` **explicitly**
  in all four callers (`mmq.cpp`, `mmvq.cpp`, `dmmv.cpp`, `getrows.cpp`),
  not by relying on `mmap_direct_failed`.
- **The refusal is terminal in `getrows`' retry loop** (r2 m-5). That loop
  halves the slice on any non-ok result (`getrows.cpp:2762-2795`), so a
  deterministic refusal would repeat `log2(slice/min)` times, each counted
  and each logging `[GET_ROWS] DMA retry`, and its fallback keys on
  `mmap_direct_failed` (`:2806`). C4 breaks out of the loop on
  `refused_file_mapped` before the retry test, and routes it to the CPU
  fallback. A host test over a fake `stream_dma` that refuses asserts one
  call, one count and no retry line.
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
  counts a device-bound copy of `FILE_MAPPED` bytes, bounced or direct,
  outside a load transaction or a cache materialization
  (`direct_stage_weight` or a fill), **at the instrumented chokepoints** (r2
  m-17):
  - `mem_copy`, `mem_copy_async` and `mem_copy_ptr_async` (`mem-ops.hpp:13-62`);
  - `unified_cache::copy_to_device_async` (`uc:10799`);
  - `stream_dma` (`uc:13959`).

  A copy through any other path is not counted. The standing
  `llama.cpp-v4jk` audit covers those. The lead's load gate reads the counter
  as 0.
- **Where the lead reads both counters** (r2 m-18). One line,
  `[FILE-MAPPED] counters: stream_dma_refusals=%llu
  device_copies_outside_materialization=%llu`, printed through
  `GGML_LOG_WARN`. WARN maps to `LOG_LEVEL_WARN` (2) (`common/log.cpp:533`),
  so it survives `llama-completion`'s default threshold, `LOG_LEVEL_INFO`
  (3, `common/log.cpp:30`). It does **not** survive `llama-cli`'s, which is
  `LOG_LEVEL_ERROR` (1, `tools/cli/cli.cpp:36`), so every `llama-cli` arm
  that reads it passes `-lv 4` (§7, r3 C). It prints:
  - once at the end of `ggml_backend_sycl_model_load_end`, after the
    preload, so a killed run still has one;
  - once at model unload.

  A missing line is a failed criterion, not a zero.
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
- `grep -E '^(MemAvailable|Shmem|AnonPages|Cached|Mlocked|Unevictable):'
  /proc/meminfo` is sampled **every 1 s**;
- the watchdog floor is `MemAvailable < 25 GiB` = 26214400 kB, the same unit
  and value as the criteria;
- each run is done once;
- `journalctl -k` is checked for `GT reset|guc_id|CAT error` after it.

**Every model gate set carries a default-context arm** (owner ruling
2026-09-27), and every such arm is scored:
- The **Mistral** gate already runs `llama-completion` with no `-c`, so it is
  its own default-context arm (n_ctx_train 32768).
- Every **Qwen** and **GPT-OSS** gate below that pins `-c 4096` has a no-`-c`
  twin.
- **The vehicle** (lead's clarification, 2026-09-27). A default-context arm
  is `llama-cli`, `llama-completion` or `llama-server` with no `-c`, whose
  context defaults to `n_ctx_train`. `llama-bench` (`n_ctx = p + n + d`) and
  `llama-perplexity` (512) stay at their fixed contexts; a "no `-c`" bench
  arm is vacuous, so C4 step 6 has none. A **programmatic** arm (a harness on
  `llama_context_default_params`, whose `n_ctx` is 512,
  `src/llama-context.cpp:5091`) must set `n_ctx = 0` explicitly (`:540`), and
  is **VOID** unless `llama_n_ctx(ctx) == n_ctx_train`. No 5efe test is such
  an arm today. The rule binds any that is added.
- **Verbosity is part of the vehicle** (r3 C, m2). A criterion that reads a
  log line is void if the vehicle drops that line's level, and an absence
  criterion then passes vacuously.
  - `common_log`'s default callback prints a line only at `verbosity <=
    common_log_verbosity_thold` (`common/log.cpp:542-546`). GGML INFO maps
    to 4 and WARN to 2 (`:532-533`).
  - `llama-completion` keeps the default threshold, 3 (`common/log.cpp:30`,
    `LOG_DEFAULT_LLAMA`), so it prints WARN and drops INFO. The Qwen arms
    read INFO lines (`pre_allocate`, `configured zones`, the `SCRATCH
    demand` line), so they pass **`-v`**, as run2 did.
  - `llama-cli` sets `params.verbosity = LOG_LEVEL_ERROR` (1) before it
    parses (`tools/cli/cli.cpp:36`), so it drops WARN too, the demotion line
    (`gs:18382`) and `[FILE-MAPPED] counters` included. `-lv N` overrides it:
    `common_params_parse_ex` first sets the threshold from that default
    (`common/arg.cpp:769`), and the `-lv` handler, which runs while the
    arguments are parsed afterwards, sets both `params.verbosity` and the
    threshold (`:3950-3962`). So **every `llama-cli` arm passes `-lv 4`**,
    which admits WARN and INFO.
  - `-lv 4` does not move the digit scoring. The answer is console output
    on stdout (`common/console.cpp:75`, `:1145-1149`), and every log line
    goes to stderr (`common/log.cpp:113-116`). The arm captures the two
    streams to **separate files** and scores `grep -cx '1, 2, 3, 4, 5'` on
    the stdout file only, since a merged capture can tear the answer line.
- llama.cpp-uize part 3 has landed (`35ce2e65f`, `812c7d790`, `7259727e6`,
  `ae4204ba7`, all in this tree). So a KV cache that does not fit VRAM is
  re-placed to the host tier, and its layers' attention runs on the CPU. The
  runtime prints `[SYCL-PLAN] KV overflow re-placed to host tier: N layer(s)
  demoted, S SWA (X MB host KV, layers A..B) on device D for n_ctx=C; the
  device has F MB free for KV. ...` (`gs:18382`).
- **PASS** needs correct tokens, and that line with the pre-registered N and
  range. **Any** `runtime KV update rejected` line is a FAIL: the refusal is
  now only the fallback for when the host tiers cannot hold the KV either, so
  no attribution to uize remains. Garbage tokens, an abort, or `Shmem`
  growth are also 5efe FAILs.

**C1. Restore `mmap_support`, and the lazy-range record** (§3, §3.1).
- RED `test-sycl-mmap-support-caps-source` (Python, `tests/`, `sycl;host;source-contract`).
  - It checks that the initializer in `ggml_backend_sycl_device_get_props`
    names `.mmap_support`, binds it to `host_unified_known &&
    !host_unified_memory`, and is neither a literal `true` nor absent.
  - Today it fails on the four-initializer list.
  - It also checks that `lm:2693-2696` is still present, since C1 must not
    remove it. **C4 amends that clause** to the narrowed form of §4.3
    (r2 m-8).
- **The record** lands in C1, with these host-only tests (`host;lazy`,
  `tests/`, libllama only, no device). C1 factors three pure functions out of
  `load_tensors` so that a host test can drive them without a device:
  - `llama_lazy_mode_resolve(mode, caps)`: the moved fallback block, over a
    list of device caps;
  - the pre-fill: (declaration, `weights_map`, `hparams`, `mode_pre`,
    `mode_post`, `taker`, `SUPPORTED`) to a record;
  - the early plan's inventory filter.

  The loader calls the same functions.
  - `test-llama-lazy-range-record`:
    - It builds `llama_tensor_weight`s from a test-written GGUF of five
      tensors in two files. Three of them are declared lazy for the test's
      arch: one at 4 GiB + 1 byte of logical size, one at 4 GiB, and one
      **sparse table above 4 GiB**, its data region an `ftruncate` hole, so
      the fixture costs no disk. The GGUF also holds one tensor of the same
      name under a second arch that does not declare it (the gemma3n shape).
    - Rows equal the declared tensors' (file, offset, size) exactly, whatever
      the mode. The second arch's same-named tensor gets **no** row (the
      (arch, name) key; addendum 5).
    - **The eligibility matrix** (addenda 4 to 6), over `taker` x `mode_pre`
      x `mode_post` x `SUPPORTED`:
      - `lazy_eligible` comes from T6a's function, called and not restated,
        with `(mode_pre, true)` when `taker` and `(mode_post, SUPPORTED)`
        otherwise;
      - `deferral` is `STORAGE` iff eligible and `taker`, `LAZY_READ` iff
        eligible and not `taker`, and `EAGER` otherwise;
      - `deferral != EAGER` iff `lazy_eligible`, asserted on every row of
        every cell.
    - **The required witness** (addendum 4). With `AUTO`, a caps list holding
      one device with `mmap_support = false`, and `taker`, the sparse row's
      bit is set, its `deferral` is `STORAGE`, and the early plan's filter
      excludes it. `llama_lazy_mode_resolve` does turn `mode_post` to `OFF`,
      which is what makes this cell discriminate. **RED mutant:** the fill
      reads `mode_post` for the taker (a post-fallback fill). The bit is then
      clear and the filter includes the table.
    - **The `-ngl 0` witness** (addendum 5). A SYCL-registered load whose
      weights are not SYCL-owned has `taker` false. Every row's `deferral` is
      then `LAZY_READ` or `EAGER`, and never `STORAGE`, across the whole
      matrix. How `taker` is derived is pinned by the source test below and
      by lead witness W3.
    - **`!SUPPORTED`, off the taker.** The bit is clear under every mode, and
      the pre-fill prints the "mmap is not available ... loaded into RAM in
      full" WARN once per declared row.
    - **The attributes are per model** (addendum 7 (a)). Two records are
      filled from two `qwen4exp` hparams sets with different `ple_n_heads`,
      and each record's `attrs(name)` returns its own value, whichever was
      filled last. **RED mutant:** the attributes cached in a static beside
      the declaration, under which the first record reads the second's
      value.
    - `deferred_for_file` holds exactly the non-`EAGER` rows.
    - `lazy_read`'s `any()`, `has()` and `for_file()`, and the ranges passed
      to `llama_mmap`, agree with the record. `any()` is false when `taker`,
      and under `no_alloc`. `add()` accepts exactly the `LAZY_READ` rows,
      under every mode value `lazy_read` holds.
    - **The fit-pass invariant** (r3 m1; `lml:1775`). With `w = nullptr`
      (the `no_alloc` call), `add()` returns true for a `LAZY_READ` row and
      leaves membership empty; with a weight it returns true and inserts the
      name. **RED mutant:** rev 3c's `w != nullptr` term in the decision,
      under which the `no_alloc` call returns false for the `LAZY_READ` row.
    - **The per-model line** (th32 2x). The pre-fill prints exactly one
      `lazy-range record: mode_pre=<m> mode_post=<m> taker=<0|1>` line per
      model, read through a callback installed with `llama_log_set`, and it
      prints even for a model with no declared name. In the addendum-4
      witness cell it reads `mode_pre=auto mode_post=off taker=1`.
    - **RED** is a mutant, compiled into the test binary only. libllama has
      no private-testing switch, so the test links a copy of the record and
      `lazy_read` sources built with a test-local define. In it
      `lazy_read::add` keeps its own range copy and its upstream mode and
      floor tests, and the record's fill is skipped for one tensor. The
      agreement check must fail naming that tensor, and the `add()` check
      must fail on the 4 GiB row under `ON` with `taker`, which is `STORAGE`
      and which upstream's `add()` would accept.
  - `test-llama-lazy-range-prefill`:
    - It feeds three names through the pre-fill and the `create_tensor`
      check: a declared lazy name, an undeclared flagged name, and a
      declared-but-unflagged name.
    - The first gives one row, filled before any `create_tensor` call, with
      the attributes `llama_model_lazy_tensor_attrs` returns for it.
    - The other two abort naming the tensor and the arch, each in its own
      process, scored on rc 134 plus the name.
    - **Mode-independent** (addendum 4). The unflagged case aborts under
      `-lzm off` and under `AUTO` at 4 GiB, both of which leave the row
      ineligible and `EAGER`. **RED mutant:** rev 3a's check, which compares
      the flag only against non-`EAGER` rows, lets both through.
    - `value(name)` equals the weight's (file, offset, size), and a write
      after the pre-fill aborts.
    - **The ineligible early-plan witness** (th32 r19 M-1, assigned to 5efe).
      With `taker`, the early plan's filter includes an ineligible row, for
      `AUTO` at 4 GiB and for `OFF`, as an ordinary weight, and excludes the
      eligible one.
    - **RED mutant:** the row filled at `create_tensor` instead, under which
      the "before any `create_tensor` call" check fails.
  - `test-llama-lazy-declaration-source` (`host;lazy;source-contract`, r2
    I-5, addendum 5):
    - It extracts every `create_tensor` call in `src/models/*.cpp` that
      passes `TENSOR_READ_LAZY`.
    - It maps the file's model class to its `LLM_ARCH` through the
      registration in `llama-model.cpp`, and each `tn(LLM_TENSOR_*, suffix
      [, bid])` to its GGUF name through `LLM_TENSOR_NAMES`
      (`llama-arch.cpp:561`), with a bid argument becoming `blk.%d.`.
    - It requires a `llama_model_lazy_tensor_names` entry **under that arch**
      for each call, and a flagged call site in that arch's file for each
      entry.
    - It also requires `llama_model_lazy_tensor_attrs` to have a case for
      every arch that has names, and no case for any other arch.
    - The declaration returns names only (addenda 7 (a) and 8 (a)). No
      static beside it stores an hparams-valued field, and
      `llama_model_lazy_tensor_attrs` writes nothing but its return value.
      Mutant: a function-local `static llama_lazy_tensor_attrs` cache in
      `llama_model_lazy_tensor_attrs`.
    - Today it passes on the two call sites once the declaration exists, and
      it fails before C1, which has no declaration.
    - **Mutants:**
      - a flagged call added in a copy of `gemma4.cpp` with no entry;
      - an entry with no call site;
      - the `qwen4exp` entry moved under `gemma3n`, whose same-named call is
        unflagged;
      - an attribute case deleted.
  - `test-llama-lazy-range-record-source` (`host;lazy;source-contract`):
    - no consumer holds its own **range** container: `lazy_read` has no
      `ranges` member (its membership set of names stays, §3.1), and
      `init_mappings` passes the record, not `lazy.for_file`;
    - **`lazy_read::add()` reads `deferral`** (addendum 5; addendum 6 (2)).
      Its body names `LAZY_READ` and the record, and contains no
      `LLAMA_LAZY_MODE_`, no `llama_mmap::SUPPORTED` and no 4 GiB floor.
      Mutant: upstream's `add()` restored (`lml:1240-1258`), as a rebase
      would restore it;
    - **`taker`** is assigned from `sycl_model_loading_guard.active`, and no
      pre-fill input names `llama_model_sycl_hooks_enabled`. Mutant: the
      process-level selector (addendum 5's RED);
    - **T6a's third argument is exactly `taker ? true :
      llama_mmap::SUPPORTED`** (th32 2x, its §4.1 and T6a row). Two mutants
      must each fail the clause: the argument always `llama_mmap::SUPPORTED`
      (the taker loses its no-mmap eligibility, addendum 6 (1)), and always
      `true` (the non-taker marks a table lazy that `lazy_read` cannot map).
      A source clause is the only discriminator: `llama_mmap::SUPPORTED` is
      a constant `true` wherever `_POSIX_MEMLOCK_RANGE` or `_WIN32` is
      defined (`src/llama-mmap.cpp:711-715`), so on Linux and Windows no
      runtime test can tell either mutant from GREEN;
    - `mode_pre` is captured before the call to `llama_lazy_mode_resolve`, and
      the pre-fill runs after the guard and before
      `llama_model_sycl_compute_early_plan`;
    - `create_tensor`'s `TENSOR_READ_LAZY` branch checks the record and does
      not fill it, and compares the flag with membership, not with
      `deferral`;
    - the early plan's `weights_map` loop consults the record;
    - today it fails on `llama-model-loader.h:118`.
- **Lead witnesses** (r1 I-7, r2 m-1; addendum 5). Three short
  `llama-completion` runs, each with `-v` (§3.1's pre-fill line is INFO, which
  default verbosity drops), each killed at its witness lines, before the
  pinned pre-size and before any tensor data is read.
  - **The margin.** In run2 the witness lines were all printed by t = 3.9 s
    (`run2.log:243`, `:308`, `:314`, `:370`). Chunk 1 of the pinned pre-size
    was allocated at t = 8.12 s (`:2827`). So the margin is about **4.2 s**,
    not the 53 s rev 3a claimed. `:2893` is the summary after the pre-size.
  - **The belt.** C1 does not split the ledger, so its load still sizes
    WEIGHT at 122.9 GB. All three runs set `GGML_SYCL_HOST_RESERVE_MB=16384`, the
    budget override (`uc:4172-4184`), so a late kill cannot commit more than
    16 GiB.
  - **W1, lazy enabled, the taker.** Qwen on the B70 with **no** `-lm`/`-lzm`
    flags. Kill it as soon as the log shows all of:
    - the per-model line `lazy-range record: mode_pre=auto mode_post=auto
      taker=1` (C1's `mmap_support` keeps `mode_post` at `auto` on a discrete
      card);
    - the pre-fill line `lazy-range record: per_layer_token_embd.weight
      (51880 MiB) eligible=1 deferral=STORAGE taker=1 mode=auto`;
    - the early `[SYCL-PLAN] Host staging zone` line;
    - the early `[PLACEMENT] Totals` line;
    - `load_mode = mmap`.

    Pre-registered: early `per_inflight` ≤ 1800 MB, and early host total =
    102383.5 MB ± 0.1 MB.
  - **W2, the ineligible case** (th32 r19 M-1). The same with `-lzm off`, so
    the row is ineligible. The per-model line reads `mode_pre=off
    mode_post=off taker=1`, and the pre-fill line `eligible=0 deferral=EAGER
    taker=1 mode=off`. The early plan must place the table as the ordinary
    weight it is: early `per_inflight` = 103762.3 MB and early host total =
    154263.6 MB, both ± 0.1 MB, the run2 figures (`:243`, `:308`).
  - **W3, `-ngl 0` in a SYCL build** (addendum 5). Qwen with `-ngl 0` under
    `ONEAPI_DEVICE_SELECTOR=level_zero:1`, so the SYCL backend and its hooks
    are registered but the weights are CPU-owned. Kill it at the pre-fill line
    plus `load_mode = mmap`. Pre-registered:
    - the per-model line reads `mode_pre=auto mode_post=auto taker=0`, and
      the pre-fill line `eligible=1 deferral=LAZY_READ taker=0
      mode=auto`, because C1's `mmap_support` keeps `mode_post` at `auto` on a
      discrete card;
    - **no** `[SYCL-PLAN] Host staging zone` line and **no** early
      `[PLACEMENT] Totals` line: the early plan does not run.

    A `deferral=STORAGE` there is the process-level-selector defect (FAIL).
  - None of them is a load. None reaches the pre-size, and none reads the
    table, so none enters the C1-to-T6 window (§3.1, **Data edges**).
- Lead gates: the Mistral gate on `:1` and `:0` (default context). Tokens are
  unchanged, since Mistral has no lazy tensor.
- **Data edge:** th32 T6a lands no later than C1 (§3.1).

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
  - the private `::free` and `sycl::free` branch counters both stay 0 at
    release.

  Mutants (r2 m-6; the rev 3a mutant was equivalent): the record registered
  as `HOST_MALLOC`; the `FILE_MAPPED` branch moved after the `zone_managed`
  branch into the `sycl::free` fallthrough.
- RED `test-sycl-raw-mmap-allowlist-source` (`sycl;host;source-contract`),
  §4.1. It uses the tokenizer, with the comment, string and `mmap (` fixture
  (r2 m-12). Mutant: an `mmap (` call added to a non-listed file.
- RED `test-sycl-alloc-tier-census-source` (`sycl;host;source-contract`). No
  `MMAP_TRACKED`, no `alloc_type::MMAP`, no `HOST_NON_USM` and no
  `memory_tier::MMAP` remain anywhere in the tree, `tests/` included. Today
  it fails on the 12 + 3 + 10 + 7 sites of §4.8 (r2 m-11).
- Also: the canonical contract §3 allowlist gains the three wrappers, and the
  static-storage census and the allowlist gates are re-run.

**C3. The segment plan and the residency gate** (§4.6).
- RED `test-sycl-file-mapping-segments` (`sycl;host`), a pure-function test.
  - Map segments are the union of the tensor ranges rounded outward, and never
    a complement (r2 m-16). `WILLNEED` and lock segments round inward.
  - At most one lazy page is mapped per edge.
  - Cases: an interleaved lazy range, a lazy range sharing a page with a
    tensor at each end, back-to-back lazy ranges, a lazy range at offset 0
    before a lock segment, and bytes that belong to no tensor of the buffer
    (never mapped).
  - Mutants: inward map rounding (fails the shared-page read), outward
    `WILLNEED` rounding (fails the zero-lazy bound), the complement reading
    of map segments (maps the no-tensor bytes).
- RED `test-sycl-file-mapping-residency` (`sycl;host`), §4.6's arms.
  - The fixture lives under `${CMAKE_BINARY_DIR}`, which is not tmpfs.
  - Each arm has its own eviction precondition, and exits 77 if it fails.
  - Arm 2 exits 77 if `RLIMIT_MEMLOCK` is below the fixture size.
  - Arms 5 and 6 exit 77 if the computed reach does not extend 2 pages past
    the shared page (r2 I-3).
  - Arms 1-2 are measured over the interior. Arms 5-6 and their twins are
    measured over the edge band.
  - Arm 4 runs last, on a fresh fixture (r2 m-14).
  - The must-fail cases: arm 2's prefix-lock mutant, arm 3, and the arm 5
    and 6 control twins.
  - A mutant that advises only the segment tail fails arm 6.
- The loader's ranged lock (§4.6) also lands here, in libllama, with RED
  `test-llama-mlock-ranged` (`host;lazy`). It is §4.6's loader lock gate: a
  lazy range before a locked range, the interior reading 0, and the
  whole-prefix `grow_to` mutant reading resident.
- Lead: none.

**C4. One published fact, the planner split, host role routing,
`SYCL_Mapped` and the loader** (§4.3, §4.4, §5.3). This lands as **one
commit**, because the fact and its only consumers must land together (r1 C-1,
I-5, §F2). It is the commit that changes every SYCL load.
- **Prefetch 0 on SYCL is an explicit C4 item** (addendum I-9, m-6).
  `init_mappings` is called with prefetch 0 on a SYCL load (`lm:2702`), and
  `llama_mmap` excludes the record's deferred ranges with the both-edge
  `MADV_RANDOM` window and the 2 MiB-widened `NOHUGEPAGE` window, whatever
  the prefetch value (r2 I-4).
- Host-only REDs:
  - `test-llama-mmap-lazy-exclusion` (`host;lazy`), §4.6's `llama_mmap`
    gates:
    - **The prefetch arm.** Under `prefetch = -1` there is no `WILLNEED` over
      a lazy range and no read-around into one. The mutant is an empty record
      (lazy ranges from a diverted `lazy_read`), which must read resident.
    - **The `HUGEPAGE` arm** (r2 I-4). Lazy boundaries sit at 2 MiB + 1 MiB
      offsets. Faulting the page adjacent to each edge leaves the folio band
      at 0. The mutant, with the window left `HUGEPAGE`, must read it
      resident, or the arm exits 77 naming why.
  - `test-sycl-host-role-routing-source` (`sycl;host;source-contract`, r2
    I-2, addendum 7):
    - It regenerates §4.4's **site table** from the tree, and fails on any
      difference: each row's holder, role, zone after C4, lifetime and owner.
      23mk's class-H rows cite that table, so the gate is its one source.
    - Every non-`GRAPH` row names an owner. A request site carrying
      `UNOWNED` that is not one of the table's 12 lines fails, as a ratchet,
      and the message cites llama.cpp-30r1 (addendum 10 (b)). A `GRAPH` row
      with no plan term is listed `UNSIZED` and ratcheted the same way (r3
      m4).
    - The three `DEVICE` holders' requests set `lifetime = DEVICE`, and
      `ggml_sycl_shutdown_global_runtime_pinned_owners` still releases each
      of them (`gs:7575-7590`).
    - `select_zone` has no category clause for WEIGHT, and sends `UNOWNED`
      to WEIGHT.
    - `SYCL_Host` takes role `WEIGHT` only under `in_model_load`.
      `SYCL_KV_Host` has its own `alloc_buffer`, with role `KV`, and keeps
      its own `get_name` (r3 B).
    - Each of `SYCL_Host`, `SYCL_KV_Host` and `SYCL_HostCompute` builds its
      intent in one function that its `alloc_buffer` and its `get_max_size`
      both call, and `get_max_size` passes that intent to `select_zone`; no
      `get_max_size` names a zone. This amends
      `tests/test-sycl-host-zone-config-source-contract.py`.
    - `llama_context` calls `ggml_backend_sycl_notify_host_buffer_sizes`
      through the registry's proc address, and nothing in `src/` calls
      `ggml_backend_sycl_notify_compute_buffer_sizes` (its VRAM half is not
      5efe's).
    - **The notify is ordered before the allocation it covers** (r3 A.2):
      in `output_reserve`, between the free of the old buffer and
      `ggml_backend_buft_alloc_buffer`; in `sched_reserve`, after the
      size-only `graph_reserve(..., true, sizes)` calls and before the first
      allocating one. It is called with one backend, `model.devices[0]`'s,
      not in a loop over backends (r3 A.5).
    - `ggml_backend_sycl_notify_host_buffer_sizes` tests
      `ggml_backend_is_sycl(backend)` before it reads `backend->context`
      (r3 A.4).
    - `runtime_bytes` in `ggml_sycl_configure_host_zones_for_plan` names
      neither `pp_pipeline_scratch_bytes` nor `pp_moe_onednn_scratch_bytes`
      (r3 m3).
    - The inventory's `kv_idx_width` is declared after `kv_layer_count`,
      and both demotion sites read `kv_demand_for_layer` (r3 B).
    - The new entry point's parameters are scalars: no pointer and no struct
      in its signature, in either the header or the `llama_context` function
      pointer type (the dlopen shape).
    - Host SCRATCH's fold at `uc:27727-27730` calls
      `unified_cache_get_planned_runtime_zone_requirement`. It reads none of
      `moe_vram_runtime_bytes`, `pp_pipeline_scratch_bytes` and
      `pp_moe_onednn_scratch_bytes`, and runs after their setters.
    - The SCRATCH boundary check counts only `GRAPH` records.
    - **Mutants:**
      - the category clause restored, which moves the `CpuExpertPool` row to
        WEIGHT;
      - `moe_vram_runtime_bytes` added back beside the function (a copied
        term);
      - a new file-scope static holder added at a `HOST_COMPUTE` site (a new
        `UNOWNED` row);
      - the notify moved after `ggml_backend_buft_alloc_buffer` in
        `output_reserve` (r3 A.2);
      - `SYCL_Host`'s `get_max_size` with WEIGHT hard-coded again;
      - `pp_moe_onednn_scratch_bytes` restored in `runtime_bytes` (r3 m3).
  - `test-sycl-host-zone-select` (`sycl;host`). It is a pure test of the
    extracted `select_zone`, over role x category x `kv_spill_to_host`, and
    of the lifetime checks over a fake registry.
    - It covers every row of §4.4's routing.
    - The graph-boundary check: a `CONTEXT`, a `MODEL`, a `DEVICE` and an
      `UNOWNED` record pass it, and a `GRAPH` record trips it.
    - Role `KV` from the `SYCL_KV_Host` intent selects zone 1, and the
      post-load `SYCL_Host` intent selects zone 3.
    - **The owner-boundary checks** (addendum 7 (b)):
      - a `CONTEXT` record live at its context's destruction prints the
        `[HOST-ZONE] leaked CONTEXT record` WARN, naming the cohort, the
        bytes and the owner. Under `GGML_SYCL_STRICT_LEASES=1` it aborts, in
        its own process, scored on rc 134 plus that text;
      - a `MODEL` record live at its model's free does the same;
      - a `DEVICE` record live at its cache's teardown does the same
        (addendum 10 (b));
      - a host-buffer `CONTEXT` record stamped with a context serial that is
        still live at the device cache's teardown does the same, naming the
        serial;
      - a record owned by a **different** context or model does not fire,
        which is the multi-model case.
    - Mutant: the owner check keyed by lifetime alone, which fires on
      another context's live record.
  - `test-sycl-host-scratch-demand` (`sycl;host`, r3 A). A pure test of the
    demand ledger behind `ggml_backend_sycl_notify_host_buffer_sizes`, over a
    fake zone that records `grow_zone` calls:
    - a context's second notify replaces its first: (100, 10) then (300, 10)
      grows for 310 in total, not 410;
    - two contexts sum, and one `grow_zone` call is made per notify whose
      sum exceeds `grown_for_demands`; a context's demand dropped at its
      destructor does not shrink the zone, and a later context of the same
      size grows nothing;
    - the request is the deficit plus one `DEFAULT_ALIGNMENT` per buffer at
      the chunk cap, and a `DEVICE` holder's demand is on the same ledger;
    - a non-SYCL backend is refused with the WARN and records nothing.
    - **Mutants:** accumulate instead of replace (the r3 A.2 double
      growth); one grow per SYCL backend (the r3 A.5 N-times growth).
  - `test-sycl-kv-demand-idx` (`sycl;host`, r3 B). A pure test of
    `kv_demand_for_layer` on a 48-layer plan with Qwen's shape: 12 FULL
    layers of 512 MiB at 262144 cells, `kv_idx_width` 128. Each carries 576
    MiB, and a 0 width gives 512. Fed to `plan_runtime_kv_demotion` with 287.8
    MB of room, it demotes 12 layers, 3..47, with 6912.0 MB of host KV.
    Mutant: the demotion input reading `kv_size_for_layer`, which prints
    6144.0.
  - `test-sycl-getrows-refusal-terminal` (`sycl;host`, a fake `stream_dma`,
    r2 m-5). One refusal gives one call, one counted refusal, no
    `[GET_ROWS] DMA retry` line, and the CPU fallback.
  - `test-sycl-mapped-loader-source` (`sycl;host;source-contract`):
    - prefetch 0 at the SYCL `init_mappings` call;
    - every site of the §4.3 table, including the preload copy (m-7) and
      `alloc_buffer` (m-10);
    - no `use_mmap` read in `create_tensor`'s SYCL buft choice;
    - `lm:2693` narrowed to the no-tier case;
    - the eight `weight_host_bytes` accumulation sites test `host_backing`,
      and the readers of §4.4 read their assigned meaning (m-3);
    - `dma_streamed` is 0 without `PINNED` host weights (m-2);
    - the eight `resolve_or_host` callers are the pinned list (§5.3);
    - the four `stream_dma` callers route `refused_file_mapped` explicitly.
  - `test-sycl-host-backing-plan`, a Qwen-shaped inventory (1223 tensors,
    B70 budget):
    - with `load_maps_files` true: host experts are `FILE_MAPPED`,
      `weight_host_bytes` = 0, `host_mapped_bytes` = 102383.5 MB, the WEIGHT
      zone is 64 MiB, the DMA pool is 0, the runtime pre-allocate is 91.9
      MB, and base + runtime = 4 + 1 chunks (r3 m3);
    - with it false: today's 122924.2 MB zone and 1700 MB DMA pool;
    - with mlock: the mapped bytes are charged.

    It is host-only through 7v5c's dry-run entry if 7v5c has landed.
    Otherwise it is registered `sycl;cache;mem-handle`, `level_zero:1`, and is
    lead-run.
- Device REDs (`sycl;cache;mem-handle`, `level_zero:1`, lead-run):
  - `test-sycl-file-mapped-stream-refusal`: `stream_dma` over a `FILE_MAPPED`
    view refuses, with and without a `copy_fn`, and the counter counts.
    Mutant: the check placed after the `copy_fn` test.
  - `test-sycl-file-mapped-bounce`: `mem_copy` of a mapped source takes the
    bounce (`host_ptr_is_usm_accessible` false). The outside-materialization
    counter stays 0 inside a materialization, and counts outside one at each
    named chokepoint (m-17).
  - `test-sycl-file-mapped-expert-owner`: `register_host_expert` over a mapped
    slice stores the owner, and the mapping outlives the buffer while an entry
    holds it.
  - `test-sycl-file-mapped-resolve-abort`, a death test in its own process:
    `resolve_or_host` on a mapped weight aborts naming the tensor. It is
    scored on rc 134 plus the tensor name.
  - **`test-sycl-pinned-meminfo-control`** (r2 m-13; the lead runs it before
    step 2 below):
    - It allocates 4 GiB through `unified_allocate` with `must_host_pinned`,
      touches every page, samples `/proc/meminfo` before and after, and frees
      it.
    - It prints each counter's delta: `AnonPages`, `Shmem`, `Mlocked`,
      `Unevictable`, `Cached`.
    - Pre-registered: at least one counter moves by ≥ 3.5 GiB, and that
      counter is the one pinned host USM lands in. run2's `Shmem` stayed at
      1.8 G while 140 GB was pinned, so it is not `Shmem`.
    - If no counter moves, the test exits 77, and the host-growth criterion
      below rests on the `Committed` line plus `MemAvailable` alone.
- Lead, in this order (r1 I-8, m-14, m-15):
  1. `POSIX_FADV_DONTNEED` over shard 3, then `fincore` on it. Pre-registered:
     ≤ 1 MiB resident.
  2. **Qwen3.8 Q8_0 on `level_zero:0`, default load flags, `-c 4096`, the
     baseline completion**: `llama-completion -m <Qwen3.8 Q8_0> -no-cnv -v
     -c 4096 -p 'The capital of France is' -n 32 --seed 42 --temp 0`, the
     run2 vehicle (r3 m2). `-v` is required: the zone, pre-size and demand
     criteria below read INFO lines, which `llama-completion` drops at its
     default threshold (§7 preamble).

     **Precondition** (§3.1, the C1-to-T6 window; accepted in addendum 8 (b)). th32
     T6 and T7 are on the tree, so the PLE table is taken by T6 and gathered
     by T7. Without T6, a taker load reads the table eagerly and the early
     plan never counted it. With T6 but not T7, nothing gathers it. Either
     tree makes steps 2 and 3 VOID, not FAIL, and the lead is told which
     tree it was.

     Pre-registered pass criteria:
     - the load completes and prints coherent text;
     - `[HOST-ARENA] configured zones: WEIGHT=64.0 MB`;
     - `Pinned pool pre_allocate: grew 4 chunks`,
       `pre_allocate_runtime: grew 1 chunks`, and `Committed 10240.0 MB in 5
       chunk(s)` (§4.4; r2 I-1, r3 m3). A reading of 6, 7 or 8 chunks names
       what did not land, by §4.4's table;
     - `DMA staging pool: 0.0 MB`;
     - **no `Pinned pool grow_zone: zone 0` line after the pre-size**, through
       the end of the completion (r2 I-2). This absence counts only because
       the same run printed the `Pinned pool pre_allocate: grew` line above,
       its positive control at the same INFO level (r3 m2);
     - **SCRATCH is scored by chunks** (r3 A.3). Every `Pinned pool
       grow_zone: zone 3` line follows a `[HOST-ARENA] SCRATCH demand` line
       whose `grown` is positive, and reads `max(2048.0, align_up(deficit +
       n x alignment))` MB for that line's deficit. A zone-3 grow with no
       demand line before it is a miss, and FAILs naming the tenant's
       cohort. The demand lines print the compute and output figures; they
       are read from the run, not pre-registered, since run2 was killed
       before its scheduler reserve;
     - no `host compute buffer ... exceeds notified` WARN (r3 A.2);
     - **no zone-1 growth**: at `-c 4096` nothing is demoted, so `Pinned pool
       grow_zone: zone 1` is absent (with the same positive control);
     - the early and late `[PLACEMENT] Totals` both print charged host 0 and
       mapped 102383.5 MB. The early `Host staging zone` has `per_inflight`
       ≤ 1800 MB;
     - "planner host placement: yes" (m-3);
     - `SYCL_Mapped` model buffer lines sum to 127615 MiB ± 1%;
     - `SYCL_Host` model buffer lines hold 0 weight bytes;
     - peak growth of the counter the control named, over the pre-load
       reading, is < 30 GiB **plus the lazy table's own bytes**, read from
       th32's `[PLACEMENT] lazy per_layer_token_embd.weight` line: its
       `resident=` figure under `residency=HOST` (51880.0 MiB, locked, th32
       §8's expected line), or its `cache=` figure under `residency=STORAGE`.
       Those bytes are th32's, not 5efe's, and the term is taken from the
       line, never assumed. A missing line is a FAIL of the precondition.
       The criterion is skipped, and said to be skipped, if the control
       exited 77;
     - `MemAvailable` never falls under 25 GiB (the watchdog unit);
     - `Shmem` stays under 5 GiB;
     - `fincore` on shard 3 after the run is ≤ 256 MiB. The bound: the prompt
       plus 32 tokens is under 40 tokens, x 16 PLE rows per token (th32 §0
       table), x a 256 KiB bcachefs extent per row fault = 160 MiB, rounded
       up;
     - the `[FILE-MAPPED] counters:` line prints `stream_dma_refusals=0
       device_copies_outside_materialization=0` (m-18);
     - no GPU fault lines.

     There is deliberately no page-cache-growth criterion, because it would
     depend on the other shards' prior cache state.
  3. **The same `llama-completion` Qwen run with no `-c`** (the
     default-context arm), at `n_ctx = n_ctx_train = 262144`
     (`run2.log:131`). The same precondition applies.
     - **The KV.** 12 of the 48 layers carry KV: qwen4exp marks layer `i`
       recurrent unless `(i + 1) % full_attention_interval == 0`
       (`src/models/qwen4exp.cpp:132-134`), and the interval is 4
       (`run2.log:37`). So layers 3, 7, ..., 47 carry it. Each holds
       262144 x (512 + 512) x 2 B = 512 MiB (`run2.log:146-147`). That is
       **6144 MiB** of main KV.
     - **The indexer** (r3 B). The `idx_` cache covers the same 12 layers
       (`lm:3580-3585`), K only, `indexer_head_size` 128: 262144 x 128 x 2 B
       = 64 MiB per layer, **768 MiB** in all. After C4 it is in the demand
       (§4.4), so each KV layer is 576 MiB, and **6912 MiB** in all. The
       recurrent state is device-resident and not in the figure (§4.4).
     - The SYCL planner's KV sizing counts all 48 layers (`run2.log:217`,
       `:307`). At 262144 that would claim 24576 MiB. That over-count is
       **llama.cpp-0edf**, outside 5efe.
     - **The room.** The demotion's capacity is `min(budget, shared zone)`
       (`kv_weight_capacity`, `kv-runtime-demotion.cpp:58`). In run2 the shared
       zone was 25519.0 MB (`run2.log:227`) and device weights 25231.2 MB
       (`:308`), with optional layouts 0.0. So the room is at most 287.8 MB,
       less than one 576 MiB layer, and no layer that carries KV stays on the
       device.
     - Demotion walks layers latest first, and skips layers with 0 recorded
       bytes (`kv-runtime-demotion.cpp:19-45`).
     - **Two pre-registered outcomes, and which one prints is 0edf's
       discriminator:**
       - real hybrid sizing: `12 layer(s) demoted, 0 SWA (6912.0 MB host KV,
         layers 3..47)`. **PASS.**
       - 0edf's over-claim: `48 layer(s) demoted, 0 SWA (27648.0 MB host KV,
         layers 0..47)`, 48 x 576. The indexer term follows the planner's
         KV-layer set, so it scales with the over-claim. **This is a FAIL**
         (r3 n1): the plan then prints 27648 MiB of host KV while the load
         pins 6912, which is plan != reality (P4). It is attributed to
         llama.cpp-0edf and is not a 5efe regression; whether it blocks the
         merge is the lead's call.
       - Without C4's indexer term, the two lines would read 6144.0 and
         24576.0. Either figure is a FAIL of the r3 B change.

       The MB figures are MiB, the unit the existing demotion lines print (29
       Mistral layers at 32768 print 3712.0).
     - **General rule, if the printed `F` differs.** With `K = floor(F /
       576)`, real sizing gives `N = 12 - K`, layers `3 + 4K..47`, and
       `N x 576` MB; the over-claim gives `N = 48 - K`, layers `K..47`. Any
       other N, range or figure is a FAIL.
     - **The criteria:**
       - every criterion of step 2 that does not depend on `n_ctx` holds;
       - **the KV zone (zone 1) must grow, scored by chunks** (r3 B). The
         main KV reaches `SYCL_KV_Host` as three 2048 MiB buffers and the
         indexer as one 768 MiB buffer (§4.4, the chunk cap), and none fits
         the 64 MiB base. So the run prints **four** `Pinned pool grow_zone:
         zone 1 grown by 2048.0 MB (1 new chunks, ...)` lines at context
         init, 8192.0 MB in all, which covers the printed 6912.0. Fewer lines,
         any zone-3 growth that is not a demand line, or a `SYCL_KV_Host`
         buffer outside zone 1 is a FAIL. The arm sets
         `GGML_SYCL_HOST_ZONE_ALLOC_TRACE=1` (`uc:3990`), whose per-allocation
         line names the zone;
       - no `[HOST-ZONE] leaked` line at context free;
       - `MemAvailable` stays above 25 GiB. The load pins 8 GiB of zone-1
         chunks on top of step 2's 10 GiB, and `MemAvailable` before the
         load was 155.0 GiB (`run2.mem`);
       - a `runtime KV update rejected` line is a FAIL.
  4. The Mistral gate on `:1` and `:0` (default context).
  5. The GPT-OSS gate on `:1`, in two arms, each under `timeout 60`, **with
     `-lv 4` added** to the canonical command (r3 C; §7 preamble), stdout and
     stderr captured to separate files, and scored with `grep -cx '1, 2, 3,
     4, 5'` = 1 on the stdout file:
     - **the canonical CLAUDE.md `llama-cli` command with `-c 4096`**;
     - **the same with no `-c`**, at `n_ctx = 131072`.
       - **The demotion line is zhcn's GDC3 row, cited and not copied**
         (§M56 (a)). GDC3 is the single source of N, the host-KV figure and
         the layer range, and of the replay point that scores them. If GDC3
         changes, this arm changes with it and nothing here is edited.
       - The line is WARN (`gs:18382`), so without `-lv` (or with `-lv` below
         2) `llama-cli` drops it and the arm cannot pass on a correct build
         (r3 C).
       - **Zone 1 grows**, one `Pinned pool grow_zone: zone 1 grown by
         2048.0 MB` line per host-KV buffer: GPT-OSS's full-attention K and
         V tensors are 128 MiB each at 131072 cells, so they pack the 2 GiB
         chunk cap exactly, and the count is `ceil(M / 2048)` for GDC3's
         host-KV figure `M`. No zone-3 miss, and no `[HOST-ZONE] leaked`
         line.
       - A refusal is a FAIL.

     Every SYCL load now takes the mapped path.
  6. `llama-bench` Mistral Q4_0 and GPT-OSS 20B on both cards, interleaved
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
  - **The override composes** (r2 m-4). With `GGML_SYCL_HOST_RESERVE_MB` above
    the formula's value, the budget is the formula's. Below it, the budget is
    the override. At zero headroom-free memory it is the named refusal
    either way.
  - Today's formula fails all three: the (246.9, 28.9) row gives 20.9, the
    sweep drops at the `available/2` cliff, and the override replaces the
    budget outright.
- RED `test-sycl-host-working-set-warn` (`sycl;host`).
  - The WARN is printed with its figures at `resident_room < mapped_ws`, not
    above it, and the plan-report line carries the same figures.
  - `resident_room` uses `host_headroom(total)`. At (62 GiB, 50 GiB
    available, 20 GiB mapped) it does not WARN, where the P3 constant would.
- It depends on th32 T2 for `host_ledger_budget()` if T2 has landed.
  Otherwise it edits `uc:4165-4231`, and T2 lifts the result.
- Lead (r3 E). C5 changes the host budget that uize-demoted host KV is
  charged against, so it can move KV placement, and every arm that places
  KV runs at the default context too:
  - the C4 Qwen `-c 4096` run once more (C4 step 2's command, `-v`),
    reading `[PLACEMENT] Totals` (the four terms) and the working-set line
    against the `/proc/meminfo` samples;
  - **the C4 step-3 Qwen no-`-c` arm**, with its pre-registered line (12
    layers, 6912.0 MB, layers 3..47) and its zone-1 criterion. A different
    N under C5's budget is a FAIL of C5, unless the arm prints the budget
    change that explains it and the general rule of step 3 predicts the
    printed N from it;
  - **the C4 step-5 GPT-OSS no-`-c` arm**, with `-lv 4`, scored on zhcn
    GDC3's line (§M56 (a)) and the digit count on stdout.
  - The same precondition as C4 steps 2 and 3 applies to both Qwen arms.

**C6. Retire unowned host registration** (§4.8).
- RED `test-sycl-unregistered-host-refusal` (`sycl;cache;mem-handle`,
  `level_zero:1`, lead-run). `register_host_expert` and `register_host_weight`
  with a `malloc`ed (`UNREGISTERED`) pointer are refused. Today they succeed as
  `HOST_MMAP`, `non_owning_external_host`.
- Lead:
  - the Mistral gate (default context);
  - the GPT-OSS gate in both arms, `-c 4096` and no `-c`, under
    `timeout 60`, with `-lv 4` and the stdout-only digit scoring (§7
    preamble), and the no-`-c` arm scored on zhcn GDC3's line;
  - the C4 Qwen run in both arms.

  The refusal must never fire on a real load.

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

Rows marked "superseded in part" did not hold as written at rev 3a (design-5efe-r2); Appendix B re-disposes them.

| finding | disposition | where |
|---|---|---|
| C-1 C4 acceptance ordered before the ledger split | **Superseded in part by r2 I-1 (pre-size figure) and I-2 (runtime term withdrawn); see Appendix B.** Fixed per §F2: the split and the WEIGHT-zone sizing that reads it move into C4, one commit with the loader and no C4-only exclusion. Qwen's WEIGHT zone is pre-registered at 64.0-128.0 MB, the pre-size at ≤ 12288 MB, with a falsifier. The non-weight `SYCL_Host` runtime term is named. | §4.4, §7 C4 |
| I-1 mincore gate cannot measure the property | **Superseded in part by r2 I-3 (read-around arms re-scoped to the edge band); see Appendix B.** Fixed per §F2: `cachestat(2)` on the fd, or `mincore` on a test-owned mapping; never the tier's hole. Synchronous `pread` positive control, must-fail mutant arm, 24 MiB hole with a 1 MiB guard band, and an mlock arm. | §4.6, §7 C3 |
| I-2 multi-segment shape under-specified | Fixed per §F2: one `PROT_NONE` reservation with `MAP_FIXED` segments, one owner, one record, one `munmap`. Map segments round outward; `WILLNEED` and lock segments round inward. Bound: at most one lazy page mapped per edge. | §4.1, §4.3, §4.6 |
| I-3 stream_dma disposition legitimized streaming | Fixed per §F2: a typed, counted refusal checked before `copy_fn`, routed to the CPU executor. The counter counts any device-bound copy of mapped bytes outside a materialization. The RED asserts the refusal. | §5.3, §7 C4 |
| I-4 headroom cap universal and discontinuous | Fixed per §F2: `max(total/32, min(29 GiB, total/4))`, one named P3 constant cited as this host's measurement, the `available/2` cliff removed in favour of a named refusal, and 32 and 62 GiB cases plus a monotonicity sweep. | §4.5, §7 C5 |
| I-5 plan and loader derive "mapped" separately, and in the wrong order | Fixed per §F2: the caps are resolved before the early plan, and one `load_maps_files` fact is published. The planner's `host_backing` is read by the loader through a hook; `create_tensor` never reads `use_mmap` for SYCL weights. | §2.1, §4.3, §4.4 |
| I-6 host-only REDs need a device; no labels | Fixed per §F2: every test is named with its registration and label. Device tests are `sycl;cache;mem-handle`, `level_zero:1`, lead-run. Host tests link a device-free TU and never touch `stream_dma` or a device `mem_copy`. | §4.1, §7 |
| I-7 C1 witness re-runs the near-OOM | **Superseded in part by r2 m-1 (margin is 4.2 s, not 53 s); see Appendix B.** Fixed per §F2: the witness is killed on the two log lines (t ≈ 3.9 s, before the pre-size). Sampling every 1 s; floor and criteria both 25 GiB = 26214400 kB. | §7 |
| I-8 two C4 criteria fail on a correct build | Fixed per §F2: the page-cache growth criterion is dropped. Shard 3 gets a pre-registered bound (≤ 1 MiB before, ≤ 256 MiB after). `SYCL_Mapped` is expected at 127615 MiB ± 1%, all SYCL host-sourced weights. | §4.3, §7 C4 |
| m-1 "pinned 110 GB" | Fixed: 139264 MB committed, 140.0 GB reached, `MemAvailable` fell 141 GiB. | §0 |
| m-2 expert figures | Fixed: 512 experts, 10 used, about 1.7 MB per matrix, about 2.4 GB per cold token, 1-6 ms per matrix. | §5.2 |
| m-3 census misses tests | Fixed: `tests/test-cpu-gpu-soa-interaction.cpp:178/180/182` added, and the census gate covers `tests/`. | §4.8, §7 C2 |
| m-4 `alloc_type::MMAP` understated | **Superseded in part by r2 m-11 (`HOST_NON_USM` replaced by `HOST_MALLOC` and `HOST_OWNED_ELSEWHERE`); see Appendix B.** Fixed: ten consumers listed, the `::free` meaning recorded, renamed `HOST_NON_USM`, and a `FILE_MAPPED` branch ordered before `::free` with a counter test. | §2.7, §4.1, §4.8 |
| m-5 fd lifetime and replacement gap | Fixed: `dup` of the loader's fd, kept by the owner for `FADV_DONTNEED`; no reopen by path. | §4.1, §4.6 |
| m-6 loader sites under-enumerated | **Superseded in part by r2 m-8 (C4 amends the C1 caps test); see Appendix B.** Fixed: the §4.3 site table (hooks, `weight_buft_supported`, `make_cpu_buft_list`, `mlock_bufs`, the `init_mappings` prefetch) and the loader `lmlock` skip. | §4.2, §4.3, §5.1 |
| m-7 null path unverified; AOS branch | Fixed: `resolve_or_host` aborts naming the tensor, with the eight callers audited and pinned, and `binbcast.cpp:394/794` cited. Both layout-availability branches are excluded for `SYCL_Mapped`. | §5.3 |
| m-8 stale th32 pin | **Superseded in part by r2 m-15 (re-pinned at rev 2u `0bd69e251`); see Appendix B.** Fixed per §F2: pinned at the latest reviewed head, rev 2q `1646da864` (see the lead-check row below); 2s `f364952d3` is noted as the revision that reads the record. Rulings relabelled §T13/§T13a/§T13c. | header, §4.6 |
| m-9 F1 fold and cold set | Fixed per §F2: the yl3v/7v5c interface is stated, the recommendation removed, and the terms yl3v must read are named so that DeepSeek's cold set is not implicitly empty. | §4.5, §8 |
| m-10 enum value collision | Fixed per §F2: the next free value at landing; censuses key on names. | §4.1, §4.8 |
| m-11 allowlist unenforced | **Superseded in part by r2 m-12 (the gate tokenizes); see Appendix B.** Fixed: `test-sycl-raw-mmap-allowlist-source` with today's eight sites, shared with th32 T1. | §4.1, §7 C2 |
| m-12 C2 RED weak | **Superseded in part by r2 m-6 (the rev 3a mutant was equivalent); see Appendix B.** Fixed: named mutants plus a private `munmap` counter and a registry query; no `mincore`-ENOMEM check. | §7 C2 |
| m-13 staging faults can populate lazy pages | Fixed: `MADV_RANDOM` on segment edges bordering lazy ranges covers execution and bounce faults; the gate reads every byte of A and B, edges included. | §4.6 |
| m-14 GPT-OSS gate lacks `timeout 60` | Fixed per §F2: the canonical command with `-c 4096` under `timeout 60`, scored with `grep -cx`. | §7 C4, C6 |
| m-15 shard-3 "~0" not pre-registered | Fixed: ≤ 1 MiB before, ≤ 256 MiB after, with the arithmetic. | §7 C4 |
| m-16 64 MiB claim too broad | Fixed: scoped to `mem_copy`; the reorder, WOQ and XMX fill paths are listed as unchanged and bounded per source tensor. | §5.4 |
| m-17 host_unified fails open | Fixed: `host_unified_known` added, and `mmap_support` requires it. | §3 |
| m-18 storage_owner type; PER_DEVICE teardown | Fixed: `storage_owner` wraps the reservation's `mem_handle`; every model unloads on every device before any cache tears down. | §4.2 |
| addendum I-2(a) §4.6 text wrong for map segments | Fixed per §F2a: already corrected at `1ddbd9d56` (map segments outward); rev 3 states it at the table, adds the `MADV_RANDOM` edge windows as an inward set, and says that only map segments round outward. | §4.6 |
| addendum I-9 lazy_ranges source disappears after th32 T6 | Fixed per §F2a and §M49: one per-file lazy-range record, owned by 5efe (filled before the early plan from the arch declaration, per addendum 2 below), landing in C1. `lazy_read`, th32's inventory, the tier and `llama_mmap` consume it and keep no copy. Prefetch 0 on SYCL is an explicit C4 item, with RED `test-llama-mmap-lazy-exclusion` (its mutant is an empty record). | §3.1, §4.6, §7 C1, C4 |
| addendum I-10 tier and loader inherit the prefix lock | Fixed per §F2a: the tier `mlock(2)`s inward-rounded host-executed segments directly; `SYCL_Mapped` is excluded from `mlock_bufs`; §2.5 is corrected; `lml:2122` becomes a ranged `lock_range` in 5efe (C3). Host-only RED `test-llama-mlock-ranged` with a lazy range before a locked range and a whole-prefix mutant; the tier's arm 2 covers the tier side. | §2.5, §4.6, §5.1, §7 C3 |
| addendum m-19 one-sided edge advice; residual undercounted | Fixed per §F2a: `MADV_RANDOM` windows on both edges, because read-around is centred on the fault. The loader's whole-file `llama_mmap` read-around is named as a second residual path, given the same record-fed window (C4), and bounded at one shared page per edge per mapping, plus bcachefs extent granularity inside the guard band. | §4.6 |
| addendum m-20 readahead case could pass vacuously | Fixed per §F2a: arms 5 (forward) and 6 (backward) fault real mapped pages, exit 77 when `read_ahead_kb` is 0, each has a no-advice control twin that must populate the hole, and both are measured with `cachestat` or a test-owned mapping. | §4.6, §7 C3 |
| §F2a I-9 addendum 2: early plan cannot see the record | **Superseded in part by r2 I-5 (declaration keyed by GGUF name, no hparams; static gate) and I-6 (STORAGE writer); see Appendix B.** Fixed: the flag is first knowable only at `create_tensor`, so C1 adds a per-arch lazy-tensor declaration and pre-fills the record from it plus `weights_map`, after the caps and lazy-mode loops move above the early plan. The early plan skips deferred rows; `create_tensor` checks the record rather than filling it, and aborts on disagreement. C-1's numbers are re-checked and unchanged, because the zones come from the late plan. Early-plan witness lines are pre-registered in C1 and C4. | §3.1, §2.6, §4.3, §4.4, §7 C1, C4 |
| §F2a I-9 addendum 2 (M-1): by-value hand-off | Done: `value(name)` returns `{file_idx, offs, size}` by value, rows are immutable after the pre-fill, and deriving the numbers independently (from `w->offs`) is named as forbidden. | §3.1 |
| lead check: th32 pin | **Superseded in part by r2 m-15 (re-pinned at rev 2u `0bd69e251`); see Appendix B.** Pinned at rev 2q `1646da864` per the lead; 2s `f364952d3` is named as the record's consumer, and §3.1's row fields match its §4.1. | header |
| §M49 note: record in C1 | Done: the record and its two host-only tests are C1 deliverables, so th32 T6 depends on C1 alone. | §3.1, §7 C1 |
| th32 r15 m-4 (relayed) mlock prefix | Fixed: the tier's own per-segment `mlock` (named, and where), no `llama_mlock` for `SYCL_Mapped`, a residency arm under mlock, and a RED with a lazy range before a locked tensor. | §4.6, §5.1, §7 C3 |

## Appendix B. Disposition of design-5efe-r2 (rev 3a `3763bb164`, 0C/6I/19m), with held rulings

| finding | disposition | where |
|---|---|---|
| I-1 pre-size omits the runtime pre-allocate | Fixed: the criterion is split, and both numbers are pre-registered from the configure path. The base is 4 chunks and the runtime is 3 chunks, so the pre-registered line was `Committed 14336.0 MB in 7 chunk(s)`. **Superseded in rev 3d by r3 m3:** the runtime pool drops the two PP terms, so it is 1 chunk and the line is `Committed 10240.0 MB in 5 chunk(s)`; §4.4's table maps 6, 7 and 8 chunks to what did not land. | §0, §4.4, §7 C4 |
| I-2 WEIGHT has non-weight tenants; the runtime term has no path | Fixed per the lead's rulings on I-2 and addendum 7 (b)/(c). 5efe routes host-side non-weight requests by role: HOST_COMPUTE and EXPERT_CACHE go to SCRATCH. The rev 3a "runtime term" is withdrawn. The criterion is **no `Pinned pool grow_zone: zone 0` line after the pre-size**. **Sizing:** `ggml_backend_sycl_notify_compute_buffer_sizes` (gs:19103-19146) has no caller, and it also reserves the VRAM arena and sums device buffers into host SCRATCH. So it stays uncalled, and C4 adds the host-only, scalar-ABI `ggml_backend_sycl_notify_host_buffer_sizes(backend, host_compute_bytes, host_output_bytes)`, reached through the proc address; a stale backend yields NULL and a WARN. Whether the old function's VRAM half should get a caller is reported to the lead. **Lifetime** (confirmed over a fifth zone): `alloc_intent::lifetime` {`GRAPH`, `CONTEXT`, `MODEL`}, with no `PROCESS`. The graph-boundary check counts `GRAPH` only. `CONTEXT` is leak-checked at backend-context destruction and `MODEL` at model free, each a WARN that aborts under STRICT_LEASES. **The site table** (35 lines) is published in §4.4, with a holder, role, zone, lifetime and owner per row: 10 `GRAPH`, 4 `CONTEXT`, 1 caller-chosen, 2 non-tenants, and **14 holders (18 lines) UNOWNED**. The UNOWNED rows are file-scope statics and `thread_local`s, including the `CpuExpertPool` and `PinnedBufferPool` rings on Qwen's default path, and are brought to the lead. **23mk:** 5efe owns role, zone, lifetime and sizing, and 23mk owns conversion and cites the table. Host SCRATCH reads `unified_cache_get_planned_runtime_zone_requirement` in place of three copied terms. | §4.4, §7 C4 |
| I-3 arms 5 and 6 cannot reach the hole interior | Fixed: read-around is centred on the fault, with reach `ra_pages/2`, doubled under `FADV_SEQUENTIAL`, where `ra_pages` comes from the bdi's `read_ahead_kb`. Arms 5 and 6 measure the **edge band** `[shared page + 1 page, edge + reach)` at each lazy edge (§4.6). `W = max(2 x read_ahead_kb, 2 MiB)` is a different quantity, the width of the `MADV_RANDOM` window C4 applies at both edges; rev 3c's row gave that name to the band (r3 m5). The interior is measured by arms 1 and 2 only. Each arm exits 77 if the computed reach does not extend two pages past the shared page. | §4.6, §7 C3 |
| I-4 `MADV_HUGEPAGE` defeats the edge windows | Fixed: `do_sync_mmap_readahead` tests `VM_HUGEPAGE` before `VM_RAND_READ`. So C4 applies `MADV_NOHUGEPAGE` to the `MADV_RANDOM` windows widened to 2 MiB, inside `llama_mmap`'s `HUGEPAGE` advice (`src/llama-mmap.cpp:517-532`). The one-page bound is restated per mapping. A faulting arm with lazy boundaries at 2 MiB + 1 MiB offsets gets a `HUGEPAGE` control mutant. | §4.6, §7 C4 |
| I-5 the declaration needs a static cross-check | Fixed per the lead ruling, as amended by addenda 5, 7 (a) and 8 (a). The declaration is 5efe's, and th32 consumes it. The final signatures (addendum 8 (a): rev 3c's stand, and th32 does not rename) are `const std::vector<std::string> & llama_model_lazy_tensor_names(llm_arch)`, keyed by (arch, exact name or one `%d`), names only; `struct llama_lazy_tensor_attrs { uint32_t rows_per_token; int32_t consumer_layer; }`; and `llama_model_lazy_tensor_attrs(arch, hparams, name)`, 5efe's. C1's pre-fill stores the attributes on the per-model record entry, and T6 reads `record.attrs(name)`. There is no static table: one that T6 filled would be shared across models (P4). An `LLM_TENSOR_INFOS` column was rejected, and the arch key is required, because `gemma3n.cpp:39` creates the same tensor unflagged. `test-llama-lazy-declaration-source` maps (arch, call site) to entry and back, with four mutants plus a static-attribute-cache mutant. The record test has a two-hparams P4 witness. The runtime abort is kept and is mode-independent. | §3.1, §7 C1 |
| I-6 no writer for `STORAGE`; eligibility superseded by addendum 3 | Fixed by folding addenda 3 to 6. The pre-fill writes every field once. `lazy_eligible = T6a(taker ? mode_pre : mode_post, size, taker ? true : SUPPORTED)`. `deferral` is `STORAGE` iff eligible and taker, `LAZY_READ` iff eligible and not taker, and `EAGER` otherwise, so `deferral != EAGER` iff eligible. `taker` is `sycl_model_loading_guard.active` (`lm:2228-2230`), not `hooks_enabled()` (rev 3b's `sycl_branch` is withdrawn). The ineligible early-plan witness is a host test plus lead witness W2. th32 was pinned at rev 2w `3afc3b020`, and is pinned at rev 2x `90be8e260` since rev 3d. | header, §3.1, §7 C1 |
| m-1 witness margin | Fixed: the lines are all printed by t = 3.9 s, and chunk 1 is at t = 8.12 s (`run2.log:2827`), a margin of about 4.2 s. `GGML_SYCL_HOST_RESERVE_MB=16384` is added as a belt. | §7 C1 |
| m-2 idle DMA pool | Fixed: the DMA staging pool is sized 0 when no host weight is `PINNED`-backed, and the `DMA staging pool: 0.0 MB` line is a C4 criterion. | §4.4, §7 C4 |
| m-3 readers not censused | Fixed: a readers table for `weight_host_bytes`, with each reader's assigned meaning, pinned by the loader source test. | §4.4, §7 C4 |
| m-4 two headroom expressions; override bypass | Fixed: `resident_room` uses `host_headroom(MemTotal)`, and the "one number" claim is scoped to MemTotal ≥ 116 GiB. `HOST_RESERVE_MB` becomes `min(override, host_ledger_budget())`, so it can lower the budget but never exceed the headroom. `test-sycl-host-budget` covers it. | §4.5, §7 C5 |
| m-5 getrows retries a refusal | Fixed: `refused_file_mapped` is terminal in the getrows slice loop, so there is one call, one counted refusal, no retry line, and the CPU fallback. `test-sycl-getrows-refusal-terminal` covers it. | §5.3, §7 C4 |
| m-6 equivalent tier mutant | Fixed: the mutants are the record registered as `HOST_MALLOC`, and the `FILE_MAPPED` branch moved after `zone_managed` into the `sycl::free` fallthrough. | §7 C2 |
| m-7 preload site | Fixed: `gs:34696-34730` is added to the §4.3 site table and §5.3, and the loader source test pins it. | §4.3, §5.3 |
| m-8 C1 test vs C4 deletion | Fixed: C4 narrows `lm:2693` to the no-tier case and amends the C1 caps test. | §4.3, §7 C1, C4 |
| m-9 pre-fill `no_alloc` | Fixed: under `no_alloc` the pre-fill still runs (rows are metadata), `lazy_read.any()` is false, and no range reaches `llama_mmap`. | §3.1, §7 C1 |
| m-10 `SYCL_Mapped` `alloc_buffer` | Fixed: size 0 returns a dummy buffer, and any other size aborts naming the buft. | §4.3 |
| m-11 `memory_tier::MMAP` | Fixed: listed in §2.7 and renamed `UNRESOLVED` in §4.8. The census gate covers all four legacy names. `alloc_type::MMAP` is split into `HOST_MALLOC` (`::free`) and `HOST_OWNED_ELSEWHERE` (the KV synthetic base, gs:39615-39619, :39636-39637, released at :38403-38410). | §2.7, §4.8, §7 C2 |
| m-12 allowlist tokenizer | Fixed: comments and strings are stripped, `mmap (` is matched with a fixture, and a non-listed-file mutant is added. | §4.1, §7 C2 |
| m-13 AnonPages has no positive control | Fixed: `test-sycl-pinned-meminfo-control` (lead-run) names the counter that pinned host USM lands in. The C4 criterion keys on that counter. If none moves, the test exits 77 and the criterion rests on `Committed` plus `MemAvailable`. | §7 C4 |
| m-14 residency-test hygiene | Fixed: a non-tmpfs fixture, eviction per arm, `cachestat` through `syscall(__NR_cachestat)`, the `RLIMIT_MEMLOCK` skip, and arm 4 last on a fresh fixture. | §4.6, §7 C3 |
| m-15 th32 pin | Fixed: re-pinned at rev 2u `0bd69e251` in rev 3b, at rev 2w `3afc3b020` in rev 3c, and at rev 2x `90be8e260` in rev 3d (r3 X7). | header |
| m-16 map segments | Fixed: exact definitions. Map segments are the union of the buffer's tensor ranges rounded outward, never a complement. `WILLNEED` and lock segments round inward. The pure-function test has a complement mutant. | §4.6, §7 C3 |
| m-17 counter chokepoints | Fixed: `mem_copy`, `mem_copy_async` and `mem_copy_ptr_async` (`mem-ops.hpp:13-62`), `copy_to_device_async` (uc:10799), and `stream_dma` (uc:13959). | §5.3, §7 C4 |
| m-18 counter log line | Fixed: `[FILE-MAPPED] counters: stream_dma_refusals=N device_copies_outside_materialization=M`, printed at WARN at load end and at unload. | §5.3, §7 C4 |
| m-19 non-POSIX builds | Fixed: `load_maps_files` requires the tier to exist in the build (POSIX only). On Windows SYCL it is false, and the no-tier caps path of `lm:2693` applies. | §4.3 |
| addendum 3 (§F2a I-9): `lazy_eligible` bit | Folded: one bit per row, filled once at the C1 pre-fill by calling T6a's predicate (a): OFF never, AUTO above 4 GiB, ON always, on the pre-fallback mode. Consumers read the bit, and an ineligible row is an ordinary weight. Membership stays flag-based. Data edge: T6a lands no later than C1. | §3.1, §7 C1 |
| owner ruling: default-context arms | Folded: every Qwen and GPT-OSS gate set has a no-`-c` arm, HARD and scored. The Mistral gate is already default-context. | §0, §7 C4, C6 |
| correction: uize part 3 landed | Folded: `35ce2e65f`, `812c7d790`, `7259727e6` and `ae4204ba7` are in the base, so a no-`-c` arm expects the `KV overflow re-placed to host tier` line (gs:18382) with a pre-registered N and range. Any refusal line is a FAIL, and so are garbage tokens, an abort, or `Shmem` growth. Qwen: 12 layers, layers 3..47 (real sizing), or 48 layers, 0..47 (the over-claim); the figures became 6912.0 and 27648.0 MB in rev 3d with the indexer term (r3 B), and the over-claim outcome is a FAIL attributed to 0edf (r3 n1). GPT-OSS on the B50: rev 3d cites zhcn's GDC3 row and carries no copy of its figures (§M56 (a)). | §7 C4 |
| llama.cpp-0edf: the x4 hybrid KV over-count | Cited, and kept out of 5efe's scope. The Qwen no-`-c` arm's two pre-registered lines discriminate it, but do not fix it. | §7 C4 |

| addendum 4 (§F2a I-9): which mode feeds the bit | Folded in rev 3c. The pre-fallback capture `mode_pre` is C1's, taken before the moved fallback block. The bit reads `mode_pre` for the taker and `mode_post` otherwise. `deferral` carries only the kind, and `STORAGE` is written at the pre-fill. The drift check is mode-independent, with a RED mutant of rev 3a's form. The required witness is a host arm with `AUTO`, an `mmap_support = false` device and a sparse table above 4 GiB (bit set, early plan excludes it; RED: a post-fallback fill). deepseek41 is dropped from the declaration. The stale "commit 1 does not depend on that" bullet (end of §3) is corrected. | §3, §3.1, §7 C1 |
| addendum 5 (§F2a I-9): the taker is per-model; (arch, name) key; `add()` reads the bit | Folded in rev 3c. `taker = sycl_model_loading_guard.active`, captured once at the pre-fill. The declaration is keyed by (arch, name), and the attributes come from one per-arch function. `lazy_read::add()` accepts iff `deferral == LAZY_READ`, and its mode, floor and `SUPPORTED` tests are deleted and pinned by a source gate (mutant: upstream's `add()`). The `-ngl 0` witness: the host matrix never yields `STORAGE` without `taker`, the source gate has a process-level-selector mutant, and lead witness W3 expects `deferral=LAZY_READ taker=0` with no early plan. The missing-attribute refusal is T6's, for `STORAGE` rows only. | §3.1, §7 C1 |
| addendum 6: th32 rev 2w interpretations | Folded in rev 3c. T6a's `SUPPORTED` input is `llama_mmap::SUPPORTED` off the taker and `true` on it. `add()`'s switch and its source gate are C1's. The attribute function's name is `llama_model_lazy_tensor_attrs(arch, hparams, name)`. | §3.1, §7 C1 |
| vehicle clarification and the programmatic `n_ctx = 0` rule | Folded in rev 3c. The no-`-c` arms are `llama-completion` (Qwen, Mistral) and `llama-cli` (GPT-OSS). `llama-bench` and `llama-perplexity` stay fixed. A programmatic arm sets `n_ctx = 0` (`src/llama-context.cpp:5091`, `:540`) and is VOID unless `llama_n_ctx == n_ctx_train`. | §7 |
| found in rev 3c: the C1-to-T6 window | Ruled in addendum 8 (b). C1 and T6 reach master in one merge, with T6 built on C1, and the tracker blocks C1's merge on T6. Interim `EAGER` is refused. C4's Qwen arms are VOID without T6 and T7, and their host-growth bound adds th32's printed table bytes; both were accepted. | §3.1, §7 C4 |

| addendum 7 (a), (b), (c) | Folded in rev 3c. (a): no mutable static table, and attributes stored on the per-model record entry; the naming follows addendum 8 (a) (see I-5). (b): lifetime {GRAPH, CONTEXT, MODEL}, owner-boundary leak checks, owners cited, UNOWNED rows brought to the lead (see I-2). (c): the published site table, 23mk citing it, and host SCRATCH reading the RUNTIME function, not copying it. The pre-size stays 4 + 3 chunks, since the base moves by at most +0.6 MB. dlopen: the new entry point is scalar-only, under a new name. | §3.1, §4.4, §7 C1, C4 |

| addendum 8 (a), (b) | Folded in rev 3c. (a): rev 3c's `_names`/`_attrs` signatures stand. (b): the C1-to-T6 window is closed by one merge. | §3.1, §7 C4 |

**design-5efe-r3** (reviewed 3763bb164..3b447ee70, with spot-checks of rev 3c;
0C/5I/7m/1n and contradictions X1-X7), folded in rev 3d:

| finding | disposition | where |
|---|---|---|
| r3 A notify unsound: VRAM half, ordering, chunk rounding, unchecked cast, N-times growth | A.1 and A.4 were fixed in rev 3c (the host-only, scalar-ABI entry point under a new proc name); rev 3d adds the `ggml_backend_is_sycl` test before `backend->context` is read (`gs:107651`). **A.2:** the notify now precedes each allocation it covers: in `output_reserve` between the old buffer's free (`src/llama-context.cpp:3550`) and the allocation (`:3566`), and in `sched_reserve` after size-only `graph_reserve(..., true, sizes)` calls (`ggml_backend_sched_reserve_size`, `:3908-3910`) and before the first allocating reserve (`:2055`); a WARN compares the allocated size with the notified one. **A.3:** SCRATCH is scored by chunks, one `grow_zone` of `max(2048 MB, align_up(deficit + n x alignment))` (`pinned-pool.cpp:613-712`, `:798-803`). **A.5:** one demand per `llama_context`, keyed by one backend context, replaced on re-notify and dropped at that context's destructor; the global zone grows by `sum - grown_for_demands`. `test-sycl-host-scratch-demand` has the accumulate and per-backend mutants. | §4.4, §7 C4 |
| r3 B demoted host KV routed to SCRATCH; zone-1 criterion vacuous; the indexer omitted | Fixed per the lead's ruling. `SYCL_KV_Host` gets its own `alloc_buffer` with role `KV`, zone 1, lifetime `CONTEXT` (it copied `SYCL_Host`'s iface, `gs:42750-42758`). The qwen4exp `idx_` cache (`src/llama-memory-hybrid-idx.cpp:45-67`, `lm:3580-3585`) takes the same path, and the demotion counts it through `kv_idx_width`, a `uint32_t` in the inventory's tail padding (measured: `sizeof` stays 176), and `kv_demand_for_layer`. Qwen's no-`-c` line is 12 layers, 6912.0 MB, 3..47. Zone 1 must grow, scored by chunks: four 2048.0 MB `grow_zone: zone 1` lines. "Plus a small recurrent state" is withdrawn: the recurrent state is device-resident (`src/llama-memory-recurrent.cpp:97-103`). | §4.4, §7 C4 |
| r3 C `llama-cli` drops WARN | Fixed. `tools/cli/cli.cpp:36` sets `LOG_LEVEL_ERROR` before parsing; `-lv N` overrides it, since the parser sets the threshold from the default first (`common/arg.cpp:769`) and the `-lv` handler sets it again (`:3950-3962`). Every `llama-cli` arm passes `-lv 4`. The digit scoring is unaffected because the answer is stdout (`common/console.cpp:75`) and logs are stderr (`common/log.cpp:113-116`); the arms capture them separately. The `[FILE-MAPPED] counters` claim is scoped to `llama-completion`. | §4.6 counters, §7 preamble, C4 step 5, C5, C6 |
| r3 D the C1-to-T6 window | Closed in rev 3c by addendum 8 (b), and the verdict's 3c spot-check confirms it: C1 and T6 merge together, the tracker blocks C1's merge on T6, and W1 kills on `deferral=STORAGE taker=1`. | §3.1, §7 C1 |
| r3 E C5 has no no-`-c` arm | Fixed: C5's lead runs add the C4 step-3 Qwen arm and the step-5 GPT-OSS arm with `-lv 4`. | §7 C5 |
| r3 m1 `add()`'s `w != nullptr` breaks the fit pass | Fixed: `add()` returns `deferral == LAZY_READ` alone, and inserts into membership only when `w` is non-null, so the `no_alloc` fit pass still reads the table as lazy (`lml:1775-1776`). The "as today" wording is gone. A RED mutant restores the rev 3c term. | §3.1, §7 C1 |
| r3 m2 Qwen arms' vehicle and `-v` | Fixed: `llama-completion -no-cnv -v`, and the zone-0 absence counts only beside its `pre_allocate` positive control. | §7 preamble, C4 step 2 |
| r3 m3 double charge of the PP terms | Confirmed still present in rev 3c: the runtime pre-allocate (`gs:5649-5653`) kept `pp_pipeline + pp_moe_onednn`. One owner, host SCRATCH (addendum 7 (c)); the runtime pool, which after zone configuration serves only the KV-zone miss fallback (`uc:15483-15495`), drops them. Re-derived: base 4 chunks + runtime 1 chunk (91.9 MB) = **5 chunks, 10240.0 MB**; 6, 7 and 8 chunks name what did not land. Both PP terms are VRAM allocations (`uc:18197-18199`), so the RUNTIME read left in host SCRATCH sizes no host tenant; removing it would amend addendum 7 (c) and is brought to the lead. | §0, §4.4, §7 C4 |
| r3 m4 rerouted SCRATCH tenants have no sizing term | Fixed: context buffers are the notified demand; `DEVICE` holders register a demand before allocating (the pool formulas from `gs:8018-8030`, `pinned-buffer-pool.cpp:38-39`, `cpu-expert-pool.cpp:56-59`); host KV sizes zone 1 through the miss path at context init; each `GRAPH` row maps to a named plan term or is ratcheted `UNSIZED`. | §4.4, §7 C4 |
| r3 m5 I-3 row misnames the band | Fixed in the I-3 row above: the band is `[shared page + 1 page, edge + reach)`, and `W` is the `MADV_RANDOM` window. | Appendix B |
| r3 m6 `gs:39954` is a P3 zero-copy pattern | Fixed: the site table cites llama.cpp-nl0f beside 23mk's row 56. | §4.4 |
| r3 m7 proc-address name | Fixed in rev 3c; the r3 spot-check confirms it. | §4.4 |
| r3 n1 0edf outcome | Changed per the lead: the 48-layer line (27648.0 MB with the indexer) is a **FAIL**, plan != reality, attributed to llama.cpp-0edf and not a 5efe regression. | §7 C4 step 3 |
| r3 X1 PROCESS lifetime | Closed in rev 3c: `{GRAPH, CONTEXT, MODEL}`, no `PROCESS` (addendum 7 (b)); rev 3d adds `DEVICE` (addendum 10 (b)). | §4.4 Lifetimes |
| r3 X2 attributes in the declaration struct | Closed in rev 3c: names only in `llama_model_lazy_tensor_names`, attributes from `llama_model_lazy_tensor_attrs(arch, hparams, name)` stored per model (addenda 7 (a), 8 (a)). | §3.1 |
| r3 X3 `sycl_branch` from `hooks_enabled()` | Closed in rev 3c: `taker = sycl_model_loading_guard.active` (`lm:2228-2230`). | §3.1 |
| r3 X4 the bit always reads `mode_pre` | Closed in rev 3c: `T6a(taker ? mode_pre : mode_post, size, taker ? true : SUPPORTED)`; rev 3d pins the third argument in source (th32 2x). | §3.1, §7 C1 |
| r3 X5 eligible-but-`EAGER` | Closed in rev 3c: `deferral != EAGER` iff `lazy_eligible`. | §3.1 |
| r3 X6 23mk overlaps unresolved | Closed in rev 3c: the published site table, which 23mk's class-H rows cite. | §4.4 |
| r3 X7 th32 pin stale | Fixed: re-pinned at rev 2x `90be8e260`. | header |
| found in rev 3d: host bufts' `get_max_size` names its own zone | `SYCL_Host`'s answers from WEIGHT (`gs:42477`) and `SYCL_HostCompute`'s from STAGING (`gs:42977`); after C4's routing both would chunk against the wrong zone (the llama.cpp-glkg shape). Each buft now builds one intent for `alloc_buffer` and `get_max_size`, which calls `select_zone`; `tests/test-sycl-host-zone-config-source-contract.py` is amended. | §4.4, §7 C4 |
| addendum 10 (a) never amend a reported commit | Followed: rev 3d is a new commit on `fef860772`. | header |
| addendum 10 (b) the 14 UNOWNED holders | Folded. `DEVICE` (owner: the device's `unified_cache`, checked at its teardown) for the two pools, and for `g_attn_host_staging` under the ruling's device-shared clause, with the reason stated; all three are released by `ggml_sycl_shutdown_global_runtime_pinned_owners` (`gs:7575-7590`) before the cache goes. The other 11 holders (12 lines) are `CONTEXT` targets pending **llama.cpp-30r1**, and carry the interim `UNOWNED` marker, routed to WEIGHT as today, until it lands. The ratchet fails on any new `UNOWNED` site and cites llama.cpp-30r1. | §4.4, §7 C4 |
| addendum 10 (c) notify | Folded: the scalar entry point stands; the old function stays uncalled and its VRAM half is 23mk's dead-code census's. | §4.4 |
| §M56 (a) GPT-OSS figures have one source | Folded: C4 step 5, C5 and C6 cite zhcn's GDC3 row and carry no copy of its N, figure or range. | header, §7 C4, C5, C6 |
| th32 2x C1 requirement 1: the third argument in source | Folded: `test-llama-lazy-range-record-source` pins `taker ? true : llama_mmap::SUPPORTED`, with the always-`SUPPORTED` and always-`true` mutants; the reason is that `SUPPORTED` is a constant `true` on Linux and Windows (`src/llama-mmap.cpp:711-715`). | §7 C1 |
| th32 2x C1 requirement 2: one per-model line | Folded: `lazy-range record: mode_pre=<m> mode_post=<m> taker=<0|1>` at INFO, once per model; a `llama_log_set` callback receives every level (`src/llama-impl.cpp:40-55`), and only `common_log` drops INFO (`common/log.cpp:542-546`). W1-W3 read it. | §3.1, §7 C1 |
