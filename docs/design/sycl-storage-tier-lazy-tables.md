# SYCL STORAGE tier for lazy row tables (PLE / n-gram / engram embeddings)

*Status: DESIGN, not implemented. Tracker: `llama.cpp-th32`. Implementation
tickets are filed after review (proposed split in §11). Written 2026-09-26
against fork `76c7f6548` and upstream `ggml-org/master` `694ec2354`.*

Citations: fork code as `76c7f6548 <file>:<line>`, upstream code as
`694ec2354 <file>:<line>` (read with `git show ggml-org/master:<file>`; the fork
does not have `qwen4exp` or `TENSOR_READ_LAZY` yet, `llama.cpp-n77l` is merging
them). `deepseek41` has no runtime anywhere yet: its conversion is upstream
PR #28696, and its runtime design is `llama.cpp-9g22`
(`docs/design/deepseek41-runtime.md`, branch `task/ds41`), which consumes the
interface in §9.1. Every number in this document was measured on this host on
2026-09-26 under its permanent ambient load (load average 39-57 for the Qwen
probes, ~90 for the DeepSeek ones); the raw logs are in Appendix A and the
probe sources are described in Appendix B.

---

## 0. Summary

**The problem.** Two models carry hashed n-gram embedding tables that are read a
few rows per token and are far too large for their value on a GPU:

| | Qwen3.8-Flash-Next (`qwen4exp`) PLE | DeepSeek-V4.1-Flash (`deepseek41`) engram |
|---|---|---|
| tables | 1: `per_layer_token_embd.weight` | 2: `blk.1.engram_embd.weight`, `blk.14.engram_embd.weight` |
| shape, type | `[160, 320001536]` Q8_0 | `[256, 384006168]` and `[256, 384016682]`, Q3_K in Q3_K_M (Q2_K in Q2_K) |
| size | **50.66 GiB** | **2 x 39.34 GiB** Q3_K (2 x 30.04 GiB Q2_K) |
| row | 170 B | 110 B Q3_K (84 B Q2_K) |
| rows per token | 16 (orders 2-3 x 8 heads) | **48**: 24 per table (orders 2-4 x 8 heads) |
| raw bytes per token | 2720 B | 5280 B Q3_K (4032 B Q2_K) |
| row indices known host-side before compute | yes: hash in `set_input` | yes: `token_map` + hash in `set_input` |
| consumed at | block 1 | block 1 (table 1) and block 14 (table 14) |
| prefetch window | block 0 | table 1: block 0; table 14: blocks 0-13 |
| tier the planner picks on this host | **HOST** (§4.5) | **STORAGE**, both tables (§4.5) |

Each row read from storage is one random 4 KiB read. DeepSeek needs **3x** the
rows per token and **3x** the unique rows per PP512 ubatch, and on this host it
cannot go to HOST at all. So DeepSeek is the harder case, and it sizes the
staging, the cache floor and the IO targets (§3.3, §6.4, §8).

Upstream marks such a table `TENSOR_READ_LAZY` and leaves it in an mmap of the
GGUF: `ggml-cpu` `get_rows` page-faults the rows in on demand. That already
keeps the table off the GPU, which is what placement-decides-the-executor asks
for. It breaks the other three principles, though: the kernel page cache is the
allocator (P1), the kernel's page reclaim is the lifetime (P2), and the
resident bytes are invisible to the planner (P4).

**The design.** Each lazy table becomes a **unified-cache-owned object**
(`lazy_table`). The planner puts it in one of two tiers, with no environment
variable:

- **HOST** (new pageable tier): the whole table is resident in cache-owned
  host memory. The planner chooses this when the table fits the host budget
  that remains after VRAM placement and host experts.
- **STORAGE** (new tier): the table stays in its file. What is resident is a
  row-granular **hot-row cache** (CLOCK), sized from the leftover host budget,
  plus a small per-context **staging** area. Missing rows are read with
  `O_DIRECT` preads that are issued as soon as the row indices exist (at
  `set_input`), in the order the tables are consumed. Each gather waits for its
  own rows later, on a CPU thread.

In both tiers the **CPU executes the gather**. A `GGML_OP_CUSTOM` node reads the
resolved rows, dequantizes them to F32 and emits `[row_elems, rows_per_token x
n_tokens]`, and the scheduler ships that activation to the device. The table
never reaches the GPU: there is no zero-copy and no weight streaming. The
consumer contract that a model implements is §9.1.

**The measured facts that shape it** (details in §2 and Appendix A):

| fact | value |
|---|---|
| where the tables live | bcachefs `/models`: user data on two **USB** SSDs, NVMe holding promoted cached copies only |
| 4 KiB `O_DIRECT` random reads, Qwen table | **1389 / 7815 / 9124 IOPS** at QD 1 / 16 / 64 |
| 4 KiB `O_DIRECT` random reads, DeepSeek `blk.1` table | 1014 / **5153 / 8400 IOPS** at QD 1 / 16 / 64 (download running) |
| bcachefs read amplification | **~60x** on both files: every 4 KiB read costs ~256 KiB of device reads, because `data_checksum=crc32c` is over whole 256 KiB extents |
| mmap page faults (the upstream path) | 1555 rows/s with 1 thread, 7866 with 16 threads: the same device ceiling |
| reference: ext4 on the root NVMe, same probe | 6019 / 82798 / 110110 IOPS (**~12x** bcachefs at QD64) |
| cold decode token, Qwen (16 rows, 16-way pread pool) | 3.75 ms mean, 13.0 ms p99 |
| cold decode token, DeepSeek (48 rows, 48-way) | **8.95 ms mean, 29.1 ms p99**; the first table's 24 rows alone 6.95 / 27.8 ms |
| cold PP512 ubatch, Qwen (~7100 unique rows) | 0.85 s mean, 8.3 k rows/s |
| cold PP512 ubatch, DeepSeek (22-25 k unique rows) | **~2.8-3.2 s**, from 7.8 k rows/s measured on a 20000-row batch |
| HOST-tier gather and dequant | 4.2 us per 16-row token; ~2 ms per 8192-row ubatch |
| steady-state decode tokens with zero misses, 512 MiB cache | Qwen: prose 32%, docs 61%, code 88%. DeepSeek: prose 14%, code 81% (48 rows must all hit) |
| `llama-bench` prompts | uniform random tokens, so **always cold**, at any cache size (§8.5) |

**What the planner will choose on this host.** The fork's own host-budget
formula gives ~163-174 GiB (it follows `MemAvailable`). Qwen: host experts
~86 GiB plus the 50.66 GiB table fit, so the table is **HOST**, and STORAGE is
its fallback. DeepSeek: the owner's primary rung is Q2_K; its backbone fits,
both tables are **STORAGE**, and they get a ~10-25 GiB row cache, past the
knee of every simulated corpus. (Q3_K_M overruns the host budget, §4.5.) STORAGE speed on this box is dominated by the
filesystem's 60x amplification, which is a question for the owner (D3), not a code
problem, and for DeepSeek it is the operating point.

**DeepSeek's rung.** The owner's primary is **Q2_K**; Q3_K_M overruns the
host budget by 30-41 GiB. Cold routed experts on STORAGE, which Q3_K_M would
need, are a later ticket (`llama.cpp-9g22` files it). §9.2 records what the
tier keeps open for it and the expert-read measurements already taken.

---

## 1. Scope and principles

This design must satisfy, and every reviewer checks:

- **P1:** the unified cache is the only allocator
  (`docs/backend/sycl-memory-design.md`, "The one allocation entry point").
- **P2:** ownership is an owner-first `mem_handle`, and its destructor is the
  sole release point (`sycl-memory-design.md`, "Owner-first allocation").
- **P3:** placement decides the executor, and layout follows residency. There
  is no GPU zero-copy of host memory and no weight streaming
  (`sycl-memory-design.md:52-120`).
- **P4:** plan == reality. The planner's accounting equals what is resident.
- **No host waits on the GPU path.** Everything is event-chained, and any
  remaining wait is named.

Owner decisions already taken on the ticket (2026-09-26):

- The planner decides placement, with no configuration.
- HOST if the table fits after VRAM and experts, otherwise STORAGE, with async
  prefetch and a hot-row cache.
- The CPU executes the gather, so only the gathered rows move.
- "Decent speed" in either tier.

Out of scope: the `qwen4exp` port itself (`llama.cpp-n77l`) and the `deepseek41`
runtime (`llama.cpp-9g22`); the GPU kernels that consume the gathered rows
(PLE's `ple_key`/`ple_value`/norms/conv, engram's `engram_q`/`engram_k`/
`engram_wkv`, all ordinary dense weights); the MTP head; and cold routed
experts on STORAGE, a later ticket (§9.2).

---

## 2. Facts

### 2.1 The consumers and their tables

#### 2.1.1 Qwen3.8-Flash-Next PLE

GGUF metadata, read with `gguf-py` (Appendix A.1):

- `qwen4exp.ple.layers = [1]`: one PLE module, at layer index 1.
- `ple.ngram_size = 3`, `ple.heads_per_ngram = 8`, so there are `ple_n_heads = (3-1)*8 = 16`
  heads (`694ec2354 src/models/qwen4exp.cpp:95`).
- `embedding_length_per_layer_input = 160` = `ple_head_dim`.
- 16 per-head vocabularies of about 20 M rows each. They are primes from
  20000003 to 20000171, concatenated at `head_offsets` 0 .. 300001275.
- `layer_multipliers = [23703573157769, 20109073645365, 8052911324071]` (u64).
- `ple.eos_token_id = 248044` (`<|endoftext|>`), which differs from
  `tokenizer.ggml.eos_token_id = 248046`; the hash uses the former.
  `ple.image_token_id = 248056`.
- `per_layer_token_embd.weight` is Q8_0 `[160, 320001536]`, **54400261120 B
  (50.66 GiB)**. It sits in split `00003-of-00006` at absolute file offset
  **192**, and that file is 54400261312 B, so the table *is* the file. A
  160-element Q8_0 row is 5 blocks x 34 B = **170 B**. Row `r` starts at file
  offset `192 + 170 r`, so 4.1% of rows straddle a 4 KiB boundary (measured
  4.0-4.2% in the batch probe).

#### 2.1.2 DeepSeek-V4.1-Flash engram

GGUF metadata of the Q3_K_M rung, read with `gguf-py` from shard 1 (keys) and
shard 2 (tensors); Appendix A.1:

- `deepseek41.block_count = 40`, `embedding_length = 5120`, 384 routed experts
  plus 1 shared, 6 used.
- `engram.layer_ids = [1, 14]`, `engram.head_count = 8`,
  `engram.key_length = 256`, `engram.max_ngram_size = 4`. That makes n-gram
  orders 2, 3 and 4, each with 8 heads: **24 rows per table per token**, and
  **48 per token** across the two tables.
- `engram.multipliers` (8 = 2 tables x 4 positions), `engram.primes` and
  `engram.offsets` (48 each = 2 tables x 3 orders x 8 heads; primes 16000057 ..
  16000889), `engram.token_map` (129280 tokenizer ids onto a compressed vocab
  of 99092), `engram.pad_id = 2`.
- **The hash** (reference `NgramHashState.forward`, `inference/engram.py` of
  `deepseek-ai/DeepSeek-V4.1-Flash`): for position `p`, `c[s] =
  token_map[tok[p - s]]` for `s = 0..3`, with `pad_id` where `p < s` (and after
  a blocking token such as an image placeholder). Per table `L`: `rolling =
  c[0] * M[L][0]`; then for `i = 1..3`: `rolling ^= c[i] * M[L][i]`, and head
  `h` of order `i + 1` reads row `rolling mod prime[L][i][h] +
  offset[L][i][h]`. Every product fits in int64 (99091 x 8.26e13 < 2^63). The
  indices depend only on token ids and the previous three tokens of the same
  sequence, so they are **computable host-side before the graph runs**, in
  `set_input`, exactly like PLE's. The sequence history is the runtime's state
  (`llama.cpp-9g22`).
- `blk.1.engram_embd.weight` is Q3_K `[256, 384006168]`, **42240678480 B
  (39.34 GiB)**, in `DeepSeek-V4.1-Flash-Q3_K_M-00002-of-00009.gguf` at data
  offset **480**; the same shard holds the block's small `engram_q`/`engram_k`
  (`[5120, 4]`) and `engram_wkv` (`[6144, 25600]`). `blk.14.engram_embd.weight`
  is `[256, 384016682]` (the last offset plus the last prime), 42241835020 B;
  its shard had not finished downloading when this was written, so its extent
  is derived, not read.
- A 256-element Q3_K row is one 110 B block; Q2_K is one 84 B block. Row `r`
  starts at `480 + 110 r`, so 2.7% of rows straddle a 4 KiB boundary (measured
  2.5-2.7%).
- The consumer (from `llama.cpp-9g22`): at the start of block 1 and block 14,
  the block's 24 rows per token are concatenated to `[6144, n_tokens]` and fed
  to `engram_wkv` on the device.
- Rung sizes (the GGUF README): Q2_K 246.3 GiB, Q3_K_M 323.4 GiB. The engram
  tables "follow the rung": 40284 MiB each at q3_K (which matches 384006168 x
  110 B), so 30762 MiB each at q2_K (x 84 B).

### 2.2 The upstream mechanism

- **Marking.** `create_tensor(..., TENSOR_READ_LAZY)` for the table
  (`694ec2354 src/models/qwen4exp.cpp:188-189`). The other in-tree user is
  gemma4's `per_layer_tok_embd` (`694ec2354 src/models/gemma4.cpp:58`). The
  `deepseek41` runtime marks both `engram_embd` tables the same way (§9.1).
- **Decision.** `llama_model_loader::lazy_read::add()`
  (`694ec2354 src/llama-model-loader.cpp:1088-1117`). `LLAMA_LAZY_MODE_OFF`
  never lazies. `AUTO` (the default, `694ec2354 src/llama-model.cpp:2861`)
  lazies marked tensors larger than 4 GiB. `ON` lazies every marked tensor.
  No mmap means no lazy. The mode enum is at `694ec2354 include/llama.h:218-222`.
  The CLI flag is `-lzm, --lazy-mode` (`694ec2354 common/arg.cpp:2706-2717`) and
  the llama-bench flag the same (`694ec2354 tools/llama-bench/llama-bench.cpp:810-835`);
  `--tensor-read-lazy` was renamed by upstream `bebc9350e`. `AUTO` falls back
  to `OFF` on devices without mmap support
  (`694ec2354 src/llama-model.cpp:1520-1528`).
- **Buffer.** A lazy tensor goes into its own context (`ctx_key.lazy`,
  `694ec2354 src/llama-model-loader.h:150-163`) with the **CPU** buffer type
  (`lazy_read::buft()`, `694ec2354 src/llama-model-loader.cpp:1080-1086`,
  forced at `:1211-1213`). Its data is the mmap
  (`694ec2354 src/llama-model.cpp:1840-1843`,
  `694ec2354 src/llama-model-loader.cpp:1640-1661`). It is never mlocked.
- **mmap.** The lazy ranges are excluded from `MAP_POPULATE`/`WILLNEED` and get
  `POSIX_MADV_RANDOM` (`694ec2354 src/llama-mmap.cpp:469-510`).
- **Row indices.** `llm_graph_input_ple::set_input` computes them **host-side**
  (`694ec2354 src/models/qwen4exp.cpp:1066-1127`), because ggml has no int64
  and no xor. The 16 indices of every token are therefore known on the host
  before the graph is computed.
- **Consumer.** `ggml_get_rows(per_layer_tok_embd, rows)` followed by a reshape
  to `[2560, n_tokens]` (`694ec2354 src/models/qwen4exp.cpp:1185-1204`). It is
  expanded into the graph **before** layer 0 ("same graph split as
  `build_inp_embd`", `694ec2354 src/models/qwen4exp.cpp:381-386`) and consumed
  by `build_ple` at layer 1 (`:397-398`).
- **The other consumer shape.** gemma4 also takes `ggml_view_1d` of row 0 of
  its lazy table for multimodal embeddings
  (`694ec2354 src/models/gemma4.cpp:462`). A generic design must cover a
  constant-row view as well as an indexed `get_rows` (§9).

### 2.3 What the fork has today, and what it lacks

- **Tiers.** `alloc_tier` is `{DEVICE_VRAM, HOST_PINNED, MMAP_TRACKED}`
  (`76c7f6548 ggml/src/ggml-sycl/unified-cache.hpp:5590-5594`), and
  `unified_alloc` refuses `MMAP_TRACKED` runtime allocations
  (`76c7f6548 ggml/src/ggml-sycl/unified-cache.cpp:15125-15126`). **There is no
  pageable host tier.** Every cache-owned host byte is pinned USM host memory,
  which the driver maps for GPU access. Pinning 50 GiB that the GPU never
  touches spends locked, device-mapped memory for nothing. Whether xe also
  counts that memory as TTM `Shmem`, the resource behind this host's OOM
  history, is unverified, and the design does not depend on it.
- **Host budget.** The pinned pool's budget is computed in the cache
  constructor as `min(total * 90%, MemAvailable - max(8 GiB, total/32))`
  (`76c7f6548 ggml/src/ggml-sycl/unified-cache.cpp:4147-4259`). **The planner
  takes no host budget:** `compute_placement_plan` receives a VRAM budget only
  (`76c7f6548 ggml/src/ggml-sycl/unified-cache.hpp:1591-1603`), and
  `plan.weight_host_bytes` (`:758`) is summed but never compared with anything.
  Host overcommit is found at allocation time, not at planning time. That is a
  P4 gap that already exists for host experts, and this design must close it
  (§4).
- **CPU-buffer weights.** A weight in a CPU-owned buffer leaves the SYCL
  inventory, and the post-`create_tensor` replan treats its layer as host.
  The canonical contract §8 items 2 and 4 place CPU-buffer tensors outside the
  contract. Upstream's lazy tensor is exactly such a tensor, so merged as-is it
  would sit outside P1/P2/P4 by construction.
- **Hooks.** The fork calls into the SYCL backend through
  `ggml_backend_reg_get_proc_address` hooks
  (`76c7f6548 src/llama-model.cpp:63-111`). This design adds its entry points
  the same way.
- **Custom ops.** SYCL declines `GGML_OP_CUSTOM`: `supports_op` falls to
  `default: return false`
  (`76c7f6548 ggml/src/ggml-sycl/ggml-sycl.cpp:108771-108772`), so a custom op
  is scheduled on the CPU backend.

### 2.4 The storage under the table (measured)

`/models` is bcachefs with three members (sysfs
`/sys/fs/bcachefs/118129b2-.../dev-*`, and `bcachefs fs usage`):

| dev | label | block device | link | user data | cached data |
|---|---|---|---|---:|---:|
| 0 | `nvme.kingston` | nvme0n1 (Kingston OM3PDP3512B) | PCIe | **0** | 182 GiB |
| 1 | `bulk.ugreen` | sdb (500SSD1 in a USB enclosure) | USB 10 Gb/s, uas | 612 GiB | 0 |
| 2 | `bulk.t7` | sda (Samsung PSSD T7) | USB 5 Gb/s, uas | 171 GiB | 0 |

The filesystem options are `foreground_target=nvme`, `background_target=bulk`,
`promote_target=nvme`, `promote_whole_extents=1`, `data_checksum=crc32c`,
`encoded_extent_max=256k`, `block_size=4k`, no compression, 1 replica. So:

- **Every user extent of the table lives on a USB SSD.** The NVMe holds
  promoted *cached* copies (182 GiB across the filesystem). `filefrag` shows two
  pointers per 256 KiB logical extent across most of the file (463214 fiemap
  extents for 13.28 M blocks), which is consistent with a cached copy on NVMe
  plus the USB home. Per-device `io_done` counters during the probes put
  ~80% of 4 KiB-read bytes on NVMe at QD1 and ~60% at QD64, with the rest on the
  two USB devices.
- **A 4 KiB read costs ~256 KiB of device reads (60x).** This was measured, not
  inferred: the sum of the per-device `io_done` read deltas divided by the
  bytes fio returned is 60.2x / 59.4x / 59.2x at QD 1/16/64. A crc32c checksum
  covers the whole encoded extent (up to `encoded_extent_max` = 256 KiB), so a
  partial read must fetch and verify the whole extent. The row-granular gather
  this table needs is the worst case for that layout.
- **At QD64 the USB devices saturate**: ~420-445 MB/s each, which is the T7's
  5 Gb/s link. That caps random 4 KiB reads at **~9.1 k IOPS** whatever the
  engine: fio `io_uring`, a pread thread pool and mmap faults all land at the
  same ceiling (§5.1).
- Sequential 1 MiB `O_DIRECT` reads of the table reach **1.78 GB/s** (a
  whole-table read takes ~31 s).

**The DeepSeek table, same filesystem.** The same probe over `blk.1`'s byte
range in shard 2 (`--offset=4096 --size=42240000000`) measured **1014 / 5153 /
8400 IOPS** at QD 1/16/64 (p99 7.7 / 18.7 / 28.2 ms), with amplification
**60.2x** at QD16 and **60.0x** at QD64. The QD1 amplification is not
attributable: the model download was still writing the next shard, and
bcachefs reconcile read 6.8 GB from the NVMe during that 15 s run to move data
to the USB targets. Load average was 90, against 39-57 for the Qwen runs, so
the lower QD1 and QD16 figures reflect contention, not a property of the file.
The ceiling at QD64 is the same ~8.4-9.1 k IOPS. The shard had been written
~15 minutes earlier, so part of it may still have been on its NVMe foreground
target; the DeepSeek numbers are therefore, if anything, optimistic for a
settled file.

---

## 3. Object model: the lazy table as a unified-cache object (item 1)

### 3.1 Tiers

`alloc_tier` gains two values. `cache_location` and `expert_resolve_tier`, the
reporting enums, gain the matching names.

| tier | memory | who may read it | new? |
|---|---|---|---|
| `DEVICE_VRAM` | device USM | that device | no |
| `HOST_PINNED` | USM host, pinned pool | CPU, and the GPU for DMA staging only | no |
| **`HOST`** | pageable anonymous memory, `mmap(MAP_ANONYMOUS)` with `MADV_HUGEPAGE` | **CPU only** | **yes** |
| **`STORAGE`** | a file extent: nothing resident by itself | CPU only, through the lazy-table read path | **yes** |
| `MMAP_TRACKED` | legacy: file mmap tracked for weights | CPU | no |

`HOST` exists because the lazy table (and, later, anything else only the CPU
executes) should cost pageable RAM, not pinned GPU-visible RAM. The unified
cache performs the allocation inside `unified-cache.cpp` through a new internal
raw wrapper, `unified_cache_raw_alloc_host_pageable(size)` /
`unified_cache_raw_free_host_pageable`. That wrapper is added to the canonical
contract's §3 allowlist exactly like `unified_cache_raw_malloc_host`, so no
other file calls `mmap` for it (P1).

A `HOST` allocation is requested through the owner-first front door:
`unified_allocate_owner(req)` with `req.intent.tier = alloc_tier::HOST` and
`role = alloc_role::WEIGHT` for the table body. The row cache and staging use
new roles (§3.3). The owner class is **`EXTERNAL_EXACT`**: each allocation is
exact and independently released, not a slice of a cache backing pool. This
keeps the two-site `CACHE_BACKING` allowlist (canonical contract §3.1)
untouched.

### 3.2 The `lazy_table` object

One `lazy_table` per lazy tensor per loaded model. It is **model-scoped**: the
rows are immutable weights, so every context of the model shares it.

```
lazy_table
  identity   : ggml_sycl_cache_id (owner = model token, name = tensor name)  // the usual weight key
  geometry   : type, ne[0] (row elements), n_rows, row_size (ggml_row_size)
  source     : file path + file index, byte offset, nbytes   // from llama_tensor_weight
  tier       : HOST | STORAGE                                // the planner's decision, immutable
  HOST:    body   : mem_handle (HOST, WEIGHT, n chunks of <= 2 GiB)
  STORAGE: fd     : O_DIRECT file descriptor, owned (closed in the destructor)
           cache  : lazy_row_cache (below)
  stats      : rows gathered, hits, misses, IO bytes, IO wait ns (§10)
```

- **Ownership (P2).** Every memory the object uses is a `mem_handle` minted by
  `unified_allocate_owner()` and held by the object. The object is destroyed
  only when its model token is unloaded (the §12 lifecycle of the canonical
  contract). Its destructor drops the handles, and that is the sole release
  point. The fd is not memory, but it follows the same rule: the object owns it
  and closes it once, in the destructor, after the in-flight-IO quiesce below.
- **In-flight IO pins the destination.** An outstanding read writes into a
  staging buffer. Every IO ticket (§5.3) therefore holds a *copy* of the
  staging `mem_handle`, in the same way `retain_handles_until_event()` retains
  handles for device work. A ticket is freed only after its last completion has
  been reaped, so a context or model teardown cannot free a buffer that the
  kernel is still DMA-ing into.
- **The teardown wait.** A destructor that finds tickets outstanding waits for
  their completions. This is a teardown path, where waits are allowed.

### 3.3 The hot-row cache and staging: the only resident STORAGE bytes

| allocation | scope | tier / role | size |
|---|---|---|---|
| row cache slab | model | `HOST` / new `alloc_role::LAZY_ROW_CACHE` | `slots x row_size`, where `slots` comes from the planner (§6) |
| row cache index | model | `HOST` / `LAZY_ROW_CACHE` | `slots x 12 B` (8 B row key, 1 B CLOCK bit, padding); open-addressed tag table: ~7% of the slab on 170 B rows, ~11% on 110 B, ~14% on 84 B |
| staging: row landing slab | context, per table | `HOST` / new `alloc_role::LAZY_STAGING` | `depth x n_ubatch x rows_per_token x row_size`: Qwen 2 x 512 x 16 x 170 B = **2.66 MiB**; DeepSeek 2 x 512 x 24 x 110 B = **2.58 MiB per table, 5.16 MiB for both** |
| staging: block ring | context, shared by its tables | `HOST` / `LAZY_STAGING` | `QD x 2 x io_block` = 64 x 8 KiB = **512 KiB** (4 KiB-aligned, for `O_DIRECT`) |

On `STORAGE`, those four lines are the **entire** resident footprint of the
table, and the planner charges exactly those bytes (P4). On `HOST` there is no
row cache (the table *is* the cache) and no block ring. The landing slab is
still used, so that one gather routine serves both tiers.

### 3.4 Pointer discipline

The CPU gather obtains raw views (a row cache slot, a chunk of the HOST body, a
landing-slab slot) by resolving the owning handle **inside** the gather call,
and never stores them. That is the same rule as the canonical contract §4, and
the new resolution entry point is added to its allowlist. No table is keyed by
a pointer: the row cache is keyed by `(row index)` within its `lazy_table`,
whose identity is the weight's `ggml_sycl_cache_id`.

---

## 4. The planner decides the tier (item 2)

### 4.1 Inventory

The loader gives lazy tensors to the SYCL inventory as a **distinct class**,
`placement_lazy_table_info`. It carries the name, type, `ne`, `nbytes`,
`row_size`, `rows_per_token`, `consumer_layer`, file index and offset. They are
never placement entries for a device: P3 forbids a device tier for them, so the
planner cannot offer one. `rows_per_token` and `consumer_layer` are arch facts
that `llama` supplies (§9.1):

| model | table | `rows_per_token` | `consumer_layer` |
|---|---|---:|---:|
| qwen4exp | `per_layer_token_embd` | 16 (`ple_n_heads`) | 1 |
| deepseek41 | `blk.1.engram_embd` | 24 (`(max_ngram_size - 1) x head_count`) | 1 |
| deepseek41 | `blk.14.engram_embd` | 24 | 14 |
| gemma4 | `per_layer_tok_embd` | 1 (token-indexed) | 0 |

`consumer_layer` orders IO issue (§5.3) and defines each table's prefetch
window (§5.4).

On the `llama` side this happens where `lazy_read::add()` records the tensor
(`694ec2354 src/llama-model-loader.cpp:1088-1117`). The SYCL path, when
`llama_model_sycl_hooks_enabled()` holds, records it in the inventory
**instead of** adding its range to the mmap lazy set. The tensor gets a
metadata-only context and **no buffer**, because nothing will ever read
`tensor->data` (§7 replaces every consumer). The memory-fit pass (`no_alloc`)
and the real load take the same decision, which preserves upstream's
invariant at `694ec2354 src/llama-model-loader.cpp:1342-1345`.

### 4.2 A host budget becomes a planner input

`compute_placement_plan` / `compute_multi_device_plan` gain a
`host_budget_bytes` argument. It comes from the **same** formula as the pinned
pool's budget (`76c7f6548 ggml/src/ggml-sycl/unified-cache.cpp:4147-4259`),
which moves into one shared function so there is one source for the fact. The
plan gets:

```
size_t host_budget;                 // the input, recorded
size_t host_planned_bytes;          // everything below, summed
std::vector<placement_lazy_table> lazy_tables;
struct placement_lazy_table {
    std::string name;
    alloc_tier  tier;               // HOST or STORAGE
    size_t      table_bytes;        // file bytes
    size_t      host_resident_bytes;// HOST: table_bytes; STORAGE: cache + index
    size_t      cache_slots;        // STORAGE only
    size_t      staging_bytes_per_context;
};
```

`host_planned_bytes` = the host weights (experts and dense overflow, after the
existing headroom ratio) + the host KV + the host zones (`host_zone_*_bytes`,
`moe_cpu_expert_staging_bytes`, `moe_host_recipe_workspace_bytes`,
`cpu_quant_buffer_bytes`) + every lazy table's `host_resident_bytes` + the
declared per-context staging x `n_seq_max`-bounded context count.

### 4.3 The decision, in order

1. VRAM is placed exactly as today: dense first, then experts.
2. Host weights, host KV and the host zones are charged against
   `host_budget`. If they alone exceed it, the plan refuses with a typed
   reason (`host-budget-exceeded`). Today it silently overcommits (§2.3).
   (Moving cold routed experts to STORAGE instead of refusing is the later
   ticket of §9.2.)
3. `leftover = host_budget - (step 2) - host_reserve`. `host_reserve` is the
   larger of 2 GiB and 1% of the budget, headroom for the process's own heap
   growth.
4. For each lazy table, largest first: if `table_bytes <= leftover`, the tier
   is **HOST**, and `leftover -= table_bytes`. Otherwise the tier is
   **STORAGE**, with `cache = clamp(leftover_share, floor, table_bytes)`, where
   `floor = 2 x n_ubatch x rows_per_token x row_size` (one double-buffered
   ubatch working set). If `leftover < floor`, the plan refuses
   (`lazy-table-cache-below-floor`) rather than run with a cache that cannot
   hold one ubatch.
5. Several STORAGE tables split the leftover in proportion to
   `table_bytes x rows_per_token`, which is a proxy for miss traffic. Open
   decision D7 (§13) covers the alternative.

Nothing in this sequence reads an environment variable. `--lazy-mode` keeps
its upstream meaning of **eligibility**: `OFF` means the tensor is an ordinary
weight and never a lazy table, and `AUTO`/`ON` pick the eligible set. The tier
is always the planner's (D4).


### 4.4 Interaction with CpuExpertPool

- **Memory.** Host experts are charged before lazy tables (step 2 before step
  4), which is the ticket's ruling ("after VRAM and experts"). A table on
  STORAGE never displaces an expert.
- **Execution.** Each gather runs once per ubatch, before its consumer layer.
  On HOST it costs microseconds (§8.2). On STORAGE the CPU cost is also small
  (84-170 B copies), and the IO waits happen in the kernel or in blocked
  worker threads. It does not compete with CpuExpertPool's compute workers for cores
  in any measurable way. The gather uses the `ggml-cpu` threadpool of the CPU
  split it belongs to, which already runs between device splits.
- **Budget pressure at runtime.** Tiers are immutable after the plan is
  published. If another load later needs host memory, the lifecycle replan
  (`MID_LOAD_REPLAN`) may re-plan *new* objects. It never demotes a live
  table: that would change a live model's plan under it, and the ownership
  rules forbid reclaiming live leased memory (CLAUDE.md, "SYCL Memory
  Ownership").

### 4.5 What the planner will choose on this host

The formula inputs vary with the host's other tenants: `MemTotal` 246.9 GiB,
and `MemAvailable` 181.8 GiB at the time of the Qwen probes and 171.4 GiB at
the time of the DeepSeek ones. So `host_budget` = min(222.2, `MemAvailable` -
8) = **163-174 GiB**. The two dual-card figures below are estimates, and the
plan log line (§10) is the witness that will check them.

- **Qwen3.8-Flash-Next Q8_0.** VRAM holds ~41 GiB of weights and KV after the
  fixed zones; dense 7 GiB goes first and experts fill the rest, so host
  experts come to ~86 of 119.5 GiB. The leftover is 163-174 - ~86 - host
  zones - reserve, **~27-37 GiB after the 50.66 GiB table**. The table goes to
  **HOST**.
- **DeepSeek-V4.1-Flash Q3_K_M.** The non-engram weights are 244.7 GiB (figure
  from `llama.cpp-9g22`), and ~41 GiB of them fit in VRAM, which leaves
  ~204 GiB for the host against a 163-174 GiB budget, **30-41 GiB short**
  before any table is considered. **Step 2 refuses** (`host-budget-exceeded`).
  No lazy-table tier fixes this; cold experts on STORAGE would, and they are
  the later ticket of §9.2. The owner's primary rung is therefore Q2_K.
- **DeepSeek-V4.1-Flash Q2_K (the owner's primary).** 246.3 GiB in total, of which the tables are
  2 x 384.0 M rows x 84 B = 60.08 GiB, so the non-engram weights are
  ~186.2 GiB and ~145 GiB of them are on the host. The leftover is
  163-174 - ~145 - host KV and zones - reserve, **~10-25 GiB**. Neither
  30.04 GiB table fits, so **both tables are STORAGE**, and they split the
  leftover evenly (equal `table_bytes x rows_per_token`): ~5-12 GiB of row
  cache each. That is past the knee of every simulated corpus (§6.2.2), so
  the cache is not what limits DeepSeek Q2_K on this host. Cold misses are,
  and they are an IOPS question (§8.4, D3).

---

## 5. The STORAGE read path (item 3)

### 5.1 mmap page faults compared with `O_DIRECT` prefetch

Both engines were measured over the table's byte range (Appendix A.2):

| engine | concurrency | rows/s | per-row latency |
|---|---|---:|---|
| mmap faults (upstream path), `MADV_RANDOM` | 1 thread | 1555 | mean 0.64 ms, p99 2.3 ms |
| mmap faults | 16 threads | 7866 | mean 2.0 ms, p99 7.3 ms |
| mmap faults | 24 threads | 7370 | mean 3.3 ms, p99 17.7 ms |
| fio `O_DIRECT` `io_uring` 4 KiB | QD1 | 1389 | p50 0.48 ms, p99 2.7 ms |
| fio `O_DIRECT` `io_uring` 4 KiB | QD16 | 7815 | p50 1.8 ms, p99 6.8 ms |
| fio `O_DIRECT` `io_uring` 4 KiB | QD64 | 9124 | p50 5.7 ms, p99 25 ms |
| pread pool, `O_DIRECT` 4/8 KiB | 64 workers | 8344 | (batch of 7100: 0.85 s) |

**The device ceiling is the same for every engine.** On this filesystem the
bottleneck is the 256 KiB extent bounce onto USB links, not the syscall
mechanism. The choice of engine therefore rests on *timing*, *ownership* and
*accounting*, and on all three `O_DIRECT` prefetch wins:

1. **Timing.** A fault happens inside `get_rows`, on a compute thread, *after*
   the graph reached the node, and at an effective QD of however many
   `ggml-cpu` threads happen to fault at once. A prefetch is issued at
   `set_input`, before graph compute starts, at a QD equal to the number of
   misses, and it overlaps with the graph's earlier work (§5.4).
2. **Ownership (P1/P2).** A fault lands the page in the kernel page cache,
   whose allocator and reclaim policy are the kernel's. `O_DIRECT` lands the
   bytes in a cache-owned `mem_handle`.
3. **Accounting (P4).** Page-cache pages are reclaimable and invisible to the
   planner. Under memory pressure the kernel evicts them, and a table the plan
   calls resident silently becomes a storage read. Worse, a "HOST tier" backed
   by mmap is not HOST at all.
4. **Efficiency.** The page cache holds 4 KiB pages, 24 rows each, when one of
   them is hot. For the same bytes, a 4 KiB-page LRU hits much less than a row
   LRU (§6.2: 19.5% against 34.5% at 32 MiB on wikitext-2 train).

Buffered `pread` is rejected for reasons 2 and 3, since it fills the page
cache too.

### 5.2 IO engine

Two engines implement one interface (D2):

- **Portable default: a pread worker pool.** `QD_max` threads (64 on this
  host) each do blocking `pread(O_DIRECT)` into a block-ring slot. This needs
  no dependency and ports to Windows as `ReadFile` +
  `FILE_FLAG_NO_BUFFERING`. It measured 8.3 k rows/s at 64 workers, at the
  device ceiling (7.8 k on the DeepSeek table). The workers sleep in the
  kernel, so they cost no CPU. One pool per process serves every table and
  context.
- **Linux fast path (later): `io_uring` through raw syscalls.** It needs no
  liburing and adds no threads. It matters when the device is faster than the
  thread pool can drive: the ext4 NVMe reference reaches 110 k IOPS, where
  per-IO thread wake-ups start to show.

The rules both engines follow:

- Every miss becomes one 4 KiB-aligned read of the block(s) that contain the
  row: 4 KiB, or 8 KiB for the rows that straddle a block (4.1% of 170 B rows,
  2.7% of 110 B rows, 2.0% of 84 B rows).
- Reads are deduplicated per ubatch by block.
- **Issue order is consumption order.** The queue is FIFO, and
  `lazy_prefetch()` (§7.3) submits the tables of one ubatch in ascending
  `consumer_layer`. DeepSeek's `blk.1` rows are therefore all queued before any
  `blk.14` row: the table with a one-block window gets the device first, and
  the table with a 13-block window takes the tail. Phase-2 lookahead reads
  (§5.4) queue behind every current-ubatch read.
- `O_DIRECT` alignment comes from `statx(STATX_DIOALIGN)`. If the filesystem
  refuses `O_DIRECT` (for example tmpfs or some FUSE mounts), the fallback is
  buffered `pread` followed by `posix_fadvise(DONTNEED)` on the range. It logs a
  once-per-process WARN that names the P4 cost.

### 5.3 The completion signal, and the only wait

A **ticket** per `(table, context, ubatch)`, so each gather waits only for its
own table's rows:

```
ticket { std::atomic<int> outstanding; futex word / condvar; mem_handle staging_ref; epoch }
```

- `prefetch(rows)` classifies each row (§6.3). A **hit** is copied from the
  row cache into the landing slab *immediately*: one row (84-170 B), done under
  the cache shard's lock. That removes any window in which another context could evict
  it. Each **miss** block is queued to the engine. `outstanding` is the number
  of queued blocks.
- When a block read completes, the reaper (a worker thread, or the io_uring
  CQ reader) copies the row(s) it carries into the landing slab, inserts them
  into the row cache, recycles the ring slot, and decrements `outstanding`.
  The last completion wakes the waiter.
- The gather op (§7) **waits for `outstanding == 0`**. That is the **one wait**
  this design adds. A CPU thread waits for *storage* IO, never for the GPU. It
  appears on the critical path only when the IO takes longer than the time
  from `set_input` to the gather node (§5.4), and the stall is counted (§10).

An existing wait sits next to it, which this design does not add but must
name. The scheduler **synchronizes the previous backend before running a split
that has no inputs** when the backend changes
(`76c7f6548 ggml/src/ggml-backend.cpp:2326-2334`; the same code is in upstream).
A CPU gather split whose only input is a host input tensor has
`n_inputs == 0`, so the host thread waits for the GPU's preceding work (layer 0
for a block-1 consumer, blocks 0-13 for DeepSeek's block-14 consumer) before
running the gather. That is a host wait on the GPU path, and it exists today
at every GPU-to-CPU split boundary (host experts included). It does not cost
the IO overlap, because the reads were issued earlier and proceed in the kernel
meanwhile. What it costs is a **drain**: the host stops running ahead of the
GPU at that point, so the GPU idles for the gather itself (microseconds when
the rows are ready), the activation's H2D copy and the first launch of the
next split. For a gather at block 1 that is a drain after one layer; for
DeepSeek's block-14 gather it is a mid-graph drain once per ubatch. It guards
the reuse of CPU compute-buffer regions against in-flight asynchronous H2D
copies, and replacing it with an event edge is a scheduler change of its own
(D6). §8.4 pre-registers how the drain is measured against the window it buys.

### 5.4 The prefetch window

- **Graph order.** Upstream expands the PLE gather **before** layer 0
  (`694ec2354 src/models/qwen4exp.cpp:381-386`). The scheduler runs splits in
  node order and CPU splits synchronously, so a gather ordered first has to
  finish before layer 0 is even submitted. The IO then overlaps only with
  input setup. The fork moves the expand to **just before the first consumer
  layer** (layer 1 here). The order becomes: CPU `inp_embd` split, GPU layer 0
  submitted asynchronously, then the CPU gather split. The window is then
  `set_input -> graph launch -> layer 0 done`. It is the window the model's own
  design places PLE for. Numerics are unchanged, since independent nodes
  reorder freely. The cost is one more split boundary, and the gathered
  activation needs its H2D copy anyway (D5).
- **Windows per table.**

  | table | issued | needed before | window |
  |---|---|---|---|
  | Qwen `per_layer_token_embd` | `set_input` | block 1 | graph launch + block 0 |
  | DeepSeek `blk.1.engram_embd` | `set_input`, first in the queue | block 1 | graph launch + block 0 |
  | DeepSeek `blk.14.engram_embd` | `set_input`, behind `blk.1` | block 14 | graph launch + blocks 0-13 |

  DeepSeek's two gathers are two custom ops in two CPU splits, each expanded
  just before its consumer block (D5). The alternative, both gathers in one
  split before block 1, avoids the mid-graph drain but shrinks `blk.14`'s
  window to block 0, which at the measured cold rate costs ~2 ms per decode
  token (the 48-row batch at 8.95 ms against the 24-row one at 6.95 ms).
- **Decode.** 16 (Qwen) or 48 (DeepSeek) rows per token, issued at
  `set_input`. The critical window is one block of one token, sub-millisecond
  to a few ms. A token cannot be prefetched earlier, because it is not known
  until it is sampled (MTP speculation, §12).
- **PP.** Every prompt token is known when `llama_decode` is entered. Phase 1
  prefetches each ubatch at its own `set_input`. Phase 2 adds **one-ubatch
  lookahead**: after `set_input(k)`, compute ubatch `k+1`'s row indices and
  prefetch them into the second landing-slab slot. That is why the depth is 2
  in §3.3. With lookahead, the IO of ubatch `k+1` overlaps the *entire*
  compute of ubatch `k`, and the exposed stall per ubatch becomes
  `max(0, IO - compute)`. The indices of ubatch `k+1` need predecessor tokens,
  which are the last `n_gram-1` tokens of ubatch `k` in the same sequence
  (2 for Qwen, 3 for DeepSeek). The model's input class gets a
  `peek_rows(next_ubatch, tail)` hook. The generic path skips lookahead when a
  model does not implement it. For DeepSeek on this host the tables are always
  on STORAGE and a cold ubatch costs ~3 s of IO, so lookahead is what makes
  long prompts tolerable; D9 asks whether it moves into v1.

---

## 6. The hot-row cache (item 4)

### 6.1 Why it works: n-gram Zipf

The row of an n-gram head is `hash(n-gram) mod p_h + offset_h`, so a row is
hot exactly when its n-gram is frequent. Natural-language and code n-grams are
heavy-tailed, and the 8 heads of one n-gram share a key, so they hit or miss
together. The simulation (Appendix A.3) replays each model's **exact hash**
over real text tokenized with that model's own vocabulary.

- **The Qwen hash.** The loop is compiled **verbatim**: `sed` of
  `694ec2354 src/models/qwen4exp.cpp:1067-1125` into a C++ harness. An
  independent Python transcription that reads its constants from the GGUF
  produced **identical** rows (320000 of 320000 compared). Every row index is
  below 320001536.
- **The DeepSeek hash.** `NgramHashState.forward` (§2.1.2) transcribed twice,
  into C++ (`ds_sim.cpp`) and independently into Python with every constant
  read from the GGUF (`ds_hash.py`). They produced **identical** rows (960000
  of 960000, on two corpora). The largest row seen is 384015792, below
  `blk.14`'s 384016682.
- **The Qwen tokenizer.** A pure-Python byte-level BPE with the `qwen35`
  pre-tokenizer regex (`76c7f6548 src/llama-vocab.cpp:382-387`). It is
  validated **50/50** against `models/ggml-vocab-qwen35.gguf.{inp,out}`, then
  run with the tokens and merges of the Qwen3.8 GGUF (248320 tokens). Every
  corpus round-trips byte-exact.
- **The DeepSeek tokenizer.** The same BPE with llama.cpp's multi-regex split
  semantics. The mechanism is validated **46/46** against
  `models/ggml-vocab-deepseek-llm.gguf.{inp,out}` (no test vectors exist for
  `joyai-llm`), then run with the `joyai-llm` regexes
  (`76c7f6548 src/llama-vocab.cpp:318-324`) and the tokens and merges of the
  DeepSeek-V4.1 GGUF (129280 tokens). Every corpus round-trips byte-exact. A
  residual difference between Python's `regex` and llama.cpp's own splitter
  would move a few token boundaries; the hit-rate statistics do not depend on
  that level of detail.
- **The corpora.** wikitext-2 raw (test 297 k tokens; train 2.51 M tokens), the
  fork's C/C++ sources (`src/`, `common/`, `ggml-cpu`, server; 3.42 M tokens),
  the fork's Markdown docs (2.29 M tokens), and a "mixed" stream that
  interleaves 2048-token chunks of train, code and docs to model a server with
  varied requests. Token counts are Qwen's; DeepSeek's vocabulary gives
  288 k / 2.42 M / 3.29 M / 2.24 M / 7.95 M.

### 6.2 Measured hit rates

#### 6.2.1 Qwen PLE

Row-granular hit rate over the whole stream. The cache size is `size / 170 B`
slots, excluding the ~7% index overhead:

| corpus | 32 MiB | 128 MiB | 512 MiB | 2 GiB | 8 GiB | distinct rows (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| wiki test (LRU) | 36.5% | 41.9% | 45.3% | 45.3% | 45.3% | 0.41 |
| wiki train (LRU) | 34.5% | 41.1% | 49.7% | 57.4% | 57.8% | 2.68 |
| wiki train (CLOCK) | 35.1% | 41.8% | 50.4% | 57.5% | 57.8% | |
| wiki train, 4 KiB-page LRU, same bytes | 19.5% | 27.6% | 33.7% | 41.4% | 55.6% | |
| code (LRU) | 86.8% | 89.1% | 90.5% | 90.7% | 90.7% | 0.80 |
| docs (LRU) | 56.3% | 63.0% | 68.2% | 71.4% | 71.4% | 1.66 |
| docs, 4 KiB-page LRU, same bytes | 33.6% | 45.1% | 54.6% | 62.8% | 72.1% | |
| mixed (LRU) | 58.8% | 64.4% | 69.6% | 74.9% | 76.9% | 4.81 |

By n-gram order (wiki train, 2 GiB): bigram heads 72.9%, trigram heads 42.0%.

How the results read:

- **CLOCK matches LRU to within ±0.7 points everywhere.** CLOCK is the policy,
  because it needs one reference bit per slot and no list relinking under a
  lock.
- **The curves saturate once the cache holds the corpus's distinct rows.** At
  that point the remaining misses are compulsory (first sight). A long-running
  server keeps climbing toward the saturation value of its own traffic.
- **The page cache is the wrong granularity.** At small sizes the row cache
  hits 1.3-1.8x as often for the same bytes. The page LRU catches up only at 8
  GiB, and there only by incidental neighbours in the same page.
- **Random tokens do not hit.** Uniform random tokens (the `llama-bench`
  control) never hit beyond hash collisions: 86% of tokens still miss all 16
  rows even with 8 GiB.

The metrics the speed targets need (CLOCK, steady state with the first 25% of
the stream excluded):

| corpus | cache | miss rows / decode token | P(0 misses) | P(16 misses) | unique-miss rows / PP512 ubatch (mean / max) |
|---|---|---:|---:|---:|---|
| random tokens | any | 15.85-16.00 | 0% | 86-100% | 8116-8192 / 8192 |
| wiki test | 0 (cold) | 16.00 | 0% | 100% | 7055 / 7848 |
| wiki test | 512 MiB | 8.47 | 31.9% | 35.3% | 4335 / 5951 |
| wiki train | 512 MiB | 7.75 | 35.0% | 28.0% | 3968 / 5820 |
| wiki train | 2 GiB | 6.29 | 42.6% | 14.5% | 3219 / 5311 |
| code | 0 (cold) | 16.00 | 0% | 100% | 3922 / 7456 |
| code | 512 MiB | 1.33 | 88.2% | 4.3% | 681 / 5419 |
| docs | 512 MiB | 4.58 | 60.5% | 15.7% | 2343 / 5700 |
| docs | 2 GiB | 3.97 | 63.9% | 10.4% | 2032 / 4930 |
| mixed | 512 MiB | 4.45 | 63.0% | 16.3% | 2275 / 6146 |
| mixed | 2 GiB | 3.52 | 67.6% | 7.1% | 1800 / 5186 |

Cold, unique rows per PP512 ubatch are 7028 (wiki test), 7104 (wiki train),
3969 (code), 6231 (docs) and 5562 (mixed) of 8192, so in-ubatch dedup saves
13-14% on prose, 24% on docs and 52% on code.

#### 6.2.2 DeepSeek engram (the harder case)

`ds_sim`: CLOCK over both tables' rows (key = table, row), at the stated
*total* size in 110 B slots. The plan splits the leftover evenly between the
two tables (§4.3 step 5), and one shared cache of the total size is equivalent
in expectation: every n-gram occurrence touches 24 rows in each table, so the
two tables see the same frequency distribution under different hashes.

| corpus | 128 MiB | 512 MiB | 2 GiB | 8 GiB | by order at 2 GiB (2 / 3 / 4-gram) | distinct rows (GiB at 110 B) |
|---|---:|---:|---:|---:|---|---:|
| random tokens | 0.2% | 0.5% | 0.9% | 0.9% | | 1.46 (300 k tokens) |
| wiki test | 28.9% | 33.3% | 34.0% | 34.0% | 59.2 / 28.6 / 14.2 | 0.93 |
| wiki train | 27.6% | 34.0% | 42.0% | 46.3% | 70.3 / 36.8 / 18.9 | 6.41 |
| code | 84.4% | 86.6% | 87.7% | 87.7% | 95.0 / 87.6 / 80.5 | 1.99 |
| docs | 52.3% | 57.5% | 62.1% | 63.4% | 82.9 / 58.8 / 44.6 | 4.04 |
| mixed | 55.1% | 60.0% | 65.1% | 70.2% | 82.7 / 62.3 / 50.5 | 11.45 |

Steady state (the first 25% of the stream excluded):

| corpus | cache (total) | miss rows / decode token (of 48) | P(0 misses) | P(48 misses) | unique-miss rows / PP512 ubatch (mean / max) |
|---|---|---:|---:|---:|---|
| random tokens | 0 (cold) | 48.00 | 0% | 100% | 24576 / 24576 |
| random tokens | 128 MiB - 8 GiB | 47.44-47.92 | 0% | 58-93% | 24290-24537 / 24553 |
| wiki test | 0 (cold) | 48.00 | 0% | 100% | 22180 / 23920 |
| wiki test | 512 MiB | 31.42 | 13.5% | 30.0% | 16092 / 20438 |
| wiki test | 2 GiB | 30.93 | 14.0% | 27.0% | 15837 / 20438 |
| wiki train | 512 MiB | 31.43 | 13.0% | 28.6% | 16091 / 20293 |
| wiki train | 2 GiB | 26.93 | 17.3% | 7.9% | 13790 / 18564 |
| wiki train | 8 GiB | 24.20 | 20.0% | 2.4% | 12389 / 18564 |
| code | 0 (cold) | 48.00 | 0% | 100% | 12964 / 22511 |
| code | 512 MiB | 5.91 | 80.5% | 3.9% | 3026 / 17368 |
| code | 2 GiB | 5.22 | 81.8% | 1.8% | 2674 / 17006 |
| docs | 512 MiB | 18.89 | 44.0% | 14.7% | 9669 / 19637 |
| docs | 2 GiB | 16.33 | 46.7% | 4.2% | 8368 / 17938 |
| docs | 8 GiB | 15.52 | 47.8% | 2.6% | 7945 / 16614 |
| mixed | 0 (cold) | 48.00 | 0% | 100% | 17452 / 24464 |
| mixed | 512 MiB | 17.97 | 49.1% | 15.9% | 9200 / 20540 |
| mixed | 2 GiB | 15.46 | 51.9% | 4.5% | 7916 / 18649 |
| mixed | 8 GiB | 12.43 | 54.9% | 0.2% | 6370 / 16572 |

Cold, unique rows per PP512 ubatch are 22117 (wiki test), 22320 (wiki train),
13114 (code), 19584 (docs) and 17762 (mixed) of 24576.

Why DeepSeek is harder, beyond having 3x the rows:

- **4-grams are sparse.** The order-4 heads hit 14% on prose at 2 GiB,
  against 59% for bigrams. A third of every token's rows are 4-gram rows.
- **A zero-stall decode token needs all 48 rows to hit.** On prose that
  happens for 14% of tokens, against 32-43% for Qwen's 16.
- **A PP512 ubatch misses 3-4x as many unique rows** at the same cache size:
  prose ~16 k against ~4.3 k, code ~3.0 k against ~0.7 k.

Prose keeps gaining up to 8 GiB (wiki train: 34% at 512 MiB, 46% at 8 GiB),
and the distinct-row sets are 0.9-11.5 GiB. On the owner's primary rung, Q2_K,
the plan gives the two tables ~10-25 GiB of cache on this host (§4.5), past
the knee of every corpus. So on DeepSeek the cache size is not the limiting
factor; the compulsory misses, and the IOPS that serve them, are. The 8 GiB
column is the one that describes Q2_K here.

### 6.3 Structure

- **Keying.** The key is the row index (int64 for generality), inside the
  table's own cache. There is no special n-gram grouping: the 8 heads of one
  n-gram are 8 separate keys that are touched together, which CLOCK already
  handles, and the generic path stays table-agnostic (§9).
- **Layout.** Set-associative, open-addressed, sharded by
  `hash(row) mod n_shards` with one mutex per shard. It is a leaf lock, **L5**
  in canonical contract §12.5 (added to its inventory by the implementation
  ticket), and is never held across an IO wait or a callback.
- **Contents.** Raw row bytes in the table's own type (170 B Q8_0, 110 B Q3_K,
  84 B Q2_K), not F32. Dequant happens in the gather (§7). An F32 cache would
  be 3.8x (Q8_0) to 12x (Q2_K) larger for the same rows.
- **Insertion.** A block completion inserts the row it carried. A row is
  evicted only by CLOCK. Nothing needs pinning: the landing slab already holds
  every row the pending gather needs (§5.3).
- **Warm start.** None in v1. The measured cold numbers are the contract. D1
  covers background prefill for the HOST tier.

### 6.4 Size

The planner provides `cache_slots` (§4.3 step 4). The floor is one
double-buffered ubatch working set at `n_ubatch = 512`: 2.66 MiB for Qwen's
table, 2.58 MiB for each DeepSeek table (2 x 512 x 24 x 110 B; 1.97 MiB at
Q2_K). The simulations flatten by ~2 GiB on Qwen and 2-8 GiB on DeepSeek for
2-8 M-token corpora, but a server's distinct-row set keeps growing, so the
recommendation is to **give the cache the whole leftover** (D7, option a).
For the harder case that means ~5-12 GiB per table on DeepSeek Q2_K, past the
knee of every simulated corpus. The planner logs the size it chose.

---

## 7. The executor (item 5)

### 7.1 The graph

Qwen's `build_inp_ple`, DeepSeek's engram block input, and every other
consumer of a lazy tensor (§9) call a new graph helper:

```
ggml_tensor * llm_graph_context::build_lazy_get_rows(ggml_tensor * lazy, ggml_tensor * rows);
```

- **Without SYCL hooks**, it returns `ggml_get_rows(lazy, rows)`: the upstream
  path, with the mmap and page faults, unchanged.
- **When the table is a SYCL `lazy_table`**, it returns
  `ggml_custom_4d(ctx, GGML_TYPE_F32, ne0 = row elements, ne1 = n_rows_requested, 1, 1, {rows}, 1, lazy_get_rows_cpu, GGML_N_TASKS_MAX, gather_ctx)`.
  The result has the shape `ggml_get_rows` produces, so each model's reshape
  (Qwen `[2560, n_tokens]`, DeepSeek `[6144, n_tokens]` per table) and
  everything downstream are untouched. `gather_ctx` is a per-context,
  per-table object that holds the proc-address entry points into the SYCL
  backend and the current ticket. SYCL declines
  `GGML_OP_CUSTOM` (§2.3), so the scheduler places the node on the **CPU**.
- **The callback.** Each of the `nth` threads waits for the ticket (§5.3), then
  dequantizes its share of rows from the landing slab into `dst` through the
  type's `to_float` trait (`ggml_get_type_traits`). No thread reads the table
  or the row cache: `prefetch` has already put every row in the slab.

### 7.2 Data movement: the gathered activation only

The scheduler copies the custom op's F32 output to the GPU as an ordinary split
input: for Qwen `[2560 x n_tokens]`, 10 KiB per token and 5 MiB per PP512
ubatch; for DeepSeek `[6144 x n_tokens]` per table, 24 KiB per token per table
and 12 MiB per PP512 ubatch per table (~1 ms of PCIe, against ~1.4 s of cold
IO for that table's rows). It is an
**activation**, the same kind of data as every other output of a CPU split
that feeds a GPU split, such as a host-resident token-embedding gather or a
host-tiered layer's output. The table itself never crosses PCIe. This is
**placement-consistent (P3)**:

- The data is host-resident, so the CPU executes the op.
- The GPU never dereferences host memory (no zero-copy).
- No weight bytes are staged to the device (no streaming): the device receives
  the *result* of a CPU op, just as it does at every CPU-to-GPU split today.
- **Layout follows residency:** the table is consumed in its materialized
  CPU layout (AOS rows of its GGUF type). For a gather, AOS row-major *is* the
  CPU-optimal layout, since each row is one contiguous run of 84-170 B. No
  repack is needed, and none is advertised.

**Why F32 and not the raw rows.** Shipping the gathered Q3_K or Q8_0 blocks
and dequantizing them on the device would be cheaper on PCIe (110 B instead of
1 KiB per DeepSeek row). It would also split one op across two executors: the
GPU would execute part of `get_rows` on per-dispatch copies of weight bytes.
That is the weight-streaming shape P3 forbids, even when the bytes are a
gathered subset, and the owner's ruling for this ticket is "the CPU executes
the gather". The PCIe cost it would save is ~1 ms per ubatch against seconds
of IO. An F16 output is a later option if the consumer takes F16 activations;
the CPU still executes the op.

### 7.3 Where the prefetch hook sits

`llm_graph_result` records every `build_lazy_get_rows` call as
`{gather_ctx, rows_input_tensor, consumer_layer}`. After
`llm_graph_result::set_inputs()` has run every input's `set_input`,
`llama_context` calls `res->lazy_prefetch()`. For each record, in ascending
`consumer_layer` (§5.2), that function reads the row indices from the rows
input tensor, which is a CPU-buffer input because its only consumer is a CPU
node, so the host pointer is valid. It then calls `prefetch`. This keeps model code
untouched apart from `build_lazy_get_rows`. Lookahead (§5.4) is the one
per-model hook.

---

## 8. Pre-registered speed targets (item 6)

Every target below is derived from the numbers measured in §2.4, §5.1 and §6.2
on this host, bcachefs as-is, under ambient load. The implementer measures
against them with the counters in §10. The lead runs the GPU parts. Qwen's
targets are §8.2-8.3; DeepSeek's, the harder case and the one that sizes the
engine, are §8.4.

### 8.1 Arms and instruments

- **Arms.**
  - **H**: HOST tier, the plan's natural choice for Qwen here. DeepSeek has
    no H arm on this host: its tables cannot be HOST alongside its backbone
    (§4.5).
  - **S**: STORAGE tier. The planner is forced into it by a *test-only* host
    budget override, `GGML_SYCL_PRIVATE_TESTING` builds only, never an
    end-user variable.
  - **S-cold**: a fresh load with an empty cache.
  - **S-warm**: steady state after at least 25% of a same-domain stream.
  - **S-hot**: the same text replayed with a cache at least as large as its
    distinct rows.
- **Instruments.**
  - `T_gather`: time inside the custom op, per ubatch.
  - `T_stall`: the part of `T_gather` spent waiting on the ticket.
  - The IO count and bytes, and the hit count (§10).
  - End-to-end: `llama-bench` (cold by construction, §8.5) and
    `llama-perplexity` on wikitext-2 test (warm).
  - **Engine rate**, the storage-side ceiling every S target leans on: miss
    rows per second over a cold PP512 ubatch, `rows_miss / stall-window`,
    from the counters. It must reach **>= 7.5 k rows/s** on the Qwen table and
    **>= 7.0 k rows/s** on the DeepSeek table (90% of the pread pool's measured
    8.3 k and 7.8 k, which are themselves at fio's QD64 ceiling). A shortfall
    here is an engine defect; every S target below it is then void, not missed.

### 8.2 Qwen decode

| arm | target | derived from |
|---|---|---|
| H | `T_gather` <= **20 us/token**; tg128 within **1%** of an `--lazy-mode off` arm (the table as an ordinary CPU weight) | measured 4.2 us for 16 rows including dequant |
| S-cold | `T_stall` mean <= **4 ms/token**, p99 <= **15 ms** | measured batch of 16: mean 3.75 ms, p99 13.0 ms, before any layer-0 overlap is subtracted |
| S-warm | mean `T_stall` <= **(1 - P0) x 4 ms**. P0 is the steady-state zero-miss share: prose 32-43%, docs 61-64%, code 88%, mixed 63-68%. The runtime hit rate must match the simulation for the same corpus to within **±5 points** | §6.2 |
| S-hot | `T_stall == 0` on **>= 99%** of tokens | every row hits |

Expressed as throughput: `tg_S >= 1 / (1/tg_H + T_stall)`. For example, if H
decodes at 20 tok/s (50 ms per token), S-cold must decode at **>= 18.5
tok/s**.

### 8.3 Qwen PP512

| arm | target | derived from |
|---|---|---|
| H | `T_gather` <= **3 ms/ubatch**; pp512 within **1%** of `--lazy-mode off` | measured ~2 ms for 8192 rows |
| S-cold (real text) | `T_stall` <= **1.0 s/ubatch** | measured 0.85 s for 7100 rows (wiki cold: 7028-7104 unique) |
| S-cold (`llama-bench`, random) | `T_stall` <= **1.15 s/ubatch** | 8116-8192 unique rows at the measured 8.3 k rows/s = 0.98 s |
| S-warm | `T_stall` <= measured-rate x unique misses: prose (512 MiB) 4335 rows -> **<= 0.6 s**; docs 2343 -> **<= 0.33 s**; code 681 -> **<= 0.1 s** | §6.2 x 8.3 k rows/s |
| S-hot | `T_stall == 0` | |
| phase 2 lookahead, multi-ubatch prompt | stall of ubatch `k >= 2` <= `max(0, IO(k) - compute(k-1)) + 10%` | §5.4 |

Expressed as throughput: `pp_S >= 512 / (512/pp_H + T_stall)`.

### 8.4 DeepSeek (the harder case)

Two gathers per ubatch, `blk.1` (window: block 0) and `blk.14` (window: blocks
0-13). `T_stall(1)` and `T_stall(14)` are their ticket waits. `T(a..b)` is the
measured GPU-and-host time of blocks `a..b` for the same ubatch, from the op
timing the lead already collects. `U` is the unique miss rows of the ubatch,
both tables together. The IO was measured on the Q3_K_M file (110 B rows),
the only rung present; the owner's primary is Q2_K (84 B rows). A miss costs
one 4 KiB read at either size (8 KiB for the 2.7% or 2.0% of rows that
straddle a block), so the targets carry over; T8 re-measures on the Q2_K file.

**Decode** (48 rows per token):

| arm | target | derived from |
|---|---|---|
| S-cold, `blk.1` | `T_stall(1)` mean <= **8 ms**, p99 <= **30 ms** | measured 24-row batch: mean 6.95 ms, p99 27.8 ms, with no overlap subtracted |
| S-cold, `blk.14` | `T_stall(14)` mean <= **max(0, 10 ms - T(1..13))**, p99 <= **max(0, 32 ms - T(1..13))** | measured 48-row batch: mean 8.95 ms, p99 29.1 ms; its reads queue behind `blk.1`'s (§5.2) |
| S-cold, per token | `T_stall(1) + T_stall(14)` mean <= **10 ms** | the 48-row batch bounds both |
| S-warm (Q2_K placement) | mean `T_stall(1)` <= **(1 - P0) x 8 ms**, where P0 is the both-tables zero-miss share at the 8 GiB rows (conservative for one table): prose (wiki train) 20%, docs 48%, code 82%, mixed 55%. The runtime hit rate must match §6.2.2 to within **±5 points** | §6.2.2 |
| S-hot | `T_stall == 0` on **>= 99%** of tokens | |

For example, if DeepSeek decodes at 10 tok/s from warm host experts (100 ms
per token), S-cold must decode at **>= 9.1 tok/s**.

**PP512** (the targets carry the same 15% headroom over the measured 7.8 k
rows/s as Qwen's do over 8.3 k):

| arm | target | derived from |
|---|---|---|
| S-cold (`llama-bench`, random) | `T_stall(1) + T_stall(14)` <= **3.6 s/ubatch**; `T_stall(1)` <= **1.8 s** | 24576 unique rows / 7.8 k = 3.15 s; half of them are `blk.1`'s |
| S-cold (prose) | <= **3.3 s**; `T_stall(1)` <= **1.65 s** | 22117 unique rows (wiki test) |
| S-cold (code) | <= **1.95 s**; `T_stall(1)` <= **1.0 s** | 13114 unique rows |
| any S arm, `blk.14` | `T_stall(14)` <= **max(0, 1.15 x U / 7.8 k - T_stall(1) - T(1..13))** | the `blk.14` reads finish during blocks 1-13 whenever they take longer than the rest of the IO |
| S-warm (Q2_K placement: >= 8 GiB of cache in total) | `T_stall(1) + T_stall(14)` <= 1.15 x U / 7.8 k at the 8 GiB rows: prose (wiki train) 12389 -> **<= 1.85 s**; docs 7945 -> **<= 1.2 s**; code 2674 -> **<= 0.4 s**; mixed 6370 -> **<= 0.95 s** | §6.2.2 |
| S-hot | `T_stall == 0` | |
| phase 2 lookahead | as §8.3 | |

**The drain, pre-registered (D5).** Arm A expands each gather before its
consumer block, as recommended. Arm B expands both before block 1, which
removes the mid-graph CPU split and its drain (§5.3) but shrinks `blk.14`'s
window to block 0. The prediction: A wins cold and warm prose (up to ~2 ms per
token, the 48-row against the 24-row batch), and B wins a fully hot cache by
the drain's cost, expected under 0.1 ms per token. Decision rule, fixed now:
**keep A iff tg(A) >= tg(B) on S-warm prose**; report code and S-hot too.

### 8.5 Two traps the targets must not fall into

- **`llama-bench` prompts are uniformly random tokens** (`std::rand() % n_vocab`,
  `694ec2354 tools/llama-bench/llama-bench.cpp:2181-2183`, `:2204-2213`). Their
  n-grams never repeat, so on STORAGE **every** `llama-bench` pp and tg number
  is the cold case, whatever the cache size. A "warm" or "hot" claim needs
  real text: `llama-perplexity`, or a fixed `llama-completion` prompt run
  twice. Do not read a flat `llama-bench` curve across cache sizes as "the
  cache does nothing".
- **Filesystem changes move every S number by an order of magnitude.** The
  ext4-on-NVMe reference measured 110 k IOPS at QD64 and 83 k at QD16. On a
  file without per-256-KiB checksums on an NVMe target (D3 option b), Qwen's
  S-cold PP512 projects to **~75-100 ms/ubatch** instead of ~0.85-1 s, and
  S-cold decode to about **1 ms**. DeepSeek's projects to **~0.25-0.3 s** per
  cold ubatch instead of ~3 s. These projections are *not* targets until
  re-measured on the actual file.

---

## 9. Generalisation to any `TENSOR_READ_LAZY` tensor (item 7)

Nothing above is specific to one model except three arch-supplied facts:
`rows_per_token` (for the planner's floor and staging size), `consumer_layer`
(for IO order and the window) and, optionally, `peek_rows` (for lookahead).
Two consumers with different geometry already exercise it: one Q8_0 table read
16 rows per token at block 1, and two Q3_K/Q2_K tables read 24 rows per token
each at blocks 1 and 14.

- **Any row table.** Type, `ne[0]` and `row_size = ggml_row_size(type, ne[0])`
  come from the tensor. Quantized rows are whole blocks by construction.
  Dequant goes through the type traits.
- **Consumers.** Every consumer of a lazy tensor must go through
  `build_lazy_get_rows`. That covers an indexed `get_rows` (qwen4exp's hashed
  rows; gemma4's `inp->tokens`, `694ec2354 src/models/gemma4.cpp:450`) and a
  constant-row view. DeepSeek's two engram gathers are indexed `get_rows`.
  gemma4's `ggml_view_1d(..., row 0)` (`:462`) becomes
  `build_lazy_get_rows` with a constant one-row index tensor. Graph build
  **asserts** that a lazy tensor has no other consumer (a `GGML_ASSERT` in
  `llm_graph_context` over `lazy->` uses). An unhandled consumer would read
  `tensor->data == nullptr`, and failing at build time names it.
- **Several tables.** Each gets its own `lazy_table`, its own tier decision, and
  a share of the leftover host budget (§4.3 step 5). DeepSeek is the first
  two-table consumer.
- **Non-SYCL builds keep upstream's mmap path verbatim.** The divergence is
  confined to the loader's SYCL branch and to `build_lazy_get_rows`.

### 9.1 The consumer contract

This is what a model implements to get the tiers, the cache, the prefetch and
the accounting. It was agreed with the `deepseek41` runtime design
(`llama.cpp-9g22`, `docs/design/deepseek41-runtime.md`), which states the same
contract from the consumer's side.

1. **Mark and declare at load.** Create the table with `TENSOR_READ_LAZY`.
   In `load_tensors`, before the plan is computed, declare its
   `rows_per_token` and `consumer_layer` through the loader
   (`ml.lazy_table_info(tensor, rows_per_token, consumer_layer)`). They feed
   the inventory (§4.1). A lazy tensor without a declaration is refused at
   load, which names it, rather than planned with a guessed floor.
2. **One entry point.** Every read of the table goes through
   `build_lazy_get_rows(lazy, rows)`. Graph build asserts that no other op
   reads a lazy tensor, because under SYCL its `tensor->data` is null.
3. **The rows input.** `rows` is an I32 graph input of
   `rows_per_token x n_tokens` elements (DeepSeek: 24 per token per table),
   fully written in `set_input` from host state only, with no dependency on
   any device result. Every value is in `[0, ne[1])`; `prefetch` aborts on an
   out-of-range id rather than read past the table. Padding, sequence starts
   and blocking tokens are the model's hash's business. int32 ids suffice
   while tables have fewer than 2^31 rows (DeepSeek: 384.0 M).
4. **The output.** F32 `[ne[0], rows_per_token x n_tokens]`, the shape
   `ggml_get_rows` returns; the model reshapes it (DeepSeek: to
   `[6144, n_tokens]` per table). It is an activation, produced on the CPU and
   copied to the device by the scheduler (§7.2). Raw blocks are not offered.
5. **Placement in the graph.** Expand each gather immediately before its
   consumer block (D5): DeepSeek's `blk.1` gather before block 1, and its
   `blk.14` gather before block 14.
6. **Optional lookahead.** `peek_rows(next_ubatch, tail)` returns the next
   ubatch's rows from the model's sequence state, so the IO of ubatch `k+1`
   overlaps the compute of ubatch `k` (§5.4). Batch-level or split-time row
   ids are welcome as the source; the provider bounds lookahead to depth 2,
   which is what the double landing slab holds.

The SYCL side of the same interface:

| piece | what it is | owner |
|---|---|---|
| `lazy_table` | model-scoped unified-cache object, keyed by the weight's `ggml_sycl_cache_id` (model token + tensor name); llama holds only an opaque ref from a proc-address hook, never a row pointer | the unified cache, under the model token (§3.2) |
| `lazy_gather_ctx` | per context, per table: the landing slab, the current ticket | its `mem_handle`s, released in its destructor |
| block ring | per context, shared by its tables | a `mem_handle`; every in-flight ticket holds a copy (§3.2) |
| `prefetch(gather_ctx, rows, n, epoch) -> ticket` | called by `lazy_prefetch()` after `set_inputs()` | |
| the one wait | inside the custom op's callback, on that table's ticket (§5.3) | |

### 9.2 Hosting cold routed experts later (not phase 1)

**Status.** On 2026-09-26 the owner first ruled that DeepSeek-V4.1-Flash runs
Q3_K_M with its coldest routed experts on STORAGE, then revised it the same
day: the **primary is Q2_K**, whose routed experts fit VRAM plus host, so only
the engram tables use STORAGE (§4.5). Cold routed experts on STORAGE feeding
the CPU executor (for Q3_K_M) become a **later ticket that `llama.cpp-9g22`
files**. This document does not design or gate that path. It does two things
for it: it keeps the tier's interface generic enough to host it, and it
records the measurements already taken, so the later ticket starts from facts.

#### 9.2.1 What the interface keeps open

- **A planned, cache-owned `O_DIRECT` landing slab.** The engine of §5.2
  reads into `mem_handle`-owned `HOST` memory whose size the planner charges
  (§3.3). Its read unit is an (fd, offset, length, destination, ticket)
  request, not a row: a 4 KiB row block and a 512 KiB slice chunk are the same
  call. The later path adds a new slab role and a slot-granular cache, not a
  new engine.
- **A hotness ranking by expected reads per step.** The planner's inputs are
  per-object sizes plus an expected-reads-per-step score. For a lazy table it
  is `rows_per_token` (§4.1). For an expert-layer it would be its expected uses
  per step, from routing counts. §4.3's ordering, "charge the score-weighted
  cost, place the coldest last", then covers both. Phase 1 implements only the
  table case.
- **The CPU executor already takes a lease.** `cpu_expert_task` carries a
  `weight_lease` `mem_handle` (`76c7f6548 ggml/src/ggml-sycl/cpu-dispatch.hpp:127`)
  and the host AOS executor consumes file-layout blocks
  (`:136-149`), so a cache slot filled from STORAGE can be executed as a host
  expert is today. P3 then holds as it does for lazy tables: the CPU executes,
  and only the expert's output reaches the GPU.

#### 9.2.2 Facts recorded for the later ticket

| fact | value |
|---|---|
| one expert of one layer, Q3_K_M | **16.77 MB**: gate and up Q3_K 5068800 B each, down Q4_K 6635520 B; **18.25 MB** at layers 0-1 (down Q5_K 8110080 B) |
| where its bytes are | three contiguous slices, one in each `ffn_{gate,up,down}_exps`, ~2 GB apart in one shard; expert `x` starts at `data_offset + x * slice`; slices are not 4 KiB-aligned (5068800 = 1237.5 x 4096) |
| one expert-layer, 16 x 512 KiB chunks in flight, fresh DeepSeek shards | **14.1 ms mean, 29.1 ms p99, 1.2 GB/s**; about 70% of the bytes came from the NVMe, because the shards were minutes old |
| the same, 2 / 6 / 12 loads concurrently | 22.5 / 122 / 222 ms per load; 1.5 / 0.83 / 0.91 GB/s aggregate |
| the same shapes on a settled Qwen split (mostly on the T7) | **0.29 GB/s** (1 load) to **0.47 GB/s** (6 loads); the T7's 5 Gb/s link is the cap |
| reading promotes | `promote_target=nvme` wrote 2.7 GB to the NVMe during a 12 s settled read, so repeatedly read extents become NVMe-cached |
| PP coverage | a PP512 ubatch uses 99.97% of a layer's 384 experts (1 - (1 - 6/384)^512), so PP reads the whole cold set once per ubatch |
| static prior | none available: `exp_probs_b` is DeepSeek-V3-style load balancing, which equalises expert load in training; `hash_layer_count = 0` |
| decode cost model | `240 x c x (1 - r) x 16.8 MB / BW` per token, where `c` is the cold share of expert uses and `r` the share a RAM cache serves |

Raw logs are in Appendix A.2 (`expert_probe`).

#### 9.2.3 DSpark verify steps

`llama.cpp-9g22` reports that DSpark verify batches carry up to 6 tokens (the
anchor plus 5 drafts). So a decode step issues up to 6 x 48 = 288 engram rows,
and rows for rejected drafts are wasted reads; the history rolls back, so
there is no correctness impact. Measured on the `blk.1` extent: a 288-row
batch takes **74 ms mean, 131 ms p99**, and a 144-row batch 24 / 52 ms (64
workers; Appendix A.2), against 8.95 ms for one token's 48. On a cold cache a
verify step therefore costs ~8x a plain token's engram IO, for at most 6
accepted tokens. The row cache serves repeated n-grams in the drafts exactly
as in PP. The §8.4 decode targets assume batch-of-1; the DSpark arm gets its
own targets from this measurement when DSpark lands.

---

## 10. Plan == reality: witnesses and counters

- **Plan log, WARN level** (default verbosity drops INFO, CLAUDE.md
  "`llama-bench` traps"):

  ```
  [PLACEMENT] lazy per_layer_token_embd.weight tier=HOST table=51880.0 MiB resident=51880.0 MiB host_budget=177971.2 MiB host_planned=... leftover=...
  [PLACEMENT] lazy blk.1.engram_embd.weight tier=STORAGE table=30762.6 MiB cache=... MiB (N slots) floor=2.0 MiB staging/ctx=... consumer_layer=1 rows_per_token=24 file=<split> off=...
  [PLACEMENT] lazy blk.14.engram_embd.weight tier=STORAGE table=30763.4 MiB cache=... MiB (N slots) floor=2.0 MiB staging/ctx=... consumer_layer=14 rows_per_token=24 file=<split> off=...
  ```

  (The DeepSeek lines are the Q2_K rung: 84 B rows. The `host_budget`
  figure above is illustrative.)

- **Activation check.** When the `lazy_table` is materialized, it sums its
  handles' sizes and compares them with `host_resident_bytes` from the plan. A
  mismatch is a `GGML_LOG_ERROR` plus a typed refusal: plan != reality is a
  defect, never tolerated.
- **Counters,** printed per context at free and exposed to tests:
  `rows_requested`, `rows_hit`, `rows_miss`, `blocks_read`, `bytes_read`,
  `stall_ns_total`, `stall_ns_max`, `gather_ns_total`, `ubatches`, per table.
  `rows_hit + rows_miss == rows_requested` is asserted.
- **Positive control for the witness.** The test-only STORAGE override arm
  must show `tier=STORAGE` and `rows_miss > 0`. The H arm must show
  `blocks_read == 0`. A run that shows neither proves nothing: an absence of
  work looks exactly like success.

---

## 11. Proposed implementation tickets (filed after review)

| # | ticket | depends on | notes |
|---|---|---|---|
| T1 | `alloc_tier::HOST` + `STORAGE`; pageable raw wrapper; roles `LAZY_ROW_CACHE`/`LAZY_STAGING`; contract §3/§4/§12.5 edits | - | header change first: every other ticket includes it |
| T2 | planner host budget input + shared budget function + host-budget refusal | T1 | closes the pre-existing P4 gap for host experts too |
| T3 | `placement_lazy_table_info` inventory class + tier decision + plan log | T2 | |
| T4 | `lazy_table` object: HOST body load (D1), STORAGE fd, row cache (CLOCK, sharded), activation check | T1, T3 | |
| T5 | IO engine: pread pool + tickets + block ring + landing slab | T4 | |
| T6 | llama side: loader SYCL branch for `TENSOR_READ_LAZY`, `lazy_table_info` declaration, `build_lazy_get_rows` + custom op + `lazy_prefetch()` (consumption order) + graph-order move (D5) + consumer assert; qwen4exp and gemma4 declarations | T4, `llama.cpp-n77l` | fork-local change to upstream model files is a few lines per model; `deepseek41` follows the §9.1 contract in `llama.cpp-9g22`'s own tickets |
| T7 | counters + test-only STORAGE override + CPU-only unit tests (both hash harnesses from Appendix B as fixtures) | T5, T6 | |
| T8 | lead-run acceptance against §8: Qwen (H, S-cold, S-warm, S-hot); DeepSeek (S arms and the §8.4 drain rule) once the `deepseek41` runtime runs | T7 | GPU work: lead only |
| T9 | `peek_rows` lookahead | T8 (Qwen); before T8's DeepSeek arms if D9 (a) | |
| T10 | optional: `io_uring` engine; sched event edge instead of the §5.3 sync (D6) | T8 | |
| (later) | cold routed experts on STORAGE for Q3_K_M (§9.2) | T5 | filed by `llama.cpp-9g22`, not part of this design |

---

## 12. Not done, and why

- **MTP-speculative prefetch.** The MTP head could predict token `t+1`, and
  its rows could be prefetched during token `t`. That turns decode's cold stall
  into a speculation-hit question. It is worth a ticket once S-cold decode is
  measured, not before.
- **A static hot set shipped with the model.** It would need a corpus-derived
  frequency list per model. The CLOCK cache reaches the same place after a few
  thousand tokens.
- **Caching dequantized F32 rows.** That would take 3.8x the memory to save a
  ~2 ms dequant per PP ubatch.
- **Placing the table on the GPU.** It is forbidden by P3, and pointless: 84-170
  B random rows are a CPU-shaped access.
- **Cold routed experts on STORAGE.** A later ticket (§9.2), filed by
  `llama.cpp-9g22`; this design keeps the interface open and records the
  measurements.

---

## 13. Decisions for the owner

Each open decision offers options, and the recommended one is marked
**(rec)**. The recommendations are proposals from this design; only D8 is an
owner ruling.

**D1. How is the HOST tier populated?**
- a) Eager: a sequential `O_DIRECT` read of the whole table at load, ~31 s at
  the measured 1.78 GB/s, blocking model load.
- b) **(rec)** Background: the load returns immediately, and the table fills
  by a background sequential read. Until a row's chunk has arrived, it is
  served through the STORAGE path (misses), so the first minute behaves like
  S-warm and then becomes H. The plan charges the full table from the start
  (P4 is about the reservation, not the fill).
- c) Pure on-demand: never prefill. The table converges to its working set,
  but a HOST-tier table would then never become the H arm.

**D2. IO engine?**
- a) **(rec)** A portable `pread` worker pool first. It measured at the device
  ceiling and adds no dependency.
- b) `io_uring` through raw syscalls only.
- c) Add a liburing dependency.

**D3. The table files on bcachefs.** This is the owner's filesystem, and
nothing here changes it.
- a) **(rec for Qwen)** Leave it. On this host the planner picks HOST for
  Qwen, so its STORAGE speed is a fallback's speed.
- b) **(rec for DeepSeek, once its runtime runs)** STORAGE is DeepSeek's
  operating point here, and every cold number in §8.4 is the filesystem's.
  Set per-file options on the shards that hold the engram tables
  (`foreground_target`/`background_target`/`promote_target=nvme`, so the user
  copy lives on the NVMe, plus `data_checksum=none` or, if bcachefs allows it
  per file, which is unverified, a smaller checksum granularity) and rewrite
  the files. That removes the 60x bounce and the USB links. The NVMe
  (512 GB, of which ~187 GiB is evictable cached data today) has room for
  both models' tables. The projected gain is ~10x in IOPS, to be re-measured
  before any target moves. `data_checksum=none` gives up integrity checking
  on those files, which is the owner's trade.
- c) Move the files to the root ext4 NVMe. **Not feasible:** 36 GiB free,
  against 50.7 GiB for Qwen's split and 30-40 GiB per DeepSeek table.

**D4. `--lazy-mode` under SYCL?**
- a) **(rec)** Keep upstream's meaning as *eligibility* (`OFF` = ordinary
  weight, `AUTO` = marked and larger than 4 GiB, `ON` = all marked). The
  planner alone picks the HOST or STORAGE tier for eligible tensors. There is
  no new flag.
- b) Also let `OFF` force the HOST tier and refuse STORAGE.
- c) Ignore the flag under SYCL.

**D5. Where is each gather expanded?**
- a) **(rec)** Generically, just before its consumer block: Qwen's before
  block 1 (upstream expands it before block 0), DeepSeek's before blocks 1
  and 14. The IO overlaps the preceding blocks. It costs one split boundary
  per gather, and for DeepSeek's block 14 a mid-graph drain (§5.3); §8.4
  pre-registers the rule that keeps or drops that choice.
- b) All gathers before the first block (upstream's order for Qwen). The IO
  overlaps only `set_input` and graph launch.

**D6. The scheduler's sync before a zero-input split (§5.3).**
- a) **(rec)** Accept it in v1 as the named pre-existing wait. For a gather
  before block 1 it costs microseconds, because the IO is already in flight;
  for DeepSeek's block-14 gather it is a mid-graph drain whose cost §8.4
  measures. Ticket T10 replaces it with an event edge.
- b) Block this design on the scheduler change.

**D7. Size of the STORAGE row cache.**
- a) **(rec)** All of the leftover host budget after the reserve, split between
  tables by `table_bytes x rows_per_token`.
- b) Cap it, for example at 8 GiB. The simulated curves flatten by 2-8 GiB,
  but a long-running server's distinct-row set keeps growing.
- c) A fixed fraction of the leftover.

**D8. DeepSeek-V4.1-Flash rung. DECIDED by the owner (2026-09-26, revised
21:00):** Q2_K is the primary; its experts fit and only the engram tables use
STORAGE. Cold routed experts on STORAGE (for Q3_K_M) are a later
`llama.cpp-9g22` ticket (§9.2).

**D9. Lookahead in v1 for STORAGE-bound models?**
- a) **(rec)** Yes for DeepSeek: land T9 before T8's DeepSeek arms. Its
  tables are always on STORAGE here, and a cold PP512 ubatch is ~3 s of IO
  that, without lookahead, only `blk.14`'s share can hide.
- b) Keep lookahead as phase 2 for every model.

---

## Appendix A. Raw measurements (2026-09-26, this host)

Probe sources and logs are in the session scratch
(`<scratchpad>/th32/{io_probe.sh,mmap_probe.cpp,batch_probe.cpp,ram_gather.cpp,qtok.py,ple_hash.py,ple_sim.cpp,ple_sim2.cpp,upstream_set_input_body.inc}`
and `io-logs/`, `sim-logs/`). That scratch is not durable, so the load-bearing
lines are reproduced verbatim below.

### A.1 GGUF metadata (gguf-py under `/usr/bin/python3`)

```
qwen4exp.block_count [48]
qwen4exp.expert_count [512]
qwen4exp.expert_used_count [10]
qwen4exp.ple.layers [1]
qwen4exp.ple.ngram_size [3]
qwen4exp.ple.heads_per_ngram [8]
qwen4exp.ple.conv_kernel [4]
qwen4exp.ple.eos_token_id [248044]
qwen4exp.embedding_length_per_layer_input [160]
qwen4exp.ple.layer_multipliers [23703573157769, 20109073645365, 8052911324071]
qwen4exp.ple.head_offsets [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297, 160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275]
qwen4exp.ple.head_vocab_sizes [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]
tokenizer.ggml.pre qwen35
tokenizer.ggml.eos_token_id [248046]
qwen4exp.ple.image_token_id [248056]
2 Qwen3.8-Flash-Next-Q8_0-00003-of-00006.gguf per_layer_token_embd.weight Q8_0 [160, 320001536] nbytes 54400261120 data_offset 192
$ ls -l .../Qwen3.8-Flash-Next-Q8_0-00003-of-00006.gguf  -> 54400261312
```

DeepSeek-V4.1-Flash Q3_K_M (`ds_meta.py`; shards 1-4 of 9 present at the time):

```
deepseek41.block_count [40]
deepseek41.embedding_length [5120]
deepseek41.expert_count [384]
deepseek41.expert_used_count [6]
deepseek41.expert_feed_forward_length [2304]
deepseek41.expert_shared_count [1]
deepseek41.engram.head_count [8]
deepseek41.engram.key_length [256]
deepseek41.engram.max_ngram_size [4]
deepseek41.engram.layer_ids len 2 [1, 14]
deepseek41.engram.multipliers len 8 [76632096046245, 4839876093313, 35959672319349, 73987337458391, 67716810739261, 51510806800915, 30921347202721, 82619226485591]
deepseek41.engram.primes len 48 [16000057, 16000079, 16000081, ... 16000889]
deepseek41.engram.offsets len 48 [0, 16000057, 32000136, ... 368005705, 0, 16000477, ... 368015793]
deepseek41.engram.token_map len 129280 (compressed vocab 99092)
deepseek41.engram.pad_id [2]      token_map[2] = 2
tokenizer.ggml.pre joyai-llm
DeepSeek-V4.1-Flash-Q3_K_M-00002-of-00009.gguf blk.1.engram_embd.weight Q3_K [256, 384006168] 42240678480 off 480
DeepSeek-V4.1-Flash-Q3_K_M-00002-of-00009.gguf blk.1.engram_k.weight Q3_K [5120, 4] 8800 off 42240678976
DeepSeek-V4.1-Flash-Q3_K_M-00002-of-00009.gguf blk.1.engram_q.weight Q3_K [5120, 4] 8800 off 42240687776
DeepSeek-V4.1-Flash-Q3_K_M-00002-of-00009.gguf blk.1.engram_wkv.weight Q3_K [6144, 25600] 67584000 off 42240696576
DeepSeek-V4.1-Flash-Q3_K_M-00003-of-00009.gguf blk.1.ffn_down_exps.weight Q5_K [2304, 5120, 384] 3114270720 off 10240
DeepSeek-V4.1-Flash-Q3_K_M-00004-of-00009.gguf blk.7.ffn_up_exps.weight Q3_K [5120, 2304, 384] 1946419200 off 11200
```

### A.2 Storage probes (read-only; each run <= 25 s)

fio 3.41, `--readonly --rw=randread --bs=4k --direct=1 --ioengine=io_uring
--offset=4096 --size=54400253952 --runtime=15 --time_based --norandommap`, with
per-device bcachefs `io_done` deltas (dev0 = NVMe, dev1 = ugreen/USB 10G,
dev2 = T7/USB 5G):

```
qd1_4k: iops=1389 MB/s=5.7 ios=20838 lat_mean=719us p50=477us p99=2671us dev_read_MB={0:4135.9, 1:508.4, 2:491.7} amp=60.2x
qd16_4k: iops=7815 MB/s=32.0 ios=117251 lat_mean=2047us p50=1810us p99=6849us dev_read_MB={0:18858.0, 1:4379.6, 2:5291.5} amp=59.4x
qd64_4k: iops=9124 MB/s=37.4 ios=136957 lat_mean=7013us p50=5734us p99=25297us dev_read_MB={0:20256.3, 1:6686.1, 2:6251.0} amp=59.2x
clat percentiles (us): qd1 p90 1171 p95 1483 p99.9 5669 | qd16 p90 3293 p95 4227 p99.9 14221 | qd64 p90 13828 p95 17433 p99.9 34341
```

mmap page-fault probe (`MAP_SHARED`, `POSIX_MADV_RANDOM`, random 170 B rows,
12 s):

```
mmap threads=1 rows=18660 rows/s=1555 mean=643us p50=512us p90=1001us p99=2271us     dev0 read_MB=4320.6  dev1 393.8  dev2 134.0
mmap threads=16 rows=94462 rows/s=7866 mean=2032us p50=1746us p90=3394us p99=7267us  dev0 15787.6  dev1 3675.6  dev2 4388.8
mmap threads=24 rows=88470 rows/s=7370 mean=3254us p50=2565us p90=5576us p99=17660us dev0 15123.3  dev1 3858.5  dev2 4064.5
```

The design's read path (`batch_probe`: B random rows, 4/8 KiB `O_DIRECT` pread
by Q workers, caller waits for the batch):

```
batch=1 workers=1 batches=11969 mean=0.84ms p50=0.65ms p90=1.29ms p99=3.55ms max=34.91ms rows/s=1197 straddle=4.2%
batch=8 workers=8 batches=2350 mean=4.26ms p50=3.78ms p90=6.23ms p99=15.58ms max=139.98ms rows/s=1880 straddle=4.0%
batch=16 workers=16 batches=3997 mean=3.75ms p50=3.17ms p90=5.31ms p99=13.02ms max=93.37ms rows/s=4264 straddle=4.1%
batch=7100 workers=64 batches=30 mean=850.89ms p50=784.79ms p90=1027.57ms p99=1187.22ms max=1268.40ms rows/s=8344 straddle=4.1%
batch=7100 workers=128 batches=27 mean=942.54ms p50=810.98ms p90=1252.12ms p99=1316.69ms max=1424.02ms rows/s=7533 straddle=4.1%
```

The batch-8 row is slower than batch-16. The per-row rates across these runs
also moved with the ambient IO of other tenants (load 39-57), which is why the
targets carry headroom over the means.

Sequential (whole-table load estimate):
`seq 1MiB QD8 O_DIRECT: 1779 MB/s` (dev0 19761.9 MB, dev1 3723.0, dev2 4121.0
over 12 s).

DeepSeek `blk.1` table, shard 2, the same fio probe with
`--size=42240000000` (`io_probe2.sh`; load average 90-92; the download was
writing shard 4 throughout). Device read deltas in MB (dev0 NVMe, dev1
ugreen, dev2 T7) and bcachefs reconcile's write delta to the T7:

```
ds_qd1_4k:  iops=1014 ios=15210  lat_mean=984us  p50=522us  p90=1221us  p99=7700us  p99.9=35389us  dev_read_MB={0:6800.7, 1:319.8, 2:15.6}   dev2_write_MB=2441.1 (reconcile: amp not attributable)
ds_qd16_4k: iops=5153 ios=77303  lat_mean=3104us p50=2073us p90=5865us p99=18743us p99.9=42205us dev_read_MB={0:12673.4, 1:2818.1, 2:3555.6} amp=60.2x
ds_qd64_4k: iops=8400 ios=126063 lat_mean=7618us p50=6390us p90=14877us p99=28180us p99.9=39059us dev_read_MB={0:19166.8, 1:5972.8, 2:5833.2} amp=60.0x
```

`batch_probe2` on the same extent (`PBASE=480 PROW=110 PNROWS=384006168`):

```
DS blk.1 batch=24 workers=24 batches=1726 mean=6.95ms p50=4.93ms p90=10.99ms p99=27.84ms max=158.02ms rows/s=3452 straddle=2.7%
DS blk.1 batch=48 workers=48 batches=1341 mean=8.95ms p50=7.57ms p90=12.68ms p99=29.13ms max=82.09ms rows/s=5361 straddle=2.5%
DS blk.1 batch=20000 workers=64 batches=11 mean=2563.06ms p50=2186.65ms p90=3251.34ms p99=3251.34ms max=3771.67ms rows/s=7803 straddle=2.6%
```

DSpark-sized batches on the same extent (6 and 3 tokens' rows):

```
DS blk.1 dspark batch=288 workers=64 batches=162 mean=74.09ms p50=75.53ms p90=98.03ms p99=131.12ms max=137.99ms rows/s=3887 straddle=2.8%
DS blk.1 dspark batch=144 workers=64 batches=500 mean=24.02ms p50=21.32ms p90=34.08ms p99=52.49ms max=93.15ms rows/s=5996 straddle=2.7%
```

Cold-expert loads (`expert_run.sh` around `expert_probe`, 12 s each; W loads in
flight, each split into CH chunks read by P threads; per-device bcachefs read
and write deltas in MB; the DeepSeek download was writing later shards during
the DS runs; the reference rows read a Qwen split written 2026-09-12 at the
same slice shapes):

```
DS-expert W=1 P=1 CH=1024KiB loads=53 loads/s=4.4 MB/s=74 load_ms mean=229.2 p50=237.3 p90=386.8 p99=469.3 max=469.3 | dev_read_MB {0: 5495.5, 1: 287.2, 2: 615.9} dev_write_MB {0: 1167.6, 1: 1369.4, 2: 3863.7} | load 34.25, 39.42, 51.91
DS-expert W=1 P=8 CH=1024KiB loads=389 loads/s=32.4 MB/s=549 load_ms mean=30.9 p50=17.2 p90=67.8 p99=126.7 max=187.4 | dev_read_MB {0: 8955.1, 1: 1507.8, 2: 1978.3} dev_write_MB {0: 850.1, 1: 883.9, 2: 2495.0} | load 34.87, 39.30, 51.67
DS-expert W=1 P=16 CH=512KiB loads=851 loads/s=70.9 MB/s=1199 load_ms mean=14.1 p50=12.9 p90=18.4 p99=29.1 max=38.4 | dev_read_MB {0: 16393.3, 1: 3133.5, 2: 3949.7} dev_write_MB {0: 0.0, 1: 0.0, 2: 0.0} | load 36.62, 39.54, 51.61
DS-expert W=2 P=8 CH=1024KiB loads=1069 loads/s=88.9 MB/s=1505 load_ms mean=22.5 p50=20.6 p90=30.5 p99=45.9 max=81.4 | dev_read_MB {0: 17916.3, 1: 3400.5, 2: 4079.0} dev_write_MB {0: 182.3, 1: 0.0, 2: 0.0} | load 40.12, 40.18, 51.69
DS-expert W=6 P=4 CH=1024KiB loads=594 loads/s=49.2 MB/s=832 load_ms mean=121.7 p50=114.5 p90=172.6 p99=328.0 max=342.3 | dev_read_MB {0: 5445.1, 1: 3091.5, 2: 4758.1} dev_write_MB {0: 909.1, 1: 0.0, 2: 0.0} | load 41.11, 40.42, 51.52
DS-expert W=6 P=8 CH=512KiB loads=576 loads/s=47.7 MB/s=807 load_ms mean=125.6 p50=115.6 p90=198.3 p99=257.1 max=312.7 | dev_read_MB {0: 7185.4, 1: 3390.5, 2: 4669.6} dev_write_MB {0: 1303.4, 1: 0.0, 2: 0.0} | load 42.47, 40.79, 51.47
DS-expert W=12 P=2 CH=1024KiB loads=652 loads/s=53.9 MB/s=911 load_ms mean=221.9 p50=216.2 p90=319.8 p99=395.6 max=448.8 | dev_read_MB {0: 6351.6, 1: 3362.3, 2: 4964.0} dev_write_MB {0: 989.9, 1: 0.0, 2: 0.0} | load 46.76, 41.77, 51.67
DS-expert W=1 P=32 CH=256KiB loads=293 loads/s=24.4 MB/s=413 load_ms mean=41.0 p50=38.4 p90=56.8 p99=121.0 max=148.3 | dev_read_MB {0: 3755.9, 1: 2220.7, 2: 3954.8} dev_write_MB {0: 1408.3, 1: 0.0, 2: 0.0} | load 50.62, 42.87, 51.87
Qwen-settled-ref W=1 P=16 CH=512KiB loads=208 loads/s=17.3 MB/s=290 load_ms mean=57.9 p50=44.4 p90=79.9 p99=275.1 max=326.6 | dev_read_MB {0: 649.2, 1: 1363.9, 2: 3713.8} dev_write_MB {0: 2672.5, 1: 143.3, 2: 409.1} | load 52.65, 43.96, 52.00
Qwen-settled-ref W=6 P=4 CH=1024KiB loads=341 loads/s=28.1 MB/s=471 load_ms mean=212.7 p50=187.3 p90=333.9 p99=599.6 max=660.9 | dev_read_MB {0: 177.2, 1: 1911.1, 2: 5353.2} dev_write_MB {0: 2665.7, 1: 31.0, 2: 89.8} | load 57.89, 45.55, 52.39
```

Reference, ext4 on the root NVMe (WD Blue SN570, shared by the whole system),
the same 4 KiB `O_DIRECT` randread over the 9.0 GiB of `build/bin`:

```
ext4 nvme1 qd1 iops=6019 p50=151us p99=922us mean=166us
ext4 nvme1 qd16 iops=82798 p50=163us p99=930us mean=193us
ext4 nvme1 qd64 iops=110110 p50=518us p99=2900us mean=581us
```

HOST-tier gather: random 170 B Q8_0 rows from a 4 GiB anonymous THP table,
dequantized to F32 (best of 20):

```
RAM gather rows=16 threads=1 best=4.2 us
RAM gather rows=8192 threads=1 best=2329.7 us
RAM gather rows=8192 threads=8 best=1941.1 us   (includes thread spawn)
```

bcachefs state: `bcachefs fs usage -h /models` gave bulk.t7 171G used,
bulk.ugreen 612G used, nvme.kingston 187G used, `cached: 182G`. Per-device
`alloc_debug`: dev0 `user 0 sectors, cached 382424032 sectors`. Options:
`foreground_target=nvme background_target=bulk promote_target=nvme
data_checksum=crc32c encoded_extent_max=256k promote_whole_extents=1
block_size=4.00k compression=none`.

### A.3 Hash and cache simulation

Validation:

```
tokenizer (qwen35 regex + byte BPE) vs models/ggml-vocab-qwen35.gguf.{inp,out}: 50/50 match
Qwen3.8 vocab 248320 sha 9c56ea103777902c eos id 248044 = '<|endoftext|>' 248046 = '<|im_end|>'
wiki_test: 1288556 chars -> 297193 tokens, roundtrip=OK
wiki_train: 10918892 chars -> 2505800 tokens, roundtrip=OK
code: 9989391 chars -> 3422500 tokens, roundtrip=OK
docs: 7694987 chars -> 2289834 tokens, roundtrip=OK
python vs verbatim-upstream C++: IDENTICAL 320000 rows; max row 320001219 < 320001536: True
ubatch_vs_decode_hash_equal=1   (ubatch 512 and ubatch 1 produce the same rows)
```

DeepSeek (`qtok2.py`, `ds_hash.py`, `ds_sim.cpp`):

```
46/46 match (deepseek-llm vectors)
DS vocab 129280 sha ab8d16d71b3a9d12 compressed vocab 99092 pad_id [2] token_map[pad] 2
wiki_test: 1288556 chars -> 287730 tokens, roundtrip=OK
wiki_train: 10918892 chars -> 2424283 tokens, roundtrip=OK
code: 9989391 chars -> 3287443 tokens, roundtrip=OK
docs: 7694987 chars -> 2241455 tokens, roundtrip=OK
DS python vs C++: IDENTICAL 960000 rows; max row 384015568   (wiki_test, first 20000 tokens)
DS python vs C++: IDENTICAL 960000 rows; max row 384015792   (code, first 20000 tokens)
```

`ple_sim` (Qwen, whole-stream hit rates):

```
wiki_test: tokens=297193 row_refs=4755088 distinct_rows=2603218 (0.41 GiB) ubatch_vs_decode_hash_equal=1
wiki_test: cold PP512 ubatch: unique rows/ubatch=7028 of 8192 (85.8%), unique 4K pages/ubatch=7026
wiki_test: cache     32 MiB (   197379 rows): LRU hit=36.5% (2nd half 34.6%, bigram 48.7%, trigram 24.2%)  CLOCK hit=36.9% (2nd half 35.1%)  4K-page LRU same bytes hit=21.0%
wiki_test: cache    128 MiB (   789516 rows): LRU hit=41.9% (2nd half 41.1%, bigram 55.5%, trigram 28.3%)  CLOCK hit=42.4% (2nd half 41.9%)  4K-page LRU same bytes hit=29.5%
wiki_test: cache    512 MiB (  3158064 rows): LRU hit=45.3% (2nd half 47.0%, bigram 59.5%, trigram 31.0%)  CLOCK hit=45.3% (2nd half 47.0%)  4K-page LRU same bytes hit=35.8%
wiki_test: cache   2048 MiB ( 12632256 rows): LRU hit=45.3% (2nd half 47.0%, bigram 59.5%, trigram 31.0%)  CLOCK hit=45.3% (2nd half 47.0%)  4K-page LRU same bytes hit=42.5%
wiki_test: cache   8192 MiB ( 50529027 rows): LRU hit=45.3% (2nd half 47.0%, bigram 59.5%, trigram 31.0%)  CLOCK hit=45.3% (2nd half 47.0%)  4K-page LRU same bytes hit=50.3%
wiki_test 17.59s 3242040KB
wiki_train: tokens=2505800 row_refs=40092800 distinct_rows=16901229 (2.68 GiB) ubatch_vs_decode_hash_equal=1
wiki_train: cold PP512 ubatch: unique rows/ubatch=7104 of 8192 (86.7%), unique 4K pages/ubatch=7102
wiki_train: cache     32 MiB (   197379 rows): LRU hit=34.5% (2nd half 34.3%, bigram 46.7%, trigram 22.4%)  CLOCK hit=35.1% (2nd half 34.8%)  4K-page LRU same bytes hit=19.5%
wiki_train: cache    128 MiB (   789516 rows): LRU hit=41.1% (2nd half 41.0%, bigram 54.9%, trigram 27.3%)  CLOCK hit=41.8% (2nd half 41.7%)  4K-page LRU same bytes hit=27.6%
wiki_train: cache    512 MiB (  3158064 rows): LRU hit=49.7% (2nd half 50.3%, bigram 65.1%, trigram 34.4%)  CLOCK hit=50.4% (2nd half 51.1%)  4K-page LRU same bytes hit=33.7%
wiki_train: cache   2048 MiB ( 12632256 rows): LRU hit=57.4% (2nd half 61.8%, bigram 72.9%, trigram 42.0%)  CLOCK hit=57.5% (2nd half 61.8%)  4K-page LRU same bytes hit=41.4%
wiki_train: cache   8192 MiB ( 50529027 rows): LRU hit=57.8% (2nd half 62.6%, bigram 73.2%, trigram 42.5%)  CLOCK hit=57.8% (2nd half 62.6%)  4K-page LRU same bytes hit=55.6%
wiki_train 275.49s 4897572KB
code: tokens=3422500 row_refs=54760000 distinct_rows=5075084 (0.80 GiB) ubatch_vs_decode_hash_equal=1
code: cold PP512 ubatch: unique rows/ubatch=3969 of 8192 (48.4%), unique 4K pages/ubatch=3968
code: cache     32 MiB (   197379 rows): LRU hit=86.8% (2nd half 84.3%, bigram 90.7%, trigram 83.0%)  CLOCK hit=86.9% (2nd half 84.4%)  4K-page LRU same bytes hit=68.6%
code: cache    128 MiB (   789516 rows): LRU hit=89.1% (2nd half 87.2%, bigram 92.8%, trigram 85.5%)  CLOCK hit=89.1% (2nd half 87.3%)  4K-page LRU same bytes hit=81.2%
code: cache    512 MiB (  3158064 rows): LRU hit=90.5% (2nd half 89.6%, bigram 94.1%, trigram 87.0%)  CLOCK hit=90.6% (2nd half 89.7%)  4K-page LRU same bytes hit=86.1%
code: cache   2048 MiB ( 12632256 rows): LRU hit=90.7% (2nd half 90.1%, bigram 94.3%, trigram 87.2%)  CLOCK hit=90.7% (2nd half 90.1%)  4K-page LRU same bytes hit=89.0%
code: cache   8192 MiB ( 50529027 rows): LRU hit=90.7% (2nd half 90.1%, bigram 94.3%, trigram 87.2%)  CLOCK hit=90.7% (2nd half 90.1%)  4K-page LRU same bytes hit=91.6%
code 125.63s 3703892KB
docs: tokens=2289834 row_refs=36637344 distinct_rows=10478884 (1.66 GiB) ubatch_vs_decode_hash_equal=1
docs: cold PP512 ubatch: unique rows/ubatch=6231 of 8192 (76.1%), unique 4K pages/ubatch=6229
docs: cache     32 MiB (   197379 rows): LRU hit=56.3% (2nd half 56.4%, bigram 66.0%, trigram 46.7%)  CLOCK hit=56.8% (2nd half 56.9%)  4K-page LRU same bytes hit=33.6%
docs: cache    128 MiB (   789516 rows): LRU hit=63.0% (2nd half 63.5%, bigram 73.2%, trigram 52.8%)  CLOCK hit=63.4% (2nd half 63.9%)  4K-page LRU same bytes hit=45.1%
docs: cache    512 MiB (  3158064 rows): LRU hit=68.2% (2nd half 69.1%, bigram 78.9%, trigram 57.6%)  CLOCK hit=68.5% (2nd half 69.5%)  4K-page LRU same bytes hit=54.6%
docs: cache   2048 MiB ( 12632256 rows): LRU hit=71.4% (2nd half 74.7%, bigram 82.0%, trigram 60.8%)  CLOCK hit=71.4% (2nd half 74.7%)  4K-page LRU same bytes hit=62.8%
docs: cache   8192 MiB ( 50529027 rows): LRU hit=71.4% (2nd half 74.7%, bigram 82.0%, trigram 60.8%)  CLOCK hit=71.4% (2nd half 74.7%)  4K-page LRU same bytes hit=72.1%
docs 271.18s 4183236KB
mixed: tokens=8218134 row_refs=131490144 distinct_rows=30389923 (4.81 GiB) ubatch_vs_decode_hash_equal=1
mixed: cold PP512 ubatch: unique rows/ubatch=5562 of 8192 (67.9%), unique 4K pages/ubatch=5561
mixed: cache     32 MiB (   197379 rows): LRU hit=58.8% (2nd half 61.5%, bigram 66.9%, trigram 50.8%)  CLOCK hit=59.3% (2nd half 62.0%)  4K-page LRU same bytes hit=41.0%
mixed: cache    128 MiB (   789516 rows): LRU hit=64.4% (2nd half 67.0%, bigram 72.9%, trigram 55.8%)  CLOCK hit=64.7% (2nd half 67.4%)  4K-page LRU same bytes hit=45.8%
mixed: cache    512 MiB (  3158064 rows): LRU hit=69.6% (2nd half 72.1%, bigram 78.7%, trigram 60.5%)  CLOCK hit=70.0% (2nd half 72.4%)  4K-page LRU same bytes hit=57.4%
mixed: cache   2048 MiB ( 12632256 rows): LRU hit=74.9% (2nd half 77.9%, bigram 84.0%, trigram 65.7%)  CLOCK hit=75.1% (2nd half 78.3%)  4K-page LRU same bytes hit=64.3%
mixed: cache   8192 MiB ( 50529027 rows): LRU hit=76.9% (2nd half 81.6%, bigram 85.7%, trigram 68.1%)  CLOCK hit=76.9% (2nd half 81.6%)  4K-page LRU same bytes hit=73.4%
mixed 819.42s 6738224KB
```

`ple_sim2` (Qwen, CLOCK, steady state = after the first 25%):

```
random_tokens cache=    0MiB decode(steady): miss rows/token=16.00  P(0 miss)=0.0%  P(<=8 miss)=0.0%  P(16 miss)=100.0% | PP512(steady): unique-miss rows/ubatch mean=8192 max=8192
random_tokens cache=   32MiB decode(steady): miss rows/token=15.99  P(0 miss)=0.0%  P(<=8 miss)=0.0%  P(16 miss)=99.0% | PP512(steady): unique-miss rows/ubatch mean=8187 max=8192
random_tokens cache=  128MiB decode(steady): miss rows/token=15.96  P(0 miss)=0.0%  P(<=8 miss)=0.0%  P(16 miss)=96.2% | PP512(steady): unique-miss rows/ubatch mean=8172 max=8184
random_tokens cache=  512MiB decode(steady): miss rows/token=15.87  P(0 miss)=0.0%  P(<=8 miss)=0.0%  P(16 miss)=87.7% | PP512(steady): unique-miss rows/ubatch mean=8125 max=8173
random_tokens cache= 2048MiB decode(steady): miss rows/token=15.85  P(0 miss)=0.0%  P(<=8 miss)=0.0%  P(16 miss)=86.1% | PP512(steady): unique-miss rows/ubatch mean=8116 max=8173
random_tokens cache= 8192MiB decode(steady): miss rows/token=15.85  P(0 miss)=0.0%  P(<=8 miss)=0.0%  P(16 miss)=86.1% | PP512(steady): unique-miss rows/ubatch mean=8116 max=8173
wiki_test cache=    0MiB decode(steady): miss rows/token=16.00  P(0 miss)=0.0%  P(<=8 miss)=0.0%  P(16 miss)=100.0% | PP512(steady): unique-miss rows/ubatch mean=7055 max=7848
wiki_test cache=   32MiB decode(steady): miss rows/token=10.13  P(0 miss)=24.2%  P(<=8 miss)=49.1%  P(16 miss)=50.4% | PP512(steady): unique-miss rows/ubatch mean=5185 max=6676
wiki_test cache=  128MiB decode(steady): miss rows/token=9.08  P(0 miss)=28.9%  P(<=8 miss)=57.3%  P(16 miss)=41.0% | PP512(steady): unique-miss rows/ubatch mean=4649 max=6362
wiki_test cache=  512MiB decode(steady): miss rows/token=8.47  P(0 miss)=31.9%  P(<=8 miss)=61.6%  P(16 miss)=35.3% | PP512(steady): unique-miss rows/ubatch mean=4335 max=5951
wiki_test cache= 2048MiB decode(steady): miss rows/token=8.47  P(0 miss)=31.9%  P(<=8 miss)=61.6%  P(16 miss)=35.3% | PP512(steady): unique-miss rows/ubatch mean=4335 max=5951
wiki_test cache= 8192MiB decode(steady): miss rows/token=8.47  P(0 miss)=31.9%  P(<=8 miss)=61.6%  P(16 miss)=35.3% | PP512(steady): unique-miss rows/ubatch mean=4335 max=5951
code cache=    0MiB decode(steady): miss rows/token=16.00  P(0 miss)=0.0%  P(<=8 miss)=0.0%  P(16 miss)=100.0% | PP512(steady): unique-miss rows/ubatch mean=3922 max=7456
code cache=   32MiB decode(steady): miss rows/token=2.01  P(0 miss)=83.7%  P(<=8 miss)=91.2%  P(16 miss)=8.8% | PP512(steady): unique-miss rows/ubatch mean=1024 max=5909
code cache=  128MiB decode(steady): miss rows/token=1.62  P(0 miss)=86.3%  P(<=8 miss)=93.4%  P(16 miss)=6.4% | PP512(steady): unique-miss rows/ubatch mean=828 max=5524
code cache=  512MiB decode(steady): miss rows/token=1.33  P(0 miss)=88.2%  P(<=8 miss)=95.0%  P(16 miss)=4.3% | PP512(steady): unique-miss rows/ubatch mean=681 max=5419
code cache= 2048MiB decode(steady): miss rows/token=1.30  P(0 miss)=88.4%  P(<=8 miss)=95.2%  P(16 miss)=4.0% | PP512(steady): unique-miss rows/ubatch mean=664 max=5392
code cache= 8192MiB decode(steady): miss rows/token=1.30  P(0 miss)=88.4%  P(<=8 miss)=95.2%  P(16 miss)=4.0% | PP512(steady): unique-miss rows/ubatch mean=664 max=5392
docs cache=    0MiB decode(steady): miss rows/token=16.00  P(0 miss)=0.0%  P(<=8 miss)=0.0%  P(16 miss)=100.0% | PP512(steady): unique-miss rows/ubatch mean=6190 max=8128
docs cache=   32MiB decode(steady): miss rows/token=6.50  P(0 miss)=49.8%  P(<=8 miss)=68.8%  P(16 miss)=30.9% | PP512(steady): unique-miss rows/ubatch mean=3323 max=7076
docs cache=  128MiB decode(steady): miss rows/token=5.42  P(0 miss)=56.1%  P(<=8 miss)=76.0%  P(16 miss)=23.1% | PP512(steady): unique-miss rows/ubatch mean=2775 max=6640
docs cache=  512MiB decode(steady): miss rows/token=4.58  P(0 miss)=60.5%  P(<=8 miss)=81.5%  P(16 miss)=15.7% | PP512(steady): unique-miss rows/ubatch mean=2343 max=5700
docs cache= 2048MiB decode(steady): miss rows/token=3.97  P(0 miss)=63.9%  P(<=8 miss)=85.2%  P(16 miss)=10.4% | PP512(steady): unique-miss rows/ubatch mean=2032 max=4930
docs cache= 8192MiB decode(steady): miss rows/token=3.97  P(0 miss)=63.9%  P(<=8 miss)=85.2%  P(16 miss)=10.4% | PP512(steady): unique-miss rows/ubatch mean=2032 max=4930
mixed cache=    0MiB decode(steady): miss rows/token=16.00  P(0 miss)=0.0%  P(<=8 miss)=0.0%  P(16 miss)=100.0% | PP512(steady): unique-miss rows/ubatch mean=5452 max=8128
mixed cache=   32MiB decode(steady): miss rows/token=6.16  P(0 miss)=53.6%  P(<=8 miss)=69.3%  P(16 miss)=30.4% | PP512(steady): unique-miss rows/ubatch mean=3154 max=7201
mixed cache=  128MiB decode(steady): miss rows/token=5.28  P(0 miss)=58.7%  P(<=8 miss)=75.1%  P(16 miss)=23.9% | PP512(steady): unique-miss rows/ubatch mean=2700 max=6841
mixed cache=  512MiB decode(steady): miss rows/token=4.45  P(0 miss)=63.0%  P(<=8 miss)=80.8%  P(16 miss)=16.3% | PP512(steady): unique-miss rows/ubatch mean=2275 max=6146
mixed cache= 2048MiB decode(steady): miss rows/token=3.52  P(0 miss)=67.6%  P(<=8 miss)=86.3%  P(16 miss)=7.1% | PP512(steady): unique-miss rows/ubatch mean=1800 max=5186
mixed cache= 8192MiB decode(steady): miss rows/token=3.13  P(0 miss)=69.6%  P(<=8 miss)=88.1%  P(16 miss)=4.2% | PP512(steady): unique-miss rows/ubatch mean=1603 max=4673
wiki_train cache=    0MiB decode(steady): miss rows/token=16.00  P(0 miss)=0.0%  P(<=8 miss)=0.0%  P(16 miss)=100.0% | PP512(steady): unique-miss rows/ubatch mean=7100 max=7968
wiki_train cache=   32MiB decode(steady): miss rows/token=10.36  P(0 miss)=22.8%  P(<=8 miss)=47.6%  P(16 miss)=51.8% | PP512(steady): unique-miss rows/ubatch mean=5298 max=6932
wiki_train cache=  128MiB decode(steady): miss rows/token=9.26  P(0 miss)=27.8%  P(<=8 miss)=56.1%  P(16 miss)=42.1% | PP512(steady): unique-miss rows/ubatch mean=4738 max=6407
wiki_train cache=  512MiB decode(steady): miss rows/token=7.75  P(0 miss)=35.0%  P(<=8 miss)=67.0%  P(16 miss)=28.0% | PP512(steady): unique-miss rows/ubatch mean=3968 max=5820
wiki_train cache= 2048MiB decode(steady): miss rows/token=6.29  P(0 miss)=42.6%  P(<=8 miss)=75.8%  P(16 miss)=14.5% | PP512(steady): unique-miss rows/ubatch mean=3219 max=5311
wiki_train cache= 8192MiB decode(steady): miss rows/token=6.21  P(0 miss)=43.0%  P(<=8 miss)=76.2%  P(16 miss)=13.8% | PP512(steady): unique-miss rows/ubatch mean=3177 max=5311
finished
```

`ds_sim` (DeepSeek, both tables, CLOCK at the stated total size in 110 B
slots; hit = whole stream; decode and PP512 = steady state after the first
25%; the random stream is 300000 tokens drawn uniformly from 129280):

```
random: tokens=300000 row_refs=14400000 distinct_rows=14265643 (1.46 GiB @110B) cold PP512 unique rows/ubatch=24576 of 24576
random cache=    0MiB hit=0.0% (2g 0.0%, 3g 0.0%, 4g 0.0%) decode(steady): miss rows/token=48.00 of 48 P(0 miss)=0.0% P(48 miss)=100.0% | PP512(steady) unique-miss rows/ubatch mean=24576 max=24576
random cache=  128MiB hit=0.2% (2g 0.2%, 3g 0.2%, 4g 0.1%) decode(steady): miss rows/token=47.92 of 48 P(0 miss)=0.0% P(48 miss)=92.7% | PP512(steady) unique-miss rows/ubatch mean=24537 max=24553
random cache=  512MiB hit=0.5% (2g 0.5%, 3g 0.5%, 4g 0.5%) decode(steady): miss rows/token=47.70 of 48 P(0 miss)=0.0% P(48 miss)=74.0% | PP512(steady) unique-miss rows/ubatch mean=24422 max=24480
random cache= 2048MiB hit=0.9% (2g 0.9%, 3g 0.9%, 4g 0.9%) decode(steady): miss rows/token=47.44 of 48 P(0 miss)=0.0% P(48 miss)=58.1% | PP512(steady) unique-miss rows/ubatch mean=24290 max=24480
random cache= 8192MiB hit=0.9% (2g 0.9%, 3g 0.9%, 4g 0.9%) decode(steady): miss rows/token=47.44 of 48 P(0 miss)=0.0% P(48 miss)=58.1% | PP512(steady) unique-miss rows/ubatch mean=24290 max=24480
wiki_test: tokens=287730 row_refs=13811040 distinct_rows=9113026 (0.93 GiB @110B) cold PP512 unique rows/ubatch=22117 of 24576
wiki_test cache=    0MiB hit=0.0% (2g 0.0%, 3g 0.0%, 4g 0.0%) decode(steady): miss rows/token=48.00 of 48 P(0 miss)=0.0% P(48 miss)=100.0% | PP512(steady) unique-miss rows/ubatch mean=22180 max=23920
wiki_test cache=  128MiB hit=28.9% (2g 51.5%, 3g 23.5%, 4g 11.6%) decode(steady): miss rows/token=34.12 of 48 P(0 miss)=11.3% P(48 miss)=44.6% | PP512(steady) unique-miss rows/ubatch mean=17460 max=21056
wiki_test cache=  512MiB hit=33.3% (2g 58.2%, 3g 27.8%, 4g 13.7%) decode(steady): miss rows/token=31.42 of 48 P(0 miss)=13.5% P(48 miss)=30.0% | PP512(steady) unique-miss rows/ubatch mean=16092 max=20438
wiki_test cache= 2048MiB hit=34.0% (2g 59.2%, 3g 28.6%, 4g 14.2%) decode(steady): miss rows/token=30.93 of 48 P(0 miss)=14.0% P(48 miss)=27.0% | PP512(steady) unique-miss rows/ubatch mean=15837 max=20438
wiki_test cache= 8192MiB hit=34.0% (2g 59.2%, 3g 28.6%, 4g 14.2%) decode(steady): miss rows/token=30.93 of 48 P(0 miss)=14.0% P(48 miss)=27.0% | PP512(steady) unique-miss rows/ubatch mean=15837 max=20438
code: tokens=3287443 row_refs=157797264 distinct_rows=19396356 (1.99 GiB @110B) cold PP512 unique rows/ubatch=13114 of 24576
code cache=    0MiB hit=0.0% (2g 0.0%, 3g 0.0%, 4g 0.0%) decode(steady): miss rows/token=48.00 of 48 P(0 miss)=0.0% P(48 miss)=100.0% | PP512(steady) unique-miss rows/ubatch mean=12964 max=22511
code cache=  128MiB hit=84.4% (2g 92.1%, 3g 84.2%, 4g 76.9%) decode(steady): miss rows/token=7.11 of 48 P(0 miss)=77.9% P(48 miss)=6.9% | PP512(steady) unique-miss rows/ubatch mean=3632 max=18348
code cache=  512MiB hit=86.6% (2g 94.1%, 3g 86.5%, 4g 79.3%) decode(steady): miss rows/token=5.91 of 48 P(0 miss)=80.5% P(48 miss)=3.9% | PP512(steady) unique-miss rows/ubatch mean=3026 max=17368
code cache= 2048MiB hit=87.7% (2g 95.0%, 3g 87.6%, 4g 80.5%) decode(steady): miss rows/token=5.22 of 48 P(0 miss)=81.8% P(48 miss)=1.8% | PP512(steady) unique-miss rows/ubatch mean=2674 max=17006
code cache= 8192MiB hit=87.7% (2g 95.0%, 3g 87.6%, 4g 80.5%) decode(steady): miss rows/token=5.22 of 48 P(0 miss)=81.8% P(48 miss)=1.8% | PP512(steady) unique-miss rows/ubatch mean=2674 max=17006
docs: tokens=2241455 row_refs=107589840 distinct_rows=39423992 (4.04 GiB @110B) cold PP512 unique rows/ubatch=19584 of 24576
docs cache=    0MiB hit=0.0% (2g 0.0%, 3g 0.0%, 4g 0.0%) decode(steady): miss rows/token=48.00 of 48 P(0 miss)=0.0% P(48 miss)=100.0% | PP512(steady) unique-miss rows/ubatch mean=19440 max=24464
docs cache=  128MiB hit=52.3% (2g 70.3%, 3g 48.9%, 4g 37.8%) decode(steady): miss rows/token=21.39 of 48 P(0 miss)=40.9% P(48 miss)=25.0% | PP512(steady) unique-miss rows/ubatch mean=10945 max=21725
docs cache=  512MiB hit=57.5% (2g 77.5%, 3g 53.9%, 4g 41.1%) decode(steady): miss rows/token=18.89 of 48 P(0 miss)=44.0% P(48 miss)=14.7% | PP512(steady) unique-miss rows/ubatch mean=9669 max=19637
docs cache= 2048MiB hit=62.1% (2g 82.9%, 3g 58.8%, 4g 44.6%) decode(steady): miss rows/token=16.33 of 48 P(0 miss)=46.7% P(48 miss)=4.2% | PP512(steady) unique-miss rows/ubatch mean=8368 max=17938
docs cache= 8192MiB hit=63.4% (2g 83.9%, 3g 60.2%, 4g 45.9%) decode(steady): miss rows/token=15.52 of 48 P(0 miss)=47.8% P(48 miss)=2.6% | PP512(steady) unique-miss rows/ubatch mean=7945 max=16614
wiki_train: tokens=2424283 row_refs=116365584 distinct_rows=62529672 (6.41 GiB @110B) cold PP512 unique rows/ubatch=22320 of 24576
wiki_train cache=    0MiB hit=0.0% (2g 0.0%, 3g 0.0%, 4g 0.0%) decode(steady): miss rows/token=48.00 of 48 P(0 miss)=0.0% P(48 miss)=100.0% | PP512(steady) unique-miss rows/ubatch mean=22306 max=24000
wiki_train cache=  128MiB hit=27.6% (2g 50.3%, 3g 22.1%, 4g 10.4%) decode(steady): miss rows/token=34.64 of 48 P(0 miss)=10.4% P(48 miss)=45.8% | PP512(steady) unique-miss rows/ubatch mean=17731 max=21498
wiki_train cache=  512MiB hit=34.0% (2g 60.3%, 3g 28.3%, 4g 13.5%) decode(steady): miss rows/token=31.43 of 48 P(0 miss)=13.0% P(48 miss)=28.6% | PP512(steady) unique-miss rows/ubatch mean=16091 max=20293
wiki_train cache= 2048MiB hit=42.0% (2g 70.3%, 3g 36.8%, 4g 18.9%) decode(steady): miss rows/token=26.93 of 48 P(0 miss)=17.3% P(48 miss)=7.9% | PP512(steady) unique-miss rows/ubatch mean=13790 max=18564
wiki_train cache= 8192MiB hit=46.3% (2g 74.0%, 3g 41.5%, 4g 23.3%) decode(steady): miss rows/token=24.20 of 48 P(0 miss)=20.0% P(48 miss)=2.4% | PP512(steady) unique-miss rows/ubatch mean=12389 max=18564
mixed: tokens=7953181 row_refs=381752688 distinct_rows=111780913 (11.45 GiB @110B) cold PP512 unique rows/ubatch=17762 of 24576
mixed cache=    0MiB hit=0.0% (2g 0.0%, 3g 0.0%, 4g 0.0%) decode(steady): miss rows/token=48.00 of 48 P(0 miss)=0.0% P(48 miss)=100.0% | PP512(steady) unique-miss rows/ubatch mean=17452 max=24464
mixed cache=  128MiB hit=55.1% (2g 70.1%, 3g 52.3%, 4g 43.0%) decode(steady): miss rows/token=20.35 of 48 P(0 miss)=45.7% P(48 miss)=25.9% | PP512(steady) unique-miss rows/ubatch mean=10421 max=21935
mixed cache=  512MiB hit=60.0% (2g 76.5%, 3g 57.0%, 4g 46.5%) decode(steady): miss rows/token=17.97 of 48 P(0 miss)=49.1% P(48 miss)=15.9% | PP512(steady) unique-miss rows/ubatch mean=9200 max=20540
mixed cache= 2048MiB hit=65.1% (2g 82.7%, 3g 62.3%, 4g 50.5%) decode(steady): miss rows/token=15.46 of 48 P(0 miss)=51.9% P(48 miss)=4.5% | PP512(steady) unique-miss rows/ubatch mean=7916 max=18649
mixed cache= 8192MiB hit=70.2% (2g 86.8%, 3g 67.7%, 4g 56.1%) decode(steady): miss rows/token=12.43 of 48 P(0 miss)=54.9% P(48 miss)=0.2% | PP512(steady) unique-miss rows/ubatch mean=6370 max=16572
finished
```

## Appendix B. Probe sources (scratch, not committed)

- `io_probe.sh`: fio randread driver plus bcachefs `io_done` snapshots.
- `mmap_probe.cpp`: N threads touch random rows through `mmap`.
- `batch_probe.cpp`: the §5.2 pread pool, measured per batch.
- `ram_gather.cpp`: the HOST-tier gather plus dequant.
- `qtok.py`: the qwen35 byte-level BPE, reading tokens and merges from a GGUF.
- `ple_hash.py`: an independent Python transcription of `set_input`.
- `upstream_set_input_body.inc`: `sed -n 1067,1125p` of
  `694ec2354 src/models/qwen4exp.cpp`, `#include`d verbatim by `ple_sim*.cpp`.
- `ple_sim.cpp` / `ple_sim2.cpp`: the row stream plus LRU, CLOCK and 4 KiB-page
  LRU; per-token and per-ubatch miss metrics.
- `io_probe2.sh`, `batch_probe2.cpp`: the same probes with the file, extent and
  row size as parameters (used for the DeepSeek table).
- `ds_meta.py`: the DeepSeek GGUF keys and engram tensor extents.
- `qtok2.py`, `ds_regex.json`, `ds_tokenize.py`: the multi-regex BPE, its
  deepseek-llm validation, and the `joyai-llm` tokenization of the corpora.
- `ds_hash.py`, `ds_sim.cpp`, `ds_consts.inc`, `ds_token_map.bin`: the two
  transcriptions of `NgramHashState.forward` and the CLOCK simulation.
- `expert_probe.cpp`, `expert_run.sh`, `ds_expert_extents.txt`,
  `qwen_ref_extents.txt`: whole-expert `O_DIRECT` loads over the routed-expert
  slices, with per-device attribution.

T7 should lift `upstream_set_input_body.inc`, `ds_sim.cpp`'s `ds_rows()` and
both Python transcriptions into CPU-only unit tests, so that a change to either
hash fails a gate instead of silently changing the rows the cache predicts.
