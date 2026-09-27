# SYCL STORAGE tier for lazy row tables (PLE / n-gram / engram embeddings)

*Status: DESIGN rev 2b, not implemented. Tracker: `llama.cpp-th32`.
Implementation tickets are filed after review (proposed split in §11). Written
2026-09-26 against fork `76c7f6548` and upstream `ggml-org/master` `694ec2354`;
the host-fit terms are re-read at fork `c69d5774d`, whose `unified-cache.cpp`
and `ggml-sycl.cpp` are byte-identical to `76c7f6548`. Rev 2 folded review
`design-th32-r1` under the lead's rulings §T1. Rev 2b folds review
`design-th32-r2` (1 Critical, 5 Important, 12 Minor) under the lead's rulings
§T2, §T3, §T4 and §D5, and the owner's decision §D4
(`lead-rulings-2026-09-26.md`).*

Citations: fork code as `76c7f6548 <file>:<line>`, upstream code as
`694ec2354 <file>:<line>` (read with `git show ggml-org/master:<file>`; the fork
does not have `qwen4exp` or `TENSOR_READ_LAZY` yet, `llama.cpp-n77l` is merging
them). `deepseek41` has no runtime anywhere yet: its conversion is upstream
PR #28696, and its runtime design is `llama.cpp-9g22`
(`docs/design/deepseek41-runtime.md`, branch `task/ds41`), which consumes the
interface in §9.1. This document owns the storage-tier contract and the
numbers it rests on; `llama.cpp-9g22` links to them by the anchors below
(`th32-contract`, `th32-host-fit`, `th32-windows`, `th32-ds-targets`,
`th32-cold-expert-facts`, `th32-dspark`, and from rev 2b `th32-p3` and
`th32-d4-fit`). Every number in this document was
measured on this host on 2026-09-26 under its permanent ambient load (load
average 31-57, ~90 for the first DeepSeek probes); the raw logs are in
Appendix A and the probe sources are described in Appendix B.

**Attribution.** A decision is called the owner's only where the owner made it
(`lead-rulings-2026-09-26.md`, or relayed by the lead as an owner answer), and
the owner's decisions are cited as **owner §D1**, **owner §D2** and **owner
§D4** after their sections there. A decision the lead made is marked *lead
ruling* with its section (§T1-§T4, §D5). Everything else in this document is
this design's **proposal**. This document's own decisions are numbered D1-D9
in §13; the one that was also called D4 (`--lazy-mode`) is **DLM** from rev 2b
on, so that "§D4" always means the owner's decision. A bare D-number is this
document's; the rulings file's sections always carry "§" and their author
("owner §D2", "*lead ruling §D5*").

---

## 0. Summary

**The problem.** Two models carry hashed n-gram embedding tables that are read a
few rows per token and are far too large for their value on a GPU:

| | Qwen3.8-Flash-Next (`qwen4exp`) PLE | DeepSeek-V4.1-Flash (`deepseek41`) engram |
|---|---|---|
| tables | 1: `per_layer_token_embd.weight` | 2: `blk.1.engram_embd.weight`, `blk.14.engram_embd.weight` |
| shape, type | `[160, 320001536]` Q8_0 | `[256, 384006168]` and `[256, 384016682]`; Q2_K in the Q2_K rung (Q3_K in Q3_K_M) |
| size | **50.66 GiB** | **2 x 30.04 GiB** Q2_K (2 x 39.34 GiB Q3_K) |
| row | 170 B | 84 B Q2_K (110 B Q3_K) |
| rows per token | 16 (orders 2-3 x 8 heads) | **48**: 24 per table (orders 2-4 x 8 heads) |
| raw bytes per token | 2720 B | 4032 B Q2_K (5280 B Q3_K) |
| row indices known host-side before compute | yes: hash in `set_input` | yes: `token_map` + hash in `set_input` |
| consumed at | block 1 | block 1 (table 1) and block 14 (table 14) |
| residency the planner picks on this host | **HOST** at every `MemAvailable` reading once lever (s) lands; on today's code **STORAGE** at the lowest reading, 171.4 GiB (§4.5) | **STORAGE**, both tables; phase 1 also puts a cold routed-expert set on STORAGE (owner §D4, §4.5) |
| who executes the gather | the CPU, as a SYCL-backend-owned `host_task`; an event-ordered H2D copy takes the output to the device | the same, after the rows' IO ticket completes |

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
(`lazy_table`). The planner gives it one of two **residencies**, with no
environment variable:

- **HOST:** the whole table is resident in cache-owned, **locked**, CPU-only
  host memory (a new `alloc_tier::HOST_CPU`, §3.1). The planner chooses this
  when the table fits the host budget that remains after VRAM placement, host
  experts, host zones and the context envelope.
- **STORAGE:** the table stays in its file. What is resident is a row-granular
  **hot-row cache** (CLOCK), a **yieldable** tenant sized from the leftover host
  budget, plus a small per-context **staging** area. Missing rows are read with
  `O_DIRECT` preads that are issued as soon as the row indices exist (after
  `set_input`), in priority order: the current ubatch before lookahead, the
  earlier consumer layer first, contexts served fairly. STORAGE is a residency
  state, not an allocation tier: no byte is ever allocated "in" it (§3.1).

In both residencies **the CPU executes the gather**, and the table never
reaches the GPU: there is no zero-copy and no weight streaming. Where the
gather sits in the graph is the **provider's** decision, not the model's
(*lead ruling §T1*, C2 and D6(c)). Both residencies take **one path**
(*lead ruling §T2*, DQ, and §T4 I-3):

- The **SYCL backend** owns the gather node and runs the gather as a CPU
  `host_task`, submitted at prefetch with no device dependency (a cross-queue
  dependency would block the submitting thread, `llama.cpp-c6ah`). A **HOST**
  table's host task dequantizes straight from the resident body, in
  microseconds. A **STORAGE** table's host task first waits for its IO ticket
  on the runtime thread. Either way it dequantizes into a planned pinned
  output slot.
- An **explicit, event-ordered H2D copy** on the main in-order queue moves that
  F32 activation into the device compute buffer just before the consumer, with
  an event edge on the host task. **No SYCL kernel reads `SYCL_Host` memory**
  on this path, and a RED fails if one does (§7.3).
- There is no scheduler split and no host wait on the GPU path, which removes
  `llama.cpp-x1xd` and `llama.cpp-jhxq` from this path (§5.3). Under
  command-graph replay the gather is a **stated segment boundary** (§7.4).

Rev 2a put the HOST gather in the leading CPU split, as a CPU op next to
`inp_embd` (*lead ruling §T1*, C2), and let the first SYCL op read its output
in place. DQ forbids that read. The copy that replaces it cannot be made
wait-free from the leading split, because the next ubatch's CPU op would
overwrite the copy's source on the submitting thread (§7.1). Rev 2b therefore
routes HOST through the STORAGE path. That changes a *lead ruling §T1*
placement, so it is put to the lead as **DQ-C2** (§13).

The consumer contract that a model implements is §9.1.

**The measured facts that shape it** (details in §2 and Appendix A):

| fact | value |
|---|---|
| where the tables live | bcachefs `/models`: user data on two **USB** SSDs, NVMe holding promoted cached copies only |
| 4 KiB `O_DIRECT` random reads, Qwen table | **1389 / 7815 / 9124 IOPS** at QD 1 / 16 / 64 |
| 4 KiB `O_DIRECT` random reads, DeepSeek Q2_K `blk.1` table | **4249 / 7070-7242 IOPS** at QD 16 / 64 (the same file read **4657** at QD64 minutes earlier, while bcachefs wrote 7.5 GB of promotions to the NVMe) |
| bcachefs read amplification | **~56-62x** on every file: each 4 KiB read costs ~256 KiB of device reads, because `data_checksum=crc32c` is over whole 256 KiB extents |
| mmap page faults (the upstream path) | 1555 rows/s with 1 thread, 7866 with 16 threads: the same device ceiling |
| reference: ext4 on the root NVMe, same probe | 6019 / 82798 / 110110 IOPS (**~12x** bcachefs at QD64); the root disk has **33 GB free**, so moving the tables there is not an option (§13 D3) |
| the pread pool against a same-session fio QD64 control | **7051 rows/s against 7070 / 7242 IOPS** (ratio 0.99) |
| cold decode token, Qwen (16 rows) | 3.75 ms mean, 13.0 ms p99 |
| cold decode token, DeepSeek Q2_K (24 + 24 rows, two files, one pool) | `blk.1`'s 24 rows land at **7.13 ms mean, 14.6 ms p99**; all 48 at 11.2 / 21.0 ms |
| a decode token queued behind another context's 12000-row prefill | **1359 ms** FIFO, **21.7 ms** with priority (§5.2) |
| cold PP512 ubatch, Qwen (~7100 unique rows) | 0.85 s mean, 8.3 k rows/s |
| cold PP512 ubatch, DeepSeek (18-25 k unique rows) | **~2.5-3.5 s** at the measured 7.0-7.8 k rows/s |
| HOST gather and dequant, straight from the table | 4.2 us per 16-row token; ~2 ms per 8192-row ubatch |
| steady-state decode tokens with zero misses | Qwen, 512 MiB: prose 32%, mixed 63%. DeepSeek, 8 GiB: prose 20%, mixed 55% (48 rows must all hit) |
| `llama-bench` prompts | uniform random tokens, so **always cold**, at any cache size (§8.5) |

**What the planner will choose on this host** (§4.5). The host weights are
counted at their **exact** bytes (`llama.cpp-rnh3`), not at the 1.2x ratio the
fork charges today, and the host zones at the bytes the sizing code actually
commits (§2.3): `dma_staging_pool`, the RUNTIME pool and the **PP MoE oneDNN
ring** included. The ring has no type gate, so every MoE model carries it
today: 7.87 GiB on DeepSeek Q2_K and 4.17 GiB on Qwen, charged twice on the
host and once in VRAM. Every fit is therefore stated twice: on **today's
code**, and **after lever (s)** (`llama.cpp-sgc5`, `llama.cpp-s4ip` and the
ring type-gate, which the owner chose in §D4). The host budget follows
`MemAvailable`, which read 171.4-190.4 GiB today, so `host_budget` =
**163.4-182.4 GiB**. Under the default EXPERT mode the B50 holds no routed
expert, so these are B70-alone numbers (`llama.cpp-9g22` phase 1,
`level_zero:0`). The numbers are reproduced by
`docs/design/sycl_storage_tier_host_fit.py` (§4.5); `llama.cpp-7v5c`'s
dry-run replaces them.

- **Qwen:** one zone set's non-weight host terms are 15.4 GiB as committed on
  today's code and 5.6 GiB after (s). The 50.66 GiB table needs a
  `MemAvailable` of 176.4-179.8 GiB on today's code, so it is **STORAGE at
  171.4** and HOST at the other three readings. After (s) it needs
  162.4-165.8 GiB and is **HOST at every reading**.
- **DeepSeek Q2_K (the owner's primary rung):** one zone set's non-weight
  terms are 31.0 GiB as committed today and 9.5 GiB after (s). On today's code
  phase 1 is refused at every reading, even with floor caches and no draft.
  After (s) the planner's fit (`MemAvailable - 8`) admits it at three
  readings, but 9g22's gate precondition P3 (predicted post-load
  `MemAvailable >= 24 GiB`) fails at all four. **The owner's §D4 closes that
  gap**: the coldest Q2_K experts go to STORAGE and the CPU executes them. The
  cold set `C` is computed at load from that load's reading (*lead ruling
  §D5*); its size at each reading is `llama.cpp-9g22`'s number
  ([its §4.5.2](deepseek41-runtime.md#452-the-host-fit-r2-c-a-r3-c-1-c-2-owner-d4)),
  not restated here. If even every expert on STORAGE cannot clear P3, the
  load refuses by name (D8, §13).

STORAGE speed on this box is dominated by the filesystem's 60x amplification.
Plain NVMe is not an option here (D3), so for DeepSeek that filesystem is the
operating point, and §8 scores against it. On that filesystem DeepSeek's
"decent speed" is bounded by the cold experts' reads on the same devices, not
by the engram rows: a PP512 ubatch reads the whole cold set once (§8.4,
`9g22` §4.5.3).

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

The owner's decisions on the ticket (2026-09-26):

- The planner decides placement, with no configuration.
- HOST if the table fits after VRAM and experts, otherwise STORAGE, with async
  prefetch and a hot-row cache.
- The CPU executes the gather, so only the gathered rows move.
- "Decent speed" in either tier.
- DeepSeek-V4.1-Flash runs the **Q2_K** rung (owner §D2).
- The Q2_K fit (owner §D4): fix the zone waste with lever (s) (`sgc5`, `s4ip`
  and the ring type-gate), and put the coldest Q2_K routed experts on STORAGE,
  executed by the CPU. That sanctions STORAGE→CPU for experts. The cold set
  is profiled, with the `exp_probs_b` bias as the fallback (owner §D2).

The lead's rulings on this design (`lead-rulings-2026-09-26.md` §D3/§T1-§T4,
§D5): host weights are charged at exact bytes (`llama.cpp-rnh3`, which T2
depends on); the provider decides where the gather sits; the gather is a
SYCL-owned CPU `host_task` with event edges (D6(c)); one host budget function
that every consumer reads; STORAGE is a residency state; planned HOST
residency is locked or refused by name; the row cache is yieldable; plain NVMe
is not an option (D3); the gather's output reaches the device by an explicit
event-ordered H2D copy, never zero-copy (§T2 DQ); the cold set is sized at load
by one computation (§D5); row caches are fixed before the cold set is sized
(§T4 I-4).

Out of scope: the `qwen4exp` port itself (`llama.cpp-n77l`) and the `deepseek41`
runtime (`llama.cpp-9g22`); the GPU kernels that consume the gathered rows
(PLE's `ple_key`/`ple_value`/norms/conv, engram's `engram_q`/`engram_k`/
`engram_wkv`, all ordinary dense weights); and the MTP head. **The cold-expert
path itself** is in scope only at its interface. Owner §D4 brings
`llama.cpp-yl3v` forward to Q2_K, and yl3v owns the selection, the
`CpuExpertPool` execution, its landing slab and the prefetch from routing.
This design provides the residency state, the shared IO engine and the host
ledger that yl3v uses, and the fit that sizes the cold set (§4.3, §9.2).

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
  is `[256, 384016682]` (the last offset plus the last prime), 42241835020 B.
  In the Q2_K rung, the owner's primary, both extents are read from the
  headers (A.1): `blk.1` in `Q2_K-00001-of-00007` at offset 5785327904
  (32256518112 B) and `blk.14` in `Q2_K-00004-of-00007` at offset 4032
  (32257401288 B), **60.08 GiB** together.
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
  (`694ec2354 src/llama-model.cpp:1520-1528`); the fork's SYCL device reports
  none today (§2.3).
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
  CPU-only host tier.** Every cache-owned host byte is pinned USM host memory,
  which the driver maps into each device's VM. Pinning 50 GiB that no GPU
  touches spends locked, device-mapped memory for nothing. Whether xe also
  counts that memory as TTM `Shmem`, the resource behind this host's OOM
  history, is unverified, and the design does not depend on it.
- **The tier is derived, not requested.** `alloc_intent` is `{role, category,
  cohort_id, constraints}` (`unified-cache.hpp:5691-5696`): it has no tier
  field. `unified_select_tier()` derives one from the constraints
  (`must_device`, `must_host_pinned`, the cohort) and falls back to
  `HOST_PINNED` when the request exceeds the device's available bytes
  (`unified-cache.cpp:14780-14804`). `memory_location::host_accessible()` is
  simply `tier != DEVICE_VRAM` (`unified-cache.hpp:5625`).
- **Host budget: one fact, several sources.**
  - Each unified cache builds its own pinned pool, capped at construction by
    `min(total * pct, MemAvailable - max(8 GiB, total/32))`, pct 90 by default
    (`unified-cache.cpp:700`, `:4147-4259`).
  - With more than one GPU, `AUTO` mode is `PER_DEVICE`
    (`unified-cache.cpp:14292-14316`), so the dual-card configuration has
    **two** pools, each capped at the whole budget, each sampled at its own
    construction time.
  - `GGML_SYCL_HOST_RESERVE_MB` replaces the computed budget (strict parser,
    `host-reserve-env.cpp`, `llama.cpp-16el`), and `GGML_SYCL_NO_PINNED` sets
    it to zero (`:4248-4251`).
  - **The planner takes no host budget:** `compute_placement_plan` receives a
    VRAM budget only (`unified-cache.hpp:1591-1603`), and
    `plan.weight_host_bytes` (`:758`) is summed but never compared with
    anything. Host overcommit is found at allocation time, not at planning
    time. That is a P4 gap that already exists for host experts, and this
    design must close it (§4).
- **The host WEIGHT zone is 1.2x the host weights.**
  `populate_host_zone_sizing()` sizes it as `weight_host_bytes x 1.2 + 64 MiB`
  (`unified-cache.cpp:27089`, `:27670-27672`), and
  `ggml_sycl_configure_host_zones_for_plan()` eagerly commits WEIGHT + KV +
  STAGING + SCRATCH through `pre_allocate_host_pool()`
  (`76c7f6548 ggml/src/ggml-sycl/ggml-sycl.cpp:5615-5645`). On DeepSeek Q2_K
  that ratio alone adds ~30 GiB of slack nobody can name. `llama.cpp-rnh3`
  replaces it with exact bytes plus named terms; T2 depends on it (§4.2).
- **The STAGING and SCRATCH zones, and a separate RUNTIME pool, scale with the
  largest tensor.** All three are host-pinned, from the same pool budget.
  - STAGING is `s1_in_flight x (max(2 x max_tensor, max_staging_pair_bytes)
    + 2 MiB)`, with `s1_in_flight = 1` (`unified-cache.hpp:5539`), plus
    16 MiB, the expert bias, `moe_routing_ids` (`n_expert x 4 x n_ctx`),
    `tp_staging_buffer_bytes` and the MMID host pool
    (`unified-cache.cpp:27685-27699`). For DeepSeek's and Qwen's expert types
    the SOA staging pair equals `2 x max_tensor`, and the TP staging buffer is
    0 on one card.
  - SCRATCH is the sum of `max_tensor`, 32 MiB, `onednn_reorder`,
    `moe_q8_workspace`, `moe_cpu_expert_staging` (the largest MoE tensor) plus
    `moe_host_recipe_workspace`, `moe_vram_runtime`, `tp_vram_runtime`,
    **`dma_staging_pool`**, `pp_pipeline_scratch` and
    **`pp_moe_onednn_scratch`** (`:27716-27742`, the ring at `:27730`).
    `ggml_sycl_configure_host_zones_for_plan()` raises it to at least the
    cache's live oneDNN scratch sizes plus `max_tensor + 32 MiB`
    (`ggml-sycl.cpp:5634-5636`).
  - `pp_moe_onednn_scratch` is the **PP MoE oneDNN ring**. The loader sizes it
    for every MoE expert weight **with no type gate**
    (`c69d5774d src/llama-model.cpp:430-507`): `n_expert` MXFP4 WOQ weight
    slots, plus activation and output slots of `512 x n_expert x
    max(n_embd, n_ff)` rows at 2 and 4 B. `ggml-sycl.cpp:15962-15970`
    publishes it unconditionally. It serves only the MXFP4 WOQ executor, which
    Q2_K, Q3_K and Q8_0 experts never reach. It is **7.87 GiB** on DeepSeek
    Q2_K (2.24 + 1.88 + 3.75) and **4.17 GiB** on Qwen (0.42 + 1.25 + 2.50).
    It is charged in host SCRATCH, again in the RUNTIME pool below, and it sets
    the floor of each card's VRAM RUNTIME zone (`unified-cache.cpp:1567-1574`,
    `:4512-4530`), so on today's code it also takes that many bytes of expert
    room from the card.
  - `dma_staging_pool` is `2 x` the largest DMA-streamed tensor. It is sized
    **unconditionally**, whether or not weight streaming can engage; the code's
    own note says so and points at `llama.cpp-iaqm` (`:27522-27541`).
  - The **RUNTIME pool** is a separate pinned pool:
    `pre_allocate_runtime_chunks(onednn_scratchpad + dma_staging_pool +
    pp_pipeline_scratch + pp_moe_onednn_scratch)` on the cache's host arena
    (`ggml-sycl.cpp:5649-5653`, `unified-cache.cpp:20479-20483`,
    `pinned-pool.cpp:272-305`: a separate `runtime_chunks_` list charged to the
    same budget). So the DMA staging pool and the ring are each charged twice
    on the host, once in SCRATCH and once here. **Lever (s)** removes both
    where no host reader exists and type-gates the ring to MXFP4 experts
    (`llama.cpp-s4ip`, comments `c-rbwa` and `c-td91`, widened to the `:5649`
    site); it is part of the owner's §D4.
  - `cpu_quant_buffer_bytes` is **not** a host term. The code states it is
    diagnostic-only: no allocation is sized from it (`:27436-27446`).

  One zone set's non-weight host memory (§4.5, KV zone included, the RUNTIME
  pool as committed in 2 GiB chunks):

  | | today's code | after lever (s) |
  |---|---:|---:|
  | DeepSeek Q2_K (`max_tensor` 1.81 GiB) | STAGING 3.65 + SCRATCH 15.26 + RUNTIME 11.58 (commits 12) + KV 0.06 = **31.0 GiB** | 3.65 + 3.77 + 0.09 (commits 2) + 0.06 = **9.5 GiB** |
  | Qwen3.8 (`max_tensor` 0.83 GiB) | 1.69 + 7.63 + 5.92 (commits 6) + 0.06 = **15.4 GiB** | 1.69 + 1.80 + 0.09 (commits 2) + 0.06 = **5.6 GiB** |

  A 30 GiB engram table counted as `max_tensor` would make STAGING alone
  ~60 GiB and move SCRATCH and RUNTIME by as much again, so the lazy tables
  must stay out of that inventory (§4.1).
- **Every participating cache builds its own zone set, WEIGHT included**
  (`llama.cpp-sgc5`). A two-card run gives every cache the same plan snapshot
  (`ggml-sycl.cpp:15325-15352`), and the S1 preload calls
  `ggml_sycl_configure_host_zones_for_plan()` for each device (`:33709-33888`).
  For a host-expert plan the second cache would commit a second ~150 GiB WEIGHT
  zone. T2 depends on sgc5 (§4.2).
- **CPU-buffer weights.** A weight in a CPU-owned buffer leaves the SYCL
  inventory, and the post-`create_tensor` replan treats its layer as host.
  The canonical contract §8 items 2 and 4 place CPU-buffer tensors outside the
  contract. Upstream's lazy tensor is exactly such a tensor, so merged as-is it
  would sit outside P1/P2/P4 by construction.
- **`mmap_support`.** The fork's `ggml_backend_dev_caps` has the field
  (`76c7f6548 ggml/include/ggml-backend.h:174`), but the SYCL device's
  initializer lists only four fields (`ggml-sycl.cpp:107797-107802`, `.async` at `:107798` through `.events` at `:107801`), so it is
  value-initialized to **false**. Upstream SYCL sets it `true`
  (`694ec2354 ggml/src/ggml-sycl/ggml-sycl.cpp:6380`), and upstream's loader
  turns `AUTO` into `OFF` when any device reports false
  (`694ec2354 src/llama-model.cpp:1519-1528`). Merged as-is, the default
  `AUTO` would silently disable every lazy table on this host (§4.1).
- **Hooks.** The fork calls into the SYCL backend through
  `ggml_backend_reg_get_proc_address` hooks
  (`76c7f6548 src/llama-model.cpp:63-111`). This design adds its entry points
  the same way.
- **Custom ops.** SYCL declines `GGML_OP_CUSTOM`: `supports_op` falls to
  `default: return false`
  (`76c7f6548 ggml/src/ggml-sycl/ggml-sycl.cpp:108771-108772`), so a custom op
  is scheduled on the CPU backend.
- **Backend-owned CPU work.** Host experts are **not** a scheduler split in the
  fork: `MUL_MAT_ID` stays a SYCL node, and the backend hands the host experts
  to `CpuExpertPool` workers (`cpu-dispatch.hpp:115-128`,
  `ggml-sycl.cpp:76048-76067`). That path still has host waits of its own (it
  waits on the activation's D2H event and on each entry's ready event before
  submitting, `:76033-76043`, e.g. `act_deferred_evt.wait()` at `:76034`), so it is the precedent for *ownership*, not for
  waits. The event-edge precedent is the queue-submitted `host_task` with
  `depends_on` (`ggml-sycl.cpp:75262-75272`).
- **Command-graph replay.** One exec-graph slot per context, keyed by
  `(graph_hash, n_nodes, device, phase)` (`ggml-sycl.cpp:99164-99173`). A
  different signature re-records; after 12 consecutive re-records with no
  replay, `exec_graph_replay_futile` turns record/replay off for the context
  (`:99226-99240`, threshold at `:99228`). MoE decode graphs can instead replay **segments** between
  direct-dispatched `MUL_MAT_ID` nodes (`:97186-97270`), and
  `GGML_SYCL_GRAPH_DIAG` counts both kinds (`:632-660`, printed at `:794-830`).
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
- Sequential 1 MiB `O_DIRECT` reads of the table reached **1.78 GB/s** (a
  whole-table read in ~31 s), but that read was served mostly from the NVMe's
  cached copies (A.2: 19.7 of 27.6 GB from dev0). Settled data reads at
  0.29-0.47 GB/s (§9.2.2), so a whole-table read is minutes (D1).

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

**The DeepSeek Q2_K tables, rev 2 (the owner's rung, now fully on disk).**
`blk.1.engram_embd` is in `Q2_K-00001-of-00007` at file offset 5785327904
(32256518112 B) and `blk.14.engram_embd` in `Q2_K-00004-of-00007` at 4032
(32257401288 B). The shards were written between 17:23 and 18:27 and probed
at ~18:36 (load average 31-52, no download running). Three things came out of
it, and each changes how targets must be scored:

- **The ceiling moves with bcachefs's own background work.** fio QD64 over
  `blk.1` read **4657 IOPS, p99 146 ms** while bcachefs wrote **7.5 GB** to the
  NVMe in the same 15 s (promotion of the extents being read, plus reconcile of
  the fresh shard). Two minutes later the same probe read **7070** and **7242
  IOPS** with 0.4-0.5 GB written. A 1.55x swing on the same extent within
  minutes is why §8 scores the engine as a **ratio to a same-session fio
  control**, never against an absolute rate.
- **The pread pool is at the device ceiling.** Bracketed by those two fio runs,
  the 64-worker pool read a 20000-row batch at **7051 rows/s** (ratio 0.99).
- **Amplification is unchanged at 56.5-61.7x**, and 1.9-2.2% of 84 B rows
  straddle a 4 KiB block.

---

## 3. Object model: the lazy table as a unified-cache object (item 1)

### 3.1 One new allocation tier, and residency as a separate state

**Allocation tier.** `alloc_tier` gains one value, `HOST_CPU`: host memory that
only the CPU reads.

| tier | memory | who may read it | new? |
|---|---|---|---|
| `DEVICE_VRAM` | device USM | that device | no |
| `HOST_PINNED` | USM host, pinned pool, mapped into each device's VM | CPU, and the GPU for DMA staging only | no |
| **`HOST_CPU`** | anonymous `mmap`, `MADV_HUGEPAGE`, **`mlock`ed**, not device-mapped | **CPU only** | **yes** |
| `MMAP_TRACKED` | legacy: file mmap tracked for weights | CPU | no |

- **Requested only explicitly.** `alloc_constraints` gains `must_host_cpu`, and
  `unified_select_tier()` maps it to `HOST_CPU`
  (`76c7f6548 ggml/src/ggml-sycl/unified-cache.cpp:14780-14804`). Nothing falls
  back **into** `HOST_CPU`: the size-exceeds-available fallback stays
  `HOST_PINNED`, and a cohort cannot inherit `HOST_CPU`. A `HOST_CPU` request
  never falls back **out** either: if it cannot be admitted or locked, it is
  refused.
- **Locked, and refused by name if it cannot be** (*lead ruling §T1*, I4). The
  raw wrapper is `unified_cache_raw_alloc_host_cpu(size)`:
  `mmap(MAP_PRIVATE | MAP_ANONYMOUS)`, `madvise(MADV_HUGEPAGE)`, then `mlock()`
  over the whole range. `mlock` without `MLOCK_ONFAULT` faults every page in
  and locks it before the call returns, so the resident bytes equal the planned
  bytes from the first moment, and a failure surfaces at load, not mid-run. If
  `mlock` fails, the wrapper unmaps and the allocation is refused as
  `host-cpu-lock-failed` with `errno`, the requested bytes and `RLIMIT_MEMLOCK`
  in the message. This is not hypothetical on this host: `SwapFree` read
  **37 MB of 8 GiB** today, so unlocked pageable memory *would* be swapped, and
  a "resident" table would silently become disk reads. `ulimit -l` is
  unlimited here. The wrapper and its `unified_cache_raw_free_host_cpu`
  counterpart are added to the canonical contract's §3 allowlist exactly like
  `unified_cache_raw_malloc_host`, with a line saying the memory is locked, so
  no other file calls `mmap`/`mlock` for it (P1).
- **Why locked host memory and not the pinned path.** Pinned USM host memory
  would also be resident, but the driver maps it into both devices' VMs for a
  table no GPU may read (P3), and it rides the pinned pool's 2 GiB chunking.
  `mlock` gives the residency guarantee without the device mappings.
- **Not device-mappable.** `memory_location::host_accessible()` stays true for
  `HOST_CPU` (`unified-cache.hpp:5625`), because the CPU reads it. A new
  `memory_location::device_mappable()` is false for it, and every path that
  hands a host-accessible pointer to a device must check it instead.
- **The owner-first front door.** A `HOST_CPU` allocation is requested through
  `unified_allocate_owner(req)` with `constraints.must_host_cpu = true`. The
  owner class is **`EXTERNAL_EXACT`**: each allocation is exact and
  independently released, not a slice of a cache backing pool. This keeps the
  two-site `CACHE_BACKING` allowlist (canonical contract §3.1) untouched.
- **The tier-switch census T1 must audit.** `alloc_tier::HOST_PINNED` or
  `MMAP_TRACKED` is named 36 times in 7 files (`unified-cache.cpp` 25,
  `ggml-sycl.cpp` 5, `common.hpp` 2, and one each in `binbcast.cpp`,
  `dmmv.cpp`, `element_wise.cpp` and `unified-cache.hpp`). The sites that must
  learn `HOST_CPU` explicitly are the two switches `alloc_tier_name()`
  (`unified-cache.cpp:2403`) and `offload_stats_note_alloc()` (`:2606`),
  `unified_free_record()` (`:14806`), `query_registered_location()`
  (`:21595`), `query_location()` (`:21643`), the allocation registry, and the
  shutdown census (`unified_cache_shutdown_runtime_alloc_census_for_test()`,
  `:19473`, and its siblings).

**Residency.** STORAGE is **not** an allocation tier (*lead ruling §T1*, I3). No
byte is ever allocated "in" it, and an allocatable enum value would be a latent
path to request it. A `lazy_table` instead carries `lazy_residency { HOST,
STORAGE }`, the planner's decision. The reporting enums `cache_location`
(`unified-cache.hpp:2167`) and `expert_resolve_tier` (`:2313`) gain `STORAGE`
as a *reported* state only; nothing allocates against them. The cold
routed-expert set of owner §D4 (`llama.cpp-yl3v`, §9.2) reports the same
`STORAGE` value.

### 3.2 The `lazy_table` object

One `lazy_table` per lazy tensor per loaded model. It is **model-scoped**: the
rows are immutable weights, so every context of the model shares it.

```
lazy_table
  identity   : ggml_sycl_cache_id (owner = model token, name = tensor name)  // the usual weight key
  geometry   : type, ne[0] (row elements), n_rows, row_size (ggml_row_size)
  source     : file path + file index, byte offset, nbytes   // from llama_tensor_weight
  residency  : HOST | STORAGE                                // the planner's decision, immutable
  consumer   : consumer_layer, the SYCL device that executes it (from the plan)
  HOST:    body   : mem_handle (HOST_CPU, WEIGHT)
  STORAGE: fd     : O_DIRECT file descriptor, owned (closed in the destructor)
           cache  : lazy_row_cache (§6.3)
  refs       : intrusive count: the model's descriptor buffer holds one, every outstanding ticket one
  stats      : §10
```

- **Ownership (P2).** Every memory the object uses is a `mem_handle` minted by
  `unified_allocate_owner()` and held by the object. The object is destroyed
  when its **last reference** drops: the model's, released at unload (the §12
  lifecycle of the canonical contract), or the last outstanding ticket's,
  released by the reaper after its final completion (§5.3). Its destructor
  drops the handles and closes the fd, once. That is the sole release point,
  and it can never run while a read into its row cache is in flight, so it
  needs no teardown wait.
- **The descriptor buffer.** The lazy tensor gets a buffer of a new SYCL
  buffer type, `ggml_backend_sycl_lazy_buffer_type(device)`, where `device` is
  the device that executes the table's consumer layer. It maps nothing: its
  base is a non-dereferenceable sentinel, its nominal size is `ggml_nbytes`,
  `is_host` is false, usage is `WEIGHTS`, and `get_tensor`, `set_tensor`,
  `memset_tensor` and `cpy_tensor` abort naming the table. Its context holds
  the `lazy_table` reference. It exists for three reasons:
  - `ggml_gallocr` treats a leaf whose `data` is null as unallocated
    (`76c7f6548 ggml/src/ggml-alloc.c:594-595`), so a bufferless 30-50 GiB
    tensor that reached a graph would be *allocated in the compute buffer*.
    The sentinel makes it read as allocated.
  - The scheduler assigns an op whose source sits in a `WEIGHTS` buffer to the
    highest-priority backend that supports that buffer type and the op
    (`ggml-backend.cpp:1594-1612`, rule `1.wgt` at `:1684`). That is how the
    SYCL backend claims the gather of either residency deterministically (§7.2).
  - `llama` sees an ordinary buffer, so model size, memory breakdown, the
    loader's mlock loop and `load_all_data` need only a narrow change (T6b):
    skip the tensor, and report the buffer's *resident* bytes, not its
    nominal size.

### 3.3 The resident bytes of a lazy table, all planned

| allocation | scope | tier / role | size |
|---|---|---|---|
| HOST body | model | `HOST_CPU` / `WEIGHT` | `table_bytes` (Qwen 50.66 GiB) |
| row cache slab | model | `HOST_CPU` / new `alloc_role::LAZY_ROW_CACHE` | `slots x row_size`, in chunks of 256 MiB so that it can yield (§6.4) |
| row cache index | model | `HOST_CPU` / `LAZY_ROW_CACHE` | `slots x 13.3 B` (§6.3): 7.8% of the slab on 170 B rows, 12.1% on 110 B, 15.9% on 84 B |
| landing slab | context, per table | `HOST_CPU` / new `alloc_role::LAZY_STAGING` | `depth x n_ubatch x rows_per_token x row_size`: Qwen 2 x 512 x 16 x 170 B = 2.66 MiB; DeepSeek Q2_K 2 x 512 x 24 x 84 B = **1.97 MiB per table** |
| block ring | context, shared by its tables | `HOST_CPU` / `LAZY_STAGING` | `QD x 2 x io_block` = 64 x 8 KiB = **512 KiB** (4 KiB-aligned, for `O_DIRECT`) |
| output slot | context, per table, both residencies | `HOST_PINNED` / new `alloc_role::LAZY_OUTPUT` | `depth x n_ubatch x rows_per_token x ne[0] x 4 B`: Qwen 2 x 512 x 16 x 160 x 4 B = **10 MiB**; DeepSeek 2 x 512 x 24 x 256 x 4 B = **24 MiB per table** |

- **STORAGE** uses the cache slab and index, the landing slab, the block ring
  and the output slot. Those are the **entire** resident footprint of a
  STORAGE table, and the planner charges exactly those bytes (P4).
- **HOST** uses the body and the output slot. Its host task dequantizes
  straight from the body (§7.2), which is what the 4.2 us/token measurement
  did, so it needs no landing slab and no ticket.
- **The output slot is pinned** because the device's copy engine reads it. It
  is the host end of the explicit H2D copy (§7.2 step 3, §7.3), which is DMA
  staging, the one GPU use `HOST_PINNED` allows (§3.1). No kernel reads it.
- **Not a th32 allocation: the cold-expert landing slab.** The owner's §D4
  cold set is read by `llama.cpp-yl3v` into its own planned, cache-owned
  `O_DIRECT` slab. Lead ruling §D5 charges it as `2C/40` (double-buffered, one
  layer's cold bytes) inside the cold-set computation; §4.2 lists it as a term
  so that `host_planned_bytes` carries it.

### 3.4 Pointer discipline

The gather obtains raw views (a chunk of the HOST body, a row cache slot, a
landing-slab slot, an output slot) by resolving the owning handle **inside**
the gather call, and never stores them. That is the same rule as the canonical
contract §4, and the new resolution entry point is added to its allowlist. No
table is keyed by a pointer: the row cache is keyed by `(row index)` within its
`lazy_table`, whose identity is the weight's `ggml_sycl_cache_id`.

---

## 4. The planner decides the residency (item 2)

### 4.1 Inventory

The loader gives lazy tensors to the SYCL inventory as a **distinct class**,
`placement_lazy_table_info`. It carries the name, type, `ne`, `nbytes`,
`row_size`, `rows_per_token`, `consumer_layer`, file index and offset. They are
never placement entries for a device: P3 forbids a device tier for them, so the
planner cannot offer one. They are also **excluded from `max_tensor_bytes`**
and every other inventory maximum: with a 30 GiB engram table counted as
`max_tensor`, the host STAGING zone alone would be ~60 GiB (§2.3). T3's RED
fixture plans a Q2_K-shaped inventory with its two tables and asserts that the
STAGING and SCRATCH zones equal what `populate_host_zone_sizing()` returns for
the same inventory **without** the tables. The expected bytes come from the
sizing function, never from a literal (*lead ruling §D5*): today that is about
3.65 GiB of STAGING, and it moves when `s4ip` lands. Counting a table would
add at least twice the table's bytes, so the check cannot pass by accident.
`rows_per_token` and `consumer_layer` are arch facts that `llama` supplies
(§9.1):

| model | table | `rows_per_token` | `consumer_layer` |
|---|---|---:|---:|
| qwen4exp | `per_layer_token_embd` | 16 (`ple_n_heads`) | 1 |
| deepseek41 | `blk.1.engram_embd` | 24 (`(max_ngram_size - 1) x head_count`) | 1 |
| deepseek41 | `blk.14.engram_embd` | 24 | 14 |
| gemma4 | `per_layer_tok_embd` | 1 (token-indexed) | 0 |

`consumer_layer` orders IO (§5.2), picks the device whose descriptor buffer
holds the table (§3.2), and defines the prefetch window (§5.4).

On the `llama` side this happens where `lazy_read::add()` records the tensor
(`694ec2354 src/llama-model-loader.cpp:1088-1117`). The SYCL path, when
`llama_model_sycl_hooks_enabled()` holds, records it in the inventory
**instead of** adding its range to the mmap lazy set, and gives the tensor a
descriptor buffer (§3.2); nothing ever reads `tensor->data` (§7 replaces every
consumer). The memory-fit pass (`no_alloc`) and the real load take the same
decision, which preserves upstream's invariant at
`694ec2354 src/llama-model-loader.cpp:1342-1345`.

**The SYCL branch does not depend on `mmap_support`.** The SYCL path reads the
table through its own `O_DIRECT` fd, so whether a device can map files is
irrelevant to it. Two changes follow (T6): the fork's SYCL device reports
`mmap_support = true`, as upstream does, and the SYCL branch resolves the
lazy mode *before* upstream's `AUTO`-to-`OFF` fallback, so that fallback cannot
disengage it (§2.3). T6's RED: with `AUTO` and a device reporting
`mmap_support = false`, the dual-card configuration still yields
`[PLACEMENT] lazy ... residency=...` lines.

### 4.2 One host ledger, and the planner's host budget

*Lead ruling §T1*, I2: one host-budget function, read by every consumer. This
design makes it a **process-global host ledger** in `unified-cache.cpp`:

- **The budget.** `host_ledger_budget()` =
  `min(total x pct, MemAvailable - max(8 GiB, total/32))`, the formula the
  pinned pools use today (`unified-cache.cpp:4147-4259`), or the
  `GGML_SYCL_HOST_RESERVE_MB` override when its strict parser accepts it.
  `MemAvailable` is **sampled once per plan publish**. Because a later sample
  already excludes this process's own resident bytes, a later model's budget
  is `ledger_committed + (MemAvailable_now - os_reserve)`, capped by
  `total x pct`.
- **Every host byte charges it.** Pinned-pool chunk growth, every `HOST_CPU`
  allocation, and every row-cache chunk ask the ledger first. The pools stop
  carrying caps of their own, so in `PER_DEVICE` mode the two pinned pools are
  capped by their **sum**, not each by the whole budget.
- **Enforced at allocation, predicted by the plan.** An allocation the ledger
  refuses after a plan said "fits" is a plan-versus-reality defect: it logs
  `[HOST-PLAN-BUG]` at ERROR with the plan's and the ledger's numbers, then
  refuses. It never falls back to an unaccounted allocator.
- **Which cache owns lazy-table memory in `PER_DEVICE` mode.** The cache of
  the device that executes the table's consumer layer (§3.2). Its registry
  holds the records, and its shutdown census counts them. Every byte still
  charges the one process-global ledger.
- **`GGML_SYCL_NO_PINNED`** disables the pinned **mechanism**, not the budget.
  The plan is the same plan, and host weights are still charged at their exact
  bytes wherever they then live. The `HOST_CPU` tier is not pinned, so
  `NO_PINNED` does not touch a lazy table. No model is refused because
  `NO_PINNED` is set.
- **`GGML_SYCL_HOST_RESERVE_MB`** *is* the budget (it is named "reserve" but,
  since `llama.cpp-glkg`, it overrides the budget, `:4152-4198`). A value too
  small for a model refuses that model, which is the override doing its job,
  with the refusal naming the variable. A malformed value is rejected with one
  WARN and the computed budget is used. A value above the computed budget logs
  one WARN naming both. No other variable changes the budget, so no model is
  refused because of an unrelated override.

`compute_placement_plan` and `compute_multi_device_plan` gain the ledger's
budget as an input. The plan gets:

```
size_t host_budget;                 // the ledger's budget at this publish, recorded
size_t host_planned_bytes;          // everything below, summed
std::vector<placement_lazy_table> lazy_tables;
struct placement_lazy_table {
    std::string    name;
    lazy_residency residency;       // HOST or STORAGE
    int            consumer_device;
    size_t         table_bytes;     // file bytes
    size_t         host_resident_bytes; // HOST: table_bytes; STORAGE: cache slab + index
    size_t         cache_slots;     // STORAGE only
    size_t         cache_floor_slots;
    size_t         staging_bytes_per_context; // landing slab + ring share + output slot
};
```

<a id="th32-host-fit"></a>
**T2's definition** (it depends on `llama.cpp-rnh3`; its enforcing phase also
depends on `llama.cpp-sgc5`):

`host_planned_bytes` is what the sizing code will commit, not a second
estimate of it. Its zone terms are the bytes `populate_host_zone_sizing()`
and `ggml_sycl_configure_host_zones_for_plan()` produce, term for term (§2.3):

  the WEIGHT zone: the host weights at their **exact** materialized bytes
  (`rnh3`: the same function as `ggml_sycl_weight_alloc_bytes`, including
  layout padding, with **no** ratio), plus `rnh3`'s named terms (the 64 MiB
  WEIGHT runtime reserve, `HOST_COMPUTE` requests such as `output_reserve`'s
  logits and the server's prompt-checkpoint reads, and a TLSF fragmentation
  bound derived from the block count)
+ the KV zone: the host KV, at least 64 MiB
+ the STAGING zone, as sized at `unified-cache.cpp:27685-27699`:
  `max(2 x max_tensor, max_staging_pair_bytes) + 2 MiB`, plus the 16 MiB
  headroom, the expert bias, `moe_routing_ids`, `tp_staging_buffer_bytes` and
  the MMID host pool
+ the SCRATCH zone, as sized at `:27716-27742` and raised at
  `ggml-sycl.cpp:5634-5636`: every term §2.3 lists, `dma_staging_pool` and the
  PP MoE oneDNN ring included on today's code
+ the RUNTIME pool that `pre_allocate_runtime_chunks` commits
  (`ggml-sycl.cpp:5649-5653`), the ring and the DMA pool included on today's
  code
+ each pool's rounding up to its commit chunk (2 GiB for the pinned pools),
  because the ledger charges committed chunks
+ the cold-expert landing slab, `2C/40` for a cold set of `C` bytes, and
  **minus** those `C` bytes from the host experts (owner §D4, *lead ruling
  §D5*; `C` is 0 for a model with no cold set)
+ the **context envelope**: per context, the host tenants a context adds (the
  CPU backend's `SYCL_Host` compute buffer, which `llama.cpp-zhcn` plans; host
  KV growth; the lazy tables' landing slabs, block ring and output slots),
  times the planned context count
+ every lazy table's `host_resident_bytes`.

Two terms are deliberately absent. `cpu_quant_buffer_bytes` is
diagnostic-only (`:27436-27446`), and `moe_cpu_expert_staging` is counted once,
inside SCRATCH.

Three details make "term for term" hold:

- **The SCRATCH raise reads live cache state.** `ggml-sycl.cpp:5634-5636`
  takes the max with `cache->onednn_weights_scratch_size() +
  onednn_activations_scratch_size()`, and a host-only dry-run has no cache. T2
  and `llama.cpp-7v5c` substitute the plan-time value of the same scratch, from
  the value function its owner defines (*lead ruling §M19*: `llama.cpp-moua`'s
  `onednn_scratchpad`). The raise is a max: it binds only when those two sizes
  exceed SCRATCH's terms other than `max_tensor + 32 MiB`, which after (s) are
  0.94 GiB on Qwen and 1.92 GiB on DeepSeek. Whether they do on these models
  is for 7v5c to print; §4.5 assumes they do not.
- **The ledger charges backing bytes.** Chunk growth charges
  `footprint.backing`, not `footprint.usable` (`pinned-pool.cpp:889-895`), so
  T2 does too.
- **A short pre-allocation becomes a refusal.** Today
  `pinned_chunk_pool::pre_allocate` logs a WARN and carries on when the budget
  runs out (`pinned-pool.cpp:241-246`), so a zone set can be committed in part
  without anyone noticing. Under T2's enforcing phase that shortfall is the
  typed refusal `host-budget-exceeded`. In the WARN-only phase it is logged
  with the plan's number next to it.

**One zone set per plan (`llama.cpp-sgc5`).** Today every participating cache
builds a full zone set, WEIGHT included, from the one plan (§2.3). T2 does not
sum that. It counts the WEIGHT zone once, in the cache whose `CpuExpertPool`
leases the host experts, and STAGING, SCRATCH and RUNTIME once per cache that
sgc5's fix still gives them (from that device's own maxima). Until sgc5 lands,
the ledger's sum cap turns the duplicate WEIGHT zone from an OOM into a
refusal that no lever can clear. So T2's enforcing phase (b) waits for sgc5;
its WARN-only phase (a) does not, and proceeds now (*lead ruling §T3*, (c)).
Phase 1 of `llama.cpp-9g22` runs on the B70 alone (`level_zero:0`), which has
one cache and one zone set whether or not sgc5 has landed.

**One source for the number** (*lead rulings §T3, §D5*). `llama.cpp-7v5c`
(9g22's T27) is the host-only plan dry-run: `compute_multi_device_plan`,
`populate_host_zone_sizing` and the `configure_host_zones` sizing, run over the
GGUF header inventory and recorded device budgets, with no GPU. It also
computes the cold set `C` for the launch's reading (§4.3). T2's
`host_planned_bytes`, 7v5c's printout and `9g22`'s `expected_host` are one
computation through the same functions. Hand arithmetic, including §4.5's, is
an estimate that 7v5c replaces. It has already missed terms three times: the
r1 text charged the diagnostic-only `cpu_quant_buffer_bytes`, the r2 text left
out the DMA staging pool and the RUNTIME pool, and rev 2a left out the ring.

**The readings are samples, not a shared fact.** This document took four
`MemAvailable` readings (171.4, 181.8, 183.7 and 190.4 GiB); `9g22` rev 5 cites
171.4 and 189.6. Both are hand samples of a quantity that moves with the other
tenants. The number that decides a load is the one 7v5c takes at that load, so
neither document's table is a gate.

### 4.3 The decision, in order

1. VRAM is placed exactly as today: dense first, then experts.
2. Host weights (exact), `rnh3`'s named terms, host KV and every carved host
   zone are charged against `host_budget`. For a model with no cold-expert
   class, a plan that exceeds the budget refuses with a typed reason
   (`host-budget-exceeded`, carrying each term); today it silently
   overcommits (§2.3). For a model that has one (DeepSeek phase 1, owner §D4),
   the host experts are **not** refused here: step 5 moves the coldest of them
   to STORAGE.
3. The **context envelope** for the planned contexts is charged next, and
   `host_reserve = max(2 GiB, 1% of the budget)` is set aside. Context-lifetime
   tenants therefore come before any lazy-table byte (*lead ruling §T1*, I8;
   `llama.cpp-moua`'s lifetime segregation puts yieldable tenants last).
4. For each lazy table, largest first: if `table_bytes <= leftover`, the
   residency is **HOST**, and `leftover -= table_bytes`. Otherwise it is
   **STORAGE**, with a cache floor of `2 x n_ubatch x rows_per_token` slots (one
   double-buffered ubatch working set) plus its index. If `leftover` cannot
   hold the floors, the plan refuses (`lazy-table-cache-below-floor`) rather
   than run with a cache that cannot hold one ubatch. With a cold-expert class,
   "leftover" here is measured with the cold set at `C_selected` (step 5): a
   table may be HOST only if that does not make step 5 demote more experts, so
   a table never pushes an expert to STORAGE (§4.4). In a plan with a cold
   set, each STORAGE cache is also **fixed** in this step at its target size,
   index included: the gate size, 2 GiB per table, which this design proposes
   as the default for DeepSeek (never below the floor) (*lead ruling §T4*,
   I-4).
5. **With a cold-expert class: the cold set** (owner §D4; *lead ruling §D5*).
   `llama.cpp-7v5c` computes `C = max(C_selected, C_P3)` from this load's
   `MemAvailable` reading. `C_selected` is what the owner's §D1/§D2 selection
   already demotes (the usage profile, then the `exp_probs_b` bias). `C_P3` is
   the smallest cold set that leaves a predicted post-load `MemAvailable` of at
   least 24 GiB (9g22's P3), counting the landing slab (`2C/40`) and the main
   pool's chunk rounding. The experts themselves are chosen by the §D1
   selection; this step only sizes `C`. Everything charged in steps 2-4, the
   fixed caches included, is charged **before** `C` is sized, and `C` absorbs
   the remainder: row caches and the cold set never trade bytes at run time
   (*lead ruling §T4*, I-4). If even `C` = every routed expert cannot clear
   P3, the load **refuses** as `cold-set-cannot-clear-p3`, naming the reading,
   the P3 threshold and the shortfall; it never loads below P3. The step is
   `llama.cpp-yl3v`'s, through 7v5c's computation; th32 consumes its `C` and
   slab bytes in `host_planned_bytes` (§4.2).
   **Without one** (Qwen): the STORAGE caches take what is left as a
   **yieldable** tenant (§6.4), split between tables in proportion to
   `table_bytes x rows_per_token`, a proxy for miss traffic.

Nothing in this sequence reads an environment variable except the ledger's
budget override (§4.2). `--lazy-mode` keeps its upstream meaning of
**eligibility**: `OFF` means the tensor is an ordinary weight and never a lazy
table, and `AUTO`/`ON` pick the eligible set. The residency is always the
planner's (DLM, *lead ruling §T2*).

### 4.4 Interaction with host experts and later tenants

- **Memory.** Host experts are charged before lazy tables (step 2 before step
  4), which is the owner's ticket ruling ("after VRAM and experts"). What
  moves an expert to STORAGE is the host budget at P3 (step 5, owner §D4),
  never a lazy table: a table becomes HOST only out of what is left once the
  warm experts, the zones, the envelope and the fixed caches are charged, at
  `C_selected`.
- **Execution.** Both residencies' gathers run on a SYCL runtime `host_task`
  thread. A HOST one takes microseconds; a STORAGE one mostly sleeps on its
  ticket. The CPU cost is the dequant of the rows (~2-3 ms per PP512 ubatch per
  table, §2.4 and A.2), which does not compete with `CpuExpertPool`'s compute
  workers in any measurable way. The pread workers sleep in the kernel.
- **IO.** The cold experts and the engram tables live on the same bcachefs USB
  members, and share the one IO engine (§5.2). How much that costs is in §8.4.
- **Later tenants.** A table's residency is immutable after the plan is
  published: a live HOST table is never demoted. Its **row cache** is
  different, because it holds only clean copies of file bytes: no lease, no
  pin, and every in-flight row lives in a landing slab, not in the cache
  (§5.3). A later context or model that needs host memory the plan cannot
  otherwise find therefore **yields** row-cache chunks (§6.4), exactly as KV
  admission yields optional layout copies today
  (`unified-cache.cpp:7798-7911`, `yield_optional_layouts`). *Lead ruling
  §T1*, I8, settles that shrinking a clean-data cache is not the forbidden
  reclaim of live memory.

<a id="th32-host-fit-numbers"></a>
### 4.5 What the planner will choose on this host (through T2's definition)

**How to reproduce these numbers.** From the repository root, run
`python3 docs/design/sycl_storage_tier_host_fit.py`. It needs no packages, and
it prints every figure in this section: the ring, the per-set terms in both
columns, the Qwen rows, DeepSeek cases A-F, the P3 check, and a cross-check of
`9g22`'s cold set. It is hand arithmetic over the terms of §4.2, and
`llama.cpp-7v5c`'s dry-run replaces it.

**The budget.** `MemTotal` is 246.9 GiB, so `total x 90%` = 222.2 GiB and the OS
reserve is 8 GiB. `MemAvailable` read **171.4, 181.8, 183.7 and 190.4 GiB** at
four points today, moved by the other tenants (the qemu VM, Emby, Frigate), so
`host_budget` = **163.4-182.4 GiB**. The tables below state each case's
requirement as the `MemAvailable` it needs at load (requirement + 8 GiB), so
that it can be checked against any reading (§4.2: these readings are samples,
not a gate).

**Two columns.** *Today's code* is the sizing code at `c69d5774d`. *After (s)*
is lever (s) of the owner's §D4: the ring type-gated to MXFP4 experts (so 0 on
both models here), and the VRAM-resident `dma_staging_pool` and ring removed
from host SCRATCH and from the RUNTIME pool (`llama.cpp-s4ip`), with one zone
set per plan (`llama.cpp-sgc5`).

**The exact weights** (GGUF tensor headers, `ds_q2k_census.py` and
`qwen_census.py`, Appendix A.1):

| model | routed experts | other weights | lazy tables | total |
|---|---:|---:|---:|---:|
| Qwen3.8-Flash-Next Q8_0 | 119.53 GiB (largest tensor `blk.0.ffn_down_exps` 850 MiB) | 5.09 GiB (dense 3.83, `token_embd` 0.63, `output` 0.64) | 50.66 GiB | 175.29 GiB |
| DeepSeek-V4.1-Flash Q2_K | 183.25 GiB (4.58 GiB per layer; largest tensor `ffn_down_exps` 1856.25 MiB) | 3.01 GiB (dense 2.30, `token_embd` 0.20, `output` 0.51) | 60.08 GiB | 246.34 GiB |

**The device side: the B70 holds every device expert** (estimates; the gate-1
plan lines are the witness). Under the default EXPERT multi-GPU mode the fork
places no routed expert on the B50 at all. The static secondary route returns
`false` for every type (`unified-cache.cpp:24961-24974`, pending a PP
secondary-route equivalence gate), and the secondary executor exists only for
MXFP4, Q4_0 and Q6_K (`:24943-24958`). So `llama.cpp-9g22`'s phase 1 pins
`level_zero:0`, the B70 alone (its §4.5.1), and the same holds for Qwen's
Q8_0 experts.

- **After (s).** With a placement budget of ~29.9 GiB, and after dense weights
  (DeepSeek ~2.8 GiB), attention state and device zones of 1.5-4 GiB, the B70
  holds **23-26 GiB** of DeepSeek experts and, with Qwen's 1.46 GiB more dense
  weight, **21.5-24.5 GiB** of Qwen's. A two-card run is `9g22`'s T25, after
  sgc5; there the cards hold ~31-38 GiB of DeepSeek experts.
- **Today's code.** The ring sets the floor of each card's VRAM RUNTIME zone
  (§2.3), so the device zones are not 1.5-4 GiB but at least 7.87 GiB
  (DeepSeek) or 4.17 GiB (Qwen). The B70 then holds up to that many bytes
  fewer experts, and they become host experts. The today's-code rows below
  charge the whole ring as displaced.
- **Two VRAM terms are inputs, not estimates here.** They are the C-class terms
  charged at the context transaction: `llama.cpp-1oxa`'s D512 no-tile ONEDNN
  term (*lead ruling §V11.3*; DeepSeek's head_dim is 512) and
  `onednn_graph_scratch` (*lead ruling §M19*, owned by `llama.cpp-23mk`). A
  context that does not fit is refused (§V11.3), and planned expert copies are
  never yielded to it (§M18.3, §M18.3a). So both terms displace host experts
  1:1, and the plan must carry them before it places experts. 7v5c prints
  them; this section carries no number for them.

**The terms, per T2's definition** (GiB; `max_tensor` is 1.81 GiB for DeepSeek
and 0.83 GiB for Qwen, both `blk.0.ffn_down_exps`):

| term | source | Qwen today | Qwen after (s) | DeepSeek today | DeepSeek after (s) |
|---|---|---:|---:|---:|---:|
| host experts, exact | total minus B70 experts; today also the ring's VRAM displacement | 99.2-102.2 | 95.0-98.0 | 165.2-168.2 | 157.3-160.3 |
| other host weights (`token_embd`, CPU `get_rows`) | header | 0.63 | 0.63 | 0.20 | 0.20 |
| `rnh3` named terms | WEIGHT reserve 64 MiB (`unified-cache.cpp:27103`) + `HOST_COMPUTE`; the TLSF bound is not yet defined | 0.1-0.5 | 0.1-0.5 | 0.1-0.5 | 0.1-0.5 |
| host KV | Qwen: attention on the card at the gate contexts; DeepSeek: the dsv41 module is non-demotable (`9g22` §2.4) | 0 | 0 | 0 | 0 |
| STAGING zone | `2 x max_tensor + 2 MiB + 16 MiB` + routing ids at `-c 4096` (§2.3; expert bias and MMID pool not yet known) | 1.69 | 1.69 | 3.65 | 3.65 |
| SCRATCH zone | `2 x max_tensor + 32 MiB` + oneDNN reorder ≈ 0.08 + `moe_q8` ≈ 0.03; today also `dma_staging_pool` (`2 x max_tensor`) and **the ring**; `moe_vram_runtime` and `pp_pipeline` not yet known | 7.63 | 1.80 | 15.26 | 3.77 |
| ... of which the ring | `llama-model.cpp:430-507` | 4.17 | 0 | 7.87 | 0 |
| RUNTIME pool, as committed | oneDNN scratchpad + `pp_pipeline` ≈ 0.09; today also the DMA pool and the ring again; 2 GiB chunks | 5.92 → 6 | 0.09 → 2 | 11.58 → 12 | 0.09 → 2 |
| KV zone minimum | `ggml-sycl.cpp:5626` | 0.06 | 0.06 | 0.06 | 0.06 |
| **non-weight host, one zone set, as committed** | | **15.4** | **5.6** | **31.0** | **9.5** |
| main-pool chunk rounding | the main pool commits 2 GiB chunks | up to 2 | up to 2 | up to 2 | up to 2 |
| **step 2, one zone set** (main-pool rounding excluded) | | **115.3-118.7** | **101.3-104.7** | **196.4-199.8** | **167.1-170.5** |
| `host_reserve` + context envelope | 2 GiB + <= 0.5 GiB (output slots: Qwen 10 MiB, DeepSeek 48 MiB; DeepSeek landing slabs 3.9 MiB and ring 0.5 MiB; the `SYCL_Host` compute buffer) | 2.5 | 2.5 | 2.5 | 2.5 |
| DSpark draft, host | `9g22` §10.3 | - | - | +7.9 | +7.9 |

The oneDNN reorder and `moe_q8` figures are `9g22`'s estimates for DeepSeek,
reused for Qwen for want of better. Every row below excludes the main pool's
chunk rounding, so each requirement can be up to 2 GiB higher.

**Qwen (B70 alone, one zone set).**

| code | needed before the table | + table (HOST) | `MemAvailable` needed | 171.4 | 181.8 | 183.7 | 190.4 |
|---|---:|---:|---:|---|---|---|---|
| today | 117.8-121.2 | 168.5-171.9 | **176.4-179.8** | **STORAGE** (42.2-45.6 left for the table) | HOST (1.9-5.3 spare) | HOST | HOST |
| after (s) | 103.8-107.2 | 154.5-157.9 | **162.4-165.8** | HOST (5.5-8.9 spare) | HOST | HOST | HOST |

- On today's code the table flips to STORAGE at the lowest reading, and 181.8
  is marginal once main-pool rounding is counted. The flip does not depend on
  the ring's device displacement: without it, today's code still needs
  172.3-175.7.
- After (s) the table is **HOST at every reading**. The spare becomes other
  tenants' headroom; Qwen has no row cache while its table is HOST.
- A second zone set (two cards before sgc5) adds another 5.6-15.4 GiB and
  moves the flip up by as much.

**DeepSeek Q2_K, the planner's fit** (`MemAvailable - 8`; the verdicts are
after (s)):

| case | today's code: `MemAvailable` needed | after (s): needed | 171.4 | 181.8 | 183.7 | 190.4 |
|---|---:|---:|---|---|---|---|
| A: phase 1 (B70 alone), no draft, floor caches | 206.9-210.3 | **177.6-181.0** | refused | fits (0.8-4.2 left) | fits (2.7-6.1 left) | fits (9.4-12.8 left) |
| B: A + gate caches (2 GiB per table, index included) | 210.9-214.3 | 181.6-185.0 | refused | low corner only (0.2 left) | low corner only (2.1 left) | fits (5.4-8.8 left) |
| C: A + DSpark draft | 214.8-218.2 | 185.5-188.9 | refused | refused | refused | fits (1.5-4.9 left) |
| D: two cards after sgc5 and `9g22` T25, one zone set | - | 165.6-173.0 | low corner only (5.8 left) | fits (8.8-16.2 left) | fits (10.7-18.1 left) | fits (17.4-24.8 left) |
| E: D + the non-weight zones a second time | - | 175.1-182.5 | refused | low corner only (6.7 left) | fits (1.2-8.6 left) | fits (7.9-15.3 left) |
| F: two cards, code as it is today (a whole second zone set, WEIGHT included) | 395.1-409.9 | - | refused (T2) or OOM without T2 | | | |

On today's code every phase-1 case is refused at every reading. Lever (s)
recovers ≈ 29.3 GiB of what case A needs: the ring three times (twice on the
host, once as displaced experts) and the DMA pool twice.

<a id="th32-p3"></a>
**DeepSeek Q2_K, 9g22's P3: the binding check** (*lead ruling §T4*, I-2).
The gate protocol launches only at a predicted post-load `MemAvailable` of at
least 24 GiB: `MemAvailable_pre - expected_host - 5 >= 24`, where
`expected_host` is step 2 as committed plus the gate caches (2 x 2 GiB), with
no draft, before any cold set, and without the main pool's chunk rounding
(0-2 GiB), as 9g22 §4.5.2 defines it. Its margin (29 GiB over
`expected_host`) is wider than the planner's (8 GiB, plus the 2.5 GiB reserve
and envelope), so **a plan that passes P3 also passes the planner**.

| code | `expected_host` | `MemAvailable_pre` needed | predicted post-load `MemAvailable` at 171.4 / 181.8 / 183.7 / 190.4 |
|---|---:|---:|---|
| today | 200.4-203.8 | 229.4-232.8 | -37.4 to -34.0 / -27.0 to -23.6 / -25.1 to -21.7 / -18.4 to -15.0 |
| after (s) | **171.1-174.5** | **200.1-203.5** | -8.1 to -4.7 / 2.3 to 5.7 / 4.2 to 7.6 / 10.9 to 14.3 |

P3 fails at every reading in both columns, so without a cold set the gates
cannot launch at all. After (s), `expected_host` agrees with `9g22` rev 5
§4.5.2's 171.0-174.4 GiB to rounding. Rev 5's today's-code row (188.0-195.4)
is 8.4-12.4 GiB lower than the one above, which charges the ring's full VRAM
displacement and the 4 GiB of gate caches. That row decides nothing (today's
code fails P3 either way) and is not reconciled by hand here; 7v5c's printout
is the one list of terms both documents cite (§4.2).

<a id="th32-d4-fit"></a>
**DeepSeek Q2_K with the owner's §D4 applied** (D8, *lead rulings §T3,
§D5*). The fit is case A after (s), minus the bytes of the cold set `C` that
moves to STORAGE, plus its landing slab (`2C/40`). `C` is not a th32 choice:
7v5c computes `C = max(C_selected, C_P3)` at load from that load's reading
(§4.3 step 5), with the gate caches fixed before it (§T4, I-4). The size it
reaches at 171.4 GiB and at 189.6 GiB is `9g22`'s figure, in
[its §4.5.2](deepseek41-runtime.md#452-the-host-fit-r2-c-a-r3-c-1-c-2-owner-d4)
(the cold-set table); this document does not restate it. The calculator's
cross-check, run from the P3 row above, lands in the same range. Because P3
binds, a load whose `C` clears P3 also fits the planner. With the DSpark draft
(production, after gates 1-4) `C` grows by about the draft's bytes. The load
refuses (`cold-set-cannot-clear-p3`) only if moving every routed expert to
STORAGE would still fail P3. At the lowest reading that is far away: P3's
deficit there (the P3 table above) is under a fifth of the 183.25 GiB of
experts.

Where the row caches land on the hit-rate curves (§6.2.2): in a plan with a
cold set they are **fixed** at the gate size, 2 GiB per table with the index
included (§4.3 step 4, §T4 I-4). That leaves each table ~1.7 GiB of 84 B slots
after its index, 44 M slots together, which is ~4.5 GiB in the simulation's
110 B units, between its 2 GiB and 8 GiB columns. That is the DeepSeek
operating point at every reading; the reading changes `C`, not the caches.

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
   calls resident silently becomes a storage read. Worse, a "HOST residency" backed
   by mmap is not HOST at all.
4. **Efficiency.** The page cache holds 4 KiB pages, 24 rows each, when one of
   them is hot. For the same bytes, a 4 KiB-page LRU hits much less than a row
   LRU (§6.2: 19.5% against 34.5% at 32 MiB on wikitext-2 train).

Buffered `pread` is rejected for reasons 2 and 3, since it fills the page
cache too.

### 5.2 IO engine

Two engines implement one interface (D2):

- **Portable default: a pread worker pool.** `QD_max` threads (64 on this
  host) each do blocking `pread(O_DIRECT)` into a block-ring slot. It needs no
  dependency and ports to Windows as `ReadFile` + `FILE_FLAG_NO_BUFFERING`. It
  measured 8.3 k rows/s on the Qwen table and 7.05 k on the DeepSeek Q2_K
  table, at the same-session fio QD64 ceiling (ratio 0.99, §2.4). The workers
  sleep in the kernel, so they cost no CPU.
- **Linux fast path (later): `io_uring` through raw syscalls.** It needs no
  liburing and adds no threads. It matters when the device is faster than the
  thread pool can drive: the ext4 NVMe reference reaches 110 k IOPS, where
  per-IO thread wake-ups start to show.

**One pool per process, with priority classes, fairness and in-flight dedup**
(I7). A process-wide FIFO has head-of-line blocking, and it was measured
(`batch_probe3 hol`, A.2): a 24 + 24-row decode token queued behind another
context's 12000-row prefill completed in **1359 ms** mean under FIFO, and in
**21.7 ms** when it was placed ahead of the waiting prefill reads (11.2 ms with
nothing else queued; the remainder is the prefill's reads already on the 64
workers). The prefill lost ~100 ms. So the engine's queue is:

- **Class 0, the current ubatch**, served before class 1. Within a context,
  reads are ordered by deadline, which is the table's `consumer_layer`:
  DeepSeek's `blk.1` rows before its `blk.14` rows.
- **Across contexts, round-robin.** Class 0 keeps one queue per context and the
  workers take from them in turn, so no context's cold PP ubatch (up to ~25 k
  reads) can sit ahead of another context's 48 decode reads.
- **Class 1, lookahead** (§5.4), served only when class 0 is empty.
- **In-flight dedup.** An in-flight table keyed by `(table, block offset)`
  records each pending read and its **waiters** (ticket, epoch, landing
  positions). A request for a row whose block is already pending joins it
  instead of reading it again: a repeated n-gram within one ubatch, the same
  row in ubatch `k` and in `k+1`'s lookahead, or two contexts reading the same
  hot row cold at the same moment. Two rows that share a 4 KiB block are one
  read.
- **Cold-expert slice reads share the pool** (owner §D4, *lead ruling §T4*
  I-4). The cold experts and the engram tables live on the same two USB
  bcachefs members, so a second engine would only hide the contention, not
  remove it. `llama.cpp-yl3v` therefore submits its expert-slice reads through
  this pool, as class 0 of the context that routed them, with the deadline set
  to the expert's layer, in chunks of at most 512 KiB (§9.2.2). One ordering
  then serves both kinds of read. An engram `blk.1` row (deadline 1) goes ahead
  of any expert read for a later layer, and nothing preempts a chunk a worker
  has already started, so an engram read waits at most for one worker to
  finish its current chunk. The interface is `lazy_io_submit(fd, offset,
  length, destination handle, ticket, class, deadline)`, the same call for a
  4 KiB row block and a 512 KiB slice chunk (§9.2.1). The cost of the sharing
  is not known until it is measured; §8.4 states targets for the shared
  device.

The rules both engines follow:

- Every miss becomes one 4 KiB-aligned read of the block(s) that contain the
  row: 4 KiB, or 8 KiB for the rows that straddle a block (4.1% of 170 B rows,
  2.7% of 110 B rows, 1.9-2.2% of 84 B rows).
- `O_DIRECT` alignment comes from `statx(STATX_DIOALIGN)`. If the filesystem
  refuses `O_DIRECT` (for example tmpfs or some FUSE mounts), the fallback is
  buffered `pread` followed by `posix_fadvise(DONTNEED)` on the range. It logs a
  once-per-process WARN that names the P4 cost.

### 5.3 Tickets, epochs, and every wait on the path

A **ticket** per `(table, context, ubatch)`, so each gather waits only for its
own table's rows:

```
ticket { lazy_table_ref table; context id; slot index; uint32 epoch;
         std::atomic<int> outstanding; std::atomic<bool> cancelled; futex word }
landing slot { uint32 epoch; ticket * owner; }   // depth 2 per table per context
```

- **`prefetch(rows)`** assigns the context's next landing slot (below), then
  classifies each row reference (§6.3). A **hit** is copied from the row cache
  into its landing position *immediately*, under the cache shard's lock: one
  row, 84-170 B. That removes any window in which another context could evict
  it. A **miss** registers a waiter in the in-flight table, joining a pending
  read or queuing a new one. `outstanding` is the number of waiters this ticket
  registered. `prefetch` then submits the gather's host task (§7.2, step 1),
  with no device dependency. A HOST table skips the classification and the
  ticket: its `prefetch` only submits the host task.
- **The reaper**, the worker whose read completed (or the io_uring CQ reader),
  removes the entry from the in-flight table, then handles each waiter: if the
  waiter's epoch still equals its slot's epoch and its ticket is not
  cancelled, it copies the row into the slot; it inserts the row into the row
  cache either way, because file bytes are always valid cache content; and it
  decrements that ticket's `outstanding`. The last decrement wakes the waiter.
- **A ticket holds a reference to its `lazy_table`** (I11), released when the
  ticket is released. A completion therefore never outlives the table it
  writes into, the table's destructor can never run under an in-flight read,
  and it needs no teardown wait (§3.2).
- **Slot reuse has epoch semantics** (I11). Assigning a slot to a new ubatch
  bumps its epoch. If the slot's previous ticket still has outstanding waiters
  (an aborted, errored or KV-retried ubatch whose gather never ran), that
  ticket is **cancelled**, not waited for: its reads still complete and feed
  the cache, but the reaper's epoch check skips their slot copies. The new
  ticket's rows that are still pending under the cancelled ticket simply join
  those reads. Nothing waits, and no stale row can land in a reassigned slot.
  T5's RED fixture: ticket A into slot 0, completions held, "abort"; ticket B
  into slot 0 with different rows; release A's completions, then B's. Slot 0
  must hold B's rows exactly, and A's rows must appear only in the cache.
- **Lock order** (M11). The pool's per-context queues, the in-flight table, the
  cache shards and the ticket futex are all **leaf** locks: none is ever held
  while another is taken. `prefetch` takes a shard lock, releases it, takes the
  in-flight lock, releases it, then takes the queue lock. The reaper takes the
  in-flight lock, releases it, copies into slots with no lock (the epoch is
  atomic), takes a shard lock to insert, releases it, and decrements an atomic.
  No lock is held across an IO wait or a callback. T5 adds them to canonical
  contract §12.5 as L5 leaves.

**Every wait on the path, named:**

1. **HOST residency: none for rows.** The host task dequantizes from the
   resident body and has no ticket to wait for. It shares item 7 with
   STORAGE.
2. **STORAGE: the gather's `host_task` waits for `outstanding == 0`** on a
   SYCL runtime thread. A CPU thread waits for *storage* IO, never for the GPU,
   and the host thread that submits the graph never waits at all. The wait is
   on the critical path only when the IO takes longer than the window (§5.4),
   and it is counted as `T_ticket` (§8.1, §10).
3. **Device side, an event edge, not a host wait:** the output copy is
   enqueued on the main in-order queue just before the consumer, with a
   `depends_on` on the host task (§7.2). If the IO is late, the *device* waits
   there; the host thread has already moved on.
4. **Removed from this path by *lead ruling §T1* D6(c):**
   - `llama.cpp-x1xd`, the scheduler's host sync before a zero-input split
     after a backend change (`ggml-backend.cpp:2326-2334`). There is no CPU
     split in the middle of the graph any more. (The sync still fires at the
     leading-CPU-to-SYCL boundary that every model has, where it synchronizes
     the CPU backend, which has nothing queued.)
   - `llama.cpp-jhxq`, the SYCL `set_tensor` GLOBAL drain that a CPU-to-SYCL
     split-input copy takes because `cpy_tensor_async` is NULL
     (`ggml-sycl.cpp:107627`, `:36076-36084`). The gathered activation is
     copied by the backend's own event-chained `memcpy`, not through
     `set_tensor`.
   - The replay fragmentation of C2: the gather adds no `graph_compute` call,
     so the context's exec-graph slot sees the same graph every token (§7.4).
5. **Nothing added at the graph's start, and nothing read in place.** The rows
   input lives in the CPU backend's compute buffer, which on this fork is the
   SYCL host buffer type (`76c7f6548 src/llama-context.cpp:888-895`), and the
   SYCL device reports that type as supported (`ggml-sycl.cpp:108809-108811`).
   The scheduler therefore creates no split input and no copy for it
   (`ggml-backend.cpp:2116`, the `!ggml_backend_sched_buffer_supported`
   condition; the per-copy input duplicates of `:2092-2112` need
   `n_copies > 1`, and pipeline parallelism is off whenever a SYCL placement
   plan is active, `llama-context.cpp:970-971`). No device reads it either:
   the gather node dispatches nothing, and `lazy_prefetch()` read the ids on
   the host at `set_input` (§7.2). The gather's **output** reaches the device
   only by the explicit copy of item 3, in both residencies (*lead ruling
   §T2*, DQ). Wait sites (2)-(4) of the `llama.cpp-jhxq` census are not
   reached by anything this design adds; `jhxq` stays open for the copies
   other paths make.
6. **A risk to verify, not a wait by design:** SYCL `host_task`s run on the
   runtime's thread pool, whose size is `SYCL_QUEUE_THREAD_POOL_SIZE` (the
   string is present in `libsycl.so.9`; the documented default is 1, which is
   unverified here). A gather blocked on a 3 s cold PP ubatch could then delay
   another context's decode gather, which is I7's head-of-line problem in a new
   place. §7.2 pre-registers four probes (HT-1 to HT-4) that T7 runs first,
   through the lead; HT-3 is this case. If any fails, the completion signal
   moves to the reaper and a host-signalled Level Zero event (D6-M, §13).
7. **The output slot's reuse check** (§7.2), in both residencies: the host
   task's thread, never the submitting thread, waits for the device, and only
   when the device runs two ubatches behind the host.
8. **The landing slot's reuse check** (§7.2): `prefetch(k+2)` waits on the
   submitting thread for ubatch `k`'s host task to consume its slot, which
   happens only when ubatch `k`'s storage IO is still outstanding two ubatches
   later. It is a wait on storage IO, not on the GPU, and `slot_wait_ns`
   counts it.

<a id="th32-windows"></a>
### 5.4 The prefetch window

- **When the IO starts.** Every table's reads are issued by `lazy_prefetch()`
  right after `set_inputs()` (§7.5), before the graph is submitted.
- **When it must be done.** With the SYCL-owned gather, the deadline is the
  moment the device reaches the copy before the consumer (§7.2). The host task
  that dequantizes is submitted at prefetch too, with no device dependency, so
  the dequant runs as soon as the rows have landed and overlaps the preceding
  blocks as well. The node's position in the graph affects neither.

  | table | residency here | issued | needed before | window |
  |---|---|---|---|---|
  | Qwen `per_layer_token_embd` | HOST | `lazy_prefetch()` submits the host task; no IO | block 1 | graph launch + block 0, against a ~4 us gather |
  | DeepSeek `blk.1.engram_embd` | STORAGE | `lazy_prefetch()`, class 0, deadline 1 | block 1 (B70) | graph launch + block 0 |
  | DeepSeek `blk.14.engram_embd` | STORAGE | `lazy_prefetch()`, class 0, deadline 14, behind `blk.1` | block 14 (B70) | graph launch + blocks 0-13 |

  Both DeepSeek tables are issued at the same moment, and `blk.14`'s reads are
  dispatched to workers after `blk.1`'s. Measured together on the two Q2_K
  files through one 64-worker pool, `blk.1`'s 24 rows landed at **7.13 ms
  mean, 14.6 ms p99** and all 48 at 11.2 / 21.0 ms (A.2).
- **Decode.** 16 (Qwen) or 48 (DeepSeek) rows per token, issued at
  `set_input`. The critical window is one block of one token, sub-millisecond
  to a few ms. A token cannot be prefetched earlier, because it is not known
  until it is sampled (MTP speculation, §12).
- **PP.** Every prompt token is known when `llama_decode` is entered. Phase 1
  prefetches each ubatch at its own `set_input`. Phase 2 adds **one-ubatch
  lookahead**: after `set_input(k)`, compute ubatch `k+1`'s row indices and
  prefetch them as **class 1** reads into the second landing slot. That is why
  the depth is 2 in §3.3. With lookahead, the IO of ubatch `k+1` overlaps the
  *entire* compute of ubatch `k`, and the exposed stall per ubatch becomes
  `max(0, IO - compute)`. In-flight dedup means a row missed by both `k` and
  `k+1` is read once. The indices of ubatch `k+1` need predecessor tokens,
  which are the last `n_gram-1` tokens of ubatch `k` in the same sequence
  (2 for Qwen, 3 for DeepSeek). The model's input class gets a
  `peek_rows(next_ubatch, tail)` hook. The generic path skips lookahead when a
  model does not implement it. For DeepSeek on this host the tables are always
  on STORAGE and a cold ubatch costs ~2.5-3.5 s of IO, so lookahead is what
  makes long prompts tolerable; D9 proposes it for v1.

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
slots, excluding the 7.8% index overhead (§6.3):

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
and the distinct-row sets are 0.9-11.5 GiB. On this host Q2_K's phase 1 has a
cold-expert set, so its caches are **fixed** at the gate size, 2 GiB per table
with the index included (§4.3 step 4, *lead ruling §T4* I-4). That is ~4.5 GiB
in simulation units, between the 2 GiB and 8 GiB columns, at every reading:
the reading moves the cold set, not the caches (§4.5). The hit rates of those
two columns bracket the operating point. A larger cache would cost the same
number of GiB of warm experts, one for one.

**Which corpora a target may quote** (M4). The code and docs corpora are this
fork's own sources, and the decode and perplexity vehicles in §8 would run on
text of the same kind: n-grams that recur across files of one codebase are
exactly what the cache catches, so those two corpora flatter the cache and are
self-similar with any in-repo test text. §8's targets therefore quote **prose**
(wiki train) and **mixed** only, and a vehicle's text must come from outside
the repo (wikitext-2 test for perplexity; a fixed prose prompt for decode).
Code and docs stay in the tables as the upper bound.

<a id="th32-row-cache"></a>
### 6.3 Structure

- **Keying.** The key is the row index (int64 for generality), inside the
  table's own cache. There is no special n-gram grouping: the 8 heads of one
  n-gram are 8 separate keys that are touched together, which CLOCK already
  handles, and the generic path stays table-agnostic (§9).
- **Index, 13.3 B per slot.** Open addressing over an array of 4 B slot ids at
  load factor <= 0.75 (5.33 B per slot), plus one 8 B key word per slot that
  holds the row index with the CLOCK reference bit folded into its top bit.
  That is **7.8%** of the slab on 170 B rows, **12.1%** on 110 B rows and
  **15.9%** on 84 B rows; §3.3 and §4.5 charge those figures, and the
  simulation's `cache_slots` are taken after them.
- **Sharding.** `hash(row) mod n_shards` with one mutex per shard. It is a leaf
  lock, **L5** in canonical contract §12.5 (added to its inventory by T4), and
  is never held across an IO wait or a callback (§5.3).
- **Contents.** Raw row bytes in the table's own type (170 B Q8_0, 110 B Q3_K,
  84 B Q2_K), not F32. Dequant happens in the gather (§7). An F32 cache would
  be 3.8x (Q8_0) to 12x (Q2_K) larger for the same rows.
- **Insertion.** A block completion inserts the row it carried (§5.3). A row is
  evicted only by CLOCK or by a yield (§6.4).
- **No leases** (*lead ruling §T1*, I8). No pointer into the cache outlives a
  shard lock: a hit is copied to its landing position under the lock, and the
  reaper copies into the landing slab *before* it inserts. The landing slab,
  not the cache, holds every row a pending gather needs. That is what makes the
  cache yieldable at any moment without waiting for anybody.
- **Warm start.** None in v1. The measured cold numbers are the contract. D1
  covers background prefill for the HOST residency.

### 6.4 Size, and yielding it back

- **The floor.** One double-buffered ubatch working set at `n_ubatch = 512`,
  plus its index: 2.66 MiB of slab for Qwen's table, 1.97 MiB for each DeepSeek
  Q2_K table. The floor is charged in step 4 (§4.3), before any leftover, and is
  never yielded. If even the floor does not fit, the load is refused as
  `lazy-table-cache-below-floor`.
- **Fixed, when the plan has a cold set** (*lead ruling §T4*, I-4). Then each
  cache is charged at its target in step 4, before the cold set is sized, and
  it does not trade bytes with the cold set at run time in either direction
  (§4.3). For DeepSeek this design proposes the gate size, 2 GiB per table
  including the index, as the target. It is still built from yieldable
  256 MiB chunks, so a context tenant can take bytes back (below).
- **The leftover, yieldable, when it has none** (*lead ruling §T1*, I8). Step 5
  gives the rest of the host budget to the row caches, split as that step
  states, in
  **256 MiB chunks**. Each chunk is its own `HOST_CPU` allocation (role
  `LAZY_ROW_CACHE`) and is charged in the ledger as **yieldable**. When a later
  tenant needs host bytes (a new context's envelope, a KV growth, a later
  model's host weights), the ledger first takes them from yieldable chunks, and
  only refuses when yieldable bytes are exhausted too. Context tenants are
  therefore never starved by a cache that got there first.
- **How a chunk is yielded.** The same shape as the device-side
  `yield_optional_layouts_begin/finish` (`76c7f6548
  ggml/src/ggml-sycl/unified-cache.cpp:7798-7911`): under each shard lock in
  turn, remove the chunk's slots from the index and mark the chunk withdrawn so
  that no insertion picks it; then, with no lock held, drop the chunk's
  `mem_handle`, whose destructor is the release point (P2). Nothing waits,
  because nothing holds a lease (§6.3). The chunk's rows become misses; a
  reaper insertion that raced the withdrawal lands in another chunk or is
  dropped, which is always correct because the landing slab already has the
  row.
- **Growing back.** None in v1. A yielded chunk is re-offered only at the next
  plan publish, when the budget is sampled again (§4.2). A server whose
  contexts come and go therefore keeps its caches at the smallest size any
  tenant forced. That is proposed for v1 (D7); regrowing between publishes
  would need a trigger the ledger does not have.
- **Logging.** The planner logs `cache_slots` per table and the yieldable bytes
  (§10), and every yield logs the chunk count and the tenant that asked.

---

## 7. The executor (item 5)

<a id="th32-executors"></a>
### 7.1 One gather path, and the provider decides the position

Every consumer of a lazy tensor (Qwen's `build_inp_ple`, DeepSeek's engram block
input, and the rest of §9) calls one graph helper:

```
ggml_tensor * llm_graph_context::build_lazy_get_rows(ggml_tensor * lazy, ggml_tensor * rows, int consumer_layer);
```

- **Without SYCL hooks**, it returns `ggml_get_rows(lazy, rows)`: the upstream
  path, with the mmap and page faults, unchanged.
- **Both residencies: `GGML_OP_GET_ROWS(lazy, rows)`, claimed by SYCL**
  (*lead ruling §T2*, DQ: one code path for both tiers). The lazy tensor sits
  in the SYCL descriptor buffer of its consumer device (§3.2), a `WEIGHTS`
  buffer, so the scheduler assigns the node to that SYCL backend (rule
  `1.wgt`, `ggml-backend.cpp:1594-1612`, `:1684`); the CPU backend cannot claim
  it, because the buffer is not a host buffer. The SYCL backend recognizes a
  `GET_ROWS` whose `src[0]` is in a lazy buffer and runs the gather of §7.2.
  The residency changes only what the host task does before it dequantizes:
  a HOST table's host task reads the resident body at once, a STORAGE table's
  first waits for its ticket. There is no CPU split in the middle of the graph,
  and the result has the shape `ggml_get_rows` produces, so each model's
  reshape and everything downstream are untouched.
- **Why HOST no longer sits in the leading CPU split** (rev 2a's form, *lead
  ruling §T1* C2; put to the lead as DQ-C2, §13). Rev 2a made the HOST gather
  a CPU custom op in the leading split and let the first SYCL op read its
  output from the `SYCL_Host` compute buffer in place. §T2's DQ forbids that
  read, and an explicit copy from the leading split cannot be made wait-free.
  The CPU op runs on the submitting thread and writes the same compute-buffer
  bytes for every ubatch, because `gallocr` gives a graph of the same topology
  the same addresses. In a multi-ubatch prefill nothing orders ubatch `k+1`'s
  CPU op after the device's copy of ubatch `k`'s output: the ubatch loop does
  not synchronize between ubatches unless pipeline parallelism is on
  (`c69d5774d src/llama-context.cpp:2724-2729`), and the SYCL split takes no
  split-input copy for a supported host buffer (§5.3 item 5). Guarding that
  source would take a wait on the submitting thread for the device, which is
  the host wait this design exists to avoid. The host task of §7.2 puts the
  same guard on a runtime thread (§7.2, the output-slot rule). So HOST takes
  the STORAGE path, and the CPU custom op survives only as arm C's executor
  (§8.4).
- **Position.** The helper expands the node at the **head of the consumer
  device's first split**: immediately before the first node of the first layer
  that runs on the table's consumer device. On one device that is right after
  `inp_embd`. With `-sm layer` and a consumer layer on the second
  card, a node placed at the very start of the graph would instead form a
  one-node split on that card ahead of the first card's split, and the
  first card's split would then pay the zero-input sync (`llama.cpp-x1xd`)
  against it. So the layer loop gets a `lazy_expand_for_layer(il)` call at the
  top of each iteration, and the helper defers the expand to the first layer
  on the consumer device. No new split is created in either case.
- **The window does not depend on the position.** The IO and the host task
  that dequantizes it are both started at prefetch (§5.4, §7.2), and the
  device waits only at the copy before the consumer, so the node's position
  changes nothing that runs; it only has to keep the split structure intact.

### 7.2 The SYCL-owned gather, both residencies (*lead rulings §T1* D6(c), *§T2* DQ)

1. **At prefetch time, a `host_task` on a per-context, per-device
   out-of-order host queue** (same SYCL context as the main queue), with
   **no** `depends_on`. `prefetch` (§5.3) submits it right after it has
   queued the table's reads (STORAGE) or at once (HOST), from
   `lazy_prefetch()`, before the graph is submitted (§7.5). Its inputs are
   host data that is complete at `set_input`, so it has nothing on the device
   to depend on. A STORAGE host task waits for its ticket, then dequantizes
   the landing slab's rows; a HOST host task dequantizes the rows straight from
   the resident body. Both write the context's pinned `LAZY_OUTPUT` slot (§3.3)
   through the type's `to_float` trait. Nothing else runs on that queue. A
   `host_task` whose `depends_on` names an in-flight event from another queue
   blocks the *submitting* thread until that event completes
   (measured on both cards for `llama.cpp-c6ah`, 2026-09-09, 6/6 reps). Had
   `blk.14`'s host task depended on device work, submitting it would have held
   the host until blocks 0-13 finished, and blocks 14-39 would have been
   submitted late. This construction never gives it a device dependency.
2. **The gather node itself dispatches nothing**, in direct and in record
   mode: its work was submitted at prefetch, and its `dst` is written by the
   copy below. The main queue proceeds with block 0 at once.

Before the **first node (or fused group) that reads the gather's `dst`**, the
backend enqueues, on the **main in-order queue**:

3. `memcpy(dst_device, output_slot, bytes)` with `depends_on(host_task event)`:
   the explicit, event-ordered H2D copy of *lead ruling §T2* DQ, the same for
   both residencies.
   Because the main queue is in-order, the copy runs after every earlier
   main-queue command, including the previous graph's readers of the same
   compute-buffer region: that is the write-after-read guard, with no barrier
   of its own. The consumer follows the copy in queue order: that is the
   read-after-write edge. The host thread waits for neither.

Rules around it:

- **The output slot is reused only after its copy completed.** It has depth 2
  (§3.3), and slot `k` is written again by ubatch `k+2`'s host task on the
  host queue. Nothing orders that host task after ubatch `k`'s copy on the
  main queue, and the host may run two ubatches ahead of the device, so T7
  adds the edge. It must not do that with a
  `depends_on`, for the reason in step 1. Instead, a device
  **marker** kernel after the copy writes the slot's epoch into a host-USM word
  (pinned, from the same `LAZY_OUTPUT` allocation), and the host task checks
  that word before it writes. It is set by the time it is read unless the
  device runs two ubatches behind the host; then the host task's thread, never
  the submitting thread, waits for it. That is the CPU executor waiting on its
  own output slot, a named wait (§5.3, item 7).
- **The landing slot is reused only after its host task consumed it**
  (STORAGE only; a HOST table has no landing slab). Slot
  `k mod 2` of the landing slab is reassigned by `prefetch(k+2)`, which copies
  that ubatch's hits into it at once (§5.3). The host task of ubatch `k` sets
  the slot's `consumed_epoch` (a host atomic) when its dequant is done, and
  `prefetch(k+2)` checks it before writing. If ubatch `k`'s IO is still
  outstanding two ubatches later, `prefetch` waits for it on the submitting
  thread. That is a wait on **storage IO** through the CPU executor, never on
  the GPU, and it is counted (`slot_wait_ns`, §10) and named (§5.3, item 8).
  With a phase-2 lookahead slot in use the same rule applies to depth 2.
- **Fusion.** The fusion matchers skip a node whose source is a lazy
  gather's `dst` until its copy has been enqueued, and never fuse the gather
  node itself.
- **The gather's `dst` lives in the device compute buffer** from the head of
  the split to its last consumer: for `blk.14` at PP512 that is 12 MiB held
  across blocks 0-13. `ggml_gallocr` measures it at reserve like any other
  tensor, so the compute buffer grows by it and the plan sees the growth (P4).
- **`rows` is never copied to the device.** It is an input, so it lives in the
  CPU backend's compute buffer, which on this fork is the SYCL host buffer
  type (`76c7f6548 src/llama-context.cpp:888-895`). The SYCL device reports
  that buffer type as supported (`ggml-sycl.cpp:108809-108811`), so the
  scheduler adds no split input for it (§5.3, item 5). The host task does not
  read it either: `lazy_prefetch()` already read it at `set_input` (§7.5).

**The risk the ruling carries, and how T7 retires it before building on it.**
This fork has twice moved away from `host_task`:

- `ggml-sycl.cpp:32755-32757`: "We avoid host_task because Level Zero driver
  has issues with it that can corrupt event handles and cause crashes", next
  to a `chain_event.wait()`.
- `:92354-92364` (`WEDGE-48330`) drains an async host_task chain before zone
  resets, and `:92806-92808` turns host-task mode off in favour of direct
  dispatch.

And the tracker's own measurement (`llama.cpp-c6ah`, 2026-09-09, both cards):
a `host_task` whose `depends_on` names an in-flight kernel event from
**another** queue blocks the submitting thread until that kernel completes.
The construction above avoids that case, but the main queue's copy in step 3
depends on a host-task event, and whether DPC++ keeps an in-order Level Zero
queue on its fast path after such a dependency is unmeasured. T7's first
deliverable is therefore four lead-run probes, each a small standalone SYCL
program with no model, a few seconds long, on `level_zero:0` and then
`level_zero:1`, with `Shmem` sampled before and after:

| probe | construction | pass |
|---|---|---|
| HT-1 | host queue: host task sleeping 200 ms, no deps. Main queue: kernel K1 (~1 ms), copy `depends_on(host task)`, a kernel K1b also `depends_on(host task)`, then kernels K2..K50 | every `submit`, including the copy's, K1b's and K2..K50's, returns in < 1 ms (a device command that depends on a host task does not hold the submitting thread); K1 completes at ~1 ms, not ~200 ms; K2 starts after the host task |
| HT-2 | HT-1 with a recorded command graph replayed on the main queue after the copy | the replay's submit returns in < 1 ms; its kernels run after the copy |
| HT-3 | one out-of-order host queue and a second one: host task A sleeps 2 s, host task B 10 ms, submitted A then B, both on the same queue and then on different queues | B completes at ~10 ms in both. If not, the runtime's host-task pool is 1 thread and a cold PP gather would stall another table's or another context's gather |
| HT-4 | HT-1 repeated 10000 times | no crash, no Level Zero error, no event-handle anomaly (the `:32755` failure), `Shmem` flat |

- If all four pass, T7 builds on `host_task`, as ruled.
- If any fails, the mechanism falls back **within** the ruling's ownership
  model to D6-M(b) (§13): the reaper, a plain thread of the IO pool, does the
  dequant into the output slot and then signals a host-signalled Level Zero
  event (`zeEventHostSignal`, from a host-visible event pool, wrapped with
  `sycl::make_event`); the copy in step 3 `depends_on` that event. No thread
  blocks, and there is no `host_task`. The fork has no precedent for that
  interop, which is why it is the fallback and not the baseline.

### 7.3 Data movement, and P1 at both ends (M7)

- **Both residencies** (*lead ruling §T2*, DQ). The copy in §7.2 step 3 is an
  **activation**: 10 KiB per token for Qwen and 24 KiB per token per DeepSeek
  table, 5 MiB and 12 MiB per PP512 ubatch, ~1 ms of PCIe at most, against
  ~1.4 s of cold IO for a DeepSeek table's rows. Its source is the
  `LAZY_OUTPUT` slot, a `HOST_PINNED` allocation from `unified_allocate_owner`
  (§3.3), which only the copy engine reads. Its destination is a planned
  device slot: the gather's `dst` in the device compute buffer, planned by
  `gallocr` and allocated from the unified cache. Both ends are unified-cache
  memory (P1) held by `mem_handle`s that outlive the queued copy (P2). It
  never goes through `set_tensor` (`llama.cpp-jhxq`) and never needs the
  scheduler's zero-input sync (`llama.cpp-x1xd`), because the backend enqueues
  it itself on its in-order queue behind an event (§5.3).
- **No GPU read of `SYCL_Host` memory on this path.** Neither the gather's
  output nor its rows input is read by a device kernel: the output goes
  through the copy, and the rows are read only on the host (§5.3 item 5). This
  is the same P3 rule as `llama.cpp-zhcn`'s eager-path fix, which found and
  removed a zero-copy read of the same shape. The general census of CPU-split
  outputs that SYCL still reads zero-copy, `inp_embd` first, is a separate
  ticket the lead files (§T2 DQ); this design does not solve it and does not
  add a member to it.
  **T7's RED.** A lead-run graph fixture on `level_zero:0` with one HOST and
  one STORAGE lazy gather feeding a SYCL consumer. For every node it
  dispatches, the SYCL backend checks whether a source that is a lazy gather's
  `dst` resolves to a host buffer (`ggml_backend_buffer_is_host`), and counts
  it in an unconditional counter, `lazy_zero_copy_reads` (§10). It counts the
  step-3 copies in `lazy_output_copies`. The fixture requires
  `lazy_zero_copy_reads == 0` and `lazy_output_copies == tables x graphs`, and
  compares the consumer's input with a CPU `get_rows` of the same rows.
  - It fails first: T7 lands the recognition and the counters before the copy,
    so `lazy_output_copies` is 0 and the consumer reads an unwritten `dst`.
  - The zero-copy probe has a positive control, so its zero means something:
    the same check pointed at `inp_embd`'s first SYCL consumer, which reads the
    `SYCL_Host` compute buffer in place today (the census ticket's case), must
    count at least one read per graph.
- **Placement consistency (P3).** The table never crosses PCIe; the device
  receives the *result* of a CPU op. No weight byte is staged to the device
  (no streaming), and no device kernel reads a table. The table is consumed
  in its materialized CPU layout, AOS rows of its GGUF type, which for a gather
  *is* the CPU-optimal layout: each row is one contiguous run of 84-170 B.
- **Why F32 and not the raw rows.** Shipping the gathered blocks and
  dequantizing them on the device would be cheaper on PCIe (84-110 B instead
  of 1 KiB per DeepSeek row), but it would split one op across two executors:
  the GPU would execute part of `get_rows` on per-dispatch copies of weight
  bytes, the weight-streaming shape P3 forbids even for a gathered subset. The
  PCIe it would save is ~1 ms per ubatch against seconds of IO. An F16 output
  is a later option if the consumer takes F16 activations; the CPU still
  executes the op.

<a id="th32-record-mode"></a>
### 7.4 Command-graph record mode: a stated segment boundary

*Lead ruling §T1* (I1) requires the record-mode behaviour to be resolved
explicitly. The two options were a recorded `host_task` node inside the
command graph, or a segment boundary around the gather. **This design takes the
segment boundary.**

- **Why not a recorded node.** A recorded `host_task` replays with the lambda
  captured at record time, so it would have to find the current ticket
  indirectly; DPC++ runs a graph containing a host task as separate partitions
  with the runtime synchronizing between them, which is a runtime-side wait on
  the device at that point; and it would put the primitive `:32755` warns
  about inside every replay, where a failure is hardest to attribute. None of
  that is measured here, and none of it is needed.
- **The boundary.** The existing segmented replay already interleaves graph
  replays with directly dispatched nodes (`ggml-sycl.cpp:97186-97270`, the
  `MUL_MAT_ID` boundaries collected by `moe_graph_collect_dispatch_indices`).
  T7 widens that collector to a **boundary set** = the MoE dispatch indices
  plus, for each lazy gather of either residency, the first node that reads
  its `dst`. At that boundary the backend submits §7.2 step 3; the gather node
  itself is a no-op
  in both modes (its host task was submitted at prefetch), and the segments
  between boundaries replay as today.
- **What it costs.** One extra segment per consumer: two per graph on
  DeepSeek, one on Qwen. On DeepSeek each consumer boundary falls inside a
  segment that already ends at the previous layer's `MUL_MAT_ID`. The
  replay-count cap
  (`MAX_PROFITABLE_SEGMENT_GRAPHS = 8`, `:97249`) applies to the total, and T7
  must show the DeepSeek decode graph stays under it or raise the cap with a
  measurement.
- **What must not change.** The gather adds no `graph_compute` call and no
  `#`-named split-copy tensor, so neither the consecutive-miss futility gate
  (`:99226-99240`) nor the fragmented-split gate (`:106688-106700`, which
  disables record/replay for the whole context on any `#` tensor) can trip
  because of it.
- **The witness** (every arm of §8): `GGML_SYCL_GRAPH_DIAG=1`, and per run
  - `seg_replay` rises with the token count and `seg_record` stays at its
    warm-up value;
  - `full_replay` and `seg_replay` are not both zero on a decode run;
  - zero `[SYCL-GRAPH] replay futility` WARN lines;
  - T8's `lazy_boundary_dispatches` counter equals `tables x graphs`;
  - **submission runs ahead** (the `llama.cpp-c6ah` failure mode): the first
    kernel of block 15 is *submitted* before block 13 *completes*, from host
    submit timestamps against the profiled completion of block 13's last
    kernel, on at least 95% of decode graphs. A host task that picked up a
    device dependency, or a copy whose submit blocks, fails this.

  A run that fails any of these is scored as record mode disengaged, and its
  timing is not used.

<a id="th32-prefetch-hook"></a>
### 7.5 Where the prefetch hook sits

`llm_graph_result` records every `build_lazy_get_rows` call as
`{gather_ctx, rows_input_tensor, consumer_layer, residency}`. After
`llm_graph_result::set_inputs()` has run every input's `set_input`,
`llama_context` calls `res->lazy_prefetch()`. For each record, in ascending
`consumer_layer`, it reads the row indices from the rows input, whose buffer
is the CPU backend's host compute buffer (§7.2), and calls `prefetch`: with
class 0 for a STORAGE table (§5.2), and for a HOST table only to submit its
host task (§5.3). Lookahead (§5.4)
is the one per-model hook. Model code is untouched apart from
`build_lazy_get_rows` and the layer loop's `lazy_expand_for_layer`.

---

## 8. Pre-registered speed targets (item 6)

Every target below is derived from the numbers measured in §2.4, §5 and §6.2 on
this host, bcachefs as-is, under ambient load. The implementer measures against
them with the counters in §10; the lead runs every GPU part, one run at a time,
selector pinned, `Shmem`/`MemAvailable` sampled before and ~5 s after. Qwen's
targets are §8.2-8.3; DeepSeek's, the harder case, are §8.4.

### 8.1 Arms, positive controls and instruments

**The arm lever is the production budget override** (*lead ruling §T1*, I5):
`GGML_SYCL_HOST_RESERVE_MB`, which sets the ledger's budget (§4.2). It changes
the budget, never the residency, so the planner still decides; and it is read by
every binary, so every arm runs the **same** build. There is no test-only
override and no `GGML_SYCL_PRIVATE_TESTING` arm (that define is per test target
and never reaches `llama-bench` or `llama-perplexity`).

- **H**: HOST residency. Qwen's natural choice here once lever (s) lands, no
  override; on today's code Qwen is STORAGE at the lowest reading (§4.5), so an
  H run records its `MemAvailable` at load. DeepSeek has no H arm on this host
  (§4.5).
- **S**: STORAGE residency, by setting the budget below the table:
  `RESERVE_MB = (step-3 requirement + floors + cache target) / 1 MiB`, the
  requirement taken from that run's own plan line in a load-only run first.
  For Qwen the step-3 requirement is ~118-121 GiB on today's code and ~104-107
  GiB after (s) (§4.5), plus a 4 GiB cache target. For DeepSeek S is the
  natural choice and its caches are fixed at the gate size, 2 GiB per table
  **with the index included**, which is also 9g22's P1 cap (§4.3 step 4). An
  S arm that wants a different cache sets `RESERVE_MB` = the case-A
  requirement + 2 x the per-table cache size, index included. No separate
  cache-size variable exists or is needed.
  - **S-cold**: a fresh load, empty cache.
  - **S-warm**: steady state after at least 25% of a same-domain stream.
  - **S-hot**: the same text replayed with a cache at least as large as its
    distinct rows.
- **The DeepSeek executor arms** A and C (§8.4), selected at run time by a
  diagnostic variable `GGML_SYCL_LAZY_GATHER_ARM=A|C` (default A), documented
  in `docs/backend/sycl-env-vars.md` and removed once T9d has decided. It
  changes the graph, not the build.

**Positive controls** (*lead ruling §T1*, I6). An arm counts only if its
engagement is shown, because an absence of work looks exactly like success:

| arm | required in that run's own log | void if |
|---|---|---|
| H | `[PLACEMENT] lazy <name> residency=HOST` **and** `rows_requested > 0` **and** `blocks_read == 0` **and** `lazy_output_copies > 0` | the line is missing, or `rows_requested == 0` (the design silently disengaged, e.g. `AUTO` fell to `OFF`) |
| S | `residency=STORAGE`, `rows_miss > 0`, `blocks_read > 0` | any is zero |
| any SYCL-gather arm | the §7.4 record-mode witness, and `lazy_zero_copy_reads == 0` (§7.3) | any witness fails |
| negative control, once | a `--lazy-mode off` run shows **no** `[PLACEMENT] lazy` line and no lazy counters | the check fires anyway (then it proves nothing) |

**Correctness gate, every arm.** Greedy decode (`--temp 0 --seed 42`, 64
tokens) must be **token-identical** across H, S and `--lazy-mode off` for Qwen,
and across A and C for DeepSeek. All paths dequantize the same rows through the
same `to_float` trait, so the F32 activations are bitwise equal and any token
difference is a defect, not noise. DeepSeek additionally passes `9g22`'s CPU
oracle gate.

**Instruments.**

- `T_gather`: the host task's run time, per ubatch. For STORAGE it includes
  the ticket wait; `T_ticket` separates it out.
- `T_ticket(t)`: the part of table `t`'s gather spent waiting on its ticket,
  measured from the host task's start (≈ graph launch). It bounds the IO
  latency the device can see.
- `T_stall(t) = max(0, T_ticket(t) - T_window(t))`, where `T_window(1)` is
  block 0's time and `T_window(14)` blocks 0-13's, from the op timing the lead
  already collects. Where the main queue has profiling enabled, T8 also reports
  the device gap before the output copy directly, and that measured gap
  replaces the estimate.
- The IO count and bytes, the hit and miss counts (§10), and end-to-end
  `llama-bench`, `llama-perplexity` on wikitext-2 test, and the decode vehicle
  (§8.5).
- **The engine rate is a ratio, never an absolute** (I6, M12).
  `rows_miss_reads / T_io` over a cold PP512 ubatch, divided by the IOPS of a
  **same-session fio QD64 control on the same extent**, run immediately before
  and after the arm (Appendix B, `io_probe2.sh` with `FOFF` at the table's
  offset). The ratio must be **>= 0.9**; the pool measured 0.99 against the
  bracketing controls (§2.4). If the two controls differ by more than 20%, the
  ceiling moved during the arm and the run is void, not scored. Each arm
  records the per-device `io_done` and write deltas from `/proc/diskstats` for
  its duration, so that fresh (NVMe-foreground) and settled (backing-device)
  placement are told apart: during the `blk.1` probes the NVMe wrote 7.5 GB in
  15 s at 4657 IOPS, and 0.4-0.5 GB at 7070-7242 two minutes later.
- **Concurrent expert IO** (*lead ruling §T4*, I-4). On DeepSeek, every arm
  also records the cold-expert reads that shared the device in the same
  window: `llama.cpp-yl3v`'s read and byte counters, per ubatch and per decode
  token, next to the engram counters. Without that record, an engram latency
  measured on the shared device cannot be told apart from contention.
- **Scaling rule for latency targets.** The latency targets below were measured
  with the control at 4657-7242 IOPS. When an arm's bracketing control reads
  below 4657, each latency bound is multiplied by `4657 / control`.

### 8.2 Qwen decode

| arm | target | derived from |
|---|---|---|
| H | `T_gather` <= **20 us/token** (the host task's run time); tg128 within **1%** of `--lazy-mode off` (scored on that arm's second run, its table pages resident) | measured 4.2 us for 16 rows including dequant |
| S-cold | `T_ticket` mean <= **4.3 ms/token**, p99 <= **15 ms** | batch of 16: mean 3.75 ms, p99 13.0 ms (A.2), + 15% |
| S-warm | mean `T_ticket` <= **(1 - P0) x 4.3 ms**, where P0 is the steady-state zero-miss share at the *planned* `cache_slots` (§8.5): prose 32-43%, mixed 63-68%. This is a ceiling, not a discriminator (M13); the runtime-versus-simulation match of §8.5 is the discriminator | §6.2.1 |
| S-hot | `T_ticket == 0` on **>= 99%** of tokens | every row hits |

Expressed as throughput: `tg_S >= 1 / (1/tg_H + T_stall)`. If H decodes at
20 tok/s (50 ms per token), S-cold must decode at **>= 18.4 tok/s**.

### 8.3 Qwen PP512

| arm | target | derived from |
|---|---|---|
| H | `T_gather` <= **3 ms/ubatch**; pp512 within **1%** of `--lazy-mode off` | measured ~2 ms for 8192 rows |
| S-cold | engine ratio >= **0.9** (§8.1); `T_ticket` <= **1.15 x U / (0.9 x control IOPS)**: at the r1 control (9124), real text (U 7028-7104) <= **1.0 s**, `llama-bench` random (U 8116-8192) <= **1.15 s** | §5.1, §6.2.1 |
| S-warm | the same formula with `U` from the simulation at the planned `cache_slots`: prose (512 MiB) 4335 -> **<= 0.6 s**; mixed (512 MiB) 2275 -> **<= 0.32 s** | §6.2.1 |
| S-hot | `T_ticket == 0` | |
| phase 2 lookahead, multi-ubatch prompt | stall of ubatch `k >= 2` <= `max(0, IO(k) - compute(k-1)) + 10%` | §5.4 |

Expressed as throughput: `pp_S >= 512 / (512/pp_H + T_stall)`.

<a id="th32-ds-targets"></a>
### 8.4 DeepSeek Q2_K (the harder case)

**Where it runs.** Inside `llama.cpp-9g22`'s real-model gate protocol, P0-P7 of
its §5.2 (rev 5, `8997c6405`), after gates 1-4 pass: phase 1 on
`level_zero:0` (the B70 alone), caches fixed at 2 GiB per table with the index
included (§4.3 step 4), no DSpark draft, **in the owner's §D4 configuration**:
lever (s) landed, and the cold set `C` that 7v5c computed for that launch's
`MemAvailable` reading resident on STORAGE (§4.5, "with the owner's §D4
applied"). P3 is claimed only as §4.5 evaluates it: on today's code, or after
(s) with no cold set, P3 fails at every reading taken today, so §8.4 is
unrunnable until (s) and `llama.cpp-yl3v` have landed. Two gathers per ubatch;
`T(a..b)` is the measured time of blocks `a..b` for the same ubatch; `U` is
the unique miss rows of the ubatch, both tables together.

**The device is shared** (*lead ruling §T4*, I-4). The cold experts and the
engram tables are read from the same USB bcachefs members through the one
pool (§5.2). The engram bounds below were measured with no competing IO. They
are scored only on the tokens and ubatches whose concurrent expert IO the arm
recorded (§8.1), and they are reported split by that IO: none, or some. The
expert-IO terms themselves are **measured-later targets, not promises**. Their
estimates are `9g22`'s (§4.5.3), from this document's facts (§9.2.2):

- **PP512.** A ubatch touches 99.97% of a layer's experts, so it reads the
  whole cold set once. At the 0.29-1.2 GB/s measured for expert slices that is
  ~25-110 s per ubatch at the lowest reading's cold set (*lead ruling §T4*),
  far more than the engram tables' ~2.5-3.5 s. PP on DeepSeek phase 1 is
  bounded by the cold experts, not by the engram IO.
- **Decode.** A token reads `240 x c x (1 - r) x 12.8 MB` of cold experts
  (§9.2.2), where `c` is the cold set's touch share, which the usage profile
  decides. The engram rows of a token are ~4 KiB reads that the deadline order
  puts ahead of any expert read for a later layer (§5.2); the bound that holds
  for them on a shared device is measured, not assumed.
- So "decent speed" for DeepSeek on this host is set by the cold-expert IO,
  and `9g22` §4.5.3 owns that number.

**Decode, S-cold** (48 rows per token). Re-derived from the probe that issues
both tables' reads together through one pool, which is what the engine does (I7;
`batch_probe3 split`, A.2, measured in the same minute as the 4657-IOPS control,
the slower state):

| quantity | measured | target (+15%) |
|---|---|---|
| `T_ticket(1)`: `blk.1`'s 24 rows landed, with `blk.14`'s 24 in flight | mean 7.13 ms, p50 6.81, p90 9.16, p99 14.63 | mean <= **8.2 ms**, p99 <= **17 ms** |
| `T_ticket(14)`: all 48 landed | mean 11.23 ms, p99 21.00 | mean <= **12.9 ms**, p99 <= **24 ms** |
| `T_stall(14)` | | mean <= **max(0, 12.9 ms - T(0..13))** |

`blk.1`'s target ignores the block-0 overlap, so it is conservative.

**Decode, S-warm, bucketed by miss count** (M13). Per token, `m` is the number
of rows (both tables, after dedup) that issued a read. `T_ticket(1)`'s mean over
the tokens of each bucket must stay within 1.15x the single-table miss curve
(`batch_probe3`, A.2) at the bucket's top, or the split probe for the last one:

| `m` | 0 | 1-4 | 5-8 | 9-16 | 17-24 | 25-48 |
|---|---:|---:|---:|---:|---:|---:|
| measured (mean / p99, ms) | - | 2.37 / 5.16 | 2.81 / 6.96 | 4.23 / 9.00 | 5.53 / 13.94 | 11.23 / 21.00 (all 48) |
| bound on the bucket's mean / p99 (ms) | `T_ticket == 0` on >= 99% | **2.7 / 5.9** | **3.2 / 8.0** | **4.9 / 10.4** | **6.4 / 16.0** | **12.9 / 24** |

The aggregate S-warm ceiling is then `sum over buckets of P(bucket) x bound`,
with `P(bucket)` from the simulation at the planned `cache_slots` (§8.5). This
discriminates where the r1 target could not: a partial-miss token is charged
for its misses, not for a full batch.

**PP512.**

| arm | target |
|---|---|
| S-cold | engine ratio >= **0.9**; `T_ticket(1) + T_ticket(14)` <= **1.15 x U / (0.9 x control)**. At a 7070-IOPS control: `llama-bench` random (U 24576) <= **4.4 s**, prose (U 22117) <= **4.0 s** |
| any S arm, `blk.14` | `T_stall(14)` <= **max(0, 1.15 x U / (0.9 x control) - T_ticket(1) - T(0..13))** |
| S-warm (gate cap: ~4.5 GiB of cache in simulation units, §4.5) | the same formula with `U` from the simulation at the planned `cache_slots`. Orientation at 7070 IOPS, bracketing columns: prose (wiki train) 13790 (2 GiB) - 12389 (8 GiB) -> **<= 2.5 - 2.2 s**; mixed 7916 - 6370 -> **<= 1.4 - 1.2 s** |
| S-hot | `T_ticket == 0` |
| phase 2 lookahead | as §8.3; required for DeepSeek (D9) |

**Position arms** (*lead ruling §T1*, C2 and M10). The provider decides the
position; these arms test that decision rather than assume it:

| arm | STORAGE gather | host waits on the path | predicted |
|---|---|---|---|
| **A** (the design) | SYCL-owned gather (§7.2): host task at prefetch, event-edged copy before the consumer | none on the GPU path (§5.3) | best or tied |
| C | upstream order: the gather as a CPU custom op in the leading split (rev 2a's HOST form); its output reaches the device through the same step-3 copy, and the op guards that copy's source by waiting for the previous graph's copy event | the submitting thread waits for the IO before layer 0 is submitted, and for the previous copy when the device runs a graph behind | loses the whole window: ~7-11 ms per cold token; wins nothing on a fully hot cache |

- **No arm B.** r1's B ("expand right before the consumer") is identical to A
  now: the host task is submitted at prefetch whatever the node's position,
  and the copy always sits before the consumer, so moving the node changes
  nothing that runs.
- **Witness on every arm**: the §7.4 record-mode witness, including its
  submission-runs-ahead check (C has no SYCL gather, so its witness is the
  same counters with `lazy_boundary_dispatches == 0`), plus the scheduler's
  split count per graph (`GGML_SCHED_DEBUG`), which must be the same for A
  and C: neither may add a split.
- **Protocol**: S-warm prose (wikitext-2 test through `llama-perplexity` for
  PP, the §8.5 decode vehicle for tg), **5 ABBA quads** of A against C, one
  process per run under P0-P7, B70 tg being the noisy axis (cv 3.3%).
- **Decision rule, fixed now**: C replaces A only if its **median paired tg
  over the 10 pairs beats A's by >= 2%**. Anything less is a tie, and a tie
  goes to the arm with fewer host waits, which is A.

**DSpark** (I9). The draft adds +7.9 GiB of host memory (`9g22` §10.3). Under
owner §D4 that is paid in cold set: 7v5c's `C` grows by about the draft's
bytes (§4.5).
Verify steps carry up to 6 tokens, so up to 288 rows per step: 74 ms mean, 131
ms p99 cold on the Q3_K_M extent (§9.2.3). The DSpark arm gets its own targets
from a Q2_K re-measurement when `9g22`'s gates 1-4 have passed; this document
sets none now.

### 8.5 Runtime against simulation, and the traps

- **The hit-rate match** (M4, M5). The runtime hit rate must match the
  simulation to within **±5 points**, with the simulation re-run under the
  run's own conditions: CLOCK at the **planned** `cache_slots` from the run's
  `[PLACEMENT]` line; the vehicle's exact token stream; `llama-perplexity`'s
  chunking (each chunk of `n_ctx` tokens restarts the n-gram history; the row
  cache persists across chunks); the first 25% of references dropped. Hit and
  miss are counted per reference, as §10 defines them.
- **The decode vehicle** (M4). `llama-perplexity` is PP-only, so S-warm decode
  uses `llama-server` `/completion` with a fixed prose prompt from outside the
  repo (the first 2048 tokens of wikitext-2 train), `n_predict 256`,
  `temperature 0`, and `return_tokens: true`, so the generated token ids come
  back exactly. Re-tokenizing the generated text is not acceptable: a sampled
  token sequence need not be the canonical tokenization of its text. (The
  `return_tokens` field exists upstream; T9d confirms it on this tree before
  relying on it.) The simulation then replays prompt plus generated ids.
- **Corpora a target may quote**: prose and mixed only (§6.2.2).
- **`llama-bench` prompts are uniformly random tokens** (`std::rand() % n_vocab`,
  `694ec2354 tools/llama-bench/llama-bench.cpp:2181-2183`, `:2204-2213`). Their
  n-grams never repeat, so on STORAGE **every** `llama-bench` pp and tg number
  is the cold case, whatever the cache size. A "warm" or "hot" claim needs the
  vehicles above. A flat `llama-bench` curve across cache sizes does not mean
  "the cache does nothing".
- **Filesystem changes move every S number by an order of magnitude.** The
  ext4-on-NVMe reference measured 110 k IOPS at QD64. On a file without
  per-256-KiB checksums on an NVMe target, Qwen's S-cold PP512 would be
  ~75-100 ms per ubatch instead of ~1 s, and DeepSeek's ~0.25-0.3 s instead of
  ~3.5 s. These are projections, not targets, and there is no such target on
  this host today (D3, §13).

---

## 9. Generalisation to any `TENSOR_READ_LAZY` tensor (item 7)

Nothing above is specific to one model except three arch-supplied facts:
`rows_per_token` (for the planner's floor and staging size), `consumer_layer`
(for IO priority, the consumer device and the position) and, optionally,
`peek_rows` (for lookahead). Two consumers with different geometry already
exercise it: one Q8_0 table read 16 rows per token at block 1, and two Q2_K
tables read 24 rows per token each at blocks 1 and 14.

- **Any row table.** Type, `ne[0]` and `row_size = ggml_row_size(type, ne[0])`
  come from the tensor. Quantized rows are whole blocks by construction.
  Dequant goes through the type traits.
- **Consumers.** Every consumer of a lazy tensor goes through
  `build_lazy_get_rows`. That covers an indexed `get_rows` (qwen4exp's hashed
  rows; DeepSeek's engram rows; gemma4's `inp->tokens`,
  `694ec2354 src/models/gemma4.cpp:450`) and a constant-row view: gemma4's
  `ggml_view_1d(..., row 0)` (`:462`) becomes `build_lazy_get_rows` with a
  constant one-row index, supplied by a new input class
  `llm_graph_input_const_rows` whose `set_input` writes the constant ids and
  whose `can_reuse` returns true when the ids are unchanged (M9).
- **The consumer check is a scan, not an assert at the call** (M9). Nothing at
  the call site can see a stray consumer (and arm C's custom op has no edge to
  the lazy tensor at all). After graph build, `llm_graph_context` scans
  every node's `src[]` and follows each `view_src` chain to its root; any node
  other than a lazy `GET_ROWS` whose chain reaches a lazy tensor fails the
  build with `GGML_ABORT` naming the node and the tensor. Under SYCL the lazy
  tensor's data is a non-dereferenceable sentinel (§3.2), so an unhandled
  consumer would otherwise fault at run time, far from its cause. T6's RED: a
  graph with a `ggml_view_1d` of a lazy tensor fails the build, naming it.
- **Several tables.** Each gets its own `lazy_table`, its own residency, and a
  share of the leftover host budget (§4.3 step 5).
- **Non-SYCL builds keep upstream's mmap path verbatim.** The divergence is
  confined to the loader's SYCL branch, the descriptor buffer and
  `build_lazy_get_rows`.

<a id="th32-contract"></a>
### 9.1 The consumer contract

What a model implements to get the residencies, the cache, the prefetch and the
accounting. `llama.cpp-9g22` (`docs/design/deepseek41-runtime.md` §3.3)
consumes this contract and links here for it; this section is the one source.

1. **Mark and declare at load.** Create the table with `TENSOR_READ_LAZY`. In
   `load_tensors`, before the plan is computed, declare its `rows_per_token`
   and `consumer_layer` through the loader
   (`ml.lazy_table_info(tensor, rows_per_token, consumer_layer)`). They feed
   the inventory (§4.1). A lazy tensor without a declaration is refused at
   load, naming it, rather than planned with a guessed floor.
2. **One entry point.** Every read of the table goes through
   `build_lazy_get_rows(lazy, rows, consumer_layer)`, and the layer loop calls
   `lazy_expand_for_layer(il)` at the top of each iteration. The post-build
   scan above enforces it.
3. **The rows input.** `rows` is an I32 graph input of
   `rows_per_token x n_tokens` elements (DeepSeek: 24 per token per table),
   fully written in `set_input` from host state only, with no dependency on any
   device result. Every value is in `[0, ne[1])`; `prefetch` aborts on an
   out-of-range id rather than read past the table. Padding, sequence starts
   and blocking tokens are the model's hash's business. int32 ids suffice while
   tables have fewer than 2^31 rows (DeepSeek: 384.0 M).
4. **The output.** F32 `[ne[0], rows_per_token x n_tokens]`, the shape
   `ggml_get_rows` returns; the model reshapes it (DeepSeek: to
   `[6144, n_tokens]` per table). It is an activation (§7.3). Raw blocks are
   not offered.
5. **Position and executor: the provider decides** (*lead ruling §T1*, C2). The
   model does **not** choose where the gather sits or what runs it. The
   helper places the gather of either residency at the head of the consumer
   device's first split as a SYCL-owned op (§7.1; DQ-C2 in §13 for the HOST
   case), and §8.4's arms test that choice. A consumer must not hard-code "before
   block N": that is the r1 contract, withdrawn.
6. **Optional lookahead.** `peek_rows(next_ubatch, tail)` returns the next
   ubatch's rows from the model's sequence state, so the IO of ubatch `k+1`
   overlaps the compute of ubatch `k` (§5.4). The provider bounds lookahead to
   depth 2, which is what the double landing slab holds.

The SYCL side of the same interface:

| piece | what it is | owner |
|---|---|---|
| `lazy_table` | model-scoped unified-cache object, keyed by the weight's `ggml_sycl_cache_id`; refcounted by the descriptor buffer and every outstanding ticket; llama holds only the descriptor buffer, never a row pointer | the unified cache (§3.2) |
| descriptor buffer | `ggml_backend_sycl_lazy_buffer_type(consumer device)`: sentinel base, nominal size, `WEIGHTS` usage | the model, like any weight buffer (§3.2) |
| `lazy_gather_ctx` | per context, per table: the output slot, the host queue; for STORAGE also the landing slab and the current ticket | its `mem_handle`s, released in its destructor |
| block ring | per context, shared by its tables | a `mem_handle` |
| `prefetch(gather_ctx, rows, n, class) -> ticket` | called by `lazy_prefetch()` after `set_inputs()` (§7.5); it submits the gather's host task, with no device dependency, and for STORAGE first queues the reads (§7.2) | |
| the gather | both residencies: SYCL host task + explicit event-ordered H2D copy (§7.2, §7.3) | th32 owns it (*lead ruling §T1*, D6(c); *§T2*, DQ) |
| `lazy_io_submit(fd, offset, length, dst handle, ticket, class, deadline)` | the one IO entry point, shared with `llama.cpp-yl3v`'s cold-expert slice reads (§5.2, §9.2) | th32 (T5) |

### 9.2 Cold routed experts on STORAGE (phase 1, owner §D4; the path is `llama.cpp-yl3v`'s)

**Status.** On 2026-09-26 the owner first ruled Q3_K_M with its coldest routed
experts on STORAGE (owner §D1), then made Q2_K the primary and moved the cold
experts to a later ticket (owner §D2). The Q2_K fit then fell short of this
host, and the owner chose **"Zones + SSD cold experts"** (owner §D4): lever (s)
plus the coldest Q2_K experts on STORAGE, executed by the CPU. That sanctions
STORAGE→CPU for experts and brings `llama.cpp-yl3v` forward to Q2_K phase 1.
So cold experts **are** on STORAGE in phase 1.

**Who owns what** (the owner's either/or in §D4: this design names the
interface). `llama.cpp-yl3v` owns the cold-expert path: choosing the experts
(the usage profile, then the `exp_probs_b` bias), executing them on
`CpuExpertPool`, its landing slab, and prefetching from routing. The expert
read path differs from the row path in exactly those places: its unit is an
expert-layer's three slices (~12.8 MB on Q2_K), not a row; its reads are known
only once the layer's router has run, not at `set_input`; and its executor is
the CPU expert pool, not the gather host task. This design provides the rest:

| th32 provides | used by yl3v as |
|---|---|
| the `STORAGE` residency state (§3.1) | the reported state of a cold expert-layer |
| `lazy_io_submit(fd, offset, length, dst handle, ticket, class, deadline)` and its pool (§5.2) | the slice reads, class 0, deadline = the expert's layer, in chunks of at most 512 KiB |
| the ticket and epoch machinery (§5.3) | completion and cancellation of a slice load |
| the host ledger and `host_planned_bytes` (§4.2) | the charge for the landing slab (`2C/40`) and the credit of `C` |
| the order of §4.3, with the cold set sized in step 5 by 7v5c (*lead ruling §D5*) | the size `C` its selection fills |

The cold-set **size** is not a th32 number either: 7v5c computes it at load
(§4.3 step 5), and `9g22` §4.5.2 records what it reaches on this host (§4.5).

#### 9.2.1 What the interface keeps open

- **A planned, cache-owned `O_DIRECT` landing slab.** The engine of §5.2
  reads into `mem_handle`-owned `HOST_CPU` memory whose size the planner charges
  (§3.3). Its read unit is an (fd, offset, length, destination, ticket)
  request, not a row: a 4 KiB row block and a 512 KiB slice chunk are the same
  call. yl3v adds its slab role, not a new engine.
- **A hotness ranking by expected reads per step.** The planner's inputs are
  per-object sizes plus an expected-reads-per-step score. For a lazy table it
  is `rows_per_token` (§4.1). For an expert-layer it would be its expected uses
  per step, from the usage profile. §4.3's ordering, "charge the warm set,
  place the coldest last", then covers both; for experts the §D1 selection
  supplies the order and §D5 the size.
- **The CPU executor already takes a lease.** `cpu_expert_task` carries a
  `weight_lease` `mem_handle` (`76c7f6548 ggml/src/ggml-sycl/cpu-dispatch.hpp:127`)
  and the host AOS executor consumes file-layout blocks
  (`:136-149`), so a cache slot filled from STORAGE can be executed as a host
  expert is today. P3 then holds as it does for lazy tables: the CPU executes,
  and only the expert's output reaches the GPU.

<a id="th32-cold-expert-facts"></a>
#### 9.2.2 Facts recorded for the cold-expert path

| fact | value |
|---|---|
| one expert of one layer, Q2_K (phase 1) | **12.8 MB**: gate and up Q2_K 3870720 B each, down Q3_K 5068800 B (4.58 GiB per layer / 384) |
| one expert of one layer, Q3_K_M | **16.77 MB**: gate and up Q3_K 5068800 B each, down Q4_K 6635520 B; **18.25 MB** at layers 0-1 (down Q5_K 8110080 B) |
| where its bytes are | three contiguous slices, one in each `ffn_{gate,up,down}_exps`, ~2 GB apart in one shard; expert `x` starts at `data_offset + x * slice`; slices are not 4 KiB-aligned (5068800 = 1237.5 x 4096) |
| one expert-layer, 16 x 512 KiB chunks in flight, fresh DeepSeek shards | **14.1 ms mean, 29.1 ms p99, 1.2 GB/s**; about 70% of the bytes came from the NVMe, because the shards were minutes old |
| the same, 2 / 6 / 12 loads concurrently | 22.5 / 122 / 222 ms per load; 1.5 / 0.83 / 0.91 GB/s aggregate |
| the same shapes on a settled Qwen split (mostly on the T7) | **0.29 GB/s** (1 load) to **0.47 GB/s** (6 loads); the T7's 5 Gb/s link is the cap |
| reading promotes | `promote_target=nvme` wrote 2.7 GB to the NVMe during a 12 s settled read, so repeatedly read extents become NVMe-cached |
| PP coverage | a PP512 ubatch uses 99.97% of a layer's 384 experts (1 - (1 - 6/384)^512), so PP reads the whole cold set once per ubatch |
| static prior | the owner's fallback is the `exp_probs_b` bias (owner §D1, §D2). **A risk to it**: `exp_probs_b` is DeepSeek-V3-style load balancing, which equalises expert load in training, so the prior may carry little signal about which experts are cold (`hash_layer_count = 0`). The usage profile is therefore the primary input, and whether it must be mandatory is `llama.cpp-9g22`'s call |
| decode cost model | `240 x c x (1 - r) x S / BW` per token, where `c` is the cold share of expert uses, `r` the share a RAM cache serves, and `S` an expert-layer's bytes: **12.8 MB on Q2_K** (phase 1), 16.8 MB on Q3_K_M |

Raw logs are in Appendix A.2 (`expert_probe`).

<a id="th32-dspark"></a>
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
  [PLACEMENT] lazy per_layer_token_embd.weight residency=HOST table=51880.0 MiB resident=51880.0 MiB locked=yes output_slot=10.0 MiB/ctx host_budget=... host_planned=... leftover=...
  [PLACEMENT] lazy blk.1.engram_embd.weight residency=STORAGE table=30762.6 MiB cache=... MiB (N slots, index ... MiB) floor=... yieldable=... MiB output_slot=24.0 MiB/ctx consumer_layer=1 device=... rows_per_token=24 file=<split> off=5785327904
  [PLACEMENT] lazy blk.14.engram_embd.weight residency=STORAGE table=30763.4 MiB cache=... consumer_layer=14 ... off=4032
  [HOST-PLAN] budget=... (source: computed|GGML_SYCL_HOST_RESERVE_MB) planned=... weights=... named_terms=... kv=... zones=... (caches carving: 1|2) envelope=... lazy_resident=... reserve=... cold_set=... (expert-layers ..., slab ...) p3_post_load=...
  ```

  (The `host_budget` figures are elided because they depend on the tenants'
  memory at load, §4.5.)
- **Activation check.** When the `lazy_table` is materialized, it sums its
  handles' sizes and compares them with `host_resident_bytes` from the plan; the
  ledger does the same for every host byte at each publish. A mismatch is a
  `GGML_LOG_ERROR` (`[HOST-PLAN-BUG]`) plus a typed refusal: plan != reality is
  a defect, never tolerated. T4's RED: a fixture whose materialization allocates
  one extra chunk fails the check, naming both numbers.
- **Counters, compiled unconditionally** (M6). They are plain relaxed atomics in
  every build, with no define and no environment gate, printed per context at
  free at WARN, and exposed to tests through a proc-address hook. Per table:
  `rows_requested`, `rows_hit`, `rows_miss`, `rows_joined`, `blocks_read`,
  `bytes_read`, `stall_ns_total` (`T_ticket`), `stall_ns_max`,
  `gather_ns_total` (`T_gather`), `ubatches`, `tickets_cancelled`,
  `chunks_yielded`, `lazy_boundary_dispatches`, `slot_wait_ns`,
  `lazy_output_copies`, `lazy_zero_copy_reads` (§7.3). Per pool:
  `class0_reads`, `class1_reads`, `inflight_joins`, and the reads and bytes
  submitted by other clients (yl3v's slices), so the shared-device record of
  §8.1 comes from the same place.
- **Definitions** (M5), chosen so that runtime counts match the simulation's
  per-reference counting:
  - Every row reference in `rows` is one of: a **hit** (the row is in the
    cache when `prefetch` classifies it), a **join** (the same row is already
    in flight, for an earlier reference of this or another ticket), or a
    **miss** (neither).
  - `rows_hit` counts hits **and** joins, because the simulation, which
    inserts on first reference, scores a repeat as a hit; `rows_joined` breaks
    the joins out.
  - `rows_miss` counts misses, including a row whose *block* is in flight for a
    different row: that is a miss in the simulation, though it issues no read.
  - `blocks_read <= rows_miss`, and the difference is the block-sharing saving.
  - `rows_hit + rows_miss == rows_requested` is asserted at free.
- **Positive controls** for all of the above are §8.1's table: H must show its
  line and `rows_requested > 0`; S must show `rows_miss > 0` and
  `blocks_read > 0`; a `--lazy-mode off` run must show neither.

---

## 11. Proposed implementation tickets (the lead files them after review)

Each ticket names the RED that must fail before its code exists (*lead ruling
§T1*, I10). "GPU" marks work the lead runs.

| # | ticket | depends on | RED (fails first) | notes |
|---|---|---|---|---|
| T1 | `alloc_tier::HOST_CPU`: raw wrapper (`mmap` + `MADV_HUGEPAGE` + `mlock`), `must_host_cpu`, `device_mappable()`, the 36-site tier census (§3.1); `lazy_residency`; `cache_location`/`expert_resolve_tier` gain reported `STORAGE`; roles `LAZY_ROW_CACHE`/`LAZY_STAGING`/`LAZY_OUTPUT`; canonical contract §3/§4/§12.5 edits | - | a host test that requests `HOST_CPU` under `RLIMIT_MEMLOCK` = 1 page and expects the typed refusal `host-cpu-lock-failed` naming `errno` and the bytes; a census test that a `HOST_CPU` allocation appears in the registry and the shutdown census | header change first: every other ticket includes it |
| T2 | the process-global host ledger: one budget function, pools capped by their sum, `NO_PINNED` and `RESERVE_MB` semantics, `host_planned_bytes` per §4.2, the plan's host fields, `host-budget-exceeded`, `[HOST-PLAN-BUG]` | phase (a): T1, **`llama.cpp-rnh3`**, `llama.cpp-moua` (the shared publish struct and lifetime order), `llama.cpp-zhcn` (the `SYCL_Host` compute buffer as a planned tenant); computed through the functions `llama.cpp-7v5c` runs. Phase (b) adds **`llama.cpp-sgc5`** (one zone set per plan) and `llama.cpp-s4ip` (lever (s), for the regression gates' expected bytes) | a planner fixture whose host weights exceed the budget expects `host-budget-exceeded` with each term; a Q2_K-shaped fixture expects `host_planned_bytes`' zone terms to equal what `populate_host_zone_sizing` and the `configure_host_zones` sizing return for the same inventory, the expected bytes **derived from those functions, never a literal** (*lead ruling §D5*): with the ring, `dma_staging_pool` and the RUNTIME pool included, `cpu_quant_buffer_bytes` excluded, and `pp_moe_onednn_scratch_bytes == 0` once `s4ip`'s type gate lands; a `PER_DEVICE` fixture with two caches expects their pools capped by the **sum**; a fixture whose pre-allocation runs short expects the refusal, not the WARN (§4.2) | **changes behaviour for every model.** Lands in two phases: (a) WARN-only, logging what it would refuse, now (*lead ruling §T3*, (c)); (b) enforcing, only after the regression gates pass: GPT-OSS 120B (host experts, `level_zero:0`), Qwen3.5-35B-A3B on the B50, the `glkg` 1% VRAM fixture, and the Mistral gate, all still admitted (GPU) |
| T3 | `placement_lazy_table_info` inventory class, excluded from `max_tensor` and every inventory maximum; the residency decision (§4.3); the plan lines (§10) | T2 | a Q2_K-shaped inventory fixture with its two tables asserts STAGING and SCRATCH equal to `populate_host_zone_sizing()`'s result for the same inventory without the tables (derived, not a literal; a counted table adds at least twice its bytes, §4.1), and the residencies of §4.5 at a fixed budget | |
| T4 | the `lazy_table` object: refcount, descriptor buffer type, HOST body load (eager, locked, D1), STORAGE fd, row cache (CLOCK, sharded, 13.3 B/slot index, 256 MiB yieldable chunks), the activation check | T1, T3 | an activation-check mismatch fixture (one extra chunk) must refuse naming both numbers; a yield fixture must free a chunk with no wait while rows remain gatherable | |
| T5 | the IO engine: pread pool, priority classes, per-context fairness, in-flight dedup, tickets with epochs, landing slabs, block ring | T4 | the abort-mid-ubatch fixture of §5.3 (slot 0 must hold ticket B's rows exactly); a head-of-line fixture (the `batch_probe3 hol` shape against a file fixture) where class-0 completion must not wait for class-0 reads of another context queued earlier | CPU-only |
| T6 | llama side: loader SYCL branch for `TENSOR_READ_LAZY` (independent of `mmap_support`, resolved before upstream's `AUTO`->`OFF`), SYCL device reports `mmap_support = true`, `lazy_table_info` declaration, `build_lazy_get_rows` + arm C's CPU custom op + `lazy_expand_for_layer` + `lazy_prefetch()` + the post-build consumer scan, gemma4's const-row input class; qwen4exp and gemma4 declarations | T4, T5, `llama.cpp-n77l` | (a) with `AUTO` and a device reporting `mmap_support = false`, the dual-card config still prints `[PLACEMENT] lazy ... residency=`; (b) a graph with a `ggml_view_1d` of a lazy tensor fails the build naming it | `deepseek41` follows the §9.1 contract in `llama.cpp-9g22`'s own tickets |
| T6b | llama handling of the descriptor buffer: memory breakdown and model size report resident bytes, `load_all_data` skips the tensor in its mapping branch, including the mlock growth there (`694ec2354 src/llama-model-loader.cpp:1640-1662`, whose `lazy.has(cur)` tests the mmap lazy set this design no longer fills) | T6 | a load of a lazy-table fixture reports model size equal to the resident bytes, and `load_all_data` reads zero bytes of the table | |
| T7 | the SYCL-owned gather for **both residencies**: lazy `GET_ROWS` recognition, the side-queue host task (body or ticket), the explicit event-ordered H2D copy, the output-slot reuse marker, fusion exclusion, the record-mode segment boundary (§7.4), the `lazy_output_copies` and `lazy_zero_copy_reads` counters | T5, T6 | **first the four probes HT-1..HT-4 (§7.2), lead-run, GPU**; then the §7.3 zero-copy fixture (fails first with `lazy_output_copies == 0`; its probe's positive control is `inp_embd`'s consumer); then a record-mode fixture: a graph with a lazy gather must replay with `seg_replay > 0`, zero futility WARNs, and `lazy_boundary_dispatches == graphs` per table; and the §7.4 submission-runs-ahead check on a two-table fixture | if a probe fails, D6-M (b) replaces the host task before anything else is built |
| T8 | counters (unconditional), their definitions (§10), and CPU-only unit tests, with both hash harnesses of Appendix B as fixtures | T5, T6 | a counter test where a repeated row in one ubatch counts as a join and a hit, and a same-block different row counts as a miss with no extra read | |
| T9q | Qwen acceptance against §8.2-8.3: H, S-cold, S-warm, S-hot, the correctness gate, the positive and negative controls, the engine ratio (GPU) | T7, T8 | an `--lazy-mode off` run must fail the engagement check (the negative control proves the check can fail) | |
| T10 | `peek_rows` lookahead (class 1, depth 2) | T9q | a lookahead fixture where rows missed by ubatch `k` and `k+1` are read once | required for DeepSeek (D9) |
| T9d | DeepSeek acceptance against §8.4: S arms, executor arms A/C, the submission-runs-ahead check, bucketed decode, the runtime-vs-simulation match, inside `9g22`'s P0-P7 (GPU) | T10, the `deepseek41` runtime (`9g22` T6, T11), `9g22` gates 1-4, **`llama.cpp-s4ip`** (lever (s), the ring and RUNTIME scope included), **`llama.cpp-yl3v`** (the cold set on STORAGE) and **`llama.cpp-7v5c`** (its size at launch) | the §8.4 witness must fail on an arm run with `GGML_SYCL_DISABLE_GRAPH=1` (record mode off: `seg_replay == 0`) | its S arms run with the §D4 cold set resident on STORAGE, and record the concurrent expert IO (§8.1) |
| T11 | optional: `io_uring` engine | T9q | engine ratio on the ext4 NVMe reference must exceed the pool's | |

**Existing tickets this design depends on or leaves open:**

- `llama.cpp-rnh3`: exact host weights plus named terms; T2 depends on it.
- `llama.cpp-sgc5`: every participating cache builds a full host zone set,
  WEIGHT included, from one plan (§2.3). T2's enforcing phase depends on it
  (§4.2): without it, the sum cap refuses a two-card host-expert load that no
  lever can clear. Part of lever (s) (owner §D4).
- `llama.cpp-s4ip` (comments `c-rbwa`, `c-td91`): lever (s)'s host side, the
  SCRATCH and RUNTIME pool terms with no host reader (`dma_staging_pool` and
  the ring, each twice) and the ring's MXFP4 type gate, which also frees the
  ring's VRAM. T2 counts whatever the sizing code commits, before or after it;
  T2's enforcing phase and T9d wait for it.
- `llama.cpp-7v5c` (9g22's T27): the host-only plan dry-run, the one
  computation behind `host_planned_bytes`, `expected_host` and the cold set
  `C` (§4.2, §4.3 step 5; *lead rulings §T3, §D5*).
- `llama.cpp-moua`: lifetime segregation and the shared publish struct; T2.
- `llama.cpp-zhcn`: the `SYCL_Host` compute buffer as a planned host tenant,
  which holds the HOST gather's output (§7.3); T2.
- `llama.cpp-n77l`: the upstream merge that brings `TENSOR_READ_LAZY`; T6.
- `llama.cpp-x1xd` and `llama.cpp-jhxq`: not on this design's path (§5.3); they
  stay open for the paths that do reach them.
- `llama.cpp-yl3v`: cold routed experts on STORAGE, executed by the CPU,
  brought forward to Q2_K phase 1 by owner §D4 (§9.2). It uses T5's engine
  and T2's ledger through the interface of §9.2; T9d waits for it.
- The census of CPU-split outputs that SYCL reads zero-copy (`inp_embd`
  first), which the lead files under §T2 DQ. This design adds no member to it
  (§7.3).

**Missing tickets from r1, now in the table:** `mmap_support` independence (T6
RED a), the `mlock` policy (T1), llama handling of the descriptor buffer (T6b),
and the host-task probes (T7). r1's T8 is split into T9q and T9d, with T10
between them, so that nothing depends on a later ticket.

---

## 12. Not done, and why

- **MTP-speculative prefetch.** The MTP head could predict token `t+1`, and
  its rows could be prefetched during token `t`. That turns decode's cold stall
  into a speculation-hit question. It is worth a ticket once S-cold decode is
  measured, not before.
- **A static hot set shipped with the model.** It would need a corpus-derived
  frequency list per model. The CLOCK cache reaches the same place after a few
  thousand tokens.
- **Caching dequantized F32 rows.** That would take 3.8-12x the memory to save
  a ~2 ms dequant per PP ubatch.
- **Placing the table on the GPU.** It is forbidden by P3, and pointless: 84-170
  B random rows are a CPU-shaped access.
- **Regrowing a yielded row cache between plan publishes** (§6.4). It needs a
  trigger the ledger does not have; v1 regrows at the next publish.
- **A scheduler event edge for the zero-input split sync** (`llama.cpp-x1xd`).
  This design no longer reaches that sync (§5.3), so it is not a dependency;
  the ticket stays open for the paths that do.
- **The cold-expert path itself.** It is in phase 1 (owner §D4), but it is
  `llama.cpp-yl3v`'s; this design names its interface (§9.2) and records the
  measurements it starts from.

---

## 13. Decisions

Each open decision offers options, and the recommended one is marked
**(rec)**. Recommendations are this design's proposals. A decision that is
already made says by whom: *lead ruling* with its section for the lead's
rulings in `lead-rulings-2026-09-26.md`, and "owner §D1/§D2/§D4" for the
owner's.

**D1. How is the HOST residency populated? Ruled (a)** (*lead ruling §T2*).
- a) **(ruled)** Eager: at load, a sequential `O_DIRECT` read of the whole table
  into the locked body, blocking model load. It is simplest, H is H from the
  first token, and a failure (lock or read) surfaces at load. r1 quoted
  "~31 s" for it; that read was NVMe-cached (A.2: 19.7 of 27.6 GB from the
  NVMe). **On settled data it is minutes**: 50.66 GiB at the 0.29-0.47 GB/s
  measured on a settled split (§9.2.2) is ~2-3 minutes, the same order as
  loading the host experts from the same filesystem.
- b) Background fill: the load returns at once and the table fills behind it,
  served through the STORAGE path until each chunk arrives. It would make the
  HOST residency carry the whole STORAGE machinery plus per-chunk ready bits,
  for a one-time saving.
- c) On-demand only. A HOST table would never become the H arm.

**D2. IO engine? Ruled (a)** (*lead ruling §T2*). Owner §D4 adds a second
client, `llama.cpp-yl3v`'s cold-expert slice reads, through the same pool
(§5.2).
- a) **(ruled)** A portable `pread` worker pool first, with priority classes,
  per-context fairness and in-flight dedup (§5.2). It measured at the device
  ceiling (ratio 0.99) and adds no dependency.
- b) `io_uring` through raw syscalls only.
- c) Add a liburing dependency.

**D3. Where the tables live. A fact, not a choice** (*lead ruling §T1*, D3).
Plain NVMe is **not** an option on this host. `/` (ext4 on NVMe) has
**33-38 GiB free** at 97% use (34 GiB at 19:00 today), and the tables come in
sets: **60.08 GiB** for the DeepSeek Q2_K pair, **78.68 GiB** for the Q3_K_M
pair, and **50.66 GiB** for Qwen's table. None fits, and a per-table comparison
would invite a half-measure on a nearly full system disk (M1). The tables stay
on bcachefs as it is; every S target in §8 is that filesystem's. The one
filesystem lever left, per-file bcachefs targets on the engram shards, is the
owner's filesystem policy, and this design does not propose it for v1.

**DLM. `--lazy-mode` under SYCL? Ruled (a)** (*lead ruling §T2*, where it is
called D4; renamed here so that "§D4" means only the owner's decision).
- a) **(ruled)** Keep upstream's meaning as *eligibility* (`OFF` = ordinary
  weight, `AUTO` = marked and larger than 4 GiB, `ON` = all marked). The
  planner alone picks HOST or STORAGE for eligible tensors, and the SYCL branch
  does not inherit upstream's `AUTO`->`OFF` fallback on `mmap_support` (§4.1).
- b) Also let `OFF` force HOST and refuse STORAGE.
- c) Ignore the flag under SYCL.

**D5. Where is each gather placed? Ruled** (*lead ruling §T1*, C2): the
provider decides, by residency. As ruled, HOST gathers stay in the leading
`inp_embd` split as a CPU op, and STORAGE gathers head the consumer device's
first split as a SYCL-owned op (§7.1). Rev 2b proposes changing the HOST half
under DQ (DQ-C2, below). §8.4's arms A and C test the STORAGE executor with a
fixed decision rule. r1's "expand before each consumer block" is withdrawn.

**D6. Who executes the STORAGE gather? Ruled** (*lead ruling §T1*, D6(c)): the
SYCL backend, as a CPU `host_task` with event edges, no scheduler split, no
host wait. th32 owns the op; `llama.cpp-9g22` consumes it through §9.1. The
record-mode question the ruling left to this design is answered in §7.4: a
stated segment boundary, not a recorded host-task node.

**D6-M. The mechanism under D6(c). Ruled** (*lead ruling §T2*): (a), with
(b) as the fallback; (c) is rejected. §7.2 pre-registers four probes
(HT-1..HT-4) against the fork's own warning about `host_task`
(`ggml-sycl.cpp:32755`) and the measured cross-queue submit block
(`llama.cpp-c6ah`).
- a) **(ruled)** `host_task`, **if** HT-1..HT-4 all pass.
- b) **(ruled fallback)** If any fails: the IO reaper dequantizes (for a HOST
  table, a pool thread does, from the body) and signals a host-signalled Level
  Zero event wrapped with `sycl::make_event`; the output copy depends on it.
  Still backend-owned, event-edged, split-free and wait-free; it needs Level
  Zero interop the fork has not used before.
- c) **Rejected** (*lead ruling §T2*): the submitting thread waiting for the
  ticket at the consumer boundary is a host wait. If (b) also fails, that goes
  back to the lead as a new question; this design does not offer (c).

**D7. Size of the STORAGE row cache. Ruled** (*lead rulings §T1* I8, *§T2*,
*§T4* I-4): sized after the context envelope and **yieldable**, since it holds
no leases; in 256 MiB chunks; yielded chunks come back only at the next plan
publish (§T2: no regrow until then). In a plan with a cold set the cache is
**fixed** at its target and charged before the cold set is sized (§T4 I-4);
without one it takes what is left (§4.3, §6.4). The target this design
proposes for DeepSeek is the gate size, 2 GiB per table with the index.

**D8. DeepSeek-V4.1-Flash: the rung and its fit. Decided by the owner**
(owner §D2: Q2_K primary; owner §D4: "Zones + SSD cold experts"). The lead
ruled that §D4 answers the fit and is not re-asked (*lead ruling §T3*). With
§D4 applied:
- **The fit** is case A after lever (s) (§4.5), minus the bytes of the cold
  set `C` on STORAGE, plus its landing slab. 9g22's P3 is the binding check:
  after (s) and before any cold set, `expected_host` is 171.1-174.5 GiB and P3
  fails at every reading taken today (§4.5).
- **The cold-set size** that clears P3 is computed at load by
  `llama.cpp-7v5c` as `C = max(C_selected, C_P3)` (*lead ruling §D5*; §4.3 step
  5). Its value at the lowest reading, 171.4 GiB, and at 189.6 GiB is
  `9g22`'s, in
  [its §4.5.2](deepseek41-runtime.md#452-the-host-fit-r2-c-a-r3-c-1-c-2-owner-d4);
  this document does not restate it. It is an input to 9g22's cold-set
  selection (owner §D1), not a th32 choice.
- **If even `C` = every routed expert cannot clear P3**, the load refuses as
  `cold-set-cannot-clear-p3`, naming the reading and the shortfall (§D5).
- **Today's code** refuses phase 1 at every reading even for the planner, so
  the gates wait for (s) and yl3v (T9d, §11).

Rev 2a's "goes back to the owner as a re-check" is withdrawn: the owner had
already decided (§T3).

**DQ. The gather's output reaches the device by an explicit copy. Ruled**
(*lead ruling §T2*): the HOST-tier gather output reaches the device by an
explicit event-ordered H2D copy on the main in-order queue, the same path as
STORAGE; the GPU does not read `SYCL_Host` memory zero-copy. The design is
§7.2 step 3 and §7.3, with a RED that fails on a zero-copy read (§7.3, T7).
The general census of CPU-split outputs that SYCL reads zero-copy is the
lead's separate ticket.

**DQ-C2. Where the HOST gather runs under DQ** (proposed; lead-level; it
changes the HOST half of *lead ruling §T1* C2).
- a) **(rec)** The one path for both residencies (§7.1, §7.2): the SYCL backend
  owns the HOST gather too; its host task dequantizes from the resident body
  into the output slot at prefetch, and the step-3 copy moves it. The copy's
  source is guarded on the host task's thread (the output-slot rule of §7.2),
  so nothing waits on the submitting thread. It is what §T2's "one code path
  for both tiers" reads as, and HOST then also depends on HT-1..HT-4 (or
  D6-M(b)).
- b) Keep C2's form: the CPU op in the leading split, and the SYCL backend
  copies its output from the `SYCL_Host` compute buffer before the consumer.
  Nothing orders ubatch `k+1`'s CPU op after ubatch `k`'s copy in a
  multi-ubatch prefill (§7.1), so the CPU op would have to wait for that copy
  on the submitting thread: a host wait on the GPU, which this design does not
  offer as a baseline. Arm C (§8.4) runs exactly this form, as a comparison
  arm.

**D9. Lookahead in v1 for STORAGE-bound models? Ruled (a)** (*lead ruling
§T2*).
- a) **(ruled)** Yes for DeepSeek: T10 lands before T9d. Its tables are always
  on STORAGE here, and a cold PP512 ubatch is ~3.5-4.4 s of IO that, without
  lookahead, only `blk.14`'s window can partly hide.
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

**Rev 2: DeepSeek-V4.1-Flash Q2_K and Qwen3.8 Q8_0 weight census** (GGUF
tensor headers only; `ds_q2k_census.py`, `qwen_census.py`):

```
ENGRAM DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf blk.1.engram_embd.weight Q2_K [256, 384006168] 32256518112 off 5785327904
ENGRAM DeepSeek-V4.1-Flash-Q2_K-00004-of-00007.gguf blk.14.engram_embd.weight Q2_K [256, 384016682] 32257401288 off 4032
output                2 tensors        542996480 B     0.506 GiB
token_embd            1 tensors        217190400 B     0.202 GiB
dense_other         921 tensors       2470050496 B     2.300 GiB
routed_experts      120 tensors     196765286400 B   183.252 GiB
engram_table          2 tensors      64513919400 B    60.083 GiB
total 246.34361562877893
max non-engram tensor 1946419200 blk.0.ffn_down_exps.weight 1856.25 MiB
expert bytes per layer (GiB): 4.581 for each of layers 0-39
expert types {('ffn_down_exps', 'Q3_K'): 40, ('ffn_gate_exps', 'Q2_K'): 40, ('ffn_up_exps', 'Q2_K'): 40}
---
output                4        682434560 B     0.636 GiB
ple_table             1      54400261120 B    50.664 GiB
token_embd            1        675430400 B     0.629 GiB
dense_other        1074       4110179840 B     3.828 GiB
routed_experts      144     128345702400 B   119.531 GiB
total 175.2879548072815 max non-PLE 850.0 MiB blk.0.ffn_down_exps.weight
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

**Rev 2 probes on the DeepSeek Q2_K tables** (`blk.1.engram_embd` in
`Q2_K-00001-of-00007` at offset 5785327904, 32256518112 B; `blk.14` in
`Q2_K-00004-of-00007` at offset 4032). Run 18:35-18:41, in this order:

```
 18:35:56 up 17 days, 23:51, 17 users,  load average: 31.30, 35.91, 35.80
ds2k_blk1_qd64: iops=4657 MB/s=19.1 ios=69989 lat_mean=13740us p50=3064us p99=145752us dev_read_MB={0:6384.9, 1:3572.6, 2:6737.2} amp=58.2x dev_write_MB={0:7500.5, 1:37.7, 2:107.3}
ds2k_blk1_qd16: iops=4249 MB/s=17.4 ios=63765 lat_mean=3764us p50=2867us p99=27656us dev_read_MB={0:6933.2, 1:3383.8, 2:5798.8} amp=61.7x dev_write_MB={0:3506.2, 1:325.1, 2:922.1}
# batch_probe3 split 24 24 64 <secs>: both tables through one 64-worker pool; per batch, time until blk.1's 24 landed, and until all 48 landed
split B1=24 B2=24 workers=64
  table1   n=1768 mean=7.13ms p50=6.81ms p90=9.16ms p99=14.63ms max=42.35ms
  all      n=1768 mean=11.23ms p50=10.84ms p90=14.15ms p99=21.00ms max=42.35ms
# bcachefs io_done around the split run (user read and user write deltas, as the run recorded them)
dev0 read +9.65 GB, write +1.54 GB
dev1 read +3.92 GB, write +0.02 GB
dev2 read +6.58 GB, write +0.04 GB
# batch_probe3 hol 12000 24+24 <order>: a 12000-row prefill queued first, a 48-row decode 200 ms later
hol P=12000 delay=200ms order=fifo workers=64
  decode   n=4 mean=1358.59ms p50=1243.48ms p90=1496.18ms p99=1496.18ms max=1531.59ms
  prefill  n=4 mean=1560.74ms p50=1447.02ms p90=1696.22ms p99=1696.22ms max=1731.20ms
hol P=12000 delay=200ms order=prio workers=64
  decode   n=4 mean=21.73ms p50=19.46ms p90=23.02ms p99=23.02ms max=27.59ms
  prefill  n=4 mean=1664.46ms p50=1626.03ms p90=1699.45ms p99=1699.45ms max=1732.42ms
# same-session control, then the pool, then the control again
ds2k_ctl_a: iops=7070 MB/s=29.0 ios=70775 lat_mean=9051us p50=6849us p99=33161us dev_read_MB={0:7977.7, 1:3974.0, 2:4487.6} amp=56.7x dev_write_MB={0:490.1, 1:0.0, 2:0.0}
batch=20000 workers=64 batches=5 mean=2836.35ms p50=2840.99ms p90=2887.11ms p99=2887.11ms max=2907.78ms rows/s=7051 straddle=1.9%
ds2k_ctl_b: iops=7242 MB/s=29.7 ios=72578 lat_mean=8836us p50=6980us p99=33161us dev_read_MB={0:8253.3, 1:4060.2, 2:4489.8} amp=56.5x dev_write_MB={0:399.8, 1:0.0, 2:0.0}
# single-table miss curve (batch_probe2, 64 workers)
batch=1 workers=64 batches=3985 mean=1.51ms p50=1.25ms p90=2.69ms p99=4.00ms max=10.10ms rows/s=664 straddle=1.9%
batch=4 workers=64 batches=2529 mean=2.37ms p50=2.08ms p90=3.55ms p99=5.16ms max=20.01ms rows/s=1686 straddle=2.0%
batch=8 workers=64 batches=2132 mean=2.81ms p50=2.48ms p90=4.01ms p99=6.96ms max=38.04ms rows/s=2842 straddle=2.2%
batch=16 workers=64 batches=1421 mean=4.23ms p50=3.92ms p90=5.90ms p99=9.00ms max=19.21ms rows/s=3787 straddle=2.0%
batch=24 workers=64 batches=1085 mean=5.53ms p50=5.10ms p90=7.28ms p99=13.94ms max=25.96ms rows/s=4338 straddle=2.1%
```

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

## Appendix B. Probe sources (scratch, not committed; the fit calculator is committed)

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

- `batch_probe3.cpp` (rev 2): two modes over one 64-worker pread pool. `split`
  issues two tables' batches together and times the first table's landing and
  the whole batch's; `hol` queues a prefill batch, then a decode batch after a
  delay, in FIFO or priority order. Files, offsets and row counts come from
  `T1F`/`T1BASE`/`T1NROWS`/`T2F`/`T2BASE`/`T2NROWS`/`ROW`.
- `ds_q2k_census.py`, `qwen_census.py` (rev 2): per-class weight bytes from the
  GGUF tensor headers, the largest non-table tensor, and the engram extents.
  Run with `/usr/bin/python3` (gguf-py needs its numpy).
- `io_probe2.sh` takes `FOFF`/`SIZE` for the extent; rev 2's same-session
  controls use it at `blk.1`'s offset.
- **`docs/design/sycl_storage_tier_host_fit.py` (rev 2b, committed with this
  document)**: the §4.5 host-fit calculator, grown from the scratch
  `c2_fit.py` by adding the PP MoE oneDNN ring, the today / after-(s) columns,
  the RUNTIME pool's chunk commit and 9g22's P3 check. Run it from the
  repository root with `python3 docs/design/sycl_storage_tier_host_fit.py`; it
  has no dependencies. Its constants are the GGUF header facts of A.1 and the
  sizing-code terms of §2.3. It is committed, unlike the probes, because
  §4.5's numbers must stay reproducible until `llama.cpp-7v5c` replaces it.

T8 should lift `upstream_set_input_body.inc`, `ds_sim.cpp`'s `ds_rows()` and
both Python transcriptions into CPU-only unit tests, so that a change to either
hash fails a gate instead of silently changing the rows the cache predicts.
