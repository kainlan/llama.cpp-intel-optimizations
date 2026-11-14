# fmcp addendum A12: the Qwen3.8 PLE table is read from storage, through a row cache of at most 2 GiB

> **Execution:** `team-driven-development`. One implementer per task, with fresh spec and quality reviewers
> per task.
> - **Models.** Opus for T4, T5, T6, T10 and T13. Sonnet for the rest.
> - **Implementers** edit, build their own targets, and run host-only tests.
> - **The lead** runs every device-registered test, every model load and every gate, one at a time, under
>   `GPU.lock`.
> - **Builds** happen only on the lead's literal GO.

**Parent:** epic `llama.cpp-fmcp` (Qwen3.8-Flash-Next on the B70).

**Base:** `integ/th32` at `ec08e5907`. Every line anchor below was read there, unified-cache ones
included. Re-read each anchor before you edit it.

**Order:** this addendum follows A11, which is merged. A10.1 stays on HOLD. This addendum does not depend on
A10.

**Revision:** r3. Plan reviews r1 and r2 are applied, with the lead rulings R1-R7 of 2026-11-14.

**Goal:** a default load of Qwen3.8-Flash-Next on the B70 stops holding the 51,880 MiB
`per_layer_token_embd.weight` table in host memory. Each ubatch reads the rows it needs from the model file.
A row cache sits in front of the file. It holds at most 2 GiB, and never more than the table's own size.

**Architecture:**
- **Who sets what.** libllama marks the table storage-only. The planner places a storage-only table STORAGE
  whatever the host room, and caps its row cache.
- **Opening.** After the load ends, libllama opens the table through four new backend procs.
- **Fetching rows.** The qwen4exp PLE graph input fetches its rows in `set_input`. Cache hits are copied.
  Misses are read through the existing O_DIRECT IO engine and waited for.
- **Host-side conversion.** The gathered rows are converted to f32 on the host with the public type traits.
  They enter the graph as an F32 input `[ne0, n_heads x n_tokens]`, in the place the `get_rows` result
  occupied. In this mode the graph has no table, no `get_rows` and no I32 row input.
- **Failure.** A failed read sets a flag. `process_ubatch` then returns FAILED between `set_inputs` and
  `graph_compute`.
- **Saving.** The model saver refuses a model whose table is STORAGE.
- **Until T13.** No arch has a STORAGE consumer, so AUTO keeps the table EAGER and the default load is
  unchanged.

**Tech stack:**
- C++17 in `ggml/src/ggml-sycl/`: the planner, the IO engine, a new SYCL-free `lazy-gather.{hpp,cpp}`, and the
  backend C API.
- C++17 in `src/`: the lazy-range record, a new `llama-lazy-storage.{h,cpp}`, the loader, the graph input
  plumbing, the model saver, and `models/qwen4exp.cpp`.
- Python source pins under `tests/`.

**Test infrastructure:** ctest.

Host-only. The implementer runs these:
- `test-sycl-lazy-io-engine-host` and its mutant registrations (`ggml/src/ggml-sycl/CMakeLists.txt:525-548`).
- `test-llama-lazy-range-record` and `test-llama-lazy-range-prefill` (`tests/CMakeLists.txt:4186-4192`).
- The pin script `test-llama-lazy-range-record-source` (`tests/CMakeLists.txt:4066-4071`).
- The lifecycle source contract (`ggml/src/ggml-sycl/CMakeLists.txt:953`).
- Two new C++ host tests and one new pin script.

Device-registered. The lead runs these:
- `test-sycl-lazy-residency-plan-{1,2}dev` (`ggml/src/ggml-sycl/CMakeLists.txt:8608-8628`).
- `test-sycl-lazy-plan-interface-{1,2}dev` (`:8636-8650`).
- Both carry `cache;mem-handle;gpu-serial` and `RUN_SERIAL`, and pin the selector.

---

## Evidence

**S1, lead spike 2026-11-14.**
- **Where the table is.** It lives in `Qwen3.8-Flash-Next-exps-IQ4_NL-00003-of-00006.gguf` (54.4 GB), on the
  USB SSD, not on the NVMe. The parent plan doc is stale on this.
- **O_DIRECT.** An O_DIRECT open through `/models` (mergerfs) succeeds.
- **Latency.** 16 reads issued together have a median of 0.66 ms through `/models` and 0.54 ms on the branch
  path.
- **Throughput.** 8192 reads on 64 threads take 218 ms through `/models` and 178 ms on the branch path. That
  is 41.7k and 52.8k reads/s sustained.
- **Branch path.** `getxattr user.mergerfs.fullpath` returns the branch path.
- **Decision.** Open the path as given. Branch-path resolution is a follow-up.

**Shape of the table** (`test-sycl-lazy-residency-plan.cpp:94`):
- It is Q8_0, `160 x 320,001,536`, so `row_size` is 170 B and a row is 5 whole blocks.
- One token reads 16 rows (`hparams.ple_n_heads`).
- `lazy_block_ring_span(170)` is 8192 B (`lazy-io-pool.hpp:46`), so a miss reads 8 KiB to deliver 170 B.
- **Cold-ubatch cost.** A 2048-token prompt reads at most 32,768 rows per 2048-token ubatch. At the S1 rate a
  fully cold ubatch costs about 0.6-0.8 s. Today's whole pp2048 takes about 55 s at 31-38 t/s.

**Precision of the existing `get_rows`.** The SYCL Q8_0 `get_rows` multiplies `d x q` in `dfloat`, and `dfloat`
is `sycl::half` under `GGML_SYCL_F16` (`ggml/src/ggml-sycl/getrows.cpp:1860-1875`). The host dequantizer
works in f32: `dequantize_row_q8_0` (`ggml/src/ggml-quants.c:553-567`) widens `d` with
`GGML_FP16_TO_FP32`, which is the inline bit conversion `ggml_compute_fp16_to_fp32`
(`ggml/src/ggml-impl.h:396`, `:448`). It reads no table, so it needs no initialisation. That is the reason
for R1.

## What the code does today

1. **Record.**
   - The pre-fill (`src/llama-lazy-range-record.cpp:144-207`) gives the table a deferral. With the SYCL taker
     and AUTO, that deferral would be STORAGE.
   - The bridge at `:173` (`!llama_lazy_storage_consumer_present()`, stub at `:243-245`, which returns false)
     turns it into EAGER and prints one WARN.
   - Mode ON keeps STORAGE. `llama_lazy_range_check_storage` (`:247-253`, called at
     `src/llama-model-loader.cpp:1930`) then throws, so the load fails.
2. **Inventory.**
   - libllama never sets `ggml_sycl_tensor_inventory::lazy_tables`. Neither builder sets it:
     `llama_model_sycl_build_early_inventory` (`src/llama-model.cpp:761-799`) and
     `llama_model_sycl_set_late_inventory` (`:825-880`).
   - So the planner's lazy path (`unified-cache.cpp:34759`) never runs for a real model.
   - And the IO pool's coverage pin is never taken. Both sites are
     `take_if(inventory->lazy_table_count > 0)`, at `ggml-sycl.cpp:18757` and `:19286`.
3. **Planner.**
   - It tests HOST first, at `unified-cache.cpp:34849`. A table becomes STORAGE only when that test fails.
   - A STORAGE cache gets its floor plus a share of all the remaining room (`:34895-34920`). There is no
     ceiling.
4. **Materialization and IO.**
   - `lazy_table_create_storage` (`lazy-table.cpp:421-608`) exists and has tests.
   - `lazy_io_submit` (`lazy-io-pool.cpp:1844-1883`) queues reads.
   - Delivery (`:1393-1451`) copies into the landing slot, inserts into the cache, and decrements
     `ticket.outstanding`.
   - Nothing can wait for a ticket to reach 0 (`lazy-io-pool.hpp:321-325`).
5. **Graph.**
   - `llm_graph_input_ple::set_input` (`src/models/qwen4exp.cpp:1195-1256`) hashes the n-grams into row ids,
     and sets the I32 `rows` input.
   - `build_inp_ple` (`:1314-1333`) gathers with `ggml_get_rows(ctx0, model.per_layer_tok_embd, rows)` and
     reshapes the result to `[ple_head_dim x n_heads, n_tokens]`.
   - `process_ubatch` calls `res->set_inputs(&ubatch)` (`src/llama-context.cpp:5367`) and then
     `graph_compute` (`:5372`). Nothing can fail in between. A reused graph also runs `set_inputs`: the reuse
     check is at `:5323`.
6. **Saver.** The saver adds `per_layer_tok_embd` (`src/llama-model-saver.cpp:492`). It fails a model only
   through `GGML_ASSERT` (`:137`), and its entry point returns `void` (`src/llama.cpp:505-510`).

## Decisions

**D1. The gather form.** Owner ruling 2026-11-14: "gather in set_input". The lead added ruling R1.
- **Rows.** The rows come from the row cache. A miss is read from disk and waited for in the input's
  `set_input`, where the ids are hashed.
- **What the graph gets.** The rows are converted to f32 on the host with
  `ggml_get_type_traits(type)->to_float` (`ggml/include/ggml.h:3055-3068`). The graph gets an F32 host input.
- **What the graph does not get.** The table never enters a graph. In STORAGE mode the graph has no
  `get_rows` and no I32 row input.
- **Why f32 on the host.** It keeps the result from depending on which backend the scheduler picks. A
  compact Q8_0 input run by SYCL `get_rows` would compute in half precision.
- **Scope.** This departs from parent D6-M (c) and DQ-C2 (b) for this table only.
- **The cost.** It is a host wait before `graph_compute`, so no device work is queued behind it.

**D2. Residency.** Owner ruling 2026-11-14: "storage-only flag for this table".
- `storage_only` is a per-table planner input. libllama sets it, for qwen4exp only.
- The planner skips the fits-in-RAM test for such a table.
- Every other table keeps HOST-if-fits.
- This departs from parent D1 and §4.3 for this table only.

**D3. The row cache.** Owner ruling 2026-11-14: "2 GiB, with hit and miss counters". The lead added R2 and R6.
- **The ceiling.** A storage-only table's `host_resident_bytes` is at most `min(2 GiB, its HOST body bytes)`,
  index included. The activation check counts the bytes the same way. A small storage-only table therefore
  never plans a cache larger than holding the table whole.
- **Where the counters come from.** Hits, misses and evictions come from `lazy_row_cache::counters()`. Only
  the count of waiters submitted is the gather's own.
- **When they print.** All four print when the model is freed.
- **The waiter count is not a disk count.** The count of waiters submitted is printed as `submitted=` and is
  not a disk-read count. Two rows in one block, or a joined read, share one disk read.

**D4. The file path.** The lead decided this after S1. The table is opened at `ml.file_paths[file_index]` as
given, through mergerfs. A file loaded from an open handle has an empty path, and the load refuses such a
STORAGE table with a named error.

**D5. Failure.**
- `set_input` never throws.
- A failed gather or conversion sets the input's `failed` flag. `process_ubatch` then returns
  `GGML_STATUS_FAILED` before `graph_compute`.
- A table that fails to open after the load fails the load.
- Saving a model with a STORAGE table aborts with a message naming the tensor. That is the saver's existing
  failure form.

**D6. The switch.**
- **The predicate.** The no-argument bridge becomes
  `static bool llama_lazy_storage_consumer_present(enum llm_arch arch)`. It returns false for every arch until
  T13. T13 makes it `arch == LLM_ARCH_QWEN4EXP`. That moment is when the default Qwen load changes.
- **After T13.** The predicate is a fact per arch, not a switch. gemma4 and DeepSeek have no consumer and keep
  the bridge and the ON refusal.
- **No new environment variable.**

**D7. Lifecycle.** Lead ruling R6. Opening and closing a table are mutating procs. Each takes
`sycl_module_mutation_guard`, as the lifecycle source contract expects
(`tests/test-sycl-lifecycle-source-contract.py:30-77`).

### S4: where each `lazy_table_desc` field comes from

The descriptor is declared at `lazy-table.hpp:46-75`, and `desc_invalid` (`lazy-table.cpp:103-157`) validates
it. The backend fills it in the T6 open proc from three sources:
- the `ggml_sycl_lazy_table_info` that libllama passes, which libllama builds from the record row and the
  weight's tensor;
- the published plan's entry for the table;
- the snapshot's own copy of the planner input, which is used to cross-check the other two.

| `desc` field | source | in libllama today? |
|---|---|---|
| `name` | `info.name`, the record row's `name` | yes |
| `identity` | left at its default. `lazy-table.cpp` never reads it | n/a |
| `model` | `llama_model::sycl_model_token` (`src/llama-model.h:833`). Valid after `sycl_model_loading_guard.finish(true)` (`src/llama-model.cpp:3372`) | yes, after finish |
| `type`, `ne0`, `n_rows`, `row_size` | `info.type`, `info.ne[0]`, `info.ne[1]`, `info.row_size`, from the weight tensor (`ml.weights_map[name].tensor`) | yes |
| `file_index`, `file_offset`, `table_bytes` | `info.file_index`, `info.file_offset`, `info.size`, from the row's `value` | yes |
| `file_path` | `ml.file_paths[file_index]` (`src/llama-model-loader.h:152`). Copied into the storage entry during the load | during the load only; **T10** keeps it |
| `residency` | `STORAGE`. The open refuses any other plan entry | plan |
| `consumer_layer` | `info.consumer_layer`, from the row's `attrs` | yes |
| `consumer_device`, `host_resident_bytes`, `cache_slots`, `cache_floor_slots` | the plan entry (`unified-cache.hpp:1097-1105`) | plan |
| `block_ring_need`, `block_ring_charge` | the plan (`unified-cache.hpp:1230-1236`) | plan |
| cross-check: `row_size`, `table_bytes`, `file_offset`, `type` | the snapshot's `kv_info.lazy_tables` entry of the same name (`unified-cache.hpp:2372-2389`) | plan |
| planner input `storage_only` | the row's `attrs.storage_only` | **no. T7 creates it** |
| gather input `file_id` | `fstat(table->storage_fd())`, its `st_dev` and `st_ino` | **no. T6 creates it** |

The plan comes from `lifecycle_find_placement_plan(model_id, load_txn_id)` (`unified-cache.hpp:2474`), as
`ggml_sycl_plan_lazy_tables_for_testing` reads it (`ggml-sycl.cpp:127590-127610`).

## Team topology

| track | tasks | files |
|---|---|---|
| A, planner and boundary | T1, T2, T3 | `unified-cache.{hpp,cpp}`, `ggml-sycl.h`, `ggml-sycl.cpp` (struct copy), two device tests |
| B, IO and backend API | T4, T5, T6 | `lazy-io-pool.{hpp,cpp}`, new `lazy-gather.{hpp,cpp}`, `ggml-sycl.{h,cpp}` (API), engine host test, plan-interface test, lifecycle contract |
| C, libllama load | T7, T8, T9, T10, T11 | the record, new `llama-lazy-storage.{h,cpp}`, `llama-model.{h,cpp}`, `llama-model-loader.{h,cpp}`, `llama-model-saver.cpp`, record tests, the shared pin script |
| D, graph | T12, T13 | `llama-graph.{h,cpp}`, `llama-context.cpp`, `models/qwen4exp.cpp`, T12's own pin script |
| lead | T14 | end-to-end validation |

**Merge order.** Tracks A and B share `ggml-sycl.h` and `ggml-sycl.cpp`, so T3 merges before T6. T12 adds one
hunk to `tests/CMakeLists.txt`, and so does T8. Merge T12 after T8. The hunks are disjoint; the order just
keeps the rebase trivial.

**The shared pin script.** It is written by T7, T8, T10, T11 and T13, in that order along the dependency chain.
T12 writes its own pin script.

**Concurrency.** No more than three implementers run at once, because each of them builds `libggml-sycl` or
`libllama`.

## Dependency graph

```
T1 ──► T2 ──► T3 ──► T6 ──────┐
T4 ──► T5 ─────────► T6       │
T7 ──► T8 (needs T3) ─────────┼──► T9 ──► T10 ──► T11 ──► T13 ──► T14 (lead)
        ┊                     │                            ▲
        ┊ merge order only    │                            │
        ▼                                                  │
T12 (merges after T8: tests/CMakeLists.txt) ───────────────┘
```

The dotted edge is a merge order, not a code dependency: T12 can be implemented at any time, and the lead holds
its merge until T8 is merged.

| after | the default Qwen load (AUTO) | `-lzm on` on Qwen |
|---|---|---|
| T1-T7 | unchanged: EAGER, mapped, placed by the plan, A11 figures | fails in `create_tensor`, as today |
| T8-T12 | unchanged | plans the table STORAGE, then fails in `create_tensor` |
| T13 | STORAGE: row cache of at most 2 GiB, rows gathered per ubatch | the same as AUTO |

## File ownership map

| file | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 | T11 | T12 | T13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ggml/src/ggml-sycl/unified-cache.hpp` | W | | | | | | | | | | | | |
| `ggml/src/ggml-sycl/unified-cache.cpp` | W | W | | | | | | | | | | | |
| `ggml/src/ggml-sycl/tests/test-sycl-lazy-residency-plan.cpp` | W | W | | | | | | | | | | | |
| `ggml/include/ggml-sycl.h` | | | W | | | W | | | | | | | |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | | | W | | | W | | | | | | | |
| `ggml/src/ggml-sycl/tests/test-sycl-lazy-plan-interface.cpp` | | | W | | | W | | | | | | | |
| `tests/test-sycl-lifecycle-source-contract.py` | | | | | | W | | | | | | | |
| `ggml/src/ggml-sycl/lazy-io-pool.{hpp,cpp}` | | | | W | | | | | | | | | |
| `ggml/src/ggml-sycl/lazy-gather.{hpp,cpp}` (new) | | | | | W | | | | | | | | |
| `tests/test-sycl-lazy-io-engine-host.cpp` | | | | W | W | | | | | | | | |
| `ggml/src/ggml-sycl/CMakeLists.txt` | | | | | W | W | | | | | | | |
| `src/llama-lazy-range-record.h` | | | | | | | W | W | | W | | | |
| `src/llama-lazy-range-record.cpp` | | | | | | | W | W | | W | | | W |
| `src/llama-model.cpp` | | | | | | | W | W | | W | | | |
| `src/llama-model-loader.cpp` | | | | | | | W | | | W | | | |
| `src/llama-model-loader.h` | | | | | | | | | | W | | | |
| `src/llama-model.h` | | | | | | | | | | W | | | |
| `src/llama-lazy-storage.{h,cpp}` (new) | | | | | | | | W | W | W | | | |
| `src/CMakeLists.txt` | | | | | | | | W | | | | | |
| `src/llama-model-saver.cpp` | | | | | | | | | | | W | | |
| `tests/test-llama-lazy-range-record.cpp` | | | | | | | W | W | | | | | |
| `tests/test-llama-lazy-range-prefill.cpp` | | | | | | | W | | | | | | W |
| `tests/test-llama-lazy-range-record-source.py` | | | | | | | W | W | | W | W | | W |
| `tests/test-llama-lazy-storage.cpp` (new) | | | | | | | | W | W | | | | |
| `tests/CMakeLists.txt` | | | | | | | | W | | | | W | |
| `src/llama-graph.{h,cpp}`, `src/llama-context.cpp` | | | | | | | | | | | | W | |
| `tests/test-llama-graph-input-failed.cpp` (new) | | | | | | | | | | | | W | |
| `tests/test-llama-graph-input-failed-source.py` (new) | | | | | | | | | | | | W | |
| `src/models/qwen4exp.cpp` | | | | | | | | | | | | | W |

The map was checked against every task's **Files** list below. Each shared file is written by tasks that are
ordered along one dependency chain. The one exception is `tests/CMakeLists.txt`, which T8 and T12 both write;
its merge order is stated under Team topology.

### A bounded wait in the new host arms (review S8)

The engine host test's mutant registrations each run the whole binary under a 120 s timeout, with a
`PASS_REGULAR_EXPRESSION` (`ggml/src/ggml-sycl/CMakeLists.txt:537-548`). A ticket wait or a gather that never
returns would turn a mutant's expected message into a timeout. So T4 and T5 run every waiting call through one
helper:

```cpp
// Runs `fn` on its own thread and waits at most `seconds`: a call that does not return fails its arm instead of hanging
// the binary, so a mutant registration still sees its own message. `fn` must own everything it touches (capture a
// shared_ptr by value): a call that never returns is detached and outlives the arm.
bool returns_within(std::function<int()> fn, int seconds, int & rc) {
    auto task = std::make_shared<std::packaged_task<int()>>(std::move(fn));
    std::future<int> done = task->get_future();
    std::thread([task] { (*task)(); }).detach();
    if (done.wait_for(std::chrono::seconds(seconds)) != std::future_status::ready) {
        return false;
    }
    rc = done.get();
    return true;
}
```

The implementer runs the new arms under every existing mutant registration:

```bash
ctest --test-dir build -R '^test-sycl-lazy-io-engine-host' --output-on-failure -j 1
```

A mutant whose expected message no longer appears is a defect in the new arm, not in the mutant.

---

## T1: a storage-only table skips the HOST test

**Track** A. **Depends** none.

**Done when** a `storage_only` table that fits the host room is planned STORAGE at its floor or above, and every
table without the flag is planned as before.

**Files**
- `ggml/src/ggml-sycl/unified-cache.hpp:514-524` (`placement_lazy_table_info`).
- `ggml/src/ggml-sycl/unified-cache.cpp:34849`, the HOST test.
- `ggml/src/ggml-sycl/tests/test-sycl-lazy-residency-plan.cpp`: a new arm (j), registered in the `arms` vector
  at `:670-680`.

**Step 1, census.**

```bash
grep -rn 'placement_lazy_table_info' ggml/src/ggml-sycl --include='*.cpp' --include='*.hpp'
```

If any site brace-initialises the struct positionally, STOP and message the lead.

**Step 2, RED. Device-registered; the lead runs it.** Add this after `arm_d_qwen_flip` (`:417`):

```cpp
// (j) A storage-only table is STORAGE at a reading where the same table without the flag is HOST.
// Engagement: the control plan (no flag, arm (d)'s "above" reading) carries the table HOST.
// Mutants: the flag not read by the HOST test (the flagged table is HOST).
// The cache's size is T2's arm (k); this arm checks residency and floor only.
arm_result arm_j_storage_only_skips_host() {
    arm_result                             r;
    const lazy_test_coverage_pin           pin    = ggml_sycl_test_make_coverage_pin(0);
    std::vector<placement_lazy_table_info> tables = qwen_tables();
    size_t                                 step2  = 0;
    if (!step2_of(r, qwen38_shape, pin.pin(), step2)) {
        return r;
    }
    const size_t reading = step2 + host_body(tables[0]) + k_p3_margin + 1 * GiB;
    expect_host(r, plan_for(qwen38_shape, kv_with(tables, reading), pin.pin()), tables[0], "control: ");

    tables[0].storage_only         = true;
    const placement_kv_info      kv = kv_with(tables, reading);
    const placement_plan         p  = plan_for(qwen38_shape, kv, pin.pin());
    const placement_lazy_table * t  = find_table(p, tables[0].name);
    expect(r, p.lazy_refusal.empty(), "the plan refused: " + p.lazy_refusal);
    expect(r, t != nullptr && t->residency == lazy_residency::STORAGE,
           std::string("the flagged table is ") + (t ? residency_name(t->residency) : "absent") + ", want STORAGE");
    expect(r, t != nullptr && t->cache_floor_slots == floor_slots(tables[0], kv) &&
                  t->cache_slots >= t->cache_floor_slots && t->host_resident_bytes >= floor_raw_bytes(tables[0], kv),
           "the flagged table is below its floor");
    expect_pin_held(r, pin);
    return r;
}
```

Register it as `{ "(j) a storage-only table skips the HOST test", arm_j_storage_only_skips_host },`.

It must first fail to compile, because the struct has no member `storage_only`.

Do not call `expect_storage` here. It asserts `host_resident_bytes < host_body` (`:189-191`), and at this
reading the uncapped share gives about 51.7 GiB against a body of about 50.7 GiB. Only T2's ceiling makes
that assertion true.

**Step 3, GREEN.** In `unified-cache.hpp`, after `table_bytes` (`:524`):

```cpp
    // libllama's answer for a table that is never held resident: the planner places it STORAGE whatever room the
    // host has, and caps its row cache.
    bool        storage_only      = false;
```

`unified-cache.cpp:34849` becomes:

```cpp
        // A storage-only table is read from its file whatever the host room: holding it costs more host memory than its
        // reads save, so it never takes the HOST branch.
        if (!info.storage_only && body != 0 && with <= fit_room && with <= p3_room) {
```

**Step 4.** The lead runs:

```bash
ctest --test-dir build -R '^test-sycl-lazy-residency-plan-1dev$' --output-on-failure
```

All arms must pass, (a)-(i) included.

**Gotchas**
- A storage-only table with no pin falls into the existing `else` and refuses with `lazy-table-ring-unpinned`.
- Default Qwen load: unchanged. libllama sets no flag yet.

**Commit:** `sycl: a storage-only lazy table skips the HOST test`.
Stage `ggml/src/ggml-sycl/unified-cache.hpp ggml/src/ggml-sycl/unified-cache.cpp ggml/src/ggml-sycl/tests/test-sycl-lazy-residency-plan.cpp`.

---

## T2: a storage-only table's row cache stops at min(2 GiB, its HOST body)

**Track** A. **Depends** T1.

**Done when** a storage-only STORAGE table's `host_resident_bytes` is at most `min(2 GiB, its HOST body
bytes)` at any reading, and a table without the flag gets the same share as before.

**Files**
- `ggml/src/ggml-sycl/unified-cache.cpp:34762-34768`, the constants.
- `ggml/src/ggml-sycl/unified-cache.cpp:34905-34920`, the share loop.
- The T1 test file: arm (j) gains one assertion, and arms (k) and (l) are new.

**Step 1, RED. Device-registered; the lead runs it.**

In arm (j), after the floor check, add:

```cpp
    expect_storage(r, p, tables[0], kv, "storage-only: ");  // its `< host_body` assertion needs the ceiling
```

Then add two arms:

```cpp
// (k) The big storage-only table's cache stops at 2 GiB, index included, where the same table without the flag gets
// tens of GiB from the same reading.
// Engagement: the control (no flag, arm (d)'s "below" reading) holds more than 2 GiB.
// Mutants: no ceiling (the flagged table holds the control's bytes); the ceiling on every table (the control holds
// 2 GiB); the ceiling ignoring the index (the flagged table exceeds 2 GiB).
arm_result arm_k_storage_only_cache_cap() {
    arm_result                             r;
    const lazy_test_coverage_pin           pin    = ggml_sycl_test_make_coverage_pin(0);
    std::vector<placement_lazy_table_info> tables = qwen_tables();
    size_t                                 step2  = 0;
    if (!step2_of(r, qwen38_shape, pin.pin(), step2)) {
        return r;
    }
    const size_t                 reading = step2 + host_body(tables[0]) + k_p3_margin - 1 * GiB;
    const placement_plan         plain   = plan_for(qwen38_shape, kv_with(tables, reading), pin.pin());
    const placement_lazy_table * c       = find_table(plain, tables[0].name);
    expect(r, c != nullptr && c->host_resident_bytes > 2 * GiB,
           "control: the unflagged table holds " + num(c ? c->host_resident_bytes : 0) + ", want more than 2 GiB");

    tables[0].storage_only         = true;
    const placement_kv_info      kv = kv_with(tables, reading);
    const placement_plan         p  = plan_for(qwen38_shape, kv, pin.pin());
    const placement_lazy_table * t  = find_table(p, tables[0].name);
    expect_storage(r, p, tables[0], kv, "capped: ");
    expect(r, t != nullptr && t->host_resident_bytes <= 2 * GiB && t->host_resident_bytes > 2 * GiB - 512 * MiB,
           "capped: the table holds " + num(t ? t->host_resident_bytes : 0) + ", want (1.5 GiB, 2 GiB]");
    expect_p3(r, p, step2, reading, "capped: ");
    expect_pin_held(r, pin);
    return r;
}

// (l) A small storage-only table never plans a cache larger than its HOST body: a 64 MiB table at a reading with
// room for every row holds at most its body, where a full cache of every row and its index would exceed it.
// Engagement: the full cache's bytes (every row, index included) exceed the body, so a ceiling of 2 GiB alone would
// let the cache pass the body.
// Mutants: the ceiling at 2 GiB without the body term (the table holds about 64 MiB of rows plus about 24 MiB of
// index, against a 64 MiB body: the helper puts the table at offset 0, so the body is the table on its mapping).
arm_result arm_l_small_storage_only_stops_at_its_body() {
    arm_result                   r;
    const lazy_test_coverage_pin pin = ggml_sycl_test_make_coverage_pin(0);
    placement_lazy_table_info    t   = lazy_table("blk.1.small.weight", GGML_TYPE_Q8_0, 32, (64 * MiB) / 34, 1, 1);
    t.storage_only                   = true;
    const std::vector<placement_lazy_table_info> tables = { t };
    size_t                                       step2  = 0;
    if (!step2_of(r, qwen38_shape, pin.pin(), step2)) {
        return r;
    }
    const size_t               reading = step2 + 64 * GiB + k_p3_margin;
    const placement_kv_info    kv      = kv_with(tables, reading);
    const placement_plan       p       = plan_for(qwen38_shape, kv, pin.pin());
    const placement_lazy_table * got   = find_table(p, t.name);
    expect(r, got != nullptr && got->residency == lazy_residency::STORAGE, "the small table is not STORAGE");
    expect(r, got != nullptr && got->host_resident_bytes <= host_body(t) &&
                  got->host_resident_bytes >= floor_raw_bytes(t, kv),
           "the small table holds " + num(got ? got->host_resident_bytes : 0) + ", want at most its body " +
               num(host_body(t)));
    expect_pin_held(r, pin);
    return r;
}
```

Register them as `(k) a storage-only cache stops at 2 GiB` and `(l) a small storage-only cache stops at its body`.

Both arms and the new (j) assertion must fail first, because the flagged table holds the control's bytes.

**Step 2, GREEN.** Add a constant beside `reserve_pct` (`:34768`):

```cpp
    constexpr size_t storage_only_cache_cap = 2ull << 30;  // a storage-only table's row cache, index included
```

At `:34914-34915`:

```cpp
            const size_t floor_res                  = t.host_resident_bytes;
            size_t       target                     = floor_res + share;
            if (info.storage_only) {
                // The owner's ceiling for a table that is never resident: 2 GiB, and never more than holding the table
                // whole (its HOST body) would cost. The floor still holds when it alone exceeds the ceiling.
                const size_t body =
                    host_cpu_mapped_size(lazy_table_host_body_request_bytes(info.file_offset, info.table_bytes));
                target = std::min(target, std::max(floor_res, std::min(storage_only_cache_cap, body)));
            }
            t.cache_slots = lazy_row_cache_slots_within(t.cache_floor_slots, rows, info.row_size, target);
```

**Gotchas**
- `given` accumulates what was actually given, so room a capped table leaves stays unused. Redistributing it is a
  follow-up.
- `expect_storage` (`:176`) checks `yieldable=` against the cache's own rule. That stays true, because yieldable
  derives from `cache_slots`.
- Arm (l)'s `lazy_table` helper (`:66-84`) sets `file_offset` 0, so `host_body` is the table rounded to its
  mapping.

**Commit:** `sycl: a storage-only lazy table's row cache stops at 2 GiB or its body`.
Stage `ggml/src/ggml-sycl/unified-cache.cpp ggml/src/ggml-sycl/tests/test-sycl-lazy-residency-plan.cpp`.

---

## T3: `ggml_sycl_lazy_table_info` carries `storage_only` across the backend boundary

**Track** A. **Depends** T2. Arm (g) asserts the ceiling.

**Done when** a table whose inventory info sets `storage_only` reaches the planner with the flag set, and the
struct is 88 bytes on both sides of the dlopen boundary.

**Files**
- `ggml/include/ggml-sycl.h:296-306`, the struct.
- `ggml/include/ggml-sycl.h:485`, the `== 80` assert, plus one new offset assert.
- `ggml/src/ggml-sycl/ggml-sycl.cpp:17266`, the mirrored assert.
- `ggml/src/ggml-sycl/ggml-sycl.cpp:17512`, the copy loop.
- `ggml/src/ggml-sycl/tests/test-sycl-lazy-plan-interface.cpp`, a new arm (g). The file already has arms (a)-(f): (f) is `arm_f_dry_run_refuses_a_count_with_no_array` (`:414`, registered `:477`).

**Step 1, census.**

```bash
grep -rn 'ggml_sycl_lazy_table_info' ggml src tests --include='*.cpp' --include='*.h' --include='*.hpp'
```

Every reader value-initialises the struct (`{}`). If a site aggregate-initialises it positionally, STOP and
message the lead.

**Step 2, RED. Device-registered; the lead runs it.** Add this after `arm_a_through_the_stage` (`:120-178`):

```cpp
// (g) storage_only crosses the boundary: arm (a)'s 64 MiB table, flagged, is STORAGE in the published plan, holding at
// most its 66 MiB body (the ceiling's body term), and its own [PLACEMENT] line says so.
// Engagement: arm (a) publishes the same table HOST, holding exactly the body; the log is cleared before the stage,
// and the line matched names this table.
// Mutants: the copy loop not copying storage_only (the table is HOST, its line says residency=HOST).
arm_result arm_g_storage_only_crosses_the_boundary() {
    arm_result                arr;
    loader_inventory          inv;
    ggml_sycl_lazy_table_info table = lazy_table("blk.1.engram_embd.weight", (64 * MiB) / 34);
    table.file_offset               = 4032;
    table.storage_only              = true;
    inv.inv.lazy_tables             = &table;
    inv.inv.lazy_table_count        = 1;
    const size_t body = host_cpu_mapped_size(lazy_table_host_body_request_bytes(table.file_offset, table.size));

    const ggml_sycl_placement_envelope envelope{ inv.inv.n_ctx, inv.inv.n_ubatch, 1, -1 };
    ggml_sycl_load_txn                 txn{};
    if (ggml_backend_sycl_model_load_begin(&txn) != GGML_SYCL_LIFECYCLE_OK) {
        void_arm(arr, "the load transaction did not begin");
        return arr;
    }
    log_clear();
    if (ggml_backend_sycl_stage_load_inventory_plan(&inv.inv, &envelope, true) != GGML_SYCL_LIFECYCLE_OK ||
        ggml_backend_sycl_stage_load_inventory_plan(&inv.inv, &envelope, false) != GGML_SYCL_LIFECYCLE_OK) {
        (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
        void_arm(arr, "a stage refused");
        return arr;
    }
    ggml_sycl_model_token model{};
    if (ggml_backend_sycl_model_load_end(txn, true, &model) != GGML_SYCL_LIFECYCLE_OK || model.model_id == 0) {
        void_arm(arr, "the load did not end with a model token");
        return arr;
    }
    const lazy_reading got = read_published(model);
    expect(arr, got.ok && got.count == 1, "the published plan carries " + num(got.count) + " tables, want 1");
    expect(arr, got.resident != 0 && got.resident <= body,
           "the table holds " + num(got.resident) + ", want a cache of at most its body " + num(body));
    expect(arr, !log_lines_containing("[PLACEMENT] lazy blk.1.engram_embd.weight residency=STORAGE").empty(),
           "this table's [PLACEMENT] lazy line does not say STORAGE");
    (void) ggml_backend_sycl_model_unloaded_token(model);
    return arr;
}
```

Register it after (f) (`:477`):
`{ "(g) storage_only crosses the boundary",                   arm_g_storage_only_crosses_the_boundary     },`.

It must first fail to compile, because the struct has no member `storage_only`.

**Step 3, GREEN.**

In `ggml-sycl.h`, after `size` (`:305`):

```c
    // Set by libllama for a table it never wants resident: the planner places it STORAGE whatever the host room and
    // caps its row cache (unified-cache.cpp, placement_plan_decide_lazy_residency).
    bool           storage_only;
```

- Change the assert at `:485` to `== 88`.
- Add `static_assert(offsetof(ggml_sycl_lazy_table_info, storage_only) == 80, "lazy table storage_only moved");`.
- Mirror both edits at `ggml-sycl.cpp:17266`.

In `ggml-sycl.cpp`, after `t.table_bytes = src.size;` (`:17512`):

```cpp
        t.storage_only   = src.storage_only;
```

**Gotchas**
- `bool` in this C struct is safe. `ggml_sycl_tensor_info` already carries `bool get_rows_only`.
- Default load: unchanged. Nothing sets the field until T8.

**Commit:** `sycl: the lazy table inventory info carries storage_only`.
Stage `ggml/include/ggml-sycl.h ggml/src/ggml-sycl/ggml-sycl.cpp ggml/src/ggml-sycl/tests/test-sycl-lazy-plan-interface.cpp`.

---

## T4: a STORAGE ticket can be waited on

**Track** B. **Depends** none.

**Done when** `lazy_io_ticket_wait` returns once every submitted waiter is delivered, with the ticket's first
errno, and no delivery touches the ticket after the waiter has returned.

**Files**
- `ggml/src/ggml-sycl/lazy-io-pool.hpp:321-325`, the ticket.
- `ggml/src/ggml-sycl/lazy-io-pool.hpp:26-32`, the includes.
- `ggml/src/ggml-sycl/lazy-io-pool.cpp:1450`, the final decrement.
- `tests/test-sycl-lazy-io-engine-host.cpp`:
  - add `<functional>` (`std::function`) and `<future>` (`std::packaged_task`, `std::future`) to the standard
    includes at `:33-46`, which `returns_within` needs;
  - add the `returns_within` helper (above);
  - add a new arm before `arm_ensure_shut_down`, which must stay last (`:1347-1348`).

**Step 1, RED. Host-only.**

```cpp
// A ticket's wait returns once its three reads are delivered, every row landed and no error; a ticket with nothing
// submitted returns at once; a failed read returns its errno. Every wait is bounded.
arm_result arm_ticket_wait() {
    arm_result r;
    ring_lease lease = lazy_io_pool_take();
    if (block_ring_ensure(k_need, k_span, 0, mapped(k_need), k_txn) != lazy_ring_ensure_status::OK) {
        expect(r, false, "the ring was not ensured");
        settle(lease);
        return r;
    }
    ggml_sycl_test_lazy_io_read_source(fill_source, nullptr);
    int rc = -1;
    {
        auto idle = std::make_shared<request_fixture>(1);
        expect(r, returns_within([idle] { return lazy_io_ticket_wait(idle->ticket); }, 5, rc) && rc == 0,
               "an empty ticket's wait did not return 0 within 5 s");
    }
    auto   f     = std::make_shared<request_fixture>(2);
    size_t added = 0;
    for (uint64_t b = 0; b < 3; ++b) {
        lazy_io_waiter w = f->waiter(0, 100);
        w.landing_offset = b * 100;
        added += lazy_io_submit(1, -1, block_span(10 + b), key0(2, 0), w);
    }
    expect(r, added == 3 && returns_within([f] { return lazy_io_ticket_wait(f->ticket); }, 5, rc) && rc == 0,
           "added " + num(added) + ", the wait returned " + num(rc) + " or did not return within 5 s");
    for (uint64_t b = 0; b < 3; ++b) {
        expect(r, f->slab[b * 100] == file_byte((10 + b) * LAZY_IO_BLOCK_BYTES), "row " + num(b) + " did not land");
    }
    static uint64_t bad = 20 * LAZY_IO_BLOCK_BYTES;
    auto            e   = std::make_shared<request_fixture>(3);
    ggml_sycl_test_lazy_io_read_source(eio_source, &bad);
    (void) lazy_io_submit(1, -1, block_span(20), key0(3, 0), e->waiter(0, 100));
    expect(r, returns_within([e] { return lazy_io_ticket_wait(e->ticket); }, 5, rc) && rc == EIO,
           "a failed read's wait did not return EIO within 5 s");
    ggml_sycl_test_lazy_io_read_source(nullptr, nullptr);
    expect(r, settle(lease) && live_charged() == 0, "the reap left " + num(live_charged()) + " B charged, want 0");
    return r;
}
```

Register it as `{ "a ticket's wait returns when its reads are delivered", arm_ticket_wait },`.

It must first fail to compile, because `lazy_io_ticket_wait` is not declared.

**Step 2, GREEN.** In `lazy-io-pool.hpp`, add `#include <condition_variable>`. The ticket becomes:

```cpp
struct lazy_io_ticket {
    lazy_gather_request *   request = nullptr;
    std::atomic<int>        outstanding{ 0 };
    std::atomic<int>        read_error{ 0 };
    // Every decrement happens under done_lock and the one that reaches 0 notifies, so a waiter that reads 0 under the
    // same lock knows no delivery will touch the ticket again and may destroy it. A leaf: nothing is taken under it.
    std::mutex              done_lock;
    std::condition_variable done;
};

// Blocks until every waiter submitted on `ticket` has been delivered and returns its read_error: 0 when every row
// landed. The caller submits all its waiters first; the count may touch 0 between two submits, which only wakes it
// early to wait again.
int lazy_io_ticket_wait(lazy_io_ticket & ticket);
```

In `lazy-io-pool.cpp`, replace `:1450`:

```cpp
    std::lock_guard<std::mutex> lock(ticket->done_lock);
    const bool                  last = ticket->outstanding.fetch_sub(1, std::memory_order_acq_rel) == 1;
    if (last) {
        ticket->done.notify_all();
    }
    return last;
}

int lazy_io_ticket_wait(lazy_io_ticket & ticket) {
    std::unique_lock<std::mutex> lock(ticket.done_lock);
    ticket.done.wait(lock, [&] { return ticket.outstanding.load(std::memory_order_acquire) == 0; });
    return ticket.read_error.load(std::memory_order_acquire);
}
```

**Step 3.**

```bash
ctest --test-dir build -R '^test-sycl-lazy-io-engine-host' --output-on-failure -j 1
```

This runs every mutant registration. Each must still match its own `PASS_REGULAR_EXPRESSION`.

**Gotchas**
- Use C++17 only. Do not use `std::atomic::wait`.
- Notify while holding the lock. If the notify comes after the unlock, the waiter could destroy a ticket that
  lives on its stack while the deliverer is still inside `notify_all`.
- Delivery never runs under the engine mutex (worker `:1598`, shutdown `:1898`), and `done_lock` nests under
  nothing. The witness records only `witnessed_mutex` instances (`witnessed-mutex.hpp:3-14`), so a plain
  `std::mutex` leaf does not register with it (Q6).
- `lazy_io_deliver` is `noexcept`. A failed lock there terminates, exactly as the landing-slot lock already
  would.

**Commit:** `sycl: a lazy IO ticket can be waited on`.
Stage `ggml/src/ggml-sycl/lazy-io-pool.hpp ggml/src/ggml-sycl/lazy-io-pool.cpp tests/test-sycl-lazy-io-engine-host.cpp`.

---

## T5: `lazy_gather_rows` copies a ubatch's rows from the cache and the file

**Track** B. **Depends** T4.

**Done when** a gather returns every requested row byte for byte, serves hits from the cache, reads misses and
waits for them, and returns EIO or EINVAL without leaving a waiter that points at its stack.

**Files**
- New `ggml/src/ggml-sycl/lazy-gather.hpp` and `lazy-gather.cpp`. They include no SYCL header. The glob at
  `ggml/src/ggml-sycl/CMakeLists.txt:86` puts them in `libggml-sycl`.
- `ggml/src/ggml-sycl/CMakeLists.txt:526-530`: add `lazy-gather.cpp` to the engine host test's sources.
- `tests/test-sycl-lazy-io-engine-host.cpp`:
  - a new arm, before `arm_ensure_shut_down`;
  - `#include "../ggml/src/ggml-sycl/lazy-gather.hpp"` and `"../ggml/src/ggml-sycl/lazy-row-cache.hpp"` beside the
    existing project includes at `:27-31` (skip the second if a fixture header already brings it in);
  - a fixture memory source copied from `tests/test-sycl-lazy-io-queue-host.cpp:473-480` (`plain_memory` and
    `plain_source`).

**Step 1, RED. Host-only.** The cache, the source and the destination live in one shared state, so a call
that never returns cannot outlive the memory it writes:

```cpp
// A gather of rows 3, 40000, 3 lands each row byte-exact from the file; a second gather of row 3 hits the cache and
// submits nothing; a row past the table is EINVAL and submits nothing; a gather whose read fails returns EIO after its
// other reads are delivered. Every gather is bounded.
arm_result arm_gather_rows() {
    arm_result r;
    ring_lease lease = lazy_io_pool_take();
    if (block_ring_ensure(k_need, k_span, 0, mapped(k_need), k_txn) != lazy_ring_ensure_status::OK) {
        expect(r, false, "the ring was not ensured");
        settle(lease);
        return r;
    }
    ggml_sycl_test_lazy_io_read_source(fill_source, nullptr);
    struct state {
        plain_source         mem;
        lazy_row_cache_build cache;
        lazy_gather_source   src;
        std::vector<uint8_t> dst = std::vector<uint8_t>(3 * 170, 0);
        lazy_gather_stats    s;
    };
    auto st = std::make_shared<state>();
    lazy_row_cache_config cfg;
    cfg.name         = "fixture.gather";
    cfg.row_size     = 170;
    cfg.n_rows       = 100000;
    cfg.cache_slots  = 64;
    cfg.floor_slots  = 64;
    cfg.n_shards     = 1;
    st->cache        = lazy_row_cache::build(cfg, st->mem);
    st->src.fd          = -1;
    st->src.file_id     = k_file;
    st->src.file_offset = 4032;
    st->src.row_size    = 170;
    st->src.n_rows      = 100000;
    const auto gather = [&r](std::shared_ptr<state> s, std::vector<int32_t> rows, uint32_t epoch, int & rc) {
        const bool done = returns_within(
            [s, rows, epoch] {
                return lazy_gather_rows(s->cache.cache.get(), s->src, 1, epoch, 0, rows.data(), rows.size(),
                                        s->dst.data(), s->dst.size(), &s->s);
            },
            10, rc);
        expect(r, done, "a gather did not return within 10 s");
        return done;
    };

    int rc = -1;
    if (gather(st, { 3, 40000, 3 }, 1, rc)) {
        const int32_t rows[3] = { 3, 40000, 3 };
        bool          exact   = rc == 0;
        for (size_t i = 0; i < 3 && exact; ++i) {
            for (size_t j = 0; j < 170; ++j) {
                exact = exact && st->dst[i * 170 + j] == file_byte(4032 + (uint64_t) rows[i] * 170 + j);
            }
        }
        expect(r, exact, "the gather returned " + num(rc) + " or a row is not the file's bytes");
        expect(r, st->s.rows == 3 && st->s.reads >= 2, "rows " + num(st->s.rows) + ", reads " + num(st->s.reads));
    }
    if (gather(st, { 3 }, 2, rc)) {
        expect(r, rc == 0 && st->s.hits == 1 && st->s.reads == 0,
               "the repeated row did not hit: hits " + num(st->s.hits) + ", reads " + num(st->s.reads));
    }
    if (gather(st, { 100000 }, 3, rc)) {
        expect(r, rc == EINVAL && st->s.reads == 0, "a row past the table did not return EINVAL unsubmitted");
    }
    static uint64_t bad = (4032 + 50000ull * 170) / LAZY_IO_BLOCK_BYTES * LAZY_IO_BLOCK_BYTES;
    ggml_sycl_test_lazy_io_read_source(eio_source, &bad);
    if (gather(st, { 50000, 7 }, 4, rc)) {
        expect(r, rc == EIO, "a failed read returned " + num(rc) + ", want EIO");
    }
    ggml_sycl_test_lazy_io_read_source(nullptr, nullptr);
    st->cache.cache.reset();
    expect(r, settle(lease) && live_charged() == 0, "the reap left " + num(live_charged()) + " B charged, want 0");
    return r;
}
```

It must first fail to compile, because `lazy-gather.hpp` does not exist.

**Step 2, GREEN.** `lazy-gather.hpp`:

```cpp
#pragma once

// The STORAGE table's ubatch gather (owner ruling 2026-11-14, gather in set_input): the rows a ubatch reads, copied from
// the row cache on a hit and read from the file through the IO engine on a miss, waited for before it returns.
// SYCL-free like lazy-io-pool.hpp, so the engine's host test links it.

#include <cstddef>
#include <cstdint>

namespace ggml_sycl {

class lazy_row_cache;

struct lazy_gather_source {
    int      fd          = -1;  // the table's O_DIRECT descriptor
    uint64_t file_id     = 0;   // the file's identity from its device and inode; the in-flight table dedups on it
    uint64_t file_offset = 0;   // the table's first byte in the file
    size_t   row_size    = 0;
    int64_t  n_rows      = 0;
};

struct lazy_gather_stats {
    uint64_t rows  = 0;  // rows looked up
    uint64_t hits  = 0;  // rows the cache served
    uint64_t reads = 0;  // waiters submitted for misses: not disk reads, since rows of one block share a read
};

// Copies row rows[i] of the table to dst + i * row_size for i in [0, n). `context` names the caller's queue, `epoch`
// this call (distinct from the caller's previous call), `deadline` the consumer layer. Returns 0, EINVAL for a bad
// row or a short dst, or the first failed read's errno. Never returns with a read in flight on its stack.
int lazy_gather_rows(lazy_row_cache *           cache,
                     const lazy_gather_source & src,
                     uint64_t                   context,
                     uint32_t                   epoch,
                     int32_t                    deadline,
                     const int32_t *            rows,
                     size_t                     n,
                     uint8_t *                  dst,
                     size_t                     dst_bytes,
                     lazy_gather_stats *        stats) noexcept;

}  // namespace ggml_sycl
```

`lazy-gather.cpp`:

```cpp
#include "lazy-gather.hpp"

#include "lazy-io-pool.hpp"
#include "lazy-row-cache.hpp"

#include <cerrno>

namespace ggml_sycl {

int lazy_gather_rows(lazy_row_cache *           cache,
                     const lazy_gather_source & src,
                     uint64_t                   context,
                     uint32_t                   epoch,
                     int32_t                    deadline,
                     const int32_t *            rows,
                     size_t                     n,
                     uint8_t *                  dst,
                     size_t                     dst_bytes,
                     lazy_gather_stats *        stats) noexcept {
    lazy_gather_stats   local;
    lazy_gather_stats & st = stats != nullptr ? *stats : local;
    st                     = {};
    if (n == 0) {
        return 0;
    }
    if (rows == nullptr || dst == nullptr || src.row_size == 0 || dst_bytes / src.row_size < n) {
        return EINVAL;
    }
    // The request, ticket and landing slot live on this frame: the wait below is what makes that safe.
    lazy_gather_request req;
    lazy_io_ticket      ticket;
    lazy_landing_slot   slot;
    req.epoch      = epoch;
    req.ticket     = &ticket;
    ticket.request = &req;
    slot.epoch     = epoch;
    slot.data      = dst;
    slot.bytes     = n * src.row_size;

    const size_t span_bytes = lazy_block_ring_span(src.row_size);
    int          stopped    = 0;  // why submitting stopped early; the reads already queued are still waited for
    for (size_t i = 0; i < n && stopped == 0; ++i) {
        const int64_t row = rows[i];
        if (row < 0 || row >= src.n_rows) {
            stopped = EINVAL;
            break;
        }
        ++st.rows;
        uint8_t * const out = dst + i * src.row_size;
        if (cache != nullptr && cache->lookup_copy(row, out)) {
            ++st.hits;
            continue;
        }
        const uint64_t byte  = src.file_offset + static_cast<uint64_t>(row) * src.row_size;
        const uint64_t start = byte / LAZY_IO_BLOCK_BYTES * LAZY_IO_BLOCK_BYTES;
        lazy_io_waiter w;
        w.ticket         = &ticket;
        w.epoch          = epoch;
        w.landing        = &slot;
        w.landing_offset = i * src.row_size;
        w.src_offset     = byte - start;
        w.bytes          = src.row_size;
        w.cache          = cache;
        w.row            = row;
        const lazy_io_key key{ lazy_io_class::PREFETCH, epoch, deadline, lazy_io_kind::TABLE_ROWS };
        try {
            if (lazy_io_submit(context, src.fd, lazy_io_span{ src.file_id, start, span_bytes }, key, w) == 0) {
                stopped = EIO;  // no live ring, or a span the ring cannot hold; submit has logged which
            } else {
                ++st.reads;
            }
        } catch (...) {
            stopped = ENOMEM;
        }
    }
    const int read_error = lazy_io_ticket_wait(ticket);
    return stopped != 0 ? stopped : read_error;
}

}  // namespace ggml_sycl
```

**Step 3.** Run the same `ctest` line as in T4. All mutant registrations must still match.

**Gotchas**
- Every path that submitted anything must wait, the error paths included. The early `return`s come before
  the first submit.
- Each lookup counts as a cache hit or a cache miss in `lazy_row_cache::counters()`. T6 prints those counters.
  `lazy_gather_stats` is this function's own view. Only its `reads` is printed, as `submitted=`.
- The insert happens at delivery, before the decrement (`lazy-io-pool.cpp:1442-1444`). A repeated row inside
  one call may therefore miss twice. Do not assert a hit within a single call.

**Commit:** `sycl: gather a lazy table's ubatch rows from the row cache and the file`.
Stage `ggml/src/ggml-sycl/lazy-gather.hpp ggml/src/ggml-sycl/lazy-gather.cpp ggml/src/ggml-sycl/CMakeLists.txt tests/test-sycl-lazy-io-engine-host.cpp`.

---

## T6: the backend opens, gathers from, counts and closes a STORAGE table by model token

**Track** B. **Depends** T3, T5.

**Done when** a fixture load whose plan carries one storage-only table can do all of the following through
the proc table:
- open the table by its model token, under the module mutation guard;
- gather rows byte-exact from a real file;
- read the cache's counters;
- close the table.

The open also refuses a geometry that disagrees with the plan.

**Files**
- `ggml/include/ggml-sycl.h`: the declarations, beside `ggml_sycl_plan_lazy_tables_for_testing` (`:2276`).
- `ggml/src/ggml-sycl/ggml-sycl.cpp`:
  - the definitions, after `:127610`;
  - the proc table entries beside the existing `strcmp(name, ...)` entries (about `:128204`).
- `ggml/src/ggml-sycl/tests/test-sycl-lazy-plan-interface.cpp`: a new arm (h), registered after T3's (g), and
  three includes at `:20-32`: `"lazy-io-test-util.hpp"`, `<cerrno>` and `<fstream>`.
- `ggml/src/ggml-sycl/CMakeLists.txt:8636-8650`: one compile definition.
- `tests/test-sycl-lifecycle-source-contract.py:30-70`: two entries in `saveable_mutating_procs`.

**Step 1, RED.**

(a) **Host-only pin.** Add these two entries to `saveable_mutating_procs`:

```python
    "ggml_backend_sycl_lazy_storage_open": "ggml_backend_sycl_lazy_storage_open",
    "ggml_backend_sycl_lazy_storage_close": "ggml_backend_sycl_lazy_storage_close",
```

The contract (`:71-74`) then requires each proc name in the proc table, and `sycl_module_mutation_guard`
within 1500 characters of the function's name. It must fail first.

(b) **Device-registered; the lead runs it.** Register the fixture directory:

```cmake
    target_compile_definitions(test-sycl-lazy-plan-interface PRIVATE
        LAZY_STORAGE_FIXTURE_DIR="${CMAKE_BINARY_DIR}")
```

Use the build directory, not `/tmp`. O_DIRECT fails on tmpfs.

The test's include block (`:20-32`) gains `"lazy-io-test-util.hpp"` (for `file_byte`, `lazy-io-test-util.hpp:68`;
today the test includes only `lazy-table-test-util.hpp`), `<cerrno>` (`EINVAL`) and `<fstream>` (for
`write_fixture_file`). The arm:

```cpp
// (h) The storage API: a 4 MiB storage-only table written to a file in the build directory opens by the published
// model token, gathers four rows byte-exact, hits on a repeated row, refuses a row past the table, and closes before
// the unload. An info whose row_size disagrees with the plan's is refused, and an open after the unload finds no plan.
// Engagement: arm (g) shows the same flag publishing STORAGE; the first gather misses (the cache is cold), so rows
// come through O_DIRECT reads.
// Mutants: the open reading the candidate instead of the published plan (no entry at model_id != 0); file_offset
// ignored (every row off by 4032 B); the geometry cross-check missing (the bad info opens).
arm_result arm_h_storage_api() {
    arm_result        arr;
    const std::string path = std::string(LAZY_STORAGE_FIXTURE_DIR) + "/lazy-storage-fixture.bin";
    const size_t      off  = 4032;
    const int64_t     rows = (4 * MiB) / 34;
    if (!write_fixture_file(path, off + rows * 34 + 4096)) {  // byte i is file_byte(i), lazy-io-test-util.hpp:68
        void_arm(arr, "the fixture file was not written");
        return arr;
    }
    loader_inventory          inv;
    ggml_sycl_lazy_table_info table = lazy_table("per_layer_token_embd.weight", rows);
    table.file_offset               = off;
    table.storage_only              = true;
    inv.inv.lazy_tables             = &table;
    inv.inv.lazy_table_count        = 1;
    const ggml_sycl_placement_envelope envelope{ inv.inv.n_ctx, inv.inv.n_ubatch, 1, -1 };
    ggml_sycl_load_txn                 txn{};
    ggml_sycl_model_token              model{};
    if (ggml_backend_sycl_model_load_begin(&txn) != GGML_SYCL_LIFECYCLE_OK) {
        void_arm(arr, "the load transaction did not begin");
        return arr;
    }
    if (ggml_backend_sycl_stage_load_inventory_plan(&inv.inv, &envelope, true) != GGML_SYCL_LIFECYCLE_OK ||
        ggml_backend_sycl_stage_load_inventory_plan(&inv.inv, &envelope, false) != GGML_SYCL_LIFECYCLE_OK ||
        ggml_backend_sycl_model_load_end(txn, true, &model) != GGML_SYCL_LIFECYCLE_OK) {
        (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
        void_arm(arr, "the fixture load did not publish");
        return arr;
    }
    ggml_sycl_lazy_table_info wrong = table;
    wrong.row_size                  = table.row_size + 2;
    expect(arr, ggml_backend_sycl_lazy_storage_open(model, &wrong, path.c_str()) == nullptr,
           "an info whose row_size disagrees with the plan opened");

    ggml_sycl_lazy_storage * s = ggml_backend_sycl_lazy_storage_open(model, &table, path.c_str());
    expect(arr, s != nullptr, "the open returned null");
    if (s != nullptr) {
        const int32_t        want[4] = { 0, 7, 7, (int32_t) rows - 1 };
        std::vector<uint8_t> dst(4 * 34);
        const int            rc = ggml_backend_sycl_lazy_storage_gather(s, want, 4, dst.data(), dst.size());
        bool                 ok = rc == 0;
        for (size_t i = 0; i < 4 && ok; ++i) {
            for (size_t j = 0; j < 34; ++j) {
                ok = ok && dst[i * 34 + j] == file_byte(off + (uint64_t) want[i] * 34 + j);
            }
        }
        expect(arr, ok, "the gather returned " + num(rc) + " or a row is not the file's bytes");
        uint64_t c[4] = {};  // hits, misses, evictions, submitted
        ggml_backend_sycl_lazy_storage_counters(s, c);
        expect(arr, c[0] + c[1] == 4 && c[1] >= 3 && c[3] >= 3 && c[3] <= c[1],
               "first gather: hits " + num(c[0]) + " misses " + num(c[1]) + " submitted " + num(c[3]));
        const int32_t again[1] = { 7 };
        (void) ggml_backend_sycl_lazy_storage_gather(s, again, 1, dst.data(), 34);
        uint64_t d[4] = {};
        ggml_backend_sycl_lazy_storage_counters(s, d);
        expect(arr, d[0] == c[0] + 1 && d[3] == c[3], "the repeated row did not hit");
        const int32_t past[1] = { (int32_t) rows };
        expect(arr, ggml_backend_sycl_lazy_storage_gather(s, past, 1, dst.data(), 34) == EINVAL, "no EINVAL");
        ggml_backend_sycl_lazy_storage_close(s);
    }
    (void) ggml_backend_sycl_model_unloaded_token(model);
    expect(arr, ggml_backend_sycl_lazy_storage_open(model, &table, path.c_str()) == nullptr,
           "an open after the unload found a plan");
    std::remove(path.c_str());
    return arr;
}
```

Register it after (g):
`{ "(h) the storage API opens, gathers, counts and closes",   arm_h_storage_api                           },`.

`write_fixture_file` is a short `std::ofstream` helper in the test file. It writes `file_byte(i)` for each
byte. Both (a) and (b) must fail first: (a) on its assertion, and (b) at compile time, because the four
functions are not declared.

**Step 2, GREEN.** In `ggml-sycl.h`:

```c
// A STORAGE lazy table opened for a published model (the storage tier's consumer side): the table's O_DIRECT
// descriptor, its row cache and the gather's state. Opaque.
struct ggml_sycl_lazy_storage;

// Opens the table `info` names for `model`, whose published plan must carry it STORAGE with the same row size, table
// bytes, file offset and type, reading `file_path`. NULL, after a WARN naming the table and the reason, when the
// module refuses mutations, no published plan exists, the plan has no such table or holds it HOST, the geometry
// disagrees, or the materialization refuses. Proc name: "ggml_backend_sycl_lazy_storage_open".
GGML_BACKEND_API struct ggml_sycl_lazy_storage * ggml_backend_sycl_lazy_storage_open(
    struct ggml_sycl_model_token model, const struct ggml_sycl_lazy_table_info * info, const char * file_path);

// Copies rows[i] of the table to dst + i * row_size and returns 0, EINVAL, or the first failed read's errno. Calls on
// one table are serialised. Proc name: "ggml_backend_sycl_lazy_storage_gather".
GGML_BACKEND_API int ggml_backend_sycl_lazy_storage_gather(struct ggml_sycl_lazy_storage * storage,
                                                           const int32_t * rows, size_t n, void * dst,
                                                           size_t dst_bytes);

// out[0] hits, out[1] misses, out[2] evictions (the row cache's own counters), out[3] waiters submitted for misses
// (not disk reads: rows of one block, and joined reads, share one). Proc name: "ggml_backend_sycl_lazy_storage_counters".
GGML_BACKEND_API void ggml_backend_sycl_lazy_storage_counters(const struct ggml_sycl_lazy_storage * storage,
                                                              uint64_t out[4]);

// Releases the table. No gather may be running. Proc name: "ggml_backend_sycl_lazy_storage_close".
GGML_BACKEND_API void ggml_backend_sycl_lazy_storage_close(struct ggml_sycl_lazy_storage * storage);
```

In `ggml-sycl.cpp`, include `lazy-gather.hpp`, `lazy-table.hpp` and `<sys/stat.h>`:

```cpp
struct ggml_sycl_lazy_storage {
    ggml_sycl::lazy_table_ref     table;
    ggml_sycl::lazy_gather_source source;
    uint64_t                      context  = 0;
    int32_t                       deadline = 0;
    std::mutex                    gather_lock;  // one gather at a time: the landing epoch is the call's
    uint32_t                      epoch = 0;
    std::atomic<uint64_t>         submitted{ 0 };
};

ggml_sycl_lazy_storage * ggml_backend_sycl_lazy_storage_open(ggml_sycl_model_token             model,
                                                             const ggml_sycl_lazy_table_info * info,
                                                             const char *                      file_path) {
    sycl_module_mutation_guard module_guard;
    if (!module_guard) {
        GGML_LOG_WARN("[LAZY-STORAGE] open refused: the backend module is not admitting mutations\n");
        return nullptr;
    }
    if (info == nullptr || info->name == nullptr || file_path == nullptr || model.model_id == 0) {
        GGML_LOG_WARN("[LAZY-STORAGE] open refused: a null argument or model id 0\n");
        return nullptr;
    }
    try {
        const auto snapshot = ggml_sycl::lifecycle_find_placement_plan(model.model_id, model.load_txn_id);
        if (!snapshot || !snapshot->plan) {
            GGML_LOG_WARN("[LAZY-STORAGE] %s: open refused: no published plan for model %llu\n", info->name,
                          (unsigned long long) model.model_id);
            return nullptr;
        }
        const ggml_sycl::placement_plan &       plan  = *snapshot->plan;
        const ggml_sycl::placement_lazy_table * entry = nullptr;
        for (const ggml_sycl::placement_lazy_table & t : plan.lazy_tables) {
            if (t.name == info->name) {
                entry = &t;
            }
        }
        const ggml_sycl::placement_lazy_table_info * planned = nullptr;
        for (const ggml_sycl::placement_lazy_table_info & t : snapshot->kv_info.lazy_tables) {
            if (t.name == info->name) {
                planned = &t;
            }
        }
        if (entry == nullptr || planned == nullptr || entry->residency != ggml_sycl::lazy_residency::STORAGE) {
            GGML_LOG_WARN("[LAZY-STORAGE] %s: open refused: the plan holds it %s\n", info->name,
                          entry == nullptr ? "nowhere" : "HOST");
            return nullptr;
        }
        // The table the caller names must be the one the plan sized: a row size or extent that differs would read
        // the wrong bytes into a cache planned for other ones.
        if (planned->row_size != info->row_size || entry->table_bytes != info->size ||
            planned->file_offset != info->file_offset || planned->type != info->type) {
            GGML_LOG_WARN("[LAZY-STORAGE] %s: open refused: row_size %zu/%zu, bytes %zu/%zu, offset %zu/%zu (caller/plan)\n",
                          info->name, info->row_size, planned->row_size, info->size, entry->table_bytes,
                          info->file_offset, planned->file_offset);
            return nullptr;
        }
        ggml_sycl::lazy_table_desc desc;
        desc.name                = info->name;
        desc.model               = model;
        desc.type                = info->type;
        desc.ne0                 = info->ne[0];
        desc.n_rows              = info->ne[1];
        desc.row_size            = info->row_size;
        desc.file_index          = info->file_index;
        desc.file_offset         = info->file_offset;
        desc.table_bytes         = info->size;
        desc.file_path           = file_path;
        desc.residency           = ggml_sycl::lazy_residency::STORAGE;
        desc.consumer_layer      = info->consumer_layer;
        desc.consumer_device     = entry->consumer_device;
        desc.host_resident_bytes = entry->host_resident_bytes;
        desc.cache_slots         = entry->cache_slots;
        desc.cache_floor_slots   = entry->cache_floor_slots;
        desc.block_ring_need     = plan.block_ring_need;
        desc.block_ring_charge   = plan.block_ring_charge;
        ggml_sycl::lazy_table_result made = ggml_sycl::lazy_table_create_storage(desc);
        if (!made) {
            return nullptr;  // the materialization logged its refusal with the figures
        }
        struct stat st = {};
        if (fstat(made.table->storage_fd(), &st) != 0) {
            GGML_LOG_WARN("[LAZY-STORAGE] %s: open refused: fstat failed (errno %d)\n", info->name, errno);
            return nullptr;
        }
        auto * s              = new ggml_sycl_lazy_storage;
        s->table              = std::move(made.table);
        s->source.fd          = s->table->storage_fd();
        // device and inode, so two descriptors on one file share reads in the engine's in-flight table
        s->source.file_id     = (static_cast<uint64_t>(st.st_dev) << 40) ^ static_cast<uint64_t>(st.st_ino);
        s->source.file_offset = info->file_offset;
        s->source.row_size    = info->row_size;
        s->source.n_rows      = info->ne[1];
        s->context            = reinterpret_cast<uintptr_t>(s);
        s->deadline           = info->consumer_layer;
        return s;
    } catch (const std::exception & e) {
        GGML_LOG_WARN("[LAZY-STORAGE] %s: open refused: %s\n", info->name, e.what());
        return nullptr;
    }
}

int ggml_backend_sycl_lazy_storage_gather(ggml_sycl_lazy_storage * s, const int32_t * rows, size_t n, void * dst,
                                          size_t dst_bytes) {
    if (s == nullptr) {
        return EINVAL;
    }
    std::lock_guard<std::mutex>  lock(s->gather_lock);
    ggml_sycl::lazy_gather_stats st;
    const int rc = ggml_sycl::lazy_gather_rows(s->table->row_cache(), s->source, s->context, ++s->epoch, s->deadline,
                                               rows, n, static_cast<uint8_t *>(dst), dst_bytes, &st);
    s->submitted += st.reads;
    return rc;
}

void ggml_backend_sycl_lazy_storage_counters(const ggml_sycl_lazy_storage * s, uint64_t out[4]) {
    if (out == nullptr) {
        return;
    }
    out[0] = out[1] = out[2] = out[3] = 0;
    if (s == nullptr || s->table->row_cache() == nullptr) {
        return;
    }
    const ggml_sycl::lazy_row_cache_counters c = s->table->row_cache()->counters();
    out[0]                                     = c.hits;
    out[1]                                     = c.misses;
    out[2]                                     = c.evictions;
    out[3]                                     = s->submitted.load();
}

void ggml_backend_sycl_lazy_storage_close(ggml_sycl_lazy_storage * s) {
    // A release is never refused: the guard only keeps the release from overlapping a module reactivation, and a
    // refused close would leak the table's MODEL-lifetime cache past its model.
    sycl_module_mutation_guard module_guard;
    delete s;
}
```

Add the four proc table entries in the existing form, for example
`if (strcmp(name, "ggml_backend_sycl_lazy_storage_open") == 0) return (void *) ggml_backend_sycl_lazy_storage_open;`.

**Gotchas**
- **Q1 is answered: an open after `model_load_end` is the designed order.**
  - The published record holds the IO-pool lease (`unified-cache.hpp:2447-2449`).
  - A model's ensure makes its ring after its load ends (`ggml-sycl.cpp:14119-14123`).
  - The ensure accepts the plan's charge for the transaction that created the ring (`lazy-io-pool.cpp:1742`).
  - **STOP and message the lead** all the same if the first device run refuses with `RING_MISMATCH`,
    `RING_UNPINNED` or `BODY_REFUSED`.
- **The struct.** Keep `ggml_sycl_lazy_storage` outside `namespace ggml_sycl`, so it matches the C declaration.
- **Teardown order.** `close` drops the table's reference, and the row cache's MODEL-lifetime allocations go
  when the table is destroyed. That must happen before the model's unload. T10 keeps this order for libllama.
- **The proc gate.** The L4 proc gate that `ggml_sycl_plan_lazy_tables_for_testing` names may scan exports.
  `ctest --test-dir build -N | grep -i proc` lists it. Run it with the contract.

**Commit:** `sycl: open, gather from and close a STORAGE lazy table by model token`.
Stage `ggml/include/ggml-sycl.h ggml/src/ggml-sycl/ggml-sycl.cpp ggml/src/ggml-sycl/CMakeLists.txt ggml/src/ggml-sycl/tests/test-sycl-lazy-plan-interface.cpp tests/test-sycl-lifecycle-source-contract.py`.

---

## T7: the record's STORAGE consumer is per arch, still absent everywhere, and qwen4exp's table is storage-only

**Track** C. **Depends** none.

**Done when**:
- the bridge and the refusal take the arch;
- both behave as today for every arch;
- `llama_model_lazy_tensor_attrs` reports `storage_only` only for the qwen4exp PLE table;
- the prefill test checks qwen4exp and gemma4 against one `consumer` boolean, which T13 flips for qwen4exp.

**Files**
- `src/llama-lazy-range-record.h`: `:33-36` (the attrs) and `:140-145` (the `check_storage` declaration).
- `src/llama-lazy-range-record.cpp`: `:129` (the forward declaration), `:173` (the bridge condition), and
  `:230-253` (the comment, the stub and the refusal).
- `src/llama-model.cpp:981-1006`.
- `src/llama-model-loader.cpp:1930`.
- `tests/test-llama-lazy-range-record.cpp:258`.
- `tests/test-llama-lazy-range-prefill.cpp:178-250`.
- `tests/test-llama-lazy-range-record-source.py`: `:259`, `:272`, `:284`, `:286` and `:297`.

**Step 1, RED. Host-only.**

**The record test.** Add after `:258`, once `none` is declared:

```cpp
        CHECK(llama_model_lazy_tensor_attrs(LLM_ARCH_QWEN4EXP, hp, t).storage_only,
              "the qwen4exp PLE table is not storage-only");
        CHECK(!g4.storage_only && !none.storage_only, "a table other than qwen4exp's PLE is storage-only");
```

**The prefill test.** Wrap the taker block (`:178-202`) and the refusal block (`:204-250`) in one loop over
both arches. Write the expectations in terms of `consumer`, so T13 changes one line:

```cpp
    for (const llm_arch a : { LLM_ARCH_QWEN4EXP, LLM_ARCH_GEMMA4 }) {
        const bool consumer = false;  // whether `a`'s graph consumes a STORAGE row; T13 makes it a == LLM_ARCH_QWEN4EXP
        // ... the two blocks, with `arch` replaced by `a` and these expectations:
        //   :191  unflagged excluded == on                                        (unchanged: AUTO at 4 GiB is EAGER)
        //   :193  excluded == (on || (mode == LLAMA_LAZY_MODE_AUTO && consumer))
        //   :196  deferral == STORAGE iff (on || (mode == LLAMA_LAZY_MODE_AUTO && consumer)), eligible likewise
        //   :200  n_warned == (mode == LLAMA_LAZY_MODE_AUTO && !consumer ? 1u : 0u)
        //   refusal cell: refused == (c.taker && c.mode == LLAMA_LAZY_MODE_ON && !consumer)
        //   :247  threw_eager == (unflagged_is_storage && !consumer)
        // and every llama_lazy_range_check_storage(rec, x) becomes llama_lazy_range_check_storage(rec, a, x).
    }
```

- Drop the `refused` column from `cells` (`:207-219`); the formula above replaces it.
- Relabel the AUTO taker cell `"AUTO, the taker"`. The parenthesis it carries today describes one outcome
  only.

**The pin script.** Change these five regexes to the per-arch form:
- `:259` keeps its `if\s*\(\s*flags\s*&\s*TENSOR_READ_LAZY\s*\)\s*\{\s*` prefix and becomes
  `...llama_lazy_range_check_storage\s*\(\s*lazy_record\s*,\s*get_arch\(\)\s*,\s*tn\.str\(\)\s*\)\s*;`.
- `:272` becomes
  `if\s*\(\s*record\s*\.\s*deferral\s*\(\s*name\s*\)\s*==\s*LLAMA_LAZY_DEFERRAL_STORAGE\s*&&\s*!\s*llama_lazy_storage_consumer_present\s*\(\s*arch\s*\)\s*\)`.
  Update its message to match.
- `:284` becomes `static\s+bool\s+llama_lazy_storage_consumer_present\s*\(\s*enum\s+llm_arch\s+arch\s*\)\s*\{`.
- `:286` becomes `!\s*llama_lazy_storage_consumer_present\s*\(\s*arch\s*\)`.
- `:297` takes the same `(arch)` in its conditioned form.

All three must fail first: the record test on a compile error (`storage_only`), the prefill test on a compile
error (the arity), and the pins on their regexes.

**Step 2, GREEN.**

`llama-lazy-range-record.h`:

```cpp
struct llama_lazy_tensor_attrs {
    uint32_t rows_per_token;  // 0 = not known for this arch
    int32_t  consumer_layer;  // -1 = not known for this arch
    bool     storage_only;    // never held resident: the planner places it STORAGE whatever the host room
};

void llama_lazy_range_check_storage(const llama_lazy_range_record & record,
                                    enum llm_arch                   arch,
                                    const std::string &             name);
```

Change the declaration's doc to "...and this arch has no STORAGE consumer...".

`llama-lazy-range-record.cpp`:
- `:129` becomes `static bool llama_lazy_storage_consumer_present(enum llm_arch arch);`.
- `:173` ends `... && !llama_lazy_storage_consumer_present(arch)) {`.
- The stub and the refusal become:

```cpp
// Whether `arch`'s graph consumes a STORAGE row. None does yet: llama_lazy_range_check_storage() below refuses a STORAGE
// row for an arch without one, and the pre-fill does not make one from the default mode (AUTO). An arch gains its
// consumer by returning true here, and its expectations move together: the `consumer` flag of
// tests/test-llama-lazy-range-prefill.cpp and the pin test_the_auto_bridge_lives_exactly_as_long_as_the_refusal in
// tests/test-llama-lazy-range-record-source.py.
static bool llama_lazy_storage_consumer_present(enum llm_arch arch) {
    GGML_UNUSED(arch);
    return false;
}

void llama_lazy_range_check_storage(const llama_lazy_range_record & record,
                                    enum llm_arch                   arch,
                                    const std::string &             name) {
    if (record.deferral(name) == LLAMA_LAZY_DEFERRAL_STORAGE && !llama_lazy_storage_consumer_present(arch)) {
        throw std::runtime_error(format("lazy-range record: %s is STORAGE and arch %s has no STORAGE consumer",
                                        name.c_str(), llm_arch_name(arch)));
    }
}
```

In `src/llama-model.cpp`:
- `:982` becomes `const llama_lazy_tensor_attrs unknown = { 0, -1, false };`.
- qwen4exp returns `{ hparams.ple_n_heads, consumer_layer, true }`.
- gemma4 returns `{ 1, 0, false }`.

`src/llama-model-loader.cpp:1930` becomes `llama_lazy_range_check_storage(lazy_record, get_arch(), tn.str());`.

**Gotchas**
- **The message.** The error message drops its ticket id. No test matches the message text.
- **The gemma4 copy.** It uses the same `declared` names. The record's rows depend on the names, and only the
  attrs and the bridge depend on the arch.
- **Default Qwen load:** unchanged.

**Commit:** `llama: the lazy-range STORAGE consumer is asked per arch; qwen4exp's PLE table is storage-only`.
Stage `src/llama-lazy-range-record.h src/llama-lazy-range-record.cpp src/llama-model.cpp src/llama-model-loader.cpp tests/test-llama-lazy-range-record.cpp tests/test-llama-lazy-range-prefill.cpp tests/test-llama-lazy-range-record-source.py`.

---

## T8: the early and late inventories carry each STORAGE row as a lazy table

**Track** C. **Depends** T3, T7.

**Done when** every STORAGE row reaches both inventories as a `ggml_sycl_lazy_table_info` with all of the S4
fields, through one converter that T9 also uses. Under AUTO nothing changes, because AUTO makes no STORAGE
row.

**Files**
- `src/llama-lazy-range-record.{h,cpp}`: a SYCL-free collector, `llama_lazy_range_storage_tables`.
- New `src/llama-lazy-storage.{h,cpp}`: the converter `llama_lazy_storage_table_info`, under the SYCL guard.
  Add the `.cpp` to `src/CMakeLists.txt` beside `llama-lazy-range-record.cpp` (`:38`).
- `src/llama-model.cpp`:
  - a helper `llama_model_sycl_lazy_table_infos`;
  - the early inventory struct (`:752-758`), and its fill after `:797`;
  - the late inventory's fill, after `:873`.
- `tests/test-llama-lazy-range-record.cpp`.
- New `tests/test-llama-lazy-storage.cpp`. Register it in the `foreach` at `tests/CMakeLists.txt:4186`.
- The pin script.

**Step 1, RED. Host-only.**

**The record test** reuses the ON-and-taker `storage` record at `:271`:

```cpp
    {
        ggml_init_params ip   = { 4 * ggml_tensor_overhead(), nullptr, true };
        ggml_context *   gctx = ggml_init(ip);
        ggml_tensor *    w    = ggml_new_tensor_2d(gctx, GGML_TYPE_Q8_0, 160, 1000);
        const auto       find = [&](const std::string &) -> const ggml_tensor * { return w; };
        const std::vector<llama_lazy_storage_table> tables = llama_lazy_range_storage_tables(storage, find);
        size_t n_storage = 0;
        for (const llama_lazy_range_row & row : storage.rows()) {
            n_storage += row.deferral == LLAMA_LAZY_DEFERRAL_STORAGE;
        }
        CHECK(n_storage > 0 && tables.size() == n_storage, "%zu tables for %zu STORAGE rows", tables.size(), n_storage);
        for (const llama_lazy_storage_table & t : tables) {
            const llama_lazy_range_value v = storage.value(t.name);
            CHECK(t.type == GGML_TYPE_Q8_0 && t.ne[0] == 160 && t.ne[1] == 1000 &&
                      t.row_size == ggml_row_size(GGML_TYPE_Q8_0, 160) && t.value.offs == v.offs &&
                      t.value.size == v.size && t.value.file_idx == v.file_idx,
                  "%s: a field differs from its row or tensor", t.name.c_str());
        }
        CHECK(llama_lazy_range_storage_tables(lazy, find).empty(), "a LAZY_READ record yields a storage table");
        ggml_free(gctx);
    }
```

**The new `tests/test-llama-lazy-storage.cpp`.** At this task it holds the converter check only; T9 adds the
handle's arms:

```cpp
#include "llama-lazy-storage.h"

#include <cstdio>
#include <cstring>

static int g_fails = 0;
#define CHECK(c, ...) do { if (!(c)) { std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); ++g_fails; } } while (0)

static void check_converter() {
#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
    llama_lazy_storage_table t = {};
    t.name                     = "per_layer_token_embd.weight";
    t.type                     = GGML_TYPE_Q8_0;
    t.ne[0]                    = 160;
    t.ne[1]                    = 1000;
    t.ne[2] = t.ne[3]          = 1;
    t.row_size                 = 170;
    t.value                    = { 2, 4032, 170000 };
    t.attrs                    = { 16, 1, true };
    const ggml_sycl_lazy_table_info i = llama_lazy_storage_table_info(t);
    CHECK(std::strcmp(i.name, t.name.c_str()) == 0 && i.name == t.name.c_str(), "name");
    CHECK(i.type == t.type && i.ne[0] == 160 && i.ne[1] == 1000 && i.row_size == 170, "shape");
    CHECK(i.file_index == 2 && i.file_offset == 4032 && i.size == 170000, "place");
    CHECK(i.rows_per_token == 16 && i.consumer_layer == 1 && i.storage_only, "attrs");
#else
    std::printf("SKIP converter: no SYCL in this build\n");
#endif
}

int main() {
    check_converter();
    return g_fails == 0 ? 0 : 1;
}
```

**The pin script.** It already defines `MODEL_CPP` at `:55`.

```python
def test_both_inventories_carry_the_storage_tables():
    src = read_code(MODEL_CPP)
    for fn in (r"static\s+bool\s+llama_model_sycl_build_early_inventory\s*\(",
               r"static\s+void\s+llama_model_sycl_set_late_inventory\s*\("):
        body = function_body(src, fn + r"(?:[^()]|\([^()]*\))*\)")
        assert "llama_model_sycl_lazy_table_infos(" in body and re.search(r"\.lazy_table_count\s*=", body), (
            f"{fn} does not hand the STORAGE rows to the planner")
    # anchored on the return type and the brace, so neither a call nor a declaration of the helper can match
    m = re.search(r"static\s+std::vector\s*<\s*ggml_sycl_lazy_table_info\s*>\s*llama_model_sycl_lazy_table_infos\s*"
                  r"\((?:[^()]|\([^()]*\))*\)\s*\{", src)
    assert m, "the inventories' helper llama_model_sycl_lazy_table_infos is not defined"
    helper = balanced(src, m.end() - 1)
    assert "llama_lazy_range_storage_tables(" in helper and "llama_lazy_storage_table_info(" in helper, (
        "the inventories' helper must collect through the record and convert through the shared converter")
```

All three must fail first.

**Step 2, GREEN.**

In `llama-lazy-range-record.h`, after `llama_lazy_range_row`:

```cpp
// A STORAGE row as the backend's inventory needs it: the row, and the shape and type of the weight it names.
struct llama_lazy_storage_table {
    std::string             name;
    enum ggml_type          type;
    int64_t                 ne[GGML_MAX_DIMS];
    size_t                  row_size;
    llama_lazy_range_value  value;
    llama_lazy_tensor_attrs attrs;
};

// Looks up the loader's tensor for a row's name.
using llama_lazy_range_tensor_fn = std::function<const ggml_tensor *(const std::string & name)>;

// The record's STORAGE rows, in record order, each with its weight's type, shape and row size. A name the lookup does
// not find aborts: a STORAGE row is a declared tensor the file carries.
std::vector<llama_lazy_storage_table> llama_lazy_range_storage_tables(const llama_lazy_range_record &    record,
                                                                      const llama_lazy_range_tensor_fn & tensor);
```

In the `.cpp`:

```cpp
std::vector<llama_lazy_storage_table> llama_lazy_range_storage_tables(const llama_lazy_range_record &    record,
                                                                      const llama_lazy_range_tensor_fn & tensor) {
    std::vector<llama_lazy_storage_table> out;
    for (const llama_lazy_range_row & row : record.rows()) {
        if (row.deferral != LLAMA_LAZY_DEFERRAL_STORAGE) {
            continue;
        }
        const ggml_tensor * t = tensor(row.name);
        if (t == nullptr) {
            GGML_ABORT("lazy-range record: STORAGE row %s has no tensor in the model file", row.name.c_str());
        }
        llama_lazy_storage_table s = {};
        s.name                     = row.name;
        s.type                     = t->type;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            s.ne[d] = t->ne[d];
        }
        s.row_size = ggml_row_size(t->type, t->ne[0]);
        s.value    = row.value;
        s.attrs    = row.attrs;
        out.push_back(std::move(s));
    }
    return out;
}
```

New `src/llama-lazy-storage.h`, as created here (T9 adds the handle):

```cpp
#pragma once

#include "llama-lazy-range-record.h"

#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
#    include "ggml-sycl.h"

// The backend's view of a STORAGE table, for the planner's inventory and for the open proc: one conversion, so the two
// cannot describe one table differently. `name` points into `table.name`: the info lives no longer than the table.
ggml_sycl_lazy_table_info llama_lazy_storage_table_info(const llama_lazy_storage_table & table);
#endif
```

New `src/llama-lazy-storage.cpp`:

```cpp
#include "llama-lazy-storage.h"

#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
ggml_sycl_lazy_table_info llama_lazy_storage_table_info(const llama_lazy_storage_table & t) {
    ggml_sycl_lazy_table_info i = {};
    i.name                      = t.name.c_str();
    i.type                      = t.type;
    i.rows_per_token            = t.attrs.rows_per_token;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        i.ne[d] = t.ne[d];
    }
    i.row_size       = t.row_size;
    i.consumer_layer = t.attrs.consumer_layer;
    i.file_index     = t.value.file_idx;
    i.file_offset    = t.value.offs;
    i.size           = t.value.size;
    i.storage_only   = t.attrs.storage_only;
    return i;
}
#endif
```

In `src/llama-model.cpp`:
- Add `#include "llama-lazy-storage.h"` to the project includes at `:3-12`, unguarded. The header guards its own
  SYCL half, and T10 uses its unguarded half from this file.
- Define the helper inside the existing SYCL guard (which closes at `:892`), after the
  `llama_model_sycl_early_inventory` struct (`:752-758`) and before `llama_model_sycl_build_early_inventory`
  (`:761`). Give it no forward declaration: the pin finds it by its return type and brace.

```cpp
// The STORAGE rows as the planner's lazy tables. `tables` owns the names the infos point at, so it must live as long
// as the inventory that carries them.
static std::vector<ggml_sycl_lazy_table_info> llama_model_sycl_lazy_table_infos(
        const llama_model_loader & ml, std::vector<llama_lazy_storage_table> & tables) {
    tables = llama_lazy_range_storage_tables(ml.lazy_record, [&](const std::string & name) -> const ggml_tensor * {
        const auto it = ml.weights_map.find(name);
        return it != ml.weights_map.end() ? it->second.tensor : nullptr;
    });
    std::vector<ggml_sycl_lazy_table_info> infos;
    for (const llama_lazy_storage_table & t : tables) {
        infos.push_back(llama_lazy_storage_table_info(t));
    }
    return infos;
}
```

**The early inventory.**
- Add two members to `llama_model_sycl_early_inventory`:
  `std::vector<llama_lazy_storage_table> lazy_tables;` and `std::vector<ggml_sycl_lazy_table_info> lazy_infos;`.
- After `:797`, set `out.lazy_infos = llama_model_sycl_lazy_table_infos(ml, out.lazy_tables);`.
- Then set `inventory.lazy_tables = out.lazy_infos.empty() ? nullptr : out.lazy_infos.data();`.
- Then set `inventory.lazy_table_count` to the vector's size, cast to the field's type (`ggml-sycl.h:474`).

**The late inventory.** Do the same with locals, after `:873` and before `apply_inventory`.

**Gotchas**
- **The tensor's shape.** The `.tensor` in a `weights_map` entry is the file's tensor, so its `ne` is the file
  shape. That is the shape the planner wants.
- **Guard placement.** `llama-model.cpp` includes `ggml-sycl.h` under the guard at `:30-32`. Keep the helper
  under the same guard.
- **Default Qwen load:** unchanged.
- **`-lzm on`.** It now prints a `[PLACEMENT] lazy per_layer_token_embd.weight residency=STORAGE` line before
  `create_tensor` refuses. The lead confirms this once:

  ```bash
  ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_TUNING_CACHE=0 timeout 600 \
    ./build/bin/llama-completion -m $QWEN -lzm on -n 1 -p hi
  ```

  Sample Shmem before and about 5 s after. Expected: the STORAGE line, then the load fails naming the tensor
  and the arch.

**Commit:** `llama: the SYCL inventories carry the STORAGE lazy-range rows as lazy tables`.
Stage `src/llama-lazy-range-record.h src/llama-lazy-range-record.cpp src/llama-lazy-storage.h src/llama-lazy-storage.cpp src/CMakeLists.txt src/llama-model.cpp tests/test-llama-lazy-range-record.cpp tests/test-llama-lazy-storage.cpp tests/CMakeLists.txt tests/test-llama-lazy-range-record-source.py`.

---

## T9: `llama_lazy_storage`, libllama's handle on an opened table

**Track** C. **Depends** T6 (the proc signatures) and T8 (the files and the converter).

**Done when** the handle:
- refuses a gather until it is opened, and forwards gathers after that;
- prints its counters once when it is destroyed;
- closes the backend handle exactly once.

**Files**
- `src/llama-lazy-storage.{h,cpp}`.
- `tests/test-llama-lazy-storage.cpp`.

**Step 1, RED. Host-only.** Add to the test file. It uses fake procs and no backend:

```cpp
#include "llama-impl.h"
#include "llama-model.h"  // llama_sycl_model_token (:24-29), taken by value

#include <cerrno>
#include <string>

static int         g_closed = 0;
static int         g_handle = 0;
static std::string g_log;

static void * fake_open(llama_sycl_model_token, const llama_lazy_storage_table &, const char *) { return &g_handle; }
static int    fake_gather(void *, const int32_t * rows, size_t n, void * dst, size_t) {
    for (size_t i = 0; i < n; ++i) {
        if (rows[i] == 99) {
            return EIO;
        }
        std::memset(static_cast<uint8_t *>(dst) + i * 4, rows[i], 4);
    }
    return 0;
}
static void fake_counters(const void *, uint64_t out[4]) { out[0] = 1; out[1] = 2; out[2] = 0; out[3] = 2; }
static void fake_close(void * h) { g_closed += h == &g_handle; }

static void check_handle() {
    llama_log_set([](ggml_log_level, const char * t, void *) { g_log += t; }, nullptr);
    const llama_lazy_storage_procs procs = { fake_open, fake_gather, fake_counters, fake_close };
    llama_lazy_storage_table       table = {};
    table.name     = "per_layer_token_embd.weight";
    table.type     = GGML_TYPE_I8;
    table.ne[0]    = 4;
    table.ne[1]    = 100;
    table.row_size = 4;
    {
        llama_lazy_storage s(table, "/models/x.gguf", procs);
        uint8_t       dst[8];
        const int32_t rows[2] = { 1, 2 };
        CHECK(s.gather(rows, 2, dst, sizeof(dst)) == ENODEV, "a gather before the open did not return ENODEV");
        llama_sycl_model_token token;
        token.model_id        = 1;
        token.load_txn_id     = 1;
        token.slot            = 0;
        token.slot_generation = 1;
        CHECK(s.open(token), "the open failed");
        CHECK(s.gather(rows, 2, dst, sizeof(dst)) == 0 && dst[4] == 2, "the gather was not forwarded");
        const int32_t bad[1] = { 99 };
        CHECK(s.gather(bad, 1, dst, 4) == EIO, "a failed gather did not return its errno");
    }
    llama_log_set(nullptr, nullptr);
    CHECK(g_closed == 1, "closed %d times", g_closed);
    CHECK(g_log.find("[LAZY-STORAGE] per_layer_token_embd.weight hits=1 misses=2 evictions=0 submitted=2") !=
              std::string::npos,
          "no counters line: %s", g_log.c_str());
}
```

In `main`, call `check_handle();` after `check_converter();`.

It must first fail to compile, because `llama_lazy_storage` does not exist.

**Step 2, GREEN.** Add to `llama-lazy-storage.h`, outside the SYCL guard:

```cpp
#include <cstddef>
#include <cstdint>
#include <string>

struct llama_sycl_model_token;

// The backend's four storage procs, as libllama calls them; the handle is the backend's, opaque here.
struct llama_lazy_storage_procs {
    void * (*open)(llama_sycl_model_token token, const llama_lazy_storage_table & table, const char * file_path);
    int (*gather)(void * handle, const int32_t * rows, size_t n, void * dst, size_t dst_bytes);
    // out[0] hits, out[1] misses, out[2] evictions, out[3] waiters submitted for misses (not disk reads)
    void (*counters)(const void * handle, uint64_t out[4]);
    void (*close)(void * handle);
};

// The SYCL backend's procs, adapted; all null when the backend lacks any of them, or in a build without SYCL.
llama_lazy_storage_procs llama_lazy_storage_sycl_procs();

// A STORAGE table the model's graph reads through set_input: created before the load's measure, so the graph builds
// its F32 input instead of a gather from the table, and opened once the load has a model token. Not copyable: the
// backend handle is closed exactly once.
class llama_lazy_storage {
  public:
    llama_lazy_storage(const llama_lazy_storage_table & table, std::string file_path,
                       const llama_lazy_storage_procs & procs);
    ~llama_lazy_storage();
    llama_lazy_storage(const llama_lazy_storage &)             = delete;
    llama_lazy_storage & operator=(const llama_lazy_storage &) = delete;

    bool open(const llama_sycl_model_token & token);  // false, after a WARN, when the backend refuses
    int  gather(const int32_t * rows, size_t n, void * dst, size_t dst_bytes) const;  // 0, ENODEV unopened, or errno

    const std::string & name() const { return table_.name; }
    ggml_type           type() const { return table_.type; }
    int64_t             ne0() const { return table_.ne[0]; }
    size_t              row_size() const { return table_.row_size; }

  private:
    llama_lazy_storage_table table_;
    std::string              file_path_;
    llama_lazy_storage_procs procs_;
    void *                   handle_ = nullptr;
};
```

In `llama-lazy-storage.cpp` (include `llama-impl.h` and `llama-model.h`):
- **The constructor** stores its arguments.
- **`open`.** With a null `procs_.open`, it WARNs and returns false. Otherwise it sets
  `handle_ = procs_.open(token, table_, file_path_.c_str())` and returns `handle_ != nullptr`.
- **`gather`** returns `ENODEV` while `handle_` is null, and forwards otherwise.
- **The destructor.** If `handle_` is set, it reads the counters and prints them with
  `LLAMA_LOG_WARN("[LAZY-STORAGE] %s hits=%llu misses=%llu evictions=%llu submitted=%llu\n", ...)`. It uses
  WARN so that a normal run shows the line. Then it calls `procs_.close(handle_)`.
- **`llama_lazy_storage_sycl_procs()`.**
  - Under the SYCL guard, it finds the SYCL reg the way `llama_model_sycl_hooks()` does
    (`src/llama-model.cpp:86-122`) and resolves the four procs by name.
  - It wraps them in four static adapters. The open adapter builds the backend's `ggml_sycl_model_token`
    field by field from `llama_sycl_model_token`, and the info with T8's `llama_lazy_storage_table_info`.
  - Without the guard, or if any proc is missing, it returns all-null.

**Gotchas**
- **The counters line is the E2E's engagement evidence.** Keep its format exactly as the test pins it.
- **A non-SYCL build** compiles the class and resolves null procs. No STORAGE row can exist there.

**Commit:** `llama: a handle on an opened STORAGE lazy table`.
Stage `src/llama-lazy-storage.h src/llama-lazy-storage.cpp tests/test-llama-lazy-storage.cpp`.

---

## T10: the loader keeps a STORAGE tensor out of every buffer, and the model opens it after the load

**Track** C. **Depends** T9.

**Done when**:
- a STORAGE row produces a tensor in a no_alloc context that no buffer, graph measure or `load_all_data` sees;
- its `llama_lazy_storage` exists before the probe measure;
- it is opened right after `finish(true)`, or the load fails;
- a file without a path refuses the load, naming the table.

This code is unreachable until T13. Today a STORAGE row throws in `create_tensor` before reaching this
branch. And between T10 and T13 the graph would still read a table that has no data.

**Files**
- `src/llama-lazy-range-record.{h,cpp}`: a query, `llama_lazy_range_is_storage`.
- `src/llama-model-loader.h`, beside `ctx_map` (`:197`).
- `src/llama-model-loader.cpp`, the lazy branch of `create_tensor` (`:1928-1934`).
- `src/llama-model.h`:
  - `class llama_lazy_storage;` forward-declared near `:20`;
  - an accessor near `:814`.
- `src/llama-model.cpp`:
  - `#include "llama-lazy-storage.h"`, which T8 adds at `:3-12` (it needs the complete class here; add it if it
    is missing);
  - `llama_model::impl`, `:2185-2205`;
  - after "populate tensors_by_name", `:2867-2871`;
  - the final finish and cancel, `:3370-3376`.
- `src/llama-lazy-storage.h`: `is_open()`.
- The pin script.

**Step 1, RED. Host-only source pins.** The branch is unreachable until T13, so its proof here is
textual. T14 proves it at runtime.

```python
def test_a_storage_tensor_bypasses_buffers_and_is_opened_after_the_load():
    ct = create_tensor_body()
    i_check = ct.find("llama_lazy_range_check_storage(")
    i_store = ct.find("create_storage_tensor(")
    i_buft  = ct.rfind("buft_for_tensor(")
    assert 0 <= i_check < i_store < i_buft, "a STORAGE tensor must leave create_tensor before the last buft_for_tensor()"
    assert "llama_lazy_range_is_storage(" in ct, "create_tensor asks the record whether a row is STORAGE"
    lt = load_tensors_body()
    i_entries = lt.find("new llama_lazy_storage(")
    i_probe   = lt.find("probe")
    assert 0 <= i_entries < i_probe, "the storage entries must exist before the probe measure builds a graph"
    tail = lt[lt.rfind("has_sycl_weight_buffer"):]
    assert re.search(r"finish\(true\);\s*for\s*\(.*lazy_storage.*\)\s*\{[^}]*->open\(", tail, re.S), (
        "every storage entry is opened right after the load's finish")
    assert re.search(r"file_paths.*\.empty\(\)", lt), "a STORAGE table with no file path refuses"
```

The test uses the script's own `create_tensor_body()` (`:138`) and `load_tensors_body()` (`:114`), which
matches `llama_model_base::load_tensors`.

The existing pin at `:265`, `"LLAMA_LAZY_DEFERRAL_STORAGE" not in` the `create_tensor` body, stays as it is.
`create_tensor` asks through `llama_lazy_range_is_storage`, so the token never appears in its body.

The new pin must fail first.

**Step 2, GREEN.**

**The record.** It gains a query:

```cpp
// Whether `name`'s row is STORAGE: the one question create_tensor() asks to keep the table out of every buffer. Aborts
// on a name without a row, like the other accessors.
bool llama_lazy_range_is_storage(const llama_lazy_range_record & record, const std::string & name);
// .cpp: return record.deferral(name) == LLAMA_LAZY_DEFERRAL_STORAGE;
```

**`llama-model-loader.h`.** Add beside `ctx_map`:

```cpp
    // The STORAGE tables' tensors: metadata only, in no buffer, so neither a buffer allocation, the load's measure nor
    // load_all_data ever sees them. Moved to the model in load_tensors.
    ggml_context_ptr storage_ctx;
    ggml_tensor *    create_storage_tensor(const std::string & name, const ggml_tensor * cur);
```

**`llama-model-loader.cpp`.** The lazy branch becomes:

```cpp
    if (flags & TENSOR_READ_LAZY) {
        // a STORAGE row is the SYCL storage tier's, not lazy_read's: refused for an arch with no consumer, and
        // otherwise kept out of buft_for_tensor(), which would make it an ordinary weight in host memory
        llama_lazy_range_check_storage(lazy_record, get_arch(), tn.str());
        if (llama_lazy_range_is_storage(lazy_record, tn.str())) {
            return create_storage_tensor(tn.str(), cur);
        }
        is_lazy = lazy.add(tn.str(), cur, no_alloc ? nullptr : &require_weight(tn.str().c_str()));
    }
```

It also defines the new member:

```cpp
ggml_tensor * llama_model_loader::create_storage_tensor(const std::string & name, const ggml_tensor * cur) {
    if (!storage_ctx) {
        ggml_init_params params = { 8 * ggml_tensor_overhead(), nullptr, true };
        storage_ctx.reset(ggml_init(params));
        if (!storage_ctx) {
            throw std::runtime_error("failed to create the STORAGE tensors' context");
        }
    }
    ggml_tensor * t = ggml_dup_tensor(storage_ctx.get(), cur);
    ggml_set_name(t, name.c_str());
    // never read in full, so not part of the load's progress: load_all_data's last-tensor cleanup waits for
    // size_done to reach size_data
    size_data -= ggml_nbytes(cur);
    n_created++;
    return t;
}
```

**`llama_model::impl`** gains:

```cpp
    ggml_context_ptr                                 storage_ctx;
    std::vector<std::unique_ptr<llama_lazy_storage>> lazy_storage;
```

**`llama-model.h`** gains two things:
- the forward declaration `class llama_lazy_storage;`;
- the accessor `const llama_lazy_storage * lazy_storage(const std::string & name) const;`, which returns the
  entry or null. Define it in `llama-model.cpp`.

**`load_tensors`.** After the populate loop at `:2871`:

```cpp
    // the STORAGE tables, before the measure below builds a graph: their consumer builds its F32 input from them, and
    // the table tensor itself must never enter a graph
    pimpl->storage_ctx = std::move(ml.storage_ctx);
    if (pimpl->storage_ctx) {
        const llama_lazy_storage_procs procs = llama_lazy_storage_sycl_procs();
        for (const llama_lazy_storage_table & t : llama_lazy_range_storage_tables(
                 ml.lazy_record, [&](const std::string & name) -> const ggml_tensor * {
                     return ggml_get_tensor(pimpl->storage_ctx.get(), name.c_str());
                 })) {
            // a file loaded from an open handle has no path (llama-model-loader.cpp:765), and the table is read
            // through its own O_DIRECT descriptor, which needs one
            if (t.value.file_idx >= ml.file_paths.size() || ml.file_paths[t.value.file_idx].empty()) {
                throw std::runtime_error(format("STORAGE table %s: its file was opened from a handle and has no path "
                                                "to read the table from; load the model from a file path",
                                                t.name.c_str()));
            }
            pimpl->lazy_storage.emplace_back(new llama_lazy_storage(t, ml.file_paths[t.value.file_idx], procs));
        }
    }
```

**The finish.** At `:3370-3376`:

```cpp
#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
    if (has_sycl_weight_buffer) {
        sycl_model_loading_guard.finish(true);
        for (auto & s : pimpl->lazy_storage) {
            if (!s->open(sycl_model_token)) {
                throw std::runtime_error(format("failed to open STORAGE table %s", s->name().c_str()));
            }
        }
    } else {
        sycl_model_loading_guard.cancel();
    }
#endif
    for (const auto & s : pimpl->lazy_storage) {
        if (!s->is_open()) {
            throw std::runtime_error(format("STORAGE table %s needs a SYCL load that finishes with a model token",
                                            s->name().c_str()));
        }
    }
    return true;
```

Add `bool is_open() const { return handle_ != nullptr; }` to the class in `llama-lazy-storage.h`.

**Gotchas**
- **The no_alloc path.** It returns before the open (`:3330-3340`), so its entries stay unopened. The fit pass
  builds graphs and never sets their inputs, so that is fine.
- **`memory_breakdown`.** It asserts one buffer per context in `ctxs_bufs` (`:3428-3444`). The storage context
  is not in `ctxs_bufs` and must never be added there.
- **`tensors_by_name`.** It does not get the table. T11 covers the saver.
- **Teardown order.** `~llama_model` resets `pimpl` before `hooks.unload(token)` (`:2284-2314`), so each table
  closes before its plan is erased. Keep that order.
- **Includes.** `llama_model::impl`'s `std::unique_ptr<llama_lazy_storage>` vector, the `new llama_lazy_storage`
  and `llama_lazy_storage_sycl_procs()` need the complete class. T8 already added
  `#include "llama-lazy-storage.h"` to `llama-model.cpp`; confirm it is there. `llama-model.h` keeps only the
  forward declaration.

**Commit:** `llama: a STORAGE lazy table stays out of every buffer and is opened after the load`.
Stage `src/llama-lazy-range-record.h src/llama-lazy-range-record.cpp src/llama-model-loader.h src/llama-model-loader.cpp src/llama-model.h src/llama-model.cpp src/llama-lazy-storage.h tests/test-llama-lazy-range-record-source.py`.

---

## T11: the model saver refuses a model whose table is STORAGE

**Track** C. **Depends** T10. Merge it before T13, so no build ever has a default STORAGE table that the saver
would silently write without data.

**Done when** saving a model that holds a STORAGE table aborts before writing anything, with a message that
names the tensor and says why. Every other model saves as before.

**Files**
- `src/llama-model-saver.cpp:131-139`, `add_tensor`.
- The pin script: add `SAVER_CPP = ROOT / "src/llama-model-saver.cpp"`.

**Step 1, RED. Host-only source pin.** A runtime test would need a SYCL load of a model with a STORAGE table.
That cannot happen before T13, so this proof is textual. T14 runs a STORAGE load, but it does not save.

```python
def test_the_saver_refuses_a_storage_table():
    body = function_body(read_code(SAVER_CPP), r"void\s+llama_model_saver::add_tensor\s*\((?:[^()]|\([^()]*\))*\)")
    i_check = body.find("lazy_storage(")
    i_add   = body.find("gguf_add_tensor")
    assert 0 <= i_check < i_add and re.search(r"GGML_ABORT\s*\([^;]*%s", body), (
        "add_tensor must abort naming a STORAGE table before it adds the tensor")
```

It must fail first.

**Step 2, GREEN.** At the top of `add_tensor`, after the null check:

```cpp
    // a STORAGE table is read from its file per ubatch and has no data in memory: saving it would write a tensor of
    // garbage, so the save stops here (the saver's failures are aborts; its entry point returns nothing)
    if (model != nullptr && model->lazy_storage(ggml_get_name(tensor)) != nullptr) {
        GGML_ABORT("cannot save %s: the table is read from its model file on demand and is not in memory; "
                   "copy the original GGUF instead", ggml_get_name(tensor));
    }
```

**Gotchas**
- **Why an abort.** `llama_model_save_to_file` returns `void` (`src/llama.cpp:505-510`), and `add_tensor`
  already fails through `GGML_ASSERT` (`:137`). An exception would leave the C API.
- **What the saver's caller does.** It calls `add_tensors_from_model()` before `save()` (`src/llama.cpp:508-509`),
  so the abort fires before any file is written.

**Commit:** `llama: the model saver refuses a model whose lazy table is STORAGE`.
Stage `src/llama-model-saver.cpp tests/test-llama-lazy-range-record-source.py`.

---

## T12: a graph input can fail its ubatch

**Track** D. **Depends** none. Merge it after T8, because both write `tests/CMakeLists.txt`.

**Done when** an input that reports a failure makes `process_ubatch` return `GGML_STATUS_FAILED` without
calling `graph_compute`. No current input reports a failure.

**Files**
- `src/llama-graph.h`: `:102-124` (`llm_graph_input_i`) and `:917`.
- `src/llama-graph.cpp:1359-1363`.
- `src/llama-context.cpp:5367-5370`.
- New `tests/test-llama-graph-input-failed.cpp`.
- New `tests/test-llama-graph-input-failed-source.py`.
- `tests/CMakeLists.txt`. Register both new tests after the lazy-range block (`:4186-4192`):

```cmake
llama_build_and_test(test-llama-graph-input-failed.cpp)
target_include_directories(test-llama-graph-input-failed PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_tests_properties(test-llama-graph-input-failed PROPERTIES LABELS "host")
if (Python3_Interpreter_FOUND)
    llama_test_pytest(${Python3_EXECUTABLE}
        NAME test-llama-graph-input-failed-source
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-llama-graph-input-failed-source.py
        LABEL "host;python;source-contract"
        TIMEOUT 120)
endif()
```

The pytest registration follows the form of its neighbour at `:4066-4071`.

**Step 1, RED. Host-only.**

```cpp
#include "llama-graph.h"

#include <cstdio>
#include <memory>

struct reporting_input : llm_graph_input_i {
    bool fail;
    explicit reporting_input(bool f) : fail(f) {}
    void set_input(const llama_ubatch *) override {}
    bool set_input_failed() const override { return fail; }
};

int main() {
    llm_graph_result res(64);
    res.add_input(std::make_unique<reporting_input>(false));
    if (res.inputs_failed()) {
        std::printf("FAIL: a clean input reported a failure\n");
        return 1;
    }
    res.add_input(std::make_unique<reporting_input>(true));
    if (!res.inputs_failed()) {
        std::printf("FAIL: a failed input was not reported\n");
        return 1;
    }
    return 0;
}
```

The pin script carries its own helpers. Copy `test-llama-lazy-range-record-source.py:59-98` whole: the
`_LEXEME_RE` regex (`:59-65`), which `strip_comments` uses, then `strip_comments`, `read_code`, `balanced` and
`function_body` (`:68-98`). Copying only `:68-98` raises NameError on the first read:

```python
"""A graph input that fails its set_input() stops the ubatch before graph_compute."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTEXT_CPP = ROOT / "src/llama-context.cpp"

# ... _LEXEME_RE and the four helpers, copied from test-llama-lazy-range-record-source.py:59-98 ...

def test_process_ubatch_stops_on_a_failed_input():
    body = function_body(read_code(CONTEXT_CPP),
                         r"llm_graph_result\s*\*\s*llama_context::process_ubatch\s*\((?:[^()]|\([^()]*\))*\)")
    i_set, i_fail, i_compute = body.find("set_inputs("), body.find("inputs_failed()"), body.find("graph_compute(")
    assert 0 <= i_set < i_fail < i_compute, "a failed input must stop the ubatch between set_inputs and graph_compute"
```

Both must fail first.

**Step 2, GREEN.**

**`llm_graph_input_i`.** After `can_reuse`:

```cpp
    // true when the last set_input() could not produce its data (a read that failed): the ubatch must not compute.
    // set_input() never throws; it reports here instead.
    virtual bool set_input_failed() const { return false; }
```

**`llm_graph_result`.** Declare `bool inputs_failed() const;` and define it:

```cpp
bool llm_graph_result::inputs_failed() const {
    for (const auto & input : inputs) {
        if (input->set_input_failed()) {
            return true;
        }
    }
    return false;
}
```

**`process_ubatch`.** After `res->set_inputs(&ubatch);`:

```cpp
        if (res->inputs_failed()) {
            LLAMA_LOG_ERROR("%s: a graph input could not be set; the ubatch is not computed\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }
```

**Gotchas**
- **Keep `set_inputs` returning `void`.** Its other callers, `llama-context.cpp:7645` and
  `llama-kv-cache.cpp:1051`, stay untouched. Neither carries a PLE input.
- **The rollback.** `decode` maps FAILED to -3 and rolls the batch back through `seq_rm` (about
  `:5818-5845`). Read that path once, and confirm a failure on the first ubatch leaves the KV unchanged.
- **Default load:** unchanged.

**Commit:** `llama: a graph input can fail its ubatch before compute`.
Stage `src/llama-graph.h src/llama-graph.cpp src/llama-context.cpp tests/test-llama-graph-input-failed.cpp tests/test-llama-graph-input-failed-source.py tests/CMakeLists.txt`.

---

## T13: qwen4exp reads the PLE rows from storage as f32, and its STORAGE consumer exists

**Track** D. **Depends** T2, T5, T10, T11, T12.

**Done when**, with a STORAGE PLE table:
- the qwen4exp graph reads one F32 input that `set_input` gathers and dequantizes on the host;
- the graph has no `get_rows` and no I32 row input in that mode;
- a failed gather fails the ubatch;
- AUTO places the table STORAGE.

This is the only task that changes the default Qwen load.

**Files**
- `src/models/qwen4exp.cpp`: the includes at `:1-7` (add `"llama-lazy-storage.h"`, whose class `llama-model.h`
  only forward-declares, and `<cstring>` for `strerror`), the class at `:1171-1193`, `set_input` at `:1195-1256`,
  and `build_inp_ple` at `:1314-1333`.
- `src/llama-lazy-range-record.cpp`: T7's stub.
- `tests/test-llama-lazy-range-prefill.cpp`: T7's `consumer` line.
- The pin script: add `QWEN4EXP_CPP = ROOT / "src/models/qwen4exp.cpp"`.

**Step 1, RED. Host-only.**

**The prefill test.** In T7's arch loop, change one line:

`const bool consumer = a == LLM_ARCH_QWEN4EXP;`

Every expectation T7 wrote in terms of `consumer` now covers the qwen4exp changes:
- `:193`, the table excluded under AUTO;
- `:196`, STORAGE and eligible under AUTO;
- `:200`, no bridge WARN;
- the refusal cells, which no longer refuse;
- `:247`, the 4 GiB ON row, which no longer throws.

The gemma4 pass keeps today's expectations.

**The pin:**

```python
def test_qwen4exp_is_the_one_storage_consumer():
    # anchored on the definition's brace: the forward declaration at the top of the file matches the same signature
    # and would hand function_body the next function's body
    cpp = read_code(RECORD_CPP)
    m = re.search(r"static\s+bool\s+llama_lazy_storage_consumer_present\s*\(\s*enum\s+llm_arch\s+arch\s*\)\s*\{", cpp)
    assert m, "the consumer predicate's definition is missing"
    body = balanced(cpp, m.end() - 1)
    assert re.search(r"return\s+arch\s*==\s*LLM_ARCH_QWEN4EXP\s*;", body), "qwen4exp's graph consumes its STORAGE table"
    q = read_code(QWEN4EXP_CPP)
    build = function_body(q, r"ggml_tensor\s*\*\s*llama_model_qwen4exp::graph::build_inp_ple\s*\((?:[^()]|\([^()]*\))*\)")
    assert "lazy_storage(" in build and "GGML_TYPE_F32" in build, "the PLE input is an F32 input in STORAGE mode"
    assert "to_float" in q and "set_input_failed" in q, "set_input dequantizes on the host and reports a failure"
```

Both must fail first.

**Step 2, GREEN.**

**The stub.**

```cpp
static bool llama_lazy_storage_consumer_present(enum llm_arch arch) {
    // qwen4exp's PLE input gathers its rows from the storage tier in set_input (models/qwen4exp.cpp)
    return arch == LLM_ARCH_QWEN4EXP;
}
```

Update the stub's comment: qwen4exp has a consumer, and the arches without one keep the bridge and the
refusal.

**The includes.** `qwen4exp.cpp` gains, in its include block (`:1-7`):

```cpp
#include "llama-lazy-storage.h"

#include <cstring>
```

**`llm_graph_input_ple`.** Its members become:

```cpp
    bool set_input_failed() const override { return failed; }

    ggml_tensor * rows = nullptr;  // I32 [ple_n_heads * n_tokens]; null when the table is STORAGE
    // F32 [ne0, ple_n_heads * n_tokens], set only when the table is STORAGE: the rows this ubatch reads, gathered from
    // the storage tier and dequantized on the host, in the order get_rows would produce them
    ggml_tensor * emb  = nullptr;
    const llama_lazy_storage * storage = nullptr;
    bool                       failed  = false;
    std::vector<uint8_t>       raw;  // scratch, reused across set_input() calls: the gathered rows in the table's type
    std::vector<float>         deq;  // scratch: the same rows as f32
```

**`can_reuse`** checks whichever input this mode has:

```cpp
    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx)->get_attn();
        const int64_t n = (int64_t) pmodel.hparams.ple_n_heads * params.ubatch.n_tokens;
        return emb != nullptr ? emb->ne[1] == n : rows->ne[0] == n;
    }
```

**`set_input`.** Replace the final `ggml_backend_tensor_set` at `:1255`:

```cpp
    if (storage == nullptr) {
        ggml_backend_tensor_set(rows, idx.data(), 0, idx.size()*ggml_element_size(rows));
        return;
    }
    // STORAGE: the rows come from the storage tier (hits from its row cache, misses read from the file and waited for
    // here) and are dequantized to f32 on the host, so the graph receives the values the eager get_rows would compute
    // in f32, whichever backend runs the graph. A failure is reported through set_input_failed(), never thrown.
    failed                         = true;  // until every step below succeeds
    const ggml_type_traits * tt    = ggml_get_type_traits(storage->type());
    const size_t             n     = idx.size();
    if (tt->to_float == nullptr) {
        LLAMA_LOG_ERROR("%s: the PLE table's type %s has no f32 conversion\n", __func__, ggml_type_name(storage->type()));
        return;
    }
    try {
        raw.resize(n * storage->row_size());
        deq.resize(n * storage->ne0());
    } catch (const std::bad_alloc &) {
        LLAMA_LOG_ERROR("%s: no host memory for %zu PLE rows\n", __func__, n);
        return;
    }
    const int rc = storage->gather(idx.data(), n, raw.data(), raw.size());
    if (rc != 0) {
        LLAMA_LOG_ERROR("%s: reading %zu PLE rows of %s failed: %s\n", __func__, n, storage->name().c_str(),
                        strerror(rc));
        return;
    }
    // one call over every row: a row is whole blocks (ne0 is a multiple of the block size, which the backend's open
    // refuses otherwise), so the gathered rows are one contiguous run of blocks
    tt->to_float(raw.data(), deq.data(), (int64_t) deq.size());
    ggml_backend_tensor_set(emb, deq.data(), 0, deq.size() * sizeof(float));
    failed = false;
```

**`build_inp_ple`.** It becomes:

```cpp
ggml_tensor * llama_model_qwen4exp::graph::build_inp_ple(
        const llama_memory_hybrid_idx_context * mctx_hyb) {
    const int64_t n_heads = hparams.ple_n_heads;

    // the attention cells see every ubatch regardless of the layer types
    auto ple_inp = std::make_unique<llm_graph_input_ple>(
            static_cast<const llama_model_qwen4exp &>(model), mctx_hyb->get_attn());

    ggml_tensor * emb = nullptr;
    if (const llama_lazy_storage * storage = model.lazy_storage(ggml_get_name(model.per_layer_tok_embd))) {
        // the table is never resident: set_input gathers this ubatch's rows and dequantizes them on the host, so the
        // graph takes them as an F32 input where the get_rows result was, with no table and no row ids
        GGML_ASSERT(storage->ne0() == hparams.ple_head_dim);
        ple_inp->storage = storage;
        ple_inp->emb     = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, storage->ne0(), n_heads * n_tokens);
        ggml_set_input(ple_inp->emb);
        emb = ple_inp->emb;
        res->add_input(std::move(ple_inp));
    } else {
        ple_inp->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_heads * n_tokens);
        ggml_set_input(ple_inp->rows);
        ggml_tensor * rows = ple_inp->rows;
        res->add_input(std::move(ple_inp));
        // gather then flatten the heads: get_rows lays the head dimension out slowest, as the reference does
        emb = ggml_get_rows(ctx0, model.per_layer_tok_embd, rows);
    }
    emb = ggml_reshape_2d(ctx0, emb, hparams.ple_head_dim * n_heads, n_tokens);
    cb(emb, "ple_embd", -1);

    return emb;
}
```

**Gotchas**
- **`to_float`'s contract.**
  - The signature is `void (*)(const void * x, float * y, int64_t k)` (`ggml/include/ggml.h:3055`).
  - For Q8_0 it is `dequantize_row_q8_0` (`ggml/src/ggml.c:737-744`). That function asserts `k % 32 == 0`,
    and loops over `k / 32` contiguous blocks (`ggml/src/ggml-quants.c:553-567`).
  - `ne0 = 160` is 5 blocks, and the backend's open refuses a geometry whose `ne0` is not whole blocks
    (`lazy-table.cpp:103-157`). So one call over the whole run of rows is valid.
- **Thread safety and initialisation.**
  - `ggml_get_type_traits` returns a pointer into the static const `type_traits` table (`ggml/src/ggml.c:948-952`).
  - The Q8_0 conversion only reads its input and writes its output.
  - It widens `d` with the inline `ggml_compute_fp16_to_fp32` (`ggml/src/ggml-impl.h:396`, `:448`), which reads
    no table.
  - So it needs no `ggml_init`, no `ggml_cpu_init`, and no ggml context. (`ggml_cpu_init` sets up the CPU
    backend's own traits, which are not used here.)
- **Shape.** The F32 input `[ne0, n_heads x n_tokens]` has exactly the shape and layout the eager `get_rows`
  result has, so the reshape and everything downstream are unchanged.
- **Size.** It is `ne0 x n_heads x 4` B per token: 10 KiB per token, so 5 MiB at a 512-token ubatch.
- **No `rows` in STORAGE mode.** `set_input` still builds `idx` (the hashed ids), passes it to `gather`, and never
  touches `rows`.
- **No throw escapes `set_input`.** The new code's allocations are caught, and `gather` is a C call. The
  existing `std::vector<int32_t> idx` and `ctx` allocations are unchanged.
- **The table tensor.** `model.per_layer_tok_embd` is the STORAGE context's tensor: its `data` is null and it has
  no buffer. Its only readers in `src/` are:
  - `qwen4exp.cpp:193-194`, which creates it;
  - the `get_rows` above, now off this path;
  - `llama-model-saver.cpp:492`, refused by T11.
- **Which loads change.** AUTO makes the row STORAGE only with the taker. A CPU-only or non-SYCL load keeps
  LAZY_READ, as today. Gemma4 is unchanged, and T7's gemma4 pass is the witness.

**Step 3, lead.** The E2E (T14) is this task's acceptance.

**Commit:** `llama: qwen4exp reads its PLE rows from the storage tier as f32 in set_input`.
Stage `src/models/qwen4exp.cpp src/llama-lazy-range-record.cpp tests/test-llama-lazy-range-prefill.cpp tests/test-llama-lazy-range-record-source.py`.

---

## T14: End-to-End Validation (lead only)

**Every GPU command follows these rules:**
- Source oneAPI.
- Hold `GPU.lock`, and wait for `Shmem < 30 GB` and `MemAvailable > 150 GB` before touching a device.
- Pin the selector, and set `GGML_SYCL_TUNING_CACHE=0`.
- Sample `grep -E '^(Shmem|MemAvailable):' /proc/meminfo` before the command and about 5 s after it.
- Run one command at a time, with the dmem.current VRAM tenant preflight before the first device binary.
- Before the first gate, check `grep -E '^GGML_SYCL:' build/CMakeCache.txt` and
  `ldd build/bin/llama-completion | grep -cE 'libggml-sycl|libsycl'`.
- Record each arm's start and end time, for row 12.

The model is
`QWEN=/models/Qwen3.8-Flash-Next-GGUF/exps-IQ4_NL/Qwen3.8-Flash-Next-exps-IQ4_NL-00001-of-00006.gguf`.
"Base" means an `ec08e5907` build in an evidence worktree.

| # | check | command (B70 = `level_zero:0`, B50 = `level_zero:1`) | pass |
|---|---|---|---|
| 1 | device tests | `ctest --test-dir build -R '^test-sycl-lazy-(residency-plan\|plan-interface)-1dev$' --output-on-failure -j 1`, after `-N` lists 2 | all arms pass, including plan-interface (g) and (h) and residency-plan (j), (k) and (l) |
| 2 | engagement | `ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_TUNING_CACHE=0 /usr/bin/time -v timeout 900 ./build/bin/llama-cli -m $QWEN -st --simple-io --no-display-prompt --reasoning-budget 0 -p 'Count from 1 to 5. Answer with only: 1, 2, 3, 4, 5' -n 48 --seed 42 --temp 0 > storage.log 2>&1` | `[PLACEMENT] lazy per_layer_token_embd.weight residency=STORAGE` is present, with `resident=` at most 2147483648 |
| 3 | host ledger | in the same log, the `[HOST-PLAN] lazy` line | `storage=1 host=0`, and `lazy_resident=` is the cache only, at most 2 GiB |
| 4 | pinned pool and RSS | the `[SYCL] Pinned pool pre_allocate` line and `Maximum resident set size` of row 2, against the same command with `-lzm off` | neither rises by tens of GiB over the `-lzm off` run, and the pinned pool stays within 2.5 GiB of A11's 2,561.5 MB |
| 5 | counters | row 2's log, at exit | `[LAZY-STORAGE] per_layer_token_embd.weight hits=H misses=M evictions=E submitted=S`, with `M > 0`, `S > 0` and `S <= M`. `submitted=` counts waiters, not disk reads |
| 6 | count gate | `cat storage.log \| grep -cx '1, 2, 3, 4, 5'`, plus the negative control `echo '> Count from 1 to 5. Answer with only: 1, 2, 3, 4, 5' \| grep -cx '1, 2, 3, 4, 5'` | 1 and 0 |
| 7 | token identity | `llama-completion -m $QWEN -p 'The capital of France is' -n 64 --seed 42 --temp 0 -no-cnv` twice, once by default and once with `-lzm off`, each with `-v` to a log and stdout to its own file, then `diff` the two stdouts | **Expected identical** when the eager run's `get_rows` on the table runs on the CPU, which is exact f32, the same arithmetic as the host dequant. **A difference is not an automatic fail.** The lead records which backend ran the eager `get_rows`, read from the `-v` log, and the first diverging token. Rows 6, 10 and 11 decide. |
| 8 | performance | `ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_TUNING_CACHE=0 timeout 900 ./build/bin/llama-bench -m $QWEN -p 2048 -n 0 -r 3 -lzm auto,off` | the STORAGE pp2048 is reported beside the eager one (today 31-38 t/s). Record the ratio. It is not a gate |
| 9 | failure path | not run on a real model. T12's test and pin cover it | — |
| 10 | Mistral regression | `ONEAPI_DEVICE_SELECTOR=level_zero:1 GGML_SYCL_TUNING_CACHE=0 timeout 900 ./build/bin/llama-bench -m /models/mistral-7b-v0.1.Q4_0.gguf -p 0 -n 4 -r 1 -v`, then the Mistral completion gate on `level_zero:1` | the `[HOST-LEDGER] ... plan` terms equal base: `weight=67108864 kv=67108864 staging=233914368 scratch=141074432 runtime=150470656 rounding=3635290240 host_planned_bytes=4294967424`. The pinned pool working set is 485.6 MB and `[HOST-ARENA] Committed` is 4096.0 MB. The gate output starts `1, 2, 3, 4, 5, 6, 7, 8, 9, 10` |
| 11 | GPT-OSS regression | the same `llama-bench -p 0 -n 4 -r 1 -v` with `/models/gpt-oss-20b-mxfp4.gguf` on `level_zero:1`, then the GPT-OSS count gate on `level_zero:1` | the plan terms equal base: `staging=1249601536 scratch=789888512 runtime=36126720 rounding=4232616640 host_planned_bytes=6442451136`. `weight=` and `kv=` equal the base capture taken alongside. The pinned pool working set is 2073.0 MB and `[HOST-ARENA] Committed` is 6144.0 MB. The gate scores `grep -cx` = 1 |
| 12 | GPU health | `journalctl -k --since "<first arm start>" --no-pager \| grep -iE 'GT reset\|guc_id\|CAT error'` | list each matching line's time and PCI id against the arm times. **Pass: no line on the card an arm was using, during that arm.** An unidentified tenant has been faulting the B50 (`0000:09:00.0`) today, so lines outside the arms, or on another card, are recorded and do not fail the row |

Rows 10 and 11 compare the planned terms, the pinned pool working set, and the HOST-ARENA commit. They never
compare whole lines, because those lines also carry the sampled `MemAvailable` and the budget, which vary
from run to run.

**Observed success:**
- Rows 1-6 pass on one build.
- Row 7 passes or is recorded as R1 describes.
- Row 8 is recorded in the merge message with both figures.
- Rows 10-12 pass.

If row 8 shows the STORAGE run far below the eager one, report the row 5 counters line with it. The hit rate
decides whether the follow-up is lookahead or a larger cache, and that decision is the owner's.

---

## Open questions

- **Q1. Is a materialization accepted after `model_load_end`? Answered, yes.**
  - The published plan's record holds the IO-pool lease (`unified-cache.hpp:2447-2449`).
  - The design makes a model's ring at its ensure, after its load ends (`ggml-sycl.cpp:14119-14123`).
  - The ensure accepts the plan's charge for the transaction that created the ring
    (`lazy-io-pool.cpp:1742-1746`).
  - The row cache's ledger uses the transaction only to skip yielding to its own load.
  - T6 keeps its STOP clause as a guard.
- **Q2. The multi-device plan shape. Open.** It blocks only the opt-in `GGML_SYCL_MOE_MULTI_GPU` path and the
  `-2dev` registrations, not the B70 default.
  - `lifecycle_find_placement_plan` returns one snapshot.
  - With a plan per device, `caches_carving` exceeds 1.
  - The `-2dev` runs of rows 1's tests show whether the open finds its entry there.
- **Q3. The model saver. Resolved by T11.** The saver refuses a STORAGE model, naming the tensor. Streaming
  the table into a save is not planned.
- **Q4. Graph reuse. Answered.** A reused graph still runs `set_inputs`: the reuse check is at
  `llama-context.cpp:5323` and `set_inputs` at `:5367`. If an upstream merge ever skips `set_inputs` on
  reuse, the F32 input would go stale.
- **Q5. The backend that runs `get_rows`. Answered by R1.** STORAGE mode has no `get_rows`. The host dequant
  in f32 matches the eager CPU path's arithmetic. Row 7 records the case where the eager path ran elsewhere.
- **Q6. The ticket lock and the witness tests. Answered.** `witnessed-mutex.hpp:3-14` records only
  `witnessed_mutex` instances, so T4's plain `std::mutex` `done_lock` is invisible to the witness. Delivery
  never runs under the engine mutex (`lazy-io-pool.cpp:1598`, `:1898`).
- **Q7. What `submitted=` counts. Stands.** It counts waiters submitted, not disk reads, and the log line
  says so by its name. A disk-read count would need the engine to report dispatches per context. That is a
  separate change.

## Follow-ups left out

- **Branch-path resolution.** Use `getxattr user.mergerfs.fullpath` to open the branch file directly. S1
  measured a 0.54 ms median against 0.66 ms.
- **Lookahead.** Hash and submit ubatch N+1's rows while ubatch N computes. The `lazy_io_class::LOOKAHEAD` class
  exists for this.
- **Gathering into the input's buffer.** When the input's buffer is host memory, gather and convert directly
  into it, which would save one copy.
- **gemma4.** Its table keeps EAGER under AUTO, and the ON refusal, until it has its own graph consumer.
- **DeepSeek.** The engram tables keep HOST-if-fits, with no `storage_only`. Their consumer is the parent
  plan's in-graph form.
- **Redistribution.** A capped table's unused share is not given to other tables. Revisit this if a model
  ever carries two STORAGE tables.
- **A stale comment.** `get-rows-support.hpp:23` calls `per_layer_token_embd` IQ4_NL, but the table is Q8_0.
  Fix the comment in its own commit.
