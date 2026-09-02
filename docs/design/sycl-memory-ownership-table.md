# SYCL memory ownership table: model-scoped vs. context-scoped

Task: `llama.cpp-32dg8.15.1` ("[32dg8-P1] Prove model-scoped vs context-scoped
memory ownership"), epic `llama.cpp-rg2ft`. Answers the epic's governing
question — can one loaded model share process-scoped weight/cache state while
multiple contexts keep KV, runtime, graph, and scratch memory isolated — with a
concrete inventory rather than a guess, so every later fix in the epic is
judged against this table instead of re-deriving it.

All citations are against worktree `/Apps/llama.cpp-wt-handles` at
`fed0b58e237cedc99db3c98e2d704b1f26390f5e` (`git -C /Apps/llama.cpp-wt-handles
rev-parse HEAD`). Line numbers drift with the tree; re-grep the cited symbol if
a line looks stale.

**Landed inputs this table builds on, not open questions:**
- `llama.cpp-1q72` (merged `4bd4211e8`, "merge: add SYCL execution lifecycle
  foundation") — added `ggml_sycl::execution::Registry`
  (`ggml/src/ggml-sycl/execution-lifecycle.hpp/.cpp`), which mints one
  `ContextId` per `llama_context` (`src/llama-context.h:366`,
  `sycl_exec_context`) and tracks session/graph-epoch/invocation state keyed by
  that id.
- `llama.cpp-o6jx` (merged `606e252b0`, "merge: owner-targeted SYCL
  context/model teardown") — added the owner-token-matched drain sequence
  (begin-drain → unlocked terminal wait → locked extract → unlocked destroy →
  finish_drain, `ggml-sycl.cpp:14395` `ggml_backend_sycl_execution_context_begin_drain`
  onward) and per-slot model teardown
  (`ggml_sycl_release_model_slot_resources`, `ggml-sycl.cpp:11745`).

**Ruling 2** (mem_handle is the sole ownership/lifetime token) and **ruling 3**
(no reset/reap as a fix for unsafe global state — propose the missing scoping
key instead) govern every row below. Where a global lacks a
model_id/context_id/slot_id/device_id key, the "Risk" column states the missing
key, and either a follow-up bead is filed (§7) or the row is explicitly
deferred to orchestrator triage with its own stated reason (§7.4) — never a
proposed reset.

## Legend

- **Scope**: `process` (one instance for the whole backend, no partition key),
  `device` (one instance per SYCL device index — an allowed key per this
  task), `model` (partitioned by `lifecycle::ModelToken`/`model_id`/slot),
  `context` (partitioned by `ggml_sycl_exec_context_id`/`llama_context`),
  `graph-epoch` (partitioned by `GraphEpoch`, nested under a context/session),
  `thread` (thread-local).
- **Reset/teardown owner**: which call path is allowed to mutate or free the
  row to zero.

## 1. Placement plan

| Object | Scope | Key fields | Reset/teardown owner | Lifetime | Risk |
|---|---|---|---|---|---|
| `g_lifecycle_plan_models` | model | `model_id -> load_txn_id -> shared_ptr<const lifecycle_plan_snapshot>` | `unified-cache.cpp:433` erases `g_lifecycle_plan_models.erase(model)` at model unload | One entry per (model, load transaction); survives across contexts of the same model | None — already scoped by model_id + load_txn_id. Two models loaded at once get disjoint sub-maps. |
| `g_lifecycle_plan_candidates` | process, keyed by `load_txn_id` | `unordered_map<uint64_t, shared_ptr<const lifecycle_plan_snapshot>>` (`unified-cache.cpp:124`) | Erased at candidate promotion/failure (`unified-cache.cpp:202`, `:243`) | Transient: only lives during one load's plan-build window | None — `load_txn_id` is itself a per-load key; not a durable global. |
| `placement_plan` struct itself | N/A (value type) | passed by reference through planner functions (`unified-cache.hpp:531`) | Owned by the `lifecycle_plan_snapshot` that wraps it | Matches the snapshot's lifetime | N/A — not a global. |

## 2. `unified_cache` instances and zones

| Object | Scope | Key fields | Reset/teardown owner | Lifetime | Risk |
|---|---|---|---|---|---|
| `g_device_caches` | device | `unordered_map<int, shared_ptr<unified_cache>>` (`unified-cache.cpp:1288`) | Backend shutdown / device teardown | One cache instance per SYCL device for the process | None — device is an allowed key; every VRAM allocation on that device flows through this one instance by design. |
| `vram_zone_id` zones (`KV`, `WEIGHT`, `ONEDNN`, `RUNTIME`, `SCRATCH`) | device (member of the per-device `unified_cache`) | `vram_zone{start,size,used,allocator}` array, `unified-cache.hpp:71-88` | `zone_settle(vram_zone_id)` (`unified-cache.cpp:19843`), refuses when live registered allocations exist | Sized once at cache construction; sub-allocations come/go via TLSF | KV and RUNTIME zones are **not** further partitioned by context — see §5. WEIGHT already ownership-tracked (§3). |
| `host_zone_id` zones (`WEIGHT`, `KV`, `STAGING`, `SCRATCH`) | device (member of the per-device `unified_cache`) | `host_zone_reserved_bytes_[COUNT]` (`unified-cache.hpp:3413`), `pinned-pool.hpp:62-67` | `host_zone_settle(host_zone_id)` (`unified-cache.cpp:16943`), same live-allocation refusal as `zone_settle` | Same as VRAM zones, host-pinned tier | Same as VRAM KV/RUNTIME above. |
| `zone_settle` / `host_zone_settle` refusal guard | device | live-allocation count checked under `rw_mutex_` before settling | Refuses (does not reset) when any allocation with a live handle exists (`unified-cache.cpp:16988` cross-refs the WEIGHT/KV refusal family) | N/A — a guard, not stored state | Working as intended per ruling 3: **do not weaken this guard**; it is the safety net for the missing context key noted above, not a bug. |
| `arena_reserve(queue, budget_bytes, ...)` reclaim path | device (per-cache) | Declared `unified-cache.hpp:2598`, defined `unified-cache.cpp:19105` | On a second-and-later call for an already-constructed arena (`arena_base_` already set — same model, new context) calls `zone_reclaim(vram_zone_id::KV)` and `zone_reclaim(vram_zone_id::RUNTIME)` (`unified-cache.cpp:19122-19123`) plus `host_zone_reclaim(host_zone_id::KV)` and `host_zone_boundary_check(host_zone_id::STAGING)` (`:19133-19134`); `zone_reclaim` is a thin, precondition-narrowed wrapper over `zone_settle` (`unified-cache.cpp:20095-20099`, `GGML_ASSERT`s KV/RUNTIME only), so it inherits the same live-allocation refusal rather than bypassing it | Runs at the graph-compute / context-switch boundary named by the RUNTIME row in canonical doc §5.2 ("arena_reserve at graph compute boundary") | Same missing key as the `zone_settle`/`host_zone_settle` row above: a genuine bulk reclaim, but still device-scoped only, not context-keyed. This is the concrete mechanism behind both the §5.2 RUNTIME row and the §5.3 zone_settle/host_zone_settle refusal this document already cross-references. |

## 3. Weight cache entries and leases

| Object | Scope | Key fields | Reset/teardown owner | Lifetime | Risk |
|---|---|---|---|---|---|
| `unified_cache::entries_` | device (per-cache `entry_map`, `unified-cache.hpp:3387`) | keyed by `(identity, type, layer/expert)`; entry has `owner_mask`, `owner_tagged`, `in_use_count` (`unified-cache.hpp:1537-1830`) | `reclaim_weight_entries(weight_reclaim_mode, slot)` (declared `unified-cache.hpp:3403`; three modes `LOAD_BOUNDARY`/`MID_LOAD_REPLAN`/`MODEL_TEARDOWN` enum'd at `unified-cache.hpp:1709-1722`) | One entry per resident weight; survives across contexts of the owning model(s) | `owner_mask` is a **32-bit bitmask** (`MODEL_SLOT_COUNT = 32`, `unified-cache.hpp:1705`) — capacity-bounded, not a missing key. Beyond 32 concurrently-attributed models an entry reads `owner_tagged=false` and is conservatively never reclaimed while any model is live (documented behavior, not a defect). |
| `live_model_mask_` | device (per-cache) | `uint32_t`, one bit per live model slot, `unified-cache.hpp:3393` | Set/cleared at model attach/detach under `rw_mutex_` | Lives for the cache instance | None — guarded by the same lock as `entries_`, so reclaim decisions and liveness cannot disagree (comment at declaration). |
| `live_buffer_owners_` | device (per-cache) | `unordered_set<uint64_t>` of tagged buffer-owner ids, `unified-cache.hpp:3398` | Same guard as `live_model_mask_` | Lives for the cache instance | None — same reasoning as above. |
| `g_sycl_weight_identities_by_name` | model+load+slot+generation (owner-scoped) | `unordered_map<string, ggml_sycl_weight_identity>` (`ggml-sycl.cpp:10887`); key is `ggml_sycl_owner_name_key(owner, name)` = `"model:load:slot:generation:name"` (`ggml-sycl.cpp:10289`) | Erased per-owner at teardown (`ggml-sycl.cpp:11670-11674` filters by `owner.model.value`/`owner.load.value`/slot/generation) | Per (model, load, slot, generation, tensor name) | None — already scoped; explicitly named in the task text but the key composition already includes model_id, load_txn_id, slot and generation. |
| `g_sycl_host_weight_extras` | model+load+slot+generation (owner-scoped) | `unordered_map<string, sycl_host_weight_extra_entry>` (`ggml-sycl.cpp:10287`), same key scheme as above (`ggml-sycl.cpp:10277-10285`) | Model-slot teardown drops exactly its own rows (comment at struct decl, `ggml-sycl.cpp:10282-10284`) | Same as above | None — already scoped. |
| `g_sycl_buffer_owners` | model (`uint64_t model_id -> lifecycle::ModelToken`, `ggml-sycl.cpp:2351`) | `ggml_sycl_lookup_buffer_owner`/`ggml_sycl_exact_wrapper_owner` (`ggml-sycl.cpp:2395-2461`) | `g_sycl_buffer_owners.erase(owner.model.value)` at buffer-owner teardown (`ggml-sycl.cpp:2406`) | Per attributed buffer id | None — already scoped by model_id. |
| `moe_mmid_workspace_registry` | model+plan (`moe_mmid_model_token`, `moe-mmid-workspace.hpp:545-590`) | keyed internally by `(token, plan_identity, submit_device, owner_device)` | `invalidate_plan(plan)` (`moe-mmid-workspace.hpp:580`) / `retire(token, plan_identity)` (`moe-mmid-workspace.hpp:583`) | Per (model, plan) | None — already scoped. |

## 4. Tiered KV manager

| Object | Scope | Key fields | Reset/teardown owner | Lifetime | Risk |
|---|---|---|---|---|---|
| `g_kv_tier_managers` | **device only** | `std::array<kv_tier_manager, GGML_SYCL_MAX_DEVICES>` (`kv-tier-manager.cpp:15`); `get_kv_tier_manager(int device)` returns a reference into this array (`kv-tier-manager.cpp:17-19`) | Overwritten in place by the next `configure()`/`configure_from_plan()`/`configure_with_weights()` call for that device — there is no reset call, each configure call is destructive | Lives for the process; state is whatever the **most recent** caller wrote | **Missing key: context_id/session_id.** `mgr.configure_from_plan(device, *kv_plan, n_layers, kv_slice)` and `mgr.configure_with_weights(...)` are called from KV-buffer allocation (`ggml-sycl.cpp:35711`, `:35865`, `:35867`) with only a `device` argument. Two `llama_context`s on the same device with different `n_ctx`/hot-layer geometry silently overwrite each other's hot/cold layer placement (`layer_on_device_`, `per_layer_kv_bytes_`) — see follow-up bead §7.1. |

## 5. Graph replay state

| Object | Scope | Key fields | Reset/teardown owner | Lifetime | Risk |
|---|---|---|---|---|---|
| `ggml_sycl::execution::Registry` (`contexts_`) | context (`ggml_sycl::execution::ContextId`, `execution-lifecycle.hpp:485`) | `unordered_map<uint64_t, context_entry>`; each `context_entry` nests one `session_entry` with `epochs` (`GraphEpoch -> persistent_epoch_entry`) and `graph` (the compatibility outer graph), `execution-lifecycle.hpp:414-436` | Owner-targeted drain (`llama.cpp-o6jx`): `begin_drain`→`begin_drain_extract`→`finish_drain`, `close_context_if_idle` (`execution-lifecycle.hpp:353-359`) | Per `llama_context`, from `create_context()` (`execution-lifecycle.cpp:170`) to drain/close | None for this registry itself — it is the landed `1q72` fix: correctly context-keyed. |
| `Registry::device_owners_` | device (`array<device_owner, max_devices>`, `execution-lifecycle.hpp:486`) | Records which `(context, session, reset_epoch, graph_epoch, invocation, token_root)` currently holds device `d`'s exclusive execution claim | Set at `begin_invocation`, cleared implicitly when a new owner claims the device (`execution-lifecycle.cpp:1141-1143`) | Per device, one owner at a time | None as a data structure — it is the exclusivity mechanism, not a leak; `device_owner_context()` (`execution-lifecycle.hpp:377`) is the read-only reverse lookup used to check this safely. |
| Per-context retained handles: `ctx->graph_retained_handles` | context (`std::vector<ggml_sycl::mem_handle>` member of `ggml_backend_sycl_context`, `ggml-sycl.cpp:39820`) | Populated during graph compute (`ggml-sycl.cpp:53589`, `:53592`), cleared via `ctx->graph_retained_handles.clear()` at graph teardown (`ggml-sycl.cpp:93263`) | The owning context's graph-boundary path | Per graph epoch within a context | None as a vector — properly context-scoped. |
| Retained-handle sink pointer: `g_graph_recording_sink` / `g_graph_retained_handle_sink` | **thread** (`thread_local graph_recording_sink_state`, `mem-handle.cpp:69-76`) | Points at whichever context's `graph_retained_handles` vector the calling thread is currently recording into; set/cleared around each `set_graph_retained_handle_sink()` call (`ggml-sycl.cpp:88149`, `:88168`, `:90141`, `:90188`, `:91074`, `:91098`, ...) | Manual save/restore discipline at each call site — no destructor-based RAII guard | Only valid for the duration of one recording scope on one thread | A thread that is **not** currently recording (sink == nullptr) has its handles fall through to `graph_unwaitable` below — this is the confirmed root cause of an existing open bug. |
| `g_retained_handles_state->graph_unwaitable` | **process** (`std::vector<mem_handle>` inside a process-global `retained_handle_state`, `mem-handle.cpp:57,64`) | Populated by (a) `retain_handles_for_current_graph()` when no thread-local sink is installed (`mem-handle.cpp:108-124`), and (b) the drain worker's "command graph" exception catch, which receives handles from whichever context originally submitted the event on the worker thread (`mem-handle.cpp:150-161`) | `release_graph_retained_handles()` (`mem-handle.cpp:2060-2070`) swaps the **entire** vector, unconditionally, from whichever context's per-context teardown path (`sycl_exec_graph_release_pool_retained`, `ggml-sycl.cpp:93259-93264`) happens to call it next | Indeterminate — handles for context A can sit here until context B's next graph invalidation frees them | **Confirmed missing key: context/owning-invocation id.** Already tracked as `llama.cpp-mhyw` ("release_graph_retained_handles() frees other contexts' leases from a per-context path", open, epic `llama.cpp-rg2ft`) — **no new bead filed here**; this table cites it as the concrete evidence for the "no host waits + thread-local sink" risk in the canonical doc. `mhyw`'s own analysis (comment `c-zrg9`/`c-ro5p`) warns against a naive per-context split: handles arriving via the drain-worker exception catch carry no context attribution today, so the fix needs to add that attribution, not just re-key the existing vector. |

## 6. Other process-wide state found during the sweep

| Object | Scope | Key fields | Reset/teardown owner | Lifetime | Risk |
|---|---|---|---|---|---|
| `g_graph_compute_active` | **process** (`std::atomic<bool>`, `unified-cache.cpp:719`) | Toggled by `unified_cache_set_graph_compute_active(bool)` (`unified-cache.cpp:13899`, declared `unified-cache.hpp:4710`) | Set/cleared around each graph-compute call, all devices | Process-wide eviction guard | **Missing key: device_id.** Documented already in `docs/design/sycl-canonical-memory-architecture.md` §5.3: "has no device argument and sets the process-global `g_graph_compute_active` eviction guard... not per-device or per-context state." Two contexts computing on different devices both gate eviction through one flag — a compute on device 0 suppresses eviction on device 1 too (conservative, not corrupting, but see follow-up bead §7.2). |
| `g_sycl_graph_compute_mutex` | process (`std::mutex`, `ggml-sycl.cpp:751`) | The only acquisition site is `ggml-sycl.cpp:99095` (`std::unique_lock<std::mutex> global_graph_lock(g_sycl_graph_compute_mutex)`) | N/A — a submission-ordering primitive, not stored ownership state | N/A | Not a missing-key case: this is a coordination primitive, not an ownership record, and the canonical doc §5.3 already states precisely which paths release it before `compute_impl` submission (direct/fallback) vs. hold it through submission (persistent-TG, deferred-copy, command-graph record/replay). No bead — narrowing this statement is exactly what `docs/design/sycl-canonical-memory-architecture.md` already does; this table does not relitigate it. |
| TP (tensor-parallel) per-layer caches: `g_tp_ffn_norm_cache`, `g_tp_ffn_inputs`, `g_tp_ffn_weights`, `g_tp_async_ffn_jobs`, `g_tp_attn_inputs`, `g_tp_attn_weights`, `g_tp_async_attn_jobs`, `g_tp_ffn_buffers`, `g_tp_attn_buffers` (9 distinct layer-keyed `unordered_map<int, ...>`s) plus `g_tp_host_staging` (a single unkeyed struct — see Risk) | **process**; the 9 maps keyed only by layer number (`int`), `g_tp_host_staging` has no key at all | `unordered_map<int, ...>` + a dedicated `std::mutex` each: `g_tp_ffn_norm_cache`/`_mutex` (`common.hpp:2370-2371`), `g_tp_ffn_inputs`/`g_tp_ffn_input_mutex` (`:2403-2404`), `g_tp_ffn_weights`/`g_tp_ffn_weight_mutex` (`:2413-2414`), `g_tp_async_ffn_jobs`/`g_tp_async_ffn_mutex` (`:2462-2463`), `g_tp_attn_inputs`/`g_tp_attn_input_mutex` (`:2431-2432`), `g_tp_attn_weights`/`g_tp_attn_weight_mutex` (`:2442-2443`), `g_tp_async_attn_jobs`/`g_tp_async_attn_mutex` (`:2481-2482`), `g_tp_ffn_buffers`/`_mutex` (`:2602-2603`), `g_tp_attn_buffers`/`_mutex` (`:2680-2681`); `g_tp_host_staging`/`g_tp_host_staging_mutex` is a plain `struct tp_host_staging_buffer` instance, not a map (`:2705-2716`) | No reset path found beyond overwrite-in-place at the next forward pass (`g_tp_current_pass_id`, `common.hpp:2372`, increments but does not gate map/struct contents by pass) | Process-wide, indexed only by transformer layer index (`g_tp_host_staging`: not indexed at all — one buffer for the whole process) | **Missing key: model_id/context_id.** Two concurrently-loaded models (or two contexts of the same model) using tensor-parallel dispatch would read/write the same layer-`N` FFN/attention intermediate buffers. `g_tp_host_staging` is the strictly worse case: it is not layer-keyed either, so even two different layers of the *same* concurrent TP dispatch would share one staging buffer. See follow-up bead §7.3 (matches the description of this global already in `llama.cpp-mgi7`). |
| `g_runtime_alloc_registry` | process, keyed by `void *` (`unordered_map<void*, runtime_alloc_record>`, `unified-cache.cpp:1283`) | `runtime_alloc_record` carries `sycl::queue * queue` (device-derivable) but the map key is the raw allocation pointer | Erased at `unified_free`/zone_free call sites | Per live runtime allocation | Keyed on a raw pointer, which is the pattern the SYCL Memory Ownership rules in `CLAUDE.md` restrict ("pointer tables... must be derived from the stable identity/hash carried by `mem_handle`, not from raw device addresses"). This is pre-existing, already-allowlisted migration debt (the doc explicitly keeps `alloc_handle` as "a migration state, not a third surface") rather than a newly-discovered model/context/device gap. **Covering bead: `llama.cpp-mg359`** ("Eliminate all pointer resolution bypasses — route everything through smart handles", open, epic `llama.cpp-rg2ft`) — its Problem section explicitly lists "cache maps or pointer tables acting as if they own placement identity" as an in-scope bypass category, and its acceptance bar ("`mem_handle` can be used directly as a hash key for backing allocation identity, with stable equality for aliases to the same memory") is exactly the property this map lacks; not filing a duplicate bead. |
| `g_offload_pool_slots` / `g_offload_pool_free` | device (via `offload_pool_key{device,...}`, `unified-cache.cpp:1406-1439`) | `g_offload_pool_free` is keyed by `offload_pool_key` (includes `device`) — correctly device-scoped. `g_offload_pool_slots` is keyed by `void*` for the in-use lookup | Freed back to the pool at `sycl_free`-equivalent release sites | Per pool slot | Same pointer-key caveat as above for the in-use side; the free-list side is already device-scoped. **Covering bead: `llama.cpp-mg359`**, same reasoning as `g_runtime_alloc_registry` above; not filing a duplicate bead. |
| `ggml_tensor_extra_gpu` (`data_device[]`, `data_handle[]`, `model_id`) | **model** (one struct per tensor, tensors belong to one model's graph) × **device** (fixed-size per-device arrays) | `common.hpp:3358-3365`; `set_data_device_handle()` (`common.hpp:3467`), `set_data_device()` (`common.hpp:3442`) | Struct lifetime follows the owning tensor/model; `refcount` field (`common.hpp:3359`) plus `retain_extra_gpu`/`release_extra_gpu` (`common.hpp:4113-4114`) | Per tensor, per model | None — already scoped by (model tensor identity, device index); this is the struct the task explicitly asked to be covered, and it is correctly keyed. |
| `g_placement_publication` | model, as an **atomic published snapshot pointer** (`shared_ptr<const lifecycle_plan_snapshot>`, `ggml-sycl.cpp:2264`), not a plain mutable global | Published via `std::atomic_store_explicit(..., memory_order_release)` at `ggml-sycl.cpp:14822`; read via `std::atomic_load_explicit(..., memory_order_acquire)` at `:2303`, `:2523`, `:2544` | Overwritten by the next model publish; nothing erases it explicitly | A reader that captured the `shared_ptr` before a new publish keeps using the OLD snapshot after a newer one lands — the point of the atomic-shared_ptr design (canonical doc §5.1) | None for a single active model; a captured snapshot protects an in-flight reader. **Not model-id partitioned**: with two models loaded, whichever publish happened last is what every *fresh* (non-captured) read sees, so an unqualified reader cannot tell which model plan it is looking at. No new bead filed here; flagged for orchestrator triage. |
| `g_tensor_inventory` | process, plain mutated global (`vector<pair<string,size_t>>`, `ggml-sycl.cpp:14142`) | Cleared/repopulated at model-load time: clear at `:14220` and `:15360`, repopulated `:15373-15381` (the `emplace_back` itself is `:15381`) | Model-load path only; **no** snapshot indirection like `g_placement_publication` | Whatever the most recent model load wrote | **Missing key: model_id**, and unlike `g_placement_publication` there is no publish/snapshot protection — a second model load overwrites this in place while an in-flight reader over the first model view has no captured-snapshot escape. No new bead filed; flagged for orchestrator triage. |
| `g_placement_kv_info` | process, plain mutated global (`ggml_sycl::placement_kv_info`, `ggml-sycl.cpp:14158`) | Cleared at `:14236`, populated during model load (same load path as `g_tensor_inventory`) | Model-load path only | Whatever the most recent model load wrote | Same pattern as `g_tensor_inventory`: missing model_id key, no snapshot protection. No new bead filed; flagged for orchestrator triage. |
| `g_model_n_layer` | process, plain mutated global (`uint32_t`, `ggml-sycl.cpp:14157`) | Set from `publication.model_n_layer` at `:14815` (mirrors the snapshot at publish time), also written independently at `:15440`, reset to 0 at `:14235`/`:15576` | Model-load / model-unload path | Whatever the most recent model load wrote | Same missing-model_id-key pattern as the two rows above; it mirrors `g_placement_publication` at publish time but is a separate plain global, so it can be read stale or torn independent of the snapshot pointer atomicity. No new bead filed; flagged for orchestrator triage. |
| `g_layer_on_cpu` | process (despite canonical doc §5.2 listing it under "context/slot-scoped state" — see Risk) | `static std::vector<bool>` (`ggml-sycl.cpp:15328`) | Rebuilt from scratch at each graph build via `.assign()`/per-layer writes (`:76874`, `:76925`, `:76944`), guarded by a dedicated mutex during incremental classification (`:76892`) | Only valid for the duration of the graph build that most recently wrote it | **No model_id/context_id key.** The "recomputed at each graph build" reset-boundary text for this global in canonical doc §5.2 describes WHEN it resets, not what key protects it between two contexts overlapping graph builds; the global itself is process-wide. Safe today only because of the documented single-active-context-per-device constraint (canonical doc §5.3); becomes a correctness hazard the moment that constraint is lifted. No new bead filed here — tracked as part of the same "context-keyed KV/RUNTIME arena reservation" gap canonical doc §5.3 already names as open; flagged for orchestrator triage. |
| `g_sycl_weight_identities_unowned` | process, keyed by **bare tensor name only**, no owner/model_id component (`unordered_map<string, ggml_sycl_weight_identity>`, `ggml-sycl.cpp:10913`) | Write site `:13511` inside the `:13490-13511` block (see Risk for the exact write-gating condition); read at `:13988-13989` (gated on an unresolved owner) and `:23978` (unconditional in its function, itself the guard) | **Never erased** — verified: exactly one write site, two read sites, zero erase sites in the whole file | Indefinite — entries persist for the process lifetime once written | Last-write-wins on name collision, zero owner scoping — the pattern the SYCL Memory Ownership rules in `CLAUDE.md` restrict, applied here to a metadata table rather than a raw allocation. The in-source design comment (`ggml-sycl.cpp:10895-10911`) and the code agree, read the right way round: the production loader passes `model_id == 0` at its one call site (`src/llama-model-loader.cpp:1370`), and `ggml-sycl.cpp:13493` (`if (model_id == 0) { return; }`) refuses to write when that is true and no load transaction is open — so nothing in a real model load ever writes here. The only way to reach the write at `:13511` is a caller passing a NONZERO `model_id` while no load transaction is open, a combination that does not occur in production, which is exactly what makes it "test-only in practice." That argument is a documented invariant, not something a dedicated test currently verifies. No new bead filed; flagged for orchestrator triage as a candidate for either a regression test of the "test-only in practice" invariant, or removal if genuinely dead in production. |

### Reconciliation with the 2026-04 finding

The original 2026-04-27 "FAIL WITH FIX" comment on this task cited four pieces
of evidence for why multi-context safety could not be proven. Re-checked
against current HEAD, per item:

1. **`unified-cache.cpp:9169-9185`, "resets KV/RUNTIME and host KV/STAGING for
   an existing arena"** — the same logic exists today at
   `unified-cache.cpp:19105-19134` (`arena_reserve`, now rowed in §2 above).
   What changed: `zone_settle`/`host_zone_settle` (which `arena_reserve`'s
   reclaim calls route through) now REFUSE to reset a zone holding live
   registered allocations instead of resetting blindly (`llama.cpp-fz2u`,
   gated by `test-sycl-zone-reset-live-refusal`) — hardening landed in the
   `1q72`/`o6jx` era, after April. What is still open: the reset remains
   device-scoped, not context-keyed, exactly the gap canonical doc §5.3 names
   as the one thing still missing. The refusal makes concurrent misuse fail
   safe instead of failing silent; it does not make two contexts KV/RUNTIME
   coexist by construction.

2. **`ggml-sycl.cpp:6200-6460`, "populates process-global inventory and
   placement state"** — the equivalent globals today are `g_tensor_inventory`
   (`:14142`), `g_placement_kv_info` (`:14158`), `g_model_n_layer` (`:14157`),
   and `g_placement_publication` (`:2264`), all newly rowed in §6 above. What
   changed: `g_placement_publication` was rebuilt on an atomic
   published-immutable-snapshot design (`c3bfd71c4`, `92b2675f6`) so an
   in-flight reader that already captured a `shared_ptr` survives a concurrent
   republish — a real improvement over the mutable `g_placement_plan` the
   canonical doc records as the April-era design. What is still open:
   `g_tensor_inventory`/`g_placement_kv_info`/`g_model_n_layer` remain plain
   mutated globals with no equivalent snapshot indirection, and none of the
   four carry a model_id partition.

3. **`ggml-sycl.cpp:38677+`, "uses global `g_layer_on_cpu` and related layer
   classification state"** — the same global exists today at
   `ggml-sycl.cpp:15328`, populated at `:76874`/`:76925`/`:76944` (line
   numbers moved; the mechanism is unchanged). What changed: nothing
   structural for this specific global. What is still open: everything — this
   is the clearest of the four where the April finding still holds verbatim,
   currently mitigated only by the documented single-active-context-per-device
   constraint (canonical doc §5.2/§5.3), not by any code-level scoping.

4. **`ggml-sycl.cpp:41455-41457`, "resets scratch and host zones at graph
   compute boundaries"** — that citation was already stale in April:
   `:41455-41457` lands on the head of `argsort_f32_i32_sycl`
   (`:41455` is a blank line, `:41456` is the
   `static void argsort_f32_i32_sycl(` signature), unrelated to any zone
   reset.
   The real graph-boundary sites today are
   `ggml_sycl::unified_cache_arena_reset(d)` (`ggml-sycl.cpp:86003`, inside
   `ggml_sycl_graph_boundary_reset_arenas`, which itself calls
   `unified_cache::arena_reset()` -> `zone_boundary_check(vram_zone_id::SCRATCH)`,
   `unified-cache.cpp:17641`) for the VRAM SCRATCH zone, and the paired
   `unified_cache_host_zone_boundary_check(STAGING)` /
   `(SCRATCH)` calls (`ggml-sycl.cpp:86515-86516`) for the host zones. What
   changed: the same hardening as item 1 — these are liveness checkpoints
   against live allocations (refuse, never force-reclaim), not blind resets;
   the canonical doc §5.2 SCRATCH row's citation is corrected to
   `ggml-sycl.cpp:86003` in this same commit. What is still open: the same
   device-only scoping gap.

**Net effect**: none of the four April findings describe a defect that has
since been fixed by adding a missing scoping key — the missing keys are still
missing, and are now tracked as the specific beads in §7 (`c781`, `2mt5`,
`mgi7`) plus the already-open `mhyw`, rather than as one undifferentiated
blocker (the now-superseded `llama.cpp-32dg8.15.10`). What HAS changed since
April is that the reset/settle paths behind all four items now fail closed
(refuse) instead of failing open (silently resetting live memory) — a real
safety improvement, and the reason this round of the task closes
PASS WITH FILED FOLLOW-UPS rather than repeating FAIL.

## 7. Follow-up beads filed

Per ruling 3, each bead below proposes the **missing scoping key**, not a
reset/reap mechanism.

- **7.1 — `llama.cpp-c781`**: `g_kv_tier_managers` (§4) is a device-only
  singleton array reconfigured destructively by every KV-buffer allocation;
  needs a context_id/session_id component in its key (or per-context instances)
  so two contexts sharing a device do not silently overwrite each other's hot/
  cold layer placement.
- **7.2 — `llama.cpp-2mt5`**: `g_graph_compute_active` (§6) and
  `unified_cache_set_graph_compute_active(bool)` (§6) have no device argument;
  needs a `device_id`-keyed variant (e.g. a per-device atomic array) so a
  compute on one device does not gate eviction on an unrelated device.
- **7.3 — `llama.cpp-mgi7`**: the TP per-layer caches (§6,
  `g_tp_ffn_norm_cache` and siblings in `common.hpp:2370-2716`) are keyed only
  by layer number; needs a model_id (or context_id, whichever the TP dispatch
  path is actually scoped to) component added to every one of these maps'
  keys.

All three are filed as children of epic `llama.cpp-rg2ft` via `task_dep` (epic
blocked-by each bead).

**Already tracked, not duplicated**: `llama.cpp-mhyw` (open, epic
`llama.cpp-rg2ft`) already covers the `graph_unwaitable`/
`release_graph_retained_handles()` cross-context release described in §5.
`llama.cpp-mg359` (open, epic `llama.cpp-rg2ft`) already covers the two
raw-`void*`-keyed maps in §6, `g_runtime_alloc_registry` and
`g_offload_pool_slots` (its Problem section names "cache maps or pointer
tables acting as if they own placement identity" as in-scope, and its
acceptance bar is exactly the missing property those two maps lack).

### 7.4 — Deferred to orchestrator triage, no bead filed

These six §6 rows have a missing-key Risk finding but no filed bead, because
each is either a different-shaped fix than "add a scoping key", or is not yet
confirmed to have a live reader that would observe the gap. Per ruling 3 this
is a stated deferral, not a silent gap:

- **`g_placement_publication`**: already an atomic-published-snapshot design;
  the residual risk is unqualified (non-snapshot-capturing) reads racing a
  second model's publish, which is a narrower defect than a missing key and
  may be better fixed by requiring callers to capture the snapshot, not by
  adding a model_id to the pointer itself. Deferred pending a design decision
  on whether unqualified reads should be disallowed outright.
- **`g_tensor_inventory`**: same missing-model_id-key shape as
  `g_placement_kv_info`/`g_model_n_layer` below, but no confirmed concurrent
  reader exists in the current single-active-load codebase. Deferred pending
  confirmation a reader can actually observe the race.
- **`g_placement_kv_info`**: same reasoning as `g_tensor_inventory`.
- **`g_model_n_layer`**: same reasoning as `g_tensor_inventory`; also
  partially mirrors `g_placement_publication` at publish time without sharing
  its snapshot protection.
- **`g_layer_on_cpu`**: **not** a KV/RUNTIME arena reservation gap, despite
  sitting next to §4/§5 rows that are — it is process-global *layer-CPU
  placement classification* consumed during graph build, a different kind of
  state than a VRAM/host zone. "Same reservation gap" would misdescribe it.
  It is safe today only because of the single-active-context-per-device
  constraint (canonical doc §5.3); closing it needs the same foundational
  context-keyed-execution work that constraint depends on, not a standalone
  scoping-key bead. Deferred pending that foundation work, not filed as an
  independent bead so as not to duplicate its scope.
- **`g_sycl_weight_identities_unowned`**: the identified risk is a write path
  that is test-only in practice (§6 Risk cell), not a missing scoping key —
  there is nothing to key, because production never reaches the write. The
  actionable follow-up is a regression test proving that unreachability, or
  removing the path, neither of which is "add a model_id"; deferred pending
  an owner decision on which.

## 8. Two-context proof

See `tests/test-sycl-two-context-ownership.cpp` (registered in
`tests/CMakeLists.txt` as `test-sycl-two-context-ownership`, label `cache`,
selector pinned to `level_zero:1`, fixture `test-download-model`, guarded by
`if (GGML_SYCL)` so a non-SYCL build never registers it). Runtime-skips (exit
77) if `llama_supports_gpu_offload()` is false, so a SYCL build with no GPU
device or a backend init failure that fell back to CPU cannot pass vacuously.
Four-part structure:

1. **References** — a single fresh context decodes each prompt's full script
   (P_A then c1..c4, P_B then c1..c3) before any two-context state exists.
2. **Interleaved proof** — creates two `llama_context` objects with different
   `n_ctx` (256 and 512 under ctest; `--ctx-a`/`--ctx-b` override), keeps both
   alive, and interleaves decode A -> B -> A -> B, checking every step's
   logits against the matching reference step: argmax equal and max |diff|
   within `--tol` (default 0.05). The two prompts differ in content and length
   so a KV clobber cannot be byte-identical (vacuous-pass guard: the reference
   first-step logits of A and B must themselves differ by more than `tol`).
3. **Clear + replay on A** — clears B's memory, then replays A's held-back
   continuation token and checks A still matches its reference, proving B's
   KV clear did not disturb A's KV/runtime memory.
4. **Positive control (replay on B)** — replays B's own prompt after the same
   clear and checks it reproduces B's fresh single-context reference. This is
   what step 3 alone cannot prove: that the clear actually emptied B's KV
   rather than being a silent no-op that would leave A looking undisturbed
   either way.

Exit 0 pass, 1 fail, 77 skip (no model file, or no GPU backend registered).

```bash
source /opt/intel/oneapi/setvars.sh --force
ctest --test-dir build -R '^test-download-model$'            # stage the fixture
ctest --test-dir build -N -R '^test-sycl-two-context-ownership$'   # must list 1 test
ctest --test-dir build -R '^test-sycl-two-context-ownership$' --output-on-failure -j 1
# expect: "[TWO-CTX] PASS: two live contexts (n_ctx 256 / 512) kept KV/runtime state isolated"
```
