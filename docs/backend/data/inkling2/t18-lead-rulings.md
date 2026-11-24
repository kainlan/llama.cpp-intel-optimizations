# Task 18: the lead's rulings

Two records kept from the execution: the rulings after the implementer's checkpoint, and the findings and rulings of
the first quality review. Line numbers are those of the commits each record names.

---

# Task 18: the lead's rulings after the implementer's checkpoint (2026-11-23)

These amend the task text in the plan (`### Task 18` in
`docs/plans/2026-11-21-inkling2-mini-sycl-enable-profile-optimize.md`). Where they differ from the plan text, they win.

Checkpoint corrections the lead accepted (they are the design):
1. `mmvq_moe_batched_dispatch` launches the UNCHECKED `mmvq_submit_q1_nvfp4_aos_id` with ids=nullptr on a compacted
   list. The checked overload is reached only through `mmvq_submit_q1_nvfp4_aos_id_admitted`.
2. `ggml_sycl_moe_decode_direct_decide` refuses when more than one device is routable. The full-local table
   (`moe_fusion_ensure_full_local_ptr_table`, used by the BF16 route) has no such limit.
3. The admitted kernel entry validates `ids_host` on the host.
4. Other per-node costs: the empty `single_task` from `mul_mat_id_ready_guard`'s destructor, a conditional wait on
   `g_cpu_tg_direct_pending_scatter`, a barrier per table-lease replacement, the terminal host task.

Decisions:
- D1: NVFP4 only. Q1_0 stays on the admitted route; one comment at the predicate says why (its admitted lifecycle
  cases are the vehicle for the failure-path tests).
- D2: a strided or non-dense src1, and any shape or stride the predicate does not take, DECLINES to the admitted
  route, which is unchanged. A decline is not a refusal: no throw, no failed graph. The route log names the reason
  for a decline, once per tensor per reason.
- D3: acceptance amended: the Q1_0, prompt, strided and mixed cases of the admitted device test pass unchanged; its
  NVFP4 all-resident dense decode cases assert the resident route.
- D4: memoize the predicate's plan answer on the tensor extra, keyed by replan epoch and expert storage generation.
  The memo must not be consulted for anything the key does not cover: shapes, strides and the ids' residency are
  checked per node. The key is pinned in the source contract with a mutant.
- D5: publish the kernel's event as dst's ready event and do not submit the empty `single_task` on this route.
  Every consumer the guard's sync event protects today (the pending CPU scatter, a following host read of dst,
  set_tensor/get_tensor sync) must still order after the kernel through the ready event.

Design points:
- Table: the full-local table at AOS (`moe_fusion_ensure_full_local_ptr_table`), as the BF16 route uses, so a
  two-card run with every expert of the tensor PRIMARY on the context's device takes the route. Not
  `ggml_sycl_moe_decode_direct_decide`.
- Kernel entry: a sibling `mmvq_submit_q1_nvfp4_aos_id_resident` using the checked overload only; the size
  arithmetic shared with the admitted entry through one helper; quantize into a Q8_1 SCRATCH allocation from the
  unified cache held as a `mem_handle`; dst written directly. No `ids_host` anywhere on the route. The launch's
  global range is checked against INT_MAX and refused by name.
- Retained until the kernel's event: the table's dispatch bundle snapshot, the ids' storage handle, the Q8_1 scratch
  handle. Nothing else keeps them alive.
- Plan says yes but the table is incomplete: decline, logged once.
- Bad ids: id out of range or a null slot writes NaN rows, id below zero writes zero rows, the graph succeeds
  (the BF16 route's behaviour). Stated in docs/backend/SYCL.md next to the BF16 route's note if one exists,
  otherwise in the route's head comment only.
- Under graph recording: throw as the BF16 route does; set `ctx.moe_graphs_disabled_once`.
- Profiler row: the kernel keeps `mulmat.mmvq.id_aos_nvfp4`; a separate quantize submission gets its own row.
- Route log behind `GGML_SYCL_MOE_ROUTE_LOG=1`, at the level its neighbours use.
- Not in this task: a shared immutable lease set per table generation, recording the node into a replay segment,
  the prompt path.

Standing rules that bind the change (CLAUDE.md, "SYCL Memory Ownership"): allocate only through the unified cache;
ownership is a `mem_handle` or an `alloc_owner`, never a raw pointer; hold the handle until the SYCL event is done;
no GPU reads of host memory; no new host waits; layout follows residency (the loaded AOS layout is the answer, no
conversion); no code parsing `ONEAPI_DEVICE_SELECTOR`; no `-ngl`.

The implementer's stated deviations (judge each):
1. Plan identity is in the memo key as a `weak_ptr` compared with `owner_before`.
2. An incomplete table is a decline (TABLE_INCOMPLETE) in the route, not in the predicate.
3. `ggml_sycl_moe_bf16_direct_own_vram_buffer` renamed `ggml_sycl_moe_direct_own_vram_buffer`; the bf16-direct
   contract expects 4 call sites, not 3.
4. Three neighbour contracts follow the geometry move (mixed-direct, resolved-batch).
5. `ctx.moe_graphs_disabled_once` is also set in the recording arm before the throw.
6. An entry that returns false is a named refusal (WARN plus `ggml_sycl_fallback_error`).
7. The SCRATCH request sets `cascade_step = true` (first writer of the field) with a new allowlist entry
   H-MOE-RESIDENT-DECODE-Q8 in scripts/sycl-alloc-zone-contract/allowlist.json.
8. The route function is a named site in the owner-first census.
Also: on a table miss the route publishes from this card's own VRAM buffer and retries.

---

# Task 18: review round 1 findings and the lead's rulings (sent to the implementer 2026-11-23)

Reviewed head was 48390d4bb. Line numbers below are that head's. The fix round is 48390d4bb..8825673f7.

- F1 (blocker, lifetime). Handles (notably the Q8_1 slice the route alone owns) could be released while submitted
  work still ran, on four paths: (a) the kernel `q.submit` throws after the quantize was submitted; (b)
  `retain_handles_until_event` throws after the kernel was submitted (the vector had been moved by value); (c) the
  profiler's `record_event` throws; (d) the entry returns false after the quantize was enqueued.
  Ruling: use `ggml_sycl::terminal_retention_ticket` as the admitted route does, or hand-roll the guarantee with a
  stated reason if the ticket costs more on the success path. A drain is allowed on failure paths only. All four
  paths pinned with mutants.
- F2 (hot path). `ggml_sycl_snapshot_moe_ptr_table_dispatch_bundle` deduplicated in O(n^2) per node (129 locked
  copies, 8128 identity compares). Ruling: this route takes a snapshot without dedup, by parameter or sibling.
- F3 (placement). src1 and dst must be in this device's memory; otherwise decline, logged once. Same gap in the BF16
  direct route, fixed through the same helper, with a BF16 contract pin.
- F4. Skip the publish-and-retry when the full-local probe already failed at this storage generation.
- F5. One ensure/publish/retry helper for both routes; the BF16 contract counts call sites inside the BF16 bodies.
- F6. The INT_MAX bound on the kernel and quantize ranges moves into the shared geometry helper (covers the admitted
  entry too). The route's own named refusal stays first. The admitted caller must end in a named refusal on false.
- F7. The D5 pin gets a mutant that publishes a non-kernel event. A device case where an NVFP4 decode node declines
  and the admitted values match the oracle asserts the resident counter did not move.
- F8 nits. `std::move(ids_handle)`; no task number in comments; one docstring.
- F9 cheap hot-path items: the fp16 aspect answered once per device; the plan owner fetched once; the quantize
  profiler label built only when the profiler is on; `ctx.stream()` read once; no default-constructed event on a
  node that does not take the route.

Lead finding on 912e0a810 (sent 2026-11-24): the keep-on-throw change added `reserve(size() + n)` on every call in
`retain_handles_for_current_graph` (three vectors), which reallocates the whole vector per retaining node. Ruling:
no reserve on the single-vector sink path (a range insert at the end has no effects when its allocation throws);
the two store vectors grow through one geometric helper before either insert, or ids first with a take-back.

What the implementer did, in its own words (verify each against the code, do not trust it):
- F1: hand-rolled. `struct ggml_sycl_moe_resident_decode_retention` holds the handles and `sycl::queue * submitted`;
  its destructor drains with `wait_and_throw` inside `catch (...)` on any exit that did not publish. New API
  `retain_handles_until_event_or_keep(std::vector<mem_handle> &, sycl::event)`; `publish_handles_until_event` and
  `retain_handles_for_current_graph` now take the vector by reference for every caller (the by-value wrappers
  remain). Reason given for no ticket: `prepare()` copies every owner into `publication_`, about 130 more locked
  handle copies and releases per node. New G-CATCH allowlist entry E-MOE-RESIDENT-DECODE-RETENTION-DRAIN.
- F2: `enum class moe_dispatch_bundle_dedup { BY_IDENTITY, NONE }`, a defaulted parameter; the route passes NONE.
- F3: `ggml_sycl_moe_direct_device_operand(tensor, device)`; decline NO_POINTER renamed OPERAND_NOT_ON_DEVICE.
- F4: weight field `moe_direct_publish_failed_generation[GGML_SYCL_MAX_DEVICES]`.
- F5: `ggml_sycl_moe_direct_ensure_aos_table(ctx, src0)`.
- F6: `mmvq_q1_nvfp4_id_fit { FITS, NO_KERNEL, LAUNCH_PAST_INT }`; both entries take `int device`.
- Reserve finding: sink reserve dropped, `retained_vector_make_room` (geometric) for the two store vectors, a
  `static_assert` that a `mem_handle` move cannot throw.
- Owed by its own account (not done): the BF16 route has the same publication window (by-value retain after submit,
  a profiler throw); `terminal_retention_ticket::prepare` has an exact reserve on `graph_sink_` since 2026-08-13;
  the admitted route does not check src1/dst are on this card; the admitted adapter's refusal line does not name
  which check failed; the shared immutable lease set per table generation.
