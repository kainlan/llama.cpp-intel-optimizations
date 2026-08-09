# W1 convergence: do the eight k7b0 lanes compose?

**Ticket:** `llama.cpp-otry` (convergence child of `llama.cpp-k7b0`).
**Evidence base:** the closing comments of the eight closed children —
`nn6z`, `nlww`, `h5m4`, `t5nq`, `o6jx`, `y36c`, `vbeb`, `x3ou`.
**Derived at HEAD `d14a0f682`**, branch `feature/sycl-b70-capability`,
working tree clean. All sixteen lane commits confirmed ancestors of HEAD.

This document answers one question: *at current HEAD, does every
mutable static-storage class the eight lanes dispositioned have a coherent
owner / reset / teardown story, with no gap between lanes and no lane
regressing another's invariant?*

Every line number below was re-derived from the tree at `d14a0f682`.
`ggml/src/ggml-sycl/ggml-sycl.cpp` (100,440 lines) is skipped by
codescout's index as oversize — confirmed in this session, the response's
`skipped.reasons.oversize` names it explicitly — so every claim about that
file comes from `cat … | grep -n`, never from `search_text`.

No build, no GPU, no `ctest` execution was performed for this review. The
verdicts below are source- and evidence-review verdicts; the hardware
proofs they rest on belong to the individual lanes and are cited to their
tickets.

---

## 1. Census closure — the k7b0 suspect table

k7b0's description named seven category-(c) candidates ("model-scoped and
leaking"). **All seven now have a disposition. None is UNHANDLED.**

| suspect | disposition at HEAD | owning lane | evidence |
|---|---|---|---|
| `g_placement_kv_info` | **Reset structurally.** Whole-struct `= ggml_sycl::placement_kv_info{}` in `ggml_sycl_reset_model_load_scratch_state()` | load-boundary reset (pre-dates the eight; refined by the immutable-placement migration) | decl `ggml-sycl.cpp:12351`, reset `:12429` |
| `g_has_placement_plan` | **Symbol eliminated.** Zero occurrences in live source | immutable placement-snapshot migration (`c3bfd71c4` and predecessors) | see §1.1 |
| `g_sycl_weight_usages` | **Owner-keyed + owner-scoped erase.** | nn6z lineage / o6jx teardown | decl `:9832`, write key `:12070`, read key `:12113`, erase `:10441-10442` |
| `g_sycl_named_weight_cache_uuids` | **Unchanged, and correctly so** — but the written justification was incomplete; corrected. See §1.2 | none (justified, not fixed) | decl `:9827`, use `:9857-9862` |
| `g_moe_n_experts_total` | **Reset structurally** in the same load-boundary scratch reset | load-boundary reset | decl `:12348`, reset `:12426` |
| layer-stream `host_ptr` registry | **Owner-gated.** `register_host_ptr()` consults the owner gate; teardown is owner-gated | vbeb (gate) + y36c (teardown call) | `layer-streaming.cpp:269`, `:92-105`; call site `ggml-sycl.cpp:10940-10943` |
| `g_sycl_canonical_checksums` | **Owner-keyed + 4th erase loop + shutdown drop** | x3ou | decl `:7960`, key `:7985-7989`, erase `:10450-10452`, shutdown drop `:7995` |

### 1.1 `g_has_placement_plan` is gone, not merely quiet

A tree-wide `search_text` for `g_has_placement_plan` returns **eight
matches, none of them live source**: three in
`docs/backend/sycl-cross-model-state-audit.md`, one in
`docs/backend/sycl-static-storage-inventory.csv`, one in
`docs/design/sycl-canonical-memory-architecture.md:375`, one in
`docs/plans/2026-04-19-cache-expert-invariant-investigation.md`, and one
inside an *example string* in `tests/test-sycl-lifecycle-source-contract.py:109`.
`cat ggml-sycl.cpp | grep -n` returns zero.

`git log -S` attributes the removal to the immutable-placement reader
migration (`c3bfd71c4`, `92b2675f6`, and predecessors). The flag was not
"fixed"; the design that needed it was replaced by atomically published
immutable placement snapshots.

**Consequence for jwy4:** the CSV and the canonical-memory doc still carry
the symbol, and `sycl-canonical-memory-architecture.md:375` cites it at
`ggml-sycl.cpp:6195`, a line that no longer holds it. This is stale prose,
not a live defect — it belongs to jwy4's regeneration, together with the
`g_moe_expert_split_active` row x3ou already flagged (`llama.cpp-a4y2`).

### 1.2 `g_sycl_named_weight_cache_uuids` — right verdict, incomplete reason

x3ou's phase-1 comment flagged this map as "still unowned … find/emplace
only, keyed by a bare `name|type|shape` signature, never erased", and
correctly declined to touch it as out of its regions. The audit doc
(`sycl-cross-model-state-audit.md:411-417`) had already ruled it category
(b), safe, never to be cleared.

**Both are right about different things, and the doc's stated reason did
not carry the verdict.** The doc argued from the key being a pure function
of tensor identity: a stale row is "either identical (harmless reuse) or
keyed differently (no collision possible)". Traced to the consumer, that
is not sufficient. The uuid becomes `id.aux_id` for a weight with no GGUF
identity (`ggml-sycl.cpp:12272-12278`). Were `aux_id` the only
discriminator, *identical* would be the harmful case: two different models
each holding a `blk.0.ffn_gate.weight` of the same type and shape would
resolve to one cache entry.

The protection is elsewhere — `cache_id_equal()`
(`unified-cache-key.hpp:57-63`): on the `has_gguf == false` branch,
`compare_logical` is true and **`model_id` and `name_hash` both participate
in equality**, with `cache_id_hash` hashing both on the same branch
(`:90-92`). The file states the rule outright at `:55-56`: *"Without GGUF
identity there is no physical fact to key on, so model_id, name_hash and
aux_id all stay in and such weights never share across models."*

So the map is safe, the uuid discriminates only *within* a model, and no
fix is required. The audit doc has been corrected in this pass to name the
load-bearing clause, so the verdict is not re-derived from a property that
does not establish it.

**Residual, non-defect:** the map grows monotonically (one `string→uint64`
row per distinct non-GGUF weight signature seen in the process) and is
never dropped, including at module shutdown. Bounded by distinct weight
shapes, negligible, and recorded here only so a future reader does not
re-open it as a leak.

---

## 2. Cross-lane seams

### 2.1 y36c's teardown call uses vbeb's `release_if_owner` — HELD at HEAD

`ggml-sycl.cpp:10934-10943`, inside `ggml_sycl_teardown_owner_effects()`:

```cpp
if (!ggml_sycl::layer_streaming_active(device)) { continue; }
ggml_sycl::get_layer_stream_manager(device).release_if_owner(dying);
```

Both halves verified against the primitive rather than its description:

- `release_if_owner()` (`layer-streaming.cpp:92-105`) takes
  `owner_transition_mutex_`, tests `owner_gate_.is_owner(owner)`, and on a
  match does **both** `owner_gate_.forget()` and `release_model_state()`.
  A release-only primitive would have left the gate claiming a dead owner
  and told the next model KEEP instead of ADOPT; it does not.
- `adopt_current_owner()` (`:107`) and `shutdown()` (`:240`) take the same
  mutex, which is what makes the check-and-release atomic rather than
  merely narrow. This closes the TOCTOU y36c reported and vbeb landed
  (`c-76ht` / `a9318f048`).
- The `layer_streaming_active()` pre-filter is **required, not
  defensive**: `get_layer_stream_manager()` (`:526-533`) does
  `try_emplace` and creates a manager on access, so an unfiltered loop
  would fabricate a working set on every device at every model teardown.
  `layer_streaming_active()` (`:535-538`) uses `find` and creates nothing.

### 2.2 x3ou's erase extends the shared per-owner release point — composition verified

`ggml_sycl_erase_weight_identities_for_owner()` (`ggml-sycl.cpp:10420-10455`)
is **four loops in three lock scopes**, and no lane invented a new clear:

| # | map | lock | selector |
|---|---|---|---|
| 1 | `g_sycl_weight_identities_by_name` | `g_sycl_weight_identity_mutex` | structural field compare (`model_id`/`load_txn_id`/`slot`/`slot_generation`) |
| 2 | `g_sycl_gguf_file_ids` | same scope | `ggml_sycl_owner_name_key_matches` |
| 3 | `g_sycl_weight_usages` | `g_sycl_weight_usage_mutex` | `ggml_sycl_owner_name_key_matches` |
| 4 | `g_sycl_canonical_checksums` | `g_sycl_canonical_checksum_mutex` | `ggml_sycl_owner_name_key_matches` |

Loop 1 compares the token's fields directly because that map stores a
struct carrying them; loops 2-4 parse the owner prefix out of the string
key. The asymmetry is structural, not an oversight, and
`ggml_sycl_owner_name_key_matches()` (`:10394-10418`) parses all four
components and requires the trailing `:`, so a prefix cannot match a
longer id by truncation.

### 2.3 Module-shutdown ordering — coherent, and no lane broke another's constraint

`ggml-sycl.cpp:99858-99861`:

```cpp
ggml_sycl_reset_pending_kv_layer_masks();   // y36c
ggml_sycl_reset_canonical_checksums();      // x3ou
ggml_sycl_reset_moe_module_state();         // nn6z / nlww
if (!ggml_sycl::shutdown_unified_cache()) { ... }
```

This is the seam most at risk of a silent cross-lane regression, because
y36c recorded (`c-zfxj`) that
`tests/test-sycl-lifecycle-source-contract.py` asserts
`ggml_sycl_reset_moe_module_state();\n    if (!ggml_sycl::shutdown_unified_cache())`
as **adjacent text** — y36c's own first attempt inserted between them and
the gate raised `ValueError: substring not found`.

x3ou landed afterwards and inserted at line 99859, *between* y36c's drop
and the MoE reset. The asserted adjacency is preserved. Each reset is
shutdown-only by design and each says so in a comment at its definition
(`:7992-7994`, `:11661-11663`), on the same stated ground: module shutdown
is the one point with no owner left for whom the state could be preserved.
None of the three is reachable from a model-load boundary, so none can
sweep a live model's state.

**Verdict on §2: the three seams hold. No lane regressed another's
invariant.**

---

## 3. Raw-pointer identity and live-handle clearing

Spot-audited every new code path the eight lanes introduced, against the
`mem_handle` rules in CLAUDE.md and the canonical contract. **No violation
found.**

- **vbeb, `layer-streaming.{cpp,hpp}`** — the strongest test, because this
  lane manages device buffers directly. `mem_handle buffer_handles_[2]`
  (`hpp:168`) are the ownership tokens; `void * buffers_[2]` (`:167`) are
  commented *"Cached raw ABI views; handles own lifetime."*
  `release_model_state()` (`cpp:70-90`) moves the handles into a local
  `mem_handle released[2]`, nulls the raw views under `host_ptr_mutex_`,
  and lets the locals destruct **after** the lock — freeing through the
  unified cache while holding a lock the class also takes on its copy path
  is exactly how the deadlock would be built, and the code says so.
  `drain_and_invalidate_buffers()` (`:51-68`) waits on `prefetch_event_`
  before any handle is dropped, so no allocation is freed with a DMA in
  flight. `allocate_buffers()` (`:191-238`) releases *everything* on
  partial failure rather than leaving the previous owner's pointer in the
  untouched slot.
  One note, not a finding: `is_active()` (`hpp:94`) reads
  `buffers_[0] != nullptr`. That is a raw pointer used as a liveness view,
  but it is set and cleared in the same lock scope as the handle it is
  derived from, and it is never used as an identity or a cache key — which
  is what the rule actually forbids.
- **y36c, pending KV masks** — `ggml_sycl_pending_kv_layer_mask`
  (`ggml-sycl.cpp:11475-11479`) is `{ModelToken owner, uint64_t
  handoff_id, std::vector<uint8_t> mask}`. All values; no pointer is
  stored, so there is nothing to dangle. Correlation is by token, not by
  address.
- **x3ou, canonical checksums** — the key is a string built from the owner
  token plus the tensor name (`:7985-7989`); the `data` pointer is read
  transiently to compute the FNV hash (`:8044-8045`) and never retained.
  Both accessors fail **closed** on an empty key (`:8036-8040`,
  `:8054-8057`) rather than falling back to a bare-name bucket.
- **Weight usages** — owner-keyed at both sites and fail-closed on an
  empty key (`:12070-12073`, `:12113-12117`).

No lane clears a live handle to reclaim memory, and no lane introduced a
pointer-addressed table.

---

## 4. A→B→A behavioural coverage, by subsystem

Registered ctest names confirmed from the `add_test` registrations, not
inferred from filenames (they differ —
`sycl-lifecycle-public-wrappers` is registered from
`tests/test-sycl-lifecycle-public-api.cpp`).

| subsystem | behavioural cross-model coverage | test + case |
|---|---|---|
| MoE discovery / popularity | **yes** | `sycl-moe-discovery-owner-state` → `A-B-A`, `simultaneous-live-models` (`test-moe-discovery-owner-state.cpp:126`, `:153`) |
| MoE bias / activation | **yes** | `sycl-moe-bias-owner-state` → `case_overlapping_layer_ids` A→B→A→B (`test-moe-bias-owner-state.cpp:243`), `case_biased_to_unbiased` (`:173`) |
| weight usages | **yes, real load transactions** | `cross-model-weight-usage` checks 4/5/6 (`test-cross-model-weight-usage.cpp:124,132,135`) |
| canonical checksums | **yes, real load transactions** | `canonical-checksum-owner-scope` checks 5/6 (`test-canonical-checksum-owner-scope.cpp:220,226`) |
| layer-stream manager | **yes, real load transactions** | `sycl-layer-stream-owner-lifecycle` → `case_5_a_then_b_then_a` (`:286`), plus cases 1/2/7 |
| pending KV-mask handoff | **yes, two load transactions** | `sycl-lifecycle-runtime-wrapper` phase at `tests/test-sycl-lifecycle-runtime-wrapper.cpp:1257`, claim 2 at `:1338` |
| KV tier sizing | **primitive level only** — see F3 | `sycl-kv-slice-sizing` case 11 (`test-kv-slice-sizing.cpp:403`) |
| TLS MoE layer-ids cache | **no** (by design) | `sycl-moe-worker-reset` is cross-thread/cross-graph (`test-moe-layer-ids-worker-reset.cpp:190`) |
| packed-K sidecar | **no** (by design) | `test-fattn-packed-k-lifecycle.cpp` is model-free; t5nq's live matrix was deliberately model-free |
| named weight cache uuids | **no** (none needed) | safe by construction, §1.2 |

Source-contract / AST gates (`sycl-lifecycle-source-contract`,
`canonical-checksum-source-contract`, `sycl-layer-stream-owner-contract`,
`sycl-moe-{discovery,bias}-owner-contract`) are **static text checks and
are not behavioural cross-model coverage**. They are counted separately
above and must not be read as substituting for it — a point y36c already
made in the narrower case (`c-a6b7`: vbeb's case 6 asserts the
*primitive*, never the *path*).

---

## 5. Findings

### F1 — the layer-stream working set has no module-shutdown drop, unlike its three siblings (medium; reachability unproven)

Module shutdown (`ggml-sycl.cpp:99858-99861`) drops the pending-KV masks,
the canonical checksums, and the MoE module state, and then calls
`shutdown_unified_cache()`. **The layer-stream managers are absent from
that sequence.**

`g_layer_managers` (`layer-streaming.cpp:523`) is a file-static
`unordered_map<int, unique_ptr<layer_stream_manager>>` that is never
erased. Its only release paths are:

1. `release_if_owner()` at model teardown — y36c's call, §2.1; and
2. `~layer_stream_manager()` (`:47-49`), which calls `shutdown()`, at
   **static destruction**.

Static destruction runs strictly after `shutdown_unified_cache()`. So a
manager still holding `buffer_handles_` (unified-cache `mem_handle`s) when
the module shuts down would present live allocations to the cache
shutdown, which throws `"SYCL unified cache arena release failed"`; and
the handles would then be destroyed later against an already-shut cache.

This is precisely the asymmetry y36c identified for the pending-KV deque —
*"this queue was the one module working set `shutdown_resources()` never
reset"* — and fixed. Layer-stream is now the remaining one, and it is the
only one of the four that owns **device memory** rather than plain values.

**Reachability is not proven and I did not run anything to test it.** On
the normal path a streaming model's teardown fires `release_if_owner`
first and the manager is empty by module shutdown. The gap needs a
streaming model still owning buffers at module-shutdown time.

**Why I did not land the fix.** It looks like four lines, but the
insertion point is constrained in a way that has already bitten this
sequence once: the source contract asserts
`ggml_sycl_reset_moe_module_state();` and the `shutdown_unified_cache()`
call as **adjacent text**, so a naive insertion between them fails the
gate (y36c hit exactly this, `c-zfxj`). A correct patch goes beside the
other two drops, before `ggml_sycl_reset_moe_module_state()`, must use the
`layer_streaming_active()` pre-filter to avoid `try_emplace` fabricating
managers, and should call `shutdown()` (unconditional, correct here —
`layer-streaming.cpp:246-248` states module/destructor teardown is exactly
its intended caller) rather than `release_if_owner()`. It also cannot be
verified without a build, in a shutdown path where a mistake throws.
Recommend a scoped follow-up ticket with a contract clause asserting the
call sits beside the other two drops.

### F2 — audit-doc justification for `g_sycl_named_weight_cache_uuids` was incomplete (corrected in this pass)

Detailed in §1.2. The verdict was right; the stated reason did not carry
it, and the clause that does (`cache_id_equal` keeping `model_id` and
`name_hash` for non-GGUF weights) was not named. Corrected in
`docs/backend/sycl-cross-model-state-audit.md`. **jwy4 should preserve
this text through the census regeneration.**

### F3 — the founding defect's own subsystem has no load-boundary test (informational, for jwy4's census)

k7b0 exists because `kv_tier_manager::per_layer_kv_bytes_.resize()` leaked
a previous architecture's per-layer sizes. `sycl-kv-slice-sizing` case 11
(`test-kv-slice-sizing.cpp:403`) does assert non-inheritance across a
sequential A-then-B reconfiguration, in four sub-scenarios, and a one-word
`assign`→`resize` revert fires it specifically — that is the specificity
bar k7b0 set, and it is met.

But case 11 calls `configure_from_plan()` **directly**: there is no
`model_load_begin`/`end` bracket and no lifecycle registry. So it proves
the primitive resets correctly and does **not** prove the production load
path calls the reconfiguration at all. Every other repaired subsystem in
§4 now has a load-transaction-level test; this one does not. Not a
regression from any of the eight lanes — the gap predates them — but it is
the weakest remaining link in the "sweep results are real, not
provisional" claim, and jwy4's census should carry it rather than let it
disappear.

### F4 — stale census artefacts (already partly tracked)

`docs/backend/sycl-static-storage-inventory.csv` still lists
`g_has_placement_plan` and `g_moe_expert_split_active` (both now removed
from source) and carries `ggml-sycl.cpp` line numbers roughly 700-800
lines stale. `docs/design/sycl-canonical-memory-architecture.md:375` cites
`g_has_placement_plan` at `ggml-sycl.cpp:6195`. x3ou already raised the
CSV half as `llama.cpp-a4y2`; the canonical-memory-doc citation is added
here. All of it is jwy4's regeneration scope.

---

## 6. What unblocks `llama.cpp-jwy4`

**The census is closed and the seams compose.** Concretely:

1. All seven k7b0 category-(c) suspects have a disposition (§1). Six are
   fixed or eliminated; one (`g_sycl_named_weight_cache_uuids`) is
   justified in writing as safe, and that justification is now complete
   and cites its actual mechanism. **Zero UNHANDLED.**
2. The three cross-lane seams hold at HEAD, and no lane regressed
   another's invariant (§2). The shutdown ordering in particular survived
   two independent insertions with a textual-adjacency contract over it.
3. No lane introduced raw-pointer identity or live-handle clearing (§3).
4. Seven of ten subsystems have behavioural cross-model coverage; the
   three that do not are either safe by construction or covered by design
   at a different axis (§4).

jwy4 can regenerate the inventory/CSV/prose at implementation HEAD. It
should carry forward, not re-litigate: **F1** (open, needs its own
ticket — it is a code gap, not a census gap), **F2** (preserve the
corrected justification text), **F3** (record the KV-tier load-boundary
gap in the census), and **F4** (the stale rows are exactly what the
regeneration removes).

Nothing found in this pass blocks jwy4. **F1 is the one item that should
not close with the tree** — it is a real asymmetry in the module-shutdown
sequence, and it wants a build and a gate that this task was scoped out
of.
