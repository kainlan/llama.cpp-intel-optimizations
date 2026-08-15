# Pending tracker writes (codescout lock unavailable, 2026-08-15 ~03:2x UTC)

> ⚠️ **See CORRECTION at the end of this file before scoring census 5.** One
> claim I made in c-62oe and repeated to the lead — "the dispatch admission path
> never reads `g_moe_expert_meta`" — is FALSE as stated. The corrected trace, a
> full reader table, and a pre-registered discriminator are at the bottom.

`task_comment_add` failed three times with
`MCP error -32603: cannot create lock file /Apps/llama.cpp/.codescout/.tasks.jsonl.lock`.
The directory itself is writable (a `touch` probe succeeded), so this is the
daemon's lock, not a permission problem — most likely contention with another
agent's write. Same handling as the 2026-08-15 02:20 outage recorded in
`artifacts/hardening-merge/pending-tracker-writes.md`: queue the text here and
post it when the tracker returns.

An earlier comment from this session DID land: **c-62oe** (implementation report
+ the pre-registration conflict). Only the build report below is unposted.

## → llama.cpp-f8ws (build green + artifact verification)

BUILD GREEN. Ready for census 5 (lead-only).

`BUILD_RC=0`, **0 error/FAILED lines**, `-j 8`, log at `artifacts/f8ws2/build.log`.
Targets: `ggml-sycl test-backend-ops test-sycl-reset-model-weight-lease-preserve
llama-completion llama-cli llama-bench`. Worktree `/Apps/llama.cpp-hm-f8ws`,
HEAD `30e5bcc27`, `git status --short` empty.

### The artifacts, not the exit code

Commits 23:00:53 and 23:01:48 → `libggml-sycl.so.0.15.3` **23:18:57**,
`test-backend-ops` / `test-sycl-reset-model-weight-lease-preserve` /
`llama-completion` / `llama-bench` **23:18:59**, `llama-cli` 23:18:58. All
postdate both commits.

**Currency, not just freshness:** `ninja -n ggml-sycl test-backend-ops
test-sycl-reset-model-weight-lease-preserve` → **"no work to do"**. A repeat
build that still did work would be evidence against currency; mtime alone is not
the check.

**Not a CPU-fallback build:** `GGML_SYCL:BOOL=ON`;
`ldd | grep -cE 'libggml-sycl|libsycl'` = **2** for both `test-backend-ops` and
the lease-preserve binary.

**New observables in the shipped `.so` — both halves scored, presence control
before absence:**

| probe | old `.so` (22:15, pre-commit) | new `.so` |
|---|---:|---:|
| `nm -C … 'note_model_load_end('` *(control: the probe works)* | 2 | 2 |
| `nm -C … 'note_buffer_owner_dead('` | **0** | **1** |
| `nm -C … 'note_buffer_owner_live('` | 0 | **1** |
| `strings … 'weight:owned_by_live_model'` *(control)* | 1 | 1 |
| `strings … 'weight:owned_by_live_buffer'` | **0** | **1** |
| `strings … 'owned-by-live-buffer=%zu'` | 0 | **1** |
| `strings … 'buffer owner 0x%llx freed with'` | 0 | **1** |

The old-artifact column is a real RED capture taken before the build, so a zero
in the new column would have been a measurement rather than an unmatchable
pattern. `nm -C` is matched on the substring **with the open paren**, never
`grep -w` (mangled-name trap).

**Commit `30e5bcc27` has no new string or symbol** — it is a `continue` inside an
existing loop. Its artifact evidence is the currency check above plus its source
gate, and I am not claiming more than that.

### Gates run (host-safe only)

```
ctest -R 'universal-provenance|prestage-block-guard'
  #355 test-sycl-universal-provenance-source ....... Passed
  #356 test-sycl-moe-prestage-block-guard-source ... Passed
  100% tests passed, 0 failed out of 2
```

### The ruling-(b) test is built, registered, and NOT run by me

⚠️ Its ctest name is **`sycl-reset-model-weight-lease-preserve`** — no `test-`
prefix (registered in `ggml/src/ggml-sycl/CMakeLists.txt:4000`, not `tests/`).
`ctest -R '^test-sycl-reset-model-weight-lease-preserve$'` selects **zero tests
and passes vacuously**; verify with `-N` first.

Both new cases are provably in the binary (`strings`: "a live buffer" ×2,
"buffer free erases idle entries" ×1, with a pre-existing case name as the
positive control). The registration supplies
`ONEAPI_DEVICE_SELECTOR=level_zero:1` and `LD_LIBRARY_PATH`, so **run it through
ctest, not by invoking the binary** — a direct run without oneAPI sourced prints
SKIP and exits 77.

```bash
ctest --test-dir build -N -R '^sycl-reset-model-weight-lease-preserve$'   # must list 1
ctest --test-dir build -R '^sycl-reset-model-weight-lease-preserve$' --output-on-failure
```

Its registration records the measured cost as `free -g` flat, peak RSS 317 MB —
it loads no model.

### Standing pre-registration for census 5, restated

- refusals **925 → 0** (661 prompt + 264 decode); `missing_key_refusal` **1 → 0**
- `not_supported` **11905 UNCHANGED** — the control
- `Expert registry` **16/16/2 unchanged**; **`Expert metadata stored` 0/0/0
  UNCHANGED** (see c-62oe — this replaces the "0→16/16/2" line, which the
  block_num ruling makes unreachable); `Expert group registry` **0**
- `range_rejected` / `warn_misaligned` / `tlsf_assert` / `alloc_abort` **0**
- failing block **0 ≤ N ≤ 925**, no floor, no point estimate
- first abort suspects per c-o4pr: **:57233 / :57288** (host-resident
  `src0_storage`) — provenance changes which tensors reach host-resident
  routing, which is exactly that watch's trigger
- real-model gates (Mistral completion, GPT-OSS count, B70/B50 PP512/TG128)
  **must not move**; registration defers to model provenance three ways so the
  model-load path is untouched, and any movement there is the signal that a
  deferral is incomplete

---

## ⚠️ CORRECTION to c-62oe, before census 5 is scored

**What I claimed** (c-62oe, and repeated to the lead, who put it in the scoring
notes): *"the dispatch admission path never reads `g_moe_expert_meta`."*

**That is false as stated.** I checked the readers by region ("they are all in
the prestaging region :3730–:6438") instead of by enclosing function. Enumerated
properly — every reference, with the definition that encloses it — one reader
sits directly on the route-resolution path:

| line | enclosing function | keyed by | on the dispatch path? |
|---:|---|---|---|
| 3793 | `moe_prestage_popular_experts` | copies the list; its own `block_num < 0` guard at :3892 | no |
| 4741 | `moe_expert_ensure_soa_cached` | `layer_id` + `expert_idx` | no |
| **5124** | **`ggml_sycl_get_canonical_moe_expert_keys`** | **tensor NAME + `expert_idx`** | **YES — called at :5705** |
| 5228 | `ggml_sycl_materialize_planned_expert_layout` | tensor NAME + `expert_idx` | only under `allow_materialize` |
| 6170 | `moe_acquire_expert_stage_descriptor` | `layer_id` + `expert_idx` | no |
| 6501/6580/6605 | `kv_expert_rebalance_check` | `layer_id` | no |
| 6986 | `moe_compute_gate_norm_placement` | `layer_id` | no |
| 7404 | `moe_hybrid_init_once` | the WRITER | n/a |

### What survives the correction, precisely

**Admission still does not read the list.** `ggml_sycl_resolve_moe_expert_route`
gates at **:5700–5704**:

```cpp
const ggml_sycl_cache_id base_key = ggml_sycl_get_moe_expert_cache_key(src0, extra, expert_id);
if (!base_key.valid) { route.reason = expert_resolve_reason::INVALID_REQUEST; return route; }
```

computed from the tensor and its extra alone. **That** is the gate universal
provenance opens, and the guard cannot touch it.

**But one line later, at :5705, the resolver DOES read the list:**

```cpp
std::vector<ggml_sycl_cache_id> canonical_keys = ggml_sycl_get_canonical_moe_expert_keys(src0, expert_id);
```

Those are *additional lookup keys*, tried only when the base-key lookup misses,
and any key equal to `base_key` is skipped. So for a layer-less tensor the guard
empties that fallback. It can only change an outcome where an entry exists under
a key that DIFFERS from `base_key` for the same (name, expert) — which is why
the equality skip exists at all.

### The part I cannot settle by reading, and the discriminator for it

All five `ggml_sycl_build_moe_resolved_batch` call sites pass
`allow_materialize=false` (:63486, :63518, :63521, :63524, :65013), so
`ggml_sycl_materialize_planned_expert_layout` is not reached *from the refusal
paths*. What I could NOT establish statically is which path populates the
per-expert cache entries for a synthetic MMID tensor. If that population is
metadata-driven (Phase-2 upload keys off `g_moe_expert_meta`), then an empty
metadata list means the route finds nothing and the case is still refused —
with a DIFFERENT reason.

**That difference is already printed, and it is the discriminator. Score census 5
on the refusal REASON, not just the count** — `[MOE-PROMPT-REFUSAL] … reason=%s
source_reason=%d` (:63490) and the decode twin (:65019):

| observed | meaning | action |
|---|---|---|
| refusals → 0 | provenance opened the gate and staging followed | as predicted; nothing to do |
| refusals persist, `reason=invalid_request` | **provenance itself still closed** — my change did not take | debug the mint/registration path, not the guard |
| refusals persist, `reason=route_unavailable` / NOT_FOUND | provenance opened the gate; **nothing staged the expert** | the empty metadata list is the first suspect |

The third row is the one my earlier "cannot suppress the refusal drop" claim
would have hidden — it would have read as a provenance failure when it is a
staging-input failure.

**Contingency, if the third row is what census 5 shows:** the fix is NOT to
delete the block_num guard. Keep the metas and move the guard onto the consumers
that actually need a layer — `moe_prestage_popular_experts` already has its own
at :3892, so the change is small and does not reopen the c-dqqi exposure.

### Why the correction does not change the ruling

The guard still stands. `layer_id` is a name hash
(`moe_cache_layer_id`, FNV-1a, :7151), not a block number, so the layer-keyed
readers were never protected by anything — the block_num guard is the only thing
keeping layer-less metas out of them. That is a stronger justification for the
guard than the one I gave, not a weaker one.

**Process note:** this is [claims-run-one-step-past-the-evidence] on my own
verification. The check I ran ("are the reader line numbers inside the
prestaging region?") is adjacent to the claim I made ("is any reader on the
dispatch path?") and fails open — a reader that lives at :5124, textually inside
the region, is called from :5705, outside it. Region membership was never
evidence about reachability. The enclosing-function enumeration above is the
check I should have run first.

---

## → llama.cpp-f8ws (STAGING DESIGN + handoff) — queued 2026-08-15 ~04:4x, tracker lock unavailable again

### The finding: there is NO "adopt an existing device pointer" primitive

The cache has exactly two external-registration entry points,
`register_host_expert` (`unified-cache.cpp:6341`) and `register_host_weight`.
Both derive location from `query_location(ptr, dev)` and can produce **only**
`HOST_PINNED` or `HOST_MMAP` (:6357-6367) — a device pointer handed to either is
misclassified as `HOST_MMAP`. Every entry that legitimately carries
`cache_location::DEVICE` (:4783, :4816, :5290) is created by a path that
**allocated the memory itself**.

The gap is precise: nothing can say *"this cache entry's bytes live at a device
address somebody else owns."* That is exactly what a tensor already resident in
its own SYCL backend buffer needs.

### The primitive to add

```cpp
// Sibling of register_host_expert, for memory this cache did not allocate.
bool register_device_expert(ggml_sycl_cache_id    key,
                            void *                device_ptr,
                            size_t                size,
                            ggml_layout_mode      layout,
                            mem_handle *          out_handle,
                            std::shared_ptr<void> allocation_owner);   // REQUIRED
```

- `cache_loc = cache_location::DEVICE`, `host_resident = false`,
  `owner_device = dev`. Do **not** derive the tier via `query_location()` —
  validate with `sycl::get_pointer_type(...) == sycl::usm::alloc::device` and
  refuse otherwise. Deriving is what confines the existing helpers to host.
- `storage_owner = allocation_owner`, **refusing when empty**. This discharges
  the lead's binding constraint where it cannot be forgotten: the owner is a
  `std::shared_ptr<mem_handle>` holding a copy of the buffer's `managed_handle`,
  so the entry cannot outlive the allocation and no raw pointer is ever the
  ownership token.
- `non_owning_external_host` stays **false**, correctly: the free predicate at
  :9863 and :10358 is `device_ptr && !storage_owner && !non_owning_external_host
  && !allocation_released_via_owner`. With `storage_owner` set, **reclaim never
  `sycl::free`s this pointer** — the buffer's `mem_handle` stays the sole
  releaser, as the contract requires.
- Range validation `offset + size <= ggml_nbytes(tensor)`, mirroring the
  arithmetic the identity gate already re-verifies.

Lifetime rides machinery already built and tested: the entry is keyed by the
canonical expert key, whose owner id carries the buffer tag, so
`note_buffer_owner_dead()` drops it at buffer free and
`weight_entry_reclaimable()` preserves it while the buffer lives.

### The question that decides whether this works — partially answered

A non-owning entry can only ever be **AOS** (it points at the tensor's own bytes).
If dispatch requests SOA or a packed layout the lookup must MISS rather than
alias, or the layout-specific key discipline breaks. So this works only where
AOS is requested.

Census evidence, with its limit: the only layout lines present read
**`layout=aos`** for both MoE tensors (`leaf_0` q8_0 `reason=default-policy`;
`as` mxfp4 `reason=xmx-tiled-not-…`). Encouraging — **but that log line is gated
on `type == Q8_0 || MXFP4`, so it says nothing about the f16/f32 cases**, which
are most of the 652.

**Cheapest next step, before writing the primitive:** one `test-backend-ops` MMID
subset with the MoE route log enabled, which emits `[MOE-AOS-REQUEST]` and
`[MOE-RESOLVE]` with requested/actual layouts per case. AOS ⇒ the primitive
collapses the 652. SOA/XMX-tiled ⇒ the copying path (buffer owner as allocation
owner) is required instead. **One run decides which design gets built.**

### Where it hooks

A **pre-pass at the MMID dispatch site**, not inside the resolver: the resolver
is the canonical decision seam and every prior round pushed allocating side
effects out of it. A pre-pass also keeps `allow_materialize=false` true at all
five `build_moe_resolved_batch` call sites, so refusal semantics do not move.

### Branch state at handoff

`f53c5c2c6`, tree clean, `BUILD_RC=0`, `ninja -n` clean, three py gates green.

| commit | what |
|---|---|
| `23f83946a` | buffer-scoped weight provenance (minting proven firing by the subset run) |
| `30e5bcc27` | block_num prestaging-planner precondition |
| `ce3f6b74a` | teardown names the lease holder; two one-shot provenance observables |
| `e2ff6b577` | refusals carry `recipe_reason`; sentinel `source_reason` fixed |
| `f53c5c2c6` | fixture owns its lease as production does; ring message disambiguated |

**Unverified:** ruling (b). The two buffer-lifetime cases have still never
executed — `f53c5c2c6` is the first build in which they can.
