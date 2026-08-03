# SYCL cross-model state audit (llama.cpp-k7b0; parser census llama.cpp-1kx3)

## Reproducible parser-grade static-storage census (llama.cpp-1kx3)

The generated inventory is
[`sycl-static-storage-inventory.csv`](sycl-static-storage-inventory.csv). Regenerate it from the
repository root with:

```sh
python3 -m pip install tree-sitter==0.25.2 tree-sitter-language-pack==1.8.1
python3 scripts/audit-sycl-static-storage.py --self-test
python3 scripts/audit-sycl-static-storage.py
python3 scripts/audit-sycl-static-storage.py --check
```

The generator pins and checks the C++ grammar ABI 15 through
`tree_sitter_language_pack` 1.8.1 and `tree-sitter` 0.25.2; it fails on a
different installed version. It walks C++
`declaration` and `field_declaration` nodes rather than matching declaration
text. Each declarator in a multi-object declaration becomes its own row. The
scope walk includes file and named-namespace objects, anonymous-namespace
objects without the `static` spelling, function-local `static`/`thread_local`
objects (including `bias_detect_flag`), and class/header static declarations.
Every row includes an initializer-free AST-derived type and top-level binding
mutability (so `const T *` and containers with const template arguments remain
mutable, while `T * const` is immutable), scope, synchronization, owner
identity, and evidence candidates. Writer, reader, and reset searches are
explicitly labeled **unscoped lexical candidates**: they do not resolve C++
bindings and therefore never establish lifecycle reset/teardown. An empty
candidate search is likewise not proof of no access.

At audited source commit `5793f2ca1089eaf27203ee171c0d73d60a3e4c83`, the
census emits **1,326 object rows**: 395 explicitly-static non-local objects,
58 non-local objects with implicit static storage duration, 869 function-local
static/thread-local objects, and 4 class static declarations. Per-file rows are 1,127
(`ggml-sycl.cpp`), 148 (`unified-cache.cpp`), 8 (`unified-cache.hpp`), 41
(`fattn.cpp`), and 2 (`layer-streaming.cpp`). The script prints SHA-256 for
every input so this result can be tied to exact source bytes.

The supplied **371 lexical candidate leads** are reconciled as leads, not as a
census: their artifact, source SHA, and extraction method were not supplied,
so a row-for-row comparison would be invented. Structurally, they expand in
both directions: multi-object declarations produce multiple object rows,
while static functions/prototypes are not storage objects; the parser also
adds the 58 implicit non-local objects, 869 local statics, and 4 class statics
that a column-zero lexical pass does not cover. No comparison is made to the
historical 329 figure because it likewise lacks a source SHA and method.

### Parse coverage and fail-closed behavior

Tree-sitter reports 41 raw recovery/missing nodes in `ggml-sycl.cpp` and 10 in
`fattn.cpp`; the other three inputs parse without recovery. The former are
from nested preprocessor alternatives, declaration-prefix macros
(`GGML_API`, `__dpct_inline__`), conditional `else` arms, and formatting-macro
tokens such as `PRId64`; the latter are dispatch macro invocations and a label
next to a conditional compilation boundary. These are **explicit raw-parser recovery sites**, not silently discarded
regions. Each of the 51 nodes must receive a structural proof category. In the
current inputs, 17 are confined to parsed function signature/storage spans and
31 are in parsed or unambiguously recovered function bodies. Recovered function
regions end at the lexically balanced closing brace (with comments and literals
masked); an oversized `ERROR` node's tail must independently consist of parsed
top-level constructs or the census fails. Parsed `function_definition` and `lambda_expression` nodes receive the same
check for every compound body containing recovery, even when the AST supplies a
non-missing closing delimiter: the body is lexically balanced
and any parser-owned tail must independently validate. This prevents recovery
from expanding the parsed body over a following valid file-scope declaration by
borrowing a brace from that declaration's initializer.
The same balanced bound controls scope attribution for recovered `ERROR`
functions, preventing later file-scope declarations from inheriting the
recovered function's scope. Only the signature span exempts a function's own
`static` storage-class marker. Body recovery is accepted only when every
`static`/`thread_local` marker in it independently belongs to a parsed
declaration; an unparsed marker fails the census. The remaining 3 sites are
confined to `#if` condition lines, which cannot contain a declaration.
Structural preprocessor recovery spanning a conditional body is not exempt. Any
other namespace/file recovery is rejected because it could conceal an
implicit-static object of a user-defined type or direct-initialization spelling.
There are no declaration-type or spelling exemptions. The generator exits 2
without writing output when a recovery node or storage marker lacks proof.

`--self-test` covers nearest method/lambda scope, const pointee versus const
pointer, mutable containers/atomics, repeated names in different bindings,
initializer-free direct initialization, fail-closed namespace recovery,
multi-object declarations, and namespaced `extern` declaration versus
definition. Negative fixtures also require rejection of malformed recovery in
a function body (`void f(){ static Widget x{; }`), across a structural
preprocessor conditional (`#if X` / `Widget implicit_global{;` / `#endif`), and
in an oversized recovered-function `ERROR` tail containing a malformed implicit
global. The parsed-function recovery fixture
`static void recovered() { #wat x }\nWidget implicit_global{};` and the exact
wrong-close fixture `static void recovered() { x; x template #if X }\nWidget
implicit_global{};` must both fail closed rather than silently dropping the valid
global after assigning it local scope. Lambda coverage includes the exact
file-scope fixture `static auto recovered = [] { x; x template #if X }\nWidget
implicit_global{};`, its templated-lambda variant, and a nested-function lambda
wrong-close fixture; each must fail closed rather than letting a recovery-expanded
lambda compound hide static-storage declarations. It also asserts that the declarations at
`ggml-sycl.cpp` lines 93499,
94640, 94700, 94801, and 94857 retain file scope.

### Static high-risk highlights (no behavior changes in this census)

- `ggml-sycl.cpp:79471` `bias_detect_flag` is a function-local
  `std::once_flag`. Its `call_once` captures MoE expert-bias device pointers
  and host copies from the first graph that reaches it, making it process-first
  semantic model state and the highest-risk review target. The inventory does
  not claim a binding-resolved reset result.
- `ggml-sycl.cpp:1973-1974` `g_moe_hybrid_init_success[]` and
  `g_moe_hybrid_init_done` are process-first MoE initialization/activation
  guards. Unscoped lexical access candidates are inventoried, but lifecycle
  reset is deliberately not inferred. They require binding-aware review.
- `ggml-sycl.cpp:57239` `tl_first_act` is thread-local pinned activation
  staging. It is reused through `ensure()` as bounded capacity storage and the
  current activation is copied before use. This is capacity retention, not by
  itself semantic cross-model state. Nearby `tl_first_tasks` has lexical
  resize/clear candidates, which require binding confirmation. The inventory
  keeps this distinction visible instead of labeling all TLS reuse a leak.
- `fattn.cpp:123-124`'s anonymous-namespace mutex and
  `g_packed_k_sidecars` are included despite lacking `static`; its erase and
  lookup lines are recorded only as unscoped lexical candidates.
  `layer-streaming.cpp:370`'s device-keyed `g_layer_managers` is also included;
  its retained per-manager model inventory remains a lifecycle-review item,
  not a claimed leak or claimed absence of teardown.

This task adds no resets or behavior changes and performs no build, GPU, or
model execution. The sections below preserve the earlier targeted/manual audit;
the generated census does not retroactively make their textual searches
binding-resolved evidence.

---

`test-llama-archs` loads ~131 architectures back-to-back in **one process**.
Any file-scope static in the SYCL backend that describes "the model
currently being loaded" and is never reset between loads leaks into the
next model. The motivating example (`kv_tier_manager::per_layer_kv_bytes_`
using `resize()` instead of `assign()`, fixed in `8e2b65553`) corrupted KV
sizing for ten architectures while seven of them still printed `OK` — an
out-of-bounds write is only fatal when something else owns the memory you
land on. This document is the audit of every other file-scope static in
`ggml-sycl.cpp` and `unified-cache.{cpp,hpp}` for the same shape of bug.

Scope owned by this pass: `ggml/src/ggml-sycl/ggml-sycl.cpp`,
`ggml/src/ggml-sycl/unified-cache.{cpp,hpp}`, plus the new regression test
`ggml/src/ggml-sycl/tests/test-cross-model-weight-usage.cpp`.
`ggml/src/ggml-sycl/kv-tier-manager.cpp` (the original defect) and
`ggml/src/ggml-sycl/layer-streaming.cpp` (assessed below, not owned) are out
of scope for edits here.

## Methodology, and why the counts don't match the ticket's 241/88 exactly

The ticket's counts (241 in `ggml-sycl.cpp`, 88 in `unified-cache.cpp`) come
from an unstated counting method. A script-based extraction of top-level
(column-0) `static` **variable** declarations — distinguishing them from
`static` **function** definitions/prototypes by looking for a top-level `;`
or aggregate-init `{` with no unmatched `(` before it — lands close but not
exact:

| file | script count | ticket count |
|---|---:|---:|
| `ggml-sycl.cpp` | 228 | 241 |
| `unified-cache.cpp` | 26 (see caveat) | 88 |

The `ggml-sycl.cpp` gap (228 vs 241, ~5%) is consistent with the script
under-counting `static constexpr` array/struct initializers that span
multiple lines in ways the brace/paren balancer doesn't fully track, plus a
few macro-generated declarations. The `unified-cache.cpp` gap is larger and
the script's own weakest point: types with `(` in them (e.g.
`std::function<void()>`) get misclassified as function protos by the same
heuristic that correctly excludes real function protos. Rather than spend
further effort perfecting a regex-based C++ parser, this audit re-targeted
effort at the part that actually matters: **finding every static whose
write pattern matches the defect's actual shape**, not producing an exact
head count.

**That targeted method:** every top-level `static` container (`unordered_map`,
`map`, `vector`, `unordered_set`, `set`) in both files was checked
programmatically for whether `<name>.clear(` or `<name>.erase(` appears
anywhere in the same file. A container with neither is a candidate by
construction: something is being added to it (every one of these is written
via `.emplace()`/`.insert()`/`operator[]` somewhere) and nothing is ever
taking members back out — exactly the `resize()`-without-touching-existing-
elements shape, generalized from vectors to maps and sets. This is a
strictly narrower net than "every static," but it is the net shaped like the
bug this audit exists to find, and it is what actually caught the three
confirmed bugs below (none of which were on the ticket's suspect list).

## Category (a): immutable / const — safe by construction

17 `static constexpr` declarations in `ggml-sycl.cpp`, 3 in
`unified-cache.cpp` (sizes, magic thresholds, table data). Not individually
enumerated here; compiler-enforced immutability makes cross-model leakage
structurally impossible.

## Category (b): process/device-scoped, correctly so

- **23 + 6 `static std::mutex`/`std::shared_mutex`** in the two files.
  Synchronization primitives are process-lifetime by definition; not a
  leak class.
- **Config globals set once at startup or via explicit API, read on every
  op**: `g_unified_cache_budget*`, `g_cache_mode`, `g_scheduler_device_count`,
  `g_total_gpu_count` (`unified-cache.cpp`) and their siblings. These
  describe the *host*, not a model; resetting them per model load would be
  the bug, not the fix.
- **Device-indexed arrays/singletons** (`g_expert_prefetchers[...]`,
  `g_pinned_buffer_pools[...]`, `g_device_caches`, etc.): scoped to the
  device slot, and the unified cache's own `mem_handle`-leased weight
  entries (ref-counted per model slot via `reset_model_weight_entries()`)
  are deliberately **not** touched by anything in this audit — per
  CLAUDE.md, another live model's lease is correct, not leaked, and
  `test-thread-safety` depends on that staying true.
- **`g_sycl_named_weight_cache_uuids`** (verified individually, ticket
  suspect): keyed by a signature string built from `name + type + ne +
  nbytes` — a pure function of the tensor's own identity. A stale entry
  from a different model is either identical (harmless reuse, and in fact
  the point: the same shape gets the same synthetic uuid across model
  reloads) or keyed differently (no collision possible). Not model-scoped
  state at all; correctly never cleared.
- **`g_sycl_weight_identities_by_name`** (verified individually, adjacent
  to the above): every real write is `g_sycl_weight_identities_by_name[name]
  = identity;` — unconditional overwrite, not emplace-if-absent. Staleness
  for a name the current model doesn't have is simply never read. Safe by
  the opposite mechanism from the uuid cache (always-fresh instead of
  pure-function), but safe.
- **`g_runtime_alloc_registry`, `g_runtime_cohort_tier`,
  `g_runtime_reset_reclaimed_allocs`, `g_offload_pool_slots`,
  `g_onednn_scratch_locks`** (`unified-cache.cpp`): all "maintained, no
  bare `.clear()`" in the systematic sweep, but all have paired
  `emplace`/`erase` at every insertion site checked — these track the
  lifetime of individual raw allocations (not model identity), which is a
  legitimate use of pointer keys (this *is* the allocator's own bookkeeping,
  not a downstream cache using a pointer as an identity key). Spot-checked,
  not exhaustively traced through every call path.

## Category (c): model-scoped and leaking — found and fixed

All three fixes are wired into a single structural entry point,
`ggml_sycl_reset_model_load_scratch_state()`, called from
`ggml_backend_sycl_set_model_loading(true)`'s `depth == 0` branch — the one
place every model load passes through exactly once, before any of the
incoming model's tensors are inventoried. It in turn calls two satellite
resets (`ggml_sycl_reset_layer_map_state()`,
`ggml_sycl_reset_moe_phase_demotion_state()`) defined next to the state they
own, each taking that state's existing mutex.

### 1. `g_sycl_weight_usages` (ticket suspect, confirmed) — real bug

`ggml_backend_sycl_register_weight_usage()` only **emplaces** a name's usage
on first sight, and forces `UNKNOWN` on a **mismatch** against whatever is
already mapped — correct within one model's own tied-weight detection
(e.g. a legacy checkpoint tying `output.weight` to `token_embd.weight`).
Never clearing the map between models means a name a PREVIOUS, unrelated
model forced to `UNKNOWN` for its own reasons poisons a DIFFERENT model's
first, and only, registration of that same name: `it->second != mapped`
reads true against the stale `UNKNOWN`, so the new model's real
classification is silently discarded before it is ever used, corrupting
layout selection. Fixed by clearing the map (and its sibling warn-once
dedup set, `g_sycl_usage_unknown_once`) at the load boundary.

### 2. `g_layer_map_initialized` / `g_layer_on_cpu` / `g_layer_plan_forced` / `g_layer_classified` — real bug, found via the systematic sweep, not on the ticket's suspect list

`g_layer_map_initialized` is a double-checked-locking "run once" guard
around `build_layer_device_map()` / `seed_layer_plan_classification()`
(two call sites, both gated identically). It was set once and **never
reset**, making it a "run once per **process**" guard when its own callers'
logic assumes "run once per **model**": in a sequential sweep, only the
first-loaded model ever ran `build_layer_device_map()` (which correctly
does `g_layer_on_cpu.assign(max_layer + 1, false)` — full overwrite, not
`resize()`), so every architecture loaded after the first kept the first
model's per-layer CPU/GPU-offload placement decisions, sized to the first
model's layer count. A later model with fewer layers reads a previous
model's stale offload decision for any layer index the first model also
had; a later model with more layers silently never classifies the extra
layers (every read site bounds-checks against `.size()`, so this fails
quiet rather than crashing) — the same "OK but wrong" shape as the
motivating KV-sizing defect, this time in CPU-offload dispatch routing
rather than KV allocation. Fixed by resetting the atomic flag (the
load-bearing part) and clearing the three vectors (defense in depth; every
read site bounds-checks, so an empty vector between reset and the next
lazy rebuild reads as the correct cold-start "not yet classified").

### 3. `g_moe_phase_tiled_demoted` / `g_moe_phase_i8_demoted` — real bug, found via the systematic sweep, low severity

Both are `unordered_set<const ggml_tensor *>`, insert-only, keyed by raw
tensor pointer — the exact anti-pattern CLAUDE.md's SYCL Memory Ownership
section names directly ("raw pointers are not ownership tokens... do not
use pointer addresses as cache keys"). Tensor object addresses are
allocator-owned and reused across model loads, so a new model's tensor that
happens to land at an address a previous model's tensor was demoted at
reads as already-demoted and skips a rematerialization attempt it never
actually failed. **Perf-only** (a missed optimization — the tensor stays on
SOA a step longer than necessary — not a wrong result), fixed opportunistically
because it was cheap and matches the documented anti-pattern exactly.

### Defense-in-depth (no confirmed live leak found, but structurally strengthened)

`g_placement_kv_info`, `g_has_placement_plan` / `g_placement_plan`,
`g_placement_envelope` / `g_placement_envelope_set`, `g_model_n_layer`,
`g_moe_n_experts_total` / `_used`, `g_moe_expert_total_bytes`,
`g_moe_expert_vram_reserve`, and the `g_tensor_inventory*` family (ticket
suspects `g_placement_kv_info` / `g_has_placement_plan`): traced
field-by-field against `populate_inventory_globals()`, the real writer.
**Every field of every one of these is already overwritten unconditionally
on the normal path** — this is not the same bug as `g_sycl_weight_usages`
(insert-if-absent); it's plain assignment, every time, covering every
struct member. The risk is structural rather than observed: both callers of
`populate_inventory_globals()`
(`ggml_backend_sycl_compute_placement_plan_early()`,
`ggml_backend_sycl_set_tensor_inventory()`) return early — skipping the
writer entirely — when `backend`/`ctx`/`inventory` is null or malformed,
which would leave every field holding the previous model's values. Whether
any of the 131 architectures actually take that early-return path was not
established (would require tracing the llama-model-loader call sites per
architecture, out of scope for a static audit). Reset unconditionally at
the load boundary regardless, since the fix is nearly free and removes the
"every future writer must remember to cover every field" invariant this
family depended on.

## Category (d): flagged, not resolved — follow-up candidates

Found by the same systematic clear()/erase() sweep, but not individually
traced far enough to confirm or rule out a leak within this pass's time
budget. Listed with the read pattern that would need checking:

| symbol | file | key | risk note |
|---|---|---|---|
| `g_moe_layer_seq[GGML_SYCL_MAX_DEVICES]` | ggml-sycl.cpp:1978 | `int` (layer id) | per-device array of maps; stale sequence numbers for a layer id present in both an old and new model would need checking against whatever writes it during load |
| `g_expert_popularity` | ggml-sycl.cpp:2389 | `int64_t` | popularity/heuristic counter for expert prefetch; likely soft (biases a heuristic, doesn't gate correctness) but not confirmed |
| `g_sycl_canonical_checksums` | ggml-sycl.cpp:7226 | tensor name | name suggests diagnostic/debug-only; not confirmed |
| `g_pending_kv_layer_masks[GGML_SYCL_MAX_DEVICES]` | ggml-sycl.cpp:9241 | n/a (deque) | uses `pop_front`/`pop_back`, not `clear`/`erase` — the sweep's `.clear(`/`.erase(` grep is a **false positive** here; likely a drained queue, not accumulating state. Re-verify with a positive-controlled probe before trusting either way |
| `g_moe_layer_ids_cache` | ggml-sycl.cpp:11033 | `int` (layer id), `thread_local` | `thread_local` does not help against a *sequential* single-thread sweep like test-llama-archs |
| `g_moe_expert_biases` / `g_moe_bias_host_copies` | ggml-sycl.cpp:14138-14140 | `int` (layer id) | per-layer MoE expert bias values; a later non-MoE or differently-shaped model could read a previous model's stale bias data if any downstream check is "does this map contain layer N" rather than "did THIS load populate layer N" |
| `g_sycl_alloc_trace_entries` | ggml-sycl.cpp:31984 | tensor name | name suggests diagnostic/trace-only; not confirmed |

Recommended next step: for each row, find the read site(s), confirm whether
the read is gated by something equivalent to "this model actually wrote
this entry" (safe) or is a bare `find()`/`count()` against process lifetime
(same shape as the three confirmed bugs). The `g_pending_kv_layer_masks`
row should be re-run through the sweep with `pop_front`/`pop_back` added to
the maintenance-pattern check before it is trusted either way — as flagged,
it is exactly the kind of empty-probe result CLAUDE.md warns about.

## `fattn.cpp:g_packed_k_sidecars` (assessed, not owned, flagged by the lead)

The lead's gemma3n investigation flagged
`ggml/src/ggml-sycl/fattn.cpp:124`'s
`std::vector<std::unique_ptr<ggml_sycl_fattn_xmx_packed_k_sidecar_entry>>
g_packed_k_sidecars` as a suspect: zero `.clear()` calls, entries carrying a
raw `const void * k_base`. `fattn.cpp` is outside this pass's ownership
(`ggml-sycl.cpp` / `unified-cache.{cpp,hpp}` only), so this is assessment
only, not a fix — but the `.clear()`-only grep that raised it is the same
shape of empty probe this document warns about for `g_pending_kv_layer_masks`
above, so it was worth checking before passing the flag along as-is.

Reading the actual read/write/erase sites (`fattn.cpp:340-357`,
`430-470`, `553-566`) changes the picture:

- The correctness-determining lookup (both the cache-hit check and the
  insert-or-reuse check) keys on `entry->k_handle.valid() &&
  entry->k_handle_hash == params.K_handle_hash` — a `ggml_sycl::mem_handle`
  and its stable identity hash, i.e. exactly the mechanism CLAUDE.md
  prescribes in place of a raw pointer. `k_base` is stored on the entry but
  is not what a lookup matches against.
- There IS an erase path:
  `ggml_sycl_fattn_xmx_unregister_packed_k_range(ptr, size)` removes every
  entry whose `k_base` falls inside `[ptr, ptr+size)`. It is not dead code —
  it has a real caller: `tiered_kv_buffer_free()` in `ggml-sycl.cpp`
  (grep-verified with `cat | grep` per the oversize-file rule; `search_text`
  reports zero callers for this symbol because it silently skips
  `ggml-sycl.cpp`, which is exactly where the caller lives), invoked for
  every layer allocation on both the vmem-pool and non-vmem KV-buffer
  teardown paths. That is erase-on-free, wired to the actual KV buffer
  lifetime — the correct way to prevent the address-reuse aliasing this
  audit's other findings suffer from, not an instance of it.

**Assessment: likely not a leak**, on the strength of the above, but not
fully confirmed within this pass — two things weren't traced: (1) whether
`tiered_kv_buffer_free()` is the *only* teardown path for every KV buffer
across all 131 architectures (a KV buffer freed through a different code
path would skip the unregister call), and (2) the whole feature is gated
behind `ggml_sycl_fattn_xmx_sidecar_enabled()`
(`GGML_SYCL_PACKED_K_SIDECAR` or `GGML_SYCL_FA_FORCE_PATH=split-packed`),
neither of which the default `test-llama-archs` invocation sets — so this
code path is dormant in the sweep that motivates this whole audit, and any
residual risk only surfaces for someone running with that env var set.
Flagging both as open questions rather than asserting either way.

## `layer_stream_manager` host_ptr registry (assessed, not owned)

The ticket asked this audit to also assess
`ggml/src/ggml-sycl/layer-streaming.cpp:137-158`'s name-keyed `host_ptr`
registry. That file is outside this pass's ownership (not
`ggml-sycl.cpp`/`unified-cache.*`), so it was not edited. Flagging for
whoever owns that file to run the same check this audit ran everywhere
else: does the registry get cleared at the model-load boundary, or only
grow/overwrite by name for the process lifetime?

## The regression test

`ggml/src/ggml-sycl/tests/test-cross-model-weight-usage.cpp` (registered in
`ggml/src/ggml-sycl/CMakeLists.txt` as ctest `cross-model-weight-usage`)
loads two different "models" through one process via the real
`ggml_backend_sycl_set_model_loading()` boundary and
`ggml_backend_sycl_register_weight_usage()` /
`ggml_sycl_get_tensor_usage()`, and asserts model B's classification does
not inherit model A's stale state. Four checks: a positive control
(first-sight registration works), a precondition check (model A's own
tied-weight case really does force `UNKNOWN`), a negative control (a name
both models use consistently survives the reset either way), and the bug
check itself (model B's first, and only, registration of a name model A
had forced to `UNKNOWN` gets model B's own answer, not the stale one).

**Mutation that proves specificity**: commenting out
`g_sycl_weight_usages.clear();` inside
`ggml_sycl_reset_model_load_scratch_state()` makes check 4 fail
("model B's shared name gets its OWN usage, not model A's stale UNKNOWN")
and only check 4 — checks 1-3 stay green, including the negative control,
which is specifically there to demonstrate the mutation doesn't just break
the reset wholesale.

This test needs a real SYCL device (it exercises the real load boundary,
which calls into `ggml_sycl_info()`) — it was not run by this pass per the
"no GPU work" constraint. See the report to `main` for what to run.
