# sycl-alloc-zone-contract data

Read by `scripts/check-sycl-alloc-zone-contract.py` (ctest `test-sycl-alloc-zone-contract`,
llama.cpp-23mk S2). JSON has no comments, so each file also carries a `_doc` field.

- `allowlist.json`: permanent exemptions, each with a `reason`. An entry names its subject, either
  `file` + `function` or a node `key`, plus the raw `name` for an E-RAW entry, and pins an exact `count` (the
  count is required). `name` is what stops a different allocator swapped into an exempt function from riding
  the exemption. An entry that matches nothing fails (a renamed exempt function protects nothing), and so does
  one whose count differs from what it covers. Hand-edited.
- `debt.json`: the tree's current violations, keyed by construction node. Shrink-only in both directions:
  a violation that is not listed fails, and a listed entry that no longer violates fails, naming it.

Allowlist entries added in S2b (clause e): the raw allocator chain's three links in `unified-cache.cpp`
(`E-CHAIN-ALIGNED`, `E-CHAIN-TRACKED`, `E-CHAIN-RAW`; canonical contract sections 3 and 9.1, 23mk census row
`:18789/:18830/:1433`) and the three raw calls in vendored `dpct/helper.hpp` (`E-DPCT-MALLOC`,
`E-DPCT-DEVMEM-DEVICE`, `E-DPCT-DEVMEM-SHARED`; canonical contract section 9.1, the dpct row). Each pins its function and count, so a second call, a
renamed function or a swapped allocator fails. Outside `dpct/helper.hpp`
the names `dpct_malloc` (identifier) and `device_memory`, `global_memory`, `constant_memory`, `shared_memory` (type
names) are clause (e) hits; a variable or parameter that is merely spelled `device_memory` is not.

Host raw allocator names (S2c): `malloc_host`, `aligned_alloc_host`, `zeMemAllocHost`, `sycl::malloc` and `sycl::aligned_alloc`
(qualified only) and the host chain's `unified_cache_raw_malloc_host` / `unified_cache_malloc_host_tracked` are clause (e) names.
Their six hits in `unified-cache.cpp` are allowlisted (`E-CHAIN-HOST-RAW`, `E-CHAIN-HOST-TRACKED`, `E-HOST-USM-BASE`, and the two
CACHE_BACKING bootstrap sites `E-BACKING-STAGING`, `E-BACKING-FLAG-SLAB`; canonical contract sections 3, 3.1 and 9.1); the
seventh, `unified_cache::allocate`'s last-resort fallback, is E-RAW debt with fate `deleted-by-D-disposition`.

Debt that came in from master: `unified-cache.cpp::runtime_registry_claim_ptr_locked`'s G-CATCH entry (master's 93tw) is a
trace-only `try { report.cohort = it->second.cohort_id; } catch (...) {}` that wraps a `std::string` copy, so it can only swallow a
`bad_alloc` of a diagnostic string. It is not 23mk's to fix; it stays debt until that code is converted.

The W7 merge of master `ec4e4eba7` brought more of the same, regenerated with `--write-debt --allow-growth` (key migration
and code that arrived from master, not new lane code). Re-keyed by a master rename, same handler or request: `graph_preload_moe_experts` to
`graph_preload_moe_experts_impl` (7pm2 Stage A, its D-FORBID and D-ZONE pair) and `flush_pending_cpu_scatter` to
`flush_pending_cpu_scatter_slot` (its G-CATCH, which aborts and cannot swallow). Arrived as master code: a fifth `catch (...)` in
`ggml_backend_sycl_graph_compute_impl`; the `catch (...)` that erases and rethrows in `runtime_registry_emplace_locked`; and the noexcept
test hooks `allocation_registry_test_assign_raw`, `_assign_host`, `_publish_host` and `_publish_irregular` (G-CATCH). Retired: the
D-FORBID and D-ZONE pair of `ggml_sycl_bf16_weight_materialize_f32`, a function master removed.

The W7 merge of master `63dfd237d` (7pm2 Stage B, the keyed per-split segment-graph cache) added six G-CATCH entries the same way.
None of them can swallow a planned refusal, and none is lane code:
- `moe_graph_record_segment_slot`'s `std::exception` handler follows a `ggml_sycl_fallback_error` handler that invalidates the Q8 cache
  and then rethrows. The clause recognises only the empty `{ throw; }` body, so it reports this one.
- The `catch (...)` around that record call in graph compute drains, then rethrows. The `catch (...)` inside it wraps only that drain,
  in a failure path. Both are keyed `<file scope>`: the clause does not resolve the lambda that encloses them.
- `ggml_sycl_retire_moe_segment_slots` is `noexcept`, and `recording_end_guard`'s destructor only ends a recording. A rethrow in
  either would terminate the process.
- `moe_segment_slots_drain_retired`'s `std::exception` handler wraps only the retire drain. It keeps the slots alive and reports failure.

Clause (q), libc allocation primitives (S3-0): `mmap`, `mmap64`, `mremap`, `posix_memalign`, `memalign`, `aligned_alloc`, `valloc`,
`pvalloc`, `malloc`, `calloc`, `realloc`, `reallocarray`, `strdup`, `strndup` and `VirtualAlloc`, called bare or through `std::` / `::` (or taken as a value, or spelled in a `#define` body), are E-LIBC findings. A member
(`pool.realloc`), a name qualified by another scope (`sycl::malloc` stays clause (e)'s) and a declaration are not. Five allowlist entries
cover seven sites: the vendored `dpct/helper.hpp` hits (`E-LIBC-DPCT-MMGR-MMAP`, `E-LIBC-DPCT-MMGR-VIRTUALALLOC`,
`E-LIBC-DPCT-HOSTBUF-MALLOC`, `E-LIBC-DPCT-DEVMEM-MALLOC`; reason "vendored dpct, unreachable from the backend"; canonical contract section 9.1) and
`cache_guard_allocator`'s `mmap` (`E-LIBC-CACHE-GUARD`; permanent, "cache bookkeeping, guard-page debug mode, no tensor/KV/scratch/pinned/USM
bytes"). The dead `weight_cache_allocator`'s `mmap` and `posix_memalign` are E-LIBC debt with fate `deleted-by-step-7` (S7's zero-caller
step 7's "dead weight_cache_allocator, whole" bullet in the 23mk design). The S2c control that used to pin `std::malloc` as a PASS flipped
to a FAIL on purpose when this clause landed.

"Unreachable from the backend" is a gated claim, not a hope: clause (e) forbids `dpct_malloc`, the dpct memory classes and the two
`dpct_memcpy` entry points (`dpct_memcpy`, `async_dpct_memcpy`) outside `dpct/helper.hpp`. The ban is on the whole name, so it also bars
the 1-D overload (which never touches `host_buffer`) along with the 3-D host-staged paths that are `host_buffer`'s only users; failing
closed on the 1-D overload is deliberate. A backend caller of any path that reaches an allowlisted dpct libc site is itself a finding.

In a `#define` body and in the lexical pass of an ERROR-root file there is no tree, so the qualifier is read backwards across spaces,
newlines and `\`-continuations. The whole scope is judged, as the tree path does: `sycl :: malloc`, `pool :: realloc`, `xstd::malloc`,
`ns::std::malloc` and `T<x>::malloc` are scoped (`sycl::malloc` is clause (e)'s), `p . malloc` and `p->malloc` are members, and
`std :: malloc`, `::std::malloc` and a bare `:: malloc` are hits. A C++ keyword before `::` (`return ::malloc(n)`, `else ::malloc(n)`,
`throw`, `sizeof`, `co_return`) is not a scope, and neither is a comparison (`a > ::malloc(n)`); a template-id counts as a scope only when
its `<` follows an identifier, so a contrived `x < y > ::malloc(n)` reads as a template (fails open; nothing like it is in the tree).

The lexical pass (an ERROR-root file, today `cpu-dispatch.cpp`) is call-shaped: it flags a `name(` and so also a declaration-shaped
`malloc(`, which fails closed. A name taken as a value (`auto f = ::malloc;`) is not seen by the lexical pass; the tree pass still finds
it wherever the parser yields an identifier node, which it does today. If a future ERROR-root file loses that, a value use would pass.

Fail-closed false positives, cleared by a rename: a local variable, a parameter or a lambda named `mmap`, `malloc`, `realloc`, `strdup`
and so on that is *called* or *taken as a value* is an E-LIBC finding, because the clause matches the identifier and does not resolve
scope; and so is an implicit-`this` member call of a method with such a name (`realloc(p, n)` inside a class that declares `realloc`),
which the tree cannot tell from the C library's. Rename the local or call it as `this->realloc(...)`; do not allowlist it.

Out of scope, on purpose: `free` (releasing is not allocating), `new`, `operator new` and the STL containers' own allocators. This is a
boundary the gate states, not one it proves: a `new` of a request type is clause (a)'s B-FORM, but a `new` or a container that holds
tensor, KV, scratch, pinned or USM bytes is a `mem_handle` ownership question and is reviewed as one.

Known gap: an aliased namespace (`namespace sy = sycl; sy::malloc(n, q, sycl::usm::alloc::host)`) escapes the qualified
`sycl::malloc` / `sycl::aligned_alloc` check, since the match is on the spelled scope. The named forms (`malloc_host` and the
rest) are matched wherever they appear.

Finding codes added by clause (c): `C-COHORT` (a copy of a request that is handed on without its own cohort
literal) and `C-SITE` (a wrapper from one request type to another that does not copy the source's site fields).
`DEFER-C` now marks only what the gate still cannot follow (a helper with several returns, a lambda or function-pointer call).

Clause (g), `G-CATCH`: every `catch (...)`, `catch (std::exception &)` and `catch (const std::exception &)` (any spelling:
east const, by value, with or without `std::`) must be preceded in the same try by a `ggml_sycl_fallback_error` handler whose
body is exactly `throw;`. The same pattern runs over the `catch_clause` nodes and over every `#define` body (a macro handler
is keyed `(file, #define NAME)`; its guard must come earlier in the body with no `try` between). The unguarded handlers
today are the shrink-only debt, keyed `file::function::catch_clause:<all|exception>#ordinal`, so a new handler in a listed
function takes the next ordinal and fails. Exemptions are allowlist entries `code: "G-CATCH"` by `file` + `function`
with a `count` and a reason (`file` + `function: "#define NAME"` for a macro).

Clause (h), `H-*`: `cascade_step` and `unconverted_ticket` have no writer today, so every write is a finding. Each is
exempted by an allowlist entry keyed by the construction or call node (the declaration of the object whose field is written,
never the function), with an `outcome` (`CASCADE`, or `UNCONVERTED` for `H-UNCONV`, which also names the `ticket` it
sets). A `TERMINAL` outcome is refused. `H-CASCADE` is `cascade_step = true`; `H-CASCADE-PARAM` is a copy of the enclosing
function's own `cascade_step` parameter, exempted by `file` + `function` for an allowlisted callee; `H-PASS-TRUE` and
`H-PASS-FORWARD` are a call that passes `true` or forwards its own parameter to a function that takes one. The `*-EXPR` codes
(any other expression) cannot be exempted, and no `H-*` finding can be debt (`--write-debt` refuses it). An allowlisted node
that stops writing the field fails as an entry matching nothing, which is witness 21's check that a DECLARED
construction keeps its flag.

Every `E-RAW` and `E-LIBC` debt entry also carries `fate` and `cite`. The one other debt entry that carries a `fate` is the `G-CATCH` entry
for the `CHECK_TRY_ERROR` macro's handler, and its fate is `converted-by-5.4a` and nothing else (the gate pins both directions: that key must
carry it, and no other entry outside E-RAW/E-LIBC may carry any fate). `fate` is `deleted-by-<step>`, `converted-by-<step>`,
`sanctioned-internal`, `sanctioned-vendored` (upstream code we do not edit) or `pending-disposition`. A `sanctioned-internal` entry is one no step will ever shrink:
it is a candidate for the allowlist, and moving it there is the lead's decision, not the implementer's.
`pending-disposition` marks an entry whose fate nobody has ruled on yet; its `cite` says what is known. Every E-RAW and E-LIBC entry, and the CHECK_TRY_ERROR entry, needs a `cite`, a ticket id or a design/census row of at
least 12 characters. An allowlist reason names no source line (`file.ext:1234`, `file.ext :12`, `file.ext#L12`, which rot with the next edit: name the
function), and neither a reason nor a cite names a ruling-ledger id (`ruling M265 R2`, `ruling M265, R2`, `rulings M247`, `§M243`, a bare
`M265`, `(R2)`: a ledger that is not in the repo); `validate_data` refuses both. A debt cite is allowed a census row id such as
`ggml-sycl.cpp:24338->:43022`, because it keys a row of the design's census table at its stated base, and a reason may carry a bare
`:18789` row id for the same reason; the source-line refusal is for a reason that points at code. A second `G-CATCH` debt key that names the
CHECK_TRY_ERROR macro (a re-key of the pinned one) is refused too, so the fate pin cannot lapse by renumbering.

## Clauses (i)-(o) and witness 9 (S2d)

Each is keyed on a subject that may not exist yet. Today's tree has no `ggml_sycl_replan_token_held` definition, no COMPLETE
reap in `ggml_sycl_run_runtime_context_transaction`, no `replace_within` (`owner_use_count` exists since S3-1, so clause (l) enforces its callers; L-FREE has no `replace_within` yet and is vacuous), no `onednn_pp_a_bytes` /
`onednn_pp_w_bytes`, and none of the four model-shaped `*_bytes()` functions, so those clauses print
`DORMANT <clause>: subject <symbol> absent` after the report: a visible state, not a pass. They wake the moment the subject is
defined. A subject name that appears in a shape the matcher does not read (a `#define` body, a lambda or variable, an alias, a
pointer to member) is an `X-LATCH` failure, so a respelling cannot keep a clause dormant for ever; X-LATCH is never debt.

- (i) `I-RETRY`: defining `ggml_sycl_replan_token_held` (a call or a `;` declaration does not count) while
  `onednn_w_retry_lost_cas` or `onednn_pp_a_relock_busy_pre_l0` is still defined.
- (k) `K-INTERIM`: `release_retained_referencing(... RETAINED_REAP_EVENTS_COMPLETE_BY_CALLER ...)` inside
  `ggml_sycl_run_runtime_context_transaction` while `onednn_pp_a_reclaim_query_interim` is still defined.
- (l) `L-CALLER` (a caller of `owner_use_count` outside `mem-handle.*` that is not an allowlist entry, `code: "L-CALLER"`,
  `file` + `function`), `L-FREE` (a free, `unified_free` / `zone_free` / `reset` / `enqueue_deferred_zone_free` or a TLSF free,
  whose execution the count controls, directly, through a local, after an early return, or through `&&`/`||`/`?:`),
  `L-GUARD` (`replace_within`'s first statement is not `replace_within_count_guard(...)`).
- (j) `J-SOURCE` (`onednn_pp_a_bytes` / `onednn_pp_w_bytes` reading `g_tensor_inventory_*` or
  `unified_cache_get_planned_(pp_moe_)onednn_*`), `J-DISPATCH` (a call of `zone_is_onednn_reorder_eligible`; the one real
  call, in `zone_scoped_maxima`, is allowlist entry `E-J-CLASSIFIER`).
- (m) `M-SCRATCH`, `M-STALE`, `M-FLOOR`, `M-DATA`. `appendix-rows.json` is the 139-row census table (zone columns) of the
  design appendix at master `2c4f5e45d`, because the appendix is not in the repository. The design in force is rev 4.19z, whose
  appendix header still reads 4.16; the file's `_doc` therefore says 4.16, and the generator run against the rev 4.19z design
  reproduces the committed file byte for byte. Regenerate it, never hand-edit it (the generator reads the `row`, `site`, `function`, `zone`, `code`, `core` and `term`
  columns of the table whose header starts `| # | site (master`):
  `python3 scripts/sycl-alloc-zone-contract/gen-appendix-rows.py --design <design.md> --master 2c4f5e45d --rev 4.16`. A missing or
  malformed file, or a row without a key the clause reads, is `M-DATA`, never a traceback. The gate's own tables (`M_TABLES`: the floor list with each row's owner ticket, the
  covered-by-peak rows, the unreachable rows) are checked against it, and `ensure_planned_arena_zones` must apply the
  `GGML_SYCL_COMPUTE_ARENA_MB` floor, by reading it itself or by calling `ggml_sycl_compute_arena_bytes` (the one reader the model-load reservation and the chunk cap's probe set also call; its body must carry the `getenv`), while the floor list is non-empty (and must not once it is empty).
- (n) `N-VOID` and `N-NODISCARD`, which may not be debt (no S2d code may). The names are the declined-result consumers
  (`DnnlGemmWrapper::gemm`, `row_gemm`, `woq_gemm_*`, the softmax / eltwise / binary wrappers, `get_scratchpad_mem`,
  `ggml_sycl_mul_mat_batched_sycl`); a class member matches as `Class::name(` anywhere or a bare `name(` inside that class.
  Test sources that spell a listed name are read for this clause only (keys are `tests/...`); clauses (a)-(h) never see them.
  Reading of the spec: a call fails only when its value is discarded (an expression statement, the left of a comma, a cast to
  `void`, a for-loop increment); an assignment, initializer, return, condition or argument counts as consuming it. The walk
  passes through parentheses, the right operand of `&&` / `||`, the arms of `?:` and the right of a comma, so `ok && f();` and
  `ok ? f() : g();` are discards. A `using` or `typedef` alias of a listed class resolves to it.
- (o) `O-NOROW`, `O-ROW`, `O-QUEUE`: each call of an acquire token (`acquire_onednn_pp_scratch`,
  `ggml_sycl_set_rows_stage_ptr`, the oneDNN graph `execute`) must sit in a function with a census row (`O_ROWS`), and each
  consuming call of that row pins its queue argument by position and exact text; a callee that takes no queue pins its own
  stream declarations.
- Witness 9 `Z9-SITE`, `Z9-SIZING`: each of `load_reorder_temp_bytes` (six sites, the sixth being
  `unified_cache::reserve_reorder_temp`), `woq_packed_bytes`, `mmq_work_counter_bytes`, `set_rows_stage_bytes` that is defined
  must be called by each of its allocation sites and, from a function that is not one of those sites, by `zone-sizing.cpp` or
  `unified-cache.cpp`.
- Latches: a subject that appears in a shape a clause cannot read is an `X-LATCH` finding. That covers a macro, a lambda or
  variable, a pointer to member, and a name declared as a type (`using`, `typedef` of a function pointer, a functor struct).
  Clause (k) latches its transaction, its reap and its mode constant (a `#define` of the mode, or a constant initialised from it);
  clauses (i) and (j) latch the retired functions and the eligibility call. No `X-LATCH` or `P-*` finding can be debt or
  allowlisted: `validate_data` refuses both.

Gaps stated rather than hidden (the gate prints a `TODO j`, `TODO l` and `TODO p-route` line for the narrowings, so they are visible
in its output):  the names of (j)'s fit function, reserve target and selector bit reads are not fixed yet, so only
the two `*_bytes` names are covered; (l)'s five count-caller functions do not exist, so the allowlist entries are added as each
lands. Today's 24 N-* debt entries were seeded once with `--write-debt --allow-growth` (834 to 858 entries). S3-3 retired the nine oneDNN softmax / eltwise / binary wrapper entries (860 to 851): the wrappers return a `[[nodiscard]] bool` and their callers test it, pinned by `scripts/check-sycl-dnnl-decline-consumers.py`.

## Clause (p): one routed predicate, one home for the support decision

Dormant until beni's b1 lands: `ggml_sycl_fattn_onednn_route_admits`, `ggml_sycl_flash_attn_ext_enabled`,
`ggml_sycl_fattn_kv_pair_of`, `ggml_sycl_kv_cache_layer_of` and `placement_plan_set_routed_head_maxima` /
`onednn_graph_scratch_bytes` are all undefined today, so the five blocks print `DORMANT p-route`, `p-home`, `p-fill`, `p-layer`
and `p-charge`. Each wakes on its own subject. The gate reads the whole repository once for the switch
`getenv("GGML_SYCL_FLASH_ATTN_EXT")` (every C or C++ source outside `docs/`, `.llm-wiki/`, `build*/`; keys `repo/...` and `tests/...`),
and `kv_is_fp8` writes and `"cache_[kv]_l` spellings under the scope. Findings: `P-ROUTE`, `P-HOME`, `P-FILL`, `P-LAYER`,
`P-CHARGE`, none of which may be debt.

The matrix cannot plant a twin beside functions the clause constrains in place, so witnesses 37 and 38 transform today's
tree into the b1 tree (`b1_tree`) and the b2 tree (`b2_tree`) the clause describes, then mutate that. Each transform is an exact
anchored edit: when a tree edit moves an anchor, the matrix fails with a setup error naming it, and the fixture is re-derived.
When beni b1 lands, the dormant lines turn into active checks of the real tree and the fixture's b1 transform becomes a no-op
to delete.

The route's D=512 hatch is exempt from the head-dim literal rule only in its exact text: the latch
`static const bool V = ggml_sycl_fa_onednn_d512_enabled();` and either `if (HD == 512 && !V) { return false; }` (the two `&&`
operands in either order, the head dim on the left of `==`) or `if (HD == 512) { if (!V) { return false; } }`; any other
head-dim literal beside it is a finding. Spellings the exemption does not recognise fail loudly as `P-HOME` head-dim-literal
rather than pass: `512 == p.ne00 && !V` (the literal on the left), a latch written `static const bool V{...}` or
`const static bool V = ...`, and a latch that is not the one function-local `static const bool`. Respell the hatch as the
pinned text, or extend `P_LATCH_RE` and `p_blank_hatch`. The routing function must read
`ggml_sycl_onednn_graph_dispatch_declined` before its first call of the routed predicate, and its whole body is pinned to
the design's text, parameter and local names included (whitespace and comments aside), and every call of
the predicate in the value function ends `..., nullptr, nullptr`.

Stated gaps: the charge side's walk helpers are unnamed, so the head-dim and helper rules cover the bodies of the two named
charge functions and the helper-name rule covers all of `unified-cache.cpp` and `.hpp`.

## Key shape

`file::function::node-kind:variable:text-hash#ordinal`. The text hash is of the construction's normalized
declaration text plus the text of every later assignment bound to that declaration (comments dropped,
whitespace collapsed), so a write added to, or removed from, a listed construction moves its key. The ordinal counts only identical VIOLATING constructions within the
same function, so adding a compliant twin cannot move a listed key. No line or byte offset appears, so adding code above a construction does not move its key;
changing the construction does, which is when its entry should be revisited.

## Tests and sharding

Ten ctests register the gate (`ggml/src/ggml-sycl/CMakeLists.txt`): `test-sycl-alloc-zone-contract` is the plain gate (fast),
`test-sycl-alloc-zone-contract-w` runs `--witnesses` (the coverage checks, the process-level witnesses that each spawn the gate,
and the check of this registration), and `test-sycl-alloc-zone-contract-m0` .. `-m7` run the mutation matrix in eight shards
(`--mutation-matrix --shard K/N`, every n-th case from K, TIMEOUT 600 each). Every shard re-checks the unmutated baseline.
The rule for N: the worst shard must take at most a third of its TIMEOUT under host load, so grow N (and the `foreach` list,
which the check pins) as the matrix grows, in preference to raising a TIMEOUT.

## Dependency

The gate parses C++ with tree-sitter: `pip install tree-sitter-language-pack` in the python3 that CMake finds.
A missing module is a FAIL naming it, never a skip.

## Shrinking the debt

Fix the violation, then delete its entry in the same commit. To rewrite the file:

    python3 scripts/check-sycl-alloc-zone-contract.py --write-debt

`--write-debt` refuses to add entries. `--allow-growth` exists for a deliberate key migration only (a change to
the key shape); do not use it to admit a new violation. The ctest never passes either flag.
