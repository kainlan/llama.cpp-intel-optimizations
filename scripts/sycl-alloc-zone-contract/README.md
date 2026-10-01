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
`E-DPCT-DEVMEM-DEVICE`, `E-DPCT-DEVMEM-SHARED`; canonical contract section 9.1, the dpct row, with rulings M247 second). Each pins its function and count, so a second call, a
renamed function or a swapped allocator fails. Outside `dpct/helper.hpp`
the names `dpct_malloc` (identifier) and `device_memory`, `global_memory`, `constant_memory`, `shared_memory` (type
names) are clause (e) hits; a variable or parameter that is merely spelled `device_memory` is not.

Host raw allocator names (S2c): `malloc_host`, `aligned_alloc_host`, `zeMemAllocHost`, `sycl::malloc` and `sycl::aligned_alloc`
(qualified only) and the host chain's `unified_cache_raw_malloc_host` / `unified_cache_malloc_host_tracked` are clause (e) names.
Their six hits in `unified-cache.cpp` are allowlisted (`E-CHAIN-HOST-RAW`, `E-CHAIN-HOST-TRACKED`, `E-HOST-USM-BASE`, and the two
CACHE_BACKING bootstrap sites `E-BACKING-STAGING`, `E-BACKING-FLAG-SLAB`; canonical contract sections 3, 3.1 and 9.1); the
seventh, `unified_cache::allocate`'s last-resort fallback, is E-RAW debt with fate `deleted-by-D-disposition`.

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

Every `E-RAW` debt entry also carries `fate` and `cite`. `fate` is `deleted-by-<step>`, `converted-by-<step>`,
`sanctioned-internal`, `sanctioned-vendored` (upstream code we do not edit) or `pending-disposition`. A `sanctioned-internal` entry is one no step will ever shrink:
it is a candidate for the allowlist, and moving it there is the lead's decision, not the implementer's.
`pending-disposition` marks an entry whose fate nobody has ruled on yet; its `cite` says what is known. Every E-RAW entry needs a `cite`, a ticket id or a design/census row of at
least 12 characters.

## Clauses (i)-(o) and witness 9 (S2d)

Each is keyed on a subject that may not exist yet. Today's tree has no `ggml_sycl_replan_token_held` definition, no COMPLETE
reap in `ggml_sycl_run_runtime_context_transaction`, no `owner_use_count` or `replace_within`, no `onednn_pp_a_bytes` /
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
  design appendix at master `2c4f5e45d` rev 4.16, because the appendix is not in the repository. Regenerate it, never hand-edit it:
  `python3 scripts/sycl-alloc-zone-contract/gen-appendix-rows.py --design <design.md> --master 2c4f5e45d --rev 4.16`. The gate's own tables (`M_TABLES`: the floor list with each row's owner ticket, the
  covered-by-peak rows, the unreachable rows) are checked against it, and `ensure_planned_arena_zones` must apply the
  `GGML_SYCL_COMPUTE_ARENA_MB` floor while the floor list is non-empty (and must not once it is empty).
- (n) `N-VOID` and `N-NODISCARD`, the only S2d codes that may be debt. The names are the declined-result consumers
  (`DnnlGemmWrapper::gemm`, `row_gemm`, `woq_gemm_*`, the softmax / eltwise / binary wrappers, `get_scratchpad_mem`,
  `ggml_sycl_mul_mat_batched_sycl`); a class member matches as `Class::name(` anywhere or a bare `name(` inside that class.
  Test sources that spell a listed name are read for this clause only (keys are `tests/...`); clauses (a)-(h) never see them.
  Reading of the spec: a call fails only when its value is discarded (an expression statement, the left of a comma, a cast to
  `void`); an assignment, initializer, return, condition or argument counts as consuming it.
- (o) `O-NOROW`, `O-ROW`, `O-QUEUE`: each call of an acquire token (`acquire_onednn_pp_scratch`,
  `ggml_sycl_set_rows_stage_ptr`, the oneDNN graph `execute`) must sit in a function with a census row (`O_ROWS`), and each
  consuming call of that row pins its queue argument by position and exact text; a callee that takes no queue pins its own
  stream declarations.
- Witness 9 `Z9-SITE`, `Z9-SIZING`: each of `load_reorder_temp_bytes`, `woq_packed_bytes`, `mmq_work_counter_bytes`,
  `set_rows_stage_bytes` that is defined must be called by each of its allocation sites and by `zone-sizing.cpp` or
  `unified-cache.cpp`.

Gaps stated rather than hidden (the gate prints a `TODO j`, `TODO l` and `TODO p-route` line for the narrowings, so they are visible
in its output):  the names of (j)'s fit function, reserve target and selector bit reads are not fixed yet, so only
the two `*_bytes` names are covered; (l)'s five count-caller functions do not exist, so the allowlist entries are added as each
lands. Today's 24 N-* debt entries were seeded once with `--write-debt --allow-growth` (834 to 858 entries).

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

Stated gaps: the route's D=512 hatch is exempt from the head-dim literal rule as any `if` whose condition spells 512 and calls
`ggml_sycl_fa_onednn_d512_enabled`; "the routing function reads the decline before the plan" is not checked (the decline reader
has no name); the charge side's walk helpers are unnamed, so the head-dim and helper rules cover the bodies of the two named
charge functions and the helper-name rule covers all of `unified-cache.cpp` and `.hpp`.

## Key shape

`file::function::node-kind:variable:text-hash#ordinal`. The text hash is of the construction's normalized
declaration text plus the text of every later assignment bound to that declaration (comments dropped,
whitespace collapsed), so a write added to, or removed from, a listed construction moves its key. The ordinal counts only identical VIOLATING constructions within the
same function, so adding a compliant twin cannot move a listed key. No line or byte offset appears, so adding code above a construction does not move its key;
changing the construction does, which is when its entry should be revisited.

## Tests and sharding

Five ctests register the gate (`ggml/src/ggml-sycl/CMakeLists.txt`): `test-sycl-alloc-zone-contract` is the plain gate (fast), and
`test-sycl-alloc-zone-contract-m0` .. `-m3` run the mutation matrix in four shards (`--mutation-matrix --shard K/N`, every n-th
case from K, TIMEOUT 600 each). Every shard re-checks the unmutated baseline; shard 0 also runs the coverage checks (every
witness has a FAIL case), the process-level witnesses and the check of this registration. The matrix costs about 0.7 s a case,
so grow N (and the `foreach` list, which the check pins) before a shard approaches its TIMEOUT.

## Dependency

The gate parses C++ with tree-sitter: `pip install tree-sitter-language-pack` in the python3 that CMake finds.
A missing module is a FAIL naming it, never a skip.

## Shrinking the debt

Fix the violation, then delete its entry in the same commit. To rewrite the file:

    python3 scripts/check-sycl-alloc-zone-contract.py --write-debt

`--write-debt` refuses to add entries. `--allow-growth` exists for a deliberate key migration only (a change to
the key shape); do not use it to admit a new violation. The ctest never passes either flag.
