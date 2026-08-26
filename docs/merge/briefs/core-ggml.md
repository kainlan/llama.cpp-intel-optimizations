# Conflict brief — core ggml (Task 10)

Pre-merge analysis for wave 2 of the upstream merge (`docs/plans/2026-08-25-phase-c-upstream-merge.md`).
Refs: merge-base `81ff7abe5`, fork tip `master`, upstream tag `b10630` (remote `ggml-org`).

Derivation command (verbatim from the plan):

```bash
comm -12 <(git diff --name-only 81ff7abe5 master | LC_ALL=C sort) \
         <(git diff --name-only 81ff7abe5 b10630 | LC_ALL=C sort) \
  | grep -E '^ggml/' | grep -vE '^ggml/src/ggml-sycl/|CMakeLists\.txt|\.cmake$'
```

Output — **26 files** (count matches; every one has an entry below):

```
ggml/include/ggml-backend.h
ggml/include/ggml-rpc.h
ggml/include/ggml.h
ggml/src/ggml-alloc.c
ggml/src/ggml-backend-meta.cpp
ggml/src/ggml-backend-reg.cpp
ggml/src/ggml-backend.cpp
ggml/src/ggml-cann/ggml-cann.cpp
ggml/src/ggml-cpu/ggml-cpu.c
ggml/src/ggml-cpu/ggml-cpu.cpp
ggml/src/ggml-cpu/kleidiai/kleidiai.cpp
ggml/src/ggml-cpu/ops.cpp
ggml/src/ggml-cpu/ops.h
ggml/src/ggml-cpu/repack.cpp
ggml/src/ggml-cpu/spacemit/ime.cpp
ggml/src/ggml-cuda/ggml-cuda.cu
ggml/src/ggml-hexagon/ggml-hexagon.cpp
ggml/src/ggml-metal/ggml-metal.cpp
ggml/src/ggml-opencl/ggml-opencl.cpp
ggml/src/ggml-openvino/ggml-openvino.cpp
ggml/src/ggml-rpc/ggml-rpc.cpp
ggml/src/ggml-virtgpu/ggml-backend-buffer-type.cpp
ggml/src/ggml-vulkan/ggml-vulkan.cpp
ggml/src/ggml-webgpu/ggml-webgpu.cpp
ggml/src/ggml-zdnn/ggml-zdnn.cpp
ggml/src/ggml.c
```

All hunk-range claims below were produced with
`git diff 81ff7abe5..<side> -- <file> | grep -n '^@@'` and, for the six highest-risk
files, verified by reading the actual hunk content (not just headers). `CONTRACT`
findings that name a symbol were checked against `ggml-sycl.cpp` with
`cat ggml/src/ggml-sycl/ggml-sycl.cpp | grep -n '<symbol>'` (codescout is blind in
that file — see CLAUDE.md) after a positive control
(`ggml_backend_sycl_buffer_get_caps`, confirmed present) proved the grep pipeline
itself works; other files used codescout `search_text` (indexed correctly there).

**Conflict shapes below are verified against an actual merge simulation, not
inferred from hunk-line-range proximity.** `git merge-tree --write-tree master
b10630` was run and its output tree (`135b7a4a7`) inspected directly: the tree
listing's stage-1/2/3 entries are the ground truth for which paths git's real
3-way merge actually leaves conflicted, and `git merge-file -p --diff3` was run
against each conflicted path's three stage blobs to see the exact conflict-marker
regions git would produce. **In this file group, exactly three files conflict:
`ggml/include/ggml-rpc.h`, `ggml/src/ggml-backend-meta.cpp`, and
`ggml/src/ggml.c`** — every other one of the 26 files auto-merges cleanly with no
conflict markers. An earlier pass through this brief used hunk-line-range
proximity as a conflict proxy and got two files wrong as a result
(`ggml-backend-reg.cpp` and `ggml-opencl.cpp`, both flagged as "genuine conflict,
hand-merge required" — corrected below to "auto-merges cleanly" once the actual
merge-tree output showed otherwise). Where a RESOLVE verdict below says a file
auto-merges, that verdict is now a **sanity-check finding** (read the merged
result, confirm it's semantically sound), not a manual-interleave instruction —
git already does the interleave. The three genuine conflicts (and the one
semantic-but-non-conflicting gap found while re-checking `ggml-backend-reg.cpp`'s
actual merged blob) are the only places wave-2 execution needs to write code by
hand.

---

## Cross-cutting findings (read first — several file entries below just cite these)

### F1. The `get_caps` mechanical pattern (fork-wide, ~14 files)

The fork added an optional `get_caps` function pointer to `ggml_backend_buffer_i`
and `ggml_backend_buffer_type_i` (declared in `ggml-backend-impl.h`, a fork-only
header not in this file group since upstream never touches it) plus a
`ggml_backend_buffer_caps` enum and accessor API in `ggml/include/ggml-backend.h`
(see that file's entry). Every other backend in this diff group gets a mechanical
one-line `/* .get_caps = */ NULL,`/`nullptr,` appended to its buffer/buffer-type
struct-literal initializers, sometimes riding along with incidental
clang-format-style reflow of the same literal (`.device =` moved to its own line).
This touches: `ggml-cann.cpp`, `ggml-cpu/kleidiai.cpp`, `ggml-cpu/repack.cpp`,
`ggml-cpu/spacemit/ime.cpp`, `ggml-cuda.cu`, `ggml-hexagon.cpp`, `ggml-metal.cpp`,
`ggml-opencl.cpp`, `ggml-openvino.cpp`, `ggml-rpc.cpp`, `ggml-backend-buffer-type.cpp`
(virtgpu), `ggml-vulkan.cpp`, `ggml-webgpu.cpp`, `ggml-zdnn.cpp`, and
`ggml-backend-meta.cpp`. Default RESOLVE for all of them: **take upstream wholesale,
re-apply the fork's `get_caps` line(s) on top** — except the two files below where
upstream's own hunk lands on the exact same struct-literal line.

`get_caps` is not decorative — SYCL already implements it extensively
(`ggml_backend_sycl_buffer_get_caps`/`_split_buffer_get_caps`/`_tp_buffer_get_caps`
and the `_type_get_caps` counterparts, all wired into their `.get_caps =`
initializers) and one accessor, `ggml_backend_buffer_has_stable_base`, is already a
live SYCL dependency: `ggml-sycl.cpp:33706` — `GGML_ASSERT(ggml_backend_buffer_has_stable_base(buffer) && ...)`.

### F2. The `mmap_support` mechanical pattern (upstream-wide, same file set)

Upstream independently added its own new field, `mmap_support`, to
`ggml_backend_dev_props` (`ggml/include/ggml-backend.h`), touched via each
backend's `device_get_props()`. This lands in the same handful of files as F1 but
at a different struct (`ggml_backend_dev_props`, not `ggml_backend_buffer_i`) and
a different function, so **F1 and F2 do not textually collide with each other**
anywhere except noted below. Confirmed via grep: **`ggml-sycl.cpp` does not set
`.mmap_support`** (`ggml_backend_sycl_device_get_props` has no `mmap_support` hit).
This is not a merge blocker — designated-init C++ zero-fills the omitted field —
but it is a real gap worth a follow-up ticket once wave 2 lands: SYCL device
loading uses `mmap`, so the field should read `true` once the merged tree carries
it (this is unrelated to, but adjacent to, the iGPU VRAM-budget CONTRACT item
under `ggml-cuda.cu` below).

### F3. `GGML_OP_*` enum insertion — auto-merges, EXCEPT one recurring line (highest-value finding in this brief)

Both sides insert new `GGML_OP_*` enum values into the *same* enum
(`ggml/include/ggml.h`) at **different, non-overlapping insertion points**:

- Fork (`master`): `GGML_OP_SET_ROWS_PAGED` right after `GGML_OP_SET_ROWS`
  (early in the enum), and `GGML_OP_ALL_REDUCE_SUM` right before `GGML_OP_COUNT`
  (very end, after `GGML_OP_GLU`). 2 new ops.
- Upstream (`b10630`): `GGML_OP_LIGHTNING_INDEXER`, `GGML_OP_DSV4_HC_COMB`,
  `GGML_OP_DSV4_HC_PRE`, `GGML_OP_DSV4_HC_POST`, all after `GGML_OP_GATED_DELTA_NET`
  (middle of the enum, between the fork's two insertion points). 4 new ops.

Because the insertion points don't overlap, the enum itself and the parallel
`GGML_OP_NAME[]`/`GGML_OP_SYMBOL[]` string tables in `ggml/src/ggml.c` will
**auto-merge cleanly via ordinary 3-way merge** — git applies both hunks to their
own line ranges and the resulting arrays stay correctly positionally aligned with
the enum (verified by reading both sides' full array diffs; each side's
insertion string(s) land at the exact same relative position as its enum
insertion). No hand-reordering needed for the enum/name/symbol arrays themselves.

**What does NOT auto-merge: the `static_assert(GGML_OP_COUNT == N, ...)` line,
which both sides edit identically three times** (`ggml/src/ggml.c` after
`GGML_OP_NAME[]`, `ggml/src/ggml.c` after `GGML_OP_SYMBOL[]`, and
`ggml/include/ggml-rpc.h`'s `RPC_PROTO_PATCH_VERSION`-adjacent assert). Fork
bumps `97 → 99` (97 base + 2 fork ops); upstream bumps `97 → 101` (97 base + 4
upstream ops). This is a **genuine textual 3-way conflict** — same line, two
different literal values, at all three sites. **The correct merged value is
`103`** (97 + 2 fork + 4 upstream), confirmed by manually reading both sides'
final `GGML_OP_NAME`/`GGML_OP_SYMBOL` arrays. Whoever performs the wave-1/wave-2
mechanical merge must hand-fix all three occurrences to `103`, not just resolve
the textual conflict marker with "ours" or "theirs" (both are wrong).

`ggml-rpc.h`'s version constants (`RPC_PROTO_MAJOR/MINOR/PATCH_VERSION`) are a
second, independent conflict on the same three lines — see that file's entry for
the version-bump reconciliation (dominant change: upstream's MAJOR/MINOR bump,
because it reflects a real wire-protocol change, not just an op-count bump).

Secondary, lower-confidence risk from the same insertion: any dispatch code using
a *range* check on `GGML_OP_*` (e.g. `op >= X && op <= Y`) rather than an
enumerated switch could silently change behavior once new values fall inside a
previously-exclusive range. No such range-based dispatch was found in
`ggml-sycl.cpp` during this review, but this brief did not exhaustively search
for the pattern — flag it for whoever executes T15/T22.

### F4. `struct ggml_tensor` ABI extension — fork-only, unopposed by upstream in this window

The fork added two fields to the public `struct ggml_tensor` (`ggml/include/ggml.h`):
`size_t buffer_offs` (right after `view_offs`) and
`struct ggml_tensor_layout * layout` (right after `extra`), shrinking
`char padding[8]` to `char padding[GGML_MEM_ALIGN + sizeof(size_t)]`, plus a new
`GGML_TENSOR_STRUCT_VERSION` (=2) ABI-guard macro guarded by an
`#if defined(GGML_TENSOR_STRUCT_VERSION_EXPECTED)` opt-in static_assert.
**Upstream makes zero changes anywhere near the tensor struct definition in this
merge window** (no `b10630` hunk falls in that region of `ggml.h`) — confirmed by
hunk-range comparison. So this survives the merge completely untouched, cleanly.

This is worth flagging anyway because of the blast radius if that ever changes:
`tensor->layout` / `GGML_LAYOUT_*` appears **1286 times** in `ggml-sycl.cpp` (it
is the backbone of the fork's "LAYOUT FOLLOWS RESIDENCY" architecture per
CLAUDE.md), and `tensor->buffer_offs` is used directly in view-offset arithmetic
(`ggml-sycl.cpp:36317-36318`). `GGML_TENSOR_STRUCT_VERSION_EXPECTED` itself is
**not wired into any build target today** — `search_text` finds it only in
`ggml.h` and two doc files (`docs/CHANGELOG.md`, `docs/development/ABI.md`), no
`.cpp`/`CMakeLists.txt` defines it before including `ggml.h` — so it is
presently a documented opt-in developer safety net, not an automatic gate. Worth
a follow-up ticket to wire it into the SYCL TU, but out of scope for this merge.

### F5. `ggml_ssm_scan()` signature change — genuine cross-brief CONTRACT item

Upstream adds a new trailing parameter to the public API
`ggml_ssm_scan(ctx, s, x, dt, A, B, C, ids, int64_t K)` (was `..., ids)`), for
multi-step SSM state snapshots. This is a real public signature change:

- Every caller of `ggml_ssm_scan()` (Mamba/SSM graph-building, expected in
  `src/llama-graph.cpp` or a `src/models/*.cpp`) must add the new `K` argument —
  **this is Task 11's (`llama-common.md`) scope to trace, flagged here so it
  isn't missed**.
- `ggml-cpu`'s `supports_op` gates on the new param
  (`ggml_get_op_params_i32(op, 0) == 1 || op->src[3]->ne[0] == 1`, per the
  CPU-backend analysis below). **`ggml-sycl.cpp`'s `GGML_OP_SSM_SCAN` `supports_op`
  (line ~99282) only checks `op->src[3]->ne[0] == 1` — it does not inspect the new
  `K` op-param.** Not a compile break (SYCL's dispatch case still compiles fine
  against the new signature), but a semantic gap: once any model emits an
  `ssm_scan` graph with `K > 1`, SYCL's `supports_op` may accept a graph its
  kernel doesn't correctly execute, silently mis-computing rather than falling
  back to CPU. Flag for whoever ports this upstream feature at wave 2/3 — the fix
  is a `supports_op` update in `ggml-sycl.cpp`, outside this brief's no-source-edits
  scope.

### F6. `ggml_backend_sched` internals — fixed arrays become growable heap arrays, entirely internal

Upstream reworks `ggml_backend_sched_split::inputs` and
`ggml_backend_sched::graph_inputs` from fixed-size arrays
(`struct ggml_tensor * inputs[GGML_SCHED_MAX_SPLIT_INPUTS]`) to growable
heap-allocated arrays (`struct ggml_tensor ** inputs` + `inputs_capacity`,
`realloc`-grown on overflow instead of asserting). Both structs are defined
inside `ggml-backend.cpp` itself, **not** in the public `ggml-backend.h` header.
`search_text` confirms `GGML_SCHED_MAX_SPLIT_INPUTS` and both struct names are
referenced only within `ggml-backend.cpp` — zero hits anywhere in
`ggml-sycl.cpp` or any other backend file. **CONTRACT: none** — purely internal
to the scheduler TU, no cross-file ABI surface.

---

## File entries

### `ggml/include/ggml-backend.h`

**Fork intent (ours):** 5 hunks (old lines 41–47, 52–65, 64–79, 230–243,
251–271). Adds the `get_caps` API (F1): `ggml_backend_buft_get_caps`,
`ggml_backend_buffer_caps` enum, `ggml_backend_buffer_get_caps`,
`ggml_backend_buft_has_cap`/`buffer_has_cap`/`has_stable_base`/`is_relocatable`,
plus `ggml_backend_tensor_has_storage`/`get_buffer_offset`, and
`ggml_backend_buffer_is_valid`. Also adds
`ggml_backend_disable_device_backends()` / `ggml_backend_device_backends_disabled()`
(process-wide CPU-only override, consumed in `src/llama-context.cpp`,
`llama-kv-cache.cpp`, `llama-model-loader.cpp`, `llama-model.cpp` — not by
`ggml-sycl.cpp`). Also replaces the old bare `ggml_backend_unload()` with a new
`ggml_backend_unload_result` enum + `ggml_backend_unload_checked()`, keeping
`ggml_backend_unload()` as an unchanged-signature compatibility wrapper (no
break for existing callers).

**Upstream intent (theirs):** 1 hunk (old line 154), purely additive:
`bool mmap_support;` on `ggml_backend_dev_props` (F2).

**RESOLVE:** Clean union merge — no overlap (fork's hunks: 41–271; upstream's
single hunk: 154, sits between fork's second and third hunks with no shared
lines).

**CONTRACT:** Yes, but favorable. `ggml_backend_buffer_has_stable_base` is a
live SYCL consumer (`ggml-sycl.cpp:33706`) — this is the fork's own API,
unaffected by upstream. `mmap_support` — confirmed via grep, **no**
`ggml-sycl.cpp` consumer yet (F2; not a merge blocker, follow-up ticket
candidate). No upstream change touches any fork-added declaration.

---

### `ggml/include/ggml-rpc.h`

**Fork intent (ours):** 1 hunk, 2 lines on the same block: `RPC_PROTO_PATCH_VERSION`
`1 → 2`, and `static_assert(GGML_OP_COUNT == 97, ...)` → `== 99` (F3).

**Upstream intent (theirs):** Same block: `RPC_PROTO_MAJOR_VERSION 4→5`,
`RPC_PROTO_MINOR_VERSION 0→1`, `RPC_PROTO_PATCH_VERSION 1→0`, and the assert
`== 97` → `== 101` (F3) — reflecting upstream's 4 new ops plus real wire-protocol
changes in `ggml-rpc.cpp` (`rpc_tensor.use_count`, new `RPC_CMD_MEMSET_TENSOR`).

**RESOLVE:** **Direct textual conflict, not a clean layer** — both sides edit
the identical `#define` lines and the identical `static_assert`. Hand-merge
required: `GGML_OP_COUNT` → `103` (per F3); version constants → take upstream's
`MAJOR 5 / MINOR 1` as the dominant bump (it encodes upstream's real protocol
change — `use_count`/`memset_tensor`), and pick a fresh `PATCH` value (e.g. `0`)
rather than just concatenating both patch bumps, since the fork's `PATCH 2` was
counting only its own op-count change against the *old* major/minor. Resolve
together with `ggml-rpc.cpp` below — same commit, same reviewer.

**CONTRACT:** Yes — this header is the RPC wire-protocol contract itself. Get
the `GGML_OP_COUNT`/version reconciliation wrong and either the `static_assert`
fails to compile, or (worse, if hand-patched past the assert without checking
semantics) a SYCL-linked `rpc-server` silently reports a version number that
doesn't uniquely identify the real wire format, breaking interop with any
`rpc-client` built from a differently-merged tree.

---

### `ggml/include/ggml.h`

**Fork intent (ours):** 9 hunks (old lines 219, 525, 586, 665, 690, 862, 1688,
2429, 2687). `GGML_TENSOR_STRUCT_VERSION` ABI guard (F4); `GGML_OP_SET_ROWS_PAGED`
/`GGML_OP_ALL_REDUCE_SUM` enum inserts (F3); the `ggml_tensor_tp_info` /
`ggml_layout_mode` / `ggml_tensor_layout` structs and the `layout`/`buffer_offs`
fields on `struct ggml_tensor` (F4); `ggml_tensor_get_layout_ptr`;
`ggml_set_rows_paged()`, `ggml_flash_attn_ext_set_paged_layout()`,
`ggml_all_reduce_sum()` builder declarations.

**Upstream intent (theirs):** 7 hunks (old lines 570, 779, 1716, 1973, 2451,
2575, 2723). 4 new `GGML_OP_*` enum values after `GATED_DELTA_NET` (F3);
`ggml_is_contiguous_to_{1,2,3}`; `ggml_clamp`/`ggml_clamp_inplace` moved earlier
+ new `ggml_rope_set_offset(a, n_offs)`; **`ggml_ssm_scan()` gains a trailing
`int64_t K` parameter** (F5); new `ggml_lightning_indexer`/`ggml_dsv4_hc_comb`/
`_pre`/`_post` builder declarations; new `ggml_build_forward_order()`.

**RESOLVE:** Union merge, no line-range overlap anywhere in this header between
the two sides (verified: fork's 9 ranges and upstream's 7 ranges are disjoint).
The one non-mechanical part is the `GGML_OP_COUNT`-adjacent bookkeeping, which
lives in `ggml.c`/`ggml-rpc.h`, not this header — see F3.

**CONTRACT:** Two items, both already covered above: F3 (enum insertion —
auto-merges here, conflict is in the arrays/assert) and F5 (`ggml_ssm_scan`
signature change — real API break, Task 11 must trace call sites). `struct
ggml_tensor`'s extension (F4) is untouched by upstream — no conflict, but it is
the single highest-blast-radius struct in the codebase for this fork and
deserves a mention in any wave-2 build-failure triage: if `ggml-sycl.cpp` fails
to compile against the merged `ggml.h` with a tensor-initializer-list error, the
`layout`/`buffer_offs` fields are the first thing to check for order drift.

---

### `ggml/src/ggml-alloc.c`

**Fork intent (ours):** 2 hunks (old lines 5, 1246). Adds `#include <math.h>`,
and a new 74-line public function `ggml_backend_probe_max_alloc_size(buft,
upper_bound, safety_margin)` — binary-search VRAM-allocation probing, declared
in `ggml/include/ggml-alloc.h` (not in this file group — untouched by upstream).

**Upstream intent (theirs):** 1 hunk (old line 40): adds `case GGML_OP_CLAMP:`
to `ggml_op_can_inplace()`'s allow-list (supports the `ggml_clamp` relocation in
`ggml.h`/`ggml.c`, see above).

**RESOLVE:** Clean union merge, zero overlap (old line 40 vs old lines 5 and
1246 — fork's second hunk is 1200+ lines away).

**CONTRACT:** `ggml_backend_probe_max_alloc_size` is fork-only and untouched by
upstream, but it is heavily load-bearing: `search_text` shows it is actively
called from `ggml-sycl.cpp`'s init path (per `docs/plans/data/e1-rca/findings.md`,
14 binary-search allocations per call, with an SYCL-specific env-gate
`GGML_SYCL_E1_RCA_DISABLE_ALLOC_PROBE` for RCA) and is exercised by
`tests/test-alloc.cpp`. No upstream interference — flagging only so wave-2
build/test triage doesn't mistake this function for something upstream touched.

---

### `ggml/src/ggml-backend-meta.cpp`

**Fork intent (ours):** 3 hunks (old lines 335, 1481, 1489) — the `get_caps`
pattern (F1) on `ggml_backend_meta_buffer_type_iface` and
`ggml_backend_meta_buffer_iface`, plus a whitespace-only comment-alignment
tweak on the `.memset_tensor` line.

**Upstream intent (theirs):** 16 hunks (old lines 132, 140, 590, 600, 745, 790,
800, 920, 984, 1063, 1248, 1345, 1481, 1834, 1934), the largest upstream diff in
this file (287 lines) — evolves the pre-existing "meta backend" (shape/split
tracking without allocation): `mmap_support` (F2, at 132/140);
`ggml_backend_meta_get_split_state` internals grow substantially (590–1119);
`ggml_backend_meta_buffer_init_tensor`/`set_tensor` get new logic (1248–1504);
**implements `ggml_backend_meta_buffer_memset_tensor`** and wires it into
`.memset_tensor` (old line 1481–1487); `graph_compute` changes (1834–2107).

**RESOLVE:** **One genuine textual conflict** — both sides edit the identical
line `/* .memset_tensor = */ nullptr, // TODO implement` inside
`ggml_backend_meta_buffer_iface` (fork: whitespace-only; upstream: real
implementation). Take upstream's implementation (it fully supersedes the
fork's cosmetic edit — verified by reading both diffs at old line 1481). The
fork's adjacent `get_caps` hunk (old line 1489, two lines later) does not
overlap upstream's hunk (which ends at old line ~1487) and applies cleanly on
top. All other upstream hunks (132–1119, 1834–2107) have zero fork-side
overlap.

**CONTRACT:** None — `ggml-backend-meta.cpp`'s "meta" device is a
shape-tracking/testing backend with no SYCL interaction; none of its symbols
are referenced from `ggml-sycl.cpp`.

---

### `ggml/src/ggml-backend-reg.cpp`

**Fork intent (ours):** 8 hunks (old lines 3, 88, 103–295 [a 192→876-line
rewrite — by far the largest hunk in this file], 296, 307, 384, 507, 563). This
is a substantial architectural rewrite of the backend registry: a
`ggml_backend_reg_state` lifecycle state machine (`ACTIVE`/`REACTIVATING`/
`UNLOADING`/`HIDDEN_FAILED`/`REMOVED`), deferred/reentrant registration
handling, `ggml_backend_disable_device_backends()`/`device_backends_disabled()`
wired into `register_builtin_backends()` (every `#ifdef GGML_USE_*` block now
guarded by `if (!disable_device_backends) { register_backend(...); }`), checked
unload with test hooks (`ggml_backend_test_block_next_unload` etc., for
deterministic test barriers), and `ggml_backend_registry_cached_name`/
`begin_call`/`end_call` for safe concurrent access during unload.

**Upstream intent (theirs):** 2 tiny hunks (old lines 86, 161–167) — adds a new
`GGML_USE_ET` backend ("ET") `#include` guard and a
`register_backend(ggml_backend_et_reg())` call inserted between the
`GGML_USE_OPENVINO` and `GGML_USE_CPU` blocks in the registration chain.

**RESOLVE:** **Corrected — auto-merges cleanly, but the auto-merged result has a
real semantic gap worth a deliberate fix, not just a sanity check.** An earlier
pass through this brief called this a "genuine overlap, hand-merge required" on
the reasoning that fork's giant hunk 3 (old lines 103–295, rewriting
`register_builtin_backends()`) range-contains upstream's insertion point (old
lines 161–167, the `#ifdef GGML_USE_OPENVINO ... #ifdef GGML_USE_CPU` region).
**`git merge-tree --write-tree master b10630` shows this file does NOT appear in
the conflict set** — git's real 3-way merge auto-resolves it with no conflict
markers. Reading the actual merged blob (`git show
135b7a4a7:ggml/src/ggml-backend-reg.cpp`) confirms why and surfaces the gap:

```c
#ifdef GGML_USE_OPENVINO
        if (!disable_device_backends) {
            register_backend(ggml_backend_openvino_reg());
        }
#endif
#ifdef GGML_USE_ET
        register_backend(ggml_backend_et_reg());
#endif
#ifdef GGML_USE_CPU
        register_backend(ggml_backend_cpu_reg());
#endif
```

Git's line-based merge inserted upstream's 3-line `#ifdef GGML_USE_ET` block
verbatim between the fork's already-guarded `OPENVINO` and `CPU` blocks — which
is textually clean (no overlapping lines) but **semantically inconsistent**:
every other device backend in this function is wrapped in
`if (!disable_device_backends) { ... }` (the fork's CPU-only override, F3-adjacent
but really a distinct fork feature — see `ggml-backend.h`'s entry), and `ET`
silently isn't. Once merged as-is, `--device none`/`GGML_BACKEND_CPU_ONLY` would
fail to suppress the `ET` backend the way it suppresses every other device
backend. **Action for wave-2 execution: after taking the auto-merge, wrap the
`register_backend(ggml_backend_et_reg());` line in the same
`if (!disable_device_backends) { ... }` guard as its siblings** — a 3-line fix,
not a hand-merge. The top-of-file `#ifdef GGML_USE_ET / #include "ggml-et.h" /
#endif` guard (old line 3 region) auto-merges with no such issue (it's a plain
`#include` guard, nothing to wrap). All other fork hunks (296–563:
`ggml_backend_device_register` tail additions, `striequals`,
`ggml_backend_init_best`, `ggml_backend_load_best`,
`ggml_backend_load_all_from_path`) have zero upstream overlap and auto-merge
cleanly.

**CONTRACT:** None beyond F1/F2/F3, already covered. The registry rewrite adds
no new public header declarations beyond what `ggml-backend.h` already lists
(`ggml_backend_disable_device_backends`/`device_backends_disabled`,
`ggml_backend_unload_checked`) — both fork-only and confirmed via grep to have
**zero references in `ggml-sycl.cpp`** (they're consumed in `src/llama-*.cpp`
instead). The public enumeration API SYCL does use
(`ggml_backend_reg_count`/`_get`, `ggml_backend_dev_count`/`_get`,
`ggml_backend_load`, `ggml_backend_device_register`) keeps its exact signature
through this rewrite.

---

### `ggml/src/ggml-backend.cpp`

**Fork intent (ours):** ~20 hunks spanning old lines 20–2367, concentrated in
two regions: (1) old lines 20–724 — a large addition (233 lines at the top) of
device-lifecycle/checked-unload-safety machinery for buffers and events
(`buffer_production_guard`, `device_call_guard`, `live_owner_store`, atomic
test-hook counters) supporting `ggml_backend_unload_checked()`'s "safe to
unload while buffers/events are still outstanding" semantics, plus smaller
hunks through `ggml_backend_buft_*`/`ggml_backend_buffer_*`/tensor-copy-async
functions; (2) old lines 2267–2367 — corresponding `get_caps`-pattern (F1) and
struct-literal touches on `ggml_backend_cpu_buffer_i`/`_from_ptr_i`/buffer-type
functions.

**Upstream intent (theirs):** ~14 hunks, entirely within old lines 765–1873 —
the scheduler (`ggml_backend_sched_split`/`ggml_backend_sched` structs and
`ggml_backend_sched_split_graph`/`_backend_id_from_cur`/`_compute_splits`/
`_new`/`_free`). Key changes: `split->inputs`/`sched->graph_inputs` become
growable heap arrays instead of fixed `GGML_SCHED_MAX_SPLIT_INPUTS`-sized arrays
(F6); `ggml_backend_sched_backend_id_from_cur` gains a `GGML_OP_FLASH_ATTN_EXT`
skip (alongside the existing `GGML_OP_ROPE` skip) when picking a backend based
on a source tensor's owning backend, "since the sinks tensor is too small to
choose a backend based on it."

**RESOLVE:** **Zero textual overlap** — fork's touched ranges (20–724,
2267–2367) and upstream's touched range (765–1873) are disjoint (verified: the
gap between fork's hunk at old line 724 and its next hunk at old line 2267 is
exactly where upstream's entire diff lives). Clean union merge; no hand
reconciliation needed for this file.

**CONTRACT:** None beyond F1 (already covered) and F6 (scheduler internals,
confirmed zero references outside this TU). The
`ggml_backend_sched_backend_id_from_cur` FLASH_ATTN_EXT-sinks skip is worth a
one-line mention for whoever validates GPT-OSS correctness post-merge (GPT-OSS
uses attention sinks per CLAUDE.md's flash-attn gates) — it changes scheduler
backend-selection heuristics, not SYCL's own kernel behavior, so it should only
ever change *which* backend a sinks tensor's consumer op is scheduled on, not
correctness of the SYCL kernel itself.

---

### `ggml/src/ggml-cann/ggml-cann.cpp`

**Fork intent (ours):** 3 one-line `get_caps` (F1) additions (old lines 1478,
1602, 1750).

**Upstream intent (theirs):** 2 hunks: `ggml_backend_cann_supports_op`'s ROPE
case (~2534) now rejects `op_params[15] != 0` (CANN doesn't yet support the new
`ggml_rope_set_offset` feature); `device_get_props` (~2818) adds `mmap_support`
(F2).

**RESOLVE:** Take upstream wholesale, re-apply the fork's 3 `get_caps` lines on
top — no proximity (1478/1602/1750 vs 2534/2818).

**Mechanical-rename check:** Yes — pure F1 pattern, no CANN-specific logic in
the fork's touch.

**CONTRACT:** None — CANN (Ascend NPU) is backend-siloed; no shared symbols
with `ggml-sycl.cpp`.

---

### `ggml/src/ggml-cpu/ggml-cpu.c`

**Fork intent (ours):** 12 lines, additive: wires `GGML_OP_SET_ROWS_PAGED` →
`ggml_compute_forward_set_rows_paged` and `GGML_OP_ALL_REDUCE_SUM` →
`ggml_compute_forward_all_reduce_sum` into both `ggml_compute_forward()`'s
op-dispatch switch and `ggml_get_n_tasks()`'s single-thread-preferred group
(inserted right after the existing `SET_ROWS` cases).

**Upstream intent (theirs):** 48 ins / 2 del. Adds dispatch for
`GGML_OP_LIGHTNING_INDEXER`/`GGML_OP_DSV4_HC_{COMB,PRE,POST}` in the same two
switch statements (inserted near `GATED_DELTA_NET`/`SSM_SCAN`, not overlapping
the fork's insertion points); `__gnu_linux__`→`__linux__` guard rename for
thread-affinity code; a `__wasi__` single-thread guard in `ggml_graph_plan`;
extends `GGML_OP_OUT_PROD` work-buffer sizing for F16 src0; a new
`GGML_OP_SET_ROWS` work-buffer case (F16→non-F16 conversion buffer); a
work-buffer case for `LIGHTNING_INDEXER`; a new `ggml_cpu_has_sme2()` feature
probe.

**RESOLVE:** Interleave — both sides add cases to the same two switch
statements at non-overlapping case labels. Take upstream wholesale for its new
ops, threading fixes, and buffer-sizing hunks, then re-insert the fork's two
`SET_ROWS_PAGED`/`ALL_REDUCE_SUM` case labels (additive, non-conflicting).

**CONTRACT:** `GGML_OP_SET_ROWS_PAGED`/`GGML_OP_ALL_REDUCE_SUM` — confirmed live
SYCL consumers: `ggml-sycl.cpp` has `case GGML_OP_SET_ROWS_PAGED:` at 5 sites
(65191, 76407, 81266, 98963, 99645) and `case GGML_OP_ALL_REDUCE_SUM:` at 2 sites
(76765, 99304) — this CPU dispatch is the fallback counterpart of a full SYCL
feature (paged KV write + tensor-parallel all-reduce); **must not be dropped**.
`LIGHTNING_INDEXER`/`DSV4_HC_*` — zero hits in `ggml-sycl.cpp`; SYCL has no
implementation, so these fall to CPU (expected gap, not a bug). Also see F5
(`ssm_scan` K param — `ggml-cpu`'s `supports_op` checks it, SYCL's doesn't).

**Mechanical-rename check:** No — fork's touch is substantive new-op wiring,
not a rename/formatting pass.

---

### `ggml/src/ggml-cpu/ggml-cpu.cpp`

**Fork intent (ours):** Trivial — removes one blank line in
`ggml_backend_cpu_get_proc_address` before the threadpool comment block.

**Upstream intent (theirs):** Adds `.mmap_support = true` to
`ggml_backend_cpu_device_get_props` (F2); extends
`ggml_backend_cpu_device_supports_op` (F16 src1 for `IM2COL_BACK`, F16 src0 for
`OUT_PROD`, new `GGML_OP_CONV_2D` and `GGML_OP_SSM_SCAN` cases — the latter
gating on the new `K` param, F5); registers `ggml_cpu_has_sme2()`.

**RESOLVE:** Take upstream wholesale; fork's one-line blank-line removal (old
line 663) is far from upstream's last hunk (ends ~609) — trivial reapply, no
real interleave needed.

**CONTRACT:** `ggml_backend_cpu_device_supports_op` mirrors scheduler
fallback logic but isn't called by `ggml-sycl.cpp` directly. `ggml-sycl.cpp`
does directly use `ggml_backend_cpu_buffer_type()->iface.{get_alignment,
get_alloc_size,is_host}` (lines 38545/38547/38548) to build its own
host-buffer-type wrapper — none of those field names are touched by this diff,
so no break. `mmap_support` — no `ggml-sycl.cpp` consumer (F2).

**Mechanical-rename check:** Fork's change is pure whitespace, not a rename.

---

### `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`

**Fork intent (ours):** 1 hunk (old line 1504) — `get_caps` (F1) +
whitespace-alignment on `ggml_backend_cpu_buffer_type_kleidiai`'s struct
literal.

**Upstream intent (theirs):** 646 lines (~24 hunks, old lines 2–1503) — a
substantial KleidiAI (Arm) rewrite: kernel-chain collection, cpu-feature
detection, expanded `tensor_traits`, a new warn-once diagnostic for
non-Q4_0/Q8_0 quantized tensors it can't accelerate, buffer-type/alloc-size
changes.

**RESOLVE:** Take upstream wholesale; fork's single tail hunk sits strictly
after upstream's last hunk (old line 1504 vs upstream's last touch ending
~1489) — no overlap, reapply the `get_caps` line to the (now-shifted ~+382
lines) struct initializer at the end.

**CONTRACT:** `get_caps` — SYCL already fully implements this field elsewhere
(F1); no break. This is Arm KleidiAI-specific otherwise; no shared symbols with
`ggml-sycl.cpp`.

**Mechanical-rename check:** Yes — same fork-wide `get_caps` + whitespace
pass as `repack.cpp`/`spacemit/ime.cpp` below. Upstream's rewrite needs this
same `get_caps = nullptr` line re-applied to its relocated buffer-type
initializer, since upstream never added the field.

---

### `ggml/src/ggml-cpu/ops.cpp`

**Fork intent (ours):** 121 lines, three changes: (1) `GGML_UNUSED(src1)` fix in
`ggml_compute_forward_get_rows_q`; (2) full implementations of
`ggml_compute_forward_set_rows_paged` (templated F32/F16 × I32/I64 paged-KV
scatter-write, explicitly commented as "Single-device CPU port of the SYCL
block-addressed KV write, see `ggml/src/ggml-sycl/set_rows_paged.hpp`") and
`ggml_compute_forward_all_reduce_sum` (CPU identity passthrough — no peer
shards to reduce on CPU); (3) a correctness fix in
`ggml_compute_forward_flash_attn_ext_f16_one_chunk`: now checks
`if (q->type == k_vec_dot_type)` and does a raw `memcpy`, only falling back to
`q_to_vec_dot` when `q->type == GGML_TYPE_F32` — the comment explicitly cites
`GGML_SYCL_FATTN_Q_TYPE` casting Q to F16 at SYCL graph-build time, fixing a bug
that manifests when such a graph executes on CPU (mixed-backend/CPU-fallback).

**Upstream intent (theirs):** 525 lines. `ggml_compute_forward_set_rows_impl`/
`set_rows` rework (relaxes the F32/F16 dtype assert, adds a float-conversion
path); rope `n_offs` partial-rotation support; im2col/conv_2d_dw cleanup;
`ggml_fp16_to_fp32_row`→`ggml_cpu_fp16_to_fp32` rename inside
`flash_attn_ext_tiled` (a **different** function from the fork's
`flash_attn_ext_f16_one_chunk` fix — no collision); `ssm_scan` `K` op-param
addition (F5); ~290-line new `dsv4_hc_{comb,pre,post}` implementation; ~87-line
new `ggml_compute_forward_lightning_indexer`.

**RESOLVE:** Interleave, no textual overlap: take upstream's `set_rows`
rework, rope `n_offs`, `ssm_scan` K, and the two new-op blocks wholesale; then
re-insert the fork's `set_rows_paged`/`all_reduce_sum` block immediately after
the (upstream-modified) `ggml_compute_forward_set_rows`, re-apply the
`GGML_UNUSED(src1)` one-liner, and re-apply the Q-type fix in
`flash_attn_ext_f16_one_chunk` (untouched by upstream's rename in the sibling
`flash_attn_ext_tiled`).

**CONTRACT:** Multiple, all confirmed. (1) `set_rows_paged`/`all_reduce_sum` —
same live SYCL consumers as `ggml-cpu.c` above. (2) The Q-type fix's
`GGML_SYCL_FATTN_Q_TYPE` macro is confirmed defined in
`ggml/include/ggml-sycl.h` and consumed in `src/llama-graph.cpp`
(`sycl_q_type = (enum ggml_type) GGML_SYCL_FATTN_Q_TYPE`) to build flash-attn
graphs with Q cast to F16 for SYCL — this is a real CPU-fallback bug-fix for
graphs SYCL constructed and **must be preserved**. (3) `ssm_scan` K (F5) —
SYCL's `supports_op` doesn't check it (flagged in F5). (4) Upstream's new rope
`n_offs` (`op_params[15]`) — `ggml-sycl.cpp`'s ROPE dispatch/`supports_op` never
references `n_offs`/`op_params[15]`; if a model sets it non-zero, SYCL's rope
kernel would silently ignore it (same "not yet a bug because nothing sets it"
caveat as `mmap_support`, but worth tracking once `ggml_rope_set_offset` gets a
caller).

**Mechanical-rename check:** No — both sides' touches are substantive new
logic (aside from the incidental, non-conflicting `ggml_fp16_to_fp32_row`
rename inside a function the fork doesn't touch).

---

### `ggml/src/ggml-cpu/ops.h`

**Fork intent (ours):** 2-line addition — forward declarations for
`ggml_compute_forward_set_rows_paged`/`all_reduce_sum`, after
`ggml_compute_forward_set_rows` (~line 56).

**Upstream intent (theirs):** 4-line addition — forward declarations for
`ggml_compute_forward_lightning_indexer`/`dsv4_hc_{comb,pre,post}`, after
`ggml_compute_forward_gated_delta_net` (~line 105).

**RESOLVE:** Trivial interleave — both insertion points are ~50 lines apart,
take both.

**CONTRACT:** None — this header is CPU-backend-internal (only included by
`ggml-cpu/*.cpp`), not by `ggml-sycl.cpp`. The underlying enum contract point is
already covered under `ggml-cpu.c`.

**Mechanical-rename check:** No — additive declarations only.

---

### `ggml/src/ggml-cpu/repack.cpp`

**Fork intent (ours):** 1-line `get_caps` (F1) addition to
`ggml_backend_cpu_repack_buffer_type`'s initializer.

**Upstream intent (theirs):** 1-line signature widen — `make_block_q4_0x4`'s
`unsigned int blck_size_interleave` param → `int` (file-local `static` helper,
no external callers possible).

**RESOLVE:** Trivially take both — different, non-adjacent locations, clean
union merge.

**CONTRACT:** None — `make_block_q4_0x4` is `static`/file-local; `get_caps`
already implemented for SYCL elsewhere (F1).

**Mechanical-rename check:** Yes for the fork's hunk — same `get_caps`
field-completeness pass as `kleidiai.cpp`/`spacemit/ime.cpp`; nothing to
re-apply from upstream's unrelated param-type hunk.

---

### `ggml/src/ggml-cpu/spacemit/ime.cpp`

**Fork intent (ours):** 8 lines across two spots — `get_caps` (F1) on
`ggml_backend_riscv64_spacemit_buffer_i` and on
`ggml_backend_cpu_riscv64_spacemit_buffer_type` (plus re-indentation of that
initializer's existing fields).

**Upstream intent (theirs):** 2-line addition — a `case GGML_TYPE_Q5_0:` added
alongside `Q4_K/Q6_K/Q8_0/Q5_1/Q5_K` in the `tensor_traits` repack-eligible
quant-type list (RISC-V IME backend).

**RESOLVE:** Trivially take both — fork's hunks are at the buffer/buffer-type
struct initializers near file end, upstream's are inside the `tensor_traits`
template switch near the top; no overlap.

**CONTRACT:** None — `get_caps` already implemented for SYCL elsewhere (F1);
`Q5_0` repack support is local to the SpaceMiT/RISC-V backend, unrelated to any
shared enum/interface SYCL touches.

**Mechanical-rename check:** Yes for the fork's hunks — same `get_caps` +
whitespace-realignment pass as `kleidiai.cpp`/`repack.cpp`.

---

### `ggml/src/ggml-cuda/ggml-cuda.cu`

**Fork intent (ours):** 3 hunks — `get_caps` (F1) on
`ggml_backend_cuda_buffer_interface`, `ggml_backend_cuda_buffer_type_interface`,
and the inline host-buffer-type literal (with incidental multi-line reflow).

**Upstream intent (theirs):** 433 ins / 80 del across ~50 hunks — device
init/parsing, pool/vmm struct growth, `mul_mat_cublas`/`mul_mat_id` dispatch
refactors, graph-fusion additions (RoPE/set_rows fusion, GDN cache fusion,
topk-moe fusion), graph capture/evaluate changes, device-count/UMA-memory
helpers, **`device_get_props` sets `mmap_support = props->type !=
GGML_BACKEND_DEVICE_TYPE_IGPU`** (gates mmap OFF specifically for the iGPU
device type — see CONTRACT below), `device_supports_op` extensions (new
`GGML_TYPE_Q2_0` support, other type/shape gates).

**RESOLVE:** Take upstream wholesale, re-apply the fork's 3 `get_caps` hunks on
top — no proximity to any upstream hunk.

**CONTRACT:** None as a compile/link dependency (CUDA-specific: pool/vmm,
graph-fusion heuristics, Q2_0 support). Worth flagging for awareness, not as a
merge blocker: CUDA's `mmap_support = (type != IGPU)` pattern is exactly the
iGPU-vs-discrete distinction CLAUDE.md documents as the root cause of this
fork's VRAM-budget iGPU hazard (the Arrow Lake-S iGPU reporting 231.7 GB of
"VRAM"; `ggml-sycl.cpp`/`unified-cache.cpp` currently consult neither
`host_unified_memory` nor an `is_integrated` check). This is a useful upstream
reference pattern for the analogous SYCL fix, not a code dependency of this
merge.

**Mechanical-rename check:** Yes — identical `get_caps` pattern; the
comment-spacing/`.device =` reflow riding along is clang-format noise, not new
logic.

---

### `ggml/src/ggml-hexagon/ggml-hexagon.cpp`

**Fork intent (ours):** 3 one-line `get_caps` (F1) additions (old lines 981,
1054, 1064).

**Upstream intent (theirs):** 588 lines across ~90 hunks — htp-event/op-profiling
additions; tiled-repack padding-alignment fixes across Q4_0/Q4_1/Q8_0/MXFP4;
a large `ggml_hexagon_opqueue`/`opbatch` restructuring; flash-attention
HMX-eligibility changes; a matmul-params precompute rewrite/relocation;
dozens of small `supports_op` predicate extensions; op-remap table and
node-fusion additions; `device_get_props`/`device_supports_op`/`proc_address`
growth.

**RESOLVE:** Take upstream wholesale, re-apply the fork's 3 `get_caps` lines on
top — the fork's hunks sit in gaps between upstream's dense hunk sequence
(981 between upstream's 955–965 and 1015–1023; 1054/1064 between upstream's
1029–1048 and 1088–1093).

**CONTRACT:** None — Qualcomm Hexagon NPU/HVX/HMX dispatch is fully
backend-siloed; no shared symbols with `ggml-sycl.cpp`.

**Mechanical-rename check:** Yes — pure `get_caps` pattern, no Hexagon-specific
logic in the fork's touch.

---

### `ggml/src/ggml-metal/ggml-metal.cpp`

**Fork intent (ours):** 5 hunks — `get_caps` (F1) on
`ggml_backend_metal_buffer_shared_i`/`_private_i` and reflow-and-add on three
inline buffer-type literals (shared/private/mapped).

**Upstream intent (theirs):** 56 lines, semantically dense: new
`ggml-metal-tuning.h` include; an OOM guard in
`ggml_backend_metal_buffer_type_alloc_buffer` (returns NULL with a logged error
instead of dereferencing a NULL `ggml_metal_buffer_init` result);
`get_alloc_size` for `GGML_OP_FLASH_ATTN_EXT` adds extra-KV-F16 accounting;
`device_get_props` adds `mmap_support` (F2); a new block of FA-vector tuning
proc-address exports; **`ggml_metal_device_get(device)` →
`ggml_metal_device_get(device, g_devices)` — a signature change**, confirmed by
reading the diff (`ggml_backend_metal_device_init`, old line ~942).

**RESOLVE:** Take upstream wholesale, re-apply the fork's 5 `get_caps`/reflow
hunks on top — no overlap (fork's struct-literal touches at 98/174/300/375/453
vs upstream's logic hunks at 203–231/681–688/868–942).

**CONTRACT:** None for `ggml-sycl.cpp` — `ggml_metal_device_get`'s signature
change is purely intra-Metal (any other caller elsewhere in
`ggml-metal-device.*` needs the extra `g_devices` argument, but that file isn't
in this diff group and isn't SYCL-adjacent). Flagging only so whoever lands the
Metal side of the merge knows to grep for other call sites of
`ggml_metal_device_get`.

**Mechanical-rename check:** Yes for the fork's diff — pure `get_caps` +
incidental struct-literal reflow, no Metal logic added by the fork here.

---

### `ggml/src/ggml-opencl/ggml-opencl.cpp`

**Fork intent (ours):** 2 lines — `get_caps` (F1) on
`ggml_backend_opencl_buffer_interface` (old line 9707) and
`ggml_backend_opencl_buffer_type_interface` (old line 9770).

**Upstream intent (theirs):** HUGE — 3312 ins / 389 del across ~190 hunks (old
lines 13–25321; characterized by hunk-header category, not read in full: the
majority (~130 hunks) are `load_cl_kernels()` new SPIR-V/CL kernel
registrations; `ggml_backend_opencl_context`/FA-kernel-variant struct growth;
`ggml_cl_init` device-capability probing growth; new tensor-extra struct fields
for q5_0/q8_0/q5_K; a large `graph_compute`/`synchronize` block; Adreno-specific
kernel-dispatch helpers and an expanded `supports_op`; buffer set/get_tensor
additions for new dtypes; **implements `ggml_backend_opencl_buffer_type_get_alloc_size`**
(was `NULL`) at old line ~9763, immediately followed by `device_get_props`
`mmap_support` (F2) at ~9815; large new elementwise/FA CL-op implementations;
`mul_mat`/`mul_mat_id`/Adreno mul_mat variant growth).

**RESOLVE:** **Corrected — auto-merges cleanly, verified against the actual
merged blob.** An earlier pass through this brief flagged this as a genuine
conflict because upstream's hunk at old line 9763–9776 (implementing
`ggml_backend_opencl_buffer_type_get_alloc_size` and wiring it into
`ggml_backend_opencl_buffer_type_interface`) and the fork's `get_caps` hunk at
old line 9770 both touch the same ~7-line struct literal. **`git merge-tree
--write-tree master b10630` shows this file is NOT in the conflict set** —
confirmed by reading the actual merged blob
(`git show 135b7a4a7:ggml/src/ggml-opencl/ggml-opencl.cpp`), which contains
exactly the correct union of both changes:

```c
static ggml_backend_buffer_type_i ggml_backend_opencl_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_opencl_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_opencl_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_opencl_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_opencl_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_opencl_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
    /* .get_caps         = */ NULL,
};
```

Git's line-based 3-way merge algorithm resolves this correctly on its own
because the two edits touch different lines within the same literal
(`.get_alloc_size` vs. an appended trailing `.get_caps` line) even though they
share hunk context — no hand-merge needed, only the sanity check above (already
done). The first fork hunk (`buffer_interface` at old line 9707) has no upstream
overlap and auto-merges trivially. Take the merge result as-is for everything
else (do not attempt to read/reconcile the ~3700-line kernel-registration body
by hand — it is purely additive new kernel variants, none of it near either
fork hunk).

**CONTRACT:** None — Adreno/OpenCL kernel dispatch is backend-siloed; no shared
symbols with `ggml-sycl.cpp`.

**Mechanical-rename check:** Yes for the fork's own 2-line diff — pure
`get_caps` pattern.

---

### `ggml/src/ggml-openvino/ggml-openvino.cpp`

**Fork intent (ours):** 3 one-line `get_caps` (F1) additions (old lines 420,
480, 536).

**Upstream intent (theirs):** 353 lines across ~20 hunks — buffer-set-tensor
additions for new dtypes; `device_get_props` (mmap_support-style, F2); two new
overflow-safe helpers `checked_mul_size`/`mul_mat_id_requires_large_tmp`; a
substantial rewrite of `is_op_unsupported_case` (includes a 47-line
deletion+replace); `device_supports_op` tail tweaks.

**RESOLVE:** Take upstream wholesale, re-apply the fork's 3 `get_caps` lines on
top — fork's hunks (420/480/536) fall in gaps between upstream's hunks (nearest
neighbors at 32–38, 135–279, 458–465, 618–624 — none intersect 420–539).

**CONTRACT:** None — OpenVINO op-support predicates and overflow helpers are
backend-siloed; no shared symbols with `ggml-sycl.cpp`.

**Mechanical-rename check:** Yes — pure `get_caps` pattern.

---

### `ggml/src/ggml-rpc/ggml-rpc.cpp`

**Fork intent (ours):** 2 one-line `get_caps` (F1) additions on
`ggml_backend_rpc_buffer_interface` (old line 539) and
`ggml_backend_rpc_buffer_type_interface` (old line 636).

**Upstream intent (theirs):** 114 lines, a real wire-protocol change (not
mechanical): `struct rpc_tensor`'s `char padding[4]` → `int32_t use_count`
(layout-compatible, semantically different); new `RPC_CMD_MEMSET_TENSOR`
command + `rpc_msg_memset_tensor_req` + client-side
`ggml_backend_rpc_buffer_memset_tensor()` wired into `.memset_tensor` (was
`NULL`) + server-side `rpc_server::memset_tensor()` with bounds-checking;
`send_msg`/`send_rpc_cmd` now `flush()` after sending; `add_tensor`/
`serialize_graph` thread `cgraph` through so `use_count` is populated from
`cgraph->use_counts`, and `rpc_server::graph_compute` reconstructs
`graph->use_counts` from the wire value — i.e. operator ref-counts (used for
backend fusion/liveness decisions) now propagate across the RPC wire;
`device_get_props` adds `mmap_support` (F2).

**RESOLVE:** Take upstream wholesale, re-apply the fork's 2 `get_caps` lines on
top. No textual overlap — upstream's struct-literal edit to
`ggml_backend_rpc_buffer_interface` (old ~531–537, changing `.memset_tensor`)
ends 2 lines before the fork's hunk starts (539, appending `.get_caps` after
`.reset`); both land in the same struct but in non-overlapping hunks and
compose cleanly. The second fork hunk (buffer_type_i at 636) has no nearby
upstream hunk at all.

**CONTRACT:** **Yes — genuine, not backend-siloed.** RPC is a generic transport
wrapping whatever backend the `rpc-server` binary links, including SYCL. Two
concrete points of contact, both confirmed via grep: (1) `RPC_CMD_MEMSET_TENSOR`
forwards to `tensor->buffer->iface.memset_tensor`, which SYCL already
implements for its default buffer type (`ggml_backend_sycl_buffer_memset_tensor`,
`ggml-sycl.cpp:33652`, wired at `:33747`) — a SYCL-backed `rpc-server` gains
real memset-over-RPC support once this lands. Its split-buffer and
tensor-parallel (`_tp_`) buffer types leave `.memset_tensor = NULL`
(`:36101`, `:36823`), so `rpc_server::memset_tensor()`'s
`iface.memset_tensor == nullptr` guard correctly refuses those over RPC —
expected, not a regression. (2) The `use_count` propagation and protocol
version bump interact with `ggml-rpc.h`'s `GGML_OP_COUNT` static_assert (see
that file's entry and F3) — must be resolved in the same commit.

**Mechanical-rename check:** Fork's own 2-line diff is pure `get_caps` pattern
— no RPC-protocol logic in the fork's touch here.

---

### `ggml/src/ggml-virtgpu/ggml-backend-buffer-type.cpp`

**Fork intent (ours):** 2 one-line `get_caps` (F1) additions —
`ggml_backend_remoting_buffer_type_interface` (old line 69) and
`ggml_backend_remoting_buffer_from_ptr_type_interface` (old line 79, file's
last line).

**Upstream intent (theirs):** 4 lines — `apir_device_get_props()` gains a 5th
out-param (`mmap_support__unused`) threaded through the virtgpu IPC layer (F2
analog), and its call site is updated to pass it.

**RESOLVE:** Take upstream wholesale, re-apply the fork's 2 `get_caps` lines on
top — fork's edits are at the struct-literal tails (69, 79), upstream's edit is
inside the function body (~11–17), no overlap.

**CONTRACT:** None — virtgpu/virtio-GPU passthrough is entirely siloed from
SYCL; `apir_device_get_props` is virtgpu-internal IPC with no SYCL analog.

**Mechanical-rename check:** Yes — pure `get_caps` pattern.

---

### `ggml/src/ggml-vulkan/ggml-vulkan.cpp`

**Fork intent (ours):** 3 hunks — `get_caps` (F1) on
`ggml_backend_vk_buffer_type_interface`, `ggml_backend_vk_buffer_interface`,
and the inline host-buffer-type literal (with reflow).

**Upstream intent (theirs):** HUGE — 1244 ins / 280 del across ~200 hunks (old
lines 128–19556; characterized by hunk-header category, not read in full): new
Vulkan feature-struct plumbing and topk-moe additions; `vk_device_struct`
capability-field growth; new push-constant structs for binary/rope/snake/
wkv7/ssm_scan ops; FLOPs-estimation and fence-wait/pipeline restructuring; a
very large `ggml_vk_load_shaders()` expansion (biggest single block, dozens of
new shader-variant registrations); device-selection growth; buffer
read/write/copy/memset signature tweaks; FA implementation growth
(`flash_attn_coopmat`); new elementwise/rwkv7/ssm_scan/pad/rope/topk_moe/snake
op-dispatch; `ggml_backend_vk_graph_compute` restructuring; `device_get_props`/
`device_supports_op` growth (many small dtype/shape gates).

**RESOLVE:** Take upstream wholesale, re-apply the fork's 3 `get_caps`/reflow
hunks on top — fork's touches (292/15295/15398–15413) fall in gaps between
upstream's hunk sequence (nearest neighbors bracket but don't cover those
ranges).

**CONTRACT:** None — SPIR-V shader variants, `vk_device_struct` capability
fields, and FA/mul_mat pipeline tuning are Vulkan-specific; no shared symbols
with `ggml-sycl.cpp`.

**Mechanical-rename check:** Yes for the fork's own diff — pure `get_caps`
pattern, no Vulkan shader/dispatch logic touched.

---

### `ggml/src/ggml-webgpu/ggml-webgpu.cpp`

**Fork intent (ours):** 2 hunks — `get_caps` (F1) on
`ggml_backend_webgpu_buffer_interface` and the inline buffer-type literal
inside `ggml_backend_webgpu_device_get_buffer_type` (with minor reflow).

**Upstream intent (theirs):** 420 lines across ~24 hunks — `ggml_webgpu_tensor_buf`
growth; new `ggml_webgpu_conv_2d_dw` (depthwise conv) op + dispatch entry;
`ggml_webgpu_ssm_scan` restructuring; `set_rows`/`mul_mat` tweaks; a
flash-attention API split (`ggml_webgpu_flash_attn_kv_direct` →
`_k_direct`/`_v_direct`); binary/concat/rms_norm/rope/glu op-dispatch tweaks;
`device_get_props` adds `mmap_support` (F2); `device_supports_op` dtype/shape
gates.

**RESOLVE:** Take upstream wholesale, re-apply the fork's 2 `get_caps` hunks on
top — no overlap (fork's hunks fall in gaps between upstream's nearest
neighbors).

**CONTRACT:** None — WebGPU's new depthwise-conv op and FA K/V-direct path
split are backend-specific; no shared symbols with `ggml-sycl.cpp`.

**Mechanical-rename check:** Yes — pure `get_caps` pattern.

---

### `ggml/src/ggml-zdnn/ggml-zdnn.cpp`

**Fork intent (ours):** 2 hunks — `get_caps` (F1) on `ggml_backend_zdnn_buffer_i`
and the inline `ggml_backend_zdnn_buffer_type()` literal (with reflow).

**Upstream intent (theirs):** 3 lines — `ggml_backend_zdnn_device_get_props`
adds `mmap_support` (F2).

**RESOLVE:** Take upstream wholesale, re-apply the fork's 2 `get_caps` hunks on
top — no overlap (fork touches old lines 318/383–396, upstream touches only
487–490).

**CONTRACT:** None — IBM Z zDNN NNPA accelerator is entirely siloed from SYCL.

**Mechanical-rename check:** Yes — the smallest, cleanest instance of the F1
pattern in the whole group; structurally identical to CANN/Hexagon/OpenVINO/
virtgpu.

---

### `ggml/src/ggml.c`

**Fork intent (ours):** 9 hunks (old lines 1034, 1094, 1145, 1205, 1796, 1903,
3942, 5458, 6195). `GGML_OP_NAME[]`/`GGML_OP_SYMBOL[]` entries for
`SET_ROWS_PAGED`/`ALL_REDUCE_SUM` plus their `static_assert(GGML_OP_COUNT ==
99, ...)` (both occur **twice** in this file — after each array — F3);
`ggml_new_tensor_impl` initializes the new `buffer_offs`/`layout` fields (F4:
`buffer_offs = view_src != NULL ? view_src->buffer_offs + view_offs : 0`,
`layout = NULL`); `ggml_get_data`-adjacent getter for `layout`; the full
`ggml_set_rows_paged()` builder implementation (validates 4D dst, 1-2D src,
1D indices, 1-2D block_table, sets `GGML_OP_SET_ROWS_PAGED` + op_params);
`ggml_flash_attn_ext_add_sinks`-adjacent `ggml_flash_attn_ext_set_paged_layout`;
`ggml_opt_step_sgd`-adjacent `ggml_all_reduce_sum()` builder.

**Upstream intent (theirs):** 22 hunks (old lines 1079, 1096, 1190, 1207, 1462,
1487, 4022, 4180, 4192, 4402, 4505, 4539, 4645, 4727, 4777, 5423, 5567, 5604,
5619, 6287, 7017, 7671). `GGML_OP_NAME[]`/`GGML_OP_SYMBOL[]` entries for the 4
new upstream ops + `static_assert(... == 101, ...)` (also twice — F3);
`ggml_is_contiguous_to_{1,2,3}` implementations; `ggml_clamp`/`_inplace`
relocated + new `ggml_rope_set_offset`; conv_1d/1d_dw/2d/3d/2d_dw signature
tweaks; `ggml_flash_attn_ext` addition; **`ggml_ssm_scan` implementation
updated for the new `K` param** (F5, three hunks: 5567/5604/5619); a 168-line
addition after `ggml_gated_delta_net` — the `lightning_indexer`/`dsv4_hc_*`
builder implementations; `ggml_build_forward_expand`-adjacent
`ggml_build_forward_order`; `ggml_set_input` tweak.

**RESOLVE:** Verified zero line-range overlap between all 9 fork hunks and all
22 upstream hunks (checked pairwise; the closest near-misses — fork's
`5458–5463` against upstream's neighbors at `5423–5429`/`5567` — still leave a
gap). **Except** the `static_assert(GGML_OP_COUNT == N, ...)` lines (F3) —
those are on the exact same line as each array (twice per file) and are a
genuine conflict requiring the hand-fix to `103` described in F3. Everything
else auto-merges cleanly via ordinary 3-way merge, and — importantly — the
`GGML_OP_NAME`/`GGML_OP_SYMBOL` array entries land in the textually-correct
relative order automatically (each side's new-op strings sit at the same
relative position as its enum insertion; verified by reading both sides' full
array diffs), so **no manual array reordering is needed**, only the
`static_assert` values.

**CONTRACT:** F3 (enum/array/assert — the headline finding of this whole
brief), F4 (`buffer_offs`/`layout` init in `ggml_new_tensor_impl` — untouched
by upstream, confirmed no b10630 hunk near old lines 1796–1812), F5
(`ggml_ssm_scan` K param — implementation-side of the signature change in
`ggml.h`; SYCL's `supports_op` gap already flagged). All three already
described above in full; nothing additional to add at the ggml.c level.

**Mechanical-rename check:** N/A (core file, not an other-backend touch) —
included here only per the acceptance criteria's per-file requirement; both
sides' changes are substantive, not mechanical.

---

## Summary — files with a genuine merge-tree conflict, and the one auto-merge semantic gap

Verified against `git merge-tree --write-tree master b10630` (tree `135b7a4a7`)
plus `git merge-file -p --diff3` on each conflicted path's three stage blobs —
this is ground truth for git's real 3-way merge, not an inference from hunk
proximity. **Exactly three of the 26 files in this group produce conflict
markers; every other file auto-merges with no markers at all:**

| File | Conflict (from the actual diff3 output) | Resolution |
|---|---|---|
| `ggml/include/ggml-rpc.h` | Two conflict blocks: `RPC_PROTO_{MAJOR,MINOR,PATCH}_VERSION` (4/0/2 vs. 5/1/0), and `static_assert(GGML_OP_COUNT == 99 ...)` vs `== 101` | `GGML_OP_COUNT → 103`; take upstream's `MAJOR 5`/`MINOR 1` bump (it reflects the real `memset_tensor`/`use_count` protocol change), pick a fresh `PATCH` (e.g. `0`) |
| `ggml/src/ggml.c` (2 conflict blocks, same file) | `static_assert(GGML_OP_COUNT == 99 ...)` vs `== 101`, once after `GGML_OP_NAME[]` and once after `GGML_OP_SYMBOL[]` — confirmed via diff3 to be the **only** two conflict regions in the file; everything else (both sides' array insertions) merged clean | Same fix, `→ 103`, both sites |
| `ggml/src/ggml-backend-meta.cpp` | One conflict block: `.memset_tensor` struct-literal line — fork's cosmetic whitespace edit vs. upstream's real `ggml_backend_meta_buffer_memset_tensor` implementation | Take upstream's implementation; drop the fork's cosmetic edit |

**Two files an earlier pass through this brief wrongly called conflicts —
corrected above after checking merge-tree, both auto-merge cleanly with no
markers:** `ggml/src/ggml-backend-reg.cpp` and
`ggml/src/ggml-opencl/ggml-opencl.cpp`. The opencl one needs nothing further —
the merged blob already contains the correct union. The backend-reg one auto-
merges textually but leaves a **semantic** gap worth a deliberate 3-line fix
(not a hand-merge): the new `GGML_USE_ET` registration lands without the
`disable_device_backends` guard every sibling backend registration has — see
that file's entry above for the exact merged snippet and fix.

All other findings in this brief are semantic, not textual — F3's array
ordering (auto-merges correctly, confirmed by diff3, no action needed), F5's
`ggml_ssm_scan` SYCL `supports_op` gap, and F2's `mmap_support` SYCL gap. The
three genuine conflicts are each a 1–10 line hand-fix once identified; the one
semantic gap in the auto-merged `ggml-backend-reg.cpp` is a similarly small,
deliberate fix.
