# RFC: Compile-time barrier on `tensor->data` in SYCL op code

**Date**: 2026-04-18 · **Author**: implementer-2 (A2) · **Bead**: `llama.cpp-2o6zy` (epic `llama.cpp-wjvse`) · **HEAD**: `56ead4cce`

## Headline recommendation

**PURSUE, incremental.** The `sycl_tensor` wrapper already exists
(`ggml/src/ggml-sycl/sycl-tensor.hpp:5-71`) and is adopted by 229 occurrences
across 47 files. The remaining work is finishing migration of the
concentrated residue (188 raw `->data` reads in `ggml-sycl.cpp`, 19 in
`common.hpp`) and adding a CI gate that fails builds on new `->data` reads
from op code. No wrapper redesign needed; no new abstraction needed. Execute
via the existing `llama.cpp-wjvse` T3-T9 plan.

## Problem

`tensor->data` is ambiguous at access time: it may be mmap (non-USM host),
pinned USM host, or device USM, depending on where the tensor lives at that
moment. Cache eviction, zone resets, buffer resets, and USM staging can
re-home `->data` between ops. Raw access bypasses the unified cache's
residency state machine and silently fails under memory pressure — the
GPT-OSS MoE use-after-free class (`llama.cpp-mqxer`, `4oi3i`) is a symptom.
The user's architectural principle ("arena owns everything, `mem_handle`-only
access, any model just works") requires raw `->data` in GPU-dispatch code to
be **impossible**, not merely discouraged.

## Current state

`ggml_sycl::sycl_tensor` (sycl-tensor.hpp:5-71) is a 16-byte wrapper
(`static_assert` line 73) over `const ggml_tensor*` + `int device`. Exposes
metadata (`ne`, `nb`, `type`, `name`, `extra_gpu()`, `src`, `view_src`, etc.)
but has **no** `data` method. Access via `resolve_ptr()` (line 60) /
`resolve_as<T>()` (line 62), routed through the cache-aware
`ggml_sycl_resolve_tensor_ptr` (common.hpp:2346-2365).

**Adoption**: 229 `sycl_tensor` occurrences across 47 files
(`grep -rcE '\bsycl_tensor\b' ggml/src/ggml-sycl` summed, HEAD 56ead4cce).
Modules already migrated: `add-id`, `norm`, `cpy`, `pad`, `pad_reflect_1d`,
`wkv`, `concat`, `roll`, `tsembd`, `ssm_conv`, `gla`, `fattn`, `softmax`,
`element_wise`, `binbcast`, `getrows`, `set_rows`, `repeat_back`,
`count-equal`, `conv`, `im2col`.

**Residue**: 215 raw `->data` reads across 7 files at HEAD 56ead4cce —
`ggml-sycl.cpp` 188, `common.hpp` 19, `cpu-dispatch.cpp` 5 (comments only),
`fattn.cpp` 2 (comments only, lines 994 and 1030), `common.cpp` 1.
Docs-only hits excluded.

**Methodology**: counts from `grep -cE '>data\b'` per file. Unbounded
`grep -c 'data'` matches many false positives (log format specifiers,
identifiers like `data_device` / `data_ptr`, etc.); the `>data\b` pattern
isolates tensor-member access. Migration progress: 229/(229+215) ≈ 52% by
callsite, up from 0 at `wjvse` creation (2026-04-07, 238 raw reads
reported).

**Residue distribution** in `ggml-sycl.cpp` (188 hits by 10k line buckets on
HEAD 56ead4cce): 0-9999 = 67, 10000-19999 = 63, 20000-29999 = 13,
30000-39999 = 5, 40000+ = 40. By *function* (stable across drift): resolver
implementation in `ggml_sycl_get_data_ptr_slow` and
`ggml_sycl_resolve_tensor_ptr`; C-API callbacks in the
`ggml_backend_sycl_buffer_{init,set,get}_tensor` family (both legitimate);
weight/expert prestage inside `moe_hybrid_init_once` and S1-PRELOAD
(mostly legitimate); **15 MoE-dispatch gap sites** tracked as A1
GAP-1..GAP-5 in `llama.cpp-zxm5m` (the mqxer/4oi3i bug class —
`ggml_sycl_get_expert_ptr` cache-miss fallback, MoE norm capture,
`g_gate_up_pairs` fused-FFN map, ADD_ID bias capture, ExpertPredictor
gate-weight registration); remainder is orchestration/debug. Locate each
function via `mcp__serena__find_symbol`.

## Proposed shape

**Shape A (recommended, in progress)**: finish the migration.
1. Fix the 15 MoE-dispatch gap sites (A1 GAP-1..GAP-5; `wjvse` T3).
2. Leave legitimate resolver/buffer-interface reads in place.
3. Add a CI grep gate (wjvse T9): fail on new `->data` in SYCL source
   outside an allowlist of known-good files.

**Shape B (rejected)**: compile-time `#pragma GCC poison data` in SYCL TUs.
Too disruptive for resolver/buffer-interface files that MUST read `->data`;
would require scope-level `pop`/`push` gymnastics or TU surgery. File-level
CI grep gives equivalent enforcement with no compiler magic.

## Cost

~400 LOC total across 15-20 commits over 1-2 weeks. Reviewer load ~30-45
min per commit (8-12 hours aggregate). No header churn (`sycl-tensor.hpp`
already in tree). No ggml-core divergence (C struct untouched; barrier is
SYCL-internal). No build-system changes. Perf-neutral: `sycl_tensor` is
pass-by-value 16 bytes, all accessors inline; `resolve_ptr` has a ~3 ns
handle-generation fast path (common.hpp:2354).

## Risks

- **Half-state silent-bug surface**: during the 1-2 week rollout,
  `sycl_tensor` and `ggml_tensor*` coexist via implicit conversion
  (sycl-tensor.hpp:10 non-explicit ctor). That enables gradual rollout but
  also means an `sycl_tensor` argument silently flowing into an unmigrated
  path loses the access discipline at the boundary. Mitigation: finish
  migration quickly; CI gate (T9) catches new regressions.
- **Incremental state**: acceptable — the 229 existing adoptions proved
  the coexistence pattern works.
- **CI allowlist maintenance**: low; legit file set is small and stable.
- **Performance regression**: none observed in the 229 already-migrated
  sites; no reason to expect one in the residue.

## Open questions

- **CI gate granularity** (`wjvse` T9 starting context):
  - *Phase 1* (ship with migration): file-level allowlist — small,
    stable set of files implementing the resolver and C-API callbacks.
  - *Phase 2* (post-migration, residue < 50 sites): reconsider
    function-level annotation, building on the existing "raw-ok" comment
    seeds in cpu-dispatch.cpp and fattn.cpp.
  This gives T9 a constraint rather than a blank canvas.
- **gate-inp host-buffer read** at `ggml-sycl.cpp:3222`: route through
  `resolve_ptr` for uniformity, or carve an explicit `host_tensor_read`
  API? Style call for `wjvse` T5 owner.

## Recommendation

**PURSUE, Shape A.** The 15 MoE-dispatch gaps (A1 GAP-1..GAP-5 in
`llama.cpp-zxm5m`) are the single biggest migration win — fixing them
directly eliminates `mqxer`/`4oi3i`-class crashes, the entire bug class
this RFC exists to address. Start there. Continue with `wjvse` T4-T8 for
the remaining orchestration residue, then T9's CI gate to lock in gains.
Do NOT pursue Shape B.
