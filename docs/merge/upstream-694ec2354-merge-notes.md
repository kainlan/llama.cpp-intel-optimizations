# Upstream 694ec2354 Merge Notes (llama.cpp-n77l)

**Merge:** ggml-org/master tip `694ec2354` (`b10944-260`), 574 commits past the
b10630 merge base `d222767c7`, into fork master `76c7f6548`.
**Why now:** brings in Qwen3.8-Flash-Next (`qwen4exp`, upstream `6c84c7d5d`
#27742) and TENSOR_READ_LAZY (`fac889fb3`, `50f068fff`).
**Shape:** one `git merge`, `ggml/src/ggml-sycl/` resolved keep-ours with a
whole-tree identity check (the b10630 precedent,
`upstream-b10630-campaign.md`), everything else resolved per hunk and built
before commit.

## Fork invariants checked

| invariant | status |
|---|---|
| unified cache is the only SYCL allocator | held: upstream's new SYCL allocators (`mem.cpp`, `memtrace.cpp`) were not taken; TENSOR_READ_LAZY allocates no SYCL memory (below) |
| `mem_handle` is the ownership token | held: no SYCL source outside keep-ours changed except the two edits listed below |
| placement decides the executor | held: ops the fork's SYCL backend refuses (below) fall to the CPU through the scheduler, which is the existing route |
| `fit_params=false` under SYCL | held: `common/common.h:483` (`#ifdef GGML_USE_SYCL` arm) survived the merge unchanged |

## `ggml/src/ggml-sycl/` - keep-ours

`git checkout HEAD -- ggml/src/ggml-sycl/`, then:

- delete-vs-modify files the fork had removed stay removed:
  `dsv4-hc.cpp`, `fattn-mkl.cpp`, `fusion.cpp`, `fwht.cpp`;
- upstream-new files are not taken: `base.hpp`, `fattn-sparse.{cpp,hpp}`,
  `mem.{cpp,hpp}`, `memtrace.{cpp,hpp}`, `topk-radix.{cpp,hpp}`.
  `mem.cpp` / `memtrace.cpp` are a second allocation layer with its own
  accounting, which the canonical memory contract forbids.

After resolution the tree differs from fork master in exactly two places:

1. **`ggml-sycl.cpp` `ggml_backend_sycl_split_buffer_type`** now takes
   `([[maybe_unused]] int main_device, const float * tensor_split)` -- a port of
   upstream `aa39d7a3e`. This fixes a latent ABI mismatch that predates the
   merge: `ggml_backend_split_buffer_type_t` (the type the registry hands out
   through `get_proc_address`) takes `(int main_device, const float *)`, while
   the fork's function took only the pointer, so a caller through the registry
   passed `main_device` where `tensor_split` was expected.
2. **The FA paged-layout flag moved from `op_params[4]` to `op_params[5]`**
   (`ggml.c` `ggml_flash_attn_ext_set_paged_layout`; readers at
   `fattn.cpp` `ggml_sycl_fattn_d512_has_paged_or_multiseq_sources` and the
   FA entry point, and the two graph-level screens in `ggml-sycl.cpp`, the
   fast-path check and `[PERSISTENT-TG] FLASH_ATTN unsupported extras`).
   Upstream now writes `n_kv_max` into slot 4
   (`ggml_flash_attn_ext_set_n_kv_max`, called by qwen4exp's sparse attention
   at `src/llama-graph.cpp:2697`). Left in slot 4, every sparse-FA node would
   read as "paged layout" to the SYCL backend: the graph screens would drop the
   fast path and the FA kernel would index K/V through a block table that does
   not exist. The fork's setter has no callers in `src/`, `common/` or `tools/`,
   so no producer moves with it. After the move the SYCL backend ignores
   `n_kv_max`, which is exact rather than approximate: the header contract is
   that `n_kv_max` only bounds the finite mask entries, the mask itself carries
   the sparsity, and the CPU backend also computes dense over the mask. CUDA
   uses it as a gather optimisation only. Slot 5 is unused by every FA setter
   upstream and in the fork.

### Upstream SYCL commits in `d222767c7..694ec2354` (34), all dropped by keep-ours except `aa39d7a3e`

Notable for follow-up (each is a feature the fork either has in its own form
or does not need to take verbatim):

- `cd74ef627` support sparse FA -- see the slot-5 move above; the fork computes
  dense over the mask. A sparse gather is a perf item, not a correctness one.
- `bb3c853c3`, `37b53fd45` DSV4_HC / HC ops for qwen4exp -- the fork's
  `supports_op` does not list them, so they run on the CPU.
- `0190529ec` SWIGLU_CLAMP -- refused by the fork's `supports_op`, runs on CPU.
- `4d7d7703f` GET_ROWS_BACK -- not supported, runs on CPU.
- `21f6b0d22`, `370cb12e8` TOP_K radix select / long rows -- the fork keeps its
  own TOP_K.
- `cd8cdf397` memtrace, `5eec3ad01` 2 GB host-pinned cap, `c9a5eeeb3` B70
  > 19.3 GB alloc, `af911149c` pinned-memory device context, `661643e43` oneDNN
  scratchpad pool order, `b6b003d2c`, `a32af33de` free-memory queries,
  `cc83d7b48` `--fit` -- all allocator / fit machinery that the unified cache
  replaces, or that `fit_params=false` disables.
- `0df974d77` peer-to-peer copy API -- no P2P on this host's topology
  (CLAUDE.md, "Patched compute-runtime & P2P topology").
- `5e48b3100`, `6703d7894`, `817e5f83e`, `dbeb37548`, `4aa6ffba2`,
  `d077b4c21`, `be876204a`, `c350a40bb`, `1aa2954bd`, `304665fe7`,
  `2a3005c23`, `384a534ce`, `4d9176092`/`1f3d31873`/`c845263f8`,
  `f9f09f02c`, `e97545d91` -- kernel fusions, tuning and quant handling in
  upstream's SYCL layout; candidates for a port ledger like
  `upstream-b10630-sycl-audit.md`, not merge material.

## TENSOR_READ_LAZY on a SYCL build

`llama_model_loader::lazy_read::buft()` (`src/llama-model-loader.cpp:1232-1238`)
returns `ggml_backend_dev_buffer_type(ggml_backend_dev_by_type(CPU))`: the
ggml-cpu buffer type, never a SYCL one. `buft_for_tensor` returns it for any
tensor `lazy_read::add` accepted (`src/llama-model-loader.cpp:1368-1369`), and
the context map is keyed by `ctx_key{buft, lazy}`. In `llama_model::load_tensors`
the lazy context is then either wrapped over the file mapping with
`ggml_backend_dev_buffer_from_host_ptr(cpu_dev, ...)` (the device comes from
the buffer type, `src/llama-model.cpp:2730`)
(`src/llama-model.cpp:2746-2765`, when `is_lazy_mapped && use_mmap_buffer &&
buffer_from_host_ptr_supported && is_default_buft`) or allocated as a plain
ggml-cpu buffer (`src/llama-model.cpp:2777`). Neither path reaches the SYCL
backend or the unified cache, so no SYCL allocation happens outside it.

The lazy tensor is host-resident and so executes on the CPU (placement decides
the executor). `AUTO`, the default (`src/llama-model.cpp:3791`), only makes a
tensor lazy above 4 GiB (`lazy_read::add`, `src/llama-model-loader.cpp:1246-1249`),
which today means qwen4exp's PLE table and very large Gemma 4
`per_layer_tok_embd` tables.

**On a SYCL build `AUTO` resolves to `OFF`, so lazy reading is inert unless
`-lzm on` is passed.** `llama_model::load_tensors` turns `AUTO` into `OFF` when
any device reports `caps.mmap_support == false` (`src/llama-model.cpp:2256-2265`,
upstream #28160), and `ggml_backend_sycl_device_get_props` fills `props->caps`
with four of its five members (`ggml-sycl.cpp:107798-107803`), so the fifth,
`mmap_support`, is zero. That predates this merge: the same check already
turns `load_mode=auto` into no-mmap for SYCL (`src/llama-model.cpp:2245-2253`,
`153d324bc`, in fork master before this merge). Upstream's SYCL backend sets
`mmap_support = true`. Whether the fork's zero is intended is a separate
question from this merge and is not changed here; with it, a default qwen4exp
load keeps its PLE table resident, and only `-lzm on` exercises the lazy path.

The fork's SYCL loops over the context map were adapted to the new key
(`it.first` -> `it.first.buft` in the late-inventory and the dev-layer sync
loops). The lazy tensor's name does not start with `blk.`, so the dev-layer
sync ignores it.

`src/llama-mmap.cpp`: the fork's Linux prefetch (`POSIX_MADV_SEQUENTIAL` +
`WILLNEED`, no `MAP_POPULATE`) now runs over `ranges_complement(lazy_ranges)`
through upstream's `advise` lambda, so lazily read ranges are not prefetched;
`MADV_HUGEPAGE` is kept for the whole mapping.

CLI: the flag is `-lzm, --lazy-mode <on|auto|off>` in `common/arg.cpp` and
`llama-bench`; there is no `--tensor-read-lazy`.

## Per-file resolutions

- **`ggml/include/ggml-sycl.h`** - ours, plus upstream's new
  `split_buffer_type` signature; dropped upstream's "pins on device 0" comment,
  which describes upstream's allocator.
- **`ggml/src/CMakeLists.txt`, `src/CMakeLists.txt`** - took upstream's
  generated `ggml-version.h` / `llama-version.h` (`configure_file`), which
  replaces the fork's per-source `GGML_COMMIT` / target `LLAMA_VERSION`
  defines and keeps the commit id off every compile command, the goal of
  llama.cpp-vuy0 reached by a different route. Took upstream's
  `LLAMA_CORE_SOURCES` + unity build, and added the fork-local sources that
  the explicit list would otherwise drop: `llama-kv-block.cpp`,
  `llama-pp-scheduler.cpp`, `llama-moe-profile.cpp`, `llama-tensor-class.cpp`
  (every `src/*.cpp` is listed).
- **`tests/test-build-info-scope.py`** - now reads the commit id from the
  generated `ggml-version.h` and requires it on no edge command (it used to
  require exactly one compile edge carrying `-DGGML_COMMIT`). The build-number
  check is unchanged.
- **`tests/test-sycl-auto-ubatch-source.py`** - upstream put
  `llama_graph_n_input_tensors()` between `sycl_select_auto_ubatch()` and
  `sched_reserve()`, so the trial-body end marker now names that helper.
  Before the change, the body took in the helper's `LLAMA_LOG_WARN`, which
  made four warnings where three are expected. The settle-reserve mutation
  witness now counts and mutates inside the trial only, because upstream's
  `opt_init()` carries the same three lines.
- **`tools/ui/CMakeLists.txt`, `tools/ui/embed.cpp`** - upstream embeds the UI
  in CMake (`scripts/ui-assets.cmake`, `emit_files()`), so the fork's
  `llama-ui-embed` executable is gone; the fork's stamp-based asset step is
  kept. The fork's "loading.html is optional" change (fdm1) is subsumed:
  upstream's `ui_validate_assets` does not require it.
- **`common/chat.cpp`, `common/parsers/gpt-oss.cpp`** - upstream moved the
  parsers into `common/parsers/`; `chat.cpp` is upstream's, and the fork's
  GPT-OSS changes moved with the parser: `common_chat_extra_context_true()`,
  the `llama_force_final_channel` block, and the Harmony thinking tags
  (`<|start|>assistant` / `<|channel|>final<|message|>`, llama.cpp-90ns).
- **`common/arg.cpp`** - upstream's video options; `--rpc` stays
  unconditional (the fork removed the `llama_supports_rpc()` guard so metadata
  commands stay backend-free).
- **`ggml/src/ggml-backend.cpp`** - ours (include superset).
- **`ggml-hexagon.cpp`, `ggml-vulkan.cpp`** - upstream; the fork's trailing
  `get_caps` interface member is zero-initialised in their aggregates.
- **`tools/rpc/rpc-server.cpp`** - upstream's CPU/ACCEL device-type check plus
  the fork's `if (!dev) continue;`.
- **`src/llama-model-loader.cpp`** - both: the fork's host weight layout /
  `select_weight_buft_cpu_owned` and upstream's `lazy_read`.
- **`src/llama-model.cpp`** - both: the fork's SYCL late-inventory block, then
  upstream's `prec_policy.load()`; the buffer loop is upstream's
  `ctx_key` form with the fork's `has_sycl_weight_buffer` flag.
  `hparams.n_expert_used` became a per-layer accessor upstream; the SYCL
  inventory now uses `n_expert_used_max()` (it sizes buffers, so the maximum
  is the right reading), as does `llama-moe-profile.cpp`.
- **`src/llama-context.cpp`** - both: the fork's auto-ubatch trial and
  upstream's `llama_graph_n_input_tensors`.
- **`tests/test-alloc.cpp`, `tests/test-backend-ops.cpp`** - both.
- **`tests/test-llama-archs.cpp`** - hand-merged from ours: upstream's `-d`
  stdev, `-v N` verbosity, `-o` save dir and `-b` backend filter added; the
  fork's exact-name `-a` with its argument-parse failure (gated by
  `test-archs-exclude-cli.sh`), `-x`, `--nan-trace`, the 77 skip exit and the
  table gate hooks kept.
- **`docs/backend/SYCL.md`** - ours.

## Test registration changes from upstream (GPU hazard)

- `test-llama-archs` and `test-backend-ops` are now `llama_build` only -- no
  longer ctests. Run them directly, pinned:
  `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/test-llama-archs -a llama`.
- `test-save-load-state` moved from label `model` (one downloaded model) to
  label **`main`** and now loads **every generated arch in one process** on
  the default devices. `test-recurrent-state-rollback-kimi-k3` is new, also
  `main`. Both depend on the `generate-models` fixture (`test-llama-archs -o`,
  which also builds every arch's model on the default devices). None of them
  pins a selector, and before this merge the CLAUDE.md "form 2" full sweep
  (`-LE 'residency|mem-handle|cache' -E '^test-backend-ops$'`) selected all of
  them. It also selected four loaders that were there before the merge and
  need a downloaded model: `test-thread-safety`,
  `test-sycl-model-lifecycle-hooks`, `test-state-restore-fragmented` and
  `test-eval-callback`. Treat them like `test-llama-archs`: pinned, single
  run, lead only.
- Fixed in the follow-up commit. `scripts/check-ctest-safety-net.sh` now
  derives the loader set from the registration. A loader is any test with
  `FIXTURES_REQUIRED`, or a fixture setup that runs a binary rather than
  cmake. The guard fails naming each loader the documented sweep selects, and
  also fails if CLAUDE.md stops carrying the sweep's exact `-E`. CLAUDE.md
  form 2 and PR step 3 now exclude all ten loaders by name.
  `tests/merge-guards/test-ctest-safety-net.sh` drives it with hermetic REDs
  plus one real-build RED: drop `test-save-load-state` from the exclusion, and
  the guard fails naming only that test.

## Host-only gates run on the merge

The merge was built in full (`sycl-build.sh`, 266/266 steps) before these runs.

- **Python-command ctests, `-j 1`:** 151 of the 157 registered were run. The
  six left out either execute a GPU binary (`sycl-lifecycle-mutation-M1..M3`,
  `mem-handle-eviction-b70-repeat-clean-exit`, `sycl-lifecycle-gpu-sequential`)
  or are the static-storage audit, which ran on its own (below). Result:
  126 passed and 25 failed.
  - Each of the 25 was re-run against a `git archive` extract of fork master
    `76c7f6548`, and each fails there too, with the same message or the same
    pytest failed/passed counts. They are rot on the fork and were not caused
    by the merge. One of them, `test-sycl-mmid-admission-source`, is
    registered in `tests/CMakeLists.txt` although its script does not exist.
  - The one merge-caused failure, `test-sycl-auto-ubatch-source`, is fixed
    above.
- **`test-sycl-static-storage-audit`:** passed, 1 test OK in 689 s at load
  average ~80. Under ctest it times out, because its registered TIMEOUT is
  300 s. The script was therefore run directly.

## Review follow-ups (review-n77l-merge, 0 Critical / 0 Important / 5 Minor)

Three minors are fixed in the follow-up commit. The other two are filed:
M2 (the split buft ignores `main_device`) as llama.cpp-oajm, and M5 (the
`prec_set_src` F32-src1 route) as llama.cpp-pj53. The SYCL `mmap_support = 0`
question is llama.cpp-5efe.

- **M1, `src/llama-moe-profile.cpp`.** `flush()` read each captured
  expert-id tensor at the profile-wide `n_expert_used`, which is now the
  maximum over layers. A narrower layer was read past its own ids, and at the
  wrong offset from its second token onward. `llama_moe_profile::update()`
  now takes the row width, and `flush()` passes the tensor's `ne[0]`.
  `test-moe-profile-stride` is host-only; its tensors sit in a CPU buffer over
  a static array. It captures a full-width layer, then a half-width one, and
  failed before the fix with layer 1's selections at 8 instead of 4, four of
  them layer 0's stale expert 7.
- **M3, `src/llama-mmap.cpp`.** `MADV_HUGEPAGE` now covers only
  `ranges_complement(lazy_ranges)`, with each range shrunk to whole pages.
  Before, a THP fault in a lazy range read a whole 2 MiB folio, despite
  `POSIX_MADV_RANDOM`. `test-mmap-lazy-hugepage` reads the per-VMA flags from
  `/proc/self/smaps`. It failed before the fix with `hg` on the lazy VMA
  (`rd mr me ms rr sd hg`), including the pages holding the range's
  unaligned ends.
- **M4, safety net.** The guard now also counts a test labelled `model` as a
  loader. That covers `test-model-load-cancel`, `test-autorelease` and
  `test-backend-sampler`, which load `LLAMACPP_TEST_MODELFILE`.
  `sycl-lifecycle-gpu-sequential` now carries `model` too. The documented
  sweep's label exclusion is `-LE 'residency|mem-handle|cache|model'`, and the
  guard ties the whole `-LE ... -E ...` string to CLAUDE.md. The driver adds
  a mock RED for a model-labelled test and a mock RED for an unreadable sweep
  listing, plus a real-build RED: drop `model` from the `-LE`, and the guard
  names all four tests.
