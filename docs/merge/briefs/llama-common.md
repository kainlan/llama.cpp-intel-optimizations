# Conflict brief: src/llama + common

Merge-base: `81ff7abe5`. Upstream target: tag `b10630` (remote `ggml-org`). Fork side: `master`.

**Verification method:** every conflict/clean claim below was checked against the actual merge
algorithm, not inferred from reading the two sides' diffs side by side:

```bash
git merge-tree --write-tree --name-only master b10630
```

This is a read-only simulation of the exact merge the eventual merge task will run — it writes
no ref, touches no tracked state, and needs no checkout. Its `CONFLICT (content): ...` lines
are the ground truth for which files actually conflict; everything else in that output
auto-merges. For this group the result is exactly six conflicts — `common/arg.cpp`,
`common/chat.cpp`, `src/llama-context.cpp`, `src/llama-graph.cpp`,
`src/llama-model-loader.cpp`, `src/llama-sampler.cpp` — matching the lead's census. The other
ten both-touched files auto-merge with no conflict markers, so their entries below are
**semantic-review notes** (what upstream changed that the fork's behavior must still hold
against) rather than hunk-level interleave guidance — interleave guidance is reserved for the
six files git actually can't resolve on its own. For files that auto-merge, I additionally
pulled the real merged blob out of the simulated tree (`git show <tree>:<path>`) wherever a
"clean" result still needed a second look — twice that surfaced a merge that was clean by
git's definition (no `<<<<<<<` markers) but wrong by the codebase's: a missing
`GGML_UNUSED(extra)` in `common/fit.cpp`, and a genuinely broken, uncompilable double-assignment
in `common/chat.cpp` hiding behind a file (`common/chat.h`) that isn't even in this
both-touched group. Both are called out in full below.

## Derivation

```bash
comm -12 <(git diff --name-only 81ff7abe5 master | LC_ALL=C sort) \
         <(git diff --name-only 81ff7abe5 b10630 | LC_ALL=C sort) \
  | grep -E '^src/|^common/' | grep -v CMakeLists
```

16 files (matches the plan's illustrative list, which named the `.cpp` stems and
implicitly covered their headers):

```
common/arg.cpp
common/chat.cpp
common/common.cpp
common/common.h
common/fit.cpp
common/sampling.cpp
src/llama-context.cpp
src/llama-graph.cpp
src/llama-kv-cache.cpp
src/llama-memory-recurrent.cpp
src/llama-model-loader.cpp
src/llama-model-loader.h
src/llama-model.cpp
src/llama-model.h
src/llama-sampler.cpp
src/llama.cpp
```

**Renames/deletes:** `git diff --name-status 81ff7abe5..b10630 -- src/ common/` contains
**zero** `R` or `D` rows (only 24 `A` and 104 `M` in this period) — verified by filtering the
full status list for `^[RD]` and getting no output. Every file in the group above is a plain
modify on both sides; there is no rename disposition to make. (Upstream's 14 additions under
`src/models/*.cpp` are covered below under the CMakeLists interaction, since that's the
acceptance criterion asking about them, not because they are renames.)

**Real merge conflicts in this group** (confirmed both by the lead's census and independently
by `git merge-tree --write-tree --name-only master b10630`, which runs the actual merge
algorithm read-only against no tracked state): `common/arg.cpp`, `common/chat.cpp`,
`src/llama-context.cpp`, `src/llama-graph.cpp`, `src/llama-model-loader.cpp`,
`src/llama-sampler.cpp`. The other 10 files auto-merge with no conflict markers — but "no
conflict markers" is not the same as "safe," and two of the ten (`common/fit.cpp`,
`common/chat.cpp`'s sibling header `common/chat.h`... see below) hide real problems that a
textual 3-way merge cannot see. Those are called out explicitly per-file.

## `src/CMakeLists.txt:9` GLOB interaction

```
file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")
```

Confirmed identical on both sides (`git show master:src/CMakeLists.txt` and
`git show b10630:src/CMakeLists.txt` both have this exact line at line 9). The fork's `master`
made **zero** changes under `src/models/` (`git diff 81ff7abe5 master --name-status --
src/models/` is empty). Upstream added 14 new files there (`bailingmoe3.cpp`, `clip.cpp`,
`dots3note.cpp`, `granite-swa.cpp`, `granite-switch.cpp`, `hy-v3.cpp`, `kimi-k3.cpp`,
`laguna.cpp`, `minimax-01.cpp`, `minimax-m3.cpp`, `muse-glimmer.cpp`, `nanbeige.cpp`,
`pockettts.cpp`, `qwen3tts.cpp`). Since the GLOB pattern is unchanged and the fork never
carved out its own `models/*.cpp` files that could collide by name, these 14 land through the
existing glob with no CMakeLists edit needed and no fork-name collision risk. RESOLVE: no
action — this is a non-issue for Task 11, confirmed rather than assumed.

## The four fork behaviors — traced to their protecting hunks

**(a) `fit_params=false` under SYCL.** Declared in `common/common.h:476` inside
`#ifdef GGML_USE_SYCL ... #else ... #endif` (commit implicit in the fork's `common.h` diff —
one hunk, 4 lines). The disabled-fitter body is `common/fit.cpp:795` inside
`common_fit_params()`, same `#ifdef` pattern: the SYCL arm short-circuits to
`COMMON_PARAMS_FIT_STATUS_FAILURE` with a `LOG_WRN` ("unified cache owns memory placement")
and `GGML_UNUSED()`s every parameter; the non-SYCL arm keeps the original body untouched. Both
hunks merge with **no conflict markers** — but `common/fit.cpp` is the one "clean" file in this
group that is not actually safe; see its own entry below for why.

**(b) `llama-tensor-class.*` call sites in `llama-model-loader.cpp`.** This premise does not
hold today, and I want to flag the correction plainly rather than build the trace on a false
floor. `src/llama-tensor-class.{h,cpp}` is a fork-only file (added by `1bb021aed`/`992c9acd5`,
tracker `llama.cpp-i7hhs`/`llama.cpp-g1f9q`; absent from both `81ff7abe5` and `b10630`).
Checked three ways for callers outside its own file: `find_references` on
`llama_tensor_classify` and `llama_tensor_priority_for` returns only the function's own
recursive call and `tests/test-tensor-class.cpp`; `search_text` for `llama-tensor-class.h`
across the whole repo returns exactly two includes (its own `.cpp` and the test); a literal
`grep -n "class\|priority_for\|tensor_class"` over `src/llama-model-loader.cpp` returns only
an unrelated `class GKV` declaration. It is compiled (`src/CMakeLists.txt:39` lists it
explicitly, not just via glob) but has **zero production call sites** anywhere in `src/` or
`common/` — staged infrastructure with test coverage, awaiting a consumer that has not landed.
**CONTRACT: none currently** — there is nothing to interleave with upstream because there is
nothing calling it. If a task in this merge wave (or a later one) wires it into
`llama-model-loader.cpp`, re-check this file's `RESOLVE` for whatever upstream did to
`create_tensor`/`check_tensor_dims` in the meantime (see the model-loader.cpp entry below —
upstream changed `check_tensor_dims`'s signature and added `TENSOR_ALLOW_RESHAPE`).

**(c) `llama-moe-profile.*` plumbing call sites.** Same shape as (b), checked the same way.
`src/llama-moe-profile.{h,cpp}` defines `llama_moe_profile` and `llama_moe_profiler` (added by
`71075d62e`, fixed onto the build by `1fb188c15`). `search_text` for `moe_profile` / grep for
`llama_moe_profiler` across the whole repo (excluding its own file) returns nothing in `src/`
or `common/` — no member of either struct is instantiated, updated, or queried anywhere.
`docs/plans/2026-02-09-unified-memory-system.md` describes an intended integration
(`moe_profile_path`/`moe_warmup_tokens`/`moe_gpu_fraction` in `common_params`, plus
`llama_load_moe_profile`/`llama_analyze_moe_profile`/`llama_save_moe_profile` C-API functions
"in `llama-context.cpp`") but none of those params or functions exist in the current tree —
that doc describes a design that was not (or not yet, or no longer) implemented. **CONTRACT:
none currently**, same reasoning as (b): zero call sites means zero interleave risk today.
Flagging this as a correction to the task's premise, not a gap in the search — three
independent lookups (symbol references, repo-wide text search, direct grep) agree.

**(d) `common/arg.cpp` metadata-only parsing guarantees.** This one is real, current, and has
two live conflicts. Commits `8cf79e8e9` ("common: keep metadata commands backend-free") and
`51d116467` ("backend: make enumeration callers raced-null safe") are the protecting hunks:
- `8cf79e8e9` removed the `if (llama_supports_rpc())` guard around `--rpc` registration and
  removed five `if (!llama_supports_gpu_offload()) fprintf(...)` warning blocks (n-gpu-layers,
  split-mode, tensor-split, main-gpu, gpu-layers-draft) — the point being that
  `common_params_parser_init()` runs on **every** invocation including `--help`/`--version`,
  so calling `llama_supports_rpc()`/`llama_supports_gpu_offload()` there (which probe the
  backend registry) is itself a violation of "metadata commands stay backend-free," independent
  of what the guarded code does.
- `51d116467` added `if (!dev) continue;` after every `ggml_backend_dev_get(i)` in a loop,
  because concurrent backend registration can return null mid-enumeration. Two sites in
  `arg.cpp` (`parse_tensor_buffer_overrides`'s `buft_list` loop, and the `--list-devices`
  handler's device-collection loop) plus one in `common/common.cpp`
  (`common_params_print_info`'s device-info loop) and matching sites in `src/llama.cpp`
  (`llama_prepare_model_devices`, twice, and `llama_print_system_info`'s backend_reg loop —
  same pattern, `ggml_backend_reg_get(i)` this time).
- `test-arg-parser` is the gate named in CLAUDE.md for `common_params_parse()` staying
  metadata-only after any parser change; re-run it after resolving the two conflicts below.

### `common/arg.cpp` — RESOLVE / CONTRACT

Two conflict hunks (verified against the actual `git merge-tree` output, not just the diffs):

**Hunk 1 — `--rpc` registration**, both sides edit the same `if (llama_supports_rpc())` line:
```
<<<<<<< master
    add_opt(common_arg({"--rpc"}, ...));                       // unconditional
=======
    if (params.is_gen_docs || llama_supports_rpc()) {
        add_opt(common_arg({"--rpc"}, ...));
    }
>>>>>>> b10630
```
Upstream independently arrived at a similar goal (`is_gen_docs` lets doc generation see the
flag without a real RPC backend) but its condition still *calls* `llama_supports_rpc()` in the
non-gen-docs case — exactly the call `8cf79e8e9` was written to eliminate.
**RESOLVE: take master's side (always register, no guard).** It's a strict superset of what
upstream's `is_gen_docs` branch achieves and it satisfies the fork's actual constraint (zero
backend probing during parser construction), which upstream's version does not.
**CONTRACT: none** — `--rpc`'s registration becomes unconditional either way; no downstream
consumer needs to detect "is RPC available" through argument presence.

**Hunk 2 — `--list-devices` body**, upstream extracted the fork-touched inline loop into a
new shared function:
```
<<<<<<< master
            ggml_backend_load_all();
            std::vector<ggml_backend_dev_t> devices;
            for (...) {
                auto * dev = ggml_backend_dev_get(i);
                if (!dev) { continue; }                 // <- 51d116467's fix
                if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) devices.push_back(dev);
            }
            printf("Available devices:\n");
            for (auto * dev : devices) { ... printf(...) ... }
=======
            common_print_available_devices();
>>>>>>> b10630
```
I pulled upstream's new `common_print_available_devices()` (`b10630:common/arg.cpp:1138`) and
it reimplements the **same loop without the null check** — upstream never had `51d116467`'s
fix, so refactoring lost nothing on its side, but a naive "take upstream" resolution silently
**reintroduces the null-deref the fork fixed**, since the fix lived in exactly the block
upstream deleted and moved. **RESOLVE: take upstream's refactor (call the shared function —
it's also used by other new call sites, not shown here, so keeping the inline duplicate would
diverge from upstream's structure for no reason), but port `if (!dev) continue;` into the body
of `common_print_available_devices()` before its `if (ggml_backend_dev_type(dev) != ...)`
check.** **CONTRACT: `common_print_available_devices()` is a new upstream-introduced function
signature (`void common_print_available_devices();`, declared wherever upstream put the
`common_arg.h`/`common.h` prototype) that the fork must patch at its one definition site, not
at a call site** — there's only one definition, so this is a single, precisely located edit.

**Unrelated non-conflicting hunks in this file, confirmed clean by diffing upstream's
`arg.cpp` against the fork's five gpu-offload-warning removals:** upstream's diff has **zero**
hits for `llama_supports_gpu_offload`, `n_gpu_layers`, `split_mode`, `tensor_split`, or
`main_gpu` — those five hunks are untouched by upstream and merge automatically. Also
unrelated: upstream's `--mlock`/`--mmap`/`--direct-io` → `--load-mode` deprecation sits
immediately after the `--rpc` hunk in the file but does not overlap it (see
`src/llama-model-loader.cpp` below for the load-bearing half of that same upstream feature).

## `common/common.h` — RESOLVE / CONTRACT

Clean per `merge-tree` (no conflict markers), but the two sides land in close proximity inside
`struct common_params` and are worth confirming rather than assuming apart. Fork's hunk
(base line ~473-479) touches only the `fit_params` declaration, wrapping it in
`#ifdef GGML_USE_SYCL`. Upstream's neighboring hunks are `@@ -456,6 +447,7@@` (adds
`n_outputs_max_per_seq` well above, near line 447) and `@@ -481,6 +473,7@@` (adds
`enum llama_load_mode load_mode = LLAMA_LOAD_MODE_AUTO;` right after `fit_params_target`,
a few lines *below* `fit_params`). The line ranges don't overlap, so git resolves this without
a conflict, and I confirmed the actual merged struct layout keeps the fork's
`#ifdef GGML_USE_SYCL / #else / #endif` block around `fit_params` intact with upstream's new
`load_mode` field landing cleanly after it. **RESOLVE: trust the auto-merge; just re-read the
resulting struct once to confirm the ifdef block wasn't visually mangled by a formatter (it
wasn't, verified) before moving on.** **CONTRACT: `common_params` gains
`enum llama_load_mode load_mode` and `int32_t n_outputs_max_per_seq` — both purely additive,
no fork consumer references either name today, so nothing to update.**

## `common/common.cpp` — RESOLVE / CONTRACT

Clean per `merge-tree`. Fork's only hunk is the same `51d116467` null-check pattern
(`common_params_print_info`'s device-info loop, `if (!dev) continue;`). Upstream's diff over
this file is much larger (225 insertions) but does not touch this loop.
**RESOLVE: trust the auto-merge.** **CONTRACT: none.**

## `common/fit.cpp` — RESOLVE / CONTRACT (the load-bearing one that looks clean but isn't-quite)

Clean per `merge-tree` — no conflict markers — but this is the file where "clean" needs the
closest scrutiny in the whole group, because both sides edit the *same function*
(`common_fit_params`) without their line ranges technically overlapping. Fork's hunk wraps the
whole function body: `#ifdef GGML_USE_SYCL <early-return, GGML_UNUSED every param> #else
<original body, untouched> #endif`. Upstream's hunk (`@@ -794,11 +883,12 @@`, landing squarely
inside that same function) adds a new parameter to the signature,
`const common_fit_extra_model * extra`, and threads it into the call to
`common_params_fit_impl(...)`.

I pulled the actual merged blob from the `merge-tree` output tree rather than trusting the
diffs in isolation, and the auto-merge **did** thread `extra` into both the signature and the
`#else`-branch call correctly:
```cpp
enum common_params_fit_status common_fit_params(
        const char * path_model, llama_model_params * mparams, llama_context_params * cparams,
        float * tensor_split, llama_model_tensor_buft_override * tensor_buft_overrides,
        size_t * margins, uint32_t n_ctx_min,
        const common_fit_extra_model * extra,          // <- upstream's new param, placed correctly
        ggml_log_level log_level) {
#ifdef GGML_USE_SYCL
    GGML_UNUSED(path_model);
    ... (existing GGML_UNUSED list, seven of them) ...
    GGML_UNUSED(log_level);
    // <- extra is NOT GGML_UNUSED'd here
    LOG_WRN(...); return COMMON_PARAMS_FIT_STATUS_FAILURE;
#else
    ...
    common_params_fit_impl(path_model, mparams, cparams, tensor_split, tensor_buft_overrides,
                            margins, n_ctx_min, extra, log_level);   // <- extra threaded correctly
    ...
#endif
}
```
**RESOLVE: the structural merge is correct and needs no hand-editing of control flow — but add
`GGML_UNUSED(extra);` to the SYCL arm's `GGML_UNUSED` list.** Without it, a SYCL build compiles
a function with an unused parameter that every other parameter in the same list is explicitly
suppressed for; whether that's a hard error depends on `-Wunused-parameter -Werror` being in
the SYCL build's flags (not confirmed either way here — treat it as a real risk to close during
the merge, not a "maybe" to defer). **CONTRACT: `common_fit_params()`'s public signature
changes (new trailing-before-log_level parameter, `common_fit_extra_model *`) — its only
caller in this fork-visible scope is `common_memory_breakdown_print` in the same file (unaffected,
different function), so the signature change has no other fork call site to chase within this
group.** Also note the fork's small unrelated addition to `common_memory_breakdown_print`
(`if (ctx == nullptr) return;`, added right after the `#endif` closing this function) sits
outside both conflict ranges and merges without any interaction.

## `common/sampling.cpp` — RESOLVE / CONTRACT

Clean per `merge-tree`. Fork's diff is a self-contained ~116-line addition: a
`LLAMA_LOGITS_TRACE`-gated diagnostic block (env-var-controlled top-N logit dump), the
`common/` counterpart to the same facility added in `src/llama-sampler.cpp` (see below) — same
env vars (`LLAMA_LOGITS_TRACE`, `_LIMIT`, `_TOPN`), same helper shapes
(`common_sampler_logits_trace_enabled/limit/topn/next_sample/piece`), independently named per
file. Upstream's diff to this file (68 lines) doesn't touch this region.
**RESOLVE: trust the auto-merge.** **CONTRACT: none** — purely additive, no shared symbol with
upstream's changes here.

## `src/llama.cpp` — RESOLVE / CONTRACT

Clean per `merge-tree`, and worth confirming precisely because it's the file where upstream's
`--load-mode` feature actually lands its call-site update. Fork's diff is three more
null-check sites of the `51d116467` pattern (introduced here by `1f15d0e3f`, confirmed via
`git log -S 'if (!dev) {' -- src/llama.cpp`, which lands `1f15d0e3f` ahead of `51d116467` on
the same day) — `llama_prepare_model_devices`'s two device-enumeration loops,
`llama_print_system_info`'s backend-registry loop — none overlapping upstream's changes.
Upstream's diff is unrelated in the same file: it (1) adds `llama_load_mode_name` /
`llama_load_mode_from_str` and `llama_version()`, (2) changes the iGPU dedup condition in
`llama_prepare_model_devices` (an `if (igpus.empty())` becomes
`if (igpus.empty() || ggml_backend_dev_backend_reg(dev) == ...)`, referencing PR 23897 — this
sits in the *same function* as one of the fork's null-check loops but a different `case` arm,
confirmed non-overlapping by `merge-tree`), and (3) **updates the `llama_model_loader`
constructor call site**: `params.use_mmap, params.use_direct_io, params.check_tensors,
params.no_alloc, ...` becomes `params.load_mode, params.check_tensors, params.no_alloc,
params.load_mtp, ...`. Since the fork never touched this exact call, upstream's version wins
outright with no interleave needed. **RESOLVE: trust the auto-merge.** **CONTRACT: this is the
call site that makes `src/llama-model-loader.cpp`'s conflict (below) non-optional to resolve
correctly — `llama.cpp` compiling against the new `load_mode`/`load_mtp` constructor signature
means the `.cpp` definition MUST match, so get that entry right or this file's clean merge
becomes a compile error one file over.**

## `src/llama-model-loader.h` — RESOLVE / CONTRACT

Clean per `merge-tree`, and this is the header half of the model-loader constructor-signature
story. Fork's hunk (base line ~83) adds one member: `std::vector<std::string> file_paths;`
plus its explanatory comment (weight-cache identity keying) and the `<string>`/`<vector>`
includes it needs. Upstream's hunks (base lines ~67, ~80, ~128, ~178) add
`TENSOR_ALLOW_RESHAPE`, a `bool load_mtp` member, change the constructor signature from
`bool use_mmap, bool use_direct_io` to `llama_load_mode load_mode` plus a new trailing
`bool load_mtp`, change `check_tensor_dims`'s signature to take a new `bool allow_reshape`
parameter, remove `create_tensor_as_view`, and add `void unmap_weight(...)`. None of these
line ranges overlap the fork's single insertion point, so git resolves it cleanly — I pulled
the actual merged header and confirmed `file_paths` lands correctly between `files` and
`ftype`, with `load_mtp` and the constructor's new signature both present and undisturbed.
**RESOLVE: trust the auto-merge.** **CONTRACT: the constructor declaration here is now
`llama_load_mode load_mode` (not `use_mmap`/`use_direct_io`), `bool load_mtp` added, and
`check_tensor_dims` grew an `allow_reshape` parameter — every one of these is consumed by the
`.cpp` definition below, which is the file actually in conflict, so resolve that entry using
this signature as the target, not the reverse.** Also: `create_tensor_as_view` is removed by
upstream — confirmed via `find_references`/grep that the fork never calls it (fork's own diff
to this header doesn't reference it and no fork-added `.cpp` code in this group calls it), so
the removal is a pure no-op for the fork.

## `src/llama-model-loader.cpp` — RESOLVE / CONTRACT (deep entry)

One conflict hunk, at the constructor body:
```
<<<<<<< master
        if (use_mmap && use_direct_io) {
            if (files.back()->has_direct_io()) {
                LLAMA_LOG_WARN("%s: direct I/O is enabled, disabling mmap\n", __func__);
                use_mmap = false;
            } else {
                LLAMA_LOG_WARN("%s: direct I/O is not available, using mmap\n", __func__);
                use_direct_io = false;
                // reopen file using std::fopen for mmap
                // Replaces the same index with the same fname, so file_paths
                // stays correct and must NOT be touched -- see the parallel
                // array invariant on file_paths in llama-model-loader.h.
                files.pop_back();
                files.emplace_back(new llama_file(fname.c_str(), "rb", false));
            }
        }
=======
>>>>>>> b10630
```
The block on master's side (`if (use_mmap && use_direct_io) {...}`) is **not** something the
fork added — it's pre-merge-base code that the fork left alone except for adding the
`file_paths` bookkeeping comment inside it. Upstream's side is empty because upstream deleted
this whole runtime reconciliation as part of the same `--load-mode` refactor traced above: its
constructor now takes `llama_load_mode load_mode` directly and *derives* `use_mmap`/
`use_direct_io` from it statically —
```cpp
this->use_mmap      = load_mode == LLAMA_LOAD_MODE_MMAP || load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK || load_mode == LLAMA_LOAD_MODE_AUTO;
this->use_direct_io = load_mode == LLAMA_LOAD_MODE_DIRECT_IO;
```
— with no runtime `has_direct_io()` capability probe or reopen-for-mmap fallback anywhere in
its version of the constructor (confirmed: grepped upstream's whole diff for `use_mmap`,
`use_direct_io`, `load_mode`, `has_direct_io` — the reopen path is gone, not moved).
**RESOLVE: take upstream's side (the block disappears) and adopt upstream's constructor
signature/derivation wholesale** — this is upstream's own designed behavior change (static
enum-driven mode selection replacing a runtime capability probe), not something the fork needs
to preserve; the fork's only stake in this exact hunk was a comment, not logic. Concretely:
(1) change the constructor parameter list to match `llama-model-loader.h`'s new declaration
(`llama_load_mode load_mode`, trailing `bool load_mtp`); (2) keep the fork's
`file_paths.emplace_back(fname)` call, which sits immediately before this hunk and is
untouched by either side's conflict — it must survive regardless of which arm of this
conflict wins; (3) drop the fork's "must NOT be touched" comment along with the block it
was inside — the reopen-for-mmap code path it was warning about no longer exists, so the
warning has nothing left to protect. **CONTRACT: this is the file that must satisfy
`src/llama.cpp`'s already-upstream-updated call site (`params.load_mode, ...,
params.load_mtp, ...`) and `src/llama-model-loader.h`'s already-clean-merged declaration —
get this one conflict resolution wrong and it's the one compile error connecting three files
in this group.** Separately, `check_tensor_dims` gained an `allow_reshape` parameter per the
header (confirmed the `.cpp` definition is in this same file, outside the conflict range, so
it inherits upstream's version automatically) — any *caller* of `check_tensor_dims` inside
this file gets upstream's version too (not fork-modified), so no additional interleave there.

## `src/llama-graph.cpp` — RESOLVE / CONTRACT

One small conflict, an add/add at the top-of-file include block:
```
<<<<<<< master
#ifdef GGML_USE_SYCL
#    include "ggml-sycl.h"
#endif
=======
#include "llama-sampler.h"
>>>>>>> b10630
```
**RESOLVE: keep both includes** (order doesn't matter — neither depends on the other). Fork's
only other hunk in this file (confirmed via full diff) is inside `build_attn_mha`: a
`#ifdef GGML_USE_SYCL` block casting `q` to `GGML_SYCL_FATTN_Q_TYPE` before the K/V-type-cast
logic already there. I checked upstream's diff for any hunk whose `@@` context names
`build_attn_mha` and found none — upstream's nearby changes are to `build_attn` (the caller,
one level up), not `build_attn_mha`'s body, so the fork's Q-cast survives untouched.
**CONTRACT: none** — `llama-sampler.h` is a new include upstream needs for something later in
the file (not investigated further, out of scope for this conflict), and the fork's SYCL
attention-path cast doesn't reference anything from it.

## `src/llama-context.cpp` — RESOLVE / CONTRACT (deep entry, largest conflict block)

One conflict, ~270 lines, but it resolves to a simple shape once you see what's actually
different: **pure add/add at one insertion point, two structurally unrelated blocks.** Master's
side is the fork's entire "owner-targeted SYCL execution-context teardown" apparatus —
`llama_context_sycl_hooks_enabled/reg_from_dev/dev_is_sycl/backend_is_sycl`,
`llama_context_sycl_device_index`, `llama_context_has_sycl_backend`,
`llama_context_sycl_attach_sched_plan`, the `llama_context_sycl_exec_hooks` struct (with its
"a SYCL DSO older than llama.cpp-o6jx exports none of these" comment), two overloads of
`llama_context_sycl_exec_procs` (static-link and `GGML_BACKEND_DL` dlopen variants),
`llama_context_sycl_exec_drain_and_close` (the mandatory quiesce → ticket → drain → extract →
release → finish sequence, referencing `llama.cpp-o6jx` and `llama.cpp-34hr`), and the
`GGML_BACKEND_DL`-only proc-address resolution block
(`llama_context_sycl_dl_compute_hooks`, `llama_context_sycl_compute_procs`,
`llama_context_sycl_runtime_proc`). Upstream's side is four small `static const
llm_fused_op_probe` constants (`llm_fused_op_lid_probe`, three `dsv4_hc_*_probe` variants) —
new entries in the same probe-table pattern as the three probes already present just above the
conflict (`llm_fused_op_flash_attn_probe`, `_gdn_ar_probe`, `_gdn_ch_probe`), which neither
side's diff touches. **RESOLVE: concatenate both blocks in full — no line inside either one
references anything in the other, so there is nothing to interleave, only to place.** Natural
ordering: keep upstream's four probe constants immediately after the three existing ones
(same declaration family, contiguous), then the fork's entire SYCL exec-hooks apparatus after
that (or before — genuinely order-independent, confirmed no cross-reference either direction).
**CONTRACT: none from this hunk itself** — but flagging for the implementer: this block is the
single largest concentration of fork-only SYCL context-lifecycle code in the whole group, so
after resolving it, a scan for any *other* upstream hunk later in the same file that might
call the pre-existing `llm_fused_op_*_probe` table (to see if the four new probes need a
corresponding dispatch-site addition beyond their declaration) is worth a follow-up check —
not done here since it's outside a `src/llama-context.cpp`-only, `src/common/`-scoped brief's
budget, but the four new probes existing without any use nearby would be worth noticing during
implementation.

## `src/llama-sampler.cpp` — RESOLVE / CONTRACT

One conflict, ~120 lines, same shape as `llama-context.cpp`: pure add/add, unrelated content.
Master's side is the `LLAMA_LOGITS_TRACE` diagnostic facility's `src/llama-sampler.cpp`
counterpart (`llama_sampler_logits_trace_enabled/limit/topn/next_sample/piece/dump` — mirrors
`common/sampling.cpp`'s version above, independently named, same env vars). Upstream's side is
one small new public API: `llama_sampler_backend_n_nodes(const llama_sampler *)`, returning
`chain->n_nodes` after asserting the sampler is an initialized chain. **RESOLVE: concatenate
both — no shared symbol, no cross-reference.** **CONTRACT: `llama_sampler_backend_n_nodes` is
a new public API surface (declared presumably in `llama.h` or `llama-sampler.h`, neither in
this both-touched group, so not chased further here) — if either header is fork-modified
elsewhere it should already have been caught by a different conflict-brief task; nothing in
this group's diff touches it.**

## The one finding that should gate this whole brief: `common/chat.cpp` × `common/chat.h`

`common/chat.cpp` has one conflict region (an add/add, same shape as the two above: master
adds a small `common_chat_extra_context_true(...)` boolean-coercion helper used once inside
`common_chat_params_init_gpt_oss`; upstream adds an entirely new
`common_chat_params_init_qwen3_coder` function, ~200 lines, that did not exist at the
merge-base at all). That part concatenates cleanly, same reasoning as the two entries above.

**But `common/chat.h` is upstream-only-touched — the fork made zero changes to it — so it is
correctly excluded from this both-touched group by the derivation command, and that is exactly
what makes the next part dangerous: git will silently take upstream's `chat.h` wholesale.**
Upstream renamed a field project-wide:
```
b10630:common/chat.h:277   std::vector<std::string> thinking_end_tags;   // was: std::string thinking_end_tag
```
Confirmed via direct diff of the two sides' `chat.h`. Fork's `chat.cpp` — because it long
predates this rename — still assigns the old singular field name at nine call sites across the
file (checked with a literal grep for `thinking_end_tag\b` against master's current
`chat.cpp`). **Eight of those nine sit outside the fork's 42-line diff to this file**, so on
merge they simply inherit upstream's already-renamed lines wholesale (git takes upstream's
version wherever the fork made no edit) — those eight are fine, no action needed, verified by
the fact `merge-tree` reports them within its one already-identified conflict-free resolution
for the rest of the file.

**The ninth is the load-bearing one, and it sits exactly at the fork's own edit inside
`common_chat_params_init_gpt_oss` — the GPT-OSS Harmony-format thinking-tag assignment that
CLAUDE.md's canonical GPT-OSS chat-correctness gate depends on.** I pulled the *actual* merged
blob from the `merge-tree` result (not just the two diffs side by side) and it auto-merges
**silently, with no conflict marker, into code that will not compile and would be semantically
wrong even if it did**:
```cpp
    data.thinking_start_tag = "<|start|>assistant";                       // master's value
    data.thinking_end_tag   = "<|channel|>final<|message|>";              // master's value, OLD FIELD NAME

    data.thinking_start_tag = "<|channel|>analysis<|message|>";           // upstream's value -- silently overwrites the line above
    data.thinking_end_tags  = {"<|end|>"};                                // upstream's value, NEW FIELD NAME
```
Both sides' hunks touch the *same insertion point* (right after `data.supports_thinking =
true;`) but git's diff3 did not detect them as overlapping — likely because the two hunks'
surrounding context lines didn't align byte-for-byte (a blank-line or comment difference is
enough) even though they are semantically the same edit site. The result, if committed
as-is: (1) `data.thinking_end_tag` no longer exists as a member once `chat.h` is upstream's
version — **this is a compile error**, not a silent bug; (2) even fixing just the compile
error by deleting that one line leaves `data.thinking_start_tag` assigned twice in sequence,
so the fork's intended value (`"<|start|>assistant"`) is silently discarded in favor of
upstream's (`"<|channel|>analysis<|message|>"`) — whichever assignment textually comes second
wins, with no error, warning, or trace of the discarded one.

**RESOLVE: this needs a human/functional call, not a mechanical merge.** These two pairs
of values encode different semantic intents for where GPT-OSS "thinking" starts and ends —
fork's values bracket "from the assistant preamble to the final channel," upstream's bracket
"the analysis channel content, ending at the model's own `<|end|>` token." Simply keeping one
side's assignment discards the other's design intent without anyone deciding that on purpose.
Concretely: (1) delete the stray `data.thinking_end_tag = ...` line (old field, would not
compile); (2) an implementer with a running B70/B50 must decide which start/end pair is
correct against the model's actual Harmony-format output — not guess from the diff — and (3)
**re-run the canonical GPT-OSS chat-correctness gate from CLAUDE.md
(`ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-cli -m
/models/gpt-oss-20b-mxfp4.gguf ... --reasoning-format none --reasoning-budget 0 ...`,
expecting the clean `1, 2, 3, 4, 5` digit sequence) after resolving this, specifically because
a wrong thinking-tag boundary would show up as garbled or channel-leaked output on exactly this
gate.** This is a build+GPU verification step outside this brief's no-build/no-GPU scope — it
belongs to whichever task actually performs and lands the merge, but it must not be skipped
because the merge tool will tell you nothing is wrong. **CONTRACT: `common_chat_params`'s
`thinking_end_tag` (string) → `thinking_end_tags` (vector<string>) rename is the load-bearing
upstream interface change for this whole file** — every other one of the nine call sites
already resolves correctly through the ordinary auto-merge; this is the only one that needs
hand intervention, and it's the one that happens to sit under the fork's GPT-OSS support.

## `src/llama-kv-cache.cpp`, `src/llama-memory-recurrent.cpp`, `src/llama-model.cpp`,
`src/llama-model.h` — lighter entries (clean, lower risk)

All four are clean per `merge-tree`. `llama-kv-cache.cpp`'s fork diff is a self-contained SYCL
KV-layer-mask handoff apparatus (`llama_kv_cache_sycl_hooks`,
`llama_kv_cache_sycl_mask_handoff` — the destructor-based single-cancel guard referencing
`llama.cpp-y36c`) that upstream's much smaller 22-line diff never approaches.
`llama-memory-recurrent.cpp`'s fork diff (26 lines) and upstream's (41 lines, mostly deletions)
don't share a hunk. `llama-model.cpp` carries the largest fork diff in this group by line count
(561 insertions, dominated by one +448-line hunk — SYCL early-layer-assignment planning,
`llama_model_sycl_compute_early_plan`/`llama_model_sycl_set_late_inventory` and their call
sites inside `load_tensors`) against upstream's 472-line diff to the same file; despite the
size on both sides, `merge-tree` found no overlapping hunk. `llama-model.h`'s fork diff is
small (17 lines) and clean against upstream's 53-line diff. **RESOLVE: for all four, trust the
auto-merge.** **CONTRACT: none identified** — I did not find a fork call site in these four
files referencing anything from the four fork behaviors above, and the two moe-profile/
tensor-class behaviors have no call sites anywhere per the (b)/(c) findings, so there is
nothing further to trace into `llama-model.cpp` specifically despite it being the natural
future home for a moe-profile consumer. Given the size of the clean merges here (especially
`llama-model.cpp`'s ~1000 combined changed lines), a full-file re-read post-merge is cheap
insurance even though no conflict marker will flag anything.
