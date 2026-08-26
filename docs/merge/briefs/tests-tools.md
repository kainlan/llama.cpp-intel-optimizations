# Wave-5 conflict brief: tests + tools

Pre-merge analysis of every both-touched `tests/`/`tools/`/`AGENTS.md` file, plus
`docs/backend/SYCL.md` (a genuine wave-5 merge conflict outside that glob, added
per the T12 spec). Base for both diffs is `81ff7abe5` (fork/upstream merge-base);
fork side is `master`, upstream side is tag `b10630`.

## Derivation

```bash
comm -12 <(git diff --name-only 81ff7abe5 master | LC_ALL=C sort) \
         <(git diff --name-only 81ff7abe5 b10630 | LC_ALL=C sort) \
  | grep -E '^tests/|^tools/|^AGENTS.md'
```

Raw output (18 files); `tests/CMakeLists.txt`, `tools/CMakeLists.txt` and
`tools/ui/CMakeLists.txt` are excluded — already covered in
`docs/merge/briefs/build-system.md`. `docs/backend/SYCL.md` is added per the
spec (wave-5 conflict, outside the glob). **16 entries below.**

**Shapes verified against the real merge, not inferred from diff hunks.**
Every "conflicts" / "auto-merges" claim in this brief was checked with
`git merge-tree --write-tree master b10630` (a read-only simulation of the
exact merge — no working tree touched) and, for every file this brief calls a
real conflict, by extracting the actual `<<<<<<</=======/>>>>>>>` markers from
the write-tree's blob (`git show <tree>:<path>`) rather than reasoning about
line-range proximity from separate two-way diffs. Confirmed conflict set for
this file group, matching the lead's ground truth exactly: `AGENTS.md`,
`docs/backend/SYCL.md`, `tests/get-model.cpp` (modify/delete),
`tests/snapshots/qwen3.5-27b.schema`→`qwen3.6-27b.schema` (rename/delete),
`tests/test-chat.cpp`, `tests/test-llama-archs.cpp`,
`tests/test-model-load-cancel.cpp`, `tests/test-quant-type-selection.cpp`,
`tools/llama-bench/llama-bench.cpp`, `tools/server/server-queue.cpp`,
`tools/ui/embed.cpp`. Confirmed clean **auto-merge** (git resolves with no
conflict markers at all): `tests/test-alloc.cpp`, `tests/test-backend-ops.cpp`,
`tests/test-gguf.cpp`, `tools/cli/README.md`, `tools/completion/README.md`,
`tools/server/README.md`.

```
AGENTS.md
tests/get-model.cpp
tests/test-alloc.cpp
tests/test-backend-ops.cpp
tests/test-chat.cpp
tests/test-gguf.cpp
tests/test-llama-archs.cpp
tests/test-model-load-cancel.cpp
tests/test-quant-type-selection.cpp
tests/snapshots/qwen3.6-27b.schema   (folded into test-quant-type-selection.cpp below)
tools/cli/README.md
tools/completion/README.md
tools/llama-bench/llama-bench.cpp
tools/server/README.md
tools/server/server-queue.cpp
tools/ui/embed.cpp
docs/backend/SYCL.md
```

---

## `AGENTS.md`

**Confirmed via `git merge-tree`: real conflict, 6 conflict blocks** — given
the fork's rewrite replaced the whole prose body upstream edited piecemeal,
this needed hand-review per block rather than a single-decision resolution;
the strategy below applies uniformly across all 6.

**Fork intent:** wholesale rewrite (2026-07-25) turning the file from a
duplicate of build/architecture/verification content into a thin pointer doc
into `CLAUDE.md` ("CLAUDE.md is the source of truth... this file holds only the
material that is not there"). Adds fork-only sections that have no upstream
analog: Intel Arc GPU memory architecture (GGTT/PPGTT), a VTune profiling
recipe, "Professional Engineering Standards" (Spinach Rule), and a "Landing the
Plane" session-completion checklist (codescout tracker, no `bd sync`). Deletes
essentially the entire original AI-usage-policy prose body.

**Upstream intent:** in-place edits to the *same* AI-usage-policy prose the fork
deleted: rewords the top banner ("AI-generated code is allowed... you are 100%
responsible"), expands Permitted/Prohibited AI usage bullets, adds a hard
*CRITICAL* non-overridable rule that only `ggml-gh-bot` may post PR
comments/replies, adds a "Common mistakes AI agents make" list (no Minja — there
is a fork-analogous fact about `common/jinja`; don't add new `tests/*` files
without maintainer approval), a good/bad comment-style example pair, and a
pointer to a new `skills/` directory.

**Interaction:** total overlap — fork replaced the whole file upstream edited
piecemeal. Every upstream hunk lands inside text the fork's diff deletes wholesale.

**RESOLVE:** take the fork's rewrite as the *document*, but this is not a pure
"ours wins" resolution — three pieces of upstream content are new policy that
did not exist in the pre-fork text and must be carried into the fork's
replacement doc (or confirmed already covered) rather than silently dropped:
1. The `ggml-gh-bot` exception to the no-auto-comment rule — CLAUDE.md/AGENTS.md
   fork rewrite has no equivalent statement; this is a maintainer-side policy
   fact, not duplicated build/arch content, so it belongs in the rewritten file
   even under the "CLAUDE.md is authoritative" model. Add a short paragraph.
2. "Do NOT add a new file in `tests/*` without maintainers' approval" — directly
   relevant to this very merge (T12/T22 add no test files, so no violation, but
   future agent-driven work in this repo should carry the rule forward).
3. The `skills/[skill]` directory pointer — check whether upstream actually
   shipped a `skills/` dir at `b10630` (`git show b10630 --stat | grep '^skills/'`
   or `git ls-tree b10630 skills/`) before adding the pointer; do not add a
   dead link if the fork's checkout has no such directory post-merge.

Everything else upstream added (the reworded banner, expanded bullet lists, the
comment-style example) is now-superseded prose that the fork's pointer-doc
model deliberately elides — safe to drop, since it duplicates `CONTRIBUTING.md`
content the fork's model already defers to.

**CONTRACT:** post-merge, `AGENTS.md` must retain the fork's pointer-doc
structure (table into `CLAUDE.md`, GGTT/PPGTT section, Landing-the-Plane
checklist) and must not regain any of the duplicated build/perf/architecture
content the 2026-07-25 rewrite removed (that duplication is the exact defect
the rewrite fixed — see the file's own "why" paragraph, corroborated by
`sycl-perf-baselines.md` history). It must gain the `ggml-gh-bot` exception
verbatim as project policy, independent of which doc style wins.

---

## `tests/get-model.cpp` (UD: upstream **D**eleted, fork **M**odified)

**Fork intent (`4c7860abe`):** `get_model_or_exit()` now calls
`test_skip_no_model()` from the fork-local `tests/test-skip.h` instead of
`fprintf(...); exit(EXIT_SUCCESS);`. `test_skip_no_model()` exits
`LLAMA_TEST_EXIT_SKIP` (77), ctest's `SKIP_RETURN_CODE`. This is the fix for
`llama.cpp-nwip`: exiting `EXIT_SUCCESS` on "no model" made
`test-autorelease`, `test-model-load-cancel` and `test-backend-sampler` pass
vacuously in ~0.18s on every model-less local run.

**Upstream intent (`47f686f53`, "tests: avoid building get-model.cpp many
times", #26317):** deletes `tests/get-model.cpp` and `tests/get-model.h`
entirely, folding the helper into `common/common.cpp`/`common/common.h` as
`common_get_model_or_exit(int, char**)` — a genuine dedup (the same tiny TU was
being compiled once per test binary). **Upstream's version still exits
`EXIT_SUCCESS`** on no model — it does not carry the fork's SKIP-return-code
fix, because that fix does not exist upstream.

Upstream's same commit rewires 5 consumers to call `common_get_model_or_exit`
instead: `test-autorelease.cpp`, `test-backend-sampler.cpp`,
`test-model-load-cancel.cpp` (also in this brief, see below),
`test-quant-type-selection.cpp` (also below), `test-rset-release.cpp`. Only the
latter two are in this brief's both-touched group; the other three
(`test-autorelease.cpp`, `test-backend-sampler.cpp`, `test-rset-release.cpp`)
are **not** touched by the fork in this range, so git will silently take
upstream's version of those three files cleanly — with no merge conflict, but
also with no SKIP-semantics fix, unless this entry's fix is applied.

**RESOLVE:** delete `tests/get-model.cpp`/`tests/get-model.h` (take upstream's
consolidation — it is a real build-time win, one fewer compiled TU per test
binary) but **port the fork's `test_skip_no_model()` call into
`common_get_model_or_exit()`** in `common/common.cpp` before deleting the
fork-local file, i.e. the merged function must exit 77, not `EXIT_SUCCESS`.
`tests/test-skip.h` is fork-local infra (per its own header comment, written
for exactly this "upstream may delete/revert the two call sites" scenario) and
survives the file deletion unaffected — `common/common.cpp` simply becomes its
new (and only) consumer.

This is a **cross-file dependency outside this brief's own scope**:
`common/common.cpp`/`common/common.h` are not in the tests/tools/AGENTS.md
group, so whichever brief/wave owns `common/` must apply this fix — flag it
explicitly to that wave rather than assuming it falls out of the tests-side
resolution. If it is missed, the regression is silent: all 5 rewired test
binaries (3 with no local conflict to force a second look) go back to passing
vacuously with no model configured, exactly as `llama.cpp-nwip` originally found.

**CONTRACT:** grep `common/common.cpp` post-merge for
`common_get_model_or_exit` and confirm its body calls something that exits 77
on the missing-model path (either by including `tests/test-skip.h` from
`common/common.cpp`, which is a layering smell worth a second look since
`test-skip.h` is a `tests/`-scoped header, or by inlining the same
fprintf+`exit(77)` there directly with a comment citing `llama.cpp-nwip`).
Then grep all 5 rewired consumers for `common_get_model_or_exit(` and confirm
none still references the deleted `get-model.h`.

---

## `tests/snapshots/qwen3.6-27b.schema` (DU: fork **D**eleted the ancestor path, upstream renamed+kept)

**What actually happened (verified, not as originally framed):** the fork
deleted **two** snapshot goldens outright —
`tests/snapshots/qwen3.5-27b.schema` (`91faa064e`) and
`tests/snapshots/qwen3.5-397b-a17b.schema` (`3842a3417`) — both via
`llama.cpp-pp72` cluster C5, because bartowski re-quantized both HF repos to add
an MTP/NextN layer, shifting `block_count` and breaking the committed golden.
Upstream's commit `47f686f53` **renames** (`R100`, byte-identical content)
`qwen3.5-27b.schema` → `qwen3.6-27b.schema` — it does **not** touch
`qwen3.5-397b-a17b.schema` at all (upstream leaves that one alone; only the
fork deletes it).

So there are two distinct dispositions, not one:
- `qwen3.5-397b-a17b.schema`: fork deletes, upstream doesn't touch → clean
  auto-delete, **no conflict**, nothing to resolve.
- `qwen3.5-27b.schema` → `qwen3.6-27b.schema`: fork deletes the pre-image,
  upstream renames it → **real DU/rename-vs-delete conflict**, surfacing as an
  unresolved add of `tests/snapshots/qwen3.6-27b.schema`.

**RESOLVE:** take upstream's file — **adopt** `qwen3.6-27b.schema`, do not
honor the fork's deletion for this one. See the `test-quant-type-selection.cpp`
entry below: upstream's `47f686f53` also swaps the `model_specs[]` row from
`bartowski/Qwen_Qwen3.5-27B-GGUF` to `ggml-org/Qwen3.6-27B-GGUF` — the exact
"swap with ggml-org if/when it's released" the fork's own dropped-row comment
was waiting for. Since the rename is content-identical (`R100`), the new
`ggml-org/Qwen3.6-27B-GGUF` model produces a structurally identical golden to
the old, pre-drift `Qwen3.5-27B` — i.e. upstream's fix and the fork's TODO
resolve to the same golden, just reached from a different (now-fixed) source
repo. No `llama.cpp-mcv8` revision-pin plumbing is needed for this row.

**CONTRACT:** `tests/snapshots/qwen3.6-27b.schema` exists post-merge with
upstream's (unchanged) content; `tests/snapshots/qwen3.5-27b.schema` and
`tests/snapshots/qwen3.5-397b-a17b.schema` do **not** exist post-merge.

---

## `tests/test-model-load-cancel.cpp`

**Fork intent:** `auto params = llama_model_params{};` →
`auto params = llama_model_default_params();` — the C++17 aggregate-init fix
([[aggregate-init-forges-defaulted-private-ctors]] class of bug: `Type{}`
value-initializes every field including ones a real constructor would leave at
a safe default, bypassing intended access control/invariants).

**Upstream intent:** two changes to the same small file, from the same
`47f686f53` commit discussed above: (1) `#include "get-model.h"` →
`#include "common.h"` and `get_model_or_exit(...)` →
`common_get_model_or_exit(...)`; (2) `params.use_mmap = false;` →
`params.load_mode = LLAMA_LOAD_MODE_NONE;` — because `include/llama.h`
(upstream, `b10630`) **removed** the `use_mmap`/`use_direct_io`/`use_mlock`
booleans from `llama_model_params` entirely, replacing them with a single
`enum llama_load_mode load_mode` field. `use_mmap` does not exist as a struct
member upstream at all — this is not optional to apply, the field is gone.

**Interaction:** confirmed via `git merge-tree` — one conflict block, verified
content:
```cpp
    llama_backend_init();
<<<<<<< master
    auto params = llama_model_default_params();
    params.use_mmap = false;
=======
    auto params = llama_model_params{};
    params.load_mode = LLAMA_LOAD_MODE_NONE;
>>>>>>> b10630
```
The `#include`/`get_model_or_exit` change auto-merged cleanly above this block
(no separate conflict marker) — the fork's diff never touched that line in
this file, so git took upstream's `common_get_model_or_exit(...)`
automatically; that call site's SKIP semantics are then resolved identically
to the `get-model.cpp` entry above (once that fix lands upstream in
`common/common.cpp`, this file needs no additional change). The
`llama_model_params{}` / `use_mmap` conflict needs BOTH sides' fixes
combined — neither git-offered side alone compiles cleanly against the
post-merge `llama.h`: taking `master`'s side loses upstream's field rename
(`use_mmap` doesn't exist as a struct member anymore, so `params.use_mmap = false;`
fails to compile); taking `b10630`'s side reintroduces the aggregate-init
hazard the fork fixed.

**RESOLVE:**
```cpp
auto params = llama_model_default_params();  // fork's aggregate-init fix
params.load_mode = LLAMA_LOAD_MODE_NONE;      // upstream's use_mmap -> load_mode rename
```
plus upstream's `#include "common.h"` / `common_get_model_or_exit(...)` (with
the SKIP-77 fix ported per the `get-model.cpp` entry).

**CONTRACT:** file must not reference `get-model.h`, `use_mmap`, or
`llama_model_params{}` post-merge; must reference `common_get_model_or_exit`,
`llama_model_default_params()`, and `LLAMA_LOAD_MODE_NONE`. Build is the
strongest gate here — `params.use_mmap` simply fails to compile if this is
missed, so a green build already proves the rename was applied; it does not by
itself prove the aggregate-init fix or the SKIP-77 fix survived, so grep those
two independently.

---

## `tests/test-quant-type-selection.cpp`

**Fork intent (`91faa064e` + `3842a3417`, `llama.cpp-pp72` C5):** drops both
`bartowski/Qwen_Qwen3.5-{27B,397B-A17B}-GGUF` rows from `model_specs[]`
(upstream re-quantized both repos with an added MTP/NextN layer, shifting
`block_count` and invalidating the committed goldens — see the schema entry
above for the mechanism). Replaces the two rows with a long comment explaining
both drifts, why regeneration was rejected (mock never sets `n_layer_nextn`),
and the restoration path (`llama.cpp-mcv8`): 27B is pinnable to a specific
revision that reproduces the old golden; 397B has no revision that ever
matched and needs a different source entirely.

**Upstream intent (`47f686f53`):** independently reaches the same "both rows
are broken" conclusion via its own route: comments out (does not delete) the
397B row with the original `// TODO: swap with ggml-org if/when it's released`
still attached, and — critically — **replaces** the 27B row with
`{ "ggml-org/Qwen3.6-27B-GGUF", "Q8_0" }`, i.e. the ggml-org release the fork's
TODO was waiting for actually shipped upstream. Also renames
`ggml-org/Nemotron-Nano-3-30B-A3B-GGUF` →
`ggml-org/NVIDIA-Nemotron-Nano-3-30B-A3B-GGUF` (org rename, unrelated to the
Qwen drift) and re-aligns column padding for the whole array (cosmetic, driven
by the longer new strings).

**Interaction:** confirmed via `git merge-tree` — exactly **one** conflict
block in the whole file, spanning the entire `model_specs[]` array literal
(both sides realigned every row's padding, so the conflict engulfs lines that
didn't semantically change). Same underlying defect, not adversarial:
upstream's fix for the 27B row is strictly better than the fork's "wait for a
replacement" comment, since the replacement now exists. Confirmed compatible:
the renamed schema is `R100` (byte-identical to the old `qwen3.5-27b.schema`),
so ggml-org's re-release reproduces the exact structural golden the drifted
bartowski repo used to. This file has **no** `get_model_or_exit`/`get-model.h`
reference at all (verified — grep returns nothing on either side), so it is
not coupled to the `get-model.cpp` entry's SKIP-77 concern the way
`test-model-load-cancel.cpp` is.

**RESOLVE:** take upstream's row set (the `ggml-org/Qwen3.6-27B-GGUF` swap,
the Nemotron-Nano rename, the realignment) rather than the fork's static
comment block — it resolves the fork's own tracked TODO. For the 397B row,
prefer upstream's **comment-out** over the fork's outright deletion: keep the
row present-but-disabled (matching upstream's shape exactly, since there is no
divergent fork-only content to preserve there) and append a one-line pointer
comment citing `llama.cpp-mcv8` for whoever eventually restores it from a real
source, so the ticket doesn't have to re-locate this spot from scratch.
Discard the fork's long explanatory comment block — its purpose (explaining
why the rows were dropped) is superseded by the 27B row no longer needing an
explanation and the 397B row keeping upstream's own TODO inline.

**CONTRACT:** `model_specs[]` contains `ggml-org/Qwen3.6-27B-GGUF` with
quant `Q8_0` and **not** `bartowski/Qwen_Qwen3.5-27B-GGUF`; the 397B row is
present but commented out with a `llama.cpp-mcv8` pointer; the Nemotron-Nano
row reads `ggml-org/NVIDIA-Nemotron-Nano-3-30B-A3B-GGUF`.

---

## `tests/test-alloc.cpp`

**Fork intent:** appends two new tests at end-of-file:
`test_backend_tensor_alloc_sets_data` (verifies `ggml_backend_tensor_alloc`
leaves a valid, in-bounds `data` pointer and round-trips a `tensor_set`/`_get`)
and `test_probe_max_alloc_size` (exercises `ggml_backend_probe_max_alloc_size`
against a synthetic buffer type with a hard allocation ceiling, including the
0-margin-factor and zero-limit edge cases).

**Upstream intent:** pure include-path cleanup —
`<ggml-alloc.h>`/`<ggml-backend-impl.h>`/`<ggml-impl.h>` (angle-bracket) become
`"ggml-alloc.h"`/`"../ggml/src/ggml-backend-impl.h"`/`"../ggml/src/ggml-impl.h"`
(quoted, with explicit relative paths for the two internal headers). No logic
touched.

**Interaction:** disjoint — upstream's hunk is the top-of-file include block;
fork's hunks are two new functions appended near the end plus two new `run(...)`
registrations in `main()`. No textual overlap.

**RESOLVE: confirmed AUTO-MERGE** — `git merge-tree --write-tree master b10630`
produces zero conflict markers for this file (no `Auto-merging ... CONFLICT`
line in the merge-tree output). Nothing to resolve by hand; both sides land
automatically.

**CONTRACT:** file compiles with the quoted/relative includes AND both new
`run("test_backend_tensor_alloc_sets_data", ...)` /
`run("test_probe_max_alloc_size", ...)` lines present in `main()`.

---

## `tests/test-gguf.cpp`

**Fork intent:** defensive null-device guard in the roundtrip-test loop —
caches `ggml_backend_dev_count()` into a local and adds
`if (!dev) { continue; }` after `ggml_backend_dev_get(i)`, before either
`test_roundtrip` call. Same defensive pattern recurs in
`tools/llama-bench/llama-bench.cpp` (5 call sites) — a fork-wide idiom against
a backend-registry slot that can apparently return null (see that entry for
detail).

**Upstream intent:** substantial new fixture coverage, all in the
`handcrafted_file_type` enum/dispatch machinery near the top third of the
file — adds `HANDCRAFTED_KV_EMPTY_KEY` (empty-string key),
`HANDCRAFTED_KV_WRONG_TYPE_ALIGN` (declares `general.alignment` with a
non-`UINT32` type; loader must reject cleanly, not assert), and
`HANDCRAFTED_TENSORS_ZERO_DIM` (a tensor dimension of 0; loader must load
without crashing, geometry check is skipped for this case specifically). Also
strengthens `handcrafted_check_tensors` to compare `gguf_get_tensor_ne(...)`
against the expected shape, not just the type.

**Interaction:** disjoint — fork's edit is at `main()`'s device-iteration loop
(near end of file); upstream's edits are all in the handcrafted-fixture
generator/checker (first ~700 lines). No overlap.

**RESOLVE: confirmed AUTO-MERGE** (`git merge-tree` produces zero conflict
markers for this file). Both sides land automatically; nothing to resolve by
hand.

**CONTRACT:** `main()` retains the cached `backend_dev_count` local and the
`if (!dev) continue;` guard; the three new `HANDCRAFTED_*` enum values, their
`handcrafted_file_type_name()` cases, and their generator/checker branches are
all present.

---

## `tests/test-chat.cpp`

By far the largest size mismatch in this group: fork +20/-0 lines in one
narrow spot; upstream +1303/-133 across the whole file (mostly new
`test_template_output_peg_parsers` blocks — DeepSeek-V4 reasoning-effort tests,
new template fixtures, PEG-parser-output assertions). **Confirmed via
`git merge-tree`: exactly three conflict blocks**, all small, everything else
in this enormous diff auto-merges cleanly (including the shared
`bool enable_thinking = true;` field itself — see below).

**Conflict 1 — include block, verified content:**
```
 #include <iostream>
<<<<<<< master
 #include <map>
 #include <nlohmann/json.hpp>
=======
 #include "json.h"
>>>>>>> b10630
 #include <set>
```
The `using json = common_json;` line immediately below is **not** part of the
conflict — it auto-merged cleanly to upstream's value, because the fork's diff
never touched that line (only upstream changed it, from
`nlohmann::ordered_json`). That settles the choice: the file now uses
`common_json`, which `"json.h"` defines — `<nlohmann/json.hpp>` is stale
against that typedef and must not be kept. **RESOLVE:** take
`#include "json.h"` (upstream) **and** `#include <map>` (fork, still needed —
see conflict 2) as two separate include lines; drop
`#include <nlohmann/json.hpp>` entirely.

**Conflict 2 & 3 — `test_template_generation_prompt`'s local `opts` struct,
verified content:**
```
        bool                         enable_thinking        = true;
<<<<<<< master
        std::map<std::string, std::string> chat_template_kwargs;
=======
>>>>>>> b10630
    };
```
and, in the `check` lambda:
```
        inputs.enable_thinking        = opts.enable_thinking;
<<<<<<< master
        inputs.chat_template_kwargs   = opts.chat_template_kwargs;
=======
>>>>>>> b10630
```
Confirms the prediction exactly: `enable_thinking` was added identically by
both sides and needed no resolution at all (git merged it silently); the only
real conflict is that upstream's side of each block is **empty** — fork's
`chat_template_kwargs` field and its wiring line are pure additions with
nothing to reconcile against. The real `common_chat_templates_inputs` struct
already carries a `chat_template_kwargs` member upstream (used directly a few
hundred lines later in the new DeepSeek-V4 test block), so this local `opts`
mirror needs no additional plumbing to compile. **RESOLVE:** take both lines
(both sides of each `<<<<<<<`/`>>>>>>>` collapse to "keep the master half,
discard the empty b10630 half").

Fork's ~15-line gpt-oss-120b `no_thinking` check block, appended immediately
after these two hunks in the source diff, produced **no separate conflict
marker** — it auto-merged cleanly ahead of upstream's DeepSeek-V4 block and
every other new PEG-parser test upstream added.

**CONTRACT:** the local `opts` struct in `test_template_generation_prompt`
carries both `enable_thinking` and `chat_template_kwargs`; the gpt-oss-120b
`no_thinking` check block (asserting
`"<|start|>assistant<|channel|>final<|message|>"`) is present; `#include "json.h"`
and `using json = common_json;` are present (not `nlohmann::ordered_json`
directly) alongside the fork's `#include <map>`.

---

## `tests/test-backend-ops.cpp`

Fork +148/-17 across 10 hunks (all in `test_rms_norm_mul_rope`, `test_argmax`,
`test_top_k`, and two small `make_test_cases_eval`/`main` additions); upstream
+1019/-149 across ~90 hunks spanning nearly the whole file (new op structs —
`test_snake_fuse`, expanded `test_ssm_scan`, new `test_diag`; new dtype
coverage on existing ops — `test_set_rows`, `test_cpy`, `test_rope`,
`test_conv_2d`/`test_conv_2d_dw` gaining an F16 variant, `test_concat`,
`test_roll`, `test_flash_attn_ext`; type-array expansion — `all_types`,
`base_types`, `other_types` gain `GGML_TYPE_Q2_0` and `GGML_TYPE_TQ2_0`).

**RESOLVE: confirmed AUTO-MERGE.** `git merge-tree --write-tree master b10630`
produces `Auto-merging tests/test-backend-ops.cpp` with **no** subsequent
`CONFLICT` line — zero conflict markers anywhere in the merged blob, despite
~100 hunks across the two diffs. (Hunk-range arithmetic corroborates why: every
fork hunk — `test_rms_norm_mul_rope` 2659-2704, `test_argmax`'s new -inf/
sentinel cases, `test_top_k`'s new fewer-than-k cases 5671-5781, the two
`make_test_cases_eval` registration blocks at 8227 and 9078, and the `main()`
edit at 10190 — sits in a gap between upstream's hunks, one case merely
adjacent at 8227/8233 with no actual line overlap.) **This entry is therefore
not a merge-resolution task** — nothing needs picking between sides. It is a
**semantic review + T22 prediction task**: confirm both sides' additions
survive the auto-merge as expected, and predict where upstream's new coverage
will land in T22's backend-ops scoring.

**What each side added, for the semantic review:** fork's additions are new
op-correctness edge cases — `test_argmax` with an all/partial `-inf` row
(exercises the sentinel-vs-index scan order: a backend that seeds its scan
with `(-inf, -1)` and admits on `>` alone never takes a `-inf` column and ships
the sentinel instead) and `test_top_k` with fewer than `k` finite values.
Upstream's additions are broad new op/dtype coverage — `test_snake_fuse`,
expanded `test_ssm_scan`, `test_diag`, F16 variants of `test_conv_2d_dw`, and
dtype-array expansion (`all_types`/`base_types`/`other_types` gain
`GGML_TYPE_Q2_0` and `GGML_TYPE_TQ2_0`). This is squarely the "upstream's new
op coverage is generally WANTED" case from the acceptance criteria — nothing
here should be dropped, and the auto-merge means nothing here needs to be.

**c-wps7 partition prediction (for T22):** the plan's three enumerated
fail-closed classes and their mechanism (SYCL `ggml_backend_sycl_device_supports_op`
unconditionally claims `MUL_MAT_ID`, but the internal MMVQ capability table
defaults **false** for anything not explicitly listed — so an unsupported type
doesn't decline/fall back to CPU, it silently computes wrong numbers that
`test-backend-ops` then reports as a numeric mismatch, not a graceful skip):

1. **float MMID** (`llama.cpp-0yi9`) — F16/F32/BF16 `MUL_MAT_ID`, ~80+73+3
   cases, because MMVQ dispatches through a quantized-activation path
   (`vec_dot_*_q8_1`) that has no meaning for float weights.
2. **iq\* family** (`llama.cpp-wh7o`) — all 9 IQ quant types' `MUL_MAT_ID`,
   ~97 cases, because each IQ type's non-indexed kernel is bespoke (different
   template arity, adjusted `qi` divisors, grid-table decode) and none is
   transcribable into the generic `_id` template the other 8 quantized types
   share.
3. **q1_0/nvfp4 oracle gate** — deliberate fail-closed policy (`docs/plans/2026-08-14-sycl-b70-hardening-merge.md` Task 8, `llama.cpp-0bot`/c-cvpx): Q1_0/NVFP4 `MUL_MAT_ID` requires an FP16 converter
   (`to_fp16_sycl`) that historically didn't exist; restoration status must be
   re-checked at merge time against current HEAD (the 2026-08-14 handoff doc
   states "No Q1_0/NVFP4 capability enable until current-HEAD B70 route...
   certification is green" — confirm whether that gate has since flipped).

**The new prediction this brief adds:** upstream's `GGML_TYPE_Q2_0` and
`GGML_TYPE_TQ2_0` additions to `all_types`/`base_types`/`other_types` feed
`make_test_cases_eval`'s `MUL_MAT_ID` generator the same way every other type
in those arrays does. **Neither type appears anywhere in
`ggml/src/ggml-sycl/moe-mmvq-tables.hpp` or `mmvq.cpp`** (verified by direct
grep — the only SYCL-side hits for `GGML_TYPE_Q2_0`/`GGML_TYPE_TQ2_0` are in
`cpu-traits-support.cpp`'s CPU-reference trait table, unrelated to the SYCL MMID
path). By the same Mechanism-A logic that produces the float and iq* classes,
new `Q2_0`/`TQ2_0` `MUL_MAT_ID` test cases will hit the identical
"unconditional supports_op claim + capability-table miss" wrong-answer failure
— but **Q2_0/TQ2_0 is neither float, nor iq\*, nor q1_0/nvfp4**, so these
failures will land **outside all three enumerated classes** and break T22's
"nothing outside them" certification bar unless dispositioned first. Recommend
either (a) extending the enumerated-classes ticket set with a fourth
`Q2_0/TQ2_0 MMID` class mirroring `llama.cpp-wh7o`'s shape before T22 runs, or
(b) confirming Q2_0/TQ2_0 plain (non-`_id`) `MUL_MAT` already works (it likely
does, via the CPU-fallback path evidenced by `cpu-traits-support.cpp`) and
scoping the new-class ticket to `MUL_MAT_ID` specifically, matching `0yi9`/`wh7o`'s
own scoping.

**CONTRACT:** post-merge `all_types`/`base_types`/`other_types` contain
`GGML_TYPE_Q2_0` and `GGML_TYPE_TQ2_0`; the fork's argmax/top_k edge-case tests
and `run()`/registration lines are present; T22's backend-ops scoring explicitly
checks for a Q2_0/TQ2_0-shaped residual before asserting the three-class
partition is exhaustive.

---

## `tests/test-llama-archs.cpp`

Fork +548/-34 across 22 hunks (spans nearly the file: `nmse`, `set_tensor_data`,
`silent_model_load_progress`, `get_model_and_ctx`, `arch_supported`,
`save_models`, `test_backends`, `main`); upstream +116/-19 across 14 hunks,
concentrated in `get_gguf_ctx` (lines ~99-300, a region the fork's diff does
not touch at all) plus smaller edits to `moe_mandatory`, `arch_supported`,
`save_models`, and `test_backends`.

**Confirmed via `git merge-tree`: exactly ONE conflict block in the entire
file**, despite 22 fork hunks and 14 upstream hunks touching overlapping
functions on paper (`arch_supported`, `save_models`, `test_backends`). Hunk-
range proximity between the two diffs (e.g. fork's `arch_supported` hunk
`@@ -426,7+584,103@@` sitting close to upstream's `@@ -412,16+505,20@@`,
ending at old 428) suggested more risk than the real merge carries — a good
demonstration of why this brief now verifies with `git merge-tree` rather than
inferring from hunk-range arithmetic across two separate two-way diffs. Only
`get_gguf_ctx` (upstream's biggest edit block, old 99-300, zero fork hunks
nearby) and the one conflict below are load-bearing to check by hand; every
other near-miss auto-merges cleanly.

**The one real conflict, verified content (inside `test_backends`'s per-arch
test loop):**
```
                bool skip = !arch_supported(arch) || (dc.split_mode == LLAMA_SPLIT_MODE_TENSOR && dc.devs.empty());
<<<<<<< master
#if defined(GGML_USE_WEBGPU)
                skip = true; // FIXME
#endif // GGML_USE_WEBGPU
#if defined(GGML_USE_SYCL)
                // [llama.cpp-l6wj] Provisional: the Meta (tensor-parallel) config cannot run on a SYCL build.
                // ... (fork's full mechanism comment, citing llama.cpp-zviv/dy1r) ...
                skip = skip || dc.split_mode == LLAMA_SPLIT_MODE_TENSOR;
#endif  // GGML_USE_SYCL
=======
>>>>>>> b10630
                if (!skip) {
```
Upstream's side of the conflict is **empty** — this is a pure fork addition
(the SYCL/WebGPU tensor-parallel skip, load-bearing per its own cited tickets:
`ggml_backend_sycl_device_supports_op()` has no residency record for Meta-
backend weights, so the scheduler mid-layer-splits a TP graph and trips a
`GGML_ASSERT` in `handle_set_rows` — this skip is the correct fix, not a
workaround to relax). **RESOLVE:** take the full `master` side of this one
conflict verbatim; there is nothing on the `b10630` side to reconcile against.

**The following fork content carries the load-bearing exit-77/no-row
semantics from `llama.cpp-4lnb`/`llama.cpp-to9m` (documented in `CLAUDE.md`
under Verification Commands) — confirmed to auto-merge cleanly with no
conflict marker anywhere near it, so no by-hand action is needed, but it is
worth naming explicitly so a reviewer knows what to spot-check post-merge:**
- The `arch_supported`-adjacent logic that makes an excluded/unsupported arch
  (`gemma4`, `gemma4-assistant`, `eagle3`, `dflash`, `gemma-embedding`, BERT
  family, RWKV) **exit 77** with an explicit "this run proves NOTHING" message,
  rather than printing an empty or all-`SKIP` table and exiting 0. This is the
  fix for the "empty table reads as verified" trap CLAUDE.md documents at
  length.
- The `n_measured` counter path that returns **77** when a targeted `-a` run
  measured nothing (distinct from the all-`FAIL`-cells table-corruption bug
  `llama.cpp-to9m` describes — that one is about `grep FAIL` returning zero on
  a genuinely failing run due to log interleaving; this one is about a
  structurally-empty run reporting success).
- Whatever change in `save_models`/`test_backends`/`main` wires the
  roundtrip-mismatch detection that CLAUDE.md's `rc=0` trustworthiness claim
  depends on (`rc=0` is described as trustworthy because `all_ok` is computed
  from in-memory comparisons — verify this computation, not just its call
  site, is fork content and not something upstream's `get_gguf_ctx` edits
  incidentally touch).

**CONTRACT:** post-merge, run
`./build/bin/test-llama-archs -a gemma-embedding; echo "rc=$?"` (or any other
`arch_supported()==false` arch) and confirm `rc=77`, not `rc=0`, and that
stderr/stdout contains an explicit "proves nothing" style message rather than
a silently-empty table. Cross-check against `CLAUDE.md`'s own worked example
before/after this fix landed.

---

## `tools/llama-bench/llama-bench.cpp`

**Fork intent:** adds `if (!dev) { continue; }` (or the `reg`-equivalent) at
**5** call sites that enumerate `ggml_backend_dev_count()`/`ggml_backend_reg_count()`
slots: `get_cpu_info()`, `get_gpu_info()`, the `--list-devices` handler inside
`parse_cmd_params`, the buffer-type enumeration inside `parse_cmd_params`, and
`test::get_backend_bench_info` (or equivalent)'s RPC-backend scan. Same
defensive idiom as the fork's `tests/test-gguf.cpp` entry above — a
backend-registry slot that can return null is apparently a real, recurring
hazard on this host (plausibly related to the iGPU / multi-device enumeration
issues `CLAUDE.md`'s VRAM-budget section documents at length, though the exact
trigger for a *null* slot as opposed to a *misbehaving* one is not established
by this diff alone).

**Correction to the task spec:** the spec described this file's fork content
as "`-v` null-callback behavior + any fork columns" — **neither is present in
this diff.** The `-v`/null-log-callback behavior (`llama_log_set(llama_null_log_callback, ...)`)
and any fork-added bench columns are pre-existing fork content from *before*
`81ff7abe5` (the merge-base), not part of this delta, so they are not at risk
in this merge and need no entry here — they simply aren't touched by either
side in this diff range. Flagging this so T22/reviewers don't go looking for a
conflict that isn't in this file's diff.

**Upstream intent:** one relevant structural change — deletes the entire
inline `--list-devices` handler body (the same loop the fork null-guards) and
replaces it with a single call to a new shared helper,
`common_print_available_devices()` (added in `common/arg.cpp`, declared in
`common/arg.h`). This is a real dedup: the same device-listing loop was
duplicated across tools before this upstream commit.

**Interaction:** confirmed via `git merge-tree` — one conflict block, verified
content:
```cpp
            } else if (arg == "--list-devices") {
<<<<<<< master
                std::vector<ggml_backend_dev_t> devices;
                for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                    auto * dev = ggml_backend_dev_get(i);
                    if (!dev) {
                        continue;
                    }
                    if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                        devices.push_back(dev);
                    }
                }
                printf("Available devices:\n");
                ... (rest of fork's inline loop, unchanged from base) ...
=======
                common_print_available_devices();
>>>>>>> b10630
                exit(0);
```
Fork's null-guard is dead code once upstream's deletion is taken, because the
loop it guards no longer exists in this file.

**RESOLVE:** take upstream's `common_print_available_devices()` call (the
dedup is worth keeping), which means fork's null-guard at *this one* call site
is moot in `llama-bench.cpp` — but **the guard's justification is not moot,
only its location moved.** Checked `common/arg.cpp` at `b10630`:
`common_print_available_devices()`'s own `for (size_t i = 0; i < ggml_backend_dev_count(); ++i) { auto * dev = ggml_backend_dev_get(i); ...}`
loop has **no equivalent null check** — upstream's consolidated helper would
silently reintroduce the exact null-deref hazard the fork's guard existed to
prevent, just in a different file. **Port the fork's `if (!dev) continue;`
into `common_print_available_devices()` in `common/arg.cpp` as part of this
merge**, not into `llama-bench.cpp` (nothing is left there to guard).

This is a **cross-file dependency outside this brief's own scope** —
`common/arg.cpp` is not in the tests/tools/AGENTS.md group (it's presumably
owned by the "core ggml"/common brief, task 10). Flag it explicitly to that
wave: without this port, the other 4 null-guard sites in `llama-bench.cpp`
(`get_cpu_info`, `get_gpu_info`, the buffer-type enumeration, the RPC scan) are
unaffected and stay guarded, but the `--list-devices` path — probably the most
frequently invoked of the five, since it's a diagnostic users run directly —
loses its protection.

**CONTRACT:** `llama-bench.cpp` retains `if (!dev) continue;` guards at the 4
sites upstream didn't touch; `common/arg.cpp`'s `common_print_available_devices()`
gains an equivalent guard (coordinate with the common/ brief — do not assume
it lands automatically).

---

## `tools/server/server-queue.cpp`

**Fork intent:** two additions. (1) `#include <exception>` / `#include <memory>`
after the existing `#include <chrono>`. (2) wraps
`result->update(states[idx])` in `server_response_reader::next(...)` in a
`try { ... } catch (const std::exception & e) { stop(); ...return an
server_task_result_error... }` block — so an exception thrown while updating a
streamed-response's generation state is converted into a clean
`ERROR_TYPE_SERVER` result (with `stop()` cancelling the rest of that reader's
work) instead of propagating uncaught out of the server's task-processing loop.

**Upstream intent:** substantial `server_queue`-internal rework — adds a
`task_resets_idle_timer(server_task_type)` helper (`SERVER_TASK_TYPE_METRICS`
does not reset the idle timer), and large edits to `terminate()`,
`start_loop()`, and `cleanup_pending_task()` (the `@@ -122,10+135,157@@` and
`@@ -138,33+298,22@@` hunks account for most of the +186/-25). Also adds
`#include <algorithm>` and `#include <thread>` to the same include block the
fork touches.

**Interaction:** confirmed via `git merge-tree` — one conflict block, verified
content:
```cpp
 #include <algorithm>
 #include <chrono>
<<<<<<< master
 #include <exception>
 #include <memory>
=======
 #include <thread>
>>>>>>> b10630
```
`#include <algorithm>` auto-merged cleanly above the conflict (fork never
touched that line, so git took upstream's insertion automatically) — only
`<exception>`/`<memory>` (fork) vs `<thread>` (upstream) actually conflict, a
narrower collision than hunk-range proximity alone suggested. The
`result->update(...)` try/catch (fork, old lines ~404-421) produced **no**
conflict marker — confirmed disjoint from every upstream hunk, auto-merges
cleanly at its original location.

**RESOLVE:** union all three conflicting includes — `<exception>`, `<memory>`,
`<thread>` — alongside the already-merged `<algorithm>`/`<chrono>`; take
upstream's `terminate()`/`start_loop()`/`cleanup_pending_task()`/
`task_resets_idle_timer` rework as-is (it auto-merged, no decision needed);
the fork's try/catch around `result->update(...)` needs no action either — it
already survived the auto-merge at its original location.

**CONTRACT:** `server_response_reader::next()` still wraps `result->update(states[idx])`
in the fork's try/catch producing `ERROR_TYPE_SERVER` on exception; all 5
`#include`s present; `task_resets_idle_timer` present and consulted wherever
upstream's rework calls it.

---

## `tools/ui/embed.cpp`

**Fork intent:** generalizes the required-asset check into an
optional-vs-required model. `struct required_check { label; match; found; }`
becomes `struct asset_check { label; match; optional; found; }`, with
`loading.html` marked `optional: true` and every other asset staying required.
Missing optional assets are collected separately, reported as a **warning**
(not a hard failure), and embedding proceeds; missing required assets still
hard-fail as before. The load-bearing comment cites `llama.cpp-fdm1`: the
prebuilt bundles at the `ggml-org/llama-ui` HF bucket stopped shipping
`loading.html` in July 2026, and its only consumer
(`tools/server/server-http.cpp`'s not-ready middleware) already falls back to
a JSON 503 when `llama_ui_find_asset()` returns null — so requiring it was
turning an external, out-of-repo bundle change into a hard local build failure.

**Upstream intent:** the minimal fix for the identical underlying problem —
deletes the `{ "loading.html", exact("loading.html"), false }` line from
`required_check checks[]` outright. No optional-asset infrastructure; `loading.html`
is simply no longer checked for at all (silently, not with a warning).
Separately (disjoint region, ~70 lines later): adds a one-line comment above
the `fnv_hash(...)` call — `// note: this is a simple hash for cache busting,
not a cryptographic hash; fnv is enough here`.

**Interaction:** confirmed via `git merge-tree` — one conflict block, verified
content:
```cpp
<<<<<<< master
        // "optional" assets degrade a feature when absent ... (fork's comment) ...
        struct asset_check { const char * label; match_fn match; bool optional; bool found; };
        asset_check checks[] = {
            { "index.html",           exact("index.html"),           false, false },
            { "loading.html",         exact("loading.html"),         true,  false },
            { "manifest.webmanifest", exact("manifest.webmanifest"), false, false },
            ...
=======
        struct required_check { const char * label; match_fn match; bool found; };
        required_check checks[] = {
            { "index.html",           exact("index.html"),           false },
            { "manifest.webmanifest", exact("manifest.webmanifest"), false },
            ...
>>>>>>> b10630
```
Both sides fix the exact same bug at the exact same array literal (upstream
simply deletes the `loading.html` row; fork keeps it but marks it optional),
but the fork's fix is a strict superset — it achieves upstream's outcome (a
missing `loading.html` no longer hard-fails the build) while also preserving
visibility (a warning line) and building general infrastructure for any
*future* asset that becomes optional, rather than a one-off deletion. The
`fnv_hash` comment (upstream, ~70 lines later) produced no conflict marker —
it auto-merges cleanly, no action needed.

**RESOLVE:** take the fork's `asset_check`/`optional` refactor wholesale — it
strictly dominates upstream's fix (same practical effect, better diagnostics,
more general) — and additionally apply upstream's `fnv_hash` comment at its
disjoint location. Do not take upstream's plain deletion; it would regress the
warning visibility the fork's fix added.

**CONTRACT:** `checks[]` (renamed `asset_check[]`) retains the `optional`
field with `loading.html` marked `true` and every other entry `false`; a
build with `loading.html` present-but-missing prints a "missing optional
asset(s)" warning and still succeeds (rc=0); a build missing a *required*
asset still fails (rc=1) exactly as before. The `fnv_hash` comment is present.

---

## `tools/cli/README.md`, `tools/completion/README.md`, `tools/server/README.md`

**Confirmed via `git merge-tree`: all three AUTO-MERGE cleanly** — zero
conflict markers despite the size mismatch (fork's 3-line edit lands on
different rows than upstream's 14-101-line regeneration in every case). This
entry is therefore not a merge-resolution task at the git level; it is a
**durability/policy finding layered on top of a clean merge** — worth fixing
regardless of this merge, since the risk it describes is about the *next*
regeneration, not this one.

**Generated-file warning, checked and confirmed relevant:** all three carry
the header `<!-- IMPORTANT: The list below is auto-generated by llama-gen-docs;
do NOT modify it manually -->`. **The fork's edit hand-violates this notice.**

**Fork intent (identical 3-line change repeated verbatim in all three files):**
appends `; ignored by SYCL unified cache builds` to the `-fit`/`-fitt`/`-fitc`
flag descriptions in the auto-generated arg table.

**Upstream intent:** each README's diff is much larger (14-101 line changes)
because it reflects a real `llama-gen-docs` regeneration against upstream's
own `common/arg.cpp` changes in this cycle — new/changed flag descriptions
elsewhere in the same tables, unrelated to `--fit`.

**The load-bearing finding:** `common/arg.cpp:2559` (fork/master, current) defines the
`--fit` description via `string_format("whether to adjust unset arguments to
fit in device memory ('on' or 'off', default: '%s')", ...)` — **with no "ignored
by SYCL unified cache builds" clause anywhere in the source string.** The fork's
annotation exists **only** in the three generated `.md` files, never in the
generator's input. This means the annotation is **not durable** — the moment
anyone runs `llama-gen-docs` to regenerate these READMEs (which is exactly
what upstream's own larger diff represents having done), the fork's hand-edit
is silently overwritten with no trace, because the generator has no idea the
annotation should exist.

**RESOLVE:** nothing at the merge level — verified in the merged blob that
the fork's `-fit`/`-fitt`/`-fitc` annotation survives unmolested alongside
every one of upstream's regenerated rows (git auto-merges cleanly because
upstream's changes land on different table rows, not these three). The action
item is entirely about the *next* regeneration, not this one: **fix the root
cause, not just the symptom** — add the "ignored by SYCL unified cache
builds" clause into the `string_format(...)` call in `common/arg.cpp` itself
(conditionally, or unconditionally as a documentation note — the flag is a
compile-time/build-configuration fact, not a runtime one, so an unconditional
string addition is simplest) so that the next `llama-gen-docs` regeneration —
by anyone, for any reason — reproduces the annotation automatically instead of
requiring someone to remember to re-hand-edit three READMEs. This is a
**cross-file dependency outside this brief's scope** (`common/arg.cpp` again,
same as the `llama-bench.cpp` entry above) — flag to whichever wave owns
`common/`.

**CONTRACT:** after the merge and after fixing `common/arg.cpp`, run the
project's `llama-gen-docs` binary (or whatever invokes it — check
`examples/gen-docs/gen-docs.cpp` / `CODEOWNERS:41`) and confirm the
regenerated `tools/{cli,completion,server}/README.md` still contain "ignored
by SYCL unified cache builds" on the `-fit`/`-fitt`/`-fitc` rows **without any
manual README edit** — i.e. the annotation now survives regeneration by
construction. If `common/arg.cpp` isn't fixed in time for this merge, at
minimum manually re-append the 3-line annotation to all three regenerated
READMEs and open a ticket for the root-cause fix, so the drift is at least
visible rather than silently reintroduced on the next `gen-docs` run by
someone unaware of the fork's convention.

---

## `docs/backend/SYCL.md`

**Confirmed via `git merge-tree`: real conflict, 5 conflict blocks.** Scale
mismatch explains why: fork 785/-223 across 28 hunks (a comprehensive
rewrite/reorganization, matching the B70/B50 hardware transition and the
extensive `GGML_SYCL_FA_ONEDNN` correction history `CLAUDE.md` documents);
upstream 55/-5 across 6 hunks (incremental content additions to the pre-fork
document structure). Because the fork's hunks span nearly the entire file,
most of upstream's 6 hunks land inside a region the fork also rewrote,
producing 5 conflict blocks — this needs judgment per topic, not a single
"ours"/"theirs" call.

**Upstream's four content additions, checked against the fork's current
(post-rewrite) document:**

1. **`--mmap` → `--load-mode auto` in two example `llama-completion` invocations
   (Linux + Windows).** Reflects the real `include/llama.h` API change (the
   same `use_mmap` → `load_mode` rename documented in the
   `test-model-load-cancel.cpp` entry above). **Checked: the fork's rewritten
   SYCL.md contains zero occurrences of `--mmap` or `--load-mode`** — the
   fork's rewrite removed these specific example command blocks entirely
   (restructured elsewhere). **No action needed** — there is nothing to
   rename in the fork's current doc. Do, however, flag to whoever resolves
   `common/arg.cpp`/`include/llama.h`: confirm whether the user-facing
   `--mmap`/`--no-mmap` CLI flag itself survives the `load_mode` struct
   rename (it's still present in `common/arg.cpp:2401` pre-merge) — if it's
   renamed or removed, any *other* fork doc referencing `--mmap` (not just
   this file) needs the same check this entry gave SYCL.md.

2. **"User can use the device management in `docs/multi-gpu.md`... `--device
   SYCL0,SYCL1`" paragraph (appears twice — Linux and Windows sections).**
   **Checked: not present in fork's rewritten doc.** This is genuinely new,
   additive content — `--device` is llama.cpp's own CLI-level device selector,
   distinct from the `ONEAPI_DEVICE_SELECTOR` environment variable the fork's
   doc and `CLAUDE.md` document as the primary mechanism. **Port this
   paragraph into the fork's rewrite**, but annotate it with the fork's own
   P2P finding: this system has confirmed no direct P2P between the B70 and
   B50 (`CLAUDE.md` "Patched compute-runtime & P2P topology"), so a bare
   `--device SYCL0,SYCL1` invocation on multi-GPU cards without a shared PCIe
   bridge silently host-bounces rather than running device-to-device — copying
   the upstream paragraph verbatim without that caveat would contradict the
   fork's own documented topology finding.

3. **Env-var table additions:** `GGML_SYCL_DEV2DEV_MEMCPY` gains a third value
   (`2`, Host Forward); new rows `GGML_SYCL_ENABLE_HOST_PINNED_MEM`,
   `GGML_SYCL_FA_ONEDNN`, `GGML_SYCL_FA_ONEDNN_MAX_KV`, `GGML_SYCL_ENABLE_MKL_FA`
   (+ its `_DEBUG`/`_DIAG` diagnostic variants), `GGML_SYCL_ENABLE_FUSION`,
   `GGML_SYCL_ENABLE_ESIMD`. **Checked against the fork's current doc:**
   `GGML_SYCL_FA_ONEDNN` is **already extensively documented** in the fork's
   rewrite (lines ~881-901), including a dedicated correction paragraph about
   the retired `GGML_SYCL_FA_ONEDNN_ALLOW` variable — **upstream's version is
   redundant and thinner; keep the fork's, discard upstream's row for this one
   variable specifically.** The other five (`GGML_SYCL_ENABLE_HOST_PINNED_MEM`,
   `GGML_SYCL_FA_ONEDNN_MAX_KV`, `GGML_SYCL_ENABLE_MKL_FA` + diagnostics,
   `GGML_SYCL_ENABLE_FUSION`, `GGML_SYCL_ENABLE_ESIMD`) have **zero** hits
   anywhere in the fork's current doc — genuinely new. **Naming collision risk
   to flag explicitly:** upstream's `GGML_SYCL_ENABLE_ESIMD` is a *different*
   variable from the fork's own `GGML_SYCL_ESIMD_DEQUANT` (documented in
   `CLAUDE.md`'s "Small-block dequant... belongs on standard SYCL, not ESIMD"
   entry, `docs/backend/sycl-env-vars.md`). Do not conflate the two rows when
   porting — they gate different things (upstream's appears to be a general
   ESIMD-kernel on/off switch; the fork's is a narrow dequant-path opt-in
   retest hatch that CLAUDE.md records as measured 1.9x *slower*). **Whether
   these five variables are worth documenting at all is contingent on the
   separate `ggml-sycl.cpp` merge (not in this brief's scope) actually
   preserving upstream's corresponding source-code additions** — a fork-side
   competing implementation could supersede or delete any of them. Flag this
   contingency to whichever wave resolves `ggml-sycl.cpp`, and don't add doc
   rows for variables that merge resolution ends up deleting.

4. **Two new FAQ entries:** `SYCL_CACHE_PERSISTENT=1` crash advisory (JIT
   cache staleness after binary changes; fix is to clear
   `~/.cache/libsycl_cache/` and `unset`), and "how to use iGPU and dGPU at the
   same time" (`--list-devices`, then `--device SYCL0,SYCL1,SYCLxxx`).
   **Checked: neither present in fork's doc.** Pure additive FAQ content, safe
   to port as-is for the first (a generic JIT-cache warning, no fork-specific
   nuance to contradict). **The second needs the same P2P caveat as item 2** —
   this fork's iGPU handling in particular is documented at length in
   `CLAUDE.md` (the VRAM-budget-for-integrated-GPU defect, `llama.cpp-403s`) —
   copying upstream's "just list devices and pass `--device`" FAQ verbatim
   without that context could lead a reader straight into the iGPU
   VRAM-budget hazard `CLAUDE.md` spends a full section warning against.

**RESOLVE:** keep the fork's rewrite as the document's structure and voice.
Port item 2 (device-management paragraph) and item 4 (both FAQ entries) with
the P2P/iGPU caveats noted above; skip item 1 (nothing to change, fork already
removed the affected examples); for item 3, keep the fork's existing
`GGML_SYCL_FA_ONEDNN` writeup unchanged and only add the five genuinely-new
env-var rows, contingent on confirming each still exists post-`ggml-sycl.cpp`-merge.

**CONTRACT:** post-merge `docs/backend/SYCL.md` contains no bare `--device
SYCL0,SYCL1` or iGPU+dGPU guidance without an adjacent P2P/VRAM-budget caveat
referencing the fork's own findings; contains exactly one `GGML_SYCL_FA_ONEDNN`
writeup (the fork's, not a duplicate upstream row); does not document
`GGML_SYCL_ENABLE_ESIMD` and `GGML_SYCL_ESIMD_DEQUANT` as if they were the same
variable; any of the five new env-var rows ported in are verified to still
exist in the merged `ggml-sycl.cpp` source before being documented as live.

---

## Summary for T22

Cross-file dependencies this brief identified that are **outside its own file
scope** and must be tracked by whoever owns `common/arg.cpp` /
`common/common.cpp` (task 10 / the common-brief wave):
1. `common_get_model_or_exit()` (`common/common.cpp`) must exit 77, not
   `EXIT_SUCCESS`, on a missing model — port the fork's `test_skip_no_model()`
   call before deleting `tests/get-model.cpp`.
2. `common_print_available_devices()` (`common/arg.cpp`) must gain the fork's
   `if (!dev) continue;` null-device guard — `llama-bench.cpp`'s own copy of
   that guard becomes dead code once this file takes upstream's dedup.
3. The `-fit`/`-fitt`/`-fitc` "ignored by SYCL unified cache builds" annotation
   must move from the three hand-edited READMEs into `common/arg.cpp`'s
   `string_format(...)` source string, or it will not survive the next
   `llama-gen-docs` regeneration.

New fail-closed class this brief predicts for T22's backend-ops partition:
**`GGML_TYPE_Q2_0`/`GGML_TYPE_TQ2_0` `MUL_MAT_ID`** — outside all three
enumerated classes (float MMID / iq* family / q1_0-nvfp4 oracle gate) per the
type-array analysis above. Disposition this before T22 asserts the partition
is exhaustive, or the "nothing outside them" bar will be surprised by it.

**Methodology note:** every conflict/auto-merge classification in this brief
was checked against `git merge-tree --write-tree master b10630` and, for real
conflicts, against the actual `<<<<<<</=======/>>>>>>>` markers extracted from
that write-tree's blobs — not inferred from comparing two separate two-way
diffs' hunk ranges. That check corrected two things an earlier hunk-range-only
pass would have gotten wrong: (1) `tests/test-backend-ops.cpp` looks
conflict-prone on paper (~100 hunks across both diffs, one pair adjacent to
within a few lines) but auto-merges with zero markers; (2)
`tests/test-llama-archs.cpp` looks like it has two overlapping regions
(`arch_supported`, `test_backends`) but has exactly one real conflict, in a
third location neither hunk-range guess would have pinpointed as precisely.
