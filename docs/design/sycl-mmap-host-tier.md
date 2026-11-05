# SYCL file-mapped host tier (mmap as a unified-cache tier)

*Status: DESIGN rev 3aa, not implemented. rev 3aa is a new commit on rev 3z
(`9c0ee6941`). design-5efe-r21 passed rev 3z with two nits, folded here: the
single-slot overload's signature gate is the name plus exactly one `&`, and
`auto & vec` is at `pp:1057`. rev 3z is a new commit on rev 3y
(`12839f154`). It folds design-5efe-r20 (`00ffc08fb..12839f154`,
0C/0I/2m/4n): the lease-drop no-allocation ban now covers `free_chunk_owner`
and the accessor, the gate's scanned-body list names every frame it bans,
and the single-slot core overload is selected by signature (the vector
overload is exempt); clause 10's allow-list is per mention (body, ordinal,
statement kind) with exact counts, plus a presence check that each decision
body calls `footprint_locked()`; the vector declaration is the `vector<`
token; the ordinal counts every mention; and `comment mention flagged` is a
negative control. Appendix B's "design-5efe-r20" table keys each item. rev 3y
is a new commit on rev 3x
(`00ffc08fb`). It folds design-5efe-r19 (`02d766337..00ffc08fb`,
0C/0I/2m/4n) under §M110: clause 7's no-allocation list covers every
frame on the lease-drop path (including the single-slot core overload
and `release_zone_chunk`) with a planted control in the overload; clause
10 allow-lists the destructor's reset (`pp:163`), scans comment-stripped
source and asserts that the nine cited log lines are the only exempt
mentions; and the nits (clause 5 names the public form's inline slot,
Appendix B's reserve-position rows are marked as moved, and clause 11
gains a control for the explicit `bad_alloc` catch). Appendix B's
"design-5efe-r19" table keys each item. rev 3x is a new commit on rev 3w
(`02d766337`). It folds design-5efe-r18 (`0ee6ad375..02d766337`,
0C/0I/3m/6n) under §M104: the lease-drop path is non-allocating (the
public release holds one inline carrier slot, never a vector) and its
callers and the blocking free are stated; `footprint_locked()` has its
own gate control and clause 10's log-only exemption is a mechanical
form; the stranded RED scores the WARN text the code prints; and the
six nits (sync/async sides, `pp:661`, the carrier count's last-statement
decrement, "equals nine" with the abort string in the design, GONE and
the §M97 cite, the lease-drop lookup scope and a status-order RED line).
Appendix B's "design-5efe-r18" table keys each item. rev 3w is a new
commit on rev 3v
(`0ee6ad375`). It folds design-5efe-r17 (`858b74433..0ee6ad375`,
0C/1I/5m/5n) under the lead's r17 rulings: every decision read of the
pool's footprint goes through one `footprint_locked()` helper that
reconciles first; the stranded reserve is made once, in `grow_into`,
before the chunk is added, and the prune is in place; the lease's last
drop releases an orphaned chunk (event-driven, no begin needed); the
no-outstanding-carrier assertion is placed and counted; and the gate,
the REDs and the nits follow. Appendix B's "design-5efe-r17" table keys
each item. rev 3v is a new commit on rev 3u
(`8cc6013e4`) that applies the lead's stranded-bound ruling: the stranded
list is reserved to the pool's chunk count wherever a chunk is added
(rev 3w: once, in `grow_into`, before the add), so
the carrier's append never allocates (no terminate path), the destructor
asserts capacity before it, and gate clause 11 pins the reserve; Appendix
B's "The stranded list's bound" table keys it. rev 3u is a new commit on
rev 3t
(`ce6226443`, which is `ab79c0f0d` plus the follow-up that moved
`test-sycl-scratch-trial-return` to `level_zero:1`). It folds
design-5efe-r16 (`858b74433..ce6226443`, 0C/1I/6m/6n) under §M92, which
corrects §M88 (1): a registry-refused free is a scheduled retry, not a
leak, so the carrier strands its footprint on a pool list and the
registry is the one source that reconciles it (the footprint leaves when
the memory does); the drop path is stated against canon 1612-1619 and the
gate pins the collect-then-release rule; the chunk's `alloc_metadata`
capture and the carrier's back-pointer are named; the status is mapped
per caller, with a one-call owner-end release; the registry accessor and
the reconcile have their own gate clauses and REDs; and the REDs' cleanup,
recipes and entry checks are fixed. Appendix B's "design-5efe-r16" table
keys each item. rev 3t is a new commit on rev 3s
(`8c61488b1`). It applies §M88 (the lead's answers to rev 3s's questions):
the footprint leaves `total_allocated_` only when the allocation is
actually gone, decided by a read-only registry query by the allocation's
stable identity, and a refused free keeps it with the ownerless-lease
WARN and the strict abort (superseded in rev 3u: §M92 makes it a stranded
entry reconciled from the registry, with a retry WARN and no abort); and
`test-sycl-configure-refusal` pins `level_zero:1`. Appendix B's
"§M88 (rev 3t)" table keys each item. rev 3s is a new commit on rev 3r
(`858b74433`). It folds design-5efe-r15 (`4db80fa62..858b74433`,
0C/1I/5m/4n) under the lead's r15 rulings (the release carrier is RAII,
so a dropped carrier releases and accounts; `release_zone_chunk` returns
a status; every caller that holds the marked set's lock uses the `_locked`
core; the lock-order gate pins the call forms, the destructor and
`deferred_frees_mutex_`; the orphaned-free claims are stated at the
callers; the strict-abort children inherit the driver's environment; and
the footprint seam's instant and build are named); Appendix B's
"design-5efe-r15" table keys each item to r15's IDs. rev 3r is a new
commit on rev 3q (`4db80fa62`). It folds design-5efe-r14 (`02ba5ed6a..4db80fa62`,
0C/1I/8m/2n) under the lead's r14 rulings (the held-lock claims derived
from the callers, the release carrier and its two forms, the footprint
that leaves with the memory, the census row, one planted control per
gate clause, and the strict-abort test's construction); Appendix B's
"design-5efe-r14" table keys each item to r14's IDs. rev 3q is a new
commit on rev 3p (`02ba5ed6a`). It folds design-5efe-r13 (`40110fd66..02ba5ed6a`,
0C/0I/5m/4n) under §M81 (c)-(e) (the marked set's lock census and order,
its insert side, and the strict-abort arm's own test); Appendix B's
"design-5efe-r13" table keys each item to r13's IDs. rev 3p is a new
commit on rev 3o (`40110fd66`). It folds design-5efe-r12 (`714081770..40110fd66`,
0C/0I/6m/4n) under the lead's r12 orders (where the kept executables are
freed, the refusal's plumbing, the buildable stale-vectors case, mutants
that print before they abort, and the marked set's lock); Appendix B's
"design-5efe-r12" table keys each item to r12's IDs. rev 3o is a new
commit on rev 3n (`714081770`). It folds design-5efe-r11
(`6b674097e..714081770`, 0C/0I/7m/3n) under the lead's r11 orders (an
entry guard on the record function, a fourth case pinned to the
moe_after site, split RETURN_TO_MARK mutants, a marked-set entry that
leaves with its chunk, and `configure_zones` refusing over a withdrawn
chunk) and the §M69 (b) amendments 1-4; Appendix B's "design-5efe-r11"
table keys each item to r11's IDs. rev 3n is a new commit on rev 3m
(`6b674097e`). It folds design-5efe-r10 (`a68508606..6b674097e`,
0C/1I/5m/8n) under the lead's r10 rulings, including the §M69 (b)
fourth amendment (submitted executables outlive the throw); Appendix B's
"design-5efe-r10" table keys each item to r10's IDs. rev 3m is a new
commit on rev 3l (`a68508606`). It folds design-5efe-r9 (`8b798c0e6..a68508606`,
0C/0I/5m/8n) under the lead's r9 rulings, corrects the tracker-edge
sentence of the C1/T6 merge ruling, and re-pins th32 at rev 3f;
Appendix B's "design-5efe-r9" table keys each item to r9's IDs. rev 3l
is a new commit on rev 3k (`8b798c0e6`). It folds design-5efe-r8 (`b29dd42d2..8b798c0e6`,
0C/1I/4m/7n) under the lead's r8 rulings, including the §M69 (b) third
amendment (the refusal exits by throwing `ggml_sycl_fallback_error`);
Appendix B's "design-5efe-r8" table keys each item to r8's IDs. rev 3k is
a new commit on rev 3j (`df0bddfed`). It folds lead ruling §M75: a losing auto-`n_ubatch`
candidate's SCRATCH growth is returned, through a zone-chunk release the
pinned pool does not have today (§4.4 staging plan, §7 C4 tests). rev 3j
is a new commit on rev 3i (`b29dd42d2`). It folds design-5efe-r7 (`87181659f..b29dd42d2`,
0C/2I/7m/6n) under the lead's r7 rulings, the §M69 (b) second amendment
(the six sites propagate `GGML_STATUS_FAILED`), §M73 and §M74; Appendix B's
"design-5efe-r7" table keys each item to r7's IDs. rev 3i is a new commit
on rev 3h (`e1b6d2750`). It folds design-5efe-r6 (`b3c08be71..87181659f`,
0C/3I/5m/4n) under the lead's r6 rulings, including the §M69 (b)
amendment to six sites (llama.cpp-573t); Appendix B's "design-5efe-r6"
table keys each item to r6's IDs. rev 3h is a new commit on rev 3g
(`87181659f`). It folds r5's added I-3 (§M66), which moved design-5efe-r5 to
0C/3I/10m/2n: each arm that scores a `[HOST-ZONE-ALLOC]` line sets the
trace variable in its literal command and names its trace-live check. rev
3g is a new commit on rev 3f (`b3c08be71`). It folds design-5efe-r5 (`fb063e845..b3c08be71`,
0C/3I/10m/2n; its I-3 was added after rev 3g and is folded in rev 3h) with
lead rulings §M63 (a), §M66, §M68 (a) and §M69 (a)-(d);
Appendix B's "design-5efe-r5" table keys each item to r5's IDs. rev 3f is a
new commit on rev 3e (`fb063e845`). It folds design-5efe-r4 as re-graded at
`3763bb164..fb063e845` (0C/3I/12m/1n), lead rulings addendum 11, §M60,
§M61 (d) and §M62 (A)-(E), and 23mk rev 4.16's relay on
`unified_cache_grow_host_scratch_zone`; Appendix B's "design-5efe-r4" table
keys each item to r4's IDs. rev 3e is a new commit on rev 3d
(`3e72abad5`). rev 3d folded the 3b-only draft of design-5efe-r3. rev 3e folds
the full r3 verdict (0C/7I/9m/1n) under lead rulings §M57 and §M58 (c), and
zhcn's GDC3 re-derivation; Appendix B's "design-5efe-r3, full verdict" rows
say where each item landed. No reported commit is amended (addendum 10 (a)).
rev 3d, a new commit on rev 3c (`fef860772`), folds
design-5efe-r3 (0C/5I/7m/1n, and its contradictions X1-X7), addendum 10 (b)
and (c), §M56 (a), and th32 rev 2x's two C1 requirements; Appendix B's r3
rows say where each landed. rev 3c added §F2a I-9 addenda 4 to 6 and rulings
addenda 7 and 8 to rev 3b (`3b447ee70`). It folds:
- design-5efe-r1 and its addendum, under lead rulings §F1, §F2 and §F2a with
  §F2a I-9 addenda 2 to 6, and rulings addenda 7 and 8;
- design-5efe-r2 (0C/6I/19m), with the lead's rulings on its I-2 and I-5;
- the th32 r15 m-4 mlock item, and the §M49 note that the lazy-range record
  lands in C1;
- the owner's default-context ruling: every model gate set carries a no-`-c`
  arm, scored, with the vehicle clarification (a CLI with no `-c`, or a
  programmatic `n_ctx = 0`). Tracker: `llama.cpp-5efe`.
Written 2026-09-27 against branch `task/5efe` at `69e70c715`. **rev 3f
re-pins every cite to master `e2461d4fb`.** Between the two trees, only
`ggml-sycl.cpp` and `common.hpp` changed among the files cited here. Every
`gs` cite, the Appendix rows' included, was moved to its `e2461d4fb` line
with a line-content check, and no cited `common.hpp` line moved. Every
cite without a revision prefix is to `e2461d4fb`. `gs` is
`ggml/src/ggml-sycl/ggml-sycl.cpp`, `uc` is
`ggml/src/ggml-sycl/unified-cache.cpp`, `uch` is `unified-cache.hpp`, `lml`
is `src/llama-model-loader.cpp` and `lm` is `src/llama-model.cpp`. Load-phase
evidence is the lead's Qwen3.8 baseline `qwen-base/run2.log` and `run2.mem`
(session scratchpad, `fefb92980`, B70 alone, `-lm mmap -lzm on -c 4096 -v`),
whose load-path code is byte-identical to `69e70c715` in every region cited
here. `e2461d4fb`'s `ggml-sycl.cpp` changes are the dense block executor, the
data-pointer cache's new-graph hook and graph-input refresh
(`gs:19341-19347`, `:89633-91233`, `:99060-99237`), none of them a region
cited as load-path evidence.
Appendix A disposes of every r1 finding, and Appendix B of every r2 finding
and later ruling.*

Sibling designs this one interfaces with, and does not restate:

- `llama.cpp-th32`: STORAGE tier for lazy row tables,
  `docs/design/sycl-storage-tier-lazy-tables.md`, abbreviated **th32**.
  Pinned at **rev 3f `d0ad45bb7`**, its current head (r9 merge unit; it
  was 3e `0426382cc` at rev 3l, 3d `c1903496f` at rev 3j, 2x `90be8e260`
  at rev 3i, 2w `3afc3b020` at rev 3c and 2u `0bd69e251` at rev 3b; r2
  m-15, r3 X7). 3f changes no section heading from 3e, nor 3e from 3d,
  and every th32 section cited here keeps its number: §0 (16 PLE rows per token),
  §2.3, §2.4, §3.1, §4.1, §4.2, §4.3, §4.5, §5.1, §9.1, and §10, which
  holds the `[PLACEMENT] lazy per_layer_token_embd` expected line (rev 3i
  cited it as §8, which was already wrong at 2x). Its §4.1 consumes
  this design's pre-fill (§3.1), the `lazy_eligible` bit, the taker and
  `deferral`, and its T6a owns the eligibility predicate that §3.1 calls. Its
  §9.1 names this design's declaration and attribute function. Its T6 fills
  the `STORAGE` branch of `create_tensor` that C1 lands (§3.1). Section
  numbers cited here are 3f's. Its §5.3 ("the backend is the authority")
  runs the backend's lazy cancel step first in each of `graph_compute`'s
  two catch paths, which is one reason this design's refusal exits
  through one of them (§4.4, §M69 (b) third amendment).
- zhcn's design, cited by row id: its **GDC3** row is the single source of
  the GPT-OSS B50 default-context demotion figures (§M56 (a)), re-derived
  at zhcn `9650b3e`. This design carries no copy of them.
  - **GDC3 is a `llama-cli` point** (§M60 (b)). The device compute buffer
    is a property of the vehicle, not of placement: the reserve graph
    carries `min(ub, n_outputs_max)` output rows
    (`src/llama-context.cpp:2051`), and `llama-cli` runs with
    `n_outputs_max = 1`. rev 3e said here that the compute buffer shrinks
    when layers demote; §M60 (a) rules that premise false, and the
    sentence is withdrawn (r4 m-6).
  - Every 5efe arm scored on GDC3 is a `llama-cli` arm (§7 C4 step 5, C5,
    C6). Each names the GDC3 point it scores against (§M61 (a), §7
    preamble), and carries zhcn's compute criterion in its §M73 (b) form,
    cited, not restated: GDC3's own statement of `compute_load +
    compute_delta == sched_reserve` for that context, and its replay's
    `compute_ctx` charge (§M73 (c)).
  - This design's own arithmetic takes no device compute-buffer size as an
    input. The Qwen room of §7 C4 step 3 is shared zone minus device
    weights, on the `llama-completion` vehicle, and is scored against that
    run's printed figure.
- `llama.cpp-7v5c`: host-only plan dry-run.
- `llama.cpp-rnh3`: exact host WEIGHT bytes.
- `llama.cpp-sgc5`: one zone set per plan.
- `llama.cpp-yl3v`: cold experts on STORAGE.
- `llama.cpp-qptd`: `-ngl 0` loses mmap.

Lead rulings are cited by their labels in the session's rulings file. The
first two th32-interface rulings, first sent as §T10/§T10a, are **§T13/§T13a**.
§T13b is withdrawn by §T13c.

---

## 0. Summary

The owner asked why SYCL disables mmap. There are **two** switches, not one:

1. **The caps slip** (the known root cause). Upstream `153d324bc` added
   `ggml_backend_dev_caps::mmap_support`. The fork's
   `ggml_backend_sycl_device_get_props` lists four initializers
   (`gs:108004-108009`), so the fifth is value-initialized to `false`, and
   `lm:2245-2265` then resolves `-lm auto` to `none` and `-lzm auto` to `off`
   on every SYCL load, discrete cards included.
2. **The fork's own override.** `lm:2693-2696` sets `ml.use_mmap = false` for
   every SYCL model *after* `create_tensor`, whatever `-lm` says ("disabling
   mmap for SYCL weight layout upload", added by `0f475d1aa`). The baseline
   shows it firing after the loader printed `load_mode = mmap`
   (`run2.log:314` then `:2902`).

Even with both removed, mmap would change nothing that matters, because of a
third fact: with the default `GGML_SYCL_WEIGHTS_EVICTABLE` (auto, `true`,
`gs:13591-13614`) the loader places **every** SYCL-layer weight in the
`SYCL_Host` buffer type, which is pinned USM from the unified cache
(`lml:1128-1135`, `:1400`; `gs:42632-42712`). mmap would only change where
the copy into that pinned buffer reads from. The baseline shows what that
costs:

- The pre-size committed 139264 MB of pinned chunks (`run2.log:2900`).
- The pool reached 140.0 GB at chunk 67 (`:2967`).
- `MemAvailable` fell by 141 GiB, from 155.0 to 13.6 GiB (`run2.mem`),
  before the watchdog killed the run.

This design therefore does four things:

- **Commit 1 restores `mmap_support`**, as `host_unified_memory` known and
  false (§3). By itself that moves one thing: `-lzm auto` engages again, so
  the 50.66 GiB Qwen PLE table leaves the pinned pool under default flags.
  Commit 1 also lands the **per-file lazy-range record** (§3.1). It is the one
  source of every `TENSOR_READ_LAZY` byte range, and lazy_read, th32's
  inventory, the tier and llama_mmap all read it.
- **A new allocation tier, `FILE_MAPPED`** (§4.1-§4.3).
  - It is a read-only file mapping that the unified cache creates inside one
    VA reservation, owns through one `alloc_owner`, and releases with one
    `munmap`.
  - A new per-device buffer type, `SYCL_Mapped`, replaces `SYCL_Host` as the
    source buffer of every SYCL weight when the load maps files.
  - Host weights that the CPU executes in file layout are served from the
    mapping.
  - Device weights are copied from it once, at load or materialization,
    through a transient pinned window. So **the double residency of every
    VRAM weight (pinned host + VRAM) ends in mmap mode** (§6).
- **One published fact, resolved before the early plan** (§4.3, §4.4). The
  load mode is resolved before the planner runs, and the planner gives every
  weight a `host_backing`. Both the WEIGHT-zone sizing and the loader's
  buffer-type choice read that. The split lands in the same commit as the
  loader (C4), so the Qwen acceptance never runs against a WEIGHT zone sized
  for pinned experts (r1 C-1).
- **Budget** (§4.5).
  - Mapped bytes are reported but not charged, unless `-lm mmap+mlock` locks
    them.
  - The host budget becomes monotone in `MemAvailable`, with one named
    headroom, `max(total/32, min(29 GiB, total/4))`.
  - A mapped working set that cannot stay resident is a WARN (§F1 Q1).
  - The host-fit refusal is th32 T2's over 7v5c's computation; 5efe supplies
    its terms.

**Acceptance** (lead-run, §7 C4): Qwen3.8-Flash-Next Q8_0 loads on the B70
(`level_zero:0`) under default load flags, and every figure below was
pre-registered before the run:

- the continuation on stdout begins `Paris` and does not degenerate into
  repetition (§7 C4 step 2);
- the WEIGHT zone is 64.0 MB, and it never grows after the pre-size;
- the pre-size commits 3 chunks, 6144.0 MB, for a 3579.2 MB base working
  set and a 91.9 MB runtime need, scored on those printed figures (§4.4);
- `MemAvailable` never falls under 25 GiB;
- `SYCL_Mapped` holds about 127615 MiB;
- shard 3, the PLE table, stays under 256 MiB resident;
- a no-`-c` arm at the default context passes too, with the KV overflow
  re-placed to the host tier (§7 C4).

---

## 1. Scope and principles

- **P1, the cache owns the bytes.** Today the fork has no cache-owned file
  mapping. A weight in a `CPU_Mapped` buffer (upstream's mmap path, the lazy
  tables) sits outside the contract (canonical contract §8 items 2 and 4), and
  host experts are registered as `non_owning_external_host` slices the cache
  never frees (`uc:8150`, `:12351`). The tier fixes both for mapped weights.
- **P2, the handle owns the lifetime.** Every reader of a mapped byte holds a
  `mem_handle`, directly or through an owner that wraps one. The last release
  unmaps (§4.2).
- **P3, placement decides the executor.** There is no GPU zero-copy of host
  memory and no weight streaming.
  - A mapped byte is read by the CPU only.
  - The GPU receives mapped bytes only as a load-time or materialization copy
    through a pinned window.
  - A mapped weight reaching the dispatch-time streaming path is a typed
    refusal (§5.3).
- **P4, plan equals reality, and no idle reservation.** One published fact
  says whether the load maps files. The plan says which bytes are pinned,
  mapped or neither, and the ledger charges exactly the resident ones (§4.4,
  §4.5).
- **Out of scope here:**
  - The lazy row tables themselves. th32 owns them; §4.6 only guarantees that
    the tier never maps, prefetches or locks them.
  - The cold-expert policy. yl3v owns it; §4.5 states the interface.
  - The `-lm none` double residency. It gets a ticket (§6).
  - `CPU_REPACK` buffers living outside the cache. This is an existing P1 gap,
    also a ticket (§4.4).

---

## 2. The load path today (item 1)

### 2.1 How `-lm` and `-lzm` resolve

- `llama_model_loader` sets `use_mmap` for `mmap`, `mmap+mlock` and `auto`
  (`lml:599`).
- `load_tensors` computes the SYCL early placement plan first (`lm:2228-2231`).
  Only then does it drop `use_mmap` under `auto` when any model device reports
  `mmap_support = false` (`lm:2245-2254`), and turn lazy `auto` into `off` on
  the same test (`lm:2256-2265`). The fork's SYCL devices all report `false`
  (§0), so the default command line loads with no mmap and no lazy tables. The
  early plan therefore runs on an unresolved load mode, which §4.3 fixes
  (r1 I-5).
- An explicit `-lm mmap` survives that check, and the loader prints
  `load_mode = mmap` (`lm:2268-2272`, `run2.log:314`).
- `create_tensor` runs next, with `use_mmap` still true. Then `lm:2693-2696`
  clears it for every SYCL model (`sycl_model_backend` is "a SYCL load
  transaction exists and some context uses a SYCL buft", `lm:2535-2542`), and
  `init_mappings` runs after that (`lm:2702`). So `load_all_data` never takes
  its mapping branch for an ordinary weight on SYCL, whatever `-lm` says.

### 2.2 Which buffer type every weight gets

| weight | buffer type chosen | cite |
|---|---|---|
| any tensor of a SYCL layer, input or output (default `weights_evictable() == true`) | `SYCL_Host` of that layer's device | `select_weight_buft(prefer_host_weights)`, `lml:1128-1135`; the flag, `lml:1396-1401`; `weight_buft_supported` returns true for it before any op check, `lml:978-985` |
| planner-host dense weight (`planned_target_device == -1`) | `CPU_REPACK`, or plain `CPU` under `GGML_SYCL_HOST_WEIGHT_LAYOUT=plain` (llama.cpp-ir18) | `lml:1473-1495` |
| `-ot <pat>=CPU` | `SYCL_Host` again, reported as not honoured (llama.cpp-lufn) | `lml:1406-1440` |
| `TENSOR_READ_LAZY` tensor chosen by `lazy.add` | the CPU device's default buft | `lml:1368-1370`, `:1232-1238`, `:1774-1777` |
| MoE `*_exps` | `SYCL_Host`; the plan has no name-indexed entry, so ir18 does not touch them | `lml:1469-1471` |

Upstream's "avoid a host buffer under mmap" demotion (`lml:1512-1518`) is
disabled for SYCL by `allow_host_buft_with_mmap` (`lml:1503-1511`), so even a
surviving `use_mmap` keeps the weights in `SYCL_Host`.

`make_cpu_buft_list` puts the first device's host buft at the front of the CPU
list unless `--no-host` (`lm:1765-1779`). That is where `token_embd` lands, and
it is `llama.cpp-qptd`'s finding: at `-ngl 0` the weights still land in
`SYCL_Host`.

### 2.3 Where the bytes land

**Host-resident weights land in pinned USM, copied, never in mapped pages.**
`SYCL_Host` allocates through `unified_alloc` with role `WEIGHT` (whenever
`weights_evictable`, even after the load, `gs:42650-42651`), category
`HOST_COMPUTE` and `must_host_pinned` (`gs:42645-42670`). That routes to the
host WEIGHT zone (`uc:15361-15372`). `load_all_data` fills it with `read_raw`
straight into `cur->data`, because the buffer is host (`lml:2134-2141`). With
mmap it would have been a `ggml_backend_tensor_set` from the mapping into the
same buffer (`lml:2129`).

**Device-resident weights get their bytes from that pinned buffer, at load
end.**
- `ggml_backend_sycl_model_load_end` runs `ggml_sycl_preload_model_weights()`
  (`gs:12476`, defined `:33520`).
- It takes `src_ptr = tensor->data` (`gs:34768`, `:35667`) and calls
  `unified_cache::direct_stage_weight` (`uc:6036`) to fill a fresh VRAM
  allocation.
- The "tiered" override that could have replaced `src_ptr` is dead:
  `get_cached_tensor_ptr` always returns `nullptr` (`gs:19236-19261`).
- Nothing releases the pinned copy afterwards, and the published entry records
  `src_ptr` into it (`uc:6213`).

**So every VRAM weight is resident twice** for the model's lifetime (lead's
trace report, point 1).

**Host experts are registered in place.** `expert_aos = src0->data +
e * expert_size` (`gs:33442`) is registered by `register_host_expert` with no
owner, so `non_owning_external_host = true` (`uc:8150`) and the cache never
frees it (`uc:12351-12353`). A second pinned allocation happens only when a
CPU reorder is needed (`register_host_aos_expert`, `gs:34147`, alloc at
`:34181`).

**`CpuExpertPool` reads that pinned slice.** Each task carries the host
lease resolved from the registered entry (`gs:76025-76033`), and the pool
executes it (`gs:76054-76072`). `ggml_sycl_host_data()` is `tensor->data`
(`common.hpp:4889-4891`), which is the `SYCL_Host` buffer.

### 2.4 `use_mmap_buffer` and `buffer_from_host_ptr` in this fork

- `use_mmap_buffer` is the constant `true` (`lm:2241`).
- The mapped branch requires `caps.buffer_from_host_ptr` **and** that the buft
  be its device's *default* buft (`lm:2740-2748`).
  - The SYCL device reports `false` (`gs:108007`), and its
    `buffer_from_host_ptr` hook is a stub returning `nullptr`
    (`gs:108054-108062`).
  - `SYCL_Host` is not a default buft anyway.
- So the only buffers built from a mapping are the CPU device's (`CPU_Mapped`,
  `ggml-cpu.cpp:398-400`): plain-CPU weights under `use_mmap`, and the lazy
  tables.
- After the load, `pimpl->mappings` keeps each `llama_mmap` alive for the
  model's lifetime (`lm:2866-2870`). The last `load_all_data` call trims the
  unused head and tail of each mapping (`lml:2236-2247`).

### 2.5 `TENSOR_READ_LAZY` on SYCL (`-lzm on`, Qwen `per_layer_token_embd`)

- `qwen4exp` creates the table with `TENSOR_READ_LAZY`
  (`src/models/qwen4exp.cpp:189`). `lazy.add` accepts it under `on` at any
  size, and under `auto` only above 4 GiB (`lml:1240-1268`). `run2.log:370`
  shows `lazy read enabled` for 51880 MiB.
- The byte range is kept in `lazy_read`'s private
  `std::map<uint32_t, llama_mmap::ranges> ranges` (`llama-model-loader.h:118`,
  filled at `lml:1258`). `llama_mmap` receives it through `lazy.for_file(idx)`
  (`lml:1890-1891`). §3.1 replaces that copy with the record.
- Its buffer type is the CPU default buft (§2.2), so the context is a lazy CPU
  context.
  - `init_mappings` maps every file whenever `lazy.any()`, even after
    `lm:2693` cleared `use_mmap` (`lml:1869-1872`), with prefetch 0
    (`lml:1888`).
  - The mapping is advised `POSIX_MADV_RANDOM` over the lazy ranges and
    excluded from `WILLNEED` and `MADV_HUGEPAGE`
    (`src/llama-mmap.cpp:503-545`). `MAP_POPULATE` is gone (`:479-481`).
- `load_tensors` wraps it as `CPU_Mapped` (`is_lazy_mapped`, `lm:2745-2748`).
  `load_all_data` points the tensor into the mapping without reading it
  (`lml:2099-2128`).
- **It can still be mlocked** (r1 addendum I-10). Under `-lm mmap+mlock`,
  `load_all_data` skips the growth call only for the lazy tensor itself
  (`!lazy.has(cur)`, `lml:2120`). A non-lazy mapped tensor that comes after the
  table in the same file calls `grow_to(weight->offs + n_size)` (`lml:2122`).
  `llama_mlock::grow_to` locks `[addr + size, target)` contiguously from the
  mapping's base (`src/llama-mmap.cpp:826-840`), so that call locks, and
  therefore faults in, every lazy byte before the tensor. §4.6 fixes it.
- It is not a SYCL inventory entry.
  - The late inventory filters out non-SYCL bufts, so the late plan's host
    total drops from 154263.6 to 102383.5 MB (`run2.log:308` vs `:2821`,
    exactly 51880.1 MB less).
  - Its `get_rows` runs on ggml-cpu, faulting rows in from the file at the
    storage rates th32 §5.1 measured.
- The table is the only tensor in shard 3 of 6 (offset 192, 54400261120 B,
  read from the GGUF headers). §7's residency gate uses that.

### 2.6 The baseline's second 104 GB

The lead asked whether `Host staging zone ... per_inflight=103762.3 MB`
(`run2.log:243`) is a second reservation. It is not.

- The STAGING zone is `max(2 x max_tensor, max_staging_pair) + 2 MiB` per
  in-flight slot (`uc:27674-27708`).
- The **early** plan (`lm:2230`, before `create_tensor`) still counts the lazy
  table as `max_tensor`: 2 x 51880.1 + 2 = 103762.3 MB.
- The late plan, which no longer sees the table, prints 1702.0 MB
  (`run2.log:2756`).
- Both configure lines carry `STAGING=1719.2 MB` (`run2.log:2893`, `:2899`).

So it is a plan-time figure that was never committed. th32 §4.1 already
requires lazy tables to be excluded from `max_tensor`. The early plan cannot
see the table without the record, so C1's pre-fill is what makes that
exclusion reach it (§3.1), and th32's T3 fixture should cover the early plan
too.

### 2.7 The legacy "mmap" vocabulary

None of it refers to a `llama_mmap` page today:

- `alloc_tier::MMAP_TRACKED` (`uch:5593`) is what `query_location` returns for
  a pointer the registry does not know ("assume host/mmap", `uc:21649-21655`).
  It is also the default of `memory_location::tier` (`uch:5614`).
  `unified_alloc` refuses it (`uc:15125-15127`), and `unified_free_record`
  returns without freeing it (`uc:14812-14814`).
- `cache_location::HOST_MMAP` (`uch:2170`) is derived from that classification
  when an entry is registered (`uc:8068-8081`, `:8235-8248`). It is also set by
  the direct-stage last resort "host arena exhausted, raw mmap fallback"
  (`uc:6504-6545`).
- `alloc_type::MMAP` (`alloc-registry.hpp:26`, commented "mmap'd from GGUF")
  has two meanings in use, neither of them a GGUF mapping:
  - the tiered KV buffer's synthetic, non-USM base (`gs:39636-39644`);
  - "regular malloc (pinned cap overflow)", which `unified_free_record` frees
    with `::free` (`uc:14922-14929`).
- `memory_tier::MMAP` (`tensor-types.hpp:36-40`, "File-backed mmap (slowest,
  fallback)") is the fourth name (r2 m-11). Its six uses (`gs:14317`,
  `:19256`, `:19279`, `:34771`, `:35670`, `:98936`) are the initial value and
  the "not found" result of `get_cached_tensor_ptr`, which always returns
  `nullptr` (§2.3). So it means "unresolved", not a mapping.

### 2.8 The host budget today

- `[HOST-ARENA]` is `min(total x 90%, MemAvailable - max(8 GiB, total/32))`
  (`uc:4200-4231`). It is the same function th32 §4.2 lifts into
  `host_ledger_budget()`.
- It is not monotone: when `MemAvailable` does not exceed the reserve, the cap
  becomes `available/2` (`uc:4227-4230`).
- The planner has only `placement_entry::on_device`, so "host" means pinned,
  and `weight_host_bytes` is never compared with any budget (lead's
  host-budget report; th32 §2.3).
- No load-time host-fit gate exists.

---

## 3. Commit 1: the caps restore (item 2)

```cpp
// gs:108004, ggml_backend_sycl_device_get_props
ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *) dev->context;
const auto & info = ggml_sycl_info().devices[ctx->device];
props->caps = {
    /* .async                 = */ async,
    /* .host_buffer           = */ host_buffer,
    /* .buffer_from_host_ptr  = */ false,
    /* .events                = */ events,
    /* .mmap_support          = */ info.host_unified_known && !info.host_unified_memory,
};
```

- **The iGPU detection already exists.**
  - `ggml_sycl_device_is_host_unified()` queries
    `sycl::info::device::host_unified_memory` (`common.cpp:827-833`), and
    `ggml_sycl_info()` caches the result per device (`gs:25200`, field at
    `common.hpp:1477`).
  - `ggml_sycl_info()` is initialized before any SYCL device object exists,
    because registration iterates it (`gs:110393`). So the cached field is
    safe to read here.
  - This matches CUDA's `type != IGPU` (`ggml-cuda.cu:5130`) and Vulkan's
    `!is_integrated_gpu` (`ggml-vulkan.cpp:15221`).
  - Upstream SYCL sets a flat `true` (`694ec2354 ggml-sycl.cpp:6380`). The
    fork keeps the iGPU off because of its own history (llama.cpp-403s).
- **The query fails open today, so C1 closes it for this cap only** (r1 m-17).
  - `ggml_sycl_device_is_host_unified()` returns `false` when the query
    throws. An iGPU whose query throws would otherwise advertise
    `mmap_support = true`.
  - C1 adds `host_unified_known` beside the cached field, true only when the
    query returned.
  - `mmap_support` requires it. The other readers of `host_unified_memory`
    (the VRAM budget's iGPU cap, `common.cpp:889-936`) keep today's semantics.
- **What changes under `-lm auto -lzm auto`**, for a device set of discrete
  cards only (`level_zero:0`, `:1`, `:0,1`):
  - `ml.use_mmap` stays true through `lm:2245-2254`, and the log reads
    `load_mode = mmap` instead of `none`.
  - Lazy `auto` stays `auto`, so every `TENSOR_READ_LAZY` tensor above 4 GiB
    becomes a lazy table (§2.5). On Qwen that takes the 50.66 GiB PLE table out
    of the pinned pool **under default flags**. This is the only material
    change of commit 1.
  - `create_tensor` sees `use_mmap = true`. The `-ot =CPU` mmap warning can
    print (`lml:1434-1439`), and the host-buft demotion stays disabled for SYCL
    (§2.2). Buffer types are otherwise identical.
  - `lm:2693` still clears `use_mmap`, so every non-lazy weight is still read
    into `SYCL_Host`. Commit 1 deliberately does **not** remove that line:
    without the tier, the only effect would be a copy from page cache into
    the same pinned buffer, doubling the transient footprint.
- **An unpinned selector** that includes the iGPU makes the iGPU a model
  device, and both fallbacks fire exactly as today. That is the intended
  behaviour, and it is never run on this host (CLAUDE.md, llama.cpp-403s).
- **th32 drops its own flip** (lead ruling §T13, point 1). The pre-fallback
  mode capture that th32's T6 used to own is now commit 1's (§F2a I-9
  addendum 4). Commit 1 captures it, with the taker, at the §3.1 pre-fill,
  before its moved fallback block.

### 3.1 Also in commit 1: the per-file lazy-range record (§F2a I-9, §M49)

**Why it exists.** Four consumers need the byte ranges of the
`TENSOR_READ_LAZY` tensors:

- `lazy_read`, for the CPU lazy path;
- th32's inventory, after T6;
- the tier's exclusion (§4.6);
- `llama_mmap`'s prefetch, `MADV` and lock exclusion (§4.6; §T13c permits
  feeding it from 5efe's code).

Today the ranges live only in `lazy_read`'s private map (§2.5). th32 T6
diverts SYCL tables *out* of `lazy_read`, so after T6 that map would hold no
SYCL table. The tier would then receive an empty exclusion set, and the
loader's mapping would `WILLNEED` the complement of that empty set, which is
every byte of every shard, the 54.4 GB table included (r1 addendum I-9). One
record, filled independently of who defers the read, removes that failure
mode.

**The record.** `llama_lazy_range_record`, one per loaded file, owned by 5efe
and held by `llama_model_loader`.

- **Membership is by declaration** (§F2a I-9 addenda 3 and 5). There is one row
  per declared name found in `weights_map`, whatever any consumer decides about
  it. A row holds (tensor name, file index, offset, size, `lazy_eligible`,
  `deferral`, and the attributes `rows_per_token` and `consumer_layer`).
- **Two per-model facts are captured once, at the pre-fill** (addenda 4 and 5):
  - **`mode_pre`**, the lazy mode as requested (`ml.lazy.mode`), captured
    **before** C1's moved `AUTO`-to-`OFF` fallback block (`lm:2256-2265`
    today). This capture is a 5efe C1 deliverable, no longer a th32 T6 one.
  - **`taker`** = `sycl_model_loading_guard.active` (`lm:2228-2230`). It is
    true when the model's weights are SYCL-owned (`n_gpu_layers() > 0` with a
    SYCL model device, or a SYCL buffer-type override, `lm:2214-2227`) and the
    hooks resolve. It is the fact the early plan is already gated on
    (`lm:2229`). It is **not** `llama_model_sycl_hooks_enabled()`, which is
    process-level: a `-ngl 0` load in a SYCL build registers the hooks
    without taking, and must not get `STORAGE`. In a build with neither
    `GGML_USE_SYCL` nor `GGML_BACKEND_DL` (`lm:2213`), `taker` is false.
  - **`mode_post`** is `ml.lazy.mode` after the fallback block. It is not
    captured separately: it is what `lazy_read` holds.
- **`lazy_eligible`** is one bit, computed once at the pre-fill by **th32's
  T6a predicate**, the DLM (a) rule (th32 rev 2w `3afc3b020` §4.1 and its T6a
  row), called and not restated:

  ```
  lazy_eligible = T6a(taker ? mode_pre : mode_post,
                      size,
                      taker ? true : llama_mmap::SUPPORTED)
  ```

  - `OFF` is never eligible, `AUTO` only above 4 GiB, and `ON` always, each
    only when the third input is true.
  - **Why the mode differs by path** (addendum 4). The taker reads through
    its own `O_DIRECT` fd and never through a mapping, so the fallback, which
    exists because the CPU lazy path needs a mapping, must not disengage it,
    and a no-mmap platform does not either (addendum 6 (1)). Off the taker,
    `lazy_read` really does read with `mode_post`, so feeding it `mode_pre`
    would mark a table that is read eagerly as lazy, and the prefetch
    exclusion would then leave a table cold that is about to be read in full.
- **`deferral`** carries only the consumer kind (addendum 5):
  - `STORAGE` iff `lazy_eligible && taker`;
  - `LAZY_READ` iff `lazy_eligible && !taker`;
  - `EAGER` otherwise.

  So **`deferral != EAGER` iff `lazy_eligible`**: one fact, one source. rev
  3b's (`3b447ee70`) eligible-but-`EAGER` case is gone, because off the taker the bit
  already reads `mode_post` and `SUPPORTED`. `STORAGE` is written here at the
  pre-fill, like every other field, since rows are immutable afterwards; th32
  T6 never writes it. T6 takes a table iff `deferral == STORAGE`, in place of
  its `hooks_enabled` gate.
- Consumers read `deferral` or the bit. None re-evaluates the mode, the floor
  or `SUPPORTED`.
- **Where it is filled: before the early plan, not at `create_tensor`**
  (§F2a I-9 addendum 2).
  - **Why `create_tensor` is too late.** The SYCL early plan runs before
    `create_tensor` (`lm:2228-2231`) and builds its inventory from
    `ml.weights_map` (`lm:639-641`), so it cannot see a row filled there. It
    would place and stage the lazy table as an ordinary weight: run2's early
    plan counted the PLE in `max_tensor` (§2.6) and in the host total
    (154263.6 MB, `run2.log:308`).
  - **When the flag is knowable.** `TENSOR_READ_LAZY` is first knowable only
    at `create_tensor`. It is an argument at each arch's call site
    (`src/models/qwen4exp.cpp:189`, `src/models/gemma4.cpp:58`), and no table
    records it (`llm_tensor_info` is `{layer, op}`, `llama-arch.h:782-785`).
  - **The declaration** (r2 I-5; lead ruling: 5efe's, since the record is
    5efe's; th32 consumes it; keyed by arch per addendum 5; names only per
    addendum 7 (a), whose naming addendum 8 (a) amends back to this):

    ```cpp
    // src/llama-model.h, C1
    // The GGUF names, or name patterns with one "%d" standing for the block
    // index, that `arch` creates with TENSOR_READ_LAZY. Static and immutable:
    // names only, no hparams-valued field.
    const std::vector<std::string> & llama_model_lazy_tensor_names(llm_arch arch);

    struct llama_lazy_tensor_attrs {
        uint32_t rows_per_token;  // 0 = not known for this arch
        int32_t  consumer_layer;  // -1 = not known for this arch
    };
    // One per-arch function, next to the declaration, reading the loaded hparams.
    llama_lazy_tensor_attrs llama_model_lazy_tensor_attrs(llm_arch arch, const llama_hparams & hparams,
                                                          const std::string & name);
    ```

    - **Keyed by (arch, GGUF name or pattern)** (addendum 5, amending r2 I-5's
      "keyed by GGUF name"). The arch key is load-bearing: `gemma3n` creates
      the same `per_layer_token_embd.weight` **without** the flag
      (`src/models/gemma3n.cpp:39`). With a name-only key, every gemma3n load
      would get a row, and the drift check below would abort it.
    - **The declaration holds names only** (addendum 7 (a); names per
      addendum 8 (a), and th32 keeps them). rev 3b put `rows_per_token` and
      `consumer_layer` in a struct for T6 to fill in. A static table that T6
      writes is mutable global state shared by every loaded model, so two
      loaded models with different hparams would overwrite each other's
      values (P4).
    - The name list takes no hparams. Presence in the GGUF already bounds it,
      so the call sites' hparams conditions (`qwen4exp.cpp:171`
      `ple_n_heads > 0`, `gemma4.cpp:57` `n_embd_per_layer > 0`) are not
      stated a third time.
    - **The attributes live on the per-model record entry.** C1's pre-fill
      calls `llama_model_lazy_tensor_attrs` once per matched name, with the
      loaded hparams, and stores the result on that model's record entry.
      th32 T6 reads it from there through `record.attrs(name)`, and never
      from a static.
      - The function is 5efe's, and **C1 lands it with the real values**,
        which T6 only reads (§M58 (c)): `qwen4exp` returns
        {`hparams.ple_n_heads` (`qwen4exp.cpp:95`), the index of the one set
        bit of `hparams.is_ple_impl`}, and `gemma4` returns {1, 0}.
      - **`consumer_layer` comes from hparams, not a literal** (r3 m4). The
        qwen4exp loader reads `qwen4exp.ple.layers`, refuses more than one
        entry, and sets that layer's bit in `is_ple_impl`
        (`src/models/qwen4exp.cpp:65-84`). This GGUF lists `[1]`, so the
        value is 1 here, but a literal 1 would be a second source, silently
        wrong for a GGUF with another PLE layer (P4). th32 §4.1's table
        carries the same literal; the lead relays this to th32.
      - The **missing-attribute refusal** is T6's (addendum 5). It applies
        only to `deferral == STORAGE` rows, and runs before the early plan
        reads the row. A non-taker row is never refused for missing
        attributes.
    - Today's entries are `per_layer_token_embd.weight` for `qwen4exp` and for
      `gemma4`. deepseek41 is **not** declared: that arch does not exist in
      the tree, and llama.cpp-9g22 (paused) adds its own entry when it lands
      (addendum 4).
    - **Not an `LLM_TENSOR_INFOS` column.** That table is keyed by tensor kind
      across every arch (`llama-arch.cpp:721`). A kind-level column would
      declare gemma3n's table lazy, for the same gemma3n reason.
  - **The pre-fill.** In `load_tensors` (`lm:2212-2232`), C1 reorders the
    code to:
    1. `resolved_sycl_weight_owner` and the guard, unchanged (`lm:2214-2228`);
    2. capture `mode_pre` and `taker`;
    3. the caps loop (`lm:2245-2254`) and the lazy-mode fallback
       (`lm:2256-2265`), moved up. They read `devices`, `params.load_mode`,
       `ml.use_mmap` and `ml.lazy.mode`, and nothing the early plan writes
       (r2 verified);
    4. print **one per-model INFO line** (th32 rev 2x's requirement, its
       §4.1 and T6 row (k)), `lazy-range record: mode_pre=<mode>
       mode_post=<mode> taker=<0|1>`, with `mode_post` read after the
       moved fallback. It prints for every model, whether or not any name is
       declared, so a harness can read both modes and the taker without a
       lazy table. It is INFO, like the row lines below. A test that installs
       a callback through `llama_log_set` receives every level, since
       `llama_log_internal_v` hands each line to the callback unfiltered
       (`src/llama-impl.cpp:40-55`); only `common_log`'s default callback
       drops INFO, at `verbosity <= common_log_verbosity_thold`
       (`common/log.cpp:542-546`, with INFO mapped to TRACE at `:532`). A
       CLI reads it with `-v` or `-lv 4`;
    5. fill one row per declared name found in `weights_map`: (name, file
       index, offset, size) from the `llama_tensor_weight`, the bit, the
       deferral, and the attributes. Each row prints one INFO line,
       `lazy-range record: <name> (<size> MiB) eligible=<0|1>
       deferral=<EAGER|LAZY_READ|STORAGE> taker=<0|1> mode=<the mode the bit
       read>`. It replaces `add()`'s "lazy read enabled" line as the witness
       §7 C1's lead runs read;
    6. `llama_model_sycl_compute_early_plan`, if `taker` (`lm:2229-2231`).

    It runs the same way under `no_alloc` (r2 m-9), so the fit pass plans what
    the real load plans. See `lazy_read` below for why that maps nothing in
    the fit pass.
  - **One source, checked twice.**
    - **Statically:** `test-llama-lazy-declaration-source` (C1) maps every
      `TENSOR_READ_LAZY` `create_tensor` call in `src/models/*.cpp`, with its
      arch, to a declaration entry, and every entry back to a call site. An
      upstream merge that adds a flagged call, or a new upstream arch that
      arrives with one, fails it at build time, on every backend's tree,
      before any load.
    - **At runtime, mode-independent** (addendum 4; r2 I-5(3)).
      `create_tensor` aborts, naming the tensor and the arch, on a
      `TENSOR_READ_LAZY` call to a tensor with **no** row, and on a non-lazy
      call to a tensor that **has** one, whatever the row's eligibility,
      `deferral` and the mode. So drift is caught under `-lzm off`, and under
      `AUTO` at or below 4 GiB, too. rev 3a checked only non-`EAGER` rows,
      which let exactly those cases through.
  - **The early plan reads it.** Its loop over `weights_map` (`lm:639-641`)
    skips every row whose `deferral` is `STORAGE`, the only kind it can see,
    since the early plan runs only when `taker` is true. That keeps an
    eligible table out of placement, out of `max_tensor` and the staging
    maxima, and out of the host total, which is th32 §4.1's requirement
    applied to the early plan too. An ineligible row (`OFF`, or `AUTO` at or
    below 4 GiB) is placed as the ordinary weight it is (th32 r19 M-1; the
    witness is in §7 C1). The late inventory already excludes deferred tables
    (§2.5).
- **Its API.**
  - `deferred_for_file(idx)` returns the sorted, merged ranges of that
    file's **`LAZY_READ`** rows, and no `STORAGE` row (r3 m3 (b), §M57 (d)).
    `storage_for_file(idx)` returns its `STORAGE` rows' ranges, and
    `excluded_for_file(idx)` the union of the two, which the two exclusion
    sites pass (§4.6). A `STORAGE` row reaches `init_mappings` only on a tree
    where T6 has claimed it (the refusal above), and T6 does not read it
    through the mapping (th32 stays off mmap, §T13c), so excluding it there
    is right. On a C1-only tree the `STORAGE` part is empty on every load that
    gets that far, and the union equals `deferred_for_file`.
  - `has(name)`, `eligible(name)`, `deferral(name)` and `attrs(name)` are the
    per-tensor queries.
  - `value(name)` returns a `llama_lazy_range_value {file_idx, offs, size}` **by
    value**. th32's backend `lazy_table` takes its (file, offset, size) from
    this once, at plan publish, through the inventory. It is a hand-off of the
    record's value, not a second source (§F2a I-9 addendum 2, M-1). What is
    forbidden is deriving those numbers independently, for example from
    `w->offs`.
- **Immutability.** Rows, the bit, `deferral` and the attributes are set
  once, by the pre-fill, and a later write aborts. So a by-value copy taken at
  plan publish stays equal to the record for the model's lifetime, and is
  safe for tickets that outlive the model.

**Consumers keep no copy of the ranges.**
- **`lazy_read`** keeps its `mode` and a **membership** set of names, as th32
  §4.1 requires (its r18 N-4).
  - **`add(name, t, w)` reads `deferral` and decides nothing** (addendum 5;
    addendum 6 (2): a C1 deliverable). **Its return value is
    `record.deferral(name) == LAZY_READ`, and nothing else** (r3 m1). Its
    own mode test, the 4 GiB floor and the `llama_mmap::SUPPORTED` test
    (`lml:1241-1256`) are deleted.
    - **Membership is a separate axis from the decision.** `add()` inserts
      the name into the membership set only when `w != nullptr`, that is,
      only when the real load will map the file. Under `no_alloc`,
      `create_tensor` passes `nullptr` (`lml:1776`), so `add()` still returns
      true for a `LAZY_READ` row and `is_lazy` is set, while membership stays
      empty and nothing is mapped. Upstream's comment at `lml:1775` requires
      exactly that: the decision "must not depend on the load mode, or the
      memory-fit pass (no_alloc, no mmap)" would size a lazy table as
      resident. rev 3c's `w != nullptr` term in the decision made `is_lazy`
      false in the fit pass, so the fit pass sized the 51880 MiB table that
      the real load leaves unread.
    - For a `STORAGE` row `add()` returns false, since `lazy_read` is not its
      consumer, and `create_tensor` must not then make the tensor an
      ordinary input-layer weight (r3 m3 (a), §M57 (d)). **C1 lands the
      branch that T6 fills.** `create_tensor` tests `record.deferral(name) ==
      STORAGE` before `buft_for_tensor`, and C1's body of that branch throws
      `lazy-range record: <name> is STORAGE and this tree has no STORAGE
      consumer (th32 T6)`. T6 replaces the body with its SYCL branch (th32 2x
      T6 row). Tensor creation runs before `init_mappings` (`lm:2702`) and
      before the late plan, so on a tree with C1 and not T6 a taker load of
      an eligible declared table stops there: it never becomes a 51880 MiB
      `SYCL_Host` weight, and is never read through the mapping. W1-W3 are
      killed before tensor creation (§7 C1), so no C1 gate reaches it.
    - That is a fork change to upstream code, so
      `test-llama-lazy-range-record-source` pins it (§7 C1). A rebase that
      restores `add()`'s own gates fails loudly.
    - Landing it in C1, where `deferral` is filled, leaves no window in which
      `add()` and the bit disagree.
    - `add()`'s "mmap is not available, so tensor %s ... is loaded into RAM in
      full" WARN (`lml:1252-1253`) moves to the pre-fill. It prints for a
      declared row whose bit is clear because `SUPPORTED` is false off the
      taker.
  - `any()` and `has()` answer from that membership. `for_file(idx)` returns
    the record's ranges for its members only.
  - Its private `ranges` map is deleted.
  - So under `taker`, where eligible rows are `STORAGE`, `any()` stays false,
    and `init_mappings` does not map every shard with `use_mmap` cleared
    (`lml:1871`, the hazard r2 I-6 names).
  - Under `no_alloc`, membership stays empty and the fit pass maps nothing,
    while a `LAZY_READ` table still reads as lazy to the fit pass (r2 m-9,
    r3 m1; the membership bullet above).
- `init_mappings` passes `record.excluded_for_file(idx)` to `llama_mmap`
  (`lml:1890-1891`).
- th32's inventory reads each table's (file, offset, size) from the record
  through `value(name)` (th32 §4.1).
- The tier hook's `lazy_ranges` argument (§4.3) is
  `record.excluded_for_file(idx)`.

**What it changes on its own.** The ranges and decisions are the ones
`lazy_read` computes today, on today's `AUTO`-to-`OFF` fallback. The one
observable change is that the early plan stops counting deferred tables. On
Qwen with lazy enabled:
- the early `[SYCL-PLAN] Host staging zone` reads `per_inflight` about
  1702 MB, like the late plan, instead of 103762.3 MB (`run2.log:243`);
- the early `[PLACEMENT] Totals` host figure reads 102383.5 MB instead of
  154263.6 MB (`run2.log:308`).

The arithmetic is exact. The early inventory's largest tensor is
`per_layer_token_embd.weight` at 51880.1 MB (`run2.log:235`), and
154263.6 - 51880.1 = 102383.5, the late host total (`:2821`). The early
`per_inflight` is 2 x 51880.1 + 2.1; the late one is 2 x 850.0 + 2.0, with
the 850.0 MB `ffn_*_exps` tensors next in line (`:235`, `:2756`).

**Data edges.** th32 T6 depends on C1 alone (§M49): C1 lands the record, the
two captures, the declaration, the attribute function and `add()`'s reading
of `deferral`. C1 depends on th32's T6a, whose predicate (mode, size,
`SUPPORTED`) the pre-fill calls, so T6a lands no later than C1 (§F2a I-9
addenda 3 and 6).

**The C1-to-T6 window (addendum 8 (b), §M57 (d)).** C1 writes `STORAGE` for
an eligible row when `taker` is true, and th32 T6 is the only consumer of
`STORAGE`. rev 3c's C1 let a tree holding C1 without T6 load such a table
anyway (r3 m3):
- `add()` rejected the row, so the table became an ordinary input-layer
  weight in `SYCL_Host` (the input layer's CPU buft list starts with the
  first device's host buft, `lm:1766-1776`), which the late inventory keeps.
  The late plan then re-acquired W2's figures (`per_inflight` 103762.3 MB,
  host 154263.6 MB) under default flags, while the early plan excluded the
  table: the two plans disagreed (P4);
- `deferred_for_file` held the row, so `llama_mmap` gave a table read in full
  `POSIX_MADV_RANDOM` and the `WILLNEED` exclusion
  (`src/llama-mmap.cpp:503-545`): 51 GB faulted at 4 KiB with no read-around.

C1 now makes both unreachable on its own: the `STORAGE` branch of
`create_tensor` refuses the load before `init_mappings`, and
`deferred_for_file` excludes `STORAGE` rows (above). **The early and late
plans agree:** on a C1-only tree a load with a `STORAGE` row stops at tensor
creation, so no late plan runs for it, and every other row is `LAZY_READ`
(out of both plans) or `EAGER` (in both). With T6, where the tensor lands is
T6's, and th32 §4.1 keeps a `STORAGE` table out of `max_tensor` and the host
total in the late plan too (§2.6).

The lead's ruling:
- **C1 and T6 reach master in one merge.** T6 is built on top of C1
  (`task/th32` carries or merges `task/5efe`'s C1), and the merge to master
  happens once, after both are reviewed. No tracker edge blocks C1's merge
  on T6: T6 depends on C1, so that edge would close a cycle, and
  `llama.cpp-5efe` has no dependency edge. The one-merge constraint is
  recorded as comments on `llama.cpp-5efe` (`c-q50c`) and `llama.cpp-th32`
  (`c-0elm`), as th32's T6 row states.
- **The merge unit** (r5 m-10; addendum 8 (b), addendum 11 (3)). 5efe C1-C4
  and th32 T6 and T7 reach master in **one** merge. C4's Qwen arms are VOID
  without T6 and T7 (§7 C4 step 2), and C4 depends on th32's scope because
  it travels in that train. So "VOID without T6/T7" is a state of the
  branch, never of master, and nothing here assumes C1 reaches master
  alone.
- The interim-`EAGER` alternative is refused: it breaks addendum 4's iff and
  creates a second source for `deferral`.
- No lead-run gate of C1 reads the table (W1-W3 are killed first, and the
  Mistral gate has no lazy tensor), so C1 needs no Qwen load of its own.
- C4's Qwen arms are VOID without T6 and T7 (§7 C4), and that precondition
  was accepted.

**C1 outputs with no consumer until T6** (§M57 (d)). Each lands with its real
value, and only T6 reads it:
- the `STORAGE` value, whose consumer is T6's body of the `create_tensor`
  branch; C1's body refuses;
- the attributes, computed from hparams (§M58 (c)). T6 reads them and owns
  the missing-attribute refusal, so C1 validates none of them;
- `mode_pre`, which feeds only the taker's bit, and so only `STORAGE` rows;
- `storage_for_file`, whose part of `excluded_for_file` is empty on every
  C1-only load that reaches `init_mappings`.

So the integration branch (`task/th32` carrying C1) is never tested at a
C1-only commit with a Qwen or gemma4 taker load: such a load stops at the
refusal, which is the designed outcome of that commit, not a result.

---

## 4. The design: a cache-owned `FILE_MAPPED` tier (item 3)

### 4.1 The tier and its object

`alloc_tier` gains `FILE_MAPPED`, distinct from th32's `HOST_CPU`:

| tier | memory | reader | charged to the host ledger |
|---|---|---|---|
| `DEVICE_VRAM` | device USM | that device | no (VRAM budget) |
| `HOST_PINNED` | USM host, pinned pool | CPU; a device for DMA staging only | yes |
| `HOST_CPU` (th32) | anonymous, `mlock`ed | CPU only | yes |
| **`FILE_MAPPED`** (new) | one `PROT_NONE` VA reservation with `mmap(PROT_READ, MAP_SHARED \| MAP_FIXED)` file segments over it, made by the cache | **CPU only** | **no**, unless locked (§4.5) |
| `UNREGISTERED` (renamed from `MMAP_TRACKED`) | not an allocation: the query result for a pointer the registry does not know | nobody may allocate it | n/a |

**Enum value.** `FILE_MAPPED` takes the next unused `alloc_tier` value at the
time C2 lands. If th32 T1's `HOST_CPU` has landed, that is 4; otherwise it is
3, and T1 then takes the next one. Every census and test keys on names,
never on values (r1 m-10).

**The raw wrappers.** These live in one new, device-free translation unit,
`ggml/src/ggml-sycl/file-mapping.cpp`, which includes no SYCL headers. Host
tests link it alone (§7).

- `unified_cache_raw_map_file(fd, file_size, reservation, segments)`:
  - takes a **`dup` of the loader's own open fd** (`llama_file::file_id()`,
    `src/llama-mmap.cpp:412-423`), passed down through the hook (§4.3);
  - `fstat`s it and refuses if the size differs from the size the loader
    recorded (`register_file_identity`, `lml:1553-1556`);
  - reserves `[page_floor(first), page_ceil(last))` as `PROT_NONE`,
    `MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE`;
  - maps each segment of §4.6 over the reservation with `MAP_FIXED`.

  Reusing the loader's open file rather than reopening by path removes the
  same-size-replacement gap (r1 m-5). The owner keeps the fd, because
  `POSIX_FADV_DONTNEED` after staging needs it (§4.6), and closes it at
  release.
- `unified_cache_raw_lock_file_segments(base, lock_segments)` and
  `unified_cache_raw_unmap_file(reservation)` are the only lock and release.
  The release is **one `munmap` of the whole reservation** (r1 I-2).
- All three are added to the canonical contract §3 allowlist, next to
  `unified_cache_raw_malloc_host` and th32's `HOST_CPU` wrapper.
- A new source gate, `test-sycl-raw-mmap-allowlist-source`, makes that
  enforceable, which the allowlist entry alone is not (r1 m-11). It fails on
  any `mmap`, `munmap`, `mlock` or `madvise` call in `ggml/src/ggml-sycl/`
  outside a listed site.
  - **It tokenizes first** (r2 m-12). It strips comments, string and
    character literals, then matches `\b(mmap|munmap|mlock|madvise)\s*\(`.
    Comments such as `uc:21652` "host/mmap (conservative)" and
    `tensor-types.hpp:40` "mmap (slowest" would otherwise be flagged by the
    `\s*` form, and a literal `mmap(` would miss a real `mmap (` call.
  - A fixture of those comment and string forms, plus one real `mmap (` call,
    is part of its test.
  - Its list is today's sites (`uch:2066`, `:2071`, `:2107`; `gs:11195`,
    `:11200`, `:11261`; `dpct/helper.hpp:1362`, `:1389`), the new TU, and
    th32 T1's `HOST_CPU` wrapper when it lands.
  - The gate is shared with th32 T1.

**The owner-first front door.**
- `unified_allocate_file_mapping(req)` mints the `alloc_owner_control` first,
  as `unified_allocate_owner` does, then maps.
- The class is **`EXTERNAL_EXACT`**: exact, independently released, and not a
  slice of cache backing (canonical contract §3.1). The two-site
  `CACHE_BACKING` allowlist is untouched.
- There is **one owner and one registry record per reservation**:
  `ptr = reservation base`, `size = reservation length`. The segments are not
  separate records (r1 I-2(c)).

**The registry type.**
- `alloc_type` gains `FILE_MAPPED`. `unified_free_record` handles
  `alloc_tier::FILE_MAPPED` in its own branch, which calls the unmap wrapper.
- That branch sits before the `MMAP_TRACKED`/`UNREGISTERED` no-free case
  (`uc:14812`), the `::free` branch (`uc:14922-14929`, `HOST_MALLOC` after
  §4.8), the `zone_managed` branch and the `sycl::free` fallthrough
  (`uc:14930-14940`). So a `FILE_MAPPED` record can never reach `::free` or
  `sycl::free` (r1 m-4).
- The `::free` branch keys on the **registry type**, which a `FILE_MAPPED`
  record never carries, so moving the `FILE_MAPPED` branch after it would
  change nothing (r2 m-6). C2's test therefore uses mutants that change
  behaviour:
  - the record registered as `HOST_MALLOC`, which reaches `::free`;
  - the branch moved after the `zone_managed` branch, into the `sycl::free`
    fallthrough.

  Private counters on the `::free` and `sycl::free` branches must both read 0
  at a correct release.

**`memory_location::device_mappable()`** (th32 §3.1 adds it) is false for
`FILE_MAPPED`. `host_accessible()` stays true, because the CPU reads it.

**Identity.** There is one reservation per (model, file index, device
buffer), keyed by the file identity the loader already publishes (path and
size, `lml:1553-1556`). Two models mapping the same file get two
reservations over one page cache; sharing them is not needed.

### 4.2 Ownership and teardown

- **Who holds references.**
  - The `SYCL_Mapped` ggml buffer (§4.3) holds the reservation's
    `mem_handle` for the model's lifetime.
  - Every cache entry registered over mapped bytes holds it through
    `storage_owner`. That field is a `std::shared_ptr<void>` (`uch:2514`), not
    a `mem_handle`, so the tier stores
    `std::make_shared<mem_handle>(reservation_handle)` in it. The owner's
    destructor drops the handle, and release still goes through the
    `alloc_owner` (r1 m-18).
  - This replaces today's `non_owning_external_host` registration for these
    entries.
  - Every `CpuExpertPool` task holds a reference through its `weight_lease`,
    which already moves into the worker (`gs:76032`).
- **Order at unload.**
  - `llama_model` frees its buffers.
  - The lifecycle unload releases the model's cache entries **on every
    device** (canonical contract §12).
  - In-flight CPU tasks drop their leases when they finish.
  - The last reference unmaps. Nothing waits for anything, and nothing unmaps
    under a reader.
- **Independent of `llama_mmap`.** The cache makes its own reservation instead
  of adopting the loader's mapping, for two reasons:
  - Adoption would leave the release in llama's hands (P1).
  - `load_all_data`'s final call `munmap`s the head and tail of each loader
    mapping (`lml:2236-2247`), which would pull pages from under a cache entry.

  Two mappings of one file share the page cache, so the cost is address space
  only. Lead ruling §T13c lets the tier do its own exclusion "from your own
  code".
- **The loader's own mapping on the SYCL path.**
  - `init_mappings` is still called, for lazy tables (until th32 lands) and
    for CPU-device buffers.
  - C4 changes its call at `lm:2702` so that a SYCL load passes prefetch 0.
    With `use_mmap` surviving, `init_mappings(true)` would otherwise set
    `prefetch_size = -1` (`lml:1888`), and `llama_mmap` would `WILLNEED` every
    non-lazy byte of every shard (`src/llama-mmap.cpp:535-539`) (r1 m-6).
  - The tier owns the prefetch policy for SYCL weights (§4.6).
- **Shutdown and `PER_DEVICE` caches** (r1 m-18).
  - A `SYCL_Mapped` buffer belongs to one device, like `SYCL_Host` today
    (`gs:42766-42790`). Its reservation is minted by that device's cache.
  - Another device's cache may hold leases on it, through `storage_owner` on
    entries that device registered over the mapped experts (the multi-device
    MoE split).
  - The required order is: **every model unloads, releasing its entries on
    every device, before any device's cache tears down.** The pre-teardown
    census of one cache therefore never sees a lease held by another cache's
    entries.
  - A live `FILE_MAPPED` control at cache teardown is a leaked lease. It is
    reported by the census like any `EXTERNAL_EXACT` control (canonical
    contract §3.1).
  - The `__cxa_finalize` abandon path (`uc:996-1007`) is safe: process exit
    unmaps.

### 4.3 One published fact, the buffer type, and the loader change

**The fact, resolved before the early plan** (r1 I-5, §F2).
- C1 already moves the two caps loops (`lm:2245-2265`) above the early plan
  (`lm:2228-2231`), because the record's pre-fill needs the resolved lazy mode
  (§3.1). They depend only on `devices` and `params`. C4 adds the publication
  below.
- The resolved answer is published once, on the SYCL load transaction:
  `load_maps_files = ml.use_mmap after resolution && the file-mapping tier
  exists in this build`. It is false under `-lm none`, and false when an iGPU
  in the device set forced `none`.
- **Builds without the tier** (r2 m-19). `file-mapping.cpp` needs
  `PROT_NONE` reservations and `MAP_FIXED`, so it is compiled for POSIX only.
  On a non-POSIX build (Windows SYCL) the hook reports the tier absent, and
  `load_maps_files` is false. Those loads keep `SYCL_Host`, and C1's cap never
  routes them to a missing hook.
- The planner reads it to give every inventory entry a `host_backing` (§4.4).
- The loader reads the planner's answer through a new hook,
  `planned_host_backing(name)`. Non-name-indexed tensors such as MoE experts
  get the transaction default, which the planner derived from the same fact.
- `create_tensor` never consults `use_mmap` for a SYCL weight.

**The buffer type.** `ggml_backend_sycl_mapped_buffer_type_for_device(dev)`
returns a per-device buft named `SYCL_Mapped`, with its own `get_name`,
because `supports_buft` keys on the name function (`gs:108985-109018`).

- `is_host` is true, so the CPU backend can execute an op on it when SYCL
  declines (the planner-host decline, `ggml_sycl_op_is_planned_on_host`).
- `supports_buft` is true for SYCL devices, exactly as for `SYCL_Host`
  (`gs:109016-109018`). Ops on device-materialized weights stay on SYCL and
  resolve the cache's VRAM copy by identity, as they do today.
- `set_tensor`, `memset_tensor` and `clear` abort, naming the tensor. The
  mapping is read-only, and weights are never written after load.
- `alloc_buffer` (r2 m-10). The `no_alloc` path calls
  `ggml_backend_buft_alloc_buffer(buft, 0)` for every buft (`lm:2770-2772`),
  so size 0 returns a dummy buffer with no storage. Any other size aborts,
  naming the caller, because a `SYCL_Mapped` buffer is made only by the hook
  below.
- `init_tensor` registers the host weight the way `SYCL_Host` weights are
  registered today (`lml:1542-1561`).

**The hook.** The buffer is created by a new proc-address hook:
`ggml_backend_sycl_mapped_buffer_from_file(dev, file_idx, dup_fd, file_size,
tensor_ranges, lazy_ranges)`, whose `lazy_ranges` is the record's
`excluded_for_file(file_idx)` (§3.1).
- The loader calls it **instead of** `ggml_backend_alloc_ctx_tensors_from_buft`
  for each (`SYCL_Mapped`, file) pair.
- It mirrors upstream's `buffer_from_host_ptr` shape (`lm:2748-2767`).
- `[first, last)` is `get_mapping_range` over the context's tensors
  (`lml:1908-1923`).
- `load_all_data` points each tensor at `base + (weight->offs - page_floor(first))`
  without reading it. It uses its mapping branch (`lml:2099-2128`), keyed on
  the buffer instead of on `ml.mappings`, and skips the loader's `lmlock`
  growth for this buffer (§5.1).
- The one contiguous reservation (§4.1) is what makes that single-base
  arithmetic valid (r1 I-2(b)).

**Contents.** `SYCL_Mapped` holds **every SYCL weight whose host backing is
`FILE_MAPPED`**: device-planned weights (the mapping is their staging source),
host experts, and host dense weights in file layout. On Qwen that is the whole
late inventory, 25231.2 + 102383.5 = 127614.7 MB (`run2.log:2821`), not only
the host experts (r1 I-8b).

**Loader sites C4 changes.** The source contract enumerates all of them
(r1 m-6):

| site | change |
|---|---|
| hook structs `lml:31-51`, `lm:76-117` | add the mapped-buft, `planned_host_backing` and `set_load_maps_files` hooks |
| caps loops `lm:2245-2265` | move above the early plan (`lm:2228-2231`); publish `load_maps_files` |
| `select_weight_buft` `lml:1128-1135`, the `prefer_host_weights` path | `SYCL_Mapped` when `planned_host_backing == FILE_MAPPED`, else `SYCL_Host` as today |
| `weight_buft_supported` `lml:978-985` | its `SYCL_Host` short-circuit also accepts `SYCL_Mapped` |
| `make_cpu_buft_list` `lm:1765-1779` | on a SYCL load that maps files, the front host buft is the device's `SYCL_Mapped`, so `token_embd` lands there |
| `allow_host_buft_with_mmap` `lml:1503-1518` | unchanged for `SYCL_Host`; `SYCL_Mapped` is never demoted |
| buffer creation `lm:2740-2791` | the `SYCL_Mapped` branch calls the hook; the `mlock_bufs` block (`lm:2782-2787`) skips `SYCL_Mapped` buffers, which lock through the tier (§5.1) |
| `lm:2693-2696` | narrowed: it clears `use_mmap` for a SYCL load only when `load_maps_files` is false while `use_mmap` is true, which happens only on a build without the tier (m-19). There the loader keeps today's behaviour instead of copying from a mapping into `SYCL_Host`. C4 amends C1's caps source test, which asserts the line is present (r2 m-8) |
| preload's host-planned copy, `gs:34701-34735` (r2 m-7) | today it copies a weight that the late plan tiers to host, after `create_tensor` put it in a SYCL buffer (the `dev_layer` sync case, `lm:2680-2689`), into a pinned WEIGHT allocation (role `WEIGHT`, category `OTHER`, `memcpy` from `tensor->data`). For a `SYCL_Mapped` source it is served in place when the CPU executes the file layout (host backing `FILE_MAPPED`, registered as a mapped slice). When a reorder is needed, the copy is planned: host backing `PINNED`, or `HOST_CPU` after th32 T1, charged in `weight_host_bytes`. It never copies mapped bytes into memory the plan does not charge |
| `init_mappings` call `lm:2702` | prefetch 0 on a SYCL load |
| `load_all_data` mapping branch `lml:2099-2129` | addresses `SYCL_Mapped` tensors from the buffer base; no `lmlock` growth for them |

With `-lm none`, or any load where `load_maps_files` is false, `SYCL_Host`
remains, with today's behaviour.

### 4.4 Which weights stay mapped, and the planner's split

A byte stays mapped only if **the CPU executes it and its materialized layout
is the file layout** (P3 with layout-follows-residency). Everything else is
copied into a cache allocation at load.

| weight | executor | materialized layout | `host_backing` | tier of the executed bytes |
|---|---|---|---|---|
| device-planned dense or expert | GPU | per-device (SOA, XMX, WOQ...) | `FILE_MAPPED` (its staging source) | `DEVICE_VRAM`, staged once from the mapping (§5.3); the mapped pages are dropped after staging (§4.6) |
| host MoE expert, AOS | CPU (`CpuExpertPool`) | AOS = file | `FILE_MAPPED` | **`FILE_MAPPED`, served in place** |
| host MoE expert needing a CPU reorder (`register_host_aos_expert`'s SOA/XMX branch) | CPU | reordered | `FILE_MAPPED` source | copied: into `HOST_CPU` once th32 T1 lands, pinned until then |
| planner-host dense, `GGML_SYCL_HOST_WEIGHT_LAYOUT=cpu` (default) | ggml-cpu | `CPU_REPACK` | `CPU_OWNED` | copied by ggml-cpu, outside the cache (ticket below) |
| planner-host dense, `=plain` | ggml-cpu | file | `FILE_MAPPED` | `FILE_MAPPED` (`is_host`, so the CPU backend runs it) |
| `token_embd` (CPU `get_rows`) | CPU | file | `FILE_MAPPED` | `FILE_MAPPED` |
| `TENSOR_READ_LAZY` table | CPU / th32 gather | row-gathered | none (not a SYCL inventory entry) | **never mapped by the tier**, nor prefetched or locked (§4.6); th32's STORAGE or `HOST_CPU` |

Under P3 no host **weight** needs pinned memory, because the GPU never reads
one. So in mmap mode the pinned pool holds only staging windows, KV, scratch,
compute buffers, and the reordered experts until th32 T1 lands.
`PINNED` host backing survives only when `load_maps_files` is false.

**Ticket to file:** `CPU_REPACK` weights are allocated by ggml-cpu's
`ggml_backend_cpu_repack_buffer_type` (`ggml-cpu/repack.cpp:5238-5250`),
outside the unified cache and its ledger. That is a P1 gap independent of
mmap (related to `llama.cpp-qptd`). The natural fix is a `HOST_CPU`-backed
repack buffer after th32 T1.

**The planner split** (the lead's point (a); moved into C4 by r1 C-1 and
§F2). `placement_entry` gains `host_backing { PINNED, FILE_MAPPED, HOST_CPU,
CPU_OWNED }` for every entry. The planner derives it once from
`load_maps_files` and the table above, and the plan then carries:

| plan term | sums | charged to the ledger | sizes |
|---|---|---|---|
| `weight_host_bytes` | host-executed entries backed `PINNED` or `HOST_CPU`, **plus the reordered copies** | yes | the WEIGHT zone (`uc:27670-27672`) |
| `host_cpu_owned_bytes` | `CPU_OWNED` entries | yes (anonymous memory, P3 counts it) | nothing: not in the zone (today they inflate it) |
| `host_mapped_bytes` | host-executed `FILE_MAPPED` entries | only when locked (§4.5) | nothing |
| `host_mapped_source_bytes` | device-planned `FILE_MAPPED` sources | no | nothing (transient page cache during staging) |

Every accumulation site of `weight_host_bytes` in the planner
(`uc:25928`, `:25933`, `:28136`, `:28196`, `:28230`, `:28322`, `:29554`,
`:29798`) tests the entry's `host_backing` before adding. The C4 source
contract lists them. `[PLACEMENT] Totals` prints all four terms.

**Who reads `weight_host_bytes`** (r2 m-3). The C4 contract lists the writers
above, and these readers too. Each is assigned one meaning, "charged" or
"placed on host":

| reader | meaning after C4 | why |
|---|---|---|
| `planned_host_bytes` / `actual_host_bytes` on the plan snapshot (`uc:171`, `:228-229`) | **placed**: `weight_host_bytes + host_cpu_owned_bytes + host_mapped_bytes` | it feeds the verdict below |
| `ggml_sycl_current_model_planner_host_placement()` (`gs:2702`), which drives `ggml_backend_sycl_is_tiered_enabled` (`gs:19168`) and the "planner host placement: yes" log (`gs:16737`, `run2.log:2826`) | **placed** | otherwise Qwen would report "no" while 100 GB of host experts execute on the CPU |
| the lifecycle LIVE publication (`gs:13197`, `model-lifecycle.cpp:668`) | both: `ModelState` gains `charged_host_bytes` beside the placed figure | multi-model accounting needs the charge; the tiering verdict needs placement |
| the load summary (`gs:1135-1230`, `:12498`) | prints both, labelled | one number would be read as the other |

**Host-side non-weight requests: routed by role** (r2 I-2; lead ruling: 5efe
owns role-based routing of **host** non-weight requests).

*Today.*
- `select_zone` sends every host request with category `HOST_COMPUTE` or
  `EXPERT_CACHE` to the WEIGHT zone, whatever its role (`uc:15368-15370`).
- `SYCL_Host` buffers allocate with role `WEIGHT` whenever
  `weights_evictable` holds, even after the load (`gs:42650-42651`).
- Those tenants fitted, unseen, in the ~20 GB of 1.2x headroom over 100 GB of
  pinned experts. After the split the WEIGHT zone is 64 MiB. So they would
  grow zone 0 one ≥ 2048 MB chunk per miss (`uc:15418-15437`), which is what
  zone 0 did in run2 (`run2.log:2965`, `:2968`).
- The known tenants are:
  - llama-context's CPU-backend compute buffer, which uses the first
    device's host buft, i.e. `SYCL_Host` (`src/llama-context.cpp:889-895`);
  - the output buffer (`src/llama-context.cpp:3562-3564`);
  - `CpuExpertPool`'s process-lifetime ring, "role=COMPUTE +
    category=EXPERT_CACHE instead routes to the WEIGHT host zone"
    (`cpu-expert-pool.cpp:66-77`), and `PinnedBufferPool`'s ring
    (`pinned-buffer-pool.cpp:69`);
  - the other `HOST_COMPUTE` sites whose role is `STAGING`, `TP_TMP`,
    `COMPUTE`, `GRAPH_TMP` or `OTHER`.

*The sizing fact the reviewer relied on does not hold.*
`ggml_backend_sycl_notify_compute_buffer_sizes` (`gs:19103-19146`), which
would grow SCRATCH by the scheduler's compute-buffer sizes (`:19145`), **has no
caller in the tree**: `git grep notify_compute_buffer_sizes` finds only its
declaration (`ggml-sycl.h:629`) and definition. So the host compute buffer is
not "sized in SCRATCH, allocated in WEIGHT". It is sized nowhere, and
allocated in WEIGHT. The comment at `uc:27716` ("fed back via
`unified_cache_grow_host_scratch_zone()` after the ggml scheduler is
created") describes a path that does not run.

*C4's change.*
- **Route by role, in `select_zone`, the one place.**
  - Role `KV`, or `kv_spill_to_host`, goes to KV, as today.
  - Role `WEIGHT` goes to WEIGHT.
  - Role `EXPERT_STAGING` goes to SCRATCH, as today.
  - A request whose lifetime is the interim `UNOWNED` goes to WEIGHT, where
    it lands today. Only the 12 lines that addendum 10 (b) leaves to
    llama.cpp-30r1 carry it (the site table below).
  - Every other role whose category is `HOST_COMPUTE` or `EXPERT_CACHE` goes
    to **SCRATCH**, per the lead's ruling.
  - Anything else goes to STAGING, as today.
  - Category no longer selects WEIGHT. No call site changes its role,
    except the three host buffer types below; the `DEVICE` and `UNOWNED`
    sites change only their lifetime.
- **`SYCL_Host` takes role `WEIGHT` only inside a model load**
  (`in_model_load`). The `|| weights_evictable` term goes. A post-load
  `SYCL_Host` buffer (compute, output, inputs, and the LoRA and
  control-vector host buffers of CPU layers, whose buft list starts with
  this buft, `lm:1766-1776`) takes role `COMPUTE`, and so SCRATCH.
- **`SYCL_KV_Host` gets its own `alloc_buffer`, with role `KV`** (r3 B; lead
  ruling on r3 B).
  - Today `ggml_backend_sycl_kv_host_buffer_type()` copies the `SYCL_Host`
    buft and overrides only `get_name` (`gs:42755-42763`). So uize-demoted
    host KV goes through `SYCL_Host`'s `alloc_buffer` (`gs:42632-42712`),
    with role `WEIGHT` or `STAGING` (`gs:42650-42651`), cohort
    `backend_host_buffer`, and lands in WEIGHT. Nothing sets
    `kv_spill_to_host` on that path; only the device-KV fallback does
    (`uc:15285-15303`). After the routing above alone, it would take role
    `COMPUTE`, zone SCRATCH and lifetime `GRAPH`: every `graph_compute`
    would trip the SCRATCH boundary WARN (`gs:92661`, `uc:20290-20298`), 6
    GiB would arrive in SCRATCH unannounced, and the KV zone would receive
    nothing.
  - C4 factors the body of `SYCL_Host`'s `alloc_buffer` into one helper that
    takes the intent, and gives `SYCL_KV_Host` its own `alloc_buffer` over
    that helper, with role `KV`, category `HOST_COMPUTE`, cohort
    `kv_host_buffer` and lifetime `CONTEXT`. `select_zone` sends role `KV` to
    the KV zone (zone 1) by its first rule. The buft keeps its own
    `get_name` (the residency decline keys on it,
    `ggml_sycl_tensor_is_in_kv_host_buft`), and every other iface slot stays
    shared.
  - **The qwen4exp indexer cache takes the same path.** `llama_memory_hybrid_idx`
    builds a second `llama_kv_cache`, prefix `idx_`, K only (the MLA trick,
    one head of `indexer_head_size`), over the layers `filter_idx` admits
    (`src/llama-memory-hybrid-idx.cpp:45-67`). qwen4exp's `filter_idx` is
    `!is_recr(il)` (`lm:3580-3585`), the same 12 layers that carry
    attention KV. The per-layer buft choice is the same hook call
    (`src/llama-kv-cache.cpp:290-306`), so a demoted layer's indexer keys go
    to `SYCL_KV_Host` too. At Qwen's default context that is 12 x 262144 x
    128 x 2 B = 768 MiB, which the printed host-KV figure must include (next
    item).
  - The recurrent state is **not** host KV: `llama_memory_recurrent`
    allocates it in the plain device buft (`src/llama-memory-recurrent.cpp:97-103`,
    `llama_recurrent_sycl_kv_buft`). rev 3c's "plus a small recurrent state"
    beside the host-KV figure is withdrawn.
- **The demotion counts the indexer** (lead ruling on r3 B: the indexer is
  in the printed figure).
  - Today the demotion's per-layer bytes come from
    `placement_plan::kv_size_for_layer(l)` (`uch:1033`), at both the demotion
    input (`gs:17089-17092`) and the residency re-fit (`gs:18037-18042`).
    Its per-layer truth comes from `llama_model_sycl_build_kv_layer_arrays`
    (`lm:350-365`) through the inventory's `kv_k_width_per_layer`,
    `kv_v_width_per_layer` and `kv_layer_kind` (`ggml-sycl.h:323-326`). No
    field carries the indexer, so the printed `X MB host KV` omits 768 MiB
    that the load then pins.
  - C4 adds **`uint32_t kv_idx_width`** to `ggml_sycl_tensor_inventory`,
    filled from `hparams.indexer_head_size` when the model's memory is
    `llama_memory_hybrid_idx` and 0 otherwise. Measured with a standalone
    compile against this tree's header: the struct is 176 bytes, 8-aligned,
    with `kv_layer_count` at offset 168, so a `uint32_t` after it lands at
    offset 172, in the tail padding, and `sizeof` stays 176, so it never
    shifts another field (memory:
    extending-a-struct-that-crosses-the-backend-dlopen-boundary).
    - **What is relied on** (r4 n-1). In-tree, `kv_idx_width` is a declared
      member, zeroed by every construction (`lm:653`, `:705`,
      `tests/test-tiered-dispatch.cpp:35`,
      `tests/test-sycl-lifecycle-runtime-wrapper.cpp:1388`, `:1426`, each
      `= {}`), and C4 sets it explicitly at both `lm` sites. That is the
      whole guarantee.
    - For a stale `libllama` the field is tail padding, whose value the
      standard leaves unspecified, so rev 3e's "a stale libllama leaves the
      byte zero" is withdrawn. That pairing is unsupported, as for the
      notify below: the commit message states that both libraries rebuild
      together.
  - One accessor, `placement_plan::kv_demand_for_layer(l)` =
    `kv_size_for_layer(l)` + `kv_idx_width` x the layer's cells x the K
    element size, for every layer the plan counts as carrying KV. Both
    demotion sites read it, and so does `rec.host_bytes`, the printed
    figure. `kv_size_for_layer` itself, which sizes the device per-layer
    slabs, is unchanged: on the device the `idx_` tensors are not remapped
    into a layer slab (`[KV-REMAP]` matches only `cache_k_l`/`cache_v_l`,
    `gs:38575-38583`). How the device accounts for them is outside 5efe's
    scope, and is ticketed as llama.cpp-ka9c (addendum 11 (5); r4 m-11).
  - So the Qwen no-`-c` line prints 12 x (512 + 64) = **6912.0 MB** of host
    KV (§7 C4 step 3).
- **One intent builder per host buft, shared by its `get_max_size` and its
  `alloc_buffer`** (found while folding r3 B; addendum 11 (2); r4 I-A; P4).
  - ggml-alloc splits a buffer into chunks of at most `get_max_size`. Today
    `SYCL_Host`'s `get_max_size` (`gs:42445`) answers from WEIGHT,
    hard-coded (`gs:42482`), on `get_current_device_id()`'s cache
    (`gs:42461`, `:42503`), while its `alloc_buffer` resolves
    `buft->device` (`gs:42639-42642`): two sources for the zone and two for
    the device. The glkg fix made the zone agree with `select_zone` as it
    stands today, and `tests/test-sycl-host-zone-config-source-contract.py`
    pins that agreement. After C4's routing a post-load `SYCL_Host` buffer
    goes to SCRATCH and a `SYCL_KV_Host` buffer to KV, so WEIGHT is wrong
    for both: chunking against the wrong zone's largest free block is the
    llama.cpp-glkg defect again.
  - **rev 3d and 3e misread one buft** (r4 I-A). `gs:42982` is inside
    `ggml_backend_sycl_cpu_offload_compute_get_max_size` (`gs:42952`),
    reached only from the `SYCL_CpuOffloadCompute` iface
    (`gs:43007-43014`), not from `SYCL_HostCompute`. `SYCL_HostCompute`'s
    iface names the **device** buft's `get_max_size` (`gs:42899`, defined
    at `gs:37956`). Its `alloc_buffer` (`gs:42813`) allocates host-pinned,
    role `COMPUTE`, only in TP mode (`gs:42818-42838`), and otherwise
    returns the device buft's `alloc_buffer` (`gs:42889`), a device
    buffer. One host-zone builder for all of `SYCL_HostCompute` would give a
    host ceiling to a VRAM allocation, the glkg defect in the other
    direction.
  - **Which buft's `get_max_size` each design owns** (addendum 11 (2),
    composed with zhcn's rule 1, §M56 (c)). No buft has two:

    | buft | `alloc_buffer` | `get_max_size` after C4 | owner |
    |---|---|---|---|
    | `SYCL<n>` device | `ggml_backend_sycl_buffer_type_alloc_buffer` | `ggml_backend_sycl_buffer_type_get_max_size` (`gs:37956`), with zhcn's in-scope branch | **zhcn** |
    | `SYCL_Host` | `gs:42632` | the host builder (role by `in_model_load`), device from `ggml_sycl_host_buft_device` | **5efe** |
    | `SYCL_KV_Host` | its own (C4) | its own, the host builder with role `KV`; the copied iface (`gs:42755-42763`) goes | **5efe** |
    | `SYCL_HostCompute`, TP mode | host `COMPUTE` (`gs:42818-42838`) | the host builder, role `COMPUTE`, device `buft_ctx->device` (`gs:42819`) | **5efe** |
    | `SYCL_HostCompute`, not TP | the device alloc (`gs:42889`) | delegates to `gs:37956` | **zhcn** |
    | `SYCL_CpuOffloadCompute` | the device alloc (`gs:43009`) | its own STAGING answer (`gs:42952-42982`) | out of scope: the mismatch is **llama.cpp-g2a7** (§M62 (B)) |

  - `SYCL_HostCompute` gets its own `get_max_size`. C4 extracts the TP test
    its `alloc_buffer` makes inline (`gs:42818`) into one function,
    `ggml_sycl_host_compute_is_tp()`, and both call it: in TP mode
    `get_max_size` returns the host builder's `select_zone` answer, and
    otherwise it calls `gs:37956`. So zhcn's branch stays the one answer
    for every device allocation, and 5efe's builder the one answer for
    every host one.
  - Each builder takes the device from the buft, never from the current
    device (r4 I-A, I-D): `ggml_sycl_host_buft_device(buft)` for
    `SYCL_Host` and `SYCL_KV_Host` (the notify, below), and
    `buft_ctx->device` for `SYCL_HostCompute`.
  - The chunk cap (2 GiB, `GGML_SYCL_HOST_BUFFER_CHUNK_MB`, `gs:42426-42443`)
    is unchanged. So Qwen's 6144 MiB of demoted main KV reaches the KV zone
    as three 2048 MiB buffers (24 tensors of 256 MiB, 8 per buffer), and the
    768 MiB indexer cache as one buffer.
- **The notify path, before allocation** (r3 A; addendum 10 (c) accepts the
  entry point).
  - **The existing entry point stays uncalled.**
    `ggml_backend_sycl_notify_compute_buffer_sizes` (`gs:19103-19146`) has
    no caller. It reserves the **VRAM** compute arena and aborts if the
    reserve is short (`unified_cache_reserve_compute_arena`, `:19140-19142`),
    and then grows host SCRATCH by the sum over **every** backend's buffer,
    device buffers included (`:19145`). The lead sends its VRAM half to
    23mk's dead-code census (addendum 10 (c)).
  - **`unified_cache_grow_host_scratch_zone` is dead code, listed for
    deletion** (23mk rev 4.16 relay). 23mk's step 7 deletes
    `ggml_backend_sycl_notify_compute_buffer_sizes` rather than give it a
    caller, and that function is the only caller of
    `unified_cache_grow_host_scratch_zone` (`uc:21935`, declared at
    `uch:6907`, named in the comment at `uc:27716`; its call is
    `gs:19145`). C4's notify does not call it: it is additive and resolves
    device 0 (`uc:21935-21940`), and §M57 (c) needs a high-water on the
    host buft's device (below).
  - **C4 alone deletes it** (§M68 (a); r5 m-5). C4 removes its one call,
    `gs:19145`, from `ggml_backend_sycl_notify_compute_buffer_sizes`,
    deletes its definition (`uc:21935`) and declaration (`uch:6907`), and
    rewrites the `uc:27716` comment to name
    `unified_cache_ensure_host_scratch_capacity`. C4 does **not** delete
    `ggml_backend_sycl_notify_compute_buffer_sizes`: that function is 23mk
    step 7's, and step 7 cites C4. rev 3f's "whichever of C4 and 23mk's
    step 7 lands second" is withdrawn. C4's source gate asserts zero
    callers **and** zero definitions of the symbol, in both states of the
    tree: with step 7 not yet landed, where the dead function is present
    without the call, and with it landed, where the function is gone.
  - **C4 adds a host-only entry point:**

    ```c
    // ggml/include/ggml-sycl.h, C4. Records one llama_context's host buffer
    // demand and grows the host pinned SCRATCH zone to cover it. Returns
    // false if the zone could not grow to cover it.
    GGML_BACKEND_API bool ggml_backend_sycl_notify_host_buffer_sizes(ggml_backend_t backend,
                                                                     size_t         host_compute_bytes,
                                                                     size_t         host_output_bytes);
    ```

    - `ggml_backend_sycl_reg_get_proc_address` (`gs:110040`) exports it by
      name. A direct call does not link under `GGML_BACKEND_DL`.
    - **`backend` is checked, not cast blindly** (r3 A.4). The function
      returns with a WARN unless `ggml_backend_is_sycl(backend)`
      (`gs:107857`). `llama_context` passes exactly one backend, the SYCL
      backend of `model.devices[0]`, from whose device its CPU backend
      takes the host buft (`src/llama-context.cpp:889-895`). The backend
      names the demand's owner; it does not choose the zone's device
      (below).
  - **The ABI shape is deliberate.** It takes scalars, so no struct crosses
    the dlopen boundary and no stride can drift between a fresh `libllama`
    and a stale `libggml-sycl` (memory:
    extending-a-struct-that-crosses-the-backend-dlopen-boundary).
    - It is a new name, not a new signature under the old name. So a stale
      backend returns `NULL` from the proc lookup, and never runs with
      misread arguments.
    - On `NULL`, `llama_context` WARNs `SYCL backend has no
      ggml_backend_sycl_notify_host_buffer_sizes; host SCRATCH is not
      grown`, and continues. The zone criteria below are then the witness.
    - **It returns `bool`** (r3 m5): false when the zone could not grow to
      its target. `llama_context` then WARNs `host SCRATCH could not grow to
      %.1f MB for this context` and allocates anyway; the allocation takes
      the miss path, and the SCRATCH criterion names it. rev 3d's `void`
      return gave `llama_context` no signal.
    - The commit message states that both libraries rebuild together.
  - **When it is called: before each allocation it covers** (r3 A.2). An
    allocation made first fits or grows zone 3 through the miss path
    (`uc:15418-15437`); a notify after it would grow SCRATCH again for the
    same bytes, an idle reservation (P4).
    - `llama_context` keeps its two current terms, `host_compute_bytes` and
      `host_output_bytes`, and every call passes both.
    - **Output buffer.** `output_reserve` computes `new_size`
      (`src/llama-context.cpp:3535-3537`) and frees the old buffer (`:3550`)
      before it allocates (`:3566`). When the chosen buft is the SYCL host
      buft (`:3559-3565`), it notifies with the new
      `host_output_bytes` between those two steps. That covers the first
      reserve in the constructor (`:850`) and every regrowth (`:2896`,
      `:3231`, `:4974`). The term is `new_size`, the bytes the allocation
      asks for.
    - **Compute buffers.** In `sched_reserve`, after the scheduler is created
      (`:2019`) and before the first allocating `graph_reserve` (`:2055`),
      C4 runs the pp graph and the tg graph through `graph_reserve(...,
      /*split_only=*/true, sizes)`, which calls
      `ggml_backend_sched_reserve_size` (`:3908-3910`) and allocates nothing.
      `host_compute_bytes` is the larger of the two sizes at the index of
      the CPU backend, whose buft is the SYCL host buft (`:889-895`). It
      notifies, then runs the allocating reserves as today. The
      pipeline-parallel retry makes a new scheduler (`:2059-2064`) and
      repeats the measure and the notify. The `no_alloc` fit pass already
      takes the size-only path (`:2055-2056`) and does not notify.
    - **The witness.** After the allocating reserves, `llama_context` compares
      `ggml_backend_sched_get_buffer_size` on the CPU backend with the
      notified figure and WARNs `[HOST-ARENA] host compute buffer %.1f MB
      exceeds notified %.1f MB` if it is larger. Plan != reality is then
      printed, not inferred.
  - **What the SYCL side does with it** (r3 A.3, A.5).
    - One demand per `llama_context`, keyed by the notifying backend's
      `ggml_backend_sycl_context`. A new call **replaces** that context's
      demand; it does not add to it. The entry is dropped in that backend
      context's destructor (`gs:43877`, reached through
      `ggml_backend_sycl_free`, `gs:83943`), its owner boundary.
    - **The target is the device the host buft allocates from** (r3 I-1
      (d), §M57 (c); r4 I-D). `llama_context` gets its CPU backend's buft
      from `ggml_backend_dev_host_buffer_type` of `model.devices[0]`
      (`lc:889-895`). The SYCL implementation ignores that `dev` and
      returns the one global `SYCL_Host` buft (`gs:108023-108026`), whose
      `.device` is registry device 0 (`gs:42729-42730`), and `SYCL_Host`'s
      `alloc_buffer` resolves that `buft->device` (`gs:42639-42642`). So
      every host buffer lands in device 0's zones. rev 3e targeted
      `ctx->device`, the notifying backend's device: in per-device cache
      mode (more than one GPU visible, `uc:14292-14317`) with
      `model.devices[0]` not registry device 0 (`-dev SYCL1`, or `-sm none
      --main-gpu 1` under `level_zero:0,1`), it would grow device 1's
      SCRATCH while the buffers land in device 0's, a zone-3 miss. No lead
      arm is multi-device, so only the source gate and the pure test below
      can catch it.
    - **One source for the host bufts' device.** C4 adds
      `ggml_sycl_host_buft_device(buft)`, the device index of
      `buft->device`. `SYCL_Host`'s and `SYCL_KV_Host`'s `alloc_buffer` and
      `get_max_size` read it (today `get_max_size` resolves
      `get_current_device_id()`, `gs:42461`, `:42503`). The notify resolves
      the buft by the call `llama_context` made,
      `ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend))`,
      and grows the helper's device. If `device_get_host_buffer_type` later
      returns the per-device buft (`gs:42766-42791`), all of them follow it
      with no other change.
    - The growth is `unified_cache_ensure_host_scratch_capacity(device,
      bytes)` on that device, and the ledger lives in that device's cache.
      Today's `unified_cache_grow_host_scratch_zone` is not called (above).
    - **Ensure capacity: a high-water mark, never additive** (r3 I-1 (b),
      §M57 (c)). The target is the plan's `host_zone_scratch_bytes` plus the
      sum of that cache's live demands. If SCRATCH's capacity already meets
      it, nothing grows; otherwise the deficit is `target - capacity`.
      `sched_reserve` runs from the constructor, from each auto-ubatch
      candidate and on every runtime `sched_need_reserve` flip (r3 I-1 (b)
      lists the sites, from `lc:1042`), and each run re-notifies: the same
      sizes replace the same demand and grow nothing, and a larger demand
      grows only by the rise. Today's `grow_scratch_zone(additional_bytes)`
      is additive and never shrinks (`uc:20442-20449`), which is why the
      notify does not call it. Outside an auto-`n_ubatch` trial, whose
      losing candidates return their growth (§M75, below), capacity is
      never returned, so a later
      context reuses what an earlier one grew, and one `llama_context` never
      grows the zone once per SYCL backend.
    - The growth is one `grow_zone(SCRATCH, deficit + n x
      DEFAULT_ALIGNMENT)` call, where `n` is the number of buffers the demand
      splits into at the chunk cap, the same slack rule as the miss path
      (`uc:15432`). `grow_zone` commits **one chunk of
      `max(2048 MB, align_up(request))`** (`pinned-pool.cpp:613-712`). So
      the zone grows by at least the deficit, **rounded up to one chunk**,
      and the criteria score it that way.
    - It prints `[HOST-ARENA] SCRATCH demand ctx=%llu compute=%.1f MB
      output=%.1f MB total=%.1f MB grown=%.1f MB` on every call, and the
      existing `[HOST-ARENA] SCRATCH zone grown by %.1f MB for compute
      buffers` (`uc:20447-20448`) when it grows.
  - The "runtime term at the runtime KV update" of rev 3a stays withdrawn,
    since it had no path into the zone.
- **Lifetimes** (addendum 7 (b), addendum 10 (b)).
  - SCRATCH's graph-boundary check runs at every `graph_compute`
    (`gs:92661`), and WARNs on any live SCRATCH record ("refusing SCRATCH
    host_zone_boundary_check with N live registered allocations",
    `uc:20290-20298`). That is exactly why the `CpuExpertPool` ring was
    routed to WEIGHT (llama.cpp-cg8j).
  - C4 adds `alloc_intent::lifetime` with four values, plus the interim
    marker `UNOWNED` described under the site table, which llama.cpp-30r1
    deletes. There is **no `PROCESS`**: a process-lifetime allocation has no
    releasing owner (P2).
    - **`GRAPH`**, the default. The graph-boundary check counts only these
      records.
    - **`CONTEXT`**. The record carries its owner, and the owner's
      boundary is where it is checked (§M57 (b)):
      - **a buffer-owned record** is owned by its `ggml_backend_buffer`: a
        post-load `SYCL_Host` buffer (`gs:42652`: compute, output, inputs,
        and the LoRA and control-vector host buffers), a `SYCL_KV_Host`
        buffer, and the `SYCL_HostCompute` buffer (`gs:42837`). The handle
        lives in the buffer's context and is released in its `free_buffer`
        (`gs:42563` for `SYCL_Host`; P2). The record carries the buffer's
        identity, and its check is **that buffer's free**: once
        `free_buffer` has released the handle, no record owned by that
        buffer may remain.
        - It is not backend-context destruction. `llama_context` destroys
          its members in reverse order: `buf_output` (`lch:429`), then
          `backends` (`lch:403`, reaching `gs:83943` and `gs:43877`), then
          `sched` (`lch:398`, the compute buffers), then `memory`
          (`lch:342`, the KV buffers, host KV included). A check at the
          backend context would see live, correctly owned compute and KV
          records on every context free, and a LoRA adapter's buffer
          belongs to no context at all.
        - So no context serial and no allocation scope is needed. rev 3d's
          use of th32 T6's context allocation scope for these owners is
          withdrawn;
      - **a backend-context member** (`readback_staging_handle`,
        `moe_ids_cache`) carries that `ggml_backend_sycl_context`'s id and
        is checked in `ggml_backend_sycl_free` **after `delete sycl_ctx`**
        (`gs:84119`), once the members are destroyed, not in the destructor
        body (`gs:43877`), which runs before them.
    - **`MODEL`**. The record carries the model's `ModelId`
      (`model-lifecycle.hpp`), whose release is the model-free point. The
      `in_model_load` branch of `gs:42652`, a model's `SYCL_Host` weights
      with role `WEIGHT`, is `MODEL` (r3 m8).
    - **`DEVICE`** (addendum 10 (b), §M62 (D)). It is allowed only for a
      holder whose contents are device-scoped and shared by every context
      on the device. It is not the refused `PROCESS`: it has a named owner
      and a release point.
      - **The intended owner is the last live backend on that device**, and
        `DEVICE` rows are classified against it.
      - **Today's release point is earlier** (r4 I-B). rev 3c-3e said "the
        module teardown". In fact
        `ggml_sycl_shutdown_global_runtime_pinned_owners` (`gs:7575-7592`),
        which shuts down `g_cpu_expert_pools[d]`, `g_attn_host_pools[d]`
        and `g_pinned_buffer_pools[d]` and frees the attention host staging
        (`ggml_sycl_attn_host_cpu_backend_free`, `gs:82182`, which calls
        `ggml_sycl_attn_host_staging_free_all`, `gs:82102`), runs inside
        **every** `ggml_backend_sycl_free` (`gs:84011`), the temporary
        backends of a model load included (`gs:83998-83999`), and in
        `ggml_backend_sycl_shutdown` (`gs:109890`). So a second live context
        on the device loses these holders when the first one's backend is
        freed. That hazard is **llama.cpp-dsab**. Moving the release to the
        last backend per device is dsab's fix, not 5efe's. This corrects the
        "freed at module teardown" premise of §M61 (d)(1); the `DEVICE`
        classification stands.
      - **5efe's checks do not fire on that release.** They run at
        `free_buffer` and after `delete sycl_ctx` (§M57 (b)), never inside
        `ggml_sycl_shutdown_global_runtime_pinned_owners`. The `DEVICE`
        check is the after-`delete sycl_ctx` one, taken only when the
        backend being freed was the last live one on its device: C4 keeps a
        per-device live-backend count beside `g_sycl_backend_refcount`
        (`gs:84121`), and then no `DEVICE` record for that device may
        remain. Today the holders are released before that point, so the
        check passes on every run. It carries weight once dsab moves the
        release there.
      - **The staging slots follow the same count** (§M69 (c); r5 I-2
        (2)). Today every backend free zeroes every device's slots,
        capacity included (`gs:84011` -> `gs:7583` -> `gs:82204` ->
        `ggml_sycl_attn_host_staging_free_all`, `gs:82102-82108`). With
        C4's plan that is no longer self-healing: a surviving context's
        slots would be at capacity 0, and its next host dispatch refuses.
        The rule: with dsab, the slots are released only when the **last**
        backend on their device is freed, by the per-device live-backend
        count C4 adds for the `DEVICE` check, and no backend free zeroes
        another live backend's slots. dsab's fix adopts that count for the
        pools and the staging together; the count is C4's, the release move
        is dsab's.
      - Until dsab lands, the premature release reaches C4's plan only as a
        refusal, which now fails the graph (§M69 (b)), never as wrong
        tokens. It needs the host dispatch on and a SYCL backend freed after
        the commit. `planned 0 B` alone does not show it: an unplanned slot
        (a wrong walk, predicate or device) reads the same. So a refusal is
        dsab's only under the ordered sequence of §4.4's criteria (r6 I-2
        (3)); otherwise it is C4's FAIL. On a single-context arm no SYCL
        backend is freed between the commit and the first dispatch (the
        load's temporary backends go first, `gs:83998-83999`), so there a
        refusal is C4's.
  - **Each lifetime keeps a leak check at its owner's boundary.** A record
    still live at its owner's boundary prints `[HOST-ZONE] leaked
    <CONTEXT|MODEL|DEVICE> record: <cohort> <bytes> owner=<id>`. It is a
    WARN that aborts under `GGML_SYCL_STRICT_LEASES=1`, through the existing
    accessor (`strict_lease_checks_enabled`, `uc:12962-12968`).
  - Every non-`GRAPH` row of the census cites its owner.
- **Sizing: every SCRATCH tenant has a term** (r3 m4). The rerouted
  tenants used to fit only inside the VRAM-RUNTIME terms folded into host
  SCRATCH, which were never sized for them.
  - **Context host buffers** (compute, output): the notified demand above.
  - **Host KV** (`SYCL_KV_Host`): not SCRATCH at all. The KV zone grows at
    context init through the miss path (`uc:15418-15437`), one chunk per
    host-KV buffer that does not fit, and the demotion line prints the bytes
    first. That is outside the PP/TG phase gate (`pinned-pool.cpp:619-640`).
  - **`DEVICE` holders** register a demand keyed by (device, holder)
    through an internal call, `unified_cache_host_scratch_demand(owner,
    bytes)`, on the same ledger as the context demands. Today the two pools
    size their buffers at first `MUL_MAT_ID` initialization
    (`gs:9024-9037`); after C4 they take the same figures from the plan
    (r6 I-3, below). The figures are `max_dispatch_count` (top-K times tokens, over the graph's
    `MUL_MAT_ID` nodes, `gs:8024-8030`), `max_K` and `max_N`
    (`gs:8018-8023`):
    - `PinnedBufferPool`: D x (K + N) x 4 B (`pinned-buffer-pool.cpp:38-39`);
    - `CpuExpertPool`: 4 ring slots x D x (K + N) x 4 B
      (`cpu-expert-pool.cpp:56-59`, `RING_SLOTS = 4`, `cpu-expert-pool.hpp:89`).

    Each `DEVICE` registration prints `[HOST-ARENA] SCRATCH demand
    holder=<name> dev=%d bytes=%.1f MB total=%.1f MB grown=%.1f MB`, where
    `<name>` is `pinned_buffer_pool`, `cpu_expert_pool` or
    `attn_host_staging`, so the zone-3 criterion below sees it beside the
    context demands.
    - **The pools' demand is planned in `sched_reserve`, not at first
      use** (r5 m-7). `moe_hybrid_init_once` runs from `graph_compute`
      (`gs:105477`) after it has set the offload phase (`gs:105352`), so a
      demand registered there would ensure capacity inside PP, under the
      phase gate (`pinned-pool.cpp:625-640`).
    - **And the pools are sized from the plan, not from the first graph**
      (r6 I-3; P4, one source). Today `moe_hybrid_init_once` sizes both
      pools from the first graph it scans, which by default is the 2-token
      all-experts warmup (`llama-graph.cpp:1475`, `common.cpp:1468-1483`).
      Both `init`s are init-once (`pinned-buffer-pool.cpp:29-31`,
      `cpu-expert-pool.cpp:42-45`), while one `MUL_MAT_ID` needs up to
      top-K x `n_ubatch` entries. The code says so itself
      (`gs:75798-75802`), and over-capacity dispatches take per-call
      `EXPERT_STAGING` buffers with a once-per-process `[MoE-CPU] pinned
      expert pool too small` WARN (`gs:75809-75836`). rev 3g planned from
      the reserve graph but let the pools allocate from the warmup: two
      sources that disagree, about 500 MiB planned against 100 MiB held on
      Qwen. rev 3g's claim that the two agree is withdrawn.
    - C4 therefore makes the plan the one source. The plan walk (below)
      scans each device's `MUL_MAT_ID` nodes of the reserve graphs, whose
      `n_tokens` is `n_ubatch`, with the dimension scan of `gs:7952-8030`
      (its `ggml_sycl_host_data(src0)` filter included) extracted into one
      function: `D = max_dispatch_count` (top-K x
      `n_ubatch`), `K = max_K`, `N = max_N`. It records (D, K, N) in the
      device's **pending** plan (below). The commit, never the walk,
      registers the pools' demand, `PinnedBufferPool` D x (K + N) x 4 B and
      `CpuExpertPool` 4 x D x (K + N) x 4 B (r7 I-1 (3)).
    - **The gate is placement's executor fact, never the buffer** (r7 I-2;
      P3, and P4's one fact, one source). A node counts only when the
      planner put at least one expert of its `src0` on the host for that
      device: `ggml_sycl_moe_plan_has_host_experts(src0, device, &known)`
      (`gs:26003-26025`), which asks the placement plan's
      `has_host_experts`. That is the fact the dispatch's CPU-island gate
      routes on for expert weights (`gs:26031-26051`), whose own comment
      says why the buffer cannot be: "the composite tensor's backing buffer
      can be the host mmap source even when every expert is planned and
      materialized in VRAM". Under C4 that is every expert, since
      `SYCL_Mapped` is the source buffer of every SYCL weight on a mapped
      load and is `is_host` (§4.3), so `ggml_backend_buffer_is_host` would
      plan the pools for device-placed experts too, an idle reservation.
      rev 3i gated on the buffer; that gate is withdrawn. `known` comes back
      false for two causes (`gs:26007-26024`): the `src0`'s usage is not
      `MOE_EXPERT_WEIGHT`, or no placement plan exists. The walk then plans
      no pool demand for the node and prints `[HOST-ARENA] DEVICE pools
      dev=%d %s: placement unknown (%s)` (WARN), naming the tensor and the
      cause, `usage not MOE_EXPERT_WEIGHT` or `no placement plan` (r8 n-5).
      Either is a FAIL on any lead arm.
    - **The walk's positive control** (r7 I-2 (3)). Every commit prints
      `[HOST-ARENA] DEVICE pools dev=%d committed D=%d K=%lld N=%lld
      host_expert_nodes=%d` (INFO), including `D=0 ... host_expert_nodes=0`
      when no node counts. So the pools' zero case is scored beside a line
      that proves the pool planning ran, never by absence alone.
    - `moe_hybrid_init_once` then initializes both pools from the device's
      committed (D, K, N), never from its own graph. Its own scan only
      checks that graph against them. If the graph needs more, it prints
      `[HOST-ARENA] DEVICE holder %s dev=%d graph needs D=%d K=%lld N=%lld,
      exceeds its planned D=%d K=%lld N=%lld` (WARN) once per holder, and
      the over-capacity dispatch takes today's per-call path. With no committed plan for the
      device, it prints `[HOST-ARENA] DEVICE pools dev=%d unplanned; not
      initialized` (WARN) and initializes nothing. The warmup's 2 x
      `n_expert` (Qwen 1024, GPT-OSS 64) is within the plan, so on the lead
      arms the check passes. Every dispatch then fits the pool, and the
      `[MoE-CPU] ... too small` cliff is retired.
    - **More than one context: the pools are max-merged** (r7 m-5). The
      device's committed (D, K, N) is the high-water, per field, over the
      live contexts' committed plans, as the staging's slots are. Both
      pools are init-once today (`pinned-buffer-pool.cpp:29-31`,
      `cpu-expert-pool.cpp:42-45`), so C4 gives the commit a re-size: when
      the device's high-water differs from what the pools hold, and the
      pools are initialized, the commit shuts both down and initializes
      them at the high-water, printing `[HOST-ARENA] DEVICE pools dev=%d
      resized D=%d->%d` (INFO). That runs where the staging's re-allocation
      runs, in `sched_reserve` after its `synchronize()` (`lc:2002`), with
      same-device concurrent inference unsupported (canonical contract
      §5), so no dispatch is using the pools. Before the pools' first
      initialization the commit only records; `moe_hybrid_init_once` then
      initializes at the high-water. A context's committed plan drops in
      its backend context's destructor, and the next commit on the device
      re-sizes down to the remaining high-water; the pools' release at the
      last backend on the device is llama.cpp-dsab's per-device count
      (§M69 (c)). The lead arms are single-context, so only the pure test
      below exercises this; `llama-bench -ub a,b` on a MoE model is the
      shipped path it covers.
    - **Pre-registered pool figures, from the reserve graph's `MUL_MAT_ID`
      dimensions** (plan == reality, scored on the same run's `holder=`
      lines against the `[MOE-POOL] Pinned buffer pool` and
      `[CPU-EXPERT-POOL] Initialized` lines):
      - **Qwen** (`llama-completion`, `n_ubatch` 512; `n_embd` 2560,
        `n_ff_exp` 640, top-10, `run2.log:133`, `:30`, `:157`): D = 5120
        and K = N = 2560, the larger of 2560 and 640 over the gate, up and
        down tensors. `PinnedBufferPool` = 5120 x 5120 x 4 B = **100.0
        MiB**, printed `act=51200 KB, out=51200 KB, max_experts=5120`.
        `CpuExpertPool` = 4 x 100.0 = **400.0 MiB**, printed `4 ring slots,
        102400.0 KB/slot (act=51200.0 KB, out=51200.0 KB)`. Qwen's experts
        are placed on the host (run2's `planner host placement: yes`), so
        both are planned, and the pools line reads `committed D=5120
        K=2560 N=2560` with `host_expert_nodes` > 0. These figures hold
        only because the commit carries the pp walk's (D, K, N), not the
        tg walk's D = 10 (r7 I-1).
      - **GPT-OSS** (`llama-cli`, `n_ubatch` 512; `n_embd` 2880,
        `expert_feed_forward_length` 2880, 32 experts, top-4, from the
        GGUF header): D = 2048 and K = N = 2880. If placement puts no
        expert on the host, **0**: the pools line reads `committed D=0 ...
        host_expert_nodes=0`, there is no `holder=` line for either pool,
        and no `[MOE-POOL]` or `[CPU-EXPERT-POOL]` line. Otherwise **45.0
        MiB** and **180.0 MiB** (`act=23040 KB, out=23040 KB,
        max_experts=2048`; `46080.0 KB/slot`), with `host_expert_nodes` >
        0. Which case applies is read from the pools line, which asks the
        same placement fact the dispatch routes on (above), so the case and
        the figures come from one source. Any other figure, or a zero case
        with no pools line, is a FAIL.
  - **The attention host staging is planned from `n_ctx` at context
    creation, when the host dispatch is on** (§M62 (E), §M69 (a); r4 m-4,
    r5 I-1). Today each slot of
    `g_attn_host_staging[dev][kind]` grows in 1 MiB steps whenever a
    request exceeds its capacity (`gs:82037-82065`), and the mask slot's
    request grows with `n_kv` during decode. Under ensure-capacity that no
    longer commits a chunk per step, but a request that crosses committed
    capacity during decode would commit a chunk of at least 2048 MB inside
    the PP/TG phase gate (`pinned-pool.cpp:619-640`).
    - **The plan follows the switch that runs the staging** (§M69 (a); r5
      I-1). The staging's only user is the host attention dispatch
      (`gs:82450-82455`, `:82645-82650`, `:82847-82849`), which is opt-in:
      `ggml_sycl_attn_host_dispatch_enabled()` is false unless
      `GGML_SYCL_ATTN_HOST_DISPATCH` is set (`gs:81901-81917`), and both
      interceptions test it (`gs:83014-83016` for `FLASH_ATTN_EXT`,
      `:83022-83024` for `SET_ROWS`). With it off, demoted-layer attention
      runs on the scheduler's CPU split (`gs:81896-81900`), in the CPU
      backend's compute buffer, which the notify covers. That default is
      placement deciding the executor (P3): host KV, so the CPU runs the op.
      So with the switch off the plan registers **no** staging demand, grows
      no slot, and prints `[ATTN-HOST] staging not planned (host dispatch
      off)` once per process, at WARN. Anything else would be an idle
      reservation (P4).
    - **The plan walk is SYCL-side** (r5 m-6, §M69 (d)), through one
      exported entry point, `bool
      ggml_backend_sycl_plan_host_holders(ggml_backend_sched_t sched,
      struct ggml_cgraph * graph)`. It was
      `ggml_backend_sycl_plan_host_attn_staging(backend, five sizes)` in
      rev 3f, and §M69 (a) names it so; it is renamed because it also plans
      the pools. Its name is added to the proc table
      (`ggml_backend_sycl_reg_get_proc_address`, `gs:110040`), and
      `llama_context` looks it up by name like the notify. A `NULL` lookup
      WARNs and continues. Both arguments are ggml-base handles, and no
      llama type crosses (r6 n-3). The SYCL side keeps neither pointer
      past the call; the graph dies at the next `graph_reserve`
      (`lc:3900`).
    - `ggml_backend_sycl_set_sched_placement_plan(sched)` (`lc:111-116`) is
      a precedent for the argument types only. It is a direct call under
      `GGML_USE_SYCL`, not in the proc table, and its body is a no-op
      (`gs:15847-15854`). Nothing in the tree reads scheduler state from
      the SYCL side yet (r6 I-1, N3).
    - **The walk runs where the split is live** (r6 I-1). The scheduler's
      node-to-backend map exists only between a split and the next reset.
      `ggml_backend_sched_reserve` ends with a reset (`gb:2703`, in
      `ggml/src/ggml-backend.cpp`), which memsets
      `hv_tensor_backend_ids` to -1 and clears the hash set
      (`gb:2665-2675`). So after an allocating `graph_reserve` (`lc:2055`,
      `:2080`, whenever `no_alloc` is false, `lc:3914`),
      `ggml_backend_sched_get_tensor_backend` returns `NULL` for every node
      (`gb:2813-2820`), and a walk placed there plans nothing. rev 3g
      placed it there. `ggml_backend_sched_reserve_size` instead resets at
      its **start** and leaves the split live at its end (`gb:2677-2689`).
      So the walk runs right after each of C4's size-only
      `graph_reserve(..., /*split_only=*/true, sizes)` calls for the pp and
      tg graphs (the notify's measure, above), before the next
      `graph_reserve`. Every node of the graph is then in the live hash
      set, so the lookup inserts nothing. The `no_alloc` fit pass is
      skipped, as the notify skips it.
    - **The size-only split is not the allocating split, and the claim is
      bounded** (r7 m-4 (a)). `reserve_size`'s start reset (`gb:2682` into
      `gb:2665-2675`) erases every pin set while the graph was built:
      llama's `norm`/`l_last` pin when `n_tokens < 32` or under full
      offload (`lc:3985-3997`), and the `!offload_kqv` pin of the KQ nodes
      to the CPU (`src/llama-graph.cpp:2779`).
      `ggml_backend_sched_reserve` keeps them, because it splits without
      resetting first (`gb:2691-2698`). The walk's claim covers only its
      three op kinds. `SET_ROWS` (its destination is a KV view) and
      `MUL_MAT_ID` (its `src0` is a weight) are assigned in the split's
      first pass by buffer (`gb:1639-1651`, `:1683-1696`), which no pin
      affects. `FLASH_ATTN_EXT` is excluded from the weight rule
      (`gb:1675`) and takes its backend by expansion. Under `-nkvo` its K
      is in a `SYCL_Host` or CPU buffer, which the predicate excludes
      anyway (below), and the `norm`/`l_last` pins move no attention node.
      No other node's attribution is read.
    - **The same fact for the notify's measure.** `host_compute_bytes` is
      measured on the size-only split, so on the full-offload lead arms it
      can differ from the allocating split's host compute buffer, in
      either direction. The witness above catches only the under-estimate.
      So at the commit `llama_context` re-notifies each context's demand
      with the allocated size (`ggml_backend_sched_get_buffer_size` on the
      CPU backend), which replaces the measured one, and the demand line
      then prints both, `measured=%.1f allocated=%.1f`. The ledger then
      holds the fact. Capacity an over-estimate already grew stays grown
      (capacity is never returned, above); the SCRATCH criterion's chunk
      rule (§7 C4 step 2) FAILs a chunk that no committed demand line
      explains.
    - **The lifecycle of a pending plan** (r7 I-1). A third proc-table
      entry, `void ggml_backend_sycl_begin_host_holders(ggml_backend_sched_t
      sched)`, runs at the **start** of every `sched_reserve`, right after
      the scheduler is created (`lc:2019-2022`) and before the measure, and
      again right after the pipeline-parallel retry creates its scheduler
      (`lc:2061-2064`). It sits after the creation, not at the
      `synchronize()` (`lc:2002`), because the creation destroys the
      previous scheduler and its compute buffers, which §M75's return
      needs (below). It clears the
      pending plan of every SYCL backend in `sched`, enumerated with
      `ggml_backend_sched_get_n_backends` and `ggml_backend_sched_get_backend`
      (`ggml-backend.h:362-363`, ggml-base). So no backend carries a
      pending plan from an earlier reserve, and a failed candidate's walk
      can never be promoted by a later commit.
    - **The walks of one reserve merge as a maximum** (r7 I-1 (2)). The pp
      walk and the tg walk of one `sched_reserve` merge into the pending
      plan per (device, kind) for the staging, and per device and field
      for the pools' (D, K, N). A walk never replaces what an earlier walk
      of the same reserve recorded. The pp graph dominates: its `n_tokens`
      is `min(n_ctx, n_ubatch)` (`lc:2007`) against the tg graph's `n_seqs`
      = `n_seq_max` (`lc:2080`), which is 1 on every lead arm (r8 n-3), and Q, DST, MASK and the pools' D all grow with
      `n_tokens`, while VAL, IDS, K and N do not depend on it. The maximum
      keeps that true without relying on the call order. A plan taken from
      the tg walk alone would size Q, DST and MASK for `n_seqs` tokens, and
      give Qwen's single-sequence pools D = 10 instead of 5120.
    - **Witness** (P4). With the switch on, a walk that attributes no
      host-KV attention node to a SYCL backend prints `[ATTN-HOST] plan
      attributed 0 host-KV nodes to a SYCL backend (%d matched the
      predicate)` (WARN). A dispatch-on arm with host KV that prints it is
      a FAIL of C4.
    - **Pending, then committed** (r6 I-1 (2), m-5; r7 I-1 (3), m-1). A
      walk records into the *pending* plan of each SYCL backend context it
      attributes nodes to, keyed like the notify's demand, by that
      `ggml_backend_sycl_context`. `llama_context` commits it through a
      second proc-table entry point, `bool
      ggml_backend_sycl_commit_host_holders(ggml_backend_t backend)`, once
      per SYCL backend. The commit consumes the pending plan, and it is the
      only step that registers demand, for the staging and the pools
      alike.
    - **Where the commit runs** (r7 m-1). `sched_reserve` commits at its
      end, after the allocating reserves succeed, for every caller except
      the auto-`n_ubatch` trial. The trial's `try_candidate` commits
      instead, and only after its host-fallback check passes
      (`lc:1622-1629`). So a candidate whose allocating reserve throws
      (`lc:1607-1619`) or whose compute buffer falls back to host commits
      nothing: it grows no slot, re-sizes no pool and ensures no SCRATCH
      for either, and the next `sched_reserve`'s begin clears its pending
      plan. The trial's settle step re-reserves the chosen candidate
      whenever the scheduler does not already match it (`lc:1931-1940`),
      so the last commit is always the chosen candidate's. There is no
      exception to this.
    - **A losing candidate's SCRATCH growth is returned** (§M75). The
      notify must ensure SCRATCH before the allocating reserve it covers
      (above), so a trial candidate's notify can grow SCRATCH for its own
      host compute buffer before the candidate is known to win. That
      growth follows the pending plan's lifecycle: it is charged to the
      candidate, kept only if the candidate wins (its `sched_reserve`
      succeeds **and** its host-fallback check passes, `lc:1622-1629`),
      and returned if it loses. After the ladder, SCRATCH capacity is the
      winner's need plus the other live contexts' demand, chunk-rounded as
      every SCRATCH figure is, and never the losers' high-water (P4).
      "Capacity is never returned" (above) holds for every other caller,
      and no longer for a trial candidate.
    - **The transaction.** A fourth proc-table entry, `bool
      ggml_backend_sycl_host_scratch_trial(ggml_backend_t backend, int
      op, uint32_t n_ubatch)`, with `op` one of `OPEN`, `KEEP` and `RETURN`
      (plain `int` constants, since no enum crosses the dlopen boundary)
      and `n_ubatch` the context's, which the witness below prints, called
      on the backend the notify is made through. `try_candidate` calls
      `OPEN` right before its `sched_reserve` (`lc:1607`). An open
      transaction records the thread that opened it, and a chunk that zone
      3 grows, by the notify's ensure or by the miss path alike, is
      recorded in the device's open transaction of the growing thread
      (canonical contract §5: the reserving thread is the allocating
      thread); a grow that no open transaction of its thread matches is
      ordinary capacity. On a
      win, `try_candidate` calls `KEEP` together with the commit, after the
      host-fallback check; the chunks become ordinary SCRATCH capacity. On
      each loss path, the allocating reserve's throw (`lc:1609-1619`) and
      the host fallback (`lc:1624-1628`), it calls `RETURN`.
    - **The marked set is the device cache's, and a marked chunk is
      withdrawn at once** (r8 m-1). `RETURN` drops the context's demand,
      moves the transaction's chunks into a **per-device marked set** held
      by that device's unified cache, and **withdraws** each from zone
      allocation. Withdrawal is one per-chunk fact, a `withdrawn` flag on
      the chunk's `zone_chunk_state`, and every zone reader derives from
      it (r9 m-3, P4):
      - withdrawal is the **one** decrement site of the zone's capacity:
        it subtracts the chunk's usable size from `zones_[zi].size`, which
        `zone_capacity` returns (`pp:583-589`), **only on the flag's
        false-to-true transition** (r10 m-4), so a second withdrawal of
        one chunk changes nothing. The release never changes
        `zones_[zi].size` again;
      - `zone_alloc` (`pp:510-521`) and `zone_alloc_segmented` skip a
        withdrawn allocator;
      - `zone_used` (`pp:570-581`) sums only the allocators that are not
        withdrawn. So `zone_alloc_segmented`'s early check (`pp:438-445`),
        which compares that sum with `zones_[zi].size`, compares like with
        like, and a loser's live buffer in a withdrawn chunk refuses no
        request the other chunks can serve;
      - `zone_largest_free_block` (`pp:594-611`) skips a withdrawn
        allocator;
      - frees into a withdrawn chunk still reach its allocator, so the
        loser's buffers can leave it.

      The marked set is independent of any transaction: a later `OPEN`
      never reads or clears it. Two windows run
      code between `RETURN` and the begin that releases: on the tuning
      cache path, a cached candidate that loses (`lc:1750`, `:1774-1776`)
      is followed by a ladder candidate's probe (`lc:1516`), publish
      (`lc:1573`) and `OPEN`, all before its `sched_reserve`; and the
      settle's publish (`lc:1936`) runs before the settle's begin. Neither
      can land an allocation in a marked chunk, and neither can drop the
      mark.
    - **When the return runs.** A returned chunk that is already empty is
      released by the return itself (below). One that is not is released
      later: at `RETURN` the losing candidate's scheduler still holds its
      host compute buffers in it, and releasing memory that a
      live handle still owns is forbidden (r9 n-8). Every loss clears
      `sched_matches_last_good` (`lc:1619`, `:1627`), so a `sched_reserve`
      always follows, for the next candidate or for the settle
      (`lc:1931-1940`). Its scheduler creation (`lc:2019`) destroys the
      losing scheduler and frees those buffers, and the begin entry, which
      runs right after it, releases every chunk in the device's marked set
      whose allocator is empty, before the measure and the notify. A
      withdrawn chunk is never counted, so no later ensure sees it either
      way.
    - **Every withdrawn chunk has an owner and a bounded release** (r9
      m-5, r10 m-3, P4). Each marked-set entry records the SYCL backend
      whose transaction returned it, the context's. A returned chunk is
      released at whichever comes first:
      - **the return itself, if the chunk's allocators are already
        empty**: `RETURN` and `RETURN_TO_MARK` apply the begin's rule at
        once, so only a chunk that still holds a scheduler's buffers stays
        withdrawn;
      - the owning context's next begin, if its allocators are empty then;
      - its owner's end. `ggml_backend_sycl_free` (`gs:83943`) runs the
        owner-end step before `delete sycl_ctx` (`gs:84119`): it releases
        each of its owner's entries whose allocators are empty and that
        hold no lease, and sets `orphaned` on the rest. The free that
        empties an orphaned chunk releases it, and so does the drop of its
        last lease (r17 M-4).

      **Where the two flags live** (r10 n-2). `withdrawn` and `orphaned`
      are the only per-chunk states, and both live on the chunk's
      `zone_chunk_state` in the pinned pool, their one home. The device
      cache's marked set holds only each chunk's owner, which no free
      reads. `zone_free` already finds the chunk's `zone_chunk_state`
      before it frees (`pp:536-541`); after the free it tests `orphaned`
      and the allocator's emptiness on that same entry, a flag test with
      no cache lookup, and returns the chunk's index to its caller, which
      releases it after the pool's lock is dropped. A free into a chunk
      that is not orphaned tests one flag and does nothing more, so
      dispatch-path frees gain no lookup.

      **Every release removes the chunk's marked-set entry** (r11 m-5).
      The marked set is the device cache's, and every release of a marked
      chunk goes through one cache-side function,
      `release_marked_chunk(dev, chunk_idx)`: it calls
      `release_zone_chunk` and erases that chunk's entry from the set in
      the same step, and no code calls `release_zone_chunk` for a marked
      chunk any other way. It has two forms (r14 M-2), and the form is
      chosen by what the caller holds (r15 M-1): **a caller that already
      holds `marked_set_mutex_` calls the core and never the public form**,
      because the lock is a plain mutex and the public form takes it. The
      core, `release_marked_chunk_locked(dev, chunk_idx, out)`, is called
      with `marked_set_mutex_` held. It calls `release_zone_chunk` and acts
      on the status that returns (below). On a release it erases the entry
      and moves the returned `released_chunk` into `out`: a vector the
      caller owns and has reserved (the walks) or, in the core's
      **single-slot overload**, an inline `released_chunk` with no vector
      at all (the public form; r18 M-1). The public `release_marked_chunk(dev,
      chunk_idx)` is for a caller that holds neither the set's lock nor the
      pool's: it holds one inline `released_chunk` slot (no vector and no
      allocation; r18 M-1), takes the lock, calls the single-slot core,
      drops the lock, and lets the slot's carrier release
      (`release_chunk_backing`, below). Four callers hold the set's lock
      and use the core, each in **one scope** that collects into a local
      vector reserved for its candidate count and releases every collected
      backing after the scope closes:
      - the begin's walk;
      - `configure_host_zones`'s release of the empty marked chunks;
      - the owner-end walk (`ggml_backend_sycl_free`'s step): for each
        entry of its owner, the core with `orphan_if_live` set (r16 M-4).
        The pool call decides under one hold of its `mutex_`: it releases
        the chunk if its allocators are empty and no lease is held, and
        otherwise (a live allocation is checked first, then a lease) sets
        `orphaned` and returns ORPHANED. A free cannot slip between the
        check and the flag, so an emptying free either sees `orphaned` and
        releases the chunk, or the walk saw the chunk empty and released
        it. An orphaned chunk is then released by whichever comes last of
        the free that empties it and the drop of its last lease (below),
        so no begin is needed;
      - `RETURN` and `RETURN_TO_MARK`: the filing (made after the pool's
        lock has dropped, since withdrawal itself runs under it) and the
        immediate release of the chunks that are already empty share one
        scope, so an empty chunk is never visible in the set between its
        filing and its release.

      Two empty chunks released by one begin, or by one owner's end, are
      both freed after the lock. The **callers of the public form** are
      the orphaned chunk's two release events (r17 M-4): the emptying
      free, where `zone_free` hands the chunk's index back to its caller,
      the cache's host-zone free path, and the last lease's drop, where
      `release_chunk_lease` hands it back to `host_release_chunk_lease`
      (below). Each holds neither lock and calls the public form after the
      pool's lock drops. So a released chunk is a tombstone in `chunks_`
      and absent
      from the marked set at the same moment, and no entry for a tombstone
      survives for a later begin to meet; its backing footprint leaves
      `total_allocated_` when the memory is freed (below). A begin's walk
      also skips an entry whose chunk is already a tombstone, with no
      print, so a stale entry could not become a null release or a bogus
      `SCRATCH return refused`.
      **The set's lock and the lock census** (r12 M-5; r13 M-2, M-3; r14
      I-1, M-4, M-5; §M81 (c), (d)). The orphaned-free caller runs on the
      thread that frees the scheduler's buffer, while another context's
      begin walks the same set, and both can pick the same empty chunk. So
      **every access to the set holds one per-device lock of the device
      cache, `marked_set_mutex_`**: the filing at `RETURN` and
      `RETURN_TO_MARK` (made after the pool's lock has dropped, since
      withdrawal itself runs under it), the begin's walk, the owner-end
      walk by owner, `release_marked_chunk_locked`'s erase, and the test
      accessor `marked_entries`.
      - **Census** (canon §12.5, where an unclassified lock is a failing
        census item): `marked_set_mutex_` is **L4**, cache registries and
        metadata, last in L4's ordinal list. `pinned_chunk_pool::mutex_`
        (`pinned-pool.hpp:313`), which the table does not list, is **L5**,
        placed **first** in L5's list, before `vram_zone::alloc_mutex`:
        `grow` already nests `g_runtime_alloc_mutex` under it (`grow_into`
        to `allocate_pinned_chunk_owner` to
        `unified_allocate_owner_backing`, `pp:939`, `uc:15578`), and that
        nesting is conforming under this row, and so is the reconcile's registry
        query (below). Any nesting under the pool's `mutex_` other than
        those two is pre-existing, tracked as llama.cpp-uilv, and not
        C4's. C4 adds both rows to canon §12.5 in the same commit, **and a
        third for `deferred_frees_mutex_`** (r15 M-2 (c)): the order
        argument below depends on nothing under `marked_set_mutex_` or the
        pool's `mutex_` taking it, and `process_deferred_frees_locked`
        holds it around owner releases (`uc:12667`) while the enqueue paths
        take it (`uc:12453-12557`), so it must be classified. It is L4,
        listed after `rw_mutex_` and before `marked_set_mutex_`; its exact
        ordinal among the other L4 locks is decided in that commit, by
        its author against the enqueue paths and reviewed there with the
        two rows above, and the gate reads it from the table once it is
        there. The
        canon's retained-handle-store row and its "last in L5" paragraph
        exist only at `d8a67422d` (canon:1458, :1502-1525, added by
        llama.cpp-uwlx); this doc does not rely on them. The order is L4
        then L5.
      - **Who takes `marked_set_mutex_`, and what each holds, derived from
        the callers at e2461d4fb** (r14 I-1). Only a caller that gets a
        chunk index back, or that releases marked chunks by design, takes
        it. The `orphaned` flag lives only on SCRATCH `zone_chunk_state`s,
        so only a SCRATCH free returns an index.
        - The begin entry and the trial's `RETURN`/`KEEP` entry are called
          from `sched_reserve` and `try_candidate`, with no cache lock.
        - The owner-end step runs in `ggml_backend_sycl_free` before
          `delete sycl_ctx` (`gs:84119`); `g_pipeline_copy_queue_mutex`
          (`gs:83957`) is a block-scoped guard that closes at `gs:83968`,
          so none is held. The planner's `backend_guard` (`gs:16818`)
          calls the same free with no census mutex held (its `module_guard`
          is a counter).
        - `configure_host_zones` is reached through
          `ggml_sycl_configure_host_zones_for_plan`. Its callers at
          e2461d4fb: the guarded path (`gs:16850`), after the inventory
          lock's scope has closed (`gs:16843-16851`); `gs:33366`, inside
          `ggml_sycl_preload_moe_experts` (`gs:33353`), a static function
          with no caller, so **not a production caller**; and `gs:33893`,
          the only other live site, inside
          `ggml_sycl_preload_model_weights` (`gs:33520`), reached from
          `ggml_sycl_model_loading_effects` (`gs:12476`) and
          `model_load_end` (`gs:13124`), which hold a
          `sycl_module_mutation_guard` (a counter, `gs:12234-12255`) and
          registry effect scopes, none a census mutex; its only
          `lock_guard` (`gs:33547`) is in a closed block.
        - The orphaned free, stated at its callers (r15 M-3). SCRATCH
          frees reach `host_zone_free` through `unified_free_record`
          (`uc:14806`), from (i) `release_registered_allocation_owned`'s
          detached release (`uc:16423`), whose own lock block
          (`uc:16373-16413`) has closed; it is reached from the registry
          release paths and from `process_deferred_frees_locked`
          (`uc:12689`, `:12714`, `:12748`, `:12760`), which runs under
          `rw_mutex_` (unique) and `deferred_frees_mutex_` (`uc:12667`);
          (ii) `unified_cache::deallocate` (`uc:20672`), whose only caller
          is the free wrapper `unified_cache_deallocate(int, void *,
          size_t, lifetime)` (`uc:21288`), and that wrapper has no caller
          in `ggml/` or `tests/` at e2461d4fb (the two-argument overload a
          policy fixture uses is a different function), so this path has no
          production caller today; (iii) the allocation-rollback calls
          (`uc:15558`, `:15572`, `:15601`), made by the allocation
          functions after their own `g_runtime_alloc_mutex` block (it
          closes at `uc:15599`); the callers of those allocation functions
          are many and are not enumerated here. A rollback record can lie
          in a withdrawn chunk, so this is a **lock-state** claim, not an
          exclusion. **For (i) and (iii) the rule stands in place of a
          census of callers**, because the gate sees bodies and not
          callers: a SCRATCH final-owner destruction, or a SCRATCH
          allocation that can roll back, made while the caller holds an L5
          lock is non-conforming under canon §12.5 (a final `mem_handle`
          destruction under a listed lock, canon:1612-1619 at e2461d4fb)
          and would put `marked_set_mutex_` (L4) under an L5 lock. Any such
          site is pre-existing and outside C4, and belongs to
          llama.cpp-uilv with the pool's other nesting; C4 adds no site,
          and the order below covers every site that follows the rule.
        - **The deferred-free path and SCRATCH.** No producer enqueues a
          SCRATCH host allocation on the deferred lists today:
          `enqueue_deferred_free` is fed by weight-cache and expert-stage
          slots (`uc:6155`, `:6651`, `:6962`, `:12368-12369`, `:12412`,
          `:13056`), DMA staging slots (`uc:13885`, `:13892`) and the
          oneDNN scratch replacement (`uc:17646`), and `defer_host_free`
          has no caller. The order below nevertheless covers the case: if a
          SCRATCH owner ever reaches `process_deferred_frees_locked`,
          `marked_set_mutex_` is taken under `rw_mutex_` and
          `deferred_frees_mutex_` (the latter is not in canon §12.5's
          table today; C4's census commit adds it, above).
        - **The WEIGHT frees** reach `host_zone_free` with caller-held
          locks and never return an orphaned index, so they never take
          `marked_set_mutex_`; they are listed because they free through
          the same function. `release_entry_allocation_locked` has 15 call
          sites at e2461d4fb (16 grep hits, the 16th being its definition
          at `uc:12345`). `uc:4925` (`shutdown_resources`, whose
          production caller is `~unified_cache`, `uc:4639`, with no lock
          held) takes `rw_mutex_` nowhere and holds no cache lock. These
          take `rw_mutex_` (unique) inside the
          function: `ensure_cached` (`uc:5648`, lock `:5430`),
          `direct_stage_weight` (`:6243`, `:6209`), `direct_stage_expert`
          (`:6589`, `:6750`, `cache_lock` `:6552`, `:6718`),
          `direct_stage_expert_tensor` (`:7045`, `:6982`),
          `drop_expert_entry` (`:7558`, `:7506`),
          `drop_expert_entries_for_tensor_layout` (`:8029`, `:7998`),
          `allocate_slot` (`:8678`, `:8637`), `remove` (`:9869`, `:9828`),
          `note_buffer_owner_dead` (`:13128`, `:13097`) and
          `ensure_cached_alloc` (`:20067`, `:19910`);
          `finalize_evictions_locked` (`:10710`) is called under
          `rw_mutex_` from `finalize_evictions` (`:10746`) and from
          `process_deferred_frees_locked` (`:12660`); `evict_one`
          (`:10626`) and `finalize_retired_entries_locked` (`:12624`)
          take it from their callers by contract only.
      - **The order.** `rw_mutex_` (and `deferred_frees_mutex_` where
        held) then `marked_set_mutex_` then the pool's `mutex_`. So
        `marked_set_mutex_` may be taken under `rw_mutex_` and other L4
        locks ranked before it, never under an L5 lock. Nothing under
        `marked_set_mutex_` or the pool's `mutex_`, in the functions C4
        touches, takes an L4 lock (the gate pins it), so no cycle exists.
        `release_marked_chunk_locked` holds `marked_set_mutex_` and takes
        the pool's `mutex_` inside `release_zone_chunk`, and nothing else.
        The pool's `mutex_` is a leaf **for these functions only**,
        because the owner moves out of the pool's scope (the release
        bullet above) and is released after both locks drop. No function
        that holds the pool's `mutex_` calls `release_marked_chunk` or
        takes `marked_set_mutex_`. The pool's other nesting is unchanged
        and is not C4's (`configure_zones` calls `grow`, and the
        destructor frees chunk owners, with `mutex_` held,
        `pp:309-328`, `pp:141-162`).
      - **The owner's release can run under a caller's L4 locks** (r14
        I-1 (c)). `release_chunk_backing` runs after both new locks drop
        but not after the caller's: a free arriving through
        `process_deferred_frees_locked` runs it under `rw_mutex_` and
        `deferred_frees_mutex_`. That is legal by the order above and by
        a non-taking property: the owner's release for a host-pinned
        record takes `g_runtime_alloc_mutex` in
        `release_registered_allocation_owned` and then frees
        (`uc:14932-14946`), and takes none of `rw_mutex_`,
        `direct_stage_mutex_` or `deferred_frees_mutex_`, so no cycle and
        no self-deadlock. It is pre-existing in kind (a final-owner
        destruction under a listed lock, canon 1612-1619) and no worse
        than today's frees through those callers.
      - **The state re-read** (r13 M-3). A chunk's state lives in
        `chunks_` and `zone_chunk_state`, which the pool's `mutex_`
        guards, so the re-read is a pool call: `release_zone_chunk` takes
        the pool's `mutex_`, finds the chunk and its flags under it, and
        returns a **status** (r15 I-1; r16 M-4), of type
        `pinned_chunk_release`. `release_marked_chunk_locked` takes
        ALREADY as no effect: no print and no second erase. Only DONE
        appends to `out`. A refusal leaves the entry in the set. The
        begin's tombstone skip uses the same pool query. The `released`
        and `return refused` lines print after both locks are dropped.
        **What each caller does with each status** (r16 M-4):

        | caller | DONE | ALREADY | REFUSED_LIVE | REFUSED_LEASE |
        |---|---|---|---|---|
        | begin's walk | erase, collect, `released` line | skip, silent | silent for another owner's chunk, entry kept (a live scheduler buffer; its owner's end or a later begin releases it); the owning context's own begin prints `SCRATCH return refused dev chunk reason=live` (r10 m-3) | silent for another owner's chunk, entry kept, retried at the next begin; the owning begin prints the line with `reason=lease` |
        | `RETURN` / `RETURN_TO_MARK` | as the begin | skip, silent | `SCRATCH return refused dev chunk reason=live`, entry kept (the chunk was judged empty at the check, so it is reported by name) | the same line with `reason=lease`, entry kept, retried at the next begin |
        | owner-end walk (`orphan_if_live`) | as the begin | skip, silent | returns ORPHANED: `orphaned` set under the same hold, released by the emptying free or the last lease's drop | returns ORPHANED too (live is checked first, then lease): released by the last lease's drop, or by the emptying free if that comes last |
        | `configure_host_zones` | as the begin | skip, silent | the chunk stays withdrawn, so `configure_zones` refuses by name and returns false (clause 6's strict read applies) | as REFUSED_LIVE |
        | orphaned emptying free (public form) | as the begin | silent (a begin released it first) | cannot occur: the free just emptied it; if it does, silent, and a later free offers it again | silent, entry kept; the last lease's drop releases it |
        | last lease's drop (public form) | as the begin | silent | silent, entry kept; the emptying free releases it | cannot occur: this is the last lease; if it does, silent |

        A lease on a chunk whose allocators are empty is transient: a
        raw-pointer lease outlives its allocation by the dispatch's own
        window. For the begin, `RETURN` and configure rows its retry point
        is the next begin of the live owner. For an orphaned chunk it is
        the lease's own drop (below), so a dying owner never leaves a
        chunk withdrawn until pool destruction.

      **A begin and a live owner** (r10 m-3). A begin releases every empty
      chunk in the device's marked set, whatever its owner. It leaves a
      non-empty chunk whose owner is another live context, and prints
      nothing: that chunk holds the other context's scheduler's buffers,
      as after a runtime throw (below). `SCRATCH return refused` prints
      only at the **owning** context's own begin, after its scheduler
      creation has freed its buffers, when the chunk is still not empty.

      **Why the owner's end needs the free path.** The owner's end comes
      before its scheduler's. `llama_context` declares `sched` before
      `backends` (`src/llama-context.h:398`, `:403`), so its members are
      destroyed backends first, and `~llama_context` (`lc:1066-1107`)
      does not reset `sched` earlier. A constructor that throws
      (`lc:2068`, `lc:2082`, reached from the constructor's
      `sched_reserve` at `lc:1042` or the trial at `lc:1040`) runs the
      same member destruction, and no `sched_reserve` follows it. So when
      the backend is freed the scheduler's compute buffers can still sit
      in the chunk; they leave it when `sched` is destroyed right after,
      and that free releases it.

      **The runtime throw** (r10 m-3). `sched_reserve` also runs from
      `decode` (`lc:3178`) and `encode` (`lc:2891`), and clears
      `sched_need_reserve` (`lc:1998`) before it can throw (`lc:2068`,
      `:2082`). `llama_decode` has no catch (`lc:5733-5742`), but a caller
      that catches, such as the server, keeps the context, and nothing
      flags another reserve. The own transaction's `RETURN` then releases
      every empty chunk at once. A chunk that holds the failed scheduler's
      buffers stays withdrawn while they live: it is in use, not idle, and
      is released at that context's next begin or at its end. Another
      context's begin meanwhile leaves it silently.

      No chunk stays withdrawn until the pool's destructor, and none is
      released while a live handle owns memory in it.
    - **The pipeline-parallel retry is the same shape, and gets the same
      return** (r8 n-6). When the pp reserve fails with pipeline
      parallelism on, `sched_reserve` replaces the scheduler and measures
      again (`lc:2055-2065`), so the first measure's notify may have grown
      SCRATCH for a scheduler that is then discarded. `sched_reserve`
      therefore brackets its measure with two more `op` values (r9 n-6):
      - `MARK`, before the measure, opens `sched_reserve`'s own
        transaction when no trial transaction is open, and records the
        open transaction's chunk count;
      - `RETURN_TO_MARK`, before the retry, **truncates the transaction to
        the mark** (r10 m-4): the chunks grown since it leave the
        transaction's set and are withdrawn and marked, or released at
        once if empty, and the transaction stays open. A later `RETURN`
        never sees them, so it cannot withdraw one twice, and a later
        `KEEP` makes only the chunks still in the set ordinary capacity; it
        never clears a `withdrawn` flag.

      So the retry's measure and reserve run inside a transaction in both
      cases: `sched_reserve`'s own, which it `KEEP`s on success (the
      re-`OPEN` of r9 n-6, with no gap between), or the trial's, which
      `try_candidate` keeps or returns as before, now without the
      discarded first measure's growth. The retry's scheduler creation
      (`lc:2061`) frees the first scheduler's buffers, and its begin
      releases the chunks. A `sched_reserve` that throws with its own
      transaction open `RETURN`s it, and the owner's end (above) releases
      the chunks when no begin follows. "Capacity is never returned" now holds only for
      growth a `KEEP` has made permanent.
    - **The support gap it closes.** The pinned pool cannot shrink a zone
      today. `grow_zone` appends a chunk to `chunks_`, a span to
      `flat_spans_` and a TLSF allocator to the zone's
      `zone_allocators_` (`pinned-pool.cpp:613-712`), and chunks are freed
      only in the pool's destructor (`:141-162`). C4 adds
      `pinned_chunk_pool::release_zone_chunk(host_zone_id zone, size_t
      chunk_idx, released_chunk & out, bool orphan_if_live = false)`, which
      returns a `pinned_chunk_release` status (r15 I-1, r16 M-4; below)
      (r8 n-4 closes four gaps in rev 3k's text):
      - it requires the chunk's zone allocator to be wholly free, **and**
        the chunk's own per-chunk `allocator` (`pinned-pool.hpp:246`) to be
        unused, and its lease count 0;
      - it requires the chunk to be withdrawn, removes its span and zone
        allocator, and does **not** change the zone's capacity, which the
        withdrawal already reduced (r9 m-3). It does **not** touch
        `total_allocated_` either (r14 M-3): the pool's footprint counts
        the chunk's **backing** size, usable plus the `DEFAULT_ALIGNMENT`
        header that `grow_into` commits (`chunk_footprint_for`,
        `pp:798-803`), a different fact from the zone's capacity, and it
        keeps counting it until the memory is physically freed (below);
      - **the status** (r15 I-1; r16 M-4, N-4): `release_zone_chunk`
        returns a `pinned_chunk_release`, one of `PINNED_CHUNK_RELEASE_DONE`,
        `PINNED_CHUNK_RELEASE_ALREADY` (the chunk is a tombstone),
        `PINNED_CHUNK_RELEASE_REFUSED_LIVE` (the zone's or the chunk's
        allocator is not empty), `PINNED_CHUNK_RELEASE_REFUSED_LEASE` (lease
        count above 0) and `PINNED_CHUNK_RELEASE_ORPHANED`, which only a call
        with `orphan_if_live` set returns, in place of REFUSED_LIVE or
        REFUSED_LEASE (a live allocation is checked first, then a lease),
        after setting `orphaned` in the same hold. It fills `out` only on
        DONE. It
        decides the status under the pool's lock, with the chunk's state, so
        a caller can tell a no-effect release from a refusal;
      - **the lease-drop trigger** (r17 M-4; r18 M-1, N-6).
        `release_chunk_lease` (`pp:1049`) returns a chunk-index hint (an
        invalid value when there is none). The lookup runs only when a
        drop takes a chunk's count to zero, the chunk is in `chunks_` (not
        `runtime_chunks_`), and it scans only `zone_allocators_[SCRATCH]`'s
        `zone_chunk_state`s for that index, the `zone_free_unlocked` shape
        (`pp:526-545`); every other drop pays one atomic and one
        comparison and nothing more. When that state has `orphaned` set and
        empty allocators, the call returns the index, as `zone_free` does
        for the last free. Its one cache-side caller,
        `unified_cache::host_release_chunk_lease` (`uc:24107-24111`, reached
        from `mem-handle.cpp:1142`), calls the **public**
        `release_marked_chunk` with the index after the pool's lock
        drops, and C4 declares it `noexcept`, because its only
        caller is `mem_handle::release_lease_state`, itself `noexcept`
        (`mem-handle.cpp:1102`). So the path is non-allocating by
        construction: the public form holds its one carrier in an inline
        slot, never a vector; the erase from the marked set and the pool's
        unlink are in place; the carrier's destructor appends to the
        already-reserved `stranded_`; and the plain mutex locks are the kind
        `release_chunk_lease` already takes in this `noexcept` chain today
        (`pp:1056`). The path contains no throwing allocation, so it has no
        failure to report. The release is event-driven: no host wait and no
        polling.
        **Callers** (r18 M-1): `release_lease_state` is reached at
        `mem-handle.cpp:982`, `:1004` and `:1067` (the resolve
        refreshes), `:1166` (`release_lease()`, which `~mem_handle` calls
        at `:1387-1388`), and `:1665` and `:1773` (the two assignment
        forms). Each takes the state under the handle's leaf lock and
        releases it after that lock's scope has closed (the assignment
        forms say so at `:1664`, "Release the old leases and allocation
        owner outside the lock"), so no handle lock is held. What is new is the exposure. Before C4 a
        handle drop took only the pool's leaf `mutex_` (`pp:1056`); now a
        drop that closes the last lease on an orphaned empty SCRATCH chunk
        takes `marked_set_mutex_` (L4), nests the pool's `mutex_` (L5)
        inside it, drops both, and then the carrier makes a blocking
        `sycl::free` and takes the registry lock. The order is L4 then L5,
        with no inversion. The blocking free is acceptable in each source
        for two reasons: it runs after every lock the path took has
        dropped, and a final `mem_handle` drop making a blocking free is
        pre-existing in kind (`~mem_handle` drops `owned_alloc_` and
        reaches `unified_free`, `mem-handle.cpp:1387` onward). Only a
        handle holding a lease on a withdrawn, orphaned SCRATCH chunk can
        reach the branch. A caller that drops a handle while holding an L5
        lock is non-conforming under canon 1612-1619 and outside C4
        (llama.cpp-uilv), the orphaned free's rule;
      - **the carrier** (r14 M-1, M-3; r15 I-1, N-3). On a release, under
        the pool's lock, it unlinks the span and the zone allocator, sets
        the tombstone in `chunks_` (base null, size 0), and moves the
        removed chunk's base, backing size and owner into `out`, a
        `pinned_chunk_pool::released_chunk`. It needs no
        `wait_for_chunk_drain`: lease count 0 at the check is enough
        because no lease can be taken afterwards, since
        `acquire_chunk_lease` and `find_chunk_handle` skip a null base
        under the same `mutex_` (`pp:1026-1047`) and the tombstone is set
        under that lock;
      - **`released_chunk` is the release, and it is RAII** (r15 I-1;
        §M88 (1); §M92; r16 M-2). It is a nested class of the pool,
        move-only, constructed only by `release_zone_chunk`. It holds the
        chunk's base, backing size, owner `mem_handle`, the owner's
        `alloc_metadata`, and a back-pointer to its pool (the destructor
        calls the non-static `free_chunk_owner` and takes `mutex_`). A
        carrier never outlives its pool: the pool is the device cache's and
        is destroyed after every caller of the release path.
        `~pinned_chunk_pool` enforces it with an atomic count of **owning**
        carriers, and its **first statement** asserts the count is 0
        (`released_chunk outstanding at pool destruction`), before the
        `lock_guard` and the free loops (`pp:141-146`). The count rises
        when `release_zone_chunk` builds an owning carrier and falls when
        an owning carrier is destroyed or consumed; a move transfers
        ownership and leaves the count unchanged, and a moved-from or
        default-constructed carrier holds no owner and is never counted.
        The decrement is the carrier destructor's **last** statement,
        after step 4 has released `mutex_`, so a concurrent
        `~pinned_chunk_pool` cannot pass its assertion while a carrier
        still touches the pool.
        **The metadata's source.** `mem_handle` exposes no `ptr`/`key()`
        accessor (`debug_info()` only), and `metadata()` exists only on
        `alloc_owner` (`mem-handle.hpp:60`). So
        `allocate_pinned_chunk_owner` (`pp:84-113`) reads
        `allocation.owner.metadata()` before `mem_handle::from_owned_alloc`
        consumes the owner, and returns it with the handle in a small
        struct, `pinned_chunk_backing {mem_handle owner; alloc_metadata
        metadata;}`. `grow_into` stores both on the `chunk` (a new field,
        `owner_metadata`), on the synchronous path and on the `std::async`
        path (`pp:904-926`, whose future now returns the struct).
        `release_zone_chunk` moves the field into the carrier with the owner.
        **Its destructor is the release.** A carrier that still holds an
        owner runs four steps, with no lock held at the first three:
        1. it takes the recorded `alloc_metadata` (`uch:5710`: `ptr` and
           `key()`, the two fields the registry's own exact checks compare,
           `uc:16428` and `:16553-16554`), before the reset;
        2. it calls the private `free_chunk_owner` (`pp:778-791`) on the
           removed chunk, whose `mem_handle` reset is the only release
           point (P2). The carrier keeps the non-null base (the null base
           that makes `free_chunk_owner` return early, `pp:779-781`, is
           the `chunks_` entry's), so the missing-owner `GGML_ASSERT`
           (`pp:788-790`) still guards the release;
        3. it asks the registry, read-only, whether that exact row is
           still registered, through a new shipped accessor,
           `unified_allocation_registered_exact(const alloc_metadata &)
           noexcept`, in the cache. It takes `g_runtime_alloc_mutex` alone,
           finds the row by `ptr`, compares `key()` and writes nothing, and
           returns the row's own state: GONE, LIVE or RELEASING (§M97: the
           accessor reports its own row and the WARN claims no cause,
           since naming one would need a second source, the retry list).
           The existing `allocation_registry_test_contains(ptr)`
           (`uc:16534`) is test-only and answers by pointer alone, which
           a reused address would satisfy, so it is not the accessor;
        4. under the pool's `mutex_`, in one short section: if the answer
           is GONE, it subtracts the backing size from `total_allocated_`;
           otherwise it appends `{metadata, backing size}` to the pool's
           `stranded_` list **without subtracting**, and prints
           `[PINNED-POOL] released chunk backing %.1f MB: release not yet
           complete after the owner reset (registry row %s); footprint
           stranded until the registry drops it` (WARN), where `%s` is LIVE
           or RELEASING. The row cannot say why the release is pending (a
           scheduled retry, other references on the control, or the
           shutdown abandon), so the line does not claim a cause.
           **The append never allocates, and the reserve comes first.** At
           most one stranded entry exists per chunk (a chunk is released
           once, and its carrier strands once), so the list never needs
           more entries than the pool has chunks. Every chunk is added in
           one place, `grow_into` (`pp:922` on the async path, `pp:945` on
           the synchronous path); `configure_zones`, `allocate_segmented`
           and `grow_zone` reach it through `grow`/`grow_into` and add
           none. `grow_into` therefore reserves `stranded_` to
           `chunks_.size() + runtime_chunks_.size() + 1` after its budget
           check (`pp:892`) and **before it allocates the chunk owner**, so
           a failed reserve never creates and releases a registered owner,
           and a chunk never exists uncounted or without room for its
           stranded entry. A `bad_alloc` from the reserve maps to `return
           false` through an explicit catch placed ahead of both paths (the
           async path catches `std::exception` at `pp:930`, the synchronous
           path only `sycl::exception` at `pp:950`). The size never
           decreases: chunks are only tombstoned (the sole clear is the
           pool's destructor, `pp:161-162`), and the reconcile prunes **in
           place** (`std::remove_if` and `erase`), which never shrinks
           capacity; a rebuild, swap or `shrink_to_fit` is forbidden. The
           destructor asserts `stranded_.size() < stranded_.capacity()`
           before its `emplace_back`, with the message `stranded_ capacity
           %zu exhausted at %zu entries`, so a violated invariant is a named
           abort and never a silent reallocation or a terminate from
           `bad_alloc`.

        **A refused free is a scheduled retry** (r16 I-1; §M92), not a
        leak. `alloc_owner_control::release_ref` (`uc:857-862`) calls the
        coordinator's `retire` (`uc:995-1019`), whose `release_physical`
        (`uc:983-993`) maps every status but RELEASED to RETRY_SCHEDULED
        and pushes the control on the retry list; `process_retries`
        (`uc:1031-1056`) frees it later, when whatever refused has gone.
        Production causes exist: `unified_free_record` returns false on a
        `sycl::exception` from `sycl::free` (`uc:14947-14960`) and on a
        `zone_free` refusal (`uc:14827-14829`). So the registry, not the
        pool, knows when the memory goes, and the pool reads that one fact
        instead of keeping a second. **`footprint_locked()`** is one
        private helper that, under `mutex_`, runs
        `reconcile_stranded_locked()` (it asks
        `unified_allocation_registered_exact` for each stranded entry and
        prunes the ones now GONE, subtracting their bytes at that moment)
        and returns `total_allocated_`. **Every decision read of the
        footprint goes through it** (r17 I-1): `allocate_segmented`
        (`pp:186`), `pre_allocate` (`pp:242`), `configure_zones`
        (`pp:323`), `grow_zone` (`pp:661`), `allocate_from_chunks`
        (`pp:827`, which serves `allocate_runtime` and the zone-base path),
        `grow_into` (`pp:892`), and the getter `allocated()` (`pp:768`).
        Several of these test before they call `grow_into`, so a reconcile
        at `pp:892` alone would never be reached. Log-only reads are
        exempt. When `stranded_` is empty the reconcile is a size test and
        takes no registry lock. `stranded_` and `total_allocated_` are
        `mutable`, since `allocated()` is `const` (`pph:206`).
        `allocated()`'s external callers are `uch:4140`
        (`pinned_pool_committed`), reached from `gs:5688`, and
        `mem-ops.cpp:392` (`stage_trace_sample`, reached from
        `mem-ops.cpp:434`, `:509`, `:518` and `gs:107656-107659`); none
        holds `g_runtime_alloc_mutex`, so the two-lock call is acyclic. The
        reconcile nests `g_runtime_alloc_mutex` under `mutex_`, the nesting
        the census row above already makes conforming (`pp:939`,
        `uc:15578`). The footprint therefore leaves exactly when the memory
        does. **Teardown:** `~pinned_chunk_pool` reconciles once and prints
        `[PINNED-POOL] stranded chunk backing %.1f MB still registered at
        pool destruction` (WARN) for each entry that is still registered.
        The registry's shutdown branch (`uc:1005-1008`) abandons a control
        with its row registered and schedules no retry, so that line is
        the report for it when the pool outlives the abandon. A strict
        abort applies only through the registry's own ownerless
        classification, unchanged; the carrier's destructor never names
        `ggml_sycl_strict_enabled`, and the gate pins that.

        A moved-from carrier holds no owner, and its destructor does
        nothing. The public `release_chunk_backing(released_chunk &&)`
        consumes a carrier early: it moves it into a local that leaves
        scope at once. There is one release path, and the accounting rides
        on it. So a carrier dropped anywhere, by an early return or an
        exception, releases and accounts exactly as a consumed one does,
        and cannot leak the footprint or free the memory on a path the
        design does not name. Callers still reserve `out` for their
        candidate count before the first release, and the append is a
        `noexcept` move into reserved storage, so nothing that can throw
        follows a release and the drop path is a backstop. The backstop is
        **not conforming**: a carrier destroyed while a caller holds
        `marked_set_mutex_` would destroy a final `mem_handle` and make a
        blocking device call (`sycl::free`) under a listed lock, which
        canon §12.5 forbids (canon:1612-1619 at e2461d4fb). Lock order is a
        different property and does not excuse it. It is reachable only if
        a caller breaks the collect-then-release rule, and gate clause 5
        pins that rule, including that each collecting vector is declared
        before its scope's lock statement, so an unwinding scope destroys
        it after the unlock. The registry query of step 3 takes
        `g_runtime_alloc_mutex` after the owner's release has dropped it,
        and step 4's section is not nested under it. The carrier is never
        destroyed under the pool's `mutex_`, since it leaves the pool's
        scope in `out`. The destructor is `noexcept`;
      - **the footprint and its window.** The footprint leaves
        `total_allocated_` **only when the allocation is actually gone**
        (§M88 (1); §M92), after the physical free, in step 4 or at a later
        reconcile. A concurrent grow (`configure_zones`, `grow_zone`,
        `allocate_segmented`) is therefore admitted (`pp:892`) only
        against a footprint that still counts a chunk whose memory is not
        yet freed, never against room that is not there (P4, plan ==
        reality). The visible cost: for the duration of one free the chunk
        is a tombstone and unmarked while its backing is still counted,
        and a concurrent grow can be refused against the still-counted
        chunk, returning false like any budget refusal, bounded by that
        one free; after a registry refusal the bound is the scheduled
        retry. The pool's lock is held across neither the free nor the
        set's lock. This applies to the release path only; the pool's
        destructor and `grow` still free or allocate under `mutex_`;
      - the `chunks_` entry stays as a tombstone (base null, size 0), so
        no other chunk's index moves: the indices in `flat_spans_`, in
        every `zone_chunk_state` and in the lease handles
        (`pinned-pool.hpp:256`) stay valid. **Every loop over `chunks_`
        skips an entry whose base is null** (r9 n-4): `contains`,
        `allocated`, `chunk_count`, `configure_zones` (`pp:361-418`, which
        rebuilds the spans from `chunks_`, for example at `llama-bench`'s
        model switch), `pre_allocate`'s free-capacity sum (`pp:228-230`)
        and `allocate_segmented`'s search (`pp:176-181`). The destructor
        already skips a null base (`pp:141-162`). The release also resets the
        chunk's per-chunk `allocator`, so a loop that missed the rule
        fails on a null allocator instead of reporting capacity that does
        not exist;
      - the header's "Chunks are only released when the pool is
        destroyed" (`pinned-pool.hpp:112`) is rewritten to name this
        release;
      - **`configure_zones` refuses while a withdrawn chunk survives**
        (r10 n-3, r11 m-6). It rebuilds every zone from scratch
        (`pp:389-417`: `zone_allocators_[z].clear()` and fresh
        `zone_chunk_state`s), so it presumes no live zone allocation, and
        a fresh state carries neither flag. The cache's
        `configure_host_zones` (`uc:20415-20436`) first releases every
        chunk in the device's marked set whose allocators are empty. A
        chunk that is not empty is still withdrawn, and
        `configure_zones` **refuses the rebuild**: under its lock, before
        it touches any state, it scans the zones' `zone_chunk_state`s
        and, if any has `withdrawn` set, prints `[HOST-ARENA]
        configure_zones refused: chunk %zu still withdrawn, %zu B
        allocated` (ERROR) and returns false. **The plumbing** (r12 M-6):
        C4 changes both `configure_zones` and the cache's
        `configure_host_zones` (`uc:20415`, `void` today) to return
        `bool`. The bool is false **only** for this refusal. The scan is
        the first thing `configure_zones` does, before the capacity growth
        (`pp:318-328`) and the shortfall check (`pp:330-358`), so a false
        return can never skip a zeroing that call owes. The capacity
        shortfall keeps today's behaviour exactly: zones unconfigured,
        every zone size 0, `configure_zones` returning true, and
        `configure_host_zones` then zeroing the reserved figures
        (`uc:20428-20430`) and returning true. `configure_host_zones`
        keeps the two returns separate: on false from the pool it returns
        false at once, before and without touching
        `host_zone_reserved_bytes_` or reading `zones_configured()`; under
        `ggml_sycl_strict_enabled()` (`uch:7259` at `d8a67422d`, added by
        llama.cpp-uwlx after this doc's base) it first `GGML_ABORT`s with
        the same line, since the refusal is a defensive invariant and not
        a capacity outcome. A refusal therefore changes nothing: the
        pool's zones, every flag, the marked set and the reserved figures
        stay one consistent state. **Callers:** a caller that gets false
        fails its load with that line and re-routes nothing (P3). No
        production caller receives it today, and the doc claims no
        production load failure: the one production caller, the void
        `ggml_sycl_configure_host_zones_for_plan` (`gs:5615`), returns
        before the call whenever zones are configured (`gs:5621-5623`);
        a future caller must propagate the false rather than log "Zones
        configured" with sizes the pool did not take.
        **Why refusing is the sound choice.** The flag's one home is the
        `zone_chunk_state` (above), so the scan reads the one fact and the
        cache's marked set is not consulted (P4). A rebuild would discard
        the chunk's allocator and its live allocation while the marked set
        still named the chunk, and the release would then meet a fresh
        allocator in a chunk the zones' sizes did not count. Carrying the
        flags and allocators across the rebuild is rejected: the rebuilt
        sizes come from the new request, so a surviving allocator would be
        a second source for the zone's capacity. The case is defensive: the
        production plan caller (`gs:5615-5623`) returns when the zones are
        already configured, and a withdrawn chunk exists only under
        configured zones, so it is reachable only from a direct
        `configure_host_zones` caller (the tests call it) or a future
        caller.
    - **A return that finds a live allocation refuses by name.** If a
      marked chunk still holds an allocation or a lease, the release
      prints `[HOST-ARENA] SCRATCH return refused dev=%d chunk=%zu: %zu B
      still allocated (cohort %s)` (WARN), keeps the chunk, and the
      transaction's figure stays charged. It never forces the release. A
      lead arm that prints it is a FAIL: something outside the losing
      candidate took SCRATCH during the trial.
    - **The witness.** `KEEP`, `RETURN` and `RETURN_TO_MARK` print
      `[HOST-ARENA] SCRATCH <txn> dev=%d n_ubatch=%u kept|returned %.1f MB
      (%zu chunk(s))` (INFO). `<txn>` is `trial` only for `try_candidate`'s
      transaction, `reserve` for `sched_reserve`'s own, and `retry` for
      `RETURN_TO_MARK` (r10 n-7). The release prints `[HOST-ARENA] SCRATCH
      released %.1f MB (%zu chunk(s)); capacity %.1f MB` (INFO) with the
      capacity after it. On a lead arm, a `returned` line of any form with
      no `released` line before the next `SCRATCH demand ctx=` line is a
      FAIL. **At zero chunks nothing prints** (r9 m-4): the three ops
      print only when they move at least one chunk, and `released` prints
      only when a release frees at least one. `sched_reserve`'s own
      transaction `KEEP`s on every successful reserve, so a `reserve ...
      kept` line means that reserve grew SCRATCH.
    - A commit **replaces** that context's committed plan, as a notify
      replaces its demand. The device's planned capacity per slot is the
      high-water over the live contexts' **committed** plans, not over
      every candidate tried (r6 m-5). A slot above it is re-allocated down
      to it at the commit. That runs in `sched_reserve`, after its own
      `synchronize()` (`lc:2002`) with no `graph_compute` since (r7 n-4),
      and same-device concurrent inference is unsupported (canonical
      contract §5), so no dispatch on the device is using the slot. The entries drop in the backend
      context's destructor (`gs:43877`), their owner boundary.
    - **On `false`** (r6 n-3). Either call returns false when the ledger
      could not ensure capacity. `llama_context` then prints `[HOST-ARENA]
      host holder plan failed on %s` (WARN) and continues. Every slot left
      unplanned refuses by name at dispatch, and unplanned pools stay
      uninitialized (above). A lead arm that prints the line is a FAIL.
    - The reserve graphs are built on `memory->init_full()` (`lc:2027`), so
      `n_kv` is the whole cache and `n_tokens` is `n_ubatch`: the largest
      request any prefill or decode of this context can make. For each node
      the SYCL side reads the backend the scheduler assigned it
      (`ggml_backend_sched_get_tensor_backend`), skips any that is not a
      SYCL backend, and takes that backend's `device`: the device whose
      `ctx.device` the dispatch indexes its slots by (`gs:82358`,
      `:82844`). One fact, one source: the device that dispatches owns the
      slot (§M69 (d)).
    - It selects nodes with the dispatch's own predicate, not a second one
      (r5 m-6): `FLASH_ATTN_EXT` where `ggml_sycl_tensor_is_in_kv_host_buft`
      holds for K or V (`src[1]`, `src[2]`), and `SET_ROWS` where it holds
      for the node, exactly the tests at `gs:83014-83016` and
      `:83022-83024`. That predicate resolves `view_src` first
      (`gs:108235-108241`). K is a view of the cache, and a view's `buffer`
      is `NULL` at creation (`ggml.c:1817`), while the cache tensor it
      views was allocated at context init. `ggml_backend_buffer_is_host`,
      rev 3f's predicate, is also true of CPU and `SYCL_Host` buffers the
      dispatch never stages (`-nkvo`, CPU-assigned layers).
    - Per (device, kind) it takes the largest `ggml_nbytes` of the operand
      the dispatch stages, the same calls the dispatch makes: Q (`src[0]`),
      DST (the node) and MASK (`src[3]`) for attention (`gs:82389-82392`,
      `:82424-82426`); VAL (`src[0]`) and IDS (`src[1]`) for `SET_ROWS`
      (`gs:82823-82824`, `:82839-82840`).
    - At commit it rounds each figure by the slot rule (`bytes + 64`, up
      to 1 MiB, `gs:82043`), takes the per-slot high-water over the live
      contexts' committed plans, and registers the slots' sum as the
      staging's `DEVICE` demand **on the dispatching device's cache**. It
      ensures SCRATCH capacity for it, then grows or shrinks each slot to
      its planned capacity through the existing grow code, with
      `req.device` set to the dispatching device (today `-1`, `gs:82048`).
      All of that runs in `sched_reserve`, before the first
      `graph_compute` and outside the PP/TG phase, and prints
      `[ATTN-HOST] planned staging dev=%d q=%zu dst=%zu mask=%zu val=%zu
      ids=%zu total=%.1f MB` at INFO, once per device and commit. A device
      with no selected node plans nothing for it.
    - **At dispatch no slot grows.** On the persistent path, `bytes >
      slot.cap` is a request beyond the plan. C4 replaces the grow there
      with a named refusal, `[ATTN-HOST] staging request %zu B for kind %s
      exceeds the planned %zu B on device %d; declining host dispatch`
      (WARN), and returns `nullptr` to the existing declines
      (`gs:82458-82461`, `:82653-82656`, `:82851-82854`). A slot that was
      never planned has capacity 0 and refuses the same way. The
      `GGML_SYCL_ATTN_HOST_STAGING_IDLE` bisection branch (`gs:81989-82013`)
      also grows its held slot today; C4 gives it the same refusal, so no
      branch grows a slot at dispatch (r5 n-1). Its per-call buffer stays
      the `GRAPH` fallback of the site table.
    - **A refusal fails the graph on every path** (§M69 (b); r5 I-2 (1)).
      The declines return `false` from the dispatch and then from
      `ggml_sycl_compute_forward`. The ordinary node loop asserts on it
      (`gs:94499-94506`), but the MoE segmented record/replay paths
      discard it at **six** sites (§M69 (b) as amended; r6 m-1).
      `moe_graph_record_segments` (`gs:97103`) discards at `gs:97234` (a
      direct segment at first record), `gs:97240` (that segment's
      `moe_after` node), `gs:97285` (recording) and `gs:97315` (the
      `moe_after` node after a recorded segment).
      `moe_graph_replay_segments` (`gs:97478`) discards at `gs:97551`
      (replay's direct segments) and `gs:97595` (replay). Direct segments
      are where `FLASH_ATTN_EXT` nodes go (`gs:97221-97234`), and the
      segmented path is the MoE default (`gs:107142-107148`). So a refusal
      there would be a WARN and a skipped op: wrong tokens, not a refusal.
    - **The six sites fail the graph by throwing; they neither assert nor
      return** (§M69 (b) second and third amendments; r8 I-1). C4 routes
      all six calls through one helper that, on `false`, prints the
      device's dispatch counter line (below), then the ordinary loop's
      `[SYCL] graph compute unsupported node` line (its `fprintf(stderr)`,
      `gs:94500`), and throws
      `ggml_sycl_fallback_error` (`common.hpp:94`, a `std::exception`, so
      no `sycl::exception` catch on the way intercepts it). Its argument is
      a **string literal**: the type stores the `const char *` it is given
      (`common.hpp:94-100`), and the catch prints `what()` after the
      unwind (`gs:107778`), so a formatted local naming the node would
      dangle; the node's name is in the unsupported-node line (r9 n-3). It
      never
      asserts or aborts: a refusal is a capability refusal, not a terminal
      plan bug, and §M72's abort, and its ban on throwing, belong to the
      TERMINAL `[ZONE-PLAN-BUG]` channel only.
    - **Why a throw, and not a return.** The recording site (`gs:97285`)
      runs inside `seg_graph.begin_recording` (`gs:97264-97271`), with
      `g_ggml_sycl_graph_recording` set, `g_recording_graph_ptr` pointing
      at the local `seg_graph`, and the retained-handle sink installed. A
      return from there would destroy `seg_graph` while the queue records
      into it and leave the pointer dangling. An early return from
      `graph_compute` would also skip the cleanup every other exit runs:
      the transient-lease unpin (`gs:107553`) and the execution release
      (`gs:107556`), a P2 leak on every failed decode, and llama-server
      carries on after -3. rev 3j's out-parameter and "return at once"
      had both defects and are withdrawn.
    - **The channel it takes.** The recording function's own catch
      (`gs:97334-97349`) resets the recording flag and pointers, removes
      the sink, trims the retained handles when nothing was submitted, and
      **rethrows** a `ggml_sycl_fallback_error` (`gs:97347-97349`), so the
      block-graphlet and per-op fallback (`gs:107209-107231`), which would
      re-dispatch nodes that already ran, never runs. The direct-segment
      sites (`gs:97234`, `:97240`) and both replay sites (`gs:97551`,
      `:97595`) are outside any try: the replay function's one catch
      (`gs:97588-97590`, its try at `gs:97583`) wraps only the
      descriptor-graph replay (r9 n-1). So every site's throw reaches
      `ggml_backend_sycl_graph_compute`'s `ggml_sycl_fallback_error` catch
      (`gs:107748-107779`), which quarantines, drains, unpins the
      transient leases, releases the execution graph, resets the recording
      globals, clears `graph_compute_active`, prints `[SYCL] recoverable
      runtime fallback failed: %s` (ERROR, `gs:107778`) and returns
      `GGML_STATUS_FAILED`. The exception never crosses the backend
      boundary; the status that leaves `graph_compute` is FAILED, which is
      what the second amendment requires.
    - **th32's cancel step.** th32 3f §5.3 runs the backend's lazy cancel
      step first in each of `graph_compute`'s two catch paths, before
      `ggml_sycl_cpu_tg_flush_pending()` and before either queue wait (it
      cites this catch's flush and waits at `c69d5774d`). A refusal
      therefore cancels or drains a lazy copy parked at a segment boundary
      before the catch waits on the queue. A non-catch exit would skip
      that step and leave the parked copy to `process_ubatch`'s failure
      path, which runs later; th32 moved its cancel out of that path for
      exactly this reason. Qwen 3b, with the host dispatch on and a lazy
      PLE table, is the arm where both are live.
    - **Submitted executables outlive the throw** (§M69 (b) fourth
      amendment; r10 I-1). `moe_graph_record_segments` keeps the
      executables it records in two locals, `recorded_segments` and
      `recorded_moe_graphs` (`gs:97159-97160`), and moves them into the
      context only on success (`gs:97355-97356`). A throw from any of its
      four sites would destroy them in the unwind: the direct sites
      (`gs:97234`, `:97240`) after earlier segments were submitted, the
      recording site (`gs:97285`), and the moe_after site (`gs:97315`),
      whose segment was submitted just before (`gs:97311`). Destroying an
      executable while its submission is outstanding is the tree's own
      recorded heap corruption: `sycl_exec_graph_clear_active`'s comment
      (`gs:99522-99538`, llama.cpp-dkw0 defect #4) records a glibc
      "corrupted size vs. prev_size" abort with an `exec_graph_impl`
      destructor frame. The clear itself drains before it destroys
      (`gs:99542-99545`), and `invalidate_moe_segments` retires first, but
      the retirement drains only when a retention epoch was published:
      `moe_graph_retention_retire_exact` returns true at once at epoch 0
      (`gs:96545-96548`), and a graph with `moe_segments` but no
      `moe_dispatch_graphs` publishes none (`gs:97361`). So not
      every destruction site drains first (r11 m-7): the invalidate at
      `gs:107202` and the exit at `gs:97363` (below) do not. This replaces
      rev 3m's "unverified". So:
      - **The entry guard** (r11 m-1). `invalidate_moe_segments`
        (`common.hpp:6234-6244`) does not always clear: when
        `ggml_sycl_retire_moe_graph_epoch` fails it sets
        `moe_graphs_disabled` and returns with `moe_segments`,
        `moe_dispatch_graphs` and `moe_segments_valid` unchanged, because
        it will not destroy an executable whose submission may be
        outstanding. `graph_compute` records anyway (`gs:107202`, then the
        call at `:107206`; nothing between them reads the flag). So "the
        context vectors are empty at the handler" is not a fact of the
        call site, and C4 makes it one: at entry to
        `moe_graph_record_segments`, before it declares its locals or
        submits anything, a non-empty `moe_segments` or
        `moe_dispatch_graphs` makes it print `[SYCL-SEG] record refused:
        the context still holds %zu segment(s) after a failed retire`
        (WARN) and `return false`, the caller's existing fallback (the
        disabled flag is already set). **The invariant: the handler, and
        the success path's move (`gs:97355-97356`), move into empty
        vectors.** Neither destroys an old executable. The guard also
        repairs the success path's same defect, which `gs:97355-97356`
        has today. **Where the kept executables are finally freed** (r12
        M-1): after the guard's `return false` the caller (`gs:107214`)
        retries `invalidate_moe_segments`, and a retry that succeeds
        destroys them after its retire drained (a retire can only have
        failed at epoch != 0, so a later success has waited the retention
        terminals' streams, `gs:96551-96553`). If the retry fails too,
        `moe_graphs_disabled` is sticky (nothing in the tree resets it
        but the context's construction, `common.hpp:6114`) and gates the
        graph path off (`gs:106423-106426`), so nothing **records** again
        (the segmented block runs only under `use_sycl_graph`). Invalidation
        still happens: the catch's cleanup step (below) and every
        `sycl_exec_graph_clear_active` caller, which calls
        `invalidate_moe_segments` (`gs:99565`; callers include `gs:15773`
        at context drain, `gs:99579` at model teardown, `gs:105417` and
        `:105558` at a phase change), each retry the retire. A retry that
        succeeds destroys the executables after its drain; one that fails
        keeps them, and they are freed in `~ggml_backend_sycl_context`
        (`gs:43877`), which waits every queue of every device
        (`gs:43880-43892`) before its own invalidate (`gs:43917`) and
        before the member destructors run. **Every destruction point
        drains.** So an old executable is destroyed only by a successful
        retire after its drain (the caller's retry, the cleanup step's
        invalidate, or a clear), by the cleanup step after the waits, or by
        the context's destructor after its queue waits;
      - C4 wraps the record loop, after the two locals' declaration, in
        `try { ... } catch (...) { ...; throw; }`. The handler moves both
        locals into `sycl_ctx->moe_segments` and `moe_dispatch_graphs`
        (empty by the guard; a vector move-assignment into an empty
        `std::allocator` vector does not throw and moves no element), sets
        `moe_segments_valid` false explicitly (the flag may be stale-true
        after a refused retire), and rethrows. The unwind that follows
        destroys two empty locals;
      - **Which exceptions reach the handler, and where each goes
        next** (r11 m-3 (c)). The handler sees every exception type from
        the loop's four sites, including the inner catch's rethrow
        (`gs:97347-97349`) of a `ggml_sycl_fallback_error`, which goes to
        `graph_compute`'s fallback catch and its cleanup step (next
        bullet). It also sees what the inner catches do not take: a
        `sycl::exception` or any exception from the two direct-dispatch
        sites (`gs:97234`, `:97240`), which sit outside the per-segment
        try, and an exception of a type the inner catches miss. Those go
        to `graph_compute`'s `std::exception` and `...` catches
        (`gs:107780-107787`), whose shared cleanup
        (`ggml_backend_sycl_graph_boundary_exception_cleanup`,
        `gs:107617-107644`) has the same queue waits and the staging
        drain but never calls `invalidate_moe_segments`. The moved
        executables then stay in the context, `moe_segments_valid` false,
        until the next record's invalidate (`gs:107202`), which runs after
        those waits, or until the context's destructor, which waits every
        queue first (`gs:43880-43892`). That is safe and is the cost of
        not widening those catches; the cleanup step below is the
        fallback-error path's;
      - no wait at the throw site: it would precede th32's cancel step;
      - `graph_compute`'s catch destroys them in its cleanup step (next
        bullet), after th32's cancel step and after its own waits.

      **Three pre-existing exits that clear without a drain** (r11 n-3).
      The inner catch's two `return false` exits (`gs:97333`, `:97350`)
      destroy their submitted locals without a drain, before the caller's
      block-graphlet and per-op fallback (`gs:107214-107231`); they return
      to `graph_compute_unchecked` (`gs:107206`), not into the
      fallback-error catch. The third is `gs:97363`: the locals were
      already moved into the context, `moe_graph_retention_publish`
      failed, and `invalidate_moe_segments` then runs a retire that is a
      no-op returning true at epoch 0 (`gs:96545-96548`), so the vectors
      clear without a drain. All three are pre-existing on master, out of
      scope here, and tracked as **llama.cpp-7l45**; none is a 5efe fold. 5efe adds no path into any
      of them: its six throw sites throw `ggml_sycl_fallback_error`,
      which the inner `std::exception` catch rethrows (it is not a
      `sycl::exception`, `common.hpp:94`).
    - **The catch's cleanup step** (r9 m-1 (c); r10 I-1, m-1). The
      transient leases are the backend context's `graph_weight_leases`
      and `graph_moe_expert_leases` (`common.hpp:7222-7223`). The unpin
      the catch calls (`gs:107768`) returns without releasing them while
      `ctx->exec_graph` is non-null or `active_exec_graph.valid` is set
      (`gs:61136-61138`). Nothing in the catch clears either before it:
      the quarantine (`gs:15570-15586`) aborts the registry invocation
      only, and the execution release, which runs after the unpin
      (`gs:107769`; `gs:15608-15637`), resets only the execution epoch
      fields, through `ggml_sycl_execution_clear_graph_tracking`
      (`gs:15632`). Nothing on the segmented path guarantees them clear
      either. So C4 inserts one step **immediately after the MMID
      recovery (`gs:107767`) and before the unpin (`gs:107768`)**: after
      th32's cancel step, both queue waits (`gs:107753-107764`), the
      scatter wait (`:107765`) and `ggml_sycl_cpu_staging_drain`
      (`:107766`, which waits the CPU chain and the staging events,
      `cpu-dispatch.cpp:3093-3107`). It is wrapped `try { ... } catch
      (...) {}` like each of its siblings, because an exception thrown
      inside a handler is not caught by the same try's other handlers and
      would cross the backend boundary the third amendment forbids:
      - if `exec_graph` is non-null or `active_exec_graph.valid` is set,
        it calls `sycl_exec_graph_clear_active(ctx, "fallback-error")`
        (`gs:99517-99567`);
      - otherwise it calls `ctx->invalidate_moe_segments()`
        (`common.hpp:6234-6244`), which the clear also calls, so the
        executables the record function moved are destroyed in either
        branch.

      What the clear does, in order: it waits the stream only when
      `exec_graph` is set (`gs:99542-99544`; the wait is `:99543`, on a
      queue already drained); destroys `exec_graph` (`:99545`) before it
      releases any lease, the right order for leases the cached
      executable bakes in; releases the pool's retained handles
      (`:99546`); resets the replay state and the input caches
      (`:99547-99555`); clears the input staging (`:99560`) and the leaf
      staging cache (`:99562`, `cpu-dispatch.cpp:2494-2496`); unpins the
      MoE and weight leases itself (`:99563-99564`), so the catch's unpin
      at `:107768` then runs on empty vectors; and invalidates the MoE
      segments and block graphs (`:99565-99566`). That invalidation
      retires the MoE graph epoch first, and the retirement waits the
      streams of the devices with a retention terminal only when a
      retention epoch was published (`gs:96545-96553`); at epoch 0 it
      returns at once without a wait. **So the destruction is safe because
      of the catch's own waits** (r11 m-7): `gs:107753-107764`, on the
      same `stream()` the segments were submitted on (`gs:97311`), which
      precede this step. The retirement is an extra drain when an epoch
      exists, not the safety argument. If retirement fails it keeps the
      vectors and sets the disabled flag rather than destroy an executable
      whose submission may be outstanding (`common.hpp:6234-6238`). The
      context then records no more (the disabled flag is sticky,
      `gs:106423-106426`), and the vectors are freed as the entry guard's
      bullet states. All of
      these waits run after th32's cancel step, and the leases and leaf
      staging are released only after the CPU drains (P2). A whole graph
      cached for another phase is dropped and re-recorded at its next
      use.
    - **Scope of the cleanup step** (r10 n-8). The step is in the catch
      that every `ggml_sycl_fallback_error` reaches, so it applies to all
      38 existing throw sites (37 in `ggml-sycl.cpp` and one at
      `getrows.cpp:291`, at `e2461d4fb`), not only the six. That is
      intended: each of them ends the graph FAILED, after which the cached
      whole-graph and segment state is not trusted, and dropping it after
      the drains costs one re-record at its next use and nothing else.
    - llama then logs `ggml_backend_sched_graph_compute_async failed with
      error` (`lc:3968-3970`), and `llama_decode` returns -3
      (`lc:3290-3294`), its existing error path (through `gb:2515-2516`
      and `lc:2836`). No segmented site gets a
      legitimate `false` for an op that then runs elsewhere, since the
      ordinary loop asserts on the same `false` for the same node
      (`gs:94322-94327`). The ordinary loop's assert (`gs:94499-94506`) is
      upstream's and outside the six. The defect class is live on master
      and is tracked as **llama.cpp-573t**, whose text
      (`GGML_STATUS_FAILED`) is the authority, so the fix stays tracked if
      C4 slips.
    - **The dispatch counter** (r6 I-2 (1), (2)). It is per device and
      module-global. `fa` counts **interceptions**, the entries into the
      host `FLASH_ATTN_EXT` dispatch, incremented before
      `ggml_sycl_attn_host_staging_get`, whatever happens after. `set_rows`
      counts `SET_ROWS` interceptions the same way. `refusals` counts the
      named staging refusals. They print as `[ATTN-HOST] host dispatches
      dev=%d fa=%zu set_rows=%zu refusals=%zu` (WARN) at every SYCL
      backend free for that device while the switch is on, and from the
      fail-closed helper before it throws, so the line survives a failed
      graph. The first interception per device also prints `[ATTN-HOST]
      first host dispatch dev=%d op=%s` (WARN), once. A score reads the
      **last** counter line per device. Load-time temporary backends print
      `fa=0` lines first, before any plan.
    - **Always compiled** (§M74 (a), (f)). Every line C4 adds and a vehicle
      arm scores is compiled in ggml-sycl itself, never behind
      `GGML_SYCL_PRIVATE_TESTING`, which only test targets define: the
      counter, `first host dispatch` and `plan attributed 0` lines;
      `planned staging` and `staging not planned`; the `[HOST-ARENA]
      SCRATCH demand` and `DEVICE pools` lines and the pools' `exceeds its
      planned`, `unplanned` and `placement unknown` WARNs; `host holder
      plan failed`; `host compute buffer ... exceeds notified`;
      `[HOST-ZONE] leaked`; `[FILE-MAPPED] counters`; and §M75's `SCRATCH
      trial ... kept|returned`, `SCRATCH released`, `SCRATCH return
      refused` and `DEVICE pools ... resized` lines (r8 n-7), the
      first now as `SCRATCH trial|reserve|retry ... kept|returned` (r10
      n-7). Each is a
      dedicated WARN or INFO line its owner prints in the shipped library,
      which §M74 (f) accepts as a vehicle source, so none moves into the
      §M74 (g)/(i) counter dump (`[SYCL-COUNTER] dev=%d name=%s
      value=%llu` lines and an `end devices=%d` line, printed at exit when
      `GGML_SYCL_COUNTER_DUMP=1`; r9 n-5). The `[ATTN-HOST]` dispatch lines are also gated by
      `GGML_SYCL_ATTN_HOST_DISPATCH=1`, which 3b and 5b set in their
      literal commands (§M66). Only test binaries keep PRIVATE_TESTING
      seams: C2's `::free` and `sycl::free` branch counters and the
      §4.6 arm-3 mutant (§M74 (d)).
    - So the staging's demand reaches the ledger once per context and
      device, from `n_ctx` and `n_ubatch`, and decode-time regrowth never
      does.
    - **A stale `libllama`** makes no plan call, so every slot stays at
      capacity 0 and every host dispatch refuses, fail-closed. The pairing
      stays unsupported (r4 n-1).
  - **`GRAPH` tenants** (the twelve `GRAPH` lines of the site table): each
    is bounded by a named plan term that the census gate records per row:
    `max_tensor_bytes` + `k_scratch_headroom` for the single-tensor staging
    copies, `moe_cpu_expert_staging_bytes` + `moe_host_recipe_workspace_bytes`
    for the MoE staging, and the planned capacity of its kind for the
    attention staging's per-call fallbacks (r4 m-12). A `GRAPH` row the gate
    cannot map to a term is listed **`UNSIZED`** and ratcheted like
    `UNOWNED`.
    - **The three `TP_TMP` lines are `UNSIZED`** (`common.cpp:1414`,
      `gs:48244`, `gs:48982`; r4 m-9). rev 3e sized them by
      `tp_vram_runtime_bytes`, which sizes device buffers and so leaves host
      SCRATCH (the term trace below). TP is opt-in and off on every lead
      arm.
  - **No VRAM-sizing term stays in host SCRATCH** (addendum 11 (1), §M61
    (d)(2), §M62 (A)). rev 3c-3e's RUNTIME read is removed, and so are the
    two terms labelled "SCRATCH (device VRAM)"; the trace is below.
- **The site table** (addendum 7 (c)). 5efe owns each host site's role, zone,
  lifetime and sizing, and publishes them here. 23mk owns each site's
  conversion, and its class-H rows (56, 118 and the others) cite this table.
  C4's source gate (`test-sycl-host-role-routing-source`) regenerates the
  table from the tree and fails on any difference.
  - A first lexical cut finds 72 host-capable lines with category
    `HOST_COMPUTE` or `EXPERT_CACHE`, across 8 files. By role:
    - 23 `EXPERT_STAGING`: already SCRATCH, `GRAPH`, unchanged;
    - 14 role `WEIGHT`: 10 are device-tier cache entries that never reach
      `select_zone`'s host branch, and 4 stay WEIGHT. One of the four is a
      real weight, the direct-stage last resort (`uc:6522`). The other three
      are non-weight bytes carrying role `WEIGHT`: `cpu-dispatch.cpp:315`
      (the CPU-offload host weight copy), `gs:39959` (a device buffer's
      host-pinned fallback on low free VRAM) and `gs:63231` (the
      tensor-split weight cache). These are the "role-WEIGHT, unplanned"
      rows. None is reached on Qwen's default path, and if one is, the
      zone-0 criterion catches it. **`gs:39959` is also a P3 violation**: a
      device buffer's bytes in host-pinned memory are read by the GPU in
      place (zero-copy). That is llama.cpp-nl0f's, beside 23mk's row 56
      (r3 m6);
    - 35 that change zone, or choose it at the site, tabulated below, plus
      the new `SYCL_KV_Host` `alloc_buffer`.
  - Two of the 35 are not tenants: `pinned-pool.cpp:93` is the pool's own
    chunk backing (`require_host_usm_base`, never zone-selected), and
    `uc:16728` is `category_to_intent`, a mapping, not a request.
    `uc:15652` (`acquire_offload_buffer`) takes the lifetime its caller
    passes, so the gate follows its callers.

  | site | holder of the handle | role | zone after C4 | lifetime | owner |
  |---|---|---|---|---|---|
  | `common.cpp:551` | local `host_handle`, released in-function | STAGING | SCRATCH | GRAPH | - |
  | `common.cpp:1414` | local `host_handle` | TP_TMP | SCRATCH | GRAPH | - |
  | `gs:8549`, `gs:8711` | local `d2h_staging` in `moe_hybrid_init_once` | STAGING | SCRATCH | GRAPH | - |
  | `gs:36787` | local `fallback_handle` (`buffer_get_tensor`) | STAGING | SCRATCH | GRAPH | - |
  | `gs:36906` | local `host_stage_handle` | STAGING | SCRATCH | GRAPH | - |
  | `gs:44294` | graph-local `host_copies`, cleared at exit | GRAPH_TMP | SCRATCH | GRAPH | - |
  | `gs:48244` | `g_tp_*` job slot, consumed and cleared by the TP sync (`gs:48401`) | TP_TMP | SCRATCH | GRAPH | - |
  | `gs:48982` | `g_tp_async_ffn_jobs[layer]` slot, consumed within the graph | TP_TMP | SCRATCH | GRAPH | - |
  | `gs:82683` | local `vh`, waited in-function (`gs:82688`) | COMPUTE | SCRATCH | GRAPH | - |
  | `gs:42652`, post-load | a `SYCL_Host` buffer (compute, output, inputs; LoRA and control-vector host buffers) | COMPUTE | SCRATCH | CONTEXT | its `ggml_backend_buffer`, checked at its `free_buffer` (`gs:42563`) |
  | `gs:42652`, `in_model_load` | a model's `SYCL_Host` weights | WEIGHT | WEIGHT | MODEL | the model's `ModelId` (r3 m8) |
  | new, `SYCL_KV_Host`'s `alloc_buffer` (C4) | a demoted layer's host KV, and its `idx_` keys | KV | KV | CONTEXT | its `ggml_backend_buffer`, checked at its `free_buffer` |
  | `gs:42837` | the `SYCL_HostCompute` buffer | COMPUTE | SCRATCH | CONTEXT | its `ggml_backend_buffer`, checked at its `free_buffer` |
  | `gs:44102` | `ggml_backend_sycl_context::readback_staging_handle` | STAGING | SCRATCH | CONTEXT | its backend context, checked after `delete sycl_ctx` (`gs:84119`) |
  | `gs:57652` | `ctx.moe_ids_cache[key]`, or a call-local entry | STAGING | SCRATCH | CONTEXT, or GRAPH for the call-local entry | its backend context, checked after `delete sycl_ctx` |
  | `uc:15652` | the caller's `offload_buffer_lease` | caller's | SCRATCH | caller's | caller's |
  | `cpu-expert-pool.cpp:77` | `g_cpu_expert_pools[dev]` (`gs:2605`), the ring | COMPUTE | SCRATCH | **DEVICE** | the last live backend on the device (§M62 (D)); released today at every `ggml_backend_sycl_free`, llama.cpp-dsab |
  | `pinned-buffer-pool.cpp:69` | `g_pinned_buffer_pools[dev]` (`gs:2604`) | COMPUTE | SCRATCH | **DEVICE** | the same |
  | `gs:82001`, `:82050` | `g_attn_host_staging[dev]` (`gs:81953`), the slots, grown only by the plan, per dispatching device, and only with `GGML_SYCL_ATTN_HOST_DISPATCH` set (§M62 (E), §M69 (a), (d)) | COMPUTE | SCRATCH | **DEVICE** | the same; device-shared, see below; released at the last backend once dsab lands (§M69 (c)) |
  | `gs:82021`, `:82032` | the caller's per-call `fallback_holder`, taken only under `GGML_SYCL_ATTN_HOST_STAGING_IDLE=1` or a `GGML_SYCL_ATTN_HOST_STAGING` mask (`gs:81967-81973`, `:81985-81988`; r4 m-12) | COMPUTE | SCRATCH | GRAPH | - |
  | `common.cpp:443` | `g_runtime_staging_cache` (file scope) | STAGING | WEIGHT until llama.cpp-30r1 | **UNOWNED**, target CONTEXT | `ggml_backend_sycl_context`, after llama.cpp-30r1 |
  | `common.cpp:1970` | `g_tp_host_staging` | TP_TMP | WEIGHT until llama.cpp-30r1 | **UNOWNED**, target CONTEXT | same |
  | `common.cpp:2083` | `g_tp_quant_comm_bufs` | TP_TMP | WEIGHT until llama.cpp-30r1 | **UNOWNED**, target CONTEXT | same |
  | `common.cpp:2279` | `g_tp_shared_reduce_handle` | TP_TMP | WEIGHT until llama.cpp-30r1 | **UNOWNED**, target CONTEXT | same |
  | `common.cpp:2328` | `g_tp_host_buf0/1_handle` | TP_TMP | WEIGHT until llama.cpp-30r1 | **UNOWNED**, target CONTEXT | same |
  | `cpu-dispatch.cpp:2160` | `g_retained_scratch_lease` | COMPUTE | WEIGHT until llama.cpp-30r1 | **UNOWNED**, target CONTEXT | same |
  | `gs:10882` | `g_routing_indices_cache` | STAGING | WEIGHT until llama.cpp-30r1 | **UNOWNED**, target CONTEXT | same |
  | `gs:21527` | `g_moe_fusions[layer].fused_handle` (`thread_local`, `gs:21669`) | COMPUTE | WEIGHT until llama.cpp-30r1 | **UNOWNED**, target CONTEXT | same |
  | `gs:62932`, `gs:62947` | `g_split_staging[dev]` (`gs:62920`) | STAGING | WEIGHT until llama.cpp-30r1 | **UNOWNED**, target CONTEXT | same |
  | `gs:63150` | `g_split_weight_staging` (`gs:63137`) | STAGING | WEIGHT until llama.cpp-30r1 | **UNOWNED**, target CONTEXT | same |
  | `gs:63553` | `s_second_out_ring_alloc` / `s_cpu_out_ring_alloc` (`gs:62650`) | STAGING | WEIGHT until llama.cpp-30r1 | **UNOWNED**, target CONTEXT | same |

  - **Addendum 10 (b) rules the 14 holders (18 lines) that rev 3c brought to
    the lead.** Each gets an owner; the refactor that moves them is
    **llama.cpp-30r1**, not 5efe's.
    - **`DEVICE`, landed in C4:** the `CpuExpertPool` ring and
      `PinnedBufferPool`, per the ruling, and the attention host staging,
      under the ruling's own clause ("if one of them turns out to be
      device-shared, it becomes DEVICE, with the reason stated"). The
      reason: `g_attn_host_staging[dev][kind]` is indexed by device, its
      slots keep a capacity that every context's host attention on that
      device reuses (`gs:81975-82065`), and it is released by the same call
      as the two pools (at every backend free today, llama.cpp-dsab; see
      `DEVICE` above). All three have one release call and are classified
      against their intended owner, the last live backend on the device
      (§M62 (D)), so C4 sets `lifetime = DEVICE` on their requests, and
      moving the holders is not needed. The per-call fallbacks at
      `gs:82021` and `:82032` are not slot capacity: each is a buffer the
      caller holds for one dispatch, so they are `GRAPH` (r4 m-12). C4
      needs this now because the pools are on Qwen's default path. The
      attention staging is **not** on a default arm's path: it serves only
      the opt-in host dispatch (`gs:82450-82455`, `:82645-82650`,
      `:82847-82849`; `GGML_SYCL_ATTN_HOST_DISPATCH`), and by default
      demoted layers run their attention on the scheduler's CPU split
      (§M69 (a); r5 I-1, which corrects rev 3f's "on every no-`-c` arm's
      path"). Its `DEVICE` classification stands for the arms that turn it
      on. Left `GRAPH` in SCRATCH, each would trip the boundary WARN at
      every `graph_compute`.
    - **`CONTEXT`, pending llama.cpp-30r1:** the other 11 holders (12
      lines). Their handles live in file-scope statics or a `thread_local`,
      so no backend context holds them yet, and stamping them `CONTEXT`
      before 30r1 moves them would make the owner check fire on every
      context free. So until 30r1 lands they carry the interim value
      **`UNOWNED`**, which `select_zone` sends to WEIGHT, where each lands
      today, and which no boundary check counts. That is today's behaviour
      made explicit, not a new lifetime: 30r1 deletes the value. C4 gives
      each of the 12 lines a cohort name when it stamps it `UNOWNED`, so
      the allocation trace names it.
    - **Which of them a lead arm reaches** (r4 m-2), from the tree, since
      run2 has no inference phase:
      - `g_moe_fusions[layer].fused_handle` is unreachable:
        `arena_pinned_alloc` (`gs:21518`) has one caller,
        `moe_fusion_state::ensure_fused_buf` (`gs:21626`), and nothing calls
        that;
      - `g_retained_scratch_lease` (32 MiB) is taken only under
        `GGML_SYCL_CPU_RETAINED=1` (`gs:93922-93927`);
      - `g_runtime_staging_cache` stages a non-USM weight for a device read
        (`gs:47945-47957`, `:24288`), which §5.3's counter requires to be 0
        on a mapped load;
      - the four `g_tp_*` holders, `g_split_staging`, `g_split_weight_staging`
        and the merge rings (inside `ggml_sycl_mul_mat_tensor_split`,
        `gs:63294`) serve TP or tensor split, off on every lead arm;
      - `g_routing_indices_cache` **is** reached on MoE graphs
        (`store_routing_indices`, `gs:10852`, called at `gs:46131`). Its
        request is `ncols x nrows` int32 (`gs:10866`, `:10879`), one
        ubatch's routing ids: at most `n_expert_used x n_ubatch x 4` B,
        20480 B on Qwen (10 x 512, `run2.log:157`) and 8192 B on GPT-OSS
        (4 x 512).
      - **Pre-registered** on C4 steps 2, 3 and 5's no-`-c` arm, each under
        `GGML_SYCL_HOST_ZONE_ALLOC_TRACE=1` (`uc:3990`): after the
        pre-size, every `[HOST-ZONE-ALLOC] ... zone=WEIGHT` line
        (`uc:4002-4008`) carries `cohort=routing_indices_cache` and `bytes`
        within that bound, and no `Pinned pool grow_zone: zone 0` line
        follows. Any other cohort on WEIGHT is a FAIL naming it, and so is a
        post-pre-size WEIGHT total above the 64 MiB reserve (`uc:27103`).
    - **The ratchet** (accepted in addendum 10 (b)): C4's census gate fails
      on any request site carrying `UNOWNED` that is not one of these 12
      lines, and it cites llama.cpp-30r1 in the failure message.
- **Host SCRATCH reads no term that sizes a device allocation** (addendum
  11 (1), §M61 (d)(2), §M62 (A); r4 m-8, m-9).
  - rev 3c-3e replaced three copied RUNTIME terms with one read of
    `unified_cache_get_planned_runtime_zone_requirement` (addendum 7 (c)).
    Addendum 11 (1) removes that read: once both PP terms are VRAM
    allocations it sizes no host tenant, an idle host reservation (P4). The
    VRAM RUNTIME zone still reads the terms through that one function
    (`uc:4520`), so 7 (c)'s single source is kept.
  - §M62 (A) generalises the rule: what a term sizes decides, not its
    label. The host SCRATCH sum (`uc:27727-27730`) has ten terms today, one
    row each:

    | term | what it sizes | where that allocation lives | host SCRATCH after C4 |
    |---|---|---|---|
    | `max_tensor_bytes` (`uch:820`) + `k_scratch_headroom` (32 MiB, `uc:27088`) | the host single-tensor staging copies, the `GRAPH` rows `common.cpp:551`, `gs:36787`, `gs:36906` | host pinned, role `STAGING` | **stays** |
    | `moe_cpu_expert_staging_bytes` + `moe_host_recipe_workspace_bytes` (`uch:831-832`) | the MoE PP CPU-island act/out staging | host pinned, role `EXPERT_STAGING` (`gs:78153-78166`) | **stays** |
    | `onednn_reorder_bytes` (`uch:826`) | the oneDNN weight reorder, the dequantized-f16 half of `onednn_scratchpad_bytes` (`uc:27543-27565`) | the device ONEDNN zone, sub-allocated by `reserve_onednn_scratch` (`uc:17514`) | **leaves** |
    | `moe_q8_workspace_bytes` (`uch:828`) | nothing: this sum is its only reader. The MoE Q8_1 activation scratch is sized by its own demand (`uc:21793-21820`) | the device scratch pool: `unified_cache_reserve_moe_q8_1_scratch` (`uc:21824`), `reserve_scratch_pool` (`uc:20917`), the VRAM arena's WEIGHT zone (`uc:21011-21013`) | **leaves** |
    | `moe_vram_runtime_bytes` (`uch:850`) | the MoE expert-pointer tables or CONTROL pool (`uc:27520`) | device RUNTIME | **leaves** |
    | `tp_vram_runtime_bytes` (`uch:889`) | TP FFN and attention compute buffers on secondary devices (`uc:27479-27491`, `:27511`); nothing else reads it | device VRAM | **leaves**; its host `TP_TMP` rows become `UNSIZED` (above) |
    | `dma_staging_pool_bytes` (`uch:856`) | the device double buffer for weight streaming | device RUNTIME | **leaves** (and is 0 in mmap mode, r2 m-2 below) |
    | `pp_pipeline_scratch_bytes` (`uch:864`) | the PP dequant prefetch staging | device RUNTIME | **leaves** |
    | `pp_moe_onednn_scratch_bytes` (`uch:871`) | the PP MoE oneDNN ring | device RUNTIME and KV zones (`uc:18197-18199`) | **leaves** |

  - So C4's host SCRATCH sum is `max_tensor_bytes + k_scratch_headroom +
    moe_cpu_expert_staging_bytes + moe_host_recipe_workspace_bytes`. No
    getter read replaces the removed terms, so rev 3c-3e's pin on the order
    of that getter's setters is moot and withdrawn. The two
    "SCRATCH (device VRAM)" planner fields have no reader left; they go to
    23mk's dead-code census rather than being deleted here.
  - **Qwen's figure**, on the `llama-completion` vehicle of §7 C4 step 2.
    Run2's configured SCRATCH was 7775.9 MB (`run2.log:2893`). The terms
    that leave are: DMA 1700.0 (`run2.log:2752`); `pp_moe_onednn` 4265.0
    and `pp_pipeline` 0.0 (`run2.log:2732`); `moe_vram_runtime` =
    max(0.56, 0.62) = 0.6 (`run2.log:2750`); `tp_vram_runtime` 0 (TP off);
    `onednn_reorder` 60.0 (`run2.log:2753`); and `moe_q8_workspace` = 10 x
    1.1 x 850.0 / 512 = 18.3 MB (`uc:27274-27277`, with `n_expert` 512 and
    `n_expert_used` 10 at `run2.log:156-157`, and the 850.0 MB largest
    tensor at `:2753`). So SCRATCH = 7775.9 - 1700.0 - 4265.0 - 0.6 - 60.0
    - 18.3 = **1732.0 MB**. The surviving terms give the same: 850.0 + 32.0
    + 850.0 (`cpu_staging`, `run2.log:2750`) + a recipe term of 0.0. The
    recipe term is not printed; it is 0 because only `Q1_0` and `NVFP4`
    expert tensors get a recipe workspace (`uc:27380-27399`), and Qwen's
    are `Q8_0` (`run2.log:81`), so the two computations are independent
    (r5 n-2). The inputs print to 0.1 MB, so the figure is scored to ±0.2
    MB.
- **The runtime pre-allocate stops carrying the same two terms** (r3 m3, the
  r2 m-2 pattern).
  - `ggml_sycl_configure_host_zones_for_plan` pre-allocates a second pinned
    pool of `onednn_scratchpad_bytes + dma_staging_pool_bytes +
    pp_pipeline_scratch_bytes + pp_moe_onednn_scratch_bytes`
    (`gs:5649-5653`). rev 3e charged the PP pipeline scratch and the PP
    MoE oneDNN ring twice on the host, in host SCRATCH through the RUNTIME
    function and here. After C4 neither host pool carries them.
  - **Neither host pool owns them**: both are VRAM allocations (the term
    trace above). The runtime pool is not where either term is used: once host
    zones are configured, `host_pool_alloc` (the runtime chunks,
    `pinned-pool.cpp:209-212`) serves only the KV-zone miss fallback
    (`uc:15483-15495`); the other callers take it only when zones are not
    configured (`uc:15440`, `:15492`, `:10565`, `:20614`,
    `common.hpp:2982`).
  - C4 drops them from `runtime_bytes`, which becomes
    `onednn_scratchpad_bytes + dma_staging_pool_bytes`.
    `reserve_pp_moe_onednn_scratch` (`gs:5654-5681`) is a VRAM reserve and
    is unchanged.
- **The criteria** (C4 lead run).
  - No `Pinned pool grow_zone: zone 0` line appears after the pre-size,
    through the end of the completion. Zone 0 is WEIGHT
    (`pinned-pool.hpp:62-67`). Its positive control is the
    `Pinned pool pre_allocate: grew` line of the same run: both are INFO,
    so the absence counts only in a run that printed the control (r3 m2).
  - **SCRATCH (zone 3)** grows only through the ledger, **under default
    flags** (r4 m-3): every zone-3 `grow_zone` line follows a `[HOST-ARENA]
    SCRATCH demand` line (a context's, or a `DEVICE` holder's, the
    attention plan's included) with `grown` > 0, and is one chunk of
    `max(2048 MB, align_up(deficit + n x alignment))`. A zone-3 grow from a
    miss is a FAIL, and names the unplanned tenant by its cohort.
    - Two opt-in paths allocate compute buffers the notify does not cover.
      `LLAMA_SPLIT_MODE_TENSOR` selects `SYCL_HostCompute`
      (`lc:901-904`), whose TP path allocates host `COMPUTE`
      (`gs:42818-42838`). `GGML_SYCL_HOST_COMPUTE=1` selects
      `SYCL_CpuOffloadCompute` (`lc:928-931`), whose `alloc_buffer` is the
      device function (`gs:43009`; llama.cpp-g2a7). A run with either set
      is VOID for this criterion. No lead arm sets them.
  - **No host zone grows during PP or TG** (§M62 (E)), on the Qwen and
    GPT-OSS no-`-c` arms: no `[HOST-POOL] grow_zone(` WARN (the phase gate,
    `pinned-pool.cpp:633-635`, which warns at its default mode 1,
    `:625-628`, once `graph_compute` has set the phase, `gs:105352`), no
    `[HOST-ARENA] DEVICE holder ... exceeds its planned` WARN and no
    `[HOST-ARENA] DEVICE pools dev=%d unplanned` WARN (r5 m-7, r6 I-3),
    and no `[MoE-CPU] pinned expert pool too small` WARN
    (`gs:75815-75819`).
  - **The pools are what was planned** (r6 I-3; P4). Each `holder=`
    line for the two pools equals the pre-registered figure (§4.4 pools
    plan) and the same run's `[MOE-POOL] Pinned buffer pool` and
    `[CPU-EXPERT-POOL] Initialized` lines
    (`pinned-buffer-pool.cpp:110`, `cpu-expert-pool.cpp:106-109`) print
    the same D, K and N. A pool line with no `holder=` line, or the
    reverse, is a FAIL. The same run's `[HOST-ARENA] DEVICE pools dev=%d
    committed` line carries that D, K and N (r7 I-2).
  - **The pools' zero case is scored beside its control** (r7 I-2 (3)). No
    `holder=` line for either pool and no `[MOE-POOL]` or
    `[CPU-EXPERT-POOL]` line counts only when the same run prints the
    device's `DEVICE pools ... committed D=0 ... host_expert_nodes=0` line
    (INFO). Without it the pool planning did not run, or INFO did not
    reach the log, and the zero case is **VOID**, which counts as failed.
  - **The staging criteria depend on the switch** (§M69 (a); r5 I-1).
    - **Default arms** (no `GGML_SYCL_ATTN_HOST_DISPATCH`): the run prints
      `[ATTN-HOST] staging not planned (host dispatch off)` exactly once,
      prints no `[ATTN-HOST] planned staging` line, and registers zero
      `DEVICE` staging demand (no `[HOST-ARENA] SCRATCH demand
      holder=attn_host_staging` line).
    - **That line is the default arms' witness, and it is required.** The
      switch also gates every other `[ATTN-HOST]` line (`planned staging`,
      `host dispatches`, `first host dispatch`, `plan attributed 0`), so
      with the switch unset their absence is automatic and proves nothing
      by itself. `staging not planned` is the one `[ATTN-HOST]` line that
      prints with the switch **off**: the plan walk prints it on that
      branch, at WARN, whether or not any host-KV node exists. So each
      default arm REQUIRES it, exactly once. A run where it is absent is
      **VOID**, which counts as failed, and every `[ATTN-HOST]` absence the
      arm scores is scored only beside it. That absence is scored only on a
      run whose log carries the INFO positive control, a `[HOST-ARENA]
      SCRATCH demand ctx=` line from the same notify (r6 n-4); without it
      INFO did not reach the log, and the absence is **VOID**.
    - **Dispatch-on arms** (§7 C4 steps 3b and 5b, which set
      `GGML_SYCL_ATTN_HOST_DISPATCH=1` on their own command line): an
      `[ATTN-HOST] planned staging` line for each dispatching device, zero
      `[ATTN-HOST] staging request ... exceeds the planned` WARNs, and no
      phase-gate WARN. The positive control is C4's dispatch counter,
      `[ATTN-HOST] host dispatches dev=%d fa=%zu set_rows=%zu refusals=%zu`
      (§4.4 staging plan), read from its **last** line per device, where
      `fa` counts `FLASH_ATTN_EXT` interceptions, whatever followed them.
    - **Terms** (r7 m-2). A *refusal* is an `[ATTN-HOST] staging request
      ... declining host dispatch` line; it names its kind. A *failed
      graph* is llama's `ggml_backend_sched_graph_compute_async failed with
      error` line (`lc:3968-3970`), whatever produced it (r9 n-2). The
      backend's `[SYCL] recoverable runtime fallback failed` line
      (`gs:107778`) before it attributes it to the refusal channel, which
      a refusal at any of the six segmented sites now takes (§M69 (b)
      second and third amendments); a failed graph without it is still a
      failed graph. An *abort* is rc 134 with the
      message the code prints: a `ggml_abort` line, `<file>.cpp:<line>:`,
      or the ordinary loop's `GGML_ASSERT(ok) failed` (`gs:94499-94506`). A
      *crash* is any other non-zero rc or signal: SIGSEGV (rc 139),
      `timeout` (rc 124), the watchdog's `[SYCL-WATCHDOG]` `_Exit(1)`, or
      a GPU-fault line. An abort or a crash is FAILED, never VOID.
    - **Precedence on a dispatch-on arm** (r6 I-2 (3), r7 m-2, m-3). The
      rules apply in this order, and the first that matches decides:
      1. **Any refusal, failed graph, abort or crash is a FAIL of C4**,
         with exactly two exceptions, both decided here and nowhere else:
         - (a) **dsab.** A refusal is dsab's premature release (§4.4
           Lifetimes, §M69 (c)) only when the stderr shows, in this order,
           for the refusing device: an `[ATTN-HOST] planned staging
           dev=<d>` line whose figure for the **refusing kind** (`q`,
           `dst`, `mask`, `val` or `ids`, as the refusal names it) is > 0
           (r7 n-2); then a counter line for that device, which marks a
           SYCL backend free after the commit; then the refusal. It is
           reported as dsab's and not scored against C4. A refusal after a
           zero figure for its kind, with no plan line, or with no free
           between the plan and the refusal is an unplanned slot, and it is
           C4's.
         - (b) **Pre-existing, 5b only** (r7 m-3). An abort or crash whose
           abort line or GPU-fault line also prints on 5b's parent-tree
           control (§7 C4 step 5b) is pre-existing, reported and not C4's.
           A refusal or a failed graph is never pre-existing: the parent
           cannot print a staging refusal, and its segmented sites discard
           the return instead of failing the graph.
      2. **No counter line for the device after its plan line** is a FAIL:
         the switch was on, so a backend free or the helper must print
         one.
      3. **Flash attention must have resolved on** (r7 n-5). The arms do
         not pass `-fa`, so llama resolves it and prints `Flash Attention
         enabled` (INFO, `lc:1281`) or `Flash Attention not supported, set
         to disabled` (WARN, `lc:1271`). With the latter there is no
         `FLASH_ATTN_EXT` node, and the staging criteria are **VOID** for
         the arm, whatever the counters read. With neither line, INFO did
         not reach the log, and they are VOID too. With the former, `fa` =
         0 while `set_rows` > 0 is a FAIL: host KV writes were intercepted
         while attention over that KV was not, so the two predicates
         disagree.
      4. **The only VOID for "no host attention ran"**: `fa` = 0,
         `set_rows` = 0 and `refusals` = 0, rc 0, and no `[ATTN-HOST] first
         host dispatch dev=<d>` line (r7 m-2). It cannot absorb a refusal,
         a failed graph, an abort or a crash, which rule 1 has already
         decided, nor a run that dispatched and then lost its last counter
         line.
      5. Otherwise (`fa` > 0), the staging criteria above are scored, and
         the first-dispatch line `[ATTN-HOST] first host dispatch dev=<d>`
         must precede the device's first counter line with a non-zero
         `fa`.
  - **KV (zone 1)** grows at context init, one chunk per host-KV buffer
    that does not fit, and the sum of its growth covers the printed host KV
    (§7 C4 step 3). On a run with no demotion it does not grow.
  - No `host compute buffer ... exceeds notified` WARN.
  - **No `refusing SCRATCH host_zone_boundary_check` WARN** in the whole run
    (r3 I-4, §M57 (c)). It is a WARN (`uc:20290-20298`), so it prints at
    `llama-completion`'s default verbosity. Its positive control is the
    zone-select test's `GRAPH` record, which trips it.
  - **Under `GGML_SYCL_STRICT_LEASES=1`, freeing a context prints no
    `[HOST-ZONE] leaked` line and no ownerless-lease WARN, and does not
    abort** (§M57 (b)), across a `llama-cli` run and a multi-`-m`
    `llama-bench` switch (§7 C4 step 5).
- **The boundary with llama.cpp-23mk** (`docs/design/sycl-alloc-zone-routing.md`,
  23mk's worktree; addendum 7 (c)). 23mk routes **device**-scope requests.
  For host sites, 5efe owns role, zone, lifetime and sizing, and publishes
  them in the site table above. 23mk owns each site's conversion. Its class-H
  rows cite that table: the nine host landings this change moves, including
  `cpu-expert-pool.cpp:62`, `pinned-buffer-pool.cpp:88` and the
  `ggml_sycl_attn_host_staging_get` rows (`gs:81975-82074`); row 56, beside
  `gs:39959`'s host-pinned fallback (and llama.cpp-nl0f for the zero-copy
  read); and row 118, the DMA staging pool (r2 m-2 below). Host SCRATCH reads
  neither the RUNTIME function nor any copy of its terms (addendum 11 (1)).

**The DMA staging pool in mmap mode** (r2 m-2).
- It is 1700 MB: 2 x `dma_streamed` (`run2.log:2752`), where `dma_streamed`
  is "largest tensor the host->device weight stream can carry"
  (`zone-sizing.hpp:80`, `uc:27537`).
- It is charged twice: in SCRATCH (`uc:27727-27730`) and in the runtime
  pre-allocate (`gs:5648-5651`).
- It exists for streaming, and §5.3 refuses streaming for every
  `FILE_MAPPED` source. So in mmap mode it is an idle reservation, which P4
  forbids.
- C4 sets `dma_streamed = 0` when no host weight is `PINNED`-backed. Both
  charges then vanish, and the log prints `DMA staging pool: 0.0 MB (no
  pinned host weights)`.
- `-lm none` keeps it, pending 23mk's row 118.

**Pre-registered Qwen zones and pre-size after C4** (r1 C-1, r2 I-1).
- Host experts are all AOS, registered in place (`gs:33442-33462`), so
  `weight_host_bytes` = 0 and `host_cpu_owned_bytes` = 0. The host total is
  experts only: 61671 host tensors = 20557 groups x 3 (`run2.log:2824`).
- `host_zone_weight_bytes = 0 x 1.2 + k_host_runtime_reserve_bytes (64 MiB,
  uc:27103)`, and nothing else reaches WEIGHT. So `[HOST-ARENA] configured
  zones: WEIGHT=` reads **64.0 MB**, against 122924.2 MB in the baseline
  (`run2.log:2893`).
- **The pre-size is two pools, and the "Committed" line counts both**
  (`gs:5615-5688`). Run2's 68 chunks were 65 + 3 (`run2.log:2892`, `:2897`,
  `:2900`). Every figure here is on run2's vehicle, `llama-completion`
  (§M60 (b)).
  - **Base** (`pre_allocate_host_pool`, `gs:5643`): WEIGHT + KV + STAGING +
    SCRATCH = 64.0 + 64.0 + 1719.2 + 1732.0 = **3579.2 MB**, which is **2
    chunks**: `Pinned pool pre_allocate: grew 2 chunks for 3579.2 MB
    working set` (the format at `pinned-pool.cpp:255`).
  - **Runtime** (`pre_allocate_runtime_chunks`, `gs:5649-5653`): today
    oneDNN scratchpad + DMA pool + PP pipeline + PP MoE oneDNN. Run2's need
    was 6056.9 MB (`:2897`): 1700.0 of DMA pool, 0.0 + 4265.0 of the two PP
    terms (`run2.log:2732`), so 91.9 MB of oneDNN scratchpad. C4 drops the
    DMA pool (r2 m-2) and the two PP terms (r3 m3), so the need is **91.9 MB,
    1 chunk**: `pre_allocate_runtime: grew 1 chunks for 2048.0 MB (total
    need 91.9 MB)`.
  - So `[HOST-ARENA] Committed` reads **6144.0 MB in 3 chunk(s)**, against
    139264.0 MB in 68 (`run2.log:2900`). rev 3e's 5 chunks, 10240.0 MB,
    kept the RUNTIME read and the two device-SCRATCH terms.
  - **Scored on the printed figures, not the chunk counts.** Chunk counts
    collide: a build that keeps the two device-SCRATCH terms also commits 3
    chunks, and one that misses r3 m3 commits 5, as rev 3e's C4 did. The
    printed working set, need and `SCRATCH=` figure tell every row apart:

    | lands | `SCRATCH=` | base working set | runtime need | Committed |
    |---|---|---|---|---|
    | C4 as specified (11 (1) and §M62 (A), r2 m-2, r3 m3) | 1732.0 MB | 3579.2 MB (2) | 91.9 MB (1) | 3 chunks, 6144.0 MB |
    | §M62 (A) missed (`onednn_reorder`, `moe_q8` kept) | 1810.3 MB | 3657.5 MB (2) | 91.9 MB (1) | 3 chunks, 6144.0 MB |
    | addendum 11 (1) missed (rev 3e's C4) | 6075.9 MB | 7923.1 MB (4) | 91.9 MB (1) | 5 chunks, 10240.0 MB |
    | r2 m-2 missed | 1732.0 MB | 3579.2 MB (2) | 1791.9 MB (1) | 3 chunks, 6144.0 MB |
    | r3 m3 missed | 1732.0 MB | 3579.2 MB (2) | 4356.9 MB (3) | 5 chunks, 10240.0 MB |
    | r2 m-2 and r3 m3 missed | 1732.0 MB | 3579.2 MB (2) | 6056.9 MB (3) | 5 chunks, 10240.0 MB |

    With r2 m-2 missed, the DMA term still leaves SCRATCH, since addendum 11 (1)
    removes it by its own rule; it stays only in the runtime need.
- **Falsifiers.**
  - A WEIGHT zone above 64.0 MB must equal 64 MiB + 1.2 x the reordered-expert
    bytes, and the placement report must print that sum. Otherwise the split
    missed an accumulation site.
  - A base working set other than 3579.2 MB (±0.2) names the zone that
    moved, from the `configured zones` line.
- **Re-checked against the early-plan fix** (§F2a I-9 addendum 2).
  - The configured zones come from the late plan: `configured zones`
    (`:2893`) follows the late `Totals` (`:2821`) and equals
    1.2 x 102383.5 + 64. The late plan already excluded the table.
  - What the record changes is the early plan (§3.1): the early STAGING
    `per_inflight` falls to about 1702 MB, and the early host total to
    102383.5 MB before the split.
  - After C4's split, the early and late `[PLACEMENT] Totals` both print
    charged host 0 and mapped 102383.5 MB. These witness lines are added to
    C4's run.

### 4.5 The budget, the ledger, the working set, and the host-fit gate

**What the ledger charges.** th32 §4.2 makes `host_ledger_budget()` the one
host budget, and every resident host byte charges it. `FILE_MAPPED` bytes are
clean page cache: the kernel can drop and re-read them, and `MemAvailable`
counts them as available. So:

- **`-lm mmap` / `auto` (unlocked):** mapped bytes are **not** charged. They
  appear in the plan as `host_mapped_bytes`, and the load prints them.
- **`-lm mmap+mlock` (locked):** the tier locks its host-executed lock
  segments (§4.6, §5.1), never a lazy range, and those bytes **are** charged
  at full size, because locked pages are no longer reclaimable.
  - A lock failure refuses the load, naming `errno`, the bytes and
    `RLIMIT_MEMLOCK`, as th32's `host-cpu-lock-failed` does.
  - No new environment variable is added: the policy is the existing load
    mode.

**The resident working-set check** (§F1 Q1: WARN, not refuse).
- An unlocked mapping keeps its pages only while memory allows.
- The plan computes:

  ```
  resident_room = MemAvailable_pre - expected_host - host_headroom(MemTotal)
  mapped_ws     = host_mapped_bytes            (host-executed ranges only)
  ```

- If `mapped_ws > resident_room`, the load logs one WARN,
  `[HOST-PLAN] mapped host weights exceed the page cache that can stay
  resident: mapped_ws=... resident_room=... (MemAvailable_pre=...
  expected_host=... headroom=...)`, and proceeds.
- It uses `host_headroom(total)` below, not the P3 constant (r2 m-4). On a
  62 or 32 GiB host a 29 GiB floor would make the WARN fire on almost every
  load, against a headroom of 15.5 or 8 GiB.
- The plan report (7v5c's dry-run and `[PLACEMENT] Totals`) carries the same
  three figures.
- A mapped overflow costs refaults, not an OOM (§5.2). Refusal stays for
  pinned and anonymous bytes, through th32 T2 over 7v5c.

**The cold-expert interface** (§F1 Q2, r1 m-9). 5efe defines the tier and its
terms and adds no cold-set logic. What yl3v and 7v5c consume is:

- per expert entry: `host_backing`, and whether it is host-executed;
- `host_mapped_bytes` and `resident_room` from the plan;
- permission to request `FILE_MAPPED` as the backing of a cold host expert,
  instead of an `O_DIRECT` STORAGE read, if yl3v chooses it.

yl3v's objective must read `mapped_ws` against `resident_room`, not only the
charged bytes. Once host experts are uncharged, a search over charged bytes
alone finds DeepSeek's cold set empty and leaves its experts faulting from USB
behind a single WARN. That policy is yl3v's to set; 5efe only guarantees the
terms exist and are published.

**The headroom and the monotone budget** (the lead's point (c); r1 I-4, §F2).
One named expression, in the one function:

```
k_host_p3_floor      = 29 GiB                       // this host's measurement:
                                                    // 9g22's 24 GiB P3 floor + 5 GiB margin (th32 §4.3)
host_headroom(total) = max(total / 32, min(k_host_p3_floor, total / 4))
host_ledger_budget() = min(total x pct, MemAvailable - host_headroom)   when MemAvailable > host_headroom
                     = refusal "host-headroom-exhausted"                 otherwise
```

- **Monotone.** The `available/2` branch at `uc:4227-4230` goes. The budget
  never rises when `MemAvailable` falls, and at zero headroom-free memory
  every pinned request is refused with a named reason that prints
  `MemAvailable`, `MemTotal` and the headroom.
- **Unknown total.** The `total_mem == 0` branch (`uc:4217-4219`) keeps its
  `min(8 GiB, available/4)` reserve, which is monotone once the `/2` cliff is
  gone.
- **Why the P3 constant.** P3 (9g22's gate precondition, which th32 §4.3 makes
  bind every load) requires `MemAvailable_pre - expected_host - 5 >= 24` on
  this host. A ledger that leaves only 8 GiB admits plans P3 refuses, which
  th32 §4.5 has to argue around.
  - The ledger and P3 read the same number **only where `total/4 >= 29 GiB`,
    that is `MemTotal >= 116 GiB`** (r2 m-4). On smaller hosts the headroom
    is `total/4` or `total/32`, and P3's measured floor does not describe
    them.
  - th32 §4.3 is asked to cite `host_headroom(total)` rather than the literal
    `24 + 5`.
  - The `min(..., total/4)` cap keeps this host's policy from becoming a
    universal allocation cap on smaller hosts.
- **`GGML_SYCL_HOST_RESERVE_MB` composes; it does not bypass** (r2 m-4).
  Today a valid value replaces the whole budget (`uc:4171-4181`). After C5 it
  is a cap: `budget = min(override, host_ledger_budget())`. It can lower the
  budget, and it can never take it past `MemAvailable - host_headroom`. The
  refusal still names the headroom when the override is the smaller term.

| host | `MemTotal` | headroom | `MemAvailable` | budget (new) | budget (old) |
|---|---:|---:|---:|---:|---:|
| this host, now | 246.9 GiB | 29.0 | 150.8 | **121.8** | 142.8 |
| this host, baseline | 246.9 | 29.0 | 154.8 | 125.8 | 146.8 |
| this host, near the floor | 246.9 | 29.0 | 29.5 / 28.9 | 0.5 / refusal | 21.5 / 20.9 |
| 62 GiB NAS | 62.0 | 15.5 | 50.0 | 34.5 | 42.0 |
| 32 GiB host | 32.0 | 8.0 | 20.0 / 7.9 | 12.0 / refusal | 12.0 / 3.95 |

(On this host `total x 90%` = 222.2 GiB never binds.) Until th32 T2 lands,
the change applies to the `[HOST-ARENA]` formula at `uc:4200-4231`; T2 then
lifts it into `host_ledger_budget()`.

**The host-fit gate** (the lead's point (b)) is not duplicated here.
- The refusal is th32 T2's `host-budget-exceeded`: WARN-only in phase (a),
  enforcing in phase (b).
- The number is 7v5c's dry-run.
- 5efe supplies their inputs: `host_backing` per entry, the four terms of §4.4,
  the working-set figures, and the locked/unlocked policy. 7v5c's printout
  gains the mapped lines and the working-set check.

### 4.6 The mapping shape, prefetch, locking, and the lazy ranges (§T13a, §T13c, th32 r15 m-4)

- **5efe owns the exclusion, because it owns the mapping.** th32 stays off
  mmap entirely (§T13c), and after T6 its tables leave `lazy_read`. The
  exclusion set therefore comes from the **lazy-range record** (§3.1), never
  from `lazy_read`'s set. Two exclusion sites read the record, both through
  `excluded_for_file(idx)`, the union of the `LAZY_READ` rows
  (`deferred_for_file`) and the `STORAGE` rows (`storage_for_file`), and so
  only eligible rows (§3.1, §F2a I-9 addendum 3):
  - the tier, through the hook's `lazy_ranges` (§4.3);
  - `llama_mmap`, through `init_mappings`.

  A table diverted to STORAGE is excluded exactly as a `LAZY_READ` one is. An
  ineligible table is not in the set, and is mapped and prefetched as the
  ordinary weight it is.
- **Three segment sets, one pure function** (r1 I-2, §F2; defined exactly per
  r2 m-16). `file_mapping_segments(tensor_ranges, lazy_ranges, page)`, where
  `tensor_ranges` is the byte ranges of the tensors in this `SYCL_Mapped`
  buffer and `lazy_ranges` is `excluded_for_file(idx)`, returns:

  | set | definition | rounding | bound on lazy bytes |
  |---|---|---|---|
  | map segments | the union of `tensor_ranges`, each range rounded **outward** to pages, then merged. It is **not** a complement: bytes that belong to no tensor of this buffer are never mapped | outward, including at a lazy boundary (GGUF alignment is 32 B, so a lazy and a non-lazy tensor can share a page) | at most one page of lazy bytes mapped per lazy edge (the shared page) |
  | `WILLNEED` segments | the host-executed subset of `tensor_ranges`, each rounded **inward**, minus every page that holds a lazy byte | inward | 0 |
  | lock segments | as `WILLNEED`, under `-lm mmap+mlock` only | inward | 0 |
  | `MADV_RANDOM` edge windows | for each lazy range, `[lazy_begin - W, lazy_begin)` and `[lazy_end, lazy_end + W)` intersected with the map segments, plus the shared pages | inward from each lazy boundary | n/a (advice only) |

  (§F2a I-2(a): only map segments round outward. Every advice and lock set
  rounds inward, so no advice or lock ever reaches a lazy page other than the
  shared page's `MADV_RANDOM`.)

  - Lazy ranges are never in `tensor_ranges`, because deferred tables are not
    `SYCL_Mapped` tensors (§4.4). A lazy byte is mapped only through a shared
    page.
  - Pages strictly between map segments stay the reservation's `PROT_NONE`
    anonymous memory. They are never faulted, and a tier read of one is a
    `SIGSEGV`: a loud bug, not a silent read.
  - `file_mapping_segments` lives in the device-free `file-mapping.cpp`.
    libllama's `ranges_complement` (`src/llama-mmap.cpp:444`) is a different
    set and is not reused.
- **Fault read-around** (r1 m-13, addendum m-19, r2 I-3).
  - Linux's synchronous mmap read-around is **centred on the fault**:
    `start = pgoff - ra_pages/2`, `size = ra_pages`. It is per file, not per
    VMA. So a fault near either edge can read into a neighbouring lazy range:
    forward from a segment that ends at it, and **backward** from a segment
    that starts after it. That covers CPU execution faults and the staging
    bounce's `memcpy` faults alike.
  - **Its reach** on each side is `ra_pages/2`. `ra_pages` is the backing
    device's `read_ahead_kb`, doubled when the open file description carries
    `POSIX_FADV_SEQUENTIAL`. The tier's `dup` fd shares the loader's
    description, and `llama_mmap` sets that advice on it
    (`src/llama-mmap.cpp:475`). Every bdi on this host reads
    `read_ahead_kb = 128`, so the reach here is 64 KiB, or 128 KiB with the
    advice.
  - The tier sets `MADV_RANDOM` on a window of
    `W = max(2 x read_ahead_kb, 2 MiB)` at **both** edges of every lazy range,
    which is at least twice the reach. That splits the VMAs and disables
    read-around there.
  - Device-only ranges are also staged through the mapping, so the same edge
    advice covers them.
- **The second residual path: the loader's own whole-file mapping** (addendum
  m-19, r2 I-4).
  - `llama_mmap` still maps whole files during the load. `load_all_data` reads
    the tensors that are not in `SYCL_Mapped` through it: `CPU_REPACK` and
    plain CPU sources, and lazy tables on the CPU path before th32.
  - Today that mapping advises `MADV_RANDOM` only over the lazy range itself
    (`src/llama-mmap.cpp:541-543`), and it sets `POSIX_FADV_SEQUENTIAL` on the
    fd (`:475`), which doubles the read-around. A fault in the neighbouring
    VMA can therefore read into the lazy range.
  - **Worse, it advises `MADV_HUGEPAGE` over every non-lazy range** shrunk to
    pages (`src/llama-mmap.cpp:517-532`), and that includes the page runs
    next to each lazy range. Its own comment names the effect: "a fault in a
    VM_HUGEPAGE area reads a whole 2 MiB folio before POSIX_MADV_RANDOM is
    honoured" (`:518-522`).
    - In `do_sync_mmap_readahead` the `VM_HUGEPAGE` branch comes before the
      `VM_RAND_READ` return. It aligns the index down to 2 MiB and reads
      2 MiB, or 4 MiB without `VM_RAND_READ`.
    - THP is `[madvise]` on this host.
    - So a `MADV_RANDOM` window inside a `VM_HUGEPAGE` VMA does nothing. A
      fault in the page before a lazy range reads up to the next 2 MiB
      boundary into it. A fault after it reads back to the previous
      boundary.
  - **C4's `llama_mmap` change, both halves.**
    - The lazy-range advice covers `[lazy_begin - W, lazy_end + W)`, from the
      record, not the lazy range alone.
    - The same window, **widened outward to 2 MiB alignment**, is excluded
      from `MADV_HUGEPAGE`, and advised `MADV_NOHUGEPAGE`. The explicit
      `NOHUGEPAGE` also covers THP `[always]`, where a VMA is huge unless
      told otherwise.
  - The tier's own mapping never advises `MADV_HUGEPAGE` (below), so it needs
    only the `MADV_RANDOM` windows.
- **The bound, stated per mapping** (r2 I-4). This counts lazy bytes that a
  non-lazy fault can bring into the page cache:
  - **The tier's mapping:** the shared page at each lazy edge, and nothing
    else. There is no `HUGEPAGE`, and the `MADV_RANDOM` windows are at least
    twice the reach.
  - **`llama_mmap`'s mapping:** the same, once C4's window is both
    `MADV_RANDOM` and `NOHUGEPAGE`.
  - **Either mapping:** plus whatever bcachefs reads at extent granularity
    (256 KiB, th32 §2.4) around a page it does read.

  The gates below measure what remains. The C3 gate measures the band that
  read-around would reach. The C4 gate measures the band a 2 MiB folio would
  reach.
- **No `MAP_POPULATE`, no `MADV_HUGEPAGE`** from the tier. Huge pages on a
  file mapping depend on the filesystem's large-folio support, which is not
  established for bcachefs here. th32 §2.4 measured that bcachefs reads whole
  256 KiB extents anyway.
- **After staging**, device-only ranges get `POSIX_FADV_DONTNEED` through the
  owner's `dup` fd (§4.1). Their page cache goes back to the host experts'
  working set. A re-materialization after a VRAM eviction re-reads them.
- **Locking** (th32 r15 m-4, addendum I-10, §F2a).
  - **The tier.** It calls `mlock(2)` directly, once per inward-rounded,
    host-executed lock segment, in `unified_cache_raw_lock_file_segments`.
    Inward rounding matters because `mlock` rounds a mid-page start down to
    the page. The tier never uses `llama_mlock`, whose `grow_to` locks
    `[addr + size, target)` contiguously from the mapping base
    (`src/llama-mmap.cpp:826-840`).
  - **`mlock_bufs`.** `SYCL_Mapped` is excluded from the `mlock_bufs` block
    (`lm:2782-2787`). Without that, `use_mlock && is_host` would `grow_to` the
    whole buffer: every device-only source (about 25 GB on Qwen) and the
    reservation's `PROT_NONE` holes.
  - **The loader's prefix lock is fixed in 5efe**, because 5efe is what turns
    mmap on for SYCL (§F2a I-10). `lml:2120-2123` stops calling
    `grow_to(weight->offs + n_size)`. It calls a new
    `llama_mlock::lock_range(offs, size)` instead, which locks the tensor's own
    range: outward to pages, except inward at any end whose page holds a
    record-deferred byte. `llama_mlock` records the ranges it has locked, and
    `grow_to` remains only for whole-buffer locks (`mlock_bufs`).
  - Qwen is immune (its table is all of shard 3). DeepSeek's layouts are
    exposed (Q3_K_M shard 2; Q2_K shards 1 and 4), which is why both lock REDs
    put a lazy range before a locked range.
- **Until th32 lands**, lazy tables keep today's path (§2.5): the loader's lazy
  mapping, `MADV_RANDOM`, no prefetch, no lock.
  - The tier changes none of it, and the loader's mapping gets prefetch 0 on
    SYCL (§4.2), which can only reduce what is prefetched.
  - The tier never double-maps a lazy range beyond the one-page edge bound
    above.
  - The upstream prefix lock that §2.5 describes is fixed in C3 (above), so a
    plain `CPU_Mapped` tensor after a lazy range under `-lm mlock` no longer
    locks the range either.
- **The residency gate** (C3; r1 I-1, §F2; hygiene per r2 m-14). It measures
  the page-cache residency of the **file**, never `mincore` on the tier's
  hole.
  - **Fixture.** A 64 MiB file in the build directory: tensor A (16 MiB), a
    lazy range (24 MiB), then tensor B (16 MiB), with 32 B-aligned,
    deliberately non-page-aligned boundaries. The test sets
    `POSIX_FADV_SEQUENTIAL` on its fd, as the loader does, so the reach
    matches production.
  - **Eviction, per arm.** Before **each** arm, every mapping of the fixture
    is unmapped (`FADV_DONTNEED` skips mapped pages), then
    `POSIX_FADV_DONTNEED` runs over the whole file, and the probe must read
    about 0 resident. If it does not, that arm exits 77: the directory
    cannot be evicted (tmpfs, or a filesystem that ignores the advice).
  - **Probe.** `cachestat(2)`, called as `syscall(__NR_cachestat, ...)`
    because glibc has no wrapper here (Linux 6.5+; this host runs 7.1). On
    `ENOSYS`, or where `__NR_cachestat` is undefined, the fallback is
    `mincore` over a **test-owned**, separate `PROT_READ` mapping of the
    file. That reports the page cache of those file pages whichever mapping
    faulted them. `mincore` reports every page resident for a file the
    caller neither owns nor can write, which does not apply to a fixture the
    test creates.
  - **The two bands.**
    - The **interior** is the hole minus a 1 MiB guard band in from each
      edge. The `WILLNEED`, lock and `pread` arms measure it.
    - The **edge band** at each lazy edge is `[shared page + 1 page,
      edge + reach)` inward from the edge, where the reach is computed from
      the fixture's bdi `read_ahead_kb` (the bdi of its `st_dev`), x2 for
      the fd's `FADV_SEQUENTIAL`, /2. The read-around arms measure it (r2
      I-3). If the reach does not extend at least 2 pages past the shared
      page, those arms exit 77, since an empty band passes vacuously.
  - **Arm 1 (the property).** Map through the tier with the lazy hole, apply
    the `WILLNEED` segments, and read every byte of A and B, including the
    bytes that share a page with the hole. That reads outward-rounded bytes
    without a `SIGSEGV`, which also tests I-2(a). Then the interior must read
    **0 resident pages**.
  - **Arm 2 (the mlock case).** Arm 1 under the lock path, with the lazy range
    placed *before* a locked tensor. Same bound. It exits 77 if
    `RLIMIT_MEMLOCK` is below the fixture size.
    - **It cannot fail by construction** (r2 m-14): the hole is `PROT_NONE`,
      not file-mapped, and edge read-around stays inside the guard band. So
      it gets its own mutant, which maps the hole and locks the prefix
      `[0, B_end)`. The mutant must read the interior resident.
  - **Arm 3 (negative control).** A `GGML_SYCL_PRIVATE_TESTING` mutant that
    passes the hole **as a tensor range**, that is, adds it to
    `tensor_ranges` and leaves it out of `lazy_ranges` (the one reading
    m-16's definition allows). It maps and `WILLNEED`s the hole, and must
    read resident pages in the interior. This shows the gate can fail.
  - **Arms 5 and 6 (read-around, forward and backward; addendum m-20, r2
    I-3).**
    - Arm 5 faults the mapped page adjacent to the hole's start, and arm 6
      the one adjacent to its end. Both are real faulting pages, not
      addresses the tier does not map.
    - With the edge `MADV_RANDOM` in place, the **edge band** must read 0.
    - Each has a **control twin** with no edge advice, which must read
      resident pages in the edge band. This shows the arm could fail.
    - A mutant that advises only the segment tail fails arm 6.
  - **Arm 4 (synchronous positive control) runs last**, on its own freshly
    evicted fixture. `pread` 1 MiB inside the interior, then the probe over
    that MiB must read resident. There is no polling and no `WILLNEED` race.
    It runs last because its `pread` populates the hole, plus readahead past
    the MiB.
- **The loader-side gates** (addendum I-9, I-10, r2 I-4). The same fixture and
  probe are driven through libllama, with no SYCL:
  - **`llama_mmap` prefetch exclusion.** Construct `llama_mmap` with
    `prefetch = -1`, the `init_mappings(true)` case, and lazy ranges from a
    record. Wait 10 s, then the interior must read 0. The mutant is an
    **empty** record (the T6 diversion with ranges taken from `lazy_read`); it
    must read resident within the same 10 s, polled.
  - **The `HUGEPAGE` arm** (r2 I-4).
    - A second fixture places the lazy boundaries at 2 MiB + 1 MiB offsets,
      not 2 MiB-aligned.
    - The arm faults the page adjacent to each lazy edge through
      `llama_mmap`'s mapping.
    - The **folio band** is the lazy bytes from the shared page + 1 page up
      to the edge's 2 MiB boundary, 1 MiB here. It must read 0.
    - The mutant leaves the window `MADV_HUGEPAGE`, and must read the folio
      band resident. If the mutant does not populate it (THP `[never]`, or
      a kernel whose file THP path ignores the advice), the arm exits 77
      naming that, rather than passing on a control that cannot fail.
  - **The loader's ranged lock.** Lock a tensor that sits after the lazy
    range through `llama_mlock::lock_range`; the interior must read 0. The
    mutant is `grow_to`, the whole-prefix lock, which must read resident. The
    test exits 77 if `RLIMIT_MEMLOCK` is below the fixture size.
  - **On Qwen** the lead gate is `fincore` on shard 3, which holds only the
    table (§2.5), against the bound pre-registered in §7 C4.

### 4.7 th32 interface, and the figures that move

- **Separate tiers, separate readers.** STORAGE reads through th32's own
  `O_DIRECT` fd and never maps (§T13a, §T13c); `HOST_CPU` is anonymous.
  `FILE_MAPPED` never holds a lazy table. th32 §5.1's argument that "a HOST
  residency backed by mmap is not HOST at all" is consistent with this design:
  `FILE_MAPPED` is not charged as resident, and the plan says so.
- **T1's census** ("`MMAP_TRACKED`: legacy file mmap tracked for weights",
  th32 §3.1) changes: `MMAP_TRACKED` becomes `UNREGISTERED`, and `FILE_MAPPED`
  is a real tier. The sites are listed in §4.8. th32 T1 and 5efe C2 can land
  in either order, because every site keys on names and the enum value is
  "next free" (§4.1).
- **Figures in th32 §4.5 that move** once 5efe lands. th32 re-pins them:
  - "Host experts, exact" (Qwen 98.7-101.7 GiB, DeepSeek 164.7-167.7 GiB
    today) moves out of step 2 into `host_mapped_bytes` under `-lm mmap`.
    Under `-lm mmap+mlock` it stays.
  - "Other host weights (`token_embd`)" moves the same way.
  - The WEIGHT zone term shrinks to 1.2 x the reordered experts, plus rnh3's
    named terms, plus the 64 MiB reserve. Non-weight host requests move to
    SCRATCH (§4.4). For Qwen the WEIGHT zone is 64.0 MB (§4.4).
  - Every "`MemAvailable_pre` needed" row drops by the moved bytes.
    - For Qwen after (s), a HOST table would need about
      180.9 - 99.3 = 81.6 GiB of `MemAvailable_pre` for step 2 plus the table,
      before the working-set check.
    - With the experts mapped, the working-set check (§4.5) then decides, and
      the PLE table and the experts compete for one page cache. That trade is
      th32's to re-pin, through 7v5c.
  - §2.3's `mmap_support` bullet and T6's flip move to 5efe commit 1.
  - The ledger headroom (§4.5) is `max(total/32, min(29 GiB, total/4))`,
    which is 29 GiB here, as P3 already carried. On this host the planner's
    fit column moves to the P3 column; on smaller hosts it moves less.

### 4.8 The tier-switch census (sites 5efe touches)

`MMAP_TRACKED` today; every site is renamed to `UNREGISTERED` or taught
`FILE_MAPPED`:

| site | today | after |
|---|---|---|
| `uch:5593` enum | `MMAP_TRACKED = 2` | `UNREGISTERED = 2`, plus `FILE_MAPPED` at the next free value (§4.1) |
| `uch:5614` `memory_location::tier` default | `MMAP_TRACKED` | `UNREGISTERED` |
| `uc:2408` `alloc_tier_name` | "mmap_tracked" | "unregistered", "file_mapped" |
| `uc:2613` `offload_stats_note_alloc` | counts as shared | `UNREGISTERED` unreachable; `FILE_MAPPED` gets its own counter |
| `uc:14812` `unified_free_record` | returns without freeing | `UNREGISTERED`: unchanged; `FILE_MAPPED`: its own branch, before this one and before the `::free` branch (§4.1) |
| `uc:15125` `unified_alloc` refusal | refuses `MMAP_TRACKED` | refuses both (mappings come only from the file-mapping front door) |
| `uc:16825` registry host sum | pinned + mmap-tracked | pinned and file-mapped, reported separately |
| `uc:21637`, `:21654` `query_registered_location` / `query_location` | unknown → `MMAP_TRACKED` | unknown → `UNREGISTERED`; registry `FILE_MAPPED` → `FILE_MAPPED` |
| `gs:38172`, `:38207` `kv_layer_alloc::tier` default | `MMAP_TRACKED` | `UNREGISTERED` |
| `gs:81268-81274` `ggml_sycl_alloc_tier_name` | 3 cases | one case per tier |
| `gs:5731` (`expert_resolve_tier` name switch) | `HOST_MMAP` | `HOST_MMAP` now means "served from a `FILE_MAPPED` owner" |
| `tests/test-cpu-gpu-soa-interaction.cpp:178`, `:180`, `:182` | `alloc_tier::MMAP_TRACKED` x3 | `alloc_tier::UNREGISTERED` (r1 m-3; the rename breaks this build otherwise) |
| `tensor-types.hpp:36-40` `memory_tier::MMAP`, used at `gs:14317`, `:19256`, `:19279`, `:34771`, `:35670`, `:98936` (r2 m-11) | "File-backed mmap (slowest, fallback)"; in fact the initial value and "not found" result of the dead `get_cached_tensor_ptr` (§2.7) | `memory_tier::UNRESOLVED`, so no name in the tree says "mmap" without meaning a `FILE_MAPPED` owner |

The registry's `alloc_type::MMAP` (r1 m-4) is split and **named for its
release semantics** (r2 m-6), because "host memory that is not USM" would
also describe a file mapping, and invite registering one under it.
- **`HOST_MALLOC`**: registered host memory that `unified_free_record`
  releases with `::free`, the pinned-cap overflow (`uc:14922-14929`).
- **`HOST_OWNED_ELSEWHERE`**: registered only so the registry can identify
  it; `unified_free_record` never frees it. This is the tiered KV's synthetic
  base, a `std::vector` that its own context clears (`gs:39620-39624`,
  `:39641-39642`, released at `:38408-38415`).
- `alloc_type::FILE_MAPPED` is new.

The ten `MMAP` consumers, all in C2:

| consumer | meaning today | after |
|---|---|---|
| `alloc-registry.hpp:26`, `:143`, `:223` | enum, host-accessible set, commented usm mapping | `HOST_MALLOC` and `HOST_OWNED_ELSEWHERE`; `FILE_MAPPED` added to the host-accessible set |
| `common.hpp:2863` | maps to `usm::alloc::unknown` | all three map to `unknown` |
| `dpct/helper.hpp:1327` | pointer-attribute classification | all three |
| `gs:38735`, `:38760` | tiered-KV memcpy for the synthetic base | `HOST_OWNED_ELSEWHERE` only |
| `gs:39643` | registers the KV synthetic base | `HOST_OWNED_ELSEWHERE` |
| `gs:87750` | (classification) | all three |
| `mem-ops.cpp:151` | non-USM host classification | all three (a `FILE_MAPPED` source takes the bounce) |
| `set_rows.cpp:321` | non-USM host classification | all three |
| `uc:14927` | malloc-overflow `::free` | `HOST_MALLOC` only; a `FILE_MAPPED` or `HOST_OWNED_ELSEWHERE` record never reaches it (§4.1) |
| `uc:21635` | `query_registered_location` | all three, each to its own tier |

`cache_location::HOST_MMAP` (39 sites) keeps its name. Its derivation changes
in one place each (`uc:8068-8081`, `:8235-8248`, `:6504-6545`): it is
`HOST_MMAP` only when `query_location` says `FILE_MAPPED`. Once every weight
source is owned (C6), `register_host_expert` and `register_host_weight` refuse
an `UNREGISTERED` pointer. That also retires the direct-stage "raw mmap
fallback" as an unowned path.

---

## 5. Hazards (item 4)

### 5.1 `mlock`

- **Which lock and where.** The tier does its own locking and never uses
  `llama_mlock`. It calls `mlock` per inward-rounded lock segment (§4.6) in
  `unified_cache_raw_lock_file_segments` (`file-mapping.cpp`, an allowlisted
  site, §4.1), at `SYCL_Mapped` buffer creation, and only when the load mode
  locks and the segment's entry is host-executed.
- **The loader's locks.** `mlock_bufs` (`lm:2782-2787`) and the per-tensor
  lock in `load_all_data` (`lml:2120-2123`) skip `SYCL_Mapped` buffers.
  Otherwise `mlock_bufs` would lock the whole buffer, device sources included,
  because `is_host` is true (r1 m-6). They keep applying to `SYCL_Host`, and
  the per-tensor lock becomes the ranged `lock_range`, not the prefix
  `grow_to` (§4.6).
- **Lock time.** `mlock` without `MLOCK_ONFAULT` faults the whole segment in
  synchronously. On Qwen that is about 100 GiB from the USB members at
  0.29-1.78 GB/s (th32 §2.4), so a locked load takes one to six minutes. That
  is the price of the guarantee, and it is paid only under `-lm mmap+mlock`.
- **Charging.** Locked file pages are unevictable, so they are charged (§4.5).
  The working-set check does not apply to them.
- **`RLIMIT_MEMLOCK`** is unlimited on this host (th32 §3.1). Elsewhere the
  refusal names it.

### 5.2 Page-fault latency on CPU-executed experts

- **Storage.** bcachefs over USB SSDs, with NVMe as a promote cache: 1.78 GB/s
  sequential when served from the NVMe cache, 0.29-0.47 GB/s settled, and a
  4 KiB random read costs about 256 KiB of device reads (th32 §2.4).
- **Expert size.** Qwen3.8 has 512 experts and uses 10 per token
  (`run2.log:26-27`, `:156-157`). Each `ffn_*_exps` tensor is 850 MB for 512
  experts, so one expert matrix is about 1.7 MB and contiguous (r1 m-2). A
  cold expert therefore faults as a short sequential run under readahead, not
  as 4 KiB randoms: roughly 1-6 ms per matrix at the rates above.
- **A fully cold decode token** touches 10 experts x 3 matrices x 48 layers x
  1.7 MB ≈ 2.4 GB, which is seconds per token. So the mapped experts must be
  warm: the tier `WILLNEED`s their ranges at load (§4.6), and the working-set
  check (§4.5) says whether they can stay warm.
- **Reclaim under other tenants.** This host's permanent tenants (the qemu VM,
  Emby, Frigate) can grow. The kernel then drops clean mapped pages first, and
  decode slows to storage speed with no error. `-lm mmap+mlock` is the
  opt-out.
- **Faults happen on `CpuExpertPool` workers**, inside
  `ggml_sycl_cpu_expert_mul_mat_batched`, so a stall holds a worker, not the
  submitting thread. The existing host waits on that path (th32 §2.3) are
  unchanged.
- **`SIGBUS`.** A mapped file that is truncated while mapped, or a bcachefs
  checksum failure on fault, raises `SIGBUS` in a worker instead of an `EIO`
  return. Mapping the loader's own open file (§4.1) removes the
  replaced-by-path case, but not truncation in place. This is the same
  exposure upstream's mmap has; it is recorded, not mitigated.

### 5.3 A GPU reading mapped pages (forbidden)

Mapped pages are not USM. A device read of one is at best a slow
driver-staged copy and at worst a device loss (`uc:14044`'s own comment). The
ways a weight's host pointer can reach a device today, and what each becomes:

| site | today | after |
|---|---|---|
| `mem_handle::resolve(device)` of a `FILE_MAPPED` handle | n/a | returns empty; `device_mappable()` is false |
| `stream_dma` (`uc:13959`), the dispatch-time streaming path reached from `mmq.cpp:7639`, `mmvq.cpp:23880`, `dmmv.cpp:3817`, `getrows.cpp:2765` | a `HOST_MMAP` source is copied straight to the device (`uc:14043-14046`); a custom `copy_fn` is tested first (`:14041-14042`) | **typed, counted refusal** (r1 I-3, §F2), described below the table |
| `ggml_sycl_resolve_or_host_tensor_ptr` (`common.hpp:4922-4945`) | returns the raw pointer for any non-device host pointer | for a `FILE_MAPPED` registration, **aborts naming the tensor** (r1 m-7), described below the table |
| layout availability (`gs:62114-62119`) | `weights_evictable && is_host` → available, and AOS is available unconditionally | for a `SYCL_Mapped` buffer neither branch applies: availability for (tensor, device, layout) comes only from a cache entry (r1 m-7) |
| `direct_stage_weight`'s plain copy (`uc:6167-6175`) | `mem_copy` from a copy handle | unchanged: `mem_copy` bounces a non-USM source through a 64 MiB pinned window (`mem-ops.cpp:700-729`); this is load or materialization, not streaming |
| reorder `fill_fn`s (`gs:34840-34845`) | read `src_ptr` on the CPU | unchanged (CPU reads are allowed); see §5.4 for their staging |
| direct-stage last resort (`uc:6504-6545`) | registers an unrecognized source in place as `HOST_MMAP` | registers the `FILE_MAPPED` slice with its owner; an `UNREGISTERED` source is refused (C6) |
| preload's host-planned copy (`gs:34701-34735`, r2 m-7) | `memcpy`s a late-plan host weight into an uncharged pinned WEIGHT allocation | served in place from the mapping, or a planned, charged copy (§4.3 site table); a CPU copy, never a device read |

**The `stream_dma` refusal.**
- The check runs at entry, before the `copy_fn` test, so custom-copy callers
  are covered.
- `dma_stream_result` (`uch:3622`) gains `refused_file_mapped`, and a
  `stream_dma_file_mapped_refusals` counter increments.
- Callers route a refusal like `mmap_direct_failed` today: to the CPU executor
  (`ggml_sycl_cpu_fallback_graph`, `mmq.cpp:7653-7658`), because a host-tier
  weight executes on the CPU. C4 routes `refused_file_mapped` **explicitly**
  in all four callers (`mmq.cpp`, `mmvq.cpp`, `dmmv.cpp`, `getrows.cpp`),
  not by relying on `mmap_direct_failed`.
- **The refusal is terminal in `getrows`' retry loop** (r2 m-5). That loop
  halves the slice on any non-ok result (`getrows.cpp:2762-2795`), so a
  deterministic refusal would repeat `log2(slice/min)` times, each counted
  and each logging `[GET_ROWS] DMA retry`, and its fallback keys on
  `mmap_direct_failed` (`:2806`). C4 breaks out of the loop on
  `refused_file_mapped` before the retry test, and routes it to the CPU
  fallback. A host test over a fake `stream_dma` that refuses asserts one
  call, one count and no retry line.
- Streaming mapped weights is forbidden by P3.

**The `resolve_or_host` abort.** Returning null would not be safe:
`binbcast.cpp:794` passes null to `needs_raw_host_staging`, which returns
false for null (`binbcast.cpp:394`), and then uses null as the kernel source.
- C4 audits all eight callers: `common.cpp:2537`, `:2538`, `binbcast.cpp:794`,
  `gs:44274`, `:51369`, `:51417`, `:64972`, `:90733`.
- A source gate pins that list.
- A caller that legitimately needs a CPU view of a weight uses
  `ggml_sycl_host_data()`.

**Gates.**
- The device-bound copy counter `file_mapped_device_copies_outside_materialization`
  counts a device-bound copy of `FILE_MAPPED` bytes, bounced or direct,
  outside a load transaction or a cache materialization
  (`direct_stage_weight` or a fill), **at the instrumented chokepoints** (r2
  m-17):
  - `mem_copy`, `mem_copy_async` and `mem_copy_ptr_async` (`mem-ops.hpp:13-62`);
  - `unified_cache::copy_to_device_async` (`uc:10799`);
  - `stream_dma` (`uc:13959`).

  A copy through any other path is not counted. The standing
  `llama.cpp-v4jk` audit covers those. The lead's load gate reads the counter
  as 0.
- **Where the lead reads both counters** (r2 m-18). One line,
  `[FILE-MAPPED] counters: stream_dma_refusals=%llu
  device_copies_outside_materialization=%llu`, printed through
  `GGML_LOG_WARN`. WARN maps to `LOG_LEVEL_WARN` (2) (`common/log.cpp:533`),
  so it survives `llama-completion`'s default threshold, `LOG_LEVEL_INFO`
  (3, `common/log.cpp:30`). It does **not** survive `llama-cli`'s, which is
  `LOG_LEVEL_ERROR` (1, `tools/cli/cli.cpp:36`), so every `llama-cli` arm
  that reads it passes `-lv 4` (§7, r3 C). It prints:
  - once at the end of `ggml_backend_sycl_model_load_end`, after the
    preload, so a killed run still has one;
  - once at model unload.

  A missing line is a failed criterion, not a zero.
- `stream_dma_file_mapped_refusals` must also read 0 on a correct build: a
  refusal means dispatch routed a host-tier weight to a device kernel.
- The standing `llama.cpp-v4jk` audit (no route stages a host operand to a
  device) covers the rest of dispatch.

### 5.4 The staging bounce

- **Waits.** `mem_copy`'s bounce waits on each 64 MiB slice before refilling
  it (`mem-ops.cpp:716`), and on the last one unless the caller retains the
  handles (`:729`). That is a host wait, but only at load or materialization,
  where the preload already runs.
- **The bounce buffer** is a standalone pinned-pool allocation, role
  `EXPERT_STAGING`, released individually when its handle drops
  (`mem-ops.cpp:472-483`, allocated at `:704-705`). The transient pinned
  footprint of **`mem_copy`** is 64 MiB per copy in flight.
- **Other staging paths are unchanged by this design, and the 64 MiB claim
  does not cover them** (r1 m-16):
  - `ggml_sycl_fill_reordered_gpu` stages the whole `src_size` through
    `ggml_sycl_staging_pool` (`gs:27190-27200`, whose non-USM branch is
    commented "mmap'd").
  - The oneDNN WOQ and XMX fills make whole-source host copies
    (`gs:32190` onward, `:32437` onward).

  Their footprint is today's, bounded per call by the source tensor (850 MB
  on Qwen).
- **When it engages.** The bounce engages only if a mapped pointer is
  classified non-USM (`host_ptr_is_usm_accessible`, `mem-ops.cpp:700`). C4's
  device-labelled RED pins that, since a mis-classified mapped source would
  take the direct-copy path instead.

---

## 6. The double residency of VRAM weights

**Decision: fixed here for mmap mode, by construction; a ticket for
`-lm none`.**

- In mmap mode the persistent host source of a device weight is the
  `FILE_MAPPED` mapping. Its pages are clean page cache: reclaimable, and
  dropped after staging (§4.6). A VRAM weight is resident once.
- The entry's recorded `src_ptr` (`uc:6213`) stays valid: it points into a
  mapping that lives as long as the entry's `storage_owner`. A
  re-materialization after a VRAM eviction re-reads the file through page
  faults, inside a cache materialization, which the §5.3 counter allows.
- **`-lm none`** stays supported on SYCL but is not the default (§F1 Q3). It
  keeps `SYCL_Host`, so it keeps the double residency. The fix there is
  different: a transient pinned window per preload slot, and a file re-read
  for re-materialization, which needs the loader's file handle to outlive the
  load. **Ticket to file after review:** "SYCL `-lm none`: device weights stay
  resident in `SYCL_Host` after preload (double residency)", depending on
  5efe.

---

## 7. Implementation plan (item 5)

Every commit is RED → GREEN. GPU work is the lead's, and each commit names
what the lead runs. Test rules (r1 I-6, §F2):

- A **host-only** test links no SYCL runtime and touches no queue, so it never
  reaches `stream_dma` or a device `mem_copy`. It carries `LABELS
  "sycl;host"`, plus `source-contract` for Python gates.
- A test that needs a device carries **`LABELS "sycl;cache;mem-handle"`** and
  `ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:1;LD_LIBRARY_PATH=..."`, and
  is **lead-run**. It is never run by an implementer or a subagent. Form 2 of
  CLAUDE.md excludes it through `-LE`.
- A RED must fail behaviourally, against a named mutant where needed, not only
  by not compiling (r1 m-12).

The lead's GPU runs share one protocol (r1 I-7):
- the selector is pinned;
- `grep -E '^(MemAvailable|Shmem|AnonPages|Cached|Mlocked|Unevictable):'
  /proc/meminfo` is sampled **every 1 s**;
- the watchdog floor is `MemAvailable < 25 GiB` = 26214400 kB, the same unit
  and value as the criteria;
- each run is done once;
- `journalctl -k` is checked for `GT reset|guc_id|CAT error` after it.

**Every model gate set carries a default-context arm** (owner ruling
2026-09-27), and every such arm is scored:
- The **Mistral** gate already runs `llama-completion` with no `-c`, so it is
  its own default-context arm (n_ctx_train 32768). 5efe's runs of it add
  `-v`, so that its context line prints (the VOID guard below).
- Every **Qwen** and **GPT-OSS** gate below that pins `-c 4096` has a no-`-c`
  twin.
- **The vehicle** (lead's clarification, 2026-09-27). A default-context arm
  is `llama-cli`, `llama-completion` or `llama-server` with no `-c`, whose
  context defaults to `n_ctx_train`. `llama-bench` (`n_ctx = p + n + d`) and
  `llama-perplexity` (512) stay at their fixed contexts; a "no `-c`" bench
  arm is vacuous, so C4 step 6 has none. A **programmatic** arm (a harness on
  `llama_context_default_params`, whose `n_ctx` is 512,
  `src/llama-context.cpp:5091`) must set `n_ctx = 0` explicitly (`:541`), and
  is **VOID** unless `llama_n_ctx(ctx) == n_ctx_train`. No 5efe test is such
  an arm today. The rule binds any that is added.
- **Verbosity is part of the vehicle** (r3 C, m2). A criterion that reads a
  log line is void if the vehicle drops that line's level, and an absence
  criterion then passes vacuously.
  - `common_log`'s default callback prints a line only at `verbosity <=
    common_log_verbosity_thold` (`common/log.cpp:542-546`). GGML INFO maps
    to 4 and WARN to 2 (`:532-533`).
  - `llama-completion` keeps the default threshold, 3 (`common/log.cpp:30`,
    `LOG_DEFAULT_LLAMA`), so it prints WARN and drops INFO. The Qwen arms
    read INFO lines (`pre_allocate`, `configured zones`, the `SCRATCH
    demand` line), so they pass **`-v`**, as run2 did.
  - `llama-cli` sets `params.verbosity = LOG_LEVEL_ERROR` (1) before it
    parses (`tools/cli/cli.cpp:36`), so it drops WARN too, the demotion line
    (`gs:18382`) and `[FILE-MAPPED] counters` included. `-lv N` overrides it:
    `common_params_parse_ex` first sets the threshold from that default
    (`common/arg.cpp:769`), and the `-lv` handler, which runs while the
    arguments are parsed afterwards, sets both `params.verbosity` and the
    threshold (`:3950-3962`). So **every `llama-cli` arm passes `-lv 4`**,
    which admits WARN and INFO.
  - `-lv 4` does not move the digit scoring. The answer is console output
    on stdout (`common/console.cpp:75`, `:1145-1149`), and every log line
    goes to stderr (`common/log.cpp:113-116`). The arm captures the two
    streams to **separate files** and scores `grep -cx '1, 2, 3, 4, 5'` on
    the stdout file only, since a merged capture can tear the answer line.
- **Every default-context arm carries the VOID guard** (§M54, §M61 (c),
  §M62 (C); r4 m-1).
  - The arm's stderr must print the context line, `llama_context: n_ctx
    = <N>` (INFO, `src/llama-context.cpp:717`), with `N` = `n_ctx_train`.
    The guard is the padded regex, never the literal: `grep -cE
    'llama_context: n_ctx +=  *<N>$' <stderr file>` must be 1.
  - The `llama_context:` prefix is load-bearing. On `llama-cli` a literal
    `n_ctx = <N>` also matches the server slot line (`slot load_model: ...
    new slot, n_ctx = <N>`), so it can pass on the wrong line. The regex is
    not anchored at `^`, because `-v` and `-lv` lines carry a timestamp
    prefix.
  - **Negative control.** The `n_ctx_seq` line and, on `llama-cli`, the
    `slot load_model` line are selected by name and must score 0 under the
    same regex: `grep -E 'n_ctx_seq|slot load_model' <stderr file> | grep
    -cE '<regex>'` = 0, with the name filter itself returning at least one
    line.
  - An arm that fails the guard is **VOID**, not FAIL. The line is INFO,
    so every such arm runs at a level that prints it: `-v` on
    `llama-completion`, `-lv 4` on `llama-cli`.
  - **The arms it covers, all of them:** C1's lead gate (the Mistral gate
    on both cards, N = 32768, with `-v`; r5 m-1); C4 steps 3 and 3b (Qwen,
    N = 262144); C4 step 4 (the Mistral gate on both cards, N = 32768); C4
    step 5's no-`-c` arm and 5b (GPT-OSS, N = 131072); C5's Qwen and GPT-OSS
    no-`-c` arms; and C6's Mistral gate, GPT-OSS no-`-c` arm and Qwen
    no-`-c` arm.
  - **Not default-context arms, named:** C1's witnesses W1-W3. They are
    no-`-c` `llama-completion` runs, but each is killed before any
    `llama_context` exists (§7 C1), so there is no context line to guard.
    No other 5efe arm runs without `-c`.
- **Every pre-registered number names its vehicle** (§M60 (b)). The device
  compute buffer depends on `n_outputs_max` (`src/llama-context.cpp:2051`),
  which is part of the vehicle. The Qwen figures are `llama-completion
  -no-cnv -v` at the default `n_ubatch` of 512, run2's vehicle. The GPT-OSS
  figures are GDC3's vehicle: `llama-cli` at its default `n_ubatch` of 512,
  with `n_outputs_max` 1 (`tools/server/server-context.cpp:1015`,
  `lc:2051`), for C4 step 5 and 5b, C5 and C6 (r5 m-2). A run with `-ub`,
  or through another tool, is a different row. No figure measured on one
  is scored on the other's arm. A `llama-completion` GPT-OSS no-`-c` arm
  would be a different row, and 5efe has none.
- **A GDC3 citation names its point** (§M61 (a); r4 m-7). GDC3 carries two
  points, today's tree and the zhcn+moua fixpoint. A 5efe arm run on a tree
  with neither zhcn's nor moua's change scores against today's-tree point.
  On a tree with both, it scores against the fixpoint. On a tree with only
  one, it is VOID for the GDC3 criterion, and the lead is told which tree it
  was. Each such arm also carries zhcn's compute criterion in its §M73 (b)
  form (r7 m-6), cited, not restated: the one GDC3 states for the context
  the arm creates, never the load envelope `c(P)` scored against it. The
  replay that yields the arm's point charges that context's `compute_ctx`
  (§M73 (c)), a property of the vehicle (§M60 (b)).
  - **The point is a replay of this run's inputs** (§M63 (a); r5 m-4). On a
    zhcn tree, GDC3's point is its replay applied to this run's own printed
    input lines, never a prior run's printed figure. So step 5's N, range
    and host-KV figure `M`, and the zone-1 count `ceil(M / 2048)`, are
    taken from that replay of the same run. A different N passes if and
    only if the replay of that run's inputs yields it.
- **An env-gated trace line is scored only when armed and live** (§M66).
  5efe scores one such emitter, `[HOST-ZONE-ALLOC]`, printed only under
  `GGML_SYCL_HOST_ZONE_ALLOC_TRACE=1` (`uc:3990`, `:4002-4008`). It is scored
  at five points: §4.4's WEIGHT pre-registration, C4 step 2's WEIGHT
  criterion, step 3's KV exact form and WEIGHT criterion, step 5's WEIGHT
  criterion, and the §M57 (a) RED. Every arm that scores it:
  - sets the variable **on its own command line**, not in an inherited
    environment;
  - is **trace-live**: its stderr holds at least one post-pre-size
    `[HOST-ZONE-ALLOC]` line of any cohort. Each arm names the line it uses
    (r5 I-3): C4 step 2, a `zone=SCRATCH cohort=backend_host_buffer` line
    (at `-c 4096` there is no `zone=KV` line); step 3, the KV exact sum;
    step 5 and its RED, the `kv_host_buffer` and `backend_host_buffer`
    lines. C5 and C6 carry the same checks;
  - is otherwise **VOID**, and VOID counts as failed. For the RED, which
    scores the presence of a line, a non-live run is VOID, never a RED that
    did not fire.
  The dispatch counter of the dispatch-on arms (§4.4 criteria) is a WARN,
  printed whenever the switch is on, not a trace line. Its arms set the
  switch on their own command line as well.
- llama.cpp-uize part 3 has landed (`35ce2e65f`, `812c7d790`, `7259727e6`,
  `ae4204ba7`, all in this tree). So a KV cache that does not fit VRAM is
  re-placed to the host tier, and its layers' attention runs on the CPU. The
  runtime prints `[SYCL-PLAN] KV overflow re-placed to host tier: N layer(s)
  demoted, S SWA (X MB host KV, layers A..B) on device D for n_ctx=C; the
  device has F MB free for KV. ...` (`gs:18382`).
- **PASS** needs correct tokens, and that line with the pre-registered N and
  range. **Any** `runtime KV update rejected` line is a FAIL: the refusal is
  now only the fallback for when the host tiers cannot hold the KV either, so
  no attribution to uize remains. Garbage tokens, an abort, or `Shmem`
  growth are also 5efe FAILs.

**C1. Restore `mmap_support`, and the lazy-range record** (§3, §3.1).
- RED `test-sycl-mmap-support-caps-source` (Python, `tests/`, `sycl;host;source-contract`).
  - It checks that the initializer in `ggml_backend_sycl_device_get_props`
    names `.mmap_support`, binds it to `host_unified_known &&
    !host_unified_memory`, and is neither a literal `true` nor absent.
  - Today it fails on the four-initializer list.
  - It also checks that `lm:2693-2696` is still present, since C1 must not
    remove it. **C4 amends that clause** to the narrowed form of §4.3
    (r2 m-8).
- **The record** lands in C1, with these host-only tests (`host;lazy`,
  `tests/`, libllama only, no device). C1 factors three pure functions out of
  `load_tensors` so that a host test can drive them without a device:
  - `llama_lazy_mode_resolve(mode, caps)`: the moved fallback block, over a
    list of device caps;
  - the pre-fill: (declaration, `weights_map`, `hparams`, `mode_pre`,
    `mode_post`, `taker`, `SUPPORTED`) to a record;
  - the early plan's inventory filter.

  The loader calls the same functions.
  - `test-llama-lazy-range-record`:
    - It builds `llama_tensor_weight`s from a test-written GGUF of five
      tensors in two files. Three of them are declared lazy for the test's
      arch: one at 4 GiB + 1 byte of logical size, one at 4 GiB, and one
      **sparse table above 4 GiB**, its data region an `ftruncate` hole, so
      the fixture costs no disk. The GGUF also holds one tensor of the same
      name under a second arch that does not declare it (the gemma3n shape).
    - Rows equal the declared tensors' (file, offset, size) exactly, whatever
      the mode. The second arch's same-named tensor gets **no** row (the
      (arch, name) key; addendum 5).
    - **The eligibility matrix** (addenda 4 to 6), over `taker` x `mode_pre`
      x `mode_post` x `SUPPORTED`:
      - `lazy_eligible` comes from T6a's function, called and not restated,
        with `(mode_pre, true)` when `taker` and `(mode_post, SUPPORTED)`
        otherwise;
      - `deferral` is `STORAGE` iff eligible and `taker`, `LAZY_READ` iff
        eligible and not `taker`, and `EAGER` otherwise;
      - `deferral != EAGER` iff `lazy_eligible`, asserted on every row of
        every cell.
    - **The required witness** (addendum 4). With `AUTO`, a caps list holding
      one device with `mmap_support = false`, and `taker`, the sparse row's
      bit is set, its `deferral` is `STORAGE`, and the early plan's filter
      excludes it. `llama_lazy_mode_resolve` does turn `mode_post` to `OFF`,
      which is what makes this cell discriminate. **RED mutant:** the fill
      reads `mode_post` for the taker (a post-fallback fill). The bit is then
      clear and the filter includes the table.
    - **The `-ngl 0` witness** (addendum 5). A SYCL-registered load whose
      weights are not SYCL-owned has `taker` false. Every row's `deferral` is
      then `LAZY_READ` or `EAGER`, and never `STORAGE`, across the whole
      matrix. How `taker` is derived is pinned by the source test below and
      by lead witness W3.
    - **`!SUPPORTED`, off the taker.** The bit is clear under every mode, and
      the pre-fill prints the "mmap is not available ... loaded into RAM in
      full" WARN once per declared row.
    - **The attributes are per model** (addendum 7 (a)). Two records are
      filled from two `qwen4exp` hparams sets with different `ple_n_heads`,
      and each record's `attrs(name)` returns its own value, whichever was
      filled last. **RED mutant:** the attributes cached in a static beside
      the declaration, under which the first record reads the second's
      value.
    - `deferred_for_file` holds exactly the `LAZY_READ` rows,
      `storage_for_file` exactly the `STORAGE` rows, and `excluded_for_file`
      their union (§M57 (d)). **Mutant:** `STORAGE` rows in
      `deferred_for_file`.
    - **The `STORAGE` branch** (§M57 (d)). `create_tensor` on a `STORAGE`
      row, with C1's body, throws the named refusal and never calls
      `buft_for_tensor`. **Mutant:** the branch removed, under which the row
      is created as an ordinary input-layer weight in the first device's
      host buft.
    - **`consumer_layer` from hparams** (r3 m4). Two qwen4exp hparams sets
      with `is_ple_impl` bit 1 and bit 5 give `consumer_layer` 1 and 5.
      **Mutant:** the literal 1.
    - `lazy_read`'s `any()`, `has()` and `for_file()`, and the ranges passed
      to `llama_mmap`, agree with the record. `any()` is false when `taker`,
      and under `no_alloc`. `add()` accepts exactly the `LAZY_READ` rows,
      under every mode value `lazy_read` holds.
    - **The fit-pass invariant** (r3 m1; `lml:1775`). With `w = nullptr`
      (the `no_alloc` call), `add()` returns true for a `LAZY_READ` row and
      leaves membership empty; with a weight it returns true and inserts the
      name. **RED mutant:** rev 3c's `w != nullptr` term in the decision,
      under which the `no_alloc` call returns false for the `LAZY_READ` row.
    - **The per-model line** (th32 2x). The pre-fill prints exactly one
      `lazy-range record: mode_pre=<m> mode_post=<m> taker=<0|1>` line per
      model, read through a callback installed with `llama_log_set`, and it
      prints even for a model with no declared name. In the addendum-4
      witness cell it reads `mode_pre=auto mode_post=off taker=1`.
    - **RED** is a mutant, compiled into the test binary only. libllama has
      no private-testing switch, so the test links a copy of the record and
      `lazy_read` sources built with a test-local define. In it
      `lazy_read::add` keeps its own range copy and its upstream mode and
      floor tests, and the record's fill is skipped for one tensor. The
      agreement check must fail naming that tensor, and the `add()` check
      must fail on the 4 GiB row under `ON` with `taker`, which is `STORAGE`
      and which upstream's `add()` would accept.
  - `test-llama-lazy-range-prefill`:
    - It feeds three names through the pre-fill and the `create_tensor`
      check: a declared lazy name, an undeclared flagged name, and a
      declared-but-unflagged name.
    - The first gives one row, filled before any `create_tensor` call, with
      the attributes `llama_model_lazy_tensor_attrs` returns for it.
    - The other two abort naming the tensor and the arch, each in its own
      process, scored on rc 134 plus the name.
    - **Mode-independent** (addendum 4). The unflagged case aborts under
      `-lzm off` and under `AUTO` at 4 GiB, both of which leave the row
      ineligible and `EAGER`. **RED mutant:** rev 3a's check, which compares
      the flag only against non-`EAGER` rows, lets both through.
    - `value(name)` equals the weight's (file, offset, size), and a write
      after the pre-fill aborts.
    - **The ineligible early-plan witness** (th32 r19 M-1, assigned to 5efe).
      With `taker`, the early plan's filter includes an ineligible row, for
      `AUTO` at 4 GiB and for `OFF`, as an ordinary weight, and excludes the
      eligible one.
    - **RED mutant:** the row filled at `create_tensor` instead, under which
      the "before any `create_tensor` call" check fails.
  - `test-llama-lazy-declaration-source` (`host;lazy;source-contract`, r2
    I-5, addendum 5):
    - It extracts every `create_tensor` call in `src/models/*.cpp` that
      passes `TENSOR_READ_LAZY`.
    - It maps the file's model class to its `LLM_ARCH` through the
      registration in `llama-model.cpp`, and each `tn(LLM_TENSOR_*, suffix
      [, bid])` to its GGUF name through `LLM_TENSOR_NAMES`
      (`llama-arch.cpp:561`), with a bid argument becoming `blk.%d.`.
    - It requires a `llama_model_lazy_tensor_names` entry **under that arch**
      for each call, and a flagged call site in that arch's file for each
      entry.
    - It also requires `llama_model_lazy_tensor_attrs` to have a case for
      every arch that has names, and no case for any other arch.
    - The declaration returns names only (addenda 7 (a) and 8 (a)). No
      static beside it stores an hparams-valued field, and
      `llama_model_lazy_tensor_attrs` writes nothing but its return value.
      Mutant: a function-local `static llama_lazy_tensor_attrs` cache in
      `llama_model_lazy_tensor_attrs`.
    - Today it passes on the two call sites once the declaration exists, and
      it fails before C1, which has no declaration.
    - **Mutants:**
      - a flagged call added in a copy of `gemma4.cpp` with no entry;
      - an entry with no call site;
      - the `qwen4exp` entry moved under `gemma3n`, whose same-named call is
        unflagged;
      - an attribute case deleted.
  - `test-llama-lazy-range-record-source` (`host;lazy;source-contract`):
    - no consumer holds its own **range** container: `lazy_read` has no
      `ranges` member (its membership set of names stays, §3.1), and
      `init_mappings` passes the record, not `lazy.for_file`;
    - **`lazy_read::add()` reads `deferral`** (addendum 5; addendum 6 (2)).
      Its body names `LAZY_READ` and the record, and contains no
      `LLAMA_LAZY_MODE_`, no `llama_mmap::SUPPORTED` and no 4 GiB floor.
      Mutant: upstream's `add()` restored (`lml:1240-1258`), as a rebase
      would restore it;
    - **`taker`** is assigned from `sycl_model_loading_guard.active`, and no
      pre-fill input names `llama_model_sycl_hooks_enabled`. Mutant: the
      process-level selector (addendum 5's RED);
    - **T6a's third argument is exactly `taker ? true :
      llama_mmap::SUPPORTED`** (th32 2x, its §4.1 and T6a row). Two mutants
      must each fail the clause: the argument always `llama_mmap::SUPPORTED`
      (the taker loses its no-mmap eligibility, addendum 6 (1)), and always
      `true` (the non-taker marks a table lazy that `lazy_read` cannot map).
      A source clause is the only discriminator: `llama_mmap::SUPPORTED` is
      a constant `true` wherever `_POSIX_MEMLOCK_RANGE` or `_WIN32` is
      defined (`src/llama-mmap.cpp:711-715`), so on Linux and Windows no
      runtime test can tell either mutant from GREEN;
    - `mode_pre` is captured before the call to `llama_lazy_mode_resolve`, and
      the pre-fill runs after the guard and before
      `llama_model_sycl_compute_early_plan`;
    - `create_tensor`'s `TENSOR_READ_LAZY` branch checks the record and does
      not fill it, and compares the flag with membership, not with
      `deferral`;
    - the early plan's `weights_map` loop consults the record;
    - today it fails on `llama-model-loader.h:118`.
- **Lead witnesses** (r1 I-7, r2 m-1; addendum 5). Three short
  `llama-completion` runs, each with `-v` (§3.1's pre-fill line is INFO, which
  default verbosity drops), each killed at its witness lines, before the
  pinned pre-size and before any tensor data is read.
  - **The margin.** In run2 the witness lines were all printed by t = 3.9 s
    (`run2.log:243`, `:308`, `:314`, `:370`). Chunk 1 of the pinned pre-size
    was allocated at t = 8.12 s (`:2827`). So the margin is about **4.2 s**,
    not the 53 s rev 3a claimed. `:2893` is the summary after the pre-size.
  - **The belt.** C1 does not split the ledger, so its load still sizes
    WEIGHT at 122.9 GB. All three runs set `GGML_SYCL_HOST_RESERVE_MB=16384`, the
    budget override (`uc:4172-4184`), so a late kill cannot commit more than
    16 GiB.
  - **W1, lazy enabled, the taker.** Qwen on the B70 with **no** `-lm`/`-lzm`
    flags. Kill it as soon as the log shows all of:
    - the per-model line `lazy-range record: mode_pre=auto mode_post=auto
      taker=1` (C1's `mmap_support` keeps `mode_post` at `auto` on a discrete
      card);
    - the pre-fill line `lazy-range record: per_layer_token_embd.weight
      (51880 MiB) eligible=1 deferral=STORAGE taker=1 mode=auto`;
    - the early `[SYCL-PLAN] Host staging zone` line;
    - the early `[PLACEMENT] Totals` line;
    - `load_mode = mmap`.

    Pre-registered: early `per_inflight` ≤ 1800 MB, and early host total =
    102383.5 MB ± 0.1 MB.
  - **W2, the ineligible case** (th32 r19 M-1). The same with `-lzm off`, so
    the row is ineligible. The per-model line reads `mode_pre=off
    mode_post=off taker=1`, and the pre-fill line `eligible=0 deferral=EAGER
    taker=1 mode=off`. The early plan must place the table as the ordinary
    weight it is: early `per_inflight` = 103762.3 MB and early host total =
    154263.6 MB, both ± 0.1 MB, the run2 figures (`:243`, `:308`).
  - **W3, `-ngl 0` in a SYCL build** (addendum 5). Qwen with `-ngl 0` under
    `ONEAPI_DEVICE_SELECTOR=level_zero:1`, so the SYCL backend and its hooks
    are registered but the weights are CPU-owned. Kill it at the pre-fill line
    plus `load_mode = mmap`. Pre-registered:
    - the per-model line reads `mode_pre=auto mode_post=auto taker=0`, and
      the pre-fill line `eligible=1 deferral=LAZY_READ taker=0
      mode=auto`, because C1's `mmap_support` keeps `mode_post` at `auto` on a
      discrete card;
    - **no** `[SYCL-PLAN] Host staging zone` line and **no** early
      `[PLACEMENT] Totals` line: the early plan does not run.

    A `deferral=STORAGE` there is the process-level-selector defect (FAIL).
  - None of them is a load. None reaches the pre-size, and none reads the
    table, so none enters the C1-to-T6 window (§3.1, **Data edges**).
- Lead gates: the Mistral gate on `:1` and `:0` (default context), with `-v`
  so its context line prints, and the VOID guard, N = 32768 (§7 preamble;
  r5 m-1). Tokens are unchanged, since Mistral has no lazy tensor.
- **Data edge:** th32 T6a lands no later than C1 (§3.1).

**C2. The `FILE_MAPPED` tier, the raw wrappers and the census** (§4.1, §4.8).
- RED `test-sycl-file-mapping-host` (C++, links `file-mapping.cpp` only,
  `sycl;host`). It checks:
  - reservation shape: one `PROT_NONE` reservation, `MAP_FIXED` segments at
    the right offsets, and the bytes read back equal to the file's;
  - `dup` fd ownership: the fd is closed at release;
  - the size-mismatch refusal;
  - the private `munmap` counter: exactly one `munmap`, of the whole
    reservation, at the last release.

  Mutants the test must fail: release skipped (counter 0); per-segment
  `munmap` (counter > 1).
- RED `test-sycl-file-mapped-tier` (links ggml-sycl, `sycl;cache;mem-handle`,
  `level_zero:1`, lead-run). It checks:
  - `query_location` returns `FILE_MAPPED`;
  - `device_mappable()` is false, and `resolve(device)` is empty;
  - the registry has one record for the reservation;
  - the private `::free` and `sycl::free` branch counters both stay 0 at
    release.

  Mutants (r2 m-6; the rev 3a mutant was equivalent): the record registered
  as `HOST_MALLOC`; the `FILE_MAPPED` branch moved after the `zone_managed`
  branch into the `sycl::free` fallthrough.
- RED `test-sycl-raw-mmap-allowlist-source` (`sycl;host;source-contract`),
  §4.1. It uses the tokenizer, with the comment, string and `mmap (` fixture
  (r2 m-12). Mutant: an `mmap (` call added to a non-listed file.
- RED `test-sycl-alloc-tier-census-source` (`sycl;host;source-contract`). No
  `MMAP_TRACKED`, no `alloc_type::MMAP`, no `HOST_NON_USM` and no
  `memory_tier::MMAP` remain anywhere in the tree, `tests/` included. Today
  it fails on the 12 + 3 + 10 + 7 sites of §4.8 (r2 m-11).
- Also: the canonical contract §3 allowlist gains the three wrappers, and the
  static-storage census and the allowlist gates are re-run.

**C3. The segment plan and the residency gate** (§4.6).
- RED `test-sycl-file-mapping-segments` (`sycl;host`), a pure-function test.
  - Map segments are the union of the tensor ranges rounded outward, and never
    a complement (r2 m-16). `WILLNEED` and lock segments round inward.
  - At most one lazy page is mapped per edge.
  - Cases: an interleaved lazy range, a lazy range sharing a page with a
    tensor at each end, back-to-back lazy ranges, a lazy range at offset 0
    before a lock segment, and bytes that belong to no tensor of the buffer
    (never mapped).
  - Mutants: inward map rounding (fails the shared-page read), outward
    `WILLNEED` rounding (fails the zero-lazy bound), the complement reading
    of map segments (maps the no-tensor bytes).
- RED `test-sycl-file-mapping-residency` (`sycl;host`), §4.6's arms.
  - The fixture lives under `${CMAKE_BINARY_DIR}`, which is not tmpfs.
  - Each arm has its own eviction precondition, and exits 77 if it fails.
  - Arm 2 exits 77 if `RLIMIT_MEMLOCK` is below the fixture size.
  - Arms 5 and 6 exit 77 if the computed reach does not extend 2 pages past
    the shared page (r2 I-3).
  - Arms 1-2 are measured over the interior. Arms 5-6 and their twins are
    measured over the edge band.
  - Arm 4 runs last, on a fresh fixture (r2 m-14).
  - The must-fail cases: arm 2's prefix-lock mutant, arm 3, and the arm 5
    and 6 control twins.
  - A mutant that advises only the segment tail fails arm 6.
- The loader's ranged lock (§4.6) also lands here, in libllama, with RED
  `test-llama-mlock-ranged` (`host;lazy`). It is §4.6's loader lock gate: a
  lazy range before a locked range, the interior reading 0, and the
  whole-prefix `grow_to` mutant reading resident.
- Lead: none.

**C4. One published fact, the planner split, host role routing,
`SYCL_Mapped` and the loader** (§4.3, §4.4, §5.3). This lands as **one
commit**, because the fact and its only consumers must land together (r1 C-1,
I-5, §F2). It is the commit that changes every SYCL load.
- **Prefetch 0 on SYCL is an explicit C4 item** (addendum I-9, m-6).
  `init_mappings` is called with prefetch 0 on a SYCL load (`lm:2702`), and
  `llama_mmap` excludes the record's deferred ranges with the both-edge
  `MADV_RANDOM` window and the 2 MiB-widened `NOHUGEPAGE` window, whatever
  the prefetch value (r2 I-4).
- Host-only REDs:
  - `test-llama-mmap-lazy-exclusion` (`host;lazy`), §4.6's `llama_mmap`
    gates:
    - **The prefetch arm.** Under `prefetch = -1` there is no `WILLNEED` over
      a lazy range and no read-around into one. The mutant is an empty record
      (lazy ranges from a diverted `lazy_read`), which must read resident.
    - **The `HUGEPAGE` arm** (r2 I-4). Lazy boundaries sit at 2 MiB + 1 MiB
      offsets. Faulting the page adjacent to each edge leaves the folio band
      at 0. The mutant, with the window left `HUGEPAGE`, must read it
      resident, or the arm exits 77 naming why.
  - `test-sycl-host-role-routing-source` (`sycl;host;source-contract`, r2
    I-2, addendum 7):
    - It regenerates §4.4's **site table** from the tree, and fails on any
      difference: each row's holder, role, zone after C4, lifetime and owner.
      23mk's class-H rows cite that table, so the gate is its one source.
    - Every non-`GRAPH` row names an owner. A request site carrying
      `UNOWNED` that is not one of the table's 12 lines fails, as a ratchet,
      and the message cites llama.cpp-30r1 (addendum 10 (b)). A `GRAPH` row
      with no plan term is listed `UNSIZED` and ratcheted the same way (r3
      m4).
    - The three `DEVICE` holders' requests set `lifetime = DEVICE`, and
      `ggml_sycl_shutdown_global_runtime_pinned_owners` still releases each
      of them (`gs:7575-7590`). When it is called is llama.cpp-dsab's.
      The `DEVICE` check follows `delete sycl_ctx` and is gated on the
      per-device live-backend count reaching 0; nothing in that function
      checks a record (§M62 (D)).
    - The attention staging's persistent and IDLE paths allocate nothing
      for a slot: a request above the slot's capacity takes the named
      refusal, and the slots grow only under
      `ggml_backend_sycl_plan_host_holders` (§M62 (E); r5 n-1). The two
      per-call fallbacks are `GRAPH` (r4 m-12).
    - The plan registers staging demand only after
      `ggml_sycl_attn_host_dispatch_enabled()` returns true, and prints the
      `staging not planned` line otherwise (§M69 (a)). It selects nodes
      with `ggml_sycl_tensor_is_in_kv_host_buft`, the interception's own
      predicate, and never calls `ggml_backend_buffer_is_host` (r5 m-6).
      It takes each node's device from the scheduler's assigned backend,
      never from the helper (§M69 (d)).
    - The pools' dimension scan is one function, called by
      `moe_hybrid_init_once` and by the plan; no other code computes
      `max_dispatch_count` (r5 m-7). The two pools' `init` calls take the
      device's committed plan figures, never a local scan's (r6 I-3).
    - **Where the walk runs** (r6 I-1). In `llama-context.cpp` each call
      through the `ggml_backend_sycl_plan_host_holders` pointer follows a
      `graph_reserve` whose `split_only` argument is `true`, with no
      `graph_reserve` between them. `ggml_backend_sycl_begin_host_holders`
      is called right after each scheduler creation in `sched_reserve`
      (`lc:2019-2022`, and the retry's `lc:2061-2064`), before the measure
      (r7 I-1, §M75; r8 n-1). A begin anywhere before the creation, such
      as at the `synchronize()`, fails the gate. The
      `ggml_backend_sycl_commit_host_holders` calls run at the end of
      `sched_reserve` on its success, except under the auto-`n_ubatch`
      trial, where `try_candidate` makes them after its host-fallback check
      (r7 m-1).
    - **§M75.** `try_candidate` calls `ggml_backend_sycl_host_scratch_trial`
      with `OPEN` before its `sched_reserve`, `KEEP` beside the commit,
      and `RETURN` on both loss paths. The begin entry follows each
      scheduler creation. The pinned pool frees a chunk only in its
      destructor and in `release_zone_chunk`, and `release_zone_chunk` is
      reached only from `release_marked_chunk`, whose callers are the
      begin entry, the returns (for an empty chunk; r10 m-3), the owner's
      end (`ggml_backend_sycl_free` and an orphaned chunk's emptying free;
      r9 m-5) and `configure_host_zones` (r13 M-4). The gate also pins the
      wiring positively (r10 m-5): `ggml_backend_sycl_free` calls the
      owner-end step before `delete sycl_ctx` (`gs:84119`), and
      `sched_reserve` calls `RETURN` on its own transaction's throw path
      and `KEEP` on its success path. That `KEEP` is excepted under the
      auto-`n_ubatch` trial, as the end commit is, because inside
      `try_candidate` it would run before the host-fallback check and keep
      a losing candidate's growth.
      `RETURN` withdraws the marked chunks and files them in the
      device cache's marked set, and nothing in `OPEN` touches that set
      (r8 m-1). Withdrawal is the only writer that lowers
      `zones_[zi].size`, and `release_zone_chunk` does not write it (r9
      m-3), and only on the flag's false-to-true transition (r10 m-4).
      `sched_reserve` calls `MARK` before its measure and `RETURN_TO_MARK`
      before the pipeline-parallel retry (r9 n-6).
    - **The lock-order gate** (r13 M-2; r14 I-1, M-6; r15 I-1, M-2;
      §M81 (c)). `test-sycl-marked-set-lock-order-gate` is a pure-Python
      source gate that allocates nothing, registered `sycl;host;source-contract`
      like its sibling `test-sycl-mmap-support-caps-source`. It checks the
      scopes of the listed bodies only, and does not claim that callers
      hold no lock. In the bodies of the begin entry, `RETURN` and
      `RETURN_TO_MARK`'s filing and immediate release, the owner-end walk,
      `release_marked_chunk` and its `_locked` core, `release_zone_chunk`,
      `release_chunk_backing`, `host_release_chunk_lease`,
      `release_chunk_lease`, `free_chunk_owner`, the `released_chunk` destructor,
      `footprint_locked`, `reconcile_stranded_locked`,
      `unified_allocation_registered_exact`, `grow_into`,
      `~pinned_chunk_pool`, `configure_host_zones`'s release and
      `marked_entries`, it pins twelve clauses, each with at least one
      **planted control** in the gate's
      own fixtures; each fixture must make the gate fail with its quoted
      reason, except the one marked a negative control, which must pass,
      and the correct tree must pass:
      1. every access to the marked set sits in a scope holding
         `marked_set_mutex_` (the core's body has no lock statement; its
         calls are checked by clause 7). Control: `marked_entries` reading
         the set with no lock (`marked set access outside
         marked_set_mutex_`);
      2. that lock is taken before any pool call. Control: a pool call
         before the lock (`pool call before marked_set_mutex_`);
      3. no function that holds the pool's `mutex_` calls
         `release_marked_chunk` or names `marked_set_mutex_`. Control: a
         pool function that does (`pool mutex_ held across
         release_marked_chunk`);
      4. no lock named in canon §12.5's table, or `deferred_frees_mutex_`,
         is named inside a scope holding either lock, other than the pool's
         `mutex_` inside `marked_set_mutex_`. The gate reads the table from
         the canon file and adds `deferred_frees_mutex_` to its list by
         name (r15 M-2 (c)), because that lock is held around owner
         releases (`uc:12667`) and taken by the enqueue paths
         (`uc:12453-12557`) and is not in the table until C4's census
         commit adds it; afterwards the two lists coincide. Controls:
         `rw_mutex_` named inside the `marked_set_mutex_` scope (`census
         lock rw_mutex_ under marked_set_mutex_`), and
         `deferred_frees_mutex_` named there (`census lock
         deferred_frees_mutex_ under marked_set_mutex_`);
      5. `release_chunk_backing` is called only outside both scopes, in
         the cache-side bodies and in the collected-vector release loop of
         the begin's, owner-end's, `RETURN`'s and `configure_host_zones`'s
         walk, which sits after its scope's closing brace, and each of
         those collecting vectors is declared before its scope's lock
         statement, so an unwinding scope destroys it after the unlock (r16
         M-1). The public form's inline `released_chunk` slot has the same
         duty (r19 N-1): it is declared before the lock statement, and its
         carrier is destroyed after the unlock by its destructor (no
         `release_chunk_backing` call is involved, so the controls below
         that name that call cannot see it). The cache-side bodies cannot
         name the pool's private
         `free_chunk_owner`, so the controls name what they can:
         `release_chunk_backing` called inside the `marked_set_mutex_`
         scope (`backing release under
         marked_set_mutex_`); the begin's collected-vector loop moved inside
         its locked scope (`collected backings released under
         marked_set_mutex_`); `release_chunk_backing` called in a pool
         body that holds `mutex_` (`backing release under pool mutex_`);
         a collecting vector declared inside the lock's scope
         (`collected vector declared inside the marked_set_mutex_
         scope`); and the public form's slot declared inside its scope
         (`released_chunk slot declared inside the marked_set_mutex_
         scope`);
      6. `configure_host_zones` reads `ggml_sycl_strict_enabled()` between
         the pool's false and its return. Control: a copy without the read
         (`configure_host_zones returns the refusal without the strict
         read`);
      7. the call forms (r15 M-2 (a); r18 M-1): every call of
         `release_marked_chunk_locked` sits in a scope holding
         `marked_set_mutex_`, and no call of the public
         `release_marked_chunk` does (a self-deadlock on the non-recursive
         lock); the public form's body is the only body that takes the lock
         and then calls the core; and `host_release_chunk_lease` calls the
         public form, never the core, and the lease-drop path contains no
         allocation. The path is every frame the drop runs through (r19
         M-1; r20 M-1; §M110): `host_release_chunk_lease`,
         `release_chunk_lease`, the public `release_marked_chunk`, the
         single-slot overload of `release_marked_chunk_locked`,
         `release_zone_chunk` (shared with the walks, so the ban costs them
         nothing), the `released_chunk` destructor, and the two frames the
         destructor calls, `free_chunk_owner` (`pp:778-791`) and
         `unified_allocation_registered_exact`. The frames above
         `host_release_chunk_lease` (`release_lease_state` and
         `get_existing_unified_cache_for_device`) are unchanged by C4 and
         are outside the gate. None of those bodies contains `reserve`,
         `resize`, `push_back` or a vector declaration, detected by the
         type token `vector<` (a reference such as `auto & vec` at
         `pp:1057` is not one). The single-slot overload is selected by
         signature: the name `release_marked_chunk_locked` whose out
         parameter is a `released_chunk` with exactly one `&`, never `&&`
         (`release_chunk_backing`'s `released_chunk &&` and
         `release_zone_chunk`'s own out parameter must not match it);
         the vector overload necessarily appends to its caller's vector and
         is exempt from the ban. Controls:
         the begin's walk calling the public form inside its scope
         (`release_marked_chunk called under marked_set_mutex_`); a core
         call outside any scope (`release_marked_chunk_locked called
         without marked_set_mutex_`); `host_release_chunk_lease` calling
         the core (`lease drop calls release_marked_chunk_locked`); and a
         `reserve` in the public form (`lease-drop path allocates`); the
         single-slot overload written as a wrapper over the vector core,
         `std::vector<released_chunk> v; v.reserve(1);` (`single-slot core
         overload builds a vector`, reason `lease-drop path allocates`);
         and the same two lines in `release_zone_chunk`, in
         `release_chunk_lease`, in `free_chunk_owner` and in
         `unified_allocation_registered_exact` (`lease-drop path
         allocates`);
      8. the carrier's destructor (r15 I-1 (4); §M88 (1); §M92) takes the
         metadata the carrier holds, calls `free_chunk_owner`, calls
         `unified_allocation_registered_exact`, and then, under the pool's
         `mutex_`, subtracts `total_allocated_` only on the branch where
         the answer is GONE and appends to `stranded_`, without
         subtracting, on the other; it never names
         `ggml_sycl_strict_enabled`; and no other body writes
         `total_allocated_` for a release except
         `reconcile_stranded_locked` (`release_zone_chunk` and
         `release_chunk_backing` contain no such write). Controls: a
         destructor with the subtraction removed (`released_chunk
         destructor does not subtract the footprint`); `release_zone_chunk`
         subtracting (`footprint subtracted outside the released_chunk
         destructor`); the subtraction with no registry query
         (`released_chunk destructor subtracts without the registry
         query`); the subtraction on the registered branch (`released_chunk
         destructor subtracts on the registered branch`); the kept branch
         without the append (`released_chunk kept branch does not strand
         the footprint`); the strict abort restored on the kept path
         (`released_chunk destructor names ggml_sycl_strict_enabled`); and
         a WARN format string that names a cause, any of retry, scheduled,
         leak or ownerless (`released_chunk WARN names a cause`, §M97);
      9. the accessor `unified_allocation_registered_exact` (r16 M-3): its
         body holds exactly one lock statement, on `g_runtime_alloc_mutex`,
         compares `key()` as well as `ptr`, and writes nothing in the
         registry (no `erase`, `emplace`, `operator[]` or assignment
         through the found row). Controls: a ptr-only compare (`accessor
         compares ptr only`) and a body that erases (`accessor writes the
         registry`);
      10. the reconcile (§M92; r17 I-1; r18 M-2; r20 M-2): the gate
          matches the **comment-stripped** source of the `pinned-pool.cpp`
          bodies, so `configure_zones`'s comment that quotes the budget test
          (`pp:334-335`) is not a mention. The allow-list is **per
          mention**, not per body, and each entry is recorded as body,
          ordinal and statement kind, the ordinal counting **every**
          mention of `total_allocated_` in that body in source order,
          exempt or not (r20 N-2). On the correct tree it holds exactly:
          `footprint_locked`, one, its return; `reconcile_stranded_locked`,
          one, its subtraction; the carrier destructor, one, its
          subtraction; `grow_into`, four, ordinals 1, 3 and 4 the log reads
          (`pp:894`, `:971`, `:988`) and ordinal 2 the `+=` commit
          (`pp:967`); `~pinned_chunk_pool`, one, the reset
          `total_allocated_ = 0` (`pp:163`). The one exemption beyond those
          is mechanical: a mention is exempt only as the numerator of a
          division inside the argument list of a `GGML_LOG_*` call, the
          form of every existing log read (`pp:256`, `:352`, `:666`,
          `:814`, `:829`, `:835`, `:894`, `:971`, `:988`). Anything else, a
          comparison, a local copy or a write, is outside the allow-list.
          The gate asserts that these nine are the **only** exempt
          mentions and that the five allow-listed mentions are the only
          others, so a tenth log read, a lost one, or a second mention in
          an allowed body (`grow_into` re-reading the member at
          `pp:892`) fails it by name. It also asserts a **positive
          presence**: each of the seven decision bodies contains a call of
          `footprint_locked()` (`allocate_segmented`, `pre_allocate`,
          `configure_zones`, `grow_zone`, `allocate_from_chunks`,
          `grow_into`, `allocated()`), which is what makes dropping the
          reconcile from a body red. `footprint_locked`
          calls `reconcile_stranded_locked` before it returns, and that
          function subtracts an entry's bytes only where
          `unified_allocation_registered_exact` answers GONE. One planted
          control per body, each dropping the reconcile from that body:
          `allocate_segmented`, `pre_allocate`, `configure_zones`,
          `grow_zone`, `allocate_from_chunks`, `grow_into` and
          `allocated()` (`<body> does not call footprint_locked`; for
          `grow_into` the mutant re-reads `total_allocated_ +
          backing_size > budget_` at `pp:892`, and fails on the presence
          check and on the per-mention count); the helper itself returning the
          member
          without the reconcile (`footprint_locked returns total_allocated_
          without reconcile_stranded_locked`); a local copy in `grow_zone`
          (`grow_zone reads total_allocated_ outside a log call`); a
          write to the member in a decision body (`total_allocated_
          written outside the allow-list`); a tenth exempt-form log read
          (`exempt log mentions are not exactly nine`); a second
          mention in an allowed body (`allow-listed mention count
          differs`); a **negative control**, which must pass: a comment
          that quotes the member must not make the gate fail with
          `comment mention flagged`; and a prune with no registry query
          (`stranded prune without registry query`);
      11. the reserve and the bound (r17 M-1, M-2, M-3; r18 N-1, N-4):
          `grow_into` reserves `stranded_` after its budget check and
          before it allocates the chunk owner, sized by the chunk count
          (`chunks_.size()` and `runtime_chunks_.size()`, never a
          constant), and returns false on a failed reserve; the carrier's
          destructor asserts `size < capacity` before its append, with the
          message `stranded_ capacity %zu exhausted at %zu entries`, and
          the append is an `emplace_back` into reserved storage and nothing
          else; and the reconcile prunes in place. Controls: `grow_into`
          with no reserve (`chunk added without stranded reserve`); a
          constant reserve (`stranded reserve not sized by the chunk
          count`); a reserve after the owner allocation (`stranded reserve
          after the owner allocation`); the assertion missing or after the
          append (`capacity assertion missing or after the append`); a
          destructor that reserves (`released_chunk destructor
          allocates`); a reserve whose `bad_alloc` is not caught and mapped
          to `return false`, which on the synchronous path would escape
          because only `sycl::exception` is caught there (`stranded reserve
          without bad_alloc catch`); and a reconcile that rebuilds the list
          (`reconcile rebuilds stranded_`);
      12. the carrier count (r17 M-5; r18 N-3): `~pinned_chunk_pool`'s
          first statement asserts that no owning carrier is outstanding,
          before its `lock_guard` and the free loops (`pp:141-146`), and the
          carrier destructor's count decrement is its last statement.
          Controls: the assertion placed after the free loop (`carrier
          assertion after the free loop`); the assertion absent (`carrier
          assertion absent`); and the decrement before step 4 (`carrier
          count lowered before the footprint section`).
    - The pools' gate calls `ggml_sycl_moe_plan_has_host_experts`, and the
      plan walk never calls `ggml_backend_buffer_is_host` (r7 I-2).
    - The helper **throws `ggml_sycl_fallback_error`** at each of the six
      sites, and no site asserts, aborts or returns on a refusal (§M69 (b)
      third amendment). Its argument is a string literal (r9 n-3). The
      recording catch's rethrow of that type (`gs:97347-97349`) is
      present, and `graph_compute`'s `ggml_sycl_fallback_error` catch
      returns `GGML_STATUS_FAILED`. Its cleanup step sits immediately
      after the MMID recovery (`gs:107767`) and before the lease unpin
      (`gs:107768`), inside its own `try { ... } catch (...) {}`, and
      calls `sycl_exec_graph_clear_active` or `invalidate_moe_segments`
      (r9 m-1 (c); r10 m-1). `moe_graph_record_segments` wraps its record
      loop in a `catch (...)` that moves `recorded_segments` and
      `recorded_moe_graphs` into the context before it rethrows (§M69 (b)
      fourth amendment), and returns false at entry, before any
      submission, while either context vector is non-empty (r11 m-1).
    - No `ggml_sycl_compute_forward(` call in the MoE segmented
      record/replay functions discards its return: each of the six goes
      through the fail-closed helper (§M69 (b) as amended; r6 m-1).
    - Each of the 12 `UNOWNED` lines names a cohort (r4 m-2).
    - `select_zone` has no category clause for WEIGHT, and sends `UNOWNED`
      to WEIGHT.
    - `SYCL_Host` takes role `WEIGHT` only under `in_model_load`.
      `SYCL_KV_Host` has its own `alloc_buffer`, with role `KV`, and keeps
      its own `get_name` (r3 B).
    - **The host bufts' size and alloc share one builder** (addendum 11
      (2); r4 I-A). The gate pins the functions each iface actually names.
      `SYCL_Host`'s `get_max_size` and `alloc_buffer` both call the host
      builder and pass its intent to `select_zone`. `SYCL_KV_Host` has its
      own iface, whose two slots call the builder with role `KV`.
      `SYCL_HostCompute`'s iface names its own `get_max_size`, which calls
      `ggml_sycl_host_compute_is_tp()`, the predicate its `alloc_buffer`
      calls, and outside TP mode delegates to
      `ggml_backend_sycl_buffer_type_get_max_size`. No host `get_max_size`
      names a zone or calls `get_current_device_id()`.
      `SYCL_CpuOffloadCompute` is not pinned (llama.cpp-g2a7). This amends
      `tests/test-sycl-host-zone-config-source-contract.py`.
    - `llama_context` calls `ggml_backend_sycl_notify_host_buffer_sizes`
      through the registry's proc address, and nothing in `src/` calls
      `ggml_backend_sycl_notify_compute_buffer_sizes` (its VRAM half is not
      5efe's).
    - **The notify is ordered before the allocation it covers** (r3 A.2):
      in `output_reserve`, between the free of the old buffer and
      `ggml_backend_buft_alloc_buffer`; in `sched_reserve`, after the
      size-only `graph_reserve(..., true, sizes)` calls and before the first
      allocating one. It is called with one backend, `model.devices[0]`'s,
      not in a loop over backends (r3 A.5).
    - `ggml_backend_sycl_notify_host_buffer_sizes` tests
      `ggml_backend_is_sycl(backend)` before it reads `backend->context`
      (r3 A.4).
    - `runtime_bytes` in `ggml_sycl_configure_host_zones_for_plan` names
      neither `pp_pipeline_scratch_bytes` nor `pp_moe_onednn_scratch_bytes`
      (r3 m3).
    - The inventory's `kv_idx_width` is declared after `kv_layer_count`,
      and both demotion sites read `kv_demand_for_layer` (r3 B).
    - The notify takes an opaque backend handle plus scalars and returns
      `bool`: no other pointer and no struct in its signature, in either the
      header or the `llama_context` function pointer type (the dlopen
      shape; r3 m5). The plan entry point takes a `ggml_backend_sched_t` and
      a `struct ggml_cgraph *`, both ggml-base handles, and no llama type.
    - The notify and `SYCL_Host`'s and `SYCL_KV_Host`'s `alloc_buffer` and
      `get_max_size` take the device from `ggml_sycl_host_buft_device` and
      nowhere else: no `ctx->device` and no `get_current_device_id()` on
      those paths, and no `resolve_effective_device(0)` on the notify's
      growth path. The growth is the ensure-capacity call, never
      `grow_scratch_zone`'s additive form (§M57 (c); r4 I-D). The staging
      and pools plan is a different path: there the dispatching backend's
      device is the right one (§M69 (d)).
    - **`unified_cache_grow_host_scratch_zone` is absent** (§M68 (a); r5
      m-5). The gate counts the token in `ggml/src/ggml-sycl/` and
      `ggml/include/`, with comments stripped: zero call sites and zero
      definitions or declarations. It runs in both states of the tree.
      Before 23mk's step 7, `ggml_backend_sycl_notify_compute_buffer_sizes`
      is present and must not contain the call. After step 7, that
      function is absent, and the count is still zero.
    - The buffer-owned check runs in each host buft's `free_buffer`, and the
      backend-member check follows `delete sycl_ctx` in
      `ggml_backend_sycl_free` (§M57 (b)).
    - Host SCRATCH's fold at `uc:27727-27730` names exactly
      `max_tensor_bytes`, `k_scratch_headroom`,
      `moe_cpu_expert_staging_bytes` and `moe_host_recipe_workspace_bytes`:
      no RUNTIME-labelled term, neither `onednn_reorder_bytes` nor
      `moe_q8_workspace_bytes`, and no call of
      `unified_cache_get_planned_runtime_zone_requirement` (addendum 11
      (1), §M62 (A)).
    - The SCRATCH boundary check counts only `GRAPH` records.
    - **Mutants:**
      - the category clause restored, which moves the `CpuExpertPool` row to
        WEIGHT;
      - a RUNTIME-labelled term (`pp_moe_onednn_scratch_bytes`) added back
        to the host SCRATCH sum, and separately `onednn_reorder_bytes` added
        back (addendum 11 (1), §M62 (A));
      - a new file-scope static holder added at a `HOST_COMPUTE` site (a new
        `UNOWNED` row);
      - the notify moved after `ggml_backend_buft_alloc_buffer` in
        `output_reserve` (r3 A.2);
      - one per host buft (r4 I-A): `SYCL_Host`'s `get_max_size` with
        WEIGHT hard-coded again, or on `get_current_device_id()`;
        `SYCL_KV_Host` back on the copied `SYCL_Host` iface (the §M57 (a)
        shape); `SYCL_HostCompute`'s `get_max_size` answering from the host
        builder outside TP mode (the glkg defect reversed);
      - `pp_moe_onednn_scratch_bytes` restored in `runtime_bytes` (r3 m3);
      - the device-0 resolve restored on the notify's growth path, and the
        additive `grow_scratch_zone` restored in place of the ensure call
        (r3 I-1);
      - the notify growing `ctx->device`, the notifying backend's device,
        instead of the helper's (r4 I-D);
      - the dispatch-time slot grow restored in
        `ggml_sycl_attn_host_staging_get` (§M62 (E)), and separately in its
        IDLE branch (r5 n-1);
      - the staging demand registered with the host dispatch off (§M69
        (a));
      - the plan's predicate replaced by `ggml_backend_buffer_is_host` of K
        (r5 m-6);
      - the staging slots planned on the helper's device, not the
        dispatching backend's (§M69 (d));
      - the return of one of the six segmented `ggml_sycl_compute_forward`
        calls discarded again, once per site (§M69 (b); r6 m-1);
      - the plan walk moved after the allocating pp reserve (`lc:2055`),
        where `ggml_backend_sched_reserve` has reset the split (r6 I-1);
      - the commit moved before the allocating reserves (r6 I-1 (2)), and
        separately the trial's commit moved before its host-fallback check
        (r7 m-1);
      - the begin call removed (r7 I-1);
      - `RETURN` removed from the host-fallback loss path (§M75), and
        separately the begin moved before the scheduler creation, where
        the losing scheduler's buffers are still live;
      - the pools gated on `ggml_backend_buffer_is_host(src0->buffer)`
        (r7 I-2);
      - the helper asserting on a refusal, and separately the helper
        returning instead of throwing (§M69 (b) second and third
        amendments);
      - the pools' demand registered in `moe_hybrid_init_once` rather than
        by the plan (r5 m-7), and separately the pools initialized from
        `moe_hybrid_init_once`'s own scan (r6 I-3);
      - the call at `gs:19145` left in place; and separately, the definition
        of `unified_cache_grow_host_scratch_zone` left with no caller, which
        a callers-only check would pass (§M68 (a), r5 m-5);
      - the backend-member check moved into `~ggml_backend_sycl_context`
        (r3 I-3).
  - RED `test-sycl-seg-forward-return-source` (Python,
    `sycl;host;source-contract`; §M69 (b) as amended, r6 m-1, n-2). It
    scans `moe_graph_record_segments` and `moe_graph_replay_segments` in
    `ggml-sycl.cpp` and fails on any `ggml_sycl_compute_forward(` call
    whose result is not consumed by the fail-closed helper. At e2461d4fb it
    fails and names six sites: `gs:97234`, `:97240`, `:97285`, `:97315`,
    `:97551` and `:97595`. At d8a67422d the same six are `gs:97244`,
    `:97250`, `:97295`, `:97325`, `:97561` and `:97605` (r7 n-1). The gate
    names the line it finds, so the drift does not matter. It also fails
    unless the helper throws `ggml_sycl_fallback_error` on failure, and if
    the helper asserts, aborts or returns (§M69 (b) third amendment). It reports each by its function and its role
    (direct segment, `moe_after` after a direct segment, recording,
    `moe_after` after a recorded segment, replay direct, replay) as well
    as its line. After C4 it passes. Its mutants are the six discards, one
    at a time, each of which must fail naming that role. A positive
    control fixture holds one discarded call, and the gate must flag it.
    It also fails if the two functions hold other than six calls, so a
    seventh call routed around the helper cannot hide.
  - `test-sycl-host-zone-select` (`sycl;host`). It is a pure test of the
    extracted `select_zone`, over role x category x `kv_spill_to_host`, and
    of the lifetime checks over a fake registry.
    - It covers every row of §4.4's routing.
    - The graph-boundary check: a `CONTEXT`, a `MODEL`, a `DEVICE` and an
      `UNOWNED` record pass it, and a `GRAPH` record trips it.
    - Role `KV` from the `SYCL_KV_Host` intent selects zone 1, and the
      post-load `SYCL_Host` intent selects zone 3.
    - **The owner-boundary checks** (addendum 7 (b)):
      - a buffer-owned `CONTEXT` record still live after its buffer's
        `free_buffer` prints the `[HOST-ZONE] leaked CONTEXT record` WARN,
        naming the cohort, the bytes and the owner, and so does a
        backend-member record live after `delete sycl_ctx`. Under `GGML_SYCL_STRICT_LEASES=1` it aborts, in
        its own process, scored on rc 134 plus that text;
      - a `MODEL` record live at its model's free does the same;
      - a `DEVICE` record still live after `delete sycl_ctx` of the **last**
        live backend on its device does the same, and one still live after
        a non-last backend's free does not (§M62 (D)). **Mutant:** the
        check taken at every backend free, which fires while a second
        backend is live;
      - **the destruction order** (§M57 (b)): a fake `llama_context` that
        frees `buf_output`, then its backends, then the scheduler's compute
        buffers, then the memory's KV buffers, in the order `lch:429`,
        `:403`, `:398` and `:342` give, prints no WARN. **Mutant:** the
        buffer-owned check moved to backend-context destruction, which fires
        on the compute and KV records still live. This test cannot see the
        real destruction order; the `GGML_SYCL_STRICT_LEASES=1` lead arms of
        step 5 can;
      - a record owned by a **different** context or model does not fire,
        which is the multi-model case.
    - Mutant: the owner check keyed by lifetime alone, which fires on
      another context's live record.
  - `test-sycl-host-scratch-demand` (`sycl;host`, r3 A). A pure test of the
    demand ledger behind `ggml_backend_sycl_notify_host_buffer_sizes`, over a
    fake zone that records `grow_zone` calls:
    - a context's second notify replaces its first: (100, 10) then (300, 10)
      leaves a demand of 310, not 410, so the target is the plan's SCRATCH
      plus 310; with the fake capacity below that target the ensure call
      grows by the deficit, rounded to one chunk, and at or above it
      nothing grows;
    - two contexts sum, and one `grow_zone` call is made per notify whose
      target exceeds SCRATCH's capacity; the same notify repeated ten times,
      as `sched_reserve`'s sites repeat it, grows once; a context's demand
      dropped at its
      destructor does not shrink the zone, and a later context of the same
      size grows nothing;
    - the request is the deficit plus one `DEFAULT_ALIGNMENT` per buffer at
      the chunk cap, and a `DEVICE` holder's demand is on the same ledger;
    - a non-SYCL backend is refused with the WARN and records nothing;
    - over two fake caches, with the fake host buft's `device` = 0, a notify
      through a device-1 backend grows device 0's SCRATCH and not device
      1's; with the buft's `device` set to 1, the same notify grows device
      1's (r4 I-D);
    - a grow that fails returns false (r3 m5).
    - **Mutants:** accumulate instead of replace (the r3 A.2 double
      growth); one grow per SYCL backend (the r3 A.5 N-times growth);
      today's additive `grow_scratch_zone` (the r3 I-1 (b) ratchet); the
      device-0 resolve (r3 I-1 (d)); the notifying backend's `ctx->device`,
      i.e. `model.devices[0]`'s, in place of `buft->device` (r4 I-D).
  - RED `test-sycl-plan-walk-sched` (`sycl;host`; r6 I-1, r7 m-4 (b)).
    Its vehicle is the **real** ggml-base scheduler, not a fake: a
    `ggml_backend_sched` over two CPU backend instances. The first, at
    index 0 (the higher priority), stands in for the SYCL backend the walk
    attributes to; the second is the scheduler's required last CPU
    backend. It needs no GPU.
    - **The injected target predicate.** The attribution function takes
      `bool (*is_target)(ggml_backend_t)`. Production passes
      `ggml_backend_is_sycl`; the test passes a predicate that matches the
      stand-in by pointer. `ggml_backend_is_sycl` would reject a CPU
      stand-in and attribute 0 at every point.
    - **The nodes reach the stand-in through the split's first pass, never
      a pin.** The graph's four nodes are `SET_ROWS` writing views of a
      tensor pre-allocated in a CPU buffer, so the first pass assigns each
      by its `view_src` buffer (`gb:1639-1651`) to the first backend that
      supports that buffer, the stand-in at index 0. The fixture never
      calls `ggml_backend_sched_set_tensor_backend`: `reserve_size`'s start
      reset erases pins (`gb:2682`), so a pinned fixture would attribute 0
      at C4's placement too, and the RED could never go green.
    - It calls the attribution function at three points:
    - after `ggml_backend_sched_reserve_size`: it attributes all four
      nodes, since the split is live (`gb:2677-2689`);
    - after `ggml_backend_sched_reserve`: it attributes **zero**, since
      that call resets at its end (`gb:2703`), and the zero-attribution
      WARN prints;
    - after a second `reserve_size` on the same scheduler: four again.

    **Its RED is the rev 3g placement.** Written first, the test asserts
    four attributed nodes at the point rev 3g called the walk, after
    `ggml_backend_sched_reserve`, and fails with `attributed 0 of 4`, the
    predicted message. Moved to after `reserve_size`, C4's placement, the
    same assertion passes; the post-`sched_reserve` point is then kept as
    the zero-attribution check above. The source gate's placement mutant
    (above) ties llama's call site to the tested point. The fake
    scheduler of the test below could not see this case, which is why the
    vehicle is the real one.
  - `test-sycl-attn-host-staging-plan` (`sycl;host`; §M62 (E), §M69, r4
    m-4, r5 I-1, I-2, m-6, m-7, r6 I-1, I-2, m-5). A pure test of the slot admission extracted
    from `ggml_sycl_attn_host_staging_get` and of the plan walk, over a fake
    allocator that records each allocation and a fake scheduler that assigns
    each node a backend and device:
    - with the host dispatch off, the plan registers no staging demand,
      grows no slot and prints the `staging not planned` line once (§M69
      (a)); the pools' demand is still registered, for a host-resident
      `MUL_MAT_ID` `src0`;
    - a walk alone changes nothing: the slots grow, and demand is
      registered, only at the commit (r6 I-1 (2), r7 I-1 (3));
    - **one reserve's walks merge** (r7 I-1 (4)): the fixture is a top-10
      MoE graph with `n_seq_max` 1 (r8 n-2, n-3); a pp walk at `n_tokens`
      512, then a tg walk at `n_seqs` = 1, then a commit leaves the pp
      figures for Q, DST, MASK and the pools' D. **Mutant** "tg replaces pp": the RED
      fails with `committed q=<1-token figure> expected <512-token
      figure>`, and on the pools `committed D=10 expected D=5120`;
    - **the begin clears every backend's pending plan** (r7 I-1 (1)): a
      first reserve's walk attributes nodes to backends A and B and then
      fails before its commit; a second reserve's begin runs, its walk
      attributes only to A, and it commits. B then commits nothing.
      **Mutant** "stale pending from a failed candidate is promoted": no
      begin, so B commits the first reserve's plan, and the RED fails with
      `backend B committed a plan its reserve never walked`;
    - after a plan of (Q, DST, MASK, VAL, IDS), each slot's capacity is its
      figure rounded by the slot rule, and the `DEVICE` demand is their
      sum;
    - every request at or below its slot's capacity returns the slot and
      records no allocation, including a MASK request that grows with
      `n_kv` up to the planned `n_ctx`;
    - a request one byte above its capacity returns null and prints the
      `exceeds the planned` WARN naming the kind, and so does any request
      to a slot never planned;
    - a second commit from the **same** context replaces its first (r6
      m-5): a larger plan grows each slot once, and a smaller one
      re-allocates each slot down to it. With a second live context
      holding the larger plan, the smaller commit leaves the slots at the
      larger; freeing that context re-allocates them down. The capacity is
      never the maximum over every plan ever walked, so an auto-`n_ubatch`
      trial of candidates 512, 1024, 2048 that settles on 512 ends at the
      512 plan. **Mutant:** the high-water over every walked plan (rev 3g),
      which ends at the 2048 plan, an idle reservation (P4).
    - the dispatch counter (r6 I-2): a host `FLASH_ATTN_EXT` that is
      refused still counts in `fa`, and in `refusals`; the first
      interception prints the `first host dispatch` line once; the
      fail-closed helper prints the counter line, then the unsupported-node
      line, and throws `ggml_sycl_fallback_error`, which the test catches,
      with the process still alive and no abort (§M69 (b) third
      amendment). **Mutants:** `fa` incremented after
      `ggml_sycl_attn_host_staging_get` succeeds, which reads `fa=0` on a
      refused dispatch; the helper asserting, which kills the test
      process.
    - The walk selects by the dispatch's predicate (r5 m-6). A synthetic
      reserve graph holds four `FLASH_ATTN_EXT` nodes: one whose K is a
      **view** (buffer `NULL`, `view_src` in a `SYCL_KV_Host` buffer), which
      counts; one whose K is in a CPU buffer and one whose K is in a
      `SYCL_Host` buffer (`-nkvo`), which do not; and one on a device
      buffer, which does not.
    - Per dispatching device (§M69 (d)): with the nodes of layers 0-1
      assigned to a device-0 backend and layers 2-3 to a device-1 backend,
      each device's slots get only its own nodes' figures, and a dispatch on
      device 1 at or below them records no allocation.
    - **The pools are sized from the plan, and the over-plan check can
      fire** (r6 I-3). A synthetic MoE graph with a `MUL_MAT_ID` `src0`
      whose experts the fixture placement plan puts on the host, top-2, is
      planned and committed at `n_tokens` 8 (D = 16). `moe_hybrid_init_once` is then given the
      same graph at `n_tokens` 16 (D = 32): it initializes both pools at D
      = 16, the plan's figure, and prints the `exceeds its planned` line
      naming `graph needs D=32` and `planned D=16`. The same graph at
      `n_tokens` 4 prints nothing. With no commit, the `unplanned; not
      initialized` line prints. The test needs no GPU: the pools take host
      allocations through the fixture cache.
    - **The gate is placement, not the buffer** (r7 I-2 (2)). The same
      `src0` in a `SYCL_Mapped`-typed buffer, which is `is_host`, with
      every expert placed on the device by the fixture plan, registers
      **zero** pool demand, and the commit prints `committed D=0 ...
      host_expert_nodes=0`. **Mutant:** the `ggml_backend_buffer_is_host`
      gate, which fails with `pool demand 0.3 MiB for an all-device src0`.
    - **Two contexts** (r7 m-5). Context A commits at `n_tokens` 8 (D =
      16), then context B at 16 (D = 32): the pools re-size once to D = 32
      and print `resized D=16->32`. B's backend context is destroyed and A
      re-commits: the pools re-size to D = 16. **Mutant:** init-once kept,
      which leaves the pools at D = 16 after B's commit and fails with
      `pools hold D=16 below committed high-water D=32`.
    - **Mutants:** the init sized from its own graph (rev 3g), which
      initializes at D = 32 and prints no line; the check removed, which
      prints nothing on the D = 32 graph.
    - **Mutants:** the dispatch-time grow restored, which records an
      allocation on the over-plan request instead of refusing it; the plan
      ignoring the switch; `ggml_backend_buffer_is_host` as the predicate,
      which counts the CPU and `SYCL_Host` K nodes and misses the view
      (`NULL` buffer); the helper's device for every slot, which leaves
      device 1's slots unplanned.
  - RED `test-sycl-scratch-trial-return` (`sycl;cache;mem-handle`,
    `level_zero:1`, lead-run; §M75; r14 M-8 (c); r15 M-4, M-5; §M88 (2)). Its
    pinned chunks come from `unified_allocate_owner_backing` on a real
    queue (`pp:84-113`), so it touches a device and is registered,
    labelled and pinned like the device REDs, and form 2 excludes it. The
    registration sets `ONEAPI_DEVICE_SELECTOR=level_zero:1` through the
    ctest `ENVIRONMENT` property (one pool on one queue), so it does not
    see the iGPU. **Build and link** (r15 M-5): a private-fixtures test,
    like `test-unified-cache-bugs` (`GGML_SYCL_PRIVATE_TESTING=1` and
    `target_link_libraries(... PRIVATE ggml-sycl-private-fixtures)`,
    `ggml/src/ggml-sycl/CMakeLists.txt:2242-2244` at e2461d4fb), and
    never linked with libllama (§M82: the private fixtures embed their own
    backend objects and must not be mixed with an ordinary libggml). It
    drives the pool, the device cache and the trial entry points
    directly, with no model load, so §M82 case (2) applies, not the
    compile-time mutation case (3): the pinned pool's release path, the
    guard it scores, is reachable at backend level. There is no
    new link recipe and no second backend build. The fixture is the
    fixture cache's pinned pool with its chunk size set to 64
    MiB (`GGML_SYCL_PINNED_CHUNK_MB=64`, read by `resolve_chunk_size`,
    `pinned-pool.cpp:49-63`), SCRATCH configured at 64 MiB, and a plan
    term `host_zone_scratch_bytes` of 0, so the ensure target is the sum
    of the live demands. It is driven through the notify and the trial
    entry points as a **non-MoE** ladder would drive them:
    - candidate 512 needs 40 MiB of host compute: it fits the base, and no
      chunk grows;
    - candidate 1024 needs 150 MiB: its notify's deficit is 150 - 64 = 86
      MiB, which grows one chunk (`chunk_footprint_for` gives it
      max(64, 86) = 86 MiB usable). The fixture then allocates an 80 MiB
      stand-in for its host compute buffer, which fits only the grown
      chunk (r8 m-2). The candidate loses at the host-fallback check, so
      `RETURN` runs while the stand-in is live, and the fixture frees the
      stand-in at the simulated scheduler destruction, before the begin;
    - right after `RETURN`, `zone_capacity` reads **64.0 MiB**, the
      withdrawal's witness (r9 m-2), and `chunk_count` is still the
      pre-ladder count plus 1;
    - the settle re-reserves 512: after its begin, SCRATCH capacity is
      **64.0 MiB**, the winner's need chunk-rounded, the `released` line
      prints the returned chunk, and the pool's `chunk_count` and
      `allocated()` are back to their pre-ladder values;
    - **the second-context case, sized per case** (r9 n-7): a second live
      context's 100 MiB demand is registered first (deficit 36 MiB, one
      64 MiB chunk: capacity 128 MiB), and the fixture holds it as one
      live filler per chunk, leaving every existing chunk's largest free
      block below 58 MiB. Candidate 512 then grows one 64 MiB chunk (target
      140 MiB, deficit 12) and wins; candidate 1024 grows one 64 MiB chunk
      (target 250 MiB, deficit 58), holds a **58 MiB** stand-in, and
      loses. Before 1024's notify the fixture also fills candidate 512's
      kept chunk, which is empty once 1024's scheduler replaces 512's and
      which `zone_alloc` would visit first (`pp:510-521`), so the stand-in
      fits only the new chunk (r10 n-5). The ladder ends at 192 MiB, the capacity
      that 100 MiB plus the winner's 40 MiB needs, chunk-rounded, with
      `chunk_count` at the pre-ladder count plus 1, and not at 256 MiB. The
      K filler is freed at 1024's simulated scheduler destruction, with the
      stand-in and before the settle's begin (r11 n-2); 512's settle
      buffer, if the fixture allocates it, needs 40 MiB and fits K, which
      is then empty, so no figure read from capacity depends on it;
    - a fixture allocation left live in the loser's chunk makes the return
      refuse by name and keeps the chunk; nothing is released under it;
    - **the cache-path window** (r8 m-1, r9 m-2): a cached candidate 1024
      loses and `RETURN`s its 86 MiB chunk, with the 80 MiB stand-in still
      live. The fixture then fills the base with a 64 MiB filler, so no
      counted chunk has room. A ladder candidate 512 runs its probe and
      publish, which make a 1 MiB SCRATCH request through the ensure and
      `zone_alloc`, as the publish does; with the marked chunk withdrawn
      the ensure grows one new 64 MiB chunk and the 1 MiB lands there.
      The stand-in and the filler are freed at the simulated scheduler
      destruction, candidate 512 `OPEN`s and wins, and only then does its
      begin run. After the ladder: the `released` line names the marked
      chunk, `chunk_count` is the pre-ladder count plus 1 (the new chunk),
      `allocated()` is the pre-ladder figure plus that chunk's backing
      size, and capacity is 128.0 MiB;
    - **the empty return** (r10 m-3): outside any trial, `sched_reserve`'s
      own transaction grows one chunk, and the reserve throws before any
      host buffer lands in it, as when the device buffer fails first.
      `RETURN` releases it at once: right after the `RETURN`, with no
      begin, `chunk_count` and `allocated()` are back to their
      pre-reserve values and the `released` line has printed;
    - **the runtime throw with the context kept** (r10 m-3): the same
      throw with a stand-in live in the chunk. The context is kept and no
      reserve follows. A second context's begin on the device then leaves
      the chunk and prints no `SCRATCH return refused` line; the chunk is
      still withdrawn, and capacity still excludes it;
    - **configure over a withdrawn chunk** (r11 m-6; r13 M-4, M-5): these
      checks run in the plain child of `test-sycl-configure-refusal`
      (below), not in this test, because they need the strict variable
      absent from the process environment and the accessor latches it.
      In the same state as the runtime throw, `configure_host_zones` with
      new sizes refuses by name and returns false; capacity,
      `chunk_count`, the chunk's `withdrawn` flag, the marked set and
      `host_zone_reserved_bytes_` read as before the call. After the
      stand-in's free and a begin, the same call, with sizes that fit the
      capacity left once the chunk is released, succeeds. **configure's own
      release**: in a second run, a marked chunk that is already empty when
      `configure_host_zones` runs is released by it, the `released` line
      prints and the marked set is empty after. A **shortfall control**
      runs after that success call has configured the zones (otherwise it
      is a fixture error, since the scan would return false first): sizes
      above the capacity the pool can grow to return true as today, with
      zones unconfigured and the reserved figures zeroed (r12 M-6);
    - **the retry's truncation** (r10 m-4): `MARK`, one chunk grown (A)
      with a stand-in live in it, `RETURN_TO_MARK`, then one more chunk
      grown (B). Right after `RETURN_TO_MARK` the transaction holds 0
      chunks and A has one marked-set entry (r11 m-4). Two endings:
      - `RETURN`: A and B are each withdrawn once; after the stand-in is
        freed and the next begin runs, capacity, `chunk_count` and
        `allocated()` are back to their pre-`MARK` values;
      - `KEEP`: B becomes ordinary capacity, A stays withdrawn, and the
        next begin, after the stand-in is freed, releases A: `chunk_count`
        is the pre-`MARK` count plus 1.
      The same two endings run once more inside a trial transaction, with
      `try_candidate`'s `RETURN` or `KEEP`.

    Every expected figure is computed as a sum of `chunk_footprint_for`
    terms, never as a difference, and every case also asserts that
    `zone_capacity` does not exceed the pool's `allocated()`, so a second
    decrement cannot hide behind a wrapped `size_t` (r9 m-3). Every case
    that checks capacity also checks `chunk_count` or `allocated()`,
    because a chunk that is withdrawn and never released is invisible to
    capacity (r9 m-2).
    - **RED: the keep-the-high-water mutant**, `RETURN` as a no-op. It
      fails with `SCRATCH capacity <C> MiB after the ladder, expected
      64.0 MiB (winner's need)`, where `<C>` is 64 plus the one chunk the
      loser grew. The ensure is a high-water, not additive, so that chunk
      is `chunk_footprint_for` of the 86 MiB deficit plus the alignment
      terms, and `<C>` is **about 150 MiB** (64 + 86), not 214 (r8 m-2).
      The test computes `<C>` from `chunk_footprint_for` rather than
      hard-coding it, and asserts that the mutant's message quotes it. In
      the second-context case the same mutant reads 256 MiB against 192.
    - A second mutant tries to release every returned chunk at `RETURN`,
      empty or not. The 80 MiB stand-in is live at `RETURN`, so the
      release refuses by name and keeps the chunk, and the test fails on
      the `SCRATCH return refused` line (r8 m-2).
    - Two mutants for the window (r9 m-2):
      - **no withdrawal**: it fails **first** on the right-after-`RETURN`
        check, with `SCRATCH capacity <C> MiB after RETURN, expected 64.0
        MiB`, and the test asserts that the mutant's message quotes that
        text, `<C>` computed as above (r10 n-4). For the record, run on,
        the counted capacity is `<C>` with 6 MiB free in the marked chunk,
        so the ensure grows nothing, the 1 MiB lands in the marked chunk,
        the only one with room, and the begin refuses with `SCRATCH
        return refused dev=0 chunk=<idx>`;
      - **marks kept per transaction**, which candidate 512's `OPEN`
        resets: the chunk stays withdrawn but unmarked, so no begin
        releases it and capacity still reads 128.0 MiB. The test fails on
        `chunk_count <n+2> after the ladder, expected <n+1>; no SCRATCH
        released line for chunk <idx>`.
    - **Double decrement** (r9 m-3): `release_zone_chunk` also lowers
      `zones_[zi].size`. The main case fails with `SCRATCH capacity
      <wrapped> MiB exceeds the pool's allocated() <x> MiB`.
    - **Release only at a begin** (r10 m-3): the empty-return case fails
      with `chunk_count <n+1> right after RETURN, expected <n>: empty
      returned chunk <idx> not released`.
    - **The footprint leaves with the memory** (r14 M-3; r15 I-1, M-5): a
      test-only seam in the carrier's destructor, compiled only under
      `GGML_SYCL_PRIVATE_TESTING` (the same guard as the hold at
      `uc:16415-16421`, so it is absent from the shipped libggml-sycl),
      samples `allocated()` **after `free_chunk_owner` returns and before
      the registry query and the last section**, the instant of "freed but
      still counted". The
      case reads it during an empty return's release: it still includes
      the chunk's backing size, and after the release completes it is back
      to the pre-reserve value. Mutant: the backing size subtracted in
      `release_zone_chunk`, before the free; it fails with `allocated()
      <x> MiB at the free point, expected <x+b> MiB: footprint decremented
      before the free`.
    - **A dropped carrier** (r15 I-1): the fixture calls
      `release_zone_chunk` on an empty withdrawn chunk and lets the
      `released_chunk` leave scope without `release_chunk_backing`, as an
      early return or a failed append in a caller would. The seam fires
      once, `allocated()` is back to the pre-reserve value, and nothing
      aborts. Mutant: a destructor that does nothing, which leaves the
      footprint over-counted; it fails with `allocated() <x> MiB after a
      dropped released_chunk, expected <x-b> MiB: footprint leaked by a
      dropped carrier`. A second mutant has the moved-from carrier's
      destructor release again; a chunk consumed by
      `release_chunk_backing` then fails with `allocated() <y> MiB after
      release_chunk_backing, expected <x-b> MiB: footprint subtracted
      twice`.
    - **A registry-refused free is stranded, then reconciled** (§M88 (1);
      §M92; r16 M-5, M-6). The arm's entry check exits 77 if
      `GGML_SYCL_STRICT_LEASES` is set: the registry's own strict
      classification could abort the held row, which would read as a mutant
      failure. The fixture holds the chunk's registry row through two new
      `GGML_SYCL_PRIVATE_TESTING` hooks:
      `allocation_registry_test_hold_physical_release(metadata)`, which
      sets the row's `test_no_physical_release` and takes one exact test
      lease (`uc:16420`, `:16518`, `:16555`), and its inverse
      `allocation_registry_test_unhold_physical_release(metadata)`, which
      drops that lease (`uc:16559-16565`) and clears the flag. With the hold
      the release is refused, the row returns to LIVE (`uc:16425-16438`) and
      the registry schedules a retry (`uc:983-993`). Then:
      - right after the empty return's release, `allocated()` still
        includes the chunk's backing size, `stranded_` holds one entry (a
        test accessor), and the line `release not yet complete after the owner
        reset (registry row LIVE)` has printed exactly once; nothing
        aborts;
      - a second `allocated()` call, still held, reads the same (the
        reconcile finds the row and prunes nothing) and prints no further
        line;
      - the fixture unholds and calls the owner's release coordinator's
        `process_retries()` (`uc:1031-1056`), which frees the row; its
        `retry_count()` reads 0 (`uc:953-957`), so nothing stays on the
        retry list;
      - the next `allocated()` reads the pre-release value less the chunk's
        backing size, and `stranded_` is empty.

      Mutants, each with one fault: no prune, which fails with `allocated()
      <x> MiB after the retry completed, expected <x-b> MiB: stranded
      footprint never reconciled`; a prune without the query, which fails
      with `allocated() <x-b> MiB at the first reconcile while the
      allocation is still registered, expected <x> MiB`; and the subtraction
      made at the refusal, which fails with `allocated() <x-b> MiB after a
      refused free, expected <x> MiB: footprint left while the allocation
      is still registered`.
    - **Teardown reports a stranded entry** (§M92): a second run holds
      the row, releases the chunk, and destroys the pool with the hold
      still in place; the destructor prints `stranded chunk backing <b>
      MiB still registered at pool destruction`. The fixture then unholds
      and calls `process_retries()`. Mutant: the destructor without the
      report, which fails with `no teardown line for the stranded entry`.
    - **The accessor compares the key** (r16 M-3): the fixture publishes
      a row with `allocation_registry_test_publish` (`uc:16511-16518`);
      `unified_allocation_registered_exact` answers true for its metadata,
      false for the same `ptr` with a different id, and false for an absent
      row, and `allocation_registry_test_size()` is unchanged after the
      three calls. Mutant: a ptr-only compare, which fails with
      `registered_exact answered true for ptr <p> with id <j>, the row
      holds id <i>`. The published row answers LIVE. The fixture removes
      the row at the end with `allocation_registry_test_erase`
      (`uch:6073`), so the registry is left as it was found (r17 N-5).
    - **More than eight chunks strand** (r17 M-2): the fixture grows nine
      64 MiB chunks (576 MiB of host pinned memory), holds and releases
      every one, and asserts that all nine strand with no abort and that
      `stranded_.size()` equals nine, a literal and not a figure read back
      from the pool's own chunk count. It then unholds every
      row, calls `process_retries()`, and `allocated()` drops all nine.
      Mutant: `reserve(8)`, which stops at the named assertion
      `stranded_ capacity <c> exhausted at <n> entries` before the abort;
      the lead-run script scores that line from the log.
    - **The status order** (r18 N-6): the owning context's begin on a
      chunk that is both live and leased prints `SCRATCH return refused
      dev chunk reason=live` (a live allocation is checked before a lease;
      with `orphan_if_live` both end in ORPHANED, so only a non-orphan call
      makes the order observable). Mutant: the lease checked first, which
      fails with `reason=lease` for a chunk that holds a live allocation.
    - **A begin refuses any owner's chunk** (r10 m-3): the runtime-throw
      case fails on the second context's `SCRATCH return refused dev=0
      chunk=<idx>` line.
    - **`RETURN_TO_MARK` mutants** (r10 m-4, r11 m-4), each with one
      fault, none depending on a second fault the design's own rules
      forbid. The fixture reads the transaction's chunk count and
      each chunk's marked-set entry count (`marked_entries(dev,
      chunk_idx)`) through test accessors. The RED evaluates every
      ending and prints every failing check rather than stopping at the
      first, and the marked set stores (chunk, owner) entries with no
      dedupe at insertion, so a second filing is observable (r12 N-4):
      - (i) *chunks left in the set only* (the transition guard and the
        `KEEP` rule intact): right after `RETURN_TO_MARK` the transaction
        still holds A, and the test fails first with `transaction holds
        <n> chunk(s) after RETURN_TO_MARK, expected 0`. Run on, the
        `RETURN` ending files A in the marked set a second time and fails
        with `chunk <idx> filed <k> times in the marked set after RETURN,
        expected 1`;
      - (ii) *chunks left in the set and a transition-blind decrement*:
        fails the same first check. Run on, the `RETURN` ending withdraws
        A twice and fails with `SCRATCH capacity <wrapped> MiB exceeds the
        pool's allocated() <x> MiB`. It carries two faults by
        construction, and is named so;
      - (iii) *`KEEP` walks the device's marked set and clears
        `withdrawn` on every chunk in it* (r12 M-4), with the
        transaction's truncation intact: B is the transaction's only
        chunk, but the walk reaches A through the marked set. The `KEEP`
        ending un-withdraws A, which no begin's release can then free
        (`release_zone_chunk` requires the flag), and fails with
        `chunk_count <n+2> after the begin, expected <n+1>`.
    - **A release leaves no marked-set entry** (r11 m-5): after the empty
      return's release, the cache-path window's begin release and the
      retry endings' releases, the device's marked set is empty
      (`marked_entries(dev)` is 0), and a following begin prints nothing
      and moves no chunk. Mutant: `release_marked_chunk` not erasing the
      entry. It fails with `marked set holds <n> entry(ies) after the
      release of chunk <idx>, expected 0`.
    - **Configure over a withdrawn chunk** (r11 m-6) mutants:
      `configure_zones` rebuilds anyway, which fails with `configure_zones
      rebuilt over withdrawn chunk <idx>: capacity <c> after, <c0>
      before, marked set holds <n>`; and `configure_host_zones` calling
      `release_zone_chunk` directly, leaving the entry behind, which fails
      configure's own-release run with `marked set holds <n> entry(ies)
      after the release of chunk <idx>, expected 0` (r13 M-4).
    - **The refusal's two arms** (r13 M-5; r14 M-8; §M81 (e)). RED
      `test-sycl-configure-refusal` (`sycl;cache;mem-handle`, lead-run). Its
      children build a unified cache and a pinned pool, whose chunks come
      from `unified_allocate_owner_backing` on a real queue (`pp:84-113`),
      so they touch a device: the test carries the `cache;mem-handle`
      labels, which form 2 excludes, and its registration sets
      `ONEAPI_DEVICE_SELECTOR=level_zero:1` through the ctest
      `ENVIRONMENT` property (§M88 (2): one pool on one queue, like every
      other 5efe device RED). It is a private-fixtures test too (r16 M-6):
      it reads `marked_entries`, a `GGML_SYCL_PRIVATE_TESTING` accessor, so
      it is built and linked like `test-sycl-scratch-trial-return` above
      (`ggml-sycl-private-fixtures`, never libllama, §M82 case (2)). The
      parent is a driver that never touches a device. It spawns its own
      binary three times with `fork` and `execve`, giving each child its
      environment through the **`envp` argument**, built as a copy of the
      driver's own `environ` (which carries the
      registration's `ONEAPI_DEVICE_SELECTOR` and `LD_LIBRARY_PATH`) with
      `GGML_SYCL_STRICT_LEASES=1` set for the strict child and the
      variable erased for the plain child (r15 M-4). A fresh `envp` would
      drop the pin, and the child would enumerate the iGPU or find no
      device. It never calls `setenv` in a child,
      because `ggml_sycl_strict_enabled()` latches the environment in a
      function-local static on first call, and the fixture's cache
      construction reaches it first. No `ENVIRONMENT_MODIFICATION` is used
      (it needs CMake 3.22, and the minimum is 3.14).
      - **Strict child**: sets `RLIMIT_CORE` to 0 first (a core-dumping
        abort holds pinned memory past its exit message), builds the
        withdrawn-chunk state, calls `configure_host_zones`, and must
        abort. The driver scores it with `waitpid`: `WIFSIGNALED` and
        `WTERMSIG == SIGABRT`, plus the `configure_zones refused: chunk`
        ERROR line on the child's stderr before the abort.
      - **Plain child**: asserts `ggml_sycl_strict_enabled()` is false at
        entry, runs the configure cases above, frees its stand-in and
        finishes cleanly (the pool's destructor waits 5 s and asserts on
        outstanding leases, `pp:1086-1109`, so a child left with a live
        stand-in could abort and make the two mutants
        indistinguishable). The driver scores `WIFEXITED` and
        `WEXITSTATUS == 0`.
      - **Outstanding-carrier child** (r17 M-5): it sets `RLIMIT_CORE` to
        0, releases an empty withdrawn chunk into a `released_chunk` and
        keeps the carrier alive, then destroys the pool. It must abort:
        `WIFSIGNALED`, `WTERMSIG == SIGABRT`, plus the `released_chunk
        outstanding at pool destruction` line on its stderr. Its envp is
        the plain child's. Mutant: the assertion removed, or placed after
        the free loop, scored `outstanding-carrier child exited <s>,
        expected SIGABRT`.
      - **VOID exits 77**: a child whose entry check finds the strict
        variable set (plain) or unset (strict), finds
        `ONEAPI_DEVICE_SELECTOR` unset, or finds no device, exits 77, and
        the driver exits 77, so a VOID case never reads as a pass.
      - Mutants: the abort dropped (`strict child exited <s>, expected
        SIGABRT`), and the abort taken in non-strict mode (`plain child
        died by signal <n>, expected exit 0`).
  - RED `test-sycl-scratch-owner-end` (r9 m-5, r10 m-5; device,
    `sycl;cache;mem-handle`, `level_zero:1`, lead-run, loads no model). It
    creates a real backend with `ggml_backend_sycl_init`, drives its
    transaction through the `ggml_backend_sycl_host_scratch_trial`
    proc-table entry, and frees it with `ggml_backend_free`, which
    reaches `ggml_backend_sycl_free` and the owner-end step, never a
    fixture call. The stand-in is a SCRATCH allocation made through the
    same host-buffer path the scheduler's host compute buffer takes, freed
    with `ggml_backend_buffer_free` after the backend, in `llama_context`'s
    member order. Two cases, each starting with `sched_reserve`'s own
    transaction growing one chunk with the stand-in live, then `RETURN`
    on a throw, with no begin after it:
    - **orphaned**: `ggml_backend_free` with the stand-in live releases
      nothing and orphans the chunk; freeing the stand-in releases it, the
      `released` line prints, and `chunk_count` and `allocated()` are back
      to their pre-reserve values before the pool is destroyed;
    - **empty at free**: the stand-in is freed first, and then
      `ggml_backend_free` releases the chunk itself.

    **A third case, two empty chunks** (r15 M-1): an owner whose
    transaction grew two chunks, both empty when `ggml_backend_free`
    runs, releases both after the walk's lock drops. Both `released`
    lines print, and `chunk_count` and `allocated()` are back to their
    pre-reserve values. The walk calls the core under one hold of the lock
    and frees every collected backing after its scope.

    **A fourth case, a lease at the owner's end** (r17 M-4): the
    stand-in's host pointer is leased through `host_acquire_chunk_lease`
    (`uc:24100-24104`), then `RETURN`, then `ggml_backend_free`, and each
    order ends with the lease's drop through `host_release_chunk_lease`
    (`uc:24107-24111`) as the last event. (a) The stand-in is freed before
    the backend, so the chunk is empty with a lease held: the owner-end
    walk orphans it and releases nothing (`chunk_count` is still `<n+1>`),
    and the lease's drop releases it, so the `released` line prints at the
    drop, `chunk_count` and `allocated()` are back to their pre-reserve
    values and the marked set is empty. (b) The stand-in is live at the
    backend's free, so a live allocation and a lease coexist: the walk
    orphans it (live is checked first), the stand-in's free finds the
    lease held and releases nothing, and the lease's drop releases it.

    Both orphaned and empty-at-free cases also assert, right after the
    release, that the device's marked set is empty and that a following
    begin on the device prints
    nothing (r11 m-5): the orphaned chunk's emptying free reaches the same
    `release_marked_chunk` as every other release.

    Mutants, each with its message: the owner-end call removed from
    `ggml_backend_sycl_free`, which fails both cases; the orphaned-free
    branch removed, `chunk_count <n+1> after the stand-in's free, expected
    <n>`; and the empty-at-free branch removed, `chunk_count <n+1> after
    the context is freed, expected <n>: withdrawn chunk <idx> idle until
    pool destruction`; and the entry left in the set by the orphaned
    free's release, which fails the orphaned case with `marked set holds
    <n> entry(ies) after the release of chunk <idx>, expected 0`; and the
    owner-end walk freeing only its first collected backing, which fails
    the third case with `chunk_count <n+1> after the context is freed,
    expected <n>: owner-end freed one of two chunks`. A walk that called
    the public form inside its scope is the self-deadlock that gate
    clause 7 pins; and the lease's drop not handing the index back, which
    fails the fourth case with `chunk_count <n+1> after the lease dropped,
    expected <n>: withdrawn chunk <idx> waits for a begin`.
  - `test-sycl-kv-demand-idx` (`sycl;host`, r3 B). A pure test of
    `kv_demand_for_layer` on a 48-layer plan with Qwen's shape: 12 FULL
    layers of 512 MiB at 262144 cells, `kv_idx_width` 128. Each carries 576
    MiB, and a 0 width gives 512. Fed to `plan_runtime_kv_demotion` with 287.8
    MB of room, it demotes 12 layers, 3..47, with 6912.0 MB of host KV.
    Mutant: the demotion input reading `kv_size_for_layer`, which prints
    6144.0.
  - `test-sycl-getrows-refusal-terminal` (`sycl;host`, a fake `stream_dma`,
    r2 m-5). One refusal gives one call, one counted refusal, no
    `[GET_ROWS] DMA retry` line, and the CPU fallback.
  - `test-sycl-mapped-loader-source` (`sycl;host;source-contract`):
    - prefetch 0 at the SYCL `init_mappings` call;
    - every site of the §4.3 table, including the preload copy (m-7) and
      `alloc_buffer` (m-10);
    - no `use_mmap` read in `create_tensor`'s SYCL buft choice;
    - `lm:2693` narrowed to the no-tier case;
    - the eight `weight_host_bytes` accumulation sites test `host_backing`,
      and the readers of §4.4 read their assigned meaning (r2 m-3);
    - `dma_streamed` is 0 without `PINNED` host weights (r2 m-2);
    - the eight `resolve_or_host` callers are the pinned list (§5.3);
    - the four `stream_dma` callers route `refused_file_mapped` explicitly.
  - `test-sycl-host-backing-plan`, a Qwen-shaped inventory (1223 tensors,
    B70 budget):
    - with `load_maps_files` true: host experts are `FILE_MAPPED`,
      `weight_host_bytes` = 0, `host_mapped_bytes` = 102383.5 MB, the WEIGHT
      zone is 64 MiB, the DMA pool is 0, SCRATCH is 1732.0 MB with no
      RUNTIME-labelled or device-sizing term, the runtime pre-allocate is
      91.9 MB, and the base working set is 3579.2 MB: 2 + 1 chunks (r3 m3,
      addendum 11 (1), §M62 (A));
    - with it false: today's 122924.2 MB zone and 1700 MB DMA pool;
    - with mlock: the mapped bytes are charged.

    It is host-only through 7v5c's dry-run entry if 7v5c has landed.
    Otherwise it is registered `sycl;cache;mem-handle`, `level_zero:1`, and is
    lead-run.
- Device REDs (`sycl;cache;mem-handle`, `level_zero:1`, lead-run):
  - The two REDs registered with the C4 tests above,
    `test-sycl-scratch-trial-return` and `test-sycl-configure-refusal`,
    belong to this list and pin `level_zero:1` like the rest (§M88 (2);
    r15 M-4, N-1).
  - RED `test-sycl-seg-refusal-cleanup` (r8 I-1; §M69 (b) third
    amendment; r9 m-1). A small MoE graph on the segmented path, run
    through `ggml_backend_sycl_graph_compute`, with a refusal forced at a
    named node through a `GGML_SYCL_PRIVATE_TESTING` seam, which is valid
    in a test binary (§M74 (d)). The test captures the log through
    `ggml_log_set`. At the refusal, before the throw, the seam snapshots
    the sizes of the context's `graph_weight_leases` and
    `graph_moe_expert_leases` (`common.hpp:7222-7223`), and whether
    `exec_graph` is non-null and `active_exec_graph.valid` is set. A size
    read before the call would be the previous graph's, since the preload
    clears both vectors at its start (`gs:60512`, `:61056`). Two cases:
    the refusal at a direct segment's node, and at a recording node inside
    `begin_recording`.
    - **Preconditions; if one fails the case is a fixture error, not a
      pass:** the seam fired at the named node, and its snapshot shows at
      least one lease pinned by the refused call, in either vector. The
      recording case also requires the `[SYCL-SEG] Segment %zu recording
      failed at node %d` WARN (`gs:97343-97345`) naming the seam's node as
      its reach witness. Without it, a segment that direct-dispatches
      (fewer than `MIN_SEGMENT_NODES` nodes, or FA in the segment,
      `gs:97221-97223`) would turn the case into a second direct case, and
      its recording asserts would pass vacuously.
    - **Assertions, in this order; the test stops at the first failure and
      prints its message:**
      1. the `[SYCL] recoverable runtime fallback failed` line printed
         (`gs:107778`);
      2. the status is `GGML_STATUS_FAILED`, and the process is alive;
      3. `g_ggml_sycl_graph_recording` is false, `g_recording_graph_ptr`
         and `g_recording_queue_ptr` are null, the retained-handle sink is
         null, and the queue is not recording;
      4. both lease vectors are empty, `exec_graph` is null and
         `active_exec_graph` is invalid. The catch cleared the whole-graph
         state before its unpin (§4.4 staging plan), so the unpin ran: one
         failed decode leaks no lease (P2);
      5. a second `graph_compute` on the same context, with the seam off,
         succeeds.
    - A third case sets `active_exec_graph.valid` alone through the seam
      before the call, with `exec_graph` left null (r10 m-2). That state
      avoids both earlier clears, the key-miss clear (`gs:106993`, inside
      the branch that needs `exec_graph`, `gs:106944`) and the fa-pointer
      drift clear (`gs:107112-107117`), and the preload and pre-allocation
      skips that a non-null `exec_graph` causes (`gs:107026-107051`).
      **Preconditions** (r11 m-2): the seam's snapshot shows
      `active_exec_graph.valid` set at the refusal (if an earlier path
      cleared it, the case is VOID) and at least one lease pinned by the
      refused call, in either vector (without it, assertion 4 passes
      vacuously). The case asserts 4: the catch's cleanup step is what makes
      the unpin run. Mutant: that step removed. It fails with `lease
      vectors <w>+<m> non-empty after a FAILED return; active_exec_graph
      live`, where `<w>` and `<m>` are the seam's snapshot.
    - A fourth case refuses at the **moe_after site of a recorded
      segment** (`gs:97315`), whose executable was submitted just before
      (`gs:97311`) (§M69 (b) fourth amendment; r10 I-1; r11 m-2, m-3). For
      the segment being processed the seam reports its index, its
      `moe_after`, whether its `exec_graph` is non-null, whether
      `segment_submitted` was set, and its last recorded node
      (`recording_node_index`); for the whole call it reports the
      identities (pointers) of the non-null executables recorded so far.
      It samples `moe_segments.size()` at three points: **s1**, the first
      statement of `graph_compute`'s fallback catch; **s2**, the entry of
      C4's cleanup step, which in source order follows th32's cancel step,
      both queue waits, the scatter wait and the staging drain; and **s3**,
      the exit of that step.
      - **Preconditions; each failure is a fixture error, not a pass:**
        (a) the seam fired at the node equal to `moe_after` of that
        segment, and that segment's `exec_graph` is non-null and
        `segment_submitted` was set, so exactly that segment's executable
        is the one newly submitted at `gs:97311`; the seam compares its
        identity with the last pointer of the recorded list, and an
        earlier recorded segment alone does not satisfy it (a refusal at a
        direct segment's `moe_after`, `gs:97240`, is a different site and
        a different case); (b) the inner catch's `[SYCL-SEG] Segment %zu
        recording failed at node %d` WARN (`gs:97343-97345`) printed for
        that segment index, naming the segment's **last recorded node**,
        which is not the seam's moe node (a `moe_after` throw prints
        `recording_node_index`, not `seg.moe_after`); (c) the lease
        snapshot shows at least one lease pinned by the refused call, as
        in the first two cases.
      - **Assertions, in this order:** assertion 1, then the case's entry
        assertion **E**, then 2-5. E precedes 2-5 because, on the mutant,
        the executables are destroyed in the unwind with their submission
        outstanding (`gs:99522-99538`), and assertion 5's second
        `graph_compute` on the same context can crash before the test
        prints the predicted message. **E:** at s1, `moe_segments` holds
        every executable the throw had recorded: none was destroyed in the
        unwind. The rest of E's group runs before 2-5 as well (r12 M-3
        (a)): s2 equals s1 (nothing was destroyed before the cleanup
        step, so none before th32's cancel step or either wait), then s3
        is 0.
      - **The seam keeps the mutants measurable** (r12 M-3 (b), (c); r13
        N-3). Both mutants destroy an executable whose `gs:97311`
        submission is outstanding (`gs:99522-99538`), which can abort
        inside `graph_compute` before any message exists. So the seam,
        which is test-only and runs at the refusal site before the throw,
        first waits the stream. **This wait is the fourth case's seam
        only**: its refusal is at a `moe_after`, outside any capture
        (`gs:97315`). In cases 1-3 the refusal is a recording node inside
        `begin_recording`, where the queue is recording and a wait on it is
        an error, so those seams take none; the destruction then cannot crash, and the test
        measures where it happens, which is the property under test. That
        wait is a test-seam action and is not in the production catch,
        where a wait at the throw site is still forbidden. The seam also
        prints each sample (`s1=<n>`, `s2=<n>`, `s3=<n>`) as it is taken,
        before any later step, so the line a mutant's message quotes
        exists even if a later step dies.
      - **Mutants, each failing alone with its derived message; the test
        asserts that the message quotes it:**
        - the local unwind restored (no move): fails E with `executables
          destroyed in the unwind: moe_segments 0 at catch entry, <n>
          recorded at the throw`;
        - the destruction placed before the waits
          (`invalidate_moe_segments` called at the top of the catch, after
          s1): fails the s2 check with `executables destroyed before the
          waits: moe_segments <n> at catch entry, 0 at the cleanup step`.
          s2 is sampled after th32's cancel step and both waits, so any
          destruction before it is seen, and th32's step needs no sample
          point of its own.
    - A fifth case, **stale vectors** (r11 m-1), pins the entry guard. A
      first `graph_compute` records K >= 1 executables. **Reaching the
      record function** (r12 M-2): the second call must miss
      `segments_match` (`gs:107184-107186`: valid, same `n_nodes`, same
      phase), or it replays and never invalidates. So it runs a graph with
      a different `n_nodes`, and **no whole-graph record has set
      `active_exec_graph.valid`**, so that the phase-change clears
      (`gs:105417`, `:105558`) do not invalidate first and set the sticky
      flag before `gs:107202`. **The optional-layout clear has its own
      precondition** (r14 M-7): `gs:106124` gates it on the context's
      `optional_layout_epoch` differing from the global
      `g_ggml_sycl_optional_layout_epoch`, and only then (`gs:106125`) on
      `exec_graph`, `moe_segments_valid` or `moe_block_graphs`, a second
      condition that holds on this fixture after the first record. So the
      two epochs must be equal at the second call, with no optional-layout
      retirement between the two calls, and the fixture-error check reads
      both. **The retire seam**
      makes `moe_graph_retention_retire_exact` fail on **every** call until
      the test clears it, not once: the caller's own retry at `gs:107214`
      is a second retire, and a one-shot seam would let it succeed and
      clear the K executables on the correct implementation. With the seam
      failing, the call's `invalidate_moe_segments` (`gs:107202`) refuses,
      keeps the vectors and sets `moe_graphs_disabled`, and the call
      reaches the record function. The test clears the seam before the
      context's destruction. **Preconditions; each failure is a fixture
      error:** the second call missed `segments_match`, the seam failed at
      `gs:107202` and again at `gs:107214`, and the seam's snapshot at the
      record function's entry shows K >= 1 segments, the disabled flag
      set, and the K executables' identities.
      **Assertions:** the `[SYCL-SEG] record refused` WARN printed once and
      quoting K; the call returned through the caller's fallback, with no
      throw; and `moe_segments` still holds the same K identities after
      it, so no old executable was destroyed while the retirement was
      refused. Mutant: the guard removed. The record function then records
      over the kept vectors and the success path's move destroys the K old
      executables; it fails with `no "record refused" line while the
      context holds <K> segment(s)`.

    **RED: the immediate-return mutant** (rev 3j's exit). It never reaches
    `gs:107778`, so both cases fail **first** on assertion 1, with `no
    "recoverable runtime fallback failed" line after a FAILED return`, and
    the test asserts that the mutant's message quotes that text (r9 m-1
    (a)). For the record, with every assertion evaluated the mutant would
    also fail 3 in the recording case (`recording flag still set`) and 4
    in both (`lease vectors <w>+<m> non-empty`); 2 passes, since rev 3j
    returned FAILED; 5 is not predicted, because the recording case may
    crash on the destroyed local graph.
  - `test-sycl-file-mapped-stream-refusal`: `stream_dma` over a `FILE_MAPPED`
    view refuses, with and without a `copy_fn`, and the counter counts.
    Mutant: the check placed after the `copy_fn` test.
  - `test-sycl-file-mapped-bounce`: `mem_copy` of a mapped source takes the
    bounce (`host_ptr_is_usm_accessible` false). The outside-materialization
    counter stays 0 inside a materialization, and counts outside one at each
    named chokepoint (m-17).
  - `test-sycl-file-mapped-expert-owner`: `register_host_expert` over a mapped
    slice stores the owner, and the mapping outlives the buffer while an entry
    holds it.
  - `test-sycl-file-mapped-resolve-abort`, a death test in its own process:
    `resolve_or_host` on a mapped weight aborts naming the tensor. It is
    scored on rc 134 plus the tensor name.
  - **`test-sycl-pinned-meminfo-control`** (r2 m-13; the lead runs it before
    step 2 below):
    - It allocates 4 GiB through `unified_allocate` with `must_host_pinned`,
      touches every page, samples `/proc/meminfo` before and after, and frees
      it.
    - It prints each counter's delta: `AnonPages`, `Shmem`, `Mlocked`,
      `Unevictable`, `Cached`.
    - Pre-registered: at least one counter moves by ≥ 3.5 GiB, and that
      counter is the one pinned host USM lands in. run2's `Shmem` stayed at
      1.8 G while 140 GB was pinned, so it is not `Shmem`.
    - If no counter moves, the test exits 77, and the host-growth criterion
      below rests on the `Committed` line plus `MemAvailable` alone.
- Lead, in this order (r1 I-8, m-14, m-15):
  1. `POSIX_FADV_DONTNEED` over shard 3, then `fincore` on it. Pre-registered:
     ≤ 1 MiB resident.
  2. **Qwen3.8 Q8_0 on `level_zero:0`, default load flags, `-c 4096`, the
     baseline completion**: `env GGML_SYCL_HOST_ZONE_ALLOC_TRACE=1
     ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion -m
     <Qwen3.8 Q8_0> -no-cnv -v -c 4096 -p 'The capital of France is' -n 32
     --seed 42 --temp 0 >step2.out 2>step2.err`, the run2 vehicle (r3 m2).
     The trace variable is part of the command, not the environment the
     lead happens to have (§M66; r5 I-3). `-v` is required: the zone, pre-size and demand
     criteria below read INFO lines, which `llama-completion` drops at its
     default threshold (§7 preamble).

     **Precondition** (§3.1, the C1-to-T6 window; accepted in addendum 8 (b)). th32
     T6 and T7 are on the tree, so the PLE table is taken by T6 and gathered
     by T7. Without T6, a taker load reads the table eagerly and the early
     plan never counted it. With T6 but not T7, nothing gathers it. Either
     tree makes steps 2 and 3 VOID, not FAIL, and the lead is told which
     tree it was.

     Pre-registered pass criteria:
     - **the answer** (r5 m-9, r6 m-3, r7 m-7, n-6): stdout and stderr go
       to separate files. `llama-completion` echoes the prompt, so the
       continuation's first word is anchored on the first line. The
       scorers, each on one line:

       ```bash
       head -n 1 step2.out | grep -cE '^The capital of France is Paris([^[:alnum:]]|$)'
       tr -cs '[:alnum:]' '\n' < step2.out | awk '{r=($0==p)?r+1:1; if(r>m)m=r; p=$0} END{print m+0}'
       tr -cs '[:alnum:]' '\n' < step2.out | awk 'NR>2{c[a" "b" "$0]++} {a=b; b=$0} END{m=0; for(k in c) if(c[k]>m) m=c[k]; print m}'
       ```

       The anchor must print 1 and the word-run check at most 3 (no word
       repeats four times in a row); those two are gated. The trigram
       check is **recorded, not gated** (r8 m-4): the lead reports its
       figure beside the answer. A correct continuation can repeat the
       prompt's own trigrams, as the enumeration control below does (4),
       so a trigram gate would fail a correct build. It still shows a
       phrase loop such as `... is Paris. The capital of France is Paris.
       ...` to the reader, which the word-level check passes.
     - **The scorers' controls**, run by the lead before scoring; each is
       fed the same way its scorer is:

       ```bash
       printf 'The capital of France is Paris.\n' | grep -cE '^The capital of France is Paris([^[:alnum:]]|$)'
       printf 'The capital of France is Lyon.\n' | grep -cE '^The capital of France is Paris([^[:alnum:]]|$)'
       printf 'The capital of France is\n' | grep -cE '^The capital of France is Paris([^[:alnum:]]|$)'
       printf 'Paris Paris Paris Paris\n' | tr -cs '[:alnum:]' '\n' | awk '{r=($0==p)?r+1:1; if(r>m)m=r; p=$0} END{print m+0}'
       printf 'The capital of France is Paris. It is\n' | tr -cs '[:alnum:]' '\n' | awk '{r=($0==p)?r+1:1; if(r>m)m=r; p=$0} END{print m+0}'
       printf 'The capital of France is Paris. The capital of France is Paris. The capital of France is Paris.\n' | tr -cs '[:alnum:]' '\n' | awk 'NR>2{c[a" "b" "$0]++} {a=b; b=$0} END{m=0; for(k in c) if(c[k]>m) m=c[k]; print m}'
       printf 'The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid.\n' | grep -cE '^The capital of France is Paris([^[:alnum:]]|$)'
       printf 'The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid.\n' | tr -cs '[:alnum:]' '\n' | awk '{r=($0==p)?r+1:1; if(r>m)m=r; p=$0} END{print m+0}'
       ```

       They must print 1, 0, 0, 4, 1, 3, 1 and 1. The last two are the
       enumeration a correct model plausibly gives (r8 m-4), which must
       pass both gated checks. The third line is the prompt echo alone, the negative control the GPT-OSS gate's `grep -cx`
       precedent asks for. Both repeat checks are `awk` over `tr`-split
       words, because this host's `grep` is `ugrep`, which rejects the
       back-reference a `grep -E` repeat pattern needs. Any other result means the scorer is broken
       and the criterion is VOID;
     - **the reference is C4's own continuation.** The parent commit cannot
       load Qwen (rc 137 in run2), so no parent run can serve. The lead
       records step 2's full stdout as the reference, and C5 and C6 rerun
       the same command and report any difference from it. It is recorded,
       not gated: two builds' device code can differ;
     - **the pools are what was planned** (§4.4 criteria; r6 I-3): the
       `holder=pinned_buffer_pool` line reads `bytes=100.0 MB` and the
       `holder=cpu_expert_pool` line `bytes=400.0 MB`, and the same run
       prints `[MOE-POOL] Pinned buffer pool: act=51200 KB, out=51200 KB,
       max_experts=5120` and `[CPU-EXPERT-POOL] Initialized: <n> threads, 4
       ring slots, 102400.0 KB/slot`. No `exceeds its planned` or
       `unplanned` pool WARN and no `[MoE-CPU] pinned expert pool too
       small` WARN;
     - `[HOST-ARENA] configured zones: WEIGHT=64.0 MB`;
     - `SCRATCH=1732.0 MB` (±0.2) in `configured zones`, `Pinned pool
       pre_allocate: grew 2 chunks for 3579.2 MB working set`,
       `pre_allocate_runtime: grew 1 chunks for 2048.0 MB (total need 91.9
       MB)`, and `Committed 6144.0 MB in 3 chunk(s)` (§4.4; r2 I-1, r3 m3,
       addendum 11 (1), §M62 (A)). They are scored on the printed MB
       figures, and §4.4's table names what did not land from them;
     - §4.4's WEIGHT pre-registration: every post-pre-size `zone=WEIGHT`
       line is `cohort=routing_indices_cache` within 20480 bytes (r4 m-2).
       **Trace-live check** (§M66; r5 I-3): `step2.err` holds at least one
       post-pre-size `[HOST-ZONE-ALLOC] ... zone=SCRATCH ...
       cohort=backend_host_buffer` line, from the context's `SYCL_Host`
       compute and output buffers that C4 routes to SCRATCH. At `-c 4096`
       nothing demotes, so no `zone=KV` line can serve here. Without that
       line the trace is dead, the WEIGHT criterion would pass on zero
       lines, and the arm is **VOID**, which counts as failed;
     - `[ATTN-HOST] staging not planned (host dispatch off)` once, and no
       staging demand (§4.4 criteria, §M69 (a));
     - `DMA staging pool: 0.0 MB`;
     - **no `Pinned pool grow_zone: zone 0` line after the pre-size**, through
       the end of the completion (r2 I-2). This absence counts only because
       the same run printed the `Pinned pool pre_allocate: grew` line above,
       its positive control at the same INFO level (r3 m2);
     - **SCRATCH is scored by chunks** (r3 A.3). Every `Pinned pool
       grow_zone: zone 3` line follows a `[HOST-ARENA] SCRATCH demand` line
       whose `grown` is positive, and reads `max(2048.0, align_up(deficit +
       n x alignment))` MB for that line's deficit. A zone-3 grow with no
       demand line before it is a miss, and FAILs naming the tenant's
       cohort. The demand lines print the compute and output figures; they
       are read from the run, not pre-registered, since run2 was killed
       before its scheduler reserve;
     - no `host compute buffer ... exceeds notified` WARN (r3 A.2);
     - **no `refusing SCRATCH host_zone_boundary_check` WARN** (r3 I-4,
       §M57 (c)). The `CpuExpertPool` ring and `PinnedBufferPool` are live
       from the first graph on Qwen's default path, so a build that leaves
       them `GRAPH` fails here;
     - **no zone-1 growth**: at `-c 4096` nothing is demoted, so `Pinned pool
       grow_zone: zone 1` is absent (with the same positive control);
     - the early and late `[PLACEMENT] Totals` both print charged host 0 and
       mapped 102383.5 MB. The early `Host staging zone` has `per_inflight`
       ≤ 1800 MB;
     - "planner host placement: yes" (m-3);
     - `SYCL_Mapped` model buffer lines sum to 127615 MiB ± 1%;
     - `SYCL_Host` model buffer lines hold 0 weight bytes;
     - peak growth of the counter the control named, over the pre-load
       reading, is < 30 GiB **plus the lazy table's own bytes**, read from
       th32's `[PLACEMENT] lazy per_layer_token_embd.weight` line: its
       `resident=` figure under `residency=HOST` (51880.0 MiB, locked, th32
       §10's expected line), or its `cache=` figure under `residency=STORAGE`.
       Those bytes are th32's, not 5efe's, and the term is taken from the
       line, never assumed. A missing line is a FAIL of the precondition.
       The criterion is skipped, and said to be skipped, if the control
       exited 77;
     - `MemAvailable` never falls under 25 GiB (the watchdog unit);
     - `Shmem` stays under 5 GiB;
     - `fincore` on shard 3 after the run is ≤ 256 MiB. The bound: the prompt
       plus 32 tokens is under 40 tokens, x 16 PLE rows per token (th32 §0
       table), x a 256 KiB bcachefs extent per row fault = 160 MiB, rounded
       up;
     - the `[FILE-MAPPED] counters:` line prints `stream_dma_refusals=0
       device_copies_outside_materialization=0` (m-18);
     - no GPU fault lines.

     There is deliberately no page-cache-growth criterion, because it would
     depend on the other shards' prior cache state.
  3. **The same `llama-completion` Qwen run with no `-c`** (the
     default-context arm), at `n_ctx = n_ctx_train = 262144`
     (`run2.log:131`): `env GGML_SYCL_HOST_ZONE_ALLOC_TRACE=1
     ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion -m
     <Qwen3.8 Q8_0> -no-cnv -v -p 'The capital of France is' -n 32 --seed 42
     --temp 0 >step3.out 2>step3.err`. The same precondition applies.
     - **The KV.** 12 of the 48 layers carry KV: qwen4exp marks layer `i`
       recurrent unless `(i + 1) % full_attention_interval == 0`
       (`src/models/qwen4exp.cpp:132-134`), and the interval is 4
       (`run2.log:37`). So layers 3, 7, ..., 47 carry it. Each holds
       262144 x (512 + 512) x 2 B = 512 MiB (`run2.log:146-147`). That is
       **6144 MiB** of main KV.
     - **The indexer** (r3 B). The `idx_` cache covers the same 12 layers
       (`lm:3580-3585`), K only, `indexer_head_size` 128: 262144 x 128 x 2 B
       = 64 MiB per layer, **768 MiB** in all. After C4 it is in the demand
       (§4.4), so each KV layer is 576 MiB, and **6912 MiB** in all. The
       recurrent state is device-resident and not in the figure (§4.4).
     - The SYCL planner's KV sizing counts all 48 layers (`run2.log:217`,
       `:307`). At 262144 that would claim 24576 MiB. That over-count is
       **llama.cpp-0edf**, outside 5efe.
     - **The room.** The demotion's capacity is `min(budget, shared zone)`
       (`kv_weight_capacity`, `kv-runtime-demotion.cpp:58`). In run2 the shared
       zone was 25519.0 MB (`run2.log:227`) and device weights 25231.2 MB
       (`:308`), with optional layouts 0.0. So the room is at most 287.8 MB,
       less than one 576 MiB layer, and no layer that carries KV stays on the
       device.
     - Demotion walks layers latest first, and skips layers with 0 recorded
       bytes (`kv-runtime-demotion.cpp:19-45`).
     - **Two pre-registered outcomes, and which one prints is 0edf's
       discriminator:**
       - real hybrid sizing: `12 layer(s) demoted, 0 SWA (6912.0 MB host KV,
         layers 3..47)`. **PASS.**
       - 0edf's over-claim: `48 layer(s) demoted, 0 SWA (27648.0 MB host KV,
         layers 0..47)`, 48 x 576. The indexer term follows the planner's
         KV-layer set, so it scales with the over-claim. It is scored **a
         5efe PASS plus a live 0edf defect** (§M61 (d)(3); r4 m-10): 5efe's
         routing did its job, while the plan prints 27648 MiB of host KV and
         the load pins 6912, a plan != reality (P4) that llama.cpp-0edf
         owns. Whether it blocks the merge is the lead's call.
       - Without C4's indexer term, the two lines would read 6144.0 and
         24576.0. Either figure is a FAIL of the r3 B change.

       The MB figures are MiB, the unit the existing demotion lines print (29
       Mistral layers at 32768 print 3712.0).
     - **General rule, if the printed `F` differs.** With `K = floor(F /
       576)`, real sizing gives `N = 12 - K`, layers `3 + 4K..47`, and
       `N x 576` MB; the over-claim gives `N = 48 - K`, layers `K..47`. Any
       other N, range or figure is a FAIL.
     - **The criteria:**
       - every criterion of step 2 that does not depend on `n_ctx` holds;
       - **the KV zone (zone 1) must grow, scored by chunks** (r3 B). The
         main KV reaches `SYCL_KV_Host` as three 2048 MiB buffers and the
         indexer as one 768 MiB buffer (§4.4, the chunk cap), and none fits
         the 64 MiB base. So the run prints **four** `Pinned pool grow_zone:
         zone 1 grown by 2048.0 MB (1 new chunks, ...)` lines at context
         init, 8192.0 MB in all, which covers the printed 6912.0. Fewer lines,
         any zone-3 growth that is not a demand line, or a `SYCL_KV_Host`
         buffer outside zone 1 is a FAIL. The arm sets
         `GGML_SYCL_HOST_ZONE_ALLOC_TRACE=1` (`uc:3990`) on its command line,
         whose per-allocation line names the zone, and is trace-live (§7
         preamble, §M66);
       - **the exact form** (§M57 (a)): the `[HOST-ZONE-ALLOC] ... zone=KV
         ... cohort=kv_host_buffer` trace lines (`uc:4002-4008`) sum to
         exactly the printed host-KV figure, 6912.0 MiB = 7247757312 bytes,
         and the four grow lines are its chunk-rounded form. No
         `cohort=kv_host_buffer` line names another zone;
       - no `[HOST-ZONE] leaked` line at context free;
       - the VOID guard, N = 262144 (§7 preamble);
       - no host zone grows during PP or TG, the `staging not planned` line
         prints once, and no staging demand is registered (§4.4 criteria,
         §M69 (a));
       - §4.4's WEIGHT pre-registration: every post-pre-size `zone=WEIGHT`
         trace line is `cohort=routing_indices_cache` within 20480 bytes
         (r4 m-2). **Its trace-live check is the exact form above** (§M66;
         r5 I-3): the `zone=KV cohort=kv_host_buffer` lines of the same
         `step3.err` summing to 7247757312 bytes. A dead trace prints none,
         so the WEIGHT criterion cannot pass vacuously: the arm is **VOID**,
         which counts as failed;
       - `MemAvailable` stays above 25 GiB. The load pins 8 GiB of zone-1
         chunks on top of step 2's 6 GiB, and `MemAvailable` before the
         load was 155.0 GiB (`run2.mem`);
       - a `runtime KV update rejected` line is a FAIL.

     **3b. The same Qwen no-`-c` run with `GGML_SYCL_ATTN_HOST_DISPATCH=1`**
     on its command line (§M69 (a)): step 3's command with
     `GGML_SYCL_ATTN_HOST_DISPATCH=1` added after `env`, writing
     `step3b.out` and `step3b.err`, and step 3's trace-live check. It is
     VOID until th32 T6/T7 and C4 are on the tree, like step 3.
     - **Its own vehicle** (§M60 (b); r6 m-4). The switch makes the SYCL
       backend accept `FLASH_ATTN_EXT` and `SET_ROWS` over host KV
       (`gs:108273-108330`), so the scheduler builds a different graph
       split from step 3's, with attention on the SYCL backend's host
       dispatch instead of the CPU split. No figure of step 3 that depends
       on the split is carried over; the ones that do not (the KV, the
       demotion line, the zones) are step 3's.
     - Every other criterion of step 3 holds, except that the staging
       criteria are the dispatch-on ones, under the precedence rules
       (§4.4 criteria; r6 I-2 (3)). The answer is scored as in step 2 and
       compared with step 2's reference (recorded, not gated).
     - **No parent control.** The parent commit cannot load Qwen (rc 137),
       so 3b has no parent-tree control, and a defect it finds is not
       classed as pre-existing on 3b's evidence alone. The host dispatch
       itself "cannot itself be GPU-verified" by its own account
       (`gs:81855-81862`), so a 3b defect is reported to the lead with
       5b's result beside it.
  4. The Mistral gate on `:1` and `:0` (default context), with `-v` added so
     the context line prints, and the VOID guard, N = 32768 (§7 preamble;
     r4 m-1). Mistral's `n_ubatch` ladder is not capped (it is not MoE),
     so this is the lead arm on which a trial candidate can lose (§M75).
     The command carries `GGML_SYCL_TUNING_CACHE=0`, so a cached terminal
     hit (`lc:1755-1760`) cannot skip the ladder and the run writes no
     tuning cache (r8 m-3). Criteria:
     - **scored always:** the last `SCRATCH demand ctx=` line's figure
       fits the final capacity with no chunk beyond it. The absence of a
       `SCRATCH return refused` line is informative only beside a
       counted `returned` line (next item), since no release is attempted
       without one; otherwise the run reports it as not exercised (r10
       n-6);
     - **the §M75 sub-criterion counts only with at least one `SCRATCH
       trial ... returned` line whose chunk count is at least 1**; the
       `reserve` and `retry` forms do not count (r10 n-7) (r9 m-4;
       a zero-chunk return prints nothing, §4.4, and the scorer checks the
       count anyway): every such line is followed by its `released` line.
       With none, the sub-criterion is **VOID**, neither PASS nor FAIL,
       and the run reports it, with the ladder's stop reason read from
       the `[SYCL-PLAN] auto n_ubatch=... (tried ...; <stop>)` WARN
       (`lc:1942-1943`). A probe or publish loss (`KV would be demoted`,
       `transaction refused`, `transaction busy`, `not the published
       model`, `lc:1530-1534`) happens before `OPEN`, and `ladder
       exhausted` has no loss, so none of them can print a `returned`
       line. **No lead arm is known to produce a growing loser:** on
       Mistral the likely post-`OPEN` losses, `compute buffers did not
       fit` (`lc:1620`) and `compute buffer fell back to host`
       (`lc:1628`), need not grow SCRATCH at all, so this sub-criterion
       may VOID on Mistral. The binding evidence for §M75 is the lead-run
       device RED `test-sycl-scratch-trial-return`.
  5. The GPT-OSS gate on `:1`, in two arms, each under `timeout 60`, **with
     `-lv 4` added** to the canonical command (r3 C; §7 preamble), stdout and
     stderr captured to separate files, and scored with `grep -cx '1, 2, 3,
     4, 5'` = 1 on the stdout file:
     - **the canonical CLAUDE.md `llama-cli` command with `-c 4096`**, under
       `GGML_SYCL_STRICT_LEASES=1` (§M57 (b)): rc 0, and no `[HOST-ZONE]
       leaked` line and no ownerless-lease WARN when the context is freed;
     - **the same with no `-c`**, at `n_ctx = 131072`, as a literal command:
       `timeout 60 env GGML_SYCL_HOST_ZONE_ALLOC_TRACE=1
       ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-cli -m
       /models/gpt-oss-20b-mxfp4.gguf -ngl 99 -lv 4 -st --simple-io
       --no-display-prompt --chat-template-kwargs
       '{"reasoning_effort":"medium"}' --reasoning-format none
       --reasoning-budget 0 -p 'Count from 1 to 5. Answer with only: 1, 2,
       3, 4, 5' -n 48 --seed 42 --temp 0 >step5.out 2>step5.err` (§M66; r5
       I-3).
       - **Trace-live check:** `step5.err` holds the `[HOST-ZONE-ALLOC] ...
         zone=KV ... cohort=kv_host_buffer` lines of the demoted host KV,
         and at least one post-pre-size `zone=SCRATCH
         cohort=backend_host_buffer` line. Without them the arm is **VOID**,
         which counts as failed.
       - **The demotion line is zhcn's GDC3 row, cited and not copied**
         (§M56 (a)). GDC3 is the single source of N, the host-KV figure and
         the layer range. The arm scores against the GDC3 point its tree
         selects (§7 preamble, §M61 (a)), on GDC3's own vehicle,
         `llama-cli`, and carries zhcn's compute criterion in its §M73 (b)
         form, cited, not restated (r7 m-6). If GDC3 changes, this arm changes with it and nothing here is
         edited.
       - The VOID guard, N = 131072 (§7 preamble).
       - No host zone grows during PP or TG, the `staging not planned` line
         prints once, and no staging demand is registered (§4.4 criteria,
         §M69 (a)).
       - With `GGML_SYCL_HOST_ZONE_ALLOC_TRACE=1` on the arm's command line
         and the run trace-live (§M66), every post-pre-size `zone=WEIGHT`
         line is `cohort=routing_indices_cache` within 8192 bytes (r4
         m-2).
       - The line is WARN (`gs:18382`), so without `-lv` (or with `-lv` below
         2) `llama-cli` drops it and the arm cannot pass on a correct build
         (r3 C).
       - **Zone 1 grows**, one `Pinned pool grow_zone: zone 1 grown by
         2048.0 MB` line per host-KV buffer: GPT-OSS's full-attention K and
         V tensors are 128 MiB each at 131072 cells, so they pack the 2 GiB
         chunk cap exactly, and the count is `ceil(M / 2048)` for the
         host-KV figure `M` of the GDC3 point the arm scores against. No
         zone-3 miss, and no `[HOST-ZONE] leaked` line.
       - A refusal is a FAIL.
     - **5b. The same no-`-c` arm with `GGML_SYCL_ATTN_HOST_DISPATCH=1`** on
       its command line (§M69 (a)), the cheapest demoting arm: step 5's
       no-`-c` command with `GGML_SYCL_ATTN_HOST_DISPATCH=1` added after
       `env`, writing `step5b.out` and `step5b.err`, and step 5's
       trace-live check.
       - **Its own vehicle** (§M60 (b); r6 m-4): `llama-cli`, GPT-OSS on
         the B50, no `-c`, with the switch on. The switch changes the split
         (`gs:108273-108330`), so 5b is not step 5's vehicle. Its GDC3
         criterion is the §M63 (a) replay of **5b's own** inputs, never
         step 5's point.
       - Every other criterion of the no-`-c` arm holds, except that the
         staging criteria are the dispatch-on ones, under the precedence
         rules (§4.4 criteria; r6 I-2 (3)). A refusal is dsab's only under
         rule 1's ordered sequence; this arm is single-context, so no SYCL
         backend is freed between the commit and the first dispatch, and a
         refusal here is C4's FAIL.
       - **Parent-tree control** (r6 m-4). GPT-OSS loads on master, so the
         lead runs 5b's literal command on the parent commit too. The
         parent has none of C4's lines, so only rc, the digit count, and
         the abort and GPU-fault lines are compared. Its place in the
         scoring is rule 1's exception (b) (§4.4 criteria; r7 m-3): only an
         abort or crash whose abort or GPU-fault line also prints on the
         parent is pre-existing, reported and not C4's. A staging refusal
         or a failed graph is never pre-existing, since the parent can
         print neither, and every other failure is C4's. The host dispatch "cannot itself be
         GPU-verified" by its own account (`gs:81855-81862`), which is why
         5b needs this control.
     - **A model switch under `GGML_SYCL_STRICT_LEASES=1`** (§M57 (b)):
       `timeout 900 env GGML_SYCL_STRICT_LEASES=1
       ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench -m
       /models/mistral-7b-v0.1.Q4_0.gguf -m /models/mistral-7b-v0.1.Q8_0.gguf
       -p 0 -n 4 -r 1 -v`. `llama-bench` frees the first model and its
       context before it loads the second (CLAUDE.md, `llama-bench` trap 2),
       so every buffer-free, context-free and model-free check runs once with
       a live successor. Pass: rc 0, both rows print, and no `[HOST-ZONE]
       leaked` line and no ownerless-lease WARN. `-v` is required, or
       `llama-bench` discards every log line.
     - **The RED for §M57 (a).** C4 lands the host routing and the
       `SYCL_KV_Host` `alloc_buffer` as two commits. The no-`-c` arm above,
       run on the first (C4's routing with master's shared `SYCL_Host` path
       for `SYCL_KV_Host`), must FAIL the SCRATCH criterion: its host KV
       arrives in SCRATCH (`[HOST-ZONE-ALLOC] ... zone=SCRATCH ...
       cohort=backend_host_buffer`) with no demand line for it. On the second
       commit it passes. The RED rides step 5's no-`-c` command, which sets
       `GGML_SYCL_HOST_ZONE_ALLOC_TRACE=1`, and must be trace-live (§M66; r5
       I-3): at least one post-pre-size `[HOST-ZONE-ALLOC]` line of any
       cohort in the same `step5.err`. With the trace off, the line it looks
       for cannot print, so a non-live RED is **VOID**, never a RED that
       silently failed to fire. GPT-OSS on the B50 is the RED's vehicle because it
       demotes host KV at a fraction of Qwen's load.

     Every SYCL load now takes the mapped path.
  6. `llama-bench` Mistral Q4_0 and GPT-OSS 20B on both cards, interleaved
     against the parent commit, gated on `docs/backend/sycl-perf-baselines.md`.
     Load time is also reported, since device weights now read from page cache
     through the bounce.

**C5. The monotone budget, the headroom and the working-set WARN** (§4.5).
- RED `test-sycl-host-budget` (`sycl;host`). The pure function lives in a
  device-free header, `host-budget.hpp`.
  - It checks every row of the §4.5 table: 121.8, 125.8, 0.5 and a refusal,
    34.5, 12.0 and a refusal.
  - It sweeps `MemAvailable` from 0 to `MemTotal` in 64 MiB steps for 32, 62
    and 246.9 GiB, and asserts the budget never decreases.
  - **The override composes** (r2 m-4). With `GGML_SYCL_HOST_RESERVE_MB` above
    the formula's value, the budget is the formula's. Below it, the budget is
    the override. At zero headroom-free memory it is the named refusal
    either way.
  - Today's formula fails all three: the (246.9, 28.9) row gives 20.9, the
    sweep drops at the `available/2` cliff, and the override replaces the
    budget outright.
- RED `test-sycl-host-working-set-warn` (`sycl;host`).
  - The WARN is printed with its figures at `resident_room < mapped_ws`, not
    above it, and the plan-report line carries the same figures.
  - `resident_room` uses `host_headroom(total)`. At (62 GiB, 50 GiB
    available, 20 GiB mapped) it does not WARN, where the P3 constant would.
- It depends on th32 T2 for `host_ledger_budget()` if T2 has landed.
  Otherwise it edits `uc:4165-4231`, and T2 lifts the result.
- Lead (r3 E). C5 changes the host budget that uize-demoted host KV is
  charged against, so it can move KV placement, and every arm that places
  KV runs at the default context too:
  - the C4 Qwen `-c 4096` run once more (C4 step 2's command, `-v`),
    reading `[PLACEMENT] Totals` (the four terms) and the working-set line
    against the `/proc/meminfo` samples;
  - **the C4 step-3 Qwen no-`-c` arm**, with its pre-registered line (12
    layers, 6912.0 MB, layers 3..47) and its zone-1 criterion. A different
    N under C5's budget is a FAIL of C5, unless the arm prints the budget
    change that explains it and the general rule of step 3 predicts the
    printed N from it;
  - **the C4 step-5 GPT-OSS no-`-c` arm**, with `-lv 4`, scored on zhcn
    GDC3's line at the point its tree selects (§M56 (a), §M61 (a)), the
    VOID guard, and the digit count on stdout.
  - The same precondition as C4 steps 2 and 3 applies to both Qwen arms.
  - Each rerun uses its C4 step's **literal command**, with
    `GGML_SYCL_HOST_ZONE_ALLOC_TRACE=1` in it, and carries that step's named
    trace-live check: step 2's `backend_host_buffer` line, step 3's KV
    exact sum, step 5's `kv_host_buffer` and `backend_host_buffer` lines. An
    arm whose trace is not live is VOID, which counts as failed (§M66; r5
    I-3).
  - Each Qwen rerun scores the answer as step 2 does, anchored and
    non-degenerate, and reports any difference from C4's recorded
    reference (r6 m-3).

**C6. Retire unowned host registration** (§4.8).
- RED `test-sycl-unregistered-host-refusal` (`sycl;cache;mem-handle`,
  `level_zero:1`, lead-run). `register_host_expert` and `register_host_weight`
  with a `malloc`ed (`UNREGISTERED`) pointer are refused. Today they succeed as
  `HOST_MMAP`, `non_owning_external_host`.
- Lead:
  - the Mistral gate (default context, `-v`, the VOID guard);
  - the GPT-OSS gate in both arms, `-c 4096` and no `-c`, under
    `timeout 60`, with `-lv 4` and the stdout-only digit scoring (§7
    preamble), and the no-`-c` arm scored on zhcn GDC3's line at the point
    its tree selects, with the VOID guard;
  - the C4 Qwen run in both arms.
  - As in C5, every rerun of a C4 arm uses that step's literal command,
    with `GGML_SYCL_HOST_ZONE_ALLOC_TRACE=1`, and its named trace-live
    check; a non-live arm is VOID, which counts as failed (§M66; r5 I-3).
    Each Qwen rerun scores the answer as step 2 does and reports any
    difference from C4's recorded reference (r6 m-3).

  The refusal must never fire on a real load.

Tickets to file after review: `-lm none` double residency (§6), and
`CPU_REPACK` outside the cache (§4.4). Already filed, per §M69 (b): the MoE
segmented record/replay paths discard `ggml_sycl_compute_forward`'s return
at six sites (`gs:97234`, `:97240`, `:97285`, `:97315`, `:97551`,
`:97595`), which C4 fixes (§4.4); that is **llama.cpp-573t**, updated for
the two `moe_after` sites under the r6 amendment, so the fix is tracked
even if C4 slips.

Out of scope and already ticketed: `SYCL_CpuOffloadCompute`'s
`get_max_size`/`alloc_buffer` mismatch (**llama.cpp-g2a7**, §M62 (B)); the
`DEVICE` holders' release at every backend free (**llama.cpp-dsab**, §M62
(D)); device-side `idx_` accounting (**llama.cpp-ka9c**, addendum 11 (5));
the `UNOWNED` holders' move (**llama.cpp-30r1**, addendum 10 (b)).

---

## 8. Answers to the open questions (lead ruling §F1)

- **Q1. A mapped working-set overflow is a WARN, not a refusal.** It costs
  refaults, not an OOM. The WARN prints the arithmetic, and the plan report
  carries the same figures (§4.5). Refusal stays for pinned and anonymous
  bytes only, through th32 T2 over 7v5c.
- **Q2. 5efe defines the `FILE_MAPPED` tier and its terms; yl3v owns the
  cold-set policy** and may use `FILE_MAPPED` as the backing for cold host
  experts. §4.5 states the interface, including the terms yl3v must read so
  that DeepSeek's cold set is not implicitly empty. 5efe adds no cold-set
  logic.
- **Q3. `-lm none` stays supported on SYCL, and is not the default.** Its
  double residency gets its own ticket after review (§6).

---

## Appendix A. Disposition of design-5efe-r1 (69e70c715..de6d0d614 with addendum, 1C/10I/20m)

Rows marked "superseded in part" did not hold as written at rev 3a (design-5efe-r2); Appendix B re-disposes them.

| finding | disposition | where |
|---|---|---|
| C-1 C4 acceptance ordered before the ledger split | **Superseded in part by r2 I-1 (pre-size figure) and I-2 (runtime term withdrawn); see Appendix B.** Fixed per §F2: the split and the WEIGHT-zone sizing that reads it move into C4, one commit with the loader and no C4-only exclusion. Qwen's WEIGHT zone is pre-registered at 64.0-128.0 MB, the pre-size at ≤ 12288 MB, with a falsifier. The non-weight `SYCL_Host` runtime term is named. | §4.4, §7 C4 |
| I-1 mincore gate cannot measure the property | **Superseded in part by r2 I-3 (read-around arms re-scoped to the edge band); see Appendix B.** Fixed per §F2: `cachestat(2)` on the fd, or `mincore` on a test-owned mapping; never the tier's hole. Synchronous `pread` positive control, must-fail mutant arm, 24 MiB hole with a 1 MiB guard band, and an mlock arm. | §4.6, §7 C3 |
| I-2 multi-segment shape under-specified | Fixed per §F2: one `PROT_NONE` reservation with `MAP_FIXED` segments, one owner, one record, one `munmap`. Map segments round outward; `WILLNEED` and lock segments round inward. Bound: at most one lazy page mapped per edge. | §4.1, §4.3, §4.6 |
| I-3 stream_dma disposition legitimized streaming | Fixed per §F2: a typed, counted refusal checked before `copy_fn`, routed to the CPU executor. The counter counts any device-bound copy of mapped bytes outside a materialization. The RED asserts the refusal. | §5.3, §7 C4 |
| I-4 headroom cap universal and discontinuous | Fixed per §F2: `max(total/32, min(29 GiB, total/4))`, one named P3 constant cited as this host's measurement, the `available/2` cliff removed in favour of a named refusal, and 32 and 62 GiB cases plus a monotonicity sweep. | §4.5, §7 C5 |
| I-5 plan and loader derive "mapped" separately, and in the wrong order | Fixed per §F2: the caps are resolved before the early plan, and one `load_maps_files` fact is published. The planner's `host_backing` is read by the loader through a hook; `create_tensor` never reads `use_mmap` for SYCL weights. | §2.1, §4.3, §4.4 |
| I-6 host-only REDs need a device; no labels | Fixed per §F2: every test is named with its registration and label. Device tests are `sycl;cache;mem-handle`, `level_zero:1`, lead-run. Host tests link a device-free TU and never touch `stream_dma` or a device `mem_copy`. | §4.1, §7 |
| I-7 C1 witness re-runs the near-OOM | **Superseded in part by r2 m-1 (margin is 4.2 s, not 53 s); see Appendix B.** Fixed per §F2: the witness is killed on the two log lines (t ≈ 3.9 s, before the pre-size). Sampling every 1 s; floor and criteria both 25 GiB = 26214400 kB. | §7 |
| I-8 two C4 criteria fail on a correct build | Fixed per §F2: the page-cache growth criterion is dropped. Shard 3 gets a pre-registered bound (≤ 1 MiB before, ≤ 256 MiB after). `SYCL_Mapped` is expected at 127615 MiB ± 1%, all SYCL host-sourced weights. | §4.3, §7 C4 |
| m-1 "pinned 110 GB" | Fixed: 139264 MB committed, 140.0 GB reached, `MemAvailable` fell 141 GiB. | §0 |
| m-2 expert figures | Fixed: 512 experts, 10 used, about 1.7 MB per matrix, about 2.4 GB per cold token, 1-6 ms per matrix. | §5.2 |
| m-3 census misses tests | Fixed: `tests/test-cpu-gpu-soa-interaction.cpp:178/180/182` added, and the census gate covers `tests/`. | §4.8, §7 C2 |
| m-4 `alloc_type::MMAP` understated | **Superseded in part by r2 m-11 (`HOST_NON_USM` replaced by `HOST_MALLOC` and `HOST_OWNED_ELSEWHERE`); see Appendix B.** Fixed: ten consumers listed, the `::free` meaning recorded, renamed `HOST_NON_USM`, and a `FILE_MAPPED` branch ordered before `::free` with a counter test. | §2.7, §4.1, §4.8 |
| m-5 fd lifetime and replacement gap | Fixed: `dup` of the loader's fd, kept by the owner for `FADV_DONTNEED`; no reopen by path. | §4.1, §4.6 |
| m-6 loader sites under-enumerated | **Superseded in part by r2 m-8 (C4 amends the C1 caps test); see Appendix B.** Fixed: the §4.3 site table (hooks, `weight_buft_supported`, `make_cpu_buft_list`, `mlock_bufs`, the `init_mappings` prefetch) and the loader `lmlock` skip. | §4.2, §4.3, §5.1 |
| m-7 null path unverified; AOS branch | Fixed: `resolve_or_host` aborts naming the tensor, with the eight callers audited and pinned, and `binbcast.cpp:394/794` cited. Both layout-availability branches are excluded for `SYCL_Mapped`. | §5.3 |
| m-8 stale th32 pin | **Superseded in part by r2 m-15 (re-pinned at rev 2u `0bd69e251`); see Appendix B.** Fixed per §F2: pinned at the latest reviewed head, rev 2q `1646da864` (see the lead-check row below); 2s `f364952d3` is noted as the revision that reads the record. Rulings relabelled §T13/§T13a/§T13c. | header, §4.6 |
| m-9 F1 fold and cold set | Fixed per §F2: the yl3v/7v5c interface is stated, the recommendation removed, and the terms yl3v must read are named so that DeepSeek's cold set is not implicitly empty. | §4.5, §8 |
| m-10 enum value collision | Fixed per §F2: the next free value at landing; censuses key on names. | §4.1, §4.8 |
| m-11 allowlist unenforced | **Superseded in part by r2 m-12 (the gate tokenizes); see Appendix B.** Fixed: `test-sycl-raw-mmap-allowlist-source` with today's eight sites, shared with th32 T1. | §4.1, §7 C2 |
| m-12 C2 RED weak | **Superseded in part by r2 m-6 (the rev 3a mutant was equivalent); see Appendix B.** Fixed: named mutants plus a private `munmap` counter and a registry query; no `mincore`-ENOMEM check. | §7 C2 |
| m-13 staging faults can populate lazy pages | Fixed: `MADV_RANDOM` on segment edges bordering lazy ranges covers execution and bounce faults; the gate reads every byte of A and B, edges included. | §4.6 |
| m-14 GPT-OSS gate lacks `timeout 60` | Fixed per §F2: the canonical command with `-c 4096` under `timeout 60`, scored with `grep -cx`. | §7 C4, C6 |
| m-15 shard-3 "~0" not pre-registered | Fixed: ≤ 1 MiB before, ≤ 256 MiB after, with the arithmetic. | §7 C4 |
| m-16 64 MiB claim too broad | Fixed: scoped to `mem_copy`; the reorder, WOQ and XMX fill paths are listed as unchanged and bounded per source tensor. | §5.4 |
| m-17 host_unified fails open | Fixed: `host_unified_known` added, and `mmap_support` requires it. | §3 |
| m-18 storage_owner type; PER_DEVICE teardown | Fixed: `storage_owner` wraps the reservation's `mem_handle`; every model unloads on every device before any cache tears down. | §4.2 |
| addendum I-2(a) §4.6 text wrong for map segments | Fixed per §F2a: already corrected at `1ddbd9d56` (map segments outward); rev 3 states it at the table, adds the `MADV_RANDOM` edge windows as an inward set, and says that only map segments round outward. | §4.6 |
| addendum I-9 lazy_ranges source disappears after th32 T6 | Fixed per §F2a and §M49: one per-file lazy-range record, owned by 5efe (filled before the early plan from the arch declaration, per addendum 2 below), landing in C1. `lazy_read`, th32's inventory, the tier and `llama_mmap` consume it and keep no copy. Prefetch 0 on SYCL is an explicit C4 item, with RED `test-llama-mmap-lazy-exclusion` (its mutant is an empty record). | §3.1, §4.6, §7 C1, C4 |
| addendum I-10 tier and loader inherit the prefix lock | Fixed per §F2a: the tier `mlock(2)`s inward-rounded host-executed segments directly; `SYCL_Mapped` is excluded from `mlock_bufs`; §2.5 is corrected; `lml:2122` becomes a ranged `lock_range` in 5efe (C3). Host-only RED `test-llama-mlock-ranged` with a lazy range before a locked range and a whole-prefix mutant; the tier's arm 2 covers the tier side. | §2.5, §4.6, §5.1, §7 C3 |
| addendum m-19 one-sided edge advice; residual undercounted | Fixed per §F2a: `MADV_RANDOM` windows on both edges, because read-around is centred on the fault. The loader's whole-file `llama_mmap` read-around is named as a second residual path, given the same record-fed window (C4), and bounded at one shared page per edge per mapping, plus bcachefs extent granularity inside the guard band. | §4.6 |
| addendum m-20 readahead case could pass vacuously | Fixed per §F2a: arms 5 (forward) and 6 (backward) fault real mapped pages, exit 77 when `read_ahead_kb` is 0, each has a no-advice control twin that must populate the hole, and both are measured with `cachestat` or a test-owned mapping. | §4.6, §7 C3 |
| §F2a I-9 addendum 2: early plan cannot see the record | **Superseded in part by r2 I-5 (declaration keyed by GGUF name, no hparams; static gate) and I-6 (STORAGE writer); see Appendix B.** Fixed: the flag is first knowable only at `create_tensor`, so C1 adds a per-arch lazy-tensor declaration and pre-fills the record from it plus `weights_map`, after the caps and lazy-mode loops move above the early plan. The early plan skips deferred rows; `create_tensor` checks the record rather than filling it, and aborts on disagreement. C-1's numbers are re-checked and unchanged, because the zones come from the late plan. Early-plan witness lines are pre-registered in C1 and C4. | §3.1, §2.6, §4.3, §4.4, §7 C1, C4 |
| §F2a I-9 addendum 2 (M-1): by-value hand-off | Done: `value(name)` returns `{file_idx, offs, size}` by value, rows are immutable after the pre-fill, and deriving the numbers independently (from `w->offs`) is named as forbidden. | §3.1 |
| lead check: th32 pin | **Superseded in part by r2 m-15 (re-pinned at rev 2u `0bd69e251`); see Appendix B.** Pinned at rev 2q `1646da864` per the lead; 2s `f364952d3` is named as the record's consumer, and §3.1's row fields match its §4.1. | header |
| §M49 note: record in C1 | Done: the record and its two host-only tests are C1 deliverables, so th32 T6 depends on C1 alone. | §3.1, §7 C1 |
| th32 r15 m-4 (relayed) mlock prefix | Fixed: the tier's own per-segment `mlock` (named, and where), no `llama_mlock` for `SYCL_Mapped`, a residency arm under mlock, and a RED with a lazy range before a locked tensor. | §4.6, §5.1, §7 C3 |

## Appendix B. Disposition of design-5efe-r2 (rev 3a `3763bb164`, 0C/6I/19m), with held rulings

| finding | disposition | where |
|---|---|---|
| I-1 pre-size omits the runtime pre-allocate | Fixed: the criterion is split, and both numbers are pre-registered from the configure path. The base is 4 chunks and the runtime is 3 chunks, so the pre-registered line was `Committed 14336.0 MB in 7 chunk(s)`. **Superseded in rev 3d by r3 m3:** the runtime pool drops the two PP terms, so it is 1 chunk and the line is `Committed 10240.0 MB in 5 chunk(s)`; §4.4's table maps 6, 7 and 8 chunks to what did not land. **Superseded again in rev 3f** (addendum 11 (1), §M62 (A); r4 m-8, m-9): host SCRATCH drops its device-sizing terms, so the line is `Committed 6144.0 MB in 3 chunk(s)`, scored on the printed MB figures. | §0, §4.4, §7 C4 |
| I-2 WEIGHT has non-weight tenants; the runtime term has no path | Fixed per the lead's rulings on I-2 and addendum 7 (b)/(c). 5efe routes host-side non-weight requests by role: HOST_COMPUTE and EXPERT_CACHE go to SCRATCH. The rev 3a "runtime term" is withdrawn. The criterion is **no `Pinned pool grow_zone: zone 0` line after the pre-size**. **Sizing:** `ggml_backend_sycl_notify_compute_buffer_sizes` (gs:19103-19146) has no caller, and it also reserves the VRAM arena and sums device buffers into host SCRATCH. So it stays uncalled, and C4 adds the host-only, scalar-ABI `ggml_backend_sycl_notify_host_buffer_sizes(backend, host_compute_bytes, host_output_bytes)`, reached through the proc address; a stale backend yields NULL and a WARN. Whether the old function's VRAM half should get a caller is reported to the lead. **Lifetime** (confirmed over a fifth zone): `alloc_intent::lifetime` {`GRAPH`, `CONTEXT`, `MODEL`}, with no `PROCESS`. The graph-boundary check counts `GRAPH` only. `CONTEXT` is leak-checked at backend-context destruction and `MODEL` at model free, each a WARN that aborts under STRICT_LEASES. **The site table** (35 lines) is published in §4.4, with a holder, role, zone, lifetime and owner per row: 10 `GRAPH`, 4 `CONTEXT`, 1 caller-chosen, 2 non-tenants, and **14 holders (18 lines) UNOWNED**. The UNOWNED rows are file-scope statics and `thread_local`s, including the `CpuExpertPool` and `PinnedBufferPool` rings on Qwen's default path, and are brought to the lead. **23mk:** 5efe owns role, zone, lifetime and sizing, and 23mk owns conversion and cites the table. Host SCRATCH reads `unified_cache_get_planned_runtime_zone_requirement` in place of three copied terms (**withdrawn in rev 3f**: addendum 11 (1), §M61 (d)(2)). | §4.4, §7 C4 |
| I-3 arms 5 and 6 cannot reach the hole interior | Fixed: read-around is centred on the fault, with reach `ra_pages/2`, doubled under `FADV_SEQUENTIAL`, where `ra_pages` comes from the bdi's `read_ahead_kb`. Arms 5 and 6 measure the **edge band** `[shared page + 1 page, edge + reach)` at each lazy edge (§4.6). `W = max(2 x read_ahead_kb, 2 MiB)` is a different quantity, the width of the `MADV_RANDOM` window C4 applies at both edges; rev 3c's row gave that name to the band (r3 m5). The interior is measured by arms 1 and 2 only. Each arm exits 77 if the computed reach does not extend two pages past the shared page. | §4.6, §7 C3 |
| I-4 `MADV_HUGEPAGE` defeats the edge windows | Fixed: `do_sync_mmap_readahead` tests `VM_HUGEPAGE` before `VM_RAND_READ`. So C4 applies `MADV_NOHUGEPAGE` to the `MADV_RANDOM` windows widened to 2 MiB, inside `llama_mmap`'s `HUGEPAGE` advice (`src/llama-mmap.cpp:517-532`). The one-page bound is restated per mapping. A faulting arm with lazy boundaries at 2 MiB + 1 MiB offsets gets a `HUGEPAGE` control mutant. | §4.6, §7 C4 |
| I-5 the declaration needs a static cross-check | Fixed per the lead ruling, as amended by addenda 5, 7 (a) and 8 (a). The declaration is 5efe's, and th32 consumes it. The final signatures (addendum 8 (a): rev 3c's stand, and th32 does not rename) are `const std::vector<std::string> & llama_model_lazy_tensor_names(llm_arch)`, keyed by (arch, exact name or one `%d`), names only; `struct llama_lazy_tensor_attrs { uint32_t rows_per_token; int32_t consumer_layer; }`; and `llama_model_lazy_tensor_attrs(arch, hparams, name)`, 5efe's. C1's pre-fill stores the attributes on the per-model record entry, and T6 reads `record.attrs(name)`. There is no static table: one that T6 filled would be shared across models (P4). An `LLM_TENSOR_INFOS` column was rejected, and the arch key is required, because `gemma3n.cpp:39` creates the same tensor unflagged. `test-llama-lazy-declaration-source` maps (arch, call site) to entry and back, with four mutants plus a static-attribute-cache mutant. The record test has a two-hparams P4 witness. The runtime abort is kept and is mode-independent. | §3.1, §7 C1 |
| I-6 no writer for `STORAGE`; eligibility superseded by addendum 3 | Fixed by folding addenda 3 to 6. The pre-fill writes every field once. `lazy_eligible = T6a(taker ? mode_pre : mode_post, size, taker ? true : SUPPORTED)`. `deferral` is `STORAGE` iff eligible and taker, `LAZY_READ` iff eligible and not taker, and `EAGER` otherwise, so `deferral != EAGER` iff eligible. `taker` is `sycl_model_loading_guard.active` (`lm:2228-2230`), not `hooks_enabled()` (rev 3b's `sycl_branch` is withdrawn). The ineligible early-plan witness is a host test plus lead witness W2. th32 was pinned at rev 2w `3afc3b020`, and is pinned at rev 2x `90be8e260` since rev 3d. | header, §3.1, §7 C1 |
| m-1 witness margin | Fixed: the lines are all printed by t = 3.9 s, and chunk 1 is at t = 8.12 s (`run2.log:2827`), a margin of about 4.2 s. `GGML_SYCL_HOST_RESERVE_MB=16384` is added as a belt. | §7 C1 |
| m-2 idle DMA pool | Fixed: the DMA staging pool is sized 0 when no host weight is `PINNED`-backed, and the `DMA staging pool: 0.0 MB` line is a C4 criterion. | §4.4, §7 C4 |
| m-3 readers not censused | Fixed: a readers table for `weight_host_bytes`, with each reader's assigned meaning, pinned by the loader source test. | §4.4, §7 C4 |
| m-4 two headroom expressions; override bypass | Fixed: `resident_room` uses `host_headroom(MemTotal)`, and the "one number" claim is scoped to MemTotal ≥ 116 GiB. `HOST_RESERVE_MB` becomes `min(override, host_ledger_budget())`, so it can lower the budget but never exceed the headroom. `test-sycl-host-budget` covers it. | §4.5, §7 C5 |
| m-5 getrows retries a refusal | Fixed: `refused_file_mapped` is terminal in the getrows slice loop, so there is one call, one counted refusal, no retry line, and the CPU fallback. `test-sycl-getrows-refusal-terminal` covers it. | §5.3, §7 C4 |
| m-6 equivalent tier mutant | Fixed: the mutants are the record registered as `HOST_MALLOC`, and the `FILE_MAPPED` branch moved after `zone_managed` into the `sycl::free` fallthrough. | §7 C2 |
| m-7 preload site | Fixed: `gs:34701-34735` is added to the §4.3 site table and §5.3, and the loader source test pins it. | §4.3, §5.3 |
| m-8 C1 test vs C4 deletion | Fixed: C4 narrows `lm:2693` to the no-tier case and amends the C1 caps test. | §4.3, §7 C1, C4 |
| m-9 pre-fill `no_alloc` | Fixed: under `no_alloc` the pre-fill still runs (rows are metadata), `lazy_read.any()` is false, and no range reaches `llama_mmap`. | §3.1, §7 C1 |
| m-10 `SYCL_Mapped` `alloc_buffer` | Fixed: size 0 returns a dummy buffer, and any other size aborts naming the buft. | §4.3 |
| m-11 `memory_tier::MMAP` | Fixed: listed in §2.7 and renamed `UNRESOLVED` in §4.8. The census gate covers all four legacy names. `alloc_type::MMAP` is split into `HOST_MALLOC` (`::free`) and `HOST_OWNED_ELSEWHERE` (the KV synthetic base, gs:39620-39624, :39641-39642, released at :38408-38415). | §2.7, §4.8, §7 C2 |
| m-12 allowlist tokenizer | Fixed: comments and strings are stripped, `mmap (` is matched with a fixture, and a non-listed-file mutant is added. | §4.1, §7 C2 |
| m-13 AnonPages has no positive control | Fixed: `test-sycl-pinned-meminfo-control` (lead-run) names the counter that pinned host USM lands in. The C4 criterion keys on that counter. If none moves, the test exits 77 and the criterion rests on `Committed` plus `MemAvailable`. | §7 C4 |
| m-14 residency-test hygiene | Fixed: a non-tmpfs fixture, eviction per arm, `cachestat` through `syscall(__NR_cachestat)`, the `RLIMIT_MEMLOCK` skip, and arm 4 last on a fresh fixture. | §4.6, §7 C3 |
| m-15 th32 pin | Fixed: re-pinned at rev 2u `0bd69e251` in rev 3b, at rev 2w `3afc3b020` in rev 3c, and at rev 2x `90be8e260` in rev 3d (r3 X7). | header |
| m-16 map segments | Fixed: exact definitions. Map segments are the union of the buffer's tensor ranges rounded outward, never a complement. `WILLNEED` and lock segments round inward. The pure-function test has a complement mutant. | §4.6, §7 C3 |
| m-17 counter chokepoints | Fixed: `mem_copy`, `mem_copy_async` and `mem_copy_ptr_async` (`mem-ops.hpp:13-62`), `copy_to_device_async` (uc:10799), and `stream_dma` (uc:13959). | §5.3, §7 C4 |
| m-18 counter log line | Fixed: `[FILE-MAPPED] counters: stream_dma_refusals=N device_copies_outside_materialization=M`, printed at WARN at load end and at unload. | §5.3, §7 C4 |
| m-19 non-POSIX builds | Fixed: `load_maps_files` requires the tier to exist in the build (POSIX only). On Windows SYCL it is false, and the no-tier caps path of `lm:2693` applies. | §4.3 |
| addendum 3 (§F2a I-9): `lazy_eligible` bit | Folded: one bit per row, filled once at the C1 pre-fill by calling T6a's predicate (a): OFF never, AUTO above 4 GiB, ON always, on the pre-fallback mode. Consumers read the bit, and an ineligible row is an ordinary weight. Membership stays flag-based. Data edge: T6a lands no later than C1. | §3.1, §7 C1 |
| owner ruling: default-context arms | Folded: every Qwen and GPT-OSS gate set has a no-`-c` arm, HARD and scored. The Mistral gate is already default-context. | §0, §7 C4, C6 |
| correction: uize part 3 landed | Folded: `35ce2e65f`, `812c7d790`, `7259727e6` and `ae4204ba7` are in the base, so a no-`-c` arm expects the `KV overflow re-placed to host tier` line (gs:18382) with a pre-registered N and range. Any refusal line is a FAIL, and so are garbage tokens, an abort, or `Shmem` growth. Qwen: 12 layers, layers 3..47 (real sizing), or 48 layers, 0..47 (the over-claim); the figures became 6912.0 and 27648.0 MB in rev 3d with the indexer term (r3 B), and the over-claim outcome is a FAIL attributed to 0edf (r3 n1). GPT-OSS on the B50: rev 3d cites zhcn's GDC3 row and carries no copy of its figures (§M56 (a)). | §7 C4 |
| llama.cpp-0edf: the x4 hybrid KV over-count | Cited, and kept out of 5efe's scope. The Qwen no-`-c` arm's two pre-registered lines discriminate it, but do not fix it. | §7 C4 |

| addendum 4 (§F2a I-9): which mode feeds the bit | Folded in rev 3c. The pre-fallback capture `mode_pre` is C1's, taken before the moved fallback block. The bit reads `mode_pre` for the taker and `mode_post` otherwise. `deferral` carries only the kind, and `STORAGE` is written at the pre-fill. The drift check is mode-independent, with a RED mutant of rev 3a's form. The required witness is a host arm with `AUTO`, an `mmap_support = false` device and a sparse table above 4 GiB (bit set, early plan excludes it; RED: a post-fallback fill). deepseek41 is dropped from the declaration. The stale "commit 1 does not depend on that" bullet (end of §3) is corrected. | §3, §3.1, §7 C1 |
| addendum 5 (§F2a I-9): the taker is per-model; (arch, name) key; `add()` reads the bit | Folded in rev 3c. `taker = sycl_model_loading_guard.active`, captured once at the pre-fill. The declaration is keyed by (arch, name), and the attributes come from one per-arch function. `lazy_read::add()` accepts iff `deferral == LAZY_READ`, and its mode, floor and `SUPPORTED` tests are deleted and pinned by a source gate (mutant: upstream's `add()`). The `-ngl 0` witness: the host matrix never yields `STORAGE` without `taker`, the source gate has a process-level-selector mutant, and lead witness W3 expects `deferral=LAZY_READ taker=0` with no early plan. The missing-attribute refusal is T6's, for `STORAGE` rows only. | §3.1, §7 C1 |
| addendum 6: th32 rev 2w interpretations | Folded in rev 3c. T6a's `SUPPORTED` input is `llama_mmap::SUPPORTED` off the taker and `true` on it. `add()`'s switch and its source gate are C1's. The attribute function's name is `llama_model_lazy_tensor_attrs(arch, hparams, name)`. | §3.1, §7 C1 |
| vehicle clarification and the programmatic `n_ctx = 0` rule | Folded in rev 3c. The no-`-c` arms are `llama-completion` (Qwen, Mistral) and `llama-cli` (GPT-OSS). `llama-bench` and `llama-perplexity` stay fixed. A programmatic arm sets `n_ctx = 0` (`src/llama-context.cpp:5091`, `:540`) and is VOID unless `llama_n_ctx == n_ctx_train`. | §7 |
| found in rev 3c: the C1-to-T6 window | Ruled in addendum 8 (b). C1 and T6 reach master in one merge, with T6 built on C1; the one-merge constraint is recorded as comments `c-q50c` and `c-0elm`, with no tracker edge (a C1-on-T6 edge would close a cycle; corrected in the post-r9 revision). Interim `EAGER` is refused. C4's Qwen arms are VOID without T6 and T7, and their host-growth bound adds th32's printed table bytes; both were accepted. | §3.1, §7 C4 |

| addendum 7 (a), (b), (c) | Folded in rev 3c. (a): no mutable static table, and attributes stored on the per-model record entry; the naming follows addendum 8 (a) (see I-5). (b): lifetime {GRAPH, CONTEXT, MODEL}, owner-boundary leak checks, owners cited, UNOWNED rows brought to the lead (see I-2). (c): the published site table, 23mk citing it, and host SCRATCH reading the RUNTIME function, not copying it (the read is **withdrawn in rev 3f**, addendum 11 (1)). The pre-size stays 4 + 3 chunks, since the base moves by at most +0.6 MB. dlopen: the new entry point is scalar-only, under a new name. | §3.1, §4.4, §7 C1, C4 |

| addendum 8 (a), (b) | Folded in rev 3c. (a): rev 3c's `_names`/`_attrs` signatures stand. (b): the C1-to-T6 window is closed by one merge. | §3.1, §7 C4 |

**design-5efe-r3** (reviewed 3763bb164..3b447ee70, with spot-checks of rev 3c;
0C/5I/7m/1n and contradictions X1-X7), folded in rev 3d:

| finding | disposition | where |
|---|---|---|
| r3 A notify unsound: VRAM half, ordering, chunk rounding, unchecked cast, N-times growth | A.1 and A.4 were fixed in rev 3c (the host-only, scalar-ABI entry point under a new proc name); rev 3d adds the `ggml_backend_is_sycl` test before `backend->context` is read (`gs:107857`). **A.2:** the notify now precedes each allocation it covers: in `output_reserve` between the old buffer's free (`src/llama-context.cpp:3550`) and the allocation (`:3566`), and in `sched_reserve` after size-only `graph_reserve(..., true, sizes)` calls (`ggml_backend_sched_reserve_size`, `:3908-3910`) and before the first allocating reserve (`:2055`); a WARN compares the allocated size with the notified one. **A.3:** SCRATCH is scored by chunks, one `grow_zone` of `max(2048 MB, align_up(deficit + n x alignment))` (`pinned-pool.cpp:613-712`, `:798-803`). **A.5:** one demand per `llama_context`, keyed by one backend context, replaced on re-notify and dropped at that context's destructor; the global zone grows by `sum - grown_for_demands`. `test-sycl-host-scratch-demand` has the accumulate and per-backend mutants. | §4.4, §7 C4 |
| r3 B demoted host KV routed to SCRATCH; zone-1 criterion vacuous; the indexer omitted | Fixed per the lead's ruling. `SYCL_KV_Host` gets its own `alloc_buffer` with role `KV`, zone 1, lifetime `CONTEXT` (it copied `SYCL_Host`'s iface, `gs:42755-42763`). The qwen4exp `idx_` cache (`src/llama-memory-hybrid-idx.cpp:45-67`, `lm:3580-3585`) takes the same path, and the demotion counts it through `kv_idx_width`, a `uint32_t` in the inventory's tail padding (measured: `sizeof` stays 176), and `kv_demand_for_layer`. Qwen's no-`-c` line is 12 layers, 6912.0 MB, 3..47. Zone 1 must grow, scored by chunks: four 2048.0 MB `grow_zone: zone 1` lines. "Plus a small recurrent state" is withdrawn: the recurrent state is device-resident (`src/llama-memory-recurrent.cpp:97-103`). | §4.4, §7 C4 |
| r3 C `llama-cli` drops WARN | Fixed. `tools/cli/cli.cpp:36` sets `LOG_LEVEL_ERROR` before parsing; `-lv N` overrides it, since the parser sets the threshold from the default first (`common/arg.cpp:769`) and the `-lv` handler sets it again (`:3950-3962`). Every `llama-cli` arm passes `-lv 4`. The digit scoring is unaffected because the answer is stdout (`common/console.cpp:75`) and logs are stderr (`common/log.cpp:113-116`); the arms capture them separately. The `[FILE-MAPPED] counters` claim is scoped to `llama-completion`. | §4.6 counters, §7 preamble, C4 step 5, C5, C6 |
| r3 D the C1-to-T6 window | Closed in rev 3c by addendum 8 (b), and the verdict's 3c spot-check confirms it: C1 and T6 merge together (recorded as comments `c-q50c` and `c-0elm`, with no tracker edge; corrected in the post-r9 revision), and W1 kills on `deferral=STORAGE taker=1`. | §3.1, §7 C1 |
| r3 E C5 has no no-`-c` arm | Fixed: C5's lead runs add the C4 step-3 Qwen arm and the step-5 GPT-OSS arm with `-lv 4`. | §7 C5 |
| r3 m1 `add()`'s `w != nullptr` breaks the fit pass | Fixed: `add()` returns `deferral == LAZY_READ` alone, and inserts into membership only when `w` is non-null, so the `no_alloc` fit pass still reads the table as lazy (`lml:1775-1776`). The "as today" wording is gone. A RED mutant restores the rev 3c term. | §3.1, §7 C1 |
| r3 m2 Qwen arms' vehicle and `-v` | Fixed: `llama-completion -no-cnv -v`, and the zone-0 absence counts only beside its `pre_allocate` positive control. | §7 preamble, C4 step 2 |
| r3 m3 double charge of the PP terms | Confirmed still present in rev 3c: the runtime pre-allocate (`gs:5649-5653`) kept `pp_pipeline + pp_moe_onednn`. One owner, host SCRATCH (addendum 7 (c)); the runtime pool, which after zone configuration serves only the KV-zone miss fallback (`uc:15483-15495`), drops them. Re-derived: base 4 chunks + runtime 1 chunk (91.9 MB) = **5 chunks, 10240.0 MB**; 6, 7 and 8 chunks name what did not land. Both PP terms are VRAM allocations (`uc:18197-18199`), so the RUNTIME read left in host SCRATCH sizes no host tenant; removing it would amend addendum 7 (c) and was brought to the lead, who removed it (§M61 (d)(2); **done in rev 3f**, r4 m-8). | §0, §4.4, §7 C4 |
| r3 m4 rerouted SCRATCH tenants have no sizing term | Fixed: context buffers are the notified demand; `DEVICE` holders register a demand before allocating (the pool formulas from `gs:8018-8030`, `pinned-buffer-pool.cpp:38-39`, `cpu-expert-pool.cpp:56-59`); host KV sizes zone 1 through the miss path at context init; each `GRAPH` row maps to a named plan term or is ratcheted `UNSIZED`. | §4.4, §7 C4 |
| r3 m5 I-3 row misnames the band | Fixed in the I-3 row above: the band is `[shared page + 1 page, edge + reach)`, and `W` is the `MADV_RANDOM` window. | Appendix B |
| r3 m6 `gs:39959` is a P3 zero-copy pattern | Fixed: the site table cites llama.cpp-nl0f beside 23mk's row 56. | §4.4 |
| r3 m7 proc-address name | Fixed in rev 3c; the r3 spot-check confirms it. | §4.4 |
| r3 n1 0edf outcome | Changed per the lead: the 48-layer line (27648.0 MB with the indexer) is a **FAIL**, plan != reality, attributed to llama.cpp-0edf and not a 5efe regression. **Relabelled in rev 3f** (§M61 (d)(3), r4 m-10): a 5efe PASS plus a live 0edf defect. | §7 C4 step 3 |
| r3 X1 PROCESS lifetime | Closed in rev 3c: `{GRAPH, CONTEXT, MODEL}`, no `PROCESS` (addendum 7 (b)); rev 3d adds `DEVICE` (addendum 10 (b)). | §4.4 Lifetimes |
| r3 X2 attributes in the declaration struct | Closed in rev 3c: names only in `llama_model_lazy_tensor_names`, attributes from `llama_model_lazy_tensor_attrs(arch, hparams, name)` stored per model (addenda 7 (a), 8 (a)). | §3.1 |
| r3 X3 `sycl_branch` from `hooks_enabled()` | Closed in rev 3c: `taker = sycl_model_loading_guard.active` (`lm:2228-2230`). | §3.1 |
| r3 X4 the bit always reads `mode_pre` | Closed in rev 3c: `T6a(taker ? mode_pre : mode_post, size, taker ? true : SUPPORTED)`; rev 3d pins the third argument in source (th32 2x). | §3.1, §7 C1 |
| r3 X5 eligible-but-`EAGER` | Closed in rev 3c: `deferral != EAGER` iff `lazy_eligible`. | §3.1 |
| r3 X6 23mk overlaps unresolved | Closed in rev 3c: the published site table, which 23mk's class-H rows cite. | §4.4 |
| r3 X7 th32 pin stale | Fixed: re-pinned at rev 2x `90be8e260`. | header |
| found in rev 3d: host bufts' `get_max_size` names its own zone | `SYCL_Host`'s answers from WEIGHT (`gs:42482`) and `SYCL_HostCompute`'s from STAGING (`gs:42982`); after C4's routing both would chunk against the wrong zone (the llama.cpp-glkg shape). Each buft now builds one intent for `alloc_buffer` and `get_max_size`, which calls `select_zone`; `tests/test-sycl-host-zone-config-source-contract.py` is amended. **Corrected in rev 3f** (r4 I-A): `gs:42982` is `SYCL_CpuOffloadCompute`'s, not `SYCL_HostCompute`'s, and the builder set is re-derived per buft. | §4.4, §7 C4 |
| addendum 10 (a) never amend a reported commit | Followed: rev 3d is a new commit on `fef860772`. | header |
| addendum 10 (b) the 14 UNOWNED holders | Folded. **The owner and release point are corrected in rev 3f** (§M62 (D), r4 I-B): the intended owner is the last live backend on the device, and the release today is at every backend free (llama.cpp-dsab). As folded in rev 3d: `DEVICE` (owner: the device's `unified_cache`, checked at its teardown) for the two pools, and for `g_attn_host_staging` under the ruling's device-shared clause, with the reason stated; all three are released by `ggml_sycl_shutdown_global_runtime_pinned_owners` (`gs:7575-7590`) before the cache goes. The other 11 holders (12 lines) are `CONTEXT` targets pending **llama.cpp-30r1**, and carry the interim `UNOWNED` marker, routed to WEIGHT as today, until it lands. The ratchet fails on any new `UNOWNED` site and cites llama.cpp-30r1. | §4.4, §7 C4 |
| addendum 10 (c) notify | Folded: the scalar entry point stands; the old function stays uncalled and its VRAM half is 23mk's dead-code census's. | §4.4 |
| §M56 (a) GPT-OSS figures have one source | Folded: C4 step 5, C5 and C6 cite zhcn's GDC3 row and carry no copy of its N, figure or range. | header, §7 C4, C5, C6 |
| th32 2x C1 requirement 1: the third argument in source | Folded: `test-llama-lazy-range-record-source` pins `taker ? true : llama_mmap::SUPPORTED`, with the always-`SUPPORTED` and always-`true` mutants; the reason is that `SUPPORTED` is a constant `true` on Linux and Windows (`src/llama-mmap.cpp:711-715`). | §7 C1 |
| th32 2x C1 requirement 2: one per-model line | Folded: `lazy-range record: mode_pre=<m> mode_post=<m> taker=<0|1>` at INFO, once per model; a `llama_log_set` callback receives every level (`src/llama-impl.cpp:40-55`), and only `common_log` drops INFO (`common/log.cpp:542-546`). W1-W3 read it. | §3.1, §7 C1 |

**design-5efe-r3, full verdict** (3763bb164..fef860772, 0C/7I/9m/1n; it
supersedes the 3b-only draft that rev 3d folded as A-E), under lead rulings
§M57 and §M58 (c), folded in rev 3e. Where rev 3d already did it, the row
says so:

| finding | disposition | where |
|---|---|---|
| I-1 notify after allocation, additive ratchet, unscoreable SCRATCH criterion, device-0 grow | (a) and (c): done in rev 3d (notify before each allocation it covers; chunk-scored). (b), per §M57 (c): the SYCL side is an **ensure-capacity high-water** call, target = plan SCRATCH + the cache's live demands; the same demand repeated by `sched_reserve`'s sites grows nothing, and today's additive `grow_scratch_zone` (`uc:20442-20449`) is not called. (d), per §M57 (c): it grows `ctx->device`'s cache, the host buft's (`gs:42639-42642`, `lc:889-895`), not `resolve_effective_device(0)`'s (`uc:21935-21940`); rev 3d's "the zone is global" is withdrawn. New demand-test cases and mutants for both. **(d) superseded in rev 3f by r4 I-D:** `ctx->device` is not the host buft's device, since `SYCL_Host` is one global buft on reg device 0; the target is `ggml_sycl_host_buft_device(buft)`. | §4.4 notify, §7 C4 |
| I-2 host KV and the indexer land in SCRATCH; zone-1 criterion vacuous | Done in rev 3d (`SYCL_KV_Host`'s own `alloc_buffer`, role `KV`; `kv_idx_width`; 6912.0 MB). §M57 (a) adds the exact form, the `[HOST-ZONE-ALLOC] zone=KV cohort=kv_host_buffer` bytes summing to exactly 6912.0 MiB (`uc:4002-4008`), and the RED: C4 lands the routing and the `SYCL_KV_Host` split as two commits, and the GPT-OSS no-`-c` arm must FAIL the SCRATCH criterion on the first. | §7 C4 steps 3 and 5 |
| I-3 CONTEXT owner check unsound for buffer-owned rows and host KV | Fixed per §M57 (b). A buft allocation is owned by its `ggml_backend_buffer` and checked at its `free_buffer` (`gs:42563`), not at backend-context destruction, since `sched` (`lch:398`) and `memory` (`lch:342`) outlive `backends` (`lch:403`). Backend-context members are checked after `delete sycl_ctx` (`gs:84119`). The `in_model_load` branch of `gs:42652` is `MODEL`. rev 3d's context allocation scope for these owners (th32 T6) is withdrawn, so 5efe no longer depends on it. Criterion: under `GGML_SYCL_STRICT_LEASES=1`, no leaked or ownerless-lease WARN and no abort across the `llama-cli` `-c 4096` arm and a multi-`-m` `llama-bench` switch. Destruction-order test with a mutant. | §4.4 Lifetimes, site table, criteria; §7 C4 |
| I-4 UNOWNED rows default to GRAPH and undo cg8j | Done in rev 3d: `DEVICE` for the two pools (and the attention host staging, device-shared, reason stated), and the other holders stay in WEIGHT, their current zone, until llama.cpp-30r1. §M57 (c)'s count, "the other 12", treats the attention staging as a CONTEXT target; rev 3d moved it to `DEVICE` under addendum 10 (b)'s device-shared clause, which leaves 11 holders (12 lines). That was brought to the lead, who confirmed it (§M61 (d)(1)); its premise, release "at module teardown", is corrected by §M62 (D) (r4 I-B). New: the zero `refusing SCRATCH host_zone_boundary_check` WARN criterion, a WARN visible at `llama-completion`'s default verbosity. | §4.4 criteria, §7 C4 step 2 |
| I-5 the PP terms charged twice across host pinned | Done in rev 3d: `runtime_bytes` drops both terms; 5 chunks, 10240.0 MB. The host SCRATCH fold's own idle VRAM-RUNTIME read was brought to the lead; **removed in rev 3f** (§M61 (d)(2), r4 m-8), so the line is now 3 chunks, 6144.0 MB. | §4.4, §7 C4 |
| I-6 `llama-cli` drops WARN | Done in rev 3d: `-lv 4` on every `llama-cli` arm; the `[FILE-MAPPED] counters` claim scoped to `llama-completion`. | §4.6, §7 |
| I-7 C5 has no no-`-c` arm | Done in rev 3d. | §7 C5 |
| m1 `add()`'s `w != nullptr` | Done in rev 3d. | §3.1, §7 C1 |
| m2 Qwen arms need `-v` | Done in rev 3d, with the `pre_allocate` positive control. | §7 C4 step 2 |
| m3 the C1-to-T6 window's mechanism | Fixed per §M57 (d). (a): C1 lands `create_tensor`'s `STORAGE` branch, whose C1 body throws a named refusal and whose T6 body is th32's; a `STORAGE` row never becomes a `SYCL_Host` weight, so the early and late plans agree (a C1-only load stops before the late plan). (b): `deferred_for_file` holds only `LAZY_READ` rows; `storage_for_file` and `excluded_for_file` are new, and the exclusion sites pass the union, whose `STORAGE` part is empty on any C1-only load that reaches `init_mappings`. The outputs with no consumer until T6 are named (the `STORAGE` value, the attributes, `mode_pre`, `storage_for_file`). Tests and mutants for both. | §3.1, §4.3, §4.6, §7 C1 |
| m4 `consumer_layer` literal 1 | Fixed: the one set bit of `hparams.is_ple_impl` (`src/models/qwen4exp.cpp:65-84`); a two-hparams test with the literal as its mutant. th32 §4.1 carries the same literal, **to relay to th32**. | §3.1, §7 C1 |
| m5 notify details | The gate text now says "an opaque backend handle plus scalars"; the `ggml_backend_is_sycl` check was done in rev 3d; the entry point returns `bool`, and `llama_context` WARNs on false. | §4.4, §7 C4 |
| m6 rerouted SCRATCH tenants unsized | Done in rev 3d (every tenant has a term or an `UNSIZED` ratchet entry). | §4.4 |
| m7 `host_output_bytes` re-derived | Done in rev 3d: the term is `output_reserve`'s own `new_size` (`src/llama-context.cpp:3535-3537`), which includes `embd_nextn`, `embd_layer_inp` and the backend-sampling buffers, notified at every realloc. | §4.4 |
| m8 `gs:42652`'s `in_model_load` branch | Fixed: its own site-table row, `MODEL`, owner the model's `ModelId`. | §4.4 |
| m9 Appendix B I-3's `W`; `gs:39959` and nl0f | Done in rev 3d. | Appendix B, §4.4 |
| n1 0edf outcome | Kept in rev 3e as rev 3d had it. **Relabelled in rev 3f** (§M61 (d)(3), r4 m-10): "a 5efe PASS plus a live 0edf defect", which keeps the two facts separate. | §7 C4 step 3 |
| §M58 (c) | The source-gate clause and the per-model line were done in rev 3d. Now stated in §3.1 too: C1 lands the real attribute values, and T6 only reads them. | §3.1 |
| zhcn GDC3 re-derived at `9650b3e` | Nothing here has copied a GDC3 figure since rev 3d; the cite names the row and `9650b3e` only (rev 3g removes the figures this row itself restated; r5 m-3). No arithmetic in this design takes the device compute buffer as an input: the Qwen room is shared zone minus device weights, and the step-5 zone-1 count, `ceil(M / 2048)`, reads `M` from the GDC3 point the arm scores against. **rev 3f** withdraws rev 3e's "the compute buffer shrinks when layers demote" (§M60 (a); r4 m-6), removes the copied figure (§M56 (a), §M61 (b); r4 m-5), and names the point (§M61 (a); r4 m-7). | header, §7 preamble, C4 step 5 |

**design-5efe-r4** (re-graded at `3763bb164..fb063e845`, 0C/3I/12m/1n; it
supersedes the draft written at `3e72abad5`), with addendum 11, §M60, §M61
(d) and §M62 (A)-(E), folded in rev 3f. Items the verdict marks
"POSTDATING" were ruled after rev 3e and are folded here like the rest.

| finding | disposition | where |
|---|---|---|
| I-A `get_max_size` finding names the wrong buft; device axis two-sourced | Fixed. `gs:42982` is `SYCL_CpuOffloadCompute`'s `get_max_size`; `SYCL_HostCompute`'s iface names the device buft's (`gs:37956`) and allocates host only in TP mode (`gs:42818`). The builder set is re-derived per buft in an ownership table (addendum 11 (2), composed with zhcn's §M56 (c)): `SYCL_Host` and `SYCL_KV_Host` take the host builder; `SYCL_HostCompute` gets its own `get_max_size` on the shared `ggml_sycl_host_compute_is_tp()`, delegating to `gs:37956` outside TP; every builder takes the device from the buft. `SYCL_CpuOffloadCompute` is out of scope, llama.cpp-g2a7 (§M62 (B)). The source gate pins the actual functions, with one mutant per buft. | §4.4 builders, §7 C4 source gate, tickets |
| I-B `DEVICE` release point is not module teardown | Fixed per §M62 (D). The doc states the true release point: `ggml_sycl_shutdown_global_runtime_pinned_owners` runs in every `ggml_backend_sycl_free` (`gs:84011`), temporary load backends included (`gs:83998-83999`), and in `ggml_backend_sycl_shutdown` (`gs:109890`). The rows are classified against the intended owner, the last live backend on the device; the hazard is llama.cpp-dsab, cited and not fixed here. 5efe's checks run at `free_buffer` and after `delete sycl_ctx` (`gs:84119`), and the `DEVICE` check only after the last backend on its device frees (a per-device count beside `g_sycl_backend_refcount`, `gs:84121`). This corrects §M61 (d)(1)'s premise. The zone-select test has a last-backend case and an every-free mutant. | §4.4 Lifetimes, site table, §7 C4 tests |
| I-D notify grows the notifying backend's device, not the host buft's | Fixed. `SYCL_Host` is one global buft whose `.device` is reg device 0 (`gs:42729-42730`), and `ggml_backend_sycl_device_get_host_buffer_type` ignores its device (`gs:108023-108026`). One helper, `ggml_sycl_host_buft_device(buft)`, is the device for `SYCL_Host`'s and `SYCL_KV_Host`'s `alloc_buffer` and `get_max_size` and for the notify, which resolves the host buft of the backend's device. The demand test pins `buft->device` (device 0 grows through a device-1 backend; a fake buft on device 1 moves it), with a `ctx->device` (`model.devices[0]`) mutant; the source gate forbids `ctx->device` on the path. | §4.4 notify, §7 C4 source gate, demand test |
| m-1 no VOID guard on default-context arms | Fixed per §M54, §M61 (c) and §M62 (C). The §7 preamble carries the guard, the padded regex `llama_context: n_ctx +=  *<N>$` (`lc:717`), not `^`-anchored, with the `n_ctx_seq` and `slot load_model` lines as the negative control, and names every arm it covers: C4 steps 3, 4 and 5's no-`-c`, C5's two arms, and C6's three. The Mistral gate adds `-v`. | §7 preamble, C4 steps 3-5, C5, C6 |
| m-2 `UNOWNED` MoE holders on the lead arms' path | Fixed by pre-registration under `GGML_SYCL_HOST_ZONE_ALLOC_TRACE=1`. Only the `routing_indices_cache` cohort can reach WEIGHT after the pre-size, within `n_expert_used x n_ubatch x 4` bytes (Qwen 20480, GPT-OSS 8192). `fused_handle` is unreachable (`ensure_fused_buf`, `gs:21626`, has no caller); the retained scratch needs `GGML_SYCL_CPU_RETAINED` (`gs:93922-93927`); `runtime_staging_cache` only stages mmap'd weights to a device (`gs:47945-47957`, `:24288`); the TP and split holders are off. Each `UNOWNED` line names its cohort. | §4.4, §7 C4 steps 2, 3, 5 |
| m-3 zone-3 "only through the notify" | Scoped to default flags: `SPLIT_MODE_TENSOR` reaches `SYCL_HostCompute`'s TP path (`lc:901-904`) and `GGML_SYCL_HOST_COMPUTE` the cpu-offload buft (`lc:928-931`); a run with either is VOID for the criterion. | §4.4 criteria |
| m-4 attention staging regrows at decode | Fixed per §M62 (E). The staging is planned in `sched_reserve` from the reserve graphs (built on `init_full`, `lc:2027`) through the scalar entry point `ggml_backend_sycl_plan_host_attn_staging`, and its slots grow there, before the first `graph_compute`. At dispatch a request above the plan is the named refusal `[ATTN-HOST] staging request ... exceeds the planned ...`. Criteria: zero `[HOST-POOL] grow_zone(` phase-gate WARNs (`pinned-pool.cpp:633-635`) and zero refusal lines on the Qwen and GPT-OSS no-`-c` arms, beside the `[ATTN-HOST] planned staging` positive control. New test `test-sycl-attn-host-staging-plan`. **Reworked in rev 3g** (§M69; r5 I-1, I-2, m-6): the plan follows `GGML_SYCL_ATTN_HOST_DISPATCH` and is not planned by default, its criteria move to the dispatch-on arms, the walk is SYCL-side through `ggml_backend_sycl_plan_host_holders(sched, graph)` with the dispatch's own predicate and per dispatching device, and a refusal fails the graph on the segmented paths. | §4.4 staging plan, criteria, §7 C4 |
| m-5 Appendix B copies GDC3's figure | Fixed: the number is removed from the zhcn GDC3 row above. | Appendix B |
| m-6 header restates the §M60 (a) premise | Fixed: the header states §M60's rule (compute size is a property of the vehicle's `n_outputs_max`, `lc:2051`), and the §7 preamble requires every pre-registered number to name its vehicle (§M60 (b)). | header, §7 preamble |
| m-7 GDC3 citations do not name the point | Fixed per §M61 (a): today's-tree point, the zhcn+moua fixpoint, or VOID when only one of the two is on the tree; each such arm also carries zhcn's plan == reality criterion (§M60 (c)). | §7 preamble, C4 step 5, C5, C6 |
| m-8 idle VRAM-RUNTIME read stays in host SCRATCH | Removed per addendum 11 (1) and §M61 (d)(2), and re-derived from run2's printed terms: SCRATCH 1732.0 MB, base 3579.2 MB (2 chunks), runtime need 91.9 MB (1 chunk), `Committed 6144.0 MB in 3 chunk(s)`, scored on the MB figures with a discriminator table. The setter-order pin is gone; the `runtime_bytes` drop stands on its own ground (both PP terms are VRAM allocations). | §0, §4.4, §7 C4 step 2 |
| m-9 no per-term trace | Fixed per §M62 (A): a trace table, one row per host SCRATCH term. `onednn_reorder_bytes` sizes the ONEDNN device zone (`uc:27543-27565`); `moe_q8_workspace_bytes` has no reader, and the real Q8_1 scratch is a device pool (`uc:21824`); `tp_vram_runtime_bytes` sizes device buffers, so all three leave the sum, and the `TP_TMP` `GRAPH` rows become `UNSIZED`. | §4.4 term trace |
| m-10 n1 label | Fixed per §M61 (d)(3): "a 5efe PASS plus a live 0edf defect", in C4 step 3 and the r3 n1 rows above. | §7 C4 step 3, Appendix B |
| m-11 device-side `idx_` accounting uncited | Fixed: llama.cpp-ka9c (addendum 11 (5)) is cited there and in the tickets list. | §4.4, tickets |
| m-12 the attention staging's per-call fallbacks listed as `DEVICE` | Fixed: `gs:82021` and `:82032` are env-only fallbacks (`gs:81967-81973`, `:81985-81988`), now their own `GRAPH` row, sized by the planned capacity of their kind. | §4.4 site table, `GRAPH` tenants |
| n-1 stale-`libllama` padding claim | Fixed: the guarantee relied on is that every in-tree construction site value-initializes (`= {}`); the stale-`libllama` pairing is stated as unsupported, not as zeroed. | §4.4 |
| 23mk rev 4.16 relay: `unified_cache_grow_host_scratch_zone` | One row, per the relay's two outcomes: **dead code, listed for deletion**. Its only caller is `ggml_backend_sycl_notify_compute_buffer_sizes` (`gs:19145`), which 23mk's step 7 deletes; C4's notify uses the ensure-capacity call instead. Whichever of C4 and step 7 lands second deletes it (`uc:21935`, `uch:6907`) and C4 rewrites the `uc:27716` comment; the source gate fails on any caller. **Superseded in rev 3g by §M68 (a)** (r5 m-5): C4 alone deletes it, removing the `gs:19145` call itself, and the gate asserts zero callers and zero definitions in both tree states. | §4.4 notify, §7 C4 source gate |
| §M60 (a)-(d) | (a) premise withdrawn (m-6); (b) vehicles named (§7 preamble); (c) plan == reality on every GDC3 arm (m-7); (d) the device compute buffer routes to RUNTIME, and nothing here charges it to SCRATCH. | header, §7 preamble |
| §M62 (A)-(E) | (A) the term trace (m-9); (B) llama.cpp-g2a7 cited (I-A); (C) the anchored guard (m-1); (D) the release point (I-B); (E) the staging plan (m-4). | as the rows above |

Closed at `fb063e845`, per the verdict's section A (doc lines are
`fb063e845`'s):

| finding | state at `fb063e845` | fb063e845 doc lines |
|---|---|---|
| r3 I-1 (b), (c); earlier-r4 I-C | fixed in 3e: ensure-capacity high-water, target = plan SCRATCH + live demands; additive grow not called; test and mutant | 1285-1299, 2726-2733 |
| r3 I-2 | fixed in 3e: `zone=KV cohort=kv_host_buffer` summing to 6912.0 MiB; the §M57 (a) two-commit RED | 2948-2952, 2976-2983 |
| r3 I-3; earlier-r4 I-3r; addendum 11 (3) | fixed in 3e: `free_buffer` check, after-`delete sycl_ctx` check, `MODEL` branch, destruction-order test | 1324-1355, 1446-1452, 2703-2712 |
| r3 I-4; earlier-r4 m-R8 | fixed in 3e: `DEVICE` rows, 11 holders (12 lines) `UNOWNED`, the zero-refusal WARN criterion | 1562-1565, 2838-2841 |
| r3 I-5 | fixed in 3e as then scoped; re-derived in rev 3f (m-8) | 1537-1545 |
| r3 m3 | fixed in 3e: `LAZY_READ`-only `deferred_for_file`, `storage_for_file`/`excluded_for_file`, the `STORAGE` branch | 617-630, 659-676, 729-736, 751-765, 2379-2388 |
| r3 m4 | fixed in 3e: `consumer_layer` from `is_ple_impl`, two-hparams test | 543-550, 2389-2391 |
| r3 m5 | fixed in 3e: returns `bool`, `llama_context` WARNs | 1208-1236, 2661-2664 |
| r3 m8 | fixed in 3e: own `MODEL` row | 1447 |
| §M58 (c); §M56 (a) body; earlier-r4 m-R7, m-R10 | fixed in 3e (§M56 (a)'s Appendix B residue is m-5 above) | as the verdict cites |

**design-5efe-r5** (`fb063e845..b3c08be71`, 0C/3I/10m/2n; every r4 item
closed as claimed; I-3 was added to the verdict after rev 3g), with lead
rulings §M63 (a), §M66, §M68 (a) and §M69 (a)-(d), folded in rev 3g, and
I-3 in rev 3h:

| finding | disposition | where |
|---|---|---|
| I-1 the staging plan reserves for a dispatch that is off by default | Fixed per §M69 (a). The host dispatch is opt-in (`GGML_SYCL_ATTN_HOST_DISPATCH`, `gs:81901-81917`, tested at `gs:83014-83016`, `:83022-83024`), and by default demoted attention runs on the scheduler's CPU split (`gs:81896-81900`). With the switch off the plan registers no demand, grows no slot, and prints `[ATTN-HOST] staging not planned (host dispatch off)` once. The default arms (C4 steps 2, 3, 5) assert that line and zero staging demand. The staging criteria move to two dispatch-on arms, 5b (GPT-OSS B50 no-`-c`) and 3b (Qwen no-`-c`, VOID until th32 T6/T7 and C4 are on the tree), whose positive control is C4's new dispatch counter with `fa` > 0. The "on every no-`-c` arm's path" claim is corrected; the `DEVICE` classification stands. | §4.4 staging plan, addendum 10 (b) paragraph, criteria; §7 C4 steps 2, 3, 3b, 5, 5b |
| I-2 (1) the refusal is discarded on the MoE segmented paths | Fixed per §M69 (b). C4 routes the four `ggml_sycl_compute_forward` calls at `gs:97234`, `:97285`, `:97551` and `:97595` through one fail-closed helper that prints the ordinary loop's unsupported-node line and asserts (`gs:94499-94506`). RED `test-sycl-seg-forward-return-source` fails at master naming the four sites, with a mutant per site and a positive-control fixture. The fix is tracked as llama.cpp-573t (filed by the lead). The r6 amendment of §M69 (b) extends it to six sites; see the design-5efe-r6 table (m-1). | §4.4 staging plan, §7 C4 tests, tickets |
| I-2 (2) the dsab release zeroes a live context's planned slots | Stated per §M69 (c): with dsab, the slots are released only when the last backend on the device is freed, by C4's per-device live-backend count, and dsab's fix adopts that count for the pools and the staging. Until dsab lands the premature release reaches the plan only as a fail-closed refusal; a dispatch-on arm that prints `planned 0 B` on a tree without dsab is VOID and reported as dsab's. Tightened in rev 3i: only the ordered sequence of r6 I-2 (2) makes a refusal dsab's. | §4.4 Lifetimes, site table, §7 C4 step 5b |
| I-2 (3) slots planned on the helper's device | Fixed per §M69 (d): the walk takes each node's device from the scheduler's assigned backend, the device whose `ctx.device` indexes the slots (`gs:82358`, `:82844`); demand and `req.device` are that device's. Test case with two devices, and a helper-device mutant. | §4.4 staging plan, §7 C4 tests |
| I-3 (§M66) trace-scored arms neither set the variable nor check it is live | Fixed in rev 3h. `GGML_SYCL_HOST_ZONE_ALLOC_TRACE=1` is in the literal command of C4 steps 2, 3, 3b, 5's no-`-c` arm, 5b and the §M57 (a) RED, which rides step 5. Each names its trace-live check: step 2 a post-pre-size `zone=SCRATCH cohort=backend_host_buffer` line (at `-c 4096` nothing demotes); step 3 the KV exact sum of 7247757312 bytes, named as the validity check for its WEIGHT criterion; step 5 its `kv_host_buffer` and `backend_host_buffer` lines; the RED any post-pre-size `[HOST-ZONE-ALLOC]` line. A non-live arm is VOID, which counts as failed. C5 and C6 rerun the arms with the same commands and checks. | §7 preamble, C4 steps 2, 3, 3b, 5, 5b, RED; C5; C6 |
| m-1 C1's Mistral default arm missing from the VOID-guard list | Fixed: C1's lead gate runs with `-v` and the guard, and is on the list; W1-W3 are named as not default-context arms, since each is killed before a `llama_context` exists. | §7 preamble, C1 |
| m-2 GPT-OSS vehicle not fully named | Fixed: `llama-cli`, default `n_ubatch` 512, `n_outputs_max` 1 (`tools/server/server-context.cpp:1015`, `lc:2051`), for step 5 and 5b, C5 and C6. | §7 preamble |
| m-3 the GDC3 row restates GDC3 figures | Fixed: the row names the GDC3 row and `9650b3e` only. | Appendix B |
| m-4 §M63 (a) not folded | Fixed: on a zhcn tree the GDC3 point is its replay applied to the run's own printed inputs; step 5's N, range, `M` and `ceil(M / 2048)` come from that replay of the same run. | §7 preamble, C4 step 5 |
| m-5 the 23mk-relay gate is not mechanical | Fixed per §M68 (a): C4 alone deletes `unified_cache_grow_host_scratch_zone`, including the `gs:19145` call; `ggml_backend_sycl_notify_compute_buffer_sizes` stays 23mk step 7's. The gate counts zero call sites and zero definitions or declarations in both tree states (step 7 not landed, and landed), with a left-call mutant and a left-definition mutant. | §4.4 notify, §7 C4 source gate |
| m-6 the plan predicate is a second source and reads a `NULL` buffer | Fixed: the walk uses `ggml_sycl_tensor_is_in_kv_host_buft` (`gs:108235-108241`), the interception's own predicate, which resolves `view_src` (a view's `buffer` is `NULL` at creation, `ggml.c:1817`). The test adds a view-K case that counts and CPU and `SYCL_Host` K cases that do not, with an `is_host` mutant. | §4.4 staging plan, §7 C4 tests |
| m-7 the `DEVICE` pools' demand arrives inside PP | Fixed: the pools' dimension scan (`gs:7952-8030`) becomes one function shared by `moe_hybrid_init_once` and the `sched_reserve` plan walk, which registers the pools' demand before the phase is set (`gs:105352`). A warmup graph stays under the planned dispatch count for both lead models; an allocation above plan prints a named WARN, which the criteria count. In rev 3i the pools are also sized from the plan, and rev 3g's claim that the two agree is withdrawn (r6 I-3). | §4.4 Sizing, criteria, §7 C4 |
| m-8 cite errors | Fixed: `ggml_sycl_attn_host_staging_get` is `gs:81975-82074`; `props->caps` is `gs:108004-108009`; `uc:21824`; `lc:541`. | §2, §4.4, §7 preamble, Appendix B |
| m-9 the Qwen correctness criterion cannot fail | Fixed: stdout is captured on its own and scored `grep -c 'Paris'` ≥ 1; the lead records the parent commit's continuation, and a difference with `Paris` present is reported. Superseded in rev 3i by r6 m-3: the first word is anchored, non-degeneracy is checked, and the reference is C4's own continuation, since the parent cannot load Qwen. | §7 C4 step 2 (and 3, 3b) |
| m-10 the merge unit | Fixed: 5efe C1-C4 and th32 T6/T7 reach master in one merge; "VOID without T6/T7" is a branch state that master never holds. | §3.1 |
| n-1 the IDLE branch still grows | Fixed: it takes the same refusal; its per-call buffer stays the `GRAPH` fallback; a gate mutant restores its grow. | §4.4 staging plan, §7 C4 source gate |
| n-2 the recipe term closed by subtraction | Fixed: 0 because only `Q1_0`/`NVFP4` experts get a recipe workspace (`uc:27380-27399`) and Qwen's are `Q8_0` (`run2.log:81`). | §4.4 |
| §M66 env-gated trace lines | Folded in rev 3g, made literal and per-arm in rev 3h (I-3 above): the one such emitter here, `[HOST-ZONE-ALLOC]` (`GGML_SYCL_HOST_ZONE_ALLOC_TRACE=1`), is scored at five points. Each sets the variable on its own command line and must be trace-live, meaning at least one `[HOST-ZONE-ALLOC]` line of any cohort in the same run, or it is VOID, which counts as failed. The §M57 (a) RED, which scores a line's presence, is VOID when not live. | §7 preamble, C4 steps 2, 3, 5, RED |
| §M68 (a) one owner per deletion | Folded (see m-5). | §4.4 notify, §7 C4 source gate |
| §M69 (a)-(d) | (a) I-1; (b) I-2 (1); (c) I-2 (2); (d) I-2 (3). | as the rows above |
| r5 H notes N1-N3 (tracker hygiene) | Not doc changes: llama.cpp-30r1's DEVICE wording, g2a7's env-var name, and dsab's cites and test switch were relayed to the lead, and the lead has done them. | - |

**design-5efe-r6** (`b3c08be71..87181659f`, 0C/3I/5m/4n; it did not see
rev 3h, `e1b6d2750`), with the lead's r6 rulings and the §M69 (b)
amendment, folded in rev 3i:

| finding | disposition | where |
|---|---|---|
| I-1 the plan walk runs after the scheduler has reset | Fixed per the ruling. `ggml_backend_sched_reserve` resets at its end (`gb:2703`), so after the allocating reserves `ggml_backend_sched_get_tensor_backend` returns `NULL` for every node (`gb:2665-2675`, `:2813-2820`). The walk now runs right after C4's size-only `reserve_size` calls, which leave the split live (`gb:2677-2689`), and records a pending plan; `ggml_backend_sycl_commit_host_holders` promotes it only after the allocating reserves succeed. The `no_alloc` pass is skipped. A walk that attributes zero host-KV nodes with the switch on WARNs, and a dispatch-on arm that prints it FAILs. New RED `test-sycl-plan-walk-sched` on the real ggml-base scheduler over CPU backends; source-gate placement and commit-order mutants. `ggml_backend_sycl_set_sched_placement_plan` is cited as a precedent for argument types only (direct call, no-op body, `gs:15847-15854`). | §4.4 staging plan, §7 C4 tests and source gate |
| I-2 (1) `fa` undefined, counter can miss the abort | Fixed: `fa` counts interceptions, incremented before `ggml_sycl_attn_host_staging_get`; the first interception per device prints a `first host dispatch` WARN; the fail-closed helper prints the counter line before it asserts; a score reads the last line per device. | §4.4 staging plan, criteria, §7 C4 staging-plan test In rev 3j the helper propagates `GGML_STATUS_FAILED` instead of asserting (§M69 (b) second amendment), and prints the counter line before it returns. rev 3l replaces the return with a throw of `ggml_sycl_fallback_error` (third amendment, design-5efe-r8 table). |
| I-2 (2) the §M69 (c) VOID is an escape hatch | Fixed: `planned 0 B` alone no longer makes a refusal dsab's. Any refusal or abort is C4's FAIL unless stderr shows, in order, a plan line with a non-zero total, a counter line (a SYCL backend free) and the refusal. Lifetimes and 5b say so; single-context 3b and 5b cannot show that sequence. | §4.4 Lifetimes, criteria, §7 C4 steps 3b, 5b |
| I-2 (3) no precedence between VOID and FAIL | Fixed: five ordered rules; the only VOID is `fa` = `set_rows` = `refusals` = 0 with no abort, and `fa` = 0 with `set_rows` > 0 is a FAIL. | §4.4 criteria |
| I-3 the pools are sized from the warmup, not the plan | Fixed per the ruling (P4, one source): `moe_hybrid_init_once` initializes both pools from the device's committed (D, K, N), and its own scan only checks, with an `exceeds its planned` WARN and an `unplanned` WARN. Pool demand is planned only for a host-resident `MUL_MAT_ID` `src0`. rev 3g's claim that the two agree is deleted. Pre-registered: Qwen 100.0 + 400.0 MiB with the matching `[MOE-POOL]` and `[CPU-EXPERT-POOL]` lines; GPT-OSS 0, or 45.0 + 180.0 MiB, by the run's placement. A pure RED plans at 8 tokens and feeds 16, so the over-plan WARN fires. | §4.4 sizing, criteria, §7 C4 staging-plan test, source gate, step 2 |
| m-1 six discard sites, not four | Fixed per the §M69 (b) amendment: the `moe_after` calls at `gs:97240` and `:97315` (e2461d4fb; `:97250`, `:97325` at d8a67422d) join the four. The helper text, the source gate, the RED's predicted names (by function and role as well as line), its six mutants, a six-call count check and the tickets paragraph all say six. llama.cpp-573t carries the two (comment c-du6m). | §4.4 staging plan, §7 C4 tests, tickets |
| m-2 step 2's command lacks the trace variable | Already closed in rev 3h: `e1b6d2750` doc lines 3371-3374 carry `env GGML_SYCL_HOST_ZONE_ALLOC_TRACE=1` in step 2's literal command, and 3, 3b, 5's no-`-c` arm, 5b and the RED have literal forms too. | §7 C4 step 2 |
| m-3 the answer criterion is weak and its reference cannot run | Fixed: the first word is anchored on the echoed prompt's line, a no-four-repeats check with a positive and a negative control is added, and the reference is C4's own recorded continuation (the parent cannot load Qwen, rc 137), compared in C5 and C6, recorded and not gated. | §7 C4 step 2, 3b, C5, C6, acceptance summary |
| m-4 3b and 5b are new vehicles | Fixed: each is its own vehicle row (§M60 (b)) since the switch changes the split (`gs:108273-108330`); 5b's GDC3 point is the §M63 (a) replay of its own inputs; 5b gets a parent-tree control, and 3b states it has none. The host dispatch's own "cannot itself be GPU-verified" (`gs:81855-81862`) is cited. | §7 C4 steps 3b, 5b |
| m-5 the staging high-water is over every candidate | Fixed: a commit replaces its context's plan, the capacity is the high-water over live contexts' committed plans, a slot above it is re-allocated down at the commit, and the settled candidate commits last (`lc:1931-1940`). Test and mutant added. | §4.4 staging plan, §7 C4 staging-plan test |
| n-1 the rev 3g header sentence | Fixed: it reads 0C/3I/10m/2n, with the I-3 added after rev 3g; the I-3 row is keyed in the r5 table. | status header |
| n-2 the seg RED's label | Fixed: `sycl;host;source-contract`. | §7 C4 tests |
| n-3 how llama reaches the entry points | Fixed: both are proc-table entries (`gs:110040`) looked up by name, a `NULL` lookup or a `false` return WARNs and continues, and the SYCL side retains neither pointer. | §4.4 staging plan |
| n-4 the default arms' absence has no INFO control | Fixed: scored only beside a `[HOST-ARENA] SCRATCH demand ctx=` line of the same run, else VOID. | §4.4 criteria |
| r6-prep: 573t | §M69 (b) is cited as llama.cpp-573t in §4.4, the tickets paragraph and the r5 I-2 (1) row; no ticket was created. | §4.4, §7 tickets, this appendix |
| r6-prep: r5 N1-N3 | The r5 row now reads that the lead has done them. | this appendix |

**design-5efe-r7** (`87181659f..b29dd42d2`, 0C/2I/7m/6n), with the lead's
r7 rulings, the §M69 (b) second amendment, §M73 and §M74, folded in rev
3j:

| finding | disposition | where |
|---|---|---|
| I-1 the pending plan has no lifecycle | Fixed as the reviewer wrote it. A third proc-table entry, `ggml_backend_sycl_begin_host_holders(sched)`, clears every SYCL backend's pending plan at the start of each `sched_reserve` (after `lc:2002`) and before the pipeline-parallel re-measure, enumerating with `ggml_backend_sched_get_n_backends`/`get_backend` (`ggml-backend.h:362-363`). The begin runs right after each scheduler creation (`lc:2019-2022`, and the retry's `lc:2061-2064`; moved there from `lc:2002` in rev 3k for §M75). One reserve's pp and tg walks max-merge per (device, kind) and per device for (D, K, N); the pp graph dominates (`lc:2007` against `lc:2080`), so Qwen plans D = 5120, not 10. The commit consumes the pending plan and is the only step that registers demand. Tests: "tg replaces pp" and "stale pending from a failed candidate is promoted" mutants, each with its predicted message. | §4.4 sizing and staging plan, §7 C4 tests and source gate |
| I-2 the pools gate on the buffer | Fixed as a P3 fix: the gate is placement's executor fact, `ggml_sycl_moe_plan_has_host_experts` (`gs:26003-26025`), the fact the dispatch's CPU-island gate routes on (`gs:26031-26051`), never `ggml_backend_buffer_is_host`, since under C4 `SYCL_Mapped` is `is_host` for device-placed experts too. A new INFO line, `DEVICE pools dev=%d committed D=... host_expert_nodes=%d`, is the positive control for the zero case, else VOID. Test: a `SYCL_Mapped` `src0` with every expert device-placed registers zero pool demand; mutant: the buffer gate. GPT-OSS's case is read from that line, one source. | §4.4 sizing, criteria, §7 C4 tests and source gate |
| m-1 a losing candidate commits | Fixed: `try_candidate` commits only after its host-fallback check (`lc:1622-1629`); other callers commit at `sched_reserve`'s end. A losing candidate grows no slot, re-sizes no pool and ensures no SCRATCH for either. No exception is stated. The notify's own pre-allocation ensure for a losing candidate was raised with the lead and is ruled by §M75 (row below). | §4.4 staging plan, §7 C4 source gate |
| m-2 rule 4 can absorb a crash | Fixed: refusal, failed graph, abort (rc 134 plus the printed abort message) and crash (any other non-zero rc or signal: rc 139, rc 124, the watchdog's `_Exit(1)`, a GPU fault) are defined; an abort or crash is FAILED, never VOID. Rule 2 requires a counter line after the plan line; rule 4 also requires no `first host dispatch` line and rc 0. | §4.4 criteria |
| m-3 5b's parent control has no place in the rules | Fixed: it is rule 1's exception (b), for an abort or crash whose abort or GPU-fault line also prints on the parent; a refusal or failed graph is never pre-existing. | §4.4 criteria, §7 C4 step 5b |
| m-4 the size-only split is not the allocating split; the RED fixture | Fixed. (a) The start reset erases the `norm`/`l_last` pins (`lc:3985-3997`) and the `!offload_kqv` pin (`src/llama-graph.cpp:2779`), which `ggml_backend_sched_reserve` keeps (`gb:2691-2698`); the walk's three op kinds are assigned by buffer or expansion and unaffected, and the claim is bounded to them. The notify re-notifies at commit with the allocated size, so the ledger holds the fact, and the demand line prints both figures. (b) The RED takes an injected `is_target` predicate and reaches the stand-in through the first pass (`view_src` in a CPU buffer, stand-in at index 0), never `set_tensor_backend`. | §4.4 staging plan, §7 C4 tests |
| m-5 no multi-context rule for the pools | Fixed: max-merged, the device high-water over live contexts' commits; the commit re-sizes initialized pools to it (`resized D=a->b`, INFO) in `sched_reserve`, which is quiescent; a context's plan drops at its backend destructor; release at the last backend is dsab's. Two-context test with an init-once mutant. | §4.4 sizing, §7 C4 tests |
| m-6 §M73 not applied | Fixed: the three GDC3 places cite zhcn's compute criterion in its §M73 (b) form and the replay's `compute_ctx` charge (§M73 (c)), cited, not restated. | sibling designs (zhcn), §7 preamble, C4 step 5 |
| m-7 the Paris anchor is fail-open | Fixed: the scorers are one-line commands in a fenced block, and the anchor has its own controls, Paris 1, Lyon 0 and the prompt echo alone 0. All six controls were run on this host and print the stated values. | §7 C4 step 2 |
| n-1 sha mixing | Fixed: all six sites at both shas. | §7 C4 tests |
| n-2 rule 1 keyed on the total | Fixed: keyed on the refusing kind's figure. | §4.4 criteria |
| n-3 the th32 pin | Re-pinned at rev 3d `c1903496f`; every cited th32 section keeps its number at 3d, and the expected-line cite is corrected to §10 (it was already §10 at 2x). | sibling designs |
| n-4 the synchronize cite | Fixed: `lc:2002`. | §4.4 staging plan |
| n-5 rule 3 presumes FA on | Fixed: rule 3 reads `Flash Attention enabled` (`lc:1281`); with `set to disabled` (`lc:1271`), or neither line, the staging criteria are VOID. | §4.4 criteria |
| n-6 word-level degeneracy only | Fixed: a trigram check (no three-word sequence three times) with a control, and its limit stated. The repeat checks use `awk`, because this host's `grep` (`ugrep`) rejects back-references. | §7 C4 step 2 |
| §M69 (b) second amendment | Folded: the six sites propagate `GGML_STATUS_FAILED` through an out-parameter, and `graph_compute` returns it before the recording fallback, which would re-dispatch nodes (`gs:107206-107229`); llama's existing path logs `lc:3968-3970` and returns -3 (`lc:3290-3294`). No assert or abort; the test and the gate check it. **Superseded in rev 3l by the third amendment**: the return left the recording state live and skipped `graph_compute`'s cleanup, so the helper throws instead (design-5efe-r8 table, I-1). | §4.4 staging plan, criteria, §7 C4 tests and source gate |
| §M74 | Folded: every line C4 adds and a vehicle arm scores is always compiled in ggml-sycl; each is a dedicated line §M74 (f) accepts; the `[ATTN-HOST]` dispatch lines are also switch-gated, and 3b/5b set the switch. | §4.4 staging plan |
| lead check: the default arms' witness | Folded: `staging not planned` is required exactly once on each default arm, absent is VOID, and every scored `[ATTN-HOST]` absence is scored only beside it. | §4.4 criteria |
| §M75 a losing candidate's SCRATCH growth | Folded in rev 3k. The growth is charged to the candidate through `ggml_backend_sycl_host_scratch_trial` (`OPEN`, `KEEP`, `RETURN`), kept only on a win (`sched_reserve` and the host-fallback check), and returned on a loss. The release runs in the begin entry, after the next scheduler creation frees the loser's buffers, and before the next notify. The zone could not shrink: C4 adds `pinned_chunk_pool::release_zone_chunk` with a tombstone so no chunk index moves, releasing through the chunk's `mem_handle`, and refusing by name over a live allocation. RED `test-sycl-scratch-trial-return` on a non-MoE ladder, with the keep-the-high-water mutant and its predicted message; Mistral's step 4 carries the lead-arm criterion. | §4.4 staging plan, §7 C4 tests, source gate, step 4 |

**design-5efe-r8** (`b29dd42d2..8b798c0e6`, 0C/1I/4m/7n), with the lead's
r8 rulings and the §M69 (b) third amendment, folded in rev 3l:

| finding | disposition | where |
|---|---|---|
| I-1 the immediate return leaks and dangles (§M69 (b) third amendment) | Fixed: the helper prints the counter and unsupported-node lines, then throws `ggml_sycl_fallback_error` (`common.hpp:94`). The recording catch rethrows it (`gs:97347-97349`) and the replay catch rethrows everything (`gs:97588-97590`), so the recording fallback never runs; `graph_compute`'s catch (`gs:107748-107779`) quarantines, drains, unpins the transient leases, releases the execution graph, resets the recording globals, prints `recoverable runtime fallback failed` (`gs:107778`) and returns `GGML_STATUS_FAILED`. An immediate return is forbidden. th32 3e §5.3's lazy cancel step runs first in that catch, so a parked lazy copy is cancelled or drained before the catch waits. The source gate requires the throw at all six sites; new RED `test-sycl-seg-refusal-cleanup` (direct and recording cases: status, recording globals, sink, transient leases at baseline, a second compute succeeds) with the immediate-return mutant and its two predicted messages. The failed-graph term now names the `gs:107778` line. The lead's `gs:107777` is off by one at `e2461d4fb`; the line is `:107778`. | §4.4 staging plan, criteria, §7 C4 tests and source gate |
| m-1 the cache-path window | Fixed: `RETURN` withdraws the marked chunks from zone allocation (alloc skips them, capacity drops them, frees still land) and files them in a per-device marked set that no `OPEN` touches, closing the cache path's window (`lc:1750` to the ladder's probe and publish, `lc:1516`, `:1573`) and the settle's publish (`lc:1936`). New RED case with a 1 MiB publish allocation between `RETURN` and the begin; mutants: marks kept per transaction, and no withdrawal. | §4.4 staging plan, §7 C4 tests and source gate |
| m-2 `<C>` and the release-at-RETURN mutant | Fixed: `<C>` is about 150 MiB (64 + 86; the ensure is a high-water), not 214. The fixture holds an 80 MiB stand-in for the loser's host compute buffer, live at `RETURN`, in the grown chunk only, freed before the begin, so the release-at-`RETURN` mutant fails on `SCRATCH return refused`. | §7 C4 tests |
| m-3 step 4 can pass with no loss | Fixed: the §M75 sub-criterion counts only with at least one `SCRATCH trial ... returned` line, else VOID, and the stop reason is read from the `lc:1942-1943` WARN; the command pins `GGML_SYCL_TUNING_CACHE=0`. | §7 C4 step 4 |
| m-4 the trigram gate fails a correct answer | Fixed: the trigram figure is recorded, not gated. The enumeration `The capital of France is Paris. The capital of Germany is Berlin. ...` is a must-pass control for the two gated checks (anchor 1, word run 1; its trigram figure is 4). All eight controls were run on this host and print 1 0 0 4 1 3 1 1. | §7 C4 step 2 |
| n-1 the gate's begin placement | Fixed: right after each scheduler creation (`lc:2019-2022`, `lc:2061-2064`); a begin before the creation fails the gate. | §7 C4 source gate |
| n-2 the fixture's top-K | Fixed: top-10, `n_seq_max` 1, pp 512, so D = 5120 against 10. | §7 C4 tests |
| n-3 tg `n_tokens` | Fixed: `n_seqs` = `n_seq_max` (`lc:2080`), 1 on every lead arm. | §4.4 staging plan, §7 C4 tests |
| n-4 `release_zone_chunk` gaps | Fixed: subtracts the backing size (`chunk_footprint_for`, `pp:798-803`) from `total_allocated_` (superseded in rev 3r and rev 3s: the footprint is subtracted by the released carrier's destructor after the physical free, design-5efe-r14 M-3, design-5efe-r15 I-1); requires the per-chunk `allocator` (`pinned-pool.hpp:246`) unused; `configure_zones` (`pp:361-418`) skips tombstones; the `pinned-pool.hpp:112` comment is rewritten. | §4.4 staging plan |
| n-5 one cause of `known == false` | Fixed: the WARN names both causes (`gs:26007-26024`), usage not `MOE_EXPERT_WEIGHT` or no placement plan. | §4.4 sizing |
| n-6 the pipeline-parallel retry's growth | Fixed: `sched_reserve` opens its own transaction when no trial is open, `KEEP`s on success, and `RETURN`s before the retry (`lc:2055-2065`), whose scheduler creation (`lc:2061`) frees the buffers and whose begin releases. | §4.4 staging plan, §7 C4 source gate |
| n-7 §M74 list | Fixed: the §M75 lines (`trial ... kept` or `returned`, `released`, `return refused`, `DEVICE pools ... resized`) are in the always-compiled list. | §4.4 staging plan |
| merge unit: th32 pin | Re-pinned at rev 3e `0426382cc`; 3e changes no heading from 3d, and the §10 expected line exists. §5.3's cancel step is cited for I-1. | sibling designs |

**design-5efe-r9** (`8b798c0e6..a68508606`, 0C/0I/5m/8n, P4 FAIL), with
the lead's r9 rulings, the queued tracker-edge correction and the th32
re-pin, folded in rev 3m:

| finding | disposition | where |
|---|---|---|
| m-1 the seg-refusal RED's order, reach and observable | Fixed. (a) The immediate-return mutant fails first on the `gs:107778` line, and the test asserts its message quotes it; the other failures are listed for the record. (b) The recording case requires the `gs:97343-97345` WARN naming the seam's node, else fixture error. (c) The observable is the context's `graph_weight_leases` and `graph_moe_expert_leases` (`common.hpp:7222-7223`), snapshotted by the seam at the refusal; at least one lease pinned by the refused call is a precondition. The unpin is a no-op while `exec_graph` or `active_exec_graph` is live (`gs:61136-61138`), and nothing in the catch clears them before `:107768` (the quarantine and the execution release do not), so C4 amends the catch to call `sycl_exec_graph_clear_active` (`gs:99517-99567`) after its queue waits and before the unpin. A third case with a live cached `exec_graph` pins that clear. Position and third case superseded in rev 3n (design-5efe-r10 table, m-1, m-2). | §4.4 staging plan, §7 C4 tests and source gate |
| m-2 the window mutants cannot fire | Fixed: the base is filled before the 1 MiB request, so without withdrawal it can only land in the marked chunk; `zone_capacity` is asserted right after `RETURN`; the marks-reset mutant is scored on `chunk_count` and the missing `released` line. Every case that checks capacity also checks `chunk_count` or `allocated()`. | §7 C4 tests |
| m-3 two decrement sites | Fixed: withdrawal is one per-chunk flag and the one decrement of `zones_[zi].size`; `release_zone_chunk` requires it and does not decrement again; `zone_used`, the segmented early check, `zone_alloc`, `zone_alloc_segmented` and `zone_largest_free_block` all exclude withdrawn chunks. The RED computes sums only and asserts capacity never exceeds `allocated()`, with a double-decrement mutant. | §4.4 staging plan, §7 C4 tests and source gate |
| m-4 a zero-chunk `returned` line | Fixed: at zero chunks `kept`, `returned` and `released` print nothing, and the own transaction's `KEEP` prints only on growth. Step 4 counts only a `returned` line with at least 1 chunk, else VOID, lists all four pre-`OPEN` losses (`lc:1530-1534` added), and states that the sub-criterion may VOID on Mistral, with the host RED as §M75's binding evidence. | §4.4 staging plan, §7 C4 step 4 |
| m-5 withdrawn chunks with no following begin | Fixed: each marked-set entry records its owning backend and is released at the next begin or at the owner's end, whichever comes first. The owner's end is `ggml_backend_sycl_free` (`gs:83943`) plus the free that empties an orphaned chunk, because `llama_context` destroys `backends` before `sched` (`src/llama-context.h:398`, `:403`), in `~llama_context` and after a constructor throw (`lc:2068`, `:2082`) alike. New RED case with a no-owner-end-release mutant. | §4.4 staging plan, §7 C4 tests and source gate |
| n-1 the replay catch | Fixed: the replay sites are outside any try; the `gs:97588-97590` catch (try at `gs:97583`) wraps only the descriptor replay. | §4.4 staging plan |
| n-2 the failed-graph term | Fixed: defined by llama's `lc:3968-3970` line alone; the `gs:107778` line attributes it to the refusal channel. | §4.4 criteria |
| n-3 the exception's message lifetime | Fixed: the helper throws a string literal, since the type stores the pointer (`common.hpp:94-100`); the source gate pins it. | §4.4 staging plan, §7 C4 source gate |
| n-4 tombstone readers | Fixed: every loop over `chunks_` skips a null base, `pre_allocate` (`pp:228-230`) and `allocate_segmented` (`pp:176-181`) named; the release resets the per-chunk allocator. | §4.4 staging plan |
| n-5 the counter dump's name | Fixed: the §M74 (g)/(i) `[SYCL-COUNTER]` lines and `end devices=%d`. | §4.4 staging plan |
| n-6 the pipeline-parallel retry | Fixed: `MARK` before the measure and `RETURN_TO_MARK` before the retry, so the retry runs inside a transaction (its own, `KEEP`t on success, or the trial's) and the discarded first measure's growth is returned in both cases. | §4.4 staging plan, §7 C4 source gate |
| n-7 the second-context stand-in | Fixed: sized per case, a 58 MiB stand-in in a 64 MiB chunk with the other chunks filled below it; the ladder ends at 192 MiB, and the keep mutant reads 256. | §7 C4 tests |
| n-8 prose | Fixed. | §4.4 staging plan |
| ungraded: the executable-graph destructor | Recorded as a known ordering item shared with today's `return false` path: the headers show a `shared_ptr` handle (`executable_graph.hpp:82`); whether the runtime's destructor waits is unverified, and C4's implementer checks it before landing. **Superseded in rev 3n**: the tree answers it (`gs:99522-99538`), and the §M69 (b) fourth amendment moves the executables out of the unwind (design-5efe-r10 table, I-1). | §4.4 staging plan |
| queued: the tracker-edge sentence | Fixed: no tracker edge blocks C1's merge on T6 (it would close a cycle); the one-merge constraint is comments `c-q50c` and `c-0elm`, as th32's T6 row states. The two Appendix B rows that repeated the claim are corrected in place. | the lead's C1/T6 ruling, this appendix |
| merge unit: th32 pin | Re-pinned at rev 3f `d0ad45bb7`; 3f changes no heading from 3e, §5.3's cancel step and the §10 expected line are unchanged. | sibling designs |

**design-5efe-r10** (`a68508606..6b674097e`, 0C/1I/5m/8n, P2 and P4
FAIL), with the lead's r10 rulings and the §M69 (b) fourth amendment,
folded in rev 3n:

| finding | disposition | where |
|---|---|---|
| I-1 the unwind destroys submitted executables (§M69 (b) fourth amendment) | Fixed: `moe_graph_record_segments` wraps its record loop in a `catch (...)` that moves `recorded_segments` and `recorded_moe_graphs` (`gs:97159-97160`) into the context before rethrowing, from all four sites (direct `gs:97234`, `:97240`; recording `gs:97285`; moe_after `gs:97315`). The catch's cleanup step destroys them after th32's cancel step and its own waits. No wait at the throw site. The evidence is `gs:99522-99538` (dkw0 #4), which replaces "unverified". New fourth seg-refusal case at the moe_after site, with the local-unwind mutant. The inner catch's `return false` exits (`gs:97333`, `:97350`) have the same shape; they are pre-existing and tracked as llama.cpp-7l45 (rev 3o, design-5efe-r11 n-3). The premise that the context vectors are empty at the handler is enforced by an entry guard in rev 3o (design-5efe-r11 m-1). | §4.4 staging plan, §7 C4 tests and source gate |
| m-1 the clear's position, wrapping and body | Fixed: one step immediately after `gs:107767` and before `:107768`, wrapped `try/catch (...)`; it calls the clear when whole-graph state is live, else `invalidate_moe_segments`. The clear's real body (`gs:99517-99567`) is described: its own unpin (`:99563-99564`), the input and leaf staging clears (`:99560`, `:99562`), and the MoE retirement waits (`gs:96549-96553`), all after th32's cancel step. | §4.4 staging plan, §7 C4 source gate |
| m-2 the third case can pass vacuously | Fixed: the seam sets `active_exec_graph.valid` alone, and the snapshot showing it set at the refusal is a precondition, else VOID; the miss-path and fa-drift clears need `exec_graph` and so cannot run first. | §7 C4 tests |
| m-3 a runtime throw leaves an empty chunk idle | Fixed: `RETURN` and `RETURN_TO_MARK` release every empty returned chunk at once; a non-empty one stays withdrawn until its owner's next begin or end. A begin leaves another live owner's non-empty chunk silently; `return refused` prints only at the owner's own begin. The runtime throw (`lc:3178`, `:2891`, with `lc:1998` and `lc:5733-5742`) is stated. New host cases: the empty return and the runtime throw with the context kept, each with a mutant. | §4.4 staging plan, §7 C4 tests |
| m-4 `RETURN_TO_MARK` can withdraw twice | Fixed: it truncates the transaction to the mark, and withdrawal lowers `zones_[zi].size` only on the flag's false-to-true transition; `KEEP` never clears the flag. New host case with `RETURN` and `KEEP` endings, in and out of a trial; mutant: the chunks left in the set. | §4.4 staging plan, §7 C4 tests and source gate |
| m-5 the owner-end wiring is not pinned | Fixed: the gate pins the owner-end call in `ggml_backend_sycl_free` before `delete sycl_ctx` (`gs:84119`), and `sched_reserve`'s `RETURN` on a throw and `KEEP` on success. The owner-end cases move to a device RED, `test-sycl-scratch-owner-end`, which frees through `ggml_backend_free`, with the orphaned and empty-at-free cases and three mutants. | §7 C4 tests and source gate |
| n-1 cites | Fixed: the clear's wait `gs:99543`, its body `gs:99517-99567`, `~llama_context` `lc:1066-1107`, and the execution release `gs:15608-15637` calling the tracking helper at `:15632`. | §4.4 staging plan |
| n-2 where `orphaned` lives | Fixed: `withdrawn` and `orphaned` both live on the chunk's `zone_chunk_state`; the marked set holds only the owner; `zone_free` tests the flag on the entry it already found (`pp:536-541`), with no cache lookup. | §4.4 staging plan |
| n-3 `configure_zones` drops the flag | Fixed: its caller releases every marked chunk first, since the rebuild (`pp:389-417`) already presumes no live zone allocation; a non-empty one refuses by name. Superseded in rev 3o: `configure_zones` itself refuses the rebuild while a withdrawn chunk survives (design-5efe-r11 m-6). | §4.4 staging plan |
| n-4 the window mutant's first failure | Fixed: the right-after-`RETURN` capacity check fails first, and the test asserts that the message quotes it. | §7 C4 tests |
| n-5 "fits only that chunk" | Fixed: candidate 512's kept chunk is filled too. | §7 C4 tests |
| n-6 step 4's zero without liveness | Fixed: the `return refused` absence is informative only beside a counted `returned` line, otherwise reported as not exercised. | §7 C4 step 4 |
| n-7 `trial` on non-trial lines | Fixed: the line names its transaction, `trial`, `reserve` or `retry`; step 4 counts only `trial` lines. | §4.4 staging plan, §7 C4 step 4 |
| n-8 the cleanup step's scope | Fixed: it applies to all 38 throw sites (37 in `ggml-sycl.cpp`, one at `getrows.cpp:291`), and that is intended. | §4.4 staging plan |

**design-5efe-r11** (`6b674097e..714081770`, 0C/0I/7m/3n), with the lead's
r11 orders (an entry guard, a pinned fourth case, split mutants, a marked-set
entry that leaves with its chunk, and `configure_zones` refusing) and the
§M69 (b) amendments 1-4, folded in rev 3o:

| finding | disposition | where |
|---|---|---|
| m-1 the handler's premise "the vectors are empty" | Fixed: `invalidate_moe_segments` keeps the vectors after a failed retire (`common.hpp:6234-6244`), and `graph_compute` records anyway (`gs:107202`, `:107206`). The record function gets an entry guard: a non-empty `moe_segments` or `moe_dispatch_graphs` prints a WARN and returns false before any submission. The stated invariant is that the handler, and the success path's move, move into empty vectors, so old executables are destroyed only by a successful retire or the cleanup step after the waits. New fifth seg-refusal case (stale vectors) with a guard-removed mutant. | §4.4 staging plan, §7 C4 tests and source gate |
| m-2 the fourth case's site and case 3's lease | Fixed: the fourth case's preconditions are per-segment (the seam fired at the segment's `moe_after`, its `exec_graph` non-null and `segment_submitted` set, its executable the last recorded identity), the WARN scored is the segment's last recorded node (`gs:97343-97345`), and cases 3 and 4 carry the lease >= 1 precondition. | §7 C4 tests |
| m-3 the fourth case's order, a mutant for the waits, the exception types | Fixed: E (entry assertion) runs right after assertion 1 and before 2-5, and its mutant quotes `executables destroyed in the unwind`. s1, s2 and s3 sample `moe_segments`; a destruction before the waits has its own mutant and message. The non-fallback exception types, which leave the moved executables in the context until the next record's invalidate (`gs:107780-107787`, `:107617-107644`), are stated. | §4.4 staging plan, §7 C4 tests |
| m-4 the `RETURN_TO_MARK` mutants | Fixed: three mutants, each with one fault and its derived message (chunks left in the set only, observed by the transaction's chunk count and the marked-set entry count; chunks left plus a transition-blind decrement; `KEEP` clearing the flag). | §7 C4 tests |
| m-5 a marked-set entry outliving its chunk | Fixed: every release goes through `release_marked_chunk`, which erases the entry in the same step, including the orphaned free's caller; a begin skips tombstones. New assertions in the host RED and the owner-end RED, with an entry-left mutant. | §4.4 staging plan, §7 C4 tests |
| m-6 `configure_zones` over a non-empty marked chunk | Fixed: it refuses the rebuild by name while any zone's `zone_chunk_state` has `withdrawn` set, returning false (C4 changes `void` to `bool`), and `configure_host_zones` leaves its reserved figures untouched. Flags carried across is rejected as a second source for capacity. The case is defensive (the plan caller returns when zones are configured, `gs:5615-5623`). New host case with a rebuild-anyway mutant. | §4.4 staging plan, §7 C4 tests |
| m-7 the retirement credited as the drain | Fixed: the destruction is safe because of the catch's own waits (`gs:107753-107764`); the retirement drains only when an epoch exists (`gs:96545-96548`). "Every other destruction site drains first" is dropped. | §4.4 staging plan |
| n-1 Appendix row cite | Fixed: the r9 m-1 row cites `gs:99517-99567` and notes the supersession in rev 3n. | Appendix B |
| n-2 the K filler | Fixed: freed at 1024's simulated scheduler destruction, before the settle's begin. | §7 C4 tests |
| n-3 the return-false paragraph | Fixed: the two inner exits and `gs:97363` (publish failure, then a no-op retire at epoch 0) are listed as pre-existing, tracked as llama.cpp-7l45, not 5efe folds; the "before the fallback" wording now says before the caller's block-graphlet and per-op fallback. | §4.4 staging plan |

**design-5efe-r12** (`714081770..40110fd66`, 0C/0I/6m/4n, P1-P4 PASS), with
the lead's r12 orders, folded in rev 3p:

| finding | disposition | where |
|---|---|---|
| M-1 where the kept executables are freed | Fixed: after the guard's `return false` the caller's retry (`gs:107214`) destroys them after a drain, or, if it fails too, the sticky `moe_graphs_disabled` (`gs:106423-106426`) stops all recording and `~ggml_backend_sycl_context` (`gs:43877`) frees them after waiting every queue (`gs:43880-43892`), before its own invalidate (`gs:43917`) and the member destructors. "The next record's entry guard" is withdrawn; the non-fallback exception types' text names the destructor too. | §4.4 staging plan |
| M-2 the fifth case is not buildable | Fixed: the second call misses `segments_match` (a different `n_nodes` or the opposite phase), and the retire seam fails every retire, including the caller's retry at `gs:107214`, until the test clears it before the context's destruction; each is a precondition. | §7 C4 tests |
| M-3 the mutants may abort before printing | Fixed: the s2 and s3 checks belong to E's group and run before 2-5; the test-only seam waits the stream at the refusal site, so the destruction cannot crash, and prints each sample as taken. The wait is not in the production catch. | §7 C4 tests |
| M-4 mutant (iii) is a no-op | Fixed: `KEEP` walks the device's marked set and clears `withdrawn` on every chunk in it, which reaches A with the truncation intact. | §7 C4 tests |
| M-5 the caller list and the set's lock | Fixed: `configure_host_zones` is a caller of `release_marked_chunk`; the set's walk, the release and the erase run under `marked_set_mutex_`, ordered before the pool's lock, with a state re-read, a no-effect release of a tombstone, and the log lines printed after the lock drops. | §4.4 staging plan |
| M-6 the refusal's plumbing | Fixed: `configure_zones` and `configure_host_zones` both return `bool`, false only for the refusal; the scan runs before the growth and the shortfall check, so no zeroing is skipped; the shortfall keeps today's behaviour; ERROR with the named line, then `GGML_ABORT` under `ggml_sycl_strict_enabled()` in the cache wrapper; no production caller receives it today, and a future caller must propagate it. The test's second call uses sizes that fit, with a shortfall control. | §4.4 staging plan, §7 C4 tests |
| N-1 the publish cite | Fixed: `gs:97361`. | §4.4 staging plan |
| N-2 the doubled parenthetical | Fixed. | §4.4 staging plan |
| N-3 the header | Fixed: the r11 items are the lead's r11 orders, separate from the §M69 (b) amendments; the Appendix B r11 heading likewise. | header, Appendix B |
| N-4 the run-on messages | Fixed: the host RED evaluates every ending and prints every failing check, and the marked set stores entries without a dedupe, so a second filing is observable. | §7 C4 tests |

**design-5efe-r13** (`40110fd66..02ba5ed6a`, 0C/0I/5m/4n), with §M81 (c)-(e),
folded in rev 3q:

| finding | disposition | where |
|---|---|---|
| M-1 "nothing records or invalidates again" | Fixed: nothing records again, but invalidation still happens at the catch's cleanup step and every `sycl_exec_graph_clear_active` caller (`gs:99565`, `:99579`, `:105417`, `:15773`), each retrying the retire; a success destroys after its drain, a failure keeps them for `~ggml_backend_sycl_context`. "Every destruction point drains." | §4.4 staging plan |
| M-2 the lock census and order | Fixed: `marked_set_mutex_` is L4 (last ordinal) and the pool's `mutex_` L5, both added to canon §12.5 in C4's commit; callers' held locks are named from source (superseded in rev 3r: derived from the callers, design-5efe-r14 I-1); the owner moves out of the pool's scope and is released after both locks drop, so the pool's lock is a leaf for these functions; new lock-order gate with two planted controls. | §4.4 staging plan, §7 C4 source gate |
| M-3 the insert side and the re-read | Fixed: every access (filing, begin's walk, owner-end walk, erase, `marked_entries`) holds `marked_set_mutex_`; the re-read is a pool call under the pool's `mutex_` returning "already released" for a tombstone. | §4.4 staging plan |
| M-4 gate and test for configure's release | Fixed: `configure_host_zones` is in the gate's `release_zone_chunk` caller list; a configure own-release run and a direct-`release_zone_chunk` mutant. | §7 C4 source gate and tests |
| M-5 the strict-abort arm | Fixed: own subprocess RED `test-sycl-configure-refusal-abort` (renamed `test-sycl-configure-refusal` in rev 3r, design-5efe-r14 M-8) (strict child: SIGABRT and the ERROR line; plain child: rc 0), two mutants; the non-strict test unsets the variable and is VOID unless `ggml_sycl_strict_enabled()` is false at entry; the gate pins the strict read. | §7 C4 tests and source gate |
| N-1 the shortfall control | Fixed: runs after the success call. | §7 C4 tests |
| N-2 the uc range | Fixed: `uc:20415-20436`. | §4.4 staging plan |
| N-3 the seam's wait | Fixed: scoped to the fourth case. | §7 C4 tests |
| N-4 "opposite phase" | Fixed: dropped; the precondition names a different `n_nodes` and no whole-graph record having set `active_exec_graph.valid` (`gs:105417`, `:105558`, `:106126` verified as clears). | §7 C4 tests |

**design-5efe-r14** (`02ba5ed6a..4db80fa62`, 0C/1I/8m/2n, P2 and P4 residues), with
the lead's r14 rulings, folded in rev 3r:

| finding | disposition | where |
|---|---|---|
| I-1 the held-lock claims were not derived from callers | Fixed: every caller of `marked_set_mutex_` is named with the locks it holds at e2461d4fb (begin and trial entries, the owner-end step, `configure_host_zones`' callers with the unreferenced static `gs:33353` noted, the orphaned-free paths including `process_deferred_frees_locked` under `rw_mutex_` and `deferred_frees_mutex_`); the 15 `release_entry_allocation_locked` sites are tabulated (`uc:4925` holds none); no SCRATCH producer reaches the deferred lists today, with cites, and the order covers it anyway; the order is `rw_mutex_` (and `deferred_frees_mutex_`), then `marked_set_mutex_`, then the pool's `mutex_`; the owner release can run under a caller's L4 lock, legally and without cycle; the "unpublished record cannot lie in a withdrawn chunk" reason is replaced by "holds no lock". The gate checks the listed bodies' scopes only. | §4.4 staging plan, §7 C4 source gate |
| M-1 the carrier | Fixed: `release_zone_chunk` returns an opaque move-only `released_chunk` (base, size, owner); a public `release_chunk_backing(released_chunk &&)` calls the private `free_chunk_owner` on it with no lock, so the null-base early return (`pp:779-781`) cannot swallow it and the missing-owner assert still guards it; no drain wait since lease count was 0 (superseded in rev 3s: the carrier is RAII and its destructor is the release, design-5efe-r15 I-1). | §4.4 staging plan |
| M-2 the non-recursive lock | Fixed: `release_marked_chunk_locked` core (lock held, collects into a vector) and a public wrapper; the begin's walk and configure's release run in one scope and free every collected backing after it closes (extended in rev 3s: the owner-end walk and `RETURN` are core users too, design-5efe-r15 M-1). | §4.4 staging plan, §7 C4 source gate |
| M-3 the footprint decremented before the free | Fixed: `total_allocated_` is subtracted in a second short pool-lock section after the physical free; a concurrent grow never sees room that is not there (superseded in rev 3t and rev 3u: the subtraction is conditional on the registry query, and a refused free is reconciled from a stranded list, §M88 (1), §M92). New seam case and a decrement-before-free mutant. | §4.4 staging plan, §7 C4 tests |
| M-4 the pool's census row | Fixed: first in L5, so `grow`'s nesting of `g_runtime_alloc_mutex` (`pp:939`, `uc:15578`) conforms; other nesting is llama.cpp-uilv; the retained-store row is cited at `d8a67422d`; "never nests" is limited to the release path. | §4.4 staging plan |
| M-5 the configure cites | Fixed: `gs:33366` (unreferenced static, not production) and `gs:33893` (the live site, via `gs:12476` and `gs:13124`, no census mutex). | §4.4 staging plan |
| M-6 controls per clause | Fixed: six clauses, six planted controls, each with a quoted reason. | §7 C4 source gate |
| M-7 `gs:106126` | Fixed: its own precondition on the two optional-layout epochs. | §7 C4 tests |
| M-8 the strict test's construction | Fixed: children get the environment through `execve` `envp` (the accessor latches); no `ENVIRONMENT_MODIFICATION`; `waitpid` scoring with `SIGABRT`; `RLIMIT_CORE` 0; VOID exits 77; the plain child frees its stand-in; the device-touching fixture is labelled `sycl;cache;mem-handle` and pinned, for this test and `test-sycl-scratch-trial-return`; the configure cases move into the plain child. | §7 C4 tests |
| N-1 the owner-end cite | Fixed: `gs:84119`, with `gs:16818` as a second path. | §4.4 staging plan |
| N-2 the destructor cite | Fixed: `pp:141-162`. | §4.4 staging plan |

**design-5efe-r15** (`4db80fa62..858b74433`, 0C/1I/5m/4n, P2 residue and P4
residues), with the lead's r15 rulings, folded in rev 3s:

| finding | disposition | where |
|---|---|---|
| I-1 the carrier's lifetime and return shape | Fixed: `released_chunk` is a move-only nested class constructed only by `release_zone_chunk`, and its destructor is the release (the `mem_handle` reset, then the footprint subtraction under the pool's `mutex_`); a moved-from carrier is a no-op; `release_chunk_backing(released_chunk &&)` only consumes it early, so one path serves a consumed and a dropped carrier; callers reserve `out` before the first release; `release_zone_chunk` returns a four-valued status (released, already released, refused live, refused lease) with the carrier as an out-parameter, and the core keys its behaviour on it; a registry-refused free still lets the footprint leave, with the registry's leak diagnosis as the evidence (superseded in rev 3t: the footprint leaves only when the allocation is gone, §M88 (1); and in rev 3u: a refused free is a scheduled retry reconciled from a stranded list, §M92); gate clause 8 and a planted control pin the destructor; a dropped-carrier case with a do-nothing-destructor mutant and a double-subtract mutant. | §4.4 staging plan, §7 C4 source gate and tests |
| M-1 the owner-end walk and `RETURN` | Fixed: a caller holding `marked_set_mutex_` calls the core and never the public form; the begin, configure, owner-end walk and `RETURN`/`RETURN_TO_MARK` (filing and immediate release in one scope) are the four core users, each releasing its collected backings after its scope; the one public-form caller is the orphaned chunk's emptying free. A two-empty-chunks owner-end case and a one-of-two mutant. | §4.4 staging plan, §7 C4 tests |
| M-2 four gate holes | Fixed: clause 7 pins the call forms (core called only inside a held scope, public form never inside one), with two controls; clause 5's controls name `release_chunk_backing` and the collected-vector loop instead of the private `free_chunk_owner`; clause 4 names `deferred_frees_mutex_` and C4's census commit adds it to canon §12.5; the gate is registered `sycl;host;source-contract` like `test-sycl-mmap-support-caps-source`. | §4.4 staging plan, §7 C4 source gate |
| M-3 callee-scoped held-lock claims | Fixed: `unified_cache::deallocate`'s only caller is the wrapper at `uc:21288`, which has no production caller; for the registry release and rollback paths the rule is stated (a SCRATCH final-owner destruction or rollback under an L5 lock is non-conforming under canon 1612-1619, outside C4, llama.cpp-uilv), and "hold no lock" is deleted for (iii); `uc:4639` is `shutdown_resources`' production caller. | §4.4 staging plan |
| M-4 the strict test's environment and the pins | Fixed: each child's `envp` is a copy of the driver's `environ` with the strict variable set or erased; a child with no selector or no device exits 77; both moved REDs pin `level_zero:0,1` as the lead ruled (the reviewer proposed `level_zero:1` for the configure test; superseded in rev 3t: both pin `level_zero:1`, §M88 (2) and the lead's follow-up), and the Device REDs list cross-references them. | §7 C4 tests |
| M-5 the footprint seam | Fixed: the seam samples after `free_chunk_owner` returns and before the second section; it is `GGML_SYCL_PRIVATE_TESTING`-compiled and absent from the shipped library; the RED is a private-fixtures test linked like `test-unified-cache-bugs` and never with libllama (§M82 case (2)). | §7 C4 tests |
| N-1 stale vocabulary and placement | Fixed: "host fixture" and "host RED" replaced; the two REDs are cross-referenced under the Device REDs header; the registration is reflowed. | §7 C4 tests, §7 C4 step 4 |
| N-2 Appendix B row n-4 | Fixed: superseded markers on the row that quoted the old footprint arrangement, and on the r14 M-1 and M-2 rows. | Appendix B |
| N-3 the lease-0 half | Fixed: no lease can be taken after the tombstone, because `acquire_chunk_lease` and `find_chunk_handle` skip a null base under the same `mutex_` (`pp:1026-1047`). | §4.4 staging plan |
| N-4 two wording points | Fixed: "15 call sites (16 grep hits including the definition at `uc:12345`)"; the window's consequence (a concurrent grow can be refused for the duration of one free) is stated. | §4.4 staging plan |

**§M88** (the lead's answers to rev 3s's questions), applied in rev 3t:

| ruling | disposition | where |
|---|---|---|
| (1) the footprint leaves only when the allocation is gone | Applied in rev 3t: the carrier's destructor takes the owner's `alloc_metadata` before the reset, resets through `free_chunk_owner`, then asks the registry read-only by exact (`ptr`, `key()`) through a new shipped accessor `unified_allocation_registered_exact`; gone subtracts. The "still registered" branch (keep, ownerless-lease WARN, strict abort, a strict refused-free child) is superseded in rev 3u by §M92: a refusal is a scheduled retry, so the footprint is stranded on a pool list and reconciled from the registry, with a retry WARN and no abort. | §4.4 staging plan, §7 C4 source gate and tests |
| (2) selector | Applied: `test-sycl-configure-refusal` pins `level_zero:1`. `test-sycl-scratch-trial-return` first stayed `level_zero:0,1` (§M88 named only the configure test) and now pins `level_zero:1` too, on the lead's follow-up ruling: the same one-pool-one-queue argument applies. | §7 C4 tests |
| (3) ticket | llama.cpp-uilv stands for the L5-held SCRATCH destruction and rollback sites; no change. | §4.4 staging plan |

**design-5efe-r16** (`858b74433..ce6226443`, 0C/1I/6m/6n, P4 FAIL), with §M92,
folded in rev 3u:

| finding | disposition | where |
|---|---|---|
| I-1 a kept footprint is never reconciled | Fixed (§M92): a refusal is a scheduled retry (`uc:857-862`, `:983-993`, `:995-1019`, `:1031-1056`), so the carrier appends `{metadata, backing size}` to the pool's `stranded_` list without subtracting and prints a retry WARN; `allocated()` and every admission test run `reconcile_stranded_locked` first and subtract each entry the registry no longer holds; teardown WARNs for entries still registered; the "ownerless lease" wording and the strict abort are dropped on this path. RED: keep, `process_retries()`, then the next `allocated()` drops it, with three mutants; a teardown case; gate clause 10. | §4.4 staging plan, §7 C4 source gate and tests |
| M-1 the drop path under a listed lock | Fixed: stated as non-conforming under canon 1612-1619, reachable only if a caller breaks the collect-then-release rule; clause 5 pins that each collecting vector is declared before its scope's lock statement, with a control. | §4.4 staging plan, §7 C4 source gate |
| M-2 the metadata's source | Fixed: `allocate_pinned_chunk_owner` reads `alloc_owner::metadata()` before `from_owned_alloc` consumes the owner and returns a `pinned_chunk_backing`; `grow_into` stores it on the chunk on both paths; the carrier holds a back-pointer and never outlives its pool (`~pinned_chunk_pool` asserts none outstanding). | §4.4 staging plan |
| M-3 the accessor and clause 8's polarity | Fixed: an accessor RED case (same `ptr`, different id, absent row; ptr-only mutant), gate clause 9 (one lock, compares `key()`, no writes) with two controls, and clause 8's inverted-branch control. | §4.4 staging plan, §7 C4 source gate and tests |
| M-4 the status per caller | Fixed: a per-caller status table with a REFUSED_LEASE handler at every row and a fifth value, ORPHANED, returned by `release_zone_chunk` with `orphan_if_live` so the owner-end walk is one pool call (extended in rev 3w: also in place of REFUSED_LEASE, with the lease's drop as a release event, r17 M-4). | §4.4 staging plan |
| M-5 the RED's cleanup | Fixed: the hooks come as a hold and an unhold; the fixture unholds, calls `process_retries()` and scores `retry_count()` 0. | §7 C4 tests |
| M-6 the recipes and entry checks | Fixed: `test-sycl-configure-refusal` is a private-fixtures test (it reads `marked_entries`), built like trial-return; the refused-free arm exits 77 when `GGML_SYCL_STRICT_LEASES` is set; the strict refused-free child is removed with the strict abort. | §7 C4 tests |
| N-1 two cites | Fixed: `uc:16553-16554`, `uc:16534`. | §4.4 staging plan |
| N-2 one canon range | Fixed: canon 1612-1619 throughout. | §4.4 staging plan |
| N-3 the 114-column line | Fixed: the recipe paragraph is reflowed. | §7 C4 tests |
| N-4 the enum type | Fixed: `pinned_chunk_release`. | §4.4 staging plan |
| N-5 markers, header, ordinal | Fixed: the r14 M-3 row is marked superseded; the header names the trial-return pin move; `deferred_frees_mutex_`'s ordinal is decided in C4's census commit and the gate reads it from the table. | Appendix B, status block, §4.4 staging plan |
| N-6 "leaked lease" | Fixed: the wording is gone from the live text; it survives only inside superseded markers (the status block and the §M88 row). | §4.4 staging plan, Appendix B |


**The stranded list's bound** (the lead's follow-up to §M92), applied in rev 3v:

| ruling | disposition | where |
|---|---|---|
| no terminate path for the stranded append | Fixed: the first draft reserved 8 entries and let a failed allocation terminate inside the `noexcept` destructor. Now `stranded_` is reserved to the pool's chunk count wherever a chunk is added (superseded in rev 3w: once, in `grow_into`, before the add, r17 M-1) (at most one stranded entry exists per chunk), the destructor asserts `size < capacity` before the append, and gate clause 11 pins the reserve-at-grow and the allocation-free append. | §4.4 staging plan, §7 C4 source gate |

**design-5efe-r17** (`858b74433..0ee6ad375`, 0C/1I/5m/5n, P4 FAIL), with the
lead's r17 rulings, folded in rev 3w:

| finding | disposition | where |
|---|---|---|
| I-1 the reconcile misses three admission tests | Fixed: one private `footprint_locked()` reconciles and returns the footprint, and every decision read goes through it: `allocate_segmented` (`pp:186`), `pre_allocate` (`pp:242`), `configure_zones` (`pp:323`), `grow_zone` (`pp:661`), `allocate_from_chunks` (`pp:827`), `grow_into` (`pp:892`) and `allocated()`; an empty list skips the registry lock; the two members are `mutable`; `allocated()`'s external callers (`uch:4140` from `gs:5688`, `mem-ops.cpp:392`) hold no registry lock. Clause 10 now names no body but the reconcile, the destructor and `grow_into`'s commit, with one control per body. | §4.4 staging plan, §7 C4 source gate |
| M-1 the reserve is after the add and is three sites | Fixed: one site, `grow_into`, which reserves to `chunks_.size() + runtime_chunks_.size() + 1` before the `emplace_back` (moved by r18 N-1 to before the owner allocation) and maps `bad_alloc` to `return false`; `configure_zones`, `allocate_segmented` and `grow_zone` add no chunk. | §4.4 staging plan, §7 C4 source gate |
| M-2 the reserve's value and the assertion are unpinned | Fixed: controls for a constant reserve, a reserve after the add (replaced in rev 3x by a reserve after the owner allocation, r18 N-1), and a missing or late assertion; the append is an `emplace_back` into reserved storage; a RED strands nine 64 MiB chunks, with a `reserve(8)` mutant that stops at the named assertion. | §7 C4 source gate and tests |
| M-3 the prune can drop the reserve | Fixed: the prune is in place (`std::remove_if` and `erase`), never shrinking capacity; a control for a rebuild. | §4.4 staging plan, §7 C4 source gate |
| M-4 REFUSED_LEASE has no release trigger | Fixed: the lease's last drop is an event-driven release (`release_chunk_lease` hands the index back, `host_release_chunk_lease` calls the public form; no host wait, no polling); `orphan_if_live` orphans for a live allocation first and then a lease; a fourth owner-end case drives both orders with a mutant. | §4.4 staging plan, §7 C4 tests |
| M-5 the carrier-count assertion | Fixed: it is `~pinned_chunk_pool`'s first statement, counts owning carriers only (a move transfers, a moved-from carrier is not counted); gate clause 12 with a control; an outstanding-carrier child in `test-sycl-configure-refusal`. | §4.4 staging plan, §7 C4 source gate and tests |
| N-1 the `release_ref` cite | Fixed: `uc:857-862`. | §4.4 staging plan, Appendix B |
| N-2 two over-wide lines | Fixed: reflowed. | §7 C4 source gate and tests |
| N-3 the census sentence | Fixed: it now names the reconcile's registry query beside `grow_into`'s allocation. | §4.4 staging plan |
| N-4 Appendix B N-6 | Fixed: "gone from the live text; kept only inside superseded markers". | Appendix B |
| N-5 the WARN and the accessor RED | Fixed: the accessor returns the row's own state (GONE, LIVE or RELEASING) and the WARN names it without claiming a cause (the row cannot tell a scheduled retry from retained references or the shutdown abandon); the accessor RED erases its published row with `allocation_registry_test_erase` (the accepted deviation is §M97). | §4.4 staging plan, §7 C4 tests |

**design-5efe-r18** (`0ee6ad375..02d766337`, 0C/0I/3m/6n), with §M104,
folded in rev 3x:

| finding | disposition | where |
|---|---|---|
| M-1 the lease-drop path from a `noexcept` frame | Fixed: the public form holds one inline `released_chunk` slot and the core has a single-slot overload, so no vector or allocation is on the path; `host_release_chunk_lease` is `noexcept`; the six `release_lease_state` call sites (`mem-handle.cpp:982`, `:1004`, `:1067`, `:1166`, `:1665`, `:1773`) are stated, all after the handle's leaf lock; the new exposure (L4 then L5, then a blocking `sycl::free` and the registry lock, with no inversion, acceptable as a final `mem_handle` drop already frees) is stated; clause 7 pins that the lease drop calls the public form and does not allocate, with two controls. | §4.4 staging plan, §7 C4 source gate |
| M-2 the helper has no control | Fixed: clause 10's allow-list names `footprint_locked`'s return, the log-only exemption is mechanical (the numerator of a division inside a `GGML_LOG_*` call), and there are controls for the helper without its reconcile and for a local copy in `grow_zone`. | §7 C4 source gate |
| M-3 a stale WARN score | Fixed: the RED scores `release not yet complete after the owner reset (registry row LIVE)` exactly once and no further line on a second call; clause 8 gains a control that the WARN names no cause. | §7 C4 tests and source gate |
| N-1 the sync/async labels | Fixed: `pp:922` is the async path, `pp:945` the synchronous path (catch `pp:950`); the reserve moves before the owner allocation, with an explicit catch. | §4.4 staging plan, §7 C4 source gate |
| N-2 `grow_zone`'s cite | Fixed: `pp:661`. | §4.4 staging plan, Appendix B |
| N-3 the carrier count's decrement | Fixed: the last statement of the destructor, with `assertion absent` and early-decrement controls. | §4.4 staging plan, §7 C4 source gate |
| N-4 "equals the chunk count" | Fixed: "equals nine"; the abort string is in §4.4 and clause 11. | §4.4 staging plan, §7 C4 tests and source gate |
| N-5 GONE and §M97 | Fixed: upper case in clause 8; §M97 cited at the accessor and the Appendix B N-5 row. | §4.4 staging plan, Appendix B |
| N-6 the lease-drop lookup scope and the status order | Fixed: the lookup runs only on a drop to zero on a `chunks_` chunk and scans only the SCRATCH zone's states; a status-order RED line (the owning begin on a live and leased chunk prints `reason=live`) with a mutant; the begin row of the status table now records that the owning begin prints. | §4.4 staging plan, §7 C4 tests |

**design-5efe-r19** (`02d766337..00ffc08fb`, 0C/0I/2m/4n), with §M110,
folded in rev 3y:

| finding | disposition | where |
|---|---|---|
| M-1 clause 7's path list misses frames | Fixed: the lease-drop path is every frame the drop runs through: `host_release_chunk_lease`, `release_chunk_lease`, the public form, the single-slot core overload, `release_zone_chunk` and the carrier destructor; a control plants `vector v; v.reserve(1)` in the overload (and in `release_zone_chunk` and `release_chunk_lease`). | §7 C4 source gate |
| M-2 clause 10 fails the correct tree | Fixed: `~pinned_chunk_pool`'s reset (`pp:163`) is allow-listed by name; the gate scans comment-stripped source (so the `pp:334-335` comment is not a mention); the nine cited log lines are asserted to be the only exempt mentions; controls for a write outside the allow-list, a tenth log read and a comment that must not trip. | §7 C4 source gate |
| N-1 clause 5 and the inline slot | Fixed: clause 5 names the public form's slot and plants `released_chunk slot declared inside the marked_set_mutex_ scope`. | §7 C4 source gate |
| N-2 the llama.cpp-uilv ticket text | Not a doc change: the lead adds the ticket comment naming the lease-drop callers; the doc cites the ticket. | llama.cpp-uilv |
| N-3 Appendix B reserve position | Fixed: the r17 M-1 and M-2 rows are marked as moved by r18 N-1. | Appendix B |
| N-4 the explicit `bad_alloc` catch | Fixed: clause 11 gains `stranded reserve without bad_alloc catch`. | §7 C4 source gate |

**design-5efe-r20** (`00ffc08fb..12839f154`, 0C/0I/2m/4n), with the lead's
r20 relay, folded in rev 3z:

| finding | disposition | where |
|---|---|---|
| M-1 frames missing from clause 7 | Fixed: the ban list adds `free_chunk_owner` and `unified_allocation_registered_exact`; the scanned-body list adds `host_release_chunk_lease`, `release_chunk_lease` and `free_chunk_owner`; the single-slot overload is selected by signature and the vector overload is exempt; frames above `host_release_chunk_lease` are stated unchanged and outside the gate; `vector v; v.reserve(1)` is planted in `free_chunk_owner` and the accessor. | §7 C4 source gate |
| M-2 the grow_into control was unfalsifiable | Fixed: the allow-list is per mention (body, ordinal, statement kind) with exact counts (5 allowed, 9 exempt); a positive presence check that each of the seven decision bodies calls `footprint_locked()`; the `grow_into` mutant re-reads the member at `pp:892` and fails both. | §7 C4 source gate |
| N-1 vector detection | Fixed: the `vector<` token (`auto & vec` at `pp:1057` is not one). | §7 C4 source gate |
| N-2 the ordinal | Fixed: it counts every mention in the body. | §7 C4 source gate |
| N-3 the llama.cpp-uilv comment | Not a doc change: the lead extends the ticket. | llama.cpp-uilv |
| N-4 `comment mention flagged` | Fixed: marked a negative control that must pass, and the clause intro says so. | §7 C4 source gate |

**design-5efe-r21** (`12839f154..9c0ee6941`, PASS with 2 nits), with §M114,
folded in rev 3aa:

| finding | disposition | where |
|---|---|---|
| N-1 the signature selector | Fixed: the gate selects the single-slot overload by the name `release_marked_chunk_locked` plus exactly one `&`, never `&&`. | §7 C4 source gate |
| N-2 `auto & vec` cite | Fixed: `pp:1057` (in clause 7 and the r20 N-1 row). | §7 C4 source gate, Appendix B |
