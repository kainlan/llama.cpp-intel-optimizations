# fmcp addendum A11: host STAGING and SCRATCH are sized from tensors that are copied

> **Execution:** `team-driven-development`. One implementer (Opus), fresh Opus spec and quality reviewers.
> GPU and model-loading runs are the lead's, serial, under `GPU.lock`. Builds only on the lead's literal GO.

**Parent:** epic `llama.cpp-fmcp` (Qwen3.8-Flash-Next on the B70). Fixes `llama.cpp-c24s`.
**Base:** `integ/th32` at `8fe8f46ec`. Line anchors below were read there; verify each before editing.
**Order (owner ruling 2026-11-14):** this addendum first, then a revised A10. A10.1 (`llama.cpp-gzzq`) is on HOLD.

**Goal:** a default load of Qwen3.8-Flash-Next stops reserving about 154 GB of pinned host memory that
nothing can use, by sizing the host STAGING and SCRATCH zones from the largest tensor that can be copied
through them, not from the largest tensor in the model.

**Architecture:** the planner keeps `plan.max_tensor_bytes` as the model-wide maximum, and gains one new
maximum, `plan.max_copied_tensor_bytes`, that leaves out a tensor the plan keeps on the host and that is
consumed only by a row gather. The three sizing sites that bound a copy read the new maximum. Every other
tensor class keeps today's bound, so no other model's zones move.

**Tech stack:** C++ in `ggml/src/ggml-sycl/unified-cache.{cpp,hpp}`; host-only planner tests that link
no device; Python source pins under `tests/`.

**Test infrastructure:** ctest. Planner tests are plain executables that call `compute_placement_plan`
and `compute_multi_device_plan` with a synthetic inventory (see `plan_q2k` and `plan_q2k_two_devices` in
`ggml/src/ggml-sycl/tests/test-sycl-host-weight-zone-terms.cpp:301-318`). Source pins are
`tests/test-sycl-*-source.py` with a `--self-test` mutant mode.

---

## Evidence (lead runs 2026-11-13, B70 `level_zero:0`, exps-IQ4_NL pack, driver 26.40)

Logs: session scratchpad `fmcp/ring-arms/mapped-default.err` and `pinned-default.err`.

| line | mapping load (default) | `-lm none`, late plan |
|---|---|---|
| `[SYCL-PLAN] Host staging zone: per_inflight` | 103,762.3 MB | 1,290.3 MB |
| `[SYCL-PLAN] Host staging zone: total` | 103,779.4 MB | 1,307.4 MB |
| ledger scratch | 54,905,674,752 B (52,362 MiB) | not read |
| `[SYCL] Pinned pool pre_allocate ... working set` | 156,269.5 MB | 52,715.3 MB |
| `[HOST-ARENA] Committed ... after plan pre-size` | 159,744.0 MB | 55,296.0 MB |

`per_inflight` on the mapping load is exactly `2 x 51,880.1 MiB + 2 MiB`. 51,880.1 MiB is
`per_layer_token_embd.weight` (type Q8_0, `160 x 320,001,536`), the model's hashed row-lookup table. The
inventory's next largest tensors are `output.weight` and `token_embd.weight` at 644.1 MiB, then the
expert tensors at 450.0 MiB.

## What the code does today

All in `ggml/src/ggml-sycl/`.

1. `populate_host_zone_sizing` (`unified-cache.cpp:35067`) sets `plan.max_tensor_bytes` to the plain
   maximum of `item.size` over the whole inventory (`:35075-35079`) and asserts it equals
   `zone_maxima.any_tensor` (`:35091`).
2. `plan.max_staging_pair_bytes` is the maximum over `plan.entries` of
   `src_size + max(dst_size, src_size)` (`:35222-35225`). The table has one dense entry with
   `src_size == dst_size == 51,880.1 MiB`, `on_device == false`.
3. STAGING: `s1_in_flight x (max(2 x max_tensor_bytes, max_staging_pair_bytes) + 2 MiB)` plus small
   terms (`:35673-35703`).
4. SCRATCH: the fold `{ plan.max_tensor_bytes, k_scratch_headroom, moe_host_scratch }` (`:35723`).
5. A second, independent SCRATCH floor in `placement_plan_host_zone_commit_terms`
   (`unified-cache.hpp:2254-2255`): `max(plan.host_zone_scratch_bytes, onednn_live + plan.max_tensor_bytes + 32 MiB)`.
   Fixing only site 4 leaves SCRATCH at about 52 GB.

Who can occupy those zones (spike, 2026-11-14):

- STAGING's main tenant is `staging_buffer_pool` (`common.hpp:3214`). It holds one tensor, or one whole
  expert tensor, on its way to a non-AOS layout **on a device**. It allocates with
  `forbid_host_zone_growth`, so an under-sized zone returns null and the caller throws or aborts.
- SCRATCH holds readbacks and fallback copies of **device-resident** tensors, single expert slices, and
  64 MiB copy chunks.
- A tensor the plan keeps on the host is read in place by the CPU. Nothing copies it through either zone.
- A device-resident gather-only tensor (an ordinary token embedding on a GPU) **can** be staged. So
  "gather-only" alone is the wrong test.
- `zone_maxima.dma_streamed` is the wrong bound: it is "per-layer family of 4 or more", which leaves out
  `output.weight` and `token_embd.weight` (644.1 MiB, device-bound, can take a staged fill).

## Decisions this addendum takes (owner confirms by approving it)

1. **The narrowest safe rule.** A tensor is left out of the new maximum only when both hold: the model
   loader marked it gather-only (`placement_tensor_info::get_rows_only`), and the plan has at least one
   entry for it and none of its entries is on a device. Every other tensor counts as today. A wider rule
   ("leave out every host-resident tensor") is probably also correct, but an under-sized STAGING zone
   aborts, and this addendum has no evidence for the wider class.
2. **`plan.max_tensor_bytes` keeps its meaning.** It stays the model-wide maximum, with its assert
   against `zone_maxima.any_tensor`, its log lines, and its two other readers (the unread
   `moe_q8_workspace_bytes` and the TP fallback). The new field is additive.
3. **The same rule filters the per-entry pair term.** An entry is skipped in `max_staging_pair_bytes`
   exactly when its tensor is left out of the new maximum.
4. **No new flag and no new log line.** The existing staging line reports the result. One existing
   `GGML_LOG_INFO` gains the new maximum next to the global one, so a log shows both.
5. **Out of scope:** reading the table from storage (its own addendum under `llama.cpp-qy3g`), the
   lazy-table inventory class, the `2 x` factor in STAGING, and A10.

## Team Topology

**Implementers:** 1 at a time (one track). **Reviewers:** spec then quality, fresh per review, Opus.

| Track | Tasks | Description |
|---|---|---|
| A | A11.1 | The new maximum, its three readers, tests and pins |

### File Ownership Map

| File | Task | Note |
|---|---|---|
| `ggml/src/ggml-sycl/unified-cache.cpp` | A11.1 | `populate_host_zone_sizing` only |
| `ggml/src/ggml-sycl/unified-cache.hpp` | A11.1 | `placement_plan` field, commit terms floor |
| `ggml/src/ggml-sycl/tests/test-sycl-host-weight-zone-terms.cpp` | A11.1 | new arms |
| `ggml/src/ggml-sycl/tests/host-ledger-commit-oracle.hpp` | A11.1 | mirrors the commit floor (`:46-47`) |
| `ggml/src/ggml-sycl/tests/test-unified-runtime-alloc.cpp` | A11.1 | mirrors the floor (`:984-988`) |
| `ggml/src/ggml-sycl/tests/test-sycl-host-ledger.cpp` | A11.1 | only if its floor asserts name the field (`:733-741`, `:812-820`, `:876-902`) |
| `tests/test-sycl-host-role-routing-source.py` | A11.1 | pins the SCRATCH fold text (`:654`, `:1075`, `:2083-2100`, `:2175`, `:2672`) |
| `docs/backend/sycl-memory-design.md` | A11.1 | "Path-scoped zone sizing" section: one paragraph |

---

### Task A11.1: STAGING and SCRATCH read the largest tensor that can be copied

**Track:** A. **Depends on:** none. **Ticket:** created by the lead at spawn.

**Done when:** on a plan whose largest tensor is a host-kept gather-only table, STAGING and SCRATCH are
sized from the next largest tensor; on the same plan with that table device-bound, and on a plan whose
largest host tensor is not gather-only, both zones are byte-identical to today.

#### Step 1: census, before any code. STOP and message the lead if any answer contradicts this addendum.

Answer each with `file:line`.

1. Does `placement_entry` carry the tensor's name and can entries be matched to inventory items by it
   (`copy_tensor_info_to_entry`, `unified-cache.cpp:31793`)? Does the entry carry `get_rows_only`, or
   must the rule join on name?
2. Are `entry.on_device` values final when `populate_host_zone_sizing` runs, in both builders (single
   device `:36995-37006`, multi device `:39054-39056`)? Name any later writer of `on_device`.
3. List every reader of `plan.max_tensor_bytes` and `plan.max_staging_pair_bytes` in non-test code
   (`cat <file> | grep -n`; `ggml-sycl.cpp` is not indexed). The expected set is the five sites above
   plus `:35245`, `:35456`, `:35461` and three log lines. Report any other.
4. Can a host-kept gather-only tensor reach `staging_buffer_pool`, a SCRATCH readback, or
   `cpu_fallback_graph`? Read `ggml_sycl_fill_reordered_gpu` (`ggml-sycl.cpp:~36940`),
   `ggml_sycl_fill_reordered_host` (`:~41958`), `buffer_get_tensor` (`:~47027`) and the CPU fallback
   copy. One path that copies a host-kept table is a STOP.
5. Which existing test asserts a STAGING or SCRATCH size that a gather-only host tensor in its fixture
   would move? Run nothing; read the fixtures. `test-zone-sizing.cpp` asserts per-model `any_tensor`
   values and must not change.
6. Does any fixture already build an inventory item with `get_rows_only = true`? Name it.

#### Step 2: RED, the planner test

File: `ggml/src/ggml-sycl/tests/test-sycl-host-weight-zone-terms.cpp`. Add one function and call it
from `main` (`:1113`). Use the file's own `check_eq` / inventory helpers; match their exact names and
signatures after reading the file, and reuse `plan_q2k` and `plan_q2k_two_devices` (`:301-318`).

The arms, each on both builders:

```cpp
// A gather-only table the plan keeps on the host is read in place, so it must not size a zone that only
// holds copies. `base` is the file's ordinary inventory; `table` is 64x its largest tensor.
static void check_copied_tensor_maximum() {
    std::vector<placement_tensor_info> base = /* the inventory the weight-zone arms use */;
    size_t base_max = 0;
    for (const auto & t : base) { base_max = std::max(base_max, t.size); }

    placement_tensor_info table("per_layer_token_embd.weight", base_max * 64);
    table.type          = GGML_TYPE_Q8_0;
    table.ne[0]         = 160;
    table.ne[1]         = static_cast<int64_t>(table.size / 170);
    table.get_rows_only = true;

    std::vector<placement_tensor_info> with_table = base;
    with_table.push_back(table);

    const placement_plan ref  = plan_q2k(base);
    const placement_plan plan = plan_q2k(with_table);

    // Arm 1: the table is the model-wide maximum and is on the host.
    check_eq("the table is the largest tensor in the model", plan.max_tensor_bytes, table.size);
    check_eq("the table's entry is host-resident", plan.find_entry_on_device(table.name), false);

    // Arm 2: neither zone is sized from it.
    check_eq("largest copied tensor leaves the host table out", plan.max_copied_tensor_bytes, base_max);
    check_eq("host STAGING is unchanged by a host-kept table", plan.host_zone_staging_bytes,
             ref.host_zone_staging_bytes);
    check_eq("host SCRATCH is unchanged by a host-kept table", plan.host_zone_scratch_bytes,
             ref.host_zone_scratch_bytes);
    const host_zone_commit_terms a = placement_plan_host_zone_commit_terms(ref, 0, 2048ull << 20);
    const host_zone_commit_terms b = placement_plan_host_zone_commit_terms(plan, 0, 2048ull << 20);
    check_eq("the commit floor is unchanged by a host-kept table", b.scratch, a.scratch);

    // Arm 3 (control): the same tensor, not gather-only, still sizes both zones as before.
    std::vector<placement_tensor_info> with_dense = base;
    table.get_rows_only = false;
    with_dense.push_back(table);
    const placement_plan dense = plan_q2k(with_dense);
    check_eq("a host tensor that is not gather-only still counts", dense.max_copied_tensor_bytes,
             table.size);
    check_true("and STAGING grows with it", dense.host_zone_staging_bytes >= 2 * table.size);
}
```

`find_entry_on_device` is a placeholder for whatever lookup the file already uses to read an entry's
tier; use the existing helper, do not add a plan method. If the fixture's 1 MiB VRAM budget cannot hold
a device-bound table, Arm 4 below needs a larger budget through a local copy of `plan_q2k`.

Arm 4 (control, device-bound): a gather-only tensor small enough that the plan puts it on the device
(give the plan a budget that fits it) must count: `max_copied_tensor_bytes >= that tensor's size`.

Run (after the lead's GO for a build of this one target):

```bash
ctest --test-dir build -R '^test-sycl-host-weight-zone-terms$' --output-on-failure
```

Expected RED: a compile error on `max_copied_tensor_bytes` first. Add the field with value
`max_tensor_bytes` (no rule yet), rebuild, and record the real RED: Arm 2 fails with STAGING about
`128x` the reference. Commit the test with that output in the message.

#### Step 3: GREEN

1. `unified-cache.hpp`, next to `max_tensor_bytes` (`:1209`):

```cpp
    // Largest tensor that can be copied through a host zone: max_tensor_bytes, leaving out a gather-only
    // tensor every one of whose entries stays on the host. The CPU reads such a table in place, so no
    // staging, readback or fallback copy of it exists to hold.
    size_t                       max_copied_tensor_bytes  = 0;
```

2. `unified-cache.cpp`, `populate_host_zone_sizing`. Keep `:35075-35079` and the assert at `:35091`.
   After `plan.entries` is known (the entries loop at `:35222`), compute the set of left-out names and
   both maxima. Basic loops, no new templates:

```cpp
    // A gather-only tensor the plan keeps wholly on the host is read in place; nothing copies it.
    std::unordered_set<std::string> host_gather_only;
    for (const auto & item : tensor_inventory) {
        if (!item.get_rows_only) {
            continue;
        }
        bool has_entry = false;
        bool any_device = false;
        for (const auto & entry : plan.entries) {
            if (entry.tensor_name == item.name) {
                has_entry  = true;
                any_device = any_device || entry.on_device;
            }
        }
        if (has_entry && !any_device) {
            host_gather_only.insert(item.name);
        }
    }
    plan.max_copied_tensor_bytes = 0;
    for (const auto & item : tensor_inventory) {
        if (host_gather_only.count(item.name) == 0) {
            plan.max_copied_tensor_bytes = std::max(plan.max_copied_tensor_bytes, item.size);
        }
    }
    for (const auto & entry : plan.entries) {
        if (host_gather_only.count(entry.tensor_name) != 0) {
            continue;
        }
        plan.max_staging_pair_bytes =
            std::max(plan.max_staging_pair_bytes, entry.src_size + std::max(entry.dst_size, entry.src_size));
    }
```

   `entry.tensor_name` stands for the entry's real name field (census 1). The inner loop is
   `gather-only tensors x entries`; gather-only tensors are a handful per model, so it is not hot. If a
   gather-only tensor has no entry at all it stays counted (conservative).

3. STAGING (`:35682`): `twice_max_tensor` is built from `plan.max_copied_tensor_bytes`.
4. SCRATCH fold (`:35723`): `{ plan.max_copied_tensor_bytes, k_scratch_headroom, moe_host_scratch }`.
5. Commit floor (`unified-cache.hpp:2255`): `plan.max_copied_tensor_bytes` in place of
   `plan.max_tensor_bytes`. Update the comment at `:1070`.
6. Mirrors: `host-ledger-commit-oracle.hpp:46-47` and `test-unified-runtime-alloc.cpp:984-988` read the
   new field.
7. One log line: the existing `[SYCL-PLAN] Host staging zone` line gains
   `copied_max=%.1f MB global_max=%.1f MB`.

Do **not** change `:35245`, `:35456`, `:35461`, the zone-sizing TU, or the assert at `:35091`.

#### Step 4: follows

- `tests/test-sycl-host-role-routing-source.py`: the `SINGLE_TENSOR` tuple (`:654`), the fold literal
  `_FOLD_FOUR` (`:2175`) and its mutant (`:2672`) name `max_tensor_bytes`. Update them to the new field
  and add one mutant: the fold reading `plan.max_tensor_bytes` again must fail the pin. Run the pin plain
  and with `--self-test`; state the mutant count before and after.
- `docs/backend/sycl-memory-design.md`, section "Path-scoped zone sizing": one paragraph stating the
  rule, why "gather-only" alone is not the test (a device-resident embedding is staged), and that
  under-sizing STAGING aborts because the pool forbids zone growth.
- Run `python3 tests/test-sycl-gate-registration-audit.py` plain and `--self-test`, and
  `python3 tests/test-sycl-static-storage-audit.py --check`.
- Build targets to request from the lead: `ggml-sycl`, `test-sycl-host-weight-zone-terms`,
  `test-sycl-host-ledger`, `test-unified-runtime-alloc`, `test-zone-sizing`, `llama-bench`,
  `llama-cli`. Also every other target that compiles `unified-cache.cpp`
  (`cat build/compile_commands.json | grep -c unified-cache.cpp` and name them), since a header field
  changes `placement_plan`'s layout.

#### Gotchas

- `placement_plan` is a struct other TUs copy by value. Add the field next to `max_tensor_bytes`; do not
  reorder existing fields. Every target that includes `unified-cache.hpp` must rebuild.
- `test-sycl-lazy-residency-plan.cpp:259-295` asserts lazy tables never move `max_tensor_bytes`,
  `max_staging_pair_bytes`, STAGING or SCRATCH. It must stay green untouched.
- `test-zone-sizing.cpp:110` asserts scoped maxima `<= any_tensor`. `max_copied_tensor_bytes` is not a
  member of `path_scoped_maxima`; leave that struct alone.
- STAGING allocations pass `forbid_host_zone_growth` (`common.hpp:3265`). An under-size is an abort at
  load, not a slow path. That is why the rule is the narrow one.
- The pair term under-states bulk expert staging (entries are per expert, the staged buffer is the whole
  expert tensor). The inventory maximum is what covers it. Keep both terms.
- Comments give the reason. No ticket ids, no review rounds, no new environment flag.
- Never run a device binary. Host planner tests only, after the lead's GO.

---

## End-to-End Validation (on the user's machine): lead only

Run from the task worktree after spec PASS and a build at the reviewed sha. One at a time, `GPU.lock`,
VRAM preflight, `GGML_SYCL_TUNING_CACHE=0`, guard on `Shmem > 100 GB`, sample before and 5 s after.

| # | run | card | expected |
|---|---|---|---|
| 1 | `llama-bench -m <exps-IQ4_NL> -p 2048 -n 0 -r 1 -v`, default flags | B70 | staging line `per_inflight` about 1,290 MB (103,762.3 before); `Pinned pool pre_allocate` about 2.6 GB (156,269.5 MB before); rc 0; pp2048 not below 22 t/s; min MemAvailable well above the 38.7 GB of the old run |
| 2 | Qwen count gate, `llama-cli`, default flags | B70 | `grep -cx '1, 2, 3, 4, 5'` = 1 |
| 3 | same bench with `-lm none` | B70 | late-plan staging line unchanged at 1,290.3 / 1,307.4 MB; early-plan line drops from 103,762.3; pp2048 about 200 t/s |
| 4 | Mistral 7B Q4_0 completion gate | B50 | output starts `1, 2, 3, 4, 5, 6, 7, 8, 9, 10`; staging line byte-identical to a run of the base build (control: no other model moves) |
| 5 | GPT-OSS 20B chat gate | B50 | `grep -cx` = 1; staging line byte-identical to the base build |
| 6 | kernel log | | no `GT reset`, `guc_id`, `CAT error` |

Controls for runs 4 and 5 come from `wt-integ-th32` rebuilt at its head, run first.

**Observed success:** run 1 prints a staging total near 1.3 GB and the load leaves more than 150 GB of
host memory that the old load pinned; runs 4 and 5 show identical staging lines on base and task builds.
