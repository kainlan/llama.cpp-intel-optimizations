# fmcp addendum A10 (revision 2): a mapped load copies its routed host experts into pinned memory for the window ring

> **Execution:** `team-driven-development` in Claude Code. Addendum to
> `docs/plans/2026-11-13-fmcp-plan-1-qwen38-b70.md`, under epic llama.cpp-fmcp, ticket llama.cpp-gzzq.
> **Revision 2, 2026-11-14.** It replaces the text approved on 2026-11-14. That text pinned the experts through the
> loader, and its cost figure was wrong: the implementer's census stopped it before any code. This revision was
> written after addendum A12 merged, a re-measure, a code census and a spike.
> **Not approved for spawning until the owner has reviewed it.**

**Goal:** on a normal load (`-lm auto`, which maps the file), the host-executed routed experts are held once, in
pinned memory, when the device has a window ring and the host budget holds them. Prompts then take the ring without
`-lm none`. When they do not fit, they stay mapped, and the exec policy stops printing `ring` for layers it cannot
stream.

**Architecture:** the loader does not change. Every tensor stays in a `SYCL_Mapped` buffer. The single-device plan
marks the routed experts `PINNED` before it sizes the ring's room, and keeps that answer when room is held and the
host plan fits its budget. All of them or none. The preload, which today serves a mapped expert in place, reads the
plan's answer and copies a `PINNED` expert from the mapping into a WEIGHT-zone allocation. That is the copy a
`-lm none` load already makes. The ring's plan index stages a host expert only when its backing is pinned memory, so
a plan that left the experts mapped sizes no ring and runs those layers on the CPU.

**Tech stack:** C++ (ggml-sycl planner, preload, ring index), three C++ tests, one Python source pin, two docs.

**Test infrastructure:**
- `tests/test-sycl-layout-choice.cpp`, namespace `ring_room_case` (`:1564`): the real planner on a small MoE
  inventory with a staged type and a planned weight slot. It is the only test whose plans hold ring room. Its ctest
  registration pins `level_zero:1`, so the lead runs it.
- `ggml/src/ggml-sycl/tests/test-sycl-host-backing-plan.cpp`: the real planner on the Qwen3.8 inventory under a
  mocked device table. Its plans hold **no** ring room (`kv_for` leaves `moe_stream_staged_types` 0). Lead runs it.
- `ggml/src/ggml-sycl/tests/test-moe-stream-plan.cpp`: host test of the ring's plan index.
- `tests/test-sycl-host-backing-source.py`: source pins, pytest style (`def test_...`).

---

## Evidence (lead runs, 2026-11-14, B70 `level_zero:0`, exps-IQ4_NL pack, driver 26.44, tuning cache off)

All runs are `llama-bench -p 2048 -n 0`, one arm at a time. The host load was 10 to 44 and no builder ran. The
figures are not baselines. The full records are on llama.cpp-gzzq.

**The first look no longer describes the tree.** At `e47ab5a60` the mapped arm gave 11.48 t/s and `-lm none` gave
200.19 t/s. A9, A11 and A12 have merged since.

**Re-measure at `integ/th32` `20e4361e0`, quiet (`-r 5 -o json`), order mapped, pinned, pinned, mapped:**

| Arm | pp2048 t/s | Wall | Min `MemAvailable` |
|-----|-----------:|-----:|-------------------:|
| mapped | 79.93 ± 0.92 | 5:27 | 176.5 GB |
| `-lm none` | 159.64 ± 2.85 | 12:07 | 51.3 GB |
| `-lm none` | 159.55 ± 3.40 | 5:54 | 50.7 GB |
| mapped, after the pinned arms | 50.31 ± 3.10 | 11:28 | 173.0 GB |

- The mapped arm also gave 112.41 and 109.14 t/s at `-r 3` in A12's end-to-end, an hour earlier.
- A mapped arm that follows a pinned arm loads in 11 to 13 minutes and is slow for the whole run. The simplest
  reading is that the pinned allocation pushed the model's pages out of the file cache. It is not proven.

**What each load holds** (`-v` runs of the same pair):

| | mapped | `-lm none` |
|---|---|---|
| loader buffers | four `SYCL_Mapped` | one `SYCL_Host`, 70,014.66 MiB |
| host experts | `mapped=48723.9 MB` | `pinned/host_cpu=48721.3 MB` |
| pinned pool at load end | 6 GB | 124 GB (`WEIGHT=115.95/117.66`) |
| ledger | `host_expert_copy_cumulative=0.00` | `sycl_host_buffers=68.37 host_expert_copy_cumulative=47.58` |
| `[MOE-EXEC]` summary | 576 ring, 0 MiB streamed, 76,270 misses | 576 ring, 224,624 MiB streamed, 0 misses |

So `-lm none` holds each host expert twice: once in the loader's whole-model buffer and once as the preload's copy.

**Spike** (evidence branch `spike/a10-copy-pin` at `d7d30d127`, 11 lines, env-gated, never to be merged). With the
variable set, the plan answers `PINNED` for a routed expert on a mapping load and the preload copies it. One binary,
two arms by environment, quiet, order pin, map, map, pin:

| Arm | pp2048 t/s | Wall | Min `MemAvailable` |
|-----|-----------:|-----:|-------------------:|
| copy-pin | 160.79 ± 3.10 | 5:04 | 123.9 GB |
| mapped, after a pinned arm | 47.81 ± 0.69 | 13:06 | 173.5 GB |
| mapped | 81.92 ± 0.88 | 5:19 | 175.4 GB |
| copy-pin | 161.21 ± 3.74 | 4:53 | 124.6 GB |

- The copy-pin load keeps four `SYCL_Mapped` buffers and no `SYCL_Host` buffer.
- Load end: `pinned pool: 54.00 GB ... WEIGHT=47.58/47.66 ... sycl_host_buffers=0.00
  host_expert_copy_cumulative=47.58`.
- The plan's ledger line: `host_budget=162056695808 host_planned_bytes=55836673600 weight=48895977472`. The plan
  uses 55.8 GB of a 150.9 GiB budget.
- `[SYCL-PLAN] MoE window ring room on device 0: held 119.5 MB` and `DMA staging pool: 900.0 MB`. The pool is
  device memory. A mapped plan sizes it 0.
- `[MOE-EXEC] summary: 288 layer-ubatches ring, 0 cpu, 0 forced, 111931 MiB streamed, 2316 windows, 288 exposed,
  0 misses` on the `-r 1 -v` run, which gave 166.22 t/s.
- The count gate under copy-pin (`llama-cli`, default flags): rc 0, `grep -cx '1, 2, 3, 4, 5'` = 1.

**Reading.** Copy-pin gives the `-lm none` throughput with a 54 GB pinned pool in place of 124 GB. It is 2.0 times
the warm mapped arm and 3.4 times the cold one.

**Not measured:** the B50, GPT-OSS, a load that does not fit, and the copy's cost in load time. The count-gate run
took 5:07 under copy-pin against 2:55 by default in A12's end-to-end, one sample each on different builds.

## What the code does today (read at `20e4361e0`; verify each anchor before editing)

**Plan, single device (`compute_placement_plan`, `unified-cache.cpp:36520`).**
- `placement_host_backing_when_host` (`:35067`) gives every routed expert `FILE_MAPPED` on a mapping load. It runs
  when the entry is made (`:36643`), before any expert is placed.
- `hold_moe_stream_ring_room` (`:36407`) is called once, at `:36948`. It returns with no room when
  `kv_info.moe_stream_staged_types == 0`, when the planned weight slot is 0, or when there is no MoE group. It
  sizes the room with `moe_stream_plan_ring_geometry`, which builds the ring's plan index.
- The expert pack follows. A group left on the host is charged at `:37001`, at its entries' `src_size`, each by its
  own backing. No routed expert is charged to a host term before that line.
- Then, at `:37065-37067`: `placement_mark_device_entries_file_mapped`, `populate_host_zone_sizing`,
  `record_host_ledger_plan`.
- `populate_host_zone_sizing` (`:35094`) assigns the plan's fields and never accumulates into them. It logs its
  `[SYCL-PLAN]` lines on every call. It holds the only reader of `has_pinned_backing()` (`:35568`), which sizes
  the DMA staging pool.
- `record_host_ledger_plan` (`:35000`) sets `plan.host_budget` and `plan.host_zone_terms`, then calls
  `placement_plan_decide_lazy_residency` (`:34777`). That pass clears the lazy tables first, decides them against
  what the budget leaves after the zone terms, and sets `plan.host_planned_bytes`. It logs WARN lines and counts
  its ring coverage reads. A plan over budget logs a would-refuse WARN and is kept.

**Plan, multi device (`compute_multi_device_plan`, `:37989`).** It never calls `hold_moe_stream_ring_room`. Its
host charges use other size bases (`:38258`, `:38509`). The same three calls end it at `:39115-39117`.

**Plan helpers (`unified-cache.hpp`).**
- `host_backing_sizes_weight_zone` (`:1515`) is true for `PINNED` and `HOST_CPU` only.
- `charge_host_weight` (`:1529`) and `uncharge_host_weight` (`:1549`) are exact inverses per backing.
- `placement_entry::host_backing` defaults to `PINNED` (`:449`).
- `lookup_expert_placement` (`:1701`) returns an `expert_placement_result` (`:378`) with no `host_backing`.

**Preload (`ggml-sycl.cpp`).** The lambda `register_host_aos_expert` (`:44477`) has one caller, at `:44822`, where
`placement` from `lookup_expert_placement` (`:44752`) is in scope.
- `served_in_place = !tensor_source_is_device && ggml_sycl_mapped_buffer_owner(tensor->buffer, &host_owner)`
  (`:44508`). In place, the entry points at the mapping and holds the mapping's owner.
- Otherwise `ggml_sycl_host_expert_alloc` allocates in the WEIGHT zone, `mem_copy` fills it from the source, and
  `host_expert_copy_cumulative_bytes` counts it (`:44534`).
- When that allocation fails and the source is host memory, the branch at `:44535` registers the source pointer
  with **no** owner.

**Ring (`moe-stream-plan.hpp`).**
- `moe_stream_plan_index_build` (`:234`) lists every host routed-expert entry as an item and reads no backing. An
  item the executor claims stages `AOS_ONLY`; any other stages `NONE`.
- `moe_stream_layer_claimed` (`:387`) is false when any item of the layer stages `NONE`.
- `moe_stream_owner_streamable` (`:505`) accepts only a `HOST_PINNED` `WEIGHT` owner in a host zone of the device.
  A mapped expert has none, so the stager skips it and the window counts a miss.

**Loader.** `ggml_backend_sycl_planned_host_backing` (`ggml-sycl.cpp:54713`) answers `FILE_MAPPED` for a
routed-expert tensor on a mapping load. **This revision leaves it and the loader alone.**

**Tests that stay true.** `test-sycl-host-weight-zone-terms.cpp` arm 3a (no weight answers a backing that keeps
`SYCL_Host` on a mapping plan) and arm 3b (the WEIGHT zone does not grow for the mapped buffer) both hold, because
the loader keeps mapping. Arm 3b builds the buffer and binds no tensor, so no existing device test runs the preload
over a mapped buffer. The lead's end-to-end runs 4 and 5 are the preload's device test.

## Decisions this addendum takes (owner to confirm by approving it)

1. **The copy is made from the mapping, by the preload.** The loader's answer for a routed-expert tensor stays
   `FILE_MAPPED`. No whole-model host buffer exists on a mapping load.
2. **All or none per plan.** Either every host-executed routed expert of the plan is pinned, or none is.
3. **The fit rule is the ledger's own figure.** The plan is sized with the experts pinned. It is kept when ring
   room is held, `host_planned_bytes` is at most the host budget, and no lazy table refuses. Otherwise the experts
   go back to `FILE_MAPPED`, the host zones are sized again, and one WARN line says why. The ledger lines print
   twice in that case only.
4. **Pinned experts come before lazy tables.** The lazy pass already decides tables against what the zone terms
   leave. A pinned plan can therefore read a table from storage that a mapped plan would have held in RAM.
5. **One opt-out, default on:** `GGML_SYCL_MOE_PIN_HOST_EXPERTS=0` keeps the experts mapped. It is read when the
   plan is made. It lets the end-to-end compare the two paths in one binary, and it is the user's way back.
6. **`-lm mlock` and `mmap+mlock` keep their experts mapped.** The user asked for a locked mapping; pinning as
   well would hold the bytes twice. With task A10.2 those layers run on the CPU and print `cpu`, not `ring`.
7. **A mapped host expert is never ring-staged.** A plan that left the experts mapped sizes no ring and holds no
   ring room. Today such a load holds 119.5 MB of device room for a ring that streams nothing.
8. **Multi-device plans are unchanged.** They hold no ring room today. Holding it per device is its own change.
9. **The mapping's pages are left to the kernel.** After the copy they are clean file cache. No `madvise` call is
   added here.
10. **`-lm none` is unchanged.** Its double holding is a separate defect and gets its own ticket.
11. **The storage path is separate.** Reading experts from the NVMe straight into pinned windows, for packs whose
    experts do not fit in RAM, is the owner's second request. It gets its own spike and addendum after this one.

---

## Team Topology

**Implementers:** 1 (Opus: ownership and placement code), the three tasks in order, in one worktree.
**Reviewers:** spec then quality, fresh each round, Opus. Each task is reviewed on its own commit range.
**Merge:** A10.1 changes no behaviour and merges when it passes. A10.2 and A10.3 merge together, after the lead's
end-to-end.

| Track | Tasks | Description |
|-------|-------|-------------|
| F | A10.1, A10.2, A10.3 | the preload reads the plan; the ring index reads the backing; the plan pins |

```dot
digraph dependencies { rankdir=LR; "A10.1 preload reads the plan" -> "A10.2 ring index reads the backing" -> "A10.3 the plan pins"; }
```

### File Ownership Map

| File | Task | Change |
|------|------|--------|
| `ggml/src/ggml-sycl/unified-cache.hpp` | A10.1 | `expert_placement_result` carries `host_backing` |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | A10.1 | `register_host_aos_expert` and its one call site |
| `ggml/src/ggml-sycl/tests/test-sycl-host-backing-plan.cpp` | A10.1 | one check in arms 1 and 2 |
| `tests/test-sycl-host-backing-source.py` | A10.1, A10.3 | one clause each; A10.3 also updates the pinned-site list |
| `ggml/src/ggml-sycl/moe-stream-plan.hpp` | A10.2 | `moe_stream_plan_index_build` reads `host_backing` |
| `ggml/src/ggml-sycl/tests/test-moe-stream-plan.cpp` | A10.2 | one new test function |
| `ggml/src/ggml-sycl/unified-cache.cpp` | A10.3 | three helpers; one call before the ring room; one call ends each builder |
| `tests/test-sycl-layout-choice.cpp` | A10.3 | one new test function in `ring_room_case` |
| `docs/backend/sycl-env-vars.md` | A10.3 | one row |
| `docs/design/sycl-mmap-host-tier.md` | A10.3 | §4.4: the expert row (`:1188`) and the sentence at `:1195` |

Run `git merge-tree` against `integ/th32` each round.

**Rules for every task.**
- No device binary, no model load, no `ctest`: the lead runs those.
- Build only on the lead's literal GO, bound to a sha. Syntax-only compile every changed `.cpp` after the last
  edit, with its own command from the worktree's `compile_commands.json`.
- Never amend a reported commit. Path-scoped commits. One trailer: `Co-Authored-By: Claude Opus 5.5
  <noreply@anthropic.com>`.
- Comments give reasons. No ticket ids, task numbers or review rounds in code.
- `clang-format-19 --dry-run -Werror` on the lines you wrote only. Comment lines are 120 columns or fewer.
- codescout is blind inside `ggml-sycl.cpp`. Search it with `cat <file> | grep -n`.
- STOP and message the lead when a census answer contradicts this addendum. Do not design around it.

---

### Task A10.1: the preload serves a host expert in place only when the plan left it mapped

**Track:** F. **Depends on:** none. Branch from `integ/th32` at `20e4361e0` or later.

**Description:** the preload decides in place or copy from the buffer's kind alone. This task makes it read the
plan's backing for that expert. No plan pins a mapped expert yet, so nothing changes at run time.

**Acceptance criteria:**

- [ ] `lookup_expert_placement` returns the entry's `host_backing`.
- [ ] `register_host_aos_expert` serves in place only when the plan's backing holds no WEIGHT-zone block.
- [ ] A failed pinned allocation over a `SYCL_Mapped` source registers the mapping **with its owner** and warns
      once per tensor.
- [ ] Every existing test and pin is green. Behaviour on today's plans is unchanged.

#### Step 1: census

1. `cat ggml/src/ggml-sycl/ggml-sycl.cpp | grep -n 'register_host_aos_expert('`: confirm one caller.
2. Read `ggml_sycl_mapped_buffer_owner`. Does it write `*owner` when it returns false? The code below assumes it
   may, and clears the handle.
3. Where is `logged_host_layout` declared? The new once-per-tensor flag goes beside it.

#### Step 2: RED

In `test-sycl-host-backing-plan.cpp`, a helper after `take_census` (`:251`):

```cpp
// Routed-expert entries whose lookup result does not carry the entry's own backing.
size_t lookup_backing_mismatches(const placement_plan & plan) {
    size_t n = 0;
    for (const placement_entry & e : plan.entries) {
        if (e.expert_id < 0 || e.layer_id < 0) {
            continue;
        }
        const auto found = plan.lookup_expert_placement(e.layer_id, e.expert_id, e.expert_role);
        n += !found.found() || found.host_backing != e.host_backing;
    }
    return n;
}
```

- In `arm1_mapping_load`, after the census checks:
  `check_eq("routed experts whose lookup result does not carry their backing", lookup_backing_mismatches(plan), 0);`
- The same line in `arm2_copying_load`.
- RED evidence: the syntax-only compile of the test TU fails on `host_backing` not being a member of
  `expert_placement_result`. Report the error lines.

In `tests/test-sycl-host-backing-source.py`, one test in the file's style, built on its `read`, `strip_comments`
and `balanced` helpers:

```python
def test_the_preload_serves_in_place_only_what_the_plan_left_mapped():
    src = strip_comments(read("ggml/src/ggml-sycl/ggml-sycl.cpp"))
    start = src.index("auto register_host_aos_expert = [&](")
    body = balanced(src, src.index("{", start))
    cond = body[body.index("served_in_place ="):]
    cond = cond[:cond.index(";")]
    assert "host_backing_sizes_weight_zone(planned_backing)" in cond, cond
    assert cond.index("host_backing_sizes_weight_zone") < cond.index("ggml_sycl_mapped_buffer_owner"), cond
    assert "getenv" not in body
```

Run it and show it failing on the first assertion.

**Commit:** `test(sycl): the preload reads the plan's backing for a host expert (RED)`, staging the two test files.

#### Step 3: GREEN

`unified-cache.hpp`, in `expert_placement_result` (`:378`), after `on_device`:

```cpp
    // The plan's host backing for this expert. For a host-executed expert, a backing that sizes the WEIGHT zone means
    // the preload holds it there; FILE_MAPPED means the CPU reads the mapping in place.
    placement_host_backing host_backing = placement_host_backing::PINNED;
```

and in `lookup_expert_placement` (`:1701`), beside the `src_size` assignment (`:1717`):
`result.host_backing = e.host_backing;`. If `placement_host_backing` is declared after the struct, move nothing:
report it and STOP.

`ggml-sycl.cpp`, the lambda's signature (`:44477`):

```cpp
                    auto register_host_aos_expert = [&](int64_t e, const uint8_t * expert_aos,
                                                        ggml_sycl::placement_host_backing planned_backing,
                                                        const char * reason) -> bool {
```

the decision (`:44503-44509`):

```cpp
                        // A SYCL_Mapped source is served in place when the plan left the expert mapped: the CPU
                        // executes the file layout (host experts are AOS), and the entry holds the mapping's owner.
                        // When the plan gave it a WEIGHT-zone block, the window ring streams it, and the ring copies
                        // only from pinned memory, so it is copied out of the mapping as any other source is.
                        ggml_sycl::mem_handle host_owner;
                        const bool            served_in_place =
                            !tensor_source_is_device &&
                            !ggml_sycl::placement_plan::host_backing_sizes_weight_zone(planned_backing) &&
                            ggml_sycl_mapped_buffer_owner(tensor->buffer, &host_owner);
```

the allocation-failure branch (`:44535-44538`):

```cpp
                        } else if (!tensor_source_is_device) {
                            // No pinned room: the source is served in place. A SYCL_Mapped source keeps its mapping's
                            // owner, as a planned in-place expert does; the window ring cannot stream this expert, so
                            // the load says so once per tensor. Any other host source is registered unowned.
                            host_owner = {};
                            if (ggml_sycl_mapped_buffer_owner(tensor->buffer, &host_owner)) {
                                if (!logged_pin_alloc_failed) {
                                    GGML_LOG_WARN("[MOE-LAYOUT] tensor=%s device=%d expert=%lld: no pinned room for "
                                                  "%zu bytes the plan pinned; served from the mapping, and the "
                                                  "window ring will not stream it\n",
                                                  tensor->name ? tensor->name : "?", device,
                                                  static_cast<long long>(e), expert_size);
                                    logged_pin_alloc_failed = true;
                                }
                            } else {
                                host_owner = {};
                            }
                            host_ptr = const_cast<uint8_t *>(expert_aos);
                        } else {
```

- Declare `bool logged_pin_alloc_failed = false;` beside `logged_host_layout`.
- The caller (`:44822`): `register_host_aos_expert(e, expert_aos, placement.host_backing, "planner-host")`.

Run the Python pin: green. Syntax-only compile `ggml-sycl.cpp` and the test TU.

**Commit:** `sycl: the preload serves a host expert in place only when the plan left it mapped`.

#### Gotchas

- **Operand order matters.** The plan test comes before `ggml_sycl_mapped_buffer_owner`, so a pinned expert never
  takes the mapping's owner into `host_owner` before the allocation replaces it.
- On a `-lm none` load every entry is `PINNED` and the source is not `SYCL_Mapped`. Both old and new conditions
  are false there.
- Do not touch the loader or `ggml_backend_sycl_planned_host_backing`.
- **Build** only on the lead's GO: targets `test-sycl-host-backing-plan llama-bench llama-cli`.
- Lead's check before review: `test-sycl-host-backing-plan` rc 0, and the count gate on the B70 with default
  flags still shows `mapped=` host experts and passes.

---

### Task A10.2: the ring's plan index stages a host expert only from pinned memory

**Track:** F. **Depends on:** A10.1.

**Description:** the index lists a mapped host expert as streamable, so the planner sizes a ring for it, the
policy prints `ring`, and the stager then skips it. This task makes the index read the backing. A plan with mapped
host experts then sizes no ring, holds no room, and runs those layers on the CPU.

**Acceptance criteria:**

- [ ] A host-planned entry whose backing does not size the WEIGHT zone stages `NONE`.
- [ ] A layer with such an entry is not claimed (`moe_stream_layer_claimed` false), and a plan made only of such
      layers has ring geometry 0.
- [ ] Entries lent from the device (`as_host`) are read exactly as today.
- [ ] Every existing case in `test-moe-stream-plan.cpp` is green unchanged: its entries default to `PINNED`.

#### Step 1: census

1. Every caller of `moe_stream_plan_index_build` and `moe_stream_plan_ring_geometry`. For each, say whether the
   plan it passes can hold `FILE_MAPPED` host experts, and what the caller does with a geometry of 0.
2. `begin_layer` in `moe-stream-plan.cpp`: name the branch a host layer takes when `moe_stream_layer_claimed` is
   false, and the `[MOE-EXEC]` counter it lands in.
3. Source pins under `tests/` naming `moe_stream_plan_index_build`, `moe_stream_layer_claimed` or
   `moe_stream_owner_streamable`. Run each and its `--self-test` before editing. Report the rc.

#### Step 2: RED

In `test-moe-stream-plan.cpp`, after `test_window_sizing`, and called from `main` beside it:

```cpp
// The ring copies only from pinned memory, so a host expert the plan left mapped stages nothing: its layer is not
// claimed and takes no slot. A lent device triplet is read as before.
void test_mapped_host_experts_are_not_staged() {
    ggml_sycl::moe_stream_window_hints hints;
    hints.weight_slot_bytes = 10 * kSrc;

    placement_plan mapped = make_typed_plan();
    for (ggml_sycl::placement_entry & e : mapped.entries) {
        if (e.expert_id >= 0 && !e.on_device) {
            e.host_backing = ggml_sycl::placement_host_backing::FILE_MAPPED;
        }
    }
    auto idx = ggml_sycl::moe_stream_plan_index_build(mapped, 0, nullptr, claims_all, hints);
    CHECK(idx.role_items_max == 0);
    CHECK(idx.window_slots == 0);
    CHECK(idx.layers.count(0) == 1 && !ggml_sycl::moe_stream_layer_claimed(idx.layers.at(0)));
    CHECK(ggml_sycl::moe_stream_plan_ring_geometry(mapped, 0, nullptr, claims_all, hints).bytes == 0);

    // One mapped expert in a pinned layer leaves the layer unclaimed, and the ring is sized over the others.
    placement_plan mixed = make_typed_plan();
    for (ggml_sycl::placement_entry & e : mixed.entries) {
        if (e.layer_id == 0 && e.expert_id == 1) {
            e.host_backing = ggml_sycl::placement_host_backing::FILE_MAPPED;
        }
    }
    idx = ggml_sycl::moe_stream_plan_index_build(mixed, 0, nullptr, claims_all, hints);
    CHECK(!ggml_sycl::moe_stream_layer_claimed(idx.layers.at(0)));
    CHECK(ggml_sycl::moe_stream_layer_claimed(idx.layers.at(2)));

    // The control: the same plan with every host expert pinned is claimed, as test_window_sizing sizes it.
    const placement_plan pinned = make_typed_plan();
    idx = ggml_sycl::moe_stream_plan_index_build(pinned, 0, nullptr, claims_all, hints);
    CHECK(idx.window_slots == 3);
    CHECK(ggml_sycl::moe_stream_layer_claimed(idx.layers.at(0)));
}
```

- Layer 1 of `make_plan` holds a fused `GATE_UP` host entry and is never claimed. The test uses layers 0 and 2.
- If `idx.layers` is not a map with `count` and `at`, use the container's own lookup and say so.
- RED evidence: the lead builds `test-moe-stream-plan` at the RED commit and runs it. Report the `CHECK failed
  line` lines you expect: the four checks on `mapped` and the first on `mixed`.

**Commit:** `test(sycl): a mapped host expert is not ring-staged (RED)`.

#### Step 3: GREEN

In `moe_stream_plan_index_build`, where the claim is asked (`if (library_claims != nullptr &&
moe_stream_item_aos_fits(item, SIZE_MAX))`):

```cpp
        // The ring copies a host expert only out of pinned memory (moe_stream_owner_streamable). A host-planned
        // expert whose backing holds no WEIGHT-zone block is read in place from its mapping, so it stages NONE and
        // its layer runs on the CPU. A lent device triplet is not asked: its host copy is the loan's business.
        const bool mapped_host = !e.on_device && !placement_plan::host_backing_sizes_weight_zone(e.host_backing);
        if (!mapped_host && library_claims != nullptr && moe_stream_item_aos_fits(item, SIZE_MAX)) {
```

Nothing else changes. Run the pins from the census again.

**Commit:** `sycl: the window ring's plan index stages a host expert only from pinned memory`.

#### Gotchas

- **This changes today's mapped load.** It holds no ring room and its host layers print `cpu`. That is the path
  decision 7 names, and the lead measures it as the opt-out arm. It does not merge before the end-to-end.
- **The planner sizes the ring's room through this index.** After this task a mapping load holds no room until
  A10.3 marks the experts pinned before the room is sized. That order is why A10.3 pins first and checks after.
- `tests/test-sycl-moe-stream-enqueue-source.py` pins the text of the owner check. This task does not edit that
  check.
- **Build** only on the lead's GO: targets `test-moe-stream-plan test-moe-stream-phase test-sycl-layout-choice
  llama-bench llama-cli`.

---

### Task A10.3: a mapping load pins its routed host experts when ring room is held and the host plan fits

**Track:** F. **Depends on:** A10.2.

**Description:** the single-device plan marks the routed experts `PINNED` before it sizes the ring's room, so the
room is sized for experts the ring can stream. The pack then charges each host group as pinned. When the plan is
finished, the pin is kept if room was held and the host plan fits; otherwise the experts go back to `FILE_MAPPED`.

**Acceptance criteria:**

- [ ] Mapping load, staged type, room held, budget fits: every host routed expert is `PINNED`, every device entry
      is `FILE_MAPPED`, `host_mapped_bytes` is 0, and `weight_host_bytes` is the host experts' `src_size` sum.
- [ ] The same load with a host budget between the mapped plan's and the pinned plan's `host_planned_bytes`:
      no host routed expert is `PINNED`, and the host zones are the mapped plan's.
- [ ] `GGML_SYCL_MOE_PIN_HOST_EXPERTS=0`, a locked mapping, or no staged type: no host routed expert is `PINNED`
      and no ring room is held.
- [ ] A `-lm none` plan and every multi-device plan are unchanged.
- [ ] The opt-out is in `docs/backend/sycl-env-vars.md`, and §4.4 of the design doc says what the code does.

#### Step 1: census

1. Between the entry loop and `:36948`, confirm no routed-expert entry is charged to a host term. Use
   `awk 'NR>=36520 && NR<=36948 && /charge_host_weight/' ggml/src/ggml-sycl/unified-cache.cpp` and read each hit.
   The addendum's reading: `:36770`, `:36829` and `:36862` charge dense weights only.
2. Is `:36948` reached on every single-device plan with MoE groups? Name the conditions around it.
3. The load plans more than once (the spike's log shows two `[HOST-LEDGER]` lines per load with one budget).
   Which plan does the preload read, and which does the ring's runtime index read? Can the two differ on the pin?
   If they can, say what each consumer then does. STOP if a mismatch can hand the stager a mapped expert as
   streamable or leave a pinned copy unowned.
4. `tests/test-sycl-host-backing-source.py`: read `test_the_planner_sites_are_the_pinned_list` (`:168`) and
   `test_only_the_helpers_write_the_host_terms` (`:146`). Say what each requires of a new site that assigns
   `PINNED`.
5. Six source pins name the three closing calls: `test-sycl-host-zone-config-source-contract.py`,
   `test-sycl-lazy-plan-slot-source.py`, `test-sycl-mapped-loader-source.py`, `test-sycl-plan-dry-run-source.py`,
   `test-sycl-moe-control-plan-contract.py`, `test-sycl-nonfa-attn-scratch-guard-source.py`. Run each before
   editing. Report which pin the calls' position inside a builder.
6. How does `ggml_check_sycl`'s non-default-settings line learn of a variable, and does
   `tests/test-sycl-env-report.cpp` list them? The opt-out follows that pattern.

#### Step 2: RED

In `tests/test-sycl-layout-choice.cpp`, inside `namespace ring_room_case`, before its closing brace (`:1681`):

```cpp
struct backing_count {
    size_t host         = 0;  // host-planned routed-expert entries
    size_t host_pinned  = 0;  // of those, PINNED
    size_t host_src     = 0;  // their src_size sum
    size_t device_other = 0;  // device-planned routed-expert entries that are not FILE_MAPPED
};

static backing_count count_backings(const ggml_sycl::placement_plan & plan) {
    backing_count c;
    for (const auto & e : plan.entries) {
        if (e.expert_id < 0) {
            continue;
        }
        if (e.on_device) {
            c.device_other += e.host_backing != ggml_sycl::placement_host_backing::FILE_MAPPED;
            continue;
        }
        c.host++;
        c.host_src += e.src_size;
        c.host_pinned += e.host_backing == ggml_sycl::placement_host_backing::PINNED;
    }
    return c;
}

// A load that maps files, as the loader's staging proc reports it.
static ggml_sycl::placement_plan plan_mapping(size_t   budget,
                                              size_t   host_budget,
                                              uint64_t staged_types = q4_0_bit,
                                              bool     locks        = false) {
    ggml_sycl::placement_kv_info kv = kv_info(staged_types, 0);
    kv.load_maps_files              = true;
    kv.load_locks_files             = locks;
    return ggml_sycl::compute_placement_plan(inventory(), budget, 0, kv, nullptr, n_exps, host_budget);
}
```

and after `run_moe_stream_ring_room_before_experts_test` (`:1686-1754`), called from `main` beside it (`:4153`):

```cpp
// A mapping load pins its routed host experts when the plan holds ring room and the host plan fits its budget, all of
// them or none. The opt-out, a budget the pinned plan exceeds, a locked mapping and a type nothing stages each leave
// them mapped.
static bool run_moe_stream_ring_pins_host_experts_test() {
    using namespace ring_room_case;
    bool ok = true;
    ggml_sycl::unified_cache_set_planned_pp_moe_onednn_scratch(0, weight_slot, 0, 0, 1);
    const size_t roomy_host = size_t(64) << 30;
    const auto   roomy      = plan_for(base + 2 * static_cast<size_t>(n_triplets) * ring_bytes, q4_0_bit);
    const size_t charge     = triplet_charge(roomy);
    const size_t budget     = base + 40 * charge;

    setenv("GGML_SYCL_MOE_PIN_HOST_EXPERTS", "0", 1);
    const auto          off = plan_mapping(budget, roomy_host);
    const backing_count oc  = count_backings(off);
    unsetenv("GGML_SYCL_MOE_PIN_HOST_EXPERTS");
    if (oc.host == 0 || oc.host_pinned != 0 || off.moe_stream_ring_reserve_bytes != 0 ||
        off.host_mapped_bytes != oc.host_src || off.weight_host_bytes != 0) {
        printf("FAIL: the opt-out keeps the host experts mapped with no ring room: host %zu, pinned %zu, room %zu, "
               "mapped %zu (want %zu), weight %zu\n",
               oc.host, oc.host_pinned, off.moe_stream_ring_reserve_bytes, off.host_mapped_bytes, oc.host_src,
               off.weight_host_bytes);
        ok = false;
    }

    const auto          pin = plan_mapping(budget, roomy_host);
    const backing_count pc  = count_backings(pin);
    if (pin.moe_stream_ring_reserve_bytes == 0 || pc.host == 0 || pc.host_pinned != pc.host || pc.device_other != 0 ||
        pin.host_mapped_bytes != 0 || pin.weight_host_bytes != pc.host_src ||
        pin.host_planned_bytes > pin.host_budget) {
        printf("FAIL: a mapping load with ring room and host budget pins every host expert: room %zu, host %zu, "
               "pinned %zu, device entries not mapped %zu, mapped %zu, weight %zu (want %zu), planned %zu of %zu\n",
               pin.moe_stream_ring_reserve_bytes, pc.host, pc.host_pinned, pc.device_other, pin.host_mapped_bytes,
               pin.weight_host_bytes, pc.host_src, pin.host_planned_bytes, pin.host_budget);
        ok = false;
    }
    if (pin.host_planned_bytes < off.host_planned_bytes + pc.host_src ||
        pin.host_zone_weight_bytes < off.host_zone_weight_bytes + pc.host_src) {
        printf("FAIL: the pinned plan's host figures hold the experts: planned %zu (mapped %zu), WEIGHT zone %zu "
               "(mapped %zu), experts %zu\n",
               pin.host_planned_bytes, off.host_planned_bytes, pin.host_zone_weight_bytes, off.host_zone_weight_bytes,
               pc.host_src);
        ok = false;
    }

    // A budget the mapped plan fits and the pinned plan does not: the experts go back, and the host zones are sized
    // for the mapped plan. The ring room, already taken out of the device pack, stays held.
    const size_t        tight = off.host_planned_bytes + pc.host_src / 2;
    const auto          back  = plan_mapping(budget, tight);
    const backing_count bc    = count_backings(back);
    if (bc.host == 0 || bc.host_pinned != 0 || back.weight_host_bytes != 0 || back.host_mapped_bytes != bc.host_src ||
        back.host_zone_weight_bytes != off.host_zone_weight_bytes || back.dma_staging_pool_bytes != 0 ||
        back.host_planned_bytes > back.host_budget) {
        printf("FAIL: a host budget the pinned plan exceeds leaves the experts mapped: host %zu, pinned %zu, weight "
               "%zu, mapped %zu (want %zu), WEIGHT zone %zu (want %zu), DMA pool %zu, planned %zu of %zu\n",
               bc.host, bc.host_pinned, back.weight_host_bytes, back.host_mapped_bytes, bc.host_src,
               back.host_zone_weight_bytes, off.host_zone_weight_bytes, back.dma_staging_pool_bytes,
               back.host_planned_bytes, back.host_budget);
        ok = false;
    }

    const backing_count lc = count_backings(plan_mapping(budget, roomy_host, q4_0_bit, /*locks=*/true));
    const backing_count nc = count_backings(plan_mapping(budget, roomy_host, /*staged_types=*/0));
    if (lc.host == 0 || lc.host_pinned != 0 || nc.host == 0 || nc.host_pinned != 0) {
        printf("FAIL: a locked mapping and a type nothing stages keep the host experts mapped: locked %zu of %zu "
               "pinned, unstaged %zu of %zu pinned\n",
               lc.host_pinned, lc.host, nc.host_pinned, nc.host);
        ok = false;
    }

    ggml_sycl::unified_cache_set_planned_pp_moe_onednn_scratch(0, 0, 0, 0, 0);
    if (ok) {
        printf("PASS: a mapping load pins its routed host experts when ring room is held and the host plan fits\n");
    }
    return ok;
}
```

- The `back` plan and the `off` plan can place different expert counts, because `back` held ring room. Each plan
  is checked against its own count. If `back.host_zone_weight_bytes` differs from `off`'s for that reason, report
  both figures and the cause; do not loosen the check without the lead.
- RED evidence: the lead builds `test-sycl-layout-choice` at the RED commit (on top of A10.2) and runs it. State
  which `FAIL:` lines you expect: the `pin` plan's two.

**Commit:** `test(sycl): a mapping load pins the routed host experts its ring will stream (RED)`.

#### Step 3: GREEN

`unified-cache.cpp`, beside `placement_mark_device_entries_file_mapped` (`:35083`):

```cpp
// GGML_SYCL_MOE_PIN_HOST_EXPERTS=0 keeps routed host experts mapped on a load that maps files. Read on every call, so
// a plan made after the variable changes sees it.
static bool placement_pin_host_experts_enabled() {
    const char * env = std::getenv("GGML_SYCL_MOE_PIN_HOST_EXPERTS");
    return env == nullptr || std::strcmp(env, "0") != 0;
}

// Marks every routed expert PINNED on a load that maps files, before the window ring's room is sized and before any
// expert is placed or charged. The ring copies a host expert only out of pinned memory, so its room is sized over
// pinned experts, and the pack then charges each host group as pinned. A locked mapping keeps its experts mapped: the
// load already holds them resident. Returns whether anything was marked; placement_plan_size_host_side keeps or
// withdraws the answer once the plan is whole.
static bool placement_pin_routed_experts_for_ring(placement_plan & plan, const placement_kv_info & kv_info) {
    if (!plan.host_backing_maps_files || plan.host_mapped_locked || kv_info.moe_stream_staged_types == 0 ||
        !placement_pin_host_experts_enabled()) {
        return false;
    }
    bool any = false;
    for (placement_entry & e : plan.entries) {
        if (e.expert_id >= 0 && e.host_backing == placement_host_backing::FILE_MAPPED) {
            e.host_backing = placement_host_backing::PINNED;
            any            = true;
        }
    }
    return any;
}

// Moves every host-executed routed expert from one backing to another with its charge, at src_size, the size the
// expert pack charges a host group. Returns the bytes moved.
static size_t placement_move_host_expert_backing(placement_plan &       plan,
                                                 placement_host_backing from,
                                                 placement_host_backing to) {
    size_t moved = 0;
    for (placement_entry & e : plan.entries) {
        if (e.expert_id >= 0 && !e.on_device && e.host_backing == from) {
            plan.uncharge_host_weight(from, e.src_size);
            e.host_backing = to;
            plan.charge_host_weight(to, e.src_size);
            moved += e.src_size;
        }
    }
    return moved;
}
```

After `record_host_ledger_plan`'s definition, the one sequence that ends both builders:

```cpp
// The host side of a finished placement: the device entries' staging source, the host zones, the ledger. Both builders
// end with it. `ring_pin` says the routed experts were marked PINNED for the window ring. That answer is kept when the
// plan holds ring room, its host bytes fit the budget and no lazy table refuses; otherwise the host experts go back to
// the mapping, all of them, and the zones and the ledger are sized again for the plan that results. The fit is the
// ledger's own figure, so a plan that keeps the pin never also logs the would-refuse line.
static void placement_plan_size_host_side(placement_plan &                           plan,
                                          const std::vector<placement_tensor_info> & tensor_inventory,
                                          int                                        n_experts,
                                          const placement_kv_info &                  kv_info,
                                          size_t                                     host_budget,
                                          const ring_lease *                         coverage_pin,
                                          bool                                       ring_pin) {
    const double mb = 1024.0 * 1024.0;
    if (ring_pin && plan.moe_stream_ring_reserve_bytes == 0) {
        // No ring exists for this plan, so nothing would stream the copies.
        placement_move_host_expert_backing(plan, placement_host_backing::PINNED, placement_host_backing::FILE_MAPPED);
        ring_pin = false;
    }
    placement_mark_device_entries_file_mapped(plan);
    populate_host_zone_sizing(plan, tensor_inventory, n_experts, kv_info.n_expert_used);
    record_host_ledger_plan(plan, kv_info, host_budget, coverage_pin);
    if (!ring_pin) {
        return;
    }
    const size_t pinned = plan.weight_host_bytes;
    if (plan.host_planned_bytes <= plan.host_budget && plan.lazy_refusal.empty()) {
        GGML_LOG_INFO("[PLACEMENT] routed host experts pinned for the window ring: %.1f MB, host plan %.1f MB of "
                      "%.1f MB\n",
                      pinned / mb, plan.host_planned_bytes / mb, plan.host_budget / mb);
        return;
    }
    GGML_LOG_WARN("[PLACEMENT] routed host experts stay mapped, so prompts run them on the CPU: pinning %.1f MB "
                  "plans %.1f MB of host memory against a budget of %.1f MB (lazy refusal: %s). The host zones "
                  "are sized again without them\n",
                  pinned / mb, plan.host_planned_bytes / mb, plan.host_budget / mb,
                  placement_lazy_refusal_text(plan).c_str());
    placement_move_host_expert_backing(plan, placement_host_backing::PINNED, placement_host_backing::FILE_MAPPED);
    populate_host_zone_sizing(plan, tensor_inventory, n_experts, kv_info.n_expert_used);
    record_host_ledger_plan(plan, kv_info, host_budget, coverage_pin);
}
```

In `compute_placement_plan`:
- after `placement_plan_copy_host_backing_fact(plan, kv_info);` (`:36535`): `bool ring_pin = false;`
- immediately before `hold_moe_stream_ring_room(...)` (`:36948`), inside the same comment block:
  `ring_pin = placement_pin_routed_experts_for_ring(plan, kv_info);`
- replace the three lines at `:37065-37067` with
  `placement_plan_size_host_side(plan, tensor_inventory, n_experts, kv_info, host_budget, coverage_pin, ring_pin);`

In `compute_multi_device_plan`, replace the three lines at `:39115-39117` with the same call and
`/*ring_pin=*/false`.

Then:
- Update the pins the census named, in this commit, and say which. A pin that listed the planner's `PINNED` sites
  gains `placement_pin_routed_experts_for_ring`.
- `docs/backend/sycl-env-vars.md`: a row for `GGML_SYCL_MOE_PIN_HOST_EXPERTS`, default ON: "`=0` keeps routed
  host experts mapped on a load that maps files; prompts then run them on the CPU and no window ring is sized".
- `docs/design/sycl-mmap-host-tier.md` §4.4: the host MoE expert row (`:1188`) gains "`PINNED`, copied from the
  mapping by the preload, when the plan holds window-ring room and the host plan fits". Correct the sentence at
  `:1195` that says no host weight needs pinned memory: the window ring copies only from pinned memory. Give the
  measured cost, 48.7 GB on the Qwen3.8 exps-IQ4_NL pack. Change nothing else in the section.

**Commit:** `sycl: a mapping load pins its routed host experts when ring room is held and the host plan fits`.

#### Gotchas

- **`weight_host_bytes` is the pinned experts' figure here** because a mapping load's host dense weights are
  `CPU_OWNED` or `FILE_MAPPED`. If the census finds a `PINNED` or `HOST_CPU` dense weight on a mapping load, sum
  the experts in the move helper's style for the log lines and say so.
- **The withdrawn pin leaves the ring's room held.** The room came out of the device pack before the fit was
  known. It is the cost of not planning twice, and the WARN line is the only place it shows.
- **The multi-device builder gets no pin**, because its host charges use other size bases and it sizes no ring.
- **Three test targets compile `unified-cache.cpp` privately.** A new symbol it calls must link in each.
- The spike's hunks (`git show d7d30d127`) show the two places the answer is read. Do not copy its `getenv`
  placement or its variable name.
- **Build** only on the lead's GO: targets `test-sycl-layout-choice test-sycl-host-backing-plan
  test-sycl-host-weight-zone-terms test-moe-stream-plan llama-bench llama-cli`.

---

## End-to-End Validation (on the user's machine): lead only

One run at a time, under `GPU.lock`, selector pinned, `GGML_SYCL_TUNING_CACHE=0`, the VRAM preflight first, `Shmem`
sampled before and 5 s after (kill above 100 GB), the kernel log checked after. Read the driver version first.
Build the task worktree at the reviewed head of A10.3. "Opt-out" means `GGML_SYCL_MOE_PIN_HOST_EXPERTS=0` on the
same binary. The mapped bench arms are reported warm and cold separately, each with its wall time.

| # | Command | Card | Pass when |
|---|---------|------|-----------|
| 1 | `test-sycl-layout-choice`, `test-sycl-host-backing-plan`, `test-moe-stream-plan`, `test-moe-stream-phase` | B50 selector for the first, as registered | rc 0 each |
| 2 | `test-sycl-host-weight-zone-terms` | B70 | rc 0, arms 3a and 3b pass |
| 3 | The source pins A10.1 to A10.3 touched or named, each with `--self-test` where it has one | none | rc 0 each |
| 4 | `llama-cli -m <exps-IQ4_NL>` count prompt, **default flags**, `-v` | B70 | `grep -cx '1, 2, 3, 4, 5'` = 1. `host weights by backing` shows the host experts under `pinned`. No `SYCL_Host model buffer` line. `host_expert_copy_cumulative` equals the WEIGHT term of the plan the preload read, within 1 GiB. Every `[HOST-LEDGER]` plan line of the load is a pinned plan. `[MOE-EXEC] summary` shows MiB streamed above 0 and 0 misses |
| 5 | Run 4 with the opt-out | B70 | the gate passes. The experts are under `mapped`. No `MoE window ring room ... held` line. `[MOE-EXEC] summary` counts the host layers under `cpu`, none under `ring`. `DMA staging pool: 0.0 MB` |
| 6 | `llama-bench -m <exps-IQ4_NL> -p 2048 -n 0 -r 5 -o json`: default, opt-out, opt-out, default | B70 | record all four with wall times and min `MemAvailable`. The first opt-out arm follows a pinned arm and is expected cold. Also record the opt-out arm against today's mapped arm at `20e4361e0` (80 warm, 48 to 50 cold): task A10.2 changed that path |
| 7 | Device experts: the `device-resident routed experts` figure of the ring-room line in runs 4 and 5 | B70 | record both. The default arm pays the ring's room and the 900 MB DMA pool |
| 8 | Run 4 with `-lm mlock` | B70 | the gate passes; the experts stay mapped; host layers under `cpu` |
| 9 | Run 4 with `-lm none` | B70 | unchanged from today's pinned arm: the ring streams, 0 misses |
| 10 | Load time: `/usr/bin/time -v` on run 4 and run 5 | B70 | record both wall times |
| 11 | GPT-OSS 20B chat gate from CLAUDE.md, default flags, `-v`, then with the opt-out | B50 `level_zero:1` | `grep -cx '1, 2, 3, 4, 5'` = 1 in both. Record whether its experts were pinned, and why if not |
| 12 | Mistral completion gate from CLAUDE.md | B50 | output starts `1, 2, 3, 4, 5, 6, 7, 8, 9, 10` |
| 13 | `llama-bench` GPT-OSS 20B `-p 512 -n 128`: default, opt-out, opt-out, default | B50, then B70 | within the spread of `docs/backend/sycl-perf-baselines.md`, or explained by the opt-out pair and by today's tree measured the same way |
| 14 | Kernel log over the whole run | both | no `GT reset`, `guc_id` or `CAT error` line on the card an arm was using, during that arm |

Run 4 is the acceptance: the default load takes the ring, answers correctly, and holds the experts once. Runs 11
and 13 matter because GPT-OSS also has host experts on the B50, so both the pin and A10.2's change can move it.
A regression there against today's tree blocks the merge.

## Follow-ups left out

- **`-lm none` holds each host expert twice** (124 GB for 48.7 GB). Its own ticket.
- **Multi-device plans hold no ring room.** Its own ticket, behind the `GGML_SYCL_MOE_MULTI_GPU` opt-in.
- **A loan on a mapping load** lends device triplets whose host bytes are the mapping. A10.2 leaves lent entries
  as they are read today. Its own ticket.
- **The DMA staging pool** is sized for any plan with pinned host weights (900 MB of device memory here), though
  the window ring does not use it. Its own ticket.
- **A withdrawn pin keeps the ring's room** (119.5 MB of device memory on this pack). Its own ticket if a host
  that small matters.
- **The mapping's pages after the copy.** The copy-pin load's peak RSS was 76.8 GB against 56.5 GB mapped, the
  difference being file pages the kernel may reclaim. An explicit release is left out until it is shown to matter.
- **The mapped path's cold state** (48 t/s against 80 t/s) and **the storage path** for packs that do not fit:
  the owner's second request, next spike.
- **Temp-0 completions on this pack vary run to run** (llama.cpp-fd88). The count gate is unaffected.
