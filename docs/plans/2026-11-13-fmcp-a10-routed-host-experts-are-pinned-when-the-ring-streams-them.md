# fmcp addendum A10: routed host experts are pinned when the window ring will stream them

> **Execution:** `team-driven-development` in Claude Code. Addendum to
> `docs/plans/2026-11-13-fmcp-plan-1-qwen38-b70.md`, under epic llama.cpp-fmcp. Written after the ring spikes and
> the lead's four-arm run. **Not approved for spawning until the owner has reviewed it.**

**Goal:** on a normal load (`-lm auto`, which maps the file), the planner pins the host-executed routed experts when
the device has a window ring and the host budget fits them, so prompts take the ring without `-lm none`. When they do
not fit, they stay mapped and the exec policy stops printing `ring` for layers it cannot stream.

**Architecture:** one decision, made once in the plan, after the pack and the ring-room hold: flip every
host-executed routed-expert entry from `FILE_MAPPED` to `PINNED`, all of them or none. The loader learns the answer
per expert tensor through the existing `planned_host_backing` hook, which today answers only for dense names. The
ring's plan index learns the same answer from the entries, so the policy and the stager cannot disagree.

**Tech stack:** C++ (ggml-sycl planner, one loader hook), one host planner test, Python source pins, one design doc.

**Test infrastructure:** `ggml/src/ggml-sycl/tests/test-sycl-host-backing-plan.cpp` (the real planner on the
Qwen3.8 inventory under a mocked device table; gpu-serial by registration, the lead runs it),
`ggml/src/ggml-sycl/tests/test-moe-stream-phase.cpp` and `test-moe-stream-plan.cpp` (host), source pins in `tests/`.

---

## Evidence (lead runs, 2026-11-13, B70 `level_zero:0`, exps-IQ4_NL pack, driver 26.40, tuning cache off)

`llama-bench -p 2048 -n 0 -r 1 -v`, binaries built at `e47ab5a60`, single runs in fixed order, load 6 to 16:

| Arm | Flags | pp2048 t/s | `[MOE-EXEC]` summary |
|-----|-------|-----------:|----------------------|
| mapped-default | none | 11.48 | 288 ring, 0 MiB streamed, 39,112 misses |
| mapped-cpu | `GGML_SYCL_MOE_EXEC=cpu` | 12.56 | 296 cpu forced |
| pinned-cpu | `-lm none`, `GGML_SYCL_MOE_EXEC=cpu` | 16.04 | 296 cpu forced |
| pinned-default | `-lm none` | 200.19 | 296 ring, 114,668 MiB streamed, 0 misses |

Count gate, `llama-cli` default flags, the same pack and card:

| Arm | `grep -cx '1, 2, 3, 4, 5'` | Prompt t/s | Generation t/s |
|-----|---:|---:|---:|
| `-lm none` | 1 | 16.8 | 8.4 |
| default (mapped) | 1 | 1.4 | 1.4 |

Both loads place the same experts: 57,048 host entries (50,139.8 MB) pinned against 56,760 (49,886.7 MB) mapped, of
73,728. The pinned load pre-allocated a 52.7 GB pinned working set. The mapped load pre-allocated 156.3 GB
(`[HOST-LEDGER] ... staging=108820543488 scratch=54905674752`), so on this pack the mapped load holds more pinned
host memory than the pinned one. Why the mapped load's staging term is that large has not been traced (ticket to be
filed by the lead; not in this addendum).

Not measured: the ring from mapped memory (needs code), any figure on the B50, and repeat runs. The 200 t/s figure
is a single run and is not a baseline.

## What the code does today (read at `9f7bd1df9`; verify each anchor before editing)

- `placement_host_backing_when_host` (`unified-cache.cpp:35040`) gives every routed expert `FILE_MAPPED` on a
  mapping load. It is called when the entry is created (`:36582` single device, `:38102` multi device), before the
  pack.
- `hold_moe_stream_ring_room` (`:36346`, called at `:36887`) sizes the ring over the host experts the pack leaves and
  sets `plan.moe_stream_ring_reserve_bytes`. It does not read backing.
- `placement_mark_device_entries_file_mapped` (`:35056`, called at `:37004`) then `populate_host_zone_sizing` and
  `record_host_ledger_plan` (`:37005`, `:37006`) size the host zones from the final backings. The ledger's
  host-budget check is a WARN and keeps the plan (`:34979`).
- `ggml_backend_sycl_planned_host_backing` (`ggml-sycl.cpp:54708`) answers the plan's fact, then a dense entry by
  name, then `FILE_MAPPED`. A routed-expert tensor name has no index, so it always answers `FILE_MAPPED`.
- The loader copies a tensor out of the mapping when its buffer type is not `SYCL_Mapped`
  (`src/llama-model-loader.cpp:2319-2360`); device weights already take that path on a mapping load.
- `moe_stream_owner_streamable` (`moe-stream-plan.hpp:502`) accepts only a `HOST_PINNED` `WEIGHT` lease in a host
  zone of this device. `moe_stream_plan_index_build` (`moe-stream-plan.hpp:234-310`) counts every non-device entry as
  a host expert and reads no backing, so the policy weighs `ring` for experts the stager will skip.
- Design `docs/design/sycl-mmap-host-tier.md` §4.4 (`:1179-1199`) says no host weight needs pinned memory on a
  mapping load because the GPU never reads one. It was written before the window ring existed.

## Decisions this addendum takes (owner to confirm by approving it)

1. **All or none per device plan.** Either every host-executed routed expert of the plan is pinned, or none is. No
   partial pinning by hotness or layer in this task.
2. **The fit rule:** pin when `plan.moe_stream_ring_reserve_bytes > 0` and the plan's host bytes after the flip are
   at most the host budget the load was given. Otherwise leave them mapped. The existing WARN stays a WARN.
3. **Per tensor at the loader.** Backing is per entry in the plan, but the loader allocates per tensor. A routed
   expert tensor answers `PINNED` when any of its entries is a host-executed `PINNED` entry.
4. **`-lm mlock` and `mmap+mlock` keep today's behaviour** (experts stay mapped and locked). The user asked for a
   locked mapping; pinning as well would hold the bytes twice.
5. **No new flag.** `GGML_SYCL_MOE_EXEC=cpu` already forces the CPU; a plan made under it still pins, because the
   variable is read at run time, not at plan time.
6. **The SSD path is separate.** Staging the ring from mapped memory or straight from the NVMe, for packs whose
   experts do not fit in RAM, is the owner's second request and gets its own spike and addendum.

---

## Team Topology

**Implementers:** 1, the two tasks in order. **Reviewers:** spec then quality, fresh each round, Opus.

| Track | Tasks | Description |
|-------|-------|-------------|
| F | A10.1 then A10.2 | the planner's decision and the loader's answer; then the ring index's honesty |

```dot
digraph dependencies { rankdir=LR; "A10.1 pin decision" -> "A10.2 ring index reads backing"; }
```

### File Ownership Map

| File | Task | Change |
|------|------|--------|
| `ggml/src/ggml-sycl/unified-cache.cpp` | A10.1 | new post-pack pass, called in both builders |
| `ggml/src/ggml-sycl/unified-cache.hpp` | A10.1 | a per-tensor expert backing lookup on `placement_plan` |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | A10.1 | `ggml_backend_sycl_planned_host_backing` only |
| `ggml/src/ggml-sycl/tests/test-sycl-host-backing-plan.cpp` | A10.1 | arm 1 follows, two new arms |
| `tests/test-sycl-mapped-loader-source.py` | A10.1 | clause F7 follows |
| `tests/test-sycl-host-backing-source.py` | A10.1 | follows if its pinned text moves |
| `docs/design/sycl-mmap-host-tier.md` | A10.1 | §4.4 table row and the "no host weight needs pinned memory" sentence |
| `ggml/src/ggml-sycl/moe-stream-plan.hpp` | A10.2 | `moe_stream_plan_index_build` reads `host_backing` |
| `ggml/src/ggml-sycl/tests/test-moe-stream-plan.cpp` | A10.2 | one new case |
| `tests/test-sycl-moe-stream-enqueue-source.py` | A10.2 | follows only if its pinned text moves |

A9 edits `ggml-sycl.cpp` near `:92316`; this task edits it near `:54708`. Run `git merge-tree` against `integ/th32`
each round.

---

### Task A10.1: the plan pins host-executed routed experts when the ring will stream them and they fit

**Track:** F. **Depends on:** none (A9 should be merged first so the branch starts from it).

**Acceptance criteria:**

- [ ] On a mapping load with a ring and room, every host-executed routed-expert entry is `PINNED`, device-planned
      entries stay `FILE_MAPPED`, and `weight_host_bytes` equals the host experts' file bytes.
- [ ] On a mapping load whose host budget does not fit them, or with no ring, every figure equals today's arm 1.
- [ ] On a locked mapping load (`load_locks_files`), every figure equals today's arm 3.
- [ ] `ggml_backend_sycl_planned_host_backing("blk.N.ffn_gate_exps.weight")` answers from the plan.
- [ ] `-lm none` plans are unchanged (arm 2 untouched and green).
- [ ] The design doc's §4.4 says what the code now does and why.

#### Step 1: census, before any code. STOP and message the lead if any answer contradicts this addendum.

1. How does a `-lm none` load place a routed expert tensor that mixes device-planned and host-executed experts?
   Which buffer type does `select_weight_buft` (`src/llama-model-loader.cpp:1623`) give it, how many bytes of it
   land in pinned memory, and where do the device-planned experts' bytes go after staging? The lead measured a
   52.7 GB pinned working set for 50.1 GB of host experts, so the whole 64.8 GB of expert tensors is not held. Name
   the mechanism; the mapping load must reuse it, not grow a second one.
2. Does a pinned expert buffer allocated during a mapping load carry owner tier `HOST_PINNED`, role `WEIGHT` and a
   host zone of the device, as `moe_stream_owner_streamable` requires? Read the allocation the loader's host buffer
   type makes and the registration at `unified-cache.cpp:9540-9553`.
3. `llama-model.cpp:3275` calls `llama_mlock` on host buffers when `use_mlock` is set. Confirm decision 4 keeps a
   pinned expert buffer out of that path.
4. The multi-device builder (`unified-cache.cpp:38102` onward): does it hold ring room per device and call
   `placement_mark_device_entries_file_mapped` in the same order as the single-device builder? Give the line of each
   call.
5. Every reader of `placement_plan::has_pinned_backing()` and of `dma_staging_pool_bytes`: what changes for each
   when a mapping load gains pinned entries.
6. Every test or pin that asserts a routed expert is `FILE_MAPPED` on a mapping load. The known ones are in the file
   map; report any other.

#### Step 2: RED, the planner test

In `test-sycl-host-backing-plan.cpp`, `plan_single` takes a budget and a `fact`. Add a host budget and a ring to the
`fact` if they are not already inputs (read `plan_single` and `fact` first; do not add a second planner entry point).
Then add two arms after arm 3 and call them from `main` beside the others:

```cpp
// Arm 4: the fact true, a ring on the device and a host budget that holds the host experts. The host-executed routed
// experts are PINNED so the ring may stream them; device-planned entries keep the mapping as their staging source.
void arm4_mapping_load_ring_fits(const std::vector<placement_tensor_info> & inv, size_t budget) {
    std::printf("(4) load_maps_files true, ring room held, host budget fits: host experts PINNED\n");
    fact f;
    f.maps_files  = true;
    f.layout_cpu  = true;
    f.host_budget = 4 * k_inventory_bytes;  // holds every weight
    captured_log         log;
    const placement_plan plan = plan_single(inv, budget, f, &log);
    const census         c    = take_census(plan);

    check(plan.moe_stream_ring_reserve_bytes > 0, "the plan holds ring room (the arm's precondition)");
    check_eq("host routed-expert groups", c.host_expert_entries / 3, k_host_groups);
    check_eq("host routed experts whose backing is not PINNED", c.host_expert_entries - c.host_pinned_or_host_cpu, 0);
    check_eq("device-planned entries whose backing is not FILE_MAPPED", c.device_not_file_mapped, 0);
    check_eq("weight_host_bytes == the host experts' file bytes", plan.weight_host_bytes, c.host_expert_src);
    check_eq("host_mapped_bytes", plan.host_mapped_bytes, 0);
    check_eq("placed host weight bytes", plan.placed_host_weight_bytes(), c.host_expert_src);
    check(plan.planned_expert_tensor_backing("blk.0.ffn_gate_exps.weight") == placement_host_backing::PINNED ||
              c.host_expert_entries == 0,
          "an expert tensor with a host-executed entry answers PINNED by name");
    check(log.dma_line.find("(no pinned host weights)") == std::string::npos,
          "the DMA pool line does not claim there are no pinned host weights");
}

// Arm 5: the same load with a host budget one byte short of the flipped plan. Nothing is pinned and every figure is
// arm 1's, so a host that cannot hold the experts keeps the mapping.
void arm5_mapping_load_ring_does_not_fit(const std::vector<placement_tensor_info> & inv, size_t budget) {
    std::printf("(5) load_maps_files true, ring room held, host budget too small: host experts stay FILE_MAPPED\n");
    fact f;
    f.maps_files  = true;
    f.layout_cpu  = true;
    f.host_budget = 1 * MiB;
    const placement_plan plan = plan_single(inv, budget, f);
    const census         c    = take_census(plan);

    check_eq("host routed experts whose backing is not FILE_MAPPED", c.host_not_file_mapped, 0);
    check_eq("weight_host_bytes", plan.weight_host_bytes, 0);
    check_eq("host_mapped_bytes == the host experts' file bytes", plan.host_mapped_bytes, c.host_expert_src);
    check_eq("DMA staging pool", plan.dma_staging_pool_bytes, 0);
    check(plan.planned_expert_tensor_backing("blk.0.ffn_gate_exps.weight") == placement_host_backing::FILE_MAPPED,
          "an expert tensor answers FILE_MAPPED by name when nothing was pinned");
}
```

Arm 1 keeps asserting today's figures: give it the too-small host budget or no ring, whichever its `fact` default
already is, and say in its header line which. If arm 1's default `fact` now pins, the header table at the top of the
file gains the row "host routed expert, ring streams it and it fits: PINNED".

RED evidence: a syntax-only compile of the test TU with its own command from the integ tree's
`compile_commands.json` fails on `planned_expert_tensor_backing` and `fact::host_budget`. Report the failing lines.

**Commit:** `test(sycl): a mapping load pins the routed host experts its ring will stream (RED)`.

#### Step 3: GREEN, the plan

In `unified-cache.hpp`, beside `find_dense_entry` (`:1860`), a per-tensor answer for routed experts. It walks the
entries; it is asked once per expert tensor at load, so no index is added:

```cpp
    // A routed-expert tensor's host backing as the loader must allocate it: the loader's buffer is per tensor while
    // the plan's entries are per expert, so one host-executed PINNED entry makes the whole tensor's answer PINNED.
    // FILE_MAPPED when the tensor has no such entry, and when the name is not a routed-expert tensor of this plan.
    placement_host_backing planned_expert_tensor_backing(const std::string & name) const {
        for (const placement_entry & e : entries) {
            if (e.expert_id >= 0 && !e.on_device && e.host_backing == placement_host_backing::PINNED &&
                e.name == name) {
                return placement_host_backing::PINNED;
            }
        }
        return placement_host_backing::FILE_MAPPED;
    }
```

Verify that a routed-expert entry's `name` is the tensor name (read `copy_tensor_info_to_entry`); if it is not, use
the field that is and say so in the report.

In `unified-cache.cpp`, a pass placed immediately before `placement_mark_device_entries_file_mapped` in **both**
builders (`:37004` and the multi-device site the census names):

```cpp
// The window ring streams a host expert only from pinned memory, so a mapping load that holds ring room pins the
// host-executed routed experts when the host budget holds them. All of them or none: a plan that pinned some would
// leave the policy weighing a ring the stager can only half fill. Runs after the pack and the ring-room hold, which
// decide which experts stay on the host and whether a ring exists, and before the host zones are sized, so the zones
// and the DMA pool see the final backings. A locked mapping keeps its experts mapped: the load already holds them.
static void placement_pin_ring_streamed_experts(placement_plan & plan, size_t host_budget) {
    if (!plan.host_backing_maps_files || plan.host_mapped_locked || plan.moe_stream_ring_reserve_bytes == 0) {
        return;
    }
    size_t pin_bytes = 0;
    for (const placement_entry & e : plan.entries) {
        if (e.expert_id >= 0 && !e.on_device && e.host_backing == placement_host_backing::FILE_MAPPED) {
            pin_bytes += e.src_size;
        }
    }
    if (pin_bytes == 0 || plan.host_bytes > host_budget || pin_bytes > host_budget - plan.host_bytes) {
        return;
    }
    for (placement_entry & e : plan.entries) {
        if (e.expert_id >= 0 && !e.on_device && e.host_backing == placement_host_backing::FILE_MAPPED) {
            plan.uncharge_host_weight(placement_host_backing::FILE_MAPPED, e.src_size);
            e.host_backing = placement_host_backing::PINNED;
            plan.charge_host_weight(placement_host_backing::PINNED, e.src_size);
        }
    }
    GGML_LOG_INFO("[PLACEMENT] routed host experts pinned for the window ring: %.1f MB (host budget %.1f MB)\n",
                  pin_bytes / (1024.0 * 1024.0), host_budget / (1024.0 * 1024.0));
}
```

Two things to verify against the code before trusting the sketch, and report:

- The size basis. The pack charges host groups through `charge_host_weight_entries` with a basis and a rounded
  figure (`unified-cache.hpp:1558`). The flip must uncharge and charge on the same basis the pack used for these
  entries, or `host_bytes` drifts. If the basis is `DST`, use `dst_size` in all three places.
- `plan.host_bytes` against the budget is the fit test only if it already holds every host term the ledger counts.
  If the ledger's `host_planned_bytes` is computed later from zone terms, compute the fit from the same terms the
  WARN at `:34979` uses, so the decision and the WARN cannot disagree.

In `ggml-sycl.cpp`, `ggml_backend_sycl_planned_host_backing`: after the dense lookup, before the default,

```cpp
        // A routed-expert tensor has no dense entry; the plan answers for it per tensor.
        return static_cast<ggml_sycl_host_backing>(plan->planned_expert_tensor_backing(tensor_name));
```

and keep the `tensor_name == nullptr` default (`FILE_MAPPED`, the load's fact) exactly as it is:
`sycl_load_maps_files()` (`src/llama-model-loader.cpp:2129`) depends on it.

**Commit:** `sycl: a mapping load pins the routed host experts its window ring will stream, when the host budget
holds them`.

#### Step 4: follows

- `tests/test-sycl-mapped-loader-source.py` clause F7 pins the order "fact, dense entry, default". Extend it to
  "fact, dense entry, expert tensor, default" and add a mutant that drops the expert branch.
- `docs/design/sycl-mmap-host-tier.md` §4.4: the table row for host MoE experts gains the pinned case, and the
  sentence at `:1196-1199` says pinned host backing also exists on a mapping load for routed experts the window ring
  streams, with the reason (the ring copies host experts to device slots during a prompt; a mapped source is refused
  by §5.3). Do not rewrite the rest of the section.

**Commit:** `docs, test: the mapped host tier's design and loader pin name the pinned routed experts`.

#### Gotchas

- **Line anchors** are from `9f7bd1df9`. Verify each.
- **Both builders.** A pass added to one builder only passes the single-device test and breaks `level_zero:0,1`.
- **Do not derive the load's fact from an expert.** `planned_host_backing(nullptr)` stays `FILE_MAPPED` on a
  mapping load.
- **No allocation code changes.** The loader's existing host buffer type allocates through the unified cache. If
  the census finds the mapping load cannot reuse the `-lm none` mechanism without new allocation code, STOP.
- **The test is gpu-serial by registration.** Do not run it; the lead does.
- **Format:** `clang-format-19 --dry-run -Werror` on files you create; in existing files fix only lines you wrote.
- **Syntax check** `unified-cache.cpp` and `ggml-sycl.cpp` under each distinct compile command they have. Three
  test targets compile `unified-cache.cpp` privately; a new symbol it calls must link in each.
- **Build** only on the lead's GO: targets `test-sycl-host-backing-plan llama-bench llama-cli`.

---

### Task A10.2: the ring's plan index does not count a mapped host expert as streamable

**Track:** F. **Depends on:** A10.1.

**Description:** when the experts stay mapped (no room, or a locked mapping), the policy must not print `ring` for
layers the stager will skip. `moe_stream_plan_index_build` already sees each entry's `host_backing`.

**Acceptance criteria:**

- [ ] A layer whose host experts are `FILE_MAPPED` is not claimed by the stream executor, so `begin_layer` takes the
      existing `ring-unavailable` branch and `[MOE-EXEC] summary` counts it under `cpu`.
- [ ] A layer whose host experts are `PINNED` is claimed exactly as today.
- [ ] No new log line, no new knob.

#### Step 1: census

Read `moe_stream_plan_index_build` (`moe-stream-plan.hpp:234-310`), `moe_stream_layer_claimed` (`:387`),
`moe_stream_executor_claims` and the post-checks in `begin_layer` (`moe-stream-plan.cpp:131-136`). Name the one
field of the per-layer index that, when set for a mapped entry, makes the layer unclaimed through the existing
check. The spike suggested `n_untriplet` or a staging of NONE; confirm by reading which of them
`moe_stream_layer_claimed` tests, and that the ring-room sizing (`hold_moe_stream_ring_room`) is not fed by the same
field in a way that would change the room for a pinned plan.

#### Step 2: RED

In `ggml/src/ggml-sycl/tests/test-moe-stream-plan.cpp`, one case built with the file's existing plan helper: a
layer with one host triplet whose three entries are `FILE_MAPPED`, and the same layer with them `PINNED`. Assert
`moe_stream_layer_claimed` is false for the first and true for the second. Write it with the helper and check
macros that file already uses; show the failing assertion line as RED.

**Commit:** `test(sycl): a mapped host expert does not make its layer ring-claimed (RED)`.

#### Step 3: GREEN

In the index build's entry loop, a host entry whose `host_backing` is not `PINNED` sets the field the census named,
with a comment giving the reason: the stager streams only from pinned memory, so a layer with a mapped host expert
cannot be served by the ring.

**Commit:** `sycl: the window ring's plan index does not claim a layer whose host experts are mapped`.

#### Gotchas

- `tests/test-sycl-moe-stream-enqueue-source.py` pins the text of the owner check. This task does not edit that
  check; run the pin and its `--self-test` and report both rc.
- Run every source pin under `tests/` that names `moe_stream_plan_index_build` or `moe_stream_layer_claimed`.
- **Build** only on the lead's GO: targets `test-moe-stream-plan test-moe-stream-phase llama-bench llama-cli`.

---

## End-to-End Validation (on the user's machine): lead only

One run at a time, under `GPU.lock`, selector pinned, `GGML_SYCL_TUNING_CACHE=0`, VRAM preflight first, `Shmem`
sampled before and 5 s after (kill above 100 GB), kernel log checked after. Build the task worktree at the reviewed
head first.

| # | Command | Card | Pass when |
|---|---------|------|-----------|
| 1 | `test-sycl-host-backing-plan` | mocked devices | rc 0, arms 0 to 5 pass |
| 2 | `test-moe-stream-plan`, `test-moe-stream-phase` | none | rc 0 |
| 3 | `llama-cli -m <exps-IQ4_NL>` count prompt, **default flags** | B70 | `grep -cx '1, 2, 3, 4, 5'` = 1; `host weights by backing` shows the host experts under `pinned`; `[MOE-EXEC] summary` shows MiB streamed above 0 and 0 misses |
| 4 | `llama-bench -m <exps-IQ4_NL> -p 2048 -n 0 -r 1 -v`, default flags | B70 | the ring streams (as run 3); record pp2048 beside the 200.19 of the `-lm none` arm; no pass bar on the number |
| 5 | Run 4 with `-lm mlock` | B70 | experts stay mapped; `[MOE-EXEC] summary` counts the host layers under `cpu`, none under `ring` |
| 6 | Run 4 with `-lm none` | B70 | unchanged from today's pinned arm (ring streams) |
| 7 | GPT-OSS 20B chat gate from CLAUDE.md, default flags | B50 `level_zero:1` | `grep -cx '1, 2, 3, 4, 5'` = 1; record whether its experts pinned and its prompt and generation t/s beside the baseline |
| 8 | Mistral completion gate from CLAUDE.md | B50 | output starts `1, 2, 3, 4, 5, 6, 7, 8, 9, 10` |
| 9 | `llama-bench` GPT-OSS 20B `-p 512 -n 128` | B50 and B70 | within the spread of `docs/backend/sycl-perf-baselines.md`, or explained by a paired run against the parent build |

Run 3 is the acceptance: the default load takes the ring and answers correctly. Run 7 matters because GPT-OSS on the
B50 also has host experts, so this change moves it too; a regression there blocks the merge.
