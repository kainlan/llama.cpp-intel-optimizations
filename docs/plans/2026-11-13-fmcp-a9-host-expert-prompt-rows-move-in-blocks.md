# fmcp addendum A9: host-expert prompt rows cross the device in blocks, not one copy per row

> **Execution:** `team-driven-development`. Addendum to
> `docs/plans/2026-11-12-fmcp-qwen38-load-fit-attribution.md`. Written after lead run S4 (llama.cpp-qp3v).
> Ticket: llama.cpp-1tw2. Base: `integ/th32` (at `cd65d765c` when written).
> **Not spawned until the owner approves this file.**

**Goal:** a prompt micro-batch moves a host-expert layer's activations to the host in a few copies and its results
back in a few copies. Today it issues one copy per row in each direction.

**Architecture:** the change is confined to `dispatch_host_expert_groups`, the lambda that runs host-resident experts
on the CPU during a prompt (the "PP CPU island"). A new SYCL-free helper plans the copies: the unique activation rows
are gathered in runs of adjacent rows, and the result rows are laid out in destination order so adjacent
destinations merge into one copy. No weight moves, no device scratch is added, and the executor does not change:
host experts still run on the CPU.

**Tech stack:** C++ (ggml-sycl backend), a host-only unit test, a Python source pin.

**Test infrastructure:** host tests in `tests/` registered with `llama_build_and_test` and the labels
`sycl;moe;hostonly` (pattern: `tests/test-sycl-moe-host-scatter.cpp`, registered in `tests/CMakeLists.txt`). Source
pins are Python files in `tests/`. Device evidence is lead-only.

---

## Evidence (S4, B70 `level_zero:0`, integ/th32 `ec4735e9e`, exps-IQ4_NL, tuning cache off)

`llama-bench -p 2048 -n 0 -r 1 -v` gave 9.9 to 12.7 t/s across four passes.

| Fact | Value | Source |
|------|-------|--------|
| Copies, device to host | 4,423,038 calls of 7,680 B | memcpy site census, `ggml-sycl.cpp:92372` |
| Copies, host to device | 4,423,038 calls of 5,120 B | memcpy site census, `ggml-sycl.cpp:92440` |
| Share of all copies | 99.9% | same census |
| Host expert rows in the run | 4,423,485 (76.6% of rows) | `[MOE-ROUTE-STATS FINAL]` |
| Kernel profiler, `sycl.memcpy.mem_ops` | n = 8,105,004, 11.46 s of device time | kernel profiler CSV |
| Main thread | 54.5% running, 33.7% in futex wait, 11.6% in page-fault wait | state sampler, pass 3 |
| Main thread CPU | 55.9 s user, 139.1 s system | same |
| Other 27 threads | about 1.2 cores on average | same |

The copy count per direction equals the host row count. So every host row of the run went through the island: 30 rows
per token per layer (10 experts used, 3 matrices), about 15,360 copies per direction per layer and 512-token
micro-batch.

Not measured: kernel stacks for the 139 s of system time, which wait the futex time is, and which mapping the page
faults hit. The gain from this task is therefore not predicted as a number.

## A second finding, which this task does not fix

The exec policy printed `ring` for 252 of 288 layer-ubatches, yet the ring staged almost nothing and the island ran
anyway:

```
[MOE-EXEC] summary: 252 layer-ubatches ring, 36 cpu, 0 forced, 18 MiB streamed, 756 windows, 252 exposed, 34244 misses
[PP-MOE] refused device=0 cause=expert-not-local detail=window-not-staged tensor=blk.11.ffn_gate_exps.weight type=iq4_nl layout=aos staged=1
[PLACEMENT] Totals: ... host weights by backing: pinned/host_cpu=0.0 MB cpu_owned=0.0 MB mapped=101766.9 MB ...
```

Read in the code at `cd65d765c`:

- `stage_window` (`moe-stream-plan.cpp`) skips any lease that is not a host-pinned WEIGHT allocation in one of the
  device's host zones (`moe_stream_owner_streamable`, `moe-stream-plan.hpp:505`).
- The ring's lease hook asks the cache with `allow_mmap_host = true` (`ggml_sycl_moe_stream_host_lease`,
  `ggml-sycl.cpp:34932`), and this pack's 101.8 GB of host weights are all memory-mapped.
- So the ring has nothing it may copy. `resolve_deferred_expert` then declines with `window-not-staged`
  (`ggml-sycl.cpp:93579`) and the op falls to the island.

The stage 0 logs show the same shape with `-lzm off` (0 MiB streamed, 14,626 misses). **The window ring has not run
on this model in any run so far.** The decision says ring and the weights cannot be streamed: one fact, two sources.

This needs an owner decision before a task is written, so it is not in this addendum's task:

1. **Pin host experts at load**, so the ring can stream them. Costs about 100 GB of pinned host memory on this pack.
2. **Let the ring stage from mapped memory.** Copies from pageable memory are slower and can fault mid-prompt.
3. **Leave host experts on the CPU for prompts**, and make the decision stop printing `ring` for a layer whose
   weights cannot be streamed.

Whichever is chosen, the island stays the executor for any layer the policy sends to the CPU, so this task is needed
in all three cases.

**Owner ruling, 2026-11-13:** this addendum is approved as written, and the three options are to be measured against
each other on this pack before one is chosen. Option 3 is what run 4 below measures once this task lands. Options 1
and 2 each need a spike that makes the arm runnable, then their own addendum, brought to the owner before any
implementer is spawned.

---

## Team Topology

**Implementers:** 1. **Reviewers:** spec then quality, fresh each round, Opus.

| Track | Tasks | Description |
|-------|-------|-------------|
| E | A9 | the island's copy plan and its use |

### File Ownership Map

| File | Change |
|------|--------|
| `ggml/src/ggml-sycl/moe-host-island.hpp` | new: the copy plan, SYCL-free |
| `tests/test-sycl-moe-host-island.cpp` | new: host unit test |
| `tests/test-sycl-moe-host-island-source.py` | new: source pin on the lambda |
| `tests/CMakeLists.txt` | two registrations |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | `dispatch_host_expert_groups` only (`:92286-92456` at `cd65d765c`) |

`ggml-sycl.cpp` is also edited by A3 (`ggml_sycl_hold_spill_fit`, near `:19743`). The regions are 70,000 lines
apart; run `git merge-tree` against `integ/th32` each round.

---

### Task A9: the island gathers unique activation rows and returns results in destination order

**Track:** E. **Depends on:** nothing open. **Tracker:** llama.cpp-1tw2.

**Description:** the island copies activation row `(i11, iid1)` of `src1` once for every expert that reads it, and
copies every result row back on its own. A token's activation is the same row for all 10 of its experts, and a
host layer's result rows are adjacent in `dst`. The plan below copies each activation row once, in runs, and writes
results into a host block ordered as `dst` is, so adjacent rows return in one copy.

**Acceptance criteria:**

- [ ] `moe_island_plan_build` passes every case of `tests/test-sycl-moe-host-island.cpp`.
- [ ] The lambda issues its device-to-host copies from `island.gather` and its host-to-device copies from
      `island.scatter`, and contains no per-row copy loop (source pin).
- [ ] Lead run: the two census sites drop from 4,423,038 calls each to at most 20,000 each on the S4 command.
- [ ] Lead run: `llama-cli` on exps-IQ4_NL at default flags still prints the answer line exactly once.
- [ ] Gate registration audit passes plain and with `--self-test`.

#### Step 1: census, before any code

Report these to the lead and wait for the literal word GO-DESIGN:

1. Every caller of `dispatch_host_expert_groups` and `enqueue_host_expert_rows` (expected: one each, at `:95163` and
   `:94622`). Use `cat ggml/src/ggml-sycl/ggml-sycl.cpp | grep -n`; the code index is blind in this file.
2. That `ggml/src/ggml-sycl/moe-decode-hostpath.hpp` (home of `moe_gather_runs_build`) and
   `moe-host-scatter.hpp` open no SYCL header. At `cd65d765c` the first includes `ggml.h` and the standard library,
   the second the standard library only. The lead compiled this file's helper and test against both with plain
   `g++ -std=c++17 -I ggml/src -I ggml/include` and the test printed `OK`. Confirm the test target's include path
   reaches `ggml.h`; another host test that includes `moe-decode-hostpath.hpp` shows how.
3. Whether any reader of `act_pinned` writes through it. `cpu_expert_task::act_host` is `const float *`
   (`:13863` in the header that declares it), so shared rows are expected to be safe. Confirm in
   `ggml_sycl_cpu_expert_mul_mat_batched` and the pool worker.
4. Every source pin under `tests/` that names `dispatch_host_expert_groups`, `scatter_dst_offsets`,
   `moe_pp_cpu_island_act` or `PP CPU island`, and what each asserts.

#### Step 2: RED, the host test

Create `tests/test-sycl-moe-host-island.cpp`:

```cpp
// The copy plan of the prompt's CPU island (ggml-sycl/moe-host-island.hpp). SYCL-free: no device is opened.
#include "ggml-sycl/moe-host-island.hpp"

#include <cstdio>
#include <vector>

using ggml_sycl::moe_island_plan;
using ggml_sycl::moe_island_plan_build;
using ggml_sycl::moe_island_row;

static int g_failed = 0;

#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) {                                                  \
            std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #cond);  \
            ++g_failed;                                                 \
        }                                                               \
    } while (0)

static const size_t K_BYTES = 8 * sizeof(float);  // one activation row
static const size_t N_BYTES = 4 * sizeof(float);  // one result row

// Row of `token` read by the expert in `slot`, for a layer with `n_slots` slots per token: the activation is the
// token's row of src1, the result is row (slot, token) of dst.
static moe_island_row row_of(size_t token, size_t slot, size_t n_slots) {
    moe_island_row r;
    r.src_offset = token * K_BYTES;
    r.dst_offset = (token * n_slots + slot) * N_BYTES;
    return r;
}

int main() {
    // 1. A layer whose every expert is on the host, rows given expert by expert as the island groups them
    //    (slot-major, not destination order): one copy down, one copy back.
    {
        std::vector<moe_island_row> rows;
        for (size_t slot = 0; slot < 3; ++slot) {
            for (size_t token = 0; token < 4; ++token) {
                rows.push_back(row_of(token, slot, 3));
            }
        }
        moe_island_plan plan;
        CHECK(moe_island_plan_build(rows, K_BYTES, N_BYTES, &plan));
        CHECK(plan.n_act_rows == 4);
        CHECK(plan.gather.size() == 1);
        CHECK(plan.gather[0].src_offset == 0 && plan.gather[0].dst_offset == 0 && plan.gather[0].bytes == 4 * K_BYTES);
        CHECK(plan.scatter.size() == 1);
        CHECK(plan.scatter[0].src_offset == 0 && plan.scatter[0].dst_offset == 0 &&
              plan.scatter[0].bytes == 12 * N_BYTES);
        for (size_t i = 0; i < rows.size(); ++i) {
            CHECK(plan.act_index[i] * K_BYTES == rows[i].src_offset);  // the row reads its token's activation
            CHECK(plan.out_index[i] * N_BYTES == rows[i].dst_offset);  // and writes where dst expects it
        }
    }
    // 2. Slot 1 of every token runs on the device: the host rows are slots 0 and 2. Slot 2 of token t and slot 0
    //    of token t + 1 are adjacent in dst, so the result copies are n_tokens + 1, not 2 * n_tokens.
    {
        std::vector<moe_island_row> rows;
        for (size_t slot : { size_t(0), size_t(2) }) {
            for (size_t token = 0; token < 4; ++token) {
                rows.push_back(row_of(token, slot, 3));
            }
        }
        moe_island_plan plan;
        CHECK(moe_island_plan_build(rows, K_BYTES, N_BYTES, &plan));
        CHECK(plan.n_act_rows == 4);
        CHECK(plan.gather.size() == 1);
        CHECK(plan.scatter.size() == 5);
        size_t bytes = 0;
        for (const ggml_sycl::moe_scatter_run & run : plan.scatter) {
            bytes += run.bytes;
        }
        CHECK(bytes == rows.size() * N_BYTES);
        // The host block is dense: result row r sits at r * N_BYTES whatever its destination.
        CHECK(plan.scatter[0].src_offset == 0 && plan.scatter[1].src_offset == 1 * N_BYTES);
        CHECK(plan.scatter[1].dst_offset == 2 * N_BYTES && plan.scatter[1].bytes == 2 * N_BYTES);
    }
    // 3. Only tokens 0, 1 and 5 have a host row: three activation rows in two copies.
    {
        std::vector<moe_island_row> rows = { row_of(5, 0, 1), row_of(0, 0, 1), row_of(1, 0, 1) };
        moe_island_plan             plan;
        CHECK(moe_island_plan_build(rows, K_BYTES, N_BYTES, &plan));
        CHECK(plan.n_act_rows == 3);
        CHECK(plan.gather.size() == 2);
        CHECK(plan.gather[0].bytes == 2 * K_BYTES && plan.gather[1].src_offset == 5 * K_BYTES);
        CHECK(plan.gather[1].dst_offset == 2 * K_BYTES);
        CHECK(plan.act_index[0] == 2 && plan.act_index[1] == 0 && plan.act_index[2] == 1);
    }
    // 4. Refusals: nothing to do, two rows bound for one destination, and activation rows that overlap without
    //    being the same row.
    {
        moe_island_plan plan;
        CHECK(!moe_island_plan_build({}, K_BYTES, N_BYTES, &plan));
        std::vector<moe_island_row> same_dst = { row_of(0, 0, 2), row_of(1, 0, 2) };
        same_dst[1].dst_offset               = same_dst[0].dst_offset;
        CHECK(!moe_island_plan_build(same_dst, K_BYTES, N_BYTES, &plan));
        std::vector<moe_island_row> torn_src = { row_of(0, 0, 2), row_of(1, 0, 2) };
        torn_src[1].src_offset               = K_BYTES / 2;
        CHECK(!moe_island_plan_build(torn_src, K_BYTES, N_BYTES, &plan));
        std::vector<moe_island_row> one = { row_of(0, 0, 1) };
        CHECK(!moe_island_plan_build(one, 0, N_BYTES, &plan));
        CHECK(!moe_island_plan_build(one, K_BYTES, 0, &plan));
    }
    if (g_failed != 0) {
        std::printf("FAILED %d check(s)\n", g_failed);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
```

Register it in `tests/CMakeLists.txt` directly after the `test-sycl-moe-host-scatter` block:

```cmake
# The copy plan of the prompt's CPU island: how host-expert activation and result rows cross the device in
# blocks. ggml-sycl/moe-host-island.hpp includes the standard library, ggml.h and two SYCL-free headers, so this
# needs no device.
if (GGML_SYCL AND NOT GGML_BACKEND_DL)
    llama_build_and_test(test-sycl-moe-host-island.cpp)
    set_tests_properties(test-sycl-moe-host-island PROPERTIES LABELS "sycl;moe;hostonly")
    target_include_directories(test-sycl-moe-host-island PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src)
endif()
```

RED evidence: the syntax-only compile of the test fails with `'ggml-sycl/moe-host-island.hpp' file not found`.
Quote that line on the ticket. No build is needed for it.

**Commit:** `test(sycl): the prompt CPU island's rows cross the device in blocks (RED)`, staging the test and
`tests/CMakeLists.txt`.

#### Step 3: GREEN, the plan

Create `ggml/src/ggml-sycl/moe-host-island.hpp`:

```cpp
#pragma once

// How the prompt's CPU island moves its rows across the device.
//
// A MUL_MAT_ID whose experts live on the host runs them on the CPU: each (token, slot) row reads the token's
// activation row of src1 and writes one result row of dst. A token's activation is the same row for every expert that
// reads it, so the plan copies each distinct activation row once, as one copy per run of adjacent rows (one copy, when
// the micro-batch's tokens are contiguous in src1). The result rows are laid out on the host in destination order,
// so rows adjacent in dst return in one copy: a layer whose every selected expert is on the host returns in one.
//
// SYCL-free on purpose so tests/test-sycl-moe-host-island.cpp can run it without a device.

#include "moe-decode-hostpath.hpp"
#include "moe-host-scatter.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ggml_sycl {

// One island row: its activation at `src_offset` bytes in src1's storage, its result bound for `dst_offset` bytes in
// dst's storage.
struct moe_island_row {
    size_t src_offset = 0;
    size_t dst_offset = 0;
};

struct moe_island_plan {
    std::vector<uint32_t>        act_index;   // per row: the activation block row it reads
    std::vector<uint32_t>        out_index;   // per row: the result block row it writes
    std::vector<moe_gather_run>  gather;      // device -> host: src1 bytes into the activation block
    std::vector<moe_scatter_run> scatter;     // host -> device: result block bytes into dst
    size_t                       n_act_rows = 0;
};

// Builds the plan for `rows`, in the caller's order. False, with nothing to submit, when there are no rows, a row
// length is zero, two rows are bound for overlapping destinations, or two activation rows overlap without being the
// same row: each means the routing that produced the rows is already wrong.
inline bool moe_island_plan_build(const std::vector<moe_island_row> & rows,
                                  size_t                              act_row_bytes,
                                  size_t                              out_row_bytes,
                                  moe_island_plan *                   plan) {
    plan->act_index.assign(rows.size(), 0);
    plan->out_index.assign(rows.size(), 0);
    plan->gather.clear();
    plan->scatter.clear();
    plan->n_act_rows = 0;
    if (rows.empty() || act_row_bytes == 0 || out_row_bytes == 0 || rows.size() > UINT32_MAX) {
        return false;
    }
    std::vector<uint32_t> order(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        order[i] = static_cast<uint32_t>(i);
    }

    // Activations: the distinct source rows, ascending, so adjacent rows of src1 are adjacent in the block.
    std::sort(order.begin(), order.end(),
              [&rows](uint32_t a, uint32_t b) { return rows[a].src_offset < rows[b].src_offset; });
    std::vector<size_t> unique_src;
    unique_src.reserve(rows.size());
    for (uint32_t i : order) {
        const size_t src = rows[i].src_offset;
        if (unique_src.empty() || unique_src.back() != src) {
            if (!unique_src.empty() && src < unique_src.back() + act_row_bytes) {
                return false;
            }
            unique_src.push_back(src);
        }
        plan->act_index[i] = static_cast<uint32_t>(unique_src.size() - 1);
    }
    moe_gather_runs_build(unique_src, 0, act_row_bytes, plan->gather);
    plan->n_act_rows = unique_src.size();

    // Results: block row r is the r-th row in destination order.
    std::sort(order.begin(), order.end(),
              [&rows](uint32_t a, uint32_t b) { return rows[a].dst_offset < rows[b].dst_offset; });
    std::vector<moe_scatter_row> placed;
    placed.reserve(rows.size());
    for (size_t r = 0; r < order.size(); ++r) {
        const uint32_t i = order[r];
        if (r > 0 && rows[i].dst_offset < rows[order[r - 1]].dst_offset + out_row_bytes) {
            plan->gather.clear();
            plan->n_act_rows = 0;
            return false;
        }
        plan->out_index[i] = static_cast<uint32_t>(r);
        moe_scatter_row row;
        row.src_offset = r * out_row_bytes;
        row.dst_offset = rows[i].dst_offset;
        row.bytes      = out_row_bytes;
        placed.push_back(row);
    }
    moe_scatter_runs_build(placed, plan->scatter);
    return true;
}

}  // namespace ggml_sycl
```

**Commit:** `sycl: a copy plan for the prompt CPU island: unique activation rows in runs, results in destination
order`, staging the header.

#### Step 4: RED then GREEN, the lambda

RED first: create `tests/test-sycl-moe-host-island-source.py`. Mirror the structure and registration of
`tests/test-sycl-moe-host-scatter-plan-claim-source.py` (read it and its `tests/CMakeLists.txt` entry; keep the
tree-layout rule its positive-control copy uses). It extracts the body of `dispatch_host_expert_groups` by brace
matching from `auto dispatch_host_expert_groups = [&]` and asserts:

1. the body calls `ggml_sycl::moe_island_plan_build(` exactly once;
2. the body contains `for (const ggml_sycl::moe_gather_run & run : island.gather)` and
   `for (const ggml_sycl::moe_scatter_run & run : island.scatter)`;
3. the body contains exactly two `mem_copy_async(` calls, one inside each of those two loops;
4. the body does not contain `scatter_dst_offsets`;
5. `act_req.size` is computed from `island.n_act_rows`.

Positive control in the same file: a copy of the tree with the per-row host-to-device loop restored fails
assertion 3. It fails on the tree as it stands. Register it; run the gate registration audit plain and with
`--self-test`.

**Commit:** `test(sycl): a source pin holds the prompt CPU island to its copy plan (RED)`.

GREEN: edit `dispatch_host_expert_groups` in `ggml/src/ggml-sycl/ggml-sycl.cpp`. Add
`#include "moe-host-island.hpp"` beside the existing `#include "moe-host-scatter.hpp"`.

(a) After the two `ensure_island_storage(...)` calls and before `act_req` is filled, build the plan:

```cpp
            // One row per (token, slot) the host runs, in the groups' order, which keeps an expert's rows together
            // for the CPU.
            std::vector<ggml_sycl::moe_island_row> island_rows;
            island_rows.reserve(n_rows);
            for (const pp_host_expert_group & group : groups) {
                for (const auto & row : group.rows) {
                    const auto [iid1, id] = row;
                    const int64_t i11     = id % ne11;
                    ggml_sycl::moe_island_row r;
                    r.src_offset =
                        src1_storage.view_offset + static_cast<size_t>(i11) * nb11 + static_cast<size_t>(iid1) * nb12;
                    r.dst_offset =
                        dst_storage.view_offset + static_cast<size_t>(id) * nb1 + static_cast<size_t>(iid1) * nb2;
                    island_rows.push_back(r);
                }
            }
            ggml_sycl::moe_island_plan island;
            if (!ggml_sycl::moe_island_plan_build(island_rows, static_cast<size_t>(K) * sizeof(float),
                                                  static_cast<size_t>(N) * sizeof(float), &island)) {
                GGML_ABORT("[MOE-ROUTE] PP CPU island rows overlap: tensor=%s groups=%zu rows=%zu",
                           src0 && src0->name ? src0->name : "?", groups.size(), n_rows);
            }
```

(b) `act_req.size = island.n_act_rows * static_cast<size_t>(K) * sizeof(float);`. `out_req.size` stays
`n_rows * N * sizeof(float)`.

(c) Replace the per-row device-to-host loop with:

```cpp
            std::vector<sycl::event> d2h_events;
            d2h_events.reserve(island.gather.size());
            for (const ggml_sycl::moe_gather_run & run : island.gather) {
                d2h_events.push_back(ggml_sycl::mem_copy_async(act_handle, run.dst_offset, src1_storage.handle,
                                                               run.src_offset, run.bytes, *stream));
            }
            sycl::event::wait(d2h_events);
```

(d) In the task loop, keep `flat_row` counting rows in the groups' order (the order `island_rows` was built in) and
set:

```cpp
                    task.act_host =
                        act_pinned + static_cast<size_t>(island.act_index[flat_row]) * static_cast<size_t>(K);
                    task.output_host =
                        out_pinned + static_cast<size_t>(island.out_index[flat_row]) * static_cast<size_t>(N);
```

Delete `scatter_dst_offsets` and its `push_back`.

(e) Replace the per-row host-to-device loop with:

```cpp
            std::vector<sycl::event> h2d_events;
            h2d_events.reserve(island.scatter.size());
            for (const ggml_sycl::moe_scatter_run & run : island.scatter) {
                h2d_events.push_back(ggml_sycl::mem_copy_async(dst_storage.handle, run.dst_offset, out_handle,
                                                               run.src_offset, run.bytes, *stream));
            }
            sycl::event::wait(h2d_events);
```

(f) The `[MOE-CPU-ISLAND]` log line gains `act_rows=%zu d2h_copies=%zu h2d_copies=%zu` (values
`island.n_act_rows`, `island.gather.size()`, `island.scatter.size()`), and `moe_host_detail` is given
`island.n_act_rows * K * sizeof(float)` as its activation bytes.

Nothing else in the lambda changes: the allocations, the `mem_fill`, the two host waits and the CPU pool call stay
as they are. They are named under "Not in this task".

**Commit:** `sycl: the prompt CPU island copies each activation row once and returns results in destination
order`, staging `ggml-sycl.cpp` and the pin if it needed a follow.

#### Gotchas

- **Line anchors** are from `cd65d765c`. Verify each before editing.
- **Row order.** `island_rows` and the task loop must walk the groups in the same order, or `act_index` and
  `out_index` address the wrong rows. The task loop's `flat_row` is the only index; do not introduce a second.
- **`mem_fill(out_handle, ...)`** zeroes the result block before the CPU writes it. Leave it: whether every task
  writes all `N` floats is not established here.
- **A gather run can be large** (512 rows of 10,240 B is 5.2 MB). `mem_copy_async` already stages large
  device-to-host copies in chunks (`mem-ops.cpp:1191`); do not chunk in the lambda.
- **No raw pointers kept.** The runs carry offsets; the handles are `act_handle`, `out_handle` and the two storage
  handles the lambda already holds across both waits.
- **Format:** `clang-format-19 --dry-run -Werror` on the new files; in `ggml-sycl.cpp` fix only lines you wrote.
- **Syntax check** `ggml-sycl.cpp` under each distinct compile command for it, and the new test TU with a
  neighbouring host test's command.
- **Build** only on the lead's GO: targets `test-sycl-moe-host-island llama-bench llama-cli`.

#### Not in this task

Each is a candidate the next sampler pass can confirm or drop:

- the two host waits (`sycl::event::wait`) and the blocking `submit_batch(...).get()`, against the no-host-waits
  ruling;
- the activation and result blocks allocated on every dispatch (`unified_allocate`), which is what
  `[SYCL-ZERO-ALLOC-CHECK] ... runtime allocation detected during pp phase` reports;
- a placement kernel for a layer whose host and device slots interleave, where the result copies are about one per
  gap. The census in the lead run below says whether any layer needs it.

---

## End-to-End Validation (on the user's machine): lead only

One run at a time, under `GPU.lock`, selector pinned, `GGML_SYCL_TUNING_CACHE=0`, VRAM preflight first,
`Shmem` sampled before and 5 s after, kernel log checked after. Build `wt-fmcp-a9` at the reviewed head first.

| # | Command | Card | Pass when |
|---|---------|------|-----------|
| 1 | `test-sycl-moe-host-island` (host test, the lead runs it before quality review) | none | prints `OK`, rc 0 |
| 2 | `GGML_SYCL_MEMCPY_SITE_CENSUS=1 llama-bench -m <exps-IQ4_NL> -p 2048 -n 0 -r 1 -v` | B70 `level_zero:0` | both island sites at most 20,000 calls each; `[MOE-ROUTE-STATS FINAL] Host rows` still about 4.42 M (the rows still run, in fewer copies) |
| 3 | `llama-cli -m <exps-IQ4_NL>` with the count prompt, default flags | B70 | `grep -cx '1, 2, 3, 4, 5'` = 1 |
| 4 | The same `llama-bench` without the census, state sampler attached | B70 | no pass bar: record pp2048 t/s and the main thread's running, futex and page-fault shares beside S4's |
| 5 | GPT-OSS 20B chat gate from CLAUDE.md | B50 `level_zero:1` | `grep -cx '1, 2, 3, 4, 5'` = 1 |

Run 3 is the correctness gate: S2's third arm produced the answer line on this pack with host experts on the island,
so a wrong row mapping shows as a wrong answer. Run 4 is the measurement the owner reads; run 2 is the acceptance.
If run 2 shows a site above 20,000 calls, read `[MOE-CPU-ISLAND]` (`GGML_SYCL_MOE_ROUTE_LOG=1`) for the layer whose
`h2d_copies` is large before changing anything.
