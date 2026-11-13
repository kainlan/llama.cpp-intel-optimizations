# fmcp plan 1, addendum A1: a `MUL_MAT_ID` decode kernel for Q8_0 experts in SOA layout

> **Execution:** `team-driven-development`, as plan 1 (`docs/plans/2026-11-12-fmcp-qwen38-load-fit-attribution.md`).
> This addendum is the task plan 1 lists as A1. It was written after lead run S1 and needs the owner's review
> before anyone is spawned. Ticket: llama.cpp-gjji. Base: `integ/th32` (at `cdeb58b57` when written).

**Goal:** Qwen3.8-Flash-Next UD-Q4_K_XL decodes on the B70 at default flags. Today the first decode stops at
`blk.2.ffn_down_exps.weight`.

**Rule this task follows:** the loaded layout is the answer. The planner materialized these experts in SOA on the
device; decode has no kernel for (Q8_0, SOA). The fix is that kernel. It is not a layout change in the planner, a
conversion at dispatch, or a re-route to another executor.

---

## What S1 measured (2026-11-13, B70, `level_zero:0`, builds `1bf1d9484` and `cdeb58b57`)

Expert census from `[MOE-LAYOUT]`, 48 layers:

| Role | Type | Layers | Device layout | Host layout |
|------|------|-------:|---------------|-------------|
| gate, up | Q4_K | 47 | AOS | AOS |
| gate, up | Q5_K | 1 | AOS | AOS |
| down | Q5_1 | 43 | AOS | AOS |
| down | Q8_0 | 5 (`blk.2`, `4`, `30`, `46`, `47`) | **SOA** | AOS |

The decode failure, identical in both runs:

```
[MOE-DECODE-REFUSAL] tensor=blk.2.ffn_down_exps.weight occurrence=0 expert=322 reason=recipe_missing source_reason=0 recipe_reason=local-layout-unsupported
[SYCL] recoverable runtime fallback failed: MUL_MAT_ID retained decode admission failed
```

`blk.0` and `blk.1` (Q5_1 down, Q4_K gate/up, AOS) decoded before it. Decode aborts at the first refusal, so this
run cannot show whether a later layer also refuses. **A2 stays conditional:** it is written only if the rerun at
the end of this task shows another refusal.

`llama-cli` exited 0 with the decode failed. Every gate in this task scores the answer line or a printed string,
never the exit code alone.

## Why it refuses (verified in the code at `cdeb58b57`)

All paths are under `ggml/src/ggml-sycl/`.

- `ggml_sycl_moe_query_route_capability` (`ggml-sycl.cpp`, near `:37456`) asks
  `moe_mmvq_capability_supports_layout(type, layout)` and falls through to `"local-layout-unsupported"`.
- `moe_mmvq_capability_supports_layout` (`moe-mmvq-tables.hpp:179-207`) answers `layout == GGML_LAYOUT_AOS` for
  one list of types that includes Q8_0. Only MXFP4 has SOA rows.
- `moe_mmvq_batched_dispatch_supports_layout` (`moe-mmvq-tables.hpp:42-84`), the executor table, is AOS-only for
  Q8_0 as well. Capability must stay a subset of the executor tables; `tests/test-sycl-moe-mmvq-tables.cpp` gates
  that.
- The planner picks SOA for Q8_0 experts where XMX is supported (`common.hpp`, near `:1922-1930`).

So one fact (the Q8_0 expert layout) has two sources that disagree: the planner materializes SOA, the decode roster
lists AOS only.

## What already exists to mirror

- `mul_mat_vec_q8_0_q8_1_id_sycl` (`mmvq.cpp:4955`): the Q8_0 AOS launcher. Grid `block_nums(1, total_batches,
  ceil(nrows_per_expert / GGML_SYCL_MOE_MMV_Y))`, `block_dims(1, GGML_SYCL_MOE_MMV_Y, WARP_SIZE)`, profile label
  `mulmat.mmvq.id_aos_q8_0`.
- `mul_mat_vec_mxfp4_q8_1_soa_id_kernel` (`mmvq.cpp:5920`): the one SOA `_id` kernel. It takes the expert base
  from the pointer table (`expert_ptrs[expert_id]`, row relative to the expert) or from the tensor base (absolute
  row `expert_id * nrows_per_expert + row`), and reads the activation in SOA Q8_1 form.
- `reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>` (`vecdotq.hpp:435`): the dense Q8_0 SOA block dot.
- `quants.hpp:164-188`: the Q8_0 SOA offsets. Quant bytes of block `i` sit at `i * 32`; its `ggml_half` scale sits
  at `nblocks * 32 + i * 2`.
- `mmvq_moe_batched_dispatch` (`mmvq.cpp:17046`) already quantizes the activation into SOA Q8_1 when the weight
  layout is SOA (`y_soa`, `:17132-17134`).

---

## Task A1: decode runs Q8_0 SOA experts through a kernel that reads SOA

**Track:** D (new). **Depends on:** nothing in flight; branch from `integ/th32`.
**Model:** Opus implementer at xhigh; Opus spec and quality reviewers.

**File scope:**
- Modify: `ggml/src/ggml-sycl/mmvq.cpp` (new kernel, new launcher, the Q8_0 arm at `:17479`, and every other
  consumer the census below finds)
- Modify: `ggml/src/ggml-sycl/moe-mmvq-tables.hpp` (two rows and the comments that describe them)
- Modify: `ggml/src/ggml-sycl/common.hpp` (only the comment near `:1930` that says the `_id` kernels for Q8_0 are
  AOS-only; it becomes false)
- Create: `ggml/src/ggml-sycl/tests/test-moe-decode-q8-0-soa.cpp` and its registration in
  `ggml/src/ggml-sycl/CMakeLists.txt`
- Modify: `tests/test-sycl-moe-mmvq-tables.cpp` (`decode_soa_gaps` at `:907`, `required[]` at `:196-215`)
- Modify: whichever source pins the census names

**Acceptance criteria:**
- [ ] The new device test fails on the base with the `local-layout-unsupported` refusal for a Q8_0 SOA expert at
      `n_tokens == 1`, and passes after, against the CPU backend, at the tolerances the existing Q8_0 case uses
      (NMSE `5e-4`, max difference `1.0`).
- [ ] A mutant of the kernel that reads the scale of the wrong block fails that test on a line only the mutant
      prints. The mutant registration carries no `SKIP_RETURN_CODE`.
- [ ] `test-sycl-moe-mmvq-tables` passes with Q8_0 removed from `decode_soa_gaps` and a `q8_0/SOA` anchor added to
      `required[]`. The subset invariant still holds.
- [ ] No consumer of the executor table runs the AOS kernel on SOA bytes. The census in step 1 is reported, with
      each consumer's branch.
- [ ] Lead device runs (below) pass.

### Step 1: census, before any code

The executor table is read by more than one consumer. Once it lists (Q8_0, SOA), each of them will accept SOA.
A consumer that still launches the AOS kernel would then compute garbage with no error. List every consumer and
state its branch:

```bash
cd <worktree>
cat ggml/src/ggml-sycl/mmvq.cpp | grep -n 'moe_mmvq_batched_dispatch_supports_layout\|moe_mmvq_any_dispatch_supports_layout\|moe_mmvq_capability_supports_layout\|mul_mat_vec_q8_0_q8_1_id_sycl'
cat ggml/src/ggml-sycl/ggml-sycl.cpp | grep -n 'moe_mmvq_batched_dispatch_supports_layout\|moe_mmvq_any_dispatch_supports_layout\|moe_mmvq_capability_supports_layout\|ggml_sycl_moe_mmvq_batched_supports_layout'
```

Known consumers to confirm: `mmvq_moe_batched_dispatch` (`mmvq.cpp:17089`, the decode path);
`ggml_sycl_mul_mat_id_vec_q` (`mmvq.cpp:21578-21585`, with a second type switch near `:22022`);
`ggml_sycl_select_moe_mmvq_layout` (`ggml-sycl.cpp`, near `:38996` and `:39194`, which downgrades SOA to AOS when
the roster lacks the row and stops doing so once it has it). codescout's index is blind inside `ggml-sycl.cpp`;
use the `cat | grep` form there.

Report the list to the lead before step 3. If a consumer cannot be given an SOA branch inside this file scope,
stop and report.

### Step 2: RED, the device test

Create `ggml/src/ggml-sycl/tests/test-moe-decode-q8-0-soa.cpp`. Model it on
`ggml/src/ggml-sycl/tests/test-moe-pp-batched-dequant.cpp`, which follows the loader's sequence: load begin,
inventory plan, file identity, weight identity, host weight (only while `ggml_backend_sycl_weights_evictable()`),
usage, tensors in `ggml_backend_sycl_host_buffer_type_for_device(dev)`, fill, `model_load_end`, plan activation,
execution context, runtime context. Do not allocate the expert tensor in the device buffer type: that fixture
reorders the whole tensor with no per-expert records and is refused at admission before any kernel runs
(`tests/test-moe-mul-mat-id-q4q8` shows this today and is red for that reason).

Cases, all Q8_0, weights `[in, out, n_experts]`, one `ggml_mul_mat_id` node, compared with the CPU backend:

| Case | in | out | n_experts | n_used | n_tokens |
|------|---:|----:|----------:|-------:|---------:|
| decode, base | 256 | 128 | 8 | 4 | 1 |
| decode, tail rows | 384 | 96 | 8 | 4 | 1 |
| decode, last expert | 256 | 128 | 8 | 1 (id 7) | 1 |

`out = 96` is not a multiple of `GGML_SYCL_MOE_MMV_Y` rounding, so it exercises the `row >= nrows_per_expert`
return. The last-expert case catches an offset that is right only for expert 0.

The test must prove it ran the path it claims. It exits 77 when no device is present or weights are not evictable,
and it fails (not skips) unless the captured log shows, for its tensor, `layout=soa` on a `[MOE-LAYOUT]` line with
a device residency. A pass on an AOS-materialized tensor would be a pass about the wrong thing.

Registration, beside `test-moe-pp-batched-dequant` in `ggml/src/ggml-sycl/CMakeLists.txt`:

```cmake
add_test(NAME test-moe-decode-q8-0-soa COMMAND test-moe-decode-q8-0-soa)
set_tests_properties(test-moe-decode-q8-0-soa PROPERTIES
    LABELS "sycl;moe;device;B50"
    SKIP_RETURN_CODE 77
    RUN_SERIAL TRUE
    ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:1;LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
    TIMEOUT 600
)
```

The implementer does not run it. The lead runs it once on the base build and quotes the refusal line; that quote
is the RED. If the base run shows the fixture did not land the tensor in SOA, the fixture is wrong: fix the fixture
before writing the kernel.

Commit: `test: a device test decodes Q8_0 SOA experts at one token against the CPU backend (llama.cpp-gjji)`,
paths: the new test and `ggml/src/ggml-sycl/CMakeLists.txt`.

### Step 3: GREEN, the kernel

In `mmvq.cpp`, beside the MXFP4 SOA `_id` kernel. The offsets are the whole task, so they are written out:

```cpp
// One row of one expert, Q8_0 weights in SOA: the quant bytes of every block first, the half scales after them.
// `expert_ptrs` set: each entry is one expert's own SOA block, so rows and the scale region are relative to it.
// `expert_ptrs` null: `vx` is one SOA block over every expert, so the row is absolute and the scale region
// starts after every expert's quant bytes.
static void mul_mat_vec_q8_0_q8_1_soa_id_kernel(const uint8_t * __restrict__ vx,
                                                const uint8_t * const * __restrict__ expert_ptrs,
                                                const void * __restrict__ vy,
                                                float * __restrict__ dst,
                                                const int32_t * __restrict__ ids,
                                                const int                ncols,
                                                const int                ncols_y,
                                                const int                nrows_per_expert,
                                                const int                n_tokens,
                                                const int                ne11,
                                                const int64_t            total_qs_size,
                                                const int64_t            total_qs_size_per_expert,
                                                const int64_t            ids_nb0,
                                                const int64_t            ids_nb1,
                                                const int64_t            nb11,
                                                const int64_t            nb12,
                                                const int64_t            nb1,
                                                const int64_t            nb2,
                                                const sycl::nd_item<3> & item_ct1) {
    const int batch_idx = item_ct1.get_group(1);
    const int row       = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);
    if (row >= nrows_per_expert) {
        return;
    }
    const int id   = batch_idx / n_tokens;
    const int iid1 = batch_idx - id * n_tokens;

    int32_t expert_id = -1;
    if (ids) {
        expert_id = *(const int32_t *) ((const char *) ids + iid1 * ids_nb1 + id * ids_nb0);
    }

    const bool      use_expert_ptrs = (expert_ptrs != nullptr);
    const uint8_t * base            = use_expert_ptrs ? (ids ? expert_ptrs[expert_id] : expert_ptrs[batch_idx]) : vx;
    const int64_t   abs_row         = use_expert_ptrs ? row : ((int64_t) expert_id * nrows_per_expert + row);
    const int64_t   d_region        = use_expert_ptrs ? total_qs_size_per_expert : total_qs_size;

    const int       blocks_per_row = ncols / QK8_0;
    const int8_t *  qs_row = (const int8_t *) (base + abs_row * blocks_per_row * QK8_0);
    const ggml_half * d_row = (const ggml_half *) (base + d_region + abs_row * blocks_per_row * sizeof(ggml_half));

    const char *        y_base = (const char *) vy + (int64_t) (id % ne11) * nb11 + (int64_t) iid1 * nb12;
    const int8_t *      y_qs   = (const int8_t *) y_base;
    const sycl::half2 * y_ds   = (const sycl::half2 *) (y_base + ncols_y);

    const int lane_id = item_ct1.get_local_id(2);
    float     acc     = 0.0f;
    for (int b = lane_id; b < blocks_per_row; b += WARP_SIZE) {
        int sumi = 0;
#pragma unroll
        for (int k = 0; k < QK8_0; k += 4) {
            sumi = dpct::dp4a(get_int_from_int8(qs_row + b * QK8_0, k / 4),
                              get_int_from_int8_aligned(y_qs + b * QK8_1, k / 4), sumi);
        }
        // acc += (weight scale d_row[b]) * (activation scale, first half of y_ds[b]) * sumi;
        // take the conversions from the dense block dot named below, do not invent them
    }
    // reduce over the sub-group; lane 0 writes dst_out[row]
}
```

The accumulation is left as a comment on purpose: the exact conversion of the weight scale and of
`y_ds[b]`, and the `get_int_from_int8` helper signatures, must be copied from
`reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>::operator()` (`vecdotq.hpp:435`), which computes
`d * ds_values[0] * sumi` for one block. Reuse that operator if its arguments allow it (it takes the base pointer,
an `(offset, 0)` pair for the quant bytes and one for the scale, the activation quant pointer, the activation
`half2` pointer and `iqs`); calling it keeps one Q8_0 SOA block dot in the tree instead of two. Writing a second
block dot by hand is acceptable only if the operator cannot take a per-expert base, and the reason goes in the
report. End the kernel as the MXFP4 one does: `reduce_over_group` over the sub-group, then lane 0 writes
`dst_out[row]` at `dst + id * nb1 + iid1 * nb2`.

`total_qs_size_per_expert = (int64_t) nrows_per_expert * (ncols / QK8_0) * QK8_0`;
`total_qs_size = total_qs_size_per_expert * n_experts`.

Launcher `mul_mat_vec_q8_0_q8_1_soa_id_sycl`: copy the grid, the `deps` handling and the profile submit of
`mul_mat_vec_q8_0_q8_1_id_sycl` (`mmvq.cpp:4955`). Profile label `mulmat.mmvq.id_soa_q8_0`. It needs its own
kernel name type; follow how the MXFP4 SOA launcher names its kernel.

Dispatch, `mmvq.cpp:17479`:

```cpp
        case GGML_TYPE_Q8_0:
            if (layout == GGML_LAYOUT_SOA) {
                mul_mat_vec_q8_0_q8_1_soa_id_sycl(/* same operands; ncols_y and the two region sizes added */);
            } else {
                mul_mat_vec_q8_0_q8_1_id_sycl(/* unchanged */);
            }
            have_kernel_event = true;
            break;
```

Confirm what `dispatch_ptrs` holds for an SOA lease (per-expert bases) by reading how the MXFP4 SOA arm at
`mmvq.cpp:18088` builds its operands, and pass the same. The activation is already in SOA Q8_1 form for this
layout (`y_soa`).

Tables, `moe-mmvq-tables.hpp`: move `GGML_TYPE_Q8_0` out of the AOS-only case group in both
`moe_mmvq_batched_dispatch_supports_layout` and `moe_mmvq_capability_supports_layout` into its own case returning
`layout == GGML_LAYOUT_AOS || layout == GGML_LAYOUT_SOA`. Update the comments in that header that describe which
types have SOA rows. Do not add a type literal at the three sites
`tests/test-sycl-moe-mmvq-layout-roster-agreement-source.py` guards.

Then every other consumer from the census gets its branch or a stated reason it cannot be reached with SOA.

### Step 4: host gates

```bash
ctest --test-dir build -R '^test-sycl-moe-mmvq-tables$' --output-on-failure
```

Before step 3's table edit this passes; after it, it fails with
`FAIL: ... SOA has a decode executor now; delete it from decode_soa_gaps` until `GGML_TYPE_Q8_0` is removed from
the list at `tests/test-sycl-moe-mmvq-tables.cpp:907`. Quote that failure as the host RED for the table change,
then remove the entry and add `{ GGML_TYPE_Q8_0, GGML_LAYOUT_SOA, "q8_0/SOA" }` to `required[]`.

Source pins: derive the list by searching `tests/*.py` for the symbols you touch
(`moe_mmvq_batched_dispatch_supports_layout`, `moe_mmvq_capability_supports_layout`,
`mul_mat_vec_q8_0_q8_1_id_sycl`, `id_aos_q8_0`, `DEDICATED_LAUNCHER_TYPES`), check each pin's registered form in
`tests/CMakeLists.txt` (pytest or script), run each before and after, and report the counts. Known to name these
tables: `test-sycl-moe-mmvq-layout-roster-agreement-source.py`, `test-sycl-moe-mmvq-consumer-coverage.py`,
`test-sycl-mmid-admission-source.py`, `test-sycl-moe-pp-batched-types-source.py`,
`test-sycl-kernel-profiler-source-mmvq.py`.

Commit: `sycl: decode runs Q8_0 SOA experts through a kernel that reads SOA (llama.cpp-gjji)`.

### Gotchas

- The implementer never runs a device binary. Builds only on the lead's literal GO bound to a sha, detached, with
  a `BUILD_RC=` marker. `mmvq.cpp` is a device TU: expect a long device link.
- `mmvq.cpp` and `ggml-sycl.cpp` carry heavy format drift. Never `clang-format-19 -i`; check changed lines only.
- A zero from `grep` on a refusal string proves nothing unless the same grep fires on the base log. Keep the base
  log as the positive control.
- Task 5 of plan 1 edits `ggml-sycl.cpp`. This task should not need to; if the census says it does, tell the lead
  so the two are sequenced.
- Host-resident Q8_0 experts stay AOS and run on the CPU path. This task does not touch them.

### Lead device runs (one at a time, under `GPU.lock`, selector pinned, `Shmem` sampled before and 5 s after)

1. Base build: `test-moe-decode-q8-0-soa` on the B50. Expect the refusal line. That is the RED.
2. Task build: the same test. Expect pass, with the `layout=soa` line present. Then the mutant. Expect its own
   failure line.
3. Task build: `test-moe-pp-batched-dequant` and `test-moe-mul-mat-id` once each on the B50, as regressions on the
   prompt path and the MXFP4 SOA kernel.
4. Task build, B70: S1's `-c 4096` command on UD-Q4_K_XL with `GGML_SYCL_TUNING_CACHE=0`. Pass is
   `grep -cx '1, 2, 3, 4, 5'` equal to 1 on the output and `MOE-DECODE-REFUSAL` absent, with S1's log as the
   positive control for that string. If another refusal appears, record its tensor, type and layout on
   llama.cpp-gjji: that writes A2.
5. Task build, B70 and B50: the Mistral completion gate and the GPT-OSS chat gate from `CLAUDE.md`, once each.
   Neither model has Q8_0 experts; these show the table change moved nothing else.

No throughput number from these runs is a baseline.
