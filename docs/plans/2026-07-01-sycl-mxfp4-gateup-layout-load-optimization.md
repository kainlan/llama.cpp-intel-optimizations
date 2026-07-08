# SYCL MXFP4 Gate/Up Layout-Load Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use team-driven-development to implement this plan with agent teams.

**Goal:** Find a correctness-safe path to improve GPT-OSS MXFP4 MoE TG by attacking resident gate/up weight layout/load efficiency first, then collecting launch and batching evidence if the layout candidate fails.

**Architecture:** The current bottleneck is not host weight streaming and not XMX compute saturation; it is resident VRAM weight-load/layout/transaction efficiency in the packed-Q8 M2 gate/up path. This plan first builds a benchmark-only aligned-payload `XMX_TILED_V2` candidate that preserves the current DPAS math and Q8 packing while changing only the A-weight layout/load helper. Runtime/planner wiring is forbidden until lead-owned synthetic evidence proves at least 20% speedup; launch instrumentation, server-batching histograms, and layout-lifecycle proof are follow-up evidence tracks, not default runtime optimizations.

**Tech Stack:** C++17, SYCL/ESIMD XMX DPAS, llama.cpp SYCL backend, `tools/sycl-kernel-bench`, Python pytest source tests, `scripts/parse-sycl-moe-profile.py`, lead-owned B50 validation.

---

## Team Topology

**Recommended implementers:** 2-3 concurrent source/build workers maximum (execution spawns one ephemeral implementer PER non-lead task or non-lead task portion; same-file tasks remain dependency-serialized)
**Reviewers:** spec + quality, spawned FRESH per review (not a standing pair; see team-driven-development)

### Parallel Tracks

| Track | Tasks | Description |
|-------|-------|-------------|
| A | 1, 2, non-lead docs portion of 12 | Source facts, direction guards, final docs preparation |
| B | 3, 4, 5 | Benchmark-only `XMX_TILED_V2` layout/load candidate |
| C | 7, non-lead source/build portion of 8 | Launch/device-time instrumentation fallback |
| D | 9, non-lead source/build portion of 10 | Batched/server grouped-reuse histogram evidence |
| E | 11 | Planner layout lifecycle proof for non-duplicate canonical layout |
| Lead | 6 plus lead-only evidence portions of 8, 10, and 12 | Lead-owned synthetic/model/evidence decisions and final validation; no worker GPU/model/probe runs |

### Dependency Graph

```dot
digraph dependencies {
    rankdir=LR;
    1 [label="Task 1: source facts guard"];
    2 [label="Task 2: V2 layout contract tests"];
    3 [label="Task 3: V2 bench CLI scaffolding"];
    4 [label="Task 4: V2 reference generator"];
    5 [label="Task 5: V2 bench kernel"];
    6 [label="Task 6: lead V2 synthetic decision"];
    7 [label="Task 7: launch parser/source gates"];
    8 [label="Task 8: launch instrumentation + evidence"];
    9 [label="Task 9: grouped histogram parser/source gates"];
    10 [label="Task 10: grouped histogram instrumentation + evidence"];
    11 [label="Task 11: layout lifecycle proof"];
    12 [label="Task 12: final docs and tracker evidence"];
    1 -> 2;
    2 -> 3;
    3 -> 4;
    4 -> 5;
    5 -> 6;
    6 -> 11 [label="layout-v2-authorized"];
    6 -> 7 [label="layout-v2-rejected"];
    7 -> 8;
    5 -> 9 [label="same source-test file serialized"];
    7 -> 9 [label="same parser/source-test files serialized"];
    6 -> 10 [label="layout-v2-rejected"];
    9 -> 10;
    6 -> 12;
    8 -> 12;
    10 -> 12;
    11 -> 12;
}
```

### File Ownership Map

| File/Directory | Tasks | Conflict Risk |
|----------------|-------|---------------|
| `tests/test-sycl-moe-gateup-optimization-path-source.py` | 1, 2, 3, 4, 5, 7, 9, 11 | Sequential source-contract expansion |
| `ggml/src/ggml-sycl/ggml-sycl-bench.hpp` | 3 | None after Task 3 |
| `tools/sycl-kernel-bench/benchmark_harness.hpp` | 3 | None after Task 3 |
| `tools/sycl-kernel-bench/kernel_registry.hpp` | 3 | None after Task 3 |
| `tools/sycl-kernel-bench/main.cpp` | 3 | None after Task 3 |
| `tools/sycl-kernel-bench/kernels/reference/reference_kernels.hpp` | 3 | None after Task 3 |
| `tools/sycl-kernel-bench/kernels/reference/mxfp4_inline_dot.cpp` | 3, 4 | Sequential, same track |
| `ggml/src/ggml-sycl/mmvq.cpp` | 5, 8, 10 | Sequential if executed in same session; Task 8/10 may be skipped by decisions |
| `scripts/parse-sycl-moe-profile.py` | 7, 9 | Sequential parser feature additions |
| `tests/test-sycl-moe-profile-parser.py` | 7, 9 | Sequential parser tests |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | 11 | None |
| `docs/backend/SYCL.md` | 12 | Final docs only |
| `docs/plans/2026-07-01-sycl-mxfp4-gateup-layout-load-optimization.md` | 6, 8, 10, 11, 12 | Lead-owned evidence updates only |

---

## Grounded Evidence And Constraints

### Current source facts

- Current XMX tiled MXFP4 A-load helper is `mxfp4_xmx_tiled_load_a_vec_from_group()` at `ggml/src/ggml-sycl/mmvq.cpp:7014`. It loads `Repeat` scale bytes from `group + xmx_row_in_group` and `Repeat * 16` packed bytes from `group + tile_n_total + xmx_row_in_group * 16`.
- Current packed-Q8 M2 gate/up baseline kernel is `mxfp4_pair_glu_xmx_tiled_dpas_m2_sycl()` at `ggml/src/ggml-sycl/mmvq.cpp:9035`. It uses `group_bytes = tile_n_total * (1 + k_per / 2)`, i.e. 272 bytes for `tile_n_total=16`, then reads gate/up groups separately.
- `GGML_SYCL_MXFP4_MOE_XMX_N` is `16` at `ggml/src/ggml-sycl/common.hpp:654`; single-stream GPT-OSS TG top-k around 4 cannot naturally fill grouped DPAS N lanes.
- Existing grouped gate/up route requires `total_batches >= exec_n` at `ggml/src/ggml-sycl/mmvq.cpp:16887-16890`, so it is a server/batched-decode path, not a single-token TG fix.
- Existing benchmark baseline name is registered at `tools/sycl-kernel-bench/kernel_registry.hpp:159` and help text at `tools/sycl-kernel-bench/main.cpp:148` as `mxfp4_pair_glu_xmx_tiled_packed_r8_m2_sparse32_bias`.
- Benchmark argument struct is `mxfp4_pair_glu_bench_args` at `ggml/src/ggml-sycl/ggml-sycl-bench.hpp:59`; benchmark launch is declared at line `123`.
- Host/reference benchmark entry is `run_mxfp4_pair_glu()` at `tools/sycl-kernel-bench/kernels/reference/mxfp4_inline_dot.cpp:1042`.
- Recent synthetic evidence rejected multi-RHS: baseline `235.588515 us`, n2 `605.034755 us`, n4 `1323.299220 us`, all exact, in `docs/plans/2026-06-30-sycl-mxfp4-multirhs-gateup-dpas-work-reduction.md:1044-1064`.

### Five Researched Paths Mapped To Tasks

1. **Grouped-route/source-facts feasibility:** Task 1 documents why the existing grouped route does not fix single-stream TG and Task 9/10 collect batched/server evidence if needed.
2. **Benchmark-only `XMX_TILED_V2` A-load layout candidate:** Tasks 2-6 define, scaffold, implement, and lead-validate the aligned-payload layout/load experiment.
3. **Launch/device-time split instrumentation:** Tasks 7-8 add default-off parser and source instrumentation to decide whether launch/graphlet work is worth a follow-up.
4. **Server/continuous-batching grouped-reuse histogram:** Tasks 9-10 measure whether real batches naturally produce same-expert groups large enough to use existing grouped kernels.
5. **Canonical layout lifecycle proof:** Task 11 proves any future winning layout can be made canonical without persistent duplicate gate/up VRAM.

### Non-negotiable constraints

- Workers must not run `/Storage/GenAI/models`, `llama-bench`, `llama-cli`, `sycl-kernel-bench` executable, B50/B580 model gates, `sycl-ls`, `/dev/dri`, DRM fdinfo, `lsof`, direct P2P probes, or real harness execution.
- Lead owns every B50 synthetic/model run and every `sycl-kernel-bench` execution.
- Runtime route remains default-off and unwired until lead Task 6 records `layout-v2-authorized`.
- Do not add a persistent duplicate gate/up VRAM layout. V2 is benchmark-only until planner-owned lifecycle proof passes.
- Preserve unified-cache `mem_handle` ownership. Raw pointers are transient ABI views only.
- Do not revive role-column gate/up fusion or the rejected `multirhs-gateup` runtime path.
- oneAPI setup commands must use `set +u` before sourcing `setvars.sh`.
- Restore `.beads/beads.db` and `.beads/issues.jsonl`, and remove `.codescout` artifacts after tracker operations.

### Acceptance gates

- Benchmark-only V2 synthetic continue gate: exact validation and `mxfp4_pair_glu_xmx_tiled_v2_packed_r8_m2_sparse32_bias <= 188.47 us` on the lead B50 synthetic command, which is at least 20% faster than the current `235.588515 us` baseline.
- If V2 synthetic fails, do not wire runtime. The next executable fallback tasks are launch timing and grouped histogram evidence; both are default-off diagnostics with lead-only model commands.
- Full model promotion is out of scope for this plan unless a later plan wires a runtime route after `layout-v2-authorized` and layout-lifecycle proof.

---

## Task 1: Source Facts Guard For Optimization Direction

**Track:** A
**Depends on:** None
**File scope:**
- Create: `tests/test-sycl-moe-gateup-optimization-path-source.py`

**Description:**
Create a source-only pytest file that locks in the facts driving this plan: current packed-Q8 M2 route, current 272-byte group layout, grouped route `exec_n=16` constraint, and rejected prior DPAS-column candidates. This prevents future implementers from drifting back to role-column or multi-RHS ideas.

**Acceptance Criteria:**

- [ ] Source test passes on current tree.
- [ ] Test proves current XMX tiled group layout uses scale prefix and packed payload after `tile_n_total`.
- [ ] Test proves current packed-Q8 M2 kernel uses `group_bytes = tile_n_total * (1 + k_per / 2)`.
- [ ] Test proves grouped route requires `total_batches >= exec_n` and `GGML_SYCL_MXFP4_MOE_XMX_N = 16`.
- [ ] Test proves recent docs record multi-RHS rejection and no production `GGML_SYCL_MOE_GATEUP_MULTIRHS` route is documented as authorized.

**Implementation Guide:**

1. **RED: prove the test file is absent.**

```bash
cd /Apps/llama.cpp-mxfp4-tg-runtime
python3 -m pytest tests/test-sycl-moe-gateup-optimization-path-source.py -q
```

Expected: FAIL with `file or directory not found`.

2. **GREEN: create the source facts test.**

Create `tests/test-sycl-moe-gateup-optimization-path-source.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
MMVQ = ROOT / "ggml" / "src" / "ggml-sycl" / "mmvq.cpp"
COMMON = ROOT / "ggml" / "src" / "ggml-sycl" / "common.hpp"
REGISTRY = ROOT / "tools" / "sycl-kernel-bench" / "kernel_registry.hpp"
PLAN = ROOT / "docs" / "plans" / "2026-06-30-sycl-mxfp4-multirhs-gateup-dpas-work-reduction.md"
SYCL_DOC = ROOT / "docs" / "backend" / "SYCL.md"


def slice_between(text: str, start_marker: str, end_marker: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[start:end]


def test_current_xmx_tiled_a_load_layout_is_scale_prefix_plus_payload() -> None:
    mmvq = MMVQ.read_text(encoding="utf-8")
    helper = slice_between(mmvq, "mxfp4_xmx_tiled_load_a_vec_from_group", "template <int Repeat>\nSYCL_ESIMD_FUNCTION inline void mxfp4_xmx_tiled_load_a_vec(")
    assert "constexpr int packed_bytes  = k_per / 2" in helper
    assert "const uint8_t * scale_ptr  = group + xmx_row_in_group" in helper
    assert "const uint8_t * packed_ptr = group + tile_n_total + xmx_row_in_group * packed_bytes" in helper
    assert "block_load<uint8_t, Repeat>(scale_ptr)" in helper
    assert "block_load<uint8_t, compact_bytes>(packed_ptr)" in helper


def test_current_packed_m2_route_uses_272_byte_groups_for_tile_n_16() -> None:
    mmvq = MMVQ.read_text(encoding="utf-8")
    body = slice_between(
        mmvq,
        "static sycl::event mxfp4_pair_glu_xmx_tiled_dpas_m2_sycl",
        "template <int Repeat, int GLU_OP, bool Prefetch>",
    )
    assert "const int64_t group_bytes     = tile_n_total * (1 + k_per / 2)" in body
    assert "const int64_t kt_group_stride = n_tile_groups_n * group_bytes" in body
    assert "mxfp4_xmx_tiled_load_a_vec_from_group<Repeat>(gate_group0" in body
    assert "mxfp4_xmx_tiled_load_a_vec_from_group<Repeat>(up_group0" in body
    registry = REGISTRY.read_text(encoding="utf-8")
    assert "mxfp4_pair_glu_xmx_tiled_packed_r8_m2_sparse32_bias" in registry


def test_single_stream_tg_cannot_use_existing_grouped_gateup_without_more_rows() -> None:
    common = COMMON.read_text(encoding="utf-8")
    assert "static constexpr size_t GGML_SYCL_MXFP4_MOE_XMX_N  = 16" in common
    mmvq = MMVQ.read_text(encoding="utf-8")
    dispatch = slice_between(
        mmvq,
        "const bool grouped_decode_shape = xmx_tiled_grouped_eligible",
        "if (device_grouped_shape)",
    )
    assert "total_batches >= exec_n" in dispatch
    assert "ids_host_count == total_batches" in dispatch


def test_rejected_dpas_column_candidates_stay_rejected() -> None:
    plan = PLAN.read_text(encoding="utf-8")
    assert "Decision: `runtime-rejected`" in plan
    assert "235.588515" in plan
    assert "605.034755" in plan
    assert "1323.299220" in plan
    sycl_doc = SYCL_DOC.read_text(encoding="utf-8")
    assert "No production" in sycl_doc
    assert "GGML_SYCL_MOE_GATEUP_MULTIRHS" in sycl_doc
    assert "route is authorized or wired" in sycl_doc
```

Run:

```bash
python3 -m pytest tests/test-sycl-moe-gateup-optimization-path-source.py -q
```

Expected: `4 passed`.

3. **Run diff check.**

```bash
git diff --check
```

Expected: no output.

**Commit:**

```bash
git add tests/test-sycl-moe-gateup-optimization-path-source.py
git commit -m "test(sycl): capture MXFP4 gateup optimization source facts"
```

**Gotchas:**

- Do not edit `mmvq.cpp` in this task.
- The test intentionally uses source strings to guard direction, not runtime behavior.
- If line drift changes nearby code, keep the semantic anchors exact: current helper, packed M2 kernel, grouped route guard, and rejection docs.

---

## Task 2: Benchmark-Only V2 Layout Contract

**Track:** A
**Depends on:** Task 1
**File scope:**
- Modify: `tests/test-sycl-moe-gateup-optimization-path-source.py`

**Description:**
Add a source/pure-Python contract for the proposed benchmark-only `XMX_TILED_V2` layout. The contract defines the only candidate layout this plan will test: payload-first 64-byte-aligned groups with a 64-byte scale slab, `320` bytes per 16-row group.

**Acceptance Criteria:**

- [ ] Test proves V2 group bytes are `320` for `tile_n_total=16` and `k_per=32`.
- [ ] Test proves payload row offsets are `row * 16`, so row `0` and row `8` M2 loads are aligned.
- [ ] Test proves scale offsets are `256 + row` and the scale slab is not in the hot payload prefix.
- [ ] Test proves V2 markers are still absent before Task 3/4 implementation, so later tasks have a meaningful RED target.

**Implementation Guide:**

1. **RED: run the future test before adding it.**

```bash
python3 -m pytest tests/test-sycl-moe-gateup-optimization-path-source.py::test_v2_aligned_payload_layout_contract -q
```

Expected: FAIL with `not found`.

2. **GREEN: append the V2 contract tests.**

Append this code to `tests/test-sycl-moe-gateup-optimization-path-source.py`:

```python

def v2_payload_offset(row: int, packed_bytes: int = 16) -> int:
    return row * packed_bytes


def v2_scale_offset(row: int, tile_n_total: int = 16, packed_bytes: int = 16) -> int:
    return tile_n_total * packed_bytes + row


def v2_group_bytes(tile_n_total: int = 16, packed_bytes: int = 16) -> int:
    payload_bytes = tile_n_total * packed_bytes
    scale_slab_bytes = 64
    return payload_bytes + scale_slab_bytes


def test_v2_aligned_payload_layout_contract() -> None:
    assert v2_group_bytes() == 320
    assert v2_payload_offset(0) == 0
    assert v2_payload_offset(8) == 128
    assert v2_payload_offset(0) % 64 == 0
    assert v2_payload_offset(8) % 64 == 0
    assert v2_scale_offset(0) == 256
    assert v2_scale_offset(15) == 271
    assert v2_scale_offset(0) >= 256


def test_v2_markers_not_implemented_before_scaffolding() -> None:
    joined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [MMVQ, REGISTRY, ROOT / "ggml" / "src" / "ggml-sycl" / "ggml-sycl-bench.hpp"]
    )
    assert "xmx_tiled_v2" not in joined
    assert "mxfp4_pair_glu_xmx_tiled_v2_packed_r8_m2_sparse32_bias" not in joined
```

Run:

```bash
python3 -m pytest tests/test-sycl-moe-gateup-optimization-path-source.py -q
```

Expected: `6 passed`.

**Commit:**

```bash
git add tests/test-sycl-moe-gateup-optimization-path-source.py
git commit -m "test(sycl): define MXFP4 gateup V2 layout contract"
```

**Gotchas:**

- This task intentionally does not add C++ V2 code.
- Keep the V2 candidate payload-first. Do not use `[scale][payload]` row interleaving; that creates 17-byte row stride and is not the candidate under test.
- Later tasks must update or replace `test_v2_markers_not_implemented_before_scaffolding()` with positive assertions.

---

## Task 3: V2 Benchmark CLI Scaffolding

**Track:** B
**Depends on:** Task 2
**File scope:**
- Modify: `tests/test-sycl-moe-gateup-optimization-path-source.py`
- Modify: `ggml/src/ggml-sycl/ggml-sycl-bench.hpp:59-123`
- Modify: `tools/sycl-kernel-bench/benchmark_harness.hpp:113-1234`
- Modify: `tools/sycl-kernel-bench/kernel_registry.hpp:159`
- Modify: `tools/sycl-kernel-bench/main.cpp:148`
- Modify: `tools/sycl-kernel-bench/kernels/reference/reference_kernels.hpp:189-207`
- Modify: `tools/sycl-kernel-bench/kernels/reference/mxfp4_inline_dot.cpp:1042-1512`

**Description:**
Expose a new benchmark-only kernel name and argument plumbing for V2 without changing any device kernel. This makes the later reference generator and `mmvq.cpp` implementation testable through `sycl-kernel-bench` while preserving current behavior for every existing name.

**Acceptance Criteria:**

- [ ] New bench args are default-off: `bool xmx_tiled_v2 = false;` and `int xmx_tiled_v2_group_bytes = 320;`.
- [ ] New registry/help name is `mxfp4_pair_glu_xmx_tiled_v2_packed_r8_m2_sparse32_bias`.
- [ ] Harness parses `_xmx_tiled_v2` from kernel name and passes it through `run_mxfp4_pair_glu()`.
- [ ] Reference API sets `args.xmx_tiled_v2` and `args.xmx_tiled_v2_group_bytes` before launch.
- [ ] Existing benchmark names remain unchanged.

**Implementation Guide:**

1. **RED: update the source test to expect V2 scaffolding.**

Replace `test_v2_markers_not_implemented_before_scaffolding()` in `tests/test-sycl-moe-gateup-optimization-path-source.py` with:

```python

def test_v2_benchmark_cli_scaffolding_exists() -> None:
    bench = (ROOT / "ggml" / "src" / "ggml-sycl" / "ggml-sycl-bench.hpp").read_text(encoding="utf-8")
    harness = (ROOT / "tools" / "sycl-kernel-bench" / "benchmark_harness.hpp").read_text(encoding="utf-8")
    registry = REGISTRY.read_text(encoding="utf-8")
    main = (ROOT / "tools" / "sycl-kernel-bench" / "main.cpp").read_text(encoding="utf-8")
    reference = (ROOT / "tools" / "sycl-kernel-bench" / "kernels" / "reference" / "mxfp4_inline_dot.cpp").read_text(encoding="utf-8")
    assert "bool  xmx_tiled_v2" in bench
    assert "int   xmx_tiled_v2_group_bytes" in bench
    assert "xmx_tiled_v2 = false" in bench
    assert "xmx_tiled_v2_group_bytes = 320" in bench
    assert "parse_moe_xmx_tiled_v2" in harness
    assert "mxfp4_pair_glu_xmx_tiled_v2_packed_r8_m2_sparse32_bias" in registry
    assert "mxfp4_pair_glu_xmx_tiled_v2_packed_r8_m2_sparse32_bias" in main
    assert "args.xmx_tiled_v2" in reference
    assert "args.xmx_tiled_v2_group_bytes" in reference
```

Run:

```bash
python3 -m pytest tests/test-sycl-moe-gateup-optimization-path-source.py::test_v2_benchmark_cli_scaffolding_exists -q
```

Expected: FAIL on missing `xmx_tiled_v2`.

2. **GREEN: add default-off bench args.**

In `ggml/src/ggml-sycl/ggml-sycl-bench.hpp`, inside `struct mxfp4_pair_glu_bench_args` after the current `xmx_tiled_m_tiles` field, add:

```cpp
    bool  xmx_tiled_v2             = false;
    int   xmx_tiled_v2_group_bytes = 320;
```

3. **GREEN: add harness parser.**

In `tools/sycl-kernel-bench/benchmark_harness.hpp`, near `parse_moe_multirhs_cols()` at line `113`, add:

```cpp
static inline bool parse_moe_xmx_tiled_v2(const std::string & kernel_name) {
    return kernel_name.find("_xmx_tiled_v2") != std::string::npos;
}
```

In the MXFP4 pair-GLU case around line `1216`, add:

```cpp
                const bool xmx_tiled_v2       = parse_moe_xmx_tiled_v2(config.kernel_name);
                const int  xmx_tiled_v2_group_bytes = 320;
```

Pass both values to `run_mxfp4_pair_glu()` immediately after `xmx_tiled_m_tiles`.

4. **GREEN: extend reference API.**

In `tools/sycl-kernel-bench/kernels/reference/reference_kernels.hpp`, add two parameters immediately after `int xmx_tiled_m_tiles` in `run_mxfp4_pair_glu()`:

```cpp
                        bool                         xmx_tiled_v2,
                        int                          xmx_tiled_v2_group_bytes,
```

In `tools/sycl-kernel-bench/kernels/reference/mxfp4_inline_dot.cpp`, add the same parameters to the definition at line `1042`, and set args near existing XMX fields:

```cpp
    args.xmx_tiled_v2             = xmx_tiled_v2;
    args.xmx_tiled_v2_group_bytes = xmx_tiled_v2_group_bytes;
```

5. **GREEN: register the benchmark name.**

In `tools/sycl-kernel-bench/kernel_registry.hpp`, add beside the existing packed M2 name:

```cpp
        { "mxfp4_pair_glu_xmx_tiled_v2_packed_r8_m2_sparse32_bias",    GGML_LAYOUT_SOA,       KernelKind::MXFP4_PAIR_GLU          },
```

In `tools/sycl-kernel-bench/main.cpp`, add the same name in the MXFP4 pair-GLU help string beside `mxfp4_pair_glu_xmx_tiled_packed_r8_m2_sparse32_bias`.

6. **Run checks.**

```bash
python3 -m pytest tests/test-sycl-moe-gateup-optimization-path-source.py -q
set +u
source /opt/intel/oneapi/setvars.sh --force >/tmp/v2_scaffold_setvars.log 2>&1
set -u
./scripts/sycl-build.sh sycl-kernel-bench
git diff --check
```

Expected: source tests pass and build succeeds. Do not run `sycl-kernel-bench` executable.

**Commit:**

```bash
git add \
  tests/test-sycl-moe-gateup-optimization-path-source.py \
  ggml/src/ggml-sycl/ggml-sycl-bench.hpp \
  tools/sycl-kernel-bench/benchmark_harness.hpp \
  tools/sycl-kernel-bench/kernel_registry.hpp \
  tools/sycl-kernel-bench/main.cpp \
  tools/sycl-kernel-bench/kernels/reference/reference_kernels.hpp \
  tools/sycl-kernel-bench/kernels/reference/mxfp4_inline_dot.cpp
git commit -m "test(sycl): expose MXFP4 gateup V2 bench route"
```

**Gotchas:**

- Do not add `mmvq.cpp` V2 implementation in this task.
- `xmx_tiled_v2` must be orthogonal to `xmx_tiled`; the V2 name still contains `_xmx_tiled` so existing parser logic should set both.
- Existing calls to `run_mxfp4_pair_glu()` must be updated to compile.

---

## Task 4: V2 Reference Generator Layout Transform

**Track:** B
**Depends on:** Task 3
**File scope:**
- Modify: `tests/test-sycl-moe-gateup-optimization-path-source.py`
- Modify: `tools/sycl-kernel-bench/kernels/reference/mxfp4_inline_dot.cpp:1180-1260`

**Description:**
Convert the benchmark's generated current XMX_TILED expert layout into the V2 payload-first aligned layout when `xmx_tiled_v2=true`. Validation reference weights must remain in the original layout so scalar reference launch stays independent of the candidate.

**Acceptance Criteria:**

- [ ] Source test proves the reference generator has a V2 transform helper.
- [ ] Transform helper maps old group layout `[16 scales][16 rows * 16 payload bytes]` to new group layout `[16 rows * 16 payload bytes][64-byte scale slab]`.
- [ ] V2 group size is exactly `320` bytes for `tile_n_total=16`.
- [ ] `logical_expert_bytes` remains based on original `weights.layout.size()` so benchmark byte accounting remains comparable.
- [ ] Scalar validation reference uses original expert layout, not V2 layout.

**Implementation Guide:**

1. **RED: append the source test.**

Append to `tests/test-sycl-moe-gateup-optimization-path-source.py`:

```python

def test_v2_reference_generator_transforms_current_xmx_layout() -> None:
    reference = (ROOT / "tools" / "sycl-kernel-bench" / "kernels" / "reference" / "mxfp4_inline_dot.cpp").read_text(encoding="utf-8")
    assert "make_xmx_tiled_v2_aligned_payload_layout" in reference
    assert "const size_t old_group_bytes = tile_n_total * (1 + packed_bytes)" in reference
    assert "const size_t new_group_bytes = tile_n_total * packed_bytes + 64" in reference
    assert "new_group + row * packed_bytes" in reference
    assert "new_group + tile_n_total * packed_bytes + row" in reference
    assert "make_xmx_tiled_v2_aligned_payload_layout(launch_layout, m, k)" in reference
    assert "logical_expert_bytes = predecoded_i8 ? launch_layout.size() : weights.layout.size()" in reference
```

Run:

```bash
python3 -m pytest tests/test-sycl-moe-gateup-optimization-path-source.py::test_v2_reference_generator_transforms_current_xmx_layout -q
```

Expected: FAIL on missing helper.

2. **GREEN: add the transform helper in `mxfp4_inline_dot.cpp`.**

Place this helper above `run_mxfp4_pair_glu()`:

```cpp
static std::vector<uint8_t> make_xmx_tiled_v2_aligned_payload_layout(const std::vector<uint8_t> & current,
                                                                     int64_t                      m,
                                                                     int64_t                      k) {
    constexpr size_t tile_n_total = GGML_SYCL_MXFP4_MOE_XMX_N;
    constexpr size_t k_per        = GGML_SYCL_MXFP4_MOE_XMX_K;
    constexpr size_t packed_bytes = k_per / 2;
    const size_t     old_group_bytes = tile_n_total * (1 + packed_bytes);
    const size_t     new_group_bytes = tile_n_total * packed_bytes + 64;
    const size_t     n_groups_n      = static_cast<size_t>((m + static_cast<int64_t>(tile_n_total) - 1) /
                                                           static_cast<int64_t>(tile_n_total));
    const size_t     k_tiles         = static_cast<size_t>(k / static_cast<int64_t>(k_per));
    std::vector<uint8_t> out(k_tiles * n_groups_n * new_group_bytes, 0);
    for (size_t kt = 0; kt < k_tiles; ++kt) {
        for (size_t group_n = 0; group_n < n_groups_n; ++group_n) {
            const uint8_t * old_group = current.data() + (kt * n_groups_n + group_n) * old_group_bytes;
            uint8_t *       new_group = out.data() + (kt * n_groups_n + group_n) * new_group_bytes;
            for (size_t row = 0; row < tile_n_total; ++row) {
                const uint8_t * old_payload = old_group + tile_n_total + row * packed_bytes;
                uint8_t *       new_payload = new_group + row * packed_bytes;
                std::copy(old_payload, old_payload + packed_bytes, new_payload);
                *(new_group + tile_n_total * packed_bytes + row) = *(old_group + row);
            }
        }
    }
    return out;
}
```

3. **GREEN: use the transform only for candidate launch layout.**

In `run_mxfp4_pair_glu()`, after the existing XMX_TILED branch has converted the original SOA `expert_layout` into the current XMX_TILED `launch_layout`, add:

```cpp
    if (xmx_tiled_v2) {
        if (!xmx_tiled || !xmx_tiled_pack_q8 || xmx_tiled_m_tiles != 2 || rows_per_wg != 8 ||
            xmx_tiled_v2_group_bytes != 320) {
            error = "mxfp4_pair_glu XMX_TILED_V2 requires packed XMX_TILED r8 m2 and 320-byte groups.";
            return false;
        }
        launch_layout = make_xmx_tiled_v2_aligned_payload_layout(launch_layout, m, k);
    }
```

Keep the existing line:

```cpp
    const size_t  logical_expert_bytes = predecoded_i8 ? launch_layout.size() : weights.layout.size();
```

4. **Run checks.**

```bash
python3 -m pytest tests/test-sycl-moe-gateup-optimization-path-source.py -q
set +u
source /opt/intel/oneapi/setvars.sh --force >/tmp/v2_reference_setvars.log 2>&1
set -u
./scripts/sycl-build.sh sycl-kernel-bench
git diff --check
```

Expected: tests pass and build succeeds. Do not run `sycl-kernel-bench` executable.

**Commit:**

```bash
git add tests/test-sycl-moe-gateup-optimization-path-source.py tools/sycl-kernel-bench/kernels/reference/mxfp4_inline_dot.cpp
git commit -m "test(sycl): add MXFP4 gateup V2 reference layout"
```

**Gotchas:**

- This task uses `sycl::malloc_device` only in the existing benchmark tool context; do not copy this allocation style into runtime backend code.
- Do not change scalar validation reference layout.
- V2 increases benchmark candidate physical bytes to `320/272 = 1.176x`; success requires latency improvement despite that.

---

## Task 5: Benchmark-Only V2 Packed-M2 Kernel

**Track:** B
**Depends on:** Task 4
**File scope:**
- Modify: `tests/test-sycl-moe-gateup-optimization-path-source.py`
- Modify: `ggml/src/ggml-sycl/mmvq.cpp:7014-7055`
- Modify: `ggml/src/ggml-sycl/mmvq.cpp:9035-9243`
- Modify: `ggml/src/ggml-sycl/mmvq.cpp:20909-21020`

**Description:**
Add a benchmark-only packed-Q8 M2 V2 kernel path. It must preserve the current baseline math, Q8 activation packing, gate/up separation, DPAS template shape, GLU math, and output layout; only A weight group byte stride and A-load helper change.

**Acceptance Criteria:**

- [ ] Source test proves V2 load helper reads payload at `row * packed_bytes` and scale at `tile_n_total * packed_bytes + row`.
- [ ] Source test proves V2 M2 kernel dispatch exists and calls separate gate/up DPAS calls.
- [ ] Benchmark launch rejects `args.xmx_tiled_v2` unless packed XMX_TILED r8 M2 with `xmx_tiled_v2_group_bytes == 320`.
- [ ] No production runtime route or env flag is added.
- [ ] `sycl-kernel-bench` builds.

**Implementation Guide:**

1. **RED: append source tests.**

Append to `tests/test-sycl-moe-gateup-optimization-path-source.py`:

```python

def test_v2_mmvq_load_helper_and_bench_branch_exist() -> None:
    mmvq = MMVQ.read_text(encoding="utf-8")
    assert "mxfp4_xmx_tiled_v2_load_a_vec_from_group" in mmvq
    assert "struct mxfp4_pair_glu_xmx_tiled_v2_dpas_m2_kernel" in mmvq
    assert "mxfp4_pair_glu_xmx_tiled_v2_dpas_m2_sycl" in mmvq
    helper = slice_between(mmvq, "mxfp4_xmx_tiled_v2_load_a_vec_from_group", "template <int Repeat>\nSYCL_ESIMD_FUNCTION inline void mxfp4_xmx_tiled_load_a_vec_from_group")
    assert "const uint8_t * packed_ptr = group + xmx_row_in_group * packed_bytes" in helper
    assert "const uint8_t * scale_ptr  = group + tile_n_total * packed_bytes + xmx_row_in_group" in helper
    body = slice_between(mmvq, "mxfp4_pair_glu_xmx_tiled_v2_dpas_m2_sycl", "static sycl::event mxfp4_pair_glu_xmx_tiled_dpas_m2_sycl")
    assert "const int64_t group_bytes     = 320" in body
    assert "mxfp4_xmx_tiled_v2_load_a_vec_from_group<Repeat>(gate_group0" in body
    assert "mxfp4_xmx_tiled_v2_load_a_vec_from_group<Repeat>(up_group0" in body
    assert "parallel_for<mxfp4_pair_glu_xmx_tiled_v2_dpas_m2_kernel" in body
    assert "gate_part0 = xmx::dpas<8, Repeat, int, int, int8_t, int8_t>" in body
    assert "up_part0   = xmx::dpas<8, Repeat, int, int, int8_t, int8_t>" in body
    launch = slice_between(mmvq, "case 8:\n                if (args.xmx_tiled_grouped)", "if (args.xmx_tiled_m_tiles == 4)")
    assert "mxfp4_dpas_pack_q8_single_col_groups_sycl" in launch
    assert "if (args.xmx_tiled_v2)" in launch
    assert "args.xmx_tiled_v2_group_bytes == 320" in launch
    assert "mxfp4_pair_glu_xmx_tiled_v2_dpas_m2_submit<8" in launch
```

Run:

```bash
python3 -m pytest tests/test-sycl-moe-gateup-optimization-path-source.py::test_v2_mmvq_load_helper_and_bench_branch_exist -q
```

Expected: FAIL on missing V2 helper.

2. **GREEN: add V2 load helper before current helper at `mmvq.cpp:7014`.**

```cpp
template <int Repeat>
SYCL_ESIMD_FUNCTION inline void mxfp4_xmx_tiled_v2_load_a_vec_from_group(
    const uint8_t *                                                             group,
    int64_t                                                                     tile_n_total,
    int64_t                                                                     xmx_row_in_group,
    sycl::ext::intel::esimd::simd<int8_t, Repeat * GGML_SYCL_MXFP4_MOE_XMX_K> & a_vec,
    sycl::ext::intel::esimd::simd<float, Repeat> &                              w_scale_vec) {
    using namespace sycl::ext::intel::esimd;
    constexpr int k_per         = GGML_SYCL_MXFP4_MOE_XMX_K;
    constexpr int packed_bytes  = k_per / 2;
    constexpr int compact_bytes = Repeat * packed_bytes;

    const uint8_t * packed_ptr = group + xmx_row_in_group * packed_bytes;
    const uint8_t * scale_ptr  = group + tile_n_total * packed_bytes + xmx_row_in_group;

    simd<uint8_t, compact_bytes> packed      = block_load<uint8_t, compact_bytes>(packed_ptr);
    simd<uint8_t, Repeat>        scale_bytes = block_load<uint8_t, Repeat>(scale_ptr);
    w_scale_vec                              = mxfp4_e8m0_to_fp32_esimd<Repeat>(scale_bytes);
#pragma unroll
    for (int r = 0; r < Repeat; ++r) {
        simd<uint8_t, packed_bytes> row = packed.template select<packed_bytes, 1>(r * packed_bytes);
        simd<uint8_t, k_per>        codes;
        codes.template select<packed_bytes, 1>(0)            = row & uint8_t{ 0x0f };
        codes.template select<packed_bytes, 1>(packed_bytes) = row >> 4;
        a_vec.template select<k_per, 1>(r * k_per)           = mxfp4_code_values_esimd<k_per>(codes);
    }
}
```

3. **GREEN: clone the packed M2 kernel as a V2 benchmark-only variant.**

Add a distinct named SYCL kernel declaration beside the existing M2 declaration near `ggml/src/ggml-sycl/mmvq.cpp:8884`:

```cpp
template <int Repeat, int GLU_OP, bool Prefetch> struct mxfp4_pair_glu_xmx_tiled_v2_dpas_m2_kernel;
```

Copy `mxfp4_pair_glu_xmx_tiled_dpas_m2_sycl()` from `ggml/src/ggml-sycl/mmvq.cpp:9035-9243` to immediately before the original. Use the same full parameter list as `mxfp4_pair_glu_xmx_tiled_dpas_m2_sycl()` and rename the function identifier to `mxfp4_pair_glu_xmx_tiled_v2_dpas_m2_sycl`.

Inside the clone make exactly these changes:

```cpp
                const int64_t group_bytes     = 320;
                const int64_t kt_group_stride = n_tile_groups_n * group_bytes;
```

Replace every call to `mxfp4_xmx_tiled_load_a_vec_from_group<Repeat>` with `mxfp4_xmx_tiled_v2_load_a_vec_from_group<Repeat>`. Also replace the named `parallel_for` kernel type in the clone with `mxfp4_pair_glu_xmx_tiled_v2_dpas_m2_kernel<Repeat, GLU_OP, Prefetch>` so the V2 kernel does not reuse the original SYCL kernel type.

Keep these lines unchanged in the clone:

```cpp
                    gate_part0 = xmx::dpas<8, Repeat, int, int, int8_t, int8_t>(gate_part0, b_vec, gate_a_vec0);
                    up_part0   = xmx::dpas<8, Repeat, int, int, int8_t, int8_t>(up_part0, b_vec, up_a_vec0);
```

Add a submit wrapper beside the existing M2 submit wrapper:

```cpp
template <int Repeat, bool Prefetch>
static sycl::event mxfp4_pair_glu_xmx_tiled_v2_dpas_m2_submit(sycl::queue &        queue,
                                                              const void * const * gate_ptrs,
                                                              const void * const * up_ptrs,
                                                              const int8_t *       b_packed,
                                                              const float *        y_scales,
                                                              float *              dst_glu,
                                                              const int32_t *      ids,
                                                              const float *        gate_bias,
                                                              const float *        up_bias,
                                                              int                  ncols,
                                                              int                  nrows_per_expert,
                                                              int                  total_batches,
                                                              int                  n_tokens,
                                                              int64_t              ids_nb0,
                                                              int64_t              ids_nb1,
                                                              int64_t              dst_nb1,
                                                              int64_t              dst_nb2,
                                                              int64_t              gate_bias_nb1,
                                                              int64_t              up_bias_nb1,
                                                              int                  glu_op,
                                                              float                alpha,
                                                              float                limit,
                                                              int                  tile_n_total,
                                                              const sycl::event &  pack_event) {
    if (glu_op == GGML_GLU_OP_SWIGLU_OAI) {
        return mxfp4_pair_glu_xmx_tiled_v2_dpas_m2_sycl<Repeat, GGML_GLU_OP_SWIGLU_OAI, Prefetch>(
            queue, gate_ptrs, up_ptrs, b_packed, y_scales, dst_glu, ids, gate_bias, up_bias, ncols,
            nrows_per_expert, total_batches, n_tokens, ids_nb0, ids_nb1, dst_nb1, dst_nb2, gate_bias_nb1,
            up_bias_nb1, alpha, limit, tile_n_total, pack_event);
    }
    return mxfp4_pair_glu_xmx_tiled_v2_dpas_m2_sycl<Repeat, GGML_GLU_OP_SWIGLU, Prefetch>(
        queue, gate_ptrs, up_ptrs, b_packed, y_scales, dst_glu, ids, gate_bias, up_bias, ncols,
        nrows_per_expert, total_batches, n_tokens, ids_nb0, ids_nb1, dst_nb1, dst_nb2, gate_bias_nb1,
        up_bias_nb1, alpha, limit, tile_n_total, pack_event);
}
```

4. **GREEN: add benchmark launch branch in the existing packed-Q8 XMX tiled switch.**

In `ggml_sycl_mxfp4_pair_glu_bench_launch()`, locate the existing `if (args.xmx_tiled_pack_q8)` block in the XMX tiled case. The current code creates:

```cpp
                    sycl::event pack_event = mxfp4_dpas_pack_q8_single_col_groups_sycl(
                        *args.stream, args.activations_q8_soa, args.dpas_b_packed, args.dpas_y_scales, args.ncols,
                        args.ncols_y, total_batches, args.n_tokens, args.ne11, args.nb11, args.nb12);
```

Immediately after that existing `pack_event` line and before the existing `if (args.xmx_tiled_m_tiles == 4)` branch, add:

```cpp
                    if (args.xmx_tiled_v2) {
#if GGML_SYCL_XMX_JOINT_MATRIX_AVAILABLE
                        const bool valid_v2 = args.xmx_tiled && args.xmx_tiled_pack_q8 && !args.xmx_tiled_grouped &&
                                              !args.xmx_tiled_prefetch && args.xmx_tiled_m_tiles == 2 &&
                                              args.rows_per_wg == 8 && args.xmx_tiled_v2_group_bytes == 320 &&
                                              args.dpas_b_packed && args.dpas_y_scales && !args.direct_xmx &&
                                              !args.split_gate_up && !args.predecoded_i8 && !args.vector_qs_load &&
                                              args.scale_stride_blocks == 0;
                        if (!valid_v2) {
                            return false;
                        }
                        mxfp4_pair_glu_xmx_tiled_v2_dpas_m2_submit<8, false>(
                            *args.stream, args.gate_ptrs, args.up_ptrs, args.dpas_b_packed, args.dpas_y_scales,
                            args.output, args.ids, args.gate_bias, args.up_bias, args.ncols, args.nrows_per_expert,
                            total_batches, args.n_tokens, args.ids_nb0, args.ids_nb1, args.dst_nb1, args.dst_nb2,
                            args.gate_bias_nb1, args.up_bias_nb1, args.glu_op, args.alpha, args.limit, tile_n_total,
                            pack_event);
                        return true;
#else
                        return false;
#endif
                    }
```

This placement deliberately reuses the existing `mxfp4_dpas_pack_q8_single_col_groups_sycl()` activation-pack event and does not invent a new pack helper or dependency vector.

5. **Run checks.**

```bash
python3 -m pytest tests/test-sycl-moe-gateup-optimization-path-source.py -q
set +u
source /opt/intel/oneapi/setvars.sh --force >/tmp/v2_kernel_setvars.log 2>&1
set -u
./scripts/sycl-build.sh sycl-kernel-bench
git diff --check
```

Expected: tests pass and build succeeds. Do not run `sycl-kernel-bench` executable.

**Commit:**

```bash
git add tests/test-sycl-moe-gateup-optimization-path-source.py ggml/src/ggml-sycl/mmvq.cpp
git commit -m "feat(sycl): add benchmark-only MXFP4 gateup V2 layout candidate"
```

**Gotchas:**

- This task must not add `GGML_SYCL_MOE_GATEUP_LAYOUT_V2` or any production dispatch.
- Keep V2 tied to packed-Q8 M2 only. Do not implement direct-Q8, grouped, prefetch, or runtime variants.
- The new branch must use existing `mem_handle` ownership indirectly through benchmark arguments only; do not store raw pointers beyond the launch ABI.

---

## Task 6: Lead-Only V2 Synthetic Decision

**Track:** Lead
**Depends on:** Task 5
**File scope:**
- Modify: `docs/plans/2026-07-01-sycl-mxfp4-gateup-layout-load-optimization.md`
- Update tracker tasks created for this plan.

**Description:**
Run only the lead-owned B50 synthetic proof for the benchmark-only V2 candidate. This task decides whether planner/runtime layout work is allowed.

**Acceptance Criteria:**

- [ ] Safe non-GPU gates pass before synthetic proof.
- [ ] Lead-owned synthetic log path is recorded.
- [ ] Decision explicitly says `layout-v2-authorized` or `layout-v2-rejected`.
- [ ] Runtime dispatch remains unmodified in this task.

**Implementation Guide:**

1. **RED: prove the synthetic evidence section is not already present.**

```bash
cd /Apps/llama.cpp-mxfp4-tg-runtime
! rg -n "^## Lead V2 Synthetic Evidence$" docs/plans/2026-07-01-sycl-mxfp4-gateup-layout-load-optimization.md
```

Expected: command succeeds because no completed V2 decision has been recorded yet.

2. **GREEN: run safe gates before synthetic proof.**

```bash
cd /Apps/llama.cpp-mxfp4-tg-runtime
python3 -m pytest tests/test-sycl-moe-gateup-optimization-path-source.py -q
set +u
source /opt/intel/oneapi/setvars.sh --force >/tmp/v2_synth_setvars.log 2>&1
set -u
./scripts/sycl-build.sh sycl-kernel-bench
git diff --check
```

Expected: all pass.

3. **GREEN: run the lead-only synthetic proof.**

```bash
cd /Apps/llama.cpp-mxfp4-tg-runtime
set +u
source /opt/intel/oneapi/setvars.sh --force >/tmp/v2_synth_setvars.log 2>&1
set -u
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/sycl-kernel-bench \
  --kernel=mxfp4_pair_glu_xmx_tiled_packed_r8_m2_sparse32_bias,mxfp4_pair_glu_xmx_tiled_v2_packed_r8_m2_sparse32_bias \
  --quant=MXFP4 --dim_m=2880 --dim_n=4 --dim_k=2880 \
  --iterations=200 --warmup=20 --validate --output=jsonl \
  > /tmp/v2_gateup_synth.jsonl
```

Decision rule:

- If V2 validates with `max_abs_error=0.000000` and latency `<= 188.47 us`, record `layout-v2-authorized` and continue to Task 11.
- Otherwise record `layout-v2-rejected` with baseline latency, V2 latency, max error, and continue to Tasks 7-10 for evidence only.

4. **GREEN: record the decision.**

Add a `## Lead V2 Synthetic Evidence` section to the plan. It must include these populated fields from `/tmp/v2_gateup_synth.jsonl`: decision, synthetic log path, baseline kernel name, baseline latency in microseconds, baseline max absolute error, V2 kernel name, V2 latency in microseconds, V2 max absolute error, and the pass/fail reason. Use the literal decision string `layout-v2-authorized` only when the V2 row validates exactly and is at least 20% faster than baseline; otherwise use `layout-v2-rejected`.

**Commit:**

```bash
git add docs/plans/2026-07-01-sycl-mxfp4-gateup-layout-load-optimization.md
git commit -m "docs(sycl): record MXFP4 gateup V2 synthetic decision"
```

**Gotchas:**

- This is the first task allowed to run `sycl-kernel-bench`; it is lead-owned only.
- Do not run full-model commands here.
- Do not wire production dispatch in this task.
- Do not commit synthetic evidence with fabricated numbers, blank fields, placeholder tokens, or example values.

---

## Lead V2 Synthetic Evidence

| Field | Value |
| --- | --- |
| Decision | `layout-v2-rejected` |
| Synthetic log path | `/tmp/v2_gateup_synth.jsonl` |
| Baseline kernel name | `mxfp4_pair_glu_xmx_tiled_packed_r8_m2_sparse32_bias` |
| Baseline latency (us) | `237.084865` |
| Baseline max absolute error | `0.000000` |
| V2 kernel name | `mxfp4_pair_glu_xmx_tiled_v2_packed_r8_m2_sparse32_bias` |
| V2 latency (us) | `251.179255` |
| V2 max absolute error | `0.000000` |
| Pass/fail reason | V2 validated exactly, but it was slower than baseline and missed the authorization threshold `<= 188.47 us`; do not wire runtime dispatch. |

Safe gates before the synthetic proof passed on 2026-07-01: `python3 -m pytest tests/test-sycl-moe-gateup-optimization-path-source.py -q` (`8 passed`), `./scripts/sycl-build.sh sycl-kernel-bench`, and `git diff --check`. The synthetic command used `ONEAPI_DEVICE_SELECTOR=level_zero:1` and did not run full-model commands.

---

## Task 7: Launch Timing Parser And Source Gates

**Track:** C
**Depends on:** Task 6 with `layout-v2-rejected`
**File scope:**
- Modify: `scripts/parse-sycl-moe-profile.py:294-324`
- Modify: `tests/test-sycl-moe-profile-parser.py`
- Modify: `tests/test-sycl-moe-gateup-optimization-path-source.py`

**Description:**
Add parser and source guards for launch/device timing fields so launch-overhead evidence can be collected without changing route behavior. This task is evidence-only and default-off.

**Acceptance Criteria:**

- [ ] Parser accepts `[MXFP4-MOE-TG-PROFILE]` lines with `launch_submit_us=`, `launch_device_us=`, and `launch_wait_us=`.
- [ ] Parser prints `profile.mxfp4_tg.launch.submit_us`, `device_us`, and `wait_us` sums.
- [ ] Source test proves instrumentation is gated behind `GGML_SYCL_MOE_TG_PROFILE_LAUNCH`.

**Implementation Guide:**

1. **RED: add parser test.**

Append to `tests/test-sycl-moe-profile-parser.py`:

```python

def test_parser_sums_mxfp4_tg_launch_timing_fields(tmp_path: pathlib.Path) -> None:
    log = tmp_path / "profile.log"
    log.write_text(
        "[MXFP4-MOE-TG-PROFILE] calls=1 total=6.0 ms pack=0.05 ms kernel=5.5 ms "
        "last_path=packed-q8-m2 launch_submit_us=120.5 launch_device_us=5100.25 launch_wait_us=80.0\n"
        "[MXFP4-MOE-TG-PROFILE] calls=1 total=6.2 ms pack=0.06 ms kernel=5.7 ms "
        "last_path=packed-q8-m2 launch_submit_us=100.0 launch_device_us=5200.0 launch_wait_us=70.0\n",
        encoding="utf-8",
    )
    out = run_parser(log)
    assert "profile.mxfp4_tg.launch.submit_us 220.500" in out
    assert "profile.mxfp4_tg.launch.device_us 10300.250" in out
    assert "profile.mxfp4_tg.launch.wait_us 150.000" in out
```

Run:

```bash
python3 -m pytest tests/test-sycl-moe-profile-parser.py::test_parser_sums_mxfp4_tg_launch_timing_fields -q
```

Expected: FAIL because fields are not printed.

2. **GREEN: update parser key handling.**

In `scripts/parse-sycl-moe-profile.py`, extend the MXFP4 TG accumulator dict near the current MXFP4 profile handling with keys:

```python
"launch_submit_us": 0.0,
"launch_device_us": 0.0,
"launch_wait_us": 0.0,
```

When parsing key/value fields from an `[MXFP4-MOE-TG-PROFILE]` line, if a key matches one of those names, add its float value. When printing the MXFP4 TG summary, emit:

```python
print(f"profile.mxfp4_tg.launch.submit_us {mxfp4_tg['launch_submit_us']:.3f}")
print(f"profile.mxfp4_tg.launch.device_us {mxfp4_tg['launch_device_us']:.3f}")
print(f"profile.mxfp4_tg.launch.wait_us {mxfp4_tg['launch_wait_us']:.3f}")
```

3. **GREEN: add source guard for future instrumentation env.**

Append to `tests/test-sycl-moe-gateup-optimization-path-source.py`:

```python

def test_launch_timing_instrumentation_is_default_off_when_implemented() -> None:
    mmvq = MMVQ.read_text(encoding="utf-8")
    if "GGML_SYCL_MOE_TG_PROFILE_LAUNCH" not in mmvq:
        return
    assert "mxfp4_moe_tg_profile_launch_enabled" in mmvq
    assert "std::getenv(\"GGML_SYCL_MOE_TG_PROFILE_LAUNCH\")" in mmvq
    assert "launch_submit_us" in mmvq
    assert "launch_device_us" in mmvq
    assert "launch_wait_us" in mmvq
```

4. **Run checks.**

```bash
python3 -m pytest tests/test-sycl-moe-profile-parser.py::test_parser_sums_mxfp4_tg_launch_timing_fields tests/test-sycl-moe-gateup-optimization-path-source.py -q
git diff --check
```

Expected: tests pass.

**Commit:**

```bash
git add scripts/parse-sycl-moe-profile.py tests/test-sycl-moe-profile-parser.py tests/test-sycl-moe-gateup-optimization-path-source.py
git commit -m "test(sycl): add MXFP4 gateup launch timing parser gates"
```

**Gotchas:**

- Do not add `mmvq.cpp` instrumentation in this task; Task 8 owns it.
- Parser must remain line-oriented and must not reintroduce whole-file mega-regex parsing.

---

## Task 8: Launch Timing Instrumentation And Lead Evidence

**Track:** C + Lead evidence
**Depends on:** Task 7
**File scope:**
- Modify: `ggml/src/ggml-sycl/mmvq.cpp:16380-17573`
- Modify: `docs/plans/2026-07-01-sycl-mxfp4-gateup-layout-load-optimization.md`

**Description:**
Add default-off launch timing instrumentation to split host submit, device event duration, and wait/drain time for the current packed-Q8 M2 path. Lead then runs a profile command only if needed after V2 rejection.

**Acceptance Criteria:**

- [ ] Instrumentation is gated by `GGML_SYCL_MOE_TG_PROFILE_LAUNCH=1`.
- [ ] Normal profile output is unchanged when env is unset.
- [ ] When env is set, `[MXFP4-MOE-TG-PROFILE]` includes `launch_submit_us`, `launch_device_us`, and `launch_wait_us`.
- [ ] Lead evidence records whether launch overhead is worth graphlet/persistent work.

**Implementation Guide:**

1. **RED: run source guard before implementation.**

```bash
python3 -m pytest tests/test-sycl-moe-gateup-optimization-path-source.py::test_launch_timing_instrumentation_is_default_off_when_implemented -q
```

Expected: PASS by early return before implementation. After implementation it must pass by positive assertions.

2. **GREEN: add env helper in `mmvq.cpp`.**

Near other `mxfp4_moe_*_requested()` helpers, add:

```cpp
static bool mxfp4_moe_tg_profile_launch_enabled() {
    const char * env = std::getenv("GGML_SYCL_MOE_TG_PROFILE_LAUNCH");
    return env && std::atoi(env) != 0;
}
```

3. **GREEN: add timing fields in `mmvq_moe_batched_dispatch_pair_glu_mxfp4_soa()`.**

Inside the function around `ggml/src/ggml-sycl/mmvq.cpp:16380`, initialize:

```cpp
    double profile_launch_submit_us = 0.0;
    double profile_launch_device_us = 0.0;
    double profile_launch_wait_us   = 0.0;
```

Around the non-aggressive packed-Q8 M2 launch branch, replace the existing direct assignment with this concrete wrapped assignment:

```cpp
                            const bool profile_launch = tg_profile && mxfp4_moe_tg_profile_launch_enabled();
                            const auto launch_submit_begin = std::chrono::high_resolution_clock::now();
                            kernel_event = mxfp4_pair_glu_xmx_tiled_dpas_m2_submit<repeat>(
                                *stream, gate_ptrs_device, up_ptrs_device, b_packed, y_scales, glu_d, ids_device,
                                gate_bias_device, up_bias_device, ne00, ne01, static_cast<int>(total_batches),
                                static_cast<int>(num_tokens), ids_nb0, ids_nb1, glu_dst->nb[1], glu_dst->nb[2],
                                gate_bias_nb1, up_bias_nb1, glu_op, alpha, limit, tile_n_total, pack_event);
                            const auto launch_submit_end = std::chrono::high_resolution_clock::now();
                            if (profile_launch) {
                                profile_launch_submit_us += std::chrono::duration<double, std::micro>(
                                                                launch_submit_end - launch_submit_begin)
                                                                .count();
                                const auto wait_begin = std::chrono::high_resolution_clock::now();
                                kernel_event.wait_and_throw();
                                const auto wait_end = std::chrono::high_resolution_clock::now();
                                profile_launch_wait_us +=
                                    std::chrono::duration<double, std::micro>(wait_end - wait_begin).count();
                                const double device_us = mmvq_sycl_event_duration_us(kernel_event);
                                if (device_us >= 0.0) {
                                    profile_launch_device_us += device_us;
                                }
                            }
```

Do not wrap the aggressive partial artifact branch in this task; this evidence track is specifically for the current `packed-q8-m2` gate/up route.

4. **GREEN: route launch fields through the existing TG profile accumulator.**

Extend `mmvq_moe_tg_profile_accum` near the top of `mmvq.cpp` with three doubles named `launch_submit_us`, `launch_device_us`, and `launch_wait_us`. Extend `mmvq_moe_tg_profile_record()` with optional trailing arguments for those three values, defaulted to `0.0`, and accumulate them into the struct.

In the `mmvq_moe_tg_profile_record()` call for gate/up/GLU near the end of `mmvq_moe_batched_dispatch_pair_glu_mxfp4_soa()`, pass `profile_launch_submit_us`, `profile_launch_device_us`, and `profile_launch_wait_us` as the new trailing arguments.

In the `[MXFP4-MOE-TG-PROFILE]` dump inside `mmvq_moe_tg_profile_record()`, keep the existing `fprintf` path byte-for-byte unchanged when `mxfp4_moe_tg_profile_launch_enabled()` is false. Add a second `fprintf` branch used only when launch profiling is enabled; that branch appends these numeric fields:

```cpp
" launch_submit_us=%.3f launch_device_us=%.3f launch_wait_us=%.3f"
```

This preserves normal profile output while allowing the parser from Task 7 to consume launch timing lines from diagnostic runs.

5. **Run safe checks.**

```bash
python3 -m pytest tests/test-sycl-moe-gateup-optimization-path-source.py tests/test-sycl-moe-profile-parser.py -q
set +u
source /opt/intel/oneapi/setvars.sh --force >/tmp/launch_timing_setvars.log 2>&1
set -u
./scripts/sycl-build.sh llama-bench
git diff --check
```

Expected: tests pass and `llama-bench` builds. Do not run model commands.

6. **GREEN: lead-only launch evidence command.**

Only the lead may run this B50/model profile; workers must not run it. Use the existing acknowledged harness so the command is reproducible and side-effect directories are explicit:

```bash
cd /Apps/llama.cpp-mxfp4-tg-runtime
OUT_DIR=/tmp/sycl_gptoss_launch_timing_$(date +%Y%m%d_%H%M%S)
GGML_SYCL_MOE_TG_PROFILE_LAUNCH=1 \
./scripts/sycl-gptoss-e2e-profile-matrix.sh \
  --run --i-understand-this-runs-gpu-models \
  --device-selector level_zero:1 \
  --out-dir "${OUT_DIR}"
```

Record a `## Lead Launch Timing Evidence` section with the output directory, parsed `profile.mxfp4_tg.launch.submit_us`, `profile.mxfp4_tg.launch.device_us`, and `profile.mxfp4_tg.launch.wait_us` values from the baseline case. Decide `launch-graphlets-worthwhile` only if submit+wait is more than 20% of gate/up time; otherwise record `launch-graphlets-not-first-target`.

**Commit:**

```bash
git add ggml/src/ggml-sycl/mmvq.cpp docs/plans/2026-07-01-sycl-mxfp4-gateup-layout-load-optimization.md
git commit -m "feat(sycl): add default-off MXFP4 gateup launch timing profile"
```

**Gotchas:**

- Event profiling can throw when queue profiling is unavailable; catch and leave device_us at zero.
- Do not enable graphlets or persistent kernels in this task.
- The profiling wait changes timing; it is diagnostic-only and must be env-gated.

## Lead Launch Timing Evidence

| Field | Value |
| --- | --- |
| Decision | `launch-graphlets-not-first-target` |
| Output directory | `/tmp/sycl_gptoss_launch_timing_20260701_091536` |
| Baseline parse path | `/tmp/sycl_gptoss_launch_timing_20260701_091536/baseline/parse.stdout` |
| Baseline model result | `pp512 1238.87 +/- 8.32 tok/s`, `tg128 36.03 +/- 0.08 tok/s` |
| Parsed `profile.mxfp4_tg.launch.submit_us` | `73723.231` |
| Parsed `profile.mxfp4_tg.launch.device_us` | `3504298.096` |
| Parsed `profile.mxfp4_tg.launch.wait_us` | `3546877.552` |
| Raw summed gate/up time | `3642520.000 us` from 641 `[MXFP4-MOE-TG-PROFILE]` lines in `baseline/bench.stderr` |
| Derived non-device launch/drain overhead | `116302.687 us` = submit + max(wait - device, 0) |
| Derived overhead ratio | `3.19%` of raw summed gate/up time |

The literal `submit_us + wait_us` total is larger than 20% of gate/up time, but the measured `wait_us` is the diagnostic drain around each kernel and is almost entirely covered by `device_us`. Treating the full wait as launch overhead would misclassify device execution as host launch overhead. The non-device launch/drain component is only `3.19%`, so graphlets/persistent launch reduction is not the first target for the TG128 gap.

---

## Task 9: Grouped-Reuse Histogram Parser And Source Gates

**Track:** D
**Depends on:** Task 5 and Task 7 (same source-test/parser file serialization)
**File scope:**
- Modify: `scripts/parse-sycl-moe-profile.py`
- Modify: `tests/test-sycl-moe-profile-parser.py`
- Modify: `tests/test-sycl-moe-gateup-optimization-path-source.py`

**Description:**
Add parser support for MoE grouped-reuse histogram evidence so server/batched decode can be evaluated without implementing a new kernel. This captures the grouped-route/source-facts path and the server/continuous-batching grouped-reuse path from the five-path map: do real batches naturally form same-expert groups large enough to use existing grouped kernels?

**Acceptance Criteria:**

- [ ] Parser accepts `[MOE-EXPERT-HIST]` lines with `tokens=`, `topk=`, `total_batches=`, `groups=`, `max_group=`, `avg_group=`, and `groups_ge2=`.
- [ ] Parser prints total lines, max `max_group`, and weighted average group size.
- [ ] Source test proves future instrumentation is env-gated by `GGML_SYCL_MOE_EXPERT_HIST=1`.

**Implementation Guide:**

1. **RED: add parser test.**

Append to `tests/test-sycl-moe-profile-parser.py`:

```python

def test_parser_summarizes_moe_expert_histograms(tmp_path: pathlib.Path) -> None:
    log = tmp_path / "hist.log"
    log.write_text(
        "[MOE-EXPERT-HIST] tensor=blk.0.ffn_gate_exps tokens=4 topk=4 total_batches=16 groups=10 max_group=3 avg_group=1.600 groups_ge2=4\n"
        "[MOE-EXPERT-HIST] tensor=blk.1.ffn_gate_exps tokens=8 topk=4 total_batches=32 groups=18 max_group=5 avg_group=1.778 groups_ge2=9\n",
        encoding="utf-8",
    )
    out = run_parser(log)
    assert "profile.moe_expert_hist.lines 2" in out
    assert "profile.moe_expert_hist.max_group 5" in out
    assert "profile.moe_expert_hist.groups_ge2 13" in out
    assert "profile.moe_expert_hist.avg_group 1.714" in out
```

Run:

```bash
python3 -m pytest tests/test-sycl-moe-profile-parser.py::test_parser_summarizes_moe_expert_histograms -q
```

Expected: FAIL because parser does not emit these lines.

2. **GREEN: update parser.**

In `scripts/parse-sycl-moe-profile.py`, when scanning lines, detect prefix `[MOE-EXPERT-HIST]`. Parse key/value pairs with the same line-oriented helper used for MXFP4 profile lines. Accumulate:

```python
moe_hist = {
    "lines": 0,
    "total_batches": 0,
    "groups": 0,
    "max_group": 0,
    "groups_ge2": 0,
}
```

For each hist line:

```python
moe_hist["lines"] += 1
moe_hist["total_batches"] += int(fields.get("total_batches", 0))
moe_hist["groups"] += int(fields.get("groups", 0))
moe_hist["max_group"] = max(moe_hist["max_group"], int(fields.get("max_group", 0)))
moe_hist["groups_ge2"] += int(fields.get("groups_ge2", 0))
```

Print when `lines > 0`:

```python
avg_group = moe_hist["total_batches"] / moe_hist["groups"] if moe_hist["groups"] else 0.0
print(f"profile.moe_expert_hist.lines {moe_hist['lines']}")
print(f"profile.moe_expert_hist.total_batches {moe_hist['total_batches']}")
print(f"profile.moe_expert_hist.groups {moe_hist['groups']}")
print(f"profile.moe_expert_hist.max_group {moe_hist['max_group']}")
print(f"profile.moe_expert_hist.groups_ge2 {moe_hist['groups_ge2']}")
print(f"profile.moe_expert_hist.avg_group {avg_group:.3f}")
```

3. **GREEN: add source guard.**

Append to `tests/test-sycl-moe-gateup-optimization-path-source.py`:

```python

def test_expert_histogram_instrumentation_is_default_off_when_implemented() -> None:
    mmvq = MMVQ.read_text(encoding="utf-8")
    if "GGML_SYCL_MOE_EXPERT_HIST" not in mmvq:
        return
    assert "mxfp4_moe_expert_hist_enabled" in mmvq
    assert "std::getenv(\"GGML_SYCL_MOE_EXPERT_HIST\")" in mmvq
    assert "[MOE-EXPERT-HIST]" in mmvq
```

4. **Run checks.**

```bash
python3 -m pytest tests/test-sycl-moe-profile-parser.py::test_parser_summarizes_moe_expert_histograms tests/test-sycl-moe-gateup-optimization-path-source.py -q
git diff --check
```

Expected: tests pass.

**Commit:**

```bash
git add scripts/parse-sycl-moe-profile.py tests/test-sycl-moe-profile-parser.py tests/test-sycl-moe-gateup-optimization-path-source.py
git commit -m "test(sycl): add MoE expert histogram parser gates"
```

**Gotchas:**

- Do not add runtime logging in this task.
- Keep parsing line-oriented; avoid regex over the whole file.

---

## Task 10: Grouped-Reuse Histogram Instrumentation And Lead Evidence

**Track:** D + Lead evidence
**Depends on:** Task 9 and Task 6 with `layout-v2-rejected`
**File scope:**
- Modify: `ggml/src/ggml-sycl/mmvq.cpp:16887-17058`
- Modify: `docs/plans/2026-07-01-sycl-mxfp4-gateup-layout-load-optimization.md`

**Description:**
Add default-off host-side expert histogram logging to the pair-GLU dispatch. This determines whether real batched/server decode can feed the existing grouped route.

**Acceptance Criteria:**

- [ ] Logging is gated by `GGML_SYCL_MOE_EXPERT_HIST=1`.
- [ ] Logging does not allocate device memory and does not change dispatch decisions.
- [ ] Histogram emits `tokens`, `topk`, `total_batches`, `groups`, `max_group`, `avg_group`, and `groups_ge2`.
- [ ] Lead evidence records whether grouped/server batching is promising.

**Implementation Guide:**

1. **RED: run source guard before implementation.**

```bash
python3 -m pytest tests/test-sycl-moe-gateup-optimization-path-source.py::test_expert_histogram_instrumentation_is_default_off_when_implemented -q
```

Expected: PASS by early return before implementation. After implementation it must pass by positive assertions.

2. **GREEN: add env helper.**

Near other `mxfp4_moe_*` helpers in `mmvq.cpp`, add:

```cpp
static bool mxfp4_moe_expert_hist_enabled() {
    const char * env = std::getenv("GGML_SYCL_MOE_EXPERT_HIST");
    return env && std::atoi(env) != 0;
}
```

3. **GREEN: add host histogram logging where `ids_host` is available.**

In `mmvq_moe_batched_dispatch_pair_glu_mxfp4_soa()` before `grouped_decode_shape` branch work, add:

```cpp
            if (mxfp4_moe_expert_hist_enabled() && ids_host && ids_host_count == total_batches &&
                gate_weight->ne[2] > 0 && n_ids > 0 && num_tokens > 0) {
                std::vector<int32_t> counts(static_cast<size_t>(gate_weight->ne[2]), 0);
                for (int64_t row = 0; row < total_batches; ++row) {
                    const int32_t eid = ids_host[row];
                    if (eid >= 0 && eid < gate_weight->ne[2]) {
                        counts[static_cast<size_t>(eid)]++;
                    }
                }
                int groups = 0;
                int max_group = 0;
                int groups_ge2 = 0;
                for (int32_t count : counts) {
                    if (count > 0) {
                        groups++;
                        max_group = std::max(max_group, static_cast<int>(count));
                        if (count >= 2) {
                            groups_ge2++;
                        }
                    }
                }
                const double avg_group = groups > 0 ? static_cast<double>(total_batches) / static_cast<double>(groups) : 0.0;
                fprintf(stderr,
                        "[MOE-EXPERT-HIST] tensor=%s tokens=%lld topk=%lld total_batches=%lld groups=%d "
                        "max_group=%d avg_group=%.3f groups_ge2=%d\n",
                        gate_weight->name ? gate_weight->name : "?", (long long) num_tokens, (long long) n_ids,
                        (long long) total_batches, groups, max_group, avg_group, groups_ge2);
            }
```

4. **Run safe checks.**

```bash
python3 -m pytest tests/test-sycl-moe-gateup-optimization-path-source.py tests/test-sycl-moe-profile-parser.py -q
set +u
source /opt/intel/oneapi/setvars.sh --force >/tmp/expert_hist_setvars.log 2>&1
set -u
./scripts/sycl-build.sh llama-bench
git diff --check
```

Expected: tests pass and build succeeds. Do not run model commands.

5. **GREEN: lead-only grouped histogram evidence command.**

Only the lead may run this B50/model profile; workers must not run it. Use the existing acknowledged harness with the histogram env enabled:

```bash
cd /Apps/llama.cpp-mxfp4-tg-runtime
OUT_DIR=/tmp/sycl_gptoss_expert_hist_$(date +%Y%m%d_%H%M%S)
GGML_SYCL_MOE_EXPERT_HIST=1 \
./scripts/sycl-gptoss-e2e-profile-matrix.sh \
  --run --i-understand-this-runs-gpu-models \
  --device-selector level_zero:1 \
  --out-dir "${OUT_DIR}"
```

Record `## Lead Grouped-Reuse Histogram Evidence` with the output directory and parser values for `profile.moe_expert_hist.lines`, `profile.moe_expert_hist.max_group`, `profile.moe_expert_hist.groups_ge2`, and `profile.moe_expert_hist.avg_group`. Decision is `server-grouping-promising` only if real batches show `max_group >= 4` and substantial `groups_ge2` across layers; otherwise record `server-grouping-not-single-stream-fix`.

**Commit:**

```bash
git add ggml/src/ggml-sycl/mmvq.cpp docs/plans/2026-07-01-sycl-mxfp4-gateup-layout-load-optimization.md
git commit -m "feat(sycl): add default-off MoE expert histogram profile"
```

**Gotchas:**

- This logging is host-only and requires `ids_host`; if `ids_host` is absent it must do nothing.
- Do not force host ID copies to make the histogram work.
- Do not change `grouped_decode_shape` eligibility.

## Lead Grouped-Reuse Histogram Evidence

| Field | Value |
| --- | --- |
| Decision | `server-grouping-not-single-stream-fix` |
| Output directory | `/tmp/sycl_gptoss_expert_hist_20260701_095825` |
| Baseline parse path | `/tmp/sycl_gptoss_expert_hist_20260701_095825/baseline/parse.stdout` |
| Baseline model result | `pp512 1237.30 +/- 4.48 tok/s`, `tg128 36.18 +/- 0.05 tok/s` |
| Baseline `diag.path.packed-q8-m2` | `641` |
| Raw `[MOE-EXPERT-HIST]` lines, baseline | `0` |
| Raw `[MOE-EXPERT-HIST]` lines, all harness cases | `0` for `baseline`, `graph_disabled`, `fa_kv_detail`, `vram_pressure`, and `cpu_sharing` |
| Parsed `profile.moe_expert_hist.lines` | not emitted because no histogram lines were present |
| Parsed `profile.moe_expert_hist.max_group` | not emitted because no histogram lines were present |
| Parsed `profile.moe_expert_hist.groups_ge2` | not emitted because no histogram lines were present |
| Parsed `profile.moe_expert_hist.avg_group` | not emitted because no histogram lines were present |

`GGML_SYCL_MOE_EXPERT_HIST=1` was enabled for the acknowledged B50/model profile, but the host-only logger intentionally does nothing unless `ids_host` is already available. The canonical single-stream B50 run continued through the `packed-q8-m2` path and produced no host-ID histogram evidence. Per the task constraint, no host ID copies were forced and `grouped_decode_shape` eligibility was not changed. This evidence does not rule out future server/continuous-batching work with host IDs, but it does not identify a grouped-reuse fix for the current single-stream TG128 gap.

---

## Task 11: Layout Lifecycle Proof For Non-Duplicate Runtime Planning

**Track:** E
**Depends on:** Task 6 with `layout-v2-authorized`
**File scope:**
- Modify: `tests/test-sycl-moe-gateup-optimization-path-source.py`
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp:46351-46440`
- Modify: `docs/plans/2026-07-01-sycl-mxfp4-gateup-layout-load-optimization.md`

**Description:**
If V2 synthetic is authorized, add default-off diagnostics proving that a future canonical gate/up layout can be phase-materialized without retaining duplicate SOA and V2/XMX layouts. This is prerequisite evidence for any runtime/planner plan.

**Acceptance Criteria:**

- [ ] Source test proves lifecycle diagnostics are gated by `GGML_SYCL_MOE_LAYOUT_LIFECYCLE_DEBUG=1`.
- [ ] Diagnostics report SOA present before materialization, XMX/V2 completion, release attempt, and still-complete flag.
- [ ] No forced release or eviction is added.
- [ ] No runtime route uses V2 yet.
- [ ] Lead-only evidence records at least one `[MOE-LAYOUT-LIFECYCLE]` line with `still_complete=1` or records why lifecycle proof failed.

**Implementation Guide:**

1. **RED: add source test.**

Append to `tests/test-sycl-moe-gateup-optimization-path-source.py`:

```python

def test_layout_lifecycle_debug_is_default_off_when_implemented() -> None:
    sycl_cpp = (ROOT / "ggml" / "src" / "ggml-sycl" / "ggml-sycl.cpp").read_text(encoding="utf-8")
    if "GGML_SYCL_MOE_LAYOUT_LIFECYCLE_DEBUG" not in sycl_cpp:
        return
    assert "ggml_sycl_moe_layout_lifecycle_debug_enabled" in sycl_cpp
    assert "std::getenv(\"GGML_SYCL_MOE_LAYOUT_LIFECYCLE_DEBUG\")" in sycl_cpp
    assert "[MOE-LAYOUT-LIFECYCLE]" in sycl_cpp
    assert "still_complete" in sycl_cpp
    assert "released_layout" in sycl_cpp
```

Run:

```bash
python3 -m pytest tests/test-sycl-moe-gateup-optimization-path-source.py::test_layout_lifecycle_debug_is_default_off_when_implemented -q
```

Expected: PASS by early return before implementation; positive assertions after implementation.

2. **GREEN: add env helper in `ggml-sycl.cpp`.**

Near `ggml_sycl_moe_phase_release_source_after_xmx_enabled()`, add:

```cpp
static bool ggml_sycl_moe_layout_lifecycle_debug_enabled() {
    const char * env = std::getenv("GGML_SYCL_MOE_LAYOUT_LIFECYCLE_DEBUG");
    return env && std::atoi(env) != 0;
}
```

3. **GREEN: add diagnostic log only.**

In `ggml_sycl_materialize_moe_phase_layouts()` around `ggml/src/ggml-sycl/ggml-sycl.cpp:46351-46440`, after `released_layout`, `still_complete`, and `source_released` are computed, add:

```cpp
                    if (ggml_sycl_moe_layout_lifecycle_debug_enabled()) {
                        fprintf(stderr,
                                "[MOE-LAYOUT-LIFECYCLE] tensor=%s device=%d had_soa_source=%d xmx_complete=%d "
                                "released_layout=%d still_complete=%d source_released=%d graph_leases=%zu\n",
                                member->name ? member->name : "?", ctx.device, had_soa_source ? 1 : 0,
                                xmx_complete ? 1 : 0, released_layout ? 1 : 0, still_complete ? 1 : 0,
                                source_released ? 1 : 0, released_graph_leases);
                    }
```

4. **GREEN: run safe checks.**

```bash
python3 -m pytest tests/test-sycl-moe-gateup-optimization-path-source.py -q
set +u
source /opt/intel/oneapi/setvars.sh --force >/tmp/layout_lifecycle_setvars.log 2>&1
set -u
./scripts/sycl-build.sh llama-bench
git diff --check
```

Expected: tests pass and build succeeds.

5. **GREEN: lead-only lifecycle proof command.**

Only the lead may run this B50/model profile; workers must not run it. Use the acknowledged harness with lifecycle diagnostics enabled:

```bash
cd /Apps/llama.cpp-mxfp4-tg-runtime
OUT_DIR=/tmp/sycl_gptoss_layout_lifecycle_$(date +%Y%m%d_%H%M%S)
GGML_SYCL_MOE_LAYOUT_LIFECYCLE_DEBUG=1 \
./scripts/sycl-gptoss-e2e-profile-matrix.sh \
  --run --i-understand-this-runs-gpu-models \
  --device-selector level_zero:1 \
  --out-dir "${OUT_DIR}"
rg -n "\[MOE-LAYOUT-LIFECYCLE\].*still_complete=1" "${OUT_DIR}"/baseline/bench.stderr
```

Record a `## Lead Layout Lifecycle Evidence` section in this plan with the output directory and representative lifecycle log lines. If the `rg` command finds no `still_complete=1` lines, record `layout-lifecycle-proof-failed` and do not authorize any runtime/planner promotion from this plan.

**Commit:**

```bash
git add tests/test-sycl-moe-gateup-optimization-path-source.py ggml/src/ggml-sycl/ggml-sycl.cpp docs/plans/2026-07-01-sycl-mxfp4-gateup-layout-load-optimization.md
git commit -m "test(sycl): add MoE layout lifecycle debug proof"
```

**Gotchas:**

- Do not call `ggml_sycl_release_moe_tensor_layout()` in new places.
- Do not add forced eviction or forced reap.
- This is diagnostic evidence only; a separate plan must own runtime layout selection.

---

## Task 12: Final Evidence And Documentation

**Track:** A + Lead
**Depends on:** Task 6 and any executed evidence fallback tasks
**File scope:**
- Modify: `docs/backend/SYCL.md:920-935`
- Modify: `docs/plans/2026-07-01-sycl-mxfp4-gateup-layout-load-optimization.md`
- Update tracker tasks created for this plan.

**Description:**
Record the final outcome of the layout-load research plan. The docs must distinguish benchmark-only candidates, diagnostics, rejected paths, and any authorized next plan.

**Acceptance Criteria:**

- [ ] SPEC review passes before QUALITY review.
- [ ] Safe gates pass.
- [ ] Docs record measured V2 synthetic decision.
- [ ] If V2 rejected, docs record launch/histogram evidence status and no runtime route.
- [ ] If V2 authorized, docs record that runtime/planner work still requires a separate plan and layout lifecycle proof.
- [ ] Parent tracker comment includes synthetic log, baseline latency, V2 latency, validation result, launch/histogram evidence if run, and decision.

**Implementation Guide:**

1. **RED: prove final evidence has not already been recorded.**

```bash
cd /Apps/llama.cpp-mxfp4-tg-runtime
! rg -n "^## Final Validation Evidence$" docs/plans/2026-07-01-sycl-mxfp4-gateup-layout-load-optimization.md
```

Expected: command succeeds before this task because the final evidence section is absent.

2. **GREEN: run safe gates.**

```bash
cd /Apps/llama.cpp-mxfp4-tg-runtime
python3 -m pytest tests/test-sycl-moe-gateup-optimization-path-source.py tests/test-sycl-moe-profile-parser.py -q
set +u
source /opt/intel/oneapi/setvars.sh --force >/tmp/v2_final_setvars.log 2>&1
set -u
./scripts/sycl-build.sh llama-bench
./scripts/sycl-build.sh sycl-kernel-bench
git diff --check
```

Expected: all pass.

3. **GREEN: update docs.**

In `docs/backend/SYCL.md` near the existing MXFP4 gate/up notes at lines `920-935`, add a paragraph matching the final decision. If V2 is rejected, the paragraph must state that the benchmark-only `XMX_TILED_V2` aligned-payload gate/up layout candidate was rejected before runtime wiring and must include the actual V2 latency, actual packed-Q8 M2 baseline latency, and actual max absolute error from the lead synthetic JSONL. If V2 is authorized, the paragraph must state that the benchmark-only candidate met the B50 synthetic continue gate but is not a runtime route yet, and that runtime promotion requires a separate planner-owned implementation that proves no persistent duplicate gate/up VRAM layout and passes canonical GPT-OSS correctness, PP512, TG128, fatal-marker, route-evidence, and gate/up profile gates.

4. **GREEN: update plan evidence.**

Add a `## Final Validation Evidence` section to this plan. It must include a table with rows for V2 synthetic log, baseline latency, V2 latency, V2 max absolute error, launch timing evidence, grouped histogram evidence, runtime route, and final decision. Every value must be an actual measurement or an explicit reason that the evidence was not run; no angle-bracket tokens, blank values, or example values may be committed.

5. **GREEN: run reviews in order.**

Run a fresh SPEC review first. Only after SPEC returns PASS, run a fresh QUALITY review. Treat every reviewer issue as a blocker and edit the docs before proceeding.

6. **GREEN: prepare commit and push.**

**Commit:**

```bash
git add docs/backend/SYCL.md docs/plans/2026-07-01-sycl-mxfp4-gateup-layout-load-optimization.md
git commit -m "docs(sycl): record MXFP4 gateup layout-load evidence"
git pull --rebase
bd sync
git push
git status --short --branch
```

Expected: push succeeds and final status is clean.

**Gotchas:**

- Do not claim full-model speedup from synthetic evidence alone.
- Do not document unavailable env flags as usable runtime features.
- Restore `.beads/beads.db` and `.beads/issues.jsonl` after `bd sync` if tracker export dirties the worktree.
- Remove `.codescout` artifacts before final status.

## Final Validation Evidence

| Evidence | Result |
| --- | --- |
| V2 synthetic log | `/tmp/v2_gateup_synth.jsonl` |
| Baseline latency | `237.084865 us` for `mxfp4_pair_glu_xmx_tiled_packed_r8_m2_sparse32_bias` |
| Baseline max absolute error | `0.000000` |
| V2 latency | `251.179255 us` for `mxfp4_pair_glu_xmx_tiled_v2_packed_r8_m2_sparse32_bias` |
| V2 max absolute error | `0.000000` |
| V2 synthetic decision | `layout-v2-rejected`: V2 was exact but slower than baseline and missed the `<= 188.47 us` continue gate. |
| Launch timing evidence | `/tmp/sycl_gptoss_launch_timing_20260701_091536`; baseline parse reported `launch_submit_us=73723.231`, `launch_device_us=3504298.096`, `launch_wait_us=3546877.552`, and derived non-device launch/drain overhead `3.19%` of raw summed gate/up time. Decision: `launch-graphlets-not-first-target`. |
| Grouped histogram evidence | `/tmp/sycl_gptoss_expert_hist_20260701_095825`; no `[MOE-EXPERT-HIST]` lines were emitted in baseline or fallback harness cases because the host-only logger requires existing `ids_host`. Decision: `server-grouping-not-single-stream-fix`. |
| Layout lifecycle evidence | Not run; Task 11 depends on `layout-v2-authorized`, but Task 6 produced `layout-v2-rejected`. |
| Runtime route | None authorized. `XMX_TILED_V2` remains benchmark-only; launch timing and expert histogram paths are default-off diagnostics only. |
| Final decision | Do not promote V2, graphlets, grouped reuse, or persistent duplicate gate/up layouts from this plan. A future runtime/planner plan must start from new evidence and still prove correctness, route evidence, performance gates, and unified-cache/mem_handle lifecycle safety. |

---

## Execution Notes

- This plan intentionally prioritizes layout/load-path research over DPAS-column tricks because role-column fusion is invalid and both single-column and same-expert multi-RHS candidates were exact but slower.
- The `XMX_TILED_V2` candidate is benchmark-only. Runtime/planner wiring requires a later plan only after `layout-v2-authorized`.
- Launch timing and expert histograms are evidence fallbacks. They are default-off diagnostics and must not change routing or correctness.
- Server/batched grouped reuse may improve throughput serving but is not expected to fix single-stream TG128 unless real batches produce enough same-expert rows.
- All worker tasks are safe source/build tasks only. Lead owns synthetic and model execution.
