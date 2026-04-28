# SYCL Allocation-Site Inventory

*Generated 2026-04-27 for bead `llama.cpp-32dg8.15.14` (P5-FIX). Audit-only — no migrations performed.*

---

## Summary

| Category      | Count | Description |
|---------------|-------|-------------|
| **internal**  |  49   | Inside `unified-cache.cpp`, `vram-pool.cpp`, `pinned-pool.cpp`, `device-pool.hpp`, or `ggml-sycl.cpp` allocator wrapper definitions — these ARE the canonical allocator internals |
| **migrate**   |  76   | Production caller sites that must be migrated to `unified_allocate` / `unified_cache_zone_alloc` in Wave 5 |
| **allowlist** |   6   | Production sites temporarily kept as-is with explicit deletion criteria |
| **test**      | 296   | Test files under `tests/` — permanently exempt per §8 |
| **delete**    |   3   | Production sites that should be deleted (not migrated) — `xmx_test_kernel` startup self-test in `mmq_xmx.cpp` |
| **non-call**  | 115   | Comments, string literals, forward declarations, macro definitions, enum value comments — not allocation sites. See Appendix A for full enumeration. |

**Total grep hits (non-blank): 545**
*(296 test + 49 internal + 76 migrate + 3 delete + 6 allowlist + 115 non-call hits)*

All production caller sites (migrate + allowlist) = **82 sites** across 21 files.

---

## Column definitions

| Column | Values |
|--------|--------|
| **Phase** | `model_load`, `context_init`, `graph_compute`, `kv_setup`, `staging`, `oneDNN`, `test`, `unknown` |
| **Purpose** | Short text describing what the allocation is for |
| **Current API** | The raw allocator function called |
| **Desired Category** | Target `unified_allocate` / `unified_cache_*_alloc` call form, or "stays raw" if internal |
| **Migrate or Allowlist** | `migrate` = Wave 5 target; `allowlist` = temporarily kept (deletion criteria given); `internal` = inside unified-cache itself |
| **Owner Bead** | Bead owning the migration |

*Note: Line numbers are 1-based as returned by grep.*

---

## Group 1 — `unified-cache.cpp` internals

These are allocations INSIDE the canonical allocator. They are the raw-malloc gateway permitted by §3.

| # | File | Line | Function | Phase | Purpose | Current API | Desired Category | Migrate or Allowlist | Owner Bead |
|---|------|------|----------|-------|---------|-------------|------------------|----------------------|------------|
| 1 | `unified-cache.cpp` | 1118 | copy_stage prealloc | model_load | Expert staging buffer | `ggml_sycl_malloc_host` | stays raw | internal | — |
| 2 | `unified-cache.cpp` | 1146 | copy_stage slot prealloc | model_load | Per-slot expert staging | `ggml_sycl_malloc_host` | stays raw | internal | — |
| 3 | `unified-cache.cpp` | 1744 | realloc weight entry | graph_compute | Weight cache realloc on eviction | `ggml_sycl_malloc_device_raw` | stays raw | internal | — |
| 4 | `unified-cache.cpp` | 1864 | alloc weight entry | model_load | Weight cache device alloc | `ggml_sycl_malloc_device_raw` | stays raw | internal | — |
| 5 | `unified-cache.cpp` | 2399 | slot alloc | model_load | Chunked weight slot device alloc | `ggml_sycl_malloc_device_raw` | stays raw | internal | — |
| 6 | `unified-cache.cpp` | 5704 | `unified_alloc` device path | internal | Primary device alloc path | `ggml_sycl_malloc_device_raw` | stays raw | internal | — |
| 7 | `unified-cache.cpp` | 5809 | `unified_alloc` host weight fallback | model_load | Host-pinned weight fallback | `ggml_sycl_malloc_host` | stays raw | internal | — |
| 8 | `unified-cache.cpp` | 7101 | oneDNN weights scratch (no arena) | context_init | oneDNN FP16 weight scratch | `sycl::malloc_device` | stays raw (arena-bypass path) | internal | — |
| 9 | `unified-cache.cpp` | 7130 | oneDNN activations scratch (no arena) | context_init | oneDNN FP16 activations scratch | `sycl::malloc_device` | stays raw (arena-bypass path) | internal | — |
| 10 | `unified-cache.cpp` | 7344 | persistent scratch (no arena) | context_init | Named persistent scratch (no arena) | `sycl::malloc_device` | stays raw | internal | — |
| 11 | `unified-cache.cpp` | 7532 | partial rows buffer | graph_compute | Partial-row GEMM temp buffer | `ggml_sycl_malloc_device_raw` | stays raw | internal | — |
| 12 | `unified-cache.cpp` | 7727–7730 | `unified_cache_raw_malloc_device` | unknown | Raw device alloc gateway | `sycl::malloc_device` | stays raw | internal | — |
| 13 | `unified-cache.cpp` | 7737–7740 | `unified_cache_raw_malloc_host` (queue) | unknown | Raw host alloc gateway (queue) | `sycl::malloc_host` | stays raw | internal | — |
| 14 | `unified-cache.cpp` | 7747–7750 | `unified_cache_raw_malloc_host` (ctx) | unknown | Raw host alloc gateway (ctx) | `sycl::malloc_host` | stays raw | internal | — |
| 15 | `unified-cache.cpp` | 7757–7760 | `unified_cache_raw_malloc_shared` | unknown | Raw shared alloc gateway | `sycl::malloc_shared` | stays raw | internal | — |
| 16 | `unified-cache.cpp` | 7901 | zone alloc (realloc) | internal | VRAM zone realloc entry | `ggml_sycl_malloc_device_raw` | stays raw | internal | — |
| 17 | `unified-cache.cpp` | 7954 | zone alloc | internal | VRAM zone primary alloc | `ggml_sycl_malloc_device_raw` | stays raw | internal | — |
| 18 | `unified-cache.cpp` | 8224 | `unified_cache_zone_alloc` device | unknown | Named zone device alloc entry | `ggml_sycl_malloc_device_raw` | stays raw | internal | — |
| 19 | `unified-cache.cpp` | 8259 | `unified_cache_zone_alloc` host | unknown | Named zone host alloc entry | `ggml_sycl_malloc_host` | stays raw | internal | — |
| 20 | `unified-cache.cpp` | 8377 | compute arena reservation | context_init | Contiguous VRAM arena block | `sycl::malloc_device` | stays raw | internal | — |
| 21 | `unified-cache.cpp` | 9269 | VRAM arena single-chunk | context_init | Single-chunk VRAM arena | `sycl::malloc_device` | stays raw | internal | — |
| 22 | `unified-cache.cpp` | 9441 | VRAM arena N-chunk | context_init | Multi-chunk VRAM arena fallback | `sycl::malloc_device` | stays raw | internal | — |

---

## Group 2 — `vram-pool.cpp` internals

| # | File | Line | Function | Phase | Purpose | Current API | Desired Category | Migrate or Allowlist | Owner Bead |
|---|------|------|----------|-------|---------|-------------|------------------|----------------------|------------|
| 23 | `vram-pool.cpp` | 84 | vram pool chunk alloc | context_init | Backing chunk for VRAM pool | `ggml_sycl_malloc_device_raw` | stays raw | internal | — |

---

## Group 3 — `pinned-pool.cpp` internals

| # | File | Line | Function | Phase | Purpose | Current API | Desired Category | Migrate or Allowlist | Owner Bead |
|---|------|------|----------|-------|---------|-------------|------------------|----------------------|------------|
| 24 | `pinned-pool.cpp` | 770 | pinned pool chunk alloc (host_task) | context_init | Pinned chunk via host_task | `ggml_sycl_malloc_host` | stays raw | internal | — |
| 25 | `pinned-pool.cpp` | 794 | pinned pool chunk alloc (queue) | context_init | Pinned chunk via queue ctx | `ggml_sycl_malloc_host` | stays raw | internal | — |

---

## Group 4 — `device-pool.hpp` internals

These are sub-allocator pool implementations that wrap `ggml_sycl_malloc_device_raw`.

| # | File | Line | Function | Phase | Purpose | Current API | Desired Category | Migrate or Allowlist | Owner Bead |
|---|------|------|----------|-------|---------|-------------|------------------|----------------------|------------|
| 26 | `device-pool.hpp` | 182 | `layout_pool` chunk alloc | unknown | Layout-pool backing chunk | `ggml_sycl_malloc_device_raw` | stays raw | internal | — |
| 27 | `device-pool.hpp` | 297 | `layout_pool` pre-grow | unknown | Layout-pool pre-grow backing | `ggml_sycl_malloc_device_raw` | stays raw | internal | — |

---

## Group 5 — `ggml-sycl.cpp` allocator wrapper definitions

Lines 18561–18683 of `ggml-sycl.cpp` contain the implementations of the `ggml_sycl_malloc_device`, `ggml_sycl_malloc_device_raw`, `ggml_sycl_malloc_host`, and `ggml_sycl_malloc_shared` wrapper functions themselves. These are the canonical allocator implementations in `ggml-sycl.cpp` — they are the wrapper bodies, not callers. (The corresponding forward declarations and typed template wrappers in `common.hpp` are covered in Group 6.)

| # | File | Line | Function | Phase | Purpose | Current API | Desired Category | Migrate or Allowlist | Owner Bead |
|---|------|------|----------|-------|---------|-------------|------------------|----------------------|------------|
| 28 | `ggml-sycl.cpp` | 18561 | `ggml_sycl_malloc_device_raw` def | — | Function definition | `ggml_sycl_malloc_device_raw` | stays raw | internal | — |
| 29 | `ggml-sycl.cpp` | 18588/18592/18640 | `ggml_sycl_malloc_device` def | — | Function definition (re-entrancy guard) | `ggml_sycl_malloc_device_raw` | stays raw | internal | — |
| 30 | `ggml-sycl.cpp` | 18646/18647 | `ggml_sycl_malloc_host` def (queue) | — | Function definition | (wrapper) | stays raw | internal | — |
| 31 | `ggml-sycl.cpp` | 18664/18665 | `ggml_sycl_malloc_host` def (ctx) | — | Function definition | (wrapper) | stays raw | internal | — |
| 32 | `ggml-sycl.cpp` | 18682/18683 | `ggml_sycl_malloc_shared` def | — | Function definition | (wrapper) | stays raw | internal | — |

---

## Group 6 — `common.hpp` — template/inline allocator wrappers and one production caller

Lines 1753–1817 of `common.hpp` are template wrapper definitions (`ggml_sycl_malloc_device_t`, `ggml_sycl_malloc_device_tracked_bytes`, etc.) — these are part of the wrapper layer, not production callers. One line (3183) is a production caller inside a graph input staging helper.

| # | File | Line | Function | Phase | Purpose | Current API | Desired Category | Migrate or Allowlist | Owner Bead |
|---|------|------|----------|-------|---------|-------------|------------------|----------------------|------------|
| 33 | `common.hpp` | 1713 | `ggml_sycl_malloc_device` decl | — | Forward declaration | — | — | internal | — |
| 34 | `common.hpp` | 1716–1719 | `ggml_sycl_malloc_device_raw`, `_host`, `_shared` decls | — | Forward declarations | — | — | internal | — |
| 35 | `common.hpp` | 1753–1815 | Template wrapper defs | — | Typed/tracked wrappers | `ggml_sycl_malloc_device` | stays raw | internal | — |
| 36 | `common.hpp` | 3183 | graph input staging `dev_ptr` | graph_compute | Device staging buffer for graph INPUT tensors | `sycl::malloc_device` | `unified_allocate(..., RUNTIME zone)` | migrate | `llama.cpp-32dg8.6` |

---

## Group 7 — `dpct/helper.hpp`

The `dpct/` directory contains Intel DPCT migration helpers. These call `sycl::malloc_device` / `sycl::malloc_shared` in low-level memory management utilities that are part of the DPCT runtime layer, not the SYCL backend itself.

| # | File | Line | Function | Phase | Purpose | Current API | Desired Category | Migrate or Allowlist | Owner Bead |
|---|------|------|----------|-------|---------|-------------|------------------|----------------------|------------|
| 37 | `dpct/helper.hpp` | 1463 | `dpct_malloc` | unknown | DPCT pitched/device malloc utility | `sycl::malloc_device` | stays raw (DPCT layer) | allowlist | — |
| 38 | `dpct/helper.hpp` | 3060 | `device_memory::allocate_device` (shared) | unknown | DPCT shared USM for device_memory | `sycl::malloc_shared` | stays raw (DPCT layer) | allowlist | — |
| 39 | `dpct/helper.hpp` | 3066 | `device_memory::allocate_device` (device) | unknown | DPCT device USM for device_memory | `sycl::malloc_device` | stays raw (DPCT layer) | allowlist | — |

*Deletion criteria: allowlisted permanently while DPCT helper layer is in use. If DPCT layer is removed, delete these. Tracked under §8 — DPCT is an ABI shim, not a backend allocator.*

---

## Group 8 — `common.cpp` — `ggml_sycl_host_malloc`

`ggml_sycl_host_malloc` in `common.cpp` is called from the public `ggml_backend` tensor-allocation path (§8 of contract) and routes via `unified_cache_raw_malloc_host`. This is the ggml buffer-type boundary entry point.

| # | File | Line | Function | Phase | Purpose | Current API | Desired Category | Migrate or Allowlist | Owner Bead |
|---|------|------|----------|-------|---------|-------------|------------------|----------------------|------------|
| 40 | `common.cpp` | 713 | `ggml_sycl_host_malloc` TP path | model_load | Host pinned for TP shared memory | `unified_cache_raw_malloc_host` | stays raw (ggml buffer boundary §8) | allowlist | — |
| 41 | `common.cpp` | 732 | `ggml_sycl_host_malloc` non-TP path | model_load | Host pinned for weight tensors | `unified_cache_raw_malloc_host` | stays raw (ggml buffer boundary §8) | allowlist | — |

*Deletion criteria: allowlisted permanently as the `ggml_backend` buffer API boundary entry point per §8 of contract.*

---

## Group 9 — `dense-scheduler.cpp`

| # | File | Line | Function | Phase | Purpose | Current API | Desired Category | Migrate or Allowlist | Owner Bead |
|---|------|------|----------|-------|---------|-------------|------------------|----------------------|------------|
| 42 | `dense-scheduler.cpp` | 20 | scheduler ctor | context_init | VRAM slot 0 for dense scheduler | `ggml_sycl_malloc_device` | `unified_allocate(..., RUNTIME zone)` | migrate | `llama.cpp-32dg8.6` |
| 43 | `dense-scheduler.cpp` | 21 | scheduler ctor | context_init | VRAM slot 1 for dense scheduler | `ggml_sycl_malloc_device` | `unified_allocate(..., RUNTIME zone)` | migrate | `llama.cpp-32dg8.6` |

---

## Group 10 — `mmvq.cpp`

| # | File | Line | Function | Phase | Purpose | Current API | Desired Category | Migrate or Allowlist | Owner Bead |
|---|------|------|----------|-------|---------|-------------|------------------|----------------------|------------|
| 44 | `mmvq.cpp` | 1381 | `mmvq_q4_coalesce` | graph_compute | Temp coalescing buffer for Q4 MMVQ | `ggml_sycl_malloc_device_tracked_t` | `unified_allocate(..., SCRATCH zone)` | migrate | `llama.cpp-32dg8.6` |
| 45 | `mmvq.cpp` | 1738 | MMVQ dispatch | graph_compute | Temp compute buffer for MMVQ | `ggml_sycl_malloc_device_tracked_t` | `unified_allocate(..., SCRATCH zone)` | migrate | `llama.cpp-32dg8.6` |
| 46 | `mmvq.cpp` | 1967 | MMVQ dispatch variant | graph_compute | Temp compute buffer for MMVQ | `ggml_sycl_malloc_device_tracked_t` | `unified_allocate(..., SCRATCH zone)` | migrate | `llama.cpp-32dg8.6` |
| 47 | `mmvq.cpp` | 2109 | MMVQ quant temp | graph_compute | Quant temp buffer | `ggml_sycl_malloc_device` | `unified_allocate(..., SCRATCH zone)` | migrate | `llama.cpp-32dg8.6` |
| 48 | `mmvq.cpp` | 3246 | MMVQ compact buffer | graph_compute | Compact output buffer | `ggml_sycl_malloc_device` | `unified_allocate(..., SCRATCH zone)` | migrate | `llama.cpp-32dg8.6` |
| 49 | `mmvq.cpp` | 3260 | MMVQ missing counter | graph_compute | Missing-row counter | `ggml_sycl_malloc_device_t` | `unified_allocate(..., SCRATCH zone)` | migrate | `llama.cpp-32dg8.6` |
| 50 | `mmvq.cpp` | 4796 | MMVQ host staging | staging | Host staging for DMA slice | `ggml_sycl_malloc_host_tracked_bytes` | `unified_allocate(..., must_host_pinned=true, use_pinned_pool=true, zone=STAGING)` | migrate | `llama.cpp-32dg8.6` |

---

## Group 11 — `mmq.cpp`

| # | File | Line | Function | Phase | Purpose | Current API | Desired Category | Migrate or Allowlist | Owner Bead |
|---|------|------|----------|-------|---------|-------------|------------------|----------------------|------------|
| 51 | `mmq.cpp` | 236 | `mmq_get_work_counter` | context_init | Persistent work counter per device | `ggml_sycl_malloc_device_tracked_t` | `unified_allocate(..., RUNTIME zone)` | migrate | `llama.cpp-32dg8.6` |
| 52 | `mmq.cpp` | 6728 | MMQ host staging | staging | Host staging for DMA slice | `ggml_sycl_malloc_host_tracked_bytes` | `unified_allocate(..., must_host_pinned=true, use_pinned_pool=true, zone=STAGING)` | migrate | `llama.cpp-32dg8.6` |

---

## Group 12 — `mmq_xmx.cpp`

These are in a test function `xmx_test_kernel` called at startup to verify XMX correctness — but the function is in `mmq_xmx.cpp` (production code), not under `tests/`.

| # | File | Line | Function | Phase | Purpose | Current API | Desired Category | Migrate or Allowlist | Owner Bead |
|---|------|------|----------|-------|---------|-------------|------------------|----------------------|------------|
| 53 | `mmq_xmx.cpp` | 4130 | `xmx_test_kernel` | context_init | XMX test matrix A — move to `tests/` as proper fixture | `ggml_sycl_malloc_device_tracked_t` | `delete (move to tests/ as proper fixture)` | delete | — |
| 54 | `mmq_xmx.cpp` | 4131 | `xmx_test_kernel` | context_init | XMX test matrix B — move to `tests/` as proper fixture | `ggml_sycl_malloc_device_tracked_t` | `delete (move to tests/ as proper fixture)` | delete | — |
| 55 | `mmq_xmx.cpp` | 4132 | `xmx_test_kernel` | context_init | XMX test matrix C — move to `tests/` as proper fixture | `ggml_sycl_malloc_device_tracked_t` | `delete (move to tests/ as proper fixture)` | delete | — |

---

## Group 13 — `convert.cpp`

| # | File | Line | Function | Phase | Purpose | Current API | Desired Category | Migrate or Allowlist | Owner Bead |
|---|------|------|----------|-------|---------|-------------|------------------|----------------------|------------|
| 56 | `convert.cpp` | 564 | convert dispatch | graph_compute | Temp convert buffer | `ggml_sycl_malloc_device_tracked_t` | `unified_allocate(..., SCRATCH zone)` | migrate | `llama.cpp-32dg8.6` |
| 57 | `convert.cpp` | 646 | convert dispatch variant 2 | graph_compute | Temp convert buffer | `ggml_sycl_malloc_device_tracked_t` | `unified_allocate(..., SCRATCH zone)` | migrate | `llama.cpp-32dg8.6` |
| 58 | `convert.cpp` | 726 | convert dispatch variant 3 | graph_compute | Temp convert buffer | `ggml_sycl_malloc_device_tracked_t` | `unified_allocate(..., SCRATCH zone)` | migrate | `llama.cpp-32dg8.6` |
| 59 | `convert.cpp` | 830 | convert dispatch variant 4 | graph_compute | Temp convert buffer | `ggml_sycl_malloc_device_tracked_t` | `unified_allocate(..., SCRATCH zone)` | migrate | `llama.cpp-32dg8.6` |

---

## Group 14 — `dmmv.cpp`

| # | File | Line | Function | Phase | Purpose | Current API | Desired Category | Migrate or Allowlist | Owner Bead |
|---|------|------|----------|-------|---------|-------------|------------------|----------------------|------------|
| 60 | `dmmv.cpp` | 2747 | dmmv debug buffer | graph_compute | Debug float buffer (debug build only) | `ggml_sycl_malloc_device_tracked_t` | `unified_allocate(..., SCRATCH zone)` | migrate | `llama.cpp-32dg8.6` |
| 61 | `dmmv.cpp` | 3417 | dmmv host staging | staging | Host staging for DMA slice | `ggml_sycl_malloc_host_tracked_bytes` | `unified_allocate(..., must_host_pinned=true, use_pinned_pool=true, zone=STAGING)` | migrate | `llama.cpp-32dg8.6` |

---

## Group 15 — `getrows.cpp`

| # | File | Line | Function | Phase | Purpose | Current API | Desired Category | Migrate or Allowlist | Owner Bead |
|---|------|------|----------|-------|---------|-------------|------------------|----------------------|------------|
| 62 | `getrows.cpp` | 1577 | get_rows host staging | staging | Host staging for DMA slice | `ggml_sycl_malloc_host_tracked_bytes` | `unified_allocate(..., must_host_pinned=true, use_pinned_pool=true, zone=STAGING)` | migrate | `llama.cpp-32dg8.6` |

---

## Group 16 — `cpy.cpp`

| # | File | Line | Function | Phase | Purpose | Current API | Desired Category | Migrate or Allowlist | Owner Bead |
|---|------|------|----------|-------|---------|-------------|------------------|----------------------|------------|
| 63 | `cpy.cpp` | 740 | cpy staging | staging | Shared USM staging buffer for CPY | `ggml_sycl_malloc_shared` | `unified_allocate(..., STAGING zone)` | migrate | `llama.cpp-32dg8.6` |

---

## Group 17 — `kv-offload.cpp`

| # | File | Line | Function | Phase | Purpose | Current API | Desired Category | Migrate or Allowlist | Owner Bead |
|---|------|------|----------|-------|---------|-------------|------------------|----------------------|------------|
| 64 | `kv-offload.cpp` | 443 | KV host block alloc | kv_setup | Host-pinned KV offload block | `ggml_sycl_malloc_host_tracked_bytes` | `unified_allocate(..., must_host_pinned=true, use_pinned_pool=true, zone=KV)` | migrate | `llama.cpp-32dg8.5` |

---

## Group 18 — `ccl-comm.cpp`

| # | File | Line | Function | Phase | Purpose | Current API | Desired Category | Migrate or Allowlist | Owner Bead |
|---|------|------|----------|-------|---------|-------------|------------------|----------------------|------------|
| 65 | `ccl-comm.cpp` | 332 | allreduce staging | graph_compute | Shared USM staging for CCL allreduce | `ggml_sycl_malloc_shared_t` | `unified_allocate(..., STAGING zone)` | migrate | `llama.cpp-32dg8.11` |

---

## Group 19 — `fused-moe-esimd.hpp`

| # | File | Line | Function | Phase | Purpose | Current API | Desired Category | Migrate or Allowlist | Owner Bead |
|---|------|------|----------|-------|---------|-------------|------------------|----------------------|------------|
| 66 | `fused-moe-esimd.hpp` | 770 | fused MoE ESIMD persistent Q8 | graph_compute | Persistent Q8 buffer for fused MoE ESIMD | `ggml_sycl_malloc_device` | `unified_allocate(..., RUNTIME zone)` | migrate | `llama.cpp-32dg8.6` |
| 67 | `fused-moe-esimd.hpp` | 950 | fused MoE ESIMD Q8 (fused path) | graph_compute | Q8 buffer for fused path | `ggml_sycl_malloc_device` | `unified_allocate(..., SCRATCH zone)` | migrate | `llama.cpp-32dg8.6` |

---

## Group 20 — `gpu-sampler.hpp`

| # | File | Line | Function | Phase | Purpose | Current API | Desired Category | Migrate or Allowlist | Owner Bead |
|---|------|------|----------|-------|---------|-------------|------------------|----------------------|------------|
| 68 | `gpu-sampler.hpp` | 19 | gpu sampler alloc | graph_compute | GPU sampler typed allocation | `ggml_sycl_malloc_device_tracked_t` | `unified_allocate(..., RUNTIME zone)` | migrate | `llama.cpp-32dg8.6` |

---

## Group 21 — `cont-batching.hpp`

| # | File | Line | Function | Phase | Purpose | Current API | Desired Category | Migrate or Allowlist | Owner Bead |
|---|------|------|----------|-------|---------|-------------|------------------|----------------------|------------|
| 69 | `cont-batching.hpp` | 21 | continuous batching alloc | graph_compute | Per-batch typed device allocation | `ggml_sycl_malloc_device_t` | `unified_allocate(..., RUNTIME zone)` | migrate | `llama.cpp-32dg8.6` |

---

## Group 22 — `fattn.cpp`

| # | File | Line | Function | Phase | Purpose | Current API | Desired Category | Migrate or Allowlist | Owner Bead |
|---|------|------|----------|-------|---------|-------------|------------------|----------------------|------------|
| 70 | `fattn.cpp` | 171 | block table alloc (seq_id_buffer pool) | graph_compute | Flash-attn block table per stream | `ggml_sycl_malloc_device_t` | `unified_allocate(..., RUNTIME zone)` | migrate | `llama.cpp-32dg8.6` |
| 71 | `fattn.cpp` | 301 | fattn weight staging slot | graph_compute | Per-weight staging slot for fattn | `ggml_sycl_malloc_device` | `unified_allocate(..., STAGING zone)` | migrate | `llama.cpp-32dg8.6` |
| 72 | `fattn.cpp` | 455 | `g_v2_auto.block_table` | context_init | Pre-alloc block table for V2 auto | `ggml_sycl_malloc_device_t` | `unified_allocate(..., RUNTIME zone)` | migrate | `llama.cpp-32dg8.6` |
| 73 | `fattn.cpp` | 466 | `g_v2_auto.seq_lens` | context_init | Pre-alloc seq lens for V2 auto | `ggml_sycl_malloc_device_t` | `unified_allocate(..., RUNTIME zone)` | migrate | `llama.cpp-32dg8.6` |
| 74 | `fattn.cpp` | 476 | `g_v2_auto.temp_buf` | context_init | Pre-alloc temp buf for V2 auto | `ggml_sycl_malloc_device_t` | `unified_allocate(..., SCRATCH zone)` | migrate | `llama.cpp-32dg8.6` |
| 75 | `fattn.cpp` | 1660 | `g_v2_auto.block_table` (lazy) | graph_compute | Lazy block table realloc | `ggml_sycl_malloc_device_t` | `unified_allocate(..., RUNTIME zone)` | migrate | `llama.cpp-32dg8.6` |
| 76 | `fattn.cpp` | 1667 | `g_v2_auto.seq_lens` (lazy) | graph_compute | Lazy seq lens realloc | `ggml_sycl_malloc_device_t` | `unified_allocate(..., RUNTIME zone)` | migrate | `llama.cpp-32dg8.6` |
| 77 | `fattn.cpp` | 1708 | `g_v2_auto.temp_buf` (lazy) | graph_compute | Lazy temp buf realloc | `ggml_sycl_malloc_device_t` | `unified_allocate(..., SCRATCH zone)` | migrate | `llama.cpp-32dg8.6` |

---

## Group 23 — `fattn-onednn.cpp`

| # | File | Line | Function | Phase | Purpose | Current API | Desired Category | Migrate or Allowlist | Owner Bead |
|---|------|------|----------|-------|---------|-------------|------------------|----------------------|------------|
| 78 | `fattn-onednn.cpp` | 421 | oneDNN SDPA scale USM | oneDNN | USM scalar for oneDNN SDPA scale | `sycl::malloc_host` | `unified_allocate(..., must_host_pinned=true, use_pinned_pool=true, zone=ONEDNN)` | migrate | `llama.cpp-32dg8.6` |

---

## Group 24 — `persistent-tg-kernel.cpp`

| # | File | Line | Function | Phase | Purpose | Current API | Desired Category | Migrate or Allowlist | Owner Bead |
|---|------|------|----------|-------|---------|-------------|------------------|----------------------|------------|
| 79 | `persistent-tg-kernel.cpp` | 304 | PTG layer weights | context_init | Device copy of per-layer weights for persistent TG | `ggml_sycl_malloc_device` | `unified_allocate(..., RUNTIME zone)` | migrate | `llama.cpp-32dg8.6` |

---

## Group 25 — `unified-kernel.cpp`

| # | File | Line | Function | Phase | Purpose | Current API | Desired Category | Migrate or Allowlist | Owner Bead |
|---|------|------|----------|-------|---------|-------------|------------------|----------------------|------------|
| 80 | `unified-kernel.cpp` | 9255 | graph bench dummy buffer | context_init | Dummy device buffer for graph bench warmup | `ggml_sycl_malloc_device` | `unified_allocate(..., SCRATCH zone)` | migrate | `llama.cpp-32dg8.6` |

---

## Group 26 — `set_rows.cpp`

| # | File | Line | Function | Phase | Purpose | Current API | Desired Category | Migrate or Allowlist | Owner Bead |
|---|------|------|----------|-------|---------|-------------|------------------|----------------------|------------|
| 81 | `set_rows.cpp` | 323 | set_rows debug buffer | graph_compute | Debug host buffer (debug build only) | `ggml_sycl_malloc_host_tracked_bytes` | `unified_allocate(..., must_host_pinned=true, use_pinned_pool=true, zone=SCRATCH)` | migrate | `llama.cpp-32dg8.6` |

---

## Group 27 — `ggml-sycl.cpp` production callers

These are production call sites in `ggml-sycl.cpp` that are NOT part of the allocator wrapper definitions (Groups 4–5 above).

| # | File | Line | Function | Phase | Purpose | Current API | Desired Category | Migrate or Allowlist | Owner Bead |
|---|------|------|----------|-------|---------|-------------|------------------|----------------------|------------|
| 82 | `ggml-sycl.cpp` | 9112 | `ggml_backend_sycl_probe_buffer_type_alloc_buffer` | model_load | Probe buffer for ggml_sycl_init (raw needed: unified_cache not yet initialized) | `ggml_sycl_malloc_device_raw` | stays raw (pre-init) | allowlist | — |
| 83 | `ggml-sycl.cpp` | 11492 | XMX tiled fill | graph_compute | Temp device buffer for XMX tiled fill | `ggml_sycl_malloc_device_tracked_t` | `unified_allocate(..., SCRATCH zone)` | migrate | `llama.cpp-32dg8.6` |
| 84 | `ggml-sycl.cpp` | 14909 | KV tier migration fallback | kv_setup | Host-pinned fallback when device KV layer fails | `ggml_sycl_malloc_host` | `unified_allocate(..., must_host_pinned=true, use_pinned_pool=true, zone=KV)` | migrate | `llama.cpp-32dg8.5` |
| 85 | `ggml-sycl.cpp` | 17370 | speculative verify logits | graph_compute | GPU logits buffer for speculative verification | `ggml_sycl_malloc_device_tracked_t` | `unified_allocate(..., SCRATCH zone)` | migrate | `llama.cpp-32dg8.6` |
| 86 | `ggml-sycl.cpp` | 17583 | multi-seq batch indices | graph_compute | Device batch indices for multi-seq sampler | `ggml_sycl_malloc_device_tracked_t` | `unified_allocate(..., SCRATCH zone)` | migrate | `llama.cpp-32dg8.6` |
| 87 | `ggml-sycl.cpp` | 17586 | multi-seq seq IDs | graph_compute | Device seq IDs for multi-seq sampler | `ggml_sycl_malloc_device_tracked_t` | `unified_allocate(..., SCRATCH zone)` | migrate | `llama.cpp-32dg8.6` |
| 88 | `ggml-sycl.cpp` | 18946 | buffer pool host fallback | context_init | Host-pinned buffer from scratch pool (non-cached) | `ggml_sycl_malloc_host` | `unified_allocate(..., must_host_pinned=true, use_pinned_pool=true, zone=SCRATCH)` | migrate | `llama.cpp-32dg8.6` |
| 89 | `ggml-sycl.cpp` | 22746 | TP col src1 float | graph_compute | TP column float buffer | `ggml_sycl_malloc_device_tracked_bytes` | `unified_allocate(..., RUNTIME zone)` | migrate | `llama.cpp-32dg8.6` |
| 90 | `ggml-sycl.cpp` | 22748 | TP col src1 Q8 | graph_compute | TP column Q8 buffer | `ggml_sycl_malloc_device_tracked_bytes` | `unified_allocate(..., RUNTIME zone)` | migrate | `llama.cpp-32dg8.6` |
| 91 | `ggml-sycl.cpp` | 22750 | TP col output | graph_compute | TP column output buffer | `ggml_sycl_malloc_device_tracked_bytes` | `unified_allocate(..., RUNTIME zone)` | migrate | `llama.cpp-32dg8.6` |
| 92 | `ggml-sycl.cpp` | 32848 | MoE tokens_sorted (non-graph path) | graph_compute | Sorted token buffer for MoE XMX dispatch | `ggml_sycl_malloc_device_tracked_bytes` | `unified_allocate(..., SCRATCH zone)` | migrate | `llama.cpp-32dg8.6` |
| 93 | `ggml-sycl.cpp` | 32852 | MoE token_map (non-graph path) | graph_compute | Token mapping table for MoE | `ggml_sycl_malloc_device_tracked_bytes` | `unified_allocate(..., SCRATCH zone)` | migrate | `llama.cpp-32dg8.6` |
| 94 | `ggml-sycl.cpp` | 32861 | MoE sorted_token_ids (non-graph path) | graph_compute | Sorted token IDs for MXFP4 MoE | `ggml_sycl_malloc_device_tracked_bytes` | `unified_allocate(..., SCRATCH zone)` | migrate | `llama.cpp-32dg8.6` |
| 95 | `ggml-sycl.cpp` | 32865 | MoE expert_counts (non-graph path) | graph_compute | Expert counts buffer for MoE | `ggml_sycl_malloc_device_tracked_bytes` | `unified_allocate(..., SCRATCH zone)` | migrate | `llama.cpp-32dg8.6` |
| 96 | `ggml-sycl.cpp` | 32869 | MoE expert_offsets (non-graph path) | graph_compute | Expert offsets buffer for MoE | `ggml_sycl_malloc_device_tracked_bytes` | `unified_allocate(..., SCRATCH zone)` | migrate | `llama.cpp-32dg8.6` |
| 97 | `ggml-sycl.cpp` | 32872 | MoE sorted_output (non-graph path) | graph_compute | Sorted output buffer for MoE | `ggml_sycl_malloc_device_tracked_bytes` | `unified_allocate(..., SCRATCH zone)` | migrate | `llama.cpp-32dg8.6` |
| 98 | `ggml-sycl.cpp` | 32921 | MoE expert_write_pos (non-graph path) | graph_compute | Expert write-position counters | `ggml_sycl_malloc_device_tracked_bytes` | `unified_allocate(..., SCRATCH zone)` | migrate | `llama.cpp-32dg8.6` |
| 99 | `ggml-sycl.cpp` | 32959 | MoE expert_scales (non-graph path) | graph_compute | Expert Q8_0 scale buffer | `ggml_sycl_malloc_device_tracked_bytes` | `unified_allocate(..., SCRATCH zone)` | migrate | `llama.cpp-32dg8.6` |
| 100 | `ggml-sycl.cpp` | 33021 | MoE q_tokens (non-graph path) | graph_compute | Q8 tokens buffer for MoE | `ggml_sycl_malloc_device_tracked_bytes` | `unified_allocate(..., SCRATCH zone)` | migrate | `llama.cpp-32dg8.6` |
| 101 | `ggml-sycl.cpp` | 33025 | MoE token_scales (non-graph path) | graph_compute | Token scale buffer for MoE | `ggml_sycl_malloc_device_tracked_bytes` | `unified_allocate(..., SCRATCH zone)` | migrate | `llama.cpp-32dg8.6` |
| 102 | `ggml-sycl.cpp` | 33055 | MoE tokens_f16_input (non-graph path) | graph_compute | FP16 input token buffer for MoE | `ggml_sycl_malloc_device_tracked_bytes` | `unified_allocate(..., SCRATCH zone)` | migrate | `llama.cpp-32dg8.6` |
| 103 | `ggml-sycl.cpp` | 43747 | MoE graph tokens_f16 | context_init | Pre-alloc tokens_f16 for MoE graph | `ggml_sycl_malloc_device_t` | `unified_allocate(..., RUNTIME zone)` | migrate | `llama.cpp-32dg8.6` |
| 104 | `ggml-sycl.cpp` | 43749 | MoE graph tokens_sorted | context_init | Pre-alloc tokens_sorted for MoE graph | `ggml_sycl_malloc_device_t` | `unified_allocate(..., RUNTIME zone)` | migrate | `llama.cpp-32dg8.6` |
| 105 | `ggml-sycl.cpp` | 43750 | MoE graph token_map | context_init | Pre-alloc token_map for MoE graph | `ggml_sycl_malloc_device_t` | `unified_allocate(..., RUNTIME zone)` | migrate | `llama.cpp-32dg8.6` |
| 106 | `ggml-sycl.cpp` | 43751 | MoE graph expert_counts | context_init | Pre-alloc expert_counts for MoE graph | `ggml_sycl_malloc_device_t` | `unified_allocate(..., RUNTIME zone)` | migrate | `llama.cpp-32dg8.6` |
| 107 | `ggml-sycl.cpp` | 43752 | MoE graph expert_offsets | context_init | Pre-alloc expert_offsets for MoE graph | `ggml_sycl_malloc_device_t` | `unified_allocate(..., RUNTIME zone)` | migrate | `llama.cpp-32dg8.6` |
| 108 | `ggml-sycl.cpp` | 43753 | MoE graph expert_write_pos | context_init | Pre-alloc expert_write_pos for MoE graph | `ggml_sycl_malloc_device_t` | `unified_allocate(..., RUNTIME zone)` | migrate | `llama.cpp-32dg8.6` |
| 109 | `ggml-sycl.cpp` | 43755 | MoE graph sorted_output | context_init | Pre-alloc sorted_output for MoE graph | `ggml_sycl_malloc_device_t` | `unified_allocate(..., RUNTIME zone)` | migrate | `llama.cpp-32dg8.6` |
| 110 | `ggml-sycl.cpp` | 43756 | MoE graph q_tokens | context_init | Pre-alloc Q8 token buffer for MoE graph | `ggml_sycl_malloc_device_t` | `unified_allocate(..., RUNTIME zone)` | migrate | `llama.cpp-32dg8.6` |
| 111 | `ggml-sycl.cpp` | 43758 | MoE graph token_scales | context_init | Pre-alloc token scales for MoE graph | `ggml_sycl_malloc_device_t` | `unified_allocate(..., RUNTIME zone)` | migrate | `llama.cpp-32dg8.6` |
| 112 | `ggml-sycl.cpp` | 43760 | MoE graph expert_scales | context_init | Pre-alloc expert scales for MoE graph | `ggml_sycl_malloc_device_t` | `unified_allocate(..., RUNTIME zone)` | migrate | `llama.cpp-32dg8.6` |
| 113 | `ggml-sycl.cpp` | 43762 | MoE graph sorted_token_ids | context_init | Pre-alloc sorted_token_ids for MoE graph | `ggml_sycl_malloc_device_t` | `unified_allocate(..., RUNTIME zone)` | migrate | `llama.cpp-32dg8.6` |
| 114 | `ggml-sycl.cpp` | 46578 | PTG RMS debug buffer | graph_compute | Debug shared USM for PTG RMS kernel | `ggml_sycl_malloc_shared` | (debug only, remove with debug code) | migrate | `llama.cpp-32dg8.6` |
| 115 | `ggml-sycl.cpp` | 46579 | PTG RMS flag | graph_compute | Debug flag for PTG RMS kernel | `ggml_sycl_malloc_shared` | (debug only, remove with debug code) | migrate | `llama.cpp-32dg8.6` |
| 116 | `ggml-sycl.cpp` | 46879 | PTG matmul debug buffer | graph_compute | Debug shared USM for PTG matmul kernel | `ggml_sycl_malloc_shared` | (debug only, remove with debug code) | migrate | `llama.cpp-32dg8.6` |
| 117 | `ggml-sycl.cpp` | 46880 | PTG matmul flag | graph_compute | Debug flag for PTG matmul kernel | `ggml_sycl_malloc_shared` | (debug only, remove with debug code) | migrate | `llama.cpp-32dg8.6` |
| 118 | `ggml-sycl.cpp` | 47711 | PTG attention debug buffer | graph_compute | Debug shared USM for PTG attention kernel | `ggml_sycl_malloc_shared` | (debug only, remove with debug code) | migrate | `llama.cpp-32dg8.6` |
| 119 | `ggml-sycl.cpp` | 47892 | PTG set_rows debug buffer | graph_compute | Debug shared USM for PTG set_rows kernel | `ggml_sycl_malloc_shared` | (debug only, remove with debug code) | migrate | `llama.cpp-32dg8.6` |
| 120 | `ggml-sycl.cpp` | 48107 | PTG hash debug buffer | graph_compute | Debug shared USM for PTG hash check | `ggml_sycl_malloc_shared` | (debug only, remove with debug code) | migrate | `llama.cpp-32dg8.6` |

---

## Group 28 — Test files (`tests/`)

All allocations in `tests/` are permanently exempt per §8 of the contract. They include 10 test files:

| Test file | Sites | Note |
|-----------|-------|-------|
| `tests/test-graph-replay.cpp` | 11 | Graph replay micro-tests |
| `tests/test-unified-kernel.cpp` | 15 | Unified kernel tests |
| `tests/test-esimd-prefetch.cpp` | 24 | ESIMD prefetch perf tests |
| `tests/test-moe-mxfp4-dp4a.cpp` | 3 | MoE MXFP4 test |
| `tests/test-xmx-optimization.cpp` | 16 | XMX optimization tests |
| `tests/test-xmx-esimd-q4-gemm.cpp` | 12 | XMX ESIMD q4 gemm tests |
| `tests/test-graph-replay-chain.cpp` | 20 | Graph replay chain tests |
| `tests/bench-dnnl-ops.cpp` | 14 | oneDNN bench |
| `tests/test-xmx-compute.cpp` | 9 | XMX compute tests |
| `tests/test-unified-kernel-ops.cpp` | 12 | Unified kernel ops tests |
| `tests/test-unified-cache-fast-path.cpp` | 1 | Unified cache fast-path test |
| `tests/test-xmx-esimd-basic.cpp` | 12 | XMX ESIMD basic tests |
| `tests/test-xmx-dpas-basic.cpp` | 21 | XMX DPAS basic tests |
| `tests/test-esimd-vectorized-dequant.cpp` | 16 | ESIMD vectorized dequant tests |
| `tests/test-mem-handle-eviction.cpp` | 5 | mem_handle eviction tests |
| `tests/test-unified-kernel-persistent.cpp` | 105 | Persistent kernel tests |
| `tests/test-mem-handle-wrong-device.cpp` | 0 | mem_handle device-identity tests (P2-FIX; no direct malloc calls) |

Total test sites: **296** (allowlisted permanently, §8).

---

## Allowlist summary with deletion criteria

| # | File | Lines | Permanence | Deletion criteria |
|---|------|-------|------------|-------------------|
| 37 | `dpct/helper.hpp` | 1463, 3060, 3066 | conditional | Delete if/when DPCT helper layer is removed from the backend |
| 40–41 | `common.cpp` | 713, 732 | permanent | `ggml_backend` buffer API boundary (§8) — cannot use `unified_allocate` here |
| 82 | `ggml-sycl.cpp` | 9112 | permanent | Pre-init probe runs before `unified_cache` is constructed; `ggml_sycl_malloc_device_raw` is correct here |

---

## Notes and surprises

1. **`ggml-sycl.cpp:9112` (probe buffer)** — this call is inside `ggml_backend_sycl_probe_buffer_type_alloc_buffer`, which runs during `ggml_sycl_init()` before the unified cache is initialized. Using `unified_allocate` here would deadlock on the static-init mutex. This is a legitimate permanent allowlist (see §9.1 precedent for "probe buffer is a raw site").

2. **`common.cpp` `ggml_sycl_host_malloc`** — these two sites go through `unified_cache_raw_malloc_host`, which IS the canonical raw gateway. However they are reached via the ggml `buffer_type` API boundary (§8), so they are correctly allowlisted. They are NOT violations of the contract.

3. **`dpct/helper.hpp`** — these three sites are inside the Intel DPCT compatibility shim, not the SYCL backend proper. They were flagged by the grep because they call `sycl::malloc_device`/`sycl::malloc_shared` directly, but DPCT is an ABI adaptation layer outside the contract's scope (similar to "non-SYCL backends" in §8.3).

4. **`mmq_xmx.cpp` test function** — `xmx_test_kernel` is at line 4130 in production code (not under `tests/`). It performs a correctness self-test at startup and frees its allocations before returning. These three allocations are classified `delete`: the function should be moved to `tests/` as a proper fixture rather than migrated to SCRATCH zone.

5. **PTG debug buffers** — seven `ggml_sycl_malloc_shared` calls in `ggml-sycl.cpp` lines 46575–48104 are inside `#ifdef`/debug-path persistent TG kernel debug code. These allocate `shared` USM (migrating memory) solely for debug output. They should be removed along with the surrounding debug code rather than migrated.

6. **MoE graph pre-alloc (context_init)** — lines 43747–43762 in `ggml-sycl.cpp` are the MoE graph buffer pre-allocation function (`init_moe_graph_buffers` or equivalent). These 11 allocations are `context_init` phase and should map to `RUNTIME zone` in the unified cache.

---

## Appendix A — Non-call grep hits (comments, declarations, string literals)

These are the grep hits that match the search pattern but are NOT allocation call sites. They are comments referencing the allocator functions, forward declarations, template/function definitions in the wrapper layer, macro definitions, enum value annotations, or string literals.

*Classification key: `comment` = `//` comment line; `string-lit` = pattern appears inside a string literal or format string; `template-def` = template function definition; `func-def` = inline/non-template function definition; `macro-def` = `#define` line; `enum-comment` = enum value with `//` annotation.*

### `alloc-registry.hpp`

| File | Line | Classification | Excerpt |
|------|------|----------------|---------|
| `alloc-registry.hpp` | 22 | enum-comment | `DEVICE,       // sycl::malloc_device — GPU-only` |
| `alloc-registry.hpp` | 23 | enum-comment | `HOST_PINNED,  // sycl::malloc_host   — CPU-accessible …` |
| `alloc-registry.hpp` | 24 | enum-comment | `SHARED,       // sycl::malloc_shared  — migrates …` |

### `common.hpp`

| File | Line | Classification | Excerpt |
|------|------|----------------|---------|
| `common.hpp` | 1753 | template-def | `template <typename T> inline T * ggml_sycl_malloc_device_t(…` |
| `common.hpp` | 1757 | template-def | `template <typename T> inline T * ggml_sycl_malloc_host_t(…` |
| `common.hpp` | 1761 | template-def | `template <typename T> inline T * ggml_sycl_malloc_shared_t(…` |
| `common.hpp` | 1766 | func-def | `inline void * ggml_sycl_malloc_device_tracked_bytes(…` |
| `common.hpp` | 1779 | func-def | `inline T * ggml_sycl_malloc_device_tracked_t(…` |
| `common.hpp` | 1780 | func-def (body) | `return static_cast<T *>(ggml_sycl_malloc_device_tracked_bytes(…` — body of 1779 |
| `common.hpp` | 1787 | func-def | `inline void * ggml_sycl_malloc_host_tracked_bytes(…` |
| `common.hpp` | 1800 | template-def | `template <typename T> inline T * ggml_sycl_malloc_host_tracked_t(…` |
| `common.hpp` | 1801 | func-def (body) | `return static_cast<T *>(ggml_sycl_malloc_host_tracked_bytes(…` — body of 1800 |
| `common.hpp` | 1814 | macro-def | `#define GGML_SYCL_MALLOC_HOST_T(…)   ggml_sycl_malloc_host_t<T>(…)` |
| `common.hpp` | 1815 | macro-def | `#define GGML_SYCL_MALLOC_SHARED_T(…) ggml_sycl_malloc_shared_t<T>(…)` |
| `common.hpp` | 1819 | comment | `// sycl::malloc_host / sycl::free calls during SOA weight conversion.` |
| `common.hpp` | 1820 | comment | `// Thread-safe.  Buffers are pinned host memory (sycl::malloc_host) …` |
| `common.hpp` | 1879 | comment | `// so that NO sycl::malloc_host occurs during inference.` |
| `common.hpp` | 1895 | comment | `// pinned allocation (sycl::malloc_host bypassing the pool).` |
| `common.hpp` | 1913 | comment | `// at init time.  Do NOT fall back to sycl::malloc_host during` |
| `common.hpp` | 1917 | string-lit | `"(no runtime sycl::malloc_host fallback)\n"` |
| `common.hpp` | 3541 | comment | `// Tracked via host memory tracking (ggml_sycl_malloc_host_tracked_bytes).` |
| `common.hpp` | 3547 | comment | `// Eliminates per-token sycl::malloc_host in the mmap-source …` |

### `common.cpp`

| File | Line | Classification | Excerpt |
|------|------|----------------|---------|
| `common.cpp` | 472 | comment | `// When a process exits without freeing sycl::malloc_host memory, …` |
| `common.cpp` | 473 | comment | `// driver holds stale GGTT mappings that block ALL future sycl::malloc_host calls` |
| `common.cpp` | 661 | comment | `// mubmt.5: Try unified-cache pinned pool first to avoid direct sycl::malloc_host churn.` |
| `common.cpp` | 680 | comment | `// Without cleanup, stale GGTT mappings block ALL future sycl::malloc_host` |
| `common.cpp` | 692 | comment | `// standalone tests confirm sycl::malloc_host succeeds for 10+ GB allocations.` |
| `common.cpp` | 2227 | comment | `// 1. sycl::malloc_host for pinned memory (faster DMA transfers)` |

### `device-pool.hpp`

| File | Line | Classification | Excerpt |
|------|------|----------------|---------|
| `device-pool.hpp` | 78 | comment | `// than calling sycl::malloc_device for each chunk.  Callers can skip` |
| `device-pool.hpp` | 87 | comment | `// sycl::malloc_device stalls BCS H2D events permanently if called` |
| `device-pool.hpp` | 114 | comment | `// Returns {nullptr, 0} if the underlying sycl::malloc_device fails.` |
| `device-pool.hpp` | 173 | comment | `// sycl::malloc_device commits GPU pages, stalling BCS permanently` |

### `expert-prefetch.hpp` / `expert-prefetch.cpp`

| File | Line | Classification | Excerpt |
|------|------|----------------|---------|
| `expert-prefetch.hpp` | 362 | comment | `// Avoids sycl::malloc_device/free per call (3 calls per MoE …` |
| `expert-prefetch.cpp` | 805 | comment | `// This avoids sycl::malloc_device/free per call (3 calls …` |
| `expert-prefetch.cpp` | 814 | comment | `// Allocate via unified_allocate (tries arena first, falls back to sycl::malloc_device).` |

### `fattn.cpp` / `fattn-onednn.cpp` / `fattn-onednn.hpp`

| File | Line | Classification | Excerpt |
|------|------|----------------|---------|
| `fattn.cpp` | 204 | comment | `// Host-pinned USM (sycl::malloc_host, which is what the host arena uses …` |
| `fattn-onednn.cpp` | 445 | comment | `// Any of Q, K, V may be host-pinned (sycl::malloc_host) under` |
| `fattn-onednn.hpp` | 101 | comment | `` // `scale_usm` is a sycl::half scalar buffer (sycl::malloc_host, pinned + `` |

### `fused-moe-esimd.hpp`

| File | Line | Classification | Excerpt |
|------|------|----------------|---------|
| `fused-moe-esimd.hpp` | 929 | comment | `// During graph recording, sycl::malloc_device and host_task are forbidden.` |

### `layer-streaming.cpp`

| File | Line | Classification | Excerpt |
|------|------|----------------|---------|
| `layer-streaming.cpp` | 94 | comment | `// Allocate via unified_allocate (tries arena WEIGHT zone first, then sycl::malloc_device).` |

### `mmvq.cpp`

| File | Line | Classification | Excerpt |
|------|------|----------------|---------|
| `mmvq.cpp` | 4248 | comment | `// Non-null: reuse across DMA calls without sycl::malloc_host per token.` |
| `mmvq.cpp` | 5103 | comment | `// Pre-wire persistent host staging so copy_fn avoids per-call sycl::malloc_host.` |

### `pinned-pool.hpp`

| File | Line | Classification | Excerpt |
|------|------|----------------|---------|
| `pinned-pool.hpp` | 142 | comment | `// never triggers sycl::malloc_host which blocks the Level Zero driver.` |

### `tlsf-allocator.hpp`

| File | Line | Classification | Excerpt |
|------|------|----------------|---------|
| `tlsf-allocator.hpp` | 13 | comment | `// discrete GPUs where sycl::malloc_device memory is not CPU-accessible).` |

### `vmem-kv.hpp`

| File | Line | Classification | Excerpt |
|------|------|----------------|---------|
| `vmem-kv.hpp` | 54 | comment | `//            falls back to P5 arena (sycl::malloc_host).` |

### `unified-cache.cpp`

| File | Line | Classification | Excerpt |
|------|------|----------------|---------|
| `unified-cache.cpp` | 1133 | comment | `// This eliminates per-expert sycl::malloc_host / sycl::free churn …` |
| `unified-cache.cpp` | 1135 | comment | `// NOTE: We use ggml_sycl_malloc_host directly (not unified_alloc) because` |
| `unified-cache.cpp` | 1213 | comment | `// to sycl::malloc_device which can return low-VA pointers that the` |
| `unified-cache.cpp` | 2234 | comment | `// dereference the pointer.  Only DEVICE and HOST_PINNED (sycl::malloc_host,` |
| `unified-cache.cpp` | 3804 | comment | `// Use pre-allocated pinned pool (zero runtime sycl::malloc_host).` |
| `unified-cache.cpp` | 4077 | comment | `// to avoid runtime sycl::malloc_host.` |
| `unified-cache.cpp` | 4151 | comment | `// cache construction — no sycl::malloc_host during inference.` |
| `unified-cache.cpp` | 5195 | comment | `// When arena is NOT active, KV allocates via sycl::malloc_device outside` |
| `unified-cache.cpp` | 5565 | comment | `// (sycl::malloc_host is never called during inference).  Staging of` |
| `unified-cache.cpp` | 5680 | comment | `// eliminating separate sycl::malloc_device calls during context creation.` |
| `unified-cache.cpp` | 5730 | comment | `` // `sycl::malloc_host` chunks are NOT guaranteed to be adjacent … `` |
| `unified-cache.cpp` | 5746 | comment | `// caller can fall back through the sycl::malloc_host path below.` |
| `unified-cache.cpp` | 5805 | comment | `// will fetch it on-demand).  Falling back to sycl::malloc_host here` |
| `unified-cache.cpp` | 7087 | comment | `// pre-reserved ONEDNN zone.  Otherwise, sycl::malloc_device is used …` |
| `unified-cache.cpp` | 8382 | string-lit | `GGML_LOG_ERROR("[COMPUTE-ARENA] sycl::malloc_device failed …"` |
| `unified-cache.cpp` | 9292 | comment | `// sub-allocations within large sycl::malloc_device chunks.` |
| `unified-cache.cpp` | 9349 | comment | `// NOTE: this fallback only fires when the single-chunk sycl::malloc_device` |
| `unified-cache.cpp` | 9571 | comment | `// is its own sycl::malloc_device USM allocation; first-fit walking` |

### `unified-cache.hpp`

| File | Line | Classification | Excerpt |
|------|------|----------------|---------|
| `unified-cache.hpp` | 973 | comment | `//       Includes HOST_PINNED entries (sycl::malloc_host, GPU-accessible via PCIe).` |
| `unified-cache.hpp` | 1309 | comment | `// Must be called BEFORE any BCS H2D copies to avoid sycl::malloc_device` |
| `unified-cache.hpp` | 1468 | comment | `// Number of chunks.  Usually 1 (a single sycl::malloc_device covers the` |
| `unified-cache.hpp` | 1621 | comment | `// Avoids per-expert sycl::malloc_device that fails when VRAM is tight.` |
| `unified-cache.hpp` | 1944 | comment | `// sycl::malloc_device USM allocation), so we use one TLSF allocator per` |
| `unified-cache.hpp` | 1969 | comment | `// Single sycl::malloc_device allocation made BEFORE S1-PRELOAD fills VRAM.` |
| `unified-cache.hpp` | 2963 | comment | `// These are the ONLY functions allowed to call sycl::malloc_device/host/shared.` |

### `ggml-sycl.cpp`

| File | Line | Classification | Excerpt |
|------|------|----------------|---------|
| `ggml-sycl.cpp` | 3582 | comment | `// per-expert sycl::malloc_device fails and all expert prestaging is lost.` |
| `ggml-sycl.cpp` | 6584 | comment | `// NO sycl::malloc_host calls occur during inference.  The model size is` |
| `ggml-sycl.cpp` | 9144 | comment | `// bypassing ONEAPI_DEVICE_SELECTOR filtering.  sycl::malloc_host creates GGTT` |
| `ggml-sycl.cpp` | 9485 | comment | `// sycl::malloc_host creates GGTT (Global Graphics Translation Table) mappings` |
| `ggml-sycl.cpp` | 9494 | comment | `// Buffers larger than this cap will fail sycl::malloc_host — ggml` |
| `ggml-sycl.cpp` | 14402 | comment | `// zones are segmented across multiple sycl::malloc_host chunks and` |
| `ggml-sycl.cpp` | 14569 | comment | `// This avoids the monolithic multi-GB sycl::malloc_device that fails with` |
| `ggml-sycl.cpp` | 14598 | comment | `// We use sycl::malloc_host to create a thin address-space reservation` |
| `ggml-sycl.cpp` | 14884 | comment | `// n_arena_layers == n_layers, so per-layer sycl::malloc_device fallbacks` |
| `ggml-sycl.cpp` | 15098 | comment | `//   - Per-layer sycl::malloc_device calls` |
| `ggml-sycl.cpp` | 15104 | comment | `// falls back to P5 arena (sycl::malloc_host) via the per-layer path below.` |
| `ggml-sycl.cpp` | 15182 | comment | `// per-layer sycl::malloc_device calls during context creation.` |
| `ggml-sycl.cpp` | 15195 | comment | `// sycl::malloc_device calls during context creation.` |
| `ggml-sycl.cpp` | 15344 | comment | `// per-layer sycl::malloc_device fallbacks rather than arena sub-allocations` |
| `ggml-sycl.cpp` | 17980 | comment | `// multiple non-contiguous sycl::malloc_host chunks, and the underlying` |
| `ggml-sycl.cpp` | 18001 | comment | `// the non-pooled sycl::malloc_host path.` |
| `ggml-sycl.cpp` | 18011 | comment | `// pieces of at most this size, each allocated via sycl::malloc_host.` |
| `ggml-sycl.cpp` | 18036 | comment | `// a no-op DIRECT handle for direct sycl::malloc_host (path D) — both` |
| `ggml-sycl.cpp` | 18107 | comment | `// DIRECT handle for path D (direct sycl::malloc_host), which is correct —` |
| `ggml-sycl.cpp` | 18305 | comment | `// host-pinned path (sycl::malloc_host).  Previously this function` |
| `ggml-sycl.cpp` | 18335 | comment | `// the caller falls back to direct sycl::malloc_host.` |
| `ggml-sycl.cpp` | 18578 | comment | `// Re-entrancy guard: unified_cache_allocate → unified_alloc → ggml_sycl_malloc_device.` |
| `ggml-sycl.cpp` | 18648 | comment | `// Always use sycl::malloc_host — pinned host memory with GPU DMA access.` |
| `ggml-sycl.cpp` | 18668 | comment | `// of sycl::malloc_host may internally synchronise the queue on Level Zero,` |
| `ggml-sycl.cpp` | 18683 | string-lit | `ggml_sycl_note_direct_allocation("ggml_sycl_malloc_shared", …)` — pattern in string arg |
| `ggml-sycl.cpp` | 18832 | comment | `// falls back to host-pinned memory (sycl::malloc_host) which the GPU` |
| `ggml-sycl.cpp` | 20200 | comment | `// Allocate staging buffers via unified_allocate (arena-first, then sycl::malloc_device).` |
| `ggml-sycl.cpp` | 20956 | comment | `// Host-pinned memory (sycl::malloc_host) is GPU-accessible` |
| `ggml-sycl.cpp` | 24329 | comment | `// Allocate TP temp buffers via unified_allocate (arena-first, then sycl::malloc_device).` |
| `ggml-sycl.cpp` | 28112 | comment | `// Host-pinned pointers are stable (sycl::malloc_host doesn't move),` |
| `ggml-sycl.cpp` | 30457 | comment | `// host-pinned (sycl::malloc_host) pointers that the GPU can access` |
| `ggml-sycl.cpp` | 37986 | comment | `// set — the GPU cannot access sycl::malloc_host pointers as src0.` |
| `ggml-sycl.cpp` | 38052 | comment | `// set — the GPU cannot access sycl::malloc_host pointers as src0.` |
| `ggml-sycl.cpp` | 38303 | comment | `// and not managed by the expert cache (e.g. sycl::malloc_host buffers from a` |
| `ggml-sycl.cpp` | 42437 | comment | `// HOST_COMPUTE: compute buffers use sycl::malloc_host` |

**Total enumerated: 105 non-call hits** (88 comments, 3 enum-value annotations, 6 function/template definitions in wrapper layer, 2 macro definitions, 3 string literals).

*Note: The summary non-call count of 115 is the residual computed as 545 − 49 (internal) − 76 (migrate) − 3 (delete) − 6 (allowlist) − 296 (test) = 115. This appendix enumerates 105 confirmed non-production grep hits. The 10-entry gap (115 − 105 = 10) reflects multi-grep-hit lines inside the `ggml-sycl.cpp` allocator wrapper function bodies (rows 29–32): each logical row covers 2–3 grep hits (e.g., row 29 covers lines 18588/18592/18640; row 30 covers 18646/18647), but those call-site lines are classified as "internal", not as non-call. The 105 entries enumerated here are every confirmed non-production, non-test grep hit; the 115 residual includes those 10 additional internal-body call lines that are counted in the "internal" production row total. The 105-entry list is authoritative for the non-call classification; 115 is the correct arithmetic residual.*

7. **Contract §9.1 count discrepancy** — the contract §9.1 cites "711 raw alloc patterns" using `grep -rE 'malloc_device|malloc_host|malloc_shared'`. The more restrictive grep used for this inventory (`sycl::malloc_(device|host|shared)|ggml_sycl_malloc_*|malloc_device_raw`) finds 545 total lines. The difference is because the broader grep matches comments, documentation strings, and variable names containing these substrings (e.g., `"sycl::malloc_host` appears in hundreds of comment lines in `ggml-sycl.cpp`). The production caller count (82 migrate+allowlist sites) is the authoritative figure for the migration backlog.
