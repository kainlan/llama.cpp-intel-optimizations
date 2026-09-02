# SYCL utilization plan — gemma4 to 80% of hardware peak on the B50 and B70

Status: APPROVED 2026-09-01 (owner: INT8 anchor for prefill; Wave 0 go) (lead session, after llama.cpp-os8k round 2a). Tracker
epic: llama.cpp-jjm7; goal ticket llama.cpp-0oad; roadmap
epic llama.cpp-wm4j (P1–P7). Evidence files (session scratchpad, lifted here where
load-bearing): kprof-os8k.csv, kprof-pp-b70.csv, e2e-pp-b70.log,
research-launch-overhead.md, research-onednn-prefill.md, research-megakernel.md.

## 0. Goal, metric, and where we stand

Owner ruling (2026-09-01): "80% efficiency" means **80% of the card's theoretical
peak** — memory-bandwidth utilization for decode, compute utilization for prefill.
The empirical Mistral-anchored targets used in 0oad c-9u5f are retired.

Hardware peaks (Intel ARK / datasheets, corroborated by the backend's own caps print):

| card | Xe2 cores / XVE | max clock | memory BW | INT8 XMX | FP16 XMX dense |
|---|---|---|---|---|---|
| Arc Pro B70 (`level_zero:0`) | 32 / 256 | 2800 MHz | **608 GB/s** (256-bit GDDR6) | 367 TOPS | ~160–183 TFLOPS |
| Arc Pro B50 (`level_zero:1`) | 16 / 128 | 2600 MHz | **224 GB/s** (128-bit) | 170 TOPS | ~85 TFLOPS |

The B70 bandwidth figure is corroborated in-tree: after the os8k head fix the
output-head kernel streams 713 MB in 1.185 ms = 602 GB/s (99% of 608).

Model (gemma-4-E4B-it Q8_0): **~5.18 GB streamed per decode token** (head 713 MB +
FFN 3510 + attention projections 624 + F32 PLE projections ~330; the 2.99 GB
`per_layer_token_embd` table is GET_ROWS-gathered, not streamed) and **~9.3 GFLOP
per token** of MUL_MAT work.

Scoreboard at 27fcc3ac0 (task/os8k), pp512 / tg128, interleaved pairs:

| axis | now | achieved | utilization | 80% bar |
|---|---:|---:|---:|---:|
| B70 tg | 38.9 t/s | 201 GB/s | **33%** | ≥ 93.9 t/s |
| B50 tg | 25.3 t/s | 131 GB/s | **58%** | ≥ 34.6 t/s |
| B70 pp | 2530 t/s | 23.5 TOPS-eq | **6.4%** of INT8 XMX | see §0.1 |
| B50 pp | 1446 t/s | 13.4 TOPS-eq | **7.9%** of INT8 XMX | see §0.1 |

### 0.1 Prefill metric — RULED 2026-09-01: anchor on INT8 XMX peak

Owner ruling: prefill compute utilization is scored against the **INT8 XMX peak**
(B70 367 TOPS, B50 170 TOPS) — Q8_0 weights × Q8_1 activations is an INT8 dot
product, so that is the card's real capability for this quantization and the kernel
choice must not define the ceiling. On that scale today: **B70 6.4%, B50 7.9%**.
A literal 80% (≈31,600 t/s on the B70) is above any published prefill implementation
on Xe2 — the best published kernels (Intel Labs Xe-Forge, arXiv:2605.26118, measured on
a B70) reach 40–50% of peak and naive ones 3–10% — so the plan reports raw
%-of-INT8-peak and tracks milestones toward the attainable ceiling (~40% ≈ 15,800 t/s
B70 / ~7,300 t/s B50). Track P's items are what move that number; P5 (an INT8 XMX
GEMM path) is the one that makes the INT8 ceiling reachable at all.

### 0.2 The decode arithmetic that sizes the whole plan

Streaming floor on the B70 = 5.18 GB / 608 GB/s = **8.5 ms/token**. 80% utilization
means ≤ 10.7 ms wall, so **≤ ~2 ms/token for every gap combined**. Today: ~630
profiled launches + ~300 unprofiled RMS-norm launches ≈ **900+ kernels/token**, and
the measured non-kernel time is ~15 ms ≈ 16 µs per kernel. Even at a 3–5 µs
per-boundary floor, 900 boundaries do not fit the budget. **Decode needs roughly a
3× reduction in kernels per token AND the whole token inside one replayed command
list.** Device-busy time is already ~81% of bandwidth (5.18 GB in ~10.5 ms of kernel
time) — the kernels are fine; the seams between them are not.

## 1. Diagnosis (measured 2026-09-01, B70, post-os8k)

### 1.1 Decode — a kernel-count and graph-coverage problem
- Kernel profile: 628 profiled submits/token; the largest non-matmul rows are
  `sycl.binbcast.mul` (85), `sycl.rope` (66), `sycl.set_rows.generic` (48),
  `sycl.memcpy.mem_ops` (44, 7.6 MB/token, 7.7 µs each), `get_rows` release markers
  (85). `norm.cpp` RMS_NORM is unwrapped (~300 launches, unmeasured).
- `llama_perf_context_print: graphs reused = 64/65` — but that does **not** mean the
  token is fully replayed. Two code facts (explore-overhead-paths report):
  - `ggml_sycl_graph_safe_memcpy()` (common.hpp:358-395) emits the capturable
    `sycl.memcpy.graph_safe` copy only while recording; the `sycl.memcpy.mem_ops`
    label is the *non-recording* path (mem-ops.cpp:540). 44 such copies per token
    means 44 copies run outside the captured graph on every replay call.
  - `ggml_sycl_stage_get_rows_indices()` (getrows.cpp:489-600) uses the persistent
    per-graph slot (`graph_input_stage_lookup`) only when
    `ggml_sycl_graph_recording_active()`; otherwise it does a fresh pinned
    `unified_allocate` + host memcpy per call, released by the `indices_release`
    marker (getrows.cpp:3083-3087). 85 markers/token ⇒ GET_ROWS runs outside the
    graph every token, with an allocation on the hot path.
  - Decode FLASH_ATTN_EXT is kept out of the graph by default
    (`GGML_SYCL_FLASH_ATTN_GRAPH_ALLOW`, ggml-sycl.cpp ~100084-100097); the
    `ggml_sycl_graph_has_host_inputs` gate (~100064-100077) can disable decode replay
    when the plan has host intermediates. Exactly what the replayed graph covers for
    gemma4 is unmeasured — spike S1.
- Fusion machinery exists but is walled off: RMSNorm→MUL_MAT fusion refuses
  `nrows < 8` (ggml-sycl.cpp:80080-80082) and its multi-consumer variant
  `can_fuse_all_projections()` is dead (returns false); fused gate/up
  (`try_fused_ffn_gate_up_swiglu`, fused-ffn.hpp:313-378) supports M=1 but is Q4_0-only,
  default-OFF (`GGML_SYCL_FFN_FUSION`), and explicitly disabled whenever graphs are on
  (ggml-sycl.cpp:87583-87589).
- gemma4's per-layer-embedding chain issues 6 kernels per layer at decode
  (src/models/gemma4.cpp:343-367: MUL_MAT F32 M=1 → GELU → MUL → MUL_MAT → RMS_NORM →
  ADD) = 252 launches/token that a single fused kernel per layer replaces.

### 1.2 Prefill — overhead-bound, neither compute- nor bandwidth-bound
pp512 on the B70 = 200 ms per ubatch (2560 t/s):
- Bandwidth: 5.18 GB / 200 ms = 26 GB/s (4%). Compute: 23.5 TFLOPS (13% of XMX).
- Attention: 35 oneDNN SDPA graph executes at 1.19 ms device each = 42 ms, but the
  `FLASH_ATTN_EXT` stage spends **95–98 ms of host time** per ubatch:
  `fattn-onednn.cpp:560-561` `wait_and_throw()`s the K/V materialization events and
  `:491` the Q repack before every execute, and the materialization buffers are
  allocated fresh with `unified_allocate()` on every call (:528-536). The
  `dnnl::graph::sycl_interop::execute` call at :1255 is passed no dependency events,
  on the belief (comment :1241-1249) that it returns void — **the bundled oneDNN 3.11
  header (`dnnl_graph_sycl.hpp:93`) declares `inline sycl::event execute(...)` with a
  `deps` vector**, so the waits are removable with the API as shipped.
- Dispatch: 123 ms of the 200 ms is inside op dispatch (800 ops ≈ 150 µs/op; the
  last `GGML_SYCL_DISPATCH_TIMING` census put enqueue at 114 µs/op vs the 5–10 µs an
  L0 submit costs), plus ~77 ms outside dispatch (per-ubatch graph build, scheduler
  alloc, planning). The per-op path runs, in order: pending-work flushes,
  `should_dispatch_to_cpu` (memoized but still a lookup), peer/split setup + event
  chaining, four `ggml_sycl_try_route_*` classifiers, then the op switch
  (ggml-sycl.cpp:78665-79352).
- GEMMs: only 18 of 237 Q8_0 M=512 GEMMs per ubatch land on the profiled
  `mmq_generic` path (~1.8 ms each ≈ 15 TFLOPS, non-XMX); the other ~219 go through
  the unprofiled oneDNN WOQ matmul. ~120 ms/ubatch is therefore dark to the profiler.
- Ubatch: pp2048 at ub512 → ub2048 = 2.6× on identical FLOPs (0oad c-o321) — the
  dominant cost shrinks with M, i.e. per-op/per-ubatch overhead, not arithmetic.
- Prior art: PP graph replay was attempted and hung (llama.cpp-6eepe).

### 1.3 Instrument gaps (must close before attribution can be trusted)
- Kernel profiler: oneDNN WOQ Q8_0 GEMM and `norm.cpp` RMS_NORM unwrapped.
- e2e stage profiler: `device_us` is 0 by construction — every call site passes 0.0
  and `e2e_tg_scope` is host-clock only (e2e-profile.hpp:86-119). Host-only today.

## 2. Feasibility evidence (web research, 2026-09-01; files in scratchpad)

| claim the plan relies on | evidence |
|---|---|
| Submission path is already the low-latency one | oneAPI 2025.3+ L0 **v2 adapter is default on Xe2** and supports only immediate command lists; batching env vars are moot. (Intel L0 immediate-command-list guide; DPC++ 2025 release notes) |
| Whole-graph replay materially cuts host cost | Intel's many-small-kernels SYCL-graph benchmark: 322 µs → 117 µs per run (2.7×). Graph nodes may be kernels, USM memcpy/memset/fill, async alloc; host-task nodes break whole-graph submission. Dynamic params only on kernel nodes; whole-graph update needs identical topology. (sycl_ext_oneapi_graph spec; Intel article) |
| Async oneDNN SDPA is an API-level fix | `dnnl::graph::sycl_interop::execute` returns `sycl::event`, takes `deps` (oneDNN header; verified in the bundled 3.11). GQA is a native oneDNN pattern (4D or 5D form). Strided KV views: no public statement — spike S3. |
| Persistent kernels are proven on this hardware class | vLLM Intel Arc Pro B blog (2025-11-11): single persistent MoE GEMM kernel with **atomic work-stealing** reaches >80% HW efficiency on a B60; naive per-launch scheme wasted ~15%. Only device-scope atomics required. |
| …but grid barriers are not | No cooperative-launch / grid-sync guarantee on Level Zero; SYCL forward progress for inter-workgroup sync is occupancy-bounded (workgroups ≤ resident capacity, sized manually). Hopper-cluster-style fusion is not portable. Hazy-style counter dependencies are plausible (atomics only) but need spike S5. |
| Realistic prefill ceiling | Xe-Forge (arXiv:2605.26118, B70): optimized kernels 60–80 TFLOPS ≈ 40–50% of peak; naive 5–15 TFLOPS. Community llama.cpp-SYCL 7–8B pp512: B580 ~2,000, B60 ~1,870, B70 up to ~3,000 t/s. |
| Per-kernel gap floor on Xe2 | **No public figure.** Intel `compute-benchmarks` has `IoqKernelSwitchLatency` / `ImmediateCommandListSubmission` — spike S2 measures it here. (NVIDIA reference: ~2.5 µs + 1 ns/node with CUDA graphs.) |
| Fusion gains on Intel | IPEX-LLM deep fusion (RMSNorm/RoPE/SDPA, GEMM+elementwise) up to 7× vs HF aggregate; vllm-xpu-kernels ships fused RMSNorm+add, SiLU-mul, RoPE. No isolated per-fusion numbers. |
| gemma PLE correctness | Upstream issue ggml-org/llama.cpp#22243 claims Gemma-4 PLE residual injection is incomplete. Our fork's gemma4.cpp:366 does `ggml_add(pe_in, cur)`; our L2 scan compares GPU against our own CPU path so a shared omission would pass — spike S7. |

## 3. Spikes (Wave 0 — measure before building; each ≤ 1 day, lead runs GPU parts)

| id | spike | method | output / acceptance |
|---|---|---|---|
| S1 (llama.cpp-w2e8) | **Decode graph-coverage census** | On a replay token: `GGML_SYCL_DISPATCH_TIMING=2` per-op trace + kernel profiler; count ops that execute per-op vs inside `ext_oneapi_graph`; check `GGML_SYCL_FLASH_ATTN_GRAPH_ALLOW` and `graph_has_host_inputs` outcomes; list every excluded op with the reason branch it hit. | Table: op type × count × reason. Decides D1's scope. |
| S2 (llama.cpp-fa0f) | **Xe2 kernel-gap floor** | Build/run Intel compute-benchmarks `IoqKernelSwitchLatency`, `ImmediateCommandListSubmission`, `EventCreation` on B70 and B50; also a 900-null-kernel in-order chain with and without a SYCL graph. | µs per boundary (eager vs graph) → the kernels/token budget for §0.2. |
| S3 (llama.cpp-hhbb) | **Async oneDNN SDPA prototype** | Pass K/V/Q materialization events as `deps` to `execute`, drop the three `wait_and_throw()`; confirm strided-KV acceptance vs the dense-materialize requirement; measure host stage time and correctness (digit gate + L2). | Host `FLASH_ATTN_EXT` stage ≪ 42 ms device time; digit/L2 green. Feeds P1. |
| S4 (llama.cpp-jmc5) | **Prefill per-op enqueue attribution** | `perf record -g` on a pp512 run + `GGML_SYCL_DISPATCH_TIMING=2`; attribute the ~150 µs/op to flush / cpu_check / resolve+event chaining / route classifiers / kernel switch / UR submit. | Ranked host-symbol table; names the top 3 slices for P2. |
| S5 (llama.cpp-3x6k) | **Persistent-kernel forward-progress test on Xe2** | Microkernel: N workgroups spinning on device-scope atomic counters (Hazy-style dependency counters) with N ≤ measured resident capacity, plus an atomic work-stealing loop; verify no hang/livelock on B70 and B50, measure resident WG capacity and per-dependency latency. | Go/no-go + safe WG count for D7. |
| S6 (llama.cpp-qmen) | **Profiler completeness** | Wrap oneDNN WOQ Q8_0 GEMM (`gemm.hpp` / the MUL_MAT oneDNN path) and `norm.cpp` RMS_NORM in the kernel profiler; either plumb `sycl::event` profiling into `e2e_tg_profile_record` or rename the stage profiler host-only. | pp512 profiled device time accounts for ≥ 90% of wall minus measured host gap. |
| S7 (llama.cpp-ddno) | **gemma4 PLE correctness vs upstream #22243** | Read gemma4.cpp against the reference architecture; compare a short logits vector against HF transformers (CPU) for the gate prompt. | Confirmed correct, or a bug filed before any PLE fusion. |

## 4. Workstreams

Acceptance for every item: digit gates both cards, L2 scan vs CPU reference (named
join, diverging=0), Mistral Q4_0/Q8_0 + GPT-OSS regression belt, interleaved-pair
perf, unified-cache/mem_handle rules (no raw allocations, no pointer-keyed caches,
no host waits), fresh spec + quality review.

### Track D — decode: fewer kernels, whole token replayed
| id | item | depends on | expected effect (B70) |
|---|---|---|---|
| D1 (llama.cpp-dyi3) | **Whole-token graph coverage**: make GET_ROWS use the persistent `graph_input_stage` slot outside recording too (no per-call `unified_allocate`), route the 44 out-of-graph memcpys through a capturable path or eliminate them (producer writes in place), allow decode FLASH_ATTN_EXT in the graph, remove host-input gating for gemma4's shape. | S1 | removes ~130 out-of-graph submissions and 85 hot-path allocations/token; first bite at the 15 ms gap |
| D2 (llama.cpp-y7q2) | **RMSNorm→MUL_MAT fusion at M=1**: lift the `nrows < 8` wall, revive `can_fuse_all_projections()` for the shared Q/K/V norm, persistent scratch instead of `scoped_unified_queue_temp`, graph-capturable. | S1 | −~170 launches/token (85 norms + 85 muls) |
| D3 (llama.cpp-51j2) | **Fused gate+up(+GEGLU) for Q8_0 at M=1**, graph-capturable (extend `try_fused_ffn_gate_up_swiglu` beyond Q4_0; remove the "graphs must be off" restriction). | D1 | 84 → 42 launches; activation stays in registers |
| D4 (llama.cpp-rnbb) | **PLE per-layer chain fusion**: one kernel per layer for MUL_MAT(F32,M=1)→GELU→MUL→MUL_MAT→RMS_NORM→ADD. | S7, D2 | 252 → 42 launches/token |
| D5 (llama.cpp-c4kf) | **RoPE into the QKV epilogue + batched KV `set_rows`** (66 + 48 launches). | D1 | −~100 launches/token |
| D6 (llama.cpp-gvt9) | **Attention decode in-graph + ESIMD partition tuning** (35 `fattn.decode.esimd_partitioned` calls at 9 µs). | D1 | small; mostly graph coverage |
| D7 (llama.cpp-q9q4) | **Persistent per-layer decode kernel** with atomic work-stealing (P4's original item), replacing the per-layer matmul chain; the endgame for ≤ ~2 ms of gaps. | S2, S5, D2–D5 | remaining gap → target ≥ 93.9 t/s |

Milestones: D1–D3 → B70 tg ≥ 55 t/s (≈47%); D4–D6 → ≥ 70 t/s (≈60%); D7 → ≥ 94 t/s
(80%). B50 tracks proportionally (bar 34.6 t/s, reachable around D3–D4).

### Track P — prefill: no host waits, cheap dispatch, replayed ubatches
| id | item | depends on | expected effect (B70, pp512) |
|---|---|---|---|
| P1 (llama.cpp-gwno) | **Async attention path**: event-chain K/V/Q materialization into `execute(..., deps)`, no `wait_and_throw()`; persistent materialization buffers (dense-f16 K/V mirror updated at `set_rows` time, Q slot reused) instead of per-call `unified_allocate`. Generalises llama.cpp-1r58. | S3 | −50–90 ms host stall per ubatch; overlap GEMM with attention |
| P2 (llama.cpp-6u2o) | **Per-op dispatch cost**: memoize per-graph-node route decisions (`try_route_*`, `should_dispatch_to_cpu`, key lookups) across ubatches of the same shape family; event-chain only across queue boundaries. | S4 | 123 ms → tens of ms per ubatch |
| P3 (llama.cpp-lez0) | **Prefill graph replay keyed on ubatch shape** (P4 item 4): record once per shape family, replay for ubatches 2..N; resolve the 6eepe hang first. | P1, P2, D1 | removes the ~77 ms/ubatch rebuild + per-op enqueue on replayed ubatches |
| P4 (llama.cpp-nphx) | **Auto-ubatch default** (llama.cpp-nphx, P7) — pick n_ubatch per card/model from fit. | — | 2.6× at pp2048 today, compounding with P1–P3 |
| P5 (llama.cpp-8s4h) | **XMX-tiled Q8_0 GEMM for M ≥ 512** (llama.cpp-vqtf, P3) replacing `mmq_generic`/WOQ, plus an attention prefill path without per-call dense K/V materialization. | S6 | toward the 40% MFU ceiling |

Milestones: P1+P2 → ≥ 4,000 t/s; P3+P4 → ≥ 6,000 t/s at pp2048; P5 → ~7,900 t/s
(80% of the attainable ceiling).

### Track I — instrumentation (Wave 0/1, unblocks attribution)
I1 = S6 productised; I2 = decode graph-coverage counter exposed permanently
(`[SYCL-GRAPH] ops_in_graph=… ops_per_op=…` at WARN once per context so it survives
default verbosity).

## 5. Sequencing

- **Wave 0 (spikes, parallel, lead runs GPU parts):** S1, S2, S3, S4, S5, S6, S7.
- **Wave 1 (cheapest decisive):** P1, D1, I1/I2.
- **Wave 2:** D2, D3, P2, P3 (P3 only after 6eepe is understood).
- **Wave 3:** D4, D5, D6, P4, P5.
- **Wave 4:** D7.

Implementers write code in the recycled worktree (`/Apps/llama.cpp-tgo37`, warm
ccache); GPU verification is lead-serialised. Any change to `common.hpp` is a ~45 min
rebuild of every SYCL TU — batch header edits.

## 6. Risks and open questions
- Graph "replay futility" and host-input gates may keep gemma4 partially out of the
  graph for structural reasons (fragmented `#` splits, host intermediates) — S1 tells.
- PP graph replay hung before (llama.cpp-6eepe); root-cause before P3.
- Forward progress for spinning workgroups on Level Zero is not guaranteed; D7 is
  gated on S5 and must size workgroups ≤ resident capacity.
- oneDNN SDPA with strided KV views is undocumented; P1 may still need a persistent
  dense mirror (planned) rather than zero-copy views.
- Fusion changes accumulation order — the L2 scan (named join, diverging=0) is the
  oracle; master-vs-master noise is ~0.2% on ~9 nodes.
- PP metric is anchored on INT8 XMX peak (ruled); the attainable-ceiling milestones
  (§0.1) are working targets, not the goal line.
