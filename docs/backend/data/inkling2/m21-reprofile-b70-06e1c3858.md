# Inkling prompt profile on the B70 after Task 21 step 1 (lead, 2026-11-26, `06e1c3858`)

Tree: fork/master `06e1c3858` (Task 21 step 1 merged). Model `/models/Inkling-2-Mini-UD-Q8_K_XL.gguf`, `-c 8192`,
the 4,737-token real-text prompt (`prompt-long-raw.txt`), 1 generated token, seed 42, temperature 0,
`ONEAPI_DEVICE_SELECTOR=level_zero:0`, `GGML_SYCL_MOE_MULTI_GPU=0`, no builder alive, each run once, under `GPU.lock`.
Runner: the lead's `run-inkling-prof.sh` (kernel profiler: `GGML_SYCL_KERNEL_PROFILE=1`, CSV; stage profiler:
`GGML_SYCL_E2E_TG_PROFILE=1`). Both runs rc=0, no refusal, no abort, no kernel fault line.

## Wall time

| run | prompt eval | t/s |
|---|---|---|
| kernel profiler | 30,790.67 ms / 4,737 tokens | 153.85 |
| stage profiler | 29,851.54 ms / 4,737 tokens | 158.69 |

Device-timed kernel time: 19,695.75 ms of the 30,791 ms wall, so the B70 sits idle for about 11.1 s (36%) of
the prompt. 383,341 directly dispatched submissions in 10 micro-batches of 512 rows (failed timestamps: 11,187,
all `sycl.memcpy.mem_ops`/`mem_fill`, the same count as at `87524b0d8`).

## Device time by kernel (top entries, ms over the whole prompt)

| kernel | launches | device ms | share of wall | measured rate |
|---|---|---|---|---|
| `fattn.compute.xmx_v2` (attention, prompt) | 480 | 7,695 | 25.0% | ~1 TFLOPS counting the full causal KV (16.0 ms mean, 5.4 ms min at the first micro-batch) |
| `mulmat.bf16_native` (dense BF16 weights) | 7,044 | 4,430 | 14.4% | 3.1 TFLOPS (2304x3072 at 512 rows: 2.31 ms) |
| `mulmat.moe.bf16_direct` (BF16 experts, blk 20-23) | 88 | 3,970 | 12.9% | 0.6 TFLOPS (2304x2304 x 4,096 expert rows: 72 ms) |
| `moe.pp.gemm.execute` (NVFP4 experts, oneDNN) | 10,016 | 1,830 | 5.9% | 183 us mean |
| `moe.pp.stage.scatter` + `.act` + `moe.pp.dequant.nvfp4` | 3 x 96,592 | 655 | 2.1% | 2-4 us each |
| `sycl.binbcast.*` (mul, add, sub, repeat, div, event) | 48,272 | 717 | 2.3% | |
| `mulmat.onednn_gemm.batched` + `.unbatched` | 3,696 | 217 | 0.7% | |
| everything else | | ~180 | 0.6% | |

B70 XMX BF16/F16 peak for reference: 32 Xe cores x ~2,044 flop/clk x 2.8 GHz (`rp0_freq`) = about 183 TFLOPS. The
three largest kernels run at 0.3% to 1.7% of it.

Shapes. Dense BF16 at 512 rows: Q and gate 2304->3072 (864 launches, 1,995 ms), output 3072->2304 (432, 988 ms),
K and V 2304->1024 (864, 685 ms), loop gate 4608->64, relative 2304->384, the rest small. BF16 experts: gate_up
2304->2304 and down 1152->2304 over 4,096 (token, expert) rows per micro-batch, one launch per node.

Attention geometry (GGUF): 24 query heads, 8 KV heads, head size 128, `sliding_window` 1024 with a sliding-window
pattern, a relative-position bias (`attn_rel_bias`, extent 1024), attention sinks, a per-head query clip. 48 attention
nodes per micro-batch (24 blocks, the loop of blocks 6-17 run three times). The window pattern is five sliding
layers to one full layer (blocks 5, 11, 17 and 23 are full). Per call: min 5.4 ms, p50 16.4, p95 33.1, max 50.0. The
median sits on a plateau while the prompt grows, and the max matches a full layer at 4,737 keys: 512 queries x 4,737
keys x 24 heads x 128 x 4 = 2.98e10 flop in 50 ms is 0.6 TFLOPS, and a sliding layer at 512 x 1,536 keys in 16 ms
is also 0.6 TFLOPS. So the window looks honoured and the kernel is slow per (query, key) pair. Not verified per layer: the profiler
aggregates all 480 calls in one row.

**Explorer's read of the attention path (2026-11-26, read-only, on `185b44aa8e`; the S·V branch checked by the
lead).** The window is honoured only through the cache size: sliding layers see 1,536 keys, full layers grow in
steps of 256, and the kernel computes every key it is given (no masked-tile skip). A model of time as query blocks x
heads x keys/32 gives 455 ns per work-group batch on this prompt and 469 ns on the 691-token one, a flat ~0.58 TFLOPS.
(Correction: the 691-token profile's 3.2 ms call is the 179-query x 768-key micro-batch; a 512 x 512 call is about
5.6 ms, the same rate.) Causes, in the order they cost:
- `src/llama-graph.cpp` pins `GGML_PREC_F32` on every attention node, so `xmx_v2` takes its float-accumulator
  branch, which does S·V as a scalar loop over local memory (`fattn-xmx-f16-v2.hpp:850-870`); only the half branch
  (`:871-906`) uses XMX for it. An in-tree comment (`fattn.cpp:3437-3441`) measured the same move on a GPT-OSS shape
  at 7.8 -> 1.9 ms per layer.
- 16 queries and two sub-groups per work-group, element-wise K/V loads without double buffering, and the three
  query heads that share a KV head each reload it.
- No skip of fully masked key tiles (a third of a sliding layer's 1,536 keys; the causal triangle of a full layer).
- The per-head bias is materialised as an [n_kv, 512, 24] F16 mask by get_rows, two copies, an add and two casts
  (about 250 GB of traffic over the prompt, mostly untimed).
- oneDNN attention declines the node because it carries sinks (`fattn-onednn.cpp:116`); its planner also does not
  check the mask's head stride (owed separately).

## Host time by stage (stage profiler run)

| stage | calls | host ms | per call |
|---|---|---|---|
| moe | 1,032 | 17,129 | 16.6 ms |
| other | 48,475 | 4,209 | 87 us |
| elementwise | 20,753 | 539 | 26 us |
| non-MoE matmul | 10,493 | 435 | 41 us |
| attention | 1,548 | 104 | 67 us |
| kv | 1,032 | 22 | 21 us |

21 profile blocks, sum of wall 28,676 ms, sum of host 22,437 ms. There are 880 NVFP4 MoE nodes in the prompt (44 NVFP4 layer
applications x 2 nodes x 10 micro-batches); each submits about 110 scatter, 110 activation and 110 dequant launches
(one per active expert group) and 11.4 GEMM executions: the per-expert stage loop of the grouped executor.

What the 11 s of device idle is made of is **not established**:
- `moe.pp.dequant.nvfp4` is a marker (0.1 us per launch), not a timed kernel. The F16 dequant of about 110 experts
  per node is real device work that no row of the table holds, so the 19.7 s device total undercounts.
- About 290,000 MoE submissions at the plan's measured 23 us each is about 7 s, not 11 s.
- The stage host total (22.4 s) is well above the submission cost of all 383,000 launches (about 9 s), so the
  `moe` stage timer includes blocking, likely the ids read-back for the counting sort, which waits for the work
  queued ahead of it.
The split between submission cost, the untimed dequant and the ids wait decides between batching the stages
(Task 21 step 2), the weight-only-quantised arm that removes the dequant (step 3) and the host-gap task. It is the
first checkpoint item of whichever of them goes next.

## Ranking and the lead's reading

1. **Attention, 7.7 s (25%).** The single largest kernel, at about 0.6 TFLOPS per needed (query, key) pair. What
   Inkling's relative bias, sinks and clip cost inside `xmx_v2`, and why it is slower per pair than on Mistral or
   GPT-OSS, is not known yet.
2. **BF16 weights, 8.4 s together (27%).** Dense BF16 (4.4 s) on `bf16_native` at 3.1 TFLOPS, and the BF16
   experts (4.0 s) on the direct kernel at 0.6 TFLOPS. This is the plan's Task 22.
3. **About 11 s of device idle around the MoE nodes.** Its parts are not established (see above).
4. NVFP4 expert GEMM 1.8 s, the stage kernels 0.65 s, element-wise 0.7 s: smaller, behind the three above.

No single item reaches the user's target alone: 1,000 t/s on this prompt is 4.7 s, and removing all BF16 device
time still leaves about 21 s (about 220 t/s). Each task's acceptance is set from this profile.

Raw data: `m21-kprof-pp-long-b70-06e1c3858.csv` next to this page (the kernel profiler's CSV). The stage log stays
in the lead's temporary directory (not durable); its totals are above.
