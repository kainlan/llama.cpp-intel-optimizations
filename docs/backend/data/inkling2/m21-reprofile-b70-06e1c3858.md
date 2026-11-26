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
nodes per micro-batch (24 blocks, the loop of blocks 6-17 run three times). The per-call time grows with the KV
length (5.4 ms at 512 keys, 16 ms median): it is not limited to the 1,024-token window.

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
(one per active expert group) and 11.4 GEMM executions: the per-expert stage loop of the grouped executor. This is the host side of the 11 s the device sits idle.

## Ranking and the lead's reading

1. **Attention, 7.7 s (25%).** The single largest kernel. It grows with the prompt and runs at about 1 TFLOPS.
   Whether the sliding-window layers compute their full KV, and what Inkling's relative bias, sinks and clip
   cost inside `xmx_v2`, is not known yet.
2. **BF16 weights, 8.4 s together (27%).** Dense BF16 (4.4 s) on `bf16_native` at 3.1 TFLOPS, and the BF16
   experts (4.0 s) on the direct kernel at 0.6 TFLOPS. This is the plan's Task 22.
3. **MoE host time, about 11 s of device idle.** The per-expert stage loop of the NVFP4 executor. This is Task 21
   step 2 (batch the stages and the GEMM over experts).
4. NVFP4 expert GEMM 1.8 s, the stage kernels 0.65 s, element-wise 0.7 s: smaller, behind the three above.

Raw data: `inkling/m21-kprof-pp-long-b70-06e1c3858/kprof.csv` and `inkling/m21-e2e-pp-long-b70-06e1c3858/inkling.err`
in the lead's temporary directory (not durable). This page is the durable record.
