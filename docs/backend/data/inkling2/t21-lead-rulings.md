# Task 21: the lead's rulings and the prompt-speed budget (2026-11-24)

Task 21 of `docs/plans/2026-11-21-inkling2-mini-sycl-enable-profile-optimize.md`. Base: master `dd59d8710`.
The implementer's checkpoint (read-only, on that base) changed the task's shape; this file records what the lead
ruled and why. Figures are from one run each on the Arc Pro B70, driver 26.44.

## Where a 512-token micro-batch goes today

From the kernel profile of a 691-token prompt at `641c7e065` (wall 6987 ms, device kernels 5094 ms, 22,825 direct
submissions, none graph-recorded), scaled to 512 tokens:

| part | seconds per 512 tokens | share |
|---|---|---|
| NVFP4 expert kernel (`mulmat.mmvq.id_aos_nvfp4`) | 2.36 | 46% |
| exposed host time (device idle) | 1.40 | 27% |
| dense BF16 weights (`mulmat.bf16_native`) | 0.53 | 10% |
| BF16 expert kernel (`mulmat.moe.bf16_direct`) | 0.43 | 8% |
| attention (`fattn.compute.xmx_v2`) | 0.33 | 6% |
| everything else on the device | 0.12 | 2% |
| total | 5.18 (99 t/s) | |

1,000 t/s is 0.51 s for all of it. No single part reaches that: every row above has to fall by about ten times.
The NVFP4 experts alone cap the gain at 1.74x on the long prompt. Order of attack, by size: the NVFP4 experts
(Task 21), the exposed host time (about 11,000 submissions per micro-batch against about 1,000 for Mistral; being
attributed by a read-only explorer), then the BF16 experts and dense BF16 weights (Task 22), then attention.

## Rulings

- **R1, acceptance.** The 2x bar on Task 21 alone is withdrawn: it is above the Amdahl ceiling of the part the
  task changes. Step 1 passes at long-prompt pp >= 1.25x of 92 t/s on the B70 with the NVFP4 expert time per
  micro-batch reported; step 2 at expert time <= 0.6 s per micro-batch and a submission count per node that does
  not grow with the number of active experts; step 3 at <= 0.3 s. Every step: text byte-equal run to run, decode
  not slower, peak host backing not higher, the B50-alone and both-cards setups not slower.
- **R2, conversion into scratch.** A per-dispatch conversion of a device-resident prompt operand into planned,
  cache-owned scratch is allowed for prompt processing (precedent: `llama.cpp-iikr` and `llama.cpp-348c`,
  seventeen quant types). For NVFP4 the F16 arm is scaffolding (role, ring, claim, batching over experts); the
  target is the weight-only-quantised arm, which re-orders the stored 4-bit codes and scales without expanding
  them. A stored plane layout or a native grouped kernel is decided only if the repack is the wall after step 3.
  The loaded AoS layout stays the only layout of the weight, and decode is not touched.
- **R3, the fused gate_up tensor gets its own role, `MOE_TENSOR_GATE_UP`.** Inkling's `ffn_gate_up_exps` matched
  none of the three role names, so the prompt claim, the ring consumer check and the loader's ring sizing all
  skipped two thirds of the NVFP4 work. It is not an alias of GATE: every role-indexed array grows, every switch
  names it, and the pair-GLU logic does not apply to it.
- **R4, NVFP4 is not a staged type.** Listing NVFP4 in the F16 table must not make it a staged type; it is
  excluded by `moe_type_has_immutable_host_recipe`. The B50 host route, the CPU-island refusal and the host
  recipe pools do not change. Staging NVFP4 host experts is a separate decision.
- **R5, the ring is planned only where it can serve.** The ring for these consumers (about 586 MB at a 512-row
  micro-batch) is reserved only on a device with at least one claimable, fully device-resident node of the type.
  The requested context's KV and the compute buffer keep their place.
- **R6.** No new host wait, no allocation outside the unified cache, expert leases and the ring claim held until
  the executor's done event.

## What is not known yet

- Whether the Xe2 oneDNN matmul accepts `f4_e2m1` weights with `f8_e4m3` scales in groups of 16 (headers of oneDNN
  3.13.2 declare it; a device probe written under this task decides it).
- Why the 3-D weight-only-quantised arm gives wrong numbers today; the checkpoint's reading is that it passes a
  three-entry scale group where the API takes zero or two.
- How many Inkling prompt nodes are fully device-resident on the B50 alone.
