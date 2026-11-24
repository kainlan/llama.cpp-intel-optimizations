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

- **R5 amended (the ring is sized before placement is known).** The loader sizes the ring and the backend
  publishes it before the planner packs experts, so "a claimable node on this device" cannot be read at sizing
  time. Ruling: plan with the ring; if no weight of a ring-served type has every expert planned on that device,
  publish the ring as 0 and plan once more (one retry, the ring-0 plan is final). It applies to the ring as such,
  MXFP4 included, and prints its decision once at WARN.
- **R7, a decline does not abort a prompt.** The direct NVFP4 route steps aside for a prompt node only when the
  claim holds and the device has a planned ring slot. A decline the executor reports before its first device
  submission re-enters the direct route with one WARN per reason; a failure after the first submission throws
  with a named reason.

## The oneDNN probe (lead's run, once, B50, `f1fe27068`, no model loaded)

`test-sycl-onednn-woq-nvfp4`: rc=0, seven arms, no kernel fault line. All through `impl=jit:gemm:any`.

| arm | result | worst relative error | us per execute |
|---|---|---|---|
| NVFP4 2-D, f16 [32,2304] x f4_e2m1 [2304,2304], f8_e4m3 scales, groups {16,1} | PASS | 6.99e-07 | 80.1 |
| NVFP4 3-D, 4 experts of [1152,2304], mask 7, groups {16,1} | PASS | 6.57e-07 | 158.9 |
| MXFP4 3-D control, e8m0 scales, groups {32,1} | PASS | 6.61e-07 | 157.3 |
| MXFP4 3-D with the group dims the backend passes today ({1,32,1}) | refused: "could not set scales primitive attribute" | | |
| NVFP4 2-D and 3-D with scales widened to f16 | PASS, same figures | | |

Scale bytes 0x00, 0x7F and 0xFF read exactly 0 and no output is non-finite. So the weight-only-quantised arm
takes NVFP4 as stored (nibbles re-ordered, scales sanitised), in one execute per group of experts. After step 1
the old steps 2 and 3 merge: NVFP4 goes to that arm in its 3-D form and leaves the F16 table again. The
backend's opt-in 3-D MXFP4 arm (`GGML_SYCL_MOE_PP_WOQ_3D`) is wrong because it passes three group dims where
the API takes two: a GPT-OSS follow-up. From the call times, 44 gate_up and 44 down nodes of 128 experts would
cost about 0.7 s per micro-batch on the B50 (an estimate from single calls, not a measured prompt).

## The exposed host time (read by an explorer on `dd59d8710`, then one experiment)

- For every prompt MoE layer application the host copies the ids from the device and waits
  (`ggml_sycl_refresh_moe_ids_cache`), then builds a resolved batch of n_tokens x 8 leased operands before it
  submits anything. The device is idle meanwhile: 96 times in the 691-token prompt.
- No prompt node of this model is ever graph-recorded: `check_graph_compatibility` admits a MUL_MAT_ID only for
  quantised Q4_0, Q8_0 or MXFP4 experts at up to 32 rows, and a global plan runs every prompt direct.
- Every binbcast op submits a second, empty job for its completion event (6,432 of the 22,825 submissions).
  `GGML_SYCL_BINBCAST_EVENT_MODE=reuse` returns the kernel's own event. Measured on the long prompt, B70,
  interleaved, each run once (barrier, reuse, reuse, barrier): pp 91.65, 91.61, 92.18, 91.89 t/s; tg 25.33,
  26.84, 26.78, 26.01 t/s; text byte-equal in all four. The extra submissions are not the prompt's host time.
  The decode difference is inside the B70's run-to-run noise and is not claimed.
- The existing fusions do not match this graph: the router has a sigmoid before its bias add, the expert sum is
  seven serial adds behind a per-expert scale, and the activation is built from unary ops.
- Consequence: the grouped executor of this task keeps one ids read per layer application but leases 128 experts
  where the direct route builds 4,096 operands. Whether that removes the exposed time is step 1's measurement.

## What is not known yet

- How much of the exposed host time is the ids wait and how much the operand build.
- How many Inkling prompt nodes are fully device-resident on the B50 alone.
