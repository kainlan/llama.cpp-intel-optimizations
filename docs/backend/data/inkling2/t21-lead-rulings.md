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

## Step 1 (a-c) measured (lead's runs, each once, `task/inkling2-t21` at `87524b0d8` against base `b44cb4b5f`)

Step 1 is the `MOE_TENSOR_GATE_UP` role, NVFP4 on the grouped executor's F16 arm, ring sizing that counts the
fused tensor, and the re-entry of the direct route on a decline.

| check | base | step 1 |
|---|---|---|
| B70 long prompt, pp t/s (order: base, new, new, base) | 92.31, 90.00 | 160.51, 159.45 |
| B70 long prompt, tg t/s | 25.92, 25.45 | 25.93, 25.45 |
| B70 peak host backing | 21 GiB | 21 GiB |
| B70 691-token prompt under the kernel profiler | 98.9 t/s | 171.1 t/s |
| B70 perplexity, 6 chunks of 2048 tokens of this repository's docs | 7.5741 +/- 0.2468 | 7.5792 +/- 0.2470 |
| B50 alone long prompt, pp / tg t/s | 17.32 / 7.00 | 16.46 / 6.56 |
| B50 alone, experts served from the host mapping | 10215.0 MB | 10774.6 MB |

- Device test `test-moe-pp-batched-dequant` on the B50: 43 cases pass; the NVFP4 cases (plain, small and large
  scales, fused gate_up) at nmse 1.41e-05 to 1.43e-05 against the CPU backend; no re-entry.
- The step 1 text is byte-equal run to run and differs from base (the activations are F16 where the old kernel
  quantised them to Q8_1); the perplexity pair says quality is unchanged.
- Profile at 691 tokens: `moe.pp.gemm.execute` 320.9 ms over 1,794 launches, gather and scatter 108 ms over
  17,086 launches each, `mulmat.mmvq.id_aos_nvfp4` down from 3,192 ms to 5 ms. The label `moe.pp.dequant.nvfp4`
  times a marker, not the converter (the converter drops its event), so the dequant's device time is in no
  label and the "exposed host time" of that profile is overstated. Next in size: dense BF16 698 ms, BF16
  experts 580 ms, attention 430 ms.
- **The B50 alone is slower with the ring**: the 558.6 MiB ring displaced about 560 MB of experts to the host
  and one more layer to the CPU, for prompt and decode. Hence the final form of R5 below.
- **R5, final form.** For a ring whose weights have the resident use only (NVFP4 today), the ring is kept on a
  device only if the plan made with it leaves no expert of those weights on the host; otherwise the ring is
  published as 0, the zones are ensured again (an arena nothing holds may be rebuilt) and the plan is made once
  more. Weights with a staged or SOA per-expert use (Qwen, GPT-OSS) keep today's ring exactly. Acceptance: the
  B50 alone gives base's plan line, base's text and base's speed; the B70 keeps its ring and its 160 t/s.
- Spec review of a-c (fresh reviewer, read-only): pass, eight minor findings, all sent to the implementer. Only
  Inkling, of the models on this machine whose headers were read, stores a fused `ffn_gate_up_exps`.

## Step 1d measured, and R5 as it stands (lead's runs, each once, `69c4ea428` against base `b44cb4b5f`)

Long real prompt as above (Inkling: 4,737 tokens then 254 generated; Qwen: 4,485 then 127), `-c 8192`, seed 42,
temp 0, `llama-completion`. Zero kernel fault lines in every run.

| run | prompt t/s | decode t/s | what the log and the text say |
|-----|------------|------------|-------------------------------|
| Inkling, B70, new | 160.89 | 26.13 | ring kept; text byte-equal to step 1's run at `87524b0d8` |
| Inkling, both cards, new | 160.92 | 25.80 | text byte-equal to the B70 run |
| Inkling, both cards, base | 91.29 | 26.02 | |
| Inkling, B50 alone, new | 15.17 | 7.38 | ring dropped, RUNTIME zone rebuilt at 512.0 MB; final plan 10215.0 MB |
| Inkling, B50 alone, base | 17.76 | 4.74 | final plan 10215.0 MB; text byte-equal to new |
| Qwen3.6-35B NVFP4, B70, base | 68.07 | 14.31 | |
| Qwen3.6-35B NVFP4, B70, new | 74.37 | 14.31 | ring kept; text differs from base inside the summary, reads right |
| Qwen3.6-35B NVFP4, B50, new | 25.02 | 7.81 | ring dropped, "no room to give back"; final plan 6572.8 MB |
| Qwen3.6-35B NVFP4, B50, base | 23.94 | 7.52 | final plan 6572.8 MB; text byte-equal to new |

- The Inkling B50 pair is **not a speed verdict**: a `ctest` was alive at the end of the new arm and `ninja` with
  four `icpx` at the end of the base arm, and prompt and decode move in opposite directions. Plan and text are
  base's. A second pair runs on the fixed tree. No builder was alive in any Qwen arm.
- The drop on the Inkling B50 at `69c4ea428`, from the log: "the plan made with it left 908 experts of its 40
  resident-only expert weights on the host ... its RUNTIME zone was rebuilt at 512.0 MB, 46.6 MB less". The three
  plans of that load served 9715.3 MB (with the ring), 9668.3 MB (after the drop) and 10215.0 MB (the load's own
  final plan, equal to base's 10215.0 MB) of experts from the mapping. Peak GPU host backing (`Shmem` plus the
  `/proc/meminfo` remainder): 21 GiB new and 24 GiB base on the B50 alone; 21 GiB on the B70 alone and 22 GiB on
  both cards. Qwen3.6-35B NVFP4: 17 GiB in every arm.
- Device checks at `69c4ea428`: `test-moe-pp-batched-dequant`, `test-moe-q1-nvfp4-dispatch-refusal` and
  `test-moe-mixed-residency-q1-nvfp4` on the B50 exit 0; the Inkling count gate passes on the B70, the B50 alone
  and both cards; the Mistral B70 and GPT-OSS B50 (no `-c`) gates pass.
- Second spec review (fresh reviewer): fail, two major and five minor findings, all sent to the implementer.
  The majors: the drop mark was a file-scope static missing from the static-storage inventory, and the text
  said "NVFP4 today" and "Qwen and GPT-OSS plans do not change" where the rule by use also reached IQ1_S, IQ1_M
  and IQ4_XS and a Qwen with NVFP4 experts.
- **R5 as it stands** (it replaces every earlier form). A device's ring is dropped only when all four hold:
  (a) every weight that uses the ring has the resident use alone (no staged window, no SOA per-expert staging);
  (b) every such weight has another device route for a resident prompt node (today the MMID direct route, so
  the set is NVFP4; the ring is the only device prompt route of IQ1_S, IQ1_M and IQ4_XS, found by the
  implementer from the code); (c) the plan made with the ring leaves an expert of those weights on the host;
  (d) dropping returns room to the weights: the zone needs less without the ring and the arena may be rebuilt.
  A drop then means ring gone, zone rebuilt smaller, one more plan. In every other case the ring stays
  published and the first plan stands. Clause (d) comes from the Qwen B50 row: there the RUNTIME zone is sized
  by another consumer (726.4 MB with and without the ring), so the drop at `69c4ea428` returned nothing and
  only took the executor from the resident layers' prompts.
- Known cost left open: a model with IQ1_S, IQ1_M or IQ4_XS experts that overflow a card still pays the ring's
  room, because those types have no device prompt route without it.
- Sweeps at `69c4ea428` (implementer): master's five red, plus `test-sycl-static-storage-audit` and three
  source contracts that pinned the structure 1d changed on purpose; each contract is strengthened to name the
  new exits, not loosened.

## The WEIGHT-zone rule on the cards (lead's runs, each once, `f011d5d1f2` against base `b44cb4b5f`, 2026-11-25)

Same long prompts as above. Correctness only: every arm ran while another session built (`ninja` and up to 17
`icpx` alive at the start or end of each arm), so no throughput figure from this chain counts, in either direction.

- Inkling, B50 alone: one drop line, "its RUNTIME zone was rebuilt at 512.0 MB, 46.6 MB less, and its WEIGHT zone
  grew from 12579.9 MB to 12626.5 MB"; plans 9715.3, 9668.3, then the load's own 10215.0 MB, base's figure; text
  byte-equal to base.
- Qwen3.6-35B NVFP4, B50: the early pass keeps the ring (no drop line; the no-room keep prints at INFO only). The
  load's late pass drops it: "the plan made with it left 12435 experts of its 120 resident-only expert weights on
  the host ... its RUNTIME zone was rebuilt at 726.4 MB, 421.0 MB less, and its WEIGHT zone grew from 12164.6 MB to
  12585.6 MB". Plans 6827.6 (early, ring kept), 6994.7 (late, with the ring), 6572.8 MB (after the drop, base's
  figure). Text byte-equal to base. So on Qwen the room is real in the late pass, and the drop returns it.
- Inkling, B70 and both cards: no drop line; text byte-equal to the `69c4ea428` run and to each other.
- Qwen3.6 NVFP4 perplexity, B70, `-c 2048 --chunks 6`: base 15.7496 +/- 0.6006, new 15.7449 +/- 0.6004.
- `test-moe-pp-batched-dequant`, `test-moe-q1-nvfp4-dispatch-refusal`, `test-moe-mixed-residency-q1-nvfp4` on the
  B50: exit 0. Inkling count gates (B70, B50 alone, both cards), Mistral B70 and GPT-OSS B50 gates: pass. Zero kernel
  fault lines. Peak GPU host backing at most 20 GiB in every arm.
- My expectation written before the run ("Qwen B50 keeps its ring") was wrong: it read the `69c4ea428` early-pass
  line as the load's answer. The late pass is the one that counts.
- Owed before a speed claim on the B50: an interleaved pair with no builder alive.

## Step 1 merged to master (lead's runs, 2026-11-26, `b67e17fcb6` = fork/master `dd59d8710` + `43ea320243`)

Task 21 step 1 closed after r8 (spec PASS, quality APPROVED, narrow re-review APPROVED). The base arm in every pair
is `b44cb4b5f`, whose tree is byte-identical to `dd59d8710` (`git diff` empty). `llama-bench -p 512 -n 128`, ABBA,
one load per arm; a builder alive at either end of an arm makes that arm's figure a lower bound.

| Pair (card) | reps | pp512 base / new (t/s) | tg128 base / new (t/s) | window |
|---|---|---|---|---|
| Inkling Q8_K_XL (B70) | 3 | 112.31, 109.51 / 259.39, 259.96 | 29.02, 29.01 / 29.06, 28.97 | clean |
| Mistral 7B Q4_0 (B70) | 3 | 3343.04 / 3343.09 | 109.49 / 109.72 | first pair clean, second pair built over |
| GPT-OSS 20B MXFP4 (B50) | 8 | medians 865.36, 858.41 / 864.05, 860.67 | medians 35.82, 35.73 / 35.82, 35.43 | clean |
| Inkling, B50 alone, tg only | 8 | - | medians 10.37, 10.42 / 10.88, 10.59 | first pair built over, second clean |

- The scoring rule was fixed before the 8-rep runs: both new medians inside the base medians' range is noise, both
  below both base medians is a regression. GPT-OSS pp is inside; GPT-OSS tg and Inkling B50 tg have one new median
  outside on each side, neither both below. An earlier 3-rep GPT-OSS B50 pair read new 4% low with one ~760 t/s
  sample per new arm; its address-masked logs were identical between builds, and the 8-rep pair does not repeat it.
- Inkling B50 alone, final state in every arm: one 14240.0 MB arena (scratch 512, runtime 512, oneDNN 256, shared
  KV+WEIGHT 12960 MB) and 4576 of 6144 experts on the device, the same as base. The new build plans once more at
  load (the ring drop), then rebuilds to that arena.
- Long real-text prompt (4737 tokens, `-c 8192`, `-n 255`), merged tree: B70 154.66 pp / 26.07 tg (base earlier
  91.29 / 26.02). B50 alone: base 13.15 pp / 7.26 tg, new 15.32 / 6.22. A 24-job build started at 05:42:44, during
  the new arm's decode, so the new tg is not a figure; the pp phase ran clean. Output byte-equal to base.
- Merged-tree gates: Inkling count gate on the B70, the B50 alone and both cards; Mistral B70; GPT-OSS B50 (`grep
  -cx` = 1); the two contract sweeps red only on master's five (`test-sycl-moe-legacy-ranking-removed-source` timed
  out at 300 s under the gate chain's load and passes alone in 91 s). Zero kernel fault lines throughout.
- `MUL_MAT_ID` in `test-backend-ops`, both cards, `9991c401ac` build against base: identical OK, FAIL and
  not-supported sets (1306 / 114 / 322); no NVFP4 case fails; the failing cases fail on base too.
- Not measured on this sha: the Qwen3.6-35B NVFP4 speed pair.

## What is not known yet

- How much of the exposed host time is the ids wait and how much the operand build.
- How many Inkling prompt nodes are fully device-resident on the B50 alone.
- Why the route change gives Qwen3.6 NVFP4 on the B70 only 9% where Inkling gains 75%.
