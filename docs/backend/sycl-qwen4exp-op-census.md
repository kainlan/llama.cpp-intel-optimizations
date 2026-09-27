# SYCL op census: qwen4exp (Qwen3.8-Flash-Next)

Tracker: `llama.cpp-k6jy`. Tree: master `fefb92980` (the upstream `694ec2354` merge, which brings in
qwen4exp from upstream `6c84c7d5d`, #27742).

**Status: static prediction done; measured columns are LEAD-RUN PENDING.** Nothing in this document
comes from a device run. Every "measured" cell is empty until the lead runs the commands in
[Lead-run census](#lead-run-census) and fills them from `census.md`.

Line numbers without a file name are `ggml/src/ggml-sycl/ggml-sycl.cpp` at `fefb92980`.

## Summary of the prediction

On an all-offloaded run (`-ngl 99`), five op families are predicted to leave the SYCL backend.
One of them is the PLE gather, which is CPU work by design. The other four are gaps.

| family | op as printed | nodes per graph | why | class |
|---|---|---:|---|---|
| hyper-connection pre-mix | `DSV4_HC_PRE` | 96 | no SYCL case, `default: false` (:108772) | gap |
| hyper-connection combine | `DSV4_HC_POST` | 96 | no SYCL case, `default: false` (:108772) | gap |
| GDN gate softplus | `SOFTPLUS` | 36 | not in the UNARY switch (:108205-108240); no softplus kernel in `ggml-sycl/` | gap |
| QSA indexer top-k | `TOP_K` | 12 | admitted only for `k <= 32` (:108684-108687); the indexer asks for up to 2051 | gap |
| token + PLE gathers | `GET_ROWS` | 2 | input-layer tensors (the PLE table is 50.66 GiB) | by design |

That is 242 CPU nodes per graph in 241 CPU splits, which comes to roughly 480 splits for each decode
token (every island costs a CPU split plus the SYCL split that follows it). The HC islands dominate,
and they exist only because of a fork-side rule; see
[The HC ops stay fused on the CPU](#the-hc-ops-stay-fused-on-the-cpu).

Two rows are **suspect** even though they are predicted to stay on SYCL:

- `GATED_DELTA_NET` (36 nodes). It is admitted unconditionally (:108728-108729). The kernel aborts
  on unsupported shapes rather than declining them, and it has no chunked PP kernel. It is also in
  the same family as `llama.cpp-30h4` (Qwen3.6-27B produces garbage prefill on SYCL). qwen4exp has 36
  of these layers.
- The BF16 indexer projections (24 `MUL_MAT`). They are admitted only through a one-time F32
  materialization at dispatch (:108305-108325), which keeps its own cache outside the planner.

A third row depends on placement: MoE `SWIGLU`. Layers whose experts the planner put on the host
may run their GLU on the CPU (:109207). That follows placement and is not a capability gap.

## Model facts (GGUF header only)

These values were read from the header of
`/models/Qwen3.8-Flash-Next-GGUF/Q8_0/Qwen3.8-Flash-Next-Q8_0-0000{1..6}-of-00006.gguf`. The
reader was pure Python and loaded no tensor data.

| key | value |
|---|---|
| layers | 48: 36 GDN (`attn_qkv`), 12 full attention at 3, 7, ..., 47 (`full_attention_interval` = 4) |
| n_embd / heads / kv heads / head dim | 2560 / 24 / 2 / 256; rope dim 64, sections [11, 11, 10, 0], IMROPE |
| GDN (ssm) | conv 4, state 128, groups (k heads) 16, dt_rank (v heads) 48, inner 6144 |
| hyper-connections | count 4, low rank 320 (so hc_dim = 10240) |
| QSA indexer | 4 heads x 128, top_k 2048, compress ratio 4 on all 12 full-attention layers |
| MoE | 512 experts, 10 used, expert FFN 640, shared-expert FFN 640 |
| PLE | layer 1; n-gram 3, 8 heads per n-gram (16 heads x 160); table `per_layer_token_embd` [160, 320001536] |
| tensor types | Q8_0 (811), F32 (389: norms, `ffn_gate_inp`, `ssm_a/dt/conv1d`, `ple_conv1d`), **BF16 (24: `indexer.{q,k}_proj`)** |
| sizes | experts 119.53 GiB, PLE table 50.66 GiB, dense 3.83 GiB, tok_embd + output 1.26 GiB |

What this implies for placement: dense and embedding weights (about 5.1 GiB) fit on either card.
The experts do not fit, so most of them are host-resident and run through the backend's
CpuExpertPool. The PLE table stays on the host.

## Graph-construction flags that decide which ops appear

- **Fused GDN is forced.** `cparams.fused_gdn_ar = fused_gdn_ch = true` and `auto_fgdn = false`
  (`src/llama-context.cpp:641-643`). Decode and PP therefore both emit one `GATED_DELTA_NET` per GDN
  layer, never the chunked `SOLVE_TRI`/`CUMSUM` expansion.
- **Fused HC is auto-resolved.** `auto_fhc = true` (`src/llama-context.cpp:651`), so
  `resolve_fused_ops()` probes it. See the next section for how that probe resolves on this fork.
- **FA is auto-resolved.** The QSA attention goes through `build_attn_mha`, which uses
  `FLASH_ATTN_EXT` when FA resolves on. SYCL admits D = 256 with F16 K/V
  (`ggml-sycl/fattn.cpp:2436-2497`), so FA is predicted on.
- `n_rs_seq` = 0 (the default) gives one conv-state rollback slot.

### The HC ops stay fused on the CPU

Upstream's `resolve_fused_ops()` disables a fused op whose node lands on a different device from its
layer. The fork added an exemption (`src/llama-context.cpp:1242-1263`): a landing **on the CPU** is
counted as "the executor follows data placement". The op then stays enabled, and the only effect is
a WARN.

The exemption was written for tiered-KV flash attention, where a CPU landing really is placement.
It applies to every probe, though. `DSV4_HC_PRE` and `DSV4_HC_POST` land on the CPU because SYCL
declines them (a capability gap), and the exemption keeps them fused there. Without it, the probe
would fall back to the unfused chain, which is all SYCL-supported ops (`SIGMOID`, `MUL`, `CONT`,
`ADD`, `SCALE`, `REPEAT`).

The resolve step prints its own WARN line, which is visible without `-lv 5`. It is a cheap
cross-check on the census:

```
resolve_fused_ops: fused DeepSeek V4 HC pre executes on CPU for 96 layer(s) (e.g. layer 47) -- the executor follows data placement, not a capability gap
```

## Op inventory and static SYCL verdict

T is the token count: 1 in decode, 512 in PP (the `-ub 512` reserve graph). S is the sequence count
(1). "Count" is the node count per graph. It is the same for T = 1 and T = 512 unless noted.

Measured columns: **lead-run pending**. B70 is `level_zero:0`, B50 is `level_zero:1`.

### Hyper-connections (2 mixers + 2 combines per layer, plus one head mixer)

| op | src types -> dst | shape class | count | verdict | cite | B70 | B50 |
|---|---|---|---:|---|---|---|---|
| REPEAT (hc_init) | f32 -> f32 | [2560,1,T] -> [2560,4,T] | 1 | SUPPORTED | :108522-108537 | | |
| RMS_NORM (grouped) | f32 -> f32 | [2560,4,T] | 97 | SUPPORTED | :108597-108607 | | |
| MUL (gamma) | f32 x f32 -> f32 | [2560,4,T] x [2560,4] | 97 | SUPPORTED | :108539-108547 | | |
| MUL_MAT `hc_*_down` | q8_0 x f32 -> f32 | [10240 -> 320], T cols | 97 | SUPPORTED | :108271-108354 | | |
| SCALE, SILU | f32 | [320,T] | 97 each | SUPPORTED | :108641, :108205-108234 | | |
| MUL_MAT `hc_*_up` | q8_0 x f32 -> f32 | [320 -> 10240] (K = 320) | 97 | SUPPORTED | :108271-108354 | | |
| **DSV4_HC_PRE** (gated) | f32, f32 -> f32 | x, gate [2560,4,T] -> [2560,T] | 96 | **UNSUPPORTED** | :108772; kept fused by `src/llama-context.cpp:1263` | | |
| MUL_MAT `hc_*_inject` | q8_0 x f32 -> f32 | [10240 -> 4] (4 output rows) | 96 | SUPPORTED | :108271-108354 | | |
| SCALE, SIGMOID, SCALE (combine weights) | f32 | [4,T] | 96 each | SUPPORTED | :108641, :108205-108234 | | |
| **DSV4_HC_POST** (identity comb) | f32 x3 -> f32 | x [2560,T], residual [2560,4,T], post [4,T] | 96 | **UNSUPPORTED** | :108772; same exemption | | |
| head mixer (il = -1, unfused) | f32 | SIGMOID/MUL [10240,T], CONT, ADD x3, SCALE | 1 each | SUPPORTED | :108205-108234, :108539-108547, :108644 | | |

### GDN linear attention (36 layers)

| op | src types -> dst | shape class | count | verdict | cite | B70 | B50 |
|---|---|---|---:|---|---|---|---|
| MUL_MAT `attn_qkv`, `attn_gate`, `ssm_out` | q8_0 x f32 | [2560 -> 10240], [2560 -> 6144], [6144 -> 2560] | 36 each | SUPPORTED | :108271-108354 | | |
| MUL_MAT `ssm_alpha`, `ssm_beta` | q8_0 x f32 | [2560 -> 48] | 36 each | SUPPORTED | :108271-108354 | | |
| SIGMOID (beta) | f32 | [1,48,T,S] | 36 | SUPPORTED | :108205-108234 | | |
| ADD (alpha + `ssm_dt`) | f32 | [48,T,S] + [48] | 36 | SUPPORTED | :108539-108547 | | |
| **SOFTPLUS** | f32 -> f32 | [48,T,S] | 36 | **UNSUPPORTED** | :108240 (UNARY default) | | |
| MUL (x `ssm_a`) | f32 | [48,T,S] x [48] | 36 | SUPPORTED | :108539-108547 | | |
| GET_ROWS (conv and ssm state rows) | f32 -> f32 | rows of [30720] and [786432] | 72 | SUPPORTED | :108356-108373 | | |
| CONCAT (state ++ x^T) | f32 | [3,10240,S] ++ [T,10240,S] | 36 | SUPPORTED | :108548-108557 | | |
| CONT + CPY (conv tail -> cache) | f32 | [3,10240,S] | 36 each | SUPPORTED | :108644, :108424-108515 | | |
| SSM_CONV | f32 x f32 -> f32 | [3+T,10240,S] x [4,10240] | 36 | SUPPORTED | :108730-108731 | | |
| SILU | f32 | [10240,T,S] | 36 | SUPPORTED | :108205-108234 | | |
| RMS_NORM + SCALE (l2 norm of q, k views) | f32 | [128,16,T,S], unit inner stride | 72 each | SUPPORTED | :108597-108607 | | |
| **GATED_DELTA_NET** | f32 x6 -> f32 | q,k [128,16,T,S], v [128,48,T,S], g,beta [1,48,T,S], s [128,128,48,S] | 36 | **SUSPECT** | :108728-108729; `gated_delta_net.cpp:193,251,293-300` | | |
| CPY (new state -> cache) | f32 | [786432,S] | 36 | SUPPORTED | :108424-108515 | | |
| RMS_NORM, MUL, SIGMOID, MUL (gated norm) | f32 | [128,48,T,S] | 36 each | SUPPORTED | :108597-108607, :108205-108234 | | |

The GDN suspicion, in detail:

- Admission does not match execution. The kernel accepts only S_v in {16, 32, 64, 128}
  (`GGML_ABORT` at `gated_delta_net.cpp:251`) and asserts contiguous g, beta and state
  (`:293-300`). supports_op returns `true` without checking any of that. qwen4exp's shapes pass
  (S_v = 128, contiguous operands), so this is a latent abort, not a qwen4exp fault.
- The broadcast convention matches the CPU reference. The kernel uses
  `iq1 = fastmodulo(h_idx, neqk1)` (`gated_delta_net.cpp:40`) and the CPU uses `iq1 = iv1 % neq1`
  (`ggml-cpu/ops.cpp:11090`); both are tiled. So H_k = 16 against H_v = 48 is not the suspect.
- There is no chunked kernel (`gated_delta_net.cpp:193`). PP with T = 512 runs the token loop
  serially in each work-group. That is a performance gap, not a split.
- `llama.cpp-30h4` is correctness evidence against the same family on this backend. Qwen3.6-27B
  gives PPL 473319 in prefill. The census cannot see wrong numbers. See ticket P5.

### QSA sparse attention (12 layers)

n_kv is the attention cache length (256-padded). n_blocks = ceil(n_kv / 4). The top-k width is
min(n_kv, 2048 + 4 - 1 = 2051).

| op | src types -> dst | shape class | count | verdict | cite | B70 | B50 |
|---|---|---|---:|---|---|---|---|
| MUL_MAT `attn_q` ([q, gate]), `attn_k`, `attn_v`, `attn_output` | q8_0 x f32 | [2560 -> 12288], [2560 -> 512] x2, [6144 -> 2560] | 12 each | SUPPORTED | :108271-108354 | | |
| RMS_NORM + MUL (q/k norm) | f32 | q view [256,24,T], k [256,2,T] | 12 each | SUPPORTED | :108597-108607 | | |
| CONT (gate view) | f32 | [6144,T] | 12 | SUPPORTED | :108644 | | |
| ROPE (IMROPE) q, k | f32 | [256,24,T], [256,2,T], n_rot 64 | 24 | SUPPORTED | :108657-108672, `rope.cpp:197` | | |
| SET_ROWS K, V -> cache | f32 -> f16 | [512,T] rows | 24 | SUPPORTED | :108375-108404 | | |
| MUL_MAT `indexer.k_proj`, `indexer.q_proj` | **bf16** x f32 | [2560 -> 128], [2560 -> 512] | 12 each | **SUSPECT** | :108305-108325 | | |
| SET_ROWS indexer key -> idx cache | f32 -> f16 | [128,T] rows | 12 | SUPPORTED | :108375-108404 | | |
| GET_ROWS block members | f16 -> f32 | [128, n_kv] by i32 [4*n_blocks, S] | 12 | SUPPORTED | :108356-108373 | | |
| CONT x4, ADD x3, SCALE (mean pool) | f32 | [128, n_blocks, S] | 12 per op | SUPPORTED | :108644, :108539-108547 | | |
| RMS_NORM + MUL + ROPE (pooled keys) | f32 | [128, 1, n_blocks*S] | 12 each | SUPPORTED | as above | | |
| RMS_NORM + MUL + ROPE (indexer q) | f32 | [128,4,T] | 12 each | SUPPORTED | as above | | |
| MUL_MAT (score) | f32 x f32 | [128, n_blocks, S] x [128, 4T, S] | 12 | SUPPORTED | :108271-108354 | | |
| RELU, CONT, ADD x3 (head sum), ADD (block bias) | f32 | [n_blocks, T, S] | 12 per op | SUPPORTED | :108205-108234 | | |
| CONT(PERMUTE) + GET_ROWS + CONT(PERMUTE) (expand to cells) | f32 | [T, n_blocks, S] by i32 [n_kv, S] | 12 each | SUPPORTED | :108356-108373 | | |
| CPY (mask f16 -> f32) + ADD | f16 -> f32 | [n_kv, T, S] | 12 each | SUPPORTED | :108424-108515 | | |
| **TOP_K** | f32 -> i32 | [n_kv, T, S], k = min(n_kv, 2051) | 12 | **UNSUPPORTED** | :108684-108687, `SYCL_TOPK_MAX_K` = 32 at :45087 | | |
| CONT (top-k) | i32 | [k, T, S] | 12 | SUPPORTED | :108644 | | |
| FILL (-inf mask; zeros) | f16; f32 | [n_kv, T_pad, 1, S]; [1, k, T, S] | 24 | **SUSPECT** | :108753; `fill.cpp:17,48` | | |
| SET_ROWS (unmask the top-k cells) | f32 -> f16 view | row size 1, i32 [k, T, S] | 12 | SUPPORTED | :108375-108404 | | |
| ADD (f16 mask + kq_mask) | f16 | [n_kv, T_pad, 1, S] | 12 | SUPPORTED | :108539-108547 | | |
| FLASH_ATTN_EXT | f32 q, f16 K/V/mask -> f32 | D 256, 24 q heads, 2 kv heads | 12 | SUPPORTED | `fattn.cpp:2436-2497` | | |
| SIGMOID, MUL (output gate) | f32 | [6144,T] | 12 each | SUPPORTED | :108205-108234 | | |

The TOP_K gap is structural: n_kv is at least 256 once the cache is padded, and the kernel's
32-wide limit comes from its shared-memory layout (256 x 32 x 8 B). No context size fits it.

### MoE (48 layers)

| op | src types -> dst | shape class | count | verdict | cite | B70 | B50 |
|---|---|---|---:|---|---|---|---|
| MUL_MAT `ffn_gate_inp` (router) | f32 x f32 | [2560 -> 512] | 48 | SUPPORTED | :108271-108354 | | |
| SOFT_MAX | f32 | [512,T] | 48 | SUPPORTED | :108648 | | |
| ARGSORT (argsort_top_k 10) | f32 -> i32 | [512,T] | 48 | SUPPORTED | :108681-108682 (2 KiB <= smpbo) | | |
| GET_ROWS (selected probs) | f32 | [1,512,T] by i32 [10,T] | 48 | SUPPORTED | :108356-108373 | | |
| SUM_ROWS, CLAMP, DIV (weight norm) | f32 | [10,T] | 48 each | SUPPORTED | :108677-108680, :108572-108583 | | |
| MUL_MAT_ID `ffn_{up,gate}_exps` | q8_0 x f32 | [2560 -> 640] x 512, 10 used | 96 | SUPPORTED | :108134-108176, `moe-mmvq-tables.hpp:78` | | |
| SWIGLU | f32 | [640,10,T] | 48 | PLACEMENT | :108244-108269; declined by :109207 when a layer's experts are host-planned | | |
| MUL_MAT_ID `ffn_down_exps` | q8_0 x f32 | [640 -> 2560] x 512 | 48 | SUPPORTED | as above | | |
| MUL, ADD x9 (weight and aggregate) | f32 | [2560,10,T] | 48 per op | SUPPORTED | :108539-108547 | | |
| MUL_MAT shared `up`/`gate`/`down` | q8_0 x f32 | [2560 -> 640] x2, [640 -> 2560] | 48 each | SUPPORTED | :108271-108354 | | |
| SWIGLU (shared) | f32 | [640,T] | 48 | SUPPORTED | :108244-108269 | | |
| MUL_MAT `ffn_gate_inp_shexp` | f32 x f32 | [2560 -> 1] (1 output row) | 48 | SUPPORTED | :108271-108354 | | |
| SIGMOID, MUL, ADD (shared gate) | f32 | [1,T]; [2560,T] | 48 each | SUPPORTED | :108205-108234 | | |

The expert FFN width 640 is not a multiple of 256. Q8_0 blocks are 32 wide, so admission is
unaffected, but no MMID kernel on this fork has been exercised at K = 640 or N = 640 before.

### PLE (layer 1 only)

| op | src types -> dst | shape class | count | verdict | cite | B70 | B50 |
|---|---|---|---:|---|---|---|---|
| GET_ROWS `per_layer_token_embd` | q8_0 -> f32 | [160, 320001536] by i32 [16T] | 1 | BY-DESIGN (CPU) | input-layer tensor, `src/llama-arch.cpp:915` | | |
| MUL_MAT `ple_key`, `ple_value` | q8_0 x f32 | [2560 -> 10240], [2560 -> 2560] | 1 each | SUPPORTED | :108271-108354 | | |
| RMS_NORM + MUL (grouped) x3 | f32 | [2560,4,T] | 3 each | SUPPORTED | :108597-108607 | | |
| MUL, SUM_ROWS, SCALE | f32 | [2560,4,T] -> [1,4,T] | 1 each | SUPPORTED | :108677-108680 | | |
| ABS, CLAMP, SQRT, SGN, MUL, SIGMOID (signed-sqrt gate) | f32 | [1,4,T] | 1 each | SUPPORTED | :108205-108234, :108572-108583 | | |
| REPEAT, MUL (value broadcast) | f32 | [2560,1,T] -> [2560,4,T] | 1 each | SUPPORTED | :108522-108537 | | |
| GET_ROWS, CONCAT, CONT, CPY (conv history, 9 columns) | f32 | [9,10240,S] | 1 each | SUPPORTED | :108548-108557 | | |
| CONT(TRANSPOSE), CONT(kernel column), MUL, ADD (dilated conv, 4 taps) | f32 | [10240,T,S] | 4 / 4 / 4 / 3 | SUPPORTED | :108644, :108539-108547 | | |
| SILU, CONT, ADD x2 | f32 | [2560,4,T] | 1 / 1 / 2 | SUPPORTED | :108205-108234 | | |

### Input and output

| op | src types -> dst | shape class | count | verdict | cite | B70 | B50 |
|---|---|---|---:|---|---|---|---|
| GET_ROWS `token_embd` | q8_0 -> f32 | [2560, 248320] by i32 [T] | 1 | BY-DESIGN (CPU) | input layer; `get_op_batch_size` returns 0 for GET_ROWS (:108846-108850) | | |
| GET_ROWS `out_ids` (last layer) | f32 | cur, inject, res_hc | 3 | SUPPORTED | :108356-108373 | | |
| MUL_MAT `output` | q8_0 x f32 | [2560 -> 248320] | 1 | SUPPORTED | :108271-108354 | | |

## What the census cannot see

The dump shows scheduler placement only. Read these gaps before calling a row clean:

- **Backend-internal host execution.** SYCL runs some admitted ops on the host inside the backend,
  where the scheduler still reports SYCL0:
  - host-resident experts go through the CpuExpertPool inside `MUL_MAT_ID`;
  - the MoE routing-subgraph interception (`ggml_sycl_op_is_moe_routing_subgraph`, :81199);
  - the KV-host flash-attention intercept.

  For qwen4exp this covers most of the 119.5 GiB of experts.
- **Wrong answers.** An op on SYCL0 can still be wrong. `GATED_DELTA_NET` needs its own correctness
  check (P5).
- **Types.** The dump prints no tensor types. The types in the tables above come from the GGUF
  header. `--tensor-types` resolves `src[0]` weight names only.
- **View ops.** VIEW, RESHAPE, PERMUTE and TRANSPOSE nodes are not printed.
- **Truncation.** The dump truncates op names to 10 characters (`GATED_DELT`, `DSV4_HC_PR`,
  `FLASH_ATTN`) and node names to 20. The parser compares at those widths.

## Lead-run census

GPU work, lead session only, one command at a time. `scripts/sycl-qwen4exp-census-run.sh` wraps
each run. It checks that `GGML_SYCL` is ON and that the binary links the SYCL backend, samples
`Shmem`/`MemAvailable` before the run and 5 s after it, and refuses to start if Shmem is at least
30 GB or MemAvailable is under the floor (qwen4exp 200 GB, control 30 GB). It then runs through
`bench-guard.sh`, which applies the timeout, the GPU-fault journal check and a VALID/SUSPECT
stamp. A watchdog samples every 2 s into `mem.log` and kills the run's process group if
MemAvailable drops under 20 GB. Finally it runs the parser.

What each run executes, for reference:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:<0|1> scripts/bench-guard.sh --budget <300|1800> -- \
  env GGML_SCHED_DEBUG=2 ./build/bin/llama-completion -m <model> -ngl 99 -c 4096 -b 512 -ub 512 \
    -fa <on|auto> --no-warmup -no-cnv -lv 5 --seed 42 --temp 0 -p '1, 2, 3, 4, 5,' -n 2 \
  > run.out 2> run.err
```

Why each flag matters:

- `GGML_SCHED_DEBUG=2`. At 1, the dump prints split headers only; at 2 it adds per-node backend
  lines (`ggml-backend.cpp:1734`). The parser returns VOID on a headers-only log.
- `-lv 5` (or `-v`). The dump is `GGML_LOG_DEBUG`, and `common_log` drops DEBUG below verbosity 5
  (`common/log.cpp:109`). Without it the log holds no dump at all, which the parser also
  reports as VOID rather than "no splits".
- Stderr goes to a separate file. The dump goes to stderr. Merging stdout into it lets token output
  tear dump lines.
- No `--log-prefix` or `--log-timestamps`. Each dump line is assembled from several log calls, and a
  prefix would land mid-line.
- `--no-warmup`. The MoE warmup graph activates all 512 experts, which changes the ARGSORT width
  and touches every host expert.
- `-c 4096`. Without it, n_ctx_train is 262144 (see `llama.cpp-uize` for what a default context does
  on this fork).
- Graph shapes. The reserve at `-ub 512` supplies the PP graph (T = 512), and the `-n 2` decode
  supplies T = 1. The parser classifies each dump by the size of the token-embedding gather.

Order and commands:

```bash
cd /Apps/llama.cpp && source /opt/intel/oneapi/setvars.sh --force
O=/Apps/llama.cpp/census-k6jy     # not /tmp: tmpfs does not survive a reboot

# 1. positive control: FLASH_ATTN_EXT declined on SYCL (fattn.cpp:2438) -> 32 CPU nodes per graph
scripts/sycl-qwen4exp-census-run.sh control-on  0 $O/control-on-b70    # parser: --require FLASH_ATTN_EXT=CPU:32
# 2. negative control: same run, no decline -> 32 on SYCL0
scripts/sycl-qwen4exp-census-run.sh control-off 0 $O/control-off-b70   # parser: --require FLASH_ATTN_EXT=SYCL:32
# 3. the census, B70 then B50 -- detached: the first load of 177 GiB is unmeasured and can pass the 10-min tool ceiling
setsid nohup scripts/sycl-qwen4exp-census-run.sh qwen4exp 0 $O/qwen-b70 > $O/qwen-b70.log 2>&1 &
#    poll $O/qwen-b70.log for "parser rc="; wait for Shmem < 30 GB and MemAvailable > 200 GB, then
setsid nohup scripts/sycl-qwen4exp-census-run.sh qwen4exp 1 $O/qwen-b50 > $O/qwen-b50.log 2>&1 &
```

Expected runtime:

- Each control: about 1 min (a 3.9 GB load plus reserve).
- Each qwen4exp run: the model load dominates and has never been measured on this host. The model
  is 177 GiB on bcachefs over NVMe, SATA and a USB SSD, so expect 5-15 min cold. The budget is
  1800 s.

Memory: host residency is roughly 120 GiB of host experts, plus the PLE table if it is copied into
a host buffer, plus 5 GiB of dense weights. That is why the floor is 200 GB and the watchdog is
armed. If the preflight refuses, `--lazy-mode on` may keep the PLE table on disk. Lazy mode resolves OFF
automatically on SYCL, and so does mmap itself: the SYCL caps initializer never sets
`mmap_support` (:107798-107803), and `src/llama-model.cpp:2245-2265` turns off both `use_mmap` and
AUTO lazy mode for such a device. The whole file is therefore read into host buffers. Forcing lazy
mode on SYCL is untested.

How to score a run:

1. Both controls must exit 0. If control-on exits 2, the parser or the log settings are broken.
   Treat any qwen4exp census taken with that setup as void.
2. qwen4exp exits 0 if its structural controls hold: `GATED_DELTA_NET` x36 and `MUL_MAT_ID` x144 in
   every dump class. Exit 2 means the dump is missing or truncated, not "no gaps".
3. `census.md` holds, for each T, the per-op backend table, the per-rule AGREE/DISAGREE/ABSENT
   diff against `scripts/sycl-qwen4exp-op-prediction.json`, and every unpredicted CPU node. Copy
   the backend and split counts into the B70/B50 columns above.
4. Cross-check `run.err` for the HC WARN line quoted above. It is present at default verbosity too.

The parser alone:

```bash
python3 scripts/parse-sycl-sched-census.py run.err --prediction scripts/sycl-qwen4exp-op-prediction.json \
    [--n-tokens 1 --n-tokens 512] [--tensor-types <"T name dims type" listing>] [--require OP=CPU|SYCL|ANY[:MIN]] [--strict]
```

Its gate is `tests/test-sycl-sched-census.py` (ctest `test-sycl-sched-census`, pure Python). The
gate plants CPU splits in a synthetic dump written with the printer's exact formats, including a
torn line. It requires the counts to come out as planted, requires an empty or headers-only log to
be VOID, and checks that every prediction rule names an op ggml prints.

## Proposed closure tickets (not filed: the lead files them once the census confirms)

**P1. sycl: GGML_UNARY_OP_SOFTPLUS kernel + supports_op case.**
- Scope: an elementwise softplus in the unary family (f32, and f16 under `GGML_SYCL_F16`),
  admitted in the UNARY switch (:108205).
- RED: `test-backend-ops -b SYCL0 -o SOFTPLUS` reports the cases "not supported" (lead-run,
  pinned selector), and the census shows `GDN-SOFTPLUS` AGREE on CPU with 36 nodes.
- GREEN: the cases pass against CPU, and the census rule flips to DISAGREE (0 CPU nodes).
  Removes 36 islands, about 72 splits per token.

**P2. sycl: TOP_K for k > 32 (up to 2051 over n_kv up to the context size).**
- Scope: a top-k whose shared memory does not scale with k, for example a radix select or a
  bitonic partial sort per row. Indices are unordered, matching the CPU semantics
  (`ggml.h`, "no particular order").
- RED: `test-backend-ops -b SYCL0 -o TOP_K` with ne0 = 4096, k = 2051 is "not supported", and the
  census shows `QSA-TOPK` on CPU with 12 nodes.
- GREEN: that case passes, and QSA-TOPK moves to SYCL0.

**P3. sycl: DSV4_HC_PRE (gated and ungated), DSV4_HC_POST and DSV4_HC_COMB kernels.**
- Scope: the three hyper-connection ops that upstream implements on CPU (`ggml-cpu.c:2100-2110`),
  shared with DeepSeek-V4.
- RED: `test-backend-ops -b SYCL0 -o DSV4_HC_PRE` / `-o DSV4_HC_POST` are "not supported", and the
  census shows 96 + 96 CPU nodes.
- GREEN: these pass, and the HC WARN line reads "enabled" with no CPU landings. This is the
  biggest item: 192 islands, about 384 splits per token.

**P4. llama-context: scope the fused-op CPU-landing exemption to placement.**
- Problem: `resolve_fused_ops` (`src/llama-context.cpp:1242-1263`) treats every CPU landing as
  placement. A landing caused by a backend declining the op is a capability gap, and upstream's
  answer to one (disable the fusion and use the unfused chain) is the better fallback.
- Proposed fix: before exempting, ask whether the layer's own device supports the node
  (`ggml_backend_dev_supports_op(device_layer, node.tensor)`). If it does not, the landing is a
  capability gap and the fusion is disabled.
- RED: on qwen4exp, the WARN reports "executes on CPU for 96 layer(s)" for HC pre. On any build
  without P3, the fix turns that into "not supported, set to disabled" and moves the unfused chain
  onto SYCL0.
- This touches an owner ruling (placement decides the executor), so it needs an owner decision.
  P3 makes it moot for HC but not for the next fused op.

**P5. sycl: GATED_DELTA_NET honesty, prefill correctness and PP throughput.**
- (a) supports_op mirrors the kernel's real envelope: S_v in {16, 32, 64, 128}, contiguous g, beta
  and state. RED: a `test-backend-ops` GDN case with S_v = 256 aborts instead of reporting "not
  supported".
- (b) Correctness at the qwen4exp shape: S_v 128, H_k 16, H_v 48, T in {1, 512}, K in {1, 2},
  plus a `llama-perplexity` repeated-paragraph run as in 30h4. RED: whatever 30h4's localisation
  shows for GDN. The `-o GATED_DELTA_NET` cases at that shape are the discriminator.
- (c) A chunked PP kernel (`gated_delta_net.cpp:193`).
- (b) blocks every GDN model on SYCL. File it against, or under, `llama.cpp-30h4`.

**P6. sycl: FILL supports_op honesty.**
- Scope: admit only a contiguous dst of type F32 or F16, matching `fill.cpp:17,48`.
- RED: a `test-backend-ops` FILL case with a BF16 or non-contiguous dst aborts instead of
  reporting "not supported".
- Small, and not a qwen4exp blocker: its FILLs are contiguous F16/F32.

**P7. sycl: BF16 weight materialization belongs to the planner.**
- Problem: the 24 BF16 indexer projections run through a dispatch-time F32 materialization with its
  own `g_sycl_bf16_materialize_cache` (:14177). The layout is decided at dispatch instead of
  by the planner, which is the "one fact, two sources" shape. Whether its allocations go through the
  unified cache also needs checking.
- Scope: the planner materializes the F32 (or a native BF16 kernel reads it), and dispatch reads the
  loaded layout.
- RED: the census shows the indexer `MUL_MAT` on SYCL0 while the materialization cache is populated
  at first dispatch. The observable is a trace of that cache, not the census.

Not a ticket: MoE `SWIGLU` on the CPU for host-expert layers (:109207) is placement deciding the
executor. If the census shows it, record the layer count, because it bounds the host-side work per
token.

## Principles check for the proposals

- **P1-P3** add kernels for layouts the loaded operands already have. They add no allocator, no
  zero-copy and no streaming: this is support-gap closure, not re-routing.
- **P4** separates a capability decline from placement. It does not move data.
- **P7** moves a layout decision from dispatch to the planner, so the layout has one source.
- None of them adds a host wait. Today's CPU islands already cost a device-host round trip each,
  which is the per-token synchronization P1-P3 remove.
