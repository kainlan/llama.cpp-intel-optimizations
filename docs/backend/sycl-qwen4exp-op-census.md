# SYCL op census: qwen4exp (Qwen3.8-Flash-Next)

Tracker: `llama.cpp-k6jy`. Tree: master `fefb92980` (the upstream `694ec2354` merge, which brings in
qwen4exp from upstream `6c84c7d5d`, #27742).

**Status: static prediction done; qwen4exp columns are LEAD-RUN PENDING.** The only device runs so
far are the two Mistral controls on the B70 (see [Lead-run census](#lead-run-census)). Every qwen4exp
"measured" cell is empty until the lead fills it from a `census.md`.

**The real model cannot be censused on this host yet.** It needs about 175 GiB of host memory, because
SYCL forces mmap off, and MemAvailable measures 151-159 GiB (162-171 GB) under the host's permanent
load. The census
runs first on a synthetic qwen4exp vehicle; see [The census vehicle](#the-census-vehicle).

Line numbers without a file name are `ggml/src/ggml-sycl/ggml-sycl.cpp` at `fefb92980`.

## Summary of the prediction

On an all-offloaded run (`-ngl 99`), four op families are predicted to leave the SYCL backend, and
all four are capability gaps. The input gathers are predicted to stay on SYCL0, but that is itself
suspect (last row): if verified zero-copy, not sanctioned.

| family | op as printed | nodes per graph | why | class |
|---|---|---:|---|---|
| hyper-connection pre-mix | `DSV4_HC_PRE` | 96 | no SYCL case, `default: false` (:108772) | gap |
| hyper-connection combine | `DSV4_HC_POST` | 96 | no SYCL case, `default: false` (:108772) | gap |
| GDN gate softplus | `SOFTPLUS` | 36 | not in the UNARY switch (:108205-108240); no softplus kernel in `ggml-sycl/` | gap |
| QSA indexer top-k | `TOP_K` | 12 | admitted only for `k <= 32` (:108684-108687); the indexer asks for up to 2051 | gap |
| token + PLE gathers | `GET_ROWS` | 2 | the input layer's first buffer type is SYCL_Host, so the scheduler runs the gather on SYCL0 over a pinned host table (measured for `token_embd` on the Mistral control) | suspect; if verified zero-copy, not sanctioned (P8, `llama.cpp-lah5`) |

That is 240 CPU nodes per graph in about 240 CPU splits, which comes to roughly 480 splits for each
decode token (every island costs a CPU split plus the SYCL split that follows it). The HC islands dominate,
and they exist only because of a fork-side rule; see
[The HC ops stay fused on the CPU](#the-hc-ops-stay-fused-on-the-cpu).

Four rows are **suspect** even though they are predicted to stay on SYCL:

- `GATED_DELTA_NET` (36 nodes). It is admitted unconditionally (:108728-108729). The kernel aborts
  on unsupported shapes rather than declining them, and it has no chunked PP kernel. It is also in
  the same family as `llama.cpp-30h4` (Qwen3.6-27B produces garbage prefill on SYCL). qwen4exp has 36
  of these layers.
- The BF16 indexer projections (24 `MUL_MAT`). They are admitted only through an F32 materialization
  decided at dispatch (:108305-108325). Its cache holds `mem_handle`s (:14177), so ownership is
  sanctioned. The layout decision is the problem: dispatch makes it, not the planner.
- `FILL` (24). It is admitted unconditionally (:108753); the kernel asserts a contiguous F32/F16 dst.
- The input gathers (`token_embd`, `per_layer_token_embd`). The input layer's buffer list is the CPU
  device's, but its first entry is the first GPU's host buffer type (`src/llama-model.cpp:1765-1777`,
  `:2338`). The scheduler hands a leaf the backend of its buffer, so SYCL0 gathers from a pinned host
  table. The B70 Mistral control shows this for `token_embd`
  (`node #0 (GET_ROWS) embd [SYCL0]`, `token_embd.weight [SYCL0]`). For qwen4exp that would put the
  50.66 GiB PLE table in pinned host memory with a GPU reading it. Suspect, not established: the
  backend label is not the executor. Ruled 2026-09-27 (P8): if verified zero-copy, it is not
  sanctioned, and the gathers belong on ggml-cpu (`llama.cpp-lah5`, which verifies first).

**CPU landings that are placement, not gaps.** `ggml_sycl_op_is_planned_on_host` (:109159) declines
an op for any of these reasons. A CPU node the census shows for one of them follows placement and is
not a ticket:
- a `MUL_MAT` whose weight the planner executes on the host (:109185-109191);
- a layer-plan op with a host-planned source (:109193-109199);
- every op of a layer with `layer_device < 0` (:109225-109227);
- MoE `SWIGLU`/`ADD_ID` for a layer whose experts are host-planned (`addid_glu_depends_host`, :109207).

The KV-host residency check (:108067-108131) also declines `FLASH_ATTN_EXT` over demoted KV. On the
full model, the experts are host-resident but `MUL_MAT_ID` is admitted residency-blind (:108134), so
the host experts show as SYCL0 nodes running through the CpuExpertPool.

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

Two corrections apply to that line:
- **"96 layer(s)" counts fused nodes, not layers.** `n_cpu_landings` is incremented once per node
  (`src/llama-context.cpp:1263`), and there are two HC pre-mixers per layer, so 96 is 48 layers x 2.
  The vehicle prints 4 for its 2 layers.
- **"not a capability gap" is false for HC.** The exemption cannot tell a capability decline from
  placement, so it prints the placement wording for both (ticket P4).

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
| GET_ROWS `per_layer_token_embd` | q8_0 -> f32 | [160, 320001536] by i32 [16T] | 1 | **SUSPECT** (SYCL0 over SYCL_Host) | input-layer tensor (`src/llama-arch.cpp:915`) in the input buft list, whose first entry is SYCL_Host (`src/llama-model.cpp:1765-1777`) | | |
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
| GET_ROWS `token_embd` | q8_0 -> f32 | [2560, 248320] by i32 [T] | 1 | **SUSPECT** (SYCL0 over SYCL_Host) | same buft list; measured SYCL0 for the Mistral control's gather | | |
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
- **Where a weight lives.** A leaf's backend label is the backend of its buffer. A SYCL_Host pinned
  buffer therefore reads `[SYCL0]` exactly like device memory, and the dump cannot tell the two
  apart. That is why the input-gather rows are SUSPECT, not measured facts.
- **Wrong answers.** An op on SYCL0 can still be wrong. `GATED_DELTA_NET` needs its own correctness
  check (P5).
- **Types.** The dump prints no tensor types. The types in the tables above come from the GGUF
  header. `--tensor-types` resolves `src[0]` weight names only.
- **View ops.** VIEW, RESHAPE, PERMUTE and TRANSPOSE nodes are not printed.
- **Truncation.** The dump truncates op names to 10 characters (`GATED_DELT`, `DSV4_HC_PR`,
  `FLASH_ATTN`) and node names to 20. The parser compares at those widths.

## Lead-run census

GPU work, lead session only, holding `GPU.lock`, one command at a time.
`scripts/sycl-qwen4exp-census-run.sh` wraps each run and does not take the lock itself. In order, it:

1. Sources oneAPI when needed, then checks that `GGML_SYCL` is ON and that `ldd` resolves at least
   two SYCL libraries (it counts only lines with `=> /`, never "not found").
2. Waits for CLAUDE.md's post-lock settle: Shmem < 30 GB and MemAvailable > max(150 GB, the mode's
   need). The qwen4exp need is 215 GB (200 GiB). After 600 s (`SETTLE_TIMEOUT_S`) it exits 3 with
   "release GPU.lock and escalate". Every memory figure the script compares or prints is decimal GB
   (10^9 bytes), CLAUDE.md's unit; it compares in bytes, so the floor is exactly 150e9 bytes, and
   prints two decimals, with the bytes on a refusal, so a refusal at the floor never reads like a
   pass. Under the host's permanent load MemAvailable can sit near or below that floor, and then an
   exit 3 "not settled" is the rule working as intended, not a defect to work around: the lead
   retries later or escalates, and never lowers the floor.
3. Launches the run in its own session through `bench-guard.sh`. bench-guard enforces the budget
   (`timeout -k`), refuses to start if Shmem net of tmpfs exceeds its 10 GiB ceiling
   (`bench-guard.sh:73,190`), checks the kernel journal for GPU faults, and stamps VALID or SUSPECT.
   Its ceiling is the binding pre-run gate; the script's Shmem < 30 GB check is the looser one.
   One second after the launch it checks the session. If bench-guard has already exited, which is
   what a refusal looks like, it says so and takes the normal rc path (steps 5 and 6). If
   `setsid` forked instead, so the run is not in the session the watchdog would kill, it kills the
   forked session, waits for it to empty, and exits 3. It never kills its own session: a child that
   has not yet reached setsid() still carries the caller's session id, and that id is skipped.
4. Runs a watchdog that samples every 2 s into `mem.log`. It kills the **session**, TERM then KILL
   after 5 s, if MemAvailable drops under 20 GB (`WATCHDOG_FLOOR_GB`), or, in the control and
   vehicle modes, if Shmem exceeds 100 GB (`WATCHDOG_SHMEM_GB`). The controls measured ~1-2 GB (the
   vehicle has not run yet), so 100 GB means a runaway. The qwen4exp mode has no Shmem abort (see
   "The real model: blocked on host memory"). A process-group kill would miss the binary, because
   `timeout` moves itself and its child into their own group.
5. Blocks until the session is empty before it stamps the post sample or exits. If a process
   survives 120 s it exits 3 with "do NOT start another GPU run".
6. Fails the run (exit 3) unless `run.err` carries `bench-guard: VALID`. bench-guard exits with the
   command's rc, so a GPU fault during an rc 0 run shows up only in that line.
7. Runs the parser with `--n-tokens 1 --n-tokens 512`, so a dump class that is missing, or that
   cannot be classified, is VOID (exit 2).

What each run executes, for reference:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:<0|1> setsid -w scripts/bench-guard.sh --budget <600|1800> -- \
  env -u LLAMA_ARG_BACKEND_SAMPLING GGML_SCHED_DEBUG=2 \
    ./build/bin/llama-completion -m <model> -ngl 99 -c 4096 -b 512 -ub 512 \
    -fa <on|auto> --no-warmup -no-cnv -lv 5 --no-log-prefix --no-log-timestamps --seed 42 --temp 0 \
    -p '1, 2, 3, 4, 5,' -n 2 [--ignore-eos] > run.out 2> run.err
```

Why each flag matters:

- `GGML_SCHED_DEBUG=2`. At 1, the dump prints split headers only; at 2 it adds per-node backend
  lines (`ggml-backend.cpp:1734`). The parser returns VOID on a headers-only log.
- `-lv 5`. The dump is `GGML_LOG_DEBUG`, and `common_log` drops DEBUG below verbosity 5
  (`common/log.cpp:109`). Without it the log holds no dump at all, which the parser reports as VOID
  rather than "no splits".
- `--no-log-prefix --no-log-timestamps`. `llama-completion` calls `common_init()`
  (`tools/completion/completion.cpp:92`) before it parses its arguments, and `common_init()` turns
  on a prefix and timestamps (`common/common.cpp:391-392`). common_log prefixes **each** log call,
  so a default capture has `<M.ss.mmm.uuu> D ` in the middle of every dump line, between the node
  and its sources. The first version of this document said the opposite, and its parser lost
  `src[0]` on every node of the lead's control runs: `n_tokens classes: [None]`. The parser now
  strips both prefix forms, and `tests/golden/sched-census-mistral-b70-excerpt.err` (an excerpt of
  that capture) gates it. The flags stay because they keep the log readable.
- Stderr goes to a separate file. The dump goes to stderr. Merging stdout into it lets token output
  tear dump lines.
- `--no-warmup`. The MoE warmup graph activates all 512 experts, which changes the ARGSORT width
  and touches every host expert.
- `-c 4096`. Without it, n_ctx_train is 262144 (see `llama.cpp-uize` for what a default context does
  on this fork).
- `--ignore-eos`, vehicle mode only, and a no-op on today's vehicle. Its "test" tokenizer has no EOS
  (`special_eos_id = LLAMA_TOKEN_NULL`, `src/llama-vocab.cpp:2095`), so `common/common.cpp:1321-1323`
  drops the flag with a WARN, and nothing can end the run before the T = 1 decode graph. The flag
  stays for a vehicle regenerated with a tokenizer that has an EOS, which random weights could sample
  first. The Mistral controls continue the digit sequence, and their measured dumps carry the T = 1
  class. The real model runs without it; if it did stop at EOS, the missing T = 1 class makes the
  parser return VOID.
- Graph shapes. The reserve at `-ub 512` supplies the PP graph (T = 512), and the `-n 2` decode
  supplies T = 1. The parser classifies each dump by the size of the token-embedding gather, whose
  `src[0]` is `token_embd.weight`. The gather node itself is unnamed or named `embd`, depending on
  the model.

### Measured so far: the controls (B70, `fefb92980` + `53eb4b700`)

Both controls were run by the lead on 2026-09-27, one at a time. Shmem stayed flat at 1 GiB (the
script then printed GiB) and the journal showed 0 GPU faults.

| run | rc | bench-guard | FLASH_ATTN_EXT | wall |
|---|---|---|---|---|
| control-on (`GGML_SYCL_FLASH_ATTN_EXT=0`, `-fa on`) | 0 | VALID | CPU: 32 nodes in 32 CPU splits, per class | ~5 min (04:39:24 to 04:44:24): CPU FA at load 128 runs at 0.13 tok/s |
| control-off | 0 | VALID | SYCL0: 32, 0 CPU splits | ~4 min (mem.log 04:44:24 to 04:48:34, load ~130) |

With the fixed parser, both logs give the classes `[1, 16, 512]`, and `--n-tokens 1 --n-tokens 512`
reports both. The placement control therefore discriminates both ways. The same logs showed
`token_embd`'s gather on SYCL0 (see the input rows). The control budget is now 600 s, because
both controls ran at 4-5 min against the old 300 s. MemAvailable at their start was 151 and 159 GiB
(162 and 171 GB), just above the 150 GB settle floor.

### The census vehicle

The real model does not fit the host today (next section), so the census runs first on a synthetic
qwen4exp. `test-llama-archs -o` already writes one. Its qwen4exp fixture has:
- n_embd 256 and 2 heads of 128;
- 2 layers: layer 0 GDN (with PLE), layer 1 QSA (`full_attention_interval` 2, `qwen4exp.cpp:133`);
- HC count 4, low rank 8;
- compress ratio 4, indexer top_k 131072, and layer 1's indexer (`blk.1.indexer.{q,k}_proj`,
  `{q,k}_norm`), which the fixture writes as F32;
- 2 experts, both used;
- a "test" tokenizer.

See `tests/test-llama-archs.cpp:250-299,418-459`.

It must then be quantized. The fixture writes F32/F16 weights, and SYCL admits `MUL_MAT_ID` only for
types in the MMID tables (:108150-108168). F32/F16 experts would put every `MUL_MAT_ID` on the CPU,
which the Q8_0 model never does.

```bash
cd /Apps/llama.cpp && source /opt/intel/oneapi/setvars.sh --force
O=/Apps/llama.cpp/census-k6jy; mkdir -p $O/vehicle
# a. write the synthetic model: loads with an empty device list (test-llama-archs.cpp:597-599,953),
#    but it is still a model load, so the lead runs it, pinned
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-llama-archs -a qwen4exp -o $O/vehicle   # -> qwen4exp-moe.gguf
# b. Q8_0 like the real model
./build/bin/llama-quantize $O/vehicle/qwen4exp-moe.gguf $O/vehicle/qwen4exp-moe-q8_0.gguf Q8_0
# b2. the real model's tensor set, from steps a and b (host-only file rewrite, no device). The
#     script's shebang is /usr/bin/python3: `python3` on this host is miniconda, whose numpy
#     cannot load (libmkl_intel_lp64.so.2 missing, even after setvars)
scripts/sycl-qwen4exp-vehicle-rewrite.py $O/vehicle/qwen4exp-moe.gguf \
    $O/vehicle/qwen4exp-moe-q8_0.gguf $O/vehicle/qwen4exp-moe-q8_0-realtensors.gguf
# c. the census, B70 then B50 (budget 600 s each)
QWEN4EXP_VEHICLE=$O/vehicle/qwen4exp-moe-q8_0-realtensors.gguf scripts/sycl-qwen4exp-census-run.sh vehicle 0 $O/vehicle-b70
QWEN4EXP_VEHICLE=$O/vehicle/qwen4exp-moe-q8_0-realtensors.gguf scripts/sycl-qwen4exp-census-run.sh vehicle 1 $O/vehicle-b50
```

Steps a and b were run by the lead on 2026-09-27, both rc 0: `qwen4exp-moe.gguf` is 19242592 B, and
the Q8_0 file is 11464096 B with 98 tensors (36 Q8_0, 6 F16, 56 F32). Tensors whose rows are not a
multiple of 32 (the HC low rank 8) get quantize's fallback type.

Step b2 exists because step b differs from the real model in more than quantize can fix. What
the real model has is read from its own headers, not written down by hand: `--derive` over the six
shards writes `scripts/sycl-qwen4exp-real-tensors.json`. It holds:
- the type of each of the 47 name classes (`blk.N.` folded to `blk.#.`; no class has two types
  across the 1224 tensors);
- the three layer tensor sets: 35 GDN layers, 1 GDN layer with PLE, and 12 QSA layers;
- the six global tensors.

Re-running `--derive` on the shards reproduces the committed file byte for byte. Diffed against
that map, the lead's step-b file differs in four ways. An earlier version of this section named only
the first two, and the vehicle carried the other two into a census run.

1. **A type quantize cannot produce.** The real model's indexer projections are BF16.
   `src/llama-quant.cpp:327-329` exempts `indexer.k_proj.weight` and `indexer.q_proj.weight` from
   quantization before any type is chosen, so a `--tensor-type 'indexer\.[qk]_proj=bf16'` override
   (this document's first version of step b) matches the names and changes nothing: the lead's run
   left both F32, with no BF16 conversion in the log. The script converts step a's F32 to BF16.
2. **The fused expert gate and up.** The fixture writes a fused `blk.*.ffn_gate_up_exps.weight`
   [256, 768, 2]; the real model has separate `ffn_gate_exps` and `ffn_up_exps`. The two reach
   different `MUL_MAT_ID` paths (detail below).
3. **Types quantize changed that the real model keeps.** `ple_norm_{key,query,conv}` became Q8_0
   (the lead's quantize log, entries [38/98]-[40/98] at lines 214-216) and `ple_conv1d` became F16
   (entry [36/98] at line 212, after the "ncols 4 not divisible by 32" fallback warning at
   line 174), where the real model has F32. The fused run's sched dump shows the grouped PLE norm's `MUL` on SYCL0 with src1
   `blk.0.ple_norm_key.w`. So the "RMS_NORM + MUL (grouped) x3 | f32" row would have scored a
   Q8_0-weight path the real model never takes. The reviewer's inferred out-of-bounds read on that
   path is llama.cpp-1z69, not this census's. The script copies step a's F32 tensors unchanged.
4. **Tensors the real model lacks.** These are every `.scale` and `.input_scale` on the attention,
   shared-expert, expert and SSM weights, and `attn_{q,k,v}.bias`: 37 on the vehicle.
   - They are not inert. `ffn_down_exps.scale` alone adds a REPEAT, GET_ROWS and MUL after the down
     `MUL_MAT_ID` (`build_lora_mm_id`, `src/llama-graph.cpp:1564-1570`, passed from
     `src/models/qwen4exp.cpp:1004`).
   - The loader treats every one as optional. The `.scale` and `.input_scale` tensors are created
     `TENSOR_NOT_REQUIRED` in its generic pass after `load_arch_tensors`
     (`src/llama-model.cpp:2366-2499`). The three biases are `TENSOR_NOT_REQUIRED` in
     `create_tensor_qkv`'s separate-weights branch (`llama-model.cpp:4302-4304`), which
     `qwen4exp.cpp:218` reaches.
   - The script drops exactly those: a pattern for the two suffixes and the three biases. It refuses
     any other tensor the real model lacks, since the loader may require it.

One delta cannot be closed, and it is the only entry in the script's allowlist: `hc_attn_up`,
`hc_ffn_up` and `output_hc_up` are Q8_0 in the real model and F16 in the vehicle. Their ne0 is the
HC low rank, 320 in the real model and 8 in the fixture, and Q8_0 needs ne0 % 32 == 0, so quantize
falls back to F16. The allowance is bound to that reason: both the rewrite and `--verify` grant F16
only while the real type's block size (from gguf-py's quant table) does not divide the tensor's
ne0, so an F16 `hc_attn_up` with ne0 = 32 is refused. What this delta costs the census is in the
table under "What the vehicle exercises".

The expert layout in detail. The real model's separate gate and up are from the lead's gguf-py read
of shards 4-6 (2026-09-27), confirmed by the derived map. The first vehicle runs (2026-09-27, on a
step-b2 file that converted only the indexer) were refused on both cards at the first prompt, with
the same lines on each:

```
[MOE-PROMPT-REFUSAL] tensor=blk.0.ffn_gate_up_exps.weight occurrence=0 expert=0 reason=recipe_missing source_reason=0 recipe_reason=local-layout-unsupported
graph_compute: ggml_backend_sched_graph_compute_async failed with error -1
```

Both exited rc 3 and scored nothing about the real model. The fused path's gap is llama.cpp-zfbl and
is out of this census's scope, because qwen4exp as shipped never takes it.
- **The split.** The script splits each fused tensor, in its place, into `ffn_gate_exps` and
  `ffn_up_exps`, [256, 384, 2] Q8_0.
- **The order.** It is the one `build_moe_ffn` views the fused result in
  (`src/llama-graph.cpp:2195-2198`): gate is the view at offset 0 and up the view at `n_ff * nb[0]`.
  So of each expert's 2 n_ff weight rows, [0, n_ff) are gate and [n_ff, 2 n_ff) are up. Q8_0 blocks
  run along ne0, so each half is whole rows, copied unchanged.
- **The loader.** qwen4exp loads the separate form: `src/models/qwen4exp.cpp:252` calls
  `create_tensor_gate_up_exps`. That creates the fused tensor `TENSOR_NOT_REQUIRED` and, when it is
  absent, both separate tensors as required (`src/llama-model.cpp:4262-4266`).

Whether the separate form clears the prompt `MUL_MAT_ID` is measured only by step c.

Run on the lead's step-a and step-b files, the script gave 7214336 B with 63 tensors: 35 Q8_0,
21 F32, 5 F16 (the allowlisted HC tensors) and 2 BF16.
- The four PLE tensors are byte-equal to step a's.
- The two projections are BF16, with a relative error of at most 3.9e-3 (bf16 rounding).
- On both layers, gate and up are byte-equal to the two halves of each expert's fused rows.
- The 37 optional tensors are gone.
- The rest, and all 162 metadata fields, are byte-identical to step b and in step b's order.

Quantize reorders tensors, so the script pairs step a with step b by the set of names and shapes.
It copies the metadata as raw bytes because the fixture has empty arrays (`tokenizer.ggml.merges`,
`classifier.output_labels`) that gguf-py's writer refuses. Whether the loader accepts the rewritten
file is measured only by step c.

The script plans every output tensor from the headers alone and diffs the plan against the map
before it reads any tensor data. So step a's all-F32 file given as step b, a step b not quantized
from step a, and an input with no indexer are refused before a byte is copied. It refuses an output
that is either input, by path or through a symlink, and writes through a temp file beside the output
plus a rename, so a failure mid-write leaves any existing output intact.

Step c cannot be pointed at the wrong file by mistake. Vehicle mode runs `--verify` first, which is
the same diff on the finished file, and exits 1 before any device work unless all of these hold:
- every tensor is in the real model;
- each has the real type, or the allowlisted one;
- every layer's tensor set is one of the real model's three;
- the global tensors are the real model's.

The sched dump prints no types, and nothing marks a tensor that should not be there. Without that
check:
- an F32 indexer `MUL_MAT` on SYCL0 would score as agreeing with IDX-PROJ-BF16;
- F32 experts would agree with MOE-MMID's "on Q8_0";
- a fused gate_up, a Q8_0 PLE norm or an expert scale's extra nodes would each score a path the real
  model never takes.

`--verify` refuses both earlier vehicles: the indexer-only file on 45 problems, and the `bb2cea44d`
file (`qwen4exp-moe-q8_0-realshape.gguf`) on 41 (the 37 extra tensors and the four PLE types). The
script's gate is `tests/test-sycl-qwen4exp-vehicle-rewrite.py` (see the gates at the end of the
lead-run section).

The fixture's `qwen4exp.attention.indexer.types = 0` does not mean "no indexer". qwen4exp never
reads that key (`src/models/qwen4exp.cpp:56-61` reads head count, key length and top_k); only the
DOTS3NOTE and HY_V4 fixtures set it (`tests/test-llama-archs.cpp:381,485`). The indexer tensors are
created on every non-recurrent layer (`qwen4exp.cpp:225-228`), and `blk.1.indexer.*` is in the file.

The structural controls for the vehicle are `GATED_DELTA_NET=ANY:1`, `MUL_MAT_ID=ANY:6` and
`TOP_K=ANY:1`. Layer 1 emits one TOP_K per graph: QSA runs when the indexer cache exists and the
layer's compress ratio is nonzero (`qwen4exp.cpp:786-788`); the cache exists because
`indexer_head_size` is 128, which sets the index-cache filter for non-recurrent layers
(`src/llama-model.cpp:3581-3586`); layer 1 is not recurrent (`recurrent_layers [1, 0]`) and its
ratio is 4. qwen4exp sets no SWA, so the model gets `llama_memory_hybrid_idx`, not the iswa variant
(`llama-model.cpp:3589,3609`). `ggml_top_k` builds a `GGML_OP_TOP_K` node (`ggml/src/ggml.c:5526`).
That is conditional, not unconditional: without the indexer cache or with a zero ratio, layer 1 runs
dense attention and emits no TOP_K. The control discriminates only because nothing else in the graph
emits `GGML_OP_TOP_K`. `qwen4exp.cpp:684` is the model's only `ggml_top_k`, and the MoE router
selects experts with `ggml_argsort_top_k` (`src/llama-graph.cpp:2121`), which is an `ARGSORT` plus a
view (`ggml.c:5500-5511`), not a TOP_K node. The same holds for the qwen4exp mode's `TOP_K=ANY:12`.
If the router ever switches to `ggml_top_k`, both controls pass with no QSA at all. Backend sampling
would do the same through `llama-sampler.cpp:1603`. It is off by default (`common/common.h:297`) and
the script does not pass `--backend-sampling`, but the option also reads
`LLAMA_ARG_BACKEND_SAMPLING` from the environment (`set_env`, `common/arg.cpp:2326`), so the script
launches with `env -u LLAMA_ARG_BACKEND_SAMPLING`. The `MUL_MAT_ID` minimum is 6: gate, up and
down on each of the two MoE layers, as the real model's 144 is 48 x 3. A fused gate_up would give 4,
so the control refuses that layout independently of `--verify`.

What the vehicle exercises, per prediction rule:

| rule(s) | exercised? | why |
|---|---|---|
| HC-PRE-FUSED, HC-POST-FUSED | yes, 4 + 4 nodes | HC count 4; the fused ops and the CPU-landing exemption are shape-independent |
| GDN-SOFTPLUS, GDN-CORE, GDN-SSM-CONV, STATE-CONCAT | yes, 1 layer | S_v = 128 as in the real model, but H_k = H_v = 2. **The H_k 16 vs H_v 48 broadcast is not exercised.** |
| QSA-TOPK | yes | k = min(n_kv, 131072 + 3) = n_kv, and n_kv >= min(kv_size, 256) = 256 at `-c 4096` (`src/llama-kv-cache.cpp:1386-1391`), so k > 32: the same decline as the real k = 2051 |
| QSA-FILL, QSA-SET-ROWS, ROPE | yes | same graph, smaller shapes |
| QSA-FA | **different kernel shape** | head dim 128, not 256 |
| IDX-PROJ-BF16 | yes, 2 nodes, on the step b2 file | quantize leaves the indexer F32 (`llama-quant.cpp:327-329`); b2 rewrites it to BF16. On the plain Q8_0 file the two `MUL_MAT`s are F32 and say nothing about BF16 |
| MOE-MMID, MOE-ARGSORT, MOE-SOFTMAX | yes, on Q8_0, separate gate and up (step b2) | 2 experts, all used; K = 256/384, not 2560/640. On the fixture's fused gate_up, `MUL_MAT_ID` takes a path the real model never does, and SYCL refuses it at the first prompt (llama.cpp-zfbl, out of scope) |
| MUL-MAT on `hc_*_up` (the `MUL_MAT hc_*_up` row, and the rule's "K=320 hc_up") | **no, different type and K** | the one allowlisted delta: the vehicle's `hc_attn_up`, `hc_ffn_up` and `output_hc_up` are F16 [8, 1024], so their `MUL_MAT` is f16 x f32 with K = 8. It says nothing about q8_0 x f32 at K = 320 |
| MOE-GLU (PLACEMENT) | **no** | the vehicle's experts fit in VRAM: no host-planned layer, no CpuExpertPool |
| IN-TOK-EMBD, IN-PLE-GATHER | yes | same input buft list; PLE on layer 0 |
| every `count` | **no** | counts are for 48 layers; the vehicle has the per-layer multiples |
| placement under memory pressure | **no** | nothing the 120 GiB of host experts does to the split graph shows up at 2 layers |

So the vehicle can confirm or refute every capability verdict (the gaps and the suspect
admissions) except the `hc_*_up` `MUL_MAT`'s, which it runs at F16 and K = 8 rather than Q8_0 and
K = 320. It cannot confirm the full-model counts, the placement rows, or the GDN head broadcast.

### The real model: blocked on host memory

The qwen4exp mode needs 215 GB (200 GiB) MemAvailable. The lead measured 151-159 GiB (162-171 GB)
under the host's permanent load, so the script's settle step times out and exits 3. The cause is
that SYCL leaves `caps.mmap_support` unset. The caps aggregate at :107798-107803 never names it, and
`src/llama-model.cpp:2245-2265` then turns off both `use_mmap` and AUTO lazy mode. The whole 177 GiB
is therefore read into host buffers.

The full-model census waits on one of these:
- the storage-tier lanes, which would place the 50.66 GiB PLE table on SSD (`llama.cpp-th32`,
  `llama.cpp-9g22`);
- a decision on SYCL mmap support. **`llama.cpp-5efe`** ("SYCL: mmap_support=0 makes lazy model
  loading AUTO resolve to OFF", open, now P2) is that ticket; as filed it covered lazy mode only.

The lead commented on 5efe on 2026-09-27 (tracker comment `c-yuc0`; read it there, this is a summary,
not a quote):
- The scope widens: the same unset `mmap_support` also turns off `use_mmap` under
  `LLAMA_LOAD_MODE_AUTO` (`src/llama-model.cpp:2245-2253`), not only AUTO lazy mode (`:2255-2265`),
  so every SYCL load reads the whole file into host buffers. Qwen3.8-Flash-Next Q8_0 (177 GiB) needs
  about 175 GiB resident against 151-159 GiB (162-171 GB) MemAvailable, so this census cannot run
  on the real model. (The comment's figures were first posted as "GB" and corrected to GiB.)
- The ticket moves toward P2 (it is P2 now). The decision is one of: SYCL advertises mmap and the
  unified cache adopts mmap-backed host tiers; the storage tier (`llama.cpp-th32`) owns those bytes;
  or `-lzm on` becomes the documented requirement. Whichever holds, the unified cache must own the
  bytes (P1), and `llama.cpp-qptd` (`-ngl 0` loses mmap) has the same root.

If the qwen4exp run does start (after mmap, or on a quieter host): Shmem may legitimately exceed
CLAUDE.md's ~100 GB abort line when the host experts are USM-backed. The MemAvailable watchdog, which
kills by session, stands in for that line on this run.

### Order and commands

```bash
cd /Apps/llama.cpp && source /opt/intel/oneapi/setvars.sh --force
O=/Apps/llama.cpp/census-k6jy     # not /tmp: tmpfs does not survive a reboot
# 0. controls: DONE on the B70 (above); repeat only after a parser or script change
scripts/sycl-qwen4exp-census-run.sh control-on  0 $O/control-on-b70    # --require FLASH_ATTN_EXT=CPU:32
scripts/sycl-qwen4exp-census-run.sh control-off 0 $O/control-off-b70   # --require FLASH_ATTN_EXT=SYCL:32
# 1. the vehicle (steps a-c above), B70 then B50
# 2. the real model, only once it fits -- detached, the first 177 GiB load can pass the 10-min tool ceiling
setsid nohup scripts/sycl-qwen4exp-census-run.sh qwen4exp 0 $O/qwen-b70 > $O/qwen-b70.log 2>&1 &
#    poll $O/qwen-b70.log for "parser rc=" (or an exit-3 message); the next run settles by itself
setsid nohup scripts/sycl-qwen4exp-census-run.sh qwen4exp 1 $O/qwen-b50 > $O/qwen-b50.log 2>&1 &
```

Expected runtime:
- The controls, about 4-5 min each under the host's load (control-on runs flash attention on the CPU).
- The vehicle: unmeasured. Its graph is tiny, but the controls show that startup under this load costs minutes, so its budget is 600 s like theirs.
- The real model: the load dominates and has never been measured here. 177 GiB on bcachefs over
  NVMe, SATA and a USB SSD, so expect 5-15 min cold. The budget is 1800 s.

How to score a run:

1. **Exit 3 is void.** The run was refused, killed, failed, or bench-guard stamped it SUSPECT. For
   SUSPECT, check the GPU (`journalctl -k --since "1 hour ago" --no-pager | grep -iE 'GT reset|guc_id|CAT error'`)
   before starting any other run.
2. Both controls must exit 0. If control-on exits 2, the parser or the log settings are broken, and
   any census taken with that setup is void.
3. A census exits 0 only if both T classes are present and classifiable, and its structural
   controls hold. Exit 2 means the dump is missing, truncated or unclassifiable. It never means "no
   gaps".
4. `census.md` holds, for T = 1 and T = 512:
   - the per-op backend table;
   - the per-rule AGREE/DISAGREE/ABSENT diff against `scripts/sycl-qwen4exp-op-prediction.json`;
   - every unpredicted CPU node.

   Copy the backend and split counts into the B70/B50 columns above. For the vehicle, record them
   with the vehicle table's caveats.
5. Cross-check `run.err` for the HC WARN line quoted above. It is present at default verbosity, and
   its count is fused nodes (vehicle 4, real model 96).

The parser alone:

```bash
python3 scripts/parse-sycl-sched-census.py run.err --prediction scripts/sycl-qwen4exp-op-prediction.json \
    --n-embd <2560|256|4096> --n-tokens 1 --n-tokens 512 [--tensor-types <"T name dims type" listing>] \
    [--require OP=CPU|SYCL|ANY[:MIN]] [--strict]
```

Its gate is `tests/test-sycl-sched-census.py` (ctest `test-sycl-sched-census`, pure Python). It works
as follows:
- It plants CPU splits in a synthetic dump. The dump is written one `GGML_LOG_DEBUG` call at a time
  with the printer's exact formats, a torn line, and an unnamed token gather.
- Every check runs once per prefix form: none, the default timestamped one, and bare `D `.
- It requires the counts to come out as planted, and requires `src[0]` to survive on every node.
- It requires an empty, headers-only or unclassifiable log to be VOID.
- It parses the golden excerpt of the real control-on capture into the classes [512, 1].
- It checks that every prediction rule names an op ggml prints.

The parser at `53eb4b700` fails it with 25 assertions (re-run 2026-09-27: 1 in the unprefixed
mode, 10 each in the timestamp and bare-prefix modes, 4 on the golden excerpt), among them
"[timestamp] n_tokens classes: expected [512, 1], got [None, None]" and the same on the golden
excerpt.

The step-b2 script's gate is `tests/test-sycl-qwen4exp-vehicle-rewrite.py` (ctest
`test-sycl-qwen4exp-vehicle-rewrite`). It is standard-library Python. It runs the script through its
`/usr/bin/python3` shebang, and exits 77 (skip) when that interpreter is missing or cannot import
gguf-py, which needs numpy and yaml.

It plants tiny GGUFs: a two-shard "real model", a step-a file (all F32) and a step-b file (quantized
and reordered). Expert rows are 64 wide, two Q8_0 blocks, so a one-block row size fails. The test
derives its own map through `--derive` and requires:
- **`--derive`** to give the expected map, and to refuse a class with two types.
- **The rewrite** to give, in step b's order and with the metadata unchanged, each tensor's real type
  from the step that has it:
  - step b's bytes;
  - step a's F32 bytes where quantize changed an F32 tensor;
  - BF16 bit-exact to ggml's round-to-nearest-even (ties both ways, NaN, inf, -0, a subnormal, max
    finite);
  - the allowlisted F16.

  It must also split the fused tensor into gate and up, byte for byte rows [0, n_ff) and
  [n_ff, 2 n_ff) of each expert's rows (every row distinct); drop the optional tensors; and create
  the output with the umask's permissions.
- **Rewrite refusals:** rc 1, no output and untouched inputs, each for its own reason, for:
  - an output that is either input (same path or symlink);
  - a step b not quantized from step a;
  - an all-F32 step b;
  - no indexer;
  - no gate and up;
  - an extra tensor the loader may require.
- **`--verify`** to accept the result and refuse, each for its own reason:
  - the step-b file;
  - Q8_0 or F16 where the real model has F32;
  - F32 experts;
  - an F32 indexer;
  - an F32 global tensor (`token_embd`);
  - F16 outside the allowlist;
  - F16 in the allowlist where ne0 = 32 lets Q8_0 hold it;
  - an extra `.scale`;
  - the fused gate_up in both layers, and in the second layer only;
  - a missing global tensor.
- **Mid-write:** an existing output to survive a write that fails partway (a file-size limit, as
  ENOSPC would).
- **Early refusal:** a wrong input to be refused before its data is read. With a 256 MiB sparse
  tensor ahead of the offending one, the script's peak RSS stays under half of it.
- **The committed map** to name six shards and give a type for every tensor in its layer sets.

Earlier versions of this gate:
- The script at `f6f0746b9` failed its first version with 9 assertions: the same-path and symlink
  overwrites (2 each), the all-F32 file accepted (2), F32 experts accepted by `--verify`, the
  existing output destroyed mid-write, and the early refusal (peak RSS 558484 KiB against
  34680 KiB after the fix).
- The script at `9bec100c4` failed the second version with 12 assertions: no split, the
  already-split file and the file with neither form accepted, and `--verify` accepting the fused
  gate_up.
- The `bb2cea44d` script has no `--derive` and takes two files, so it fails this version at its
  first steps.

This version is shown to discriminate by mutants of the current script, each run in a git-archive
extract. Every one fails it (the last three were survivors of the previous version):
- gate and up swapped at the call;
- a contiguous gate-then-up layout with no expert stride;
- a wrong expert stride;
- a one-block (34 B) row;
- a diff that checks only `blk.0`;
- no restore from step a;
- `--verify` ignoring extra tensors;
- no allowlist;
- no chmod;
- dropping any tensor;
- BF16 by truncation;
- no layer-set check;
- no type check on global tensors;
- `--derive` letting the first type seen win;
- the F16 allowance granted whatever ne0 is.

## Proposed closure tickets (not filed: the lead files them once the census confirms)

**P1. sycl: GGML_UNARY_OP_SOFTPLUS kernel + supports_op case.**
- Scope: an elementwise softplus in the unary family (f32, and f16 under `GGML_SYCL_F16`),
  admitted in the UNARY switch (:108205).
- RED: `test-backend-ops -b SYCL0 -o SOFTPLUS` (pinned selector, lead-run) reports the `test_unary`
  SOFTPLUS cases "not supported". SOFTPLUS is in `test_unary`'s op list (`test-backend-ops.cpp:2084`),
  and the `test_unary_mul` SOFTPLUS cases (`:9552`) decline too. The census shows `GDN-SOFTPLUS` on
  CPU (36 nodes, or 1 on the vehicle).
- GREEN: the cases pass against CPU at the default NMSE <= 1e-7, and the census rule flips to
  DISAGREE (0 CPU nodes). That removes 36 islands, about 72 splits per token.

**P2. sycl: TOP_K for k > 32 (up to 2051, over n_kv up to the context size).**
- Scope: a top-k whose shared memory does not scale with k, for example a radix select or a
  bitonic partial sort per row. Indices are unordered, matching the CPU semantics
  (`ggml.h`, "no particular order").
- RED: the existing cases with k in {100, 500, 1023, 9999} over ne0 = 2^i (+11)
  (`test-backend-ops.cpp:11144-11151`) report "not supported" on SYCL0. The ticket adds the
  qwen4exp shapes, which do not exist yet: `test_top_k(F32, {4096, 1, 1, 1}, 2051)` and
  `{4096, 512, 1, 1}, 2051`. The census shows `QSA-TOPK` on CPU.
- GREEN: all of those pass, and QSA-TOPK moves to SYCL0.

**P3. sycl: DSV4_HC_PRE (gated and ungated), DSV4_HC_POST and DSV4_HC_COMB kernels.**
- Scope: the three hyper-connection ops that upstream implements on CPU (`ggml-cpu.c:2100-2110`),
  shared with DeepSeek-V4.
- RED: the existing cases decline on SYCL0 (`test-backend-ops.cpp:9586-9612`):
  - `test_dsv4_hc_comb` x4 and a loop;
  - `test_dsv4_hc_pre`, gated and ungated, n_embd up to 4096, n_hc {2..} loop;
  - `test_dsv4_hc_post`, identity and not.

  The ticket adds the qwen4exp shape, `test_dsv4_hc_pre(2560, 4, {1, 512}, gated = true)`. The
  census shows 96 + 96 CPU nodes (vehicle 4 + 4).
- GREEN: these pass, and the HC WARN line reads "enabled" with no CPU landings. This is the biggest
  item: 192 islands, about 384 splits per token.

**P4. llama-context + sycl: tell a capability decline from a placement decline, from one source.**
- Problem: `resolve_fused_ops` (`src/llama-context.cpp:1242-1263`) treats every CPU landing as
  placement, and prints "not a capability gap" even for HC, where it is one. Asking
  `ggml_backend_dev_supports_op(device_layer, node)` cannot fix this, because this fork's
  supports_op also declines for placement:
  - the KV-host residency check (:108067-108131), which declines FA over demoted KV unless
    ATTN_HOST_DISPATCH is set;
  - `ggml_sycl_op_is_planned_on_host` (:108185-108192).

  For tiered-KV FA, that check would disable FA globally. That brings back `llama.cpp-7nzm` (the
  17.3 GB compute buffer, `:1247-1252`) and overrides "placement decides the executor".
- Scope: the backend is the only party that knows why it declined, so the reason comes from it:
  - Split SYCL's supports_op into a capability predicate and a placement predicate. The public
    supports_op stays their conjunction.
  - Expose the capability half through the registry's `get_proc_address`, for example
    `"ggml_backend_sycl_dev_supports_op_capability"`. No public ggml API change.
  - `resolve_fused_ops` looks the function up on the layer's device. It exempts a CPU landing only
    when the capability answer is yes (so the decline was placement). A capability no disables the
    fusion, which is upstream's behaviour, and prints upstream's "not supported" wording.
  - A backend that does not export the function keeps today's behaviour.
- RED:
  1. On the vehicle, the WARN reads "executes on CPU ... not a capability gap" for HC pre/post, and
     the census shows `DSV4_HC_PR` on CPU.
  2. With the fix, HC resolves "not supported, set to disabled", and the unfused chain appears on
     SYCL0.
  3. A tiered-KV run keeps FA enabled. Use the GPT-OSS n_ctx 131072 configuration from 7nzm and
     measure the compute buffer: it must not regrow to 17.3 GB.
- **Accepted** (lead ruling, 2026-09-27). P3 makes it moot for HC, but not for the next fused op.

**P5. sycl: GATED_DELTA_NET honesty, prefill correctness and PP throughput.**
- (a) supports_op mirrors the kernel's real envelope: S_v in {16, 32, 64, 128}, and contiguous g,
  beta and state. RED: a GDN case with head_size 256 aborts in the kernel
  (`gated_delta_net.cpp:251`) instead of reporting "not supported". The case is
  `test_gated_delta_net(F32, 4, 256, 1, 1)` and does not exist yet.
- (b) Correctness at the qwen4exp shape. The ticket adds
  `test_gated_delta_net(F32, head_count=16, head_size=128, n_seq_tokens={1, 512}, n_seqs=1, v_repeat=3)`,
  which gives H_k 16 and H_v 48. Tolerance is the test's default NMSE <= 1e-7
  (`test-backend-ops.cpp:1217`). RED: the pre-registered prediction is that T=1 passes and T=512
  fails. That follows from `llama.cpp-30h4`: Qwen3.6-27B is garbage in prefill only, PPL 473319. If
  both pass, GDN is exonerated for this shape, and 30h4's cause lies elsewhere (host-tiered dense
  weights).
- (c) A chunked PP kernel (`gated_delta_net.cpp:193`). This is throughput only; RED/GREEN is PP512
  tok/s on the vehicle and then the real model.

**P6. sycl: FILL supports_op honesty.**
- Scope: admit only a contiguous dst of type F32 or F16, matching `fill.cpp:17,48`.
- RED: a `test_fill` with a BF16 or non-contiguous dst (the existing cases, `test-backend-ops.cpp:11326-11329`,
  are all F32 contiguous) aborts instead of reporting "not supported".
- Small, and not a qwen4exp blocker: its FILLs are contiguous F16/F32.

**P7. sycl: the BF16 weight layout is decided at dispatch, not by the planner.**
- The ownership half is already right: `g_sycl_bf16_materialize_cache` holds
  `ggml_sycl::mem_handle` values (:14177).
- The defect is the one-fact-two-sources shape. Whether the indexer projections run as F32 is
  decided at first dispatch (:108305-108325), not by the planner and unified cache that materialized
  the operand. "The loaded layout is the answer" requires the planner to materialize the F32 layout
  (or a BF16 kernel to consume BF16), and dispatch to read the loaded layout.
- RED: on the vehicle (the step b2 file, BF16 indexer) or the real model, the census shows the indexer
  `MUL_MAT` on SYCL0 while the planner's layout record for those weights is BF16/AOS and a
  materialization entry appears at first dispatch. The observable is a trace of that cache, not the
  census.

**P8. The input-layer gathers run on SYCL0 over a SYCL_Host buffer. Suspect; if verified zero-copy, not sanctioned.**
- `make_cpu_buft_list` puts the first GPU's host buffer type ahead of the CPU buffer
  (`src/llama-model.cpp:1765-1777`). The input layer (`:2338`) therefore lands `token_embd`, and on
  qwen4exp the 50.66 GiB `per_layer_token_embd`, in pinned host memory, and SYCL0 runs the gathers.
  Measured for `token_embd` on the Mistral control.
- **Ruling (2026-09-27): the existing owner rule applies, with no exception.** Placement decides the
  executor, and a GPU gather over SYCL_Host-pinned `token_embd` or PLE, if that is what executes, is
  a zero-copy read of host memory. A small T-row gather gets no carve-out for moving few bytes. The
  `[SYCL0]` label is the scheduler's backend assignment, not proof of the executor.
- `llama.cpp-lah5` first verifies which executor actually runs the gathers, then moves them to
  ggml-cpu. The vehicle census plus a `GGML_SYCL_DEBUG` trace of the PLE gather give the evidence;
  after the fix, IN-TOK-EMBD and IN-PLE-GATHER expect CPU.
- The gathered rows, T x n_embd per layer, then cross to the device as an ordinary activation copy
  at the split boundary. That is not weight streaming: no weight moves, only the op's output.
- Where the PLE table's bytes live, RAM or SSD, is not part of this ticket. That stays the storage
  planner's decision (`llama.cpp-th32`); lah5 only moves the executor to where the bytes already are.

Not a ticket: MoE `SWIGLU` on the CPU for host-expert layers (:109207), and the other
`planned_on_host` declines listed in the summary, are placement deciding the executor. If the census
shows them, record the layer count, because it bounds the host-side work per token.

## Principles check for the proposals

- **P1-P3, P5 and P6** close support gaps, or make advertising honest, for layouts the loaded
  operands already have. They add no allocator, no zero-copy, no weight streaming and no host wait.
- **P4** gives "why did this backend decline" one source, the backend, and leaves placement
  declines exempt. The first version asked supports_op, which would have overridden placement.
- **P7** moves a layout decision from dispatch to the planner, so the layout has one source.
- **P8** removes a GPU zero-copy read of host memory, once lah5 verifies that one executes: placement
  decides the executor, and the input gathers go to ggml-cpu.
- Today's CPU islands already cost a device-host round trip each. That is the per-token
  synchronization P1-P3 remove.
