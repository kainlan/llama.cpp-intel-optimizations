# Path-Scoped Zone Sizing — Measured Findings

Ground-truth measurements produced by the tasks in
`docs/plans/2026-07-25-sycl-path-scoped-zone-sizing.md`. No task in that plan may
quote a byte figure that does not appear in this document.

---

## Task 1 — Inventory

**Instrument:** `[SYCL-PLAN] inventory top-8` line added to
`populate_host_zone_sizing` (`ggml/src/ggml-sycl/unified-cache.cpp`), printed once
per plan from the same `tensor_inventory` that feeds `plan.max_tensor_bytes`.
Observation only — no sizing arithmetic changed.

**Capture command** (B50, `level_zero:1`):

```bash
timeout 900 env ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/<model>.gguf -p 0 -n 4 -r 1 -v 2>&1 | grep '\[SYCL-PLAN\]'
```

> `-v` is **required**. `llama-bench` installs a null log callback, so every
> `GGML_LOG_INFO` — including all pre-existing `[SYCL-PLAN]` lines — is silently
> discarded without it. The plan's capture command omitted `-v` and yields zero
> `[SYCL-PLAN]` lines of any kind, which is indistinguishable from "the log line
> was never added."

Artifacts: `/tmp/zone-sizing/` (not committed).

### RED state

Against the pre-change build at `57db1e693`:

```
$ cat /tmp/zone-sizing/before.txt | grep -c 'inventory top'
0
```

Eleven other `[SYCL-PLAN]` lines were present in the same capture, so the zero is
absence of this information specifically, not absence of plan logging.

### GPT-OSS 20B MXFP4

459 tensors, inventory total **11536.2 MB**.

| # | Tensor | MB | type | printed shape |
|---|---|---:|---|---|
| 1 | `output.weight`            | 586.8 | 8 (Q8_0)  | 2880 × 201088 |
| 2 | `token_embd.weight`        | 586.8 | 8 (Q8_0)  | 2880 × 201088 |
| 3 | `blk.0.ffn_up_exps.weight`   | 134.5 | 39 (MXFP4) | 2880 × 2880 |
| 4 | `blk.0.ffn_gate_exps.weight` | 134.5 | 39 (MXFP4) | 2880 × 2880 |
| 5 | `blk.1.ffn_down_exps.weight` | 134.5 | 39 (MXFP4) | 2880 × 2880 |
| 6 | `blk.1.ffn_up_exps.weight`   | 134.5 | 39 (MXFP4) | 2880 × 2880 |
| 7 | `blk.1.ffn_gate_exps.weight` | 134.5 | 39 (MXFP4) | 2880 × 2880 |
| 8 | `blk.0.ffn_down_exps.weight` | 134.5 | 39 (MXFP4) | 2880 × 2880 |

Ranks 3–8 are a tie at 134.5 MB; `std::partial_sort` is not stable, so their
order varies between runs. The set is stable.

### Mistral 7B v0.1 Q4_0

291 tensors, inventory total **3917.9 MB**.

| # | Tensor | MB | type | printed shape |
|---|---|---:|---|---|
| 1 | `output.weight`         | 102.5 | 14 (Q6_K) | 4096 × 32000 |
| 2 | `token_embd.weight`     |  70.3 | 2 (Q4_0)  | 4096 × 32000 |
| 3 | `blk.0.ffn_gate.weight` |  31.5 | 2 (Q4_0)  | 4096 × 14336 |
| 4 | `blk.0.ffn_down.weight` |  31.5 | 2 (Q4_0)  | 14336 × 4096 |
| 5 | `blk.0.ffn_up.weight`   |  31.5 | 2 (Q4_0)  | 4096 × 14336 |
| 6 | `blk.1.ffn_up.weight`   |  31.5 | 2 (Q4_0)  | 4096 × 14336 |
| 7 | `blk.1.ffn_gate.weight` |  31.5 | 2 (Q4_0)  | 4096 × 14336 |
| 8 | `blk.1.ffn_down.weight` |  31.5 | 2 (Q4_0)  | 14336 × 4096 |

Ranks 3–8 are again a tie at 31.5 MB with unstable ordering.

### Headline figures

```
MAX TENSOR (gpt-oss-20b-mxfp4): output.weight at 586.8 MB
MAX TENSOR (mistral-7b-Q4_0):   output.weight at 102.5 MB
```

`token_embd.weight` ties the maximum exactly on GPT-OSS (same shape, same Q8_0
type) and is second on Mistral (70.3 MB, Q4_0 rather than Q6_K).

**The 586.8 MiB assumption the plan flagged as unverified is confirmed correct**,
but for a different reason than the plan's arithmetic assumed. The GPT-OSS
201088 × 2880 embedding and output tensors are **Q8_0, not MXFP4**:
579,133,440 elements ÷ 32 × 34 B = 615,329,280 B = 586.82 MiB. The plan's
"roughly half" estimate came from applying the MXFP4 block size (17 B) to a
tensor that is not MXFP4.

### IMPLIED ZONE COST at HEAD

Multipliers read from `unified-cache.cpp` at HEAD, all applied to
`plan.max_tensor_bytes`:

| Consumer | Multiplier | GPT-OSS 20B | Mistral 7B |
|---|---:|---:|---:|
| `onednn_reorder_bytes`    | 1 × |  586.8 MB | 102.5 MB |
| `cpu_quant_buffer_bytes`  | 3 × | 1760.4 MB | 307.6 MB |
| `dma_staging_pool_bytes`  | 2 × | 1173.6 MB | 205.1 MB |
| `onednn_scratchpad_bytes` | 2 × | 1173.6 MB | 205.1 MB |
| **Total (4 repointed consumers)** | **8 ×** | **4694.4 MB** | **820.2 MB** |

```
IMPLIED ZONE COST at HEAD: onednn_scratchpad=1173.6 MB, cpu_quant=1760.4 MB,
dma_staging=1173.6 MB, onednn_reorder=586.8 MB.   [gpt-oss-20b-mxfp4]
IMPLIED ZONE COST at HEAD: onednn_scratchpad=205.1 MB, cpu_quant=307.6 MB,
dma_staging=205.1 MB, onednn_reorder=102.5 MB.    [mistral-7b-Q4_0]
```

The `dma_staging_pool_bytes` figures are corroborated directly by the pre-existing
log line in both captures: `DMA staging pool: 1173.6 MB (2 x max_tensor 586.8 MB)`
and `DMA staging pool: 205.1 MB (2 x max_tensor 102.5 MB)`.

Two further consumers are deliberately left on the global maximum by the plan and
are excluded from the table above: `moe_q8_workspace_bytes`
(`max_tensor_bytes / n_experts × 1.1 × n_expert_used`) and `s1_per_inflight_bytes`
(`max(max_tensor_bytes × 2, max_staging_pair_bytes) + 2 MB`).

### Headroom available to the path-scoped predicates

If a predicate excludes `output.weight` and `token_embd.weight`, the next
inventory maximum is the largest FFN tensor:

| Model | Global max | Max excl. embed/output | Ratio | 8 × cost at HEAD | 8 × cost if narrowed |
|---|---:|---:|---:|---:|---:|
| GPT-OSS 20B | 586.8 MB | 134.5 MB | 4.36 × | 4694.4 MB | 1076.0 MB |
| Mistral 7B  | 102.5 MB |  31.5 MB | 3.25 × |  820.2 MB |  252.0 MB |

These are arithmetic implications of the measured inventory, **not measured
reclaim**. Task 8 measures actual reclaimed VRAM and granted MoE layers; nothing
here licenses a reclaim claim.

### Notes binding on later tasks

- **Real tensor names are `token_embd.weight` and `output.weight`.** There is no
  `lm_head` tensor in either model — llama.cpp's GGUF conversion names the LM head
  `output.weight`. A predicate keyed on `lm_head` matches nothing in either model.
  Both winners carry the `.weight` suffix, so a predicate must use a substring or
  prefix match, not string equality against `token_embd` / `output`.
- **`output.weight` and `token_embd.weight` do not share a type.** GPT-OSS has both
  at Q8_0; Mistral has `output.weight` at Q6_K and `token_embd.weight` at Q4_0. A
  predicate keyed on type rather than name will not generalize.
- **The logged shape is truncated to two dimensions and understates expert tensors
  by the expert count.** `blk.*.ffn_*_exps.weight` prints as `2880 × 2880`, which at
  MXFP4 is 4.2 MB, yet the tensor is 134.5 MB — the real tensor is 3D with
  `ne[2] = 32` experts. Any later predicate that computes a size from `ne[0] × ne[1]`
  will be wrong by 32 × on GPT-OSS expert tensors. Use `placement_tensor_info::size`,
  which is the byte size the sizing chain actually consumes.
- **`placement_tensor_info::ne` is only valid when `has_shape()` is true**
  (`unified-cache.hpp:359`). Every entry in both captures printed a non-zero shape,
  so no shapeless entries were observed on these two models — but that is an
  observation about two models, not a guarantee.

### No-behaviour-change evidence

The GPT-OSS `[SYCL-PLAN]` capture with the new line removed is byte-identical to
the pre-change capture:

```bash
$ diff /tmp/zone-sizing/before.txt \
       <(cat /tmp/zone-sizing/gpt-oss-20b-mxfp4.plan.txt | grep -v 'inventory top-8')
$ echo $?
0
```

All zone sizes — DMA staging pool, PP MoE oneDNN scratch ring, host staging zone,
MoE routing buffers, KV inputs — are unchanged.
