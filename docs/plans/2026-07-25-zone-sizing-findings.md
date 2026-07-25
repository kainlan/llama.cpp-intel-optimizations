# Path-Scoped Zone Sizing — Measured Findings

Ground-truth measurements produced by the tasks in
`docs/plans/2026-07-25-sycl-path-scoped-zone-sizing.md`. No task in that plan may
quote a byte figure that does not appear in this document.

**Units:** every figure labelled `MB` here is binary — bytes ÷ 1024², i.e. a MiB.
The label matches the `[SYCL-PLAN]` log lines it is quoted from and the `MB`
convention used throughout `ggml/src/ggml-sycl/unified-cache.cpp`; the few places
that spell it `MiB` are the same unit, not a different one.

---

## Task 1 — Inventory

**Instrument:** two `[SYCL-PLAN]` lines added to `populate_host_zone_sizing`
(`ggml/src/ggml-sycl/unified-cache.cpp`), printed once per plan from the same
`tensor_inventory` that feeds `plan.max_tensor_bytes`:

- `inventory top-8` — the eight largest tensors by name, size, type and shape.
- `inventory groups` — the `(type, ne[0..3])` group cardinality distribution, for
  validating the structural per-layer-weight predicate.

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

---

## Task 1 (addendum) — Structural group cardinality

Path-scoped sizing classifies tensors by structure, not name:

```
key  = (type, ne[0], ne[1], ne[2], ne[3])
freq = number of inventory tensors sharing that key
is_per_layer_weight(t)  <=>  freq[key(t)] >= max(2, n_layer/2)
```

The `inventory groups` line reports the raw distribution so the threshold can be
checked rather than assumed.

### GPT-OSS 20B MXFP4 — `n_layer = 24`, threshold `max(2, 12) = 12`

```
[SYCL-PLAN] inventory groups: 11 distinct (type,ne) groups over 459 tensors; cardinality histogram: 2x1 24x5 48x2 72x1 73x1 96x1; largest card>=4: blk.12.ffn_up_exps.weight 134.5MB (card=72); largest card<=2: output.weight 586.8MB (card=2)
```

| Cardinality | Groups | Tensors |
|---:|---:|---:|
| 2  | 1 |   2 |
| 24 | 5 | 120 |
| 48 | 2 |  96 |
| 72 | 1 |  72 |
| 73 | 1 |  73 |
| 96 | 1 |  96 |
| **Total** | **11** | **459** |

Two populations: `{2}` and `{24, 48, 72, 73, 96}`. The threshold of 12 falls inside
a gap spanning 2 → 24, a **12× margin**. Clean separation.

`output.weight` and `token_embd.weight` share both type (Q8_0) and shape
(2880 × 201088), so they collapse into a *single* group of cardinality 2 rather
than two singletons.

### Mistral 7B v0.1 Q4_0 — `n_layer = 32`, threshold `max(2, 16) = 16`

```
[SYCL-PLAN] inventory groups: 7 distinct (type,ne) groups over 291 tensors; cardinality histogram: 1x2 32x1 64x3 65x1; largest card>=4: blk.27.ffn_gate.weight 31.5MB (card=64); largest card<=2: output.weight 102.5MB (card=1)
```

| Cardinality | Groups | Tensors |
|---:|---:|---:|
| 1  | 2 |   2 |
| 32 | 1 |  32 |
| 64 | 3 | 192 |
| 65 | 1 |  65 |
| **Total** | **7** | **291** |

Two populations: `{1, 1}` and `{32, 64, 64, 64, 65}`. The threshold of 16 falls
inside a gap spanning 1 → 32, a **32× margin**. Clean separation.

Here `output.weight` (Q6_K) and `token_embd.weight` (Q4_0) differ in type, so they
form two separate singletons. Both configurations — paired and split — land below
the threshold, so the rule holds either way.

Cardinalities above `n_layer` come from families that share a key across roles:
`64 = 32 + 32` is two same-shaped families merged (e.g. `attn_q` + `attn_o`), and
`65 = 32 + 32 + 1` is the per-layer norms plus `output_norm`. Merging is harmless
for this predicate — it only pushes a family further above the threshold.

### The reclaim, computed name-free

| Model | Largest with card ≤ 2 | Largest with card ≥ 4 | Delta | 8 × delta |
|---|---:|---:|---:|---:|
| GPT-OSS 20B | **586.8 MB** (`output.weight`, card=2) | **134.5 MB** (`blk.*.ffn_*_exps.weight`, card=72) | 452.3 MB | **3618.4 MB** |
| Mistral 7B  | **102.5 MB** (`output.weight`, card=1) | **31.5 MB** (`blk.*.ffn_*.weight`, card=64)      |  71.0 MB |  **568.0 MB** |

Per consumer, if all four repointed lines take the card ≥ 4 maximum:

| Consumer | × | GPT-OSS narrowed | Mistral narrowed |
|---|---:|---:|---:|
| `onednn_reorder_bytes`    | 1 × | 134.5 MB |  31.5 MB |
| `cpu_quant_buffer_bytes`  | 3 × | 403.5 MB |  94.5 MB |
| `dma_staging_pool_bytes`  | 2 × | 269.0 MB |  63.0 MB |
| `onednn_scratchpad_bytes` | 2 × | 269.0 MB |  63.0 MB |
| **Total** | **8 ×** | **1076.0 MB** | **252.0 MB** |

versus 4694.4 MB and 820.2 MB at HEAD. These remain **arithmetic implications of
the measured inventory, not measured reclaim** — a consumer only realizes the
delta if the narrowed maximum is genuinely the largest tensor its path can see,
which Tasks 3–5 establish and Task 8 measures.

### Notes on the structural predicate

- **The separation is far wider than the threshold needs.** The gap is 2 → 24 on
  GPT-OSS and 1 → 32 on Mistral. A **fixed threshold of 4** separates both models
  identically, with no `n_layer` term. Since `n_layer` is not a parameter of
  `populate_host_zone_sizing` (signature `(plan, tensor_inventory, n_experts,
  n_expert_used)`), a constant threshold avoids a signature change for no measured
  loss of discrimination on these two models. `max(2, n_layer/2)` is also safe on
  both; the choice is about plumbing cost, not correctness here.
- **The dominant histogram mode is not `n_layer`.** GPT-OSS's largest bucket is
  cardinality 24 with 5 groups (`n_layer = 24`, matches), but Mistral's is
  cardinality 64 with 3 groups against `n_layer = 32` — merged same-shape families
  double it. Deriving `n_layer` from the mode would be wrong on Mistral. Use the
  *smallest* family cardinality, or a fixed threshold.
- **All 459 / 291 entries carried a valid shape** on these two models, so
  `has_shape()` never gated a grouping decision here. A shapeless entry would have
  `ne = {0,0,0,0}` and would group with every other shapeless entry of the same
  type — a potentially large spurious family. A predicate keyed on this
  distribution should treat `!has_shape()` as "not a per-layer weight" explicitly
  rather than letting the zeros vote.
- **Ties within a group have unstable order**, so the reported `largest card>=4`
  tensor name varies between runs (`blk.12.ffn_up_exps.weight` and
  `blk.3.ffn_gate_exps.weight` on two GPT-OSS runs). The *size* is stable; only
  which tied member is named varies.

---

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

The GPT-OSS `[SYCL-PLAN]` capture with the two new lines removed is byte-identical
to the pre-change capture:

```bash
$ diff /tmp/zone-sizing/before.txt \
       <(cat /tmp/zone-sizing/gpt-oss-20b-mxfp4.plan.txt | grep -vE 'inventory (top-8|groups)')
$ echo $?
0
```

All zone sizes — DMA staging pool, PP MoE oneDNN scratch ring, host staging zone,
MoE routing buffers, KV inputs — are unchanged.
