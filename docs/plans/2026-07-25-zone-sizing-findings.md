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
  -m /models/<model>.gguf -p 0 -n 4 -r 1 -v 2>&1 | grep '\[SYCL-PLAN\]'
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

---

## Task 8 — Reclaim measurement

Measured reclaim, granted MoE layers, and an interleaved paired throughput
comparison of the plan's 19 commits.

### Builds compared

| arm | SHA | build |
|---|---|---|
| **after** | `b36bb603b` (HEAD of `feature/sycl-b70-capability`) | `/Apps/llama.cpp/build` |
| **before** | `57db1e693` | separate source+build tree (`git archive` extract, own `build/`) |

**The baseline is `57db1e693`, not `git merge-base HEAD master`.** The merge-base
resolves to `dc02e1f83`, 50 commits back; this plan contributed only 19, so 31
commits of unrelated branch work sit between. `57db1e693` is the commit
immediately before this plan's first (`f509f291d`) and is the true "before".

A separate extracted source tree was used rather than a `git worktree` so the
main tree's `build/` (and its ccache warmth) was never disturbed.

### Method

- Zone figures: `llama-bench -p 0 -n 4 -r 1 -v` with `GGML_SYCL_DEBUG=1`.
  **`-v` is mandatory** — without it `llama-bench` installs a null log callback
  and every `GGML_LOG_INFO` line is discarded.
- One model per `llama-bench` process throughout (a second `-m` aborts on a
  leaked model-weight lease — `llama.cpp-ljb9`, pre-existing).
- Throughput: `-p 512 -n 128 -r 1 -o csv`, **interleaved paired**, 8 pairs per
  model per card, with the arm order alternating on every pair.
- B70 runs set `GGML_SYCL_OP_TIMEOUT_MS=180000`.
- Free VRAM confirmed at the top of every run: **32603 MiB on the B70**, **16250
  MiB on the B50** — both full cards, no foreign workload holding VRAM.
- `journalctl -k --since "2 hours ago"` matched **zero** `GT reset` / `guc_id` /
  `GPU hang` / `xe.*reset` lines before or after the measurements. (`dmesg` is
  privilege-denied on this host.)

### Machine contention — the machine was NOT quiet

Stated plainly because it bounds what the absolute numbers mean:

- The codescout daemon ran a semantic re-index for the entire session,
  **1,171,451 of 1,181,563 chunks still pending**, consuming 600–1430 % CPU on a
  20-core host.
- A runaway `python3` from another session held ~100 % CPU throughout (277 min
  accumulated).
- Load average stayed between **14.6 and 22.5** across all measurements.

The load was *sustained and steady*, not bursty, and every comparison is
interleaved with the arm order alternating per pair, so it loads both arms
equally. The **paired deltas below are therefore trustworthy; the absolute
throughput figures are depressed** and should not be quoted as baselines. The
clearest evidence of that: GPT-OSS on the B70 measured 1366–1368 pp512 in *both*
arms against a 1414.62 baseline — a 3.4 % deficit present in the pre-plan build
too, i.e. attributable to the machine, not to this plan.

### Zone figures — GPT-OSS 20B MXFP4

Planner figures are identical on both cards (they derive from the tensor
inventory, not the device).

| figure | before | after | delta |
|---|---:|---:|---:|
| `[SYCL-PLAN] DMA staging pool` | 1173.6 MB (2 × max_tensor 586.8) | 268.9 MB (2 × dma_streamed 134.5) | **−904.7 MB** |
| `[SYCL-PLAN] oneDNN scratchpad` | 1173.6 MB (2 × max_tensor 586.8) | 268.9 MB (2 × onednn_eligible 134.5) | **−904.7 MB** |
| `[SYCL-PLAN] CPU quant buffers` | 1760.4 MB (3 × 586.8) | 403.4 MB (3 × cpu_quant_eligible 134.5) | −1357.0 MB (**write-only — reclaims nothing**) |
| `onednn_reorder_bytes` | 586.8 MB (1 × 586.8) | 134.5 MB | **−452.3 MB** |
| `[SYCL-PLAN] Host staging zone` | 1191.7 MB | 1191.7 MB | 0 (left on global max by design) |
| `[SYCL-PLAN] PP MoE oneDNN scratch ring` | 49.6 MB | 49.6 MB | 0 |

The before build does not log the oneDNN-scratchpad, CPU-quant, or reorder lines
(they are new in this plan); their before values are the documented multipliers
applied to `max_tensor_bytes` = 586.8 MB, and both are corroborated by the two
*measured* aggregates below.

**VRAM zones (measured, `[VRAM-ARENA] Reserved …`):**

| card | zone | before | after | delta |
|---|---|---:|---:|---:|
| B50 | ONEDNN | 1173.6 MB | 268.9 MB | **−904.7 MB** |
| B50 | weight | 13348.4 MB | 14253.1 MB | **+904.7 MB** |
| B70 | ONEDNN | 1173.6 MB | 268.9 MB | **−904.7 MB** |
| B70 | weight | 29578.1 MB | 30482.9 MB | **+904.8 MB** |

**Conservation check reproduced, three independent ways.** The arena weight zone
grows by what the ONEDNN zone gives up, on both cards — this is the proof the
VRAM reclaim is real and not a dropped budget term. It is **exact on the B50**
(−904.7 / +904.7). The B70 row reads −904.7 against **+904.8**: that 0.1 MB is
the same rounded-MB artifact explained under "On 452.3 vs 452.4" below — the two
zones' MB figures are each rounded independently before being differenced, so a
0.1 MB residual is expected and is not a budget leak. A third confirmation comes
from the B70 `[MOE-LAYOUT] gateup-i8 … remaining=` figure, which moves
11752.8 → 12657.5 MB, again **+904.7 MB** — an independent counter on the same
card that lands on the B50's exact value.

The proof line for the zone change is
`[VRAM-ARENA] Rebuilding unused early arena for planned zones: scratch
512.0->512.0 MB, oneDNN 256.0->1173.6 MB` (before) versus `oneDNN 256.0->268.9
MB` (after).

**Host arena zones (measured, `[HOST-ARENA] configured zones`):**

| card | zone | before | after | delta |
|---|---|---:|---:|---:|
| both | SCRATCH | 2644.1 MB | 1287.0 MB | **−1357.1 MB** |
| both | STAGING | 1191.7 MB | 1191.7 MB | 0 |

The SCRATCH delta decomposes exactly into the two consumers that feed
`host_zone_scratch_bytes` (`unified-cache.cpp`, the `plan.host_zone_scratch_bytes
= std::max(...)` sum): `onednn_reorder` −452.3 MB + `dma_staging_pool` −904.7 MB
= −1357.0 MB, versus −1357.1 MB measured (rounding). This reconciles T3's
oneDNN-reorder reclaim and T5's DMA-staging reclaim into one measured host-side
figure.

**On 452.3 vs 452.4.** T3 records the oneDNN-reorder reclaim as 452.4 MB; the
table above says 452.3 MB. **These are the same quantity**, differing only in
where the rounding happens. Subtracting the two already-rounded MB figures gives
586.8 − 134.5 = **452.3**. Subtracting in bytes and rounding once gives
615,329,280 − 141,004,800 = 474,324,480 B = **452.4** MiB. (The tensors:
`output.weight` at Q8_0, 579,133,440 elements ÷ 32 × 34 B; `blk.*.ffn_*_exps.weight`
at MXFP4, 2880 × 2880 × 32 = 265,420,800 elements ÷ 32 × 17 B.) The byte-exact
452.4 is the more accurate figure; 452.3 is used in the table above for internal
consistency with the rounded MB columns it is derived from. Neither is an error.

### Zone figures — Mistral 7B v0.1 Q4_0

| figure | before | after | delta |
|---|---:|---:|---:|
| `[SYCL-PLAN] DMA staging pool` | 205.1 MB (2 × max_tensor 102.5) | 63.0 MB (2 × dma_streamed 31.5) | −142.1 MB |
| `[SYCL-PLAN] oneDNN scratchpad` | 205.1 MB | 63.0 MB | −142.1 MB |
| `[SYCL-PLAN] CPU quant buffers` | 307.6 MB | 94.5 MB | −213.1 MB (write-only) |
| `onednn_reorder_bytes` | 102.5 MB | 31.5 MB | −71.0 MB |
| `[HOST-ARENA] SCRATCH` (measured) | 442.2 MB | 229.0 MB | **−213.2 MB** |
| `[HOST-ARENA] STAGING` (measured) | 223.1 MB | 223.1 MB | 0 |
| VRAM ONEDNN zone, B50 & B70 (measured) | 256.0 MB | 256.0 MB | **0** |
| VRAM weight zone, B50 (measured) | 14266.0 MB | 14266.0 MB | **0** |
| VRAM weight zone, B70 (measured) | 30495.8 MB | 30495.8 MB | **0** |

Host SCRATCH again decomposes exactly: −71.0 (reorder) + −142.1 (DMA staging) =
−213.1 MB vs −213.2 measured.

**Mistral reclaims zero VRAM, and that is floor behaviour, not a failure.**
Neither build emits `ONEDNN zone raised` or `[VRAM-ARENA] Rebuilding …` on
Mistral: the before estimate (205.1 MB) and the after estimate (63.0 MB) both sit
*below* the 256 MB ONEDNN zone floor, so the zone is 256 MB either way. An absent
`Rebuilding` line here is the floor working, not a failed capture.

### Granted MoE layers (B50, GPT-OSS 20B)

From `[MOE-LAYOUT] down-i8 device=0 …`:

| | before | after |
|---|---:|---:|
| candidates | 24 | 24 |
| **`upgraded_tensors`** | **6** | **10** |
| `upgraded_entries` | 192 | 320 |
| `preserved_soa` | 192 | 320 |
| `remaining` headroom | 221.7 MB | 82.2 MB |

**The reclaim bought 4 additional down-i8 layers on the B50, 6/24 → 10/24.**

The arithmetic closes: the pass consumed the 904.7 MB of reclaimed weight-zone
VRAM *plus* 139.5 MB of the headroom it already had (221.7 → 82.2), i.e. 1044.2
MB for 4 tensors = **261.05 MB per tensor** — matching the ~261 MB/tensor figure
the plan assumed. The remaining 14 candidates would need a further ~3.6 GB; both
builds report the shortfall as the VRAM headroom guard, not an eligibility
failure.

**On the B70 the count cannot move: both builds already grant all 24.**
`[MOE-LAYOUT] summary device=0 … down=mxfp4_i8(768)` in both arms, and the
`down-i8` diagnostic line — which prints only when something is declined — is
absent from both captures. The B70's extra 904.7 MB shows up purely as unused
headroom (11752.8 → 12657.5 MB).

Gate/up is unaffected on both cards and in both builds: `gateup-i8 … candidates=0
… skip_executor=1536`, an eligibility outcome unrelated to VRAM.

### Throughput — interleaved paired, 8 pairs each

Every pair alternates which arm runs first. Deltas are after − before.

| card / model | metric | before | after | paired delta | t (df=7) | p |
|---|---|---:|---:|---:|---:|---:|
| B50 / GPT-OSS 20B | pp512 | 898.41 ± 6.48 | 889.33 ± 3.07 | **−9.08 ± 4.77 (−1.01 %)** | −5.38 | **0.0010** |
| B50 / GPT-OSS 20B | tg128 | 35.43 ± 0.73 | 35.63 ± 0.64 | +0.21 ± 1.13 (+0.58 %) | +0.52 | 0.6197 |
| B70 / GPT-OSS 20B | pp512 | 1368.05 ± 17.18 | 1366.26 ± 7.71 | −1.78 ± 17.63 (−0.13 %) | −0.29 | 0.7830 |
| B70 / GPT-OSS 20B | tg128 | 48.46 ± 1.92 | 49.15 ± 1.43 | +0.69 ± 1.55 (+1.42 %) | +1.26 | 0.2492 |
| B50 / Mistral 7B | pp512 | 1207.18 ± 14.29 | 1209.83 ± 11.95 | +2.65 ± 24.37 (+0.22 %) | +0.31 | 0.7673 |
| B50 / Mistral 7B | tg128 | 46.96 ± 0.10 | 46.93 ± 0.14 | −0.03 ± 0.19 (−0.06 %) | −0.41 | 0.6953 |
| B70 / Mistral 7B | pp512 | 2542.36 ± 76.22 | 2549.44 ± 110.04 | +7.08 ± 92.86 (+0.28 %) | +0.22 | 0.8353 |
| B70 / Mistral 7B | tg128 | 109.62 ± 1.50 | 109.95 ± 1.05 | +0.33 ± 2.02 (+0.30 %) | +0.46 | 0.6590 |

Seven of the eight comparisons are indistinguishable from zero. **One is not: B50
GPT-OSS pp512 is 1.01 % slower after the plan, p = 0.0010.** Reported rather than
rounded away.

#### Isolating the layout change from the sizing change

Two pieces of evidence, strongest first. Neither is a proof; together they make
the extra MoE layers — not the sizing arithmetic — the leading explanation.

**1. The B70 control (structurally strongest, and mechanism-independent).** The
B50 is the only configuration whose *MoE layout changes* between the arms
(6 → 10 down-i8 tensors). The B70 grants all 24 down tensors in both arms, so
there the *only* difference between the builds is the zone sizes — and there is
no pp512 effect at all: **−0.13 %, p = 0.78**, over the same 8 interleaved pairs.
This is a genuine control: it does not depend on any forced configuration, and it
says the sizing change on its own has no measurable pp512 cost.

**2. A forced-headroom experiment on the B50 (corroborating, not conclusive).**
Re-running the **after** build with
`GGML_SYCL_VRAM_ARENA_EXTERNAL_HEADROOM_MB=1795` shrinks the arena until the
down-i8 pass grants **6** tensors, the same count the before build gets. The
comparison then inverts:

| B50 / GPT-OSS, down-i8 count matched at 6/24 | before | after | paired delta | t (df=5) | p |
|---|---:|---:|---:|---:|---:|
| pp512 | 900.62 ± 3.29 | 905.54 ± 2.48 | **+4.92 ± 1.28 (+0.55 %)** | +9.41 | 0.0002 |
| tg128 | 35.72 ± 0.38 | 36.52 ± 0.29 | **+0.80 ± 0.54 (+2.25 %)** | +3.66 | 0.0146 |

**What this experiment does not establish.** Three limits, stated so the result
is not cited beyond its strength:

- it matches the down-i8 **count** (6), not per-tensor **identity** — the two
  builds are not verified to have upgraded the *same* six tensors;
- forcing external arena headroom is a **different mechanism** from the before
  build's genuinely larger ONEDNN zone, so the two configurations are not
  identical in the way a true controlled swap would be;
- the two B50 experiments are separately paired against the same before build
  rather than being one paired design, so the comparison between them is
  quasi-experimental.

**Conclusion at the strength the evidence supports.** The −1.01 % is most likely
the cost of the four extra down-i8 tensors the reclaimed VRAM buys, rather than a
cost of this plan's sizing change; the B70 control alone already shows the sizing
change carries no pp512 penalty where the layout is held fixed. A dedicated
"is down-i8 worth it at the margin on the B50?" experiment — one that pins tensor
identity and varies only the layout — is worth filing to close the question.

#### Raw per-pair measurements

Every `avg_ts` behind the t-tests above, embedded here rather than referenced by
path. Task 1's artifacts went to `/tmp/zone-sizing/` and were destroyed by a
mid-plan reboot — `/tmp` is tmpfs on this host — so a path would be a pointer
that dangles and would leave the statistics permanently unverifiable. The data is
small, so it goes in the committed record instead, on the same principle that put
the log lines in the zone sections: **every t and p below is recomputable from
this document alone, by anyone, without the machine.**

Each row is one interleaved pair. `first arm` is the arm that ran first in that
pair — it alternates by construction, so drift within a pair loads both arms
equally over the set. Units are tok/s; each figure is a single
`llama-bench -p 512 -n 128 -r 1 -o csv` process.

**B50 / GPT-OSS 20B**

| pair | first arm | pp512 before | pp512 after | tg128 before | tg128 after |
|---:|---|---:|---:|---:|---:|
| 1 | after | 903.51 | 891.90 | 36.27 | 34.91 |
| 2 | before | 905.59 | 890.43 | 35.95 | 34.89 |
| 3 | after | 903.99 | 894.09 | 35.95 | 36.17 |
| 4 | before | 901.51 | 886.75 | 35.33 | 34.83 |
| 5 | after | 891.26 | 886.37 | 35.63 | 36.30 |
| 6 | before | 894.99 | 886.91 | 34.79 | 35.87 |
| 7 | after | 898.67 | 891.79 | 35.47 | 36.10 |
| 8 | before | 887.80 | 886.44 | 34.01 | 36.00 |

**B70 / GPT-OSS 20B**

| pair | first arm | pp512 before | pp512 after | tg128 before | tg128 after |
|---:|---|---:|---:|---:|---:|
| 1 | after | 1363.28 | 1360.24 | 47.46 | 48.40 |
| 2 | before | 1338.55 | 1370.44 | 45.24 | 49.08 |
| 3 | after | 1355.42 | 1365.58 | 47.19 | 49.05 |
| 4 | before | 1360.70 | 1358.02 | 48.97 | 48.23 |
| 5 | after | 1383.42 | 1355.94 | 48.21 | 47.59 |
| 6 | before | 1383.35 | 1377.64 | 50.65 | 51.30 |
| 7 | after | 1369.50 | 1368.75 | 48.68 | 48.16 |
| 8 | before | 1390.13 | 1373.48 | 51.24 | 51.36 |

**B50 / Mistral 7B**

| pair | first arm | pp512 before | pp512 after | tg128 before | tg128 after |
|---:|---|---:|---:|---:|---:|
| 1 | after | 1196.13 | 1213.87 | 47.02 | 47.15 |
| 2 | before | 1222.52 | 1216.12 | 46.98 | 47.03 |
| 3 | after | 1196.04 | 1218.38 | 46.95 | 46.73 |
| 4 | before | 1226.44 | 1188.16 | 46.98 | 46.77 |
| 5 | after | 1187.19 | 1226.21 | 46.96 | 46.98 |
| 6 | before | 1219.98 | 1202.35 | 46.74 | 47.04 |
| 7 | after | 1204.27 | 1201.40 | 47.06 | 46.89 |
| 8 | before | 1204.90 | 1212.18 | 46.99 | 46.87 |

**B70 / Mistral 7B**

| pair | first arm | pp512 before | pp512 after | tg128 before | tg128 after |
|---:|---|---:|---:|---:|---:|
| 1 | after | 2508.37 | 2439.38 | 110.84 | 109.27 |
| 2 | before | 2507.81 | 2669.40 | 107.04 | 110.50 |
| 3 | after | 2413.81 | 2355.21 | 107.82 | 109.96 |
| 4 | before | 2571.91 | 2677.14 | 110.81 | 110.06 |
| 5 | after | 2566.57 | 2582.57 | 111.16 | 110.89 |
| 6 | before | 2661.64 | 2542.20 | 109.49 | 110.99 |
| 7 | after | 2608.34 | 2600.89 | 110.40 | 107.74 |
| 8 | before | 2500.43 | 2528.75 | 109.40 | 110.17 |

**B50 / GPT-OSS 20B — down-i8 count matched at 6/24** (after arm carries
`GGML_SYCL_VRAM_ARENA_EXTERNAL_HEADROOM_MB=1795`)

| pair | first arm | pp512 before | pp512 after | tg128 before | tg128 after |
|---:|---|---:|---:|---:|---:|
| 1 | after | 903.15 | 908.32 | 36.29 | 36.64 |
| 2 | before | 903.85 | 907.80 | 35.45 | 36.49 |
| 3 | after | 902.53 | 906.46 | 35.42 | 36.72 |
| 4 | before | 898.07 | 902.48 | 35.78 | 36.69 |
| 5 | after | 900.68 | 905.40 | 35.35 | 36.61 |
| 6 | before | 895.43 | 902.78 | 36.00 | 35.96 |

The reported statistic is a two-sided paired t-test on the per-pair differences
(after − before), df = pairs − 1; the ± figures in the summary tables are sample
standard deviations, not standard errors.

Recomputing all ten t-tests from the tables above reproduces the summary tables.
The figures here are rounded to 2 dp for the record while the summary statistics
were computed at full `avg_ts` precision, so a recomputation drifts in the last
digit — the largest divergence across all ten is p = 0.6197 → 0.6166 (B50
GPT-OSS tg128, null either way). No sign, no significance verdict, and no delta
percentage to 2 dp changes.

### Gates

Both gates, both cards, on the after build (`b36bb603b`):

| gate | B70 (`level_zero:0`) | B50 (`level_zero:1`) |
|---|---|---|
| Mistral 7B Q4_0 deterministic completion | **PASS** — `1, 2, 3, 4, 5, 6, 7, 8, 9, 10` | **PASS** — `1, 2, 3, 4, 5, 6, 7, 8, 9, 10` |
| GPT-OSS 20B MXFP4 count gate | **PASS** — `1, 2, 3, 4, 5` | **PASS** — `1, 2, 3, 4, 5` |

**Performance gated against `docs/backend/sycl-perf-baselines.md`, not CLAUDE.md.**
CLAUDE.md's ">= 1100 PP512, ~50+ TG128" GPT-OSS guardrail predates the 26.27
driver refresh and would report a false ~20 % catastrophe. The baselines document
records this branch on this driver at 893.77 ± 2.32 pp512 / 32.06 ± 0.22 tg128
(B50 GPT-OSS) and 1414.62 ± 11.13 / 43.57 ± 1.46 (B70 GPT-OSS).

| run (after build) | baselines row | measured | verdict |
|---|---|---|---|
| B50 GPT-OSS | 893.77 / 32.06 | 889.33 / 35.63 | pp −0.50 %, tg **+11.1 %** — pass |
| B70 GPT-OSS | 1414.62 / 43.57 | 1366.26 / 49.15 | pp −3.4 %, tg **+12.8 %** — pass; the pp deficit is present in the *before* arm too (1368.05) and is machine load |
| B50 Mistral | 1187.83 / 46.53 | 1209.83 / 46.93 | both above baseline — pass |
| B70 Mistral | 2495.42 / 107.66 | 2549.44 / 109.95 | both above baseline — pass |

### Predicate underestimates

`GGML_SYCL_DEBUG=1` prints `[SYCL-PLAN] zone sizing coverage: onednn
observations=N underestimates=M` once per plan.

| run | observations | underestimates |
|---|---:|---:|
| B50 GPT-OSS, `-p 0 -n 4` | 0 | 0 |
| B70 GPT-OSS, `-p 0 -n 4` | 0 | 0 |
| B50 GPT-OSS, `-p 512` | 0 | 0 |
| B50 Mistral, `-p 0 -n 4` | 0 | 0 |
| B70 Mistral, `-p 0 -n 4` | 0 | 0 |
| **B50 Mistral, `-p 512`** | **3** | **0** |
| **B70 Mistral, `-p 512`** | **3** | **0** |

**Zero underestimates across every run — with two caveats that must travel with
that zero.**

1. **GPT-OSS never enters `reserve_onednn_scratch` at all** (observations = 0 in
   every GPT-OSS run, including a prompt-processing run at `-p 512`). The check is
   structurally blind on that model; its entire value rests on the Mistral runs.
2. **Mistral's zero is partly the 256 MB floor absorbing a real ~1.8× predicate
   under-sizing** (filed as `llama.cpp-2wgg`). The floor is doing work the
   predicate is not. That zero means "no observed request exceeded the *zone*",
   not "the predicate is right".

Observations only appear on a run that actually processes a prompt — `-p 0 -n 4`
returns 0 on every model/card combination. A capture that omits prompt processing
cannot detect an underestimate.

### Summary

```
RECLAIMED (VRAM):  GPT-OSS 20B  -904.7 MB ONEDNN zone -> +904.7 MB arena weight zone
                                (B50 13348.4 -> 14253.1; B70 29578.1 -> 30482.9)
                   Mistral 7B    0 MB -- both estimates sit under the 256 MB floor
RECLAIMED (HOST):  GPT-OSS 20B  -1357.1 MB host SCRATCH zone (2644.1 -> 1287.0)
                   Mistral 7B    -213.2 MB host SCRATCH zone (442.2 -> 229.0)
                   [cpu_quant_buffer_bytes is write-only and reclaims nothing]
MOE LAYERS GRANTED: B50 GPT-OSS down-i8  6/24 -> 10/24  (+4 layers, 261.05 MB each)
                    B70 GPT-OSS down-i8  24/24 -> 24/24 (already full; +904.7 MB idle headroom)
                    Mistral: no MoE path
THROUGHPUT:        7 of 8 interleaved paired comparisons null. B50 GPT-OSS pp512
                   -1.01% (p=0.0010) -- most likely the cost of the 4 extra
                   down-i8 tensors rather than the sizing. Evidence: the B70,
                   where the layout is identical in both arms, shows no pp512
                   effect (-0.13%, p=0.78); and a forced-headroom B50 run with
                   the down-i8 count matched at 6/24 inverts the sign (+0.55%
                   pp512 p=0.0002, +2.25% tg128 p=0.0146). Not conclusive -- see
                   "Isolating the layout change from the sizing change".
GATES:             4/4 PASS (Mistral completion + GPT-OSS count, both cards).
                   Perf gated on docs/backend/sycl-perf-baselines.md, not CLAUDE.md.
UNDERESTIMATES:    0 observed. But GPT-OSS records 0 observations (structurally
                   blind), and Mistral's 3 observations / 0 underestimates rest on
                   the 256 MB floor absorbing a ~1.8x under-size (llama.cpp-2wgg).
```

Machine was **not** quiet (codescout semantic re-index, 600–1430 % CPU, load
14.6–22.5 throughout). All comparisons interleaved with alternating arm order;
paired deltas hold, absolute throughput figures are depressed and are not
baselines.
