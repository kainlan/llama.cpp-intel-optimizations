# Qwen3.8-Flash-Next (qwen4exp) MTP: feasibility findings

Tracker: `llama.cpp-0rhb`. Written 2026-10-03. Status: **acceptance not yet measured; SYCL wiring on hold.**

Question: can we draft with the model's MTP head and verify with the trunk (Strata's "verify window",
1.6-1.8x at 2-bit), and is it worth wiring on the SYCL backend given our per-token expert-read cost?

## Facts

**Our tree before this work** had the generic MTP framework (NextN tensor names, `LLAMA_CONTEXT_TYPE_MTP`,
`LLM_GRAPH_TYPE_DECODER_MTP`, `common_speculative_impl_draft_mtp`, `--spec-type draft-mtp`, `-md`) but no
qwen4exp MTP graph, block loading, hidden-state tap or draft memory filter. `qwen4exp + draft-mtp` did nothing.

**Upstream.** ggml-org/llama.cpp#29761 "Qwen4Exp: add MTP" (am17an) merged 2026-10-01. It supersedes
#28243 (Unsloth, closed unmerged) and #27836 (draft). Unsloth's `qwen4exp/mtp` branch is the #28243 lineage.
Reported by the PR: 1.55x decode speedup, 0.640 acceptance (Qwen3.8-Flash-Next IQ4_XS, `--spec-draft-n-max 3`,
DGX Spark, greedy, 24 prompts over 7 categories; per category 0.58-0.78). Unsloth reports 66% acceptance at
n-max 2 and 1.34-1.67x on a B200. Both are measured with everything in VRAM or unified memory, which is
not our placement.

**Neither local GGUF has an MTP head.**
- Unsloth Q8_0 (6 splits): 1224 tensors, `blk.0` to `blk.47` only, no `nextn.*`, no
  `qwen4exp.nextn_predict_layers` key. `n_layer_all == n_layer`, so the loader's MTP path stays inert.
- ISTA-DASLab GSQ-RCO IQ3_XXS (2 splits, 47.0 + 28.8 GB, downloaded 2026-10-03): dumped from the real
  files, 1224 tensors (`split.tensors.count` = 1224), `blk.0` to `blk.47`, zero nextn/mtp/eh_proj/enorm/
  hnorm/shared_head matches, `block_count = 48`, no `nextn_predict_layers`. Mixed per-tensor types (BF16 dense
  tensors, IQ2/IQ3/IQ4 and 38 Q2_0 tensors, Q4_K/Q5_K/Q6_K); `general.file_type = 23`.

**The head is a separate sidecar GGUF**, not embedded: ggml-org/Qwen3.8-Flash-Next-GGUF
`mtp-Qwen3.8-Flash-Next-{Q8_0 (4.14 GB), Q4_0 (2.20 GB), BF16 (7.77 GB)}.gguf`. Self-contained: it carries
`token_embd`, `output` and the block. Unsloth's `shared-*` heads borrow the target's embedding and output
projection; the merged code does not support borrowing ("needs requants" per the PR thread), so use the
ggml-org files. They work with any quant of the target: the head consumes only the trunk's residual and the
next token.

**What the head file contains** (dumped, ggml-org `mtp-...-Q8_0.gguf`, 4,137,429,280 B, 34 tensors):
`general.architecture = qwen4exp`, `block_count = 49`, `nextn_predict_layers = 1`, `compress_ratios` and
`recurrent_layers` with 49 entries (block 48: ratio 4, not recurrent, so QSA full attention). Tensors:
`token_embd` and `output` (635.7 M elements each, Q8_0), and `blk.48.*` only: attention q/k/v/o and norms, the
indexer (BF16), MoE `ffn_{gate,up,down}_exps` (512 experts, 2.5 G elements together), the shared expert, the
two hc modules, and `blk.48.nextn.{eh_proj,enorm,hnorm,hc_head_norm,hc_head_down,hc_head_up}` -- exactly the
names the port registers. The Q4_0 head has the same 34 tensors. Downloaded to
`/models/Qwen3.8-Flash-Next-MTP/`.

**Architecture** (from the port): one block, `n_layer_nextn == 1`: a full-attention QSA layer with its own
indexer cache, plus a MoE FFN (512 experts, top 10, like a trunk layer), fed by `eh_proj([enorm(tok_embd);
hnorm(h)])` per hyper-connection stream, collapsed by its own `hc_head` mixer, then the target's output
projection. `h` is the trunk's hc-wide residual (n_embd * hc = 10240 floats per token), tapped before the
final mixer. Verification is exact: the output is identical to plain greedy decoding.

## The port in this tree

`a15fe6317` carries #29761 over, credited to upstream. This tree predates upstream #29751 (the QSA rewrite
that #29761 builds on), so `graph_mtp` was adapted to call `build_layer_attn` without the kpool input
(QSA here is derived from `mctx_hyb->get_idx()` and the layer's compress ratio), and two upstream hunks
with no target here were dropped. Also fixed in the same series: the one-line `speculative.cpp` bug where
the draft model was loaded from `params.model.path` instead of the draft path (`4679665af` separately guards
a pre-existing `-DGGML_SYCL=OFF` compile break in `llama-model-loader.cpp`).

Verified: a CPU-only build compiles and links `llama-speculative-simple`. **Not verified**: any model load
(the lead runs those), the converter (`conversion/qwen4exp.py`, needs the BF16 checkpoint), and anything
on SYCL.

## Expected speedup at our expert-read cost (estimate, not a measurement)

Per-token decode `t1 = A + E`, where `A` is everything that is not the expert reads (GPU dense, GDN, QSA,
launch) and `E` is the host expert-read time. A round with `k` drafts verifies `n = k + 1` tokens:

    T_round = A + m(n) * E + k * D        D = one draft step, ~5 ms (MTP head on the GPU)
    tokens per round T = 1 + a + a^2 + ... + a^k        a = per-position acceptance
    speedup S = (A + E) * T / T_round

`m(n)` is the number of distinct experts the verify batch reads, relative to one token: 2.9 at n = 3 and 3.9
at n = 4 if routing were independent (512 experts, top 10); about 2.4 and 3.0 if routing is skewed.
Reported acceptance numbers are consistent with `a` of 0.75-0.8 (PR: 0.64 accepted/generated at n-max 3,
mean length 2.76).

**The point: MTP amortizes `A`, never `E`.** Every drafted token that is accepted still has its own experts
read, and the verify batch reads their union, which grows nearly linearly with `n` for top-10 of 512. A
draft that is rejected costs its experts too. So MTP pays only when `E << A`: experts mostly VRAM-resident
or aggressively quantized.

| scenario (A, E in ms) | a=0.7 | a=0.8 |
|---|---|---|
| Q8_0, ~83% of experts host-resident (A=20, E=80; 1.9 GiB at 25 GB/s) | 0.65-0.98x | 0.79-1.10x |
| IQ3_XXS, CPU kernels bandwidth-bound (A=20, E=21) | 0.81-1.11x | 0.98-1.24x |
| IQ3_XXS, CPU kernels compute-bound (A=20, E=40) | 0.72-1.04x | 0.87-1.16x |
| mostly VRAM-resident (A=20, E=8) | 0.99-1.24x | 1.20-1.39x |

Ranges span k = 2..4 and independent-to-skewed routing. These are ordering, not predictions: even the
mostly-resident row is below Strata's measured 1.6-1.8x, so `A` may be overstated. On these numbers Q8_0 as
it stands gets nothing from MTP. The expert cache (Strata lesson 1) and lower-bit experts matter more, and
they are prerequisites for MTP to pay.

## Risks for SYCL wiring (not investigated)

- A second `llama_model` (draft head: 1 layer + `token_embd` + `output`, ~4.1 GB at Q8_0) and a second
  `llama_context` on the same device. The canonical contract (section 5) records that same-device concurrent
  inference lacks context-keyed KV/RUNTIME arena ownership; draft and target alternate on one thread, but
  this needs checking against the planner and unified cache.
- The verify batch is a (k+1)-token decode through the multi-token MoE path with host-resident experts, not
  today's n = 1 fast path.
- Upstream reports a very large draft-context compute buffer (22 GB at `-ub 4096` with the Q8 head; it needed
  `-ub 512` or the Q4 head), which matters under our placement budgeting.

## How to measure acceptance (CPU-only, no SYCL)

Acceptance and tokens per round depend on the target and the prompts, not the backend.

    scripts/qwen4exp-mtp-acceptance.sh --dry-run      # print the 24 commands
    scripts/qwen4exp-mtp-acceptance.sh                # run them, serially (SETS=base or SETS=p05 for 12)
    # THREADS defaults to 16 (-t/-tb/-td/-tbd); the binary's own default of 4 ran 0.225 t/s.  WARM=1 pre-reads every shard.

Target: the IQ3_XXS pair (76 GB, fits in page cache; the 177 GB Q8_0 would thrash it). Heads: ggml-org Q8_0
and Q4_0, in `/models/Qwen3.8-Flash-Next-MTP/`. Arms: n-max {2, 3} x prompts {code, chat, reasoning},
greedy, seed 42, 256 new tokens, in two sets: `base` (binary default p-min 0.00, drafts never filtered) and
`p05` (`--spec-draft-p-min 0.5`, Strata's keep-while-p>=0.5 filter, for the comparison). Built with `cmake -B build-cpu -G Ninja -DGGML_SYCL=OFF -DLLAMA_CURL=OFF`
and `--target llama-speculative-simple`. It is a model load: run it by hand, one process at a time. The
parser (`scripts/parse-qwen4exp-mtp-acceptance.py`, gate `test-qwen4exp-mtp-acceptance-parser`) rejects a
log with `n_drafted = 0` instead of reporting 0% acceptance, because that is speculation that never ran.

To turn the table above into a number, two more measurements are needed: a `llama-bench` decode baseline on
the SYCL build, and the GPU-busy versus CPU-expert-pool-busy split per token (that gives `A` and `E`).

## Decision

Hold the SYCL wiring until measured acceptance and measured `A`/`E` give `S > ~1.2x`. On the estimate that
happens for IQ3_XXS with a hot VRAM expert cache or a mostly-resident configuration, not for Q8_0 as it
stands.
