# Qwen3.8-Flash-Next (qwen4exp) MTP: feasibility findings

Tracker: `llama.cpp-0rhb`. Written 2026-10-03. Status: **CPU-only acceptance and decode speedup measured 2026-10-03 (24 arms + 3 interleaved baseline pairs); SYCL/hybrid not measured; SYCL wiring on hold.**

Question: can we draft with the model's MTP head and verify with the trunk (Strata's "verify window",
1.6-1.8x at 2-bit), and is it worth wiring on the SYCL backend given our per-token expert-read cost?

## Facts

**Our tree before this work** had the generic MTP framework (NextN tensor names, `LLAMA_CONTEXT_TYPE_MTP`,
`LLM_GRAPH_TYPE_DECODER_MTP`, `common_speculative_impl_draft_mtp`, `--spec-type draft-mtp`, `-md`) but no
qwen4exp MTP graph, block loading, hidden-state tap or draft memory filter. `qwen4exp + draft-mtp` did nothing.

**Upstream.** ggml-org/llama.cpp#29761 "Qwen4Exp: add MTP" (am17an) merged 2026-10-01. It supersedes
#28243 (Unsloth, closed unmerged) and #27836 (draft). Unsloth's `qwen4exp/mtp` branch is the #28243 lineage.
Reported by the PR: 1.55x decode speedup, 0.640 acceptance (Qwen3.8-Flash-Next IQ4_XS, `--spec-draft-n-max 3`,
DGX Spark, greedy, 24 prompts over 7 categories; per category 0.58-0.78). That is measured with everything in
VRAM or unified memory, which is not our placement. (Unsloth speedup and acceptance figures from the Phase-1
web research were not traceable to a source and are dropped.)

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
final mixer. Verification is greedy: a draft token is kept only if it equals the target's own argmax at that
position. Bit-identical output to a plain greedy run is not established here, though: in our pairs the MTP output
and the no-MTP output diverge early on all three prompts (see "Output equivalence").

## The port in this tree

`a15fe6317` carries #29761 over, credited to upstream. This tree predates upstream #29751 (the QSA rewrite
that #29761 builds on), so `graph_mtp` was adapted to call `build_layer_attn` without the kpool input
(QSA here is derived from `mctx_hyb->get_idx()` and the layer's compress ratio), and two upstream hunks
with no target here were dropped. The one-line `speculative.cpp` change (the draft model is loaded from the
draft path rather than `params.model.path`) is a no-op cleanup for the in-tree callers that matches upstream,
not a bug fix. `4679665af` separately guards a pre-existing `-DGGML_SYCL=OFF` compile break in
`llama-model-loader.cpp`, and `825238c8c` guards three SYCL-only statics there that were unused in a CPU build.

**One deliberate deviation from upstream #29761.** `llama_memory_recurrent::is_empty()` carries
`assert(total_size() == 0)` ahead of `return ctxs_bufs.empty();`. In a Debug build that assert aborts in the
constructor for every non-empty recurrent or hybrid model, so it is dropped here (`825238c8c`). It is worth
reporting upstream.

Verified: a CPU-only build compiles and links `llama-speculative-simple`, and the lead ran it on the IQ3_XXS
target with both heads (see "Measured results"): speculation is active, with 50-96% of drafted tokens
accepted depending on prompt and n-max. **Not verified**: that the output equals plain greedy decoding (it
diverged from the baseline on all three prompts; see "Output equivalence"), the converter
(`conversion/qwen4exp.py`, needs the BF16 checkpoint) and anything on SYCL.

## Expected speedup at our expert-read cost (Phase-1 estimate; superseded for CPU decode by the measurements below)

Per-token decode `t1 = A + E`, where `A` is everything that is not the expert reads (GPU dense, GDN, QSA,
launch) and `E` is the host expert-read time. A round with `k` drafts verifies `n = k + 1` tokens:

    T_round = A + m(n) * E + k * D        D = one draft step, ~5 ms (MTP head on the GPU)
    tokens per round T = 1 + a + a^2 + ... + a^k        a = per-position acceptance
    speedup S = (A + E) * T / T_round

`m(n)` is the number of distinct experts the verify batch reads, relative to one token: 2.9 at n = 3 and 3.9
at n = 4 if routing were independent (512 experts, top 10); about 2.4 and 3.0 if routing is skewed.
Reported acceptance numbers are consistent with `a` of 0.75-0.8 (PR: 0.64 accepted/generated at n-max 3, which
implies a mean length of about 1 + 3 x 0.64 = 2.9 tokens per round if every round drafts 3; the PR does not
state a mean length).

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
    # THREADS defaults to 16 (-t/-tb/-td/-tbd); the binary's own default of 4 ran 0.225 t/s.  WARM=1 reads every shard through dd and prints fincore.
    # NOMMAP=1 (default) passes -lm none (-lzm on is kept; the lazy PLE table still works, RSS 47 GB): /models is bcachefs and does not keep an mmapped IQ3 file cached (2.78 TB read in 28 min; decode ~50x slower than with -lm none).

Target: the IQ3_XXS pair (76 GB, fits in page cache; the 177 GB Q8_0 would thrash it). Heads: ggml-org Q8_0
and Q4_0, in `/models/Qwen3.8-Flash-Next-MTP/`. Arms: n-max {2, 3} x prompts {code, chat, reasoning},
greedy, seed 42, 256 new tokens, in two sets: `base` (binary default p-min 0.00, drafts never filtered) and
`p05` (`--spec-draft-p-min 0.5`, Strata's keep-while-p>=0.5 filter, for the comparison). Built with `cmake -B build-cpu -G Ninja -DGGML_SYCL=OFF -DLLAMA_CURL=OFF`
and `--target llama-speculative-simple`. It is a model load: run it by hand, one process at a time. The
parser (`scripts/parse-qwen4exp-mtp-acceptance.py`, gate `test-qwen4exp-mtp-acceptance-parser`) rejects a
log with `n_drafted = 0` instead of reporting 0% acceptance, because that is speculation that never ran.

To turn the estimate table above into a number for SYCL, two more measurements are needed: a `llama-bench`
decode baseline on the SYCL build, and the GPU-busy versus CPU-expert-pool-busy split per token (that gives
`A` and `E`).

## Measured results (CPU-only, IQ3_XXS target, 2026-10-03)

Setup. Binary: CPU-only `build-cpu`, built from the C++ of `4679665af` (no C++ file changed between it and
`b8fedecc7`, which only touched scripts and docs; the later `825238c8c` changes nothing in a Release CPU
build). Target: ISTA-DASLab GSQ-RCO IQ3_XXS. Settings of every
arm: `-t/-tb/-td/-tbd 16`, `-lm none -lzm on` (RSS 47 GB), `-c 4096 -ub 512 -ngl 0`, 256 new tokens, greedy,
seed 42, prompts = the three pre-rendered files in `scripts/qwen4exp-mtp-prompts/` (code: a Python function with
doctests; chat: a three-day Lisbon itinerary; reasoning: a two-trains word problem, "reasoning-flavoured" only:
its prompt opens an empty `<think>` and the model closes it at once in both runs, so it is not a thinking trace). Host load average was
~36-57 during the arm sets and ~28-38 during the pairs (a codescout re-index plus ambient load). Data, all committed in `docs/backend/qwen4exp-mtp-data/`:
`acceptance-base.txt` and `acceptance-p05.txt` (the parser tables below), `pairs.txt` (parser `--pairs`),
`run-base.out`, `run-p05.out` and `run-pairs.out` (the run logs), and `raw-logs.tar.xz` (the 24 arm logs under
`out/` and the 6 pair logs under `pairs/`). The pairs were run with a one-off script whose flags match
`qwen4exp-mtp-acceptance.sh --pairs` (checked against its dry-run); `--pairs` is the reproducible form. Acceptance is deterministic (the same config run twice gave
identical `n_drafted`/`n_accept`); decode t/s is not (see Caveats).

**Terms.** `accept%` = `n_accept / n_drafted`. `mean_len` = tokens produced per verify round, i.e. 1 + accepted
drafts (a round is one verify batch; for `q4_n3_code`: 189 accepted / 71 rounds + 1 = 3.66, matching the
`#mean acc len` line). Pooled rows are token-weighted for accept% and a plain mean over the 3 prompts for
`mean_len`.

Per arm, base set (p-min 0.00, never filtered; from `run-base.out`):

| arm | n_drafted | n_accept | accept% | mean_len | acc/pos | decode t/s |
|---|---:|---:|---:|---:|---|---:|
| q8_n2_code | 180 | 169 | 93.89 | 2.88 | 0.98, 0.90 | 1.92 |
| q8_n2_chat | 230 | 143 | 62.17 | 2.24 | 0.73, 0.51 | 1.57 |
| q8_n2_reasoning | 185 | 166 | 89.73 | 2.78 | 0.94, 0.85 | 1.93 |
| q8_n3_code | 210 | 189 | 90.00 | 3.70 | 0.96, 0.91, 0.83 | 2.29 |
| q8_n3_chat | 311 | 155 | 49.84 | 2.49 | 0.75, 0.48, 0.26 | 1.30 |
| q8_n3_reasoning | 213 | 189 | 88.73 | 3.66 | 0.94, 0.90, 0.82 | 1.73 |
| q4_n2_code | 178 | 170 | 95.51 | 2.91 | 0.98, 0.93 | 2.01 |
| q4_n2_chat | 222 | 147 | 66.22 | 2.32 | 0.78, 0.54 | 1.51 |
| q4_n2_reasoning | 188 | 165 | 87.77 | 2.76 | 0.92, 0.84 | 2.06 |
| q4_n3_code | 211 | 189 | 89.57 | 3.66 | 0.96, 0.89, 0.82 | 2.30 |
| q4_n3_chat | 311 | 155 | 49.84 | 2.49 | 0.77, 0.46, 0.26 | 1.33 |
| q4_n3_reasoning | 225 | 185 | 82.22 | 3.47 | 0.92, 0.84, 0.71 | 1.95 |

Per arm, p-min 0.5 set (`--spec-draft-p-min 0.5`; from `run-p05.out`):

| arm | n_drafted | n_accept | accept% | mean_len | acc/pos | decode t/s (unreliable) |
|---|---:|---:|---:|---:|---|---:|
| q8_n2_p05_code | 177 | 169 | 95.48 | 2.88 | 0.98, 0.90 | 1.75 |
| q8_n2_p05_chat | 189 | 136 | 71.96 | 2.11 | 0.68, 0.43 | 1.77 |
| q8_n2_p05_reasoning | 177 | 164 | 92.66 | 2.74 | 0.90, 0.84 | 1.82 |
| q8_n3_p05_code | 205 | 190 | 92.68 | 3.71 | 0.97, 0.91, 0.83 | 2.20 |
| q8_n3_p05_chat | 227 | 148 | 65.20 | 2.35 | 0.69, 0.40, 0.26 | 1.71 |
| q8_n3_p05_reasoning | 202 | 188 | 93.07 | 3.61 | 0.96, 0.86, 0.79 | 2.37 |
| q4_n2_p05_code | 178 | 170 | 95.51 | 2.91 | 0.98, 0.93 | 2.33 |
| q4_n2_p05_chat | 190 | 140 | 73.68 | 2.19 | 0.73, 0.46 | 2.00 |
| q4_n2_p05_reasoning | 179 | 163 | 91.06 | 2.70 | 0.93, 0.77 | 2.89 |
| q4_n3_p05_code | 211 | 189 | 89.57 | 3.66 | 0.96, 0.89, 0.82 | 2.33 |
| q4_n3_p05_chat | 217 | 152 | 70.05 | 2.45 | 0.70, 0.49, 0.25 | 1.96 |
| q4_n3_p05_reasoning | 202 | 187 | 92.57 | 3.56 | 0.94, 0.86, 0.75 | 2.46 |

Pooled per head x n-max x p-min (3 prompts each, from the parser's group rows):

| head | n-max | p-min none: accept% / mean_len | p-min 0.5: accept% / mean_len |
|---|---:|---|---|
| q8 | 2 | 80.34 / 2.63 | 86.37 / 2.58 |
| q4 | 2 | 81.97 / 2.66 | 86.47 / 2.60 |
| q8 | 3 | 72.62 / 3.28 | 82.97 / 3.22 |
| q4 | 3 | 70.82 / 3.21 | 83.81 / 3.22 |

**Findings.**
- **The Q4_0 head equals the Q8_0 head within noise; use Q4_0 (2.2 GB against 4.1 GB).** Pooled accept% differs by
  at most 1.8 points (n2 base 81.97 vs 80.34; n3 base 70.82 vs 72.62; p05 86.47 vs 86.37 and 83.81 vs 82.97) and the
  sign flips between rows. The largest single-arm gap is 6.5 points (n3 reasoning, 88.73 for Q8 vs 82.22 for Q4);
  the n3 chat arms are identical (155/311 for both heads).
- **p-min 0.5 raises acceptance and does not lengthen rounds.** Pooled accept% goes 80-82% to ~86% at n2 and
  71-73% to ~83-84% at n3, while pooled `mean_len` stays at 2.58-2.66 (n2) and 3.21-3.28 (n3); at n2 it is a
  little lower (-0.05 to -0.06). The gain is wasted drafts removed, not more accepted tokens: `n_drafted`
  for the n3 chat arms falls from 311 to 227 (Q8) and 217 (Q4), about -27% to -30%, with `n_accept` 155 to 148/152.
  So it saves draft compute, and nothing else. Chat benefits most (accept% n2 62-66 to 72-74; n3 50 to 65-70).
  Whether the saved draft steps show up as t/s is not measured, because the p05 decode t/s is unreliable (Caveats).
- **Acceptance depends strongly on the prompt.** At n2: code 93.9-95.5% and reasoning 87.8-89.7% (base), 91.1-95.5%
  (p05), against chat 62.2-66.2% (base), 72.0-73.7% (p05). At n3 chat is the outlier, 49.8% base and 65.2-70.1%
  with p-min 0.5; code and reasoning stay at 82-93%. Position 1 accepts 0.90-0.98 on code and reasoning;
  chat is 0.68-0.78 at position 1 and falls to 0.25-0.26 at position 3.
- **n3 buys longer rounds only where acceptance is high.** `mean_len` n3 over n2: code 3.66-3.70 vs 2.88-2.91,
  reasoning 3.47-3.66 vs 2.76-2.78, chat 2.49 vs 2.24-2.32 (base). Base-set decode t/s agrees (code n3 2.29-2.30
  against n2 1.92-2.01; chat n3 1.30-1.33 against n2 1.51-1.57, so n3 is slower on chat), but under load; see
  Caveats.

**Decode speedup against no MTP (CPU decode, Q4_0 head, n-max 2).** One interleaved pair per prompt on the same
host state (order AB for code, BA for chat, AB for reasoning), same flags, target loaded identically. Baseline:
`llama-completion -no-cnv`, t/s from `common_perf_print` "eval time"; MTP: `llama-speculative-simple`, t/s from
"decoded ... speed". Logs: `pairs/base_<prompt>_1.log`, `pairs/mtp_<prompt>_1.log`.

| prompt | baseline t/s (ms/token) | MTP t/s | speedup (this pair) | MTP t/s, same config in the arm run | speedup using that |
|---|---|---:|---:|---:|---:|
| code | 1.28 (782.8) | 2.24 | 1.75x | 2.01 | 1.57x |
| chat | 1.39 (717.4) | 2.13 | 1.53x | 1.51 | 1.08x |
| reasoning | 1.39 (719.1) | 2.06 | 1.48x | 2.06 | 1.48x |

The pair figures, 1.48-1.75x, are what the interleaved protocol produced. The last two columns are the same MTP
configuration (q4_n2, base) as measured in the arm set under a different load phase: identical acceptance (170,
147, 165 accepted), but the pair-run rate is 12% above the arm-run rate for code, **41% above for chat**, and equal for reasoning. So one
run's t/s carries roughly that much load noise, and the chat speedup in particular could be anywhere from 1.08x to
1.53x on this evidence; the lower number is the one to quote if only one is quoted. All of these are well above the
Phase-1 estimate for IQ3_XXS (0.72-1.24x).

*Hypothesis for the gap (not tested):* the estimate assumed the verify batch reads `m(n)` x the experts of one
token. On this CPU path the baseline decodes at ~0.75 s/token for roughly 0.75 GiB of expert reads per token, about
1 GB/s, so decode is limited by compute and per-token overhead (IQ3 dequant, many small expert matmuls, thread
sync), not by DRAM bandwidth. A 2-3 token batch can reuse dequantized expert rows across tokens that route to the
same expert and pays the per-layer overhead once, so it costs much less than 2-3 single-token steps. If so, the
estimate's `E` term was pessimistic for this path and it says nothing about a GPU or hybrid path, where the
cost structure differs.

**Output equivalence: not established, and the speedups above are for numerically divergent output.** Speculative
decoding with greedy verification is meant to reproduce a plain greedy run. Here it does not, on any of the three
prompts: in each pair the MTP text and the no-MTP text agree for the first 130-200 characters and then differ
(`pairs/base_<prompt>_1.log` against `pairs/mtp_<prompt>_1.log`). The first divergent token, with what the baseline
(X) and the MTP run (Y) each emitted:

| prompt | text both runs agree on, up to | X (baseline) | Y (MTP) |
|---|---|---|---|
| code | "...closed intervals,\n    merge" | ` any` | ` all` |
| chat | "...the light is soft and golden." | ` This` | `\n\n` |
| reasoning | "...analyze the situation before the second train departs" | `\n` | `\n\n` |

All three are near-tie style choices (word, sentence-versus-paragraph break), which fits floating-point
differences between a batch-1 and a batch-3 forward pass on a near-tied logit pair: the MTP verify pass computes
the target's logits for the draft positions as one small batch, while the baseline computes them one token at a
time, and the CPU kernels do not promise bit-identical results across batch widths. It is equally consistent with
a verify/rollback defect in the port (target GDN-state rollback, `seq_rm`, QSA indexer positions after a rejected
draft), and the logs cannot tell the two apart. Consequently:
- this document does not claim that MTP output equals plain greedy decoding;
- the speedups above compare two different texts of the same length, not the same tokens produced faster;
- acceptance rates describe agreement between the MTP head and the target's batched argmax.

*Discriminator (CPU-only, existing `llama-completion`, `-n 1`; no new tool).* For each prompt, feed the prompt plus
the agreed text above as a new prompt and let the baseline emit one token, at `-ub 512` (the prefix is one batched
pass, like a verify batch) and at `-ub 1` (batch-1, the decode shape). The `-ub 1` run is the positive control and
must print X. If `-ub 512` prints Y, batch-width numerics on a near tie explain the divergence for that prompt. If
it prints X, batch width alone does not explain it and the verify/rollback path stays suspect. Optionally add
`--logit-bias <id of Y>+<delta>` at `-ub 1` and take the smallest delta that flips X to Y as the logit margin.
Run: `scripts/qwen4exp-mtp-divergence-probe.sh` (the prefixes are `docs/backend/qwen4exp-mtp-data/disc/*_prefix.txt`;
the lead's run, with the build-cpu binaries, is `disc/run.out`, rc 0, Shmem flat at 12.24 GB before and after).

| prompt | X (baseline in the pair) | -ub 512 printed | -ub 1 printed | Y (MTP in the pair) |
|---|---|---|---|---|
| code | ` any` | ` any` | ` any` | ` all` |
| chat | ` This` | ` This` | ` This` | `\n\n` |
| reasoning | `\n` | `\n` | `\n` | `\n\n` |

Reading. The positive control holds (`-ub 1` reproduces X on all three prompts, so the prefixes reconstruct the
baseline). `-ub 512` also prints X on all three: widening the pass from one token to the whole prefix does not
move the pick to MTP's token. So batch-width numerics on a near tie, at least of the kind a 512-wide prefill
exercises, do not explain Y, and the weight shifts toward the MTP verify/rollback path (target GDN-state
rollback, `seq_rm`, QSA indexer positions after a rejected draft). **This is not a proven defect.** The probe
varies the prefill batch width over a fresh context. It does not reproduce the exact verify shape (a 3-token
decode step over a KV and recurrent state that has been through accept/reject cycles), so a numerics
difference that only appears with that state history is not excluded.

Which rollback path runs. Not checkpoints: `examples/speculative-simple` restores the target from a checkpoint only
when `common_context_can_seq_rm(ctx_tgt)` is `COMMON_CONTEXT_SEQ_RM_TYPE_FULL` and logs "speculative decoding will
use checkpoints" at INFO, and none of the `-lv 4` logs contains that line (other INFO lines such as "encoded" do
appear). The logs show the bounded-rollback path instead: the target context prints `n_rs_seq = 2` (3 for the
n-max 3 arms; `llama_memory_recurrent: ... 2 rs_seq`, 337.71 MiB against 450.28 MiB at 3) and
`common_conte: the context supports bounded partial sequence removal`. After each verified round
`llama_memory_seq_rm(ctx_tgt, seq_id, n_past, -1)` (`speculative-simple.cpp`, "clear kv cache from any extra
tokens") rewinds the recurrent state by up to `n_rs_seq` positions from saved per-step planes, and the compress-state
planes (`dsv4_build_comp_plan`, rollback `<= n_rs_seq`) carry the same bound. That is the code the divergence
points at, and it is shared with `llama_memory_recurrent`, which this port touched (the draft context prints
`n_rs_seq = 0` and a 0 MiB recurrent memory: the `is_empty()` case).

Next discriminating runs (not run). (a) `--spec-draft-n-max 1` on the same prompts: if the output still diverges
from greedy, the 3-token verify batch is not needed for the divergence. (b) Locate the first divergence relative
to the first rejected draft. This cannot be done from the existing logs: they are `-lv 4` and carry only the
aggregate counters (for `mtp_code_1`: 89 rounds, 178 drafted, 170 accepted, so 8 draft tokens were rejected
somewhere). Per-round rejections are logged at DEBUG, which is `-lv 5`: `accepted <k>/<n> draft tokens, the last
target token is: (<id>)` after each round (`speculative-simple.cpp:324`, a rejection is k < n; the generated text
of the round is printed just before it). The `partial acceptance ... restoring checkpoint` line (`:268`) belongs to the
checkpoint path and is not expected here. So (b) needs one `-lv 5` re-run of an MTP arm, for example
`mtp_code_1`'s command with `-lv 5`; the divergence then sits in the round whose text contains the first
differing character. If no rejection precedes it, rollback is implicated directly.

**Compared with upstream's report** (PR #29761: 0.640 acceptance, 1.55x, IQ4_XS, n-max 3, all-VRAM DGX Spark, 24
prompts over 7 categories): our pooled n3 acceptance without p-min is 70.8-72.6% over 3 prompts only, higher than
0.640, but the spread is large (chat 49.8%, code and reasoning 82-90%) and the target quant, prompts and head
differ, so this is not a like-for-like confirmation. Our 1.48-1.75x is at n-max 2 on CPU and the PR's 1.55x is at
n-max 3 on VRAM-resident weights: similar magnitude, different configuration, no direct comparison.

**Caveats.**
- CPU-only, IQ3_XXS target, no GPU involved. Hybrid (device-resident dense + host experts) and SYCL are not measured.
- One pair per prompt for the speedup; the run-to-run spread above (up to 41% on chat) is larger than the
  differences between most arms. Acceptance is exact and repeatable; decode t/s is not.
- Load average ~28-57 over the campaign. A build ran concurrently with the p05 set, so p05 decode t/s is
  not comparable with the base set and is marked unreliable above. The base-set t/s are also load-affected.
- The pair baseline and the MTP arm are different binaries (`llama-completion` vs `llama-speculative-simple`)
  and take t/s from different timers; the same flags and the same host state, not the same code path, make them
  comparable.
- Three prompts, 256 tokens each, one seed, pre-rendered chat-template prompts fed raw; pooled figures
  inherit that sample size.

## Decision

Acceptance is no longer the open question: at n-max 2 it is 62-96% depending on the prompt (80-82% pooled,
~86% with p-min 0.5) and the Q4_0 head is as good as the Q8_0 head. The CPU decode speedup measured here is
1.48-1.75x (single interleaved pairs, load-affected; chat as low as 1.08x on a repeat; and for output that diverges from the baseline, see "Output equivalence"), well above the Phase-1
estimate for this target.

Still holding the SYCL wiring, because the measurement does not cover the placement we run. What is left to
establish, and where the hold is decided: (1) the GPU-busy versus CPU-expert-pool-busy split per token (`A` and
`E`) on the SYCL build; (2) whether the host-resident-expert verify batch (k+1 tokens through the CpuExpertPool
path) is cheaper than k+1 single-token steps: the CPU result is evidence for that only if the hypothesis above
(batched expert matmuls reuse dequantized rows) is right and the hybrid path's host experts run through the
same batched CPU kernels, and neither is tested; (3) the SYCL risks listed above (second model and context on
one device, draft-context compute buffer); (4) whether the output divergence from the baseline is numerics or a
port defect ("Output equivalence"): the discriminator ruled out prefill batch-width numerics and left the verify
and rollback path suspect but unproven, so MTP is not to be treated as output-equivalent, and the speedups stay
"two different texts", until the rollback path is cleared (next runs (a) and (b) there). If (2) and (4) come
out well, the working recommendation for a first SYCL
attempt is the Q4_0 head and n-max 2. `--spec-draft-p-min 0.5` is worth trying for chat-like workloads, but that
rests on acceptance alone: its effect on decode t/s was not measured (the p05 timings are unreliable). Keep the
`S > ~1.2x` bar for deciding whether to wire it.
