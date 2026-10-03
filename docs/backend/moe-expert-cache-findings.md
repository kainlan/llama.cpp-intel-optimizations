# What an adaptive VRAM expert cache is worth: routing-trace simulation for Qwen3.8 Flash-Next

Tracker: `llama.cpp-05mh`. Model: Qwen3.8 Flash-Next (`qwen4exp`): 48 MoE layers,
512 experts per layer, top-10 routing. Tools: `examples/moe-trace` (records the
router's choices, `llama-moe-trace`) and `scripts/moe-cache-sim.py` (replays a trace
against a cache policy). Data: [`moe-expert-cache-data/`](moe-expert-cache-data/).

**What this is.** A prediction of the hit rate and of the bytes the CPU must still read
per token if an adaptive cache kept part of each layer's experts in VRAM, made from
recorded routing before anyone builds the cache. **What it is not.** It models hit
rate, not time; the routing was captured on the CPU from the IQ3_XXS file; there are
10 traces from three kinds of text. Section 9 lists what that does and does not
support. Every number below is a row of a committed CSV, named under each table;
"token-weighted" means `sum(value * tokens) / sum(tokens)` over the held-out test
groups of one cell (`moe-expert-cache-data/analyze.py` computes it).

## 1. Result in brief

All figures are decode, leave-one-prompt-set-out, token-weighted, IQ3_XXS routing and
sizes unless a line says Q8_0.

1. **Adaptive beats everything that is achievable, most clearly when the cache is
   small.** IQ3_XXS with 7 GiB resident (18 % of the experts): hit rate
   0.71 against 0.60 for first-touch and 0.35 for a static profile trained on other
   text; CPU bytes per token 0.232 GiB against 0.322 and 0.510. Q8_0 with 7 GiB (5.7 %
   of the experts): 0.47 against 0.18 and 0.13.
2. **A shipped static profile is not worth it.** A profile trained on two other prompt
   sets scores 0.12 to 0.79 across 2 to 24 GiB for IQ3_XXS, below first-touch (which needs
   no training) at every budget. For Q8_0 it is below first-touch from 7 GiB up (0.13
   against 0.18 at 7 GiB); at 2 and 4 GiB both are near zero and within 0.01 (0.04 and
   0.03; 0.08 and 0.08).
3. **With a large cache, first-touch is nearly as good.** At 14 GiB of IQ3_XXS (36 % of
   the experts) adaptive is 1.1 points above first-touch (0.84 against 0.83); at 2 to
   4 GiB the gap is 20 to 29 points, and it widens again at 24 GiB (3.6 points; a hypothesis
   is that first-touch never evicts, which is not tested). For Q8_0 the gap stays 9 to
   29 points up to 24 GiB.
4. **Adaptive can exceed an in-sample static profile.** Against the profile fitted on the
   held-out set itself, adaptive is ahead at 2 and 4 GiB (0.45 against 0.37, 0.59 against
   0.55), level at 7 GiB (0.71 against 0.72) and behind from 10 GiB. It is **not** ahead
   of a profile fitted to the single test trace (0.49 and 0.67 at 2 and 4 GiB), which is
   a stricter and unachievable bound (section 3).
5. **Score decode and prefill separately; the prefill policy is a different, open
   question.** With prompt rounds included (`phase=all`) the aggregate collapses (IQ3_XXS
   7 GiB: 0.34, below first-touch's 0.47) because a 512-token ubatch is not a decode
   round: the rule's counters and its cap of 96 swaps per 4 rounds are in decode units.
   Freezing the swaps in prefill was run as an arm and is not clearly better: prefill
   hit rate falls (0.220 against 0.293 at 7 GiB, long set), and decode after it moves
   little (0.716 against 0.703; section 4). Whether prefill should swap, seed the
   counters, or not take part is not settled by this data.
6. **The parameters Strata chose are near the optimum.** What matters is the adaptation
   rate: the cap of 96 swaps per 4 rounds binds at small budgets (21.5 of a possible 24
   swaps per token at 4 GiB), and a land delay of 4 rounds or more cuts the number of
   swaps by roughly half at small budgets (21.5 to 11.9 per token at 4 GiB; at 14 GiB
   only 10.4 to 9.9) because of the in-flight skip rule: with the rule disabled the
   count stays at 21.5 (section 5).
7. **Swap traffic is small next to what it saves.** IQ3_XXS 7 GiB: 29 MiB per token
   moved, 284 MiB per token fewer CPU reads than the static profile (not than no cache;
   9.7 MiB saved per MiB moved); about 1.2 GB/s at 40 tokens/s. Q8_0 7 GiB: 112 MiB per token moved, about 4.7 GB/s at
   40 tokens/s, 7.2 MiB saved per MiB moved (section 6).
8. **What it buys in tokens/s, as the CPU-limited rate at the median CPU bandwidth**
   (estimate, section 7): IQ3_XXS 7 GiB adaptive at 21 GB/s of CPU expert bandwidth gives
   84 tokens/s, against 38 for static and 25 with no cache. Q8_0 7 GiB at 30 GB/s: 23
   against 14 and 12. This ignores everything but the CPU's expert reads.

One open question is not a measurement: the placement ruling says placement "is made
once, in the planning pass". An adaptive cache re-places experts while running. Section
8 sets out the two readings and leaves the decision to the owner.

## 2. Method

### Traces

Ten traces, all recorded with `llama-moe-trace` built from master `f2d0e3dad`, on the
CPU from the IQ3_XXS target (GSQ-RCO), `-t 16 -lm none`, greedy decoding (no sampling), 256 decode tokens each, `n_batch` 2048, `n_ubatch` 512.
`traces-manifest.csv` lists each file with its sha256; **the trace files themselves are
not committed** (the two long ones are 10.1 and 8.2 MB).

| set | traces | prompt tokens | what the prompts are |
|---|---|---|---|
| chat | chat-0 .. chat-3 | 39 to 46 | ChatML questions |
| code | code-0 .. code-3 | 62 to 104 | ChatML coding requests |
| long | long-0, long-1 | 10218, 8218 | ChatML wrapped around a cut of `docs/backend/SYCL.md` and a cut of `build_moe_ffn` in `src/llama-graph.cpp`; `-c 12288` |

Decode tokens: chat 1024, code 1024, long 512, total 2560. For `phase=all`, the long
prompts add 18436 prefill tokens and the short ones 487 (21483 tokens in all:
`summary-curves.csv`, `tokens`).
`n_prompt` and the step counts are in `traces-manifest.csv`.

### Policies (all per layer; slots = floor(budget bytes per layer / expert bytes))

| policy | what it does |
|---|---|
| static | resident set = top-S experts by routing count over the training traces; scored on held-out traces |
| oracle | the same, profiled on the test traces themselves (in-sample; not achievable) |
| first-touch | starts empty, admits each missed expert until the layer is full, never evicts |
| adaptive | starts from the static profile and follows Strata's `adapt()` (from the Strata source, research-strata `src/program/generate.cpp:4794-4860`, not part of this tree): usage counts per routed (token, expert); every 4 rounds candidates with usage >= 2.0 are paired with the coldest residents while gain >= 1.5; the 96 largest gains across all layers swap; the victim is evicted at once, the incoming expert lands 1 round later; usage times 0.7 after each adaptation; an adaptation is skipped (no decay) while swaps are in flight |

A round is one step of the trace: one decoded token, or one prefill ubatch. Each test
trace starts with fresh caches. "Leave-one-set-out" (`--loo-by set`) trains on the other
two sets and tests on the held-out set, so the profile is always from other kinds of
text. "Leave-one-id-out" trains on the other nine traces, which includes the same set.
`oracle` in leave-one-set-out is a profile of the whole held-out set pooled; in
leave-one-id-out it is a profile of the single trace.

### Sizes and budgets

Expert bytes come from the GGUF tensor tables (gate + up + down, divided by 512;
`expert-sizes.csv`):

| format | per layer | mean | all experts (48 x 512) |
|---|---|---|---|
| IQ3_XXS (mixed per layer) | 1.245 to 2.222 MiB | 1.665 MiB | 39.97 GiB |
| Q8_0 | 4.980 MiB | 4.980 MiB | 119.53 GiB |

Budgets are 42.667 to 512 MiB per layer, labelled here by the nominal total (x 48):
2, 4, 7, 10, 14, 18, 24 GiB. Slots are floored, so the resident bytes are slightly
below the label (IQ3_XXS 1.96 to 23.97 GiB; Q8_0 1.87 to 23.81 GiB). The resident
fraction of all experts is 5.1 % to 62.3 % for IQ3_XXS and 1.6 % to 19.9 % for Q8_0
(`summary-curves.csv`, `resident_fraction_of_experts`).

**The Q8_0 curves are IQ3_XXS routing applied to Q8_0 sizes.** No Q8_0 capture was made.
That quantisation changes routing little is an assumption, not measured.

## 3. Decode hit rate and CPU bytes

Source: `summary-curves.csv`, `phase=decode`, `loo_by=set`; per-set rows in
`curves-decode-by-set.csv`.

**IQ3_XXS, hit rate (token-weighted)**

| policy | 2 GiB | 4 | 7 | 10 | 14 | 18 | 24 |
|---|---:|---:|---:|---:|---:|---:|---:|
| resident fraction of experts | 5.1 % | 10.3 % | 18.1 % | 25.9 % | 36.3 % | 46.7 % | 62.3 % |
| adaptive | 0.45 | 0.59 | 0.71 | 0.78 | 0.84 | 0.89 | 0.94 |
| first-touch | 0.16 | 0.39 | 0.60 | 0.73 | 0.83 | 0.88 | 0.90 |
| static (other sets) | 0.12 | 0.22 | 0.35 | 0.45 | 0.57 | 0.67 | 0.79 |
| oracle (held-out set pooled) | 0.37 | 0.55 | 0.72 | 0.83 | 0.92 | 0.96 | 0.99 |

**IQ3_XXS, CPU GiB read per token** (all experts on the CPU: 0.781)

| policy | 2 GiB | 4 | 7 | 10 | 14 | 18 | 24 |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive | 0.439 | 0.326 | 0.232 | 0.175 | 0.124 | 0.089 | 0.051 |
| first-touch | 0.667 | 0.490 | 0.322 | 0.220 | 0.137 | 0.095 | 0.075 |
| static | 0.688 | 0.610 | 0.510 | 0.429 | 0.338 | 0.261 | 0.171 |

**Q8_0 sizes, IQ3_XXS routing, hit rate**

| policy | 2 GiB | 4 | 7 | 10 | 14 | 18 | 24 |
|---|---:|---:|---:|---:|---:|---:|---:|
| resident fraction of experts | 1.6 % | 3.3 % | 5.7 % | 8.2 % | 11.5 % | 15.0 % | 19.9 % |
| adaptive | 0.25 | 0.37 | 0.47 | 0.55 | 0.62 | 0.68 | 0.73 |
| first-touch | 0.03 | 0.08 | 0.18 | 0.31 | 0.44 | 0.55 | 0.65 |
| static (other sets) | 0.04 | 0.08 | 0.13 | 0.19 | 0.25 | 0.31 | 0.38 |
| oracle (held-out set pooled) | 0.16 | 0.28 | 0.40 | 0.49 | 0.59 | 0.67 | 0.76 |

**Q8_0, CPU GiB read per token** (all experts on the CPU: 2.335)

| policy | 2 GiB | 4 | 7 | 10 | 14 | 18 | 24 |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive | 1.760 | 1.475 | 1.236 | 1.058 | 0.890 | 0.756 | 0.623 |
| first-touch | 2.265 | 2.157 | 1.914 | 1.620 | 1.305 | 1.062 | 0.825 |
| static | 2.235 | 2.142 | 2.023 | 1.902 | 1.756 | 1.617 | 1.452 |

**Reductions in CPU bytes per token by adaptive**, from the same rows: against
first-touch 33 % at 4 GiB, 28 % at 7, 9 % at 14 (IQ3_XXS) and 32 %, 35 %, 32 % (Q8_0);
against a static profile 47 %, 54 %, 63 % (IQ3_XXS) and 31 %, 39 %, 49 % (Q8_0).

**The held-out sets agree on the ordering.** IQ3_XXS hit rates by held-out set (each is
one fold; chat and code have 1024 test tokens, long 512):

| policy | 4 GiB chat / code / long | 7 GiB chat / code / long | 14 GiB chat / code / long |
|---|---|---|---|
| adaptive | 0.63 / 0.55 / 0.59 | 0.73 / 0.68 / 0.71 | 0.86 / 0.83 / 0.86 |
| first-touch | 0.40 / 0.39 / 0.37 | 0.61 / 0.61 / 0.57 | 0.85 / 0.82 / 0.83 |
| static | 0.17 / 0.22 / 0.32 | 0.28 / 0.35 / 0.48 | 0.49 / 0.59 / 0.71 |
| oracle | 0.55 / 0.54 / 0.60 | 0.73 / 0.70 / 0.76 | 0.93 / 0.89 / 0.93 |

The long set behaves like chat and code: its adaptive hit rate (0.59, 0.71, 0.86) lies
between theirs at 4 and 7 GiB and is level with chat's at 14 GiB (0.857 against 0.855). Its static profile does better (0.32, 0.48, 0.71 against
0.17 to 0.22, 0.28 to 0.35, 0.49 to 0.59), plausibly because the other sets resemble its
text more than they resemble each other; the data do not test that.

**Static profiles transfer poorly across kinds of text.** Leave-one-set-out static is
0.35 at 7 GiB; leave-one-trace-out, where the same set's other traces are in training,
is 0.44; the in-sample profile of the pooled set is 0.72 and of the single trace 0.82
(`summary-curves.csv`, `loo_by=id`).

**Adaptive against the oracle, stated exactly.** The held-out-set oracle is fitted to
the test tokens, and adaptive is ahead of it at small budgets. A hypothesis for why:
temporal locality (the experts used in the last few tokens predict the next ones) is
worth more than a long-run frequency profile pooled over several prompts when only 5 to
10 % of the experts fit. The leave-one-trace-out oracle below weakens it: a profile of
the single trace, which has that trace's own locality built in, beats adaptive, so the
pooled profile may simply be a worse fit than a per-prompt one. This was not tested
further. Adaptive is not ahead of a profile fitted to one trace: leave-one-trace-out IQ3_XXS gives adaptive 0.45 / 0.60 /
0.72 against that oracle's 0.49 / 0.67 / 0.82 at 2 / 4 / 7 GiB. The per-trace oracle
overfits 256 tokens and cannot be shipped; it bounds what a static profile could do with
perfect knowledge of the next prompt.

**Spread over traces** (leave-one-trace-out, IQ3_XXS adaptive, min to max of the ten
traces): 0.55 to 0.67 at 4 GiB, 0.68 to 0.76 at 7 GiB, 0.84 to 0.88 at 14 GiB
(`summary-curves.csv`, `hit_rate_min_group`, `hit_rate_max_group`).

## 4. Prefill (`phase=all`)

Source: `summary-curves.csv`, `phase=all`; `long-0-rounds.csv`, `long-0-windows.csv`,
`long-0-blocks.csv` (`prefill-rounds.py`).

Hit rate with every round scored, prefill included, against decode only:

| format, policy | decode 4 / 7 / 14 GiB | all rounds 4 / 7 / 14 GiB |
|---|---|---|
| IQ3_XXS adaptive | 0.59 / 0.71 / 0.84 | 0.25 / 0.34 / 0.53 |
| IQ3_XXS first-touch | 0.39 / 0.60 / 0.83 | 0.26 / 0.47 / 0.74 |
| IQ3_XXS static | 0.22 / 0.35 / 0.57 | 0.14 / 0.23 / 0.44 |
| IQ3_XXS oracle | 0.55 / 0.72 / 0.92 | 0.46 / 0.61 / 0.81 |
| Q8_0 adaptive | 0.37 / 0.47 / 0.62 | 0.16 / 0.20 / 0.27 |
| Q8_0 first-touch | 0.08 / 0.18 / 0.44 | 0.04 / 0.07 / 0.30 |

The aggregate is dominated by prefill: 18436 of 21483 tokens (86 %) are prompt tokens,
and 97.6 % of the 10474 tokens scored for long-0 are (10218 of 10474). So the
all-round number is a prefill number, and in prefill adaptive is behind first-touch.

The replay of one trace (long-0 held out, trained on chat and code, IQ3_XXS 7 GiB, 4441
resident slots, 276 rounds; the script asserts its totals equal the simulator's) shows
the cause:

| measure | value |
|---|---|
| prefill rounds | 20 (19 of 512 tokens, one of 490) |
| mean fraction of a layer's 512 experts touched by one prefill round | 0.617 to 0.775 (decode round: 0.0195, i.e. 10 of 512) |
| usage counters at or above `min_count` 2.0 at the first adaptation (round 3) | 18567 of 24576 (76 %); 22048 (90 %) at the fifth |
| candidate swaps that pass the margin at the first adaptation | 3401 |
| swaps taken (cap 96) | 96 per adaptation: 480 in the 5 prefill adaptations, 10.8 % of the 4441 slots |
| adaptations taken at the cap | all of them up to round 83; round 87 is the first shortfall (91); 96 recurs later, in 30 of the 69 adaptations (all 5 in prefill, 25 of 64 in decode) |

**Why a 512-token ubatch fools the swap rule.** The constants (`min_count` 2.0, margin
1.5, decay 0.7, 96 swaps, every 4 rounds) were chosen for rounds that add 10 counts per
layer. A prefill round adds 5120. Three consequences, each in the table:

1. The two gates stop discriminating. After four ubatches three quarters of all counters
   already clear `min_count`, and thousands of candidates clear the margin, so the rule
   reduces to "swap the 96 largest count differences", chosen from the routing of the
   prompt so far (the counters decay by 0.7 per adaptation, so recent chunks weigh most).
2. The cadence is per round, not per token. Four prefill rounds are 2048 tokens; the cap
   of 96 swaps per adaptation moves at most 2.2 % of the slots every 2048 tokens, so the
   cache cannot follow a 10K-token prompt (5 adaptations, 480 swaps).
3. A prefill round touches 62 to 78 % of a layer's experts, so a resident subset of 18 %
   serves only the popular part of each round: on the long set the in-sample oracle
   reaches 0.60 and first-touch 0.46 over all rounds (`curves-all-by-set.csv`, 7 GiB; the
   0.45 in the table below is long-0's prefill alone).

Per policy on long-0 (`long-0-blocks.csv`, token-weighted within the block):

| rounds | adaptive | static | first-touch | first-touch, admitted at end of round |
|---|---:|---:|---:|---:|
| prefill (20 rounds, 10218 tokens) | 0.294 | 0.216 | 0.451 | 0.421 |
| decode rounds 0-31 | 0.596 | 0.534 | 0.421 | 0.421 |
| decode rounds 96-127 | 0.752 | 0.414 | 0.469 | 0.469 |
| decode rounds 224-255 | 0.741 | 0.384 | 0.387 | 0.387 |
| decode, all 256 rounds | 0.702 | 0.446 | 0.399 | 0.399 |

- Adaptive is better than static in prefill (0.294 against 0.216) and far from
  first-touch. First-touch fills its 92 slots per layer from the first ubatch in the
  order experts are met, which favours the frequent ones, and reaches 0.42 to 0.45 at
  once.
- **A modelling caveat that favours first-touch.** The simulator admits a missed expert
  immediately, so later tokens of the same 512-token round already hit. A copy cannot
  land in the middle of a ubatch. The last column admits at the end of the round
  instead: 0.421 against 0.451 in prefill. First-touch is ahead of adaptive in prefill
  either way (the column is identical in decode, where a round is one token).
- **The decode that follows is not visibly harmed.** The decode part of this trace scores
  0.702 for adaptive, 0.446 static, 0.399 first-touch, after the prefill's 480 swaps. The
  control is the same trace, same training, decode rounds only, no prefill in the run:
  adaptive 0.705 (last row of `long-0-blocks.csv`; mean of the four quarters in
  `decode-quarters.csv`). So the prefill swaps cost 0.003 here. First-touch loses a lot
  from being filled by the prompt (0.399 after prefill; its decode-only quarters average
  0.562 on long-0), which is the same effect seen from the other policy.
- **Adaptive's rise over the 256 decode rounds is real on the short traces and is
  confounded on the long ones.** Decode-only hit rate per 64-round quarter at 7 GiB
  (`decode-quarters.csv`, leave-one-set-out, IQ3_XXS):

  | held out | policy | Q1 | Q2 | Q3 | Q4 |
  |---|---|---:|---:|---:|---:|
  | chat | adaptive | 0.568 | 0.750 | 0.804 | 0.813 |
  | chat | first-touch | 0.631 | 0.606 | 0.600 | 0.585 |
  | chat | static | 0.363 | 0.246 | 0.262 | 0.260 |
  | code | adaptive | 0.577 | 0.704 | 0.710 | 0.723 |
  | code | first-touch | 0.623 | 0.630 | 0.632 | 0.549 |
  | code | static | 0.397 | 0.386 | 0.345 | 0.279 |
  | long | adaptive | 0.654 | 0.729 | 0.750 | 0.724 |
  | long | first-touch | 0.695 | 0.608 | 0.505 | 0.492 |
  | long | static | 0.568 | 0.495 | 0.424 | 0.425 |

  On chat and code adaptive keeps climbing through the quarters, so a 256-token trace
  understates the steady state for such sessions (by how much is not measured). On the
  long set it gains in the second quarter and is flat after (long-0 0.652 / 0.714 /
  0.704 / 0.751; long-1 0.656 / 0.744 / 0.797 / 0.697). The rise through the 32-round
  blocks of the phase=all long-0 table above (0.596, then 0.71 to 0.77) is not evidence
  of continued learning: its counters start from the prefill, which decays over about 64
  rounds (0.7 per 4 rounds), and the decode-only run of the same trace is flat after the
  first quarter.

**Arms for what the cache does in prefill.** The simulator has `--prefill-swaps
on|off|seed` (`prefill-arms.py`, `prefill-arms.csv`, `summary-prefill-arms.csv`): `on`
is the default; `off` scores prefill rounds but neither counts nor swaps in them (the
cache stays at the static profile until decode, and the cadence counts only the
remaining rounds); `seed` counts usage in prefill and decays it, without swapping.
Long set held out, IQ3_XXS, prefill hit rate / decode hit rate, token-weighted. The
decode share is the full run's pairs minus a run cut to the prefill rounds; the two
shares recombine to the all-rounds rate in the file.

| arm | 4 GiB | 7 GiB | 14 GiB |
|---|---|---|---|
| adaptive, swaps on | 0.212 / 0.550 | 0.293 / 0.703 | 0.490 / 0.868 |
| adaptive, frozen (`off`) | 0.137 / 0.588 | 0.220 / 0.716 | 0.425 / 0.855 |
| adaptive, `seed` | 0.137 / 0.552 | 0.220 / 0.703 | 0.425 / 0.868 |
| static (reference) | 0.137 / 0.310 | 0.220 / 0.466 | 0.425 / 0.697 |
| first-touch (reference) | 0.250 / 0.191 | 0.463 / 0.358 | 0.730 / 0.616 |

Q8_0, same cut: on 0.132 / 0.293, 0.161 / 0.398, 0.227 / 0.587; frozen 0.054 / 0.337,
0.083 / 0.451, 0.152 / 0.620 (4, 7, 14 GiB).

What the arms say, and do not say:

- A frozen cache in prefill scores exactly the static profile there (the `off` row
  equals the static row), 0.07 below swapping at 7 GiB. Freezing is not obviously better
  in prefill, and it is below first-touch (0.463 at 7 GiB) in either case.
- Decode after a frozen prefill is slightly better than after a swapping one at 4 GiB
  (0.588 against 0.550) and 7 GiB (0.716 against 0.703; Q8_0 0.451 against 0.398), and
  slightly worse at 14 GiB (0.855 against 0.868). The per-trace values at 7 GiB are
  long-0 0.708 against 0.702 and long-1 0.725 against 0.705.
- `seed` decodes like `on` (0.703 and 0.703 at 7 GiB; 0.868 against 0.868 at 14 GiB):
  the prompt's counts are what matter, not its swaps, and carrying them costs nothing
  measurable here. At 4 GiB it is 0.552 against 0.550.
- First-touch's decode share (0.358 at 7 GiB) is far below its decode-only rate on this
  set (0.575, the mean of the quarters above): the prompt fills the cache and it never
  evicts.

**Recommendation.** Score decode and prefill separately; the aggregate over all rounds
is a prefill number and hides the decode result. The prefill policy is a different,
open question, and these arms do not decide it: swapping gains 0.07 in prefill, and
freezing gains 0.01 to 0.04 in decode at small budgets, on one held-out set of two traces.
Under "placement decides the executor" the experts a prefill ubatch touches mostly will
not be resident, so prefill is a separate design question, not something this cache
answers.

## 5. Parameter sensitivity

Source: `summary-ablations.csv` (from `ablations/*.csv`). Adaptive, decode,
leave-one-set-out, token-weighted, one parameter changed at a time from the defaults
(every 4, swap-n 96, land 1, decay 0.7, min-count 2.0, margin 1.5). The cell is the
hit rate at IQ3_XXS 4 / 7 / 14 GiB and its change from the default row
(0.589 / 0.708 / 0.845).

| parameter | value | hit at 4 / 7 / 14 GiB | change |
|---|---|---|---|
| every | 1 | 0.594 / 0.701 / 0.833 | +0.005 / -0.006 / -0.012 |
| every | 2 | 0.595 / 0.702 / 0.833 | +0.005 / -0.006 / -0.012 |
| every | 8 | 0.558 / 0.674 / 0.836 | -0.031 / -0.033 / -0.009 |
| every | 16 | 0.495 / 0.609 / 0.794 | -0.094 / -0.099 / -0.051 |
| swap-n | 16 | 0.452 / 0.569 / 0.767 | -0.138 / -0.138 / -0.078 |
| swap-n | 32 | 0.523 / 0.637 / 0.811 | -0.066 / -0.071 / -0.034 |
| swap-n | 192 | 0.602 / 0.719 / 0.848 | +0.013 / +0.012 / +0.003 |
| swap-n | 384 | 0.604 / 0.720 / 0.848 | +0.014 / +0.012 / +0.003 |
| land-delay | 0 | 0.598 / 0.714 / 0.849 | +0.009 / +0.007 / +0.004 |
| land-delay | 2 | 0.583 / 0.703 / 0.842 | -0.007 / -0.005 / -0.003 |
| land-delay | 3 | 0.576 / 0.698 / 0.839 | -0.013 / -0.010 / -0.006 |
| land-delay | 4 | 0.546 / 0.663 / 0.827 | -0.043 / -0.045 / -0.018 |
| land-delay | 8 | 0.504 / 0.619 / 0.799 | -0.085 / -0.089 / -0.046 |
| land-delay 4, skip rule disabled | 4 | 0.571 / 0.694 / 0.836 | -0.018 / -0.014 / -0.009 |
| land-delay 8, skip rule disabled | 8 | 0.551 / 0.678 / 0.826 | -0.039 / -0.029 / -0.019 |
| decay | 0.0 | 0.469 / 0.594 / 0.785 | -0.120 / -0.114 / -0.060 |
| decay | 0.5 | 0.583 / 0.699 / 0.838 | -0.006 / -0.009 / -0.007 |
| decay | 0.85 | 0.597 / 0.718 / 0.855 | +0.007 / +0.011 / +0.010 |
| decay | 0.95 | 0.594 / 0.720 / 0.863 | +0.005 / +0.013 / +0.018 |
| margin | 1.0 / 1.2 / 2.0 | 0.589 / 0.708 / 0.845 | within 0.001 |
| margin | 3.0 | 0.565 / 0.661 / 0.805 | -0.025 / -0.046 / -0.040 |
| min-count | 1 | 0.592 / 0.716 / 0.858 | +0.003 / +0.009 / +0.013 |
| min-count | 4 | 0.517 / 0.613 / 0.774 | -0.072 / -0.095 / -0.071 |

Reading it:

- **The defaults are near the optimum.** Nothing changes the result by more than about
  +0.02 (decay 0.95 at 14 GiB); the large moves are all losses. The Q8_0 rows in the
  same file show the same losses (every >= 8, swap-n <= 32, land delay >= 4, `min_count`
  4, margin 3, decay 0) and the same small gains, except that decay 0.85 to 0.95 is
  mixed (-0.004 to +0.007).
- **Hurts:** adapting less often (every >= 8), a smaller swap cap (<= 32), a land delay
  of 4 or more, `min_count` 4, margin 3, and forgetting everything (decay 0). Margin
  between 1.0 and 2.0 does nothing.
- **Small gains:** decay 0.85 to 0.95, `min_count` 1, a cap of 192 or more, and a land
  delay of 0. Adapting every 1 or 2 rounds is slightly negative for IQ3_XXS at 7 and 14 GiB (-0.006,
  -0.012) and slightly positive at 4 GiB (+0.005); for Q8_0 it is slightly positive
  (+0.003 to +0.008).
- **The cap binds at small budgets.** The most the defaults can swap is 96 / 4 = 24 per
  token; the measured rate is 21.5 per token at 4 GiB and 10.4 at 14 GiB
  (`swap-traffic.csv`). That is why raising the cap helps most at 4 GiB, and why a cap of
  192 or 384 costs more traffic for little: at 4 GiB 30.0 swaps per token against 21.5
  (+40 %) for +0.014; at 7 GiB 20.6 against 17.6 (+17 %) for +0.012.
- **The land delay cliff in swap count is the in-flight skip rule (tested by disabling
  it); the hit-rate loss is partly the rule and partly late landing.** The rule skips an
  adaptation while swaps are pending. With `every` 4, a swap decided at round r lands at
  round r + 1 + land; for land 3 or less it has landed before the next window (r + 4),
  for land 4 it has not. Land 3 takes the same number of swaps as the default (21.5 per
  token at 4 GiB, 17.6 at 7, 10.4 at 14) and costs 0.013 at 4 GiB; the cliff is at land
  4, where swaps fall from 21.5 to 11.9 per token at 4 GiB (21.5 to 8.3 at land 8). With
  the rule disabled (`--no-skip-pending`; an expert already in flight is never swapped
  in twice) the count stays at 21.5 / 17.6 / 10.4 at land 4 and 8, and the hit rate
  falls less (rows above). So at land 4 the rule itself costs 0.025 / 0.031 / 0.009 at
  4 / 7 / 14 GiB, and the remainder of the loss against the default (0.018 / 0.014 /
  0.009) is placement that is stale by the landing time. A land delay of 2 costs only
  0.003 to 0.007. This is the key engineering number: a batch of swaps must land within
  `every - 1` = 3 token periods to keep the rule idle. A full batch of 96
  experts is 160 MiB (IQ3_XXS mean size) or 478 MiB (Q8_0); landing within 3 tokens at
  40 tokens/s (75 ms) needs about 2.2 GB/s (IQ3_XXS) or 6.7 GB/s (Q8_0) of copy rate from
  host memory to the device. The host-to-device rate of this machine's cards was not
  measured here. The skip rule is modelled as read from Strata's `adapt()`; whether the
  production rule has this exact form is not tested here.

## 6. Swap traffic

Source: `swap-traffic.csv`: adaptive, decode, leave-one-set-out, from the `swaps` column
of `curves-decode-by-set.csv` divided by tokens.

Expert size is **not** taken from the swap list (the CSV has only counts): MiB per token
is `swaps per token x mean expert size` (IQ3_XXS 1.665 MiB, Q8_0 4.980 MiB, from
`expert-sizes.csv`). For IQ3_XXS the layers differ (1.245 to 2.222 MiB), so the columns
`mib_per_token_if_min_layer` and `..._max_layer` bound it.

| format | budget | swaps/token | MiB/token moved | GB/s at 40 tokens/s | CPU MiB/token saved vs static | saved per MiB moved |
|---|---|---:|---:|---:|---:|---:|
| IQ3_XXS | 2 GiB | 22.4 | 37.2 | 1.56 | 255 | 6.8 |
| IQ3_XXS | 4 | 21.5 | 35.8 | 1.50 | 291 | 8.1 |
| IQ3_XXS | 7 | 17.6 | 29.2 | 1.23 | 284 | 9.7 |
| IQ3_XXS | 14 | 10.4 | 17.3 | 0.73 | 218 | 12.6 |
| IQ3_XXS | 24 | 4.3 | 7.2 | 0.30 | 123 | 17.1 |
| Q8_0 | 2 GiB | 13.7 | 68.4 | 2.87 | 487 | 7.1 |
| Q8_0 | 4 | 20.8 | 103.6 | 4.35 | 683 | 6.6 |
| Q8_0 | 7 | 22.6 | 112.4 | 4.72 | 806 | 7.2 |
| Q8_0 | 14 | 20.7 | 103.0 | 4.32 | 888 | 8.6 |
| Q8_0 | 24 | 16.2 | 80.8 | 3.39 | 849 | 10.5 |

IQ3_XXS at 40 tokens/s is 0.30 to 1.56 GB/s (1.17 to 2.08 at 2 GiB if every swap were
the smallest or the largest layer's expert); Q8_0 is 2.9 to 4.7 GB/s. The tokens/s is a
chosen operating point, not a prediction (the file also has 20 tokens/s).

Two cautions. A swap-in is a read of host memory by the copy, so it competes with the
CPU's expert reads for the same DRAM bandwidth: 1.23 GB/s is 5.9 % of 21 GB/s (section
7), and Q8_0's 4.72 GB/s is 16 % of 30 GB/s. And first-touch's fills are not in this
table: the model counts no cost for them, and they are bounded by the cache size and paid
once per cold cache.

A swap pays for itself when the expert is hit more than once before it is evicted. The
`swap-traffic.csv` column `extra_hits_per_swap_vs_static` is the extra routed hits adaptive
has over static per swap: 7.0 to 16.4 for IQ3_XXS, 6.6 to 10.5 for Q8_0.

## 7. What it could mean for decode speed (estimate)

Source: `cpu-bound-estimate.csv`. **This is the rate the CPU's expert reads alone would
allow at a given bandwidth, computed from the simulated bytes; it is not a decode rate
and not a bound.** It ignores attention, the dense layers, GPU time for the resident
experts, dispatch overhead and the swap copies' share of memory bandwidth, and a kernel
that beat the median bandwidth would exceed it.

CPU expert throughput comes from [`sycl-cpu-expert-bandwidth.md`](sycl-cpu-expert-bandwidth.md)
(medians of bursts, under the host's ambient load). The production kernel as built
(unpinned arena workers) reaches 16 to 43 GB/s at 22 threads over 13 type and shape
configs; the same code with pinned workers, `prod+pin`, is a variant of production and
reaches 14.4 to 46.7 GB/s. The columns below use the `prod+pin` figures: its 14.4 (taken
as 14) and 46.7 (47) bound the range, and the two configs nearest this model are Qwen
IQ3_XXS gate, 21 GB/s, and Qwen Q8_0 gate, 30 GB/s (Qwen Q8_0 down, 47 GB/s, is the
fastest of the 13). **The IQ3_XXS figure is a proxy:** the measured config is a uniform
IQ3_XXS gate matrix, but this model's IQ3_XXS file is a mixed allocation in which only 6
of 48 layers carry IQ3_XXS gate and up tensors (the rest use IQ2_XXS, IQ2_XS, IQ2_S or
IQ3_S, with Q2_0 down tensors in 30 layers and IQ4_NL in 18; `expert-sizes.csv`,
`tensor_types`), and Q2_0 measured 16 to 22 GB/s at 22 threads, so the real rate may be
lower. The rate is
`bandwidth / (cpu_gib_per_token x 1.0737)`.

| format, budget | policy | CPU GB/token | tokens/s at 14 GB/s | at the format's point (21 / 30) | at 47 GB/s |
|---|---|---:|---:|---:|---:|
| IQ3_XXS, no cache | none | 0.838 | 16.7 | 25.1 | 56.1 |
| IQ3_XXS 4 GiB | adaptive | 0.350 | 40.0 | 60.0 | 134.4 |
| IQ3_XXS 4 GiB | static | 0.655 | 21.4 | 32.1 | 71.8 |
| IQ3_XXS 7 GiB | adaptive | 0.249 | 56.2 | 84.3 | 188.6 |
| IQ3_XXS 7 GiB | first-touch | 0.346 | 40.5 | 60.7 | 135.8 |
| IQ3_XXS 7 GiB | static | 0.547 | 25.6 | 38.4 | 85.9 |
| IQ3_XXS 14 GiB | adaptive | 0.134 | 104.7 | 157.1 | 351.6 |
| Q8_0, no cache | none | 2.507 | 5.6 | 12.0 (at 30) | 18.7 |
| Q8_0 4 GiB | adaptive | 1.584 | 8.8 | 18.9 | 29.7 |
| Q8_0 7 GiB | adaptive | 1.327 | 10.5 | 22.6 | 35.4 |
| Q8_0 7 GiB | first-touch | 2.055 | 6.8 | 14.6 | 22.9 |
| Q8_0 7 GiB | static | 2.172 | 6.4 | 13.8 | 21.6 |
| Q8_0 14 GiB | adaptive | 0.955 | 14.7 | 31.4 | 49.2 |

(The IQ3_XXS 21 GB/s column is `tps_bound_at_21_gbps` in the CSV; the Q8_0 column is
`tps_bound_at_30_gbps`; the column names say `bound`, the text says rate.) Reading: at the median bandwidth the CPU kernel measures, an adaptive
7 GiB cache lifts the CPU-limited ceiling from 25 to 84 tokens/s for IQ3_XXS and from
12 to 23 for Q8_0; against a static profile of the same size the ceilings are 2.2x and
1.6x higher. The CPU's share of a token overlaps the GPU's, so the real gain is smaller
than these ratios wherever the GPU part is the longer path.

## 8. Rules context and what to build

**What the simulator models.** A cache that copies host-resident experts into VRAM over
time, driven by recent routing, with the executor following the current placement: the
CPU runs an expert until its copy has landed, and the evicted slot's victim is gone at
once. That is what `land-delay` and the in-flight skip rule do, and it is why the land
delay (section 5) decides whether the cache works. It is not a GPU zero-copy read of host
memory, and it does not copy a weight per dispatch.

**Whether the placement ruling allows it is an open question for the owner, and this
document does not answer it.** The memory design's "Placement decides the executor"
(`docs/backend/sycl-memory-design.md`, line 52, and CLAUDE.md, Architecture) says the
planning pass decides where data lives and inference executes each op where its data
already is. It says placement "is made once, in the planning pass" and that "the
dispatcher never re-litigates it at op time" (`sycl-memory-design.md:62-65`). It forbids
GPU zero-copy reads of host memory and "weight streaming" (copying host-resident weights
into device scratch per dispatch). Two readings are open:

- *Allowed.* A swap is a placement change made between dispatches by the planner; every
  dispatch still runs where its data is, and no weight is copied per dispatch or into
  scratch. The "streaming" clause is about per-dispatch copies.
- *Not allowed.* Placement is decided once, and a cache that re-places experts while the
  model runs is outside the ruling, however it is dispatched. The clause that says a
  VRAM-starved configuration that is too slow on the CPU is fixed by "placement (budget,
  eviction priority)" would then mean changes to the plan, not a run-time policy.

The data in this document are what the owner would weigh against either reading.

If run-time re-placement is allowed, what the data support:

1. **The gain is largest when the cache is small.** Against first-touch, CPU bytes per
   token fall 20 to 34 % for IQ3_XXS at 2 to 10 GiB (5 to 26 % of the experts) and 22 to
   35 % for Q8_0 at every budget measured (2 to 20 % of the experts). For IQ3_XXS the gain
   dips to 7 to 9 % at 14 to 18 GiB and returns to 33 % at 24 GiB (first-touch plateaus
   there; the cause is not tested).
2. **Where the cache is large (IQ3_XXS 14 to 18 GiB, 36 to 47 % of the experts), first-touch
   is the cheaper choice.** It is within 1.1 and 0.9 points of adaptive there and needs no
   swap traffic, no counters and no copy engine. At 24 GiB adaptive is 3.6 points ahead
   (33 % fewer CPU bytes); a hypothesis is that first-touch never evicts, which was not
   tested. It also does not follow a change of topic; this simulation runs each trace alone and does not test a
   mid-session shift.
3. **Do not ship a static profile.** It is below first-touch at every IQ3_XXS budget and
   from 7 GiB up for Q8_0; at Q8_0's 2 and 4 GiB both are near zero.
4. **Use Strata's parameters; do not tune them.** The one thing to engineer to is the
   land delay of at most 3 token periods, which sets a minimum copy rate (section 5).
5. **Score prefill apart from decode; its policy is open** (section 4). Freezing swaps,
   seeding counters and swapping all decode within 0.04 of each other for IQ3_XXS at 4 to 14 GiB
   (0.05 for Q8_0 at 7 GiB).

What would change the answer: a Q8_0 capture (the Q8_0 curves are a proxy), a capture on
the SYCL backend, longer generations, a trace with a topic change inside one session,
the measured host-to-device copy rate and landing latency of the B70, and timing. None
of these was done here.

## 9. Caveats

- **Ten traces from one model**, three kinds of text, one CPU run each. Chat and code
  prompts are 39 to 104 tokens; the long prompts are documentation and source text.
  There is no confidence interval: the spread over traces in section 3 is the only
  uncertainty estimate here, and the held-out groups have 4, 4 and 2 traces.
- **IQ3_XXS routing used for Q8_0.** Quantisation changes the hidden states and so the
  routing a little. Nothing here measures how much. Treat the Q8_0 curves as the same
  routing stream cut into larger experts.
- **The simulator models hit rate, not time.** There is no PCIe, no copy latency beyond
  an integer land delay in tokens, no overlap of CPU and GPU, no contention between swap
  copies and the CPU's expert reads. Section 7 is a rate at a median bandwidth, labelled
  as an estimate.
- **A round is one decoded token, not a speculative-decoding window.** Strata's windows
  union several tokens' experts; the cadence "every 4 rounds" is in tokens here. A
  single sequence is traced; concurrent slots are not.
- **Traces are 256 decode tokens**, each started from a fresh cache. On chat and code
  adaptive is still improving at the end of the trace (section 4, quarters), so its
  numbers are conservative for such sessions by an amount not measured; on the long
  set it is flat after the first quarter. First-touch needs no warm-up beyond the fill.
- **Token-weighted means mix sets of different sizes** (decode: 1024, 1024, 512 test
  tokens for chat, code, long; `phase=all`: 1199, 1336, 18948).
  The per-set table in section 3 shows the spread; no set-balanced mean is used.
- **The two oracles differ.** "Oracle" in leave-one-set-out is the held-out set pooled;
  in leave-one-trace-out it is the single trace. Statements about adaptive beating the
  oracle hold for the first only.
- **The tracer's effect on inference is expected, not measured to be nil**: with an eval
  callback the scheduler splits the graph after each top-k node. Logits against an
  untraced run were not compared.
- **First-touch admits instantly** in the model; this matters only in prefill (section 4).
- **Swap bytes use the mean expert size**, not the experts actually swapped.
- **The budgets are nominal totals** (budget per layer x 48); real resident bytes are a
  little lower because slots are floored.
- **The trace files are not committed** (about 22 MB). `traces-manifest.csv` has their
  sha256; they were in the session scratchpad. The curves and tables here can be
  regenerated from the committed CSVs with `analyze.py`; regenerating the curves
  themselves needs the traces.

## Data

All in [`moe-expert-cache-data/`](moe-expert-cache-data/).

| file | contents |
|---|---|
| `curves-{decode,all}-by-{set,id}.csv` | raw simulator output: 7 budgets x 2 formats x 4 policies x each held-out group (`run-curves.sh`) |
| `curves-all-by-set-prefill-{off,seed}.csv` | the `phase=all` leave-one-set-out run with `--prefill-swaps off` and `seed` (`run-curves.sh`) |
| `ablations/*.csv` | the same for one parameter varied, decode, leave-one-set-out, budgets 4, 7, 14 GiB, including land 3 and the skip rule disabled (`landnoskip*`) (`run-ablations.sh`) |
| `summary-curves.csv` | token-weighted hit rate, CPU GiB per token, swaps per token, resident fraction, min and max over groups |
| `summary-ablations.csv` | section 5 |
| `swap-traffic.csv` | section 6 |
| `cpu-bound-estimate.csv` | section 7 |
| `long-0-rounds.csv`, `long-0-windows.csv`, `long-0-blocks.csv` | section 4 (`prefill-rounds.py`; the decode-only control row comes from `decode-quarters.csv`) |
| `decode-quarters.csv` | section 4: decode hit rate per 64-round quarter at 7 GiB, per set and per long trace (`decode-quarters.py`) |
| `prefill-arms.csv`, `summary-prefill-arms.csv` | section 4: prefill and decode hit rate per prefill arm (`prefill-arms.py`; summary from `analyze.py`) |
| `expert-sizes.csv`, `traces-manifest.csv` | section 2 (`gen-sizes-and-manifest.py`) |
| `analyze.py` | derives every `summary-*`, `swap-traffic`, `cpu-bound-estimate` and `long-0-blocks` file from the raw CSVs |

`run-curves.sh OUT_DIR TRACE...` and `run-ablations.sh OUT_DIR TRACE...` take the output
directory and the traces as arguments (`SIM`, `PYTHON`, `GGUF_IQ3` and `GGUF_Q8` override
the simulator, interpreter and GGUF paths) and write the file names above. The
simulator needs `gguf-py` and numpy, so run it with `PYTHONPATH=gguf-py` and an
interpreter that has numpy.
