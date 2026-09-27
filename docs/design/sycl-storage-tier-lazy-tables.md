# SYCL STORAGE tier for lazy row tables (PLE / n-gram / engram embeddings)

*Status: DESIGN rev 3h, not implemented. Tracker: `llama.cpp-th32`.
Implementation tickets are filed after review (proposed split in §11). Written
2026-09-26 against fork `76c7f6548` and upstream `ggml-org/master` `694ec2354`;
the host-fit terms are re-read at fork `c69d5774d`, whose `unified-cache.cpp`
and `ggml-sycl.cpp` are byte-identical to `76c7f6548`. Rev 2 folded review
`design-th32-r1` under the lead's rulings §T1. Rev 2b folds review
`design-th32-r2` (1 Critical, 5 Important, 12 Minor) under the lead's rulings
§T2, §T3, §T4, §D5 and §D6 (§D6 amends §D5 after `9g22`'s r4 review), and the
owner's decision §D4 (`lead-rulings-2026-09-26.md`). It takes `9g22`'s figures
from that design's rev 10 (`054069548`), which moves one figure of rev 9's that
this document quotes (§4.5, `th32-p3`). Rev 2c folds the lead's §T5, which rules
on rev 2b's open items: DQ-C2, the row-cache size and the cold-expert order
source. Rev 2d folds review `design-th32-r3` (3 Important, 14 Minor) under the
lead's §T6, and §D7 I-B's consumer-copy criterion. Rev 2e folds review
`design-th32-r4` (7 Important, 13 Minor) under the lead's §T8, which makes
D6-M(b) the one refill mechanism for every model and depth, with the relays of
§T7, §D9 and §D10. Rev 2f folds review `design-th32-r5` (7 Important, 15 Minor)
under the lead's §T10, with §T9 and §T9a, which settle the epoch ring's bound
(DQ-E): its depth derives from the context's in-flight ubatch depth, and
exhaustion is a plan bug. It re-pins `9g22` at rev 10 and takes that design's
device-written cancellation marker. Rev 2g folds review `design-th32-r6` (1
Critical, 4 Important, 9 Minor) under the lead's §T9b, which names the real
input guard (the SYCL host buffer's pre-write sync), and §T12, with §T11, §D12,
§D12a, §D12b, §D12c, §G1 and §G1a: the backend cancels before the catch-path
flush, and a failed-graph mark scopes the cancel at `synchronize()`. Rev 2h
folds review `design-th32-r7` (4 Important, 12 Minor) with *§T9c* and *§T9d*,
which make the guard a matter of queue identity (the host buffer's device's
cache queue alone) and refuse the pre-cache state by name, and *§G1b* and
*§G1c*, which give the strict accessor's export to `llama.cpp-uwlx` alone: the
guard's input fact now comes from a llama-side declaration that each ubatch's
sync counter verifies, and the input sync itself runs inside a watchdog scope.
Rev 2i folds *§D12d*, one shared rule for the failed-graph mark: it carries its
epoch scope, only a completed cancel step clears it, and `synchronize()` and
`graph_compute`'s entry each run the scoped step when they find it still set.
Rev 2j folds review `design-th32-r8` (2 Important, 11 Minor) under *§D12f*, with
*§D12e* and *§T9e*: the cancel step stores first and keys on what has been
enqueued, never on the stage signal, the mark arms hold their reads with the
backstop off and count their sites, the guard's witness runs once per ubatch,
the input sync's watchdog scope is engaged only for a context with a lazy plan,
and the queue refusal prints its cause as `tp`, `device`, `pre-cache` or
`other`. Rev 2k folds review `design-th32-r9` (3 Important, 6 Minor) under
*§D12h*, with *§D12g*: one llama-owned ubatch delimiter keys every witness, the
lazy plan flag is stamped onto every host compute buffer the context allocates,
the witness re-baselines on a new buffer, the counters mean one thing in both
designs, poisoning needs an unwritten enqueued marker, and device loss is
classified from the exception's text at the sync's catch; it also takes
*§D12j*'s counter prefixes and *§D12k*'s re-baseline and delimiter name. Rev 2l
folds review `design-th32-r10` (1 Important, 10 Minor) under *§D12m*, which
makes this document the one statement of the rules it shares with `9g22`: T6
refuses `llama_opt_init` on a lazy or cold plan, the delimiter is called on
every SYCL backend and fails the ubatch before `graph_compute`, per-backend
counters witness it, the allocation scope is `thread_local` and opened inside
`graph_reserve()`, the stamp is a pair of serials with the plan flag read from a
context-serial registry, the poison check covers drained epochs, and no fixture
forks after SYCL init. Rev 2m folds review `design-th32-r11` (2 Important, 11
Minor) under *§D12n*, which makes three shared mechanisms precise: an unstamped
guard buffer fails the witness closed, a consumed per-ubatch record on the
backend carries the delimiter to each gather, and the device-lost classifier is
one helper at four enumerated site classes, with device loss always recorded.
Rev 2n folds review `design-th32-r12` (2 Important, 7 Minor) under *§D12q*, with
*§D12o* and *§D12p*: the context at the host-buffer sync comes from the guard's
stamp, the carrier's post lives on the persistent `lazy_gather_ctx` with stated
`none` sentinels and is checked at every gather of every ubatch, and the
classifier prints a named line and counts every recorded device loss, so each
site class has an observable; T6 (e)'s training RED pins its parameter to
`output_norm.weight`, downstream of every `MUL_MAT_ID` and custom op, so the
backward build cannot abort first (*§D12r*). Rev 2o folds review
`design-th32-r13` (1 Important, 7 Minor) under *§D12s*: every delimited ubatch
ends at an end point on the delimited path alone, which sets each post back to
`none` on the success and the failure returns, so a post whose ubatch never
reached its gather cannot pass an undelimited graph, and T6 (e)'s training RED
fires on the k branch. Rev 2p folds review `design-th32-r14` (7 Minor): the end
point's contract (`noexcept`, resolved or refused by name, counted), cancel
scopes that do not read the posted `k`, a scorable failure-before-gather arm,
and, under *§T13* and *§T13a*, the SYCL device's `mmap_support` and the mapping
of lazy ranges left to `llama.cpp-5efe`, behind an order edge and an interface
requirement, with §4.5's host terms stated for today's anonymous-copy host tier
and one host budget, §4.2's. Rev 2q folds review `design-th32-r15` (5 Minor),
scored on the §T13a state (*§T13c*): T6b's skip, at the top of `load_all_data`'s
loop and counted into `size_done`, is the one source of "the table is not
loaded", the guard runs the backstop when a throw unwinds past it, and the T6
edge to 5efe is order-only. Rev 2r folds review `design-th32-r16` (3 Minor, 1
nit): T6 (i) rolls back the failed ubatch's cells and proves its guard was
built, T6b's abort RED is scored in its own process, and the citation note names
`fefb92980`. Rev 2s folds review `design-th32-r17` (PASS, 2 nits) with *§F2a*
I-9: the inventory reads each table's byte range from 5efe's per-file lazy-range
record, which `lazy_read`'s set and the mmap prefetch exclusion also consume,
and keeps no copy. Rev 2t folds review `design-th32-r18` (2 Minor, 4 nits) under
*§F2a* I-9 addendum 2: `lazy_table` takes its range by value from the record
through the inventory at plan publish, a new T6 arm (j) gates that the inventory
reads the record, and the record holds every `TENSOR_READ_LAZY` tensor whatever
`add()`'s gates decide. Rev 2u folds review `design-th32-r19` (2 Minor, 3 nits)
under *§F2a* I-9 addendum 3. It takes `llama.cpp-5efe`'s rev 3a pre-fill
(`3763bb164`), in which 5efe pre-fills the record before the early plan from a
per-arch declaration and `create_tensor` only checks it, and it states addendum
3's requirement on 5efe, which that rev 3a did not yet carry: each entry has one
`lazy_eligible` bit, computed by th32's DLM (a) predicate (a new T6a, which
lands no later than 5efe's C1), which the early plan and 5efe's exclusions read;
T6 refuses a missing range as `lazy-range-missing`, T6 (j) runs on a two-file
fixture, and every §8 gate set carries a default-context arm. Rev 2v folds
review `design-th32-r20` (2 Important, 2 Minor, 2 nits) under *§F2a* I-9
addendum 4 and the lead's default-context vehicle clarification: 5efe's C1
captures the pre-fallback lazy mode, the bit's mode input is that mode when the
SYCL branch is the taker and `lazy_read`'s resolved mode otherwise, `deferral`
carries no eligibility, T6 reads the bit, a new T6 arm (k) witnesses the input,
the drift check is mode-independent, `lazy-range-missing` is the not-found
branch of the SYCL branch's own range lookup, and the default-context arm runs
on a vehicle that reaches the training context and scores any refusal as FAIL.
Rev 2w folds review `design-th32-r21` (1 Important, 3 Minor, 2 nits) under
*§F2a* I-9 addendum 5: the taker is the per-model
`sycl_model_loading_guard.active`, captured at 5efe's C1 pre-fill, not the
process-level hooks check; `deferral` is `STORAGE` for an eligible table the
taker diverts and `LAZY_READ` for one it does not, and T6 takes a table exactly
when `deferral == STORAGE`; T6 (k) gains engagement checks and a `-ngl 0` run;
the declaration is keyed by arch and GGUF name with no hparams, its attributes
come from one per-arch function at the pre-fill, and `lazy_read::add()` reads
the bit on the non-taker path. Rev 2x folds review `design-th32-r22` (1
Important, 2 Minor, 3 nits) under *§F2a* I-9 addenda 6, 8 and 9, and re-pins
`llama.cpp-5efe` at rev 3c (`fef860772`): T6 serves a lazy table on the executor
the plan assigns to its consumer layer, a CPU-executed consumer getting a
host-tier descriptor and the CPU gather, with no device descriptor; a source
clause pins the pre-fill's mmap-support input; T6 (k) runs on the gemma4 shape
with named observables, stub arrangement and stop points; 5efe's rev 3c names
are adopted; and 5efe's C1 and T6 reach master in one merge. Rev 2y folds review
`design-th32-r23` (2 Important, 4 Minor, 4 nits) under *lead ruling §M58*: the
consumer's executor is one exported query over the plan's residency, which
`llama.cpp-30h4` and T6 both call and which `-ngl` never feeds; a CPU-executed
consumer's table has its own lifecycle in §5.3, with no ring entry, output slot,
marker or event, a carrier check in the CPU op, a post-build scan that counts
that op, and a CPU column in §3.3 and §10 that charges nothing device-side; T6
(l) is re-derived on a plan-tiered configuration with a supports_op-decline
engagement check, two REDs and a `D + 2` ubatch sub-arm; the descriptor is bound
in a T6-owned context after the late replan; and C1 lands the attribute values
that T6 reads. Rev 2z folds review `design-th32-r24` (2 Important, 8 Minor, 5
nits) and the lead's `-ngl` probes 3-5 (`llama.cpp-wj27`): the `no_alloc` pass
has no plan, so T6 binds the host form there and refuses `NO_PLAN` on the real
load only (T6 (o)); a lazy failure on the CPU form returns -3, not the abort
status 2 that the server path retries as a full KV cache (T6 (p), (q)), and the
op releases its landing slot on every return; `planned_target_device` becomes a
second entry point onto the query's body; `lazy_ctx` joins `ctx_buf_maps`; the
reduction's change to 30h4's `dev_layer` is named and gated (T6 (l-sync)), and
(l) and (l-sync) score correctness on the known-correct answer, recording token
identity with the pre-change build rather than gating on it (*lead ruling
§M65(a)*); (l) and (m) name their vehicle, the calibration of `<p>`, HOST and
STORAGE sub-arms and the default-context VOID guard; (k) cites 5efe's landed
per-model line; and qwen4exp's `consumer_layer` is the one set bit of
`is_ple_impl` (5efe rev 3e). T5's read-fault variable is accepted on *lead
ruling §M65(b)*'s terms. Rev 3a folds review `design-th32-r25` (2 Important, 4
Minor, 5 nits) and audits the arms against *lead ruling §M66*: the name-keyed
entry points answer `NO_PLAN` for an `_exps` name, so ir18's expert answer is
unchanged (T6 (n); rev 3c replaces that substring with the plan's predicate);
the lazy tensor gets an identity-only registration and never enters the
host-weight registry that the preload and layer streaming read (T6 (r), and a
preload count in (l)); (q) scores its slot release by site; (p) pins its line
and rc; the `no_alloc` pass has no `lazy_table`; (l)'s same-build identity is
VOID, not FAIL, unless the `off` run's gather is on the CPU and the plans agree;
and every criterion that scores an env-gated line sets its variable and names
its trace-live check (§10). Rev 3b folds the two additions of that review's
final verdict (3 Important, 5 Minor): (l-sync) sets
`GGML_SYCL_SUPPORTS_OP_DEBUG=1` in both builds' commands and names its `blk.0.`
decline as each build's trace-live check (I-3), and (l)'s answer scorer cuts at
30 characters, after the comma that ends the tenth number, with a `…9, 100`
negative control (M-5). Rev 3c folds review `design-th32-r26` (1 Important, 4
Minor, 4 nits). The expert skip, in both name-keyed entry points and in the
reduction, is one predicate taken from the planner's own split gate,
`infer_tensor_usage(name) == MOE_EXPERT_WEIGHT` for a name with no dense entry,
and no name substring remains: grovemoe's `_chexps` experts answer `NO_PLAN`,
and GPT-OSS's `_exps.bias` and arctic's `ffn_norm_exps` keep their dense answer
(I-1, T6 (n)). (l)'s plan agreement compares the per-layer `[PLACEMENT]` lines
exactly, and the priority and total lines by exactly the table's one host entry,
and both runs must print them (M-1). The descriptor's `init_tensor` is null,
every route into the host-weight registry and layer streaming is named with what
closes it, and a SYCL-form arm, T7 (8), counts the preload (M-2). (r)'s stub
exports `try_register_usage` answering true (M-3). The merge unit is 5efe C1-C4
plus T6 and T7 in one merge, and (m) and T9q carry *lead ruling §M56(b)*'s
precondition and watchdog (M-4). *Lead rulings §M71* and *§M72* touch no th32
arm: no th32 criterion scores an SDPA decline, a `[CONTEXT-PLAN] graph scratch`
line, the commit line's `admitted=` suffix or `[ZONE-PLAN-BUG]`, and a TERMINAL
abort in a th32 GPU arm fails that arm on its rc. Rev 3d folds review
`design-th32-r27` (3 Minor, 4 nits): T6 (n) gains a host-only half that calls
the real predicate on a fixture plan, with a plan-only and a classifier-only
mutant as REDs, and the r24 M-1 case's value is pinned (m-1); both planner split
gates, `compute_placement_plan`'s and `compute_multi_device_plan`'s, call the
one classifier, under a source pin (m-2); the predicate reads
`infer_tensor_usage` only, never the loader's usage registry, whose disagreement
on the same names is `llama.cpp-elot`, and (n)'s fixture registers usages as the
loader does (m-3); two staggered-landing residues go, and the §4.5 re-pin gates
T9q (n-1); (n)'s names are the real ones (n-2); T7 (8)'s RED binds through a
real SYCL buffer (n-3); and four more SYCL consumers of a `WEIGHTS` tensor are
named with what closes them (n-4). Rev 3e folds review `design-th32-r28` (3
Minor, 3 nits): T6 (n)'s real half computes its plan with the real
`compute_placement_plan`, opens a load transaction before it registers usages,
reads the registry back as a control, and adds a no-dense-entry case and a
no-plan case, so the registry-reading RED can fail (m-1); it links
`ggml-sycl-private-fixtures` and its plan installer lives in the
`GGML_SYCL_PRIVATE_TESTING` block (m-2); block-exec declines a lazy leaf through
`trace_prepare_fail("lazy-table-leaf")` with a once-per-process WARN and falls
back to the ordinary path, with a RED in T7 (9) (m-3); the source pin names the
two `placement_tensor_info` bodies, strips comments and forbids every
`MOE_EXPERT_WEIGHT` token (n-1); the `ggml_sycl_is_device_vram_buffer` guard is
the descriptor's alone, and its effect on every buffer type is stated (n-2); and
the `n_experts == 0` case pins layer 0's `layer_device` (n-3). Rev 3f folds
review `design-th32-r29` (3 Minor, 2 nits) and three lead relays: T6 (n) pins
the registry-reading mutant to the bound load effect's owner and gives its
read-back tensors the load's `model_id`, and states that under the
identity-snapshot route no registry RED fires (m-1); T7 (9) is a private-testing
fixture that calls `prepare_context` with two synthetic blocks, since no
default-config vehicle reaches block-exec (m-2); (n) and (9) are lead-run and
GPU-serial, not host-only (m-3); the reason not to reuse
`test_set_kv_placement_plan` is corrected (n-1); the output-allocation probe is
cited at its call (n-2); the §10 counters' line is its own liveness check; the
undefined read is cited as `llama.cpp-nb87`; 5efe's refusal reaches the cancel
step by a throw (*lead ruling §M69(b) third amendment*); and 5efe is re-cited at
rev 3l (`a68508606`). Rev 3g folds review `design-th32-r30` (2 Minor, 2 nits):
T7 (9)'s fixture gives each of its two blocks its own layer-indexed node with
one forward edge, so `prepare_context` reaches the missing-leaf loop, derives
the blocks from the node names, scores the existing trace lines, and counts a
layout rejection as VOID (m-1); every T2 and T3 fixture that runs the planner on
a shaped inventory or builds a cache is lead-run and GPU-serial, and 7v5c's
dry-run is said to initialise SYCL (m-2); §11 scopes its load claim to th32's
own `gpu-serial` tests (n-1); and the class-C rows follow *lead ruling §M75*
across the auto-ubatch ladder (n-2). Rev 3h folds review `design-th32-r31` (2
Minor, 2 nits): T7 (9) keeps the leaf cap in force below the fixture table's
bytes, so its GREEN reason alone places the decline before the cap check and the
probe, and its RED returns `leaf-materialize-too-large` before any allocation
(m-1); §11 classifies lead-run tests by rule and lists every member (m-2); the
fd-2 capture's mechanics are stated (n-1); and the class-C return names the
losing returns after the candidate's publish (n-2).*

Citations: fork code as `76c7f6548 <file>:<line>`, upstream code as `694ec2354
<file>:<line>` (read with `git show ggml-org/master:<file>`; the fork does not
have `qwen4exp` or `TENSOR_READ_LAZY` at `76c7f6548` or `c69d5774d`;
`llama.cpp-n77l` merged them at `fefb92980`, the merge of upstream `694ec2354`
that is now on master, which is cited as `fefb92980 <file>:<line>`; r16 m-2).
`deepseek41` has no runtime anywhere yet: its conversion is upstream PR #28696,
and its runtime design is `llama.cpp-9g22` (`docs/design/deepseek41-runtime.md`,
branch `task/ds41`), which consumes the interface in §9.1. This document owns
the storage-tier contract and the numbers it rests on; `llama.cpp-9g22` links to
them by the anchors below (`th32-contract`, `th32-host-fit`, `th32-windows`,
`th32-ds-targets`, `th32-cold-expert-facts`, `th32-dspark`, and from rev 2b
`th32-p3` and `th32-d4-fit`). Every number in this document was measured on this
host on 2026-09-26 under its permanent ambient load (load average 31-57, ~90 for
the first DeepSeek probes); the raw logs are in Appendix A and the probe sources
are described in Appendix B.

**Attribution.** A decision is called the owner's only where the owner made it
(`lead-rulings-2026-09-26.md`, or relayed by the lead as an owner answer), and
the owner's decisions are cited as **owner §D1**, **owner §D2** and **owner
§D4** after their sections there; one more, the default-context gate arm of
§8.1, was relayed by the lead on 2026-09-27 and is cited as **owner, relayed
2026-09-27**. A decision the lead made is marked *lead ruling* with its section
(§T1-§T13, §T9c-§T9e, §T13a, §T13c, §D5-§D12s, §F2a, §G1-§G1d). Everything else
in this document is this design's **proposal**. This document's own decisions
are numbered D1-D9 in §13; the one that was also called D4 (`--lazy-mode`) is
**DLM** from rev 2b on, so that "§D4" always means the owner's decision, and
"§D6" is always the lead's ruling, never this document's D6. A bare D-number is
this document's; the rulings file's sections always carry "§" and, where the
context does not already say, their author ("owner §D2", "*lead ruling §D5*").

---

## 0. Summary

**The problem.** Two models carry hashed n-gram embedding tables that are read a
few rows per token and are far too large for their value on a GPU:

| | Qwen3.8-Flash-Next (`qwen4exp`) PLE | DeepSeek-V4.1-Flash (`deepseek41`) engram |
|---|---|---|
| tables | 1: `per_layer_token_embd.weight` | 2: `blk.1.engram_embd.weight`, `blk.14.engram_embd.weight` |
| shape, type | `[160, 320001536]` Q8_0 | `[256, 384006168]` and `[256, 384016682]`; Q2_K in the Q2_K rung (Q3_K in Q3_K_M) |
| size | **50.66 GiB** | **2 x 30.04 GiB** Q2_K (2 x 39.34 GiB Q3_K) |
| row | 170 B | 84 B Q2_K (110 B Q3_K) |
| rows per token | 16 (orders 2-3 x 8 heads) | **48**: 24 per table (orders 2-4 x 8 heads) |
| raw bytes per token | 2720 B | 4032 B Q2_K (5280 B Q3_K) |
| row indices known host-side before compute | yes: hash in `set_input` | yes: `token_map` + hash in `set_input` |
| consumed at | block 1 | block 1 (table 1) and block 14 (table 14) |
| residency the planner picks on this host | P3 binds (§4.3): after lever (s), **STORAGE** at the lowest `MemAvailable` reading, 171.4 GiB, HOST at 190.4, and HOST only at the low corner at 181.8 and 183.7; on today's code **STORAGE** at every reading (§4.5; a fact, not a gap to close, *lead ruling §T7* 2) | **STORAGE**, both tables; phase 1 also puts a cold routed-expert set on STORAGE (owner §D4, §4.5) |
| who executes the gather | the CPU, on the SYCL backend's IO-engine worker; an H2D copy ordered on a host-signalled event takes the output to the device | the same, after the rows' IO ticket completes |

Each row read from storage is one random 4 KiB read. DeepSeek needs **3x** the
rows per token and **3x** the unique rows per PP512 ubatch, and on this host it
cannot go to HOST at all. So DeepSeek is the harder case, and it sizes the
staging, the cache floor and the IO targets (§3.3, §6.4, §8).

Upstream marks such a table `TENSOR_READ_LAZY` and leaves it in an mmap of the
GGUF: `ggml-cpu` `get_rows` page-faults the rows in on demand. That already
keeps the table off the GPU, which is what placement-decides-the-executor asks
for. It breaks the other three principles, though: the kernel page cache is the
allocator (P1), the kernel's page reclaim is the lifetime (P2), and the
resident bytes are invisible to the planner (P4).

**The design.** Each lazy table becomes a **unified-cache-owned object**
(`lazy_table`). The planner gives it one of two **residencies**, with no
environment variable:

- **HOST:** the whole table is resident in cache-owned, **locked**, CPU-only
  host memory (a new `alloc_tier::HOST_CPU`, §3.1). The planner chooses this
  when the table fits the host budget that remains after VRAM placement, host
  experts, host zones and the context envelope, and P3 (a predicted post-load
  `MemAvailable` of at least 24 GiB) still holds with the table resident
  (§4.3).
- **STORAGE:** the table stays in its file. What is resident is a row-granular
  **hot-row cache** (CLOCK), a **yieldable** tenant sized from the host budget
  left above the P3 floor (fixed at 2 GiB per table in a cold-set plan), plus a small per-context **staging** area. Missing rows are read with
  `O_DIRECT` preads that are issued as soon as the row indices exist (after
  `set_input`), in priority order: prefetched ubatches before lookahead, the
  older epoch first and within it the earlier consumer layer, contexts served
  fairly. STORAGE is a residency
  state, not an allocation tier: no byte is ever allocated "in" it (§3.1).

In both residencies **the CPU executes the gather**, and the table never
reaches the GPU: there is no zero-copy and no weight streaming. Where the
gather sits in the graph is the **provider's** decision, not the model's
(*lead ruling §T1*, C2 and D6(c)). Both residencies take **one path**
(*lead ruling §T2*, DQ, and §T4 I-3):

- The **SYCL backend** owns the gather node. At prefetch the submitting thread
  only copies the ubatch's row ids into a planned ring and posts a request;
  it never waits (§5.3). The IO engine's worker then runs the gather on the
  CPU: a **HOST** table's rows come straight from the resident body in
  microseconds, and a **STORAGE** table's after its IO ticket. Either way it
  dequantizes into a planned pinned output slot and signals a host-signalled
  Level Zero event. This is D6-M(b), the one mechanism for every model and
  every depth (*lead ruling §T8*), shared with `9g22`'s cold experts.
- An **explicit, event-ordered H2D copy** on the main in-order queue moves that
  F32 activation into the device compute buffer just before the consumer, with
  an event edge on that event. **No SYCL kernel reads `SYCL_Host` memory**
  on this path, and a RED fails if one does (§7.3).
- There is no scheduler split, and this design adds no host wait: it removes
  `llama.cpp-x1xd` and `llama.cpp-jhxq`'s split-input copy from this path
  (§5.3). The ring's depth derives from the context's in-flight ubatch depth,
  which the existing input guard sets (the SYCL host buffer's sync before
  each input write), and exhausting it is a plan bug (*lead rulings §T9a,
  §T9b, §T10*). Under command-graph replay the gather is a
  **stated segment boundary**, dense graphs included (§7.4).

Rev 2a put the HOST gather in the leading CPU split, as a CPU op next to
`inp_embd` (*lead ruling §T1*, C2), and let the first SYCL op read its output
in place. DQ forbids that read. The copy that replaces it cannot be made
wait-free from the leading split, because the next ubatch's CPU op would
overwrite the copy's source on the submitting thread (§7.1). So HOST takes
the STORAGE path: **one path, no host wait**. The lead ruled this
(*lead ruling §T5*, DQ-C2 option (a)), and it supersedes *§T1* C2's placement
of HOST in the leading CPU split.

The consumer contract that a model implements is §9.1.

**The measured facts that shape it** (details in §2 and Appendix A):

| fact | value |
|---|---|
| where the tables live | bcachefs `/models`: user data on two **USB** SSDs, NVMe holding promoted cached copies only |
| 4 KiB `O_DIRECT` random reads, Qwen table | **1389 / 7815 / 9124 IOPS** at QD 1 / 16 / 64 |
| 4 KiB `O_DIRECT` random reads, DeepSeek Q2_K `blk.1` table | **4249 / 7070-7242 IOPS** at QD 16 / 64 (the same file read **4657** at QD64 minutes earlier, while bcachefs wrote 7.5 GB of promotions to the NVMe) |
| bcachefs read amplification | **~56-62x** on every file: each 4 KiB read costs ~256 KiB of device reads, because `data_checksum=crc32c` is over whole 256 KiB extents |
| mmap page faults (the upstream path) | 1555 rows/s with 1 thread, 7866 with 16 threads: the same device ceiling |
| reference: ext4 on the root NVMe, same probe | 6019 / 82798 / 110110 IOPS (**~12x** bcachefs at QD64); the root disk has **33 GB free**, so moving the tables there is not an option (§13 D3) |
| the pread pool against a same-session fio QD64 control | **7051 rows/s against 7070 / 7242 IOPS** (ratio 0.99) |
| cold decode token, Qwen (16 rows) | 3.75 ms mean, 13.0 ms p99 |
| cold decode token, DeepSeek Q2_K (24 + 24 rows, two files, one pool) | `blk.1`'s 24 rows land at **7.13 ms mean, 14.6 ms p99**; all 48 at 11.2 / 21.0 ms |
| a decode token queued behind another context's 12000-row prefill | **1359 ms** FIFO, **21.7 ms** with priority (§5.2) |
| cold PP512 ubatch, Qwen (~7100 unique rows) | 0.85 s mean, 8.3 k rows/s |
| cold PP512 ubatch, DeepSeek (18-25 k unique rows) | **~2.5-3.5 s** at the measured 7.0-7.8 k rows/s |
| HOST gather and dequant, straight from the table | 4.2 us per 16-row token; ~2 ms per 8192-row ubatch |
| steady-state decode tokens with zero misses | Qwen, 512 MiB: prose 32%, mixed 63%. DeepSeek, 8 GiB: prose 20%, mixed 55% (48 rows must all hit) |
| `llama-bench` prompts | uniform random tokens, so **always cold**, at any cache size (§8.5) |

**What the planner will choose on this host** (§4.5). The host weights are
counted at their **exact** bytes (`llama.cpp-rnh3`), not at the 1.2x ratio the
fork charges today, and the host zones at the bytes the sizing code actually
commits (§2.3): `dma_staging_pool`, the RUNTIME pool and the **PP MoE oneDNN
ring** included. The ring has no type gate, so every MoE model carries it
today: 7.87 GiB on DeepSeek Q2_K and 4.17 GiB on Qwen, charged twice on the
host and once in VRAM. Every fit is therefore stated twice: on **today's
code**, and **after lever (s)** (`llama.cpp-sgc5`, `llama.cpp-s4ip` and the
ring type-gate, which the owner chose in §D4). The host budget follows
`MemAvailable`, which read 171.4-190.4 GiB today, so `host_budget` =
**163.4-182.4 GiB**. Under the default EXPERT mode the B50 holds no routed
expert, so these are B70-alone numbers (`llama.cpp-9g22` phase 1,
`level_zero:0`). The numbers are reproduced by
`docs/design/sycl_storage_tier_host_fit.py` (§4.5); `llama.cpp-7v5c`'s
dry-run replaces them.

- **Qwen:** one zone set's non-weight host terms are 15.4 GiB as committed on
  today's code and 5.5 GiB after (s). P3 binds every load (*lead ruling
  §T6*, m-14), so the 50.66 GiB table is HOST only if the predicted post-load
  `MemAvailable` stays at least 24 GiB with it resident. That needs a
  `MemAvailable_pre` of 194.4-197.8 GiB on today's code (STORAGE at every
  reading) and 180.9-184.3 GiB after (s): **STORAGE at 171.4**, HOST at
  190.4, and HOST only at the low corner at 181.8 and 183.7.
- **DeepSeek Q2_K (the owner's primary rung):** one zone set's non-weight
  terms are 31.0 GiB as committed today and 9.5 GiB after (s). On today's code
  phase 1 is refused at every reading, even with floor caches and no draft.
  After (s) the planner's fit (`MemAvailable - 8`) admits it at three
  readings, but 9g22's gate precondition P3 (predicted post-load
  `MemAvailable >= 24 GiB`) fails at all four. **The owner's §D4 closes that
  gap**: the coldest Q2_K experts go to STORAGE and the CPU executes them. The
  cold set `C` is found at load, from that load's reading, by a search down
  the profile order (*lead rulings §D5, §D6*); its size at each reading is
  `llama.cpp-9g22`'s
  ([its §4.5.2](deepseek41-runtime.md#452-the-host-fit-r2-c-a-r3-c-1-c-2-owner-d4)),
  not restated here. The same computation places the engram tables under one
  P3 budget, experts first: a table goes HOST only if P3 holds with it HOST and
  no cold set, which on this host needs 228-232 GiB, so both tables are
  STORAGE at every reading. If even every expert on STORAGE cannot clear P3,
  the load refuses by name (D8, §13).

STORAGE speed on this box is dominated by the filesystem's 60x amplification.
Plain NVMe is not an option here (D3), so for DeepSeek that filesystem is the
operating point, and §8 scores against it. On that filesystem DeepSeek's
"decent speed" is bounded by the cold experts' reads on the same devices, not
by the engram rows: a PP512 ubatch reads the whole cold set once (§8.4,
`9g22` §4.5.3).

---

## 1. Scope and principles

This design must satisfy, and every reviewer checks:

- **P1:** the unified cache is the only allocator
  (`docs/backend/sycl-memory-design.md`, "The one allocation entry point").
- **P2:** ownership is an owner-first `mem_handle`, and its destructor is the
  sole release point (`sycl-memory-design.md`, "Owner-first allocation").
- **P3:** placement decides the executor, and layout follows residency. There
  is no GPU zero-copy of host memory and no weight streaming
  (`sycl-memory-design.md:52-120`).
- **P4:** plan == reality. The planner's accounting equals what is resident.
- **No host waits on the GPU path.** Everything is event-chained, and any
  remaining wait is named. The lazy path adds no wait on the submitting
  thread: slot reuse is an event edge (§7.2, *lead ruling §T6*, I-2).

The owner's decisions on the ticket (2026-09-26):

- The planner decides placement, with no configuration.
- HOST if the table fits after VRAM and experts, otherwise STORAGE, with async
  prefetch and a hot-row cache.
- The CPU executes the gather, so only the gathered rows move.
- "Decent speed" in either tier.
- DeepSeek-V4.1-Flash runs the **Q2_K** rung (owner §D2).
- The Q2_K fit (owner §D4): fix the zone waste with lever (s) (`sgc5`, `s4ip`
  and the ring type-gate), and put the coldest Q2_K routed experts on STORAGE,
  executed by the CPU. That sanctions STORAGE→CPU for experts. The cold set
  is profiled, and the profile is not mandatory (*lead ruling §T5*): the
  source order is `llama.cpp-9g22`'s, stated once in
  [its §7 D1](deepseek41-runtime.md#7-owner-decisions). P3 holds with any
  source; only speed depends on it.

The lead's rulings on this design (`lead-rulings-2026-09-26.md` §D3/§T1-§T8,
§D5-§D10): host weights are charged at exact bytes (`llama.cpp-rnh3`, which T2
depends on); the provider decides where the gather sits; the gather is
SYCL-owned CPU work with event edges (D6(c)), by D6-M(b)'s host-signalled
event for every model and depth (§T8); one host budget function
that every consumer reads; STORAGE is a residency state; planned HOST
residency is locked or refused by name; the row cache is yieldable; plain NVMe
is not an option (D3); the gather's output reaches the device by an explicit
event-ordered H2D copy, never zero-copy (§T2 DQ); the cold set is sized at load
by one computation (§D5), a search down the profile order with a landing slab
of `2 x` the largest per-layer cold bytes (§D6 I-4); one P3 budget is placed
jointly, routed experts before engram tables (§D6 I-5); row caches are fixed
before the cold set is sized (§T4 I-4); lever (s) removes `dma_staging_pool`
entirely (§D6 I-1); T2 is split into a WARN-only phase (a) and an enforcing
phase (b) (§D6 I-3); HOST gathers take the STORAGE path, which
supersedes §T1 C2's HOST placement (§T5 DQ-C2); the row caches are fixed at
2 GiB per table, index included, in the P3 budget before the cold-set search
(§T5); the cold-expert order is `9g22`'s §7 D1, not restated here (§T5).

Out of scope: the `qwen4exp` port itself (`llama.cpp-n77l`) and the `deepseek41`
runtime (`llama.cpp-9g22`); the GPU kernels that consume the gathered rows
(PLE's `ple_key`/`ple_value`/norms/conv, engram's `engram_q`/`engram_k`/
`engram_wkv`, all ordinary dense weights); and the MTP head. **The cold-expert
path itself** is in scope only at its interface. Owner §D4 brings
`llama.cpp-yl3v` forward to Q2_K, and yl3v owns the selection, the
`CpuExpertPool` execution, its landing slab and the prefetch from routing.
This design provides the residency state, the shared IO engine and the host
ledger that yl3v uses, and the fit that sizes the cold set (§4.3, §9.2).

---

## 2. Facts

### 2.1 The consumers and their tables

#### 2.1.1 Qwen3.8-Flash-Next PLE

GGUF metadata, read with `gguf-py` (Appendix A.1):

- `qwen4exp.ple.layers = [1]`: one PLE module, at layer index 1.
- `ple.ngram_size = 3`, `ple.heads_per_ngram = 8`, so there are `ple_n_heads = (3-1)*8 = 16`
  heads (`694ec2354 src/models/qwen4exp.cpp:95`).
- `embedding_length_per_layer_input = 160` = `ple_head_dim`.
- 16 per-head vocabularies of about 20 M rows each. They are primes from
  20000003 to 20000171, concatenated at `head_offsets` 0 .. 300001275.
- `layer_multipliers = [23703573157769, 20109073645365, 8052911324071]` (u64).
- `ple.eos_token_id = 248044` (`<|endoftext|>`), which differs from
  `tokenizer.ggml.eos_token_id = 248046`; the hash uses the former.
  `ple.image_token_id = 248056`.
- `per_layer_token_embd.weight` is Q8_0 `[160, 320001536]`, **54400261120 B
  (50.66 GiB)**. It sits in split `00003-of-00006` at absolute file offset
  **192**, and that file is 54400261312 B, so the table *is* the file. A
  160-element Q8_0 row is 5 blocks x 34 B = **170 B**. Row `r` starts at file
  offset `192 + 170 r`, so 4.1% of rows straddle a 4 KiB boundary (measured
  4.0-4.2% in the batch probe).

#### 2.1.2 DeepSeek-V4.1-Flash engram

GGUF metadata of the Q3_K_M rung, read with `gguf-py` from shard 1 (keys) and
shard 2 (tensors); Appendix A.1:

- `deepseek41.block_count = 40`, `embedding_length = 5120`, 384 routed experts
  plus 1 shared, 6 used.
- `engram.layer_ids = [1, 14]`, `engram.head_count = 8`,
  `engram.key_length = 256`, `engram.max_ngram_size = 4`. That makes n-gram
  orders 2, 3 and 4, each with 8 heads: **24 rows per table per token**, and
  **48 per token** across the two tables.
- `engram.multipliers` (8 = 2 tables x 4 positions), `engram.primes` and
  `engram.offsets` (48 each = 2 tables x 3 orders x 8 heads; primes 16000057 ..
  16000889), `engram.token_map` (129280 tokenizer ids onto a compressed vocab
  of 99092), `engram.pad_id = 2`.
- **The hash** (reference `NgramHashState.forward`, `inference/engram.py` of
  `deepseek-ai/DeepSeek-V4.1-Flash`): for position `p`, `c[s] =
  token_map[tok[p - s]]` for `s = 0..3`, with `pad_id` where `p < s` (and after
  a blocking token such as an image placeholder). Per table `L`: `rolling =
  c[0] * M[L][0]`; then for `i = 1..3`: `rolling ^= c[i] * M[L][i]`, and head
  `h` of order `i + 1` reads row `rolling mod prime[L][i][h] +
  offset[L][i][h]`. Every product fits in int64 (99091 x 8.26e13 < 2^63). The
  indices depend only on token ids and the previous three tokens of the same
  sequence, so they are **computable host-side before the graph runs**, in
  `set_input`, exactly like PLE's. The sequence history is the runtime's state
  (`llama.cpp-9g22`).
- `blk.1.engram_embd.weight` is Q3_K `[256, 384006168]`, **42240678480 B
  (39.34 GiB)**, in `DeepSeek-V4.1-Flash-Q3_K_M-00002-of-00009.gguf` at data
  offset **480**; the same shard holds the block's small `engram_q`/`engram_k`
  (`[5120, 4]`) and `engram_wkv` (`[6144, 25600]`). `blk.14.engram_embd.weight`
  is `[256, 384016682]` (the last offset plus the last prime), 42241835020 B.
  In the Q2_K rung, the owner's primary, both extents are read from the
  headers (A.1): `blk.1` in `Q2_K-00001-of-00007` at offset 5785327904
  (32256518112 B) and `blk.14` in `Q2_K-00004-of-00007` at offset 4032
  (32257401288 B), **60.08 GiB** together.
- A 256-element Q3_K row is one 110 B block; Q2_K is one 84 B block. Row `r`
  starts at `480 + 110 r`, so 2.7% of rows straddle a 4 KiB boundary (measured
  2.5-2.7%).
- The consumer (from `llama.cpp-9g22`): at the start of block 1 and block 14,
  the block's 24 rows per token are concatenated to `[6144, n_tokens]` and fed
  to `engram_wkv` on the device.
- Rung sizes (the GGUF README): Q2_K 246.3 GiB, Q3_K_M 323.4 GiB. The engram
  tables "follow the rung": 40284 MiB each at q3_K (which matches 384006168 x
  110 B), so 30762 MiB each at q2_K (x 84 B).

### 2.2 The upstream mechanism

- **Marking.** `create_tensor(..., TENSOR_READ_LAZY)` for the table
  (`694ec2354 src/models/qwen4exp.cpp:188-189`). The other in-tree user is
  gemma4's `per_layer_tok_embd` (`694ec2354 src/models/gemma4.cpp:58`). The
  `deepseek41` runtime marks both `engram_embd` tables the same way (§9.1).
- **Decision.** `llama_model_loader::lazy_read::add()`
  (`694ec2354 src/llama-model-loader.cpp:1088-1117`). `LLAMA_LAZY_MODE_OFF`
  never lazies. `AUTO` (the default, `694ec2354 src/llama-model.cpp:2861`)
  lazies marked tensors larger than 4 GiB. `ON` lazies every marked tensor.
  No mmap means no lazy. The mode enum is at `694ec2354 include/llama.h:218-222`.
  The CLI flag is `-lzm, --lazy-mode` (`694ec2354 common/arg.cpp:2706-2717`) and
  the llama-bench flag the same (`694ec2354 tools/llama-bench/llama-bench.cpp:810-835`);
  `--tensor-read-lazy` was renamed by upstream `bebc9350e`. `AUTO` falls back
  to `OFF` on devices without mmap support
  (`694ec2354 src/llama-model.cpp:1520-1528`); the fork's SYCL device reports
  none today (§2.3).
- **Buffer.** A lazy tensor goes into its own context (`ctx_key.lazy`,
  `694ec2354 src/llama-model-loader.h:150-163`) with the **CPU** buffer type
  (`lazy_read::buft()`, `694ec2354 src/llama-model-loader.cpp:1080-1086`, forced
  at `:1211-1213`). Its data is the mmap (`694ec2354
  src/llama-model.cpp:1840-1843`, `694ec2354
  src/llama-model-loader.cpp:1640-1661`). The tensor itself is never mlocked
  (`:1661`), but a mapping-backed tensor at a higher offset in the same shard
  locks the mapping's whole prefix through `llama_mlock::grow_to` (`694ec2354
  src/llama-mmap.cpp:790-803`), the table's range included: an upstream defect
  that §4.1's requirement on `llama.cpp-5efe` covers (r15 m-3).
- **mmap.** The lazy ranges are excluded from `MAP_POPULATE`/`WILLNEED` and get
  `POSIX_MADV_RANDOM` (`694ec2354 src/llama-mmap.cpp:469-510`). The SYCL branch
  keeps the table out of that flow entirely (§4.1).
- **Row indices.** `llm_graph_input_ple::set_input` computes them **host-side**
  (`694ec2354 src/models/qwen4exp.cpp:1066-1127`), because ggml has no int64
  and no xor. The 16 indices of every token are therefore known on the host
  before the graph is computed.
- **Consumer.** `ggml_get_rows(per_layer_tok_embd, rows)` followed by a reshape
  to `[2560, n_tokens]` (`694ec2354 src/models/qwen4exp.cpp:1185-1204`). It is
  expanded into the graph **before** layer 0 ("same graph split as
  `build_inp_embd`", `694ec2354 src/models/qwen4exp.cpp:381-386`) and consumed
  by `build_ple` at layer 1 (`:397-398`).
- **The other consumer shape.** gemma4 also takes `ggml_view_1d` of row 0 of
  its lazy table for multimodal embeddings
  (`694ec2354 src/models/gemma4.cpp:462`). A generic design must cover a
  constant-row view as well as an indexed `get_rows` (§9).

### 2.3 What the fork has today, and what it lacks

- **Tiers.** `alloc_tier` is `{DEVICE_VRAM, HOST_PINNED, MMAP_TRACKED}`
  (`76c7f6548 ggml/src/ggml-sycl/unified-cache.hpp:5590-5594`), and
  `unified_alloc` refuses `MMAP_TRACKED` runtime allocations
  (`76c7f6548 ggml/src/ggml-sycl/unified-cache.cpp:15125-15126`). **There is no
  CPU-only host tier.** Every cache-owned host byte is pinned USM host memory,
  which the driver maps into each device's VM. Pinning 50 GiB that no GPU
  touches spends locked, device-mapped memory for nothing. Whether xe also
  counts that memory as TTM `Shmem`, the resource behind this host's OOM
  history, is unverified, and the design does not depend on it.
- **The tier is derived, not requested.** `alloc_intent` is `{role, category,
  cohort_id, constraints}` (`unified-cache.hpp:5691-5696`): it has no tier
  field. `unified_select_tier()` derives one from the constraints
  (`must_device`, `must_host_pinned`, the cohort) and falls back to
  `HOST_PINNED` when the request exceeds the device's available bytes
  (`unified-cache.cpp:14780-14804`). `memory_location::host_accessible()` is
  simply `tier != DEVICE_VRAM` (`unified-cache.hpp:5625`).
- **Host budget: one fact, several sources.**
  - Each unified cache builds its own pinned pool, capped at construction by
    `min(total * pct, MemAvailable - max(8 GiB, total/32))`, pct 90 by default
    (`unified-cache.cpp:700`, `:4147-4259`).
  - With more than one GPU, `AUTO` mode is `PER_DEVICE`
    (`unified-cache.cpp:14292-14316`), so the dual-card configuration has
    **two** pools, each capped at the whole budget, each sampled at its own
    construction time.
  - `GGML_SYCL_HOST_RESERVE_MB` replaces the computed budget (strict parser,
    `host-reserve-env.cpp`, `llama.cpp-16el`), and `GGML_SYCL_NO_PINNED` sets
    it to zero (`:4248-4251`).
  - **The planner takes no host budget:** `compute_placement_plan` receives a
    VRAM budget only (`unified-cache.hpp:1591-1603`), and
    `plan.weight_host_bytes` (`:758`) is summed but never compared with
    anything. Host overcommit is found at allocation time, not at planning
    time. That is a P4 gap that already exists for host experts, and this
    design must close it (§4).
- **The host WEIGHT zone is 1.2x the host weights.**
  `populate_host_zone_sizing()` sizes it as `weight_host_bytes x 1.2 + 64 MiB`
  (`unified-cache.cpp:27089`, `:27670-27672`), and
  `ggml_sycl_configure_host_zones_for_plan()` eagerly commits WEIGHT + KV +
  STAGING + SCRATCH through `pre_allocate_host_pool()`
  (`76c7f6548 ggml/src/ggml-sycl/ggml-sycl.cpp:5615-5645`). On DeepSeek Q2_K
  that ratio alone adds ~30 GiB of slack nobody can name. `llama.cpp-rnh3`
  replaces it with exact bytes plus named terms; T2 depends on it (§4.2).
- **The STAGING and SCRATCH zones, and a separate RUNTIME pool, scale with the
  largest tensor.** All three are host-pinned, from the same pool budget.
  - STAGING is `s1_in_flight x (max(2 x max_tensor, max_staging_pair_bytes)
    + 2 MiB)`, with `s1_in_flight = 1` (`unified-cache.hpp:5539`), plus
    16 MiB, the expert bias, `moe_routing_ids` (`n_expert x 4 x n_ctx`),
    `tp_staging_buffer_bytes` and the MMID host pool
    (`unified-cache.cpp:27685-27699`). For DeepSeek's and Qwen's expert types
    the SOA staging pair equals `2 x max_tensor`, and the TP staging buffer is
    0 on one card.
  - SCRATCH is the sum of `max_tensor`, 32 MiB, `onednn_reorder`,
    `moe_q8_workspace`, `moe_cpu_expert_staging` (the largest MoE tensor) plus
    `moe_host_recipe_workspace`, `moe_vram_runtime`, `tp_vram_runtime`,
    **`dma_staging_pool`**, `pp_pipeline_scratch` and
    **`pp_moe_onednn_scratch`** (`:27716-27742`, the ring at `:27730`).
    `ggml_sycl_configure_host_zones_for_plan()` raises it to at least the
    cache's live oneDNN scratch sizes plus `max_tensor + 32 MiB`
    (`ggml-sycl.cpp:5634-5636`).
  - `pp_moe_onednn_scratch` is the **PP MoE oneDNN ring**. The loader sizes it
    for every MoE expert weight **with no type gate**
    (`c69d5774d src/llama-model.cpp:430-507`): `n_expert` MXFP4 WOQ weight
    slots, plus activation and output slots of `512 x n_expert x
    max(n_embd, n_ff)` rows at 2 and 4 B. `ggml-sycl.cpp:15962-15970`
    publishes it unconditionally. It serves only the MXFP4 WOQ executor, which
    Q2_K, Q3_K and Q8_0 experts never reach. It is **7.87 GiB** on DeepSeek
    Q2_K (2.24 + 1.88 + 3.75) and **4.17 GiB** on Qwen (0.42 + 1.25 + 2.50).
    It is charged in host SCRATCH, again in the RUNTIME pool below, and it sets
    the floor of each card's VRAM RUNTIME zone (`unified-cache.cpp:1567-1574`,
    `:4512-4530`), so on today's code it also takes that many bytes of expert
    room from the card.
  - `dma_staging_pool` is `2 x` the largest DMA-streamed tensor. It is sized
    **unconditionally**, whether or not weight streaming can engage; the code's
    own note says so and points at `llama.cpp-iaqm` (`:27522-27541`).
  - The **RUNTIME pool** is a separate pinned pool:
    `pre_allocate_runtime_chunks(onednn_scratchpad + dma_staging_pool +
    pp_pipeline_scratch + pp_moe_onednn_scratch)` on the cache's host arena
    (`ggml-sycl.cpp:5649-5653`, `unified-cache.cpp:20479-20483`,
    `pinned-pool.cpp:272-305`: a separate `runtime_chunks_` list charged to the
    same budget). So the DMA staging pool and the ring are each charged twice
    on the host, once in SCRATCH and once here. **Lever (s)** type-gates the
    ring to MXFP4 experts and removes it where no host reader exists, and it
    removes `dma_staging_pool` entirely: no host SCRATCH charge, no `:5649`
    charge and no device charge, because weight streaming cannot engage
    under an arena and the pool has no legal consumer (*lead ruling §D6*,
    I-1; `llama.cpp-s4ip`, comments `c-rbwa` and `c-td91`, widened to the
    `:5649` site). It is part of the owner's §D4.
  - `cpu_quant_buffer_bytes` is **not** a host term. The code states it is
    diagnostic-only: no allocation is sized from it (`:27436-27446`).

  One zone set's non-weight host memory (§4.5, KV zone included, the RUNTIME
  pool as committed in 2 GiB chunks):

  | | today's code | after lever (s) |
  |---|---:|---:|
  | DeepSeek Q2_K (`max_tensor` 1.81 GiB) | STAGING 3.65 + SCRATCH 15.26 + RUNTIME 11.58 (commits 12) + KV 0.06 = **31.0 GiB** | 3.65 + 3.77 + 0.09 (commits 2) + 0.06 = **9.5 GiB** |
  | Qwen3.8 (`max_tensor` 0.83 GiB) | 1.69 + 7.63 + 5.92 (commits 6) + 0.06 = **15.4 GiB** | 1.69 + 1.80 + 0.09 (commits 2) + 0.06 = **5.5 GiB** |

  A 30 GiB engram table counted as `max_tensor` would make STAGING alone
  ~60 GiB and move SCRATCH and RUNTIME by as much again, so the lazy tables
  must stay out of that inventory (§4.1).
- **Every participating cache builds its own zone set, WEIGHT included**
  (`llama.cpp-sgc5`). A two-card run gives every cache the same plan snapshot
  (`ggml-sycl.cpp:15325-15352`), and the S1 preload calls
  `ggml_sycl_configure_host_zones_for_plan()` for each device (`:33709-33888`).
  For a host-expert plan the second cache would commit a second ~150 GiB WEIGHT
  zone. T2 depends on sgc5 (§4.2).
- **CPU-buffer weights.** A weight in a CPU-owned buffer leaves the SYCL
  inventory, and the post-`create_tensor` replan treats its layer as host.
  The canonical contract §8 items 2 and 4 place CPU-buffer tensors outside the
  contract. Upstream's lazy tensor is exactly such a tensor, so merged as-is it
  would sit outside P1/P2/P4 by construction.
- **`mmap_support`.** The fork's `ggml_backend_dev_caps` has the field
  (`76c7f6548 ggml/include/ggml-backend.h:174`), but the SYCL device's
  initializer lists only four fields (`ggml-sycl.cpp:107797-107802`, `.async` at
  `:107798` through `.events` at `:107801`), so it is value-initialized to
  **false**. Upstream SYCL sets it `true` (`694ec2354
  ggml/src/ggml-sycl/ggml-sycl.cpp:6380`), and upstream's loader turns `AUTO`
  into `OFF` when any device reports false (`694ec2354
  src/llama-model.cpp:1519-1528`). Merged as-is, the default `AUTO` would
  silently disable every lazy table on this host (§4.1). The field is
  `llama.cpp-5efe`'s to set; th32's branch does not depend on it.
- **Hooks.** The fork calls into the SYCL backend through
  `ggml_backend_reg_get_proc_address` hooks
  (`76c7f6548 src/llama-model.cpp:63-111`). This design adds its entry points
  the same way.
- **Custom ops.** SYCL declines `GGML_OP_CUSTOM`: `supports_op` falls to
  `default: return false`
  (`76c7f6548 ggml/src/ggml-sycl/ggml-sycl.cpp:108771-108772`), so a custom op
  is scheduled on the CPU backend.
- **Backend-owned CPU work.** Host experts are **not** a scheduler split in the
  fork: `MUL_MAT_ID` stays a SYCL node, and the backend hands the host experts
  to `CpuExpertPool` workers (`cpu-dispatch.hpp:115-128`,
  `ggml-sycl.cpp:76048-76067`). That path still has host waits of its own (it
  waits on the activation's D2H event and on each entry's ready event before
  submitting, `:76033-76043`, e.g. `act_deferred_evt.wait()` at `:76034`), so it is the precedent for *ownership*, not for
  waits. The event-edge precedent is the queue-submitted `host_task` with
  `depends_on` (`ggml-sycl.cpp:75262-75272`); this design's edge is a
  host-signalled Level Zero event instead (§7.2), which has no precedent in
  the fork and is why T7 probes it first.
- **Command-graph replay.** One exec-graph slot per context, keyed by
  `(graph_hash, n_nodes, device, phase)` (`ggml-sycl.cpp:99164-99173`). A
  different signature re-records; after 12 consecutive re-records with no
  replay, `exec_graph_replay_futile` turns record/replay off for the context
  (`:99226-99240`, threshold at `:99228`). MoE decode graphs can instead replay **segments** between
  direct-dispatched `MUL_MAT_ID` nodes (`:97186-97270`), and
  `GGML_SYCL_GRAPH_DIAG` counts both kinds (`:632-660`, printed at `:794-830`).
### 2.4 The storage under the table (measured)

`/models` is bcachefs with three members (sysfs
`/sys/fs/bcachefs/118129b2-.../dev-*`, and `bcachefs fs usage`):

| dev | label | block device | link | user data | cached data |
|---|---|---|---|---:|---:|
| 0 | `nvme.kingston` | nvme0n1 (Kingston OM3PDP3512B) | PCIe | **0** | 182 GiB |
| 1 | `bulk.ugreen` | sdb (500SSD1 in a USB enclosure) | USB 10 Gb/s, uas | 612 GiB | 0 |
| 2 | `bulk.t7` | sda (Samsung PSSD T7) | USB 5 Gb/s, uas | 171 GiB | 0 |

The filesystem options are `foreground_target=nvme`, `background_target=bulk`,
`promote_target=nvme`, `promote_whole_extents=1`, `data_checksum=crc32c`,
`encoded_extent_max=256k`, `block_size=4k`, no compression, 1 replica. So:

- **Every user extent of the table lives on a USB SSD.** The NVMe holds
  promoted *cached* copies (182 GiB across the filesystem). `filefrag` shows two
  pointers per 256 KiB logical extent across most of the file (463214 fiemap
  extents for 13.28 M blocks), which is consistent with a cached copy on NVMe
  plus the USB home. Per-device `io_done` counters during the probes put
  ~80% of 4 KiB-read bytes on NVMe at QD1 and ~60% at QD64, with the rest on the
  two USB devices.
- **A 4 KiB read costs ~256 KiB of device reads (60x).** This was measured, not
  inferred: the sum of the per-device `io_done` read deltas divided by the
  bytes fio returned is 60.2x / 59.4x / 59.2x at QD 1/16/64. A crc32c checksum
  covers the whole encoded extent (up to `encoded_extent_max` = 256 KiB), so a
  partial read must fetch and verify the whole extent. The row-granular gather
  this table needs is the worst case for that layout.
- **At QD64 the USB devices saturate**: ~420-445 MB/s each, which is the T7's
  5 Gb/s link. That caps random 4 KiB reads at **~9.1 k IOPS** whatever the
  engine: fio `io_uring`, a pread thread pool and mmap faults all land at the
  same ceiling (§5.1).
- Sequential 1 MiB `O_DIRECT` reads of the table reached **1.78 GB/s** (a
  whole-table read in ~31 s), but that read was served mostly from the NVMe's
  cached copies (A.2: 19.7 of 27.6 GB from dev0). Settled data reads at
  0.29-0.47 GB/s (§9.2.2), so a whole-table read is minutes (D1).

**The DeepSeek table, same filesystem.** The same probe over `blk.1`'s byte
range in shard 2 (`--offset=4096 --size=42240000000`) measured **1014 / 5153 /
8400 IOPS** at QD 1/16/64 (p99 7.7 / 18.7 / 28.2 ms), with amplification
**60.2x** at QD16 and **60.0x** at QD64. The QD1 amplification is not
attributable: the model download was still writing the next shard, and
bcachefs reconcile read 6.8 GB from the NVMe during that 15 s run to move data
to the USB targets. Load average was 90, against 39-57 for the Qwen runs, so
the lower QD1 and QD16 figures reflect contention, not a property of the file.
The ceiling at QD64 is the same ~8.4-9.1 k IOPS. The shard had been written
~15 minutes earlier, so part of it may still have been on its NVMe foreground
target; the DeepSeek numbers are therefore, if anything, optimistic for a
settled file.

**The DeepSeek Q2_K tables, rev 2 (the owner's rung, now fully on disk).**
`blk.1.engram_embd` is in `Q2_K-00001-of-00007` at file offset 5785327904
(32256518112 B) and `blk.14.engram_embd` in `Q2_K-00004-of-00007` at 4032
(32257401288 B). The shards were written between 17:23 and 18:27 and probed
at ~18:36 (load average 31-52, no download running). Three things came out of
it, and each changes how targets must be scored:

- **The ceiling moves with bcachefs's own background work.** fio QD64 over
  `blk.1` read **4657 IOPS, p99 146 ms** while bcachefs wrote **7.5 GB** to the
  NVMe in the same 15 s (promotion of the extents being read, plus reconcile of
  the fresh shard). Two minutes later the same probe read **7070** and **7242
  IOPS** with 0.4-0.5 GB written. A 1.55x swing on the same extent within
  minutes is why §8 scores the engine as a **ratio to a same-session fio
  control**, never against an absolute rate.
- **The pread pool is at the device ceiling.** Bracketed by those two fio runs,
  the 64-worker pool read a 20000-row batch at **7051 rows/s** (ratio 0.99).
- **Amplification is unchanged at 56.5-61.7x**, and 1.9-2.2% of 84 B rows
  straddle a 4 KiB block.

---

## 3. Object model: the lazy table as a unified-cache object (item 1)

### 3.1 One new allocation tier, and residency as a separate state

**Allocation tier.** `alloc_tier` gains one value, `HOST_CPU`: host memory that
only the CPU reads.

| tier | memory | who may read it | new? |
|---|---|---|---|
| `DEVICE_VRAM` | device USM | that device | no |
| `HOST_PINNED` | USM host, pinned pool, mapped into each device's VM | CPU, and the GPU for DMA staging only | no |
| **`HOST_CPU`** | anonymous `mmap`, `MADV_HUGEPAGE`, **`mlock`ed**, not device-mapped | **CPU only** | **yes** |
| `MMAP_TRACKED` | legacy: file mmap tracked for weights | CPU | no |

- **Requested only explicitly.** `alloc_constraints` gains `must_host_cpu`, and
  `unified_select_tier()` maps it to `HOST_CPU`
  (`76c7f6548 ggml/src/ggml-sycl/unified-cache.cpp:14780-14804`). Nothing falls
  back **into** `HOST_CPU`: the size-exceeds-available fallback stays
  `HOST_PINNED`, and a cohort cannot inherit `HOST_CPU`. A `HOST_CPU` request
  never falls back **out** either: if it cannot be admitted or locked, it is
  refused.
- **Locked, and refused by name if it cannot be** (*lead ruling §T1*, I4). The
  raw wrapper is `unified_cache_raw_alloc_host_cpu(size)`:
  `mmap(MAP_PRIVATE | MAP_ANONYMOUS)`, `madvise(MADV_HUGEPAGE)`, then `mlock()`
  over the whole range. `mlock` without `MLOCK_ONFAULT` faults every page in
  and locks it before the call returns, so the resident bytes equal the planned
  bytes from the first moment, and a failure surfaces at load, not mid-run. If
  `mlock` fails, the wrapper unmaps and the allocation is refused as
  `host-cpu-lock-failed` with `errno`, the requested bytes and `RLIMIT_MEMLOCK`
  in the message. This is not hypothetical on this host: `SwapFree` read
  **37 MB of 8 GiB** today, so unlocked pageable memory *would* be swapped, and
  a "resident" table would silently become disk reads. `ulimit -l` is
  unlimited here. The wrapper and its `unified_cache_raw_free_host_cpu`
  counterpart are added to the canonical contract's §3 allowlist exactly like
  `unified_cache_raw_malloc_host`, with a line saying the memory is locked, so
  no other file calls `mmap`/`mlock` for it (P1).
- **Why locked host memory and not the pinned path.** Pinned USM host memory
  would also be resident, but the driver maps it into both devices' VMs for a
  table no GPU may read (P3), and it rides the pinned pool's 2 GiB chunking.
  `mlock` gives the residency guarantee without the device mappings.
- **Not device-mappable.** `memory_location::host_accessible()` stays true for
  `HOST_CPU` (`unified-cache.hpp:5625`), because the CPU reads it. A new
  `memory_location::device_mappable()` is false for it, and every path that
  hands a host-accessible pointer to a device must check it instead.
- **The owner-first front door.** A `HOST_CPU` allocation is requested through
  `unified_allocate_owner(req)` with `constraints.must_host_cpu = true`. The
  owner class is **`EXTERNAL_EXACT`**: each allocation is exact and
  independently released, not a slice of a cache backing pool. This keeps the
  two-site `CACHE_BACKING` allowlist (canonical contract §3.1) untouched.
- **The tier-switch census T1 must audit.** `alloc_tier::HOST_PINNED` or
  `MMAP_TRACKED` is named 36 times in 7 files (`unified-cache.cpp` 25,
  `ggml-sycl.cpp` 5, `common.hpp` 2, and one each in `binbcast.cpp`,
  `dmmv.cpp`, `element_wise.cpp` and `unified-cache.hpp`). The sites that must
  learn `HOST_CPU` explicitly are the two switches `alloc_tier_name()`
  (`unified-cache.cpp:2403`) and `offload_stats_note_alloc()` (`:2606`),
  `unified_free_record()` (`:14806`), `query_registered_location()`
  (`:21595`), `query_location()` (`:21643`), the allocation registry, and the
  shutdown census (`unified_cache_shutdown_runtime_alloc_census_for_test()`,
  `:19473`, and its siblings).

**Residency.** STORAGE is **not** an allocation tier (*lead ruling §T1*, I3). No
byte is ever allocated "in" it, and an allocatable enum value would be a latent
path to request it. A `lazy_table` instead carries `lazy_residency { HOST,
STORAGE }`, the planner's decision. The reporting enums `cache_location`
(`unified-cache.hpp:2167`) and `expert_resolve_tier` (`:2313`) gain `STORAGE`
as a *reported* state only; nothing allocates against them. The cold
routed-expert set of owner §D4 (`llama.cpp-yl3v`, §9.2) reports the same
`STORAGE` value.

### 3.2 The `lazy_table` object

One `lazy_table` per lazy tensor per loaded model. It is **model-scoped**: the
rows are immutable weights, so every context of the model shares it.

```
lazy_table
  identity   : ggml_sycl_cache_id (owner = model token, name = tensor name)  // the usual weight key
  geometry   : type, ne[0] (row elements), n_rows, row_size (ggml_row_size)
  source     : file index, offset, size  // the lazy-range record's, by value
               file path                 // the loader's file table
  residency  : HOST | STORAGE                                // the planner's decision, immutable
  consumer   : consumer_layer; its executor: the plan's, through the query below
  HOST:    body   : mem_handle (HOST_CPU, WEIGHT)
  STORAGE: fd     : O_DIRECT file descriptor, owned (closed in the destructor)
           cache  : lazy_row_cache (§6.3)
  refs       : intrusive count: the model's descriptor buffer holds one, every outstanding ticket one
  stats      : §10
```

- **Ownership (P2).** Every memory the object uses is a `mem_handle` minted by
  `unified_allocate_owner()` and held by the object. The object is destroyed
  when its **last reference** drops: the model's, released at unload (the §12
  lifecycle of the canonical contract), or the last outstanding ticket's,
  released by the reaper after its final completion (§5.3). Its destructor drops
  the handles and closes the fd, once. That is the sole release point, and it
  can never run while a read into its row cache is in flight, so it needs no
  teardown wait.
- **The source is a hand-off, not a second source** (*lead ruling §F2a* I-9
  addendum 2, r18 M-1). The object receives its (file index, offset, size) **by
  value**, once, at plan publish, from the per-file lazy-range record through
  the inventory entry (§4.1). The record's value is immutable for the model's
  lifetime, so the object holds the record's value handed over, not an
  independent source; what is forbidden is deriving it independently, from
  `llama_tensor_weight`'s `offs` or from `ggml_nbytes`. As a value it needs no
  counted reference to the llama-side record, and it stays valid for a ticket
  that outlives the model's reference (above).
- **The consumer's executor is the plan's, through one query** (*lead rulings
  §F2a* I-9 addendum 9, *§M58a*; r22 I-1, r23 I-1). On this fork neither `-ngl`
  nor the buffer type decides which executor runs a layer; the plan's residency
  does. The table's `deferral` is unchanged: `STORAGE` means the SYCL branch
  takes the table, and the executor is chosen inside T6, never as a new
  `deferral` value.
  - **`-ngl` does not decide it.** The early plan never sees it:
    `llama_model_sycl_compute_early_plan` builds its inventory from
    `weights_map`, the hparams and an argument-less envelope (`fefb92980
    src/llama-model.cpp:622-658`), and `n_gpu_layers` is read only by
    `load_tensors`' upstream split (`:2215`, `:2239`, `:2322-2330`). A layer
    that `-ngl` puts below `i_gpu_start` still gets `SYCL_Host` weights:
    `make_cpu_buft_list` puts the first device's host buffer type ahead of
    `CPU_REPACK` and `CPU` (`:1771-1779`), `create_tensor` prefers host weights
    whenever the list's front device is SYCL (`fefb92980
    src/llama-model-loader.cpp:1396-1401`), and `select_weight_buft` returns
    that type (`:1128-1136`); `-ot <pat>=CPU` resolves the same way
    (`:1411-1433`). This is measured (the lead's probes 3-5, 2026-09-27,
    ticketed as `llama.cpp-wj27`): gemma4 E4B Q8_0 on the B70 under
    `llama-completion -no-cnv -v` has one model buffer, `SYCL_Host` 8478.32 MiB,
    at `-ngl 1` as at `-ngl 99`, and with `GGML_SCHED_DEBUG=2` every weighted op
    of layer 0 is assigned to SYCL0 at `-ngl 1`, the only CPU ops being
    weightless `RMS_NORM` nodes (an earlier probe without `-no-cnv` died in the
    chat template and is void). So `-ngl` reaches the weight executor only
    through the buffer type it chooses, which does not decide it either (next
    item). What it still moves is the KV residency and the placement of
    weightless norms: in those probes a CPU KV buffer is copied into SYCL0's
    attention splits, the mixed state `llama.cpp-wj27` tickets, which this
    design cites and does not design around.
  - **The buffer type does not decide it either.** The SYCL backend supports
    `SYCL_Host` (`fefb92980 ggml/src/ggml-sycl/ggml-sycl.cpp:108810-108812`) and
    refuses an op only when `ggml_sycl_op_is_planned_on_host` holds
    (`:108185-108192`), which reads the plan's per-weight and per-layer
    residency (`:109078-109107`, `:109159-109214`; the plan fields at
    `ggml/src/ggml-sycl/common.hpp:5514-5559`). A `SYCL_Host` weight that the
    plan tiers to host is therefore executed by the CPU whenever ir18 does not
    move it to a ggml-cpu type, because no such type can hold it
    (`src/llama-model-loader.cpp:1474-1494`, `cpu_buft == nullptr`) or because
    `GGML_SYCL_HOST_WEIGHT_LAYOUT=off`. There, `llama.cpp-30h4`'s reading,
    "`SYCL_Host` means a SYCL device executes"
    (`src/llama-model.cpp:2584-2593`), is false, and T6 does not inherit it.
  - **The query.** A layer is CPU-executed iff one of its non-expert weights (a
    `blk.<il>.` weight that the plan does not split into per-expert entries, by
    the one expert predicate below, which replaces 30h4's `_exps` substring
    filter at `:2615-2617`: host-resident experts are the CpuExpertPool's
    business, not the layer's executor) sits in a buffer type that is not
    SYCL-owned, which the plan's entries do not index (30h4's finding,
    `:2562-2575`), or is SYCL-owned and planned host. The planned-host test is
    one exported function, `ggml_backend_sycl_weight_executor(name, buft)`,
    which returns a device index, -1 for the CPU, or
    `GGML_SYCL_PLANNED_NO_PLAN`. It is keyed by the weight's name and its
    settled buffer type, not by a tensor, because until the buffers exist
    (`src/llama-model.cpp:2705`) a tensor has no buffer and the tensor-keyed
    predicate answers "unknown" (`ggml_sycl_tensor_is_weight`,
    `common.hpp:4817-4820`). Its plan read is the one body that
    `ggml_sycl_get_planned_weight_residency` runs after its tensor checks, so
    supports_op and the query cannot drift apart while a plan exists (with none,
    the query answers `NO_PLAN`, below). That body returns the target itself,
    the layer's `layer_device` value when the layer has one and otherwise the
    dense entry's `get_target_device`, or `NO_PLAN`; it never compares with a
    caller's device, as the residency function's multi-device branch does today
    (`common.hpp:5535-5538`, `:5552-5558`), and supports_op makes that
    comparison as the body's caller. The existing name-keyed reader,
    `ggml_backend_sycl_planned_target_device(name)` (`fefb92980
    ggml-sycl.cpp:19173-19183`), which ir18 calls at `create_tensor` for every
    tensor (`src/llama-model-loader.cpp:1473-1475`) and which reads only the
    dense entry, becomes a second entry point onto the same body, so a
    non-expert weight with a `layer_device` entry and no dense entry gets one
    answer through both (r24 M-1). **Both name-keyed entry points answer
    `NO_PLAN` for a name the plan splits into per-expert entries, before they
    read the body, and the reduction skips the same names** (r25 I-1, r26 I-1).
    Which names those are is one fact, the planner's: it splits a tensor into
    per-expert entries iff `infer_tensor_usage(name) ==
    tensor_usage::MOE_EXPERT_WEIGHT` (`fefb92980
    ggml/src/ggml-sycl/unified-cache.cpp:27969-27974`), and only those entries
    stay out of the dense name index (`unified-cache.hpp:1406-1426`). A name
    substring is not that fact, in either direction, as the classifier's own
    comment warns (`ggml/src/ggml-sycl/common.hpp:1854-1873`): grovemoe's
    chunked experts, `blk.N.ffn_{gate,up,down}_chexps.weight`
    (`src/models/grovemoe.cpp:57-59`), are split but contain no `_exps`, while
    GPT-OSS's `blk.N.ffn_{gate,up,down}_exps.bias`
    (`src/models/openai-moe.cpp:52-54`) and arctic's `blk.N.ffn_norm_exps`
    contain it but are dense entries (`MOE_INTERMEDIATE` and `NORM`). So the
    skip is one function (proposed): `ggml_sycl_weight_is_split_expert(name)`
    holds iff the settled plan has no dense entry for the name and
    `ggml_sycl_usage_is_moe_expert(name)` holds. That second function is
    `infer_tensor_usage(name) == tensor_usage::MOE_EXPERT_WEIGHT`, factored out
    of the planner's split gates. There are two, one in each planner, and both
    call it in place of their inline test: `compute_placement_plan`'s at
    `:27974` and `compute_multi_device_plan`'s at `:29436` (`:28032` and
    `:29494` at `d8a67422d`), the planner that §4.2 gives the ledger's budget.
    So the two planners and the skip read one classifier, and T6's source pin
    forbids any other `MOE_EXPERT_WEIGHT` test in either planner (r27 m-2, r28
    n-1); T6 also rewrites the planner's comment that says an expert tensor's
    "name contains \"_exps\"" (`fefb92980 :27963`, `d8a67422d :28021`), the
    substring rule r26 I-1 retired. The classifier reads
    `infer_tensor_usage(name)` **only**, never `ggml_sycl_get_tensor_usage` or
    the usage registry `g_sycl_weight_usages` behind it (r27 m-3). That registry
    is a second usage source, and it disagrees on exactly these names: the
    loader registers `MOE_EXPERT_WEIGHT` for every `*_EXPS` and `*_CHEXPS`
    tensor and for `FFN_NORM_EXPS`, whatever the suffix, so `.bias` included
    (`fefb92980 src/llama-model-loader.cpp:1622-1633`, through
    `try_register_usage` at `:1715`), and `ggml_sycl_get_tensor_usage` returns
    the registered value before it falls back to the classifier (`e2461d4fb
    ggml/src/ggml-sycl/ggml-sycl.cpp:14103-14121`). A predicate that read it
    would bring back the false hit on `_exps.bias` and `ffn_norm_exps`. The
    backend-wide disagreement is `llama.cpp-elot`, which th32 does not fix;
    th32's predicate follows the planner, the source of the split, and is
    revisited when elot picks the authority. It is exported once, as
    `ggml_backend_sycl_weight_is_split_expert(name)`, which the reduction
    reaches through the SYCL hook table, and both entry points call it inside
    their plan lock. It answers from the classifier when no plan exists, and for
    a tensor the late inventory never saw, a ggml-cpu-owned weight
    (`src/llama-model.cpp:2562-2575`), for which a plan lookup alone would have
    no entry to consult; a name with a dense entry keeps today's dense answer,
    which also covers a model with no experts, whose expert-usage names the
    planner makes dense (`:27974`'s `n_experts > 0`). No name match remains on
    the path. ir18 relies on the answer: `NO_PLAN` sends a split expert
    untouched to the CpuExpertPool path
    (`src/llama-model-loader.cpp:1466-1472`), and a dense `_exps.bias` or
    `ffn_norm_exps` weight gets `planned_target_device`'s dense answer, so ir18
    moves a host-planned one to a ggml-cpu type as it does today. The body keys
    on the layer id before any dense-entry test (`common.hpp:5528-5540`), while
    the planner places expert triplets first-fit on the VRAM the dense layers
    leave, independently of `layer_device` (`fefb92980
    ggml/src/ggml-sycl/unified-cache.cpp:28299-28338`); without the skip, ir18
    under the default `GGML_SYCL_HOST_WEIGHT_LAYOUT=cpu` would move every expert
    of a host-tiered layer into a ggml-cpu type, device-planned groups included,
    so the executor would contradict the placement on every MoE load with a
    host-tiered layer. The skip sits in the entry points, not in the
    tensor-keyed body, so supports_op's expert answers, and its multi-GPU MoE
    routing exceptions (`ggml-sycl.cpp:108182-108185`), are unchanged. That body
    already answers an expert by its layer's `layer_device`
    (`ggml-sycl.cpp:109094`) although the expert's triplet carries its own
    target: a pre-existing one-fact, two-sources seam on the executor side,
    outside th32, which the lead may ticket.
  - **One reduction, two callers.** The llama side reduces the query per layer
    in one function, `llama_model_sycl_layer_executor(ml, il)`. It walks
    `ml.ctx_map` as 30h4 does, skips every weight for which
    `ggml_backend_sycl_weight_is_split_expert` holds, and reaches that predicate
    and the query through the SYCL hook table as `planned_target_device` is
    reached (`src/llama-model-loader.cpp:1475`), and returns the CPU if any
    non-expert weight answers -1 or sits in a ggml-cpu type, and otherwise the
    device they answer. It reads `NO_PLAN` on a SYCL-owned weight as not host,
    as 30h4 does today, except on a lazy table's consumer layer on the real
    load, where T6 refuses it as `[LAZY-PLAN-BUG]`, naming the layer and the
    weight, since the late inventory covers every SYCL-owned tensor there; the
    `no_alloc` pass has no plan, and T6 binds the host form there without asking
    (below; r24 I-1). 30h4's `dev_layer` sync and T6 both call it.
    `GGML_SYCL_DEV_LAYER_SYNC` (`src/llama-model.cpp:134-137`) gates only 30h4's
    mutation of `dev_layer`, never the query, and T6's read runs with it set to
    `0`. Reading `i_gpu_start`, `dev_layer` before the sync, or any `-ngl` split
    for the executor is forbidden, and T6 (l)'s REDs prove it (§11).
  - **What the change does to 30h4** (r24 M-2). Today 30h4 skips every
    SYCL-owned buffer type, `SYCL_Host` included
    (`src/llama-model.cpp:2605-2606`), so a layer whose weights sit in
    `SYCL_Host` keeps its `-ngl` device in `dev_layer`. Through the reduction, a
    planned-host `SYCL_Host` layer becomes the CPU there: the case of
    `GGML_SYCL_HOST_WEIGHT_LAYOUT=off`, and of any weight for which ir18 finds
    no ggml-cpu type (`src/llama-model-loader.cpp:1489-1492`). That moves the
    two consumers 30h4 names (`src/llama-model.cpp:2553-2554`): the layer's KV
    buffer type, which for a CPU `dev_layer` takes no SYCL hook and becomes a
    ggml-cpu buffer (`src/llama-kv-cache.cpp:287-318`), and the fused-attention
    check, which warns and disables fused attention for every layer when a
    layer's attention lands on a device other than its `dev_layer` and not on
    the CPU (`src/llama-context.cpp:1236-1256`). This is `llama.cpp-wj27`'s
    mixed state reached through the plan instead of `-ngl`. The change also
    replaces 30h4's `_exps` substring filter (`src/llama-model.cpp:2615-2616`)
    with the expert predicate (r26 I-1). On the gate models that moves no layer:
    a GPT-OSS `_exps.bias` or an arctic `ffn_norm_exps` is a dense entry of its
    layer, at `NORM_EMBED` priority (`fefb92980
    ggml/src/ggml-sycl/unified-cache.cpp:24148-24150`, `:24178-24179`), which
    the query answers with its layer's `layer_device`, so it is host-planned, or
    moved by ir18 to a ggml-cpu type, only where its layer's other dense weights
    are too. The one layer it changes is grovemoe's: a `_chexps` expert in a
    ggml-cpu type, which 30h4 counts today as a layer weight, no longer marks
    its layer CPU. th32 does not design around it: it names the change, gates it
    with T6 (l-sync), and scores the correctness of (l) and (l-sync) on the
    known-correct answer, not on token identity with any build, so neither
    passes through a regression in that routing, and a numerics change that the
    move of layer 0's attention to the CPU legitimately makes (P3) fails neither
    (*lead ruling §M65(a)*; §11).
- **When the descriptor is bound, and through which context** (r23 M-1). The
  loader buckets a tensor into `ml.ctx_map` by the buffer type chosen at
  `create_tensor`, and the model's buffers are made from those buckets
  (`fefb92980 src/llama-model.cpp:2705-2807`), after the late replan
  (`:2543-2545`). So T6 does not let `create_tensor` choose one for a `STORAGE`
  lazy tensor: it creates the tensor in a T6-owned context of the model,
  `lazy_ctx`, which no buffer-type selection sees. After the late replan and
  before the buffers are made, T6 runs the query for each table's consumer
  layer, sets the `lazy_table`'s `consumer` from the answer, creates the
  descriptor buffer on that executor for `lazy_ctx`'s tensors, and adds it to
  the model's buffer list (`pimpl->ctxs_bufs`), so its lifetime is the model's.
  T6's branch substitutes `lazy_ctx` for `ctx_for_buft(buft)` in `create_tensor`
  (`src/llama-model-loader.cpp:1800`) and keeps the tail's `ggml_dup_tensor`,
  name and count, so the tensor increments `n_created` (`:1817-1819`) and
  `done_getting_tensors` finds the count it expects (`:1826-1832`). It replaces
  the tail's `register_sycl_tensor_metadata` (`:1822`) with an identity-only
  registration (r25 I-2): `register_file_identity` and `register_identity`
  (`:1547-1558`), which the `lazy_table`'s key, the weight's
  `ggml_sycl_cache_id`, needs (§9.1), and never `register_host_weight`
  (`:1560-1562`). It keeps `register_sycl_tensor_metadata`'s two gates, the
  hooks resolved through `selected_final_dev` (`:1541-1545`) and the
  `weights_map` hit (`:1547-1548`), and it drops the `try_register_usage` call
  (`:1715-1717`), which changes no answer: `infer_tensor_usage`'s `token_embd`
  match (`ggml/src/ggml-sycl/common.hpp:1924`) gives `EMBEDDING`, the value the
  loader registers for `LLM_TENSOR_PER_LAYER_TOKEN_EMBD` (`:1638`), and
  qwen4exp's table has the same name (`src/models/qwen4exp.cpp:188`) (r26 N-4).
  `register_host_weight` puts a tensor in `g_sycl_host_weight_extras` (`fefb92980
  ggml-sycl.cpp:13646-13734`), which two consumers read as loadable host
  weights: S1-PRELOAD, whose only skip is a null `data` (`:33611`), which the
  sentinel base does not trigger, stages or uploads `ggml_nbytes` from
  `tensor->data` (`:34679-34777`), and layer streaming registers `ggml_nbytes`
  from it (`:16663-16674`). For the table that would be weight streaming (P3)
  and a second, contradicting source for "the table is not loaded" beside T6b's
  skip (P4). `lazy_ctx` stays out of `ml.ctx_map`, so no buffer-type selection,
  the late inventory or the reduction sees it, but it joins `ctx_buf_maps`, the
  list `load_all_data` runs over (`src/llama-model.cpp:2706-2808`,
  `:2857-2861`), with a `buf_map` that sends every file index to the descriptor
  buffer, so T6b's skip is reached in `load_all_data`'s per-tensor loop and its
  `size_done` credit keeps the final progress callback (r24 M-3). Joining
  `ctx_buf_maps` is a second step after the bind, because that list is a local
  declared at `:2706`: T6 adds `lazy_ctx` after the bucket loop (`:2808`) and
  before the `no_alloc` return (`:2834`) and the no-mmap `stable_partition`
  (`:2846`), which sorts the descriptor, whose `is_host` is false, into its
  first group; that is harmless, since T6b's skip reads no byte for it (r25
  N-1). T6 binds at the same point on the real load and on the `no_alloc` pass,
  which differ in one respect (r24 I-1). **The `no_alloc` pass has no plan by
  construction:** the early plan and the late inventory both return first under
  `no_alloc` (`:631`, `:670`), so the query answers `NO_PLAN` for every weight.
  A taker reaches that pass, because the guard ignores `no_alloc` and 5efe's
  pre-fill still writes `STORAGE` there, whenever `common_fit_params` sets it
  (`fefb92980 common/fit.cpp:56-60`) for `-fit on` or `llama-fit-params`, and
  for any API caller that sets it. There T6 binds the host form,
  `ggml_backend_sycl_lazy_host_buffer_type()`, without asking the query, and
  never reads `NO_PLAN` as a plan bug: the `[LAZY-PLAN-BUG]` refusal is the real
  load's alone. The pass's buffers are zero-size dummies
  (`src/llama-model.cpp:2771-2775`), so the descriptor there is
  `ggml_backend_buft_alloc_buffer(buft, 0)`, a context-less `buffer_init(buft,
  {}, NULL, 0)` that never reaches the buffer type's `alloc_buffer`
  (`ggml/src/ggml-backend.cpp:278-280`), with a null base rather than the
  sentinel (`:618-621`), which `memory_breakdown`'s `no_alloc` branch asserts
  (`src/llama-model.cpp:2936`). **No `lazy_table` exists on that pass** (r25
  M-3): the dummy holds no table reference and none is created through it, and
  T6b reports 0 bytes for it where that branch would charge the table's nominal
  size. Neither form puts a table byte on a device on either pass, so the host
  form claims nothing device-side for a fit to count. `common_fit_params`'
  context only reads the memory breakdown and never decodes
  (`common/fit.cpp:65-100`); with no plan it carries no lazy plan flag and is
  created as today (T6 (g)'s no-plan case), with the table's `lazy_gather_ctx`
  in its CPU form, no landing slot and no table reference: it takes the gather's
  shape from the record entry (`rows_per_token`) and the tensor (its type and
  row size), which is all a graph reserve needs to size the CPU op's output, and
  it is never executed. The late replan runs only when `sycl_model_backend`
  holds (`:2543`); when it does not, the early plan is the settled one, and the
  query reads it. The descriptor holds no bytes, so binding it after the replan
  moves nothing.
- **The descriptor buffer** (r23 N-2). The lazy tensor gets a zero-byte
  descriptor buffer **on its consumer's executor**. It owns no bytes and no
  `mem_handle`, only a `lazy_table` reference, so "a device" or "the host tier"
  below names where the descriptor is bound, never an allocation: every lazy
  byte is one of §3.3's owner-first handles (the body, the row cache, the
  landing slab, the ids ring), and a CPU consumer's gathered rows are an
  activation in its context's host compute buffer (P1, P2). For a SYCL-executed
  consumer it is a new SYCL buffer type,
  `ggml_backend_sycl_lazy_buffer_type(device)`, where `device` is the device the
  query above returns for the table's consumer layer. For a CPU-executed
  consumer it is the host tier's, `ggml_backend_sycl_lazy_host_buffer_type()`,
  with the same sentinel contract, and the gather runs on the CPU (§7.1, §5.3's
  CPU form): there is no device descriptor, no device zone and no device copy
  for a CPU consumer, and no device ever reads the table's host bytes. A lead
  run observes which form was bound through the plan line's `device=` field
  (§10), which prints the query's answer, a SYCL device or `CPU`. The contract
  below holds for both forms, except the scheduler reason, which is the SYCL
  form's alone: no node of a CPU consumer's graph has an edge to the lazy tensor
  (§7.1). It maps nothing: on the real load its base is a non-dereferenceable
  sentinel (the `no_alloc` pass binds a null-based dummy, above), its nominal
  size is `ggml_nbytes`, `is_host` is false, usage is `WEIGHTS`, and
  `get_tensor`, `set_tensor`, `memset_tensor` and `cpy_tensor` abort naming the
  table. On the real load its context holds the `lazy_table` reference; the
  `no_alloc` dummy holds none (above). It exists for three reasons:
  - `ggml_gallocr` treats a leaf whose `data` is null as unallocated
    (`76c7f6548 ggml/src/ggml-alloc.c:594-595`), so a bufferless 30-50 GiB
    tensor that reached a graph would be *allocated in the compute buffer*.
    The sentinel makes it read as allocated.
  - The scheduler assigns an op whose source sits in a `WEIGHTS` buffer to the
    highest-priority backend that supports that buffer type and the op
    (`ggml-backend.cpp:1594-1612`, rule `1.wgt` at `:1684`). That is how the
    SYCL backend claims the gather of either residency deterministically (§7.2).
  - `llama` sees an ordinary buffer, so model size, memory breakdown, the
    loader's mlock loop and `load_all_data` need only a narrow change (T6b):
    skip the tensor, and report the buffer's *resident* bytes, not its
    nominal size.
- **Its `init_tensor`, and every route into the host-weight registry and layer
  streaming** (r26 M-2). The descriptor's buffer interface is its own, never the
  SYCL buffer's, and its `init_tensor` is null, which
  `ggml_backend_buffer_init_tensor` treats as success
  (`ggml/src/ggml-backend.cpp:646-650`), so binding the lazy tensor sets no
  `extra` and registers nothing. The loader's route is closed by the
  identity-only registration (above). The SYCL side has four more, cited at
  `e2461d4fb ggml/src/ggml-sycl/ggml-sycl.cpp`, and each is closed:
  - The SYCL buffer's `init_tensor` registers any weight when weights are
    evictable (`:26603-26606`). Closed by the descriptor's null `init_tensor`.
    Its RED is T7 (8): with the SYCL `init_tensor` reused, the table enters the
    registry, and S1-PRELOAD's host-planned branch copies `ggml_nbytes` from the
    sentinel (`:34704-34724`).
  - `ggml_sycl_get_data_ptr_slow` registers any tensor of a `WEIGHTS` buffer
    with non-null `data` into layer streaming, with `ggml_nbytes`
    (`:24201-24206`). Closed by construction: the gather node dispatches nothing
    (§7.2), and the post-build scan admits no other reader of the lazy tensor,
    so nothing asks for its pointer. T4 also puts a guard at the function's
    entry, before the registration and before any device call: a tensor whose
    buffer is a descriptor, of either form, aborts with `[LAZY-PLAN-BUG] data
    pointer requested for lazy table <name>`, so a reader nobody found fails
    closed instead of registering the sentinel. Its RED is T4's.
  - `ggml_backend_sycl_get_weight_cache_key` registers a tensor of a host buffer
    that has no `extra` (`:14398-14408`). Closed by the descriptor's `is_host`
    being false.
  - `set_tensor` registers the weight it writes (`:36118-36121`). Closed by the
    descriptor's `set_tensor` abort.

  Four more SYCL-side consumers of a `WEIGHTS` tensor read it without
  registering it; each is closed or fails closed (r27 n-4):

  - `graph_preload_weights` (`:60880-60941`) takes each weight's cache key and
    preloads from `src->data`. Closed by its planner-authoritative return at
    `:60881`, since T6 requires a plan on the real load, so it never reaches its
    later reads.
  - `ggml_sycl_is_device_vram_buffer` (`:25995-26001`) casts any
    `buffer->context` to `ggml_backend_sycl_buffer_context` with no type check.
    Six of its call sites have no check of their own (`:8071`, `:8101`,
    `:26065`, `:60930`, `:81307`, `:81495`; `:52039` sits behind
    `ggml_backend_buffer_has_sycl_context`). T4 closes it for the descriptor
    alone (r28 n-2): the function first returns false for a buffer whose type is
    a descriptor's, through T4's `ggml_backend_sycl_buffer_is_lazy`,
    forward-declared ahead of `:25995`. Its effect by buffer type: for the
    descriptor, an undefined read of a `lazy_table` reference as a SYCL buffer
    context becomes `false`; for every other buffer type it changes nothing,
    since buffers with a real `ggml_backend_sycl_buffer_context` (the SYCL
    buffer type, `SYCL_HostCompute` and the CPU-offload compute type, the three
    names `:26107-26115` accepts) read their tier as today, and the rest keep
    today's read. th32 does not guard with
    `ggml_backend_buffer_has_sycl_context` instead, although that would also
    close the descriptor: it would turn today's read of a foreign context, which
    is undefined behaviour (byte 92 of the object read as `managed_meta.tier`,
    layout `:25855-25874`, where a zero reads as `DEVICE_VRAM`,
    `alloc_tier::DEVICE_VRAM == 0` at `unified-cache.hpp:5590-5594`), into
    `false` for `SYCL_Host`, `SYCL_KV_Host`, the split buffer (`:40169`), TP
    (`:40914`), tiered KV (`:39377`, `:39711`) and 5efe's `SYCL_Mapped`, and at
    `:26065` and `:81495` `false` means a host-resident weight, so a split or TP
    weight reaching them would change route with no th32 arm to see it. That
    pre-existing undefined read is `llama.cpp-nb87`, which th32 does not fix.
    One consumer of it moves on the merge unit whichever guard is used:
    `moe_hybrid_init_once`, always on (`:105464-105477`), counts host experts
    through this function (`:8101`) and prints them with its `host_src_ptr`
    count (`[MOE-HYBRID]`, `:8158-8165`); under 5efe's `SYCL_Mapped`, which
    holds the late inventory, that count carries no residency information, and
    no th32 or 5efe arm scores it. Its buffer-based residency is a second source
    beside the plan (P4), ticketed as `llama.cpp-0qe1`, which th32 does not fix.
  - The default-on dense-split executor (`:89612-89614`) classifies the lazy
    tensor as a `WEIGHT` root whose `resident_mask` is 0, because
    `ggml_sycl_tensor_metadata_owner` finds no `extra` and no SYCL buffer
    context (`:52001-52045`, `:91093-91104`), and refuses the graph
    (`ggml/src/ggml-sycl/block-exec-dense.hpp:714-718`). That fails closed, and
    is reachable only on multi-device.
  - Block-exec's leaf materialize (`:88836-88847`, `:87953-87975`) would
    allocate the table's span on the device and copy it from `tensor->data`. It
    is opt-in, behind `GGML_SYCL_BLOCK_EXEC_EXECUTE` (`:88576-88582`), and
    running it on a lazy-table model is a legitimate configuration, not a plan
    bug, so T7 makes block-exec decline the graph rather than fail it (r28 m-3):
    the missing-leaf collection notes a source whose buffer is a descriptor, and
    `ggml_sycl_block_exec_prepare_context` returns
    `trace_prepare_fail("lazy-table-leaf")` (`:88760-88766`) before its
    leaf-bytes check (`:88902`) and its output-allocation probe (the call at
    `:88912-88913`, whose failure return is `:88924`), after printing, once per
    process, the WARN `[SYCL-BLOCK-EXEC] lazy table <name> is a graph leaf;
    block-exec declines and the ordinary path gathers it`. The caller turns that
    into its `PREPARE_CONTEXT` gate and returns false (`:89410-89416`), and
    `graph_compute` runs the graph on the ordinary path (`:93263-93273`), where
    T7's gather serves the table. Its RED is T7 (9), a fixture, since no
    default-config vehicle reaches block-exec (r29 m-2).

### 3.3 The resident bytes of a lazy table, all planned

| allocation | scope | tier / role | size | CPU-executed consumer (§5.3's CPU form) |
|---|---|---|---|---|
| HOST body | model | `HOST_CPU` / `WEIGHT` | `table_bytes` (Qwen 50.66 GiB) | the same |
| row cache slab | model | `HOST_CPU` / new `alloc_role::LAZY_ROW_CACHE` | `slots x row_size`, in chunks of 256 MiB so that it can yield (§6.4) | the same (STORAGE) |
| row cache index | model | `HOST_CPU` / `LAZY_ROW_CACHE` | `slots x 13.3 B` (§6.3): 7.8% of the slab on 170 B rows, 12.1% on 110 B, 15.9% on 84 B | the same (STORAGE) |
| landing slab | context (class C), per table | `HOST_CPU` / new `alloc_role::LAZY_STAGING` | `depth x n_ubatch x rows_per_token x row_size`: Qwen 2 x 512 x 16 x 170 B = 2.66 MiB; DeepSeek Q2_K 2 x 512 x 24 x 84 B = **1.97 MiB per table** | STORAGE: **one** slot, `n_ubatch x rows_per_token x row_size` (Qwen 1.33 MiB; DeepSeek 0.98 MiB per table); HOST: none |
| epoch ring: ids captures | context (class C), per table, both residencies | `HOST_CPU` / `LAZY_STAGING` | `D x n_ubatch x rows_per_token x 4 B`, with `D = in_flight_depth + 2` (§5.3; *lead rulings §T9a, §T9b*): today `in_flight_depth` is 1, so `D` = 3, and Qwen 3 x 32 KiB = **96 KiB**; DeepSeek 3 x 48 KiB = **144 KiB per table** | none: `prefetch` classifies the ids in place, from the rows input |
| block ring | the IO pool (process), shared by every context and table | `HOST_CPU` / `LAZY_STAGING` | `QD x span_max`, where `span_max` is the largest row read of any lazy table (§5.2): 64 x 8 KiB = **512 KiB** for rows of at most 4 KiB, 64 x 16 KiB = 1 MiB if gemma4 E4B's 11424 B PLE rows are STORAGE; once, 4 KiB-aligned for `O_DIRECT`; each in-flight read holds a reference on its slot (§5.3) | the same share |
| output slot | context (class C), per table, both residencies | `HOST_PINNED` / new `alloc_role::LAZY_OUTPUT` | `depth x n_ubatch x rows_per_token x ne[0] x 4 B`: Qwen 2 x 512 x 16 x 160 x 4 B = **10 MiB**; DeepSeek 2 x 512 x 24 x 256 x 4 B = **24 MiB per table** | none: the CPU op writes its rows into the context's host compute buffer, which the scheduler's compute-buffer reservation sizes like any activation |
| epoch ring: marker words | context (class C), per table, both residencies | `HOST_PINNED` / `LAZY_OUTPUT`, its own allocation | `D` words of 4 B, one per ring entry (§7.2): the only host bytes a device kernel writes on this path | none |
| epoch ring: interop events | context (class C), per table | driver objects, not cache bytes (P1) | `D` host-signalled Level Zero events and their host-visible pool, created at context init, owned by a refcounted `lazy_event_pool` that every enqueued copy's wait also holds, and destroyed at its last reference, which the marker watcher drops after that copy's marker (§5.3, *§T10* I-7). The SYCL wrapper (`make_event`, `ownership::keep`) is made per copy, never kept (§7.2). The `zeEventPoolCreate`/`zeEventCreate` site is a new native Level Zero object site in the backend, so T1 gives it an allowlist line in the canonical contract, next to the `HOST_CPU` wrapper (r5 m-15) | none |

- **Class C, charged at the context transaction** (*lead rulings §T9, §T10* I-1,
  r5 m-13). Every row whose scope is "context" scales with that context's own
  `n_ubatch` and its in-flight depth, so it is class C: sized at the context's
  transaction from the context's own values, never from the plan's `n_ubatch`,
  and never carried in a model-scope plan field (§4.1). The context records its
  `in_flight_depth`, its `D` and the guard input that sets the depth (§5.3) in
  the same transaction. On the auto-ubatch path the transaction runs once per
  candidate (§5.3, "Which transaction run creates the entry"), and each run
  sizes the class-C rows at that candidate's `n_ubatch`; they follow *lead
  ruling §M75*, as a candidate's SCRATCH growth does: charged to the candidate,
  kept only if it wins, and returned, charge and allocation, by the losing
  candidate's context transaction at its rollback, which is each losing return
  after the candidate's publish (`d8a67422d src/llama-context.cpp:1577`,
  `:1620`, `:1628`); the probe-stage returns before that publish (`:1531`,
  `:1534`, `:1537`, `:1540`) run no transaction and charge nothing (r31 n-2).
  After the ladder the class-C bytes equal the winner's need, never the losers'
  high-water mark (P4). No class-C witness exists in §11 to extend; T7's
  transaction arms are where one would go.
- **STORAGE** uses the cache slab and index, the landing slab, the epoch ring, a
  share of the pool's block ring and the output slot. Those are the **entire**
  resident footprint of a STORAGE table, and the planner charges exactly those
  bytes (P4).
- **A CPU-executed consumer's table** (*lead ruling §M58b*; r23 I-2 (d)) uses
  the right-hand column instead: HOST the body alone; STORAGE the cache slab and
  index, one landing slot and its share of the block ring. It has no epoch ring,
  no output slot, no marker words and no events, so it charges nothing
  device-side and no `HOST_PINNED` byte, and the planner charges exactly those
  bytes (P4).
- **HOST** uses the body, the epoch ring and the output slot. Its worker
  dequantizes straight from the body (§7.2), which is what the 4.2 us/token
  measurement did, so it needs no landing slab and no ticket.
- **Both residencies capture the ids** (*lead rulings §T6* I-1, *§T8*). The rows
  input sits at the same `gallocr` address every ubatch and `set_input(k+1)`
  overwrites it with no sync (§7.1), and the worker now runs after `prefetch`
  returns. So `prefetch` copies the ids into the ring entry, and the worker
  reads only that. A CPU-executed consumer's table captures nothing: its
  `prefetch` and its op both read the rows input before the next `set_inputs()`
  can run (§5.3's CPU form).
- **The output slot is pinned** because the device's copy engine reads it. It
  is the host end of the explicit H2D copy (§7.2 step 3, §7.3), which is DMA
  staging, the one GPU use `HOST_PINNED` allows (§3.1). No kernel reads or
  writes it.
- **The marker words are a device-written completion flag, sanctioned here**
  (*lead ruling §T6*, m-13). A marker kernel after each output copy stores the
  epoch into its ring entry's word, so the host can tell, without blocking,
  that the copy is done (the non-blocking predicate of `llama.cpp-c6ah`). It
  follows the canonical contract's completion-flag precedent (§3.1, the
  oneDNN Graph-scratch flag slab) in kind only: it is a separate, ordinary
  owner-first allocation from `unified_allocate_owner()`, not `CACHE_BACKING`
  (which only that contract's two allowlisted sites may mint), and it holds
  flags, never data. A census of host memory read or written by device
  kernels lists it under that name.
- **Not a th32 allocation: the cold-expert landing slab.** The owner's §D4
  cold set is read by `llama.cpp-yl3v` into its own planned, cache-owned
  `O_DIRECT` slab. *Lead ruling §D6* (I-4, amending §D5) charges it as `2 x`
  the largest per-layer cold bytes (double-buffered; one fully cold layer
  makes it 2 x 4.58 = 9.16 GiB on Q2_K) inside the cold-set computation; §4.2
  lists it as a term so that `host_planned_bytes` carries it.

### 3.4 Pointer discipline

The gather obtains raw views (a chunk of the HOST body, a row cache slot, a
landing-slab slot, an output slot) by resolving the owning handle **inside**
the gather call, and never stores them. That is the same rule as the canonical
contract §4, and the new resolution entry point is added to its allowlist. No
table is keyed by a pointer: the row cache is keyed by `(row index)` within its
`lazy_table`, whose identity is the weight's `ggml_sycl_cache_id`.

---

## 4. The planner decides the residency (item 2)

### 4.1 Inventory

The loader gives lazy tensors to the SYCL inventory as a **distinct class**,
`placement_lazy_table_info`. It carries the name, type, `ne`, `row_size`,
`rows_per_token` and `consumer_layer`, and it takes the table's byte range, file
index, offset and size, by value from `llama.cpp-5efe`'s per-file lazy-range
record (below), never derived independently, from `llama_tensor_weight::offs` or
from `ggml_nbytes` (*lead ruling §F2a* I-9 addendum 2; r19 N-1). They are never
placement entries for a device: P3 forbids a device tier for them, so the
planner cannot offer one. They are also **excluded from `max_tensor_bytes`** and
every other inventory maximum: with a 30 GiB engram table counted as
`max_tensor`, the host STAGING zone alone would be ~60 GiB (§2.3). T3's RED
fixture plans a Q2_K-shaped inventory with its two tables and asserts that the
STAGING and SCRATCH zones equal what `populate_host_zone_sizing()` returns for
the same inventory **without** the tables. The expected bytes come from the
sizing function, never from a literal (*lead ruling §D5*): today that is about
3.65 GiB of STAGING, and it moves when `s4ip` lands. Counting a table would add
at least twice the table's bytes, so the check cannot pass by accident.
`rows_per_token` and `consumer_layer` are arch facts that `llama` supplies
(§9.1), returned for each declared name by the per-arch attribute function at
5efe's pre-fill (below; *§F2a* I-9 addendum 5):

| model | table | `rows_per_token` | `consumer_layer` |
|---|---|---:|---:|
| qwen4exp | `per_layer_token_embd.weight` | 16 (`ple_n_heads`) | the one set bit of `hparams.is_ple_impl` (1 on this GGUF) |
| gemma4 | `per_layer_token_embd.weight` (the member `per_layer_tok_embd`) | 1 (token-indexed) | 0 |

deepseek41's two tables, `blk.1.engram_embd.weight` and
`blk.14.engram_embd.weight` (24 rows per token each, `(max_ngram_size - 1) x
head_count`; consumer layers 1 and 14), join the declaration when
`llama.cpp-9g22` lands that arch: it does not exist in the tree, so 5efe's C1
cannot list it (*§F2a* I-9 addendum 4; r20 N-1).

`consumer_layer` orders IO (§5.2), names the layer whose executor, read after
the late replan through §3.2's query, holds the table's descriptor buffer (a
SYCL device's for a SYCL-executed consumer, the host tier's for a CPU-executed
one; r23 N-3), and defines the prefetch window (§5.4).

On the `llama` side this happens where `lazy_read::add()` records the tensor
(`694ec2354 src/llama-model-loader.cpp:1088-1117`). The SYCL path takes the
tensor exactly when its record entry's `deferral` is `STORAGE` (below; *lead
ruling §F2a* I-9 addendum 5), with no taker test of its own, records it in the
inventory **instead of** adding its range to the mmap lazy set, and creates the
tensor in T6's own context, whose descriptor buffer is bound after the late
replan, not at `create_tensor` (§3.2; r23 M-1). **One lazy-range record per
loaded file** (*lead ruling §F2a* I-9), which `llama.cpp-5efe` owns. The loader
**pre-fills** it before the early plan, independently of T6, from a per-arch
declaration keyed by (arch, GGUF name) (*§F2a* I-9 addendum 5; r21 M-2),
`llama_model_lazy_tensor_names(arch)`, which returns `const
std::vector<std::string> &` of the GGUF names the arch creates with
`TENSOR_READ_LAZY`, each an exact name or one with a single `%d` for the block
index, and takes no hparams, and from each matched name's `weights_map` entry,
its (file index, offset, size) (5efe rev 3l `a68508606`
`docs/design/sycl-mmap-host-tier.md:566-569`; *§F2a* I-9 addendum 8). The arch
key is load-bearing: gemma3n creates the same `per_layer_token_embd.weight`
**without** the flag (`fefb92980 src/models/gemma3n.cpp:39`), so a name-only key
would give every gemma3n load an entry, and the drift check below would abort
it. `create_tensor` does not fill it; it **checks** it, and aborts naming the
tensor and the arch when a `TENSOR_READ_LAZY` call has no entry or a non-lazy
call has one, whatever the entry's eligibility and the mode, so the upstream
flag and the declaration cannot drift apart silently. That mode-independent form
is th32's requirement on 5efe (*§F2a* I-9 addendum 4; r20 M-2), which 5efe's rev
3a narrowed to a non-`EAGER` row, a form that under `-lzm off`, or `AUTO` at or
below 4 GiB, lets a declared name created without the flag pass. §9.1's
`rows_per_token` and `consumer_layer` are hparams-valued, so they attach at the
pre-fill, for each matched name, through **one** named per-arch function that
reads the loaded hparams, 5efe's, next to the declaration:
`llama_model_lazy_tensor_attrs(arch, hparams, name)`, returning
`llama_lazy_tensor_attrs { uint32_t rows_per_token; int32_t consumer_layer; }`,
where 0 and -1 mean "not known" (addenda 5, 6 and 8). The pre-fill stores the
result on the per-model record entry, never in a static table. C1 lands the
qwen4exp and gemma4 values, the table above (5efe rev 3l `a68508606`
`docs/design/sycl-mmap-host-tier.md:600-610`). qwen4exp's `consumer_layer` is
the index of the one set bit of `hparams.is_ple_impl`, which the qwen4exp loader
sets from `qwen4exp.ple.layers` and refuses for more than one entry (`fefb92980
src/models/qwen4exp.cpp:65-84`); it is 1 on this GGUF, and a literal 1 would be
a second source (r24 N-1). T6 reads the values from the record entry (*§F2a* I-9
addenda 7(a) and 8(a); *lead ruling §M58c*; r23 M-4). A `deferral == STORAGE`
entry for which that function returns a "not known" `rows_per_token` or
`consumer_layer` is refused before the early plan reads it, naming the tensor;
T6 owns that refusal, and a non-taker entry, which has no use for the
attributes, is never refused for them (addendum 5; r21 N-2). Each entry also
carries one **`lazy_eligible`** bit (*lead ruling §F2a* I-9 addenda 3 and 4),
computed once, at the pre-fill, by th32's DLM (a) predicate (T6a) from a lazy
mode, the entry's size and whether mmap is supported, with the 4 GiB floor under
`AUTO`. **The mode input** (addendum 4, r20 I-1): 5efe's C1 captures the lazy
mode before its moved `AUTO`-to-`OFF` fallback block, and the pre-fill passes
that pre-fallback mode when the SYCL branch is the **taker**, so the fallback
cannot disengage the branch (DLM (a)); otherwise it passes `lazy_read`'s
resolved, post-fallback mode, so that on a load the SYCL branch does not take, a
table `lazy_read` reads eagerly is not treated as lazy by the prefetch
exclusion. **The taker** (*§F2a* I-9 addendum 5; r21 I-1) is the per-model fact
the early plan is already gated on, `sycl_model_loading_guard.active`
(`fefb92980 src/llama-model.cpp:2214-2230`): the model's weights are SYCL-owned
(`n_gpu_layers() > 0` with a SYCL model device, or a SYCL buffer-type override)
and the hooks resolve. C1 captures it once at the pre-fill, beside the
pre-fallback mode. It is **not** `llama_model_sycl_hooks_enabled()`, which says
only that a hook-exporting SYCL backend is registered in the process: under it,
a `-ngl 0` load in a SYCL build would get the bit set and a SYCL descriptor on a
CPU-owned load with no early plan. T6a's third input is `llama_mmap::SUPPORTED`
on the non-taker path, so a platform without mmap yields a clear bit there, and
`true` on the taker path, whose reads go through the SYCL branch's own
`O_DIRECT` fd (§5.1) and never through a mapping (DLM (a); *§F2a* I-9 addendum 6
(1)). `SUPPORTED` is a constant `true` on Linux and Windows (`fefb92980
src/llama-mmap.cpp:710-714`), so no runtime arm on those hosts can tell the two
inputs apart; a source-contract clause in C1's source gate therefore pins the
pre-fill's third argument as `taker ? true : llama_mmap::SUPPORTED`, with two
mutants that must each fail it: the taker passing `SUPPORTED`, and the non-taker
passing `true` (r22 M-1). th32 states the clause as a requirement on 5efe's C1,
and the lead relays it (*lead ruling §M58c*); 5efe folded it as
`test-llama-lazy-range-record-source` (rev 3l `a68508606`
`docs/design/sycl-mmap-host-tier.md:4691`). The bit is the one eligibility fact,
and 5efe's `deferral` carries only the consumer kind: `STORAGE` exactly when the
entry is eligible and the SYCL branch is the taker, `LAZY_READ` exactly when it
is eligible and the SYCL branch is not, and `EAGER` otherwise (addendum 5), so
`deferral != EAGER` exactly when `lazy_eligible` is set. `STORAGE` here names
the SYCL lazy-table consumer, whatever residency, HOST or STORAGE, the planner
later gives the table (§4.3). On the non-taker path `lazy_read::add()` **reads**
that fact, deferring exactly the `LAZY_READ` entries, and no longer re-evaluates
the mode, the floor or `SUPPORTED` (`fefb92980
src/llama-model-loader.cpp:1241-1255`); that is a fork change to upstream code,
and a source gate pins it, so a rebase that restores add()'s own gates fails
loudly (addendum 5; r21 M-3). th32 states it as a requirement on 5efe's C1,
which fills `deferral`. Every consumer that acts on "lazy" reads the bit and
never re-evaluates the predicate: T6's SYCL branch at `create_tensor` (through
`deferral`), `lazy_read::add()` on the non-taker path (likewise), the early
plan, 5efe's tier and map-segment exclusion, and llama_mmap's
`WILLNEED`/`POSIX_MADV_RANDOM` prefetch exclusion. An ineligible entry is an
ordinary weight everywhere: placed, mapped and prefetched as one. The
inventory's byte ranges, `lazy_read`'s set, 5efe's tier exclusion and
llama_mmap's prefetch exclusion all consume that record, and none derives the
ranges independently. **`lazy-range-missing`** (r20 M-1): the SYCL branch looks
a taken table's range up by name; the not-found branch of that lookup, whatever
source the lookup reads, refuses the load naming the GGUF tensor and its
`weights_map` file index (`llama_tensor_weight::idx`; T6 (j)'s source clause
forbids only `offs` and `ggml_nbytes` for the range). On GREEN it is
unreachable, because the drift check aborts a lazy call with no entry first; it
is defensive, and its one witness is T6 (j)'s RED. Membership stays flag-based:
the record holds **every** `TENSOR_READ_LAZY` tensor, whatever upstream
`lazy_read::add()`'s three gates decide (the mode `OFF`, the `AUTO` 4 GiB floor
and `!llama_mmap::SUPPORTED`, `fefb92980 src/llama-model-loader.cpp:1241-1255`),
and eligibility is the bit's alone; upstream's add() is today the only filler,
behind all three (after C1, add() reads `deferral` and has no gates, above),
which is why the record is 5efe's new object and not `lazy_read`'s set (r18
M-2). So keeping the table out of `lazy_read`'s set does not empty the
exclusion. If the exclusion read that set instead, `init_mappings(true)` under
`use_mmap` would prefetch the whole file (`prefetch_size` -1, `694ec2354
src/llama-model-loader.cpp:1429`) and `WILLNEED` its complement of the lazy
list, the table included (`694ec2354 src/llama-mmap.cpp:500-503`), and for a
file whose lazy list is empty it would also set `MAP_POPULATE`, a synchronous
prefault (`:480`): the hazard *§T13a* forbids (r18 N-3). The mirror hazard runs
the other way (r18 N-4): `lazy_read`'s membership, `has()` and `any()`, stays
its own and excludes the table on the SYCL branch, and it consumes the record's
ranges only for its own members. Were `any()` re-pointed at the whole record, it
would be true on the SYCL branch, and `init_mappings` would map every shard even
with `use_mmap` cleared (`694ec2354 src/llama-model-loader.cpp:1412`), the
loader-semantics change for which *§T13c* withdrew *§T13b*. The early plan
(`llama_model_sycl_compute_early_plan`, `fefb92980 src/llama-model.cpp:2230`)
runs before `create_tensor` and builds its inventory from `weights_map`, which
is why the record is pre-filled ahead of it. It reads the record and applies DLM
(a)'s eligibility to its entries through the `lazy_eligible` bit, the same bit
T6's SYCL branch reads at `create_tensor` (the pre-fallback mode and `true` for
mmap support when the SYCL branch is the taker,
`sycl_model_loading_guard.active`, and the 4 GiB floor under `AUTO`). It never
places or stages an **eligible** entry as an ordinary weight, and it places an
ineligible one (`OFF`, or `AUTO` at or below 4 GiB) as the ordinary weight it is
(*§F2a* I-9 addenda 2 and 3; r19 M-1). On the merge unit's tree, which carries
5efe's C1-C4 (T6), run 1 of T6 (j) is the witness for the ineligible case, and
T6 (k) for an eligible table under the fallback (addendum 4); the early-plan
assertion in (j) is 5efe's. The STORAGE tier reads the table through its own
`O_DIRECT` fd (§5.1) and stays off mmap entirely, so upstream's load-time
exclusions as they stand, which read the `lazy_ranges` that `init_mappings`
passes from `lazy_read`'s set through `lazy.for_file(idx)` (`694ec2354
src/llama-model-loader.cpp:1431-1432`, `src/llama-mmap.cpp:469-510`), do not
cover it on the SYCL branch; today nothing needs them, because the fork's SYCL
model path clears `use_mmap` whenever `sycl_model_backend` holds, whatever the
load mode (`c69d5774d src/llama-model.cpp:2558-2561`, `fefb92980
src/llama-model.cpp:2693-2696`), before `init_mappings` runs (`c69d5774d :2564`,
`fefb92980 :2702`), so no weight is loaded from a mapping and no mapping is
built (`694ec2354 src/llama-model-loader.cpp:1412`). `mmap_support` is not the
reason: upstream's check clears `use_mmap` only under `LLAMA_LOAD_MODE_AUTO`
(`694ec2354 src/llama-model.cpp:1508-1517`), so an explicit `-lm mmap` passes
it. That disable is what 5efe removes, and the requirement below binds from then
on (r15 m-5). **Interface requirement on `llama.cpp-5efe`** (*lead ruling
§T13a*, *§T13c*): once 5efe maps model files on SYCL, its file-mapped tier must
never populate, `WILLNEED` or prefault the range of an **eligible**
`TENSOR_READ_LAZY` entry, one whose `lazy_eligible` bit is set (*§F2a* I-9
addendum 3), and preferably does not map one at all, with a residency gate of
its own that shows it. Its exclusion, and llama_mmap's prefetch exclusion
beneath it, read the per-file lazy-range record above, never `lazy_read`'s set
(*§F2a* I-9), and act on its eligible entries only, by the bit; an ineligible
entry is mapped and prefetched as the ordinary weight it is. Its mlock growth
skips eligible lazy **ranges**, not only lazy tensors, and its residency gate
also runs under mmap with mlock: `llama_mlock::grow_to` locks the mapping's
prefix `[0, target)` (`694ec2354 src/llama-mmap.cpp:790-803`), and the loader
skips growth only for the lazy tensor itself (`694ec2354
src/llama-model-loader.cpp:1661`), so any mapping-backed tensor after the table
in its shard would lock, and so prefault, the whole range; llama_mmap's
exclusion path governs `MAP_POPULATE`, `WILLNEED` and `POSIX_MADV_RANDOM`
(`694ec2354 src/llama-mmap.cpp:480`, `:500-507`), not mlock growth. Qwen's table
fills its own shard after the header (Appendix A.1), so it is immune; DeepSeek's
tables share shards with later mapping-backed tensors (§2.1.2), so they are
exposed. This is an upstream defect, not th32's to fix (r15 m-3). Otherwise the
50.66 GiB table would be faulted into the page cache at load, duplicating
STORAGE's `O_DIRECT` path, moving `MemAvailable` during the P3 decision and
evicting other tenants' cache. th32 adds no mechanism for it. T6's edge to 5efe
is an order edge for this requirement and for `mmap_support`, and a data
dependency only on the lazy-range record, which T6's inventory reads (*§F2a*
I-9). A second data edge runs the other way: 5efe's C1 pre-fill calls T6a's
predicate, so T6a lands no later than 5efe's C1 (*§F2a* I-9 addendum 3). **What
keeps `load_all_data` off the table is T6b's skip, and only that** (r15 m-2,
P4). The table is not in `lazy.has()`'s set, but that does not protect it:
`from_mapping` is `use_mmap || lazy.has(cur)` (`694ec2354
src/llama-model-loader.cpp:1640`), so under `use_mmap` the table would take the
mapping branch and reach the descriptor's `set_tensor` (`:1669-1670`), and
without `use_mmap` the file-read branch (`:1672` on) would do the same, and the
descriptor aborts either way (§3.2). So the skip sits at the top of the
per-tensor loop, before `from_mapping` is computed, keyed on the SYCL branch's
inventory entry for the tensor, never on the lazy-range record, which holds
every `TENSOR_READ_LAZY` tensor and not only those the SYCL branch took, and it
adds the tensor's bytes to `size_done` (`:1750`) before it continues, because
`size_data` counts every weight (`:1443-1446`) and the last-call cleanup and the
final `progress_callback(1.0)` run only when `size_done >= size_data`
(`:1776-1793`). Nothing ever reads `tensor->data` (§7 replaces every consumer).
The memory-fit pass (`no_alloc`) and the real load take the same decision, which
preserves upstream's invariant at `694ec2354
src/llama-model-loader.cpp:1342-1345`.

**The SYCL branch does not depend on `mmap_support`.** The SYCL path reads the
table through its own `O_DIRECT` fd, so whether a device can map files is
irrelevant to it. One change follows, and it is 5efe's C1, not T6 (*lead ruling
§F2a* I-9 addendum 4; r20 I-1): C1 captures the lazy mode *before* its moved
`AUTO`-to-`OFF` fallback block, and the pre-fill feeds that pre-fallback mode to
T6a's predicate for a table the SYCL branch will take, so the fallback cannot
disengage the branch (§2.3, §4.1). The SYCL device's `mmap_support` itself
belongs to `llama.cpp-5efe`, which restores file mapping on SYCL; th32 does not
set it. For the field, T6's edge to 5efe is an order edge, not a data
dependency, since T6's branch does not read it (*lead ruling §T13*, *§T13c*).
T6's RED: with `AUTO` and a device reporting `mmap_support = false`, the
dual-card configuration still yields `[PLACEMENT] lazy ... residency=...` lines,
and T6 (k) checks the bit and the early plan in the same configuration,
host-only.

### 4.2 One host ledger, and the planner's host budget

*Lead ruling §T1*, I2: one host-budget function, read by every consumer. This
design makes it a **process-global host ledger** in `unified-cache.cpp`:

- **The budget.** `host_ledger_budget()` =
  `min(total x pct, MemAvailable - max(8 GiB, total/32))`, the formula the
  pinned pools use today (`unified-cache.cpp:4147-4259`), or the
  `GGML_SYCL_HOST_RESERVE_MB` override when its strict parser accepts it.
  `MemAvailable` is **sampled once per plan publish**. Because a later sample
  already excludes this process's own resident bytes, a later model's budget
  is `ledger_committed + (MemAvailable_now - os_reserve)`, capped by
  `total x pct`.
- **Every host byte charges it.** Pinned-pool chunk growth, every `HOST_CPU`
  allocation, and every row-cache chunk ask the ledger first. The pools stop
  carrying caps of their own, so in `PER_DEVICE` mode the two pinned pools are
  capped by their **sum**, not each by the whole budget.
- **One fact, one source** (*lead ruling §T13*). `host_ledger_budget()` is the
  only host budget. When `llama.cpp-5efe` lands, it charges its pinned bytes
  into it, and page-cache-backed `FILE_MAPPED` bytes are not charged; any
  headroom change it needs goes into this one formula, and its host-fit gate
  extends or depends on `llama.cpp-7v5c` rather than duplicating it.
- **Enforced at allocation, predicted by the plan.** An allocation the ledger
  refuses after a plan said "fits" is a plan-versus-reality defect: it logs
  `[HOST-PLAN-BUG]` at ERROR with the plan's and the ledger's numbers, then
  refuses. It never falls back to an unaccounted allocator.
- **Which cache owns lazy-table memory in `PER_DEVICE` mode.** The cache of
  the device that executes the table's consumer layer (§3.2). Its registry
  holds the records, and its shutdown census counts them. Every byte still
  charges the one process-global ledger.
- **`GGML_SYCL_NO_PINNED`** disables the pinned **mechanism**, not the budget.
  The plan is the same plan, and host weights are still charged at their exact
  bytes wherever they then live. The `HOST_CPU` tier is not pinned, so
  `NO_PINNED` does not touch a lazy table. No model is refused because
  `NO_PINNED` is set.
- **`GGML_SYCL_HOST_RESERVE_MB`** *is* the budget (it is named "reserve" but,
  since `llama.cpp-glkg`, it overrides the budget, `:4152-4198`). A value too
  small for a model refuses that model, which is the override doing its job,
  with the refusal naming the variable. A malformed value is rejected with one
  WARN and the computed budget is used. A value above the computed budget logs
  one WARN naming both. No other variable changes the budget, so no model is
  refused because of an unrelated override.

`compute_placement_plan` and `compute_multi_device_plan` gain the ledger's
budget as an input. The plan gets:

```
size_t host_budget;                 // the ledger's budget at this publish, recorded
size_t host_planned_bytes;          // everything below, summed
std::vector<placement_lazy_table> lazy_tables;
struct placement_lazy_table {
    std::string    name;
    lazy_residency residency;       // HOST or STORAGE
    int            consumer_device;
    size_t         table_bytes;     // file bytes
    size_t         host_resident_bytes; // HOST: table_bytes; STORAGE: cache slab + index
    size_t         cache_slots;     // STORAGE only
    size_t         cache_floor_slots;
    // no per-context bytes here: §3.3's context rows are class C, charged at
    // the context's own transaction (§3.3; lead ruling §T10, I-1)
};
```

<a id="th32-host-fit"></a>
**T2's definition** (it depends on `llama.cpp-rnh3`; its enforcing phase also
depends on `llama.cpp-sgc5`):

`host_planned_bytes` is what the sizing code will commit, not a second
estimate of it. Its zone terms are the bytes `populate_host_zone_sizing()`
and `ggml_sycl_configure_host_zones_for_plan()` produce, term for term (§2.3):

  the WEIGHT zone: the host weights at their **exact** materialized bytes
  (`rnh3`: the same function as `ggml_sycl_weight_alloc_bytes`, including
  layout padding, with **no** ratio), plus `rnh3`'s named terms (the 64 MiB
  WEIGHT runtime reserve, `HOST_COMPUTE` requests such as `output_reserve`'s
  logits and the server's prompt-checkpoint reads, and a TLSF fragmentation
  bound derived from the block count)
+ the KV zone: the host KV, at least 64 MiB
+ the STAGING zone, as sized at `unified-cache.cpp:27685-27699`:
  `max(2 x max_tensor, max_staging_pair_bytes) + 2 MiB`, plus the 16 MiB
  headroom, the expert bias, `moe_routing_ids`, `tp_staging_buffer_bytes` and
  the MMID host pool
+ the SCRATCH zone, as sized at `:27716-27742` and raised at
  `ggml-sycl.cpp:5634-5636`: every term §2.3 lists, `dma_staging_pool` and the
  PP MoE oneDNN ring included on today's code
+ the RUNTIME pool that `pre_allocate_runtime_chunks` commits
  (`ggml-sycl.cpp:5649-5653`), the ring and the DMA pool included on today's
  code. Lever (s) removes `dma_staging_pool` entirely, with no device charge
  either: weight streaming cannot engage under an arena, so the pool has no
  legal consumer (*lead ruling §D6*, I-1)
+ each pool's rounding up to its commit chunk (2 GiB for the pinned pools),
  because the ledger charges committed chunks
+ the cold-expert landing slab, `2 x` the largest per-layer cold bytes of the
  cold set `C` (up to 9.16 GiB on Q2_K), and **minus** those `C` bytes from
  the host experts (owner §D4, *lead rulings §D5, §D6* I-4; `C` and the slab
  are 0 for a model with no cold set)
+ the **context envelope**: per context, the host tenants a context adds (the
  CPU backend's `SYCL_Host` compute buffer, which `llama.cpp-zhcn` plans; host
  KV growth; the lazy tables' per-context rows of §3.3, which is the one
  list),
  times the planned context count
+ every lazy table's `host_resident_bytes`.

Two terms are deliberately absent. `cpu_quant_buffer_bytes` is
diagnostic-only (`:27436-27446`), and `moe_cpu_expert_staging` is counted once,
inside SCRATCH.

Three details make "term for term" hold:

- **The SCRATCH raise reads live cache state.** `ggml-sycl.cpp:5634-5636`
  takes the max with `cache->onednn_weights_scratch_size() +
  onednn_activations_scratch_size()`, and a host-only dry-run has no cache.
  Those are the live sizes `reserve_onednn_scratch` holds
  (`unified-cache.cpp:17514`, `:17863-17865`, `:18003`) for the oneDNN PP
  matmul scratch reserved at dispatch (`acquire_onednn_pp_scratch`,
  `ggml-sycl.cpp:1480-1497`): `llama.cpp-23mk`'s `onednn_pp_w` and
  `onednn_pp_a` (*lead rulings §V13* I-2, *§Z15*, *§T6* m-4), not `moua`'s
  `onednn_scratchpad`, which is already in the RUNTIME pool (`:5649`) and
  would be counted twice. At a load's first configure nothing has dispatched,
  so the code reads 0, and T2 and 7v5c substitute exactly that: whatever the
  cache holds at the configure, which is 0 on a fresh load.
- **The ledger charges backing bytes.** Chunk growth charges
  `footprint.backing`, not `footprint.usable` (`pinned-pool.cpp:889-895`), so
  T2 does too.
- **A short pre-allocation becomes a refusal.** Today
  `pinned_chunk_pool::pre_allocate` logs a WARN and carries on when the budget
  runs out (`pinned-pool.cpp:241-246`), so a zone set can be committed in part
  without anyone noticing. Under T2's enforcing phase that shortfall is the
  typed refusal `host-budget-exceeded`. In the WARN-only phase it is logged
  with the plan's number next to it.

**One zone set per plan (`llama.cpp-sgc5`).** Today every participating cache
builds a full zone set, WEIGHT included, from the one plan (§2.3). T2 does not
sum that. It counts the WEIGHT zone once, in the cache whose `CpuExpertPool`
leases the host experts, and STAGING, SCRATCH and RUNTIME once per cache that
sgc5's fix still gives them (from that device's own maxima). Until sgc5 lands,
the ledger's sum cap turns the duplicate WEIGHT zone from an OOM into a
refusal that no lever can clear.

**T2 has two phases** (*lead ruling §D6*, I-3). Phase (a) is the WARN-only
single ledger: it computes `host_planned_bytes` and logs what it would refuse,
and it proceeds now (*lead ruling §T3*, (c)). It depends on none of sgc5,
`llama.cpp-moua` or `llama.cpp-zhcn`: with one card there is no PER_DEVICE
sum to cap, and `GGML_SYCL_HOST_RESERVE_MB` is its only budget. Phase (b) is
the enforcing phase with the sum cap, and waits for sgc5, moua, zhcn and
s4ip.
Phase 1 of `llama.cpp-9g22` runs on the B70 alone (`level_zero:0`), which has
one cache and one zone set whether or not sgc5 has landed, so its gates (its
T13) need phase (a) only; P3 and the P4 watchdog are their enforcement.

**One source for the number** (*lead rulings §T3, §D5*). `llama.cpp-7v5c`
(9g22's T27) is the host-only plan dry-run: `compute_multi_device_plan`,
`populate_host_zone_sizing` and the `configure_host_zones` sizing, run over the
GGUF header inventory and recorded device budgets, with no model and no GPU work
of its own; on a host with devices it still initialises SYCL, since the
planner's shaped path calls `layout_policy::get_optimal` and so
`ggml_sycl_info()` (T6 (n)), which is why the fixtures of T2 and T3 that run it
are lead-run (§11; r30 m-2). It also computes the cold set `C` for the launch's
reading (§4.3). T2's `host_planned_bytes`, 7v5c's printout and `9g22`'s
`expected_host` are one computation through the same functions. Hand arithmetic,
including §4.5's, is an estimate that 7v5c replaces. It has already missed terms
three times: the r1 text charged the diagnostic-only `cpu_quant_buffer_bytes`,
the r2 text left out the DMA staging pool and the RUNTIME pool, and rev 2a left
out the ring.

**The readings are samples, not a shared fact.** This document took four
`MemAvailable` readings (171.4, 181.8, 183.7 and 190.4 GiB); `9g22` cites
171.4 and 189.6. Both are hand samples of a quantity that moves with the other
tenants. The number that decides a load is the one 7v5c takes at that load, so
neither document's table is a gate.

### 4.3 The decision, in order

1. VRAM is placed exactly as today: dense first, then experts.
2. Host weights (exact), `rnh3`'s named terms, host KV and every carved host
   zone are charged against `host_budget`. For a model with no cold-expert
   class, a plan that exceeds the budget refuses with a typed reason
   (`host-budget-exceeded`, carrying each term); today it silently
   overcommits (§2.3). For a model that has one (DeepSeek phase 1, owner §D4),
   the host experts are **not** refused here: step 5 moves the coldest of them
   to STORAGE.
3. The **context envelope** for the planned contexts is charged next, and
   `host_reserve = max(2 GiB, 1% of the budget)` is set aside. Context-lifetime
   tenants therefore come before any lazy-table byte (*lead ruling §T1*, I8;
   `llama.cpp-moua`'s lifetime segregation puts yieldable tenants last).
4. For each lazy table, in ascending `consumer_layer` (the tighter window
   first; ties go to the larger table), the residency is **HOST** only if both
   hold with the table resident: `table_bytes <= leftover` (the planner's
   fit), and **P3**, a predicted post-load `MemAvailable` of at least 24 GiB
   (`MemAvailable_pre - expected_host - 5 >= 24`, `9g22`'s gate
   precondition, which binds every load: *lead ruling §T6*, m-14). Then
   `leftover -= table_bytes`. Otherwise it is **STORAGE**, with a cache floor
   of `2 x n_ubatch x rows_per_token` slots (one double-buffered ubatch
   working set) plus its index. If what is left above the P3 floor cannot
   hold the floors, the plan refuses (`lazy-table-cache-below-floor`) rather
   than run with a cache that cannot hold one ubatch.
   **With a cold-expert class, steps 4 and 5 are one placement** over one P3
   budget, computed by `llama.cpp-7v5c` (*lead ruling §D6*, I-5). Routed
   experts stay host-resident before any table goes HOST. A table is HOST
   only if, with it HOST and the cold set **empty**, the predicted post-load
   `MemAvailable` stays at least 24 GiB (9g22's P3); otherwise it is STORAGE.
   The `leftover` rule above is not used there, and this step never spends
   the residual that step 5 sizes `C` to leave. P3 binds production loads and
   gate loads alike. On DeepSeek phase 1 one HOST table needs a
   `MemAvailable_pre` of about 228-232 GiB (§4.5), so both tables are STORAGE
   at every reading so far. Each STORAGE cache is then **fixed** at 2 GiB
   per table, index included (never below the floor), and charged in the
   one P3 budget before step 5's search (*lead rulings §T4* I-4, *§T5*);
   7v5c prints it.
5. **With a cold-expert class: the cold set** (owner §D4; *lead ruling §D5*,
   amended by *§D6*, I-4). `llama.cpp-7v5c` finds `C` from this load's
   `MemAvailable` reading by a **search** down the §D1 profile order, coldest
   first, in the source order `9g22` states once in
   [its §7 D1](deepseek41-runtime.md#7-owner-decisions) (*lead ruling §T5*).
   Each step moves the next expert-layer to STORAGE and recomputes the warm
   WEIGHT bytes, the main pool's exact chunk rounding and the landing slab
   (`2 x` the largest per-layer cold bytes). The search stops at the first `C`
   whose predicted post-load `MemAvailable` is at least 24 GiB (9g22's P3). The
   profile orders the experts; it does not size the set, so there is no separate
   selected size (§D6 withdraws `C_selected`). Everything charged in steps 2-4,
   the fixed caches included, is charged **before** `C` is sized, and `C`
   absorbs the remainder: row caches and the cold set never trade bytes at run
   time (*lead ruling §T4*, I-4). If even every host expert cold cannot clear
   P3, the load **refuses** as `cold-set-cannot-clear-p3`, naming the reading,
   the P3 threshold and the shortfall; it never loads below P3. The step is
   `llama.cpp-yl3v`'s, through 7v5c's computation; th32 consumes its `C` and
   slab bytes in `host_planned_bytes` (§4.2). **Without one** (Qwen): the
   STORAGE caches take what is left above the P3 floor as a **yieldable**
   tenant (§6.4), split between tables in proportion to
   `table_bytes x rows_per_token`, a proxy for miss traffic.

Nothing in this sequence reads an environment variable except the ledger's
budget override (§4.2). `--lazy-mode` keeps its upstream meaning of
**eligibility**: `OFF` means the tensor is an ordinary weight and never a lazy
table, and `AUTO`/`ON` pick the eligible set. The residency is always the
planner's (DLM, *lead ruling §T2*).

### 4.4 Interaction with host experts and later tenants

- **Memory.** Host experts are charged before lazy tables (step 2 before step
  4), which is the owner's ticket ruling ("after VRAM and experts"). What
  moves an expert to STORAGE is the host budget at P3 (step 5, owner §D4),
  never a lazy table. With a cold set, a table becomes HOST only if P3 holds
  with it HOST and the cold set empty (step 4, *lead ruling §D6*, I-5);
  without one, only out of what is left once the host experts, the zones and
  the envelope are charged.
- **Execution.** Both residencies' gathers run on the IO engine's workers
  (§5.3). A HOST one takes microseconds; a STORAGE one is parked on its ticket
  and holds no thread. The CPU cost is the dequant of the rows (~2-3 ms per
  PP512 ubatch per table, §2.4 and A.2), which does not compete with
  `CpuExpertPool`'s compute workers in any measurable way. The pread workers
  sleep in the kernel.
- **IO.** The cold experts and the engram tables live on the same bcachefs USB
  members, and share the one IO engine (§5.2). How much that costs is in §8.4.
- **Later tenants.** A table's residency is immutable after the plan is
  published: a live HOST table is never demoted. Its **row cache** is
  different, because it holds only clean copies of file bytes: no lease, no
  pin, and every in-flight row lives in a landing slab, not in the cache
  (§5.3). A later context or model that needs host memory the plan cannot
  otherwise find therefore **yields** row-cache chunks (§6.4), exactly as KV
  admission yields optional layout copies today
  (`unified-cache.cpp:7798-7911`, `yield_optional_layouts`). *Lead ruling
  §T1*, I8, settles that shrinking a clean-data cache is not the forbidden
  reclaim of live memory.

<a id="th32-host-fit-numbers"></a>
### 4.5 What the planner will choose on this host (through T2's definition)

**How to reproduce these numbers.** From the repository root, run
`python3 docs/design/sycl_storage_tier_host_fit.py`. It needs no packages, and
it prints every figure in this section: the ring, the per-set terms in both
columns, the Qwen rows, DeepSeek cases A-F, the P3 check, and a cross-check of
`9g22`'s cold set. It is hand arithmetic over the terms of §4.2, and
`llama.cpp-7v5c`'s dry-run replaces it.

**The budget.** `MemTotal` is 246.9 GiB, so `total x 90%` = 222.2 GiB and the OS
reserve is 8 GiB. `MemAvailable` read **171.4, 181.8, 183.7 and 190.4 GiB** at
four points today, moved by the other tenants (the qemu VM, Emby, Frigate), so
`host_budget` = **163.4-182.4 GiB**. The tables below state each case's
requirement as the `MemAvailable` it needs at load (requirement + 8 GiB), so
that it can be checked against any reading (§4.2: these readings are samples,
not a gate).

**Two columns.** *Today's code* is the sizing code at `c69d5774d`. *After (s)*
is lever (s) of the owner's §D4: the ring type-gated to MXFP4 experts (so 0 on
both models here) and removed from host SCRATCH and from the RUNTIME pool,
`dma_staging_pool` removed entirely, with no host SCRATCH, RUNTIME pool or
device charge (`llama.cpp-s4ip`; *lead ruling §D6*, I-1), and one zone set
per plan (`llama.cpp-sgc5`).

**The exact weights** (GGUF tensor headers, `ds_q2k_census.py` and
`qwen_census.py`, Appendix A.1):

| model | routed experts | other weights | lazy tables | total |
|---|---:|---:|---:|---:|
| Qwen3.8-Flash-Next Q8_0 | 119.53 GiB (largest tensor `blk.0.ffn_down_exps` 850 MiB) | 5.09 GiB (dense 3.83, `token_embd` 0.63, `output` 0.64) | 50.66 GiB | 175.29 GiB |
| DeepSeek-V4.1-Flash Q2_K | 183.25 GiB (4.58 GiB per layer; largest tensor `ffn_down_exps` 1856.25 MiB) | 3.01 GiB (dense 2.30, `token_embd` 0.20, `output` 0.51) | 60.08 GiB | 246.34 GiB |

**The device side: the B70 holds every device expert** (estimates; the gate-1
plan lines are the witness). Under the default EXPERT multi-GPU mode the fork
places no routed expert on the B50 at all. The static secondary route returns
`false` for every type (`unified-cache.cpp:24961-24974`, pending a PP
secondary-route equivalence gate), and the secondary executor exists only for
MXFP4, Q4_0 and Q6_K (`:24943-24958`). So `llama.cpp-9g22`'s phase 1 pins
`level_zero:0`, the B70 alone (its §4.5.1), and the same holds for Qwen's
Q8_0 experts.

- **After (s).** With a placement budget of ~29.9 GiB, and after dense weights
  (DeepSeek ~2.8 GiB), attention state and device zones of 1.5-4 GiB, the B70
  holds **23-26 GiB** of DeepSeek experts and, with Qwen's 1.46 GiB more dense
  weight, **21.5-24.5 GiB** of Qwen's. A two-card run is `9g22`'s T25, after
  sgc5; there the cards hold ~31-38 GiB of DeepSeek experts.
- **Today's code.** The ring sets the floor of each card's VRAM RUNTIME zone
  (§2.3). The expert room it takes is the ring less that zone's own 512 MiB
  floor: about 7.4 GiB on DeepSeek and 3.7 GiB on Qwen, which become host
  experts. The figure and its rule are `9g22`'s (rev 10, r4 m4:
  [its §4.5.2](deepseek41-runtime.md#452-the-host-fit-r2-c-a-r3-c-1-c-2-owner-d4)); the
  today's-code rows below use that rule, one source (*lead ruling §T6*, m-9).
- **Two VRAM terms are inputs, not estimates here.** They are the C-class terms
  charged at the context transaction: `llama.cpp-1oxa`'s D512 no-tile ONEDNN
  term (*lead ruling §V11.3*; DeepSeek's head_dim is 512) and
  `onednn_graph_scratch` (*lead ruling §M19*, owned by `llama.cpp-23mk`). A
  context that does not fit is refused (§V11.3), and planned expert copies are
  never yielded to it (§M18.3, §M18.3a). So both terms displace host experts
  1:1, and the plan must carry them before it places experts. 7v5c prints
  them; this section carries no number for them.

**The terms, per T2's definition** (GiB; `max_tensor` is 1.81 GiB for DeepSeek
and 0.83 GiB for Qwen, both `blk.0.ffn_down_exps`):

| term | source | Qwen today | Qwen after (s) | DeepSeek today | DeepSeek after (s) |
|---|---|---:|---:|---:|---:|
| host experts, exact | total minus B70 experts; today also the ring's VRAM displacement (the ring less the 512 MiB floor) | 98.7-101.7 | 95.0-98.0 | 164.7-167.7 | 157.3-160.3 |
| other host weights (`token_embd`, CPU `get_rows`) | header | 0.63 | 0.63 | 0.20 | 0.20 |
| `rnh3` named terms | WEIGHT reserve 64 MiB (`unified-cache.cpp:27103`) + `HOST_COMPUTE`; the TLSF bound is not yet defined | 0.1-0.5 | 0.1-0.5 | 0.1-0.5 | 0.1-0.5 |
| host KV | Qwen: attention on the card at the gate contexts; DeepSeek: the dsv41 module is non-demotable (`9g22` §2.4) | 0 | 0 | 0 | 0 |
| STAGING zone | `2 x max_tensor + 2 MiB + 16 MiB` + routing ids at `-c 4096` (§2.3; expert bias and MMID pool not yet known) | 1.69 | 1.69 | 3.65 | 3.65 |
| SCRATCH zone | `2 x max_tensor + 32 MiB` + oneDNN reorder ≈ 0.08 + `moe_q8` ≈ 0.03; today also `dma_staging_pool` (`2 x max_tensor`) and **the ring**; `moe_vram_runtime` and `pp_pipeline` not yet known | 7.63 | 1.80 | 15.26 | 3.77 |
| ... of which the ring | `llama-model.cpp:430-507` | 4.17 | 0 | 7.87 | 0 |
| RUNTIME pool, as committed | oneDNN scratchpad + `pp_pipeline` ≈ 0.09; today also the DMA pool and the ring again; 2 GiB chunks | 5.92 → 6 | 0.09 → 2 | 11.58 → 12 | 0.09 → 2 |
| KV zone minimum | `ggml-sycl.cpp:5626` | 0.06 | 0.06 | 0.06 | 0.06 |
| **non-weight host, one zone set, as committed** | | **15.4** | **5.5** | **31.0** | **9.5** |
| main-pool chunk rounding | the main pool commits 2 GiB chunks | up to 2 | up to 2 | up to 2 | up to 2 |
| **step 2, one zone set** (main-pool rounding excluded) | | **114.8-118.2** | **101.3-104.7** | **195.9-199.3** | **167.1-170.5** |
| `host_reserve` + context envelope | 2 GiB + <= 0.5 GiB (the class-C context rows of §3.3, charged at the context transaction: output slots Qwen 10 MiB, DeepSeek 48 MiB; landing slabs Qwen 2.66 MiB when STORAGE, DeepSeek 3.9 MiB; epoch rings at `D` = 3, Qwen 0.1 MiB, DeepSeek 0.3 MiB; plus the pool's block ring 0.5 MiB once, and the `SYCL_Host` compute buffer) | 2.5 | 2.5 | 2.5 | 2.5 |
| DSpark draft: its host effect on phase 1 | its ≈ 7.9 GiB of weights are **device** demand; with its ring they move ≈ 10.0 GiB of experts to the host (`9g22` rev 10, `054069548`, [its §4.5.2](deepseek41-runtime.md#452-the-host-fit-r2-c-a-r3-c-1-c-2-owner-d4)). What that does to the cold set is `9g22`'s figure, linked, not modelled here | - | - | +10.0 | +10.0 |
| cold set: credit and landing slab | `-C` from the host experts, plus `2 x` the largest per-layer cold bytes (*lead ruling §D6*, I-4): `2C/40` if the cold set is spread evenly, 9.16 GiB if whole layers are cold | - | - | - | `-C` + slab |

The oneDNN reorder and `moe_q8` figures are `9g22`'s estimates for DeepSeek,
reused for Qwen for want of better. Every row below excludes the main pool's
chunk rounding, so each requirement can be up to 2 GiB higher.

**These terms assume today's anonymous-copy host tier** (*lead ruling §T13*; r14
N-2, N-3). Every host weight above is charged as resident anonymous memory,
because that is how the fork holds host weights until `llama.cpp-5efe` restores
file mapping on SYCL (§2.3). 5efe lists which figures move, and th32 re-pins
them after the merge unit (5efe C1-C4 with T6 and T7, §11) lands and before T9q
runs; the re-pin gates T9q (r27 n-1). If its file-mapped tier holds host
weights, the rows that move are the two weight rows, the host experts (Qwen 95.0-101.7 GiB, DeepSeek
157.3-167.7 GiB across both columns) and the other host weights (0.63 and 0.20
GiB), and with them step 2, `expected_host`, the `MemAvailable_pre` needed, and
the P3 verdicts below, since unlocked page cache is partly counted in
`MemAvailable` and locked file pages are not. The non-weight zones,
`host_reserve` with the context envelope, and the table's own HOST bytes, which
stay anonymous and locked in `HOST_CPU` (§3.1), do not move. T1's tier-switch
census (36 sites in 7 files) and the "`MMAP_TRACKED`, legacy" wording of §2.3
and §3.1 are stated against today's `alloc_tier` too, and T1 re-derives them if
5efe adds or revives a file-mapped tier. `llama.cpp-7v5c`'s dry-run takes the
host tier as an input, and the budget stays the one of §4.2.

**Qwen (B70 alone, one zone set). P3 binds** (*lead ruling §T6*, m-14). The
table is HOST only if, with it resident, the predicted post-load
`MemAvailable` stays at least 24 GiB: `MemAvailable_pre - expected_host - 5
>= 24`, with `expected_host` = step 2 + the table (no row cache while it is
HOST). Main-pool chunk rounding (0-2 GiB) is excluded, as everywhere here.

| code | step 2 | `expected_host`, table HOST | `MemAvailable_pre` needed | 171.4 | 181.8 | 183.7 | 190.4 |
|---|---:|---:|---:|---|---|---|---|
| today | 114.8-118.2 | 165.4-168.8 | **194.4-197.8** | STORAGE | STORAGE | STORAGE | STORAGE |
| after (s) | 101.3-104.7 | 151.9-155.3 | **180.9-184.3** | **STORAGE** (post-load 11.1-14.5) | low corner only (21.5-24.9) | low corner only (23.4-26.8) | HOST (30.1-33.5) |

- The planner's own fit (`MemAvailable - 8`) would put the table HOST at
  171.4 after (s): it needs only 162.4-165.8 GiB. That leaves a post-load
  `MemAvailable` of 11.1-14.5 GiB, under P3 and under `9g22`'s 16 GiB
  watchdog, which is why P3, not the planner's fit, decides. At a "low
  corner only" reading the load's own reading and 7v5c's exact rounding
  decide; the H arm launches only where P3 holds (§8.1).
- On STORAGE, Qwen's caches are a yieldable tenant sized from what is left
  above the P3 floor (§4.3 step 5): 37.7-41.1 GiB after (s) at 171.4.
- A second card's zone set before sgc5 includes WEIGHT (case F below shows
  the DeepSeek size). After sgc5 it adds the non-weight terms again, 5.5-15.4
  GiB, and moves the threshold up by as much (*lead ruling §T6*, m-8).

**DeepSeek Q2_K, the planner's fit** (`MemAvailable - 8`; the verdicts are
after (s)):

| case | today's code: `MemAvailable` needed | after (s): needed | 171.4 | 181.8 | 183.7 | 190.4 |
|---|---:|---:|---|---|---|---|
| A: phase 1 (B70 alone), no draft, floor caches (cannot occur in a cold-set plan, whose caches are fixed) | 206.4-209.8 | **177.6-181.0** | refused | fits (0.8-4.2 left) | fits (2.7-6.1 left) | fits (9.4-12.8 left) |
| B: A + the fixed caches (2 GiB per table, index included) | 210.4-213.8 | 181.6-185.0 | refused | low corner only (0.2 left) | low corner only (2.1 left) | fits (5.4-8.8 left) |
| C: A + the experts the DSpark draft displaces | 216.4-219.8 | 187.6-191.0 | refused | refused | refused | low corner only (2.8 left) |
| D: two cards after sgc5 and `9g22` T25, non-weight zones counted once (a lower bound only: after sgc5 each card keeps its own STAGING, SCRATCH and RUNTIME, which is E). With a lazy table, admitted only when both consumer layers (1 and 14) run on the host buffer's card (§5.3) | - | 165.6-173.0 | low corner only (5.8 left) | fits (8.8-16.2 left) | fits (10.7-18.1 left) | fits (17.4-24.8 left) |
| E: D + the non-weight zones a second time; the same card rule as D | - | 175.1-182.5 | refused | low corner only (6.7 left) | fits (1.2-8.6 left) | fits (7.9-15.3 left) |
| F: two cards, code as it is today (a whole second zone set, WEIGHT included) | 393.1-407.9 | - | refused (T2) or OOM without T2 | | | |

On today's code every phase-1 case is refused at every reading. Lever (s)
recovers ≈ 28.9 GiB of what the load needs: the ring three times (twice on the
host, 15.7, and once as displaced experts, 7.4) and the DMA pool twice (7.3),
less 1.5 GiB that the RUNTIME pool's 2 GiB chunk rounding gives back (it
commits 2 GiB for 0.09 after (s), against 12 for 11.58 today). Case C counts
the draft's host effect as `9g22` states it: the draft's weights are device
demand, so its host bytes are the ≈ 10.0 GiB of experts it displaces (rev
2e carried the 7.9 GiB of weights here as host bytes, r5 m-12). What the
draft does to the cold set is `9g22`'s figure, linked, not restated (rev 10,
[its §4.5.2](deepseek41-runtime.md#452-the-host-fit-r2-c-a-r3-c-1-c-2-owner-d4)).

<a id="th32-p3"></a>
**DeepSeek Q2_K, 9g22's P3: the binding check** (*lead ruling §T4*, I-2).
The gate protocol launches only at a predicted post-load `MemAvailable` of at
least 24 GiB: `MemAvailable_pre - expected_host - 5 >= 24`, where
`expected_host` is step 2 as committed plus the gate caches (2 x 2 GiB), with
no draft, before any cold set, and without the main pool's chunk rounding
(0-2 GiB), as 9g22 §4.5.2 defines it. Its margin (29 GiB over
`expected_host`) is wider than the planner's (8 GiB, plus the 2.5 GiB reserve
and envelope), so **a plan that passes P3 also passes the planner**.

| code | `expected_host` | `MemAvailable_pre` needed | predicted post-load `MemAvailable` at 171.4 / 181.8 / 183.7 / 190.4 |
|---|---:|---:|---|
| today | 199.9-203.3 | 228.9-232.3 | -36.9 to -33.5 / -26.5 to -23.1 / -24.6 to -21.2 / -17.9 to -14.5 |
| after (s) | **171.1-174.5** | **200.1-203.5** | -8.1 to -4.7 / 2.3 to 5.7 / 4.2 to 7.6 / 10.9 to 14.3 |

P3 fails at every reading in both columns, so without a cold set the gates
cannot launch at all. `9g22` rev 10 §4.5.2 (`054069548`) agrees with both
rows:
- **Today's code:** its "B70 alone, today's code + rnh3" row (its r4 m1)
  gives `E₀` = **201.1-204.0 GiB**, against 199.9-203.3 above; both now take
  the ring's displacement less the 512 MiB floor. That closes the 8.4-12.4
  GiB gap rev 2b recorded against rev 5, whose 188.0-195.4 was two-card room
  misread as one card (*lead ruling §T5*).
- **After (s):** it predicts -8.9 to -6.0 GiB post-load at 171.4 (two cards
  -10.7 to -3.3), 0.8-1.3 GiB below the row above (rev 9 said -8.7 to -5.8).
  9g22 adds the RUNTIME zone's 512 MiB floor to the device side (its r4 m4),
  the 0.06 WEIGHT reserve (its m11) and, from rev 10, its class-C cold slots;
  the rest is hand rounding.

7v5c's printout is the one list of terms both documents cite (§4.2).

**An engram table HOST under P3** (§4.3 step 4, *lead ruling §D6*, I-5).
With the cold set empty and one table HOST, its 2 GiB gate cache is no longer
charged, so P3 needs `MemAvailable_pre` >= `expected_host` - 2 + 30.04 + 29 =
**228.1-231.5 GiB** after (s). No reading comes close, and both tables HOST
would need 256.2-259.6 GiB on a 246.9 GiB host. Both tables are therefore
STORAGE at every reading; `9g22` §4.5.2 reaches the same.

<a id="th32-d4-fit"></a>
**DeepSeek Q2_K with the owner's §D4 applied** (D8, *lead rulings §T3, §D5,
§D6*). The fit is step 2 after (s) plus the fixed caches, which is case B (case
A's floor caches cannot occur in a cold-set plan, *lead ruling §T6*, m-7), with
both engram tables on STORAGE (above), minus the bytes of the cold set `C` that
moves to STORAGE, plus its landing slab (`2 x` the largest per-layer cold
bytes). `C` is not a th32 choice: 7v5c finds it at load by the search of §4.3
step 5, from that load's reading, with the gate caches fixed before it (§T4,
I-4). The size it reaches at 171.4 GiB and at 189.6 GiB is `9g22`'s: its
**spread** band (every layer equally cold, slab `2C/40`) and its
**concentrated** band (whole layers cold, slab 9.16 GiB), in [its
§4.5.2](deepseek41-runtime.md#452-the-host-fit-r2-c-a-r3-c-1-c-2-owner-d4) (rev
7, the cold-set table); this document does not restate them. The calculator's
cross-check, run from the P3 row above, lands within about a GiB of both bands.
Because P3 binds, a load whose `C` clears P3 also fits the planner. With the
DSpark draft (production, after gates 1-4) `C` grows by `9g22`'s figure for it,
in the same section; this document does not restate it. The load refuses
(`cold-set-cannot-clear-p3`) only if every host expert cold still fails P3;
`9g22` §4.5.2 puts that far below any reading taken.

Where the row caches land on the hit-rate curves (§6.2.2): in a plan with a
cold set they are **fixed** at 2 GiB per table with the index included,
charged in the P3 budget before the cold-set search (§4.3 step 4; *lead
rulings §T4* I-4, *§T5*). That leaves each table ~1.7 GiB of 84 B slots
after its index, 44 M slots together, which is ~4.5 GiB in the simulation's
110 B units, between its 2 GiB and 8 GiB columns. That is the DeepSeek
operating point at every reading; the reading changes `C`, not the caches.

---

## 5. The STORAGE read path (item 3)

### 5.1 mmap page faults compared with `O_DIRECT` prefetch

Both engines were measured over the table's byte range (Appendix A.2):

| engine | concurrency | rows/s | per-row latency |
|---|---|---:|---|
| mmap faults (upstream path), `MADV_RANDOM` | 1 thread | 1555 | mean 0.64 ms, p99 2.3 ms |
| mmap faults | 16 threads | 7866 | mean 2.0 ms, p99 7.3 ms |
| mmap faults | 24 threads | 7370 | mean 3.3 ms, p99 17.7 ms |
| fio `O_DIRECT` `io_uring` 4 KiB | QD1 | 1389 | p50 0.48 ms, p99 2.7 ms |
| fio `O_DIRECT` `io_uring` 4 KiB | QD16 | 7815 | p50 1.8 ms, p99 6.8 ms |
| fio `O_DIRECT` `io_uring` 4 KiB | QD64 | 9124 | p50 5.7 ms, p99 25 ms |
| pread pool, `O_DIRECT` 4/8 KiB | 64 workers | 8344 | (batch of 7100: 0.85 s) |

**The device ceiling is the same for every engine.** On this filesystem the
bottleneck is the 256 KiB extent bounce onto USB links, not the syscall
mechanism. The choice of engine therefore rests on *timing*, *ownership* and
*accounting*, and on all three `O_DIRECT` prefetch wins:

1. **Timing.** A fault happens inside `get_rows`, on a compute thread, *after*
   the graph reached the node, and at an effective QD of however many
   `ggml-cpu` threads happen to fault at once. A prefetch is issued at
   `set_input`, before graph compute starts, at a QD equal to the number of
   misses, and it overlaps with the graph's earlier work (§5.4).
2. **Ownership (P1/P2).** A fault lands the page in the kernel page cache,
   whose allocator and reclaim policy are the kernel's. `O_DIRECT` lands the
   bytes in a cache-owned `mem_handle`.
3. **Accounting (P4).** Page-cache pages are reclaimable and invisible to the
   planner. Under memory pressure the kernel evicts them, and a table the plan
   calls resident silently becomes a storage read. Worse, a "HOST residency" backed
   by mmap is not HOST at all.
4. **Efficiency.** The page cache holds 4 KiB pages, 24 rows each, when one of
   them is hot. For the same bytes, a 4 KiB-page LRU hits much less than a row
   LRU (§6.2: 19.5% against 34.5% at 32 MiB on wikitext-2 train).

Buffered `pread` is rejected for reasons 2 and 3, since it fills the page
cache too.

### 5.2 IO engine

Two engines implement one interface (D2):

- **Portable default: a pread worker pool.** `QD_max` threads (64 on this
  host) each do blocking `pread(O_DIRECT)` into a block-ring slot. It needs no
  dependency and ports to Windows as `ReadFile` + `FILE_FLAG_NO_BUFFERING`. It
  measured 8.3 k rows/s on the Qwen table and 7.05 k on the DeepSeek Q2_K
  table, at the same-session fio QD64 ceiling (ratio 0.99, §2.4). The workers
  sleep in the kernel, so they cost no CPU.
- **Linux fast path (later): `io_uring` through raw syscalls.** It needs no
  liburing and adds no threads. It matters when the device is faster than the
  thread pool can drive: the ext4 NVMe reference reaches 110 k IOPS, where
  per-IO thread wake-ups start to show.

The rules both engines follow:

- Every miss becomes one 4 KiB-aligned read of the block(s) that contain the
  row: 4 KiB, or 8 KiB for the rows that straddle a block (4.1% of 170 B rows,
  2.7% of 110 B rows, 1.9-2.2% of 84 B rows). In general the read is
  `ceil((offset mod 4 KiB + row_size) / 4 KiB)` blocks, sized from the
  table's `row_size`, and the block ring's slot from the largest such span
  (§3.3): gemma4 E4B's PLE row (10752 x Q8_0 = 11424 B) is 3 or 4 blocks
  (r5 m-10).
- `O_DIRECT` alignment comes from `statx(STATX_DIOALIGN)`. If the filesystem
  refuses `O_DIRECT` (for example tmpfs or some FUSE mounts), the fallback is
  buffered `pread` followed by `posix_fadvise(DONTNEED)` on the range. It logs a
  once-per-process WARN that names the P4 cost.

<a id="th32-io-order"></a>
#### 5.2.1 IO ordering: the single source (*lead rulings §D10* I-3, *§T8* I-6)

This subsection is the one statement of the shared engine's priority and
deadline scheme. `llama.cpp-9g22` and `llama.cpp-yl3v` cite it and define no
rule of their own. **One pool per process**, with priority classes,
fairness and in-flight dedup (I7). A process-wide FIFO has head-of-line
blocking, and it was measured (`batch_probe3 hol`, A.2): a 24 + 24-row decode
token queued behind another context's 12000-row prefill completed in
**1359 ms** mean under FIFO, and in **21.7 ms** when it was placed ahead of the
waiting prefill reads (11.2 ms with nothing else queued; the remainder is the
prefill's reads already on the 64 workers). The prefill lost ~100 ms. So the
engine's queue is:

- **Class 0, the prefetched ubatches**, served before class 1. Every class-0
  read carries a key **(epoch, deadline)**. The epoch is the ubatch's epoch
  (§5.3); for a cold-expert slice (`llama.cpp-yl3v`) it is the ubatch that
  routed it. The deadline is the layer whose execution needs the read: a lazy
  table's `consumer_layer`, or the expert's layer. Within a context, class 0 is
  ordered by that key **lexicographically** (*lead ruling §T10*, I-4): every
  read of epoch `e` before any of epoch `e+1`, and **earliest deadline first**
  within an epoch. Reads with an equal key dispatch in the order they were
  queued, by a sequence number taken at enqueue (r7 m-6); a read that a join
  re-keys keeps its number. At an equal layer, lazy-table rows go before expert
  slices, because the engram consumer runs at the head of its layer and the
  expert FFN runs at its end. So DeepSeek's `blk.1` rows go before layer 2's
  cold slices, and `blk.14`'s rows go after the cold slices of layers 2-13.
- **Why the epoch comes first** (r5 I-4 (a)). The submitter may post up to
  `D` epochs before the device finishes the oldest (§5.3), so "the current
  ubatch" alone is not defined. Keyed by layer only, epoch `e+1`'s `blk.1`
  rows would be served ahead of epoch `e`'s `blk.14` rows and its layer 2-13
  slices while the device is stalled on epoch `e`: no deadlock, since IO
  never waits on the device, but a priority inversion of up to several cold
  PP ubatches.
- **Why earliest deadline, and not "cold reads lowest"** (the rule
  `9g22` rev 8 stated, retired by *§D10* I-3). The device reaches layer 2
  before layer 14, and a layer's cold experts run on the CPU inside that
  layer. A layer-2 slice is therefore needed before a `blk.14` row, and
  serving the `blk.14` row first can only delay the layer the device is
  stalled on.
- **Across contexts, round-robin.** Class 0 keeps one queue per context, and
  the workers take from them in turn. So no context's cold PP ubatch (up to
  ~25 k row reads, or a whole cold set of slices) can sit ahead of another
  context's 48 decode reads. A cold slice is class 0 of **the context that
  routed it**.
- **Class 1, lookahead** (§5.4), served only when class 0 is empty. A class-0
  request for a block that a class-1 read already has in flight joins it and
  promotes it to class 0.
- **Chunks.** A slice is read in chunks of at most 512 KiB, and a row block
  is 4 or 8 KiB. Nothing preempts a chunk a worker has already started, so a
  newly queued read with an earlier key waits at most for one worker to finish
  its current chunk.
- **In-flight dedup.** An in-flight table keyed by `(file, block offset)`
  records each pending read and its **waiters** (ticket, epoch, landing
  positions). A request for a block that is already pending joins it instead
  of reading it again: a repeated n-gram within one ubatch, the same row in
  ubatch `k` and in `k+1`'s lookahead, or two contexts reading the same hot
  row cold at the same moment. Two rows that share a 4 KiB block are one read.
  A join takes references to the read's ring slot (§5.3).
- **A join inherits the more urgent key** (*lead ruling §T12*, I-C). A join on a
  read that is still **queued** must not inherit that read's position: context
  B's decode ticket would otherwise wait for whatever sits ahead of the block in
  context A's backlog, which is the 1359 ms FIFO case again, and the same holds
  inside one context when epoch `e` joins a read that epoch `e+1` queued. So a
  join **from the same context** re-keys the queued read to the more urgent of
  the two keys, `(class, epoch, deadline)` compared lexicographically. Epochs of
  two contexts come from independent counters and do not compare (r7 m-7), so a
  join **from another context** leaves the owner's entry alone and adds a second
  entry for the read to the joiner's per-context queue, at the joiner's key: one
  read with two queue entries, dispatched by whichever the workers reach first,
  and the other entry is dropped when a worker finds it already dispatched. A
  join on a read already **dispatched** has nothing to reorder: it waits at most
  for that read, which witness 1's one-read term covers. yl3v's cold slices take
  the same rule, since two contexts routing the same cold expert join the same
  `(file, offset)` chunks routinely.
- **The interface** is `lazy_io_submit(fd, offset, length, destination handle,
  ticket, class, epoch, deadline)`: the same call for a 4 KiB row block and a
  512 KiB slice chunk, callable from any thread (§9.2).

**The ordering witnesses**, all T5 REDs on a file fixture with the mock or
the pread engine, CPU-only. The engine records a **dispatch log**: a
sequence number taken when a worker picks a read up. The order the rule sets
is the dispatch order, not the completion order, since 64 reads in flight
complete in whatever order the device returns them (r5 I-4 (b)); so the
separating REDs are pre-registered on the dispatch log. The engine carries a
**test-only FIFO ordering override** (`batch_probe3`'s `order=fifo`, A.2, is
its model), which serves reads in enqueue order. It is witness 1's positive
control, which must show the backlog delay under it, and `9g22`'s yl3v arms use
the same switch (its r9 m10 (b)); nothing outside a test sets it.
1. **Head-of-line, the backlog shape** (`batch_probe3 hol`, P = 12000). One
   context queues a 12000-read class-0 backlog, more than the 64 workers can
   hold, and a second context then queues a 48-read decode ticket. That
   ticket must complete within its no-backlog latency **plus the p99 of one
   read at QD64**, both measured by the fixture in the same run, because at
   worst it waits for one worker to finish a read already started (r5 m-5).
   On this host that is 11.2 + 33.2 ms (the same session's QD64 control p99,
   A.2) = 44.4 ms, against 21.7 ms measured under this rule and 1359 ms under
   FIFO. A single held read on a 64-worker pool cannot fail this check, so the
   backlog is the shape (*§D10* I-3).
2. **The separating RED** (*lead ruling §T8*, I-6). One context queues cold
   slice chunks for layers 2-13 of one epoch, more than 64 of them, so that
   every worker is busy and a backlog waits. Then it queues cold chunks for
   layer 20 and a `blk.14` engram ticket, in that order. Under this rule the
   pre-registered dispatch log shows:
   - every `blk.14` read dispatched after every queued layer-2-13 chunk;
   - and before every layer-20 chunk that was still queued when the ticket
     arrived.

   Under "cold reads lowest" the `blk.14` reads dispatch before the layer-2-13
   backlog, so the RED fails there. A `blk.1` ticket queued at the same point
   dispatches ahead of the backlog under both rules, so it separates nothing
   and is recorded only as a sanity arm.
3. **The epoch RED** (*lead ruling §T10*, I-4). One context queues two epochs,
   `e` and `e+1`, each with a `blk.1` and a `blk.14` ticket, behind its own
   epoch `e-1` backlog that keeps every worker busy. The pre-registered dispatch
   log shows every read of epoch `e`'s `blk.14` ticket dispatched before any
   read of epoch `e+1`'s `blk.1` ticket. Keyed by deadline alone, `e+1`'s
   `blk.1` reads dispatch first, so the RED fails there.
4. **The shared-block RED** (*lead ruling §T12*, I-C), two arms. None of the
   witnesses above shares a block, so none could see a join inherit a
   position.
   - **Across contexts.** Context A queues witness 1's 12000-read backlog, with
     block X placed near its end. Context B then queues a 48-read decode
     ticket that includes X. B's ticket must meet witness 1's bound, its
     no-backlog latency plus one QD64 read's p99, measured in the same run.
   - **Across epochs.** One context queues epoch `e+1`'s `blk.14` ticket,
     with block Y as its **last** read, behind an epoch `e-1` backlog that
     keeps every worker busy, and then epoch `e`'s `blk.14` ticket, which also
     includes Y. The dispatch log must show Y dispatched before any other read
     of epoch `e+1` **and** before epoch `e`'s last read: re-keyed to
     `(e, 14)`, Y keeps its earlier sequence number, so it leads epoch `e`'s
     `blk.14` reads.

   The pre-registered mutant joins without re-keying, so the read keeps its
   queued position: B's ticket then takes about A's backlog time (of the order
   of the 1359 ms FIFO figure), and Y dispatches in epoch `e+1`'s order: after
   every read of epoch `e`, and, being last in its ticket, after every other
   read of epoch `e+1`'s `blk.14` ticket. Both of the arm's conditions fail
   there.

### 5.3 Epochs, the gather worker, and every wait on the path

*Lead ruling §T8* (I-1, I-2, I-3, I-7) makes the structural form of D6-M(b)
the one mechanism for every lazy-table refill in v1: every model, dense ones
included, and every depth, lookahead included. It is the same mechanism as
`llama.cpp-9g22`'s cold-expert path (*lead ruling §D6a*). Nothing in it
depends on how far the submitting thread runs ahead of the device, so rev 2d's
invariant L is no longer a precondition anywhere. The ring's depth is the one
place where that distance enters, and it is a planned fact this design reads,
not one it sets (below).

Each ubatch a table is gathered for is an **epoch** `e` of that table in that
context. Its resources split by the thread that writes them:

- **Submitter side, the epoch ring**, depth `D` (§3.3). Entry `e mod D` holds
  the ubatch's **ids capture** (its row indices), one **host-signalled Level
  Zero event**, and one **marker word**.
- **Per epoch, the gather request**, never reused. It carries the epoch's
  **cancel flag** (and the `drain` flag the cancel step sets on an enqueued
  epoch) and, for STORAGE, its ticket. The flag lives here, off the ring entry,
  so the `prefetch` that takes an entry over can never observe a half-cancelled
  one, and HOST, which has no ticket, has the same flag (*lead ruling §T10*,
  I-2).
- **Worker side, the depth-2 slots**, indexed `e mod 2`: the landing slot
  (STORAGE) and the output slot (both residencies).

```
epoch entry    { uint32 epoch; ids capture; ze event; marker word;
                 request ref; bool copy_enqueued;
                 bool marker_enqueued; }             // submitter-private
gather request { lazy_gather_ctx ref; epoch; ticket (STORAGE only);
                 std::atomic<bool> cancelled, drain; }  // one per epoch
ticket         { request ref; std::atomic<int> outstanding; }
landing slot   { uint32 epoch; leaf lock; }
```

**The ring's depth** (*lead rulings §T9a*, *§T9b*). The caller's run-ahead is
bounded today by the RAW guard on graph inputs, and *lead ruling §T9b* names
it. Verified at `c69d5774d`:

- **The guard is the SYCL host buffer's sync before each input write.** Graph
  inputs are assigned to the CPU backend (`ggml-backend.cpp:1651-1654`), whose
  buffer type on this fork is the SYCL host buffer type llama asks the first
  device for (`src/llama-context.cpp:888-895`); that type's device is always
  SYCL backend device 0 (below). That buffer's `set_tensor` is overridden:
  `ggml_backend_sycl_host_buffer_set_tensor` (`ggml-sycl.cpp:42584`) calls
  `ggml_backend_sycl_host_buffer_sync` (`:42538`, `llama.cpp-30h4`) before its
  memcpy, and the sync waits the unified cache's queue and then the device's
  default queue (`:42543-42551`). `inp_tokens`, `inp_embd` and `inp_pos` are
  written through `ggml_backend_tensor_set` (`src/llama-graph.cpp:76, 84, 101,
  107, 116, 145, 147`), so each ubatch's input write drains everything
  submitted earlier on those queues. `KQ_mask` and `out_ids` are written
  through `->data` (`:455`, `:207`), so they are not guards. The device
  buffer's `set_tensor` GLOBAL drain (`ggml-sycl.cpp:36084`), which rev 2f
  named, is never reached by a graph input (r6 C-1).
- **What the sync covers: queue identity** (*§T9b*, *§T9c*). The sync waits
  two queue objects: the unified cache's queue for the host buffer's device
  first, then that device's default queue if it is a different object
  (`:42543-42551`). So the guard is a fact about which queue object lazy work
  is submitted on, not about a stream index. At `c69d5774d`, `stream(device,
  s)` returns `ggml_sycl_execution_queue_for_device(device)` for every `s` once
  the cache exists (`common.hpp:5808-5825`). That execution queue is the TP
  shared-context queue when TP is on (`common.hpp:5647-5655`,
  `common.cpp:194-197`), and otherwise the unified cache's own queue, a
  single-device-context queue the cache creates for itself
  (`unified-cache.cpp:14213-14231`, `:14472-14477`), not the device's default
  queue; it is the first queue the sync waits. (Only if creating it fails does
  the cache take the default queue, `:14477`, and the two are then one object,
  which the identity comparison below handles.) The sync's own comment
  (`ggml-sycl.cpp:42530-42534`) excludes streams 1 and up and the
  deferred-decode path. It predates `stream()` resolving to the execution
  queue, and at `c69d5774d` neither exclusion holds: every stream index is the
  same queue object, and the deferred-decode exit (`:107322-107323`) submits
  its barrier on `sycl_ctx->stream()` (`:107324`), that same queue, so the next
  ubatch's token write drains it. With TP off on one card the rule reduces to
  "on `stream(d, 0)`", and that is the only form in which this document states
  it. The predicate below accepts only the cache queue, the one the sync
  waits first (*§T9d*).
- **The in-flight depth `n`** is one planned quantity, owned by the guard, not
  by th32. Depth `n` means that at `prefetch(k)` every copy up to epoch `k-n`
  has completed. `prefetch(k)` runs after `set_inputs(k)` (§7.5), whose token
  write waited for everything ubatch `k-1` submitted, so `n` = 1. Nothing else
  is relied on: the W1 exit drain, prestage's input copy and
  `GGML_SYCL_DRAIN_INTERVAL`'s periodic drain are each specific to a path
  (*§T9b*), and item 8 lists them only as existing waits that can now span IO.
- **`D = in_flight_depth + 2`** (*§T9b*), so `D` = 3 today. The requirement is
  `D >= max(n, 2)` (r6 m-4): `prefetch(k)`'s free test needs copy `k-D`
  complete, which the guard gives for any `D >= n`, and stage 2's read of copy
  `e-2`'s marker is ordered by the strand for any `D >= 2`. So `n + 2` holds
  one entry more than required at `n` = 1. Rev 2f called it "margin 0", which
  was wrong. The spare entry is ruled and stated, and `D` still follows `n` by
  formula; the exhaustion assertion below remains the safety net.
- **The guard check is one planner function,
  `ggml_sycl_plan_input_raw_guard_present()`** (*§T9b*, *§T9c*). `9g22` calls
  the same function for a cold set with no lazy table (*§D12a*), and each
  caller raises its own family's tag (*§G1a*). At context creation, once the
  reserve graph is scheduled, it holds when both of these are true:
  1. **llama's declaration names a guard input** (*lead ruling §T9e*, r7 I-3).
     Whether llama writes an input through `ggml_backend_tensor_set`, which
     syncs, or through `->data`, which does not, is a property of llama's
     `set_input` code (`src/llama-graph.cpp:76` against `:455`), not of the
     graph: `KQ_mask`, `out_ids`, `inp_tokens` and `inp_pos` sit in the same
     SYCL host compute buffer (`src/llama-context.cpp:888-895`), so the backend
     cannot tell them apart. T6 therefore declares it: the context transaction
     carries the set of graph inputs llama writes through
     `ggml_backend_tensor_set`, today `inp_tokens`, `inp_embd` and `inp_pos`
     (the graph result's `t_inp_tokens`, `t_inp_embd` and `inp_pos`). The clause
     holds when that set names at least one INPUT leaf of the reserve graph
     whose buffer is a SYCL host buffer.
  2. **every lazy submission is on the queue the sync waits first, compared as
     an object** (*§T9c*, *§T9d*). Every lazy copy, every cancellation marker
     and every first reader of a gather's `dst` is submitted on
     `ggml_sycl_execution_queue_for_device(d)`, and that queue object must be
     the host buffer's device's unified-cache queue. Nothing else is
     accepted: once the cache exists the execution queue is never the default
     queue, so the sync's second wait, on the default queue, covers no work
     this design submits (*§T9d*). The predicate never compares stream
     indices. The context is refused **whenever the two queue objects
     differ**, and the cause is classified in order, the first that applies
     printed as `cause=<c>`:
     - `tp`, when `ggml_sycl_get_tp_queue(d)` is non-null, which is exactly
       when the execution queue is the shared-context TP queue, which the
       sync never waits; it is read from that function itself, so the test
       seam that overrides it reaches this case;
     - `device`, when `d` is not the host buffer's device
       (`sycl_host_buf_ctx::device`, the buffer type's device,
       `ggml-sycl.cpp:42633-42642`), whose queues the sync never touches. That
       device is **always SYCL backend device 0**, not `model.devices[0]`:
       `ggml_backend_sycl_device_get_host_buffer_type` ignores its device and
       returns the one generic host buffer type (`:107816-107819`), whose device
       is `ggml_backend_reg_dev_get(reg, 0)` (`:42724-42725`) (*§D12f* m-3,
       amending *§T9c* and *§T9d*'s gloss);
     - `pre-cache`, when **the execution queue is `nullptr`**
       (`common.hpp:5654`). Only `stream()` falls back then, to the device's
       default queue (`:5818-5822`), and it switches to the cache's queue once
       the cache exists, which would leave the work on two queues with no order
       between them. This case is expected to be unreachable, because the host
       buffer's own allocation creates the cache (`ggml-sycl.cpp:42637`,
       `unified-cache.cpp:14728`). The predicate still checks it by name, and no
       RED arm is required for it (*§T9d*);
     - otherwise `other`: a queue on the right device, with TP off and the
       cache present, that is still not the cache's queue.

     The comparison is a plan-time snapshot, while `stream()` re-resolves on
     every call, so a TP enable after planning is not seen here: that is
     `llama.cpp-0j5w`'s known mixed-TP straddle (§M35), named and not solved
     by this design (*§T9d*).

     Clause (2) is evaluated over a context's SYCL-executed tables only. A
     CPU-executed consumer's table (§5.3's CPU form) submits no copy, no marker
     and no first reader, and has no consumer device `d`, so the clause has
     nothing to compare for it, and no implementation evaluates it there with
     `d` undefined (r24 N-5). Clause (1) is unchanged.

  If (1) fails, the declaration is a defect, because llama writes its tokens or
  embeddings through `ggml_backend_tensor_set` on every model. th32 logs
  `[LAZY-PLAN-BUG] input guard missing` at WARN, naming the context and its
  declaration, and then, with `GGML_SYCL_STRICT_LEASES` unset, refuses the
  context as `lazy-input-unguarded` without aborting; with it set, aborts
  (*§D12f* m-6; the switch as in exhaustion, below). If (2) fails, the context
  is refused by name, `lazy-work-off-guarded-stream`, naming the table, the
  consumer's device, the queue and its `cause`. That is a
  refusal, not a plan bug (*§T9b*). The context records the declared guard
  inputs, `n` and `D`.
- **Why the pre-cache state is refused rather than admitted** (*§T9c* case 3,
  *§T9d*; the lead's confirmation for rev 2h, recorded here so it is not argued
  again). The sync does wait the default queue, so work left there would be
  drained. But lazy work is planned and allocated through the unified cache
  (P1), and work whose execution queue resolves before the cache exists is
  outside any plan: the sync's second wait covering it is an accident, not a
  property of the design. The state is expected to be unreachable, since the
  host buffer's allocation creates the cache and a plan exists only once the
  cache does, so a named refusal costs nothing and keeps one rule: the cache's
  queue, on the host buffer's device, with TP off.
- **The ubatch delimiter** (*lead rulings §D12g* I-1, *§D12m* 1-3). This
  document is the one statement of the shared rules below; `9g22` cites it
  (*§D12m*). `graph_compute` is not a ubatch boundary: the scheduler calls it
  once per split and once per `cb_eval` range (`ggml-backend.cpp:2459-2493`).
  The boundary is one llama-side call per ubatch, which T6 defines next to the
  *§T9e* declaration: `process_ubatch` calls
  `ggml_backend_sycl_ubatch_begin(backend, k, guard inputs)`, through a
  proc-address hook (§2.3), once per ubatch **on every SYCL backend the
  scheduler holds**, in scheduler order, after `res->set_inputs()` and before
  its `graph_compute()`, on the graph-reuse path and the rebuild path alike
  (*§D12m* 3). A backend with no lazy or cold plan treats the call as a no-op;
  the one with a lazy plan, which case 2 of the predicate puts on SYCL device 0,
  runs the lazy witness once for ubatch `k`, and `lazy_prefetch()` is its
  consumer: it runs at the same point, after every call has succeeded, for the
  `k` the calls carry (§7.5). The counters are per backend: `ubatch_begin_calls`
  equals the ubatch count on every SYCL backend, and `lazy_witness_checks`
  equals it on the backend with the lazy plan (§10), so a hook that never
  resolved cannot pass a GREEN vacuously; on a context with a lazy or cold plan
  an unresolved hook, `ggml_backend_sycl_ubatch_begin` or
  `ggml_backend_sycl_ubatch_end`, is refused at creation by name,
  `lazy-hook-unresolved`, naming the hook (r14 m-3). `9g22`'s cold witness keys
  on the same call, and the fixtures call the same entry point. No witness
  derives the boundary from `graph_compute`.
  - **The failure contract** (*§D12m* 2): a failed witness fails the ubatch
    **before** its `graph_compute`, whatever the strict switch: with
    `GGML_SYCL_STRICT_LEASES` unset it returns -3, and with it set it aborts
    at the same point, so the graph never runs on an unwitnessed guard.
  - **Every path that runs a ubatch's graph goes through it.** Encode, decode
    and embeddings all go through `process_ubatch`
    (`src/llama-context.cpp:2845`, `:3198`). The training path does not:
    `opt_epoch_iter` runs `res->set_inputs()` and then `ggml_opt_eval` on the
    context's own scheduler (`:4790-4830`), and `llama_set_param`'s default
    filter marks every F32 tensor but `token_embd` and `rope_freqs` a parameter
    (`:4673-4687`, `:5561`). So T6 **refuses `llama_opt_init` by name** on any
    context that carries a lazy or cold plan (*§D12m* 1). `llama_opt_init`
    returns `void` (`include/llama.h:1621`, `src/llama-context.cpp:5567`), so
    the refusal is a precondition at its entry: it logs
    `lazy-plan-training-unsupported` at ERROR, naming the context and its
    plan, and then aborts through `GGML_ABORT`, whatever the strict switch
    (*§D12n* 4, r11 m-1). No training step can run on such a context, and a
    caller learns it from the named line and the abort, never from a status.
    A graph that reaches a lazy gather with no delimiter for its ubatch
    anyway is caught by the carrier below.
  - **The carrier: how a gather knows its ubatch had a delimiter** (*lead
    rulings §D12n* 2, *§D12p*, *§D12q* I-2; one definition, which `9g22`'s
    cold hand-off cites). One ubatch may reach `graph_compute` several times
    (the `cb_eval` ranges and CPU islands above), so no flag that
    `ubatch_begin` sets and nothing clears can tell a later call of the same
    ubatch from the first call of a ubatch that had no delimiter. The
    carrier is therefore a post that is consumed:
    - **State, and where it lives.** Each SYCL backend context holds a
      **current-ubatch record**, the pair (context serial, `k`), where the
      context serial is the one the context transaction stored on the
      backend (the stamp bullet below). Each **`lazy_gather_ctx`**, the
      backend's per-context, per-table gather state (§9.1's ownership table),
      which survives graph rebuilds, holds the pair
      (posted `k`, `consumed`) for its table. It lives there, not on
      llama's per-build gather record of §7.5, because `llm_graph_result`
      is rebuilt with every new graph (`res->reset()`,
      `src/llama-context.cpp:2733`, `:4798`), and because the reader is
      the SYCL gather dispatch, which reaches `lazy_gather_ctx` and never
      llama's record. One pair per table is enough: the epoch ring already
      takes one epoch per table per ubatch (§5.3), so a graph gathers each
      lazy table at most once, and T6's post-build scan refuses a graph
      that gathers one table twice, naming it.
    - **The `none` sentinels.** `k` is an unsigned sequence number, and
      `none` is a reserved value that no ubatch takes. The backend record's
      `k` is `none` from the backend's creation, and again once teardown has
      cleared it. A `lazy_gather_ctx`'s posted `k` is `none` from its
      creation, with `consumed` clear. `none` never equals a real `k`, and
      two `none`s compare unequal for this check, so a gather with no post
      since its creation reports the line whatever the record holds.
    - **Setters.** `ggml_backend_sycl_ubatch_begin(backend, k, ...)` writes
      the record's `k` at its entry, before its witness runs.
      `lazy_prefetch()` then posts each table's epoch for that `k`, which
      sets its `lazy_gather_ctx`'s posted `k` to `k` and clears `consumed`.
    - **Reader, on every gather of every ubatch** (*§D12p* option (a)). The
      gather's dispatch in `graph_compute`, before it enqueues anything for that
      node, requires that its `lazy_gather_ctx`'s posted `k` equals the backend
      record's `k` and that `consumed` is clear; it then sets `consumed`. The
      check does not depend on what the gather reads: a gather whose rows are
      all cache hits, the lazy analogue of a cold layer with no pick, is checked
      and consumed like any other. A CPU gather makes the same check and consume
      in its op, on ggml-cpu's thread 0, before it reads anything (§5.3's CPU
      form; r23 I-2 (c)). T6's post-build scan also requires that every graph
      built for a ubatch gathers every lazy table **its graph builder
      declares**: the declared set is per builder, and the
      `LLM_GRAPH_TYPE_DECODER_MTP` builder (`src/llama-graph.h:40`), whose head
      is out of scope (§1), declares none, so its head-only graph is not refused
      for omitting the decoder's tables (r13 m-7). The equality holds in both
      directions: the scan also refuses a graph that gathers a table its builder
      does not declare, so an MTP head that used a decoder table would be
      refused by name at build time, not run out of scope (r14 m-5). A table's
      gather is its lazy `GET_ROWS` node for a SYCL-executed consumer, or, for a
      CPU-executed one, the CPU gather op, which the scan identifies by its user
      data, the table's `lazy_gather_ctx`, since the op has no edge to the lazy
      tensor; the twice check and both directions count the two forms alike
      (§5.3's CPU form; r23 I-2 (b)).
    - **The end point** (*§D12p* option (b), *§D12s*). Option (a) alone does not
      bound a post whose ubatch never reaches its gather: a recoverable
      `graph_compute` failure at a node before the gather (T7 (4)'s pre-reader
      arm is one), an earlier split that fails or aborts, after which the
      scheduler returns at once (`ggml/src/ggml-backend.cpp:2460-2463`,
      `:2482-2485`), and a `cb_eval` that returns false, which ends that split's
      node loop with success (`:2490-2491`). Each leaves the post with its `k`
      equal to the record's and `consumed` clear, and an undelimited graph that
      follows would pass that gather silently (T7 (5)'s arm (iv) is this case).
      So every delimited ubatch **ends**: `process_ubatch` calls
      `ggml_backend_sycl_ubatch_end(backend)`, through the same proc-address
      hook table, on every SYCL backend it called `ubatch_begin` on, and that
      call sets every one of the backend's `lazy_gather_ctx` posted `k`s to
      `none`. It is `noexcept`, since the guard's destructor may run during
      unwinding, where a throw calls `std::terminate`, and it takes no lock that
      teardown can hold: the pair is touched only by the thread that drives the
      context (its prefetch, its gathers' dispatch and its end point), so the
      end point is host stores per table and nothing else: one, or three for a
      CPU-form epoch that was posted and not consumed (§5.3's CPU form; r14
      m-3). The call is made by an RAII guard constructed at the delimiter,
      immediately before the `ubatch_begin` calls (after `set_inputs()`,
      `src/llama-context.cpp:2762`), so its destructor runs on every return
      after it: the delimiter's own failure, `graph_compute`'s failure (`:2771`)
      and its success (`:2776`). It also runs when a throw unwinds past it, such
      as a `std::bad_alloc` in the scheduler or a CPU backend (SYCL's
      `graph_compute` catches its own), and that throw is not terminal for every
      caller: `llama_decode` has no catch (`src/llama-context.cpp:5512-5521`),
      but the server catches `decode()`'s throw, aborts the slots and decodes on
      the same context at its next `update_slots`
      (`tools/server/server-context.cpp:2756-2780`). So the guard records
      `std::uncaught_exceptions()` at construction, and when its destructor
      finds the count risen it runs the backstop, `lazy_cancel_unenqueued()`,
      before the clear, inside the `noexcept` body with its own `try`; the
      backend step handles its own failures by the poison rule, and the `try`
      only keeps the destructor from throwing (r15 m-4). The returns before the
      guard (`:2711`, `:2747`, `:2753`) precede this ubatch's posts, and the
      previous ubatch's end has already cleared the rest. After the end point,
      any gather that runs before the next delimiter fails the k branch at once,
      whatever its ubatch reached. **The end point sits only on the delimited
      path** (*§D12s*). A detector for a missing event cannot be reset by an
      event that the undelimited graph also produces, so the end point is never
      in `graph_compute`, never in the gather, and never on a path that an
      undelimited graph runs: in production, `process_ubatch` is the only caller
      of `ubatch_begin` and `ubatch_end` (T7 (5)'s direct-drive arms call both
      themselves, r14 m-3), and the training path does not pass through it,
      since `llama_opt_epoch` (`src/llama-context.cpp:5571-5579`) reaches
      `opt_epoch_iter` (`:4869`, `:4880`), which calls `res->set_inputs()` and
      `ggml_opt_eval` directly (`src/llama-context.cpp:4790-4830`); nor does the
      K-shift graph (`llama-kv-cache.cpp:1002-1009`), which has no gather. Two
      placement mutants are pre-registered: an end point that re-arms each post
      for the current record (posted `k` := record `k`, `consumed` cleared) at
      every `graph_compute` entry, the form a per-call end point takes when it
      has no delimiter to key on, makes T6 (e)'s training graph pass its gather
      silently; and a clear to `none` at every `graph_compute` exit fires the
      line at the second SYCL `graph_compute` of a delimited ubatch whose later
      range holds a gather, which T7 (5)'s CPU-island GREEN catches by
      construction, since it has a gather on each side of its CPU node; the
      `cb_eval` GREEN catches it only when some gather lies outside its first
      SYCL range, which its fixture does not pin (r14 m-6).
    - **Clear points.** The next `ubatch_begin` overwrites the record; the
      lazy teardown sets it to `none` as its first action. A post is
      replaced by the next post for its table, and set to `none` by its
      ubatch's end point. So no post outlives its delimited ubatch, whether
      or not the ubatch reached its gathers, and a graph run without its
      `ubatch_begin` is caught at its first gather, whichever table that
      is.
    - **The line and its action.** A gather that finds its posted `k`
      different from the backend record's `k` (the **k branch**), or its
      post already consumed (the **consumed branch**), reports
      `[LAZY-PLAN-BUG] ubatch delimiter missing`, naming the context, the
      table, both numbers and the branch, at WARN; with
      `GGML_SYCL_STRICT_LEASES` unset it throws the backend's failure, so
      `graph_compute` returns `GGML_STATUS_FAILED` through its catch path
      with nothing enqueued for that node, and with it set it aborts,
      through `ggml_sycl_strict_enabled()`. The cold family's line is
      `[COLD-PLAN-BUG] ubatch delimiter missing`, with the same action.
    Each branch has its own positive control in T7 (5)'s direct-drive arm.
    After the end point, an undelimited graph takes the k branch; the
    consumed branch is reached by a second gather of one post inside a
    delimited ubatch, before its end point. T6 (e)'s training RED reaches
    the k branch deterministically: the training graph runs after the last
    decode's end point, so its gather finds its posted `k` at `none`.
  - **The K-shift graph** (`llama-kv-cache.cpp:1002-1009`) runs
    `graph_compute` between two delimiters. It has no lazy consumer, so it
    needs no witness; its allocation is scoped like every other (below).
- **The declaration is verified once per ubatch, per family** (*lead rulings
  §T9e*, *§D12f* m-2, *§D12h* I-3, *§D12k* 2, *§D12l*, *§D12n* 1; r7 I-3). At
  the delimiter, the lazy witness reads the buffer of the guard input the
  declaration names. **It first requires, on every check, that the buffer's
  stamp names this context's serial**, the one the transaction stored on the
  backend. A guard buffer allocated outside the scope carries no stamp, and the
  sync would treat it as a non-lazy buffer, with no watchdog and no fault
  attribution, and print nothing; so an unstamped guard fails closed: the
  witness logs `[LAZY-PLAN-BUG] input guard unstamped`, naming the context, the
  buffer's allocation serial and `k`, and the ubatch fails under the failure
  contract above (the cold family's line is `[COLD-PLAN-BUG] input guard
  unstamped`). It then compares the pair (allocation serial, count), that
  buffer's serial and its `host_buffer_syncs`, with the lazy family's snapshot,
  taken at the previous delimiter (or at context creation). With the same
  serial, it requires that the count has moved. With a new serial, because a
  re-reserve replaced the buffer (the watchdog bullet below), the baseline is 0,
  since a fresh buffer's count starts at 0, and the check requires count >= 1,
  the same "moved" test measured against 0; it never compares counts across two
  buffers, and it counts the event in `lazy_witness_rebaselines` (§10). The
  serial is a counter the allocation assigns, never the buffer's address (§10).
  The count is per buffer, so another context's input writes cannot satisfy it;
  the snapshot is per family, so `9g22`'s cold witness, which checks the same
  buffer at the same call with a snapshot of its own, never consumes the lazy
  family's movement, nor the reverse. If the check fails, the declaration is
  false: it logs `[LAZY-PLAN-BUG] input guard not observed`, naming the context,
  the buffer and `k`, `lazy_prefetch()` does not run, so no ring entry is taken,
  and the ubatch fails as the failure contract above states. So the plan-time
  check reads a declared fact and the runtime check witnesses it; nothing in the
  plan is revised mid-run.
- **Two cards** (r7 m-12). A two-card placement that puts a lazy consumer's
  layer on the second card is refused today by case 2. §4.5's two-card rows D
  and E are host-fit numbers and do not say which card runs a consumer, so T7's
  first plan-time run reports which two-card configurations the predicate
  admits, against a pre-registered expectation: phase 1 is the B70 alone, and
  a two-card DeepSeek plan is admitted only when both consumer layers, 1 and
  14, are on SYCL device 0, the host buffer's device. A single-card model
  placed on in-process device 1 under a two-device selector (`-mg 1`, split
  none) is refused by case 2, and the report pre-registers that too (r8
  m-3).
- **`llama.cpp-0j5w` carries the guard forward** (*§T9b*, §M35). When the
  execution queue becomes per context, the sync must wait on the executing
  context's queue, or the guard silently vanishes. 0j5w carries that
  obligation, with a RED in which this predicate refuses a context whose sync
  queue is not its executing queue. Any other change to the guard must
  publish the in-flight depth as a planned quantity for `D` to read (§11).
- **Initial state** (r5 m-3). Epochs start at `D`. Context init seeds entry
  `i` (`0 <= i < D`) with marker `i`, and both slots as released, so the
  first epochs' checks read epochs `0..D-1` as completed, never uninitialised
  memory. Epochs compare modulo 2^31; the marker's top bit is `CANCELLED`.

**The flow** (a SYCL-executed consumer's table; a CPU-executed consumer's is the
CPU form, below).

- **`prefetch(e)`, on the submitting thread, never waits.** It first checks
  the context's poison flag and its device's fault fact (below): a poisoned
  context fails the ubatch with its named cause and never evaluates the ring.
  The input guard's witness has already run once for the ubatch, before its
  first table (above); `prefetch` then reads entry
  `e mod D`'s marker word with an **acquire** load (`std::atomic_ref<uint32_t>`,
  `memory_order_acquire`; r6 m-8). The entry is free when that word holds
  epoch `e-D` (its copy completed) or `e-D | CANCELLED` (the strand retired
  it, below). Only device marker kernels write the word after context init:
  the copy's, or a cancellation marker, each through a `sycl::atomic_ref`
  store with `memory_order::release` at system scope, since the word is
  `HOST_PINNED`. The strand's last touch of an entry is a release fence
  followed by `zeEventHostSignal`, and the marker kernel runs only after that
  signal, so the acquire load orders the takeover after the strand's last
  access to the ids capture and the slots. (HT-5 reads the marker with this
  load.) `prefetch` then resets the entry's event (`zeEventHostReset`: no
  command can still wait on it and no thread can still signal it, because the
  one command that waited on it, copy `e-D` or its cancellation marker, has
  completed, and that command ran only after the strand's signal), copies the
  rows input into the ids capture (at most 48 KiB), creates the gather
  request, and posts it `{table, context, e}` to the IO engine. That is its
  whole cost. When the entry is not free, see **exhaustion** below.
- **The gather worker.** The IO engine runs each `(table, context)` as a
  **strand**: its epochs in order, in two stages, on the engine's **stage
  executor**, two threads of its own that are not the pread workers (r5
  m-9). A stage never blocks: it is a classification, hit copies, or one
  dequant (at most 12 MiB, ~2 ms at PP512; ~4 us per decode token). So a HOST
  dequant or a hit copy never queues behind a blocking `pread`, and T8 counts
  the time a ready stage waits for a thread as `stage_queue_ns`. The engine
  starts a stage when its condition holds, and no thread blocks on the
  condition:
  - **Stage 1, refill,** once epoch `e-2`'s dequant has released landing slot
    `e mod 2` (a worker-side fact). It takes the slot under its lock and sets
    the slot's epoch to `e`. Then it classifies each id (§6.3): a **hit** is
    copied from the row cache into its landing position at once, under the
    shard lock; a **miss** joins a pending read or queues a class-0 read keyed
    `(e, consumer_layer)` (§5.2.1), and registers a waiter. HOST has nothing
    to classify.
  - **Stage 2, dequant,** once the ticket's `outstanding` is 0 (STORAGE; at
    once for HOST) **and** copy `e-2`, the last reader of output slot
    `e mod 2`, has completed (its marker holds `e-2`, or `e-2 | CANCELLED`).
    It dequantizes into the output slot: STORAGE from the landing slot, HOST
    from the resident body through the ids capture. Then it releases the
    landing slot and signals the entry's event (`zeEventHostSignal`).
- **The copy** (§7.2 step 3). The submitter enqueues the output slot's H2D
  copy with `depends_on` the entry's event, through a `sycl::make_event`
  wrapper made for this copy (§7.2), then a marker kernel that writes `e`
  into the entry's marker word. It takes a reference on the context's
  `lazy_event_pool` for that wait, registers the copy with the marker
  watcher, and sets `copy_enqueued`.
- **The marker watcher.** One thread per process watches the marker words that
  parked stage-2 requests need and those of enqueued copies, with back-off, and
  resumes the stage whose marker is set; it reads each word with the acquire
  load `prefetch` uses. When it sees an enqueued copy's marker, it drops that
  copy's references, the event pool's included. It is the only thread that waits
  on device progress. That wait is on its own thread, which *lead ruling §T8*
  sanctions, and it is never the submitter's.

**No cycle.** Copy `e` waits on event `e`. Event `e` is signalled by stage 2
of `e`, which waits on IO and on copy `e-2`. Copy `e-2` sits earlier in the
same in-order queue and waits only on event `e-2`. Every wait points at IO or
at a strictly earlier epoch, so the chain ends at the first epoch, and the
submitting thread is in none of it.

**The CPU form: a CPU-executed consumer's table** (*lead ruling §M58b*; r23
I-2). When §3.2's query returns the CPU for a table's consumer layer, that
table's `lazy_gather_ctx` in each context takes none of the machinery above: no
epoch ring entry, no output slot, no marker word, no interop event, no
marker-watcher registration and no copy. The CPU gather op (§7.1) reads the
table on its own executor. Addendum 9's temporary refusal is not taken
(*§M58b*). The lifecycle:

- **`prefetch(e)`, on the submitting thread.** It checks the poison flag as
  above and posts the carrier (posted `k`, `consumed` clear). For HOST it does
  nothing else: the op reads the resident body through the rows input. For
  STORAGE it takes the table's one landing slot, which must read released
  (below), sets the slot's epoch to `e`, and classifies the ids **in place, from
  the rows input**, as stage 1 does: a hit is copied from the row cache into its
  landing position under the shard lock, and a miss joins a pending read or
  queues a class-0 read keyed `(e, consumer_layer)` (§5.2.1). Nothing is
  captured, because nothing overwrites the rows input before the op reads it:
  the next `set_inputs()` runs on this same thread after this ubatch's
  `graph_compute` has returned, and a CPU split completes inside that call. So
  this table has no stage 1 or stage 2 on the stage executor. The classification
  counts as stage 1's does, by §10's definitions: `rows_requested`, `rows_hit`,
  `rows_joined` and `rows_miss`, and `blocks_read` and `bytes_read` for the
  reads it queues (r24 M-6). For HOST it classifies nothing, so those counters
  stay 0 and `lazy_cpu_rows` is the table's engagement count (§8.1).
  `prefetch`'s cost is the classification and the hit copies, at most `n_ubatch
  x rows_per_token x row_size` (gemma4 E4B at PP512: 512 x 11424 B = 5.6 MiB,
  about 1 ms; one row at decode), and it never waits on IO or a device.
- **The op, on ggml-cpu's thread 0.** ggml-cpu calls a custom op on every thread
  of the graph (`fefb92980 ggml/src/ggml-cpu/ggml-cpu.c:3107-3153`), so the op
  acts on `ith == 0` alone, which is the thread that called `graph_compute`
  (`:3436-3459`), the thread that drives the context; the barrier after the node
  orders its readers after it. It first makes the carrier's check and consume,
  with the same rule and line as the SYCL gather (above). For STORAGE it then
  waits for its ticket's `outstanding` to reach 0 (`std::atomic::wait`, which
  the reaper's last decrement notifies), reads the landing slot, and stores the
  slot released; for HOST it reads the body through the rows input. It
  dequantizes through `to_float` into its output and counts `lazy_cpu_gathers`
  and `lazy_cpu_rows` (§10).
- **A failure inside the op.** A custom op cannot return a status through
  ggml-cpu, and a throw across its C frames is undefined, so the op records the
  failure (either carrier branch, or a failed read) on its `lazy_gather_ctx` and
  on the context's lazy-failure flag, logs its line, stores the slot released
  (below) and returns. The context's abort callback on the CPU backend
  (`src/llama-context.cpp:2584-2595`), which T6 wraps as "a lazy failure is
  recorded for this context, or the user's callback says abort", recording which
  of the two fired, then stops the CPU graph after the node
  (`ggml-cpu.c:3155-3160`), and `graph_compute` returns `GGML_STATUS_ABORTED`.
  **One failure fact has one status** (r24 I-2): `process_ubatch`, where T6
  already owns the delimiter and the end point, maps `GGML_STATUS_ABORTED` to
  `GGML_STATUS_FAILED` when the wrap recorded a lazy failure and the user's
  callback did not ask for the abort, before its end point runs, and the end
  point clears the flag. `llama_decode` then returns -3
  (`src/llama-context.cpp:3290-3292`), as the SYCL form does. Left as 2, the
  status would read as a user abort (`include/llama.h:1007`), and the server
  path that `llama-cli` runs would take it for a full KV cache: it sets an error
  only for 1, -1 and anything below -1, and leaves 2 as a TODO (`fefb92980
  tools/server/server-context.cpp:3688-3707`), so a 2 halves `n_batch`, warns
  `failed to find free space in the KV cache, retrying with smaller batch size`
  and retries (`:3726-3735`). A transient read failure followed by a successful
  retry would then leave correct tokens and no sign that the lazy path failed.
  With -3 the server logs `Compute error.`, fails the processing slots and stops
  (`:3702-3724`). With `GGML_SYCL_STRICT_LEASES` set the op aborts at the line,
  as the SYCL form does.
- **The landing slot is free by construction.** The op stores the slot released,
  and its request cancelled, on every return after its consume, success or
  failure, so a failed read never leaves it held (r24 M-7). For an epoch posted
  and not consumed (a failed graph, or a `cb_eval` range that ended before the
  op), host stores set the request's `cancelled` flag and store the slot
  released: the ubatch's end point makes them, and so do the backstop, the catch
  path's cancel step and teardown, whichever runs first, all before
  `process_ubatch` returns. The end point also releases a slot still held whose
  epoch is this ubatch's, consumed or not, so no path of this ubatch can hand
  the next `prefetch` a held slot. A read still in flight keeps its block-ring
  slot, and the reaper copies into the landing slot only while the slot's epoch
  matches and the request is neither cancelled nor draining (Cancellation,
  below), so a stale completion cannot reach the next epoch's rows.
  `prefetch(e+1)` therefore finds the slot released. If it does not, that is a
  plan bug with exhaustion's action: `[LAZY-PLAN-BUG] cpu landing slot held`,
  naming the table, the context, `e` and the slot's epoch, counted in
  `cpu_slot_held`, an abort under the strict switch and otherwise a fatal -3.
  The releases are counted in `lazy_cpu_epochs_released{site}`, the op's under
  `op` and the end point's under `end` (r25 M-1), never in
  `lazy_epochs_cancelled`, which counts marker enqueues.
- **One slot is enough, and `D` does not apply.** Nothing in the CPU form
  outlives its ubatch: the op completes inside `graph_compute`, and every
  release above runs before `process_ubatch` returns. T6 (l)'s sub-arm (l-D)
  decodes `D + 2` ubatches, where a CPU form that took a ring entry with nothing
  to retire it would fail at the fourth (r23 I-2 (a)).
- **Everything else is shared.** The input guard's witness, the delimiter, the
  carrier, the end point, the backstop and the poison flag apply unchanged, and
  lookahead (§5.4) lands in the row cache only.

**Exhaustion is a plan bug** (*lead rulings §T9a*, *§T9b*, *§T10* I-1). If entry
`e mod D` is not free at `prefetch(e)` and the context is not poisoned (below),
the submitter is further ahead of the device than the planned in-flight depth,
which the input guard excludes. `llama_decode`'s contract has no recoverable
return for it (`c69d5774d include/llama.h:976-990`: 1 means no KV slot, and a
`GGML_STATUS_FAILED` surfaces as -3, which is fatal), so rev 2e's recoverable
`lazy-epoch-ring-exhausted` is withdrawn. Nothing waits: `prefetch` logs
`[LAZY-PLAN-BUG] epoch ring exhausted` at ERROR, naming the table, the context,
`e`, the entry's marker, `in_flight_depth`, `D` and the guard input, and counts
`epoch_ring_exhausted`. Under `GGML_SYCL_STRICT_LEASES=1` it aborts. *Lead
rulings §G1* and *§G1b* fold the fork's ownership, lifetime and plan-defect
switches into that one variable, and th32 reads it only through
`ggml_sycl_strict_enabled()` (*§G1a*), with no `getenv` of its own.
`llama.cpp-uwlx` is its sole definer (*§G1b*, *§G1c*), so T7's commits that call
it land after uwlx's merge, with no fallback (§11). The tag stays
`[LAZY-PLAN-BUG]`: the rulings unify the switch, not the tags. Otherwise it is
that line plus a fatal -3: `process_ubatch` returns `GGML_STATUS_FAILED` through
the cancellation path below. T7's public-API fixture (§11, T7 (5)) is the
RED/GREEN that shows it unreachable on today's code, including through
llama-server's long-prompt path: the server synchronizes only after an
output-carrying batch (`tools/server/server-context.cpp:3565-3569`), but that
does not remove the per-ubatch input guard.

**Cancelled epochs are inside the guard too.** A cancelled epoch still puts
one command on the queue, its cancellation marker (below), so the next
ubatch's input sync waits for it exactly as it waits for a copy. The depth
fact therefore covers cancelled epochs with no separate bound. This holds by
construction, and no fixture can separate it: the catch path's own queue wait
already orders the marker before `llama_decode` returns, so T7 (4) has no
guard arm (r7 m-3). (Rev 2f's
first draft had the strand write the marker from the host, which left a
cancelled epoch outside the guard; the device-written marker closes that,
§13 DQ-F.)

**Cancellation** (*lead rulings §T8* I-2, *§T10* I-2, *§D12* I-1, *§D12b*,
*§D12c*). A failed ubatch never aborts the process, and nothing on the
failure path waits for IO.

- **The backend is the authority** (*§D12* I-1, the same step as `9g22`'s cold
  path). Each of `graph_compute`'s two catch paths runs the backend's **lazy
  cancel step** first, before `ggml_sycl_cpu_tg_flush_pending()` and before
  either of its queue waits: the boundary cleanup
  `ggml_backend_sycl_graph_boundary_exception_cleanup` (its flush at `c69d5774d
  ggml-sycl.cpp:107413`) and the `ggml_sycl_fallback_error` catch (its flush at
  `:107543`). 5efe's capability refusal at its six discarded-return sites
  reaches the second: it exits by throwing `ggml_sycl_fallback_error`, the
  recording catch rethrows it, and `graph_compute`'s `ggml_sycl_fallback_error`
  catch (`e2461d4fb ggml-sycl.cpp:107748-107779`) runs its full cleanup, this
  cancel step first, and returns `GGML_STATUS_FAILED` after its ERROR line
  (`:107778`) (*lead ruling §M69(b) third amendment*; 5efe rev 3l agrees). So
  the cancel step stays in the catch paths, and th32 adds no cancel site for
  that refusal. Rev 2f cancelled only from `process_ubatch`'s failure path,
  which runs after those waits. A copy enqueued at its first reader's boundary
  before a later node failed then sat parked on its event, and the catch path's
  `last_graph_event->wait_and_throw()` and `stream()->wait_and_throw()`
  (`:107418`, `:107424`; `:107548`, `:107554`) held the host until that epoch's
  reads finished: ~2.5-3.5 s on a cold DeepSeek PP ubatch, and without bound
  under the `hold` controls or a stalled read. The flush itself enqueues no lazy
  copy (it drains CPU-TG scatter and pipeline work, `:23910-23925`); a copy is
  enqueued only at its first reader (§7.2). The backend context holds its
  context's lazy gather state from the context transaction, so the step needs
  nothing from llama.
- **What the step does** (*lead rulings §D12e*, *§D12f* I-1). It acts on the
  mark's scope, the failed graph's whole epoch set (below; at teardown, in
  backend free and at a throwing `synchronize()`, every live epoch of the
  context). Within that set, **`copy_enqueued` partitions the action**, the
  same rule as `9g22`'s:
  - **copy not enqueued**: cancel plus marker, with the host stores first.
    These epochs are the **cancel set**; each needs a cancellation marker, or
    its entry never frees;
  - **copy enqueued, stage 2 unsignalled**: the drain. The step sets the
    request's `drain` flag and re-posts the stage, and enqueues nothing: the
    flag is what turns the re-post into the force-signal, because the re-posted
    stage reads it, skips its dequant if it has not begun it, and signals its
    event at once without waiting for its reads (below). The copy already
    `depends_on` that event, so it then runs, moving output-slot rows that
    nothing reads, since the graph failed, and its own copy marker writes `e`
    and frees the entry. The flag and the re-post are the whole of the drain; no
    cancellation marker is enqueued for this case, and it is counted in
    `lazy_epochs_drained`, not `lazy_epochs_cancelled` (*§D12h* I-2). th32 uses
    a `drain` flag of its own where `9g22` sets its cancel flag for this case;
    both are the same rule, the flag the re-posted stage reads;
  - **copy enqueued, stage 2 signalled**: no action. The epoch completes
    normally: its marker `e` was already enqueued with the copy, which writes
    it on completion.

  The stage signal never selects an epoch on its own. An epoch whose stage 2
  has signalled but whose copy was never enqueued is the normal case of a
  throw before the first reader, since a warm stage signals in microseconds
  (above), so by the time the catch path runs the stage has usually
  signalled; it is in the first case, selected and freed. A step that filtered
  on the signal would skip it, nothing would ever write its marker, stage 2 of
  `e+2` would park on "copy `e` completed" forever, and the ring would surface
  a recoverable failure as a hang or as `[LAZY-PLAN-BUG]` exhaustion (r8 I-1).
  An epoch whose marker is already enqueued (`marker_enqueued`) is skipped, so
  the step is idempotent. The step runs in a fixed order:
  1. **Host stores, which cannot throw.** It sets the request's `cancelled`
     flag of every cancel-set epoch and the `drain` flag of every draining
     one, which is all the host state th32's cancel needs, since output slot
     `e mod 2` is reused only after the device-written marker and the landing
     slot is released by the stage. It then re-posts each flagged epoch whose
     stage has not signalled to the stage executor: a `noexcept` post into
     the executor's preallocated queue, recorded per epoch, so a repeat is a
     no-op. The queue's capacity is bounded by the live epochs it can carry,
     `D` per strand summed over the strands on that executor, plus `9g22`'s
     `D_cold` terms if the executor is shared, so a post never allocates and
     never finds the queue full (*§D12m* 8). A signalled stage is not
     re-posted.
  2. **Device submissions, which can throw.** For each cancel-set epoch not
     yet marker-enqueued it enqueues **the marker kernel alone**, `depends_on`
     the entry's event, writing `e | CANCELLED`, and sets `marker_enqueued`; a
     `depends_on` on an event that has already signalled is satisfied at
     once. It does not write the marker from the host and does not signal the
     event. The step is complete when every cancel-set epoch is
     marker-enqueued; until then, the mark stays set.

  The re-post is needed because a stage parked on outstanding reads is otherwise
  resumed only by the last read's completion. **A cancelled or draining stage
  signals exactly once** (*§D12h* m-6): each request carries one resume claim,
  taken by compare-and-swap, and the stage is posted only by whichever of the
  step's re-post and the reaper's last decrement takes it. After the re-post has
  taken it, the last read of that request only releases its block-ring reference
  and resumes nothing, so a read that retires after `prefetch(e+D)` has taken
  the entry over cannot signal epoch `e+D`'s event early (r5 I-2's race). The
  executor runs cancelled and draining stages ahead of ready ones (*lead ruling
  §T11*). The stage sees its flag, skips the dequant and signals the event at
  once, without waiting for its reads; a stage already inside its dequant
  finishes it (at most ~2 ms) and then signals. So after the step the catch
  path's waits hold only for a stage pickup, never for IO, even when step 2
  throws. The counters mean one thing in both designs (*lead rulings §D12h* I-2,
  *§D12j* m-7), and each carries its family's prefix, because on DeepSeek the
  lazy engram tables and `9g22`'s cold experts share one context and unprefixed
  names would collide: th32's are `lazy_epochs_cancelled{site}` and
  `lazy_epochs_drained{site}`, `9g22`'s `cold_epochs_cancelled{site}` and
  `cold_epochs_drained{site}`. `lazy_epochs_cancelled{site}` counts
  cancel-plus-marker actions only, one per marker enqueue, and
  `lazy_epochs_drained{site}` counts drains, one per flag and re-post; the
  no-action case is not counted. The site is one of `catch`, `sync`, `entry`,
  `teardown` and `backstop`, and the unlabelled forms are the sums. A step that
  completes under the mark counts `lazy_mark_consumed{site}` for its site (§10).
  The partition is recorded as *lead ruling §D12g* m-7.
- **Reads still in flight** keep their block-ring slot, the memory the read
  writes, by their own reference until they retire (References, below). The
  landing slot is only the reaper's copy target, and the reaper copies into
  it only while the slot's epoch matches and the request is neither
  cancelled nor draining (the reaper's item 1), so the stage releases it, and
  T5's cancelled-epoch RED shows a stale completion cannot land in it. That
  is how §D12's "the slot stays held until the reads retire" is met here: the
  memory a read writes stays held, and the slot it would be copied into is
  guarded by the check. Completed reads still feed the row cache.
- **`synchronize()` and teardown: the failed-graph mark** (*§D12b*, *§D12d*, one
  rule shared with `9g22`). On a successful decode
  `ggml_backend_sycl_synchronize` (its flush at `:84304`) is the normal output
  wait, and the copies of the in-flight ubatches are legitimately parked there,
  so it cancels nothing and waits for them, as today. It runs the step only in
  three cases:
  - **the context carries the failed-graph mark** (*§D12d*). The mark is per
    context and **carries its scope**: the epochs the failed graph's ubatch
    prefetched for itself, one per lazy table, which the catch path reads from
    the backend context's lazy gather state. A step run under the mark cancels
    only those epochs, never an epoch prefetched later for another ubatch. Its
    only writer is a catch path, which sets it before its own cancel step.
    Neither this scope nor the backstop's (below) is keyed on the carrier's
    posted `k`, which the end point sets to `none`: both are the epochs the ring
    recorded at prefetch for the failed ubatch's `k`, read through the backend
    record's `k`, which the end point leaves alone (r14 m-2). It is cleared
    **only by a completed** cancel step, at any site, and nothing clears it
    blindly. It is one fact with one writer and one clear condition, and it is
    never derived at llama level from a return code. The catch path's own step
    is its primary consumer, and normally sets and clears it in the same path.
    When that step did not complete (each cleanup action is its own `try`), the
    mark is still set, and **both `synchronize()` and `graph_compute`'s entry**
    run the scoped step if they find it, then proceed: `synchronize()` before
    its wait, and `graph_compute` before the new graph's first node. At
    `graph_compute`'s entry the order is the poison check, then the scoped step,
    then the graph (*§D12e*). The entry site covers a caller that retries
    `llama_decode` after the fatal status without synchronizing; the scope is
    what keeps that retry's own epoch, prefetched at its delimiter, out of the
    cancel. The test hook that leaves a step incomplete (T7 (4)) throws at the
    marker enqueue, after the host stores and before the SYCL submit (*§D12e*,
    *§D12f* I-2). It throws a test-only exception type, not a `sycl::exception`,
    and no command was submitted, so it is not a poison source: poison needs a
    marker submission that the queue refuses, or a wait that reports a device
    error (below);
  - **teardown is running.** T6's lazy teardown is `~llama_context`'s first
    statement, before its `synchronize()` (`src/llama-context.cpp:1067`), and
    `ggml_backend_sycl_free` runs the step before its own flush (`:83948`,
    *§D12c*);
  - **`synchronize()` itself throws** (r7 m-8, the same rule as `9g22` rev
    12's). The boundary cleanup that holds the step is also `synchronize()`'s
    own catch handler (`:84352`, `:84354`), so there the step runs
    unconditionally. That is intended: a throwing `synchronize()` has failed its
    outputs, so its unsignalled epochs are cancelled. The cleanup already takes
    a stage name, and only the `"graph_compute"` calls and the
    `ggml_sycl_fallback_error` catch write the mark, so this path writes none; a
    mark still set is cleared by its step, as by any step that completes. A
    throw from its queue wait never reaches that handler: the wait is
    `CHECK_TRY_ERROR` inside `SYCL_CHECK` (`:84307-84308`, `:84334`), whose
    `ggml_sycl_error` aborts (`common.hpp:709-717`), after the classifier's
    class-2 call has seen the exception and, for a device-lost text, printed its
    `[SYCL-FAULT]` line (below). The handler sees every throw outside the
    wrapped wait, those before it (`ggml_sycl_cpu_tg_flush_pending()`,
    `stream()`, the strict validate at `:84301-84304`) and those after it
    (*§D12q* I-1). T7 leaves `synchronize()`'s existing strict-mode registry
    checks (`:84300`, `:84348`) alone; retiring them belongs to `llama.cpp-1obo`
    (*§G1b*).
- **The backstop.** `process_ubatch`'s failure path still calls
  `lazy_cancel_unenqueued()` (T6), for a failure between `prefetch` and the
  copy's enqueue that did not come through a catch path. It calls the backend's
  step, with the same rule over the failed ubatch's epochs, so it is idempotent:
  an epoch whose marker is enqueued, or that is draining, is skipped (r8 I-1).
  It runs in `process_ubatch`'s failure branch before that branch returns, so
  before the end point's guard is destroyed, and, on a throw that unwinds past
  the guard, from the guard's destructor before its clear (§5.3, r15 m-4), and
  its scope is not keyed on the posted `k` (above); so for an earlier CPU split
  that fails (`ggml/src/ggml-backend.cpp:2460-2463`), where no SYCL catch runs
  and the backstop is the only canceller, it still cancels the failed ubatch's
  epochs (r14 m-2). Its cancellations are counted as
  `lazy_epochs_cancelled{backstop}`, a backstop step that completes under the
  failed-graph mark counts `lazy_mark_consumed{backstop}` (*§D12m* 8), and a
  test hook turns it off, so that the mark arms of T7 (4) see only the site
  under test, with `lazy_epochs_cancelled{backstop}` = 0 and
  `lazy_mark_consumed{backstop}` = 0 as that hook's engagement check (*§D12f*
  I-2).
- **Why the device writes the marker.** The device writes `e | CANCELLED`
  only after the strand's signal, which is the strand's last touch of the
  entry, so only then can `prefetch(e+D)` take the entry over. The signal
  cannot land late on a later occupant's event, which was r5 I-2's race: the
  takeover needs the marker, and the marker needs the signal. The flags live
  on the request, off the ring entry, so a takeover never sees a
  half-cancelled entry. The strand runs a table's epochs in order, so no
  later epoch touches a slot while the cancelled epoch's stage is still in
  it. This is `9g22`'s cold-path form, which *lead ruling §D12* ratifies as
  the one cancellation mechanism of both designs.
- **The cancelled epoch's window** (DQ-F, re-derived under *§T9b*; r6 m-3).
  A failure happens inside ubatch `k`, after `set_inputs(k)`'s input sync
  drained every copy of ubatch `k-1`. So every earlier epoch has signalled,
  and the strand's next stage is `k`'s own. Run first (*§T11*), that stage is
  one stage pickup away, and its marker is set one device command later. The
  guard is on every path this design admits (the predicate refuses the
  rest), so the window is the same on each of them, the deferred-decode exit
  included.
- **A queue that cannot run the marker poisons the context** (*§D12*'s m3 rider;
  *lead rulings §D12h* m-1, *§D12i* m-8, *§D12m* 8). After the drain that
  follows the step (the catch path's own waits, or `synchronize()`'s), the
  backend reads, with a plain load (*§D12e*), the marker word of **every scope
  epoch the step acted on**: each cancel-set epoch whose cancellation marker a
  step actually enqueued (`marker_enqueued`), which must hold `e | CANCELLED`,
  and each drained epoch, whose copy marker is an enqueued marker too and must
  hold `e`. A flagged entry with no marker enqueued is **cancel pending**, never
  a fault: the mark is still set, and a later consumer's step enqueues its
  marker. A word that holds neither `e` nor `e | CANCELLED` means the queue
  could not run the marker, a device-side failure, and **poisons** the context,
  so a queue that dies after a drain poisons from the marker check itself and
  leaves no drained entry unfree. A wait's exception alone never poisons: today
  every catch-path wait is `try { ... } catch (...) {}` (`:107416-107427`,
  `:107546-107557`), and T7 changes those catches to record the exception as the
  cause that a later poisoning reports (r7 m-9 (a)), so the failed graph's own
  async error, surfacing at the catch wait after every marker ran, leaves a
  recoverable context recoverable. Two more sources poison, the same as
  `9g22`'s, each recording the first failure's text, the table and the epoch:
  - a marker submission that the queue itself refuses with a
    `sycl::exception`, a device-side refusal;
  - the host buffer's sync fails. It catches `sycl::exception`, logs it at
    ERROR and lets the write proceed (`:42553-42555`), and on the
    deferred-exit path it is often the first wait to see a lost device (r7
    m-9 (b)).

  **The device-lost classifier, stated once for both designs** (*lead rulings
  §D12h* m-2, *§D12m* 6, *§D12n* 3, *§D12o*, *§D12q* I-1). It is **one
  helper**, `ggml_sycl_exception_classify(device, what, context serial,
  site)`, declared `noexcept`, called from an **enumerated list of sites**,
  never "every catch": `ggml-sycl.cpp` alone has 147 `sycl::exception` catch
  clauses at `c69d5774d`, and most of them sit where no lazy work is in
  flight. The helper tests `std::strstr(what,
  "UR_RESULT_ERROR_DEVICE_LOST")`, the fork's own precedent for
  `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` (`ggml-sycl.cpp:36766`, `:37195`)
  (amending *§D12g* m-1), on the text as it was thrown, before anything
  reduces it to `dpct::default_error`. What it does:
  - **Device loss is always recorded, and says so.** A device-lost text
    records the **per-device** fault fact for `device`, whatever the context
    attribution, so a site outside any ubatch, or a buffer with no stamp or
    no registry entry, still records it; only the context-scoped poison
    below needs a context (*§D12n* 3, r11 m-8). Each time it records the
    fact, it logs `[SYCL-FAULT] device lost: device=<d> site=<site>
    ctx=<serial>|none` at ERROR and bumps the always-compiled per-device
    counter `device_lost_faults` (§10), so every site has an observable even
    where the process aborts right after the helper returns (*§D12q* I-1).
    `prefetch` reads the fault fact with the poison flag.
  - **Context attribution.** From the backend's current-ubatch record, or,
    at the host-buffer sync, from the guard buffer's stamp, the two being
    equal by the witness's stamp check (*§D12n* 1, *§D12o*): at the host
    buffer's sync, the context is the one its buffer's stamp names in the
    registry, since the sync has no backend; at every other site, it is the
    one the ambient backend's current-ubatch record names (the carrier,
    above). The ambient backend is a `thread_local` that an RAII guard sets
    for the duration of `graph_compute`, `synchronize()` and
    `ubatch_begin`, since the `CHECK_TRY_ERROR` catch has no backend
    argument. The guard is declared **before** each function's `try`, as
    `stage_trace_exit_guard` is in `graph_compute` (`:107451-107453`), so it
    outlives the catch handlers in which classes 3 and 4 run
    (`:107573-107580`, `:84351-84355`) (r12 m-5). Outside those, and
    before a backend's first `ubatch_begin`, there is no attribution.
  - **The device.** At classes 1, 3 and 4, and at class 2 inside an ambient
    backend, it is that buffer's or backend's ggml device index. Otherwise
    it is the thread's current device, taken with a lookup that cannot
    throw, since the helper can be entered from `ggml_sycl_set_device`'s
    own `CHECK_TRY_ERROR(get_current_device_id())` (`common.hpp:918`):
    `get_current_device_id()` (`common.cpp:137`) returns a dpct index, which
    the helper maps to a ggml index through the inverse of
    `ggml_sycl_map_device_id` (`common.cpp:222-230`, set at
    `ggml-sycl.cpp:25140`). A lookup or mapping that fails records the fact
    for every device, which fails closed. The current device need not be
    the one that threw (the init-time `get_memory_info` of device `i`,
    `:25213`, runs whatever device is current on the thread), but those
    sites precede any lazy work (r12 m-3).
  - **Any other exception** at the host buffer's sync poisons only the
    context attributed there, when that context's registry entry has its
    plan flag set, and is logged at ERROR as today. So an async error from a
    recovered failed graph, surfacing at the next wait, cannot poison a
    healthy device for every context (*§D12f* m-8). At the other sites it is
    recorded as the cause that a later poisoning reports, and never poisons
    by itself (the poison rule above).

  The sites, by class, each site calling the helper once:
  1. **the host buffer's sync** (`site=host_sync`), its `sycl::exception`
     catch (`ggml-sycl.cpp:42552`); it never aborts;
  2. **the `CHECK_TRY_ERROR` catch** (`site=check_try_error`,
     `common.hpp:638-648`), one edit to the
     macro that covers every wrapped op (46 live uses in `ggml-sycl.cpp` at
     `c69d5774d`; a 47th, `:45276`, is commented out), taking the
     `std::exception` it catches; the macro is also edited by
     `llama.cpp-23mk`'s I-1, and the two edits are made as one change to it.
     Every use on a `graph_compute` or `synchronize()` path, `synchronize()`'s
     own wait at `:84307-84308` among them, sits inside `SYCL_CHECK`, whose
     `ggml_sycl_error` (`common.hpp:709-717`) is `[[noreturn]]` and calls
     `GGML_ABORT`; the only non-aborting uses are init-time (`:24686`,
     `:25213`, `:25292`, `:25793`). So at this class the fact's observable is
     the helper's line in a process that then aborts;
  3. **the graph boundary** (`site=boundary`), the cleanup function
     `ggml_backend_sycl_graph_boundary_exception_cleanup` (`:107410`),
     which `graph_compute`'s top-level `std::exception` catch
     (`:107573-107576`) and `synchronize()`'s (`:84351-84352`) both call;
  4. **the catch-path waits** (`site=catch_wait`), the queue and event waits
     inside that cleanup (`:107416-107427`) and inside `graph_compute`'s
     fallback catch (`:107546-107557`), whose `catch (...)` T7 turns into
     catches that record the exception (the poison rule above).

  The list is checked by one census, which T7 carries as a pure-Python source
  gate over every `.cpp` and `.hpp` under `ggml/src/ggml-sycl/`, recursively, so
  `dpct/` is included, with `ggml/src/ggml-sycl/tests/` excluded by name, since
  a unit test may call the helper directly (r12 m-2, r13 m-2). It strips
  comments, then attributes every remaining occurrence of
  `ggml_sycl_exception_classify(` by a brace scan to its enclosing function, or,
  for a line inside a `#define`, to that macro. It allows exactly two non-call
  occurrences, the declaration in `common.hpp` and the definition, and exactly
  these calls: one in `ggml_backend_sycl_host_buffer_sync`, one in the
  `CHECK_TRY_ERROR` macro, three in
  `ggml_backend_sycl_graph_boundary_exception_cleanup` (its own and its two
  waits) and two in `ggml_backend_sycl_graph_compute` (the fallback catch's two
  waits). A call anywhere else, or a missing one, fails it; the macro's
  expansions in `common.cpp`, `concat.cpp`, `conv3d.cpp`, `fattn-buffers.cpp`,
  `mmq_xmx.cpp` and `common.hpp:918`, `:927` are uses of the macro, not calls,
  and the scanner sees none of them. A plain `git grep -n
  'ggml_sycl_exception_classify(' -- ggml/src/ggml-sycl` lists the same lines
  for a reader, plus any comment that names the helper. `synchronize()`'s
  `[SYCL-SYNC-DIAG] DEVICE_LOST` label (`:84306-84334`) is printed for every
  error and is never the classifier. Each class has a positive control, scored
  on the `[SYCL-FAULT]` line and `device_lost_faults`, class 2's as a death test
  on rc 134 and the line alone (r13 m-3); the "always recorded" control sits at
  a non-aborting, unattributed site, the unstamped `buf_output`'s sync; and the
  sync also has the negative control, an exception whose text carries
  `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` (T7 (4)).

  Every later `prefetch` then fails its ubatch with
  `lazy-context-poisoned` at ERROR, naming them, before it evaluates the
  ring. `llama_decode` reports that as its fatal status. So a device fault
  never surfaces `D` epochs later as `[LAZY-PLAN-BUG] epoch ring exhausted`,
  which would blame a plan bug for it. T7 (4)'s poison arm checks it.

T5's RED (r5 I-3): epoch A's reads are held and cancelled, then epoch B takes
landing slot 0 with different rows. **B's completions are released first,
then A's stale ones.** Pre-registered: the check-and-copy build holds B's
rows in slot 0 exactly, and A's rows appear only in the cache. The positive
control is a pre-registered mutant without the reaper's epoch and cancel
check, in which A's late completions overwrite B's positions, so slot 0 shows
A's rows. Releasing A first could not fail: B's completions would overwrite
the same positions, and the mutant would pass.

**The reaper** is the worker whose read completed (or the `io_uring` CQ
reader). It removes the entry from the in-flight table, then handles each
waiter:
1. It takes the waiter's landing-slot lock and copies the row **only if** the
   slot's epoch still equals the waiter's epoch and its request is neither
   cancelled nor draining. The check and the copy happen under one lock, and the
   slot's epoch changes only under it, so a stale row can never land in a
   reassigned slot.
2. It inserts the row into the row cache either way, because file bytes are
   always valid cache content.
3. It decrements `outstanding`. The last decrement resumes stage 2 only if
   it takes the request's resume claim; a cancelled or draining request whose
   stage the cancel step re-posted has lost it, so its last read resumes
   nothing (§5.3, *§D12h* m-6).

**References, and teardown never waits on IO** (*lead rulings §T8* I-7,
*§T10* I-7, P2).
- Every in-flight read holds a reference on its block-ring slot. The ring is
  owned by the IO pool, not a context (§3.3), so a read that another
  context's ticket joined writes into memory that neither context owns.
  **The slot is the join's source, and it is held until every waiter's
  delivery completes** (*lead ruling §D12m* 10, the rule `9g22`'s landing
  halves take): the reaper copies out of the slot once per waiter, the
  owner's and every joiner's, and drops the read's reference only after the
  last of those copies, so the slot returns to the ring, and a later read can
  reuse it, only when no waiter still reads it. T5's join-source RED checks
  it.
- Every waiter holds its ticket and a handle to the landing slot it fills.
- Every gather request holds, by value, the handles of its ids capture,
  landing slot, output slot and marker word.
- **Every enqueued wait holds the event and its pool.** The `sycl::event` from
  `make_event` uses `ownership::keep`, so it does not keep the Level Zero
  event alive. So a copy's enqueue takes a reference on the context's
  `lazy_event_pool`, which owns the pool and its `D` events. The marker watcher
  drops that reference only after it has seen the copy's marker, which the
  device writes after the copy, and so after its wait on the event.
  `zeEventDestroy` and `zeEventPoolDestroy` run at the pool's last reference,
  never while a command can still wait on the event.
- The marker watcher holds an enqueued copy's output-slot and marker handles,
  and its event-pool reference, until it sees that copy's marker.
- A ticket also holds its `lazy_table` (I11).

**Teardown order.** `llama_context::~llama_context` begins with `synchronize()`
(`c69d5774d src/llama-context.cpp:1067`), which waits on the queue, copies
parked on IO events included. So T6 runs the lazy teardown **before** that
`synchronize()`, as the destructor's first statement, and it is the cancel step
above in its teardown form. It cancels every epoch not yet enqueued
(`cancelled`, with no marker: no `prefetch` follows, so nothing needs the entry
back, as in `9g22`'s cold path), and sets `drain` on every enqueued epoch whose
stage 2 has not signalled; each such stage is re-posted, skips its dequant,
waits for no IO and signals, so the enqueued copies run in queue order and the
device drains. The two flags differ because only an enqueued copy has a waiter
to release at teardown. The `synchronize()` then waits only for device work
already enqueued, and none of this context's work depends on IO any more.

That wait is not only this context's (r6 m-7). The execution queue is one per
device, shared by every context on it (`common.hpp:5647-5654`, `:5808-5822`;
*lead ruling §M35*), so context A's `synchronize()` also waits for context B's
copies on the same device, and those may be parked on B's IO. The wait is
bounded by B's IO, and unbounded only while B's `hold` control is engaged.
`llama.cpp-0j5w`'s per-context queue removes it; until then it is the one
teardown wait left. Teardown then drops the context's own references and
returns. The `lazy_gather_ctx` destructor, the sole release point, runs at the
last release, and the memory is freed then. T5's REDs:
- free context A with a read in flight that context B has joined. B's rows
  must arrive, there must be no use-after-free (ASAN, or the mock engine's
  poisoned slabs), and the release must be observed only after the read
  completes;
- **the event arm** (r5 I-7): free the context between an epoch's signal and
  its copy's execution. The event and its pool must be destroyed only after
  that copy's marker is observed (a destroy-order counter on the mock's
  event pool).

**Lock order** (M11). The pool's per-context queues, the in-flight table, the
cache shards, the landing-slot locks and the strand queues are all **leaf**
locks: none is ever held while another is taken, and none is held across an
IO wait or a callback. T5 adds them to canonical contract §12.5 as L5 leaves.

**Every wait on the path, named:**

1. **The submitting thread** (*lead ruling §M58b*; r23 I-2). For a SYCL-executed
   consumer: none. `prefetch` reads a word, copies the ids and pushes a request.
   The copy's submit carries an event edge, and HT-1 measures that its submit
   does not block (§7.2). For a CPU-executed consumer's STORAGE table,
   `prefetch` classifies and copies hits, and the op waits for its ticket on
   ggml-cpu's thread 0, which is this thread, the scheduler's caller (the CPU
   form above). That is the sanctioned IO wait of the CPU executor's own data
   dependency, as a page fault is on upstream's path: it is counted as
   `cpu_ticket_wait_ns`, bounded by the IO and never by a device, and unbounded
   only while a `hold` control is engaged. While thread 0 waits, ggml-cpu's
   other threads spin in `ggml_barrier` after the node (`fefb92980
   ggml/src/ggml-cpu/ggml-cpu.c:3162-3164`), so the wait also holds `n_threads -
   1` cores busy for its length, on a host whose B70 decode slow mode was
   attributed to host CPU contention (r24 N-4). `cpu_ticket_wait_ns` measures
   the length, and the cores are its price.
2. **Storage IO:** stage 2 waits for its ticket. No thread blocks; the reaper
   resumes the strand. It is counted as `T_ticket` (§8.1, §10), and it is on
   the critical path only when the IO takes longer than the window (§5.4).
3. **A device that runs behind:** stage 2 waits for copy `e-2`'s marker,
   parked on the marker watcher. This is the device part of rev 2d's
   `slot_wait_ns`, now `slot_wait_device_ns`; its IO part is
   `stall_ns_total`. A marker still unset after `N` = 300 s, which is above
   every per-ubatch time this design or `9g22` §4.5.3 predicts, gives one
   WARN per table per context and counts `slot_marker_stale`: the
   bounded-age witness of *lead ruling §T8*. It never aborts and never waits.
4. **Device side, an event edge, not a host wait:** the output copy waits on
   the epoch's event. If the IO is late, the *device* waits there. The
   submitting thread does not wait at the copy; on today's MoE models and on
   the record paths it can still wait later, at one of the existing waits of
   item 8. How long a device may stay parked on a host-signalled event is
   HT-6's question (§7.2).
5. **Removed from this path by *lead ruling §T1* D6(c):**
   - `llama.cpp-x1xd`, the scheduler's host sync before a zero-input split
     after a backend change (`ggml-backend.cpp:2326-2334`). There is no CPU
     split in the middle of the graph any more. (The sync still fires at the
     leading-CPU-to-SYCL boundary that every model has, where it synchronizes
     the CPU backend, which has nothing queued.)
   - `llama.cpp-jhxq`, the SYCL `set_tensor` GLOBAL drain that a CPU-to-SYCL
     split-input copy takes because `cpy_tensor_async` is NULL
     (`ggml-sycl.cpp:107627`, `:36076-36084`). The gathered activation is
     copied by the backend's own event-chained `memcpy`, not through
     `set_tensor`.
   - The replay fragmentation of C2: the gather adds no `graph_compute` call,
     so the context's exec-graph slot sees the same graph every token (§7.4).
6. **Nothing added at the graph's start, and nothing read in place.** The rows
   input lives in the CPU backend's compute buffer, which on this fork is the
   SYCL host buffer type (`76c7f6548 src/llama-context.cpp:888-895`), and the
   SYCL device reports that type as supported (`ggml-sycl.cpp:108809-108811`).
   The scheduler therefore creates no split input and no copy for it
   (`ggml-backend.cpp:2116`, the `!ggml_backend_sched_buffer_supported`
   condition; the per-copy input duplicates of `:2092-2112` need
   `n_copies > 1`, and pipeline parallelism is off whenever a SYCL placement
   plan is active, `llama-context.cpp:970-971`). No device reads it either:
   the gather node dispatches nothing, and `prefetch` copied the ids on the
   host at `set_input` (§7.5). The gather's **output** reaches the device
   only by the explicit copy of item 4, in both residencies (*lead ruling
   §T2*, DQ). Wait sites (2)-(4) of the `llama.cpp-jhxq` census are not
   reached by anything this design adds, and site (4), the device buffer's
   `set_tensor` drain, is not the ring's guard either: no graph input reaches
   it (r6 C-1). The guard is the host buffer's sync (above), so this design
   depends on that staying, or on a replacement that publishes its depth.
   `jhxq` stays open for the copies other paths make.
7. **No SYCL host-task thread is held** (*lead ruling §T8*). This path has no
   `host_task`, so rev 2d's HT-3 question about the size of the runtime's
   host-task pool no longer applies. Its analogue, a slow read that holds
   back another context's gather, is the IO engine's head-of-line rule
   (§5.2.1), and T5 tests it.
8. **Existing waits that now span IO** (r5 m-4, r6 C-1). This design adds
   none of them, but each can now include `T_ticket`, because the device work
   it waits for can sit behind an epoch's copy (`c69d5774d ggml-sycl.cpp`):
   - the input guard, the host buffer's sync (`:42538`, reached from
     `:42584`), the one that bounds the ring (above; `llama.cpp-30h4`, and
     `llama.cpp-0j5w` for its queue). It runs in `set_inputs`, outside
     `graph_compute`, and on the deferred-exit decode path it can wait for a
     copy parked on IO, so it runs inside the watchdog's non-graph scope
     (below; r7 I-4);
   - `graph_compute`'s W1 exit drain (`:107337-107346`, traced as
     `graph-compute-exit`), on every graph that does not defer its exit wait,
     so on every PP ubatch;
   - the periodic drain every `GGML_SYCL_DRAIN_INTERVAL` nodes on the direct
     path (`:94315-94321`; default 256, off at 0);
   - the MMID ids admission (`:72160-72162`);
   - the synchronous router ids copy (`:74069` -> `:57241`);
   - `graph_prestage_leaf_tensors`' INPUT staging, run before record,
     segmented and block execution (`:98768-98780`, `graph_input_stage`);
   - sampling's `llama_synchronize`, the output `synchronize()`, where the
     last MoE layer's copy can still be parked (the watchdog scope below
     covers it).

   The catch paths' waits are not on the list: the cancel step runs before
   them (above). None of the waits listed here is relied on as a bound
   (*§T9b*). `llama.cpp-jhxq` and `llama.cpp-54sf0` own the rest.

**The watchdog counts IO progress** (*lead ruling §T10*, I-6 (a)). The fork's
watchdog `_Exit(1)`s after `GGML_SYCL_OP_TIMEOUT_MS` (default 30 s) without
a heartbeat while a `graph_compute` is in flight (`c69d5774d
ggml-sycl/common.cpp:655-677`; the in-flight count is an RAII guard around
`graph_compute`, `ggml-sycl.cpp:105097-105111`). Under today's MoE lockstep a
submitter sits inside `graph_compute` while the device waits on IO, and
`9g22` §4.5.3 predicts 28-145 s per DeepSeek PP512 ubatch. So the rule is:
**a stall is no device progress AND no IO-engine progress for the timeout.**
The reaper calls `ggml_sycl_watchdog_heartbeat()` for every class-0 read that
completes bytes; a class-1 lookahead read, which no ubatch waits for, never
does. The queued class-0 reads of a context are finite, so a hung device is
still caught, one timeout after the IO in front of it has drained, and an
engine that completes nothing is caught as before. The lead relays the rule
to `9g22`, whose cold path has the same exposure at a larger size (§11).

- **The heartbeat is process-global** (r6 m-6). There is one timestamp for
  the process (`g_watchdog_last_tick_ns`, `common.cpp:686`), so progress
  anywhere defers the timeout everywhere: context B's reads can keep the
  watchdog quiet while another device is hung. That is not new in kind:
  B's own `graph_compute` exits and prestage loops already heartbeat the same
  timestamp today, and mask a hung device in exactly the same way. The rule
  adds no masking beyond that, because only class-0 reads heartbeat, and each
  belongs to a ubatch some context has submitted and is waiting on. A
  per-device watchdog is outside this design.
- **Waits outside `graph_compute` run in a non-graph scope** (*lead ruling
  §D12a*, m16; th32 owns it under *§T10* I-6 (a)). The watchdog heartbeats
  and skips its check whenever no `graph_compute` is in flight and no
  non-graph scope is open (`common.cpp:723-725`). The last MoE layer's copy
  can still be parked at the output `synchronize()`, which is outside
  `graph_compute`, so a dead strand there would hang `synchronize()` with no
  watchdog, and only `slot_marker_stale`'s 300 s WARN would fire. So every
  host wait that can reach a lazy copy's event outside `graph_compute` runs
  inside a non-graph scope:
  - **the host buffer's sync** (`ggml_backend_sycl_host_buffer_sync`,
    `ggml-sycl.cpp:42538`; r7 I-4), the input guard itself. On the
    deferred-exit decode path `set_inputs(k+1)` waits there for copy `k`,
    which can be parked on IO. The scope goes inside the sync, so it covers
    all five of the host buffer's callers of it: `clear` (`:42576`),
    `set_tensor` (`:42584`), `get_tensor`, whose output reads have the same
    exposure, `memset_tensor` and `cpy_tensor`. **It is engaged only for a
    context that holds a lazy or cold plan** (*§D12f* m-7, *§D12h* I-3). That
    fact is context state, not a property of one buffer, because llama
    replaces a context's host compute buffers mid-life: `decode` calls
    `sched_reserve()` (`src/llama-context.cpp:3111`), which rebuilds the
    scheduler and frees its buffers (`:1980`) whenever a public setter
    (`llama_set_causal_attn`, the embeddings setters, an adapter change) has
    set `sched_need_reserve`, and gallocr replaces a buffer that a graph
    outgrows (`ggml-alloc.c:916-940`). So every host compute buffer the
    context allocates is **stamped**, each rebuild included, and never
    stamped once (*§D12m* 4, 5):
    - **The scope.** T6 opens a context allocation scope, through a
      proc-address hook, **inside `graph_reserve()`**, which covers
      `sched_reserve()`'s reserves and its pipeline-parallel retry,
      `memory_update` (`src/llama-context.cpp:2212`) and the `llama-ext.h`
      `llama_graph_reserve` (`:5240`), and at **every
      `ggml_backend_sched_alloc_graph` site on the context's scheduler**
      (`:2750`, the K-shift graph's `llama-kv-cache.cpp:1002`), which covers
      gallocr's replacement of an outgrown buffer. The training path is left
      to its refusal (above). The scope is `thread_local` and RAII, saving
      and restoring the previous scope so it nests; an allocation outside any
      scope carries no stamp, so another thread's allocation on the SYCL host
      buffer type while this context is inside its scope, another context's
      compute buffer or a model load's host weights, is never stamped with
      this context (r10 m-1).
    - **The stamp** is the pair (context serial, allocation serial), set by
      the SYCL host buffer type's `alloc_buffer`: the scope's context serial,
      a process-wide counter assigned at the context transaction, never a
      pointer to the context, and a fresh allocation serial (§10). The
      transaction's hook also stores the context serial on each of the
      context's SYCL backends, which is how the witness knows its own serial
      and the carrier's record names it. The plan flag is not copied into
      the stamp. It is context state, set at the context transaction and
      immutable afterwards; the sync and the fault attribution read it from a
      **context-serial registry** at check time, one fact with one source
      (P4). A refused context never decodes.
    - **Which transaction run creates the entry** (r11 m-9). The
      transaction runs before the first scoped allocation on every path, but
      more than once: on the default path at `src/llama-context.cpp:810`,
      before `sched_reserve()` at `:1041`; on the auto-ubatch path at each
      candidate's `:1571`, before `:1606`; and at the conditional settle,
      `:1934` before `:1937`. The **first** run creates the registry entry,
      with the context serial and the plan flag. Every later run looks the
      entry up and recomputes the flag, which is a property of the model's
      declared tables and the context's `--lazy-mode`, not of the ubatch
      candidate; a later run that computes a different flag reports
      `[LAZY-PLAN-BUG] plan flag changed` and refuses the context, so no run
      ever changes the flag.
    - **The registry's concurrency** (r11 m-6). A sync on one thread can look
      an entry up while teardown on another erases it, so the registry is a
      hash map from context serial to entry under one mutex, held across the
      lookup and the copy of the flag out of the entry, and taken by the
      insert and the erase. The lookup runs only for a buffer whose stamp
      names a context serial, so an unstamped buffer, the model's weights
      included, pays nothing; a stamped buffer pays one uncontended lock and
      one hash lookup per `set_tensor`, `get_tensor`, `memset` or `cpy`,
      against the queue waits the same sync then makes.
    - **Teardown.** `~llama_context` destroys its backends before its
      scheduler (`src/llama-context.h:395`, `:390`), so the buffers outlive
      the context's backend state: teardown removes the registry entry first,
      and a sync on a buffer whose context serial has no entry finds no plan,
      opens no scope and attributes no context fault, since that context is
      being torn down and has no witness; a device-lost text there still
      records the device's fault fact (the classifier, above).
    - **Unstamped by design** (r11 m-7). The context's output buffer
      `buf_output` is allocated on the output device's host buffer type
      (`src/llama-context.cpp:3499`), which is the SYCL host buffer type, in
      `output_reserve`, outside any scope, and its clear (`:3504`) enters the
      host buffer's sync. It stays unstamped, deliberately: it is never a
      guard input, so no witness reads it; its first allocation precedes any
      decode, and a reallocation runs `synchronize()` first (`:3480`), so no
      lazy copy can be pending behind its sync; and the witness's stamp
      check (above) fails closed on the one buffer whose stamp matters. The
      sites above are the list of scoped allocations, and `9g22`'s wording
      follows it rather than "every llama allocation site".

    The sync opens the scope only when its buffer's context serial names a
    registry entry whose plan flag is set. Every other model's sync behaves
    exactly as today and never starts the watchdog, so a long
    non-lazy wait there cannot newly `_Exit`;
  - the output `synchronize()`;
  - teardown's `synchronize()` after the cancel step;
  - `ggml_backend_sycl_free`'s wait;
  - a catch path's wait once the graph count has dropped.

  The scope is the RAII form the s1 wait guard already uses
  (`ggml-sycl.cpp:33851-33857`). Its constructor calls
  `ggml_sycl_watchdog_start()` (`common.cpp:796`), then
  `ggml_sycl_watchdog_heartbeat()`, then `ggml_sycl_watchdog_non_graph_begin()`
  (`common.cpp:707-714`), and its destructor calls `_non_graph_end()`, which
  heartbeats. The `start()` call is needed: only `graph_compute`
  (`ggml-sycl.cpp:105107`) and the s1 guard (`:33853`) start the watchdog
  thread, so a process whose first lazy wait comes before any `graph_compute`
  would otherwise have no watchdog at all. `start()` returns at once when the
  thread is running. IO-engine liveness counts as progress inside the scope, as
  above, and the scope adds no wait of its own (P4). T7 (4)'s watchdog arm and
  its input-sync sub-arm check it.

<a id="th32-windows"></a>
### 5.4 The prefetch window

- **When the IO starts.** Every table's reads are issued by `lazy_prefetch()`
  right after `set_inputs()` (§7.5), before the graph is submitted.
- **When it must be done.** With the SYCL-owned gather, the deadline is the
  moment the device reaches the copy before the consumer (§7.2). The gather
  request is posted at prefetch too, with no device dependency (§5.3), so
  the dequant runs as soon as the rows have landed and overlaps the preceding
  blocks as well. The node's position in the graph affects neither.

  | table | residency here | issued | needed before | window |
  |---|---|---|---|---|
  | Qwen `per_layer_token_embd` | HOST | `lazy_prefetch()` posts the gather request; no IO | block 1 | graph launch + block 0, against a ~4 us gather |
  | DeepSeek `blk.1.engram_embd` | STORAGE | `lazy_prefetch()`, class 0, deadline 1 | block 1 (B70) | graph launch + block 0 |
  | DeepSeek `blk.14.engram_embd` | STORAGE | `lazy_prefetch()`, class 0, deadline 14, behind `blk.1` | block 14 (B70) | graph launch + blocks 0-13 |

  Both DeepSeek tables are issued at the same moment, and `blk.14`'s reads are
  dispatched to workers after `blk.1`'s. Measured together on the two Q2_K
  files through one 64-worker pool, `blk.1`'s 24 rows landed at **7.13 ms
  mean, 14.6 ms p99** and all 48 at 11.2 / 21.0 ms (A.2).
- **Decode.** 16 (Qwen) or 48 (DeepSeek) rows per token, issued at
  `set_input`. The critical window is one block of one token, sub-millisecond
  to a few ms. A token cannot be prefetched earlier, because it is not known
  until it is sampled (MTP speculation, §12).
- **PP.** Every prompt token is known when `llama_decode` is entered. Phase 1
  prefetches each ubatch at its own `set_input`. Phase 2 adds **one-ubatch
  lookahead**: after `set_input(k)`, compute ubatch `k+1`'s row indices and
  prefetch them as **class 1** reads into the **row cache** only. They touch no
  slot, no ring entry and no marker, and ubatch `k+1`'s own refill then finds
  them as hits, or joins the reads still in flight and promotes them to class 0
  (*lead ruling §T8*, I-3; §7.2). With lookahead, the IO of ubatch `k+1`
  overlaps the *entire* compute of ubatch `k`, and the exposed stall per ubatch
  becomes `max(0, IO - compute)`. In-flight dedup means a row missed by both `k`
  and `k+1` is read once. The indices of ubatch `k+1` need predecessor tokens,
  which are the last `n_gram-1` tokens of ubatch `k` in the same sequence (2 for
  Qwen, 3 for DeepSeek). The model's input class gets a `peek_rows(next_ubatch,
  tail)` hook. The generic path skips lookahead when a model does not implement
  it. For DeepSeek on this host the tables are always on STORAGE and a cold
  ubatch costs ~2.5-3.5 s of IO, so lookahead is what makes long prompts
  tolerable; D9 puts it in v1 (*lead ruling §T2*).

---

## 6. The hot-row cache (item 4)

### 6.1 Why it works: n-gram Zipf

The row of an n-gram head is `hash(n-gram) mod p_h + offset_h`, so a row is
hot exactly when its n-gram is frequent. Natural-language and code n-grams are
heavy-tailed, and the 8 heads of one n-gram share a key, so they hit or miss
together. The simulation (Appendix A.3) replays each model's **exact hash**
over real text tokenized with that model's own vocabulary.

- **The Qwen hash.** The loop is compiled **verbatim**: `sed` of
  `694ec2354 src/models/qwen4exp.cpp:1067-1125` into a C++ harness. An
  independent Python transcription that reads its constants from the GGUF
  produced **identical** rows (320000 of 320000 compared). Every row index is
  below 320001536.
- **The DeepSeek hash.** `NgramHashState.forward` (§2.1.2) transcribed twice,
  into C++ (`ds_sim.cpp`) and independently into Python with every constant
  read from the GGUF (`ds_hash.py`). They produced **identical** rows (960000
  of 960000, on two corpora). The largest row seen is 384015792, below
  `blk.14`'s 384016682.
- **The Qwen tokenizer.** A pure-Python byte-level BPE with the `qwen35`
  pre-tokenizer regex (`76c7f6548 src/llama-vocab.cpp:382-387`). It is
  validated **50/50** against `models/ggml-vocab-qwen35.gguf.{inp,out}`, then
  run with the tokens and merges of the Qwen3.8 GGUF (248320 tokens). Every
  corpus round-trips byte-exact.
- **The DeepSeek tokenizer.** The same BPE with llama.cpp's multi-regex split
  semantics. The mechanism is validated **46/46** against
  `models/ggml-vocab-deepseek-llm.gguf.{inp,out}` (no test vectors exist for
  `joyai-llm`), then run with the `joyai-llm` regexes
  (`76c7f6548 src/llama-vocab.cpp:318-324`) and the tokens and merges of the
  DeepSeek-V4.1 GGUF (129280 tokens). Every corpus round-trips byte-exact. A
  residual difference between Python's `regex` and llama.cpp's own splitter
  would move a few token boundaries; the hit-rate statistics do not depend on
  that level of detail.
- **The corpora.** wikitext-2 raw (test 297 k tokens; train 2.51 M tokens), the
  fork's C/C++ sources (`src/`, `common/`, `ggml-cpu`, server; 3.42 M tokens),
  the fork's Markdown docs (2.29 M tokens), and a "mixed" stream that
  interleaves 2048-token chunks of train, code and docs to model a server with
  varied requests. Token counts are Qwen's; DeepSeek's vocabulary gives
  288 k / 2.42 M / 3.29 M / 2.24 M / 7.95 M.

### 6.2 Measured hit rates

#### 6.2.1 Qwen PLE

Row-granular hit rate over the whole stream. The cache size is `size / 170 B`
slots, excluding the 7.8% index overhead (§6.3):

| corpus | 32 MiB | 128 MiB | 512 MiB | 2 GiB | 8 GiB | distinct rows (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| wiki test (LRU) | 36.5% | 41.9% | 45.3% | 45.3% | 45.3% | 0.41 |
| wiki train (LRU) | 34.5% | 41.1% | 49.7% | 57.4% | 57.8% | 2.68 |
| wiki train (CLOCK) | 35.1% | 41.8% | 50.4% | 57.5% | 57.8% | |
| wiki train, 4 KiB-page LRU, same bytes | 19.5% | 27.6% | 33.7% | 41.4% | 55.6% | |
| code (LRU) | 86.8% | 89.1% | 90.5% | 90.7% | 90.7% | 0.80 |
| docs (LRU) | 56.3% | 63.0% | 68.2% | 71.4% | 71.4% | 1.66 |
| docs, 4 KiB-page LRU, same bytes | 33.6% | 45.1% | 54.6% | 62.8% | 72.1% | |
| mixed (LRU) | 58.8% | 64.4% | 69.6% | 74.9% | 76.9% | 4.81 |

By n-gram order (wiki train, 2 GiB): bigram heads 72.9%, trigram heads 42.0%.

How the results read:

- **CLOCK matches LRU to within ±0.7 points everywhere.** CLOCK is the policy,
  because it needs one reference bit per slot and no list relinking under a
  lock.
- **The curves saturate once the cache holds the corpus's distinct rows.** At
  that point the remaining misses are compulsory (first sight). A long-running
  server keeps climbing toward the saturation value of its own traffic.
- **The page cache is the wrong granularity.** At small sizes the row cache
  hits 1.3-1.8x as often for the same bytes. The page LRU catches up only at 8
  GiB, and there only by incidental neighbours in the same page.
- **Random tokens do not hit.** Uniform random tokens (the `llama-bench`
  control) never hit beyond hash collisions: 86% of tokens still miss all 16
  rows even with 8 GiB.

The metrics the speed targets need (CLOCK, steady state with the first 25% of
the stream excluded):

| corpus | cache | miss rows / decode token | P(0 misses) | P(16 misses) | unique-miss rows / PP512 ubatch (mean / max) |
|---|---|---:|---:|---:|---|
| random tokens | any | 15.85-16.00 | 0% | 86-100% | 8116-8192 / 8192 |
| wiki test | 0 (cold) | 16.00 | 0% | 100% | 7055 / 7848 |
| wiki test | 512 MiB | 8.47 | 31.9% | 35.3% | 4335 / 5951 |
| wiki train | 512 MiB | 7.75 | 35.0% | 28.0% | 3968 / 5820 |
| wiki train | 2 GiB | 6.29 | 42.6% | 14.5% | 3219 / 5311 |
| code | 0 (cold) | 16.00 | 0% | 100% | 3922 / 7456 |
| code | 512 MiB | 1.33 | 88.2% | 4.3% | 681 / 5419 |
| docs | 512 MiB | 4.58 | 60.5% | 15.7% | 2343 / 5700 |
| docs | 2 GiB | 3.97 | 63.9% | 10.4% | 2032 / 4930 |
| mixed | 512 MiB | 4.45 | 63.0% | 16.3% | 2275 / 6146 |
| mixed | 2 GiB | 3.52 | 67.6% | 7.1% | 1800 / 5186 |

Cold, unique rows per PP512 ubatch are 7028 (wiki test), 7104 (wiki train),
3969 (code), 6231 (docs) and 5562 (mixed) of 8192, so in-ubatch dedup saves
13-14% on prose, 24% on docs and 52% on code.

#### 6.2.2 DeepSeek engram (the harder case)

`ds_sim`: CLOCK over both tables' rows (key = table, row), at the stated *total*
size in 110 B slots. The plan fixes both tables' caches at the same size (§4.3
step 4; a cold-set plan splits nothing), and one shared cache of the total size
is equivalent in expectation: every n-gram occurrence touches 24 rows in each
table, so the two tables see the same frequency distribution under different
hashes.

| corpus | 128 MiB | 512 MiB | 2 GiB | 8 GiB | by order at 2 GiB (2 / 3 / 4-gram) | distinct rows (GiB at 110 B) |
|---|---:|---:|---:|---:|---|---:|
| random tokens | 0.2% | 0.5% | 0.9% | 0.9% | | 1.46 (300 k tokens) |
| wiki test | 28.9% | 33.3% | 34.0% | 34.0% | 59.2 / 28.6 / 14.2 | 0.93 |
| wiki train | 27.6% | 34.0% | 42.0% | 46.3% | 70.3 / 36.8 / 18.9 | 6.41 |
| code | 84.4% | 86.6% | 87.7% | 87.7% | 95.0 / 87.6 / 80.5 | 1.99 |
| docs | 52.3% | 57.5% | 62.1% | 63.4% | 82.9 / 58.8 / 44.6 | 4.04 |
| mixed | 55.1% | 60.0% | 65.1% | 70.2% | 82.7 / 62.3 / 50.5 | 11.45 |

Steady state (the first 25% of the stream excluded):

| corpus | cache (total) | miss rows / decode token (of 48) | P(0 misses) | P(48 misses) | unique-miss rows / PP512 ubatch (mean / max) |
|---|---|---:|---:|---:|---|
| random tokens | 0 (cold) | 48.00 | 0% | 100% | 24576 / 24576 |
| random tokens | 128 MiB - 8 GiB | 47.44-47.92 | 0% | 58-93% | 24290-24537 / 24553 |
| wiki test | 0 (cold) | 48.00 | 0% | 100% | 22180 / 23920 |
| wiki test | 512 MiB | 31.42 | 13.5% | 30.0% | 16092 / 20438 |
| wiki test | 2 GiB | 30.93 | 14.0% | 27.0% | 15837 / 20438 |
| wiki train | 512 MiB | 31.43 | 13.0% | 28.6% | 16091 / 20293 |
| wiki train | 2 GiB | 26.93 | 17.3% | 7.9% | 13790 / 18564 |
| wiki train | 8 GiB | 24.20 | 20.0% | 2.4% | 12389 / 18564 |
| code | 0 (cold) | 48.00 | 0% | 100% | 12964 / 22511 |
| code | 512 MiB | 5.91 | 80.5% | 3.9% | 3026 / 17368 |
| code | 2 GiB | 5.22 | 81.8% | 1.8% | 2674 / 17006 |
| docs | 512 MiB | 18.89 | 44.0% | 14.7% | 9669 / 19637 |
| docs | 2 GiB | 16.33 | 46.7% | 4.2% | 8368 / 17938 |
| docs | 8 GiB | 15.52 | 47.8% | 2.6% | 7945 / 16614 |
| mixed | 0 (cold) | 48.00 | 0% | 100% | 17452 / 24464 |
| mixed | 512 MiB | 17.97 | 49.1% | 15.9% | 9200 / 20540 |
| mixed | 2 GiB | 15.46 | 51.9% | 4.5% | 7916 / 18649 |
| mixed | 8 GiB | 12.43 | 54.9% | 0.2% | 6370 / 16572 |

Cold, unique rows per PP512 ubatch are 22117 (wiki test), 22320 (wiki train),
13114 (code), 19584 (docs) and 17762 (mixed) of 24576.

Why DeepSeek is harder, beyond having 3x the rows:

- **4-grams are sparse.** The order-4 heads hit 14% on prose at 2 GiB,
  against 59% for bigrams. A third of every token's rows are 4-gram rows.
- **A zero-stall decode token needs all 48 rows to hit.** On prose that
  happens for 14% of tokens, against 32-43% for Qwen's 16.
- **A PP512 ubatch misses 3-4x as many unique rows** at the same cache size:
  prose ~16 k against ~4.3 k, code ~3.0 k against ~0.7 k.

Prose keeps gaining up to 8 GiB (wiki train: 34% at 512 MiB, 46% at 8 GiB),
and the distinct-row sets are 0.9-11.5 GiB. On this host Q2_K's phase 1 has a
cold-expert set, so its caches are **fixed** at the gate size, 2 GiB per table
with the index included (§4.3 step 4, *lead ruling §T4* I-4). That is ~4.5 GiB
in simulation units, between the 2 GiB and 8 GiB columns, at every reading:
the reading moves the cold set, not the caches (§4.5). The hit rates of those
two columns bracket the operating point. A larger cache would cost the same
number of GiB of warm experts, one for one.

**Which corpora a target may quote** (M4). The code and docs corpora are this
fork's own sources, and the decode and perplexity vehicles in §8 would run on
text of the same kind: n-grams that recur across files of one codebase are
exactly what the cache catches, so those two corpora flatter the cache and are
self-similar with any in-repo test text. §8's targets therefore quote **prose**
(wiki train) and **mixed** only, and a vehicle's text must come from outside
the repo (wikitext-2 test for perplexity; a fixed prose prompt for decode).
Code and docs stay in the tables as the upper bound.

<a id="th32-row-cache"></a>
### 6.3 Structure

- **Keying.** The key is the row index (int64 for generality), inside the
  table's own cache. There is no special n-gram grouping: the 8 heads of one
  n-gram are 8 separate keys that are touched together, which CLOCK already
  handles, and the generic path stays table-agnostic (§9).
- **Index, 13.3 B per slot.** Open addressing over an array of 4 B slot ids at
  load factor <= 0.75 (5.33 B per slot), plus one 8 B key word per slot that
  holds the row index with the CLOCK reference bit folded into its top bit.
  That is **7.8%** of the slab on 170 B rows, **12.1%** on 110 B rows and
  **15.9%** on 84 B rows; §3.3 and §4.5 charge those figures, and the
  simulation's `cache_slots` are taken after them.
- **Sharding.** `hash(row) mod n_shards` with one mutex per shard. It is a leaf
  lock, **L5** in canonical contract §12.5 (added to its inventory by T4), and
  is never held across an IO wait or a callback (§5.3).
- **Contents.** Raw row bytes in the table's own type (170 B Q8_0, 110 B Q3_K,
  84 B Q2_K), not F32. Dequant happens in the gather (§7). An F32 cache would
  be 3.8x (Q8_0) to 12x (Q2_K) larger for the same rows.
- **Insertion.** A block completion inserts the row it carried (§5.3). A row is
  evicted only by CLOCK or by a yield (§6.4).
- **No leases** (*lead ruling §T1*, I8). No pointer into the cache outlives a
  shard lock: a hit is copied to its landing position under the lock, and the
  reaper copies into the landing slab *before* it inserts. The landing slab,
  not the cache, holds every row a pending gather needs. That is what makes the
  cache yieldable at any moment without waiting for anybody.
- **Warm start.** None in v1. The measured cold numbers are the contract. D1
  covers background prefill for the HOST residency.

### 6.4 Size, and yielding it back

- **The floor.** One double-buffered ubatch working set at `n_ubatch = 512`,
  plus its index: 2.66 MiB of slab for Qwen's table, 1.97 MiB for each DeepSeek
  Q2_K table. The floor is charged in step 4 (§4.3), before any leftover, and is
  never yielded. If even the floor does not fit, the load is refused as
  `lazy-table-cache-below-floor`.
- **Fixed, when the plan has a cold set** (*lead ruling §T4*, I-4). Then each
  cache is charged at its target in step 4, before the cold set is sized, and
  it does not trade bytes with the cold set at run time in either direction
  (§4.3). For DeepSeek the target is 2 GiB per table including the index,
  charged in the P3 budget before the cold-set search (*lead ruling §T5*).
  It is still built from yieldable
  256 MiB chunks, so a context tenant can take bytes back (below).
- **The leftover, yieldable, when it has none** (*lead ruling §T1*, I8). Step 5
  gives the rest of the host budget to the row caches, split as that step
  states, in
  **256 MiB chunks**. Each chunk is its own `HOST_CPU` allocation (role
  `LAZY_ROW_CACHE`) and is charged in the ledger as **yieldable**. When a later
  tenant needs host bytes (a new context's envelope, a KV growth, a later
  model's host weights), the ledger first takes them from yieldable chunks, and
  only refuses when yieldable bytes are exhausted too. Context tenants are
  therefore never starved by a cache that got there first.
- **How a chunk is yielded.** The same shape as the device-side
  `yield_optional_layouts_begin/finish` (`76c7f6548
  ggml/src/ggml-sycl/unified-cache.cpp:7798-7911`): under each shard lock in
  turn, remove the chunk's slots from the index and mark the chunk withdrawn so
  that no insertion picks it; then, with no lock held, drop the chunk's
  `mem_handle`, whose destructor is the release point (P2). Nothing waits,
  because nothing holds a lease (§6.3). The chunk's rows become misses; a
  reaper insertion that raced the withdrawal lands in another chunk or is
  dropped, which is always correct because the landing slab already has the
  row.
- **Growing back.** None in v1. A yielded chunk is re-offered only at the next
  plan publish, when the budget is sampled again (§4.2). A server whose
  contexts come and go therefore keeps its caches at the smallest size any
  tenant forced. That is proposed for v1 (D7); regrowing between publishes
  would need a trigger the ledger does not have.
- **Logging.** The planner logs `cache_slots` per table and the yieldable bytes
  (§10), and every yield logs the chunk count and the tenant that asked.

---

## 7. The executor (item 5)

<a id="th32-executors"></a>
### 7.1 One gather path, and the provider decides the position

Every consumer of a lazy tensor (Qwen's `build_inp_ple`, DeepSeek's engram block
input, and the rest of §9) calls one graph helper:

```
ggml_tensor * llm_graph_context::build_lazy_get_rows(ggml_tensor * lazy, ggml_tensor * rows, int consumer_layer);
```

- **Without SYCL hooks**, it returns `ggml_get_rows(lazy, rows)`: the upstream
  path, with the mmap and page faults, unchanged.
- **A SYCL-executed consumer, both residencies: `GGML_OP_GET_ROWS(lazy, rows)`,
  claimed by SYCL** (*lead ruling §T2*, DQ: one code path for both tiers). For a
  SYCL-executed consumer (a CPU-executed one is below; r23 N-3), the lazy tensor
  sits in the SYCL descriptor buffer of its consumer device (§3.2), a `WEIGHTS`
  buffer, so the scheduler assigns the node to that SYCL backend (rule `1.wgt`,
  `ggml-backend.cpp:1594-1612`, `:1684`); the CPU backend cannot claim it,
  because the buffer is not a host buffer. The SYCL backend recognizes a
  `GET_ROWS` whose `src[0]` is in a lazy buffer and runs the gather of §7.2. The
  residency changes only what the worker does before it dequantizes: for a HOST
  table it reads the resident body at once, and for a STORAGE table it first
  waits for its ticket. There is no CPU split in the middle of the graph, and
  the result has the shape `ggml_get_rows` produces, so each model's reshape and
  everything downstream are untouched.
- **Why HOST no longer sits in the leading CPU split** (rev 2a's form, *lead
  ruling §T1* C2, superseded by *§T5* DQ-C2 (a); §13). Rev 2a made the HOST
  gather a CPU custom op in the leading split and let the first SYCL op read
  its output from the `SYCL_Host` compute buffer in place. §T2's DQ forbids that
  read, and an explicit copy from the leading split cannot be made wait-free.
  The CPU op runs on the submitting thread and writes the same compute-buffer
  bytes for every ubatch, because `gallocr` gives a graph of the same topology
  the same addresses. In a multi-ubatch prefill nothing orders ubatch `k+1`'s
  CPU op after the device's copy of ubatch `k`'s output: the ubatch loop does
  not synchronize between ubatches unless pipeline parallelism is on
  (`c69d5774d src/llama-context.cpp:2724-2729`), and the SYCL split takes no
  split-input copy for a supported host buffer (§5.3 item 6). Guarding that
  source would take a wait on the submitting thread for the device, which is
  the host wait this design exists to avoid. The worker of §5.3 puts the
  same guard on its own thread (stage 2 waits for the previous copy's
  marker), and the ids live in the epoch ring, so the overwrite cannot reach
  them (*lead rulings §T6* I-1, *§T8*). So HOST takes
  the STORAGE path, and the CPU custom op survives only as arm C's executor
  (§8.4).
- **A CPU-executed consumer** (*lead rulings §F2a* I-9 addendum 9, *§M58a*,
  *§M58b*; r22 I-1, r23 I-1, I-2). When §3.2's query returns the CPU for the
  consumer layer, the helper emits the gather as a CPU custom op at the head of
  that layer, inside its CPU split. It is arm C's op in kind, not arm C's
  lifecycle, which keeps the step-3 copy and its marker (§8.4). The op has no
  edge to the lazy tensor, whose host-tier descriptor no node reads, so the
  stray-consumer check never sees it; the post-build scan counts it as its
  table's gather through its user data, the table's `lazy_gather_ctx` (§5.3). It
  performs the `get_rows` itself, through the same `to_float` trait, on the
  lifecycle of §5.3's CPU form: no epoch ring entry, no output slot, no marker,
  no event and no copy. Its F32 output lands in the CPU split's compute buffer,
  which the CPU-executed consumer reads in place: CPU reading host memory on its
  own executor, so arm C's step-3 copy and its previous-copy guard do not exist
  here. On gemma4 every layer consumes the gathered rows:
  `project_per_layer_inputs` adds them to the projection of `inp_batch` by
  `per_layer_model_proj` (`fefb92980 src/models/gemma4.cpp:59`, `:484-503`, the
  `ggml_add` at `:497`), and each layer reads its slice (`gemma4.cpp:378`).
  `per_layer_model_proj.weight` has no `blk.` prefix (`src/llama-arch.cpp:562`),
  so the plan charges it as a standalone entry, before any layer (`fefb92980
  ggml/src/ggml-sycl/unified-cache.cpp:28123-28139`), and the reduction's
  `blk.<il>.` walk never sees it (r24 N-3). Under T6 (l) it is likely on the
  device, so the CPU op's output crosses to the SYCL split at that `ggml_add` by
  the scheduler's ordinary split-input path, as `inpL` does, which this design
  does not govern (r23 N-4); no device reads the table's host bytes. The CPU
  form lands in T6 with the op.
- **Position.** For a SYCL-executed consumer, the helper expands the node at the
  **head of the consumer device's first split**: immediately before the first
  node of the first layer that runs on the table's consumer device. On one
  device that is right after `inp_embd`. With `-sm layer` and a consumer layer
  on the second card, a node placed at the very start of the graph would instead
  form a one-node split on that card ahead of the first card's split, and the
  first card's split would then pay the zero-input sync (`llama.cpp-x1xd`)
  against it. So the layer loop gets a `lazy_expand_for_layer(il)` call at the
  top of each iteration, and the helper defers the expand to the first layer on
  the consumer device. No new split is created in either case.
- **The window does not depend on the position.** The IO and the dequant
  are both started from prefetch (§5.4, §5.3), and the
  device waits only at the copy before the consumer, so the node's position
  changes nothing that runs; it only has to keep the split structure intact.

### 7.2 The SYCL-owned gather, both residencies (*lead rulings §T1* D6(c), *§T2* DQ, *§T8*)

1. **At prefetch time** (§5.3), on the submitting thread: the ids are copied
   into the epoch's ring entry, and a gather request is posted to the IO
   engine. No SYCL call is made, and nothing depends on the device.
   Whatever the ids need to become an output (the ticket's IO for STORAGE,
   then the dequant into the pinned `LAZY_OUTPUT` slot through the type's
   `to_float` trait) runs on the IO engine's strand, and ends with
   `zeEventHostSignal` on the epoch's event. There is no `host_task`: the
   mechanism is D6-M(b), which *lead ruling §T8* makes the v1 form for every
   model and every depth, shared with `9g22`'s cold path (*§D6a*). So the
   fork's two retreats from `host_task` (`ggml-sycl.cpp:32755-32757`;
   `:92354-92364`, `WEDGE-48330`) and `llama.cpp-c6ah`'s cross-queue
   submit block no longer bear on this path.
2. **The gather node itself dispatches nothing**, in direct and in record
   mode: its work was posted at prefetch, and its `dst` is written by the copy
   below. The main queue proceeds with block 0 at once.

Before the **first node (or fused group) that reads the gather's `dst`**, the
backend enqueues, on the **main in-order queue**:

3. `memcpy(dst_device, output_slot, bytes)` with `depends_on` the epoch's
   host-signalled event, then a marker kernel that writes the epoch into the
   ring entry's marker word (§3.3). **The wrapper form, pinned** (r5 m-2): the
   native event is the context's, created at context init (§3.3); each copy
   wraps it with a fresh `sycl::make_event` (`ownership::keep`) and drops the
   wrapper with its request, so no runtime-cached status can outlive a reset.
   HT-4 and HT-5 use exactly this form. This is the explicit, event-ordered H2D
   copy of *lead ruling §T2* DQ, the same for both residencies. Because the main
   queue is in-order, the copy runs after every earlier main-queue command,
   including the previous graph's readers of the same compute-buffer region:
   that is the write-after-read guard, with no barrier of its own. The consumer
   follows the copy in queue order: that is the read-after-write edge. The host
   thread waits for neither.

Rules around it:

- **The event is the copy's only edge.** Nothing in rev 2d's slot-reuse edge
  survives on the submitting thread (*lead ruling §T8* removes it, which also
  retires r4's m-4 and m-5). Slot reuse is now the worker's business (§5.3,
  stage 2), and the submitter needs only a free ring entry, which it checks
  with one plain load.
- **The event reset rule.** An event is reset only by the `prefetch` that takes
  over its ring entry, and only after it has read that entry's marker word
  showing the previous epoch's copy, or its cancellation marker, completed.
  Either command runs only after the strand's signal, and the strand signals a
  cancelled epoch only as its last touch of the entry (§5.3, *lead ruling §T10*
  I-2). So no command can wait on the event then and no thread can still signal
  it, and no copy can ever find its event still signalled from an earlier epoch.
  A reset by a worker *after* it sees the marker, with the event reused at depth
  2, would race the device: copy `e+2` can start the moment copy `e` completes,
  before any host thread has reset the event, and it would then read the output
  slot before stage 2 of `e+2` has written it. HT-5 probes this hazard, with a
  negative arm, and the lead relays the rule to `9g22` (§11).
- **What today's code does, which nothing here depends on** (r4 m-3). On
  today's code the submitter rarely runs far ahead of the device:
  - decode: the sampling step's `llama_synchronize` bounds it before the next
    `set_input`, for every model. On host-expert layers the synchronous
    router ids copy does too (`c69d5774d ggml-sycl.cpp:74069` ->
    `ggml_sycl_safe_memcpy_sync`, `:57241`); the MMVQ probe returns before it
    on device-resident layers (`:73860-73940`);
  - PP: the ids admission wait `:72160-72162`, whose `entry.wait()` is on the
    in-order memcpy event (`:57337-57368`).

  (`:22917` is a CPU future join and `:22867` waits on specific scatter
  events, so neither is a queue wait.) Dense graphs have none of these, and
  `llama.cpp-54sf0` may remove them. The design is the same either way, so
  *lead ruling §T8* withdraws the 54sf0 dependency edge for th32.
- **Lookahead needs no slot** (*lead ruling §T8*, I-3). Class-1 reads
  (§5.4) land in the row cache only; no waiter copies them into a slot, so
  they touch no ring entry and no marker. When ubatch `k+1` is then prefetched
  at its own `set_input`, its refill finds those rows as hits, or joins the
  class-1 reads still in flight and promotes them to class 0. There is one
  refill mechanism at every depth.
- **No start-order dependency.** A strand parked on a marker or a ticket
  holds no thread, so progress does not depend on the order in which any pool
  starts work (r3 I-1's question; *lead ruling §T6* I-1). The remaining
  head-of-line risk is the IO engine's, and §5.2.1 orders it.
- **Fusion.** The fusion matchers skip a node whose source is a lazy
  gather's `dst` until its copy has been enqueued, and never fuse the gather
  node itself.
- **The gather's `dst` lives in the device compute buffer** from the head of
  the split to its last consumer: for `blk.14` at PP512 that is 12 MiB held
  across blocks 0-13. `ggml_gallocr` measures it at reserve like any other
  tensor, so the compute buffer grows by it and the plan sees the growth (P4).
- **`rows` is never copied to the device.** It is an input, so it lives in the
  CPU backend's compute buffer, which on this fork is the SYCL host buffer
  type (`76c7f6548 src/llama-context.cpp:888-895`). The SYCL device reports
  that buffer type as supported (`ggml-sycl.cpp:108809-108811`), so the
  scheduler adds no split input for it (§5.3, item 6). The backend's graph
  paths are a second route, and T7 closes it (*lead ruling §T6*, m-12):
  `graph_prestage_leaf_tensors` (`c69d5774d ggml-sycl.cpp:98614`, run before
  record, segmented and block execution) sends every `INPUT` source of a node
  to `graph_input_stage` (`:~98768-98780`). The lazy `GET_ROWS` node's
  `src[1]` is one such source; its `src[0]` is skipped as `WEIGHTS`
  (`:~98718`). `llama.cpp-zhcn`'s `ggml_sycl_node_is_host_dispatched()` is
  defined as true exactly for the nodes its KV-host funnel intercepts (zhcn
  design l.720, l.1918). Enrolling the lazy gather in it as it stands would
  hand the gather to that funnel. So T7 needs the predicate split into
  "sources not staged", which the lazy gather joins, and "intercepted by the
  funnel", which it does not, or needs the funnel to dispatch by node kind.
  The split is agreed with zhcn before T7 starts (r4 m-12; relayed, §11).
  Nothing reads `rows` after `prefetch` has copied it into the ring entry
  (§7.5).
- **Counters.** `slot_wait_device_ns` and `stall_ns_total` (§5.3 items 2-3),
  `slot_marker_stale`, `epoch_ring_exhausted`, `lazy_epochs_cancelled`,
  `lazy_epochs_drained`, `lazy_context_poisoned` and `stage_queue_ns` (§10).

**The interop the mechanism rests on, and how T7 retires it first.** The fork
has one precedent, in a test (r5 m-1):
`c69d5774d ggml/src/ggml-sycl/tests/test-fattn-packed-k-lifecycle.cpp:262-342`
(`level_zero_host_gate`) creates a `HOST_VISIBLE` pool and an event whose
signal and wait scopes are `ZE_EVENT_SCOPE_FLAG_HOST`, wraps it with
`make_event(ownership::keep)`, and gates a kernel on it with `depends_on`
(`submit_half_payload`, `:345-350`); `submit_retry_after_accepted_signal`
(`:407-429`) requires a dependent submit to be accepted before the gate is
released. It never resets or reuses its event and never holds the device
long, so it does not settle this design's questions. T7's first deliverable
is six lead-run probes built on that gate class, with its pool flags and
event scopes. Production uses the same scopes: the host signals after it has
written host memory, and the device only DMA-reads that memory. Each probe is
a small standalone SYCL program with no model, run on `level_zero:0` and then
`level_zero:1` with the selector pinned, and `Shmem` sampled before and ~5 s
after. `E` below is such an event. Each copy wraps it with a fresh
`make_event`, as production does, and a probe that needs its status reads
`zeEventQueryStatus` on the native handle, never through the SYCL wrapper:

| probe | construction | pass |
|---|---|---|
| HT-1 | main queue: kernel K1 (~1 ms), copy `depends_on(E)`, a kernel K1b also `depends_on(E)`, then kernels K2..K50; a host thread signals `E` after 200 ms. **Must-fire control:** `c6ah`'s exact shape, a `host_task` whose `depends_on` names another queue's in-flight ~200 ms spin kernel | every `submit` in the main arm, the copy's and K1b's included, returns in < 1 ms; K1 completes at ~1 ms, not ~200 ms; K2 starts after `E` is signalled. The control's submit takes ~200 ms, which proves the timer sees a blocking submit |
| HT-2 | HT-1 with a recorded command graph replayed on the main queue after the copy | the replay's submit returns in < 1 ms, and its kernels run after the copy |
| HT-3 | the production execution queue, one per device and shared by every context on it (`ggml_sycl_execution_queue_for_device`, `common.hpp:5808-5822`), on `level_zero:0` and `level_zero:1` in one process, each with a copy `depends_on` its own event; `E_A` (card 0) is signalled after 2 s, `E_B` (card 1) after 10 ms | B's copy completes at ~10 ms: one device's late event does not hold back another device's. The claim is two devices only (r5 m-14): on one device a late event holds every later command of every context, so §5.2.1's per-context round-robin is an IO-only property, and same-device multi-context is unsupported anyway (canonical contract §5) |
| HT-4 | HT-1 repeated 10000 times, reusing one event under the reset rule | no crash, no Level Zero error, no event-handle anomaly (the `:32755` failure), `Shmem` flat |
| HT-5 | reset before reuse: copy `c1` `depends_on(E)`, then a marker kernel (its store a system-scope release, §5.3); `E` is signalled and `c1` completes; the host reads the marker with an acquire load, resets `E`, and enqueues copy `c2` `depends_on(E)`; `E` is signalled again 200 ms later. **Negative arm:** skip the reset | `c2` starts ~200 ms after its enqueue in the main arm, and at once in the negative arm, so the probe can tell a stale signal from a fresh one |
| HT-6 | **the long hold** (*lead ruling §T10*, I-6 (b)): HT-1's main arm with `E` signalled after **60 s**, then a second run after **300 s**; lead-run, selector pinned. The probe runs no `graph_compute`, so the fork's watchdog is not armed | no `GT reset`, `guc_id`, CAT error or job-timeout line in `journalctl -k` over the run's window; the copy completes, and correctly, after the signal; `Shmem` flat. The fork's own watchdog comment says xe resets the GT engine after ~10 s of job inactivity (`ggml-sycl/common.cpp`, its watchdog block at `:655-677`), so the outcome is not assumed. **The report records the loaded xe module** (r6 m-9): its path, `version` and `srcversion` (`modinfo xe`, `/sys/module/xe`), the kernel release, and the engines' `job_timeout_ms` and `job_timeout_max` from sysfs. `ggml-sycl.cpp:94322-94325` says the fork assumes a patched `xe.ko` built with `CONFIG_DRM_XE_JOB_TIMEOUT_MAX=600000`, so a pass on a patched module does not transfer to a stock one |

- If HT-1..HT-5 pass, T7 builds the mechanism above.
- HT-6's result goes to the lead whatever it is (*§T10* I-6). If xe resets or
  times out the parked job, the device must not be parked on IO, and the
  copy may be enqueued only once its epoch's IO has completed. That is not
  enough on its own (*lead ruling §T12*, I-D; r6 I-D): the submitter enqueues
  the consumer, and everything after it, on the same in-order queue right
  after the gather's boundary, so a copy submitted later by the strand lands
  **behind** the consumer, which then reads an unwritten `dst`. Keeping the
  copy ahead of the consumer without parking the device leaves two forms:
  - **a submitter stop:** the submitter stops at the first reader until the
    IO completes, then enqueues the copy and continues. That is a host wait
    on IO, which this design forbids (*§T8*);
  - **continuation submission:** `graph_compute` submits the graph up to the
    gather's first reader and returns control of the rest to the IO side,
    and the IO completion (stage 2's signal point) submits the copy and then
    the remainder of the graph from that boundary on. That changes
    `graph_compute`'s return contract, the W1 exit drain and the
    process-global graph mutex, and it is a new executor.

  *Lead ruling §T12* rules it: **if HT-6 resets, the design is continuation
  submission**, shared with `9g22`'s cold path, with no host wait. T7 then
  builds it before the fixtures that depend on a parked copy. Its
  interaction with the W1 drain and the mutex goes to the lead with HT-6's
  result.
- If any of HT-1..HT-5 fails, there is no in-design fallback. `host_task`
  (rev 2d's D6-M(a)) is superseded by *lead ruling §T8*, and D6-M(c) is
  rejected. The failure goes to the lead as a new question before anything
  else is built (§13).

### 7.3 Data movement, and P1 at both ends (M7)

- **Both residencies** (*lead ruling §T2*, DQ). The copy in §7.2 step 3 is an
  **activation**: 10 KiB per token for Qwen and 24 KiB per token per DeepSeek
  table, 5 MiB and 12 MiB per PP512 ubatch, ~1 ms of PCIe at most, against
  ~1.4 s of cold IO for a DeepSeek table's rows. Its source is the
  `LAZY_OUTPUT` slot, a `HOST_PINNED` allocation from `unified_allocate_owner`
  (§3.3), which only the copy engine reads. Its destination is a planned
  device slot: the gather's `dst` in the device compute buffer, planned by
  `gallocr` and allocated from the unified cache. Both ends are unified-cache
  memory (P1) held by `mem_handle`s that outlive the queued copy (P2). It
  never goes through `set_tensor` (`llama.cpp-jhxq`) and never needs the
  scheduler's zero-input sync (`llama.cpp-x1xd`), because the backend enqueues
  it itself on its in-order queue behind an event (§5.3).
- **No GPU read of `SYCL_Host` memory on this path.** Neither the gather's
  output nor its rows input is read by a device kernel: the output goes
  through the copy, and the rows are read only on the host (§5.3 item 5). This
  is the same P3 rule as `llama.cpp-zhcn`'s eager-path fix, which found and
  removed a zero-copy read of the same shape. The general census of CPU-split
  outputs that SYCL still reads zero-copy, `inp_embd` first, is a separate
  ticket the lead files (§T2 DQ); this design does not solve it and does not
  add a member to it.
  **T7's RED** (*lead ruling §T6*, I-3). A lead-run graph fixture on
  `level_zero:0` with one HOST and one STORAGE lazy gather feeding a SYCL
  consumer. The zero-copy check keys on the **resolved pointer the kernel
  reads**, not on a buffer type: `gallocr` puts a lazy gather's `dst` in the
  device compute buffer by construction, so a buffer-type test cannot fire,
  and the one zero-copy that could creep back, `dst->data` aliased to the
  pinned output slot to save the copy, leaves `dst->buffer` a device buffer.
  For every node it dispatches, the backend tests each resolved source address
  against the context's `LAZY_OUTPUT` slot ranges and counts a hit in
  `lazy_zero_copy_reads` (§10). When `llama.cpp-zhcn` lands, the counter
  becomes a view of zhcn's production check, a staged source that resolved to
  host memory, restricted to lazy sources: one fact, one source. It counts
  the step-3 copies in `lazy_output_copies`. The fixture requires
  `lazy_zero_copy_reads == 0` and `lazy_output_copies == tables x graphs`,
  and compares the consumer's input with a CPU `get_rows` of the same rows.
  - It fails first: T7 lands the recognition and the counters before the copy,
    so `lazy_output_copies` is 0 and the consumer reads an unwritten `dst`.
  - **The positive control is a pre-registered aliasing mutant**, not
    `inp_embd` (whose consumer reads device memory once zhcn stages it, which
    would leave a buffer-keyed control firing on nothing). The mutant aliases
    `dst->data` to the output slot and skips the copy; it must give
    `lazy_zero_copy_reads >= 1` per graph and `lazy_output_copies == 0`.
  - **Arm C (§8.4) keys on its own output** (r4 m-7). Its regression is its
    consumer reading the CPU op's output in the `SYCL_Host` compute buffer
    in place, which never touches a `LAZY_OUTPUT` slot, so the arm-A range
    test cannot see it. For arm C the counter tests each resolved source
    against the arm-C op's `dst` range; after zhcn lands, it is zhcn's
    "resolved to host memory" count for that consumer. Its pre-registered
    mutant skips arm C's step-3 copy and lets the consumer read the op's
    output in place. It must give `lazy_zero_copy_reads >= 1` per graph. A
    zero-copy arm C would bias the A/C comparison.
- **Placement consistency (P3).** The table never crosses PCIe; the device
  receives the *result* of a CPU op. No weight byte is staged to the device
  (no streaming), and no device kernel reads a table. The table is consumed
  in its materialized CPU layout, AOS rows of its GGUF type, which for a gather
  *is* the CPU-optimal layout: each row is one contiguous run of 84-170 B.
- **Why F32 and not the raw rows.** Shipping the gathered blocks and
  dequantizing them on the device would be cheaper on PCIe (84-110 B instead
  of 1 KiB per DeepSeek row), but it would split one op across two executors:
  the GPU would execute part of `get_rows` on per-dispatch copies of weight
  bytes, the weight-streaming shape P3 forbids even for a gathered subset. The
  PCIe it would save is ~1 ms per ubatch against seconds of IO. An F16 output
  is a later option if the consumer takes F16 activations; the CPU still
  executes the op.

<a id="th32-record-mode"></a>
### 7.4 Command-graph record mode: a stated segment boundary

*Lead ruling §T1* (I1) requires the record-mode behaviour to be resolved
explicitly. The two options were a recorded `host_task` node inside the
command graph, or a segment boundary around the gather. **This design takes the
segment boundary.**

- **Why not a recorded node.** A recorded `host_task` replays with the lambda
  captured at record time, so it would have to find the current ticket
  indirectly; DPC++ runs a graph containing a host task as separate partitions
  with the runtime synchronizing between them, which is a runtime-side wait on
  the device at that point; and it would put the primitive `:32755` warns
  about inside every replay, where a failure is hardest to attribute. Since
  *lead ruling §T8* the gather has no `host_task` at all, and the copy
  `depends_on` a fresh host-signalled event every ubatch, which a recorded
  segment could not carry. So the copy sits at the boundary between
  segments.
- **The boundary.** The existing segmented replay already interleaves graph
  replays with directly dispatched nodes (`ggml-sycl.cpp:97186-97270`, the
  `MUL_MAT_ID` boundaries collected by `moe_graph_collect_dispatch_indices`). T7
  widens that collector to a **boundary set** = the MoE dispatch indices plus,
  for each lazy gather of either residency, the first node that reads its `dst`,
  and, for `9g22`'s cold path, the first reader of each cold-bearing down output
  (its r9 m10 (a)). One predicate finds every first reader, the eager hooks' own
  `ggml_sycl_op_consumes_tensor` (`:23082-23093`). At that boundary the backend
  submits §7.2 step 3; the gather node itself is a no-op in both modes (its work
  was posted at prefetch), and the segments between boundaries replay as today.
- **Which lazy-bearing graphs replay, and which run direct and are counted**
  (*lead rulings §T10* I-5, *§T12* I-B). Direct dispatch is correct for a lazy
  graph by construction: the step-3 copy is submitted at its boundary on every
  call, with that ubatch's event, and nothing is recorded. What §T10 I-5
  forbids is a silent third form, a lazy graph that neither replays its
  segments nor is counted. So every lazy-bearing graph either takes segmented
  replay or runs direct and counts `lazy_graph_direct` under a named reason,
  and T7 (6) asserts which, by graph kind. Today a dense graph never reaches
  the segmented path (`use_segmented` needs `moe_graph_rerecord`, `:106941`,
  and only MoE paths set it), and the full-graph record path would record the
  step-3 copy with its record-time event, a stale edge on every replay. At
  `c69d5774d`:
  - **Decode, flash attention engaged in graphs**
    (`GGML_SYCL_FLASH_ATTN_GRAPH_ALLOW=1`, or `AUTO` once every observed
    decode FA kernel is verified replay-safe, `:106130-106141`). The topology
    site (`:106352`, `if (stats.moe_ops > 0)`, its setter at `:106361`) tests
    "boundaries > 0", where `moe_decode_segmented_graph_analyze` counts the
    **boundary set** (the MoE dispatch indices plus each lazy gather's first
    reader), not only `MUL_MAT_ID` dispatch indices (`:97205-97207`).
    `moe_graph_record_segments`' early return tests the same set
    (`:96912-96918`). In `moe_decode_segmented_graph_profitable` the early
    return keyed on `moe_ops == 0` (`:97240-97244`) becomes "no boundaries",
    keyed on the boundary set, so a lazy-bearing graph never takes it: it
    faces the cap, `graph_segments <= MAX_PROFITABLE_SEGMENT_GRAPHS` (8,
    `:97249`), and `segment_win`, which needs more graph segments than direct
    ones and more graph nodes than direct ones (`:97253`), like a MoE graph.
    (*§T12* I-B puts the cap before that return; once the return is keyed on
    the boundary set the order no longer matters for a lazy-bearing graph, r7
    m-5.) The analyzer leaves each boundary node out of every segment and sends
    a segment of fewer than `MIN_SEGMENT_NODES` = 3 nodes direct (`:97201`,
    `:97213`, `:97228`). The dense decode path runs this test at
    `:106340-106341`, before the `:106352` trigger. By shape:
    - **Dense gemma4 does not replay** (r7 I-2). gemma4's first
      `ggml_build_forward_expand` is `inp_per_layer`'s (`694ec2354
      src/models/gemma4.cpp:176-181`, `build_inp_per_layer` from `:440`), so
      node 0 is the lazy `GET_ROWS` and its first reader is node 1, or node 2
      if views are skipped. The head segment has 1-2 nodes and runs direct,
      and the tail is one graph segment, so `graph_segments` = 1 =
      `direct_segments`, `segment_win` fails, and the graph runs direct,
      counting `lazy_graph_direct{not_profitable}`. Rev 2g said this graph
      replays segmented, which was wrong. This design does not change the
      analyzer: a rule that a segment of no-op nodes is not a direct segment
      would move the policy for every MoE graph too, and that is a measured
      policy. T7 takes the pre-registration from a node dump of the real
      gemma4 decode graph, not from a segment count alone.
    - **The MoE graphs** (Qwen, DeepSeek) segment already; each lazy consumer
      adds one boundary to their `MUL_MAT_ID` boundaries (What it costs,
      below).
    - **A dense graph that does replay** has its lazy gather's first reader
      at least `MIN_SEGMENT_NODES` nodes into the graph and at least as many
      before its end. Head and tail are then both graph segments, two against
      none, and `segment_win` holds. T7 (6)'s synthetic dense arm has that
      shape.
  - **Decode, flash attention not engaged** (the default `AUTO` on a shape
    whose FA kernel is not verified replay-safe, and `=0`). Today a dense graph
    there turns graphs off (`use_sycl_graph = false`, `:106164`), and a MoE
    graph segments with every FA-bearing segment forced direct
    (`moe_graph_record_segments` from `:96900`; the analyzer likewise,
    `:97213-97233`). gemma4's PLE boundary sits before layer 0, so its decode
    graph would be a head of a few nodes and one tail holding every layer's
    `FLASH_ATTN_EXT`: both direct, zero graph segments. So T7 does **not**
    widen the `:106161` site. A dense lazy-bearing graph keeps
    `use_sycl_graph = false` and counts `lazy_graph_direct{fa_not_engaged}`
    per graph; a MoE graph keeps today's form, its FA segments direct.
  - **PP.** Both trigger sites are decode-only (`:106143`,
    `decode_has_flash_attn_ext`; `:106350`, `cached_is_decode`). A PP graph
    with a lazy gather that reaches the full-graph record path is refused
    there by name (`[SYCL-GRAPH] lazy gather needs segmented replay`) and runs
    direct, counting `lazy_graph_direct{pp}`. The full-graph path never
    records a copy's event edge.
  - **Unprofitable.** A lazy-bearing decode graph that fails the
    profitability test runs direct and counts
    `lazy_graph_direct{not_profitable}`.
  - **Graphs disabled** (`GGML_SYCL_DISABLE_GRAPH=1`): direct, as today, and
    not counted, since nothing tried to record them.

  If T7 finds a site where its recognition of the graph's lazy gathers is not
  available, the graph runs direct, named by that line and counted as
  `lazy_graph_direct{site}`, and the case goes to the lead. What direct costs
  is a replay's saving on those graphs; T9q and T9d measure it against the same
  arm with graphs engaged.
- **What it costs.** One extra segment per consumer: two per graph on
  DeepSeek, one on Qwen. On DeepSeek each consumer boundary falls inside a
  segment that already ends at the previous layer's `MUL_MAT_ID`. The
  replay-count cap
  (`MAX_PROFITABLE_SEGMENT_GRAPHS = 8`, `:97249`) applies to the total, and T7
  must show the DeepSeek decode graph stays under it or raise the cap with a
  measurement.
- **What must not change.** The gather adds no `graph_compute` call and no
  `#`-named split-copy tensor, so neither the consecutive-miss futility gate
  (`:99226-99240`) nor the fragmented-split gate (`:106688-106700`, which
  disables record/replay for the whole context on any `#` tensor) can trip
  because of it.
- **The witness** (every arm of §8): `GGML_SYCL_GRAPH_DIAG=1`, and per run
  - in a mode that replays (above), `seg_replay` rises with the token count,
    `seg_record` stays at its warm-up value, and on a decode run
    `full_replay` and `seg_replay` are not both zero;
  - in a mode that runs direct, `lazy_graph_direct` equals the graph count
    under its named reason;
  - zero `[SYCL-GRAPH] replay futility` WARN lines;
  - T8's `lazy_boundary_dispatches` counter equals `tables x graphs`;
  - **the consumer copy does not block its submitter** (*lead rulings §D7* I-B,
    *§T8* I-5, the same criterion for both models), **scored per table** (r4
    m-13). For every §7.2 step-3 copy, T8 records two things: host chrono around
    its submit, including its `depends_on`; and whether the epoch's event was
    still unsignalled at that moment (`copy_submit_io_pending`, read with
    `zeEventQueryStatus` on the native handle, never through the SYCL wrapper,
    r5 m-2), so the claim about pending copies is measured, not assumed. The
    criterion: submit < 1 ms on at least 99% of each table's copies, and on at
    least 99% of those with `copy_submit_io_pending` set. The old ordering
    witness ("block 15 submitted before block 13 completes") is withdrawn
    everywhere. The synchronous router ids copy lives in the general
    `ggml_sycl_mul_mat_id` (`ggml-sycl.cpp:74069` ->
    `ggml_sycl_safe_memcpy_sync`, `:57241`; *lead ruling §D8* moves the earlier
    `:57239` cite to this synchronous branch), and it keeps every MoE graph's
    submitter in lockstep with the device, so that ordering could not fail.

  **The engram controls**, defined here and nowhere else (*lead rulings §D9*
  2, *§T7* 3, *§T8* I-5; `9g22` cites them). T7 owns them. They are
  always compiled into the same release build as every arm (§8.1), so they
  are variables, not mutants. They are read once at init. Any non-default
  value emits a WARN through `ggml_check_sycl()`'s non-default-settings line
  (the `e7e2667bc` convention), and an unknown value is a WARN and is treated
  as `off`. T7 lists them in `docs/backend/sycl-env-vars.md`:

  | variable | values | effect |
  |---|---|---|
  | `GGML_SYCL_LAZY_CONTROL` | `off` (default), `hold`, `hold_blocking` | `hold`: the IO engine completes each read of one STORAGE table late (below); a HOST table has no reads, so for it the same timer holds its stage-2 signal instead, and the control is never a no-op (r5 m-8). `hold_blocking`: the same hold, and the submitter also waits on the epoch's event before it enqueues that table's step-3 copy. That is the blocking build, selectable in release, and it must show the hold in its submit time |
  | `GGML_SYCL_LAZY_CONTROL_TABLE` | a lazy tensor's name (default: the table with the highest `consumer_layer`, `blk.14.engram_embd.weight` on DeepSeek) | the one table `hold` applies to, so `blk.14` is reachable. A name that is not a lazy tensor of the model is a WARN, and the control is `off` |
  | `GGML_SYCL_LAZY_CONTROL_HOLD_MS` | a positive integer, default 50; `<= 0` or non-numeric is a WARN, and the control is `off` | the extra hold. Each read of the table completes (for HOST, each stage-2 signal is made) `T_window(t) + HOLD_MS` after its gather request is posted. `T_window(t)` is engine-observable (r5 m-8): the p99, over the 8 warm ubatches before the hold engages, of the time from the table's request post to its copy's submit, two instants the backend records. (§8.1's `T_window` comes from the lead's op timing, which the engine does not have) |

  The hold delays the reaper's completion (HOST: the stage's signal) on a timer.
  It does not sleep an IO worker, and there is no SYCL host-task thread to hold,
  so it depends on neither the pool size nor HT-3's outcome (*§T8* I-5). One
  table is held at a time, and the hold outlasts the table's own window. So
  under the lockstep of today's decode, the held table's copies are still
  pending when they are submitted. That makes the rev 2d uniform hold's blind
  spot (r4 I-5 (b): `blk.14`'s host task finished before its copy was submitted)
  impossible.
  - **Positive control, `hold`:** the held table's copies must show
    `copy_submit_io_pending` on at least 90% of them, submit < 1 ms, and
    `T_ticket` up by at least `HOLD_MS`.
  - **Positive control, `hold_blocking`:** the held table's submit
    times must rise by at least `HOLD_MS`, so the check can see a blocking
    submit.

  A run that fails any of these is scored as record mode disengaged, and its
  timing is not used.

<a id="th32-prefetch-hook"></a>
### 7.5 Where the prefetch hook sits

`llm_graph_result` records every `build_lazy_get_rows` call as
`{gather_ctx, rows_input_tensor, consumer_layer, residency}`. After
`llm_graph_result::set_inputs()` has run every input's `set_input`,
`llama_context` makes the ubatch delimiter's call on every SYCL backend the
scheduler holds (§5.3, *lead rulings §D12g* I-1, *§D12m* 3) and, once every
call has succeeded, calls `res->lazy_prefetch()` with the calls' ubatch
number, before `graph_compute()`. For each record, in ascending
`consumer_layer`, it reads the row indices from the rows input, whose buffer
is the CPU backend's host compute buffer (§7.2), and calls `prefetch`, which
copies them into the epoch ring and posts the gather request (§5.3); a
STORAGE table's misses become class-0 reads (§5.2.1). Lookahead (§5.4)
is the one per-model hook. Model code is untouched apart from
`build_lazy_get_rows` and the layer loop's `lazy_expand_for_layer`.

---

## 8. Pre-registered speed targets (item 6)

Every target below is derived from the numbers measured in §2.4, §5 and §6.2 on
this host, bcachefs as-is, under ambient load. The implementer measures against
them with the counters in §10; the lead runs every GPU part, one run at a time,
selector pinned, `Shmem`/`MemAvailable` sampled before and ~5 s after. Qwen's
targets are §8.2-8.3; DeepSeek's, the harder case, are §8.4.

### 8.1 Arms, positive controls and instruments

**The arm lever is the production budget override** (*lead ruling §T1*, I5):
`GGML_SYCL_HOST_RESERVE_MB`, which sets the ledger's budget (§4.2). It changes
the budget, never the residency, so the planner still decides; and it is read by
every binary, so every arm runs the **same** build. There is no test-only
override and no `GGML_SYCL_PRIVATE_TESTING` arm (that define is per test target
and never reaches `llama-bench` or `llama-perplexity`).

- **H**: HOST residency, no override. **Launch precondition:** P3 holds with
  the table resident, a predicted post-load `MemAvailable` of at least
  24 GiB, which after (s) needs a `MemAvailable_pre` of 180.9-184.3 GiB (plus
  up to 2 GiB of chunk rounding; §4.5). At 171.4 the planner makes Qwen
  STORAGE, so there is no H arm at that reading (*lead ruling §T6*, m-14). An
  H run records its `MemAvailable` before and after load. DeepSeek has no H
  arm on this host (§4.5).
- **S**: STORAGE residency, by setting the budget below the table: `RESERVE_MB =
  (step-3 requirement + floors + cache target) / 1 MiB`, the requirement taken
  from that run's own plan line in a load-only run first. For Qwen the step-3
  requirement is ~117-121 GiB on today's code and ~104-107 GiB after (s) (§4.5,
  "before table"), plus a 4 GiB cache target. For DeepSeek S is the natural
  choice and its caches are fixed at the gate size, 2 GiB per table **with the
  index included**, which is also 9g22's P1 cap (§4.3 step 4); no `RESERVE_MB`
  value gives it a different cache, it would only move `C` or refuse (*lead
  ruling §T6*, m-6). For **Qwen**, an S arm that wants a different cache sets
  `RESERVE_MB` = the step-3 requirement + the per-table cache size, index
  included. No separate cache-size variable exists or is needed.
  - **S-cold**: a fresh load, empty cache.
  - **S-warm**: steady state after at least 25% of a same-domain stream.
  - **S-hot**: the same text replayed with a cache at least as large as its
    distinct rows.
- **The DeepSeek executor arms** A and C (§8.4), selected at run time by a
  diagnostic variable `GGML_SYCL_LAZY_GATHER_ARM=A|C` (default A), documented
  in `docs/backend/sycl-env-vars.md` and removed once T9d has decided. It
  changes the graph, not the build.

**Positive controls** (*lead ruling §T1*, I6). An arm counts only if its
engagement is shown, because an absence of work looks exactly like success:

| arm | required in that run's own log | void if |
|---|---|---|
| H, SYCL-executed consumer | `[PLACEMENT] lazy <name> residency=HOST` **and** `rows_requested > 0` **and** `blocks_read == 0` **and** `lazy_output_copies > 0` | the line is missing, or `rows_requested == 0` (the design silently disengaged, e.g. `AUTO` fell to `OFF`) |
| S, SYCL-executed consumer | `residency=STORAGE`, `rows_miss > 0`, `blocks_read > 0` | any is zero |
| H, CPU-executed consumer (§5.3's CPU form) | `residency=HOST` **and** `device=CPU` **and** `lazy_cpu_rows > 0` **and** `blocks_read == 0` **and** `lazy_output_copies == 0` | the line is missing, or `lazy_cpu_rows == 0` |
| S, CPU-executed consumer (§5.3's CPU form) | `residency=STORAGE` **and** `device=CPU`, `rows_miss > 0`, `blocks_read > 0`, and `cpu_ticket_wait_ns` printed | any count is zero, or `cpu_ticket_wait_ns` is missing |
| any SYCL-gather arm | the §7.4 record-mode witness, and `lazy_zero_copy_reads == 0` (§7.3) | any witness fails |
| negative control, once | a `--lazy-mode off` run shows **no** `[PLACEMENT] lazy` line and no lazy counters | the check fires anyway (then it proves nothing) |

**Correctness gate, every arm.** Greedy decode (`--temp 0 --seed 42`, 64 tokens)
must be **token-identical** across H, S and `--lazy-mode off` for Qwen, and
across A and C for DeepSeek. th32's gather paths, the SYCL-owned gather of §7.2
and the CPU op of §5.3's CPU form, dequantize the same rows through the same
`to_float` trait, so their F32 activations are bitwise equal and any token
difference between them is a defect, not noise. The backend's own SYCL
`GET_ROWS` is not one of them: under the default `GGML_SYCL_F16` it dequantizes
Q8_0 in half precision (`fefb92980 ggml/src/ggml-sycl/getrows.cpp:1788-1796`,
`dequantize.hpp:642-645`), so a `--lazy-mode off` run whose table gather runs
there is not a bitwise baseline (r25 M-4). DeepSeek additionally passes `9g22`'s
CPU oracle gate.

**Every gate set carries a default-context arm** (owner, relayed 2026-09-27,
with the lead's vehicle clarification of the same day). Its vehicle is one whose
context defaults to the training context: `llama-cli`, `llama-completion` or
`llama-server` with no `-c` (`n_ctx` 0 means the context the model was trained
with, `fefb92980 common/common.h:451`). There it runs the correctness gate
(greedy, 64 tokens) and the S and H decode arms, and DeepSeek's A and C, each
with §8.1's positive controls in that run's own log; a run that pins `-c` is an
additional arm, never the only one. The `llama-bench` and `llama-perplexity`
arms stay at their fixed contexts and are never the default-context arm:
`llama-bench` has no `-c` and sizes its context as prompt + generated + depth
tokens (`fefb92980 tools/llama-bench/llama-bench.cpp:1286`), so a "no `-c`"
bench arm never reaches the training context and would meet the ruling
vacuously, and `llama-perplexity` sets `n_ctx` to 512 before parsing (`fefb92980
tools/perplexity/perplexity.cpp:2016`) and refuses a text of fewer than 2 x
`n_ctx` tokens (`:480-484`), so wikitext-2's 297193 tokens (Appendix A.3) cap it
at `n_ctx` <= 148596. **A default-context run that refuses, at load or at
context init, scores FAIL**, th32's own `host-budget-exceeded` included, and is
never "no arm": KV that does not fit on the card is re-placed to host rather
than refused (`llama.cpp-uize` part 3, `35ce2e65f`), host KV enters the ledger
(§4.2), and the planner must answer with STORAGE for the table. The default
context is where KV placement meets the table's host budget: §4.5's host-KV term
is 0 at the gate contexts, where Qwen's attention stays on the card, while at
the default context KV may be placed on host, and the ledger counts it. So a
default-context arm takes its S `RESERVE_MB` and its H launch precondition from
its own load-only plan line, as S already does, never from §4.5's figures. Where
that plan line shows the H precondition unmet, as at a `MemAvailable` of 171.4
GiB, the H arm is not run at that context; that is a launch precondition read
from a completed plan, not a refusal, and the S arm still runs and must pass.

**Instruments.**

- `T_gather`: the worker's time from the gather request to the event's
  signal, per ubatch. For STORAGE it includes the ticket wait; `T_ticket`
  separates it out.
- `T_ticket(t)`: the part of table `t`'s gather spent waiting on its ticket,
  measured from the request's post (≈ graph launch). It bounds the IO
  latency the device can see.
- `T_stall(t) = max(0, T_ticket(t) - T_window(t))`, where `T_window(1)` is
  block 0's time and `T_window(14)` blocks 0-13's, from the op timing the lead
  already collects. Where the main queue has profiling enabled, T8 also reports
  the device gap before the output copy directly, and that measured gap
  replaces the estimate.
- The IO count and bytes, the hit and miss counts (§10), and end-to-end
  `llama-bench`, `llama-perplexity` on wikitext-2 test, and the decode vehicle
  (§8.5).
- **The engine rate is a ratio, never an absolute** (I6, M12).
  `rows_miss_reads / T_io` over a cold PP512 ubatch, divided by the IOPS of a
  **same-session fio QD64 control on the same extent**, run immediately before
  and after the arm (Appendix B, `io_probe2.sh` with `FOFF` at the table's
  offset). The ratio must be **>= 0.9**; the pool measured 0.99 against the
  bracketing controls (§2.4). If the two controls differ by more than 20%, the
  ceiling moved during the arm and the run is void, not scored. Each arm
  records the per-device `io_done` and write deltas from `/proc/diskstats` for
  its duration, so that fresh (NVMe-foreground) and settled (backing-device)
  placement are told apart: during the `blk.1` probes the NVMe wrote 7.5 GB in
  15 s at 4657 IOPS, and 0.4-0.5 GB at 7070-7242 two minutes later.
- **Concurrent expert IO** (*lead ruling §T4*, I-4). On DeepSeek, every arm
  also records the cold-expert reads that shared the device in the same
  window: `llama.cpp-yl3v`'s read and byte counters, per ubatch and per decode
  token, next to the engram counters. Without that record, an engram latency
  measured on the shared device cannot be told apart from contention.
- **Scaling rule for latency targets.** The latency targets below were measured
  with the control at 4657-7242 IOPS. When an arm's bracketing control reads
  below 4657, each latency bound is multiplied by `4657 / control`.

### 8.2 Qwen decode

| arm | target | derived from |
|---|---|---|
| H | `T_gather` <= **20 us/token** (the worker's gather time); tg128 within **1%** of `--lazy-mode off` (scored on that arm's second run, its table pages resident) | measured 4.2 us for 16 rows including dequant |
| S-cold | `T_ticket` mean <= **4.3 ms/token**, p99 <= **15 ms** | batch of 16: mean 3.75 ms, p99 13.0 ms (A.2), + 15% |
| S-warm | mean `T_ticket` <= **(1 - P0) x 4.3 ms**, where P0 is the steady-state zero-miss share at the *planned* `cache_slots` (§8.5): prose 32-43%, mixed 63-68%. This is a ceiling, not a discriminator (M13); the runtime-versus-simulation match of §8.5 is the discriminator | §6.2.1 |
| S-hot | `T_ticket == 0` on **>= 99%** of tokens | every row hits |

Expressed as throughput: `tg_S >= 1 / (1/tg_H + T_stall)`. If H decodes at
20 tok/s (50 ms per token), S-cold must decode at **>= 18.4 tok/s**.

### 8.3 Qwen PP512

| arm | target | derived from |
|---|---|---|
| H | `T_gather` <= **3 ms/ubatch**; pp512 within **1%** of `--lazy-mode off` | measured ~2 ms for 8192 rows |
| S-cold | engine ratio >= **0.9** (§8.1); `T_ticket` <= **1.15 x U / (0.9 x control IOPS)**: at the r1 control (9124), real text (U 7028-7104) <= **1.0 s**, `llama-bench` random (U 8116-8192) <= **1.15 s** | §5.1, §6.2.1 |
| S-warm | the same formula with `U` from the simulation at the planned `cache_slots`: prose (512 MiB) 4335 -> **<= 0.6 s**; mixed (512 MiB) 2275 -> **<= 0.32 s** | §6.2.1 |
| S-hot | `T_ticket == 0` | |
| phase 2 lookahead, multi-ubatch prompt | stall of ubatch `k >= 2` <= `max(0, IO(k) - compute(k-1)) + 10%` | §5.4 |

Expressed as throughput: `pp_S >= 512 / (512/pp_H + T_stall)`.

<a id="th32-ds-targets"></a>
### 8.4 DeepSeek Q2_K (the harder case)

**Where it runs.** Inside `llama.cpp-9g22`'s real-model gate protocol, P0-P7 of
its §5.2 (rev 10, `054069548`), after gates 1-4 pass: phase 1 on
`level_zero:0` (the B70 alone), caches fixed at 2 GiB per table with the index
included (§4.3 step 4), no DSpark draft, **in the owner's §D4 configuration**:
lever (s) landed, and the cold set `C` that 7v5c computed for that launch's
`MemAvailable` reading resident on STORAGE (§4.5, "with the owner's §D4
applied"). P3 is claimed only as §4.5 evaluates it: on today's code, or after
(s) with no cold set, P3 fails at every reading taken today, so §8.4 is
unrunnable until (s) and `llama.cpp-yl3v` have landed. Two gathers per ubatch;
`T(a..b)` is the measured time of blocks `a..b` for the same ubatch; `U` is
the unique miss rows of the ubatch, both tables together.

**The device is shared** (*lead ruling §T4*, I-4). The cold experts and the
engram tables are read from the same USB bcachefs members through the one
pool (§5.2). The engram bounds below were measured with no competing IO. They
are scored only on the tokens and ubatches whose concurrent expert IO the arm
recorded (§8.1), and they are reported split by that IO: none, or some. The
expert-IO terms themselves are **measured-later targets, not promises**. Their
estimates are `9g22`'s (§4.5.3), from this document's facts (§9.2.2):

- **PP512.** A ubatch touches 99.97% of a layer's experts, so it reads the whole
  cold set once, at the 0.29-1.2 GB/s measured for expert slices (§9.2.2). The
  per-ubatch time that gives at each reading's cold set is `9g22`'s (rev 10,
  [its §4.5.3](deepseek41-runtime.md#453-speed-estimates-not-targets); *lead
  ruling §T6*, m-10), and it is far more than the engram tables' ~2.5-3.5 s. PP
  on DeepSeek phase 1 is bounded by the cold experts, not by the engram IO.
- **Decode.** A token reads `240 x c x (1 - r) x 12.8 MB` of cold experts
  (§9.2.2), where `c` is the cold set's touch share, which the usage profile
  decides. The engram rows of a token are ~4 KiB reads that the deadline order
  puts ahead of any expert read for a later layer (§5.2); the bound that holds
  for them on a shared device is measured, not assumed.
- So "decent speed" for DeepSeek on this host is set by the cold-expert IO,
  and `9g22` §4.5.3 owns that number.

**Decode, S-cold** (48 rows per token). Re-derived from the probe that issues
both tables' reads together through one pool, which is what the engine does (I7;
`batch_probe3 split`, A.2, measured in the same minute as the 4657-IOPS control,
the slower state):

| quantity | measured | target (+15%) |
|---|---|---|
| `T_ticket(1)`: `blk.1`'s 24 rows landed, with `blk.14`'s 24 in flight | mean 7.13 ms, p50 6.81, p90 9.16, p99 14.63 | mean <= **8.2 ms**, p99 <= **17 ms** |
| `T_ticket(14)`: all 48 landed | mean 11.23 ms, p99 21.00 | mean <= **12.9 ms**, p99 <= **24 ms** |
| `T_stall(14)` | | mean <= **max(0, 12.9 ms - T(0..13))** |

`blk.1`'s target ignores the block-0 overlap, so it is conservative.

**Decode, S-warm, bucketed by miss count** (M13). Per token, `m` is the number
of rows (both tables, after dedup) that issued a read. `T_ticket(1)`'s mean over
the tokens of each bucket must stay within 1.15x the single-table miss curve
(`batch_probe3`, A.2) at the bucket's top, or the split probe for the last one:

| `m` | 0 | 1-4 | 5-8 | 9-16 | 17-24 | 25-48 |
|---|---:|---:|---:|---:|---:|---:|
| measured (mean / p99, ms) | - | 2.37 / 5.16 | 2.81 / 6.96 | 4.23 / 9.00 | 5.53 / 13.94 | 11.23 / 21.00 (all 48) |
| bound on the bucket's mean / p99 (ms) | `T_ticket == 0` on >= 99% | **2.7 / 5.9** | **3.2 / 8.0** | **4.9 / 10.4** | **6.4 / 16.0** | **12.9 / 24** |

The aggregate S-warm ceiling is then `sum over buckets of P(bucket) x bound`,
with `P(bucket)` from the simulation at the planned `cache_slots` (§8.5). This
discriminates where the r1 target could not: a partial-miss token is charged
for its misses, not for a full batch.

**PP512.**

| arm | target |
|---|---|
| S-cold | engine ratio >= **0.9**; `T_ticket(1) + T_ticket(14)` <= **1.15 x U / (0.9 x control)**. At a 7070-IOPS control: `llama-bench` random (U 24576) <= **4.4 s**, prose (U 22117) <= **4.0 s** |
| any S arm, `blk.14` | `T_stall(14)` <= **max(0, 1.15 x U / (0.9 x control) - T_ticket(1) - T(0..13))** |
| S-warm (gate cap: ~4.5 GiB of cache in simulation units, §4.5) | the same formula with `U` from the simulation at the planned `cache_slots`. Orientation at 7070 IOPS, bracketing columns: prose (wiki train) 13790 (2 GiB) - 12389 (8 GiB) -> **<= 2.5 - 2.2 s**; mixed 7916 - 6370 -> **<= 1.4 - 1.2 s** |
| S-hot | `T_ticket == 0` |
| phase 2 lookahead | as §8.3; required for DeepSeek (D9) |

**Position arms** (*lead ruling §T1*, C2 and M10). The provider decides the
position; these arms test that decision rather than assume it:

| arm | STORAGE gather | host waits on the path | predicted |
|---|---|---|---|
| **A** (the design) | SYCL-owned gather (§7.2): the worker's gather posted at prefetch, event-edged copy before the consumer | none on the GPU path (§5.3) | best or tied |
| C | upstream order: the gather as a CPU custom op in the leading split (rev 2a's HOST form); its output reaches the device through the same step-3 copy, and the op guards that copy's source by waiting for the previous graph's copy event | the submitting thread waits for the IO before layer 0 is submitted, and for the previous copy when the device runs a graph behind | loses the whole window: ~7-11 ms per cold token; wins nothing on a fully hot cache |

- **No arm B.** r1's B ("expand right before the consumer") is identical to A
  now: the gather request is posted at prefetch whatever the node's position,
  and the copy always sits before the consumer, so moving the node changes
  nothing that runs.
- **Witness on every arm**: the §7.4 record-mode witness, including its
  consumer-copy submit check (C has no SYCL gather, so its witness is the
  same counters with `lazy_boundary_dispatches == 0`, and §7.3's
  resolved-pointer `lazy_zero_copy_reads == 0`), plus the scheduler's
  split count per graph (`GGML_SCHED_DEBUG`), which must be the same for A
  and C: neither may add a split.
- **Protocol**: S-warm prose (wikitext-2 test through `llama-perplexity` for
  PP, the §8.5 decode vehicle for tg), **5 ABBA quads** of A against C, one
  process per run under P0-P7, B70 tg being the noisy axis (cv 3.3%).
- **Decision rule, fixed now**: C replaces A only if its **median paired tg
  over the 10 pairs beats A's by >= 2%**. Anything less is a tie, and a tie
  goes to the arm with fewer host waits, which is A.

**DSpark** (I9). The draft's ≈ 7.9 GiB of weights are **device** demand; with
its ring they move ≈ 10.0 GiB of experts to the host, the +10.0 of §4.5 case C
(`9g22` rev 10 §4.5.2, r6 m-1). Under owner §D4 its whole effect is paid in cold
set, and that effect, VRAM displaced as well as host bytes, is `9g22`'s figure
([its
§4.5.2](deepseek41-runtime.md#452-the-host-fit-r2-c-a-r3-c-1-c-2-owner-d4), rev
10); this document does not restate it. Verify steps carry up to 6 tokens, so up
to 288 rows per step: 74 ms mean, 131 ms p99 cold on the Q3_K_M extent (§9.2.3).
The DSpark arm gets its own targets from a Q2_K re-measurement when `9g22`'s
gates 1-4 have passed; this document sets none now.

### 8.5 Runtime against simulation, and the traps

- **The hit-rate match** (M4, M5). The runtime hit rate must match the
  simulation to within **±5 points**, with the simulation re-run under the
  run's own conditions: CLOCK at the **planned** `cache_slots` from the run's
  `[PLACEMENT]` line; the vehicle's exact token stream; `llama-perplexity`'s
  chunking (each chunk of `n_ctx` tokens restarts the n-gram history; the row
  cache persists across chunks); the first 25% of references dropped. Hit and
  miss are counted per reference, as §10 defines them.
- **The decode vehicle** (M4). `llama-perplexity` is PP-only, so S-warm decode
  uses `llama-server` `/completion` with a fixed prose prompt from outside the
  repo (the first 2048 tokens of wikitext-2 train), `n_predict 256`,
  `temperature 0`, and `return_tokens: true`, so the generated token ids come
  back exactly. Re-tokenizing the generated text is not acceptable: a sampled
  token sequence need not be the canonical tokenization of its text. (The
  `return_tokens` field exists upstream; T9d confirms it on this tree before
  relying on it.) The simulation then replays prompt plus generated ids.
- **Corpora a target may quote**: prose and mixed only (§6.2.2).
- **`llama-bench` prompts are uniformly random tokens** (`std::rand() % n_vocab`,
  `694ec2354 tools/llama-bench/llama-bench.cpp:2181-2183`, `:2204-2213`). Their
  n-grams never repeat, so on STORAGE **every** `llama-bench` pp and tg number
  is the cold case, whatever the cache size. A "warm" or "hot" claim needs the
  vehicles above. A flat `llama-bench` curve across cache sizes does not mean
  "the cache does nothing".
- **Filesystem changes move every S number by an order of magnitude.** The
  ext4-on-NVMe reference measured 110 k IOPS at QD64. On a file without
  per-256-KiB checksums on an NVMe target, Qwen's S-cold PP512 would be
  ~75-100 ms per ubatch instead of ~1 s, and DeepSeek's ~0.25-0.3 s instead of
  ~3.5 s. These are projections, not targets, and there is no such target on
  this host today (D3, §13).

---

## 9. Generalisation to any `TENSOR_READ_LAZY` tensor (item 7)

Nothing above is specific to one model except three arch-supplied facts:
`rows_per_token` (for the planner's floor and staging size), `consumer_layer`
(for IO priority, the consumer device and the position) and, optionally,
`peek_rows` (for lookahead). Two consumers with different geometry already
exercise it: one Q8_0 table read 16 rows per token at block 1, and two Q2_K
tables read 24 rows per token each at blocks 1 and 14.

- **Any row table.** Type, `ne[0]` and `row_size = ggml_row_size(type, ne[0])`
  come from the tensor. Quantized rows are whole blocks by construction.
  Dequant goes through the type traits. The read span and the block ring's
  slot are sized from `row_size` (§5.2, §3.3), so a row wider than 4 KiB,
  such as gemma4 E4B's 11424 B PLE row, is read as 3-4 blocks (r5 m-10).
- **Consumers.** Every consumer of a lazy tensor goes through
  `build_lazy_get_rows`. That covers an indexed `get_rows` (qwen4exp's hashed
  rows; DeepSeek's engram rows; gemma4's `inp->tokens`,
  `694ec2354 src/models/gemma4.cpp:450`) and a constant-row view: gemma4's
  `ggml_view_1d(..., row 0)` (`:462`) becomes `build_lazy_get_rows` with a
  constant one-row index, supplied by a new input class
  `llm_graph_input_const_rows` whose `set_input` writes the constant ids and
  whose `can_reuse` returns true when the ids are unchanged (M9).
- **The consumer check is a scan, not an assert at the call** (M9). Nothing at
  the call site can see a stray consumer (and arm C's custom op has no edge to
  the lazy tensor at all). After graph build, `llm_graph_context` scans
  every node's `src[]` and follows each `view_src` chain to its root; any node
  other than a lazy `GET_ROWS` whose chain reaches a lazy tensor fails the
  build with `GGML_ABORT` naming the node and the tensor. Under SYCL the lazy
  tensor's data is a non-dereferenceable sentinel (§3.2), so an unhandled
  consumer would otherwise fault at run time, far from its cause. T6's RED: a
  graph with a `ggml_view_1d` of a lazy tensor fails the build, naming it.
- **Several tables.** Each gets its own `lazy_table`, its own residency, and
  its own cache: fixed in a cold-set plan (§4.3 step 4), otherwise a share of
  what is left above the P3 floor (§4.3 step 5).
- **Non-SYCL builds keep upstream's mmap path verbatim.** The divergence is
  confined to the loader's SYCL branch, the descriptor buffer and
  `build_lazy_get_rows`.

<a id="th32-contract"></a>
### 9.1 The consumer contract

What a model implements to get the residencies, the cache, the prefetch and the
accounting. `llama.cpp-9g22` (`docs/design/deepseek41-runtime.md` §3.3)
consumes this contract and links here for it; this section is the one source.

1. **Mark and declare at load.** Create the table with `TENSOR_READ_LAZY`, and
   add an entry for it to the lazy-tensor declaration, keyed by (arch, GGUF name
   or name pattern), `llama_model_lazy_tensor_names(arch)` (`llama.cpp-5efe`'s
   C1, rev 3l `a68508606`), whose entries today are
   `per_layer_token_embd.weight` for qwen4exp and for gemma4. The name list
   takes no hparams; the arch key is required, because another arch may create
   the same name without the flag (gemma3n's `per_layer_token_embd.weight`,
   `fefb92980 src/models/gemma3n.cpp:39`). A table repeated per block is
   declared by a name with one `%d` for the block index; deepseek41's engram
   tables are declared that way by `llama.cpp-9g22` when that arch lands, not by
   C1 (*§F2a* I-9 addenda 4 and 8). Its `rows_per_token` and `consumer_layer`
   are returned, for each matched name, by the arch's case in the one per-arch
   attribute function beside the declaration, which reads the loaded hparams:
   `llama_model_lazy_tensor_attrs(arch, hparams, name)`, returning
   `llama_lazy_tensor_attrs { uint32_t rows_per_token; int32_t consumer_layer;
   }` with 0 and -1 for "not known" (5efe's; *§F2a* I-9 addenda 5, 6 and 8). The
   pre-fill stores the result on the per-model record entry. The loader
   pre-fills the lazy-range record from the declaration before the plan is
   computed, and `create_tensor` aborts, naming the tensor and the arch, when
   the flag and the declaration disagree, whatever the mode and the entry's
   eligibility (§4.1; *§F2a* I-9 addendum 4). The entry feeds the inventory
   (§4.1). An entry the SYCL branch takes (`deferral == STORAGE`, §4.1) for
   which the attribute function returns a "not known" `rows_per_token` or
   `consumer_layer` is refused by T6 before the early plan reads it, naming the
   tensor, rather than planned with a guessed floor; a non-taker entry is never
   refused for them.
2. **One entry point.** Every read of the table goes through
   `build_lazy_get_rows(lazy, rows, consumer_layer)`, and the layer loop calls
   `lazy_expand_for_layer(il)` at the top of each iteration. The post-build
   scan above enforces it.
3. **The rows input.** `rows` is an I32 graph input of
   `rows_per_token x n_tokens` elements (DeepSeek: 24 per token per table),
   fully written in `set_input` from host state only, with no dependency on any
   device result. Every value is in `[0, ne[1])`; `prefetch` aborts on an
   out-of-range id rather than read past the table. Padding, sequence starts
   and blocking tokens are the model's hash's business. int32 ids suffice while
   tables have fewer than 2^31 rows (DeepSeek: 384.0 M).
4. **The output.** F32 `[ne[0], rows_per_token x n_tokens]`, the shape
   `ggml_get_rows` returns; the model reshapes it (DeepSeek: to
   `[6144, n_tokens]` per table). It is an activation (§7.3). Raw blocks are
   not offered.
5. **Position and executor: the provider decides** (*lead ruling §T1*, C2). The
   model does **not** choose where the gather sits or what runs it. The helper
   places the gather of either residency on the executor that §3.2's query
   returns for the consumer layer from the plan's residency, never from `-ngl`
   (*lead ruling §M58a*): at the head of the consumer device's first split as a
   SYCL-owned op when that is a SYCL device (§7.1; DQ-C2 in §13, *lead ruling
   §T5*), and as the CPU op at the head of the layer when the plan executes it
   on the CPU (§7.1 and §5.3's CPU form, *§F2a* I-9 addendum 9, *§M58b*), and
   §8.4's arms and T6's arms (l), (m) and (n) test that choice. A consumer must
   not hard-code "before block N": that is the r1 contract, withdrawn.
6. **Optional lookahead.** `peek_rows(next_ubatch, tail)` returns the next
   ubatch's rows from the model's sequence state, so the IO of ubatch `k+1`
   overlaps the compute of ubatch `k` (§5.4). Lookahead reads land in the row
   cache only, so they need no slot at any depth (§7.2).

The SYCL side of the same interface:

| piece | what it is | owner |
|---|---|---|
| `lazy_table` | model-scoped unified-cache object, keyed by the weight's `ggml_sycl_cache_id`; refcounted by the descriptor buffer and every outstanding ticket; llama holds only the descriptor buffer, never a row pointer; none exists on the `no_alloc` pass (§3.2) | the unified cache (§3.2) |
| descriptor buffer | on the consumer's executor: `ggml_backend_sycl_lazy_buffer_type(consumer device)` for a SYCL consumer, `ggml_backend_sycl_lazy_host_buffer_type()` for a CPU one (addendum 9); sentinel base, nominal size, `WEIGHTS` usage; on the `no_alloc` pass, a context-less size-0 dummy of the host form with a null base and no table (§3.2) | the model, like any weight buffer (§3.2) |
| `lazy_gather_ctx` | per context, per table: the per-context rows of §3.3 (the output slot, the epoch ring with its marker words and events; for STORAGE also the landing slab; for a CPU-executed consumer only §3.3's right-hand column), its strand, and the carrier's (posted `k`, `consumed`), created as (`none`, clear) and set back to posted `none` by each ubatch's end point (§5.3, r13 m-6) | refcounted: every in-flight request, waiter and enqueued copy holds it or the handles it writes (§5.3); its destructor is the sole release point |
| block ring | the IO pool's, shared by every context | a `mem_handle`; each in-flight read holds a reference on its slot |
| `prefetch(gather_ctx, rows, n, class) -> ticket` | called by `lazy_prefetch()` after `set_inputs()` (§7.5); it copies the ids into the epoch ring after one non-blocking marker load and posts the gather request; for a CPU-executed consumer's table it classifies the ids in place instead; it never waits (§5.3) | |
| the gather | both residencies: the IO engine's worker (D6-M(b)) + an H2D copy ordered on its host-signalled event (§5.3, §7.2, §7.3); for a CPU-executed consumer, the CPU op of §5.3's CPU form | th32 owns it (*lead ruling §T1*, D6(c); *§T2*, DQ) |
| `lazy_io_submit(fd, offset, length, dst handle, ticket, class, epoch, deadline)` | the one IO entry point, shared with `llama.cpp-yl3v`'s cold-expert slice reads (§5.2, §9.2) | th32 (T5) |

### 9.2 Cold routed experts on STORAGE (phase 1, owner §D4; the path is `llama.cpp-yl3v`'s)

**Status.** On 2026-09-26 the owner first ruled Q3_K_M with its coldest routed
experts on STORAGE (owner §D1), then made Q2_K the primary and moved the cold
experts to a later ticket (owner §D2). The Q2_K fit then fell short of this
host, and the owner chose **"Zones + SSD cold experts"** (owner §D4): lever (s)
plus the coldest Q2_K experts on STORAGE, executed by the CPU. That sanctions
STORAGE→CPU for experts and brings `llama.cpp-yl3v` forward to Q2_K phase 1.
So cold experts **are** on STORAGE in phase 1.

**Who owns what** (the owner's either/or in §D4: this design names the
interface). `llama.cpp-yl3v` owns the cold-expert path: choosing the experts
(in the source order of `9g22`'s §7 D1, *lead ruling §T5*), executing them on
`CpuExpertPool`, its landing slab, and prefetching from routing. The expert
read path differs from the row path in exactly those places: its unit is an
expert-layer's three slices (~12.8 MB on Q2_K), not a row; its reads are known
only once the layer's router has run, not at `set_input`; and its executor is
the CPU expert pool, not the gather worker. This design provides the rest:

| th32 provides | used by yl3v as |
|---|---|
| the `STORAGE` residency state (§3.1) | the reported state of a cold expert-layer |
| `lazy_io_submit(fd, offset, length, dst handle, ticket, class, epoch, deadline)` and its pool (§5.2); callable from any thread, including one that is not the graph's submitter | the slice reads, class 0, epoch = the routing ubatch's, deadline = the expert's layer, in chunks of at most 512 KiB. Because they are known only once the router has run, the edge from the router's device event to the read must not be a `host_task` that `depends_on` that in-flight event, which blocks the submitter (`llama.cpp-c6ah`); it is D6-M(b)'s host-signalled event or a non-submitting thread (§13; *lead ruling §T6*, relay). The ordering rule is §5.2.1, the single source (*§D10* I-3) |
| the ticket and epoch machinery (§5.3) | completion and cancellation of a slice load |
| the host ledger and `host_planned_bytes` (§4.2) | the charge for the landing slab (`2 x` the largest per-layer cold bytes, *lead ruling §D6* I-4) and the credit of `C` |
| the order of §4.3, with the cold set found in step 5 by 7v5c's search (*lead rulings §D5, §D6*) | the size `C` that its profile order fills |

The cold-set **size** is not a th32 number either: 7v5c computes it at load
(§4.3 step 5), and `9g22` §4.5.2 records what it reaches on this host (§4.5).

#### 9.2.1 What the interface keeps open

- **A planned, cache-owned `O_DIRECT` landing slab.** The engine of §5.2
  reads into `mem_handle`-owned `HOST_CPU` memory whose size the planner charges
  (§3.3). Its read unit is an (fd, offset, length, destination, ticket)
  request, not a row: a 4 KiB row block and a 512 KiB slice chunk are the same
  call. yl3v adds its slab role, not a new engine.
- **A hotness ranking by expected reads per step.** The planner's inputs are
  per-object sizes plus an expected-reads-per-step score. For a lazy table it
  is `rows_per_token` (§4.1). For an expert-layer it would be its expected uses
  per step, from the usage profile. §4.3's ordering, "charge the warm set,
  place the coldest last", then covers both; for experts the §D1 selection
  supplies the order and the §D5/§D6 search the size.
- **The CPU executor already takes a lease.** `cpu_expert_task` carries a
  `weight_lease` `mem_handle` (`76c7f6548 ggml/src/ggml-sycl/cpu-dispatch.hpp:127`)
  and the host AOS executor consumes file-layout blocks
  (`:136-149`), so a cache slot filled from STORAGE can be executed as a host
  expert is today. P3 then holds as it does for lazy tables: the CPU executes,
  and only the expert's output reaches the GPU.

<a id="th32-cold-expert-facts"></a>
#### 9.2.2 Facts recorded for the cold-expert path

| fact | value |
|---|---|
| one expert of one layer, Q2_K (phase 1) | **12.8 MB**: gate and up Q2_K 3870720 B each, down Q3_K 5068800 B (4.58 GiB per layer / 384) |
| one expert of one layer, Q3_K_M | **16.77 MB**: gate and up Q3_K 5068800 B each, down Q4_K 6635520 B; **18.25 MB** at layers 0-1 (down Q5_K 8110080 B) |
| where its bytes are | three contiguous slices, one in each `ffn_{gate,up,down}_exps`, ~2 GB apart in one shard; expert `x` starts at `data_offset + x * slice`; slices are not 4 KiB-aligned (5068800 = 1237.5 x 4096) |
| one expert-layer, 16 x 512 KiB chunks in flight, fresh DeepSeek shards | **14.1 ms mean, 29.1 ms p99, 1.2 GB/s**; about 70% of the bytes came from the NVMe, because the shards were minutes old |
| the same, 2 / 6 / 12 loads concurrently | 22.5 / 122 / 222 ms per load; 1.5 / 0.83 / 0.91 GB/s aggregate |
| the same shapes on a settled Qwen split (mostly on the T7) | **0.29 GB/s** (1 load) to **0.47 GB/s** (6 loads); the T7's 5 Gb/s link is the cap |
| reading promotes | `promote_target=nvme` wrote 2.7 GB to the NVMe during a 12 s settled read, so repeatedly read extents become NVMe-cached |
| PP coverage | a PP512 ubatch uses 99.97% of a layer's 384 experts (1 - (1 - 6/384)^512), so PP reads the whole cold set once per ubatch |
| static prior | the owner's fallback is the `exp_probs_b` bias (owner §D1, §D2). **A risk to it**: `exp_probs_b` is DeepSeek-V3-style load balancing, which equalises expert load in training, so the prior may carry little signal about which experts are cold (`hash_layer_count = 0`). The profile is **not** mandatory (*lead ruling §T5*); where this prior sits in the source order is `9g22`'s [§7 D1](deepseek41-runtime.md#7-owner-decisions), the one place it is stated. P3 holds with any source; a weak prior costs speed, not fit |
| decode cost model | `240 x c x (1 - r) x S / BW` per token, where `c` is the cold share of expert uses, `r` the share a RAM cache serves, and `S` an expert-layer's bytes: **12.8 MB on Q2_K** (phase 1), 16.8 MB on Q3_K_M |

Raw logs are in Appendix A.2 (`expert_probe`).

<a id="th32-dspark"></a>
#### 9.2.3 DSpark verify steps

`llama.cpp-9g22` reports that DSpark verify batches carry up to 6 tokens (the
anchor plus 5 drafts). So a decode step issues up to 6 x 48 = 288 engram rows,
and rows for rejected drafts are wasted reads; the history rolls back, so
there is no correctness impact. Measured on the `blk.1` extent: a 288-row
batch takes **74 ms mean, 131 ms p99**, and a 144-row batch 24 / 52 ms (64
workers; Appendix A.2), against 8.95 ms for one token's 48. On a cold cache a
verify step therefore costs ~8x a plain token's engram IO, for at most 6
accepted tokens. The row cache serves repeated n-grams in the drafts exactly
as in PP. The §8.4 decode targets assume batch-of-1; the DSpark arm gets its
own targets from this measurement when DSpark lands.

---

## 10. Plan == reality: witnesses and counters

- **Plan log, WARN level** (default verbosity drops INFO, CLAUDE.md
  "`llama-bench` traps"):

  ```
  [PLACEMENT] lazy per_layer_token_embd.weight residency=HOST table=51880.0 MiB resident=51880.0 MiB locked=yes output_slot=10.0 MiB/ctx host_budget=... host_planned=... leftover=...
  [PLACEMENT] lazy blk.1.engram_embd.weight residency=STORAGE table=30762.6 MiB cache=... MiB (N slots, index ... MiB) floor=... yieldable=... MiB output_slot=24.0 MiB/ctx consumer_layer=1 device=... rows_per_token=24 file=<split> off=5785327904
  [PLACEMENT] lazy blk.14.engram_embd.weight residency=STORAGE table=30763.4 MiB cache=... consumer_layer=14 ... off=4032
  [HOST-PLAN] budget=... (source: computed|GGML_SYCL_HOST_RESERVE_MB) planned=... weights=... named_terms=... kv=... zones=... (caches carving: 1|2) envelope=... lazy_resident=... reserve=... cold_set=... (expert-layers ..., slab ...) p3_post_load=...
  ```

  (The `host_budget` figures are elided because they depend on the tenants'
  memory at load, §4.5. `device=` prints §3.2's query answer for the consumer
  layer, a SYCL device or `CPU` (addendum 9, *lead ruling §M58a*), and is the
  observable of which descriptor form was bound. A CPU-executed consumer's table
  prints `output_slot=none` and, for STORAGE, its one landing slot as
  `landing=<MiB>/ctx`: §3.3's right-hand column, *§M58b*.)
- **Activation check.** When the `lazy_table` is materialized, it sums its
  handles' sizes and compares them with `host_resident_bytes` from the plan; the
  ledger does the same for every host byte at each publish. A mismatch is a
  `GGML_LOG_ERROR` (`[HOST-PLAN-BUG]`) plus a typed refusal: plan != reality is
  a defect, never tolerated. T4's RED: a fixture whose materialization allocates
  one extra chunk fails the check, naming both numbers.
- **Counters, compiled unconditionally** (M6). They are plain relaxed atomics in
  every build, with no define and no environment gate, printed per context at
  free at WARN, and exposed to tests through a proc-address hook. An arm that
  scores them takes the per-context line's presence as its liveness check, since
  an abort or `_Exit` skips the context free and with it the line (the reason
  *lead ruling §M74(g)* gives for its own dump; the line itself is admitted by
  *lead ruling §M74* (f)). Per table: `rows_requested`, `rows_hit`, `rows_miss`,
  `rows_joined`, `blocks_read`, `bytes_read`, `stall_ns_total` (`T_ticket`),
  `stall_ns_max`, `gather_ns_total` (`T_gather`), `ubatches`,
  `tickets_cancelled`, `chunks_yielded`, `lazy_boundary_dispatches`,
  `slot_wait_device_ns` (the device part of rev 2d's `slot_wait_ns`; its IO part
  is `stall_ns_total`, §5.3), `slot_marker_stale`, `epoch_ring_exhausted`,
  `lazy_epochs_cancelled` and `lazy_epochs_drained`, each by site (`catch`,
  `sync`, `entry`, `teardown`, `backstop`): the cancel-plus-marker actions and
  the drains (flag and re-post) of each site's step, the no-action case
  uncounted (§5.3, *lead ruling §D12h* I-2), `copy_submit_ns` and
  `copy_submit_io_pending` (§7.4, per copy, per table), `lazy_output_copies`,
  `lazy_zero_copy_reads` (§7.3), `stage_queue_ns` (§5.3). For a CPU-executed
  consumer's table (§5.3's CPU form; *lead ruling §M58b*), CPU-side counters
  that the op bumps on its thread 0: `lazy_cpu_gathers` (the op's runs),
  `lazy_cpu_rows` (the rows it gathered) and `cpu_ticket_wait_ns` (STORAGE: its
  ticket wait); beside them `cpu_slot_held` and `lazy_cpu_epochs_released{site}`
  (`op`, the op's own release on each return; `end`, `catch`, `backstop`,
  `teardown`). The SYCL gather's `lazy_output_copies`, `slot_wait_device_ns`,
  `slot_marker_stale` and `epoch_ring_exhausted` stay 0 for such a table. Per
  context: `lazy_graph_direct`, by reason (`fa_not_engaged`, `pp`,
  `not_profitable`, `site`; §7.4), `lazy_context_poisoned` (§5.3),
  `lazy_mark_consumed`, by site (`catch`, `sync`, `entry`, `backstop`: the steps
  that completed under the failed-graph mark, §5.3; *lead rulings §D12e*,
  *§D12f* I-2), and `graph_exit_deferred`, the `graph_compute` calls that took
  the deferred exit (T7's fixtures assert that path with it). Per SYCL backend
  (*lead ruling §D12m* 3, 8): `ubatch_begin_calls`, the calls into
  `ggml_backend_sycl_ubatch_begin`, which equals the ubatch count on every SYCL
  backend the scheduler holds; `ubatch_end_calls`, the calls into
  `ggml_backend_sycl_ubatch_end`, which equals `ubatch_begin_calls` on every
  backend, on the failure returns as on the success return (r14 m-3);
  `lazy_witness_checks`, the lazy witness's runs, which equals it on the backend
  with the lazy plan and is 0 elsewhere; `lazy_witness_rebaselines`, the checks
  that found a new allocation serial (§5.3); and `graph_compute_calls`, the
  entries into the backend's `graph_compute`, which the fixtures use to show
  that a failed ubatch never entered it, and that a path with no delimiter did
  (T6 (e), T7 (5)'s direct-drive arm; r11 m-1, m-5). Per device:
  `device_lost_faults`, the classifier's recordings of the fault fact, one per
  `[SYCL-FAULT] device lost` line (§5.3, *§D12q* I-1). Per pool: `class0_reads`,
  `class1_reads`, `inflight_joins`, and the reads and bytes submitted by other
  clients (yl3v's slices), so the shared-device record of §8.1 comes from the
  same place; for tests, the dispatch log of §5.2.1. Per host buffer:
  `host_buffer_syncs`, which counts the calls that enter
  `ggml_backend_sycl_host_buffer_sync` past its no-device return
  (`ggml-sycl.cpp:42539`), whether or not they wait; this is its one definition
  (*lead rulings §T9e*, *§D12h* m-3), and the only per-buffer form that can be
  kept, since the `!ctx` return has no buffer to count on and an allocated
  buffer never has `device < 0` (`:42633-42635`). Beside it, the buffer's stamp,
  the pair (context serial, allocation serial) set at `alloc_buffer`: the
  context serial names the context's entry in the context-serial registry, from
  which the sync and the fault attribution read the plan flag, and the
  allocation serial is a process-wide counter assigned at allocation, which the
  witness reads as the buffer's identity (§5.3, *§D12h* I-3, *§D12m* 5). The
  count is the guard declaration's witness, checked once per ubatch per family
  (§5.3, *§D12f* m-2), and T7 (5)'s GREEN counts it per ubatch as the input
  guard's positive control (*lead rulings §T9b, §T9c*).
  `lazy_mark_consumed{site}`, `lazy_epochs_cancelled{site}`,
  `lazy_epochs_drained{site}`, `ubatch_begin_calls`, `lazy_witness_checks`,
  `lazy_witness_rebaselines` and `graph_compute_calls` join the always-compiled
  witness counters, with `cold_mark_consumed`, `cold_witness_checks` and
  `cold_witness_rebaselines` on `9g22`'s side (*§D12e*, *§D12m* 8). The
  test-only hook that removes the sync's waits does not remove the count, so the
  REDs that use the hook still pass the witness.
- **Definitions** (M5), chosen so that runtime counts match the simulation's
  per-reference counting:
  - Every row reference in `rows` is one of: a **hit** (the row is in the
    cache when `prefetch` classifies it), a **join** (the same row is already
    in flight, for an earlier reference of this or another ticket), or a
    **miss** (neither).
  - `rows_hit` counts hits **and** joins, because the simulation, which
    inserts on first reference, scores a repeat as a hit; `rows_joined` breaks
    the joins out.
  - `rows_miss` counts misses, including a row whose *block* is in flight for a
    different row: that is a miss in the simulation, though it issues no read.
  - `blocks_read <= rows_miss`, and the difference is the block-sharing saving.
  - `rows_hit + rows_miss == rows_requested` is asserted at free.
- **Env-gated lines** (*lead ruling §M66*). A criterion that scores a line
  printed only under an environment variable sets that variable in the arm's
  command and carries a trace-live check from the same emitter, or the arm is
  VOID, which counts as failed. In this design: the `[SYCL-SUPPORTS-OP-DEBUG]
  decline` lines of T6 (l), (l-sync) and (m) need
  `GGML_SYCL_SUPPORTS_OP_DEBUG=1`, which each sets, (l-sync) in both builds'
  commands, and each criterion on them counts at least one line (the
  `blk.<consumer>.` decline, and in (l) the `off` run's table `GET_ROWS`
  decline), so the count is its own live check, and no criterion counts zero of
  them; the `[LAZY-READ-FAULT] injected` lines of T6 (p) need
  `GGML_SYCL_LAZY_READ_FAULT`, which it sets, and (p)'s own count of at least
  one injected line is the live check, since the variable's init WARN is a
  different emitter (r26 N-3); the `[SYCL-SEG-POLICY]` line and the replay
  counts of §7.4's witness and T7 (6) need `GGML_SYCL_GRAPH_DIAG=1`, which each
  sets, and each scores fields of a line that must be present, so a missing line
  is VOID. The lines gated by verbosity rather than by a variable are read under
  `-v` and scored as present: `llama_context: n_ctx`, `print_info: n_ctx_train`,
  `[S1-PRELOAD]`, the `[PLACEMENT]` priority, per-layer and total lines that T6
  (l) and T7 (8) compare, (k)'s per-model line, which (k) reads through a log
  callback (all INFO), and the KV cache's `layer   0: dev =` (DEBUG). The lines
  scored as zero, the mixed-state line, the retry warning and `[LAZY-PLAN-BUG]`,
  are WARN or ERROR and print at every level, and so does the ERROR `Compute
  error.` line that (p) scores as exactly 1 (r26 N-3).
- **Positive controls** for all of the above are §8.1's table: for a
  SYCL-executed consumer, H must show its line and `rows_requested > 0`, and S
  must show `rows_miss > 0` and `blocks_read > 0`; for a CPU-executed consumer,
  H must show `lazy_cpu_rows > 0` and S the CPU form's S row (r24 M-6); a
  `--lazy-mode off` run must show neither.

---

## 11. Proposed implementation tickets (the lead files them after review)

Each ticket names the RED that must fail before its code exists (*lead ruling
§T1*, I10). "GPU" marks work the lead runs. The lead also runs every test
labelled `gpu-serial`. **The rule** (r31 m-2): every th32 test that constructs a
`unified_cache`, calls `unified_allocate_owner` or `unified_alloc` for any tier,
initialises SYCL, or loads a model is lead-run and `gpu-serial`, and one that
loads a model also carries the `model` label, which form 2's `-LE` excludes. A
`HOST_CPU` request is no exception: it names a device, `unified_allocate_owner`
refuses a negative one and takes that device's coordinator (`d8a67422d
unified-cache.cpp:16116-16128`), and `unified_alloc` resolves the device's queue
(`:15115-15116`). th32 adds no seam to keep any of them off the device. **The
members**, beyond the arms marked GPU: loading a model, so also `model`: T6 (a),
which runs the real dual-card configuration, whose cards report `mmap_support =
false` until 5efe restores it, not a stub device; T6 (e), (i) and (q), which
decode (e)'s gemma4-shaped fixture; T6b's load arm; and T7 (1)-(7). Loading no
model: T1's host test and census test; T4's activation-check and yield fixtures,
whose row-cache chunks are `HOST_CPU`; each T5 arm that allocates a real
`HOST_CPU` landing slot or block ring rather than the mock engine's memory;
every fixture of T2 and T3, which run the planner on a shaped inventory or
construct a cache (r30 m-2); T6 (n); and T7 (9) (r29 m-3; r30 n-1).

| # | ticket | depends on | RED (fails first) | notes |
|---|---|---|---|---|
| T1 | `alloc_tier::HOST_CPU`: raw wrapper (`mmap` + `MADV_HUGEPAGE` + `mlock`), `must_host_cpu`, `device_mappable()`, the 36-site tier census (§3.1); `lazy_residency`; `cache_location`/`expert_resolve_tier` gain reported `STORAGE`; roles `LAZY_ROW_CACHE`/`LAZY_STAGING`/`LAZY_OUTPUT`; canonical contract §3/§4/§12.5 edits, including the allowlist line for T7's native Level Zero event-pool site, next to the `HOST_CPU` wrapper (§3.3, r5 m-15) | - | a host test that requests `HOST_CPU` under `RLIMIT_MEMLOCK` = 1 page and expects the typed refusal `host-cpu-lock-failed` naming `errno` and the bytes; a census test that a `HOST_CPU` allocation appears in the registry and the shutdown census | header change first: every other ticket includes it |
| T2 | the process-global host ledger: one budget function, pools capped by their sum, `NO_PINNED` and `RESERVE_MB` semantics, `host_planned_bytes` per §4.2, the plan's host fields, `host-budget-exceeded`, `[HOST-PLAN-BUG]` | phase (a): T1 and **`llama.cpp-rnh3`**, computed through the functions `llama.cpp-7v5c` runs; **not** sgc5, moua or zhcn (*lead ruling §D6*, I-3). Phase (b) adds **`llama.cpp-sgc5`** (one zone set per plan), `llama.cpp-moua` (the shared publish struct and lifetime order), `llama.cpp-zhcn` (the `SYCL_Host` compute buffer as a planned tenant) and `llama.cpp-s4ip` (lever (s), for the regression gates' expected bytes) | **phase (a), WARN-only** (*lead ruling §T6*, m-2): a planner fixture whose host weights exceed the budget expects the load to proceed and one would-refuse WARN line carrying each term and the plan's number, and no refusal; a Q2_K-shaped fixture expects `host_planned_bytes`' zone terms to equal what `populate_host_zone_sizing` and the `configure_host_zones` sizing return for the same inventory, the expected bytes **derived from those functions, never a literal** (*lead ruling §D5*): with the ring, `dma_staging_pool` and the RUNTIME pool included, `cpu_quant_buffer_bytes` excluded, and `pp_moe_onednn_scratch_bytes == 0` and `dma_staging_pool` absent from host SCRATCH, the RUNTIME pool and the device once `s4ip` lands; and, because that equality can only fail by a term's absence (m-3), a **plan-against-committed mutant**: a fixture that commits one pool chunk the plan does not count must give `[HOST-PLAN-BUG]` naming both numbers (§10's ledger check). **Phase (b), enforcing:** the same over-budget fixture expects `host-budget-exceeded` with each term, not the WARN; a `PER_DEVICE` fixture with two caches expects their pools capped by the **sum**; a fixture whose pre-allocation runs short expects the refusal, not the WARN (§4.2) | **changes behaviour for every model.** Lands in two phases (*lead ruling §D6*, I-3): (a) WARN-only, the single ledger logging what it would refuse, now (*lead ruling §T3*, (c)), and all `9g22`'s phase-1 gates (its T13) need; (b) enforcing, with the sum cap, only after the regression gates pass: GPT-OSS 120B (host experts, `level_zero:0`), Qwen3.5-35B-A3B on the B50, the `glkg` 1% VRAM fixture, and the Mistral gate, all still admitted (GPU). Every fixture here, phase (a)'s over-budget and `host_planned_bytes` fixtures and phase (b)'s `PER_DEVICE` two-cache fixture included, is lead-run and GPU-serial, and loads no model: `host_planned_bytes` is 7v5c's computation, whose `compute_multi_device_plan` shaped path calls `planner_default_device_layout` and so `get_optimal` (`d8a67422d unified-cache.cpp:29507`), `configure_host_zones` is a cache member (`:20472`), and a two-cache fixture constructs two caches, each with its queue and pinned pool (r30 m-2) |
| T3 | `placement_lazy_table_info` inventory class, excluded from `max_tensor` and every inventory maximum; the residency decision (§4.3); the plan lines (§10) | T2, **`llama.cpp-7v5c`** (the joint placement is its computation, §4.3 step 5) and, for step 5's cold set, **`llama.cpp-yl3v`** | a Q2_K-shaped inventory fixture with its two tables asserts STAGING and SCRATCH equal to `populate_host_zone_sizing()`'s result for the same inventory without the tables (derived, not a literal; a counted table adds at least twice its bytes, §4.1). The residency decision is driven by a synthetic `MemAvailable_pre`, not a budget, and must take **both branches** (*lead ruling §T6*, m-5): the DeepSeek fixture at the lowest §4.5 reading gives both tables STORAGE, and at a synthetic `MemAvailable_pre` 1 GiB above the fixture's own one-table threshold (`expected_host - 2 + one table + 29`, computed from its inventory as the Qwen pair is, never §4.5's hand figure; r5 m-11), which the fixture first asserts lies below its own both-tables threshold (otherwise both tables go HOST), gives `blk.1` HOST (the tie-break of §4.3 step 4: ascending `consumer_layer`) and `blk.14` STORAGE; a Qwen flip pair at `MemAvailable_pre` = the fixture's own P3 threshold (`expected_host + 5 + 24`, computed by the fixture from its inventory, never taken from §4.5's hand band) minus and plus 1 GiB gives STORAGE and HOST. A stub that never returns HOST fails the second and third (r4 m-8) | every fixture here runs the planner on a shaped inventory, `expected_host` included, which is 7v5c's computation, so it is lead-run and GPU-serial, and loads no model (§11; r30 m-2) |
| T4 | the `lazy_table` object: refcount, descriptor buffer type, with its own interface, a null `init_tensor` and aborting data calls, the descriptor guard at `ggml_sycl_get_data_ptr_slow`'s entry, and `ggml_sycl_is_device_vram_buffer`'s descriptor test before its cast, with `ggml_backend_sycl_buffer_is_lazy` forward-declared ahead of it, which changes no other buffer type's answer (§3.2; r26 M-2, r27 n-4, r28 n-2), HOST body load (eager, locked, D1), STORAGE fd, row cache (CLOCK, sharded, 13.3 B/slot index, 256 MiB yieldable chunks), the activation check | T1, T3 | an activation-check mismatch fixture (one extra chunk) must refuse naming both numbers; a yield fixture must free a chunk with no wait while rows remain gatherable; host-only, a `WEIGHTS` tensor bound to a host-form descriptor keeps `tensor->extra` null, and, passed to `ggml_sycl_get_data_ptr_slow` in its own process, prints exactly one `[LAZY-PLAN-BUG] data pointer requested for lazy table` line and exits 134, where the RED, the guard absent, prints none (r26 M-2) | |
| T5 | the IO engine: pread pool, the ordering of §5.2.1 (classes, the `(epoch, deadline)` key, per-context round-robin, chunking, promotion) and its dispatch log, in-flight dedup with joins that take the more urgent key within a context and add a second queue entry across contexts (§5.2.1, *§T12* I-C, r7 m-7), FIFO order within an equal key by enqueue sequence, the test-only FIFO ordering override (§5.2.1; `9g22` r9 m10 (b)), tickets with epochs, gather requests with their cancel flags, landing slabs, the pool's block ring sized from `row_size`, the per-`(table, context)` strands on their own stage executor, the strand's retire of a cancelled epoch, the request's single resume claim (§5.3, *§D12h* m-6), and the marker watcher with its event-pool references (§5.3), references on every read, join and request, the watchdog heartbeat on class-0 read completions (§5.3), and the read-fault seam of T6 (p) and (q) (r24 I-2, M-7): `GGML_SYCL_LAZY_READ_FAULT=<k>`, a production diagnostic variable on *lead ruling §M65(b)*'s terms: off by default, read once through the accessor pattern of `GGML_SYCL_LAZY_GATHER_ARM` (§8.1), documented in `docs/backend/sycl-env-vars.md` as fault injection for tests, and announced by one WARN at init whenever it is set, naming its value, so a stray setting is never silent; set, it completes every class-0 read of the `k`-th ubatch's epoch on the lazy-plan backend (counted as `ubatch_begin_calls`) with `EIO` instead of issuing it and prints one `[LAZY-READ-FAULT] injected` WARN per read naming the table, and its mock-engine test hook, which does the same for a named epoch | T4 | the cancelled-epoch fixture of §5.3 (r5 I-3): **B's completions released first, then A's**, slot 0 holds B's rows exactly and A's appear only in the cache; positive control, the pre-registered mutant without the reaper's epoch and cancel check, shows A's rows in slot 0. **The single-signal RED** (*§D12h* m-6, *§D12m* 11): cancel epoch `e` with its reads held past `prefetch(e+D)`'s takeover of the entry, hold stage 2 of `e+D` with a stage hook, as T7's retire arm does, then release `e`'s reads while that hold stays. Its engagement check is pre-registered: event `e+D` is unsignalled immediately before the release; if it is not, the arm is VOID, which counts as failed. Pre-registered: event `e+D` stays unsignalled until stage 2 of `e+D` is released and signals it; the mutant whose reaper resumes the stage on the last read regardless of the claim signals it early, at the release, while stage 2 of `e+D` is still held. **The join-source RED** (*§D12m* 10, r11 m-3), on the mock with a one-slot block ring: context B joins a read of block X that context A's ticket dispatched, a test hook holds the reaper's copy into B's landing slot after A's copy has completed, and a new request is submitted for block Y, a different block whose bytes differ from X's in every row B reads (pre-registered from the fixture file). Its engagement check is pre-registered: the new read was submitted while B's copy was held; if it was not, the arm is VOID, which counts as failed. The hook releases B's copy on an observed event, never on a timer, under which the mutant could race and pass: when the new read has completed, or when it is observed waiting for the slot, whichever comes first, within a 5 s bound past which the arm is VOID. Pre-registered: the new read is observed waiting for the slot, and B's rows equal block X's; the mutant that drops the read's slot reference at the owner's delivery lets the new read complete into the slot, so B's rows equal block Y's. The three ordering witnesses of §5.2.1, scored on the dispatch log: the head-of-line backlog shape (P = 12000; the decode ticket within its no-backlog latency plus one QD64 read's p99, both measured in the run, r5 m-5; its positive control is the FIFO override, under which the ticket takes the backlog's time), the **separating RED** (every `blk.14` read dispatched after every queued layer-2-13 chunk and before the still-queued layer-20 chunks; it fails under "cold reads lowest", *lead ruling §T8* I-6), and the **epoch RED** (epoch `e`'s `blk.14` reads dispatched before epoch `e+1`'s `blk.1` reads; it fails under a deadline-only key, *§T10* I-4), and the **shared-block RED**, across two contexts and across two epochs (witness 4; it fails under a join that keeps the queued read's position, *§T12* I-C; the across-epochs arm puts Y last in epoch `e+1`'s ticket and also requires Y before epoch `e`'s last read, r7 m-6). The teardown fixtures of §5.3: context A freed with a read in flight that B joined (B's rows arrive, no use-after-free under ASAN or the mock's poisoned slabs, release observed after the read, *§T8* I-7), and the event arm (the context freed between a signal and its copy's execution; the event pool destroyed only after the copy's marker is observed, *§T10* I-7) | CPU-only |
| T6a | th32's DLM (a) eligibility predicate, exposed as one function of a lazy mode, the entry's size and an mmap-supported flag: never eligible when the flag is false, and otherwise `OFF` is never eligible, `AUTO` only above 4 GiB, `ON` always (§4.1, *lead ruling §F2a* I-9 addenda 3 and 5). 5efe's C1 pre-fill calls it once per entry to set the entry's `lazy_eligible` bit, passing, when the SYCL branch is the taker (`sycl_model_loading_guard.active`, captured by C1), the pre-fallback mode C1 captures and `true`, and otherwise `lazy_read`'s resolved mode and `llama_mmap::SUPPORTED`; T6's SYCL branch and `lazy_read::add()` read the entry's `deferral` and never call it (*§F2a* I-9 addenda 4 and 5). So a taker load on a platform without mmap passes `true` and is eligible by the flag-true rows (addendum 6 (1)); the caller's choice is pinned by C1's source clause (§4.1; r22 M-1), since `SUPPORTED` is constant on the hosts this project runs | `llama.cpp-n77l`; it lands **no later than `llama.cpp-5efe`'s C1**, whose pre-fill calls it (a data edge, *§F2a* I-9 addendum 3) | a host-only table over `OFF`, `AUTO` and `ON` at 4 GiB and at 4 GiB + 1 byte, with the mmap flag true: only `AUTO` at 4 GiB + 1 and both `ON` rows are eligible; and the same six rows with the flag false, none eligible; its RED is a predicate returning `mode != OFF`, which fails the `AUTO` row at 4 GiB and every flag-false row but the two `OFF` ones | the one eligibility source; no consumer re-evaluates it |
| T6 | llama side: loader SYCL branch for `TENSOR_READ_LAZY` (independent of `mmap_support`), taking a table exactly when its record entry's `deferral` is `STORAGE`, with no taker test of its own and never re-evaluating T6a's predicate (the pre-fallback mode and the taker, `sycl_model_loading_guard.active`, are captured by 5efe's C1, *§F2a* I-9 addenda 4 and 5), the refusal of a `deferral == STORAGE` entry that has no `rows_per_token` or `consumer_layer` (§9.1 item 1, addendum 5), T6's read of the attributes from the per-model record entry, whose qwen4exp and gemma4 values C1 lands in `llama_model_lazy_tensor_attrs` (5efe rev 3l `a68508606` `docs/design/sycl-mmap-host-tier.md:600-610`, qwen4exp's `consumer_layer` from `hparams.is_ple_impl`; §9.1, addenda 7(a) and 8(a), *lead ruling §M58c*; r23 M-4, r24 N-1), the executor query `ggml_backend_sycl_weight_executor(name, buft)`, whose plan read is one body with supports_op's, and its layer reduction `llama_model_sycl_layer_executor`, which `llama.cpp-30h4`'s `dev_layer` sync is changed to call, neither gated by `GGML_SYCL_DEV_LAYER_SYNC` (§3.2, *lead ruling §M58a*; r23 I-1), with `ggml_backend_sycl_planned_target_device` made a second entry point onto the query's body, both name-keyed entry points answering `NO_PLAN` for a name the plan splits into per-expert entries, and the reduction skipping the same names, through one exported predicate, `ggml_backend_sycl_weight_is_split_expert(name)`, whose classifier, `infer_tensor_usage` only and never the usage registry, both planners' split gates also call, with no name substring anywhere and a source pin that forbids any `MOE_EXPERT_WEIGHT` token, comments stripped, in the `placement_tensor_info` bodies of `compute_placement_plan` and `compute_multi_device_plan`, and the planner's stale `_exps` comment rewritten (§3.2; r24 M-1, r25 I-1, r26 I-1, r27 m-2, m-3, r28 n-1), the lazy tensor created in T6's own `lazy_ctx`, counted in `n_created`, with an identity-only registration and never `register_host_weight` (r25 I-2), and its descriptor bound after the late replan and before the buffers are made, with `lazy_ctx` in `ctx_buf_maps` (§3.2; r23 M-1, r24 M-3), and on the `no_alloc` pass the host form bound as a null-based zero-size dummy with no query and no `NO_PLAN` refusal (§3.2; r24 I-1), `ggml_backend_sycl_lazy_host_buffer_type()` and, for a CPU-executed consumer, the CPU gather with its own lifecycle: classification in `prefetch` from the rows input, one landing slot, the op's carrier check and ticket wait on ggml-cpu's thread 0, the release stores at the end point, backstop, catch step and teardown, the wrap of the context's CPU abort callback, and `process_ubatch`'s mapping of a lazy failure's `GGML_STATUS_ABORTED` to `GGML_STATUS_FAILED` (§5.3's CPU form, §7.1, *§F2a* I-9 addendum 9, *§M58b*; r23 I-2, r24 I-2, M-7), the inventory's byte ranges taken by value from 5efe's per-file lazy-range record, never derived independently (from `llama_tensor_weight::offs` or `ggml_nbytes`) (§4.1, *§F2a* I-9 addendum 2; r19 N-1), and the load refusal `lazy-range-missing`, the not-found branch of the SYCL branch's own range lookup, naming the GGUF tensor and its `weights_map` file index, unreachable on GREEN behind the drift check (§4.1; r19 M-2, r20 M-1), `build_lazy_get_rows` + arm C's CPU custom op + `lazy_expand_for_layer` + `lazy_prefetch()` + `lazy_cancel_unenqueued()` on `process_ubatch`'s failure path and from the end-point guard's destructor when a throw unwinds past it (r15 m-4), the backstop behind T7's backend cancel step (§5.3, *§D12*) + the lazy teardown as `~llama_context`'s first statement, before its `synchronize()` (§5.3, *lead ruling §T10* I-7) + the post-build consumer scan, which also refuses a graph built for a ubatch that gathers a lazy table twice, omits one its builder declares, or gathers one its builder does not declare, naming the table, and which counts a CPU gather op, through its user data, as its table's gather (r23 I-2 (b)) (§5.3, *§D12p*, *§D12q* I-2), gemma4's const-row input class; the qwen4exp and gemma4 builders' lists of the lazy tables each graph gathers, which the post-build scan checks (§5.3, *§D12q* I-2; these are not 5efe's name-list entries); the guard-input declaration on the context transaction, the graph inputs llama writes through `ggml_backend_tensor_set` (`inp_tokens`, `inp_embd`, `inp_pos`; §5.3, r7 I-3, *§T9e*); beside it, the **ubatch delimiter**, one call per ubatch into `ggml_backend_sycl_ubatch_begin(backend, k, guard inputs)` on every SYCL backend the scheduler holds, in scheduler order, made in `process_ubatch` after `set_inputs()` and before `graph_compute()`, which runs every family's witness, fails the ubatch with -3 before `graph_compute()` when a witness fails, whatever the strict switch, and of which `lazy_prefetch()` is the consumer, with the `lazy-hook-unresolved` refusal (§5.3, *lead rulings §D12g* I-1, *§D12m* 2, 3); the delimiter's **end point**, an RAII guard constructed in `process_ubatch` immediately before the `ubatch_begin` calls, whose destructor calls `ggml_backend_sycl_ubatch_end(backend)` on every SYCL backend it began, through the same hook table, on the failure and the success returns alike, and which is on no other path (§5.3, *§D12p* option (b), *§D12s*); the refusal of `llama_opt_init` by name on a context with a lazy or cold plan, an ERROR line `lazy-plan-training-unsupported` and then `GGML_ABORT`, since the function returns `void` (*§D12m* 1, *§D12n* 4); the **context-serial registry**, a hash map under one mutex held across each lookup and flag copy and taken by the insert and the erase, whose entry the first context-transaction run creates with the immutable plan flag, whose later runs refuse a changed flag as `[LAZY-PLAN-BUG] plan flag changed`, and which teardown removes first, with the transaction hook that stores the context serial on each SYCL backend (§5.3, *§D12m* 5, r11 m-6, m-9); and the **context allocation scope**, `thread_local` and RAII-nested, opened inside `graph_reserve()` and at every `ggml_backend_sched_alloc_graph` site on the context's scheduler (`src/llama-context.cpp:2750`, `llama-kv-cache.cpp:1002`), through which every host compute buffer the context allocates is stamped with (context serial, allocation serial) (§5.3, *§D12h* I-3, *§D12m* 4) | T4, T5, T6a, `llama.cpp-n77l`; `llama.cpp-5efe`'s per-file lazy-range record, its `lazy_eligible` bit and `deferral`, the pre-fallback mode and taker captures, the per-arch attribute function and `lazy_read::add()`'s read of `deferral`, landed in 5efe's C1, which T6's inventory and branch read (*§F2a* I-9, addendum 4); **T6 is built on C1, and the merge unit is 5efe C1-C4 plus th32 T6 and T7, which reach master in one merge** (*§F2a* I-9 addendum 8 (b), addendum 11 (3), *lead ruling §M56(b)*; 5efe rev 3l `a68508606` `docs/design/sycl-mmap-host-tier.md:805-810`; r26 M-4): C1 alone writes `STORAGE` with no consumer, so a taker Qwen load would leave the table out of the early plan and then load it eagerly, 51880 MiB pinned and uncounted; C4 travels in the same train (addendum 11 (3)), and no th32 arm assumes that any part of the unit reaches master alone; task/th32 carries or merges task/5efe's C1; there is **no** tracker edge blocking C1's merge on T6, because T6 depends on C1 and that edge would close a cycle, so the one-merge constraint is recorded as comments on `llama.cpp-5efe` (`c-q50c`) and `llama.cpp-th32` (`c-0elm`); otherwise `llama.cpp-5efe` is an **order edge only**, since T6's branch reads neither the SYCL device's `mmap_support` nor 5efe's mapping, and 5efe owns §4.1's interface requirement (*§T13*, *§T13a*, *§T13c*) | (a) with `AUTO` and a device reporting `mmap_support = false`, the dual-card config still prints `[PLACEMENT] lazy ... residency=`; (b) a graph with a `ggml_view_1d` of a lazy tensor fails the build naming it, and so do a graph that gathers one lazy table twice, a graph that omits a declared table, and a graph whose builder gathers a table it does not declare (*§D12p*, *§D12q* I-2, r14 m-5); (c) the guard declaration (*lead ruling §T9e*) of a gemma4 and of a qwen4exp context names `inp_pos` and one of `inp_tokens` or `inp_embd`, and never `KQ_mask` or `out_ids`, which llama writes through `->data`; (d) with two counting stub SYCL backends in one context, `ggml_backend_sycl_ubatch_begin` is called exactly once per ubatch on each, in scheduler order, before the ubatch's first `graph_compute`, on a reused graph and on a rebuilt one, with `k` strictly increasing, and `ubatch_begin_calls` equals the ubatch count on each, and `ubatch_end_calls` equals `ubatch_begin_calls` on each, including a ubatch whose `graph_compute` a stub fails (*§D12m* 3, r14 m-3); (e) `llama_opt_init` on the lazy fixture, in its own process, prints `lazy-plan-training-unsupported` and aborts, scored on rc 134 plus the line, with the process on the synchronous default log sinks, llama's for this `LLAMA_LOG_ERROR` line (`src/llama-impl.cpp:62-67`) and ggml's for the backend's `GGML_LOG_ERROR` lines (`ggml/src/ggml.c:313-318`), each of which writes and flushes `stderr` per line, with no callback installed through `llama_log_set` or `ggml_log_set` (`llama_log_set` sets both, `src/llama-impl.cpp:32-36`), since a custom asynchronous callback loses lines the same way, and no `common_init()`, whose asynchronous `common_log` worker can lose the lines queued before an abort (`common/log.cpp:374-378`; r13 m-3); its RED is the refusal removed, run after at least one decode, on the gemma4-shaped lazy fixture, with `idata_split` = 0 and a param filter that admits only `output_norm.weight`, a non-lazy F32 tensor applied after the last layer (*§D12r*): `ggml_opt_alloc` builds the backward graph on its first call whatever `idata_split` is (`ggml/src/ggml-opt.cpp:448-456`, `:489-490`), and that build aborts on an op with no backward case, `MUL_MAT_ID` among them (`ggml/src/ggml.c:7218-7220`), before any `graph_compute`; every `MUL_MAT_ID` and arm C's `GGML_OP_CUSTOM` sit upstream of `output_norm.weight`, so `ggml_build_backward_expand` marks none of them as needing a gradient (`ggml.c:7398-7406`) and `ggml_compute_backward` returns early for each (`:6749-6751`); the property the choice rests on is that **every op downstream of `output_norm.weight` has a backward case** (r13 m-5), and on this fixture that set is the norm's `MUL`, the lm-head `MUL_MAT`, the `MUL` by `output_s` when the model carries one (`src/llama-graph.cpp:1520-1522`), a `GET_ROWS` only when unmasked `embeddings_nextn` is requested (`src/models/gemma4.cpp:395-397`), which the RED's context does not request (the `out_ids` crop is inside the last layer, upstream of the norm, `:251-256`), `CROSS_ENTROPY_LOSS`, and the `SCALE` that `ggml_opt` appends when `opt_period` > 1 (`src/llama-context.cpp:4698`, `ggml/src/ggml-opt.cpp:408-411`; its backward case is at `ggml.c:6945`), while the `ARGMAX` prediction is `I32` and is skipped (`ggml.c:7367`); the fixture carries no `final_logit_softcapping` (an optional key, `src/models/gemma4.cpp:20`, `:405-409`), whose `ggml_tanh` has no backward case; its engagement check is pre-registered: `graph_compute_calls` rises during `llama_opt_epoch`, or the arm is VOID, which counts as failed; pre-registered: `ubatch_begin_calls` and `lazy_witness_checks` stay at their post-decode values, and the first training graph's gather prints the carrier's `[LAZY-PLAN-BUG] ubatch delimiter missing`, naming the **k branch**, since the training graph runs after the last decode's end point, so its gather finds its posted `k` at `none` (§5.3; *§D12m* 1, *§D12n* 2, *§D12s*, r11 m-1); its placement mutant is pre-registered: the end point moved into `graph_compute` as a re-arm at entry (posted `k` := the record's `k`, `consumed` cleared) makes the training graph pass its gather with no line, so the RED goes silent (*§D12s*); (f) the scope is `thread_local`: a stub allocator called from a second thread while the first thread holds a scope reads no context serial, and a nested scope restores the outer one on exit; its RED is a process-global scope, under which the second thread reads the first thread's context serial (*§D12m* 4), and the nesting half's RED is a scope whose exit resets the ambient to none, under which the outer scope's serial reads as none once the inner scope has exited (r11 m-2); (g) with a SYCL registration whose proc address for `ggml_backend_sycl_ubatch_begin` returns null, and in a second run one whose `ggml_backend_sycl_ubatch_end` returns null, creating a context with a lazy plan is refused by name, `lazy-hook-unresolved`, naming the hook (r14 m-3), and a context with no plan is created as today (r11 m-4); (h) a host-only unit arm: a transaction stub that computes a different plan flag on its second run prints `[LAZY-PLAN-BUG] plan flag changed` and refuses the context, and the same stub computing the same flag creates it (r12 m-6); (i) a throw that unwinds past the guard, on (e)'s gemma4-shaped lazy fixture, whose one lazy table is `per_layer_tok_embd` (§9.1): a one-shot test hook, which disarms itself before it throws so that the `D + 1` follow-up decodes run unhooked (r17 nit-1), armed by `process_ubatch` around its own `graph_compute` call (`src/llama-context.cpp:2767`) so that the K-shift's call (`llama-kv-cache.cpp:1009`), which runs before the guard exists, never throws, throws `std::runtime_error` from `llama_context::graph_compute` after `lazy_prefetch()` and before its scheduler call (`src/llama-context.cpp:3876`), and the test catches it around `llama_decode`, as the server does. A throw skips `decode()`'s rollback of the ubatch's cells (`:3200-3221`), which `mctx->apply()` wrote before the guard existed (`:2708`), so after the catch the test calls `llama_memory_seq_rm(mem, seq, pos_min, -1)` for the failed ubatch's lowest position, mirroring that rollback; otherwise a re-decode at the same explicit positions fails batch init (`src/llama-batch.cpp:295-310`) with the backstop working. Its engagement check is that the test caught the hook's exception **and** `ubatch_begin_calls` rose by one on every SYCL backend during the throwing `llama_decode`, the counters being per backend (§10; r17 nit-2), so the guard was built, or the arm is VOID, which counts as failed; pre-registered: `lazy_epochs_cancelled{backstop}` rises by 1, `ubatch_end_calls` equals `ubatch_begin_calls`, and the next `D + 1` decodes succeed with `epoch_ring_exhausted == 0`; its RED is a guard that only clears, under which `lazy_epochs_cancelled{backstop}` is unmoved (r15 m-4, r16 m-1); (j) the inventory's byte range comes from the record (r18 M-2, *§F2a* I-9), on a two-file variant of (e)'s gemma4-shaped fixture that puts `per_layer_tok_embd` in the second shard, so its file index is 1 and a hard-coded 0 fails (r19 N-3), with a device reporting `mmap_support = false`, as in (a), and the table below `AUTO`'s 4 GiB floor, in two runs. Run 1, under `AUTO`, which upstream forces to `OFF` (`694ec2354 src/llama-model.cpp:1520-1528`), asserts that the record holds `per_layer_tok_embd`'s entry, equal to what the fixture wrote (file index 1, the shard's offset, the table's size), with `lazy_eligible` clear, **and** that the SYCL branch takes no inventory entry, since under DLM (a) `AUTO` makes only a table above 4 GiB eligible (r19 N-2); it discriminates two of upstream `lazy_read::add()`'s three gates, the mode `OFF` and the `AUTO` floor (`fefb92980 src/llama-model-loader.cpp:1241-1249`), and not the third, because `llama_mmap::SUPPORTED` is a constant `true` wherever `_POSIX_MEMLOCK_RANGE` (or `_WIN32`) is defined, Linux included (`fefb92980 src/llama-mmap.cpp:710-714`; r19 N-2, r20 N-2); on the merge unit's tree, which carries 5efe's C1-C4 (T6), it is also the witness that the early plan places the table as an ordinary weight, an assertion 5efe owns (r19 M-1). Run 2, under `ON`, asserts that the inventory entry's (file index, offset, size) is non-empty and equals the record's entry and the fixture's, with `lazy_eligible` set. Its RED is an inventory that looks the range up in `lazy_read`'s set, which lacks the table on the SYCL branch, so the branch's range lookup takes its not-found branch (§4.1) and run 2 stops on the pre-registered line `lazy-range-missing` naming the GGUF tensor `per_layer_token_embd.weight` (the member `per_layer_tok_embd`, `fefb92980 src/models/gemma4.cpp:58`) and file index 1 (r19 M-2). A source-contract clause, scoped to the SYCL lazy branch and not to 5efe's record check beside it in `create_tensor`, which legitimately reads `w->offs` (r19 N-3), pins that the branch takes the inventory entry's range fields by value from the record's accessor and reads neither `llama_tensor_weight::offs` nor `ggml_nbytes` for the range; its two mutants, a branch that fills the offset from `w->offs` and one that fills the size from `ggml_nbytes`, each fail the clause; (k) the bit's mode input and the taker (*§F2a* I-9 addenda 4 and 5; r20 I-1, r21 I-1, M-1), host-only and scored at the early plan, before any tensor data is read, on (e)'s gemma4 shape, whose attributes are the constants 1 and 0, so T6's missing-attribute refusal cannot fire on GREEN (r22 M-2 (a)), with `per_layer_token_embd.weight` a sparse region above 4 GiB (4 GiB plus one row), under `AUTO`, with a stub device reporting `mmap_support = false` in the model's device list in every run, so that upstream's fallback (`fefb92980 src/llama-model.cpp:2256-2265`) resolves `lazy_read`'s mode to `OFF`. The host-only mechanism is a stub backend registration named `"SYCL"` that owns that device, exports all ten lifecycle hooks (`:79-122`), returns `GGML_SYCL_LIFECYCLE_OK` from `begin` and `end` (the guard's constructor throws otherwise, `:158-162`) and captures the inventory handed to its `stage_inventory` hook (`llama_model_sycl_apply_inventory`, `:592`, called from the early plan at `:658`); fork precedent for fixture registrations is `tests/test-sycl-lifecycle-runtime-wrapper.cpp:112-236`. `llama_model_sycl_hooks()` binds the first device whose registration is named `"SYCL"` (`:79-87`), so the stub must be the only such registration that enumerates a device: `params.devices` holds the stub device alone, and the process either does not load the real SYCL module or runs it with a device selector that enumerates none (r22 M-2 (c)). Engagement, pre-registered for every run, or the run is VOID, which counts as failed: `lazy_read`'s resolved mode is `OFF` after the fallback, read from C1's per-model line through a test-installed `llama_log_set` callback, which receives INFO lines (only `common_log` drops them) (r22 M-2 (b)). 5efe's C1 prints that line as a separate INFO line per model, `lazy-range record: mode_pre=<m> mode_post=<m> taker=<t>`, with `<t>` 0 or 1, for every model whether or not a name is declared (5efe rev 3l `a68508606` `docs/design/sycl-mmap-host-tier.md:630-635`, witness cell `:3368-3372`; *lead ruling §M58c*). It shares the `lazy-range record:` prefix with the per-row line (`:642-647`), whose `mode=` is the mode the bit read, `mode_pre` for the taker, and whose verbatim form 5efe's W1-W3 pre-register (`:3468-3499`), so (k)'s scorer keys on `mode_pre=`, never on the prefix alone (r24 M-8). Pre-registered per run: run 1 reads `mode_pre=auto mode_post=off taker=1`, 5efe's own witness cell, and runs 2 and 3 read `mode_pre=auto mode_post=off taker=0`. Until C1's code prints it, every (k) run is VOID, which counts as failed (r23 M-3). Run 1, the taker: the stub SYCL registration and `n_gpu_layers` > 0, so `sycl_model_loading_guard.active` holds; engagement adds that `stage_inventory` is called once with `early` set, with the fixture's weight count minus one; pre-registered: the entry's `lazy_eligible` is set, its `deferral` is `STORAGE`, and the table is absent from the captured inventory, from `max_tensor` and from the host total; its stop point is pre-registered too: after capturing, the stub's `stage_inventory` returns non-OK, so `load_tensors` throws `SYCL inventory planning failed: early=1` (`:592-594`), before any tensor is created or read (r22 M-2 (d)). Run 2, no SYCL: the stub SYCL registration is not registered, and the same device comes from a stub registration of another name; pre-registered: the bit clear, `deferral` `EAGER`, no `stage_inventory` call. Runs 2 and 3 load with `no_alloc` set, under which the pre-fill still runs and nothing is read (r22 M-2 (d)). Run 3, SYCL registered but not the taker: the stub SYCL registration with `n_gpu_layers` = 0 and no SYCL buffer-type override, so the hooks resolve but the guard is inactive; pre-registered: the bit clear, `deferral` `EAGER` (so no descriptor buffer can be bound, read from the pre-fill's `deferral` under `no_alloc`) and no `stage_inventory` call. Its REDs: the pre-fill fed `lazy_read`'s post-fallback mode, which clears the bit in run 1, so the early plan places the table as an ordinary weight; the pre-fallback mode fed whatever the taker, which sets the bit in runs 2 and 3; and the process-level `llama_model_sycl_hooks_enabled()` as the taker, which sets the bit, and a `STORAGE` `deferral`, in run 3 only; (l) a CPU-executed consumer on a plan-tiered configuration (*§F2a* I-9 addendum 9, *lead rulings §M58a*, *§M58b*; r22 I-1, r23 I-1, I-2), GPU, run by the lead. **Vehicle** (*§M60(b)*; r24 M-5): `llama-completion -no-cnv -v` at the default `-ub 512`, with a fixed prompt, `--temp 0 --seed 42` and `-n 64`; `-no-cnv` because gemma4 E4B's template otherwise turns conversation mode on and the run dies in the chat template (the lead's first probe, void; probes 3-5 ran this way, `llama.cpp-wj27`), and `-v` because the counts and the context line below are INFO. The model is gemma4 E4B (`/models/stock-gemma-4-E4B-it.Q8_0.gguf`) on the B70 (`level_zero:0`) at `-ngl 99`, under `ON`, with `GGML_SYCL_VRAM_BUDGET_PCT=<p>` (`docs/backend/sycl-env-vars.md`), `GGML_SYCL_HOST_WEIGHT_LAYOUT=off` so that the tiered weights stay in `SYCL_Host`, the case in which the buffer type and the executor disagree (§3.2), and `GGML_SYCL_SUPPORTS_OP_DEBUG=1`; `-ngl` alone is not a vehicle (*§M58a*), which also retires r23 N-1's `-ngl 0` hole. Every run below is made twice, pinned at `-c 4096` and at the default context with no `-c` (§8.1). The default-context run's VOID guard (*§M61(c)*, *§M62(C)*) is the padded regex `llama_context: n_ctx +=  *<N>$`, not anchored at `^`, with `<N>` read from the same run's `print_info: n_ctx_train`; it must score exactly 1, and its negative control, the `n_ctx_seq` line, must score 0. **The calibration of `<p>`** (r24 M-4), which the lead records before the arm: the table is in the early inventory under `--lazy-mode off` (`src/llama-model.cpp:639-641`) and absent under `ON`, standalone entries are charged before any layer (`fefb92980 ggml/src/ggml-sycl/unified-cache.cpp:28123-28139`) and layers fill in ascending order (`:28107`, `:28141`), so layer 0 is the last dense layer to tier, and `ON` tiers it only at a lower `<p>` than `off`. The search starts at 50 and steps down by 5, running the arm's `ON` command and its `off` command with `-n 1` at each step, and records the highest `<p>` at which both print the engagement line below; where one step goes from no `blk.0.` decline to a load that aborts, it is refined by 1 inside that step. The floor is the load's own: a budget whose usable VRAM after the external headroom does not cover the mandatory arena aborts before planning (the `glkg` 1% B50 fixture, `.llm-wiki/wiki/sources/glkg-one-percent-vram-fixture-aborts-before-ordering.md`). If no `<p>` above the floor engages both commands, the arm is VOID, which counts as failed, and the lead escalates it rather than substituting another lever. Engagement, pre-registered for the `ON` run **and** the `off` run, or the run is VOID, which counts as failed: at least one `[SYCL-SUPPORTS-OP-DEBUG] decline` line (`fefb92980 ggml-sycl.cpp:107892-107915`, printed through `fprintf`, so at any verbosity) whose `src[i]` is a `blk.0.` weight, since the token comparison is valid only when both runs tier layer 0. The `off` run's validity for that comparison (r25 M-4): a `[SYCL-SUPPORTS-OP-DEBUG] decline` line with `op=GET_ROWS` whose `src[0]` is `per_layer_token_embd.weight`, which puts its gather on the CPU's `to_float`, and the two runs' plans agreeing (r26 M-1), read from each run's last `[PLACEMENT]` block, the settled plan's (`fefb92980 ggml/src/ggml-sycl/unified-cache.cpp:28383-28473`, INFO, printed at `-v`), which both runs must print: every per-layer line, `[PLACEMENT] layer <il> weight=... kv=... total=... dense_target=... kv_target=...` (`:28427-28433`), is equal byte for byte, in the same non-zero number and order, since the table has no layer and adds to none; every priority line (`:28401`), the `Totals:` line (`:28466`) and the `Total:` line (`:28471`) are equal except that, `off` minus `ON`, the `NORM/EMBED` line's `host=` count is exactly 1 more and its host MB, the `Totals:` host weight MB and the `Total:` host MB are each more by the table's `ggml_nbytes` / 2^20, to within 0.1 MB, the two `%.1f` roundings; the table's usage is `EMBEDDING` (`common.hpp:1924`), which the planner ranks `NORM_EMBED` (`unified-cache.cpp:24148-24150`), and under `ON` it is not a placement entry (§4.1), so its bytes appear only on the `[PLACEMENT] lazy` and `[HOST-PLAN]` lines (§10). The other `[PLACEMENT]` lines (the envelope, the zone reservation, the KV and MoE lines) are not scored. If the `GET_ROWS` check or the plan agreement fails, or either run prints no `[PLACEMENT] layer` line, the identity comparison below is VOID, not FAIL, and the answer alone decides. The host-weight registry (r25 I-2): each `[S1-PRELOAD] N host-registered weights found for preload` line of the `ON` run (INFO, at `-v`; `fefb92980 ggml-sycl.cpp:33568`) reads one less than the `off` run's matching line, where the table is an ordinary `SYCL_Host` weight; both runs must print it, or the check is VOID, which counts as failed. **Residency is pinned per sub-arm** (r24 M-6): (l-H) runs with no override and pre-registers `residency=HOST`, with §8.1's H launch precondition read from its own load-only plan line; (l-S) sets `GGML_SYCL_HOST_RESERVE_MB` by §8.1's S recipe from its own load-only plan line and pre-registers `residency=STORAGE`. Pre-registered for both: the `[PLACEMENT] lazy` line prints `consumer_layer=0 device=CPU` (the descriptor form's observable, §10); `lazy_cpu_gathers` equals the lazy-plan backend's `ubatch_begin_calls`, the vehicle's own ubatch count with its warmup; `lazy_output_copies == 0`; §8.1's CPU-form row for that residency; greedy output token-identical to the `--lazy-mode off` run of the same build at the same `<p>`, layout, vehicle and context, which isolates the lazy path, since both runs take the same routing and the gather dequantizes the same rows bitwise (§8.1); and the answer below, scored against the known-correct output, which does not pass through the changed `dev_layer` sync at all (r24 M-2). The **pre-change build** is the tree T6 is built on, before its 30h4 call. **The answer** (*lead ruling §M65(a)*): each command also runs with the prompt `1, 2, 3, 4, 5,` and `-n 24`, and the first line of its stdout, cut to its first 30 characters (`head -n 1`, then `cut -c1-30`), is scored with `grep -cx '1, 2, 3, 4, 5, 6, 7, 8, 9, 10,'`, which must print 1: an exact line, cut just after the comma that ends the tenth number, so that where the generation stops does not matter and a longer number cannot pass (r25 M-5; the 29-character cut passed `…9, 100`). The scorer suits gemma's tokenizer: its vocabulary has no token that holds an ASCII digit together with any other character (read from the GGUF's `tokenizer.ggml.tokens`), so each digit, each comma and each `▁` is one token, and the continuation ` 6, 7, 8, 9, 10,` is 16 of the 24 tokens; a correct run reaches the comma and scores 1. Its negative controls, run through the same scorer before any arm, must each print 0: the prompt alone (`echo '1, 2, 3, 4, 5,'`), a wrong count (`echo '1, 2, 3, 4, 5, 6, 7, 8, 9, 11,'`) and a longer number (`echo '1, 2, 3, 4, 5, 6, 7, 8, 9, 100'`). Its positive control is the pre-change build's `--lazy-mode off` run, which must print 1, or the prompt is unfit for the model and the arm is VOID, which counts as failed; that is a check of the prompt, not a comparison of tokens, and it is also what catches a model that stops at EOS right after `10`, which would score 0 on every arm rather than pass falsely. Its REDs are the two mutants this arm separates, because here `-ngl` and the buffer type both disagree with the plan: the executor read from `i_gpu_start` (at `-ngl 99` layer 0 is above it), and the executor read from the buffer type as 30h4 reads it (`SYCL_Host` means SYCL); each binds a SYCL descriptor, prints `device=` a SYCL device and gives `lazy_output_copies > 0`. Its sub-arm (l-D), the lifecycle's positive control (*§M58b*), in (l-H) and in (l-S): the same command with a one-ubatch prompt and `-n 8`, so that at least `D + 2` = 5 ubatches decode one token each; pre-registered: `lazy_cpu_gathers` equals `ubatch_begin_calls`, `epoch_ring_exhausted == 0` and rc 0, and in (l-S), where the slot exists, `rows_requested` rises in every ubatch and `cpu_slot_held == 0`; its RED is a CPU form that takes an epoch-ring entry nothing retires, which fails at `prefetch(D)`, the fourth ubatch, with `[LAZY-PLAN-BUG] epoch ring exhausted`; (l-sync) the gate on the 30h4 change (r24 M-2; `llama.cpp-wj27`), GPU, run by the lead: (l)'s model, card and vehicle (gemma4 E4B on the B70, `level_zero:0`, `-ngl 99`, `llama-completion -no-cnv -v`), `<p>` and layout, and both contexts, under `--lazy-mode off`, on the pre-change build and on T6's build, with `GGML_SYCL_SUPPORTS_OP_DEBUG=1` set in **both** builds' commands (r25 I-3, *lead ruling §M66*). Engagement, or the run is VOID, which counts as failed: each build prints (l)'s `blk.0.` decline, a `[SYCL-SUPPORTS-OP-DEBUG] decline` line whose `src[i]` is a `blk.0.` weight, and that line, counted non-empty on each build's own log, is also that build's trace-live check, since the variable is off by default and a run without it prints no decline at all; and T6's build's `llama_kv_cache: layer   0: dev =` line (DEBUG, printed at `-v`, `src/llama-kv-cache.cpp:320`) reads `CPU`, which shows that the sync took effect. The gate, pre-registered (*lead ruling §M65(a)*): T6's build's answer scores 1 with (l)'s scorer and controls; and T6's build prints no mixed-state line, counted with `grep -c 'is assigned to device CPU but .* is assigned to device SYCL'` on its log, which must print 0 (`src/llama-context.cpp:1255-1256`; the line disables fused attention for every layer). Its positive control is the stderr of the lead's `-ngl 1` probe, on which the same count prints 1; the scratch copy does not outlive the session, so the lead regenerates it on any build before T6 with `ONEAPI_DEVICE_SELECTOR=level_zero:0 llama-completion -no-cnv -m /models/stock-gemma-4-E4B-it.Q8_0.gguf -ngl 1 -c 4096 -p 'Hello' -n 4 --seed 42 --temp 0 -v` (probe 4's command without its `GGML_SCHED_DEBUG=1`, which does not gate the mixed-state WARN, `src/llama-context.cpp:1254-1258`; probe 4's stderr also counts 1) (r25 N-3, r26 N-2). Recorded and reported, not gated: whether the two builds' greedy outputs are token-identical, and each build's layer-0 KV `dev =` line. Placement moving layer 0's attention to the CPU (P3) can legitimately change the numerics, so a token difference with a correct answer is reported and does not fail the gate. A wrong answer or a mixed-state line is a FAIL, `llama.cpp-wj27`'s mixed state reached through the plan: it blocks T6's 30h4 call and goes to wj27, whose owner ruling decides the fix, and th32 does not design around it; (m) a planner-tiered consumer on a MoE model, GPU, run by the lead: Qwen3.8 (qwen4exp) on the B50 (`level_zero:1`), with (l)'s vehicle, layout and debug settings, its (m-H) and (m-S) sub-arms, both contexts with the VOID guard, and the calibration of `<p>`, whose engagement line here is a `blk.1.` weight for which the expert predicate (§3.2) does not hold, on both commands. Its consumer layer, 1 on this GGUF, must be CPU-executed through its **non-expert** weights: host experts do not make a layer CPU-executed, since the reduction skips every name the plan splits into per-expert entries (§3.2; r26 I-1). Engagement, pre-registered, or the run is VOID, which counts as failed (r23 M-2): that decline line. Pre-registered and RED as in (l), with `consumer_layer=1`. (m) is a 177 GiB Qwen3.8 load, so *lead ruling §M56(b)* binds it (r26 M-4): it runs only on a tree that contains 5efe C1-C4 and th32 T6 and T7, the merge unit above, and on any other tree it is VOID-precondition, not FAIL and not skipped-as-pass; what makes the model load on this host is 5efe's C4 (5efe §4.5), and the host-fit refusal is th32 T2's. It runs under §M56(b)'s watchdog, which samples `MemAvailable` every 5 s and aborts the arm as VOID-host if it falls below 40 GB (the lead's own baseline of this load reached 14 GB); (n) the executor query, host-only, CPU: a fixture `ml.ctx_map` and a stub SYCL registration whose hook answers the query, and whose `ggml_backend_sycl_weight_is_split_expert` hook answers as the fixture plan does (true for the split names below, false for the dense ones); pre-registered: a `blk.0.` weight in a `CPU_REPACK` bucket gives the CPU; a `SYCL_Host` weight the stub answers -1 gives the CPU; a `SYCL_Host` weight answered 0 gives SYCL device 0; a layer whose only host answer is the split `blk.0.ffn_down_exps.weight` gives the device; on the real load (the fixture's `no_alloc` clear), a `NO_PLAN` answer on a SYCL-owned weight of a lazy consumer layer is refused as `[LAZY-PLAN-BUG]`, naming it; a non-expert weight with a `layer_device` entry of 0 and no dense entry gets 0, the same answer, from `ggml_backend_sycl_weight_executor` and from `ggml_backend_sycl_planned_target_device` (r24 M-1; value pinned, r27 m-1); `planned_target_device` answers `NO_PLAN` for an `_exps` weight of a layer whose `layer_device` is -1 (r25 I-1); both entry points answer `NO_PLAN` for grovemoe's `blk.1.ffn_down_chexps.weight`, split into per-expert entries with no dense entry, in a layer whose `layer_device` is -1; `blk.1.ffn_down_exps.bias` and `blk.1.ffn_norm_exps.weight` (the real arctic name, `src/models/arctic.cpp:45`), each a dense entry of a layer whose `layer_device` and dense entries read 0, answer 0 from both entry points, the value master's `planned_target_device` returns for them; a layer whose only ggml-cpu-owned weight is a `_chexps` expert gives the device (r26 I-1); and every answer is the same with `GGML_SYCL_DEV_LAYER_SYNC=0`. Its REDs: the reduction reading `i_gpu_start` or `dev_layer[il].dev` fails the `SYCL_Host` cases, one reading the buffer type alone fails the `SYCL_Host`-answered-`-1` case, `planned_target_device` left on its own dense-entry read answers `NO_PLAN` for the `layer_device`-only weight, the entry without the expert skip answers -1 for the expert; and rev 3a's `_exps` substring skip answers -1 for the `_chexps` weight and `NO_PLAN` for the `.bias` and `ffn_norm_exps` weights, and 30h4's substring filter in the reduction gives the CPU for the `_chexps` layer (r26 I-1). **The real predicate** (r27 m-1, m-3; r28 m-1, m-2, n-3; r29 m-1, m-3), lead-run and GPU-serial, loading no model: a new target, `test-sycl-split-expert-predicate`, compiled with `GGML_SYCL_PRIVATE_TESTING=1` and linked against `ggml-sycl-private-fixtures`, never beside the ordinary `ggml-sycl` (`d8a67422d ggml/src/ggml-sycl/CMakeLists.txt:117-118`, `:5838-5887`, as `test-sycl-kv-view-extra-reuse` does at `:3104-3112`), so it builds under `BUILD_TESTING=ON` and `GGML_BACKEND_DL=OFF`, this host's configuration, at the cost of one link; registered as that sibling is (`:3113-3119`), with `LABELS "sycl;placement;gpu-serial"`, `RUN_SERIAL`, `SKIP_RETURN_CODE 77` and the pinned selector. It touches the device although it loads no model: the planner's layout choice calls `layout_policy::get_optimal(.., device_id)` (`unified-cache.cpp:24935-24961`, `common.hpp:1635-1638`), which calls `ggml_sycl_info()` and so `ggml_sycl_init()` (`ggml-sycl.cpp:25442-25450`), and the installer's publish constructs a `unified_cache`, with its queue and its pinned staging allocation, for every enumerated device (`ggml-sycl.cpp:15314-15340`; `unified-cache.cpp:14735-14783`, `:14530-14534`, `:4065-4067`). So the lead runs it (§11's preamble), and th32 adds no seam to keep it off the device (r29 m-3). Its plan installer is a sibling of `ggml_sycl::test_set_kv_placement_plan` inside the same `#if defined(GGML_SYCL_PRIVATE_TESTING)` block (`d8a67422d ggml-sycl.cpp:16156-16177`, declared at `ggml-sycl-test.hpp:476-485`), so no mutable plan hook ships in the production DSO (*§M74*); it publishes the plan as computed, through `ggml_sycl_publish_plan_locked` under `g_tensor_inventory_mutex`, the lock and plan `planned_target_device` reads. It does not reuse `test_set_kv_placement_plan`, which rewrites the plan's `kv_per_layer`, `planner_n_ctx` and `multi_device` before it publishes (`:16156-16161`), so what it installs is not the plan as computed (r29 n-1). The run, in order: (1) it opens a load transaction as the loader does, `begin_outer()` then `bind_candidate()` on the lifecycle registry (the pattern of `d8a67422d tests/test-sycl-weight-identity-multi-model.cpp:195-212`), with module admission `ACTIVE`, its default (`ggml-sycl.cpp:12230`, `:12234-12245`), since otherwise `ggml_backend_sycl_try_register_weight_usage` returns false (`:14050-14051`) or stores no row (`:14062-14075`); (2) it registers each fixture tensor's usage through that function with the loader's values (`fefb92980 src/llama-model-loader.cpp:1622-1633`), so `MOE_EXPERT_WEIGHT` for `blk.1.ffn_down_exps.bias`, `blk.1.ffn_norm_exps.weight` and `blk.2.ffn_down_exps.bias`, and `ATTENTION_WEIGHT` for `blk.0.attn_q.weight`; (3) **the read-back control**, before any case: `ggml_sycl_get_tensor_usage` (`e2461d4fb ggml-sycl.cpp:14103-14121`) on fixture tensors of those names, each carrying an extra whose `model_id` is the `begin_outer` token's model, so that its owner lookup lands on the bound load effect's owner (`d8a67422d ggml-sycl.cpp:2804-2809`), the owner registration keys each row by, slot and generation included (`:14071-14085`, `model-lifecycle.cpp:190-218`), reads `MOE_EXPERT_WEIGHT` for `blk.1.ffn_down_exps.bias` and `ATTENTION_WEIGHT` for `blk.0.attn_q.weight`, or the run is VOID, which counts as failed; (4) it computes the plan with experts by running the real `compute_placement_plan`, the `placement_tensor_info` overload (`d8a67422d unified-cache.cpp:27960`), with `n_experts > 0` on a fixture inventory of `blk.1.attn_q.weight`, the 3D `blk.1.ffn_down_exps.weight` and `blk.1.ffn_down_chexps.weight`, `blk.1.ffn_down_exps.bias` and `blk.1.ffn_norm_exps.weight`, under a budget that places every entry on device 0, installs it, and reads back layer 1's `layer_device` as 0, or the run is VOID; `blk.0.attn_q.weight` and `blk.2.ffn_down_exps.bias` are registered but not in the inventory, as a weight ir18 moved to `CPU_REPACK` is not. Pre-registered, on that plan: `blk.1.ffn_down_exps.weight` and `blk.1.ffn_down_chexps.weight`, split, answer true; `blk.0.attn_q.weight` answers false; `blk.1.ffn_down_exps.bias` and `blk.1.ffn_norm_exps.weight`, dense entries of the computed plan, answer false and 0 from both entry points; `blk.2.ffn_down_exps.bias`, which no plan entry holds, answers false; with the plan cleared, `blk.1.ffn_down_exps.bias` answers false and both entry points answer `NO_PLAN`; and on a second computed plan with `n_experts == 0`, whose budget places every entry on device 0 so that layer 0's `layer_device` is 0, read back or the run is VOID (r28 n-3), `blk.0.ffn_down_exps.weight`, which that planner keeps dense, answers false and 0 from both entry points. Its REDs: the plan-only form ("no dense entry" alone) answers true for `blk.0.attn_q.weight` and `blk.2.ffn_down_exps.bias`; the classifier-only form (no dense-entry test) answers true, and `NO_PLAN` from the entry points, for the `n_experts == 0` dense expert name; and a classifier that reads the usage registry, keying `g_sycl_weight_usages` by the bound load effect's owner as `ggml_backend_sycl_try_register_weight_usage` does (`:14062-14078`), answers true for `blk.2.ffn_down_exps.bias` and for the cleared plan, and, because the planner calls the same classifier (r27 m-2), the computed plan splits `blk.1.ffn_down_exps.bias` and `blk.1.ffn_norm_exps.weight`, so they have no dense entry and answer true and `NO_PLAN`. Since the plan is computed, not installed, that last RED is also r27 m-2's behavioural gate: the planner's split and the predicate move together. The mutant's owner route is pinned because the other route fires none of the three registry REDs (r29 m-1): a reader that takes its owner from the identity snapshot, as the backend's name-keyed registry reads do (`:26545-26547`) and as a tensor without an extra does (`:2851-2854`, `:2789-2790`, `:2680-2683`, `:2878-2892`), finds no snapshot at plan time or with the plan cleared, so it forms no key, and after the install finds the provisional snapshot, which carries no slot (`slot = UINT32_MAX`, generation 0; `:2966-2982`, `unified-cache.hpp:1483-1487`), so its key misses every registered row. (n) claims its registry REDs for the bound-effect route only. The plan-only form is the one that would break the default configuration: under `GGML_SYCL_HOST_WEIGHT_LAYOUT=cpu`, ir18 moves a host-tiered layer's dense weights to `CPU_REPACK`, the late inventory never sees them (`fefb92980 src/llama-model.cpp:2562-2575`), so it would read every one as a split expert, the reduction would skip them, and the CPU-executed layer would read as device: T6 would bind a SYCL descriptor for a CPU consumer (P3), and 30h4's sync would make 0 corrections again, the `c-pj9b` regression (`:2568-2572`); (l) and (m) run `GGML_SYCL_HOST_WEIGHT_LAYOUT=off`, so no GPU arm would see it. **The source pin** (r27 m-2, r28 n-1), a pure-Python gate on `unified-cache.cpp`: it takes the bodies of the two `placement_tensor_info` overloads, `compute_placement_plan` at `d8a67422d :27960` and `compute_multi_device_plan` at `:29348`, not the forwarders that take pairs (`:27950-27958`, `:29337-29346`), which hold no gate; it strips comments and string literals; and it requires `ggml_sycl_usage_is_moe_expert(` exactly once in each body and no `MOE_EXPERT_WEIGHT` token at all in either, so neither a commented call nor a reversed operand or a `switch` case can pass it. Its RED is today's file, whose two bodies carry the token only at their gates (`:28032`, `:29494`; `fefb92980 :27974`, `:29436`); (o) the `no_alloc` pass (r24 I-1), host-only, CPU: (k)'s stub registration, exporting also the descriptor buffer-type hooks, with `n_gpu_layers` > 0 and `no_alloc` set, on (e)'s gemma4 shape and on a qwen4exp shape whose `qwen4exp.ple.layers` is `[1]`, each with (k)'s table, `per_layer_token_embd.weight` a sparse region of 4 GiB plus one row, under `AUTO`, which `common_fit_params` copies with the rest of the model params, so that the entry is eligible (r25 N-4), each driven twice, through `llama_model_load_from_file` and through `common_fit_params`, the path of `-fit on` and `llama-fit-params` (`common/fit.cpp:56-60`), with the stub device reporting its memory. Engagement, or the run is VOID, which counts as failed: the per-model line reads `taker=1` and the entry's `deferral` is `STORAGE`. Pre-registered: `load_tensors` returns true with no `[LAZY-PLAN-BUG]` line; the descriptor's buffer type is `ggml_backend_sycl_lazy_host_buffer_type()`, with a null base and size 0; `llama_model::memory_breakdown()` reports 0 for it; and `common_fit_params` returns. Its REDs: the refusal left active under `no_alloc`, which prints `[LAZY-PLAN-BUG]` naming layer 0 (gemma4) or layer 1 (qwen4exp) and fails the load; the descriptor bound with the sentinel base, which fails `memory_breakdown`'s null-base assertion (`src/llama-model.cpp:2936`), scored on rc 134 in its own process; and the breakdown left on `ggml_backend_alloc_ctx_tensors_from_buft_size`, which reports the table's size under the host form's type; (p) one failure status on the server path (r24 I-2), GPU, run by the lead: (l-S)'s configuration at `-c 4096`, a pinned arm, on `llama-cli -lv 4 --jinja -st --simple-io --no-display-prompt` with (l)'s fixed prompt, `--temp 0 --seed 42` and `-n 16`, since `llama-cli` runs the server context (*§M60(a)*), and `GGML_SYCL_LAZY_READ_FAULT=2`, so that the first ubatch after the warmup fails its reads (T5). Engagement, or the run is VOID, which counts as failed: the variable's init WARN and at least one `[LAZY-READ-FAULT] injected` line. Pre-registered (r25 M-2): `grep -c 'Compute error\. off = [0-9]*, n_batch = [0-9]*, ret = -3$'` prints 1 (the `SRV_ERR` line, `tools/server/server-context.cpp:3710`); the `decode() failed: Compute error.` line that the caught throw prints (`:2892-2894`) is expected, and a bare `grep -c 'Compute error'` therefore reads at least 2 on a correct tree; `grep -c 'failed to find free space in the KV cache'` prints 0; and rc is 0, because `-st` leaves the turn loop after its one turn whatever the turn's result (`tools/cli/cli-context.cpp:658-665`). The pinned scorer's negative control is the RED's retry line, on which it prints 0. Its RED is the status left untranslated, under which the run prints that retry warning with `ret = 2` and the pinned scorer prints 0; (q) the CPU form's failure path, host-only, on (e)'s gemma4-shaped fixture with the stub query answering -1 for layer 0 and the table at STORAGE on T5's mock engine, whose read-fault hook fails epoch `e`'s reads (r24 I-2, M-7). Engagement, or the run is VOID, which counts as failed: the hook fired and the op recorded the failure. Pre-registered: the failing `llama_decode` returns -3; during it `lazy_cpu_epochs_released{op}` rises by 1 and `lazy_cpu_epochs_released{end}` does not; `cpu_slot_held == 0`; and the next `llama_decode` returns 0 with the rows the fixture wrote. Its REDs (r25 M-1): the status left untranslated, under which the failing decode returns 2; a release on success only, under which the end point's release catches the slot, so the failing decode's count moves from `{op}` to `{end}`; and both failure-path releases removed, the op's and the end point's, under which the next decode's `prefetch` prints `[LAZY-PLAN-BUG] cpu landing slot held` and fails; (r) the lazy tensor's registration (r25 I-2), host-only, CPU, on (e)'s gemma4-shaped fixture with (k)'s stub registration, which exports lifecycle hooks, extended with the loader's hook set that `register_sycl_tensor_metadata` requires before it does anything (`src/llama-model-loader.cpp:1541-1545`, resolved at `:49-55`), and the stub device as the tensors' `selected_final_dev` (r26 M-3): `weights_evictable` answers true; `try_register_usage` (`ggml_backend_sycl_try_register_weight_usage`) answers true, since `:1715-1716` throws otherwise; and `register_host_weight` and `register_identity` count their calls by tensor name. The load is a taker load under `ON`. Engagement, or the run is VOID, which counts as failed: `register_host_weight` is called for at least one non-lazy weight. Pre-registered: `register_host_weight` is never called for `per_layer_token_embd.weight`, and `register_identity` is called once for it. Its RED is the unchanged tail, under which `register_host_weight` is called once for the table | `deepseek41` follows the §9.1 contract in `llama.cpp-9g22`'s own tickets |
| T6b | llama handling of the descriptor buffer: memory breakdown and model size report resident bytes; `load_all_data`'s **one skip** for the table, the one source of "the table is not loaded" (P4), at the top of its per-tensor loop after the weight lookup and the progress callback (`694ec2354 src/llama-model-loader.cpp:1625-1636`) and before `from_mapping` (`:1640`), keyed on the SYCL branch's inventory entry and never on `lazy.has()`, `use_mmap` or the lazy-range record (r18 N-2), which adds the tensor's bytes to `size_done` (`:1750`) before it continues, so it covers the mapping branch with its mlock growth (`:1640-1671`) and the file-read branch (`:1672` on) alike, and the last-call cleanup and final progress callback (`:1776-1793`) still run (§4.1, r15 m-2); the skip is reached because `lazy_ctx` joins `ctx_buf_maps` with a `buf_map` that sends every file index to the descriptor buffer (§3.2; r24 M-3); on the `no_alloc` pass `load_all_data` never runs (`fefb92980 src/llama-model.cpp:2834-2842`), and `memory_breakdown`'s `no_alloc` branch reports 0 for `lazy_ctx` instead of `ggml_backend_alloc_ctx_tensors_from_buft_size`, the table's nominal size (`:2933-2938`; r24 I-1) | T6 | a load of a lazy-table fixture reports model size equal to the resident bytes and completes, which shows that no byte of the table reached the descriptor, whose `set_tensor` aborts naming the table (§3.2), and the test's progress callback last reports 1.0; no clause needs a mapping: the arm runs in the SYCL load mode, where `use_mmap` is false (`c69d5774d src/llama-model.cpp:2558-2561`) and a missing skip sends the table to the file-read branch, and on a tree that carries 5efe's C1-C4 it runs again under `-lm mmap` (r15 m-2; r26 M-4); RED 1 is the skip moved into the mapping branch, under which the file-read branch reaches the descriptor and the load aborts naming the table, run in its own process and scored on rc 134 plus the descriptor's line naming the table, under T6 (e)'s sink conditions (the synchronous default sinks, no callback installed, no `common_init()`; r16 m-3); RED 2 is the skip without the `size_done` add, under which the load returns with the progress callback's last value below 1.0 (r15 m-2) | |
| T7 | block-exec's decline of a lazy leaf: `ggml_sycl_block_exec_prepare_context` returns `trace_prepare_fail("lazy-table-leaf")` before its leaf-bytes check and its output-allocation probe, with a once-per-process WARN naming the table, so the graph runs on the ordinary path (§3.2; r27 n-4, r28 m-3); the SYCL-owned gather for **both residencies** under D6-M(b) (*lead ruling §T8*): lazy `GET_ROWS` recognition, with the carrier's (posted `k`, `consumed`) on each `lazy_gather_ctx`, its `none` sentinels, the check and consume at every gather's dispatch whatever the gather reads, `ggml_backend_sycl_ubatch_end(backend)`, `noexcept`, which sets every one of the backend's posted `k`s to `none` and in production is called only by T6's end point, with its `ubatch_end_calls` counter, and its `ubatch delimiter missing` line with the branch named (§5.3, *§D12n* 2, *§D12p*, *§D12q* I-2, *§D12s*); the epoch ring (ids captures, marker words with their acquire load and system-scope release store and their initial seeding, host-signalled events in the refcounted `lazy_event_pool`, the per-copy wrapper, the reset rule); `D = in_flight_depth + 2`, read at the context transaction, and the planner function `ggml_sycl_plan_input_raw_guard_present()`, with its `lazy-work-off-guarded-stream` refusal whenever the queue objects differ, its cause printed as `cause=tp`, `device`, `pre-cache` or `other`, classified in that order, the host buffer's device's cache queue the only accepted queue, and the witness of llama's guard declaration, once per ubatch at T6's delimiter, keyed on (allocation serial, count) after its stamp check with the `input guard unstamped` line (§5.3, *§D12g* I-1, *§D12h* I-3, *§D12m* 3, *§D12n* 1, *lead rulings §T9a, §T9b, §T9c, §T9d*; r7 I-3), and the plan-time report of the two-card configurations it admits (r7 m-12); the worker's dequant and signal; the explicit H2D copy on the event; the backend's lazy cancel step, run before `ggml_sycl_cpu_tg_flush_pending()` in both `graph_compute` catch paths, in `synchronize()` and at `graph_compute`'s entry only under the failed-graph mark, which carries its epoch scope and is cleared only by a completed step, at teardown, and in `ggml_backend_sycl_free` (§5.3, *§D12*, *§D12b*, *§D12c*, *§D12d*), with the device-written cancellation marker, the `drain` force-signal and the stage re-post, the stage executor's cancelled-first order (*§T11*), the step at a throwing `synchronize()` (r7 m-8), and the poison flag with its three sources, an unwritten enqueued marker, a queue-refused marker submission and a failed host buffer sync, the catch-path waits recording their exception as the cause without poisoning, and the host buffer's sync classifying device loss into its per-device fault fact (*§D12*'s m3 rider, r7 m-9, *§D12h* m-1, m-2); the `[LAZY-PLAN-BUG]` exhaustion and guard-witness lines, their abort read only through `ggml_sycl_strict_enabled()` (*§G1*, *§G1a*, *§G1b*); the non-graph watchdog scope, which starts the watchdog, on the host buffer's sync, engaged only for a context with a lazy or cold plan, read at check time from the context-serial registry through the context serial stamped onto every host compute buffer the context allocates, each re-reserve included (*§D12h* I-3, *§D12m* 5), and on every other lazy wait outside `graph_compute` (§5.3, *§D12a* m16, r7 I-4, *§D12f* m-7); the "sources not staged" predicate split agreed with zhcn (§7.2); fusion exclusion; the record-mode boundary set, which takes `9g22`'s cold first readers by the same predicate, the widened decode triggers with the early return keyed on the boundary set, and the named direct reasons (§7.4); if HT-6 resets, continuation submission (§7.2, *§T12* I-D); the engram controls `GGML_SYCL_LAZY_CONTROL*` (§7.4), listed in `sycl-env-vars.md` (the strict accessor's export, the `GGML_SYCL_STRICT_LEASES` catalog row and the corrections to `sycl-env-vars.md:590` and the canonical contract's `:178-179` are `llama.cpp-uwlx`'s, *§G1b*; T7 adds only its own tag to uwlx's family list, if that list lacks it); the `lazy_output_copies`, `lazy_zero_copy_reads` (resolved-pointer, and arm C's own range), `copy_submit_*`, `slot_*`, `epoch_*`, `lazy_graph_direct` (by reason), `lazy_context_poisoned`, `lazy_mark_consumed{catch,sync,entry,backstop}`, the per-backend `ubatch_begin_calls`, `lazy_witness_checks`, `lazy_witness_rebaselines` and `graph_compute_calls` (*§D12m* 3, 8), `lazy_epochs_cancelled` and `lazy_epochs_drained` by site `{catch,sync,entry,teardown,backstop}` (always compiled, *§D12e*, *§D12f* I-2, *§D12h* I-2), `graph_exit_deferred` and `host_buffer_syncs` (per buffer, counting sync entries past the no-device return, *§T9e*, *§D12h* m-3) counters; the host buffer's stamp, (context serial, allocation serial), set at `alloc_buffer`, whose context serial the watchdog scope and the fault attribution resolve through the registry and whose allocation serial the witness reads (*§D12h* I-3, *§D12m* 5); the `noexcept` classifier helper `ggml_sycl_exception_classify`, called once at each of its four enumerated site classes, with the per-device device-lost fact always recorded, its `[SYCL-FAULT] device lost` line and the per-device `device_lost_faults` counter, the non-throwing current-device lookup through the inverse device map, the ambient-backend guard declared before each `try`, and the census source gate with its comment-stripping, per-function scanner (§5.3, *§D12h* m-2, *§D12m* 6, *§D12n* 3, *§D12o*, *§D12q* I-1; r12 m-2, m-3, m-5); the poison check over every scope epoch the step acted on, drained ones included (*§D12i* m-8, *§D12m* 8); the `noexcept`, capacity-bounded stage re-post (*§D12m* 8); the test-only hook that removes the host-buffer sync's waits (*§T9b*); the test-only hooks that leave a cancel step incomplete at its marker enqueue and turn the llama backstop off (*§D12e*, *§D12f* I-2); and the test-only queue seam `ggml_sycl_test_queue_override`, kinds TP and FOREIGN, shared with `9g22` (*§T9c*, *§D12f* m-5) | T5, T6; `llama.cpp-zhcn` for the predicate split and the host-memory check; **`llama.cpp-uwlx`** for `ggml_sycl_strict_enabled()`, a hard edge (*§G1b*, *§G1c*): uwlx is the sole definer of `ggml_sycl_strict_enabled()`, and every T7 commit that calls it depends on uwlx's merge; the tracker edge th32 -> uwlx exists | **first the six probes HT-1..HT-6 (§7.2), lead-run, GPU**, built on `test-fattn-packed-k-lifecycle.cpp:262-342`'s gate class, HT-3 on the production execution queue per device (r5 m-1, m-14). Then the fixtures below. **No fixture forks after SYCL init** (*§D12m* 7): every watchdog, rebuild and classifier control runs as its own process, scored on its rc plus its named log line. **Every one** runs under a pre-registered 60 s timeout with `GGML_SYCL_OP_TIMEOUT_MS=120000`, so the watchdog cannot fire first and an `rc=1` stays unambiguous (the watchdog arm of (4) alone sets its own timeout), and with `GGML_SYCL_DISABLE_GRAPH=1` unless it is about record mode (r5 m-6). **How a fixture reaches `prefetch(e+D)` with a copy held** (r6 C-1 (c)): the test-only hook removes the host-buffer sync's wait (*§T9b*); the ubatches are single-token decodes (`n_ubatch` 1, logits only on the last token), so `graph_compute` takes the deferred exit, which the fixture asserts first (`graph_exit_deferred` rises per ubatch); `GGML_SYCL_DRAIN_INTERVAL=0`; and the graph is dense, so no MMID wait applies. Otherwise the W1 exit drain or the periodic drain would hold the submitter in `graph_compute` behind the held copy. (1) the §7.3 zero-copy fixture (fails first with `lazy_output_copies == 0`; positive control: the pre-registered aliasing mutant, `lazy_zero_copy_reads >= 1` per graph; arm C's own mutant likewise); (2) **the dense held-copy fixture** (*§T8* I-1, r5 m-7): a dense graph, gemma4-shaped with `-lzm on`, of exactly `D` = 3 ubatches in one decode, with one lazy gather, and copy 0 held behind a test-controlled event. It uses the sync-removal hook, since the input sync would otherwise hold `set_inputs(1)` until copy 0 completes. The pre-registered RED is a mutant in rev 2d's form, whose `prefetch` treats an unset marker two epochs back as overrun: it aborts at `prefetch(2)`. GREEN completes with rows equal to CPU `get_rows` for every ubatch, with no abort and no exhaustion. Positive control: a mutant whose stage 2 skips the "copy `e-2` completed" guard overwrites output slot 0 while copy 0 is held, so ubatch 0's rows mismatch; (3) **the ids-capture race fixture** (*§T8* I-4): a dense 2-ubatch fixture, with the sync-removal hook for the same reason, whose test hook holds the worker's gather for ubatch 0 until `set_input(1)` has returned. The pre-registered result: an in-place mutant, reading the rows input instead of the ring entry, gathers ubatch 1's rows for ubatch 0 (a mismatch against CPU `get_rows`), and the ring build matches; (4) **the recoverable-failure fixture** (*§T8* I-2, *§D12*, r5 m-6), its arms in separate processes. **The pre-reader arm:** a test hook throws `ggml_sycl_fallback_error` in ubatch `k` after its prefetch and **before the first reader of the gather's `dst`**, so before its step-3 copy is enqueued (gemma4 reads the PLE before layer 0, `694ec2354 src/models/gemma4.cpp:176-181`, so "in block 0" would be too late). `llama_decode` returns its failure status with no abort; the next three ubatches complete with correct rows; `lazy_epochs_cancelled == 1`; no slot is written while the cancelled epoch's stage is in it (a slot-owner assertion in the mock). **Its signalled-stage sub-arm** (*§D12f* I-1, *§D12h* I-1), the reads NOT held and the llama backstop off: a warm ubatch `k` (HOST, or STORAGE with every row a hit), with the throw hook firing only once event `k` has signalled (the test hook itself polls the entry's event; production code never does), as in `9g22`. Its engagement check is pre-registered: event `k` has signalled at the catch step's entry, recorded by the step's test probe; if it has not, the arm is VOID, which counts as failed. Pre-registered: `lazy_epochs_cancelled == 1`, `k \| CANCELLED` is written, stage 2 of `k+2` runs (no parked stage), there is no exhaustion through `prefetch(k+D)`, and the next ubatches' rows are correct. Its RED is a step that keeps rev 2i's filter on the stage signal: it skips `k`, stage 2 of `k+2` parks on copy `k`, copy `k+2` blocks the in-order queue, and the fixture hits its 60 s timeout in `set_inputs(k+3)`'s input sync. **The backend arm** (*§D12* I-1): the table's reads held 5 s (`GGML_SYCL_LAZY_CONTROL=hold`, `HOLD_MS=5000`), and the throw placed at a node after the first reader, so the copy is enqueued and parked. Pre-registered: `graph_compute` returns `GGML_STATUS_FAILED` in < 100 ms from the throw, with no abort; `lazy_epochs_drained == 1`; the held reads' block-ring slots are released only after they complete; the next ubatch's rows are correct. The same with the throw before the first reader also returns in < 100 ms (`lazy_epochs_cancelled == 1`). A teardown sub-arm frees the context with an enqueued epoch whose reads are held: the lazy teardown and the destructor's `synchronize()` return in < 100 ms. The RED is rev 2f's order, the cancel only in `process_ubatch` after the catch path's waits: it takes the 5 s. A throw between the hand-off and the first reader, `9g22`'s placement, cannot separate the two orders here, because th32 enqueues no copy before its first reader and the flush enqueues none, so it is the second sub-arm, not the RED. **The successful-decode arm** (*§D12b*): a decode that succeeds while an enqueued epoch's reads are held, then `llama_synchronize()`. Pre-registered: `synchronize()` waits out the hold, `lazy_epochs_drained` and `lazy_epochs_cancelled` stay 0, and the rows are correct. Its RED is an unconditional cancel at `synchronize()`, which force-signals the epoch, so its copy moves an unwritten output slot and the rows mismatch. (The ruling's form, PREFETCHED epochs `e+1` and `e+2` kept intact, does not arise in th32: `prefetch(k+1)` runs at ubatch `k+1`'s delimiter, after ubatch `k`'s `synchronize()`, so the epochs live at a `synchronize()` are enqueued ones, as in `9g22`'s arm.) **The mark arms** (*§D12d*, *§D12e*, *§D12f* I-2), each in its own process, with the llama backstop off (`lazy_epochs_cancelled{backstop} == 0` and `lazy_mark_consumed{backstop} == 0` pre-registered as that hook's engagement check) and the incomplete-step hook pinned at the catch step's marker enqueue: it throws after the host stores and before the SYCL submit, as a test-only exception, so it is not a poison source (§5.3). Ubatch `k` fails before its first reader with `k`'s reads held (`HOLD_MS=2000`), so its stage has not signalled. **(a) Incomplete catch step:** the hook, then `llama_synchronize()`. Pre-registered: after the catch path, `lazy_mark_consumed{catch} == 0`, `lazy_epochs_cancelled{catch} == 0` and the mark is set; after `synchronize()`, `lazy_mark_consumed{sync} == 1`, `lazy_epochs_cancelled{sync}` equals the size of the mark's scope (ubatch `k`'s epochs, one per lazy table, none with its copy enqueued), `k \| CANCELLED` is written, and the mark is clear. Its RED is the `synchronize()` site removed: the mark survives, `lazy_mark_consumed{sync} == 0`, and `k`'s marker is never written. **(a2) The order clause** (*§D12e*, *§D12h* m-1), on a two-table fixture so the scope holds a draining epoch: the throw is placed after table A's first reader and before table B's, so A's copy is enqueued with its reads held 5 s and B's is not, and the hook fires at B's marker enqueue. Pre-registered: `graph_compute` returns in < 100 ms, because A's drain flag and re-post were stored before the hook threw, with `lazy_epochs_drained{catch} == 1` and `lazy_epochs_cancelled{catch} == 0`. Its RED is the step with its host stores after the marker enqueue: the hook throws before A's flag is set, and the catch wait holds for A's IO, ~5 s. **(b) Retry without `synchronize()`:** the hook, then `D` single-token deferred-exit decodes `k+1..k+D` with distinct cold ids and no `synchronize()`, `k+1`'s reads held past its `graph_compute` entry. Pre-registered: `graph_compute(k+1)`'s entry runs the poison check and then the scoped step, `lazy_mark_consumed{entry} == 1`, `lazy_epochs_cancelled{entry}` equals the scope, epoch `k+1` stays intact, its rows equal CPU `get_rows` once its hold ends, and there is no exhaustion through `prefetch(k+D)`. Its RED is the entry site removed: `lazy_mark_consumed{entry} == 0`, `k` is never cancelled, stage 2 of `k+2` parks on copy `k`, and the retry fails at its 60 s timeout in `set_inputs(k+3)`'s input sync. Its false-poison RED (*§D12h* m-1) is a poison check that reads an unenqueued marker: `k`'s entry is cancel pending with no marker, so the retry fails at entry with `lazy-context-poisoned`. Its second RED is an unscoped step at entry: it also cancels epoch `k+1`, whose copy then moves an unwritten output slot, so `k+1`'s rows equal epoch `k-1`'s, the last dequant into slot `(k+1) mod 2`, not CPU `get_rows`. **(b2) The mark survives to `synchronize()`** (*§D12e*, for parity with `9g22`; th32's (b) already isolates the scope at entry, because `k+1` is prefetched at its delimiter, before its `graph_compute`): the hook, then one decode `k+1` whose entry step also throws at its marker enqueue, then `llama_synchronize()` while `k+1`'s reads are still held. The entry step is its own `try`: its throw leaves the mark set and the graph runs (*§D12d*, *§D12h* m-4), so it never reaches `graph_compute`'s catch handlers, which would write a new mark scoped to `k+1`. Pre-registered (*§D12j* m-7 for the counters): `graph_compute(k+1)` returns `GGML_STATUS_SUCCESS`, `lazy_mark_consumed{entry} == 0`, `lazy_epochs_cancelled{entry} == 0`, `lazy_epochs_drained{entry} == 0`, `lazy_mark_consumed{sync} == 1`, `lazy_epochs_cancelled{sync}` equals the scope, `lazy_epochs_drained{sync} == 0`, epoch `k+1` stays intact, and its rows equal CPU `get_rows`. Its RED is an unscoped step at `synchronize()`, which drains `k+1`: `lazy_epochs_drained{sync} == 1`, and its rows equal epoch `k-1`'s. **The retire arm** (r5 I-2, rebuilt by *§T12* I-A): with the sync-removal hook, wait until copies up to `e-1` have completed; hold the strand's stage for `e` and stage 2 of `e+D`; throw in ubatch `e` before its first reader, so `e` is cancelled; drive the submitter on toward `prefetch(e+D)`. **`e`'s stage is released** when `prefetch(e+D)` has returned or 1000 ms after the throw, whichever comes first, while stage 2 of `e+D` stays held (r7 I-1). Pre-registered for the build: `graph_compute(e)` returns only after that release, ~1000 ms after the throw, because the catch path's queue wait (`:107424`, `:107554`), which the hook does not remove, waits for the cancellation marker, and the marker waits on the held stage's signal; `prefetch(e+D)` then finds `e \| CANCELLED`, `epoch_ring_exhausted == 0`, and once stage 2 of `e+D` is released, copy `e+D`'s rows equal CPU `get_rows`. The pre-registered mutant writes `e \| CANCELLED` from the host at cancel and lets the strand's cancelled stage signal the event. Its `prefetch(e+D)` takes the entry at once and resets the event **before** the release, so the released stage's signal lands on epoch `e+D`'s event and survives: copy `e+D` then runs as soon as copies `e+1..e+D-1` complete, while stage 2 of `e+D` is still held. Pre-registered mismatch: copy `e+D`'s rows equal epoch `e+D-2`'s (the last dequant into output slot `(e+D) mod 2`), not CPU `get_rows` for `e+D`; every ubatch has distinct ids, so the two differ. The mutant's `graph_compute(e)` also returns without waiting for the release, since nothing on its queue waits on event `e`. So the arms separate on copy `e+D`'s rows and on `graph_compute(e)`'s return time, not on exhaustion. (Rev 2g had a guard arm here; it could not fail, because the catch path's own queue wait orders the marker, and it is dropped, r7 m-3.) **The poison arm** (*§D12*'s m3 rider): a test hook makes the cancellation marker's SYCL submit throw a `sycl::exception`. The next `llama_decode` fails with `lazy-context-poisoned` naming the cause, `D + 1` more decodes each fail the same way with `epoch_ring_exhausted == 0`, and a second context on the device still decodes correctly (*§D12g* m-1). A second sub-arm makes the host buffer's sync throw a `sycl::exception` whose `what()` carries `UR_RESULT_ERROR_DEVICE_LOST`, the classifier's positive control, and expects the same from the device's fault fact for every lazy context on that device; each other site class of the classifier has its own positive control, one process each, scored on the helper's `[SYCL-FAULT] device lost` line with its `site=` label and on `device_lost_faults` (*§D12n* 3, *§D12q* I-1): class 2, `synchronize()`'s wrapped wait throwing a `sycl::exception` whose text carries `UR_RESULT_ERROR_DEVICE_LOST` inside a scoped ubatch, is a death test, scored on rc 134 from `SYCL_CHECK`'s abort plus the line with `site=check_try_error` printed before it, never on `device_lost_faults`, which dies with the process, and run, as T6 (e) is, on ggml's synchronous default sink for this `GGML_LOG_ERROR` line (`ggml/src/ggml.c:313-318`), with no callback installed through `llama_log_set` or `ggml_log_set` and no `common_init()` (r14 m-4), since `common_log`'s asynchronous worker can lose the line queued before the abort (`common/log.cpp:374-378`; r13 m-3); class 3, a node in `graph_compute` that throws the same text past every inner catch to the top-level catch, returns `GGML_STATUS_FAILED` with the line and `site=boundary`, and the next decode fails `lazy-context-poisoned`; class 4, a catch-path wait throwing the same text, is entered through an exception that does not carry it, a `ggml_sycl_fallback_error` for the fallback catch's waits and a plain `std::runtime_error` for the cleanup's, so only the wait's own call can record the fact, and prints the line with `site=catch_wait` (r12 m-4). **The always-recorded control** (r11 m-8, *§D12q* I-1) sits at a non-aborting, unattributed site: a test hook throws a `sycl::exception` with the device-lost text at the host buffer's sync on `buf_output`'s clear during context creation (`src/llama-context.cpp:849`, `:3504`), a buffer with no stamp, so no context is attributed. Pre-registered: the line with `ctx=none`, `device_lost_faults == 1`, and the context's first decode fails `lazy-context-poisoned` from the device's fault fact. Its RED is a helper that records the fact only with an attributed context: no line, and that decode succeeds; a third throws one whose `what()` carries `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` instead, the negative control, and expects only the context that the buffer's context serial names in the registry poisoned, a second context on the device decoding correctly (*§D12f* m-8, *§D12h* m-2, *§D12m* 6). **Its drained-poisoned sub-arm** (*§D12i* m-8, *§D12m* 8): the backend arm's throw after the first reader, so the catch step drains the enqueued epoch `e`, with a test hook that turns the drained copy's marker write into a no-op, the stand-in for a queue that dies after the drain. Pre-registered: the marker check after the step reads `e`'s word as neither `e` nor `e \| CANCELLED`, so `lazy_context_poisoned == 1` when `graph_compute` returns, before any later input sync, with the marker word as its cause, and the next `llama_decode` fails with `lazy-context-poisoned`. Its RED is rev 2k's check, which reads only cancel-set markers: `lazy_context_poisoned == 0` when `graph_compute` returns. The poison arm's own RED is the poison check placed after the ring evaluation, which reports `[LAZY-PLAN-BUG] epoch ring exhausted` at the `D`-th decode after the poisoning, with `epoch_ring_exhausted == 1` (r7 m-9 (c)). **The watchdog arm** (*§D12a* m16), with `GGML_SYCL_OP_TIMEOUT_MS=2000` and the fixture's own 60 s timeout: a decode whose table's reads are held with no completions (`HOLD_MS=30000`), so neither the strand nor the IO engine makes progress, then `llama_synchronize()`. Pre-registered: the process `_Exit(1)`s with its `[SYCL-WATCHDOG]` line within ~2.5 s of the wait's start. The hold must stop IO progress as well as the strand, or the IO-liveness rule would correctly keep the process alive. Both watchdog arms use a synthetic STORAGE table whose one row spans 20 blocks of 4 KiB (80 KiB), so a single-token decode issues exactly 20 block reads (*§D12f* m-9). Liveness control: the same wait with a test hook completing one of those 20 held reads every 500 ms, each heartbeating. Pre-registered: every held read has completed by 10 s, `synchronize()` returns rc 0, and no `[SYCL-WATCHDOG]` line appears. The RED is the wait without the non-graph scope: the watchdog idles, the reads complete when the hold ends, and `synchronize()` returns rc 0 at ~30 s with no `[SYCL-WATCHDOG]` line (r7 m-4). **Its input-sync sub-arm** (r7 I-4), with the sync-removal hook off: a single-token deferred-exit decode `k` whose table's reads are held with no completions (`HOLD_MS=30000`), then a second `llama_decode`, whose `set_inputs(k+1)` waits in the host buffer's sync for copy `k`. Pre-registered: `_Exit(1)` with the `[SYCL-WATCHDOG]` line within ~2.5 s of that wait's start, and the same liveness control. Its RED, the sync without the scope, returns at ~30 s when the hold ends, with no line. **Its regression sub-arm** (*§D12f* m-7): a model with no lazy or cold plan (the Mistral gate model), `GGML_SYCL_OP_TIMEOUT_MS=2000`, and a test-held event parking the queue 5 s ahead of a `set_tensor` on its host compute buffer: rc 0 and no `[SYCL-WATCHDOG]` line, because the scope is not engaged there; its engagement check, as in `9g22`, is that the held sync returns at least 2 s after its entry, or the arm is VOID, which counts as failed, since a hold that missed the sync's queue would also give rc 0 and no line (r11 m-10); its positive control is the same hold on a lazy-plan context, which prints `[SYCL-WATCHDOG]` and exits within ~2.5 s. **Its re-reserve sub-arm** (*§D12h* I-3, *§D12m* 9): a lazy context runs at least 8 warm decodes, then, in this order, a decode, `llama_set_causal_attn(false)`, a decode, `llama_set_causal_attn(true)` and a decode. Each setter only sets `sched_need_reserve` (`src/llama-context.cpp:2566-2576`), so the decode after it runs `sched_reserve()` and allocates a new host compute buffer with a new allocation serial; two setters before one decode would give one. Then it runs the input-sync sub-arm. Pre-registered: two new allocation serials, `lazy_witness_rebaselines == 2`, `lazy_witness_checks` equal to the ubatch count, no `input guard not observed` line across the re-reserves, and the input-sync sub-arm still `_Exit(1)`s within ~2.5 s. Its RED is the stamp-once form: the new buffer carries no stamp, so the sync opens no scope and returns rc 0 at ~30 s with no `[SYCL-WATCHDOG]` line. Its cross-buffer RED is a witness that requires the new buffer's count `>` the old buffer's snapshot (a `!=` comparator is not the mutant, since it passes): the warm decodes measure and pre-register `s`, the buffer's syncs per ubatch; the old buffer's snapshot is then at least 8 × `s` and the new buffer's count at its first check is `s`, so it reports `input guard not observed` at the first ubatch after the first toggle. A second process runs the same sequence and then, in place of the input-sync sub-arm, the poison arm's negative-control sub-arm, an injected exception whose text carries `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` at the sync (*§D12h* I-3, *§D12m* 6): pre-registered, only this context is poisoned; its stamp-once RED poisons no context, and the next decode proceeds; (5) **the public-API exhaustion fixture** (*§T9a*, *§T9b*, *§T10* I-1): `llama_decode` through the public API with no output read between calls, `GGML_SYCL_LAZY_CONTROL=hold` with `HOLD_MS=2000` engaged after 8 warm ubatches, each shape in its own process under its own 60 s timeout (r6 m-5). **Shape (a)**, the deferred-exit arm (r6 C-1 (4)): at least `2 x D` single-token decodes with `GGML_SYCL_DISABLE_GRAPH=1` and `GGML_SYCL_DRAIN_INTERVAL=0`, the deferred exit asserted engaged, so the next ubatch's token write is the only bound; about 12 s of hold. Every held ubatch reads ids no earlier ubatch read, so each has reads for the hold to delay, and `copy_submit_io_pending` rising on every held ubatch is the arm's own positive control (r7 m-11). **Shape (b)**, the llama-server shape: `n_ubatch` 16 and `n_batch` 32, one 80-token prompt (more than `2 x n_batch`) split into `n_batch` decodes with logits only on the last token of the last one; about 10 s of hold. GREEN, on today's code: no exhaustion over every ubatch, and `host_buffer_syncs` rises by at least one per ubatch, the positive control that the guard fires on the real path. RED, scored on shape (a): the same run with the sync-removal hook reports `[LAZY-PLAN-BUG] epoch ring exhausted` at the (`D`+1)-th ubatch after the hold engages, with -3 and `epoch_ring_exhausted == 1` (STRICT off). Shape (b)'s ubatches are PP graphs, whose W1 exit drain also holds the submitter, so shape (b) cannot separate the guard from that drain and is a GREEN only. (Its cold arm, shape (a) on a cold-set fixture, is yl3v's acceptance, not T7's, since T7 does not depend on yl3v; see yl3v below, *§D12f* m-10.) **The plan-time REDs**, whose arms differ in the predicate's inputs (*§T9e*, r7 I-3), each with both strict forms (*§D12f* m-6): a context whose guard declaration names no input (as if llama wrote every input through `->data`, as it writes `KQ_mask`) gives, with `GGML_SYCL_STRICT_LEASES` unset, the `[LAZY-PLAN-BUG] input guard missing` WARN and the refusal `lazy-input-unguarded` with no abort, and, with it set before the first read, an abort (a death test); the same context declaring `inp_tokens` is accepted. The witness's RED is a fixture that declares `inp_tokens` while a test hook routes **every** syncing access to that host buffer, `inp_tokens`' and `inp_pos`' `set_tensor`, the output `get_tensor`, `clear`, `memset_tensor` and `cpy_tensor`, to the CPU implementation without the sync, with no output read between ubatches; `host_buffer_syncs` unchanged across the ubatch is the hook's own engagement check (*§D12f* m-1). Unset, it reports `[LAZY-PLAN-BUG] input guard not observed` at ubatch 1's witness and fails it with -3, with `graph_compute_calls` unmoved across that ubatch, so it never entered `graph_compute` (r11 m-5); set, it aborts (a death test). Its GREEN is every shape above, where the witness never fires. **The unstamped-guard REDs** (*§D12n* 1), each its own process: with the scope removed at `graph_reserve()`, the first host compute buffer carries no stamp, and ubatch 0's witness reports `[LAZY-PLAN-BUG] input guard unstamped` and fails it with -3, `graph_compute_calls` unmoved; with the scope removed at the `ggml_backend_sched_alloc_graph` site (`src/llama-context.cpp:2750`), `LLAMA_GRAPH_REUSE_DISABLE=1` (`:688-689`) so that every ubatch takes the non-reuse branch (`:2721`) to that site, and a test hook that makes gallocr replace the host compute buffer at the next allocation, as an outgrown chunk does (`ggml-alloc.c:916-940`), the replacement carries no stamp, and the first ubatch after it reports the line and fails with -3; its engagement check is that the line's allocation serial differs from the guard buffer's serial before the hook, or the arm is VOID, which counts as failed (r12 m-1). Their GREEN is the same two runs with the scope in place: no `input guard unstamped` line, and the replacement's new serial re-baselines the witness (`lazy_witness_rebaselines` rises by 1). **The delimiter arms** (*§D12g* I-1): a `cb_eval` GREEN, a lazy fixture decoded with `cparams.cb_eval` set to a callback that asks for every node, so the scheduler calls `graph_compute` once per range (`ggml-backend.cpp:2459-2493`), and a CPU-island GREEN, a synthetic lazy graph with a CPU-only node (a `GGML_OP_CUSTOM`, which SYCL declines) between its two lazy consumer layers, so each ubatch reaches SYCL's `graph_compute` at least twice. Each pre-registers 0 `input guard not observed` lines, `ubatch_begin_calls` equal to the decode count on every SYCL backend, `lazy_witness_checks` equal to it on the backend with the lazy plan, and `ubatches` equal to the decode count (*§D12m* 3, 8). Each RED is the witness keyed on `graph_compute` instead of the delimiter, which finds the count unmoved at the ubatch's second `graph_compute`: at least 1 `input guard not observed` line. Both GREENs also pre-register 0 `ubatch delimiter missing` lines, although each ubatch makes several `graph_compute` calls. **The direct-drive arm, the carrier's positive controls** (*§D12n* 2, *§D12p*, *§D12q* I-2; r12 m-7), each its own process, with `graph_compute_calls` rising on the call under test as its engagement check (VOID otherwise, which counts as failed). (i) **The consumed branch:** ubatch 0 through `ggml_backend_sycl_ubatch_begin`, `lazy_prefetch()` and `graph_compute`, then a second `graph_compute` of the same graph, which gathers each table again, before the test calls `ggml_backend_sycl_ubatch_end`. Pre-registered, with `GGML_SYCL_STRICT_LEASES` unset: the second call's first gather prints `[LAZY-PLAN-BUG] ubatch delimiter missing` naming the consumed branch, and `graph_compute` returns `GGML_STATUS_FAILED`; with it set, an abort (a death test). Its RED is a sticky carrier, a flag that `ubatch_begin` sets and nothing clears, under which the second call prints no line. (ii) **The k branch:** ubatch 0 as in (i) with one `graph_compute`, then `ggml_backend_sycl_ubatch_end`, then ubatch 1 calls `lazy_prefetch(1)` with no `ubatch_begin`, so each posted `k` is 1 against the record's 0, and runs `graph_compute`. Pre-registered: the line naming the k branch. Its RED is a reader that checks only `consumed`, which prints nothing, since the prefetch cleared it. (iii) **The no-pick shape** (*§D12p*): a two-table fixture, table A at layer 1 and table B at layer 14, A HOST-resident so that it reads no block (r13 m-1), whose ubatch 0 is delimited, runs with B reading rows, and ends, then ubatch 1 with no `ubatch_begin`, whose first gather is A's. Its engagement check is pre-registered: A's `rows_miss` is unmoved across ubatch 0 and B's rose, or the arm is VOID, which counts as failed. Pre-registered: the first `ubatch delimiter missing` line names table A. Its RED is the stale-post mutant, a reader that checks and consumes only a gather whose post carried a miss: A's gather is never checked, so the first line names table B. (iv) **Failure before the gather** (*§D12s*): ubatch 0 on (iii)'s fixture goes through `llama_decode`, so through `process_ubatch` and its end point, with a **one-shot** test hook that throws `ggml_sycl_fallback_error` at a SYCL-dispatched node ahead of A's gather in node order, named from a node dump of the fixture's graph taken first; it is never `token_embd`'s `get_rows`, a CPU weight in this plan (§4.5), since a hook on a CPU node never reaches SYCL's fallback catch. The hook fires in ubatch 0 only and is disarmed before the second call. `graph_compute` fails and `llama_decode` returns an error. The test then reaches llama's scheduler and ubatch 0's graph through `llama_context_last_graph_for_test()`, a test-only accessor T7 adds in an internal header that only the test links, and runs `ggml_backend_sched_graph_compute()` on them, a `graph_compute` with no `ubatch_begin`. Its engagement check is pre-registered, and runs in the GREEN and the RED alike: a test hook in `ggml_backend_sycl_ubatch_end` records each post before the clear, and A's reads posted `k` 0 with `consumed` clear, or the arm is VOID, which counts as failed. Pre-registered: the first `ubatch delimiter missing` line names table A, on the k branch. Its RED is `ggml_backend_sycl_ubatch_end` called with its clear removed: the hook still records A at (0, clear), A's post still matches the record, and no line is printed (r14 m-1). The none-clear placement mutant, a clear to `none` at every `graph_compute` exit, is caught by the CPU-island GREEN above, whose ubatch has a gather on each side of the CPU node, so its second SYCL `graph_compute` prints the line (*§D12s*, r14 m-6). The queue REDs are real mismatches (*§T9c*), built through the shared seam `ggml_sycl_test_queue_override` with no TP state (*§D12f* m-5): kind TP, pre-registered to print `cause=tp`, and kind FOREIGN, a queue on the host buffer's device with TP off and the cache present, pre-registered to print `cause=other`, each refused with `lazy-work-off-guarded-stream`. The pre-cache case is checked by name but has no RED arm, since it is expected to be unreachable (*§T9d*). A lazy first reader placed on a different stream index is the same queue object, not a mismatch, and is accepted. The two-card report runs here, against its pre-registration: a two-card plan is admitted only when layers 1 and 14 are both on SYCL device 0, the host buffer's device, and a single-card model on in-process device 1 (`-mg 1`, split none) is refused by case 2 (§5.3, r7 m-12, r8 m-3); (6) **the record-mode fixture** (*§T10* I-5, *§T12* I-B), three graphs, a MoE graph, the dense gemma4 shape and a synthetic dense graph, each with a lazy gather and `GGML_SYCL_GRAPH_DIAG=1`, scored by graph kind (§7.4): the FA mode is stated per arm (*§D12f* m-4): decode with `GGML_SYCL_FLASH_ATTN_GRAPH_ALLOW=1`: the MoE arm has `seg_replay > 0`, a `[SYCL-SEG-POLICY]` line with `graph_segments > 0`, zero futility WARNs, `lazy_boundary_dispatches == graphs` per table, and no `lazy_graph_direct` on its decode graphs; the dense gemma4 arm counts `lazy_graph_direct{not_profitable}` equal to its decode graph count with `seg_replay == 0` (r7 I-2), a pre-registration first checked against a node dump of its decode graph (the lazy `GET_ROWS` at node 0, its first reader at node 1 or 2); the synthetic dense arm, which has no `FLASH_ATTN_EXT` and whose lazy first reader has at least 3 nodes on each side, replays: `seg_replay > 0`, with `graph_segments` 2 and `direct_segments` 0 on its `[SYCL-SEG-POLICY]` line, and the same at every FA mode, since it has no FA node; decode with `GGML_SYCL_FLASH_ATTN_GRAPH_ALLOW=0`: the gemma4 arm counts `lazy_graph_direct{fa_not_engaged}` equal to its decode graph count with `seg_replay == 0`, and the MoE arm replays as today, its FA segments direct; at the default `AUTO`, gemma4 may engage, since its D=512 global-attention layers use the verified `D512_TILE` kernel (`ggml-sycl.cpp:106123`), but only after an observation: `all_verified_safe()` needs a non-zero observed count (`common.hpp:7204-7206`) and engagement also needs `coverage_ok` (`ggml-sycl.cpp:106136-106138`). So the arm runs a warm-up decode first, asserts the observation counts are above 0, and scores the decodes after it from the counters, never from the once-per-process INFO `[SYCL-GRAPH]` lines (*§D12h* m-5): engaged (observed > 0 and `coverage_ok`) gives `lazy_graph_direct{not_profitable}` equal to the scored decode graph count, otherwise `fa_not_engaged`; PP: `lazy_graph_direct{pp}` equals the PP graph count. Every arm's rows equal CPU `get_rows`, and its greedy tokens equal the same model's run with the table resident. **The cap arm**: a synthetic dense graph with no `FLASH_ATTN_EXT`, run at `GGML_SYCL_FLASH_ATTN_GRAPH_ALLOW=1`, with more lazy boundaries than `MAX_PROFITABLE_SEGMENT_GRAPHS` allows counts `lazy_graph_direct{not_profitable}`; its RED is the early return still keyed on the `MUL_MAT_ID` count (today's `:97242`, with `dispatch_indices` from `:97205`), which returns before the cap and admits it (r7 m-5); (7) the §7.4 consumer-copy submit check, per table, with its `hold` and `hold_blocking` controls on a two-table fixture, after at least 8 warm ubatches (r5 m-8), where the guard witness never fires although two tables prefetch per ubatch (*§D12f* m-2); (8) **the SYCL form's registry check** (r26 M-2), GPU, run by the lead on the merge unit's tree: T6 (l)'s model, card and vehicle (gemma4 E4B, `/models/stock-gemma-4-E4B-it.Q8_0.gguf`, on the B70, `level_zero:0`, `-ngl 99`, `llama-completion -no-cnv -v`, (l)'s fixed prompt and its answer run), with no `GGML_SYCL_VRAM_BUDGET_PCT`, so that layer 0 stays on the card and the consumer is SYCL-executed, both contexts with (l)'s VOID guard, under `ON` and under `--lazy-mode off`. Engagement, or the run is VOID, which counts as failed: the `ON` run's `[PLACEMENT] lazy` line prints `consumer_layer=0` and a SYCL `device=`, with §8.1's SYCL-executed row for the residency it prints; and the `off` run holds the table on the card, read as in T6 (l) from the two runs' settled `[PLACEMENT]` blocks, with every per-layer line equal and, `off` minus `ON`, exactly the table's one entry on the device side (the `NORM/EMBED` `device=` count 1 more, and its device MB, the `Totals:` device weight MB and the `Total:` device MB each more by the table's size to within 0.1 MB) and every host field equal, since that is the case in which the loader registers the table in the `off` run. Pre-registered: each `[S1-PRELOAD] N host-registered weights found for preload` line of the `ON` run reads one less than the `off` run's matching line, and both runs print it, or the check is VOID; no `[LAZY-PLAN-BUG]` line; rc 0; and the answer scores 1 with (l)'s scorer and controls. Its RED is the lazy tensor bound through a real SYCL buffer on the consumer's device, both the SYCL buffer interface and its `ggml_backend_sycl_buffer_context`, so that `ggml_backend_sycl_buffer_init_tensor` reads a real context and device (`e2461d4fb ggml/src/ggml-sycl/ggml-sycl.cpp:26597`, `:26603-26606`); a mutant that swapped only the interface function would leave the descriptor's own context there, read a garbage device, possibly register nothing, and pass the "one less" count, so it is not this RED (r27 n-3). Under the real RED, the table enters the registry, the two counts are equal, and S1-PRELOAD's host-planned branch copies `ggml_nbytes` from the sentinel (`:34704-34724`); (9) **block-exec declines a lazy leaf** (r28 m-3, r29 m-2, r30 m-1, r31 m-1), a fixture, lead-run and GPU-serial as T6 (n) is, in a test built with `GGML_SYCL_PRIVATE_TESTING=1` against `ggml-sycl-private-fixtures`. A hook T7 adds in a new `#if defined(GGML_SYCL_PRIVATE_TESTING)` region after `ggml_sycl_block_exec_prepare_context`'s definition (`e2461d4fb ggml-sycl.cpp:88755`), since the hook constructs a `block_exec_execution_context` (`:87024`), calls that static function with `log_fail` true and two synthetic `placement_plan::layer_block`s, layer 0 and layer 1. The function derives each node's block from the node's layer index and the blocks' ranges (`:88772-88799`, `:87086-87125`), so the fixture supplies node names, not a block assignment. **Precondition:** each block has its own layer-indexed node, in ascending layer order, with only a forward edge from block 0's node to block 1's, so `active_blocks == blocks.size()`, nothing interleaves, the edge check passes (`:88806-88824`, `:88857-88862`, `:88884-88890`), and the missing-leaf loop is reached; block 1's node reads a descriptor-bound leaf. A rejection with reason `invalid-graph-or-block-count`, `active-block-layout` or `non-forward-adjacent-edges` means the precondition failed, and the run is VOID, which counts as failed. It scores the existing reason line and adds no statement inside `prepare_context`: `[SYCL-BLOCK-EXEC] prepare_context rejected: <reason>` (`:88762`, printed whenever `log_fail` holds), an `fprintf(stderr)` line that never reaches a log callback. So the fixture captures file descriptor 2 around each call: it saves the original with `dup(2)` once, calls `fflush(stderr)` before each `dup2` in both directions, and restores fd 2 in a scope guard on every path, so a throw or a failed assertion reports to the real stderr, not into the capture file (r31 n-1). The WARN is a `GGML_LOG_WARN` line, counted through a callback installed with `ggml_log_set`. **The leaf cap stays in force, below the table** (r31 m-1): before its first call the fixture sets `GGML_SYCL_BLOCK_EXEC_MAX_PREP_LEAF_MB=1`, read once and cached (`:88597-88607`), a cap of 1048576 bytes, and its descriptor-bound table is F32 of 256 x 2048, whose `ggml_nbytes` is 2097152; the fixture asserts 2097152 > 1048576 before the first call, or the run is VOID. It calls the hook twice. Pre-registered: each call returns false and prints `prepare_context rejected: lazy-table-leaf`, and the WARN naming the fixture's table is counted exactly once over the two calls. That reason alone places the decline before the leaf-bytes check (`:88902`) and the output-allocation probe (`:88912`), since a decline placed after the check would print `leaf-materialize-too-large`, so no zero is scored. Its RED is the allocate-the-span mutant, the lazy leaf left in `missing_leaves`, under which the 2097152 bytes exceed the cap and the call returns cleanly with `prepare_context rejected: leaf-materialize-too-large` (`:88909`), or any other reason outside the VOID set and not `lazy-table-leaf`, before any allocation. No GPU arm is kept, because no default-config vehicle reaches block-exec: it is opt-in (`GGML_SYCL_BLOCK_EXEC_EXECUTE`, `:88576-88582`), and even then `prepare_context` is reached only with at least 2 `candidate_layer_blocks` (`block-exec-gate.hpp:54-72`; `:89389-89407`), which only `compute_multi_device_plan` fills, for a MoE model on more than one device with no direct peer path (`unified-cache.cpp:29637-29638`, `:30448-30461`); gemma4, which (8) loads on one card, is dense and has none. A GPU arm would also need a lazy consumer node whose name carries a layer index, because the missing-leaf loop skips every node outside a block (`:88832-88834`) | if one of HT-1..HT-5 fails there is no in-design fallback: it goes to the lead before anything else is built. HT-6's result goes to the lead whatever it is, and if it resets, T7 builds continuation submission (§7.2) |
| T8 | counters (unconditional), their definitions (§10), and CPU-only unit tests, with both hash harnesses of Appendix B as fixtures | T5, T6 | a counter test where a repeated row in one ubatch counts as a join and a hit, and a same-block different row counts as a miss with no extra read | |
| T9q | Qwen acceptance against §8.2-8.3: H, S-cold, S-warm, S-hot, the correctness gate, the positive and negative controls, the engine ratio (GPU) | T7, T8 | an `--lazy-mode off` run must fail the engagement check (the negative control proves the check can fail) | gated by §4.5's re-pin of the host-fit figures, made on the merge unit's tree before any T9q arm, since its anonymous-copy figures, `MemAvailable_pre` included, do not hold once C4 maps host weights (r27 n-1); every T9q arm is a 177 GiB Qwen3.8 load, so *lead ruling §M56(b)* binds it: it runs only on a tree that contains 5efe C1-C4 and th32 T6 and T7, the merge unit (T6), and is otherwise VOID-precondition, not FAIL and not skipped-as-pass; and it runs under a watchdog that samples `MemAvailable` every 5 s and aborts the arm as VOID-host below 40 GB (r26 M-4) |
| T10 | `peek_rows` lookahead (class 1, into the row cache only; §5.4, §7.2) | T9q | a lookahead fixture where rows missed by ubatch `k` and `k+1` are read once; and, with every ring entry's copy held, lookahead for `k+1` still issues its class-1 reads and writes no slot, no ring entry and no marker (*lead ruling §T8*, I-3) | required for DeepSeek (D9) |
| T9d | DeepSeek acceptance against §8.4: S arms, executor arms A/C, the §7.4 consumer-copy submit check per table with its `GGML_SYCL_LAZY_CONTROL` arms, bucketed decode, the runtime-vs-simulation match, inside `9g22`'s P0-P7 (GPU) | T10, the `deepseek41` runtime (`9g22` T6, T11), `9g22` gates 1-4, **`llama.cpp-s4ip`** (lever (s), the ring and RUNTIME scope included), **`llama.cpp-yl3v`** (the cold set on STORAGE) and **`llama.cpp-7v5c`** (its size at launch) | the §8.4 witness must fail on an arm run with `GGML_SYCL_DISABLE_GRAPH=1` (record mode off: `seg_replay == 0`) | its S arms run with the §D4 cold set resident on STORAGE, and record the concurrent expert IO (§8.1) |
| T11 | optional: `io_uring` engine | T9q | engine ratio on the ext4 NVMe reference must exceed the pool's | |

**Existing tickets this design depends on or leaves open:**

- `llama.cpp-rnh3`: exact host weights plus named terms; T2 depends on it.
- `llama.cpp-sgc5`: every participating cache builds a full host zone set,
  WEIGHT included, from one plan (§2.3). T2's enforcing phase depends on it
  (§4.2): without it, the sum cap refuses a two-card host-expert load that no
  lever can clear. Part of lever (s) (owner §D4).
- `llama.cpp-s4ip` (comments `c-rbwa`, `c-td91`): lever (s)'s host side, the
  SCRATCH and RUNTIME pool terms with no host reader (`dma_staging_pool` and
  the ring, each twice) and the ring's MXFP4 type gate, which also frees the
  ring's VRAM. `dma_staging_pool` is removed entirely, with no device charge
  (*lead ruling §D6*, I-1). T2 counts whatever the sizing code commits,
  before or after it; T2's enforcing phase and T9d wait for it.
- `llama.cpp-7v5c` (9g22's T27): the host-only plan dry-run, the one
  computation behind `host_planned_bytes`, `expected_host` and the cold set
  `C` (§4.2, §4.3 step 5; *lead rulings §T3, §D5*).
- `llama.cpp-moua`: lifetime segregation and the shared publish struct; T2
  phase (b).
- `llama.cpp-zhcn`: the `SYCL_Host` compute buffer as a planned host tenant;
  T2 phase (b).
- `llama.cpp-n77l`: the upstream merge that brings `TENSOR_READ_LAZY`; T6.
- `llama.cpp-x1xd`: not on this design's path (§5.3); it stays open for the
  paths that do reach it.
- `llama.cpp-jhxq`: its split-input copy sites are not on this design's path
  either, and its site (4), the device buffer's `set_tensor` drain, is not
  the ring's guard: no graph input reaches it (§5.3, r6 C-1, *lead ruling
  §T9b*).
- `llama.cpp-30h4`: the SYCL host buffer's sync before each input write, the
  guard that bounds the epoch ring (§5.3). th32 relies on it and adds no wait
  (*§T9b*).
- `llama.cpp-0j5w` (§M35): when the execution queue becomes per context, the
  host buffer's sync must wait on the executing context's queue, with a RED in
  which `ggml_sycl_plan_input_raw_guard_present()` refuses a context whose
  sync queue is not its executing queue (*§T9b*, the lead's comment on 0j5w).
  Until it lands, a teardown `synchronize()` also waits for other contexts'
  copies on the same device (§5.3).
- `llama.cpp-54sf0`: owns the MoE path's existing submitter waits (§7.2).
  Nothing in this design relies on them any more, so the dependency edge that
  *lead ruling §T7* 1 asked for is **withdrawn** for th32 (*§T8* I-1).

**Relays to `9g22`** (through the lead, *lead rulings §T8, §T10, §D12, §T12,
§D12e, §D12f, §D12m, §D12n, §D12p, §D12q*). This document is the one statement
of the rules the two designs share, and `9g22` cites it rather than restating
them (*§D12m*, *§D12n*). th32 pins `9g22` at two revisions, each for its own
part (*§D12j*): rev 17 (`076757b6f`) for the shared rules, and rev 10
(`054069548`) as the anchor for the figures this document quotes. Rev 17 pins
th32 at rev 2m and has, checked there (most of it since rev 14a): the
device-written cancellation marker, the backend cancel step before the flush
with the failed-graph mark and at a throwing `synchronize()`, `D_cold` =
(`in_flight_depth` + 1) x 40 (*§D12a*), the removal of its slot-overrun abort,
the marker word, the `(epoch, deadline)` key, the shared predicate
`ggml_sycl_plan_input_raw_guard_present()` refusing whenever the queue objects
differ, with its cause classified as `tp`, `device`, `pre-cache` or `other`,
llama's guard declaration (*§T9e*), the input guard's sync in its list of
non-graph watchdog scopes, `llama.cpp-uwlx` as the accessor's sole definer
(*§G1b*, *§G1c*), *§D12d*'s mark with its scope and three consumers, and every
shared item of *§D12f*: the three-case partition by `copy_enqueued`, with
throw-free host stores before the marker enqueue, and the drain as the flag and
the re-post with nothing enqueued (I-1, *§D12e*); the mark arms with reads held
and the backstop off, the site labels `catch`, `sync`, `entry`, `teardown` and
`backstop`, and `cold_mark_consumed` with `backstop` among its sites (I-2); the
witness once per ubatch per family with its own snapshot, and its full-bypass
RED with the engagement check (m-1, m-2); SYCL backend device 0 as the host
buffer's device (m-3); the queue seam with kinds TP and FOREIGN (m-5); both
strict forms with `cold-input-unguarded` (m-6); the watchdog scope engaged only
for a lazy or cold context, with a Mistral regression arm and a cold-plan
positive control (m-7); the fault fact per device only for device loss, with a
queue-refused marker submission as a poison source (m-8). Rev 15 adds, checked
there: the delimiter `ggml_backend_sycl_ubatch_begin(backend, k, guard inputs)`,
with its `cb_eval` and CPU-island GREENs (*§D12g* I-1, *§D12k* 3); the
family-prefixed `cold_epochs_cancelled{site}` and `cold_epochs_drained{site}`,
one meaning each (*§D12h* I-2, *§D12j* m-7); the cold plan flag carried on every
host compute buffer, the re-baseline to 0 on a new buffer, and a rebuild arm
with a stamp-once RED and a cross-buffer RED (*§D12h* I-3, *§D12k* 2); the
classifier by `strstr` at the catch, with both controls (*§D12h* m-2); the
poison check over every scope epoch the step acted on, with a drained sub-arm
(*§D12i* m-8); and a non-throwing re-post into a queue sized at the context
transaction to the sum of `D_cold` over its strands. Rev 17 adds, checked there
against 2m's relay list: the cold witness's stamp check with th32 T7's two
unstamped-guard REDs; the carrier and the classifier cited from th32 §5.3;
th32's enumerated scoped sites with `buf_output` unstamped by design; the
transaction-run rule with `[COLD-PLAN-BUG] plan flag changed`; and its own r15
minors. Rev 16 added every item th32 2l left open for it, checked there against
2l's relay list: the failure contract, -3 before `graph_compute` with no latch;
the call on every SYCL backend with per-backend `cold_witness_checks` and
`cold_witness_rebaselines`; the `llama_opt_init` refusal with a RED; the stamp
and registry cited from th32; the toggle order with the `>` comparator and 8 ×
`s` against `s`; the classifier cited from th32; no fork after SYCL init; th32's
`D` terms in its re-post bound; the join source held until every waiter's
delivery; and th32's line names, `ubatch delimiter missing`,
`lazy-plan-training-unsupported` and `lazy-hook-unresolved`. Earlier rounds also
closed th32's relays: a failed host buffer sync poisons through th32's fault
fact (r7 m-9 (b)); a join re-keys only within a context, and a join from another
context adds its own entry at its own key (r7 m-7, *§T12* I-C); its IO ordering
and heartbeat are th32's, with no rule of its own (*§T10* I-6 (a)); HT-1..HT-6
are th32's lead-run HTs, which it cites; its site row 2 has the HOST output in
the `SYCL_Host` buffer; its backend RED keeps the throw between the hand-off and
the first reader, with a sub-arm after the first reader; its r9 m10 relays (a)
and (b) are in th32, the cold first reader in T7's boundary set (§7.4) and T5's
test-only FIFO override (§5.2.1), and its (c), the cold arm, is yl3v's
acceptance, on th32's fixture (*§D12f* m-10). Open for it from revs 2n, 2o and
2p, each a rule stated once in this document (*§D12q*, *§D12s*, relayed by the
lead after each fold):
- re-pin th32 at rev 2p; the three th32 facts it checks against 2m, the -8.9
  to -6.0 quote, the T2 phases and the §5.2.1 FIFO override, are unchanged
  in 2n, 2o and 2p;
- **the carrier's home and sentinels** (*§D12q* I-2): the cold post
  (posted `k`, `consumed`) lives on the backend's persistent per-context
  cold hand-off state, never on `llm_graph_result`, which is rebuilt at
  `src/llama-context.cpp:2733` and `:4798`, with the `none` sentinels stated
  in §5.3 and the end point below, so its training RED fires through the k
  branch deterministically (*§D12s*);
- **stale posts** (*§D12p*): th32 takes option (a), so the cold check runs
  at every cold-bearing layer's hand-off on every ubatch, pick or not, and
  every post is consumed or reported; its RED is the no-pick shape, the
  uncalled graph's first hand-off at a layer with no pick in the prior
  ubatch, which must print the line; this replaces its interim arm-level
  closures;
- **the end point** (*§D12s*): every delimited ubatch's end sets each cold
  post's posted `k` to `none`, on the success and the failure returns alike,
  from an end point that sits only on the delimited path, the one that runs
  `ubatch_begin`, and never in `graph_compute`, the hand-off, or any path an
  undelimited graph runs, naming `opt_epoch_iter` as a path that does not
  pass through it; its RED is a failure before the hand-off followed by an
  undelimited graph, whose line must name the table whose post was never
  consumed, and its placement mutant, the end point as a re-arm at
  `graph_compute`'s entry, must make its training RED silent;
- **death-test logging** (r13 m-3): a death test scored on a line runs on
  llama's synchronous default log sink with no `common_init()`, or the line
  is written to `stderr` and flushed before the abort, and it is scored on
  rc 134 plus the line, never on a counter;
- **the end point's contract** (r14 m-2, m-3, m-5): `ubatch_end` is `noexcept`
  and takes no lock teardown can hold, is refused by name when unresolved, and
  is counted against `ubatch_begin`; no cancel scope reads the posted `k`; the
  post-build scan's set equality runs both ways; and a death test cites the
  default sink of whichever logger prints its line;
- **the op property** (r13 m-5): its training RED's pinned parameter is
  justified by the property that every op downstream of it has a backward
  case, stated with its own instance of that set;
- **the k branch** (*§D12q* I-2): a direct-drive control that posts the next
  `k` with no `ubatch_begin`, whose RED is a reader that checks only
  `consumed`;
- **the classifier's observables** (*§D12q* I-1): the helper's
  `[SYCL-FAULT] device lost` line and `device_lost_faults`; class 2 scored
  as a death test; the always-recorded control and its attributed-only RED
  at the unstamped `buf_output` sync, followed by a decode that fails
  `lazy-context-poisoned` (the cold lane's form is the same with its own
  poisoned status); class 4 entered through a non-device-lost exception; the
  ambient guard declared before each `try`; the non-throwing device lookup
  (r12 m-3 to m-5).

**Relay to `zhcn`:** split `ggml_sycl_node_is_host_dispatched()` into "sources
not staged" and "intercepted by the funnel" before T7 (§7.2, r4 m-12).

**Relay to the owners of the input guard** (*lead rulings §T9a, §T9b*): any
change to the host buffer's sync, `llama.cpp-0j5w`'s included, must keep it
waiting on the queue lazy work is submitted to, compared as a queue object
(*§T9c*), must keep its non-graph watchdog scope and its sync counter, and must
publish the context's in-flight ubatch depth as a planned quantity, which
th32's `D` reads (§5.3).

**Alongside, outside the table:**
- `llama.cpp-yl3v`: cold routed experts on STORAGE, executed by the CPU,
  brought forward to Q2_K phase 1 by owner §D4 (§9.2). It uses T5's engine
  and T2's ledger through the interface of §9.2; T9d waits for it. Its
  acceptance carries the cold arm of the guard fixture (`9g22`'s r9 m10 (c),
  *§D12f* m-10): T7 (5)'s shape (a) on a cold-set fixture with no lazy table,
  the cold reads held the same way; GREEN, no `[COLD-PLAN-BUG] ring exhausted`
  and the buffer's sync count rising per ubatch; its RED, the direct drive at
  a test-only `D_cold`, is `9g22`'s. It reuses th32's fixture, since T7 does
  not depend on yl3v.
- The census of CPU-split outputs that SYCL reads zero-copy (`inp_embd`
  first), which the lead files under §T2 DQ. This design adds no member to it
  (§7.3).

**Missing tickets from r1, now in the table:** `mmap_support` independence (T6
RED a), the `mlock` policy (T1), llama handling of the descriptor buffer (T6b),
and the host-task probes (T7). r1's T8 is split into T9q and T9d, with T10
between them, so that nothing depends on a later ticket.

---

## 12. Not done, and why

- **MTP-speculative prefetch.** The MTP head could predict token `t+1`, and
  its rows could be prefetched during token `t`. That turns decode's cold stall
  into a speculation-hit question. It is worth a ticket once S-cold decode is
  measured, not before.
- **A static hot set shipped with the model.** It would need a corpus-derived
  frequency list per model. The CLOCK cache reaches the same place after a few
  thousand tokens.
- **Caching dequantized F32 rows.** That would take 3.8-12x the memory to save
  a ~2 ms dequant per PP ubatch.
- **Placing the table on the GPU.** It is forbidden by P3, and pointless: 84-170
  B random rows are a CPU-shaped access.
- **Regrowing a yielded row cache between plan publishes** (§6.4). It needs a
  trigger the ledger does not have; v1 regrows at the next publish.
- **A scheduler event edge for the zero-input split sync** (`llama.cpp-x1xd`).
  This design no longer reaches that sync (§5.3), so it is not a dependency;
  the ticket stays open for the paths that do.
- **The cold-expert path itself.** It is in phase 1 (owner §D4), but it is
  `llama.cpp-yl3v`'s; this design names its interface (§9.2) and records the
  measurements it starts from.

---

## 13. Decisions

Each open decision offers options, and the recommended one is marked
**(rec)**. Recommendations are this design's proposals. A decision that is
already made says by whom: *lead ruling* with its section for the lead's
rulings in `lead-rulings-2026-09-26.md`, and "owner §D1/§D2/§D4" for the
owner's.

**D1. How is the HOST residency populated? Ruled (a)** (*lead ruling §T2*).
- a) **(ruled)** Eager: at load, a sequential `O_DIRECT` read of the whole table
  into the locked body, blocking model load. It is simplest, H is H from the
  first token, and a failure (lock or read) surfaces at load. r1 quoted
  "~31 s" for it; that read was NVMe-cached (A.2: 19.7 of 27.6 GB from the
  NVMe). **On settled data it is minutes**: 50.66 GiB at the 0.29-0.47 GB/s
  measured on a settled split (§9.2.2) is ~2-3 minutes, the same order as
  loading the host experts from the same filesystem.
- b) Background fill: the load returns at once and the table fills behind it,
  served through the STORAGE path until each chunk arrives. It would make the
  HOST residency carry the whole STORAGE machinery plus per-chunk ready bits,
  for a one-time saving.
- c) On-demand only. A HOST table would never become the H arm.

**D2. IO engine? Ruled (a)** (*lead ruling §T2*). Owner §D4 adds a second
client, `llama.cpp-yl3v`'s cold-expert slice reads, through the same pool
(§5.2).
- a) **(ruled)** A portable `pread` worker pool first, with priority classes,
  per-context fairness and in-flight dedup (§5.2). It measured at the device
  ceiling (ratio 0.99) and adds no dependency.
- b) `io_uring` through raw syscalls only.
- c) Add a liburing dependency.

**D3. Where the tables live. A fact, not a choice** (*lead ruling §T1*, D3).
Plain NVMe is **not** an option on this host. `/` (ext4 on NVMe) has
**33-38 GiB free** at 97% use (34 GiB at 19:00 today), and the tables come in
sets: **60.08 GiB** for the DeepSeek Q2_K pair, **78.68 GiB** for the Q3_K_M
pair, and **50.66 GiB** for Qwen's table. None fits, and a per-table comparison
would invite a half-measure on a nearly full system disk (M1). The tables stay
on bcachefs as it is; every S target in §8 is that filesystem's. The one
filesystem lever left, per-file bcachefs targets on the engram shards, is the
owner's filesystem policy, and this design does not propose it for v1.

**DLM. `--lazy-mode` under SYCL? Ruled (a)** (*lead ruling §T2*, where it is
called D4; renamed here so that "§D4" means only the owner's decision).
- a) **(ruled)** Keep upstream's meaning as *eligibility* (`OFF` = ordinary
  weight, `AUTO` = marked and larger than 4 GiB, `ON` = all marked). The
  planner alone picks HOST or STORAGE for eligible tensors, and the SYCL branch
  does not inherit upstream's `AUTO`->`OFF` fallback on `mmap_support` (§4.1).
- b) Also let `OFF` force HOST and refuse STORAGE.
- c) Ignore the flag under SYCL.

**D5. Where is each gather placed? Ruled** (*lead rulings §T1* C2, *§T5*
DQ-C2): the provider decides. §T1 C2 put HOST gathers in the leading
`inp_embd` split as a CPU op; §T5 supersedes that half, so both residencies
head the consumer device's first split as a SYCL-owned op (§7.1). §8.4's
arms A and C test the STORAGE executor with a fixed decision rule. r1's
"expand before each consumer block" is withdrawn.

**D6. Who executes the STORAGE gather? Ruled** (*lead ruling §T1*, D6(c)): the
SYCL backend, as CPU work with event edges, no scheduler split, no host wait;
its mechanism is D6-M. th32 owns the op; `llama.cpp-9g22` consumes it through
§9.1. The record-mode question the ruling left to this design is answered in
§7.4: a stated segment boundary, not a recorded host-task node.

**D6-M. The mechanism under D6(c). Ruled (b)** (*lead ruling §T8*, amending
*§T2*'s "(a), with (b) as the fallback"; *§D6a* makes (b) mandatory for
`9g22`'s cold path, and §T8 makes it the one mechanism for every lazy-table
refill, every model and every depth). §7.2 pre-registers six probes
(HT-1..HT-6) of the interop it needs; HT-6's result goes to the lead.
- a) **Superseded** (*§T8*): `host_task`. Under rev 2d it needed invariant L,
  which dense graphs, recoverable failures and lookahead each break (r4 I-1,
  I-2, I-3).
- b) **(ruled)** The IO engine's strand refills and dequantizes (for a HOST
  table from the body), then signals a host-signalled Level Zero event
  wrapped with `sycl::make_event`, and the output copy depends on it (§5.3).
  It is backend-owned, event-edged, split-free and wait-free on the
  submitting thread. If HT-1..HT-5 fail, it goes back to the lead.
- c) **Rejected** (*lead ruling §T2*): the submitting thread waiting for the
  ticket at the consumer boundary is a host wait. If (b) also fails, that goes
  back to the lead as a new question; this design does not offer (c).

**D7. Size of the STORAGE row cache. Ruled** (*lead rulings §T1* I8, *§T2*,
*§T4* I-4): sized after the context envelope and **yieldable**, since it holds
no leases; in 256 MiB chunks; yielded chunks come back only at the next plan
publish (§T2: no regrow until then). In a plan with a cold set the cache is
**fixed** at its target and charged before the cold set is sized (§T4 I-4);
without one it takes what is left (§4.3, §6.4). For DeepSeek the target is
2 GiB per table with the index, charged in the P3 budget before the cold-set
search and printed by 7v5c (*lead ruling §T5*).

**D8. DeepSeek-V4.1-Flash: the rung and its fit. Decided by the owner**
(owner §D2: Q2_K primary; owner §D4: "Zones + SSD cold experts"). The lead
ruled that §D4 answers the fit and is not re-asked (*lead ruling §T3*). With
§D4 applied:
- **The fit** is case B after lever (s) (§4.5; step 2 plus the fixed caches,
  since case A cannot occur in a cold-set plan), minus the bytes of the cold set
  `C` on STORAGE, plus its landing slab. 9g22's P3 is the binding check: after
  (s) and before any cold set, `expected_host` is 171.1-174.5 GiB and P3 fails
  at every reading taken today (§4.5).
- **The cold-set size** that clears P3 is found at load by
  `llama.cpp-7v5c`'s search down the §D1 profile order, with a landing slab
  of `2 x` the largest per-layer cold bytes (*lead ruling §D5* as amended by
  *§D6*, I-4; §4.3 step 5). Its value at the lowest reading, 171.4 GiB, and
  at 189.6 GiB is `9g22`'s spread and concentrated bands, in
  [its §4.5.2](deepseek41-runtime.md#452-the-host-fit-r2-c-a-r3-c-1-c-2-owner-d4);
  this document does not restate them. It is 9g22's, not a th32 choice.
- **The engram tables share that one P3 budget** (*lead ruling §D6*, I-5):
  routed experts stay host-resident first, and a table goes HOST only if P3
  holds with it HOST and the cold set empty. On this host that needs
  228-232 GiB, so both tables are STORAGE at every reading (§4.5).
- **If even every host expert cold cannot clear P3**, the load refuses as
  `cold-set-cannot-clear-p3`, naming the reading and the shortfall (§D5).
- **Today's code** refuses phase 1 at every reading even for the planner, so
  the gates wait for (s) and yl3v (T9d, §11).

Rev 2a's "goes back to the owner as a re-check" is withdrawn: the owner had
already decided (§T3).

**DQ. The gather's output reaches the device by an explicit copy. Ruled**
(*lead ruling §T2*): the HOST-tier gather output reaches the device by an
explicit event-ordered H2D copy on the main in-order queue, the same path as
STORAGE; the GPU does not read `SYCL_Host` memory zero-copy. The design is
§7.2 step 3 and §7.3, with a RED that fails on a zero-copy read (§7.3, T7).
The general census of CPU-split outputs that SYCL reads zero-copy is the
lead's separate ticket.

**DQ-C2. Where the HOST gather runs under DQ. Ruled (a)** (*lead ruling
§T5*; it supersedes *§T1* C2's placement of HOST in the leading CPU split).
- a) **(ruled)** The one path for both residencies (§7.1, §7.2): the SYCL
  backend owns the HOST gather too; the worker dequantizes from the resident
  body into the output slot, and the step-3 copy moves it. The copy's source
  is guarded on the worker's own thread (§5.3, stage 2), so nothing waits on
  the submitting thread. It is what §T2's "one code path for both tiers"
  reads as, and HOST depends on HT-1..HT-6 like STORAGE.
- b) Keep C2's form: the CPU op in the leading split, and the SYCL backend
  copies its output from the `SYCL_Host` compute buffer before the consumer.
  Nothing orders ubatch `k+1`'s CPU op after ubatch `k`'s copy in a
  multi-ubatch prefill (§7.1), so the CPU op would have to wait for that copy
  on the submitting thread: a host wait on the GPU, which this design does not
  offer as a baseline. Arm C (§8.4) runs exactly this form, as a comparison
  arm.

**DQ-E. What bounds the epoch ring? Ruled** (*lead rulings §T9, §T9a* and its
follow-up, *§T9b*, *§T10* I-1). The caller's run-ahead is bounded by the RAW
guard on graph inputs, which *§T9b* names: the SYCL host buffer's sync before
each write of `inp_tokens`, `inp_embd` or `inp_pos` (`llama.cpp-30h4`), not the
device buffer's `set_tensor` drain that rev 2f named, which no graph input
reaches (r6 C-1). It gives an in-flight depth of 1, one planned quantity owned
by the guard; th32 reads it. `D = in_flight_depth + 2` (`D` = 3 today), one
entry above the requirement `D >= max(n, 2)` (r6 m-4). Exhaustion is
`[LAZY-PLAN-BUG]`: a STRICT abort through `ggml_sycl_strict_enabled()` (*§G1*,
*§G1a*, *§G1b*), otherwise a WARN and a fatal -3. Rev 2e's recoverable
`lazy-epoch-ring-exhausted` is rejected, because `llama_decode`'s contract has
no recoverable return for it. `ggml_sycl_plan_input_raw_guard_present()` checks
llama's declared guard input at context creation (*§T9e*), the ubatch delimiter
witnesses it once per ubatch through the host buffer's sync count and identity
(*§D12f* m-2, *§D12g* I-1, *§D12h* I-3), and lazy work on a queue the sync does
not wait is refused by name whenever the queue objects differ, its cause
classified as `tp`, `device`, `pre-cache` or `other` (*§T9c*, *§T9d*): the cache
queue is the only accepted queue, and the pre-cache case is checked though
expected unreachable (r7 I-3, m-1). The public-API fixture, llama-server's shape
included, shows exhaustion unreachable on today's code (§5.3, T7 (5)).
`llama.cpp-0j5w` carries the guard to the per-context queue (§11). No host wait
is added.

**DQ-F. The cancelled-epoch window. Withdrawn** (rev 2f; *lead ruling §T11*
accepted its option (a), and rev 2g keeps both). A host-written cancellation
marker would leave a cancelled epoch outside the input guard, since it would
have no device work. Adopting `9g22`'s device-written cancellation marker,
which *lead ruling §D12* ratifies as the one mechanism of both designs, gives
every cancelled epoch one command on the queue, which the guard drains like a
copy. §T11's rule stays: the stage executor runs cancelled and draining stages
first. Re-derived under *§T9b* (r6 m-3): a failure in ubatch `k` comes after
`set_inputs(k)`'s input sync drained every copy of ubatch `k-1`, so the
cancelled epoch's stage is the strand's next, one stage pickup away, and its
marker follows one device command later, on every path the predicate admits
(§5.3).

**D9. Lookahead in v1 for STORAGE-bound models? Ruled (a)** (*lead ruling
§T2*).
- a) **(ruled)** Yes for DeepSeek: T10 lands before T9d. Its tables are always
  on STORAGE here, and a cold PP512 ubatch is ~2.5-3.5 s of IO at the
  measured rates (§0; §8.4's S-cold bound allows up to 4.4 s) that, without
  lookahead, only `blk.14`'s window can partly hide.
- b) Keep lookahead as phase 2 for every model.

---

## Appendix A. Raw measurements (2026-09-26, this host)

Probe sources and logs are in the session scratch
(`<scratchpad>/th32/{io_probe.sh,mmap_probe.cpp,batch_probe.cpp,ram_gather.cpp,qtok.py,ple_hash.py,ple_sim.cpp,ple_sim2.cpp,upstream_set_input_body.inc}`
and `io-logs/`, `sim-logs/`). That scratch is not durable, so the load-bearing
lines are reproduced verbatim below.

### A.1 GGUF metadata (gguf-py under `/usr/bin/python3`)

```
qwen4exp.block_count [48]
qwen4exp.expert_count [512]
qwen4exp.expert_used_count [10]
qwen4exp.ple.layers [1]
qwen4exp.ple.ngram_size [3]
qwen4exp.ple.heads_per_ngram [8]
qwen4exp.ple.conv_kernel [4]
qwen4exp.ple.eos_token_id [248044]
qwen4exp.embedding_length_per_layer_input [160]
qwen4exp.ple.layer_multipliers [23703573157769, 20109073645365, 8052911324071]
qwen4exp.ple.head_offsets [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297, 160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275]
qwen4exp.ple.head_vocab_sizes [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]
tokenizer.ggml.pre qwen35
tokenizer.ggml.eos_token_id [248046]
qwen4exp.ple.image_token_id [248056]
2 Qwen3.8-Flash-Next-Q8_0-00003-of-00006.gguf per_layer_token_embd.weight Q8_0 [160, 320001536] nbytes 54400261120 data_offset 192
$ ls -l .../Qwen3.8-Flash-Next-Q8_0-00003-of-00006.gguf  -> 54400261312
```

DeepSeek-V4.1-Flash Q3_K_M (`ds_meta.py`; shards 1-4 of 9 present at the time):

```
deepseek41.block_count [40]
deepseek41.embedding_length [5120]
deepseek41.expert_count [384]
deepseek41.expert_used_count [6]
deepseek41.expert_feed_forward_length [2304]
deepseek41.expert_shared_count [1]
deepseek41.engram.head_count [8]
deepseek41.engram.key_length [256]
deepseek41.engram.max_ngram_size [4]
deepseek41.engram.layer_ids len 2 [1, 14]
deepseek41.engram.multipliers len 8 [76632096046245, 4839876093313, 35959672319349, 73987337458391, 67716810739261, 51510806800915, 30921347202721, 82619226485591]
deepseek41.engram.primes len 48 [16000057, 16000079, 16000081, ... 16000889]
deepseek41.engram.offsets len 48 [0, 16000057, 32000136, ... 368005705, 0, 16000477, ... 368015793]
deepseek41.engram.token_map len 129280 (compressed vocab 99092)
deepseek41.engram.pad_id [2]      token_map[2] = 2
tokenizer.ggml.pre joyai-llm
DeepSeek-V4.1-Flash-Q3_K_M-00002-of-00009.gguf blk.1.engram_embd.weight Q3_K [256, 384006168] 42240678480 off 480
DeepSeek-V4.1-Flash-Q3_K_M-00002-of-00009.gguf blk.1.engram_k.weight Q3_K [5120, 4] 8800 off 42240678976
DeepSeek-V4.1-Flash-Q3_K_M-00002-of-00009.gguf blk.1.engram_q.weight Q3_K [5120, 4] 8800 off 42240687776
DeepSeek-V4.1-Flash-Q3_K_M-00002-of-00009.gguf blk.1.engram_wkv.weight Q3_K [6144, 25600] 67584000 off 42240696576
DeepSeek-V4.1-Flash-Q3_K_M-00003-of-00009.gguf blk.1.ffn_down_exps.weight Q5_K [2304, 5120, 384] 3114270720 off 10240
DeepSeek-V4.1-Flash-Q3_K_M-00004-of-00009.gguf blk.7.ffn_up_exps.weight Q3_K [5120, 2304, 384] 1946419200 off 11200
```

**Rev 2: DeepSeek-V4.1-Flash Q2_K and Qwen3.8 Q8_0 weight census** (GGUF
tensor headers only; `ds_q2k_census.py`, `qwen_census.py`):

```
ENGRAM DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf blk.1.engram_embd.weight Q2_K [256, 384006168] 32256518112 off 5785327904
ENGRAM DeepSeek-V4.1-Flash-Q2_K-00004-of-00007.gguf blk.14.engram_embd.weight Q2_K [256, 384016682] 32257401288 off 4032
output                2 tensors        542996480 B     0.506 GiB
token_embd            1 tensors        217190400 B     0.202 GiB
dense_other         921 tensors       2470050496 B     2.300 GiB
routed_experts      120 tensors     196765286400 B   183.252 GiB
engram_table          2 tensors      64513919400 B    60.083 GiB
total 246.34361562877893
max non-engram tensor 1946419200 blk.0.ffn_down_exps.weight 1856.25 MiB
expert bytes per layer (GiB): 4.581 for each of layers 0-39
expert types {('ffn_down_exps', 'Q3_K'): 40, ('ffn_gate_exps', 'Q2_K'): 40, ('ffn_up_exps', 'Q2_K'): 40}
---
output                4        682434560 B     0.636 GiB
ple_table             1      54400261120 B    50.664 GiB
token_embd            1        675430400 B     0.629 GiB
dense_other        1074       4110179840 B     3.828 GiB
routed_experts      144     128345702400 B   119.531 GiB
total 175.2879548072815 max non-PLE 850.0 MiB blk.0.ffn_down_exps.weight
```

### A.2 Storage probes (read-only; each run <= 25 s)

fio 3.41, `--readonly --rw=randread --bs=4k --direct=1 --ioengine=io_uring
--offset=4096 --size=54400253952 --runtime=15 --time_based --norandommap`, with
per-device bcachefs `io_done` deltas (dev0 = NVMe, dev1 = ugreen/USB 10G,
dev2 = T7/USB 5G):

```
qd1_4k: iops=1389 MB/s=5.7 ios=20838 lat_mean=719us p50=477us p99=2671us dev_read_MB={0:4135.9, 1:508.4, 2:491.7} amp=60.2x
qd16_4k: iops=7815 MB/s=32.0 ios=117251 lat_mean=2047us p50=1810us p99=6849us dev_read_MB={0:18858.0, 1:4379.6, 2:5291.5} amp=59.4x
qd64_4k: iops=9124 MB/s=37.4 ios=136957 lat_mean=7013us p50=5734us p99=25297us dev_read_MB={0:20256.3, 1:6686.1, 2:6251.0} amp=59.2x
clat percentiles (us): qd1 p90 1171 p95 1483 p99.9 5669 | qd16 p90 3293 p95 4227 p99.9 14221 | qd64 p90 13828 p95 17433 p99.9 34341
```

mmap page-fault probe (`MAP_SHARED`, `POSIX_MADV_RANDOM`, random 170 B rows,
12 s):

```
mmap threads=1 rows=18660 rows/s=1555 mean=643us p50=512us p90=1001us p99=2271us     dev0 read_MB=4320.6  dev1 393.8  dev2 134.0
mmap threads=16 rows=94462 rows/s=7866 mean=2032us p50=1746us p90=3394us p99=7267us  dev0 15787.6  dev1 3675.6  dev2 4388.8
mmap threads=24 rows=88470 rows/s=7370 mean=3254us p50=2565us p90=5576us p99=17660us dev0 15123.3  dev1 3858.5  dev2 4064.5
```

The design's read path (`batch_probe`: B random rows, 4/8 KiB `O_DIRECT` pread
by Q workers, caller waits for the batch):

```
batch=1 workers=1 batches=11969 mean=0.84ms p50=0.65ms p90=1.29ms p99=3.55ms max=34.91ms rows/s=1197 straddle=4.2%
batch=8 workers=8 batches=2350 mean=4.26ms p50=3.78ms p90=6.23ms p99=15.58ms max=139.98ms rows/s=1880 straddle=4.0%
batch=16 workers=16 batches=3997 mean=3.75ms p50=3.17ms p90=5.31ms p99=13.02ms max=93.37ms rows/s=4264 straddle=4.1%
batch=7100 workers=64 batches=30 mean=850.89ms p50=784.79ms p90=1027.57ms p99=1187.22ms max=1268.40ms rows/s=8344 straddle=4.1%
batch=7100 workers=128 batches=27 mean=942.54ms p50=810.98ms p90=1252.12ms p99=1316.69ms max=1424.02ms rows/s=7533 straddle=4.1%
```

The batch-8 row is slower than batch-16. The per-row rates across these runs
also moved with the ambient IO of other tenants (load 39-57), which is why the
targets carry headroom over the means.

Sequential (whole-table load estimate):
`seq 1MiB QD8 O_DIRECT: 1779 MB/s` (dev0 19761.9 MB, dev1 3723.0, dev2 4121.0
over 12 s).

DeepSeek `blk.1` table, shard 2, the same fio probe with
`--size=42240000000` (`io_probe2.sh`; load average 90-92; the download was
writing shard 4 throughout). Device read deltas in MB (dev0 NVMe, dev1
ugreen, dev2 T7) and bcachefs reconcile's write delta to the T7:

```
ds_qd1_4k:  iops=1014 ios=15210  lat_mean=984us  p50=522us  p90=1221us  p99=7700us  p99.9=35389us  dev_read_MB={0:6800.7, 1:319.8, 2:15.6}   dev2_write_MB=2441.1 (reconcile: amp not attributable)
ds_qd16_4k: iops=5153 ios=77303  lat_mean=3104us p50=2073us p90=5865us p99=18743us p99.9=42205us dev_read_MB={0:12673.4, 1:2818.1, 2:3555.6} amp=60.2x
ds_qd64_4k: iops=8400 ios=126063 lat_mean=7618us p50=6390us p90=14877us p99=28180us p99.9=39059us dev_read_MB={0:19166.8, 1:5972.8, 2:5833.2} amp=60.0x
```

`batch_probe2` on the same extent (`PBASE=480 PROW=110 PNROWS=384006168`):

```
DS blk.1 batch=24 workers=24 batches=1726 mean=6.95ms p50=4.93ms p90=10.99ms p99=27.84ms max=158.02ms rows/s=3452 straddle=2.7%
DS blk.1 batch=48 workers=48 batches=1341 mean=8.95ms p50=7.57ms p90=12.68ms p99=29.13ms max=82.09ms rows/s=5361 straddle=2.5%
DS blk.1 batch=20000 workers=64 batches=11 mean=2563.06ms p50=2186.65ms p90=3251.34ms p99=3251.34ms max=3771.67ms rows/s=7803 straddle=2.6%
```

DSpark-sized batches on the same extent (6 and 3 tokens' rows):

```
DS blk.1 dspark batch=288 workers=64 batches=162 mean=74.09ms p50=75.53ms p90=98.03ms p99=131.12ms max=137.99ms rows/s=3887 straddle=2.8%
DS blk.1 dspark batch=144 workers=64 batches=500 mean=24.02ms p50=21.32ms p90=34.08ms p99=52.49ms max=93.15ms rows/s=5996 straddle=2.7%
```

Cold-expert loads (`expert_run.sh` around `expert_probe`, 12 s each; W loads in
flight, each split into CH chunks read by P threads; per-device bcachefs read
and write deltas in MB; the DeepSeek download was writing later shards during
the DS runs; the reference rows read a Qwen split written 2026-09-12 at the
same slice shapes):

```
DS-expert W=1 P=1 CH=1024KiB loads=53 loads/s=4.4 MB/s=74 load_ms mean=229.2 p50=237.3 p90=386.8 p99=469.3 max=469.3 | dev_read_MB {0: 5495.5, 1: 287.2, 2: 615.9} dev_write_MB {0: 1167.6, 1: 1369.4, 2: 3863.7} | load 34.25, 39.42, 51.91
DS-expert W=1 P=8 CH=1024KiB loads=389 loads/s=32.4 MB/s=549 load_ms mean=30.9 p50=17.2 p90=67.8 p99=126.7 max=187.4 | dev_read_MB {0: 8955.1, 1: 1507.8, 2: 1978.3} dev_write_MB {0: 850.1, 1: 883.9, 2: 2495.0} | load 34.87, 39.30, 51.67
DS-expert W=1 P=16 CH=512KiB loads=851 loads/s=70.9 MB/s=1199 load_ms mean=14.1 p50=12.9 p90=18.4 p99=29.1 max=38.4 | dev_read_MB {0: 16393.3, 1: 3133.5, 2: 3949.7} dev_write_MB {0: 0.0, 1: 0.0, 2: 0.0} | load 36.62, 39.54, 51.61
DS-expert W=2 P=8 CH=1024KiB loads=1069 loads/s=88.9 MB/s=1505 load_ms mean=22.5 p50=20.6 p90=30.5 p99=45.9 max=81.4 | dev_read_MB {0: 17916.3, 1: 3400.5, 2: 4079.0} dev_write_MB {0: 182.3, 1: 0.0, 2: 0.0} | load 40.12, 40.18, 51.69
DS-expert W=6 P=4 CH=1024KiB loads=594 loads/s=49.2 MB/s=832 load_ms mean=121.7 p50=114.5 p90=172.6 p99=328.0 max=342.3 | dev_read_MB {0: 5445.1, 1: 3091.5, 2: 4758.1} dev_write_MB {0: 909.1, 1: 0.0, 2: 0.0} | load 41.11, 40.42, 51.52
DS-expert W=6 P=8 CH=512KiB loads=576 loads/s=47.7 MB/s=807 load_ms mean=125.6 p50=115.6 p90=198.3 p99=257.1 max=312.7 | dev_read_MB {0: 7185.4, 1: 3390.5, 2: 4669.6} dev_write_MB {0: 1303.4, 1: 0.0, 2: 0.0} | load 42.47, 40.79, 51.47
DS-expert W=12 P=2 CH=1024KiB loads=652 loads/s=53.9 MB/s=911 load_ms mean=221.9 p50=216.2 p90=319.8 p99=395.6 max=448.8 | dev_read_MB {0: 6351.6, 1: 3362.3, 2: 4964.0} dev_write_MB {0: 989.9, 1: 0.0, 2: 0.0} | load 46.76, 41.77, 51.67
DS-expert W=1 P=32 CH=256KiB loads=293 loads/s=24.4 MB/s=413 load_ms mean=41.0 p50=38.4 p90=56.8 p99=121.0 max=148.3 | dev_read_MB {0: 3755.9, 1: 2220.7, 2: 3954.8} dev_write_MB {0: 1408.3, 1: 0.0, 2: 0.0} | load 50.62, 42.87, 51.87
Qwen-settled-ref W=1 P=16 CH=512KiB loads=208 loads/s=17.3 MB/s=290 load_ms mean=57.9 p50=44.4 p90=79.9 p99=275.1 max=326.6 | dev_read_MB {0: 649.2, 1: 1363.9, 2: 3713.8} dev_write_MB {0: 2672.5, 1: 143.3, 2: 409.1} | load 52.65, 43.96, 52.00
Qwen-settled-ref W=6 P=4 CH=1024KiB loads=341 loads/s=28.1 MB/s=471 load_ms mean=212.7 p50=187.3 p90=333.9 p99=599.6 max=660.9 | dev_read_MB {0: 177.2, 1: 1911.1, 2: 5353.2} dev_write_MB {0: 2665.7, 1: 31.0, 2: 89.8} | load 57.89, 45.55, 52.39
```

Reference, ext4 on the root NVMe (WD Blue SN570, shared by the whole system),
the same 4 KiB `O_DIRECT` randread over the 9.0 GiB of `build/bin`:

```
ext4 nvme1 qd1 iops=6019 p50=151us p99=922us mean=166us
ext4 nvme1 qd16 iops=82798 p50=163us p99=930us mean=193us
ext4 nvme1 qd64 iops=110110 p50=518us p99=2900us mean=581us
```

HOST-tier gather: random 170 B Q8_0 rows from a 4 GiB anonymous THP table,
dequantized to F32 (best of 20):

```
RAM gather rows=16 threads=1 best=4.2 us
RAM gather rows=8192 threads=1 best=2329.7 us
RAM gather rows=8192 threads=8 best=1941.1 us   (includes thread spawn)
```

bcachefs state: `bcachefs fs usage -h /models` gave bulk.t7 171G used,
bulk.ugreen 612G used, nvme.kingston 187G used, `cached: 182G`. Per-device
`alloc_debug`: dev0 `user 0 sectors, cached 382424032 sectors`. Options:
`foreground_target=nvme background_target=bulk promote_target=nvme
data_checksum=crc32c encoded_extent_max=256k promote_whole_extents=1
block_size=4.00k compression=none`.

**Rev 2 probes on the DeepSeek Q2_K tables** (`blk.1.engram_embd` in
`Q2_K-00001-of-00007` at offset 5785327904, 32256518112 B; `blk.14` in
`Q2_K-00004-of-00007` at offset 4032). Run 18:35-18:41, in this order:

```
 18:35:56 up 17 days, 23:51, 17 users,  load average: 31.30, 35.91, 35.80
ds2k_blk1_qd64: iops=4657 MB/s=19.1 ios=69989 lat_mean=13740us p50=3064us p99=145752us dev_read_MB={0:6384.9, 1:3572.6, 2:6737.2} amp=58.2x dev_write_MB={0:7500.5, 1:37.7, 2:107.3}
ds2k_blk1_qd16: iops=4249 MB/s=17.4 ios=63765 lat_mean=3764us p50=2867us p99=27656us dev_read_MB={0:6933.2, 1:3383.8, 2:5798.8} amp=61.7x dev_write_MB={0:3506.2, 1:325.1, 2:922.1}
# batch_probe3 split 24 24 64 <secs>: both tables through one 64-worker pool; per batch, time until blk.1's 24 landed, and until all 48 landed
split B1=24 B2=24 workers=64
  table1   n=1768 mean=7.13ms p50=6.81ms p90=9.16ms p99=14.63ms max=42.35ms
  all      n=1768 mean=11.23ms p50=10.84ms p90=14.15ms p99=21.00ms max=42.35ms
# bcachefs io_done around the split run (user read and user write deltas, as the run recorded them)
dev0 read +9.65 GB, write +1.54 GB
dev1 read +3.92 GB, write +0.02 GB
dev2 read +6.58 GB, write +0.04 GB
# batch_probe3 hol 12000 24+24 <order>: a 12000-row prefill queued first, a 48-row decode 200 ms later
hol P=12000 delay=200ms order=fifo workers=64
  decode   n=4 mean=1358.59ms p50=1243.48ms p90=1496.18ms p99=1496.18ms max=1531.59ms
  prefill  n=4 mean=1560.74ms p50=1447.02ms p90=1696.22ms p99=1696.22ms max=1731.20ms
hol P=12000 delay=200ms order=prio workers=64
  decode   n=4 mean=21.73ms p50=19.46ms p90=23.02ms p99=23.02ms max=27.59ms
  prefill  n=4 mean=1664.46ms p50=1626.03ms p90=1699.45ms p99=1699.45ms max=1732.42ms
# same-session control, then the pool, then the control again
ds2k_ctl_a: iops=7070 MB/s=29.0 ios=70775 lat_mean=9051us p50=6849us p99=33161us dev_read_MB={0:7977.7, 1:3974.0, 2:4487.6} amp=56.7x dev_write_MB={0:490.1, 1:0.0, 2:0.0}
batch=20000 workers=64 batches=5 mean=2836.35ms p50=2840.99ms p90=2887.11ms p99=2887.11ms max=2907.78ms rows/s=7051 straddle=1.9%
ds2k_ctl_b: iops=7242 MB/s=29.7 ios=72578 lat_mean=8836us p50=6980us p99=33161us dev_read_MB={0:8253.3, 1:4060.2, 2:4489.8} amp=56.5x dev_write_MB={0:399.8, 1:0.0, 2:0.0}
# single-table miss curve (batch_probe2, 64 workers)
batch=1 workers=64 batches=3985 mean=1.51ms p50=1.25ms p90=2.69ms p99=4.00ms max=10.10ms rows/s=664 straddle=1.9%
batch=4 workers=64 batches=2529 mean=2.37ms p50=2.08ms p90=3.55ms p99=5.16ms max=20.01ms rows/s=1686 straddle=2.0%
batch=8 workers=64 batches=2132 mean=2.81ms p50=2.48ms p90=4.01ms p99=6.96ms max=38.04ms rows/s=2842 straddle=2.2%
batch=16 workers=64 batches=1421 mean=4.23ms p50=3.92ms p90=5.90ms p99=9.00ms max=19.21ms rows/s=3787 straddle=2.0%
batch=24 workers=64 batches=1085 mean=5.53ms p50=5.10ms p90=7.28ms p99=13.94ms max=25.96ms rows/s=4338 straddle=2.1%
```

### A.3 Hash and cache simulation

Validation:

```
tokenizer (qwen35 regex + byte BPE) vs models/ggml-vocab-qwen35.gguf.{inp,out}: 50/50 match
Qwen3.8 vocab 248320 sha 9c56ea103777902c eos id 248044 = '<|endoftext|>' 248046 = '<|im_end|>'
wiki_test: 1288556 chars -> 297193 tokens, roundtrip=OK
wiki_train: 10918892 chars -> 2505800 tokens, roundtrip=OK
code: 9989391 chars -> 3422500 tokens, roundtrip=OK
docs: 7694987 chars -> 2289834 tokens, roundtrip=OK
python vs verbatim-upstream C++: IDENTICAL 320000 rows; max row 320001219 < 320001536: True
ubatch_vs_decode_hash_equal=1   (ubatch 512 and ubatch 1 produce the same rows)
```

DeepSeek (`qtok2.py`, `ds_hash.py`, `ds_sim.cpp`):

```
46/46 match (deepseek-llm vectors)
DS vocab 129280 sha ab8d16d71b3a9d12 compressed vocab 99092 pad_id [2] token_map[pad] 2
wiki_test: 1288556 chars -> 287730 tokens, roundtrip=OK
wiki_train: 10918892 chars -> 2424283 tokens, roundtrip=OK
code: 9989391 chars -> 3287443 tokens, roundtrip=OK
docs: 7694987 chars -> 2241455 tokens, roundtrip=OK
DS python vs C++: IDENTICAL 960000 rows; max row 384015568   (wiki_test, first 20000 tokens)
DS python vs C++: IDENTICAL 960000 rows; max row 384015792   (code, first 20000 tokens)
```

`ple_sim` (Qwen, whole-stream hit rates):

```
wiki_test: tokens=297193 row_refs=4755088 distinct_rows=2603218 (0.41 GiB) ubatch_vs_decode_hash_equal=1
wiki_test: cold PP512 ubatch: unique rows/ubatch=7028 of 8192 (85.8%), unique 4K pages/ubatch=7026
wiki_test: cache     32 MiB (   197379 rows): LRU hit=36.5% (2nd half 34.6%, bigram 48.7%, trigram 24.2%)  CLOCK hit=36.9% (2nd half 35.1%)  4K-page LRU same bytes hit=21.0%
wiki_test: cache    128 MiB (   789516 rows): LRU hit=41.9% (2nd half 41.1%, bigram 55.5%, trigram 28.3%)  CLOCK hit=42.4% (2nd half 41.9%)  4K-page LRU same bytes hit=29.5%
wiki_test: cache    512 MiB (  3158064 rows): LRU hit=45.3% (2nd half 47.0%, bigram 59.5%, trigram 31.0%)  CLOCK hit=45.3% (2nd half 47.0%)  4K-page LRU same bytes hit=35.8%
wiki_test: cache   2048 MiB ( 12632256 rows): LRU hit=45.3% (2nd half 47.0%, bigram 59.5%, trigram 31.0%)  CLOCK hit=45.3% (2nd half 47.0%)  4K-page LRU same bytes hit=42.5%
wiki_test: cache   8192 MiB ( 50529027 rows): LRU hit=45.3% (2nd half 47.0%, bigram 59.5%, trigram 31.0%)  CLOCK hit=45.3% (2nd half 47.0%)  4K-page LRU same bytes hit=50.3%
wiki_test 17.59s 3242040KB
wiki_train: tokens=2505800 row_refs=40092800 distinct_rows=16901229 (2.68 GiB) ubatch_vs_decode_hash_equal=1
wiki_train: cold PP512 ubatch: unique rows/ubatch=7104 of 8192 (86.7%), unique 4K pages/ubatch=7102
wiki_train: cache     32 MiB (   197379 rows): LRU hit=34.5% (2nd half 34.3%, bigram 46.7%, trigram 22.4%)  CLOCK hit=35.1% (2nd half 34.8%)  4K-page LRU same bytes hit=19.5%
wiki_train: cache    128 MiB (   789516 rows): LRU hit=41.1% (2nd half 41.0%, bigram 54.9%, trigram 27.3%)  CLOCK hit=41.8% (2nd half 41.7%)  4K-page LRU same bytes hit=27.6%
wiki_train: cache    512 MiB (  3158064 rows): LRU hit=49.7% (2nd half 50.3%, bigram 65.1%, trigram 34.4%)  CLOCK hit=50.4% (2nd half 51.1%)  4K-page LRU same bytes hit=33.7%
wiki_train: cache   2048 MiB ( 12632256 rows): LRU hit=57.4% (2nd half 61.8%, bigram 72.9%, trigram 42.0%)  CLOCK hit=57.5% (2nd half 61.8%)  4K-page LRU same bytes hit=41.4%
wiki_train: cache   8192 MiB ( 50529027 rows): LRU hit=57.8% (2nd half 62.6%, bigram 73.2%, trigram 42.5%)  CLOCK hit=57.8% (2nd half 62.6%)  4K-page LRU same bytes hit=55.6%
wiki_train 275.49s 4897572KB
code: tokens=3422500 row_refs=54760000 distinct_rows=5075084 (0.80 GiB) ubatch_vs_decode_hash_equal=1
code: cold PP512 ubatch: unique rows/ubatch=3969 of 8192 (48.4%), unique 4K pages/ubatch=3968
code: cache     32 MiB (   197379 rows): LRU hit=86.8% (2nd half 84.3%, bigram 90.7%, trigram 83.0%)  CLOCK hit=86.9% (2nd half 84.4%)  4K-page LRU same bytes hit=68.6%
code: cache    128 MiB (   789516 rows): LRU hit=89.1% (2nd half 87.2%, bigram 92.8%, trigram 85.5%)  CLOCK hit=89.1% (2nd half 87.3%)  4K-page LRU same bytes hit=81.2%
code: cache    512 MiB (  3158064 rows): LRU hit=90.5% (2nd half 89.6%, bigram 94.1%, trigram 87.0%)  CLOCK hit=90.6% (2nd half 89.7%)  4K-page LRU same bytes hit=86.1%
code: cache   2048 MiB ( 12632256 rows): LRU hit=90.7% (2nd half 90.1%, bigram 94.3%, trigram 87.2%)  CLOCK hit=90.7% (2nd half 90.1%)  4K-page LRU same bytes hit=89.0%
code: cache   8192 MiB ( 50529027 rows): LRU hit=90.7% (2nd half 90.1%, bigram 94.3%, trigram 87.2%)  CLOCK hit=90.7% (2nd half 90.1%)  4K-page LRU same bytes hit=91.6%
code 125.63s 3703892KB
docs: tokens=2289834 row_refs=36637344 distinct_rows=10478884 (1.66 GiB) ubatch_vs_decode_hash_equal=1
docs: cold PP512 ubatch: unique rows/ubatch=6231 of 8192 (76.1%), unique 4K pages/ubatch=6229
docs: cache     32 MiB (   197379 rows): LRU hit=56.3% (2nd half 56.4%, bigram 66.0%, trigram 46.7%)  CLOCK hit=56.8% (2nd half 56.9%)  4K-page LRU same bytes hit=33.6%
docs: cache    128 MiB (   789516 rows): LRU hit=63.0% (2nd half 63.5%, bigram 73.2%, trigram 52.8%)  CLOCK hit=63.4% (2nd half 63.9%)  4K-page LRU same bytes hit=45.1%
docs: cache    512 MiB (  3158064 rows): LRU hit=68.2% (2nd half 69.1%, bigram 78.9%, trigram 57.6%)  CLOCK hit=68.5% (2nd half 69.5%)  4K-page LRU same bytes hit=54.6%
docs: cache   2048 MiB ( 12632256 rows): LRU hit=71.4% (2nd half 74.7%, bigram 82.0%, trigram 60.8%)  CLOCK hit=71.4% (2nd half 74.7%)  4K-page LRU same bytes hit=62.8%
docs: cache   8192 MiB ( 50529027 rows): LRU hit=71.4% (2nd half 74.7%, bigram 82.0%, trigram 60.8%)  CLOCK hit=71.4% (2nd half 74.7%)  4K-page LRU same bytes hit=72.1%
docs 271.18s 4183236KB
mixed: tokens=8218134 row_refs=131490144 distinct_rows=30389923 (4.81 GiB) ubatch_vs_decode_hash_equal=1
mixed: cold PP512 ubatch: unique rows/ubatch=5562 of 8192 (67.9%), unique 4K pages/ubatch=5561
mixed: cache     32 MiB (   197379 rows): LRU hit=58.8% (2nd half 61.5%, bigram 66.9%, trigram 50.8%)  CLOCK hit=59.3% (2nd half 62.0%)  4K-page LRU same bytes hit=41.0%
mixed: cache    128 MiB (   789516 rows): LRU hit=64.4% (2nd half 67.0%, bigram 72.9%, trigram 55.8%)  CLOCK hit=64.7% (2nd half 67.4%)  4K-page LRU same bytes hit=45.8%
mixed: cache    512 MiB (  3158064 rows): LRU hit=69.6% (2nd half 72.1%, bigram 78.7%, trigram 60.5%)  CLOCK hit=70.0% (2nd half 72.4%)  4K-page LRU same bytes hit=57.4%
mixed: cache   2048 MiB ( 12632256 rows): LRU hit=74.9% (2nd half 77.9%, bigram 84.0%, trigram 65.7%)  CLOCK hit=75.1% (2nd half 78.3%)  4K-page LRU same bytes hit=64.3%
mixed: cache   8192 MiB ( 50529027 rows): LRU hit=76.9% (2nd half 81.6%, bigram 85.7%, trigram 68.1%)  CLOCK hit=76.9% (2nd half 81.6%)  4K-page LRU same bytes hit=73.4%
mixed 819.42s 6738224KB
```

`ple_sim2` (Qwen, CLOCK, steady state = after the first 25%):

```
random_tokens cache=    0MiB decode(steady): miss rows/token=16.00  P(0 miss)=0.0%  P(<=8 miss)=0.0%  P(16 miss)=100.0% | PP512(steady): unique-miss rows/ubatch mean=8192 max=8192
random_tokens cache=   32MiB decode(steady): miss rows/token=15.99  P(0 miss)=0.0%  P(<=8 miss)=0.0%  P(16 miss)=99.0% | PP512(steady): unique-miss rows/ubatch mean=8187 max=8192
random_tokens cache=  128MiB decode(steady): miss rows/token=15.96  P(0 miss)=0.0%  P(<=8 miss)=0.0%  P(16 miss)=96.2% | PP512(steady): unique-miss rows/ubatch mean=8172 max=8184
random_tokens cache=  512MiB decode(steady): miss rows/token=15.87  P(0 miss)=0.0%  P(<=8 miss)=0.0%  P(16 miss)=87.7% | PP512(steady): unique-miss rows/ubatch mean=8125 max=8173
random_tokens cache= 2048MiB decode(steady): miss rows/token=15.85  P(0 miss)=0.0%  P(<=8 miss)=0.0%  P(16 miss)=86.1% | PP512(steady): unique-miss rows/ubatch mean=8116 max=8173
random_tokens cache= 8192MiB decode(steady): miss rows/token=15.85  P(0 miss)=0.0%  P(<=8 miss)=0.0%  P(16 miss)=86.1% | PP512(steady): unique-miss rows/ubatch mean=8116 max=8173
wiki_test cache=    0MiB decode(steady): miss rows/token=16.00  P(0 miss)=0.0%  P(<=8 miss)=0.0%  P(16 miss)=100.0% | PP512(steady): unique-miss rows/ubatch mean=7055 max=7848
wiki_test cache=   32MiB decode(steady): miss rows/token=10.13  P(0 miss)=24.2%  P(<=8 miss)=49.1%  P(16 miss)=50.4% | PP512(steady): unique-miss rows/ubatch mean=5185 max=6676
wiki_test cache=  128MiB decode(steady): miss rows/token=9.08  P(0 miss)=28.9%  P(<=8 miss)=57.3%  P(16 miss)=41.0% | PP512(steady): unique-miss rows/ubatch mean=4649 max=6362
wiki_test cache=  512MiB decode(steady): miss rows/token=8.47  P(0 miss)=31.9%  P(<=8 miss)=61.6%  P(16 miss)=35.3% | PP512(steady): unique-miss rows/ubatch mean=4335 max=5951
wiki_test cache= 2048MiB decode(steady): miss rows/token=8.47  P(0 miss)=31.9%  P(<=8 miss)=61.6%  P(16 miss)=35.3% | PP512(steady): unique-miss rows/ubatch mean=4335 max=5951
wiki_test cache= 8192MiB decode(steady): miss rows/token=8.47  P(0 miss)=31.9%  P(<=8 miss)=61.6%  P(16 miss)=35.3% | PP512(steady): unique-miss rows/ubatch mean=4335 max=5951
code cache=    0MiB decode(steady): miss rows/token=16.00  P(0 miss)=0.0%  P(<=8 miss)=0.0%  P(16 miss)=100.0% | PP512(steady): unique-miss rows/ubatch mean=3922 max=7456
code cache=   32MiB decode(steady): miss rows/token=2.01  P(0 miss)=83.7%  P(<=8 miss)=91.2%  P(16 miss)=8.8% | PP512(steady): unique-miss rows/ubatch mean=1024 max=5909
code cache=  128MiB decode(steady): miss rows/token=1.62  P(0 miss)=86.3%  P(<=8 miss)=93.4%  P(16 miss)=6.4% | PP512(steady): unique-miss rows/ubatch mean=828 max=5524
code cache=  512MiB decode(steady): miss rows/token=1.33  P(0 miss)=88.2%  P(<=8 miss)=95.0%  P(16 miss)=4.3% | PP512(steady): unique-miss rows/ubatch mean=681 max=5419
code cache= 2048MiB decode(steady): miss rows/token=1.30  P(0 miss)=88.4%  P(<=8 miss)=95.2%  P(16 miss)=4.0% | PP512(steady): unique-miss rows/ubatch mean=664 max=5392
code cache= 8192MiB decode(steady): miss rows/token=1.30  P(0 miss)=88.4%  P(<=8 miss)=95.2%  P(16 miss)=4.0% | PP512(steady): unique-miss rows/ubatch mean=664 max=5392
docs cache=    0MiB decode(steady): miss rows/token=16.00  P(0 miss)=0.0%  P(<=8 miss)=0.0%  P(16 miss)=100.0% | PP512(steady): unique-miss rows/ubatch mean=6190 max=8128
docs cache=   32MiB decode(steady): miss rows/token=6.50  P(0 miss)=49.8%  P(<=8 miss)=68.8%  P(16 miss)=30.9% | PP512(steady): unique-miss rows/ubatch mean=3323 max=7076
docs cache=  128MiB decode(steady): miss rows/token=5.42  P(0 miss)=56.1%  P(<=8 miss)=76.0%  P(16 miss)=23.1% | PP512(steady): unique-miss rows/ubatch mean=2775 max=6640
docs cache=  512MiB decode(steady): miss rows/token=4.58  P(0 miss)=60.5%  P(<=8 miss)=81.5%  P(16 miss)=15.7% | PP512(steady): unique-miss rows/ubatch mean=2343 max=5700
docs cache= 2048MiB decode(steady): miss rows/token=3.97  P(0 miss)=63.9%  P(<=8 miss)=85.2%  P(16 miss)=10.4% | PP512(steady): unique-miss rows/ubatch mean=2032 max=4930
docs cache= 8192MiB decode(steady): miss rows/token=3.97  P(0 miss)=63.9%  P(<=8 miss)=85.2%  P(16 miss)=10.4% | PP512(steady): unique-miss rows/ubatch mean=2032 max=4930
mixed cache=    0MiB decode(steady): miss rows/token=16.00  P(0 miss)=0.0%  P(<=8 miss)=0.0%  P(16 miss)=100.0% | PP512(steady): unique-miss rows/ubatch mean=5452 max=8128
mixed cache=   32MiB decode(steady): miss rows/token=6.16  P(0 miss)=53.6%  P(<=8 miss)=69.3%  P(16 miss)=30.4% | PP512(steady): unique-miss rows/ubatch mean=3154 max=7201
mixed cache=  128MiB decode(steady): miss rows/token=5.28  P(0 miss)=58.7%  P(<=8 miss)=75.1%  P(16 miss)=23.9% | PP512(steady): unique-miss rows/ubatch mean=2700 max=6841
mixed cache=  512MiB decode(steady): miss rows/token=4.45  P(0 miss)=63.0%  P(<=8 miss)=80.8%  P(16 miss)=16.3% | PP512(steady): unique-miss rows/ubatch mean=2275 max=6146
mixed cache= 2048MiB decode(steady): miss rows/token=3.52  P(0 miss)=67.6%  P(<=8 miss)=86.3%  P(16 miss)=7.1% | PP512(steady): unique-miss rows/ubatch mean=1800 max=5186
mixed cache= 8192MiB decode(steady): miss rows/token=3.13  P(0 miss)=69.6%  P(<=8 miss)=88.1%  P(16 miss)=4.2% | PP512(steady): unique-miss rows/ubatch mean=1603 max=4673
wiki_train cache=    0MiB decode(steady): miss rows/token=16.00  P(0 miss)=0.0%  P(<=8 miss)=0.0%  P(16 miss)=100.0% | PP512(steady): unique-miss rows/ubatch mean=7100 max=7968
wiki_train cache=   32MiB decode(steady): miss rows/token=10.36  P(0 miss)=22.8%  P(<=8 miss)=47.6%  P(16 miss)=51.8% | PP512(steady): unique-miss rows/ubatch mean=5298 max=6932
wiki_train cache=  128MiB decode(steady): miss rows/token=9.26  P(0 miss)=27.8%  P(<=8 miss)=56.1%  P(16 miss)=42.1% | PP512(steady): unique-miss rows/ubatch mean=4738 max=6407
wiki_train cache=  512MiB decode(steady): miss rows/token=7.75  P(0 miss)=35.0%  P(<=8 miss)=67.0%  P(16 miss)=28.0% | PP512(steady): unique-miss rows/ubatch mean=3968 max=5820
wiki_train cache= 2048MiB decode(steady): miss rows/token=6.29  P(0 miss)=42.6%  P(<=8 miss)=75.8%  P(16 miss)=14.5% | PP512(steady): unique-miss rows/ubatch mean=3219 max=5311
wiki_train cache= 8192MiB decode(steady): miss rows/token=6.21  P(0 miss)=43.0%  P(<=8 miss)=76.2%  P(16 miss)=13.8% | PP512(steady): unique-miss rows/ubatch mean=3177 max=5311
finished
```

`ds_sim` (DeepSeek, both tables, CLOCK at the stated total size in 110 B
slots; hit = whole stream; decode and PP512 = steady state after the first
25%; the random stream is 300000 tokens drawn uniformly from 129280):

```
random: tokens=300000 row_refs=14400000 distinct_rows=14265643 (1.46 GiB @110B) cold PP512 unique rows/ubatch=24576 of 24576
random cache=    0MiB hit=0.0% (2g 0.0%, 3g 0.0%, 4g 0.0%) decode(steady): miss rows/token=48.00 of 48 P(0 miss)=0.0% P(48 miss)=100.0% | PP512(steady) unique-miss rows/ubatch mean=24576 max=24576
random cache=  128MiB hit=0.2% (2g 0.2%, 3g 0.2%, 4g 0.1%) decode(steady): miss rows/token=47.92 of 48 P(0 miss)=0.0% P(48 miss)=92.7% | PP512(steady) unique-miss rows/ubatch mean=24537 max=24553
random cache=  512MiB hit=0.5% (2g 0.5%, 3g 0.5%, 4g 0.5%) decode(steady): miss rows/token=47.70 of 48 P(0 miss)=0.0% P(48 miss)=74.0% | PP512(steady) unique-miss rows/ubatch mean=24422 max=24480
random cache= 2048MiB hit=0.9% (2g 0.9%, 3g 0.9%, 4g 0.9%) decode(steady): miss rows/token=47.44 of 48 P(0 miss)=0.0% P(48 miss)=58.1% | PP512(steady) unique-miss rows/ubatch mean=24290 max=24480
random cache= 8192MiB hit=0.9% (2g 0.9%, 3g 0.9%, 4g 0.9%) decode(steady): miss rows/token=47.44 of 48 P(0 miss)=0.0% P(48 miss)=58.1% | PP512(steady) unique-miss rows/ubatch mean=24290 max=24480
wiki_test: tokens=287730 row_refs=13811040 distinct_rows=9113026 (0.93 GiB @110B) cold PP512 unique rows/ubatch=22117 of 24576
wiki_test cache=    0MiB hit=0.0% (2g 0.0%, 3g 0.0%, 4g 0.0%) decode(steady): miss rows/token=48.00 of 48 P(0 miss)=0.0% P(48 miss)=100.0% | PP512(steady) unique-miss rows/ubatch mean=22180 max=23920
wiki_test cache=  128MiB hit=28.9% (2g 51.5%, 3g 23.5%, 4g 11.6%) decode(steady): miss rows/token=34.12 of 48 P(0 miss)=11.3% P(48 miss)=44.6% | PP512(steady) unique-miss rows/ubatch mean=17460 max=21056
wiki_test cache=  512MiB hit=33.3% (2g 58.2%, 3g 27.8%, 4g 13.7%) decode(steady): miss rows/token=31.42 of 48 P(0 miss)=13.5% P(48 miss)=30.0% | PP512(steady) unique-miss rows/ubatch mean=16092 max=20438
wiki_test cache= 2048MiB hit=34.0% (2g 59.2%, 3g 28.6%, 4g 14.2%) decode(steady): miss rows/token=30.93 of 48 P(0 miss)=14.0% P(48 miss)=27.0% | PP512(steady) unique-miss rows/ubatch mean=15837 max=20438
wiki_test cache= 8192MiB hit=34.0% (2g 59.2%, 3g 28.6%, 4g 14.2%) decode(steady): miss rows/token=30.93 of 48 P(0 miss)=14.0% P(48 miss)=27.0% | PP512(steady) unique-miss rows/ubatch mean=15837 max=20438
code: tokens=3287443 row_refs=157797264 distinct_rows=19396356 (1.99 GiB @110B) cold PP512 unique rows/ubatch=13114 of 24576
code cache=    0MiB hit=0.0% (2g 0.0%, 3g 0.0%, 4g 0.0%) decode(steady): miss rows/token=48.00 of 48 P(0 miss)=0.0% P(48 miss)=100.0% | PP512(steady) unique-miss rows/ubatch mean=12964 max=22511
code cache=  128MiB hit=84.4% (2g 92.1%, 3g 84.2%, 4g 76.9%) decode(steady): miss rows/token=7.11 of 48 P(0 miss)=77.9% P(48 miss)=6.9% | PP512(steady) unique-miss rows/ubatch mean=3632 max=18348
code cache=  512MiB hit=86.6% (2g 94.1%, 3g 86.5%, 4g 79.3%) decode(steady): miss rows/token=5.91 of 48 P(0 miss)=80.5% P(48 miss)=3.9% | PP512(steady) unique-miss rows/ubatch mean=3026 max=17368
code cache= 2048MiB hit=87.7% (2g 95.0%, 3g 87.6%, 4g 80.5%) decode(steady): miss rows/token=5.22 of 48 P(0 miss)=81.8% P(48 miss)=1.8% | PP512(steady) unique-miss rows/ubatch mean=2674 max=17006
code cache= 8192MiB hit=87.7% (2g 95.0%, 3g 87.6%, 4g 80.5%) decode(steady): miss rows/token=5.22 of 48 P(0 miss)=81.8% P(48 miss)=1.8% | PP512(steady) unique-miss rows/ubatch mean=2674 max=17006
docs: tokens=2241455 row_refs=107589840 distinct_rows=39423992 (4.04 GiB @110B) cold PP512 unique rows/ubatch=19584 of 24576
docs cache=    0MiB hit=0.0% (2g 0.0%, 3g 0.0%, 4g 0.0%) decode(steady): miss rows/token=48.00 of 48 P(0 miss)=0.0% P(48 miss)=100.0% | PP512(steady) unique-miss rows/ubatch mean=19440 max=24464
docs cache=  128MiB hit=52.3% (2g 70.3%, 3g 48.9%, 4g 37.8%) decode(steady): miss rows/token=21.39 of 48 P(0 miss)=40.9% P(48 miss)=25.0% | PP512(steady) unique-miss rows/ubatch mean=10945 max=21725
docs cache=  512MiB hit=57.5% (2g 77.5%, 3g 53.9%, 4g 41.1%) decode(steady): miss rows/token=18.89 of 48 P(0 miss)=44.0% P(48 miss)=14.7% | PP512(steady) unique-miss rows/ubatch mean=9669 max=19637
docs cache= 2048MiB hit=62.1% (2g 82.9%, 3g 58.8%, 4g 44.6%) decode(steady): miss rows/token=16.33 of 48 P(0 miss)=46.7% P(48 miss)=4.2% | PP512(steady) unique-miss rows/ubatch mean=8368 max=17938
docs cache= 8192MiB hit=63.4% (2g 83.9%, 3g 60.2%, 4g 45.9%) decode(steady): miss rows/token=15.52 of 48 P(0 miss)=47.8% P(48 miss)=2.6% | PP512(steady) unique-miss rows/ubatch mean=7945 max=16614
wiki_train: tokens=2424283 row_refs=116365584 distinct_rows=62529672 (6.41 GiB @110B) cold PP512 unique rows/ubatch=22320 of 24576
wiki_train cache=    0MiB hit=0.0% (2g 0.0%, 3g 0.0%, 4g 0.0%) decode(steady): miss rows/token=48.00 of 48 P(0 miss)=0.0% P(48 miss)=100.0% | PP512(steady) unique-miss rows/ubatch mean=22306 max=24000
wiki_train cache=  128MiB hit=27.6% (2g 50.3%, 3g 22.1%, 4g 10.4%) decode(steady): miss rows/token=34.64 of 48 P(0 miss)=10.4% P(48 miss)=45.8% | PP512(steady) unique-miss rows/ubatch mean=17731 max=21498
wiki_train cache=  512MiB hit=34.0% (2g 60.3%, 3g 28.3%, 4g 13.5%) decode(steady): miss rows/token=31.43 of 48 P(0 miss)=13.0% P(48 miss)=28.6% | PP512(steady) unique-miss rows/ubatch mean=16091 max=20293
wiki_train cache= 2048MiB hit=42.0% (2g 70.3%, 3g 36.8%, 4g 18.9%) decode(steady): miss rows/token=26.93 of 48 P(0 miss)=17.3% P(48 miss)=7.9% | PP512(steady) unique-miss rows/ubatch mean=13790 max=18564
wiki_train cache= 8192MiB hit=46.3% (2g 74.0%, 3g 41.5%, 4g 23.3%) decode(steady): miss rows/token=24.20 of 48 P(0 miss)=20.0% P(48 miss)=2.4% | PP512(steady) unique-miss rows/ubatch mean=12389 max=18564
mixed: tokens=7953181 row_refs=381752688 distinct_rows=111780913 (11.45 GiB @110B) cold PP512 unique rows/ubatch=17762 of 24576
mixed cache=    0MiB hit=0.0% (2g 0.0%, 3g 0.0%, 4g 0.0%) decode(steady): miss rows/token=48.00 of 48 P(0 miss)=0.0% P(48 miss)=100.0% | PP512(steady) unique-miss rows/ubatch mean=17452 max=24464
mixed cache=  128MiB hit=55.1% (2g 70.1%, 3g 52.3%, 4g 43.0%) decode(steady): miss rows/token=20.35 of 48 P(0 miss)=45.7% P(48 miss)=25.9% | PP512(steady) unique-miss rows/ubatch mean=10421 max=21935
mixed cache=  512MiB hit=60.0% (2g 76.5%, 3g 57.0%, 4g 46.5%) decode(steady): miss rows/token=17.97 of 48 P(0 miss)=49.1% P(48 miss)=15.9% | PP512(steady) unique-miss rows/ubatch mean=9200 max=20540
mixed cache= 2048MiB hit=65.1% (2g 82.7%, 3g 62.3%, 4g 50.5%) decode(steady): miss rows/token=15.46 of 48 P(0 miss)=51.9% P(48 miss)=4.5% | PP512(steady) unique-miss rows/ubatch mean=7916 max=18649
mixed cache= 8192MiB hit=70.2% (2g 86.8%, 3g 67.7%, 4g 56.1%) decode(steady): miss rows/token=12.43 of 48 P(0 miss)=54.9% P(48 miss)=0.2% | PP512(steady) unique-miss rows/ubatch mean=6370 max=16572
finished
```

## Appendix B. Probe sources (scratch, not committed; the fit calculator is committed)

- `io_probe.sh`: fio randread driver plus bcachefs `io_done` snapshots.
- `mmap_probe.cpp`: N threads touch random rows through `mmap`.
- `batch_probe.cpp`: the §5.2 pread pool, measured per batch.
- `ram_gather.cpp`: the HOST-tier gather plus dequant.
- `qtok.py`: the qwen35 byte-level BPE, reading tokens and merges from a GGUF.
- `ple_hash.py`: an independent Python transcription of `set_input`.
- `upstream_set_input_body.inc`: `sed -n 1067,1125p` of
  `694ec2354 src/models/qwen4exp.cpp`, `#include`d verbatim by `ple_sim*.cpp`.
- `ple_sim.cpp` / `ple_sim2.cpp`: the row stream plus LRU, CLOCK and 4 KiB-page
  LRU; per-token and per-ubatch miss metrics.
- `io_probe2.sh`, `batch_probe2.cpp`: the same probes with the file, extent and
  row size as parameters (used for the DeepSeek table).
- `ds_meta.py`: the DeepSeek GGUF keys and engram tensor extents.
- `qtok2.py`, `ds_regex.json`, `ds_tokenize.py`: the multi-regex BPE, its
  deepseek-llm validation, and the `joyai-llm` tokenization of the corpora.
- `ds_hash.py`, `ds_sim.cpp`, `ds_consts.inc`, `ds_token_map.bin`: the two
  transcriptions of `NgramHashState.forward` and the CLOCK simulation.
- `expert_probe.cpp`, `expert_run.sh`, `ds_expert_extents.txt`,
  `qwen_ref_extents.txt`: whole-expert `O_DIRECT` loads over the routed-expert
  slices, with per-device attribution.

- `batch_probe3.cpp` (rev 2): two modes over one 64-worker pread pool. `split`
  issues two tables' batches together and times the first table's landing and
  the whole batch's; `hol` queues a prefill batch, then a decode batch after a
  delay, in FIFO or priority order. Files, offsets and row counts come from
  `T1F`/`T1BASE`/`T1NROWS`/`T2F`/`T2BASE`/`T2NROWS`/`ROW`.
- `ds_q2k_census.py`, `qwen_census.py` (rev 2): per-class weight bytes from the
  GGUF tensor headers, the largest non-table tensor, and the engram extents.
  Run with `/usr/bin/python3` (gguf-py needs its numpy).
- `io_probe2.sh` takes `FOFF`/`SIZE` for the extent; rev 2's same-session
  controls use it at `blk.1`'s offset.
- **`docs/design/sycl_storage_tier_host_fit.py` (rev 2b-2c, committed with this
  document)**: the §4.5 host-fit calculator, grown from the scratch
  `c2_fit.py` by adding the PP MoE oneDNN ring, the today / after-(s) columns,
  the RUNTIME pool's chunk commit, 9g22's P3 check, the engram tables' joint
  P3 placement and a cross-check of 9g22's spread and concentrated cold-set
  bands (*lead ruling §D6*). Run it from the
  repository root with `python3 docs/design/sycl_storage_tier_host_fit.py`; it
  has no dependencies. Its constants are the GGUF header facts of A.1 and the
  sizing-code terms of §2.3. It is committed, unlike the probes, because
  §4.5's numbers must stay reproducible until `llama.cpp-7v5c` replaces it.

T8 should lift `upstream_set_input_body.inc`, `ds_sim.cpp`'s `ds_rows()` and
both Python transcriptions into CPU-only unit tests, so that a change to either
hash fails a gate instead of silently changing the rows the cache predicts.
