# Abort and wrong-output census of open tickets (2026-11-16)

Source: codescout task_list (open and in_progress), read in full with descriptions and comments. Titles and text are taken from the tickets. "(my reading)" marks my own judgment.

Columns: Default = does a default, flag-free run reach it. Last evidence = latest date the ticket text shows it still happening. "ticket updated" means no run date is stated and I give the ticket's update date instead.

| Id | Title | Pri | Trigger | Default | Last evidence | Workaround | Root cause per ticket | Related |
|---|---|---|---|---|---|---|---|---|
| 83qz | GPT-OSS 20B on two cards loses the B70 at first prompt expert op (GPU page faults, exit 69) | P1 | `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 llama-cli` GPT-OSS 20B chat gate, default settings, B70+B50, driver 26.44 | yes | 2026-11-16 | yes: `GGML_SYCL_SCHED_ALL_DEVICES=1` passes (cross-card expert path off); also `GGML_SYCL_MOE_MULTI_GPU=0` per project memory | unknown (cross-card expert path is the fault path) | not caused by the exec2 spike; same on integ/th32 d0118bb6c; see dp4p |
| dp4p | GPT-OSS 120B on two cards: prompt batch refused, decode fails, process aborts at device < dev_count | P2 | `level_zero:0,1 llama-perplexity` GPT-OSS 120B `-c 512 -b 512 -ub 512`, default settings | yes | 2026-11-16 | partial: spike branch with `GGML_SYCL_SCHED_ALL_DEVICES=1` gave correct PPL on Qwen 3.8 (not a fix) | unknown; ticket tracks a scheduler-visible second backend (my reading: design work) | see 83qz, iv5a |
| iv5a | llama-perplexity exits 0 and prints no perplexity when every decode fails | P2 | same two-card 120B run as dp4p | yes | 2026-11-16 | no | unknown; exit-status handling in perplexity tool | follow-on of dp4p |
| 8aw1 | Qwen3.6-27B on B70 at default context aborts: oneDNN Graph scratch DIRECT allocation exhausted (718.59 MB) | P1 | llama-server, Qwen3.6-27B-UD-Q4_K_XL, B70, no `-c` (n_ctx 262144, auto ubatch 2048), ~2840 token prompt | yes | 2026-10-01 | unclear (my reading: smaller `-c` or `-ub 512`) | plan never accounted for the runtime SDPA scratch request when KV is demoted and VRAM is full (lead triage) | see fkpg, zhcn, 6rdu, t31w |
| 6rdu | Qwen3.5-9B Q6_K_XL `-d 32768` at auto ubatch hangs on B70 in 9th prompt graph (n_kv 16384) | P1 | `llama-bench -p 0 -n 4 -d 32768 -fa 1`, auto ubatch 2048, B70 | yes | 2026-10-17 | yes: `-ub 512` completes; `GGML_SYCL_FA_ONEDNN=0` completes | hang is in the oneDNN SDPA flash-attention route (attribution run) | see fkpg, t31w |
| t31w | Qwen3-14B Q4_0 on B70 refuses every prompt graph at n_ubatch 2048 (DEQUANT-F16, res=-2) | P1 | `llama-bench -p 0 -n 32 -d 8192 -fa 1`, auto ubatch | yes | 2026-10-17 | not stated (my reading: `-ub 512`) | oneDNN PP pair bound planned at ubatch 512 but runtime uses 2048 | reproduced on Mistral 7B Q4_0; see fkpg |
| 1rc4 | B50 GPT-OSS 20B llama-bench cannot create its second context (tg128): planned dense scratch spills outside arena | P1 | `level_zero:1 llama-bench -m gpt-oss-20b-mxfp4.gguf -p 512 -n 128 -r 1 -v` | yes | 2026-11-06 | none stated | planned dense scratch compute buffers spill outside the VRAM arena | see pgay, zhcn |
| 6uje | GPT-OSS 120B host-experts count gate on B70 answers "1," then NBSP garbage | P1 | CLAUDE.md GPT-OSS gate with the 120B model, `level_zero:0` | yes | 2026-11-02 | none stated | unknown; defect is on master, pinned expert pool involved | not introduced by th32 lane |
| 30h4 | Qwen3.6-27B produces garbage logits in prompt processing (PPL 473319 vs Mistral 1.0097) | P1 | `llama-perplexity` Qwen3.6-27B on B50, model larger than card so weights host-tier | yes (my reading: any host-tiered run) | 2026-10-01 (still wrong at master ce348d2f7) | not stated | ticket says root cause found 2026-09-19 and a fix landed (64ec60199), but PP is still wrong on host-tiered B50 | duplicates va1g; related fzrj, eyqw |
| wb3d | Qwen3.6-27B default-context output is nondeterministic run to run on B70 | P1 | `llama-completion` Qwen3.6-27B, no `-c`, B70, seed 42 temp 0; two md5s seen | yes | 2026-10-01 | none stated | unknown; SAFE_MODE gave the same alternate text so not a missing wait | see 8aw1 |
| 57w3 | Qwen3.8 IQ3_XXS temp-0 completion on B70 is two-valued run to run on ring/CPU mixed path | P2 | `llama-completion` Qwen3.8-Flash-Next IQ3_XXS, B70, default MoE exec | yes (my reading) | 2026-10-30 | none stated | unknown | see wb3d |
| etaq | Oversized model on B50 SIGABRTs in layer_stream_manager::load_layer_sync (zero-extent source handle, mem-ops.cpp:61) | P1 | Qwen3.6-27B-UD-Q4_K_XL (16.67 GiB) on B50 `level_zero:1`, decode | yes | 2026-09-18 | none stated | weight-streaming path that the placement ruling forbids; zero-extent handle | same mem_copy extent=0 family as 5d14, mpps, 1gp0 |
| 37na | test-llama-archs -a llama fails to load on `level_zero:0,1`: "compute-slot measure failed ... plan override nested (refused)" | P1 | `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./build/bin/test-llama-archs -a llama` | yes (my reading: also any two-card load) | 2026-11-02 | none stated | unknown; not introduced by 5eb8407fc, present at 0665bc65e | |
| 1gp0 | llama-eval-callback aborts in deferred CPU scatter flush at layer 28 of Qwen3.8 prefill (mem_copy range rejected) | P2 | `llama-eval-callback` Qwen3.8 IQ3_XXS on B70, code.txt prompt | yes (tool run, no special flags) | 2026-10-09 | none stated | unknown; extent=25600 vs size 81920 on ffn_moe_swiglu-28 | mem_copy family (etaq, 5d14, mpps) |
| sw2oj | GPT-OSS produces garbage output on fresh boot, all kernel paths | P0 | B50/B580 GPT-OSS after fresh boot | yes | ticket updated 2026-09-02 (triage refuted a proposed closure) | none stated | unknown | epic jx71 |
| fzrj | Sparse token corruption on host-tiered runs, ~2/3 of runs | P2 | host-tiered runs (Mistral PPL runs, B50) | unclear | ticket updated 2026-09-19 | none stated | unknown; separate from 30h4 | split from 30h4 |
| l9i1 | MXFP4 MUL_MAT_ID produces NaN for every n > 32 (route_layout=soa vs materialized aos) | P1 | `test-backend-ops test -o MUL_MAT_ID` on `level_zero:0,1` | unclear (GPT-OSS gate passes per CLAUDE.md) | 2026-09-18 | none stated | layout selector disagreement; owner ruling: dispatch must read the loaded layout | see yitq, t98c |
| yitq | SYCL claims MUL_MAT_ID unconditionally in supports_op; 234 uncovered-type cases give wrong numbers (in progress, P0) | P0 | `test-backend-ops -o MUL_MAT_ID` f32, f16, iq2_xxs, bf16 | unclear | 2026-09-18 | none stated | supports_op returns true before any src0 type check | see l9i1, t98c |
| ve38 | Prompt micro-batch above 512 rows can fault the GPU on a MoE model (CAT error, DEVICE_LOST) | P1 | `llama-perplexity` Qwen3.8 exps-IQ4_NL, `-ub 1024`, B70, driver 26.44 | no (needs `-ub 1024`) | 2026-11-15 | yes: stay at `-ub 512` or lower | unknown; 1 fault in 4 finished plain runs | related ohkx |
| 7q93 | Host-resident KV cache corrupts logits non-deterministically | P1 | Mistral 7B Q4_0 PPL on B50 with `GGML_SYCL_KV_HOST=1` | no (debug env) | 2026-09-19 | not setting the env | unknown | see mnqa |
| q0j5 | B50 FLASH_ATTN OUT_OF_RESOURCES, failed CPU fallback, then hang and GPU page fault | P1 | `llama-perplexity` Qwen3.6-27B B50 `-ub 1024 -b 2048` | no (needs `-ub 1024`) | 2026-10-02 | smaller `-ub` (my reading) | context-init check wrongly admitted `-ub 1024` (fixed in kpjw per text) | dup class of 8dd9 |
| 8dd9 | Process hangs after "CPU fallback also failed ... failing this graph cleanly" (FA OUT_OF_RESOURCES) | P2 | `llama-perplexity` Qwen3.6-27B B50, auto-ub 1024 | unclear | 2026-10-01 | none stated | VRAM 107.8 MB free after a hold-induced spill | see q0j5, qyza |
| 5d14 | Gemma4 E4B aborts at mem-ops.cpp:61 (mem_copy extent=0) under low VRAM budget | P1 | `GGML_SYCL_VRAM_BUDGET_PCT=30`, B70, Gemma4 E4B Q8_0 | no (budget env) | 2026-09-27 | not setting budget | unknown; 3210 residency declines before abort | mem_copy family |
| 34lq | GPT-OSS 20B on B50 emits garbage once any expert is host-tiered | P1 | `GGML_SYCL_VRAM_BUDGET_PCT=88`, B50, chat gate, CpuExpertPool path | no (budget env) (my reading: reachable when the card is full) | 2026-09-26 | PCT=89 passes | unknown; pre-existing on master | |
| va1g | Forced whole-layer host demotion emits garbage tokens (Mistral B50) | P2 | `GGML_SYCL_VRAM_BUDGET_PCT=20` Mistral Q4_0 B50 | no | 2026-09-19 | none | same defect as 30h4 (ticket comment) | duplicate of 30h4 |
| nx1f | Weights forced onto a SYCL DEVICE buffer with `-ot` while weights_evictable ON give silently wrong tokens | P1 | `-ot` device placement, B50 llama-completion | no (`-ot` flag) | ticket updated 2026-09-02 | not using `-ot` | dual residency | |
| 480a | MoE MMID h2d staging exhausts pinned pool: 32-byte alloc aborts at mem-ops.cpp:393 | P1 | `test-backend-ops` with 192 GB free; also 2 failures in 1156 staging allocs on Mistral | unclear (model-agnostic race per ticket) | ticket updated 2026-09-02 | retry rescued 2 of 2 in Mistral | low-probability race in metadata publication step | |
| lhmy | Fattn Q-type assert reachable via quantized KV plus partial CPU offload | P2 | `-ctk q8_0 -ctv q8_0`, GPT-OSS 20B on B50, no `-c` (planner demotes KV) | no (needs KV type flags) | 2026-10-17 | use f16 KV | CPU fattn asserts q type is F32 (ops.cpp:8849) | same as o6bi |
| o6bi | CPU flash-attn aborts with quantized K cache (Q pre-cast to F16) | P2 | type_k=Q8_0, CPU-only context, SYCL build | no | 2026-10-01 | f16 K cache | Q pre-cast to F16 by SYCL graph path | same as lhmy |
| pgay | Second model load in one process with wider micro-batch aborts at VRAM zone sizing (0.1 MB short) | P2 | `llama-bench` GPT-OSS 20B two contexts, B50, wide ubatch | unclear | 2026-11-16 | none stated | first model's leases still live | see 1rc4, ouur |
| ouur | Second model's late arena-zone rebuild aborts instead of refusing the load | P2 | second model load via compute_and_store_plan_for_inventory | unclear | ticket updated 2026-10-09 | none | GGML_ABORT where a named refusal is wanted | see pgay |
| i7wd | GPT-OSS on B70 with MOE_GRAPHLET_REPLAY_PROBE=1 + FLASH_ATTN_GRAPH_ALLOW=1 faults the GPU (CAT, DEVICE_LOST) | P2 | those two env vars, B70 | no | 2026-09-25 | do not set the probes | unknown | |
| rtf1 | B50 f16 opt-out arm: error-40 in FLASH_ATTN_EXT at p512 plus hang after throw | P2 | `GGML_SYCL_MOE_PP_ONEDNN_F16_BATCHED=1 GGML_SYCL_MOE_PP_WOQ=0` | no | 2026-08-22 | do not set | unknown | |
| mpps | `GGML_SYCL_PERSISTENT_TG=1` aborts in extract_persistent_plan (mem_copy extent=0) | P2 | env var, Mistral llama-bench, both cards | no | 2026-09-01 | leave env off | zero-extent handle | mem_copy family |
| zw4b | `GGML_SYCL_MOE_GROUPED_DECODE=1` fails first decode batch (res=-3) | P2 | env var, GPT-OSS 20B, B70 | no | 2026-09-01 | leave env off | unknown | |
| r0e7 | `GGML_SYCL_MOE_MULTI_GPU=1` aborts in sycl_tensor::resolve_as (null pointer) on Qwen3.8 decode | P3 | env var, `level_zero:0,1` | no | 2026-10-16 | leave env off | null pointer in resolve_as | see 83qz |
| raky | CPU-offload device discovery terminates instead of falling back | P2 | `GGML_SYCL_CPU_OFFLOAD=1` + opencl:cpu selector | no | 2026-09-18 | do not set | catch narrower than throw | |
| mhyw | release_graph_retained_handles frees other contexts' leases (test-thread-safety) | P1 | test-thread-safety | no (test) | 2026-09-02 | none | ticket re-scoped; exit is 134 not 139 | see 09ep |
| 09ep | test-thread-safety SEGFAULTs in 2.5 s under a single visible GPU | P1 | `level_zero:1 ctest -R test-thread-safety` | no (test) | 2026-08-01 | none | fix chain in tree; one pinned run missing | |
| eslf | B70 Engine memory CAT error during test-sycl-cpu-job-signal-ring (1 of 6 runs) | P1 | ctest gate on B70 | no (test) | 2026-10-18 | none | unknown | |
| vp53 | Unidentified tenant faults the B50 (engine resets, bcs CAT errors) with no llama process | P2 | host kernel log 2026-11-14 | no | 2026-11-14 | none | unknown, no llama process involved | |
| kjoc | Backend queues had no async_handler: async errors hit the SYCL default handler and terminated | P1 | Q1/NVFP4 route terminal failure injection | unclear | 2026-10-10 | n/a | fixed: every backend queue and context has the recording handler (async-failure.hpp); a delivered error fails its device's contexts at the next graph boundary (`llama_decode` -3), one graph late | |
| r2zl | TP row-parallel FFN silently drops device 1's partial sum when buffer alloc fails | P1 | tensor-parallel path | no (TP opt-in, my reading) | 2026-09-26 | none | code path skips accumulation after fprintf | found statically |
| m49a | set_rows on device N reads i64 row index from another device's host USM | P1 | any whole-graph two-block config | unclear | 2026-09-27 | none | cross-context read | found by trace |
| r7fz | populate_inventory_globals releases another live model's PP MoE ring at load | P1 | second model load | no (my reading: multi-model) | 2026-09-26 | none | release at every load | predates u1bb |
| 94z1 | stage_inventory_plan temporary backends run process-global teardown inside a model load | P1 | model load | unclear | 2026-09-27 | none | verified by lead at c69d5774d | |
| wais | ESIMD partitioned decode tier wrong for masked multi-query at D=512 | P1 | D=512 FA case ne01>1 | unclear | 2026-09-04 | none | garbage in test case | |
| sfal | Pinned buffer pool sized from init graph: host-resident MoE prompt aborts GGML_ASSERT | P1 | Qwen3.6-35B-A3B Q5_K_S on B50 | unclear | 2026-09-27 (does not reproduce) | n/a | pool sized at 16 | see glkg |
| 1lrh | Zero-device SYCL registration aborts (test-sycl-env-report exits 134) | P1 | host with no GPU | no | 2026-09-02 | n/a | non-fatal dev_mgr in 33867aa21 | see qhm5 |
| 5a28 | gemma4 llama-completion aborts at default env: custom template not supported | P3 | gemma4 E4B no `--jinja` | yes | 2026-09-03 | yes: add `--jinja` | chat template parse | |
| 71hx | PP staging fallback consumes weight pointers without verifying layout: silent garbage | P2 | staging route with XMX_TILED | no | ticket updated 2026-09-01 | none | triage refuted a closure | |

## Probably fixed but still open (my reading, from comments)

- glkg: comment says "the described defect is fixed and proven on this host with the reporter's exact model" (master 70d701849).
- sfal: 2026-09-27 probe rc=0, abort does not reproduce.
- 1lrh: commit 33867aa21 made dev_mgr non-fatal on zero devices; 2026-09-02 comment proves the zero-device path on the converged branch.
- qhm5: depends on the 1lrh fix landing.
- mhyw: comment says "RESOLVED, premise dissolved"; ticket re-scoped.
- t98c: comment says resolved at the reporting layer; 25 cases structurally unfixable in that harness.
- va1g: comment says same defect as 30h4; close as duplicate once 30h4 lands.
- 09ep: fix chain is in tree; only one pinned test run is missing.
- u3usl: comment says FA-off chat gate now passes on B50 (partial).
- zhcn, moua, 23mk: comments say W7 landed on master 43f9d0e4e; tickets still in progress.
- 5a28: workaround known (`--jinja`), gate text only.
- mnqa: owner decision made 2026-09-26, fix path stated.

## Counts

- Open tickets scanned: 993 (977 open plus 16 in progress), all with descriptions and comments.
- Tickets picked: 51 (rows in the table above).
- Note: I picked by keyword and priority filters over descriptions (P0 to P2 mostly, plus a few P3). Low-priority P3 and P4 tickets and test-gate tickets were not individually read. Some defects of this type may be missing.

## Re-verification on 2026-11-16 (lead, integ/th32 `d0118bb6c`, driver 26.44, one run per ticket)

Tracked under llama.cpp-eb3s. Every run ended with 0 kernel GPU lines. The table above is the census as read from
the tickets before these runs.

| Ticket | Result | What the run showed |
|---|---|---|
| 1rc4, t31w, 6rdu, 34lq, etaq, 1gp0 | closed, does not reproduce | the ticket's command completes with correct output |
| 30h4 | closed, does not reproduce | Qwen3.6-27B on the B50: chunks 1.0032, 1.0017, 1.0012, 1.0010 |
| va1g | closed, does not reproduce | Mistral Q4_0 on the B50 at a 20% VRAM budget (3628 MB of weights on the host): chunks 1.0087, 1.0072, 1.0063, 1.0063 |
| 37na | reproduces | two-card `test-llama-archs -a llama`: `plan override nested (refused)`, rc 255 |
| 5a28 | reproduces | gemma4 `llama-completion` without `--jinja`: `terminate called`, rc 134, core dump. Fix: llama.cpp-bgg9 |
| 6uje | reproduces | GPT-OSS 120B chat gate on the B70: `1,` then non-breaking spaces, also at `-c 4096`. A raw 10-token completion on the same model is correct |
| 8aw1 | open, re-scoped | no abort through `llama-completion` at auto micro-batch 1024; the oneDNN Graph direct fallback fired 3 times. Its exhaustion must become a named refusal (width plan, task W3.2d) |
| 83qz | reproduces by design of the default | root cause read: one variable gates both the two-card plan and the cross-card expert path. Fix: llama.cpp-5r7k |

Not yet re-run: wb3d, 57w3, fzrj, iv5a, 480a, kjoc, sw2oj, l9i1, yitq, dp4p.
