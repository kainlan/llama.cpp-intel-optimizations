# TKV-3 — Recurrent state size census (llama.cpp-zsyj, llama.cpp-tlb4)

CPU-only. No GPU runs, no model loading. All numbers below are computed from
hparams read directly out of the fixture-generating source
(`tests/test-llama-archs.cpp`) and a live HF Hub `config.json` for the
realistic model, using the exact formula the fork uses at runtime. No
`test-generate-models` fixture GGUFs were available in the checked-out
`build/` (its `CTestTestfile.cmake` is stale relative to `tests/CMakeLists.txt`
— `test-generate-models`/`test-download-model`'s newer registrations are
missing; ctest only lists `test-recurrent-state-rollback` itself, not its
`generate-models` fixture setup). Per this task's scope ("No builds needed; if
you believe you need one, ask the lead") this census does not reconfigure or
rebuild; every hparam below is instead cited from the exact source lines that
would produce that GGUF, which is equivalent for a purely arithmetic census
and keeps the answer reproducible without touching `build/`.

## 1. The formula

`llama_memory_recurrent`'s per-layer state tensors (`src/llama-memory-recurrent.cpp:125-127`):

```cpp
const uint32_t n_rows = mem_size * (1 + n_rs_seq);
ggml_tensor * r = ggml_new_tensor_2d(ctx, type_r, hparams.n_embd_r(), n_rows);
ggml_tensor * s = ggml_new_tensor_2d(ctx, type_s, hparams.n_embd_s(), n_rows);
```

so per-layer bytes are `n_embd_r() * n_rows * sizeof(type_r)` (`cache_r_l%d`)
and `n_embd_s() * n_rows * sizeof(type_s)` (`cache_s_l%d`), summed only over
layers where the arch's recurrent-layer filter passes (non-recurrent layers
get no `r_l[i]`/`s_l[i]` tensor at all — `:94-98`). Both the pure-recurrent
construction path (`src/llama-model.cpp:2977-2985`) and the hybrid path
(`src/llama-memory-hybrid.cpp:54-65`, called from `src/llama-model.cpp`'s
`default:` case) always pass `type_r = type_s = GGML_TYPE_F32` (4 bytes).
`mem_size` is `std::max(1, cparams.n_seq_max)` and `n_rs_seq = cparams.n_rs_seq`
for both paths.

`n_embd_r()`/`n_embd_s()` (`src/llama-hparams.cpp:183-229`) branch on
model-specific flags (RWKV `wkv_head_size`, LFM2 `n_shortconv_l_cache`, Kimi/
MiniMax `n_embd_head_kda`/`n_embd_head_la`); none of those are set for
QWEN35/NEMOTRON_H/DEEPSEEK4, so all three fall through to the base Mamba
formula:

```
n_embd_r = (ssm_d_conv - 1) * (ssm_d_inner + 2*ssm_n_group*ssm_d_state)
n_embd_s = ssm_d_state * ssm_d_inner
```

## 2. Fixture archs

Base hparams shared by all `test-llama-archs.cpp` fixtures unless overridden
(`:224-273`): `n_vocab=128, n_embd=256, n_head=2, n_ff=384, n_layer=2`. SSM
keys (`:438-443`, arch-conditional only for `ssm_d_inner`):

```
ssm_d_inner = 256                      if QWEN3NEXT/QWEN35/QWEN35MOE else 2*n_embd
ssm_d_conv  = 4                        (all archs)
ssm_d_state = 128                      (all archs)
ssm_n_group = 2                        (all archs except PLAMO2, which is 0)
```

Both fixture tests are driven by the same binary/context params
(`tests/test-recurrent-state-rollback.cpp:11-14`, `make_ctx`): `n_seq_max = 1`,
`n_rs_seq = 8` ⇒ `mem_size = max(1,1) = 1`, `n_rows = 1*(1+8) = 9`.

### 2a. `qwen35-dense` (LLM_ARCH_QWEN35)

No per-arch override to `n_embd`/`n_head`/`n_ff`/`n_layer` (not in any branch
at `:234-273`), so `n_embd=256, n_layer=2`; `ssm_d_inner=256` (QWEN35 special
case, `:439`).

Recurrent-layer pattern: `src/models/qwen35.cpp:17-25` — no
`ATTENTION_RECURRENT_LAYERS` array is supplied by the fixture, so it falls to
`full_attn_interval` read from `LLM_KV_FULL_ATTENTION_INTERVAL`, which the
fixture sets to `2` unconditionally (`tests/test-llama-archs.cpp:306`):
`is_recr[i] = (i+1) % 2 != 0` ⇒ layer 0 recurrent, layer 1 not. **1 of 2
layers recurrent.**

| field | value | source |
|---|---|---|
| n_layer (total / recurrent) | 2 / 1 | `test-llama-archs.cpp:233`, `models/qwen35.cpp:17-25` |
| n_embd_r | (4-1)×(256+2×2×128) = **2304** | `llama-hparams.cpp:204` |
| n_embd_s | 128×256 = **32768** | `llama-hparams.cpp:228` |
| mem_size | 1 | `test-recurrent-state-rollback.cpp:13` |
| n_rs_seq | 8 | `test-recurrent-state-rollback.cpp:14` |
| n_rows | 1×(1+8) = 9 | formula |
| r bytes/recurrent layer | 2304×9×4 = 82,944 B | |
| s bytes/recurrent layer | 32768×9×4 = 1,179,648 B | |
| **total (1 recurrent layer)** | **1,262,592 B ≈ 1.20 MiB** | |

Sanity check against the live evidence in `llama.cpp-zsyj`'s description: the
observed failing write was `new_state-0 (131072 B) -> cache_s_l0`. `131072 =
32768 × 4` exactly — one row of `cache_s_l0` at this arch's `n_embd_s`,
confirming the formula and the 32768 value independently of this census.

### 2b. `nemotron_h-dense` (LLM_ARCH_NEMOTRON_H)

Override: `n_layer = 3` only (`test-llama-archs.cpp:267-268`); `n_embd=256`
stays default, so `ssm_d_inner = 2×256 = 512` (not in the QWEN35/NEXT special
case, `:439` else-branch).

Recurrent-layer pattern: `src/models/nemotron-h.cpp:13-18` —
`is_recr[i] = n_head_kv(i)==0 && n_ff(i)==0`. The fixture's per-layer arrays
for NEMOTRON_H (`:308-318` head count, `:291-297` FF count) give:

| il | n_head(_kv) | n_ff | is_recr |
|---|---|---|---|
| 0 | 2 | 0 | false (head≠0) |
| 1 | 0 | 0 | **true** |
| 2 | 2 | 384 | false |

**1 of 3 layers recurrent** (layer 1).

| field | value | source |
|---|---|---|
| n_layer (total / recurrent) | 3 / 1 | above |
| n_embd_r | (4-1)×(512+2×2×128) = **3072** | `llama-hparams.cpp:204` |
| n_embd_s | 128×512 = **65536** | `llama-hparams.cpp:228` |
| mem_size | 1 | same test binary |
| n_rs_seq | 8 | same test binary |
| n_rows | 9 | formula |
| r bytes/recurrent layer | 3072×9×4 = 110,592 B | |
| s bytes/recurrent layer | 65536×9×4 = 2,359,296 B | |
| **total (1 recurrent layer)** | **2,469,888 B ≈ 2.36 MiB** | |

### 2c. `deepseek4-moe` (dsv4 / LLM_ARCH_DEEPSEEK4) — does NOT go through this formula

**Load-bearing finding, not a fixture-shaped guess:** `LLM_ARCH_DEEPSEEK4` has
its own `case` in the memory-construction switch (`src/llama-model.cpp:2894-2936`)
that runs *before* the `default:` branch which builds `llama_memory_hybrid`/
`llama_memory_recurrent` (`:2963-2985`). DEEPSEEK4 always constructs
`llama_kv_cache_dsv4` instead — it never creates `cache_r_l%d`/`cache_s_l%d`
tensors, `hparams.n_embd_r()`/`n_embd_s()` are never called for it, and
`llm_arch_is_hybrid(DEEPSEEK4) == true` (`llama-arch.cpp:1012`) is consulted
elsewhere (e.g. `llm_arch_supports_rs_rollback`, `llama-arch.cpp:1036`) but
**not** by the memory-construction switch, precisely because DEEPSEEK4 is
matched by its own case first.

DEEPSEEK4's analogous per-layer state lives in `llama_dsv4_comp_state`
(`src/llama-kv-cache-dsv4.cpp:893-998`), named `dsv4_%s_state_kv_l%d` /
`dsv4_%s_state_score_l%d` (`:967-968`) — a different prefix than
`cache_r_l`/`cache_s_l`, sized by `n_embd_state × state_size × n_planes`
(`n_planes = n_stream×(1+n_rs_seq)`, `:962`), not by `n_embd_r()`/`n_embd_s()`.

**Further finding: `llama_dsv4_comp_state` does not route through the tiered
SYCL KV buft at all.** Its buft selection (`:950-955`) calls
`ggml_backend_dev_buffer_type(dev)` directly — the same call
`llama_recurrent_sycl_kv_buft`'s existing fallback already uses
(`llama-memory-recurrent.cpp:109`) — never
`ggml_backend_sycl_kv_buffer_type_from_dev`. So this tensor class was never
subject to the `tiered_kv_buffer_init_tensor` prefix-matching bug in the first
place, and Task 10's direction-(b) fix (which only edits
`llama-memory-recurrent.cpp`) **cannot be what makes
`test-recurrent-state-rollback-dsv4` RED today, and will have zero effect on
it either way** — it is a disjoint code path from the one this ticket's fix
touches.

**Recommendation for Task 10 / the owner:** do not assume the direction-(b)
patch turns `test-recurrent-state-rollback-dsv4` green. Root-cause its RED
failure separately (`GGML_SYCL_DEBUG=1`, one lead-run, to see which tensor's
copy is actually rejected for `deepseek4-moe.gguf`) — most likely one of the
four inner attention KV caches (`kv_raw`/`kv_csa`/`kv_hca`/`kv_lid`, ordinary
`llama_kv_cache`/`llama_kv_cache_iswa` instances that DO use the standard
`cache_k_l%d`/`cache_v_l%d` naming and DO go through the tiered SYCL buft), not
the recurrent-state mechanism this census was scoped to measure.

For completeness, the dsv4 fixture's own comp-state size (informational only,
not gated by this census's verdict, computed from
`tests/test-llama-archs.cpp:246-251` giving `n_embd=512, n_head=8, n_layer=4`;
`:400-403` giving `compress_ratios=[0,0,4,128]`, `indexer_head_count=n_head=8`,
`indexer_head_size=64`; and the ctor wiring at `llama-kv-cache-dsv4.cpp:1315-1329`):

| comp_state | ratio | state_size | n_embd_state | applies to (of 4 layers) | bytes (2 tensors × n_planes=9) |
|---|---|---|---|---|---|
| csa | 4 | 8 | 2×n_embd_head_k=128 | layer 2 only (ratio==4) | 73,728 B |
| hca | 128 | 128 | n_embd_head_k=64 | layer 3 only (ratio==128) | 589,824 B |
| lid | 4 (uses filter_csa) | 8 | 2×indexer_head_size=128 | layer 2 only | 73,728 B |
| **total** | | | | | **737,280 B ≈ 720 KiB** |

## 3. Realistic full-size hybrid model config

Fixtures are 2-3 layer toy models (`fixture-shapes-take-different-paths` risk
per the plan's gotchas), so the direction decision is anchored on a real,
currently-shipping hybrid model instead: **NVIDIA Nemotron-Nano-9B-v2**
(`nemotron_h` arch — same family as the `nemotron_h-dense` fixture), config
fetched live from `hf://models/nvidia/NVIDIA-Nemotron-Nano-9B-v2/config.json`:

```
num_hidden_layers = 56
hidden_size       = 4480
mamba_num_heads   = 128, mamba_head_dim = 80   -> ssm_d_inner = 128*80 = 10240
mamba_num_groups  = 8                          -> ssm_n_group = 8
ssm_state_size    = 128                        -> ssm_d_state = 128
conv_kernel       = 4                          -> ssm_d_conv  = 4
hybrid_override_pattern = "M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-"
```

The pattern string is exactly 56 characters (one per layer, no separators
beyond the characters themselves): `Counter = {'M': 27, '-': 25, '*': 4}` —
**27 of 56 layers are recurrent** (Mamba/SSM), 25 are FFN-only dense, 4 are
attention.

```
n_embd_r = (4-1) * (10240 + 2*8*128) = 3 * 12288 = 36,864
n_embd_s = 128 * 10240               = 1,310,720
```

`llama_context_params` defaults (`src/llama-context.cpp:3964-3970`):
`n_seq_max = 1`, `n_rs_seq = 0` (documented `[EXPERIMENTAL], 0 = no rollback`,
`include/llama.h:356`, and clamped to 0 for any arch/config where the caller
doesn't explicitly request it — `llama-context.cpp:359-364`) ⇒
`mem_size = 1`, `n_rows = 1`. This is the realistic default (a single chat
session, no rollback feature enabled):

| n_seq_max | n_rs_seq | n_rows | r bytes/layer | s bytes/layer | total (27 layers) | % of 16 GiB |
|---|---|---|---|---|---|---|
| **1 (default)** | **0 (default)** | 1 | 147,456 B | 5,242,880 B | **145,539,072 B ≈ 138.8 MiB** | **0.847 %** |
| 8 (concurrent-server) | 0 | 8 | 1,179,648 B | 41,943,040 B | 1,164,312,576 B ≈ 1.08 GiB | 6.78 % |
| 1 | 8 (rollback on) | 9 | 1,327,104 B | 47,185,920 B | 1,309,851,648 B ≈ 1.25 GiB | 7.62 % |

The default single-session, no-rollback configuration — the common case, and
the only one that doesn't require a caller to opt into an experimental feature
— sits at **0.85 % of a 16 GiB budget**, comfortably under the 5 % gate. Note
the caveat, though: `n_seq_max` scales the total *linearly* (it's a row-count
multiplier applied uniformly, not a fixed cost), so a concurrent-serving
deployment with `n_seq_max ≳ 6` on this exact model would cross 5 % on its
own, independent of whether rollback is enabled. That is a real scaling axis
this decision should stay aware of, but it does not change the verdict for the
default/typical case this gate is meant to answer, and the 5% line is not a
hard cliff either way — it is a heuristic for "is this worth the design cost
of tiering recurrent state" vs. "just keep it device-resident."

## 4. Verdict

**Total recurrent-state bytes at the realistic default configuration
(NVIDIA Nemotron-Nano-9B-v2, `n_seq_max=1`, `n_rs_seq=0`): 145,539,072 B ≈
138.8 MiB ≈ 0.847 % of a 16 GiB budget.** The two fixture archs that actually
exercise `llama_memory_recurrent` (`qwen35-dense`, `nemotron_h-dense`) are
smaller still (≈1.2 MiB and ≈2.36 MiB respectively — sub-0.02 %). All three
numbers are well under the 5 % gate.

**`< 5% of a 16 GB budget` ⇒ DIRECTION (b): route `cache_r_l`/`cache_s_l`
through the plain device buft (`llama_recurrent_sycl_kv_buft` returns
`nullptr`), not direction (a) (teach the tiered arena recurrent geometry).**

Caveats carried into Task 10:

1. **This verdict does not cover `deepseek4-moe`/DEEPSEEK4.** That arch does
   not use `llama_memory_recurrent` at all (§2c) — Task 10's direction-(b)
   edit to `llama-memory-recurrent.cpp` cannot affect, explain, or fix
   whatever currently makes `test-recurrent-state-rollback-dsv4` RED.
   Root-cause that test's failure separately before assuming the fix covers
   it; expect its rejected tensor to be one of `kv_raw`/`kv_csa`/`kv_hca`/
   `kv_lid`'s standard `cache_k_l`/`cache_v_l` tensors, not a recurrent-state
   tensor.
2. A concurrent-serving deployment (`n_seq_max ≳ 6-8`) on a large hybrid model
   pushes recurrent-state bytes past 5% on `n_seq_max` scaling alone (§3) —
   worth a follow-up note if/when a server-mode gate is defined for this
   feature, but out of scope for the direction decision here (the default,
   single-session case is what "the recurrent state doesn't fit tiered
   attention geometry" is actually about).

## Sources cited

- `src/llama-memory-recurrent.cpp:39-47,94-98,104-116,125-127` (formula, buft
  fallback, `type_r`/`type_s=F32`)
- `src/llama-hparams.cpp:183-229` (`n_embd_r()`/`n_embd_s()`)
- `src/llama-model.cpp:2894-2985` (memory-construction switch: DEEPSEEK4's own
  case vs. the `default:` hybrid/recurrent case)
- `src/llama-memory-hybrid.cpp:11-65` (hybrid ctor, `type_r=type_s=F32`,
  `rs_size` argument)
- `src/llama-arch.cpp:996-1018,1032-1040` (`llm_arch_is_hybrid`,
  `llm_arch_supports_rs_rollback`)
- `tests/test-llama-archs.cpp:224-273,306,388-403,438-443` (fixture hparams)
- `src/models/qwen35.cpp:17-25`, `src/models/nemotron-h.cpp:13-18`
  (recurrent-layer masks)
- `tests/test-recurrent-state-rollback.cpp:11-18` (test context params)
- `src/llama-kv-cache-dsv4.cpp:893-998,950-955,1210-1330` (dsv4 comp-state
  sizing and buft selection)
- `src/llama-context.cpp:359-364,3964-3970`; `include/llama.h:351-356`
  (`n_seq_max`/`n_rs_seq` defaults and semantics)
- `hf://models/nvidia/NVIDIA-Nemotron-Nano-9B-v2/config.json` (live HF Hub
  fetch, realistic model hparams)
- `llama.cpp-zsyj` ticket body (evidence cross-check: observed `size=131072`
  matches `qwen35-dense`'s computed `n_embd_s×4` exactly)
