# MoE `down` MXFP4_I8 layout decline — diagnosis and options

**Date:** 2026-07-24
**Task:** `llama.cpp-s88q` (B70 plan, T8) — diagnosis only, no fix implemented.
**Model under test:** `gpt-oss-20b-mxfp4.gguf` (24 layers, 32 experts, 2880 × 2880 expert matrices).
**Cards:** Arc Pro B50 (16 GB, `level_zero:1`), Arc Pro B70 (32.6 GB, `level_zero:0`).

> **Units.** Every log line in this area prints `%.1f MB` after dividing by 1024², i.e. the
> printed "MB" are MiB. This document uses MiB throughout and matches the logs.

---

## 1. What this settles

The `down` expert tensors decline the MXFP4_I8 layout upgrade on the B50 **purely for lack of
VRAM headroom**. Every eligibility counter in the pass is zero. The original plan hypothesis —
that the `GGML_SYCL_MOE_DOWN_SUM_DPAS_DIRECT_FINAL{,_I8}` executor gates blocked the upgrade —
is refuted: `skip_executor=0`, and enabling both gates provably did not change the outcome.

Three independent lines of evidence agree:

| evidence | result | source |
|---|---|---|
| B50 counter dump | 6 of 24 upgraded, every skip counter zero, `remaining=221.7 MB` | `llama.cpp-r5ib` |
| B50 dose–response | +164.3 MiB of budget bought **exactly one** more layer (5 → 6) | `llama.cpp-r5ib` |
| B70 clean card (32.6 GB free) | **24 of 24** upgraded | `llama.cpp-fuo6` |

The verbatim B50 line (build `bbb8d5529`, ComfyUI stopped, `free=16251.0 MB`, `budget=15675.0 MB`):

```
[MOE-LAYOUT] down-i8 device=0 n_experts=32 considered=2304 candidates=24 upgraded_tensors=6
  upgraded_entries=192 preserved_soa=192 skip_not_down=1536 skip_already_i8=0 skip_not_device=0
  skip_wrong_dev=0 skip_executor=0 skip_unsupported=0 requested=mxfp4_i8 granted=mixed
  remaining=221.7 MB
```

`skip_not_down=1536` is role-filter bookkeeping (the gate/up entries), not a rejection.
Layers `blk.0`–`blk.5` are `layout=mxfp4_i8`; `blk.6`–`blk.23` stay on `soa`.

**The B50 cannot reach 24 of 24 under any charging policy** (§4). That conclusion survives all
three arithmetic corrections in §3 and §4.

---

## 2. The mechanism, read from the code

`maybe_upgrade_moe_down_layouts_to_i8()` — `ggml/src/ggml-sycl/unified-cache.cpp:16375`.

**Candidate construction** (`:16407`–`:16441`) applies every eligibility test: expert role,
already-I8, on-device, target-device, `static_i8_executor_supported`, `planner_mxfp4_i8_supported`.
All 24 down tensors survive it on the B50.

**The upgrade loop** (`:16498`–`:16525`) contains exactly one `continue`, the headroom guard at
`:16499`:

```cpp
if (remaining <= k_layout_upgrade_guard || candidate.extra_charge > remaining - k_layout_upgrade_guard) {
    continue;
}
```

with `k_layout_upgrade_guard = 64 MiB` (`:16380`). So for a tensor that reached `candidates`,
declining is *by construction* a headroom decline. Nothing else can reject it.

### 2.1 The charge is the full I8 cost — and that is honest accounting, not a bug

The lead comment on this task, and `llama.cpp-r5ib`'s notes, describe `:16477`

```cpp
extra_charge += preserve_primary_soa ? new_charge : (new_charge > old_charge ? new_charge - old_charge : 0);
```

as a "double-charge". **It is not a double-charge.** When `preserve_primary_soa` holds, the
upgrade loop at `:16511`–`:16519` explicitly registers the SOA copy as a retained alternate
layout via `planner_entry_add_alternate_layout_on_device(entry, GGML_LAYOUT_SOA, …)` before
flipping the primary to I8 — which is what `preserved_soa=192 == upgraded_entries=192` reports.
Both copies are genuinely resident: SOA for prompt processing, I8 for decode. Charging
`old_charge` (already counted) plus `new_charge` (the added I8 copy) is exactly the memory the
plan will consume.

This matters for the recommendation: **there is no arithmetic error to fix.** The only way to
reach delta pricing is to stop keeping two copies of `down`. Options 1 and 5 in the lead's
comment are therefore the same option, not two.

`preserve_primary_soa` (`:16469`) is false when either of two existing opt-ins is set —
so the delta-charging behaviour already exists behind env gates and needs no new code:

| env var | effect | scope | defined at |
|---|---|---|---|
| `GGML_SYCL_MOE_PP_DOWN_SPECIALIZED_LAYOUTS=1` | PP consumes I8/DPAS `down` directly; no SOA alternate is kept | any plan | `:14871` |
| `GGML_SYCL_MOE_PROMPT_DOWN_TRANSIENT=1` | PP materialises SOA transiently for the selected experts | single-device only | `:14879` |

Setting either also reorders the pass relative to `add_single_moe_pp_executable_alternates`
(`:17488`–`:17505`, and `:19431`–`:19442` for the single-device budget path).

### 2.2 Corrected byte model (derived from the code, not from a byte count)

`planner_layout_bytes_for_dims()` — `:15047`:

* `GGML_LAYOUT_SOA` / `AOS` → `ggml_row_size(MXFP4, ncols) * nrows`; MXFP4 blocks are 17 B per
  32 elements (1 scale + 16 nibble bytes).
* `GGML_LAYOUT_MXFP4_I8` → `nblocks * (QK_MXFP4 + 1)` = **33 B per block** (32 int8 weights +
  1 scale). The scales are *not* dropped.

Per expert (2880 × 2880 → 259,200 blocks), and per `down` tensor (× 32 experts):

| | per expert | per tensor | |
|---|---|---|---|
| SOA (today) | 4,406,400 B | 141,004,800 B | **134.5 MiB** |
| MXFP4_I8 | 8,553,600 B | 273,715,200 B | **261.0 MiB** |
| delta | 4,147,200 B | 132,710,400 B | **126.6 MiB** |

`placement_vram_charge_bytes()` (`unified-cache.hpp:293`) rounds to 256 B — negligible.

The modelled full charge of 261.0 MiB matches the **measured 261.3 MiB per tensor** from the
dose–response. The lead's figures (I8 253.1 MiB, delta 118.7 MiB) omitted the block scales;
they are ~3 % low. Use 261.0 / 126.6.

---

## 3. The arithmetic, corrected

At entry to the down pass, `remaining` is reconstructible exactly, because the loop decrements
`remaining` only by `candidate.extra_charge`:

```
R0 = remaining_final + upgraded_tensors × 261.3
```

| B50 run | free | budget | upgraded | remaining | R0 |
|---|---|---|---|---|---|
| ComfyUI running | 16086.7 | 15510.7 | 5 | 318.7 | **1625.2 MiB** |
| ComfyUI stopped | 16251.0 | 15675.0 | 6 | 221.7 | **1789.5 MiB** |

The two R0 values differ by 164.3 MiB — exactly the budget delta. The model reproduces both
observations:

* `floor((1625.2 − 64) / 261.3) = 5` ✓
* `floor((1789.5 − 64) / 261.3) = 6` ✓ (the 6th missed by 6.6 MiB in the first run)

### 3.1 Counterfactuals on the B50 (usable headroom = R0 − 64 MiB = 1725.5 MiB)

| policy | charge / tensor | layers upgraded | note |
|---|---|---|---|
| full charge (today) | 261.3 MiB | **6** of 24 | observed |
| delta charge (drop the SOA copy) | 126.6 MiB | **13** of 24 | predicted |
| full charge + oneDNN zone at its 256 MiB default (+917.7 MiB) | 261.3 MiB | **10** of 24 | predicted |
| delta charge + oneDNN zone at default | 126.6 MiB | **20** of 24 | predicted |
| all 24 at delta pricing | 126.6 MiB | needs **3102 MiB** at pass entry; has 1789.5 | short by **1313 MiB** |

### 3.2 Correction to the lead's stated ceiling

The task comment compares "24 × 118.7 MiB = 2.78 GiB needed" against "the B50 has 0.47 GiB".
Those two numbers use different baselines: 0.47 GiB (483.0 MiB) is the headroom *after five
layers had already been charged at full price*, so the five paid-for layers are counted on the
cost side and subtracted on the supply side. The consistent comparison is at pass entry:
**3.03 GiB of delta charges needed (plus the 64 MiB guard) against 1.75 GiB available.**

Consequences:

* The B50 shortfall is ~1.28 GiB, not ~2.3 GiB.
* Delta charging is worth roughly **+7 layers (6 → 13)**, not +3.
* **The headline conclusion is unchanged:** the B50 cannot reach 24 of 24 by changing charging
  policy. Full `down`-I8 is a large-VRAM feature. The B70 gets it because it has ~20 GB spare
  after weights; the B50 has ~1.75 GiB at the moment the pass runs.

---

## 4. Where the B50's memory actually goes

The arena reserves three fixed zones out of the VRAM budget *before* weights are packed
(`:17296`–`:17321`; `remaining = min(budget, zone_capacity(WEIGHT))`):

| zone | size | how it is sized |
|---|---|---|
| SCRATCH | 512 MiB | constant, `GGML_SYCL_COMPUTE_ARENA_MB` overrides (`:1752`) |
| RUNTIME | 512 MiB | constant, `GGML_SYCL_RUNTIME_ARENA_MB` overrides; raised if PP scratch planning needs more (`:1766`–`:1780`) |
| ONEDNN | **1173.7 MiB** | `max(256 MiB, planned oneDNN scratchpad)` (`:1758`–`:1764`) |

The oneDNN figure is not measured demand. `plan.onednn_scratchpad_bytes = plan.max_tensor_bytes * 2`
(`:17054`), and `max_tensor_bytes` is simply the largest tensor in the model inventory
(`:16880`). For this GGUF that is ~586.8 MiB — consistent with the Q8_0 vocab projection
(201,088 × 2,880 at 34 B/32 elements = 615,329,280 B = 586.8 MiB); 2 × that is 1173.7 MiB,
matching the observed zone to 0.1 MiB. (I inferred the tensor identity from the arithmetic;
I did not open the GGUF.)

Two things are worth flagging:

1. The comment at `:17053` says "the ONEDNN zone is pre-sized at 256 MB; this estimate
   validates the zone is adequate" — but `:1760` actually *raises the zone to the estimate*.
   Comment and behaviour disagree.
2. The sizing is driven by a dense vocab tensor, while the workload this plan cares about is
   MoE. Whether oneDNN ever reorders a 586 MiB weight here is unmeasured.

`1173.7 − 256 = 917.7 MiB` is therefore the largest single B50 lever identified — bigger than
the entire charging question, and nobody has looked at it. It is also the *least verified*: if
oneDNN genuinely needs the space, shrinking the zone makes oneDNN PP paths fall back (there is
already a warning for that case at `:17093`).

---

## 5. gate/up: a pass that can never succeed

`llama.cpp-o752` observed `skip_executor=1536` — every gate/up entry — with `remaining=221.7 MB`,
so VRAM is definitively not involved. The cause is
`planner_moe_primary_executor_supports_layout_on_device()` at `:15267`:

```cpp
if (layout == GGML_LAYOUT_MXFP4_I8) {
    return entry.expert_role == expert_tensor_role::DOWN && planner_mxfp4_i8_supported(entry, device_id);
}
```

MXFP4_I8 is DOWN-only, so `maybe_upgrade_moe_gate_up_layouts_to_i8()` (`:16639`) declines its
entire population on every plan.

**Is the restriction deliberate or vestigial? Deliberate — no gate/up kernel consumes I8.**
The MoE kernel roster in `mmvq.cpp` splits cleanly by role:

* I8 consumers, all `role=down`: `mxfp4.down.direct_final_i8` (`mmvq.cpp:7540`),
  `mxfp4.down.same_expert_grouped` (`:8254`), `mxfp4.down.q8_dpas_tile2/4` (`:19508`).
* gate/up kernels consume XMX_TILED or SOA: `mxfp4.gateup.xmx_tiled_dpas_m2/m4` (`:9838`, `:10896`),
  `mxfp4.gateup.xmx_tiled_bundle4_m2` (`:9633`), `mxfp4.soa.pair_glu_batched` (`:15538`).

The one layout-generic I8 dispatch site, `mmvq_moe_batched_dispatch()` (`mmvq.cpp:16024`,
I8 handling at `:16223`–`:16250`), **has no callers anywhere in the tree** — only its definition
and its `mmvq.hpp:68` declaration. The live MoE entry points are the role-specific variants
(`…_pair_glu_mxfp4_soa`, `…_down_from_cached_q8_mxfp4`, `…_down_sum_from_cached_q8_mxfp4`).
So it is not a latent gate/up I8 path.

**Would an I8 gate/up path be worth building? Almost certainly not.** From `llama.cpp-clhz`:
gate/up already moves 33.62 MiB/layer/token at **150.0 GB/s** on SOA (the kernel packs to XMX
tiles at dispatch). I8 would multiply its bytes by 33/17 = **1.94×** to 65.2 MiB, so it would
have to reach ~291 GB/s just to break even — above what any kernel on this card has
demonstrated. gate/up is already the efficient half.

Note the irony without over-reading it: the executor **is** the real blocker — for the gate/up
pass, which nobody was examining — while the original hypothesis aimed that same explanation at
`down`, where it is false. These are two different passes with two different causes. The
coincidence is not partial vindication of the hypothesis.

---

## 6. The premise nobody has tested: is `down`-I8 a win at all?

This plan has spent its effort on *why* the upgrade declines and none on *whether the upgrade
helps*. The arithmetic is not obviously favourable:

* `down` on SOA moves 16.81 MiB/layer/token and achieves **66.8 GB/s** (`mxfp4.soa.batched`).
* `down` on I8 moves 32.63 MiB/layer/token — 1.94× the bytes.
* Decode at batch = 1 is bandwidth-bound. **I8 `down` must sustain ≈130 GB/s just to break even
  against the SOA path**, and more than that to be a win.

The case for I8 is that `down`'s 66.8 GB/s is far below the 150 GB/s the gate/up kernel proves
is reachable on the same card, i.e. `down` is ALU/latency-bound on nibble unpacking rather than
bandwidth-saturated, and I8 buys direct DPAS consumption. That is plausible — and unproven.

The B50 is currently in a *mixed* state (6 layers I8, 18 layers SOA), which means a single
`GGML_SYCL_KERNEL_PROFILE` capture on the B50 already contains both kernels and can settle this
with the CPU-only analyser. **Do that before spending any VRAM on this.** If I8 `down` lands
below ~130 GB/s, the B50's "failure" to upgrade is protecting it, and the correct fix is to stop
requesting the upgrade on small cards rather than to make room for it.

---

## 7. Ranked recommendations

Ranked by expected value per unit of risk. Confidence is stated explicitly; where a value is
unquantified it says so.

### R1. Measure whether `down`-I8 pays for its 1.94× bytes — before anything else
**Cost:** one B50 `GGML_SYCL_KERNEL_PROFILE` capture + `scripts/parse-sycl-kernel-profile.py
--geometry gpt-oss-20b`. No code.
**Value:** decides whether R2–R4 are worth doing at all. Break-even is ≈130 GB/s.
**Confidence:** high that the measurement is decisive; genuinely open which way it goes.
This is first because every other recommendation assumes a win that has never been demonstrated.

### R2. Test the two existing delta-charging opt-ins on the B50 — no code change
**Cost:** two `llama-bench` runs plus the GPT-OSS count gate, per env var.
**Value:** `GGML_SYCL_MOE_PP_DOWN_SPECIALIZED_LAYOUTS=1` and (single-device)
`GGML_SYCL_MOE_PROMPT_DOWN_TRANSIENT=1` both already remove the retained SOA copy and hence
charge the delta. Predicted: `upgraded_tensors` rises from 6 to **12–14** (13 modelled;
the ± comes from the pass-ordering change at `:17488`).
**Risk:** these gates move PP off the SOA `down` path onto I8/DPAS or a transient
materialisation. PP throughput and chat correctness must both be checked — they are opt-in
precisely because that path is unproven. `GGML_SYCL_MOE_PP_DOWN_SPECIALIZED_LAYOUTS` is not in
the "keep opt-in" guardrail list in `CLAUDE.md`, but treat it as if it were.
**Confidence:** high on the layer count (the model reproduces both existing data points
exactly); unknown on throughput and correctness.
This subsumes the lead's option 1 ("fix the double-charge") and option 5 ("drop the SOA
alternate") — they are the same change, and it is already implemented behind these flags.

### R3. Establish what the oneDNN zone actually uses — the largest single lever
**Cost:** one B50 run with `GGML_SYCL_ARENA_PP_PROFILE=1`; read the `[ARENA-PP]` line's
`onednn_mb=<used_begin>-><used_end>/<capacity>` field (`unified-cache.cpp:1322`–`:1340`).
Prompt-phase only.
**Value if the zone is over-reserved:** +917.7 MiB to the weight budget → 6 → **10** layers on
its own, or 13 → **20** combined with R2. Larger than R2. Also frees budget for everything else
on a 16 GB card.
**Risk:** if oneDNN really needs it, shrinking the zone pushes oneDNN PP paths to fall back
(`:17093`) and costs PP throughput. Unquantified until measured.
**Follow-up if over-reserved:** size the zone from measured demand rather than
`2 × max_tensor_bytes`, or add a `GGML_SYCL_ONEDNN_ARENA_MB` knob to match the SCRATCH/RUNTIME
zones (`:1753`, `:1767`), which today have overrides while ONEDNN does not.
**Confidence:** high that the measurement is cheap and conclusive; genuinely unknown which way.

### R4. Accept partial upgrade on 16 GB cards, and keep saying so plainly
**Cost:** none — T7's summary line already reports `granted=mixed` and the partial-decline
message already names headroom as the cause.
**Value:** prevents the next investigator from re-deriving this. Consider adding the modelled
"would need N MiB for all M candidates" to the decline line so the shortfall is legible without
reconstructing R0 by hand.
**Confidence:** high. This is the honest state for the B50 regardless of R1–R3.

### R5. Delete or narrow the gate/up I8 upgrade pass
**Cost:** small, localised (`:16639`–`:16860`).
**Value:** no throughput. It removes a pass that declines 1536 entries on every plan and emits a
`skip_executor`-shaped rejection that already cost this plan time by pointing at the executor
for the wrong pass. Purely a maintenance and misdiagnosis fix.
**Recommendation:** do not delete outright — the pass is correct machinery waiting on a kernel
that does not exist. Prefer an early return with a one-line "no gate/up I8 executor exists"
note, so the log stops implying a per-entry rejection. Revisit only if §5's 291 GB/s break-even
is ever plausible.
**Confidence:** high that no gate/up I8 kernel exists today (kernel roster + planner gate +
the uncalled generic dispatcher).

### Not recommended
* **Chasing 24 of 24 on the B50.** §3.1: it does not fit, at any charging policy, by ~1.3 GiB.
* **Raising the B50's VRAM budget percentage.** `min(total × pct, free_at_init)` is correct by
  design (`CLAUDE.md`); the card genuinely has ~16.25 GB free and holds ~11.3 GB of weights.
* **Building an I8 gate/up kernel** (§5).

### Incidental, out of scope
`mmvq_moe_batched_dispatch()` (`mmvq.cpp:16024`) has no in-tree callers. It is a large,
layout-generic MoE dispatcher carrying live-looking I8 and occupancy logic. Worth a separate
dead-code decision; not part of this diagnosis.

---

## 8. What would change these conclusions

| claim | what would overturn it |
|---|---|
| Headroom is the sole blocker for `down` | Any B50 run where a `down` skip counter other than `skip_not_down` is non-zero, or where `upgraded_tensors` changes without `remaining` changing. |
| Charge = 261.3 MiB/tensor, delta = 126.6 MiB | A `[MOE-LAYOUT] down-i8` line whose `upgraded_tensors × 261.3 + remaining` does not reconstruct a stable R0 across runs at the same budget. A dimension or block-size change in `planner_layout_bytes_for_dims` invalidates the table in §2.2. |
| Delta charging yields ~13 of 24 on the B50 | R2's run reporting materially outside 12–14. That would mean either R0 shifts under the reordered pass or another consumer takes the freed budget first. **This is the single cheapest falsification test in the document.** |
| The B50 cannot reach 24 of 24 | A layout for `down` costing ≤ 71 MiB/tensor more than SOA, or ~1.3 GiB freed elsewhere (R3 alone is not enough), or a smaller expert geometry. Not achievable by charging policy. |
| The oneDNN zone is over-reserved | `[ARENA-PP]` showing `onednn_mb` used approaching its 1173.7 MiB capacity during PP. That would close R3 immediately. |
| No gate/up kernel consumes I8 | Any dispatch site selecting a `role=gateup` kernel with `layout == GGML_LAYOUT_MXFP4_I8` and a live caller. The one candidate found is uncalled. |
| `down`-I8 is worth having at all | R1. A measured I8 `down` kernel below ~130 GB/s inverts the entire objective of this track. |

### Evidence-handling rules that apply to any follow-up run
Carried forward from `llama.cpp-r5ib`, which nearly misread its own output twice:

1. The logged `device=N` is the in-process SYCL index after `ONEAPI_DEVICE_SELECTOR` filtering.
   B50-only and B70-only runs **both** print `device=0`. Key off the selector you passed.
2. Record the reported free VRAM with every result. A background ComfyUI process held 18.3 GiB
   on the B70 and 164.3 MiB on the B50 and silently invalidated a day of measurements.
3. `-v` is mandatory on `llama-bench` or ggml's log callback is muted and nothing prints.
4. Grep the sub-tag (`down-i8`, `gateup-i8`, `summary`), never the bare `[MOE-LAYOUT]` — it is
   shared with a pre-existing per-tensor trace.
5. Silence from the decline line is a result, not a failure: it fires only when something
   declined. Check the per-tensor tally instead
   (`… | grep ffn_down_exps | grep -oE "layout=[a-z0-9_]+" | sort | uniq -c`).

---

## 9. Cross-references

* `llama.cpp-r5ib` — `[MOE-LAYOUT] down-i8` counters; the B50 dose–response.
* `llama.cpp-o752` — gate/up `skip_executor=1536`; the per-role summary line.
* `llama.cpp-fuo6` — clean-card B70 profiling (24 of 24); the ComfyUI confound.
* `llama.cpp-clhz` — `scripts/parse-sycl-kernel-profile.py`, achieved GB/s
  (gate/up 33.62 MiB → 150.0 GB/s, down 16.81 MiB → 66.8 GB/s).
* Code: `ggml/src/ggml-sycl/unified-cache.cpp` `:15047`, `:15267`, `:16375`, `:16639`, `:17054`,
  `:1752`–`:1780`; `ggml/src/ggml-sycl/mmvq.cpp` `:16024`.
* `docs/backend/sycl-memory-design.md`, `docs/design/sycl-canonical-memory-architecture.md`.
