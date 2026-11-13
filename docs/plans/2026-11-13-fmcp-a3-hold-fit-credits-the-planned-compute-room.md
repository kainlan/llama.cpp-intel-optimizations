# fmcp addendum A3: the hold-spill fit credits the compute room the load reserved

> **Execution:** `team-driven-development`. Addendum to
> `docs/plans/2026-11-12-fmcp-qwen38-load-fit-attribution.md`. Written after lead spike S2 (llama.cpp-5pkp,
> comments c-igoc and c-0eak). **Not spawned until the owner approves this file.**

**Goal:** auto `-ub` keeps 512 on the B70 for the exps-IQ4_NL pack at n_ctx 2048, because the fit stops charging a
compute buffer the RUNTIME zone was sized to hold.

**Architecture:** the plan's row for A3 said "zone sizing and admission call one requirement function". S2 shows the
two sums that disagree are not the constructor's (Task 5's line never printed). They are the RUNTIME zone requirement,
which includes the load's measured compute term, and the hold-spill fit, which credits that term only for an explicit
`-ub` the plan PLACEd. Under auto `-ub` the fit charges the rung's compute buffer as a possible spill although the zone
holds room for it. A3 makes the fit read the same fact the zone was sized from, and makes its refusal print its terms.

**Tech stack:** C++ (ggml-sycl backend), host test `test-zone-sizing`, host test `test-sycl-layout-choice`, source
pin `tests/test-sycl-explicit-ubatch-placement-source.py`.

---

## Evidence (S2, B70, integ/th32 ec4735e9e, tuning cache off)

`llama-bench -p 2048 -n 0 -r 1` on exps-IQ4_NL settled at `-ub 256`, 10.22 t/s. The one refusal of 512:

```
sched_reserve:      SYCL0 compute buffer size =   490.00 MiB
[SYCL-PLAN] runtime context update rejected: at n_ubatch=512 the compute buffers the planned dense scratch keeps
out of the RUNTIME zone (worst case 1599.6 MB, ...; 0 request(s) totalling 0.0 MB were held out and now live
outside the arena) leave device 0 250.6 MB free (of 1850.1 MB free before them), under the 256.0 MB driver headroom
[SYCL-PLAN] auto n_ubatch candidate 512: hold spill left no headroom (the largest -ub that fits is about 256)
```

Read with the load's own lines from the same log:

```
llama_load_measure: [LOAD-PLAN] compute trace side=measure stage=probe n_ctx=262144 n_ubatch=512 ...
[LOAD-PLAN] compute term reserved in RUNTIME on device 0: 2057.7 MiB planned (..., 2 chunk(s), n_ctx 262144)
[UNIFIED-CACHE] Runtime zone raised to 7664.1 MB for PP scratch planning (..., compute=2057.7 MB, state=112.6 MB)
```

Facts:

1. The zone was sized with 2057.7 MiB for compute buffers at micro-batch 512.
2. The rung's compute buffer was 490 MiB and none of it was held out of the zone (0 requests, 0.0 MB).
3. The fit still charged a 1599.6 MB worst case and refused by 5.4 MB.
4. The credit that would have removed the 490 MiB exists (`zone_hold_fit_rung_has_compute_room`,
   `ggml/src/ggml-sycl/zone-sizing.cpp:568`) but its inputs are filled only from
   `placement_explicit_ubatch_compute_room` (`ggml/src/ggml-sycl/unified-cache.cpp:36142`), which returns 0 unless an
   explicit `-ub` was PLACEd (`ggml/src/ggml-sycl/ggml-sycl.cpp:19786-19797`).

**Inferred, not read from the log:** the split of 1599.6 MB. The refusal prints only the sum. From
`zone_hold_fit_demand` the sum is `plan + request - kv_room + ring_uncovered`. With `request` = 490 MiB (513.8 MB) the
rest is about 1086 MB. With the credit the demand would be about 1086 MB and leave about 764 MB free, well above the
256 MB headroom. Cycle 1 below prints the terms so the lead's rerun replaces this inference with a reading.

**Not in scope:** the hint "the largest -ub that fits is about N" was one rung optimistic in both `llama-cli` runs
(recorded on llama.cpp-0tew). The ring that came up 2 slots short and sent 36 layers to the CPU is A4/A5.

---

## Team Topology

One implementer (Opus, xhigh). Fresh Opus spec reviewer, then fresh Opus quality reviewer. Worktree
`wt-fmcp-a3` on branch `task/fmcp-a3`, base integ/th32.

| File | Change |
|------|--------|
| `ggml/src/ggml-sycl/unified-cache.hpp`, `unified-cache.cpp` | new pure helper `placement_plan_compute_room` |
| `ggml/src/ggml-sycl/zone-sizing.hpp`, `zone-sizing.cpp` | new pure `zone_hold_fit_terms` |
| `ggml/src/ggml-sycl/ggml-sycl.cpp:19705-19900` | fit fills the room from the new helper; both refusals print the terms |
| `ggml/src/ggml-sycl/tests/test-zone-sizing.cpp` | new case |
| `tests/test-sycl-layout-choice.cpp` | new block beside `:845` |
| `tests/test-sycl-explicit-ubatch-placement-source.py:144-157, 322-334` | `FIT_ROOM` and the claim follow the new call |

---

### Task A3: the fit credits the planned compute room at the load's planning micro-batch, and prints its terms

**Depends on:** nothing open. **Tracker:** llama.cpp-5pkp.

**Description.** Two RED to GREEN cycles, two commits each. Cycle 1 adds a pure function that returns the terms of
the demand and prints them in both hold-spill refusals. Cycle 2 adds the credit for the load's planning micro-batch.

**Acceptance criteria**

- [ ] Both `[SYCL-PLAN] runtime context update rejected` hold-spill lines end with
      `terms: plan=<MB> request=<MB> request_credited=<0|1> kv_room=<MB> ring_uncovered=<MB> compute_room=<MB>@<n_ubatch>`.
- [ ] With no explicit `-ub` placed, a single-device plan on the asked device gives the fit the planned compute term
      at `plan.planner_n_ubatch`. An explicit PLACE keeps its present answer. Every other case gives 0.
- [ ] In the realized check (`rung_live`), a rung that recorded any request held outside the arena gets no credit.
- [ ] `test-zone-sizing`, `test-sycl-layout-choice`, the source pin and the gate registration audit pass.
- [ ] Lead device run (below): 512 is kept on exps-IQ4_NL at `-p 2048`.

#### Cycle 1: the terms

**RED** (`ggml/src/ggml-sycl/tests/test-zone-sizing.cpp`, new case after the block ending near `:1745`, same `CHECK`
macro and style):

```cpp
    // The terms the hold-spill refusal prints are the ones the demand is made of.
    {
        const size_t MiB  = 1024 * 1024;
        auto         plan = [](void *, uint32_t) -> size_t {
            return 1100 * 1024 * 1024;
        };
        ggml_sycl::zone_hold_rung_request rungs[] = {
            { 512, 490 * MiB }
        };
        ggml_sycl::zone_hold_fit_inputs in = {};
        in.headroom_target                 = 256 * MiB;
        in.free_before                     = 1764 * MiB;
        in.kv_room                         = 64 * MiB;
        in.rungs                           = rungs;
        in.n_rungs                         = 1;
        in.plan_of                         = plan;
        in.ring_bytes                      = 2 * MiB;
        ggml_sycl::zone_hold_fit_terms t   = ggml_sycl::zone_hold_fit_terms_at(in, 512);
        CHECK(t.plan == 1100 * MiB && t.request == 490 * MiB && !t.request_credited, "plan and the uncredited request");
        CHECK(t.kv_room == 64 * MiB && t.ring_uncovered == 2 * MiB, "the KV room and the uncovered ring");
        CHECK(t.demand == ggml_sycl::zone_hold_fit_demand(in, 512), "the terms' demand is the fit's demand");
        CHECK(t.demand == 1100 * MiB + 490 * MiB - 64 * MiB + 2 * MiB, "1528 MiB");
        in.compute_room_n_ubatch = 512;
        in.compute_room_bytes    = 2057 * MiB;
        t                        = ggml_sycl::zone_hold_fit_terms_at(in, 512);
        CHECK(t.request == 490 * MiB && t.request_credited, "a credited request is still named, and marked");
        CHECK(t.demand == 1100 * MiB - 64 * MiB + 2 * MiB && t.demand == ggml_sycl::zone_hold_fit_demand(in, 512),
              "and adds nothing");
        t = ggml_sycl::zone_hold_fit_terms_at(in, 0);
        CHECK(t.demand == 0 && t.plan == 0 && t.request == 0, "an unknown n_ubatch names nothing");
    }
```

Run (host only, no device): `./scripts/sycl-build.sh test-zone-sizing` on the lead's GO, then
`ctest --test-dir build -R '^test-zone-sizing$' --output-on-failure`. Before the build, the syntax-only compile fails
with "no member named 'zone_hold_fit_terms'": that is the RED.

**GREEN.** `zone-sizing.hpp`, after `zone_hold_fit_demand`'s declaration:

```cpp
// The terms zone_hold_fit_demand is made of at `n_ubatch`, for a refusal to print. `request` is the rung's request
// whether or not the compute room credits it; `request_credited` says the demand left it out. `demand` is the value
// zone_hold_fit_demand returns for the same inputs.
struct zone_hold_fit_terms {
    size_t plan             = 0;
    size_t request          = 0;
    bool   request_credited = false;
    size_t kv_room          = 0;
    size_t ring_uncovered   = 0;
    size_t demand           = 0;
};
zone_hold_fit_terms zone_hold_fit_terms_at(const zone_hold_fit_inputs & in, uint32_t n_ubatch);
```

`zone-sizing.cpp`: move the body of `zone_hold_fit_demand` into `zone_hold_fit_terms_at` so there is one arithmetic,
and make the demand a one-line caller:

```cpp
zone_hold_fit_terms zone_hold_fit_terms_at(const zone_hold_fit_inputs & in, uint32_t n_ubatch) {
    zone_hold_fit_terms t;
    if (n_ubatch == 0) {
        return t;
    }
    // No plan function, no hold: the ring is still a demand.
    t.plan             = in.plan_of ? in.plan_of(in.plan_ctx, n_ubatch) : 0;
    // The request is already the rung's own (or scaled to it), so the bound is asked with it at this n_ubatch. One
    // the zone was planned to serve is not held back, so it spills nothing; the plan stays in the bound as it is.
    t.request          = zone_hold_fit_request(in, n_ubatch);
    t.request_credited = zone_hold_fit_rung_has_compute_room(in, n_ubatch, t.request);
    t.kv_room          = in.kv_room;
    t.ring_uncovered   = in.ring_bytes;
    const size_t spill = zone_hold_spill_raw_demand(
        zone_hold_spill_bound(t.plan, t.request_credited ? 0 : t.request, n_ubatch, n_ubatch), in.kv_room);
    // Saturating, so a wrapped sum never reads as a fit.
    t.demand = in.ring_bytes > SIZE_MAX - spill ? SIZE_MAX : spill + in.ring_bytes;
    return t;
}

size_t zone_hold_fit_demand(const zone_hold_fit_inputs & in, uint32_t n_ubatch) {
    return zone_hold_fit_terms_at(in, n_ubatch).demand;
}
```

`ggml-sycl.cpp`: add `ggml_sycl::zone_hold_fit_terms terms;` and `uint32_t compute_room_n_ubatch = 0;
size_t compute_room_bytes = 0;` to `ggml_sycl_hold_fit_answer` (`:19713`); in `ggml_sycl_hold_spill_fit` replace
`a.demand = ggml_sycl::zone_hold_fit_demand(in, q.n_ubatch);` with

```cpp
    a.terms                 = ggml_sycl::zone_hold_fit_terms_at(in, q.n_ubatch);
    a.demand                = a.terms.demand;
    a.compute_room_n_ubatch = in.compute_room_n_ubatch;
    a.compute_room_bytes    = in.compute_room_bytes;
```

and append to the format string of both refusals (`ggml_sycl_check_hold_spill_headroom` `:19840` and
`ggml_sycl_check_hold_spill_realized` `:19882`), before the final `\n`:

```
"; terms: plan=%.1f MB request=%.1f MB request_credited=%d kv_room=%.1f MB ring_uncovered=%.1f MB "
"compute_room=%.1f MB@%u"
```

with arguments `a.terms.plan / mb, a.terms.request / mb, a.terms.request_credited ? 1 : 0, a.terms.kv_room / mb,
a.terms.ring_uncovered / mb, a.compute_room_bytes / mb, a.compute_room_n_ubatch`.

**Gotcha:** `tests/test-sycl-explicit-ubatch-placement-source.py` pins `FIT_DEMAND`
(`a.demand = ggml_sycl::zone_hold_fit_demand(in, q.n_ubatch);`) and `DEMAND_CREDIT`/`DEMAND_BOUND` in
`zone_hold_fit_demand`'s body. Both move. Change `FIT_DEMAND` to the `zone_hold_fit_terms_at` line and point
`DEMAND_SIG` at `zone_hold_fit_terms_at`, with `DEMAND_CREDIT` and `DEMAND_BOUND` rewritten to the new text. Run the
pin both ways it is registered: `python3 tests/test-sycl-explicit-ubatch-placement-source.py` and under pytest. Grep
every other pin for the touched symbols before reporting:
`cat tests/*.py | grep -l -E 'zone_hold_fit_demand|ggml_sycl_hold_spill_fit|ggml_sycl_hold_fit_answer'` (run per file).

**Commits:** `test(sycl): the hold-spill fit's terms are the demand's (RED)`, then
`sycl: both hold-spill refusals print the terms of their demand`.

#### Cycle 2: the credit at the planning micro-batch

**RED** (`tests/test-sycl-layout-choice.cpp`, new block after the one ending `:863`; `placed`, `unnamed`
and `compute_term` are the plans and value that block already builds; `none` there is a REFUSE plan, not this case):

```cpp
    // A plan that placed no explicit micro-batch holds the load's compute term at its planning micro-batch.
    {
        ggml_sycl::placement_plan shaped = unnamed;  // action NONE: no micro-batch was named
        shaped.multi_device              = false;
        shaped.device_id                 = 0;
        shaped.planner_n_ubatch          = 512;
        uint32_t     n      = 1;
        const size_t at_512 = ggml_sycl::placement_plan_compute_room(shaped, 0, compute_term, false, &n);
        bool         good   = at_512 == compute_term && n == 512;
        // a rung that recorded a held-out request is charged, whatever the plan says
        good = good && ggml_sycl::placement_plan_compute_room(shaped, 0, compute_term, true, &n) == 0 && n == 0;
        // another device, a multi-device plan and an unknown planning micro-batch hold nothing
        good = good && ggml_sycl::placement_plan_compute_room(shaped, 1, compute_term, false, &n) == 0 && n == 0;
        shaped.multi_device = true;
        good = good && ggml_sycl::placement_plan_compute_room(shaped, 0, compute_term, false, &n) == 0 && n == 0;
        shaped.multi_device     = false;
        shaped.planner_n_ubatch = 0;
        good = good && ggml_sycl::placement_plan_compute_room(shaped, 0, compute_term, false, &n) == 0 && n == 0;
        // an explicit PLACE keeps its own micro-batch, not the planning one
        ggml_sycl::placement_plan explicit_plan = placed;
        explicit_plan.multi_device              = false;
        explicit_plan.device_id                 = 0;
        explicit_plan.planner_n_ubatch          = 512;
        good = good &&
               ggml_sycl::placement_plan_compute_room(explicit_plan, 0, compute_term, false, &n) == compute_term &&
               n == 2048;
        if (!good) {
            printf("FAIL: the compute room of a plan with no explicit micro-batch is the planned term at the "
                   "planning micro-batch on its own device, and nothing once a request was held out\n");
            ok = false;
        }
    }
```

**GREEN.** `unified-cache.hpp`, after `placement_explicit_ubatch_compute_room`'s declaration:

```cpp
// The compute room the RUNTIME zone of `plan` holds on `device`, for the hold-spill fit to credit: `compute_term` at
// the micro-batch the zone was sized for. A placed explicit micro-batch names its own. Otherwise the zone was sized
// at the load's planning micro-batch (planner_n_ubatch, the shape the load measures the compute term at), so the
// room is credited there. 0 with *n_ubatch 0 for a multi-device plan, another device, an unknown planning micro-batch,
// and when `held_out` says the rung already recorded a request outside the arena: a measured spill overrules the plan.
size_t placement_plan_compute_room(const placement_plan & plan,
                                   int                    device,
                                   size_t                 compute_term,
                                   bool                   held_out,
                                   uint32_t *             n_ubatch);
```

`unified-cache.cpp`, after `placement_explicit_ubatch_compute_room`:

```cpp
size_t placement_plan_compute_room(const placement_plan & plan,
                                   int                    device,
                                   size_t                 compute_term,
                                   bool                   held_out,
                                   uint32_t *             n_ubatch) {
    uint32_t n    = 0;
    size_t   room = 0;
    if (!held_out && !plan.multi_device && plan.device_id == device) {
        room = placement_explicit_ubatch_compute_room(plan.explicit_ubatch, compute_term, &n);
        if (n == 0 && plan.explicit_ubatch.action == pp_moe_onednn_explicit_ubatch_action::NONE &&
            plan.planner_n_ubatch != 0) {
            n    = plan.planner_n_ubatch;
            room = compute_term;
        }
    }
    if (n_ubatch != nullptr) {
        *n_ubatch = n;
    }
    return room;
}
```

`ggml-sycl.cpp:19786-19797` becomes:

```cpp
    {
        const auto room_plan     = ggml_sycl_global_plan_owner();
        uint32_t   room_n_ubatch = 0;
        // The realized check has the rung's own record: a request the hold kept outside the arena is charged.
        bool       held_out      = false;
        if (q.rung_live) {
            ggml_sycl::planned_hold_spill_totals totals;
            ggml_sycl::unified_cache_peek_recent_planned_hold_spills(q.device, q.owner, &totals);
            held_out = totals.raw_count != 0;
        }
        const size_t room_bytes = ggml_sycl::placement_plan_compute_room(
            *room_plan, q.device, ggml_sycl::unified_cache_get_planned_compute_term_bytes(q.device), held_out,
            &room_n_ubatch);
        in.compute_room_n_ubatch = room_n_ubatch;
        in.compute_room_bytes    = room_bytes;
    }
```

**Gotchas**

- `unified_cache_get_recent_planned_hold_spills` (`unified-cache.cpp`, body ending `:3571`) RESETS the counters it
  reads, and `ggml_sycl_check_hold_spill_realized` reads them after the fit for its message. The fit must not consume
  them. Add `unified_cache_peek_recent_planned_hold_spills` beside it: same lock, same copy into `*out`, no reset.
  Factor the copy into one static helper both call. Its declaration goes beside the other at `unified-cache.hpp:~2748`.
- `pp_moe_onednn_explicit_ubatch_action::NONE` is the enumerator for "no micro-batch above the planning shape was
  named". `tests/test-sycl-layout-choice.cpp:~833` checks it on the `unnamed` plan. GROW_ZONE and REFUSE must keep giving 0.
- The planning micro-batch is 512 on both sides today: `src/llama-model.cpp:463` (`inventory.n_ubatch = 512`) and
  `src/llama-load-measure.h:212` (`llama_auto_ubatch_ladder[0]`). The credit is only true while they agree. Add one
  case to `tests/test-sycl-explicit-ubatch-placement-source.py` that reads both files and fails when the literal at
  `inventory.n_ubatch =` is not the first element of `llama_auto_ubatch_ladder` in `src/llama-auto-ubatch.h:59`.
- The pin's `FIT_ROOM` and `claim_fit_credits_the_placed_compute_room` (`:144`, `:322`) follow the new call. Keep
  `ROOM_HELPER_RULE` as it is: the explicit helper does not change.
- The header comment on `zone_hold_fit_rung_has_compute_room` (`zone-sizing.hpp`) names the over-credit limit for a
  rung whose buffers span several chunks. It applies to this credit too. Extend that comment by one sentence; do not
  change the rule.
- `ggml-sycl.cpp` is a 60k-line file the code index cannot see into. Find every caller with
  `cat ggml/src/ggml-sycl/ggml-sycl.cpp | grep -n '<symbol>'`.
- Syntax-only compile every changed `.cpp` for every `compile_commands.json` entry before reporting. Three test
  targets compile `unified-cache.cpp` on their own.
- No device binary, no model load. Builds only on the lead's GO.

**Commits:** `test(sycl): a plan with no explicit micro-batch holds the compute term at its planning micro-batch
(RED)`, then `sycl: the hold-spill fit credits the compute room the load reserved at its planning micro-batch`.

---

## End-to-End Validation (on the user's machine): lead only

Environment: this host, B70 (`ONEAPI_DEVICE_SELECTOR=level_zero:0`), under `GPU.lock`, `GGML_SYCL_TUNING_CACHE=0`,
Shmem guard at 100 GB, one run at a time. Binaries from the merged integ tree.

1. After cycle 1 only (terms, no credit): `llama-bench -m <exps-IQ4_NL pack> -p 2048 -n 0 -r 1 -v`. Expected: the
   512 refusal now carries `terms:`. Record the five values on llama.cpp-5pkp. If `request_credited=0` and
   `plan - kv_room` alone leaves less than 256 MB, the credit cannot admit 512: stop and bring that to the owner.
2. After cycle 2: the same command. Expected: no `auto n_ubatch lowered from 512` line, the result row shows
   `n_ubatch` 512, rc 0, and `[SCRATCH-STATS] ... zone=runtime` for the compute buffer with no `zone=raw`.
3. Control, same binary: `llama-cli` on the same pack at the default context with the count prompt. Expected:
   `grep -cx '1, 2, 3, 4, 5'` is 1, as before the change.
4. B50 control (`level_zero:1`), GPT-OSS gate from CLAUDE.md, default context: the digit line count is 1 and the
   settled `-ub` is the one integ gives before the change. This is the card whose 1024 rung the realized check was
   written for; a credit at 512 must not change it.

Observed success is the result row at `n_ubatch` 512 in step 2 with steps 3 and 4 unchanged.
