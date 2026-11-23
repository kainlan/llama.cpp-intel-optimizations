# Task 4 decision record and Task 4b: the zone-sizing abort becomes a refusal (llama.cpp-ouur)

Part of `docs/plans/2026-11-21-inkling2-mini-sycl-enable-profile-optimize.md`. Every file:line below is at master
`1dcdf49b1`; re-find each site by its text before editing.

### Task 4 decision record (spike, 2026-11-21, read-only on master `1dcdf49b1`)

**Channel chosen: (c), the inventory-plan entry the loader already calls.** `llama_model_sycl_apply_inventory`
(`src/llama-model.cpp:631-642`) calls `ggml_backend_sycl_stage_inventory_plan` (resolved by name at `:106-107`,
declared `ggml/include/ggml-sycl.h:1376-1379`). That entry already returns `ggml_sycl_lifecycle_result`, and the loader
already throws `std::runtime_error` on any non-OK answer (`src/llama-model.cpp:639-641`). The entry already has a
rollback guard (`ggml/src/ggml-sycl/ggml-sycl.cpp:18295-18304`) and a C-boundary `catch (...)` (`:18374`). It reaches
both abort sites: the early arm goes through `ggml_backend_sycl_compute_placement_plan_early` (`:18320`, `:17935-17951`)
and the late arm through `ggml_sycl_set_tensor_inventory_impl` (`:18322`, `:17978`, `:18218`). Both end in
`compute_and_store_plan_for_inventory` (`:17611`). The enum already has the right answer:
`GGML_SYCL_LIFECYCLE_PLAN_REJECTED` (`ggml-sycl.h:1360-1366`, "a deterministic refusal ... Callers must NOT retry").
The two public `void` entries are not on the loader's path: llama calls neither of them directly. The cause's text is
printed once, by the backend, on its own `[SYCL-PLAN]` ERROR line in the stage entry's catch. The loader's exception
names the rejection as deterministic and points at that line. The text comes from the code that sized the zones.

**Lead ruling on the text carrier (2026-11-21).** No new entry carries the text to the loader: no new proc-table
entry, no new `ggml-sycl.h` declaration, no thread-local out-pointer and no new hook in `src/llama-model.cpp`. The
reasons:
- A thread-local that exists so that source-contract body pins survive is a test shaping the design.
- A new entry crossing the dlopen boundary is a cost this refusal does not need.
- The backend's own ERROR line, which every tool prints, is the honest place for the text.

**Follow-up (tracker item, not this task).** A capacity refusal during a rebuild comes after `arena_destroy()`
(`unified-cache.cpp:6923`). Moving the fit check before the destroy needs the arena geometry (`alloc_size`,
`per_chunk_cap`, `arena_reserve:27340-27386`) as a pure function.

**Rejected:**
- (a) The reservation proc (`ggml_backend_sycl_load_reserve_compute_term`, `ggml-sycl.cpp:46821-46876`) never sizes
  zones. It stores a planner term, and false means "declined, unplanned" (`:46814-46816`). A distinct refusal there
  would need a new proc, which is allowed: new optional procs by name are the established pattern. But it would be a
  second, earlier answer computed from different inputs, and it would not remove the aborts, which the late plan
  still reaches (live allocations, the ring re-publish).
- (b) A dry run at reservation time cannot see final inputs. The late plan publishes RUNTIME terms again after the
  reservation: `ggml_sycl_publish_load_ubatch_scratch` at `ggml-sycl.cpp:17791-17792`, `:17801` and `:17812`, and the
  GROW path at `:17795-17821`. The live-allocation state can also change between the two points. So it would be a second
  computation of "what the zones can give", which breaks one fact, one source.

**What makes `ensure_planned_arena_zones` (`ggml/src/ggml-sycl/unified-cache.cpp:6628-6936`) return false:**
1. The zones must grow, and live allocations block the rebuild (`:6850-6907`). Live allocations are zone bytes in use,
   chunk leases, or live scratch such as an earlier context's PP ring. This is cause B: a second model beside a live
   one (tickets ouur, pgay).
2. `arena_reserve` (`:27278`) refuses (`:6926-6929`) for one of these reasons:
   - The tail zones plus the 16 MiB shared floor exceed the arena (`:27416-27421`, `:27546-27560`).
   - The arena spans chunks and the tail (one allocation, `:27522-27533`) exceeds the per-allocation cap
     (`:27535-27541`).
   - An allocation fails.

   This is cause A, capacity. The sizing never looks at weights: the planner packs them into whatever shared zone is
   left (`ggml-sycl.cpp:17756-17778`).

**Plan premise corrected (UNVERIFIED on hardware).** The Task 4 description says Inkling at `-c 131072` on the B70
(about 18 GiB of compute term) reaches the abort. The code does not support that. The B70's per-allocation cap is
29472 MiB (`docs/design/sycl-kv-region-segregation.md:1640`), and an 18.5 GiB RUNTIME zone plus the scratch and oneDNN
zones fits under it and inside the roughly 30 GB arena. The zones are reserved, and the planner places the weights
that fit the remaining ~11 GB and tiers the rest. The cases left between Task 3 (term larger than the device total)
and a successful sizing are:
- on the B70, a RUNTIME zone of more than about 28.7 GiB (more than about 205k tokens of Inkling context, a linear
  estimate, UNVERIFIED), and on a single-chunk card, a zone larger than the arena budget;
- live allocations blocking a rebuild.

Task 4b refuses all of them by name. The lead runs Inkling at `-c 131072` once in Task 9 to see what the planner does.

**Size:** one task, about the size of Task 3. The loader change is one added branch.

---

### Task 4b: A load whose arena zones cannot be sized is refused by name, not aborted (llama.cpp-ouur)

**Track:** A (lead orders the `ggml-sycl.cpp` part after Task 7, per the File Ownership Map)
**Depends on:** Task 4 (this decision); Tasks 2, 3 and 5 (same `src/llama-model.cpp`, Track A, sequential); Task 7
(same `ggml/src/ggml-sycl/ggml-sycl.cpp`)
**File scope:**
- Modify: `ggml/src/ggml-sycl/zone-sizing.hpp` (after `zone_runtime_kept_above_plan`, `:499`)
- Modify: `ggml/src/ggml-sycl/zone-sizing.cpp` (after `zone_runtime_kept_above_plan`, `:501-505`)
- Modify: `ggml/src/ggml-sycl/unified-cache.hpp:4291-4304` (`arena_reserve`, `ensure_planned_arena_zones`), `:8078`
- Modify: `ggml/src/ggml-sycl/unified-cache.cpp:6628-6631` (signature, reset), before `:6835` (needs), `:6907` (live
  cause), `:6926-6929` (reserve call), `:26108-26114` (free wrapper), `:27278-27285` (signature), `:27394` (floor), before
  `:27408` (fit check)
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` before `:17608` (exception type), `:17621-17628` and `:17813-17815` (the
  two aborts), `:18374` (typed catch)
- Modify: `ggml/include/ggml-sycl.h:427-443` (a comment sentence on the two `void` entries; no declaration changes)
- Modify: `src/llama-model.cpp:638-641` (`llama_model_sycl_apply_inventory`: one added branch, the call unchanged)
- Modify: `ggml/src/ggml-sycl/tests/test-zone-sizing.cpp` (Case 34, before `:2145`)
- Create: `tests/test-sycl-zone-sizing-refusal-source.py`; Modify: `tests/CMakeLists.txt` (after `:4692`)
- Modify: `tests/test-sycl-explicit-ubatch-placement-source.py:190` (`ENSURE_SIG`)
- Modify: `docs/backend/sycl-memory-design.md:3382-3384`

**Description:** `compute_and_store_plan_for_inventory` ends the process with
`GGML_ABORT("[SYCL-PLAN] failed to size VRAM arena zones from tensor inventory")` (`ggml-sycl.cpp:17627`, `:17814`)
whenever the arena's zones cannot be sized. That happens when the zones do not fit the device (cause A, capacity: the
arena budget, or the per-allocation cap on a chunked arena) or when live allocations block a rebuild (cause B, a second
model beside a live one). This task makes the sizing say why and makes the plan throw that text instead of aborting.
The stage entry prints the text on its `[SYCL-PLAN]` ERROR line and answers `GGML_SYCL_LIFECYCLE_PLAN_REJECTED`.
`load_tensors` then throws a `std::runtime_error` that names the rejection as deterministic and points at that line,
the same way it throws `probe.refusal` (`src/llama-model.cpp:2962`). Nothing is shrunk, no fallback context is
chosen, no global is read after a call, and no abort is added. The GROW path (`:17796-17821`)
keeps its non-fatal one-argument try-grow.

**Acceptance Criteria:**
- [ ] `ggml-sycl.cpp` has no `GGML_ABORT("[SYCL-PLAN] failed to size VRAM arena zones`. Both sites ask
      `unified_cache_ensure_planned_arena_zones(ctx->device, &why)` and throw `zone_sizing_refusal_text(why)`.
- [ ] `zone_arena_tail_fit` passes test-zone-sizing Case 34 (below):
      - it fits Inkling's 18944 MiB RUNTIME zone on the B70 geometry;
      - it refuses 28800 MiB as `CHUNK_CAP` and 150528 MiB as `BUDGET`, with the room left for each;
      - the floor boundary falls on the byte `arena_reserve` checks;
      - a wrapped sum refuses.
- [ ] `arena_reserve` refuses by `zone_arena_tail_fit` before its first allocation, and the 16 MiB floor is
      `k_zone_arena_min_shared_bytes` (one source).
- [ ] The text names `load refused`, the SYCL device, the RUNTIME zone needed with the planned compute term, and what
      can be given: the room for a capacity refusal, or the live zones for a live one. It also names the cause:
      `the device's capacity` or `live allocations`. Only the capacity text names `with -c`.
- [ ] `ggml_backend_sycl_stage_inventory_plan` catches the zone refusal before its `catch (...)`, logs
      `GGML_LOG_ERROR("[SYCL-PLAN] %s\n", e.what())` in the catch block itself, and returns `PLAN_REJECTED`. Every
      other failure still returns `EFFECT_FAILED`. This catch is the entry's only `PLAN_REJECTED`.
- [ ] `llama_model_sycl_apply_inventory` keeps its one call `hooks.stage_inventory(&inventory, &envelope, early)`
      (`tests/test-sycl-lifecycle-source-contract.py:997` pins it). On `PLAN_REJECTED` it throws a message that says
      three things: the SYCL backend refused this load's placement plan; the refusal is deterministic for these
      inputs (a retry with the same model, context and devices gets the same answer); the cause is the backend's
      `[SYCL-PLAN]` error line printed just before. Every other non-OK answer still throws
      `SYCL inventory planning failed: early=%d result=%d`.
- [ ] No new public declaration, proc-table entry or llama hook.
- [ ] `test-sycl-zone-sizing-refusal-source` passes (4 claims, 4 mutants), and so do the existing contracts listed
      under step 6.
- [ ] (lead, GPU, once) The B50 capacity vehicle in step 7 prints the refusal and no abort.

**Implementation Guide:**

1. **Test (RED), the arithmetic and the text:** `ggml/src/ggml-sycl/tests/test-zone-sizing.cpp`. Insert this before
   `std::printf("PASS: ...")` (`:2145`):

```cpp
    // ---- Case 34: a load whose arena zones cannot be had is refused by name (llama.cpp-ouur) ----
    // arena_reserve() refuses by zone_arena_tail_fit() before it allocates, and the load's refusal names the device, the
    // compute term, what the arena can give and which cause it is. The B70's per-allocation cap is the recorded
    // 29472 MiB (docs/design/sycl-kv-region-segregation.md); the arena size is illustrative.
    {
        using ggml_sycl::zone_arena_tail_fit;
        using ggml_sycl::zone_sizing_cause;
        using ggml_sycl::zone_sizing_refusal;
        using ggml_sycl::zone_sizing_refusal_text;
        const size_t mib     = 1024ull * 1024ull;
        const size_t floor   = ggml_sycl::k_zone_arena_min_shared_bytes;
        const size_t arena   = 30720 * mib;
        const size_t cap     = 29472 * mib;
        const size_t scratch = 512 * mib;
        const size_t onednn  = 256 * mib;
        CHECK(floor == 16 * mib, "the shared floor is not arena_reserve()'s 16 MiB");

        // Inkling at -c 131072: an 18 GiB term in an 18944 MiB RUNTIME zone fits; the planner, not the zones, decides
        // which weights stay on the device
        zone_sizing_refusal why;
        CHECK(zone_arena_tail_fit(arena, cap, scratch, onednn, 18944 * mib, &why), "a tail within the cap was refused");
        CHECK(why.cause == zone_sizing_cause::NONE, "a fit wrote a cause");

        // within the arena but past one allocation: the cap decides, and names the room it leaves
        CHECK(!zone_arena_tail_fit(arena, cap, scratch, onednn, 28800 * mib, &why), "a tail past the cap was accepted");
        CHECK(why.cause == zone_sizing_cause::CHUNK_CAP, "a tail past the cap is not CHUNK_CAP");
        CHECK(why.limit_bytes == cap, "the cap's refusal does not name the cap");
        CHECK(why.runtime_room == cap - scratch - onednn - floor, "the cap's room is not the cap less the rest");
        CHECK(why.need_runtime == 28800 * mib, "the refusal does not carry the RUNTIME zone it needed");

        // Inkling at its 1M default: 150528 MiB is past the arena itself
        CHECK(!zone_arena_tail_fit(arena, cap, scratch, onednn, 150528 * mib, &why), "a tail past the arena was accepted");
        CHECK(why.cause == zone_sizing_cause::BUDGET && why.limit_bytes == arena, "a tail past the arena is not BUDGET");
        CHECK(why.runtime_room == arena - scratch - onednn - floor, "the arena's room is not the arena less the rest");

        // one chunk (no cap, or the arena within it): only the arena binds, at the byte arena_reserve() checks
        const size_t b50 = 15360 * mib;
        CHECK(zone_arena_tail_fit(b50, 0, scratch, onednn, b50 - scratch - onednn - floor, &why),
              "a tail that leaves exactly the floor was refused");
        CHECK(!zone_arena_tail_fit(b50, 0, scratch, onednn, b50 - scratch - onednn - floor + 1, &why) &&
                  why.cause == zone_sizing_cause::BUDGET,
              "a tail one byte into the floor was accepted");
        CHECK(zone_arena_tail_fit(b50, b50, scratch, onednn, 14000 * mib, &why), "an arena equal to its cap was chunked");
        // a sum that wraps is larger than any arena, never smaller, and leaves no room
        CHECK(!zone_arena_tail_fit(arena, 0, SIZE_MAX - mib, onednn, 512 * mib, &why) &&
                  why.cause == zone_sizing_cause::BUDGET && why.runtime_room == 0,
              "a wrapped tail was accepted");
        CHECK(!zone_arena_tail_fit(arena, cap, scratch, onednn, 150528 * mib, nullptr), "a null why changed the answer");

        // the capacity text: device, term, room, cause, remedy
        CHECK(!zone_arena_tail_fit(arena, cap, scratch, onednn, 150528 * mib, &why), "the 1M tail was accepted");
        why.device       = 0;
        why.compute_term = 150016 * mib;
        const std::string capacity = zone_sizing_refusal_text(why);
        CHECK(capacity.find("load refused") != std::string::npos, "no refusal in the capacity text");
        CHECK(capacity.find("SYCL device 0") != std::string::npos, "the capacity text does not name the device");
        CHECK(capacity.find("150016.0 MiB") != std::string::npos, "the capacity text does not name the compute term");
        CHECK(capacity.find("29936.0 MiB") != std::string::npos, "the capacity text does not name the room");
        CHECK(capacity.find("the device's capacity") != std::string::npos, "the capacity text does not name its cause");
        CHECK(capacity.find("with -c") != std::string::npos, "the capacity text does not name the remedy");
        CHECK(capacity.find("live allocations") == std::string::npos, "the capacity text names the other cause");

        // the live text: what the live zones hold and what keeps them; no context remedy
        zone_sizing_refusal live;
        live.cause        = zone_sizing_cause::LIVE_ALLOCATIONS;
        live.device       = 1;
        live.compute_term = 2048 * mib;
        live.need_scratch = scratch;
        live.need_onednn  = onednn;
        live.need_runtime = 2560 * mib;
        live.have_scratch = scratch;
        live.have_onednn  = onednn;
        live.have_runtime = 1024 * mib;
        live.live_bytes   = 9000 * mib;
        live.chunk_leases = 2;
        const std::string held = zone_sizing_refusal_text(live);
        CHECK(held.find("SYCL device 1") != std::string::npos, "the live text does not name the device");
        CHECK(held.find("2048.0 MiB") != std::string::npos, "the live text does not name the compute term");
        CHECK(held.find("1024.0 MiB") != std::string::npos, "the live text does not name the live RUNTIME zone");
        CHECK(held.find("9000.0 MiB") != std::string::npos, "the live text does not name the bytes in use");
        CHECK(held.find("live allocations") != std::string::npos, "the live text does not name its cause");
        CHECK(held.find("the device's capacity") == std::string::npos, "the live text names the other cause");
        CHECK(held.find("with -c") == std::string::npos, "the live text names a context remedy");
    }
```

   Run: `./scripts/sycl-build.sh test-zone-sizing`. Expected: a compile error. `zone_arena_tail_fit` is not a member
   of `ggml_sycl`.

2. **Test (RED), the travel:** create `tests/test-sycl-zone-sizing-refusal-source.py`. The spike checked the text
   below in two ways:
   - On master `1dcdf49b1` all 8 tests fail. It was run against `/Apps/llama.cpp` read-only, with `ROOT` pointed
     there.
   - With step 5's stage catch and loader lines spliced into an in-memory copy of master's text, the stage and loader
     claims pass and their mutants fail. The plan and sizing claims were not probed that way.

```python
"""Source contract for llama.cpp-ouur: a load whose arena zones cannot be sized is refused by name, never aborted.

Host-only: reads sources, builds nothing, loads no model and touches no device.

The defect: compute_and_store_plan_for_inventory() ended the process with GGML_ABORT("[SYCL-PLAN] failed to size VRAM
arena zones from tensor inventory") whenever unified_cache::ensure_planned_arena_zones() returned false: a load whose
zones (its planned compute term is in the RUNTIME zone) do not fit the device, and a load that needs larger zones while
live allocations keep the arena from being rebuilt (a second model beside a live one).

The fix this gate pins: ensure_planned_arena_zones() says why through a zone_sizing_refusal, and arena_reserve()
decides its refusals before it allocates by zone_arena_tail_fit(), the rule the text then names. The plan throws the
refusal's text where it aborted; ggml_backend_sycl_stage_inventory_plan() catches it, prints it on the backend's
[SYCL-PLAN] ERROR line and answers GGML_SYCL_LIFECYCLE_PLAN_REJECTED; the loader throws a refusal from load_tensors that
names PLAN_REJECTED as deterministic and points at that line. The arithmetic and the text are unit-tested in
ggml/src/ggml-sycl/tests/test-zone-sizing.cpp (Case 34); this gate pins how the refusal travels.

Every claim is checked on COMMENT-STRIPPED, whitespace-normalized text and has a mutant that must make it fail.

Pytest-style (module-level test_* functions): register with llama_test_pytest.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_LEXEME_RE = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'",
    re.DOTALL,
)


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def norm(text: str) -> str:
    """Comment-stripped, whitespace-canonical text, so a clang-format re-wrap cannot turn a claim red."""
    t = re.sub(r"\s+", " ", strip_comments(text))
    t = re.sub(r"\( ", "(", t)
    t = re.sub(r" \)", ")", t)
    t = re.sub(r" ,", ",", t)
    t = re.sub(r",(?! )", ", ", t)
    return t


def read(rel: str) -> str:
    return (ROOT / rel).read_text()


SYCL = read("ggml/src/ggml-sycl/ggml-sycl.cpp")
CACHE = read("ggml/src/ggml-sycl/unified-cache.cpp")
MODEL = read("src/llama-model.cpp")


def body(text: str, signature: str) -> str:
    """The brace-balanced body of the first definition whose normalized text starts with `signature` (string
    literals are skipped while balancing). Empty when there is none."""
    at = text.find(signature)
    if at < 0:
        return ""
    open_at = text.find("{", at + len(signature))
    if open_at < 0:
        return ""
    depth = 0
    i = open_at
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "\"'":
            j = i + 1
            while j < n and text[j] != ch:
                j += 2 if text[j] == "\\" else 1
            i = j + 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_at:i + 1]
        i += 1
    return ""


def ordered(text: str, *needles: str) -> bool:
    """Every needle is present, each after the one before it."""
    at = 0
    for needle in needles:
        at = text.find(needle, at)
        if at < 0:
            return False
        at += len(needle)
    return True


PLAN_SIG = "static void compute_and_store_plan_for_inventory("
STAGE_SIG = "ggml_sycl_lifecycle_result ggml_backend_sycl_stage_inventory_plan(const ggml_sycl_tensor_inventory * inventory,"
ENSURE_SIG = "bool unified_cache::ensure_planned_arena_zones("
RESERVE_SIG = "bool unified_cache::arena_reserve("
APPLY_SIG = "static void llama_model_sycl_apply_inventory("

ABORT = 'GGML_ABORT("[SYCL-PLAN] failed to size VRAM arena zones from tensor inventory");'
SIZED = "ggml_sycl::unified_cache_ensure_planned_arena_zones(ctx->device, &why)"
THROW = "throw ggml_sycl_zone_sizing_refused(ggml_sycl::zone_sizing_refusal_text(why));"
CATCH = "} catch (const ggml_sycl_zone_sizing_refused & e) {"
LOG = 'GGML_LOG_ERROR("[SYCL-PLAN] %s\\n", e.what());'
REJECT = "return GGML_SYCL_LIFECYCLE_PLAN_REJECTED;"
CATCH_ALL = "} catch (...) {"
LIVE = "why->cause = zone_sizing_cause::LIVE_ALLOCATIONS;"
RESERVE_CALL = ("arena_reserve(queue_, budget_, max_alloc, safe_max_alloc, scratch_zone, onednn_zone, runtime_zone, "
                "device_total_vram_, why)")
RESERVE_FAILED = "why->cause = zone_sizing_cause::RESERVE_FAILED;"
FIT = "zone_arena_tail_fit("
ALLOC = "sycl_aligned_malloc_device("
CALL = "const auto rc = hooks.stage_inventory(&inventory, &envelope, early);"
REJECTED_IF = "if (rc == GGML_SYCL_LIFECYCLE_PLAN_REJECTED) {"
REJECTED_THROW = "throw std::runtime_error(format(\"SYCL backend refused this load's placement plan"
GENERIC = ('if (rc != GGML_SYCL_LIFECYCLE_OK) { throw std::runtime_error(format("SYCL inventory planning failed: '
           'early=%d result=%d", early, (int) rc)); }')


def claim_the_plan_throws_where_it_aborted(sycl: str) -> bool:
    """Neither zone-sizing site aborts: both ask the sizing why and throw its text."""
    text = norm(sycl)
    plan = body(text, PLAN_SIG)
    return ABORT not in text and bool(plan) and plan.count(SIZED) == 2 and plan.count(THROW) == 2


def claim_stage_refuses_by_name(sycl: str) -> bool:
    """The stage entry catches the zone refusal before the catch-all, prints it at ERROR and answers PLAN_REJECTED,
    its only PLAN_REJECTED."""
    stage = body(norm(sycl), STAGE_SIG)
    return bool(stage) and ordered(stage, CATCH, LOG, REJECT, CATCH_ALL) and stage.count(REJECT) == 1


def claim_the_loader_names_the_rejection(model: str) -> bool:
    """One stage call; PLAN_REJECTED throws its own deterministic-refusal message, every other non-OK answer the old
    numbered one."""
    apply = body(norm(model), APPLY_SIG)
    return (bool(apply) and apply.count("hooks.stage_inventory(") == 1
            and ordered(apply, CALL, REJECTED_IF, REJECTED_THROW, GENERIC))


def claim_the_sizing_names_its_cause(cache: str) -> bool:
    """The sizing names a live-allocation refusal and a failed reservation, and decides the fit before allocating."""
    text = norm(cache)
    ensure = body(text, ENSURE_SIG)
    reserve = body(text, RESERVE_SIG)
    return (bool(ensure) and ordered(ensure, LIVE, "return false;") and ordered(ensure, RESERVE_CALL, RESERVE_FAILED)
            and bool(reserve) and ordered(reserve, FIT, ALLOC))


def test_the_plan_throws_where_it_aborted():
    assert claim_the_plan_throws_where_it_aborted(SYCL)


def test_the_stage_refuses_by_name():
    assert claim_stage_refuses_by_name(SYCL)


def test_the_loader_names_the_rejection():
    assert claim_the_loader_names_the_rejection(MODEL)


def test_the_sizing_names_its_cause():
    assert claim_the_sizing_names_its_cause(CACHE)


# ---- mutants: every claim must fail against the thing it forbids --------------------------------------------------


def _once(raw: str, old: str, new: str, within: str = "", matches: int = 1) -> str:
    """The mutant: the first `old` replaced by `new`, all three compared as norm() text. The anchor must match exactly
    `matches` times, in the body of `within` when it is given, else in the whole file."""
    text, o, n = norm(raw), norm(old), norm(new)
    scope = body(text, within) if within else text
    assert scope.count(o) == matches, f"mutant anchor must match {matches}x: {old!r} x{scope.count(o)}"
    return text.replace(scope, scope.replace(o, n, 1), 1) if within else text.replace(o, n, 1)


def test_mutant_the_abort_back_at_a_site_fails():
    assert not claim_the_plan_throws_where_it_aborted(_once(SYCL, THROW, ABORT, PLAN_SIG, matches=2))


def test_mutant_the_stage_calling_it_an_effect_failure_fails():
    assert not claim_stage_refuses_by_name(
        _once(SYCL, REJECT, "return GGML_SYCL_LIFECYCLE_EFFECT_FAILED;", STAGE_SIG))


def test_mutant_the_loader_numbering_the_rejection_fails():
    assert not claim_the_loader_names_the_rejection(
        _once(MODEL, REJECTED_IF, "if (rc == GGML_SYCL_LIFECYCLE_EFFECT_FAILED) {", APPLY_SIG))


def test_mutant_the_sizing_forgetting_the_live_cause_fails():
    assert not claim_the_sizing_names_its_cause(_once(CACHE, LIVE, "(void) why;", ENSURE_SIG))
```

   Register it in `tests/CMakeLists.txt` after the `test-sycl-load-compute-term-source` block (`:4682-4692`):

```cmake
# llama.cpp-ouur: a load whose arena zones cannot be sized is refused by name, never aborted: the sizing names its
# cause, the plan throws its text, the stage entry prints it at ERROR and answers PLAN_REJECTED, and the loader names
# that rejection. A mutant per claim. Text only: no device, no model.
if (Python3_Interpreter_FOUND)
    llama_test_pytest(${Python3_EXECUTABLE}
        NAME test-sycl-zone-sizing-refusal-source
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-zone-sizing-refusal-source.py
        LABEL "sycl;python;source-assert"
        TIMEOUT 120)
endif()
```

   Run: `python3 -m pytest -q -p no:cacheprovider tests/test-sycl-zone-sizing-refusal-source.py`. Expected:
   `8 failed`.

3. **Implement (GREEN), the rule and the text:** `ggml/src/ggml-sycl/zone-sizing.hpp`. Insert after
   `bool zone_runtime_kept_above_plan(...)` (`:499`):

```cpp
// The smallest shared KV+WEIGHT zone the arena carves beside its tail zones. Below it the arena would only alias the
// tail zones on top of each other (nryi9), so arena_reserve() refuses a reservation that cannot leave it.
constexpr size_t k_zone_arena_min_shared_bytes = 16ull * 1024ull * 1024ull;

// Why a load's arena zones could not be had (llama.cpp-ouur). The load is refused by this name, never aborted.
enum class zone_sizing_cause {
    NONE,              // not refused
    BUDGET,            // the tail zones and the shared floor are larger than the arena
    CHUNK_CAP,         // the arena spans chunks, and the tail zones (one allocation) are larger than one chunk may be
    LIVE_ALLOCATIONS,  // the zones must grow, and live allocations keep the arena from being rebuilt
    RESERVE_FAILED,    // the zones fit, and reserving the arena still failed (an allocation the driver refused)
};

// What the refusal names. ensure_planned_arena_zones() fills the device, the compute term and the zones it needs;
// the step that refused fills the cause and what the device or the live arena can give.
struct zone_sizing_refusal {
    zone_sizing_cause cause        = zone_sizing_cause::NONE;
    int               device       = -1;
    size_t            compute_term = 0;  // the planned compute term, part of need_runtime
    size_t            need_scratch = 0;
    size_t            need_onednn  = 0;
    size_t            need_runtime = 0;
    size_t            have_scratch = 0;  // LIVE_ALLOCATIONS: the zones the live arena holds
    size_t            have_onednn  = 0;
    size_t            have_runtime = 0;
    size_t            live_bytes   = 0;  // LIVE_ALLOCATIONS: bytes in use in the arena's zones
    uint32_t          chunk_leases = 0;
    bool              live_scratch = false;
    size_t            limit_bytes  = 0;  // BUDGET: the arena; CHUNK_CAP: the per-allocation cap; RESERVE_FAILED: budget
    size_t            runtime_room = 0;  // BUDGET, CHUNK_CAP: the largest RUNTIME zone the limit holds beside the rest
};

// Whether an arena of `arena_bytes` holds the tail zones (scratch, oneDNN and RUNTIME: one allocation at the end of
// the last chunk) beside the shared floor. `chunk_cap_bytes` is the per-allocation cap, 0 for none; it binds only when
// the arena is larger than it, so the arena spans chunks. The sum saturates: a wrapped tail is larger than any arena.
// False fills why's cause, needs, limit and room (why may be null); true leaves why untouched. arena_reserve() refuses
// by this rule before it allocates, so the load's refusal names the rule that decided.
bool zone_arena_tail_fit(size_t                arena_bytes,
                         size_t                chunk_cap_bytes,
                         size_t                scratch_bytes,
                         size_t                onednn_bytes,
                         size_t                runtime_bytes,
                         zone_sizing_refusal * why);

// The load's refusal text: the device, the zones it needs with the compute term, what the device or the live arena can
// give, and which of the two causes it is. A capacity refusal names -c as the remedy; a live one does not.
std::string zone_sizing_refusal_text(const zone_sizing_refusal & why);
```

   `ggml/src/ggml-sycl/zone-sizing.cpp`. Insert after `zone_runtime_kept_above_plan` (`:501-505`). The file already
   includes `<cstdio>` and `<string>` (`:13-18`):

```cpp
// a + b, or SIZE_MAX when it wraps
static size_t zone_saturating_add(size_t a, size_t b) {
    return b > SIZE_MAX - a ? SIZE_MAX : a + b;
}

bool zone_arena_tail_fit(size_t                arena_bytes,
                         size_t                chunk_cap_bytes,
                         size_t                scratch_bytes,
                         size_t                onednn_bytes,
                         size_t                runtime_bytes,
                         zone_sizing_refusal * why) {
    const size_t others =
        zone_saturating_add(zone_saturating_add(scratch_bytes, onednn_bytes), k_zone_arena_min_shared_bytes);
    const size_t      needed = zone_saturating_add(others, runtime_bytes);
    zone_sizing_cause cause  = zone_sizing_cause::NONE;
    size_t            limit  = 0;
    if (arena_bytes < needed) {
        cause = zone_sizing_cause::BUDGET;
        limit = arena_bytes;
    } else if (chunk_cap_bytes != 0 && arena_bytes > chunk_cap_bytes && chunk_cap_bytes < needed) {
        cause = zone_sizing_cause::CHUNK_CAP;
        limit = chunk_cap_bytes;
    } else {
        return true;
    }
    if (why != nullptr) {
        why->cause        = cause;
        why->need_scratch = scratch_bytes;
        why->need_onednn  = onednn_bytes;
        why->need_runtime = runtime_bytes;
        why->limit_bytes  = limit;
        why->runtime_room = limit > others ? limit - others : 0;
    }
    return false;
}

std::string zone_sizing_refusal_text(const zone_sizing_refusal & why) {
    const double mib = 1024.0 * 1024.0;
    char         needs[384];
    std::snprintf(needs, sizeof(needs),
                  "load refused: SYCL device %d cannot size its arena zones for this load: they need a RUNTIME zone of "
                  "%.1f MiB (the planned compute term is %.1f MiB of it), oneDNN %.1f MiB and scratch %.1f MiB; ",
                  why.device, why.need_runtime / mib, why.compute_term / mib, why.need_onednn / mib,
                  why.need_scratch / mib);
    char cause[512];
    switch (why.cause) {
        case zone_sizing_cause::BUDGET:
            std::snprintf(cause, sizeof(cause),
                          "the device's capacity: the arena is %.1f MiB, so the RUNTIME zone can be at most %.1f MiB "
                          "beside the other zones; request a smaller context with -c",
                          why.limit_bytes / mib, why.runtime_room / mib);
            break;
        case zone_sizing_cause::CHUNK_CAP:
            std::snprintf(cause, sizeof(cause),
                          "the device's capacity: the RUNTIME, oneDNN and scratch zones are one allocation and the "
                          "device allows at most %.1f MiB in one, so the RUNTIME zone can be at most %.1f MiB; request "
                          "a smaller context with -c",
                          why.limit_bytes / mib, why.runtime_room / mib);
            break;
        case zone_sizing_cause::LIVE_ALLOCATIONS:
            std::snprintf(cause, sizeof(cause),
                          "live allocations: the arena's zones are RUNTIME %.1f MiB, oneDNN %.1f MiB and scratch %.1f "
                          "MiB, and %.1f MiB in use in them, %u chunk lease(s)%s keep them from being rebuilt larger "
                          "(another loaded model, or an earlier context's scratch); load this model before the other, "
                          "or free it",
                          why.have_runtime / mib, why.have_onednn / mib, why.have_scratch / mib, why.live_bytes / mib,
                          (unsigned) why.chunk_leases, why.live_scratch ? " and live scratch" : "");
            break;
        case zone_sizing_cause::RESERVE_FAILED:
            std::snprintf(cause, sizeof(cause),
                          "the zones fit the device, and reserving the arena within its %.1f MiB budget failed (the "
                          "[VRAM-ARENA] lines above say why)",
                          why.limit_bytes / mib);
            break;
        case zone_sizing_cause::NONE:
        default:
            std::snprintf(cause, sizeof(cause), "no cause was recorded");
            break;
    }
    return std::string(needs) + cause;
}
```

   Run: `./scripts/sycl-build.sh test-zone-sizing && ctest --test-dir build -R '^zone-sizing$' --output-on-failure`.
   Expected: `PASS: zone-sizing structural path-scoped maxima`. The ctest name is `zone-sizing`
   (`ggml/src/ggml-sycl/CMakeLists.txt:6740`).

4. **Implement (GREEN), the sizing says why:** `ggml/src/ggml-sycl/unified-cache.hpp`.
   - `arena_reserve` (`:4291-4299`): append the parameter `zone_sizing_refusal * why = nullptr`, and add to its
     comment "False says why in `why` when its zones do not fit (llama.cpp-ouur)."
   - `ensure_planned_arena_zones` (`:4304`): becomes `bool ensure_planned_arena_zones(zone_sizing_refusal * why = nullptr);`.
     Add to its comment "False says why in `why`."
   - The free function (`:8078`): becomes
     `bool unified_cache_ensure_planned_arena_zones(int device_id, zone_sizing_refusal * why = nullptr);`.

   The defaults keep every existing call unchanged. That includes the constructor (`unified-cache.cpp:6378`),
   `:22359`, the GROW try-grow (`ggml-sycl.cpp:17802`, pinned as `GROW_PUBLISH` in
   `tests/test-sycl-explicit-ubatch-placement-source.py:229-230`) and `ggml-sycl.cpp:19139`.

   `ggml/src/ggml-sycl/unified-cache.cpp`:
   - `:6628` becomes `bool unified_cache::ensure_planned_arena_zones(zone_sizing_refusal * why) {`. After the
     `if (!vram_arena_enabled()) { return true; }` block (`:6629-6631`), add:

```cpp
    if (why != nullptr) {
        *why = zone_sizing_refusal{};
    }
```

   - Immediately before `auto bind_compute_arena = [&]() {` (`:6835`):

```cpp
    // What the load's refusal names when these zones cannot be had (llama.cpp-ouur): the device, the planned compute
    // term and the zones this plan needs. Each refusal below adds its cause and what could be given.
    if (why != nullptr) {
        why->device       = dev_id;
        why->compute_term = unified_cache_get_planned_compute_term_bytes(dev_id);
        why->need_scratch = scratch_zone;
        why->need_onednn  = onednn_zone;
        why->need_runtime = runtime_zone;
    }
```

   - In the live branch, immediately before its `return false;` (`:6907`):

```cpp
            if (why != nullptr) {
                why->cause        = zone_sizing_cause::LIVE_ALLOCATIONS;
                why->have_scratch = zone_capacity(vram_zone_id::SCRATCH);
                why->have_onednn  = zone_capacity(vram_zone_id::ONEDNN);
                why->have_runtime = zone_capacity(vram_zone_id::RUNTIME);
                why->live_bytes   = live_zone_bytes;
                why->chunk_leases = chunk_leases;
                why->live_scratch = has_live_scratch;
            }
```

   - The reservation (`:6926-6929`) becomes:

```cpp
    if (!arena_reserve(queue_, budget_, max_alloc, safe_max_alloc, scratch_zone, onednn_zone, runtime_zone,
                       device_total_vram_, why)) {
        if (why != nullptr && why->cause == zone_sizing_cause::NONE) {
            why->cause       = zone_sizing_cause::RESERVE_FAILED;
            why->limit_bytes = budget_;
        }
        return false;
    }
```

   - The free wrapper (`:26108-26114`): add the parameter and pass it on with
     `return cache->ensure_planned_arena_zones(why);`.
   - `arena_reserve` (`:27278-27285`): append `zone_sizing_refusal * why` to the definition's parameters. `:27394`
     becomes `constexpr size_t k_min_shared_bytes = k_zone_arena_min_shared_bytes;  // 16 MB`. Immediately before
     `const bool try_single_chunk = ...` (`:27408`), add:

```cpp
    // The zones' fit is decided before anything is allocated, by the rule the load's refusal names (llama.cpp-ouur):
    // the tail zones beside the shared floor within the arena, and within one allocation when the arena spans chunks.
    // A raw cap of 0 is no cap: the single chunk below is then tried at any size.
    zone_sizing_refusal   fit_local;
    zone_sizing_refusal * fit = why != nullptr ? why : &fit_local;
    if (!zone_arena_tail_fit(alloc_size, max_alloc_size == 0 ? 0 : per_chunk_cap, scratch_bytes, onednn_bytes,
                             runtime_bytes, fit)) {
        GGML_LOG_WARN(
            "[VRAM-ARENA] zones scratch=%.1f MB oneDNN=%.1f MB runtime=%.1f MB do not fit %s of %.1f MB; refusing "
            "reservation\n",
            scratch_bytes / (1024.0 * 1024.0), onednn_bytes / (1024.0 * 1024.0), runtime_bytes / (1024.0 * 1024.0),
            fit->cause == zone_sizing_cause::CHUNK_CAP ? "one allocation" : "the arena",
            fit->limit_bytes / (1024.0 * 1024.0));
        return false;
    }
```

   This rule matches `arena_reserve`'s own checks exactly. When the arena is at most the cap (or the raw cap is 0),
   `try_single_chunk` is true. A single chunk smaller than tail plus floor falls through to the N-chunk path, and the
   N-chunk refusals at `:27535-27561` refuse it there: that is `BUDGET`. When the arena exceeds a nonzero cap, the
   N-chunk path refuses a cap below tail plus floor (`:27536`): that is `CHUNK_CAP`. Leave the later refusals as they
   are, because they still guard the path after a failed single allocation.

5. **Implement (GREEN), the plan throws and the stage refuses:** `ggml/src/ggml-sycl/ggml-sycl.cpp`.
   - Before the `// Phase C helper` comment (`:17608`):

```cpp
// A load whose arena zones cannot be had, carrying the sizing's own text (llama.cpp-ouur). The plan throws it where
// the process used to abort, and ggml_backend_sycl_stage_inventory_plan() catches it and refuses the load with it.
struct ggml_sycl_zone_sizing_refused : std::runtime_error {
    using std::runtime_error::runtime_error;
};
```

   - `:17621-17628` (the first abort, with its comment) becomes:

```cpp
    // A load whose zones the device cannot hold, or that live allocations (a second model beside a live one) keep
    // from being rebuilt larger, is refused by name: the sizing says which, and what it could give.
    ggml_sycl::zone_sizing_refusal why;
    if (ggml_sycl::get_unified_cache_for_device(ctx->device) &&
        !ggml_sycl::unified_cache_ensure_planned_arena_zones(ctx->device, &why)) {
        throw ggml_sycl_zone_sizing_refused(ggml_sycl::zone_sizing_refusal_text(why));
    }
```

   - `:17813-17815` (the second abort, in the GROW fallback) becomes:

```cpp
                if (!ggml_sycl::unified_cache_ensure_planned_arena_zones(ctx->device, &why)) {
                    throw ggml_sycl_zone_sizing_refused(ggml_sycl::zone_sizing_refusal_text(why));
                }
```

   - In `ggml_backend_sycl_stage_inventory_plan`, immediately before `} catch (...) {` (`:18374`):

```cpp
    } catch (const ggml_sycl_zone_sizing_refused & e) {
        // a deterministic refusal of this load by name: its inputs cannot change, so it is not retried
        GGML_LOG_ERROR("[SYCL-PLAN] %s\n", e.what());
        return GGML_SYCL_LIFECYCLE_PLAN_REJECTED;
```

   The rollback guard (`:18295-18304`) still runs on this path, because `rollback.released` is set only on success
   (`:18372`). The ERROR line is the only place the cause's text is printed. Print it here, in the catch block,
   with no helper.

   `ggml/include/ggml-sycl.h`: add one sentence to the comments of `ggml_backend_sycl_set_tensor_inventory` and
   `ggml_backend_sycl_compute_placement_plan_early` (`:427-443`): "A plan whose arena zones cannot be sized throws
   std::runtime_error to a direct C++ caller; the loader goes through stage_inventory_plan, which refuses the load by
   name instead." No declaration changes.

   `src/llama-model.cpp`: in `llama_model_sycl_apply_inventory`, `:638-641` becomes:

```cpp
    const auto rc = hooks.stage_inventory(&inventory, &envelope, early);
    if (rc == GGML_SYCL_LIFECYCLE_PLAN_REJECTED) {
        throw std::runtime_error(
            format("SYCL backend refused this load's placement plan (early=%d): the refusal is deterministic for these "
                   "inputs, so a retry with the same model, context and devices gets the same answer; the cause is the "
                   "backend's [SYCL-PLAN] error line printed just before this one",
                   early));
    }
    if (rc != GGML_SYCL_LIFECYCLE_OK) {
        throw std::runtime_error(format("SYCL inventory planning failed: early=%d result=%d", early, (int) rc));
    }
```

   Keep the single call. Add no hook field (`:77`) and no resolve (`:106-107`).

   `tests/test-sycl-explicit-ubatch-placement-source.py:190`: `ENSURE_SIG = "bool unified_cache::ensure_planned_arena_zones("`
   (prefix form; its `body()` balances from the next `{`).

   `docs/backend/sycl-memory-design.md:3382-3384`: replace the bullet with:
   "A second model whose term is larger than the live one's raises the RUNTIME requirement, and the late zone rebuild
   is refused while the first model holds allocations. `llama.cpp-ouur` refuses that load by name. The same applies to
   any load whose zones do not fit the device: the arena's size, or the per-allocation cap on a chunked arena.
   `ensure_planned_arena_zones()` says why in a `zone_sizing_refusal`, and the plan throws its text.
   `ggml_backend_sycl_stage_inventory_plan()` prints that text on the backend's `[SYCL-PLAN]` ERROR line, which
   carries the cause, and answers `GGML_SYCL_LIFECYCLE_PLAN_REJECTED`. `load_tensors` names the rejection as
   deterministic and points at that line. A RUNTIME zone within the zones' limits is never refused by them: the
   planner packs the weights into what is left."

6. **Run (GREEN):**

```bash
./scripts/sycl-build.sh test-zone-sizing && ctest --test-dir build -R '^zone-sizing$' --output-on-failure
python3 -m pytest -q -p no:cacheprovider tests/test-sycl-zone-sizing-refusal-source.py      # 8 passed
ctest --test-dir build --output-on-failure -R '^(test-sycl-zone-sizing-refusal-source|test-sycl-explicit-ubatch-placement-source|test-sycl-load-compute-term-source|test-sycl-replan-census-source|test-sycl-zone-reset-audit-source|test-sycl-nonfa-attn-scratch-guard-source|test-sycl-dense-scratch-owner-source)$'
python3 tests/test-sycl-lifecycle-source-contract.py --self-test
python3 tests/test-sycl-host-zone-config-source-contract.py --self-test
./scripts/sycl-build.sh llama-completion     # compile check of libggml-sycl and libllama; run nothing
```

   The tests above are text-only or host-only. Run no model.

7. **(lead, GPU, once; not the implementer)** Capacity vehicle on the B50. `GGML_SYCL_RUNTIME_ARENA_MB`
   (`unified-cache.cpp`, `unified_cache_runtime_zone_floor_bytes`) raises the RUNTIME floor past the arena, so the
   first zone sizing of the load refuses. Under the usual Shmem guard and `GPU.lock`:

```bash
GGML_SYCL_RUNTIME_ARENA_MB=40000 ONEAPI_DEVICE_SELECTOR=level_zero:1 timeout 120 ./build/bin/llama-completion \
  -m /models/mistral-7b-v0.1.Q4_0.gguf -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0 > ouur-b50.log 2>&1; echo "rc=$?"
grep -c 'load refused: SYCL device 0 cannot size its arena zones' ouur-b50.log   # >= 1
grep -c "the device's capacity" ouur-b50.log                                     # >= 1
grep -c 'failed to size VRAM arena zones' ouur-b50.log                           # 0
grep -n -E '\[SYCL-PLAN\] load refused|refused this load.s placement plan' ouur-b50.log   # the [SYCL-PLAN] line first
```

   Expected: `rc` is not 134 (abort) and not 124 (timeout), and the backend's `[SYCL-PLAN]` line comes before the
   loader's "refused this load's placement plan" line. The Task 9 Mistral gate, run without the variable, is the
   positive control. The live-allocation cause has no deterministic vehicle yet. The pgay reproduction (a second
   `llama-bench` GPT-OSS 20B context on the B50 with a wider micro-batch, per its ticket) is a candidate. It is
   UNVERIFIED that it reaches this path; the lead decides whether to run it.

**Commit:** write the message to a file, then
`git add -- <paths> && git commit -F <msgfile> -- ggml/src/ggml-sycl/zone-sizing.hpp ggml/src/ggml-sycl/zone-sizing.cpp ggml/src/ggml-sycl/unified-cache.hpp ggml/src/ggml-sycl/unified-cache.cpp ggml/src/ggml-sycl/ggml-sycl.cpp ggml/include/ggml-sycl.h src/llama-model.cpp ggml/src/ggml-sycl/tests/test-zone-sizing.cpp tests/test-sycl-zone-sizing-refusal-source.py tests/CMakeLists.txt tests/test-sycl-explicit-ubatch-placement-source.py docs/backend/sycl-memory-design.md`.
Subject: `sycl: refuse a load whose arena zones cannot be sized, by name, instead of aborting (llama.cpp-ouur)`. The
body says that `test-sycl-explicit-ubatch-placement-source`'s `ENSURE_SIG` moved to the prefix form because the
definition gained a parameter. It ends with the team trailer from TEAM-BRIEF rule 3.

**Gotchas:**
- The Task 4 description's middle case is not refused by the zones (decision record above). Do not write an
  acceptance check that expects Inkling at `-c 131072` to be refused. What the planner does there is Task 9's question.
- One fact, one source. "What the zones can give" comes only from `zone_arena_tail_fit` (capacity) and the live zone
  capacities (`zone_capacity`, live). Do not compute a room anywhere else. Task 3's total-memory check stays as the
  earlier, cheaper refusal. It is a weaker necessary condition, not a second answer to the zone question.
- Contract pins that must survive:
  - `GROW_PUBLISH` (`tests/test-sycl-explicit-ubatch-placement-source.py:229-230`) pins the one-argument
    `unified_cache_ensure_planned_arena_zones(ctx->device);` in the GROW branch. Leave it.
  - `claim_runtime_zone_is_one_reservation` requires exactly one `zone_runtime_reserved_bytes(` in the bodies of both
    `arena_reserve` and `ensure_planned_arena_zones` (`:362-364`). The new code must not call it.
  - `tests/test-sycl-lifecycle-source-contract.py:996-1007` pins `hooks.stage_inventory(&inventory, &envelope, early)`
    in `src/llama-model.cpp` and `strcmp(name, "ggml_backend_sycl_stage_inventory_plan")` in the backend. The loader
    change keeps the call, and the proc table is not touched.
  - `test-sycl-load-compute-term-source`'s `claim_early_plan_stages_the_probe_plan` (`:378-386`) pins
    `if (early) { ggml_backend_sycl_compute_placement_plan_early(backend, inventory); }` in the stage body. Do not
    reroute the early arm.
- The exception passes out through the two public `void` entries (`ggml_backend_sycl_compute_placement_plan_early`,
  `ggml_backend_sycl_set_tensor_inventory`). Only test binaries call those directly; the loader's path catches it in
  `stage_inventory_plan`. This is the same C++ unwinding the existing `"stale host provisioning candidate"` throws in
  that function use. Do not catch-and-abort inside the `void` entries: the early arm goes through one of them.
- The cause's text reaches the user only through the backend's `[SYCL-PLAN]` ERROR line (lead ruling, decision
  record). Do not copy it into the loader's exception through any side channel. Do not log it a second time.
  `GGML_LOG_ERROR` survives the default verbosity of every tool, but a program that installs its own log callback
  and drops ERROR lines sees only the loader's message, which then points at a line it did not print.
- `PLAN_REJECTED` sources from `ggml_backend_sycl_stage_inventory_plan`, checked on master `1dcdf49b1`: none today.
  The entry returns `BUSY` (`:18268`), `NULL_OUTPUT` (`:18270`), `WRONG_TRANSACTION` (`:18276`), `EFFECT_FAILED`
  (`:18308`, and `:18375` from `catch (...)`), and `NOT_FOUND` or `OK` (`:18373`). Its callees
  `ggml_backend_sycl_compute_placement_plan_early` (`void`, `:17935-17951`) and `ggml_sycl_set_tensor_inventory_impl`
  (`:17978-18229`) return no lifecycle result, and `compute_and_store_plan_for_inventory` (`:17611-17930`) has no
  `PLAN_REJECTED`. The other `PLAN_REJECTED` returns in `ggml-sycl.cpp` (`:25101`, `:25165`, `:25215`, `:25220`,
  `:25232`, `:25382`, `:25725`) belong to the runtime-context publish and recheck functions, which this entry does not
  reach. After this task the new typed catch is the only source, so the loader's message (deterministic, the cause is
  on the `[SYCL-PLAN]` line) is true for every `PLAN_REJECTED` it sees. A later change that adds another
  `PLAN_REJECTED` to this entry must print its cause on a `[SYCL-PLAN]` ERROR line too, or the loader's message
  becomes false. The contract's `stage.count(REJECT) == 1` turns red when that happens.
- A capacity refusal during a rebuild comes after `arena_destroy()` (`unified-cache.cpp:6923`). The device is then
  left with no arena, which held nothing. The next load's first `ensure` takes the `!arena_active()` path and
  reserves again. Moving the fit check earlier is the follow-up in the decision record, not part of this task.
- Multi-device: `stage_inventory_plan` sizes each device in turn (`:18305`). A refusal on device 1 after device 0
  grew its zones leaves device 0's zones larger. `ggml_sycl_warn_runtime_zone_kept_from_earlier_load` (`:17957-17976`)
  already names that for the next load.
- `why.compute_term` is the device's planned term, `unified_cache_get_planned_compute_term_bytes`
  (`unified-cache.cpp:2244`). It is the merge over owners: beside a live draft or target it reads the larger. The
  device index is the backend's post-selector index, the same one the `[VRAM-ARENA]` lines print.
- Formatting: check only your lines with `clang-format-19 --dry-run -Werror <file>`. Never use `-i`. The
  `ggml-sycl.cpp` and `unified-cache.cpp` drift is not yours.
- The tracker ticket llama.cpp-ouur is updated by the lead (the codescout tools do not work in implementer sessions).
  pgay is a related symptom, a second load blocked by the first model's still-live leases. This task turns its abort
  into a named refusal. It does not fix why the leases are live.

**UNVERIFIED in this spike:**
- The B70 arena's exact size. The test fixture uses an illustrative 30720 MiB; the cap of 29472 MiB comes from the doc.
- That Inkling at `-c 131072` loads, and what the planner places on the device beside an 18.5 GiB RUNTIME zone.
- The linear estimate of about 205k tokens for the B70 cap window.
- That the B50 vehicle refuses at the early stage with exit code 1. A no-arena constructor fallback comes first
  (`unified-cache.cpp:6378-6381`).
- That the rollback guard's `ggml_sycl_abort_owner_effects_noexcept` drops the refused load's planned compute term.
  `claim_abort_drops_only_its_owner` in `test-sycl-load-compute-term-source.py:341` pins it; the guard body was not
  read.
- That the `[SYCL-PLAN]` ERROR line appears before the loader's error in each tool's output. Both go to stderr in
  order on the same thread, but a tool with a custom log callback, or one that buffers its own error print
  differently, was not checked.
- Which cause a refusal had, for a caller that sees only the exception (an embedding program without the log). By
  the ruling, the exception carries no cause and no numbers, so capacity and live allocations cannot be told apart
  from it alone.
- None of the C++ in this draft was compiled. The spike was read-only and built nothing.
