# Handoff — SYCL branch session, 2026-08-02

Branch: `feature/sycl-b70-capability` · ~340 commits ahead of `master`, 0 behind.

Read this top to bottom before running anything. Section 1 is housekeeping that will
bite you immediately; section 6 is the one open item that should block a merge.

---

## 1. FIRST: clean up locks and in-flight processes

A session was interrupted mid-build. Before doing anything:

```bash
cd /Apps/llama.cpp
ls -d BUILD.lock GPU.lock 2>/dev/null          # may be stale
pgrep -af 'ninja|icpx|test-llama-archs|test-'  # is anything actually running?
```

- **If a process is alive**, let it finish or kill it deliberately — do NOT remove a lock
  another process still holds.
- **If no process is alive but a lock directory exists, it is stale** — `rmdir BUILD.lock`
  / `rmdir GPU.lock`. Lock files cannot record a live pid here (`$$` dies with the shell),
  so liveness must come from `pgrep`, never from the lock's contents.

There was an unfinished build at HEAD and a chained Case C check queued behind it
(background tasks `b0pcood7l`, `bx553yh6q`). Neither completed. Their results are lost;
re-run from section 5.

Working tree was clean apart from untracked `.pi/remote-pi/`.

⚠️ **`build/bin` IS NOT TRUSTWORTHY — do a rebuild before running anything.** The session was
interrupted during the final link of `libggml-sycl.so.0.15.3`, and that link was killed
mid-write. The file is present and plausibly-sized (134 MB), which is exactly why you must
not trust it — a truncated or partially-linked `.so` will produce results you cannot
interpret, and this session already lost hours to results that were artifacts of how they
were produced. Just run `./scripts/sycl-build.sh` first; ninja will relink.

Locks were released at handoff (both absent). Some compiler processes may still be draining —
check `pgrep -af 'ninja|icpx'` and let them exit or kill them before building.

---

## 2. Ground rules that were learned the hard way this session

- **GPU work is serialised through ONE session.** Subagents write code; they do not run it.
  Two agents (`impl-w2`, `impl-w3`) worked all session without touching the GPU.
- **Always pin the selector**: `ONEAPI_DEVICE_SELECTOR=level_zero:0`. Measured this session:
  the FULL 130-arch sweep peaks at **3 GB `Shmem`** pinned. `CLAUDE.md` previously recorded
  only the single-arch figure and flagged the full sweep as unmeasured — that gap is now
  filled. Unpinned costs 195–206 GB.
- **Always `source /opt/intel/oneapi/setvars.sh --force`** before any binary. Omitting it
  aborts in `dpct::dev_mgr::dev_mgr()` at init with `rc=134` — it does NOT skip quietly.
  (I lost one full sweep to this.)
- **The full sweep needs `GGML_SYCL_OP_TIMEOUT_MS=120000`** on a loaded host. At the 30 s
  default it watchdog-trips on `grok` under load ~64 (codescout re-index at 1044 % CPU).
  grok alone passes `OK (2.29e-12)` — the trip is load, not a regression.
- **Never build while tests run from `build/bin`**, and never build with dirty backend
  sources — you get a binary matching no commit, and results you cannot attribute.
  Guard used:
  ```bash
  DIRTY=$(git status --porcelain -- ggml/src/ggml-sycl/ | grep -v '^??' | wc -l)
  [ "$DIRTY" -ne 0 ] && { echo "ABORT: source dirty"; exit 1; }
  ```

---

## 3. VERIFIED — fixed and confirmed on hardware

| what | commit | evidence |
|---|---|---|
| gemma3n bug 1 — bit1 fused a view-at-nonzero-offset operand | `d293bf2b3` | Mutation A/B: guard forced `return true` → `-a gemma3n` `FAIL (7.04e-01)`; guard intact → `OK (2.17e-06)`, same build |
| `test-layout-bytes` Q8_0 stale expectation | `1107a53b0` | Predicted statically (`expected 340, got 2068`), then `Layout bytes test: PASS` |
| `cpu_mul_mat` missing `thread_local` init (partial) | `cd02cd5f3` | `test-mul-mat-host-streaming`: SIGABRT → `rc=0`, `nmse=5.13e-14` |
| `EXPERT_STAGING` + `HOST_COMPUTE` misrouted to WEIGHT zone | `282069dd4` | **VERIFIED just before handoff.** `test-sycl-moe-handle-resolution`: was `FAIL: EXPERT_STAGING should route to SCRATCH host zone` → now `SYCL MoE handle resolution tests: PASS`, `rc=0`. A pre-existing RED (not written for this fix) flipping green — the best provenance available |

**Full sweep at `1107a53b0`: `rc=0`, 130 archs, 0 watchdog, 0 FAIL rows.** See section 6
before trusting that.

---

## 4. COMMITTED BUT NOT BUILD-VERIFIED — do these first

| commit | what | how to verify |
|---|---|---|
| `dd81d76be` | Full `g_cpu_dispatch_buffers` consumer audit; 6 more sites | `test-sycl-cpu-dispatch`, `test-mul-mat-host-streaming`, `test-sycl-xmx-unified-correctness` |
| `2f0df918d` | Case C `ggml_add` argument swap so it reaches bit1 | section 5 |

---

## 5. THE VERIFICATION QUEUE (exact commands)

Run in this order. Every one needs oneAPI sourced + `GPU.lock` + pinned selector.

### 5a. cpu_dispatch audit (dd81d76be) — EXPERT_STAGING already verified, see section 3
```bash
./scripts/sycl-build.sh
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-sycl-moe-handle-resolution
for t in test-sycl-cpu-dispatch test-mul-mat-host-streaming test-sycl-xmx-unified-correctness; do
  ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/$t; echo "$t rc=$?"
done
```
Watch: `test-sycl-xmx-unified-correctness` previously progressed SIGABRT → `rc=1` with
`SKIP: no graph-pinned entries` then `FAIL: SYCL backend run failed`. **If it fails
identically after `dd81d76be`, that is a separate second defect** — file it, don't fold it
into `llama.cpp-ktlb`. Nobody has checked whether that SKIP causes the FAIL or is merely
adjacent.

### 5b. Case C reachability — pass criteria recorded BEFORE the run
```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_FUSION_BIT1_REACH_DEBUG=1 \
  ./build/bin/test-rms-norm-mul-add-broadcast 2>&1 | grep FUSION-BIT1
```
Want: **6 `[FUSION-BIT1-GUARD]` lines** (was 4) and **0 `op_seq=0` lines** (was 2).

### 5c. Case C mutation A/B — the proof it guards anything
Case C has NEVER been observed to fail. Until it does, it is decoration.
1. Force `ggml_sycl_fusion_operand_view_offset_safe` to `return true;` at its top
   (`ggml-sycl.cpp`), rebuild → **Case C must FAIL**.
2. `git checkout -- ggml/src/ggml-sycl/ggml-sycl.cpp`, rebuild → must pass.

Both directions required. One is not enough — that is how this test got shipped void.

### 5d. THE IMPORTANT ONE — see section 6.

---

## 6. ⚠️ OPEN AND UNEXPLAINED — gemma3n's failure vanished with no known fix

**This is the item that should block a merge.**

```
sweep at 4407acc83 (pre-fix):   gemma3n FAIL (1.14e+00)
sweep at d293bf2b3 (post-fix):  gemma3n FAIL (1.14e+00)   bit-identical, all 413 rows matched
sweep at 1107a53b0 (now):       gemma3n OK   (7.57e-07)   clean sweep, 0 FAILs
```

Every commit in that range is behaviour-neutral, and I verified rather than assumed:
- `e2df03733` — block gated on `static const bool ... = getenv("GGML_SYCL_FUSION_BIT1_REACH_DEBUG")`
- `77f7d7c8a` — refactor is genuinely equivalent: `if (A && B) { walk; if (offs) return false; }`
  → `if (A && B && walk() != 0) return false;`
- `98e26fac2` artifacts only, `5bdb2fd6f` comment, `1107a53b0` test file

**Leading theory: binary layout.** Env-gated diagnostic code still changes binary size and
shifts allocation addresses. A defect sensitive to alignment/address values would flip on
that alone — and would explain bit-identical failures across two builds sharing a layout,
then a pass on a third that doesn't. **If true, the bug is intact and merely invisible.**

**The decisive experiment, NOT YET RUN:**
```bash
git checkout d293bf2b3 -- ggml/src/ggml-sycl/ggml-sycl.cpp   # reverts both diagnostics
./scripts/sycl-build.sh
ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_OP_TIMEOUT_MS=120000 ./build/bin/test-llama-archs
git checkout -- ggml/src/ggml-sycl/ggml-sycl.cpp             # RESTORE
```
- `FAIL (1.14e+00)` returns → commit-linked; one of those "neutral" commits isn't. Bisect.
- Passes → nondeterministic; both earlier bit-identical FAILs were a shared-layout
  coincidence, and `llama.cpp-8t4s` needs reframing around an intermittent defect.

Use file-restore, **not** `git checkout <sha>` — agents may hold live state in the checkout.

**Also withdrawn:** my "one preceding architecture is enough" claim in `llama.cpp-8t4s`
rested on a SINGLE observation (`FAIL 1.03e+00`) against two later OKs. Treat that repro as
unverified.

**Unaffected:** the `81gx` guard stays. It is mutation-proven for the isolated case.

---

## 7. Tickets filed this session

| id | pri | what |
|---|---|---|
| `llama.cpp-8t4s` | P1 | gemma3n cross-model failure — **see section 6, status changed, do not close** |
| `llama.cpp-ktlb` | P1 | `g_cpu_dispatch_buffers` `thread_local` vs `CpuExpertPool` workers — **live production abort** at MXFP4 `chunk4>256` |
| `llama.cpp-99ke` | P1 | MMVQ launch requires `nrows % 16 == 0`; **11 launch sites**; `nrows` is a *slice* (`row_diff`), so multi-device row-splitting is the risk. Needs an owner ruling on the work-group perf tradeoff — `mmvq.cpp` is a hot path |
| `llama.cpp-7f2e` | P1 | `9a0670712` audit. Downgraded P0→P1 with justification |
| `llama.cpp-81gx` | open | gemma3n bug 1 — fix verified; Case C work pending (5b/5c) |
| `llama.cpp-0igs` | open | the restoration; comment `c-wlic` has the full consumer audit |

---

## 8. DECISIONS NEEDED FROM THE OWNER

1. **Tied-weight cache identity.** `5de614b8c` made `cache_id_equal` compare `name_hash`
   unconditionally — correctly stopping distinct MoE experts aliasing to one cache entry.
   Side effect: tied embedding/lm_head (different names, same GGUF offset) no longer share
   a device entry, so each gets its own VRAM copy. Not recorded anywhere, including
   `docs/design/sycl-canonical-memory-architecture.md`. Direction is almost certainly right;
   nobody decided the tied case explicitly. `test-sycl-weight-key-uniqueness` is red on it.
2. **`GGML_SYCL_UNIFIED_CACHE=0` was silently removed** by `9a0670712` — `unified_cache_enabled()`
   is gone entirely (it gated ~30 call sites). "Unified cache always on" IS a documented
   decision (`CLAUDE.md`, Feb 9 2026), so the direction is intentional. The question is the
   *convention*: this fork documents its opt-outs (`GGML_SYCL_UNIFIED_SOA=0`,
   `GGML_SYCL_TG_FAST=0`, …) and this one vanished undocumented.
3. **`g_tiered_enabled`** conflates *"is the unified cache active"* with *"does this model
   need tiering"* — unrelated questions. **This predates `9a0670712`**; that commit removed
   the escape hatch that could mask it. Confirmed: `test-tiered-dispatch` never set
   `GGML_SYCL_UNIFIED_CACHE`, so it was already failing before. The test's premise (tiered
   varies by inventory size) is CORRECT; the production flag is wrong.

---

## 9. Why any of this was found — the `0igs` result

41 previously-deselected tests were restored. **14 failed. Only ONE was a merely-stale test.**

Real defects found, all from tests that had been sitting unregistered:
- `g_cpu_dispatch_buffers` live production abort (`ktlb`)
- `g_tiered_enabled` conflation — fires on every model load
- MMVQ `nrows % 16` crash, 11 sites (`99ke`)
- `EXPERT_STAGING` misrouting (leak-shaped)
- Q4_0 coalesced DMMV wrong answers — 12–31 % max_rel on ~35–40 % of rows, on a real
  dispatch path (the "not implemented" placeholder is a *different, uncalled* function)

Not defects on closer evidence: `test-sycl-unified-memory-e2e` (VRAM budget starvation —
the test doesn't constrain its budget, arena takes 32026 of 32656 MB at pct=100, then
`malloc_device` returns nullptr ×7), `test-layout-bytes` (genuinely stale, fixed).

Three of these trace to **`9a0670712`** ("checkpoint unified memory ownership work") — a
commit `CLAUDE.md` already indicts for a *fourth*, loud regression that was fixed in days.
The loud damage got repaired; the silent damage inherited the repair's credibility and
lived ~7 weeks.

---

## 10. Traps hit this session — do not repeat

- **A filter can hide the signal and leave an absence you read as a finding.** I grepped
  `-vE '^\[UNIFIED…'` as "banner noise", deleting the 7 `malloc_device returned nullptr`
  lines, then read the empty tail as "silent for 240 s ⇒ deadlock". It was budget
  starvation. When a conclusion rests on ABSENCE, re-read unfiltered.
- **A test can run fully — allocate, take time, print real numbers — and never reach the
  code it names.** Case C: correct shape (view at non-zero offset, exactly `slice_rest`),
  `op_seq=0`, never touched bit1. Its green was cited as a finding ("bit1's kernel is
  correct") that redirected the investigation for rounds.
- **`test-llama-archs`: `-a` OVERWRITES (single variable), `-x` ACCUMULATES (set).** To build
  a multi-arch subset, exclude everything else. And filter the arch list with
  `grep -E '[a-z0-9]'` or the table's `|-----|` separator parses as an arch name and the run
  dies at argument validation.
- **`GGML_ABORT` never reaches a log** — grep the string the site prints
  (`ggml-sycl\.cpp:[0-9]+:`), never the macro name.
- **codescout's index is blind in `ggml-sycl.cpp`** (94k lines). It surfaced 1 of 8+ call
  sites in one search here. Use `cat … | grep -n`.
- Two agents each corrected the lead this session, and both were right. Reports that revise
  their own severity *downward* (or bound their own scan's limits) were the most reliable.

---

## 11. Agent state at handoff

`impl-w2` (bit1 fusion / gemma3n) and `impl-w3` (`0igs` triage) were both idle-available,
all work committed, nothing dirty. `impl-w3` was about to write the closing `0igs` report.
Neither has GPU access by design. Messages crossed frequently — check a teammate's latest
message before assuming an instruction is current.

---

## 12. impl-w3's closing report — scope not covered above

Delivered just before shutdown. Adds detail this handoff did not otherwise capture.

### Corrected count on `llama.cpp-ktlb`
**8 touch sites across 6 public entry points** (not "6 sites"). All fixed across
`cd02cd5f3` + `dd81d76be`, placed at function entry or inside the specific parallel
lambda depending on whether the touch can migrate threads.

### Still unresolved — read but NOT root-caused
- `test-q8-0-layout-cache-path` and `-mmvq` — both fail identically at
  `Failed to resolve SoA layout pointer (source=wrong_layout)`, never reaching their
  numerical comparison. Partially traced, no specific defect named. Open question worth
  checking first: is `wrong_layout` downstream of the same Q8_0 warp-tile padding change
  (`d87d54cdd`) that made `test-layout-bytes` stale? If so, one root cause covers three
  tests.
- `test-q6k-dispatch` — Test 2 fails on accuracy (max_rel 1.4442 % against a 1 % gate);
  Tests 3 (determinism) and 4 pass. Determinism passing **rules out a race**. Unresolved
  whether 1 % was ever a justified threshold — `git log -S` on the constant would settle
  (a) real regression vs (b) marginal gate.
- `test-sycl-xmx-unified-correctness` — progressed past its abort thanks to `ktlb`'s fix;
  now `FAIL: SYCL backend run failed` preceded by `SKIP: no graph-pinned entries`.
  Cause not chased. **Nobody has checked whether that SKIP causes the FAIL or is merely
  adjacent** — do not assume ordering means causation.
- **DMMV Q4_0 coalesced numerical error has NO OWNER.** Confirmed real, reachable through
  production `graph_compute`, ~12–31 % max_rel. Root-caused only as far as "the GPU
  disagrees with CPU and the test's own decoder is exonerated" (its Part 1 CPU layout
  check passes). Needs a numerical investigation nobody started.

### Restoration scope NOT completed
- **22 of the 64-file set remain**: 12 GPU-touching, 5 model-loading hazards (do NOT
  register as-is — see the never-loop rule), 5 never-registered-at-`3c8f296fd`
  (mock-vs-real triage). Deferred deliberately, not started.
- **~83 files outside this branch's changed surface** — explicitly out of scope, tracked
  in `llama.cpp-0igs`.

### Process traps impl-w3 hit (additional to section 10)
- **Two of its own harness bugs had the same shape as the defects it was auditing** —
  `tail | grep` reading the wrong command's exit code. The tooling you write to find a
  defect class is susceptible to that same class.
- **`ctest -LE` is an unanchored substring match**, which cost a fix-then-refix cycle:
  `cache-hostonly` still matched `-LE cache`. Verification used equality where ctest uses
  substring.
- **codescout's oversize blind spot on `ggml-sycl.cpp` hid 7 of 8** production
  `ggml_sycl_cpu_dispatch_buffers_init()` call sites. `cat … | grep -n` was required
  throughout. Never conclude "no other callers" there from the index.
