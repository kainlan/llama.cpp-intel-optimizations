# SYCL crash-work completion plan

> Execution: `pi-team-driven-development`, with **Astra at low effort for owners and fresh independent reviewers**, replacing the skill's Luna/Sol model defaults at the user's request.

**Status:** Draft for approval, 2026-09-13. Planning only; no production changes, builds, GPU runs, merges, or existing task ownership changes authorized by this document alone.
**Planning tracker:** `llama.cpp-c2wq`. Existing feature tickets remain authoritative; this document is not a second status tracker.
**Goal:** Finish the recovered Gemma4 parallel-slot, allocation, auto-ubatch, build-safety and directly associated follow-up work without losing the previous agents' changes or treating historical test results as current proof.
**Architecture:** Complete existing fixes and their integration contracts. Preserve canonical SYCL memory ownership and FULL/SWA/SHARED layer accounting. Do not redesign allocation or add another planner/cache path. The separate `fkpg` caller-to-load context contract needs evidence and a bounded design decision before production edits.
**Test infrastructure:** Existing C++ numerical/cache/runtime fixtures, Python source gates, SYCL real builds, lead-run CTest and model/server acceptance, and the existing Docker-flags configure/import audit. Source checks are supplementary, not proof of runtime behavior.

## 1. Scope and evidence baseline

Two read-only planning workers verified `PI_MODEL=gpt-6-astra` and `PI_REASONING_LEVEL=low`. Their snapshot is not evidence of current runtime correctness or a live previous owner.

| Workspace | Branch / observed HEAD | State to preserve |
| --- | --- | --- |
| `/Apps/llama.cpp` | `master`, `7b03ea03fbb2fa6f1da755e7c0d099a3de22dd7b` | Pre-existing wiki changes and untracked artifacts, test output and notes; do not stage or remove incidentally. |
| `/Apps/llama.cpp-wt-0oxf` | `task/3aos`, `2e52f24bfb4186a03bafb58bbd4eb21224fb7bb8` | Already merges reviewed `652ff12f2` with master `7b03ea03f`; seven uncommitted continuation paths. Do not repeat the merge blindly. |
| `/Apps/llama.cpp-wt-o3a0` | `task/pyu4`, `d0e0e197377652dab6cf06d88fd0c8e3596b3011` | Uncommitted context/source-test continuation. |
| `/Apps/llama.cpp-wt-glkg` | `task/glkg`, `7b03ea03fbb2fa6f1da755e7c0d099a3de22dd7b` | Staged allocation, fixture and documentation changes. |
| `/Apps/llama.cpp-wt-o65k` | `task/6rno`, `542d0d880` | Reviewed commit not an ancestor of snapshot master; tracker says merge launched. Reconcile process/log/worktree state before touching it. |

### Protected continuation paths

- **3aos:** `docs/backend/sycl-env-vars.md`, `ggml/include/ggml-sycl.h`, `ggml/src/ggml-sycl/tuning-cache-io.hpp`, `ggml/src/ggml-sycl/ubatch-tuning-cache.cpp`, `src/llama-context.cpp`, `tests/test-sycl-auto-ubatch-source.py`, `tests/test-tuning-cache-io.cpp`.
- **pyu4:** `src/llama-context.cpp`, `tests/test-sycl-auto-ubatch-source.py`.
- **glkg:** `ggml/src/ggml-sycl/{ggml-sycl.cpp,unified-cache.cpp,pinned-pool.cpp,CMakeLists.txt,tests/test-unified-runtime-alloc.cpp}`, `tests/test-sycl-host-zone-config-source-contract.py`, `docs/backend/sycl-env-vars.md`.

Read and reserve those exact existing changes before adopting them. Do not replace them with a fresh implementation. The lead must first establish that no previous owner or merge/build process is still writing the workspace. Stale tracker assignment alone is not permission to take it over.

**Closed prerequisites:** `7n6n` (persisted ubatch cache, merge `fd9b30fd9`) and `n4ee` (DL symbol fix, merge `7b03ea03f`) remain closed. Verify ancestry; do not redo their work. Their results do not qualify new deltas.

**Historical evidence only:** 3aos records GPU results on `ed2927552` and review counts of 124 numerical / 20 source checks on `652ff12f2`. `scratchpad/3aos-gpu-run2/`, `3aos-branch-gpu-v2.sh`, `merge-3aos.sh`, and pyu4's referenced `merge-xojq.sh` were not found locally. Recover their exact commands/artifacts or replace them with committed, reviewed reproducers. Do not claim those unavailable scripts ran on the current continuation.

## 2. Team topology, safety and delivery

- **Default up to two feature owners plus one review/specialist slot**, with an aggregate cap of three or the actual collaboration limit, whichever is lower. Retained owners and reviewers count against that limit; reduce owner fan-out when fewer slots are available so independent review can run. No nested delegation under this session's recursion restriction.
- Use `openai-codex/gpt-6-astra:low`. At execution setup verify the actual `team_spawn` route supports the model/effort selector, then verify each child's reported model and reasoning level. Stop and report unsupported configuration rather than silently substituting another model/effort.
- Use one persistent `team_*` execution team. Each owner retains one coherent feature through implementation, continuous tests, integration and its review fixes. Fresh read-only Astra-low SPEC, then fresh Astra-low QUALITY reviewers; never reuse reviewers. Model substitution does not weaken evidence or review policy.
- Preserve the supplied skill's required role/evidence clauses. Unique member names; reconcile old process/member identity before replacing a previous implementer. No speculative changes to an approved feature contract.
- **Lead owns all GPU/model-loading runs and CTest sweeps, serialized.** Owners may run approved CPU-only focused tests and builds. Maximum two builds across the entire team, normally one while qualifying; each explicitly granted build defaults to `-j6` and the lead lowers concurrency for memory pressure. No competing throughput workload. Inspect actual devices, host load, MemAvailable and Shmem before reservations; do not infer current hardware from old task titles.
- Source oneAPI for binaries. Respect `CLAUDE.md` device selection, no unsupported FP64 override, GPT-OSS template, timeout, warning, load supervision and performance-baseline rules. Do not run GPU tests merely to develop this plan.
- Unknown/overlapping ownership uses a task worktree outside the lead checkout. Existing worktrees may be adopted only after owner/liveness and dirty-change reconciliation. New independent work gets a unique task branch/worktree. One writer per shared file; isolated branches do not eliminate dependency/conflict review.
- Every worktree prompt passes `root=<absolute workspace>` on Codescout tools accepting it and verifies echoed roots. Indexed misses are not absence; use live `search_text` and direct reads for dirty files.
- Commits are path-scoped after checking staged paths. No reset, stash, force removal, broad cleanup or unrelated staging. No new file under `tests/*` without maintainer approval: extend existing fixtures. A worker's proposed new census test is not approved by this plan.
- **No push or release is authorized yet.** Ask the repository owner before pushing. Execution merges are serialized and must qualify the resulting immutable candidate, not just a branch build.

## 3. Workstreams and ordering

All shortened ticket IDs below have prefix `llama.cpp-`. Grouping related tickets does not erase their individual acceptance. At execution setup record dependency edges only after checking current records; retain one accountable owner for each grouped outcome.

| Lane | Outcome / tickets | Ordering and ownership boundary |
| --- | --- | --- |
| A | Parallel-slot KV correctness + cache-v3 integration: `3aos`, symptom `17fe` | First core integration; closed `7n6n`/`n4ee` prerequisite ancestry. |
| B | Safe auto-ubatch reserve and fallback: `pyu4` | After A for shared context/source-gate edits. |
| C | Allocation/headroom correctness: `glkg`, placement explanation `lzok` | Can inspect/test existing staged work alongside A; serialize backend writes/integration with A. NAS proof remains a release gate. |
| D | Reliable CTest registration/selection: `6rno`, `6ism` | Reconcile 6rno's launched merge first; then census/labels. Single `tests/CMakeLists.txt` writer. |
| E | Robust shared cache I/O: `oxpe` | After A's cache-v3 contract; parallel with B only while files remain disjoint. |
| F | Gemma tensor registry remainder: `ttws` | After A; serialize `ggml-sycl.cpp` with C. Independent root cause, not automatic A closure. |
| G | Actual-context scratch planning: `fkpg` | After A/C memory contracts; contract/evidence hold described below. Not a duplicate of A. |
| H | Complete DL forbidden-call source coverage: `f2p0` | Independent test scope; CMake edits serialized with D. |
| I | Operational documentation: `wbci`, `p84f`, `0v28`, `0su3` | After owning behavior stabilizes; serialize shared docs. 0su3 uses final measured E2E outcomes. |
| Lead gate | Merged-master DL proof: `mzvn` | On final immutable integrated SHA after relevant backend/API changes; closed n4ee is not repeated implementation. |

**Suggested scheduling:** after liveness preflight, reconcile D's already-reviewed integration and give A the first core owner. Use the other slot for C's bounded CPU work or H while GPU/build resources are unavailable. B and E follow A; F follows A/C; G advances only through its bounded evidence/contract gate. Finish I against real results. Fill freed slots from this campaign's useful ready tickets, not the entire repository backlog. Run `task_ready` at setup and after each closed feature and completed cleanup.

**Shared-file serialization:** A before B (`src/llama-context.cpp`, auto-ubatch source gate); A before E (cache I/O + fixture); A/C/F/G serialize backend memory interfaces and `unified-cache.cpp`; D/H serialize `tests/CMakeLists.txt`; all producers finish before final edits to `sycl-env-vars.md`/`CLAUDE.md`. If actual deltas introduce another overlap, reserve it explicitly before writing.

### A. Finish Gemma4 parallel slots and cache integration

**Acceptance:** Gemma4 server loads at normal/default parallelism and explicit four slots without KV remap overflow; real overlapping requests complete coherently. Preserve FULL/SWA/SHARED per-layer accounting and both `kv_unified` modes. Keep historical layer-zero width correction: width 512 is real; do not implement the superseded title diagnosis. Persisted ubatch keys include `kv_unified` in serialization, equality, conversion and caller assignment; cache version 3 invalidates incompatible data and distinguishes alternating cache modes.

**Scope:** seven protected continuation files above, plus existing committed changes in `src/llama-model.cpp`, `ggml/src/ggml-sycl/{ggml-sycl.cpp,unified-cache.cpp,unified-cache.hpp,kv-runtime-demotion.cpp,kv-runtime-demotion.hpp}` and their existing registered fixtures. No scratch transport redesign or registry-warning suppression.

**Continuous checks (from owning workspace):**

```bash
python3 -m pytest -q tests/test-sycl-kv-layer-sizing-source.py tests/test-sycl-auto-ubatch-source.py
CMAKE_BUILD_PARALLEL_LEVEL=6 ./scripts/sycl-build.sh
# Lead-run registration proof, then execution:
ctest --test-dir build -N -R '^(sycl-kv-layer-sizing|test-tuning-cache-io)$'
ctest --test-dir build -R '^(sycl-kv-layer-sizing|test-tuning-cache-io)$' --output-on-failure -j 1
```

**RED/GREEN:** extend existing numerical/cache fixtures for the changed production boundary; observe the committed reproducer fail on an immutable pre-fix candidate, then pass on the frozen completion candidate. Recover or commit a server acceptance harness with exact model/digest, default and `-np 4` launches, concurrent requests, both cache modes, readiness/timeout/error checks and sandboxed cache location. A completion-only invocation does not prove server-slot behavior. Lead also runs the existing GPT-OSS/Mistral correctness and single-slot controls using verified templates/flags.

**Done:** real target build and nonzero relevant tests, current-SHA runtime evidence, independent reviews, reviewed merge. Close `17fe` only with its exact server symptom covered; otherwise keep it open with executable residual evidence.

### B. Finish reserve/fallback ladder behavior

**Scope:** protected pyu4 context/source-gate edits; related `sycl-env-vars.md` documentation after A. Preserve valid cache-attempt handling and rollback on reserve failure.
**Acceptance:** early exit detects no rung satisfying `fallback_ubatch <= rung <= cap`; no empty trial sweep or bogus terminal cache entry. Cache hits, misses, reserve exception cleanup, fallback and WARN behavior remain correct.

```bash
python3 -m pytest -q tests/test-sycl-auto-ubatch-source.py
CMAKE_BUILD_PARALLEL_LEVEL=6 ./scripts/sycl-build.sh
```

**Evidence hold before review:** current G1 checks around `test-sycl-auto-ubatch-source.py:1017-1143` are regex/text mutation witnesses, not runtime reserve/cache tests. Extend an existing suitable executable fixture to cross the changed boundary and exercise the documented `(1000,1000,600)` and `(600,2048,1024)` cases with argument meanings explicitly recorded. Freeze and execute RED/GREEN, including cache writes/attempts and zero valid candidates. If existing fixtures cannot reach the seam, ask for approval before a new `tests/*` file. Recover historical harness details rather than inventing an interface.
**Done:** focused behavioral evidence, real build, affected broader gate, documentation and independent SPEC/QUALITY on the integrated feature. Static-source GREEN alone cannot close pyu4.

### C. Finish host allocation ordering and honest placement reporting

**Scope:** protected glkg staged files; `docs/backend/sycl-memory-design.md` if needed for `lzok`. Preserve existing changes before editing. No GGTT-based pinned-memory cap.
**Acceptance:** host zones are provisioned before host weight allocations consume the pool; max-size reporting and actual allocation route agree; context buffers retain adequate headroom; staging/fallback remains accounted and diagnosed. Explain host-resident expert execution versus the nominal offloaded-layer count; do not emit a misleading host-expert warning for Mistral.

```bash
python3 tests/test-sycl-host-zone-config-source-contract.py --self-test
# Source-contract comparison against snapshot master, NOT runtime allocation proof:
python3 tests/test-sycl-host-zone-config-source-contract.py --root /Apps/llama.cpp
CMAKE_BUILD_PARALLEL_LEVEL=6 ./scripts/sycl-build.sh
```

**RED/GREEN:** extend existing `ggml/src/ggml-sycl/tests/test-unified-runtime-alloc.cpp` for ordering, headroom and allocation/max-size routing; discover its actual CTest registration before running it. Execute the real changed allocation boundary, not just its source gate. Do not assume the staged 64-MiB reserve is sufficient from inspection. Lead replays Qwen load/context/generation and a dense control locally. NAS reporter must replay the exact failing Qwen model/launch on the same candidate and return digest/config/log/results.
**External gate:** exact NAS GGUF, launch, digest and captured error are not available locally. Obtain them; do not substitute a different model and close the reported NAS defect. Historical 13 chunks / 26 GiB ordering evidence is a root-cause lead, not current GREEN.
**Done:** local boundary proof, NAS reproduction and same-scenario GREEN, current target build, placement documentation, reviews and merged evidence.

### D. Land safety registration and select every source gate

**Scope:** `scripts/check-ctest-safety-net.sh`, existing `tests/test-sycl-ctest-registration-order-source.py`, `tests/CMakeLists.txt`, relevant `CLAUDE.md` guidance. Extend existing tests for census coverage rather than creating the proposed new census file without approval.
**Acceptance:** reconcile and qualify `542d0d880`; registration safety floor counts 400 occurrences, not 400 distinct tests. All six omitted source gates enter the intended family/label selection. Compare actual selected names/counts and rebaseline failures against `uv62`, not an assumed clean suite.

```bash
python3 -m pytest -q tests/test-sycl-ctest-registration-order-source.py
bash scripts/check-ctest-safety-net.sh --build-dir build
# Lead only:
ctest --test-dir build -N -L source-assert -E '^mem-handle-byte-contract$'
ctest --test-dir build -L source-assert -E '^mem-handle-byte-contract$' --output-on-failure -j 1
```

**RED/GREEN:** preserve registration-order regression coverage; commit a census/selector witness that fails for the six currently omitted names and passes with complete labels. Execute the selected gates and record non-vacuous counts, skips and baseline failures. Missing selection is not test success.
**Done:** each ticket's specific gate covered, candidate build configured correctly, independent review and verified merged commit. Do not start a second merge process while the old launch is unresolved.

### E. Finish shared cache parsing

**Scope:** `ggml/src/ggml-sycl/tuning-cache-io.hpp`, existing `tests/test-tuning-cache-io.cpp`; preserve A's v3 `kv_unified` contract.
**Acceptance:** shared reader/writer handles brace-containing model names through a real write/read round trip, with existing valid/invalid-cache behavior preserved. No new cache format redesign beyond the documented contract.
**RED/GREEN:** add the brace round-trip to the existing executable fixture; freeze the pre-fix candidate with that durable test, observe failure, implement and replay. Build the real target and run the `test-tuning-cache-io` registration/execution commands from A, then affected cache regressions. Source matching is not parser proof.
**Done:** integrated fixture/cache correctness, independent reviews; leave unrelated optional refactoring out.

### F. Resolve the separate Gemma weight registry warning

**Scope:** `ggml/src/ggml-sycl/ggml-sycl.cpp`; inspect `src/llama-model-loader.cpp` and existing cross-buffer duplicated-tensor / displaced-extra ownership explanation. Use existing `GGML_SYCL_DKW0_PTR_CHECK` diagnostics where appropriate.
**Acceptance:** classify the actual `token_embd.weight` mismatch after A, preserve ownership/lifetimes and output correctness, and resolve the warning's cause or prove/document a safe case without hiding corruption. Determine whether the old master's `result=19` was KV planning or registry failure.
**Evidence hold:** no dedicated executable reproducer was located. Owner recovers A's load scenario and commits the smallest supported fixture/harness; lead executes it on the immutable candidate before production changes. A source-level suspicion about duplicates is HYPOTHESIS, not permission to delete ownership checks.
**Done:** exact warning scenario reproduced and resolved, appropriate behavioral memory/ownership evidence, model correctness control, build and independent review. Separate from A because runtime cause remains unproved.

### G. Finish real-context scratch planning without guessing an API

**Established outcome:** `fkpg` still needs actual-context scratch sizing, including agreed multi-slot/non-FA behavior. A does not close it. Historical RED was planning at 512 and servicing a 768-MB request through DIRECT, not necessarily a crash.
**Grounded scope:** `common/common.cpp`, `tools/llama-bench/llama-bench.cpp`, `include/llama.h`, `src/llama-model.cpp`, `ggml/include/ggml-sycl.h`, `ggml/src/ggml-sycl/unified-cache.cpp`, existing floor/lifecycle wrapper fixtures.
**Contract gate:** current `llama_model_params` has no intended-context/envelope field. `llama_model_sycl_make_placement_envelope()` takes no arguments and emits zero context/ubatch with one sequence. Both model inventory paths call it. The backend envelope calls `n_ctx` per-slot, while the common caller supplies its context value; bench reuses a model across rows with differing context shapes. Do not blindly add a public field or copy the first row. Freeze and reproduce the existing shortfall, then propose the smallest transport/units/model-reuse decision for approval before production API changes. This draft does not approve that design by implication.

**Existing numerical checks, lead-run after real build:**

```bash
ctest --test-dir build -N -R '^test-sycl-onednn-graph-floor(-override)?$'
ctest --test-dir build -R '^test-sycl-onednn-graph-floor(-override)?$' --output-on-failure -j 1
```

Extend `tests/test-sycl-onednn-graph-floor.cpp` for agreed arithmetic and consider existing `tests/test-sycl-lifecycle-runtime-wrapper.cpp` for envelope propagation. Existing tests do not yet prove common/bench-to-model transport. Unsupported oneDNN/SYCL skips are not GREEN.

**Verified bench shape for future lead-only reproduction:** after sourcing oneAPI and confirming the selector still identifies the intended device,

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:1 GGML_SYCL_DEBUG=1 ./build/bin/llama-bench \
  -m /models/mistral-7b-v0.1.Q4_0.gguf -p 8192 -n 0 -ub 512 -fa on -r 1 -v
```

Repeat separately with `-fa off`. Bench has no `-c` flag; `-p 8192 -n 128` creates separate PP/TG instances, so it is not a single 8320-token context. GREEN must show intended shape reaching sizing, bounded allocation and actual PP8192 completion; graceful refusal alone does not meet non-FA acceptance. Record multi-slot producer semantics/tests after the bounded contract is approved.
**Done:** approved, evidenced caller/load contract; runtime producer and floor tests; FA/non-FA and multi-slot acceptance; docs/build/reviews. If contract approval or evidence is unavailable, keep G explicitly blocked rather than declaring the campaign complete.

### H. Extend the DL source safety net

**Scope:** existing `tests/test-sycl-attn-host-dl-source.py` and its registration comment in `tests/CMakeLists.txt`; production SYCL `.cpp/.hpp` census excludes backend tests.

```bash
python3 -m pytest -q tests/test-sycl-attn-host-dl-source.py
```

**RED/GREEN:** executable source-check mutation removing the getrows guard must fail and identify file:line; derive the real production-file census rather than hard-coding an unexplained count. GREEN covers all intended translation units and headers with nonzero executed cases. This is a test-tooling outcome, not evidence of runtime DL loading; that belongs to `mzvn`.
**Done:** complete non-vacuous census/mutation checks, registration/docs as needed, independent reviews; no unrelated backend edits.

### I. Finish associated operational documentation

**Scope:** `CLAUDE.md`, `docs/backend/sycl-env-vars.md`, `docs/build.md`, comment-only `.devops/intel.Dockerfile`; `0su3`'s existing auto-ubatch documentation and measured acceptance references. `lzok` stays with C's behavior owner.
**Acceptance:** `wbci` isolates persisted tuning caches during WARN gates and accurately states the POST_BUILD audit exception; `p84f` documents host-attention dispatch and the DL decline path; `0v28` documents RAM-aware compiler parallelism and observed NAS compiler crashes without asserting an unproved compiler fix; `0su3` records final measured auto-ubatch/cache outcomes with immutable evidence and limitations.

```bash
git diff --check -- CLAUDE.md docs/backend/sycl-env-vars.md docs/build.md .devops/intel.Dockerfile
```

Also validate documented option/env literals against implementation and the environment catalog's inventory instructions, and verify changed links and acceptance statements. Documentation-only work uses static validators; no fabricated product runtime RED requirement. Do not remove unrelated documentation or duplicate drifting performance numbers.
**Done:** exact ticket requirements reflected accurately, no claimed unrun checks, independent SPEC/QUALITY. Documentation is not a substitute for B/C runtime acceptance.

## 4. Evidence, integration and closure contract

For every behavioral fix, preserve the existing dirty candidate first, then freeze commit/config/model/reproducer identities. Commit a durable regression artifact and observe executable RED on the exact pre-fix candidate before new production edits. Replay that scenario for GREEN, then focused/broader risk-relevant gates on one immutable feature candidate. For already-written fixes, recover historical RED with provenance or reproduce it in a safe isolated historical checkout; never undo protected work or invent a past run. Mark incomplete current proof as HYPOTHESIS.

Owner handoff includes base/final SHAs, scoped diff, exact commands, fixture counts, configuration/model digest, stdout/stderr/exit codes, baseline failures/skips, docs changes and remaining risks. Source checks and builds are necessary where relevant but cannot substitute for exercising changed runtime behavior. A zero-test selector is failure of validation, not GREEN.

Fresh independent SPEC then QUALITY reviews inspect the complete feature and its evidence. Ordinary blocking FAIL needs executable current-candidate reproduction; static-only concerns are non-blocking. A narrow safety/security/data-loss substitute requires concrete unsafe path and explicit approval. After two failed re-reviews or a new blocker class, rescope/triage rather than endlessly micro-fix. Send findings only to the retained owner, never a reused reviewer. Existing approvals can be reused as historical coverage, not approval of dirty or conflict-resolution deltas.

Merge reviewed commits serially. Validate the changed seams after merge; qualify the final aggregate candidate once, rerunning affected gates if the SHA/config changes. No automatic closing of related tickets just because a similar ticket merged.

### Final lead-only qualification

1. Freeze aggregate SHA and clean owned paths; enumerate included ticket commits and exact runtime/build environments. Use disposable, recorded tuning-cache locations to avoid stale hits or poisoning normal caches.
2. Reconfigure/build the real SYCL target; prove CTest registration/selection is non-vacuous; run required host gates with known baseline failures explicitly attributed.
3. Run the recovered/committed Gemma4 default/four-slot overlapping-request scenarios, both KV modes, single-slot controls; GPT-OSS with the actual server template; Mistral correctness; auto-ubatch cold/warm/cache-boundary behavior; allocation/Qwen local and NAS evidence; fkpg FA/non-FA shapes after its contract gate. Record timeouts, warnings, outputs, memory and workload identity. Compare applicable performance gates in `docs/backend/sycl-perf-baselines.md` under idle host/device conditions; do not duplicate broad benchmarks for every intermediate commit.
4. **`mzvn` DL proof** on that integrated source SHA in an isolated build directory, without launching Docker merely to test Docker flags:

```bash
source /opt/intel/oneapi/setvars.sh --force
./scripts/sycl-dockerfile-configure-check.sh .
cmake -S . -B build-dl -G Ninja \
  -DGGML_NATIVE=OFF -DGGML_SYCL=ON \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
  -DLLAMA_BUILD_TESTS=OFF -DGGML_SYCL_F16=ON
ninja -C build-dl -j6
```

Reserve the directory first; use another recorded directory if `build-dl` already belongs to someone else. Require the POST_BUILD RTLD_NOW/dependency/seam audit to execute and pass: no forbidden imports and no ggml-cpu NEEDED dependency. A historical task-branch audit is insufficient.
5. After ordinary owners are integrated and cleaned up, use a fresh independent final reviewer for cross-feature seams/release consistency, not redundant re-review of local steps. New confirmed findings get a new/reopened tracker fix task, fresh owner and the same review/cleanup lifecycle.

**Closure order:** review PASS -> reviewed merge and required qualification -> close the feature -> terminate/remove its implementer -> verify process/member removal -> remove only clean, proved-merged authorized task worktree/branch -> prune -> `task_ready`. Reviewed but dirty/unmerged worktrees cannot be removed. Pre-existing unrelated artifacts/worktrees remain untouched. If external acceptance is outstanding, report the feature blocked, not complete.

At team teardown verify no feature-owned live processes/members or temporary resources remain; preserve evidence before authorized cleanup. Teardown acknowledgements alone are not proof. Use supported operation/status/events APIs if the installed team tool surface lacks `team_await`; do not invent a tool or busy-poll. Reset leases for work no longer actively owned, with a concrete handoff. Push only after owner confirmation, then verify remote ancestry/status; otherwise report committed but unpushed work plainly.

## 5. Explicit exclusions and approval holds

- Older `sbky`, `hhbb`, `qmen`, `dyi3`, `gwno` are not silently included in this recent-work campaign. `dyi3` may be unblocked by the slot fix but its decode replay/performance scope remains separate. No closure/status changes to those tickets in planning.
- No backlog-wide optimization, new model support, driver changes, system configuration changes or unrelated test expansion.
- **Before all-work completion can be promised:** recover exact runtime acceptance artifacts, obtain the NAS model/reproducer and candidate-matched rerun, establish executable pyu4 G1 coverage, and approve fkpg's evidenced transport/units/model-reuse contract. These are explicit holds, not GREEN results.
- Proposed approval is for the recovered outcomes, bounded ordering, ownership, Astra-low execution/reviews and safety policy. It is not approval of an unspecified public API change, new top-level test file, push, release or remote NAS access.

**Observed during planning:** read-only tracker/source/worktree inspection; Astra-low configuration verified. **Not run:** builds, source suites, CTest, GPU/server/NAS acceptance or DL loading. Plan validation is documentation-only. The execution lead fills real results into the existing tracker records, not this plan's prose.
