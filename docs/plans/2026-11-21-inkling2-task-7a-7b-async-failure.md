# Tasks 7a, 7b and 7c: asynchronous SYCL errors fail closed instead of terminating (llama.cpp-kjoc)

Part of `docs/plans/2026-11-21-inkling2-mini-sycl-enable-profile-optimize.md`. Every file:line below is at
`7809ffac8` (branch `feat/inkling2`). Re-find each site by its text before editing (`cat FILE | grep -n 'TEXT'`; codescout is
blind inside `ggml-sycl.cpp`).

## Decision record

### What the code does today

- **The execution queue has no handler.** `ensure_single_device_context_queue` (`unified-cache.cpp:17477-17490`) builds
  `sycl::context ctx(dev)` and `new sycl::queue(ctx, dev, default_queue_properties())`, neither with an
  `async_handler`. Every backend context's `stream()` resolves to it (`common.hpp:6227-6235`, `:6532-6547`), so every
  model's decode, and the Q1/NVFP4 route's `exact_queue` (`ggml-sycl.cpp:87079`), runs there. When an asynchronous error
  reaches it, the runtime's default handler prints `Default async_handler caught exceptions:` and calls `std::terminate`
  (oracle: `docs/backend/data/inkling2/t1-oracle-b70.txt:414-416`, `:494`).
- **22 sites in non-test backend sources construct a queue or context directly** (list in Task 7b). There is no shared
  constructor, and none passes a ggml handler. `queue(dev, props)` delegates to `detail::defaultAsyncHandler`
  (`sycl/queue.hpp:316-317`). The dpct per-device queues (`dpct/helper.hpp:880-905`) use `dpct::exception_handler`
  (`helper.hpp:149-164`), which prints a `sycl::exception` and drops it.
- **A throwing drain in synchronize aborts.** `ggml_backend_sycl_synchronize` (`ggml-sycl.cpp:97097-97179`) runs
  `CHECK_TRY_ERROR(...wait_and_throw())`, which returns 999 on a throw, prints `[SYCL-SYNC-DIAG] DEVICE_LOST`, then calls
  `SYCL_CHECK(err)`, which ends in `GGML_ABORT` (`common.hpp:899-909`, `:967-985`).
- **graph_compute already fails closed on a synchronous throw.** `ggml_backend_sycl_graph_compute` (`:122580-122729`)
  catches it, runs `ggml_backend_sycl_graph_boundary_exception_cleanup` (`:122545-122578`: quarantine, drain, recover
  exact-MMID workspaces, release the graph, reset recording state, log `[SYCL] <stage> failed: <what>`) and returns
  `GGML_STATUS_FAILED`.
- **The route's terminal.** `moe_direct_event_terminal::ready()` (`:79165-79185`) calls `event->wait_and_throw()` on a
  complete event, and a throw marks it failed. The route's terminal host task (`:87190-87201`) recycles the workspace
  only when `observe_success()` is true.
- **The seventh oracle case** (`tests/test-q1-nvfp4-admitted-device.cpp:494-514`) injects a host task that throws after
  the route's final copy (`ggml-sycl.cpp:87177-87187`). It asserts that `graph_compute` returns SUCCESS, that synchronize
  returns, and the counters (`failure_consumed +1`, `submit +1`, `quarantine +1`, `recycle +0`). Today it terminates.
- **The retained-graph terminal does not check for failure.** `moe_sycl_event_terminal::ready()` (`:109444-109455`)
  treats `command_execution_status == complete` as success, so a failed replay can recycle its retained handles.
- **ggml status flow.** `ggml_backend_synchronize` returns void (`ggml-backend.cpp:1324-1331`).
  `ggml_backend_graph_compute` returns the async status (`:1354-1358`). The scheduler returns the first non-success
  status (`:2974-2999`). `llama_context::decode` maps `GGML_STATUS_FAILED` to `-3` (`src/llama-context.cpp:5852-5882`).
  `llama_get_logits_ith` synchronizes, a void call, and then returns the logits (`:8229-8240`).

### Measured runtime facts

Source: the lead's standalone host-task probe, run once on the B50 (`level_zero:1`), compute-runtime 26.44, oneAPI
2026.2. Log: `docs/backend/data/inkling2/t7a-runtime-probe-b50.txt`; source: `t7a-runtime-probe.cpp` beside it. `test-async-handler-delivery.cpp` (Task 7a) pins (a) to (e) against a runtime
upgrade.

| | Measured |
|---|---|
| (a) | Handler on both queue and context. `queue::wait_and_throw` returns, and the error goes to the **queue's** handler (queue +1, context +0). |
| (b) | `event::wait_and_throw` on the failed event returns. The error also goes to the **queue's** handler, once: a later `queue::wait_and_throw` delivers nothing more. |
| (c) | A queue on the platform default context, with a handler on the queue only. Both waits reach the queue's handler. |
| (d) | A host task that `depends_on` a failed event **runs**. Its `failed.wait_and_throw()` does not throw, and it sees the failure in the latch. A failed command does not cancel its dependents. |
| (e) | A queue **without** its own handler, in a context that has one. Both waits deliver to the **context's** handler, so the context handler is the fallback for any queue later built on it. |
| (f) | A handled queue destroyed with an undelivered error (only a plain `wait()` before the destructor). The error is dropped silently: no delivery, no terminate. |
| (g) | A plain `event.wait()` on a failed event neither throws nor delivers. The error stays pending until a `wait_and_throw`/`throw_asynchronous` on that queue. |
| CPU | This host HAS a CPU SYCL device: `Intel(R) OpenCL`, registered by `/etc/OpenCL/vendors/intel64.icd` (`libintelocl.so`). Corrected 2026-11-21: this row first said there was none, inferred from the absence of a `libur_adapter_opencl*.so` file in `/opt/intel/oneapi/compiler/2026.2/lib/`. The Task 7a implementer's host-only run of the delivery probe selected that device (`SYCL_UR_TRACE=1`: `platform: Intel(R) OpenCL`) and printed `async handler delivery: PASS`. Facts (a) to (e) therefore hold on the OpenCL CPU device as well as on the B50. |

The header comment at `sycl/event.hpp:101-106` says `event::wait_and_throw` calls the context's handler. That is wrong
for this runtime: (b) shows it uses the queue's handler, falling back to the context's (e).

### Decisions

1. **Design.** A recording-only handler on every backend queue and context feeds a per-device latch. The graph
   boundaries read the latch. The handler never rethrows, logs, locks or allocates. It can run on a decode thread, on a
   SYCL host-task worker, on a thread holding the workspace registry or pool mutexes (`recover_quarantined` ->
   `terminal->ready()` -> `event->wait_and_throw()`), and during static destruction.
2. **Granularity.** One slot per backend device index, plus one shared slot for queues bound to no single device (the
   dpct per-device queues, the CPU-offload queue, the multi-GPU control context). A failure on a device fails every
   context on it. A failure on the shared slot fails every device.
3. **Reset point.** Counts never reset. A backend context takes a cursor at construction and is failed for life once
   either of its counts moves. A new `ggml_backend_sycl_init` starts clean.
4. **A throwing drain in synchronize becomes a latched failure instead of an abort.** The `[SYCL-SYNC-DIAG]` line
   carries the drain's own exception text, because the latch's first message is the process's first and may be an
   older one.
5. **graph_compute** refuses, before any submission, every graph of a context that has seen a failure.
   **synchronize** of such a context reports and runs the boundary cleanup on its first failed call, then returns. Both
   tasks add no host wait.
6. **Terminals.** The direct route's terminal (7a) and the retained-graph terminal (7b) never report success once the
   latch has moved past a cursor taken before their first submission.
7. **Last drain before a queue is destroyed.** Where a queue is destroyed before process exit and its last drain is a
   plain `wait()`, that wait becomes `wait_and_throw()` (7b). Where it has no drain at all, the site is listed under
   "Known limits" instead of getting a new wait. The other plain drains of every non-execution queue become
   `wait_and_throw()` too (7c), so that a process-lifetime queue delivers before exit.
8. **Oracle.** The seventh case also asserts that the next graph on that context is refused and never reaches the route.
9. **Gates.** One Mistral B70 completion gate and one GPT-OSS B50 chat gate accept each of 7a, 7b and 7c. There is no
   bench A/B inside these tasks: the Stage 1 exit (Task 9) runs an interleaved A/B of the stage tip against the commit
   before Task 2. The execution queue's context is already an explicit single-device context (`unified-cache.cpp:17486`), so
   7a adds a handler and changes no context identity.

### Known limits and follow-ups

The lead filed nothing in the tracker yet (the tracker tools are down in this session); the follow-ups are listed
here, in the plan's "Known limits".


- **The one-token window.** The failure of graph N is first seen in the void synchronize. `llama_get_logits_ith` then
  returns graph N's logits, and one token sampled from failed work can reach the caller before graph N+1 returns
  FAILED (`llama_decode` -3). The `[SYCL-ASYNC-FAILURE]` line and one paragraph in `docs/backend/SYCL.md` say so.
  Follow-up, two candidate designs: a status-returning synchronize, or the backend poisoning a failed context's pending
  readbacks.
- **Failed-terminal bundles stay quarantined, with their handles, until process exit.** `retire` keeps a registry
  context that has a quarantined slot (`moe-mmid-workspace.cpp:1656-1690`). Follow-up: release them at model unload
  once the queue is drained.
- **Granularity is coarse** (decision 2): one failing context fails its neighbours on the same device.
- **A context does not read the slot of another device it submits to.** A context's cursor covers its own device's
  slot and the shared slot (decisions 2 and 3). Work it puts on another device's execution queue, which the two-card
  paths do (`GGML_SYCL_MOE_MULTI_GPU`, TP, split, the secondary prestage), latches that device's slot, and this
  context's boundaries never see it. For that reason Task 7c leaves the waits on other devices' execution queues
  plain (its rule X). Follow-up: a context that submits to another device's queue takes a cursor on that device's
  slot too, and its boundaries read both.
- **Queues destroyed with no drain at all drop their undelivered errors** (fact f). No wait is added for them,
  because the fork adds no host waits (decision 7). Follow-up: each needs its work chained to an event that some
  boundary already waits on.

  | Queue | Built | Destroyed | Work it carries |
  |-------|-------|-----------|-----------------|
  | `control_queue` | `ggml-sycl.cpp:34794` | end of `ggml_sycl_init_cross_device_control_tensor` | Nothing is submitted to it directly. It is the `alloc_request` queue for the host-pinned control `unified_alloc` (`:34796-34800`). Whether that allocation path submits anything on `req.queue` was not established. |
  | `ccs` (`GGML_SYCL_PRIVATE_TESTING`) | `:24803` | end of scope: the early returns at `:24812` and `:24816` leave before the drain at `:24818`, which 7b makes a `wait_and_throw` | the test's expert-refill commands |
  | split secondary, `g_split_secondary_queue_owner` | `:73797` (it becomes `g_split_config.queue[1]` and the secondary device's registered cache queue, `:73799-73813`) | `split_config_shutdown_reset` `:73724`; also reset at `:73832`, right after construction, with nothing submitted | the secondary device's rows of a split matmul |
  | split coordinator, `g_split_coord_queue_owner` | `:73598-73600` | `split_config_shutdown_reset` `:73722` | the persistent kernel's merge-progress copies (`:120480`, `:120572`) |
  | MoE stream timing queue, `timing_queue_` | `moe-stream-plan.cpp:285` | reset at `:419` (`prepare` on a new compute queue) and `:798` (test seam), and destroyed with the phase (`~moe_stream_phase`, `:55-59`) | the stream window's timing marks, markers that wait on the window's copies (`:310`, `:328`) |

  Both split queues are also drained by the replan with `wait_and_throw` after Task 7c. That drain runs only when a
  replan happens, not before the reset.

---

### Task 7a: An asynchronous error on the execution queue fails the next graph boundary instead of terminating

**Track:** L-ordered. One implementer makes the host-side edits and runs the host tests. The lead runs the GPU checks.
**Depends on:** Task 1 (its oracle logs are the GPU RED). Blocks Task 7.
**File scope:**
- Create: `ggml/src/ggml-sycl/async-failure.hpp`
- Modify: `ggml/src/ggml-sycl/common.cpp` (append at end of file)
- Modify: `ggml/src/ggml-sycl/common.hpp`: include block `:16-41`; backend context `:6514-6520`
- Modify: `ggml/src/ggml-sycl/unified-cache.cpp:17485-17487` (`ensure_single_device_context_queue`)
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp`:
  - `:52659-52663` ("A FAILED DRAIN" comment)
  - `:79163-79206` (`moe_direct_event_terminal`)
  - `:87079` (cursor)
  - `:87189` (`arm`)
  - `:87212-87215`, `:87221-87223` (catch comments)
  - `:97097-97179` (`ggml_backend_sycl_synchronize`)
  - `:122580-122584` (`ggml_backend_sycl_graph_compute` entry)
- Modify: `ggml/src/ggml-sycl/tests/test-q1-nvfp4-admitted-device.cpp:95-99` (comment), `:513-514` (case 7)
- Create: `ggml/src/ggml-sycl/tests/test-async-failure-latch.cpp`, `ggml/src/ggml-sycl/tests/test-async-handler-delivery.cpp`
- Modify: `ggml/src/ggml-sycl/CMakeLists.txt` (after the `test-attn-host-pool` block, `:2363-2398`)
- Create: `tests/test-sycl-async-failure-source.py`; Modify: `tests/CMakeLists.txt` (after `:1011-1015`)
- Modify: `docs/backend/SYCL.md:1143` (the empty `## Known Issues` section)

**Description:** The execution queue and its context carry no `async_handler`. An asynchronous error there reaches the
runtime's default handler, which terminates the process, and the oracle's seventh case does exactly that. This task:
- adds the latch and the handler;
- installs the handler on the execution queue and its context;
- makes `synchronize` report and clean up instead of aborting, for a delivered error and for a drain that throws;
- makes `graph_compute` refuse every graph of a context whose device failed after the context was created;
- makes the route's terminal deny success after a failure;
- extends the oracle's seventh case to assert fail-closed.

Two ordered commits: commit 1 is the latch, the helper and their tests; commit 2 is the behaviour.

**Acceptance Criteria:**
- [ ] Commit 1: `sycl-async-failure-latch` passes (host-only).
- [ ] Commit 1: `sycl-async-handler-delivery` passes under ctest on the OpenCL CPU device (host tasks only). It
      reports **Skipped** (77) only where no CPU SYCL device is visible, and a skip is not evidence. (Corrected
      2026-11-21: this line first expected a certain skip on this host.)
- [ ] Commit 2: `test-sycl-async-failure-source` passes with `--self-test`, so each mutant is caught (host-only).
- [ ] Commit 2: full `./scripts/sycl-build.sh` (all targets) succeeds.
- [ ] Lead, once, on the B50: `test-sycl-async-handler-delivery --gpu` prints `async handler delivery: PASS`, rc 0.
- [ ] Lead, once per card. The oracle ends with `Q1/NVFP4 scoped production-route lifecycle: PASS` on the B70 (ctest)
      and rc 0 on the B50 (binary). Neither log contains `Default async_handler` or `terminate called`. The B70 log
      contains `[SYCL-ASYNC-FAILURE]`.
- [ ] Lead, once each: the Mistral B70 completion gate and the GPT-OSS B50 chat gate (CLAUDE.md commands) pass.
- [ ] No text in `ggml/src/ggml-sycl/` says the execution queue has no handler, or that synchronize swallows a throwing
      drain.

**Implementation Guide:**

#### Commit 1: the latch and the helper

1. **Tests (RED).**

`ggml/src/ggml-sycl/tests/test-async-failure-latch.cpp` (host-only; it enumerates no device):

```cpp
// The per-device asynchronous-failure latch (async-failure.hpp) that turns a delivered SYCL async error into a failed
// graph boundary. Host-only: it calls the latch's record and read functions directly; no device is enumerated.
#include "../async-failure.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

// The build is Release (-DNDEBUG): an explicit check, never assert().
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

int main() {
    // 1. A cursor taken before any failure sees none.
    const ggml_sycl_async_failure_cursor dev0 = ggml_sycl_async_failure_snapshot(0);
    CHECK(!ggml_sycl_async_failure_since(dev0), "case 1: a fresh cursor reports a failure");

    // 2. A failure on device 0 fails device 0's cursor and not device 1's.
    const ggml_sycl_async_failure_cursor dev1 = ggml_sycl_async_failure_snapshot(1);
    ggml_sycl_async_failure_record(0, "first failure");
    CHECK(ggml_sycl_async_failure_since(dev0), "case 2: device 0's cursor missed its failure");
    CHECK(!ggml_sycl_async_failure_since(dev1), "case 2: a device 0 failure failed device 1");
    const char * first = ggml_sycl_async_failure_first_message(0);
    CHECK(first && std::strcmp(first, "first failure") == 0, "case 2: the first message was not kept");

    // 3. Sticky, and the first message stays the first.
    ggml_sycl_async_failure_record(0, "second failure");
    CHECK(ggml_sycl_async_failure_since(dev0), "case 3: the failure is not sticky");
    CHECK(std::strcmp(ggml_sycl_async_failure_first_message(0), "first failure") == 0, "case 3: first message replaced");
    CHECK(ggml_sycl_async_failure_count(0) == 2, "case 3: count is not 2");

    // 4. A cursor taken after the failure (a context created later) starts clean.
    const ggml_sycl_async_failure_cursor later = ggml_sycl_async_failure_snapshot(0);
    CHECK(!ggml_sycl_async_failure_since(later), "case 4: a later cursor inherited an earlier failure");

    // 5. A failure on the shared slot (a queue bound to no device) fails every device's cursor.
    const ggml_sycl_async_failure_cursor dev1b = ggml_sycl_async_failure_snapshot(1);
    ggml_sycl_async_failure_record(GGML_SYCL_ASYNC_FAILURE_SLOT_SHARED, "shared failure");
    CHECK(ggml_sycl_async_failure_since(dev1b) && ggml_sycl_async_failure_since(later),
          "case 5: a shared failure did not fail every device");

    // 6. Out-of-range slots go to the shared slot, never out of bounds.
    const uint64_t shared_before = ggml_sycl_async_failure_count(GGML_SYCL_ASYNC_FAILURE_SLOT_SHARED);
    ggml_sycl_async_failure_record(-1, "negative slot");
    ggml_sycl_async_failure_record(GGML_SYCL_MAX_DEVICES + 5, "slot past the end");
    CHECK(ggml_sycl_async_failure_count(GGML_SYCL_ASYNC_FAILURE_SLOT_SHARED) == shared_before + 2,
          "case 6: out-of-range slots were not counted on the shared slot");

    // 7. The handler runs on any thread that waits: concurrent records are all counted; null and long messages are safe.
    const uint64_t    c7 = ggml_sycl_async_failure_count(7);
    const std::string long_message(4096, 'x');
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&long_message, t] {
            for (int i = 0; i < 1000; ++i) {
                ggml_sycl_async_failure_record(7, (i + t) % 2 ? nullptr : long_message.c_str());
            }
        });
    }
    for (auto & thread : threads) {
        thread.join();
    }
    CHECK(ggml_sycl_async_failure_count(7) == c7 + 8000, "case 7: concurrent records were lost");
    const char * m7 = ggml_sycl_async_failure_first_message(7);
    CHECK(m7 && std::strlen(m7) < 256, "case 7: the first message is missing or unbounded");

    std::puts("async failure latch: PASS");
    return 0;
}
```

`ggml/src/ggml-sycl/tests/test-async-handler-delivery.cpp` (real runtime; a CPU device by default, `--gpu` for the lead):

```cpp
// Pins the runtime facts the asynchronous-failure latch rests on, so a runtime upgrade that changes them fails here:
// (a) queue::wait_and_throw delivers to the handler of the queue the command ran on, not its context's, and returns;
// (b) event::wait_and_throw on the failed event also goes to the queue's handler, once;
// (c) a queue on the platform default context with a handler on the queue alone receives both;
// (d) a host task that depends on a failed event still runs, and sees the failure in the latch (the route's terminal
//     relies on this);
// (e) a queue built without a handler on a context that has one delivers to the context's handler.
// Host tasks only: no kernel, no allocation. Without --gpu it needs a CPU SYCL device and skips when there is none.
#include "../../../../tests/test-skip.h"  // LLAMA_TEST_EXIT_SKIP
#include "../async-failure.hpp"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <sycl/sycl.hpp>

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

namespace {
// Distinct latch slots tell the queue's handler from the context's. The probe runs alone in its process, so device
// slots are free to borrow.
constexpr int k_queue_slot       = 0;
constexpr int k_context_slot     = 1;
constexpr int k_default_ctx_slot = 2;

sycl::event submit_throw(sycl::queue & q, const char * what) {
    return q.submit([what](sycl::handler & h) { h.host_task([what] { throw std::runtime_error(what); }); });
}

bool waits_without_throwing(sycl::queue & q) {
    try {
        q.wait_and_throw();
        return true;
    } catch (...) {
        return false;
    }
}

bool waits_without_throwing(sycl::event & e) {
    try {
        e.wait_and_throw();
        return true;
    } catch (...) {
        return false;
    }
}

uint64_t count(int slot) {
    return ggml_sycl_async_failure_count(slot);
}
}  // namespace

int main(int argc, char ** argv) {
    const bool                  gpu = argc > 1 && std::strcmp(argv[1], "--gpu") == 0;
    std::optional<sycl::device> dev;
    try {
        dev = gpu ? sycl::device{ sycl::gpu_selector_v } : sycl::device{ sycl::cpu_selector_v };
    } catch (const sycl::exception & e) {
        std::fprintf(stderr, "SKIP: no %s SYCL device: %s\n", gpu ? "GPU" : "CPU", e.what());
        return LLAMA_TEST_EXIT_SKIP;
    }
    const sycl::property_list in_order{ sycl::property::queue::in_order{} };
    sycl::context             ctx = ggml_sycl_make_context(*dev, k_context_slot);
    sycl::queue               q   = ggml_sycl_make_queue(ctx, *dev, k_queue_slot, in_order);

    // (a)
    uint64_t nq = count(k_queue_slot), nc = count(k_context_slot);
    submit_throw(q, "probe (a)");
    CHECK(waits_without_throwing(q), "(a) queue::wait_and_throw threw: the handler must record, not rethrow");
    CHECK(count(k_queue_slot) == nq + 1 && count(k_context_slot) == nc,
          "(a) queue::wait_and_throw did not deliver to the queue's handler alone");

    // (b)
    nq              = count(k_queue_slot);
    nc              = count(k_context_slot);
    sycl::event bad = submit_throw(q, "probe (b)");
    CHECK(waits_without_throwing(bad), "(b) event::wait_and_throw threw");
    CHECK(count(k_queue_slot) == nq + 1 && count(k_context_slot) == nc,
          "(b) event::wait_and_throw did not deliver to the queue's handler alone");
    CHECK(waits_without_throwing(q) && count(k_queue_slot) == nq + 1, "(b) the error was delivered twice");

    // (c)
    sycl::queue qd = ggml_sycl_make_queue(*dev, k_default_ctx_slot, in_order);
    uint64_t    nd = count(k_default_ctx_slot);
    sycl::event bd = submit_throw(qd, "probe (c) event wait");
    CHECK(waits_without_throwing(bd) && count(k_default_ctx_slot) == nd + 1,
          "(c) event::wait_and_throw on a default-context queue missed the queue's handler");
    submit_throw(qd, "probe (c) queue wait");
    CHECK(waits_without_throwing(qd) && count(k_default_ctx_slot) == nd + 2,
          "(c) queue::wait_and_throw on a default-context queue missed the queue's handler");

    // (d)
    nq                    = count(k_queue_slot);
    sycl::event      failed = submit_throw(q, "probe (d)");
    std::atomic<int> saw{ -1 };  // -1 never ran, 0 ran but latch unmoved, 1 saw the failure, 2 the wait threw
    q.submit([&saw, failed, nq](sycl::handler & h) {
        h.depends_on(failed);
        h.host_task([&saw, failed, nq]() mutable {
            bool threw = false;
            try {
                failed.wait_and_throw();
            } catch (...) {
                threw = true;
            }
            saw.store(threw ? 2 : (count(k_queue_slot) != nq ? 1 : 0), std::memory_order_release);
        });
    });
    q.wait();  // a plain wait delivers nothing (fact g); the host task above did the delivery
    CHECK(saw.load(std::memory_order_acquire) == 1,
          "(d) a host task depending on a failed event did not run, threw, or did not see the failure in the latch");
    CHECK(waits_without_throwing(q), "(d) final drain threw");

    // (e)
    sycl::queue qe(ctx, *dev, in_order);  // no handler of its own: the context's is the fallback
    nq = count(k_queue_slot);
    nc = count(k_context_slot);
    submit_throw(qe, "probe (e) queue wait");
    CHECK(waits_without_throwing(qe) && count(k_context_slot) == nc + 1 && count(k_queue_slot) == nq,
          "(e) queue::wait_and_throw on a handlerless queue did not reach its context's handler");
    sycl::event be = submit_throw(qe, "probe (e) event wait");
    CHECK(waits_without_throwing(be) && count(k_context_slot) == nc + 2,
          "(e) event::wait_and_throw on a handlerless queue did not reach its context's handler");

    std::puts("async handler delivery: PASS");
    return 0;
}
```

Registration, in `ggml/src/ggml-sycl/CMakeLists.txt` directly after the `test-attn-host-pool` `endif()`
(`:2398`), in the same shape:

```cmake
# The asynchronous-failure latch (llama.cpp-kjoc). The latch test is host-only and enumerates no device. The delivery
# probe asks for a CPU SYCL device and skips (77) without one, which is certain on a host whose oneAPI ships only the
# Level Zero UR adapters; the lead runs it with --gpu by hand. Both link ggml-sycl for common.cpp's latch.
if (NOT GGML_BACKEND_DL)
    foreach(_t test-sycl-async-failure-latch test-sycl-async-handler-delivery)
        if (_t STREQUAL "test-sycl-async-failure-latch")
            add_executable(${_t} tests/test-async-failure-latch.cpp)
        else()
            add_executable(${_t} tests/test-async-handler-delivery.cpp)
        endif()
        target_include_directories(${_t} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/..
            ${CMAKE_CURRENT_SOURCE_DIR}/../../include
            ${SYCL_INCLUDE_DIR}
        )
        target_compile_options(${_t} PRIVATE ${SYCL_LINK_ONLY_TEST_OPTIONS})
        target_link_options(${_t} PRIVATE ${SYCL_LINK_ONLY_TEST_OPTIONS})
        target_link_libraries(${_t} PRIVATE ggml-base ggml ggml-sycl Threads::Threads)
        if (IntelSYCL_FOUND)
            target_link_libraries(${_t} PRIVATE IntelSYCL::SYCL_CXX)
        endif()
        set_target_properties(${_t} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    endforeach()
    add_test(NAME sycl-async-failure-latch COMMAND test-sycl-async-failure-latch)
    set_tests_properties(sycl-async-failure-latch PROPERTIES
        ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
        LABELS "sycl;kjoc;hostonly"
        TIMEOUT 30
    )
    add_test(NAME sycl-async-handler-delivery COMMAND test-sycl-async-handler-delivery)
    set_tests_properties(sycl-async-handler-delivery PROPERTIES
        ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib:$ENV{LD_LIBRARY_PATH}"
        LABELS "sycl;kjoc;hostonly"
        SKIP_RETURN_CODE 77
        TIMEOUT 30
    )
else()
    foreach(_n sycl-async-failure-latch sycl-async-handler-delivery)
        add_test(NAME ${_n} COMMAND ${CMAKE_COMMAND} -E echo "not built: needs -DGGML_BACKEND_DL=OFF")
        set_tests_properties(${_n} PROPERTIES DISABLED TRUE LABELS "sycl;kjoc;hostonly;not-built")
    endforeach()
endif()
```

Run in your own worktree:
```bash
./scripts/sycl-build.sh test-sycl-async-failure-latch
```
Expected RED: the build fails with `fatal error: '../async-failure.hpp' file not found`. Paste that line into your
report.

2. **Implement (GREEN).**

`ggml/src/ggml-sycl/async-failure.hpp` (new):

```cpp
#pragma once

#include "ggml-sycl.h"  // GGML_SYCL_MAX_DEVICES

#include <cstdint>
#include <sycl/sycl.hpp>

// Asynchronous SYCL errors (a host task that threw, a device fault reported after submission) reach the backend only
// through an async_handler, and only when a wait_and_throw() or throw_asynchronous() runs on the queue the command ran
// on. On this runtime both queue::wait_and_throw and event::wait_and_throw deliver to that queue's handler, a queue
// built without one falls back to its context's handler, and with neither the runtime's default handler prints and
// calls std::terminate. Each error is delivered once.
//
// A plain wait() -- on a queue or on an event -- neither throws nor delivers, so it is never a failure check. The error
// stays pending on its queue until a wait_and_throw() there, and a queue destroyed with an undelivered error drops it
// silently. So the last drain of a queue before it is destroyed is a wait_and_throw().
//
// A failed command does not cancel the commands that depend on it: they run on inputs that were never produced. That is
// why a backend context whose device latched a failure stays failed for the rest of its life (its outputs and the KV
// rows it wrote may come from that work), and why only a context created afterwards starts clean.
//
// The handler records each error in a per-device latch and does nothing else. The graph boundaries read the latch:
// ggml_backend_sycl_synchronize reports and cleans up, ggml_backend_sycl_graph_compute refuses.

// The slot of queues that serve no single backend device (dpct's per-device queues, the CPU-offload queue, the
// multi-GPU control context). A failure there fails every device.
constexpr int GGML_SYCL_ASYNC_FAILURE_SLOT_SHARED = GGML_SYCL_MAX_DEVICES;

// What a reader saw: a failure happened after the snapshot when either count has moved since.
struct ggml_sycl_async_failure_cursor {
    int      device       = -1;
    uint64_t device_count = 0;
    uint64_t shared_count = 0;
};

// Records one delivered error. Lock-free and allocation-free: it runs on whatever thread delivers the error (a decode
// thread, a SYCL host-task worker, a thread holding backend mutexes) and during static destruction.
void         ggml_sycl_async_failure_record(int slot, const char * what) noexcept;
uint64_t     ggml_sycl_async_failure_count(int slot) noexcept;
// The first message recorded on the slot since process start, or nullptr when none is readable yet.
const char * ggml_sycl_async_failure_first_message(int slot) noexcept;
ggml_sycl_async_failure_cursor ggml_sycl_async_failure_snapshot(int device) noexcept;
bool                           ggml_sycl_async_failure_since(const ggml_sycl_async_failure_cursor & cursor) noexcept;

// The handler bound to one slot, and the constructors the backend uses for its queues and contexts.
sycl::async_handler ggml_sycl_async_handler(int slot);
sycl::context       ggml_sycl_make_context(const sycl::device & dev, int slot);
sycl::queue         ggml_sycl_make_queue(const sycl::context &       ctx,
                                         const sycl::device &        dev,
                                         int                         slot,
                                         const sycl::property_list & props);
sycl::queue ggml_sycl_make_queue(const sycl::device & dev, int slot, const sycl::property_list & props);
```

In `common.hpp`, add `#include "async-failure.hpp"` to the include block, after `#include "alloc-registry.hpp"` (`:16`).

`common.cpp`, appended at the end of the file:

```cpp
// ---------------------------------------------------------------------------
// Asynchronous SYCL failure latch (async-failure.hpp)
// ---------------------------------------------------------------------------
namespace {
constexpr int    k_async_failure_slots       = GGML_SYCL_MAX_DEVICES + 1;
constexpr size_t k_async_failure_message_len = 256;

struct async_failure_slot {
    std::atomic<uint64_t> count{ 0 };
    std::atomic<int>      message_state{ 0 };  // 0 empty, 1 being written, 2 readable
    char                  message[k_async_failure_message_len] = {};
};

// Namespace scope, constant-initialized and trivially destructible: a queue drained during static destruction can still
// deliver an error here.
async_failure_slot g_async_failure[k_async_failure_slots];

int async_failure_clamp(int slot) noexcept {
    return slot >= 0 && slot < GGML_SYCL_MAX_DEVICES ? slot : GGML_SYCL_ASYNC_FAILURE_SLOT_SHARED;
}
}  // namespace

void ggml_sycl_async_failure_record(int slot, const char * what) noexcept {
    async_failure_slot & s        = g_async_failure[async_failure_clamp(slot)];
    int                  expected = 0;
    if (s.message_state.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
        const char * text = what ? what : "(no message)";
        size_t       n    = 0;
        for (; n + 1 < k_async_failure_message_len && text[n] != '\0'; ++n) {
            s.message[n] = text[n];
        }
        s.message[n] = '\0';
        s.message_state.store(2, std::memory_order_release);
    }
    s.count.fetch_add(1, std::memory_order_acq_rel);
}

uint64_t ggml_sycl_async_failure_count(int slot) noexcept {
    return g_async_failure[async_failure_clamp(slot)].count.load(std::memory_order_acquire);
}

const char * ggml_sycl_async_failure_first_message(int slot) noexcept {
    const async_failure_slot & s = g_async_failure[async_failure_clamp(slot)];
    return s.message_state.load(std::memory_order_acquire) == 2 ? s.message : nullptr;
}

ggml_sycl_async_failure_cursor ggml_sycl_async_failure_snapshot(int device) noexcept {
    return { device, ggml_sycl_async_failure_count(device),
             ggml_sycl_async_failure_count(GGML_SYCL_ASYNC_FAILURE_SLOT_SHARED) };
}

bool ggml_sycl_async_failure_since(const ggml_sycl_async_failure_cursor & cursor) noexcept {
    return ggml_sycl_async_failure_count(cursor.device) != cursor.device_count ||
           ggml_sycl_async_failure_count(GGML_SYCL_ASYNC_FAILURE_SLOT_SHARED) != cursor.shared_count;
}

sycl::async_handler ggml_sycl_async_handler(int slot) {
    const int bound = async_failure_clamp(slot);
    // Record only. Rethrowing would surface the error from whichever wait_and_throw delivers it, including ones in
    // noexcept code and in host tasks; the graph boundaries read the latch instead.
    return [bound](sycl::exception_list errors) noexcept {
        for (const std::exception_ptr & error : errors) {
            try {
                std::rethrow_exception(error);
            } catch (const std::exception & e) {
                ggml_sycl_async_failure_record(bound, e.what());
            } catch (...) {
                ggml_sycl_async_failure_record(bound, "non-standard exception");
            }
        }
    };
}

sycl::context ggml_sycl_make_context(const sycl::device & dev, int slot) {
    return sycl::context(dev, ggml_sycl_async_handler(slot));
}

sycl::queue ggml_sycl_make_queue(const sycl::context &       ctx,
                                 const sycl::device &        dev,
                                 int                         slot,
                                 const sycl::property_list & props) {
    return sycl::queue(ctx, dev, ggml_sycl_async_handler(slot), props);
}

sycl::queue ggml_sycl_make_queue(const sycl::device & dev, int slot, const sycl::property_list & props) {
    return sycl::queue(dev, ggml_sycl_async_handler(slot), props);
}
```

`common.cpp` already includes `<atomic>` through `common.hpp`. If the build says otherwise, add `#include <atomic>`.

Run:
```bash
./scripts/sycl-build.sh test-sycl-async-failure-latch test-sycl-async-handler-delivery
source /opt/intel/oneapi/setvars.sh --force
ctest --test-dir build -R '^sycl-async-(failure-latch|handler-delivery)$' --output-on-failure
```
Expected: `sycl-async-failure-latch` Passed (its output ends `async failure latch: PASS`). `sycl-async-handler-delivery`
Passed on the OpenCL CPU device (its output ends `async handler delivery: PASS`). Where no CPU SYCL device is visible
it reports `***Skipped` with `SKIP: no CPU SYCL device`, which is not a pass. The Level Zero run is the lead's `--gpu`
run. Never run it with `--gpu` yourself. The CMake comment in the block above is as first written; the committed
comment says the probe skips on a host that has no CPU SYCL device, without the claim about which adapters ship.

**Commit 1:**
```bash
git commit -F <msgfile> -- ggml/src/ggml-sycl/async-failure.hpp ggml/src/ggml-sycl/common.cpp \
  ggml/src/ggml-sycl/common.hpp ggml/src/ggml-sycl/tests/test-async-failure-latch.cpp \
  ggml/src/ggml-sycl/tests/test-async-handler-delivery.cpp ggml/src/ggml-sycl/CMakeLists.txt
```
Subject: `sycl: a recording async_handler and a per-device latch for asynchronous SYCL errors (kjoc)`. Body: what the
latch records, why the handler only records, and the probe's five pinned facts. End it with your own Co-Authored-By
trailer.

#### Commit 2: the boundaries

3. **Test (RED).** `tests/test-sycl-async-failure-source.py`:

```python
#!/usr/bin/env python3
"""Asynchronous SYCL errors fail closed at the graph boundaries (llama.cpp-kjoc).

With comments blanked:
  Q. (unified-cache.cpp ensure_single_device_context_queue) the execution queue's context and queue are built with
     ggml_sycl_make_context(dev, device) and ggml_sycl_make_queue(ctx, dev, device, ...).
  S. (ggml-sycl.cpp ggml_backend_sycl_synchronize) no SYCL_CHECK(err) and no CHECK_TRY_ERROR; a drain that throws is
     recorded with ggml_sycl_async_failure_record(sycl_ctx->device, ...); the failure branch tests
     ggml_sycl_async_failure_since(sycl_ctx->async_failure_cursor), puts the drain's own error in the
     [SYCL-SYNC-DIAG] line and calls ggml_backend_sycl_graph_boundary_exception_cleanup(sycl_ctx, "synchronize", ...)
     before the tail releases the graph.
  G. (ggml-sycl.cpp ggml_backend_sycl_graph_compute) ggml_sycl_async_failure_since( is tested and GGML_STATUS_FAILED
     returned before the first ggml_backend_sycl_graph_compute_unchecked( call.
  R. (ggml-sycl.cpp ggml_sycl_async_failure_report) the [SYCL-ASYNC-FAILURE] line says the outputs of the context's
     last graph are not valid.
  T. (ggml-sycl.cpp moe_direct_event_terminal) arm takes a ggml_sycl_async_failure_cursor; ready() tests
     ggml_sycl_async_failure_since( after event->wait_and_throw(); the route arms with a snapshot taken before its
     first submission.
  H. (common.cpp ggml_sycl_async_handler) the handler body has no throw statement, no GGML_LOG, no lock or mutex, no
     std::string and no new.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"


def blank_comments(text):
    # Line comments first: a "/*" inside one (unified-cache.cpp has `weight-reclaim/*`) would otherwise open a block
    # comment that swallows code up to the next "*/".
    text = re.sub(r"//[^\n]*", lambda m: " " * len(m.group(0)), text)
    return re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)


def body(text, signature):
    """The brace-balanced body after the first occurrence of `signature`, or '' if absent."""
    start = text.find(signature)
    if start < 0:
        return ""
    open_at = text.find("{", start)
    depth = 0
    for i in range(open_at, len(text)):
        depth += text[i] == "{"
        depth -= text[i] == "}"
        if depth == 0:
            return text[open_at:i + 1]
    return ""


def claims(cache, sycl, common):
    q = body(cache, "static sycl::queue * ensure_single_device_context_queue(int device)")
    s = body(sycl, "static void ggml_backend_sycl_synchronize(ggml_backend_t backend)")
    g = body(sycl, "static ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph)")
    r = body(sycl, "static void ggml_sycl_async_failure_report(ggml_backend_sycl_context * ctx, const char * stage)")
    t = body(sycl, "class moe_direct_event_terminal final")
    h = body(common, "sycl::async_handler ggml_sycl_async_handler(int slot)")
    g_check = g.find("ggml_sycl_async_failure_since(")
    g_first = g.find("ggml_backend_sycl_graph_compute_unchecked(")
    s_since = s.find("ggml_sycl_async_failure_since(sycl_ctx->async_failure_cursor)")
    s_clean = s.find('ggml_backend_sycl_graph_boundary_exception_cleanup(sycl_ctx, "synchronize"')
    s_release = s.find("(void) ggml_sycl_execution_release_graph(sycl_ctx);")
    return {
        "Q execution context has the handler": "ggml_sycl_make_context(dev, device)" in q,
        "Q execution queue has the handler": "ggml_sycl_make_queue(ctx, dev, device," in q,
        "S no abort on a throwing drain": s != "" and "SYCL_CHECK(err)" not in s and "CHECK_TRY_ERROR" not in s,
        "S a throwing drain is latched": "ggml_sycl_async_failure_record(sycl_ctx->device, drain_error.c_str());" in s,
        "S the diag line carries the drain's error": "[SYCL-SYNC-DIAG]" in s and "drain error: %s" in s,
        "S failure branch cleans up before the tail": 0 <= s_since < s_clean < s_release,
        "G refusal before compute": 0 <= g_check < g_first and "GGML_STATUS_FAILED" in g[g_check:g_first],
        "R the report says the last outputs are invalid": "are not valid" in r,
        "T arm takes a cursor": "void arm(sycl::event event, ggml_sycl_async_failure_cursor cursor)" in t,
        "T ready reads the latch after the wait": 0 <= t.find("event->wait_and_throw();") <
            t.find("ggml_sycl_async_failure_since(cursor)"),
        "T route arms with its cursor": "terminal->arm(final, failure_cursor);" in sycl and
            "const ggml_sycl_async_failure_cursor failure_cursor = ggml_sycl_async_failure_snapshot(ctx.device);" in sycl,
        "H handler is bounded": h != "" and not re.search(r"\bthrow\b|GGML_LOG|lock_guard|unique_lock|mutex|"
                                                      r"std::string|\bnew\b", h.replace("rethrow_exception", "")),
    }


def check(cache, sycl, common):
    return [name for name, ok in claims(cache, sycl, common).items() if not ok]


def main():
    cache = blank_comments((SYCL / "unified-cache.cpp").read_text())
    sycl = blank_comments((SYCL / "ggml-sycl.cpp").read_text())
    common = blank_comments((SYCL / "common.cpp").read_text())
    failed = check(cache, sycl, common)
    if failed:
        print("async failure source contract failed: " + ", ".join(failed), file=sys.stderr)
        return 1
    if "--self-test" in sys.argv:
        mutants = {
            "execution queue without handler": (cache.replace("ggml_sycl_make_queue(ctx, dev, device,",
                                                              "sycl::queue(ctx, dev,", 1), sycl, common),
            "synchronize aborts again": (cache, sycl.replace(
                "ggml_sycl_async_failure_record(sycl_ctx->device, drain_error.c_str());", "SYCL_CHECK(err);", 1),
                common),
            "graph_compute does not refuse": (cache, sycl.replace("ggml_sycl_async_failure_since(failed_ctx->",
                                                                  "false && (failed_ctx->", 1), common),
            "terminal ignores the latch": (cache, sycl.replace("terminal->arm(final, failure_cursor);",
                                                               "terminal->arm(final, {});", 1), common),
            "handler logs": (cache, sycl, common.replace("ggml_sycl_async_failure_record(bound, e.what());",
                                                         "GGML_LOG_ERROR(\"%s\", e.what());", 1)),
        }
        for name, args in mutants.items():
            if not check(*args):
                print(f"async failure source contract: mutant '{name}' was not caught", file=sys.stderr)
                return 1
    print("async failure source contract: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

The "graph_compute does not refuse" mutant depends on the GREEN naming its local `failed_ctx`, as below. Keep the two
in step.

Register it in `tests/CMakeLists.txt`, directly after `test-sycl-chain-same-queue-source` (`:1011-1015`) and inside
the same guard:

```cmake
    # Asynchronous SYCL errors fail closed at the graph boundaries (llama.cpp-kjoc). Run WITH --self-test so each pin
    # is proven to catch its mutant.
    add_test(NAME test-sycl-async-failure-source
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-async-failure-source.py --self-test)
    set_tests_properties(test-sycl-async-failure-source PROPERTIES
        LABELS "sycl;source;source-contract;hostonly;python" TIMEOUT 120)
```

Run: `python3 tests/test-sycl-async-failure-source.py --self-test; echo "rc=$?"`
Expected RED: `async failure source contract failed:` followed by all twelve claim names (Q, Q, S x4, G, R, T x3, H),
and `rc=1`. Both results were checked against a scratch copy before this spec was handed over: RED against
`7809ffac8`, and a pass with all five mutants caught once the GREEN blocks below were applied verbatim.

The GPU RED for the behaviour is already recorded in Task 1's logs: `docs/backend/data/inkling2/t1-oracle-b70.txt:414-416`
and `t1-oracle-b50.txt`, where the seventh case terminates.

4. **Implement (GREEN).**

`common.hpp:6514-6520`, the backend context. After `queue_ptr qptrs[...]` add:

```cpp
    // The asynchronous-failure counts when this context was created. Once either moves, this context's graph boundaries
    // fail for the rest of its life: a failed command does not cancel its dependents, so its outputs and the KV rows it
    // wrote may come from work that never ran, and the handler cannot tell whose command failed on a queue every
    // context of the device shares. A context created later starts from the counts at its creation.
    ggml_sycl_async_failure_cursor async_failure_cursor;
    bool                           async_failure_reported = false;
```

Then replace `matmul_orchestrator(*this) {}` with
`matmul_orchestrator(*this) { async_failure_cursor = ggml_sycl_async_failure_snapshot(device); }`.

`unified-cache.cpp:17485-17487`, in `ensure_single_device_context_queue`, replace the two construction lines with:

```cpp
        auto & dev = ggml_sycl_get_device(device);
        // The execution queue and its context both carry the asynchronous-failure handler: both waits deliver to the
        // queue's handler, a queue later built on this context without one falls back to the context's, and without
        // either the runtime's default handler terminates the process.
        sycl::context ctx           = ggml_sycl_make_context(dev, device);
        g_shared_ctx_queues[device] = new sycl::queue(ggml_sycl_make_queue(ctx, dev, device, default_queue_properties()));
```

In `ggml-sycl.cpp`, add the report helper directly above `ggml_backend_sycl_synchronize` (`:97097`):

```cpp
// The boundary's report of a latched asynchronous failure, once per context (synchronize runs several times per
// token). The handler itself never logs: it can run on a host-task worker or under backend mutexes.
static void ggml_sycl_async_failure_report(ggml_backend_sycl_context * ctx, const char * stage) {
    if (ctx->async_failure_reported) {
        return;
    }
    ctx->async_failure_reported = true;
    const char * device_msg     = ggml_sycl_async_failure_first_message(ctx->device);
    const char * shared_msg     = ggml_sycl_async_failure_first_message(GGML_SYCL_ASYNC_FAILURE_SLOT_SHARED);
    GGML_LOG_ERROR("[SYCL-ASYNC-FAILURE] device %d: %s: an asynchronous SYCL error was delivered after this context "
                   "was created (device errors=%llu, shared errors=%llu; first on device: %s; first shared: %s). "
                   "The outputs of this context's last graph are not valid, and this context computes nothing more.\n",
                   ctx->device, stage, (unsigned long long) ggml_sycl_async_failure_count(ctx->device),
                   (unsigned long long) ggml_sycl_async_failure_count(GGML_SYCL_ASYNC_FAILURE_SLOT_SHARED),
                   device_msg ? device_msg : "(none)", shared_msg ? shared_msg : "(none)");
}
```

In `ggml_backend_sycl_synchronize`, replace the span from
`auto err = use_deferred_decode_event ? CHECK_TRY_ERROR(...` (`:97116`) through `SYCL_CHECK(err);` (`:97148`) with:

```cpp
        std::string drain_error;
        try {
            if (use_deferred_decode_event) {
                sycl_ctx->last_graph_event->wait_and_throw();
            } else {
                stream->wait_and_throw();
            }
        } catch (const std::exception & e) {
            drain_error = e.what();
        } catch (...) {
            drain_error = "unknown exception";
        }
        ggml_sycl_cpu_job_gates_clear(gate_sleep);
        if (!drain_error.empty()) {
            // An asynchronous error goes to the queue's handler, not to this catch: a throw here is the wait itself
            // failing (a lost device, for one). Either way the queue's work cannot be trusted, so latch it like a
            // delivered error instead of aborting.
            ggml_sycl_async_failure_record(sycl_ctx->device, drain_error.c_str());
        }
        if (ggml_sycl_async_failure_since(sycl_ctx->async_failure_cursor)) {
            // The first failed synchronize of a context cleans up its graph; later ones have nothing to clean, because
            // graph_compute refuses this context from now on.
            const bool first_report = !sycl_ctx->async_failure_reported;
            if (first_report || !drain_error.empty()) {
                const int dev          = sycl_ctx->device;
                size_t    cache_used   = 0;
                size_t    cache_budget = 0;
                size_t    free_mem     = 0;
                size_t    total_mem    = 0;
                if (auto * cache = ggml_sycl::get_unified_cache_for_device(dev)) {
                    cache_used   = cache->used();
                    cache_budget = cache->budget();
                }
                try {
                    ggml_backend_sycl_get_device_memory(dev, &free_mem, &total_mem);
                } catch (...) {
                }
                // The drain's own error, not the latch's first message, which may be an older failure's.
                GGML_LOG_ERROR("[SYCL-SYNC-DIAG] failed drain on device %d: drain error: %s; cache_used=%.1f MB, "
                               "cache_budget=%.1f MB, L0_free=%.1f MB, L0_total=%.1f MB\n",
                               dev,
                               drain_error.empty() ? "none (an asynchronous error was delivered to the handler)" :
                                                     drain_error.c_str(),
                               cache_used / (1024.0 * 1024.0), cache_budget / (1024.0 * 1024.0),
                               free_mem / (1024.0 * 1024.0), total_mem / (1024.0 * 1024.0));
            }
            ggml_sycl_async_failure_report(sycl_ctx, "synchronize");
            if (first_report) {
                // Quarantines the invocation and recovers only exact-MMID workspaces whose terminals report success
                // or quiescence, so a bundle whose terminal failed stays quarantined with its handles.
                ggml_backend_sycl_graph_boundary_exception_cleanup(sycl_ctx, "synchronize",
                                                                   "asynchronous SYCL failure latched");
            }
            return;
        }
        // GGML_SYCL_MOE_EXEC_TRACE=2's per-window timing lines: read here, after the wait, so reading the windows'
        // device times never waits on the compute path. Nothing below trace level 2.
        if (sycl_ctx->moe_stream) {
            sycl_ctx->moe_stream->flush_window_timing();
        }
```

The deleted `if (err != 0) { ggml_sycl_recover_exact_mmid_after_drain(sycl_ctx); ... }` block is covered: the cleanup
calls `ggml_sycl_recover_exact_mmid_after_drain` (`:122566`), releases the graph, resets `last_graph_event` and sets
graph-compute-active false. Leave the outer `catch` blocks (`:97174-97178`) as they are.

`ggml_backend_sycl_graph_compute` (`:122580`), directly after the `GGML_SYCL_WITNESS(...)` statement (`:122584`):

```cpp
    // A context whose device delivered an asynchronous failure after the context was created computes nothing more,
    // refused before any submission, so llama_decode returns -3 instead of building on failed outputs and KV rows.
    if (auto * failed_ctx = backend ? static_cast<ggml_backend_sycl_context *>(backend->context) : nullptr;
        failed_ctx && ggml_sycl_async_failure_since(failed_ctx->async_failure_cursor)) {
        ggml_sycl_async_failure_report(failed_ctx, "graph_compute");
        return GGML_STATUS_FAILED;
    }
```

`ggml_sycl_async_failure_report` is defined above synchronize (`:97097`), which precedes this function, so it needs no
forward declaration.

`moe_direct_event_terminal` (`:79163-79206`). Replace `ready()` and `arm()`, and add the member `cursor_`:

```cpp
    bool ready() const noexcept override {
        std::optional<sycl::event>     event;
        ggml_sycl_async_failure_cursor cursor{};
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (quiescent_) return true;
            if (failed_ || !event_) return false;
            event  = event_;
            cursor = cursor_;
        }
        try {
            if (event->get_info<sycl::info::event::command_execution_status>() !=
                sycl::info::event_command_status::complete) return false;
            // Delivers a pending error to the queue's handler; it does not block, the event is complete.
            event->wait_and_throw();
            // The backend's handler records instead of throwing, so a failure shows only in the latch. Any failure on
            // this device since the route's first submission denies success: the terminal cannot tell whose it was,
            // and a failed command does not stop the commands after it.
            if (ggml_sycl_async_failure_since(cursor)) {
                std::lock_guard<std::mutex> lock(mutex_);
                failed_ = true;
                return false;
            }
            std::lock_guard<std::mutex> lock(mutex_);
            succeeded_ = true;
            return true;
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            failed_ = true;
            return false;
        }
    }
```
```cpp
    void arm(sycl::event event, ggml_sycl_async_failure_cursor cursor) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        event_  = std::move(event);
        cursor_ = cursor;
    }
```
Add the member `ggml_sycl_async_failure_cursor cursor_{};` after `std::optional<sycl::event> event_;`.

At `:87079`, after `sycl::queue * exact_queue = ctx.stream();`, add this. It sits before admission, which allows no
construction afterwards (`:87203-87206`), and the snapshot is a plain stack value:

```cpp
            // Taken before the route's first submission: a failure delivered after it denies the terminal success.
            const ggml_sycl_async_failure_cursor failure_cursor = ggml_sycl_async_failure_snapshot(ctx.device);
```

At `:87189`, replace `terminal->arm(final);` with `terminal->arm(final, failure_cursor);`.

At `:87212-87215`, replace the catch's first comment (four lines, from "wait_and_throw may report an asynchronous error
after" to "explicit quiescent terminal and recover the quarantine.") with:

```cpp
                            // A throw here is synchronous: nothing reached the device, or a submit itself failed.
                            // wait_and_throw drains the exact queue; an asynchronous error it delivers goes to the
                            // backend's handler (the latch), not to this catch, so returning means drained, and the
                            // latch fails this context at its next boundary. Only a drain that itself throws leaves
                            // the bundle quarantined for the outer boundary.
```

Replace the inner comment at `:87221-87223` (three lines, from "An asynchronously failed event is not terminal" to
"keep this bundle quarantined.") with
`// The wait itself failed: quiescence is unproven, so the bundle stays quarantined.`

At `:52659-52661` ("A FAILED DRAIN"), the paragraph's first sentence says synchronize swallows its own failure, which is
false today: the drain aborts. Replace the text from `A FAILED DRAIN: "synchronize" is an attempt:` through
`and the call returns).` with the following, and keep the rest of the paragraph from "So after a failed drain":

```cpp
        // A FAILED DRAIN: "synchronize" is an attempt. A drain that throws, or an asynchronous error any wait delivered
        // to the backend's handler, latches the device (async-failure.hpp): ggml_backend_sycl_synchronize logs, runs
        // the boundary cleanup and returns, and every later graph_compute of a context that saw it returns
        // GGML_STATUS_FAILED.
```

In the same paragraph, change "and the synchronize's own ERROR line is the only report" to "and the synchronize's
ERROR lines are the only report".

**Oracle**, `tests/test-q1-nvfp4-admitted-device.cpp`. After the last `require_stages` of
`injected_async_terminal_failure_case` (`:513-514`) add:

```cpp
    // Fail closed: the device delivered an asynchronous failure, so this context refuses its next graph before
    // anything reaches the route (no new candidate, no new submit).
    require_stages(ggml_backend_graph_compute(life.backend, c.graph) != GGML_STATUS_SUCCESS,
                   "a graph after an asynchronous failure was computed", before);
    ggml_sycl_q1_nvfp4_test_counters refused{};
    ggml_sycl_q1_nvfp4_test_counters_read(&refused);
    require_stages(refused.candidate == after.candidate && refused.submit == after.submit,
                   "the refused graph reached the route", before);
```

Replace the comment at `:95-99` with:

```cpp
// If any case terminates, an asynchronous error reached a queue with no backend handler, and the runtime's default
// handler called std::terminate (llama.cpp-kjoc). Name the running case, or the last one that ended when none is
// running, and print its stage deltas first. Then hand over to the previous handler (ggml's, which prints a
// backtrace), so the run still aborts red (rc 134).
```

**Doc**, `docs/backend/SYCL.md`. Under the empty `## Known Issues` heading (`:1143`), before
`## Runtime Allocation Policy`, insert this text, with one blank line after the heading and one before the next:

```markdown
- **An asynchronous SYCL error fails its device's contexts one graph late.** A command that fails after it was
  submitted (a device fault, a host task that throws) is recorded by the backend's async handler for its device; it no
  longer terminates the process. The backend logs `[SYCL-ASYNC-FAILURE]`, and every later `graph_compute` of every
  context on that device that existed before the failure returns `GGML_STATUS_FAILED`, so `llama_decode` returns -3.
  The failure is first seen in `ggml_backend_synchronize`, which cannot return a status, so the outputs of the graph
  that failed (its logits or embeddings, and a token sampled from them) reach the caller before the next
  `llama_decode` reports it. Treat every output read after the error line as invalid, and create a new context: a
  failed context never computes again. A context created after the failure starts clean.
```

Run:
```bash
./scripts/sycl-build.sh          # all targets: common.hpp and the context member touch every TU
source /opt/intel/oneapi/setvars.sh --force
ctest --test-dir build -R '^(sycl-async-failure-latch|sycl-async-handler-delivery|test-sycl-async-failure-source)$' --output-on-failure
clang-format-19 --dry-run -Werror ggml/src/ggml-sycl/async-failure.hpp \
  ggml/src/ggml-sycl/tests/test-async-failure-latch.cpp ggml/src/ggml-sycl/tests/test-async-handler-delivery.cpp
```
Expected: the latch test, the source contract and the delivery probe pass. For the large files you
edited (`common.cpp`, `common.hpp`, `unified-cache.cpp`, `ggml-sycl.cpp`, the oracle), format only the lines you
wrote. Never `clang-format-19 -i`. Then build the oracle so the lead can run it:
`./scripts/sycl-build.sh test-sycl-q1-nvfp4-admitted-device`. It links the PRIVATE_TESTING fixture library, which is a
second full compile of the backend sources.

5. **Verify (lead, GPU, once each, under the CLAUDE.md memory guard):**
```bash
source /opt/intel/oneapi/setvars.sh --force
ONEAPI_DEVICE_SELECTOR=level_zero:1 timeout 60 ./build/bin/test-sycl-async-handler-delivery --gpu; echo "rc=$?"
ctest --test-dir build -R '^sycl-q1-nvfp4-admitted-device$' -V > <scratch>/t7a-oracle-b70.txt 2>&1
ONEAPI_DEVICE_SELECTOR=level_zero:1 GGML_SYCL_OP_TIMEOUT_MS=110000 timeout 120 \
  ./build/bin/test-sycl-q1-nvfp4-admitted-device > <scratch>/t7a-oracle-b50.txt 2>&1; echo "rc=$?" >> <scratch>/t7a-oracle-b50.txt
cat <scratch>/t7a-oracle-b70.txt <scratch>/t7a-oracle-b50.txt | grep -c 'Default async_handler\|terminate called'   # want 0
cat <scratch>/t7a-oracle-b70.txt | grep -c 'SYCL-ASYNC-FAILURE'                                                       # want >= 1
```
Expected:
- the delivery probe prints `async handler delivery: PASS`, rc 0;
- the B70 ctest reports Passed;
- the B50 log ends `rc=0`.

Then run the Mistral B70 completion gate and the GPT-OSS B50 chat gate, verbatim from CLAUDE.md.

**Commit 2:**
```bash
git commit -F <msgfile> -- ggml/src/ggml-sycl/common.hpp ggml/src/ggml-sycl/unified-cache.cpp \
  ggml/src/ggml-sycl/ggml-sycl.cpp ggml/src/ggml-sycl/tests/test-q1-nvfp4-admitted-device.cpp \
  tests/test-sycl-async-failure-source.py tests/CMakeLists.txt docs/backend/SYCL.md
```
Subject: `sycl: an asynchronous error on the execution queue fails the next graph boundary instead of terminating (kjoc)`.
Body: the execution queue's handler, synchronize no longer aborting, the graph_compute refusal, the terminal check,
the oracle's new assertion, and the accepted one-token window. End it with your own Co-Authored-By trailer.

**Gotchas:**
- `ggml-sycl.cpp` is ~125k lines. Find each site with `cat ... | grep -n` and edit by exact string.
- Keep the handler record-only (claim H). It runs on host-task workers, and under the workspace registry and pool
  mutexes (`recover_quarantined` -> `terminal->ready()` -> `event->wait_and_throw()`).
- Keep the latch out of line in `common.cpp`. An `inline` variable in the header could become one copy per shared
  object wherever visibility hides it. `dpct/helper.hpp:853-878` explains the same interposition hazard.
- `ggml_backend_graph_compute` still returns SUCCESS for the graph that failed asynchronously
  (`ggml-backend.cpp:1354-1358`), and the seventh case asserts that. The refusal applies to the next graph.
- The first message per slot is the process's first, which can predate a context's cursor. The counts decide; the
  message is a hint. The `[SYCL-SYNC-DIAG]` line carries the drain's own error for this reason.
- In case 7 the internal synchronize inside `ggml_backend_graph_compute` reports and cleans up. The test's explicit
  `ggml_backend_synchronize` then returns quietly (`first_report` is false), and the new `graph_compute` is refused.
  That is the expected log shape: one `[SYCL-ASYNC-FAILURE] ... synchronize` line and one `[SYCL] synchronize failed`
  line.
- The failed bundle stays quarantined until process exit (an accepted limit). If the oracle's teardown prints an
  ownerless-lease warning, report it. Do not force-release it.

---

### Task 7b: Every backend queue and context carries the async-failure handler

**Track:** L-ordered. One implementer makes the host-side edits; the lead runs the gates.
**Depends on:** Task 7a merged (it adds the helper and the latch). Runs after 7a, since both edit `ggml-sycl.cpp`.
**File scope:**
- Modify: `ggml/src/ggml-sycl/dpct/helper.hpp`: top includes `:1-30`; `exception_handler` `:148-164`;
  `create_queue_impl` `:878-905`
- Modify: `ggml/src/ggml-sycl/common.cpp:177`
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp`:
  - queue and context constructions: `:24803`, `:31316-31317`, `:33539`, `:33549`, `:34726`, `:34794`, `:45971`,
    `:58660`, `:73599-73600`, `:73797-73798`, `:73820`, `:80611-80612`, `:100241-100242`, `:106138-106140`
  - drains: `:24818`, `:46123`, `:73646`, `:96756`
  - `moe_sycl_event_terminal` and its construction: `:109442-109478`, `:109592`
- Modify: `ggml/src/ggml-sycl/unified-cache.cpp:6388-6389`, `:6415-6416`, `:12727`, `:12749-12750`
- Modify: `ggml/src/ggml-sycl/expert-prefetch.cpp:57-59`, `ggml/src/ggml-sycl/moe-stream-plan.cpp:285`
- Modify: `docs/plans/data/2026-11-16-no-crash-census.md:50` (the kjoc row)
- Create: `tests/test-sycl-async-handler-source.py`; Modify: `tests/CMakeLists.txt`

**Description:** Task 7a covers the execution queue. Every other queue the backend builds can still terminate the
process on an asynchronous error: those with no handler, and those built with `queue(dev, props)`, which installs
`detail::defaultAsyncHandler` (`queue.hpp:316-317`). The dpct per-device queues print errors and drop them. This task:
- routes every construction through `ggml_sycl_make_queue` / `ggml_sycl_make_context` / `ggml_sycl_async_handler`;
- makes the last drain before each early-destroyed queue a `wait_and_throw`, so its errors reach the handler instead of
  being dropped (fact f);
- makes the retained-graph terminal deny success after a failure;
- pins all of it with a source contract.

**Acceptance Criteria:**
- [ ] `test-sycl-async-handler-source --self-test` passes, and each of its mutants fails.
- [ ] `test-sycl-async-failure-source --self-test` (7a's contract) still passes.
- [ ] `dpct::exception_handler` no longer exists, and both `create_queue_impl` overloads install
      `ggml_sycl_async_handler(GGML_SYCL_ASYNC_FAILURE_SLOT_SHARED)` on context and queue.
- [ ] Full build succeeds (`./scripts/sycl-build.sh`, all targets).
- [ ] Lead: the Mistral B70 completion gate and the GPT-OSS B50 chat gate pass once each.

**Implementation Guide:**

1. **Test (RED).** `tests/test-sycl-async-handler-source.py`:

```python
#!/usr/bin/env python3
"""Every SYCL queue and context the backend constructs carries the async-failure handler (llama.cpp-kjoc).

Without a handler an asynchronous error reaches the runtime's default handler, which calls std::terminate; dpct's old
handler printed and dropped it. ggml_sycl_make_queue / ggml_sycl_make_context (common.cpp) install
ggml_sycl_async_handler.

  C. Every construction in the backend's non-test sources -- `new sycl::queue|context`, `make_unique|make_shared<
     sycl::queue|context>`, a `sycl::queue|context(...)` or `{...}` temporary, a named `sycl::queue|context x(...)` or
     `x{...}`, and an `emplace(` on an optional queue -- has, in its statement, ggml_sycl_make_queue(,
     ggml_sycl_make_context( or ggml_sycl_async_handler(. dpct/helper.hpp's create_queue_impl may pass `eh` (claim D).
  D. dpct/helper.hpp defines no exception_handler, never assigns `eh = exception_handler`, and installs
     ggml_sycl_async_handler(GGML_SYCL_ASYNC_FAILURE_SLOT_SHARED) in both create_queue_impl overloads.
  E. moe_sycl_event_terminal takes its cursor at construction, the retained graphs' publication snapshots it per
     device before any replay is submitted, and ready() denies success once the latch moved, after a wait_and_throw
     on the complete event.
  F. The last drain of each queue destroyed before process exit is a wait_and_throw: the test's ccs queue, the shared
     readback's kernel copy, the split merge events, and the pipeline copy queues freed in ggml_backend_sycl_free.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"

CONSTRUCTIONS = [
    re.compile(r"\bnew\s+sycl::(queue|context)\b"),
    re.compile(r"\bmake_(unique|shared)\s*<\s*sycl::(queue|context)\s*>\s*\("),
    re.compile(r"\bsycl::(queue|context)\s*[({]"),
    re.compile(r"\bsycl::(queue|context)\s+[A-Za-z_]\w*\s*[({]"),
    re.compile(r"\b\w*queue_?\s*\.\s*emplace\s*\("),  # an optional queue member: timing_queue_.emplace(
]
# A function declared or defined to return a queue or context by value, not a construction.
DECLARATION = re.compile(
    r"\bsycl::(queue|context)\s+[A-Za-z_]\w*\s*\(\s*(\)|(const\s+)?[\w:<>]+\s*[&*]?\s+[A-Za-z_]\w*\s*[,)=])")
HANDLED = ("ggml_sycl_make_queue(", "ggml_sycl_make_context(", "ggml_sycl_async_handler(")


def blank_comments(text):
    # Line comments first: a "/*" inside one (unified-cache.cpp has `weight-reclaim/*`) would otherwise open a block
    # comment that swallows code up to the next "*/".
    text = re.sub(r"//[^\n]*", lambda m: " " * len(m.group(0)), text)
    return re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)


def statement(text, pos):
    end = text.find(";", pos)
    return text[pos:end if end >= 0 else len(text)]


def body(text, signature):
    start = text.find(signature)
    if start < 0:
        return ""
    open_at = text.find("{", start)
    depth = 0
    for i in range(open_at, len(text)):
        depth += text[i] == "{"
        depth -= text[i] == "}"
        if depth == 0:
            return text[open_at:i + 1]
    return ""


def violations(name, text):
    found = {}
    for pattern in CONSTRUCTIONS:
        for m in pattern.finditer(text):
            if DECLARATION.match(text, m.start()):
                continue
            stmt = statement(text, m.start())
            if any(h in stmt for h in HANDLED):
                continue
            if name == "dpct/helper.hpp" and re.search(r"\beh\b", stmt):
                continue
            line = text.count("\n", 0, m.start()) + 1
            # `new sycl::queue(...)` also matches the temporary pattern: report each line once.
            found.setdefault(line, f"{name}:{line}: {' '.join(stmt.split())[:120]}")
    return [found[line] for line in sorted(found)]


def sources():
    for path in sorted(SYCL.rglob("*")):
        if path.suffix in (".cpp", ".hpp", ".h") and "tests" not in path.relative_to(SYCL).parts:
            yield path.relative_to(SYCL).as_posix(), blank_comments(path.read_text(errors="replace"))


def semantic(files):
    texts = dict(files)
    helper = texts.get("dpct/helper.hpp", "")
    sycl = texts.get("ggml-sycl.cpp", "")
    term = body(sycl, "class moe_sycl_event_terminal final")
    t_wait = term.find("event->wait_and_throw();")
    t_since = term.find("return !ggml_sycl_async_failure_since(cursor_);")
    merge = body(sycl, "static void split_merge_drain()")
    free_fn = body(sycl, "static void ggml_backend_sycl_free(ggml_backend_t backend)")
    return {
        "D no dpct exception_handler": not re.search(r"\binline\s+auto\s+exception_handler\b", helper) and
            not re.search(r"\beh\s*=\s*exception_handler\b", helper),
        "D both overloads install the shared handler":
            helper.count("ggml_sycl_async_handler(GGML_SYCL_ASYNC_FAILURE_SLOT_SHARED)") >= 2,
        "E terminal cursor at construction":
            "explicit moe_sycl_event_terminal(ggml_sycl_async_failure_cursor cursor)" in term,
        "E ready denies after the latch moved": 0 <= t_wait < t_since,
        "E publication snapshots before replay":
            "std::make_shared<moe_sycl_event_terminal>(ggml_sycl_async_failure_snapshot(device))" in sycl,
        "F ccs drains with wait_and_throw": "ccs.wait_and_throw();" in sycl and not re.search(r"\bccs\.wait\(\);", sycl),
        "F readback kernel copy drains with wait_and_throw": not re.search(
            r"\bev\.wait\(\);\s*std::memcpy\(dst \+ offset, host_stage, chunk\);", sycl),
        "F split merge events drain with wait_and_throw": "e.wait_and_throw();" in merge and "e.wait();" not in merge,
        "F freed pipeline copy queues drain with wait_and_throw":
            "g_pipeline_copy_queue[d]->wait_and_throw();" in free_fn and "g_pipeline_copy_queue[d]->wait();" not in free_fn,
    }


def check(files):
    bad = [v for name, text in files for v in violations(name, text)]
    bad += [name for name, ok in semantic(files).items() if not ok]
    return bad


def mutate(files, name, old, new):
    out = [(n, t.replace(old, new, 1) if n == name else t) for n, t in files]
    if out == files:
        raise SystemExit(f"self-test: mutant text not found in {name}: {old!r}")
    return out


def main():
    files = list(sources())
    bad = check(files)
    if bad:
        print("async handler source contract failed:\n  " + "\n  ".join(bad), file=sys.stderr)
        return 1
    if "--self-test" in sys.argv:
        mutants = {
            "unhandled new queue": files + [("mutant.cpp", "void f(sycl::device dev) { auto * q = new sycl::queue(dev, props); }")],
            "dpct handler back": mutate(files, "dpct/helper.hpp",
                                        "ggml_sycl_async_handler(GGML_SYCL_ASYNC_FAILURE_SLOT_SHARED)", "exception_handler"),
            "terminal ignores the latch": mutate(files, "ggml-sycl.cpp",
                                                 "return !ggml_sycl_async_failure_since(cursor_);", "return true;"),
            "terminal cursor after replay": mutate(
                files, "ggml-sycl.cpp", "std::make_shared<moe_sycl_event_terminal>(ggml_sycl_async_failure_snapshot(device))",
                "std::make_shared<moe_sycl_event_terminal>(ggml_sycl_async_failure_cursor{})"),
            "plain wait before destruction": mutate(files, "ggml-sycl.cpp", "ccs.wait_and_throw();", "ccs.wait();"),
        }
        for name, mutant in mutants.items():
            if not check(mutant):
                print(f"async handler source contract: mutant '{name}' was not caught", file=sys.stderr)
                return 1
    print("async handler source contract: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Register it in `tests/CMakeLists.txt` directly after 7a's `test-sycl-async-failure-source`, with the same labels and
timeout.

Run: `python3 tests/test-sycl-async-handler-source.py --self-test; echo "rc=$?"`. Expected RED: 22 claim-C lines, one
per construction site in the table below; the dpct overloads are covered by claim D, not C. Then the nine D, E and F
claim names, and `rc=1`. This was checked before 7a, against the tree at `7809ffac8`, plus the two execution-queue lines
that 7a removes. After the table's edits, the dpct edit, the terminal edit and the four drain edits were applied to a
scratch copy, both this contract and 7a's passed with `--self-test`. Read the list against the table. A hit that is
not a construction is a scanner bug: fix the scanner and add a mutant for the fix. A construction the table lacks is a
site to convert by the same rule; name it in your report.

2. **Implement (GREEN).** One rule per site. A queue made from an existing queue, or from a known backend device
index, uses that device's slot. A queue or context that serves no single backend device uses
`GGML_SYCL_ASYNC_FAILURE_SLOT_SHARED`. Every queue keeps exactly its current properties: in order or not, profiling or
not. The comments at `unified-cache.cpp:6396-6413` and `:12714-12723` say why some queues have no profiling. A queue
on the platform default context keeps that context (fact c), because USM sharing between the TP and split queues
depends on it.

The Today column shows each expression with its whitespace collapsed. Several span two or three aligned lines in the
source, so match on the tokens.

| Site | Today | Replacement |
|------|-------|-------------|
| `common.cpp:177` | `new sycl::queue(dev, default_queue_properties())` | `new sycl::queue(ggml_sycl_make_queue(dev, dev_id, default_queue_properties()))` |
| `ggml-sycl.cpp:31316-31317` | `new sycl::queue(dev0.default_queue().get_context(), dev.default_queue().get_device(), default_queue_properties())` | `new sycl::queue(ggml_sycl_make_queue(dev0.default_queue().get_context(), dev.default_queue().get_device(), device, default_queue_properties()))` |
| `ggml-sycl.cpp:33539` | `return new sycl::queue{ cpu_dev };` | `return new sycl::queue(ggml_sycl_make_queue(cpu_dev, GGML_SYCL_ASYNC_FAILURE_SLOT_SHARED, sycl::property_list{}));` |
| `ggml-sycl.cpp:33549` | `return new sycl::queue{ dev };` | `return new sycl::queue(ggml_sycl_make_queue(dev, GGML_SYCL_ASYNC_FAILURE_SLOT_SHARED, sycl::property_list{}));` |
| `ggml-sycl.cpp:34726` | `std::make_unique<sycl::context>(devices)` | `std::make_unique<sycl::context>(devices, ggml_sycl_async_handler(GGML_SYCL_ASYNC_FAILURE_SLOT_SHARED))` (`context.hpp:160-161`) |
| `ggml-sycl.cpp:34794` | `sycl::queue control_queue(control_ctx, ctx->stream->get_device(), default_queue_properties());` | `sycl::queue control_queue = ggml_sycl_make_queue(control_ctx, ctx->stream->get_device(), ctx->device, default_queue_properties());` |
| `ggml-sycl.cpp:45971` | `exec_q = sycl::queue(q.get_context(), q.get_device(), default_queue_properties());` | `exec_q = ggml_sycl_make_queue(q.get_context(), q.get_device(), ggml_sycl_get_device_id_from_queue(q), default_queue_properties());` |
| `ggml-sycl.cpp:58660` | `new sycl::queue(dev, default_queue_properties())` | `new sycl::queue(ggml_sycl_make_queue(dev, device, default_queue_properties()))` |
| `ggml-sycl.cpp:73599-73600` | `std::make_unique<sycl::queue>(primary_queue.get_context(), primary_queue.get_device())` | `std::make_unique<sycl::queue>(ggml_sycl_make_queue(primary_queue.get_context(), primary_queue.get_device(), ggml_sycl_get_device_id_from_queue(primary_queue), sycl::property_list{}))` |
| `ggml-sycl.cpp:73797-73798` | `std::make_unique<sycl::queue>(secondary_dev, sycl::property::queue::in_order{})` | `std::make_unique<sycl::queue>(ggml_sycl_make_queue(secondary_dev, ggml_sycl_get_device_id_from_device(secondary_dev), sycl::property_list{ sycl::property::queue::in_order{} }))` |
| `ggml-sycl.cpp:73820` | `std::make_unique<sycl::queue>(primary_dev)` | `std::make_unique<sycl::queue>(ggml_sycl_make_queue(primary_dev, ggml_sycl_get_device_id_from_device(primary_dev), sycl::property_list{}))` |
| `ggml-sycl.cpp:80611-80612` | `sycl::context pri_ctx{};` / `sycl::context sec_ctx{};` | `std::optional<sycl::context> pri_ctx;` / `std::optional<sycl::context> sec_ctx;`. They are only assigned, from existing queues' contexts (`:80665-80671`), and never read. A default-constructed `sycl::context` is a real context with no handler. Comment: `// Keep the contexts the ring's buffers live in. Optional: a default-constructed sycl::context is a real context, built with no handler.` |
| `ggml-sycl.cpp:100241-100242` | `std::make_unique<sycl::queue>(base->get_context(), base->get_device(), default_queue_properties())` | `std::make_unique<sycl::queue>(ggml_sycl_make_queue(base->get_context(), base->get_device(), device, default_queue_properties()))` |
| `ggml-sycl.cpp:106138-106140` | `std::make_unique<sycl::queue>(compute_q.get_context(), compute_q.get_device(), sycl::property_list{ sycl::property::queue::in_order{} })` | `std::make_unique<sycl::queue>(ggml_sycl_make_queue(compute_q.get_context(), compute_q.get_device(), sycl_ctx->device, sycl::property_list{ sycl::property::queue::in_order{} }))` |
| `ggml-sycl.cpp:24803` (PRIVATE_TESTING) | `sycl::queue ccs(fx.queue->get_context(), fx.queue->get_device(), sycl::property::queue::in_order());` | `sycl::queue ccs = ggml_sycl_make_queue(fx.queue->get_context(), fx.queue->get_device(), fx.device, sycl::property_list{ sycl::property::queue::in_order() });` |
| `unified-cache.cpp:6388-6389`, `:6415-6416` | `std::make_unique<sycl::queue>(queue_.get_context(), queue_.get_device(), sycl::property::queue::in_order{})` | `std::make_unique<sycl::queue>(ggml_sycl_make_queue(queue_.get_context(), queue_.get_device(), ggml_sycl_get_device_id_from_queue(queue_), sycl::property_list{ sycl::property::queue::in_order{} }))` |
| `unified-cache.cpp:12727` | `std::make_unique<sycl::queue>(queue_.get_context(), queue_.get_device())` | `std::make_unique<sycl::queue>(ggml_sycl_make_queue(queue_.get_context(), queue_.get_device(), ggml_sycl_get_device_id_from_queue(queue_), sycl::property_list{}))` |
| `unified-cache.cpp:12749-12750` | `... queue_.get_device(), sycl::property::queue::in_order{})` | as for `:6388` (in order) |
| `expert-prefetch.cpp:57-59` | queue built, then `device_id_` resolved | move `device_id_ = ggml_sycl_get_device_id_from_queue(compute_q);` above, then `dma_queue_ = std::make_shared<sycl::queue>(ggml_sycl_make_queue(compute_q.get_context(), compute_q.get_device(), device_id_, sycl::property_list{}));` |
| `moe-stream-plan.cpp:285` | `timing_queue_.emplace(ccs_->get_context(), ccs_->get_device(), default_queue_properties());` | `timing_queue_.emplace(ggml_sycl_make_queue(ccs_->get_context(), ccs_->get_device(), ggml_sycl_get_device_id_from_queue(*ccs_), default_queue_properties()));` |

The cache uses `ggml_sycl_get_device_id_from_queue(queue_)` for its own device (`unified-cache.cpp:6373`, `:6379`), so
these queues name the same slot. That function and `ggml_sycl_get_device_id_from_device` fall back to the current
device for an unmapped device (`common.cpp:262-291`). That is acceptable for a device queue. Never use them for the CPU
queue, which takes the shared slot.

**`dpct/helper.hpp`.** Add `#include "../async-failure.hpp"` after `#include "../alloc-registry.hpp"`. Delete
`exception_handler` (`:148-164`, from the `/// SYCL default exception handler` comment through the lambda's `};`). In
both `create_queue_impl` overloads, replace

```cpp
        sycl::async_handler eh = {};
        if (enable_exception_handler) {
          eh = exception_handler;
        }
```
with
```cpp
        // Every dpct queue and its context carry the backend's recording handler, whatever the caller asked for:
        // without one an asynchronous error terminates the process. device_ext does not know the backend's device
        // index, so these queues latch the shared slot, which fails every device.
        (void) enable_exception_handler;
        const sycl::async_handler eh = ggml_sycl_async_handler(GGML_SYCL_ASYNC_FAILURE_SLOT_SHARED);
```
Keep the parameter: `create_queue` and its siblings (`:791-816`) pass it through. Every TU now compiles an identical
copy of these inlines, which matches the interposition rule at `:853-878`.

**Drains before destruction (fact f).** For each queue destroyed before process exit:

| Queue | Destroyed | Last drain today | Edit |
|-------|-----------|------------------|------|
| `ccs` (`ggml-sycl.cpp:24803`, PRIVATE_TESTING) | end of scope | `ccs.wait();` `:24818` | `ccs.wait_and_throw();` |
| `exec_q` (`:45971`) | end of scope | staged path: `ggml_sycl::mem_copy(..., exec_q)`, which ends in `.wait_and_throw()` on the copy's event (`mem-ops.cpp:1361`). Kernel path: `ev.wait();` `:46123` | `ev.wait_and_throw();` |
| split merge queue (`g_split_merge_queue_owner`, `:73820`) | `split_config_shutdown_reset` `:73723` | `split_merge_drain` (`:73641-73650`): `e.wait();` `:73646` on each pending merge event | `e.wait_and_throw();` (an event wait delivers to the queue's handler, fact b) |
| pipeline copy queues (`g_pipeline_copy_queue[d]`, `:31316`) | `delete` in `ggml_backend_sycl_free` `:96759` | `g_pipeline_copy_queue[d]->wait();` `:96756` | `g_pipeline_copy_queue[d]->wait_and_throw();` |
| the cache's six queues (`unified-cache.cpp:6388`, `:6415`, `:12727`, `:12749`, plus `queue_` and `compute_queue_`) | cache member destruction | `drain_all_queues_noexcept` runs `q->wait_and_throw()` on each (`unified-cache.cpp:6954-6990`) before members go | none |
| expert-prefetch `dma_queue_` | `shutdown()` `expert-prefetch.cpp:74` | `cancel_all()`: `queue->wait_and_throw()` `:401` | none |

None of these edits adds a wait. Each changes an existing wait so that it delivers. The kernel path in `exec_q` keeps
its current result: `ggml_sycl_readback_via_shared_kernel` still returns true after a delivered error. The latch fails
the device at its next boundary, which is the same accepted one-graph window.

**`moe_sycl_event_terminal`** (`:109442-109478`). One terminal per device is created when the retained graphs are
published, in `moe_graph_retention_publish` (`:109590-109597`). It is re-armed after each replay's submission, in
`moe_graph_retention_finish_replay` (`:109640-109642`). All four replay sites (`:107197-107208`, `:109992-110007`,
`:111210-111220`, `:111467-111528`) follow begin -> `graph_exec_submit` -> finish, and `arm` runs after the
submission. A cursor taken at `arm` would therefore come after the submission it guards. A cursor taken at each
replay's begin would let a failure of replay k, delivered after replay k+1 begins, be absorbed into k+1's cursor. The
point before the first submission is construction: `publish` runs after recording and before any replay of the
published graphs. Recorded sequence graphs are explicitly not submitted in their recording token (`:109981-109984`).
So the cursor is taken per device at construction:

```cpp
class moe_sycl_event_terminal final : public ggml_sycl::moe::device_terminal {
  public:
    // Taken when the retained graphs are published, before any replay of them is submitted. Once this device latches
    // a failure, a replay may have run on inputs that were never produced (a failed command does not cancel its
    // dependents), so the terminal never reports success again.
    explicit moe_sycl_event_terminal(ggml_sycl_async_failure_cursor cursor) : cursor_(cursor) {}

    bool ready() const noexcept override {
        std::optional<sycl::event> event;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!event_.has_value()) {
                return true;  // nothing was submitted under this terminal
            }
            event = event_;
        }
        try {
            if (event->get_info<sycl::info::event::command_execution_status>() !=
                sycl::info::event_command_status::complete) {
                return false;
            }
            // "complete" says the command ended, not that it succeeded; a status query never delivers an error.
            // This wait does not block (the event is complete) and delivers a pending error to the queue's handler.
            event->wait_and_throw();
        } catch (...) {
            return false;
        }
        return !ggml_sycl_async_failure_since(cursor_);
    }

    void wait() noexcept override {
        // unchanged
    }

    void arm(sycl::event event) {
        // unchanged
    }

  private:
    mutable std::mutex                   mutex_;
    std::optional<sycl::event>           event_;
    const ggml_sycl_async_failure_cursor cursor_;
};
```

At `:109592`, replace `auto terminal = std::make_shared<moe_sycl_event_terminal>();` with
`auto terminal = std::make_shared<moe_sycl_event_terminal>(ggml_sycl_async_failure_snapshot(device));`.

**Census row**, `docs/plans/data/2026-11-16-no-crash-census.md:50`. Replace the `kjoc` row with:

```markdown
| kjoc | Backend queues had no async_handler: async errors hit the SYCL default handler and terminated | P1 | Q1/NVFP4 route terminal failure injection | unclear | 2026-10-10 | n/a | fixed: every backend queue and context has the recording handler (async-failure.hpp); a delivered error fails its device's contexts at the next graph boundary (`llama_decode` -3), one graph late | |
```

Run:
```bash
python3 tests/test-sycl-async-handler-source.py --self-test     # async handler source contract: PASS
./scripts/sycl-build.sh                                           # all targets
source /opt/intel/oneapi/setvars.sh --force
ctest --test-dir build -L source-contract -j 4 --output-on-failure
ctest --test-dir build -R '^sycl-async-(failure-latch|handler-delivery)$' --output-on-failure
```

**Commit:**
```bash
git commit -F <msgfile> -- ggml/src/ggml-sycl/dpct/helper.hpp ggml/src/ggml-sycl/common.cpp \
  ggml/src/ggml-sycl/ggml-sycl.cpp ggml/src/ggml-sycl/unified-cache.cpp ggml/src/ggml-sycl/expert-prefetch.cpp \
  ggml/src/ggml-sycl/moe-stream-plan.cpp docs/plans/data/2026-11-16-no-crash-census.md \
  tests/test-sycl-async-handler-source.py tests/CMakeLists.txt
```
Subject: `sycl: every backend queue and context carries the async-failure handler (kjoc)`. End it with your own
Co-Authored-By trailer.

**Gotchas:**
- `dpct/helper.hpp` is included by every backend TU and by test fixtures. Its `device_ext` inlines now call
  `ggml_sycl_async_handler`, which is defined in `common.cpp`. A target that instantiates `dev_mgr` without linking
  ggml-sycl will fail to link. Build all targets and report any such target. Do not move the latch inline into the
  header.
- The source contract's claim C is a text scanner. If it flags a non-construction (a declaration, a cast), fix the
  scanner with a mutant for the fix. Do not reword the code to dodge it.
- Do not convert any other plain `wait()` in this task. Task 7c converts the drains of the auxiliary queues.

### Task 7c: A drain of a handled queue delivers its asynchronous errors

**Track:** L-ordered, the same track as 7a and 7b. One implementer makes the host-side edits; the lead runs the gates.
**Depends on:** Task 7b merged (every queue below gets its handler there, except the two member queues this task
builds).
**Blocks:** Task 9 (its interleaved A/B measures this task's per-graph change, listed under "Per-token sites").
**File scope:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp`:
  - cache DMA/BCS and their events: `:4951`, `:5110-5111`, `:5125-5126`, `:5356`, `:5598`, `:5864`, `:6415`, `:33138`,
    `:44171`, `:105217`, `:105233`
  - PP pipeline DMA: `:76593`, `:76610`, `:76662`, `:108039`
  - TP worker: `:58792`, `:58801`, `:58814`, `:58845`
  - pipeline copy queues: `:31402`, `:125031`
  - dpct default queues: `:51377`, `:51647`
  - replan: `:114013`, `:114016`, `:114019`, `:114033`, `:114038`, `:114045`, `:114049`, `:114057`, `:114067`
- Modify: `ggml/src/ggml-sycl/unified-cache.cpp:16568`, `:16571`
- Modify: `ggml/src/ggml-sycl/device-pool.hpp:192`, `:317`
- Modify: `ggml/src/ggml-sycl/common.cpp:2524`
- Modify: `ggml/src/ggml-sycl/layer-streaming.cpp:58`, `:353`, `:379`, `:410`, `:464`
- Modify: `ggml/src/ggml-sycl/l2-prefetch.hpp:155`, `:164`
- Modify: `ggml/src/ggml-sycl/dense-scheduler.cpp:48`, `:71`, `:113`, `:177`, `:202`, `:212`
- Create: `tests/test-sycl-async-drain-source.py`; Modify: `tests/CMakeLists.txt`

**Description:** A plain `wait()` neither throws nor delivers (fact g). After 7b every backend queue has the recording
handler, but an error on a queue whose drains are all plain still reaches the latch late or never. A process-lifetime
queue's error is held until the queue is destroyed at exit, and fact (f) says it is dropped then. This task changes
the existing drains of every non-execution queue from `wait()` to `wait_and_throw()`, so each delivers what has
accumulated so far. With the handler installed, `wait_and_throw()` does not throw for an asynchronous error: the
handler takes the list and returns, and the call returns as `wait()` does. Synchronous exceptions are thrown exactly as
they are today. The task adds no wait and removes none.

Which waits are in scope. A scan of the backend's non-test sources at `7809ffac8` finds 426 `wait(` call sites. 42 are
`std::condition_variable` waits, which take arguments. The other 384 are SYCL waits, and each falls under one of five
rules:
- **Auxiliary queue (converted).** The 50 sites in the table below. These are waits on a queue that is not a device
  execution queue, or on an event submitted to one.
- **E. Own execution queue (stays plain).** A context's synchronize ends every graph with a `wait_and_throw` on its own
  device's execution queue (`ggml-sycl.cpp:97097`ff, after 7a) or, for a deferred decode graph, on that graph's last
  event. An error on that queue reaches the latch at the end of the graph it belongs to. A load-time error on it
  reaches the latch at the end of the first graph: the accepted one-graph window. This covers the `cache->get_queue()`
  lines that sit next to converted lines (`:5109`, `:5124`, `:5136`, `:44170`, `:44172`, `:114032`,
  `unified-cache.cpp:16566`, `device-pool.hpp:187`, `:312`), the op and kernel waits on `ctx.stream()`, the KV offload
  manager (`kv-offload.cpp`, built on `*stream()` at `ggml-sycl.cpp:54467`), the sampler's buffer stream (`:51374`),
  and the secondary scatter's submissions on the primary context's stream (`:31599`, `:31683`;
  `g_pending_secondary_scatter.stream` is `ctx.stream`, `:81131`). `g_cpu_tg_direct_pending_scatter` is never
  filled, so its waits (`:31784`, `:83812`) never run.
  The deferred-decode case rests on an unmeasured fact; see "Not verified".
- **X. Another device's execution queue (stays plain; known limit).** These are:
  - the secondary-device prestage `q.wait()` (`:5420`, `b.cache->get_queue()`);
  - the MoE scatter's producer events `e.last_event.wait()` (`:31348`, `:31385`, `:31511`, `:31532`, `:31569`), which
    come from `target_queue`, the target device's execution queue (`:80515-80520`);
  - the TP row-parallel `ctx.stream(device, 0)` waits and their events (`:59523`-`:60906`);
  - the replan's loops over every device's execution and shared-context queues (`:114006`, `:114026`).

  Delivery from these would record into the other device's slot, and a context's cursor reads only its own device's
  slot and the shared slot (decisions 2 and 3). Converting them would change nothing that this context's boundaries
  read. All of them are two-card paths (`GGML_SYCL_MOE_MULTI_GPU`, TP, split, secondary prestage). See "Known limits".
- **P. Caller-passed queue in a generic helper (stays plain; not traced site by site).** These are the waits in
  `moe-sort.hpp`, `moe-xmx.hpp`, `moe-xmx-fused.hpp`, `quantized-comm.hpp`, the `mmvq.cpp`/`mmq.cpp`/`dmmv.cpp`
  staging events, `cpu-dispatch.cpp:192` (`offload_wait_queue(gpu_q)`), `dpct/helper.hpp:1920`, and the remaining
  35 queue-parameter sites. Each waits on the queue its caller passes. I did not trace every caller. The ones I read
  pass the execution queue. See "Not verified".
- **F. `GGML_SYCL_PRIVATE_TESTING` fixtures (stay plain).** These are the route and pointer-table tests on
  `ggml_sycl_get_device(0).default_queue()`: the `sycl::event::wait(stage_events)` calls at `:10395`-`:10784` and
  `:26470`-`:26784`, and their `table_event`, `entry` and barrier event waits. dpct queues latch the shared slot
  (7b), and a latched shared slot fails every device. A fixture error delivered there would show up as refused graphs
  in later, unrelated cases of the same test process, not in the fixture that caused it. The fixtures keep their own
  assertions. This is the only place where delivery would change behaviour beyond recording, and the reason they
  stay plain.

Sites that already deliver, and are unchanged: `ggml_sycl_drain_direct_stage_queue` (`:3744-3752`), the PP pipeline's
`cancel_graph_noexcept` (`:2528-2530`), the cache's `drain_all_queues_noexcept` (`unified-cache.cpp:6954-6990`),
expert prefetch `cancel_all` (`expert-prefetch.cpp:401`), every `mem_copy` (it ends in `wait_and_throw`,
`mem-ops.cpp:1361`), and 7b's four destruction drains.

**Sites that must stay plain within the auxiliary set: none.** I read each site. The handler takes no lock and
allocates nothing, so delivery is safe where the waits run:
- **Under a mutex:** the replan's `g_block_exec_copy_queue_mutex` and `g_pipeline_copy_queue_mutex`, shutdown's
  `g_pipeline_copy_queue_mutex`, and the dense scheduler's `mutex_`.
- **On a background thread:** the scatter's detached thread (`:31776`) and the TP worker thread.
- **In a destructor:** L2 prefetch and the dense scheduler.

No site depends on the wait not delivering:
- **`catch (...)` wrappers.** Prestage, device pool, graph entry, shutdown, and both destructors wrap their waits in
  `catch (...)`, and preload and reclaim catch `sycl::exception` and log it. Because the handler does not rethrow,
  these blocks see the same exceptions as today.
- **The graph-entry drains** (`:105217`, `:105233`) are skipped while a graph is being recorded, because Level Zero
  forbids a host-blocking call on a queue that shares the recording queue's context. `wait_and_throw` blocks exactly
  as `wait` does, so the same guard (`:105201`) covers it, and no converted site enters a recording region.
- **Destruction paths.** On the L2 and dense destructors and the shutdown loop, the converted wait is the queue's last
  chance to deliver (fact f).

**Implementation Guide:**

1. **Test (RED).** `tests/test-sycl-async-drain-source.py`:

```python
#!/usr/bin/env python3
"""A drain of an auxiliary backend queue delivers its asynchronous errors (llama.cpp-kjoc).

A plain wait() neither throws nor calls the queue's async handler, so an error on a queue whose drains are all plain
reaches the latch late or never. wait_and_throw() hands it to the handler, which records it and never rethrows: the
call returns as wait() does. Execution queues are exempt: a context's synchronize ends every graph with a
wait_and_throw on its own device's execution queue or on that graph's last event.

  W. No plain wait() in the backend's non-test sources has an auxiliary queue in its receiver: the cache's DMA and BCS
     queues, the PP pipeline DMA queue, the pipeline copy queues, the split owner queues, the TP device-1 worker queue,
     the dpct default queues, the L2 prefetch queue and the dense scheduler's copy queue.
  A. Where such a queue, or an event submitted to one, is waited through a local name, that name is never waited
     plainly in that function (or file).
  R. synchronize_for_replan waits plainly only on execution queues.
  T. The TP worker no longer claims that a plain wait catches asynchronous errors.
  M. The dense scheduler's and the L2 prefetch manager's queue members are built through ggml_sycl_make_queue.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"

PLAIN_WAIT = re.compile(r"(?:\.|->)\s*wait\s*\(\s*\)|\bsycl::event::wait\s*\(")
AUX_RECEIVER = re.compile(
    r"\bget_dma_queue\s*\(|\bget_bcs_queue\s*\(|\bdma_queue_\b|\bbcs_queue_\b|\bpipe\s*\.\s*dma_queue\b|"
    r"\bg_pipeline_copy_queue\b|\bslot\s*->\s*copy_queue\b|\bg_split_\w+_queue_owner\b|\bg_tp_device1_worker_queue\b|"
    r"\bdefault_queue\s*\(|\bprefetch_queue_\b|\bcopy_queue_\b")
# (file, function signature or None for the whole file, local names whose plain waits are forbidden there)
ALIASES = [
    ("ggml-sycl.cpp", "static void tp_device1_worker_thread_func()", r"stream"),
    ("ggml-sycl.cpp", "void ggml_backend_sycl_shutdown(void)", r"queue"),
    ("ggml-sycl.cpp", "int ggml_backend_sycl_verify_speculative_with_tokens(", r"default_q"),
    ("ggml-sycl.cpp", "static void ggml_sycl_measure_h2d_pinned(", r"q"),
    ("ggml-sycl.cpp", "static void moe_prestage_popular_experts()", r"source\s*\.\s*ready"),
    ("ggml-sycl.cpp", "void * moe_expert_ensure_soa_cached(", r"source\s*\.\s*ready"),
    ("ggml-sycl.cpp", "static bool ggml_sycl_materialize_planned_expert_layout(", r"source\s*\.\s*ready"),
    ("common.cpp", "void ggml_sycl_tp_copy_weight_shard(", r"stream"),
    ("dense-scheduler.cpp", None, r"pending_prefetch_"),
    ("layer-streaming.cpp", None, r"prefetch_event_|last_event"),
]
REPLAN = "bool ggml_backend_sycl_synchronize_for_replan(ggml_backend_t backend)"
REPLAN_EXEC_RECEIVER = re.compile(r"^(ctx->stream\(ctx->device, 0\)|cache->get_queue\(\))$")
REPLAN_EXEC_BINDING = re.compile(r"ggml_sycl_execution_queue_for_device\s*\(|get_shared_context_queue\s*\(")


def blank_comments(text):
    # Line comments first: a "/*" inside one (unified-cache.cpp has `weight-reclaim/*`) would otherwise open a block
    # comment that swallows code up to the next "*/".
    text = re.sub(r"//[^\n]*", lambda m: " " * len(m.group(0)), text)
    return re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)


def body(text, signature):
    start = text.find(signature)
    if start < 0:
        return None
    open_at = text.find("{", start)
    depth = 0
    for i in range(open_at, len(text)):
        depth += text[i] == "{"
        depth -= text[i] == "}"
        if depth == 0:
            return text[open_at:i + 1]
    return None


def receiver(text, match):
    # The expression the wait is called on: back to the start of its statement.
    start = max(text.rfind(c, 0, match.start()) for c in ";{}") + 1
    return " ".join(text[start:match.start()].split())


def plain_waits(text):
    for m in PLAIN_WAIT.finditer(text):
        yield m, receiver(text, m), text.count("\n", 0, m.start()) + 1


def sources():
    for path in sorted(SYCL.rglob("*")):
        if path.suffix in (".cpp", ".hpp", ".h") and "tests" not in path.relative_to(SYCL).parts:
            yield path.relative_to(SYCL).as_posix(), path.read_text(errors="replace")


def check(raw_files):
    # Claims W, A, R and M read code with comments blanked; claim T reads the comment itself.
    files = [(name, blank_comments(text)) for name, text in raw_files]
    texts = dict(files)
    bad = {}
    for name, text in files:
        for m, recv, line in plain_waits(text):
            if AUX_RECEIVER.search(recv):
                bad.setdefault(f"{name}:{line}", f"W {name}:{line}: {recv}{' '.join(m.group(0).split())}")
    for name, signature, alias in ALIASES:
        text = texts.get(name, "")
        scope = text if signature is None else body(text, signature)
        if scope is None:
            bad[f"A {name} {signature}"] = f"A {name}: function not found: {signature}"
            continue
        for m, recv, line in plain_waits(scope):
            if re.fullmatch(rf"(?:{alias})", recv):
                bad.setdefault(f"A {name} {signature} {line}", f"A {name} ({signature}): plain wait on {recv}")
    replan = body(texts.get("ggml-sycl.cpp", ""), REPLAN)
    if replan is None:
        bad["R"] = "R synchronize_for_replan not found"
    else:
        for m, recv, line in plain_waits(replan):
            if REPLAN_EXEC_RECEIVER.match(recv):
                continue
            binding = []
            if recv.isidentifier():
                binding = re.findall(rf"\b{re.escape(recv)}\s*=\s*([^;)]*)", replan[:m.start()])
            if binding and REPLAN_EXEC_BINDING.search(binding[-1]):
                continue
            bad.setdefault(f"R {line}", f"R synchronize_for_replan: plain wait on {recv}")
    if "Add wait to catch any async errors" in dict(raw_files).get("ggml-sycl.cpp", ""):
        bad["T"] = "T the TP worker still says a plain wait catches asynchronous errors"
    if not re.search(r"\bcopy_queue_\s*\(\s*ggml_sycl_make_queue\s*\(", texts.get("dense-scheduler.cpp", "")):
        bad["M dense"] = "M dense_layer_scheduler::copy_queue_ is not built through ggml_sycl_make_queue"
    if not re.search(r"\bprefetch_queue_\s*\(\s*ggml_sycl_make_queue\s*\(", texts.get("l2-prefetch.hpp", "")):
        bad["M l2"] = "M L2PrefetchManager::prefetch_queue_ is not built through ggml_sycl_make_queue"
    return list(bad.values())


def mutate(files, name, old, new, signature=None):
    # With a signature, the replacement lands inside that function's body only.
    def apply(text):
        a, b = 0, len(text)
        if signature is not None:
            scope = body(text, signature) or ""
            a = text.find(scope) if scope else -1
            if a < 0:
                return text
            b = a + len(scope)
        return text[:a] + text[a:b].replace(old, new, 1) + text[b:]
    out = [(n, apply(t) if n == name else t) for n, t in files]
    if out == files:
        raise SystemExit(f"self-test: mutant text not found in {name}: {old!r}")
    return out


def main():
    files = list(sources())
    bad = check(files)
    if bad:
        print("async drain source contract failed:\n  " + "\n  ".join(bad), file=sys.stderr)
        return 1
    if "--self-test" in sys.argv:
        sycl = "ggml-sycl.cpp"
        tp_fn = "static void tp_device1_worker_thread_func()"
        tp_new = "stream->wait_and_throw();  // delivers this queue's asynchronous errors to the latch"
        shutdown = "void ggml_backend_sycl_shutdown(void)"
        mutants = {
            "direct auxiliary wait":
                files + [("mutant.cpp", "void f(unified_cache * cache) { cache->get_bcs_queue().wait(); }")],
            "slash-star in a line comment":
                files + [("mutant.cpp", "// see weight-reclaim/* below\nvoid f() { dma_queue_->wait(); }\n// */\n")],
            "TP worker plain again": mutate(files, sycl, "stream->wait_and_throw();", "stream->wait();", tp_fn),
            "false TP comment back":
                mutate(files, sycl, tp_new, "stream->wait_and_throw();  // Add wait to catch any async errors", tp_fn),
            "replan CPU queue plain": mutate(files, sycl, "q->wait_and_throw();", "q->wait();", REPLAN),
            "prestage ready event plain": mutate(files, sycl, "source.ready.wait_and_throw();", "source.ready.wait();"),
            "shutdown copy queue plain": mutate(files, sycl, " queue->wait_and_throw();", " queue->wait();", shutdown),
            "unhandled dense copy queue":
                mutate(files, "dense-scheduler.cpp", "copy_queue_(ggml_sycl_make_queue(", "copy_queue_(sycl::queue("),
        }
        for name, mutant in mutants.items():
            if not check(mutant):
                print(f"async drain source contract: mutant '{name}' was not caught", file=sys.stderr)
                return 1
    print("async drain source contract: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Register it in `tests/CMakeLists.txt` directly after 7b's `test-sycl-async-handler-source`, with the same labels and
timeout.

Run: `python3 tests/test-sycl-async-drain-source.py --self-test; echo "rc=$?"`. Expected RED on a tree that has 7b:
59 lines and `rc=1`:
- 25 W, one per direct site in the table;
- 22 A: the 4 TP worker waits, shutdown, the speculative verify, the H2D probe, the 4 `source.ready` waits, the TP
  shard upload, 5 dense scheduler and 5 layer-streaming waits;
- 9 R, the replan's non-execution waits;
- 1 T;
- 2 M.

This was checked against the tree at `7809ffac8`, which gives 60 lines: the extra W is `:96756`, 7b's conversion. The
contract then passed with `--self-test` on a scratch copy with 7a, 7b and the table below applied, and each of the
eight mutants failed under its own claim. 7a's and 7b's contracts still passed on that copy. If a W or A hit is not
in the table, it is a site to convert by the same rule: name it in your report. If a hit is not a wait on an auxiliary
queue, it is a scanner bug: fix the scanner, and add a mutant for the fix.

2. **Implement (GREEN).** Change each site from the plain form to `wait_and_throw()`. Keep each statement's receiver
and its surrounding `try`. Runs: **L** = per model load, replan, pool growth or shutdown; **G** = once per graph (one
decode graph is one token); **O** = per op or per expert inside a graph. *opt-in* = reached only with the named
feature on.

**Cache DMA and BCS queues** (process lifetime; the cache drains them at destruction with `wait_and_throw`)

| Site | Function | Today | Replacement | Runs |
|------|----------|-------|-------------|------|
| `ggml-sycl.cpp:5110`, `:5125` | `moe_prestage_popular_experts` | `cache->get_dma_queue().wait();` | `cache->get_dma_queue().wait_and_throw();` | L |
| `:5111`, `:5126` | same | `cache->get_bcs_queue().wait();` | `cache->get_bcs_queue().wait_and_throw();` | L |
| `:5598` | same, secondary devices | `b.cache->get_bcs_queue().wait();` | `b.cache->get_bcs_queue().wait_and_throw();` | L, two-card |
| `:4951`, `:5356` | same | `source.ready.wait();` | `source.ready.wait_and_throw();` | L |
| `:5864` | `moe_expert_ensure_soa_cached` (from `kv_expert_rebalance_check`) | `source.ready.wait();` | `source.ready.wait_and_throw();` | L (rebalance) |
| `:6415` | `ggml_sycl_materialize_planned_expert_layout` | `source.ready.wait();` | `source.ready.wait_and_throw();` | O, on a layout miss |
| `:33138` | `ggml_sycl_measure_h2d_pinned` (`q` is `&cache->get_bcs_queue()`, `:33110`) | `q->wait();` | `q->wait_and_throw();` | L |
| `:44171` | `ggml_sycl_preload_model_weights` | `cache->get_bcs_queue().wait();` | `cache->get_bcs_queue().wait_and_throw();` | L |
| `:105217` | `ggml_backend_sycl_graph_compute_impl`, graph entry | `cache->get_bcs_queue().wait();` | `cache->get_bcs_queue().wait_and_throw();` | **G** |
| `:105233` | same | `cache->get_dma_queue().wait();` | `cache->get_dma_queue().wait_and_throw();` | **G** |
| `unified-cache.cpp:16568` | `reclaim_weight_entries` | `dma_queue_->wait();` | `dma_queue_->wait_and_throw();` | L |
| `unified-cache.cpp:16571` | same | `bcs_queue_->wait();` | `bcs_queue_->wait_and_throw();` | L |
| `device-pool.hpp:192`, `:317` | chunk growth (`bcs_queue_` set by `set_bcs_queue`) | `bcs_queue_->wait();` | `bcs_queue_->wait_and_throw();` | L |

`source.ready` is a cache entry's producer event. It comes from `copy_to_device` on the cache's execution queue
(`unified-cache.cpp:13888`ff) or from direct staging on the staging queue that the caller passes
(`unified-cache.cpp:8591`). On the execution-queue origin the conversion is redundant, and harmless. On the staging
origin, it is the only delivery before the expert is used.

**PP pipeline DMA queue** (`g_pp_pipeline.dma_queue`, a static; built at `:106138-106140`)

| Site | Function | Today | Replacement | Runs |
|------|----------|-------|-------------|------|
| `:76593`, `:76610`, `:76662` | `ggml_sycl_mul_mat` | `pipe.dma_queue->wait();` | `pipe.dma_queue->wait_and_throw();` | O, prompt graphs, opt-in `GGML_SYCL_PP_PIPELINE` |
| `:108039` | `ggml_backend_sycl_graph_compute_impl`, end of graph | `pipe.dma_queue->wait();` | `pipe.dma_queue->wait_and_throw();` | G, opt-in, only with a prefetch pending |

**TP device-1 worker queue** (`g_tp_device1_worker_queue`, `:58660`; `stream` in the worker is that queue, `:58662`)

| Site | Function | Today | Replacement | Runs |
|------|----------|-------|-------------|------|
| `:58792` | `tp_device1_worker_thread_func` | `stream->wait();  // Add wait to catch any async errors` | `stream->wait_and_throw();  // delivers this queue's asynchronous errors to the latch` | O (per layer), opt-in threaded TP FFN |
| `:58801`, `:58814`, `:58845` | same | `stream->wait();` | `stream->wait_and_throw();` | O, opt-in |
| `:114049` | `ggml_backend_sycl_synchronize_for_replan` | `g_tp_device1_worker_queue->wait();` | `g_tp_device1_worker_queue->wait_and_throw();` | L |

The `:58792` comment was false: by fact (g) a plain wait catches nothing. With `wait_and_throw` it becomes true. The
worker's other three waits get no comment.

**TP shared queues** (`ggml_sycl_get_tp_queue`, `common.cpp:212`)

| Site | Function | Today | Replacement | Runs |
|------|----------|-------|-------------|------|
| `common.cpp:2524` | `ggml_sycl_tp_copy_weight_shard` (its caller passes `ggml_sycl_get_tp_queue(device)`, `ggml-sycl.cpp:50701-50706`) | `stream->wait();` | `stream->wait_and_throw();` | L, opt-in TP |
| `ggml-sycl.cpp:114045` | `ggml_backend_sycl_synchronize_for_replan` | `q->wait();` (bound to `ggml_sycl_get_tp_queue(d)`) | `q->wait_and_throw();` | L |

**Pipeline copy queues** (`g_pipeline_copy_queue[d]`; `slot->copy_queue` is the same queue, `:31759`)

| Site | Function | Today | Replacement | Runs |
|------|----------|-------|-------------|------|
| `:31402` | `pipeline_scatter_background` (detached thread) | `slot->copy_queue->wait();` | `slot->copy_queue->wait_and_throw();` | per scatter, opt-in MoE pipeline |
| `:114067` | `ggml_backend_sycl_synchronize_for_replan`, under `g_pipeline_copy_queue_mutex` | `g_pipeline_copy_queue[d]->wait();` | `g_pipeline_copy_queue[d]->wait_and_throw();` | L |
| `:125031` | `ggml_backend_sycl_shutdown`, under the same mutex, then `delete queue` | `queue->wait();` | `queue->wait_and_throw();` | L (shutdown) |

`:125031` falls under 7b's destruction-drain rule (decision 7): the queue is deleted on the next line. 7b converts the
same queue's drain in `ggml_backend_sycl_free` (`:96756`) and misses this one. It is here because 7b's scope is
fixed.

**Split owner, block-exec copy and CPU-dispatch queues** (replan only)

| Site | Function | Today | Replacement | Runs |
|------|----------|-------|-------------|------|
| `:114013` | `ggml_backend_sycl_synchronize_for_replan` | `g_split_secondary_queue_owner->wait();` | `g_split_secondary_queue_owner->wait_and_throw();` | L |
| `:114016` | same | `g_split_merge_queue_owner->wait();` | `g_split_merge_queue_owner->wait_and_throw();` | L |
| `:114019` | same | `g_split_coord_queue_owner->wait();` | `g_split_coord_queue_owner->wait_and_throw();` | L |
| `:114033` | same | `cache->get_bcs_queue().wait();` | `cache->get_bcs_queue().wait_and_throw();` | L |
| `:114038` | same | `q->wait();` (bound to `ggml_sycl_get_cpu_queue()`) | `q->wait_and_throw();` | L |
| `:114057` | same, under `g_block_exec_copy_queue_mutex` | `q->wait();` (each of `g_block_exec_copy_queues`) | `q->wait_and_throw();` | L |

The replan's execution-queue waits stay plain under rule E or X. These are `ctx->stream(ctx->device, 0)->wait()`
(`:114002`), the loops over `ggml_sycl_execution_queue_for_device` (`:114006`) and `get_shared_context_queue`
(`:114026`), and `cache->get_queue().wait()` (`:114032`). Claim R allows exactly these.

**dpct default queues** (one per device, shared slot after 7b)

| Site | Function | Today | Replacement | Runs |
|------|----------|-------|-------------|------|
| `ggml-sycl.cpp:51377` | `ggml_backend_sycl_sample_token_full` | `ggml_sycl_get_device(buf_ctx->device).default_queue().wait();` | `ggml_sycl_get_device(buf_ctx->device).default_queue().wait_and_throw();` | per sampled token, when the backend sampler API is used |
| `:51647` | `ggml_backend_sycl_verify_speculative_with_tokens` | `default_q.wait();` | `default_q.wait_and_throw();` | per verification, same API |
| `layer-streaming.cpp:58` | `drain_and_invalidate_buffers` | `prefetch_event_.wait();` | `prefetch_event_.wait_and_throw();` | L |
| `layer-streaming.cpp:353` | `load_layer_sync` | `last_event.wait();` | `last_event.wait_and_throw();` | O, only while layer streaming is active |
| `layer-streaming.cpp:379`, `:410`, `:464` | `ensure_layer`, `prefetch_next_layer` | `prefetch_event_.wait();` | `prefetch_event_.wait_and_throw();` | O, same |

The layer-streaming events come from `mem_copy_async` on the queue the caller passes. Both product callers pass
`ggml_sycl_get_device(d).default_queue()` (`ggml-sycl.cpp:58428`, `:75884`, `:75893`).

**L2 prefetch queue and dense scheduler copy queue.** 7b's scanner does not see either construction: both are member
initializers, `x_(ctx, dev)`, which none of its patterns covers. A scan of the backend for member initializers built
from `get_context()`, `get_device()` or queue properties finds only these two.

| Site | Today | Replacement | Runs |
|------|-------|-------------|------|
| `l2-prefetch.hpp:155` | `prefetch_queue_(compute_queue->get_context(), compute_queue->get_device())` | `prefetch_queue_(ggml_sycl_make_queue(compute_queue->get_context(), compute_queue->get_device(), ggml_sycl_get_device_id_from_queue(*compute_queue), sycl::property_list{}))` | per context, when `is_l2_prefetch_enabled()` |
| `l2-prefetch.hpp:164` (destructor) | `prefetch_queue_.wait();` | `prefetch_queue_.wait_and_throw();` | L (context free) |
| `dense-scheduler.cpp:48` | `copy_queue_(compute_queue.get_context(), compute_queue.get_device()),` | `copy_queue_(ggml_sycl_make_queue(compute_queue.get_context(), compute_queue.get_device(), ggml_sycl_get_device_id_from_queue(compute_queue), sycl::property_list{})),` | per scheduler |
| `dense-scheduler.cpp:71` (destructor), `:113`, `:177`, `:202`, `:212` | `pending_prefetch_.wait();` | `pending_prefetch_.wait_and_throw();` | see below |

`pending_prefetch_` comes from `mem_copy_async(..., copy_queue_)` (`dense-scheduler.cpp:193`). The product never
constructs a `dense_layer_scheduler`. Only `tests/test-dense-scheduler.cpp` does. The edits keep the file under the
same rule as the rest of the backend. The L2 destructor drain is a destruction drain like 7b's four: the queue goes
with the manager, which is destroyed with its context (`ggml-sycl.cpp:54105`).

**Per-token sites that gain a runtime call.** `wait()` and `wait_and_throw()` share the wait. The difference is
`throw_asynchronous()` on the queue. In the open DPC++ sources, that call locks the queue's own mutex to swap out its
exception list. `event::wait_and_throw()` also has to find the event's queue, which in those sources takes the
scheduler's graph read lock for an event that has a scheduler command. I could not confirm either for the 2026.2
runtime, which ships here as a binary. Task 9's A/B is where a cost shows. In the default build, decode reaches:
- `ggml-sycl.cpp:105217` and `:105233`: one `queue::wait_and_throw` on the BCS queue and one on the DMA queue at every
  graph entry, prompt and decode. This is the only change to the default decode path.
- `:6415`: one `event::wait_and_throw` per expert layout miss during route resolution. In steady state there is no
  miss.

Only with a feature on, or off the decode thread:
- `:58792`, `:58801`, `:58814`, `:58845`: four per layer per token (threaded TP FFN).
- `:31402`: one per scatter, on the background thread (MoE pipeline).
- `:51377`: one per sampled token through the backend sampler API. Whether a product path calls it was not checked.
- `layer-streaming.cpp:353`, `:379`, `:410`, `:464`: per layer transition while layer streaming is active.
- `:76593`, `:76610`, `:76662`, `:108039`: prompt graphs only (`GGML_SYCL_PP_PIPELINE`).

Not changed, listed for Task 9: synchronize's deferred-decode branch runs `last_graph_event->wait_and_throw()` and does
not drain the queue (`ggml-sycl.cpp:97097`ff). If that call does not deliver errors from other commands on the same
queue (see "Not verified"), the fix would be `stream->throw_asynchronous()` (`queue.hpp:527-531`) in that branch. It
would be a new call per decode token, taking one queue-mutex lock. 7c does not add it.

**Not verified** (host-only design; the lead decides which, if any, to measure):
- Whether `event::wait_and_throw()` delivers asynchronous errors of other commands on the event's queue, or only its
  own. Facts (a) and (b) measured one failing command waited by its own event. A probe case would be: one failing
  command, then a second command on the same queue, then `wait_and_throw` on the second command's event; count the
  handler calls. This decides the deferred-decode branch above. It changes no site in this task.
- Rule P: that every caller of the listed helpers passes an execution queue. If one passes an auxiliary queue, its
  wait belongs in the table.
- Which lock `throw_asynchronous` takes in the 2026.2 runtime.

**Acceptance Criteria:**
- [ ] `test-sycl-async-drain-source --self-test` passes, and each of its eight mutants fails.
- [ ] 7a's `test-sycl-async-failure-source` and 7b's `test-sycl-async-handler-source` still pass with `--self-test`.
- [ ] No plain wait remains on the queues in the table, outside the execution-queue waits that rules E and X keep
      (claims W, A and R).
- [ ] `:58792` no longer says "Add wait to catch any async errors" (claim T).
- [ ] Full build succeeds (`./scripts/sycl-build.sh`, all targets; `tests/test-dense-scheduler.cpp` compiles against
      the new member initializer).
- [ ] Lead: the Mistral B70 completion gate and the GPT-OSS B50 chat gate pass once each.

Run:
```bash
python3 tests/test-sycl-async-drain-source.py --self-test      # async drain source contract: PASS
python3 tests/test-sycl-async-handler-source.py --self-test    # still PASS
python3 tests/test-sycl-async-failure-source.py --self-test    # still PASS
./scripts/sycl-build.sh                                          # all targets
source /opt/intel/oneapi/setvars.sh --force
ctest --test-dir build -L source-contract -j 4 --output-on-failure
```

**Commit:**
```bash
git commit -F <msgfile> -- ggml/src/ggml-sycl/ggml-sycl.cpp ggml/src/ggml-sycl/unified-cache.cpp \
  ggml/src/ggml-sycl/device-pool.hpp ggml/src/ggml-sycl/common.cpp ggml/src/ggml-sycl/layer-streaming.cpp \
  ggml/src/ggml-sycl/l2-prefetch.hpp ggml/src/ggml-sycl/dense-scheduler.cpp \
  tests/test-sycl-async-drain-source.py tests/CMakeLists.txt
```
Subject: `sycl: drains of auxiliary queues deliver their asynchronous errors (kjoc)`. End it with your own
Co-Authored-By trailer.

**Gotchas:**
- Several Today strings are substrings of others. `b.cache->get_bcs_queue().wait();` contains
  `cache->get_bcs_queue().wait();`, and `pipe.dma_queue->wait();` contains `queue->wait();`. Edit by line, not by a
  global replace. The scratch check hit the first trap.
- Do not convert the execution-queue lines that sit next to converted ones (rule E). The contract allows them, and
  converting them changes nothing the latch reads. Keep the statement-per-line layout, so a review sees which line
  changed.
- Leave the `catch (...)` blocks as they are. They now see only synchronous exceptions, which is all they ever saw:
  with no handler, the runtime terminated on an asynchronous error instead of throwing it to them.
- `l2-prefetch.hpp` and `dense-scheduler.cpp` include `common.hpp`, which includes `async-failure.hpp` after 7a. Add
  no include.
- The contract is a text scanner. It sees a wait through a local name only where claim A names that name. A new alias
  of an auxiliary queue added later escapes it. Claim A's list is the place to extend.
