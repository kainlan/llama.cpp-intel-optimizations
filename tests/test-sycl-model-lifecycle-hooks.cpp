// test-sycl-model-lifecycle-hooks: the backend must actually CALL its weight
// ownership hooks at the real llama_model lifecycle boundaries (llama.cpp-viga).
//
// ---------------------------------------------------------------------------
// The hole this fills
// ---------------------------------------------------------------------------
//
// tests/test-sycl-reset-model-weight-lease-preserve.cpp proves the reclaim
// DECISION is right: a live model's idle weights survive a load boundary,
// MID_LOAD_REPLAN frees its own staging while sparing a live model, and so on.
// It drives unified_cache directly, which is what makes it cheap (317 MB peak
// RSS) and what makes its mutation control an INPUT rather than a rebuild.
//
// The price of driving the cache directly is that it bypasses the call site.
// NOTHING asserted that the backend ever REACHES note_model_load_end() or
// release_model_slot() on a real model load and teardown.  So the ownership
// logic could be flawless and simply never invoked in production, and the suite
// would stay green -- while a llama_model destructor silently stopped reclaiming
// its weights and VRAM leaked for the process lifetime.
//
// The chain under test, all of which must run for this test to pass:
//
//   llama_model_load_from_file
//     -> llama_model_base::load_tensors           (llama-model.cpp)
//        -> llama_model_sycl_loading_guard(true, &sycl_model_slot)
//           -> ggml_backend_sycl_set_model_loading(false)   [outermost exit]
//              -> ggml_sycl::unified_cache_note_model_load_end(slot)
//           -> ggml_backend_sycl_model_slot_current()  stores the token
//
//   llama_model_free
//     -> ~llama_model                              (llama-model.cpp)
//        -> hooks.unload(token)
//           -> ggml_backend_sycl_model_unloaded_token(token)
//              -> ggml_sycl::unified_cache_release_model_slot(token.slot)
//
// Each link has its own way of silently disappearing: the #ifdef GGML_USE_SYCL
// around the destructor hook, llama_model_sycl_hooks_enabled(), the
// sycl_model_slot != NONE guard, the out_slot plumbing on the OUTERMOST loading
// guard only, and the depth counter that decides which set_model_loading(false)
// is outermost.  Delete any one of them and this test goes red; none of them is
// covered by any other test that can be run affordably.
//
// ---------------------------------------------------------------------------
// Why not extend test-thread-safety
// ---------------------------------------------------------------------------
//
// It is the only other thing that exercises this path, and it is unusable as a
// verdict channel: it peaks at 156-180 GB of Shmem for a 19 MB model, and its
// exit status is already consumed by two unrelated crashes (llama.cpp-oze0,
// llama.cpp-09ep).  Assertions bolted onto it would report through a channel
// that is known-broken, which is coverage on paper only.
//
// This test loads the same 19 MB model ONCE, with no context and no decode.
//
// ---------------------------------------------------------------------------
// What makes this non-vacuous
// ---------------------------------------------------------------------------
//
// A test that can report green while doing no work is worse than no test.  Four
// independent guards, three of them checked inside the run itself:
//
//  1. No SYCL device, or no model file  -> exit 77, NOT 0.  ctest reports the
//     run as skipped and a bare shell invocation gets a non-zero status.  The
//     device precondition is load-bearing rather than cosmetic: with no device
//     there is no unified_cache, and the counters would still advance because
//     the backend entry points run regardless -- a green run proving nothing.
//
//  2. Both counters must read ZERO before the model is loaded.  A counter
//     bumped by static init, backend registration or device enumeration would
//     make every later delta meaningless.  This catches that.
//
//  3. An EXECUTED negative control: ggml_backend_sycl_model_unloaded() is
//     called with GGML_SYCL_MODEL_SLOT_NONE, which it must reject before doing
//     any work.  The teardown counter must NOT move.  This separates "the
//     counter tracks a real slot release" from "the counter increments on
//     function entry".
//
//  4. Cross-discrimination: the load counter must not move at teardown and the
//     teardown counter must not move at load.  Each hook is asserted to fire at
//     its own boundary AND to stay silent at the other one, so a counter wired
//     to the wrong event, or to both, fails.
//
// The slot TOKEN is checked too, not just the call counts: the slot the backend
// handed the model at load must be the slot its destructor released.  A hook
// that fires with the wrong token reclaims another model's weights, which is
// llama.cpp-0qlw's defect wearing different clothes.
//
// ---------------------------------------------------------------------------
// Scope
// ---------------------------------------------------------------------------
//
// This proves the hooks are REACHED with the right token.  It does not re-prove
// what they decide once reached -- that is
// tests/test-sycl-reset-model-weight-lease-preserve.cpp's job, and the two are
// the two halves of one claim.  Neither is sufficient alone.

#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml-sycl/model-lifecycle-probe.hpp"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__linux__)
#    include <sys/resource.h>
#endif

// ctest's SKIP_RETURN_CODE.  Never 0: a skip that exits 0 reads as verification.
static const int EXIT_SKIP = 77;

static int g_failures = 0;

static void check(bool cond, const char * what, const char * why) {
    if (cond) {
        printf("  PASS: %s\n", what);
        return;
    }
    printf("  FAIL: %s\n", what);
    printf("        %s\n", why);
    g_failures++;
}

// Shmem is the figure that tracks the GPU-BO TTM backing hazard; MemAvailable
// alone conflates it with reclaimable cache (CLAUDE.md, Running Tests).  Printed
// so a single run also yields the cost evidence for this test's registration.
static void report_meminfo(const char * when) {
#if !defined(__linux__)
    (void) when;
#else
    FILE * f = fopen("/proc/meminfo", "r");
    if (!f) {
        return;
    }
    char line[256];
    long shmem_kb = -1, avail_kb = -1;
    while (fgets(line, sizeof(line), f)) {
        if (std::strncmp(line, "Shmem:", 6) == 0) {
            sscanf(line + 6, "%ld", &shmem_kb);
        } else if (std::strncmp(line, "MemAvailable:", 13) == 0) {
            sscanf(line + 13, "%ld", &avail_kb);
        }
    }
    fclose(f);
    printf("[mem] %-18s Shmem %6.1f GB   MemAvailable %6.1f GB\n", when, shmem_kb / 1048576.0, avail_kb / 1048576.0);
#endif
}

static void report_peak_rss(void) {
#if defined(__linux__)
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        printf("[mem] peak RSS         %6.1f MB\n", ru.ru_maxrss / 1024.0);
    }
#endif
}

static const char * model_path_from(int argc, char ** argv) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 || std::strcmp(argv[i], "--model") == 0) {
            return argv[i + 1];
        }
    }
    const char * env = std::getenv("LLAMACPP_TEST_MODELFILE");
    if (env && env[0] != '\0') {
        return env;
    }
    return nullptr;
}

static int sycl_gpu_device_count(void) {
    ggml_backend_reg_t reg = ggml_backend_sycl_reg();
    if (!reg) {
        return 0;
    }
    int n = 0;
    for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); ++i) {
        ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, i);
        if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            printf("[env] SYCL GPU device %d: %s\n", n, ggml_backend_dev_name(dev));
            n++;
        }
    }
    return n;
}

int main(int argc, char ** argv) {
    printf("=== test-sycl-model-lifecycle-hooks (llama.cpp-viga) ===\n");

    const char * model_path = model_path_from(argc, argv);
    if (!model_path) {
        printf("SKIP: no model given (-m <gguf> or LLAMACPP_TEST_MODELFILE).\n");
        printf("      This run proves NOTHING about the lifecycle hooks.\n");
        return EXIT_SKIP;
    }
    if (FILE * f = fopen(model_path, "rb")) {
        fclose(f);
    } else {
        printf("SKIP: model not readable: %s\n", model_path);
        printf("      The test-download-model fixture supplies it; it did not run or failed.\n");
        printf("      This run proves NOTHING about the lifecycle hooks.\n");
        return EXIT_SKIP;
    }

    llama_backend_init();

    // Precondition, not decoration.  With no SYCL device there is no
    // unified_cache, yet ggml_backend_sycl_set_model_loading() and
    // ggml_backend_sycl_model_unloaded() still run and would still bump these
    // counters -- a green run that verified nothing.  Refuse to be that run.
    if (sycl_gpu_device_count() == 0) {
        printf("SKIP: no SYCL GPU device.  Source oneAPI (setvars.sh) and check\n");
        printf("      ONEAPI_DEVICE_SELECTOR.  Without a device there is no unified\n");
        printf("      cache, so the counters would advance while nothing was cached.\n");
        printf("      This run proves NOTHING about the lifecycle hooks.\n");
        llama_backend_free();
        return EXIT_SKIP;
    }

    ggml_backend_sycl_model_lifecycle_probe before   = {};
    ggml_backend_sycl_model_lifecycle_probe rejected = {};
    ggml_backend_sycl_model_lifecycle_probe loaded   = {};
    ggml_backend_sycl_model_lifecycle_probe freed    = {};

    ggml_backend_sycl_model_lifecycle_probe_read(&before);

    printf("\n[1] baseline: the counters must be untouched before any model exists\n");
    // Observed values precede every verdict in this file, including here. A
    // surprise in the baseline or the negative control is the most informative
    // failure in the test -- it says the probe itself is not measuring what it
    // claims -- so it must be the most legible, not the least.
    printf("  [info] load_end_calls = %llu, release_slot_calls = %llu\n", (unsigned long long) before.load_end_calls,
           (unsigned long long) before.release_slot_calls);
    check(before.load_end_calls == 0, "note_model_load_end has not been called",
          "a counter that is already nonzero before a model is loaded makes every\n"
          "        later delta meaningless -- it would be bumped by static init,\n"
          "        backend registration or device enumeration, not by a load.");
    check(before.release_slot_calls == 0, "release_model_slot has not been called",
          "same: the teardown counter must start at zero for the delta to mean anything.");

    printf("\n[2] negative control: a rejected slot must not bump anything\n");
    // GGML_SYCL_MODEL_SLOT_NONE is rejected at the top of the function, before
    // any cache call.  If the counter moved here it would be counting function
    // ENTRY rather than a real slot release, and every assertion below would be
    // satisfiable without the cache ever being touched.
    ggml_backend_sycl_model_unloaded(GGML_SYCL_MODEL_SLOT_NONE);
    ggml_backend_sycl_model_lifecycle_probe_read(&rejected);
    printf("  [info] release_slot_calls = %llu (was %llu), release_slot_last_slot = %u\n",
           (unsigned long long) rejected.release_slot_calls, (unsigned long long) before.release_slot_calls,
           rejected.release_slot_last_slot);
    check(rejected.release_slot_calls == 0, "ggml_backend_sycl_model_unloaded(SLOT_NONE) bumped nothing",
          "the counter tracks entry to the function, not a slot release that reached\n"
          "        the cache.  Everything this test asserts afterwards would then be\n"
          "        satisfiable with the unified cache never touched at all.");

    report_meminfo("before load");

    printf("\n[3] load the model -- the load hook must fire, the teardown hook must not\n");
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers       = 99;

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        printf("  FAIL: llama_model_load_from_file failed for %s\n", model_path);
        printf("        Cannot test lifecycle hooks without a model. This is a FAILURE,\n");
        printf("        not a skip: the model file was readable, so the load path broke.\n");
        llama_backend_free();
        return 1;
    }
    report_meminfo("after load");

    const uint32_t slot = ggml_backend_sycl_model_slot_current();
    ggml_backend_sycl_model_lifecycle_probe_read(&loaded);
    // Observed values are printed alongside every verdict so a surprising count
    // is diagnosable from the one run this test is allowed, without a re-run.
    printf("  [info] model slot = %u, params = %llu\n", slot, (unsigned long long) llama_model_n_params(model));
    printf("  [info] load_end_calls = %llu, release_slot_calls = %llu, load_end_last_slot = %u\n",
           (unsigned long long) loaded.load_end_calls, (unsigned long long) loaded.release_slot_calls,
           loaded.load_end_last_slot);

    check(loaded.load_end_calls == 1, "note_model_load_end() ran exactly once for one model load",
          "the backend never told the cache the load finished.  Either the OUTERMOST\n"
          "        llama_model_sycl_loading_guard did not run, or the depth counter in\n"
          "        ggml_backend_sycl_set_model_loading() decided no exit was outermost.\n"
          "        The weight ownership logic is then correct and unreachable: no entry\n"
          "        ever gets an owner_mask bit, so a later load boundary treats this\n"
          "        model's weights as unattributed.");
    check(loaded.release_slot_calls == 0, "release_model_slot() did NOT run at load",
          "the teardown hook fired at a LOAD boundary.  A counter that moves at both\n"
          "        boundaries proves nothing about either; worse, a real teardown here\n"
          "        would reclaim the weights of the model that is still loading.");
    check(slot != GGML_SYCL_MODEL_SLOT_NONE, "the model was given an ownership slot",
          "ggml_backend_sycl_model_slot_current() returned NONE, so llama_model has no\n"
          "        token to hand back at teardown and ~llama_model will skip the hook\n"
          "        entirely (its `sycl_model_slot != NONE` guard). Weights then stay\n"
          "        pinned for the process lifetime -- eviction skips pinned entries.");
    // `slot != NONE` is repeated here on purpose: both sides start life as the
    // NONE sentinel, so a bare equality would pass by comparing two sentinels --
    // green precisely in the case where nothing happened.
    check(slot != GGML_SYCL_MODEL_SLOT_NONE && loaded.load_end_last_slot == slot, "the cache was told about THAT slot",
          "note_model_load_end() ran with a different slot than the one handed to the\n"
          "        model. The entries tagged as owned belong to a model that does not\n"
          "        exist, and this model's weights are left unattributed.");

    printf("\n[4] free the model -- the teardown hook must fire with the same token\n");
    llama_model_free(model);
    model = nullptr;
    ggml_backend_sycl_model_lifecycle_probe_read(&freed);
    report_meminfo("after free");
    printf("  [info] load_end_calls = %llu, release_slot_calls = %llu, release_slot_last_slot = %u\n",
           (unsigned long long) freed.load_end_calls, (unsigned long long) freed.release_slot_calls,
           freed.release_slot_last_slot);

    check(freed.release_slot_calls == 1, "release_model_slot() ran exactly once at teardown",
          "~llama_model did not reach ggml_backend_sycl_model_unloaded(), or that call\n"
          "        did not reach the cache. This is the defect the hook exists to prevent:\n"
          "        ggml_sycl_preload_model_weights() PINS every dense weight it caches and\n"
          "        evict_one() skips pinned entries, so this call is the only reclaimer in\n"
          "        the system. Without it a dead model's VRAM is unreachable by LRU for the\n"
          "        rest of the process.");
    check(freed.load_end_calls == loaded.load_end_calls, "note_model_load_end() did NOT run at teardown",
          "the load hook fired at a TEARDOWN boundary. Re-claiming every cache entry for a\n"
          "        model being destroyed is the exact inverse of what teardown must do.");
    check(slot != GGML_SYCL_MODEL_SLOT_NONE && freed.release_slot_last_slot == slot,
          "the released slot is the slot the model was given",
          "the destructor released a DIFFERENT slot than the one this model was assigned.\n"
          "        That reclaims another model's weights while leaving this model's own\n"
          "        entries owned by a dead slot -- llama.cpp-0qlw's failure, reached by a\n"
          "        different route.");
    check(ggml_backend_sycl_model_slot_current() == GGML_SYCL_MODEL_SLOT_NONE, "the slot was retired after teardown",
          "ggml_backend_sycl_model_slot_current() still reports the destroyed model's slot,\n"
          "        so the next load reads a stale token.");

    printf(
        "  [info] entries reclaimed by the teardown = %llu (diagnostic only; zero is\n"
        "         legitimate when another live model owns every entry)\n",
        (unsigned long long) freed.release_slot_last_reclaimed);

    report_peak_rss();
    llama_backend_free();

    printf("\n=== %s: %d failure(s) ===\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}
