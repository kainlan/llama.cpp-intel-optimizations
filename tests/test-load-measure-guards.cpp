// C7h-3 (zhcn): the plan-override guard and the weight stand-ins of a load-time measure. Header-only
// over src/llama-load-measure.h; no device and no model. The measure that uses them is
// test-measure-context's (CPU) and the lead-run GPU arm's.
//
// The guard installs the backend's plan override in its constructor and clears it in its destructor, once, on
// every exit, a throw included; a missing entry point and a refused install each leave it uninstalled with
// a name and never call clear. The stand-ins are size-0 weight buffers set on the tensors that have none
// and removed again, leaving a tensor that already had a buffer alone.

#include "../ggml/src/ggml-backend-impl.h"
#include "../src/llama-load-measure.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static int n_failed = 0;

#define CHECK(cond, ...)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            n_failed++;                                                       \
            fprintf(stderr, "FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
            fprintf(stderr, __VA_ARGS__);                                     \
            fprintf(stderr, "\n");                                            \
        }                                                                     \
    } while (0)

static int                          g_installs   = 0;
static int                          g_clears     = 0;
static uint64_t                     g_txn        = 0;
static enum ggml_sycl_measure_stage g_stage      = GGML_SYCL_MEASURE_STAGE_PROBE;
static bool                         g_install_ok = true;

static bool fake_install(uint64_t load_txn, enum ggml_sycl_measure_stage stage) {
    g_installs++;
    g_txn   = load_txn;
    g_stage = stage;
    return g_install_ok;
}

static void fake_clear() {
    g_clears++;
}

static void reset_fakes(bool install_ok) {
    g_installs   = 0;
    g_clears     = 0;
    g_txn        = 0;
    g_stage      = GGML_SYCL_MEASURE_STAGE_PROBE;
    g_install_ok = install_ok;
}

static void test_guard() {
    // installed: the entry sees the load and the stage, and the destructor clears once
    reset_fakes(true);
    {
        llama_measure_plan_override guard({ &fake_install, &fake_clear }, 42, GGML_SYCL_MEASURE_STAGE_CANDIDATE_B);
        CHECK(guard.installed(), "the override did not install");
        CHECK(guard.failure() == nullptr, "an installed guard names a failure");
        CHECK(g_installs == 1 && g_txn == 42 && g_stage == GGML_SYCL_MEASURE_STAGE_CANDIDATE_B,
              "install saw txn %llu stage %d", (unsigned long long) g_txn, (int) g_stage);
        CHECK(g_clears == 0, "cleared while still held");
    }
    CHECK(g_clears == 1, "the destructor cleared %d times", g_clears);

    // a throw through the scope clears once
    reset_fakes(true);
    try {
        llama_measure_plan_override guard({ &fake_install, &fake_clear }, 7, GGML_SYCL_MEASURE_STAGE_PROBE);
        throw std::runtime_error("unwind");
    } catch (const std::runtime_error &) {
    }
    CHECK(g_installs == 1 && g_clears == 1, "unwind: %d installs, %d clears", g_installs, g_clears);

    // a refused install (a nest, or no plan staged) names itself and never clears
    reset_fakes(false);
    {
        llama_measure_plan_override guard({ &fake_install, &fake_clear }, 1, GGML_SYCL_MEASURE_STAGE_PROBE);
        CHECK(!guard.installed(), "a refused install reads as installed");
        CHECK(guard.failure() != nullptr && std::strcmp(guard.failure(), "plan override nested") == 0,
              "the refusal is named %s", guard.failure() ? guard.failure() : "(null)");
    }
    CHECK(g_installs == 1 && g_clears == 0, "refused: %d installs, %d clears", g_installs, g_clears);

    // a missing entry point installs nothing, names itself and never calls the other
    for (int which = 0; which < 3; which++) {
        reset_fakes(true);
        llama_measure_override_procs procs = { which == 0 ? nullptr : &fake_install,
                                               which == 1 ? nullptr : &fake_clear };
        if (which == 2) {
            procs = { nullptr, nullptr };
        }
        {
            llama_measure_plan_override guard(procs, 1, GGML_SYCL_MEASURE_STAGE_PROBE);
            CHECK(!guard.installed(), "case %d: installed with a missing proc", which);
            CHECK(guard.failure() != nullptr && std::strcmp(guard.failure(), "plan override proc missing") == 0,
                  "case %d: the refusal is named %s", which, guard.failure() ? guard.failure() : "(null)");
        }
        CHECK(g_installs == 0 && g_clears == 0, "case %d: %d installs, %d clears", which, g_installs, g_clears);
    }
}

static ggml_context * new_ctx() {
    ggml_init_params ip = { 16 * 1024, nullptr, /*no_alloc =*/true };
    return ggml_init(ip);
}

static void test_dummies() {
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

    ggml_context * wctx = new_ctx();
    ggml_tensor *  a    = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, 16);
    ggml_tensor *  b    = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, 32);

    // a tensor that already lives in a real buffer is left alone, in and out of the scope
    ggml_context *        rctx = new_ctx();
    ggml_tensor *         r    = ggml_new_tensor_1d(rctx, GGML_TYPE_F32, 8);
    ggml_backend_buffer_t real = ggml_backend_alloc_ctx_tensors_from_buft(rctx, buft);
    CHECK(real != nullptr && r->buffer == real, "the control tensor has no real buffer");

    const size_t live_before = ggml_backend_test_live_buffer_count();
    {
        const std::vector<llama_measure_dummy_entry> entries = {
            { buft,    wctx    },
            { buft,    rctx    },
            { nullptr, wctx    },
            { buft,    nullptr }
        };
        llama_measure_dummy_scope scope(entries);
        CHECK(!scope.failed(), "the stand-in buffer was refused");
        CHECK(scope.n_buffers() == 2, "%zu stand-in buffers", scope.n_buffers());
        CHECK(a->buffer != nullptr && a->buffer == b->buffer, "the tensors without a buffer got no shared stand-in");
        CHECK(a->buffer != nullptr && ggml_backend_buffer_get_usage(a->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS,
              "the stand-in is not marked as weights");
        CHECK(ggml_backend_buffer_get_size(a->buffer) == 0, "the stand-in is not size 0");
        CHECK(r->buffer == real, "the tensor with a real buffer was rewritten");
        CHECK(ggml_backend_test_live_buffer_count() == live_before + 2, "live buffers %zu -> %zu", live_before,
              ggml_backend_test_live_buffer_count());
    }
    CHECK(a->buffer == nullptr && b->buffer == nullptr, "a stand-in survived the scope on a tensor");
    CHECK(r->buffer == real, "the real buffer was removed by the scope");
    CHECK(ggml_backend_test_live_buffer_count() == live_before, "live buffers %zu after, %zu before",
          ggml_backend_test_live_buffer_count(), live_before);

    // an unwind through the scope restores the same way
    try {
        llama_measure_dummy_scope scope({
            { buft, wctx }
        });
        CHECK(a->buffer != nullptr, "no stand-in during the unwind arm");
        throw std::runtime_error("unwind");
    } catch (const std::runtime_error &) {
    }
    CHECK(a->buffer == nullptr && ggml_backend_test_live_buffer_count() == live_before, "unwind left a stand-in");

    ggml_backend_buffer_free(real);
    ggml_free(rctx);
    ggml_free(wctx);
}

int main() {
    test_guard();
    test_dummies();
    if (n_failed != 0) {
        fprintf(stderr, "%d check(s) failed\n", n_failed);
        return 1;
    }
    printf("test-load-measure-guards: ok\n");
    return 0;
}
