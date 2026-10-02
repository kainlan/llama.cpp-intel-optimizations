// C7h-3 (zhcn): the plan-override guard and the weight stand-ins of a load-time measure. Header-only
// over src/llama-load-measure.h; no device and no model. The measure that uses them is
// test-measure-context's (CPU) and the lead-run GPU arm's.
//
// The guard installs the backend's plan override in its constructor and clears it in its destructor, once, on
// every exit, a throw included; a missing entry point and a refused install each leave it uninstalled with
// a name and never call clear. The stand-ins are size-0 weight buffers set on the tensors that have none
// and removed again, leaving a tensor that already had a buffer alone.
//
// It also runs the late-check fold (every answer of the backend lands in its own list: NOT_RECORDED is
// never a pass), the one refusal text, and the measure's context shape, which is tied to the auto-ubatch
// ladder's bottom rung.

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

// --- a multi-buft scope (M5) ---------------------------------------------------------------------
//
// A real model's weights sit in several buffer types at once (a device buffer per card, a pinned host one),
// and the measure's dummies must be one buffer per entry, each tensor on its own entry's. A size-0 buffer is
// made without the buffer type's allocator, so the buffer's type is the observable: it must be the type of the
// entry whose context the tensor belongs to. (ggml answers a size-0 request with a buffer on every path but a
// device that is shutting down, so the refused-buffer arm of the scope has no host test.)

static ggml_backend_buffer_type make_fake_buft(void * tag) {
    ggml_backend_buffer_type b = *ggml_backend_cpu_buffer_type();
    b.context                  = tag;
    return b;
}

static void test_dummies_multi_buft() {
    int                                          tag_a = 0, tag_b = 0;
    ggml_backend_buffer_type                     buft_a  = make_fake_buft(&tag_a);
    ggml_backend_buffer_type                     buft_b  = make_fake_buft(&tag_b);
    ggml_context *                               ca      = new_ctx();
    ggml_context *                               cb      = new_ctx();
    ggml_context *                               cc      = new_ctx();
    ggml_tensor *                                ta      = ggml_new_tensor_1d(ca, GGML_TYPE_F32, 4);
    ggml_tensor *                                tb      = ggml_new_tensor_1d(cb, GGML_TYPE_F32, 4);
    ggml_tensor *                                tc      = ggml_new_tensor_1d(cc, GGML_TYPE_F32, 4);
    const size_t                                 live0   = ggml_backend_test_live_buffer_count();
    const std::vector<llama_measure_dummy_entry> entries = {
        { &buft_a, ca },
        { &buft_b, cb },
        { &buft_a, cc }
    };

    {
        llama_measure_dummy_scope scope(entries);
        CHECK(!scope.failed(), "a stand-in was refused");
        CHECK(scope.n_buffers() == 3, "%zu buffers for three entries", scope.n_buffers());
        CHECK(ta->buffer != nullptr && tb->buffer != nullptr && tc->buffer != nullptr, "a tensor has no stand-in");
        CHECK(ta->buffer != tb->buffer && tb->buffer != tc->buffer && ta->buffer != tc->buffer,
              "two entries share one buffer");
        CHECK(ggml_backend_buffer_get_type(ta->buffer) == &buft_a &&
                  ggml_backend_buffer_get_type(tb->buffer) == &buft_b &&
                  ggml_backend_buffer_get_type(tc->buffer) == &buft_a,
              "a tensor's stand-in is not of its own entry's buffer type");
        CHECK(ggml_backend_test_live_buffer_count() == live0 + 3, "live buffers %zu",
              ggml_backend_test_live_buffer_count());
    }
    CHECK(ta->buffer == nullptr && tb->buffer == nullptr && tc->buffer == nullptr, "a stand-in survived");
    CHECK(ggml_backend_test_live_buffer_count() == live0, "live buffers %zu after, %zu before",
          ggml_backend_test_live_buffer_count(), live0);

    ggml_free(ca);
    ggml_free(cb);
    ggml_free(cc);
}

// --- the late-check fold (I2) -------------------------------------------------------------------

static std::vector<int32_t> g_late_seen;
static std::vector<int>     g_late_answers;  // indexed by device

static ggml_sycl_late_check_result fake_late_check(struct ggml_sycl_load_txn, int32_t device, uint64_t) {
    g_late_seen.push_back(device);
    return (ggml_sycl_late_check_result) g_late_answers[(size_t) device];
}

static llama_load_measure_device measured(int32_t device, bool host, size_t total) {
    llama_load_measure_device d;
    d.device = device;
    d.host   = host;
    d.total  = total;
    return d;
}

static void test_late_check_fold() {
    llama_sycl_l4_procs procs;
    procs.late_check = &fake_late_check;

    // every answer is its own list, the host tier is skipped, and nothing refuses
    g_late_seen.clear();
    g_late_answers = { GGML_SYCL_LATE_CHECK_EQUAL, GGML_SYCL_LATE_CHECK_SHRINK_ADMITTED,
                       GGML_SYCL_LATE_CHECK_NOT_RECORDED, GGML_SYCL_LATE_CHECK_EQUAL };
    {
        const std::vector<llama_load_measure_device> devs = { measured(0, false, 10), measured(1, false, 20),
                                                              measured(2, false, 30), measured(-1, true, 40),
                                                              measured(3, false, 50) };
        const llama_late_check_result                r    = llama_late_check_fold(procs, ggml_sycl_load_txn{ 9 }, devs);
        CHECK(r.checked, "a fold that ran reads as unchecked");
        CHECK(r.refusal.empty(), "refused: %s", r.refusal.c_str());
        CHECK(r.shrunk == std::vector<int32_t>({ 1 }), "shrunk list has %zu entries", r.shrunk.size());
        CHECK(r.not_recorded == std::vector<int32_t>({ 2 }), "not_recorded has %zu entries", r.not_recorded.size());
        CHECK(g_late_seen == std::vector<int32_t>({ 0, 1, 2, 3 }), "the host tier reached the backend");
    }

    // the first REFUSED ends the fold with the named text and no later device is asked
    g_late_seen.clear();
    g_late_answers = { GGML_SYCL_LATE_CHECK_EQUAL, GGML_SYCL_LATE_CHECK_REFUSED, GGML_SYCL_LATE_CHECK_EQUAL };
    {
        const std::vector<llama_load_measure_device> devs = { measured(0, false, 1), measured(1, false, 2),
                                                              measured(2, false, 3) };
        const llama_late_check_result                r    = llama_late_check_fold(procs, ggml_sycl_load_txn{ 9 }, devs);
        CHECK(r.refusal.rfind("[LOAD-PLAN] compute-slot measure failed at late on device 1: ", 0) == 0,
              "refusal text: %s", r.refusal.c_str());
        CHECK(r.refusal.size() > 10 && r.refusal.compare(r.refusal.size() - 10, 10, " (refused)") == 0,
              "refusal tail: %s", r.refusal.c_str());
        CHECK(g_late_seen == std::vector<int32_t>({ 0, 1 }), "the fold went on past a refusal");
    }

    // a table without the late-check proc compares nothing: every device is NOT_RECORDED, none passes
    {
        llama_sycl_l4_procs                          none;
        const std::vector<llama_load_measure_device> devs = { measured(0, false, 1), measured(1, false, 2) };
        const llama_late_check_result                r    = llama_late_check_fold(none, ggml_sycl_load_txn{ 9 }, devs);
        CHECK(r.refusal.empty() && r.shrunk.empty() && r.not_recorded == std::vector<int32_t>({ 0, 1 }),
              "a missing proc was read as a pass");
    }

    // a value outside the enum is NOT_RECORDED too
    g_late_answers = { 77 };
    {
        const std::vector<llama_load_measure_device> devs = { measured(0, false, 1) };
        const llama_late_check_result                r    = llama_late_check_fold(procs, ggml_sycl_load_txn{ 9 }, devs);
        CHECK(r.not_recorded == std::vector<int32_t>({ 0 }), "an unknown answer was read as a pass");
    }
}

static void test_refusal_text() {
    CHECK(llama_load_measure_refusal_text(GGML_SYCL_MEASURE_STAGE_PROBE, 2, "r") ==
              "[LOAD-PLAN] compute-slot measure failed at probe on device 2: r (refused)",
          "probe text");
    CHECK(llama_load_measure_refusal_text(GGML_SYCL_MEASURE_STAGE_CANDIDATE_B, 0, "r") ==
              "[LOAD-PLAN] compute-slot measure failed at admitted on device 0: r (refused)",
          "admitted text");
    CHECK(llama_load_measure_refusal_text(GGML_SYCL_MEASURE_STAGE_CANDIDATE_C, -1, "r") ==
              "[LOAD-PLAN] compute-slot measure failed at late on device -1: r (refused)",
          "late text");
}

// --- the measure's context shape (I3) -----------------------------------------------------------

static void test_measure_params_tie_to_the_ladder() {
    const llama_context_params p = llama_load_measure_context_params(0, 4096);
    CHECK(p.n_ubatch == llama_auto_ubatch_ladder[0], "n_ubatch %u, ladder bottom %u", p.n_ubatch,
          llama_auto_ubatch_ladder[0]);
    CHECK(p.n_batch >= p.n_ubatch, "n_batch %u below n_ubatch %u", p.n_batch, p.n_ubatch);
    CHECK(p.n_ctx == 4096, "n_ctx 0 did not become the training context: %u", p.n_ctx);
    CHECK(llama_load_measure_context_params(1234, 4096).n_ctx == 1234, "the caller's n_ctx was replaced");
    for (size_t i = 1; i < llama_auto_ubatch_ladder_size; ++i) {
        CHECK(llama_auto_ubatch_ladder[i - 1] < llama_auto_ubatch_ladder[i], "ladder is not ascending at %zu", i);
    }
}

int main() {
    test_guard();
    test_dummies();
    test_dummies_multi_buft();
    test_late_check_fold();
    test_refusal_text();
    test_measure_params_tie_to_the_ladder();
    if (n_failed != 0) {
        fprintf(stderr, "%d check(s) failed\n", n_failed);
        return 1;
    }
    printf("test-load-measure-guards: ok\n");
    return 0;
}
