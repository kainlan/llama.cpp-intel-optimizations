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
// ladder's bottom rung. And the quiet log scope a measure-only context holds: it drops every line below
// ERROR on its own thread, nests, ends with its owner, and leaves other threads alone.

#include "../ggml/src/ggml-backend-impl.h"
#include "../src/llama-impl.h"
#include "../src/llama-load-measure.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
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
static const ggml_sycl_measure_kv_shape * g_kv_shape   = nullptr;

static bool fake_install(uint64_t                           load_txn,
                         enum ggml_sycl_measure_stage       stage,
                         const ggml_sycl_measure_kv_shape * kv_shape) {
    g_installs++;
    g_txn      = load_txn;
    g_stage    = stage;
    g_kv_shape = kv_shape;
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
    const ggml_sycl_measure_kv_shape shape = { 4096, 512, 1, false, false };

    // installed: the entry sees the load, the stage and the KV shape, and the destructor clears once
    reset_fakes(true);
    {
        llama_measure_plan_override guard({ &fake_install, &fake_clear }, 42, GGML_SYCL_MEASURE_STAGE_CANDIDATE_B,
                                          &shape);
        CHECK(guard.installed(), "the override did not install");
        CHECK(guard.failure() == nullptr, "an installed guard names a failure");
        CHECK(g_installs == 1 && g_txn == 42 && g_stage == GGML_SYCL_MEASURE_STAGE_CANDIDATE_B,
              "install saw txn %llu stage %d", (unsigned long long) g_txn, (int) g_stage);
        CHECK(g_kv_shape == &shape, "install did not see the measure's KV shape");
        CHECK(g_clears == 0, "cleared while still held");
    }
    CHECK(g_clears == 1, "the destructor cleared %d times", g_clears);

    // a throw through the scope clears once
    reset_fakes(true);
    try {
        llama_measure_plan_override guard({ &fake_install, &fake_clear }, 7, GGML_SYCL_MEASURE_STAGE_PROBE, &shape);
        throw std::runtime_error("unwind");
    } catch (const std::runtime_error &) {
    }
    CHECK(g_installs == 1 && g_clears == 1, "unwind: %d installs, %d clears", g_installs, g_clears);

    // a refused install (a nest, or no plan staged) names itself and never clears
    reset_fakes(false);
    {
        llama_measure_plan_override guard({ &fake_install, &fake_clear }, 1, GGML_SYCL_MEASURE_STAGE_PROBE, &shape);
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
            llama_measure_plan_override guard(procs, 1, GGML_SYCL_MEASURE_STAGE_PROBE, &shape);
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

    // NOT_RECORDED is its own list (a SHRINK_ADMITTED is admitted, not a miss), the host tier is skipped, nothing refuses
    g_late_seen.clear();
    g_late_answers = { GGML_SYCL_LATE_CHECK_EQUAL, GGML_SYCL_LATE_CHECK_SHRINK_ADMITTED,
                       GGML_SYCL_LATE_CHECK_NOT_RECORDED, GGML_SYCL_LATE_CHECK_EQUAL };
    {
        const std::vector<llama_load_measure_device> devs = { measured(0, false, 10), measured(1, false, 20),
                                                              measured(2, false, 30), measured(-1, true, 40),
                                                              measured(3, false, 50) };
        const llama_late_check_result r = llama_late_check_fold(procs, ggml_sycl_load_txn{ 9 }, devs, 512);
        CHECK(r.n_ubatch == 512, "the fold dropped the ubatch: %u", r.n_ubatch);
        CHECK(r.refusal.empty(), "refused: %s", r.refusal.c_str());
        CHECK(r.not_recorded == std::vector<int32_t>({ 2 }), "not_recorded has %zu entries", r.not_recorded.size());
        CHECK(r.not_recorded_bytes == std::vector<size_t>({ 30 }), "the measured term beside device 2 is not its own");
        CHECK(g_late_seen == std::vector<int32_t>({ 0, 1, 2, 3 }), "the host tier reached the backend");
    }

    // the first REFUSED ends the fold with the named text and no later device is asked
    g_late_seen.clear();
    g_late_answers = { GGML_SYCL_LATE_CHECK_EQUAL, GGML_SYCL_LATE_CHECK_REFUSED, GGML_SYCL_LATE_CHECK_EQUAL };
    {
        const std::vector<llama_load_measure_device> devs = { measured(0, false, 1), measured(1, false, 2),
                                                              measured(2, false, 3) };
        const llama_late_check_result r = llama_late_check_fold(procs, ggml_sycl_load_txn{ 9 }, devs, 512);
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
        const llama_late_check_result r = llama_late_check_fold(none, ggml_sycl_load_txn{ 9 }, devs, 512);
        CHECK(r.refusal.empty() && r.not_recorded == std::vector<int32_t>({ 0, 1 }),
              "a missing proc was read as a pass");
        CHECK(r.not_recorded_bytes == std::vector<size_t>({ 1, 2 }), "the measured terms are not in device order");
    }

    // a value outside the enum is NOT_RECORDED too
    g_late_answers = { 77 };
    {
        const std::vector<llama_load_measure_device> devs = { measured(0, false, 1) };
        const llama_late_check_result r = llama_late_check_fold(procs, ggml_sycl_load_txn{ 9 }, devs, 512);
        CHECK(r.not_recorded == std::vector<int32_t>({ 0 }), "an unknown answer was read as a pass");
    }
}

// --- the admitted check (stage (b), llama.cpp-p6i0) ----------------------------------------------
//
// c(P), measured at the admitted placement, is compared per device with the probe bound C-hat measured before
// the pack, both in the reservation's units (the backend's term-bytes proc): c(P) <= C-hat is admitted and c(P)
// (never C-hat, and as the measure's raw total) is what is recorded; c(P) > C-hat refuses the load by name with
// both values. A device the backend declined to reserve for is neither compared nor recorded. The host tier is
// skipped, as in the late fold.

static void test_measure_n_ctx() {
    CHECK(llama_load_measure_n_ctx(0, 262144) == 262144, "n_ctx 0 did not become the training context");
    CHECK(llama_load_measure_n_ctx(4096, 262144) == 4096, "the caller's n_ctx was replaced");
    // the params use the same helper, so the recorded n_ctx is the one the measure ran at
    CHECK(llama_load_measure_context_params(0, 262144).n_ctx == llama_load_measure_n_ctx(0, 262144),
          "the measure's n_ctx and the recorded n_ctx disagree");
}

// The test's stand-in for the reservation's units: each nonzero chunk rounded up to 256 B, summed. The fold must
// compare what this answers, not the raw totals; the backend's own rule is pinned by test-zone-sizing.
static bool fake_term_bytes(const uint64_t * chunks, uint32_t n, uint64_t * out) {
    uint64_t sum = 0;
    for (uint32_t i = 0; i < n; ++i) {
        sum += (chunks[i] + 255) / 256 * 256;
    }
    *out = sum;
    return true;
}

static llama_load_measure_device chunked(int32_t device, bool host, std::vector<size_t> chunks) {
    llama_load_measure_device d;
    d.device      = device;
    d.host        = host;
    d.chunk_bytes = chunks;
    d.total       = 0;
    for (size_t c : chunks) {
        d.total += c;
    }
    return d;
}

static void test_admitted_fold() {
    llama_sycl_l4_procs procs;
    procs.term_bytes = &fake_term_bytes;

    // equal and shrink are admitted; each records c(P)'s raw total; the host tier is skipped
    {
        const std::vector<llama_load_measure_device> probe = { chunked(0, false, { 100 }), chunked(1, false, { 600 }),
                                                               chunked(-1, true, { 999 }) };
        const std::vector<llama_load_measure_device> admitted = { chunked(0, false, { 100 }),
                                                                  chunked(1, false, { 300 }),
                                                                  chunked(-1, true, { 5000 }) };
        const llama_admitted_check_result r = llama_admitted_check_fold(procs, probe, {}, admitted, 262144, 512);
        CHECK(r.refusal.empty(), "refused: %s", r.refusal.c_str());
        CHECK(r.terms.size() == 2, "%zu terms for two SYCL devices", r.terms.size());
        if (r.terms.size() == 2) {
            CHECK(r.terms[0].device == 0 && r.terms[0].reserved && r.terms[0].probe_term == 256 &&
                      r.terms[0].admitted_term == 256 && r.terms[0].admitted_bytes == 100,
                  "device 0's term is wrong");
            CHECK(r.terms[1].device == 1 && r.terms[1].probe_term == 768 && r.terms[1].admitted_term == 512 &&
                      r.terms[1].admitted_bytes == 300,
                  "a shrink must record c(P), not the probe bound");
        }
    }

    // the reservation's units decide, not the raw sums: 310 B fits under 512 B raw, but its two chunks occupy 768 B
    // against the 512 B reserved, so it is refused
    {
        const std::vector<llama_load_measure_device> probe    = { chunked(0, false, { 256, 256 }) };
        const std::vector<llama_load_measure_device> admitted = { chunked(0, false, { 300, 10 }) };
        const llama_admitted_check_result r = llama_admitted_check_fold(procs, probe, {}, admitted, 4096, 512);
        CHECK(!r.refusal.empty(), "a c(P) larger than the reservation in its own units was admitted");
        CHECK(r.refusal.find("768 B") != std::string::npos && r.refusal.find("512 B") != std::string::npos,
              "the refusal does not print both values in the reservation's units: %s", r.refusal.c_str());
    }
    // and the other way: 250 B is above 200 B raw, but both occupy one 256 B grain, so it is admitted
    {
        const std::vector<llama_load_measure_device> probe    = { chunked(0, false, { 200 }) };
        const std::vector<llama_load_measure_device> admitted = { chunked(0, false, { 250 }) };
        const llama_admitted_check_result r = llama_admitted_check_fold(procs, probe, {}, admitted, 4096, 512);
        CHECK(r.refusal.empty(), "a c(P) equal to the reservation in its own units was refused: %s", r.refusal.c_str());
        CHECK(r.terms.size() == 1 && r.terms[0].admitted_bytes == 250, "the recorded c(P) is not the raw total");
    }

    // c(P) > C-hat refuses by name, with both values, the n_ctx and the ubatch the measure ran at
    {
        const std::vector<llama_load_measure_device> probe = { chunked(0, false, { 100 }), chunked(1, false, { 512 }) };
        const std::vector<llama_load_measure_device> admitted = { chunked(0, false, { 100 }),
                                                                  chunked(1, false, { 513 }) };
        const llama_admitted_check_result r = llama_admitted_check_fold(procs, probe, {}, admitted, 262144, 512);
        CHECK(r.refusal.rfind("[LOAD-PLAN] compute-slot-exceeds-probe-bound on device 1: ", 0) == 0, "refusal text: %s",
              r.refusal.c_str());
        CHECK(r.refusal.find("c(P) 768 B > probe bound C-hat 512 B") != std::string::npos,
              "the refusal does not print both values: %s", r.refusal.c_str());
        CHECK(r.refusal.find("n_ctx 262144") != std::string::npos && r.refusal.find("ubatch 512") != std::string::npos,
              "the refusal does not print the measure's shape: %s", r.refusal.c_str());
        CHECK(r.refusal.size() > 10 && r.refusal.compare(r.refusal.size() - 10, 10, " (refused)") == 0,
              "refusal tail: %s", r.refusal.c_str());
        CHECK(r.terms.empty(), "a refused check still hands out terms to record");
    }

    // a device measured at the admitted placement with no probe bound has nothing that bounds it: refused
    {
        const std::vector<llama_load_measure_device> probe    = { chunked(0, false, { 100 }) };
        const std::vector<llama_load_measure_device> admitted = { chunked(0, false, { 100 }),
                                                                  chunked(1, false, { 1 }) };
        const llama_admitted_check_result r = llama_admitted_check_fold(procs, probe, {}, admitted, 4096, 512);
        CHECK(r.refusal.rfind("[LOAD-PLAN] compute-slot-exceeds-probe-bound on device 1: ", 0) == 0,
              "a device without a probe bound was admitted: %s", r.refusal.c_str());
        CHECK(r.refusal.find("no probe bound") != std::string::npos, "the refusal does not say why: %s",
              r.refusal.c_str());
    }

    // a device the backend declined to reserve for has no reservation to compare with: listed, not compared
    {
        const std::vector<llama_load_measure_device> probe = { chunked(0, false, { 100 }), chunked(1, false, { 100 }) };
        const std::vector<llama_load_measure_device> admitted = { chunked(0, false, { 100 }),
                                                                  chunked(1, false, { 5000 }) };
        const llama_admitted_check_result r = llama_admitted_check_fold(procs, probe, { 1 }, admitted, 4096, 512);
        CHECK(r.refusal.empty(), "a device with no reservation was compared and refused: %s", r.refusal.c_str());
        CHECK(r.terms.size() == 2 && r.terms[0].reserved && !r.terms[1].reserved,
              "the device with no reservation is not listed as such");
    }

    // without the reservation's units nothing can be compared: a reserved device refuses by name, an unreserved one
    // is still only listed
    {
        llama_sycl_l4_procs                          none;
        const std::vector<llama_load_measure_device> probe    = { chunked(0, false, { 100 }) };
        const std::vector<llama_load_measure_device> admitted = { chunked(0, false, { 100 }) };
        const llama_admitted_check_result r = llama_admitted_check_fold(none, probe, {}, admitted, 4096, 512);
        CHECK(r.refusal.find("cannot be sized in the reservation's units") != std::string::npos,
              "an unsized term was compared as raw bytes: %s", r.refusal.c_str());
        const llama_admitted_check_result u = llama_admitted_check_fold(none, probe, { 0 }, admitted, 4096, 512);
        CHECK(u.refusal.empty() && u.terms.size() == 1 && !u.terms[0].reserved,
              "an unreserved device was refused for a term nobody compares");
    }

    // nothing on a SYCL device: nothing to record, nothing refused
    {
        const llama_admitted_check_result r =
            llama_admitted_check_fold(procs, {}, {}, { chunked(-1, true, { 7 }) }, 4096, 512);
        CHECK(r.refusal.empty() && r.terms.empty(), "a host-only measure produced a term or a refusal");
    }
}

static std::vector<int32_t>  g_record_seen;
static std::vector<uint64_t> g_record_bytes;
static std::vector<uint32_t> g_record_n_ctx;
static uint64_t              g_record_txn = 0;
static bool                  g_record_ok  = true;

static bool fake_record(struct ggml_sycl_load_txn txn, int32_t device, uint64_t bytes, uint32_t n_ctx) {
    g_record_txn = txn.id;
    g_record_seen.push_back(device);
    g_record_bytes.push_back(bytes);
    g_record_n_ctx.push_back(n_ctx);
    return g_record_ok;
}

static llama_admitted_term admitted_term(int32_t device, bool reserved, size_t probe_term, size_t admitted_bytes) {
    llama_admitted_term t;
    t.device         = device;
    t.reserved       = reserved;
    t.probe_term     = probe_term;
    t.admitted_term  = admitted_bytes;
    t.admitted_bytes = admitted_bytes;
    return t;
}

static void test_admitted_record() {
    llama_admitted_check_result admitted;
    admitted.terms = { admitted_term(0, true, 100, 90), admitted_term(1, true, 200, 200),
                       admitted_term(2, false, 300, 300) };

    llama_sycl_l4_procs procs;
    procs.record_term = &fake_record;

    // each reserved device records c(P) at the measure's n_ctx, never the probe bound and never n_ctx 0; a device
    // with no reservation records nothing
    g_record_seen.clear();
    g_record_bytes.clear();
    g_record_n_ctx.clear();
    g_record_ok = true;
    CHECK(llama_admitted_record(procs, ggml_sycl_load_txn{ 7 }, admitted, 262144) == 2, "not every term recorded");
    CHECK(g_record_txn == 7, "the record went to transaction %llu", (unsigned long long) g_record_txn);
    CHECK(g_record_seen == std::vector<int32_t>({ 0, 1 }), "the record visited the wrong devices");
    CHECK(g_record_bytes == std::vector<uint64_t>({ 90, 200 }), "the record carried the probe bound, not c(P)");
    CHECK(g_record_n_ctx == std::vector<uint32_t>({ 262144, 262144 }), "the record carried the wrong n_ctx");

    // a record the backend refuses is counted as not recorded (the late check then says NOT_RECORDED)
    g_record_ok = false;
    CHECK(llama_admitted_record(procs, ggml_sycl_load_txn{ 7 }, admitted, 262144) == 0, "a refused record was counted");

    // a table without the record proc records nothing, and says so by its count
    llama_sycl_l4_procs none;
    CHECK(llama_admitted_record(none, ggml_sycl_load_txn{ 7 }, admitted, 262144) == 0,
          "a missing proc was counted as a record");
}

// The KV shape the override's re-fit sizes for is the measure context's own: every field from the params the
// measure builds the context with (llama.cpp-p6i0).
static void test_measure_kv_shape() {
    for (uint32_t n_ctx : { 0u, 4096u }) {
        llama_context_params params            = llama_load_measure_context_params(n_ctx, 262144);
        params.n_seq_max                       = 3;
        params.kv_unified                      = true;
        params.swa_full                        = true;
        const ggml_sycl_measure_kv_shape shape = llama_load_measure_kv_shape(params);
        CHECK(shape.n_ctx == params.n_ctx && shape.n_ubatch == params.n_ubatch && shape.n_seq_max == 3 &&
                  shape.kv_unified && shape.swa_full,
              "n_ctx %u: the shape is %u/%u/%u/%d/%d", n_ctx, shape.n_ctx, shape.n_ubatch, shape.n_seq_max,
              (int) shape.kv_unified, (int) shape.swa_full);
    }
    const ggml_sycl_measure_kv_shape d = llama_load_measure_kv_shape(llama_load_measure_context_params(0, 262144));
    CHECK(d.n_ctx == 262144 && !d.kv_unified == !llama_context_default_params().kv_unified &&
              !d.swa_full == !llama_context_default_params().swa_full,
          "the default shape is %u/%d/%d", d.n_ctx, (int) d.kv_unified, (int) d.swa_full);
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

static void test_not_recorded_text() {
    // 2133928064 B is 2035.07 MiB
    const std::string t = llama_late_check_not_recorded_text(2, 512, 2133928064);
    CHECK(t.find("device 2") != std::string::npos, "the text does not name the device: %s", t.c_str());
    CHECK(t.find("ubatch 512") != std::string::npos, "the text does not carry the ubatch: %s", t.c_str());
    CHECK(t.find("nothing was compared") != std::string::npos, "the text does not say nothing was compared: %s",
          t.c_str());
    CHECK(t.find("measured compute term 2035.1 MiB on device 2)") != std::string::npos,
          "the text does not carry the measured term in MiB: %s", t.c_str());
    CHECK(llama_late_check_not_recorded_text(2, 1024, 0).find("ubatch 1024") != std::string::npos,
          "the ubatch is fixed");
    CHECK(llama_late_check_not_recorded_text(2, 512, 1024 * 1024).find("term 1.0 MiB") != std::string::npos,
          "the measured term is fixed");
}

// --- the quiet log scope -------------------------------------------------------------------------

struct log_lines {
    std::vector<std::pair<int, std::string>> seen;

    static void callback(ggml_log_level level, const char * text, void * user_data) {
        static_cast<log_lines *>(user_data)->seen.emplace_back((int) level, text);
    }
};

static void emit_all_levels() {
    LLAMA_LOG_DEBUG("d\n");
    LLAMA_LOG_INFO("i\n");
    LLAMA_LOG_WARN("w\n");
    LLAMA_LOG_ERROR("e\n");
    LLAMA_LOG_CONT("c\n");
    LLAMA_LOG("n\n");
}

static void test_quiet_scope() {
    log_lines         lines;
    ggml_log_callback old_cb   = nullptr;
    void *            old_data = nullptr;
    llama_log_get(&old_cb, &old_data);
    llama_log_set(&log_lines::callback, &lines);

    // control: with no scope every level reaches the callback
    emit_all_levels();
    CHECK(lines.seen.size() == 6, "without a scope %zu of 6 lines arrived", lines.seen.size());

    // in a scope the ERROR line does, and the continuation that follows it (it belongs to that line); the
    // continuation after the INFO line and everything else is dropped
    lines.seen.clear();
    {
        llama_log_quiet_scope scope;
        emit_all_levels();
        CHECK(lines.seen.size() == 2 && lines.seen[0].first == (int) GGML_LOG_LEVEL_ERROR &&
                  lines.seen[0].second == "e\n" && lines.seen[1].first == (int) GGML_LOG_LEVEL_CONT &&
                  lines.seen[1].second == "c\n",
              "in a scope %zu lines arrived", lines.seen.size());

        // a continuation passes only straight after an ERROR: not after a dropped line, and not once another
        // level has intervened
        lines.seen.clear();
        LLAMA_LOG_INFO("i\n");
        LLAMA_LOG_CONT("c1\n");
        CHECK(lines.seen.empty(), "a continuation of a dropped INFO line passed");
        LLAMA_LOG_ERROR("e\n");
        LLAMA_LOG_CONT("c2\n");
        LLAMA_LOG_CONT("c3\n");
        CHECK(lines.seen.size() == 3 && lines.seen[1].second == "c2\n" && lines.seen[2].second == "c3\n",
              "an ERROR's continuations: %zu lines arrived", lines.seen.size());
        lines.seen.clear();
        LLAMA_LOG_WARN("w\n");
        LLAMA_LOG_CONT("c4\n");
        CHECK(lines.seen.empty(), "a continuation after an intervening WARN passed");

        // a nested scope does not end the outer one when it closes
        {
            llama_log_quiet_scope inner;
        }
        lines.seen.clear();
        LLAMA_LOG_INFO("still quiet\n");
        CHECK(lines.seen.empty(), "the outer scope ended with the inner one");

        // another thread is not silenced
        lines.seen.clear();
        std::thread other([]() { LLAMA_LOG_WARN("other thread\n"); });
        other.join();
        CHECK(lines.seen.size() == 1 && lines.seen[0].second == "other thread\n", "another thread was silenced");
    }

    // the scope ends with its owner, and a continuation that was riding on an ERROR does not outlive it
    // as a pass in the next one
    lines.seen.clear();
    emit_all_levels();
    CHECK(lines.seen.size() == 6, "after the scope %zu of 6 lines arrived", lines.seen.size());
    {
        llama_log_quiet_scope first;
        LLAMA_LOG_ERROR("e\n");
    }
    lines.seen.clear();
    {
        llama_log_quiet_scope second;
        LLAMA_LOG_CONT("stale\n");
        CHECK(lines.seen.empty(), "a continuation passed on an ERROR from an earlier scope");
    }

    // a throw through the scope closes it
    lines.seen.clear();
    try {
        llama_log_quiet_scope scope;
        throw std::runtime_error("unwind");
    } catch (const std::runtime_error &) {
    }
    LLAMA_LOG_INFO("after the throw\n");
    CHECK(lines.seen.size() == 1, "a throw left the scope open");

    llama_log_set(old_cb, old_data);
}

int main() {
    test_guard();
    test_dummies();
    test_dummies_multi_buft();
    test_late_check_fold();
    test_not_recorded_text();
    test_quiet_scope();
    test_refusal_text();
    test_measure_params_tie_to_the_ladder();
    test_measure_kv_shape();
    test_measure_n_ctx();
    test_admitted_fold();
    test_admitted_record();
    if (n_failed != 0) {
        fprintf(stderr, "%d check(s) failed\n", n_failed);
        return 1;
    }
    printf("test-load-measure-guards: ok\n");
    return 0;
}
