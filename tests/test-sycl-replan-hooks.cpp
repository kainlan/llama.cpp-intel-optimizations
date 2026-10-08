// Test: ggml_backend_sycl_synchronize_for_replan, ggml_backend_sycl_graph_invalidate
// (llama.cpp-zhcn) and ggml_backend_sycl_measure_backend_init; H6a.
//
// Both are reached by llama through the backend proc table, so this resolves them the
// same way.  A backend that is not a SYCL backend is refused (false / no-op) without
// touching any context.  The context arms need a real GPU backend (the SYCL backend
// admits no CPU device), so they run only when GGML_SYCL_TEST_REPLAN_DEVICE=1 is set
// together with a ONEAPI_DEVICE_SELECTOR naming a card: a context with nothing recorded
// waits clean and its invalidation is a no-op.  Without it the test exits 77, so a
// host run reports SKIPPED for those arms instead of passing them.
//
// What a recorded graph does under the clear, and which queues the wait reaches, is
// lead-run on a card (gate 30 censuses the queue list).
//
// Usage:
//   ./build/bin/test-sycl-replan-hooks                                   # refusals only
//   GGML_SYCL_TEST_REPLAN_DEVICE=1 ONEAPI_DEVICE_SELECTOR=level_zero:1 \
//       ./build/bin/test-sycl-replan-hooks                               # lead-run

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "test-skip.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

int g_failures = 0;

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            ++g_failures;                                                       \
        }                                                                       \
    } while (0)

typedef bool (*sync_fn)(ggml_backend_t);
typedef void (*invalidate_fn)(ggml_backend_t, const char *);
typedef ggml_backend_t (*measure_init_fn)(int);

}  // namespace

int main() {
    ggml_backend_reg_t reg = ggml_backend_sycl_reg();
    CHECK(reg != nullptr, "the SYCL backend registry exists");
    if (!reg) {
        return 1;
    }
    auto sync = (sync_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_synchronize_for_replan");
    auto inv  = (invalidate_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_graph_invalidate");
    auto measure_init =
        (measure_init_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_measure_backend_init");
    CHECK(measure_init != nullptr, "measure_backend_init is registered");
    CHECK((void *) measure_init == (void *) ggml_backend_sycl_measure_backend_init,
          "the proc is the exported function");
    if (measure_init) {
        CHECK(measure_init(-1) == nullptr, "a negative device index is refused");
        CHECK(measure_init(100000) == nullptr, "an out-of-range device index is refused");
    }
    CHECK(sync != nullptr, "synchronize_for_replan is registered");
    CHECK(inv != nullptr, "graph_invalidate is registered");
    CHECK((void *) sync == (void *) ggml_backend_sycl_synchronize_for_replan, "the proc is the exported function");
    CHECK((void *) inv == (void *) ggml_backend_sycl_graph_invalidate, "the proc is the exported function");
    if (!sync || !inv) {
        return 1;
    }

    // Refusals: no backend, and a backend that is not SYCL.
    CHECK(!sync(nullptr), "a null backend is refused");
    inv(nullptr, "test");
    {
        static const ggml_guid other_guid = { 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8,
                                              0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf, 0x10 };
        ggml_backend           fake{};
        fake.guid    = const_cast<ggml_guid_t>(&other_guid);
        fake.context = nullptr;
        CHECK(!sync(&fake), "a non-SYCL backend is refused");
        inv(&fake, "test");  // must not dereference the null context
    }

    const char * want_device = std::getenv("GGML_SYCL_TEST_REPLAN_DEVICE");
    if (!want_device || std::strcmp(want_device, "1") != 0) {
        std::printf("test-sycl-replan-hooks: GGML_SYCL_TEST_REPLAN_DEVICE not set; context arms SKIPPED\n");
        if (g_failures != 0) {
            std::fprintf(stderr, "test-sycl-replan-hooks: %d failure(s)\n", g_failures);
            return 1;
        }
        return LLAMA_TEST_EXIT_SKIP;
    }

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    CHECK(backend != nullptr, "a SYCL backend initialises on the selected device");
    if (!backend) {
        return 1;
    }

    CHECK(sync(backend), "a context with nothing recorded waits clean");
    CHECK(sync(backend), "the wait is repeatable");
    inv(backend, "test");
    inv(backend, nullptr);
    CHECK(sync(backend), "the context still waits clean after an invalidate that found nothing");

    // The tenant staging a recording or replaying call parks past its exit.  An eager exit releases
    // it; a call that recorded, and a call that only replayed a recorded graph (it begins no
    // recording, so the begin counter cannot tell it from an eager call), keep it, because the graph
    // has its addresses baked in; and the re-plan's invalidation then names the parked entry and
    // clears it, with no graph left for the predicate to find.
    CHECK(ggml_backend_sycl_test_tenant_staging_count(backend) == 0, "no tenant staging is parked at the start");
    CHECK(ggml_backend_sycl_test_park_tenant_staging(backend, 4096) == 1, "a tenant staging entry parks");
    CHECK(ggml_backend_sycl_test_graph_exit(backend, /*recorded=*/false, /*replayed=*/false), "an eager exit succeeds");
    CHECK(ggml_backend_sycl_test_tenant_staging_count(backend) == 0, "an eager exit releases the tenant staging");
    CHECK(ggml_backend_sycl_test_park_tenant_staging(backend, 4096) == 1, "a tenant staging entry parks again");
    CHECK(ggml_backend_sycl_test_graph_exit(backend, /*recorded=*/true, /*replayed=*/false),
          "a recording exit succeeds");
    CHECK(ggml_backend_sycl_test_tenant_staging_count(backend) == 1, "a recording exit keeps the tenant staging");
    CHECK(ggml_backend_sycl_test_graph_exit(backend, /*recorded=*/false, /*replayed=*/true),
          "a replay-only exit succeeds");
    CHECK(ggml_backend_sycl_test_tenant_staging_count(backend) == 1,
          "a replay-only exit keeps the tenant staging: it is not an eager call");
    inv(backend, "test");
    CHECK(ggml_backend_sycl_test_tenant_staging_count(backend) == 0,
          "the re-plan's invalidation names and clears the parked tenant staging");
    CHECK(sync(backend), "the context still waits clean after the parked entry was cleared");
    ggml_backend_free(backend);

    // The measure backend: its own object, not a SYCL one, with no synchronize slot.
    ggml_backend_t measure = ggml_backend_sycl_measure_backend_init(0);
    CHECK(measure != nullptr, "a measure backend is built for device 0");
    if (measure) {
        CHECK(!ggml_backend_is_sycl(measure), "the measure backend is not a SYCL backend");
        CHECK(measure->context == nullptr, "the measure backend holds no SYCL context");
        CHECK(measure->iface.synchronize == nullptr && measure->iface.graph_compute == nullptr &&
                  measure->iface.set_tensor_async == nullptr && measure->iface.event_record == nullptr,
              "every slot but get_name and free is NULL");
        CHECK(measure->iface.free != nullptr, "free is set");
        CHECK(ggml_backend_dev_type(ggml_backend_get_device(measure)) == GGML_BACKEND_DEVICE_TYPE_GPU,
              "its device is the SYCL device");
        CHECK(!sync(measure), "the replan hooks refuse a measure backend");
        inv(measure, "test");
        ggml_backend_synchronize(measure);  // a NULL slot returns
        ggml_backend_free(measure);
    }
    // Freeing it did not disturb a real backend's refcount: a later init and free pair up.
    backend = ggml_backend_sycl_init(0);
    CHECK(backend != nullptr, "a SYCL backend still initialises after a measure backend was freed");
    if (backend) {
        ggml_backend_free(backend);
    }

    if (g_failures != 0) {
        std::fprintf(stderr, "test-sycl-replan-hooks: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test-sycl-replan-hooks: all ok\n");
    return 0;
}
