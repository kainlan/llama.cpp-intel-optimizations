#include "ggml-sycl.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <future>
#include <stdexcept>
#include <thread>

extern "C" void ggml_backend_test_block_next_unload();
extern "C" void ggml_backend_test_wait_unload_blocked();
extern "C" void ggml_backend_test_reentrant_mutation_on_next_unload();
extern "C" void ggml_backend_test_release_unload();
extern "C" size_t ggml_backend_test_active_calls(ggml_backend_reg_t reg);
extern "C" const char * ggml_backend_test_registry_state(ggml_backend_reg_t reg);
extern "C" void ggml_backend_sycl_test_seed_moe_module_state();
extern "C" bool ggml_backend_sycl_test_moe_module_state_clean();
extern "C" bool ggml_backend_sycl_test_allocate_predictor_scores();
extern "C" bool ggml_backend_sycl_test_seed_global_runtime_pinned_owners();
extern "C" void ggml_backend_sycl_test_fail_next_backend_publish();
extern "C" bool ggml_backend_sycl_test_hold_live_update(ggml_sycl_model_token model);
extern "C" void ggml_backend_sycl_test_release_live_update();

namespace {
enum class registry_fixture_mode {
    NORMAL, RESOLVER_THROW, SHUTDOWN_THROW, DEFERRED_REGISTER_ON_SHUTDOWN, PREPARE_CROSS_THREAD,
    PREPARE_REACTIVATE_THROW, LEGACY_THREE_HOOK_REACTIVATE,
    COMMIT_REACTIVATE_THROW, COMMIT_AND_ROLLBACK_THROW, FINALIZE_REACTIVATE_THROW
};
static registry_fixture_mode g_registry_fixture_mode = registry_fixture_mode::NORMAL;
static ggml_backend_reg_t     g_registry_fixture_reg = nullptr;
static int                    g_registry_fixture_recursive_registrations = 0;
static int                    g_registry_fixture_shutdowns = 0;
static int                    g_registry_fixture_cancels = 0;
static bool                   g_registry_fixture_reactivation_pending = false;
static ggml_backend_reg_t     g_cross_thread_fixture_reg = nullptr;
static std::mutex              g_device_callback_mutex;
static std::condition_variable g_device_callback_cv;
static bool                    g_device_callback_block = false;
static bool                    g_fixture_event_free_throw = false;
static bool                    g_device_callback_entered = false;
static int                     g_fixture_event_free_calls = 0;
static int                     g_fixture_buffer_free_calls = 0;
static std::mutex              g_commit_reactivate_mutex;
static std::condition_variable g_commit_reactivate_cv;
static bool                    g_commit_reactivate_block = false;
static bool                    g_commit_reactivate_entered = false;

static const char * registry_fixture_dev_name(ggml_backend_dev_t) { return "TEST-LIFECYCLE0"; }
static const char * registry_fixture_dev_description(ggml_backend_dev_t) {
    std::unique_lock<std::mutex> lock(g_device_callback_mutex);
    if (g_device_callback_block) {
        g_device_callback_entered = true;
        g_device_callback_cv.notify_all();
        g_device_callback_cv.wait(lock, [] { return !g_device_callback_block; });
    }
    return "registry lifecycle fixture";
}
static enum ggml_backend_dev_type registry_fixture_dev_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}
static ggml_backend_t registry_fixture_dev_init(ggml_backend_dev_t, const char *) { return nullptr; }
struct legacy_v2_event_layout { ggml_backend_dev_t device; void * context; };
static ggml_backend_event_t registry_fixture_event_new(ggml_backend_dev_t device) {
    static_assert(sizeof(legacy_v2_event_layout) == sizeof(ggml_backend_event), "event ABI v2 layout changed");
    return reinterpret_cast<ggml_backend_event_t>(new legacy_v2_event_layout{ device, nullptr });
}
static void registry_fixture_event_free(ggml_backend_dev_t, ggml_backend_event_t event) {
    ++g_fixture_event_free_calls;
    delete reinterpret_cast<legacy_v2_event_layout *>(event);
    if (g_fixture_event_free_throw) throw std::runtime_error("fixture event free failure");
}
struct fixture_owned_context {
    int * live;
    int * cleanup_calls = nullptr;
};
static void registry_fixture_context_cleanup(void * opaque) {
    auto * context = static_cast<fixture_owned_context *>(opaque);
    if (context->cleanup_calls) {
        ++*context->cleanup_calls;
    }
    --*context->live;
    delete context;
}
static void registry_fixture_buffer_free_owned(ggml_backend_buffer_t buffer) {
    ++g_fixture_buffer_free_calls;
    registry_fixture_context_cleanup(buffer->context);
    buffer->context = nullptr;
}
static void registry_fixture_buffer_free_passthrough(ggml_backend_buffer_t) {
    ++g_fixture_buffer_free_calls;
}
static void registry_fixture_buffer_free_throw(ggml_backend_buffer_t) {
    throw std::runtime_error("fixture buffer free failure");
}
static const char * registry_fixture_reg_name(ggml_backend_reg_t) { return "TEST-LIFECYCLE"; }
static size_t registry_fixture_reg_count(ggml_backend_reg_t) { return 1; }
static ggml_backend_dev_t registry_fixture_reg_get(ggml_backend_reg_t reg, size_t index) {
    return index == 0 ? static_cast<ggml_backend_dev_t>(reg->context) : nullptr;
}
static bool registry_fixture_can_unload() {
    ++g_registry_fixture_recursive_registrations;
    ggml_backend_register(g_registry_fixture_reg);
    return true;
}
static ggml_backend_reg_t deferred_hook_fixture();
static void registry_fixture_shutdown() {
    ++g_registry_fixture_shutdowns;
    if (g_registry_fixture_mode == registry_fixture_mode::DEFERRED_REGISTER_ON_SHUTDOWN) {
        ggml_backend_register(deferred_hook_fixture());
    }
    if (g_registry_fixture_mode == registry_fixture_mode::SHUTDOWN_THROW) {
        throw std::runtime_error("fixture partial shutdown");
    }
}
static void registry_fixture_cancel() { ++g_registry_fixture_cancels; }
static bool registry_fixture_prepare_reactivate() {
    if (g_registry_fixture_reactivation_pending) return false;
    g_registry_fixture_reactivation_pending = true;
    if (g_registry_fixture_mode == registry_fixture_mode::PREPARE_CROSS_THREAD) {
        static ggml_backend_device nested_dev{};
        static ggml_backend_reg nested_reg{};
        nested_dev.reg = &nested_reg;
        nested_reg.api_version = GGML_BACKEND_API_VERSION;
        nested_reg.iface.get_name = registry_fixture_reg_name;
        nested_reg.iface.get_device_count = registry_fixture_reg_count;
        nested_reg.iface.get_device = registry_fixture_reg_get;
        nested_reg.context = &nested_dev;
        g_cross_thread_fixture_reg = &nested_reg;
        auto nested = std::async(std::launch::async, [] { ggml_backend_register(g_cross_thread_fixture_reg); });
        if (nested.wait_for(std::chrono::seconds(2)) != std::future_status::ready) {
            throw std::runtime_error("cross-thread prepare reentry deadlocked");
        }
        nested.get();
    }
    if (g_registry_fixture_mode == registry_fixture_mode::PREPARE_REACTIVATE_THROW) {
        throw std::runtime_error("fixture partial prepare failure");
    }
    return true;
}
static void registry_fixture_commit_reactivate() {
    {
        std::unique_lock<std::mutex> lock(g_commit_reactivate_mutex);
        if (g_commit_reactivate_block) {
            g_commit_reactivate_entered = true;
            g_commit_reactivate_cv.notify_all();
            g_commit_reactivate_cv.wait(lock, [] { return !g_commit_reactivate_block; });
        }
    }
    if (g_registry_fixture_mode == registry_fixture_mode::COMMIT_REACTIVATE_THROW ||
        g_registry_fixture_mode == registry_fixture_mode::COMMIT_AND_ROLLBACK_THROW) {
        throw std::runtime_error("fixture reactivation commit failure");
    }
    if (g_registry_fixture_mode == registry_fixture_mode::LEGACY_THREE_HOOK_REACTIVATE) {
        g_registry_fixture_reactivation_pending = false;
    }
}
static void registry_fixture_finalize_reactivate() {
    g_registry_fixture_reactivation_pending = false;
    if (g_registry_fixture_mode == registry_fixture_mode::FINALIZE_REACTIVATE_THROW) {
        throw std::runtime_error("fixture finalize failure");
    }
}
static void registry_fixture_rollback_reactivate() {
    ++g_registry_fixture_cancels;
    g_registry_fixture_reactivation_pending = false;
    if (g_registry_fixture_mode == registry_fixture_mode::COMMIT_AND_ROLLBACK_THROW) {
        throw std::runtime_error("fixture rollback failure");
    }
}
static void * registry_fixture_resolve(ggml_backend_reg_t, const char * name) {
    if (g_registry_fixture_mode == registry_fixture_mode::RESOLVER_THROW) {
        throw std::runtime_error("fixture resolver failure");
    }
    if (std::strcmp(name, "ggml_backend_can_unload") == 0) return (void *) registry_fixture_can_unload;
    if (std::strcmp(name, "ggml_backend_shutdown") == 0) return (void *) registry_fixture_shutdown;
    if (std::strcmp(name, "ggml_backend_complete_unload") == 0 ||
        std::strcmp(name, "ggml_backend_cancel_unload") == 0) return (void *) registry_fixture_cancel;
    if (std::strcmp(name, "ggml_backend_prepare_reactivate") == 0) return (void *) registry_fixture_prepare_reactivate;
    if (std::strcmp(name, "ggml_backend_commit_reactivate") == 0) return (void *) registry_fixture_commit_reactivate;
    if (std::strcmp(name, "ggml_backend_finalize_reactivate") == 0) {
        return g_registry_fixture_mode == registry_fixture_mode::LEGACY_THREE_HOOK_REACTIVATE ? nullptr :
                                                                                                 (void *) registry_fixture_finalize_reactivate;
    }
    if (std::strcmp(name, "ggml_backend_rollback_reactivate") == 0) return (void *) registry_fixture_rollback_reactivate;
    return nullptr;
}

static ggml_backend_reg_t registry_fixture() {
    static ggml_backend_device dev{};
    static ggml_backend_reg reg{};
    static const bool initialized = [] {
        dev.iface.get_name = registry_fixture_dev_name;
        dev.iface.get_description = registry_fixture_dev_description;
        dev.iface.get_type = registry_fixture_dev_type;
        dev.iface.init_backend = registry_fixture_dev_init;
        dev.iface.event_new = registry_fixture_event_new;
        dev.iface.event_free = registry_fixture_event_free;
        dev.reg = &reg;
        reg.api_version = GGML_BACKEND_API_VERSION;
        reg.iface.get_name = registry_fixture_reg_name;
        reg.iface.get_device_count = registry_fixture_reg_count;
        reg.iface.get_device = registry_fixture_reg_get;
        reg.iface.get_proc_address = registry_fixture_resolve;
        reg.context = &dev;
        g_registry_fixture_reg = &reg;
        return true;
    }();
    (void) initialized;
    return &reg;
}

static ggml_backend_reg_t deferred_hook_fixture() {
    static ggml_backend_device dev{};
    static ggml_backend_reg reg{};
    static const bool initialized = [] {
        dev.iface.get_name = registry_fixture_dev_name;
        dev.iface.get_description = registry_fixture_dev_description;
        dev.iface.get_type = registry_fixture_dev_type;
        dev.iface.init_backend = registry_fixture_dev_init;
        dev.iface.event_new = registry_fixture_event_new;
        dev.iface.event_free = registry_fixture_event_free;
        dev.reg = &reg;
        reg.api_version = GGML_BACKEND_API_VERSION;
        reg.iface.get_name = [](ggml_backend_reg_t) { return "TEST-DEFERRED-HOOK"; };
        reg.iface.get_device_count = registry_fixture_reg_count;
        reg.iface.get_device = registry_fixture_reg_get;
        reg.context = &dev;
        return true;
    }();
    (void) initialized;
    return &reg;
}

static void phase(const char * name) {
    std::fprintf(stderr, "[sycl-runtime-wrapper] %s\n", name);
    std::fflush(stderr);
}

static bool fixture_expect(bool condition, const char * detail) {
    if (!condition) {
        std::fprintf(stderr, "[sycl-runtime-wrapper] assert failed: %s\n", detail);
        std::fflush(stderr);
    }
    return condition;
}

static bool run_registry_failure_fixture() {
    phase("generic fixture: construct pre-registry owners");
    auto reg = registry_fixture();
    static ggml_backend_buffer_type pre_registry_buft{};
    pre_registry_buft.device = static_cast<ggml_backend_dev_t>(reg->context);
    auto pre_registry_buffer = ggml_backend_buffer_init(&pre_registry_buft, {}, nullptr, 0);
    auto pre_registry_event = ggml_backend_event_new(static_cast<ggml_backend_dev_t>(reg->context));
    static ggml_backend_reg orphan_reg{};
    static ggml_backend_device orphan_dev{};
    static ggml_backend_buffer_type orphan_buft{};
    orphan_dev.reg = &orphan_reg;
    orphan_dev.iface.event_new = registry_fixture_event_new;
    orphan_dev.iface.event_free = registry_fixture_event_free;
    orphan_buft.device = &orphan_dev;
    static ggml_backend_reg failed_free_reg{};
    static ggml_backend_device failed_free_dev{};
    static ggml_backend_buffer_type failed_free_buft{};
    failed_free_dev.reg = &failed_free_reg;
    failed_free_dev.iface.event_new = registry_fixture_event_new;
    failed_free_dev.iface.event_free = registry_fixture_event_free;
    failed_free_buft.device = &failed_free_dev;
    ggml_backend_buffer_i orphan_iface{};
    orphan_iface.free_buffer = registry_fixture_buffer_free_passthrough;
    auto orphan_buffer = ggml_backend_buffer_init(&orphan_buft, orphan_iface, nullptr, 0);
    auto orphan_event = ggml_backend_event_new(&orphan_dev);
    auto failed_free_buffer = ggml_backend_buffer_init(&failed_free_buft, orphan_iface, nullptr, 0);
    auto failed_free_event = ggml_backend_event_new(&failed_free_dev);
    if (!pre_registry_buffer || !pre_registry_event || !orphan_buffer || !orphan_event || !failed_free_buffer || !failed_free_event) {
        if (!pre_registry_buffer) std::fprintf(stderr, "[sycl-runtime-wrapper] init returned null: pre_registry_buffer\n");
        if (!pre_registry_event) std::fprintf(stderr, "[sycl-runtime-wrapper] init returned null: pre_registry_event\n");
        if (!orphan_buffer) std::fprintf(stderr, "[sycl-runtime-wrapper] init returned null: orphan_buffer\n");
        if (!orphan_event) std::fprintf(stderr, "[sycl-runtime-wrapper] init returned null: orphan_event\n");
        if (!failed_free_buffer) std::fprintf(stderr, "[sycl-runtime-wrapper] init returned null: failed_free_buffer\n");
        if (!failed_free_event) std::fprintf(stderr, "[sycl-runtime-wrapper] init returned null: failed_free_event\n");
        std::fflush(stderr);
        return false;
    }
    phase("generic fixture: initialize registry with owners still pending");
    (void) ggml_backend_reg_count();
    g_fixture_event_free_calls = 0;
    g_fixture_buffer_free_calls = 0;
    phase("generic fixture: overlap lifecycle adoption and buffer free");
    ggml_backend_test_block_owner_adoption(true);
    auto adopting_register = std::async(std::launch::async, [reg] { ggml_backend_register(reg); });
    phase("generic fixture: await adoption barrier");
    for (int i = 0; i < 1000 && !ggml_backend_test_owner_adoption_blocked(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!ggml_backend_test_owner_adoption_blocked()) {
        ggml_backend_test_block_owner_adoption(false);
        adopting_register.get();
        return false;
    }
    if (ggml_backend_reg_by_name("TEST-LIFECYCLE") != nullptr) {
        ggml_backend_test_block_owner_adoption(false);
        adopting_register.get();
        return false;
    }
    phase("generic fixture: concurrent unload blocked by hidden publication");
    const size_t unload_attempts_before = ggml_backend_test_unload_attempts();
    auto publishing_unload = std::async(std::launch::async, [reg] { return ggml_backend_unload_checked(reg); });
    for (int i = 0; i < 1000 && ggml_backend_test_unload_attempts() == unload_attempts_before; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (ggml_backend_test_unload_attempts() == unload_attempts_before) {
        ggml_backend_test_block_owner_adoption(false);
        adopting_register.get();
        (void) publishing_unload.get();
        return false;
    }
    if (publishing_unload.wait_for(std::chrono::milliseconds(0)) != std::future_status::timeout) {
        const auto early_result = publishing_unload.get();
        std::fprintf(stderr,
                     "publication-raced unload completed early: result=%d visible=%d durable=%zu shutdowns=%d\n",
                     static_cast<int>(early_result), ggml_backend_reg_by_name("TEST-LIFECYCLE") != nullptr,
                     ggml_backend_test_durable_owners(reg), g_registry_fixture_shutdowns);
        ggml_backend_test_block_owner_adoption(false);
        adopting_register.get();
        return false;
    }
    const size_t transfer_attempts_before = ggml_backend_test_owner_transfer_attempts();
    auto overlapping_transfer = std::async(std::launch::async,
        [pre_registry_buffer] { return ggml_backend_buffer_set_type(pre_registry_buffer, &orphan_buft); });
    for (int i = 0; i < 1000 && ggml_backend_test_owner_transfer_attempts() == transfer_attempts_before; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (ggml_backend_test_owner_transfer_attempts() == transfer_attempts_before) {
        ggml_backend_test_block_owner_adoption(false);
        adopting_register.get();
        (void) overlapping_transfer.get();
        (void) publishing_unload.get();
        return false;
    }
    const size_t close_attempts_before = ggml_backend_test_owner_close_attempts();
    auto overlapping_free = std::async(std::launch::async,
        [pre_registry_buffer] { ggml_backend_buffer_free(pre_registry_buffer); });
    for (int i = 0; i < 1000 && ggml_backend_test_owner_close_attempts() == close_attempts_before; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (ggml_backend_test_owner_close_attempts() == close_attempts_before ||
        overlapping_free.wait_for(std::chrono::milliseconds(0)) != std::future_status::timeout) {
        ggml_backend_test_block_owner_adoption(false);
        adopting_register.get();
        (void) overlapping_transfer.get();
        overlapping_free.get();
        (void) publishing_unload.get();
        return false;
    }
    phase("generic fixture: release adoption barrier");
    ggml_backend_test_block_owner_adoption(false);
    adopting_register.get();
    phase("generic fixture: adoption publication joined");
    if (overlapping_transfer.get()) return false;
    phase("generic fixture: overlapping ownership transfer reconciled");
    if (publishing_unload.get() != GGML_BACKEND_UNLOAD_BUSY) return false;
    phase("generic fixture: publication-raced unload rejected by live owner");
    overlapping_free.get();
    phase("generic fixture: overlapping free joined");
    phase("generic fixture: verify adopted event durable owner");
    if (ggml_backend_test_durable_owners(reg) != 1) return false;
    // The unrelated owner remained pending. Its first publication failure must
    // not run reactivation hooks; retry adopts its exact live owners.
    orphan_reg.api_version = GGML_BACKEND_API_VERSION;
    orphan_reg.iface.get_name = registry_fixture_reg_name;
    orphan_reg.iface.get_device_count = registry_fixture_reg_count;
    orphan_reg.iface.get_device = registry_fixture_reg_get;
    orphan_reg.context = &orphan_dev;
    ggml_backend_test_fail_next_owner_adoption();
    ggml_backend_register(&orphan_reg);
    if (!fixture_expect(ggml_backend_test_durable_owners(&orphan_reg) == 0,
                        "failed first publication durable owner count != 0")) return false;
    if (!fixture_expect(std::strcmp(ggml_backend_test_registry_state(&orphan_reg), "REMOVED") == 0,
                        "failed first publication registry tombstone state != REMOVED")) return false;
    phase("generic fixture: retry failed first publication with exact live owners");
    ggml_backend_register(&orphan_reg);
    if (!fixture_expect(ggml_backend_test_durable_owners(&orphan_reg) == 2,
                        "retry publication durable owner count != 2") ||
        !fixture_expect(ggml_backend_unload_checked(&orphan_reg) == GGML_BACKEND_UNLOAD_BUSY,
                        "retry publication unload did not report BUSY")) return false;
    ggml_backend_buffer_free(orphan_buffer);
    ggml_backend_event_free(orphan_event);
    if (ggml_backend_unload_checked(&orphan_reg) != GGML_BACKEND_UNLOAD_OK) return false;

    g_fixture_buffer_free_calls = 0;
    g_fixture_event_free_calls = 0;
    phase("generic fixture: construct failed-publication owners for immediate free");
    failed_free_reg.api_version = GGML_BACKEND_API_VERSION;
    failed_free_reg.iface.get_name = registry_fixture_reg_name;
    failed_free_reg.iface.get_device_count = registry_fixture_reg_count;
    failed_free_reg.iface.get_device = registry_fixture_reg_get;
    failed_free_reg.context = &failed_free_dev;
    ggml_backend_test_fail_next_owner_adoption();
    ggml_backend_register(&failed_free_reg);
    if (!fixture_expect(ggml_backend_test_durable_owners(&failed_free_reg) == 0,
                        "immediate-free failed publication durable owner count != 0")) return false;
    phase("generic fixture: immediate free after failed first publication");
    ggml_backend_buffer_free(failed_free_buffer);
    if (!fixture_expect(g_fixture_buffer_free_calls == 1,
                        "failed first publication buffer cleanup callback count != 1")) {
        std::fprintf(stderr, "[sycl-runtime-wrapper] buffer cleanup diagnostics: calls=%d durable_owners=%zu registry_state=%s lookup_visible=%d\n",
                     g_fixture_buffer_free_calls, ggml_backend_test_durable_owners(&failed_free_reg),
                     ggml_backend_test_registry_state(&failed_free_reg), ggml_backend_reg_by_name("TEST-LIFECYCLE") == &failed_free_reg);
        return false;
    }
    phase("generic fixture: immediate free after failed first publication / free legacy-v2 event owner");
    ggml_backend_event_free(failed_free_event);
    if (!fixture_expect(g_fixture_event_free_calls == 1,
                        "failed first publication legacy-v2 event cleanup callback count != 1")) {
        std::fprintf(stderr, "[sycl-runtime-wrapper] legacy-v2 event cleanup diagnostics: calls=%d durable_owners=%zu registry_state=%s lookup_visible=%d\n",
                     g_fixture_event_free_calls, ggml_backend_test_durable_owners(&failed_free_reg),
                     ggml_backend_test_registry_state(&failed_free_reg), ggml_backend_reg_by_name("TEST-LIFECYCLE") == &failed_free_reg);
        return false;
    }
    if (!fixture_expect(ggml_backend_test_durable_owners(&failed_free_reg) == 0,
                        "failed first publication durable owner count changed after immediate free")) {
        std::fprintf(stderr, "[sycl-runtime-wrapper] durable owner diagnostics: count=%zu registry_state=%s lookup_visible=%d\n",
                     ggml_backend_test_durable_owners(&failed_free_reg), ggml_backend_test_registry_state(&failed_free_reg),
                     ggml_backend_reg_by_name("TEST-LIFECYCLE") == &failed_free_reg);
        return false;
    }
    if (!fixture_expect(std::strcmp(ggml_backend_test_registry_state(&failed_free_reg), "REMOVED") == 0,
                        "failed first publication registry tombstone state != REMOVED")) {
        std::fprintf(stderr, "[sycl-runtime-wrapper] registry state diagnostics: state=%s durable_owners=%zu lookup_visible=%d\n",
                     ggml_backend_test_registry_state(&failed_free_reg), ggml_backend_test_durable_owners(&failed_free_reg),
                     ggml_backend_reg_by_name("TEST-LIFECYCLE") == &failed_free_reg);
        return false;
    }
    phase("generic fixture: free adopted legacy-v2 event");
    ggml_backend_event_free(pre_registry_event);
    if (ggml_backend_test_durable_owners(reg) != 0) return false;
    phase("generic fixture: unload after all adopted owners freed");
    if (ggml_backend_unload_checked(reg) != GGML_BACKEND_UNLOAD_OK) return false;
    phase("generic fixture: cross-thread prepare reentry fails closed");
    g_registry_fixture_mode = registry_fixture_mode::PREPARE_CROSS_THREAD;
    ggml_backend_register(reg);
    if (ggml_backend_reg_by_name("TEST-LIFECYCLE") != reg ||
        ggml_backend_unload_checked(g_cross_thread_fixture_reg) != GGML_BACKEND_UNLOAD_OK ||
        ggml_backend_unload_checked(reg) != GGML_BACKEND_UNLOAD_OK) return false;
    phase("generic fixture: throwing partial prepare rolls back");
    g_registry_fixture_mode = registry_fixture_mode::PREPARE_REACTIVATE_THROW;
    ggml_backend_register(reg);
    if (g_registry_fixture_reactivation_pending || ggml_backend_reg_by_name("TEST-LIFECYCLE") != nullptr) return false;
    g_registry_fixture_mode = registry_fixture_mode::NORMAL;
    phase("generic fixture: adoption allocation failure leaves retryable tombstone");
    ggml_backend_test_fail_next_owner_adoption();
    ggml_backend_register(reg);
    if (ggml_backend_reg_by_name("TEST-LIFECYCLE") != nullptr) return false;
    phase("generic fixture: register after adopted-owner unload");
    ggml_backend_register(reg);
    phase("generic fixture: multi-buffer aggregate owns no duplicate lease");
    auto part_a = ggml_backend_buffer_init(&pre_registry_buft, {}, nullptr, 0);
    auto part_b = ggml_backend_buffer_init(&pre_registry_buft, {}, nullptr, 0);
    auto unrelated = ggml_backend_buffer_init(&pre_registry_buft, {}, nullptr, 0);
    ggml_backend_buffer_t parts[] = { part_a, part_b };
    auto aggregate = part_a && part_b ? ggml_backend_multi_buffer_alloc_buffer(parts, 2) : nullptr;
    if (!aggregate || !unrelated || ggml_backend_test_durable_owners(reg) != 3) return false;
    ggml_backend_buffer_free(aggregate);
    if (ggml_backend_test_durable_owners(reg) != 1) return false;
    ggml_backend_buffer_free(unrelated);
    if (ggml_backend_test_durable_owners(reg) != 0) return false;
    phase("generic fixture: owner-map allocation and throwing teardown recovery");
    int wrapper_owned_resources = 1;
    int wrapper_cleanup_calls = 0;
    ggml_backend_buffer_i owned_iface{};
    owned_iface.free_buffer = registry_fixture_buffer_free_passthrough;
    ggml_backend_test_fail_next_buffer_wrapper();
    if (ggml_backend_buffer_init_with_cleanup(&pre_registry_buft, owned_iface,
                                 new fixture_owned_context{ &wrapper_owned_resources, &wrapper_cleanup_calls }, 0,
                                 registry_fixture_context_cleanup) != nullptr ||
        wrapper_owned_resources != 0 || wrapper_cleanup_calls != 1 || ggml_backend_test_durable_owners(reg) != 0) return false;
    int acquire_owned_resources = 1;
    int acquire_cleanup_calls = 0;
    if (ggml_backend_buffer_init_with_cleanup(&orphan_buft, owned_iface,
                                 new fixture_owned_context{ &acquire_owned_resources, &acquire_cleanup_calls }, 0,
                                 registry_fixture_context_cleanup) != nullptr ||
        acquire_owned_resources != 0 || acquire_cleanup_calls != 1 || ggml_backend_test_durable_owners(reg) != 0) return false;
    int emplace_owned_resources = 1;
    int emplace_cleanup_calls = 0;
    ggml_backend_test_fail_next_buffer_emplace();
    if (ggml_backend_buffer_init_with_cleanup(&pre_registry_buft, owned_iface,
                                 new fixture_owned_context{ &emplace_owned_resources, &emplace_cleanup_calls }, 0,
                                 registry_fixture_context_cleanup) != nullptr ||
        emplace_owned_resources != 0 || emplace_cleanup_calls != 1 || ggml_backend_test_durable_owners(reg) != 0) return false;
    int refresh_owned_resources = 1;
    int refresh_cleanup_calls = 0;
    ggml_backend_test_fail_next_buffer_refresh();
    if (ggml_backend_buffer_init_with_cleanup(nullptr, owned_iface,
                                 new fixture_owned_context{ &refresh_owned_resources, &refresh_cleanup_calls }, 0,
                                 registry_fixture_context_cleanup) != nullptr ||
        refresh_owned_resources != 0 || refresh_cleanup_calls != 1 || ggml_backend_test_durable_owners(reg) != 0) return false;
    ggml_backend_test_fail_next_event_emplace();
    if (ggml_backend_event_new(static_cast<ggml_backend_dev_t>(reg->context)) != nullptr ||
        ggml_backend_test_durable_owners(reg) != 0) return false;
    ggml_backend_buffer_i throwing_iface{};
    throwing_iface.free_buffer = registry_fixture_buffer_free_throw;
    auto throwing_buffer = ggml_backend_buffer_init(&pre_registry_buft, throwing_iface, nullptr, 0);
    if (!throwing_buffer || ggml_backend_test_durable_owners(reg) != 1) return false;
    ggml_backend_buffer_free(throwing_buffer);
    g_fixture_event_free_throw = true;
    auto throwing_event = ggml_backend_event_new(static_cast<ggml_backend_dev_t>(reg->context));
    if (!throwing_event || ggml_backend_test_durable_owners(reg) != 1) return false;
    ggml_backend_event_free(throwing_event);
    g_fixture_event_free_throw = false;
    if (ggml_backend_test_durable_owners(reg) != 0) return false;
    const size_t initial_reg_count = ggml_backend_reg_count();
#if defined(GGML_SYCL_RUNTIME_MODULE)
    if (initial_reg_count < 1) return false;
#else
    // Static first-use registration must finish publishing built-in SYCL/CPU
    // entries before the custom fixture is admitted.
    if (initial_reg_count < 2 || ggml_backend_reg_by_name("SYCL") == nullptr) return false;
#endif

    phase("generic fixture: resolver failure tombstone");
    g_registry_fixture_mode = registry_fixture_mode::RESOLVER_THROW;
    if (ggml_backend_unload_checked(reg) != GGML_BACKEND_UNLOAD_BUSY ||
        ggml_backend_reg_by_name("TEST-LIFECYCLE") != nullptr) return false;
    for (int i = 0; i < 1000; ++i) {
        if (ggml_backend_reg_by_name("TEST-LIFECYCLE") != nullptr) return false;
    }
    phase("generic fixture: retry resolver-failure unload");
    g_registry_fixture_mode = registry_fixture_mode::NORMAL;
    if (ggml_backend_unload_checked(reg) != GGML_BACKEND_UNLOAD_OK) return false;

    phase("generic fixture: enumerate reactivated generation");
    const size_t before_reg_count = ggml_backend_reg_count();
    const size_t before_dev_count = ggml_backend_dev_count();
    ggml_backend_register(reg);
    const size_t reg_count = ggml_backend_reg_count();
    const size_t dev_count = ggml_backend_dev_count();
    if (reg_count != before_reg_count + 1 || dev_count != before_dev_count + 1) return false;
    size_t reg_index = reg_count;
    size_t dev_index = dev_count;
    for (size_t i = 0; i < reg_count; ++i) if (ggml_backend_reg_get(i) == reg) reg_index = i;
    for (size_t i = 0; i < dev_count; ++i) if (ggml_backend_dev_get(i) == reg->context) dev_index = i;

    phase("generic fixture: shutdown failure tombstone");
    g_registry_fixture_mode = registry_fixture_mode::SHUTDOWN_THROW;
    if (reg_index == reg_count || dev_index == dev_count ||
        ggml_backend_unload_checked(reg) != GGML_BACKEND_UNLOAD_BUSY ||
        ggml_backend_reg_get(reg_index) != nullptr || ggml_backend_dev_get(dev_index) != nullptr ||
        ggml_backend_reg_by_name("TEST-LIFECYCLE") != nullptr) return false;
    phase("generic fixture: retry shutdown-failure unload");
    g_registry_fixture_mode = registry_fixture_mode::NORMAL;
    if (ggml_backend_unload_checked(reg) != GGML_BACKEND_UNLOAD_OK) return false;
    phase("generic fixture: active callback drain");
    ggml_backend_register(reg);
    if (ggml_backend_reg_by_name("TEST-LIFECYCLE") != reg) return false;
    {
        std::lock_guard<std::mutex> lock(g_device_callback_mutex);
        g_device_callback_block = true;
        g_device_callback_entered = false;
    }
    auto callback = std::async(std::launch::async, [&] {
        return ggml_backend_dev_description(static_cast<ggml_backend_dev_t>(reg->context));
    });
    phase("generic fixture: await active callback entry");
    {
        std::unique_lock<std::mutex> lock(g_device_callback_mutex);
        if (!g_device_callback_cv.wait_for(lock, std::chrono::seconds(10),
                                           [] { return g_device_callback_entered; })) {
            g_device_callback_block = false;
            g_device_callback_cv.notify_all();
            lock.unlock();
            callback.get();
            return false;
        }
    }
    phase("generic fixture: launch unload into bounded callback drain");
    auto blocked_unload = std::async(std::launch::async, [&] { return ggml_backend_unload_checked(reg); });
    const bool unload_waited_for_callback =
        blocked_unload.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout;
    phase("generic fixture: release active callback");
    {
        std::lock_guard<std::mutex> lock(g_device_callback_mutex);
        g_device_callback_block = false;
        g_device_callback_cv.notify_all();
    }
    const char * callback_result = callback.get();
    phase("generic fixture: active callback joined");
    const auto blocked_unload_result = blocked_unload.get();
    phase("generic fixture: callback-drain unload joined");
    if (!unload_waited_for_callback || !callback_result ||
        blocked_unload_result != GGML_BACKEND_UNLOAD_OK) return false;

    phase("generic fixture: bounded stalled callback cancellation");
    ggml_backend_register(reg);
    {
        std::lock_guard<std::mutex> lock(g_device_callback_mutex);
        g_device_callback_block = true;
        g_device_callback_entered = false;
    }
    auto stalled_callback = std::async(std::launch::async, [&] {
        return ggml_backend_dev_description(static_cast<ggml_backend_dev_t>(reg->context));
    });
    phase("generic fixture: await deliberately stalled callback entry");
    {
        std::unique_lock<std::mutex> lock(g_device_callback_mutex);
        if (!g_device_callback_cv.wait_for(lock, std::chrono::seconds(10),
                                           [] { return g_device_callback_entered; })) {
            g_device_callback_block = false;
            g_device_callback_cv.notify_all();
            lock.unlock();
            stalled_callback.get();
            return false;
        }
    }
    phase("generic fixture: run bounded stalled-callback unload");
    const auto stalled_unload_result = ggml_backend_unload_checked(reg);
    {
        std::lock_guard<std::mutex> lock(g_device_callback_mutex);
        g_device_callback_block = false;
        g_device_callback_cv.notify_all();
    }
    phase("generic fixture: released stalled callback");
    if (stalled_unload_result != GGML_BACKEND_UNLOAD_BUSY || !stalled_callback.get() ||
        ggml_backend_unload_checked(reg) != GGML_BACKEND_UNLOAD_OK) return false;
    phase("generic fixture: stalled callback and retry unload joined");

    phase("generic fixture: unload-hook deferred registration publishes eventually");
    g_registry_fixture_mode = registry_fixture_mode::DEFERRED_REGISTER_ON_SHUTDOWN;
    ggml_backend_register(reg);
    auto * deferred_reg = deferred_hook_fixture();
    if (ggml_backend_unload_checked(reg) != GGML_BACKEND_UNLOAD_OK ||
        ggml_backend_reg_by_name("TEST-DEFERRED-HOOK") != deferred_reg ||
        ggml_backend_unload_checked(deferred_reg) != GGML_BACKEND_UNLOAD_OK) return false;
    g_registry_fixture_mode = registry_fixture_mode::NORMAL;

    phase("generic fixture: hidden blocked reactivation");
    {
        std::lock_guard<std::mutex> lock(g_commit_reactivate_mutex);
        g_commit_reactivate_block = true;
        g_commit_reactivate_entered = false;
    }
    auto blocked_reactivation = std::async(std::launch::async, [&] { ggml_backend_register(reg); });
    {
        std::unique_lock<std::mutex> lock(g_commit_reactivate_mutex);
        g_commit_reactivate_cv.wait(lock, [] { return g_commit_reactivate_entered; });
    }
    const bool reactivation_hidden = ggml_backend_reg_by_name("TEST-LIFECYCLE") == nullptr;
    {
        std::lock_guard<std::mutex> lock(g_commit_reactivate_mutex);
        g_commit_reactivate_block = false;
        g_commit_reactivate_cv.notify_all();
    }
    blocked_reactivation.get();
    if (!reactivation_hidden || ggml_backend_unload_checked(reg) != GGML_BACKEND_UNLOAD_OK) return false;

    phase("generic fixture: legacy 3-hook reactivation ABI");
    g_registry_fixture_mode = registry_fixture_mode::LEGACY_THREE_HOOK_REACTIVATE;
    ggml_backend_register(reg);
    if (ggml_backend_reg_by_name("TEST-LIFECYCLE") != reg || ggml_backend_unload_checked(reg) != GGML_BACKEND_UNLOAD_OK) {
        return false;
    }
    g_registry_fixture_mode = registry_fixture_mode::NORMAL;

    phase("generic fixture: throwing commit rollback and recovery");
    g_registry_fixture_mode = registry_fixture_mode::COMMIT_AND_ROLLBACK_THROW;
    ggml_backend_register(reg);
    if (ggml_backend_reg_by_name("TEST-LIFECYCLE") != nullptr) return false;
    g_registry_fixture_mode = registry_fixture_mode::NORMAL;
    ggml_backend_register(reg);
    if (ggml_backend_reg_by_name("TEST-LIFECYCLE") != reg ||
        ggml_backend_unload_checked(reg) != GGML_BACKEND_UNLOAD_OK) return false;
    phase("generic fixture: throwing finalize rolls committed state forward");
    g_registry_fixture_mode = registry_fixture_mode::FINALIZE_REACTIVATE_THROW;
    ggml_backend_register(reg);
    if (ggml_backend_reg_by_name("TEST-LIFECYCLE") != reg) return false;
    auto finalize_owner = ggml_backend_buffer_init(&pre_registry_buft, {}, nullptr, 0);
    if (!finalize_owner || ggml_backend_unload_checked(reg) != GGML_BACKEND_UNLOAD_BUSY) return false;
    ggml_backend_buffer_free(finalize_owner);
    g_registry_fixture_mode = registry_fixture_mode::NORMAL;
    if (ggml_backend_unload_checked(reg) != GGML_BACKEND_UNLOAD_OK) return false;
    return g_registry_fixture_recursive_registrations >= 2 && g_registry_fixture_shutdowns >= 2 &&
           g_registry_fixture_cancels >= 2;
}
} // namespace

int main() {
    phase("generic registry tombstone/failure fixture");
    if (!run_registry_failure_fixture()) {
        std::fprintf(stderr, "generic registry lifecycle fixture failed\n");
        return 1;
    }
#if defined(GGML_SYCL_RUNTIME_MODULE)
    // Exercise real registry late registration and module lifetime: load,
    // unregister/unload, then reload before resolving lifecycle operations.
    phase("initial module load");
    auto * reg = ggml_backend_load(GGML_SYCL_RUNTIME_MODULE);
    if (!reg) {
        std::fprintf(stderr, "failed to register SYCL backend module\n");
        return 1;
    }
    auto shutdown = reinterpret_cast<decltype(&ggml_backend_sycl_shutdown)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_shutdown"));
    if (!shutdown) {
        std::fprintf(stderr, "missing registry procedure ggml_backend_sycl_shutdown\n");
        return 1;
    }
    auto seed_moe_state = reinterpret_cast<decltype(&ggml_backend_sycl_test_seed_moe_module_state)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_test_seed_moe_module_state"));
    if (!seed_moe_state) {
        std::fprintf(stderr, "missing MoE reload-state seed procedure\n");
        return 1;
    }
    auto seed_cpu_retained = reinterpret_cast<bool (*)()>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_test_seed_cpu_retained"));
    if (!seed_cpu_retained) {
        std::fprintf(stderr, "missing CPU-retained reload fixture\n");
        return 1;
    }
    auto seed_global_runtime_pinned_owners = reinterpret_cast<decltype(&ggml_backend_sycl_test_seed_global_runtime_pinned_owners)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_test_seed_global_runtime_pinned_owners"));
    if (!seed_global_runtime_pinned_owners) {
        std::fprintf(stderr, "missing global runtime/pinned owner seed procedure\n");
        return 1;
    }
    auto initial_get_device_memory = reinterpret_cast<decltype(&ggml_backend_sycl_get_device_memory)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_get_device_memory"));
    size_t free_before_cache = 0;
    size_t total_before_cache = 0;
    if (!initial_get_device_memory) {
        std::fprintf(stderr, "missing initial device-memory procedure\n");
        return 1;
    }
    initial_get_device_memory(0, &free_before_cache, &total_before_cache);
    if (!seed_cpu_retained()) {
        std::fprintf(stderr, "failed to seed CPU-retained reload fixture\n");
        return 1;
    }
    if (!seed_global_runtime_pinned_owners()) {
        std::fprintf(stderr, "failed to seed global runtime/pinned owner fixture\n");
        return 1;
    }
    auto allocate_predictor_scores = reinterpret_cast<decltype(&ggml_backend_sycl_test_allocate_predictor_scores)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_test_allocate_predictor_scores"));
    if (!allocate_predictor_scores || !allocate_predictor_scores()) {
        std::fprintf(stderr, "failed to create real predictor unified allocation\n");
        return 1;
    }
    size_t free_with_cache = 0;
    size_t total_with_cache = 0;
    initial_get_device_memory(0, &free_with_cache, &total_with_cache);
    constexpr size_t minimum_measured_drop = 1ull * 1024ull * 1024ull * 1024ull;
    if (total_with_cache != total_before_cache || free_before_cache < free_with_cache + minimum_measured_drop) {
        std::fprintf(stderr, "cache allocation did not produce a meaningful measured VRAM drop: before=%zu after=%zu\n",
                     free_before_cache, free_with_cache);
        return 1;
    }
    const size_t measured_cache_drop = free_before_cache - free_with_cache;
    auto saved_model_load_begin = reinterpret_cast<decltype(&ggml_backend_sycl_model_load_begin)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_model_load_begin"));
    auto saved_host_compute = reinterpret_cast<decltype(&ggml_backend_sycl_host_compute_buffer_type)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_host_compute_buffer_type"));
    auto saved_push_kv = reinterpret_cast<decltype(&ggml_backend_sycl_push_kv_layer_mask_from_dev)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_push_kv_layer_mask_from_dev"));
    auto saved_dev = ggml_backend_reg_dev_get(reg, 0);
    auto saved_buft = saved_host_compute ? saved_host_compute(0) : nullptr;
    if (!saved_model_load_begin || !saved_host_compute || !saved_push_kv || !saved_dev || !saved_buft) {
        std::fprintf(stderr, "missing saved mutation/buffer procedures\n");
        return 1;
    }
    auto fail_next_arena_free = reinterpret_cast<void (*)()>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_test_fail_next_arena_free"));
    auto fail_next_shutdown_clean = reinterpret_cast<void (*)()>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_test_fail_next_shutdown_clean"));
    auto fail_next_registry_stage = reinterpret_cast<void (*)()>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_test_fail_next_registry_stage"));
    using shutdown_owner_census_fn = bool (*)(uint64_t out[4]);
    auto shutdown_owner_census = reinterpret_cast<shutdown_owner_census_fn>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_test_shutdown_owner_census"));
    using shutdown_runtime_alloc_census_fn = bool (*)(uint64_t out[8]);
    auto shutdown_runtime_alloc_census = reinterpret_cast<shutdown_runtime_alloc_census_fn>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_test_shutdown_runtime_alloc_census"));
    auto block_next_kv_push = reinterpret_cast<void (*)()>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_test_block_next_kv_push"));
    auto wait_kv_push_blocked = reinterpret_cast<void (*)()>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_test_wait_kv_push_blocked"));
    auto release_kv_push = reinterpret_cast<void (*)()>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_test_release_kv_push"));
    if (!fail_next_arena_free || !fail_next_shutdown_clean || !fail_next_registry_stage || !shutdown_owner_census ||
        !shutdown_runtime_alloc_census || !block_next_kv_push || !wait_kv_push_blocked || !release_kv_push) {
        std::fprintf(stderr, "missing deterministic unload/admission procedures\n");
        return 1;
    }
    seed_moe_state();
    phase("initial module unload drains saved KV mutation and preserves failure closure");
    shutdown = nullptr;
    fail_next_arena_free();
    fail_next_shutdown_clean();
    fail_next_registry_stage();
    block_next_kv_push();
    const uint8_t test_kv_mask = 1;
    auto blocked_push = std::async(std::launch::async, [&] { saved_push_kv(saved_dev, &test_kv_mask, 1); });
    wait_kv_push_blocked();
    auto draining_unload = std::async(std::launch::async, [&] { return ggml_backend_unload_checked(reg); });
    if (draining_unload.wait_for(std::chrono::milliseconds(100)) != std::future_status::timeout) {
        std::fprintf(stderr, "unload did not drain saved KV mutation\n");
        release_kv_push();
        return 1;
    }
    release_kv_push();
    blocked_push.get();
    if (draining_unload.get() != GGML_BACKEND_UNLOAD_BUSY) {
        std::fprintf(stderr, "arena free failure did not fail unload transactionally\n");
        return 1;
    }
    ggml_sycl_load_txn stale_txn{};
    if (saved_model_load_begin(&stale_txn) != GGML_SYCL_LIFECYCLE_LOAD_BUSY) {
        std::fprintf(stderr, "failure-window saved model-load procedure reopened admission\n");
        return 1;
    }
    // shutdown_resources() may legitimately free arena bytes before the later
    // retryable dirty postcondition runs. Assert retained owner authority, not
    // transient free-VRAM deltas.
    uint64_t owner_census_before_dirty[4]{};
    if (!shutdown_owner_census(owner_census_before_dirty)) {
        std::fprintf(stderr, "shutdown owner census unavailable before dirty retry\n");
        return 1;
    }
    phase("retry module unload detects shutdown-clean failure");
    if (ggml_backend_unload_checked(reg) != GGML_BACKEND_UNLOAD_BUSY) {
        std::fprintf(stderr, "shutdown clean=false did not retain owners transactionally\n");
        return 1;
    }
    uint64_t owner_census_after_dirty[4]{};
    if (!shutdown_owner_census(owner_census_after_dirty)) {
        std::fprintf(stderr, "shutdown owner census unavailable after dirty retry\n");
        return 1;
    }
    uint64_t runtime_alloc_census_after_dirty[8]{};
    if (!shutdown_runtime_alloc_census(runtime_alloc_census_after_dirty)) {
        std::fprintf(stderr, "shutdown runtime allocation census unavailable after dirty retry\n");
        return 1;
    }
    if (runtime_alloc_census_after_dirty[3] != 0 ||
        runtime_alloc_census_after_dirty[0] != runtime_alloc_census_after_dirty[2]) {
        std::fprintf(stderr,
                     "dirty shutdown retry retained runtime allocation owners: total=%llu host=%llu cache_owned=%llu offload=%llu ptr=0x%llx size=%llu owner=%s device=%lld\n",
                     static_cast<unsigned long long>(runtime_alloc_census_after_dirty[0]),
                     static_cast<unsigned long long>(runtime_alloc_census_after_dirty[1]),
                     static_cast<unsigned long long>(runtime_alloc_census_after_dirty[2]),
                     static_cast<unsigned long long>(runtime_alloc_census_after_dirty[3]),
                     static_cast<unsigned long long>(runtime_alloc_census_after_dirty[4]),
                     static_cast<unsigned long long>(runtime_alloc_census_after_dirty[5]),
                     runtime_alloc_census_after_dirty[7] ? "cache-pinned" : "external",
                     static_cast<long long>(runtime_alloc_census_after_dirty[6] == UINT64_MAX ? -1 : runtime_alloc_census_after_dirty[6]));
        return 1;
    }
    if (owner_census_before_dirty[0] == 0 || owner_census_after_dirty[0] != owner_census_before_dirty[0]) {
        std::fprintf(stderr,
                     "dirty shutdown retry lost retained cache owners: before=%llu dirty=%llu active_before=%llu active_dirty=%llu chunks_before=%llu chunks_dirty=%llu\n",
                     static_cast<unsigned long long>(owner_census_before_dirty[0]),
                     static_cast<unsigned long long>(owner_census_after_dirty[0]),
                     static_cast<unsigned long long>(owner_census_before_dirty[1]),
                     static_cast<unsigned long long>(owner_census_after_dirty[1]),
                     static_cast<unsigned long long>(owner_census_before_dirty[2]),
                     static_cast<unsigned long long>(owner_census_after_dirty[2]));
        return 1;
    }
    if (saved_model_load_begin(&stale_txn) != GGML_SYCL_LIFECYCLE_LOAD_BUSY) {
        std::fprintf(stderr, "clean-failure window reopened saved mutation admission\n");
        return 1;
    }
    phase("final retry module unload after retained owners");
    if (ggml_backend_unload_checked(reg) != GGML_BACKEND_UNLOAD_OK) {
        std::fprintf(stderr, "retained owner retry did not complete unload\n");
        return 1;
    }
    uint64_t owner_census_after_retry[4]{};
    if (!shutdown_owner_census(owner_census_after_retry)) {
        std::fprintf(stderr, "shutdown owner census unavailable after retry success\n");
        return 1;
    }
    if (owner_census_after_retry[0] != 0) {
        std::fprintf(stderr, "retained owner retry did not clear owner census: owners=%llu active=%llu chunks=%llu queues=%llu\n",
                     static_cast<unsigned long long>(owner_census_after_retry[0]),
                     static_cast<unsigned long long>(owner_census_after_retry[1]),
                     static_cast<unsigned long long>(owner_census_after_retry[2]),
                     static_cast<unsigned long long>(owner_census_after_retry[3]));
        return 1;
    }
    uint64_t runtime_alloc_census_after_retry[8]{};
    if (!shutdown_runtime_alloc_census(runtime_alloc_census_after_retry)) {
        std::fprintf(stderr, "shutdown runtime allocation census unavailable after retry success\n");
        return 1;
    }
    if (runtime_alloc_census_after_retry[0] != 0 || runtime_alloc_census_after_retry[2] != 0 ||
        runtime_alloc_census_after_retry[3] != 0) {
        std::fprintf(stderr,
                     "retained owner retry did not clear runtime allocation census: total=%llu cache_owned=%llu offload=%llu\n",
                     static_cast<unsigned long long>(runtime_alloc_census_after_retry[0]),
                     static_cast<unsigned long long>(runtime_alloc_census_after_retry[2]),
                     static_cast<unsigned long long>(runtime_alloc_census_after_retry[3]));
        return 1;
    }
    reg = nullptr;
    if (saved_model_load_begin(&stale_txn) != GGML_SYCL_LIFECYCLE_LOAD_BUSY) {
        std::fprintf(stderr, "saved model-load procedure admitted after completed unload\n");
        return 1;
    }
    if (saved_host_compute(0) != nullptr || ggml_backend_buft_alloc_buffer(saved_buft, 64) != nullptr) {
        std::fprintf(stderr, "saved buffer-type path allocated after completed unload\n");
        return 1;
    }
    phase("module reload staging failure remains closed and removed");
    if (ggml_backend_load(GGML_SYCL_RUNTIME_MODULE) != nullptr ||
        saved_model_load_begin(&stale_txn) != GGML_SYCL_LIFECYCLE_LOAD_BUSY ||
        ggml_backend_reg_by_name("SYCL") != nullptr) {
        std::fprintf(stderr, "reactivation staging failure published or reopened module\n");
        return 1;
    }
    phase("module reload");
    reg = ggml_backend_load(GGML_SYCL_RUNTIME_MODULE);
    if (!reg) {
        std::fprintf(stderr, "failed to reload SYCL backend module\n");
        return 1;
    }
    // Generic loader hooks and public SYCL names must both be rebuilt by the
    // reloaded registry. Checking alias identity catches a stale/incomplete
    // proc table before any lifecycle work begins.
    auto reloaded_get_device_memory = reinterpret_cast<decltype(&ggml_backend_sycl_get_device_memory)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_get_device_memory"));
    size_t free_after_shutdown = 0;
    size_t total_after_shutdown = 0;
    if (!reloaded_get_device_memory) {
        std::fprintf(stderr, "missing reloaded device-memory procedure\n");
        return 1;
    }
    constexpr size_t reclaim_driver_noise = 256ull * 1024ull * 1024ull;
    constexpr auto reclaim_poll_interval = std::chrono::milliseconds(100);
    constexpr auto reclaim_deadline = std::chrono::seconds(10);
    const auto reclaim_started = std::chrono::steady_clock::now();
    bool memory_recovered = false;
    do {
        reloaded_get_device_memory(0, &free_after_shutdown, &total_after_shutdown);
        const size_t measured_recovery = free_after_shutdown > free_with_cache ?
                                             free_after_shutdown - free_with_cache : 0;
        memory_recovered = total_after_shutdown == total_before_cache &&
                           measured_recovery + reclaim_driver_noise >= measured_cache_drop;
        if (memory_recovered || std::chrono::steady_clock::now() - reclaim_started >= reclaim_deadline) {
            break;
        }
        std::this_thread::sleep_for(reclaim_poll_interval);
    } while (true);
    const auto reclaim_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                        std::chrono::steady_clock::now() - reclaim_started)
                                        .count();
    if (!memory_recovered) {
        std::fprintf(stderr,
                     "logical unload retained device memory after %lld ms: before=%zu after=%zu "
                     "total_before=%zu total_after=%zu\n",
                     static_cast<long long>(reclaim_elapsed_ms), free_before_cache, free_after_shutdown,
                     total_before_cache, total_after_shutdown);
        return 1;
    }
    std::fprintf(stderr,
                 "device memory reclaim settled after %lld ms: allocated_free=%zu recovered_free=%zu delta=%zu\n",
                 static_cast<long long>(reclaim_elapsed_ms), free_with_cache, free_after_shutdown,
                 free_after_shutdown - free_with_cache);
    auto block_finalize_reactivate = reinterpret_cast<void (*)()>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_test_block_finalize_reactivate"));
    auto wait_finalize_reactivate_blocked = reinterpret_cast<void (*)()>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_test_wait_finalize_reactivate_blocked"));
    auto release_finalize_reactivate = reinterpret_cast<void (*)()>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_test_release_finalize_reactivate"));
    if (!block_finalize_reactivate || !wait_finalize_reactivate_blocked || !release_finalize_reactivate) {
        std::fprintf(stderr, "missing finalize reactivation control procedures\n");
        return 1;
    }
    phase("module reload finalize window keeps saved procedures closed until publication");
    if (ggml_backend_unload_checked(reg) != GGML_BACKEND_UNLOAD_OK) {
        std::fprintf(stderr, "failed to unload reloaded SYCL backend before finalize-window probe\n");
        return 1;
    }
    block_finalize_reactivate();
    auto blocked_reload = std::async(std::launch::async, [] { return ggml_backend_load(GGML_SYCL_RUNTIME_MODULE); });
    wait_finalize_reactivate_blocked();
    if (saved_model_load_begin(&stale_txn) != GGML_SYCL_LIFECYCLE_LOAD_BUSY || saved_host_compute(0) != nullptr ||
        ggml_backend_reg_by_name("SYCL") != nullptr) {
        std::fprintf(stderr, "reactivation commit/finalize window reopened saved procedures or published early\n");
        release_finalize_reactivate();
        return 1;
    }
    release_finalize_reactivate();
    reg = blocked_reload.get();
    if (!reg || saved_host_compute(0) == nullptr) {
        std::fprintf(stderr, "saved procedures did not reopen after finalize publication\n");
        return 1;
    }
    auto moe_state_clean = reinterpret_cast<decltype(&ggml_backend_sycl_test_moe_module_state_clean)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_test_moe_module_state_clean"));
    if (!moe_state_clean || !moe_state_clean()) {
        std::fprintf(stderr, "NODELETE reload retained model-bound MoE state\n");
        return 1;
    }
    using admission_snapshot_fn = void (*)(uint64_t out[8]);
    auto admission_snapshot = reinterpret_cast<admission_snapshot_fn>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_test_admission_snapshot"));
    auto * generic_can_unload = ggml_backend_reg_get_proc_address(reg, "ggml_backend_can_unload");
    auto * named_can_unload = ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_can_unload");
    auto * generic_cancel_unload = ggml_backend_reg_get_proc_address(reg, "ggml_backend_cancel_unload");
    auto * named_cancel_unload = ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_cancel_unload");
    if (!admission_snapshot || !generic_can_unload || generic_can_unload != named_can_unload ||
        !generic_cancel_unload || generic_cancel_unload != named_cancel_unload) {
        std::fprintf(stderr, "reload did not rebuild unload reservation procedure aliases\n");
        return 1;
    }
#    define LOAD_SYCL(name)                                                                                \
        auto name##_fn = reinterpret_cast<decltype(&name)>(ggml_backend_reg_get_proc_address(reg, #name)); \
        if (!name##_fn) {                                                                                  \
            std::fprintf(stderr, "missing registry procedure %s\n", #name);                                \
            return 1;                                                                                      \
        }
    LOAD_SYCL(ggml_backend_sycl_shutdown)
    LOAD_SYCL(ggml_backend_sycl_test_fail_next_backend_publish)
    LOAD_SYCL(ggml_backend_sycl_can_unload)
    LOAD_SYCL(ggml_backend_sycl_cancel_unload)
    LOAD_SYCL(ggml_backend_sycl_test_hold_live_update)
    LOAD_SYCL(ggml_backend_sycl_test_release_live_update)
    using seed_control_host_allocs_fn = bool (*)(ggml_backend_t, uint32_t);
    auto ggml_backend_sycl_test_seed_control_host_allocs_fn = reinterpret_cast<seed_control_host_allocs_fn>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_test_seed_control_host_allocs"));
    if (!ggml_backend_sycl_test_seed_control_host_allocs_fn) {
        std::fprintf(stderr, "missing registry procedure %s\n", "ggml_backend_sycl_test_seed_control_host_allocs");
        return 1;
    }
    LOAD_SYCL(ggml_backend_sycl_activate_model_plan)
    LOAD_SYCL(ggml_backend_sycl_set_runtime_context_for_model)
    LOAD_SYCL(ggml_backend_sycl_execution_context_create)
    LOAD_SYCL(ggml_backend_sycl_execution_context_bind_backend)
    LOAD_SYCL(ggml_backend_sycl_execution_context_extract)
    LOAD_SYCL(ggml_backend_sycl_execution_context_close_if_idle)
    LOAD_SYCL(ggml_backend_sycl_execution_context_begin_drain)
    LOAD_SYCL(ggml_backend_sycl_execution_context_extract_control_host_allocs)
    LOAD_SYCL(ggml_backend_sycl_execution_context_finish_drain)
    LOAD_SYCL(ggml_backend_sycl_execution_session_begin_reset)
    LOAD_SYCL(ggml_backend_sycl_execution_session_finish_reset)
    LOAD_SYCL(ggml_backend_sycl_stage_inventory_plan)
    LOAD_SYCL(ggml_backend_sycl_kv_buffer_type_from_dev)
    LOAD_SYCL(ggml_backend_sycl_push_kv_layer_mask_from_dev)
    LOAD_SYCL(ggml_backend_sycl_host_compute_buffer_type)
    LOAD_SYCL(ggml_backend_sycl_cpu_offload_compute_buffer_type)
    LOAD_SYCL(ggml_backend_sycl_cpu_offload_available)
    LOAD_SYCL(ggml_backend_sycl_has_active_placement_plan)
    LOAD_SYCL(ggml_backend_sycl_weights_evictable)
    LOAD_SYCL(ggml_backend_sycl_host_buffer_type)
    LOAD_SYCL(ggml_backend_sycl_host_buffer_type_for_device)
    LOAD_SYCL(ggml_backend_sycl_register_host_weight_tensor)
    LOAD_SYCL(ggml_backend_sycl_register_weight_identity)
    LOAD_SYCL(ggml_backend_sycl_register_weight_usage)
    LOAD_SYCL(ggml_backend_sycl_try_register_weight_usage)
    LOAD_SYCL(ggml_backend_sycl_model_quarantine_token)
    LOAD_SYCL(ggml_backend_sycl_model_load_begin)
    LOAD_SYCL(ggml_backend_sycl_model_load_end)
    LOAD_SYCL(ggml_backend_sycl_model_unloaded_token)
#    define CALL_SYCL(name) name##_fn
#else
#    define CALL_SYCL(name) name
#endif

#if defined(GGML_SYCL_RUNTIME_MODULE)
    phase("post-check admission reservation");
    uint64_t admission_before[8]{};
    uint64_t admission_after[8]{};
    admission_snapshot(admission_before);
    const size_t generic_calls_before = ggml_backend_test_active_calls(reg);
    const bool reserved = CALL_SYCL(ggml_backend_sycl_can_unload)();
    ggml_sycl_load_txn reserved_begin{};
    const auto reserved_begin_rc = reserved ? CALL_SYCL(ggml_backend_sycl_model_load_begin)(&reserved_begin) :
                                              GGML_SYCL_LIFECYCLE_BUSY;
    admission_snapshot(admission_after);
    if (!reserved || reserved_begin_rc != GGML_SYCL_LIFECYCLE_LOAD_BUSY) {
        std::fprintf(stderr,
                     "shutdown reservation check failed: reserved=%d begin_rc=%d generic_calls=%zu "
                     "before=[phase=%llu mutations=%llu txn=%llu models=%llu contexts=%llu updates=%llu "
                     "reserved=%llu completed=%llu] after=[phase=%llu mutations=%llu txn=%llu models=%llu "
                     "contexts=%llu updates=%llu reserved=%llu completed=%llu]\n",
                     reserved ? 1 : 0, (int) reserved_begin_rc, generic_calls_before,
                     (unsigned long long) admission_before[0], (unsigned long long) admission_before[1],
                     (unsigned long long) admission_before[2], (unsigned long long) admission_before[3],
                     (unsigned long long) admission_before[4], (unsigned long long) admission_before[5],
                     (unsigned long long) admission_before[6], (unsigned long long) admission_before[7],
                     (unsigned long long) admission_after[0], (unsigned long long) admission_after[1],
                     (unsigned long long) admission_after[2], (unsigned long long) admission_after[3],
                     (unsigned long long) admission_after[4], (unsigned long long) admission_after[5],
                     (unsigned long long) admission_after[6], (unsigned long long) admission_after[7]);
        return 1;
    }
    CALL_SYCL(ggml_backend_sycl_cancel_unload)();
#endif

    phase("backend construction failpoint rollback");
    CALL_SYCL(ggml_backend_sycl_test_fail_next_backend_publish)();
#if defined(GGML_SYCL_RUNTIME_MODULE)
    ggml_backend_t failed_backend = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
#else
    ggml_backend_t failed_backend = ggml_backend_sycl_init(0);
#endif
    if (failed_backend != nullptr || !CALL_SYCL(ggml_backend_sycl_can_unload)()) {
        std::fprintf(stderr, "backend construction failure leaked publication or admission\n");
        return 1;
    }
    CALL_SYCL(ggml_backend_sycl_cancel_unload)();

    // Allocation-parity procedures must resolve from the current registry in
    // both static and DL modes. Null metadata calls are defined no-ops.
    (void) CALL_SYCL(ggml_backend_sycl_weights_evictable)();
    if (!CALL_SYCL(ggml_backend_sycl_host_buffer_type)() ||
        CALL_SYCL(ggml_backend_sycl_host_buffer_type_for_device)(nullptr) != nullptr) {
        std::fprintf(stderr, "missing exact SYCL host buffer type\n");
        return 1;
    }
    CALL_SYCL(ggml_backend_sycl_register_host_weight_tensor)(nullptr, nullptr);
    CALL_SYCL(ggml_backend_sycl_register_weight_identity)(nullptr, 0, 0, 0, 0);
    CALL_SYCL(ggml_backend_sycl_register_weight_usage)(nullptr, GGML_SYCL_TENSOR_USAGE_UNKNOWN);
    if (CALL_SYCL(ggml_backend_sycl_try_register_weight_usage)(nullptr, GGML_SYCL_TENSOR_USAGE_UNKNOWN)) {
        std::fprintf(stderr, "status weight-usage API accepted null metadata\n");
        return 1;
    }

    ggml_sycl_model_token zero{};
    if (CALL_SYCL(ggml_backend_sycl_activate_model_plan)(zero) != GGML_SYCL_LIFECYCLE_STALE_IDENTITY ||
        CALL_SYCL(ggml_backend_sycl_set_runtime_context_for_model)(nullptr, zero, 0, 0, 0) !=
            GGML_SYCL_LIFECYCLE_NULL_OUTPUT) {
        std::fprintf(stderr, "activation/runtime API signature or invalid-input result mismatch\n");
        return 1;
    }
    if (CALL_SYCL(ggml_backend_sycl_model_quarantine_token)(zero) != GGML_SYCL_LIFECYCLE_STALE_IDENTITY) {
        std::fprintf(stderr, "zero quarantine token accepted\n");
        return 1;
    }
    ggml_sycl_model_token unknown{ 0x1234, 0x5678, 0, 1 };
    if (CALL_SYCL(ggml_backend_sycl_model_quarantine_token)(unknown) != GGML_SYCL_LIFECYCLE_STALE_IDENTITY) {
        std::fprintf(stderr, "unknown quarantine token accepted\n");
        return 1;
    }
    ggml_sycl_execution_snapshot exec_snapshot{};
    ggml_sycl_exec_drain_ticket drain_ticket{};
    ggml_sycl_exec_reset_ticket reset_ticket{};
    ggml_sycl_exec_control_host_alloc_batch drain_batch{ nullptr, 0 };
    if (CALL_SYCL(ggml_backend_sycl_execution_context_extract)({}, &exec_snapshot) != GGML_SYCL_EXECUTION_STALE ||
        CALL_SYCL(ggml_backend_sycl_execution_context_close_if_idle)({}) != GGML_SYCL_EXECUTION_STALE ||
        CALL_SYCL(ggml_backend_sycl_execution_context_begin_drain)({}, &drain_ticket) != GGML_SYCL_EXECUTION_STALE ||
        CALL_SYCL(ggml_backend_sycl_execution_context_extract_control_host_allocs)(&drain_ticket, &drain_batch) != GGML_SYCL_EXECUTION_STALE ||
        CALL_SYCL(ggml_backend_sycl_execution_context_finish_drain)(drain_ticket, &drain_batch) != GGML_SYCL_EXECUTION_STALE ||
        CALL_SYCL(ggml_backend_sycl_execution_session_begin_reset)({}, {}, {}, &reset_ticket) != GGML_SYCL_EXECUTION_STALE ||
        CALL_SYCL(ggml_backend_sycl_execution_session_finish_reset)(reset_ticket, &exec_snapshot.reset_epoch) != GGML_SYCL_EXECUTION_STALE) {
        std::fprintf(stderr, "execution lifecycle invalid-input contract mismatch\n");
        return 1;
    }

    phase("lifecycle abort checks");
    ggml_sycl_load_txn aborted{};
    if (CALL_SYCL(ggml_backend_sycl_model_load_begin)(&aborted) != GGML_SYCL_LIFECYCLE_OK || aborted.id == 0) {
        std::fprintf(stderr, "runtime begin failed\n");
        return 1;
    }
#if defined(GGML_SYCL_RUNTIME_MODULE)
    if (ggml_backend_unload_checked(reg) != GGML_BACKEND_UNLOAD_BUSY) {
        std::fprintf(stderr, "active load did not reject checked unload\n");
        return 1;
    }
#endif
    const auto abort_rc = CALL_SYCL(ggml_backend_sycl_model_load_end)(aborted, false, nullptr);
    if (abort_rc != GGML_SYCL_LIFECYCLE_MISSING_SUCCESS && abort_rc != GGML_SYCL_LIFECYCLE_POISONED) {
        std::fprintf(stderr, "runtime abort returned %d\n", (int) abort_rc);
        return 1;
    }
    // CPU-only and no_alloc ownership cancellation uses this abort path. More
    // than the slot count must remain reusable rather than leaking speculative
    // SYCL model ownership.
    const bool authority_before_cpu_cancels = CALL_SYCL(ggml_backend_sycl_has_active_placement_plan)();
    for (int i = 0; i < 40; ++i) {
        ggml_sycl_load_txn cpu_only{};
        if (CALL_SYCL(ggml_backend_sycl_model_load_begin)(&cpu_only) != GGML_SYCL_LIFECYCLE_OK) {
            std::fprintf(stderr, "speculative CPU lifecycle consumed slots at iteration %d\n", i);
            return 1;
        }
        const auto cancel_rc = CALL_SYCL(ggml_backend_sycl_model_load_end)(cpu_only, false, nullptr);
        if (cancel_rc != GGML_SYCL_LIFECYCLE_MISSING_SUCCESS && cancel_rc != GGML_SYCL_LIFECYCLE_POISONED) {
            std::fprintf(stderr, "speculative CPU lifecycle cancel returned %d\n", (int) cancel_rc);
            return 1;
        }
    }
    if (CALL_SYCL(ggml_backend_sycl_has_active_placement_plan)() != authority_before_cpu_cancels) {
        std::fprintf(stderr, "speculative CPU lifecycle changed placement authority\n");
        return 1;
    }

    phase("empty early and late planning commit");
    ggml_sycl_load_txn    committed{};
    ggml_sycl_model_token token{};
    ggml_sycl_tensor_inventory inventory{};
    inventory.n_ctx    = 32;
    inventory.n_ubatch = 8;
    ggml_sycl_placement_envelope envelope{};
    envelope.n_ctx           = 32;
    envelope.n_ubatch        = 8;
    envelope.n_seq_max       = 1;
    envelope.flash_attn_type = -1;
    if (CALL_SYCL(ggml_backend_sycl_model_load_begin)(&committed) != GGML_SYCL_LIFECYCLE_OK ||
        CALL_SYCL(ggml_backend_sycl_stage_inventory_plan)(&inventory, &envelope, true) != GGML_SYCL_LIFECYCLE_OK ||
        CALL_SYCL(ggml_backend_sycl_stage_inventory_plan)(&inventory, &envelope, false) != GGML_SYCL_LIFECYCLE_OK ||
        CALL_SYCL(ggml_backend_sycl_model_load_end)(committed, true, &token) != GGML_SYCL_LIFECYCLE_OK ||
        token.model_id == 0) {
        std::fprintf(stderr, "runtime no-allocation commit failed\n");
        return 1;
    }
#if defined(GGML_SYCL_RUNTIME_MODULE)
    phase("live-owner and teardown-overlap unload rejection");
    if (ggml_backend_unload_checked(reg) != GGML_BACKEND_UNLOAD_BUSY ||
        !CALL_SYCL(ggml_backend_sycl_test_hold_live_update)(token)) {
        std::fprintf(stderr, "live-owner unload was not observably deferred\n");
        return 1;
    }
    auto overlapping_teardown = std::async(std::launch::async, [&] {
        return CALL_SYCL(ggml_backend_sycl_model_unloaded_token)(token);
    });
    if (overlapping_teardown.wait_for(std::chrono::milliseconds(20)) != std::future_status::timeout ||
        ggml_backend_unload_checked(reg) != GGML_BACKEND_UNLOAD_BUSY) {
        std::fprintf(stderr, "DRAINING_UPDATES unload overlap was not rejected\n");
        return 1;
    }
    CALL_SYCL(ggml_backend_sycl_test_release_live_update)();
    if (overlapping_teardown.get() != GGML_SYCL_LIFECYCLE_OK) {
        std::fprintf(stderr, "overlapping teardown did not recover\n");
        return 1;
    }
#else
    phase("model teardown");
    if (CALL_SYCL(ggml_backend_sycl_model_unloaded_token)(token) != GGML_SYCL_LIFECYCLE_OK) {
        std::fprintf(stderr, "runtime teardown failed\n");
        return 1;
    }
#endif
    const auto repeat = CALL_SYCL(ggml_backend_sycl_model_unloaded_token)(token);
    if (repeat != GGML_SYCL_LIFECYCLE_OK_ALREADY_DEAD && repeat != GGML_SYCL_LIFECYCLE_STALE_IDENTITY) {
        std::fprintf(stderr, "runtime repeated teardown returned %d\n", (int) repeat);
        return 1;
    }
#if defined(GGML_SYCL_RUNTIME_MODULE)
    phase("quarantined-owner unload rejection");
    ggml_sycl_load_txn quarantine_txn{};
    ggml_sycl_model_token quarantine_token{};
    if (CALL_SYCL(ggml_backend_sycl_model_load_begin)(&quarantine_txn) != GGML_SYCL_LIFECYCLE_OK ||
        CALL_SYCL(ggml_backend_sycl_stage_inventory_plan)(&inventory, &envelope, false) != GGML_SYCL_LIFECYCLE_OK ||
        CALL_SYCL(ggml_backend_sycl_model_load_end)(quarantine_txn, true, &quarantine_token) !=
            GGML_SYCL_LIFECYCLE_OK ||
        CALL_SYCL(ggml_backend_sycl_model_quarantine_token)(quarantine_token) != GGML_SYCL_LIFECYCLE_OK ||
        ggml_backend_unload_checked(reg) != GGML_BACKEND_UNLOAD_BUSY ||
        CALL_SYCL(ggml_backend_sycl_model_unloaded_token)(quarantine_token) != GGML_SYCL_LIFECYCLE_OK) {
        std::fprintf(stderr, "quarantined owner unload guard/recovery failed\n");
        return 1;
    }
    if (CALL_SYCL(ggml_backend_sycl_has_active_placement_plan)()) {
        std::fprintf(stderr, "teardown retained placement authority\n");
        return 1;
    }
    phase("durable buffer owner unload rejection");
    auto durable_buft = CALL_SYCL(ggml_backend_sycl_host_buffer_type)();
    auto durable_compute_buft = CALL_SYCL(ggml_backend_sycl_host_compute_buffer_type)(0);
    auto durable_buffer = durable_buft ? ggml_backend_buft_alloc_buffer(durable_buft, 64) : nullptr;
    auto durable_compute_buffer = durable_compute_buft ?
                                      ggml_backend_buft_alloc_buffer(durable_compute_buft, 64) : nullptr;
    if (!durable_buffer || !durable_compute_buffer ||
        ggml_backend_unload_checked(reg) != GGML_BACKEND_UNLOAD_BUSY) {
        std::fprintf(stderr, "live backend buffer did not block checked unload\n");
        return 1;
    }
    (void) ggml_backend_buffer_get_base(durable_buffer);
    ggml_backend_buffer_clear(durable_buffer, 0);
    ggml_backend_buffer_reset(durable_buffer);
    ggml_backend_buffer_free(durable_buffer);
    if (ggml_backend_unload_checked(reg) != GGML_BACKEND_UNLOAD_BUSY) {
        std::fprintf(stderr, "remaining host-compute buffer lost durable ownership\n");
        return 1;
    }
    ggml_backend_buffer_free(durable_compute_buffer);

    phase("durable event owner unload rejection");
    auto durable_event = ggml_backend_event_new(ggml_backend_reg_dev_get(reg, 0));
    if (!durable_event || ggml_backend_unload_checked(reg) != GGML_BACKEND_UNLOAD_BUSY) {
        std::fprintf(stderr, "live backend event did not block checked unload\n");
        return 1;
    }
    ggml_backend_event_synchronize(durable_event);
    ggml_backend_event_free(durable_event);

    phase("stateful module unload");
    size_t unloaded_reg_index = ggml_backend_reg_count();
    for (size_t i = 0; i < unloaded_reg_index; ++i) {
        if (ggml_backend_reg_get(i) == reg) {
            unloaded_reg_index = i;
            break;
        }
    }
    ggml_backend_dev_t sycl_dev = ggml_backend_reg_dev_get(reg, 0);
    size_t unloaded_dev_index = ggml_backend_dev_count();
    for (size_t i = 0; i < unloaded_dev_index; ++i) {
        if (ggml_backend_dev_get(i) == sycl_dev) {
            unloaded_dev_index = i;
            break;
        }
    }
    if (unloaded_reg_index == ggml_backend_reg_count() || unloaded_dev_index == ggml_backend_dev_count() ||
        ggml_backend_unload_checked(reg) != GGML_BACKEND_UNLOAD_OK) {
        std::fprintf(stderr, "dead-owner module enumeration/unload failed\n");
        return 1;
    }
    // These get() calls consume the count snapshots captured before removal;
    // both must reject the now-tombstoned identity deterministically.
    const char * tombstoned_name = ggml_backend_reg_name(reg);
    if (ggml_backend_reg_by_name("SYCL") != nullptr || ggml_backend_reg_get(unloaded_reg_index) != nullptr ||
        ggml_backend_dev_get(unloaded_dev_index) != nullptr || ggml_backend_reg_dev_get(reg, 0) != nullptr ||
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_model_load_begin") != nullptr ||
        ggml_backend_dev_init(sycl_dev, nullptr) != nullptr || !tombstoned_name ||
        std::strcmp(tombstoned_name, "SYCL") != 0) {
        std::fprintf(stderr, "logical unload lookup or retained raw-handle lease failed\n");
        return 1;
    }
    reg = nullptr;

    phase("reload after stateful shutdown");
    reg = ggml_backend_load(GGML_SYCL_RUNTIME_MODULE);
    if (!reg) {
        std::fprintf(stderr, "stateful module reload failed\n");
        return 1;
    }
    auto begin_again = reinterpret_cast<decltype(&ggml_backend_sycl_model_load_begin)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_model_load_begin"));
    auto end_again = reinterpret_cast<decltype(&ggml_backend_sycl_model_load_end)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_model_load_end"));
    auto unload_again = reinterpret_cast<decltype(&ggml_backend_sycl_model_unloaded_token)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_model_unloaded_token"));
    auto stage_again = reinterpret_cast<decltype(&ggml_backend_sycl_stage_inventory_plan)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_stage_inventory_plan"));
    auto authority_again = reinterpret_cast<decltype(&ggml_backend_sycl_has_active_placement_plan)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_has_active_placement_plan"));
    ggml_sycl_load_txn reload_txn{};
    ggml_sycl_model_token reload_token{};
    if (!begin_again || !end_again || !unload_again || !stage_again || !authority_again || authority_again() ||
        begin_again(&reload_txn) != GGML_SYCL_LIFECYCLE_OK ||
        stage_again(&inventory, &envelope, false) != GGML_SYCL_LIFECYCLE_OK ||
        end_again(reload_txn, true, &reload_token) != GGML_SYCL_LIFECYCLE_OK || reload_token.model_id == 0 ||
        unload_again(reload_token) != GGML_SYCL_LIFECYCLE_OK || authority_again()) {
        std::fprintf(stderr, "post-shutdown lifecycle reuse left dirty authority/token state\n");
        return 1;
    }
    phase("unlocked tombstone load/enumeration overlap");
    ggml_backend_test_reentrant_mutation_on_next_unload();
    ggml_backend_test_block_next_unload();
    auto final_unload = std::async(std::launch::async, [&] { return ggml_backend_unload_checked(reg); });
    ggml_backend_test_wait_unload_blocked();
    auto concurrent_load = std::async(std::launch::async, [&] { return ggml_backend_load(GGML_SYCL_RUNTIME_MODULE); });
    auto concurrent_enumeration = std::async(std::launch::async, [] { return ggml_backend_dev_count(); });
    const bool load_failed_closed =
        concurrent_load.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
    const bool enumeration_unlocked =
        concurrent_enumeration.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
    if (!load_failed_closed || !enumeration_unlocked) {
        std::fprintf(stderr, "hook-contended load did not fail closed or enumeration blocked\n");
        ggml_backend_test_release_unload();
        (void) final_unload.get();
        (void) concurrent_load.get();
        (void) concurrent_enumeration.get();
        return 1;
    }
    (void) concurrent_enumeration.get();
    if (concurrent_load.get() != nullptr || ggml_backend_reg_by_name("SYCL") != nullptr) {
        std::fprintf(stderr, "hook-contended load published a backend instead of returning null\n");
        ggml_backend_test_release_unload();
        (void) final_unload.get();
        return 1;
    }
    ggml_backend_test_release_unload();
    if (final_unload.get() != GGML_BACKEND_UNLOAD_OK ||
        !(reg = ggml_backend_load(GGML_SYCL_RUNTIME_MODULE))) {
        std::fprintf(stderr, "post-hook logical reload retry failed\n");
        return 1;
    }
    if (ggml_backend_unload_checked(reg) != GGML_BACKEND_UNLOAD_OK) {
        std::fprintf(stderr, "pre-renamed fixture unload failed\n");
        return 1;
    }
#    if defined(GGML_SYCL_RENAMED_RUNTIME_MODULE)
    phase("renamed DSO load/checked-unload/reload");
    auto * renamed_reg = ggml_backend_load(GGML_SYCL_RENAMED_RUNTIME_MODULE);
    if (!renamed_reg || ggml_backend_unload_checked(renamed_reg) != GGML_BACKEND_UNLOAD_OK ||
        !(renamed_reg = ggml_backend_load(GGML_SYCL_RENAMED_RUNTIME_MODULE)) ||
        ggml_backend_unload_checked(renamed_reg) != GGML_BACKEND_UNLOAD_OK) {
        std::fprintf(stderr, "renamed DSO lifecycle path failed: %s\n", GGML_SYCL_RENAMED_RUNTIME_MODULE);
        return 1;
    }
#    endif
    phase("complete");
#endif
    return 0;
}
