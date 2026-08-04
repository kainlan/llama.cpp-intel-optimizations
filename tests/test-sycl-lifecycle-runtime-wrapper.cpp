#include "ggml-sycl.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <future>
#include <stdexcept>
#include <thread>

extern "C" void ggml_backend_test_block_next_unload();
extern "C" void ggml_backend_test_wait_unload_blocked();
extern "C" void ggml_backend_test_reentrant_mutation_on_next_unload();
extern "C" void ggml_backend_test_release_unload();
extern "C" void ggml_backend_sycl_test_seed_moe_module_state();
extern "C" bool ggml_backend_sycl_test_moe_module_state_clean();
extern "C" bool ggml_backend_sycl_test_allocate_predictor_scores();
extern "C" void ggml_backend_sycl_test_fail_next_backend_publish();
extern "C" bool ggml_backend_sycl_test_hold_live_update(ggml_sycl_model_token model);
extern "C" void ggml_backend_sycl_test_release_live_update();

namespace {
enum class registry_fixture_mode { NORMAL, RESOLVER_THROW, SHUTDOWN_THROW };
static registry_fixture_mode g_registry_fixture_mode = registry_fixture_mode::NORMAL;
static ggml_backend_reg_t     g_registry_fixture_reg = nullptr;
static int                    g_registry_fixture_recursive_registrations = 0;
static int                    g_registry_fixture_shutdowns = 0;
static int                    g_registry_fixture_cancels = 0;

static const char * registry_fixture_dev_name(ggml_backend_dev_t) { return "TEST-LIFECYCLE0"; }
static const char * registry_fixture_dev_description(ggml_backend_dev_t) { return "registry lifecycle fixture"; }
static enum ggml_backend_dev_type registry_fixture_dev_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}
static ggml_backend_t registry_fixture_dev_init(ggml_backend_dev_t, const char *) { return nullptr; }
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
static void registry_fixture_shutdown() {
    ++g_registry_fixture_shutdowns;
    if (g_registry_fixture_mode == registry_fixture_mode::SHUTDOWN_THROW) {
        throw std::runtime_error("fixture partial shutdown");
    }
}
static void registry_fixture_cancel() { ++g_registry_fixture_cancels; }
static void * registry_fixture_resolve(ggml_backend_reg_t, const char * name) {
    if (g_registry_fixture_mode == registry_fixture_mode::RESOLVER_THROW) {
        throw std::runtime_error("fixture resolver failure");
    }
    if (std::strcmp(name, "ggml_backend_can_unload") == 0) return (void *) registry_fixture_can_unload;
    if (std::strcmp(name, "ggml_backend_shutdown") == 0) return (void *) registry_fixture_shutdown;
    if (std::strcmp(name, "ggml_backend_complete_unload") == 0 ||
        std::strcmp(name, "ggml_backend_cancel_unload") == 0) return (void *) registry_fixture_cancel;
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

static bool run_registry_failure_fixture() {
    auto reg = registry_fixture();
    ggml_backend_register(reg);
    const size_t initial_reg_count = ggml_backend_reg_count();
#if defined(GGML_SYCL_RUNTIME_MODULE)
    if (initial_reg_count < 1) return false;
#else
    // Static first-use registration must finish publishing built-in SYCL/CPU
    // entries before the custom fixture is admitted.
    if (initial_reg_count < 2 || ggml_backend_reg_by_name("SYCL") == nullptr) return false;
#endif

    g_registry_fixture_mode = registry_fixture_mode::RESOLVER_THROW;
    if (ggml_backend_unload_checked(reg) != GGML_BACKEND_UNLOAD_BUSY ||
        ggml_backend_reg_by_name("TEST-LIFECYCLE") != nullptr) return false;
    for (int i = 0; i < 1000; ++i) {
        if (ggml_backend_reg_by_name("TEST-LIFECYCLE") != nullptr) return false;
    }
    g_registry_fixture_mode = registry_fixture_mode::NORMAL;
    if (ggml_backend_unload_checked(reg) != GGML_BACKEND_UNLOAD_OK) return false;

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

    g_registry_fixture_mode = registry_fixture_mode::SHUTDOWN_THROW;
    if (reg_index == reg_count || dev_index == dev_count ||
        ggml_backend_unload_checked(reg) != GGML_BACKEND_UNLOAD_BUSY ||
        ggml_backend_reg_get(reg_index) != nullptr || ggml_backend_dev_get(dev_index) != nullptr ||
        ggml_backend_reg_by_name("TEST-LIFECYCLE") != nullptr) return false;
    g_registry_fixture_mode = registry_fixture_mode::NORMAL;
    if (ggml_backend_unload_checked(reg) != GGML_BACKEND_UNLOAD_OK) return false;
    return g_registry_fixture_recursive_registrations >= 2 && g_registry_fixture_shutdowns >= 2 &&
           g_registry_fixture_cancels >= 2;
}
} // namespace

static void phase(const char * name) {
    std::fprintf(stderr, "[sycl-runtime-wrapper] %s\n", name);
    std::fflush(stderr);
}

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
    auto allocate_predictor_scores = reinterpret_cast<decltype(&ggml_backend_sycl_test_allocate_predictor_scores)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_test_allocate_predictor_scores"));
    if (!allocate_predictor_scores || !allocate_predictor_scores()) {
        std::fprintf(stderr, "failed to create real predictor unified allocation\n");
        return 1;
    }
    seed_moe_state();
    phase("initial module unload with shutdown hook");
    shutdown = nullptr;
    ggml_backend_unload(reg);
    reg = nullptr;
    phase("module reload");
    reg = ggml_backend_load(GGML_SYCL_RUNTIME_MODULE);
    if (!reg) {
        std::fprintf(stderr, "failed to reload SYCL backend module\n");
        return 1;
    }
    // Generic loader hooks and public SYCL names must both be rebuilt by the
    // reloaded registry. Checking alias identity catches a stale/incomplete
    // proc table before any lifecycle work begins.
    auto moe_state_clean = reinterpret_cast<decltype(&ggml_backend_sycl_test_moe_module_state_clean)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_test_moe_module_state_clean"));
    if (!moe_state_clean || !moe_state_clean()) {
        std::fprintf(stderr, "NODELETE reload retained model-bound MoE state\n");
        return 1;
    }
    auto * generic_can_unload = ggml_backend_reg_get_proc_address(reg, "ggml_backend_can_unload");
    auto * named_can_unload = ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_can_unload");
    auto * generic_cancel_unload = ggml_backend_reg_get_proc_address(reg, "ggml_backend_cancel_unload");
    auto * named_cancel_unload = ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_cancel_unload");
    if (!generic_can_unload || generic_can_unload != named_can_unload || !generic_cancel_unload ||
        generic_cancel_unload != named_cancel_unload) {
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
    LOAD_SYCL(ggml_backend_sycl_activate_model_plan)
    LOAD_SYCL(ggml_backend_sycl_set_runtime_context_for_model)
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
    ggml_sycl_load_txn reserved_begin{};
    if (!CALL_SYCL(ggml_backend_sycl_can_unload)() ||
        CALL_SYCL(ggml_backend_sycl_model_load_begin)(&reserved_begin) != GGML_SYCL_LIFECYCLE_LOAD_BUSY) {
        std::fprintf(stderr, "shutdown reservation admitted a post-check load\n");
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
    if (ggml_backend_reg_by_name("SYCL") != nullptr || ggml_backend_reg_get(unloaded_reg_index) != nullptr ||
        ggml_backend_dev_get(unloaded_dev_index) != nullptr || ggml_backend_reg_dev_get(reg, 0) != nullptr ||
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_model_load_begin") != nullptr ||
        ggml_backend_dev_init(sycl_dev, nullptr) != nullptr || std::strcmp(ggml_backend_reg_name(reg), "SYCL") != 0) {
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
    if (concurrent_load.wait_for(std::chrono::milliseconds(100)) != std::future_status::timeout ||
        concurrent_enumeration.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready) {
        std::fprintf(stderr, "same-module load was not deferred or enumeration held the registry lock\n");
        return 1;
    }
    (void) concurrent_enumeration.get();
    ggml_backend_test_release_unload();
    if (final_unload.get() != GGML_BACKEND_UNLOAD_OK || !(reg = concurrent_load.get())) {
        std::fprintf(stderr, "deferred logical reload failed\n");
        return 1;
    }
    phase("complete");
#endif
    return 0;
}
