#include "ggml-sycl.h"

#include <chrono>
#include <cstdio>
#include <future>
#include <thread>

extern "C" bool ggml_backend_sycl_test_hold_live_update(ggml_sycl_model_token model);
extern "C" void ggml_backend_sycl_test_release_live_update();

static void phase(const char * name) {
    std::fprintf(stderr, "[sycl-runtime-wrapper] %s\n", name);
    std::fflush(stderr);
}

int main() {
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
    if (ggml_backend_unload_checked(reg) != GGML_BACKEND_UNLOAD_OK) {
        std::fprintf(stderr, "dead-owner module unload failed\n");
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
    phase("final clean module unload");
    if (ggml_backend_unload_checked(reg) != GGML_BACKEND_UNLOAD_OK) {
        return 1;
    }
    reg = nullptr;
    phase("complete");
#endif
    return 0;
}
