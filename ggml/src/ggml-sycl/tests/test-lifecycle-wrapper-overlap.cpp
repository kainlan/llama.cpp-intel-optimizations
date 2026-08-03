#include "ggml-sycl.h"

#include <cstdio>
#include <future>

extern "C" void ggml_backend_sycl_test_fail_next_candidate_binding_allocation();
extern "C" void ggml_backend_sycl_test_block_next_candidate_binding_allocation();
extern "C" void ggml_backend_sycl_test_wait_for_candidate_binding_failure();
extern "C" void ggml_backend_sycl_test_release_candidate_binding_failure();

int main() {
#if defined(GGML_SYCL_RUNTIME_MODULE)
    auto * reg = ggml_backend_load(GGML_SYCL_RUNTIME_MODULE);
    if (!reg) {
        std::fprintf(stderr, "failed to load SYCL backend module\n");
        return 1;
    }
#    define LOAD_SYCL(name)                                                                                \
        auto name##_fn = reinterpret_cast<decltype(&name)>(ggml_backend_reg_get_proc_address(reg, #name)); \
        if (!name##_fn) {                                                                                  \
            std::fprintf(stderr, "missing registry procedure %s\n", #name);                                \
            return 1;                                                                                      \
        }
    LOAD_SYCL(ggml_backend_sycl_model_quarantine_token)
    LOAD_SYCL(ggml_backend_sycl_model_load_begin)
    LOAD_SYCL(ggml_backend_sycl_model_load_enter_nested)
    LOAD_SYCL(ggml_backend_sycl_model_load_end)
    LOAD_SYCL(ggml_backend_sycl_model_unloaded_token)
    LOAD_SYCL(ggml_backend_sycl_test_fail_next_candidate_binding_allocation)
    LOAD_SYCL(ggml_backend_sycl_test_block_next_candidate_binding_allocation)
    LOAD_SYCL(ggml_backend_sycl_test_wait_for_candidate_binding_failure)
    LOAD_SYCL(ggml_backend_sycl_test_release_candidate_binding_failure)
#    define CALL_SYCL(name) name##_fn
#else
#    define CALL_SYCL(name) name
#endif

    ggml_sycl_load_txn    model_load{};
    ggml_sycl_model_token model{};
    if (CALL_SYCL(ggml_backend_sycl_model_load_begin)(&model_load) != GGML_SYCL_LIFECYCLE_OK ||
        CALL_SYCL(ggml_backend_sycl_model_load_end)(model_load, true, &model) != GGML_SYCL_LIFECYCLE_OK) {
        std::fprintf(stderr, "failed to publish model fixture\n");
        return 1;
    }

    ggml_sycl_load_txn blocker{};
    if (CALL_SYCL(ggml_backend_sycl_model_load_begin)(&blocker) != GGML_SYCL_LIFECYCLE_OK ||
        CALL_SYCL(ggml_backend_sycl_model_quarantine_token)(model) != GGML_SYCL_LIFECYCLE_OK) {
        std::fprintf(stderr, "failed to establish destructor/load overlap\n");
        return 1;
    }

    // load_begin runs one reaper pass before attempting another begin. The
    // active blocker makes exact-token teardown BUSY, so the reaper must retain
    // it. A duplicate destructor enqueue during that interval must deduplicate.
    ggml_sycl_load_txn contender{};
    if (CALL_SYCL(ggml_backend_sycl_model_load_begin)(&contender) != GGML_SYCL_LIFECYCLE_LOAD_BUSY ||
        CALL_SYCL(ggml_backend_sycl_model_quarantine_token)(model) != GGML_SYCL_LIFECYCLE_OK) {
        std::fprintf(stderr, "BUSY reaper/destructor overlap lost wrapper semantics\n");
        return 1;
    }
    if (CALL_SYCL(ggml_backend_sycl_model_load_end)(blocker, false, nullptr) != GGML_SYCL_LIFECYCLE_MISSING_SUCCESS) {
        std::fprintf(stderr, "failed to release blocking load\n");
        return 1;
    }

    ggml_sycl_load_txn reaper_trigger{};
    if (CALL_SYCL(ggml_backend_sycl_model_load_begin)(&reaper_trigger) != GGML_SYCL_LIFECYCLE_OK ||
        CALL_SYCL(ggml_backend_sycl_model_load_end)(reaper_trigger, false, nullptr) !=
            GGML_SYCL_LIFECYCLE_MISSING_SUCCESS ||
        CALL_SYCL(ggml_backend_sycl_model_unloaded_token)(model) != GGML_SYCL_LIFECYCLE_OK_ALREADY_DEAD) {
        std::fprintf(stderr, "retained destructor token was not reaped after BUSY cleared\n");
        return 1;
    }

    ggml_sycl_load_txn binding_failure{};
    if (CALL_SYCL(ggml_backend_sycl_model_load_begin)(&binding_failure) != GGML_SYCL_LIFECYCLE_OK) {
        std::fprintf(stderr, "binding-failure load begin failed\n");
        return 1;
    }
    CALL_SYCL(ggml_backend_sycl_test_fail_next_candidate_binding_allocation)();
    auto failed_end = std::async(std::launch::async, [&] {
        ggml_sycl_model_token unused{};
        return CALL_SYCL(ggml_backend_sycl_model_load_end)(binding_failure, true, &unused);
    });
    if (failed_end.get() != GGML_SYCL_LIFECYCLE_EFFECT_FAILED ||
        CALL_SYCL(ggml_backend_sycl_model_load_end)(binding_failure, false, nullptr) != GGML_SYCL_LIFECYCLE_POISONED) {
        std::fprintf(stderr, "cross-thread binding allocation failure was not durably aborted\n");
        return 1;
    }
    ggml_sycl_load_txn after_failure{};
    if (CALL_SYCL(ggml_backend_sycl_model_load_begin)(&after_failure) != GGML_SYCL_LIFECYCLE_OK ||
        CALL_SYCL(ggml_backend_sycl_model_load_end)(after_failure, false, nullptr) !=
            GGML_SYCL_LIFECYCLE_MISSING_SUCCESS) {
        std::fprintf(stderr, "binding allocation failure wedged later model loads\n");
        return 1;
    }

    ggml_sycl_load_txn nested_begin_failure{};
    if (CALL_SYCL(ggml_backend_sycl_model_load_begin)(&nested_begin_failure) != GGML_SYCL_LIFECYCLE_OK) {
        std::fprintf(stderr, "nested-begin fault fixture failed\n");
        return 1;
    }
    CALL_SYCL(ggml_backend_sycl_test_fail_next_candidate_binding_allocation)();
    ggml_sycl_model_token nested_begin_token{};
    if (CALL_SYCL(ggml_backend_sycl_model_load_enter_nested)(nested_begin_failure) !=
            GGML_SYCL_LIFECYCLE_EFFECT_FAILED ||
        CALL_SYCL(ggml_backend_sycl_model_load_end)(nested_begin_failure, true, &nested_begin_token) !=
            GGML_SYCL_LIFECYCLE_POISONED) {
        std::fprintf(stderr, "nested begin binding failure changed outer depth or wedged load\n");
        return 1;
    }

    ggml_sycl_load_txn nested_end_failure{};
    if (CALL_SYCL(ggml_backend_sycl_model_load_begin)(&nested_end_failure) != GGML_SYCL_LIFECYCLE_OK ||
        CALL_SYCL(ggml_backend_sycl_model_load_enter_nested)(nested_end_failure) != GGML_SYCL_LIFECYCLE_OK) {
        std::fprintf(stderr, "nested-end fault fixture failed\n");
        return 1;
    }
    CALL_SYCL(ggml_backend_sycl_test_fail_next_candidate_binding_allocation)();
    auto                  failed_nested_end = std::async(std::launch::async, [&] {
        return CALL_SYCL(ggml_backend_sycl_model_load_end)(nested_end_failure, false, nullptr);
    });
    ggml_sycl_model_token nested_end_token{};
    if (failed_nested_end.get() != GGML_SYCL_LIFECYCLE_EFFECT_FAILED ||
        CALL_SYCL(ggml_backend_sycl_model_load_end)(nested_end_failure, true, &nested_end_token) !=
            GGML_SYCL_LIFECYCLE_POISONED) {
        std::fprintf(stderr, "nested end binding failure changed outer depth or wedged load\n");
        return 1;
    }

    ggml_sycl_load_txn race_load{};
    if (CALL_SYCL(ggml_backend_sycl_model_load_begin)(&race_load) != GGML_SYCL_LIFECYCLE_OK) {
        std::fprintf(stderr, "binding/commit race fixture failed\n");
        return 1;
    }
    CALL_SYCL(ggml_backend_sycl_test_block_next_candidate_binding_allocation)();
    ggml_sycl_model_token raced_replay{};
    auto                  raced_end = std::async(std::launch::async, [&] {
        return CALL_SYCL(ggml_backend_sycl_model_load_end)(race_load, true, &raced_replay);
    });
    CALL_SYCL(ggml_backend_sycl_test_wait_for_candidate_binding_failure)();
    ggml_sycl_model_token canonical{};
    const auto            canonical_rc = CALL_SYCL(ggml_backend_sycl_model_load_end)(race_load, true, &canonical);
    CALL_SYCL(ggml_backend_sycl_test_release_candidate_binding_failure)();
    if (canonical_rc != GGML_SYCL_LIFECYCLE_OK || raced_end.get() != GGML_SYCL_LIFECYCLE_OK ||
        canonical.model_id == 0 || raced_replay.model_id != canonical.model_id ||
        raced_replay.load_txn_id != canonical.load_txn_id || raced_replay.slot != canonical.slot ||
        raced_replay.slot_generation != canonical.slot_generation) {
        std::fprintf(stderr, "binding failure raced canonical commit into a noncanonical result\n");
        return 1;
    }
    if (CALL_SYCL(ggml_backend_sycl_model_unloaded_token)(canonical) != GGML_SYCL_LIFECYCLE_OK) {
        std::fprintf(stderr, "binding/commit race fixture teardown failed\n");
        return 1;
    }

    // Terminal and wrong-transaction classification must bypass candidate
    // binding entirely. One pending injected failure therefore survives all
    // three calls while their canonical durable results are replayed.
    CALL_SYCL(ggml_backend_sycl_test_fail_next_candidate_binding_allocation)();
    ggml_sycl_load_txn    unknown{ UINT64_MAX - 1 };
    ggml_sycl_model_token committed_replay{};
    if (CALL_SYCL(ggml_backend_sycl_model_load_end)(unknown, false, nullptr) != GGML_SYCL_LIFECYCLE_WRONG_TRANSACTION ||
        CALL_SYCL(ggml_backend_sycl_model_load_end)(model_load, true, &committed_replay) != GGML_SYCL_LIFECYCLE_OK ||
        committed_replay.model_id != model.model_id ||
        CALL_SYCL(ggml_backend_sycl_model_load_end)(after_failure, false, nullptr) !=
            GGML_SYCL_LIFECYCLE_MISSING_SUCCESS) {
        std::fprintf(stderr, "terminal/wrong replay consulted fallible candidate binding\n");
        return 1;
    }

#if defined(GGML_SYCL_RUNTIME_MODULE)
    ggml_backend_unload(reg);
#endif
    return 0;
}
