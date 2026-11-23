#pragma once

// Private BUILD_TESTING-only seam for exercising the Q1/NVFP4 direct
// production route. This header is deliberately kept beside its sole test and
// is not installed with ggml-sycl.h.
#include "ggml-backend.h"
#include "ggml-sycl.h"

#include <cstdint>

struct ggml_sycl_q1_nvfp4_test_scope_token {
    uint64_t nonce;
    uintptr_t backend_identity;
    uint64_t context_id;
    ggml_sycl_model_token model;
};

struct ggml_sycl_q1_nvfp4_test_counters {
    uint64_t candidate;
    uint64_t admit;
    uint64_t submit;
    uint64_t terminal;
    uint64_t recycle;
    uint64_t quarantine;
    uint64_t failure_consumed;
    uint32_t workspace_slot;
    uint64_t workspace_generation;
    // The direct route's host leg: the mixed batches whose host occurrences it submitted, and how many occurrences.
    uint64_t host_submit;
    uint64_t host_occurrences;
    // Its card leg: the occurrences it submitted to its device kernel.
    uint64_t card_occurrences;
};

enum ggml_sycl_q1_nvfp4_test_failure {
    GGML_SYCL_Q1_NVFP4_TEST_FAILURE_NONE = 0,
    GGML_SYCL_Q1_NVFP4_TEST_FAILURE_PRE_MARK,
    GGML_SYCL_Q1_NVFP4_TEST_FAILURE_POST_MARK,
    GGML_SYCL_Q1_NVFP4_TEST_FAILURE_ASYNC_TERMINAL,
    // The completion host task skips its release once. It stands for a completion task that did not release, its
    // release refused or skipped; whether one can still be pending after the drain's deferred-decode wait is unmeasured.
    GGML_SYCL_Q1_NVFP4_TEST_FAILURE_COMPLETION_NO_RELEASE,
    // The host leg's job refuses its first task once, after admission. It stands for a host recipe that refuses a task
    // its check before admission passed, which the check rules out in a correct build.
    GGML_SYCL_Q1_NVFP4_TEST_FAILURE_HOST_JOB,
    // mmvq_moe_batched_dispatch refuses a Q1_0/NVFP4 operand once, just before its submit.
    GGML_SYCL_Q1_NVFP4_TEST_FAILURE_BATCHED_DISPATCH,
    // The MoE prompt path of ggml_sycl_mul_mat_id, any expert type: one staging allocation acts as if the card had
    // not granted it, once, so the op refuses by name (test-moe-pp-staging-refusal). The row-map staging, the
    // per-expert path's output staging, and its F32 activation staging at whichever site allocates it first.
    GGML_SYCL_Q1_NVFP4_TEST_FAILURE_PP_ROW_MAP_STAGING,
    GGML_SYCL_Q1_NVFP4_TEST_FAILURE_PP_DST_STAGING,
    GGML_SYCL_Q1_NVFP4_TEST_FAILURE_PP_SRC1_F32_STAGING,
};

#if defined(__GNUC__) || defined(__clang__)
#    define GGML_SYCL_Q1_TEST_LOCAL __attribute__((visibility("hidden")))
#else
#    define GGML_SYCL_Q1_TEST_LOCAL
#endif

// Tokens are minted for one backend/context/full-model-token tuple, must be
// explicitly entered, and are revoked by leave or backend destruction. A token
// from another backend, context, slot generation, model, or load fails shut.
GGML_SYCL_Q1_TEST_LOCAL bool ggml_sycl_q1_nvfp4_test_scope_mint(
    ggml_backend_t backend, ggml_sycl_exec_context_id context,
    ggml_sycl_model_token model, ggml_sycl_q1_nvfp4_test_scope_token * token);
GGML_SYCL_Q1_TEST_LOCAL bool ggml_sycl_q1_nvfp4_test_scope_enter(
    ggml_backend_t backend, const ggml_sycl_q1_nvfp4_test_scope_token * token);
GGML_SYCL_Q1_TEST_LOCAL void ggml_sycl_q1_nvfp4_test_scope_leave(
    ggml_backend_t backend, const ggml_sycl_q1_nvfp4_test_scope_token * token);
GGML_SYCL_Q1_TEST_LOCAL void ggml_sycl_q1_nvfp4_test_counters_read(
    ggml_sycl_q1_nvfp4_test_counters * counters);
GGML_SYCL_Q1_TEST_LOCAL void ggml_sycl_q1_nvfp4_test_failure_once(
    enum ggml_sycl_q1_nvfp4_test_failure failure);
// While armed, the direct route's next completion object gets a second holder that outlives its host task, as a copy
// of the host-task functor kept by the SYCL runtime would. Disarming releases that holder; held reports whether one
// exists, so a case can show the seam took effect.
GGML_SYCL_Q1_TEST_LOCAL void ggml_sycl_q1_nvfp4_test_hold_completion(bool hold);
GGML_SYCL_Q1_TEST_LOCAL bool ggml_sycl_q1_nvfp4_test_completion_held();
// The most routed-expert triplet groups a single-device placement plan puts on its device, below what its budget
// holds; -1 removes the cap. The planner's real budgets move in steps far above a synthetic expert's few kilobytes, so
// a mixed fixture uses this to get a plan with experts on both the card and the host: the planner's own first-fit pack
// still decides which groups go where, and the fixture reads the split from the plan.
GGML_SYCL_Q1_TEST_LOCAL void ggml_sycl_q1_nvfp4_test_cap_device_expert_groups(int groups);
// Copies the first `count` slots of `weight`'s expert pointer table on `backend`'s device into `out`, after the
// backend's queue has finished. False when the weight has no device table there or it is shorter than `count`.
GGML_SYCL_Q1_TEST_LOCAL bool ggml_sycl_q1_nvfp4_test_read_expert_table(ggml_backend_t      backend,
                                                                       const ggml_tensor * weight,
                                                                       size_t              count,
                                                                       const void **       out);
// Consumes an armed failure if it is this one (counted in failure_consumed). For the backend's own translation
// units outside ggml-sycl.cpp, which arm nothing themselves.
GGML_SYCL_Q1_TEST_LOCAL bool ggml_sycl_q1_nvfp4_test_failure_take(enum ggml_sycl_q1_nvfp4_test_failure failure);
