#pragma once

// Private BUILD_TESTING-only seam for exercising the closed Q1/NVFP4 direct
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
};

enum ggml_sycl_q1_nvfp4_test_failure {
    GGML_SYCL_Q1_NVFP4_TEST_FAILURE_NONE = 0,
    GGML_SYCL_Q1_NVFP4_TEST_FAILURE_PRE_MARK,
    GGML_SYCL_Q1_NVFP4_TEST_FAILURE_POST_MARK,
    GGML_SYCL_Q1_NVFP4_TEST_FAILURE_ASYNC_TERMINAL,
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
