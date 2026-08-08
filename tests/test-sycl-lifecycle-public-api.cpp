// COMPILE-TIME ABI GATE -- not a runtime test (llama.cpp-u2mz).
//
// Every check below is a static_assert, so `main` has exactly one reachable
// exit and the ctest entry `sycl-lifecycle-public-wrappers` cannot go red for
// any ABI reason. THE GATE IS THE BUILD: to exercise it, change a signature in
// ggml-sycl.h and confirm `ninja test-sycl-lifecycle-public-api` fails to
// compile. Do NOT run ctest to check it -- ctest would happily run a stale
// binary from a previous build and print a pass.
//
// Retained rather than deleted because it pins the public C ABI, which is real
// coverage that nothing else provides. Its one honest limitation: the operands
// of `decltype(&f)` are unevaluated, so these assertions constrain the declared
// SIGNATURES only -- they do not odr-use the functions and therefore prove
// nothing about whether the symbols link. Giving it that property would mean
// linking ggml-sycl into this target, which is not possible under
// GGML_BACKEND_DL (the backend is then a MODULE library); see the note on its
// registration in ggml/src/ggml-sycl/CMakeLists.txt.

#include "ggml-sycl.h"

#include <cstdio>
#include <type_traits>

using begin_fn = ggml_sycl_lifecycle_result (*)(ggml_sycl_load_txn *);
using end_fn = ggml_sycl_lifecycle_result (*)(ggml_sycl_load_txn, bool, ggml_sycl_model_token *);
using teardown_fn = ggml_sycl_lifecycle_result (*)(ggml_sycl_model_token);
using usage_abi_fn = void (*)(const char *, ggml_backend_sycl_tensor_usage);
using usage_status_fn = bool (*)(const char *, ggml_backend_sycl_tensor_usage);
using exec_create_fn = ggml_sycl_execution_result (*)(ggml_sycl_exec_context_id *);
using exec_bind_fn = ggml_sycl_execution_result (*)(ggml_backend_t, ggml_sycl_exec_context_id);
using exec_extract_fn = ggml_sycl_execution_result (*)(ggml_sycl_exec_context_id, ggml_sycl_execution_snapshot *);
using exec_close_if_idle_fn = ggml_sycl_execution_result (*)(ggml_sycl_exec_context_id);
using exec_begin_drain_fn = ggml_sycl_execution_result (*)(ggml_sycl_exec_context_id, ggml_sycl_exec_drain_ticket *);
using exec_extract_allocs_fn = ggml_sycl_execution_result (*)(ggml_sycl_exec_drain_ticket *, ggml_sycl_exec_control_host_alloc_batch *);
using exec_finish_drain_fn = ggml_sycl_execution_result (*)(ggml_sycl_exec_drain_ticket, ggml_sycl_exec_control_host_alloc_batch *);
using exec_begin_reset_fn = ggml_sycl_execution_result (*)(ggml_sycl_exec_context_id, ggml_sycl_exec_session_id, ggml_sycl_exec_session_reset_epoch, ggml_sycl_exec_reset_ticket *);
using exec_finish_reset_fn = ggml_sycl_execution_result (*)(ggml_sycl_exec_reset_ticket, ggml_sycl_exec_session_reset_epoch *);

static_assert(std::is_same_v<decltype(&ggml_backend_sycl_model_load_begin), begin_fn>);
static_assert(std::is_same_v<decltype(&ggml_backend_sycl_model_load_end), end_fn>);
static_assert(std::is_same_v<decltype(&ggml_backend_sycl_model_unloaded_token), teardown_fn>);
static_assert(std::is_same_v<decltype(&ggml_backend_sycl_model_quarantine_token), teardown_fn>);
static_assert(std::is_same_v<decltype(&ggml_backend_sycl_register_weight_usage), usage_abi_fn>);
static_assert(std::is_same_v<decltype(&ggml_backend_sycl_try_register_weight_usage), usage_status_fn>);
static_assert(std::is_same_v<decltype(&ggml_backend_sycl_execution_context_create), exec_create_fn>);
static_assert(std::is_same_v<decltype(&ggml_backend_sycl_execution_context_bind_backend), exec_bind_fn>);
static_assert(std::is_same_v<decltype(&ggml_backend_sycl_execution_context_extract), exec_extract_fn>);
static_assert(std::is_same_v<decltype(&ggml_backend_sycl_execution_context_close_if_idle), exec_close_if_idle_fn>);
static_assert(std::is_same_v<decltype(&ggml_backend_sycl_execution_context_begin_drain), exec_begin_drain_fn>);
static_assert(std::is_same_v<decltype(&ggml_backend_sycl_execution_context_extract_control_host_allocs), exec_extract_allocs_fn>);
static_assert(std::is_same_v<decltype(&ggml_backend_sycl_execution_context_finish_drain), exec_finish_drain_fn>);
static_assert(std::is_same_v<decltype(&ggml_backend_sycl_execution_session_begin_reset), exec_begin_reset_fn>);
static_assert(std::is_same_v<decltype(&ggml_backend_sycl_execution_session_finish_reset), exec_finish_reset_fn>);
static_assert(GGML_SYCL_LIFECYCLE_BUSY != GGML_SYCL_LIFECYCLE_EFFECT_FAILED);
static_assert(GGML_SYCL_EXECUTION_DEVICE_BUSY != GGML_SYCL_EXECUTION_BUSY);
static_assert(GGML_SYCL_LIFECYCLE_FOREIGN_BACKEND != GGML_SYCL_LIFECYCLE_STALE_IDENTITY);

int main() {
    // Deliberately says what it did and did not prove. A silent `return 0` here
    // was indistinguishable from a stale binary.
    std::puts(
        "PASS: public lifecycle/execution ABI signatures (18 static_asserts, "
        "verified at COMPILE time -- this run proves only that a binary exists)");
    return 0;
}
