#include "ggml-sycl.h"

#include <type_traits>

using begin_fn = ggml_sycl_lifecycle_result (*)(ggml_sycl_load_txn *);
using end_fn = ggml_sycl_lifecycle_result (*)(ggml_sycl_load_txn, bool, ggml_sycl_model_token *);
using teardown_fn = ggml_sycl_lifecycle_result (*)(ggml_sycl_model_token);
using usage_abi_fn = void (*)(const char *, ggml_backend_sycl_tensor_usage);
using usage_status_fn = bool (*)(const char *, ggml_backend_sycl_tensor_usage);

static_assert(std::is_same_v<decltype(&ggml_backend_sycl_model_load_begin), begin_fn>);
static_assert(std::is_same_v<decltype(&ggml_backend_sycl_model_load_end), end_fn>);
static_assert(std::is_same_v<decltype(&ggml_backend_sycl_model_unloaded_token), teardown_fn>);
static_assert(std::is_same_v<decltype(&ggml_backend_sycl_model_quarantine_token), teardown_fn>);
static_assert(std::is_same_v<decltype(&ggml_backend_sycl_register_weight_usage), usage_abi_fn>);
static_assert(std::is_same_v<decltype(&ggml_backend_sycl_try_register_weight_usage), usage_status_fn>);
static_assert(GGML_SYCL_LIFECYCLE_BUSY != GGML_SYCL_LIFECYCLE_EFFECT_FAILED);
static_assert(GGML_SYCL_LIFECYCLE_FOREIGN_BACKEND != GGML_SYCL_LIFECYCLE_STALE_IDENTITY);

int main() {
    return 0;
}
