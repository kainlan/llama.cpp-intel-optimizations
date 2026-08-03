#include "ggml-sycl.h"

#include <type_traits>

using begin_fn = ggml_sycl_lifecycle_result (*)(ggml_sycl_load_txn *);
using end_fn = ggml_sycl_lifecycle_result (*)(ggml_sycl_load_txn, bool, ggml_sycl_model_token *);
using teardown_fn = ggml_sycl_lifecycle_result (*)(ggml_sycl_model_token);

static_assert(std::is_same_v<decltype(&ggml_backend_sycl_model_load_begin), begin_fn>);
static_assert(std::is_same_v<decltype(&ggml_backend_sycl_model_load_end), end_fn>);
static_assert(std::is_same_v<decltype(&ggml_backend_sycl_model_unloaded_token), teardown_fn>);
static_assert(std::is_same_v<decltype(&ggml_backend_sycl_model_quarantine_token), teardown_fn>);
static_assert(GGML_SYCL_LIFECYCLE_BUSY != GGML_SYCL_LIFECYCLE_EFFECT_FAILED);

int main() {
    return 0;
}
