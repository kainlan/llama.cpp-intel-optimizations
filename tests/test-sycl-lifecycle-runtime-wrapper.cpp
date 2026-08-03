#include "ggml-sycl.h"

#include <cstdio>

int main() {
    ggml_sycl_model_token zero{};
    if (ggml_backend_sycl_model_quarantine_token(zero) != GGML_SYCL_LIFECYCLE_STALE_IDENTITY) {
        std::fprintf(stderr, "zero quarantine token accepted\n");
        return 1;
    }
    ggml_sycl_model_token unknown{ 0x1234, 0x5678, 0, 1 };
    if (ggml_backend_sycl_model_quarantine_token(unknown) != GGML_SYCL_LIFECYCLE_STALE_IDENTITY) {
        std::fprintf(stderr, "unknown quarantine token accepted\n");
        return 1;
    }

    // Exercise production-linked begin/end rollback effects (including the
    // explicit no-plan candidate) rather than only stale-token rejection.
    ggml_sycl_load_txn aborted{};
    if (ggml_backend_sycl_model_load_begin(&aborted) != GGML_SYCL_LIFECYCLE_OK || aborted.id == 0) {
        std::fprintf(stderr, "runtime begin failed\n");
        return 1;
    }
    const auto abort_rc = ggml_backend_sycl_model_load_end(aborted, false, nullptr);
    if (abort_rc != GGML_SYCL_LIFECYCLE_MISSING_SUCCESS && abort_rc != GGML_SYCL_LIFECYCLE_POISONED) {
        std::fprintf(stderr, "runtime abort returned %d\n", (int) abort_rc);
        return 1;
    }

    ggml_sycl_load_txn    committed{};
    ggml_sycl_model_token token{};
    if (ggml_backend_sycl_model_load_begin(&committed) != GGML_SYCL_LIFECYCLE_OK ||
        ggml_backend_sycl_model_load_end(committed, true, &token) != GGML_SYCL_LIFECYCLE_OK || token.model_id == 0) {
        std::fprintf(stderr, "runtime no-allocation commit failed\n");
        return 1;
    }
    if (ggml_backend_sycl_model_unloaded_token(token) != GGML_SYCL_LIFECYCLE_OK) {
        std::fprintf(stderr, "runtime teardown failed\n");
        return 1;
    }
    const auto repeat = ggml_backend_sycl_model_unloaded_token(token);
    if (repeat != GGML_SYCL_LIFECYCLE_OK_ALREADY_DEAD && repeat != GGML_SYCL_LIFECYCLE_STALE_IDENTITY) {
        std::fprintf(stderr, "runtime repeated teardown returned %d\n", (int) repeat);
        return 1;
    }
    return 0;
}
