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
    return 0;
}
