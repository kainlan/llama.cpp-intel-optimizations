#include "ggml-sycl.h"

#include <cstdio>

#if defined(GGML_SYCL_RUNTIME_MODULE)
#    include <dlfcn.h>
#endif

int main() {
#if defined(GGML_SYCL_RUNTIME_MODULE)
    void * module = dlopen(GGML_SYCL_RUNTIME_MODULE, RTLD_NOW | RTLD_LOCAL);
    if (!module) {
        std::fprintf(stderr, "failed to load SYCL backend module: %s\n", dlerror());
        return 1;
    }
#    define LOAD_SYCL(name)                                                       \
        auto name##_fn = reinterpret_cast<decltype(&name)>(dlsym(module, #name)); \
        if (!name##_fn) {                                                         \
            std::fprintf(stderr, "missing module symbol %s\n", #name);            \
            return 1;                                                             \
        }
    LOAD_SYCL(ggml_backend_sycl_model_quarantine_token)
    LOAD_SYCL(ggml_backend_sycl_model_load_begin)
    LOAD_SYCL(ggml_backend_sycl_model_load_end)
    LOAD_SYCL(ggml_backend_sycl_model_unloaded_token)
#    define CALL_SYCL(name) name##_fn
#else
#    define CALL_SYCL(name) name
#endif

    ggml_sycl_model_token zero{};
    if (CALL_SYCL(ggml_backend_sycl_model_quarantine_token)(zero) != GGML_SYCL_LIFECYCLE_STALE_IDENTITY) {
        std::fprintf(stderr, "zero quarantine token accepted\n");
        return 1;
    }
    ggml_sycl_model_token unknown{ 0x1234, 0x5678, 0, 1 };
    if (CALL_SYCL(ggml_backend_sycl_model_quarantine_token)(unknown) != GGML_SYCL_LIFECYCLE_STALE_IDENTITY) {
        std::fprintf(stderr, "unknown quarantine token accepted\n");
        return 1;
    }

    ggml_sycl_load_txn aborted{};
    if (CALL_SYCL(ggml_backend_sycl_model_load_begin)(&aborted) != GGML_SYCL_LIFECYCLE_OK || aborted.id == 0) {
        std::fprintf(stderr, "runtime begin failed\n");
        return 1;
    }
    const auto abort_rc = CALL_SYCL(ggml_backend_sycl_model_load_end)(aborted, false, nullptr);
    if (abort_rc != GGML_SYCL_LIFECYCLE_MISSING_SUCCESS && abort_rc != GGML_SYCL_LIFECYCLE_POISONED) {
        std::fprintf(stderr, "runtime abort returned %d\n", (int) abort_rc);
        return 1;
    }

    ggml_sycl_load_txn    committed{};
    ggml_sycl_model_token token{};
    if (CALL_SYCL(ggml_backend_sycl_model_load_begin)(&committed) != GGML_SYCL_LIFECYCLE_OK ||
        CALL_SYCL(ggml_backend_sycl_model_load_end)(committed, true, &token) != GGML_SYCL_LIFECYCLE_OK ||
        token.model_id == 0) {
        std::fprintf(stderr, "runtime no-allocation commit failed\n");
        return 1;
    }
    if (CALL_SYCL(ggml_backend_sycl_model_unloaded_token)(token) != GGML_SYCL_LIFECYCLE_OK) {
        std::fprintf(stderr, "runtime teardown failed\n");
        return 1;
    }
    const auto repeat = CALL_SYCL(ggml_backend_sycl_model_unloaded_token)(token);
    if (repeat != GGML_SYCL_LIFECYCLE_OK_ALREADY_DEAD && repeat != GGML_SYCL_LIFECYCLE_STALE_IDENTITY) {
        std::fprintf(stderr, "runtime repeated teardown returned %d\n", (int) repeat);
        return 1;
    }
#if defined(GGML_SYCL_RUNTIME_MODULE)
    dlclose(module);
#endif
    return 0;
}
