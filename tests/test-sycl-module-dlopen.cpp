#include <cstdio>

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

#ifndef GGML_SYCL_RUNTIME_MODULE
#    error "GGML_SYCL_RUNTIME_MODULE must name the dynamic SYCL backend"
#endif

int main() {
#if defined(_WIN32)
    HMODULE module = LoadLibraryA(GGML_SYCL_RUNTIME_MODULE);
    if (!module) {
        std::fprintf(stderr, "LoadLibrary failed for %s: error %lu\n", GGML_SYCL_RUNTIME_MODULE,
                     static_cast<unsigned long>(GetLastError()));
        return 1;
    }
    using lifetime_policy_fn = unsigned int (*)();
    auto policy = reinterpret_cast<lifetime_policy_fn>(GetProcAddress(module, "ggml_backend_lifetime_policy_v1"));
    if (!policy || policy() != 1u) {
        std::fprintf(stderr, "missing PROCESS_LIFETIME policy export\n");
        FreeLibrary(module);
        return 1;
    }
    HMODULE pinned = nullptr;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                            reinterpret_cast<LPCSTR>(policy), &pinned)) {
        std::fprintf(stderr, "failed to pin module: error %lu\n", static_cast<unsigned long>(GetLastError()));
        FreeLibrary(module);
        return 1;
    }
    FreeLibrary(module); // logical unload; the positive pinned reference remains
#else
    dlerror();
    void * module = dlopen(GGML_SYCL_RUNTIME_MODULE, RTLD_NOW | RTLD_LOCAL);
    if (!module) {
        const char * error = dlerror();
        std::fprintf(stderr, "dlopen(RTLD_NOW) failed for %s: %s\n", GGML_SYCL_RUNTIME_MODULE,
                     error ? error : "unknown error");
        return 1;
    }
    using lifetime_policy_fn = unsigned int (*)();
    auto policy = reinterpret_cast<lifetime_policy_fn>(dlsym(module, "ggml_backend_lifetime_policy_v1"));
    if (!policy || policy() != 1u) {
        std::fprintf(stderr, "missing PROCESS_LIFETIME policy export\n");
        dlclose(module);
        return 1;
    }
#    if defined(RTLD_NODELETE)
    void * pinned = dlopen(GGML_SYCL_RUNTIME_MODULE, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
#    else
    void * pinned = dlopen(GGML_SYCL_RUNTIME_MODULE, RTLD_NOW | RTLD_LOCAL);
#    endif
    if (!pinned) {
        std::fprintf(stderr, "failed to process-pin validated SYCL module\n");
        dlclose(module);
        return 1;
    }
    dlclose(module); // logical unload; intentionally retain the positive pin
#endif
    return 0;
}
