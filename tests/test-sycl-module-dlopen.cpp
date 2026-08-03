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
    FreeLibrary(module);
#else
    dlerror();
    void * module = dlopen(GGML_SYCL_RUNTIME_MODULE, RTLD_NOW | RTLD_LOCAL);
    if (!module) {
        const char * error = dlerror();
        std::fprintf(stderr, "dlopen(RTLD_NOW) failed for %s: %s\n", GGML_SYCL_RUNTIME_MODULE,
                     error ? error : "unknown error");
        return 1;
    }
    dlclose(module);
#endif
    return 0;
}
