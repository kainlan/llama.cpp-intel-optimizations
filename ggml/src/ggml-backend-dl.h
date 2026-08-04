#pragma once

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#else
#    include <dlfcn.h>
#    include <unistd.h>
#endif
#include <cstdint>
#include <filesystem>

namespace fs = std::filesystem;

#ifdef _WIN32

using dl_handle = std::remove_pointer_t<HMODULE>;

struct dl_handle_deleter {
    void operator()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

#else

using dl_handle = void;

struct dl_handle_deleter {
    void operator()(void * handle) {
        dlclose(handle);
    }
};

#endif

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

// Versioned backend lifetime policy queried immediately after RTLD_NOW open,
// before score/init or any backend-owned destructor can be registered.
#define GGML_BACKEND_LIFETIME_POLICY_ABI_V1 1u
#define GGML_BACKEND_LIFETIME_POLICY_NORMAL 0u
#define GGML_BACKEND_LIFETIME_POLICY_PROCESS 1u
using ggml_backend_lifetime_policy_v1_t = uint32_t (*)(void);

dl_handle * dl_load_library(const fs::path & path);
// Pin an already-open module for process lifetime. Used both for the pre-score
// export and the initialized-registry compatibility fallback.
bool dl_pin_library(dl_handle * handle, const fs::path & path);
void * dl_get_sym(dl_handle * handle, const char * name);
const char * dl_error();

