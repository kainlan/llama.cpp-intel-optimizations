#include "ggml-backend-dl.h"

#ifdef _WIN32

dl_handle * dl_load_library(const fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);
    HMODULE handle = LoadLibraryW(path.wstring().c_str());
    SetErrorMode(old_mode);

    if (handle) {
        auto policy = reinterpret_cast<ggml_backend_lifetime_policy_v1_t>(
            GetProcAddress(handle, "ggml_backend_lifetime_policy_v1"));
        if (policy && policy() == GGML_BACKEND_LIFETIME_POLICY_PROCESS && !dl_pin_library(handle, path)) {
            FreeLibrary(handle);
            return nullptr;
        }
    }
    return handle;
}

bool dl_pin_library(dl_handle * handle, const fs::path & path) {
    (void) path;
    HMODULE pinned = nullptr;
    // PIN also acquires a positive module reference and makes it permanent.
    return GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                              reinterpret_cast<LPCWSTR>(handle), &pinned) != 0 && pinned != nullptr;
}

void * dl_get_sym(dl_handle * handle, const char * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);
    void * p = (void *) GetProcAddress(handle, name);
    SetErrorMode(old_mode);
    return p;
}

const char * dl_error() {
    return "";
}

#else

dl_handle * dl_load_library(const fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle) {
        auto policy = reinterpret_cast<ggml_backend_lifetime_policy_v1_t>(
            dlsym(handle, "ggml_backend_lifetime_policy_v1"));
        if (policy && policy() == GGML_BACKEND_LIFETIME_POLICY_PROCESS && !dl_pin_library(handle, path)) {
            dlclose(handle);
            return nullptr;
        }
    }
    return handle;
}

bool dl_pin_library(dl_handle * handle, const fs::path & path) {
#if defined(RTLD_NODELETE)
    // Reopen with NOW to retain dependency validation while upgrading the
    // module mapping to process lifetime. Intentionally retain this positive
    // reference; the pin store must never participate in static destruction.
    void * pinned = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
#else
    // NODELETE is unavailable: retain a positive reference to the same DSO.
    void * pinned = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
    (void) handle;
    return pinned != nullptr;
}

void * dl_get_sym(dl_handle * handle, const char * name) {
    return dlsym(handle, name);
}

const char * dl_error() {
    const char * rslt = dlerror();
    return rslt != nullptr ? rslt : "";
}

#endif
