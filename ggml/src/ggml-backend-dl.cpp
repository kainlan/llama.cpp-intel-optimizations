#include "ggml-backend-dl.h"

#include <mutex>
#include <string>
#include <unordered_set>

namespace {
std::mutex                      g_dl_pin_mutex;
std::unordered_set<std::string> g_dl_pinned_paths;

std::string dl_pin_key(const fs::path & path) {
    std::error_code ec;
    const fs::path canonical = fs::weakly_canonical(path, ec);
    return (ec ? path : canonical).string();
}
}

#ifdef _WIN32

dl_handle * dl_load_library(const fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);
    HMODULE handle = LoadLibraryW(path.wstring().c_str());
    SetErrorMode(old_mode);

    if (handle) {
        try {
            auto policy = reinterpret_cast<ggml_backend_lifetime_policy_v1_t>(
                GetProcAddress(handle, "ggml_backend_lifetime_policy_v1"));
            if (policy && policy() == GGML_BACKEND_LIFETIME_POLICY_PROCESS && !dl_pin_library(handle, path)) {
                FreeLibrary(handle);
                return nullptr;
            }
        } catch (...) {
            FreeLibrary(handle);
            return nullptr;
        }
    }
    return handle;
}

bool dl_pin_library(dl_handle * handle, const fs::path & path) try {
    std::lock_guard<std::mutex> lock(g_dl_pin_mutex);
    const std::string key = dl_pin_key(path);
    if (g_dl_pinned_paths.count(key) != 0) {
        return true;
    }
    HMODULE pinned = nullptr;
    // PIN also acquires a positive module reference and makes it permanent.
    const bool ok = GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                                       reinterpret_cast<LPCWSTR>(handle), &pinned) != 0 && pinned != nullptr;
    if (ok) {
        g_dl_pinned_paths.insert(key);
    }
    return ok;
} catch (...) {
    return false;
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
        try {
            auto policy = reinterpret_cast<ggml_backend_lifetime_policy_v1_t>(
                dlsym(handle, "ggml_backend_lifetime_policy_v1"));
            if (policy && policy() == GGML_BACKEND_LIFETIME_POLICY_PROCESS && !dl_pin_library(handle, path)) {
                dlclose(handle);
                return nullptr;
            }
        } catch (...) {
            dlclose(handle);
            return nullptr;
        }
    }
    return handle;
}

bool dl_pin_library(dl_handle * handle, const fs::path & path) try {
    std::lock_guard<std::mutex> lock(g_dl_pin_mutex);
    const std::string key = dl_pin_key(path);
    if (g_dl_pinned_paths.count(key) != 0) {
        return true;
    }
#if defined(RTLD_NODELETE)
    // Reopen once with NODELETE. The deduplicated positive reference is kept
    // for process lifetime because raw C handles have no ownership release API.
    void * pinned = dlopen(key.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
#else
    void * pinned = dlopen(key.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
    (void) handle;
    if (pinned) {
        g_dl_pinned_paths.insert(key);
    }
    return pinned != nullptr;
} catch (...) {
    return false;
}

void * dl_get_sym(dl_handle * handle, const char * name) {
    return dlsym(handle, name);
}

const char * dl_error() {
    const char * rslt = dlerror();
    return rslt != nullptr ? rslt : "";
}

#endif
