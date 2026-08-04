#include "ggml-backend-dl.h"

#ifdef _WIN32

dl_handle * dl_load_library(const fs::path & path, bool keep_loaded) {
    (void) keep_loaded;
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    return handle;
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

dl_handle * dl_load_library(const fs::path & path, bool keep_loaded) {
    int flags = RTLD_NOW | RTLD_LOCAL;
#    if defined(RTLD_NODELETE)
    if (keep_loaded) {
        flags |= RTLD_NODELETE;
    }
#    else
    (void) keep_loaded;
#    endif
    dl_handle * handle = dlopen(path.string().c_str(), flags);
    return handle;
}

void * dl_get_sym(dl_handle * handle, const char * name) {
    return dlsym(handle, name);
}

const char * dl_error() {
    const char *rslt = dlerror();
    return rslt != nullptr ? rslt : "";
}

#endif
