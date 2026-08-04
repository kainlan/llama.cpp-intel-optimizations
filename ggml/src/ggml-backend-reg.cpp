#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-backend-dl.h"
#include "ggml-impl.h"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>
#include <cctype>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#elif defined(__APPLE__)
#    include <mach-o/dyld.h>
#    include <dlfcn.h>
#else
#    include <dlfcn.h>
#    include <unistd.h>
#endif

// Backend registry
#ifdef GGML_USE_CPU
#include "ggml-cpu.h"
#endif

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#ifdef GGML_USE_SYCL
#include "ggml-sycl.h"
#endif

#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#ifdef GGML_USE_WEBGPU
#include "ggml-webgpu.h"
#endif

#ifdef GGML_USE_ZDNN
#include "ggml-zdnn.h"
#endif

#ifdef GGML_USE_OPENCL
#include "ggml-opencl.h"
#endif

#ifdef GGML_USE_HEXAGON
#include "ggml-hexagon.h"
#endif

#ifdef GGML_USE_BLAS
#include "ggml-blas.h"
#endif

#ifdef GGML_USE_RPC
#include "ggml-rpc.h"
#endif

#ifdef GGML_USE_VIRTGPU_FRONTEND
#include "ggml-virtgpu.h"
#endif

#ifdef GGML_USE_CANN
#include "ggml-cann.h"
#endif

#ifdef GGML_USE_ZENDNN
#include "ggml-zendnn.h"
#endif

#ifdef GGML_USE_OPENVINO
#include "ggml-openvino.h"
#endif

namespace fs = std::filesystem;

static std::atomic<bool> g_disable_device_backends{false};

// Deterministic host-test barrier: blocks one checked unload while it owns the
// registry transaction mutex, proving loads and enumeration readers cannot
// overlap its vector mutation.
static std::mutex              g_registry_test_mutex;
static std::condition_variable g_registry_test_cv;
static bool                    g_registry_test_block_next_unload = false;
static bool                    g_registry_test_unload_blocked = false;
static bool                    g_registry_test_release_unload = false;
static bool                    g_registry_test_reentrant_mutation = false;

extern "C" void ggml_backend_test_block_next_unload() {
    std::lock_guard<std::mutex> lock(g_registry_test_mutex);
    g_registry_test_block_next_unload = true;
    g_registry_test_unload_blocked = false;
    g_registry_test_release_unload = false;
}

extern "C" void ggml_backend_test_wait_unload_blocked() {
    std::unique_lock<std::mutex> lock(g_registry_test_mutex);
    g_registry_test_cv.wait(lock, [] { return g_registry_test_unload_blocked; });
}

extern "C" void ggml_backend_test_reentrant_mutation_on_next_unload() {
    std::lock_guard<std::mutex> lock(g_registry_test_mutex);
    g_registry_test_reentrant_mutation = true;
}

extern "C" void ggml_backend_test_release_unload() {
    std::lock_guard<std::mutex> lock(g_registry_test_mutex);
    g_registry_test_release_unload = true;
    g_registry_test_cv.notify_all();
}

static void ggml_backend_test_unload_barrier() {
    std::unique_lock<std::mutex> lock(g_registry_test_mutex);
    if (!g_registry_test_block_next_unload) {
        return;
    }
    g_registry_test_unload_blocked = true;
    g_registry_test_cv.notify_all();
    g_registry_test_cv.wait(lock, [] { return g_registry_test_release_unload; });
    g_registry_test_block_next_unload = false;
    g_registry_test_unload_blocked = false;
    g_registry_test_release_unload = false;
}

bool ggml_backend_device_backends_disabled(void) {
    if (g_disable_device_backends.load(std::memory_order_acquire)) {
        return true;
    }

    const char * env = std::getenv("GGML_BACKEND_CPU_ONLY");
    return env != nullptr && std::strcmp(env, "0") != 0;
}

static std::string path_str(const fs::path & path) {
    try {
#if defined(__cpp_lib_char8_t)
        // C++20 and later: u8string() returns std::u8string
        const std::u8string u8str = path.u8string();
        return std::string(reinterpret_cast<const char *>(u8str.data()), u8str.size());
#else
        // C++17: u8string() returns std::string
        return path.u8string();
#endif
    } catch (...) {
        return std::string();
    }
}

static bool striequals(const char * a, const char * b);

struct ggml_backend_reg_entry {
    ggml_backend_reg_t reg;
    dl_handle_ptr      handle;
};

using ggml_backend_reg_entry_ptr = std::shared_ptr<ggml_backend_reg_entry>;

struct ggml_backend_device_entry {
    ggml_backend_dev_t           dev;
    ggml_backend_reg_entry_ptr   owner;
};

// Raw public registry/device handles cannot carry a C++ lease. Retain every
// dynamic entry returned to a thread so its DSO remains mapped for as long as
// that thread can use the raw handle. Logical removal is still immediate: all
// lookups validate against the live vectors before adding a lease.
static thread_local std::vector<ggml_backend_reg_entry_ptr> g_backend_raw_handle_leases;

struct ggml_backend_registry {
    mutable std::recursive_mutex             mutex;
    std::vector<ggml_backend_reg_entry_ptr>  backends;
    std::vector<ggml_backend_device_entry>   devices;

    ggml_backend_registry() {
        // Only referenced inside the per-backend GGML_USE_* blocks below, so it
        // is unused in a CPU-only build.
        [[maybe_unused]] const bool disable_device_backends = ggml_backend_device_backends_disabled();

#ifdef GGML_USE_CUDA
        if (!disable_device_backends) {
            register_backend(ggml_backend_cuda_reg());
        }
#endif
#ifdef GGML_USE_METAL
        if (!disable_device_backends) {
            register_backend(ggml_backend_metal_reg());
        }
#endif
#ifdef GGML_USE_SYCL
        if (!disable_device_backends) {
            register_backend(ggml_backend_sycl_reg());
        }
#endif
#ifdef GGML_USE_VULKAN
    // Add runtime disable check
    if (!disable_device_backends && getenv("GGML_DISABLE_VULKAN") == nullptr) {
        register_backend(ggml_backend_vk_reg());
    } else if (!disable_device_backends) {
        GGML_LOG_DEBUG("Vulkan backend disabled by GGML_DISABLE_VULKAN environment variable\n");
    }
#endif
#ifdef GGML_USE_WEBGPU
        if (!disable_device_backends) {
            register_backend(ggml_backend_webgpu_reg());
        }
#endif
#ifdef GGML_USE_ZDNN
        if (!disable_device_backends) {
            register_backend(ggml_backend_zdnn_reg());
        }
#endif
#ifdef GGML_USE_VIRTGPU_FRONTEND
        if (!disable_device_backends) {
            register_backend(ggml_backend_virtgpu_reg());
        }
#endif

#ifdef GGML_USE_OPENCL
        if (!disable_device_backends) {
            register_backend(ggml_backend_opencl_reg());
        }
#endif
#ifdef GGML_USE_ZENDNN
        register_backend(ggml_backend_zendnn_reg());
#endif
#ifdef GGML_USE_HEXAGON
        if (!disable_device_backends) {
            register_backend(ggml_backend_hexagon_reg());
        }
#endif
#ifdef GGML_USE_CANN
        if (!disable_device_backends) {
            register_backend(ggml_backend_cann_reg());
        }
#endif
#ifdef GGML_USE_BLAS
        register_backend(ggml_backend_blas_reg());
#endif
#ifdef GGML_USE_RPC
        if (!disable_device_backends) {
            register_backend(ggml_backend_rpc_reg());
        }
#endif
#ifdef GGML_USE_OPENVINO
        if (!disable_device_backends) {
            register_backend(ggml_backend_openvino_reg());
        }
#endif
#ifdef GGML_USE_CPU
        register_backend(ggml_backend_cpu_reg());
#endif
    }

    ~ggml_backend_registry() {
        // FIXME: backends cannot be safely unloaded without a function to destroy all the backend resources,
        // since backend threads may still be running and accessing resources from the dynamic library
        for (auto & entry : backends) {
            if (entry->handle) {
                entry->handle.release(); // NOLINT
            }
        }
    }

    void register_backend(ggml_backend_reg_t reg, dl_handle_ptr handle = nullptr) {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        if (!reg) {
            return;
        }

        for (auto & entry : backends) {
            if (entry->reg == reg) {
                return;
            }
        }

#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
            __func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_count(reg));
#endif
        auto entry = std::make_shared<ggml_backend_reg_entry>(ggml_backend_reg_entry{ reg, std::move(handle) });
        backends.push_back(entry);
        for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); i++) {
            register_device(ggml_backend_reg_dev_get(reg, i), entry);
        }
    }

    void register_device(ggml_backend_dev_t device, ggml_backend_reg_entry_ptr owner = {}) {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        if (!owner && device) {
            const auto reg = ggml_backend_dev_backend_reg(device);
            const auto it = std::find_if(backends.begin(), backends.end(),
                [reg](const ggml_backend_reg_entry_ptr & entry) { return entry->reg == reg; });
            if (it != backends.end()) {
                owner = *it;
            }
        }
        for (auto & entry : devices) {
            if (entry.dev == device) {
                return;
            }
        }

#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: registered device %s (%s)\n", __func__, ggml_backend_dev_name(device), ggml_backend_dev_description(device));
#endif
        devices.push_back({ device, std::move(owner) });
    }

    ggml_backend_reg_t load_backend(const fs::path & path, bool silent) {
        // Serialize dlopen/init/register with the full checked-unload
        // reservation/shutdown/erase/completion transaction.
        std::lock_guard<std::recursive_mutex> lock(mutex);
        dl_handle_ptr handle{ dl_load_library(path) };
        if (!handle) {
            if (!silent) {
                GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(path).c_str(), dl_error());
            }
            return nullptr;
        }

        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
        if (score_fn && score_fn() == 0) {
            if (!silent) {
                GGML_LOG_INFO("%s: backend %s is not supported on this system\n", __func__, path_str(path).c_str());
            }
            return nullptr;
        }

        auto backend_init_fn = (ggml_backend_init_t) dl_get_sym(handle.get(), "ggml_backend_init");
        if (!backend_init_fn) {
            if (!silent) {
                GGML_LOG_ERROR("%s: failed to find ggml_backend_init in %s\n", __func__, path_str(path).c_str());
            }
            return nullptr;
        }

        ggml_backend_reg_t reg = backend_init_fn();
        // Compatibility fallback for an older initialized SYCL registry that
        // predates the pre-score export. Renamed DSOs remain safe because this
        // uses registry identity, never the filename.
        if (reg && !dl_get_sym(handle.get(), "ggml_backend_lifetime_policy_v1") &&
            std::strcmp(ggml_backend_reg_name(reg), "SYCL") == 0 && !dl_pin_library(handle.get(), path)) {
            if (!silent) {
                GGML_LOG_ERROR("%s: failed to process-pin SYCL backend %s\n", __func__, path_str(path).c_str());
            }
            return nullptr;
        }
        if (!reg || reg->api_version != GGML_BACKEND_API_VERSION) {
            if (!silent) {
                if (!reg) {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: ggml_backend_init returned NULL\n",
                        __func__, path_str(path).c_str());
                } else {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: incompatible API version (backend: %d, current: %d)\n",
                        __func__, path_str(path).c_str(), reg->api_version, GGML_BACKEND_API_VERSION);
                }
            }
            return nullptr;
        }

        GGML_LOG_INFO("%s: loaded %s backend from %s\n", __func__, ggml_backend_reg_name(reg), path_str(path).c_str());

        register_backend(reg, std::move(handle));

        return reg;
    }

    ggml_backend_unload_result unload_backend(ggml_backend_reg_t reg, bool silent) {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        ggml_backend_test_unload_barrier();
        auto it = std::find_if(backends.begin(), backends.end(),
                               [reg](const ggml_backend_reg_entry_ptr & entry) { return entry->reg == reg; });

        if (it == backends.end()) {
            if (!silent) {
                GGML_LOG_ERROR("%s: backend not found\n", __func__);
            }
            return GGML_BACKEND_UNLOAD_NOT_FOUND;
        }

        if (!silent) {
            GGML_LOG_DEBUG("%s: unloading %s backend\n", __func__, ggml_backend_reg_name(reg));
        }
        // Capture lifetime-owned identity before any reentrant plugin callback.
        const ggml_backend_reg_entry_ptr unloading = *it;

        // A module with exact live lifecycle owners must retain both its
        // devices and procedure table so model destruction can still perform
        // exact-token teardown. This optional typed gate runs before shutdown.
        using can_unload_fn = bool (*)();
        auto can_unload = reinterpret_cast<can_unload_fn>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_can_unload"));
        bool shutdown_reserved = false;
        bool eligibility_threw = false;
        try {
            if (can_unload) {
                shutdown_reserved = can_unload();
                if (!shutdown_reserved) {
                    return GGML_BACKEND_UNLOAD_BUSY;
                }
            }
        } catch (...) {
            // Conservatively assume a throwing hook reserved admission before
            // it failed; cancellation below is required to make retry possible.
            shutdown_reserved = can_unload != nullptr;
            eligibility_threw = true;
        }

        // Give a dynamic backend one last chance to join module-owned threads
        // and destroy queues/caches while its code and dependent runtimes are
        // still loaded. The hook is optional and must be idempotent.
        using shutdown_fn = void (*)();
        auto shutdown = reinterpret_cast<shutdown_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_shutdown"));
        auto complete = reinterpret_cast<shutdown_fn>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_complete_unload"));
        auto cancel = reinterpret_cast<shutdown_fn>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_cancel_unload"));
        const auto cancel_noexcept = [&] {
            if (shutdown_reserved && cancel) {
                try {
                    cancel();
                } catch (...) {
                }
            }
        };
        if (eligibility_threw || (can_unload && (!shutdown_reserved || !complete)) || (!shutdown && can_unload)) {
            cancel_noexcept();
            return GGML_BACKEND_UNLOAD_BUSY;
        }

        // Reentrant hooks may mutate and reallocate the vectors. Re-find the
        // shared entry identity, then remove it logically before shutdown so
        // recursive lookup/init cannot enter a module being torn down.
        {
            std::lock_guard<std::mutex> test_lock(g_registry_test_mutex);
            if (g_registry_test_reentrant_mutation) {
                // Deterministically emulate a registration callback that grows
                // the vector after iterator capture. Only shared identity may
                // be used after this point.
                backends.reserve(backends.capacity() + 1);
                g_registry_test_reentrant_mutation = false;
            }
        }
        it = std::find(backends.begin(), backends.end(), unloading);
        if (it == backends.end()) {
            cancel_noexcept();
            return GGML_BACKEND_UNLOAD_NOT_FOUND;
        }
        std::vector<ggml_backend_device_entry> removed_devices;
        for (auto dev = devices.begin(); dev != devices.end();) {
            if (dev->owner == unloading || ggml_backend_dev_backend_reg(dev->dev) == reg) {
                removed_devices.push_back(*dev);
                dev = devices.erase(dev);
            } else {
                ++dev;
            }
        }
        backends.erase(it);

        try {
            if (shutdown) {
                shutdown();
            }
        } catch (...) {
            // Restore logical visibility when shutdown itself failed. The
            // shared entry kept the DSO mapped throughout rollback.
            backends.push_back(unloading);
            devices.insert(devices.end(), removed_devices.begin(), removed_devices.end());
            cancel_noexcept();
            return GGML_BACKEND_UNLOAD_BUSY;
        }
        if (complete) {
            try {
                complete();
            } catch (...) {
                // Completion is optional plugin code too. Its failure must not
                // escape or leave admission reserved after logical removal.
                cancel_noexcept();
            }
        }
        return GGML_BACKEND_UNLOAD_OK;
    }

    size_t backend_count() const {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        return backends.size();
    }

    ggml_backend_reg_t backend_get(size_t index) const {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        if (index >= backends.size()) {
            return nullptr;
        }
        g_backend_raw_handle_leases.push_back(backends[index]);
        return backends[index]->reg;
    }

    ggml_backend_reg_t backend_by_name(const char * name) const {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        for (const auto & entry : backends) {
            if (striequals(ggml_backend_reg_name(entry->reg), name)) {
                g_backend_raw_handle_leases.push_back(entry);
                return entry->reg;
            }
        }
        return nullptr;
    }

    size_t device_count() const {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        return devices.size();
    }

    ggml_backend_dev_t device_get(size_t index) const {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        if (index >= devices.size()) {
            return nullptr;
        }
        if (devices[index].owner) {
            g_backend_raw_handle_leases.push_back(devices[index].owner);
        }
        return devices[index].dev;
    }

    ggml_backend_dev_t device_by_name(const char * name) const {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        for (const auto & entry : devices) {
            if (striequals(ggml_backend_dev_name(entry.dev), name)) {
                if (entry.owner) {
                    g_backend_raw_handle_leases.push_back(entry.owner);
                }
                return entry.dev;
            }
        }
        return nullptr;
    }

    ggml_backend_dev_t device_by_type(enum ggml_backend_dev_type type) const {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        for (const auto & entry : devices) {
            if (ggml_backend_dev_type(entry.dev) == type) {
                if (entry.owner) {
                    g_backend_raw_handle_leases.push_back(entry.owner);
                }
                return entry.dev;
            }
        }
        return nullptr;
    }
};

static ggml_backend_registry & get_reg() {
    static ggml_backend_registry reg;
    return reg;
}

// Internal API
void ggml_backend_register(ggml_backend_reg_t reg) {
    get_reg().register_backend(reg);
}

void ggml_backend_device_register(ggml_backend_dev_t device) {
    get_reg().register_device(device);
}

void ggml_backend_disable_device_backends(void) {
    g_disable_device_backends.store(true, std::memory_order_release);
}

// Backend (reg) enumeration
static bool striequals(const char * a, const char * b) {
    for (; *a && *b; a++, b++) {
        if (std::tolower(*a) != std::tolower(*b)) {
            return false;
        }
    }
    return *a == *b;
}

size_t ggml_backend_reg_count() {
    return get_reg().backend_count();
}

ggml_backend_reg_t ggml_backend_reg_get(size_t index) {
    return get_reg().backend_get(index);
}

ggml_backend_reg_t ggml_backend_reg_by_name(const char * name) {
    return get_reg().backend_by_name(name);
}

size_t ggml_backend_dev_count() {
    return get_reg().device_count();
}

ggml_backend_dev_t ggml_backend_dev_get(size_t index) {
    return get_reg().device_get(index);
}

ggml_backend_dev_t ggml_backend_dev_by_name(const char * name) {
    return get_reg().device_by_name(name);
}

ggml_backend_dev_t ggml_backend_dev_by_type(enum ggml_backend_dev_type type) {
    return get_reg().device_by_type(type);
}

// Convenience functions
ggml_backend_t ggml_backend_init_by_name(const char * name, const char * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(name);
    if (!dev) {
        return nullptr;
    }
    return ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_by_type(enum ggml_backend_dev_type type, const char * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(type);
    if (!dev) {
        return nullptr;
    }
    return ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_best(void) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!dev) {
        return nullptr;
    }
    return ggml_backend_dev_init(dev, nullptr);
}

// Dynamic loading
ggml_backend_reg_t ggml_backend_load(const char * path) {
    return get_reg().load_backend(path, false);
}

ggml_backend_unload_result ggml_backend_unload_checked(ggml_backend_reg_t reg) {
    return get_reg().unload_backend(reg, true);
}

void ggml_backend_unload(ggml_backend_reg_t reg) {
    (void) ggml_backend_unload_checked(reg);
}

static fs::path get_executable_path() {
#if defined(__APPLE__)
    // get executable path
    std::vector<char> path;
    uint32_t size;
    while (true) {
        size = path.size();
        if (_NSGetExecutablePath(path.data(), &size) == 0) {
            break;
        }
        path.resize(size);
    }
    std::string base_path(path.data(), size);
    // remove executable name
    auto last_slash = base_path.find_last_of('/');
    if (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    return base_path + "/";
#elif defined(__linux__) || defined(__FreeBSD__)
    std::string base_path = ".";
    std::vector<char> path(1024);
    while (true) {
        // get executable path
#    if defined(__linux__)
        ssize_t len = readlink("/proc/self/exe", path.data(), path.size());
#    elif defined(__FreeBSD__)
        ssize_t len = readlink("/proc/curproc/file", path.data(), path.size());
#    endif
        if (len == -1) {
            break;
        }
        if (len < (ssize_t) path.size()) {
            base_path = std::string(path.data(), len);
            // remove executable name
            auto last_slash = base_path.find_last_of('/');
            if (last_slash != std::string::npos) {
                base_path = base_path.substr(0, last_slash);
            }
            break;
        }
        path.resize(path.size() * 2);
    }

    return base_path + "/";
#elif defined(_WIN32)
    std::vector<wchar_t> path(MAX_PATH);
    DWORD len = GetModuleFileNameW(NULL, path.data(), path.size());
    if (len == 0) {
        return {};
    }
    std::wstring base_path(path.data(), len);
    // remove executable name
    auto last_slash = base_path.find_last_of('\\');
    if (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    return base_path + L"\\";
#else
    return {};
#endif
}

static fs::path backend_filename_prefix() {
#ifdef _WIN32
    return fs::u8path("ggml-");
#else
    return fs::u8path("libggml-");
#endif
}

static fs::path backend_filename_extension() {
#ifdef _WIN32
    return fs::u8path(".dll");
#else
    return fs::u8path(".so");
#endif
}

static ggml_backend_reg_t ggml_backend_load_best(const char * name, bool silent, const char * user_search_path) {
    // enumerate all the files that match [lib]ggml-name-*.[so|dll] in the search paths
    const fs::path name_path = fs::u8path(name);
    const fs::path file_prefix = backend_filename_prefix().native() + name_path.native() + fs::u8path("-").native();
    const fs::path file_extension = backend_filename_extension();

    std::vector<fs::path> search_paths;
    if (user_search_path == nullptr) {
#ifdef GGML_BACKEND_DIR
        search_paths.push_back(fs::u8path(GGML_BACKEND_DIR));
#endif
        // default search paths: executable directory, current directory
        search_paths.push_back(get_executable_path());
        search_paths.push_back(fs::current_path());
    } else {
        search_paths.push_back(fs::u8path(user_search_path));
    }

    int best_score = 0;
    fs::path best_path;
    std::error_code ec;

    for (const auto & search_path : search_paths) {
        if (!fs::exists(search_path, ec)) {
            if (ec) {
                GGML_LOG_DEBUG("%s: posix_stat(%s) failure, error-message: %s\n", __func__, path_str(search_path).c_str(), ec.message().c_str());
            } else {
                GGML_LOG_DEBUG("%s: search path %s does not exist\n", __func__, path_str(search_path).c_str());
            }
            continue;
        }
        fs::directory_iterator dir_it(search_path, fs::directory_options::skip_permission_denied);
        for (const auto & entry : dir_it) {
            if (entry.is_regular_file(ec)) {
                auto filename = entry.path().filename();
                auto ext = entry.path().extension();
                if (filename.native().find(file_prefix) == 0 && ext == file_extension) {
                    dl_handle_ptr handle{ dl_load_library(entry) };
                    if (!handle && !silent) {
                        GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(entry.path()).c_str(), dl_error());
                    }
                    if (handle) {
                        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
                        if (score_fn) {
                            int s = score_fn();
#ifndef NDEBUG
                            GGML_LOG_DEBUG("%s: %s score: %d\n", __func__, path_str(entry.path()).c_str(), s);
#endif
                            if (s > best_score) {
                                best_score = s;
                                best_path = entry.path();
                            }
                        } else {
                            if (!silent) {
                                GGML_LOG_INFO("%s: failed to find ggml_backend_score in %s\n", __func__, path_str(entry.path()).c_str());
                            }
                        }
                    }
                }
            }
        }
    }

    if (best_score == 0) {
        // try to load the base backend
        for (const auto & search_path : search_paths) {
            fs::path filename = backend_filename_prefix().native() + name_path.native() + backend_filename_extension().native();
            fs::path path = search_path / filename;
            if (std::error_code ec; fs::exists(path, ec)) {
                return get_reg().load_backend(path, silent);
            } else {
                if (ec) {
                    GGML_LOG_DEBUG("%s: posix_stat(%s) failure, error-message: %s\n", __func__, path_str(path).c_str(), ec.message().c_str());
                }
            }
        }
        return nullptr;
    }

    return get_reg().load_backend(best_path, silent);
}

void ggml_backend_load_all() {
    ggml_backend_load_all_from_path(nullptr);
}

void ggml_backend_load_all_from_path(const char * dir_path) {
#ifdef NDEBUG
    bool silent = true;
#else
    bool silent = false;
#endif

    const bool disable_device_backends = ggml_backend_device_backends_disabled();

    ggml_backend_load_best("blas", silent, dir_path);
    ggml_backend_load_best("zendnn", silent, dir_path);
    if (!disable_device_backends) {
        ggml_backend_load_best("cann", silent, dir_path);
        ggml_backend_load_best("cuda", silent, dir_path);
        ggml_backend_load_best("hip", silent, dir_path);
        ggml_backend_load_best("metal", silent, dir_path);
        ggml_backend_load_best("rpc", silent, dir_path);
        ggml_backend_load_best("sycl", silent, dir_path);
        ggml_backend_load_best("vulkan", silent, dir_path);
        ggml_backend_load_best("virtgpu", silent, dir_path);
        ggml_backend_load_best("opencl", silent, dir_path);
        ggml_backend_load_best("hexagon", silent, dir_path);
        ggml_backend_load_best("musa", silent, dir_path);
        ggml_backend_load_best("openvino", silent, dir_path);
    }
    ggml_backend_load_best("cpu", silent, dir_path);
    // check the environment variable GGML_BACKEND_PATH to load an out-of-tree backend
    const char * backend_path = std::getenv("GGML_BACKEND_PATH");
    if (backend_path) {
        ggml_backend_load(backend_path);
    }
}
