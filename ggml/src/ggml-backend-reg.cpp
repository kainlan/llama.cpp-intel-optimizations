#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-backend-dl.h"
#include "ggml-impl.h"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <chrono>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <thread>
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

// Deterministic host-test barrier: pauses one checked unload after publishing
// its UNLOADING tombstone. Readers and recursive registration remain unlocked
// and must observe/reject that identity without blocking.
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

enum class ggml_backend_reg_state : uint8_t {
    ACTIVE,
    REACTIVATING,
    UNLOADING,
    HIDDEN_FAILED,
    REMOVED,
};

struct ggml_backend_reg_entry {
    ggml_backend_reg_t       reg;
    dl_handle_ptr            handle;
    ggml_backend_reg_state   state = ggml_backend_reg_state::ACTIVE;
    bool                     dynamic = false;
    size_t                   active_calls = 0;
    size_t                   durable_owners = 0;
    uint64_t                 current_generation = 0;
    uint64_t                 generation_counter = 0;
    std::string              cached_name;
    std::string              module_path;
};

using ggml_backend_reg_entry_ptr = std::shared_ptr<ggml_backend_reg_entry>;

struct ggml_backend_device_entry {
    ggml_backend_dev_t         dev;
    ggml_backend_reg_entry_ptr owner;
    uint64_t                   generation;
};

// count()/get() use one bounded per-thread snapshot. It is enumeration state,
// never a DSO lifetime mechanism: every dynamic module is process-pinned before
// score/init/publication because raw C handles have no transferable release API.
static thread_local std::vector<ggml_backend_reg_entry_ptr> g_backend_reg_enumeration;
static thread_local std::vector<ggml_backend_device_entry>  g_backend_dev_enumeration;

struct ggml_backend_registry {
    mutable std::mutex                       mutex;
    mutable std::condition_variable          cv;
    std::recursive_mutex                     module_operation_mutex;
    std::vector<ggml_backend_reg_entry_ptr>  backends;
    std::vector<ggml_backend_device_entry>   devices;

    ggml_backend_registry() = default;

    // Run only after the registry object is fully constructed and published by
    // get_reg(). Backend constructors may enter complex runtime code; invoking
    // them from this object's constructor exposes a partially-initialized
    // registry to any indirect generic registration path.
    void register_builtin_backends() {
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

    ~ggml_backend_registry() = default;

    bool register_backend(ggml_backend_reg_t reg, dl_handle_ptr handle = nullptr,
                          std::string module_path = {}) noexcept {
        if (!reg) {
            return false;
        }
        try {
            bool needs_reactivation = false;
            {
                std::lock_guard<std::mutex> lock(mutex);
                const auto known = std::find_if(backends.begin(), backends.end(),
                    [reg](const ggml_backend_reg_entry_ptr & entry) { return entry->reg == reg; });
                if (known != backends.end() && (*known)->state != ggml_backend_reg_state::REMOVED) {
                    return false;
                }
                needs_reactivation = known != backends.end();
            }
            using prepare_reactivate_fn = bool (*)();
            using settle_reactivate_fn = void (*)();
            prepare_reactivate_fn prepare_reactivate = nullptr;
            settle_reactivate_fn commit_reactivate = nullptr;
            settle_reactivate_fn rollback_reactivate = nullptr;
            bool reactivation_prepared = false;
            if (needs_reactivation && reg->iface.get_proc_address) {
                prepare_reactivate = reinterpret_cast<prepare_reactivate_fn>(
                    reg->iface.get_proc_address(reg, "ggml_backend_prepare_reactivate"));
                commit_reactivate = reinterpret_cast<settle_reactivate_fn>(
                    reg->iface.get_proc_address(reg, "ggml_backend_commit_reactivate"));
                rollback_reactivate = reinterpret_cast<settle_reactivate_fn>(
                    reg->iface.get_proc_address(reg, "ggml_backend_rollback_reactivate"));
                const bool any_hook = prepare_reactivate || commit_reactivate || rollback_reactivate;
                if (any_hook && (!prepare_reactivate || !commit_reactivate || !rollback_reactivate ||
                                 !prepare_reactivate())) {
                    return false;
                }
                reactivation_prepared = any_hook;
            }
            struct reactivation_rollback_guard {
                settle_reactivate_fn rollback;
                bool * prepared;
                ~reactivation_rollback_guard() { if (*prepared && rollback) rollback(); }
            } rollback_guard{ rollback_reactivate, &reactivation_prepared };
            // Stage all plugin-owned metadata while mutation admission remains closed.
            const char * name = ggml_backend_reg_name_unchecked(reg);
            size_t count = 0;
            if (!ggml_backend_reg_dev_count_unchecked(reg, &count)) {
                return false;
            }
            std::vector<ggml_backend_dev_t> staged_devices;
            staged_devices.reserve(count);
            for (size_t i = 0; i < count; ++i) {
                ggml_backend_dev_t device = nullptr;
                if (!ggml_backend_reg_dev_get_unchecked(reg, i, &device)) {
                    return false;
                }
                staged_devices.push_back(device);
            }
            const bool dynamic = handle != nullptr;
            auto candidate = std::make_shared<ggml_backend_reg_entry>(
                ggml_backend_reg_entry{ reg, std::move(handle), ggml_backend_reg_state::ACTIVE, dynamic, 0, 0, 0, 0,
                                        name ? name : "", std::move(module_path) });

            ggml_backend_reg_entry_ptr published_entry;
            uint64_t staged_generation = 0;
            {
                std::lock_guard<std::mutex> lock(mutex);
                auto existing = std::find_if(backends.begin(), backends.end(),
                    [reg](const ggml_backend_reg_entry_ptr & entry) { return entry->reg == reg; });
                if (existing != backends.end() && (*existing)->state != ggml_backend_reg_state::REMOVED) {
                    return false;
                }
                const bool is_new = existing == backends.end();
                ggml_backend_reg_entry_ptr entry = is_new ? candidate : *existing;
                backends.reserve(backends.size() + (is_new ? 1 : 0));
                devices.reserve(devices.size() + staged_devices.size());
                staged_generation = ++entry->generation_counter;
                if (is_new) {
                    backends.push_back(entry);
                } else {
                    // Retain every prior generation row forever: saved raw
                    // device identities must remain recognizable tombstones.
                    entry->handle = std::move(candidate->handle);
                    entry->dynamic = candidate->dynamic;
                    entry->module_path = std::move(candidate->module_path);
                    entry->state = reactivation_prepared ? ggml_backend_reg_state::REACTIVATING :
                                                           ggml_backend_reg_state::ACTIVE;
                }
                for (auto dev : staged_devices) devices.push_back({ dev, entry, staged_generation });
                if (!reactivation_prepared) entry->current_generation = staged_generation;
                published_entry = entry;
            }
            if (reactivation_prepared) {
                try {
                    commit_reactivate();
                } catch (...) {
                    std::lock_guard<std::mutex> lock(mutex);
                    // Keep staged device identities linked to their REMOVED
                    // owner tombstone so saved handles reject rather than
                    // falling through as standalone/unmanaged devices.
                    published_entry->state = ggml_backend_reg_state::REMOVED;
                    return false;
                }
                std::lock_guard<std::mutex> lock(mutex);
                published_entry->current_generation = staged_generation;
                published_entry->state = ggml_backend_reg_state::ACTIVE;
                reactivation_prepared = false;
            }
            ggml_backend_refresh_buffer_lifecycle();
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n", __func__, name, count);
#endif
            return true;
        } catch (...) {
            return false;
        }
    }

    void register_device(ggml_backend_dev_t device) noexcept {
        if (!device) {
            return;
        }
        try {
            const auto reg = device->reg;
            std::lock_guard<std::mutex> lock(mutex);
            const auto owner = std::find_if(backends.begin(), backends.end(),
                [reg](const ggml_backend_reg_entry_ptr & entry) {
                    return entry->reg == reg && entry->state == ggml_backend_reg_state::ACTIVE;
                });
            if (owner == backends.end() || std::any_of(devices.begin(), devices.end(),
                    [device](const ggml_backend_device_entry & entry) { return entry.dev == device; })) {
                return;
            }
            devices.push_back({ device, *owner, (*owner)->current_generation });
        } catch (...) {
        }
    }

    ggml_backend_reg_t load_backend(const fs::path & path, bool silent) noexcept {
        std::lock_guard<std::recursive_mutex> operation_lock(module_operation_mutex);
        std::error_code canonical_error;
        const fs::path canonical_path = fs::weakly_canonical(path, canonical_error);
        const std::string module_path = path_str(canonical_error ? path : canonical_path);
        {
            std::lock_guard<std::mutex> lock(mutex);
            const auto blocked = std::find_if(backends.begin(), backends.end(), [&](const auto & entry) {
                return entry->dynamic && entry->module_path == module_path &&
                       entry->state != ggml_backend_reg_state::REMOVED;
            });
            if (blocked != backends.end()) {
                return nullptr;
            }
        }
        dl_handle_ptr handle;
        try {
            handle.reset(dl_load_library(path));
        } catch (...) {
            if (!silent) {
                GGML_LOG_ERROR("%s: backend loader callback threw for %s\n", __func__, path_str(path).c_str());
            }
            return nullptr;
        }
        if (!handle) {
            if (!silent) {
                GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(path).c_str(), dl_error());
            }
            return nullptr;
        }
        // Generic raw-handle policy: pin before score/init can publish any DSO
        // pointer. Logical unload remains supported, physical unload does not.
        if (!dl_pin_library(handle.get(), path)) {
            if (!silent) {
                GGML_LOG_ERROR("%s: failed to process-pin backend %s\n", __func__, path_str(path).c_str());
            }
            return nullptr;
        }
        try {
            auto score_fn = reinterpret_cast<ggml_backend_score_t>(dl_get_sym(handle.get(), "ggml_backend_score"));
            if (score_fn && score_fn() == 0) {
                return nullptr;
            }
            auto backend_init_fn = reinterpret_cast<ggml_backend_init_t>(dl_get_sym(handle.get(), "ggml_backend_init"));
            if (!backend_init_fn) {
                return nullptr;
            }
            ggml_backend_reg_t reg = backend_init_fn();
            if (!reg || reg->api_version != GGML_BACKEND_API_VERSION) {
                return nullptr;
            }
            if (!register_backend(reg, std::move(handle), module_path)) {
                return nullptr;
            }
            GGML_LOG_INFO("%s: loaded %s backend from %s\n", __func__, ggml_backend_reg_name(reg), path_str(path).c_str());
            return reg;
        } catch (...) {
            if (!silent) {
                GGML_LOG_ERROR("%s: backend resolver/init threw for %s\n", __func__, path_str(path).c_str());
            }
            return nullptr;
        }
    }

    ggml_backend_unload_result unload_backend(ggml_backend_reg_t reg, bool silent) noexcept {
        std::lock_guard<std::recursive_mutex> operation_lock(module_operation_mutex);
        ggml_backend_reg_entry_ptr unloading;
        bool was_hidden_failed = false;
        {
            std::unique_lock<std::mutex> lock(mutex);
            const auto it = std::find_if(backends.begin(), backends.end(),
                [reg](const ggml_backend_reg_entry_ptr & entry) {
                    return entry->reg == reg && (entry->state == ggml_backend_reg_state::ACTIVE ||
                                                 entry->state == ggml_backend_reg_state::HIDDEN_FAILED);
                });
            if (it == backends.end()) {
                return GGML_BACKEND_UNLOAD_NOT_FOUND;
            }
            unloading = *it;
            was_hidden_failed = unloading->state == ggml_backend_reg_state::HIDDEN_FAILED;
            if (unloading->durable_owners != 0) {
                return GGML_BACKEND_UNLOAD_BUSY;
            }
            // Tombstone first: enumeration, lookup, init and same-identity
            // registration all reject this entry while callbacks run unlocked.
            unloading->state = ggml_backend_reg_state::UNLOADING;
        }

        ggml_backend_test_unload_barrier();
        bool test_reentrant_registration = false;
        {
            std::lock_guard<std::mutex> test_lock(g_registry_test_mutex);
            test_reentrant_registration = g_registry_test_reentrant_mutation;
            g_registry_test_reentrant_mutation = false;
        }
        if (test_reentrant_registration) {
            // Exercise the same public registration path a recursive plugin
            // callback would use; the UNLOADING tombstone must reject it.
            (void) register_backend(reg);
        }
        using can_unload_fn = bool (*)();
        using shutdown_fn = void (*)();
        can_unload_fn can_unload = nullptr;
        shutdown_fn shutdown = nullptr;
        shutdown_fn complete = nullptr;
        shutdown_fn cancel = nullptr;
        bool resolver_failed = false;
        try {
            // Resolver is plugin code too; the process pin guards every call.
            const auto resolve = [&](const char * name) {
                return reg->iface.get_proc_address ? reg->iface.get_proc_address(reg, name) : nullptr;
            };
            can_unload = reinterpret_cast<can_unload_fn>(resolve("ggml_backend_can_unload"));
            shutdown = reinterpret_cast<shutdown_fn>(resolve("ggml_backend_shutdown"));
            complete = reinterpret_cast<shutdown_fn>(resolve("ggml_backend_complete_unload"));
            cancel = reinterpret_cast<shutdown_fn>(resolve("ggml_backend_cancel_unload"));
        } catch (...) {
            resolver_failed = true;
        }
        bool reserved = false;
        const auto cancel_noexcept = [&] {
            if (reserved && cancel) {
                try {
                    cancel();
                } catch (...) {
                }
            }
            reserved = false;
        };
        dl_handle_ptr released_handle;
        const auto settle_state = [&](ggml_backend_reg_state state) {
            std::lock_guard<std::mutex> lock(mutex);
            if (unloading->state == ggml_backend_reg_state::UNLOADING) {
                if (state == ggml_backend_reg_state::REMOVED) {
                    released_handle = std::move(unloading->handle);
                }
                unloading->state = state;
            }
        };
        if (resolver_failed) {
            settle_state(ggml_backend_reg_state::HIDDEN_FAILED);
            return GGML_BACKEND_UNLOAD_BUSY;
        }
        // Never acquire a reservation unless every settlement path is already
        // resolved. This avoids an un-cancellable reservation on malformed DSOs.
        if (can_unload && (!shutdown || !complete || !cancel)) {
            settle_state(was_hidden_failed ? ggml_backend_reg_state::HIDDEN_FAILED :
                                               ggml_backend_reg_state::ACTIVE);
            return GGML_BACKEND_UNLOAD_BUSY;
        }

        try {
            if (can_unload) {
                reserved = can_unload();
                if (!reserved) {
                    settle_state(was_hidden_failed ? ggml_backend_reg_state::HIDDEN_FAILED :
                                                       ggml_backend_reg_state::ACTIVE);
                    return GGML_BACKEND_UNLOAD_BUSY;
                }
            }
        } catch (...) {
            // The hook may throw after reservation; conservatively cancel.
            reserved = can_unload != nullptr;
            cancel_noexcept();
            settle_state(ggml_backend_reg_state::HIDDEN_FAILED);
            return GGML_BACKEND_UNLOAD_BUSY;
        }

        {
            std::unique_lock<std::mutex> lock(mutex);
            constexpr auto active_call_drain_timeout = std::chrono::seconds(5);
            if (!cv.wait_for(lock, active_call_drain_timeout, [&] { return unloading->active_calls == 0; })) {
                const size_t stuck_calls = unloading->active_calls;
                lock.unlock();
                GGML_LOG_ERROR("%s: active callback drain timed out with %zu call(s)\n", __func__, stuck_calls);
                cancel_noexcept();
                settle_state(was_hidden_failed ? ggml_backend_reg_state::HIDDEN_FAILED :
                                                 ggml_backend_reg_state::ACTIVE);
                return GGML_BACKEND_UNLOAD_BUSY;
            }
        }

        try {
            if (shutdown) {
                shutdown();
            }
        } catch (...) {
            cancel_noexcept();
            // Shutdown may have destroyed arbitrary module state. Never make
            // this identity discoverable again; retain a retryable tombstone.
            settle_state(ggml_backend_reg_state::HIDDEN_FAILED);
            return GGML_BACKEND_UNLOAD_BUSY;
        }
        try {
            if (complete) {
                complete();
                reserved = false;
            }
        } catch (...) {
            cancel_noexcept();
            settle_state(ggml_backend_reg_state::HIDDEN_FAILED);
            return GGML_BACKEND_UNLOAD_BUSY;
        }
        settle_state(ggml_backend_reg_state::REMOVED);
        // Drop only the ordinary dlopen reference outside the registry lock;
        // the generic process pin preserves all previously returned raw handles.
        released_handle.reset();
        if (!silent) {
            GGML_LOG_DEBUG("%s: logically unloaded backend\n", __func__);
        }
        return GGML_BACKEND_UNLOAD_OK;
    }

    bool begin_call(const ggml_backend_reg_entry_ptr & entry) const noexcept {
        try {
            std::lock_guard<std::mutex> lock(mutex);
            if (entry->state != ggml_backend_reg_state::ACTIVE || entry->active_calls == SIZE_MAX) {
                return false;
            }
            ++entry->active_calls;
            return true;
        } catch (...) {
            return false;
        }
    }

    void end_call(const ggml_backend_reg_entry_ptr & entry) const noexcept {
        try {
            std::lock_guard<std::mutex> lock(mutex);
            if (entry->active_calls != 0) {
                --entry->active_calls;
                cv.notify_all();
            }
        } catch (...) {
        }
    }

    size_t backend_count() const noexcept {
        try {
            std::lock_guard<std::mutex> lock(mutex);
            g_backend_reg_enumeration.clear();
            for (const auto & entry : backends) {
                if (entry->state == ggml_backend_reg_state::ACTIVE) {
                    g_backend_reg_enumeration.push_back(entry);
                }
            }
            return g_backend_reg_enumeration.size();
        } catch (...) {
            g_backend_reg_enumeration.clear();
            return 0;
        }
    }

    ggml_backend_reg_t backend_get(size_t index) const noexcept {
        try {
            std::lock_guard<std::mutex> lock(mutex);
            if (index >= g_backend_reg_enumeration.size() ||
                g_backend_reg_enumeration[index]->state != ggml_backend_reg_state::ACTIVE) {
                return nullptr;
            }
            return g_backend_reg_enumeration[index]->reg;
        } catch (...) {
            return nullptr;
        }
    }

    ggml_backend_reg_t backend_by_name(const char * name) const noexcept {
        try {
            std::vector<ggml_backend_reg_entry_ptr> live;
            {
                std::lock_guard<std::mutex> lock(mutex);
                for (const auto & entry : backends) {
                    if (entry->state == ggml_backend_reg_state::ACTIVE) {
                        live.push_back(entry);
                    }
                }
            }
            for (const auto & entry : live) {
                if (!begin_call(entry)) {
                    continue;
                }
                bool matches = false;
                try {
                    matches = striequals(ggml_backend_reg_name(entry->reg), name);
                } catch (...) {
                }
                end_call(entry);
                if (matches) {
                    std::lock_guard<std::mutex> lock(mutex);
                    return entry->state == ggml_backend_reg_state::ACTIVE ? entry->reg : nullptr;
                }
            }
        } catch (...) {
        }
        return nullptr;
    }

    size_t device_count() const noexcept {
        try {
            std::lock_guard<std::mutex> lock(mutex);
            g_backend_dev_enumeration.clear();
            for (const auto & entry : devices) {
                if (entry.owner->state == ggml_backend_reg_state::ACTIVE &&
                    entry.generation == entry.owner->current_generation) {
                    g_backend_dev_enumeration.push_back(entry);
                }
            }
            return g_backend_dev_enumeration.size();
        } catch (...) {
            g_backend_dev_enumeration.clear();
            return 0;
        }
    }

    ggml_backend_dev_t device_get(size_t index) const noexcept {
        try {
            std::lock_guard<std::mutex> lock(mutex);
            if (index >= g_backend_dev_enumeration.size() ||
                g_backend_dev_enumeration[index].owner->state != ggml_backend_reg_state::ACTIVE ||
                g_backend_dev_enumeration[index].generation !=
                    g_backend_dev_enumeration[index].owner->current_generation) {
                return nullptr;
            }
            return g_backend_dev_enumeration[index].dev;
        } catch (...) {
            return nullptr;
        }
    }

    ggml_backend_dev_t device_by_name(const char * name) const noexcept {
        try {
            std::vector<ggml_backend_device_entry> live;
            {
                std::lock_guard<std::mutex> lock(mutex);
                for (const auto & entry : devices) {
                    if (entry.owner->state == ggml_backend_reg_state::ACTIVE &&
                        entry.generation == entry.owner->current_generation) {
                        live.push_back(entry);
                    }
                }
            }
            for (const auto & entry : live) {
                if (!begin_call(entry.owner)) {
                    continue;
                }
                bool matches = false;
                try {
                    matches = striequals(ggml_backend_dev_name(entry.dev), name);
                } catch (...) {
                }
                end_call(entry.owner);
                if (matches) {
                    std::lock_guard<std::mutex> lock(mutex);
                    return entry.owner->state == ggml_backend_reg_state::ACTIVE &&
                           entry.generation == entry.owner->current_generation ? entry.dev : nullptr;
                }
            }
        } catch (...) {
        }
        return nullptr;
    }

    ggml_backend_dev_t device_by_type(enum ggml_backend_dev_type type) const noexcept {
        try {
            std::vector<ggml_backend_device_entry> live;
            {
                std::lock_guard<std::mutex> lock(mutex);
                for (const auto & entry : devices) {
                    if (entry.owner->state == ggml_backend_reg_state::ACTIVE &&
                        entry.generation == entry.owner->current_generation) {
                        live.push_back(entry);
                    }
                }
            }
            for (const auto & entry : live) {
                if (!begin_call(entry.owner)) {
                    continue;
                }
                bool matches = false;
                try {
                    matches = ggml_backend_dev_type(entry.dev) == type;
                } catch (...) {
                }
                end_call(entry.owner);
                if (matches) {
                    std::lock_guard<std::mutex> lock(mutex);
                    return entry.owner->state == ggml_backend_reg_state::ACTIVE &&
                           entry.generation == entry.owner->current_generation ? entry.dev : nullptr;
                }
            }
        } catch (...) {
        }
        return nullptr;
    }
};

const char * ggml_backend_registry_cached_name(ggml_backend_reg_t reg) noexcept;
bool ggml_backend_registry_begin_call(ggml_backend_reg_t reg) noexcept;
void ggml_backend_registry_end_call(ggml_backend_reg_t reg) noexcept;
bool ggml_backend_device_begin_call(ggml_backend_dev_t device) noexcept;
void ggml_backend_device_end_call(ggml_backend_dev_t device) noexcept;
bool ggml_backend_device_owner_acquire(ggml_backend_dev_t device) noexcept;
void ggml_backend_device_owner_release(ggml_backend_dev_t device) noexcept;

static ggml_backend_registry & get_reg() {
    static ggml_backend_registry reg;
    static const ggml_backend_registry_lifecycle_i lifecycle_iface = {
        ggml_backend_registry_cached_name,
        ggml_backend_registry_begin_call,
        ggml_backend_registry_end_call,
        ggml_backend_device_begin_call,
        ggml_backend_device_end_call,
        ggml_backend_device_owner_acquire,
        ggml_backend_device_owner_release,
    };
    static const bool installed = [] {
        ggml_backend_set_registry_lifecycle(&lifecycle_iface);
        return true;
    }();
    (void) installed;

    enum class builtin_init_state { UNINITIALIZED, INITIALIZING, COMPLETE };
    static std::mutex builtin_mutex;
    static std::condition_variable builtin_cv;
    static builtin_init_state builtin_state = builtin_init_state::UNINITIALIZED;
    static std::thread::id builtin_owner;
    {
        std::unique_lock<std::mutex> lock(builtin_mutex);
        if (builtin_state == builtin_init_state::INITIALIZING && builtin_owner == std::this_thread::get_id()) {
            // Constructor/callback reentry sees the fully constructed registry,
            // but the currently initializing backend is not published yet.
            return reg;
        }
        while (builtin_state == builtin_init_state::INITIALIZING) {
            builtin_cv.wait(lock);
        }
        if (builtin_state == builtin_init_state::COMPLETE) {
            return reg;
        }
        builtin_state = builtin_init_state::INITIALIZING;
        builtin_owner = std::this_thread::get_id();
    }
    try {
        reg.register_builtin_backends();
        // Built-in devices are now published; transactionally backfill buffers
        // that were created before lifecycle hooks/owners were available.
        ggml_backend_set_registry_lifecycle(&lifecycle_iface);
    } catch (...) {
        std::lock_guard<std::mutex> lock(builtin_mutex);
        builtin_state = builtin_init_state::UNINITIALIZED;
        builtin_owner = {};
        builtin_cv.notify_all();
        throw;
    }
    {
        std::lock_guard<std::mutex> lock(builtin_mutex);
        builtin_state = builtin_init_state::COMPLETE;
        builtin_owner = {};
        builtin_cv.notify_all();
    }
    return reg;
}

// Internal API
void ggml_backend_register(ggml_backend_reg_t reg) {
    get_reg().register_backend(reg);
}

void ggml_backend_device_register(ggml_backend_dev_t device) {
    get_reg().register_device(device);
}

const char * ggml_backend_registry_cached_name(ggml_backend_reg_t reg) noexcept {
    if (!reg) return nullptr;
    try {
        auto & registry = get_reg();
        std::lock_guard<std::mutex> lock(registry.mutex);
        const auto found = std::find_if(registry.backends.begin(), registry.backends.end(),
            [reg](const ggml_backend_reg_entry_ptr & entry) { return entry->reg == reg; });
        // Entry names are copied at staging and never mutated. Tombstones stay
        // retained in the registry, so this pointer remains stable after unlock.
        return found == registry.backends.end() ? nullptr : (*found)->cached_name.c_str();
    } catch (...) {
        return nullptr;
    }
}

bool ggml_backend_registry_begin_call(ggml_backend_reg_t reg) noexcept {
    if (!reg) {
        return false;
    }
    try {
        auto & registry = get_reg();
        std::lock_guard<std::mutex> lock(registry.mutex);
        const auto found = std::find_if(registry.backends.begin(), registry.backends.end(),
            [reg](const ggml_backend_reg_entry_ptr & entry) { return entry->reg == reg; });
        if (found == registry.backends.end()) {
            return true;
        }
        if ((*found)->state != ggml_backend_reg_state::ACTIVE || (*found)->active_calls == SIZE_MAX) {
            return false;
        }
        ++(*found)->active_calls;
        return true;
    } catch (...) {
        return false;
    }
}

void ggml_backend_registry_end_call(ggml_backend_reg_t reg) noexcept {
    try {
        auto & registry = get_reg();
        std::lock_guard<std::mutex> lock(registry.mutex);
        const auto found = std::find_if(registry.backends.begin(), registry.backends.end(),
            [reg](const ggml_backend_reg_entry_ptr & entry) { return entry->reg == reg; });
        if (found != registry.backends.end() && (*found)->active_calls != 0) {
            --(*found)->active_calls;
            registry.cv.notify_all();
        }
    } catch (...) {
    }
}

extern "C" size_t ggml_backend_test_active_calls(ggml_backend_reg_t reg) {
    try {
        auto & registry = get_reg();
        std::lock_guard<std::mutex> lock(registry.mutex);
        const auto found = std::find_if(registry.backends.begin(), registry.backends.end(),
            [reg](const ggml_backend_reg_entry_ptr & entry) { return entry->reg == reg; });
        return found == registry.backends.end() ? 0 : (*found)->active_calls;
    } catch (...) {
        return SIZE_MAX;
    }
}

bool ggml_backend_device_begin_call(ggml_backend_dev_t device) noexcept {
    if (!device) {
        return false;
    }
    try {
        auto & registry = get_reg();
        std::lock_guard<std::mutex> lock(registry.mutex);
        const auto found = std::find_if(registry.devices.begin(), registry.devices.end(),
            [device](const ggml_backend_device_entry & entry) { return entry.dev == device; });
        // Standalone/custom devices are outside the registry lifecycle. Known
        // registry devices acquire admission only while discoverable.
        if (found == registry.devices.end()) {
            return true;
        }
        if (found->owner->state != ggml_backend_reg_state::ACTIVE ||
            found->generation != found->owner->current_generation || found->owner->active_calls == SIZE_MAX) {
            return false;
        }
        ++found->owner->active_calls;
        return true;
    } catch (...) {
        return false;
    }
}

bool ggml_backend_device_owner_acquire(ggml_backend_dev_t device) noexcept {
    if (!device) return false;
    try {
        auto & registry = get_reg();
        std::lock_guard<std::mutex> lock(registry.mutex);
        const auto found = std::find_if(registry.devices.begin(), registry.devices.end(),
            [device](const ggml_backend_device_entry & entry) { return entry.dev == device; });
        if (found == registry.devices.end()) return device->reg == nullptr;
        if (found->owner->state != ggml_backend_reg_state::ACTIVE ||
            found->generation != found->owner->current_generation || found->owner->durable_owners == SIZE_MAX) {
            return false;
        }
        ++found->owner->durable_owners;
        return true;
    } catch (...) { return false; }
}

void ggml_backend_device_owner_release(ggml_backend_dev_t device) noexcept {
    try {
        auto & registry = get_reg();
        std::lock_guard<std::mutex> lock(registry.mutex);
        const auto found = std::find_if(registry.devices.begin(), registry.devices.end(),
            [device](const ggml_backend_device_entry & entry) { return entry.dev == device; });
        if (found != registry.devices.end() && found->owner->durable_owners != 0) {
            --found->owner->durable_owners;
            registry.cv.notify_all();
        }
    } catch (...) {}
}

void ggml_backend_device_end_call(ggml_backend_dev_t device) noexcept {
    try {
        auto & registry = get_reg();
        std::lock_guard<std::mutex> lock(registry.mutex);
        const auto found = std::find_if(registry.devices.begin(), registry.devices.end(),
            [device](const ggml_backend_device_entry & entry) { return entry.dev == device; });
        if (found != registry.devices.end() && found->owner->active_calls != 0) {
            --found->owner->active_calls;
            registry.cv.notify_all();
        }
    } catch (...) {
    }
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
                    if (handle && !dl_pin_library(handle.get(), entry.path())) {
                        if (!silent) {
                            GGML_LOG_ERROR("%s: failed to process-pin score candidate %s\n", __func__,
                                           path_str(entry.path()).c_str());
                        }
                        continue;
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
