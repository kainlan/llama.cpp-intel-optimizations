#pragma once

// A std::mutex whose holders a test build can see (docs/design/sycl-storage-tier-lazy-tables.md §3.3, §11 T5;
// canonical contract §12.5). The IO pool's reap, the yield entry and host_ledger_sample() must run with none of the
// five witnessed locks held (the IO pool registry's leaf mutex, g_tensor_inventory_mutex, g_lifecycle_plan_mutex, the
// IO engine's queue lock and each table's strand lock), and a std::mutex cannot say whether the calling thread holds
// it: try_lock from the owner thread is undefined behaviour.
//
// This form is std::mutex itself in every build and records no holder: witnessed_mutex_depth() reads 0 and
// witnessed_mutex_held_any() reads none whatever the thread holds.
//
// Include it only from backend translation units compiled with the same GGML_SYCL_PRIVATE_TESTING setting as the
// library they link: no public header and no libllama translation unit includes it, and the lock objects are file-scope
// statics and members of backend-private types, so no type that embeds one is compiled under both settings. Guards
// spell the lock type through class template argument deduction (std::lock_guard lock(m)), so a guard reads the same
// under both settings.

#include <mutex>

namespace ggml_sycl {

// An instance's name, which a declaration passes as witnessed_mutex m{ GGML_SYCL_WITNESS_NAME("m") };. This form
// drops it, so the declaration value-initializes a std::mutex.
#define GGML_SYCL_WITNESS_NAME(name)

using witnessed_mutex = std::mutex;

static_assert(sizeof(witnessed_mutex) == sizeof(std::mutex), "a release build's witnessed_mutex is std::mutex");

// The calling thread's depth on `m`: 1 while it holds it, else 0. This form records nothing and reads 0.
inline int witnessed_mutex_depth(const witnessed_mutex & m) noexcept {
    (void) m;
    return 0;
}

// Whether the calling thread holds `m`.
inline bool witnessed_mutex_held(const witnessed_mutex & m) noexcept {
    return witnessed_mutex_depth(m) > 0;
}

// The name of a witnessed lock the calling thread holds, or nullptr when it holds none. This form reads none.
inline const char * witnessed_mutex_held_any() noexcept {
    return nullptr;
}

}  // namespace ggml_sycl
