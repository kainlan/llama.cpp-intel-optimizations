#pragma once

// A std::mutex whose holders a test build can see (docs/design/sycl-storage-tier-lazy-tables.md §3.3, §11 T5;
// canonical contract §12.5). The IO pool's reap, the yield entry and host_ledger_sample() must run with none of the
// five witnessed locks held (the IO pool registry's leaf mutex, g_tensor_inventory_mutex, g_lifecycle_plan_mutex, the
// IO engine's queue lock and each table's strand lock), and a std::mutex cannot say whether the calling thread holds
// it: try_lock from the owner thread is undefined behaviour.
//
// A release build's witnessed_mutex is std::mutex itself and records no holder: witnessed_mutex_depth() reads 0 and
// witnessed_mutex_held_any() reads none whatever the thread holds. A GGML_SYCL_PRIVATE_TESTING build's wraps a
// std::mutex and records each instance the calling thread holds, with its name, in an eight-entry thread_local table,
// so a lock taken through a guard, a unique_lock relock or a condition_variable_any wait is recorded and erased with
// the mutex itself. A ninth lock held at once aborts naming it: eight is above the deepest nesting of the five, and a
// lock the witness cannot record is one the reap would miss.
//
// Both settings meet in one binary: test-mem-handle-wrong-device, test-mem-handle-byte-contract and
// test-sycl-runtime-alloc compile unified-cache.cpp with GGML_SYCL_PRIVATE_TESTING=1 and link the backend library,
// built without it. No public header and no libllama translation unit includes this one, and three invariants keep such
// a binary sound. Each lock instance has internal linkage, a file-scope static or one in an unnamed namespace, or is a
// member of a type that one translation unit defines, so no instance is named across the two settings. The two forms'
// functions live in distinct inline namespaces, so the binary holds two definitions under two names, never two
// definitions of one name. No type that embeds a witnessed_mutex is defined in a header, so none is compiled under both
// settings with two layouts. Guards spell the lock type through class template argument deduction (std::lock_guard
// lock(m)), so a guard reads the same under both settings.

#include <mutex>

#if defined(GGML_SYCL_PRIVATE_TESTING)
#    include <cstdio>
#    include <cstdlib>
#endif

namespace ggml_sycl {

#if !defined(GGML_SYCL_PRIVATE_TESTING)

// An instance's name, which a declaration passes as witnessed_mutex m{ GGML_SYCL_WITNESS_NAME("m") };. This form
// drops it, so the declaration value-initializes a std::mutex.
#    define GGML_SYCL_WITNESS_NAME(name)

inline namespace witness_release {

using witnessed_mutex = std::mutex;

static_assert(sizeof(witnessed_mutex) == sizeof(std::mutex), "a release build's witnessed_mutex is std::mutex");

// The calling thread's depth on `m`: 1 while it holds it, else 0. This form records nothing and reads 0.
inline int witnessed_mutex_depth(const witnessed_mutex & m) noexcept {
    (void) m;
    return 0;
}

// The name of a witnessed lock the calling thread holds, or nullptr when it holds none. This form reads none.
inline const char * witnessed_mutex_held_any() noexcept {
    return nullptr;
}

// Whether the calling thread holds `m`.
inline bool witnessed_mutex_held(const witnessed_mutex & m) noexcept {
    return witnessed_mutex_depth(m) > 0;
}

}  // namespace witness_release

#else

// An instance's name, which a declaration passes as witnessed_mutex m{ GGML_SYCL_WITNESS_NAME("m") };. This form
// keeps it: the witness reports it, and the reap's refusal names it.
#    define GGML_SYCL_WITNESS_NAME(name) name

inline namespace witness_private_testing {

class witnessed_mutex;

// One held lock of the calling thread: the instance and its name. An empty entry has a null instance.
struct witnessed_mutex_entry {
    const witnessed_mutex * m;
    const char *            name;
};

constexpr int k_witnessed_mutex_table_size = 8;

// The calling thread's held witnessed locks; an unlock empties its entry and the next lock fills the first empty one.
inline witnessed_mutex_entry * witnessed_mutex_table() noexcept {
    thread_local witnessed_mutex_entry table[k_witnessed_mutex_table_size] = {};
    return table;
}

class witnessed_mutex {
  public:
    // constexpr, as std::mutex's constructor is, so a namespace-scope lock is constant-initialized and is never
    // taken before its constructor runs.
    constexpr explicit witnessed_mutex(const char * name) noexcept : name_(name != nullptr ? name : "(unnamed)") {}

    witnessed_mutex(const witnessed_mutex &)             = delete;
    witnessed_mutex & operator=(const witnessed_mutex &) = delete;

    void lock() {
        mutex_.lock();
        record();
    }

    bool try_lock() {
        if (!mutex_.try_lock()) {
            return false;
        }
        record();
        return true;
    }

    // The entry goes first: once the mutex is free another thread may take it, and this thread no longer holds it.
    void unlock() {
        erase();
        mutex_.unlock();
    }

    const char * name() const noexcept { return name_; }

  private:
    void record() noexcept {
        witnessed_mutex_entry * table = witnessed_mutex_table();
        for (int i = 0; i < k_witnessed_mutex_table_size; ++i) {
            if (table[i].m == nullptr) {
                table[i] = { this, name_ };
                return;
            }
        }
        std::fprintf(stderr, "witnessed_mutex: the depth table is full taking %s\n", name_);
        std::fflush(stderr);
        std::abort();
    }

    void erase() noexcept {
        witnessed_mutex_entry * table = witnessed_mutex_table();
        for (int i = 0; i < k_witnessed_mutex_table_size; ++i) {
            if (table[i].m == this) {
                table[i] = { nullptr, nullptr };
                return;
            }
        }
    }

    std::mutex   mutex_;
    const char * name_;
};

// The calling thread's depth on `m`: 1 while it holds it, else 0.
inline int witnessed_mutex_depth(const witnessed_mutex & m) noexcept {
    const witnessed_mutex_entry * table = witnessed_mutex_table();
    for (int i = 0; i < k_witnessed_mutex_table_size; ++i) {
        if (table[i].m == &m) {
            return 1;
        }
    }
    return 0;
}

// The name of the witnessed lock in the calling thread's first occupied entry, or nullptr when it holds none.
inline const char * witnessed_mutex_held_any() noexcept {
    const witnessed_mutex_entry * table = witnessed_mutex_table();
    for (int i = 0; i < k_witnessed_mutex_table_size; ++i) {
        if (table[i].m != nullptr) {
            return table[i].name;
        }
    }
    return nullptr;
}

// Whether the calling thread holds `m`.
inline bool witnessed_mutex_held(const witnessed_mutex & m) noexcept {
    return witnessed_mutex_depth(m) > 0;
}

}  // namespace witness_private_testing

#endif

}  // namespace ggml_sycl
