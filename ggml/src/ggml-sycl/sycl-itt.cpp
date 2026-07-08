#include "sycl-itt.hpp"

#include <atomic>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <string_view>

namespace ggml_sycl {
namespace {

constexpr const char * SYCL_ITT_ENV = "GGML_SYCL_VTUNE_ITT";

std::atomic<uint64_t> g_begin_count{ 0 };
std::atomic<uint64_t> g_end_count{ 0 };
thread_local int      g_thread_task_depth = 0;

bool is_space(char ch) {
    return std::isspace(static_cast<unsigned char>(ch)) != 0;
}

std::string_view trim_ascii(const char * value) {
    if (value == nullptr) {
        return {};
    }

    const char * first = value;
    while (*first != '\0' && is_space(*first)) {
        ++first;
    }

    const char * last = first + std::strlen(first);
    while (last > first && is_space(*(last - 1))) {
        --last;
    }

    return { first, static_cast<size_t>(last - first) };
}

#if defined(__GNUC__) || defined(__clang__)
struct __itt_domain;
struct __itt_string_handle;

struct __itt_id {
    uint64_t d1;
    uint64_t d2;
    uint64_t d3;
};

extern "C" {
__attribute__((weak)) __itt_domain *        __itt_domain_create(const char * name);
__attribute__((weak)) __itt_string_handle * __itt_string_handle_create(const char * name);
__attribute__((weak)) void                  __itt_task_begin(const __itt_domain *  domain,
                                                             __itt_id              taskid,
                                                             __itt_id              parentid,
                                                             __itt_string_handle * name);
__attribute__((weak)) void                  __itt_task_end(const __itt_domain * domain);
}

bool itt_symbols_available() {
    return __itt_domain_create != nullptr && __itt_string_handle_create != nullptr && __itt_task_begin != nullptr &&
           __itt_task_end != nullptr;
}

__itt_domain * itt_domain() {
    static __itt_domain * domain = itt_symbols_available() ? __itt_domain_create("ggml-sycl.timeline") : nullptr;
    return domain;
}

__itt_string_handle * itt_string_handle(const char * category, const char * name) {
    const char * task_name = name != nullptr && name[0] != '\0' ? name : category;
    if (task_name == nullptr || task_name[0] == '\0') {
        task_name = "sycl.timeline";
    }
    return __itt_string_handle_create(task_name);
}
#else
bool itt_symbols_available() {
    return false;
}
#endif

void itt_runtime_task_begin(const char * category, const char * name) {
#if defined(__GNUC__) || defined(__clang__)
    if (!itt_symbols_available()) {
        return;
    }

    __itt_domain * domain = itt_domain();
    if (domain == nullptr) {
        return;
    }

    __itt_string_handle * handle = itt_string_handle(category, name);
    if (handle == nullptr) {
        return;
    }

    static constexpr __itt_id null_id = { 0, 0, 0 };
    __itt_task_begin(domain, null_id, null_id, handle);
#else
    (void) category;
    (void) name;
#endif
}

void itt_runtime_task_end() {
#if defined(__GNUC__) || defined(__clang__)
    if (!itt_symbols_available()) {
        return;
    }

    __itt_domain * domain = itt_domain();
    if (domain == nullptr) {
        return;
    }

    __itt_task_end(domain);
#endif
}

}  // namespace

bool sycl_itt_enabled_from_env(const char * value) {
    const std::string_view trimmed = trim_ascii(value);
    return trimmed == "1";
}

bool sycl_itt_enabled() {
    return sycl_itt_enabled_from_env(std::getenv(SYCL_ITT_ENV));
}

void sycl_itt_task_begin(const char * category, const char * name) {
    if (!sycl_itt_enabled()) {
        return;
    }

    ++g_thread_task_depth;
    g_begin_count.fetch_add(1, std::memory_order_relaxed);
    itt_runtime_task_begin(category, name);
}

void sycl_itt_task_end() {
    if (g_thread_task_depth <= 0) {
        return;
    }

    --g_thread_task_depth;
    g_end_count.fetch_add(1, std::memory_order_relaxed);
    itt_runtime_task_end();
}

void sycl_itt_reset_for_tests() {
    g_begin_count.store(0, std::memory_order_relaxed);
    g_end_count.store(0, std::memory_order_relaxed);
    g_thread_task_depth = 0;
}

uint64_t sycl_itt_begin_count_for_tests() {
    return g_begin_count.load(std::memory_order_relaxed);
}

uint64_t sycl_itt_end_count_for_tests() {
    return g_end_count.load(std::memory_order_relaxed);
}

}  // namespace ggml_sycl
