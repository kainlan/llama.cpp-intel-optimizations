#include <string>
#include <unordered_map>

namespace ggml_sycl {
struct mem_handle {};
}

void bad_cpu_dispatch_host_ptr_split_registry_legacy(const std::string & key, const void * host_ptr) {
    static std::unordered_map<std::string, const void *>          g_host_ptr_map;
    static std::unordered_map<std::string, ggml_sycl::mem_handle> g_host_ptr_owned_handles;
    g_host_ptr_map[key] = host_ptr;
    (void) g_host_ptr_owned_handles;
}
