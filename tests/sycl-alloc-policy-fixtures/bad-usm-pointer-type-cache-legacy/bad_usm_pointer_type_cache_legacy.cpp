#include <unordered_map>

void bad_usm_pointer_type_cache_legacy() {
    static std::unordered_map<void *, bool> cache;
    cache[nullptr] = true;
}
