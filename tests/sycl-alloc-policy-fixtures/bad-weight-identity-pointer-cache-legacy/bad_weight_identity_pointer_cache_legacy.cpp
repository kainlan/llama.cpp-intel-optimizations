#include <unordered_map>

struct ggml_tensor;
struct ggml_sycl_weight_identity {};

void bad_weight_identity_pointer_cache_legacy(const ggml_tensor * tensor) {
    static std::unordered_map<const ggml_tensor *, ggml_sycl_weight_identity> g_sycl_weight_identities;
    (void) g_sycl_weight_identities.find(tensor);
}
