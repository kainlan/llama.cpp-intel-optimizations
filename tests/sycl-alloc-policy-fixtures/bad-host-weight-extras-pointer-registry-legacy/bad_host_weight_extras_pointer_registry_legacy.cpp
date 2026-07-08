#include <unordered_map>

struct ggml_tensor;
struct ggml_tensor_extra_gpu;

void bad_host_weight_extras_pointer_registry_legacy(ggml_tensor * tensor, ggml_tensor_extra_gpu * extra) {
    static std::unordered_map<ggml_tensor *, ggml_tensor_extra_gpu *> g_sycl_host_weight_extras;
    (void) g_sycl_host_weight_extras.emplace(tensor, extra);
}
