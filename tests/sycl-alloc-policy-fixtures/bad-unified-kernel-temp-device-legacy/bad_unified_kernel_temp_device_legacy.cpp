#include <vector>

namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

struct PersistentPlan {
    std::vector<ggml_sycl::alloc_handle> temp_device_allocs;
};

class UnifiedKernel {
    std::vector<ggml_sycl::alloc_handle> cached_temp_device_allocs_;

    void add_temp_device_alloc_handle(ggml_sycl::alloc_handle handle);
};
