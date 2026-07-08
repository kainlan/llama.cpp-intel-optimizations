#include <vector>

namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

static const void * ggml_sycl_set_rows_stage_ptr(std::vector<ggml_sycl::alloc_handle> & staged_allocs) {
    ggml_sycl::alloc_handle alloc{};
    staged_allocs.push_back(alloc);
    return nullptr;
}

static void ggml_sycl_set_rows_free_staged(std::vector<ggml_sycl::alloc_handle> & staged_allocs) {
    staged_allocs.clear();
}

struct ggml_sycl_set_rows_staged_alloc_guard {
    std::vector<ggml_sycl::alloc_handle> handles;
};
