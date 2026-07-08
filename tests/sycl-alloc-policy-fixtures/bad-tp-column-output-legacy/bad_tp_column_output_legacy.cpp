#include <unordered_map>

struct ggml_tensor;

namespace ggml_sycl {
struct alloc_handle {
    void * ptr;
};
}  // namespace ggml_sycl

struct ggml_sycl_tp_column_parallel_output {
    void *                  ptr;
    size_t                  size;
    ggml_sycl::alloc_handle alloc;
};

std::unordered_map<const ggml_tensor *, ggml_sycl_tp_column_parallel_output> g_tp_column_parallel_outputs;
