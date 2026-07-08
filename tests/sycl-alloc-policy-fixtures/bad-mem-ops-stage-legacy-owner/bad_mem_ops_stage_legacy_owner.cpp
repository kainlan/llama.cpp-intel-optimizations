#include <utility>

namespace ggml_sycl {
struct alloc_handle {};

struct mem_handle {
    static mem_handle from_owned_alloc(alloc_handle &&, int) { return {}; }
};
}  // namespace ggml_sycl

bool alloc_pinned_stage_handle_legacy() {
    ggml_sycl::alloc_handle stage_alloc{};
    auto                    owner = ggml_sycl::mem_handle::from_owned_alloc(std::move(stage_alloc), 0);
    (void) owner;
    return true;
}
