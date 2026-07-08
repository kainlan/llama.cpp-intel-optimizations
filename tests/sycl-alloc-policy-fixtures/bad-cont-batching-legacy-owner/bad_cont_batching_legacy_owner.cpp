#include <utility>

namespace ggml_sycl {
struct alloc_handle {};

struct mem_handle {
    static mem_handle from_owned_alloc(alloc_handle, int) { return {}; }
};
}  // namespace ggml_sycl

void bad_cont_batching_legacy_owner() {
    ggml_sycl::alloc_handle batch_owner{};
    auto                    owner = ggml_sycl::mem_handle::from_owned_alloc(std::move(batch_owner), 0);
    (void) owner;
}
