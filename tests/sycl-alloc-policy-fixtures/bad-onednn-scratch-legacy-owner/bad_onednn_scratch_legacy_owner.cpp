#include <memory>

namespace ggml_sycl {
struct alloc_handle {
    void * ptr;
};

bool unified_free(const alloc_handle &);
}  // namespace ggml_sycl

struct unified_cache {
    std::shared_ptr<ggml_sycl::alloc_handle> onednn_weights_scratch_owner_;
};

void bad_onednn_scratch_legacy_owner(unified_cache & cache) {
    if (cache.onednn_weights_scratch_owner_ && cache.onednn_weights_scratch_owner_->ptr) {
        (void) ggml_sycl::unified_free(*cache.onednn_weights_scratch_owner_);
    }
}
