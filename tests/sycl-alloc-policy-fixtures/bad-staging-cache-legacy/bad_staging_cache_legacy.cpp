static constexpr int GGML_SYCL_MAX_DEVICES = 4;

namespace ggml_sycl {
struct alloc_handle {
    void * ptr;
};
}  // namespace ggml_sycl

struct StagedBuffer {
    void *                  ptrs[GGML_SYCL_MAX_DEVICES];
    ggml_sycl::alloc_handle handles[GGML_SYCL_MAX_DEVICES];
};

struct RuntimeStagedData {
    void *                  ptr;
    ggml_sycl::alloc_handle alloc;
};

void bad_runtime(RuntimeStagedData & data) {
    if (data.alloc.ptr) {
        data.alloc = {};
    }
}

void bad_tp(StagedBuffer * entry) {
    entry->handles[0] = {};
}
