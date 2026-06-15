namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

constexpr int GGML_SYCL_MAX_DEVICES = 4;

struct bad_tensor_extra_gpu {
    ggml_sycl::alloc_handle data_alloc[GGML_SYCL_MAX_DEVICES];

    void reset(int device) {
        ggml_sycl::alloc_handle data_alloc{};
        this->data_alloc[device] = data_alloc;
    }
};
