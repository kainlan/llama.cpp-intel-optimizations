namespace ggml_sycl {
struct alloc_handle {};
}

class UnifiedKernel {
    ggml_sycl::alloc_handle dag_ready_counter_alloc_;
    ggml_sycl::alloc_handle dag_tile_claimed_alloc_;
    ggml_sycl::alloc_handle dag_tiles_done_alloc_;
    ggml_sycl::alloc_handle dag_completed_alloc_;
    ggml_sycl::alloc_handle dag_successor_off_alloc_;
    ggml_sycl::alloc_handle dag_successor_list_alloc_;
    ggml_sycl::alloc_handle dag_n_tiles_alloc_;
    ggml_sycl::alloc_handle dag_initial_ready_counter_alloc_;
};
