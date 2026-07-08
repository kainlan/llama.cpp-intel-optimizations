struct bad_pp_pipeline_state {
    void * scratch_buf[2];
    void * scratch_alloc[2];
};

namespace ggml_sycl {
struct alloc_handle {};
}  // namespace ggml_sycl

void bad_pp_pipeline_legacy_fixture(bad_pp_pipeline_state * pipe, void * ptr) {
    pipe->scratch_alloc[0] = ptr;
    pipe->scratch_buf[0]   = ptr;

    ggml_sycl::alloc_handle scratch_alloc;
    (void) scratch_alloc;
}
