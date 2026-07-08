namespace ggml_sycl {
struct scoped_unified_alloc {
    bool   allocate(int);
    void * get() const;
};
}  // namespace ggml_sycl

static void * bad_stage_raw_host_source(ggml_sycl::scoped_unified_alloc & stage) {
    int req = 0;
    if (!stage.allocate(req) || stage.get() == nullptr) {
        return nullptr;
    }
    return stage.get();
}

void bad_binbcast_raw_host_stage_scoped() {
    ggml_sycl::scoped_unified_alloc src0_stage;
    ggml_sycl::scoped_unified_alloc src1_stage;
    (void) bad_stage_raw_host_source(src0_stage);
    (void) bad_stage_raw_host_source(src1_stage);
}
