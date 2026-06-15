namespace ggml_sycl {
struct scoped_unified_alloc {
    explicit scoped_unified_alloc(int) {}

    int as_mem_handle() { return 0; }
};
}  // namespace ggml_sycl

void bad_moe_readback_stage_scoped(int gate_req, int early_req, int weights_req) {
    ggml_sycl::scoped_unified_alloc gate_stage(gate_req);
    (void) gate_stage.as_mem_handle();

    ggml_sycl::scoped_unified_alloc early_stage(early_req);
    (void) early_stage.as_mem_handle();

    ggml_sycl::scoped_unified_alloc weights_stage(weights_req);
    (void) weights_stage.as_mem_handle();
}
