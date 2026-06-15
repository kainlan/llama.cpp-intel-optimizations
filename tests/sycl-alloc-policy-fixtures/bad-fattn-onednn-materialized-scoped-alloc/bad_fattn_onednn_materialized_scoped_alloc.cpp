namespace ggml_sycl {
struct mem_handle {};

struct scoped_unified_alloc {
    mem_handle as_mem_handle() const;
};
}  // namespace ggml_sycl

struct bad_materialized_kv {
    ggml_sycl::scoped_unified_alloc Q_alloc;
    ggml_sycl::scoped_unified_alloc K_alloc;
    ggml_sycl::scoped_unified_alloc V_alloc;
    ggml_sycl::mem_handle           Q;
    ggml_sycl::mem_handle           K;
    ggml_sycl::mem_handle           V;
};

void bad_materialized_as_mem_handle(bad_materialized_kv & out) {
    out.Q = out.Q_alloc.as_mem_handle();
    out.K = out.K_alloc.as_mem_handle();
    out.V = out.V_alloc.as_mem_handle();
}
