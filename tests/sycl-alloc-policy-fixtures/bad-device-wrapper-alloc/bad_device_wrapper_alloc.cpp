struct queue;

void * ggml_sycl_malloc_device(unsigned long bytes, const queue & q, const char * tag);
void * ggml_sycl_malloc_device_tracked_bytes(unsigned long bytes, queue & q, const char * tag);
void * ggml_sycl_malloc_host(unsigned long bytes, const queue & q);
void * ggml_sycl_malloc_host_tracked_bytes(unsigned long bytes, queue & q, const char * tag);
void * ggml_sycl_malloc_shared(unsigned long bytes, const queue & q);
void * ggml_sycl_host_malloc(unsigned long bytes);
void   ggml_sycl_host_free(void * ptr);

template <typename T> T * ggml_sycl_malloc_host_t(unsigned long count, const queue & q);

template <typename T> T * ggml_sycl_malloc_shared_t(unsigned long count, const queue & q);

#define GGML_SYCL_MALLOC_HOST_T(T, count, q)   ggml_sycl_malloc_host_t<T>(count, q)
#define GGML_SYCL_MALLOC_SHARED_T(T, count, q) ggml_sycl_malloc_shared_t<T>(count, q)

void * bad_device_wrapper_alloc_fixture(const queue & q) {
    return ggml_sycl_malloc_device(4096, q, "bad-device-wrapper");
}

void * bad_device_tracked_wrapper_alloc_fixture(queue & q) {
    return ggml_sycl_malloc_device_tracked_bytes(4096, q, "bad-device-tracked-wrapper");
}

void * bad_host_wrapper_alloc_fixture(const queue & q) {
    return ggml_sycl_malloc_host(4096, q);
}

void * bad_host_tracked_wrapper_alloc_fixture(queue & q) {
    return ggml_sycl_malloc_host_tracked_bytes(4096, q, "bad-host-tracked-wrapper");
}

void * bad_shared_wrapper_alloc_fixture(const queue & q) {
    return ggml_sycl_malloc_shared(4096, q);
}

float * bad_host_template_wrapper_alloc_fixture(const queue & q) {
    return GGML_SYCL_MALLOC_HOST_T(float, 1024, q);
}

float * bad_shared_template_wrapper_alloc_fixture(const queue & q) {
    return GGML_SYCL_MALLOC_SHARED_T(float, 1024, q);
}

void * bad_host_boundary_alloc_fixture() {
    return ggml_sycl_host_malloc(4096);
}

void bad_host_boundary_free_fixture(void * ptr) {
    ggml_sycl_host_free(ptr);
}
