// Device test: FlashAttention's sequence-id device buffers belong to the backend context.
//
// The buffers used to be thread_local, so they ended with the thread that ran the op and, on a thread that
// never exits, outlived every model. They are members of the backend context now. This test allocates a
// context's four buffers from a worker thread and checks that
//   - they are still allocated after that thread has exited (a thread's end releases nothing), and
//   - freeing the backend releases all four, on a thread that stays alive.
// The file keeps its name from the thread_local era: its registration and the test inventory carry it.
//
// Usage:
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-fattn-thread-local

#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml.h"
#include "sycl-selector-fallback.hpp"
#include "test-skip.h"

#include <cstdio>
#include <cstdlib>
#include <thread>

#if !defined(GGML_USE_SYCL)
int main() {
    fprintf(stderr, "GGML_USE_SYCL not enabled; skipping test.\n");
    return LLAMA_TEST_EXIT_SKIP;
}
#else

extern "C" {
int  ggml_sycl_test_seq_id_buffer_allocs();
bool ggml_sycl_test_seq_id_buffers_touch(ggml_backend_t backend);
}

// The buffers a context's sequence-id group holds once every one of them is allocated.
static const int SEQ_ID_BUFFERS_PER_CONTEXT = 4;

int main(int, char ** argv) {
    sycl_test_selector_fallback(argv, "level_zero:0");

    if (ggml_backend_sycl_get_device_count() <= 0) {
        fprintf(stderr, "SKIP: no usable SYCL GPU\n");
        return LLAMA_TEST_EXIT_SKIP;
    }

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        fprintf(stderr, "ggml_backend_sycl_init(0) returned null\n");
        return 1;
    }

    const int allocs_before = ggml_sycl_test_seq_id_buffer_allocs();

    bool        touched = false;
    std::thread worker([&]() { touched = ggml_sycl_test_seq_id_buffers_touch(backend); });
    worker.join();
    if (!touched) {
        fprintf(stderr, "Failed to allocate the context's seq-id buffers in the worker thread\n");
        ggml_backend_free(backend);
        return 1;
    }

    // The worker has exited. The buffers are the context's, so they are still allocated.
    const int allocs_after_thread_exit = ggml_sycl_test_seq_id_buffer_allocs();
    if (allocs_after_thread_exit != allocs_before + SEQ_ID_BUFFERS_PER_CONTEXT) {
        fprintf(stderr, "Seq-id buffers did not outlive the thread that allocated them (before=%d after=%d)\n",
                allocs_before, allocs_after_thread_exit);
        ggml_backend_free(backend);
        return 1;
    }

    // A second touch finds the buffers and allocates nothing.
    if (!ggml_sycl_test_seq_id_buffers_touch(backend) ||
        ggml_sycl_test_seq_id_buffer_allocs() != allocs_after_thread_exit) {
        fprintf(stderr, "A second touch did not reuse the context's seq-id buffers (count=%d, expected %d)\n",
                ggml_sycl_test_seq_id_buffer_allocs(), allocs_after_thread_exit);
        ggml_backend_free(backend);
        return 1;
    }

    // The context's end is their end: this thread does not exit, and nothing else releases them.
    ggml_backend_free(backend);

    const int allocs_after_free = ggml_sycl_test_seq_id_buffer_allocs();
    if (allocs_after_free != allocs_before) {
        fprintf(stderr, "Freeing the backend left seq-id buffers allocated (before=%d after=%d)\n", allocs_before,
                allocs_after_free);
        return 1;
    }

    printf("Context-owned seq-id buffers released with the backend\n");
    return 0;
}

#endif
