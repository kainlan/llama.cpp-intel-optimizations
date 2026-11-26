// Device test: FlashAttention's device buffers belong to the backend context.
//
// The sequence-id buffers and the operand-staging slots (sinks and mask) used to be thread_local, so they ended
// with the thread that ran the op and, on a thread that never exits, outlived every model. They are members of the
// backend context now. This test allocates both groups on a context from a worker thread and checks, for each, that
//   - the allocations are still held after that thread has exited (a thread's end releases nothing),
//   - a second touch reuses them, and
//   - freeing the backend releases all of them, on a thread that stays alive.
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
int  ggml_sycl_test_fattn_operand_stage_allocs();
bool ggml_sycl_test_fattn_operand_stage_touch(ggml_backend_t backend);
}

// One group of context-owned buffers: how to allocate it, how to count what is live, and how many allocations a
// context holds once the whole group is allocated.
struct buffer_group {
    const char * name;
    bool (*touch)(ggml_backend_t backend);
    int (*live)();
    int per_context;
};

static const buffer_group BUFFER_GROUPS[] = {
    { "seq-id buffers",        ggml_sycl_test_seq_id_buffers_touch,      ggml_sycl_test_seq_id_buffer_allocs,       4 },
    { "operand-staging slots", ggml_sycl_test_fattn_operand_stage_touch, ggml_sycl_test_fattn_operand_stage_allocs, 2 },
};

static const size_t N_BUFFER_GROUPS = sizeof(BUFFER_GROUPS) / sizeof(BUFFER_GROUPS[0]);

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

    int before[N_BUFFER_GROUPS];
    for (size_t g = 0; g < N_BUFFER_GROUPS; ++g) {
        before[g] = BUFFER_GROUPS[g].live();
    }

    for (size_t g = 0; g < N_BUFFER_GROUPS; ++g) {
        const buffer_group & group = BUFFER_GROUPS[g];

        bool        touched = false;
        std::thread worker([&]() { touched = group.touch(backend); });
        worker.join();
        if (!touched) {
            fprintf(stderr, "Failed to allocate the context's %s in the worker thread\n", group.name);
            ggml_backend_free(backend);
            return 1;
        }

        // The worker has exited. The allocations are the context's, so they are still held.
        const int after_thread_exit = group.live();
        if (after_thread_exit != before[g] + group.per_context) {
            fprintf(stderr, "The %s did not outlive the thread that allocated them (before=%d after=%d)\n", group.name,
                    before[g], after_thread_exit);
            ggml_backend_free(backend);
            return 1;
        }

        // A second touch finds them and allocates nothing.
        if (!group.touch(backend) || group.live() != after_thread_exit) {
            fprintf(stderr, "A second touch did not reuse the context's %s (count=%d, expected %d)\n", group.name,
                    group.live(), after_thread_exit);
            ggml_backend_free(backend);
            return 1;
        }
    }

    // The context's end is their end: this thread does not exit, and nothing else releases them.
    ggml_backend_free(backend);

    for (size_t g = 0; g < N_BUFFER_GROUPS; ++g) {
        const int after_free = BUFFER_GROUPS[g].live();
        if (after_free != before[g]) {
            fprintf(stderr, "Freeing the backend left %s allocated (before=%d after=%d)\n", BUFFER_GROUPS[g].name,
                    before[g], after_free);
            return 1;
        }
    }

    printf("Context-owned seq-id buffers and operand-staging slots released with the backend\n");
    return 0;
}

#endif
