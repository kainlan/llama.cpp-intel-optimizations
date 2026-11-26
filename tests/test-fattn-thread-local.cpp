// Device test: FlashAttention's device buffers belong to the backend context.
//
// The sequence-id buffers and the operand-staging slots (sinks and mask) are members of the backend context. This
// test reserves both groups on a context from a worker thread and checks, for each group, that
//   - the allocations are still held after that thread has exited (a thread's end releases nothing),
//   - a second reserve of the same size reuses every buffer,
//   - a larger reserve replaces every buffer, one freed for one allocated,
//   - a reserve on another queue replaces every buffer, so none stays on the queue the context left, and
//   - freeing the backend releases all of them, on a thread that stays alive.
// It reads the counters the buffers' release keeps and the allocator ids of their handles, which are never reused.
// It does not read the device arena.
// The file name is the one its registration and the test inventory carry.
//
// Usage:
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-fattn-thread-local

#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml-sycl/unified-cache.hpp"
#include "ggml.h"
#include "sycl-selector-fallback.hpp"
#include "test-skip.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sycl/sycl.hpp>
#include <thread>

#if !defined(GGML_USE_SYCL)
int main() {
    fprintf(stderr, "GGML_USE_SYCL not enabled; skipping test.\n");
    return LLAMA_TEST_EXIT_SKIP;
}
#else

extern "C" {
int           ggml_sycl_test_seq_id_buffer_allocs();
bool          ggml_sycl_test_seq_id_buffers_touch(ggml_backend_t backend, size_t bytes, sycl::queue * stream);
size_t        ggml_sycl_test_seq_id_buffer_ids(ggml_backend_t backend, uint64_t * ids, size_t capacity);
int           ggml_sycl_test_fattn_operand_stage_allocs();
bool          ggml_sycl_test_fattn_operand_stage_touch(ggml_backend_t backend, size_t bytes, sycl::queue * stream);
size_t        ggml_sycl_test_fattn_operand_stage_ids(ggml_backend_t backend, uint64_t * ids, size_t capacity);
sycl::queue * ggml_sycl_test_fattn_context_stream(ggml_backend_t backend);
}

static const size_t MAX_PER_CONTEXT = 4;
static const size_t SMALL_BYTES     = 64 * sizeof(float);
static const size_t LARGE_BYTES     = 16 * SMALL_BYTES;

// One group of context-owned buffers: how to reserve it (a null stream is the context's own queue), how to count
// what is live, how to read the allocator ids of what the context holds, and how many buffers the group has.
struct buffer_group {
    const char * name;
    bool (*touch)(ggml_backend_t backend, size_t bytes, sycl::queue * stream);
    int (*live)();
    size_t (*ids)(ggml_backend_t backend, uint64_t * ids, size_t capacity);
    size_t per_context;
};

static const buffer_group BUFFER_GROUPS[] = {
    { "seq-id buffers",        ggml_sycl_test_seq_id_buffers_touch,      ggml_sycl_test_seq_id_buffer_allocs,
     ggml_sycl_test_seq_id_buffer_ids,       4 },
    { "operand-staging slots", ggml_sycl_test_fattn_operand_stage_touch, ggml_sycl_test_fattn_operand_stage_allocs,
     ggml_sycl_test_fattn_operand_stage_ids, 2 },
};

static const size_t N_BUFFER_GROUPS = sizeof(BUFFER_GROUPS) / sizeof(BUFFER_GROUPS[0]);

struct buffer_ids {
    uint64_t id[MAX_PER_CONTEXT] = {};
};

// Reads the ids of the group's buffers. False, with its own message, when the group has another number of buffers
// than the table says or when one of them has no id: the checks below could not tell reused from replaced.
static bool buffer_ids_read(const buffer_group & group, ggml_backend_t backend, buffer_ids & out) {
    const size_t n = group.ids(backend, out.id, MAX_PER_CONTEXT);
    if (n != group.per_context) {
        fprintf(stderr, "The context reports %zu %s, expected %zu\n", n, group.name, group.per_context);
        return false;
    }
    for (size_t i = 0; i < group.per_context; ++i) {
        if (out.id[i] == 0) {
            fprintf(stderr, "Buffer %zu of the context's %s has no allocator id\n", i, group.name);
            return false;
        }
    }
    return true;
}

// Reserves `bytes` in every buffer of the group on `stream` and reads the ids back. False, with its own message,
// when the reserve fails or an id cannot be read.
static bool buffer_group_touch(const buffer_group & group,
                               ggml_backend_t       backend,
                               size_t               bytes,
                               sycl::queue *        stream,
                               buffer_ids &         out) {
    if (!group.touch(backend, bytes, stream)) {
        fprintf(stderr, "Failed to reserve %zu bytes in the context's %s\n", bytes, group.name);
        return false;
    }
    return buffer_ids_read(group, backend, out);
}

// How many of the group's buffers are the same allocation in both readings.
static size_t buffer_ids_kept(const buffer_group & group, const buffer_ids & a, const buffer_ids & b) {
    size_t kept = 0;
    for (size_t i = 0; i < group.per_context; ++i) {
        kept += a.id[i] == b.id[i] ? 1 : 0;
    }
    return kept;
}

// Whether the context still allocates on `home`. The reuse check below means something only then: a context whose
// queue changed between two touches is right to replace every buffer. False with its own message.
static bool context_queue_is(ggml_backend_t backend, const sycl::queue * home, const char * when) {
    if (ggml_sycl_test_fattn_context_stream(backend) != home) {
        fprintf(stderr, "The context's queue changed between touches (%s)\n", when);
        return false;
    }
    return true;
}

// Runs the checks that need a live context on one group. `before` is the group's live count before the context
// allocated anything, `home` is the queue the context allocates on, and `other` is a second queue on the same
// SYCL context and device.
static bool buffer_group_check(const buffer_group & group,
                               ggml_backend_t       backend,
                               int                  before,
                               const sycl::queue *  home,
                               sycl::queue &        other) {
    const int expected = before + static_cast<int>(group.per_context);

    bool        touched = false;
    std::thread worker([&]() { touched = group.touch(backend, SMALL_BYTES, nullptr); });
    worker.join();
    if (!touched) {
        fprintf(stderr, "Failed to allocate the context's %s in the worker thread\n", group.name);
        return false;
    }
    if (!context_queue_is(backend, home, "after the first")) {
        return false;
    }

    // The worker has exited. The allocations are the context's, so they are still held.
    if (group.live() != expected) {
        fprintf(stderr, "The %s did not outlive the thread that allocated them (before=%d after=%d)\n", group.name,
                before, group.live());
        return false;
    }
    buffer_ids first;
    if (!buffer_ids_read(group, backend, first)) {
        return false;
    }

    // The same size again: every buffer is found, none is allocated.
    buffer_ids again;
    if (!buffer_group_touch(group, backend, SMALL_BYTES, nullptr, again) ||
        !context_queue_is(backend, home, "after the second")) {
        return false;
    }
    if (group.live() != expected || buffer_ids_kept(group, first, again) != group.per_context) {
        fprintf(stderr, "A second touch did not reuse the context's %s (count=%d, expected %d, kept %zu of %zu)\n",
                group.name, group.live(), expected, buffer_ids_kept(group, first, again), group.per_context);
        return false;
    }

    // A larger size: every buffer is replaced, and each replacement frees the buffer it replaces.
    buffer_ids grown;
    if (!buffer_group_touch(group, backend, LARGE_BYTES, nullptr, grown)) {
        return false;
    }
    if (group.live() != expected || buffer_ids_kept(group, again, grown) != 0) {
        fprintf(stderr, "Growing the context's %s did not replace each one for one (count=%d, expected %d, kept %zu)\n",
                group.name, group.live(), expected, buffer_ids_kept(group, again, grown));
        return false;
    }

    // Another queue, the size unchanged: the whole group moves, although no buffer is too small. A buffer that
    // stayed would be read through a queue it was not allocated on.
    buffer_ids moved;
    if (!buffer_group_touch(group, backend, LARGE_BYTES, &other, moved)) {
        return false;
    }
    if (group.live() != expected || buffer_ids_kept(group, grown, moved) != 0) {
        fprintf(stderr,
                "A touch on another queue left the context's %s on the queue they came from (count=%d, expected %d, "
                "kept %zu)\n",
                group.name, group.live(), expected, buffer_ids_kept(group, grown, moved));
        return false;
    }

    return true;
}

int main(int, char ** argv) {
    sycl_test_selector_fallback(argv, "level_zero:0");

    if (ggml_backend_sycl_get_device_count() <= 0) {
        fprintf(stderr, "SKIP: no usable SYCL GPU\n");
        return LLAMA_TEST_EXIT_SKIP;
    }

    const int      device  = 0;
    ggml_backend_t backend = ggml_backend_sycl_init(device);
    if (!backend) {
        fprintf(stderr, "ggml_backend_sycl_init(%d) returned null\n", device);
        return 1;
    }

    // The context allocates on the queue of its device's unified cache, and answers the device's default queue until
    // that cache exists. A loaded model has created the cache before any op runs. This test loads none, so it creates
    // the cache here: otherwise the first touch would create it, and the context would be on another queue for the
    // second touch.
    if (!ggml_sycl::get_unified_cache_for_device(device)) {
        fprintf(stderr, "Could not create the unified cache of device %d\n", device);
        ggml_backend_free(backend);
        return 1;
    }

    sycl::queue * home = ggml_sycl_test_fattn_context_stream(backend);
    if (!home) {
        fprintf(stderr, "The backend context has no queue\n");
        ggml_backend_free(backend);
        return 1;
    }
    // A second queue on the SYCL context and device of `home`, the cache's queue: the last check is a change of
    // queue on one device. It is declared here so that it outlives the backend: the buffers that check leaves on it
    // are released by ggml_backend_free().
    sycl::queue other(home->get_context(), home->get_device());

    int before[N_BUFFER_GROUPS];
    for (size_t g = 0; g < N_BUFFER_GROUPS; ++g) {
        before[g] = BUFFER_GROUPS[g].live();
    }

    for (size_t g = 0; g < N_BUFFER_GROUPS; ++g) {
        if (!buffer_group_check(BUFFER_GROUPS[g], backend, before[g], home, other)) {
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
