// Legacy device-pool release rule (VRAM arena off).
//
// Host-only: no SYCL queue, no device. pool_leg's arena-off free() calls
// pool_legacy_release(); the slot and owner types below mirror its
// ggml_sycl_buffer and ggml_sycl_pool_owned_buffer, and the handle counts how
// many owners are alive so a test can see exactly when memory is released.

#include "pool-legacy-release.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ggml_sycl;

namespace {

struct counted_handle {
    int * live = nullptr;

    counted_handle() = default;

    explicit counted_handle(int * live_) : live(live_) { ++*live; }

    counted_handle(counted_handle && other) noexcept : live(other.live) { other.live = nullptr; }

    counted_handle & operator=(counted_handle && other) noexcept {
        if (this != &other) {
            reset();
            live       = other.live;
            other.live = nullptr;
        }
        return *this;
    }

    ~counted_handle() { reset(); }

    void reset() {
        if (live) {
            --*live;
            live = nullptr;
        }
    }

    bool valid() const { return live != nullptr; }
};

struct slot {
    void *         ptr  = nullptr;
    size_t         size = 0;
    counted_handle handle;
};

struct owned {
    size_t         size = 0;
    counted_handle handle;
};

constexpr int kSlots = 4;

// The pool state pool_leg keeps on the arena-off path.
struct legacy_pool {
    slot                              slots[kSlots];
    std::unordered_map<void *, owned> active;
    std::vector<counted_handle>       graph_retained;
    size_t                            pool_size = 0;
    int                               live      = 0;

    void * alloc(uintptr_t addr, size_t size) {
        void * ptr  = reinterpret_cast<void *>(addr);
        active[ptr] = { size, counted_handle(&live) };
        pool_size += size;
        return ptr;
    }

    pool_legacy_release_result release(void * ptr, bool recording) {
        return pool_legacy_release(ptr, recording, slots, kSlots, active, graph_retained, pool_size);
    }

    bool on_free_list(void * ptr) const {
        for (const slot & s : slots) {
            if (s.ptr == ptr) {
                return true;
            }
        }
        return false;
    }
};

int failures = 0;

void check(bool condition, const std::string & message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
        ++failures;
    }
}

// Control: outside recording a freed block is cached for reuse, so the free
// list really is where a later alloc() finds memory.
void test_free_outside_recording_is_cached() {
    legacy_pool pool;
    void *      ptr = pool.alloc(0x1000, 4096);

    check(pool.release(ptr, false) == pool_legacy_release_result::CACHED, "outside recording: result is not CACHED");
    check(pool.on_free_list(ptr), "outside recording: freed block is not on the free list");
    check(pool.graph_retained.empty(), "outside recording: a block was parked with a graph");
    check(pool.live == 1, "outside recording: the cached block's owner was released");
    check(pool.pool_size == 4096, "outside recording: pool accounting dropped a cached block");
}

// The bug: a block freed while a graph is recording went back on the free
// list, so the next alloc() could hand out memory the recorded graph still
// points into.
void test_free_during_recording_is_retained_with_the_graph() {
    legacy_pool pool;
    void *      ptr = pool.alloc(0x2000, 8192);

    const pool_legacy_release_result result = pool.release(ptr, true);
    check(!pool.on_free_list(ptr),
          "during recording: freed block went back on the free list, so a later alloc() can reuse memory a "
          "recorded graph still references");
    check(result == pool_legacy_release_result::GRAPH_RETAINED, "during recording: result is not GRAPH_RETAINED");
    check(pool.graph_retained.size() == 1, "during recording: the owner was not parked with the graph");
    check(pool.live == 1, "during recording: the owner was released while the graph lives");
    check(pool.active.empty(), "during recording: the block is still tracked as active");
    check(pool.pool_size == 0, "during recording: pool accounting still counts a block the graph now owns");

    // Dropping the graph drops its retained handles; only then is the memory
    // released.
    pool.graph_retained.clear();
    check(pool.live == 0, "graph dropped: the retained owner was not released");
}

// With the free list full, outside recording the owner is released at once;
// during recording that would free memory under the graph.
void test_full_free_list_during_recording_is_retained_not_dropped() {
    legacy_pool pool;
    for (int i = 0; i < kSlots; ++i) {
        void * filler = pool.alloc(0x10000 + 0x1000 * i, 256);
        check(pool.release(filler, false) == pool_legacy_release_result::CACHED, "setup: filler was not cached");
    }
    const int cached_live = pool.live;

    void * dropped = pool.alloc(0x20000, 512);
    check(pool.release(dropped, false) == pool_legacy_release_result::DROPPED,
          "full free list, outside recording: result is not DROPPED");
    check(pool.live == cached_live, "full free list, outside recording: the owner was not released");

    void * kept = pool.alloc(0x30000, 512);
    check(pool.release(kept, true) == pool_legacy_release_result::GRAPH_RETAINED,
          "full free list, during recording: the owner was dropped, freeing memory a recorded graph references");
    check(pool.live == cached_live + 1, "full free list, during recording: the owner is not alive");
    check(pool.pool_size == kSlots * 256, "full free list: pool accounting is off");
}

// A free the pool never handed out has no owner to route; the caller asserts.
void test_free_without_owner_fails_closed() {
    for (bool recording : { false, true }) {
        legacy_pool pool;
        check(pool.release(reinterpret_cast<void *>(0x4000), recording) == pool_legacy_release_result::MISSING_OWNER,
              recording ? "unknown block during recording: not MISSING_OWNER" :
                          "unknown block outside recording: not MISSING_OWNER");
        check(pool.graph_retained.empty() && !pool.on_free_list(reinterpret_cast<void *>(0x4000)),
              "unknown block: it was routed anyway");
    }
}

}  // namespace

int main() {
    test_free_outside_recording_is_cached();
    test_free_during_recording_is_retained_with_the_graph();
    test_full_free_list_during_recording_is_retained_not_dropped();
    test_free_without_owner_fails_closed();

    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::puts("PASS: legacy pool release parks graph-recorded frees with the graph");
    return 0;
}
