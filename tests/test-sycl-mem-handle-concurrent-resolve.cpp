// Regression test for llama.cpp-c48l.
//
// mem_handle::resolve() is declared const but MUTATES the handle: it rewrites
// cached_ (a resolved_ptr, which carries a sycl::event -- i.e. a shared_ptr),
// gen_, leased_entry_ and the chunk-lease fields.  Until this test's fix landed
// the handle carried no synchronisation at all.
//
// That is a live defect rather than a theoretical one because a mem_handle is
// SHARED state: every weight tensor's ggml_tensor_extra_gpu::data_handle[dev]
// is resolved by whichever context thread is dispatching, llama.cpp supports
// many contexts per model (tests/test-thread-safety.cpp creates n_parallel of
// them), and ggml_backend_sycl_graph_compute releases the process-wide
// g_sycl_graph_compute_mutex before dispatch on nearly every path
// (compute_impl_unlocked), so those resolves genuinely overlap.  The observed
// symptom was a SIGSEGV inside mem_handle::resolve() itself, reached from
// ggml_sycl_tensor_metadata_owner -> ggml_sycl_find_tensor_device_owner.
//
// Both tests below are CPU-only: they construct DIRECT/WEIGHT handles by hand
// and never touch a device, a queue, or the unified cache.  They can be run
// anywhere, including on a machine with no GPU.
//
// Usage:
//   ./build/bin/test-sycl-mem-handle-concurrent-resolve

#include "mem-handle.hpp"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

namespace {

// Kept small enough to finish in a couple of seconds, large enough that an
// unsynchronised publish is observed essentially every run.  The pre-fix
// failure mode is a torn read within the first few thousand iterations.
constexpr int  kReaderThreads = 4;
constexpr long kIterations    = 400000;

// A resolve concurrent with a state change must observe one WHOLE published
// state, never a mixture of the old and the new.  The two states below differ
// in all three fields of resolved_ptr that a caller acts on, so any field-by-
// field publish is detectable without relying on a crash.
int test_resolve_never_observes_a_half_written_state() {
    alignas(64) std::uint8_t buf_a[64] = {};
    alignas(64) std::uint8_t buf_b[64] = {};

    const ggml_sycl::mem_handle state_a = ggml_sycl::mem_handle::from_direct(buf_a, GGML_LAYOUT_AOS, false);
    const ggml_sycl::mem_handle state_b = ggml_sycl::mem_handle::from_direct(buf_b, GGML_LAYOUT_SOA, true);

    ggml_sycl::mem_handle shared = state_a;

    std::atomic<bool> stop{ false };
    std::atomic<long> torn{ 0 };

    std::vector<std::thread> readers;
    readers.reserve(kReaderThreads);
    for (int t = 0; t < kReaderThreads; ++t) {
        readers.emplace_back([&]() {
            while (!stop.load(std::memory_order_relaxed)) {
                const ggml_sycl::resolved_ptr r    = shared.resolve();
                const bool                    is_a = r.ptr == buf_a && r.layout == GGML_LAYOUT_AOS && !r.on_device;
                const bool                    is_b = r.ptr == buf_b && r.layout == GGML_LAYOUT_SOA && r.on_device;
                if (!is_a && !is_b) {
                    torn.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    for (long i = 0; i < kIterations; ++i) {
        shared = (i & 1) ? state_b : state_a;
    }
    stop.store(true, std::memory_order_relaxed);
    for (auto & th : readers) {
        th.join();
    }

    const long observed = torn.load(std::memory_order_relaxed);
    if (observed != 0) {
        std::fprintf(stderr,
                     "FAIL: %ld resolve() results mixed fields from two different published states\n"
                     "      (resolve() read cached_ while another thread wrote it)\n",
                     observed);
        return 1;
    }
    std::puts("PASS: concurrent resolve() never observes a half-written state");
    return 0;
}

// The production shape: many threads resolving ONE shared handle while the
// global cache generation churns, which is what invalidates every handle at
// once and drives them all into the mutating slow path simultaneously.  Before
// the fix this raced on cached_'s sycl::event refcount and on leased_entry_;
// the assertion here is simply that it completes without corrupting memory.
int test_concurrent_resolve_under_generation_churn() {
    ggml_sycl::unified_cache_key key = {};
    key.type                         = ggml_sycl::cache_entry_type::DENSE_WEIGHT;
    key.layer_id                     = -1;
    key.expert_id                    = -1;

    // HOST_DEVICE keeps resolve_slow() out of the unified cache entirely, so
    // this exercises the handle's own state machine with no device present.
    const ggml_sycl::mem_handle shared = ggml_sycl::mem_handle::from_weight(key, ggml_sycl::mem_handle::HOST_DEVICE);

    std::atomic<bool> stop{ false };

    std::vector<std::thread> readers;
    readers.reserve(kReaderThreads);
    for (int t = 0; t < kReaderThreads; ++t) {
        readers.emplace_back([&]() {
            while (!stop.load(std::memory_order_relaxed)) {
                const ggml_sycl::resolved_ptr r = shared.resolve();
                (void) r;
                (void) shared.valid();
                (void) shared.stable_identity_hash();
            }
        });
    }

    for (long i = 0; i < kIterations; ++i) {
        ggml_sycl::cache_generation_bump();
    }
    stop.store(true, std::memory_order_relaxed);
    for (auto & th : readers) {
        th.join();
    }

    std::puts("PASS: concurrent resolve() under generation churn is memory-safe");
    return 0;
}

}  // namespace

int main() {
    if (int rc = test_resolve_never_observes_a_half_written_state()) {
        return rc;
    }
    if (int rc = test_concurrent_resolve_under_generation_churn()) {
        return rc;
    }
    std::puts("PASS: mem_handle concurrent resolve");
    return 0;
}
