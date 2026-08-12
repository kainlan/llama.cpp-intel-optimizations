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
// The three tests below cover the two failure modes separately, because they do
// NOT fail the same way:
//
//   1. test_resolve_never_observes_a_half_written_state -- the LOUD one.  A
//      resolve reading cached_ while another thread writes it corrupts the
//      shared_ptr inside resolved_ptr::ready_event, which faults inside
//      resolve() and aborts the allocator.
//   2. test_lease_is_released_exactly_once -- the SILENT one, and the more
//      dangerous.  A lease released twice drives unified_cache_entry::
//      in_use_count below the number of live references, so the evictor is free
//      to free an allocation still in use.  Nothing reports it.
//   3. test_concurrent_resolve_under_generation_churn -- the mirror of (2):
//      resolve_slow()'s early exit for a non-cache device dropped the lease
//      pointer WITHOUT decrementing, leaking the entry instead of releasing it.
//
// All three are CPU-only: they construct DIRECT/WEIGHT handles and a stack
// unified_cache_entry by hand and never touch a device, a queue, or the unified
// cache.  They can be run anywhere, including on a machine with no GPU.
//
// Usage:
//   ./build/bin/test-sycl-mem-handle-concurrent-resolve            # all three
//   ./build/bin/test-sycl-mem-handle-concurrent-resolve mixed-state
//   ./build/bin/test-sycl-mem-handle-concurrent-resolve lease-once
//   ./build/bin/test-sycl-mem-handle-concurrent-resolve gen-churn
//
// The selector exists because test 1's pre-fix failure mode is an ABORT
// (allocator corruption), which would stop the process before tests 2 and 3
// ever ran.  Without it, the two lease tests could not be shown failing on an
// unfixed build -- and a regression test whose RED cannot be demonstrated is
// exactly the shape this file exists to avoid.

#include "mem-handle.hpp"
#include "unified-cache.hpp"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace {

// Kept small enough to finish in a couple of seconds, large enough that an
// unsynchronised publish is observed essentially every run.  The pre-fix
// failure mode is a torn read within the first few thousand iterations.
constexpr int  kReaderThreads = 4;
constexpr int  kWriterThreads = 3;
constexpr long kIterations    = 400000;

// A handle key that is distinct per test but otherwise inert: nothing here ever
// reaches the unified cache, so only the key's identity matters.
ggml_sycl::unified_cache_key make_key(uint64_t tag) {
    ggml_sycl::unified_cache_key key = {};
    key.type                         = ggml_sycl::cache_entry_type::DENSE_WEIGHT;
    key.layer_id                     = -1;
    key.expert_id                    = -1;
    key.id.name_hash                 = tag;
    return key;
}

uint32_t lease_count(const ggml_sycl::unified_cache_entry & entry) {
    return entry.in_use_count.v.load(std::memory_order_acquire);
}

void initialize_host_entry(ggml_sycl::unified_cache_entry & entry,
                           void *                            ptr,
                           size_t                            size,
                           ggml_layout_mode                  layout) {
    entry.device_ptr    = ptr;
    entry.src_ptr       = ptr;
    entry.size          = size;
    entry.type          = ggml_sycl::cache_entry_type::DENSE_WEIGHT;
    entry.layer_id      = -1;
    entry.expert_id     = -1;
    entry.layout        = layout;
    entry.state         = ggml_sycl::cache_entry_state::READY;
    entry.location      = ggml_sycl::cache_location::HOST_PINNED;
    entry.host_resident = true;
    entry.owner_device  = ggml_sycl::mem_handle::HOST_DEVICE;
}

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

// THE LEASE-ACCOUNTING TEST, and the one that matters most.
//
// The mixed-state race above crashes loudly.  This one corrupts a refcount
// SILENTLY: mem_handle's lease on a unified_cache_entry is what stops the
// evictor freeing an allocation that is still in use, so a lease released twice
// drives in_use_count below the number of live references and makes an in-use
// entry evictable.  Nothing reports that; the damage surfaces later as freed
// memory being read.
//
// The pre-fix defect: every transition (operator=, dtor, resolve_slow) called
// release_lease(), which read leased_entry_, decremented it, then nulled it.
// Two threads reaching that on the SAME handle both saw the non-null pointer and
// both decremented -- one lease, two releases.
//
// Device-free by construction: unified_cache_entry is a plain struct, and
// from_weight_lease_snapshot() skips its chunk-lease cache lookup at HOST_DEVICE
// (mem-handle.cpp gates it on valid_cache_device_id), so no cache, queue or
// device is ever touched.  Only the refcount arithmetic is under test.
int test_lease_is_released_exactly_once() {
    alignas(64) std::uint8_t buf_a[64] = {};
    alignas(64) std::uint8_t buf_b[64] = {};

    ggml_sycl::unified_cache_entry entry_a = {};
    ggml_sycl::unified_cache_entry entry_b = {};
    initialize_host_entry(entry_a, buf_a, sizeof(buf_a), GGML_LAYOUT_AOS);
    initialize_host_entry(entry_b, buf_b, sizeof(buf_b), GGML_LAYOUT_SOA);

    // Each template handle owns one lease -- from_weight_lease_snapshot() takes ownership
    // of a bump the caller is expected to have already made.
    entry_a.in_use_count.v.store(1, std::memory_order_release);
    entry_b.in_use_count.v.store(1, std::memory_order_release);

    const ggml_sycl::mem_handle lease_a = ggml_sycl::mem_handle::from_weight_lease_snapshot(
        make_key(0xa11a), ggml_sycl::mem_handle::HOST_DEVICE, buf_a, GGML_LAYOUT_AOS, false, &entry_a,
        entry_a.storage_owner, entry_a.has_ready_event, entry_a.ready_event);
    const ggml_sycl::mem_handle lease_b = ggml_sycl::mem_handle::from_weight_lease_snapshot(
        make_key(0xb22b), ggml_sycl::mem_handle::HOST_DEVICE, buf_b, GGML_LAYOUT_SOA, false, &entry_b,
        entry_b.storage_owner, entry_b.has_ready_event, entry_b.ready_event);

    {
        ggml_sycl::mem_handle shared = lease_a;  // +1 on entry_a

        std::atomic<bool> stop{ false };

        std::vector<std::thread> readers;
        readers.reserve(kReaderThreads);
        for (int t = 0; t < kReaderThreads; ++t) {
            readers.emplace_back([&]() {
                while (!stop.load(std::memory_order_relaxed)) {
                    const ggml_sycl::resolved_ptr r = shared.resolve();
                    (void) r;
                    (void) shared.valid();
                }
            });
        }

        // Several writers, because ONE writer cannot double-release: the race
        // needs two threads inside a lease transition on the same handle.
        std::vector<std::thread> writers;
        writers.reserve(kWriterThreads);
        for (int w = 0; w < kWriterThreads; ++w) {
            writers.emplace_back([&, w]() {
                for (long i = 0; i < kIterations; ++i) {
                    shared = ((i + w) & 1) ? lease_b : lease_a;
                }
            });
        }

        for (auto & th : writers) {
            th.join();
        }
        stop.store(true, std::memory_order_relaxed);
        for (auto & th : readers) {
            th.join();
        }
        // `shared` releases its one remaining lease here.
    }

    // Exactly the two template handles remain, one lease each.  Anything else
    // means releases and acquisitions did not pair up.
    const uint32_t count_a = lease_count(entry_a);
    const uint32_t count_b = lease_count(entry_b);
    if (count_a != 1 || count_b != 1) {
        std::fprintf(stderr,
                     "FAIL: lease refcounts drifted: entry_a=%u entry_b=%u (expected 1 and 1)\n"
                     "      a count below 1 is a double release -- an in-use entry the evictor may now free;\n"
                     "      a count above 1 is a leaked lease -- an entry that can never be reclaimed\n",
                     count_a, count_b);
        return 1;
    }
    std::puts("PASS: concurrent lease transitions release exactly once");
    return 0;
}

// The production shape: many threads resolving ONE shared handle while the
// global cache generation churns, which is what invalidates every handle at
// once and drives them all into the mutating slow path simultaneously.
//
// This covers a SECOND lease defect, distinct from the double-release above.
// resolve_slow()'s early exit for a non-cache device used to drop the lease by
// writing `leased_entry_ = nullptr` WITHOUT decrementing -- the mirror image:
// not a double release but no release at all, leaving in_use_count permanently
// above zero so the entry could never be evicted.  The handle is seeded with a
// lease here so the leak is observable as a count that never reaches 0.
int test_concurrent_resolve_under_generation_churn() {
    alignas(64) std::uint8_t buf[64] = {};

    ggml_sycl::unified_cache_entry entry = {};
    initialize_host_entry(entry, buf, sizeof(buf), GGML_LAYOUT_AOS);
    entry.in_use_count.v.store(1, std::memory_order_release);

    {
        // HOST_DEVICE keeps resolve_slow() out of the unified cache entirely, so
        // this exercises the handle's own state machine with no device present.
        const ggml_sycl::mem_handle shared = ggml_sycl::mem_handle::from_weight_lease_snapshot(
            make_key(0xc33c), ggml_sycl::mem_handle::HOST_DEVICE, buf, GGML_LAYOUT_AOS, false, &entry,
            entry.storage_owner, entry.has_ready_event, entry.ready_event);

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
    }

    const uint32_t count = lease_count(entry);
    if (count != 0) {
        std::fprintf(stderr,
                     "FAIL: lease still held after the handle was destroyed: in_use_count=%u (expected 0)\n"
                     "      resolve_slow()'s early exit dropped leased_entry_ without releasing it\n",
                     count);
        return 1;
    }
    std::puts("PASS: concurrent resolve() under generation churn releases its lease");
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    const char * only     = argc > 1 ? argv[1] : nullptr;
    auto         selected = [only](const char * name) {
        return !only || std::strcmp(only, name) == 0;
    };

    if (only && !selected("mixed-state") && !selected("lease-once") && !selected("gen-churn")) {
        std::fprintf(stderr, "unknown test '%s' (mixed-state | lease-once | gen-churn)\n", only);
        return 2;
    }

    if (selected("mixed-state")) {
        if (int rc = test_resolve_never_observes_a_half_written_state()) {
            return rc;
        }
    }
    if (selected("lease-once")) {
        if (int rc = test_lease_is_released_exactly_once()) {
            return rc;
        }
    }
    if (selected("gen-churn")) {
        if (int rc = test_concurrent_resolve_under_generation_churn()) {
            return rc;
        }
    }
    std::puts("PASS: mem_handle concurrent resolve");
    return 0;
}
