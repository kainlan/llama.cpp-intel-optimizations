//
// The fixture side of lazy-io-pool.cpp's two link seams (docs/design/sycl-storage-tier-lazy-tables.md §3.3, §10), for
// a host test that links the pool's translation unit without the backend: the block ring's memory from the aligned
// operator new, with a fixture ledger that counts allocations and charges each at host_cpu_mapped_size, and the
// strict-leases switch, off unless a death child turns it on, so no host arm reads GGML_SYCL_STRICT_LEASES. Each such
// test includes it once.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include "../host-cpu-mapping.hpp"
#include "../lazy-io-pool.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>

namespace lazy_io_host_fixture {

// The fixture ledger and its switches. The counters only grow, except live_charged, which a ring's destructor credits.
struct state {
    std::atomic<uint64_t> allocations{ 0 };      // rings the seam handed out
    std::atomic<uint64_t> live_charged{ 0 };     // the charged bytes of the rings still alive
    std::atomic<uint64_t> last_load_txn{ 0 };    // the transaction the last allocation was charged under
    std::atomic<uint64_t> last_pool_gen{ 0 };    // the pool generation the last allocation was owned by
    std::atomic<bool>     refuse_next{ false };  // the next allocation returns null
    std::atomic<bool>     throw_next{ false };   // the next allocation throws std::bad_alloc
    std::atomic<size_t>   short_next{ 0 };       // the next allocation's usable bytes fall short by this many
    std::atomic<bool>     strict{ false };       // ggml_sycl_strict_enabled(): GGML_SYCL_STRICT_LEASES=1's stand-in
};

inline state g_state;

class memory final : public ggml_sycl::lazy_io_host_memory {
  public:
    memory(uint8_t * data, size_t bytes, size_t charged) : lazy_io_host_memory(data, bytes, charged) {}

    ~memory() override {
        g_state.live_charged.fetch_sub(charged_bytes(), std::memory_order_relaxed);
        ::operator delete(data(), std::align_val_t{ ggml_sycl::LAZY_IO_BLOCK_BYTES });
    }
};

}  // namespace lazy_io_host_fixture

namespace ggml_sycl {

std::unique_ptr<lazy_io_host_memory> lazy_io_allocate_host(size_t   bytes,
                                                           int      device,
                                                           uint64_t pool_generation,
                                                           uint64_t load_txn_id) {
    using lazy_io_host_fixture::g_state;
    (void) device;
    if (g_state.throw_next.exchange(false)) {
        throw std::bad_alloc();
    }
    if (g_state.refuse_next.exchange(false) || bytes == 0) {
        return nullptr;
    }
    const size_t usable  = bytes - std::min(bytes, g_state.short_next.exchange(0));
    auto *       data    = static_cast<uint8_t *>(::operator new(bytes, std::align_val_t{ LAZY_IO_BLOCK_BYTES }));
    const size_t charged = host_cpu_mapped_size(bytes);
    std::memset(data, 0, bytes);
    g_state.allocations.fetch_add(1, std::memory_order_relaxed);
    g_state.live_charged.fetch_add(charged, std::memory_order_relaxed);
    g_state.last_load_txn.store(load_txn_id, std::memory_order_relaxed);
    g_state.last_pool_gen.store(pool_generation, std::memory_order_relaxed);
    return std::unique_ptr<lazy_io_host_memory>(new lazy_io_host_fixture::memory(data, usable, charged));
}

bool ggml_sycl_strict_enabled() {
    return lazy_io_host_fixture::g_state.strict.load(std::memory_order_relaxed);
}

}  // namespace ggml_sycl
