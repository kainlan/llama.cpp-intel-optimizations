#pragma once

// Per-thread scratch for the CPU dispatch paths (host-resident expert execution).
// Pure std types: kept out of common.hpp so a host-only test can include it.
//
// llama.cpp-z5fn: init() used to size scratch_nk to MAX_N * MAX_K floats (224 MiB)
// on every thread that called it, including every TBB arena worker of
// ggml_sycl_cpu_expert_mul_mat_batched, which never reads scratch_nk. Zero-filled
// by std::vector::resize, that was ~10 GiB of untracked host RSS at 46 threads.
// scratch_nk is now sized on demand, by the two dequant fallbacks that use it,
// to the N * K they actually need.

#include "host-mem-ledger.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

struct cpu_dispatch_buffers {
    std::vector<uint8_t> src1_q;      // Quantization buffer: max M * max_q_row_size
    std::vector<float>   accs;        // Accumulator buffer: reused as __m256* via reinterpret_cast
    std::vector<float>   scratch_nk;  // Weight dequantization buffer: N * K, grown on demand

    // Note: accs is reinterpreted as __m256 array. Since we only use _mm256_setzero_ps()
    // and array indexing (no aligned load/store), alignment is not critical.

    // Size the small, always-needed buffers. Cheap to repeat: resize to the same size is a no-op.
    void init(size_t max_m, size_t max_q_row_size) {
        src1_q.resize(max_m * max_q_row_size);
        // __m256 is 32 bytes = 8 floats; allocate for max chunk4 (256 + max_m) accumulators
        accs.resize((256 + max_m) * 8);  // Conservative upper bound: 256 stack + max_m heap
    }

    // Grow scratch_nk to at least n_floats and return it. Never shrinks.
    float * ensure_scratch_nk(size_t n_floats) {
        if (scratch_nk.size() < n_floats) {
            const size_t before = scratch_nk.capacity();
            scratch_nk.resize(n_floats);
            ledger_add(scratch_nk.capacity(), before);
        }
        return scratch_nk.data();
    }

    cpu_dispatch_buffers() = default;
    cpu_dispatch_buffers(const cpu_dispatch_buffers &)             = delete;
    cpu_dispatch_buffers & operator=(const cpu_dispatch_buffers &) = delete;

    ~cpu_dispatch_buffers() {
        ggml_sycl::host_mem_ledger_get().cpu_dispatch_scratch_bytes.fetch_sub(scratch_nk.capacity() * sizeof(float),
                                                                              std::memory_order_relaxed);
    }

private:
    static void ledger_add(size_t after_cap, size_t before_cap) {
        ggml_sycl::host_mem_ledger_get().cpu_dispatch_scratch_bytes.fetch_add((after_cap - before_cap) * sizeof(float),
                                                                              std::memory_order_relaxed);
    }

public:
    size_t resident_bytes() const {
        return src1_q.capacity() * sizeof(uint8_t) + accs.capacity() * sizeof(float) +
               scratch_nk.capacity() * sizeof(float);
    }
};
