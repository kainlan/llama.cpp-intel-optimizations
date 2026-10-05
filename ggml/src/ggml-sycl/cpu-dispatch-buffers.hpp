//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// cpu-dispatch-buffers.hpp -- per-thread scratch for the CPU dispatch paths
// (host-resident expert execution).
//
// Pure std types, kept out of common.hpp so a host-only test can include it.
//
// Every thread that runs CPU dispatch work (each TBB arena worker included)
// owns one cpu_dispatch_buffers. init() sizes the two small buffers every path
// needs, the quantized activation row and the accumulator tile, to about 1 MiB
// in total. The weight-dequant buffer (scratch_nk, N * K floats) is a different
// size class: only the two dequant GEMM fallbacks use it, only on the thread
// that calls them, and N * K is model-sized. It is therefore not sized by
// init(); ensure_scratch_nk() grows it on demand, uninitialised (both callers
// write every element before reading it), and counts it in the host-memory
// ledger so [HOSTMEM] can account for it.

#pragma once

#include "host-mem-ledger.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

// Upper bounds init() sizes for:
//  - batch size for TG: 16 tokens
//  - n_ff (feedforward hidden): 14336 (typical 7B; ~3.5x n_embd)
//  - quantized row size: (K / 32 + 1) blocks of at most 128 bytes, a safe bound for any quant type
constexpr size_t CPU_DISPATCH_MAX_M          = 16;
constexpr size_t CPU_DISPATCH_MAX_K          = 14336;
constexpr size_t CPU_DISPATCH_MAX_Q_ROW_SIZE = (CPU_DISPATCH_MAX_K / 32 + 1) * 128;

struct cpu_dispatch_buffers {
    std::vector<uint8_t> src1_q;  // Quantization buffer: CPU_DISPATCH_MAX_M * CPU_DISPATCH_MAX_Q_ROW_SIZE
    std::vector<float>   accs;    // Accumulator buffer: reused as __m256* via reinterpret_cast

    // Note: accs is reinterpreted as __m256 array. Since we only use _mm256_setzero_ps()
    // and array indexing (no aligned load/store), alignment is not critical.

    cpu_dispatch_buffers()                                         = default;
    cpu_dispatch_buffers(const cpu_dispatch_buffers &)             = delete;
    cpu_dispatch_buffers & operator=(const cpu_dispatch_buffers &) = delete;

    // scratch_nk is counted in host_mem_ledger.cpu_dispatch_scratch_bytes; give it back with the thread.
    ~cpu_dispatch_buffers() { ledger_sub(scratch_nk_cap_); }

    // Size the small, always-needed buffers. Cheap to repeat: resize to the same size is a no-op.
    void init() {
        src1_q.resize(CPU_DISPATCH_MAX_M * CPU_DISPATCH_MAX_Q_ROW_SIZE);
        // __m256 is 32 bytes = 8 floats; allocate for max chunk4 (256 + max_m) accumulators
        accs.resize((256 + CPU_DISPATCH_MAX_M) * 8);  // Conservative upper bound: 256 stack + max_m heap
    }

    // Return a buffer of at least n_floats floats. The contents are UNSPECIFIED (not zeroed, and not
    // carried over when the buffer grows): the caller must write before it reads. Growth is geometric
    // (at least twice the old capacity, or the request if larger), so a stream of slowly growing
    // requests does not reallocate per call. A pointer from an earlier call is invalid after a call
    // that grows the buffer.
    float * ensure_scratch_nk(size_t n_floats) {
        if (n_floats > scratch_nk_cap_) {
            const size_t new_cap = std::max(n_floats, 2 * scratch_nk_cap_);
            scratch_nk_.reset(new float[new_cap]);  // default-init: no zero fill, no copy of the old contents
            ledger_add(new_cap);
            ledger_sub(scratch_nk_cap_);
            scratch_nk_cap_ = new_cap;
        }
        return scratch_nk_.get();
    }

    size_t scratch_nk_capacity() const { return scratch_nk_cap_; }

    // Test-only: lets test-cpu-dispatch-buffers bound the per-thread footprint.
    size_t resident_bytes() const {
        return src1_q.capacity() * sizeof(uint8_t) + accs.capacity() * sizeof(float) + scratch_nk_cap_ * sizeof(float);
    }

  private:
    static void ledger_add(size_t n_floats) {
        ggml_sycl::host_mem_ledger_get().cpu_dispatch_scratch_bytes.fetch_add(n_floats * sizeof(float),
                                                                              std::memory_order_relaxed);
    }

    static void ledger_sub(size_t n_floats) {
        ggml_sycl::host_mem_ledger_get().cpu_dispatch_scratch_bytes.fetch_sub(n_floats * sizeof(float),
                                                                              std::memory_order_relaxed);
    }

    std::unique_ptr<float[]> scratch_nk_;
    size_t                   scratch_nk_cap_ = 0;
};
