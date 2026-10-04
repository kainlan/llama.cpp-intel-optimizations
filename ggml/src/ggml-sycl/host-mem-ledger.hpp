//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// host-mem-ledger.hpp -- per-consumer host-memory breakdown (llama.cpp-z5fn).
//
// RssAnon cannot say which consumer holds the bytes, and the pinned pool's own
// counters only say how many chunks exist. This ledger counts the bytes the
// backend knowingly puts in host memory per consumer, and
// ggml_sycl_log_host_mem() prints them next to the process counters and the
// pinned pool's zone figures, with an explicit residual, so one WARN line per
// phase can be scored against RssAnon.
//
// WARN, not INFO: GGML_LOG_INFO is dropped at default verbosity in every tool.

#pragma once

#include <atomic>
#include <cstddef>

namespace ggml_sycl {

struct host_mem_ledger {
    // Live SYCL_Host-family buffers (at load: every non-CPU weight, device-tier and host-tier).
    std::atomic<size_t> sycl_host_buffer_bytes{ 0 };
    // Host-tier expert / dense weights copied into a second pinned allocation.
    std::atomic<size_t> host_expert_copy_bytes{ 0 };
    std::atomic<size_t> host_dense_copy_bytes{ 0 };
};

inline host_mem_ledger & host_mem_ledger_get() {
    static host_mem_ledger ledger;
    return ledger;
}

// Parses "<key>:   <n> kB" out of a /proc file's text. Returns bytes, or 0 when absent.
size_t host_mem_proc_kb_bytes(const char * path, const char * key);

}  // namespace ggml_sycl

// Prints one `[HOSTMEM] <phase> ...` WARN line. Safe at any phase; reads only
// process counters and the pinned pool.
void ggml_sycl_log_host_mem(const char * phase);
