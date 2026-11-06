//
// The host bytes configuring a plan's zones commits on a fresh pinned pool, derived from the configure rules rather
// than asked of placement_plan_host_zone_commit_terms(), the function under test (th32 §4.2: host_planned_bytes is
// what the sizing code will commit, not a second estimate of it). Both host ledger tests score against this one
// derivation: the host-only leg on hand-built plans, the lead-run leg on a planned one and the real pool.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include "../unified-cache.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace host_ledger_commit_oracle {

struct derived_commit {
    size_t   weight    = 0;
    size_t   kv        = 0;
    size_t   staging   = 0;
    size_t   scratch   = 0;
    size_t   runtime   = 0;
    uint64_t committed = 0;
};

// The rules ggml_sycl_configure_host_zones_for_plan() and the pinned pool apply: no entries, no zones; the KV zone at
// least 64 MiB; the SCRATCH zone at least the live oneDNN scratch plus the largest tensor plus 32 MiB; the four zones
// each aligned to 64 bytes and grown in whole chunks, each chunk carrying a 64-byte header; the RUNTIME pool's bytes,
// the oneDNN scratchpad and the DMA staging pool and neither PP term, in chunks of their own.
inline derived_commit derive_commit(const ggml_sycl::placement_plan & plan, size_t onednn_live, size_t chunk) {
    constexpr size_t MiB = 1024ull * 1024ull;
    derived_commit   d;
    if (plan.entries.empty() || chunk == 0) {
        return d;
    }
    const auto align64 = [](size_t b) {
        return (b + 63) & ~static_cast<size_t>(63);
    };
    d.weight  = plan.host_zone_weight_bytes;
    d.kv      = std::max(plan.host_zone_kv_bytes, 64 * MiB);
    d.staging = plan.host_zone_staging_bytes;
    d.scratch = std::max(plan.host_zone_scratch_bytes, onednn_live + plan.max_tensor_bytes + 32 * MiB);
    d.runtime = plan.onednn_scratchpad_bytes + plan.dma_staging_pool_bytes;

    const uint64_t zones = align64(d.weight) + align64(d.kv) + align64(d.staging) + align64(d.scratch);
    d.committed          = ((zones + chunk - 1) / chunk) * (chunk + 64);
    if (d.runtime > 0) {
        d.committed += ((d.runtime + chunk - 1) / chunk) * (chunk + 64);
    }
    return d;
}

}  // namespace host_ledger_commit_oracle
