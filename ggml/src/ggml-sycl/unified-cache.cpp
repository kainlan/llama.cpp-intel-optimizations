//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "unified-cache.hpp"

#include "alloc-registry.hpp"
#include "common.hpp"
#include "expert-prefetch.hpp"
#include "ggml-impl.h"
#include "ggml-sycl.h"
#include "kv-tier-manager.hpp"
#include "mem-handle.hpp"
#include "mem-ops.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <thread>
#include <unordered_set>

#if defined(_WIN32)
#    include <windows.h>
#else
#    include <unistd.h>
#endif

namespace ggml_sycl {

// Per-device cache storage (for PER_DEVICE and AUTO modes)
static std::unordered_map<int, std::unique_ptr<unified_cache>> g_device_caches;
static std::shared_mutex                                       g_cache_rw_mutex;
static size_t                                                  g_unified_cache_budget      = 0;  // 0 = auto-calculate
static int                                                     g_unified_cache_budget_pct  = 100;
static size_t                                                  g_unified_cache_host_budget = 0;  // 0 = auto-calc
static int                                                     g_unified_cache_host_budget_pct = 90;
static unified_cache_mode                                      g_cache_mode             = unified_cache_mode::AUTO;
static int                                                     g_scheduler_device_count = -1;
static int                                                     g_total_gpu_count        = -1;
static std::atomic<bool>   g_cache_mode_locked{ false };   // Locked after first cache access
static std::atomic<bool>   g_sycl_shutting_down{ false };  // Set during shutdown to skip sycl::free()
// VRAM runtime counters removed — arena zones (zone_used) are the single source of truth.
// Host runtime counters retained until host-pool zone tracking is implemented.
static std::atomic<size_t> g_runtime_reserved_host_bytes{};
static std::atomic<size_t> g_runtime_host_cat_bytes[static_cast<int>(runtime_category::COUNT)]{};
static std::atomic<size_t> g_runtime_managed_reserved_host_bytes{};
static std::atomic<size_t> g_planned_pp_pipeline_scratch_bytes[GGML_SYCL_MAX_DEVICES]{};
static std::atomic<bool>   g_atexit_registered{ false };  // Ensure atexit handler registered once
static std::atomic<int>    g_cache_assert_enabled{ -1 };
static std::atomic<int>    g_copy_trace_enabled{ -1 };
static std::atomic<bool>   g_graph_compute_active{ false };

static std::mutex            g_runtime_alloc_mutex;
static std::atomic<uint64_t> g_runtime_alloc_id{ 1 };

struct runtime_alloc_record {
    alloc_handle  handle{};
    sycl::queue * queue            = nullptr;
    bool          uses_pinned_pool = false;
    bool          zone_managed     = false;
    bool          from_arena       = false;                // True if sub-allocated from arena (KV/RUNTIME/etc zone)
    vram_zone_id  vram_zone        = vram_zone_id::COUNT;  // Non-COUNT: zone_free on unified_free
    std::string   cohort_id;
};

static std::unordered_map<void *, runtime_alloc_record> g_runtime_alloc_registry;
static std::unordered_map<std::string, alloc_tier>      g_runtime_cohort_tier;
static std::mutex                                       g_offload_pool_mutex;
static std::atomic<uint64_t>                            g_offload_pool_lease_id{ 1 };

constexpr size_t k_device_alloc_alignment = 2ull * 1024ull * 1024ull;

static void * sycl_aligned_malloc_device(size_t size, const sycl::queue & queue) {
    return sycl::aligned_alloc_device(k_device_alloc_alignment, size, queue);
}

struct offload_pool_key {
    int                 device    = -1;
    offload_buffer_role role      = offload_buffer_role::OTHER;
    alloc_tier          tier      = alloc_tier::HOST_PINNED;
    runtime_category    category  = runtime_category::OTHER;
    size_t              alignment = 64;
};

struct offload_pool_key_hash {
    size_t operator()(const offload_pool_key & key) const {
        size_t h = 0;
        h        = detail::cache_hash_combine(h, std::hash<int>()(key.device));
        h        = detail::cache_hash_combine(h, std::hash<int>()(static_cast<int>(key.role)));
        h        = detail::cache_hash_combine(h, std::hash<int>()(static_cast<int>(key.tier)));
        h        = detail::cache_hash_combine(h, std::hash<int>()(static_cast<int>(key.category)));
        h        = detail::cache_hash_combine(h, std::hash<size_t>()(key.alignment));
        return h;
    }
};

static bool operator==(const offload_pool_key & a, const offload_pool_key & b) {
    return a.device == b.device && a.role == b.role && a.tier == b.tier && a.category == b.category &&
           a.alignment == b.alignment;
}

struct offload_pool_slot {
    alloc_handle     handle{};
    offload_pool_key key{};
    bool             in_use   = false;
    uint64_t         lease_id = 0;
};

static std::unordered_map<void *, offload_pool_slot>                                    g_offload_pool_slots;
static std::unordered_map<offload_pool_key, std::vector<void *>, offload_pool_key_hash> g_offload_pool_free;

static std::atomic<uint64_t> g_offload_wait_count{ 0 };
static std::atomic<uint64_t> g_offload_wait_count_forced{ 0 };
static std::atomic<uint64_t> g_offload_wait_count_fallback{ 0 };
static std::atomic<uint64_t> g_offload_alloc_count_host{ 0 };
static std::atomic<uint64_t> g_offload_alloc_count_device{ 0 };
static std::atomic<uint64_t> g_offload_alloc_count_shared{ 0 };
static std::atomic<uint64_t> g_offload_pool_hit_count{ 0 };
static std::atomic<uint64_t> g_offload_pool_miss_count{ 0 };
static std::atomic<uint64_t> g_offload_cross_domain_transfer_count{ 0 };
static std::atomic<uint64_t> g_offload_cross_domain_transfer_count_pp{ 0 };
static std::atomic<uint64_t> g_offload_cross_domain_transfer_count_tg{ 0 };
static std::atomic<uint64_t> g_offload_transfer_bytes_h2d{ 0 };
static std::atomic<uint64_t> g_offload_transfer_bytes_d2h{ 0 };

void unified_cache_set_planned_pp_pipeline_scratch_bytes(int device_id, size_t bytes) {
    if (device_id < 0 || device_id >= GGML_SYCL_MAX_DEVICES) {
        return;
    }
    g_planned_pp_pipeline_scratch_bytes[device_id].store(bytes, std::memory_order_release);
}

size_t unified_cache_get_planned_pp_pipeline_scratch_bytes(int device_id) {
    if (device_id < 0 || device_id >= GGML_SYCL_MAX_DEVICES) {
        return 0;
    }
    return g_planned_pp_pipeline_scratch_bytes[device_id].load(std::memory_order_acquire);
}

static std::atomic<uint64_t> g_offload_transfer_bytes_h2d_pp{ 0 };
static std::atomic<uint64_t> g_offload_transfer_bytes_h2d_tg{ 0 };
static std::atomic<uint64_t> g_offload_transfer_bytes_d2h_pp{ 0 };
static std::atomic<uint64_t> g_offload_transfer_bytes_d2h_tg{ 0 };
static std::atomic<uint64_t> g_offload_dispatch_count_cpu{ 0 };
static std::atomic<uint64_t> g_offload_dispatch_count_gpu{ 0 };
static std::atomic<uint64_t> g_offload_dispatch_count_gpu_island{ 0 };
static std::atomic<uint64_t> g_offload_dispatch_count_cpu_pp{ 0 };
static std::atomic<uint64_t> g_offload_dispatch_count_cpu_tg{ 0 };
static std::atomic<uint64_t> g_offload_dispatch_count_gpu_pp{ 0 };
static std::atomic<uint64_t> g_offload_dispatch_count_gpu_tg{ 0 };
static std::atomic<uint64_t> g_offload_dispatch_count_gpu_island_pp{ 0 };
static std::atomic<uint64_t> g_offload_dispatch_count_gpu_island_tg{ 0 };
static std::atomic<uint64_t> g_offload_transition_wait_count{ 0 };
static std::atomic<uint64_t> g_offload_transition_wait_count_pp{ 0 };
static std::atomic<uint64_t> g_offload_transition_wait_count_tg{ 0 };
static std::atomic<uint64_t> g_offload_transition_wait_elided_count{ 0 };
static std::atomic<uint64_t> g_offload_transition_wait_elided_count_pp{ 0 };
static std::atomic<uint64_t> g_offload_transition_wait_elided_count_tg{ 0 };
static std::atomic<uint64_t> g_offload_host_alloc_call_count{ 0 };
static std::atomic<uint64_t> g_offload_host_alloc_bytes{ 0 };
static std::atomic<uint64_t> g_offload_host_alloc_calls_unified_alloc_host{ 0 };
static std::atomic<uint64_t> g_offload_host_alloc_bytes_unified_alloc_host{ 0 };
static std::atomic<uint64_t> g_offload_host_alloc_calls_unified_cache_host_chunk{ 0 };
static std::atomic<uint64_t> g_offload_host_alloc_bytes_unified_cache_host_chunk{ 0 };
static std::atomic<uint64_t> g_offload_host_alloc_calls_host_malloc{ 0 };
static std::atomic<uint64_t> g_offload_host_alloc_bytes_host_malloc{ 0 };
static std::atomic<uint64_t> g_offload_host_alloc_calls_other{ 0 };
static std::atomic<uint64_t> g_offload_host_alloc_bytes_other{ 0 };
static std::mutex            g_offload_host_alloc_by_tag_mutex;
static std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> g_offload_host_alloc_by_tag;
static std::atomic<int> g_offload_phase{ static_cast<int>(offload_phase::UNKNOWN) };

static int get_device_id_from_queue(sycl::queue & queue);

static const char * alloc_tier_name(alloc_tier tier) {
    switch (tier) {
        case alloc_tier::DEVICE_VRAM:
            return "device_vram";
        case alloc_tier::HOST_PINNED:
            return "host_pinned";
        case alloc_tier::MMAP_TRACKED:
            return "mmap_tracked";
        default:
            return "unknown";
    }
}

static inline size_t align_up(size_t value, size_t alignment) {
    if (alignment <= 1) {
        return value;
    }
    return (value + alignment - 1) / alignment * alignment;
}

static runtime_category category_from_role(alloc_role role) {
    switch (role) {
        case alloc_role::KV:
            return runtime_category::KV_CACHE;
        case alloc_role::COMPUTE:
            return runtime_category::COMPUTE;
        case alloc_role::STAGING:
            return runtime_category::STAGING;
        case alloc_role::GRAPH_TMP:
            return runtime_category::GRAPH;
        case alloc_role::TP_TMP:
            return runtime_category::GRAPH;
        case alloc_role::EXPERT_STAGING:
            return runtime_category::EXPERT_CACHE;
        case alloc_role::CONTROL:
            return runtime_category::CONTROL;
        case alloc_role::WEIGHT:
        case alloc_role::OTHER:
        default:
            return runtime_category::OTHER;
    }
}

// unified_managed_add/sub_device_bytes removed — arena zones are the single source of truth for VRAM.
// Host managed bytes tracking retained until host-pool zone tracking is implemented.

static inline void unified_managed_add_host_bytes(size_t bytes) {
    if (bytes == 0) {
        return;
    }
    g_runtime_managed_reserved_host_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

static inline void unified_managed_sub_host_bytes(size_t bytes) {
    if (bytes == 0) {
        return;
    }
    g_runtime_managed_reserved_host_bytes.fetch_sub(bytes, std::memory_order_relaxed);
}

bool unified_alloc_strict_mode() {
    static std::atomic<int> s_strict{ -1 };
    int                     cached = s_strict.load(std::memory_order_acquire);
    if (cached >= 0) {
        return cached != 0;
    }
    const char * env    = std::getenv("GGML_SYCL_UNIFIED_ALLOC_STRICT");
    const int    strict = (env && std::atoi(env) != 0) ? 1 : 0;
    s_strict.store(strict, std::memory_order_release);
    return strict != 0;
}

bool offload_stats_enabled() {
    const char * env = std::getenv("GGML_SYCL_OFFLOAD_STATS");
    return env && std::atoi(env) != 0;
}

void offload_stats_reset() {
    g_offload_wait_count.store(0, std::memory_order_relaxed);
    g_offload_wait_count_forced.store(0, std::memory_order_relaxed);
    g_offload_wait_count_fallback.store(0, std::memory_order_relaxed);
    g_offload_alloc_count_host.store(0, std::memory_order_relaxed);
    g_offload_alloc_count_device.store(0, std::memory_order_relaxed);
    g_offload_alloc_count_shared.store(0, std::memory_order_relaxed);
    g_offload_pool_hit_count.store(0, std::memory_order_relaxed);
    g_offload_pool_miss_count.store(0, std::memory_order_relaxed);
    g_offload_cross_domain_transfer_count.store(0, std::memory_order_relaxed);
    g_offload_cross_domain_transfer_count_pp.store(0, std::memory_order_relaxed);
    g_offload_cross_domain_transfer_count_tg.store(0, std::memory_order_relaxed);
    g_offload_transfer_bytes_h2d.store(0, std::memory_order_relaxed);
    g_offload_transfer_bytes_d2h.store(0, std::memory_order_relaxed);
    g_offload_transfer_bytes_h2d_pp.store(0, std::memory_order_relaxed);
    g_offload_transfer_bytes_h2d_tg.store(0, std::memory_order_relaxed);
    g_offload_transfer_bytes_d2h_pp.store(0, std::memory_order_relaxed);
    g_offload_transfer_bytes_d2h_tg.store(0, std::memory_order_relaxed);
    g_offload_dispatch_count_cpu.store(0, std::memory_order_relaxed);
    g_offload_dispatch_count_gpu.store(0, std::memory_order_relaxed);
    g_offload_dispatch_count_gpu_island.store(0, std::memory_order_relaxed);
    g_offload_dispatch_count_cpu_pp.store(0, std::memory_order_relaxed);
    g_offload_dispatch_count_cpu_tg.store(0, std::memory_order_relaxed);
    g_offload_dispatch_count_gpu_pp.store(0, std::memory_order_relaxed);
    g_offload_dispatch_count_gpu_tg.store(0, std::memory_order_relaxed);
    g_offload_dispatch_count_gpu_island_pp.store(0, std::memory_order_relaxed);
    g_offload_dispatch_count_gpu_island_tg.store(0, std::memory_order_relaxed);
    g_offload_transition_wait_count.store(0, std::memory_order_relaxed);
    g_offload_transition_wait_count_pp.store(0, std::memory_order_relaxed);
    g_offload_transition_wait_count_tg.store(0, std::memory_order_relaxed);
    g_offload_transition_wait_elided_count.store(0, std::memory_order_relaxed);
    g_offload_transition_wait_elided_count_pp.store(0, std::memory_order_relaxed);
    g_offload_transition_wait_elided_count_tg.store(0, std::memory_order_relaxed);
    g_offload_host_alloc_call_count.store(0, std::memory_order_relaxed);
    g_offload_host_alloc_bytes.store(0, std::memory_order_relaxed);
    g_offload_host_alloc_calls_unified_alloc_host.store(0, std::memory_order_relaxed);
    g_offload_host_alloc_bytes_unified_alloc_host.store(0, std::memory_order_relaxed);
    g_offload_host_alloc_calls_unified_cache_host_chunk.store(0, std::memory_order_relaxed);
    g_offload_host_alloc_bytes_unified_cache_host_chunk.store(0, std::memory_order_relaxed);
    g_offload_host_alloc_calls_host_malloc.store(0, std::memory_order_relaxed);
    g_offload_host_alloc_bytes_host_malloc.store(0, std::memory_order_relaxed);
    g_offload_host_alloc_calls_other.store(0, std::memory_order_relaxed);
    g_offload_host_alloc_bytes_other.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(g_offload_host_alloc_by_tag_mutex);
        g_offload_host_alloc_by_tag.clear();
    }
    g_offload_phase.store(static_cast<int>(offload_phase::UNKNOWN), std::memory_order_relaxed);
}

void offload_stats_set_phase(offload_phase phase) {
    g_offload_phase.store(static_cast<int>(phase), std::memory_order_relaxed);
}

const char * offload_phase_name(offload_phase phase) {
    switch (phase) {
        case offload_phase::LOAD:
            return "load";
        case offload_phase::WARMUP:
            return "warmup";
        case offload_phase::PP:
            return "pp";
        case offload_phase::TG:
            return "tg";
        default:
            return "unknown";
    }
}

offload_phase offload_stats_phase() {
    const int phase = g_offload_phase.load(std::memory_order_relaxed);
    switch (phase) {
        case static_cast<int>(offload_phase::LOAD):
            return offload_phase::LOAD;
        case static_cast<int>(offload_phase::WARMUP):
            return offload_phase::WARMUP;
        case static_cast<int>(offload_phase::PP):
            return offload_phase::PP;
        case static_cast<int>(offload_phase::TG):
            return offload_phase::TG;
        default:
            return offload_phase::UNKNOWN;
    }
}

static inline offload_phase offload_stats_current_phase() {
    const int phase = g_offload_phase.load(std::memory_order_relaxed);
    switch (phase) {
        case static_cast<int>(offload_phase::LOAD):
            return offload_phase::LOAD;
        case static_cast<int>(offload_phase::WARMUP):
            return offload_phase::WARMUP;
        case static_cast<int>(offload_phase::PP):
            return offload_phase::PP;
        case static_cast<int>(offload_phase::TG):
            return offload_phase::TG;
        default:
            return offload_phase::UNKNOWN;
    }
}

void offload_stats_note_wait(bool fallback) {
    g_offload_wait_count.fetch_add(1, std::memory_order_relaxed);
    if (fallback) {
        g_offload_wait_count_fallback.fetch_add(1, std::memory_order_relaxed);
    } else {
        g_offload_wait_count_forced.fetch_add(1, std::memory_order_relaxed);
    }
}

void offload_stats_note_alloc(alloc_tier tier) {
    switch (tier) {
        case alloc_tier::DEVICE_VRAM:
            g_offload_alloc_count_device.fetch_add(1, std::memory_order_relaxed);
            break;
        case alloc_tier::HOST_PINNED:
            g_offload_alloc_count_host.fetch_add(1, std::memory_order_relaxed);
            break;
        case alloc_tier::MMAP_TRACKED:
            g_offload_alloc_count_shared.fetch_add(1, std::memory_order_relaxed);
            break;
        default:
            break;
    }
}

void offload_stats_note_pool_hit() {
    g_offload_pool_hit_count.fetch_add(1, std::memory_order_relaxed);
}

void offload_stats_note_pool_miss() {
    g_offload_pool_miss_count.fetch_add(1, std::memory_order_relaxed);
}

void offload_stats_note_transfer(bool h2d, size_t bytes) {
    const offload_phase phase = offload_stats_current_phase();
    if (h2d) {
        g_offload_transfer_bytes_h2d.fetch_add(bytes, std::memory_order_relaxed);
        if (phase == offload_phase::PP) {
            g_offload_transfer_bytes_h2d_pp.fetch_add(bytes, std::memory_order_relaxed);
        } else if (phase == offload_phase::TG) {
            g_offload_transfer_bytes_h2d_tg.fetch_add(bytes, std::memory_order_relaxed);
        }
    } else {
        g_offload_transfer_bytes_d2h.fetch_add(bytes, std::memory_order_relaxed);
        if (phase == offload_phase::PP) {
            g_offload_transfer_bytes_d2h_pp.fetch_add(bytes, std::memory_order_relaxed);
        } else if (phase == offload_phase::TG) {
            g_offload_transfer_bytes_d2h_tg.fetch_add(bytes, std::memory_order_relaxed);
        }
    }
}

void offload_stats_note_cross_domain_transfer(size_t bytes) {
    GGML_UNUSED(bytes);
    const offload_phase phase = offload_stats_current_phase();
    g_offload_cross_domain_transfer_count.fetch_add(1, std::memory_order_relaxed);
    if (phase == offload_phase::PP) {
        g_offload_cross_domain_transfer_count_pp.fetch_add(1, std::memory_order_relaxed);
    } else if (phase == offload_phase::TG) {
        g_offload_cross_domain_transfer_count_tg.fetch_add(1, std::memory_order_relaxed);
    }
}

void offload_stats_note_dispatch(bool cpu, bool gpu_island) {
    const offload_phase phase = offload_stats_current_phase();
    if (cpu) {
        g_offload_dispatch_count_cpu.fetch_add(1, std::memory_order_relaxed);
        if (phase == offload_phase::PP) {
            g_offload_dispatch_count_cpu_pp.fetch_add(1, std::memory_order_relaxed);
        } else if (phase == offload_phase::TG) {
            g_offload_dispatch_count_cpu_tg.fetch_add(1, std::memory_order_relaxed);
        }
        return;
    }

    g_offload_dispatch_count_gpu.fetch_add(1, std::memory_order_relaxed);
    if (phase == offload_phase::PP) {
        g_offload_dispatch_count_gpu_pp.fetch_add(1, std::memory_order_relaxed);
    } else if (phase == offload_phase::TG) {
        g_offload_dispatch_count_gpu_tg.fetch_add(1, std::memory_order_relaxed);
    }

    if (gpu_island) {
        g_offload_dispatch_count_gpu_island.fetch_add(1, std::memory_order_relaxed);
        if (phase == offload_phase::PP) {
            g_offload_dispatch_count_gpu_island_pp.fetch_add(1, std::memory_order_relaxed);
        } else if (phase == offload_phase::TG) {
            g_offload_dispatch_count_gpu_island_tg.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

void offload_stats_note_transition_wait(bool waited) {
    const offload_phase phase = offload_stats_current_phase();
    if (waited) {
        g_offload_transition_wait_count.fetch_add(1, std::memory_order_relaxed);
        if (phase == offload_phase::PP) {
            g_offload_transition_wait_count_pp.fetch_add(1, std::memory_order_relaxed);
        } else if (phase == offload_phase::TG) {
            g_offload_transition_wait_count_tg.fetch_add(1, std::memory_order_relaxed);
        }
        return;
    }

    g_offload_transition_wait_elided_count.fetch_add(1, std::memory_order_relaxed);
    if (phase == offload_phase::PP) {
        g_offload_transition_wait_elided_count_pp.fetch_add(1, std::memory_order_relaxed);
    } else if (phase == offload_phase::TG) {
        g_offload_transition_wait_elided_count_tg.fetch_add(1, std::memory_order_relaxed);
    }
}

void offload_stats_note_host_alloc(const char * tag, size_t bytes) {
    if (bytes == 0) {
        return;
    }
    g_offload_host_alloc_call_count.fetch_add(1, std::memory_order_relaxed);
    g_offload_host_alloc_bytes.fetch_add(bytes, std::memory_order_relaxed);

    const std::string tag_s = tag && tag[0] != '\0' ? std::string(tag) : std::string("(unknown)");
    {
        std::lock_guard<std::mutex> lock(g_offload_host_alloc_by_tag_mutex);
        auto &                      entry = g_offload_host_alloc_by_tag[tag_s];
        entry.first += 1;
        entry.second += bytes;
    }

    if (tag_s == "unified_alloc:host") {
        g_offload_host_alloc_calls_unified_alloc_host.fetch_add(1, std::memory_order_relaxed);
        g_offload_host_alloc_bytes_unified_alloc_host.fetch_add(bytes, std::memory_order_relaxed);
        return;
    }
    if (tag_s.find("unified_cache:host_chunk") != std::string::npos ||
        tag_s.find("unified_cache:host_temp") != std::string::npos) {
        g_offload_host_alloc_calls_unified_cache_host_chunk.fetch_add(1, std::memory_order_relaxed);
        g_offload_host_alloc_bytes_unified_cache_host_chunk.fetch_add(bytes, std::memory_order_relaxed);
        return;
    }
    if (tag_s == "host_malloc" || tag_s == "tp_shared_host") {
        g_offload_host_alloc_calls_host_malloc.fetch_add(1, std::memory_order_relaxed);
        g_offload_host_alloc_bytes_host_malloc.fetch_add(bytes, std::memory_order_relaxed);
        return;
    }
    g_offload_host_alloc_calls_other.fetch_add(1, std::memory_order_relaxed);
    g_offload_host_alloc_bytes_other.fetch_add(bytes, std::memory_order_relaxed);
}

offload_stats_snapshot offload_stats_get() {
    offload_stats_snapshot s{};
    s.wait_count                      = g_offload_wait_count.load(std::memory_order_relaxed);
    s.wait_count_forced               = g_offload_wait_count_forced.load(std::memory_order_relaxed);
    s.wait_count_fallback             = g_offload_wait_count_fallback.load(std::memory_order_relaxed);
    s.alloc_count_host                = g_offload_alloc_count_host.load(std::memory_order_relaxed);
    s.alloc_count_device              = g_offload_alloc_count_device.load(std::memory_order_relaxed);
    s.alloc_count_shared              = g_offload_alloc_count_shared.load(std::memory_order_relaxed);
    s.pool_hit_count                  = g_offload_pool_hit_count.load(std::memory_order_relaxed);
    s.pool_miss_count                 = g_offload_pool_miss_count.load(std::memory_order_relaxed);
    s.cross_domain_transfer_count     = g_offload_cross_domain_transfer_count.load(std::memory_order_relaxed);
    s.cross_domain_transfer_count_pp  = g_offload_cross_domain_transfer_count_pp.load(std::memory_order_relaxed);
    s.cross_domain_transfer_count_tg  = g_offload_cross_domain_transfer_count_tg.load(std::memory_order_relaxed);
    s.transfer_bytes_h2d              = g_offload_transfer_bytes_h2d.load(std::memory_order_relaxed);
    s.transfer_bytes_d2h              = g_offload_transfer_bytes_d2h.load(std::memory_order_relaxed);
    s.transfer_bytes_h2d_pp           = g_offload_transfer_bytes_h2d_pp.load(std::memory_order_relaxed);
    s.transfer_bytes_h2d_tg           = g_offload_transfer_bytes_h2d_tg.load(std::memory_order_relaxed);
    s.transfer_bytes_d2h_pp           = g_offload_transfer_bytes_d2h_pp.load(std::memory_order_relaxed);
    s.transfer_bytes_d2h_tg           = g_offload_transfer_bytes_d2h_tg.load(std::memory_order_relaxed);
    s.dispatch_count_cpu              = g_offload_dispatch_count_cpu.load(std::memory_order_relaxed);
    s.dispatch_count_gpu              = g_offload_dispatch_count_gpu.load(std::memory_order_relaxed);
    s.dispatch_count_gpu_island       = g_offload_dispatch_count_gpu_island.load(std::memory_order_relaxed);
    s.dispatch_count_cpu_pp           = g_offload_dispatch_count_cpu_pp.load(std::memory_order_relaxed);
    s.dispatch_count_cpu_tg           = g_offload_dispatch_count_cpu_tg.load(std::memory_order_relaxed);
    s.dispatch_count_gpu_pp           = g_offload_dispatch_count_gpu_pp.load(std::memory_order_relaxed);
    s.dispatch_count_gpu_tg           = g_offload_dispatch_count_gpu_tg.load(std::memory_order_relaxed);
    s.dispatch_count_gpu_island_pp    = g_offload_dispatch_count_gpu_island_pp.load(std::memory_order_relaxed);
    s.dispatch_count_gpu_island_tg    = g_offload_dispatch_count_gpu_island_tg.load(std::memory_order_relaxed);
    s.transition_wait_count           = g_offload_transition_wait_count.load(std::memory_order_relaxed);
    s.transition_wait_count_pp        = g_offload_transition_wait_count_pp.load(std::memory_order_relaxed);
    s.transition_wait_count_tg        = g_offload_transition_wait_count_tg.load(std::memory_order_relaxed);
    s.transition_wait_elided_count    = g_offload_transition_wait_elided_count.load(std::memory_order_relaxed);
    s.transition_wait_elided_count_pp = g_offload_transition_wait_elided_count_pp.load(std::memory_order_relaxed);
    s.transition_wait_elided_count_tg = g_offload_transition_wait_elided_count_tg.load(std::memory_order_relaxed);
    s.host_alloc_call_count           = g_offload_host_alloc_call_count.load(std::memory_order_relaxed);
    s.host_alloc_bytes                = g_offload_host_alloc_bytes.load(std::memory_order_relaxed);
    s.host_alloc_calls_unified_alloc_host =
        g_offload_host_alloc_calls_unified_alloc_host.load(std::memory_order_relaxed);
    s.host_alloc_bytes_unified_alloc_host =
        g_offload_host_alloc_bytes_unified_alloc_host.load(std::memory_order_relaxed);
    s.host_alloc_calls_unified_cache_host_chunk =
        g_offload_host_alloc_calls_unified_cache_host_chunk.load(std::memory_order_relaxed);
    s.host_alloc_bytes_unified_cache_host_chunk =
        g_offload_host_alloc_bytes_unified_cache_host_chunk.load(std::memory_order_relaxed);
    s.host_alloc_calls_host_malloc = g_offload_host_alloc_calls_host_malloc.load(std::memory_order_relaxed);
    s.host_alloc_bytes_host_malloc = g_offload_host_alloc_bytes_host_malloc.load(std::memory_order_relaxed);
    s.host_alloc_calls_other       = g_offload_host_alloc_calls_other.load(std::memory_order_relaxed);
    s.host_alloc_bytes_other       = g_offload_host_alloc_bytes_other.load(std::memory_order_relaxed);
    return s;
}

static std::vector<std::pair<std::string, std::pair<uint64_t, uint64_t>>> offload_host_alloc_top_by_calls(
    size_t top_n) {
    std::vector<std::pair<std::string, std::pair<uint64_t, uint64_t>>> rows;
    {
        std::lock_guard<std::mutex> lock(g_offload_host_alloc_by_tag_mutex);
        rows.reserve(g_offload_host_alloc_by_tag.size());
        for (const auto & it : g_offload_host_alloc_by_tag) {
            rows.push_back(it);
        }
    }
    std::sort(rows.begin(), rows.end(), [](const auto & a, const auto & b) {
        if (a.second.first != b.second.first) {
            return a.second.first > b.second.first;
        }
        return a.second.second > b.second.second;
    });
    if (rows.size() > top_n) {
        rows.resize(top_n);
    }
    return rows;
}

// Standalone zero-alloc check — runs even when offload stats are disabled.
// GGML_SYCL_ZERO_ALLOC_CHECK: 0=off, 1=warn (default), 2=abort
void zero_alloc_check(const char * tag, int device) {
    static const int zero_alloc_mode = []() {
        const char * env = std::getenv("GGML_SYCL_ZERO_ALLOC_CHECK");
        if (env) {
            return std::atoi(env);
        }
        return 1;  // Default ON (warn mode)
    }();
    if (zero_alloc_mode <= 0) {
        return;
    }
    const offload_phase phase = offload_stats_current_phase();
    if (phase != offload_phase::PP && phase != offload_phase::TG) {
        return;
    }

    const size_t runtime_bytes = unified_cache_get_runtime_bytes(device);

    // Baseline snapshot: captured once on first PP/TG check.
    // All pre-existing runtime allocations (KV cache, graph buffers, etc.)
    // are expected and should not trigger warnings.
    static std::atomic<size_t> s_baseline{ 0 };
    static std::atomic<bool>   s_baseline_set{ false };
    if (!s_baseline_set.load(std::memory_order_relaxed)) {
        s_baseline.store(runtime_bytes, std::memory_order_relaxed);
        s_baseline_set.store(true, std::memory_order_relaxed);
        return;  // First call — just record baseline
    }

    const size_t baseline = s_baseline.load(std::memory_order_relaxed);
    if (runtime_bytes > baseline) {
        const size_t delta      = runtime_bytes - baseline;
        const char * phase_name = offload_phase_name(phase);
        GGML_LOG_WARN(
            "[SYCL-ZERO-ALLOC-CHECK] %s: runtime allocation detected during %s phase: +%.1f MB "
            "(total %.1f MB, baseline %.1f MB). "
            "Expected zero new allocations during steady-state inference.\n",
            tag ? tag : "graph", phase_name, delta / (1024.0 * 1024.0), runtime_bytes / (1024.0 * 1024.0),
            baseline / (1024.0 * 1024.0));
        if (zero_alloc_mode >= 2) {
            GGML_ASSERT(false && "ZERO_ALLOC_CHECK: runtime allocation during inference");
        }
    }
}

void offload_stats_log_summary(const char * tag, int device) {
    if (!offload_stats_enabled()) {
        return;
    }
    const offload_stats_snapshot s          = offload_stats_get();
    const offload_phase          phase      = offload_stats_current_phase();
    const char *                 phase_name = offload_phase_name(phase);

    fprintf(
        stderr,
        "[SYCL-OFFLOAD-STATS] tag=%s device=%d wait_count=%llu wait_count_forced=%llu "
        "wait_count_fallback=%llu alloc_count_host=%llu alloc_count_device=%llu "
        "alloc_count_shared=%llu pool_hit_count=%llu pool_miss_count=%llu phase=%s "
        "cross_domain_transfer_count=%llu cross_domain_transfer_count_pp=%llu cross_domain_transfer_count_tg=%llu "
        "transfer_bytes_h2d=%llu transfer_bytes_h2d_pp=%llu transfer_bytes_h2d_tg=%llu "
        "transfer_bytes_d2h=%llu transfer_bytes_d2h_pp=%llu transfer_bytes_d2h_tg=%llu "
        "dispatch_count_cpu=%llu dispatch_count_gpu=%llu dispatch_count_gpu_island=%llu "
        "dispatch_count_cpu_pp=%llu dispatch_count_cpu_tg=%llu "
        "dispatch_count_gpu_pp=%llu dispatch_count_gpu_tg=%llu "
        "dispatch_count_gpu_island_pp=%llu dispatch_count_gpu_island_tg=%llu "
        "transition_wait_count=%llu transition_wait_count_pp=%llu transition_wait_count_tg=%llu "
        "transition_wait_elided_count=%llu transition_wait_elided_count_pp=%llu transition_wait_elided_count_tg=%llu "
        "host_alloc_call_count=%llu host_alloc_bytes=%llu "
        "host_alloc_calls_unified_alloc_host=%llu host_alloc_bytes_unified_alloc_host=%llu "
        "host_alloc_calls_unified_cache_host_chunk=%llu host_alloc_bytes_unified_cache_host_chunk=%llu "
        "host_alloc_calls_host_malloc=%llu host_alloc_bytes_host_malloc=%llu "
        "host_alloc_calls_other=%llu host_alloc_bytes_other=%llu\n",
        tag ? tag : "graph", device, (unsigned long long) s.wait_count, (unsigned long long) s.wait_count_forced,
        (unsigned long long) s.wait_count_fallback, (unsigned long long) s.alloc_count_host,
        (unsigned long long) s.alloc_count_device, (unsigned long long) s.alloc_count_shared,
        (unsigned long long) s.pool_hit_count, (unsigned long long) s.pool_miss_count, phase_name,
        (unsigned long long) s.cross_domain_transfer_count, (unsigned long long) s.cross_domain_transfer_count_pp,
        (unsigned long long) s.cross_domain_transfer_count_tg, (unsigned long long) s.transfer_bytes_h2d,
        (unsigned long long) s.transfer_bytes_h2d_pp, (unsigned long long) s.transfer_bytes_h2d_tg,
        (unsigned long long) s.transfer_bytes_d2h, (unsigned long long) s.transfer_bytes_d2h_pp,
        (unsigned long long) s.transfer_bytes_d2h_tg, (unsigned long long) s.dispatch_count_cpu,
        (unsigned long long) s.dispatch_count_gpu, (unsigned long long) s.dispatch_count_gpu_island,
        (unsigned long long) s.dispatch_count_cpu_pp, (unsigned long long) s.dispatch_count_cpu_tg,
        (unsigned long long) s.dispatch_count_gpu_pp, (unsigned long long) s.dispatch_count_gpu_tg,
        (unsigned long long) s.dispatch_count_gpu_island_pp, (unsigned long long) s.dispatch_count_gpu_island_tg,
        (unsigned long long) s.transition_wait_count, (unsigned long long) s.transition_wait_count_pp,
        (unsigned long long) s.transition_wait_count_tg, (unsigned long long) s.transition_wait_elided_count,
        (unsigned long long) s.transition_wait_elided_count_pp, (unsigned long long) s.transition_wait_elided_count_tg,
        (unsigned long long) s.host_alloc_call_count, (unsigned long long) s.host_alloc_bytes,
        (unsigned long long) s.host_alloc_calls_unified_alloc_host,
        (unsigned long long) s.host_alloc_bytes_unified_alloc_host,
        (unsigned long long) s.host_alloc_calls_unified_cache_host_chunk,
        (unsigned long long) s.host_alloc_bytes_unified_cache_host_chunk,
        (unsigned long long) s.host_alloc_calls_host_malloc, (unsigned long long) s.host_alloc_bytes_host_malloc,
        (unsigned long long) s.host_alloc_calls_other, (unsigned long long) s.host_alloc_bytes_other);

    int top_n = 5;
    if (const char * env = std::getenv("GGML_SYCL_HOST_ALLOC_TOP")) {
        top_n = std::max(1, std::atoi(env));
    }
    const auto rows = offload_host_alloc_top_by_calls(static_cast<size_t>(top_n));
    if (!rows.empty()) {
        fprintf(stderr, "[SYCL-OFFLOAD-HOST-ALLOC] tag=%s device=%d top=%d", tag ? tag : "graph", device, top_n);
        for (const auto & row : rows) {
            fprintf(stderr, " [%s calls=%llu bytes=%llu]", row.first.c_str(), (unsigned long long) row.second.first,
                    (unsigned long long) row.second.second);
        }
        fprintf(stderr, "\n");
    }

    long warn_calls = 0;
    if (const char * env = std::getenv("GGML_SYCL_HOST_ALLOC_WARN_CALLS")) {
        warn_calls = std::strtol(env, nullptr, 10);
    }
    if (warn_calls > 0 && s.host_alloc_call_count > static_cast<uint64_t>(warn_calls)) {
        fprintf(stderr,
                "[SYCL-OFFLOAD-HOST-ALLOC] WARN tag=%s device=%d host_alloc_call_count=%llu exceeds threshold=%ld\n",
                tag ? tag : "graph", device, (unsigned long long) s.host_alloc_call_count, warn_calls);
    }
}

namespace {

using arena_profile_clock = std::chrono::high_resolution_clock;

struct arena_pp_profile_state {
    bool active = false;
    int  device = -1;

    arena_profile_clock::time_point graph_start{};

    std::array<uint64_t, static_cast<size_t>(vram_zone_id::COUNT)> zone_alloc_calls{};
    std::array<uint64_t, static_cast<size_t>(vram_zone_id::COUNT)> zone_alloc_failures{};
    std::array<uint64_t, static_cast<size_t>(vram_zone_id::COUNT)> zone_alloc_bytes{};
    std::array<double, static_cast<size_t>(vram_zone_id::COUNT)>   zone_alloc_us{};
    std::array<uint64_t, static_cast<size_t>(vram_zone_id::COUNT)> zone_free_calls{};
    std::array<double, static_cast<size_t>(vram_zone_id::COUNT)>   zone_free_us{};

    std::array<uint64_t, static_cast<size_t>(vram_zone_id::COUNT)> zone_reset_calls{};
    std::array<double, static_cast<size_t>(vram_zone_id::COUNT)>   zone_reset_us{};

    std::array<size_t, static_cast<size_t>(vram_zone_id::COUNT)> zone_used_begin{};
    std::array<size_t, static_cast<size_t>(vram_zone_id::COUNT)> zone_used_end{};
    std::array<size_t, static_cast<size_t>(vram_zone_id::COUNT)> zone_used_peak{};
    std::array<size_t, static_cast<size_t>(vram_zone_id::COUNT)> zone_capacity{};
    std::array<size_t, static_cast<size_t>(vram_zone_id::COUNT)> zone_largest_free_begin{};
    std::array<size_t, static_cast<size_t>(vram_zone_id::COUNT)> zone_largest_free_end{};
    std::array<size_t, static_cast<size_t>(vram_zone_id::COUNT)> zone_largest_free_min{};

    uint64_t compute_arena_alloc_calls = 0;
    uint64_t compute_arena_alloc_fail  = 0;
    uint64_t compute_arena_alloc_bytes = 0;
    double   compute_arena_alloc_us    = 0.0;
    uint64_t compute_arena_reset_calls = 0;
    double   compute_arena_reset_us    = 0.0;
    size_t   compute_arena_used_begin  = 0;
    size_t   compute_arena_used_end    = 0;
    size_t   compute_arena_capacity    = 0;

    uint64_t onednn_reserve_calls       = 0;
    uint64_t onednn_reserve_reuse_hits  = 0;
    uint64_t onednn_reserve_arena_hits  = 0;
    uint64_t onednn_reserve_direct_hits = 0;
    uint64_t onednn_reserve_failures    = 0;
    uint64_t onednn_reserve_weights_req = 0;
    uint64_t onednn_reserve_acts_req    = 0;
    double   onednn_reserve_us          = 0.0;

    uint64_t onednn_get_calls       = 0;
    uint64_t onednn_get_failures    = 0;
    uint64_t onednn_get_weights_req = 0;
    uint64_t onednn_get_acts_req    = 0;
    double   onednn_get_us          = 0.0;
};

thread_local arena_pp_profile_state t_arena_pp_profile;

static const char * arena_zone_name(vram_zone_id zone) {
    switch (zone) {
        case vram_zone_id::KV:
            return "kv";
        case vram_zone_id::WEIGHT:
            return "weight";
        case vram_zone_id::ONEDNN:
            return "onednn";
        case vram_zone_id::RUNTIME:
            return "runtime";
        case vram_zone_id::SCRATCH:
            return "scratch";
        case vram_zone_id::COUNT:
            break;
    }
    return "unknown";
}

static double arena_profile_elapsed_us(const arena_profile_clock::time_point & start) {
    return std::chrono::duration<double, std::micro>(arena_profile_clock::now() - start).count();
}

static void arena_pp_profile_snapshot_zones(unified_cache *                                                cache,
                                            std::array<size_t, static_cast<size_t>(vram_zone_id::COUNT)> & used_out,
                                            std::array<size_t, static_cast<size_t>(vram_zone_id::COUNT)> & cap_out) {
    if (!cache || !cache->arena_active()) {
        used_out.fill(0);
        cap_out.fill(0);
        return;
    }
    for (size_t i = 0; i < static_cast<size_t>(vram_zone_id::COUNT); ++i) {
        const auto zid = static_cast<vram_zone_id>(i);
        used_out[i]    = cache->zone_used(zid);
        cap_out[i]     = cache->zone_capacity(zid);
    }
}

}  // namespace

bool arena_pp_profile_enabled() {
    static const bool enabled = []() {
        const char * env = std::getenv("GGML_SYCL_ARENA_PP_PROFILE");
        return env && std::atoi(env) != 0;
    }();
    return enabled;
}

bool arena_pp_profile_active() {
    return t_arena_pp_profile.active;
}

bool arena_pp_profile_begin(int device, bool is_prompt_phase) {
    t_arena_pp_profile = {};

    if (!arena_pp_profile_enabled() || !is_prompt_phase) {
        return false;
    }

    unified_cache * cache = get_unified_cache_for_device(device);
    if (!cache || !cache->arena_active()) {
        return false;
    }

    t_arena_pp_profile.active      = true;
    t_arena_pp_profile.device      = device;
    t_arena_pp_profile.graph_start = arena_profile_clock::now();
    arena_pp_profile_snapshot_zones(cache, t_arena_pp_profile.zone_used_begin, t_arena_pp_profile.zone_capacity);
    for (size_t i = 0; i < static_cast<size_t>(vram_zone_id::COUNT); ++i) {
        const auto zid                                = static_cast<vram_zone_id>(i);
        t_arena_pp_profile.zone_used_peak[i]          = t_arena_pp_profile.zone_used_begin[i];
        t_arena_pp_profile.zone_largest_free_begin[i] = cache->zone_largest_free(zid);
        t_arena_pp_profile.zone_largest_free_min[i]   = t_arena_pp_profile.zone_largest_free_begin[i];
    }
    t_arena_pp_profile.compute_arena_used_begin = cache->compute_arena_used();
    t_arena_pp_profile.compute_arena_capacity   = cache->compute_arena_capacity();
    return true;
}

void arena_pp_profile_end(const char * tag, int device) {
    arena_pp_profile_state stats = t_arena_pp_profile;
    t_arena_pp_profile           = {};

    if (!stats.active || stats.device != device) {
        return;
    }

    unified_cache * cache = get_unified_cache_for_device(device);
    arena_pp_profile_snapshot_zones(cache, stats.zone_used_end, stats.zone_capacity);
    if (cache) {
        for (size_t i = 0; i < static_cast<size_t>(vram_zone_id::COUNT); ++i) {
            stats.zone_largest_free_end[i] = cache->zone_largest_free(static_cast<vram_zone_id>(i));
        }
        stats.compute_arena_used_end = cache->compute_arena_used();
        stats.compute_arena_capacity = cache->compute_arena_capacity();
    }

    const double total_ms =
        std::chrono::duration<double, std::milli>(arena_profile_clock::now() - stats.graph_start).count();
    const char * onednn_source = cache ? cache->onednn_scratch_source_name() : "unavailable";

    fprintf(
        stderr,
        "[ARENA-PP] tag=%s device=%d total_ms=%.3f onednn_source=%s compute_arena_mb=%.1f->%.1f/%.1f "
        "kv_mb=%.1f->%.1f/%0.1f onednn_mb=%.1f->%.1f/%0.1f runtime_mb=%.1f->%.1f/%0.1f scratch_mb=%.1f->%.1f/%0.1f\n",
        tag ? tag : "graph_compute", device, total_ms, onednn_source,
        stats.compute_arena_used_begin / (1024.0 * 1024.0), stats.compute_arena_used_end / (1024.0 * 1024.0),
        stats.compute_arena_capacity / (1024.0 * 1024.0),
        stats.zone_used_begin[static_cast<size_t>(vram_zone_id::KV)] / (1024.0 * 1024.0),
        stats.zone_used_end[static_cast<size_t>(vram_zone_id::KV)] / (1024.0 * 1024.0),
        stats.zone_capacity[static_cast<size_t>(vram_zone_id::KV)] / (1024.0 * 1024.0),
        stats.zone_used_begin[static_cast<size_t>(vram_zone_id::ONEDNN)] / (1024.0 * 1024.0),
        stats.zone_used_end[static_cast<size_t>(vram_zone_id::ONEDNN)] / (1024.0 * 1024.0),
        stats.zone_capacity[static_cast<size_t>(vram_zone_id::ONEDNN)] / (1024.0 * 1024.0),
        stats.zone_used_begin[static_cast<size_t>(vram_zone_id::RUNTIME)] / (1024.0 * 1024.0),
        stats.zone_used_end[static_cast<size_t>(vram_zone_id::RUNTIME)] / (1024.0 * 1024.0),
        stats.zone_capacity[static_cast<size_t>(vram_zone_id::RUNTIME)] / (1024.0 * 1024.0),
        stats.zone_used_begin[static_cast<size_t>(vram_zone_id::SCRATCH)] / (1024.0 * 1024.0),
        stats.zone_used_end[static_cast<size_t>(vram_zone_id::SCRATCH)] / (1024.0 * 1024.0),
        stats.zone_capacity[static_cast<size_t>(vram_zone_id::SCRATCH)] / (1024.0 * 1024.0));

    fprintf(stderr, "[ARENA-PP-ZONE]");
    for (size_t i = 0; i < static_cast<size_t>(vram_zone_id::COUNT); ++i) {
        const auto zid = static_cast<vram_zone_id>(i);
        fprintf(stderr,
                " %s_calls=%llu %s_fail=%llu %s_mb=%.1f %s_ms=%.3f %s_frees=%llu %s_free_ms=%.3f "
                "%s_resets=%llu %s_reset_ms=%.3f",
                arena_zone_name(zid), (unsigned long long) stats.zone_alloc_calls[i], arena_zone_name(zid),
                (unsigned long long) stats.zone_alloc_failures[i], arena_zone_name(zid),
                stats.zone_alloc_bytes[i] / (1024.0 * 1024.0), arena_zone_name(zid), stats.zone_alloc_us[i] / 1000.0,
                arena_zone_name(zid), (unsigned long long) stats.zone_free_calls[i], arena_zone_name(zid),
                stats.zone_free_us[i] / 1000.0, arena_zone_name(zid), (unsigned long long) stats.zone_reset_calls[i],
                arena_zone_name(zid), stats.zone_reset_us[i] / 1000.0);
    }
    fprintf(stderr, "\n");

    const size_t scratch_idx = static_cast<size_t>(vram_zone_id::SCRATCH);
    fprintf(stderr,
            "[ARENA-PP-SCRATCH] peak_live_mb=%.1f largest_free_mb=%.1f->%.1f min_largest_free_mb=%.1f "
            "alloc_free_ratio=%.2f req_to_peak_ratio=%.2f\n",
            stats.zone_used_peak[scratch_idx] / (1024.0 * 1024.0),
            stats.zone_largest_free_begin[scratch_idx] / (1024.0 * 1024.0),
            stats.zone_largest_free_end[scratch_idx] / (1024.0 * 1024.0),
            stats.zone_largest_free_min[scratch_idx] / (1024.0 * 1024.0),
            stats.zone_free_calls[scratch_idx] ? static_cast<double>(stats.zone_alloc_calls[scratch_idx]) /
                                                     static_cast<double>(stats.zone_free_calls[scratch_idx]) :
                                                 0.0,
            stats.zone_used_peak[scratch_idx] ? static_cast<double>(stats.zone_alloc_bytes[scratch_idx]) /
                                                    static_cast<double>(stats.zone_used_peak[scratch_idx]) :
                                                0.0);

    fprintf(stderr,
            "[ARENA-PP-COMPUTE] alloc_calls=%llu alloc_fail=%llu alloc_mb=%.1f alloc_ms=%.3f "
            "reset_calls=%llu reset_ms=%.3f\n",
            (unsigned long long) stats.compute_arena_alloc_calls, (unsigned long long) stats.compute_arena_alloc_fail,
            stats.compute_arena_alloc_bytes / (1024.0 * 1024.0), stats.compute_arena_alloc_us / 1000.0,
            (unsigned long long) stats.compute_arena_reset_calls, stats.compute_arena_reset_us / 1000.0);

    fprintf(stderr,
            "[ARENA-PP-ONEDNN] reserve_calls=%llu reserve_reuse=%llu reserve_arena=%llu reserve_direct=%llu "
            "reserve_fail=%llu reserve_req_mb=%.1f/%.1f reserve_ms=%.3f "
            "get_calls=%llu get_fail=%llu get_req_mb=%.1f/%.1f get_ms=%.3f\n",
            (unsigned long long) stats.onednn_reserve_calls, (unsigned long long) stats.onednn_reserve_reuse_hits,
            (unsigned long long) stats.onednn_reserve_arena_hits, (unsigned long long) stats.onednn_reserve_direct_hits,
            (unsigned long long) stats.onednn_reserve_failures, stats.onednn_reserve_weights_req / (1024.0 * 1024.0),
            stats.onednn_reserve_acts_req / (1024.0 * 1024.0), stats.onednn_reserve_us / 1000.0,
            (unsigned long long) stats.onednn_get_calls, (unsigned long long) stats.onednn_get_failures,
            stats.onednn_get_weights_req / (1024.0 * 1024.0), stats.onednn_get_acts_req / (1024.0 * 1024.0),
            stats.onednn_get_us / 1000.0);
}

void arena_pp_profile_note_onednn_reserve(size_t weights_size,
                                          size_t activations_size,
                                          bool   reused,
                                          bool   arena_attempted,
                                          bool   arena_success,
                                          bool   direct_attempted,
                                          bool   ok,
                                          double elapsed_us) {
    if (!t_arena_pp_profile.active) {
        return;
    }
    t_arena_pp_profile.onednn_reserve_calls++;
    t_arena_pp_profile.onednn_reserve_weights_req += weights_size;
    t_arena_pp_profile.onednn_reserve_acts_req += activations_size;
    t_arena_pp_profile.onednn_reserve_us += elapsed_us;
    if (reused) {
        t_arena_pp_profile.onednn_reserve_reuse_hits++;
    }
    if (arena_attempted && arena_success) {
        t_arena_pp_profile.onednn_reserve_arena_hits++;
    }
    if (direct_attempted && ok) {
        t_arena_pp_profile.onednn_reserve_direct_hits++;
    }
    if (!ok) {
        t_arena_pp_profile.onednn_reserve_failures++;
    }
}

void arena_pp_profile_note_onednn_get(size_t weights_needed, size_t activations_needed, bool ok, double elapsed_us) {
    if (!t_arena_pp_profile.active) {
        return;
    }
    t_arena_pp_profile.onednn_get_calls++;
    t_arena_pp_profile.onednn_get_weights_req += weights_needed;
    t_arena_pp_profile.onednn_get_acts_req += activations_needed;
    t_arena_pp_profile.onednn_get_us += elapsed_us;
    if (!ok) {
        t_arena_pp_profile.onednn_get_failures++;
    }
}

static bool parse_env_mb_value(const char * name, size_t & out_mb) {
    const char * env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return false;
    }
    char * end = nullptr;
    long   mb  = std::strtol(env, &end, 10);
    if (end == env || mb < 0) {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Invalid %s='%s'\n", name, env);
        return false;
    }
    out_mb = static_cast<size_t>(mb);
    return true;
}

static bool parse_env_count_value(const char * name, size_t & out_count) {
    const char * env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return false;
    }
    char * end   = nullptr;
    long   count = std::strtol(env, &end, 10);
    if (end == env || count < 0) {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Invalid %s='%s'\n", name, env);
        return false;
    }
    out_count = static_cast<size_t>(count);
    return true;
}

static void resolve_dma_defaults(size_t & slice_bytes, size_t & buffer_count) {
    size_t slice_mb = 1024;
    size_t buffers  = 2;
    size_t env_val  = 0;

    const bool slice_env_set = parse_env_mb_value("GGML_SYCL_DMA_SLICE_MB", env_val);
    if (slice_env_set) {
        slice_mb = env_val;
    }
    const bool buffers_env_set = parse_env_count_value("GGML_SYCL_DMA_BUFFERS", env_val) ||
                                 parse_env_count_value("GGML_SYCL_DMA_SLICES", env_val);
    if (buffers_env_set) {
        buffers = env_val;
    }
    if (!slice_env_set && !buffers_env_set && ggml_backend_sycl_weights_evictable()) {
        // Use smaller defaults for evictable weights to reduce staging OOM risk.
        slice_mb = std::min<size_t>(slice_mb, 32);
        buffers  = std::min<size_t>(buffers, 1);
    }

    if (slice_bytes == 0) {
        slice_bytes = slice_mb * 1024ULL * 1024ULL;
    }
    if (buffer_count == 0) {
        buffer_count = buffers;
    }
}

static size_t resolve_host_staging_bytes() {
    size_t staging_mb = 64;
    size_t env_mb     = 0;
    if (parse_env_mb_value("GGML_SYCL_HOST_STAGING_MB", env_mb) ||
        parse_env_mb_value("GGML_SYCL_MMAP_STAGING_MB", env_mb)) {
        staging_mb = env_mb;
    }
    return staging_mb * 1024ULL * 1024ULL;
}

static size_t resolve_host_reserve_bytes(size_t staging_bytes) {
    size_t reserve_mb = 0;
    size_t env_mb     = 0;
    if (parse_env_mb_value("GGML_SYCL_HOST_RESERVE_MB", env_mb)) {
        reserve_mb = env_mb;
    } else {
        reserve_mb = staging_bytes / (1024ULL * 1024ULL);
    }
    return reserve_mb * 1024ULL * 1024ULL;
}

static bool cache_assert_enabled() {
    int enabled = g_cache_assert_enabled.load(std::memory_order_acquire);
    if (enabled >= 0) {
        return enabled != 0;
    }
    const char * env = std::getenv("GGML_SYCL_CACHE_ASSERT");
    enabled          = (env && std::atoi(env) != 0) ? 1 : 0;
    g_cache_assert_enabled.store(enabled, std::memory_order_release);
    return enabled != 0;
}

static bool copy_trace_enabled() {
    int enabled = g_copy_trace_enabled.load(std::memory_order_acquire);
    if (enabled >= 0) {
        return enabled != 0;
    }
    const char * env = std::getenv("GGML_SYCL_COPY_TRACE");
    enabled          = (env && std::atoi(env) != 0) ? 1 : 0;
    g_copy_trace_enabled.store(enabled, std::memory_order_release);
    return enabled != 0;
}

static bool copy_to_device_sync_enabled() {
    static std::atomic<int> cached{ -1 };
    int                     enabled = cached.load(std::memory_order_acquire);
    if (enabled >= 0) {
        return enabled != 0;
    }
    const char * env = std::getenv("GGML_SYCL_COPY_TO_DEVICE_SYNC");
    enabled          = (env && std::atoi(env) != 0) ? 1 : 0;
    cached.store(enabled, std::memory_order_release);
    return enabled != 0;
}

static size_t copy_to_device_stage_slots() {
    static std::atomic<size_t> cached{ 0 };
    size_t                     slots = cached.load(std::memory_order_acquire);
    if (slots != 0) {
        return slots;
    }
    size_t parsed = 3;
    if (const char * env = std::getenv("GGML_SYCL_COPY_TO_DEVICE_STAGE_SLOTS")) {
        parsed = static_cast<size_t>(std::max(1, std::atoi(env)));
    }
    parsed = std::min<size_t>(parsed, 16);
    cached.store(parsed, std::memory_order_release);
    return parsed;
}

// atexit handler to prevent SYCL cleanup during static destruction
static void unified_cache_atexit_handler() {
    g_sycl_shutting_down.store(true, std::memory_order_release);
}

// Forward declarations needed by unified_cache constructor (defined later in file)
static size_t get_total_system_memory_bytes();
static size_t get_available_system_memory_bytes();

bool ggml_sycl_is_shutting_down() {
    return g_sycl_shutting_down.load(std::memory_order_acquire);
}

unified_cache::unified_cache(sycl::queue & queue,
                             size_t        budget_bytes,
                             size_t        staging_size,
                             size_t        dma_reserved_bytes,
                             size_t        device_total_vram) :
    queue_(queue),
    budget_(budget_bytes),
    base_budget_(budget_bytes),
    reserved_(0),
    dma_reserved_bytes_(dma_reserved_bytes) {
    // Register atexit handler once to set shutdown flag before static destructors run
    // This prevents the destructor from calling sycl::free() on invalid queue
    bool expected = false;
    if (g_atexit_registered.compare_exchange_strong(expected, true)) {
        std::atexit(unified_cache_atexit_handler);
    }

    // Allocate staging buffer (pinned host memory)
    try {
        staging_ = ggml_sycl_malloc_host(staging_size, queue_, "unified_cache:staging");
        if (staging_) {
            staging_size_ = staging_size;
        }
    } catch (const sycl::exception & e) {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Failed to allocate staging buffer: %s\n", e.what());
        staging_      = nullptr;
        staging_size_ = 0;
    }

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] Initialized: budget=%.1f MB, staging=%.1f MB, dma-reserve=%.1f MB\n",
                    budget_ / (1024.0f * 1024.0f), staging_size_ / (1024.0f * 1024.0f),
                    dma_reserved_bytes_ / (1024.0f * 1024.0f));

    // Pre-allocate reusable host-pinned staging slots for copy_to_device_async.
    // This eliminates per-expert sycl::malloc_host / sycl::free churn during
    // inference — each alloc/free does GGTT page table ops in the kernel driver.
    // NOTE: We use ggml_sycl_malloc_host directly (not unified_alloc) because
    // the constructor runs under g_cache_rw_mutex and unified_alloc would deadlock
    // via unified_cache_add_runtime_host_bytes -> g_cache_rw_mutex.
    {
        constexpr size_t k_fallback_chunk = 64 * 1024 * 1024;
        const size_t     slot_capacity    = staging_size_ > 0 ? staging_size_ : k_fallback_chunk;
        const size_t     n_slots          = copy_to_device_stage_slots();

        copy_stage_slots_.reserve(n_slots);
        for (size_t i = 0; i < n_slots; ++i) {
            try {
                void * ptr = ggml_sycl_malloc_host(slot_capacity, queue_, "copy_stage_slot_prealloc");
                if (ptr) {
                    copy_stage_slot slot{};
                    slot.ptr       = ptr;
                    slot.device    = -1;  // host-pinned, no device association
                    slot.capacity  = slot_capacity;
                    slot.in_flight = false;
                    copy_stage_slots_.push_back(slot);
                } else {
                    GGML_SYCL_DEBUG("[UNIFIED-CACHE] Failed to pre-allocate staging slot %zu\n", i);
                }
            } catch (const sycl::exception & e) {
                GGML_SYCL_DEBUG("[UNIFIED-CACHE] Failed to pre-allocate staging slot %zu: %s\n", i, e.what());
            }
        }
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Pre-allocated %zu/%zu staging slots (%.1f MB each)\n",
                        copy_stage_slots_.size(), n_slots, slot_capacity / (1024.0f * 1024.0f));
    }

    // Initialize layout pool for consolidating layout allocations into
    // contiguous chunks (reduces GPU TLB misses from scattered USM mappings).
    layout_pool_ = std::make_unique<sycl_device_pool>(queue_, ggml_sycl_get_device_id_from_queue(queue_));

    // VRAM Arena: pre-allocate a single VRAM block when GGML_SYCL_VRAM_ARENA=1.
    if (vram_arena_enabled()) {
        const int    dev_id    = ggml_sycl_get_device_id_from_queue(queue_);
        // Per-allocation cap = the runtime's reported max_mem_alloc_size from
        // the L0 device query.  This is what the runtime actually accepts;
        // the alloc-probe `safe_max_alloc_size` (in ggml_sycl_info) saturates
        // at a smaller value (~1.5 GB on Arc B580 + patched libze 1.14.37435),
        // but that's the probe's stop-condition, NOT a hard runtime cap.
        // Empirical: 5825 + 4660 MB chunks succeed on this hardware.  Using
        // the probe value here would manufacture artificial caps that force
        // unnecessary chunking on post-init paths.
        const size_t max_alloc = queue_.get_device().get_info<sycl::info::device::max_mem_alloc_size>();

        // Default zone sizes.  Scratch arena default is 256 MB.
        size_t       scratch_zone = 256 * 1024 * 1024;
        const char * arena_mb_env = std::getenv("GGML_SYCL_COMPUTE_ARENA_MB");
        if (arena_mb_env) {
            scratch_zone = static_cast<size_t>(std::max(0, std::atoi(arena_mb_env))) * 1024 * 1024;
        }

        // oneDNN scratch: 0 by default, sized later by reserve_onednn_scratch.
        // Pre-reserve a generous 256 MB for oneDNN to avoid later realloc.
        size_t onednn_zone = 256 * 1024 * 1024;

        // Runtime zone: 512 MB default for KV buffers, staging, MoE pools.
        // For TP-enabled models, the placement_plan::tp_vram_runtime_bytes field
        // documents the additional RUNTIME bytes needed for secondary devices;
        // the secondary device planner is expected to pass that value here.
        // Zone size is fixed at arena creation — cannot grow after freezing.
        size_t runtime_zone = 512 * 1024 * 1024;
        if (const char * env = std::getenv("GGML_SYCL_RUNTIME_ARENA_MB")) {
            runtime_zone = static_cast<size_t>(std::max(0, std::atoi(env))) * 1024 * 1024;
        }
        const size_t planned_pp_pipeline = unified_cache_get_planned_pp_pipeline_scratch_bytes(dev_id);
        if (planned_pp_pipeline > 0 && runtime_zone < planned_pp_pipeline) {
            runtime_zone = planned_pp_pipeline;
            GGML_LOG_INFO("[UNIFIED-CACHE] Runtime zone raised to %.1f MB for PP pipeline scratch planning\n",
                          runtime_zone / (1024.0 * 1024.0));
        }

        if (arena_reserve(queue_, budget_bytes, max_alloc, scratch_zone, onednn_zone, runtime_zone,
                          device_total_vram)) {
            // Point compute_arena at the arena's SCRATCH zone immediately so
            // pool_leg can route through it.  Without this, pool_leg falls back
            // to sycl::malloc_device which can return low-VA pointers that the
            // L0 driver doesn't recognize as USM (compute-runtime bug).
            const auto & cz_info = get_zone(vram_zone_id::SCRATCH);
            compute_arena_ptr_   = offset_to_ptr(cz_info.start);
            compute_arena_size_  = cz_info.size;
            compute_arena_off_.store(0, std::memory_order_relaxed);
            GGML_LOG_INFO("[VRAM-ARENA] Active on device %d: %d chunk(s), %.1f MB total\n", dev_id, chunk_count(),
                          arena_total_size() / (1024.0 * 1024.0));
            GGML_LOG_INFO("[VRAM-ARENA] Scratch zone: %p + %.1f MB (offset %.1f MB from base)\n", compute_arena_ptr_,
                          compute_arena_size_ / (1024.0 * 1024.0), cz_info.start / (1024.0 * 1024.0));
            // Bind layout pool to the arena so new layout allocations come from the
            // arena's weight zone instead of allocating separate chunks.
            if (layout_pool_) {
                layout_pool_->set_arena(this);
            }
        } else {
            GGML_LOG_WARN("[VRAM-ARENA] Failed on device %d, falling back to per-entry allocation\n", dev_id);
        }
    }

    // Create a separate in-order DMA queue for cache operations (CCS engine).
    // This keeps cache DMA/fill work off the compute queue, preventing
    // >20s accumulated queue work that triggers L0 DirectSubmission timeouts.
    try {
        dma_queue_ =
            std::make_unique<sycl::queue>(queue_.get_context(), queue_.get_device(), sycl::property::queue::in_order{});
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Created separate DMA queue for cache operations\n");
    } catch (const sycl::exception & e) {
        GGML_LOG_WARN("[UNIFIED-CACHE] Failed to create DMA queue, falling back to compute queue: %s\n", e.what());
        dma_queue_.reset();
    }

    // Create a BCS (copy-only) queue for H2D transfers during expert prestaging.
    // On Intel GPUs, Level Zero exposes queue groups: ordinal 0 = CCS (compute+copy),
    // ordinal 1 = BCS (copy-only / blitter).  By routing H2D memcpy to a separate
    // in-order queue, the runtime can assign it to the BCS engine, keeping CCS free
    // for SOA reorder kernels.  This prevents CCS monopolization during the ~6000
    // kernel submissions of MoE expert prestaging that trigger GT engine resets.
    //
    // Even if the runtime routes both queues to CCS, having separate queues still
    // enables pipelining: H2D copies and reorder kernels interleave via event deps
    // instead of serializing on a single command list.
    try {
        bcs_queue_ =
            std::make_unique<sycl::queue>(queue_.get_context(), queue_.get_device(), sycl::property::queue::in_order{});
        GGML_LOG_INFO("[UNIFIED-CACHE] Created BCS queue for H2D copy pipelining\n");
        // Give the device pool a reference so it can drain BCS before chunk allocs.
        if (layout_pool_) {
            layout_pool_->set_bcs_queue(bcs_queue_.get());
        }
    } catch (const sycl::exception & e) {
        GGML_LOG_WARN("[UNIFIED-CACHE] Failed to create BCS queue, falling back to DMA queue: %s\n", e.what());
        bcs_queue_.reset();
    }

    // P7: async DMA eviction — default ON when arena is active.
    // Env var GGML_SYCL_ASYNC_EVICT=0 disables.
    {
        const char * env     = std::getenv("GGML_SYCL_ASYNC_EVICT");
        bool         dflt    = arena_active();  // Default ON when arena active
        async_evict_enabled_ = env ? (std::string(env) != "0") : dflt;
        if (async_evict_enabled_) {
            GGML_LOG_INFO("[UNIFIED-CACHE] Async DMA eviction enabled (preserves layouts during migration)\n");
        }
    }

    // Host arena: allocate pinned host memory via 2GB chunks to bypass Level Zero's
    // ~11GB per-allocation limit.  Budget is derived from queried system RAM
    // and bounded by currently available RAM minus an OS reserve, NOT by VRAM.
    // The pool grows lazily; this budget is a cap, not committed memory.
    {
        size_t host_mem_budget = g_unified_cache_host_budget;
        size_t total_mem       = 0;
        size_t available_mem   = 0;
        size_t os_reserve      = 0;
        if (host_mem_budget == 0) {
            total_mem     = get_total_system_memory_bytes();
            available_mem = get_available_system_memory_bytes();
            int pct       = g_unified_cache_host_budget_pct;
            if (pct < 1) {
                pct = 1;
            } else if (pct > 100) {
                pct = 100;
            }

            if (total_mem > 0) {
                host_mem_budget =
                    static_cast<size_t>(static_cast<double>(total_mem) * (static_cast<double>(pct) / 100.0));
                os_reserve = std::max<size_t>(8ull << 30, total_mem / 32);
            } else if (available_mem > 0) {
                host_mem_budget = available_mem;
                os_reserve      = std::min<size_t>(8ull << 30, available_mem / 4);
            } else {
                host_mem_budget = std::numeric_limits<size_t>::max() / 4;
                GGML_LOG_WARN(
                    "[HOST-ARENA] System RAM detection failed; using lazy-growth host arena without a detected RAM "
                    "cap\n");
            }

            if (available_mem > 0 && os_reserve > 0) {
                const size_t available_cap =
                    available_mem > os_reserve ? available_mem - os_reserve : available_mem / 2;
                host_mem_budget = std::min(host_mem_budget, available_cap);
            }
        }

        // On Intel Arc discrete GPUs (Xe architecture), USM host memory
        // (sycl::malloc_host / zeMemAllocHost) is mapped through the PPGTT
        // (Per-Process GTT), NOT the GGTT.  The PPGTT provides a 256 TB
        // address space per device — effectively unlimited for practical
        // allocations.  The GGTT is reserved for kernel/privileged resources
        // (GuC firmware, display engine) and is NOT consumed by user-space
        // USM allocations.  Therefore there is no GGTT aperture constraint
        // on pinned host memory, and no budget cap is needed for multi-GPU.
        //
        // See: Intel GPU PRM Vol06 "Memory Views" — PPGTT is 48-bit (256 TB),
        //      GGTT is 32-bit (4 GB) and reserved for kernel resources only.
        //      Level Zero spec: host allocations are accessible by all devices
        //      in the context via PPGTT mappings, not GGTT.

        if (getenv("GGML_SYCL_NO_PINNED") != nullptr) {
            GGML_LOG_INFO("[HOST-ARENA] GGML_SYCL_NO_PINNED set: disabling pinned host memory\n");
            host_mem_budget = 0;
        }

        host_arena_ = std::make_unique<pinned_chunk_pool>(queue_, host_mem_budget);
        GGML_LOG_INFO(
            "[HOST-ARENA] Created with %.1f GB budget, committed=0.0 GB "
            "(system total=%.1f GB available=%.1f GB reserve=%.1f GB pct=%d)\n",
            host_mem_budget / (1024.0 * 1024.0 * 1024.0), total_mem / (1024.0 * 1024.0 * 1024.0),
            available_mem / (1024.0 * 1024.0 * 1024.0), os_reserve / (1024.0 * 1024.0 * 1024.0),
            g_unified_cache_host_budget_pct);
    }

    // Ensure unordered_map has buckets before any find() calls.
    entries_.rehash(1);
    id_to_key_.rehash(1);
}

unified_cache::~unified_cache() {
    // Stop the prefetch worker thread first (before any resource cleanup).
    // This is safe even if the SYCL runtime is shutting down since the worker
    // only does cache lookups and pinning, not SYCL memory operations.
    stop_prefetch_worker();

    // Skip cleanup if SYCL runtime is shutting down (static destruction order issue)
    // This can happen when the program exits and static destructors run in undefined order
    if (g_sycl_shutting_down.load()) {
        // Abandon pool chunks without calling sycl::free() (context is invalid)
        if (layout_pool_) {
            layout_pool_->abandon();
        }
        // Abandon arena without calling sycl::free — context is invalid.
        arena_abandon();
        compute_arena_ptr_ = nullptr;
        // Leak host_arena_ to prevent pinned_chunk_pool destructor calling sycl::free
        // on an invalid SYCL context (safe shutdown pattern).
        (void) host_arena_.release();
        return;
    }

    // Try to verify SYCL context is still valid before freeing
    // This guards against static destruction order issues where SYCL runtime
    // has been torn down before this destructor runs
    try {
        // Simple validity check - if this throws, SYCL is gone
        (void) queue_.get_context();
    } catch (...) {
        // SYCL runtime already destroyed, skip cleanup
        if (layout_pool_) {
            layout_pool_->abandon();
        }
        compute_arena_ptr_ = nullptr;
        // Leak host_arena_ to avoid sycl::free on an invalid context.
        (void) host_arena_.release();
        return;
    }

    // Check arena state before destroying anything.
    const bool had_arena = arena_active();

    // Free all cached entries (skip pool-allocated and arena-owned entries)
    for (auto & pair : entries_) {
        if (pair.second.device_ptr && !pair.second.pool_allocated &&
            !(had_arena && vram_owns(pair.second.device_ptr))) {
            try {
                sycl::free(pair.second.device_ptr, queue_);
            } catch (...) {
            }
        }
    }

    // Free compute arena BEFORE destroying the VRAM arena (which would invalidate owns check).
    if (compute_arena_ptr_ && !(had_arena && vram_owns(compute_arena_ptr_))) {
        try {
            sycl::free(compute_arena_ptr_, queue_);
        } catch (...) {
        }
        saturating_sub_used(compute_arena_size_);
    }
    compute_arena_ptr_  = nullptr;
    compute_arena_size_ = 0;
    compute_arena_off_.store(0, std::memory_order_relaxed);

    // Free scratch pool BEFORE destroying the VRAM arena (which would invalidate owns check).
    if (scratch_pool_ptr_ && !(had_arena && vram_owns(scratch_pool_ptr_))) {
        try {
            sycl::free(scratch_pool_ptr_, queue_);
        } catch (...) {
        }
        saturating_sub_used(scratch_pool_size_);
    }
    scratch_pool_ptr_  = nullptr;
    scratch_pool_size_ = 0;
    scratch_pool_off_.store(0, std::memory_order_relaxed);

    // Free oneDNN scratch buffers BEFORE arena destroy (vram_owns() needs live arena).
    if (onednn_weights_scratch_) {
        if (!(had_arena && vram_owns(onednn_weights_scratch_))) {
            try {
                sycl::free(onednn_weights_scratch_, queue_);
            } catch (...) {
            }
        }
        onednn_weights_scratch_ = nullptr;
    }
    if (onednn_activations_scratch_) {
        if (!(had_arena && vram_owns(onednn_activations_scratch_))) {
            try {
                sycl::free(onednn_activations_scratch_, queue_);
            } catch (...) {
            }
        }
        onednn_activations_scratch_ = nullptr;
    }

    // Free reorder temp buffer BEFORE arena destroy.
    if (reorder_temp_buffer_) {
        if (!(had_arena && vram_owns(reorder_temp_buffer_))) {
            alloc_registry::instance().unregister_alloc(reorder_temp_buffer_);
            saturating_sub_used(reorder_temp_size_);
            try {
                sycl::free(reorder_temp_buffer_, queue_);
            } catch (...) {
            }
        }
        reorder_temp_buffer_ = nullptr;
        reorder_temp_size_   = 0;
    }

    // Free persistent scratch buffers BEFORE arena destroy.
    for (auto & pair : persistent_scratches_) {
        if (pair.second.device_ptr) {
            if (!(had_arena && vram_owns(pair.second.device_ptr))) {
                try {
                    sycl::free(pair.second.device_ptr, queue_);
                } catch (...) {
                }
            }
        }
    }
    persistent_scratches_.clear();

    // Destroy the VRAM arena (frees the pre-allocated chunks).
    arena_destroy();

    // Destroy layout pool before SYCL context goes away.
    // The pool's reset() returns physical bytes freed so we can decrement used_.
    if (layout_pool_) {
        const size_t pool_freed = layout_pool_->reset();
        if (pool_freed > 0 && used_.load(std::memory_order_relaxed) >= pool_freed) {
            used_.fetch_sub(pool_freed, std::memory_order_relaxed);
        }
        layout_pool_.reset();
    }

    // Free staging buffer
    if (staging_) {
        try {
            sycl::free(staging_, queue_);
        } catch (...) {
        }
    }

    // Free DMA staging buffers
    for (size_t i = 0; i < dma_staging_buffers_.size(); ++i) {
        if (i < dma_staging_allocs_.size() && dma_staging_allocs_[i].ptr != nullptr) {
            alloc_handle handle{};
            handle.ptr      = dma_staging_allocs_[i].ptr;
            handle.size     = dma_staging_allocs_[i].size;
            handle.device   = dma_staging_allocs_[i].device;
            handle.alloc_id = dma_staging_allocs_[i].alloc_id;
            unified_free(handle);
            continue;
        }
        void * ptr = dma_staging_buffers_[i];
        if (!ptr) {
            continue;
        }
        try {
            sycl::free(ptr, queue_);
        } catch (...) {
        }
    }
    dma_staging_allocs_.clear();
    dma_staging_buffers_.clear();

    // Free reusable host-pinned async copy staging slots.
    // These were allocated with ggml_sycl_malloc_host (not unified_alloc) to
    // avoid g_cache_rw_mutex deadlock during construction, so free via sycl::free.
    for (auto & slot : copy_stage_slots_) {
        if (slot.ptr != nullptr) {
            try {
                sycl::free(slot.ptr, queue_);
            } catch (...) {
            }
            slot.ptr = nullptr;
        }
    }
    copy_stage_slots_.clear();

    // Free any deferred frees that haven't been released yet.
    for (auto & entry : deferred_frees_) {
        if (entry.ptr) {
            try {
                sycl::free(entry.ptr, queue_);
            } catch (...) {
            }
        }
    }
    deferred_frees_.clear();

    // Free partial row entries (multi-device tensor split)
    for (auto & pair : partial_cache_) {
        if (pair.second.ptr) {
            try {
                sycl::free(pair.second.ptr, queue_);
            } catch (...) {
            }
        }
    }
    partial_cache_.clear();
}

// Fast 64-bit hash of entire data buffer (xxHash-style)
// Computes full content hash for robust change detection
// ~10 GB/s on modern CPUs - acceptable for one-time cache miss
static uint64_t compute_content_hash(const void * data, size_t size) {
    if (!data || size == 0) {
        return 0;
    }

    const uint8_t * bytes = static_cast<const uint8_t *>(data);

    // xxHash-style constants
    constexpr uint64_t PRIME1 = 0x9E3779B185EBCA87ULL;
    constexpr uint64_t PRIME2 = 0xC2B2AE3D27D4EB4FULL;
    constexpr uint64_t PRIME3 = 0x165667B19E3779F9ULL;
    constexpr uint64_t PRIME4 = 0x85EBCA77C2B2AE63ULL;
    constexpr uint64_t PRIME5 = 0x27D4EB2F165667C5ULL;

    uint64_t hash = PRIME5 + size;

    auto load_u64_unaligned = [](const void * ptr) -> uint64_t {
        uint64_t value;
        std::memcpy(&value, ptr, sizeof(value));
        return value;
    };

    // Process 32-byte chunks for speed
    size_t num_chunks = size / 32;

    if (num_chunks > 0) {
        uint64_t v1 = hash + PRIME1 + PRIME2;
        uint64_t v2 = hash + PRIME2;
        uint64_t v3 = hash;
        uint64_t v4 = hash - PRIME1;

        for (size_t i = 0; i < num_chunks; i++) {
            const uint8_t * chunk = bytes + i * 32;
            uint64_t        k1    = load_u64_unaligned(chunk + 0);
            uint64_t        k2    = load_u64_unaligned(chunk + 8);
            uint64_t        k3    = load_u64_unaligned(chunk + 16);
            uint64_t        k4    = load_u64_unaligned(chunk + 24);

            v1 += k1 * PRIME2;
            v1 = (v1 << 31) | (v1 >> 33);
            v1 *= PRIME1;

            v2 += k2 * PRIME2;
            v2 = (v2 << 31) | (v2 >> 33);
            v2 *= PRIME1;

            v3 += k3 * PRIME2;
            v3 = (v3 << 31) | (v3 >> 33);
            v3 *= PRIME1;

            v4 += k4 * PRIME2;
            v4 = (v4 << 31) | (v4 >> 33);
            v4 *= PRIME1;
        }

        hash =
            ((v1 << 1) | (v1 >> 63)) + ((v2 << 7) | (v2 >> 57)) + ((v3 << 12) | (v3 >> 52)) + ((v4 << 18) | (v4 >> 46));

        hash ^= ((v1 * PRIME2) << 31 | (v1 * PRIME2) >> 33) * PRIME1;
        hash = hash * PRIME1 + PRIME4;
        hash ^= ((v2 * PRIME2) << 31 | (v2 * PRIME2) >> 33) * PRIME1;
        hash = hash * PRIME1 + PRIME4;
        hash ^= ((v3 * PRIME2) << 31 | (v3 * PRIME2) >> 33) * PRIME1;
        hash = hash * PRIME1 + PRIME4;
        hash ^= ((v4 * PRIME2) << 31 | (v4 * PRIME2) >> 33) * PRIME1;
        hash = hash * PRIME1 + PRIME4;
    }

    // Process remaining 8-byte chunks
    size_t          remaining = size - (num_chunks * 32);
    const uint8_t * tail      = bytes + (num_chunks * 32);

    while (remaining >= 8) {
        uint64_t k = load_u64_unaligned(tail);
        k *= PRIME2;
        k = (k << 31) | (k >> 33);
        k *= PRIME1;
        hash ^= k;
        hash = ((hash << 27) | (hash >> 37)) * PRIME1 + PRIME4;
        tail += 8;
        remaining -= 8;
    }

    // Process remaining bytes
    while (remaining > 0) {
        hash ^= static_cast<uint64_t>(*tail) * PRIME5;
        hash = ((hash << 11) | (hash >> 53)) * PRIME1;
        tail++;
        remaining--;
    }

    // Final avalanche
    hash ^= hash >> 33;
    hash *= PRIME2;
    hash ^= hash >> 29;
    hash *= PRIME3;
    hash ^= hash >> 32;

    return hash;
}

static size_t get_total_system_memory_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        return static_cast<size_t>(status.ullTotalPhys);
    }
    return 0;
#else
    long pages     = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages <= 0 || page_size <= 0) {
        return 0;
    }
    return static_cast<size_t>(pages) * static_cast<size_t>(page_size);
#endif
}

static size_t get_available_system_memory_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        return static_cast<size_t>(status.ullAvailPhys);
    }
    return 0;
#else
    FILE * f = std::fopen("/proc/meminfo", "r");
    if (f) {
        char               key[64]  = {};
        char               unit[32] = {};
        unsigned long long value_kb = 0;
        while (std::fscanf(f, "%63s %llu %31s", key, &value_kb, unit) == 3) {
            if (std::strcmp(key, "MemAvailable:") == 0) {
                std::fclose(f);
                return static_cast<size_t>(value_kb) * 1024ULL;
            }
        }
        std::fclose(f);
    }
#    ifdef _SC_AVPHYS_PAGES
    long pages     = sysconf(_SC_AVPHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) {
        return static_cast<size_t>(pages) * static_cast<size_t>(page_size);
    }
#    endif
    return 0;
#endif
}

static bool is_host_accessible_ptr(const void * ptr, const sycl::queue & queue) {
    if (!ptr) {
        return false;
    }
    try {
        const sycl::usm::alloc alloc = ggml_sycl_get_alloc_type(ptr);
        return alloc == sycl::usm::alloc::host || alloc == sycl::usm::alloc::shared;
    } catch (...) {
        return false;
    }
}

static const char * usm_alloc_name(sycl::usm::alloc alloc) {
    switch (alloc) {
        case sycl::usm::alloc::host:
            return "host";
        case sycl::usm::alloc::shared:
            return "shared";
        case sycl::usm::alloc::device:
            return "device";
        default:
            return "unknown";
    }
}

void * unified_cache::ensure_cached(const ggml_sycl_cache_id & key_id,
                                    const void *               src_ptr,
                                    size_t                     size,
                                    cache_entry_type           type,
                                    int                        layer_id,
                                    int                        expert_id,
                                    ggml_layout_mode           layout,
                                    bool                       validate_content) {
    if (!key_id.valid || !src_ptr || size == 0) {
        return nullptr;
    }

    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    // Only process deferred frees when no GPU kernels are in-flight.
    // During graph_compute (MoE inference), freed VRAM pages may still be
    // referenced by earlier MUL_MAT_ID kernels — processing frees here
    // causes GPU page faults (DEVICE_LOST).
    if (!g_graph_compute_active.load(std::memory_order_acquire)) {
        process_deferred_frees();
    }

    // Create key for lookup (identity-only, no layout)
    unified_cache_key key{ type, key_id, layer_id, expert_id };

    // Check if already cached
    auto it = entries_.find(key);
    if (it != entries_.end()) {
        auto id_it = id_to_key_.find(key_id);
        if (id_it == id_to_key_.end()) {
            id_to_key_.emplace(key_id, key);
        } else if (!(id_it->second == key)) {
            GGML_LOG_ERROR("[UNIFIED-CACHE] identity collision in ensure_cached model=%llu name_hash=0x%llx\n",
                           (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash);
            if (cache_assert_enabled()) {
                GGML_ABORT("unified_cache id_to_key mismatch");
            }
        }
        if (it->second.layout != layout) {
            GGML_LOG_ERROR(
                "[UNIFIED-CACHE] layout mismatch in ensure_cached model=%llu name_hash=0x%llx have=%d want=%d\n",
                (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash, (int) it->second.layout,
                (int) layout);
            if (cache_assert_enabled()) {
                GGML_ABORT("unified_cache layout mismatch");
            }
            return nullptr;
        }
        // Entry exists - check if size or content changed
        // This handles ABA: same identity with new src_ptr/size
        bool need_realloc = (size != it->second.size);
        bool need_recopy  = need_realloc || (it->second.src_ptr != src_ptr) || validate_content;

        if (need_recopy) {
            uint64_t new_hash        = compute_content_hash(src_ptr, size);
            bool     content_changed = (it->second.content_hash != new_hash);

            if (need_realloc) {
                // Size changed - need to reallocate device buffer
                GGML_SYCL_DEBUG(
                    "[UNIFIED-CACHE] Size changed for model=%llu name_hash=0x%llx (%zu -> %zu bytes), reallocating\n",
                    (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash, it->second.size, size);

                const bool   was_pinned          = it->second.pinned;
                const size_t old_size            = it->second.size;
                const bool   was_device_resident = !it->second.host_resident;
                bool         use_host_fallback   = false;
                it->second.pinned                = true;
                while (used_.load() - old_size + size > budget_) {
                    if (evict_one(size) == 0) {
                        GGML_SYCL_DEBUG(
                            "[UNIFIED-CACHE] Cannot evict for realloc (used=%.1f MB, need=%.1f MB), trying host "
                            "fallback\n",
                            used_.load() / (1024.0f * 1024.0f), size / (1024.0f * 1024.0f));
                        use_host_fallback = true;
                        break;
                    }
                }

                // Allocate new buffer with correct size
                void *         new_device_ptr   = nullptr;
                bool           is_host_resident = false;
                cache_location new_location     = cache_location::DEVICE;

                if (!use_host_fallback) {
                    try {
                        new_device_ptr = ggml_sycl_malloc_device_raw(size, queue_, "unified_cache:realloc");
                    } catch (const sycl::exception & e) {
                        GGML_SYCL_DEBUG("[UNIFIED-CACHE] realloc malloc_device failed: %s, trying host fallback\n",
                                        e.what());
                        use_host_fallback = true;
                    }

                    if (!new_device_ptr && !use_host_fallback) {
                        GGML_SYCL_DEBUG(
                            "[UNIFIED-CACHE] realloc malloc_device returned nullptr, trying host fallback\n");
                        use_host_fallback = true;
                    }
                }

                // Host fallback for realloc
                if (use_host_fallback) {
                    void * host_ptr = host_zone_alloc(host_zone_id::WEIGHT, size, 256);
                    if (host_ptr) {
                        std::memcpy(host_ptr, src_ptr, size);

                        GGML_SYCL_DEBUG(
                            "[UNIFIED-CACHE] Realloc to host-resident for model=%llu name_hash=0x%llx (%.2f MB)\n",
                            (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash,
                            size / (1024.0f * 1024.0f));

                        new_device_ptr   = host_ptr;
                        is_host_resident = true;
                        new_location     = cache_location::HOST_PINNED;
                    }

                    if (!new_device_ptr) {
                        GGML_LOG_ERROR("[UNIFIED-CACHE] Both device and host realloc failed\n");
                        it->second.pinned = was_pinned;
                        return nullptr;
                    }
                }

                // Release old buffer after new allocation succeeds (only if it was on device)
                if (!it->second.host_resident && it->second.device_ptr) {
                    enqueue_deferred_free(it->second.device_ptr, it->second.size);
                    // Device pointer freed — baked graph pointers to this entry are now stale
                    has_evictions_.store(true, std::memory_order_release);
                }

                // Copy new data (only if on device, host buffer already filled above)
                if (!is_host_resident) {
                    sycl::event copy_evt       = copy_to_device(new_device_ptr, src_ptr, size);
                    it->second.has_ready_event = true;
                    it->second.ready_event     = copy_evt;
                    it->second.state           = cache_entry_state::IN_PROGRESS;
                }

                // Update entry with new allocation
                it->second.device_ptr    = new_device_ptr;
                it->second.size          = size;
                it->second.content_hash  = new_hash;
                it->second.src_ptr       = src_ptr;
                it->second.host_resident = is_host_resident;
                it->second.location      = new_location;
                if (!is_host_resident) {
                    if (size > old_size) {
                        used_.fetch_add(size - old_size, std::memory_order_relaxed);
                    } else if (old_size > size) {
                        saturating_sub_used(old_size - size);
                    }
                } else if (was_device_resident) {
                    // Migrated from device to host, reduce device usage
                    saturating_sub_used(old_size);
                }
                it->second.pinned = was_pinned;
            } else if (content_changed) {
                // Same size but content changed - just re-upload
                GGML_SYCL_DEBUG(
                    "[UNIFIED-CACHE] Content changed for model=%llu name_hash=0x%llx (hash %llx -> %llx), "
                    "re-uploading\n",
                    (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash,
                    (unsigned long long) it->second.content_hash, (unsigned long long) new_hash);
                sycl::event copy_evt       = copy_to_device(it->second.device_ptr, src_ptr, size);
                it->second.has_ready_event = true;
                it->second.ready_event     = copy_evt;
                it->second.state           = cache_entry_state::IN_PROGRESS;
                it->second.content_hash    = new_hash;
                it->second.src_ptr         = src_ptr;
            } else {
                // Same content from different pointer - just update src_ptr
                it->second.src_ptr = src_ptr;
            }
        }
        hits_++;
        // Update access stats
        it->second.access_count++;
        it->second.last_access = time_++;
        return it->second.device_ptr;
    }

    misses_++;

    // Need to allocate - check if we have space
    bool use_host_fallback = false;
    while (used_.load() + size > budget_) {
        // Need to evict
        if (evict_one(size) == 0) {
            // All entries pinned, cannot evict - try host fallback
            GGML_SYCL_DEBUG(
                "[UNIFIED-CACHE] Cannot evict: all entries pinned (used=%.1f MB, need=%.1f MB), trying host fallback\n",
                used_.load() / (1024.0f * 1024.0f), size / (1024.0f * 1024.0f));
            use_host_fallback = true;
            break;
        }
    }

    // Allocate device memory (unless we need host fallback)
    void *         device_ptr       = nullptr;
    bool           is_host_resident = false;
    cache_location entry_location   = cache_location::DEVICE;
    bool           has_copy_event   = false;
    sycl::event    copy_evt;

    if (!use_host_fallback) {
        try {
            device_ptr = ggml_sycl_malloc_device_raw(size, queue_, "unified_cache:alloc");
        } catch (const sycl::exception & e) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] malloc_device failed: %s, trying host fallback\n", e.what());
            use_host_fallback = true;
        }

        if (!device_ptr && !use_host_fallback) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] malloc_device returned nullptr, trying host fallback\n");
            use_host_fallback = true;
        }
    }

    // Host fallback when device allocation fails or eviction is impossible
    if (use_host_fallback) {
        void * host_ptr = host_zone_alloc(host_zone_id::WEIGHT, size, 256);
        if (host_ptr) {
            std::memcpy(host_ptr, src_ptr, size);

            GGML_SYCL_DEBUG(
                "[UNIFIED-CACHE] Device full, using host-resident for model=%llu name_hash=0x%llx (%.2f MB)\n",
                (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash,
                size / (1024.0f * 1024.0f));

            device_ptr       = host_ptr;
            is_host_resident = true;
            entry_location   = cache_location::HOST_PINNED;
        }

        if (!device_ptr) {
            GGML_LOG_ERROR("[UNIFIED-CACHE] Both device and host allocation failed\n");
            return nullptr;
        }
    } else {
        // Copy data from source to device — deferred completion via ready_event
        has_copy_event = true;
        copy_evt       = copy_to_device(device_ptr, src_ptr, size);
    }

    // Compute content hash for new entry (only computed once on cache miss)
    uint64_t content_hash = compute_content_hash(src_ptr, size);

    // Create cache entry
    unified_cache_entry entry{};
    entry.device_ptr   = device_ptr;
    entry.src_ptr      = src_ptr;       // Track source for change detection
    entry.content_hash = content_hash;  // Track content for change detection
    entry.size         = size;
    entry.type         = type;
    entry.layer_id     = layer_id;
    entry.expert_id    = expert_id;
    entry.layout       = layout;
    entry.access_count = 1;
    entry.last_access  = time_++;
    entry.pinned       = false;
    entry.hot          = false;
    if (has_copy_event) {
        entry.state           = cache_entry_state::IN_PROGRESS;
        entry.has_ready_event = true;
        entry.ready_event     = copy_evt;
    } else {
        entry.state           = cache_entry_state::READY;
        entry.has_ready_event = false;
    }
    entry.host_resident = is_host_resident;
    entry.location      = entry_location;
    // NOTE: Reorder state is tracked in tensor->extra->optimized_feature, not here

    // Store in cache
    entries_[key] = entry;
    auto id_it    = id_to_key_.find(key_id);
    if (id_it == id_to_key_.end()) {
        id_to_key_.emplace(key_id, key);
    } else if (!(id_it->second == key)) {
        GGML_LOG_ERROR("[UNIFIED-CACHE] identity collision on insert model=%llu name_hash=0x%llx\n",
                       (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash);
        if (cache_assert_enabled()) {
            GGML_ABORT("unified_cache id_to_key mismatch");
        }
    }

    // Only track device memory usage, not host-resident entries
    if (!is_host_resident) {
        used_.fetch_add(size, std::memory_order_relaxed);
    }

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] Cached %s%s: %.2f MB (used=%.1f/%.1f MB)\n",
                    type == cache_entry_type::DENSE_WEIGHT ? "dense" : "expert",
                    is_host_resident ? " (host-resident)" : "", size / (1024.0f * 1024.0f),
                    used_.load() / (1024.0f * 1024.0f), budget_ / (1024.0f * 1024.0f));

    return device_ptr;
}

// === Direct Staging API ===

direct_stage_result unified_cache::direct_stage_weight(ggml_sycl_cache_id   key,
                                                       const void *         src_ptr,
                                                       size_t               src_size,
                                                       size_t               dst_size,
                                                       ggml_layout_mode     layout,
                                                       cache_layout_fill_fn fill_fn,
                                                       const void *         fill_ctx,
                                                       sycl::queue *        queue) {
    direct_stage_result result{};
    if (!key.valid || !src_ptr || src_size == 0 || dst_size == 0) {
        return result;
    }
    if (!queue) {
        GGML_LOG_ERROR("[DIRECT-STAGE] null queue for weight staging\n");
        return result;
    }

    {
        std::shared_lock<std::shared_mutex> lock(rw_mutex_);
        const unified_cache_key             cache_key{ cache_entry_type::DENSE_WEIGHT, key, -1, -1 };
        auto                                it = entries_.find(cache_key);
        if (it != entries_.end() && it->second.device_ptr && it->second.layout == layout &&
            it->second.location == cache_location::DEVICE) {
            result.ptr = it->second.device_ptr;
            if (it->second.has_ready_event) {
                result.event = it->second.ready_event;
            }
            result.ok = true;
            return result;
        }
    }

    // 1. Allocate from WEIGHT zone
    void * ptr = zone_alloc(vram_zone_id::WEIGHT, dst_size);
    if (!ptr) {
        GGML_LOG_ERROR("[DIRECT-STAGE] weight zone_alloc failed for %zu bytes\n", dst_size);
        return result;
    }

    // 2. Fill: reorder or plain memcpy
    sycl::event fill_event;
    if (fill_fn) {
        fill_event = fill_fn(*queue, ptr, dst_size, src_ptr, src_size, fill_ctx, {});
    } else {
        fill_event = queue->memcpy(ptr, src_ptr, src_size);
    }

    // 3. Zero-fill padding if dst_size > src_size
    sycl::event last_event = fill_event;
    if (dst_size > src_size && !fill_fn) {
        last_event = queue->submit([&](sycl::handler & cgh) {
            cgh.depends_on(fill_event);
            cgh.memset(static_cast<char *>(ptr) + src_size, 0, dst_size - src_size);
        });
    }

    // 4. Store in lookup table (keyed by full cache_id for collision safety)
    {
        std::unique_lock<std::shared_mutex> lock(direct_stage_mutex_);
        weight_entry                        entry{};
        entry.ptr                   = ptr;
        entry.size                  = dst_size;
        entry.layout                = layout;
        entry.location              = cache_location::DEVICE;
        entry.has_ready_event       = true;
        entry.ready_event           = last_event;
        direct_weight_entries_[key] = std::move(entry);
    }
    {
        std::unique_lock<std::shared_mutex> lock(rw_mutex_);
        unified_cache_key                   cache_key{ cache_entry_type::DENSE_WEIGHT, key, -1, -1 };
        unified_cache_entry                 entry{};
        entry.device_ptr       = ptr;
        entry.src_ptr          = src_ptr;
        entry.content_hash     = 0;
        entry.size             = dst_size;
        entry.type             = cache_entry_type::DENSE_WEIGHT;
        entry.layer_id         = -1;
        entry.expert_id        = -1;
        entry.layout           = layout;
        entry.access_count     = 0;
        entry.last_access      = time_++;
        entry.pinned           = true;
        entry.hot              = true;
        entry.state            = cache_entry_state::IN_PROGRESS;
        entry.has_ready_event  = true;
        entry.ready_event      = last_event;
        entry.host_resident    = false;
        entry.location         = cache_location::DEVICE;
        entry.pool_allocated   = false;
        entry.last_write_event = last_event;
        entry.has_write_event  = true;
        auto old               = entries_.find(cache_key);
        if (old != entries_.end() && old->second.device_ptr && old->second.device_ptr != ptr) {
            const uint32_t live = old->second.in_use_count.load();
            if (live != 0) {
                GGML_LOG_WARN(
                    "[DIRECT-STAGE] replacing dense entry with live leases model=%llu name_hash=0x%llx leases=%u\n",
                    (unsigned long long) key.model_id, (unsigned long long) key.name_hash, live);
            } else if (old->second.host_resident || old->second.location == cache_location::HOST_PINNED) {
                if (host_zones_configured()) {
                    host_zone_free(host_zone_id::WEIGHT, old->second.device_ptr);
                } else {
                    host_pool_free(old->second.device_ptr, old->second.size);
                }
            } else if (vram_owns(old->second.device_ptr)) {
                zone_free(vram_zone_id::WEIGHT, old->second.device_ptr);
            } else if (layout_pool_ && layout_pool_->owns(old->second.device_ptr)) {
                saturating_sub_used(old->second.size);
            } else {
                enqueue_deferred_free(old->second.device_ptr, old->second.size);
            }
        }
        entries_[cache_key] = entry;
        id_to_key_[key]     = cache_key;
    }

    result.ptr   = ptr;
    result.event = last_event;
    result.ok    = true;
    return result;
}

static bool moe_direct_trace_enabled();
static void moe_direct_trace_key(const char *               op,
                                 const ggml_sycl_cache_id & key,
                                 ggml_layout_mode           layout,
                                 const char *               detail,
                                 size_t                     direct_entries,
                                 const weight_entry *       entry);

direct_stage_result unified_cache::direct_stage_expert(ggml_sycl_cache_id   key,
                                                       const void *         src_ptr,
                                                       size_t               src_size,
                                                       size_t               dst_size,
                                                       ggml_layout_mode     layout,
                                                       cache_layout_fill_fn fill_fn,
                                                       const void *         fill_ctx,
                                                       sycl::queue *        queue) {
    direct_stage_result result{};
    if (!key.valid || !src_ptr || src_size == 0 || dst_size == 0) {
        return result;
    }
    if (!queue) {
        GGML_LOG_ERROR("[DIRECT-STAGE] null queue for expert staging\n");
        return result;
    }

    {
        std::shared_lock<std::shared_mutex> lock(rw_mutex_);
        const unified_cache_key             cache_key{ cache_entry_type::MOE_EXPERT, key, -1, -1 };
        auto                                it = entries_.find(cache_key);
        if (it != entries_.end() && it->second.device_ptr && it->second.layout == layout &&
            it->second.location == cache_location::DEVICE) {
            result.ptr = it->second.device_ptr;
            if (it->second.has_ready_event) {
                result.event = it->second.ready_event;
            }
            result.ok = true;
            return result;
        }
    }

    if (has_placement_plan() && unified_cache_is_graph_compute_active()) {
        GGML_LOG_ERROR(
            "[DIRECT-STAGE] inference-time expert staging is forbidden when placement-plan preload is active\n");
        GGML_ASSERT(false && "direct_stage_expert called during inference with placement plan");
        return result;
    }

    // 1. Allocate from WEIGHT zone
    void * ptr = zone_alloc(vram_zone_id::WEIGHT, dst_size);
    if (!ptr) {
        if (has_placement_plan() && layout != GGML_LAYOUT_AOS) {
            static std::atomic<int> planned_stage_fail_log{ 0 };
            if (planned_stage_fail_log.fetch_add(1, std::memory_order_relaxed) < 10) {
                size_t entry_count         = 0;
                size_t direct_expert_count = 0;
                {
                    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
                    entry_count = entries_.size();
                }
                {
                    std::shared_lock<std::shared_mutex> lock(direct_stage_mutex_);
                    direct_expert_count = direct_expert_entries_.size();
                }
                GGML_LOG_WARN(
                    "[DIRECT-STAGE] WEIGHT zone full (%zu bytes) for planned layout=%d expert; refusing host AoS "
                    "fallback (used=%.1f MB avail=%.1f MB largest=%.1f MB entries=%zu direct_experts=%zu)\n",
                    dst_size, (int) layout, zone_used(vram_zone_id::WEIGHT) / (1024.0 * 1024.0),
                    zone_available(vram_zone_id::WEIGHT) / (1024.0 * 1024.0),
                    zone_largest_free(vram_zone_id::WEIGHT) / (1024.0 * 1024.0), entry_count, direct_expert_count);
            }
            return result;
        }

        static std::atomic<int> host_fallback_log{ 0 };
        if (host_fallback_log.fetch_add(1, std::memory_order_relaxed) < 10) {
            GGML_LOG_WARN("[DIRECT-STAGE] WEIGHT zone full (%zu bytes) — host arena fallback\n", dst_size);
        }

        const sycl::context & ctx   = queue->get_context();
        sycl::usm::alloc      atype = sycl::get_pointer_type(src_ptr, ctx);

        void *         host_ptr = nullptr;
        cache_location loc      = cache_location::HOST_MMAP;

        if (atype == sycl::usm::alloc::host) {
            host_ptr = const_cast<void *>(src_ptr);
            loc      = cache_location::HOST_PINNED;
        } else {
            host_ptr = host_zone_alloc(host_zone_id::WEIGHT, src_size, 256);
            if (host_ptr) {
                std::memcpy(host_ptr, src_ptr, src_size);
                loc = cache_location::HOST_PINNED;
            } else {
                host_ptr = const_cast<void *>(src_ptr);
                loc      = cache_location::HOST_MMAP;
                GGML_LOG_WARN("[DIRECT-STAGE] host arena exhausted, raw mmap fallback\n");
            }
        }

        {
            std::unique_lock<std::shared_mutex> lock(rw_mutex_);
            unified_cache_key                   cache_key{ cache_entry_type::MOE_EXPERT, key, -1, -1 };
            unified_cache_entry                 entry{};
            entry.device_ptr       = host_ptr;
            entry.src_ptr          = src_ptr;
            entry.content_hash     = 0;
            entry.size             = src_size;
            entry.type             = cache_entry_type::MOE_EXPERT;
            entry.layer_id         = -1;
            entry.expert_id        = -1;
            entry.layout           = GGML_LAYOUT_AOS;
            entry.access_count     = 0;
            entry.last_access      = time_++;
            entry.pinned           = true;
            entry.hot              = true;
            entry.state            = cache_entry_state::READY;
            entry.has_ready_event  = false;
            entry.host_resident    = loc != cache_location::DEVICE;
            entry.location         = loc;
            entry.pool_allocated   = false;
            entry.last_write_event = {};
            entry.has_write_event  = false;
            entries_[cache_key]    = entry;
            id_to_key_[key]        = cache_key;
        }
        {
            std::unique_lock<std::shared_mutex> lock(direct_stage_mutex_);
            weight_entry                        entry{};
            entry.ptr                   = host_ptr;
            entry.size                  = src_size;
            entry.layout                = GGML_LAYOUT_AOS;
            entry.location              = loc;
            direct_expert_entries_[key] = std::move(entry);
            moe_direct_trace_key("insert-host", key, GGML_LAYOUT_AOS, "zone-full", direct_expert_entries_.size(),
                                 &direct_expert_entries_.find(key)->second);
        }
        result.ptr = host_ptr;
        result.ok  = true;
        return result;
    }

    // 2. Fill: reorder or plain memcpy
    sycl::event fill_event;
    if (fill_fn) {
        fill_event = fill_fn(*queue, ptr, dst_size, src_ptr, src_size, fill_ctx, {});
    } else {
        fill_event = queue->memcpy(ptr, src_ptr, src_size);
    }

    // 3. Zero-fill padding if dst_size > src_size
    sycl::event last_event = fill_event;
    if (dst_size > src_size && !fill_fn) {
        last_event = queue->submit([&](sycl::handler & cgh) {
            cgh.depends_on(fill_event);
            cgh.memset(static_cast<char *>(ptr) + src_size, 0, dst_size - src_size);
        });
    }

    // 4. Store in lookup table (keyed by full cache_id for collision safety)
    {
        std::unique_lock<std::shared_mutex> lock(direct_stage_mutex_);
        weight_entry                        entry{};
        entry.ptr                   = ptr;
        entry.size                  = dst_size;
        entry.layout                = layout;
        entry.location              = cache_location::DEVICE;
        entry.has_ready_event       = true;
        entry.ready_event           = last_event;
        direct_expert_entries_[key] = std::move(entry);
        moe_direct_trace_key("insert-device", key, layout, "", direct_expert_entries_.size(),
                             &direct_expert_entries_.find(key)->second);
    }
    {
        std::unique_lock<std::shared_mutex> lock(rw_mutex_);
        unified_cache_key                   cache_key{ cache_entry_type::MOE_EXPERT, key, -1, -1 };
        unified_cache_entry                 entry{};
        entry.device_ptr       = ptr;
        entry.src_ptr          = src_ptr;
        entry.content_hash     = 0;
        entry.size             = dst_size;
        entry.type             = cache_entry_type::MOE_EXPERT;
        entry.layer_id         = -1;
        entry.expert_id        = -1;
        entry.layout           = layout;
        entry.access_count     = 0;
        entry.last_access      = time_++;
        entry.pinned           = true;
        entry.hot              = true;
        entry.state            = cache_entry_state::IN_PROGRESS;
        entry.has_ready_event  = true;
        entry.ready_event      = last_event;
        entry.host_resident    = false;
        entry.location         = cache_location::DEVICE;
        entry.pool_allocated   = false;
        entry.last_write_event = last_event;
        entry.has_write_event  = true;
        auto old               = entries_.find(cache_key);
        if (old != entries_.end() && old->second.device_ptr && old->second.device_ptr != ptr) {
            const uint32_t live = old->second.in_use_count.load();
            if (live != 0) {
                GGML_LOG_WARN(
                    "[DIRECT-STAGE] replacing expert entry with live leases model=%llu name_hash=0x%llx leases=%u\n",
                    (unsigned long long) key.model_id, (unsigned long long) key.name_hash, live);
            } else if (old->second.host_resident || old->second.location == cache_location::HOST_PINNED) {
                if (host_zones_configured()) {
                    host_zone_free(host_zone_id::WEIGHT, old->second.device_ptr);
                } else {
                    host_pool_free(old->second.device_ptr, old->second.size);
                }
            } else if (vram_owns(old->second.device_ptr)) {
                zone_free(vram_zone_id::WEIGHT, old->second.device_ptr);
            } else if (layout_pool_ && layout_pool_->owns(old->second.device_ptr)) {
                saturating_sub_used(old->second.size);
            } else {
                enqueue_deferred_free(old->second.device_ptr, old->second.size);
            }
        }
        entries_[cache_key] = entry;
        id_to_key_[key]     = cache_key;
    }

    result.ptr   = ptr;
    result.event = last_event;
    result.ok    = true;
    return result;
}

static bool moe_direct_trace_enabled() {
    static std::atomic<int> cached{ -1 };
    int                     v = cached.load(std::memory_order_acquire);
    if (v < 0) {
        const char * env_route  = std::getenv("GGML_SYCL_MOE_ROUTE_LOG");
        const char * env_direct = std::getenv("GGML_SYCL_MOE_DIRECT_LOG");
        v = ((env_route && std::atoi(env_route) != 0) || (env_direct && std::atoi(env_direct) != 0)) ? 1 : 0;
        cached.store(v, std::memory_order_release);
    }
    return v != 0;
}

static void moe_direct_trace_key(const char *               op,
                                 const ggml_sycl_cache_id & key,
                                 ggml_layout_mode           layout,
                                 const char *               detail,
                                 size_t                     direct_entries,
                                 const weight_entry *       entry) {
    if (!moe_direct_trace_enabled()) {
        return;
    }
    bool         force_log = false;
    const char * hash_env  = std::getenv("GGML_SYCL_MOE_DIRECT_HASH");
    if (hash_env && hash_env[0] != '\0') {
        char *   end  = nullptr;
        uint64_t want = std::strtoull(hash_env, &end, 0);
        force_log     = end != hash_env && want == key.name_hash;
    }
    static std::atomic<int> log_count{ 0 };
    if (!force_log && log_count.fetch_add(1, std::memory_order_relaxed) >= 400) {
        return;
    }
    fprintf(stderr,
            "[MOE-DIRECT] op=%s model=%llu name_hash=0x%llx aux=%llu type=%d ne=[%lld,%lld,%lld,%lld] "
            "layout=%d entries=%zu %s ptr=%p entry_layout=%d loc=%d ready=%d\n",
            op ? op : "?", (unsigned long long) key.model_id, (unsigned long long) key.name_hash,
            (unsigned long long) key.aux_id, (int) key.type, (long long) key.ne[0], (long long) key.ne[1],
            (long long) key.ne[2], (long long) key.ne[3], (int) layout, direct_entries, detail ? detail : "",
            entry ? entry->ptr : nullptr, entry ? (int) entry->layout : -1, entry ? (int) entry->location : -1,
            entry ? (entry->has_ready_event ? 1 : 0) : 0);
}

const weight_entry * unified_cache::lookup_weight(ggml_sycl_cache_id key) const {
    std::shared_lock<std::shared_mutex> lock(direct_stage_mutex_);
    auto                                it = direct_weight_entries_.find(key);
    if (it == direct_weight_entries_.end()) {
        return nullptr;
    }
    return &it->second;
}

const weight_entry * unified_cache::lookup_expert(ggml_sycl_cache_id key) const {
    std::shared_lock<std::shared_mutex> lock(direct_stage_mutex_);
    auto                                it = direct_expert_entries_.find(key);
    if (it == direct_expert_entries_.end()) {
        moe_direct_trace_key("lookup-miss", key, GGML_LAYOUT_AOS, "", direct_expert_entries_.size(), nullptr);
        return nullptr;
    }
    moe_direct_trace_key("lookup-hit", key, it->second.layout, "", direct_expert_entries_.size(), &it->second);
    return &it->second;
}

static expert_resolve_tier expert_tier_from_location(cache_location location) {
    switch (location) {
        case cache_location::DEVICE:
            return expert_resolve_tier::DEVICE_VRAM;
        case cache_location::HOST_PINNED:
            return expert_resolve_tier::HOST_PINNED;
        case cache_location::HOST_MMAP:
            return expert_resolve_tier::HOST_MMAP;
        case cache_location::UNKNOWN:
        default:
            return expert_resolve_tier::UNAVAILABLE;
    }
}

static bool expert_resolution_allowed(const expert_resolve_request & req,
                                      cache_location                 location,
                                      int                            owning_device,
                                      expert_resolve_reason *        out_reason) {
    if (out_reason) {
        *out_reason = expert_resolve_reason::FOUND;
    }

    if (location == cache_location::HOST_PINNED || location == cache_location::HOST_MMAP) {
        if (!req.allow_host) {
            if (out_reason) {
                *out_reason = expert_resolve_reason::HOST_DISALLOWED;
            }
            return false;
        }
        if (location == cache_location::HOST_MMAP && !req.allow_mmap_host) {
            if (out_reason) {
                *out_reason = expert_resolve_reason::MMAP_HOST_DISALLOWED;
            }
            return false;
        }
        return true;
    }

    if (location == cache_location::DEVICE) {
        if (req.device_policy == expert_resolve_device_policy::CURRENT_ONLY && req.current_device >= 0 &&
            owning_device != req.current_device) {
            if (out_reason) {
                *out_reason = expert_resolve_reason::DEVICE_MISMATCH;
            }
            return false;
        }
        if (req.device_policy == expert_resolve_device_policy::PREFERRED_ONLY && req.preferred_device >= 0 &&
            owning_device != req.preferred_device) {
            if (out_reason) {
                *out_reason = expert_resolve_reason::DEVICE_MISMATCH;
            }
            return false;
        }
        return true;
    }

    if (out_reason) {
        *out_reason = expert_resolve_reason::NOT_FOUND;
    }
    return false;
}

expert_resolve_result unified_cache::resolve_expert(const expert_resolve_request & req) {
    expert_resolve_result result{};
    result.reason = expert_resolve_reason::NOT_FOUND;

    if (!req.key.valid) {
        result.reason = expert_resolve_reason::INVALID_REQUEST;
        return result;
    }

    const int cache_device = ggml_sycl_get_device_id_from_queue(queue_);
    auto      apply_miss   = [&result](expert_resolve_reason reason) {
        if (result.reason == expert_resolve_reason::NOT_FOUND) {
            result.reason = reason;
        }
    };

    {
        std::shared_lock<std::shared_mutex> lock(rw_mutex_);
        auto                                entry_it = entries_.end();
        const unified_cache_key explicit_key{ cache_entry_type::MOE_EXPERT, req.key, req.layer_id, req.expert_id };
        entry_it = entries_.find(explicit_key);
        if (entry_it == entries_.end()) {
            auto id_it = id_to_key_.find(req.key);
            if (id_it != id_to_key_.end() && id_it->second.type == cache_entry_type::MOE_EXPERT) {
                entry_it = entries_.find(id_it->second);
            }
        }

        if (entry_it != entries_.end()) {
            auto & entry                     = entry_it->second;
            bool   entry_ready               = entry.state == cache_entry_state::READY;
            bool   entry_in_progress_allowed = entry.state == cache_entry_state::IN_PROGRESS && !req.require_ready;
            if (entry.layout != req.requested_layout) {
                apply_miss(expert_resolve_reason::LAYOUT_MISMATCH);
            } else if (!entry.device_ptr) {
                apply_miss(expert_resolve_reason::NOT_FOUND);
            } else if (entry.state == cache_entry_state::EVICTING || entry.state == cache_entry_state::FAILED) {
                apply_miss(expert_resolve_reason::NOT_READY);
            } else if (entry.state == cache_entry_state::IN_PROGRESS && req.require_ready) {
                entry_ready = entry.has_ready_event && event_complete(entry.ready_event);
                if (!entry_ready) {
                    apply_miss(expert_resolve_reason::NOT_READY);
                }
            }

            if ((entry_ready || entry_in_progress_allowed) && entry.device_ptr &&
                entry.layout == req.requested_layout) {
                const cache_location  location      = entry.location;
                const bool            on_device     = location == cache_location::DEVICE;
                const int             owner         = on_device ? cache_device : mem_handle::HOST_DEVICE;
                expert_resolve_reason reject_reason = expert_resolve_reason::FOUND;
                if (expert_resolution_allowed(req, location, owner, &reject_reason)) {
                    entry.in_use_count.fetch_add(1);
                    result.ptr             = entry.device_ptr;
                    result.size            = entry.size;
                    result.tier            = expert_tier_from_location(location);
                    result.location        = location;
                    result.owning_device   = owner;
                    result.actual_layout   = entry.layout;
                    result.cpu_accessible  = !on_device;
                    result.has_ready_event = entry.state == cache_entry_state::IN_PROGRESS && entry.has_ready_event;
                    if (result.has_ready_event) {
                        result.ready_event = entry.ready_event;
                    }
                    result.reason   = expert_resolve_reason::FOUND;
                    result.lifetime = std::make_unique<mem_handle>(mem_handle::from_weight_lease(
                        entry_it->first, owner, entry.device_ptr, entry.layout, on_device, &entry));
                    return result;
                }
                apply_miss(reject_reason);
            }
        }
    }

    {
        std::shared_lock<std::shared_mutex> lock(direct_stage_mutex_);
        auto                                it = direct_expert_entries_.find(req.key);
        if (it == direct_expert_entries_.end()) {
            moe_direct_trace_key("resolve-direct-miss", req.key, req.requested_layout, "",
                                 direct_expert_entries_.size(), nullptr);
            return result;
        }

        const weight_entry & entry = it->second;
        moe_direct_trace_key("resolve-direct-hit", req.key, req.requested_layout, "", direct_expert_entries_.size(),
                             &entry);
        if (entry.layout != req.requested_layout) {
            result.reason = expert_resolve_reason::LAYOUT_MISMATCH;
            return result;
        }
        if (!entry.ptr) {
            result.reason = expert_resolve_reason::NOT_FOUND;
            return result;
        }
        if (req.require_ready && entry.has_ready_event && !event_complete(entry.ready_event)) {
            result.reason = expert_resolve_reason::NOT_READY;
            return result;
        }

        const bool            on_device     = entry.location == cache_location::DEVICE;
        const int             owner         = on_device ? cache_device : mem_handle::HOST_DEVICE;
        expert_resolve_reason reject_reason = expert_resolve_reason::FOUND;
        result.tier                         = expert_tier_from_location(entry.location);
        result.location                     = entry.location;
        result.owning_device                = owner;
        result.actual_layout                = entry.layout;
        result.cpu_accessible               = !on_device;
        result.has_ready_event              = entry.has_ready_event;
        if (entry.has_ready_event) {
            result.ready_event = entry.ready_event;
        }
        if (!expert_resolution_allowed(req, entry.location, owner, &reject_reason)) {
            result.reason = reject_reason;
            return result;
        }

        void *                 entry_ptr             = entry.ptr;
        const size_t           entry_size            = entry.size;
        const cache_location   entry_location        = entry.location;
        const ggml_layout_mode entry_layout          = entry.layout;
        const bool             entry_has_ready_event = entry.has_ready_event;
        unified_cache_key      mirror_key{ cache_entry_type::MOE_EXPERT, req.key, -1, -1 };
        lock.unlock();

        auto lease = acquire_entry_lease(mirror_key);
        if (lease) {
            result.ptr             = lease.ptr;
            result.size            = entry_size;
            result.tier            = expert_tier_from_location(entry_location);
            result.location        = entry_location;
            result.owning_device   = owner;
            result.actual_layout   = lease.layout;
            result.cpu_accessible  = !lease.on_device;
            result.has_ready_event = entry_has_ready_event;
            result.reason          = expert_resolve_reason::FOUND;
            result.lifetime        = std::make_unique<mem_handle>(mem_handle::from_weight_lease(
                mirror_key, owner, lease.ptr, lease.layout, lease.on_device, lease.entry));
            return result;
        }

        if (on_device) {
            static std::atomic<int> unowned_direct_device_log{ 0 };
            if (unowned_direct_device_log.fetch_add(1, std::memory_order_relaxed) < 10) {
                GGML_LOG_WARN(
                    "[MOE-DIRECT] refusing unleased device direct entry model=%llu name_hash=0x%llx; "
                    "canonical unified-cache entry is missing\n",
                    (unsigned long long) req.key.model_id, (unsigned long long) req.key.name_hash);
            }
            result.reason = expert_resolve_reason::NOT_FOUND;
            return result;
        }

        static std::atomic<int> unowned_direct_host_log{ 0 };
        if (unowned_direct_host_log.fetch_add(1, std::memory_order_relaxed) < 10) {
            GGML_LOG_WARN(
                "[MOE-DIRECT] refusing unleased host direct entry model=%llu name_hash=0x%llx; "
                "canonical unified-cache entry is missing\n",
                (unsigned long long) req.key.model_id, (unsigned long long) req.key.name_hash);
        }
        result.reason = expert_resolve_reason::NOT_FOUND;
        return result;
    }
}

void unified_cache::register_host_expert(ggml_sycl_cache_id key, void * ptr, size_t size, ggml_layout_mode layout) {
    const int       dev = ggml_sycl_get_device_id_from_queue(queue_);
    memory_location loc = query_location(ptr, dev);
    cache_location  cache_loc =
        loc.tier == alloc_tier::HOST_PINNED ? cache_location::HOST_PINNED : cache_location::HOST_MMAP;
    if (cache_loc != cache_location::HOST_PINNED) {
        try {
            const sycl::usm::alloc atype = sycl::get_pointer_type(ptr, queue_.get_context());
            if (atype == sycl::usm::alloc::host || atype == sycl::usm::alloc::shared) {
                cache_loc = cache_location::HOST_PINNED;
            }
        } catch (...) {
            cache_loc = cache_location::HOST_MMAP;
        }
    }

    {
        std::unique_lock<std::shared_mutex> cache_lock(rw_mutex_);
        unified_cache_key                   cache_key{ cache_entry_type::MOE_EXPERT, key, -1, -1 };
        unified_cache_entry                 cache_entry{};
        cache_entry.device_ptr       = ptr;
        cache_entry.src_ptr          = ptr;
        cache_entry.content_hash     = 0;
        cache_entry.size             = size;
        cache_entry.type             = cache_entry_type::MOE_EXPERT;
        cache_entry.layer_id         = -1;
        cache_entry.expert_id        = -1;
        cache_entry.layout           = layout;
        cache_entry.access_count     = 0;
        cache_entry.last_access      = time_++;
        cache_entry.pinned           = true;
        cache_entry.hot              = true;
        cache_entry.state            = cache_entry_state::READY;
        cache_entry.has_ready_event  = false;
        cache_entry.host_resident    = cache_loc != cache_location::DEVICE;
        cache_entry.location         = cache_loc;
        cache_entry.pool_allocated   = false;
        cache_entry.last_write_event = {};
        cache_entry.has_write_event  = false;
        entries_[cache_key]          = cache_entry;
        id_to_key_[key]              = cache_key;
    }

    std::unique_lock<std::shared_mutex> lock(direct_stage_mutex_);
    weight_entry                        entry{};
    entry.ptr                   = ptr;
    entry.size                  = size;
    entry.layout                = layout;
    entry.location              = cache_loc;
    direct_expert_entries_[key] = std::move(entry);
}

void unified_cache::register_host_weight(ggml_sycl_cache_id key, void * ptr, size_t size, ggml_layout_mode layout) {
    const int       dev = ggml_sycl_get_device_id_from_queue(queue_);
    memory_location loc = query_location(ptr, dev);
    cache_location  cache_loc =
        loc.tier == alloc_tier::HOST_PINNED ? cache_location::HOST_PINNED : cache_location::HOST_MMAP;
    if (cache_loc != cache_location::HOST_PINNED) {
        try {
            const sycl::usm::alloc atype = sycl::get_pointer_type(ptr, queue_.get_context());
            if (atype == sycl::usm::alloc::host || atype == sycl::usm::alloc::shared) {
                cache_loc = cache_location::HOST_PINNED;
            }
        } catch (...) {
            cache_loc = cache_location::HOST_MMAP;
        }
    }

    {
        std::unique_lock<std::shared_mutex> cache_lock(rw_mutex_);
        unified_cache_key                   cache_key{ cache_entry_type::DENSE_WEIGHT, key, -1, -1 };
        unified_cache_entry                 cache_entry{};
        cache_entry.device_ptr       = ptr;
        cache_entry.src_ptr          = ptr;
        cache_entry.content_hash     = 0;
        cache_entry.size             = size;
        cache_entry.type             = cache_entry_type::DENSE_WEIGHT;
        cache_entry.layer_id         = -1;
        cache_entry.expert_id        = -1;
        cache_entry.layout           = layout;
        cache_entry.access_count     = 0;
        cache_entry.last_access      = time_++;
        cache_entry.pinned           = true;
        cache_entry.hot              = true;
        cache_entry.state            = cache_entry_state::READY;
        cache_entry.has_ready_event  = false;
        cache_entry.host_resident    = cache_loc != cache_location::DEVICE;
        cache_entry.location         = cache_loc;
        cache_entry.pool_allocated   = false;
        cache_entry.last_write_event = {};
        cache_entry.has_write_event  = false;
        entries_[cache_key]          = cache_entry;
        id_to_key_[key]              = cache_key;
    }

    std::unique_lock<std::shared_mutex> lock(direct_stage_mutex_);
    weight_entry                        entry{};
    entry.ptr                   = ptr;
    entry.size                  = size;
    entry.layout                = layout;
    entry.location              = cache_loc;
    direct_weight_entries_[key] = std::move(entry);
}

bool unified_cache::is_cached(const ggml_sycl_cache_id & key_id, ggml_layout_mode layout) const {
    if (!key_id.valid) {
        return false;
    }
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    auto                                id_it = id_to_key_.find(key_id);
    if (id_it == id_to_key_.end()) {
        return false;
    }
    auto entry_it = entries_.find(id_it->second);
    if (entry_it == entries_.end()) {
        return false;
    }
    if (entry_it->second.layout != layout) {
        return false;
    }
    return true;
}

bool unified_cache::is_cached_any(const ggml_sycl_cache_id & key_id) const {
    if (!key_id.valid) {
        return false;
    }
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    auto                                id_it = id_to_key_.find(key_id);
    if (id_it == id_to_key_.end()) {
        return false;
    }
    return entries_.find(id_it->second) != entries_.end();
}

void * unified_cache::get(const ggml_sycl_cache_id & key_id, ggml_layout_mode layout) {
    if (!key_id.valid) {
        return nullptr;
    }
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    auto                                id_it = id_to_key_.find(key_id);
    if (id_it == id_to_key_.end()) {
        return nullptr;
    }
    auto entry_it = entries_.find(id_it->second);
    if (entry_it == entries_.end()) {
        return nullptr;
    }
    auto & entry = entry_it->second;
    if (entry.layout != layout) {
        return nullptr;
    }
    if (entry.state == cache_entry_state::IN_PROGRESS) {
        if (entry.has_ready_event && event_complete(entry.ready_event)) {
            entry.state           = cache_entry_state::READY;
            entry.has_ready_event = false;
        } else {
            GGML_SYCL_DEBUG(
                "[UNIFIED-CACHE] get pending: model=%llu name_hash=0x%llx layout=%d size=%zu has_event=%d\n",
                (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash, (int) layout, entry.size,
                entry.has_ready_event ? 1 : 0);
            return nullptr;
        }
    }
    return entry.device_ptr;
}

void * unified_cache::try_get_cached_fast(const ggml_sycl_cache_id & key_id, ggml_layout_mode layout) {
    if (!key_id.valid) {
        return nullptr;
    }
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    auto                                id_it = id_to_key_.find(key_id);
    if (id_it == id_to_key_.end()) {
        static std::atomic<int> miss_log{ 0 };
        if (miss_log.fetch_add(1, std::memory_order_relaxed) < 3) {
            GGML_LOG_WARN(
                "[CACHE-LOOKUP] MISS: model=%llu hash=0x%llx aux=0x%llx layout=%d id_to_key_size=%zu "
                "entries_size=%zu\n",
                (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash,
                (unsigned long long) key_id.aux_id, (int) layout, id_to_key_.size(), entries_.size());
        }
        return nullptr;
    }
    auto entry_it = entries_.find(id_it->second);
    if (entry_it == entries_.end()) {
        return nullptr;
    }
    const auto & entry = entry_it->second;
    if (entry.layout != layout) {
        return nullptr;
    }
    if (entry.state != cache_entry_state::READY) {
        return nullptr;
    }
    // HOST_MMAP entries contain raw mmap pointers that are NOT GPU-accessible.
    // Returning them from lookup would cause GPU page faults when kernels
    // dereference the pointer.  Only DEVICE and HOST_PINNED (sycl::malloc_host,
    // GPU-accessible via PCIe zero-copy) entries are safe for GPU dispatch.
    if (entry.location == cache_location::HOST_MMAP) {
        return nullptr;
    }
    return entry.device_ptr;
}

void * unified_cache::try_get_cached_with_event(const ggml_sycl_cache_id & key_id,
                                                ggml_layout_mode           layout,
                                                sycl::event *              out_event,
                                                bool *                     out_has_event) {
    if (out_event) {
        *out_event = sycl::event{};
    }
    if (out_has_event) {
        *out_has_event = false;
    }
    if (!key_id.valid) {
        return nullptr;
    }
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    auto                                id_it = id_to_key_.find(key_id);
    if (id_it == id_to_key_.end()) {
        return nullptr;
    }
    auto entry_it = entries_.find(id_it->second);
    if (entry_it == entries_.end()) {
        return nullptr;
    }
    const auto & entry = entry_it->second;
    if (entry.layout != layout) {
        return nullptr;
    }
    if (entry.location == cache_location::HOST_MMAP) {
        return nullptr;
    }
    // READY entries: return pointer, no event needed.
    if (entry.state == cache_entry_state::READY) {
        return entry.device_ptr;
    }
    // IN_PROGRESS entries: return pointer + ready_event so the caller can
    // chain the subsequent kernel/memcpy after the fill completes.
    if (entry.state == cache_entry_state::IN_PROGRESS && entry.device_ptr) {
        if (entry.has_ready_event) {
            if (out_event) {
                *out_event = entry.ready_event;
            }
            if (out_has_event) {
                *out_has_event = true;
            }
        }
        return entry.device_ptr;
    }
    return nullptr;
}

// --- Decomposed cache operations ---

void * unified_cache::allocate_slot(const ggml_sycl_cache_id & key,
                                    size_t                     size,
                                    ggml_layout_mode           layout,
                                    cache_entry_type           type,
                                    int                        layer_id,
                                    int                        expert_id) {
    if (!key.valid || size == 0) {
        return nullptr;
    }

    const unified_cache_key cache_key{ type, key, layer_id, expert_id };

    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    if (!g_graph_compute_active.load(std::memory_order_acquire)) {
        process_deferred_frees();
    }

    // Check if entry already exists with matching layout and size
    auto it = entries_.find(cache_key);
    if (it != entries_.end()) {
        auto & entry = it->second;
        if (entry.layout == layout && entry.size == size && entry.device_ptr &&
            entry.location != cache_location::HOST_MMAP) {
            // Already allocated (may be READY or IN_PROGRESS) — return existing ptr.
            // HOST_MMAP entries are raw mmap pointers, not device allocations;
            // treat them as a mismatch so a real device allocation is made.
            return entry.device_ptr;
        }
        // Layout/size mismatch or HOST_MMAP — evict old entry.
        // llama.cpp-vtf7f: refuse to erase an entry that any mem_handle
        // leases.  Returning nullptr forces the caller to retry later when
        // the lease is released (same semantics as the existing OOM fallback
        // for pinned entries).
        const uint32_t alloc_slot_leases = entry.in_use_count.load();
        if (alloc_slot_leases > 0) {
            GGML_LOG_WARN(
                "[UNIFIED-CACHE] allocate_slot refused reclaim: model=%llu name_hash=0x%llx "
                "in_use=%u (returning nullptr so caller retries)\n",
                (unsigned long long) key.model_id, (unsigned long long) key.name_hash, alloc_slot_leases);
            return nullptr;
        }
        if (entry.device_ptr && !entry.host_resident) {
            if (!entry.pool_allocated) {
                enqueue_deferred_free(entry.device_ptr, entry.size);
            }
        }
        if (type == cache_entry_type::MOE_EXPERT && layer_id == 0 && expert_id == 0) {
            fprintf(stderr,
                    "[MOE-ENTRY-ERASE] path=allocate_slot-reclaim layer=%d expert=%d have_layout=%d want_layout=%d "
                    "size=%zu pinned=%d state=%d\n",
                    layer_id, expert_id, (int) entry.layout, (int) layout, entry.size, entry.pinned ? 1 : 0,
                    (int) entry.state);
        }
        entries_.erase(it);
    }

    // Check VRAM budget and evict if needed.
    // Skip the legacy layout_pool_ when the cache has a placement plan: the arena manages
    // all VRAM allocations in that mode, making pool sub-allocation redundant.
    const bool skip_pool     = has_placement_plan();
    bool       is_pool_alloc = false;
    void *     device_ptr    = nullptr;

    if (layout_pool_ && !skip_pool) {
        auto pool_result = layout_pool_->allocate(size);
        device_ptr       = pool_result.ptr;
        if (device_ptr) {
            is_pool_alloc = true;
            if (pool_result.new_physical_bytes > 0) {
                used_.fetch_add(pool_result.new_physical_bytes, std::memory_order_relaxed);
            }
        }
    }

    if (!device_ptr && arena_active()) {
        // VRAM arena path: sub-allocate from the weight zone.
        device_ptr = zone_alloc(vram_zone_id::WEIGHT, size);
        if (device_ptr) {
            is_pool_alloc = true;  // Arena-owned: don't free individually.
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] allocate_slot: arena weight zone alloc %zu bytes\n", size);
        }
    }

    if (!device_ptr) {
        // Check live VRAM and evict if needed
        size_t free_mem = 0, total_mem = 0;
        try {
            const int device_id = get_device_id_from_queue(queue_);
            ggml_backend_sycl_get_device_memory(device_id, &free_mem, &total_mem);
        } catch (...) {
        }

        if (total_mem > 0) {
            const size_t min_headroom = 256ull * 1024ull * 1024ull;
            const size_t headroom     = std::max(min_headroom, total_mem / 10);
            const size_t usable_free  = free_mem > headroom ? free_mem - headroom : 0;
            if (size > usable_free) {
                size_t evicted_total = 0;
                while (evicted_total < size) {
                    size_t evicted = evict_one(size);
                    if (evicted == 0) {
                        break;
                    }
                    evicted_total += evicted;
                }
                if (evicted_total < size) {
                    GGML_SYCL_DEBUG("[UNIFIED-CACHE] allocate_slot: VRAM insufficient after eviction\n");
                    return nullptr;
                }
            }
        }

        try {
            device_ptr = ggml_sycl_malloc_device_raw(size, queue_, "unified_cache:slot");
        } catch (const sycl::exception & e) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] allocate_slot: malloc_device failed: %s\n", e.what());
            return nullptr;
        }
    }

    if (!device_ptr) {
        return nullptr;
    }

    // Create entry in IN_PROGRESS state (not yet READY — caller must register_ready)
    unified_cache_entry entry{};
    entry.device_ptr      = device_ptr;
    entry.src_ptr         = nullptr;
    entry.content_hash    = 0;
    entry.size            = size;
    entry.type            = type;
    entry.layer_id        = layer_id;
    entry.expert_id       = expert_id;
    entry.layout          = layout;
    entry.onednn_pack_m   = 0;
    entry.xmx_info        = {};
    entry.access_count    = 1;
    entry.last_access     = time_++;
    entry.pinned          = is_pool_alloc;
    entry.hot             = false;
    entry.state           = cache_entry_state::IN_PROGRESS;
    entry.has_ready_event = false;
    entry.host_resident   = false;
    entry.location        = cache_location::DEVICE;
    entry.pool_allocated  = is_pool_alloc;

    entries_[cache_key] = entry;
    auto id_it          = id_to_key_.find(key);
    if (id_it == id_to_key_.end()) {
        if (id_to_key_.bucket_count() == 0) {
            id_to_key_.rehash(1);
        }
        id_to_key_.emplace(key, cache_key);
    }

    if (!is_pool_alloc) {
        used_.fetch_add(size, std::memory_order_relaxed);
    }

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] allocate_slot: layout=%d size=%zu ptr=%p\n", (int) layout, size, device_ptr);
    return device_ptr;
}

void unified_cache::register_ready(const ggml_sycl_cache_id & key,
                                   void *                     device_ptr,
                                   ggml_layout_mode           layout,
                                   size_t                     size,
                                   cache_entry_type           type,
                                   int                        layer_id,
                                   int                        expert_id,
                                   const void *               src_ptr,
                                   int64_t                    onednn_pack_m) {
    if (!key.valid || !device_ptr) {
        return;
    }

    const unified_cache_key cache_key{ type, key, layer_id, expert_id };

    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    auto                                it = entries_.find(cache_key);
    if (it != entries_.end()) {
        auto & entry          = it->second;
        entry.device_ptr      = device_ptr;
        entry.state           = cache_entry_state::READY;
        entry.has_ready_event = false;
        entry.layout          = layout;
        entry.size            = size;
        entry.src_ptr         = src_ptr;
        entry.onednn_pack_m   = onednn_pack_m;
        entry.access_count++;
        entry.last_access = time_++;
    } else {
        // Entry was not pre-allocated via allocate_slot — create it directly as READY
        unified_cache_entry entry{};
        entry.device_ptr      = device_ptr;
        entry.src_ptr         = src_ptr;
        entry.content_hash    = 0;
        entry.size            = size;
        entry.type            = type;
        entry.layer_id        = layer_id;
        entry.expert_id       = expert_id;
        entry.layout          = layout;
        entry.onednn_pack_m   = onednn_pack_m;
        entry.xmx_info        = {};
        entry.access_count    = 1;
        entry.last_access     = time_++;
        entry.pinned          = false;
        entry.hot             = false;
        entry.state           = cache_entry_state::READY;
        entry.has_ready_event = false;
        entry.host_resident   = false;
        entry.location        = cache_location::DEVICE;
        entry.pool_allocated  = false;

        entries_[cache_key] = entry;
        auto id_it          = id_to_key_.find(key);
        if (id_it == id_to_key_.end()) {
            if (id_to_key_.bucket_count() == 0) {
                id_to_key_.rehash(1);
            }
            id_to_key_.emplace(key, cache_key);
        }
    }
    entry_cv_.notify_all();

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] register_ready: layout=%d size=%zu ptr=%p\n", (int) layout, size, device_ptr);
}

// ---------------------------------------------------------------------------
// Atomic expert group staging: all 3 tensors stage or none stage.
// ---------------------------------------------------------------------------
bool unified_cache::stage_expert_group(int                          block_id,
                                       int                          expert_id_arg,
                                       const expert_tensor_group &  keys,
                                       const staging_tensor_data &  gate_data,
                                       const staging_tensor_data &  up_data,
                                       const staging_tensor_data &  down_data,
                                       ggml_layout_mode             layout,
                                       const cache_layout_request * gate_req,
                                       const cache_layout_request * up_req,
                                       const cache_layout_request * down_req) {
    // Collect the tensors that need staging (skip unregistered roles)
    struct slot_info {
        const ggml_sycl_cache_id *   key;
        const staging_tensor_data *  data;
        const cache_layout_request * req;
        void *                       ptr;
        bool                         was_existing;
    };

    std::vector<slot_info> slots;
    slots.reserve(3);

    if (keys.has_gate && gate_data.src_ptr && gate_data.dst_size > 0) {
        slots.push_back({ &keys.gate_key, &gate_data, gate_req, nullptr, false });
    }
    if (keys.has_up && up_data.src_ptr && up_data.dst_size > 0) {
        slots.push_back({ &keys.up_key, &up_data, up_req, nullptr, false });
    }
    if (keys.has_down && down_data.src_ptr && down_data.dst_size > 0) {
        slots.push_back({ &keys.down_key, &down_data, down_req, nullptr, false });
    }

    if (slots.empty()) {
        return false;
    }

    // Calculate total VRAM needed (only for tensors not already cached)
    size_t total_needed = 0;
    for (auto & s : slots) {
        void * existing = lookup(*s.key, layout);
        if (existing) {
            s.ptr          = existing;  // Already cached -- skip allocation
            s.was_existing = true;
        } else {
            total_needed += s.data->dst_size;
        }
    }

    // All already cached? Success.
    if (total_needed == 0) {
        GGML_SYCL_DEBUG(
            "[UNIFIED-CACHE] stage_expert_group: blk=%d exp=%d "
            "all %zu tensors already cached\n",
            block_id, expert_id_arg, slots.size());
        return true;
    }

    // Check VRAM availability and evict if needed
    if (available() < total_needed) {
        size_t freed = evict(total_needed - available());
        (void) freed;
    }

    if (available() < total_needed) {
        GGML_SYCL_DEBUG(
            "[UNIFIED-CACHE] stage_expert_group: blk=%d exp=%d "
            "insufficient VRAM (need=%zu avail=%zu)\n",
            block_id, expert_id_arg, total_needed, available());
        return false;
    }

    // Explicit allocate + fill + register path.
    {
        bool all_ok = true;
        for (auto & s : slots) {
            if (s.was_existing) {
                continue;
            }

            // 1. Allocate VRAM slot (device only, no host fallback)
            s.ptr = allocate_slot(*s.key, s.data->dst_size, layout, cache_entry_type::MOE_EXPERT, s.data->layer_id,
                                  s.data->expert_id);
            if (!s.ptr) {
                all_ok = false;
                break;
            }
        }

        if (!all_ok) {
            // Rollback partial allocations
            for (auto & s : slots) {
                if (s.was_existing || !s.ptr) {
                    continue;
                }
                remove(*s.key, cache_entry_type::MOE_EXPERT, s.data->layer_id, s.data->expert_id, layout);
                s.ptr = nullptr;
            }
            return false;
        }

        // 2. Fill all slots (DMA + optional reorder via fill_fn from request).
        //    Fills are submitted WITHOUT a per-expert dq.wait().  This allows
        //    BCS H2D copies and CCS reorder kernels to pipeline across experts.
        //    The caller batches experts and calls get_dma_queue().wait() +
        //    Raw H2D copies go through BCS (copy-only engine) to keep CCS free.
        //    Fill functions receive DMA queue and internally route H2D to BCS.
        //
        //    IMPORTANT: When using a pre-allocated temp buffer (prealloc_temp),
        //    consecutive fills share the same staging VRAM.  The BCS H2D for
        //    tensor N+1 must not overwrite the temp buffer while the CCS reorder
        //    for tensor N is still reading it.  We chain fills via deps: each
        //    fill's H2D depends on the previous fill's reorder completion event.
        //    Without this, the BCS and CCS queues race on the shared temp buffer
        //    (WAR hazard) causing corrupted SOA layouts or BCS CAT faults.
        sycl::queue & dq  = get_dma_queue();
        sycl::queue & bcs = get_bcs_queue();
        sycl::event   last_event;
        bool          has_last_event = false;
        for (auto & s : slots) {
            if (s.was_existing) {
                continue;
            }
            if (s.req && s.req->fill_fn) {
                // Use the caller-provided fill function (GPU reorder, CPU reorder, etc.)
                // Fill functions route H2D to BCS internally via ctx->bcs_queue.
                // Pass last_event as dependency so BCS H2D waits for previous
                // CCS reorder to finish reading the shared prealloc_temp buffer.
                std::vector<sycl::event> fill_deps;
                if (has_last_event) {
                    fill_deps.push_back(last_event);
                }
                last_event     = s.req->fill_fn(dq, s.ptr, s.data->dst_size, s.data->src_ptr, s.data->src_size,
                                                s.req->fill_ctx, fill_deps);
                has_last_event = true;
            } else if (s.data->src_ptr && s.data->src_size > 0) {
                // Raw DMA copy — route to BCS (copy engine) to keep CCS free
                last_event     = bcs.memcpy(s.ptr, s.data->src_ptr, std::min(s.data->dst_size, s.data->src_size));
                has_last_event = true;
            }
        }

        // 3. Mark entries as IN_PROGRESS with ready_event (deferred READY).
        //    Entries remain IN_PROGRESS with ready_event until downstream
        //    try_get_cached_with_event chains the event.  This avoids the
        //    per-expert dq.wait() that serialized all BCS/CCS work and caused
        //    GT engine resets.
        {
            std::unique_lock<std::shared_mutex> lock(rw_mutex_);
            for (auto & s : slots) {
                if (s.was_existing) {
                    continue;
                }
                unified_cache_key ckey{ cache_entry_type::MOE_EXPERT, *s.key, s.data->layer_id, s.data->expert_id };
                auto              it = entries_.find(ckey);
                if (it != entries_.end()) {
                    it->second.has_ready_event = true;
                    it->second.ready_event     = last_event;
                    it->second.layout          = layout;
                    it->second.size            = s.data->dst_size;
                    if (has_last_event) {
                        it->second.last_write_event = last_event;
                        it->second.has_write_event  = true;
                    }
                }
            }
        }

        // Verify (first 3 experts only)
        for (auto & s : slots) {
            if (s.was_existing) {
                continue;
            }
            static std::atomic<int> verify_log{ 0 };
            if (verify_log.fetch_add(1, std::memory_order_relaxed) < 3) {
                fprintf(stderr,
                        "[STAGE-SUBMIT] blk=%d exp=%d layout=%d ptr=%p "
                        "model=%llu hash=0x%llx aux=0x%llx (deferred READY)\n",
                        block_id, expert_id_arg, (int) layout, s.ptr, (unsigned long long) s.key->model_id,
                        (unsigned long long) s.key->name_hash, (unsigned long long) s.key->aux_id);
            }
        }

        return true;
    }

    // Fallback: allocate + raw DMA fill + register_ready
    bool alloc_ok = true;
    for (auto & s : slots) {
        if (s.was_existing) {
            continue;  // Already cached
        }
        s.ptr = allocate_slot(*s.key, s.data->dst_size, layout, cache_entry_type::MOE_EXPERT, s.data->layer_id,
                              s.data->expert_id);
        if (!s.ptr) {
            alloc_ok = false;
            break;
        }
    }

    // Rollback on partial allocation failure
    if (!alloc_ok) {
        for (auto & s : slots) {
            if (s.ptr && !s.was_existing) {
                std::shared_lock<std::shared_mutex> lock(rw_mutex_);
                unified_cache_key ckey{ cache_entry_type::MOE_EXPERT, *s.key, s.data->layer_id, s.data->expert_id };
                auto              it = entries_.find(ckey);
                if (it != entries_.end() && it->second.state == cache_entry_state::IN_PROGRESS) {
                    lock.unlock();
                    remove(*s.key, cache_entry_type::MOE_EXPERT, s.data->layer_id, s.data->expert_id, layout);
                }
            }
        }
        GGML_LOG_WARN(
            "[UNIFIED-CACHE] stage_expert_group: blk=%d exp=%d "
            "allocation rollback\n",
            block_id, expert_id_arg);
        return false;
    }

    // Fill all tensors using BCS queue (copy-only engine) — no per-expert wait.
    // Raw H2D copies go through BCS to keep CCS free and prevent GT engine resets.
    sycl::queue & bcs = get_bcs_queue();
    sycl::event   last_event;
    for (auto & s : slots) {
        if (s.was_existing) {
            continue;
        }

        try {
            size_t copy_size = std::min(s.data->src_size, s.data->dst_size);
            last_event       = bcs.memcpy(s.ptr, s.data->src_ptr, copy_size);
        } catch (...) {
            GGML_LOG_WARN(
                "[UNIFIED-CACHE] stage_expert_group: DMA memcpy "
                "failed blk=%d exp=%d\n",
                block_id, expert_id_arg);
            // Rollback all new allocations
            for (auto & s2 : slots) {
                if (s2.ptr && !s2.was_existing) {
                    remove(*s2.key, cache_entry_type::MOE_EXPERT, s2.data->layer_id, s2.data->expert_id, layout);
                }
            }
            return false;
        }
    }

    // Mark entries as IN_PROGRESS with ready_event (deferred READY).
    // try_get_cached_with_event chains the event for downstream callers.
    {
        std::unique_lock<std::shared_mutex> lock(rw_mutex_);
        for (auto & s : slots) {
            if (s.was_existing) {
                continue;
            }
            unified_cache_key ckey{ cache_entry_type::MOE_EXPERT, *s.key, s.data->layer_id, s.data->expert_id };
            auto              it = entries_.find(ckey);
            if (it != entries_.end()) {
                it->second.has_ready_event  = true;
                it->second.ready_event      = last_event;
                it->second.layout           = layout;
                it->second.size             = s.data->dst_size;
                it->second.src_ptr          = s.data->src_ptr;
                it->second.last_write_event = last_event;
                it->second.has_write_event  = true;
            }
        }
    }

    GGML_SYCL_DEBUG(
        "[UNIFIED-CACHE] stage_expert_group: blk=%d exp=%d "
        "staged %zu tensors (total=%zu bytes)\n",
        block_id, expert_id_arg, slots.size(), total_needed);
    return true;
}

// ---------------------------------------------------------------------------
// Atomic expert group eviction: all 3 tensors evict together.
// ---------------------------------------------------------------------------
void unified_cache::evict_expert_group(const expert_tensor_group & keys, ggml_layout_mode layout) {
    auto evict_one_key = [&](const ggml_sycl_cache_id & key) {
        if (!key.valid) {
            return;
        }
        std::shared_lock<std::shared_mutex> lock(rw_mutex_);
        auto                                id_it = id_to_key_.find(key);
        if (id_it == id_to_key_.end()) {
            return;
        }
        auto entry_it = entries_.find(id_it->second);
        if (entry_it == entries_.end()) {
            return;
        }
        int lid = entry_it->second.layer_id;
        int eid = entry_it->second.expert_id;
        lock.unlock();
        remove(key, cache_entry_type::MOE_EXPERT, lid, eid, layout);
    };

    if (keys.has_gate) {
        evict_one_key(keys.gate_key);
    }
    if (keys.has_up) {
        evict_one_key(keys.up_key);
    }
    if (keys.has_down) {
        evict_one_key(keys.down_key);
    }
}

// ---------------------------------------------------------------------------
// Expert-granularity LRU eviction: find coldest expert group, evict all 3.
// ---------------------------------------------------------------------------
size_t unified_cache::evict_coldest_expert_group(const std::unordered_map<int64_t, expert_tensor_group> & expert_groups,
                                                 ggml_layout_mode                                         layout) {
    // Scan all expert groups to find the coldest one (lowest combined frequency).
    int64_t  coldest_gkey        = -1;
    uint32_t coldest_freq        = UINT32_MAX;
    int64_t  coldest_last_access = std::numeric_limits<int64_t>::max();
    size_t   coldest_total_bytes = 0;

    auto get_entry_info = [&](const ggml_sycl_cache_id & key) -> std::tuple<bool, uint32_t, int64_t, size_t, bool> {
        // Returns: (found, access_count, last_access, size, pinned_or_leased)
        if (!key.valid) {
            return { false, 0, 0, 0, false };
        }
        std::shared_lock<std::shared_mutex> lock(rw_mutex_);
        auto                                id_it = id_to_key_.find(key);
        if (id_it == id_to_key_.end()) {
            return { false, 0, 0, 0, false };
        }
        auto entry_it = entries_.find(id_it->second);
        if (entry_it == entries_.end()) {
            return { false, 0, 0, 0, false };
        }
        const auto & e    = entry_it->second;
        // llama.cpp-vtf7f: treat live mem_handle leases as pinned for the
        // purposes of group eviction.  evict_expert_group() below issues
        // per-tensor remove() which itself refuses on in_use_count > 0, so
        // this is defence-in-depth — without it, the scoring loop might
        // select a leased group only to have the actual remove() skip.
        const bool   held = e.pinned || e.in_use_count.load() > 0;
        return { true, e.access_count, e.last_access, e.size, held };
    };

    for (const auto & [gkey, grp] : expert_groups) {
        uint32_t combined_freq = 0;
        int64_t  oldest_access = std::numeric_limits<int64_t>::max();
        size_t   total_bytes   = 0;
        bool     any_found     = false;
        bool     any_pinned    = false;

        auto check_tensor = [&](const ggml_sycl_cache_id & key, bool has_key) {
            if (!has_key || !key.valid) {
                return;
            }
            auto [found, freq, last_access, sz, pinned] = get_entry_info(key);
            if (!found) {
                return;
            }
            any_found = true;
            combined_freq += freq;
            oldest_access = std::min(oldest_access, last_access);
            total_bytes += sz;
            if (pinned) {
                any_pinned = true;
            }
        };

        check_tensor(grp.gate_key, grp.has_gate);
        check_tensor(grp.up_key, grp.has_up);
        check_tensor(grp.down_key, grp.has_down);

        if (!any_found || any_pinned || total_bytes == 0) {
            continue;
        }

        // Compare: lower frequency wins, then older last_access as tiebreaker
        bool is_colder =
            (combined_freq < coldest_freq) || (combined_freq == coldest_freq && oldest_access < coldest_last_access);

        if (is_colder) {
            coldest_gkey        = gkey;
            coldest_freq        = combined_freq;
            coldest_last_access = oldest_access;
            coldest_total_bytes = total_bytes;
        }
    }

    if (coldest_gkey < 0) {
        GGML_SYCL_DEBUG(
            "[UNIFIED-CACHE] evict_coldest_expert_group: "
            "no evictable group found\n");
        return 0;
    }

    // Evict the coldest group
    auto it = expert_groups.find(coldest_gkey);
    if (it == expert_groups.end()) {
        return 0;
    }

    const int block = static_cast<int>(coldest_gkey >> 16);
    const int exp   = static_cast<int>(coldest_gkey & 0xFFFF);
    GGML_SYCL_DEBUG(
        "[UNIFIED-CACHE] evict_coldest_expert_group: blk=%d exp=%d "
        "freq=%u bytes=%zu\n",
        block, exp, coldest_freq, coldest_total_bytes);

    evict_expert_group(it->second, layout);
    return coldest_total_bytes;
}

void * unified_cache::lookup(const ggml_sycl_cache_id & key, ggml_layout_mode layout) {
    // Identical semantics to try_get_cached_fast -- shared_lock read-only path
    // NOTE: try_get_cached_fast already filters HOST_MMAP entries.
    return try_get_cached_fast(key, layout);
}

void * unified_cache::lookup_device_only(const ggml_sycl_cache_id & key, ggml_layout_mode layout) {
    if (!key.valid) {
        return nullptr;
    }
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    auto                                id_it = id_to_key_.find(key);
    if (id_it == id_to_key_.end()) {
        return nullptr;
    }
    auto entry_it = entries_.find(id_it->second);
    if (entry_it == entries_.end()) {
        return nullptr;
    }
    const auto & entry = entry_it->second;
    if (entry.layout != layout) {
        return nullptr;
    }
    if (entry.state != cache_entry_state::READY) {
        return nullptr;
    }
    // Only return VRAM-resident entries. Host-pinned and mmap entries are excluded.
    if (entry.host_resident) {
        return nullptr;
    }
    return entry.device_ptr;
}

unified_cache::weight_ptr_result unified_cache::get_weight_ptr(const ggml_sycl_cache_id & key) {
    weight_ptr_result result{};
    if (!key.valid) {
        return result;
    }
    // Try layouts in priority order: COALESCED > SOA > AOS.
    // This ensures the best available layout is returned, not whatever
    // id_to_key_ happens to point at (which can be ONEDNN_PACKED from PP).
    static const ggml_layout_mode       try_layouts[] = { GGML_LAYOUT_COALESCED, GGML_LAYOUT_SOA, GGML_LAYOUT_AOS };
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    for (auto layout : try_layouts) {
        // Build the full cache key with this layout
        unified_cache_key ckey{ cache_entry_type::DENSE_WEIGHT, key, -1, -1 };
        auto              entry_it = entries_.find(ckey);
        if (entry_it == entries_.end()) {
            continue;
        }
        const auto & entry = entry_it->second;
        if (entry.state != cache_entry_state::READY) {
            continue;
        }
        if (!entry.device_ptr) {
            continue;
        }
        // HOST_MMAP entries contain raw mmap pointers that are NOT GPU-accessible.
        // Returning them would cause CCS page faults when GPU kernels dereference
        // the pointer.  Only DEVICE and HOST_PINNED entries are safe.
        if (entry.location == cache_location::HOST_MMAP) {
            continue;
        }
        result.ptr       = entry.device_ptr;
        result.layout    = entry.layout;
        result.on_device = !entry.host_resident;
        return result;
    }
    lock.unlock();

    // Fallback: check direct_weight_entries_ populated by direct_stage_weight().
    // S1-PRELOAD uses direct_stage_weight which stores entries in a separate map
    // (direct_weight_entries_) for lock-free staging.  Without this fallback,
    // mem_handle WEIGHT resolution via resolve_slow() -> get_weight_ptr() misses
    // all staged weights, causing inference to fall back to host AOS pointers
    // and producing garbage output during TG.
    {
        std::shared_lock<std::shared_mutex> dlock(direct_stage_mutex_);
        auto                                it = direct_weight_entries_.find(key);
        if (it != direct_weight_entries_.end() && it->second.ptr) {
            result.ptr       = it->second.ptr;
            result.layout    = it->second.layout;
            result.on_device = true;  // direct_stage_weight always stages to device VRAM
            return result;
        }
    }
    return result;
}

// Lease-acquiring variant of get_weight_ptr — bumps in_use_count on the
// resolved entry while holding shared_lock.  Eviction paths take the unique
// lock (writer), so the increment is safely visible to a subsequent eviction
// scan.  Caller (mem_handle) MUST release via entry->in_use_count.fetch_sub(1).
//
// direct_weight_entries_ (S1-PRELOAD) is NOT refcounted — those entries live
// for the duration of the host arena and are never individually evicted.  If
// the result comes from direct_weight_entries_, entry == nullptr, meaning the
// caller has no lease obligation (and no lifetime protection against an
// arena-wide teardown, which only happens at model unload).
unified_cache::weight_ptr_lease_result unified_cache::acquire_weight_lease(const ggml_sycl_cache_id & key) {
    unified_cache_key ckey{ cache_entry_type::DENSE_WEIGHT, key, -1, -1 };
    return acquire_entry_lease(ckey);
}

unified_cache::weight_ptr_lease_result unified_cache::acquire_entry_lease(const unified_cache_key & key) {
    weight_ptr_lease_result result{};
    if (!key.id.valid) {
        return result;
    }
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    {
        auto entry_it = entries_.find(key);
        if (entry_it != entries_.end()) {
            auto & entry = entry_it->second;
            if (entry.state != cache_entry_state::READY) {
                return result;
            }
            if (!entry.device_ptr) {
                return result;
            }
            // HOST_MMAP entries hold raw mmap pointers — not GPU-accessible; same
            // filter as get_weight_ptr so mem_handle semantics are consistent.
            if (entry.location == cache_location::HOST_MMAP) {
                return result;
            }
            // Bump the lease refcount under shared_lock.  Visible to any evictor
            // that later acquires the unique_lock (acq_rel ordering).
            entry.in_use_count.fetch_add(1);
            result.ptr       = entry.device_ptr;
            result.layout    = entry.layout;
            result.on_device = !entry.host_resident;
            result.entry     = &entry;  // pointer stable across unordered_map inserts
            return result;
        }
    }
    lock.unlock();

    // Fallback: S1-PRELOAD direct_weight_entries_ (no refcount needed — these
    // entries live for the lifetime of the host arena and are not evicted
    // individually).  mem_handle receives ptr with entry == nullptr; its dtor
    // will correctly skip the release.
    //
    // on_device reflects the stored location: HOST_PINNED direct entries are
    // host-accessible (PCIe zero-copy), DEVICE direct entries are GPU VRAM.
    // Callers of acquire_weight_lease() that want a host-accessible pointer
    // (cpu_mul_mat → DNNL) must check on_device=false.
    {
        std::shared_lock<std::shared_mutex> dlock(direct_stage_mutex_);
        const auto &                        entries =
            key.type == cache_entry_type::MOE_EXPERT ? direct_expert_entries_ : direct_weight_entries_;
        auto it = entries.find(key.id);
        if (it != entries.end() && it->second.ptr) {
            result.ptr       = it->second.ptr;
            result.layout    = it->second.layout;
            result.on_device = (it->second.location == cache_location::DEVICE);
            result.entry     = nullptr;
            return result;
        }
    }
    return result;
}

sycl::queue & unified_cache::get_dma_queue() {
    // Return dedicated DMA queue if available, otherwise fall back to compute queue
    if (dma_queue_) {
        return *dma_queue_;
    }
    return queue_;
}

sycl::queue & unified_cache::get_bcs_queue() {
    // Return BCS (copy-only) queue if available, otherwise fall back to DMA queue
    if (bcs_queue_) {
        return *bcs_queue_;
    }
    return get_dma_queue();
}

// get_or_wait REMOVED — legacy synchronous blocking pattern
// get_by_data_ptr REMOVED — O(N) scan with zero callers

cache_ptr_view unified_cache::get_view(const ggml_sycl_cache_id & key_id, ggml_layout_mode layout) {
    cache_ptr_view view{};
    if (!key_id.valid) {
        return view;
    }
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    auto                                id_it = id_to_key_.find(key_id);
    if (id_it == id_to_key_.end()) {
        return view;
    }
    auto entry_it = entries_.find(id_it->second);
    if (entry_it == entries_.end()) {
        return view;
    }
    auto & entry = entry_it->second;
    if (entry.layout != layout) {
        return view;
    }
    if (entry.state == cache_entry_state::EVICTING) {
        // Entry is being async-evicted to host — device pointer is stale-bound.
        // Return empty view so caller falls back to mmap.
        return view;
    }
    if (entry.state == cache_entry_state::IN_PROGRESS) {
        if (entry.has_ready_event && event_complete(entry.ready_event)) {
            entry.state           = cache_entry_state::READY;
            entry.has_ready_event = false;
        } else {
            return view;
        }
    }
    view.ptr           = entry.device_ptr;
    view.size          = entry.size;
    view.layout        = entry.layout;
    view.onednn_pack_m = entry.onednn_pack_m;
    view.location      = entry.location;
    view.type          = entry.type;
    view.layer_id      = entry.layer_id;
    view.expert_id     = entry.expert_id;
    view.xmx_info      = entry.xmx_info;
    return view;
}

void unified_cache::remove(const ggml_sycl_cache_id & key_id,
                           cache_entry_type           type,
                           int                        layer_id,
                           int                        expert_id,
                           ggml_layout_mode           layout) {
    if (!key_id.valid) {
        return;
    }
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    if (!g_graph_compute_active.load(std::memory_order_acquire)) {
        process_deferred_frees();
    }
    unified_cache_key key{ type, key_id, layer_id, expert_id };

    auto it = entries_.find(key);
    if (it == entries_.end()) {
        return;
    }
    if (it->second.layout != layout) {
        GGML_LOG_ERROR("[UNIFIED-CACHE] layout mismatch in remove model=%llu name_hash=0x%llx have=%d want=%d\n",
                       (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash,
                       (int) it->second.layout, (int) layout);
        if (cache_assert_enabled()) {
            GGML_ABORT("unified_cache layout mismatch");
        }
        return;
    }
    if (it->second.state == cache_entry_state::IN_PROGRESS) {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] remove skipped: entry in progress model=%llu name_hash=0x%llx\n",
                        (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash);
        return;
    }

    // llama.cpp-vtf7f: refuse to remove an entry that any mem_handle leases.
    // remove() is typically called for dead weights / cache eviction; a live
    // lease at this moment implies the caller is about to dereference a
    // pointer we are about to free.  Log loudly and skip.
    const uint32_t remove_leases = it->second.in_use_count.load();
    if (remove_leases > 0) {
        GGML_LOG_WARN("[UNIFIED-CACHE] remove refused: model=%llu name_hash=0x%llx layout=%d in_use=%u (entry kept)\n",
                      (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash,
                      (int) it->second.layout, remove_leases);
        return;
    }

    enqueue_deferred_free(it->second.device_ptr, it->second.size);

    if (type == cache_entry_type::MOE_EXPERT && layer_id == 0 && expert_id == 0) {
        fprintf(stderr, "[MOE-ENTRY-ERASE] path=remove layer=%d expert=%d layout=%d size=%zu pinned=%d state=%d\n",
                layer_id, expert_id, (int) it->second.layout, it->second.size, it->second.pinned ? 1 : 0,
                (int) it->second.state);
    }
    entries_.erase(it);
    id_to_key_.erase(key_id);
}

// NOTE: mark_reordered/is_reordered removed - reorder state tracked in tensor->extra->optimized_feature

void unified_cache::pin(const ggml_sycl_cache_id & key_id, ggml_layout_mode layout) {
    if (!key_id.valid) {
        return;
    }
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    auto                                id_it = id_to_key_.find(key_id);
    if (id_it == id_to_key_.end()) {
        return;
    }
    auto entry_it = entries_.find(id_it->second);
    if (entry_it != entries_.end()) {
        if (entry_it->second.layout != layout) {
            // Layout changed since caller's last lookup (PP→TG switch).
            // Pin with the entry's current layout — caller still gets protection.
            GGML_SYCL_DEBUG(
                "[UNIFIED-CACHE] pin layout mismatch model=%llu name_hash=0x%llx have=%d want=%d — pinning with "
                "current layout\n",
                (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash,
                (int) entry_it->second.layout, (int) layout);
        }
        entry_it->second.pinned = true;
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] pin model=%llu name_hash=0x%llx layout=%d\n",
                        (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash,
                        (int) entry_it->second.layout);
    }
}

void unified_cache::unpin(const ggml_sycl_cache_id & key_id, ggml_layout_mode layout) {
    if (!key_id.valid) {
        return;
    }
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    auto                                id_it = id_to_key_.find(key_id);
    if (id_it == id_to_key_.end()) {
        return;
    }
    auto entry_it = entries_.find(id_it->second);
    if (entry_it != entries_.end()) {
        if (entry_it->second.layout != layout) {
            // Entry layout changed (e.g. PP→TG layout switch).  Unpin anyway
            // since the caller's pinned handle is stale but the entry must be
            // released for future eviction.
            GGML_SYCL_DEBUG(
                "[UNIFIED-CACHE] unpin layout mismatch model=%llu name_hash=0x%llx have=%d want=%d — unpinning "
                "anyway\n",
                (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash,
                (int) entry_it->second.layout, (int) layout);
        }
        entry_it->second.pinned = false;
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] unpin model=%llu name_hash=0x%llx layout=%d\n",
                        (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash,
                        (int) entry_it->second.layout);
    }
}

void unified_cache::unpin_experts() {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    for (auto & pair : entries_) {
        if (pair.second.type == cache_entry_type::MOE_EXPERT) {
            pair.second.pinned = false;
        }
    }
}

void unified_cache::unpin_all() {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    for (auto & pair : entries_) {
        pair.second.pinned = false;
    }
}

bool unified_cache::is_pinned(const ggml_sycl_cache_id & key_id, ggml_layout_mode layout) const {
    if (!key_id.valid) {
        return false;
    }
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    auto                                id_it = id_to_key_.find(key_id);
    if (id_it == id_to_key_.end()) {
        return false;
    }
    auto entry_it = entries_.find(id_it->second);
    if (entry_it == entries_.end()) {
        return false;
    }
    if (entry_it->second.layout != layout) {
        GGML_LOG_ERROR("[UNIFIED-CACHE] layout mismatch in is_pinned model=%llu name_hash=0x%llx have=%d want=%d\n",
                       (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash,
                       (int) entry_it->second.layout, (int) layout);
        if (cache_assert_enabled()) {
            GGML_ABORT("unified_cache layout mismatch");
        }
        return false;
    }
    return entry_it->second.pinned;
}

// === Bulk Pinning Implementation ===

int unified_cache::pin_layer_weights(int layer_id, const layer_weight_set & weights, ggml_layout_mode layout) {
    int pinned = 0;

    // Helper lambda to try pinning a single key
    auto try_pin = [&](const ggml_sycl_cache_id & key) {
        if (!key.valid) {
            return;
        }
        // Use fast path to check if entry exists with correct layout
        void * ptr = try_get_cached_fast(key, layout);
        if (ptr) {
            pin(key, layout);
            pinned++;
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] bulk pin layer=%d model=%llu name_hash=0x%llx layout=%d\n", layer_id,
                            (unsigned long long) key.model_id, (unsigned long long) key.name_hash, (int) layout);
        }
    };

    // Pin all weights in the set
    try_pin(weights.attn_norm);
    try_pin(weights.q_proj);
    try_pin(weights.k_proj);
    try_pin(weights.v_proj);
    try_pin(weights.o_proj);
    try_pin(weights.ffn_norm);
    try_pin(weights.gate_proj);
    try_pin(weights.up_proj);
    try_pin(weights.down_proj);
    // Optional fused weights
    try_pin(weights.attn_qkv_proj);
    try_pin(weights.ffn_gate_up_proj);

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] pin_layer_weights layer=%d pinned=%d layout=%d\n", layer_id, pinned, (int) layout);
    return pinned;
}

void unified_cache::unpin_layer_weights(int layer_id, const layer_weight_set & weights, ggml_layout_mode layout) {
    // Helper lambda to try unpinning a single key
    auto try_unpin = [&](const ggml_sycl_cache_id & key) {
        if (!key.valid) {
            return;
        }
        unpin(key, layout);
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] bulk unpin layer=%d model=%llu name_hash=0x%llx layout=%d\n", layer_id,
                        (unsigned long long) key.model_id, (unsigned long long) key.name_hash, (int) layout);
    };

    // Unpin all weights in the set
    try_unpin(weights.attn_norm);
    try_unpin(weights.q_proj);
    try_unpin(weights.k_proj);
    try_unpin(weights.v_proj);
    try_unpin(weights.o_proj);
    try_unpin(weights.ffn_norm);
    try_unpin(weights.gate_proj);
    try_unpin(weights.up_proj);
    try_unpin(weights.down_proj);
    // Optional fused weights
    try_unpin(weights.attn_qkv_proj);
    try_unpin(weights.ffn_gate_up_proj);

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] unpin_layer_weights layer=%d layout=%d\n", layer_id, (int) layout);
}

int unified_cache::pin_model_weights(int                                   n_layers,
                                     const std::vector<layer_weight_set> & layers,
                                     ggml_layout_mode                      layout) {
    if (n_layers <= 0 || layers.empty()) {
        return 0;
    }

    int       total_pinned  = 0;
    const int actual_layers = std::min(n_layers, (int) layers.size());

    for (int layer_id = 0; layer_id < actual_layers; layer_id++) {
        total_pinned += pin_layer_weights(layer_id, layers[layer_id], layout);
    }

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] pin_model_weights n_layers=%d total_pinned=%d layout=%d\n", actual_layers,
                    total_pinned, (int) layout);
    return total_pinned;
}

// === Async Layer Prefetch Implementation ===

void unified_cache::queue_layer_prefetch(int                      layer_id,
                                         const layer_weight_set & weights,
                                         ggml_layout_mode         layout,
                                         prefetch_priority        priority) {
    // Lazily start the worker thread on first call
    if (!prefetch_started_.load()) {
        start_prefetch_worker();
    }

    {
        std::lock_guard<std::mutex> lock(prefetch_mutex_);
        prefetch_request            req;
        req.layer_id = layer_id;
        req.weights  = weights;
        req.layout   = layout;
        req.priority = priority;

        // HIGH priority goes to the front of the queue
        if (priority == prefetch_priority::HIGH) {
            prefetch_queue_.push_front(std::move(req));
        } else {
            prefetch_queue_.push_back(std::move(req));
        }
    }
    prefetch_cv_.notify_one();

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] queue_layer_prefetch layer=%d priority=%d layout=%d\n", layer_id, (int) priority,
                    (int) layout);
}

void unified_cache::prefetch_worker_loop() {
    while (true) {
        prefetch_request req;

        // Wait for a request or shutdown signal
        {
            std::unique_lock<std::mutex> lock(prefetch_mutex_);
            prefetch_cv_.wait_for(lock, std::chrono::seconds(2),
                                  [this] { return !prefetch_queue_.empty() || prefetch_shutdown_.load(); });

            if (prefetch_shutdown_.load() && prefetch_queue_.empty()) {
                return;
            }

            // Spurious wakeup or timeout with empty queue — loop back
            if (prefetch_queue_.empty()) {
                continue;
            }

            req = std::move(prefetch_queue_.front());
            prefetch_queue_.pop_front();
        }

        GGML_SYCL_DEBUG("[UNIFIED-CACHE] prefetch_worker processing layer=%d layout=%d\n", req.layer_id,
                        (int) req.layout);

        // Pin all weights for this layer.
        // The weights should already be in cache from model loading; we just pin them
        // to prevent eviction during persistent kernel execution.
        int pinned = pin_layer_weights(req.layer_id, req.weights, req.layout);

        // Store the weight set and layout for later release_layer
        {
            std::lock_guard<std::mutex> lock(layer_state_mutex_);
            layer_weights_[req.layer_id] = req.weights;
            layer_layouts_[req.layer_id] = req.layout;
            layer_ready_[req.layer_id]   = true;
        }
        layer_ready_cv_.notify_all();

        GGML_SYCL_DEBUG("[UNIFIED-CACHE] prefetch_worker layer=%d ready (pinned=%d)\n", req.layer_id, pinned);
    }
}

layer_weight_pointers unified_cache::await_layer(int layer_id) {
    // Wait until the layer is marked ready by the prefetch worker,
    // then read layout and weights in the same critical section to avoid TOCTOU.
    ggml_layout_mode layout;
    layer_weight_set weights;
    {
        std::unique_lock<std::mutex> lock(layer_state_mutex_);
        bool                         ready = layer_ready_cv_.wait_for(lock, std::chrono::seconds(5), [this, layer_id] {
            auto it = layer_ready_.find(layer_id);
            return it != layer_ready_.end() && it->second;
        });
        if (!ready) {
            GGML_LOG_WARN("[PREFETCH] await_layer %d timed out after 5s, falling back to direct lookup\n", layer_id);
            return {};
        }
        layout  = layer_layouts_[layer_id];
        weights = layer_weights_[layer_id];
    }

    // Build the result by looking up each weight pointer in the cache.
    // try_get_cached_fast uses a shared_lock on rw_mutex_ (read-only, no deadlock risk).
    layer_weight_pointers ptrs;
    ptrs.attn_norm = try_get_cached_fast(weights.attn_norm, layout);
    ptrs.q_proj    = try_get_cached_fast(weights.q_proj, layout);
    ptrs.k_proj    = try_get_cached_fast(weights.k_proj, layout);
    ptrs.v_proj    = try_get_cached_fast(weights.v_proj, layout);
    ptrs.o_proj    = try_get_cached_fast(weights.o_proj, layout);
    ptrs.ffn_norm  = try_get_cached_fast(weights.ffn_norm, layout);
    ptrs.gate_proj = try_get_cached_fast(weights.gate_proj, layout);
    ptrs.up_proj   = try_get_cached_fast(weights.up_proj, layout);
    ptrs.down_proj = try_get_cached_fast(weights.down_proj, layout);

    // Fused weight lookups (optional, zero cache_id means not set)
    if (weights.attn_qkv_proj.valid) {
        ptrs.attn_qkv_proj = try_get_cached_fast(weights.attn_qkv_proj, layout);
    }
    if (weights.ffn_gate_up_proj.valid) {
        ptrs.ffn_gate_up_proj = try_get_cached_fast(weights.ffn_gate_up_proj, layout);
    }

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] await_layer layer=%d pointers resolved\n", layer_id);
    return ptrs;
}

bool unified_cache::get_layer_weight_set(int layer_id, layer_weight_set & out) const {
    std::lock_guard<std::mutex> lock(layer_state_mutex_);
    auto                        it = layer_weights_.find(layer_id);
    if (it == layer_weights_.end()) {
        return false;
    }
    out = it->second;
    return true;
}

bool unified_cache::is_layer_ready(int layer_id) const {
    std::lock_guard<std::mutex> lock(layer_state_mutex_);
    auto                        it = layer_ready_.find(layer_id);
    return it != layer_ready_.end() && it->second;
}

void unified_cache::release_layer(int layer_id) {
    layer_weight_set weights;
    ggml_layout_mode layout;

    // Retrieve and remove the layer's tracking state
    {
        std::lock_guard<std::mutex> lock(layer_state_mutex_);
        auto                        wit = layer_weights_.find(layer_id);
        auto                        lit = layer_layouts_.find(layer_id);
        if (wit == layer_weights_.end() || lit == layer_layouts_.end()) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] release_layer layer=%d not found\n", layer_id);
            return;
        }
        weights = wit->second;
        layout  = lit->second;

        layer_ready_.erase(layer_id);
        layer_weights_.erase(wit);
        layer_layouts_.erase(lit);
    }

    // Unpin the layer weights (uses rw_mutex_ internally, safe since we dropped layer_state_mutex_)
    unpin_layer_weights(layer_id, weights, layout);

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] release_layer layer=%d unpinned\n", layer_id);
}

void unified_cache::start_prefetch_worker() {
    std::lock_guard<std::mutex> lock(prefetch_lifecycle_mutex_);
    if (prefetch_started_.load()) {
        return;  // Already started
    }

    prefetch_shutdown_.store(false);
    prefetch_worker_ = std::thread([this] { prefetch_worker_loop(); });
    prefetch_started_.store(true);

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] prefetch worker started\n");
}

void unified_cache::stop_prefetch_worker() {
    std::lock_guard<std::mutex> lock(prefetch_lifecycle_mutex_);
    if (!prefetch_started_.load()) {
        return;  // Never started
    }

    // Signal shutdown and wake the worker
    {
        std::lock_guard<std::mutex> qlock(prefetch_mutex_);
        prefetch_shutdown_.store(true);
    }
    prefetch_cv_.notify_one();

    // Join the worker thread with timeout
    if (prefetch_worker_.joinable()) {
        auto future = std::async(std::launch::async, [this] { prefetch_worker_.join(); });
        if (future.wait_for(std::chrono::seconds(5)) == std::future_status::timeout) {
            GGML_LOG_WARN("[UNIFIED-CACHE] prefetch worker did not exit within 5s\n");
            prefetch_worker_.detach();
        }
    }

    prefetch_started_.store(false);

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] prefetch worker stopped\n");
}

size_t unified_cache::evict(size_t bytes_needed) {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    if (!g_graph_compute_active.load(std::memory_order_acquire)) {
        process_deferred_frees();
    }

    size_t freed = 0;
    while (freed < bytes_needed && !entries_.empty()) {
        const size_t n_before = entries_.size();
        size_t       evicted  = evict_one(0);
        if (evicted == 0) {
            // evict_one returns 0 for two reasons:
            //   1. No eligible entry found (all pinned/in-progress/graph-active)
            //   2. Evicted a host-resident entry (freed 0 device bytes but
            //      removed from entries_)
            // If entries_ shrank, a host-resident entry was evicted — keep
            // trying, device-resident entries may follow.
            if (entries_.size() < n_before) {
                continue;
            }
            break;  // Genuinely nothing evictable
        }
        freed += evicted;
    }
    return freed;
}

size_t unified_cache::evict_and_flush(size_t bytes_needed) {
    // Phase 1: evict entries (defers the actual sycl::free behind barrier events)
    size_t evicted = evict(bytes_needed);
    if (evicted == 0) {
        return 0;
    }

    // Invalidate per-tensor resolved pointer cache — evicted weights have
    // stale device pointers.  The generation bump forces re-resolution.
    ggml_sycl_invalidate_resolve_cache();

    // Phase 2: wait on the queue so all pending operations complete, making
    //          the deferred frees eligible for processing.
    try {
        queue_.wait_and_throw();
    } catch (const sycl::exception & e) {
        GGML_LOG_WARN("[UNIFIED-CACHE] evict_and_flush: queue wait failed: %s\n", e.what());
    } catch (...) {
        GGML_LOG_WARN("[UNIFIED-CACHE] evict_and_flush: queue wait failed (unknown)\n");
    }

    // Phase 3: process deferred frees — this calls sycl::free and
    //          saturating_sub_used, so used_ reflects the freed memory.
    {
        std::unique_lock<std::shared_mutex> lock(rw_mutex_);
        if (!g_graph_compute_active.load(std::memory_order_acquire)) {
            process_deferred_frees();
        }
    }

    return evicted;
}

static int eviction_tier(const unified_cache_entry & entry) {
    // Tiered eviction priority (lower = evict first):
    // -1: host-resident (already slow, evict first to reclaim tracking)
    // Derived from alloc_category_priority (lower priority number = higher VRAM
    // priority = harder to evict = HIGHER eviction tier).
    //   MoE experts: category priority 2 → inverted base 2 → tiers 4-5 (cold/hot)
    //   Dense weights: category priority 1 → inverted base 3 → tiers 6-7 (cold/hot)
    // Hot entries get +1 within their category to resist eviction.
    if (entry.host_resident) {
        return -1;  // Host-resident entries evict first (they're already slow)
    }
    const alloc_category cat =
        (entry.type == cache_entry_type::MOE_EXPERT) ? alloc_category::EXPERT_CACHE : alloc_category::WEIGHT;
    constexpr int k_max_priority = 4;  // max value from alloc_category_priority
    const int     inverted       = k_max_priority - alloc_category_priority(cat);
    const int     base           = inverted * 2;
    return base + (entry.hot ? 1 : 0);
}

size_t unified_cache::evict_one(size_t /* new_size */) {
    // NOTE: process_deferred_frees() was removed from here to fix a BCS CAT
    // error [18].  The race: evict_one is called from stage_expert_group
    // during prestage, which also submits BCS copies.  Processing deferred
    // frees here would call sycl::free on entries whose barrier was submitted
    // BEFORE the current batch's BCS copies — unmapping VRAM while BCS is
    // still writing to nearby pages.  Callers must drain all queues and call
    // process_deferred_frees() explicitly at safe synchronization points
    // (e.g., the prestage yield loop after queue drain).

    // Block eviction while GPU kernels are in flight (graph_compute_impl).
    // Evicting frees VRAM that MUL_MAT_ID kernels may still reference via
    // the expert pointer table → GPU page fault → DEVICE_LOST.
    // Callers fall back to host-pinned zero-copy when eviction returns 0.
    if (g_graph_compute_active.load(std::memory_order_acquire)) {
        GGML_LOG_WARN("[UNIFIED-CACHE] evict_one: blocked by graph_compute_active\n");
        return 0;
    }

    unified_cache_key evict_key{};
    int               best_tier        = std::numeric_limits<int>::max();
    uint32_t          best_freq        = UINT32_MAX;  // Lower frequency = evict first (MoE only)
    int64_t           best_last_access = std::numeric_limits<int64_t>::max();
    bool              found            = false;

    for (auto & pair : entries_) {
        auto & entry = pair.second;
        if (entry.state == cache_entry_state::EVICTING) {
            continue;  // Already being evicted asynchronously
        }
        if (entry.state == cache_entry_state::IN_PROGRESS) {
            if (entry.has_ready_event && event_complete(entry.ready_event)) {
                entry.state           = cache_entry_state::READY;
                entry.has_ready_event = false;
            } else {
                GGML_SYCL_DEBUG(
                    "[UNIFIED-CACHE] evict skip: model=%llu name_hash=0x%llx layout=%d in-progress size=%zu\n",
                    (unsigned long long) pair.first.id.model_id, (unsigned long long) pair.first.id.name_hash,
                    (int) entry.layout, entry.size);
                continue;
            }
        }
        if (entry.pinned) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] evict skip: model=%llu name_hash=0x%llx layout=%d pinned size=%zu\n",
                            (unsigned long long) pair.first.id.model_id, (unsigned long long) pair.first.id.name_hash,
                            (int) entry.layout, entry.size);
            continue;
        }
        // llama.cpp-vtf7f: skip entries with outstanding mem_handle leases.
        // A live lease means a DNNL call / kernel submit / CPU dispatch is
        // using entry.device_ptr right now; freeing would dangle the pointer.
        // Acquire-ordered load pairs with the reader's fetch_add under the
        // shared rw_mutex_, so we see every prior lease increment.
        const uint32_t entry_leases = entry.in_use_count.load();
        if (entry_leases > 0) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] evict skip: model=%llu name_hash=0x%llx layout=%d in_use=%u size=%zu\n",
                            (unsigned long long) pair.first.id.model_id, (unsigned long long) pair.first.id.name_hash,
                            (int) entry.layout, entry_leases, entry.size);
            continue;
        }

        const int tier = eviction_tier(entry);

        // For MoE expert entries (tier 0/1): use frequency as primary signal
        // within the same tier, with last_access as tiebreaker.
        // For dense weights (tier 2/3) and host-resident (tier -1): pure LRU.
        uint32_t freq = 0;
        if (entry.type == cache_entry_type::MOE_EXPERT && entry.expert_id >= 0) {
            freq = ggml_sycl::get_expert_frequency(entry.layer_id, entry.expert_id);
        }

        bool is_better = false;
        if (tier < best_tier) {
            is_better = true;
        } else if (tier == best_tier) {
            if (entry.type == cache_entry_type::MOE_EXPERT && entry.expert_id >= 0) {
                // Frequency-weighted: evict lowest frequency first, LRU tiebreaker
                is_better = (freq < best_freq) || (freq == best_freq && entry.last_access < best_last_access);
            } else {
                // Pure LRU for dense weights and host-resident entries
                is_better = (entry.last_access < best_last_access);
            }
        }

        if (is_better) {
            best_tier        = tier;
            best_freq        = freq;
            best_last_access = entry.last_access;
            evict_key        = pair.first;
            found            = true;
        }
    }

    if (!found) {
        GGML_LOG_WARN("[UNIFIED-CACHE] evict_one: no eligible entries found in %zu entries\n", entries_.size());
        return 0;  // All entries pinned
    }

    // Evict the entry
    size_t evicted_bytes = 0;
    auto   it            = entries_.find(evict_key);
    if (it != entries_.end()) {
        // llama.cpp-pxvih 4c: eviction guard assert.
        // The scan loop above already skips entries with in_use_count > 0, so
        // reaching here with a live lease is a bug — it means an evictor path
        // bypassed the scan (e.g. via a direct erase) or a lease was acquired
        // without holding rw_mutex_.  Log with full detail and skip instead of
        // silently producing a UAF.
        const uint32_t live_leases = it->second.in_use_count.load();
        if (live_leases > 0) {
            GGML_LOG_ERROR(
                "[UNIFIED-CACHE] BUG: evict_one reached free path with live leases "
                "(model=%llu name_hash=0x%llx layout=%d in_use=%u size=%zu) — "
                "skipping to prevent UAF.  This is a bug; please report.\n",
                (unsigned long long) evict_key.id.model_id, (unsigned long long) evict_key.id.name_hash,
                (int) it->second.layout, live_leases, it->second.size);
            GGML_ASSERT(live_leases == 0 &&
                        "evict_one: attempted to free a WEIGHT entry with outstanding mem_handle leases");
            return 0;  // Unreachable if GGML_ASSERT aborts; fallback if asserts disabled.
        }

        size_t entry_size    = it->second.size;
        void * ptr           = it->second.device_ptr;
        bool   host_resident = it->second.host_resident;
        int    entry_layout  = static_cast<int>(it->second.layout);
        if (it->second.type == cache_entry_type::MOE_EXPERT && it->second.layer_id == 0 && it->second.expert_id == 0) {
            fprintf(stderr,
                    "[MOE-ENTRY-ERASE] path=evict_one layer=%d expert=%d layout=%d size=%zu pinned=%d state=%d "
                    "host_resident=%d\n",
                    it->second.layer_id, it->second.expert_id, entry_layout, entry_size, it->second.pinned ? 1 : 0,
                    (int) it->second.state, host_resident ? 1 : 0);
        }
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] evict model=%llu name_hash=0x%llx layout=%d size=%zu host_resident=%d\n",
                        (unsigned long long) evict_key.id.model_id, (unsigned long long) evict_key.id.name_hash,
                        (int) it->second.layout, entry_size, host_resident ? 1 : 0);

        if (!host_resident) {
            // Check if async D2H eviction is available: device entry with a
            // transformed layout (SOA/COALESCED/XMX) worth preserving.
            const bool has_transformed_layout = it->second.layout != GGML_LAYOUT_AOS;
            const bool async_evict_enabled    = async_evict_enabled_ && has_transformed_layout;

            if (async_evict_enabled) {
                // P7: Async D2H eviction — preserve transformed layout in host-pinned memory.
                // Use pre-allocated pinned pool (zero runtime sycl::malloc_host).
                void * host_dst = host_zones_configured() ? host_zone_alloc(host_zone_id::WEIGHT, entry_size, 64) :
                                                            host_pool_alloc(entry_size, 64);

                if (host_dst) {
                    // Issue async D2H copy via DMA queue
                    sycl::queue & dq = get_dma_queue();
                    sycl::event   evt;
                    try {
                        evt = dq.memcpy(host_dst, ptr, entry_size);
                    } catch (...) {
                        if (host_zones_configured()) {
                            host_zone_free(host_zone_id::WEIGHT, host_dst);
                        } else {
                            host_pool_free(host_dst, entry_size);
                        }
                        host_dst = nullptr;
                    }

                    if (host_dst) {
                        // Mark entry as EVICTING — VRAM stays occupied until finalize
                        it->second.state              = cache_entry_state::EVICTING;
                        it->second.eviction_event     = evt;
                        it->second.has_eviction_event = true;
                        it->second.eviction_host_ptr  = host_dst;
                        it->second.pinned             = true;  // Prevent re-eviction

                        GGML_SYCL_DEBUG(
                            "[UNIFIED-CACHE] async evict started: model=%llu name_hash=0x%llx layout=%d "
                            "size=%zu\n",
                            (unsigned long long) evict_key.id.model_id, (unsigned long long) evict_key.id.name_hash,
                            entry_layout, entry_size);

                        // Return 0: VRAM not yet freed. Caller should call finalize_evictions()
                        // or retry with another entry. VRAM is reclaimed asynchronously.
                        // We still report success so the caller knows progress was made.
                        evicted_bytes = 0;
                        // Increment eviction counter so callers can poll finalize
                        evictions_in_flight_++;
                        has_evictions_.store(true, std::memory_order_release);

                        // Return entry_size to indicate the eviction is in progress
                        // even though VRAM hasn't been freed yet. The caller can
                        // call finalize_evictions() to reclaim after DMA completes.
                        return entry_size;
                    }
                }
            }

            // Synchronous eviction fallback (original path)
            // Always defer device memory frees.  Direct sycl::free inside evict_one
            // causes BCS CAT errors: eviction during prestage unmaps VRAM pages
            // while concurrent BCS copies (from stage_expert_group in the same
            // prestage loop) are still writing to nearby pages in the same L0
            // allocation region.  Deferred frees are processed at safe sync points
            // (after queue drains) where no BCS work is in-flight.
            const bool is_arena = vram_owns(ptr);
            const bool is_pool  = !is_arena && layout_pool_ && layout_pool_->owns(ptr);
            if (is_arena) {
                // Arena entries: free via TLSF zone_free.
                size_t offset = ptr_to_offset(ptr);
                if (offset != SIZE_MAX) {
                    zone_free(vram_zone_id::WEIGHT, offset_to_ptr(offset));
                }
                // No budget adjustment — arena bytes stay in used_ until arena is destroyed.
            } else if (!is_pool) {
                enqueue_deferred_free(ptr, entry_size);
            } else {
                // Pool entries: memory stays in pool, just update accounting
                saturating_sub_used(entry_size);
            }
            has_evictions_.store(true, std::memory_order_release);
        }
        if (host_resident && ptr) {
            // Host-resident entry: free the host WEIGHT zone allocation.
            if (host_zones_configured()) {
                host_zone_free(host_zone_id::WEIGHT, ptr);
            } else {
                host_pool_free(ptr, entry_size);
            }
        }

        // Remove from lookup
        id_to_key_.erase(evict_key.id);

        // Remove from entries — invalidates iterator, must not dereference `it` after this
        entries_.erase(it);

        // Bump generation so all mem_handles see that pointers may have moved.
        // Coverage: see tests/test-mem-handle-eviction.cpp.
        cache_generation_bump();

        GGML_SYCL_DEBUG(
            "[UNIFIED-CACHE] Evicted: model=%llu name_hash=0x%llx layout=%d %.2f MB (used=%.1f/%.1f MB) "
            "host_resident=%d\n",
            (unsigned long long) evict_key.id.model_id, (unsigned long long) evict_key.id.name_hash, entry_layout,
            entry_size / (1024.0f * 1024.0f), used_.load() / (1024.0f * 1024.0f), budget_ / (1024.0f * 1024.0f),
            host_resident ? 1 : 0);
        evicted_bytes = host_resident ? 0 : entry_size;  // Only count device bytes freed
    }

    return evicted_bytes;
}

// Internal: caller MUST already hold rw_mutex_ (unique).
size_t unified_cache::finalize_evictions_locked() {
    if (evictions_in_flight_.load(std::memory_order_relaxed) == 0) {
        return 0;
    }

    std::vector<unified_cache_key> finalized_keys;
    finalized_keys.reserve(4);

    for (auto & pair : entries_) {
        auto & entry = pair.second;
        if (entry.state != cache_entry_state::EVICTING || !entry.has_eviction_event) {
            continue;
        }

        // Check if D2H copy is complete
        if (!event_complete(entry.eviction_event)) {
            continue;
        }

        // DMA complete — free the host buffer (can re-read from mmap on next access).
        if (entry.eviction_host_ptr) {
            if (host_zones_configured()) {
                host_zone_free(host_zone_id::WEIGHT, entry.eviction_host_ptr);
            } else {
                host_pool_free(entry.eviction_host_ptr, entry.size);
            }
        }

        // Reclaim device VRAM
        void *     ptr        = entry.device_ptr;
        size_t     entry_size = entry.size;
        const bool is_arena   = vram_owns(ptr);
        const bool is_pool    = !is_arena && layout_pool_ && layout_pool_->owns(ptr);
        if (is_arena) {
            size_t offset = ptr_to_offset(ptr);
            if (offset != SIZE_MAX) {
                zone_free(vram_zone_id::WEIGHT, offset_to_ptr(offset));
            }
        } else if (!is_pool) {
            enqueue_deferred_free(ptr, entry_size);
        } else {
            saturating_sub_used(entry_size);
        }

        GGML_SYCL_DEBUG("[UNIFIED-CACHE] async evict finalized: model=%llu name_hash=0x%llx layout=%d size=%zu\n",
                        (unsigned long long) pair.first.id.model_id, (unsigned long long) pair.first.id.name_hash,
                        (int) entry.layout, entry_size);

        finalized_keys.push_back(pair.first);
    }

    // Remove finalized entries
    for (const auto & key : finalized_keys) {
        id_to_key_.erase(key.id);
        entries_.erase(key);
        evictions_in_flight_--;
    }

    if (!finalized_keys.empty()) {
        // Async-evict equivalent of the sync bump in evict_one.
        // Coverage: see tests/test-mem-handle-eviction.cpp (async case).
        cache_generation_bump();
    }

    return finalized_keys.size();
}

size_t unified_cache::finalize_evictions() {
    // Poll in-flight async D2H evictions.  For each completed one:
    // 1. Free the host-pinned buffer (can re-read from mmap on next access)
    // 2. Reclaim VRAM (arena zone_free or deferred free)
    // 3. Remove entry from device cache
    if (evictions_in_flight_.load(std::memory_order_relaxed) == 0) {
        return 0;
    }

    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    return finalize_evictions_locked();
}

void * unified_cache::promote_to_device(const unified_cache_key & key, size_t size) {
    // With plan-driven placement, re-promotion is handled by the normal
    // ensure_cached path (re-reads from mmap).  No separate host weight
    // cache to promote from.
    (void) key;
    (void) size;
    return nullptr;
}

float unified_cache::compute_score(const unified_cache_entry & entry) const {
    int64_t age        = time_.load() - entry.last_access;
    float   decay      = std::exp(-DECAY_ALPHA * static_cast<float>(age));
    float   base_score = static_cast<float>(entry.access_count) * decay;

    // Higher VRAM priority (lower priority number) → higher score → harder to evict.
    // Dense weights (priority 1) get more boost than MoE experts (priority 2).
    const alloc_category cat =
        (entry.type == cache_entry_type::MOE_EXPERT) ? alloc_category::EXPERT_CACHE : alloc_category::WEIGHT;
    constexpr int k_max_priority = 4;
    const float   priority_boost = static_cast<float>(k_max_priority - alloc_category_priority(cat) + 1);
    base_score *= priority_boost;  // WEIGHT → 4x, EXPERT_CACHE → 3x

    if (entry.type == cache_entry_type::DENSE_WEIGHT) {
        return base_score;
    }
    if (entry.hot) {
        constexpr float k_hot_boost = 1.5f;
        return base_score * k_hot_boost;
    }
    // Boost MoE experts with high popularity (low rank = more popular).
    // This makes popular experts resist VRAM eviction after warmup profiling.
    if (entry.type == cache_entry_type::MOE_EXPERT && entry.layer_id >= 0 && entry.expert_id >= 0) {
        if (is_expert_popularity_initialized()) {
            int pop_rank = get_expert_popularity_rank(entry.layer_id, entry.expert_id);
            if (pop_rank >= 0) {
                int boost_slots = 4;
                if (pop_rank < boost_slots) {
                    float boost = static_cast<float>(boost_slots - pop_rank);
                    base_score *= (1.0f + boost);
                }
            }
        }
    }
    return base_score;
}

sycl::event unified_cache::copy_to_device(void * dst, const void * src, size_t size) {
    // Host-pinned USM can be read directly by the GPU via DMA — skip staging.
    const sycl::usm::alloc src_type = ggml_sycl_get_alloc_type(src);
    if (src_type == sycl::usm::alloc::host || src_type == sycl::usm::alloc::device) {
        return queue_.memcpy(dst, src, size);
    }
    // Use staging buffer for mmap'd / non-USM data
    if (staging_ && size <= staging_size_) {
        std::lock_guard<std::mutex> lock(staging_mutex_);
        // Copy mmap -> staging (may trigger page fault)
        std::memcpy(staging_, src, size);
        // Copy staging -> device — return event instead of blocking
        return queue_.memcpy(dst, staging_, size);
    } else if (staging_) {
        std::lock_guard<std::mutex> lock(staging_mutex_);
        // Chunked transfer for large entries — must serialize chunks through
        // the single staging buffer, so each chunk waits on prior via depends_on.
        const char *                src_ptr   = static_cast<const char *>(src);
        char *                      dst_ptr   = static_cast<char *>(dst);
        size_t                      remaining = size;
        sycl::event                 last_event;

        while (remaining > 0) {
            size_t chunk = std::min(remaining, staging_size_);
            // Must wait for previous chunk's GPU read of staging_ to complete
            // before overwriting staging_ with next chunk's memcpy.
            if (src_ptr != static_cast<const char *>(src)) {
                last_event.wait();
            }
            std::memcpy(staging_, src_ptr, chunk);
            last_event = queue_.memcpy(dst_ptr, staging_, chunk);
            src_ptr += chunk;
            dst_ptr += chunk;
            remaining -= chunk;
        }
        // Wait for the final chunk since we hold staging_mutex_ and the staging
        // buffer must not be reused until the GPU finishes reading it.
        last_event.wait();
        return sycl::event{};
    } else {
        // No staging buffer — this should not happen since staging_ is always
        // pre-allocated in the constructor.  Fall back to host_arena pinned pool
        // to avoid runtime sycl::malloc_host.
        void * temp =
            host_zones_configured() ? host_zone_alloc(host_zone_id::SCRATCH, size, 64) : host_pool_alloc(size, 64);
        if (temp) {
            std::memcpy(temp, src, size);
            sycl::event evt = queue_.memcpy(dst, temp, size);
            // Must wait before freeing temp buffer back to pool
            evt.wait();
            if (!host_zones_configured()) {
                host_pool_free(temp, size);
            }
            return sycl::event{};
        } else {
            GGML_LOG_ERROR("[UNIFIED-CACHE] copy_to_device: no staging buffer and pinned pool exhausted\n");
            return sycl::event{};
        }
    }
}

sycl::event unified_cache::copy_to_device_async(void *                           dst,
                                                const void *                     src,
                                                size_t                           size,
                                                const std::vector<sycl::event> & deps,
                                                sycl::queue *                    override_q) {
    if (!src || !dst) {
        GGML_LOG_ERROR("[UNIFIED-CACHE] copy_to_device_async: null pointer (src=%p dst=%p size=%zu)\n", src, dst, size);
        return sycl::event{};
    }
    const sycl::usm::alloc src_type = ggml_sycl_get_alloc_type(src);
    const sycl::usm::alloc dst_type = ggml_sycl_get_alloc_type(dst);
    if (g_ggml_sycl_debug >= 2 || copy_trace_enabled()) {
        GGML_LOG_INFO("[SYCL] copy_to_device_async ptr types: dst=%p type=%d src=%p type=%d size=%zu\n", dst,
                      (int) dst_type, src, (int) src_type, size);
        if (copy_trace_enabled()) {
        }
    }
    if (dst_type == sycl::usm::alloc::unknown && cache_assert_enabled()) {
        GGML_ABORT("copy_to_device_async called with non-USM destination");
    }
    // Route memcpy through override queue when provided (e.g. BCS queue for
    // expert prefetch).  Falls back to the cache's internal queue_ otherwise.
    sycl::queue & q = override_q ? *override_q : queue_;
    if (copy_to_device_sync_enabled()) {
        for (const auto & dep : deps) {
            const_cast<sycl::event &>(dep).wait();
        }
        copy_to_device(dst, src, size).wait();
        return submit_barrier_all();
    }

    // Stage non-USM source memory through host-pinned buffer.
    // This handles:
    // - unknown: mmap'd or non-USM pointers (must stage — GPU cannot DMA)
    // - shared: can fail on Level Zero if allocated on different context
    // Host-pinned (sycl::usm::alloc::host) and device sources skip staging —
    // the GPU can DMA directly from host-pinned USM via queue.memcpy.
    // This is critical for large tensors (e.g. 615 MB token_embd.weight in
    // 120B models) where staging through intermediate buffers can segfault.
    const bool needs_staging = (src_type != sycl::usm::alloc::device && src_type != sycl::usm::alloc::host);
    if (needs_staging) {
        // Non-USM source pointers are staged through reusable host-pinned chunks.
        constexpr size_t         k_fallback_chunk = 64 * 1024 * 1024;
        const size_t             chunk_size = std::min(size, staging_size_ > 0 ? staging_size_ : k_fallback_chunk);
        const char *             src_ptr    = static_cast<const char *>(src);
        char *                   dst_ptr    = static_cast<char *>(dst);
        size_t                   remaining  = size;
        sycl::event              last;
        std::vector<sycl::event> chain = deps;

        while (remaining > 0) {
            const size_t chunk = std::min(remaining, chunk_size);

            // Acquire a pre-allocated staging slot.  Slots are allocated once at
            // cache construction — no sycl::malloc_host during inference.
            size_t slot_idx = std::numeric_limits<size_t>::max();
            if (copy_stage_slots_.empty()) {
                throw sycl::exception(sycl::make_error_code(sycl::errc::memory_allocation),
                                      "Cannot copy non-USM pointer to device: no staging slots pre-allocated");
            }
            while (slot_idx == std::numeric_limits<size_t>::max()) {
                sycl::event wait_evt{};
                bool        has_wait_evt = false;

                {
                    std::lock_guard<std::mutex> lock(copy_stage_mutex_);
                    // Scan for a free slot with sufficient capacity.
                    for (size_t i = 0; i < copy_stage_slots_.size(); ++i) {
                        const size_t idx  = (copy_stage_next_ + i) % copy_stage_slots_.size();
                        auto &       slot = copy_stage_slots_[idx];
                        if (slot.capacity < chunk) {
                            continue;
                        }
                        if (slot.in_flight && !event_complete(slot.done_event)) {
                            continue;
                        }
                        slot.in_flight   = false;
                        copy_stage_next_ = (idx + 1) % std::max<size_t>(copy_stage_slots_.size(), 1);
                        slot_idx         = idx;
                        break;
                    }
                    if (slot_idx != std::numeric_limits<size_t>::max()) {
                        break;
                    }
                    // All slots busy — wait on the next one in round-robin order.
                    const size_t idx = copy_stage_next_ % copy_stage_slots_.size();
                    copy_stage_next_ = (idx + 1) % copy_stage_slots_.size();
                    auto & slot      = copy_stage_slots_[idx];
                    if (slot.in_flight) {
                        wait_evt     = slot.done_event;
                        has_wait_evt = true;
                    }
                    slot_idx = idx;
                }

                if (has_wait_evt) {
                    wait_evt.wait();
                    std::lock_guard<std::mutex> lock(copy_stage_mutex_);
                    if (slot_idx < copy_stage_slots_.size()) {
                        copy_stage_slots_[slot_idx].in_flight = false;
                    }
                }
            }

            void * stage_ptr = nullptr;
            {
                std::lock_guard<std::mutex> lock(copy_stage_mutex_);
                GGML_ASSERT(slot_idx < copy_stage_slots_.size());
                stage_ptr = copy_stage_slots_[slot_idx].ptr;
            }

            std::memcpy(stage_ptr, src_ptr, chunk);
            sycl::event ev;
            if (chain.empty()) {
                ev = q.memcpy(dst_ptr, stage_ptr, chunk);
            } else {
                ev = q.submit([&](sycl::handler & cgh) {
                    cgh.depends_on(chain);
                    cgh.memcpy(dst_ptr, stage_ptr, chunk);
                });
            }
            {
                std::lock_guard<std::mutex> lock(copy_stage_mutex_);
                GGML_ASSERT(slot_idx < copy_stage_slots_.size());
                copy_stage_slots_[slot_idx].in_flight  = true;
                copy_stage_slots_[slot_idx].done_event = ev;
            }

            src_ptr += chunk;
            dst_ptr += chunk;
            remaining -= chunk;
            last = ev;
            chain.clear();
            chain.push_back(ev);
        }
        return last;
    }

    if (deps.empty()) {
        return q.memcpy(dst, src, size);
    }
    return q.submit([&](sycl::handler & cgh) {
        cgh.depends_on(deps);
        cgh.memcpy(dst, src, size);
    });
}

bool unified_cache::event_complete(const sycl::event & evt) {
    try {
        auto status = evt.get_info<sycl::info::event::command_execution_status>();
        return status == sycl::info::event_command_status::complete;
    } catch (...) {
        return false;
    }
}

sycl::event unified_cache::submit_barrier(const std::vector<sycl::event> & deps) {
    if (deps.empty()) {
        return sycl::event{};
    }
    return queue_.ext_oneapi_submit_barrier(deps);
}

sycl::event unified_cache::submit_barrier_all() {
    // Submit barrier that depends on ALL queues — cache, DMA, BCS, and compute.
    // This ensures deferred frees don't execute until in-flight work on every
    // queue has completed.  Missing any queue causes use-after-free:
    //   - compute queue: MUL_MAT_ID kernels reference expert pointer table
    //   - dma queue: CCS reorder kernels write to freshly-allocated VRAM slots
    //   - bcs queue: H2D copies write to temp VRAM or destination slots
    // Without the dma_queue_ barrier, sycl::free() in process_deferred_frees()
    // can unmap pages while dma_queue_ reorder kernels still reference VRAM,
    // causing BCS CAT errors when the L0 driver reuses freed pages under
    // high VRAM pressure (85-90% budget with 1000+ expert groups).
    std::vector<sycl::event> no_deps;
    std::vector<sycl::event> deps;
    try {
        deps.push_back(queue_.ext_oneapi_submit_barrier(no_deps));
    } catch (...) {
    }
    if (dma_queue_) {
        try {
            deps.push_back(dma_queue_->ext_oneapi_submit_barrier(no_deps));
        } catch (...) {
        }
    }
    if (bcs_queue_) {
        try {
            deps.push_back(bcs_queue_->ext_oneapi_submit_barrier(no_deps));
        } catch (...) {
        }
    }
    if (compute_queue_) {
        try {
            deps.push_back(compute_queue_->ext_oneapi_submit_barrier(no_deps));
        } catch (...) {
        }
    }
    // Return a barrier on the cache queue that depends on all collected events
    return queue_.ext_oneapi_submit_barrier(deps);
}

void unified_cache::enqueue_deferred_free(void * ptr, size_t size) {
    if (!ptr || size == 0) {
        return;
    }

    // Pool-owned and arena-owned pointers cannot be individually freed; skip the
    // deferred free entirely to avoid unnecessary barrier events and invalid sycl::free() calls.
    if (layout_pool_ && layout_pool_->owns(ptr)) {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] skipping deferred free for pool-owned ptr=%p size=%zu\n", ptr, size);
        return;
    }
    if (vram_owns(ptr)) {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] skipping deferred free for arena-owned ptr=%p size=%zu\n", ptr, size);
        return;
    }

    deferred_free_entry entry{};
    entry.ptr  = ptr;
    entry.size = size;
    try {
        entry.event     = submit_barrier_all();
        entry.has_event = true;
    } catch (...) {
        entry.has_event = false;
    }

    deferred_frees_.push_back(entry);
    GGML_SYCL_DEBUG("[UNIFIED-CACHE] deferred free: ptr=%p size=%zu\n", ptr, size);
}

void unified_cache::enqueue_deferred_free(const managed_alloc_ref & handle) {
    if (handle.ptr == nullptr || handle.size == 0) {
        return;
    }

    if (vram_owns(handle.ptr)) {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] skipping deferred free for arena-owned managed ptr=%p size=%zu\n", handle.ptr,
                        handle.size);
        return;
    }

    deferred_free_entry entry{};
    entry.ptr     = handle.ptr;
    entry.size    = handle.size;
    entry.handle  = handle;
    entry.managed = true;
    try {
        entry.event     = submit_barrier_all();
        entry.has_event = true;
    } catch (...) {
        entry.has_event = false;
    }

    deferred_frees_.push_back(entry);
    GGML_SYCL_DEBUG("[UNIFIED-CACHE] deferred managed free: ptr=%p size=%zu alloc_id=%llu\n", handle.ptr, handle.size,
                    static_cast<unsigned long long>(handle.alloc_id));
}

void unified_cache::enqueue_deferred_host_free(void * ptr, size_t size, const sycl::event & event) {
    if (!ptr) {
        return;
    }
    deferred_host_free_entry entry{};
    entry.ptr       = ptr;
    entry.size      = size;
    entry.has_event = true;
    entry.event     = event;
    deferred_host_frees_.push_back(entry);
    GGML_SYCL_DEBUG("[UNIFIED-CACHE] deferred host free: ptr=%p\n", ptr);
}

void unified_cache::defer_host_free(void * ptr, size_t size, const sycl::event & event) {
    enqueue_deferred_host_free(ptr, size, event);
}

void unified_cache::process_deferred_frees() {
    // P7: finalize any completed async D2H evictions first.
    // This reclaims VRAM from entries whose D2H copies have completed.
    // NOTE: Use _locked variant — caller already holds rw_mutex_.
    if (evictions_in_flight_.load(std::memory_order_relaxed) > 0) {
        finalize_evictions_locked();
    }

    auto it = deferred_frees_.begin();
    while (it != deferred_frees_.end()) {
        const bool ready = !it->has_event || event_complete(it->event);
        if (!ready) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] deferred free pending: ptr=%p size=%zu\n", it->ptr, it->size);
            ++it;
            continue;
        }

        if (it->ptr) {
            const bool is_arena = vram_owns(it->ptr);
            const bool is_pool  = !is_arena && layout_pool_ && layout_pool_->owns(it->ptr);
            if (is_arena) {
                // Arena entries: just remove from deferred list, no sycl::free needed.
                GGML_SYCL_DEBUG("[UNIFIED-CACHE] deferred free skip arena ptr=%p size=%zu\n", it->ptr, it->size);
            } else if (it->managed) {
                alloc_handle handle{};
                handle.ptr      = it->handle.ptr;
                handle.size     = it->handle.size;
                handle.device   = it->handle.device;
                handle.alloc_id = it->handle.alloc_id;
                unified_free(handle);
            } else if (!is_pool) {
                if (!it->has_event) {
                    // Instead of queue_.wait() under rw_mutex_ (deadlock risk),
                    // submit a barrier event and defer to the next cycle.
                    try {
                        it->event     = submit_barrier_all();
                        it->has_event = true;
                    } catch (...) {
                        // If barrier submission fails, fall back to skipping
                    }
                    GGML_SYCL_DEBUG("[UNIFIED-CACHE] deferred free: added barrier for ptr=%p\n", it->ptr);
                    ++it;
                    continue;
                }
                try {
                    sycl::free(it->ptr, queue_);
                } catch (...) {
                }
                saturating_sub_used(it->size);
            }
            // Pool entries: used_ stays at chunk level, memory stays in pool
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] deferred free done: ptr=%p size=%zu pool=%d\n", it->ptr, it->size,
                            is_pool ? 1 : 0);
        }

        it = deferred_frees_.erase(it);
    }

    auto host_it = deferred_host_frees_.begin();
    while (host_it != deferred_host_frees_.end()) {
        const bool ready = !host_it->has_event || event_complete(host_it->event);
        if (!ready) {
            ++host_it;
            continue;
        }
        if (host_it->ptr) {
            try {
                if (host_it->has_event) {
                    host_it->event.wait_and_throw();
                }
            } catch (...) {
            }
            try {
                sycl::free(host_it->ptr, queue_);
            } catch (...) {
            }
            if (host_it->size > 0) {
                ggml_sycl::unified_cache_sub_runtime_host_bytes(host_it->size);
            }
        }
        host_it = deferred_host_frees_.erase(host_it);
    }

    auto pin_it = inflight_unpins_.begin();
    while (pin_it != inflight_unpins_.end()) {
        const bool ready = !pin_it->has_event || event_complete(pin_it->event);
        if (g_ggml_sycl_debug) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] unpin check model=%llu name_hash=0x%llx layout=%d has_event=%d ready=%d\n",
                            (unsigned long long) pin_it->key.model_id, (unsigned long long) pin_it->key.name_hash,
                            (int) pin_it->layout, pin_it->has_event ? 1 : 0, ready ? 1 : 0);
        }
        if (!ready) {
            ++pin_it;
            continue;
        }
        if (pin_it->has_event) {
            try {
                pin_it->event.wait_and_throw();
            } catch (...) {
                // Best-effort cleanup; event_complete already said ready.
            }
        }
        auto id_it = id_to_key_.find(pin_it->key);
        if (id_it != id_to_key_.end()) {
            auto entry_it = entries_.find(id_it->second);
            if (entry_it != entries_.end()) {
                if (entry_it->second.layout != pin_it->layout) {
                    // Rate-limit: log once, then suppress. Common for MoE models where
                    // MMVQ unpins with AOS but unified cache stores SOA/COALESCED for experts.
                    static std::atomic<int> unpin_mismatch_count{ 0 };
                    int                     count = unpin_mismatch_count.fetch_add(1, std::memory_order_relaxed) + 1;
                    if (count == 1) {
                        GGML_LOG_WARN(
                            "[UNIFIED-CACHE] layout mismatch in inflight unpin: have=%d want=%d "
                            "(MoE expert layout mismatch, benign — further occurrences suppressed)\n",
                            (int) entry_it->second.layout, (int) pin_it->layout);
                    }
                    if (cache_assert_enabled()) {
                        GGML_ABORT("unified_cache layout mismatch");
                    }
                } else {
                    entry_it->second.pinned = false;
                    GGML_SYCL_DEBUG("[UNIFIED-CACHE] in-flight unpin model=%llu name_hash=0x%llx layout=%d\n",
                                    (unsigned long long) pin_it->key.model_id,
                                    (unsigned long long) pin_it->key.name_hash, (int) pin_it->layout);
                }
            }
        }
        pin_it = inflight_unpins_.erase(pin_it);
    }
}

bool unified_cache::has_pending_deferred_frees() const {
    return !deferred_frees_.empty() || !deferred_host_frees_.empty();
}

size_t unified_cache::dense_count() const {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    size_t                              count = 0;
    for (const auto & pair : entries_) {
        if (pair.second.type == cache_entry_type::DENSE_WEIGHT) {
            count++;
        }
    }
    return count;
}

size_t unified_cache::expert_count() const {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    size_t                              count = 0;
    for (const auto & pair : entries_) {
        if (pair.second.type == cache_entry_type::MOE_EXPERT) {
            count++;
        }
    }
    return count;
}

size_t unified_cache::used_bytes(cache_entry_type type) const {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    size_t                              total = 0;
    for (const auto & pair : entries_) {
        if (pair.second.type == type) {
            total += pair.second.size;
        }
    }
    return total;
}

size_t unified_cache::get_layer_vram_bytes(int layer_id) const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    size_t                              total = 0;
    for (const auto & [key, entry] : entries_) {
        if (entry.layer_id == layer_id && entry.state == cache_entry_state::READY && !entry.host_resident &&
            entry.location != cache_location::HOST_MMAP && entry.location != cache_location::HOST_PINNED) {
            total += entry.size;
        }
    }
    return total;
}

size_t unified_cache::evictable_expert_bytes() const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    size_t                              total = 0;
    for (const auto & [key, entry] : entries_) {
        if (entry.type == cache_entry_type::MOE_EXPERT && entry.state == cache_entry_state::READY && !entry.pinned &&
            !entry.host_resident) {
            total += entry.size;
        }
    }
    return total;
}

void unified_cache::print_stats() const {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);

    size_t total = hits_.load() + misses_.load();
    float  rate  = total > 0 ? 100.0f * hits_.load() / total : 0.0f;

    size_t        dense = 0, experts = 0;
    size_t        dense_bytes = 0, expert_bytes = 0;
    constexpr int layout_count                = GGML_LAYOUT_MXFP4_DPAS + 1;
    size_t        layout_counts[layout_count] = {};
    size_t        layout_bytes[layout_count]  = {};
    for (const auto & pair : entries_) {
        if (pair.second.type == cache_entry_type::DENSE_WEIGHT) {
            dense++;
            dense_bytes += pair.second.size;
        } else {
            experts++;
            expert_bytes += pair.second.size;
        }
        const int layout_idx = static_cast<int>(pair.second.layout);
        if (layout_idx >= 0 && layout_idx < layout_count) {
            layout_counts[layout_idx]++;
            layout_bytes[layout_idx] += pair.second.size;
        }
    }

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] Stats: %zu hits, %zu misses (%.1f%% hit rate)\n", hits_.load(), misses_.load(),
                    rate);
    GGML_SYCL_DEBUG("[UNIFIED-CACHE] Entries: %zu dense (%.1f MB), %zu experts (%.1f MB), total %.1f/%.1f MB\n", dense,
                    dense_bytes / (1024.0f * 1024.0f), experts, expert_bytes / (1024.0f * 1024.0f),
                    used_.load() / (1024.0f * 1024.0f), budget_ / (1024.0f * 1024.0f));
    GGML_LOG_INFO(
        "[UNIFIED-CACHE] Layouts: aos=%zu (%.1f MB), soa=%zu (%.1f MB), coalesced=%zu (%.1f MB), "
        "mxfp4_i8=%zu (%.1f MB), mxfp4_dpas=%zu (%.1f MB), "
        "xmx_tiled=%zu (%.1f MB), xmx_gemm_tiled=%zu (%.1f MB), "
        "onednn_packed=%zu (%.1f MB), onednn_woq=%zu (%.1f MB)\n",
        layout_counts[GGML_LAYOUT_AOS], layout_bytes[GGML_LAYOUT_AOS] / (1024.0f * 1024.0f),
        layout_counts[GGML_LAYOUT_SOA], layout_bytes[GGML_LAYOUT_SOA] / (1024.0f * 1024.0f),
        layout_counts[GGML_LAYOUT_COALESCED], layout_bytes[GGML_LAYOUT_COALESCED] / (1024.0f * 1024.0f),
        layout_counts[GGML_LAYOUT_MXFP4_I8], layout_bytes[GGML_LAYOUT_MXFP4_I8] / (1024.0f * 1024.0f),
        layout_counts[GGML_LAYOUT_MXFP4_DPAS], layout_bytes[GGML_LAYOUT_MXFP4_DPAS] / (1024.0f * 1024.0f),
        layout_counts[GGML_LAYOUT_XMX_TILED], layout_bytes[GGML_LAYOUT_XMX_TILED] / (1024.0f * 1024.0f),
        layout_counts[GGML_LAYOUT_XMX_GEMM_TILED], layout_bytes[GGML_LAYOUT_XMX_GEMM_TILED] / (1024.0f * 1024.0f),
        layout_counts[GGML_LAYOUT_ONEDNN_PACKED], layout_bytes[GGML_LAYOUT_ONEDNN_PACKED] / (1024.0f * 1024.0f),
        layout_counts[GGML_LAYOUT_ONEDNN_WOQ], layout_bytes[GGML_LAYOUT_ONEDNN_WOQ] / (1024.0f * 1024.0f));
}

void unified_cache::reset_stats() {
    hits_   = 0;
    misses_ = 0;
}

void unified_cache::reset_model_weight_entries() {
    // Model load is a quiescent boundary for inference. Drain cache queues so
    // direct-entry metadata cannot be cleared while an S1 copy/reorder from the
    // previous model is still in flight.
    try {
        queue_.wait();
        if (dma_queue_) {
            dma_queue_->wait();
        }
        if (bcs_queue_) {
            bcs_queue_->wait();
        }
    } catch (const sycl::exception & e) {
        GGML_LOG_WARN("[UNIFIED-CACHE] reset_model_weight_entries queue drain failed: %s\n", e.what());
    }

    struct stale_alloc {
        void *         ptr            = nullptr;
        size_t         size           = 0;
        cache_location location       = cache_location::UNKNOWN;
        bool           host_resident  = false;
        bool           pool_allocated = false;
    };

    std::vector<stale_alloc> to_free;
    const bool               trace_reset           = moe_direct_trace_enabled();
    const size_t             weight_used_before    = zone_used(vram_zone_id::WEIGHT);
    const size_t             weight_avail_before   = zone_available(vram_zone_id::WEIGHT);
    const size_t             weight_largest_before = zone_largest_free(vram_zone_id::WEIGHT);
    size_t                   entries_seen          = 0;
    size_t                   entries_erased        = 0;
    size_t                   entries_preserved     = 0;
    size_t                   bytes_erased          = 0;
    size_t                   direct_weights_before = 0;
    size_t                   direct_experts_before = 0;
    {
        std::shared_lock<std::shared_mutex> lock(direct_stage_mutex_);
        direct_weights_before = direct_weight_entries_.size();
        direct_experts_before = direct_expert_entries_.size();
    }
    {
        std::unique_lock<std::shared_mutex> lock(rw_mutex_);
        for (auto it = entries_.begin(); it != entries_.end();) {
            unified_cache_entry & entry = it->second;
            const uint32_t        live  = entry.in_use_count.load();
            entries_seen++;
            if (live != 0) {
                entries_preserved++;
                GGML_LOG_WARN(
                    "[UNIFIED-CACHE] preserving stale model weight with live leases model=%llu name_hash=0x%llx "
                    "leases=%u\n",
                    (unsigned long long) it->first.id.model_id, (unsigned long long) it->first.id.name_hash, live);
                ++it;
                continue;
            }
            if (entry.device_ptr) {
                to_free.push_back(
                    { entry.device_ptr, entry.size, entry.location, entry.host_resident, entry.pool_allocated });
                bytes_erased += entry.size;
            }
            id_to_key_.erase(it->first.id);
            it = entries_.erase(it);
            entries_erased++;
        }
        layer_ready_.clear();
        layer_weights_.clear();
        layer_layouts_.clear();
    }
    for (const stale_alloc & alloc : to_free) {
        if (!alloc.ptr) {
            continue;
        }
        if (alloc.host_resident || alloc.location == cache_location::HOST_PINNED) {
            if (host_zones_configured()) {
                host_zone_free(host_zone_id::WEIGHT, alloc.ptr);
            } else {
                host_pool_free(alloc.ptr, alloc.size);
            }
            continue;
        }
        if (vram_owns(alloc.ptr)) {
            zone_free(vram_zone_id::WEIGHT, alloc.ptr);
        } else if (layout_pool_ && layout_pool_->owns(alloc.ptr)) {
            saturating_sub_used(alloc.size);
        } else {
            enqueue_deferred_free(alloc.ptr, alloc.size);
        }
    }
    {
        std::unique_lock<std::shared_mutex> lock(direct_stage_mutex_);
        direct_weight_entries_.clear();
        direct_expert_entries_.clear();
    }

    has_evictions_.store(false, std::memory_order_relaxed);
    evictions_in_flight_.store(0, std::memory_order_relaxed);
    if (trace_reset) {
        GGML_LOG_INFO(
            "[UNIFIED-CACHE] reset model weight entries: entries seen=%zu erased=%zu preserved_live=%zu "
            "queued_free=%.1f MB direct_before weight=%zu expert=%zu "
            "weight_zone used %.1f->%.1f MB avail %.1f->%.1f MB largest %.1f->%.1f MB\n",
            entries_seen, entries_erased, entries_preserved, bytes_erased / (1024.0 * 1024.0), direct_weights_before,
            direct_experts_before, weight_used_before / (1024.0 * 1024.0),
            zone_used(vram_zone_id::WEIGHT) / (1024.0 * 1024.0), weight_avail_before / (1024.0 * 1024.0),
            zone_available(vram_zone_id::WEIGHT) / (1024.0 * 1024.0), weight_largest_before / (1024.0 * 1024.0),
            zone_largest_free(vram_zone_id::WEIGHT) / (1024.0 * 1024.0));
    }
    GGML_SYCL_DEBUG("[UNIFIED-CACHE] reset model weight entries\n");
}

bool unified_cache::validate() const {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    bool                                ok = true;

    for (const auto & pair : entries_) {
        const auto & key   = pair.first;
        const auto & entry = pair.second;
        auto         it    = id_to_key_.find(key.id);
        if (it == id_to_key_.end() || !(it->second == key)) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] validate: id_to_key mismatch model=%llu name_hash=0x%llx\n",
                            (unsigned long long) key.id.model_id, (unsigned long long) key.id.name_hash);
            ok = false;
        }
        if (!entry.device_ptr || entry.size == 0) {
            GGML_SYCL_DEBUG(
                "[UNIFIED-CACHE] validate: entry missing data model=%llu name_hash=0x%llx layout=%d size=%zu\n",
                (unsigned long long) key.id.model_id, (unsigned long long) key.id.name_hash, (int) entry.layout,
                entry.size);
            ok = false;
        }
    }

    for (const auto & pair : id_to_key_) {
        auto it = entries_.find(pair.second);
        if (it == entries_.end()) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] validate: dangling id_to_key entry model=%llu name_hash=0x%llx\n",
                            (unsigned long long) pair.first.model_id, (unsigned long long) pair.first.name_hash);
            ok = false;
        }
    }

    return ok;
}

// NOTE (792vn.5 cleanup): Arena zones now handle VRAM partitioning for runtime
// allocations (KV, oneDNN, scratch).  Dynamic budget pressure from runtime
// allocations is handled by the overcommit guard in unified_cache_total_committed_bytes()
// rather than by shrinking the weight cache budget via this method.  Callers that
// previously used g_runtime_reserved_bytes to drive update_reserved_bytes() have been
// removed.  The method is retained for unified_cache deferred-reserve patterns
// during cache creation.
void unified_cache::update_reserved_bytes(size_t reserved_bytes) {
    size_t effective_budget = 0;
    {
        std::unique_lock<std::shared_mutex> lock(rw_mutex_);
        reserved_ = reserved_bytes;
        if (reserved_ >= base_budget_) {
            budget_ = 0;
            GGML_LOG_INFO("[UNIFIED-CACHE] Reserve %.1f MB >= base budget %.1f MB; cache budget now 0 (used %.1f MB)\n",
                          reserved_ / (1024.0f * 1024.0f), base_budget_ / (1024.0f * 1024.0f),
                          used_.load() / (1024.0f * 1024.0f));
        } else {
            budget_ = base_budget_ - reserved_;
        }
        effective_budget = budget_;
        while (used_.load() > budget_ && !entries_.empty()) {
            if (evict_one(0) == 0) {
                break;
            }
        }
        const size_t used = used_.load();
        if (used > budget_) {
            if (!budget_exceeded_) {
                GGML_LOG_WARN(
                    "[UNIFIED-CACHE] Budget exceeded: used %.1f MB > budget %.1f MB, "
                    "eviction exhausted (reserved %.1f MB)\n",
                    used / (1024.0f * 1024.0f), budget_ / (1024.0f * 1024.0f), reserved_ / (1024.0f * 1024.0f));
            }
            budget_exceeded_ = true;
        } else {
            budget_exceeded_ = false;
        }
    }
}

void unified_cache::unpin_on_event(const ggml_sycl_cache_id & key_id,
                                   ggml_layout_mode           layout,
                                   const sycl::event &        event) {
    if (!key_id.valid) {
        return;
    }
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    inflight_unpin_entry                entry{};
    entry.key       = key_id;
    entry.layout    = layout;
    entry.event     = event;
    entry.has_event = true;
    inflight_unpins_.push_back(entry);
    if (g_ggml_sycl_debug) {
        const bool ready = entry.has_event ? event_complete(entry.event) : true;
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] in-flight pin model=%llu name_hash=0x%llx layout=%d ready=%d\n",
                        (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash, (int) layout,
                        ready ? 1 : 0);
    }
}

bool unified_cache::get_dma_staging_buffers(size_t slice_bytes, size_t count, dma_staging_buffers & out) {
    out = {};
    if (slice_bytes == 0 || count == 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(dma_staging_mutex_);
    if (!dma_staging_buffers_.empty()) {
        if (dma_buffer_count_ >= count && dma_slice_bytes_ >= slice_bytes) {
            out.buffers     = dma_staging_buffers_.data();
            out.count       = count;
            out.slice_bytes = slice_bytes;
            return true;
        }
        GGML_SYCL_DEBUG(
            "[UNIFIED-CACHE] DMA staging pool mismatch: have=%zu x %.1f MB, need=%zu x %.1f MB; reallocating\n",
            dma_buffer_count_, dma_slice_bytes_ / (1024.0 * 1024.0), count, slice_bytes / (1024.0 * 1024.0));
        for (size_t i = 0; i < dma_staging_buffers_.size(); ++i) {
            if (i < dma_staging_allocs_.size() && dma_staging_allocs_[i].ptr != nullptr) {
                enqueue_deferred_free(dma_staging_allocs_[i]);
                continue;
            }
            void * ptr = dma_staging_buffers_[i];
            if (!ptr) {
                continue;
            }
            enqueue_deferred_free(ptr, dma_slice_bytes_);
        }
        dma_staging_allocs_.clear();
        dma_staging_buffers_.clear();
        dma_slice_bytes_  = 0;
        dma_buffer_count_ = 0;
    }

    const size_t old_reserved = dma_reserved_bytes_;
    const size_t new_reserved = slice_bytes * count;

    dma_staging_allocs_.assign(count, {});
    dma_staging_buffers_.resize(count, nullptr);
    size_t allocated = 0;
    for (size_t i = 0; i < count; ++i) {
        alloc_request req{};
        req.queue                          = &queue_;
        req.device                         = get_device_id_from_queue(queue_);
        req.size                           = slice_bytes;
        req.intent.role                    = alloc_role::STAGING;
        req.intent.category                = runtime_category::STAGING;
        req.intent.constraints.must_device = true;

        alloc_handle handle{};
        if (!unified_alloc(req, &handle) || handle.ptr == nullptr) {
            break;
        }
        dma_staging_allocs_[i]  = { handle.ptr, handle.size, handle.device, handle.alloc_id };
        dma_staging_buffers_[i] = handle.ptr;
        allocated++;
    }

    if (allocated != count) {
        for (auto & handle : dma_staging_allocs_) {
            if (handle.ptr == nullptr) {
                continue;
            }
            alloc_handle managed{};
            managed.ptr      = handle.ptr;
            managed.size     = handle.size;
            managed.device   = handle.device;
            managed.alloc_id = handle.alloc_id;
            unified_free(managed);
        }
        dma_staging_allocs_.clear();
        dma_staging_buffers_.clear();
        dma_slice_bytes_    = 0;
        dma_buffer_count_   = 0;
        dma_reserved_bytes_ = old_reserved;
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] DMA staging pool allocation failed (need=%zu x %.1f MB)\n", count,
                        slice_bytes / (1024.0 * 1024.0));
        return false;
    }

    dma_reserved_bytes_ = new_reserved;
    dma_slice_bytes_    = slice_bytes;
    dma_buffer_count_   = count;
    GGML_SYCL_DEBUG("[UNIFIED-CACHE] DMA staging pool allocated: %zu x %.1f MB\n", count,
                    slice_bytes / (1024.0 * 1024.0));
    out.buffers     = dma_staging_buffers_.data();
    out.count       = count;
    out.slice_bytes = dma_slice_bytes_;
    return true;
}

unified_cache::dma_stream_result unified_cache::stream_dma(const cache_ptr_view &           src,
                                                           size_t                           total_bytes,
                                                           size_t                           slice_bytes,
                                                           size_t                           buffer_count,
                                                           dma_stream_slice_fn              slice_fn,
                                                           const void *                     ctx,
                                                           const std::vector<sycl::event> & deps,
                                                           dma_stream_copy_fn               copy_fn) {
    dma_stream_result result{};
    if (!src.ptr || !slice_fn) {
        return result;
    }

    size_t bytes = src.size;
    if (total_bytes > 0) {
        bytes = std::min(total_bytes, src.size);
    }
    if (bytes == 0) {
        return result;
    }

    resolve_dma_defaults(slice_bytes, buffer_count);
    if (slice_bytes == 0 || buffer_count == 0) {
        return result;
    }
    if (slice_bytes > bytes) {
        slice_bytes = bytes;
    }

    result.slice_bytes  = slice_bytes;
    result.buffer_count = buffer_count;
    GGML_SYCL_DEBUG("[UNIFIED-CACHE] DMA stream: ptr=%p bytes=%zu slice=%.1f MB buffers=%zu loc=%d type=%d\n", src.ptr,
                    bytes, slice_bytes / (1024.0 * 1024.0), buffer_count, static_cast<int>(src.location),
                    static_cast<int>(src.type));

    if (src.location == cache_location::DEVICE) {
        result.event  = slice_fn(queue_, src.ptr, bytes, 0, ctx, deps);
        result.ok     = true;
        result.slices = 1;
        return result;
    }

    if (src.location == cache_location::HOST_MMAP) {
        result.used_mmap_direct = true;
        if (std::getenv("GGML_SYCL_TEST_DMA_FAIL") != nullptr) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] DMA test override: forcing mmap DMA failure\n");
            result.mmap_direct_failed = true;
            return result;
        }
    }

    dma_staging_buffers staging{};
    if (!get_dma_staging_buffers(slice_bytes, buffer_count, staging)) {
        return result;
    }

    if (src.location == cache_location::HOST_MMAP) {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] DMA streaming from mmap pointer %p (bytes=%zu)\n", src.ptr, bytes);
    }

    std::vector<sycl::event> all_events;
    std::vector<sycl::event> buffer_events(buffer_count);
    std::vector<bool>        buffer_has_event(buffer_count, false);

    size_t offset = 0;
    size_t slices = 0;
    while (offset < bytes) {
        const size_t cur  = std::min(slice_bytes, bytes - offset);
        const size_t slot = slices % buffer_count;

        std::vector<sycl::event> copy_deps = deps;
        if (buffer_has_event[slot]) {
            copy_deps.push_back(buffer_events[slot]);
        }

        sycl::event copy_evt;
        try {
            if (copy_fn) {
                copy_evt = copy_fn(queue_, staging.buffers[slot], cur, offset, src.ptr, src.size, ctx, copy_deps);
            } else if (src.location == cache_location::HOST_MMAP) {
                // Avoid direct queue_.memcpy from mmap'd pointers (can trigger device loss).
                GGML_SYCL_DEBUG("[UNIFIED-CACHE] DMA stream staging mmap slice offset=%zu size=%zu\n", offset, cur);
                copy_evt = copy_to_device_async(staging.buffers[slot], static_cast<const char *>(src.ptr) + offset, cur,
                                                copy_deps);
            } else {
                copy_evt =
                    queue_.memcpy(staging.buffers[slot], static_cast<const char *>(src.ptr) + offset, cur, copy_deps);
            }
        } catch (const sycl::exception & e) {
            GGML_LOG_ERROR("[UNIFIED-CACHE] DMA copy failed: %s\n", e.what());
            if (src.location == cache_location::HOST_MMAP) {
                result.mmap_direct_failed = true;
            }
            return result;
        }

        std::vector<sycl::event> kernel_deps;
        kernel_deps.push_back(copy_evt);
        sycl::event kernel_evt = slice_fn(queue_, staging.buffers[slot], cur, offset, ctx, kernel_deps);

        buffer_events[slot]    = kernel_evt;
        buffer_has_event[slot] = true;
        all_events.push_back(kernel_evt);

        offset += cur;
        slices++;
    }

    result.slices = slices;
    if (!all_events.empty()) {
        if (queue_.has_property<sycl::property::queue::in_order>()) {
            // In-order queues already serialize submissions; avoid ext_oneapi_submit_barrier.
            result.event = all_events.back();
        } else {
            result.event = submit_barrier(all_events);
        }
    }
    result.ok = true;
    if (result.used_mmap_direct && !result.mmap_direct_failed) {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] DMA mmap direct ok: slices=%zu bytes=%zu\n", result.slices, bytes);
    }
    return result;
}

void unified_cache::set_hot(const ggml_sycl_cache_id & key_id,
                            cache_entry_type           type,
                            int                        layer_id,
                            int                        expert_id,
                            ggml_layout_mode           layout,
                            bool                       hot) {
    if (!key_id.valid) {
        return;
    }
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    unified_cache_key                   key{ type, key_id, layer_id, expert_id };
    auto                                it = entries_.find(key);
    if (it != entries_.end()) {
        if (it->second.layout != layout) {
            GGML_LOG_ERROR("[UNIFIED-CACHE] layout mismatch in set_hot model=%llu name_hash=0x%llx have=%d want=%d\n",
                           (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash,
                           (int) it->second.layout, (int) layout);
            if (cache_assert_enabled()) {
                GGML_ABORT("unified_cache layout mismatch");
            }
            return;
        }
        it->second.hot = hot;
    }
}

void unified_cache::clear_hot_experts(int layer_id) {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    for (auto & pair : entries_) {
        if (pair.second.type == cache_entry_type::MOE_EXPERT && pair.second.layer_id == layer_id) {
            pair.second.hot = false;
        }
    }
}

// === Mode and Global Functions ===

unified_cache_mode get_unified_cache_mode() {
    // Check environment variable
    const char * env = std::getenv("GGML_SYCL_UNIFIED_CACHE_MODE");
    if (env) {
        if (strcmp(env, "global") == 0) {
            return unified_cache_mode::GLOBAL;
        }
        if (strcmp(env, "per_device") == 0) {
            return unified_cache_mode::PER_DEVICE;
        }
        if (strcmp(env, "auto") == 0) {
            return unified_cache_mode::AUTO;
        }
    }
    return g_cache_mode;
}

void set_unified_cache_mode(unified_cache_mode mode) {
    if (g_cache_mode_locked) {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Mode change ignored: cache already initialized\n");
        return;
    }
    g_cache_mode = mode;
}

void set_scheduler_device_count(int count) {
    g_scheduler_device_count = count;
}

void set_total_gpu_count(int count) {
    g_total_gpu_count = count;
}

// === Per-Device Queue Pool ===
// Per-device queues each using their OWN SYCL context (single-device context).
//
// IMPORTANT: Prior implementations used a shared context (device 0's context
// spanning all GPUs), which triggers DEVICE_LOST / OUT_OF_DEVICE_MEMORY errors
// on compute-runtime 26.x when the Level Zero context spans multiple discrete
// Arc GPUs.  See intel/compute-runtime#916 and #921.
//
// With per-device contexts:
//   - Each secondary GPU gets its own sycl::context containing only that device.
//   - Cross-device data transfer uses host-staged copies (already the pattern
//     in dispatch_experts_secondary_gpu_impl).
//   - No cross-device depends_on() or USM sharing — each context is isolated.
//   - Synchronization uses host-side event.wait() between queues.
static sycl::queue * g_shared_ctx_queues[GGML_SYCL_MAX_DEVICES] = {};
static bool          g_shared_ctx_queues_initialized            = false;

void init_shared_context_queues(int total_gpus) {
    if (g_shared_ctx_queues_initialized) {
        return;
    }
    g_shared_ctx_queues_initialized = true;

    if (total_gpus < 2) {
        return;
    }

    int n_created = 0;
    for (int d = 1; d < total_gpus && d < GGML_SYCL_MAX_DEVICES; d++) {
        try {
            auto & dev_d = ggml_sycl_get_gpu_device(d);
            if (!g_shared_ctx_queues[d]) {
                // Create a SINGLE-DEVICE context for device d.  This avoids the
                // multi-device Level Zero context regression (compute-runtime#916,
                // #921) that causes DEVICE_LOST when a context spans 2+ discrete
                // Arc GPUs.  Each device operates in its own context; data moves
                // between devices via host-staged memcpy.
                sycl::context dev_d_ctx(dev_d);
                g_shared_ctx_queues[d] = new sycl::queue(dev_d_ctx, dev_d, default_queue_properties());
            }
            n_created++;
        } catch (const std::exception & e) {
            GGML_LOG_WARN("[PER-DEV-QUEUE] Device %d unavailable: %s\n", d, e.what());
        } catch (...) {
            GGML_LOG_WARN("[PER-DEV-QUEUE] Device %d unavailable (unknown error)\n", d);
        }
    }
    if (n_created > 0) {
        GGML_LOG_INFO("[PER-DEV-QUEUE] %d secondary queues created with per-device contexts\n", n_created);
    }
}

sycl::queue * get_shared_context_queue(int device) {
    if (device < 0 || device >= GGML_SYCL_MAX_DEVICES) {
        return nullptr;
    }
    return g_shared_ctx_queues[device];
}

// Helper: Determine effective mode (resolves AUTO)
static unified_cache_mode get_effective_mode() {
    unified_cache_mode mode = get_unified_cache_mode();
    if (mode == unified_cache_mode::AUTO) {
        // Use g_total_gpu_count (physical GPUs) to determine cache mode.
        // When multiple physical GPUs are visible, each gets its own
        // PER_DEVICE cache so the unified cache planner can place weights
        // on the correct device.  The scheduler may still see only device 0
        // (no tensor split), but the planner distributes weights across all
        // physical GPUs.
        //
        // IMPORTANT: Do NOT call ggml_sycl_info() here.  This function is
        // called from get_unified_cache_for_device_impl(), which is invoked
        // inside ggml_sycl_init()'s static initialization.  Calling
        // ggml_sycl_info() would re-enter the static guard and deadlock.
        // Use g_total_gpu_count instead — it is set in ggml_sycl_init()
        // BEFORE cache pre-initialization runs.
        int total_gpus = g_total_gpu_count;
        // If not yet initialized (called before ggml_sycl_init sets it),
        // fall back to dpct device count (safe — no static guard).
        if (total_gpus < 0) {
            total_gpus = dpct::dev_mgr::instance().device_count();
        }
        return (total_gpus > 1) ? unified_cache_mode::PER_DEVICE : unified_cache_mode::GLOBAL;
    }
    return mode;
}

// Helper: Get device ID from queue.
// Uses gpu_dpct_ids[] (pre-scheduler-hiding GPU map) for secondary devices that
// are hidden from the scheduler.  The scheduler-filtered map's identity fallback
// is wrong when non-GPU devices are interleaved in dpct enumeration.
static int get_device_id_from_queue(sycl::queue & queue) {
    try {
        sycl::device dev  = queue.get_device();
        const auto & info = ggml_sycl_info();
        // First: check all physical GPUs via gpu_dpct_ids[] (pre-hiding map).
        // This correctly finds secondary GPUs that were hidden from the scheduler.
        for (int i = 0; i < info.total_gpu_count && i < GGML_SYCL_MAX_DEVICES; i++) {
            if (ggml_sycl_get_gpu_device(i) == dev) {
                return i;
            }
        }
        // Fallback: check all dpct devices via scheduler-filtered map.
        int device_count = dpct::dev_mgr::instance().device_count();
        for (int i = 0; i < device_count; i++) {
            if (ggml_sycl_get_device(i) == dev) {
                return i;
            }
        }
    } catch (...) {
    }
    return dpct::dev_mgr::instance().current_device_id();
}

// Helper: Resolve effective device ID (GLOBAL mode maps everything to device 0)
static int resolve_effective_device(int device) {
    unified_cache_mode mode             = get_effective_mode();
    int                effective_device = (mode == unified_cache_mode::GLOBAL) ? 0 : device;
    if (effective_device < 0 || effective_device >= GGML_SYCL_MAX_DEVICES) {
        return -1;
    }
    return effective_device;
}

// Helper: Look up existing cache under shared (read) lock.
// Returns nullptr if cache doesn't exist yet.  Safe for hot-path use
// because g_device_caches entries are never erased during inference.
static unified_cache * get_cache_shared(int effective_device) {
    std::shared_lock<std::shared_mutex> lock(g_cache_rw_mutex);
    auto                                it = g_device_caches.find(effective_device);
    if (it == g_device_caches.end() || !it->second) {
        return nullptr;
    }
    return it->second.get();
}

// Non-initializing cache lookup for planner diagnostics.  Placement planning can
// be called from unit tests before ggml_sycl_info() has finished its static
// initialization; creating a cache from there re-enters that static guard.
static unified_cache * get_existing_cache_for_device(int device_id) {
    const int effective_device = resolve_effective_device(device_id);
    if (effective_device < 0) {
        return nullptr;
    }
    return get_cache_shared(effective_device);
}

// Direct arena zone query — caller MUST hold g_cache_rw_mutex.
// Accesses g_device_caches without re-acquiring the lock.
static size_t arena_non_weight_used_locked(int device_id) {
    auto it = g_device_caches.find(device_id);
    if (it == g_device_caches.end() || !it->second || !it->second->arena_active()) {
        return 0;
    }
    return it->second->zone_used(vram_zone_id::KV) + it->second->zone_used(vram_zone_id::ONEDNN) +
           it->second->zone_used(vram_zone_id::RUNTIME) + it->second->zone_used(vram_zone_id::SCRATCH);
}

static size_t runtime_reserved_bytes_nolock(int device_id) {
    return arena_non_weight_used_locked(device_id);
}

static size_t runtime_reserved_adjusted_nolock(int device_id) {
    return arena_non_weight_used_locked(device_id);
}

static size_t runtime_reserved_host_bytes_nolock() {
    return g_runtime_reserved_host_bytes.load(std::memory_order_relaxed);
}

// Helper: Create cache for a device.
// deferred_reserved_out: if non-null, stores the reserved bytes that the
// caller must apply via update_reserved_bytes() AFTER releasing g_cache_rw_mutex.
// This prevents a deadlock: update_reserved_bytes() → recalc → layer streaming
// → zone_alloc() → tries to re-lock g_cache_rw_mutex.
// Optional device memory hints to avoid calling ggml_sycl_info() during
// static init (which would deadlock on the static local guard).
struct device_mem_hint {
    size_t free_mem          = 0;
    size_t total_mem         = 0;
    size_t free_vram_at_init = 0;
};

static unified_cache * create_cache_for_device(int                     device_id,
                                               size_t *                deferred_reserved_out = nullptr,
                                               const device_mem_hint * hint                  = nullptr) {
    // Get queue for this device
    sycl::queue & queue = ggml_sycl_get_device(device_id).default_queue();

    // Reserve VRAM headroom for DMA staging buffers.
    // Match the resolved defaults (including evictable-weight sizing) unless explicitly overridden.
    size_t dma_reserve_mb    = 0;
    size_t dma_reserve_bytes = 0;
    size_t reserve_mb_env    = 0;
    bool   reserve_env_set   = parse_env_mb_value("GGML_SYCL_DMA_RESERVE_MB", reserve_mb_env);
    size_t slice_bytes       = 0;
    size_t buffers           = 0;
    if (reserve_env_set) {
        dma_reserve_mb    = reserve_mb_env;
        dma_reserve_bytes = dma_reserve_mb * 1024ULL * 1024ULL;
    } else {
        resolve_dma_defaults(slice_bytes, buffers);
        if (slice_bytes == 0 || buffers == 0) {
            dma_reserve_bytes = 0;
        } else {
            dma_reserve_bytes = slice_bytes * buffers;
        }
    }

    if (dma_reserve_bytes > 0) {
        // DMA reserve is now tracked via arena zones — no separate counter needed.
        if (reserve_env_set) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] Reserving %.1f MB for DMA staging (fixed)\n",
                            dma_reserve_bytes / (1024.0 * 1024.0));
        } else {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] Reserving %.1f MB for DMA staging (buffers=%zu, slice=%.1f MB)\n",
                            dma_reserve_bytes / (1024.0 * 1024.0), buffers, slice_bytes / (1024.0 * 1024.0));
        }
    }

    // Auto-calculate budget if not set
    size_t budget                = g_unified_cache_budget;
    bool   budget_capped_to_free = false;
    if (budget == 0) {
        size_t free_mem = 0, total_mem = 0;
        if (hint && hint->total_mem > 0) {
            // Use caller-provided values to avoid ggml_sycl_info() reentry deadlock
            free_mem  = hint->free_mem;
            total_mem = hint->total_mem;
        } else {
            ggml_backend_sycl_get_device_memory(device_id, &free_mem, &total_mem);
        }
        size_t base_mem =
            (hint && hint->total_mem > 0) ? hint->total_mem : ggml_sycl_info().devices[device_id].total_vram;
        if (base_mem == 0) {
            base_mem = total_mem > 0 ? total_mem : free_mem;
        }

        int          pct     = g_unified_cache_budget_pct;
        // Allow env var override for testing host fallback paths
        const char * env_pct = getenv("GGML_SYCL_VRAM_BUDGET_PCT");
        if (env_pct) {
            pct = std::atoi(env_pct);
            GGML_LOG_INFO("[UNIFIED-CACHE] Budget override via GGML_SYCL_VRAM_BUDGET_PCT=%d%%\n", pct);
        }
        if (pct < 1) {
            pct = 1;
        } else if (pct > 100) {
            pct = 100;
        }

        budget = static_cast<size_t>(base_mem * (static_cast<double>(pct) / 100.0));

        // Cap budget to actual free VRAM to account for system overhead
        // (display compositor, driver structures, etc.).
        //
        // IMPORTANT: Use the pre-probe free VRAM snapshot, NOT the current
        // get_memory_info() value.  The alloc probe at init does binary-search
        // malloc_device/free cycles whose freed memory lingers in the L0 USM
        // pool, making post-probe get_memory_info() report artificially low
        // free_mem (e.g. 600 MB on a 12 GB GPU).  The pre-probe snapshot
        // reflects the true available VRAM before our process consumed any.
        size_t clean_free = (hint && hint->free_vram_at_init > 0) ? hint->free_vram_at_init :
                                                                    ggml_sycl_get_free_vram_at_init(device_id);
        if (clean_free == 0) {
            clean_free = free_mem;  // fallback to current if pre-probe unavailable
        }
        if (clean_free > 0 && budget > clean_free) {
            GGML_LOG_INFO(
                "[UNIFIED-CACHE] Capping budget from %.1f MB to %.1f MB "
                "(pre-probe free VRAM)\n",
                budget / (1024.0f * 1024.0f), clean_free / (1024.0f * 1024.0f));
            budget                = clean_free;
            budget_capped_to_free = true;
        }

        // Reserve generic device slack outside the unified cache. This is not
        // the PP pipeline budget; that is modeled explicitly in the placement
        // plan and reserved inside the arena's RUNTIME zone. This coarse
        // cushion covers other out-of-cache runtime consumers such as driver
        // overhead, transient kernel temporaries, and allocations that still
        // bypass zone management.
        constexpr size_t device_runtime_slack_headroom = 512ull * 1024ull * 1024ull;
        if (budget > device_runtime_slack_headroom) {
            budget -= device_runtime_slack_headroom;
            GGML_LOG_INFO(
                "[UNIFIED-CACHE] Generic device slack headroom: %.0f MB reserved "
                "(weight budget=%.1f MB)\n",
                device_runtime_slack_headroom / (1024.0 * 1024.0), budget / (1024.0 * 1024.0));
        }

        // Leave additional headroom for KV cache and ggml compute buffers.
        // When the VRAM arena is active, KV allocates from the shared KV+weight
        // zone inside the arena, so no external headroom is needed — the planner
        // charges KV alongside weights in the shared budget.
        // When arena is NOT active, KV allocates via sycl::malloc_device outside
        // the cache, so we must reserve headroom.
        if (!vram_arena_enabled()) {
            constexpr size_t kv_compute_headroom = 2048ull * 1024ull * 1024ull;
            if (budget > kv_compute_headroom + device_runtime_slack_headroom) {
                budget -= kv_compute_headroom;
                GGML_LOG_INFO("[UNIFIED-CACHE] KV+compute headroom: %.0f MB reserved\n",
                              kv_compute_headroom / (1024.0 * 1024.0));
            }
        }

        char desc[256] = { 0 };
        ggml_backend_sycl_get_device_description(device_id, desc, sizeof(desc));
        GGML_LOG_INFO("[UNIFIED-CACHE] Device %d (%s): total=%.1f MB free=%.1f MB budget=%.1f MB (%d%%)\n", device_id,
                      desc, base_mem / (1024.0f * 1024.0f), free_mem / (1024.0f * 1024.0f),
                      budget / (1024.0f * 1024.0f), pct);
    }

    const size_t staging_bytes       = resolve_host_staging_bytes();
    // Total VRAM for arena_reserve's runtime-headroom calculation.  Sourced
    // from the hint when supplied (mid ggml_sycl_init() — calling
    // ggml_backend_sycl_get_device_memory() there would reenter
    // ggml_sycl_info() and deadlock), otherwise queried directly.
    size_t       total_vram_for_ctor = hint ? hint->total_mem : size_t(0);
    if (total_vram_for_ctor == 0) {
        size_t free_unused = 0;
        ggml_backend_sycl_get_device_memory(device_id, &free_unused, &total_vram_for_ctor);
    }
    try {
        g_device_caches[device_id] =
            std::make_unique<unified_cache>(queue, budget, staging_bytes, dma_reserve_bytes, total_vram_for_ctor);
        // Always baseline pre-existing runtime reservations so they don't
        // eat into the cache's weight budget.  Allocations that existed before
        // cache creation (DMA pre-reserve, probe residuals) are already
        // accounted for in the budget calculation — only NEW runtime
        // allocations after this point should reduce available cache space.
        const size_t reserved_total    = runtime_reserved_bytes_nolock(device_id);
        // Baseline no longer stored separately — arena zones provide the single source of truth.
        const size_t baseline          = 0;  // No baseline adjustment needed with arena zones.
        const size_t reserved_adjusted = runtime_reserved_adjusted_nolock(device_id);
        // Defer update_reserved_bytes to caller (after releasing g_cache_rw_mutex)
        // to avoid deadlock: update_reserved_bytes → recalc → layer streaming
        // → zone_alloc → re-lock g_cache_rw_mutex
        if (deferred_reserved_out) {
            *deferred_reserved_out = reserved_adjusted;
        } else if (reserved_adjusted > 0) {
            g_device_caches[device_id]->update_reserved_bytes(reserved_adjusted);
        }
        return g_device_caches[device_id].get();
    } catch (const sycl::exception & e) {
        GGML_LOG_ERROR("[UNIFIED-CACHE] Failed to initialize device %d: %s\n", device_id, e.what());
        return nullptr;
    }
}

unified_cache * get_unified_cache(sycl::queue & queue) {
    unified_cache_mode mode      = get_effective_mode();
    int                device_id = (mode == unified_cache_mode::GLOBAL) ? 0 : get_device_id_from_queue(queue);

    // Fast path: check under shared lock (no contention during inference)
    {
        std::shared_lock<std::shared_mutex> read_lock(g_cache_rw_mutex);
        auto                                it = g_device_caches.find(device_id);
        if (it != g_device_caches.end()) {
            return it->second.get();
        }
    }

    // Slow path: create cache under exclusive lock
    unified_cache * result           = nullptr;
    size_t          deferred_reserve = 0;
    {
        std::unique_lock<std::shared_mutex> lock(g_cache_rw_mutex);
        g_cache_mode_locked = true;

        // Double-check after acquiring write lock
        auto it = g_device_caches.find(device_id);
        if (it != g_device_caches.end()) {
            return it->second.get();
        }

        result = create_cache_for_device(device_id, &deferred_reserve);
    }
    // Apply deferred reserved bytes outside the mutex to avoid deadlock
    if (result && deferred_reserve > 0) {
        result->update_reserved_bytes(deferred_reserve);
    }
    return result;
}

// Internal implementation: accepts optional device memory hints to avoid
// ggml_sycl_info() reentry deadlock during ggml_sycl_init().
static unified_cache * get_unified_cache_for_device_impl(int device_id, const device_mem_hint * hint) {
    unified_cache_mode mode             = get_effective_mode();
    int                effective_device = (mode == unified_cache_mode::GLOBAL) ? 0 : device_id;

    // Fast path: check under shared lock (no contention during inference)
    {
        std::shared_lock<std::shared_mutex> read_lock(g_cache_rw_mutex);
        auto                                it = g_device_caches.find(effective_device);
        if (it != g_device_caches.end()) {
            return it->second.get();
        }
    }

    // Slow path: create cache under exclusive lock
    unified_cache * result           = nullptr;
    size_t          deferred_reserve = 0;
    {
        std::unique_lock<std::shared_mutex> lock(g_cache_rw_mutex);
        g_cache_mode_locked = true;

        // Double-check after acquiring write lock
        auto it = g_device_caches.find(effective_device);
        if (it != g_device_caches.end()) {
            return it->second.get();
        }

        result = create_cache_for_device(effective_device, &deferred_reserve, hint);
    }
    // Apply deferred reserved bytes outside the mutex to avoid deadlock
    if (result && deferred_reserve > 0) {
        result->update_reserved_bytes(deferred_reserve);
    }
    return result;
}

unified_cache * get_unified_cache_for_device(int device_id) {
    return get_unified_cache_for_device_impl(device_id, nullptr);
}

unified_cache * get_unified_cache_for_device(int    device_id,
                                             size_t free_mem,
                                             size_t total_mem,
                                             size_t free_vram_at_init) {
    device_mem_hint hint;
    hint.free_mem          = free_mem;
    hint.total_mem         = total_mem;
    hint.free_vram_at_init = free_vram_at_init;
    return get_unified_cache_for_device_impl(device_id, &hint);
}

bool unified_cache_enabled() {
    // Check if explicitly disabled
    const char * env = std::getenv("GGML_SYCL_UNIFIED_CACHE");
    if (env && std::atoi(env) == 0) {
        return false;  // Explicitly disabled
    }
    // Unified cache is now the default for MoE expert caching
    // Set GGML_SYCL_UNIFIED_CACHE=0 to disable
    return true;
}

void set_unified_cache_budget(size_t bytes) {
    std::unique_lock<std::shared_mutex> lock(g_cache_rw_mutex);
    if (g_cache_mode_locked) {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Budget change ignored: cache already initialized\n");
        return;
    }
    g_unified_cache_budget = bytes;
}

void set_unified_cache_budget_pct(int pct) {
    std::unique_lock<std::shared_mutex> lock(g_cache_rw_mutex);
    if (g_cache_mode_locked) {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Budget pct change ignored: cache already initialized\n");
        return;
    }
    if (pct < 1) {
        pct = 1;
    } else if (pct > 100) {
        pct = 100;
    }
    g_unified_cache_budget_pct = pct;
}

void set_unified_cache_host_budget_pct(int pct) {
    std::unique_lock<std::shared_mutex> lock(g_cache_rw_mutex);
    if (g_cache_mode_locked) {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Host budget pct change ignored: cache already initialized\n");
        return;
    }
    if (pct < 1) {
        pct = 1;
    } else if (pct > 100) {
        pct = 100;
    }
    g_unified_cache_host_budget_pct = pct;
}

alloc_tier unified_select_tier(const alloc_request & req) {
    if (req.intent.constraints.must_device) {
        return alloc_tier::DEVICE_VRAM;
    }
    if (req.intent.constraints.must_host_pinned) {
        return alloc_tier::HOST_PINNED;
    }

    if (req.intent.constraints.prefer_same_tier_as_cohort && req.intent.cohort_id && req.intent.cohort_id[0] != '\0') {
        std::lock_guard<std::mutex> lock(g_runtime_alloc_mutex);
        auto                        it = g_runtime_cohort_tier.find(req.intent.cohort_id);
        if (it != g_runtime_cohort_tier.end()) {
            return it->second;
        }
    }

    if (req.device >= 0) {
        if (auto * cache = get_unified_cache_for_device(req.device)) {
            if (req.size > cache->available()) {
                return alloc_tier::HOST_PINNED;
            }
        }
    }
    return alloc_tier::DEVICE_VRAM;
}

static bool unified_free_record(const runtime_alloc_record & rec) {
    bool ok = true;
    // Arena zones handle their own accounting via zone_reset/zone_alloc.
    // No separate counter bookkeeping needed.

    try {
        if (rec.handle.tier == alloc_tier::MMAP_TRACKED) {
            return true;
        }
        // Zone-managed allocations: route through handle.zone_managed fields.
        // VRAM TLSF zones (zone_managed=true, vram_zone!=COUNT): call zone_free() for O(1) reclaim.
        // Host zones use TLSF. WEIGHT and persistent KV zones support
        // per-alloc zone_free. EXPERT_STAGING in SCRATCH is also scoped and
        // reclaimable: PP MoE CPU fallback allocates act/out slabs per expert
        // and releases them after synchronous D2H → CPU → H2D completion.
        // Other SCRATCH/STAGING users remain reset-scoped and are reclaimed by
        // host_zone_reset().
        // Bump-arena device allocations (from_arena=true, vram_zone==COUNT): freed by arena_reset().
        if (rec.handle.zone_managed) {
            if (rec.handle.vram_zone != vram_zone_id::COUNT) {
                unified_cache_zone_free(rec.handle.device, rec.handle.vram_zone, rec.handle.ptr);
            } else if (rec.handle.host_zone == host_zone_id::WEIGHT || rec.handle.host_zone == host_zone_id::KV ||
                       (rec.handle.host_zone == host_zone_id::SCRATCH &&
                        rec.handle.role == alloc_role::EXPERT_STAGING)) {
                // These zones/roles have per-allocation lifetimes and must be
                // reclaimable during rollback or scoped MoE CPU fallback.
                auto * cache = get_unified_cache_for_device(rec.handle.device);
                if (cache) {
                    if (!rec.handle.all_segments.empty()) {
                        for (const auto & seg : rec.handle.all_segments) {
                            cache->host_zone_free(rec.handle.host_zone, seg.ptr);
                        }
                    } else {
                        cache->host_zone_free(rec.handle.host_zone, rec.handle.ptr);
                    }
                }
            }
            // SCRATCH/STAGING host zones: reset-only by design — freed by host_zone_reset().
            // Assertion: every zone-managed alloc must be accounted for by a known zone type.
            GGML_ASSERT(rec.handle.vram_zone != vram_zone_id::COUNT || rec.handle.host_zone != host_zone_id::COUNT);
            return true;
        }
        if (rec.from_arena) {
            // Legacy: bump-arena allocation without zone routing.
            // Freed by arena_reset() — no individual free needed.
            return true;
        }

        // Handle segmented allocations: release all segments
        if (!rec.handle.all_segments.empty()) {
            if (rec.uses_pinned_pool) {
                // Return all segments to the pinned pool (via unified_cache host_arena)
                auto * cache = get_unified_cache_for_device(rec.handle.device);
                if (cache) {
                    for (const auto & seg : rec.handle.all_segments) {
                        if (!rec.zone_managed) {
                            cache->host_pool_free(seg.ptr, seg.size);
                        }
                        // Zone-managed segments are reclaimed by zone reset or pool destruction
                    }
                }
            } else {
                // Non-pinned pool: free each segment individually
                for (const auto & seg : rec.handle.all_segments) {
                    if (rec.queue != nullptr && seg.ptr != nullptr) {
                        sycl::free(seg.ptr, *rec.queue);
                    } else if (seg.ptr != nullptr && rec.handle.device >= 0 &&
                               rec.handle.device < GGML_SYCL_MAX_DEVICES) {
                        auto & q = ggml_sycl_get_device(rec.handle.device).default_queue();
                        sycl::free(seg.ptr, q);
                    }
                }
            }
            return ok;
        }

        // Check if this pointer was allocated via regular malloc (pinned cap overflow).
        // The alloc_registry tracks these as MMAP type.  Using sycl::free on them
        // would crash, so use ::free instead.
        const auto * reg_info = ggml_sycl::alloc_registry::instance().lookup(rec.handle.ptr);
        if (reg_info && reg_info->type == ggml_sycl::alloc_type::MMAP) {
            ggml_sycl::alloc_registry::instance().unregister_alloc(rec.handle.ptr);
            ::free(rec.handle.ptr);
            return true;
        }
        if (rec.uses_pinned_pool) {
            if (!rec.zone_managed) {
                if (auto * cache = get_unified_cache_for_device(rec.handle.device)) {
                    cache->host_pool_free(rec.handle.ptr, rec.handle.size);
                }
            }
            // Zone-managed host allocations are reclaimed by zone reset or pool
            // destruction, not individual free().
        } else if (rec.queue != nullptr && rec.handle.ptr != nullptr) {
            sycl::free(rec.handle.ptr, *rec.queue);
        } else if (rec.handle.ptr != nullptr && rec.handle.device >= 0 && rec.handle.device < GGML_SYCL_MAX_DEVICES) {
            auto & q = ggml_sycl_get_device(rec.handle.device).default_queue();
            sycl::free(rec.handle.ptr, q);
        }
    } catch (const sycl::exception & e) {
        GGML_LOG_ERROR("[UNIFIED-ALLOC] free failed ptr=%p size=%zu dev=%d tier=%s: %s\n", rec.handle.ptr,
                       rec.handle.size, rec.handle.device, alloc_tier_name(rec.handle.tier), e.what());
        ok = false;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("[UNIFIED-ALLOC] free failed ptr=%p size=%zu dev=%d tier=%s: %s\n", rec.handle.ptr,
                       rec.handle.size, rec.handle.device, alloc_tier_name(rec.handle.tier), e.what());
        ok = false;
    }

    if (!ok) {
        // Arena zones handle accounting — no separate counter bookkeeping to restore on error.
    }
    return ok;
}

bool unified_alloc(const alloc_request & req_in, alloc_handle * out) {
    if (out == nullptr) {
        return false;
    }
    *out = {};

    if (req_in.size == 0) {
        return true;
    }

    alloc_request req = req_in;
    if (req.queue == nullptr && req.device >= 0 && req.device < GGML_SYCL_MAX_DEVICES) {
        req.queue = &ggml_sycl_get_device(req.device).default_queue();
    }
    if (req.queue == nullptr) {
        GGML_LOG_ERROR("[UNIFIED-ALLOC] invalid request: missing queue\n");
        return false;
    }
    if (req.device < 0) {
        req.device = get_device_id_from_queue(*req.queue);
    }
    if (req.device < 0 || req.device >= GGML_SYCL_MAX_DEVICES) {
        GGML_LOG_ERROR("[UNIFIED-ALLOC] invalid request: device=%d\n", req.device);
        return false;
    }

    if (req.intent.constraints.must_device && req.intent.constraints.must_host_pinned) {
        GGML_LOG_ERROR("[UNIFIED-ALLOC] invalid request: both must_device and must_host_pinned are set\n");
        return false;
    }

    // Phase gate: block new unified_alloc() calls during steady-state inference.
    // All memory must be pre-allocated during LOAD/WARMUP phases.
    // Zone sub-allocations (arena_alloc, zone_alloc) are still permitted.
    {
        static std::atomic<int> s_phase_gate_mode{ -1 };
        int                     mode = s_phase_gate_mode.load(std::memory_order_relaxed);
        if (mode < 0) {
            const char * env = std::getenv("GGML_SYCL_ALLOC_PHASE_GATE");
            // Diagnostic gate only. Default logging here interleaves with token
            // output and breaks deterministic completion checks; enable it
            // explicitly when auditing steady-state allocation regressions.
            mode             = (env != nullptr) ? std::atoi(env) : 0;  // 0=off, 1=warn, 2=assert
            s_phase_gate_mode.store(mode, std::memory_order_relaxed);
        }
        if (mode > 0) {
            // Exempt host-pinned allocations from the phase gate.
            // All host-pinned memory comes from pre-allocated pinned pool chunks
            // (sycl::malloc_host is never called during inference).  Staging of
            // INPUT leaf tensors, CPU dispatch D2H buffers, and offload scratch
            // all route through here — they're pool sub-allocations, not new
            // driver allocations.
            const bool          is_pool_suballoc = req_in.intent.constraints.must_host_pinned;
            const offload_phase phase            = offload_stats_phase();
            if (!is_pool_suballoc && (phase == offload_phase::PP || phase == offload_phase::TG)) {
                if (mode >= 2) {
                    GGML_LOG_ERROR(
                        "[UNIFIED-ALLOC] PHASE GATE: unified_alloc() called during %s phase "
                        "(size=%zu, device=%d, role=%d). All memory must be pre-allocated.\n",
                        offload_phase_name(phase), req_in.size, req_in.device, static_cast<int>(req_in.intent.role));
                    GGML_ASSERT(false && "unified_alloc called during PP/TG inference phase");
                } else {
                    GGML_LOG_WARN(
                        "[UNIFIED-ALLOC] PHASE GATE WARNING: unified_alloc() during %s (size=%zu, device=%d)\n",
                        offload_phase_name(phase), req_in.size, req_in.device);
                }
            }
        }
    }

    runtime_category cat =
        req.intent.category == runtime_category::OTHER ? category_from_role(req.intent.role) : req.intent.category;
    alloc_tier tier = unified_select_tier(req);
    if (tier == alloc_tier::MMAP_TRACKED) {
        GGML_LOG_ERROR("[UNIFIED-ALLOC] MMAP_TRACKED runtime allocations are not supported in unified_alloc\n");
        return false;
    }
    if (req.intent.constraints.must_device) {
        tier = alloc_tier::DEVICE_VRAM;
    }
    if (req.intent.constraints.must_host_pinned) {
        tier = alloc_tier::HOST_PINNED;
    }

    const size_t alloc_size     = std::max<size_t>(req.size, 1);
    bool         reserve_device = (tier == alloc_tier::DEVICE_VRAM);
    bool         reserve_host   = (tier == alloc_tier::HOST_PINNED);

    bool   uses_pinned_pool = false;
    bool   zone_managed     = false;
    bool   from_arena       = false;
    bool   kv_spill_to_host = false;
    void * ptr              = nullptr;
    if (tier == alloc_tier::DEVICE_VRAM) {
        // Guard against Level Zero overcommit: if this allocation would exceed
        // the device's total VRAM, fail early so the caller can retry with
        // host-pinned.  Without this check, L0 may return a valid pointer but
        // a subsequent memset/memcpy triggers DEVICE_LOST.
        if (req.device >= 0 && req.device < GGML_SYCL_MAX_DEVICES) {
            const size_t total_vram   = ggml_sycl_info().devices[req.device].total_vram;
            const size_t runtime_vram = unified_cache_arena_non_weight_used(req.device);
            // Include weight cache usage (SOA entries on device VRAM)
            size_t       cache_vram   = 0;
            auto *       cache        = get_unified_cache_for_device(req.device);
            if (cache) {
                cache_vram = cache->used();
            }
            size_t used_vram = runtime_vram + cache_vram;
            if (total_vram > 0 && used_vram + alloc_size > total_vram) {
                // Try evicting cache entries to make room before failing.
                // This is the key mechanism that lets the 120B model work:
                // model load fills VRAM with cached weights, then compute
                // buffer allocation triggers eviction to free space.
                if (cache) {
                    const size_t needed = used_vram + alloc_size - total_vram;
                    GGML_LOG_INFO(
                        "[UNIFIED-ALLOC] Device %d VRAM pressure: "
                        "used=%.1f MB + alloc=%.1f MB > total=%.1f MB, "
                        "evicting %.1f MB from cache\n",
                        req.device, used_vram / (1024.0 * 1024.0), alloc_size / (1024.0 * 1024.0),
                        total_vram / (1024.0 * 1024.0), needed / (1024.0 * 1024.0));
                    const size_t freed = cache->evict_and_flush(needed);
                    // Re-check after eviction
                    cache_vram         = cache->used();
                    used_vram          = unified_cache_arena_non_weight_used(req.device) + cache_vram;
                    if (freed > 0) {
                        GGML_LOG_INFO(
                            "[UNIFIED-ALLOC] Evicted %.1f MB, "
                            "used now=%.1f MB (cache=%.1f MB + runtime=%.1f MB)\n",
                            freed / (1024.0 * 1024.0), used_vram / (1024.0 * 1024.0), cache_vram / (1024.0 * 1024.0),
                            unified_cache_arena_non_weight_used(req.device) / (1024.0 * 1024.0));
                    }
                }
                if (total_vram > 0 && used_vram + alloc_size > total_vram) {
                    GGML_SYCL_DEBUG(
                        "[UNIFIED-ALLOC] Device %d VRAM overcommit guard: "
                        "used=%.1f MB + alloc=%.1f MB > total=%.1f MB, failing\n",
                        req.device, used_vram / (1024.0f * 1024.0f), alloc_size / (1024.0f * 1024.0f),
                        total_vram / (1024.0f * 1024.0f));
                    return false;
                }
            }
        }
        // prefer_vram_zone: explicit zone routing for RUNTIME/SCRATCH/ONEDNN/KV zones.
        // Caller sets alloc_constraints::prefer_vram_zone to request a specific TLSF zone.
        // unified_free() calls cache->zone_free(vram_zone, ptr) for TLSF reclaim.
        if (req.intent.constraints.prefer_vram_zone != vram_zone_id::COUNT && vram_arena_enabled()) {
            auto * cache = get_unified_cache_for_device(req.device);
            if (cache && cache->arena_active()) {
                const vram_zone_id zid = req.intent.constraints.prefer_vram_zone;
                ptr                    = cache->zone_alloc(zid, alloc_size);
                if (ptr) {
                    from_arena        = true;
                    out->zone_managed = true;
                    out->vram_zone    = zid;
                    GGML_SYCL_DEBUG("[UNIFIED-ALLOC] zone alloc: dev=%d zone=%d size=%.1f MB ptr=%p\n", req.device,
                                    static_cast<int>(zid), alloc_size / (1024.0 * 1024.0), ptr);
                }
                // If zone is full, fall through to raw device malloc below.
            }
        }
        // P5: Route KV allocations through the arena's KV zone when active.
        // This co-locates KV cache with weights in the same pre-allocated VRAM block,
        // eliminating separate sycl::malloc_device calls during context creation.
        if (!ptr && req.intent.role == alloc_role::KV && vram_arena_enabled()) {
            auto * cache = get_unified_cache_for_device(req.device);
            if (cache && cache->arena_active()) {
                ptr = cache->zone_alloc(vram_zone_id::KV, alloc_size);
                if (ptr) {
                    from_arena        = true;
                    out->zone_managed = true;
                    out->vram_zone    = vram_zone_id::KV;
                    GGML_SYCL_DEBUG("[UNIFIED-ALLOC] KV arena alloc: dev=%d size=%.1f MB ptr=%p\n", req.device,
                                    alloc_size / (1024.0 * 1024.0), ptr);
                } else if (req.intent.constraints.must_device) {
                    GGML_SYCL_DEBUG(
                        "[UNIFIED-ALLOC] KV arena zone full for must-device request "
                        "(need %.1f MB, avail %.1f MB); trying raw device allocation\n",
                        alloc_size / (1024.0 * 1024.0), cache->zone_available(vram_zone_id::KV) / (1024.0 * 1024.0));
                } else {
                    // Shared zone full (weights consumed most of it).
                    // Fall back to host-pinned KV instead of raw device alloc
                    // to avoid VRAM overcommit → DEVICE_LOST.
                    GGML_LOG_WARN(
                        "[UNIFIED-ALLOC] KV arena zone full (need %.1f MB, avail %.1f MB), "
                        "falling back to host-pinned KV\n",
                        alloc_size / (1024.0 * 1024.0), cache->zone_available(vram_zone_id::KV) / (1024.0 * 1024.0));
                    kv_spill_to_host = true;
                }
            }
        }
        if (!ptr && !kv_spill_to_host) {
            ptr = ggml_sycl_malloc_device_raw(alloc_size, *req.queue, "unified_alloc:device");
        }
        // KV arena spill: redirect to host-pinned path.
        if (!ptr && kv_spill_to_host) {
            reserve_device = false;
            reserve_host   = true;
            tier           = alloc_tier::HOST_PINNED;
        }
    }
    if (!ptr && (tier == alloc_tier::HOST_PINNED || kv_spill_to_host)) {
        // Always try the pre-allocated pinned chunk pool first (lock-free path).
        if (auto * ucache = get_unified_cache_for_device(req.device)) {
            // Route to the correct host zone based on role. KV spills go to KV
            // zone; permanent weight data (WEIGHT role) goes to WEIGHT zone so
            // it is never recycled by reset-scoped zones. EXPERT_STAGING is
            // CPU compute scratch (act/out slabs for MoE CPU fallback), so it
            // uses SCRATCH and can be reclaimed by scoped smart-handle owners.
            // Transfer/readback staging stays in STAGING.
            auto select_zone = [&]() {
                if (kv_spill_to_host || req.intent.role == alloc_role::KV) {
                    return host_zone_id::KV;
                }
                if (req.intent.role == alloc_role::WEIGHT) {
                    return host_zone_id::WEIGHT;
                }
                if (req.intent.role == alloc_role::EXPERT_STAGING) {
                    return host_zone_id::SCRATCH;
                }
                return host_zone_id::STAGING;
            };
            // CRITICAL: unified_alloc() must return a pointer whose [ptr, ptr+size)
            // is fully mapped and contiguous in the process address space.
            // `sycl::malloc_host` chunks are NOT guaranteed to be adjacent in
            // virtual memory (Linux DRM render-node mappings get whatever
            // addresses the kernel picks, with mmap guard pages in between).
            // `pinned_chunk_pool::zone_alloc_segmented` can therefore return
            // a vector of non-contiguous segments that span multiple chunks.
            //
            // Returning only segments[0].ptr while pretending the full
            // alloc_size is contiguous (as the previous code did) silently
            // fragments the address space: the caller writes to
            // base + offset and lands in the unmapped gap between chunks,
            // which manifests as CPU-backend SIGSEGV during mul_mat_id.
            //
            // Fix: use the single-segment-contiguous `zone_alloc`. If the
            // zone cannot satisfy the request contiguously, grow the zone by
            // one chunk (or more, capped by budget) and retry. If growth is
            // blocked (phase gate or budget exhausted), fail cleanly so the
            // caller can fall back through the sycl::malloc_host path below.
            auto try_zone_alloc_contiguous = [&](host_zone_id zone) -> void * {
                if (!ucache->host_zones_configured() && req.intent.constraints.use_pinned_pool) {
                    // Zones not yet configured (pre-configure model-load path):
                    // fall through to host_pool_alloc below.
                    return nullptr;
                }
                void * p = ucache->host_zone_alloc(zone, alloc_size, pinned_chunk_pool::DEFAULT_ALIGNMENT);
                if (p) {
                    return p;
                }
                // Fragmentation path: the zone has enough aggregate free bytes
                // somewhere (possibly across multiple chunks) but no single
                // chunk has `alloc_size` contiguous. Grow the zone by at least
                // one chunk to add a fresh TLSF arena whose `largest_free_block`
                // covers the request, then retry.
                const size_t largest = ucache->host_zone_largest_free_block(zone);
                if (largest >= alloc_size) {
                    // There IS a single-chunk free block big enough, but the
                    // first-attempt allocation lost to a race or alignment
                    // detail. Do not grow — return failure so the caller
                    // surfaces the real error.
                    return nullptr;
                }
                const size_t need = alloc_size - largest + pinned_chunk_pool::DEFAULT_ALIGNMENT;
                if (!ucache->host_zone_grow(zone, need)) {
                    return nullptr;
                }
                return ucache->host_zone_alloc(zone, alloc_size, pinned_chunk_pool::DEFAULT_ALIGNMENT);
            };
            if (req.intent.constraints.use_pinned_pool) {
                if (!ucache->host_zones_configured()) {
                    ptr = ucache->host_pool_alloc(alloc_size, pinned_chunk_pool::DEFAULT_ALIGNMENT);
                } else {
                    host_zone_id pool_zone = select_zone();
                    ptr                    = try_zone_alloc_contiguous(pool_zone);
                    if (ptr) {
                        zone_managed      = true;
                        out->zone_managed = true;
                        out->host_zone    = pool_zone;
                    }
                }
                if (ptr) {
                    uses_pinned_pool = true;
                }
            } else if (ucache->host_zones_configured()) {
                host_zone_id zone = select_zone();
                ptr               = try_zone_alloc_contiguous(zone);
                zone_managed      = (ptr != nullptr);
                if (zone_managed) {
                    out->zone_managed = true;
                    out->host_zone    = zone;
                }
            } else {
                // Zones not configured: direct runtime allocation.  host_pool_alloc
                // is itself backed by a single-chunk TLSF and returns nullptr
                // rather than spanning chunks.
                ptr = ucache->host_pool_alloc(alloc_size, pinned_chunk_pool::DEFAULT_ALIGNMENT);
            }
            uses_pinned_pool = (ptr != nullptr);
        }
        if (!ptr) {
            // When host zones are configured, a WEIGHT zone miss means the zone is
            // intentionally sized and the caller should skip this allocation (e.g.
            // S1-PRELOAD will leave the expert un-registered and the inference path
            // will fetch it on-demand).  Falling back to sycl::malloc_host here
            // causes memory pressure that can evict mmap-backed tensor pages and
            // trigger SIGSEGV in the preload memcpy.
            if (req.intent.role == alloc_role::WEIGHT) {
                ptr = ggml_sycl_malloc_host(alloc_size, *req.queue, "unified_alloc:weight");
            } else {
                if (req.suppress_failure_log) {
                    GGML_SYCL_DEBUG("[UNIFIED-ALLOC] allocation miss: %zu bytes, tier=%s\n", alloc_size,
                                    alloc_tier_name(tier));
                } else {
                    GGML_LOG_ERROR(
                        "[UNIFIED-ALLOC] allocation failed: %zu bytes, tier=%s. "
                        "All inference allocations must go through the unified cache zone system. "
                        "Consider increasing host zone budgets or enabling dynamic pool growth.\n",
                        alloc_size, alloc_tier_name(tier));
                }
                return false;
            }
        }
    }

    if (!ptr) {
        return false;
    }

    // Track the allocation now that it succeeded.
    // Weight buffers are the primary model data allocated by ggml framework;
    // they must NOT count against the cache budget (reserved_) because the
    // cache manages SOA/XMX layouts in the REMAINING VRAM after weights.
    // Arena zones track non-weight allocations via zone_alloc/zone_reset.
    // No separate counter bookkeeping needed — zones are the single source of truth.
    // Weight allocations are excluded from runtime tracking (they live in the WEIGHT zone).

    runtime_alloc_record rec;
    rec.handle.ptr          = ptr;
    rec.handle.size         = alloc_size;
    rec.handle.device       = req.device;
    rec.handle.tier         = tier;
    rec.handle.role         = req.intent.role;
    rec.handle.category     = cat;
    rec.handle.alloc_id     = g_runtime_alloc_id.fetch_add(1, std::memory_order_relaxed);
    rec.handle.vram_zone    = out->vram_zone;  // Propagate zone routing set by zone_alloc paths above
    rec.handle.zone_managed = out->zone_managed;
    rec.handle.host_zone    = out->host_zone;
    rec.handle.all_segments = std::move(out->all_segments);  // Preserve segments from zone alloc path
    rec.queue               = req.queue;
    rec.uses_pinned_pool    = uses_pinned_pool;
    rec.zone_managed        = zone_managed;
    rec.from_arena          = from_arena;
    rec.vram_zone           = out->vram_zone;
    if (req.intent.cohort_id && req.intent.cohort_id[0] != '\0') {
        rec.cohort_id = req.intent.cohort_id;
    }

    {
        std::lock_guard<std::mutex> lock(g_runtime_alloc_mutex);
        if (g_runtime_alloc_registry.find(ptr) != g_runtime_alloc_registry.end()) {
            GGML_LOG_ERROR("[UNIFIED-ALLOC] duplicate pointer registration ptr=%p size=%zu tier=%s\n", ptr, alloc_size,
                           alloc_tier_name(tier));
            // Arena zones handle their own accounting — no separate counter sub needed.
            if (uses_pinned_pool && !zone_managed) {
                if (auto * ucache = get_unified_cache_for_device(req.device)) {
                    ucache->host_pool_free(ptr, alloc_size);
                }
            } else if (uses_pinned_pool) {
                // Zone-managed host allocations are released by zone reset.
            } else {
                sycl::free(ptr, *req.queue);
            }
            return false;
        }
        g_runtime_alloc_registry.emplace(ptr, rec);
        if (!rec.cohort_id.empty()) {
            g_runtime_cohort_tier[rec.cohort_id] = tier;
        }
    }

    *out = rec.handle;
    offload_stats_note_alloc(tier);
    return true;
}

bool acquire_offload_buffer(const offload_buffer_request & req_in, offload_buffer_lease * out) {
    if (out == nullptr) {
        return false;
    }
    *out = {};

    if (req_in.queue == nullptr) {
        GGML_LOG_ERROR("[UNIFIED-ALLOC] offload pool acquire failed: missing queue\n");
        return false;
    }

    offload_buffer_request req = req_in;
    if (req.device < 0) {
        req.device = get_device_id_from_queue(*req.queue);
    }
    if (req.device < 0 || req.device >= GGML_SYCL_MAX_DEVICES) {
        GGML_LOG_ERROR("[UNIFIED-ALLOC] offload pool acquire failed: invalid device=%d\n", req.device);
        return false;
    }

    const size_t alignment  = std::max<size_t>(req.alignment, 64);
    const size_t alloc_size = align_up(std::max<size_t>(req.size, 1), alignment);

    alloc_request areq{};
    areq.queue  = req.queue;
    areq.device = req.device;
    areq.size   = alloc_size;
    areq.intent = req.intent;
    if (areq.intent.role == alloc_role::OTHER) {
        areq.intent.role = alloc_role::STAGING;
    }
    if (areq.intent.category == runtime_category::OTHER) {
        areq.intent.category = runtime_category::HOST_COMPUTE;
    }
    const bool has_explicit_tier_constraint =
        areq.intent.constraints.must_host_pinned || areq.intent.constraints.must_device;

    // Default staging roles to host-pinned, but preserve explicit caller
    // constraints so compute-buffer users can intentionally target VRAM.
    if (!has_explicit_tier_constraint) {
        switch (req.role) {
            case offload_buffer_role::STAGING_SRC0:
            case offload_buffer_role::STAGING_SRC1:
            case offload_buffer_role::STAGING_DST:
            case offload_buffer_role::RETAINED_SCRATCH:
                areq.intent.constraints.must_host_pinned = true;
                areq.intent.constraints.use_pinned_pool  = true;
                break;
            default:
                break;
        }
    }

    const alloc_tier tier = unified_select_tier(areq);
    offload_pool_key key{};
    key.device    = req.device;
    key.role      = req.role;
    key.tier      = tier;
    key.category  = areq.intent.category;
    key.alignment = alignment;

    {
        std::lock_guard<std::mutex> lock(g_offload_pool_mutex);
        auto                        it_bucket = g_offload_pool_free.find(key);
        if (it_bucket != g_offload_pool_free.end()) {
            size_t best_idx  = std::numeric_limits<size_t>::max();
            size_t best_size = std::numeric_limits<size_t>::max();
            for (size_t i = 0; i < it_bucket->second.size(); ++i) {
                void * ptr     = it_bucket->second[i];
                auto   slot_it = g_offload_pool_slots.find(ptr);
                if (slot_it == g_offload_pool_slots.end() || slot_it->second.in_use) {
                    continue;
                }
                if (slot_it->second.handle.size < alloc_size) {
                    continue;
                }
                if (slot_it->second.handle.size < best_size) {
                    best_size = slot_it->second.handle.size;
                    best_idx  = i;
                }
            }
            if (best_idx != std::numeric_limits<size_t>::max()) {
                void * ptr = it_bucket->second[best_idx];
                it_bucket->second.erase(it_bucket->second.begin() + static_cast<ptrdiff_t>(best_idx));
                if (it_bucket->second.empty()) {
                    g_offload_pool_free.erase(it_bucket);
                }

                auto slot_it = g_offload_pool_slots.find(ptr);
                GGML_ASSERT(slot_it != g_offload_pool_slots.end());
                slot_it->second.in_use   = true;
                slot_it->second.lease_id = g_offload_pool_lease_id.fetch_add(1, std::memory_order_relaxed);
                out->handle              = slot_it->second.handle;
                out->lease_id            = slot_it->second.lease_id;
                out->valid               = true;
                offload_stats_note_pool_hit();
                return true;
            }
        }
    }

    alloc_handle h{};
    if (!unified_alloc(areq, &h) || h.ptr == nullptr) {
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(g_offload_pool_mutex);
        offload_pool_slot           slot{};
        slot.handle                 = h;
        slot.key                    = key;
        slot.in_use                 = true;
        slot.lease_id               = g_offload_pool_lease_id.fetch_add(1, std::memory_order_relaxed);
        g_offload_pool_slots[h.ptr] = slot;
        out->handle                 = h;
        out->lease_id               = slot.lease_id;
        out->valid                  = true;
    }
    offload_stats_note_pool_miss();
    return true;
}

bool release_offload_buffer(const offload_buffer_lease & lease) {
    if (!lease.valid || lease.handle.ptr == nullptr) {
        return true;
    }

    std::lock_guard<std::mutex> lock(g_offload_pool_mutex);
    auto                        it = g_offload_pool_slots.find(lease.handle.ptr);
    if (it == g_offload_pool_slots.end()) {
        if (unified_alloc_strict_mode()) {
            GGML_LOG_ERROR("[UNIFIED-ALLOC] strict offload-pool release unknown ptr=%p lease=%llu\n", lease.handle.ptr,
                           static_cast<unsigned long long>(lease.lease_id));
        }
        return false;
    }
    if (!it->second.in_use || it->second.lease_id != lease.lease_id) {
        if (unified_alloc_strict_mode()) {
            GGML_LOG_ERROR("[UNIFIED-ALLOC] strict offload-pool stale lease ptr=%p lease=%llu active=%llu in_use=%d\n",
                           lease.handle.ptr, static_cast<unsigned long long>(lease.lease_id),
                           static_cast<unsigned long long>(it->second.lease_id), it->second.in_use ? 1 : 0);
        }
        return false;
    }

    it->second.in_use = false;
    g_offload_pool_free[it->second.key].push_back(lease.handle.ptr);
    return true;
}

void offload_buffer_pool_trim(int device) {
    std::vector<alloc_handle> free_list;
    {
        std::lock_guard<std::mutex> lock(g_offload_pool_mutex);
        for (auto it = g_offload_pool_slots.begin(); it != g_offload_pool_slots.end();) {
            const offload_pool_slot & slot = it->second;
            if (!slot.in_use && (device < 0 || slot.key.device == device)) {
                free_list.push_back(slot.handle);
                it = g_offload_pool_slots.erase(it);
            } else {
                ++it;
            }
        }
        g_offload_pool_free.clear();
        for (const auto & kv : g_offload_pool_slots) {
            if (!kv.second.in_use) {
                g_offload_pool_free[kv.second.key].push_back(kv.first);
            }
        }
    }

    for (const alloc_handle & h : free_list) {
        (void) unified_free(h);
    }
}

mem_handle unified_allocate(const alloc_request & req) {
    alloc_handle handle;
    if (!unified_alloc(req, &handle) || !handle.ptr) {
        return mem_handle{};
    }
    bool on_device = (handle.tier == alloc_tier::DEVICE_VRAM);
    int  dev       = on_device ? handle.device : mem_handle::HOST_DEVICE;
    return mem_handle::from_direct(handle.ptr, GGML_LAYOUT_AOS, on_device, dev);
}

mem_handle scoped_unified_alloc::as_mem_handle() const {
    if (!handle_.ptr) {
        return mem_handle{};
    }
    bool on_device = (handle_.tier == alloc_tier::DEVICE_VRAM);
    int  dev       = on_device ? handle_.device : mem_handle::HOST_DEVICE;
    return mem_handle::from_direct(handle_.ptr, GGML_LAYOUT_AOS, on_device, dev);
}

mem_handle alloc_handle::as_mem_handle() const {
    if (!ptr) {
        return mem_handle{};
    }
    bool on_device = (tier == alloc_tier::DEVICE_VRAM);
    int  dev       = on_device ? device : mem_handle::HOST_DEVICE;
    return mem_handle::from_direct(ptr, GGML_LAYOUT_AOS, on_device, dev);
}

bool unified_lookup(void * ptr, alloc_handle * out) {
    if (out == nullptr) {
        return false;
    }
    *out = {};
    if (ptr == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_runtime_alloc_mutex);
    auto                        it = g_runtime_alloc_registry.find(ptr);
    if (it == g_runtime_alloc_registry.end()) {
        return false;
    }
    *out = it->second.handle;
    return true;
}

bool unified_free_ptr(void * ptr, int expected_device) {
    if (ptr == nullptr) {
        return true;
    }

    runtime_alloc_record rec;
    {
        std::lock_guard<std::mutex> lock(g_runtime_alloc_mutex);
        auto                        it = g_runtime_alloc_registry.find(ptr);
        if (it == g_runtime_alloc_registry.end()) {
            if (unified_alloc_strict_mode()) {
                GGML_LOG_ERROR("[UNIFIED-ALLOC] strict unknown free ptr=%p expected_device=%d\n", ptr, expected_device);
            }
            return false;
        }
        if (expected_device >= 0 && expected_device != it->second.handle.device) {
            GGML_LOG_ERROR("[UNIFIED-ALLOC] free device mismatch ptr=%p expected=%d actual=%d\n", ptr, expected_device,
                           it->second.handle.device);
            return false;
        }
        rec = it->second;
    }

    if (!unified_free_record(rec)) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_runtime_alloc_mutex);
    auto                        it = g_runtime_alloc_registry.find(ptr);
    if (it != g_runtime_alloc_registry.end() && it->second.handle.alloc_id == rec.handle.alloc_id) {
        g_runtime_alloc_registry.erase(it);
    }
    return true;
}

bool unified_free(const alloc_handle & handle) {
    if (handle.ptr == nullptr) {
        return true;
    }

    {
        std::lock_guard<std::mutex> lock(g_runtime_alloc_mutex);
        auto                        it = g_runtime_alloc_registry.find(handle.ptr);
        if (it == g_runtime_alloc_registry.end()) {
            if (unified_alloc_strict_mode()) {
                GGML_LOG_ERROR("[UNIFIED-ALLOC] strict stale/unknown handle free ptr=%p alloc_id=%llu\n", handle.ptr,
                               static_cast<unsigned long long>(handle.alloc_id));
            }
            return false;
        }
        if (handle.alloc_id != 0 && it->second.handle.alloc_id != handle.alloc_id) {
            GGML_LOG_ERROR("[UNIFIED-ALLOC] stale handle free ptr=%p expected_alloc_id=%llu actual_alloc_id=%llu\n",
                           handle.ptr, static_cast<unsigned long long>(handle.alloc_id),
                           static_cast<unsigned long long>(it->second.handle.alloc_id));
            return false;
        }
    }

    return unified_free_ptr(handle.ptr, handle.device);
}

// ============================================================================
// Simplified allocation facade
// ============================================================================

// Map alloc_category to the internal alloc_role / runtime_category / constraints
// used by the existing unified_alloc machinery.
static void category_to_intent(alloc_category cat, alloc_intent & intent) {
    switch (cat) {
        case alloc_category::WEIGHT:
            intent.role     = alloc_role::WEIGHT;
            intent.category = runtime_category::OTHER;
            break;
        case alloc_category::KV_CACHE:
            intent.role     = alloc_role::KV;
            intent.category = runtime_category::KV_CACHE;
            break;
        case alloc_category::COMPUTE_SCRATCH:
            intent.role     = alloc_role::COMPUTE;
            intent.category = runtime_category::COMPUTE;
            break;
        case alloc_category::STAGING:
            intent.role     = alloc_role::STAGING;
            intent.category = runtime_category::STAGING;
            break;
        case alloc_category::CONTROL:
            intent.role                         = alloc_role::CONTROL;
            intent.category                     = runtime_category::CONTROL;
            intent.constraints.must_host_pinned = true;
            intent.constraints.must_device      = false;
            intent.constraints.use_pinned_pool  = false;
            break;
        case alloc_category::EXPERT_CACHE:
            intent.role     = alloc_role::EXPERT_STAGING;
            intent.category = runtime_category::EXPERT_CACHE;
            break;
    }
}

// Returns true if a category is eligible for host-pinned fallback when VRAM is full.
static bool category_allows_host_fallback(alloc_category cat) {
    return cat == alloc_category::COMPUTE_SCRATCH || cat == alloc_category::STAGING;
}

int alloc_category_priority(alloc_category cat) {
    switch (cat) {
        case alloc_category::COMPUTE_SCRATCH:
            return 0;  // pre-reserved arena, must stay in VRAM
        case alloc_category::KV_CACHE:
            return 1;  // hot KV: latency-critical, always VRAM
        case alloc_category::WEIGHT:
            return 2;  // attention weights: used every token
        case alloc_category::CONTROL:
            return 4;  // host-control, not a VRAM eviction participant
        case alloc_category::EXPERT_CACHE:
            return 3;  // MoE experts: evictable via LRU/frequency
        case alloc_category::STAGING:
            return 4;  // always host-ok
    }
    return 4;          // unknown → treat as staging
}

unified_alloc_result unified_cache_allocate(int device, size_t size, alloc_category category, sycl::queue & queue) {
    unified_alloc_result result{};
    if (size == 0) {
        return result;
    }

    alloc_request req{};
    req.queue  = &queue;
    req.device = device;
    req.size   = size;
    category_to_intent(category, req.intent);

    alloc_handle handle{};
    if (unified_alloc(req, &handle)) {
        result.ptr  = handle.ptr;
        result.tier = handle.tier;
        result.size = handle.size;
        return result;
    }

    // VRAM allocation failed — try host-pinned fallback for eligible categories.
    if (category_allows_host_fallback(category) && !req.intent.constraints.must_device) {
        req.intent.constraints.must_host_pinned = true;
        req.intent.constraints.must_device      = false;
        if (unified_alloc(req, &handle)) {
            result.ptr  = handle.ptr;
            result.tier = handle.tier;
            result.size = handle.size;
            return result;
        }
    }

    // All attempts failed
    return result;
}

void unified_cache_deallocate(void * ptr, int device) {
    if (ptr == nullptr) {
        return;
    }
    unified_free_ptr(ptr, device);
}

bool unified_alloc_validate_registry(int device, const char * where) {
    std::array<size_t, GGML_SYCL_MAX_DEVICES> registry_device{};
    size_t                                    registry_host = 0;
    {
        std::lock_guard<std::mutex> lock(g_runtime_alloc_mutex);
        for (const auto & kv : g_runtime_alloc_registry) {
            const alloc_handle & h = kv.second.handle;
            if (h.tier == alloc_tier::DEVICE_VRAM) {
                if (h.device >= 0 && h.device < GGML_SYCL_MAX_DEVICES) {
                    registry_device[h.device] += h.size;
                }
            } else if (h.tier == alloc_tier::HOST_PINNED || h.tier == alloc_tier::MMAP_TRACKED) {
                registry_host += h.size;
            }
        }
    }

    bool ok = true;
    for (int d = 0; d < GGML_SYCL_MAX_DEVICES; ++d) {
        if (device >= 0 && d != device) {
            continue;
        }
        const size_t tracked = unified_cache_arena_non_weight_used(d);
        const size_t reg     = registry_device[d];
        if (tracked != reg) {
            ok = false;
            GGML_LOG_WARN("[UNIFIED-ALLOC] managed registry mismatch%s%s dev=%d tracked=%zu registry=%zu\n",
                          where ? " at " : "", where ? where : "", d, tracked, reg);
        }
    }
    const size_t tracked_host = g_runtime_reserved_host_bytes.load(std::memory_order_relaxed);
    if (tracked_host != registry_host) {
        ok = false;
        GGML_LOG_WARN("[UNIFIED-ALLOC] managed registry mismatch%s%s host tracked=%zu registry=%zu\n",
                      where ? " at " : "", where ? where : "", tracked_host, registry_host);
    }
    return ok;
}

void unified_cache_set_graph_compute_active(bool active) {
    g_graph_compute_active.store(active, std::memory_order_release);
}

bool unified_cache_is_graph_compute_active() {
    return g_graph_compute_active.load(std::memory_order_acquire);
}

bool unified_cache_has_pending_deferred_frees(int device) {
    int effective_device = resolve_effective_device(device);
    if (effective_device < 0) {
        return false;
    }
    unified_cache * cache = get_cache_shared(effective_device);
    if (!cache) {
        return false;
    }
    return cache->has_pending_deferred_frees();
}

// unified_cache_add/sub_runtime_bytes removed — arena zones are the single source of truth.
// Zone_alloc/zone_reset handle accounting internally.

size_t unified_cache_get_runtime_bytes(int device) {
    return unified_cache_arena_non_weight_used(device);
}

size_t unified_cache_get_runtime_bytes_by_category(int device, runtime_category cat) {
    // Per-category tracking replaced by zone-based queries.
    // Note: not every runtime_category maps 1:1 to a zone; categories that
    // previously shared counters are collapsed to the best-fit zone.
    auto * cache = get_unified_cache_for_device(device);
    if (!cache || !cache->arena_active()) {
        return 0;
    }
    switch (cat) {
        case runtime_category::KV_CACHE:
            return cache->zone_used(vram_zone_id::KV);
        case runtime_category::COMPUTE:
            return cache->zone_used(vram_zone_id::RUNTIME);
        case runtime_category::STAGING:
            return cache->zone_used(vram_zone_id::SCRATCH);
        case runtime_category::GRAPH:
            return cache->zone_used(vram_zone_id::RUNTIME);
        case runtime_category::HOST_COMPUTE:
            return 0;  // Host memory — not tracked by VRAM zones
        case runtime_category::EXPERT_CACHE:
            return cache->zone_used(vram_zone_id::RUNTIME);
        case runtime_category::CONTROL:
            return 0;  // Host-pinned control memory, not VRAM
        case runtime_category::OTHER:
            return cache->zone_used(vram_zone_id::ONEDNN);
        case runtime_category::COUNT:
            return 0;  // Sentinel — not a real category
    }
    return 0;
}

// Host runtime tracking: atomic counters for diagnostics.
// Host arena zones handle their own capacity tracking via TLSF.
void unified_cache_add_runtime_host_bytes(size_t bytes) {
    if (bytes == 0) {
        return;
    }
    g_runtime_reserved_host_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

void unified_cache_sub_runtime_host_bytes(size_t bytes) {
    if (bytes == 0) {
        return;
    }
    size_t cur  = g_runtime_reserved_host_bytes.load(std::memory_order_relaxed);
    size_t next = cur > bytes ? cur - bytes : 0;
    g_runtime_reserved_host_bytes.store(next, std::memory_order_relaxed);
}

size_t unified_cache_get_runtime_host_bytes() {
    // Pure atomic read — no lock needed
    return g_runtime_reserved_host_bytes.load(std::memory_order_relaxed);
}

// === Budget Query API ===

size_t unified_cache_available_for_compute(int device) {
    int effective_device = resolve_effective_device(device);
    if (effective_device < 0) {
        return 0;
    }
    unified_cache * cache = get_cache_shared(effective_device);
    if (!cache) {
        return 0;
    }

    // Use the unified total-available view that sums BOTH budget channels
    // (weights from used_ + runtime from arena zone usage).
    // Arena zones are the single source of truth for runtime tracking —
    // zone_used() values are always current, eliminating the TOCTOU gap
    // that existed when reserved_ was driven by global counters.
    //
    // We still query live VRAM as a safety cap: if the driver reports less free
    // memory than the budget says, use the lower value.
    const size_t budget_avail = unified_cache_total_available_bytes(device);
    size_t       free_vram = 0, total_vram = 0;
    ggml_backend_sycl_get_device_memory(effective_device, &free_vram, &total_vram);
    if (free_vram == 0) {
        return budget_avail;
    }
    // Reserve 256 MB headroom for driver structures, compute scratch, and
    // transient allocations that come and go during inference.
    const size_t headroom   = size_t(256) << 20;
    const size_t live_avail = free_vram > headroom ? free_vram - headroom : 0;
    return std::min(budget_avail, live_avail);
}

size_t unified_cache_total_committed_bytes(int device) {
    int effective_device = resolve_effective_device(device);
    if (effective_device < 0) {
        return 0;
    }
    // Runtime bytes from arena zones: KV + ONEDNN + RUNTIME + SCRATCH usage.
    const size_t    runtime = unified_cache_arena_non_weight_used(effective_device);
    unified_cache * cache   = get_cache_shared(effective_device);
    const size_t    weights = cache ? cache->weight_bytes() : 0;
    return runtime + weights;
}

size_t unified_cache_total_available_bytes(int device) {
    int effective_device = resolve_effective_device(device);
    if (effective_device < 0) {
        return 0;
    }
    unified_cache * cache = get_cache_shared(effective_device);
    if (!cache) {
        return 0;
    }
    const size_t base      = cache->base_budget();
    const size_t committed = unified_cache_total_committed_bytes(device);
    return base > committed ? base - committed : 0;
}

size_t unified_cache_total_managed(int device) {
    int effective_device = resolve_effective_device(device);
    if (effective_device < 0) {
        return 0;
    }
    unified_cache * cache = get_cache_shared(effective_device);
    if (!cache) {
        return 0;
    }
    return cache->base_budget();
}

size_t unified_cache_weight_bytes(int device) {
    int effective_device = resolve_effective_device(device);
    if (effective_device < 0) {
        return 0;
    }
    unified_cache * cache = get_cache_shared(effective_device);
    if (!cache) {
        return 0;
    }
    return cache->weight_bytes();
}

size_t unified_cache_get_layer_vram_bytes(int device, int layer_id) {
    int effective_device = resolve_effective_device(device);
    if (effective_device < 0) {
        return 0;
    }
    unified_cache * cache = get_cache_shared(effective_device);
    if (!cache) {
        return 0;
    }
    return cache->get_layer_vram_bytes(layer_id);
}

size_t unified_cache_evictable_expert_bytes(int device) {
    int effective_device = resolve_effective_device(device);
    if (effective_device < 0) {
        return 0;
    }
    unified_cache * cache = get_cache_shared(effective_device);
    if (!cache) {
        return 0;
    }
    return cache->evictable_expert_bytes();
}

// === Budget Summary Diagnostic ===

void unified_cache_log_budget_summary(int device) {
    int effective_device = resolve_effective_device(device);
    if (effective_device < 0) {
        return;
    }
    unified_cache * cache_ptr = get_cache_shared(effective_device);
    if (!cache_ptr) {
        return;
    }
    auto &       cache = *cache_ptr;
    const size_t base  = cache.base_budget();
    const size_t wt    = cache.weight_bytes();
    const size_t rt    = unified_cache_arena_non_weight_used(effective_device);
    const size_t eff   = cache.budget();
    const size_t avl   = cache.available();

    const size_t avail_for_wt = base > rt ? base - rt : 0;
    int          budget_pct   = 90;
    const char * env_pct      = std::getenv("GGML_SYCL_VRAM_BUDGET_PCT");
    if (env_pct) {
        budget_pct = std::max(1, std::min(100, std::atoi(env_pct)));
    }
    // Compute model-exceeds-VRAM directly from model size vs available budget
    const size_t model_size    = ggml_sycl_get_model_size();
    size_t       moe_total_log = 0;
    int          n_exp_log = 0, n_exp_used_log = 0;
    ggml_sycl_get_moe_info(&moe_total_log, &n_exp_log, &n_exp_used_log);
    const size_t effective_model =
        compute_moe_effective_weight_bytes(model_size, moe_total_log, n_exp_log, n_exp_used_log);
    const bool exceeds = (model_size > 0) && (effective_model > avail_for_wt);

    const size_t committed = unified_cache_total_committed_bytes(device);
    const size_t total_avl = unified_cache_total_available_bytes(device);

    GGML_LOG_INFO(
        "[UNIFIED-CACHE] Budget summary for device %d:\n"
        "  Total VRAM budget:    %8.1f MB\n"
        "  Weight bytes (used_): %8.1f MB\n"
        "  Runtime reserved:     %8.1f MB\n"
        "  Total committed:      %8.1f MB  (weights + runtime)\n"
        "  Total available:      %8.1f MB  (budget - committed)\n"
        "  Effective budget:     %8.1f MB\n"
        "  Available for alloc:  %8.1f MB\n"
        "  Avail for weights:    %8.1f MB\n"
        "  Budget pct:           %8d %%\n"
        "  Model exceeds VRAM:   %8s\n",
        device, base / (1024.0f * 1024.0f), wt / (1024.0f * 1024.0f), rt / (1024.0f * 1024.0f),
        committed / (1024.0f * 1024.0f), total_avl / (1024.0f * 1024.0f), eff / (1024.0f * 1024.0f),
        avl / (1024.0f * 1024.0f), avail_for_wt / (1024.0f * 1024.0f), budget_pct, exceeds ? "yes" : "no");

    // Per-category runtime breakdown from arena zones
    static const char * cat_names[] = { "KV_CACHE",     "COMPUTE",      "STAGING", "GRAPH",
                                        "HOST_COMPUTE", "EXPERT_CACHE", "OTHER" };
    GGML_LOG_INFO("[UNIFIED-CACHE] Runtime breakdown for device %d:\n", device);
    auto * cat_cache = get_unified_cache_for_device(effective_device);
    if (cat_cache && cat_cache->arena_active()) {
        GGML_LOG_INFO("  %-12s %8.1f MB\n", "KV_ZONE", cat_cache->zone_used(vram_zone_id::KV) / (1024.0 * 1024.0));
        GGML_LOG_INFO("  %-12s %8.1f MB\n", "ONEDNN_ZONE",
                      cat_cache->zone_used(vram_zone_id::ONEDNN) / (1024.0 * 1024.0));
        GGML_LOG_INFO("  %-12s %8.1f MB\n", "RUNTIME_ZONE",
                      cat_cache->zone_used(vram_zone_id::RUNTIME) / (1024.0 * 1024.0));
        GGML_LOG_INFO("  %-12s %8.1f MB\n", "SCRATCH_ZONE",
                      cat_cache->zone_used(vram_zone_id::SCRATCH) / (1024.0 * 1024.0));
    }
    // Legacy per-category counters removed — arena zones are the single source of truth.
    size_t cat_sum = 0;
    if (cat_cache && cat_cache->arena_active()) {
        cat_sum = cat_cache->zone_used(vram_zone_id::KV) + cat_cache->zone_used(vram_zone_id::ONEDNN) +
                  cat_cache->zone_used(vram_zone_id::RUNTIME) + cat_cache->zone_used(vram_zone_id::SCRATCH);
    }
    if (rt > cat_sum + (1024 * 1024)) {  // >1 MB untagged
        GGML_LOG_INFO("  %-12s %8.1f MB (tracked outside zones)\n", "UNTAGGED", (rt - cat_sum) / (1024.0f * 1024.0f));
    }

    // Validate accounting consistency:
    //   weight_bytes + available should equal effective budget
    //   (effective budget = base_budget - internal reserved)
    const size_t tolerance   = 1024 * 1024;  // 1 MB
    const size_t wt_plus_avl = wt + avl;
    if (wt_plus_avl > eff + tolerance || eff > wt_plus_avl + tolerance) {
        GGML_LOG_WARN(
            "[UNIFIED-CACHE] Accounting mismatch on device %d: "
            "weights(%.1f) + available(%.1f) = %.1f MB, "
            "but effective_budget = %.1f MB (delta = %.1f MB)\n",
            device, wt / (1024.0f * 1024.0f), avl / (1024.0f * 1024.0f), wt_plus_avl / (1024.0f * 1024.0f),
            eff / (1024.0f * 1024.0f),
            (double) (wt_plus_avl > eff ? wt_plus_avl - eff : eff - wt_plus_avl) / (1024.0 * 1024.0));
    }

    // Sanity: used_ should not exceed effective budget
    if (wt > eff + tolerance) {
        GGML_LOG_WARN(
            "[UNIFIED-CACHE] Over-allocation on device %d: "
            "weight_bytes(%.1f MB) > effective_budget(%.1f MB)\n",
            device, wt / (1024.0f * 1024.0f), eff / (1024.0f * 1024.0f));
    }

    // Diagnostic: flag if external runtime tracker diverges from internal reserved
    const size_t implied_reserved = (base > eff) ? (base - eff) : 0;
    if (rt > implied_reserved + tolerance || implied_reserved > rt + tolerance) {
        GGML_LOG_INFO(
            "[UNIFIED-CACHE] Note: external runtime tracker (%.1f MB) "
            "differs from internal reserved (%.1f MB) on device %d\n",
            rt / (1024.0f * 1024.0f), implied_reserved / (1024.0f * 1024.0f), device);
    }
}

void unified_cache_seal_layout_pool(int device) {
    int effective_device = resolve_effective_device(device);
    if (effective_device < 0) {
        return;
    }
    unified_cache * cache = get_cache_shared(effective_device);
    if (!cache) {
        return;
    }
    cache->seal_layout_pool();
    GGML_LOG_INFO("[UNIFIED-CACHE] Layout pool sealed on device %d\n", effective_device);
}

bool unified_cache_is_budget_exceeded(int device) {
    int effective_device = resolve_effective_device(device);
    if (effective_device < 0) {
        return false;
    }
    unified_cache * cache = get_cache_shared(effective_device);
    if (!cache) {
        return false;
    }
    // Check the unified view: total committed (weights + runtime) > base budget.
    // The per-cache is_budget_exceeded() only checks used_ > budget_ which can
    // miss over-allocation when arena zone usage grew after weight placement.
    const size_t committed = unified_cache_total_committed_bytes(device);
    const size_t base      = cache->base_budget();
    return cache->is_budget_exceeded() || committed > base;
}

bool unified_cache_has_evictions() {
    std::shared_lock<std::shared_mutex> read_lock(g_cache_rw_mutex);
    for (auto & [device_id, cache] : g_device_caches) {
        if (cache && cache->has_evictions()) {
            return true;
        }
    }
    return false;
}

// === Budget Export API ===

size_t compute_moe_effective_weight_bytes(size_t total_weight_bytes,
                                          size_t expert_total_bytes,
                                          int    n_expert,
                                          int    n_expert_used) {
    if (n_expert <= 0 || n_expert_used <= 0 || expert_total_bytes == 0) {
        return total_weight_bytes;  // Dense model, no savings
    }
    // Active expert fraction + headroom for expert cache churn
    // Use 1.5x active ratio to account for recently-used experts still in cache
    double active_ratio    = static_cast<double>(n_expert_used) / n_expert;
    double effective_ratio = std::min(1.0, active_ratio * 1.5);
    size_t expert_savings  = static_cast<size_t>(expert_total_bytes * (1.0 - effective_ratio));
    return (expert_savings <= total_weight_bytes) ? total_weight_bytes - expert_savings : 0;
}

unified_budget_info unified_cache_get_budget_info(int device) {
    unified_budget_info info = {};
    info.device_id           = device;

    if (device < 0 || device >= GGML_SYCL_MAX_DEVICES) {
        return info;  // Return zeroed struct for invalid device
    }

    // Read budget percentage once, clamp to [1,100]
    int          pct     = 90;
    const char * env_pct = std::getenv("GGML_SYCL_VRAM_BUDGET_PCT");
    if (env_pct) {
        pct = std::max(1, std::min(100, std::atoi(env_pct)));
    }
    info.budget_pct = pct;

    size_t free_mem = 0, total_mem = 0;
    ggml_backend_sycl_get_device_memory(device, &free_mem, &total_mem);
    info.total_vram = ggml_sycl_info().devices[device].total_vram;
    if (info.total_vram == 0) {
        info.total_vram = total_mem > 0 ? total_mem : free_mem;
    }

    auto * cache = get_unified_cache_for_device(device);
    if (cache) {
        info.budget_bytes    = unified_cache_total_managed(device);
        info.weight_bytes    = unified_cache_weight_bytes(device);
        info.runtime_bytes   = unified_cache_get_runtime_bytes(device);
        info.total_committed = unified_cache_total_committed_bytes(device);
        info.available_for_weights =
            info.budget_bytes > info.runtime_bytes ? info.budget_bytes - info.runtime_bytes : 0;
        info.total_available = unified_cache_total_available_bytes(device);
    } else {
        // Cache not yet initialized — use raw calculation
        info.budget_bytes     = static_cast<size_t>(info.total_vram * (static_cast<double>(pct) / 100.0));
        const size_t headroom = std::max(size_t(256) << 20, info.total_vram / 10);
        if (info.total_vram > headroom && info.budget_bytes > info.total_vram - headroom) {
            info.budget_bytes = info.total_vram - headroom;
        }
        info.available_for_weights = info.budget_bytes;
        info.total_committed       = 0;
        info.total_available       = info.budget_bytes;
    }

    // Populate MoE fields from tensor inventory
    size_t moe_total = 0;
    int    n_exp = 0, n_exp_used = 0;
    ggml_sycl_get_moe_info(&moe_total, &n_exp, &n_exp_used);
    info.expert_weight_bytes = moe_total;
    info.n_expert_total      = n_exp;
    info.n_expert_used       = n_exp_used;
    info.active_expert_bytes = compute_moe_effective_weight_bytes(moe_total, moe_total, n_exp, n_exp_used);

    // model_exceeds_vram removed — unified non-blocking cache handles all model sizes

    return info;
}

size_t unified_cache_get_margin_bytes(int device) {
    auto info = unified_cache_get_budget_info(device);
    if (info.available_for_weights > info.weight_bytes) {
        return info.available_for_weights - info.weight_bytes;
    }
    return 0;
}

bool unified_cache_should_offload_kv(int device, size_t kv_estimate_bytes) {
    // Check env var override first
    static std::atomic<int> cached_env{ -2 };  // -2 = not checked
    int                     env_val = cached_env.load(std::memory_order_acquire);
    if (env_val == -2) {
        const char * env_kv = std::getenv("GGML_SYCL_KV_HOST");
        env_val             = env_kv ? std::atoi(env_kv) : -1;
        cached_env.store(env_val, std::memory_order_release);
    }
    if (env_val == 1) {
        return true;
    }
    if (env_val == 0) {
        return false;
    }

    // HOST_COMPUTE mode: auto-offload KV to host-pinned memory.
    // When CPU offload runs with host-pinned compute buffers, KV on host
    // eliminates GPU islands (SET_ROWS writes + FLASH_ATTN reads go through
    // PCIe zero-copy instead of requiring CPU↔GPU transitions).
    static std::atomic<int> cached_hc{ -1 };
    int                     hc_val = cached_hc.load(std::memory_order_acquire);
    if (hc_val == -1) {
        const char * env_hc = std::getenv("GGML_SYCL_HOST_COMPUTE");
        hc_val              = (env_hc && std::atoi(env_hc) != 0) ? 1 : 0;
        cached_hc.store(hc_val, std::memory_order_release);
    }
    if (hc_val == 1) {
        return true;
    }

    auto info = unified_cache_get_budget_info(device);

    // Check if weight bytes already exceed available budget (model doesn't fit in VRAM)
    if (info.weight_bytes > 0 && info.weight_bytes > info.available_for_weights) {
        return true;
    }

    // Check total model size vs budget for models not yet fully loaded
    const size_t model_size = ggml_sycl_get_model_size();
    if (model_size > 0 && model_size > info.available_for_weights) {
        return true;
    }

    // If KV estimate provided, check if it would push us over budget
    if (kv_estimate_bytes > 0 && info.available_for_weights > info.weight_bytes) {
        size_t margin = info.available_for_weights - info.weight_bytes;
        if (kv_estimate_bytes > margin) {
            return true;
        }
    }

    return false;
}

// === MoE Cache Helpers ===

void unpin_all_experts() {
    std::shared_lock<std::shared_mutex> read_lock(g_cache_rw_mutex);
    // Unpin in all caches (cache pointers stable during inference)
    for (auto & [device_id, cache] : g_device_caches) {
        if (cache) {
            cache->unpin_experts();
        }
    }
}

// === Routing-Aware Expert Pre-staging ===

// Helper: Create a cache ID for an expert that matches the dispatch path's key generation.
// Uses semantic tensor identity (model id + tensor name + expert id + type/dims), not graph-local
// tensor/extra wrapper identity, so prestaged entries are found across PP/TG graph transitions.
static ggml_sycl_cache_id make_expert_cache_id(const char * tensor_name,
                                               uint64_t     cache_uuid,
                                               uint32_t     model_id,
                                               int          expert_id,
                                               ggml_type    tensor_type = GGML_TYPE_COUNT,
                                               int64_t      ne0         = 0,
                                               int64_t      ne1         = 0) {
    (void) cache_uuid;
    ggml_sycl_cache_id id{};

    // Use name-based key with expert_id suffix for per-expert uniqueness.
    // Matches ggml_sycl_get_moe_expert_cache_key in ggml-sycl.cpp.
    std::string expert_name = (tensor_name && tensor_name[0]) ? std::string(tensor_name) : std::string("unknown");
    expert_name += ":e";
    expert_name += std::to_string(expert_id);
    uint64_t name_hash = static_cast<uint64_t>(std::hash<std::string>()(expert_name));

    id.valid         = true;
    id.model_id      = model_id;
    id.has_gguf      = false;
    id.file_idx      = 0;
    id.file_offs     = 0;
    id.nbytes        = 0;
    id.name_hash     = name_hash;
    id.type          = tensor_type;
    id.tp_sharded    = false;
    id.tp_rank       = 0;
    id.tp_world_size = 1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        id.ne[i]           = 0;
        id.tp_local_ne[i]  = 0;
        id.tp_offset_ne[i] = 0;
    }
    id.ne[0] = ne0;
    id.ne[1] = ne1;
    id.ne[2] = 1;
    id.ne[3] = 1;

    id.aux_id = 0;

    return id;
}

prestage_result prestage_routed_experts(void *          queue_ptr,
                                        const int32_t * expert_ids,
                                        int             n_expert_used,
                                        int             n_tokens,
                                        const void *    weight_base_ptr,
                                        size_t          expert_stride,
                                        size_t          expert_size,
                                        int             layer_id,
                                        int             n_experts_total,
                                        int             device_id,
                                        const char *    tensor_name,
                                        uint64_t        cache_uuid,
                                        uint32_t        model_id,
                                        ggml_type       tensor_type,
                                        int64_t         ne0,
                                        int64_t         ne1) {
    prestage_result result{};
    result.n_unique = 0;
    result.success  = false;

    // Validate inputs
    if (!expert_ids || n_expert_used <= 0 || n_tokens <= 0 || !weight_base_ptr) {
        GGML_SYCL_DEBUG("[PRESTAGE] Invalid inputs: expert_ids=%p, n_expert_used=%d, n_tokens=%d, weight_base=%p\n",
                        (const void *) expert_ids, n_expert_used, n_tokens, weight_base_ptr);
        return result;
    }

    // Get unified cache for this device
    unified_cache * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        GGML_SYCL_DEBUG("[PRESTAGE] No unified cache for device %d\n", device_id);
        return result;
    }

    // Step 1: Deduplicate expert IDs with bounds checking
    std::unordered_set<int32_t> unique_experts;
    const int                   total_ids = n_expert_used * n_tokens;

    for (int i = 0; i < total_ids; i++) {
        const int32_t expert_id = expert_ids[i];
        if (expert_id >= 0 && expert_id < n_experts_total) {
            unique_experts.insert(expert_id);
        }
    }

    result.n_unique = static_cast<int>(unique_experts.size());

    GGML_SYCL_DEBUG("[PRESTAGE] Layer %d: %d unique experts from %d IDs (n_experts_total=%d)\n", layer_id,
                    result.n_unique, total_ids, n_experts_total);

    if (result.n_unique == 0) {
        result.success = true;  // Nothing to do, but not an error
        return result;
    }

    // Runtime prestage is lookup-only. Every routed expert must already be present either
    // in the device cache or in the host expert registry established at model load.
    static constexpr ggml_layout_mode pin_layouts[] = {
        GGML_LAYOUT_AOS,
        GGML_LAYOUT_SOA,
        GGML_LAYOUT_COALESCED,
        GGML_LAYOUT_XMX_TILED,
    };
    for (int32_t expert_id : unique_experts) {
        ggml_sycl_cache_id key =
            make_expert_cache_id(tensor_name, cache_uuid, model_id, expert_id, tensor_type, ne0, ne1);

        bool device_resolved = false;
        for (ggml_layout_mode layout : pin_layouts) {
            if (cache->lookup(key, layout)) {
                cache->pin(key, layout);
                device_resolved = true;
                break;
            }
        }

        if (device_resolved) {
            result.n_gpu++;
            continue;
        }

        if (const weight_entry * host_entry = cache->lookup_expert(key);
            host_entry && host_entry->ptr && host_entry->location != cache_location::DEVICE) {
            result.n_cpu++;
            continue;
        }

        result.n_miss++;
        result.expert_locations[expert_id] = cache_location::UNKNOWN;
        GGML_LOG_WARN("[PRESTAGE] Layer %d expert %d unresolved during lookup-only prestage\n", layer_id, expert_id);
    }

    GGML_UNUSED(queue_ptr);
    GGML_UNUSED(weight_base_ptr);
    GGML_UNUSED(expert_stride);
    GGML_UNUSED(expert_size);

    result.success = (result.n_miss == 0);

    GGML_SYCL_DEBUG("[PRESTAGE] Layer %d: Completed - n_gpu=%d, n_cpu=%d, n_miss=%d, unique=%d\n", layer_id,
                    result.n_gpu, result.n_cpu, result.n_miss, result.n_unique);

    return result;
}

void unpin_routed_experts(const int32_t * expert_ids,
                          int             n_expert_used,
                          int             n_tokens,
                          const void *    weight_base_ptr,
                          size_t          expert_stride,
                          int             layer_id,
                          int             n_experts_total,
                          int             device_id,
                          const char *    tensor_name,
                          uint64_t        cache_uuid,
                          uint32_t        model_id,
                          ggml_type       tensor_type,
                          int64_t         ne0,
                          int64_t         ne1) {
    // Validate inputs
    if (!expert_ids || n_expert_used <= 0 || n_tokens <= 0 || !weight_base_ptr) {
        return;
    }

    // Get unified cache for this device
    unified_cache * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        return;
    }

    // Deduplicate expert IDs (same as prestage)
    std::unordered_set<int32_t> unique_experts;
    const int                   total_ids = n_expert_used * n_tokens;

    for (int i = 0; i < total_ids; i++) {
        const int32_t expert_id = expert_ids[i];
        if (expert_id >= 0 && expert_id < n_experts_total) {
            unique_experts.insert(expert_id);
        }
    }

    // Unpin all unique experts
    for (int32_t expert_id : unique_experts) {
        ggml_sycl_cache_id key =
            make_expert_cache_id(tensor_name, cache_uuid, model_id, expert_id, tensor_type, ne0, ne1);

        cache->unpin(key, GGML_LAYOUT_AOS);
    }

    GGML_SYCL_DEBUG("[UNPIN] Layer %d: Unpinned %zu experts\n", layer_id, unique_experts.size());
}

// (ExpertPlacementTable removed — the cache IS the placement.
//  Popularity tracking now in g_expert_popularity via
//  get_expert_popularity_rank() / set_expert_popularity_rank().)

// === OneDNN FP16 Scratch Buffer Implementation ===

bool unified_cache::reserve_onednn_scratch(size_t weights_size, size_t activations_size) {
    const bool profile_active = arena_pp_profile_active();
    const auto t0             = profile_active ? arena_profile_clock::now() : arena_profile_clock::time_point{};
    bool       reused         = false;
    bool       arena_attempt  = false;
    bool       arena_success  = false;
    bool       direct_attempt = false;
    auto       finish         = [&](bool ok) {
        if (profile_active) {
            arena_pp_profile_note_onednn_reserve(weights_size, activations_size, reused, arena_attempt, arena_success,
                                                               direct_attempt, ok, arena_profile_elapsed_us(t0));
        }
        return ok;
    };

    std::lock_guard<std::mutex> lock(onednn_scratch_mutex_);

    // Already reserved with sufficient size?
    if (onednn_weights_scratch_ && onednn_activations_scratch_ && onednn_weights_scratch_size_ >= weights_size &&
        onednn_activations_scratch_size_ >= activations_size) {
        reused = true;
        return finish(true);
    }

    // VRAM arena path: sub-allocate from the oneDNN zone.
    if (arena_active()) {
        arena_attempt             = true;
        const size_t total_needed = weights_size + activations_size;
        size_t       zone_cap     = zone_capacity(vram_zone_id::ONEDNN);
        if (total_needed <= zone_cap) {
            // Reset the oneDNN zone to reclaim any previous allocation.
            zone_reset(vram_zone_id::ONEDNN);

            void * w = zone_alloc(vram_zone_id::ONEDNN, weights_size);
            void * a = w ? zone_alloc(vram_zone_id::ONEDNN, activations_size) : nullptr;
            if (w && a) {
                onednn_weights_scratch_          = w;
                onednn_weights_scratch_size_     = weights_size;
                onednn_activations_scratch_      = a;
                onednn_activations_scratch_size_ = activations_size;
                arena_success                    = true;
                // Budget already charged when arena was reserved.
                GGML_LOG_INFO("[UNIFIED-CACHE] oneDNN scratch from arena: weights=%.1f MB, activations=%.1f MB\n",
                              weights_size / (1024.0f * 1024.0f), activations_size / (1024.0f * 1024.0f));
                return finish(true);
            }
            // Reset zone on partial failure.
            zone_reset(vram_zone_id::ONEDNN);
        }
        GGML_LOG_WARN(
            "[UNIFIED-CACHE] oneDNN scratch arena zone too small (need %.1f MB, have %.1f MB), "
            "falling back to direct alloc\n",
            total_needed / (1024.0f * 1024.0f), zone_cap / (1024.0f * 1024.0f));
    }
    direct_attempt = true;

    // Free existing if resizing — subtract old sizes from budget first
    const size_t old_total = onednn_weights_scratch_size_ + onednn_activations_scratch_size_;
    if (onednn_weights_scratch_ && !vram_owns(onednn_weights_scratch_)) {
        try {
            sycl::free(onednn_weights_scratch_, queue_);
        } catch (...) {
        }
        onednn_weights_scratch_      = nullptr;
        onednn_weights_scratch_size_ = 0;
    } else {
        onednn_weights_scratch_      = nullptr;
        onednn_weights_scratch_size_ = 0;
    }
    if (onednn_activations_scratch_ && !vram_owns(onednn_activations_scratch_)) {
        try {
            sycl::free(onednn_activations_scratch_, queue_);
        } catch (...) {
        }
        onednn_activations_scratch_      = nullptr;
        onednn_activations_scratch_size_ = 0;
    } else {
        onednn_activations_scratch_      = nullptr;
        onednn_activations_scratch_size_ = 0;
    }
    if (old_total > 0 && !arena_active()) {
        saturating_sub_used(old_total);
    }

    // Note: we do NOT check cache budget here.  oneDNN scratch is a temporary
    // compute buffer (not cached weights), so it should not be gated by the
    // weight-cache available() budget.  When all weights are device-resident
    // (must_device=true), available() is near-zero but the device still has
    // physical VRAM for scratch.  When arena is active, scratch comes from the
    // pre-reserved ONEDNN zone.  Otherwise, sycl::malloc_device is used and
    // failure is handled in the catch blocks below.
    const size_t total_needed = weights_size + activations_size;
    GGML_SYCL_DEBUG("[UNIFIED-CACHE] oneDNN scratch: need %.1f MB, cache-available %.1f MB (bypassing budget check)\n",
                    total_needed / (1024.0f * 1024.0f), available() / (1024.0f * 1024.0f));

    // Allocate weights scratch
    if (arena_active()) {
        onednn_weights_scratch_ = zone_alloc(vram_zone_id::ONEDNN, weights_size);
        if (!onednn_weights_scratch_) {
            GGML_LOG_WARN("[UNIFIED-CACHE] Arena ONEDNN zone full for weights scratch (%.1f MB)\n",
                          weights_size / (1024.0f * 1024.0f));
            return finish(false);
        }
        onednn_weights_scratch_size_ = weights_size;
    } else {
        try {
            onednn_weights_scratch_ = sycl_aligned_malloc_device(weights_size, queue_);
            if (!onednn_weights_scratch_) {
                GGML_SYCL_DEBUG("[UNIFIED-CACHE] Failed to allocate oneDNN weights scratch (%.1f MB)\n",
                                weights_size / (1024.0f * 1024.0f));
                return finish(false);
            }
            alloc_registry::instance().register_alloc(onednn_weights_scratch_, weights_size,
                                                      ggml_sycl_get_device_id_from_queue(queue_), alloc_type::DEVICE);
            onednn_weights_scratch_size_ = weights_size;
        } catch (const sycl::exception & e) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] oneDNN weights scratch allocation failed: %s\n", e.what());
            return finish(false);
        }
    }

    // Allocate activations scratch
    if (arena_active()) {
        onednn_activations_scratch_ = zone_alloc(vram_zone_id::ONEDNN, activations_size);
        if (!onednn_activations_scratch_) {
            GGML_LOG_WARN("[UNIFIED-CACHE] Arena ONEDNN zone full for activations scratch (%.1f MB)\n",
                          activations_size / (1024.0f * 1024.0f));
            // Cleanup weights (arena-owned, just null the pointer — no sycl::free)
            onednn_weights_scratch_      = nullptr;
            onednn_weights_scratch_size_ = 0;
            return finish(false);
        }
        onednn_activations_scratch_size_ = activations_size;
    } else {
        try {
            onednn_activations_scratch_ = sycl_aligned_malloc_device(activations_size, queue_);
            if (!onednn_activations_scratch_) {
                GGML_SYCL_DEBUG("[UNIFIED-CACHE] Failed to allocate oneDNN activations scratch (%.1f MB)\n",
                                activations_size / (1024.0f * 1024.0f));
                // Cleanup weights
                alloc_registry::instance().unregister_alloc(onednn_weights_scratch_);
                sycl::free(onednn_weights_scratch_, queue_);
                onednn_weights_scratch_      = nullptr;
                onednn_weights_scratch_size_ = 0;
                return finish(false);
            }
            alloc_registry::instance().register_alloc(onednn_activations_scratch_, activations_size,
                                                      ggml_sycl_get_device_id_from_queue(queue_), alloc_type::DEVICE);
            onednn_activations_scratch_size_ = activations_size;
        } catch (const sycl::exception & e) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] oneDNN activations scratch allocation failed: %s\n", e.what());
            sycl::free(onednn_weights_scratch_, queue_);
            onednn_weights_scratch_      = nullptr;
            onednn_weights_scratch_size_ = 0;
            return finish(false);
        }
    }

    // Track in budget (skip when arena-backed — budget already charged at arena reservation)
    if (!arena_active()) {
        used_.fetch_add(total_needed, std::memory_order_relaxed);
    }

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] Reserved oneDNN scratch: weights=%.1f MB, activations=%.1f MB\n",
                    weights_size / (1024.0f * 1024.0f), activations_size / (1024.0f * 1024.0f));
    return finish(true);
}

bool unified_cache::get_onednn_scratch(size_t weights_needed, size_t activations_needed, onednn_scratch_buffers & out) {
    // Note: caller must hold onednn_scratch_mutex_ via lock_onednn_scratch()
    if (!onednn_weights_scratch_ || !onednn_activations_scratch_) {
        return false;
    }
    if (weights_needed > onednn_weights_scratch_size_ || activations_needed > onednn_activations_scratch_size_) {
        return false;
    }
    out.weights          = onednn_weights_scratch_;
    out.activations      = onednn_activations_scratch_;
    out.weights_size     = onednn_weights_scratch_size_;
    out.activations_size = onednn_activations_scratch_size_;
    return true;
}

bool unified_cache::reserve_reorder_temp(size_t size_bytes) {
    // Already reserved with sufficient size?
    if (reorder_temp_buffer_ && reorder_temp_size_ >= size_bytes) {
        return true;
    }

    // Free existing if resizing
    if (reorder_temp_buffer_) {
        if (!vram_owns(reorder_temp_buffer_)) {
            alloc_registry::instance().unregister_alloc(reorder_temp_buffer_);
            try {
                sycl::free(reorder_temp_buffer_, queue_);
            } catch (...) {
            }
            saturating_sub_used(reorder_temp_size_);
        }
        reorder_temp_buffer_ = nullptr;
        reorder_temp_size_   = 0;
    }

    // Allocate temp buffer for GPU-side AOS→SOA reorder.
    // Called from moe_hybrid_init_once under std::call_once — single-threaded.
    if (arena_active()) {
        reorder_temp_buffer_ = zone_alloc(vram_zone_id::SCRATCH, size_bytes);
        if (!reorder_temp_buffer_) {
            GGML_LOG_WARN("[UNIFIED-CACHE] Arena SCRATCH zone full for reorder temp (%.1f MB)\n",
                          size_bytes / (1024.0f * 1024.0f));
            return false;
        }
        reorder_temp_size_ = size_bytes;
        GGML_LOG_INFO("[UNIFIED-CACHE] Reserved GPU reorder temp buffer (arena): %.1f MB\n",
                      size_bytes / (1024.0f * 1024.0f));
        return true;
    }
    // Arena is not active — cannot allocate reorder temp buffer.
    // This is acceptable at early model-load before arena is configured;
    // the reorder path will be disabled for this invocation.
    GGML_LOG_WARN(
        "[UNIFIED-CACHE] reserve_reorder_temp: arena inactive, cannot allocate "
        "%.1f MB reorder temp buffer — reorder path disabled\n",
        size_bytes / (1024.0f * 1024.0f));
    return false;
}

// Global scratch buffer state for lock management
static std::mutex                                            g_onednn_scratch_lock_mutex;
static std::unordered_map<int, std::unique_lock<std::mutex>> g_onednn_scratch_locks;

bool unified_cache_reserve_onednn_scratch(int device_id, size_t weights_size, size_t activations_size) {
    unified_cache * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        return false;
    }
    return cache->reserve_onednn_scratch(weights_size, activations_size);
}

onednn_scratch_result unified_cache_get_onednn_scratch(int    device_id,
                                                       size_t weights_needed,
                                                       size_t activations_needed) {
    const bool            profile_active = arena_pp_profile_active();
    const auto            t0 = profile_active ? arena_profile_clock::now() : arena_profile_clock::time_point{};
    onednn_scratch_result result;
    unified_cache *       cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        if (profile_active) {
            arena_pp_profile_note_onednn_get(weights_needed, activations_needed, false, arena_profile_elapsed_us(t0));
        }
        return result;
    }

    // Acquire lock and store it for later release
    auto lock = cache->lock_onednn_scratch();

    unified_cache::onednn_scratch_buffers buffers;
    if (!cache->get_onednn_scratch(weights_needed, activations_needed, buffers)) {
        if (profile_active) {
            arena_pp_profile_note_onednn_get(weights_needed, activations_needed, false, arena_profile_elapsed_us(t0));
        }
        return result;
    }

    // Store lock for release
    {
        std::lock_guard<std::mutex> guard(g_onednn_scratch_lock_mutex);
        g_onednn_scratch_locks[device_id] = std::move(lock);
    }

    result.weights     = buffers.weights;
    result.activations = buffers.activations;
    result.ok          = true;
    if (profile_active) {
        arena_pp_profile_note_onednn_get(weights_needed, activations_needed, true, arena_profile_elapsed_us(t0));
    }
    return result;
}

void unified_cache_release_onednn_scratch(int device_id) {
    std::lock_guard<std::mutex> guard(g_onednn_scratch_lock_mutex);
    auto                        it = g_onednn_scratch_locks.find(device_id);
    if (it != g_onednn_scratch_locks.end()) {
        // Unlock by destroying the unique_lock
        g_onednn_scratch_locks.erase(it);
    }
}

bool unified_cache_has_onednn_scratch(int device_id) {
    unified_cache * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        return false;
    }
    return cache->has_onednn_scratch();
}

// === Persistent Scratch Buffer Implementation ===

bool unified_cache::reserve_persistent_scratch(const std::string & buffer_name, size_t size_bytes, bool pin) {
    std::lock_guard<std::mutex> lock(persistent_scratch_mutex_);

    // Check if we already have this buffer with sufficient size
    auto it = persistent_scratches_.find(buffer_name);
    if (it != persistent_scratches_.end()) {
        auto & entry = it->second;
        if (entry.size >= size_bytes) {
            // Already have sufficient size, just update pin state if needed
            entry.pinned = pin;
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] Persistent scratch '%s' already reserved (%.1f MB >= %.1f MB)\n",
                            buffer_name.c_str(), entry.size / (1024.0f * 1024.0f), size_bytes / (1024.0f * 1024.0f));
            return true;
        }
        // Existing buffer too small, need to free and reallocate
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Persistent scratch '%s' resize: %.1f MB -> %.1f MB\n", buffer_name.c_str(),
                        entry.size / (1024.0f * 1024.0f), size_bytes / (1024.0f * 1024.0f));
        if (entry.device_ptr) {
            if (!vram_owns(entry.device_ptr)) {
                try {
                    sycl::free(entry.device_ptr, queue_);
                } catch (...) {
                }
                saturating_sub_used(entry.size);
            }
        }
        persistent_scratches_.erase(it);
    }

    // Allocate device memory
    void * ptr = nullptr;
    if (arena_active()) {
        ptr = zone_alloc(vram_zone_id::SCRATCH, size_bytes);
        if (!ptr) {
            GGML_LOG_WARN("[UNIFIED-CACHE] Arena SCRATCH zone full for persistent scratch '%s' (%.1f MB)\n",
                          buffer_name.c_str(), size_bytes / (1024.0f * 1024.0f));
            return false;
        }
    } else {
        // Check if we have budget
        if (size_bytes > available()) {
            // Try to evict to make room
            size_t freed = evict(size_bytes - available());
            if (freed < size_bytes - available()) {
                GGML_LOG_ERROR(
                    "[UNIFIED-CACHE] Cannot reserve persistent scratch '%s': need %.1f MB, available %.1f MB\n",
                    buffer_name.c_str(), size_bytes / (1024.0f * 1024.0f), available() / (1024.0f * 1024.0f));
                return false;
            }
        }
        try {
            ptr = sycl_aligned_malloc_device(size_bytes, queue_);
            if (!ptr) {
                GGML_LOG_ERROR("[UNIFIED-CACHE] Failed to allocate persistent scratch '%s' (%.1f MB)\n",
                               buffer_name.c_str(), size_bytes / (1024.0f * 1024.0f));
                return false;
            }
            alloc_registry::instance().register_alloc(ptr, size_bytes, ggml_sycl_get_device_id_from_queue(queue_),
                                                      alloc_type::DEVICE);
        } catch (const sycl::exception & e) {
            GGML_LOG_ERROR("[UNIFIED-CACHE] Persistent scratch '%s' allocation failed: %s\n", buffer_name.c_str(),
                           e.what());
            return false;
        }
        // Track in budget (skip when arena-backed)
        used_.fetch_add(size_bytes, std::memory_order_relaxed);
    }

    persistent_scratches_[buffer_name] = { ptr, size_bytes, pin };

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] Reserved persistent scratch '%s': %.1f MB (pinned=%d)\n", buffer_name.c_str(),
                    size_bytes / (1024.0f * 1024.0f), pin ? 1 : 0);
    return true;
}

void * unified_cache::get_persistent_scratch(const std::string & buffer_name) {
    std::lock_guard<std::mutex> lock(persistent_scratch_mutex_);

    auto it = persistent_scratches_.find(buffer_name);
    if (it == persistent_scratches_.end()) {
        return nullptr;
    }
    return it->second.device_ptr;
}

void unified_cache::release_persistent_scratch(const std::string & buffer_name) {
    std::lock_guard<std::mutex> lock(persistent_scratch_mutex_);

    auto it = persistent_scratches_.find(buffer_name);
    if (it == persistent_scratches_.end()) {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Persistent scratch '%s' not found for release\n", buffer_name.c_str());
        return;
    }

    auto & entry = it->second;
    if (entry.device_ptr) {
        if (!vram_owns(entry.device_ptr)) {
            try {
                sycl::free(entry.device_ptr, queue_);
            } catch (...) {
            }
            saturating_sub_used(entry.size);
        }
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Released persistent scratch '%s' (%.1f MB)\n", buffer_name.c_str(),
                        entry.size / (1024.0f * 1024.0f));
    }
    persistent_scratches_.erase(it);
}

bool unified_cache::has_persistent_scratch(const std::string & buffer_name) const {
    std::lock_guard<std::mutex> lock(persistent_scratch_mutex_);
    return persistent_scratches_.find(buffer_name) != persistent_scratches_.end();
}

size_t unified_cache::get_persistent_scratch_size(const std::string & buffer_name) const {
    std::lock_guard<std::mutex> lock(persistent_scratch_mutex_);
    auto                        it = persistent_scratches_.find(buffer_name);
    if (it == persistent_scratches_.end()) {
        return 0;
    }
    return it->second.size;
}

// === Persistent Scratch Buffer C API Wrappers ===

bool unified_cache_reserve_persistent_scratch(int device_id, const char * buffer_name, size_t size_bytes, bool pin) {
    unified_cache * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        return false;
    }
    return cache->reserve_persistent_scratch(buffer_name, size_bytes, pin);
}

void * unified_cache_get_persistent_scratch(int device_id, const char * buffer_name) {
    unified_cache * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        return nullptr;
    }
    return cache->get_persistent_scratch(buffer_name);
}

void unified_cache_release_persistent_scratch(int device_id, const char * buffer_name) {
    unified_cache * cache = get_unified_cache_for_device(device_id);
    if (cache) {
        cache->release_persistent_scratch(buffer_name);
    }
}

bool unified_cache_has_persistent_scratch(int device_id, const char * buffer_name) {
    unified_cache * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        return false;
    }
    return cache->has_persistent_scratch(buffer_name);
}

size_t unified_cache_get_persistent_scratch_size(int device_id, const char * buffer_name) {
    unified_cache * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        return 0;
    }
    return cache->get_persistent_scratch_size(buffer_name);
}

// === Bulk Weight Pinning C API Wrappers ===

int unified_cache_pin_layer_weights(int device_id, int layer_id, const layer_weight_set * weights, int layout) {
    unified_cache * cache = get_unified_cache_for_device(device_id);
    if (!cache || !weights) {
        return 0;
    }
    return cache->pin_layer_weights(layer_id, *weights, static_cast<ggml_layout_mode>(layout));
}

void unified_cache_unpin_layer_weights(int device_id, int layer_id, const layer_weight_set * weights, int layout) {
    unified_cache * cache = get_unified_cache_for_device(device_id);
    if (!cache || !weights) {
        return;
    }
    cache->unpin_layer_weights(layer_id, *weights, static_cast<ggml_layout_mode>(layout));
}

int unified_cache_pin_model_weights(int device_id, int n_layers, const layer_weight_set * layers, int layout) {
    unified_cache * cache = get_unified_cache_for_device(device_id);
    if (!cache || !layers || n_layers <= 0) {
        return 0;
    }
    // Convert C array to vector
    std::vector<layer_weight_set> layers_vec(layers, layers + n_layers);
    return cache->pin_model_weights(n_layers, layers_vec, static_cast<ggml_layout_mode>(layout));
}

void unified_cache_unpin_model_weights(int device_id, int n_layers, const layer_weight_set * layers, int layout) {
    unified_cache * cache = get_unified_cache_for_device(device_id);
    if (!cache || !layers || n_layers <= 0) {
        return;
    }
    // Unpin each layer
    for (int i = 0; i < n_layers; i++) {
        cache->unpin_layer_weights(i, layers[i], static_cast<ggml_layout_mode>(layout));
    }
}

// =============================================================================
// Multi-Device Partial Row Loading
// =============================================================================

void * unified_cache::load_partial_rows(const char * tensor_name,
                                        const void * src_host,
                                        ggml_type    type,
                                        int64_t      ncols,
                                        int64_t      row_count,
                                        int          device_idx) {
    if (!tensor_name || !src_host || row_count <= 0 || ncols <= 0) {
        return nullptr;
    }

    // Build cache key: "tensor_name:device_idx"
    std::string key = std::string(tensor_name) + ":" + std::to_string(device_idx);

    // Check if already loaded
    {
        std::lock_guard<std::mutex> lock(partial_mutex_);
        auto                        it = partial_cache_.find(key);
        if (it != partial_cache_.end()) {
            return it->second.ptr;
        }
    }

    const size_t row_bytes     = ggml_row_size(type, ncols);
    const size_t partial_bytes = static_cast<size_t>(row_count) * row_bytes;

    if (partial_bytes == 0) {
        return nullptr;
    }

    // Allocate device memory on this cache's queue
    void * dev_ptr = nullptr;
    try {
        dev_ptr = ggml_sycl_malloc_device_raw(partial_bytes, queue_, "unified_cache:partial_rows");
    } catch (const sycl::exception & e) {
        GGML_LOG_ERROR("[PARTIAL-ROWS] malloc_device failed for '%s' device %d: %s\n", tensor_name, device_idx,
                       e.what());
        return nullptr;
    }
    if (!dev_ptr) {
        GGML_LOG_ERROR("[PARTIAL-ROWS] malloc_device returned nullptr for '%s' device %d (%.2f MB)\n", tensor_name,
                       device_idx, partial_bytes / (1024.0f * 1024.0f));
        return nullptr;
    }

    // Copy AOS data from host to device — no CPU wait needed since the reorder
    // kernel below is submitted to the same in-order queue_ and will implicitly
    // depend on this memcpy completing on the GPU.
    queue_.memcpy(dev_ptr, src_host, partial_bytes);

    // Apply in-place SOA reorder on device (same queue, implicitly ordered)
    bool reordered =
        reorder_rows_to_soa(static_cast<uint8_t *>(dev_ptr), type, ncols, row_count, partial_bytes, &queue_);
    if (!reordered) {
        GGML_LOG_ERROR("[PARTIAL-ROWS] SOA reorder failed for '%s' device %d type %d\n", tensor_name, device_idx,
                       (int) type);
        sycl::free(dev_ptr, queue_);
        return nullptr;
    }

    // Track in partial cache
    {
        std::lock_guard<std::mutex> lock(partial_mutex_);
        partial_cache_[key] = { dev_ptr, device_idx, partial_bytes };
    }

    // Update budget tracking (count as weight bytes on this device)
    used_.fetch_add(partial_bytes, std::memory_order_relaxed);

    GGML_SYCL_DEBUG("[PARTIAL-ROWS] Loaded '%s' device %d: %lld rows, %.2f MB SOA\n", tensor_name, device_idx,
                    (long long) row_count, partial_bytes / (1024.0f * 1024.0f));

    return dev_ptr;
}

void * unified_cache::get_split_weight_ptr(const char * tensor_name, int device_idx) {
    if (!tensor_name) {
        return nullptr;
    }
    std::string key = std::string(tensor_name) + ":" + std::to_string(device_idx);

    std::lock_guard<std::mutex> lock(partial_mutex_);
    auto                        it = partial_cache_.find(key);
    return (it != partial_cache_.end()) ? it->second.ptr : nullptr;
}

void unified_cache::free_partial_entries() {
    std::lock_guard<std::mutex> lock(partial_mutex_);
    for (auto & pair : partial_cache_) {
        if (pair.second.ptr && !g_sycl_shutting_down.load()) {
            sycl::free(pair.second.ptr, queue_);
            saturating_sub_used(pair.second.bytes);
        }
    }
    partial_cache_.clear();
}

// Free-standing wrappers for multi-device partial row API

void * unified_cache_load_partial_rows(const char * tensor_name,
                                       const void * src_host,
                                       ggml_type    type,
                                       int64_t      ncols,
                                       int64_t      row_count,
                                       int          target_device) {
    auto * cache = get_unified_cache_for_device(target_device);
    if (!cache) {
        GGML_LOG_ERROR("[PARTIAL-ROWS] No cache for device %d\n", target_device);
        return nullptr;
    }
    return cache->load_partial_rows(tensor_name, src_host, type, ncols, row_count, target_device);
}

void * unified_cache_get_split_weight_ptr(const char * tensor_name, int device) {
    auto * cache = get_unified_cache_for_device(device);
    if (!cache) {
        return nullptr;
    }
    return cache->get_split_weight_ptr(tensor_name, device);
}

void unified_cache_free_partial_entries(int device) {
    auto * cache = get_unified_cache_for_device(device);
    if (cache) {
        cache->free_partial_entries();
    }
}

unified_cache * unified_cache_register_for_queue(int device_id, sycl::queue & queue) {
    // Fast path: check under shared lock
    {
        std::shared_lock<std::shared_mutex> read_lock(g_cache_rw_mutex);
        auto                                it = g_device_caches.find(device_id);
        if (it != g_device_caches.end()) {
            return it->second.get();
        }
    }

    // Slow path: create under exclusive lock
    std::unique_lock<std::shared_mutex> lock(g_cache_rw_mutex);

    // Double-check after acquiring write lock
    auto it = g_device_caches.find(device_id);
    if (it != g_device_caches.end()) {
        return it->second.get();
    }

    // Query VRAM from the queue's device (no dpct dependency)
    sycl::device dev    = queue.get_device();
    size_t       total  = dev.get_info<sycl::info::device::global_mem_size>();
    size_t       budget = static_cast<size_t>(total * 0.80);  // 80% budget for secondary GPU

    const size_t min_headroom = 256ull * 1024ull * 1024ull;
    if (total > min_headroom && budget > total - min_headroom) {
        budget = total - min_headroom;
    }

    GGML_LOG_INFO("[UNIFIED-CACHE] Registering device %d (%s): total=%.1f MB budget=%.1f MB\n", device_id,
                  dev.get_info<sycl::info::device::name>().c_str(), total / (1024.0f * 1024.0f),
                  budget / (1024.0f * 1024.0f));

    const size_t staging_bytes = 16 * 1024 * 1024;  // 16 MB staging for secondary device
    try {
        g_device_caches[device_id] = std::make_unique<unified_cache>(queue, budget, staging_bytes, 0, total);
        return g_device_caches[device_id].get();
    } catch (const sycl::exception & e) {
        GGML_LOG_ERROR("[UNIFIED-CACHE] Failed to register device %d: %s\n", device_id, e.what());
        return nullptr;
    }
}

// === Direct Staging API — Free-Standing Wrappers ===

direct_stage_result unified_cache_direct_stage_weight(int                  device_id,
                                                      ggml_sycl_cache_id   key,
                                                      const void *         src,
                                                      size_t               src_size,
                                                      size_t               dst_size,
                                                      ggml_layout_mode     layout,
                                                      cache_layout_fill_fn fill_fn,
                                                      const void *         fill_ctx,
                                                      sycl::queue *        queue) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        GGML_LOG_ERROR("[DIRECT-STAGE] no cache for device %d\n", device_id);
        return {};
    }
    return cache->direct_stage_weight(key, src, src_size, dst_size, layout, fill_fn, fill_ctx, queue);
}

direct_stage_result unified_cache_direct_stage_expert(int                  device_id,
                                                      ggml_sycl_cache_id   key,
                                                      const void *         src,
                                                      size_t               src_size,
                                                      size_t               dst_size,
                                                      ggml_layout_mode     layout,
                                                      cache_layout_fill_fn fill_fn,
                                                      const void *         fill_ctx,
                                                      sycl::queue *        queue) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        GGML_LOG_ERROR("[DIRECT-STAGE] no cache for device %d\n", device_id);
        return {};
    }
    return cache->direct_stage_expert(key, src, src_size, dst_size, layout, fill_fn, fill_ctx, queue);
}

const weight_entry * unified_cache_lookup_weight(int device_id, ggml_sycl_cache_id key) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        return nullptr;
    }
    return cache->lookup_weight(key);
}

const weight_entry * unified_cache_lookup_expert(int device_id, ggml_sycl_cache_id key) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        return nullptr;
    }
    return cache->lookup_expert(key);
}

// === Raw allocation primitives — only place in the codebase that calls sycl::malloc_* ===
// All consumer code must route through these functions (or higher-level cache APIs).
// unified-cache.cpp and pinned-pool.cpp are the sole owners of sycl::malloc_* calls.

void * unified_cache_raw_malloc_device(size_t size, const sycl::queue & queue) {
    void * ptr = nullptr;
    try {
        ptr = sycl_aligned_malloc_device(size, queue);
    } catch (...) {
        return nullptr;
    }
    return ptr;
}

void * unified_cache_raw_malloc_host(size_t size, const sycl::queue & queue) {
    void * ptr = nullptr;
    try {
        ptr = sycl::malloc_host(size, queue);
    } catch (...) {
        return nullptr;
    }
    return ptr;
}

void * unified_cache_raw_malloc_host(size_t size, const sycl::context & ctx) {
    void * ptr = nullptr;
    try {
        ptr = sycl::malloc_host(size, ctx);
    } catch (...) {
        return nullptr;
    }
    return ptr;
}

void * unified_cache_raw_malloc_shared(size_t size, const sycl::queue & queue) {
    void * ptr = nullptr;
    try {
        ptr = sycl::malloc_shared(size, queue);
    } catch (...) {
        return nullptr;
    }
    return ptr;
}

void shutdown_unified_cache() {
    // Set shutdown flag FIRST so destructors skip sycl::free() calls
    g_sycl_shutting_down.store(true);

    // Clear all device caches
    // The destructors will skip cleanup due to the shutdown flag
    std::unique_lock<std::shared_mutex> lock(g_cache_rw_mutex);
    g_device_caches.clear();

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] Shutdown complete\n");
}

// ============================================================================
// Phase 3: Unified Allocation API
// ============================================================================

// --- unified_cache::available_device() ---
// Queries Level Zero for the actual free VRAM on the device, subtracts a safety
// margin for driver internals.  Unlike available() which is pure budget math,
// this reflects real hardware state.

size_t unified_cache::available_device() const {
    const int device_id = get_device_id_from_queue(const_cast<sycl::queue &>(queue_));
    if (device_id < 0 || device_id >= GGML_SYCL_MAX_DEVICES) {
        return 0;
    }
    size_t free_mem = 0, total_mem = 0;
    ggml_backend_sycl_get_device_memory(device_id, &free_mem, &total_mem);
    if (free_mem <= VRAM_SAFETY_MARGIN) {
        return 0;
    }
    return free_mem - VRAM_SAFETY_MARGIN;
}

// --- Deprecated shim: ensure_cached_alloc ---
// Restored verbatim from pre-kcru9 deletion to keep test suite building.
// New code must use unified_alloc() instead.
// TODO: Port test suite (tests/test-sycl-unified-cache*, ~40 call sites across 5 files) to
// unified_alloc(), then delete this shim. Tracked in bd: llama.cpp-og9dt.

void * unified_cache::ensure_cached_alloc(const ggml_sycl_cache_id & key_id,
                                          const void *               src_ptr,
                                          size_t                     src_size,
                                          size_t                     alloc_size,
                                          cache_entry_type           type,
                                          int                        layer_id,
                                          int                        expert_id,
                                          ggml_layout_mode           layout,
                                          bool                       validate_content,
                                          bool *                     needs_fill) {
    if (needs_fill) {
        *needs_fill = true;
    }
    if (!key_id.valid || !src_ptr || src_size == 0 || alloc_size == 0) {
        return nullptr;
    }

    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    if (!g_graph_compute_active.load(std::memory_order_acquire)) {
        process_deferred_frees();
    }

    unified_cache_key key{ type, key_id, layer_id, expert_id };
    const uint64_t    new_hash = compute_content_hash(src_ptr, src_size);

    auto it = entries_.find(key);
    if (it != entries_.end()) {
        auto id_it = id_to_key_.find(key_id);
        if (id_it == id_to_key_.end()) {
            id_to_key_.emplace(key_id, key);
        } else if (!(id_it->second == key)) {
            GGML_LOG_ERROR("[UNIFIED-CACHE] identity collision in ensure_cached_alloc model=%llu name_hash=0x%llx\n",
                           (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash);
            if (cache_assert_enabled()) {
                GGML_ABORT("unified_cache id_to_key mismatch");
            }
        }
        if (it->second.layout != layout) {
            // llama.cpp-vtf7f: if any mem_handle leases this entry, its
            // caller (mmvq/mmq/dnnl/etc.) is mid-use.  Returning the
            // existing pointer under the old layout is safe (caller already
            // committed to it); erasing and switching layouts would dangle
            // that pointer.  Fall through to "same layout" handling below.
            const uint32_t ensure_leases = it->second.in_use_count.load();
            if (ensure_leases > 0) {
                GGML_LOG_WARN(
                    "[UNIFIED-CACHE] ensure_cached_alloc: layout switch refused "
                    "model=%llu name_hash=0x%llx in_use=%u have=%d want=%d (reusing old layout)\n",
                    (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash, ensure_leases,
                    (int) it->second.layout, (int) layout);
                // Fall through: use existing entry as-is under its current layout.
            } else {
                if (it->second.pinned) {
                    GGML_SYCL_DEBUG(
                        "[UNIFIED-CACHE] layout switch: unpinning model=%llu name_hash=0x%llx have=%d want=%d\n",
                        (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash,
                        (int) it->second.layout, (int) layout);
                    it->second.pinned = false;
                }
                void * stale_ptr      = it->second.device_ptr;
                size_t stale_size     = it->second.size;
                bool   stale_host_res = it->second.host_resident;
                if (type == cache_entry_type::MOE_EXPERT && layer_id == 0 && expert_id == 0) {
                    fprintf(stderr,
                            "[MOE-ENTRY-ERASE] path=ensure_cached_alloc-layout-switch layer=%d expert=%d "
                            "have_layout=%d want_layout=%d size=%zu pinned=%d state=%d\n",
                            layer_id, expert_id, (int) it->second.layout, (int) layout, it->second.size,
                            it->second.pinned ? 1 : 0, (int) it->second.state);
                }
                entries_.erase(it);
                it = entries_.end();
                if (!stale_host_res && stale_ptr && stale_size > 0) {
                    enqueue_deferred_free(stale_ptr, stale_size);
                }
            }
        }
        if (it == entries_.end()) {
            // Fall through to allocation path below
        } else {
            bool need_realloc = (alloc_size != it->second.size);
            bool content_changed =
                validate_content || (it->second.src_ptr != src_ptr) || (it->second.content_hash != new_hash);

            if (need_realloc) {
                const bool   was_pinned = it->second.pinned;
                const size_t old_size   = it->second.size;
                it->second.pinned       = true;
                while (used_.load() - old_size + alloc_size > budget_) {
                    if (evict_one(alloc_size) == 0) {
                        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Cannot evict for alloc (used=%.1f MB, need=%.1f MB)\n",
                                        used_.load() / (1024.0f * 1024.0f), alloc_size / (1024.0f * 1024.0f));
                        it->second.pinned = was_pinned;
                        if (needs_fill) {
                            *needs_fill = false;
                        }
                        return nullptr;
                    }
                }

                void * new_device_ptr = nullptr;
                try {
                    new_device_ptr = ggml_sycl_malloc_device_raw(alloc_size, queue_, "unified_cache:alloc");
                } catch (const sycl::exception & e) {
                    GGML_LOG_ERROR("[UNIFIED-CACHE] alloc malloc_device failed: %s\n", e.what());
                    it->second.pinned = was_pinned;
                    if (needs_fill) {
                        *needs_fill = false;
                    }
                    return nullptr;
                }
                if (!new_device_ptr) {
                    GGML_LOG_ERROR("[UNIFIED-CACHE] alloc malloc_device returned nullptr\n");
                    it->second.pinned = was_pinned;
                    if (needs_fill) {
                        *needs_fill = false;
                    }
                    return nullptr;
                }
                it->second.pinned = was_pinned;
                enqueue_deferred_free(it->second.device_ptr, it->second.size);
                it->second.device_ptr = new_device_ptr;
                it->second.size       = alloc_size;
                if (alloc_size > old_size) {
                    used_.fetch_add(alloc_size - old_size, std::memory_order_relaxed);
                } else if (old_size > alloc_size) {
                    saturating_sub_used(old_size - alloc_size);
                }
                content_changed = true;
            }

            it->second.src_ptr      = src_ptr;
            it->second.content_hash = new_hash;
            it->second.access_count++;
            it->second.last_access     = time_++;
            it->second.state           = cache_entry_state::READY;
            it->second.has_ready_event = false;

            if (needs_fill) {
                *needs_fill = need_realloc || content_changed;
            }
            return it->second.device_ptr;
        }
    }

    while (used_.load() + alloc_size > budget_) {
        if (evict_one(alloc_size) == 0) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] Cannot evict for alloc (used=%.1f MB, need=%.1f MB)\n",
                            used_.load() / (1024.0f * 1024.0f), alloc_size / (1024.0f * 1024.0f));
            return nullptr;
        }
    }

    void * device_ptr = nullptr;
    try {
        device_ptr = ggml_sycl_malloc_device_raw(alloc_size, queue_, "unified_cache:alloc");
    } catch (const sycl::exception & e) {
        GGML_LOG_ERROR("[UNIFIED-CACHE] alloc malloc_device failed: %s\n", e.what());
        return nullptr;
    }
    if (!device_ptr) {
        GGML_LOG_ERROR("[UNIFIED-CACHE] alloc malloc_device returned nullptr\n");
        return nullptr;
    }

    unified_cache_entry entry{};
    entry.device_ptr      = device_ptr;
    entry.src_ptr         = src_ptr;
    entry.content_hash    = new_hash;
    entry.size            = alloc_size;
    entry.type            = type;
    entry.layer_id        = layer_id;
    entry.expert_id       = expert_id;
    entry.layout          = layout;
    entry.access_count    = 1;
    entry.last_access     = time_++;
    entry.pinned          = false;
    entry.hot             = false;
    entry.state           = cache_entry_state::READY;
    entry.has_ready_event = false;
    entry.host_resident   = false;
    entry.location        = cache_location::DEVICE;

    entries_[key] = entry;
    auto id_it    = id_to_key_.find(key_id);
    if (id_it == id_to_key_.end()) {
        id_to_key_.emplace(key_id, key);
    } else if (!(id_it->second == key)) {
        GGML_LOG_ERROR("[UNIFIED-CACHE] identity collision on insert model=%llu name_hash=0x%llx\n",
                       (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash);
        if (cache_assert_enabled()) {
            GGML_ABORT("unified_cache id_to_key mismatch");
        }
    }
    used_.fetch_add(alloc_size, std::memory_order_relaxed);

    if (needs_fill) {
        *needs_fill = true;
    }
    return device_ptr;
}

// --- Host arena methods ---

void * unified_cache::host_pool_alloc(size_t size, size_t alignment) {
    if (!host_arena_ || !size) {
        return nullptr;
    }
    return host_arena_->allocate_runtime(size, alignment);
}

void unified_cache::host_pool_free(void * ptr, size_t size) {
    if (!host_arena_ || !ptr || size == 0) {
        return;
    }
    host_arena_->deallocate(ptr, size);
}

void unified_cache::host_zone_reset(host_zone_id zone) {
    if (!host_arena_ || !host_arena_->zones_configured()) {
        return;
    }
    GGML_ASSERT(zone != host_zone_id::WEIGHT && "WEIGHT host zone must not be reset");

    // Purge registry entries that belong to this host zone.  Without this,
    // zone_reset() recycles TLSF addresses while the registry still maps them,
    // causing duplicate-pointer rejection on retry.
    {
        std::lock_guard<std::mutex> lock(g_runtime_alloc_mutex);
        for (auto it = g_runtime_alloc_registry.begin(); it != g_runtime_alloc_registry.end();) {
            if (it->second.handle.host_zone == zone) {
                it = g_runtime_alloc_registry.erase(it);
            } else {
                ++it;
            }
        }
    }

    // Purge offload pool entries whose backing memory came from this host zone.
    // The offload pool caches (slot registry + free list) hold raw pointers into
    // the TLSF arena.  zone_reset() recycles that memory, so any surviving pool
    // entry becomes a dangling pointer.  Purge before zone_reset so that the
    // next acquire_offload_buffer() allocates fresh memory from the reset zone.
    {
        std::lock_guard<std::mutex> lock(g_offload_pool_mutex);
        for (auto it = g_offload_pool_slots.begin(); it != g_offload_pool_slots.end();) {
            if (it->second.handle.host_zone == zone) {
                it = g_offload_pool_slots.erase(it);
            } else {
                ++it;
            }
        }
        // Rebuild free list from surviving slots to remove stale pointers.
        g_offload_pool_free.clear();
        for (const auto & kv : g_offload_pool_slots) {
            if (!kv.second.in_use) {
                g_offload_pool_free[kv.second.key].push_back(kv.first);
            }
        }
    }

    host_arena_->zone_reset(zone);
}

void unified_cache::host_zone_free(host_zone_id zone, void * ptr) {
    if (!host_arena_ || !ptr) {
        return;
    }
    host_arena_->zone_free(zone, ptr);
}

size_t unified_cache::host_zone_used(host_zone_id zone) const {
    if (!host_arena_ || !host_arena_->zones_configured()) {
        return 0;
    }
    return host_arena_->zone_used(zone);
}

size_t unified_cache::host_zone_capacity(host_zone_id zone) const {
    if (!host_arena_ || !host_arena_->zones_configured()) {
        return 0;
    }
    return host_arena_->zone_capacity(zone);
}

size_t unified_cache::host_zone_largest_free_block(host_zone_id zone) const {
    if (!host_arena_ || !host_arena_->zones_configured()) {
        return 0;
    }
    return host_arena_->zone_largest_free_block(zone);
}

bool unified_cache::host_zone_grow(host_zone_id zone, size_t additional_bytes) {
    if (!host_arena_ || !host_arena_->zones_configured() || additional_bytes == 0) {
        return false;
    }
    return host_arena_->grow_zone(zone, additional_bytes);
}

void unified_cache::configure_host_zones(size_t weight_bytes,
                                         size_t kv_bytes,
                                         size_t staging_bytes,
                                         size_t scratch_bytes) {
    if (!host_arena_) {
        return;
    }
    host_arena_->configure_zones(weight_bytes, kv_bytes, staging_bytes, scratch_bytes);
}

bool unified_cache::host_zones_configured() const {
    return host_arena_ && host_arena_->zones_configured();
}

void unified_cache::grow_scratch_zone(size_t additional_bytes) {
    if (!host_arena_ || !host_arena_->zones_configured() || additional_bytes == 0) {
        return;
    }
    host_arena_->grow_zone(host_zone_id::SCRATCH, additional_bytes);
    GGML_LOG_INFO("[HOST-ARENA] SCRATCH zone grown by %.1f MB for compute buffers\n",
                  additional_bytes / (1024.0 * 1024.0));
}

bool unified_cache::contains_pinned(const void * ptr) const {
    if (!host_arena_ || !ptr) {
        return false;
    }
    return host_arena_->contains(ptr);
}

size_t unified_cache::pre_allocate_host_pool(size_t total_bytes) {
    if (!host_arena_) {
        return 0;
    }
    return host_arena_->pre_allocate(total_bytes);
}

size_t unified_cache::pre_allocate_all(size_t model_weight_bytes) {
    if (!host_arena_) {
        return 0;
    }
    return host_arena_->pre_allocate_all(model_weight_bytes);
}

size_t unified_cache::pre_allocate_runtime_chunks(size_t total_bytes) {
    if (!host_arena_) {
        return 0;
    }
    return host_arena_->pre_allocate_runtime_chunks(total_bytes);
}

segmented_buffer unified_cache::host_zone_alloc_segmented(host_zone_id zone, size_t size, size_t alignment) {
    if (!host_arena_) {
        return {};
    }
    if (!host_arena_->zones_configured()) {
        return host_arena_->allocate_segmented(size, alignment);
    }
    return host_arena_->zone_alloc_segmented(zone, size, alignment);
}

void * unified_cache::host_zone_alloc(host_zone_id zone, size_t size, size_t alignment) {
    segmented_buffer buf = host_zone_alloc_segmented(zone, size, alignment);
    if (buf.segments.empty()) {
        return nullptr;
    }
    if (buf.segments.size() > 1) {
        // Should never happen with TLSF (all zone allocations are contiguous within a chunk).
        // Free each segment and return nullptr as a safety guard.
        for (auto & seg : buf.segments) {
            host_arena_->zone_free(zone, seg.ptr);
        }
        return nullptr;
    }
    return buf.segments[0].ptr;
}

// --- unified_cache::allocate() ---
// The primary allocation path for Phase 3.  Steps:
//   1. Check budget: used_ + runtime_reserved + size <= budget_
//   2. If insufficient, try evict_and_flush
//   3. Query L0 free VRAM as a second safety guard
//   4. Allocate on device or fall back to host-pinned
//   5. Track in managed_allocs_ and adjust budget

unified_cache::vram_alloc_result unified_cache::allocate(size_t size, alloc_lifetime lifetime, const char * tag) {
    vram_alloc_result result{};
    if (size == 0) {
        return result;
    }

    const char * label = tag ? tag : "unified_cache::allocate";

    // Step 1: Check budget headroom (available() = budget_ - used_,
    // where budget_ = base_budget_ - reserved_, and reserved_ tracks
    // KV + compute + staging + graph runtime bytes).
    bool try_device = true;
    if (size > available()) {
        // Try eviction to make room.
        const size_t needed = size - available();
        const size_t freed  = evict_and_flush(needed);
        if (size > available()) {
            GGML_SYCL_DEBUG(
                "[UNIFIED-CACHE] allocate(%s): budget exhausted after evicting "
                "%.1f MB (need %.1f MB, avail %.1f MB)\n",
                label, freed / (1024.0 * 1024.0), size / (1024.0 * 1024.0), available() / (1024.0 * 1024.0));
            try_device = false;
        }
    }

    // Step 2: Cross-check with L0 free VRAM.
    if (try_device) {
        const size_t hw_avail = available_device();
        if (size > hw_avail) {
            GGML_SYCL_DEBUG(
                "[UNIFIED-CACHE] allocate(%s): L0 free VRAM too low "
                "(need %.1f MB, L0 avail %.1f MB), falling back\n",
                label, size / (1024.0 * 1024.0), hw_avail / (1024.0 * 1024.0));
            try_device = false;
        }
    }

    // Step 3: Attempt device allocation.
    void * ptr = nullptr;
    if (try_device) {
        try {
            ptr = ggml_sycl_malloc_device_raw(size, queue_, label);
        } catch (...) {
            ptr = nullptr;
        }
    }

    if (ptr) {
        // Device allocation succeeded.
        result.ptr       = ptr;
        result.on_device = true;
        result.size      = size;

        // Track against budget as runtime reservation.
        used_.fetch_add(size, std::memory_order_relaxed);

        std::lock_guard<std::mutex> lock(managed_allocs_mutex_);
        managed_allocs_[ptr] = { size, true, false, host_zone_id::COUNT, lifetime };
        return result;
    }

    // Step 4: Host-pinned fallback — route through the pre-allocated pinned pool.
    // All host-pinned memory is pre-allocated at init; zero runtime malloc_host.
    bool         uses_pinned_pool = false;
    host_zone_id zone             = host_zone_id::COUNT;
    if (host_zones_configured()) {
        zone = host_zone_id::STAGING;
        ptr  = host_zone_alloc(zone, size, pinned_chunk_pool::DEFAULT_ALIGNMENT);
    } else {
        ptr = host_pool_alloc(size, pinned_chunk_pool::DEFAULT_ALIGNMENT);
    }
    uses_pinned_pool = (ptr != nullptr);
    if (!ptr) {
        // Pool exhausted — last resort: raw malloc_host (should not happen
        // after pre_allocate_all, but avoids hard failure during init).
        try {
            ptr = ggml_sycl_malloc_host(size, queue_, label);
        } catch (...) {
            ptr = nullptr;
        }
    }

    if (!ptr) {
        GGML_LOG_ERROR(
            "[UNIFIED-CACHE] allocate(%s): both device and host alloc failed "
            "(size=%.1f MB)\n",
            label, size / (1024.0 * 1024.0));
        return result;
    }

    result.ptr       = ptr;
    result.on_device = false;
    result.size      = size;

    // Host-pinned does NOT consume VRAM budget (it is malloc_host).
    std::lock_guard<std::mutex> lock(managed_allocs_mutex_);
    managed_allocs_[ptr] = { size, false, uses_pinned_pool, zone, lifetime };
    return result;
}

// --- unified_cache::deallocate() ---

void unified_cache::deallocate(void * ptr, size_t size, alloc_lifetime lifetime) {
    if (!ptr) {
        return;
    }
    (void) lifetime;  // Used for future per-lifetime diagnostics.

    managed_alloc_entry entry{};
    {
        std::lock_guard<std::mutex> lock(managed_allocs_mutex_);
        auto                        it = managed_allocs_.find(ptr);
        if (it == managed_allocs_.end()) {
            GGML_LOG_WARN("[UNIFIED-CACHE] deallocate: unknown pointer %p (size=%zu)\n", ptr, size);
            return;
        }
        entry = it->second;
        managed_allocs_.erase(it);
    }

    if (entry.on_device) {
        // Reverse the budget charge.
        saturating_sub_used(entry.size);
    }

    try {
        if (!entry.on_device && entry.uses_pinned_pool) {
            if (entry.host_zone == host_zone_id::COUNT) {
                host_pool_free(ptr, entry.size);
            }
            return;
        }
        sycl::free(ptr, queue_);
    } catch (const sycl::exception & e) {
        GGML_LOG_ERROR("[UNIFIED-CACHE] deallocate: sycl::free failed ptr=%p: %s\n", ptr, e.what());
    }
}

// --- Compute Arena ---
// Pre-reserved VRAM region for compute scratch buffers.
// In VRAM arena mode this aliases the SCRATCH TLSF zone. Outside arena mode,
// it falls back to the legacy single-allocation bump allocator.
// Must be called BEFORE S1-PRELOAD.

bool unified_cache::reserve_compute_arena(size_t arena_bytes) {
    if (compute_arena_ptr_ && compute_arena_size_ >= arena_bytes) {
        return true;  // Already reserved with sufficient capacity.
    }

    // VRAM arena path: scratch zone is already pre-allocated.
    if (arena_active()) {
        size_t zone_cap = zone_capacity(vram_zone_id::SCRATCH);
        if (zone_cap >= arena_bytes) {
            // Point compute_arena at the arena's scratch zone base.
            const auto & szone  = get_zone(vram_zone_id::SCRATCH);
            compute_arena_ptr_  = offset_to_ptr(szone.start);
            compute_arena_size_ = zone_cap;
            compute_arena_off_.store(0, std::memory_order_relaxed);
            GGML_LOG_INFO("[COMPUTE-ARENA] Using VRAM arena scratch zone: %.1f MB (offset=%.1f MB)\n",
                          zone_cap / (1024.0 * 1024.0), szone.start / (1024.0 * 1024.0));
            // Budget already charged when arena was reserved — don't double-count.
            return true;
        }
        // Arena owns ALL VRAM — no raw malloc_device possible.
        // Use the available zone capacity even if smaller than requested.
        if (zone_cap > 0) {
            const auto & szone  = get_zone(vram_zone_id::SCRATCH);
            compute_arena_ptr_  = offset_to_ptr(szone.start);
            compute_arena_size_ = zone_cap;
            compute_arena_off_.store(0, std::memory_order_relaxed);
            GGML_LOG_WARN(
                "[COMPUTE-ARENA] Arena scratch zone (%.1f MB) < requested (%.1f MB), "
                "using available capacity\n",
                zone_cap / (1024.0 * 1024.0), arena_bytes / (1024.0 * 1024.0));
            return true;
        }
        GGML_LOG_ERROR("[COMPUTE-ARENA] Arena scratch zone is 0 bytes, cannot reserve\n");
        return false;
    }

    // Free existing arena if resizing.
    if (compute_arena_ptr_) {
        try {
            sycl::free(compute_arena_ptr_, queue_);
        } catch (...) {
        }
        saturating_sub_used(compute_arena_size_);
        compute_arena_ptr_  = nullptr;
        compute_arena_size_ = 0;
        compute_arena_off_.store(0, std::memory_order_relaxed);
    }

    // Allocate a single contiguous VRAM block.
    try {
        compute_arena_ptr_ = sycl_aligned_malloc_device(arena_bytes, queue_);
    } catch (const sycl::exception & e) {
        GGML_LOG_ERROR("[COMPUTE-ARENA] sycl::aligned_alloc_device failed (%.1f MB): %s\n",
                       arena_bytes / (1024.0 * 1024.0), e.what());
        compute_arena_ptr_ = nullptr;
    }

    if (!compute_arena_ptr_) {
        GGML_LOG_ERROR("[COMPUTE-ARENA] Failed to reserve %.1f MB of VRAM\n", arena_bytes / (1024.0 * 1024.0));
        return false;
    }

    compute_arena_size_ = arena_bytes;
    compute_arena_off_.store(0, std::memory_order_relaxed);

    // Track arena bytes against the unified cache budget so S1-PRELOAD
    // sees reduced available VRAM and loads fewer weights.
    used_.fetch_add(arena_bytes, std::memory_order_relaxed);

    GGML_LOG_INFO("[COMPUTE-ARENA] Reserved %.1f MB VRAM for compute scratch\n", arena_bytes / (1024.0 * 1024.0));
    return true;
}

void * unified_cache::arena_alloc(size_t size) {
    if (!compute_arena_ptr_ || size == 0) {
        return nullptr;
    }

    const bool   profile_active = arena_pp_profile_active();
    const auto   t0             = profile_active ? arena_profile_clock::now() : arena_profile_clock::time_point{};
    const size_t aligned        = (size + 255) & ~size_t(255);

    if (arena_active()) {
        void * ptr = zone_alloc(vram_zone_id::SCRATCH, aligned, 256);
        if (profile_active) {
            t_arena_pp_profile.compute_arena_alloc_calls++;
            t_arena_pp_profile.compute_arena_alloc_bytes += aligned;
            if (!ptr) {
                t_arena_pp_profile.compute_arena_alloc_fail++;
            }
            t_arena_pp_profile.compute_arena_alloc_us += arena_profile_elapsed_us(t0);
        }
        return ptr;
    }

    // Align to 256 bytes for GPU coalescing.

    // Atomic bump allocator — lock-free.
    size_t off = compute_arena_off_.fetch_add(aligned, std::memory_order_relaxed);
    if (off + aligned > compute_arena_size_) {
        // Arena exhausted — roll back.
        compute_arena_off_.fetch_sub(aligned, std::memory_order_relaxed);
        if (profile_active) {
            t_arena_pp_profile.compute_arena_alloc_calls++;
            t_arena_pp_profile.compute_arena_alloc_fail++;
            t_arena_pp_profile.compute_arena_alloc_bytes += aligned;
            t_arena_pp_profile.compute_arena_alloc_us += arena_profile_elapsed_us(t0);
        }
        return nullptr;
    }

    if (profile_active) {
        t_arena_pp_profile.compute_arena_alloc_calls++;
        t_arena_pp_profile.compute_arena_alloc_bytes += aligned;
        t_arena_pp_profile.compute_arena_alloc_us += arena_profile_elapsed_us(t0);
    }
    return static_cast<uint8_t *>(compute_arena_ptr_) + off;
}

void unified_cache::arena_free(void * ptr, size_t size) {
    if (!compute_arena_ptr_ || !ptr) {
        return;
    }

    if (arena_active()) {
        zone_free(vram_zone_id::SCRATCH, ptr);
        return;
    }

    // Same alignment as arena_alloc().
    const size_t aligned = (size + 255) & ~size_t(255);
    const size_t off     = static_cast<uint8_t *>(ptr) - static_cast<uint8_t *>(compute_arena_ptr_);

    // Watermark reclaim: if this allocation sits at the current arena top,
    // rewind the bump pointer.  Unlike the previous last-alloc-only tracking,
    // this handles cascading LIFO frees correctly: when multiple pool_alloc
    // instances in a single op are freed in reverse order (C++ destructor
    // order), each free reclaims its space because each successive free lands
    // at the new arena top after the previous reclaim.
    //
    // Without cascading reclaim, only the last allocation per op was reclaimed
    // (the rest leaked until arena_reset), causing the 256 MB SCRATCH zone to
    // exhaust during PP512's 1158-node graph.
    //
    // Single-threaded assumption: pool_leg processes graph nodes sequentially
    // on one compute stream per device.
    size_t current_off = compute_arena_off_.load(std::memory_order_relaxed);
    if (off + aligned == current_off) {
        compute_arena_off_.store(off, std::memory_order_relaxed);
    }
    // Non-watermark free: no-op (space reclaimed at arena_reset).
}

void unified_cache::arena_reset() {
    const bool profile_active = arena_pp_profile_active();
    const auto t0             = profile_active ? arena_profile_clock::now() : arena_profile_clock::time_point{};
    if (arena_active()) {
        zone_reset(vram_zone_id::SCRATCH);
        if (profile_active) {
            t_arena_pp_profile.compute_arena_reset_calls++;
            t_arena_pp_profile.compute_arena_reset_us += arena_profile_elapsed_us(t0);
        }
        return;
    }
    compute_arena_off_.store(0, std::memory_order_relaxed);
    if (profile_active) {
        t_arena_pp_profile.compute_arena_reset_calls++;
        t_arena_pp_profile.compute_arena_reset_us += arena_profile_elapsed_us(t0);
    }
}

bool unified_cache::arena_owns(const void * ptr) const {
    if (!compute_arena_ptr_ || !ptr) {
        return false;
    }
    const auto p    = reinterpret_cast<uintptr_t>(ptr);
    const auto base = reinterpret_cast<uintptr_t>(compute_arena_ptr_);
    return p >= base && p < base + compute_arena_size_;
}

size_t unified_cache::compute_arena_capacity() const {
    return compute_arena_size_;
}

size_t unified_cache::compute_arena_used() const {
    if (arena_active()) {
        return zone_used(vram_zone_id::SCRATCH);
    }
    return compute_arena_off_.load(std::memory_order_relaxed);
}

// --- Inference Scratch Pool ---

bool unified_cache::reserve_scratch_pool(size_t pool_bytes) {
    if (scratch_pool_ptr_ && scratch_pool_size_ >= pool_bytes) {
        return true;  // Already large enough.
    }

    // Free existing pool if it exists but is too small.
    // Arena-owned pointers must NOT be sycl::free'd — free via TLSF zone_free instead.
    if (scratch_pool_ptr_) {
        if (arena_active() && vram_owns(scratch_pool_ptr_)) {
            zone_free(vram_zone_id::WEIGHT, scratch_pool_ptr_);
            scratch_pool_ptr_ = nullptr;
        } else {
            saturating_sub_used(scratch_pool_size_);
            try {
                sycl::free(scratch_pool_ptr_, queue_);
            } catch (...) {
            }
        }
        scratch_pool_ptr_  = nullptr;
        scratch_pool_size_ = 0;
        scratch_pool_off_.store(0, std::memory_order_relaxed);
    }

    // VRAM arena path: sub-allocate from the weight zone (persistent allocation).
    if (arena_active()) {
        void * ptr = zone_alloc(vram_zone_id::WEIGHT, pool_bytes);
        if (ptr) {
            scratch_pool_ptr_  = ptr;
            scratch_pool_size_ = pool_bytes;
            scratch_pool_off_.store(0, std::memory_order_relaxed);
            scratch_pool_hwm_ = 0;
            GGML_LOG_INFO("[UNIFIED-CACHE] Scratch pool reserved from arena weight zone: %.1f MB\n",
                          pool_bytes / (1024.0 * 1024.0));
            return true;
        }
        GGML_LOG_WARN(
            "[UNIFIED-CACHE] Arena weight zone full for scratch pool (%.1f MB), "
            "falling back to direct alloc\n",
            pool_bytes / (1024.0 * 1024.0));
    }

    // Allocate through our own allocate() path so budget/L0 checks apply.
    // Scratch pool MUST be on device — if allocate() falls back to host, reject it.
    vram_alloc_result res = allocate(pool_bytes, alloc_lifetime::PERSISTENT, "scratch_pool");
    if (!res.ptr || !res.on_device) {
        if (res.ptr && !res.on_device) {
            // Got host-pinned — not useful as scratch pool, release it.
            deallocate(res.ptr, pool_bytes, alloc_lifetime::PERSISTENT);
        }
        res.ptr = nullptr;
    }
    if (!res.ptr) {
        GGML_LOG_WARN("[UNIFIED-CACHE] reserve_scratch_pool: failed to allocate %.1f MB\n",
                      pool_bytes / (1024.0 * 1024.0));
        return false;
    }

    scratch_pool_ptr_  = res.ptr;
    scratch_pool_size_ = pool_bytes;
    scratch_pool_off_.store(0, std::memory_order_relaxed);
    scratch_pool_hwm_ = 0;

    GGML_LOG_INFO("[UNIFIED-CACHE] Scratch pool reserved: %.1f MB on device\n", pool_bytes / (1024.0 * 1024.0));
    return true;
}

void * unified_cache::get_scratch(size_t size) {
    if (!scratch_pool_ptr_ || size == 0) {
        return nullptr;
    }

    // Align to 256 bytes for GPU coalescing.
    const size_t aligned = (size + 255) & ~size_t(255);

    // Atomic bump allocator — lock-free.
    size_t off = scratch_pool_off_.fetch_add(aligned, std::memory_order_relaxed);
    if (off + aligned > scratch_pool_size_) {
        // Pool exhausted — roll back.
        scratch_pool_off_.fetch_sub(aligned, std::memory_order_relaxed);
        return nullptr;
    }

    // Track high-water mark (relaxed — diagnostic only).
    size_t new_hwm = off + aligned;
    size_t cur_hwm = scratch_pool_hwm_;
    while (new_hwm > cur_hwm) {
        // Not atomic — this is best-effort diagnostic.
        scratch_pool_hwm_ = new_hwm;
        cur_hwm           = new_hwm;
    }

    return static_cast<uint8_t *>(scratch_pool_ptr_) + off;
}

void unified_cache::return_scratch(void * ptr, size_t size) {
    // Stack discipline: we don't actually free individual allocations.
    // The pool is reset wholesale via reset_scratch_pool().
    (void) ptr;
    (void) size;
}

void unified_cache::reset_scratch_pool() {
    scratch_pool_off_.store(0, std::memory_order_relaxed);
}

// --- Expert Allocation ---

unified_cache::vram_alloc_result unified_cache::allocate_expert(size_t size) {
    return allocate(size, alloc_lifetime::SCRATCH, "expert");
}

// ============================================================================
// Phase 3: Free-Standing Wrappers
// ============================================================================

unified_cache::vram_alloc_result unified_cache_allocate(int                           device_id,
                                                        size_t                        size,
                                                        unified_cache::alloc_lifetime lifetime,
                                                        const char *                  tag) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        return {};
    }
    return cache->allocate(size, lifetime, tag);
}

void unified_cache_deallocate(int device_id, void * ptr, size_t size, unified_cache::alloc_lifetime lifetime) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (cache) {
        cache->deallocate(ptr, size, lifetime);
    }
}

bool unified_cache_reserve_compute_arena(int device_id, size_t arena_bytes) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        return false;
    }
    return cache->reserve_compute_arena(arena_bytes);
}

void * unified_cache_arena_alloc(int device_id, size_t size) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        return nullptr;
    }
    return cache->arena_alloc(size);
}

void unified_cache_arena_free(int device_id, void * ptr, size_t size) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (cache) {
        cache->arena_free(ptr, size);
    }
}

void * unified_cache_arena_alloc_weight(int device_id, size_t size) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (!cache || !cache->arena_active()) {
        return nullptr;
    }
    return cache->zone_alloc(vram_zone_id::WEIGHT, size, 64);
}

void unified_cache_arena_reset(int device_id) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (cache) {
        cache->arena_reset();
    }
}

bool unified_cache_arena_owns(int device_id, const void * ptr) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        return false;
    }
    return cache->arena_owns(ptr);
}

size_t unified_cache_compute_arena_capacity(int device_id) {
    auto * cache = get_unified_cache_for_device(device_id);
    return cache ? cache->compute_arena_capacity() : 0;
}

size_t unified_cache_compute_arena_used(int device_id) {
    auto * cache = get_unified_cache_for_device(device_id);
    return cache ? cache->compute_arena_used() : 0;
}

size_t unified_cache_kv_arena_capacity(int device_id) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (!cache || !cache->arena_active()) {
        return 0;
    }
    return cache->zone_available(vram_zone_id::KV);
}

size_t unified_cache_kv_arena_used(int device_id) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (!cache || !cache->arena_active()) {
        return 0;
    }
    return cache->zone_used(vram_zone_id::KV);
}

size_t unified_cache_arena_non_weight_used(int device) {
    auto * cache = get_unified_cache_for_device(device);
    if (!cache || !cache->arena_active()) {
        return 0;
    }
    return cache->zone_used(vram_zone_id::KV) + cache->zone_used(vram_zone_id::ONEDNN) +
           cache->zone_used(vram_zone_id::RUNTIME) + cache->zone_used(vram_zone_id::SCRATCH);
}

void * unified_cache_kv_arena_alloc(int device_id, size_t size) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (!cache || !cache->arena_active()) {
        return nullptr;
    }
    return cache->zone_alloc(vram_zone_id::KV, size);
}

memory_location query_location(const void * ptr, int device_hint) {
    memory_location loc{};
    if (!ptr) {
        return loc;
    }

    loc.ptr = const_cast<void *>(ptr);

    {
        std::lock_guard<std::mutex> lock(g_runtime_alloc_mutex);
        auto                        it = g_runtime_alloc_registry.find(loc.ptr);
        if (it != g_runtime_alloc_registry.end()) {
            const alloc_handle & h = it->second.handle;
            loc.device             = h.device;
            loc.tier               = h.tier;
            loc.role               = h.role;
            loc.zone               = h.vram_zone;
            loc.from_arena         = h.zone_managed && h.vram_zone != vram_zone_id::COUNT;
            loc.layout             = GGML_LAYOUT_AOS;
            if (h.host_zone != host_zone_id::COUNT && loc.role == alloc_role::OTHER) {
                loc.role = h.host_zone == host_zone_id::KV     ? alloc_role::KV :
                           h.host_zone == host_zone_id::WEIGHT ? alloc_role::WEIGHT :
                                                                 alloc_role::STAGING;
            }
            return loc;
        }
    }

    // Step 1: Check alloc_registry for tier classification (O(1) binary search).
    const auto * info = alloc_registry::instance().lookup(ptr);
    if (info) {
        loc.device = info->device_id;
        switch (info->type) {
            case alloc_type::DEVICE:
                loc.tier = alloc_tier::DEVICE_VRAM;
                break;
            case alloc_type::HOST_PINNED:
                loc.tier = alloc_tier::HOST_PINNED;
                break;
            case alloc_type::SHARED:
                loc.tier = alloc_tier::HOST_PINNED;  // treat shared as host-accessible
                break;
            case alloc_type::MMAP:
                loc.tier = alloc_tier::MMAP_TRACKED;
                break;
            default:
                loc.tier = alloc_tier::MMAP_TRACKED;
                break;
        }
    } else if (device_hint >= 0) {
        // Not registered — assume host/mmap (conservative).
        loc.device = device_hint;
        loc.tier   = alloc_tier::MMAP_TRACKED;
    }

    // Step 2: Check arena zone ownership (device allocations only).
    if (loc.tier == alloc_tier::DEVICE_VRAM) {
        int dev = (loc.device >= 0) ? loc.device : device_hint;
        if (dev >= 0) {
            auto * cache = get_unified_cache_for_device(dev);
            if (cache && cache->arena_active()) {
                if (cache->vram_owns(ptr)) {
                    loc.from_arena = true;
                    if (cache->zone_owns(vram_zone_id::SCRATCH, ptr)) {
                        loc.zone = vram_zone_id::SCRATCH;
                        loc.role = alloc_role::COMPUTE;
                    } else if (cache->zone_owns(vram_zone_id::RUNTIME, ptr)) {
                        loc.zone = vram_zone_id::RUNTIME;
                        loc.role = alloc_role::COMPUTE;
                    } else if (cache->zone_owns(vram_zone_id::KV, ptr)) {
                        loc.zone = vram_zone_id::KV;
                        loc.role = alloc_role::KV;
                    } else if (cache->zone_owns(vram_zone_id::ONEDNN, ptr)) {
                        loc.zone = vram_zone_id::ONEDNN;
                        loc.role = alloc_role::COMPUTE;
                    } else if (cache->zone_owns(vram_zone_id::WEIGHT, ptr)) {
                        loc.zone = vram_zone_id::WEIGHT;
                        loc.role = alloc_role::WEIGHT;
                    }
                }
            }
        }
    }

    return loc;
}

memory_location query_kv_location(int layer_id, int device) {
    memory_location loc{};
    loc.device = device;
    loc.role   = alloc_role::KV;
    loc.layout = GGML_LAYOUT_AOS;  // KV cache is always row-major

    if (device < 0) {
        // No device specified — assume host
        loc.tier = alloc_tier::HOST_PINNED;
        return loc;
    }

    auto & mgr = get_kv_tier_manager(device);
    if (!mgr.is_active()) {
        // No tiering configured — all KV on device (default path)
        loc.tier = alloc_tier::DEVICE_VRAM;
        return loc;
    }

    if (mgr.is_hot(static_cast<uint32_t>(layer_id))) {
        loc.tier = alloc_tier::DEVICE_VRAM;
        loc.zone = vram_zone_id::KV;
    } else {
        loc.tier = alloc_tier::HOST_PINNED;
    }

    return loc;
}

bool unified_cache_reserve_scratch_pool(int device_id, size_t pool_bytes) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        return false;
    }
    return cache->reserve_scratch_pool(pool_bytes);
}

void * unified_cache_get_scratch(int device_id, size_t size) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        return nullptr;
    }
    return cache->get_scratch(size);
}

void unified_cache_return_scratch(int device_id, void * ptr, size_t size) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (cache) {
        cache->return_scratch(ptr, size);
    }
}

void unified_cache_reset_scratch_pool(int device_id) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (cache) {
        cache->reset_scratch_pool();
    }
}

static size_t align_256(size_t n) {
    return (n + 255) & ~size_t(255);
}

static const char * scratch_layout_name(ggml_layout_mode layout) {
    switch (layout) {
        case GGML_LAYOUT_AOS:
            return "aos";
        case GGML_LAYOUT_SOA:
            return "soa";
        case GGML_LAYOUT_COALESCED:
            return "coalesced";
        case GGML_LAYOUT_MXFP4_I8:
            return "mxfp4_i8";
        case GGML_LAYOUT_MXFP4_DPAS:
            return "mxfp4_dpas";
        case GGML_LAYOUT_XMX_TILED:
            return "xmx_tiled";
        case GGML_LAYOUT_XMX_GEMM_TILED:
            return "xmx_gemm_tiled";
        case GGML_LAYOUT_ONEDNN_PACKED:
            return "onednn_packed";
        case GGML_LAYOUT_ONEDNN_WOQ:
            return "onednn_woq";
    }
    return "unknown";
}

static bool moe_q8_1_scratch_supported_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
            return true;
        default:
            return false;
    }
}

moe_q8_1_scratch_demand unified_cache_plan_moe_q8_1_scratch(const moe_q8_1_scratch_shape & shape) {
    moe_q8_1_scratch_demand demand;
    demand.weight_type   = shape.weight_type;
    demand.layout        = shape.layout;
    demand.input_cols    = shape.input_cols;
    demand.input_rows    = shape.input_rows;
    demand.buffer_count  = shape.graph_op_count;
    demand.device_id     = shape.device_id;
    demand.segment_id    = shape.segment_id;
    demand.layer_id      = shape.layer_id;
    demand.n_experts     = shape.n_experts;
    demand.n_expert_used = shape.n_expert_used;

    if (!moe_q8_1_scratch_supported_type(shape.weight_type) || shape.input_cols <= 0 || shape.input_rows <= 0 ||
        shape.graph_op_count <= 0) {
        return demand;
    }

    constexpr int64_t qk8_1            = 32;
    constexpr size_t  q8_1_ds_bytes    = 4;  // two fp16 scale values per QK8_1 block
    demand.supported                   = true;
    demand.input_cols_padded           = GGML_PAD(shape.input_cols, qk8_1);
    const size_t q8_1_qs_bytes_per_row = static_cast<size_t>(demand.input_cols_padded);
    const size_t q8_1_ds_bytes_per_row = static_cast<size_t>(demand.input_cols_padded / qk8_1) * q8_1_ds_bytes;
    const size_t q8_1_row_bytes        = q8_1_qs_bytes_per_row + q8_1_ds_bytes_per_row;
    demand.bytes_per_buffer            = static_cast<size_t>(shape.input_rows) * q8_1_row_bytes;
    demand.aligned_bytes_per_buffer    = align_256(demand.bytes_per_buffer);
    demand.total_bytes                 = static_cast<size_t>(shape.graph_op_count) * demand.aligned_bytes_per_buffer;
    return demand;
}

bool unified_cache_reserve_moe_q8_1_scratch(int device_id, const moe_q8_1_scratch_demand & demand) {
    if (!demand.supported || demand.total_bytes == 0) {
        return true;
    }

    auto * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        GGML_LOG_ERROR("[MOE-GRAPH-SCRATCH] device=%d no unified cache for Q8_1 scratch reserve\n", device_id);
        return false;
    }

    const size_t before_capacity = cache->scratch_pool_capacity();
    if (before_capacity >= demand.total_bytes) {
        GGML_SYCL_DEBUG(
            "[MOE-GRAPH-SCRATCH] reserve satisfied device=%d requested=%zu capacity=%zu layout=%s dtype=%d "
            "buffers=%d per_buffer=%zu aligned=%zu cols=%lld padded=%lld rows=%lld segment=%d layer=%d "
            "experts=%d used=%d\n",
            device_id, demand.total_bytes, before_capacity, scratch_layout_name(demand.layout), demand.weight_type,
            demand.buffer_count, demand.bytes_per_buffer, demand.aligned_bytes_per_buffer,
            (long long) demand.input_cols, (long long) demand.input_cols_padded, (long long) demand.input_rows,
            demand.segment_id, demand.layer_id, demand.n_experts, demand.n_expert_used);
        return true;
    }

    const bool   ok           = cache->reserve_scratch_pool(demand.total_bytes);
    const size_t new_capacity = cache->scratch_pool_capacity();
    if (!ok || new_capacity < demand.total_bytes) {
        GGML_LOG_ERROR(
            "[MOE-GRAPH-SCRATCH] reserve failed device=%d requested=%zu capacity_before=%zu capacity_after=%zu "
            "layout=%s dtype=%d buffers=%d per_buffer=%zu aligned=%zu cols=%lld padded=%lld rows=%lld "
            "segment=%d layer=%d experts=%d used=%d\n",
            device_id, demand.total_bytes, before_capacity, new_capacity, scratch_layout_name(demand.layout),
            demand.weight_type, demand.buffer_count, demand.bytes_per_buffer, demand.aligned_bytes_per_buffer,
            (long long) demand.input_cols, (long long) demand.input_cols_padded, (long long) demand.input_rows,
            demand.segment_id, demand.layer_id, demand.n_experts, demand.n_expert_used);
        return false;
    }

    GGML_SYCL_DEBUG(
        "[MOE-GRAPH-SCRATCH] reserved device=%d requested=%zu capacity_before=%zu capacity_after=%zu layout=%s "
        "dtype=%d buffers=%d per_buffer=%zu aligned=%zu cols=%lld padded=%lld rows=%lld segment=%d layer=%d "
        "experts=%d used=%d\n",
        device_id, demand.total_bytes, before_capacity, new_capacity, scratch_layout_name(demand.layout),
        demand.weight_type, demand.buffer_count, demand.bytes_per_buffer, demand.aligned_bytes_per_buffer,
        (long long) demand.input_cols, (long long) demand.input_cols_padded, (long long) demand.input_rows,
        demand.segment_id, demand.layer_id, demand.n_experts, demand.n_expert_used);
    return true;
}

bool unified_cache_allocate_moe_q8_1_graph_scratch(int                             device_id,
                                                   const moe_q8_1_scratch_demand & demand,
                                                   alloc_handle *                  out) {
    if (!out) {
        return false;
    }
    *out = {};

    if (!demand.supported || demand.total_bytes == 0) {
        return true;
    }

    if (device_id < 0 || device_id >= GGML_SYCL_MAX_DEVICES) {
        GGML_LOG_ERROR("[MOE-GRAPH-SCRATCH] graph allocation invalid device=%d requested=%zu\n", device_id,
                       demand.total_bytes);
        return false;
    }

    sycl::queue * queue = nullptr;
    try {
        queue = &ggml_sycl_get_device(device_id).default_queue();
    } catch (...) {
        GGML_LOG_ERROR("[MOE-GRAPH-SCRATCH] graph allocation missing queue device=%d requested=%zu\n", device_id,
                       demand.total_bytes);
        return false;
    }

    alloc_request req{};
    req.queue                               = queue;
    req.device                              = device_id;
    req.size                                = demand.total_bytes;
    req.intent.role                         = alloc_role::GRAPH_TMP;
    req.intent.category                     = runtime_category::GRAPH;
    req.intent.constraints.must_device      = true;
    req.intent.constraints.prefer_vram_zone = vram_zone_id::WEIGHT;

    auto *       cache            = get_unified_cache_for_device(device_id);
    const size_t scratch_capacity = cache ? cache->scratch_pool_capacity() : 0;
    if (!unified_alloc(req, out) || !out->ptr) {
        GGML_LOG_ERROR(
            "[MOE-GRAPH-SCRATCH] graph allocation failed device=%d requested=%zu scratch_capacity=%zu layout=%s "
            "dtype=%d buffers=%d per_buffer=%zu aligned=%zu cols=%lld padded=%lld rows=%lld segment=%d "
            "layer=%d experts=%d used=%d\n",
            device_id, demand.total_bytes, scratch_capacity, scratch_layout_name(demand.layout), demand.weight_type,
            demand.buffer_count, demand.bytes_per_buffer, demand.aligned_bytes_per_buffer,
            (long long) demand.input_cols, (long long) demand.input_cols_padded, (long long) demand.input_rows,
            demand.segment_id, demand.layer_id, demand.n_experts, demand.n_expert_used);
        *out = {};
        return false;
    }

    GGML_SYCL_DEBUG(
        "[MOE-GRAPH-SCRATCH] graph allocated device=%d ptr=%p requested=%zu scratch_capacity=%zu zone=%d "
        "layout=%s dtype=%d buffers=%d per_buffer=%zu aligned=%zu cols=%lld padded=%lld rows=%lld segment=%d "
        "layer=%d experts=%d used=%d\n",
        device_id, out->ptr, demand.total_bytes, scratch_capacity, static_cast<int>(out->vram_zone),
        scratch_layout_name(demand.layout), demand.weight_type, demand.buffer_count, demand.bytes_per_buffer,
        demand.aligned_bytes_per_buffer, (long long) demand.input_cols, (long long) demand.input_cols_padded,
        (long long) demand.input_rows, demand.segment_id, demand.layer_id, demand.n_experts, demand.n_expert_used);
    return true;
}

void unified_cache_grow_host_scratch_zone(size_t additional_bytes) {
    auto * cache = get_unified_cache_for_device(resolve_effective_device(0));
    if (cache) {
        cache->grow_scratch_zone(additional_bytes);
    }
}

void * unified_cache_host_zone_alloc(host_zone_id zone, size_t size, size_t alignment) {
    auto * cache = get_unified_cache_for_device(resolve_effective_device(0));
    if (!cache) {
        return nullptr;
    }
    return cache->host_zone_alloc(zone, size, alignment);
}

void * unified_cache_zone_alloc(int device_id, vram_zone_id zone, size_t size, size_t align) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        return nullptr;
    }
    return cache->zone_alloc(zone, size, align);
}

void unified_cache_zone_reset(int device_id, vram_zone_id zone) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (cache) {
        cache->zone_reset(zone);
    }
}

void unified_cache_reset_model_weight_entries(int device_id) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (cache) {
        cache->reset_model_weight_entries();
    }
}

void unified_cache_zone_free(int device_id, vram_zone_id zone, void * ptr) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (cache) {
        cache->zone_free(zone, ptr);
    }
}

void unified_cache_host_zone_reset(host_zone_id zone) {
    auto * cache = get_unified_cache_for_device(resolve_effective_device(0));
    if (cache) {
        cache->host_zone_reset(zone);
    }
}

size_t unified_cache_host_zone_used(host_zone_id zone) {
    auto * cache = get_unified_cache_for_device(resolve_effective_device(0));
    return cache ? cache->host_zone_used(zone) : 0;
}

size_t unified_cache_host_zone_capacity(host_zone_id zone) {
    auto * cache = get_unified_cache_for_device(resolve_effective_device(0));
    return cache ? cache->host_zone_capacity(zone) : 0;
}

size_t unified_cache_host_zone_largest_free_block(host_zone_id zone) {
    auto * cache = get_unified_cache_for_device(resolve_effective_device(0));
    return cache ? cache->host_zone_largest_free_block(zone) : 0;
}

unified_cache::vram_alloc_result unified_cache_allocate_expert(int device_id, size_t size) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        return {};
    }
    return cache->allocate_expert(size);
}

size_t unified_cache_available_device(int device_id) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        return 0;
    }
    return cache->available_device();
}

size_t unified_cache_available_budget(int device_id) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        return 0;
    }
    return cache->available_budget();
}

// ============================================================================
// Phase 4: Pre-allocated MoE Inference Buffers
// ============================================================================

// Per-device storage for pre-allocated MoE buffers.
static std::array<moe_inference_buffers, GGML_SYCL_MAX_DEVICES> g_moe_buffers{};
static std::mutex                                               g_moe_buffers_mutex;

// Internal: allocate the expert pointer table block as a single contiguous
// allocation, then partition it into per-table pointers.
// Each table is `n_experts * sizeof(void*)` bytes.
// We allocate one big block: n_tables * table_bytes, then point into it.
// This minimizes the number of cache allocations.

bool moe_preallocate_inference_buffers(int device_id, const moe_buffer_params & params) {
    if (device_id < 0 || device_id >= GGML_SYCL_MAX_DEVICES) {
        GGML_LOG_ERROR("[MOE-PREALLOC] invalid device_id=%d\n", device_id);
        return false;
    }
    if (params.n_experts <= 0) {
        GGML_LOG_WARN("[MOE-PREALLOC] n_experts=%d, skipping pre-allocation\n", params.n_experts);
        return true;  // Not an error — model may not be MoE.
    }

    std::lock_guard<std::mutex> lock(g_moe_buffers_mutex);
    moe_inference_buffers &     bufs = g_moe_buffers[device_id];
    if (bufs.initialized) {
        return true;  // Already done.
    }

    const int    n_tables          = params.n_moe_layers * std::max(params.n_moe_tensors, 1);
    const size_t table_bytes       = static_cast<size_t>(params.n_experts) * sizeof(void *);
    const size_t total_table_bytes = static_cast<size_t>(n_tables) * table_bytes;

    // IDs staging: n_expert_used * max_batch * sizeof(int32_t)
    const size_t ids_bytes =
        static_cast<size_t>(params.n_expert_used) * static_cast<size_t>(params.max_batch) * sizeof(int32_t);
    const size_t compact_bytes =
        static_cast<size_t>(params.n_expert_used) * static_cast<size_t>(params.max_batch) * sizeof(void *);

    GGML_LOG_INFO(
        "[MOE-PREALLOC] device %d: n_tables=%d table_bytes=%zu "
        "total_table=%.1f KB ids_staging=%.1f KB compact=%.1f KB\n",
        device_id, n_tables, table_bytes, total_table_bytes / 1024.0, ids_bytes / 1024.0, compact_bytes / 1024.0);

    // --- Allocate expert pointer tables ---
    if (n_tables > 0 && table_bytes > 0) {
        auto table_result = unified_cache_allocate(device_id, total_table_bytes,
                                                   unified_cache::alloc_lifetime::PERSISTENT, "moe_expert_ptr_tables");

        if (!table_result) {
            GGML_LOG_ERROR(
                "[MOE-PREALLOC] failed to allocate expert pointer tables "
                "(%.1f KB)\n",
                total_table_bytes / 1024.0);
            return false;
        }

        // Allocate the host-side array of pointers into the contiguous block.
        bufs.expert_ptr_tables = static_cast<void **>(::malloc(static_cast<size_t>(n_tables) * sizeof(void *)));
        if (!bufs.expert_ptr_tables) {
            unified_cache_deallocate(device_id, table_result.ptr, total_table_bytes,
                                     unified_cache::alloc_lifetime::PERSISTENT);
            GGML_LOG_ERROR("[MOE-PREALLOC] failed to allocate host table pointer array\n");
            return false;
        }

        // Partition the contiguous device block into per-table pointers.
        auto * base = static_cast<uint8_t *>(table_result.ptr);
        for (int i = 0; i < n_tables; i++) {
            bufs.expert_ptr_tables[i] = base + static_cast<size_t>(i) * table_bytes;
        }

        bufs.n_tables         = n_tables;
        bufs.table_bytes      = table_bytes;
        bufs.tables_on_device = table_result.on_device;

        // Zero-fill the tables (they'll be populated by update_moe_ptr_table).
        auto * cache = get_unified_cache_for_device(device_id);
        if (cache && table_result.on_device) {
            try {
                cache->get_queue().memset(table_result.ptr, 0, total_table_bytes).wait();
            } catch (...) {
                GGML_LOG_WARN("[MOE-PREALLOC] memset of expert pointer tables failed\n");
            }
        }
    }

    // --- Allocate MoE IDs staging ---
    if (ids_bytes > 0) {
        auto ids_result =
            unified_cache_allocate(device_id, ids_bytes, unified_cache::alloc_lifetime::PERSISTENT, "moe_ids_staging");

        if (!ids_result) {
            GGML_LOG_ERROR(
                "[MOE-PREALLOC] failed to allocate IDs staging "
                "(%.1f KB)\n",
                ids_bytes / 1024.0);
            // Tables were allocated — leave them, partial success.
        } else {
            bufs.ids_staging       = ids_result.ptr;
            bufs.ids_staging_bytes = ids_bytes;
            bufs.ids_on_device     = ids_result.on_device;
        }
    }

    // --- Allocate compact selected-row pointer list ---
    if (compact_bytes > 0) {
        auto compact_result = unified_cache_allocate(device_id, compact_bytes,
                                                     unified_cache::alloc_lifetime::PERSISTENT, "moe_compact_ptrs");

        if (!compact_result) {
            GGML_LOG_ERROR(
                "[MOE-PREALLOC] failed to allocate compact pointer list "
                "(%.1f KB)\n",
                compact_bytes / 1024.0);
        } else {
            bufs.compact_ptrs       = compact_result.ptr;
            bufs.compact_ptrs_bytes = compact_bytes;
            bufs.compact_on_device  = compact_result.on_device;
        }

        auto missing_result = unified_cache_allocate(device_id, sizeof(int), unified_cache::alloc_lifetime::PERSISTENT,
                                                     "moe_compact_missing");
        if (!missing_result) {
            GGML_LOG_ERROR("[MOE-PREALLOC] failed to allocate compact missing flag\n");
        } else {
            bufs.compact_missing           = static_cast<int *>(missing_result.ptr);
            bufs.compact_missing_bytes     = sizeof(int);
            bufs.compact_missing_on_device = missing_result.on_device;
        }
    }

    bufs.initialized = true;

    GGML_LOG_INFO("[MOE-PREALLOC] device %d: tables=%s (%d x %zu B) ids=%s (%.1f KB) compact=%s (%.1f KB)\n", device_id,
                  bufs.tables_on_device ? "VRAM" : "host", bufs.n_tables, bufs.table_bytes,
                  bufs.ids_on_device ? "VRAM" : "host", bufs.ids_staging_bytes / 1024.0,
                  bufs.compact_on_device ? "VRAM" : "host", bufs.compact_ptrs_bytes / 1024.0);

    return true;
}

const moe_inference_buffers * moe_get_inference_buffers(int device_id) {
    if (device_id < 0 || device_id >= GGML_SYCL_MAX_DEVICES) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_moe_buffers_mutex);
    if (!g_moe_buffers[device_id].initialized) {
        return nullptr;
    }
    return &g_moe_buffers[device_id];
}

void * moe_get_expert_ptr_table(int device_id, int table_index) {
    if (device_id < 0 || device_id >= GGML_SYCL_MAX_DEVICES) {
        return nullptr;
    }
    // No lock needed — read-only after initialization.
    const moe_inference_buffers & bufs = g_moe_buffers[device_id];
    if (!bufs.initialized || !bufs.expert_ptr_tables) {
        return nullptr;
    }
    if (table_index < 0 || table_index >= bufs.n_tables) {
        return nullptr;
    }
    return bufs.expert_ptr_tables[table_index];
}

void * moe_get_ids_staging(int device_id, size_t needed_bytes) {
    if (device_id < 0 || device_id >= GGML_SYCL_MAX_DEVICES) {
        return nullptr;
    }
    // No lock needed — read-only after initialization.
    const moe_inference_buffers & bufs = g_moe_buffers[device_id];
    if (!bufs.initialized || !bufs.ids_staging) {
        return nullptr;
    }
    if (needed_bytes > bufs.ids_staging_bytes) {
        GGML_SYCL_DEBUG("[MOE-PREALLOC] ids_staging too small: need %zu have %zu\n", needed_bytes,
                        bufs.ids_staging_bytes);
        return nullptr;
    }
    return bufs.ids_staging;
}

void * moe_get_compact_ptrs(int device_id, size_t needed_bytes) {
    if (device_id < 0 || device_id >= GGML_SYCL_MAX_DEVICES) {
        return nullptr;
    }
    const moe_inference_buffers & bufs = g_moe_buffers[device_id];
    if (!bufs.initialized || !bufs.compact_ptrs) {
        return nullptr;
    }
    if (needed_bytes > bufs.compact_ptrs_bytes) {
        GGML_SYCL_DEBUG("[MOE-PREALLOC] compact pointer list too small: need %zu have %zu\n", needed_bytes,
                        bufs.compact_ptrs_bytes);
        return nullptr;
    }
    return bufs.compact_ptrs;
}

int * moe_get_compact_missing_flag(int device_id) {
    if (device_id < 0 || device_id >= GGML_SYCL_MAX_DEVICES) {
        return nullptr;
    }
    const moe_inference_buffers & bufs = g_moe_buffers[device_id];
    if (!bufs.initialized || !bufs.compact_missing || bufs.compact_missing_bytes < sizeof(int)) {
        return nullptr;
    }
    return bufs.compact_missing;
}

void moe_free_inference_buffers(int device_id) {
    if (device_id < 0 || device_id >= GGML_SYCL_MAX_DEVICES) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_moe_buffers_mutex);
    moe_inference_buffers &     bufs = g_moe_buffers[device_id];
    if (!bufs.initialized) {
        return;
    }

    // Free the contiguous table block (first table pointer is the base).
    if (bufs.expert_ptr_tables && bufs.n_tables > 0) {
        void * base = bufs.expert_ptr_tables[0];
        if (base) {
            const size_t total = static_cast<size_t>(bufs.n_tables) * bufs.table_bytes;
            unified_cache_deallocate(device_id, base, total, unified_cache::alloc_lifetime::PERSISTENT);
        }
        ::free(bufs.expert_ptr_tables);
    }

    // Free IDs staging.
    if (bufs.ids_staging) {
        unified_cache_deallocate(device_id, bufs.ids_staging, bufs.ids_staging_bytes,
                                 unified_cache::alloc_lifetime::PERSISTENT);
    }

    if (bufs.compact_ptrs) {
        unified_cache_deallocate(device_id, bufs.compact_ptrs, bufs.compact_ptrs_bytes,
                                 unified_cache::alloc_lifetime::PERSISTENT);
    }
    if (bufs.compact_missing) {
        unified_cache_deallocate(device_id, bufs.compact_missing, bufs.compact_missing_bytes,
                                 unified_cache::alloc_lifetime::PERSISTENT);
    }

    bufs = {};  // Reset to zero state.
}

// === VRAM Arena Implementation ===

bool vram_arena_enabled() {
    static const bool enabled = []() {
        const char * env = std::getenv("GGML_SYCL_VRAM_ARENA");
        // Default ON — opt out with GGML_SYCL_VRAM_ARENA=0
        return env == nullptr || std::atoi(env) != 0;
    }();
    return enabled;
}

// --- Device pool arena helpers (called from device-pool.hpp inline methods) ---

void * device_pool_arena_alloc(unified_cache * cache, size_t size, size_t align) {
    if (!cache || !cache->arena_active()) {
        return nullptr;
    }
    return cache->zone_alloc(vram_zone_id::WEIGHT, size, align);
}

bool device_pool_arena_owns(const unified_cache * cache, const void * ptr) {
    if (!cache || !cache->arena_active()) {
        return false;
    }
    return cache->vram_owns(ptr);
}

bool unified_cache::arena_reserve(sycl::queue & queue,
                                  size_t        budget_bytes,
                                  size_t        max_alloc_size,
                                  size_t        scratch_bytes,
                                  size_t        onednn_bytes,
                                  size_t        runtime_bytes,
                                  size_t        device_total_vram) {
    if (arena_base_) {
        // Same model, new context — reset ephemeral zones so KV/runtime
        // space from the previous context is reclaimable.  Weight zone is
        // untouched (weights persist across contexts).  SCRATCH resets at
        // graph_compute start via arena_reset().
        zone_reset(vram_zone_id::KV);
        zone_reset(vram_zone_id::RUNTIME);

        // Host zones: reset KV and STAGING so host-pinned buffers from the
        // previous context (output buffer, KV spill) are reclaimable.
        // WEIGHT is persistent, SCRATCH resets at graph_compute.
        // Use host_zone_reset() (not host_arena_->zone_reset()) to also
        // purge the registry of stale host zone entries.
        host_zone_reset(host_zone_id::KV);
        host_zone_reset(host_zone_id::STAGING);

        // Arena zones are the single source of truth for runtime tracking.
        // zone_reset() above already zeroes zone_used() for KV and RUNTIME.
        return true;
    }

    arena_queue_ = &queue;

    // Align zone sizes to 256 bytes.
    scratch_bytes = (scratch_bytes + 255) & ~size_t(255);
    onednn_bytes  = (onednn_bytes + 255) & ~size_t(255);
    runtime_bytes = (runtime_bytes + 255) & ~size_t(255);

    // Try single-chunk first.  All zones share one contiguous allocation.
    // Falls back to N-chunk (tail zones in last chunk, weights spread across
    // earlier chunks) if a single allocation would exceed the runtime's
    // per-allocation cap (~1.5 GB on Arc B580 under patched libze 1.14.37435).
    int dev_id = ggml_sycl_get_device_id_from_queue(queue);

    void * ptr        = nullptr;
    size_t alloc_size = budget_bytes;

    // Per-chunk cap used ONLY for N-chunk piece sizing (single-chunk attempt
    // below is unconditional — let the runtime tell us if it doesn't fit).
    // Use the L0-advertised `max_mem_alloc_size` directly: that is the size
    // the runtime promises to accept.  No artificial safety margin — empirical
    // evidence (master HEAD on patched libze 1.14.37435) shows allocations up
    // to the cap succeed, and trimming the value here would manufacture
    // unnecessary chunking on hardware with high reported caps.
    const size_t per_chunk_cap = max_alloc_size;

    // Cap upfront reservation so non-arena runtime allocations (transient
    // ggml compute outputs, DMA staging, planner-side mallocs) still fit in
    // residual VRAM (bead llama.cpp-w2ptt: oversubscribed arenas leave
    // <2 GB free and graph-compute ops fail with err 40 / OUT_OF_RESOURCES on
    // Mistral 7B GET_ROWS).  Caller passes total VRAM rather than us probing
    // because this routine may run inside ggml_sycl_init() static init,
    // where ggml_backend_sycl_get_device_memory() would reenter
    // ggml_sycl_info() and deadlock.  Caller passes 0 to opt out of the cap
    // entirely (e.g. mid-init paths where the live VRAM query would deadlock
    // and no hint was supplied) — in that case we trust the caller's budget
    // and skip the headroom subtraction.
    if (device_total_vram > 0) {
        const size_t runtime_headroom = std::max<size_t>(2ull * 1024ull * 1024ull * 1024ull, device_total_vram / 8);
        if (device_total_vram > runtime_headroom) {
            const size_t arena_cap = device_total_vram - runtime_headroom;
            if (alloc_size > arena_cap) {
                const size_t prev_alloc = alloc_size;
                alloc_size              = arena_cap;
                GGML_LOG_INFO(
                    "[VRAM-ARENA] Capping arena at %.1f MB (was %.1f MB) to leave %.0f MB headroom for runtime "
                    "allocs\n",
                    alloc_size / (1024.0 * 1024.0), prev_alloc / (1024.0 * 1024.0),
                    runtime_headroom / (1024.0 * 1024.0));
            }
        }
    } else {
        GGML_SYCL_DEBUG(
            "[VRAM-ARENA] device_total_vram=0, skipping runtime-headroom cap "
            "(arena reserved at full budget %.1f MB)\n",
            alloc_size / (1024.0 * 1024.0));
    }

    // Keep every zone base naturally aligned.  The headroom cap above can
    // produce byte-granular arena sizes (for example total_vram - total_vram/6),
    // and the tail zones are carved from the end of the arena.  If alloc_size is
    // not rounded down first, RUNTIME/ONEDNN/SCRATCH suballocations inherit a
    // misaligned base pointer even though each zone size is aligned.
    constexpr size_t k_arena_size_alignment = 2ull * 1024ull * 1024ull;
    alloc_size                              = (alloc_size / k_arena_size_alignment) * k_arena_size_alignment;

    // Minimum shared (KV+WEIGHT) zone that is worth carving out.  Below this
    // the arena is useless for weight/KV storage and only aliases the tail
    // zones on top of each other — see nryi9 rootcause.  Refuse to reserve;
    // caller falls back to per-entry allocation via the existing path at
    // unified_cache::unified_cache() (log line "[VRAM-ARENA] Failed on
    // device %d, falling back to per-entry allocation").
    constexpr size_t k_min_shared_bytes = 16ull * 1024ull * 1024ull;  // 16 MB

    // Always try single-chunk first.  per_chunk_cap is the L0-advertised
    // max_mem_alloc_size; the runtime promises to accept up to that, so a
    // single-shot attempt is the right default.  Fall through to N-chunk
    // only if the single allocation actually fails.
    {
        const size_t tail_bytes = onednn_bytes + runtime_bytes + scratch_bytes;
        if (alloc_size < tail_bytes + k_min_shared_bytes) {
            GGML_LOG_WARN(
                "[VRAM-ARENA] Insufficient budget for single-chunk arena: "
                "%.1f MB < tail %.1f MB + min shared %.1f MB; refusing reservation\n",
                alloc_size / (1024.0 * 1024.0), tail_bytes / (1024.0 * 1024.0), k_min_shared_bytes / (1024.0 * 1024.0));
        } else {
            try {
                ptr = sycl_aligned_malloc_device(alloc_size, queue);
            } catch (const sycl::exception & e) {
                GGML_LOG_WARN("[VRAM-ARENA] Single alloc (%.1f MB) failed: %s\n", alloc_size / (1024.0 * 1024.0),
                              e.what());
                ptr = nullptr;
            }
        }
        if (ptr) {
            alloc_registry::instance().register_alloc(ptr, alloc_size, dev_id, alloc_type::DEVICE);

            arena_chunks_.clear();
            arena_chunks_.push_back({ ptr, alloc_size });
            arena_base_ = ptr;
            arena_size_ = alloc_size;

            // Single-chunk zone layout:
            //   [KV→ ...free... ←WEIGHT] [ONEDNN] [RUNTIME] [SCRATCH]
            // ONEDNN, RUNTIME, and SCRATCH zones are placed at the END (high
            // addresses) to work around a compute-runtime bug where
            // sycl::get_pointer_type() returns 'unknown' for low-offset
            // sub-allocations within large sycl::malloc_device chunks.
            // oneDNN's dnnl_memory_create validates USM pointer type and
            // rejects 'unknown' pointers.
            //
            // Invariant: alloc_size >= tail_bytes + k_min_shared_bytes
            // (verified above) so shared_bytes > 0 here — there is no
            // silent zero-sized KV+WEIGHT zone.
            const size_t tail_bytes_final = onednn_bytes + runtime_bytes + scratch_bytes;
            const size_t shared_bytes     = alloc_size - tail_bytes_final;

            auto & kz = arena_zones_[static_cast<int>(vram_zone_id::KV)];
            kz.start  = 0;
            kz.size   = shared_bytes;
            kz.used.store(0, std::memory_order_relaxed);

            auto & wz = arena_zones_[static_cast<int>(vram_zone_id::WEIGHT)];
            wz.start  = 0;
            wz.size   = shared_bytes;
            wz.used.store(0, std::memory_order_relaxed);

            auto & oz = arena_zones_[static_cast<int>(vram_zone_id::ONEDNN)];
            oz.start  = shared_bytes;
            oz.size   = onednn_bytes;
            oz.used.store(0, std::memory_order_relaxed);

            auto & rz = arena_zones_[static_cast<int>(vram_zone_id::RUNTIME)];
            rz.start  = shared_bytes + onednn_bytes;
            rz.size   = runtime_bytes;
            rz.used.store(0, std::memory_order_relaxed);

            auto & sz = arena_zones_[static_cast<int>(vram_zone_id::SCRATCH)];
            sz.start  = shared_bytes + onednn_bytes + runtime_bytes;
            sz.size   = scratch_bytes;
            sz.used.store(0, std::memory_order_relaxed);

            GGML_LOG_INFO(
                "[VRAM-ARENA] Reserved single chunk: %.1f MB "
                "(scratch=%.1f, runtime=%.1f, oneDNN=%.1f, shared KV+weight=%.1f MB)\n",
                alloc_size / (1024.0 * 1024.0), scratch_bytes / (1024.0 * 1024.0), runtime_bytes / (1024.0 * 1024.0),
                onednn_bytes / (1024.0 * 1024.0), shared_bytes / (1024.0 * 1024.0));

            // Initialize TLSF allocators per zone.
            // Single-chunk: KV and WEIGHT share one allocator in KV zone.
            for (int i = 0; i < static_cast<int>(vram_zone_id::COUNT); i++) {
                auto & z = arena_zones_[i];
                if (i == static_cast<int>(vram_zone_id::WEIGHT)) {
                    // WEIGHT delegates to KV's allocator in single-chunk mode.
                    z.allocator = nullptr;
                } else if (z.size > 0) {
                    z.allocator = std::make_unique<tlsf_allocator>(z.size);
                }
            }

            return true;
        }
    }

    // NOTE: this fallback only fires when the single-chunk sycl::malloc_device
    // above actually fails (returned nullptr or threw).  On Arc B580 + patched
    // libze 1.14.37435 the single alloc currently always succeeds and this
    // fallback is dormant.  Activates on any GPU/runtime combo that enforces
    // a real per-allocation cap below the budget (stock libze with under-
    // reporting drivers, smaller GPUs, or future drivers that fail oversized
    // allocs cleanly).
    // N-chunk fallback.  Layout invariant: tail zones (oneDNN, RUNTIME,
    // SCRATCH) live entirely in the LAST chunk so they remain contiguous.
    // The single-chunk path documents why: oneDNN's dnnl_memory_create
    // rejects USM pointers whose sycl::get_pointer_type() returns 'unknown',
    // and the compute-runtime returns 'unknown' for low-offset sub-allocs
    // within large malloc_device chunks.  Keeping the tail zones at the END
    // of the LAST chunk reproduces the working single-chunk layout for those
    // zones while letting weights overflow into earlier chunks.
    //
    // Plan:
    //   - last chunk: KV zone (head) + ONEDNN + RUNTIME + SCRATCH (tail).
    //     Sized at min(per_chunk_cap, kv_share + tail_bytes).
    //   - chunks [0..N-2]: WEIGHT zone, each sized at per_chunk_cap, total
    //     summing to alloc_size - last_chunk_size.
    //
    // Rationale for "tail-last": logical offset 0 maps to chunks[0] (the
    // first weight chunk), preserving the single-chunk invariant that the
    // WEIGHT zone starts at offset 0 from the layout pool's perspective.
    const size_t tail_bytes = onednn_bytes + runtime_bytes + scratch_bytes;

    // Pick the last (tail) chunk size: at most per_chunk_cap, at least
    // tail_bytes + k_min_shared_bytes (so KV has a sane carve-out).  If the
    // total budget is smaller than that, refuse — single-chunk path would
    // already have caught this case via its own min-shared check.
    const size_t tail_chunk_min = tail_bytes + k_min_shared_bytes;
    if (per_chunk_cap < tail_chunk_min) {
        GGML_LOG_WARN(
            "[VRAM-ARENA] Per-chunk cap %.1f MB too small to hold tail zones (%.1f MB) + min KV (%.1f MB); "
            "refusing reservation\n",
            per_chunk_cap / (1024.0 * 1024.0), tail_bytes / (1024.0 * 1024.0), k_min_shared_bytes / (1024.0 * 1024.0));
        return false;
    }

    // Want as much KV in the tail chunk as the cap allows; any leftover
    // becomes weight chunks.  Aim for tail = per_chunk_cap when the budget
    // is large enough, else what budget allows.
    size_t last_chunk_size = std::min<size_t>(per_chunk_cap, alloc_size);
    if (last_chunk_size < tail_chunk_min) {
        GGML_LOG_WARN(
            "[VRAM-ARENA] Insufficient budget for N-chunk arena: total %.1f MB < tail %.1f MB + min KV %.1f MB; "
            "refusing reservation\n",
            alloc_size / (1024.0 * 1024.0), tail_bytes / (1024.0 * 1024.0), k_min_shared_bytes / (1024.0 * 1024.0));
        return false;
    }

    const size_t weight_total = alloc_size - last_chunk_size;
    if (weight_total == 0) {
        // alloc_size fit in one chunk but the single-chunk path couldn't allocate it.
        // The N-chunk path would just retry the same alloc — refuse instead.
        GGML_LOG_WARN(
            "[VRAM-ARENA] N-chunk path reached with single-chunk-sized budget %.1f MB after single alloc failure; "
            "refusing reservation\n",
            alloc_size / (1024.0 * 1024.0));
        return false;
    }
    if (weight_total < k_min_shared_bytes) {
        GGML_LOG_WARN(
            "[VRAM-ARENA] Insufficient budget for N-chunk arena: weight total %.1f MB < min %.1f MB; "
            "refusing reservation\n",
            weight_total / (1024.0 * 1024.0), k_min_shared_bytes / (1024.0 * 1024.0));
        return false;
    }

    // Plan weight chunk sizes: each ≤ per_chunk_cap; final sliver picks up the remainder.
    std::vector<size_t> chunk_sizes;
    {
        size_t remaining = weight_total;
        while (remaining > per_chunk_cap) {
            chunk_sizes.push_back(per_chunk_cap);
            remaining -= per_chunk_cap;
        }
        if (remaining > 0) {
            chunk_sizes.push_back(remaining);
        }
    }
    chunk_sizes.push_back(last_chunk_size);  // tail chunk last

    GGML_LOG_INFO("[VRAM-ARENA] Reserving N-chunk arena: %zu chunks, %.1f MB total (per-chunk cap %.1f MB)\n",
                  chunk_sizes.size(), alloc_size / (1024.0 * 1024.0), per_chunk_cap / (1024.0 * 1024.0));

    // Allocate chunks; on any failure, free the ones already reserved.
    arena_chunks_.clear();
    arena_chunks_.reserve(chunk_sizes.size());
    for (size_t i = 0; i < chunk_sizes.size(); ++i) {
        void * p = nullptr;
        try {
            p = sycl_aligned_malloc_device(chunk_sizes[i], queue);
        } catch (const sycl::exception & e) {
            GGML_LOG_WARN("[VRAM-ARENA] N-chunk: chunk %zu (%.1f MB) failed: %s\n", i,
                          chunk_sizes[i] / (1024.0 * 1024.0), e.what());
            p = nullptr;
        } catch (...) {
            p = nullptr;
        }
        if (!p) {
            GGML_LOG_WARN("[VRAM-ARENA] N-chunk: chunk %zu (%.1f MB) failed\n", i, chunk_sizes[i] / (1024.0 * 1024.0));
            // Roll back any successful allocs.  Log on free failure so a
            // silently leaked chunk leaves a breadcrumb in the operator log.
            for (size_t j = 0; j < arena_chunks_.size(); ++j) {
                auto & c = arena_chunks_[j];
                if (c.ptr) {
                    alloc_registry::instance().unregister_alloc(c.ptr);
                    try {
                        sycl::free(c.ptr, queue);
                    } catch (const sycl::exception & e) {
                        GGML_LOG_WARN("[VRAM-ARENA] N-chunk rollback: free of chunk %zu failed (leaking): %s\n", j,
                                      e.what());
                    } catch (...) {
                        GGML_LOG_WARN("[VRAM-ARENA] N-chunk rollback: free of chunk %zu failed (leaking)\n", j);
                    }
                }
            }
            arena_chunks_.clear();
            return false;
        }
        alloc_registry::instance().register_alloc(p, chunk_sizes[i], dev_id, alloc_type::DEVICE);
        arena_chunks_.push_back({ p, chunk_sizes[i] });
    }

    arena_base_ = arena_chunks_.front().ptr;
    arena_size_ = alloc_size;

    // Logical-offset layout (cumulative across chunks; offset_to_ptr/ptr_to_offset translate):
    //   [WEIGHT (chunks 0..N-2, weight_total bytes)]
    //   [KV (head of last chunk, up to last_chunk_size - tail_bytes bytes;
    //        0 if tail consumes the chunk — guarded above by tail_chunk_min)]
    //   [ONEDNN] [RUNTIME] [SCRATCH] (tail of last chunk)
    auto & wz = arena_zones_[static_cast<int>(vram_zone_id::WEIGHT)];
    wz.start  = 0;
    wz.size   = weight_total;
    wz.used.store(0, std::memory_order_relaxed);

    const size_t kv_size = last_chunk_size > tail_bytes ? last_chunk_size - tail_bytes : 0;

    size_t off = weight_total;
    auto & kz  = arena_zones_[static_cast<int>(vram_zone_id::KV)];
    kz.start   = off;
    kz.size    = kv_size;
    kz.used.store(0, std::memory_order_relaxed);
    off += kz.size;

    auto & oz = arena_zones_[static_cast<int>(vram_zone_id::ONEDNN)];
    oz.start  = off;
    oz.size   = onednn_bytes;
    oz.used.store(0, std::memory_order_relaxed);
    off += oz.size;

    auto & rz = arena_zones_[static_cast<int>(vram_zone_id::RUNTIME)];
    rz.start  = off;
    rz.size   = runtime_bytes;
    rz.used.store(0, std::memory_order_relaxed);
    off += rz.size;

    auto & sz = arena_zones_[static_cast<int>(vram_zone_id::SCRATCH)];
    sz.start  = off;
    sz.size   = scratch_bytes;
    sz.used.store(0, std::memory_order_relaxed);

    GGML_LOG_INFO(
        "[VRAM-ARENA] Reserved %zu chunks, %.1f MB total "
        "(weight=%.1f, KV=%.1f, oneDNN=%.1f, runtime=%.1f, scratch=%.1f MB)\n",
        arena_chunks_.size(), alloc_size / (1024.0 * 1024.0), wz.size / (1024.0 * 1024.0), kz.size / (1024.0 * 1024.0),
        oz.size / (1024.0 * 1024.0), rz.size / (1024.0 * 1024.0), sz.size / (1024.0 * 1024.0));

    // Initialize TLSF allocators (N-chunk path).
    //   - WEIGHT spans chunks 0..N-2 — one TLSF per physical weight chunk so
    //     no allocation crosses a chunk boundary.  z.allocator is left null;
    //     zone_alloc/zone_free walk weight_chunk_allocators_ instead.
    //   - KV/ONEDNN/RUNTIME/SCRATCH all live entirely in the last (tail)
    //     chunk — one TLSF each over their zone size, as before.
    weight_chunk_allocators_.clear();
    {
        size_t logical = 0;
        for (int i = 0; i + 1 < static_cast<int>(arena_chunks_.size()); ++i) {
            weight_chunk_alloc wca;
            wca.chunk_idx     = i;
            wca.logical_start = logical;
            wca.allocator     = std::make_unique<tlsf_allocator>(arena_chunks_[i].size);
            weight_chunk_allocators_.push_back(std::move(wca));
            logical += arena_chunks_[i].size;
        }
    }
    for (int i = 0; i < static_cast<int>(vram_zone_id::COUNT); i++) {
        auto & z = arena_zones_[i];
        if (i == static_cast<int>(vram_zone_id::WEIGHT)) {
            z.allocator = nullptr;  // routed through weight_chunk_allocators_
            continue;
        }
        if (z.size > 0) {
            z.allocator = std::make_unique<tlsf_allocator>(z.size);
        }
    }

    return true;
}

void * unified_cache::zone_alloc(vram_zone_id zone, size_t size, size_t align) {
    if (!arena_base_ || size == 0) {
        return nullptr;
    }

    const bool profile_active = arena_pp_profile_active();
    const auto t0             = profile_active ? arena_profile_clock::now() : arena_profile_clock::time_point{};

    auto & z = arena_zones_[static_cast<int>(zone)];

    if (align == 0 || (align & (align - 1)) != 0) {
        align = 256;
    }

    // N-chunk WEIGHT path: try each per-chunk allocator in turn.  A single
    // weight allocation must not cross a chunk boundary because each chunk
    // is its own sycl::malloc_device USM allocation; first-fit walking
    // gives any one chunk's TLSF the chance to satisfy the request.
    //
    // Single mutex across the per-chunk walk: intentional simplicity for the
    // dormant N-chunk path.  Per-chunk mutexes would scale better but pay no
    // dividend until a config actually triggers N>=2.
    if (zone == vram_zone_id::WEIGHT && !weight_chunk_allocators_.empty()) {
        std::lock_guard<std::mutex> lock(z.alloc_mutex);
        for (auto & wca : weight_chunk_allocators_) {
            const size_t before = wca.allocator->used();
            const size_t offset = wca.allocator->allocate(size, align);
            if (offset == SIZE_MAX) {
                continue;
            }
            // Incremental aggregate: alloc/free both update z.used under
            // z.alloc_mutex so there's no race window.  Use the TLSF
            // before/after delta (rather than `size`) so we account for
            // alignment + block-header overhead consistently with what
            // tlsf_allocator::used() reports.
            const size_t after      = wca.allocator->used();
            const size_t delta      = after - before;
            const size_t total_used = z.used.fetch_add(delta, std::memory_order_relaxed) + delta;
            if (profile_active) {
                const size_t idx = static_cast<size_t>(zone);
                t_arena_pp_profile.zone_alloc_calls[idx]++;
                t_arena_pp_profile.zone_alloc_bytes[idx] += size;
                t_arena_pp_profile.zone_alloc_us[idx] += arena_profile_elapsed_us(t0);
                t_arena_pp_profile.zone_used_peak[idx] = std::max(t_arena_pp_profile.zone_used_peak[idx], total_used);
                t_arena_pp_profile.zone_largest_free_min[idx] =
                    std::min(t_arena_pp_profile.zone_largest_free_min[idx], wca.allocator->largest_free_block());
            }
            return offset_to_ptr(wca.logical_start + offset);
        }
        if (profile_active) {
            const size_t idx = static_cast<size_t>(zone);
            t_arena_pp_profile.zone_alloc_calls[idx]++;
            t_arena_pp_profile.zone_alloc_failures[idx]++;
            t_arena_pp_profile.zone_alloc_bytes[idx] += size;
            t_arena_pp_profile.zone_alloc_us[idx] += arena_profile_elapsed_us(t0);
        }
        return nullptr;
    }

    tlsf_allocator * alloc = z.allocator.get();

    // Shared zone delegation (single-chunk mode): WEIGHT delegates to KV allocator.
    if (!alloc && zone == vram_zone_id::WEIGHT) {
        alloc = arena_zones_[static_cast<int>(vram_zone_id::KV)].allocator.get();
    }
    if (!alloc) {
        return nullptr;
    }

    // Use KV zone's mutex for serialization in shared-zone mode.
    std::mutex &                mtx = (!z.allocator && zone == vram_zone_id::WEIGHT) ?
                                          arena_zones_[static_cast<int>(vram_zone_id::KV)].alloc_mutex :
                                          z.alloc_mutex;
    std::lock_guard<std::mutex> lock(mtx);

    // TLSF returns OFFSET into the managed region; convert to device pointer.
    size_t offset = alloc->allocate(size, align);
    if (offset != SIZE_MAX) {
        const size_t current_used         = alloc->used();
        const size_t current_largest_free = alloc->largest_free_block();
        z.used.store(current_used, std::memory_order_relaxed);
        if (profile_active) {
            const size_t idx = static_cast<size_t>(zone);
            t_arena_pp_profile.zone_alloc_calls[idx]++;
            t_arena_pp_profile.zone_alloc_bytes[idx] += size;
            t_arena_pp_profile.zone_alloc_us[idx] += arena_profile_elapsed_us(t0);
            t_arena_pp_profile.zone_used_peak[idx] = std::max(t_arena_pp_profile.zone_used_peak[idx], current_used);
            t_arena_pp_profile.zone_largest_free_min[idx] =
                std::min(t_arena_pp_profile.zone_largest_free_min[idx], current_largest_free);
        }
        return offset_to_ptr(z.start + offset);
    }
    if (profile_active) {
        const size_t idx = static_cast<size_t>(zone);
        t_arena_pp_profile.zone_alloc_calls[idx]++;
        t_arena_pp_profile.zone_alloc_failures[idx]++;
        t_arena_pp_profile.zone_alloc_bytes[idx] += size;
        t_arena_pp_profile.zone_alloc_us[idx] += arena_profile_elapsed_us(t0);
        t_arena_pp_profile.zone_largest_free_min[idx] =
            std::min(t_arena_pp_profile.zone_largest_free_min[idx], alloc->largest_free_block());
    }
    return nullptr;
}

void unified_cache::zone_reset(vram_zone_id zone) {
    const bool                  profile_active = arena_pp_profile_active();
    const auto                  t0 = profile_active ? arena_profile_clock::now() : arena_profile_clock::time_point{};
    auto &                      z  = arena_zones_[static_cast<int>(zone)];
    std::lock_guard<std::mutex> lock(z.alloc_mutex);

    // Purge registry entries whose pointer falls within this zone's address
    // range.  Without this, TLSF reset recycles addresses while the registry
    // still maps them, causing duplicate-pointer rejection on the next
    // unified_alloc() that gets a recycled address.
    //
    // KV/ONEDNN/RUNTIME/SCRATCH all live in a SINGLE physical chunk (the tail
    // chunk in N-chunk mode, or the only chunk in single-chunk mode), so the
    // zone's address range is one contiguous span starting at offset_to_ptr(z.start).
    // WEIGHT in N-chunk mode is the only zone that spans multiple physical
    // chunks; zone_reset is never called on WEIGHT in production paths but
    // handle it correctly anyway.
    if (arena_base_ && z.size > 0) {
        std::lock_guard<std::mutex> reg_lock(g_runtime_alloc_mutex);
        if (zone == vram_zone_id::WEIGHT && !weight_chunk_allocators_.empty()) {
            // Walk each weight chunk's physical range.
            for (const auto & wca : weight_chunk_allocators_) {
                const auto      cbase = reinterpret_cast<uintptr_t>(arena_chunks_[wca.chunk_idx].ptr);
                const uintptr_t lo    = cbase;
                const uintptr_t hi    = cbase + arena_chunks_[wca.chunk_idx].size;
                for (auto it = g_runtime_alloc_registry.begin(); it != g_runtime_alloc_registry.end();) {
                    const uintptr_t p = reinterpret_cast<uintptr_t>(it->first);
                    if (p >= lo && p < hi) {
                        it = g_runtime_alloc_registry.erase(it);
                    } else {
                        ++it;
                    }
                }
            }
        } else {
            const auto      base    = reinterpret_cast<uintptr_t>(offset_to_ptr(z.start));
            const uintptr_t zone_lo = base;
            const uintptr_t zone_hi = base + z.size;
            for (auto it = g_runtime_alloc_registry.begin(); it != g_runtime_alloc_registry.end();) {
                const uintptr_t p = reinterpret_cast<uintptr_t>(it->first);
                if (p >= zone_lo && p < zone_hi) {
                    it = g_runtime_alloc_registry.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }

    if (zone == vram_zone_id::WEIGHT && !weight_chunk_allocators_.empty()) {
        for (auto & wca : weight_chunk_allocators_) {
            wca.allocator->reset();
        }
    } else if (z.allocator) {
        z.allocator->reset();
    }
    z.used.store(0, std::memory_order_relaxed);
    if (profile_active) {
        const size_t idx = static_cast<size_t>(zone);
        t_arena_pp_profile.zone_reset_calls[idx]++;
        t_arena_pp_profile.zone_reset_us[idx] += arena_profile_elapsed_us(t0);
    }
}

void unified_cache::zone_free(vram_zone_id zone, void * ptr) {
    if (!ptr || !arena_base_) {
        return;
    }

    const bool profile_active = arena_pp_profile_active();
    const auto t0             = profile_active ? arena_profile_clock::now() : arena_profile_clock::time_point{};

    auto & z = arena_zones_[static_cast<int>(zone)];

    // N-chunk WEIGHT path: route the free to the per-chunk allocator that owns ptr.
    if (zone == vram_zone_id::WEIGHT && !weight_chunk_allocators_.empty()) {
        const int chunk_idx = arena_find_chunk(ptr);
        if (chunk_idx < 0) {
            return;
        }
        weight_chunk_alloc * target = nullptr;
        for (auto & wca : weight_chunk_allocators_) {
            if (wca.chunk_idx == chunk_idx) {
                target = &wca;
                break;
            }
        }
        if (!target) {
            return;
        }
        std::lock_guard<std::mutex> lock(z.alloc_mutex);
        const auto                  chunk_base = reinterpret_cast<uintptr_t>(arena_chunks_[chunk_idx].ptr);
        const auto                  p          = reinterpret_cast<uintptr_t>(ptr);
        const size_t                chunk_off  = static_cast<size_t>(p - chunk_base);
        const size_t                before     = target->allocator->used();
        target->allocator->free(chunk_off);
        // Incremental aggregate: see matching fetch_add comment in zone_alloc.
        const size_t after = target->allocator->used();
        const size_t delta = before - after;
        z.used.fetch_sub(delta, std::memory_order_relaxed);
        if (profile_active) {
            const size_t idx = static_cast<size_t>(zone);
            t_arena_pp_profile.zone_free_calls[idx]++;
            t_arena_pp_profile.zone_free_us[idx] += arena_profile_elapsed_us(t0);
        }
        return;
    }

    tlsf_allocator * alloc = z.allocator.get();

    // Shared zone delegation (single-chunk mode).
    if (!alloc && zone == vram_zone_id::WEIGHT) {
        auto & kv = arena_zones_[static_cast<int>(vram_zone_id::KV)];
        alloc     = kv.allocator.get();
        if (!alloc) {
            return;
        }
        // Convert device pointer to zone-relative offset for TLSF.
        size_t arena_offset = ptr_to_offset(ptr);
        if (arena_offset == SIZE_MAX || arena_offset < kv.start || arena_offset >= kv.start + kv.size) {
            return;
        }
        size_t                      zone_offset = arena_offset - kv.start;
        std::lock_guard<std::mutex> lock(kv.alloc_mutex);
        alloc->free(zone_offset);
        kv.used.store(alloc->used(), std::memory_order_relaxed);
        if (profile_active) {
            const size_t idx = static_cast<size_t>(zone);
            t_arena_pp_profile.zone_free_calls[idx]++;
            t_arena_pp_profile.zone_free_us[idx] += arena_profile_elapsed_us(t0);
        }
        return;
    }
    if (!alloc) {
        return;
    }

    // Convert device pointer to zone-relative offset for TLSF.
    size_t arena_offset = ptr_to_offset(ptr);
    if (arena_offset == SIZE_MAX || arena_offset < z.start || arena_offset >= z.start + z.size) {
        return;
    }
    size_t zone_offset = arena_offset - z.start;

    std::lock_guard<std::mutex> lock(z.alloc_mutex);
    alloc->free(zone_offset);
    z.used.store(alloc->used(), std::memory_order_relaxed);
    if (profile_active) {
        const size_t idx = static_cast<size_t>(zone);
        t_arena_pp_profile.zone_free_calls[idx]++;
        t_arena_pp_profile.zone_free_us[idx] += arena_profile_elapsed_us(t0);
    }
}

bool unified_cache::vram_owns(const void * ptr) const {
    if (!arena_base_ || !ptr) {
        return false;
    }
    for (const auto & c : arena_chunks_) {
        auto base = reinterpret_cast<uintptr_t>(c.ptr);
        auto p    = reinterpret_cast<uintptr_t>(ptr);
        if (p >= base && p < base + c.size) {
            return true;
        }
    }
    return false;
}

bool unified_cache::zone_owns(vram_zone_id zone, const void * ptr) const {
    if (!arena_base_ || !ptr) {
        return false;
    }
    size_t off = ptr_to_offset(ptr);
    if (off == SIZE_MAX) {
        return false;
    }
    const auto & z = arena_zones_[static_cast<int>(zone)];
    return off >= z.start && off < z.start + z.size;
}

size_t unified_cache::ptr_to_offset(const void * ptr) const {
    if (!ptr) {
        return SIZE_MAX;
    }
    size_t logical_base = 0;
    for (const auto & c : arena_chunks_) {
        auto base = reinterpret_cast<uintptr_t>(c.ptr);
        auto p    = reinterpret_cast<uintptr_t>(ptr);
        if (p >= base && p < base + c.size) {
            return logical_base + static_cast<size_t>(p - base);
        }
        logical_base += c.size;
    }
    return SIZE_MAX;
}

void * unified_cache::offset_to_ptr(size_t offset) const {
    size_t logical_base = 0;
    for (const auto & c : arena_chunks_) {
        if (offset < logical_base + c.size) {
            return static_cast<uint8_t *>(c.ptr) + (offset - logical_base);
        }
        logical_base += c.size;
    }
    return nullptr;
}

size_t unified_cache::zone_capacity(vram_zone_id zone) const {
    return arena_zones_[static_cast<int>(zone)].size;
}

size_t unified_cache::zone_used(vram_zone_id zone) const {
    return arena_zones_[static_cast<int>(zone)].used.load(std::memory_order_relaxed);
}

size_t unified_cache::zone_available(vram_zone_id zone) const {
    const auto & z = arena_zones_[static_cast<int>(zone)];

    // In single-chunk mode, WEIGHT delegates to KV's allocator.
    if (arena_chunks_.size() == 1 && zone == vram_zone_id::WEIGHT) {
        const auto & kv = arena_zones_[static_cast<int>(vram_zone_id::KV)];
        if (kv.allocator) {
            return kv.allocator->available();
        }
        return 0;
    }

    // N-chunk WEIGHT: sum across per-chunk allocators.
    if (zone == vram_zone_id::WEIGHT && !weight_chunk_allocators_.empty()) {
        size_t total = 0;
        for (const auto & wca : weight_chunk_allocators_) {
            total += wca.allocator->available();
        }
        return total;
    }

    // Zones with their own allocator.
    if (z.allocator) {
        return z.allocator->available();
    }

    // Fallback for zones without allocator (shouldn't happen after arena_reserve).
    size_t used = z.used.load(std::memory_order_relaxed);
    return z.size > used ? z.size - used : 0;
}

size_t unified_cache::zone_largest_free(vram_zone_id zone) const {
    const auto & z = arena_zones_[static_cast<int>(zone)];

    if (arena_chunks_.size() == 1 && zone == vram_zone_id::WEIGHT) {
        const auto & kv = arena_zones_[static_cast<int>(vram_zone_id::KV)];
        if (kv.allocator) {
            return kv.allocator->largest_free_block();
        }
        return 0;
    }

    // N-chunk WEIGHT: max across per-chunk allocators (a single alloc cannot
    // span chunks, so the largest fittable size is the largest single chunk's
    // largest free block).
    if (zone == vram_zone_id::WEIGHT && !weight_chunk_allocators_.empty()) {
        size_t best = 0;
        for (const auto & wca : weight_chunk_allocators_) {
            best = std::max(best, wca.allocator->largest_free_block());
        }
        return best;
    }

    if (z.allocator) {
        return z.allocator->largest_free_block();
    }

    return zone_available(zone);
}

void unified_cache::arena_destroy() {
    if (!arena_queue_) {
        return;
    }
    for (auto & c : arena_chunks_) {
        if (c.ptr) {
            // llama.cpp-dyhdl: wait for chunk leases to drain before sycl::free.
            // A lease outstanding at destruction means a downstream caller holds
            // a raw pointer into this chunk — assert-with-detail after 5 s rather
            // than SEGV silently after free.
            const auto deadline    = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            uint32_t   outstanding = c.lease_count.load();
            if (outstanding > 0) {
                GGML_LOG_WARN(
                    "[VRAM-ARENA] chunk %p size=%.1f MB has %u outstanding lease(s) at arena_destroy, waiting ≤5s\n",
                    c.ptr, c.size / (1024.0 * 1024.0), outstanding);
                while ((outstanding = c.lease_count.load()) > 0) {
                    if (std::chrono::steady_clock::now() > deadline) {
                        GGML_LOG_ERROR(
                            "[VRAM-ARENA] chunk %p size=%.1f MB still has %u outstanding lease(s) after 5s — "
                            "aborting (llama.cpp-dyhdl)\n",
                            c.ptr, c.size / (1024.0 * 1024.0), outstanding);
                        GGML_ASSERT(false && "VRAM arena chunk freed while leases outstanding (dyhdl timeout)");
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
            try {
                sycl::free(c.ptr, *arena_queue_);
            } catch (...) {
            }
        }
    }
    weight_chunk_allocators_.clear();
    arena_chunks_.clear();
    arena_base_ = nullptr;
    arena_size_ = 0;
    for (int i = 0; i < static_cast<int>(vram_zone_id::COUNT); i++) {
        auto &                      z = arena_zones_[i];
        std::lock_guard<std::mutex> lock(z.alloc_mutex);
        z.allocator.reset();  // unique_ptr::reset — destroys the TLSF allocator
        z.start = 0;
        z.size  = 0;
        z.used.store(0, std::memory_order_relaxed);
    }
}

// === Arena chunk lease API (llama.cpp-dyhdl) ===

int unified_cache::arena_find_chunk(const void * ptr) const {
    if (!ptr) {
        return -1;
    }
    const char * p = static_cast<const char *>(ptr);
    for (size_t i = 0; i < arena_chunks_.size(); ++i) {
        const char * base = static_cast<const char *>(arena_chunks_[i].ptr);
        if (base && p >= base && p < base + arena_chunks_[i].size) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

int unified_cache::arena_acquire_chunk_lease(const void * ptr) {
    const int idx = arena_find_chunk(ptr);
    if (idx < 0) {
        return -1;
    }
    arena_chunks_[idx].lease_count.fetch_add(1);
    return idx;
}

void unified_cache::arena_release_chunk_lease(int chunk_idx) {
    if (chunk_idx < 0 || chunk_idx >= static_cast<int>(arena_chunks_.size())) {
        return;
    }
    const uint32_t prev = arena_chunks_[chunk_idx].lease_count.fetch_sub(1);
    if (prev == 0) {
        GGML_LOG_ERROR("[VRAM-ARENA] arena_release_chunk_lease: underflow on chunk idx=%d\n", chunk_idx);
        arena_chunks_[chunk_idx].lease_count.fetch_add(1);
    }
}

bool unified_cache::arena_chunk_has_leases(int chunk_idx) const {
    if (chunk_idx < 0 || chunk_idx >= static_cast<int>(arena_chunks_.size())) {
        return false;
    }
    return arena_chunks_[chunk_idx].lease_count.load() > 0;
}

// llama.cpp-dyhdl: host pinned-pool chunk lease pass-through.
uint64_t unified_cache::host_acquire_chunk_lease(const void * ptr) {
    if (!host_arena_) {
        return pinned_chunk_pool::INVALID_CHUNK_HANDLE;
    }
    return host_arena_->acquire_chunk_lease(ptr);
}

void unified_cache::host_release_chunk_lease(uint64_t handle) {
    if (!host_arena_ || handle == pinned_chunk_pool::INVALID_CHUNK_HANDLE) {
        return;
    }
    host_arena_->release_chunk_lease(handle);
}

void unified_cache::arena_abandon() {
    // Null everything without calling sycl::free — used during shutdown
    // when the SYCL context is already invalid.
    weight_chunk_allocators_.clear();
    arena_chunks_.clear();
    arena_base_ = nullptr;
    arena_size_ = 0;
    for (int i = 0; i < static_cast<int>(vram_zone_id::COUNT); i++) {
        auto & z = arena_zones_[i];
        z.allocator.reset();
        z.start = 0;
        z.size  = 0;
        z.used.store(0, std::memory_order_relaxed);
    }
}

// === P4: Priority-based static placement planner ===

// Map tensor_usage + name to placement priority.
// For MoE experts, name sub-classification distinguishes gate/down/up.
static placement_priority tensor_to_placement_priority(tensor_usage usage, const char * name) {
    switch (usage) {
        case tensor_usage::NORM:
        case tensor_usage::EMBEDDING:
            return placement_priority::NORM_EMBED;
        case tensor_usage::ATTENTION_WEIGHT:
            return placement_priority::ATTENTION;
        case tensor_usage::FFN_WEIGHT:
            return placement_priority::FFN;
        case tensor_usage::MOE_GATE:
            return placement_priority::MOE_GATE;
        case tensor_usage::MOE_EXPERT_WEIGHT:
            // Unsloth-informed sub-classification: down > up > gate
            if (name && strstr(name, "ffn_down_exps")) {
                return placement_priority::MOE_DOWN;
            }
            if (name && strstr(name, "ffn_up_exps")) {
                return placement_priority::MOE_UP;
            }
            // ffn_gate_exps or unknown expert pattern
            return placement_priority::MOE_GATE_PROJ;
        case tensor_usage::MOE_INTERMEDIATE:
            return placement_priority::NORM_EMBED;  // Small, treat like norms
        case tensor_usage::UNKNOWN:
        default:
            // Unknown tensors get FFN priority (middle of the pack)
            return placement_priority::FFN;
    }
}

// Extract layer number from tensor name (e.g. "blk.5.attn_q" -> 5).
static int p4_extract_layer_id(const char * name) {
    if (!name) {
        return -1;
    }
    const char * blk = strstr(name, "blk.");
    if (!blk) {
        return -1;
    }
    return std::atoi(blk + 4);
}

static int count_observed_moe_layers(const placement_plan & plan) {
    std::unordered_set<int> layers;
    for (const auto & entry : plan.entries) {
        if (entry.expert_id >= 0 && entry.layer_id >= 0) {
            layers.insert(entry.layer_id);
        }
    }
    return static_cast<int>(layers.size());
}

static const char * placement_target_name(int target, char * buf, size_t buf_size) {
    if (target < 0) {
        return "HOST";
    }
    std::snprintf(buf, buf_size, "device%d", target);
    return buf;
}

struct moe_triplet_group {
    int                 layer_id  = -1;
    int                 expert_id = -1;
    std::vector<size_t> indices;
    size_t              bytes           = 0;
    uint8_t             role_mask       = 0;
    int                 popularity_rank = -1;
};

static uint8_t moe_role_bit(expert_tensor_role role) {
    switch (role) {
        case expert_tensor_role::GATE:
            return 1u << 0;
        case expert_tensor_role::UP:
            return 1u << 1;
        case expert_tensor_role::DOWN:
            return 1u << 2;
        case expert_tensor_role::UNKNOWN:
        default:
            return 0;
    }
}

static bool moe_triplet_complete(const moe_triplet_group & group) {
    constexpr uint8_t all_roles = (1u << 0) | (1u << 1) | (1u << 2);
    return (group.role_mask & all_roles) == all_roles;
}

static std::vector<moe_triplet_group> build_moe_triplet_groups(const placement_plan &      plan,
                                                               const std::vector<size_t> & moe_indices,
                                                               const char **               hotness_source) {
    std::map<std::pair<int, int>, size_t> group_index;
    std::vector<moe_triplet_group>        groups;
    groups.reserve(moe_indices.size() / 3 + 1);

    for (size_t idx : moe_indices) {
        const auto & entry = plan.entries[idx];
        const auto   key   = std::make_pair(entry.layer_id, entry.expert_id);
        auto         it    = group_index.find(key);
        if (it == group_index.end()) {
            moe_triplet_group group;
            group.layer_id  = entry.layer_id;
            group.expert_id = entry.expert_id;
            groups.push_back(std::move(group));
            const size_t new_idx = groups.size() - 1;
            group_index.emplace(key, new_idx);
            it = group_index.find(key);
        }

        auto & group = groups[it->second];
        group.indices.push_back(idx);
        group.bytes += entry.dst_size;
        group.role_mask |= moe_role_bit(entry.expert_role);
    }

    bool has_ranked_groups = false;
    if (is_expert_popularity_initialized()) {
        for (auto & group : groups) {
            group.popularity_rank = get_expert_popularity_rank(group.layer_id, group.expert_id);
            has_ranked_groups     = has_ranked_groups || group.popularity_rank >= 0;
        }
    }

    if (hotness_source != nullptr) {
        *hotness_source = has_ranked_groups ? "routing-popularity" : "static(layer,expert)";
    }

    // When routing/popularity data is available, place hot routed triplets
    // first. Otherwise use the layer-first fallback: it preserves fully
    // device-resident early MoE layers under pressure and avoids making every
    // layer mixed-resident based on an unproven expert-ID prior.
    std::sort(groups.begin(), groups.end(),
              [has_ranked_groups](const moe_triplet_group & a, const moe_triplet_group & b) {
                  if (has_ranked_groups) {
                      const int a_rank = a.popularity_rank >= 0 ? a.popularity_rank : std::numeric_limits<int>::max();
                      const int b_rank = b.popularity_rank >= 0 ? b.popularity_rank : std::numeric_limits<int>::max();
                      if (a_rank != b_rank) {
                          return a_rank < b_rank;
                      }
                  }
                  if (a.layer_id != b.layer_id) {
                      return a.layer_id < b.layer_id;
                  }
                  if (a.expert_id != b.expert_id) {
                      return a.expert_id < b.expert_id;
                  }
                  return a.bytes > b.bytes;
              });
    return groups;
}

struct moe_triplet_pack_stats {
    size_t       groups            = 0;
    size_t       complete_groups   = 0;
    size_t       incomplete_groups = 0;
    size_t       device_groups     = 0;
    size_t       host_groups       = 0;
    size_t       device_entries    = 0;
    size_t       host_entries      = 0;
    size_t       device_bytes      = 0;
    size_t       host_bytes        = 0;
    const char * hotness           = "static(layer,expert)";
};

static void log_moe_triplet_pack_stats(const char * tag, const moe_triplet_pack_stats & stats, size_t remaining) {
    if (stats.groups == 0) {
        return;
    }
    GGML_LOG_INFO(
        "[%s] triplet packing: groups=%zu complete=%zu incomplete=%zu device_groups=%zu host_groups=%zu "
        "device_entries=%zu (%.1f MB) host_entries=%zu (%.1f MB) remaining=%.1f MB "
        "hotness=%s\n",
        tag, stats.groups, stats.complete_groups, stats.incomplete_groups, stats.device_groups, stats.host_groups,
        stats.device_entries, stats.device_bytes / (1024.0 * 1024.0), stats.host_entries,
        stats.host_bytes / (1024.0 * 1024.0), remaining / (1024.0 * 1024.0), stats.hotness);
}

static void log_moe_device_policy(const char * tag,
                                  int          device_id,
                                  size_t       vram_budget,
                                  size_t       weight_budget,
                                  size_t       arena_total,
                                  size_t       arena_scratch,
                                  size_t       arena_runtime,
                                  size_t       arena_onednn,
                                  int          n_experts) {
    if (n_experts <= 0 || !get_existing_cache_for_device(device_id)) {
        return;
    }
    if (device_id < 0 || device_id >= ggml_sycl_info().device_count) {
        return;
    }

    const auto & dinfo  = ggml_sycl_info().devices[device_id];
    const auto   policy = ggml_sycl_make_moe_device_policy(
        dinfo.xmx_caps, device_id, dinfo.total_vram, dinfo.free_vram_at_init, dinfo.max_alloc_size,
        dinfo.safe_max_alloc_size, vram_budget, weight_budget, arena_total, arena_scratch, arena_runtime, arena_onednn,
        /*in_dim=*/0, /*out_dim=*/0, n_experts, /*device_resident=*/true, /*tiled_kernel_validated=*/false);

    GGML_LOG_INFO(
        "[%s] device=%d total=%.1f MB free_init=%.1f MB plan_budget=%.1f MB weight_budget=%.1f MB "
        "max_alloc=%.1f MB safe_alloc=%.1f MB arena=%.1f MB zones(scratch/runtime/onednn)=%.1f/%.1f/%.1f MB "
        "cu=%u wg=%zu sg_pref=%zu sg_max=%zu slm=%zu KB usm(d/s/h)=%d/%d/%d fp16(type/xmx)=%d/%d "
        "xmx_int8=%d tile=%zux%zux%zu tiles=%dx%d executor=%s reason=%s\n",
        tag, device_id, dinfo.total_vram / (1024.0 * 1024.0), dinfo.free_vram_at_init / (1024.0 * 1024.0),
        vram_budget / (1024.0 * 1024.0), weight_budget / (1024.0 * 1024.0), dinfo.max_alloc_size / (1024.0 * 1024.0),
        dinfo.safe_max_alloc_size / (1024.0 * 1024.0), arena_total / (1024.0 * 1024.0),
        arena_scratch / (1024.0 * 1024.0), arena_runtime / (1024.0 * 1024.0), arena_onednn / (1024.0 * 1024.0),
        policy.caps.compute_units, policy.caps.max_work_group_size, policy.caps.preferred_sub_group_size,
        policy.caps.max_sub_group_size, policy.caps.slm_size / 1024, policy.caps.supports_usm_device ? 1 : 0,
        policy.caps.supports_usm_shared ? 1 : 0, policy.caps.supports_usm_host ? 1 : 0,
        policy.caps.supports_fp16_type ? 1 : 0, policy.caps.supports_fp16 ? 1 : 0, policy.xmx_int8_candidate ? 1 : 0,
        policy.caps.M, policy.caps.N, policy.caps.K, policy.caps.optimal_tiles_m, policy.caps.optimal_tiles_n,
        policy.device_executor, policy.executor_reason);
}

static void log_moe_placement_diagnostics(const char * tag, const placement_plan & plan, int n_experts) {
    if (n_experts <= 0) {
        return;
    }

    const int observed_layers = count_observed_moe_layers(plan);
    if (observed_layers <= 0) {
        return;
    }

    const auto summary = plan.summarize_expert_placements(observed_layers, n_experts);
    GGML_LOG_INFO("[%s] expected=%zu planned=%zu unclassified=%zu duplicates=%zu missing=%zu\n", tag, summary.expected,
                  summary.planned, summary.unclassified, summary.duplicates, summary.missing);
    GGML_LOG_INFO("[%s] roles: gate=%zu up=%zu down=%zu unknown=%zu\n", tag,
                  summary.role_counts[static_cast<size_t>(expert_tensor_role::GATE)],
                  summary.role_counts[static_cast<size_t>(expert_tensor_role::UP)],
                  summary.role_counts[static_cast<size_t>(expert_tensor_role::DOWN)],
                  summary.role_counts[static_cast<size_t>(expert_tensor_role::UNKNOWN)]);

    std::vector<int> targets;
    targets.reserve(summary.target_counts.size());
    for (const auto & [target, count] : summary.target_counts) {
        GGML_UNUSED(count);
        targets.push_back(target);
    }
    std::sort(targets.begin(), targets.end());
    for (int target : targets) {
        char target_buf[32];
        GGML_LOG_INFO("[%s] target %s: %zu expert tensors\n", tag,
                      placement_target_name(target, target_buf, sizeof(target_buf)), summary.target_counts.at(target));
    }

    if (observed_layers > 8 && n_experts > 30) {
        for (expert_tensor_role role : { expert_tensor_role::GATE, expert_tensor_role::UP, expert_tensor_role::DOWN }) {
            const auto placement = plan.lookup_expert_placement(8, 30, role);
            char       target_buf[32];
            GGML_LOG_INFO("[%s] sample blk.8 expert 30 %s: %s\n", tag, expert_tensor_role_name(role),
                          placement.found() ?
                              placement_target_name(placement.target_device, target_buf, sizeof(target_buf)) :
                              "MISSING");
        }
    }
}

static void populate_host_zone_sizing(placement_plan &                                    plan,
                                      const std::vector<std::pair<std::string, size_t>> & tensor_inventory,
                                      int                                                 n_experts,
                                      int                                                 n_expert_used) {
    constexpr size_t k_min_zone_bytes        = 64ull * 1024ull * 1024ull;
    constexpr size_t k_tp_staging_headroom   = 16ull * 1024ull * 1024ull;
    constexpr size_t k_scratch_headroom      = 32ull * 1024ull * 1024ull;
    constexpr double k_weight_headroom_ratio = 1.2;

    plan.max_tensor_bytes = 0;
    for (const auto & item : tensor_inventory) {
        plan.max_tensor_bytes = std::max(plan.max_tensor_bytes, item.second);
    }

    // --- Inference memory category sizing ---
    // Computed first so zone sizing below can include these costs.

    // 1. oneDNN reorder: one temp buffer sized to the largest weight tensor, reused per layer.
    plan.onednn_reorder_bytes = plan.max_tensor_bytes;

    // 2. MoE Q8_1 workspace: n_expert_used activation rows quantized to Q8_1 for batched dispatch.
    //    Coarse heuristic: estimate Q8_1 workspace from weight tensor size.
    //    The actual Q8_1 row size depends on the activation K dimension (ne10),
    //    not weight bytes. This over-estimates because quantized weight bytes
    //    exceed the raw K dimension, making the sizing conservative.
    //    1.1x headroom covers QK8_1 alignment padding.  Zero for dense models.
    if (n_experts > 0 && n_expert_used > 0) {
        const size_t expert_ffn_bytes = plan.max_tensor_bytes / static_cast<size_t>(n_experts);
        const size_t q8_1_row_bytes   = static_cast<size_t>(static_cast<double>(expert_ffn_bytes) * 1.1);
        plan.moe_q8_workspace_bytes   = static_cast<size_t>(n_expert_used) * q8_1_row_bytes;
    } else {
        plan.moe_q8_workspace_bytes = 0;
    }

    // 3. Expert bias D2H + MoE routing buffers: computed from MoE layer count.
    //    Count MoE layers from plan entries (expert_id >= 0) to avoid needing n_layer separately.
    if (n_experts > 0) {
        int max_layer_id = 0;
        for (const auto & entry : plan.entries) {
            if (entry.expert_id >= 0 && entry.layer_id > max_layer_id) {
                max_layer_id = entry.layer_id;
            }
        }
        const int n_moe_layers = max_layer_id + 1;

        // 3a. Expert bias D2H: float32 bias per expert across all MoE layers, staged to host.
        plan.expert_bias_bytes = static_cast<size_t>(n_experts) * sizeof(float) * static_cast<size_t>(n_moe_layers);

        // 3b. MoE routing IDs: per-batch expert assignment control buffer.
        //     n_expert rows × max_batch_tokens cols, each entry is int32_t.
        //     planner_n_ctx is the maximum context length = conservative max batch size.
        //     These are CPU-produced and read by any participating GPU, so they
        //     are planned as shared-context host-pinned CONTROL memory rather
        //     than VRAM runtime bytes.
        const size_t max_batch_tokens = static_cast<size_t>(std::max<uint32_t>(plan.planner_n_ctx, 1));
        plan.moe_routing_ids_bytes    = static_cast<size_t>(n_experts) * sizeof(int32_t) * max_batch_tokens;

        // 3c. MoE expert pointer tables: one void* per expert per MoE layer, pre-allocated
        //     for graph recording (fixed addresses required).
        //     MAX_EXPERTS (256) is used for the pre-allocation size regardless of n_experts.
        constexpr int k_max_experts_prealloc = 256;  // moe_tile_mapping_state::MAX_EXPERTS
        plan.moe_expert_ptrs_bytes =
            static_cast<size_t>(k_max_experts_prealloc) * sizeof(void *) * static_cast<size_t>(n_moe_layers);

        // 3d. PP CPU expert act/out staging. These are scoped EXPERT_STAGING
        //     HOST_COMPUTE allocations routed to host SCRATCH, so the zone only
        //     needs the peak live act/out pair, not graph-wide accumulation.
        //     Tensor inventory does not carry K/N, so use the largest MoE
        //     tensor as a shape-free conservative proxy.
        size_t max_moe_tensor_bytes = 0;
        for (const auto & [name, sz] : tensor_inventory) {
            if (expert_tensor_role_from_tensor_name(name.c_str()) != expert_tensor_role::UNKNOWN) {
                max_moe_tensor_bytes = std::max(max_moe_tensor_bytes, sz);
            }
        }
        plan.moe_cpu_expert_staging_bytes = max_moe_tensor_bytes;

        GGML_LOG_INFO(
            "[SYCL-PLAN] MoE routing buffer sizing: n_experts=%d n_moe_layers=%d "
            "max_batch=%zu control_ids=%.2f MB host-pinned, expert_ptrs=%.2f MB VRAM, cpu_staging=%.2f MB\n",
            n_experts, n_moe_layers, max_batch_tokens, plan.moe_routing_ids_bytes / (1024.0 * 1024.0),
            plan.moe_expert_ptrs_bytes / (1024.0 * 1024.0), plan.moe_cpu_expert_staging_bytes / (1024.0 * 1024.0));
    } else {
        plan.expert_bias_bytes            = 0;
        plan.moe_routing_ids_bytes        = 0;
        plan.moe_expert_ptrs_bytes        = 0;
        plan.moe_cpu_expert_staging_bytes = 0;
    }

    // 4. CPU quantization temp buffers: 3 pre-allocated slots (cpu_dispatch_buffers) each
    //    sized conservatively to max_tensor_bytes to cover any weight's row/col dimensions.
    constexpr size_t k_cpu_quant_slots = 3;
    plan.cpu_quant_buffer_bytes        = k_cpu_quant_slots * plan.max_tensor_bytes;

    // 5. Graph metadata: layer classification vectors + MoE routing tables.
    //    Fixed 4 MB base + per-expert int entries per layer.
    constexpr size_t k_graph_metadata_base = 4ull * 1024ull * 1024ull;
    plan.graph_metadata_bytes = k_graph_metadata_base + static_cast<size_t>(std::max(n_experts, 0)) * 32 * sizeof(int);

    // 6. Tensor Parallelism buffer estimates (only when TP is active with >1 device).
    //    TP FFN compute buffers live on secondary device VRAM; sized conservatively from
    //    max_tensor_bytes since we cannot recover exact n_embd/n_ff from quantized byte counts
    //    without knowing the quant type.  The constants below bound the Q8_1-quantized inputs,
    //    intermediate float buffers, and partial-result buffers computed by
    //    ggml_sycl_tp_ensure_ffn_buffers().
    //
    //    TP host staging is a single persistent float32 buffer holding one copy of the
    //    n_embd-wide activation row (≤16 MB for any current model at max PP batch).
    if (g_sycl_tp_config.enabled && g_sycl_tp_config.world_size > 1) {
        // Find the largest FFN and attention weight tensor byte counts.
        // FFN gate/up/down tensors: size ≈ n_embd × n_ff × bytes_per_quant_elem
        // Attention q/k/v/o tensors: size ≈ n_embd × (n_embd or n_kv_heads×head_dim) × ...
        size_t max_ffn_weight_bytes  = 0;
        size_t max_attn_weight_bytes = 0;
        for (const auto & [name, sz] : tensor_inventory) {
            const tensor_usage usage = infer_tensor_usage(name.c_str());
            if (usage == tensor_usage::FFN_WEIGHT) {
                max_ffn_weight_bytes = std::max(max_ffn_weight_bytes, sz);
            } else if (usage == tensor_usage::ATTENTION_WEIGHT) {
                max_attn_weight_bytes = std::max(max_attn_weight_bytes, sz);
            }
        }
        // TP FFN device buffers: input Q8_1 + 3× hidden float + hidden Q8_1 + partial float.
        // At max PP batch (512) these scale as O(batch × n_embd_or_n_ff).
        // Conservative bound: use 4× the float32-equivalent of the largest FFN weight.
        // Float32 equivalent ≈ weight_bytes × (32 / QK4_0) = weight_bytes × 2 (for Q4_0).
        // Multiply by 4 to cover all intermediate buffers at max batch.
        constexpr int k_tp_ffn_float_scale = 8;  // float32_equiv × buffer_count
        plan.tp_ffn_buffer_bytes           = (max_ffn_weight_bytes > 0) ? max_ffn_weight_bytes * k_tp_ffn_float_scale :
                                                                          plan.max_tensor_bytes * k_tp_ffn_float_scale;

        // TP attention device buffers: QKV projections similarly sized.
        constexpr int k_tp_attn_float_scale = 4;
        plan.tp_attn_buffer_bytes = (max_attn_weight_bytes > 0) ? max_attn_weight_bytes * k_tp_attn_float_scale :
                                                                  plan.max_tensor_bytes * k_tp_attn_float_scale;

        // TP host staging: single buffer for activation D2H/H2D copies.
        // Bounded by n_embd × max_batch × sizeof(float) ≤ k_tp_staging_headroom.
        plan.tp_staging_buffer_bytes = k_tp_staging_headroom;

        GGML_LOG_INFO(
            "[SYCL-PLAN] TP buffer sizing: world_size=%d "
            "ffn=%.1f MB attn=%.1f MB staging=%.1f MB\n",
            g_sycl_tp_config.world_size, plan.tp_ffn_buffer_bytes / (1024.0 * 1024.0),
            plan.tp_attn_buffer_bytes / (1024.0 * 1024.0), plan.tp_staging_buffer_bytes / (1024.0 * 1024.0));
    } else {
        plan.tp_ffn_buffer_bytes     = 0;
        plan.tp_attn_buffer_bytes    = 0;
        plan.tp_staging_buffer_bytes = 0;
    }

    // Aggregate TP VRAM compute buffers into a single field consumed by arena_reserve_compute()
    // for secondary TP devices. Primary device planner uses this to pre-size its RUNTIME zone
    // reservation. 0 when TP disabled (both components are 0).
    plan.tp_vram_runtime_bytes = plan.tp_ffn_buffer_bytes + plan.tp_attn_buffer_bytes;

    // Aggregate MoE device-side routing buffers into a single VRAM RUNTIME
    // reservation field. CPU-produced routing IDs are host-pinned CONTROL data
    // and are accounted in host staging, not VRAM.
    plan.moe_vram_runtime_bytes = plan.moe_expert_ptrs_bytes;

    // 7. DMA staging pool: device-resident double-buffer for host→device weight streaming.
    //    Two slices, each sized to the largest weight tensor (conservative — actual slice size
    //    defaults to 1 GB but may be smaller than max_tensor_bytes for small models).
    //    Only relevant when weight streaming is active; 0 otherwise (streaming not pre-enabled).
    //    GGML_SYCL_FORCE_STREAMING enables streaming; planner uses a conservative per-model estimate.
    {
        constexpr size_t k_dma_pipeline_depth = 2;  // Double-buffer (matches resolve_dma_defaults)
        plan.dma_staging_pool_bytes           = plan.max_tensor_bytes * k_dma_pipeline_depth;
        GGML_LOG_INFO("[SYCL-PLAN] DMA staging pool: %.1f MB (%zu x max_tensor %.1f MB)\n",
                      plan.dma_staging_pool_bytes / (1024.0 * 1024.0), k_dma_pipeline_depth,
                      plan.max_tensor_bytes / (1024.0 * 1024.0));
    }

    // 8. oneDNN scratchpad: ONEDNN zone workspace for weight reorder + activation buffer.
    //    reserve_onednn_scratch(weights_size, activations_size) sub-allocates from this zone.
    //    Conservative estimate: max_tensor_bytes for weights reorder + max_tensor_bytes for activations.
    //    The ONEDNN zone is pre-sized at 256 MB; this estimate validates the zone is adequate.
    plan.onednn_scratchpad_bytes = plan.max_tensor_bytes * 2;

    // 9. PP pipeline scratch: double-buffered FP16 weight staging for prompt-processing
    //    dequant prefetch. This is computed exactly at inventory collection time and
    //    plumbed through as an explicit planner field so the RUNTIME zone accounts for it.
    if (plan.pp_pipeline_scratch_bytes > 0) {
        GGML_LOG_INFO("[SYCL-PLAN] PP pipeline scratch: %.1f MB\n", plan.pp_pipeline_scratch_bytes / (1024.0 * 1024.0));
    }

    constexpr size_t k_onednn_zone_bytes = 256ull * 1024ull * 1024ull;
    if (plan.onednn_scratchpad_bytes > k_onednn_zone_bytes) {
        GGML_LOG_WARN(
            "[SYCL-PLAN] oneDNN scratchpad estimate (%.1f MB) exceeds zone (%.1f MB) — "
            "oneDNN may fall back to direct alloc\n",
            plan.onednn_scratchpad_bytes / (1024.0 * 1024.0), k_onednn_zone_bytes / (1024.0 * 1024.0));
    }

    // --- Host zone sizing (uses inference category fields computed above) ---

    plan.host_zone_weight_bytes =
        static_cast<size_t>(static_cast<double>(plan.weight_host_bytes) * k_weight_headroom_ratio);
    plan.host_zone_kv_bytes               = std::max<size_t>(k_min_zone_bytes, plan.kv_host_bytes);
    // Staging zone: S1 preload may keep multiple async H2D/reorder submissions
    // alive. CPU-side layout conversion can hold both the source staging copy
    // and the converted destination staging copy until the corresponding copy
    // event completes, so size this zone from the same in-flight window that
    // preload uses instead of a fixed two-buffer heuristic.
    const size_t s1_in_flight             = s1_preload_max_in_flight_limit();
    const size_t s1_reorder_guard_bytes   = 2ull * 1024ull * 1024ull;
    const size_t s1_per_inflight_bytes    = plan.max_tensor_bytes * 2 + s1_reorder_guard_bytes;
    const size_t s1_preload_staging_bytes = s1_per_inflight_bytes * s1_in_flight;
    plan.host_zone_staging_bytes =
        std::max<size_t>(k_min_zone_bytes, s1_preload_staging_bytes + k_tp_staging_headroom + plan.expert_bias_bytes +
                                               plan.moe_routing_ids_bytes + plan.tp_staging_buffer_bytes);
    GGML_LOG_INFO("[SYCL-PLAN] Host staging zone: s1_window=%zu per_inflight=%.1f MB total=%.1f MB\n", s1_in_flight,
                  s1_per_inflight_bytes / (1024.0 * 1024.0), plan.host_zone_staging_bytes / (1024.0 * 1024.0));
    // SCRATCH zone: baseline (max_tensor + headroom) plus oneDNN reorder, MoE Q8_1 workspace,
    // PP CPU expert act/out staging, MoE device routing buffers (expert pointer tables),
    // TP compute buffers, and DMA staging pool (device double-buffer for weight streaming).
    // VRAM RUNTIME fields (moe/tp/dma) are folded here as conservative zone budgeting —
    // the actual allocations live in the VRAM RUNTIME zone, but the host SCRATCH zone
    // grows proportionally to ensure the planner accounts for peak VRAM usage.
    // Actual compute buffer needs are fed back via unified_cache_grow_host_scratch_zone()
    // after the ggml scheduler is created.
    // Note: onednn_scratchpad_bytes goes to the ONEDNN zone (separate 256 MB allocation),
    // not counted here.
    plan.host_zone_scratch_bytes = std::max<size_t>(
        k_min_zone_bytes, plan.max_tensor_bytes + k_scratch_headroom + plan.onednn_reorder_bytes +
                              plan.moe_q8_workspace_bytes + plan.moe_cpu_expert_staging_bytes +
                              plan.moe_vram_runtime_bytes + plan.tp_vram_runtime_bytes + plan.dma_staging_pool_bytes);
}

// Maps the int32 flash_attn_type encoding on ggml_sycl_placement_envelope
// (mirror of the llama_flash_attn_type enum, see ggml-sycl.h declaration)
// to a printable name for the [PLACEMENT] envelope log line.  Single source
// of truth: any new enum values must extend both this switch and the field
// comment in ggml-sycl.h in lockstep.
static const char * placement_envelope_fa_name(int32_t t) {
    switch (t) {
        case -1:
            return "AUTO";
        case 0:
            return "DISABLED";
        case 1:
            return "ENABLED";
        default:
            return "UNKNOWN";
    }
}

placement_plan compute_placement_plan(const std::vector<std::pair<std::string, size_t>> & tensor_inventory,
                                      size_t                                              vram_budget,
                                      int                                                 device_id,
                                      const placement_kv_info &                           kv_info,
                                      const ggml_sycl_placement_envelope *                envelope,
                                      int                                                 n_experts) {
    if (envelope != nullptr) {
        GGML_LOG_INFO(
            "[PLACEMENT] envelope (from sycl-snapshot): n_ctx=%u n_ubatch=%u n_seq_max=%u flash_attn_type=%s\n",
            envelope->n_ctx, envelope->n_ubatch, envelope->n_seq_max,
            placement_envelope_fa_name(envelope->flash_attn_type));
    }
    placement_plan plan;
    plan.vram_budget               = vram_budget;
    plan.device_id                 = device_id;
    plan.multi_device              = false;
    plan.vram_bytes                = 0;
    plan.host_bytes                = 0;
    plan.weight_vram_bytes         = 0;
    plan.weight_host_bytes         = 0;
    plan.kv_vram_bytes             = 0;
    plan.kv_host_bytes             = 0;
    plan.kv_per_layer              = kv_info.kv_bytes_per_layer();
    plan.kv_per_swa_layer          = kv_info.kv_bytes_per_swa_layer();
    plan.swa_layer_mask            = kv_info.swa_layer_mask;
    plan.planner_n_ctx             = kv_info.n_ctx;
    plan.planner_n_ctx_is_runtime  = kv_info.n_ctx_is_runtime;
    plan.pp_pipeline_scratch_bytes = unified_cache_get_planned_pp_pipeline_scratch_bytes(device_id);

    std::map<int, bool>   layer_has_attention;
    std::map<int, size_t> layer_weight_bytes;

    // Build entries with priority classification.
    // Dense weights get one entry per tensor.
    // MoE expert tensors (n_experts > 0, name contains "_exps") are split into
    // per-expert entries so each expert competes individually for VRAM.
    plan.entries.reserve(tensor_inventory.size() + (n_experts > 0 ? tensor_inventory.size() * n_experts : 0));
    for (const auto & [name, src_size] : tensor_inventory) {
        const tensor_usage usage = infer_tensor_usage(name.c_str());

        // MoE expert tensors: split into per-expert entries when n_experts > 0.
        // Each expert gets its own placement_entry with expert_id >= 0.
        // The composite tensor entry is NOT added — only per-expert entries.
        if (n_experts > 0 && usage == tensor_usage::MOE_EXPERT_WEIGHT) {
            const int                layer_id    = p4_extract_layer_id(name.c_str());
            const size_t             expert_size = src_size / static_cast<size_t>(n_experts);
            const placement_priority prio        = tensor_to_placement_priority(usage, name.c_str());
            const expert_tensor_role role        = expert_tensor_role_from_tensor_name(name.c_str());

            for (int e = 0; e < n_experts; ++e) {
                placement_entry entry;
                entry.name          = name;
                entry.src_size      = expert_size;
                entry.dst_size      = expert_size;  // Default: same as source (AOS)
                entry.kv_size       = 0;
                entry.layer_id      = layer_id;
                entry.expert_id     = e;
                entry.expert_role   = role;
                entry.priority      = prio;
                entry.on_device     = false;
                entry.target_device = -1;
                plan.entries.push_back(std::move(entry));
            }
            continue;
        }

        // Dense weight: single entry with expert_id = -1.
        placement_entry entry;
        entry.name      = name;
        entry.src_size  = src_size;
        entry.dst_size  = src_size;  // Default: same as source (AOS)
        entry.kv_size   = 0;
        entry.layer_id  = p4_extract_layer_id(name.c_str());
        entry.expert_id = -1;

        entry.priority = tensor_to_placement_priority(usage, name.c_str());
        if (entry.layer_id >= 0) {
            layer_weight_bytes[entry.layer_id] += entry.dst_size;
            if (entry.priority == placement_priority::ATTENTION) {
                layer_has_attention[entry.layer_id] = true;
            } else if (layer_has_attention.find(entry.layer_id) == layer_has_attention.end()) {
                layer_has_attention[entry.layer_id] = false;
            }
        }

        entry.on_device     = false;  // Will be set during packing below
        entry.target_device = -1;     // -1 = host (updated during packing)
        plan.entries.push_back(std::move(entry));
    }

    // Sort by (priority ASC, layer_id ASC, dst_size DESC).
    // This ensures highest-priority weights fill VRAM first, earlier layers
    // before later ones, and larger weights before smaller within the same
    // priority+layer (to minimize fragmentation from small leftovers).
    std::sort(plan.entries.begin(), plan.entries.end(), [](const placement_entry & a, const placement_entry & b) {
        if (a.priority != b.priority) {
            return static_cast<uint8_t>(a.priority) < static_cast<uint8_t>(b.priority);
        }
        if (a.layer_id != b.layer_id) {
            return a.layer_id < b.layer_id;
        }
        if (a.dst_size != b.dst_size) {
            return a.dst_size > b.dst_size;
        }
        if (a.expert_id != b.expert_id) {
            return a.expert_id < b.expert_id;
        }
        return static_cast<uint8_t>(a.expert_role) < static_cast<uint8_t>(b.expert_role);
    });

    // Planner contract:
    //   - dense non-layer tensors (embeddings/output/etc.) may pack individually
    //   - dense per-layer tensors move as a single execution unit
    //   - KV for a layer is charged alongside that dense layer
    //   - MoE experts remain individually placeable
    //
    // When the VRAM arena is active, subtract compute scratch and oneDNN
    // scratch zone capacities from the budget BEFORE packing weights.
    // Without this, weights fill all available VRAM and leave zero space
    // for compute scratch → "VRAM exhaustion" abort on large models.
    size_t remaining       = vram_budget;
    size_t arena_total     = 0;
    size_t scratch_reserve = 0;
    size_t onednn_reserve  = 0;
    size_t runtime_reserve = 0;
    if (vram_arena_enabled()) {
        auto * cache = get_existing_cache_for_device(device_id);
        if (cache && cache->arena_active()) {
            arena_total                = cache->arena_total_size();
            scratch_reserve            = cache->zone_capacity(vram_zone_id::SCRATCH);
            onednn_reserve             = cache->zone_capacity(vram_zone_id::ONEDNN);
            runtime_reserve            = cache->zone_capacity(vram_zone_id::RUNTIME);
            const size_t total_reserve = scratch_reserve + onednn_reserve + runtime_reserve;
            if (remaining > total_reserve) {
                remaining -= total_reserve;
            } else {
                remaining = 0;
            }
            GGML_LOG_INFO(
                "[PLACEMENT] Zone reservation: scratch=%.1f MB + oneDNN=%.1f MB + "
                "runtime=%.1f MB = %.1f MB (weight budget=%.1f MB)\n",
                scratch_reserve / (1024.0 * 1024.0), onednn_reserve / (1024.0 * 1024.0),
                runtime_reserve / (1024.0 * 1024.0), total_reserve / (1024.0 * 1024.0), remaining / (1024.0 * 1024.0));
        }
    }
    log_moe_device_policy("MOE-POLICY", device_id, vram_budget, remaining, arena_total, scratch_reserve,
                          runtime_reserve, onednn_reserve, n_experts);
    std::vector<size_t>                standalone_dense_indices;
    std::map<int, std::vector<size_t>> dense_layer_indices;
    std::vector<size_t>                moe_indices;
    for (size_t i = 0; i < plan.entries.size(); ++i) {
        auto & entry  = plan.entries[i];
        entry.kv_size = 0;
        if (entry.expert_id >= 0) {
            moe_indices.push_back(i);
            continue;
        }
        if (entry.layer_id >= 0) {
            dense_layer_indices[entry.layer_id].push_back(i);
        } else {
            standalone_dense_indices.push_back(i);
        }
    }

    for (size_t idx : standalone_dense_indices) {
        auto & entry = plan.entries[idx];
        if (entry.dst_size <= remaining) {
            entry.on_device     = true;
            entry.target_device = device_id;
            remaining -= entry.dst_size;
            plan.weight_vram_bytes += entry.dst_size;
            plan.vram_bytes += entry.dst_size;
        } else {
            entry.on_device     = false;
            entry.target_device = -1;
            plan.weight_host_bytes += entry.dst_size;
            plan.host_bytes += entry.dst_size;
        }
    }

    for (const auto & [layer_id, indices] : dense_layer_indices) {
        const size_t weight_bytes = layer_weight_bytes[layer_id];
        // Charge each attention layer at its actual KV cost: SWA layers only
        // need min(n_ctx, n_swa) tokens of KV (typically ~8 MB at 4096 tokens)
        // while full-attention layers need the full per-layer allocation (~256 MB
        // at 131K context).  TLSF supports heterogeneous slot sizes so the old
        // uniform-charging workaround is no longer needed.
        const size_t kv_cost =
            layer_has_attention[layer_id] ?
                (kv_info.is_swa_layer(layer_id) ? kv_info.kv_bytes_per_swa_layer() : plan.kv_per_layer) :
                0;
        const size_t total_cost = weight_bytes + kv_cost;
        const bool   on_device  = total_cost <= remaining;
        const int    target     = on_device ? device_id : -1;

        if (on_device) {
            remaining -= total_cost;
            plan.weight_vram_bytes += weight_bytes;
            plan.kv_vram_bytes += kv_cost;
            plan.vram_bytes += total_cost;
        } else {
            plan.weight_host_bytes += weight_bytes;
            plan.kv_host_bytes += kv_cost;
            plan.host_bytes += total_cost;
        }

        plan.layer_device[layer_id] = target;
        plan.kv_device[layer_id]    = kv_cost > 0 ? target : -1;

        size_t kv_anchor = indices.front();
        if (kv_cost > 0) {
            for (size_t idx : indices) {
                if (plan.entries[idx].priority == placement_priority::ATTENTION) {
                    kv_anchor = idx;
                    break;
                }
            }
        }

        for (size_t idx : indices) {
            auto & entry        = plan.entries[idx];
            entry.on_device     = on_device;
            entry.target_device = target;
            entry.kv_size       = idx == kv_anchor ? kv_cost : 0;
        }
    }

    // MoE expert entries: budget-aware placement at (layer, expert) triplet
    // granularity.  A layer executor consumes gate/up/down for the same routed
    // expert together; splitting the roles across device and host creates the
    // worst possible execution shape (device kernels plus CPU handoff for the
    // same routed expert).  Under pressure, keep a complete triplet on one
    // domain or mark the triplet as an explicit host/CPU domain.
    {
        const char *           hotness_source = nullptr;
        const auto             moe_groups     = build_moe_triplet_groups(plan, moe_indices, &hotness_source);
        moe_triplet_pack_stats stats;
        stats.groups  = moe_groups.size();
        stats.hotness = hotness_source != nullptr ? hotness_source : stats.hotness;
        for (const auto & group : moe_groups) {
            if (moe_triplet_complete(group)) {
                stats.complete_groups++;
            } else {
                stats.incomplete_groups++;
            }

            const bool on_device = group.bytes <= remaining;
            const int  target    = on_device ? device_id : -1;
            if (on_device) {
                remaining -= group.bytes;
                plan.weight_vram_bytes += group.bytes;
                plan.vram_bytes += group.bytes;
                stats.device_groups++;
                stats.device_entries += group.indices.size();
                stats.device_bytes += group.bytes;
            } else {
                plan.weight_host_bytes += group.bytes;
                plan.host_bytes += group.bytes;
                stats.host_groups++;
                stats.host_entries += group.indices.size();
                stats.host_bytes += group.bytes;
            }

            for (size_t idx : group.indices) {
                auto & entry        = plan.entries[idx];
                entry.on_device     = on_device;
                entry.target_device = target;
            }
        }
        log_moe_triplet_pack_stats("PLACEMENT-MOE", stats, remaining);
    }

    populate_host_zone_sizing(plan, tensor_inventory, n_experts, kv_info.n_expert_used);

    // Build the name->index lookup for O(1) queries
    plan.build_index();
    log_moe_placement_diagnostics("PLACEMENT-MOE", plan, n_experts);

    // Log placement summary per priority level
    static const char * priority_names[] = { "NORM/EMBED", "ATTENTION", "FFN",          "MOE_GATE",
                                             "MOE_DOWN",   "MOE_UP",    "MOE_GATE_PROJ" };
    for (int p = 0; p < static_cast<int>(placement_priority::COUNT); ++p) {
        size_t device_count = 0, host_count = 0;
        size_t device_bytes = 0, host_bytes = 0;
        for (const auto & e : plan.entries) {
            if (static_cast<int>(e.priority) == p) {
                if (e.on_device) {
                    device_count++;
                    device_bytes += e.dst_size;
                } else {
                    host_count++;
                    host_bytes += e.dst_size;
                }
            }
        }
        if (device_count + host_count > 0) {
            GGML_LOG_INFO("[PLACEMENT] %-14s  device=%3zu (%.1f MB)  host=%3zu (%.1f MB)\n", priority_names[p],
                          device_count, device_bytes / (1024.0 * 1024.0), host_count, host_bytes / (1024.0 * 1024.0));
        }
    }
    if (!layer_weight_bytes.empty()) {
        std::vector<int> layer_ids;
        layer_ids.reserve(layer_weight_bytes.size());
        for (const auto & [layer_id, bytes] : layer_weight_bytes) {
            GGML_UNUSED(bytes);
            layer_ids.push_back(layer_id);
            plan.kv_device.try_emplace(layer_id, -1);
            plan.layer_device.try_emplace(layer_id, -1);
        }
        std::sort(layer_ids.begin(), layer_ids.end());
        for (int layer_id : layer_ids) {
            const size_t weight_bytes = layer_weight_bytes[layer_id];
            const bool   has_attn     = layer_has_attention[layer_id];
            const bool   is_swa       = has_attn && kv_info.is_swa_layer(layer_id);
            const size_t kv_bytes     = has_attn ? (is_swa ? kv_info.kv_bytes_per_swa_layer() : plan.kv_per_layer) : 0;
            const char * kv_label     = has_attn ? (is_swa ? "swa" : "full") : "none";
            const int    dense_target = plan.get_layer_device(layer_id);
            const int    kv_target    = plan.get_kv_device(layer_id);
            GGML_LOG_INFO(
                "[PLACEMENT] layer %3d  weight=%.1f MB  kv=%.1f MB (%s)  total=%.1f MB  "
                "dense_target=%s%d  kv_target=%s%d\n",
                layer_id, weight_bytes / (1024.0 * 1024.0), kv_bytes / (1024.0 * 1024.0), kv_label,
                (weight_bytes + kv_bytes) / (1024.0 * 1024.0), dense_target >= 0 ? "device " : "host ",
                dense_target >= 0 ? dense_target : -1, kv_target >= 0 ? "device " : "host ",
                kv_target >= 0 ? kv_target : -1);
        }
    }
    // KV breakdown summary for heterogeneous attention
    if (kv_info.n_swa_layers > 0) {
        GGML_LOG_INFO(
            "[PLACEMENT] KV breakdown: full_attn=%u layers at %.1f MB/layer, "
            "swa=%u layers at %.1f MB/layer (heterogeneous charging)\n",
            kv_info.n_full_attn_layers(), plan.kv_per_layer / (1024.0 * 1024.0), kv_info.n_swa_layers,
            kv_info.kv_bytes_per_swa_layer() / (1024.0 * 1024.0));
    }
    // Log MoE expert placement breakdown if per-expert entries exist
    {
        size_t expert_device_count = 0, expert_host_count = 0;
        for (const auto & e : plan.entries) {
            if (e.expert_id >= 0) {
                if (e.on_device) {
                    expert_device_count++;
                } else {
                    expert_host_count++;
                }
            }
        }
        if (expert_device_count + expert_host_count > 0) {
            GGML_LOG_INFO("[PLACEMENT] MoE experts: %zu on device, %zu on host (of %zu total)\n", expert_device_count,
                          expert_host_count, expert_device_count + expert_host_count);
        }
    }
    GGML_LOG_INFO(
        "[PLACEMENT] KV sizing: n_layer=%u n_embd_k_gqa=%u n_embd_v_gqa=%u n_ctx=%u (%s) kv_per_layer=%.1f MB\n",
        kv_info.n_layer, kv_info.n_embd_k_gqa, kv_info.n_embd_v_gqa, kv_info.n_ctx,
        kv_info.n_ctx_is_runtime ? "runtime" : "conservative", plan.kv_per_layer / (1024.0 * 1024.0));
    GGML_LOG_INFO("[PLACEMENT] Totals: weights=%.1f MB device + %.1f MB host, kv=%.1f MB device + %.1f MB host\n",
                  plan.weight_vram_bytes / (1024.0 * 1024.0), plan.weight_host_bytes / (1024.0 * 1024.0),
                  plan.kv_vram_bytes / (1024.0 * 1024.0), plan.kv_host_bytes / (1024.0 * 1024.0));
    GGML_LOG_INFO("[PLACEMENT] Total: %.1f MB device + %.1f MB host (budget=%.1f MB)\n",
                  plan.vram_bytes / (1024.0 * 1024.0), plan.host_bytes / (1024.0 * 1024.0),
                  vram_budget / (1024.0 * 1024.0));

    return plan;
}

// ---------------------------------------------------------------------------
// P4.5: Multi-device placement planning with hybrid parallelism.
// ---------------------------------------------------------------------------
//
// Algorithm:
//   1. Compute per-device budgets and total VRAM pool.
//   2. For DENSE layers: assign contiguous layer ranges proportional to VRAM.
//      Each device gets a range [layer_start, layer_end).  Dense weights for
//      a layer go entirely to the owning device.
//   3. For MoE EXPERT tensors: pool remaining VRAM across all devices and
//      fill by Unsloth priority (gate > down > up, earlier layers first).
//      Each expert tensor is assigned to the device with the most remaining
//      budget (first-fit-decreasing across devices).
//   4. Overflow (dense or expert) spills to host (-1).
//
// Falls back to single-device compute_placement_plan() when only 1 device.

multi_gpu_mode get_multi_gpu_mode(bool is_moe) {
    const char * env = std::getenv("GGML_SYCL_MULTI_GPU_MODE");
    if (env) {
        if (std::strcmp(env, "layer") == 0) {
            return multi_gpu_mode::LAYER;
        }
        if (std::strcmp(env, "expert") == 0) {
            return multi_gpu_mode::EXPERT;
        }
        if (std::strcmp(env, "hybrid") == 0) {
            return multi_gpu_mode::HYBRID;
        }
        GGML_LOG_WARN(
            "[PLACEMENT-MULTI] Unknown GGML_SYCL_MULTI_GPU_MODE='%s', "
            "using default\n",
            env);
    }
    // Default: expert parallelism for MoE, layer placement for dense-only.
    // Hybrid/layer MoE can place dense activations and KV on hidden physical
    // devices; keep that an explicit opt-in until all cross-device paths are
    // fully covered.
    return is_moe ? multi_gpu_mode::EXPERT : multi_gpu_mode::LAYER;
}

placement_plan compute_multi_device_plan(const std::vector<device_budget> &                  device_budgets,
                                         const std::vector<std::pair<std::string, size_t>> & tensor_inventory,
                                         int                                                 n_layers,
                                         multi_gpu_mode                                      mode,
                                         const placement_kv_info &                           kv_info,
                                         const ggml_sycl_placement_envelope *                envelope,
                                         int                                                 n_experts) {
    // Single device: delegate to existing P4 path
    if (device_budgets.size() <= 1) {
        const int    dev    = device_budgets.empty() ? 0 : device_budgets[0].device_id;
        const size_t budget = device_budgets.empty() ? 0 : device_budgets[0].vram_budget;
        return compute_placement_plan(tensor_inventory, budget, dev, kv_info, envelope, n_experts);
    }

    const size_t n_devs = device_budgets.size();

    placement_plan plan;
    plan.device_id                = -1;  // Multi-device
    plan.multi_device             = true;
    plan.vram_bytes               = 0;
    plan.host_bytes               = 0;
    plan.weight_vram_bytes        = 0;
    plan.weight_host_bytes        = 0;
    plan.kv_vram_bytes            = 0;
    plan.kv_host_bytes            = 0;
    plan.vram_budget              = 0;
    plan.kv_per_layer             = kv_info.kv_bytes_per_layer();
    plan.kv_per_swa_layer         = kv_info.kv_bytes_per_swa_layer();
    plan.swa_layer_mask           = kv_info.swa_layer_mask;
    plan.planner_n_ctx            = kv_info.n_ctx;
    plan.planner_n_ctx_is_runtime = kv_info.n_ctx_is_runtime;
    for (const auto & db : device_budgets) {
        plan.pp_pipeline_scratch_bytes =
            std::max(plan.pp_pipeline_scratch_bytes, unified_cache_get_planned_pp_pipeline_scratch_bytes(db.device_id));
    }

    std::map<int, bool>   layer_has_attention;
    std::map<int, size_t> layer_weight_bytes;

    // Store device list and per-device tracking
    plan.devices.resize(n_devs);
    plan.per_device_vram.resize(n_devs, 0);
    std::vector<size_t> remaining(n_devs);
    for (size_t d = 0; d < n_devs; d++) {
        plan.devices[d]        = device_budgets[d].device_id;
        remaining[d]           = device_budgets[d].vram_budget;
        size_t arena_total     = 0;
        size_t scratch_reserve = 0;
        size_t onednn_reserve  = 0;
        size_t runtime_reserve = 0;

        // Subtract arena scratch + oneDNN + runtime zone reservations from
        // per-device budget so weights don't fill space needed for compute.
        if (vram_arena_enabled()) {
            auto * cache = get_existing_cache_for_device(device_budgets[d].device_id);
            if (cache && cache->arena_active()) {
                arena_total          = cache->arena_total_size();
                scratch_reserve      = cache->zone_capacity(vram_zone_id::SCRATCH);
                onednn_reserve       = cache->zone_capacity(vram_zone_id::ONEDNN);
                runtime_reserve      = cache->zone_capacity(vram_zone_id::RUNTIME);
                const size_t reserve = scratch_reserve + onednn_reserve + runtime_reserve;
                remaining[d]         = remaining[d] > reserve ? remaining[d] - reserve : 0;
            }
        }

        log_moe_device_policy("MOE-POLICY-MULTI", device_budgets[d].device_id, device_budgets[d].vram_budget,
                              remaining[d], arena_total, scratch_reserve, runtime_reserve, onednn_reserve, n_experts);

        plan.vram_budget += device_budgets[d].vram_budget;
    }

    // Step 1: Compute layer-to-device assignment (dense layers).
    // Proportional to total VRAM, so bigger GPU gets more layers.
    // layer_owner[l] = device index in device_budgets (not device_id).
    std::vector<int> layer_owner(n_layers, 0);
    {
        size_t total_vram = 0;
        for (const auto & db : device_budgets) {
            total_vram += db.total_vram;
        }
        int layer_cursor = 0;
        for (size_t d = 0; d < n_devs; d++) {
            double fraction     = static_cast<double>(device_budgets[d].total_vram) / static_cast<double>(total_vram);
            int    n_dev_layers = static_cast<int>(fraction * n_layers + 0.5);
            if (d == n_devs - 1) {
                n_dev_layers = n_layers - layer_cursor;
            }
            for (int l = 0; l < n_dev_layers && layer_cursor < n_layers; l++, layer_cursor++) {
                layer_owner[layer_cursor] = static_cast<int>(d);
            }
        }
    }

    // Step 2: Build entries with priority classification (same as P4).
    // MoE expert tensors are split into per-expert entries when n_experts > 0.
    plan.entries.reserve(tensor_inventory.size() + (n_experts > 0 ? tensor_inventory.size() * n_experts : 0));
    for (const auto & [name, src_size] : tensor_inventory) {
        const tensor_usage usage = infer_tensor_usage(name.c_str());

        // MoE expert tensors: split into per-expert entries when n_experts > 0.
        if (n_experts > 0 && usage == tensor_usage::MOE_EXPERT_WEIGHT) {
            const int                layer_id    = p4_extract_layer_id(name.c_str());
            const size_t             expert_size = src_size / static_cast<size_t>(n_experts);
            const placement_priority prio        = tensor_to_placement_priority(usage, name.c_str());
            const expert_tensor_role role        = expert_tensor_role_from_tensor_name(name.c_str());

            for (int e = 0; e < n_experts; ++e) {
                placement_entry entry;
                entry.name          = name;
                entry.src_size      = expert_size;
                entry.dst_size      = expert_size;
                entry.kv_size       = 0;
                entry.layer_id      = layer_id;
                entry.expert_id     = e;
                entry.expert_role   = role;
                entry.priority      = prio;
                entry.on_device     = false;
                entry.target_device = -1;
                plan.entries.push_back(std::move(entry));
            }
            continue;
        }

        // Dense weight: single entry with expert_id = -1.
        placement_entry entry;
        entry.name          = name;
        entry.src_size      = src_size;
        entry.dst_size      = src_size;
        entry.kv_size       = 0;
        entry.layer_id      = p4_extract_layer_id(name.c_str());
        entry.expert_id     = -1;
        entry.on_device     = false;
        entry.target_device = -1;

        entry.priority = tensor_to_placement_priority(usage, name.c_str());
        if (entry.layer_id >= 0) {
            layer_weight_bytes[entry.layer_id] += entry.dst_size;
            if (entry.priority == placement_priority::ATTENTION) {
                layer_has_attention[entry.layer_id] = true;
            } else if (layer_has_attention.find(entry.layer_id) == layer_has_attention.end()) {
                layer_has_attention[entry.layer_id] = false;
            }
        }

        plan.entries.push_back(std::move(entry));
    }

    // Step 3: Sort by (priority ASC, layer_id ASC, dst_size DESC, expert_id ASC).
    std::sort(plan.entries.begin(), plan.entries.end(), [](const placement_entry & a, const placement_entry & b) {
        if (a.priority != b.priority) {
            return static_cast<uint8_t>(a.priority) < static_cast<uint8_t>(b.priority);
        }
        if (a.layer_id != b.layer_id) {
            return a.layer_id < b.layer_id;
        }
        if (a.dst_size != b.dst_size) {
            return a.dst_size > b.dst_size;
        }
        if (a.expert_id != b.expert_id) {
            return a.expert_id < b.expert_id;
        }
        return static_cast<uint8_t>(a.expert_role) < static_cast<uint8_t>(b.expert_role);
    });

    // Step 4: Pack entries into devices.
    // Behavior depends on parallelism mode:
    //   LAYER:  All tensors (dense + MoE) assigned by layer owner.
    //   EXPERT: Dense to primary device; MoE experts distributed across all.
    //   HYBRID: Dense by layer owner; MoE experts distributed across all.

    auto is_moe_priority = [](placement_priority p) {
        return p == placement_priority::MOE_GATE || p == placement_priority::MOE_DOWN ||
               p == placement_priority::MOE_UP || p == placement_priority::MOE_GATE_PROJ;
    };

    static const char * mode_names[] = { "LAYER", "EXPERT", "HYBRID" };
    GGML_LOG_INFO("[PLACEMENT-MULTI] Mode: %s\n", mode_names[static_cast<int>(mode)]);

    std::vector<size_t>                standalone_dense_indices;
    std::map<int, std::vector<size_t>> dense_layer_indices;
    std::vector<size_t>                moe_indices;
    for (size_t i = 0; i < plan.entries.size(); ++i) {
        auto & entry  = plan.entries[i];
        entry.kv_size = 0;
        if (entry.expert_id >= 0 || is_moe_priority(entry.priority)) {
            moe_indices.push_back(i);
            continue;
        }
        if (entry.layer_id >= 0) {
            dense_layer_indices[entry.layer_id].push_back(i);
        } else {
            standalone_dense_indices.push_back(i);
        }
    }

    auto commit_bytes = [&](int target_dev_idx, size_t weight_bytes, size_t kv_cost) {
        const size_t total_cost = weight_bytes + kv_cost;
        if (target_dev_idx >= 0 && static_cast<size_t>(target_dev_idx) < n_devs &&
            remaining[target_dev_idx] >= total_cost) {
            remaining[target_dev_idx] -= total_cost;
            plan.weight_vram_bytes += weight_bytes;
            plan.kv_vram_bytes += kv_cost;
            plan.vram_bytes += total_cost;
            plan.per_device_vram[target_dev_idx] += total_cost;
            return device_budgets[target_dev_idx].device_id;
        }

        plan.weight_host_bytes += weight_bytes;
        plan.kv_host_bytes += kv_cost;
        plan.host_bytes += total_cost;
        return -1;
    };

    for (size_t idx : standalone_dense_indices) {
        auto & entry          = plan.entries[idx];
        int    target_dev_idx = 0;
        if (remaining[target_dev_idx] < entry.dst_size) {
            target_dev_idx = -1;
        }
        const int target    = commit_bytes(target_dev_idx, entry.dst_size, 0);
        entry.on_device     = target >= 0;
        entry.target_device = target;
    }

    for (const auto & [layer_id, indices] : dense_layer_indices) {
        int target_dev_idx = 0;
        if (mode != multi_gpu_mode::EXPERT && layer_id >= 0 && layer_id < n_layers) {
            target_dev_idx = layer_owner[layer_id];
        }

        const size_t weight_bytes = layer_weight_bytes[layer_id];
        // Charge each attention layer at its actual KV cost (SWA vs full-attn).
        // TLSF supports heterogeneous slot sizes — see single-device path for details.
        const size_t kv_cost =
            layer_has_attention[layer_id] ?
                (kv_info.is_swa_layer(layer_id) ? kv_info.kv_bytes_per_swa_layer() : plan.kv_per_layer) :
                0;
        const size_t total_cost = weight_bytes + kv_cost;
        if (target_dev_idx >= 0 && static_cast<size_t>(target_dev_idx) < n_devs &&
            remaining[target_dev_idx] < total_cost) {
            target_dev_idx = -1;
        }

        const int target            = commit_bytes(target_dev_idx, weight_bytes, kv_cost);
        plan.layer_device[layer_id] = target;
        plan.kv_device[layer_id]    = kv_cost > 0 ? target : -1;

        size_t kv_anchor = indices.front();
        if (kv_cost > 0) {
            for (size_t idx : indices) {
                if (plan.entries[idx].priority == placement_priority::ATTENTION) {
                    kv_anchor = idx;
                    break;
                }
            }
        }

        for (size_t idx : indices) {
            auto & entry        = plan.entries[idx];
            entry.on_device     = target >= 0;
            entry.target_device = target;
            entry.kv_size       = idx == kv_anchor ? kv_cost : 0;
        }
    }

    {
        const char *           hotness_source = nullptr;
        const auto             moe_groups     = build_moe_triplet_groups(plan, moe_indices, &hotness_source);
        moe_triplet_pack_stats stats;
        stats.groups  = moe_groups.size();
        stats.hotness = hotness_source != nullptr ? hotness_source : stats.hotness;
        for (const auto & group : moe_groups) {
            if (moe_triplet_complete(group)) {
                stats.complete_groups++;
            } else {
                stats.incomplete_groups++;
            }

            int target_dev_idx = -1;
            if (mode != multi_gpu_mode::LAYER) {
                size_t best_remaining = 0;
                for (size_t d = 0; d < n_devs; d++) {
                    if (remaining[d] >= group.bytes && remaining[d] > best_remaining) {
                        best_remaining = remaining[d];
                        target_dev_idx = static_cast<int>(d);
                    }
                }
            } else if (group.layer_id >= 0 && group.layer_id < n_layers) {
                target_dev_idx = layer_owner[group.layer_id];
                if (remaining[target_dev_idx] < group.bytes) {
                    target_dev_idx = -1;
                }
            }

            const int target = commit_bytes(target_dev_idx, group.bytes, 0);
            if (target >= 0) {
                stats.device_groups++;
                stats.device_entries += group.indices.size();
                stats.device_bytes += group.bytes;
            } else {
                stats.host_groups++;
                stats.host_entries += group.indices.size();
                stats.host_bytes += group.bytes;
            }

            for (size_t idx : group.indices) {
                auto & entry        = plan.entries[idx];
                entry.on_device     = target >= 0;
                entry.target_device = target;
            }
        }
        const size_t total_remaining = std::accumulate(remaining.begin(), remaining.end(), size_t(0));
        log_moe_triplet_pack_stats("PLACEMENT-MOE-MULTI", stats, total_remaining);
    }

    // The multi-device plan places all KV on-device (kv_host_bytes = 0) because it
    // assumes all GPUs are available. At runtime ONEAPI_DEVICE_SELECTOR may restrict
    // execution to a single device, causing KV to spill to host. Size the KV zone
    // conservatively: the full KV footprint across all layers, so the host zone is
    // always large enough regardless of runtime GPU availability.
    if (plan.kv_host_bytes == 0 && kv_info.valid()) {
        const size_t total_kv = kv_info.n_full_attn_layers() * kv_info.kv_bytes_per_layer() +
                                kv_info.n_swa_layers * kv_info.kv_bytes_per_swa_layer();
        plan.kv_host_bytes = total_kv;
    }

    populate_host_zone_sizing(plan, tensor_inventory, n_experts, kv_info.n_expert_used);

    plan.build_index();
    log_moe_placement_diagnostics("PLACEMENT-MOE", plan, n_experts);

    // Build runtime query maps for multi-device inference routing.

    // device_layers: per-device contiguous layer range
    for (size_t d = 0; d < n_devs; d++) {
        int dev_id = device_budgets[d].device_id;
        int first = n_layers, last = -1;
        for (int l = 0; l < n_layers; l++) {
            if (layer_owner[l] == static_cast<int>(d)) {
                first = std::min(first, l);
                last  = std::max(last, l);
            }
        }
        if (first <= last) {
            plan.device_layers[dev_id] = { first, last };
        }
    }

    for (int l = 0; l < n_layers; l++) {
        if (plan.kv_device.find(l) == plan.kv_device.end()) {
            plan.kv_device[l] = -1;
        }
        if (plan.layer_device.find(l) == plan.layer_device.end()) {
            plan.layer_device[l] = -1;
        }
    }

    // expert_device: per-layer per-expert device assignment (from entry target_device).
    // With per-expert splitting (n_experts > 0), each expert has its own entry
    // and can be assigned to a different device.  Without splitting, the whole
    // MoE tensor goes to one device and all experts share that target.
    for (const auto & entry : plan.entries) {
        if (!is_moe_priority(entry.priority)) {
            continue;
        }
        if (entry.layer_id < 0) {
            continue;
        }
        const int eid                           = (entry.expert_id >= 0) ? entry.expert_id : 0;
        plan.expert_device[entry.layer_id][eid] = entry.target_device;
    }

    // Log multi-device placement summary
    GGML_LOG_INFO("[PLACEMENT-MULTI] %zu devices, %d layers, hybrid parallelism\n", n_devs, n_layers);
    for (size_t d = 0; d < n_devs; d++) {
        int first_layer = n_layers, last_layer = -1;
        for (int l = 0; l < n_layers; l++) {
            if (layer_owner[l] == static_cast<int>(d)) {
                first_layer = std::min(first_layer, l);
                last_layer  = std::max(last_layer, l);
            }
        }
        GGML_LOG_INFO(
            "[PLACEMENT-MULTI] Device %d: layers [%d, %d], "
            "%.1f MB planned bytes used (%.1f MB budget)\n",
            device_budgets[d].device_id, first_layer, last_layer, plan.per_device_vram[d] / (1024.0 * 1024.0),
            device_budgets[d].vram_budget / (1024.0 * 1024.0));
    }

    // Per-priority breakdown
    static const char * priority_names[] = { "NORM/EMBED", "ATTENTION", "FFN",          "MOE_GATE",
                                             "MOE_DOWN",   "MOE_UP",    "MOE_GATE_PROJ" };
    for (int p = 0; p < static_cast<int>(placement_priority::COUNT); ++p) {
        size_t dev_count = 0, host_count = 0;
        size_t dev_bytes = 0, host_bytes = 0;
        for (const auto & e : plan.entries) {
            if (static_cast<int>(e.priority) == p) {
                if (e.on_device) {
                    dev_count++;
                    dev_bytes += e.dst_size;
                } else {
                    host_count++;
                    host_bytes += e.dst_size;
                }
            }
        }
        if (dev_count + host_count > 0) {
            GGML_LOG_INFO("[PLACEMENT-MULTI] %-14s  device=%3zu (%.1f MB)  host=%3zu (%.1f MB)\n", priority_names[p],
                          dev_count, dev_bytes / (1024.0 * 1024.0), host_count, host_bytes / (1024.0 * 1024.0));
        }
    }
    // KV breakdown summary for heterogeneous attention
    if (kv_info.n_swa_layers > 0) {
        GGML_LOG_INFO(
            "[PLACEMENT-MULTI] KV breakdown: full_attn=%u layers at %.1f MB/layer, "
            "swa=%u layers at %.1f MB/layer (heterogeneous charging)\n",
            kv_info.n_full_attn_layers(), plan.kv_per_layer / (1024.0 * 1024.0), kv_info.n_swa_layers,
            kv_info.kv_bytes_per_swa_layer() / (1024.0 * 1024.0));
    }
    GGML_LOG_INFO(
        "[PLACEMENT-MULTI] KV sizing: n_layer=%u n_embd_k_gqa=%u n_embd_v_gqa=%u n_ctx=%u (%s) kv_per_layer=%.1f MB\n",
        kv_info.n_layer, kv_info.n_embd_k_gqa, kv_info.n_embd_v_gqa, kv_info.n_ctx,
        kv_info.n_ctx_is_runtime ? "runtime" : "conservative", plan.kv_per_layer / (1024.0 * 1024.0));
    GGML_LOG_INFO("[PLACEMENT-MULTI] Totals: weights=%.1f MB device + %.1f MB host, kv=%.1f MB device + %.1f MB host\n",
                  plan.weight_vram_bytes / (1024.0 * 1024.0), plan.weight_host_bytes / (1024.0 * 1024.0),
                  plan.kv_vram_bytes / (1024.0 * 1024.0), plan.kv_host_bytes / (1024.0 * 1024.0));
    GGML_LOG_INFO("[PLACEMENT-MULTI] Total: %.1f MB device + %.1f MB host (budget=%.1f MB)\n",
                  plan.vram_bytes / (1024.0 * 1024.0), plan.host_bytes / (1024.0 * 1024.0),
                  plan.vram_budget / (1024.0 * 1024.0));

    return plan;
}

void unified_cache::memset(const mem_handle & h, int value, size_t size, sycl::queue & stream) {
    mem_fill(h, value, size, stream);
}

void unified_cache::memcpy(const mem_handle & dst, const mem_handle & src, size_t size, sycl::queue & stream) {
    mem_copy(dst, src, size, stream);
}

}  // namespace ggml_sycl
