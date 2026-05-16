//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_FATTN_ONEDNN_HPP
#define GGML_SYCL_FATTN_ONEDNN_HPP

#include "common.hpp"

#if GGML_SYCL_DNNL

#    include "fattn-common.hpp"
#    include "fattn.hpp"
#    include "oneapi/dnnl/dnnl_graph.hpp"
#    include "oneapi/dnnl/dnnl_graph_sycl.hpp"
#    include "oneapi/dnnl/dnnl_sycl.hpp"

#    include <cstdint>
#    include <functional>
#    include <memory>
#    include <mutex>
#    include <unordered_map>

struct ggml_sycl_onednn_fa_materialized_kv {
    ggml_sycl::scoped_unified_alloc K_alloc;
    ggml_sycl::scoped_unified_alloc V_alloc;
    ggml_sycl::mem_handle           K;
    ggml_sycl::mem_handle           V;
};

// Cache key for oneDNN graph compiled_partition.
// Per plan §8.7: keyed by (device_id, shape) — compiled_partitions are device-specific.
// Strides + dtypes are part of the key because two calls with identical logical
// shapes but different memory layouts (e.g. permuted Q) or different mask dtypes
// must NOT share a compiled partition.
struct sdpa_shape_key {
    int  device_id;
    int  D;
    int  ncols;  // ne01 (query count)
    int  ne11;   // KV length
    int  H_q;
    int  H_kv;
    bool has_mask;
    bool is_gqa;  // H_q != H_kv

    // Source tensor types (ggml_type ids) — mask may be f16 or f32 depending
    // on whether flash_attn is set (ggml casts kq_mask to f16 in that case).
    int Q_type;
    int K_type;
    int V_type;
    int mask_type;

    // Byte strides from the underlying ggml tensors (nb[1], nb[2], nb[3]).
    // Stored at key granularity so that compiled partitions are not reused
    // across permuted / non-contiguous source layouts.
    int64_t q_nb1, q_nb2, q_nb3;
    int64_t k_nb1, k_nb2, k_nb3;
    int64_t v_nb1, v_nb2, v_nb3;
    int64_t m_nb1, m_nb2, m_nb3;

    bool operator==(const sdpa_shape_key & o) const {
        return device_id == o.device_id && D == o.D && ncols == o.ncols && ne11 == o.ne11 && H_q == o.H_q &&
               H_kv == o.H_kv && has_mask == o.has_mask && is_gqa == o.is_gqa && Q_type == o.Q_type &&
               K_type == o.K_type && V_type == o.V_type && mask_type == o.mask_type && q_nb1 == o.q_nb1 &&
               q_nb2 == o.q_nb2 && q_nb3 == o.q_nb3 && k_nb1 == o.k_nb1 && k_nb2 == o.k_nb2 && k_nb3 == o.k_nb3 &&
               v_nb1 == o.v_nb1 && v_nb2 == o.v_nb2 && v_nb3 == o.v_nb3 && m_nb1 == o.m_nb1 && m_nb2 == o.m_nb2 &&
               m_nb3 == o.m_nb3;
    }
};

struct sdpa_shape_key_hash {
    size_t operator()(const sdpa_shape_key & k) const {
        auto mix = [](size_t h, size_t v) {
            return h ^ (v + 0x9e3779b9 + (h << 6) + (h >> 2));
        };
        size_t h = std::hash<int>{}(k.device_id);
        h        = mix(h, std::hash<int>{}(k.D));
        h        = mix(h, std::hash<int>{}(k.ncols));
        h        = mix(h, std::hash<int>{}(k.ne11));
        h        = mix(h, std::hash<int>{}(k.H_q));
        h        = mix(h, std::hash<int>{}(k.H_kv));
        h        = mix(h, std::hash<bool>{}(k.has_mask));
        h        = mix(h, std::hash<bool>{}(k.is_gqa));
        h        = mix(h, std::hash<int>{}(k.Q_type));
        h        = mix(h, std::hash<int>{}(k.K_type));
        h        = mix(h, std::hash<int>{}(k.V_type));
        h        = mix(h, std::hash<int>{}(k.mask_type));
        h        = mix(h, std::hash<int64_t>{}(k.q_nb1));
        h        = mix(h, std::hash<int64_t>{}(k.q_nb2));
        h        = mix(h, std::hash<int64_t>{}(k.q_nb3));
        h        = mix(h, std::hash<int64_t>{}(k.k_nb1));
        h        = mix(h, std::hash<int64_t>{}(k.k_nb2));
        h        = mix(h, std::hash<int64_t>{}(k.k_nb3));
        h        = mix(h, std::hash<int64_t>{}(k.v_nb1));
        h        = mix(h, std::hash<int64_t>{}(k.v_nb2));
        h        = mix(h, std::hash<int64_t>{}(k.v_nb3));
        h        = mix(h, std::hash<int64_t>{}(k.m_nb1));
        h        = mix(h, std::hash<int64_t>{}(k.m_nb2));
        h        = mix(h, std::hash<int64_t>{}(k.m_nb3));
        return h;
    }
};

// Per-shape cache entry: compiled_partition + port lists + a per-shape USM
// scratch for the scalar Divide op.
//
// `scale_usm` is a sycl::half scalar buffer (sycl::malloc_host, pinned +
// GPU-accessible via PCIe zero-copy). It is allocated AND populated exactly
// once inside build_and_compile_sdpa() — which runs under the cache mutex —
// so the execute path only READS it; there is no WRITE race.
//
// The value stored is the divisor sqrt(D) derived from `key.D` at compile
// time. A runtime assertion at dispatch verifies that 1/params.scale matches
// sqrt(key.D) within a tight epsilon, so any future model that does NOT
// follow the 1/sqrt(D) convention fails loudly instead of silently producing
// wrong outputs.
//
// `usm_queue` is a non-owning pointer to the sycl::queue used for allocation;
// it is valid for the full lifetime of the entry because the queue is owned
// by the global device registry (ggml_sycl_get_device()), which outlives
// every backend context.
struct sdpa_compiled_entry {
    dnnl::graph::compiled_partition          cp;
    std::vector<dnnl::graph::logical_tensor> in_ports;
    std::vector<dnnl::graph::logical_tensor> out_ports;
    sycl::half *                             scale_usm = nullptr;
    sycl::queue *                            usm_queue = nullptr;

    sdpa_compiled_entry()                                        = default;
    sdpa_compiled_entry(const sdpa_compiled_entry &)             = delete;
    sdpa_compiled_entry & operator=(const sdpa_compiled_entry &) = delete;

    ~sdpa_compiled_entry() {
        if (scale_usm && usm_queue) {
            sycl::free(scale_usm, *usm_queue);
        }
    }
};

// SDPA compiled_partition cache owned by backend context (one per device).
// Stored as a pointer on ggml_backend_sycl_context so the header doesn't
// need to pull in all graph headers downstream.
//
// Concurrency contract (two distinct safety properties):
// - Entry lifetime: `hits` stores `std::shared_ptr<sdpa_compiled_entry>`. The
//   dispatch entry copies the shared_ptr under `m`, then unlocks. The entry
//   survives any subsequent map rehash/erase because the copied shared_ptr
//   holds the last reference as needed. This is how callers safely touch
//   `in_ports` / `out_ports` / `cp` / `scale_usm` after releasing the lock.
// - Cache lifetime: the `sdpa_partition_cache` itself is NOT kept alive by
//   shared_ptr. It relies on ggml's backend-teardown contract — no FA
//   dispatch is in flight when `ggml_sycl_sdpa_cache_destroy` is called.
//   shared_ptr on entries does not, by itself, make destroying the cache
//   concurrently with a dispatch safe; the teardown contract is what makes
//   it safe.
// - `m` serialises all reads/writes of `hits` and `negative`. One mutex per
//   ggml_backend_sycl_context gives lock-free cross-device dispatch on
//   multi-GPU systems. scale_usm is no longer on the cache — it moved to
//   `sdpa_compiled_entry` (per-shape, set-once), so no write-site remains
//   on the execute path.
struct sdpa_partition_cache {
    std::mutex                                                                                    m;
    std::unordered_map<sdpa_shape_key, std::shared_ptr<sdpa_compiled_entry>, sdpa_shape_key_hash> hits;
    std::unordered_map<sdpa_shape_key, bool, sdpa_shape_key_hash>                                 negative;
};

// Check whether the current op is eligible for the oneDNN graph SDPA path.
// Eligibility returns true when: no sinks, no softcap, no FP8 KV, f16 Q/K/V
// (and if mask is present, f16 or f32 mask), D <= 512, ncols >= 8 (PP batch),
// not multi-seq. Device-USM-ness of Q/K/V is NOT checked here — that
// rejection happens one level later inside ggml_sycl_flash_attn_ext_onednn
// via the kDeviceVAThreshold guards, because the buffers can move between
// host-pinned and device between eligibility and dispatch under
// GGML_SYCL_HOST_COMPUTE / GGML_SYCL_KV_HOST.
// Caller must additionally check: !safe_decode, g_sycl_fa_onednn_enabled,
// !g_sycl_paged_v2_enabled (all three are file-scope / template-local flags
// that this function cannot see directly).
bool ggml_sycl_flash_attn_ext_onednn_eligible(const fattn_params & params,
                                              int                  H_q,
                                              int                  H_kv,
                                              bool                 kv_is_fp8,
                                              bool                 multi_seq);

// Execute flash attention via oneDNN graph SDPA (compile-and-cache on miss).
// Falls back silently to false on any oneDNN error; caller routes to kernel path.
// Returns true if oneDNN executed, false if fell through.
bool ggml_sycl_flash_attn_ext_onednn(ggml_backend_sycl_context & ctx, const fattn_params & params);

bool ggml_sycl_flash_attn_ext_onednn_materialize_kv(
    const ggml_sycl_onednn_fa_materialization_desc & desc,
    const fattn_params &                            params,
    sycl::queue &                                   stream,
    ggml_sycl_onednn_fa_materialized_kv *           out);

#endif  // GGML_SYCL_DNNL
#endif  // GGML_SYCL_FATTN_ONEDNN_HPP
