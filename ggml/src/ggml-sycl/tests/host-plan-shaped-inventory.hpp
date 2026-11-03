//
// The shaped MoE inventories the host-plan tests share, each in the planner's form (placement_tensor_info rows) and in
// the loader's (the ggml_sycl_tensor_inventory libllama hands the backend), built from one row list so the two forms
// cannot drift. Each shape names the model it stands for; no tensor is ever allocated.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include "../unified-cache.hpp"
#include "ggml-sycl.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace host_plan_shaped_inventory {

struct inventory_row {
    std::string name;
    ggml_type   type;
    int64_t     ne[3];
};

// One MoE model's shape: every weight in `type`, `experts` experts per layer of which `experts_used` route, the
// loader's planning shape (n_ctx, n_ubatch) and the PP MoE oneDNN ring the loader would size for it (0: none).
struct moe_shape {
    ggml_type type;
    int       layers;
    int       experts;
    int       experts_used;
    int64_t   embd;
    int64_t   ffn;
    int64_t   vocab;
    uint32_t  n_ctx;
    uint32_t  n_ubatch;
    size_t    ring_weight_slot_bytes;
    size_t    ring_activation_slot_bytes;
    size_t    ring_output_slot_bytes;
};

// The T2 fixture: DeepSeek's quant type, at a size a test can hold, with no ring.
constexpr moe_shape q2k_shape = { GGML_TYPE_Q2_K, 2, 8, 2, 1024, 512, 4096, 16, 1, 0, 0, 0 };

// Qwen3.8-Flash-Next Q8_0 (qwen4exp): 48 layers, 512 experts of which 10 route, hidden 2560, expert FFN 640, at its
// real dimensions, so its ~128 GiB of experts overflow any card on this host and the plan puts experts on the host.
// The loader's planning shape is n_ctx = n_ubatch = 512. The ring is the IQ3_XXS load's (test-kv-room-cost.cpp): a
// 425 MiB weight slot, 2.5 MiB activation and 5 MiB output per row at n_ubatch 512, depth 1. Each row size is a
// multiple of 256, so the slots are already at the ring's alignment.
constexpr moe_shape qwen38_shape = {
    GGML_TYPE_Q8_0, 48, 512, 10, 2560, 640, 151936, 512, 512, 425ull << 20, 512ull * 2621440ull, 512ull * 5242880ull,
};

inline std::vector<inventory_row> shaped_rows(const moe_shape & s) {
    std::vector<inventory_row> rows;
    rows.push_back({
        "token_embd.weight", s.type, { s.embd, s.vocab, 1 }
    });
    for (int l = 0; l < s.layers; ++l) {
        const std::string blk = "blk." + std::to_string(l) + ".";
        rows.push_back({
            blk + "attn_norm.weight", GGML_TYPE_F32, { s.embd, 1, 1 }
        });
        rows.push_back({
            blk + "attn_q.weight", s.type, { s.embd, s.embd, 1 }
        });
        rows.push_back({
            blk + "attn_k.weight", s.type, { s.embd, s.embd, 1 }
        });
        rows.push_back({
            blk + "attn_v.weight", s.type, { s.embd, s.embd, 1 }
        });
        rows.push_back({
            blk + "attn_output.weight", s.type, { s.embd, s.embd, 1 }
        });
        rows.push_back({
            blk + "ffn_norm.weight", GGML_TYPE_F32, { s.embd, 1, 1 }
        });
        rows.push_back({
            blk + "ffn_gate_inp.weight", GGML_TYPE_F32, { s.embd, s.experts, 1 }
        });
        rows.push_back({
            blk + "ffn_gate_exps.weight", s.type, { s.embd, s.ffn, s.experts }
        });
        rows.push_back({
            blk + "ffn_up_exps.weight", s.type, { s.embd, s.ffn, s.experts }
        });
        rows.push_back({
            blk + "ffn_down_exps.weight", s.type, { s.ffn, s.embd, s.experts }
        });
    }
    rows.push_back({
        "output_norm.weight", GGML_TYPE_F32, { s.embd, 1, 1 }
    });
    rows.push_back({
        "output.weight", s.type, { s.embd, s.vocab, 1 }
    });
    return rows;
}

inline size_t row_bytes(const inventory_row & row) {
    return ggml_row_size(row.type, row.ne[0]) * static_cast<size_t>(row.ne[1]) * static_cast<size_t>(row.ne[2]);
}

inline std::vector<ggml_sycl::placement_tensor_info> planner_inventory(const moe_shape & s = q2k_shape) {
    std::vector<ggml_sycl::placement_tensor_info> inv;
    for (const inventory_row & row : shaped_rows(s)) {
        ggml_sycl::placement_tensor_info t(row.name, row_bytes(row));
        t.type  = row.type;
        t.ne[0] = row.ne[0];
        t.ne[1] = row.ne[1];
        t.ne[2] = row.ne[2];
        t.ne[3] = 1;
        inv.push_back(t);
    }
    return inv;
}

// The loader's form. The rows own the name strings the tensor infos point at, so the object is not copyable.
struct loader_inventory {
    std::vector<inventory_row>         rows;
    std::vector<ggml_sycl_tensor_info> tensors;
    ggml_sycl_tensor_inventory         inv{};

    explicit loader_inventory(const moe_shape & s = q2k_shape) : rows(shaped_rows(s)) {
        size_t total = 0;
        for (const inventory_row & row : rows) {
            ggml_sycl_tensor_info t{};
            t.name  = row.name.c_str();
            t.size  = row_bytes(row);
            t.type  = row.type;
            t.ne[0] = row.ne[0];
            t.ne[1] = row.ne[1];
            t.ne[2] = row.ne[2];
            t.ne[3] = 1;
            total += t.size;
            tensors.push_back(t);
        }
        inv.tensors                             = tensors.data();
        inv.count                               = tensors.size();
        inv.total_size                          = total;
        inv.n_expert                            = s.experts;
        inv.n_expert_used                       = s.experts_used;
        inv.n_layer                             = static_cast<uint32_t>(s.layers);
        inv.n_ctx                               = s.n_ctx;
        inv.n_ubatch                            = s.n_ubatch;
        inv.pp_moe_onednn_weight_slot_bytes     = s.ring_weight_slot_bytes;
        inv.pp_moe_onednn_activation_slot_bytes = s.ring_activation_slot_bytes;
        inv.pp_moe_onednn_output_slot_bytes     = s.ring_output_slot_bytes;
        inv.pp_moe_onednn_ring_depth            = s.ring_weight_slot_bytes > 0 ? 1 : 0;
        // The per-row bytes behind the two row slots: a stage under a load owner re-publishes the ring from these at
        // n_ubatch, so they must give back the slots above or the stage plans another ring than the inventory names.
        if (s.n_ubatch > 0) {
            inv.pp_moe_onednn_activation_bytes_per_row = s.ring_activation_slot_bytes / s.n_ubatch;
            inv.pp_moe_onednn_output_bytes_per_row     = s.ring_output_slot_bytes / s.n_ubatch;
        }
        inv.pp_moe_onednn_scratch_bytes =
            s.ring_weight_slot_bytes + s.ring_activation_slot_bytes + s.ring_output_slot_bytes;
    }

    loader_inventory(const loader_inventory &)             = delete;
    loader_inventory & operator=(const loader_inventory &) = delete;
};

}  // namespace host_plan_shaped_inventory
