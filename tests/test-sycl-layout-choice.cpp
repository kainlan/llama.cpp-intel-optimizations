// SYCL unified-cache layout choice test.
// Verifies that resolve() returns the correct layout for cached weights.

#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#ifndef GGML_SYCL_WARP_SIZE
#    define GGML_SYCL_WARP_SIZE 32
#endif
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/ggml-sycl-test.hpp"
#include "ggml-sycl/unified-cache.hpp"

static XMXCapabilities test_mxfp4_caps() {
    XMXCapabilities caps{};
    caps.supported                = true;
    caps.supports_int8            = true;
    caps.M                        = GGML_SYCL_MXFP4_MOE_XMX_M;
    caps.N                        = GGML_SYCL_MXFP4_MOE_XMX_N;
    caps.K                        = GGML_SYCL_MXFP4_MOE_XMX_K;
    caps.slm_size                 = 64 * 1024;
    caps.max_work_group_size      = 256;
    caps.preferred_sub_group_size = GGML_SYCL_MXFP4_MOE_XMX_SG;
    caps.sub_group_sizes[0]       = GGML_SYCL_MXFP4_MOE_XMX_SG;
    caps.sub_group_size_count     = 1;
    caps.optimal_tiles_n          = 4;
    return caps;
}

static bool run_mxfp4_moe_policy_test() {
    XMXCapabilities caps = test_mxfp4_caps();

    auto decision = ggml_sycl_select_mxfp4_moe_layout(caps,
                                                      /*in_dim=*/2880,
                                                      /*out_dim=*/2880,
                                                      /*n_experts=*/128,
                                                      /*device_resident=*/true,
                                                      /*tiled_kernel_validated=*/false);
    if (decision.layout != GGML_LAYOUT_SOA || !decision.tiled_eligible || decision.tiled_selected ||
        std::strcmp(decision.reason, "xmx-tiled-not-validated-shared-soa") != 0) {
        printf(
            "FAIL: expected eligible-but-not-selected tiled MXFP4 decision, got layout=%d eligible=%d selected=%d "
            "reason=%s\n",
            (int) decision.layout, decision.tiled_eligible ? 1 : 0, decision.tiled_selected ? 1 : 0, decision.reason);
        return false;
    }

    decision = ggml_sycl_select_mxfp4_moe_layout(caps,
                                                 /*in_dim=*/2880,
                                                 /*out_dim=*/2880,
                                                 /*n_experts=*/128,
                                                 /*device_resident=*/true,
                                                 /*tiled_kernel_validated=*/true);
    if (decision.layout != GGML_LAYOUT_XMX_TILED || !decision.tiled_selected ||
        decision.tile_n_total != (int64_t) (caps.N * caps.optimal_tiles_n)) {
        printf("FAIL: expected validated tiled MXFP4 decision, got layout=%d selected=%d tile_n=%lld\n",
               (int) decision.layout, decision.tiled_selected ? 1 : 0, (long long) decision.tile_n_total);
        return false;
    }

    XMXCapabilities unsupported = caps;
    unsupported.supports_int8   = false;
    decision                    = ggml_sycl_select_mxfp4_moe_layout(unsupported,
                                                                    /*in_dim=*/4096,
                                                                    /*out_dim=*/2880,
                                                                    /*n_experts=*/128,
                                                                    /*device_resident=*/true,
                                                                    /*tiled_kernel_validated=*/true);
    if (decision.layout != GGML_LAYOUT_COALESCED || decision.tiled_eligible) {
        printf("FAIL: expected non-XMX coalesced fallback, got layout=%d eligible=%d reason=%s\n",
               (int) decision.layout, decision.tiled_eligible ? 1 : 0, decision.reason);
        return false;
    }

    XMXCapabilities different_tile = caps;
    different_tile.K               = 64;
    decision                       = ggml_sycl_select_mxfp4_moe_layout(different_tile,
                                                                       /*in_dim=*/4096,
                                                                       /*out_dim=*/2880,
                                                                       /*n_experts=*/128,
                                                                       /*device_resident=*/true,
                                                                       /*tiled_kernel_validated=*/true);
    if (decision.layout != GGML_LAYOUT_COALESCED || decision.tiled_eligible ||
        std::strcmp(decision.reason, "kernel-tile-shape") != 0) {
        printf("FAIL: expected tile-shape fallback, got layout=%d eligible=%d reason=%s\n", (int) decision.layout,
               decision.tiled_eligible ? 1 : 0, decision.reason);
        return false;
    }

    decision = ggml_sycl_select_mxfp4_moe_layout(caps,
                                                 /*in_dim=*/2880,
                                                 /*out_dim=*/2880,
                                                 /*n_experts=*/128,
                                                 /*device_resident=*/false,
                                                 /*tiled_kernel_validated=*/true);
    if (decision.layout != GGML_LAYOUT_AOS || decision.tiled_eligible ||
        std::strcmp(decision.reason, "host-resident-aos") != 0) {
        printf("FAIL: expected host-resident AOS fallback, got layout=%d eligible=%d reason=%s\n",
               (int) decision.layout, decision.tiled_eligible ? 1 : 0, decision.reason);
        return false;
    }

    printf("PASS: MXFP4 MoE tiled policy is capability/shape/residency driven\n");
    return true;
}

static bool run_mxfp4_grouped_dpas_policy_test() {
    XMXCapabilities caps = test_mxfp4_caps();

    auto decision = ggml_sycl_select_mxfp4_grouped_dpas(caps,
                                                        /*in_dim=*/2880,
                                                        /*out_dim=*/2880,
                                                        /*grouped_rows=*/caps.N - 1, GGML_LAYOUT_MXFP4_DPAS,
                                                        /*local_device_resident=*/true);
    if (decision.shape_eligible || decision.layout_ready || std::strcmp(decision.reason, "rows") != 0) {
        printf("FAIL: expected grouped DPAS row-count rejection, got eligible=%d ready=%d reason=%s\n",
               decision.shape_eligible ? 1 : 0, decision.layout_ready ? 1 : 0, decision.reason);
        return false;
    }

    decision = ggml_sycl_select_mxfp4_grouped_dpas(caps,
                                                   /*in_dim=*/2880,
                                                   /*out_dim=*/2880,
                                                   /*grouped_rows=*/caps.N, GGML_LAYOUT_SOA,
                                                   /*local_device_resident=*/true);
    if (!decision.shape_eligible || decision.layout_ready ||
        std::strcmp(decision.reason, "layout-not-materialized") != 0 || decision.n_tile_repeat != 1) {
        printf("FAIL: expected grouped DPAS shape-only eligibility, got eligible=%d ready=%d repeat=%zu reason=%s\n",
               decision.shape_eligible ? 1 : 0, decision.layout_ready ? 1 : 0, decision.n_tile_repeat, decision.reason);
        return false;
    }

    decision = ggml_sycl_select_mxfp4_grouped_dpas(caps,
                                                   /*in_dim=*/2880,
                                                   /*out_dim=*/2880,
                                                   /*grouped_rows=*/4 * caps.N, GGML_LAYOUT_MXFP4_DPAS,
                                                   /*local_device_resident=*/true);
    if (!decision.shape_eligible || !decision.layout_ready || decision.n_tile_repeat != 2 ||
        std::strcmp(decision.reason, "layout-ready") != 0) {
        printf("FAIL: expected grouped DPAS ready decision, got eligible=%d ready=%d repeat=%zu reason=%s\n",
               decision.shape_eligible ? 1 : 0, decision.layout_ready ? 1 : 0, decision.n_tile_repeat, decision.reason);
        return false;
    }

    decision = ggml_sycl_select_mxfp4_grouped_dpas(caps,
                                                   /*in_dim=*/2880,
                                                   /*out_dim=*/2880,
                                                   /*grouped_rows=*/4 * caps.N, GGML_LAYOUT_MXFP4_DPAS,
                                                   /*local_device_resident=*/false);
    if (decision.shape_eligible || std::strcmp(decision.reason, "residency") != 0) {
        printf("FAIL: expected grouped DPAS residency rejection, got eligible=%d reason=%s\n",
               decision.shape_eligible ? 1 : 0, decision.reason);
        return false;
    }

    XMXCapabilities unsupported = caps;
    unsupported.supports_int8   = false;
    decision                    = ggml_sycl_select_mxfp4_grouped_dpas(unsupported,
                                                                      /*in_dim=*/2880,
                                                                      /*out_dim=*/2880,
                                                                      /*grouped_rows=*/4 * caps.N, GGML_LAYOUT_MXFP4_DPAS,
                                                                      /*local_device_resident=*/true);
    if (decision.shape_eligible || std::strcmp(decision.reason, "capability") != 0) {
        printf("FAIL: expected grouped DPAS capability rejection, got eligible=%d reason=%s\n",
               decision.shape_eligible ? 1 : 0, decision.reason);
        return false;
    }

    const int32_t dense_counts[] = { 16, 16, 16, 16 };
    auto          occupancy      = ggml_sycl_select_mxfp4_grouped_dpas_occupancy(caps, dense_counts,
                                                                                 sizeof(dense_counts) / sizeof(dense_counts[0]));
    if (!occupancy.dispatch_ready || occupancy.total_rows != 64 || occupancy.padded_rows != 64 ||
        std::strcmp(occupancy.reason, "ok") != 0) {
        printf("FAIL: expected dense grouped occupancy ready, got ready=%d rows=%zu padded=%zu reason=%s\n",
               occupancy.dispatch_ready ? 1 : 0, occupancy.total_rows, occupancy.padded_rows, occupancy.reason);
        return false;
    }

    const int32_t sparse_counts[] = { 16, 1, 1, 1 };
    occupancy                     = ggml_sycl_select_mxfp4_grouped_dpas_occupancy(caps, sparse_counts,
                                                                                  sizeof(sparse_counts) / sizeof(sparse_counts[0]));
    if (occupancy.dispatch_ready || std::strcmp(occupancy.reason, "occupancy") != 0) {
        printf("FAIL: expected sparse grouped occupancy rejection, got ready=%d rows=%zu padded=%zu reason=%s\n",
               occupancy.dispatch_ready ? 1 : 0, occupancy.total_rows, occupancy.padded_rows, occupancy.reason);
        return false;
    }

    const int32_t underfilled_counts[] = { 15, 15, 15, 15 };
    occupancy                          = ggml_sycl_select_mxfp4_grouped_dpas_occupancy(
        caps, underfilled_counts, sizeof(underfilled_counts) / sizeof(underfilled_counts[0]));
    if (occupancy.dispatch_ready || std::strcmp(occupancy.reason, "rows-per-expert") != 0) {
        printf("FAIL: expected underfilled grouped occupancy rejection, got ready=%d max=%zu reason=%s\n",
               occupancy.dispatch_ready ? 1 : 0, occupancy.max_rows, occupancy.reason);
        return false;
    }

    const int32_t pp_counts[] = { 512, 512, 512, 512 };
    occupancy = ggml_sycl_select_mxfp4_grouped_dpas_occupancy(caps, pp_counts, sizeof(pp_counts) / sizeof(pp_counts[0]),
                                                              ggml_sycl_mxfp4_grouped_dpas_row_list_limit(caps));
    if (occupancy.dispatch_ready || std::strcmp(occupancy.reason, "kernel-row-limit") != 0) {
        printf("FAIL: expected PP row-list kernel rejection, got ready=%d rows=%zu limit=%zu reason=%s\n",
               occupancy.dispatch_ready ? 1 : 0, occupancy.total_rows, occupancy.max_total_rows, occupancy.reason);
        return false;
    }

    printf("PASS: grouped DPAS policy is capability/shape/layout/residency driven\n");
    return true;
}

static bool run_moe_device_policy_mock_test() {
    XMXCapabilities full_caps     = test_mxfp4_caps();
    full_caps.compute_units       = 160;
    full_caps.global_mem_size     = 12ull * 1024ull * 1024ull * 1024ull;
    full_caps.max_mem_alloc_size  = 11ull * 1024ull * 1024ull * 1024ull;
    full_caps.supports_usm_device = true;
    full_caps.supports_usm_shared = true;
    full_caps.supports_usm_host   = true;
    full_caps.supports_fp16       = true;
    full_caps.supports_fp16_type  = true;

    auto full_policy = ggml_sycl_make_moe_device_policy(full_caps,
                                                        /*device_id=*/0,
                                                        /*total_vram=*/12ull * 1024ull * 1024ull * 1024ull,
                                                        /*free_vram_at_init=*/11ull * 1024ull * 1024ull * 1024ull,
                                                        /*max_alloc_size=*/11ull * 1024ull * 1024ull * 1024ull,
                                                        /*safe_max_alloc_size=*/10ull * 1024ull * 1024ull * 1024ull,
                                                        /*vram_budget=*/10ull * 1024ull * 1024ull * 1024ull,
                                                        /*weight_budget=*/9ull * 1024ull * 1024ull * 1024ull,
                                                        /*arena_total=*/10ull * 1024ull * 1024ull * 1024ull,
                                                        /*arena_scratch=*/256ull * 1024ull * 1024ull,
                                                        /*arena_runtime=*/512ull * 1024ull * 1024ull,
                                                        /*arena_onednn=*/256ull * 1024ull * 1024ull,
                                                        /*in_dim=*/2880,
                                                        /*out_dim=*/2880,
                                                        /*n_experts=*/128,
                                                        /*device_resident=*/true,
                                                        /*tiled_kernel_validated=*/false);
    if (!full_policy.xmx_int8_candidate || !full_policy.onednn_candidate || !full_policy.cpu_island_candidate ||
        full_policy.mxfp4_device_layout.layout != GGML_LAYOUT_SOA ||
        std::strcmp(full_policy.device_executor, "xmx-mmvq") != 0) {
        printf(
            "FAIL: full mocked policy should select XMX-backed SOA executor, got xmx=%d onednn=%d cpu=%d "
            "layout=%d executor=%s reason=%s\n",
            full_policy.xmx_int8_candidate ? 1 : 0, full_policy.onednn_candidate ? 1 : 0,
            full_policy.cpu_island_candidate ? 1 : 0, (int) full_policy.mxfp4_device_layout.layout,
            full_policy.device_executor, full_policy.executor_reason);
        return false;
    }

    XMXCapabilities limited_caps          = full_caps;
    limited_caps.supported                = false;
    limited_caps.supports_int8            = false;
    limited_caps.supports_fp16            = false;
    limited_caps.supports_fp16_type       = false;
    limited_caps.supports_usm_shared      = false;
    limited_caps.supports_usm_host        = false;
    limited_caps.compute_units            = 32;
    limited_caps.slm_size                 = 16 * 1024;
    limited_caps.sub_group_size_count     = 0;
    limited_caps.max_sub_group_size       = 8;
    limited_caps.preferred_sub_group_size = 8;

    auto limited_policy = ggml_sycl_make_moe_device_policy(limited_caps,
                                                           /*device_id=*/1,
                                                           /*total_vram=*/6ull * 1024ull * 1024ull * 1024ull,
                                                           /*free_vram_at_init=*/5ull * 1024ull * 1024ull * 1024ull,
                                                           /*max_alloc_size=*/2ull * 1024ull * 1024ull * 1024ull,
                                                           /*safe_max_alloc_size=*/1ull * 1024ull * 1024ull * 1024ull,
                                                           /*vram_budget=*/4ull * 1024ull * 1024ull * 1024ull,
                                                           /*weight_budget=*/3ull * 1024ull * 1024ull * 1024ull,
                                                           /*arena_total=*/4ull * 1024ull * 1024ull * 1024ull,
                                                           /*arena_scratch=*/128ull * 1024ull * 1024ull,
                                                           /*arena_runtime=*/128ull * 1024ull * 1024ull,
                                                           /*arena_onednn=*/0,
                                                           /*in_dim=*/4096,
                                                           /*out_dim=*/2880,
                                                           /*n_experts=*/128,
                                                           /*device_resident=*/true,
                                                           /*tiled_kernel_validated=*/true);
    if (limited_policy.xmx_int8_candidate || limited_policy.onednn_candidate || limited_policy.cpu_island_candidate ||
        limited_policy.mxfp4_device_layout.layout != GGML_LAYOUT_COALESCED ||
        std::strcmp(limited_policy.mxfp4_device_layout.reason, "no-xmx-coalesced") != 0) {
        printf(
            "FAIL: limited mocked policy should fall back by queried capabilities, got xmx=%d onednn=%d cpu=%d "
            "layout=%d reason=%s\n",
            limited_policy.xmx_int8_candidate ? 1 : 0, limited_policy.onednn_candidate ? 1 : 0,
            limited_policy.cpu_island_candidate ? 1 : 0, (int) limited_policy.mxfp4_device_layout.layout,
            limited_policy.mxfp4_device_layout.reason);
        return false;
    }

    printf("PASS: MoE device policy derives executor/layout from mocked hardware facts\n");
    return true;
}

static bool run_moe_triplet_planner_test() {
    constexpr int                                     n_experts = 4;
    const std::vector<std::pair<std::string, size_t>> inventory = {
        { "blk.0.ffn_gate_exps.weight", 1024 },
        { "blk.0.ffn_up_exps.weight",   2048 },
        { "blk.0.ffn_down_exps.weight", 3072 },
        { "blk.0.ffn_gate_exps.bias",   256  },
        { "blk.0.ffn_up_exps.bias",     256  },
        { "blk.0.ffn_down_exps.bias",   256  },
    };

    ggml_sycl::placement_kv_info kv_info{};
    const size_t                 bias_bytes               = 3 * 256;
    const size_t                 triplet_bytes_per_expert = 256 + 512 + 768;
    const size_t                 budget                   = bias_bytes + 2 * triplet_bytes_per_expert + 1;
    auto       plan    = ggml_sycl::compute_placement_plan(inventory, budget, 0, kv_info, nullptr, n_experts);
    const auto summary = plan.summarize_expert_placements(1, n_experts);

    if (summary.expected != 12 || summary.planned != summary.expected || summary.duplicates != 0 ||
        summary.unclassified != 0 || summary.missing != 0) {
        printf("FAIL: expected 12 unique expert weight placements, got planned=%zu expected=%zu dup=%zu miss=%zu\n",
               summary.planned, summary.expected, summary.duplicates, summary.missing);
        return false;
    }
    if (!plan.has_dense_entry("blk.0.ffn_gate_exps.bias") || !plan.has_dense_entry("blk.0.ffn_up_exps.bias") ||
        !plan.has_dense_entry("blk.0.ffn_down_exps.bias")) {
        printf("FAIL: expert bias tensors should remain dense planner entries, not per-expert placements\n");
        return false;
    }

    size_t device_triplets = 0;
    size_t host_triplets   = 0;
    for (int e = 0; e < n_experts; ++e) {
        const auto gate = plan.lookup_expert_placement(0, e, ggml_sycl::expert_tensor_role::GATE);
        const auto up   = plan.lookup_expert_placement(0, e, ggml_sycl::expert_tensor_role::UP);
        const auto down = plan.lookup_expert_placement(0, e, ggml_sycl::expert_tensor_role::DOWN);
        if (!gate.found() || !up.found() || !down.found()) {
            printf("FAIL: missing triplet placement for expert %d\n", e);
            return false;
        }
        if (gate.on_device != up.on_device || gate.on_device != down.on_device ||
            gate.target_device != up.target_device || gate.target_device != down.target_device) {
            printf("FAIL: mixed residency in expert %d triplet: gate=%d/%d up=%d/%d down=%d/%d\n", e,
                   gate.on_device ? 1 : 0, gate.target_device, up.on_device ? 1 : 0, up.target_device,
                   down.on_device ? 1 : 0, down.target_device);
            return false;
        }
        if (gate.on_device) {
            device_triplets++;
        } else {
            host_triplets++;
        }
    }
    if (device_triplets != 2 || host_triplets != 2) {
        printf("FAIL: expected 2 device and 2 host triplets, got device=%zu host=%zu\n", device_triplets,
               host_triplets);
        return false;
    }

    const std::vector<std::pair<std::string, size_t>> balanced_inventory = {
        { "blk.0.ffn_gate_exps.weight", 1024 },
        { "blk.0.ffn_up_exps.weight",   2048 },
        { "blk.0.ffn_down_exps.weight", 3072 },
        { "blk.1.ffn_gate_exps.weight", 1024 },
        { "blk.1.ffn_up_exps.weight",   2048 },
        { "blk.1.ffn_down_exps.weight", 3072 },
    };
    ggml_sycl::set_expert_popularity_rank(/*layer_id=*/1, /*expert_id=*/0, /*rank=*/0);
    ggml_sycl::set_expert_popularity_rank(/*layer_id=*/1, /*expert_id=*/1, /*rank=*/1);
    ggml_sycl::set_expert_popularity_rank(/*layer_id=*/0, /*expert_id=*/0, /*rank=*/2);
    ggml_sycl::set_expert_popularity_rank(/*layer_id=*/0, /*expert_id=*/1, /*rank=*/3);

    auto balanced_plan = ggml_sycl::compute_placement_plan(balanced_inventory, 4 * triplet_bytes_per_expert, 0, kv_info,
                                                           nullptr, n_experts);
    auto triplet_on_device = [&](int layer_id, int expert_id) {
        const auto gate =
            balanced_plan.lookup_expert_placement(layer_id, expert_id, ggml_sycl::expert_tensor_role::GATE);
        const auto up = balanced_plan.lookup_expert_placement(layer_id, expert_id, ggml_sycl::expert_tensor_role::UP);
        const auto down =
            balanced_plan.lookup_expert_placement(layer_id, expert_id, ggml_sycl::expert_tensor_role::DOWN);
        return gate.found() && up.found() && down.found() && gate.on_device && up.on_device && down.on_device;
    };
    if (!triplet_on_device(1, 0) || !triplet_on_device(1, 1) || !triplet_on_device(0, 0) || !triplet_on_device(0, 1)) {
        printf("FAIL: expected popularity-aware MoE packing to keep ranked routed experts resident\n");
        return false;
    }
    if (triplet_on_device(0, 2) || triplet_on_device(1, 2) || triplet_on_device(0, 3) || triplet_on_device(1, 3)) {
        printf("FAIL: popularity-aware MoE packing should spill unranked colder experts first\n");
        return false;
    }

    const std::vector<std::pair<std::string, size_t>> unaligned_inventory = {
        { "blk.2.ffn_gate_exps.weight", 257 * 2 },
        { "blk.2.ffn_up_exps.weight",   257 * 2 },
        { "blk.2.ffn_down_exps.weight", 257 * 2 },
    };
    auto   unaligned_plan = ggml_sycl::compute_placement_plan(unaligned_inventory, 257 * 3 * 2, 0, kv_info, nullptr, 2);
    size_t unaligned_device_triplets = 0;
    for (int e = 0; e < 2; ++e) {
        const auto gate = unaligned_plan.lookup_expert_placement(2, e, ggml_sycl::expert_tensor_role::GATE);
        const auto up   = unaligned_plan.lookup_expert_placement(2, e, ggml_sycl::expert_tensor_role::UP);
        const auto down = unaligned_plan.lookup_expert_placement(2, e, ggml_sycl::expert_tensor_role::DOWN);
        if (gate.found() && up.found() && down.found() && gate.on_device && up.on_device && down.on_device) {
            unaligned_device_triplets++;
        }
    }
    if (unaligned_device_triplets != 1) {
        printf("FAIL: planner should charge allocator-rounded VRAM bytes, got %zu resident triplets\n",
               unaligned_device_triplets);
        return false;
    }

    printf("PASS: MoE planner packs gate/up/down as triplets and excludes expert biases\n");
    return true;
}

static bool run_layout_choice_test() {
    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("SKIP: SYCL backend unavailable\n");
        return true;
    }

    ggml_backend_buffer_type_t host_buft = ggml_backend_sycl_host_buffer_type();
    ggml_backend_buffer_type_t dev_buft  = ggml_backend_get_default_buffer_type(backend);
    if (!host_buft || !dev_buft) {
        printf("SKIP: buffer types unavailable\n");
        ggml_backend_free(backend);
        return true;
    }

    ggml_init_params params = {
        16 * 1024 * 1024,
        nullptr,
        true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("FAIL: ggml_init failed\n");
        ggml_backend_free(backend);
        return false;
    }

    const int ncols   = 1024;
    const int nrows   = 128;
    const int ntokens = 1;

    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, ncols, nrows);
    ggml_set_name(weight, "layout_choice_weight");
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, ntokens);
    ggml_set_name(input, "layout_choice_input");
    ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
    ggml_set_name(output, "layout_choice_output");

    const size_t weight_size = ggml_backend_buft_get_alloc_size(host_buft, weight);
    const size_t input_size  = ggml_backend_buft_get_alloc_size(dev_buft, input);
    const size_t output_size = ggml_backend_buft_get_alloc_size(dev_buft, output);

    ggml_backend_buffer_t weight_buf = ggml_backend_buft_alloc_buffer(host_buft, weight_size);
    ggml_backend_buffer_t input_buf  = ggml_backend_buft_alloc_buffer(dev_buft, input_size);
    ggml_backend_buffer_t output_buf = ggml_backend_buft_alloc_buffer(dev_buft, output_size);

    if (!weight_buf || !input_buf || !output_buf) {
        printf("FAIL: buffer allocation failed\n");
        if (weight_buf) {
            ggml_backend_buffer_free(weight_buf);
        }
        if (input_buf) {
            ggml_backend_buffer_free(input_buf);
        }
        if (output_buf) {
            ggml_backend_buffer_free(output_buf);
        }
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    ggml_backend_buffer_set_usage(weight_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_set_usage(input_buf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_buffer_set_usage(output_buf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    ggml_backend_tensor_alloc(weight_buf, weight, ggml_backend_buffer_get_base(weight_buf));
    ggml_backend_tensor_alloc(input_buf, input, ggml_backend_buffer_get_base(input_buf));
    ggml_backend_tensor_alloc(output_buf, output, ggml_backend_buffer_get_base(output_buf));

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev) {
        ggml_backend_sycl_register_host_weight_tensor(dev, weight);
    }

    std::vector<uint8_t> weight_data(ggml_nbytes(weight), 0);
    std::vector<float>   input_data(ncols * ntokens, 0.1f);

    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size());
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    void * pre_cached = ggml_sycl_get_weight_layout_ptr(weight, 0, GGML_LAYOUT_COALESCED);
    if (!pre_cached) {
        printf("SKIP: coalesced layout unavailable, cannot validate purge\n");
        ggml_backend_buffer_free(weight_buf);
        ggml_backend_buffer_free(input_buf);
        ggml_backend_buffer_free(output_buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return true;
    }

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        printf("FAIL: graph compute failed\n");
        ggml_backend_buffer_free(weight_buf);
        ggml_backend_buffer_free(input_buf);
        ggml_backend_buffer_free(output_buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    auto        resolved_layout = ggml_sycl_resolve(weight, 0);
    layout_mode chosen_layout   = resolved_layout ? static_cast<layout_mode>(resolved_layout.layout) : GGML_LAYOUT_AOS;
    if (!resolved_layout) {
        printf("FAIL: missing cache entry for weight after finalize\n");
        ggml_backend_buffer_free(weight_buf);
        ggml_backend_buffer_free(input_buf);
        ggml_backend_buffer_free(output_buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }
    if (chosen_layout != GGML_LAYOUT_SOA) {
        printf("FAIL: expected SoA layout choice, got %d\n", (int) chosen_layout);
        ggml_backend_buffer_free(weight_buf);
        ggml_backend_buffer_free(input_buf);
        ggml_backend_buffer_free(output_buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    sycl::queue &              q     = dpct::dev_mgr::instance().get_device(0).default_queue();
    ggml_sycl::unified_cache * cache = ggml_sycl::get_unified_cache(q);
    if (!cache) {
        printf("SKIP: unified cache unavailable\n");
        ggml_backend_buffer_free(weight_buf);
        ggml_backend_buffer_free(input_buf);
        ggml_backend_buffer_free(output_buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return true;
    }

    ggml_sycl_cache_id key        = ggml_backend_sycl_get_weight_cache_key(weight, 0);
    const bool         soa_cached = key.valid && cache->is_cached(key, GGML_LAYOUT_SOA);
    const bool         coa_cached = key.valid && cache->is_cached(key, GGML_LAYOUT_COALESCED);

    ggml_backend_buffer_free(weight_buf);
    ggml_backend_buffer_free(input_buf);
    ggml_backend_buffer_free(output_buf);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (!soa_cached) {
        printf("FAIL: expected SoA layout cached after finalize\n");
        return false;
    }
    if (coa_cached) {
        printf("FAIL: coalesced layout should be purged after finalize\n");
        return false;
    }

    printf("PASS: layout choice enforced (SoA only)\n");
    return true;
}

int main() {
    if (!run_mxfp4_moe_policy_test()) {
        return 1;
    }
    if (!run_mxfp4_grouped_dpas_policy_test()) {
        return 1;
    }
    if (!run_moe_device_policy_mock_test()) {
        return 1;
    }
    if (!run_moe_triplet_planner_test()) {
        return 1;
    }
    ggml_sycl::test_layout_override_guard guard(GGML_LAYOUT_SOA);
    bool                                  ok = run_layout_choice_test();
    return ok ? 0 : 1;
}
