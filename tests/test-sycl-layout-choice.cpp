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

static ggml_sycl_device_info make_mock_sycl_info(int requested_device_count = 2) {
    ggml_sycl_device_info info{};
    int                   device_count = requested_device_count;
    if (device_count < 1) {
        device_count = 1;
    }
    if (device_count > GGML_SYCL_MAX_DEVICES) {
        device_count = GGML_SYCL_MAX_DEVICES;
    }

    info.device_count        = device_count;
    info.total_gpu_count     = device_count;
    info.host_max_alloc_size = 128ull * 1024ull * 1024ull * 1024ull;

    for (int d = 0; d < device_count; ++d) {
        const size_t total_vram = d == 0 ? 12ull * 1024ull * 1024ull * 1024ull : 16ull * 1024ull * 1024ull * 1024ull;

        sycl_device_info & dev           = info.devices[d];
        dev.cc                           = 1200;
        dev.nsm                          = d == 0 ? 160 : 96;
        dev.smpbo                        = 64ull * 1024ull;
        dev.vmm                          = true;
        dev.total_vram                   = total_vram;
        dev.free_vram_at_init            = total_vram;
        dev.max_alloc_size               = total_vram - 512ull * 1024ull * 1024ull;
        dev.safe_max_alloc_size          = dev.max_alloc_size;
        dev.supports_soa_reorder         = true;
        dev.xmx_caps                     = test_mxfp4_caps();
        dev.xmx_caps.compute_units       = dev.nsm;
        dev.xmx_caps.global_mem_size     = total_vram;
        dev.xmx_caps.max_mem_alloc_size  = dev.max_alloc_size;
        dev.xmx_caps.supports_usm_device = true;
        std::snprintf(dev.device_name, sizeof(dev.device_name), "mock-bmg-xmx-%d", d);

        info.max_work_group_sizes[d] = 256;
        info.default_tensor_split[d] = static_cast<float>(d + 1) / static_cast<float>(device_count);
        info.gpu_dpct_ids[d]         = d;
    }

    for (int src = 0; src < device_count; ++src) {
        for (int dst = 0; dst < device_count; ++dst) {
            sycl_peer_link_info & link   = info.peer_links[src][dst];
            link.valid                   = true;
            link.src_device              = src;
            link.dst_device              = dst;
            link.same_device             = src == dst;
            link.same_backend            = true;
            link.same_platform           = true;
            link.same_sycl_context       = src == dst;
            link.level_zero              = true;
            link.l0_peer_query_supported = true;
            link.l0_can_access_peer      = src == dst;
            std::snprintf(link.preferred_transfer, sizeof(link.preferred_transfer), "%s",
                          src == dst ? "same-device" : "host-bounce");
            std::snprintf(link.unsupported_reason, sizeof(link.unsupported_reason), "%s",
                          src == dst ? "" : "mock-no-p2p");
        }
    }

    return info;
}

static size_t planned_weight_charge_for_device(const ggml_sycl::placement_plan & plan, int device_id) {
    size_t total = 0;
    for (const ggml_sycl::placement_entry & entry : plan.entries) {
        if (!entry.on_device || entry.target_device != device_id) {
            continue;
        }
        total += entry.vram_charge_size != 0 ? entry.vram_charge_size :
                                               ggml_sycl::placement_vram_charge_bytes(entry.dst_size);
        for (const ggml_sycl::placement_alternate_layout & alt : entry.alternate_layouts) {
            const int alt_target = alt.target_device >= 0 ? alt.target_device : entry.target_device;
            if (alt_target != device_id) {
                continue;
            }
            total +=
                alt.vram_charge_size != 0 ? alt.vram_charge_size : ggml_sycl::placement_vram_charge_bytes(alt.dst_size);
        }
    }
    return total;
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

static bool run_multi_device_layer_block_plan_test() {
    constexpr size_t mib = 1024u * 1024u;

    const std::vector<std::pair<std::string, size_t>> inventory = {
        { "blk.0.attn_q.weight", 4u * mib },
        { "blk.1.attn_q.weight", 4u * mib },
    };
    const std::vector<ggml_sycl::device_budget> devices = {
        { 0, 64u * mib, 64u * mib, 1.0, true },
        { 1, 64u * mib, 64u * mib, 4.0, true },
    };

    ggml_sycl::placement_kv_info kv_info{};
    kv_info.n_ubatch = 512;
    auto plan = ggml_sycl::compute_multi_device_plan(devices, inventory, 2, ggml_sycl::multi_gpu_mode::LAYER, kv_info,
                                                     nullptr, 0);

    if (plan.fastest_dense_device != 1 || plan.fastest_dense_score != 4.0) {
        printf("FAIL: expected raw fastest dense device 1, got device=%d score=%.2f\n", plan.fastest_dense_device,
               plan.fastest_dense_score);
        return false;
    }
    if (plan.layer_blocks.empty()) {
        printf("FAIL: expected multi-device planner to emit explicit layer blocks\n");
        return false;
    }
    if (plan.get_layer_device(0) != plan.fastest_dense_device ||
        plan.get_layer_device(1) != plan.fastest_dense_device) {
        printf("FAIL: expected dense layers to prefer fastest dense device %d, got layer0=%d layer1=%d\n",
               plan.fastest_dense_device, plan.get_layer_device(0), plan.get_layer_device(1));
        return false;
    }
    if (plan.layer_blocks.front().start_layer != 0 || plan.layer_blocks.back().end_layer != 1) {
        printf("FAIL: expected layer blocks to cover layers [0,1], got first=%d last=%d\n",
               plan.layer_blocks.front().start_layer, plan.layer_blocks.back().end_layer);
        return false;
    }
    if (plan.layer_blocks.size() != 1 || plan.layer_blocks.front().dense_device != plan.fastest_dense_device ||
        !plan.layer_blocks.front().dense_on_fastest_device) {
        printf("FAIL: expected one fastest-dense layer block, blocks=%zu dense=%d fastest=%d flag=%d\n",
               plan.layer_blocks.size(), plan.layer_blocks.front().dense_device, plan.fastest_dense_device,
               plan.layer_blocks.front().dense_on_fastest_device ? 1 : 0);
        return false;
    }

    printf("PASS: multi-device planner records explicit layer blocks and raw fastest dense device\n");
    return true;
}

static bool run_multi_device_layer_boundary_metadata_test() {
    constexpr size_t   mib        = 1024u * 1024u;
    constexpr int64_t  hidden_dim = 4096;
    constexpr uint32_t tokens     = 512;

    std::vector<ggml_sycl::placement_tensor_info> inventory;
    ggml_sycl::placement_tensor_info              expert_block{ "blk.0.ffn_gate_exps.weight", 1 };
    expert_block.type  = GGML_TYPE_MXFP4;
    expert_block.ne[0] = 32;
    expert_block.ne[1] = hidden_dim;
    inventory.push_back(expert_block);

    ggml_sycl::placement_tensor_info layer0{ "blk.0.attn_q.weight", 4u * mib };
    layer0.type  = GGML_TYPE_F16;
    layer0.ne[0] = hidden_dim;
    layer0.ne[1] = 512;
    inventory.push_back(layer0);

    ggml_sycl::placement_tensor_info layer1{ "blk.1.attn_q.weight", 4u * mib };
    layer1.type  = GGML_TYPE_F16;
    layer1.ne[0] = hidden_dim;
    layer1.ne[1] = 512;
    inventory.push_back(layer1);

    const std::vector<ggml_sycl::device_budget> devices = {
        { 0, 64u * mib, 64u * mib, 1.0, true },
        { 1, 6u * mib,  6u * mib,  4.0, true },
    };

    ggml_sycl::placement_kv_info kv_info{};
    kv_info.n_ubatch = tokens;
    auto plan = ggml_sycl::compute_multi_device_plan(devices, inventory, 2, ggml_sycl::multi_gpu_mode::LAYER, kv_info,
                                                     nullptr, 0);

    if (plan.layer_blocks.size() != 2) {
        printf("FAIL: expected two layer blocks after fastest-device capacity spill, got %zu\n",
               plan.layer_blocks.size());
        return false;
    }
    const auto & first  = plan.layer_blocks[0];
    const auto & second = plan.layer_blocks[1];
    if (first.execution_device == second.execution_device) {
        printf("FAIL: expected cross-device block boundary, got both blocks on %d\n", first.execution_device);
        return false;
    }

    const size_t expected_bytes = static_cast<size_t>(hidden_dim) * static_cast<size_t>(tokens) * sizeof(float);
    if (first.boundary_to_next_bytes != expected_bytes || second.boundary_from_prev_bytes != expected_bytes) {
        printf("FAIL: expected boundary bytes %zu, got out=%zu in=%zu\n", expected_bytes, first.boundary_to_next_bytes,
               second.boundary_from_prev_bytes);
        return false;
    }
    if (first.boundary_batch_tokens != tokens || second.boundary_batch_tokens != tokens ||
        first.boundary_activation_hidden_dim != hidden_dim || second.boundary_activation_hidden_dim != hidden_dim) {
        printf("FAIL: boundary shape metadata mismatch tokens=[%u,%u] hidden=[%lld,%lld]\n",
               first.boundary_batch_tokens, second.boundary_batch_tokens,
               (long long) first.boundary_activation_hidden_dim, (long long) second.boundary_activation_hidden_dim);
        return false;
    }
    if (first.next_execution_device != second.execution_device ||
        second.previous_execution_device != first.execution_device) {
        printf("FAIL: boundary adjacency mismatch first.next=%d second.prev=%d\n", first.next_execution_device,
               second.previous_execution_device);
        return false;
    }
    if (first.boundary_to_next_est_us <= 0.0 || second.boundary_from_prev_est_us <= 0.0) {
        printf("FAIL: expected positive boundary transfer estimate, got out=%.2f in=%.2f\n",
               first.boundary_to_next_est_us, second.boundary_from_prev_est_us);
        return false;
    }

    printf("PASS: multi-device planner records boundary activation metadata and transfer cost\n");
    return true;
}

static bool run_multi_device_moe_i8_executor_support_test() {
    constexpr size_t mib       = 1024u * 1024u;
    constexpr int    n_experts = 2;
    constexpr int    ncols     = 2880;
    constexpr int    nrows     = 2880;

    auto make_mxfp4 = [&](const char * name) {
        ggml_sycl::placement_tensor_info tensor;
        tensor.name  = name;
        tensor.type  = GGML_TYPE_MXFP4;
        tensor.ne[0] = ncols;
        tensor.ne[1] = nrows;
        tensor.ne[2] = n_experts;
        tensor.ne[3] = 1;
        tensor.size =
            ggml_row_size(GGML_TYPE_MXFP4, ncols) * static_cast<size_t>(nrows) * static_cast<size_t>(n_experts);
        return tensor;
    };

    const std::vector<ggml_sycl::placement_tensor_info> inventory = {
        make_mxfp4("blk.0.ffn_gate_exps.weight"),
        make_mxfp4("blk.0.ffn_up_exps.weight"),
        make_mxfp4("blk.0.ffn_down_exps.weight"),
    };
    const std::vector<ggml_sycl::device_budget> devices = {
        { 0, 512u * mib, 512u * mib, 4.0, true },
        { 1, 1u * mib,   1u * mib,   1.0, true },
    };

    ggml_sycl::placement_kv_info kv_info{};
    kv_info.n_ubatch = 512;
    auto plan = ggml_sycl::compute_multi_device_plan(devices, inventory, 1, ggml_sycl::multi_gpu_mode::EXPERT, kv_info,
                                                     nullptr, n_experts);

    for (int e = 0; e < n_experts; ++e) {
        const auto down = plan.lookup_expert_placement(0, e, ggml_sycl::expert_tensor_role::DOWN);
        if (!down.found() || !down.on_device || down.target_device != 0) {
            printf("FAIL: expected down expert %d to stay on primary device0, found=%d on_device=%d target=%d\n", e,
                   down.found() ? 1 : 0, down.on_device ? 1 : 0, down.target_device);
            return false;
        }
        if (down.layout == GGML_LAYOUT_MXFP4_I8) {
            printf("FAIL: multi-device planner must not static-plan MXFP4_I8 down without executor support\n");
            return false;
        }

        const auto gate = plan.lookup_expert_placement(0, e, ggml_sycl::expert_tensor_role::GATE);
        const auto up   = plan.lookup_expert_placement(0, e, ggml_sycl::expert_tensor_role::UP);
        if (!gate.found() || !up.found() || !gate.on_device || !up.on_device || gate.target_device != 0 ||
            up.target_device != 0) {
            printf("FAIL: expected gate/up expert %d to stay on primary device0, gate=%d/%d/%d up=%d/%d/%d\n", e,
                   gate.found() ? 1 : 0, gate.on_device ? 1 : 0, gate.target_device, up.found() ? 1 : 0,
                   up.on_device ? 1 : 0, up.target_device);
            return false;
        }
        if (gate.layout == GGML_LAYOUT_XMX_TILED || up.layout == GGML_LAYOUT_XMX_TILED) {
            printf("FAIL: multi-device planner must not static-plan XMX_TILED gate/up without executor support\n");
            return false;
        }
    }

    printf("PASS: multi-device MoE planner does not advertise I8/XMX layouts without executor support\n");
    return true;
}

static bool run_single_device_moe_pp_complete_soa_layout_test() {
    const auto & info = ggml_sycl_info();
    if (info.device_count <= 0 ||
        !xmx_capabilities_match_int8_tile(info.devices[0].xmx_caps, GGML_SYCL_MXFP4_MOE_XMX_M,
                                          GGML_SYCL_MXFP4_MOE_XMX_N, GGML_SYCL_MXFP4_MOE_XMX_K) ||
        !xmx_capabilities_support_sub_group(info.devices[0].xmx_caps, GGML_SYCL_MXFP4_MOE_XMX_SG)) {
        printf("PASS: single-device XMX PP alternate test skipped on non-XMX device\n");
        return true;
    }

    constexpr size_t mib       = 1024u * 1024u;
    constexpr int    n_experts = 2;
    constexpr int    ncols     = 2880;
    constexpr int    nrows     = 2880;

    auto make_mxfp4 = [&](const char * name) {
        ggml_sycl::placement_tensor_info tensor;
        tensor.name  = name;
        tensor.type  = GGML_TYPE_MXFP4;
        tensor.ne[0] = ncols;
        tensor.ne[1] = nrows;
        tensor.ne[2] = n_experts;
        tensor.ne[3] = 1;
        tensor.size =
            ggml_row_size(GGML_TYPE_MXFP4, ncols) * static_cast<size_t>(nrows) * static_cast<size_t>(n_experts);
        return tensor;
    };

    const std::vector<ggml_sycl::placement_tensor_info> inventory = {
        make_mxfp4("blk.0.ffn_gate_exps.weight"),
        make_mxfp4("blk.0.ffn_up_exps.weight"),
        make_mxfp4("blk.0.ffn_down_exps.weight"),
    };
    const std::vector<ggml_sycl::device_budget> devices = {
        { 0, 1024u * mib, 1024u * mib, 4.0, true },
    };

    ggml_sycl::placement_kv_info kv_info{};
    kv_info.n_ubatch      = 512;
    kv_info.n_expert_used = 4;
    auto plan = ggml_sycl::compute_multi_device_plan(devices, inventory, 1, ggml_sycl::multi_gpu_mode::EXPERT, kv_info,
                                                     nullptr, n_experts);

    auto has_soa_layout_on_target = [](const ggml_sycl::expert_placement_result & placement) {
        if (placement.layout == GGML_LAYOUT_SOA) {
            return true;
        }
        for (const auto & alt : placement.alternate_layouts) {
            const int alt_target = alt.target_device >= 0 ? alt.target_device : placement.target_device;
            if (alt.layout == GGML_LAYOUT_SOA && alt_target == placement.target_device) {
                return true;
            }
        }
        return false;
    };

    for (int e = 0; e < n_experts; ++e) {
        const auto gate = plan.lookup_expert_placement(0, e, ggml_sycl::expert_tensor_role::GATE);
        const auto up   = plan.lookup_expert_placement(0, e, ggml_sycl::expert_tensor_role::UP);
        const auto down = plan.lookup_expert_placement(0, e, ggml_sycl::expert_tensor_role::DOWN);
        if (!gate.found() || !up.found() || !down.found() || !gate.on_device || !up.on_device || !down.on_device ||
            gate.target_device != 0 || up.target_device != 0 || down.target_device != 0) {
            printf("FAIL: expected complete local MoE triplet for expert %d, gate=%d/%d/%d up=%d/%d/%d down=%d/%d/%d\n",
                   e, gate.found() ? 1 : 0, gate.on_device ? 1 : 0, gate.target_device, up.found() ? 1 : 0,
                   up.on_device ? 1 : 0, up.target_device, down.found() ? 1 : 0, down.on_device ? 1 : 0,
                   down.target_device);
            return false;
        }
        if (gate.layout != GGML_LAYOUT_SOA || up.layout != GGML_LAYOUT_SOA) {
            printf("FAIL: prompt gate/up expert %d must use PP-safe SOA primaries, gate=%d up=%d\n", e,
                   (int) gate.layout, (int) up.layout);
            return false;
        }
        if (!has_soa_layout_on_target(gate) || !has_soa_layout_on_target(up) || !has_soa_layout_on_target(down)) {
            printf("FAIL: expert %d is missing complete SOA PP executable coverage, gate=%d up=%d down=%d\n", e,
                   (int) gate.layout, (int) up.layout, (int) down.layout);
            return false;
        }
        if (down.layout != GGML_LAYOUT_MXFP4_I8) {
            printf("FAIL: down expert %d should retain MXFP4_I8 TG primary after reserving SOA PP alternate, got %d\n",
                   e, (int) down.layout);
            return false;
        }
    }
    if (!plan.moe_pp_soa_promoted) {
        printf("FAIL: planner should record PP SOA promotion for prompt-incompatible gate/up primaries\n");
        return false;
    }
    const size_t planned_charge = planned_weight_charge_for_device(plan, 0);
    if (plan.weight_vram_bytes != planned_charge || plan.vram_bytes != planned_charge) {
        printf("FAIL: planner under/over-counted PP-safe MoE layout charge, plan weight=%zu vram=%zu actual=%zu\n",
               plan.weight_vram_bytes, plan.vram_bytes, planned_charge);
        return false;
    }
    if (planned_charge > devices[0].vram_budget) {
        printf("FAIL: planner charged beyond device budget, charged=%zu budget=%zu\n", planned_charge,
               devices[0].vram_budget);
        return false;
    }

    const std::vector<ggml_sycl::device_budget> tight_devices = {
        { 0, planned_charge - mib, planned_charge - mib, 4.0, true },
    };
    auto tight_plan = ggml_sycl::compute_multi_device_plan(
        tight_devices, inventory, 1, ggml_sycl::multi_gpu_mode::EXPERT, kv_info, nullptr, n_experts);
    const size_t tight_charge = planned_weight_charge_for_device(tight_plan, 0);
    if (tight_plan.weight_vram_bytes != tight_charge || tight_plan.vram_bytes != tight_charge) {
        printf(
            "FAIL: tight-budget planner under/over-counted PP-safe MoE layout charge, plan weight=%zu vram=%zu "
            "actual=%zu\n",
            tight_plan.weight_vram_bytes, tight_plan.vram_bytes, tight_charge);
        return false;
    }
    if (tight_charge > tight_devices[0].vram_budget) {
        printf("FAIL: tight-budget planner exceeded budget, charged=%zu budget=%zu\n", tight_charge,
               tight_devices[0].vram_budget);
        return false;
    }

    ggml_tensor down_tensor{};
    down_tensor.type  = GGML_TYPE_MXFP4;
    down_tensor.ne[0] = ncols;
    down_tensor.ne[1] = nrows;
    down_tensor.ne[2] = n_experts;
    down_tensor.ne[3] = 1;
    ggml_set_name(&down_tensor, "blk.0.ffn_down_exps.weight");

    struct probe_override_guard {
        ~probe_override_guard() { ggml_sycl::test_clear_moe_planned_layout_probe_overrides(); }
    } probe_guard;

    ggml_sycl::test_clear_moe_planned_layout_probe_overrides();
    ggml_sycl::test_set_moe_planned_layout_probe_override(&down_tensor, 0, GGML_LAYOUT_MXFP4_I8, n_experts, 0, 0, 0);
    if (ggml_sycl::test_prompt_down_specialized_layout_proven(&down_tensor, 0, GGML_LAYOUT_MXFP4_I8,
                                                              /*n_tokens=*/512)) {
        printf("FAIL: prompt-down I8 should not be proven without complete SOA fallback coverage\n");
        return false;
    }
    if (ggml_sycl::test_moe_layout_for_selected_rows(&down_tensor, 0, GGML_LAYOUT_MXFP4_I8,
                                                     /*selected_rows=*/512, /*exact_override=*/false,
                                                     /*n_tokens=*/512) != GGML_LAYOUT_SOA) {
        printf("FAIL: prompt-down I8 should fall back to SOA until specialized and SOA layouts are complete\n");
        return false;
    }

    ggml_sycl::test_set_moe_planned_layout_probe_override(&down_tensor, 0, GGML_LAYOUT_SOA, n_experts, 0, 0, 0);
    if (!ggml_sycl::test_prompt_down_specialized_layout_proven(&down_tensor, 0, GGML_LAYOUT_MXFP4_I8,
                                                               /*n_tokens=*/512)) {
        printf("FAIL: prompt-down I8 should be proven when I8 and SOA coverage are both complete\n");
        return false;
    }
    if (ggml_sycl::test_moe_layout_for_selected_rows(&down_tensor, 0, GGML_LAYOUT_MXFP4_I8,
                                                     /*selected_rows=*/512, /*exact_override=*/false,
                                                     /*n_tokens=*/512) != GGML_LAYOUT_MXFP4_I8) {
        printf("FAIL: prompt-down I8 should remain selected after proof-gated SOA fallback coverage is complete\n");
        return false;
    }

    ggml_sycl::test_clear_moe_planned_layout_probe_overrides();
    ggml_sycl::test_set_moe_planned_layout_probe_override(&down_tensor, 0, GGML_LAYOUT_MXFP4_DPAS, n_experts, 0, 0, 0);
    if (ggml_sycl::test_prompt_down_specialized_layout_proven(&down_tensor, 0, GGML_LAYOUT_MXFP4_DPAS,
                                                              /*n_tokens=*/512)) {
        printf("FAIL: prompt-down DPAS should not be proven without complete SOA fallback coverage\n");
        return false;
    }
    if (ggml_sycl::test_moe_layout_for_selected_rows(&down_tensor, 0, GGML_LAYOUT_MXFP4_DPAS,
                                                     /*selected_rows=*/512, /*exact_override=*/false,
                                                     /*n_tokens=*/512) != GGML_LAYOUT_SOA) {
        printf("FAIL: prompt-down DPAS should fall back to SOA until specialized and SOA layouts are complete\n");
        return false;
    }

    ggml_sycl::test_set_moe_planned_layout_probe_override(&down_tensor, 0, GGML_LAYOUT_SOA, n_experts, 0, 0, 0);
    if (!ggml_sycl::test_prompt_down_specialized_layout_proven(&down_tensor, 0, GGML_LAYOUT_MXFP4_DPAS,
                                                               /*n_tokens=*/512)) {
        printf("FAIL: prompt-down DPAS should be proven when DPAS and SOA coverage are both complete\n");
        return false;
    }
    if (ggml_sycl::test_moe_layout_for_selected_rows(&down_tensor, 0, GGML_LAYOUT_MXFP4_DPAS,
                                                     /*selected_rows=*/512, /*exact_override=*/false,
                                                     /*n_tokens=*/512) != GGML_LAYOUT_MXFP4_DPAS) {
        printf("FAIL: prompt-down DPAS should remain selected after proof-gated SOA fallback coverage is complete\n");
        return false;
    }

    printf("PASS: single-device MoE planner budgets and proves complete PP-safe SOA executable layouts\n");
    return true;
}

static bool run_multi_device_no_p2p_cohesive_moe_layer_test() {
    constexpr size_t mib       = 1024u * 1024u;
    constexpr int    n_experts = 2;
    constexpr int    ncols     = 4096;
    constexpr int    nrows     = 1928;

    auto make_dense = [](const char * name) {
        constexpr int hidden_dim = 4096;
        constexpr int tokens     = 512;

        ggml_sycl::placement_tensor_info tensor;
        tensor.name  = name;
        tensor.type  = GGML_TYPE_F16;
        tensor.ne[0] = hidden_dim;
        tensor.ne[1] = tokens;
        tensor.size  = ggml_row_size(GGML_TYPE_F16, hidden_dim) * static_cast<size_t>(tokens);
        return tensor;
    };

    auto make_mxfp4 = [&](const char * name) {
        ggml_sycl::placement_tensor_info tensor;
        tensor.name  = name;
        tensor.type  = GGML_TYPE_MXFP4;
        tensor.ne[0] = ncols;
        tensor.ne[1] = nrows;
        tensor.ne[2] = n_experts;
        tensor.ne[3] = 1;
        tensor.size =
            ggml_row_size(GGML_TYPE_MXFP4, ncols) * static_cast<size_t>(nrows) * static_cast<size_t>(n_experts);
        return tensor;
    };

    const std::vector<ggml_sycl::placement_tensor_info> inventory = {
        make_dense("blk.0.attn_q.weight"),      make_mxfp4("blk.0.ffn_gate_exps.weight"),
        make_mxfp4("blk.0.ffn_up_exps.weight"), make_mxfp4("blk.0.ffn_down_exps.weight"),
        make_dense("blk.1.attn_q.weight"),      make_mxfp4("blk.1.ffn_gate_exps.weight"),
        make_mxfp4("blk.1.ffn_up_exps.weight"), make_mxfp4("blk.1.ffn_down_exps.weight"),
    };
    const std::vector<ggml_sycl::device_budget> devices = {
        { 0, 128u * mib, 128u * mib, 1.0, true },
        { 1, 40u * mib,  40u * mib,  4.0, true },
    };

    ggml_sycl::placement_kv_info kv_info{};
    kv_info.n_ubatch = 512;
    auto plan = ggml_sycl::compute_multi_device_plan(devices, inventory, 2, ggml_sycl::multi_gpu_mode::EXPERT, kv_info,
                                                     nullptr, n_experts);

    if (plan.fastest_dense_device != 1) {
        printf("FAIL: expected fastest dense device 1, got %d\n", plan.fastest_dense_device);
        return false;
    }
    if (plan.get_layer_device(0) != 0 || plan.get_layer_device(1) != 0) {
        printf("FAIL: expected no-P2P cohesive layer set to stay on primary [0,0], got [%d,%d]\n",
               plan.get_layer_device(0), plan.get_layer_device(1));
        return false;
    }
    if (plan.layer_blocks.size() != 1 || plan.layer_blocks[0].dense_device != 0) {
        printf("FAIL: expected one primary cohesive layer block for no-P2P fit, blocks=%zu device=%d\n",
               plan.layer_blocks.size(), plan.layer_blocks.empty() ? -1 : plan.layer_blocks[0].dense_device);
        return false;
    }

    for (int layer = 0; layer < 2; ++layer) {
        const int expected_device = plan.get_layer_device(layer);
        for (int expert = 0; expert < n_experts; ++expert) {
            for (ggml_sycl::expert_tensor_role role :
                 { ggml_sycl::expert_tensor_role::GATE, ggml_sycl::expert_tensor_role::UP,
                   ggml_sycl::expert_tensor_role::DOWN }) {
                const auto placement = plan.lookup_expert_placement(layer, expert, role);
                if (!placement.found() || !placement.on_device || placement.target_device != expected_device) {
                    printf("FAIL: layer %d expert %d role %d expected target %d, found=%d on_device=%d target=%d\n",
                           layer, expert, static_cast<int>(role), expected_device, placement.found() ? 1 : 0,
                           placement.on_device ? 1 : 0, placement.target_device);
                    return false;
                }
            }
        }
    }

    printf("PASS: no-P2P multi-device MoE planner keeps dense/KV/experts cohesive by layer block\n");
    return true;
}

static bool run_regression_guard_policy_test() {
    if (ggml_sycl::test_moe_primary_uses_expert_handles(
            /*planner_primary_fastpath_guard_active=*/false,
            /*planner_has_moe_plan_for_primary=*/true,
            /*is_moe_expert_weight=*/true)) {
        printf("FAIL: single-GPU MoE should keep direct resolve instead of forcing expert handles\n");
        return false;
    }
    if (!ggml_sycl::test_moe_primary_uses_expert_handles(
            /*planner_primary_fastpath_guard_active=*/true,
            /*planner_has_moe_plan_for_primary=*/true,
            /*is_moe_expert_weight=*/true)) {
        printf("FAIL: planner-owned multi-device MoE should use expert handles\n");
        return false;
    }
    if (ggml_sycl::test_moe_primary_uses_expert_handles(
            /*planner_primary_fastpath_guard_active=*/true,
            /*planner_has_moe_plan_for_primary=*/false,
            /*is_moe_expert_weight=*/true)) {
        printf("FAIL: MoE expert handles require an active primary plan\n");
        return false;
    }
    if (ggml_sycl::test_moe_primary_uses_expert_handles(
            /*planner_primary_fastpath_guard_active=*/true,
            /*planner_has_moe_plan_for_primary=*/true,
            /*is_moe_expert_weight=*/false)) {
        printf("FAIL: non-MoE tensors should not use MoE expert handles\n");
        return false;
    }

    constexpr size_t mib                  = 1024ull * 1024ull;
    constexpr size_t b50_total            = 16304ull * mib;
    constexpr size_t caller_reserved      = 565ull * mib;
    constexpr size_t safe_headroom        = 576ull * mib;
    constexpr size_t b50_public_budget    = b50_total - caller_reserved;
    const size_t     b50_full_headroom    = b50_total / 10;
    const size_t     b50_clamped_headroom = ggml_sycl::test_arena_external_headroom_bytes(b50_total, b50_public_budget);
    if (b50_clamped_headroom != safe_headroom) {
        printf("FAIL: arena headroom should raise undersized caller slack to the safe floor, got %zu expected %zu\n",
               b50_clamped_headroom, safe_headroom);
        return false;
    }
    const size_t b50_default_headroom = ggml_sycl::test_arena_external_headroom_bytes(b50_total, b50_total);
    if (b50_default_headroom != b50_full_headroom) {
        printf("FAIL: arena headroom should use proportional B50 default, got %zu expected %zu\n", b50_default_headroom,
               b50_full_headroom);
        return false;
    }
    constexpr size_t b580_total         = 12288ull * mib;
    constexpr size_t b580_public_budget = b580_total - 512ull * mib;
    const size_t     b580_headroom      = ggml_sycl::test_arena_external_headroom_bytes(b580_total, b580_public_budget);
    if (b580_headroom != safe_headroom) {
        printf("FAIL: arena headroom should not leave only the exact graph scratch floor, got %zu expected %zu\n",
               b580_headroom, safe_headroom);
        return false;
    }
    if (ggml_sycl::test_arena_external_headroom_bytes(0, b50_total) != 0) {
        printf("FAIL: zero VRAM total should opt out of arena headroom capping\n");
        return false;
    }

    constexpr size_t weight_slot = 64ull * mib;
    constexpr size_t act_slot    = 16ull * mib;
    constexpr size_t out_slot    = 32ull * mib;
    if (ggml_sycl::test_pp_moe_onednn_runtime_ring_depth(0) != 1) {
        printf("FAIL: missing PP MoE oneDNN scratch plan should fall back to one serialized slot\n");
        return false;
    }
    if (ggml_sycl::test_pp_moe_onednn_runtime_ring_depth(2) != 2) {
        printf("FAIL: explicit PP MoE oneDNN scratch plan should be preserved\n");
        return false;
    }
    if (ggml_sycl::test_pp_moe_onednn_effective_ring_depth(4) != 1) {
        printf("FAIL: serialized PP MoE oneDNN scratch should budget one reusable slot\n");
        return false;
    }
    const size_t planned_pp_moe_scratch =
        ggml_sycl::test_pp_moe_onednn_planned_scratch_bytes(weight_slot, act_slot, out_slot, 4);
    const size_t expected_pp_moe_scratch = weight_slot + act_slot + out_slot;
    if (planned_pp_moe_scratch != expected_pp_moe_scratch) {
        printf("FAIL: PP MoE oneDNN scratch should not reserve unused future slots, got %zu expected %zu\n",
               planned_pp_moe_scratch, expected_pp_moe_scratch);
        return false;
    }

    printf("PASS: SYCL regression guards preserve single-GPU MoE resolve and arena headroom\n");
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
    {
        const ggml_sycl_device_info              mock_info = make_mock_sycl_info();
        ggml_sycl::test_sycl_info_override_guard info_guard(mock_info);

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
        if (!run_multi_device_layer_block_plan_test()) {
            return 1;
        }
        if (!run_multi_device_layer_boundary_metadata_test()) {
            return 1;
        }
        if (!run_multi_device_moe_i8_executor_support_test()) {
            return 1;
        }
        if (!run_single_device_moe_pp_complete_soa_layout_test()) {
            return 1;
        }
        if (!run_multi_device_no_p2p_cohesive_moe_layer_test()) {
            return 1;
        }
        if (!run_regression_guard_policy_test()) {
            return 1;
        }
    }

    const char * run_backend_layout_test = std::getenv("GGML_SYCL_TEST_LAYOUT_CHOICE_BACKEND");
    if (!run_backend_layout_test || std::atoi(run_backend_layout_test) == 0) {
        printf("SKIP: backend layout choice purge requires GGML_SYCL_TEST_LAYOUT_CHOICE_BACKEND=1\n");
        return 0;
    }
    if (!std::getenv("GGML_SYCL_VRAM_BUDGET_PCT")) {
        setenv("GGML_SYCL_VRAM_BUDGET_PCT", "20", 0);
    }
    ggml_sycl::test_layout_override_guard guard(GGML_LAYOUT_SOA);
    bool                                  ok = run_layout_choice_test();
    return ok ? 0 : 1;
}
