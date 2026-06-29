#pragma once

#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>

struct ggml_sycl_device_info;

namespace ggml_sycl {
struct placement_plan;

// Test-only layout override hooks (no env handling inside the library).
// Use the guard in tests to temporarily force a layout during a scoped operation.
void               test_set_layout_override(ggml_layout_mode layout);
void               test_clear_layout_override();
bool               test_get_layout_override(ggml_layout_mode * out);
void               test_clear_host_weight_registry();
bool               test_backend_supports_graphs(ggml_backend_t backend);
bool               test_backend_graphs_disabled(ggml_backend_t backend);
bool               test_backend_has_exec_graph(ggml_backend_t backend);
size_t             test_graph_pinned_entry_count(ggml_backend_t backend);
uint64_t           test_backend_graph_replay_count(ggml_backend_t backend);
bool               test_graph_recording_uses_gpu_only_dispatch();
bool               test_backend_graph_recording_uses_gpu_only_dispatch(ggml_backend_t backend);
bool               test_moe_route_preserves_ready_event_for_chaining();
bool               test_moe_ptr_table_retains_route_lease_until_event();
bool               test_moe_ptr_table_cached_reuse_retains_lease_and_ready_event();
bool               test_moe_ptr_table_cached_reuse_is_tensor_specific();
bool               test_moe_ptr_table_does_not_persist_pointer_cache();
bool               test_moe_ptr_table_lease_covers_populated_slots();
constexpr uint32_t TEST_GRAPH_BOUNDARY_DRAIN_RETAINED = 1u << 0;
constexpr uint32_t TEST_GRAPH_BOUNDARY_CLEAR_ACTIVE   = 1u << 1;
constexpr uint32_t TEST_GRAPH_BOUNDARY_RESET_ARENAS   = 1u << 2;
uint32_t           test_graph_boundary_policy_actions(bool active_graph_valid, bool same_phase, bool graph_key_matches);
int                test_moe_block_graphlet_requested_size_from_env(const char * enabled_env, const char * size_env);
bool               test_moe_block_graphlet_bulk_xmx_phase_disabled_from_env(const char * enabled_env,
                                                                            const char * bulk_xmx_env,
                                                                            const char * phase_materialize_env);
size_t             test_layout_bytes(const ggml_tensor * tensor, ggml_layout_mode layout, int device);
const char *       test_layout_name(ggml_layout_mode layout);
void               test_set_moe_planned_layout_probe_override(const ggml_tensor * tensor,
                                                              int                 device,
                                                              ggml_layout_mode    layout,
                                                              size_t              local,
                                                              size_t              secondary,
                                                              size_t              host,
                                                              size_t              missing);
void               test_clear_moe_planned_layout_probe_overrides();
bool               test_prompt_down_specialized_layout_proven(const ggml_tensor * tensor,
                                                              int                 device,
                                                              ggml_layout_mode    layout,
                                                              int64_t             n_tokens);
bool               test_moe_primary_uses_expert_handles(bool planner_primary_fastpath_guard_active,
                                                        bool planner_has_moe_plan_for_primary,
                                                        bool is_moe_expert_weight);
uint32_t           test_pp_moe_onednn_runtime_ring_depth(uint32_t planned_ring_depth);
bool               test_xmx_moe_allow_unsafe_pp_from_env(const char * unsafe_env, const char * legacy_env);

struct test_moe_default_fast_path_policy_input {
    bool default_fast_path_enabled = false;
    bool decode_phase = false;
    bool has_limited_graph = false;
    bool safe_baseline_enabled = false;
    bool sequence_identity_stable = false;
    bool aggregation_available = true;
    bool fusion_metadata_complete = false;
    bool fusion_kernel_proven = false;
    bool unsafe_fused_q8_requested = false;
    bool context_quarantined = false;
};

struct test_moe_default_fast_path_policy_result {
    bool         attempt_sequence = false;
    bool         attempt_fusion   = false;
    const char * reason           = "default-fast-disabled";
    const char * selected_path    = "baseline-fallback";
};

test_moe_default_fast_path_policy_result test_moe_default_fast_path_policy(
    const test_moe_default_fast_path_policy_input & in);

struct test_moe_down_sum_direct_final_input {
    bool             env_enabled                        = false;
    bool             dpas_direct_final_env_enabled                  = false;
    bool             dpas_direct_final_i8_enabled                   = false;
    bool             dpas_direct_final_dpas_enabled                 = false;
    bool             dpas_direct_final_rank_parallel_atomic_enabled = false;
    bool             dpas_direct_final_scratch_reduce_enabled       = false;
    bool             dpas_direct_final_same_expert_grouped_enabled  = false;
    int64_t          n_tokens                                       = 0;
    ggml_layout_mode down_layout                        = GGML_LAYOUT_AOS;
    bool             cached_q8_direct_ids               = false;
    bool             has_ids_host                       = false;
    int64_t          ids_host_count                     = 0;
    int64_t          down_entries                       = 0;
    bool             has_weighted_dst                   = false;
    bool             has_moe_weights                    = false;
    bool             has_down_sum_final                 = false;
    int              down_sum_final_index               = -1;
    bool             glu_handle_device                  = false;
    bool             final_handle_device                = false;
    bool             down_weight_resident               = false;
    bool             fused_q8_quarantined               = false;
    bool             q8_capacity_ok                     = true;
    bool             device_xmx_ok                      = true;
    bool             token_major_deterministic_metadata        = false;
    int64_t          token_major_entries                       = 0;
    int64_t          token_major_expected_entries              = 0;
    bool             token_major_weights_valid                 = false;
    bool             same_expert_grouping_metadata             = false;
    bool             same_expert_grouping_has_lane_filled_group = false;
};

struct test_moe_down_sum_direct_final_result {
    bool         accepted                = false;
    const char * reason                  = "none";
    int          expected_saved_launches = 0;
    const char * selected_variant        = "none";
};

struct test_moe_down_sum_direct_final_counters {
    uint64_t candidates   = 0;
    uint64_t accepted     = 0;
    uint64_t rejected     = 0;
    uint64_t submit_saved = 0;
};

test_moe_down_sum_direct_final_result test_moe_down_sum_direct_final_policy(
    const test_moe_down_sum_direct_final_input & in);
void                                    test_moe_down_sum_direct_final_counters_reset();
test_moe_down_sum_direct_final_counters test_moe_down_sum_direct_final_counters_snapshot();
void                                    test_moe_down_sum_direct_final_record_submit_saved_for_test();

struct test_moe_token_major_metadata_entry {
    int32_t token  = -1;
    int32_t rank   = -1;
    int32_t expert = -1;
    int32_t slot   = -1;
    float   weight = 0.0f;
};

struct test_moe_token_major_metadata_input {
    const int32_t *                       ids             = nullptr;
    const float *                         weights         = nullptr;
    int64_t                               n_ids           = 0;
    int64_t                               n_tokens        = 0;
    int64_t                               n_experts       = 0;
    int64_t                               ids_nb0         = sizeof(int32_t);
    int64_t                               ids_nb1         = 0;
    int64_t                               weights_nb1     = sizeof(float);
    int64_t                               weights_nb2     = 0;
    test_moe_token_major_metadata_entry * out_entries     = nullptr;
    int64_t                               out_capacity    = 0;
    bool                                  require_weights = true;
};

struct test_moe_token_major_metadata_result {
    bool         accepted  = false;
    const char * reason    = "none";
    int64_t      entries   = 0;
    uint64_t     signature = 0;
};

test_moe_token_major_metadata_result test_moe_build_token_major_metadata(
    const test_moe_token_major_metadata_input & in);

struct test_moe_same_expert_group {
    int32_t token  = -1;
    int32_t expert = -1;
    int32_t offset = 0;
    int32_t count  = 0;
};

struct test_moe_same_expert_grouping_input {
    const test_moe_token_major_metadata_entry * entries            = nullptr;
    int64_t                                     entry_count        = 0;
    int64_t                                     expected_entries   = 0;
    int64_t                                     n_tokens           = 0;
    int64_t                                     n_experts          = 0;
    int32_t                                     max_lanes          = 0;
    test_moe_same_expert_group *                out_groups         = nullptr;
    int64_t                                     out_group_capacity = 0;
    int32_t *                                   out_entry_indices  = nullptr;
    int64_t                                     out_entry_capacity = 0;
};

struct test_moe_same_expert_grouping_result {
    bool         accepted              = false;
    const char * reason                = "none";
    int64_t      groups                = 0;
    int64_t      grouped_entries       = 0;
    int32_t      max_group_lanes       = 0;
    bool         has_lane_filled_group = false;
    uint64_t     signature             = 0;
};

test_moe_same_expert_grouping_result test_moe_build_same_expert_grouping(
    const test_moe_same_expert_grouping_input & in);

struct test_moe_direct_final_scratch_plan_input {
    int64_t n_tokens         = 0;
    int64_t n_ids            = 0;
    int64_t nrows_per_expert = 0;
    size_t  element_size     = 0;
};

struct test_moe_direct_final_scratch_plan_result {
    bool         accepted        = false;
    const char * reason          = "none";
    size_t       bytes           = 0;
    bool         requires_reduce = false;
};

test_moe_direct_final_scratch_plan_result test_moe_direct_final_scratch_plan(
    const test_moe_direct_final_scratch_plan_input & in);

struct test_moe_glu_q8_artifact_input {
    bool fused_candidate    = false;
    bool kernel_event_set   = false;
    bool fused_store_ok     = false;
    bool glu_row_contiguous = false;
    bool publish_ok         = false;
    bool handle_valid       = false;
    bool layer_match        = false;
    bool dims_match         = false;
    bool layout_soa         = false;
    bool ready_event_set    = false;
};

struct test_moe_glu_q8_artifact_result {
    bool         accepted                = false;
    const char * action                  = "none";
    const char * reason                  = "none";
    bool         invalidate_artifact     = false;
    bool         cached_down_allowed     = false;
    bool         completion_event_set    = false;
    int          expected_saved_launches = 0;
};

struct test_moe_glu_q8_counters {
    uint64_t candidates    = 0;
    uint64_t fused_store   = 0;
    uint64_t publish       = 0;
    uint64_t invalidated   = 0;
    uint64_t cached_allow  = 0;
    uint64_t cached_reject = 0;
};

test_moe_glu_q8_artifact_result test_moe_glu_q8_artifact_policy(const test_moe_glu_q8_artifact_input & in);
void                            test_moe_glu_q8_counters_reset();
test_moe_glu_q8_counters        test_moe_glu_q8_counters_snapshot();

enum class test_moe_glu_q8_kernel_path {
    NONE,
    SPLIT_SG16,
    DIRECT_Q8,
    PACKED_Q8_M2,
    GROUPED_DIRECT_Q8,
    GROUPED_PACKED_Q8_M2,
};

struct test_moe_glu_q8_fused_store_input {
    bool                        env_enabled       = false;
    bool                        artifact_enabled  = false;
    int64_t                     n_tokens          = 0;
    bool                        glu_row_contig    = false;
    bool                        glu_handle_valid  = false;
    bool                        glu_handle_device = false;
    bool                        kernel_event_set  = false;
    size_t                      q8_capacity_bytes = 0;
    size_t                      q8_required_bytes = 0;
    test_moe_glu_q8_kernel_path kernel_path       = test_moe_glu_q8_kernel_path::NONE;
    bool                        kernel_writes_q8  = false;
};

struct test_moe_glu_q8_fused_store_result {
    bool         accepted                = false;
    const char * reason                  = "none";
    int          expected_saved_launches = 0;
};

struct test_moe_glu_q8_fused_store_counters {
    uint64_t candidates = 0;
    uint64_t accepted   = 0;
    uint64_t rejected   = 0;
};

test_moe_glu_q8_fused_store_result   test_moe_glu_q8_fused_store_policy(const test_moe_glu_q8_fused_store_input & in);
void                                 test_moe_glu_q8_fused_store_counters_reset();
test_moe_glu_q8_fused_store_counters test_moe_glu_q8_fused_store_counters_snapshot();

size_t           test_arena_external_headroom_bytes(size_t device_total_vram, size_t budget_bytes);
uint32_t         test_pp_moe_onednn_effective_ring_depth(uint32_t requested_ring_depth);
size_t           test_pp_moe_onednn_planned_scratch_bytes(size_t   weight_slot_bytes,
                                                          size_t   activation_slot_bytes,
                                                          size_t   output_slot_bytes,
                                                          uint32_t requested_ring_depth);
ggml_layout_mode test_moe_layout_for_selected_rows(const ggml_tensor * tensor,
                                                   int                 device,
                                                   ggml_layout_mode    layout,
                                                   size_t              selected_rows,
                                                   bool                exact_override,
                                                   int64_t             n_tokens);

struct test_moe_decode_down_layout_policy_input {
    bool             valid_mxfp4_down                  = true;
    bool             selected_dpas_materialize_enabled = false;
    bool             effective_layout_is_dpas          = false;
    bool             explicit_decode_dpas_enabled      = false;
    bool             decode_i8_candidate               = false;
    bool             dpas_shape_eligible               = true;
    ggml_layout_mode adjusted_dpas_layout              = GGML_LAYOUT_MXFP4_DPAS;
    int64_t          n_tokens                          = 1;
};

struct test_moe_decode_down_layout_policy_result {
    ggml_layout_mode layout = GGML_LAYOUT_SOA;
    const char *     reason = "decode-soa-default";
};

test_moe_decode_down_layout_policy_result test_moe_decode_down_layout_policy(
    const test_moe_decode_down_layout_policy_input & in);
void             test_reset_orchestrator_call_count();
int              test_get_orchestrator_call_count();
int              test_physical_device_count();
void             test_set_kv_placement_plan(const placement_plan & plan, uint32_t n_layers, size_t kv_per_layer);
void             test_clear_kv_placement_plan();
bool             test_cache_replacement_allowed_for_test(uint32_t live_leases, bool retired);
void             test_set_sycl_info_override(const ggml_sycl_device_info & info);
void             test_clear_sycl_info_override();

inline ggml_sycl_cache_id test_make_cache_id(const void * tag, uint64_t model_id = 1) {
    ggml_sycl_cache_id id{};
    id.valid    = true;
    id.model_id = model_id;
    id.aux_id   = reinterpret_cast<uintptr_t>(tag);
    return id;
}

struct test_layout_override_guard {
    explicit test_layout_override_guard(ggml_layout_mode layout) { test_set_layout_override(layout); }

    ~test_layout_override_guard() { test_clear_layout_override(); }
};

struct test_sycl_info_override_guard {
    explicit test_sycl_info_override_guard(const ggml_sycl_device_info & info) { test_set_sycl_info_override(info); }

    ~test_sycl_info_override_guard() { test_clear_sycl_info_override(); }
};

}  // namespace ggml_sycl

// =============================================================================
// Blind preload threshold functions (for testing)
// =============================================================================
// Get the blind preload threshold (cached, reads env var once)
int ggml_sycl_get_blind_preload_threshold();

// Check if blind preload should be skipped for given expert count
// Returns true if should SKIP blind preload (expert count exceeds threshold)
bool ggml_sycl_should_skip_blind_preload(int64_t n_experts);
