#pragma once

#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml.h"
#include "unified-cache.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

struct ggml_sycl_device_info;

namespace ggml_sycl {
struct placement_plan;

// Constructs a stable synthetic WEIGHT lease without probing cache/device
// state. Test-only: production ownership paths must use canonical factories.
mem_handle test_make_stable_weight_lease(const ggml_sycl_cache_id & key,
                                         int                        device,
                                         void *                     ptr,
                                         ggml_layout_mode           layout,
                                         bool                       on_device,
                                         std::shared_ptr<void>      storage_owner);
bool       test_moe_resolved_batch_accepts_actual_planned_alternate(mem_handle lease);

// ggml_sycl_plan_has_cpu_work() on a synthetic plan: true when the CPU would
// execute any part of the graph (llama.cpp-38af).
bool test_plan_has_cpu_work(const placement_plan & plan);
// The clause of that predicate which fired, by the name the "[SYCL-CPU-ACT] plan CPU work:" line prints, or nullptr
// when the plan leaves the CPU no work.
const char * test_plan_cpu_work_reason(const placement_plan & plan);

// Mutable controls are declared only for private direct-source fixtures.
#if defined(GGML_SYCL_PRIVATE_TESTING)
void test_set_layout_override(ggml_layout_mode layout);
void test_clear_layout_override();
bool test_get_layout_override(ggml_layout_mode * out);
// Sets the model-load flag the SYCL_Host buffer type reads to choose its role and lifetime.
void test_set_in_model_load(bool loading);
#endif
void               test_clear_host_weight_registry();
bool               test_backend_supports_graphs(ggml_backend_t backend);
bool               test_backend_graphs_disabled(ggml_backend_t backend);
bool               test_backend_has_exec_graph(ggml_backend_t backend);
size_t             test_graph_pinned_entry_count(ggml_backend_t backend);
uint64_t           test_backend_graph_replay_count(ggml_backend_t backend);
bool               test_graph_recording_uses_gpu_only_dispatch();
bool               test_backend_graph_recording_uses_gpu_only_dispatch(ggml_backend_t backend);
bool               test_moe_route_preserves_ready_event_for_chaining();
bool               test_exact_wrapper_owner_matches(uint64_t wrapper_model_id, uint64_t expected_model_id);
bool               test_moe_ptr_table_retains_route_lease_until_event();
bool               test_moe_ptr_table_cached_reuse_retains_lease_and_ready_event();
bool               test_moe_ptr_table_cached_reuse_is_tensor_specific();
bool               test_moe_ptr_table_does_not_persist_pointer_cache();
bool               test_moe_ptr_table_lease_covers_populated_slots();
bool               test_moe_ptr_table_dispatch_bundle_retains_table_compact_missing();

#if defined(GGML_SYCL_PRIVATE_TESTING)
struct moe_resolved_batch;
struct moe_resolved_batch_result;

// The staged ring route (llama.cpp-o9uw). insert_staged puts one entry in the backend's stream-phase stage table
// (making the phase when it has none), read by the staged route, or by the batched prompt executor when
// executor_consumer is set (llama.cpp-3460); staged_ring_build runs the resolver on the backend's device and queue as a
// prompt MUL_MAT_ID does; staged_route_counts, staged_route_handed_off and staged_route_refusals read the phase's tally
// of the resolver's lookups (the refusals as the trace prints them); admission_entry_reusable is the prompt admission
// cache's reuse predicate.
bool                      test_moe_stream_insert_staged(ggml_backend_t     backend,
                                                        int                layer,
                                                        int                expert_id,
                                                        expert_tensor_role role,
                                                        mem_handle         aos,
                                                        sycl::event        ready,
                                                        ggml_layout_mode   layout,
                                                        bool               executor_consumer = false);
bool test_moe_stream_begin_ubatch(ggml_backend_t backend, uint32_t n_tokens);
bool test_moe_stream_end_prompt(ggml_backend_t backend);
bool test_moe_stream_staged_route_counts(ggml_backend_t backend, size_t * served, size_t * refused);
bool test_moe_stream_staged_route_handed_off(ggml_backend_t backend, size_t * handed_off);
bool test_moe_stream_staged_route_refusals(ggml_backend_t backend, std::string * refusals);
moe_resolved_batch_result test_moe_staged_ring_build(ggml_backend_t      backend,
                                                     const ggml_tensor * src0,
                                                     const int32_t *     ids,
                                                     size_t              count,
                                                     size_t              slots_per_token,
                                                     ggml_layout_mode    requested_layout);
bool                      test_moe_admission_entry_reusable(bool     valid,
                                                            bool     names_staged_ring,
                                                            uint64_t entry_graph_seq,
                                                            uint64_t graph_seq);
// A staged slot's lifetime (llama.cpp-o9uw). reserve_ring gives the backend's phase (making it when it has none) a ring
// of n_slots slots of slot_bytes, in two regions, its barriers on the backend's queue, as the first claimed layer's
// ring_ready() reserves one; acquire_slot takes a slot of it as stage_window() does; staged_ring_upload uploads
// `batch`'s pointer table for src0 as a prompt MUL_MAT_ID does, recording each payload slot it gives a staged pointer;
// payload_records is the number of weights the backend holds a record for. test_moe_stream_end_prompt gives the ring
// back.
bool                      test_moe_stream_reserve_ring(ggml_backend_t backend, size_t slot_bytes, uint32_t n_slots);
mem_handle                test_moe_stream_acquire_slot(ggml_backend_t backend, ggml_layout_mode layout);
const void * const *      test_moe_staged_ring_upload(ggml_backend_t             backend,
                                                      const ggml_tensor *        src0,
                                                      const moe_resolved_batch & batch,
                                                      ggml_layout_mode           layout);
size_t                    test_moe_stream_payload_records(ggml_backend_t backend);
#endif

// Exact test seam for the staged IDs ABI. The pointer, allocation lease, ready
// event, and normalized strides are one receipt and must be consumed together.
struct test_moe_ids_device_receipt {
    const int32_t * ptr = nullptr;
    mem_handle      handle;
    sycl::event     ready_event;
    int64_t         nb0 = 0;
    int64_t         nb1 = 0;
};
test_moe_ids_device_receipt test_stage_moe_ids_device(const ggml_tensor * ids, int device);

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

// Read-only: the two layout records on a tensor's SYCL extra. reorder_record names what a kernel last resolved for
// the tensor (optimized_feature.get_reorder(): "none", "soa", "coalesced", "xmx_coalesced" or "xmx_gemm_tiled"),
// which may be a cache copy elsewhere; own_range_layout is the order of the bytes in the tensor's own range (the
// extra's own_data_layout). Without an extra, has_extra is false and the other fields keep their defaults.
struct test_tensor_layout_records {
    bool             has_extra        = false;
    const char *     reorder_record   = "none";
    ggml_layout_mode own_range_layout = GGML_LAYOUT_AOS;
};

test_tensor_layout_records test_get_tensor_layout_records(const ggml_tensor * tensor);
#if defined(GGML_SYCL_PRIVATE_TESTING)
void test_set_moe_planned_layout_probe_override(const ggml_tensor * tensor,
                                                int                 device,
                                                ggml_layout_mode    layout,
                                                size_t              local,
                                                size_t              secondary,
                                                size_t              host,
                                                size_t              missing);
void test_clear_moe_planned_layout_probe_overrides();
#endif
bool               test_prompt_down_specialized_layout_proven(const ggml_tensor * tensor,
                                                              int                 device,
                                                              ggml_layout_mode    layout,
                                                              int64_t             n_tokens);
bool               test_moe_primary_uses_expert_handles(bool planner_primary_fastpath_guard_active,
                                                        bool planner_has_moe_plan_for_primary,
                                                        bool is_moe_expert_weight);
uint32_t           test_pp_moe_onednn_runtime_ring_depth(uint32_t planned_ring_depth);
bool               test_xmx_moe_allow_unsafe_pp_from_env(const char * unsafe_env, const char * legacy_env);

struct test_moe_single_xmx_planner_input {
    const char *       single_xmx_env               = nullptr;
    expert_tensor_role role                         = expert_tensor_role::UNKNOWN;
    ggml_type          type                         = GGML_TYPE_COUNT;
    ggml_layout_mode   current_layout               = GGML_LAYOUT_AOS;
    bool               device_resident              = false;
    bool               device_xmx_int8_ok           = false;
    bool               shape_aligned                = false;
    size_t             pp_rows                      = 0;
    bool               pp_xmx_supported             = false;
    bool               tg_xmx_supported             = false;
    bool               current_default_wants_pp_soa = false;
};

struct test_moe_single_xmx_planner_result {
    bool         single_xmx_selected = false;
    bool         needs_pp_soa        = true;
    bool         adds_soa_alternate  = true;
    const char * reason              = "env";
};

test_moe_single_xmx_planner_result test_moe_single_xmx_planner_decision(const test_moe_single_xmx_planner_input & in);
bool test_moe_single_xmx_pp_runtime_optins_from_env(const char * force_env,
                                                    const char * unsafe_env,
                                                    const char * legacy_env);

struct test_moe_xmx_tiled_materialization_input {
    int              n_experts                 = 0;
    ggml_layout_mode expected_layout           = GGML_LAYOUT_AOS;
    int              handles_present           = 0;
    int              handles_device_resident   = 0;
    int              handles_matching_layout   = 0;
    int              ready_events_present      = 0;
    bool             single_xmx_mode           = false;
    bool             materialization_succeeded = false;
};

struct test_moe_xmx_tiled_materialization_result {
    bool complete              = false;
    bool release_soa_after_xmx = false;
    int  reason_code           = 0;
};

test_moe_xmx_tiled_materialization_result test_moe_xmx_tiled_materialization_invariants(
    const test_moe_xmx_tiled_materialization_input & in);

struct test_moe_phase_xmx_auto_policy_input {
    const char * phase_materialize_env = nullptr;
    const char * bulk_xmx_env          = nullptr;
    bool         has_placement_plan    = false;
    bool         pp_soa_promoted       = false;
    bool         xmx_supported         = false;
    bool         xmx_int8_supported    = false;
};

struct test_moe_phase_xmx_auto_policy_result {
    bool         materialization_enabled = false;
    bool         bulk_xmx_enabled        = false;
    const char * reason                  = "capability";
};

test_moe_phase_xmx_auto_policy_result test_moe_phase_xmx_auto_policy(const test_moe_phase_xmx_auto_policy_input & in);

bool test_moe_single_xmx_chunked_fallback_policy(const char *     env_value,
                                                 const char *     tensor_name,
                                                 ggml_layout_mode target_layout,
                                                 bool             bulk_materialization_ok);
bool test_moe_single_xmx_allows_packed_q8(const char * env_value, bool packed_q8_default_enabled);

struct test_moe_default_fast_path_policy_input {
    bool default_fast_path_enabled = false;
    bool decode_phase              = false;
    bool has_limited_graph         = false;
    bool safe_baseline_enabled     = false;
    bool sequence_identity_stable  = false;
    bool aggregation_available     = true;
    bool fusion_metadata_complete  = false;
    bool fusion_kernel_proven      = false;
    bool unsafe_fused_q8_requested = false;
    bool context_quarantined       = false;
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
    bool             env_enabled                                    = false;
    bool             dpas_direct_final_env_enabled                  = false;
    bool             dpas_direct_final_i8_enabled                   = false;
    bool             dpas_direct_final_dpas_enabled                 = false;
    bool             dpas_direct_final_rank_parallel_atomic_enabled = false;
    bool             dpas_direct_final_scratch_reduce_enabled       = false;
    bool             dpas_direct_final_same_expert_grouped_enabled  = false;
    int64_t          n_tokens                                       = 0;
    ggml_layout_mode down_layout                                    = GGML_LAYOUT_AOS;
    bool             cached_q8_direct_ids                           = false;
    bool             has_ids_host                                   = false;
    int64_t          ids_host_count                                 = 0;
    int64_t          down_entries                                   = 0;
    bool             has_weighted_dst                               = false;
    bool             has_moe_weights                                = false;
    bool             has_down_sum_final                             = false;
    int              down_sum_final_index                           = -1;
    bool             glu_handle_device                              = false;
    bool             final_handle_device                            = false;
    bool             down_weight_resident                           = false;
    bool             fused_q8_quarantined                           = false;
    bool             q8_capacity_ok                                 = true;
    bool             device_xmx_ok                                  = true;
    bool             token_major_deterministic_metadata             = false;
    int64_t          token_major_entries                            = 0;
    int64_t          token_major_expected_entries                   = 0;
    bool             token_major_weights_valid                      = false;
    bool             same_expert_grouping_metadata                  = false;
    bool             same_expert_grouping_has_lane_filled_group     = false;
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

struct test_moe_gateup_singlecol_policy_input {
    bool env_enabled         = false;
    bool is_tg               = false;
    bool packed_q8_m2_route  = false;
    bool has_gate_up_handles = false;
    bool graph_recording     = false;
};

struct test_moe_gateup_singlecol_policy_result {
    bool         accepted = false;
    const char * reason   = "none";
};

test_moe_gateup_singlecol_policy_result test_moe_gateup_singlecol_policy(
    const test_moe_gateup_singlecol_policy_input & in);

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

struct test_moe_gateup_prepack_policy_input {
    const char * prepack_env            = nullptr;
    int64_t      ne12                   = 0;
    int          layer                  = -1;
    int64_t      selected_entries       = 0;
    int64_t      selected_batches       = 0;
    bool         metadata_complete      = false;
    bool         metadata_deterministic = false;
    bool         gate_handle_valid      = false;
    bool         gate_handle_device     = false;
    bool         up_handle_valid        = false;
    bool         up_handle_device       = false;
    bool         scratch_handle_valid   = false;
    size_t       scratch_required_bytes = 0;
    size_t       scratch_capacity_bytes = 0;
    bool         graph_recording        = false;
};

struct test_moe_gateup_prepack_policy_result {
    bool         accepted       = false;
    const char * reason         = "none";
    const char * selected_route = "fallback";
};

struct test_moe_gateup_prepack_policy_counters {
    uint64_t candidates = 0;
    uint64_t accepted   = 0;
    uint64_t rejected   = 0;
};

const char *                            test_moe_gateup_prepack_env_name();
bool                                    test_moe_gateup_prepack_enabled_from_env(const char * env);
test_moe_gateup_prepack_policy_result   test_moe_gateup_prepack_policy(const test_moe_gateup_prepack_policy_input & in);
void                                    test_moe_gateup_prepack_policy_counters_reset();
test_moe_gateup_prepack_policy_counters test_moe_gateup_prepack_policy_counters_snapshot();

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
#if defined(GGML_SYCL_PRIVATE_TESTING)
void test_reset_orchestrator_call_count();
int  test_get_orchestrator_call_count();
bool test_plan_publication_prepare_failure_is_caught();
bool test_provisional_placement_id_exhaustion_is_caught();
void test_set_kv_placement_plan(const placement_plan & plan, uint32_t n_layers, size_t kv_per_layer);
void test_clear_kv_placement_plan();
// The executor-deciding latch read (override-aware), the process latch itself, and the
// writer's plan condition.
bool test_moe_multi_gpu_for_executor();
bool test_moe_multi_gpu_latch();
bool test_moe_multi_gpu_wanted(const placement_plan & plan);
// The stage's refusal of a cross-card expert plan, given whether the cards have a peer path, and the peer-path
// answer the stage passes it for a set of devices, read from the device info's peer-link records.
bool test_cross_card_expert_plan_refused(const placement_plan & plan, bool peer_path);
bool test_cards_have_peer_path(const std::vector<int> & devices);
void test_set_sycl_info_override(const ggml_sycl_device_info & info);
void test_clear_sycl_info_override();
// Forces the planner's XMX_TILED grouped-DPAS PP route predicate on or off (its env read is latched per process).
// The setter returns the override it replaced (-1 none, 0 off, 1 on), for
// test_restore_xmx_tiled_pp_route_override() to put back.
int  test_set_xmx_tiled_pp_route_override(bool active);
void test_restore_xmx_tiled_pp_route_override(int previous);
// Forces ggml_sycl_moe_mmid_route_reachable_in_build() (defined in ggml-sycl.cpp) on or off. The setter returns the
// override it replaced (-1 none, 0 closed, 1 open), for test_restore_moe_mmid_route_reachable_override() to put back.
int  test_set_moe_mmid_route_reachable_override(bool reachable);
void test_restore_moe_mmid_route_reachable_override(int previous);
#endif
int test_physical_device_count();
#if defined(GGML_SYCL_PRIVATE_TESTING)
bool test_cache_replacement_allowed_for_test(uint32_t live_leases, bool retired);
#endif

inline ggml_sycl_cache_id test_make_cache_id(const void * tag, uint64_t model_id = 1) {
    ggml_sycl_cache_id id{};
    id.valid    = true;
    id.model_id = model_id;
    id.aux_id   = reinterpret_cast<uintptr_t>(tag);
    return id;
}

#if defined(GGML_SYCL_PRIVATE_TESTING)
struct test_layout_override_guard {
    explicit test_layout_override_guard(ggml_layout_mode layout) { test_set_layout_override(layout); }
    ~test_layout_override_guard() { test_clear_layout_override(); }
};

struct test_sycl_info_override_guard {
    explicit test_sycl_info_override_guard(const ggml_sycl_device_info & info) { test_set_sycl_info_override(info); }
    ~test_sycl_info_override_guard() { test_clear_sycl_info_override(); }
};

// Restores the override it replaced, so guards nest.
struct test_xmx_tiled_pp_route_override_guard {
    explicit test_xmx_tiled_pp_route_override_guard(bool active) :
        previous(test_set_xmx_tiled_pp_route_override(active)) {}

    ~test_xmx_tiled_pp_route_override_guard() { test_restore_xmx_tiled_pp_route_override(previous); }

    test_xmx_tiled_pp_route_override_guard(const test_xmx_tiled_pp_route_override_guard &)             = delete;
    test_xmx_tiled_pp_route_override_guard & operator=(const test_xmx_tiled_pp_route_override_guard &) = delete;

    const int previous;
};

// Restores the override it replaced, so guards nest.
struct test_moe_mmid_route_reachable_override_guard {
    explicit test_moe_mmid_route_reachable_override_guard(bool reachable) :
        previous(test_set_moe_mmid_route_reachable_override(reachable)) {}

    ~test_moe_mmid_route_reachable_override_guard() { test_restore_moe_mmid_route_reachable_override(previous); }

    test_moe_mmid_route_reachable_override_guard(const test_moe_mmid_route_reachable_override_guard &) = delete;
    test_moe_mmid_route_reachable_override_guard & operator=(const test_moe_mmid_route_reachable_override_guard &) =
        delete;

    const int previous;
};
#endif

#if defined(GGML_SYCL_PRIVATE_TESTING)
// The plan dry run's record of one plan (th32 §4.2): every term and residency line it prints for `plan` under
// `index`, and the result it fills. A hand-built plan reaches backing mixes and charges a shaped inventory does not.
ggml_sycl_plan_dry_run_plan test_plan_dry_run_record(uint32_t index, const placement_plan & plan);
#endif

#if defined(GGML_SYCL_PRIVATE_TESTING)
// The expert loan transaction (llama.cpp-mwmr) on synthetic MoE tensors staged on logical device 0. Each returns an
// empty string on a pass and otherwise names the first failed assertion. The shared-storage case reserves a WEIGHT
// arena, so run it last: with an arena the other cases' experts become zone sub-allocations instead of raw device rows.
// Without host zones the loan refuses by name and lends nothing (llama.cpp-sif1). Host zones cannot be unconfigured
// and every other case configures them, so run it first.
std::string test_expert_loan_refuses_without_host_zones();
std::string test_expert_loan_lends_two_triplets();
std::string test_expert_loan_probe_changes_nothing();
std::string test_expert_loan_refuses_replacing_a_host_entry();
std::string test_expert_loan_refuses_shared_storage();
// The refill's design evidence (llama.cpp-sif1): outside a load, under a published plan and a planned materialization
// scope, direct staging takes one expert from a pinned host copy, and its copy takes the staged branch. It reserves the
// WEIGHT arena itself.
std::string test_expert_direct_stage_outside_a_load_copies_through_staging();
// The refill (llama.cpp-sif1): after the loan, two calls re-admit one triplet each, the bytes and the device's zone
// room come back and no host entry is left; a loan between two calls cancels the second; two boundaries after one
// decode graph re-admit once; a lent triplet no boundary could prepare leaves the ledger, and a full zone keeps its;
// two triplets per call both move, and the third stays lent.
std::string test_expert_refill_readmits_one_triplet_per_call();
std::string test_expert_refill_cancelled_by_a_loan();
std::string test_expert_refill_runs_once_per_decode_graph();
std::string test_expert_refill_drops_an_unpreparable_triplet();
std::string test_expert_refill_moves_two_triplets_per_call();
// The loan and the refill size their copies from the tensor, never from a padded cache entry: the loan's host copy
// holds the expert's own bytes, and the refill's device copy is the staged size with its tail zero-filled. Host
// storage that cannot hold the bytes, and an expert longer than its device entry, are refused.
std::string test_expert_loan_and_refill_size_from_the_tensor();
// With no entry to copy from, a loan stops "not resident" and a refill "not lent", the latter only at the
// ledger's back, and either moves and publishes nothing (llama.cpp-pu8v).
std::string test_expert_loan_and_refill_stop_on_a_missing_entry();
// A loan leaves no lease on a withdrawn device entry in the pointer tables a prompt graph's preload built, and marks
// each such table stale, one whose leases are the entry's only view included: the loan's own boundary drains the views
// it dropped, so the one deferred-free pass after it erases the entry and returns its zone room, and the first decode
// boundary re-admits it (llama.cpp-6aj4).
std::string test_expert_loan_reclaims_pointer_table_leases();
// The same loan with the boundary's drain timed out on a held drain worker: the entry survives that boundary's pass,
// and the pass after the worker goes erases it and returns its room (llama.cpp-6aj4).
std::string test_expert_loan_reclaims_after_a_timed_out_boundary_drain();
// A prompt whose ring did not engage returns its whole loan at the first decode graph's boundary, in one transaction,
// with one WARN that names the position (llama.cpp-i4ny).
std::string test_expert_refill_returns_an_unengaged_prompts_loan_at_once();
// A prompt whose ring engaged owes no bulk refill: the first decode boundary is the per-graph refill alone.
std::string test_expert_refill_keeps_the_per_graph_refill_after_an_engaged_prompt();
// A context that ends with its loan lent owes it to the next context: its free copies nothing and its report says
// how many it owes, and the next context's first graph entry returns it in one transaction (llama.cpp-i4ny).
std::string test_expert_refill_returns_a_loan_an_ended_context_owes();
// The bulk refill with WEIGHT zone room for one of three lent triplets re-admits that one with one "1 of 3" line; with
// room for none it is refused with no line. Either way the debt is spent, and once the room is back the per-graph
// refill re-admits the rest, one per decode graph, with no second bulk line (llama.cpp-i4ny).
std::string test_expert_refill_returns_a_partial_bulk_refills_leftovers_per_graph();
std::string test_expert_refill_returns_a_refused_bulk_refills_loan_per_graph();
// On a plan that maps files, a loan publishes its lent members PINNED, the pinned copy it made, and a refill publishes
// them FILE_MAPPED again; on a plan that maps nothing they read PINNED throughout.
std::string test_expert_loan_on_a_mapping_plan_publishes_its_lent_members_as_pinned();
// The named move's admit, on the same fixture with each host triplet copied to the pinned pool as S1-PRELOAD copies
// one. A padded width's tail reads zero in storage poisoned before the admit, and the payload is the expert's bytes; a
// member served from a file mapping refuses its triplet and leaves it as it was, while a triplet named after it moves;
// a loan outstanding and a WEIGHT zone with room for one member are BUSY and leave no state; the host zone gets the
// moved copies' bytes back once their retired entries go; and the admit that completes a block is read from the card
// through the selected-ids pointer table built right after it, behind the copy's ready event, and through the decode
// direct route's table, which refused the block before the admit and serves it after.
std::string test_expert_admit_zero_fills_a_padded_tail();
std::string test_expert_admit_refuses_a_file_mapped_member();
std::string test_expert_admit_is_busy_while_lent_or_the_zone_is_full();
std::string test_expert_admit_returns_the_host_zone_bytes();
std::string test_expert_admit_completes_a_block_read_through_its_table();
// An admit sizes from the tensor, never the host copy: from a host copy longer than the expert it copies only the
// tensor's bytes, mints their staged size, zero-fills the tail and records the tensor's bytes; a host copy shorter
// than the expert is refused "host copy size" and leaves the triplet as it was.
std::string test_expert_admit_reads_the_tensor_bytes_from_a_longer_host_copy();
std::string test_expert_admit_refuses_a_host_copy_shorter_than_the_tensor();
std::string test_expert_admit_completes_a_block_the_direct_route_serves();
// The named move's evict, on the same fixture. A resident triplet moves to pinned host storage through the loan's
// cache body, writes no ledger, so the refill never re-admits it, and its device entries go only when a deferred-free
// pass reaps them. A bulk-staged triplet is REFUSED "shared storage" while one staged per expert moves in the same
// call; right after an admit, or with a loan outstanding, an evict is BUSY and changes nothing; a swap keeps the
// block's resident count and, once reaped, the zone's room; evict and admit cycles keep no storage on either side, and
// each cycle's evict shows its copies in the host zone and its reaped room in the WEIGHT zone; a refused evict leaves
// the admit beside it to move; an evict's device room is never an admit's in the same call; and an evict out of a
// block the decode direct route served moves the tensor's storage generation, so the route refuses the block instead
// of its old table.
std::string test_expert_evict_moves_a_resident_triplet();
std::string test_expert_evict_is_not_refilled();
std::string test_expert_evict_refuses_shared_storage();
std::string test_expert_evict_is_busy_right_after_an_admit();
std::string test_expert_evict_is_busy_while_lent();
std::string test_expert_evict_and_admit_swap_in_one_call();
std::string test_expert_evict_and_admit_cycles_return_the_room();
std::string test_expert_evict_refusal_leaves_the_admit_list_alone();
std::string test_expert_evict_room_is_not_an_admit_in_the_same_call();
std::string test_expert_evict_takes_a_block_off_the_direct_route();
// A named evict's host copies never grow the pinned host zone: into a full host WEIGHT zone it is BUSY "host pinned
// pool exhausted" and the zone's capacity is unchanged; once the room is back the same evict moves.
std::string test_expert_evict_without_host_room_is_busy_and_keeps_the_zone();
// A move hands each member's id to the copy it publishes before it retires the entry the id named, so no retire and
// no reaping walks the cache's entries for an entry to hand the id to, and a loan's rollback hands it back.
std::string test_expert_evict_walks_no_entry_to_hand_over_its_id();
std::string test_expert_loan_rollback_hands_the_id_back_to_the_device_entry();
// A move's retires leave the cache's eviction latch clear, since the move's publication bumps the exec-graph epoch: a
// named evict, an admit and their reaps set nothing, while a drop of an expert's device entry outside a move latches.
std::string test_expert_moves_leave_the_eviction_latch_clear();
// A move whose plan swap loses publishes nothing and bumps no epoch, so the transaction sets the eviction latch as it
// returns.
std::string test_expert_move_whose_swap_lost_sets_the_eviction_latch();
// A host copy registered in place of a device entry whose mirror cannot be published hands the id it took back to that
// entry, and the same registration without the failure maps the id to the copy.
std::string test_expert_failed_moved_copy_hands_its_id_back();
// Each copy a move publishes belongs to the model that owned the entry it replaces, so that model's release reclaims
// it: an admitted device entry, an evicted triplet's host copy, and a refilled device entry after its loan. An entry
// no model ever owned stays through the release. A copy made while its source's load is still pending takes that
// load, so the load's end tags it.
std::string test_expert_admit_belongs_to_the_host_copys_model();
std::string test_expert_evict_host_copy_belongs_to_the_device_entrys_model();
std::string test_expert_evict_host_copy_takes_the_sources_pending_load();
std::string test_expert_refill_belongs_to_the_lending_model();
#endif

}  // namespace ggml_sycl

int         ggml_sycl_moe_down_q8_dpas_tile_from_env(const char * env);
int         ggml_sycl_moe_down_q8_dpas_tile_active_from_env(const char * env,
                                                            bool         is_down_role,
                                                            ggml_layout_mode down_layout,
                                                            int64_t      n_tokens);
const char * ggml_sycl_moe_down_q8_dpas_tile_label(int tile_rows);

// =============================================================================
// Blind preload threshold functions (for testing)
// =============================================================================
// Get the blind preload threshold (cached, reads env var once)
int ggml_sycl_get_blind_preload_threshold();

// Check if blind preload should be skipped for given expert count
// Returns true if should SKIP blind preload (expert count exceeds threshold)
bool ggml_sycl_should_skip_blind_preload(int64_t n_experts);

#if defined(GGML_SYCL_PRIVATE_TESTING)
// test-sycl-seg-refusal-cleanup's seams in ggml-sycl.cpp (docs/design/sycl-mmap-host-tier.md §4.4, §7 C4). The
// refusal is forced at the armed node through ggml_sycl::host_dispatch_fail_closed, the one function a real refusal
// throws from, and the report records what the design's preconditions and assertions read. Identities are the
// executables' addresses, compared only while the test holds the context.
#    define GGML_SYCL_TEST_SEG_IDS_MAX 8

struct ggml_sycl_test_seg_seam {
    // The refusal, and the context's state the seam read just before its throw.
    bool         fired;
    size_t       weight_leases;  // graph_weight_leases.size()
    size_t       moe_leases;     // graph_moe_expert_leases.size()
    bool         exec_graph;     // exec_graph non-null
    bool         active_valid;   // active_exec_graph.valid
    // The note taken at a recorded segment's moe_after site, just before its helper call; the last one taken.
    bool         noted;
    size_t       seg_idx;
    int          moe_after;
    bool         submitted;           // segment_submitted
    int          last_recorded_node;  // recording_node_index
    size_t       n_recorded;          // recorded_segments.size()
    const void * seg_exec;            // recorded_segments.back().exec_graph
    size_t       n_recorded_ids;      // the non-null executables recorded so far, in order
    const void * recorded_ids[GGML_SYCL_TEST_SEG_IDS_MAX];
    // moe_segments.size() at the fallback catch's first statement (s1), the cleanup step's entry (s2) and its exit
    // (s3); -1 until taken. s1's executables too.
    long         s1;
    long         s2;
    long         s3;
    size_t       n_s1_ids;
    const void * s1_ids[GGML_SYCL_TEST_SEG_IDS_MAX];
    // The record function's entries since the arm, and the context's segments at the last one.
    size_t       record_entries;
    size_t       entry_segments;
    bool         entry_disabled;  // moe_graphs_disabled
    size_t       n_entry_ids;
    const void * entry_ids[GGML_SYCL_TEST_SEG_IDS_MAX];
    // The retires the retire seam failed since the arm.
    int          retire_hits;
};

struct ggml_sycl_test_seg_state {
    bool         recording;        // g_ggml_sycl_graph_recording
    bool         graph_ptr;        // g_recording_graph_ptr non-null
    bool         queue_ptr;        // g_recording_queue_ptr non-null
    bool         sink;             // a retained-handle sink attached
    bool         queue_recording;  // the context's queue is in the recording state
    size_t       weight_leases;
    size_t       moe_leases;
    bool         exec_graph;
    bool         active_valid;
    bool         rerecord;                      // moe_graph_rerecord
    bool         moe_graphs_disabled;
    uint64_t     optional_layout_epoch;         // the context's
    uint64_t     global_optional_layout_epoch;  // g_ggml_sycl_optional_layout_epoch
    size_t       n_segments;                    // moe_segments.size()
    size_t       n_ids;
    const void * ids[GGML_SYCL_TEST_SEG_IDS_MAX];
};

// Arms the refusal at the node named node_name when the helper runs it in `role` (null disarms), and the samples.
// Every call clears the report.
extern "C" void ggml_backend_sycl_test_seg_seam_arm(const char * node_name, const char * role, bool samples);
extern "C" void ggml_backend_sycl_test_seg_seam_read(ggml_sycl_test_seg_seam * out);
// While on, moe_graph_retention_retire_exact fails on every call, and each failure counts in retire_hits.
extern "C" void ggml_backend_sycl_test_seg_retire_fail(bool on);
// Sets active_exec_graph.valid for the prompt phase with exec_graph left null; false when exec_graph is set.
extern "C" bool ggml_backend_sycl_test_seg_set_active_valid_alone(ggml_backend_t backend);
extern "C" bool ggml_backend_sycl_test_seg_state_read(ggml_backend_t backend, ggml_sycl_test_seg_state * out);
#endif
