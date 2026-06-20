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
size_t             test_arena_external_headroom_bytes(size_t device_total_vram, size_t budget_bytes);
uint32_t           test_pp_moe_onednn_effective_ring_depth(uint32_t requested_ring_depth);
size_t             test_pp_moe_onednn_planned_scratch_bytes(size_t   weight_slot_bytes,
                                                            size_t   activation_slot_bytes,
                                                            size_t   output_slot_bytes,
                                                            uint32_t requested_ring_depth);
ggml_layout_mode   test_moe_layout_for_selected_rows(const ggml_tensor * tensor,
                                                     int                 device,
                                                     ggml_layout_mode    layout,
                                                     size_t              selected_rows,
                                                     bool                exact_override,
                                                     int64_t             n_tokens);
void               test_reset_orchestrator_call_count();
int                test_get_orchestrator_call_count();
int                test_physical_device_count();
void               test_set_kv_placement_plan(const placement_plan & plan, uint32_t n_layers, size_t kv_per_layer);
void               test_clear_kv_placement_plan();
void               test_set_sycl_info_override(const ggml_sycl_device_info & info);
void               test_clear_sycl_info_override();

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
