#pragma once

#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml.h"

#include <cstdint>

namespace ggml_sycl {
struct placement_plan;

// Test-only layout override hooks (no env handling inside the library).
// Use the guard in tests to temporarily force a layout during a scoped operation.
void         test_set_layout_override(ggml_layout_mode layout);
void         test_clear_layout_override();
bool         test_get_layout_override(ggml_layout_mode * out);
void         test_clear_host_weight_registry();
bool         test_backend_supports_graphs(ggml_backend_t backend);
bool         test_backend_graphs_disabled(ggml_backend_t backend);
bool         test_backend_has_exec_graph(ggml_backend_t backend);
size_t       test_graph_pinned_entry_count(ggml_backend_t backend);
uint64_t     test_backend_graph_replay_count(ggml_backend_t backend);
bool         test_graph_recording_uses_gpu_only_dispatch();
bool         test_backend_graph_recording_uses_gpu_only_dispatch(ggml_backend_t backend);
bool         test_moe_route_preserves_ready_event_for_chaining();
bool         test_moe_ptr_table_retains_route_lease_until_event();
bool         test_moe_ptr_table_cached_reuse_retains_lease_and_ready_event();
bool         test_moe_ptr_table_cached_reuse_is_tensor_specific();
bool         test_moe_ptr_table_does_not_persist_pointer_cache();
bool         test_moe_ptr_table_lease_covers_populated_slots();
size_t       test_layout_bytes(const ggml_tensor * tensor, ggml_layout_mode layout, int device);
const char * test_layout_name(ggml_layout_mode layout);
void         test_reset_orchestrator_call_count();
int          test_get_orchestrator_call_count();
int          test_physical_device_count();
void         test_set_kv_placement_plan(const placement_plan & plan, uint32_t n_layers, size_t kv_per_layer);
void         test_clear_kv_placement_plan();

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

}  // namespace ggml_sycl

// =============================================================================
// Blind preload threshold functions (for testing)
// =============================================================================
// Get the blind preload threshold (cached, reads env var once)
int ggml_sycl_get_blind_preload_threshold();

// Check if blind preload should be skipped for given expert count
// Returns true if should SKIP blind preload (expert count exceeds threshold)
bool ggml_sycl_should_skip_blind_preload(int64_t n_experts);
