#!/usr/bin/env python3
"""Source gate for owner-first staging and w295 transactional growth contracts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"
FATTN = (SYCL / "fattn.cpp").read_text()
RUNTIME = (SYCL / "ggml-sycl.cpp").read_text()
CACHE = (SYCL / "unified-cache.cpp").read_text()
COMMON = (SYCL / "common.hpp").read_text()
COMMON_IMPL = (SYCL / "common.cpp").read_text()


def region(source: str, start: str, end: str) -> str:
    begin = source.index(start)
    finish = source.index(end, begin)
    return source[begin:finish]


def owner_first(block: str, mutation: str) -> None:
    allocation = block.index("unified_allocate_owner(req)")
    owner = block.index("from_owned_alloc", allocation)
    resolved = block.index("resolve(", owner)
    validation = block.index("if (!resolved.ptr", resolved)
    publication = block.index(mutation, validation)
    assert allocation < owner < resolved < validation < publication
    assert "unified_alloc(" not in block
    assert "from_legacy_owned_alloc" not in block


# All four FATTN legacy-owner sites are closed, including both KV-zone users.
assert FATTN.count("unified_alloc(") == 0
assert FATTN.count("from_legacy_owned_alloc(") == 0
assert FATTN.count("unified_allocate_owner(") == 4
assert FATTN.count("prefer_vram_zone = ggml_sycl::vram_zone_id::KV") == 2

owner_first(
    region(FATTN, "bool ggml_sycl_fattn_xmx_update_packed_k_from_set_rows", "void ggml_sycl_fattn_xmx_unregister"),
    "packed.handle = std::move(handle)",
)
owner_first(
    region(FATTN, "static bool ggml_sycl_fattn_alloc_device_owner", "template <typename T>"),
    "owner = new ggml_sycl::mem_handle",
)
owner_first(
    region(FATTN, "bool ggml_sycl_fattn_xmx_materialize_packed_k", "static void ggml_sycl_fattn_xmx_v2_free"),
    "out->handle      = std::move(handle)",
)
owner_first(
    region(FATTN, "static bool ggml_sycl_fattn_xmx_v2_alloc_split_workspace_buffer", "static bool ggml_sycl_fattn_xmx_v2_ensure"),
    "*out = std::move(handle)",
)
print("PASS fattn-owner-first-source-gate")

# Failure atomicity: no existing output is cleared before allocation and
# resolution succeed, so allocation failure cannot publish a raw pointer or
# erase the previous owner.
sidecar_prefix = region(FATTN, "if (!reuse_alloc) {", "const auto resolved = handle.resolve(target_device);")
assert "packed.reset()" not in sidecar_prefix
materializer = region(FATTN, "bool ggml_sycl_fattn_xmx_materialize_packed_k", "// Publish every field used by retry reuse")
assert materializer.index("const auto resolved") < materializer.index("out->reset()")
print("PASS fattn-allocation-failure-leaves-output-untouched")

# Ten coherent runtime staging/workspace owner sites were migrated. The exact
# compatibility inventory prevents either a silent regression or an unreviewed
# widening of this bounded batch.
assert RUNTIME.count("unified_alloc(") == 56
assert RUNTIME.count("from_legacy_owned_alloc(") == 44
assert RUNTIME.count("unified_allocate_owner(") == 25
assert CACHE.count("unified_alloc(") == 28
assert CACHE.count("from_legacy_owned_alloc(") == 12
assert CACHE.count("unified_allocate_owner(") == 10
assert COMMON.count("unified_alloc(") == 4
assert COMMON.count("from_legacy_owned_alloc(") == 4
assert COMMON.count("unified_allocate_owner(") == 3
assert COMMON_IMPL.count("unified_alloc(") == 8
assert COMMON_IMPL.count("from_legacy_owned_alloc(") == 8
assert COMMON_IMPL.count("unified_allocate_owner(") == 4

runtime_regions = (
    ("ggml_backend_sycl_context::get_staging_buffer", "ggml_backend_sycl_context::free_staging_buffer"),
    ("ggml_backend_sycl_context::ensure_mmvq_host_staging", "ggml_backend_sycl_context::ensure_readback_staging"),
    ("ggml_backend_sycl_context::ensure_readback_staging", "ggml_backend_sycl_context::new_pool_for_device"),
    ("struct scoped_mmvq_scratch_handle", "constexpr bool quantize_enabled"),
    ("auto                               allocate_owned_scratch", "auto release_owned_scratch"),
    ("bool ensure_device(T *&", "static bool ggml_sycl_expert_entry_weight_ptr"),
    ("auto ensure_secondary_device_buffer", "// Ensure ALL slots"),
    ("static bool ggml_sycl_moe_down_sum_shadow_record", "static void ggml_sycl_moe_down_sum_shadow_compare"),
    ("// Allocate and validate a replacement before disturbing the published buffer.", "if (pipe.enabled)"),
    ("static void ggml_sycl_mmvq_soa_pre_allocate_buffers", "static void ggml_sycl_xmx_moe_pre_allocate_buffers"),
)
for start, end in runtime_regions:
    block = region(RUNTIME, start, end)
    assert "unified_allocate_owner(" in block, start
    assert "unified_alloc(" not in block, start
    assert "from_legacy_owned_alloc" not in block, start

# Representative replacement paths prove the old owner/raw view is not reset
# before the replacement has an accepted allocation and validated resolution.
for start, end, forbidden in (
    ("ggml_backend_sycl_context::get_staging_buffer", "ggml_sycl::allocation_result allocation", "free_staging_buffer()"),
    ("ggml_backend_sycl_context::ensure_mmvq_host_staging", "ggml_sycl::allocation_result allocation", "mmvq_host_staging_handle = {}"),
    ("ggml_backend_sycl_context::ensure_readback_staging", "ggml_sycl::allocation_result allocation", "readback_staging_handle = {}"),
    ("bool ensure_device(T *&", "ggml_sycl::allocation_result allocation", "reset_device(ptr, handle)"),
    ("auto ensure_secondary_device_buffer", "ggml_sycl::allocation_result allocation", "handle = {}"),
    ("// Allocate and validate a replacement before disturbing the published buffer.", "ggml_sycl::allocation_result allocation", "pipe.scratch_handle[b] = {}"),
):
    assert forbidden not in region(RUNTIME, start, end), start
print("PASS runtime-workspace-owner-first-source-gate")
print("PASS runtime-allocation-failure-leaves-output-untouched")

# The next coherent STAGING batch closes 17 legacy sites: 13 in the runtime,
# two in unified-cache, and one each in common.hpp/common.cpp. Exact legacy
# variable names are forbidden so migrated adapters cannot silently regress.
staging_runtime_regions = (
    ("struct ggml_sycl_scoped_staging_handle", "struct ggml_sycl_f16_attention_route"),
    ("struct scoped_staging_handle", "auto set_root_override"),
    ("struct ggml_sycl_pool_host", "void ggml_sycl::L2PrefetchManagerDeleter"),
    ("static bool convert_tensor_layout", "static bool ggml_sycl_select_mul_mat_layout"),
    ("static bool ggml_sycl_ensure_moe_ptr_table", "static void ggml_sycl_update_moe_hotset"),
    ("static const int32_t * ggml_sycl_get_moe_ids_device_ptr_exact", "static bool ggml_sycl_release_moe_tensor_layout"),
    ("static bool graph_preload_moe_experts", "static void graph_unpin_moe_experts"),
    ("static bool ensure_split_persistent_resources", "static sycl::queue *            g_split_merge_queue"),
    ("static bool split_secondary_gpu_ensure", "static const void * split_secondary_weight_load"),
    ("static ggml_sycl::mem_handle ggml_sycl_block_exec_alloc_host_stage_handle", "static bool ggml_sycl_block_exec_queue_matches_device"),
)
staging_runtime = "\n".join(region(RUNTIME, a, b) for a, b in staging_runtime_regions)
for start, end in staging_runtime_regions:
    assert "unified_allocate_owner(" in region(RUNTIME, start, end), start

for forbidden in (
    "alloc_handle xmx_staging_owner", "alloc_handle table_owner", "alloc_handle ids_pack_owner",
    "alloc_handle device_owner", "alloc_handle q8_owner", "alloc_handle f32_owner",
    "alloc_handle second_out_owner", "alloc_handle host_stage_owner",
    "ggml_sycl_take_owned_alloc_handle(alloc",
):
    assert forbidden not in staging_runtime, forbidden
for forbidden_call in (
    "unified_alloc(staging_req", "unified_alloc(req, &table_owner", "unified_alloc(req, &ids_pack_owner",
    "unified_alloc(req, &device_owner", "unified_alloc(req, &q8_owner", "unified_alloc(req, &f32_owner",
    "unified_alloc(req, &second_out_owner", "unified_alloc(req, &host_stage_owner",
):
    assert forbidden_call not in RUNTIME, forbidden_call

cache_reorder = region(CACHE, "bool unified_cache::reserve_reorder_temp", "bool unified_cache::reserve_persistent_scratch")
cache_fill = region(CACHE, "bool unified_cache_fill_with_host_copy", "bool unified_cache_copy_from_host_async")
for block in (cache_reorder, cache_fill):
    assert "unified_allocate_owner(" in block
    assert "unified_alloc(" not in block
    assert "from_legacy_owned_alloc" not in block

common_stage = region(COMMON, "void * ensure_buffer(size_t required_size", "ggml_sycl::mem_handle handle() const")
pp_stage = region(COMMON_IMPL, "void * ggml_sycl_pp_ensure_stage_buffer", "sycl::event ggml_sycl_pp_stage_transfer")
for block in (common_stage, pp_stage):
    assert "unified_allocate_owner(" in block
    assert "unified_alloc(" not in block
    assert "from_legacy_owned_alloc" not in block

# Representative failure seam: replacement owners are resolved and routing is
# validated before old staging metadata is published or cleared.
for block, mutation in (
    (region(RUNTIME, "static bool split_secondary_gpu_ensure", "static const void * split_secondary_weight_load"),
     "g_split_secondary_gpu.q8_handle = std::move(q8_replacement)"),
    (cache_reorder, "reorder_temp_owner_  = std::move(replacement)"),
    (common_stage, "backing_handle   = std::move(replacement)"),
    (pp_stage, "g_sycl_pp_config.stage_output_handle[stage] = std::move(replacement)"),
):
    allocation = block.index("unified_allocate_owner(")
    resolved = block.index("resolve(", allocation)
    validation = block.index("resolved.ptr", resolved)
    publication = block.index(mutation, validation)
    assert allocation < resolved < validation < publication
print("PASS staging-owner-first-source-gate-17")
print("PASS staging-replacement-failure-seam")

# w288: resize helpers qualify success against the requested geometry.  A
# surviving smaller pointer is never accepted after a failed growth attempt.
secondary = region(RUNTIME, "static bool split_secondary_gpu_ensure", "// Secondary GPU weight loading")
assert "q8_size >= q8_bytes" in secondary
assert "f32_size >= f32_bytes" in secondary
secondary_call = region(RUNTIME, "// Secondary GPU: H2D src1", "// CPU vec_dot:")
assert "!split_secondary_gpu_ensure(q8_bytes, src1_f32_bytes, second_out_bytes" in secondary_call
assert "s_second_out_dev_sz, s_second_out_dev_handle, stream_second" in secondary_call
persistent = region(RUNTIME, "static bool ensure_split_persistent_resources", "// OOQ merge queue")
assert "r.q8_staging_size >= need_q8" in persistent
assert "return false;" in persistent
assert "if (!ensure_split_persistent_resources(" in RUNTIME
print("PASS staging-resize-capacity-qualified-source-gate")

# Successful secondary replacements escrow q8/f32/output old owners as one
# transaction behind the exact queue terminal. The retirement ticket predates
# owner-vector growth and failed barrier/publication paths drain before unwind.
retirement = region(RUNTIME, "class ggml_sycl_old_owner_retirement", "// Construct this before direct_stage_expert")
# Field order (ticket before the owner vector) is the property; match on the
# declarations rather than on their column alignment, which clang-format moves
# whenever a neighbouring member name changes length.
assert retirement.index("publish_ticket_ =") < retirement.index("old_owners_;")
assert "old_owners_.reserve(owner_capacity)" in retirement
# The queue is held by pointer so the private host fixture can drive this
# transaction with no device; null means there is no submission to fence or
# drain, and every production caller passes a live queue.
assert "queue_->ext_oneapi_submit_barrier()" in retirement
assert "retain_handles_until_event_transactional(old_owners_, prior_queue_terminal, publish_ticket_)" in retirement
assert "ggml_sycl_drain_direct_stage_queue(*queue_)" in retirement
for owner in ("g_split_secondary_gpu.q8_handle", "g_split_secondary_gpu.f32_handle", "output_handle"):
    assert f"retirement.hold({owner})" in secondary
assert secondary.count("retirement.secure()") == 1
assert secondary.index("retirement.secure()") < secondary.index("g_split_secondary_gpu.q8_handle = std::move")
assert "secondary_queue.ext_oneapi_submit_barrier()" in persistent
print("PASS staging-in-flight-resize-owner-retention-source-gate")

# XMX staging and MoE pointer-table growth are failure atomic: no old state is
# cleared before a replacement resolves, and table payload vectors are built
# locally before the owner/size/validity tuple is published.
xmx_stage = region(RUNTIME, "const bool staging_too_small", "// Always use host staging")
assert "xmx_mxfp4_tiled_aos_staging_handle[device_id] = {}" not in xmx_stage
assert xmx_stage.index("staging_resolved && staging_resolved.on_device") < xmx_stage.index("std::move(staging_handle)")
moe_table = region(RUNTIME, "static bool ggml_sycl_ensure_moe_ptr_table", "static void ggml_sycl_update_moe_hotset")
assert "moe_expert_ptrs_handle[device]        = {};" not in moe_table
# The three publication sites share one constructor helper, so the
# same-allocation host-vector failure has a single seam. The "built off to the
# side, published only once complete" property moves with it: assert the call
# count here and the ordering inside the helper itself.
assert moe_table.count("ggml_sycl_build_moe_table_views(count,") == 3
table_views = region(RUNTIME, "static void ggml_sycl_build_moe_table_views", "#if defined(GGML_SYCL_PRIVATE_TESTING)")
assert "std::vector<ggml_sycl::mem_handle> new_handles(count)" in table_views
assert "std::vector<void *>                new_payload(count, nullptr)" in table_views
assert table_views.index("new_payload(count, nullptr)") < table_views.index("handles.swap(new_handles)")
assert table_views.index("handles.swap(new_handles)") < table_views.index("payload.swap(new_payload)")
assert moe_table.count("catch (const std::bad_alloc &)") >= 3
assert "ggml_sycl_checked_mul_size(count, sizeof(void *), &bytes)" in moe_table
assert "return true;" in moe_table and "return false;" in moe_table
assert "retirement.hold(extra->moe_expert_ptrs_handle[device])" in moe_table
print("PASS xmx-moe-replacement-failure-atomic-source-gate")

# w295 geometry is rejected before pointer arithmetic/allocation and a surviving
# undersized XMX pointer cannot pass the capacity gate.
xmx_convert = region(RUNTIME, "static bool convert_tensor_layout", "static bool ggml_sycl_select_mul_mat_layout")
assert "ggml_sycl_checked_mul_size(info.total_bytes" in xmx_convert
assert "xmx_mxfp4_tiled_aos_staging_size[device_id] < aos_expert_size" in xmx_convert
assert xmx_convert.index("xmx_mxfp4_tiled_aos_staging_size[device_id] < aos_expert_size") < xmx_convert.index("// Always use host staging")
assert "ggml_sycl_checked_round_up_size(static_cast<size_t>(K), MATRIX_ROW_PADDING" in RUNTIME
assert "ggml_sycl_checked_mul_size(q8_blocks, sizeof(block_q8_1), &q8_bytes)" in RUNTIME
assert "ggml_sycl_checked_mul_size(static_cast<size_t>(N_second), sizeof(float), &second_out_bytes)" in RUNTIME
assert "ggml_sycl_checked_mul_size(total_batches_size, sizeof(int32_t), &ids_bytes)" in RUNTIME
print("PASS overflow-safe-staging-geometry-source-gate")

# Graph-preload failure propagates into graph suppression, rather than logging
# and continuing through a stale graph path.
refresh = region(RUNTIME, "if (refresh_moe_after_pp)", "const int descriptor_moe_graph_candidates")
assert "if (!graph_preload_moe_experts(*sycl_ctx, cgraph))" in refresh
assert "sycl_ctx->moe_graphs_disabled = true" in refresh
assert "use_sycl_graph                = false" in refresh
assert "graph_unpin_moe_experts(sycl_ctx)" in refresh
print("PASS graph-preload-bool-propagation-source-gate")

# Metadata and its derived group registry are built locally and atomically
# swapped under both writer locks; bad_alloc preserves the old epoch.
metadata = region(RUNTIME, "// Build both views off to the side", "// Early multi-GPU setup")
assert "new_expert_meta" in metadata and "new_expert_groups" in metadata
assert "catch (const std::bad_alloc &)" in metadata
assert "std::scoped_lock lock(g_moe_expert_meta_mutex, g_expert_groups_mutex)" in metadata
assert metadata.index("catch (const std::bad_alloc &)") < metadata.index("g_moe_expert_meta.swap")
assert "g_expert_groups.swap(new_expert_groups)" in metadata
print("PASS moe-metadata-atomic-publication-source-gate")
