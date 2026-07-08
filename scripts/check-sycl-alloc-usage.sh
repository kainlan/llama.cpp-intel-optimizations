#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCAN_ROOT="${1:-ggml/src/ggml-sycl}"
if [[ "$SCAN_ROOT" = "$ROOT_DIR"* ]]; then
    SCAN_ROOT="${SCAN_ROOT#$ROOT_DIR/}"
fi

# This checker was written against ripgrep, but the generic CI runners do not
# ship it. When rg is unavailable, provide a GNU-grep shim covering the three
# invocation forms this script uses, so the SYCL alloc-policy check still runs
# everywhere. Requires GNU grep (\s and \b extensions, PCRE -P for the
# multiline patterns, -r, --include/--exclude-dir), which is present on the CI
# images and dev hosts. When a real rg binary exists it is used unchanged.
if ! command -v rg >/dev/null 2>&1; then
    rg() {
        # Form 1: quiet match on stdin -> rg -q PATTERN
        if [[ "${1:-}" == "-q" ]]; then
            grep -Eq -- "$2"
            return
        fi
        # Form 3: multiline scan -> rg -U "${RG_ARGS[@]}" PATTERN PATH
        local multiline=0
        if [[ "${1:-}" == "-U" ]]; then
            multiline=1
            shift
        fi
        # Form 2: recursive scan -> rg "${RG_ARGS[@]}" PATTERN PATH
        # RG_ARGS = -n --no-heading --with-filename --glob '*.ext'... --glob '!**/dir/**'
        local includes=() excludes=() positional=()
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --glob)
                    shift
                    local g="$1"
                    if [[ "$g" == '!'* ]]; then
                        g="${g#!}"; g="${g#\*\*/}"; g="${g%/\*\*}"
                        excludes+=("--exclude-dir=$g")
                    else
                        includes+=("--include=$g")
                    fi
                    ;;
                -n|--no-heading|--with-filename) ;;  # implied by grep -Hn
                *) positional+=("$1") ;;
            esac
            shift
        done
        if [[ "$multiline" == 1 ]]; then
            # -U/--multiline: treat each file as one NUL record (-z) so patterns
            # with embedded \n match across lines; -P (PCRE) honors \n in classes.
            # -l lists matching files -> non-empty output signals a match, which
            # is all the -U call sites test.
            grep -rPlz "${includes[@]}" "${excludes[@]}" -e "${positional[0]}" "${positional[1]}"
        else
            # grep -Hn output "path:line:text" matches rg --with-filename --no-heading -n.
            # -e keeps the (long, regex-heavy) pattern from being read as a path.
            # grep exits 1 on no matches; callers wrap with `|| true`.
            grep -rHnE "${includes[@]}" "${excludes[@]}" -e "${positional[0]}" "${positional[1]}"
        fi
    }
fi

PATTERN='sycl::malloc_(device|host|shared)\s*\(|sycl_aligned_malloc_device\s*\(|sycl::free\s*\(|zeMemAlloc[^[:space:]]*\s*\(|zeMemFree\s*\(|ggml_sycl_(malloc|free)_device_raw\s*\(|unified_cache_raw_(malloc_(device|host)|free_device)\s*\('
ALLOW_RE='^ggml/src/ggml-sycl/unified-cache\.cpp$'
HOST_ALLOC_PATTERN='std::(malloc|calloc|realloc|aligned_alloc|free)\s*\(|(^|[^[:alnum:]_:])(malloc|calloc|aligned_alloc)\s*\('
VM_ALLOC_PATTERN='(^|[^[:alnum:]_:])(posix_memalign|mmap|munmap)\s*\('
RAW_CACHE_API_PATTERN='unified_cache_(allocate|deallocate)\s*\('
RAW_CACHE_API_ALLOW_RE='^ggml/src/ggml-sycl/unified-cache\.(cpp|hpp)$'
RAW_ZONE_API_PATTERN='unified_cache_zone_(alloc|free)\s*\('
RAW_ZONE_API_ALLOW_RE='^ggml/src/ggml-sycl/unified-cache\.(cpp|hpp)$'
SYNTHETIC_ZONE_HANDLE_PATTERN='\.(zone_managed|vram_zone|host_zone)[[:space:]]*=[[:space:]]*[^=]'
SYNTHETIC_ZONE_HANDLE_ALLOW_RE='^ggml/src/ggml-sycl/unified-cache\.(cpp|hpp)$'
XMX_LEGACY_OWNERSHIP_PATTERN='xmx_mxfp4_tiled(_aos_staging)?(_alloc|_owned|[[:space:]]*\[)'
XMX_DEVICE_TMP_LEGACY_PATTERN='ggml_sycl::alloc_handle[[:space:]]+tmp_alloc'
SYCL_EXT_DEVICE_TEMP_LEGACY_PATTERN='sycl_ext_(device_temp|malloc_device|free)'
XMX_TILED_OWNER_LEGACY_PATTERN='ggml_sycl::alloc_handle[[:space:]]+tiled_alloc'
SPEC_VERIFY_LOGITS_LEGACY_PATTERN='\blogits_alloc\b'
PP_STAGE_LEGACY_OWNERSHIP_PATTERN='stage_output_(buf|alloc)'
PP_PIPELINE_LEGACY_OWNERSHIP_PATTERN='(scratch_alloc\[[^]]+\]|ggml_sycl::alloc_handle[[:space:]]+scratch_alloc)'
FATTN_SPLIT_LEGACY_OWNERSHIP_PATTERN='split_partial_(max|sum|out)_alloc'
FATTN_PACKED_K_LEGACY_OWNERSHIP_PATTERN='(packed\.alloc|out->alloc)'
FATTN_ONEDNN_MATERIALIZED_RAW_RELEASE_PATTERN='(ggml_sycl_release_materialized_after_event|unified_free\([[:space:]]*[qkv][[:space:]]*\))'
FATTN_ONEDNN_MATERIALIZED_SCOPED_ALLOC_PATTERN='(scoped_unified_alloc[[:space:]]+[QKV]_alloc|[QKV]_alloc\.as_mem_handle\(\)|materialized\.[QKV]_alloc|from_owned_alloc\(materialized\.[QKV]_alloc\.release\(\))'
FATTN_BUFFERS_KV_LEGACY_OWNER_PATTERN='(alloc_handle[[:space:]]+kv_buffer_owner|from_owned_alloc\(std::move\(kv_buffer_owner\))'
ONEDNN_XMX_FILL_SCOPED_STAGE_PATTERN='(scoped_unified_alloc[[:space:]]+(tmp_alloc|dev_alloc)|scoped_unified_alloc[[:space:]]+host_alloc\(host_req\)|tmp_alloc\.(allocate|get)\(|dev_alloc\.as_mem_handle\(\))'
GGML_SYCL_CPP_SCOPED_UNIFIED_ALLOC_PATTERN='scoped_unified_alloc'
LAYER_STREAM_LEGACY_OWNERSHIP_PATTERN='buffer_allocs_\[[^]]+\]'
LAYER_STREAM_LEGACY_OWNER_PATTERN='(alloc_handle[[:space:]]+buffer_alloc|unified_free\(buffer_alloc\)|from_owned_alloc\([^\n]*buffer_alloc)'
COMPUTE_BUFFER_LEGACY_OWNERSHIP_PATTERN='alloc_handle[[:space:]]+scratch_handle_'
COMPUTE_BUFFER_MANAGER_LEGACY_OWNER_PATTERN='(alloc_handle[[:space:]]+(buffer_owner|scratch_owner)|from_owned_alloc\([^\n]*(buffer_owner|scratch_owner))'
GPU_SAMPLER_LEGACY_OWNER_PATTERN='(alloc_handle[[:space:]]+sampler_owner|from_owned_alloc\([^\n]*sampler_owner)'
BACKEND_CONTEXT_LEGACY_STAGING_PATTERN='(readback_staging_alloc|mmvq_host_staging_alloc|staging_buffer_alloc_|ggml_sycl::alloc_handle[[:space:]]+(device_staging_alloc|host_staging_alloc|readback_alloc))'
CONTROL_HOST_LEGACY_PATTERN='ggml_sycl::alloc_handle[[:space:]]+control_alloc'
FFN_NORM_LEGACY_OWNERSHIP_PATTERN='data_dev1_alloc'
TP_INPUT_LEGACY_OWNERSHIP_PATTERN='data_ptr\(\) const \{ return alloc\.ptr \? alloc\.ptr : data; \}'
TP_INPUT_CAPTURE_LEGACY_PATTERN='input_dev1_alloc'
CPU_DISPATCH_HOST_COPY_LEGACY_PATTERN='host_ptr_owned_alloc'
CPU_DISPATCH_HOST_COPY_LEGACY_OWNER_PATTERN='(alloc_handle[[:space:]]+copy_alloc|from_owned_alloc\([^\n]*copy_alloc)'
USM_POINTER_TYPE_CACHE_LEGACY_PATTERN='(tl_ptr_type_cache|static[[:space:]]+std::unordered_map<void[[:space:]]*\*,[[:space:]]*bool>[[:space:]]+cache)'
WEIGHT_IDENTITY_POINTER_CACHE_LEGACY_PATTERN='(std::unordered_map<const[[:space:]]+ggml_tensor[[:space:]]*\*,[[:space:]]*ggml_sycl_weight_identity>[[:space:]]+g_sycl_weight_identities\b|g_sycl_weight_identities\[[^]]+\]|g_sycl_weight_identities\.find\()'
HOST_WEIGHT_EXTRAS_POINTER_REGISTRY_LEGACY_PATTERN='(std::unordered_map<ggml_tensor[[:space:]]*\*,[[:space:]]*ggml_tensor_extra_gpu[[:space:]]*\*>[[:space:]]+g_sycl_host_weight_extras\b|g_sycl_host_weight_extras\.emplace\([[:space:]]*tensor[[:space:]]*,)'
CPU_DISPATCH_HOST_PTR_SPLIT_REGISTRY_LEGACY_PATTERN='(std::unordered_map<std::string,[[:space:]]*const[[:space:]]+void[[:space:]]*\*>[[:space:]]+g_host_ptr_map\b|g_host_ptr_owned_handles\b|g_host_ptr_map\[[^]]+\][[:space:]]*=[[:space:]]*host_ptr|g_host_ptr_map\[[^]]+\][[:space:]]*=[[:space:]]*copy)'
CPU_DISPATCH_EXPERT_GROUP_POINTER_PATTERN='(std::unordered_map<const[[:space:]]+void[[:space:]]*\*,[[:space:]]*std::vector<int>>[[:space:]]+expert_groups\b|expert_groups\[[^]]*weight_host[^]]*\]\.push_back)'
STAGING_CACHE_LEGACY_OWNERSHIP_PATTERN='(alloc_handle[[:space:]]+handles\[GGML_SYCL_MAX_DEVICES\]|it->second\.alloc|entry->handles\[|it->second\.handles\[|entry\.second\.handles\[)'
MEM_OPS_STAGE_LEGACY_OWNER_PATTERN='(alloc_handle[[:space:]]+stage_alloc|from_owned_alloc\([^\n]*stage_alloc)'
CPY_HOST_STAGE_LEGACY_OWNER_PATTERN='(alloc_handle[[:space:]]+host_stage_owner|from_owned_alloc\(std::move\(host_stage_owner\))'
SET_ROWS_HOST_STAGE_LEGACY_OWNER_PATTERN='(alloc_handle[[:space:]]+host_stage_owner|from_owned_alloc\(std::move\(host_stage_owner\))'
DMMV_STAGE_SCRATCH_LEGACY_OWNER_PATTERN='(alloc_handle[[:space:]]+(host_stage_owner|device_scratch_owner)|from_owned_alloc\(std::move\((host_stage_owner|device_scratch_owner)\))'
CONVERT_DEVICE_SCRATCH_LEGACY_OWNER_PATTERN='(alloc_handle[[:space:]]+device_scratch_owner|from_owned_alloc\(std::move\(device_scratch_owner\))'
MMQ_XMX_DEVICE_SCRATCH_LEGACY_OWNER_PATTERN='(alloc_handle[[:space:]]+device_scratch_owner|from_owned_alloc\(std::move\(device_scratch_owner\))'
DENSE_SCHEDULER_SLOT_LEGACY_OWNER_PATTERN='(alloc_handle[[:space:]]+slot_owner|from_owned_alloc\(std::move\(slot_owner\))'
MMQ_STAGE_COUNTER_LEGACY_OWNER_PATTERN='(alloc_handle[[:space:]]+(host_stage_owner|work_counter_owner)|from_owned_alloc\(std::move\((host_stage_owner|work_counter_owner)\))'
MMVQ_STAGE_REUSE_LEGACY_OWNER_PATTERN='(alloc_handle[[:space:]]+(device_scratch_owner|host_stage_owner|q8_owner|scratch_owner)|from_owned_alloc\(std::move\((device_scratch_owner|host_stage_owner|q8_owner|scratch_owner)\))'
MMVQ_GRAPH_COMPACT_LEGACY_OWNER_PATTERN='(alloc_handle[[:space:]]+(compact_alloc|missing_alloc|graph_owner)|from_owned_alloc\(std::move\((compact_alloc|missing_alloc|graph_owner)\))'
MMVQ_RMSNORM_SCALES_LEGACY_OWNER_PATTERN='(alloc_handle[[:space:]]+scales_alloc|from_owned_alloc\([^\n]*scales_alloc)'
GET_ROWS_STAGE_LEGACY_OWNER_PATTERN='(alloc_handle[[:space:]]+(host_stage_owner|device_temp_owner|indices_owner)|from_owned_alloc\(std::move\((host_stage_owner|device_temp_owner|indices_owner)\))'
SET_ROWS_STAGING_LEGACY_PATTERN='(std::vector<ggml_sycl::alloc_handle>[[:space:]]*&[[:space:]]*staged_allocs|ggml_sycl_set_rows_free_staged|ggml_sycl_set_rows_staged_alloc_guard|staged_allocs\.push_back\(alloc\)|std::vector<ggml_sycl::alloc_handle>[[:space:]]+handles)'
SET_ROWS_STAGING_SCOPED_PATTERN='(scoped_unified_alloc[[:space:]]+scoped_alloc|scoped_alloc\.(get|release)\(\))'
BINBCAST_RAW_HOST_STAGE_SCOPED_PATTERN='(scoped_unified_alloc[[:space:]]+src[01]_stage|scoped_unified_alloc[[:space:]]*&[[:space:]]*stage|stage\.(allocate|get)\()'
COMMON_HOST_STAGING_SCOPED_PATTERN='(scoped_unified_alloc[[:space:]]+host_alloc|host_alloc\.(get|as_mem_handle)\(\))'
READBACK_FALLBACK_SCOPED_PATTERN='(scoped_unified_alloc[[:space:]]+(fallback_alloc|host_alloc)|(^|[^[:alnum:]_])(fallback_alloc|host_alloc)\.(allocate|get|as_mem_handle)\()'
MOE_READBACK_STAGE_SCOPED_PATTERN='(scoped_unified_alloc[[:space:]]+(gate_stage|early_stage|weights_stage)|(^|[^[:alnum:]_])(gate_stage|early_stage|weights_stage)\.as_mem_handle\(\))'
MOE_PHASE2_D2H_SCOPED_PATTERN='(scoped_unified_alloc[[:space:]]+d2h_staging|d2h_staging\.(allocate|get|as_mem_handle)\()'
SPLIT_WEIGHT_STAGE_SCOPED_PATTERN='(scoped_unified_alloc[[:space:]]+staging_alloc|staging_alloc\.(allocate|get|as_mem_handle)\()'
GRAPH_Q8_LEGACY_OWNERSHIP_PATTERN='(alloc_handle[[:space:]]+q8_1_owner|q8_1_owner\.ptr|backing_alloc)'
GET_ROWS_HOST_STAGE_SCOPED_PATTERN='(scoped_unified_alloc[[:space:]]+host_stage|host_stage\.as_mem_handle\(\))'
TP_REDUCE_LEGACY_OWNERSHIP_PATTERN='g_tp_(shared_reduce|host_buf[01])_alloc'
TP_HOST_STAGING_LEGACY_OWNERSHIP_PATTERN='host_staging\.alloc'
TP_QUANT_COMM_LEGACY_OWNERSHIP_PATTERN='(dev_q_alloc|dev_minmax_alloc|host_q[01]_alloc|host_result_alloc)'
TP_ASYNC_RESULT_LEGACY_OWNERSHIP_PATTERN='result_alloc'
TP_ALLOC_TMP_LEGACY_OUT_PATTERN='ggml_sycl_tp_alloc_tmp\([^)]*alloc_handle[[:space:]]*\*|ggml_sycl_tp_alloc_tmp\(.*&[[:space:]]*[A-Za-z0-9_]*(owner|alloc)'
TP_COMPUTE_BUFFER_LEGACY_OWNERSHIP_PATTERN='(input_q8_alloc|gate_out_alloc|up_out_alloc|hidden_out_alloc|hidden_q8_alloc|partial_out_alloc|q_out_alloc|k_out_alloc|v_out_alloc|attn_out_alloc|attn_q8_alloc|attn_scores_alloc)'
DEV1_KV_CACHE_LEGACY_OWNERSHIP_PATTERN='\b(k_alloc|v_alloc)\b'
TP_COLUMN_OUTPUT_LEGACY_PATTERN='(ggml_sycl_tp_column_parallel_output|g_tp_column_parallel_outputs)'
ROUTING_INDICES_LEGACY_OWNERSHIP_PATTERN='(g_routing_indices_cache\.host_alloc|host_alloc;[[:space:]]*// Ownership for host_indices|ggml_sycl::alloc_handle[[:space:]]+new_alloc)'
MOE_IDS_LEGACY_OWNERSHIP_PATTERN='((entry|ids_entry)\.(device_alloc|staging_alloc)|ggml_sycl::alloc_handle[[:space:]]+(device_alloc|staging_alloc)([;{]|$)|device_ids_ptr\(\) const \{ return device_alloc\.ptr|staging_ids_ptr\(\) const \{ return staging_alloc\.ptr)'
MOE_IDS_PACK_LEGACY_PATTERN='ids_pack_alloc'
MOE_FUSION_LEGACY_PATTERN='fused_alloc'
MOE_PTR_TABLE_LEGACY_PATTERN='table_alloc'
MOE_GPU_PROBE_LEGACY_PATTERN='\bprobe_alloc\b'
GPU_REORDER_TEMP_LEGACY_PATTERN='(std::vector<ggml_sycl::alloc_handle>[[:space:]]+temp_vram_bufs|std::unordered_map<int, std::vector<ggml_sycl::alloc_handle>>[[:space:]]+sec_temp_bufs|alloc_handle[[:space:]]+temp_vram_h|arena_device_alloc\(.*alloc_handle)'
REORDER_D2H_STAGE_LEGACY_PATTERN='\bsrc_stage_alloc\b'
PAYLOAD_STAGE_LEGACY_PATTERN='from_owned_alloc\(std::move\(host_alloc\)'
GENERIC_OWNED_ALLOC_HANDOFF_PATTERN='from_owned_alloc\(std::move\(alloc\)'
GENERIC_OWNED_ALLOC_HANDOFF_ENFORCED_RE='^ggml/src/ggml-sycl/.*'
RUNTIME_LOOKUP_OWNER_ALLOC_PATTERN='\bowner_alloc\b'
BACKEND_BUFFER_CONTEXT_LEGACY_OWNER_PATTERN='ggml_sycl::alloc_handle[[:space:]]+(managed_alloc|tp_allocs\[[^]]+\])'
BACKEND_BUFFER_CONTEXT_DIRECT_ASSIGN_PATTERN='ctx->(managed_alloc[[:space:]]*=|tp_allocs\[[^]]+\][[:space:]]*=)'
BACKEND_BUFFER_CONTEXT_AS_MEM_HANDLE_PATTERN='(managed_alloc|tp_allocs\[[^]]+\])\.as_mem_handle\('
BACKEND_BUFFER_CONTEXT_LOOKUP_FREE_PATTERN='(unified_lookup\(dev_ptr|unified_free\(looked_up\))'
SYCL_POOL_LEGACY_OWNERSHIP_PATTERN='(std::unordered_map<void \*, ggml_sycl::alloc_handle>[[:space:]]+active_handles|ggml_sycl::alloc_handle[[:space:]]+handle;|unified_free\(it->second\)|unified_free\(b\.handle\))'
TENSOR_EXTRA_DATA_ALLOC_LEGACY_PATTERN='(data_alloc\[[^]]+\]|ggml_sycl::alloc_handle[[:space:]]+data_alloc\b)'
HOST_BUFFER_CONTEXT_LEGACY_PATTERN='(ggml_sycl::alloc_handle[[:space:]]+alloc;|ctx->alloc(\.ptr|\b)|unified_free\(ctx->alloc\)|new sycl_host_buf_ctx\{[^}]*alloc)'
VRAM_POOL_LEGACY_OWNERSHIP_PATTERN='(alloc_handle[[:space:]]+handle;|unified_free\((alloc|it->second)\.handle\)|allocations_\[tensor_id\][[:space:]]*=[[:space:]]*\{[[:space:]]*handle)'
UNIFIED_ALLOCATE_DIRECT_HANDLE_PATTERN='return[[:space:]]+mem_handle::from_direct\(handle\.ptr'
MANAGED_HOST_PINNED_LEGACY_PATTERN='(auto[[:space:]]+mh[[:space:]]*=[[:space:]]*handle\.as_mem_handle\(\)|\(void\)[[:space:]]+ggml_sycl::unified_free\(handle\)|return[[:space:]]+handle\.as_mem_handle\(\))'
ALLOCATE_MANAGED_HOST_PINNED_LEGACY_PATTERN='(out->as_mem_handle\(\)|unified_free\(\*out\)|ggml_sycl::alloc_handle[[:space:]]+(act_owner|out_owner)\{\}|unified_free\((act_owner|out_owner|weight_owner)\))'
STAGING_BUFFER_POOL_LEGACY_PATTERN='(has_unified_handle|unified_free\(s\.unified_handle\)|new_slot\.unified_handle[[:space:]]*=[[:space:]]*std::move\(unified_handle\))'
CPU_FALLBACK_HOST_COPY_LEGACY_PATTERN='(entry\.alloc|unified_free\(entry\.alloc\)|fallback_host_copy.*alloc_handle)'
TIERED_KV_ZONE_H_LEGACY_PATTERN='(alloc_handle[[:space:]]+zone_h|\.zone_h(\.|[[:space:]]*=)|zone_h[[:space:]]*=|unified_free\([^)]*zone_h\)|zone_h\.as_mem_handle\()'
ONEDNN_SCRATCH_LEGACY_OWNER_PATTERN='(std::shared_ptr<alloc_handle>[[:space:]]+onednn_(weights|activations)_scratch_owner_|onednn_(weights|activations)_scratch_owner_->|unified_free\(\*onednn_(weights|activations)_scratch_owner_|release_direct_scratch[^\n]*std::shared_ptr<alloc_handle>|allocate_direct_scratch[^\n]*std::shared_ptr<alloc_handle>)'
PERSISTENT_SCRATCH_LEGACY_OWNER_PATTERN='(std::shared_ptr<alloc_handle>[[:space:]]+owner[[:space:]]*=|persistent_scratch_entry[^;{]*std::shared_ptr<alloc_handle>|entry\.owner[[:space:]]*&&[[:space:]]*entry\.owner->ptr|unified_free\(\*entry\.owner\)|entry\.owner\.reset\(\))'
COMPUTE_ARENA_LEGACY_OWNER_PATTERN='(std::shared_ptr<alloc_handle>[[:space:]]+compute_arena_owner_|compute_arena_owner_->|unified_free\(\*compute_arena_owner_|compute_arena_owner_\.reset\(\))'
SCRATCH_POOL_LEGACY_OWNER_PATTERN='(std::shared_ptr<alloc_handle>[[:space:]]+scratch_pool_owner_|scratch_pool_owner_->|unified_free\(\*scratch_pool_owner_|scratch_pool_owner_\.reset\(\))'
STAGING_OWNER_LEGACY_PATTERN='(std::shared_ptr<alloc_handle>[[:space:]]+staging_owner_|staging_owner_->|unified_free\(\*staging_owner_|staging_owner_\.reset\(\))'
PARTIAL_ROW_CACHE_LEGACY_OWNER_PATTERN='(partial_entry[^;{]*std::shared_ptr<alloc_handle>|partial_owner\.as_mem_handle\(\)|std::make_shared<alloc_handle>\(partial_owner\)|unified_free\(partial_owner\)|pair\.second\.owner[[:space:]]*&&[[:space:]]*pair\.second\.owner->ptr|unified_free\(\*pair\.second\.owner\))'
DIRECT_ALLOC_OWNER_LEGACY_PATTERN='(std::shared_ptr<alloc_handle>[[:space:]]+(new_)?direct_alloc_owner|direct_alloc_owner->|direct_alloc_owner[[:space:]]*&&|(new_)?direct_alloc_owner[[:space:]]*=[[:space:]]*std::make_shared<alloc_handle>|std::make_shared<alloc_handle>\(handle\)|unified_free\(\*direct_alloc_owner\)|(dma_staging_allocs_\[[^]]+\]|entry\.handle|it->handle)\.owner[[:space:]]*&&[[:space:]]*(dma_staging_allocs_\[[^]]+\]|entry\.handle|it->handle)\.owner->ptr|unified_free\(\*(dma_staging_allocs_\[[^]]+\]|entry\.handle|it->handle)\.owner\))'
MOE_PRESTAGE_CPU_REORDER_LEGACY_PATTERN='reorder_alloc'
SET_TENSOR_REORDER_FALLBACK_LEGACY_PATTERN='reorder_fallback_alloc'
EXPERT_PREFETCH_LEGACY_PATTERN='scores_alloc_'
FP16_CACHE_LEGACY_PATTERN='slab_alloc'
PINNED_BUFFER_POOL_LEGACY_PATTERN='(act_alloc_|out_alloc_)'
PINNED_BUFFER_POOL_LEGACY_OWNER_PATTERN='(alloc_handle[[:space:]]+(act_owner|out_owner)|from_owned_alloc\([^\n]*(act_owner|out_owner))'
PINNED_POOL_CHUNK_LEGACY_OWNER_PATTERN='(std::shared_ptr<alloc_handle>[[:space:]]+allocate_pinned_chunk_owner|alloc_handle[[:space:]]+owner|std::make_shared<alloc_handle>\(std::move\(owner\)\)|unified_free\(\*c\.owner\)|c\.owner->ptr)'
PENDING_CPU_LEGACY_OWNERSHIP_PATTERN='(ggml_sycl::alloc_handle[[:space:]]+(out_alloc|act_alloc|weight_alloc)([;{]|$)|g_pending_(scatter|cpu_pipeline)\.(out_alloc|act_alloc|weight_alloc)|pb\.(out_alloc|act_alloc|weight_alloc)|result\.(out_alloc|act_alloc))'
CPU_EXPERT_POOL_LEGACY_PATTERN='ring_alloc_'
CPU_EXPERT_POOL_LEGACY_OWNER_PATTERN='(alloc_handle[[:space:]]+ring_owner|from_owned_alloc\([^\n]*ring_owner)'
CONT_BATCH_ALLOC_LEGACY_OWNER_PATTERN='(alloc_handle[[:space:]]+batch_owner|from_owned_alloc\(std::move\(batch_owner\))'
UNIFIED_KERNEL_MICROGRAPH_LEGACY_PATTERN='(micro_tile_counters_alloc_|micro_gen_alloc_|mmvq_q8_buf_allocs_|mmvq_gate_scratch_alloc_|mmvq_up_scratch_alloc_)'
UNIFIED_KERNEL_LIGHT_FLAGS_LEGACY_PATTERN='light_flags_alloc_'
UNIFIED_KERNEL_ROLE_SCHEDULE_LEGACY_PATTERN='(role_sync_alloc_|role_elem_alloc_|role_matmul_alloc_)'
UNIFIED_KERNEL_OPS_POOL_LEGACY_PATTERN='ops_pool_alloc_'
UNIFIED_KERNEL_PHASE_SCHEDULE_LEGACY_PATTERN='(phase_entries_alloc_|phase_offset_alloc_|phase_tiles_alloc_|phase_type_alloc_)'
UNIFIED_KERNEL_GET_ROWS_LEGACY_PATTERN='\bget_rows_alloc_\b'
UNIFIED_KERNEL_GET_ROWS_SHARED_HANDLE_PATTERN='\bget_rows_handle_\b'
UNIFIED_KERNEL_SCRATCH_POOL_LEGACY_PATTERN='scratch_pool_alloc_'
UNIFIED_KERNEL_DEFERRED_COPY_RAW_SCRATCH_PATTERN='dc\.source_op_idx[[:space:]]*>=.*copy_handle_for_raw_ptr\(src,[[:space:]]*device_id_'
UNIFIED_KERNEL_DEFERRED_COPY_EXEC_RAW_FALLBACK_PATTERN='copy_handle_for_raw_ptr\((src|dc\.dst),[[:space:]]*device_id_'
UNIFIED_KERNEL_DEFERRED_COPY_RAW_API_BRIDGE_PATTERN='copy_handle_for_raw_ptr\((src_ptr|dst),[[:space:]]*device_id_?'
UNIFIED_KERNEL_FINAL_COPY_RAW_DST_PATTERN='copy_handle_for_raw_ptr\(copy_back_dst,[[:space:]]*device_id_'
UNIFIED_KERNEL_RAW_HANDLE_BRIDGE_DEF_PATTERN='copy_handle_for_raw_ptr\(void[[:space:]]*\*[[:space:]]*ptr,[[:space:]]*int[[:space:]]+device_id\)'
UNIFIED_KERNEL_FINAL_COPY_OP_INDEX_PATTERN='add_deferred_copy\(final_op_idx,[[:space:]]*nullptr,[[:space:]]*copy_back_dst'
PERSISTENT_TG_DEFERRED_COPY_RAW_PATTERN='kernel\.add_deferred_copy\(source_op_idx,[[:space:]]*src_ptr,[[:space:]]*dst_ptr,[[:space:]]*nbytes\)'
PERSISTENT_SPLIT_INPUT_RAW_REWRAP_PATTERN='from_chunk_ptr\([[:space:]\n]*const_cast<void[[:space:]]*\*>\(info\.input_device_ptr\)'
PENDING_DEVICE_TOKEN_RAW_REWRAP_PATTERN='from_chunk_ptr\([[:space:]\n]*src_device,[[:space:]]*0,[[:space:]]*GGML_LAYOUT_AOS,[[:space:]]*true\)'
PENDING_DEVICE_TOKEN_RAW_STATE_PATTERN='(void[[:space:]]*\*[[:space:]]*\btoken_ptr[[:space:]]*=|g_sycl_device_token_cache\.token_ptr)'
SEQ_IDS_RAW_STATE_PATTERN='(const[[:space:]]+int32_t[[:space:]]*\*[[:space:]]*(q_seq_ids|kv_seq_ids)[[:space:]]*=[[:space:]]*nullptr|g_sycl_seq_ids_cache\.(q_seq_ids|kv_seq_ids))'
MOE_WEIGHTS_BY_LAYER_POINTER_PATTERN='(std::unordered_map<int,[[:space:]]*const[[:space:]]+ggml_tensor[[:space:]]*\*>[[:space:]]+g_moe_weights_by_layer\b|g_moe_weights_by_layer\[[^]]+\][[:space:]]*=[[:space:]]*node|const[[:space:]]+ggml_tensor[[:space:]]*\*[[:space:]]+wt[[:space:]]*=[[:space:]]*wit->second)'
MOE_PRECOMPUTED_SKIP_POINTER_SET_PATTERN='(std::unordered_set<const[[:space:]]+ggml_tensor[[:space:]]*\*>[[:space:]]+g_moe_(precomputed_(mmid|node)_skip|down_sum_fusion_disabled)\b|g_moe_precomputed_(mmid|node)_skip\.(count|insert|erase)\(|g_moe_down_sum_fusion_disabled\.(find|insert|erase|count)\()'
GRAPH_PRESTAGE_POINTER_DEDUPE_PATTERN='(std::unordered_set<void[[:space:]]*\*>[[:space:]]+staged_pointers\b|staged_pointers\.(insert|count)\([[:space:]]*tensor->data[[:space:]]*\))'
GRAPH_INPUT_DISCOVERY_POINTER_DEDUPE_PATTERN='(std::unordered_set<const[[:space:]]+void[[:space:]]*\*>[[:space:]]+seen_ptrs\b|seen_ptrs\.insert\([[:space:]]*tensor->data[[:space:]]*\))'
PREFETCH_SCHEDULER_POINTER_ACTIVE_SET_PATTERN='(std::unordered_set<const[[:space:]]+void[[:space:]]*\*>[[:space:]]+active_prefetches_\b|active_prefetches_\.(insert|erase|count)\([[:space:]]*(req\.tensor_data|tensor_data)[[:space:]]*\))'
MOE_Q8_POINTER_CACHE_PATTERN='(const[[:space:]]+void[[:space:]]*\*[[:space:]]+cached_src\b|cached_src[[:space:]]*==[[:space:]]*src|cached_src[[:space:]]*=[[:space:]]*src1?_d|ctx\.moe_q8_cache\.cached_src[[:space:]]*=)'
BACKEND_BUFFER_RUNTIME_ZONE_RESET_PATTERN='unified_cache_zone_reset\([^)]*vram_zone_id::RUNTIME'
UNIFIED_KERNEL_SYNC_BLOCK_LEGACY_PATTERN='sync_block_alloc_'
UNIFIED_KERNEL_PERSISTENT_BUFFER_LEGACY_PATTERN='persistent_buf_allocs_'
UNIFIED_KERNEL_DEVICE_PERSISTENT_LEGACY_OWNER_PATTERN='(alloc_handle[[:space:]]+(persistent_owner|sync_block_owner|ready_owner|claimed_owner|done_owner|completed_owner|role_sync_owner|flags_owner)|from_owned_alloc\(std::move\((persistent_owner|sync_block_owner|ready_owner|claimed_owner|done_owner|completed_owner|role_sync_owner|flags_owner)\))'
UNIFIED_KERNEL_PINNED_PERSISTENT_LEGACY_OWNER_PATTERN='(alloc_handle[[:space:]]+(successor_off_owner|successor_list_owner|n_tiles_owner|initial_ready_owner|entries_owner|offset_owner|tiles_owner|type_owner|role_elem_owner|role_matmul_owner|generation_owner|ops_pool_owner|profile_alloc)|from_owned_alloc\(std::move\((successor_off_owner|successor_list_owner|n_tiles_owner|initial_ready_owner|entries_owner|offset_owner|tiles_owner|type_owner|role_elem_owner|role_matmul_owner|generation_owner|ops_pool_owner|profile_alloc)\)|profile_alloc\.ptr|unified_free\(profile_alloc\))'
UNIFIED_KERNEL_DEVICE_SCRATCH_LEGACY_OWNER_PATTERN='(alloc_handle[[:space:]]+(get_rows_owner|scratch_owner|counters_owner|q8_owner|gate_owner|up_owner)|from_owned_alloc\(std::move\((get_rows_owner|scratch_owner|counters_owner|q8_owner|gate_owner|up_owner)\))'
UNIFIED_KERNEL_DAG_LEGACY_PATTERN='(dag_ready_counter_alloc_|dag_tile_claimed_alloc_|dag_tiles_done_alloc_|dag_completed_alloc_|dag_successor_off_alloc_|dag_successor_list_alloc_|dag_n_tiles_alloc_|dag_initial_ready_counter_alloc_)'
UNIFIED_KERNEL_TEMP_DEVICE_LEGACY_PATTERN='(temp_device_allocs|cached_temp_device_allocs_|add_temp_device_alloc_handle)'
UNIFIED_KERNEL_GRAPH_OVERHEAD_LEGACY_PATTERN='(alloc_handle[[:space:]]+(dummy_alloc|dummy_owner)|from_owned_alloc\(std::move\(dummy_owner\))'
SPLIT_PERSISTENT_Q8_LEGACY_PATTERN='q8_staging_alloc'
SPLIT_PERSISTENT_SYNC_LEGACY_PATTERN='(progress_counter_alloc|merge_complete_alloc)'
MMVQ_SOA_BULK_LEGACY_PATTERN='ggml_sycl::alloc_handle[[:space:]]+bulk_alloc'
SPLIT_SECONDARY_GPU_LEGACY_PATTERN='(g_split_secondary_gpu\.(q8_alloc|f32_alloc)|ggml_sycl::alloc_handle[[:space:]]+(q8_alloc|f32_alloc);)'
SPLIT_SECONDARY_OUTPUT_LEGACY_PATTERN='s_second_out_dev_alloc'
SPLIT_WEIGHT_CACHE_LEGACY_PATTERN='(split_weight_cache_entry.*alloc_handle|unified_free\(entry\.handle\))'
MXFP4_TG_REUSE_LEGACY_PATTERN='(ggml_sycl::alloc_handle[[:space:]]+(q8_alloc|dpas_b_alloc|dpas_y_alloc)|cache\.q8_alloc|reuse\.dpas_[by]_alloc|unified_free\(cache\.q8_alloc\))'
SECONDARY_LAYER_TG_LEGACY_PATTERN='(q8_gate_up_alloc|q8_down_alloc|act_full_dev_alloc|act_batch_primary_dev_alloc|scatter_primary_dev_alloc|row_indices_primary_alloc)'
SECONDARY_RING_LEGACY_PATTERN='(dev_q8_1_alloc|dev_out_alloc|dev_q8_batch_alloc|dev_ptrs_alloc|dev_ids_alloc|dev_agg_alloc|dev_reduce_alloc|gate_dev_alloc)'
PERSISTENT_SET_ROWS_VALIDATE_SCOPED_PATTERN='(scoped_unified_alloc[[:space:]]+tmp_alloc\(tmp_req\)|tmp_alloc\.as_mem_handle\(\))'
POINTER_ALLOC_PATTERN='ggml_sycl_malloc_(device|host|shared)(_tracked_bytes|_tracked_t|_t)?\s*(<|\()|GGML_SYCL_MALLOC_(HOST|SHARED)_T\s*\(|ggml_sycl_host_malloc\s*\('
POINTER_FREE_PATTERN='unified_free_ptr\s*\(|ggml_sycl_host_free\s*\('
POINTER_FREE_ALLOW_RE='^ggml/src/ggml-sycl/unified-cache\.(cpp|hpp)$'
COPY_PATTERN='(dpct::async_dpct_memcpy|ctx\.stream\(\)->memcpy|main_stream->memcpy|target_queue->memcpy|primary_queue->memcpy|stream->memcpy|->memcpy\(|queue_\.memcpy|(^|[^[:alnum:]_:])q\.memcpy|(^|[^[:alnum:]_:])cgh\.memcpy|(^|[^[:alnum:]_:])host_q\.memcpy|\.memcpy\()'
COPY_ALLOW_RE='^ggml/src/ggml-sycl/(common\.hpp|mem-ops\.cpp|unified-cache\.cpp)$'
FILL_PATTERN='(ctx\.stream\(\)->(memset|fill)|main_stream->(memset|fill)|target_queue->(memset|fill)|primary_queue->(memset|fill)|stream->(memset|fill)|->(memset|fill)\(|queue_\.memset|queue_\.fill|(^|[^[:alnum:]_:])q\.memset|(^|[^[:alnum:]_:])q\.fill|(^|[^[:alnum:]_:])cgh\.memset|(^|[^[:alnum:]_:])cgh\.fill|\.memset\()'
FILL_ALLOW_RE='^ggml/src/ggml-sycl/mem-ops\.cpp$'

RG_ARGS=(-n --no-heading --with-filename --glob '*.cpp' --glob '*.cc' --glob '*.cxx' --glob '*.hpp' --glob '*.h' --glob '!**/dpct/**')
# The default scan root is already inside ggml/src/ggml-sycl, but keep this
# exclusion when callers pass that tree explicitly.
if [[ "$SCAN_ROOT" == "ggml/src/ggml-sycl" || "$SCAN_ROOT" == "ggml/src/ggml-sycl/"* ]]; then
    RG_ARGS+=(--glob '!**/tests/**')
fi

violations=0

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$PATTERN" <<<"$code"; then
        continue
    fi
    if [[ "$rel" =~ $ALLOW_RE ]]; then
        continue
    fi
    if [[ "$rel" == "ggml/src/ggml-sycl/unified-cache.hpp" ]] &&
       [[ "$code" =~ unified_cache_raw_(malloc_(device|host)|free_device)[[:space:]]*\( ]]; then
        continue
    fi
    if [[ "$rel" == "ggml/src/ggml-sycl/ggml-sycl.cpp" ]]; then
        if [[ "$code" =~ ^[[:space:]]*static[[:space:]]+(void[[:space:]]*\*|bool)[[:space:]]+ggml_sycl_(malloc|free)_device_raw[[:space:]]*\( ]]; then
            continue
        fi
        if [[ "$code" == *"ggml_sycl_malloc_device_raw"* && "$code" == *'"probe_buffer"'* ]]; then
            continue
        fi
        if [[ "$code" == *"ggml_sycl_malloc_device_raw"* && "$code" == *'"peer_probe_src"'* ]]; then
            continue
        fi
        if [[ "$code" == *"ggml_sycl_malloc_device_raw"* && "$code" == *'"peer_probe_dst"'* ]]; then
            continue
        fi
        if [[ "$code" == *"ggml_sycl_free_device_raw"* && "$code" == *'"probe_buffer"'* ]]; then
            continue
        fi
        if [[ "$code" == *"ggml_sycl_free_device_raw"* && "$code" == *'"peer_probe_src"'* ]]; then
            continue
        fi
        if [[ "$code" == *"ggml_sycl_free_device_raw"* && "$code" == *'"peer_probe_dst"'* ]]; then
            continue
        fi
        if [[ "$code" == *"unified_cache_raw_malloc_device(size, queue)"* ]]; then
            continue
        fi
        if [[ "$code" == *"unified_cache_raw_free_device(ptr, queue)"* ]]; then
            continue
        fi
    fi
    echo "forbidden raw SYCL alloc: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$HOST_ALLOC_PATTERN" <<<"$code"; then
        continue
    fi
    if [[ "$rel" == "ggml/src/ggml-sycl/ggml-sycl.cpp" ]] &&
       [[ "$code" == *"std::free(header)"* ]]; then
        continue
    fi
    echo "forbidden raw host alloc/free: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$HOST_ALLOC_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$VM_ALLOC_PATTERN" <<<"$code"; then
        continue
    fi
    if [[ "$rel" == "ggml/src/ggml-sycl/ggml-sycl.cpp" ]]; then
        if [[ "$code" == *"mmap(nullptr, mapping_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS"* ]]; then
            continue
        fi
        if [[ "$code" == *"munmap(mapping, mapping_size)"* ]]; then
            continue
        fi
        if [[ "$code" == *"munmap(header_direct->mapping_base, header_direct->mapping_size)"* ]]; then
            continue
        fi
        if [[ "$code" == *"posix_memalign(&base, alignment, aligned_total)"* ]]; then
            continue
        fi
    fi
    if [[ "$rel" == "ggml/src/ggml-sycl/unified-cache.hpp" ]]; then
        if [[ "$code" == *"mmap(nullptr, mapping_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS"* ]]; then
            continue
        fi
        if [[ "$code" == *"munmap(mapping, mapping_size)"* ]]; then
            continue
        fi
        if [[ "$code" == *"munmap(header->mapping_base, header->mapping_size)"* ]]; then
            continue
        fi
    fi
    echo "forbidden raw host VM/posix allocation API: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$VM_ALLOC_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$RAW_CACHE_API_PATTERN" <<<"$code"; then
        continue
    fi
    if [[ "$rel" =~ $RAW_CACHE_API_ALLOW_RE ]]; then
        continue
    fi
    echo "forbidden raw unified-cache allocation API: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$RAW_CACHE_API_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$RAW_ZONE_API_PATTERN" <<<"$code"; then
        continue
    fi
    if [[ "$rel" =~ $RAW_ZONE_API_ALLOW_RE ]]; then
        continue
    fi
    echo "forbidden raw unified-cache zone allocation/free API: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$RAW_ZONE_API_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$SYNTHETIC_ZONE_HANDLE_PATTERN" <<<"$code"; then
        continue
    fi
    if [[ "$rel" =~ $SYNTHETIC_ZONE_HANDLE_ALLOW_RE ]]; then
        continue
    fi
    echo "forbidden synthetic zone-managed alloc_handle construction: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$SYNTHETIC_ZONE_HANDLE_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$XMX_LEGACY_OWNERSHIP_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden XMX MXFP4 raw ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$XMX_LEGACY_OWNERSHIP_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$XMX_DEVICE_TMP_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden XMX tiled temp legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$XMX_DEVICE_TMP_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$SYCL_EXT_DEVICE_TEMP_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden legacy sycl_ext temp allocation helper name: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$SYCL_EXT_DEVICE_TEMP_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$XMX_TILED_OWNER_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden XMX tiled legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$XMX_TILED_OWNER_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$SPEC_VERIFY_LOGITS_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden speculative logits legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$SPEC_VERIFY_LOGITS_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$PP_STAGE_LEGACY_OWNERSHIP_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden PP stage raw ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$PP_STAGE_LEGACY_OWNERSHIP_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$PP_PIPELINE_LEGACY_OWNERSHIP_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden PP pipeline scratch alloc_handle ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$PP_PIPELINE_LEGACY_OWNERSHIP_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$FATTN_SPLIT_LEGACY_OWNERSHIP_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden FATTN split workspace legacy allocation metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$FATTN_SPLIT_LEGACY_OWNERSHIP_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$FATTN_PACKED_K_LEGACY_OWNERSHIP_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden FATTN packed-K legacy allocation metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$FATTN_PACKED_K_LEGACY_OWNERSHIP_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$FATTN_ONEDNN_MATERIALIZED_RAW_RELEASE_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden oneDNN FATTN materialized Q/K/V raw release path: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$FATTN_ONEDNN_MATERIALIZED_RAW_RELEASE_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$FATTN_ONEDNN_MATERIALIZED_SCOPED_ALLOC_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden oneDNN FATTN materialized Q/K/V scoped alloc/as_mem_handle path: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$FATTN_ONEDNN_MATERIALIZED_SCOPED_ALLOC_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ "$rel" != "ggml/src/ggml-sycl/fattn-buffers.cpp" && "$rel" != tests/sycl-alloc-policy-fixtures/bad-fattn-buffers-kv-legacy-owner/* ]]; then
        continue
    fi
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$FATTN_BUFFERS_KV_LEGACY_OWNER_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden FATTN KV buffer legacy alloc_handle handoff: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$FATTN_BUFFERS_KV_LEGACY_OWNER_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ "$rel" != "ggml/src/ggml-sycl/ggml-sycl.cpp" && "$rel" != tests/sycl-alloc-policy-fixtures/bad-onednn-xmx-fill-scoped-stage/* ]]; then
        continue
    fi
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$ONEDNN_XMX_FILL_SCOPED_STAGE_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden oneDNN/XMX fill scoped staging allocation/as_mem_handle bridge: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$ONEDNN_XMX_FILL_SCOPED_STAGE_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ "$rel" != "ggml/src/ggml-sycl/ggml-sycl.cpp" && "$rel" != tests/sycl-alloc-policy-fixtures/bad-ggml-sycl-cpp-scoped-unified-alloc/* ]]; then
        continue
    fi
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$GGML_SYCL_CPP_SCOPED_UNIFIED_ALLOC_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden scoped_unified_alloc in ggml-sycl.cpp production path: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$GGML_SYCL_CPP_SCOPED_UNIFIED_ALLOC_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$LAYER_STREAM_LEGACY_OWNERSHIP_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden layer-streaming legacy buffer allocation metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$LAYER_STREAM_LEGACY_OWNERSHIP_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ "$rel" != "ggml/src/ggml-sycl/layer-streaming.cpp" && "$rel" != tests/sycl-alloc-policy-fixtures/bad-layer-stream-legacy-owner/* ]]; then
        continue
    fi
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$LAYER_STREAM_LEGACY_OWNER_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden layer-streaming alloc_handle/from_owned_alloc handoff: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$LAYER_STREAM_LEGACY_OWNER_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$COMPUTE_BUFFER_LEGACY_OWNERSHIP_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden compute-buffer legacy allocation metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$COMPUTE_BUFFER_LEGACY_OWNERSHIP_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ "$rel" != "ggml/src/ggml-sycl/compute-buffer-manager.cpp" && "$rel" != tests/sycl-alloc-policy-fixtures/bad-compute-buffer-manager-legacy-owner/* ]]; then
        continue
    fi
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$COMPUTE_BUFFER_MANAGER_LEGACY_OWNER_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden compute-buffer-manager alloc_handle/from_owned_alloc handoff: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$COMPUTE_BUFFER_MANAGER_LEGACY_OWNER_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ "$rel" != "ggml/src/ggml-sycl/gpu-sampler.hpp" && "$rel" != tests/sycl-alloc-policy-fixtures/bad-gpu-sampler-legacy-owner/* ]]; then
        continue
    fi
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$GPU_SAMPLER_LEGACY_OWNER_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden GPU sampler alloc_handle/from_owned_alloc handoff: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$GPU_SAMPLER_LEGACY_OWNER_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$BACKEND_CONTEXT_LEGACY_STAGING_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden backend-context raw staging ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$BACKEND_CONTEXT_LEGACY_STAGING_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$CONTROL_HOST_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden host-control legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$CONTROL_HOST_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$FFN_NORM_LEGACY_OWNERSHIP_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden FFN norm cache legacy allocation metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$FFN_NORM_LEGACY_OWNERSHIP_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$TP_INPUT_LEGACY_OWNERSHIP_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden TP input cache legacy allocation metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$TP_INPUT_LEGACY_OWNERSHIP_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$TP_INPUT_CAPTURE_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden TP captured input legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$TP_INPUT_CAPTURE_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$CPU_DISPATCH_HOST_COPY_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden CPU-dispatch host-copy legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$CPU_DISPATCH_HOST_COPY_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ "$rel" != "ggml/src/ggml-sycl/cpu-dispatch.cpp" && "$rel" != tests/sycl-alloc-policy-fixtures/bad-cpu-dispatch-host-copy-legacy-owner/* ]]; then
        continue
    fi
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$CPU_DISPATCH_HOST_COPY_LEGACY_OWNER_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden CPU-dispatch host-copy alloc_handle/from_owned_alloc handoff: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$CPU_DISPATCH_HOST_COPY_LEGACY_OWNER_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$USM_POINTER_TYPE_CACHE_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden raw-pointer USM type cache: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$USM_POINTER_TYPE_CACHE_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$WEIGHT_IDENTITY_POINTER_CACHE_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden pointer-keyed SYCL weight identity cache: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$WEIGHT_IDENTITY_POINTER_CACHE_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$HOST_WEIGHT_EXTRAS_POINTER_REGISTRY_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden pointer-keyed SYCL host-weight extras registry: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$HOST_WEIGHT_EXTRAS_POINTER_REGISTRY_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$CPU_DISPATCH_HOST_PTR_SPLIT_REGISTRY_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden split raw-pointer CPU-dispatch host pointer registry: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$CPU_DISPATCH_HOST_PTR_SPLIT_REGISTRY_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$CPU_DISPATCH_EXPERT_GROUP_POINTER_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden CPU-dispatch expert grouping by raw weight_host pointer: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$CPU_DISPATCH_EXPERT_GROUP_POINTER_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$STAGING_CACHE_LEGACY_OWNERSHIP_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden staging-cache legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$STAGING_CACHE_LEGACY_OWNERSHIP_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ "$rel" != "ggml/src/ggml-sycl/mem-ops.cpp" && "$rel" != tests/sycl-alloc-policy-fixtures/bad-mem-ops-stage-legacy-owner/* ]]; then
        continue
    fi
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$MEM_OPS_STAGE_LEGACY_OWNER_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden mem-ops staging alloc_handle/from_owned_alloc handoff: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$MEM_OPS_STAGE_LEGACY_OWNER_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ "$rel" != "ggml/src/ggml-sycl/cpy.cpp" && "$rel" != tests/sycl-alloc-policy-fixtures/bad-cpy-host-stage-legacy-owner/* ]]; then
        continue
    fi
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$CPY_HOST_STAGE_LEGACY_OWNER_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden CPY host-stage alloc_handle/from_owned_alloc handoff: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$CPY_HOST_STAGE_LEGACY_OWNER_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ "$rel" != "ggml/src/ggml-sycl/set_rows.cpp" && "$rel" != tests/sycl-alloc-policy-fixtures/bad-set-rows-host-stage-legacy-owner/* ]]; then
        continue
    fi
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$SET_ROWS_HOST_STAGE_LEGACY_OWNER_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden SET_ROWS host-stage alloc_handle/from_owned_alloc handoff: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$SET_ROWS_HOST_STAGE_LEGACY_OWNER_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ "$rel" != "ggml/src/ggml-sycl/dmmv.cpp" && "$rel" != tests/sycl-alloc-policy-fixtures/bad-dmmv-stage-scratch-legacy-owner/* ]]; then
        continue
    fi
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$DMMV_STAGE_SCRATCH_LEGACY_OWNER_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden DMMV staging/scratch alloc_handle/from_owned_alloc handoff: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$DMMV_STAGE_SCRATCH_LEGACY_OWNER_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ "$rel" != "ggml/src/ggml-sycl/convert.cpp" && "$rel" != tests/sycl-alloc-policy-fixtures/bad-convert-device-scratch-legacy-owner/* ]]; then
        continue
    fi
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$CONVERT_DEVICE_SCRATCH_LEGACY_OWNER_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden CONVERT device-scratch alloc_handle/from_owned_alloc handoff: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$CONVERT_DEVICE_SCRATCH_LEGACY_OWNER_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ "$rel" != "ggml/src/ggml-sycl/mmq_xmx.cpp" && "$rel" != tests/sycl-alloc-policy-fixtures/bad-mmq-xmx-device-scratch-legacy-owner/* ]]; then
        continue
    fi
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$MMQ_XMX_DEVICE_SCRATCH_LEGACY_OWNER_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden MMQ_XMX device-scratch legacy alloc_handle handoff: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$MMQ_XMX_DEVICE_SCRATCH_LEGACY_OWNER_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ "$rel" != "ggml/src/ggml-sycl/dense-scheduler.cpp" && "$rel" != tests/sycl-alloc-policy-fixtures/bad-dense-scheduler-slot-legacy-owner/* ]]; then
        continue
    fi
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$DENSE_SCHEDULER_SLOT_LEGACY_OWNER_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden dense-scheduler slot alloc_handle/from_owned_alloc handoff: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$DENSE_SCHEDULER_SLOT_LEGACY_OWNER_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ "$rel" != "ggml/src/ggml-sycl/mmq.cpp" && "$rel" != tests/sycl-alloc-policy-fixtures/bad-mmq-stage-counter-legacy-owner/* ]]; then
        continue
    fi
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$MMQ_STAGE_COUNTER_LEGACY_OWNER_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden MMQ staging/counter alloc_handle/from_owned_alloc handoff: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$MMQ_STAGE_COUNTER_LEGACY_OWNER_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ "$rel" != "ggml/src/ggml-sycl/mmvq.cpp" && "$rel" != tests/sycl-alloc-policy-fixtures/bad-mmvq-stage-reuse-legacy-owner/* ]]; then
        continue
    fi
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$MMVQ_STAGE_REUSE_LEGACY_OWNER_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden MMVQ staging/reuse alloc_handle/from_owned_alloc handoff: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$MMVQ_STAGE_REUSE_LEGACY_OWNER_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ "$rel" != "ggml/src/ggml-sycl/mmvq.cpp" && "$rel" != tests/sycl-alloc-policy-fixtures/bad-mmvq-graph-compact-legacy-owner/* ]]; then
        continue
    fi
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$MMVQ_GRAPH_COMPACT_LEGACY_OWNER_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden MMVQ graph/compact alloc_handle/from_owned_alloc handoff: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$MMVQ_GRAPH_COMPACT_LEGACY_OWNER_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ "$rel" != "ggml/src/ggml-sycl/mmvq-rmsnorm.hpp" && "$rel" != tests/sycl-alloc-policy-fixtures/bad-mmvq-rmsnorm-scales-legacy-owner/* ]]; then
        continue
    fi
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$MMVQ_RMSNORM_SCALES_LEGACY_OWNER_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden MMVQ RMSNorm scales alloc_handle/from_owned_alloc handoff: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$MMVQ_RMSNORM_SCALES_LEGACY_OWNER_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ "$rel" != "ggml/src/ggml-sycl/getrows.cpp" && "$rel" != tests/sycl-alloc-policy-fixtures/bad-get-rows-stage-legacy-owner/* ]]; then
        continue
    fi
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$GET_ROWS_STAGE_LEGACY_OWNER_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden GET_ROWS staging alloc_handle/from_owned_alloc handoff: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$GET_ROWS_STAGE_LEGACY_OWNER_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$SET_ROWS_STAGING_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden set_rows staging legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$SET_ROWS_STAGING_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$SET_ROWS_STAGING_SCOPED_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden SET_ROWS staging scoped allocation/release bridge: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$SET_ROWS_STAGING_SCOPED_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$BINBCAST_RAW_HOST_STAGE_SCOPED_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden BINBCAST raw-host staging scoped allocation bridge: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$BINBCAST_RAW_HOST_STAGE_SCOPED_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ "$rel" != "ggml/src/ggml-sycl/common.cpp" && "$rel" != tests/sycl-alloc-policy-fixtures/bad-common-host-staging-scoped/* ]]; then
        continue
    fi
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$COMMON_HOST_STAGING_SCOPED_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden common.cpp host staging scoped allocation/as_mem_handle bridge: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$COMMON_HOST_STAGING_SCOPED_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ "$rel" != "ggml/src/ggml-sycl/ggml-sycl.cpp" && "$rel" != tests/sycl-alloc-policy-fixtures/bad-readback-fallback-scoped/* ]]; then
        continue
    fi
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$READBACK_FALLBACK_SCOPED_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden readback fallback scoped staging/as_mem_handle bridge: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$READBACK_FALLBACK_SCOPED_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ "$rel" != "ggml/src/ggml-sycl/ggml-sycl.cpp" && "$rel" != tests/sycl-alloc-policy-fixtures/bad-moe-readback-stage-scoped/* ]]; then
        continue
    fi
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$MOE_READBACK_STAGE_SCOPED_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden MoE readback scoped staging/as_mem_handle bridge: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$MOE_READBACK_STAGE_SCOPED_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ "$rel" != "ggml/src/ggml-sycl/ggml-sycl.cpp" && "$rel" != tests/sycl-alloc-policy-fixtures/bad-moe-phase2-d2h-scoped/* ]]; then
        continue
    fi
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$MOE_PHASE2_D2H_SCOPED_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden MOE phase2 D2H scoped staging/as_mem_handle bridge: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$MOE_PHASE2_D2H_SCOPED_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ "$rel" != "ggml/src/ggml-sycl/ggml-sycl.cpp" && "$rel" != tests/sycl-alloc-policy-fixtures/bad-split-weight-stage-scoped/* ]]; then
        continue
    fi
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$SPLIT_WEIGHT_STAGE_SCOPED_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden split-weight D2H scoped staging/as_mem_handle bridge: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$SPLIT_WEIGHT_STAGE_SCOPED_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$GRAPH_Q8_LEGACY_OWNERSHIP_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden graph Q8 cache legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$GRAPH_Q8_LEGACY_OWNERSHIP_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$GET_ROWS_HOST_STAGE_SCOPED_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden GET_ROWS host index staging scoped allocation/as_mem_handle path: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$GET_ROWS_HOST_STAGE_SCOPED_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$TP_REDUCE_LEGACY_OWNERSHIP_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden TP reduce legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$TP_REDUCE_LEGACY_OWNERSHIP_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$TP_HOST_STAGING_LEGACY_OWNERSHIP_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden TP host-staging legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$TP_HOST_STAGING_LEGACY_OWNERSHIP_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$TP_QUANT_COMM_LEGACY_OWNERSHIP_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden TP quant-comm legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$TP_QUANT_COMM_LEGACY_OWNERSHIP_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$TP_ASYNC_RESULT_LEGACY_OWNERSHIP_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden TP async-result legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$TP_ASYNC_RESULT_LEGACY_OWNERSHIP_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$TP_ALLOC_TMP_LEGACY_OUT_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden TP temp allocation helper legacy alloc_handle output: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$TP_ALLOC_TMP_LEGACY_OUT_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$TP_COMPUTE_BUFFER_LEGACY_OWNERSHIP_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden TP compute-buffer legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$TP_COMPUTE_BUFFER_LEGACY_OWNERSHIP_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$DEV1_KV_CACHE_LEGACY_OWNERSHIP_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden dev1 KV-cache legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$DEV1_KV_CACHE_LEGACY_OWNERSHIP_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$TP_COLUMN_OUTPUT_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden TP column-output legacy allocation cache: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$TP_COLUMN_OUTPUT_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$ROUTING_INDICES_LEGACY_OWNERSHIP_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden routing-indices legacy host allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$ROUTING_INDICES_LEGACY_OWNERSHIP_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$MOE_IDS_LEGACY_OWNERSHIP_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden MoE IDs legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$MOE_IDS_LEGACY_OWNERSHIP_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$MOE_IDS_PACK_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden MoE IDs packed staging legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$MOE_IDS_PACK_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$MOE_FUSION_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden MoE fusion legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$MOE_FUSION_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$MOE_PTR_TABLE_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden MoE pointer-table legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$MOE_PTR_TABLE_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$MOE_GPU_PROBE_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden MoE GPU probe legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$MOE_GPU_PROBE_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$GPU_REORDER_TEMP_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden GPU reorder temp VRAM legacy ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$GPU_REORDER_TEMP_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$REORDER_D2H_STAGE_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden reorder D2H staging legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$REORDER_D2H_STAGE_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$PAYLOAD_STAGE_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden payload staging legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$PAYLOAD_STAGE_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if [[ "$rel" == ggml/src/ggml-sycl/* ]] && ! [[ "$rel" =~ $GENERIC_OWNED_ALLOC_HANDOFF_ENFORCED_RE ]]; then
        continue
    fi
    if ! rg -q "$GENERIC_OWNED_ALLOC_HANDOFF_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden generic alloc_handle ownership handoff: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$GENERIC_OWNED_ALLOC_HANDOFF_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$RUNTIME_LOOKUP_OWNER_ALLOC_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden runtime lookup owner_alloc scratch name: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$RUNTIME_LOOKUP_OWNER_ALLOC_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$BACKEND_BUFFER_CONTEXT_LEGACY_OWNER_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden backend buffer context alloc_handle owner field: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$BACKEND_BUFFER_CONTEXT_LEGACY_OWNER_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$BACKEND_BUFFER_CONTEXT_DIRECT_ASSIGN_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden backend buffer context raw alloc_handle assignment: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$BACKEND_BUFFER_CONTEXT_DIRECT_ASSIGN_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$BACKEND_BUFFER_CONTEXT_AS_MEM_HANDLE_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden backend buffer context alloc_handle.as_mem_handle rebuild: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$BACKEND_BUFFER_CONTEXT_AS_MEM_HANDLE_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$BACKEND_BUFFER_CONTEXT_LOOKUP_FREE_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden backend buffer context raw lookup/free fallback: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$BACKEND_BUFFER_CONTEXT_LOOKUP_FREE_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$SYCL_POOL_LEGACY_OWNERSHIP_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden SYCL pool legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$SYCL_POOL_LEGACY_OWNERSHIP_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$TENSOR_EXTRA_DATA_ALLOC_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden tensor extra legacy data_alloc ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$TENSOR_EXTRA_DATA_ALLOC_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$HOST_BUFFER_CONTEXT_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden SYCL host buffer context legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$HOST_BUFFER_CONTEXT_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if [[ "$rel" == "ggml/src/ggml-sycl/unified-cache.cpp" ]]; then
        continue
    fi
    if ! rg -q "$VRAM_POOL_LEGACY_OWNERSHIP_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden VRAM pool legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$VRAM_POOL_LEGACY_OWNERSHIP_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$UNIFIED_ALLOCATE_DIRECT_HANDLE_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden unified_allocate direct mem_handle without allocation ownership: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$UNIFIED_ALLOCATE_DIRECT_HANDLE_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$MANAGED_HOST_PINNED_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden managed_host_pinned_buffer legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$MANAGED_HOST_PINNED_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$ALLOCATE_MANAGED_HOST_PINNED_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden allocate_managed_host_pinned legacy allocation ownership handoff: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$ALLOCATE_MANAGED_HOST_PINNED_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$STAGING_BUFFER_POOL_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden staging_buffer_pool legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$STAGING_BUFFER_POOL_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$CPU_FALLBACK_HOST_COPY_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden CPU fallback host-copy legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$CPU_FALLBACK_HOST_COPY_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$TIERED_KV_ZONE_H_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden tiered KV zone_h legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$TIERED_KV_ZONE_H_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$ONEDNN_SCRATCH_LEGACY_OWNER_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden oneDNN scratch legacy alloc_handle ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$ONEDNN_SCRATCH_LEGACY_OWNER_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$PERSISTENT_SCRATCH_LEGACY_OWNER_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden persistent scratch legacy alloc_handle ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$PERSISTENT_SCRATCH_LEGACY_OWNER_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$COMPUTE_ARENA_LEGACY_OWNER_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden compute arena legacy alloc_handle ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$COMPUTE_ARENA_LEGACY_OWNER_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$SCRATCH_POOL_LEGACY_OWNER_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden scratch pool legacy alloc_handle ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$SCRATCH_POOL_LEGACY_OWNER_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$STAGING_OWNER_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden staging buffer legacy alloc_handle ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$STAGING_OWNER_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$PARTIAL_ROW_CACHE_LEGACY_OWNER_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden partial row cache legacy alloc_handle ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$PARTIAL_ROW_CACHE_LEGACY_OWNER_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$DIRECT_ALLOC_OWNER_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden direct/deferred legacy alloc_handle ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$DIRECT_ALLOC_OWNER_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$MOE_PRESTAGE_CPU_REORDER_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden MoE prestage CPU reorder legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$MOE_PRESTAGE_CPU_REORDER_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$SET_TENSOR_REORDER_FALLBACK_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden set_tensor reorder fallback legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$SET_TENSOR_REORDER_FALLBACK_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$EXPERT_PREFETCH_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden expert-prefetch legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$EXPERT_PREFETCH_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$FP16_CACHE_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden FP16 cache legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$FP16_CACHE_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$PINNED_BUFFER_POOL_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden pinned-buffer-pool legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$PINNED_BUFFER_POOL_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ "$rel" != "ggml/src/ggml-sycl/pinned-buffer-pool.cpp" && "$rel" != tests/sycl-alloc-policy-fixtures/bad-pinned-buffer-pool-legacy-owner/* ]]; then
        continue
    fi
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$PINNED_BUFFER_POOL_LEGACY_OWNER_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden pinned-buffer-pool alloc_handle/from_owned_alloc handoff: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$PINNED_BUFFER_POOL_LEGACY_OWNER_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ "$rel" != "ggml/src/ggml-sycl/pinned-pool.cpp" && "$rel" != tests/sycl-alloc-policy-fixtures/bad-pinned-pool-chunk-legacy-owner/* ]]; then
        continue
    fi
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$PINNED_POOL_CHUNK_LEGACY_OWNER_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden pinned-pool chunk alloc_handle owner/free path: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$PINNED_POOL_CHUNK_LEGACY_OWNER_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$PENDING_CPU_LEGACY_OWNERSHIP_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden pending CPU scatter/pipeline legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$PENDING_CPU_LEGACY_OWNERSHIP_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$CPU_EXPERT_POOL_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden CPU expert pool legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$CPU_EXPERT_POOL_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ "$rel" != "ggml/src/ggml-sycl/cpu-expert-pool.cpp" && "$rel" != tests/sycl-alloc-policy-fixtures/bad-cpu-expert-pool-legacy-owner/* ]]; then
        continue
    fi
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$CPU_EXPERT_POOL_LEGACY_OWNER_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden CPU expert pool alloc_handle/from_owned_alloc handoff: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$CPU_EXPERT_POOL_LEGACY_OWNER_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ "$rel" != "ggml/src/ggml-sycl/cont-batching.hpp" && "$rel" != tests/sycl-alloc-policy-fixtures/bad-cont-batching-legacy-owner/* ]]; then
        continue
    fi
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$CONT_BATCH_ALLOC_LEGACY_OWNER_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden continuous-batching alloc_handle/from_owned_alloc handoff: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$CONT_BATCH_ALLOC_LEGACY_OWNER_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$UNIFIED_KERNEL_MICROGRAPH_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden unified-kernel micro-graph legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$UNIFIED_KERNEL_MICROGRAPH_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$UNIFIED_KERNEL_LIGHT_FLAGS_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden unified-kernel light-flags legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$UNIFIED_KERNEL_LIGHT_FLAGS_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$UNIFIED_KERNEL_ROLE_SCHEDULE_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden unified-kernel role-schedule legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$UNIFIED_KERNEL_ROLE_SCHEDULE_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$UNIFIED_KERNEL_OPS_POOL_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden unified-kernel ops-pool legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$UNIFIED_KERNEL_OPS_POOL_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$UNIFIED_KERNEL_PHASE_SCHEDULE_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden unified-kernel phase-schedule legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$UNIFIED_KERNEL_PHASE_SCHEDULE_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$UNIFIED_KERNEL_GET_ROWS_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden unified-kernel get-rows legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$UNIFIED_KERNEL_GET_ROWS_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$UNIFIED_KERNEL_GET_ROWS_SHARED_HANDLE_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden unified-kernel shared GET_ROWS handle ownership: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$UNIFIED_KERNEL_GET_ROWS_SHARED_HANDLE_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$UNIFIED_KERNEL_SCRATCH_POOL_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden unified-kernel scratch-pool legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$UNIFIED_KERNEL_SCRATCH_POOL_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$UNIFIED_KERNEL_DEFERRED_COPY_RAW_SCRATCH_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden unified-kernel deferred-copy raw scratch handle reconstruction: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$UNIFIED_KERNEL_DEFERRED_COPY_RAW_SCRATCH_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$UNIFIED_KERNEL_DEFERRED_COPY_EXEC_RAW_FALLBACK_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden unified-kernel deferred-copy execution raw handle fallback: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$UNIFIED_KERNEL_DEFERRED_COPY_EXEC_RAW_FALLBACK_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$UNIFIED_KERNEL_DEFERRED_COPY_RAW_API_BRIDGE_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden unified-kernel deferred-copy raw API handle bridge: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$UNIFIED_KERNEL_DEFERRED_COPY_RAW_API_BRIDGE_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$UNIFIED_KERNEL_FINAL_COPY_RAW_DST_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden unified-kernel final copy-back raw destination handle bridge: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$UNIFIED_KERNEL_FINAL_COPY_RAW_DST_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$UNIFIED_KERNEL_RAW_HANDLE_BRIDGE_DEF_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden unified-kernel raw pointer handle bridge definition: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$UNIFIED_KERNEL_RAW_HANDLE_BRIDGE_DEF_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$UNIFIED_KERNEL_FINAL_COPY_OP_INDEX_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden unified-kernel final copy-back op-index/raw registration: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$UNIFIED_KERNEL_FINAL_COPY_OP_INDEX_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$PERSISTENT_TG_DEFERRED_COPY_RAW_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden persistent TG deferred-copy raw registration: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$PERSISTENT_TG_DEFERRED_COPY_RAW_PATTERN" "$SCAN_ROOT" || true
)

if persistent_split_input_matches="$(cd "$ROOT_DIR" &&
    rg -U "${RG_ARGS[@]}" "$PERSISTENT_SPLIT_INPUT_RAW_REWRAP_PATTERN" "$SCAN_ROOT" || true)"; then
    if [[ -n "$persistent_split_input_matches" ]]; then
        echo "forbidden persistent split activation input raw mem_handle rewrap:" >&2
        echo "$persistent_split_input_matches" >&2
        violations=$((violations + 1))
    fi
fi

if pending_device_token_matches="$(cd "$ROOT_DIR" &&
    rg -U "${RG_ARGS[@]}" "$PENDING_DEVICE_TOKEN_RAW_REWRAP_PATTERN" "$SCAN_ROOT" || true)"; then
    if [[ -n "$pending_device_token_matches" ]]; then
        echo "forbidden pending device-token raw mem_handle rewrap:" >&2
        echo "$pending_device_token_matches" >&2
        violations=$((violations + 1))
    fi
fi

if pending_device_token_state_matches="$(cd "$ROOT_DIR" &&
    rg -U "${RG_ARGS[@]}" "$PENDING_DEVICE_TOKEN_RAW_STATE_PATTERN" "$SCAN_ROOT" || true)"; then
    if [[ -n "$pending_device_token_state_matches" ]]; then
        echo "forbidden pending device-token raw retained state:" >&2
        echo "$pending_device_token_state_matches" >&2
        violations=$((violations + 1))
    fi
fi

if seq_ids_state_matches="$(cd "$ROOT_DIR" &&
    rg -U "${RG_ARGS[@]}" "$SEQ_IDS_RAW_STATE_PATTERN" "$SCAN_ROOT" || true)"; then
    if [[ -n "$seq_ids_state_matches" ]]; then
        echo "forbidden seq_ids raw retained state:" >&2
        echo "$seq_ids_state_matches" >&2
        violations=$((violations + 1))
    fi
fi

if moe_weights_by_layer_matches="$(cd "$ROOT_DIR" &&
    rg -U "${RG_ARGS[@]}" "$MOE_WEIGHTS_BY_LAYER_POINTER_PATTERN" "$SCAN_ROOT" || true)"; then
    if [[ -n "$moe_weights_by_layer_matches" ]]; then
        echo "forbidden MoE weights-by-layer tensor pointer cache:" >&2
        echo "$moe_weights_by_layer_matches" >&2
        violations=$((violations + 1))
    fi
fi

if moe_precomputed_skip_matches="$(cd "$ROOT_DIR" &&
    rg -U "${RG_ARGS[@]}" "$MOE_PRECOMPUTED_SKIP_POINTER_SET_PATTERN" "$SCAN_ROOT" || true)"; then
    if [[ -n "$moe_precomputed_skip_matches" ]]; then
        echo "forbidden MoE precomputed-skip tensor pointer set:" >&2
        echo "$moe_precomputed_skip_matches" >&2
        violations=$((violations + 1))
    fi
fi

if graph_prestage_pointer_matches="$(cd "$ROOT_DIR" &&
    rg -U "${RG_ARGS[@]}" "$GRAPH_PRESTAGE_POINTER_DEDUPE_PATTERN" "$SCAN_ROOT" || true)"; then
    if [[ -n "$graph_prestage_pointer_matches" ]]; then
        echo "forbidden graph-prestage raw pointer dedupe:" >&2
        echo "$graph_prestage_pointer_matches" >&2
        violations=$((violations + 1))
    fi
fi

if graph_input_pointer_matches="$(cd "$ROOT_DIR" &&
    rg -U "${RG_ARGS[@]}" "$GRAPH_INPUT_DISCOVERY_POINTER_DEDUPE_PATTERN" "$SCAN_ROOT" || true)"; then
    if [[ -n "$graph_input_pointer_matches" ]]; then
        echo "forbidden graph input-discovery raw pointer dedupe:" >&2
        echo "$graph_input_pointer_matches" >&2
        violations=$((violations + 1))
    fi
fi

if prefetch_scheduler_pointer_matches="$(cd "$ROOT_DIR" &&
    rg -U "${RG_ARGS[@]}" "$PREFETCH_SCHEDULER_POINTER_ACTIVE_SET_PATTERN" "$SCAN_ROOT" || true)"; then
    if [[ -n "$prefetch_scheduler_pointer_matches" ]]; then
        echo "forbidden PrefetchScheduler raw pointer active set:" >&2
        echo "$prefetch_scheduler_pointer_matches" >&2
        violations=$((violations + 1))
    fi
fi

if moe_q8_pointer_cache_matches="$(cd "$ROOT_DIR" &&
    rg -U "${RG_ARGS[@]}" "$MOE_Q8_POINTER_CACHE_PATTERN" "$SCAN_ROOT" || true)"; then
    if [[ -n "$moe_q8_pointer_cache_matches" ]]; then
        echo "forbidden MoE Q8 activation cache raw source pointer key:" >&2
        echo "$moe_q8_pointer_cache_matches" >&2
        violations=$((violations + 1))
    fi
fi

if backend_runtime_reset_matches="$(cd "$ROOT_DIR" &&
    rg -U "${RG_ARGS[@]}" "$BACKEND_BUFFER_RUNTIME_ZONE_RESET_PATTERN" "$SCAN_ROOT" || true)"; then
    if [[ -n "$backend_runtime_reset_matches" ]]; then
        echo "forbidden backend-buffer RUNTIME zone reset retry:" >&2
        echo "$backend_runtime_reset_matches" >&2
        violations=$((violations + 1))
    fi
fi

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$UNIFIED_KERNEL_SYNC_BLOCK_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden unified-kernel sync-block legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$UNIFIED_KERNEL_SYNC_BLOCK_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$UNIFIED_KERNEL_PERSISTENT_BUFFER_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden unified-kernel persistent-buffer legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$UNIFIED_KERNEL_PERSISTENT_BUFFER_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ "$rel" != "ggml/src/ggml-sycl/unified-kernel.cpp" && "$rel" != tests/sycl-alloc-policy-fixtures/bad-unified-kernel-device-persistent-owner/* ]]; then
        continue
    fi
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$UNIFIED_KERNEL_DEVICE_PERSISTENT_LEGACY_OWNER_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden unified-kernel device-persistent alloc_handle/from_owned_alloc handoff: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$UNIFIED_KERNEL_DEVICE_PERSISTENT_LEGACY_OWNER_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ "$rel" != "ggml/src/ggml-sycl/unified-kernel.cpp" && "$rel" != tests/sycl-alloc-policy-fixtures/bad-unified-kernel-pinned-persistent-owner/* ]]; then
        continue
    fi
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$UNIFIED_KERNEL_PINNED_PERSISTENT_LEGACY_OWNER_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden unified-kernel pinned-persistent alloc_handle/from_owned_alloc handoff: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$UNIFIED_KERNEL_PINNED_PERSISTENT_LEGACY_OWNER_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ "$rel" != "ggml/src/ggml-sycl/unified-kernel.cpp" && "$rel" != tests/sycl-alloc-policy-fixtures/bad-unified-kernel-device-scratch-owner/* ]]; then
        continue
    fi
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$UNIFIED_KERNEL_DEVICE_SCRATCH_LEGACY_OWNER_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden unified-kernel device-scratch alloc_handle/from_owned_alloc handoff: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$UNIFIED_KERNEL_DEVICE_SCRATCH_LEGACY_OWNER_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$UNIFIED_KERNEL_DAG_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden unified-kernel DAG legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$UNIFIED_KERNEL_DAG_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$UNIFIED_KERNEL_TEMP_DEVICE_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden unified-kernel temp-device legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$UNIFIED_KERNEL_TEMP_DEVICE_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$UNIFIED_KERNEL_GRAPH_OVERHEAD_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden unified-kernel graph-overhead legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$UNIFIED_KERNEL_GRAPH_OVERHEAD_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$SPLIT_PERSISTENT_Q8_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden split persistent Q8 legacy staging ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$SPLIT_PERSISTENT_Q8_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$SPLIT_PERSISTENT_SYNC_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden split persistent sync legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$SPLIT_PERSISTENT_SYNC_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$MMVQ_SOA_BULK_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden MMVQ SoA bulk legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$MMVQ_SOA_BULK_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$SPLIT_SECONDARY_GPU_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden split secondary GPU legacy staging ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$SPLIT_SECONDARY_GPU_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$SPLIT_SECONDARY_OUTPUT_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden split secondary output legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$SPLIT_SECONDARY_OUTPUT_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$SPLIT_WEIGHT_CACHE_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden split weight cache legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$SPLIT_WEIGHT_CACHE_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$MXFP4_TG_REUSE_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden MXFP4 TG reuse legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$MXFP4_TG_REUSE_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$SECONDARY_LAYER_TG_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden secondary layer TG legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$SECONDARY_LAYER_TG_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$SECONDARY_RING_LEGACY_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden secondary ring-buffer legacy allocation ownership metadata: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$SECONDARY_RING_LEGACY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ "$rel" != "ggml/src/ggml-sycl/ggml-sycl.cpp" && "$rel" != tests/sycl-alloc-policy-fixtures/bad-persistent-set-rows-validate-scoped/* ]]; then
        continue
    fi
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$PERSISTENT_SET_ROWS_VALIDATE_SCOPED_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden persistent SET_ROWS validate scoped tmp allocation/as_mem_handle bridge: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$PERSISTENT_SET_ROWS_VALIDATE_SCOPED_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$POINTER_ALLOC_PATTERN" <<<"$code"; then
        continue
    fi
    echo "forbidden pointer-only device allocation API: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$POINTER_ALLOC_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$POINTER_FREE_PATTERN" <<<"$code"; then
        continue
    fi
    if [[ "$rel" =~ $POINTER_FREE_ALLOW_RE ]]; then
        continue
    fi
    echo "forbidden pointer-only unified free: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$POINTER_FREE_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$COPY_PATTERN" <<<"$code"; then
        continue
    fi
    if [[ "$rel" =~ $COPY_ALLOW_RE ]]; then
        continue
    fi
    if [[ "$rel" == "ggml/src/ggml-sycl/unified-kernel.cpp" ]] &&
       [[ "$code" == *"queue_.memcpy(dag_state_.ready_counter"* ]]; then
        continue
    fi
    if [[ "$rel" == "ggml/src/ggml-sycl/unified-kernel.cpp" ]] &&
       [[ "$code" == *"queue_.memcpy(dc.dst, src, dc.bytes)"* ]]; then
        continue
    fi
    if [[ "$rel" == "ggml/src/ggml-sycl/ggml-sycl.cpp" ]] &&
       [[ "$code" == *"h.memcpy(dst.ptr, ptr_payload.data(), ptr_payload.size() * sizeof(void *))"* ]]; then
        continue
    fi
    echo "forbidden raw SYCL queue copy: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$COPY_PATTERN" "$SCAN_ROOT" || true
)

while IFS= read -r match; do
    rel="${match%%:*}"
    code="${match#*:*:}"
    code="${code%%//*}"
    if [[ -z "${code//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "$code" =~ ^[[:space:]]*// ]] || [[ "$code" =~ ^[[:space:]]*/\* ]] || [[ "$code" =~ ^[[:space:]]*\* ]]; then
        continue
    fi
    if ! rg -q "$FILL_PATTERN" <<<"$code"; then
        continue
    fi
    if [[ "$rel" =~ $FILL_ALLOW_RE ]]; then
        continue
    fi
    if [[ "$rel" == "ggml/src/ggml-sycl/unified-kernel.cpp" ]] &&
       [[ "$code" == *"queue_.memset(dag_state_.tile_claimed"* ]]; then
        continue
    fi
    if [[ "$rel" == "ggml/src/ggml-sycl/unified-kernel.cpp" ]] &&
       [[ "$code" == *"queue_.memset(dag_state_.tiles_done"* ]]; then
        continue
    fi
    if [[ "$rel" == "ggml/src/ggml-sycl/unified-kernel.cpp" ]] &&
       [[ "$code" == *"queue_.memset(dag_state_.completed_count"* ]]; then
        continue
    fi
    echo "forbidden raw SYCL queue fill: $match" >&2
    violations=$((violations + 1))
done < <(
    cd "$ROOT_DIR"
    rg "${RG_ARGS[@]}" "$FILL_PATTERN" "$SCAN_ROOT" || true
)

if [[ $violations -ne 0 ]]; then
    echo "SYCL alloc/copy/fill policy check failed: $violations violation(s)" >&2
    exit 1
fi

echo "SYCL alloc/copy/fill policy check passed"
