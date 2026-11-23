// Settings of upstream's XMX dequant-GEMM (fused-gemm.cpp, fused-gemm-esimd.cpp) and the refusal report of the
// device-built MUL_MAT_ID schedule's scratch admission (mmid-sched-admission.hpp).
//
// Upstream defines these globals in ggml-sycl.cpp and reads most of them from the environment. In this fork only the
// device-built MUL_MAT_ID schedule reaches that code, so only its switch is a knob: ggml_check_sycl() reads
// GGML_SYCL_MMID_SCHED (default 0, docs/backend/sycl-env-vars.md). The others keep upstream's names and defaults as
// fixed values.

#include "common.hpp"
#include "mmid-sched-admission.hpp"

#include <atomic>

// GGML_SYCL_MMID_SCHED: 0 off (default), bit 0 the device-built schedule, bit 1 drops the rows-per-expert width
// heuristic. Written once by ggml_check_sycl().
int g_ggml_sycl_mmid_sched = 0;

// Fixed at upstream's defaults. Upstream's GGML_SYCL_ENABLE_ESIMD switches every upstream ESIMD path and the fork has
// none of those paths, so it stays 1; g_ggml_sycl_mmid_esimd (upstream GGML_SYCL_MMID_ESIMD) picks the grouped ESIMD
// GEMM over the joint_matrix one.
int g_ggml_sycl_enable_esimd               = 1;
int g_ggml_sycl_mmid_esimd                 = 1;
int g_ggml_sycl_xmx_gather_types           = GGML_SYCL_XMX_GATHER_TYPES_DEFAULT;
int g_ggml_sycl_xmx_gather_shapes          = GGML_SYCL_XMX_GATHER_SHAPES_DEFAULT;
int g_ggml_sycl_dynamic_precision          = GGML_SYCL_DYNAMIC_PRECISION_DEFAULT;
int g_ggml_sycl_dynamic_required_precision = GGML_SYCL_DYNAMIC_PRECISION_F32;

namespace ggml_sycl {

void mmid_sched_warn_refusal_once(const mmid_sched_scratch_admission & admission) noexcept {
    static_assert(static_cast<unsigned>(mmid_sched_scratch_reason::COUNT) <= 32, "one bit per reason");
    static std::atomic<uint32_t> logged{ 0 };
    if (admission.allowed) {
        return;
    }
    const uint32_t bit = 1u << static_cast<unsigned>(admission.reason);
    if (logged.fetch_or(bit, std::memory_order_relaxed) & bit) {
        return;
    }
    GGML_LOG_WARN(
        "%s: GGML_SYCL_MMID_SCHED: device MUL_MAT_ID schedule refused (%s): required rows=%zu tiles=%zu offsets=%zu "
        "packed_b=%zu bytes, planned rows=%zu tiles=%zu offsets=%zu packed_b=%zu bytes; the op takes its planned MoE "
        "route\n",
        __func__, mmid_sched_scratch_reason_name(admission.reason), admission.required.row_mapping_bytes,
        admission.required.tile_bytes, admission.required.expert_offset_bytes, admission.required.packed_b_bytes,
        admission.planned.row_mapping_bytes, admission.planned.tile_bytes, admission.planned.expert_offset_bytes,
        admission.planned.packed_b_bytes);
}

}  // namespace ggml_sycl
