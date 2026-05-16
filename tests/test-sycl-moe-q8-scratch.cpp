// Regression coverage for MoE graph Q8_1 scratch sizing.

#include "ggml-sycl.h"
#include "ggml-sycl/unified-cache.hpp"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>

#define TEST_ASSERT(cond, msg)                         \
    do {                                               \
        if (!(cond)) {                                 \
            std::fprintf(stderr, "FAIL: %s\n", (msg)); \
            return false;                              \
        }                                              \
    } while (0)

static size_t align_256(size_t n) {
    return (n + 255u) & ~size_t(255u);
}

static bool test_mxfp4_coalesced_decode_scratch_uses_graph_shape() {
    ggml_sycl::moe_q8_1_scratch_shape shape;
    shape.weight_type    = GGML_TYPE_MXFP4;
    shape.layout         = GGML_LAYOUT_COALESCED;
    shape.input_cols     = 2880;
    shape.input_rows     = 4;
    shape.graph_op_count = 24;
    shape.device_id      = 1;
    shape.segment_id     = 7;
    shape.layer_id       = 8;
    shape.n_experts      = 128;
    shape.n_expert_used  = 4;

    const auto demand = ggml_sycl::unified_cache_plan_moe_q8_1_scratch(shape);

    const size_t padded_cols      = 2880;
    const size_t q8_1_row_bytes   = padded_cols + (padded_cols / 32u) * 4u;
    const size_t per_buffer_bytes = 4u * q8_1_row_bytes;

    TEST_ASSERT(demand.supported, "MXFP4 coalesced decode should require Q8_1 scratch");
    TEST_ASSERT(demand.input_cols_padded == padded_cols, "input column padding should use QK8_1 alignment");
    TEST_ASSERT(demand.bytes_per_buffer == per_buffer_bytes, "per-buffer Q8_1 bytes should derive from src1 shape");
    TEST_ASSERT(demand.aligned_bytes_per_buffer == align_256(per_buffer_bytes),
                "each scratch buffer should be 256-byte aligned");
    TEST_ASSERT(demand.buffer_count == 24, "buffer count should derive from graph MUL_MAT_ID count");
    TEST_ASSERT(demand.total_bytes == 24u * align_256(per_buffer_bytes),
                "total scratch should reserve every graph Q8_1 buffer");
    TEST_ASSERT(demand.layout == GGML_LAYOUT_COALESCED, "diagnostic demand should preserve selected layout");
    TEST_ASSERT(demand.weight_type == GGML_TYPE_MXFP4, "diagnostic demand should preserve weight dtype");
    TEST_ASSERT(demand.device_id == 1 && demand.segment_id == 7 && demand.layer_id == 8,
                "diagnostic demand should preserve device/segment/layer shape");
    return true;
}

static bool test_unsupported_dense_dtype_has_zero_demand() {
    ggml_sycl::moe_q8_1_scratch_shape shape;
    shape.weight_type    = GGML_TYPE_F16;
    shape.layout         = GGML_LAYOUT_AOS;
    shape.input_cols     = 4096;
    shape.input_rows     = 4;
    shape.graph_op_count = 12;

    const auto demand = ggml_sycl::unified_cache_plan_moe_q8_1_scratch(shape);

    TEST_ASSERT(!demand.supported, "dense F16 MoE graph should not reserve Q8_1 MMVQ scratch");
    TEST_ASSERT(demand.total_bytes == 0, "unsupported dtype should have zero scratch demand");
    return true;
}

#if defined(GGML_USE_SYCL)
static bool test_graph_scratch_owner_survives_pool_reset_and_growth() {
    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {
        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:1", 1);
    }

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        std::puts("SKIP: SYCL backend unavailable");
        return true;
    }

    ggml_sycl::moe_q8_1_scratch_shape shape;
    shape.weight_type    = GGML_TYPE_MXFP4;
    shape.layout         = GGML_LAYOUT_SOA;
    shape.input_cols     = 2880;
    shape.input_rows     = 4;
    shape.graph_op_count = 4;
    shape.device_id      = 0;
    shape.layer_id       = 3;
    shape.n_experts      = 32;
    shape.n_expert_used  = 4;

    const auto demand = ggml_sycl::unified_cache_plan_moe_q8_1_scratch(shape);

    ggml_sycl::alloc_handle graph_owner{};
    const bool              allocated = ggml_sycl::unified_cache_allocate_moe_q8_1_graph_scratch(0, demand, &graph_owner);
    TEST_ASSERT(allocated && graph_owner.ptr != nullptr, "graph scratch should allocate through unified-cache owner");

    void * graph_ptr = graph_owner.ptr;

    TEST_ASSERT(ggml_sycl::unified_cache_reserve_scratch_pool(0, demand.total_bytes),
                "transient scratch pool reservation should succeed");
    void * transient_a = ggml_sycl::unified_cache_get_scratch(0, demand.bytes_per_buffer);
    TEST_ASSERT(transient_a != nullptr, "transient scratch allocation should succeed");
    TEST_ASSERT(transient_a != graph_ptr, "graph scratch must not alias transient scratch pool");

    ggml_sycl::unified_cache_reset_scratch_pool(0);
    void * transient_b = ggml_sycl::unified_cache_get_scratch(0, demand.bytes_per_buffer);
    TEST_ASSERT(transient_b != nullptr, "transient scratch allocation after reset should succeed");
    TEST_ASSERT(transient_b != graph_ptr, "scratch reset must not recycle graph scratch ownership");

    TEST_ASSERT(ggml_sycl::unified_cache_reserve_scratch_pool(0, demand.total_bytes * 2),
                "transient scratch pool growth should succeed");
    void * transient_c = ggml_sycl::unified_cache_get_scratch(0, demand.bytes_per_buffer);
    TEST_ASSERT(transient_c != nullptr, "transient scratch allocation after growth should succeed");
    TEST_ASSERT(transient_c != graph_ptr, "scratch growth must not invalidate or alias graph scratch ownership");

    const auto graph_loc = ggml_sycl::query_location(graph_ptr, 0);
    TEST_ASSERT(graph_loc.tier == ggml_sycl::alloc_tier::DEVICE_VRAM,
                "graph scratch owner should remain registered as device VRAM after pool reset/growth");

    ggml_sycl::unified_free(graph_owner);
    ggml_backend_free(backend);
    return true;
}
#endif

int main() {
    int failed = 0;
    failed += !test_mxfp4_coalesced_decode_scratch_uses_graph_shape();
    failed += !test_unsupported_dense_dtype_has_zero_demand();
#if defined(GGML_USE_SYCL)
    failed += !test_graph_scratch_owner_survives_pool_reset_and_growth();
#endif
    if (failed) {
        return 1;
    }
    std::puts("PASS: sycl MoE Q8_1 scratch sizing");
    return 0;
}
