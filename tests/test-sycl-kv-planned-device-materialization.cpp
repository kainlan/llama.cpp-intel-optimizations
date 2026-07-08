// Focused regression test for cross-device KV root materialization.
//
// It installs a synthetic placement plan where layer 1 KV belongs to physical
// device 1, allocates a SYCL_KV_Tiered buffer from the scheduler-visible device
// 0 path, and verifies cache_k_l1 records a smart handle for device 1.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml.h"
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/ggml-sycl-test.hpp"
#include "ggml-sycl/unified-cache.hpp"

#include <cstdio>
#include <cstdlib>

#if !defined(GGML_USE_SYCL)
int main() {
    fprintf(stderr, "GGML_USE_SYCL not enabled; skipping test.\n");
    return 0;
}
#else

#    define TEST_ASSERT(cond, msg)                    \
        do {                                          \
            if (!(cond)) {                            \
                fprintf(stderr, "  FAIL: %s\n", msg); \
                return 1;                             \
            }                                         \
        } while (0)

int main() {
    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {
        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:0,1", 1);
    }
    setenv("GGML_SYCL_KV_HOST", "0", 1);
    setenv("GGML_SYCL_VRAM_ARENA", "0", 1);
    setenv("GGML_SYCL_ALLOC_PHASE_GATE", "0", 1);

    const int physical_devices = ggml_sycl::test_physical_device_count();
    if (physical_devices < 2) {
        printf("SKIP: need at least two physical SYCL devices, got %d\n", physical_devices);
        return 0;
    }

    ggml_sycl::unified_cache * cache0 = ggml_sycl::get_unified_cache_for_device(0);
    TEST_ASSERT(cache0 != nullptr, "device 0 unified cache must be initialized");
    if (!cache0->host_zones_configured()) {
        cache0->configure_host_zones(128 * 1024, 256 * 1024, 128 * 1024, 128 * 1024);
    }
    const size_t host_kv_capacity = ggml_sycl::unified_cache_host_zone_capacity(ggml_sycl::host_zone_id::KV);
    TEST_ASSERT(host_kv_capacity > 0, "host KV zone must be configured for rollback test");
    const size_t host_kv_before = ggml_sycl::unified_cache_host_zone_used(ggml_sycl::host_zone_id::KV);
    ggml_sycl::alloc_request host_kv_req{};
    host_kv_req.device                              = 0;
    host_kv_req.size                                = 64 * 1024;
    host_kv_req.intent.role                         = ggml_sycl::alloc_role::KV;
    host_kv_req.intent.category                     = ggml_sycl::runtime_category::KV_CACHE;
    host_kv_req.intent.constraints.must_host_pinned = true;
    host_kv_req.intent.constraints.use_pinned_pool  = true;
    ggml_sycl::alloc_handle host_kv_h{};
    TEST_ASSERT(ggml_sycl::unified_alloc(host_kv_req, &host_kv_h) && host_kv_h.ptr,
                "host KV rollback allocation failed");
    TEST_ASSERT(host_kv_h.host_zone == ggml_sycl::host_zone_id::KV, "host KV allocation must use KV host zone");
    const size_t host_kv_after_alloc = ggml_sycl::unified_cache_host_zone_used(ggml_sycl::host_zone_id::KV);
    TEST_ASSERT(host_kv_after_alloc >= host_kv_before + host_kv_req.size, "host KV zone usage did not increase");
    TEST_ASSERT(ggml_sycl::unified_free(host_kv_h), "host KV rollback free failed");
    const size_t host_kv_after_free = ggml_sycl::unified_cache_host_zone_used(ggml_sycl::host_zone_id::KV);
    TEST_ASSERT(host_kv_after_free == host_kv_before, "host KV zone usage was not reclaimed by unified_free");

    constexpr size_t layer_bytes = 128 * 1024;
    constexpr size_t total_bytes = 2 * layer_bytes;

    ggml_sycl::placement_plan plan;
    plan.multi_device       = true;
    plan.device_id          = -1;
    plan.kv_per_layer       = layer_bytes;
    plan.kv_device[0]       = 0;
    plan.kv_device[1]       = 1;
    plan.layer_device[0]    = 0;
    plan.layer_device[1]    = 1;
    plan.kv_vram_bytes      = total_bytes;
    plan.kv_host_bytes      = 0;
    plan.devices            = { 0, 1 };
    plan.per_device_vram    = { layer_bytes, layer_bytes };
    ggml_sycl::test_set_kv_placement_plan(plan, 2, layer_bytes);

    struct ggml_init_params params = {
        /* .mem_size   = */ 16 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    TEST_ASSERT(ctx != nullptr, "ggml_init failed");

    ggml_tensor * k0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, layer_bytes / sizeof(ggml_fp16_t));
    ggml_tensor * k1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, layer_bytes / sizeof(ggml_fp16_t));
    ggml_set_name(k0, "cache_k_l0");
    ggml_set_name(k1, "cache_k_l1");

    ggml_backend_buffer_type_t buft = ggml_backend_sycl_kv_buffer_type(0);
    ggml_backend_buffer_t      buf  = ggml_backend_buft_alloc_buffer(buft, total_bytes);
    TEST_ASSERT(buf != nullptr, "KV buffer allocation failed");

    TEST_ASSERT(ggml_backend_tensor_alloc_offset(buf, k0, 0) == GGML_STATUS_SUCCESS, "cache_k_l0 allocation failed");
    TEST_ASSERT(ggml_backend_tensor_alloc_offset(buf, k1, layer_bytes) == GGML_STATUS_SUCCESS,
                "cache_k_l1 allocation failed");

    TEST_ASSERT(k1->extra != nullptr, "cache_k_l1 extra must be populated");
    auto * extra = static_cast<ggml_tensor_extra_gpu *>(k1->extra);

    auto dev1 = extra->data_handle[1].resolve();
    TEST_ASSERT(dev1, "cache_k_l1 smart handle for planned device 1 must resolve");
    TEST_ASSERT(dev1.on_device, "cache_k_l1 planned device 1 handle must be device-resident");
    TEST_ASSERT(dev1.ptr == k1->data, "cache_k_l1 tensor data should point at the device-1 materialization");

    const auto * info = ggml_sycl::alloc_registry::instance().lookup(k1->data);
    TEST_ASSERT(info != nullptr, "cache_k_l1 materialization must be registered");
    TEST_ASSERT(info->type == ggml_sycl::alloc_type::DEVICE, "cache_k_l1 must not be represented as HOST_PINNED");
    TEST_ASSERT(info->device_id == 1, "cache_k_l1 registry owner must be device 1");

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_sycl::test_clear_kv_placement_plan();

    printf("SYCL planned-device KV materialization test: PASS\n");
    return 0;
}

#endif
