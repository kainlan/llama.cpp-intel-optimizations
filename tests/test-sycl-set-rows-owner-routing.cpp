// Regression tests for owner-aware SYCL SET_ROWS routing.

#include "ggml-sycl/common.hpp"
#include "ggml-sycl/set_rows.hpp"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

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
                return false;                         \
            }                                         \
        } while (0)

static void init_tensor(ggml_tensor & tensor, const char * name, ggml_type type) {
    std::memset(&tensor, 0, sizeof(tensor));
    tensor.type  = type;
    tensor.ne[0] = 16;
    tensor.ne[1] = 4;
    tensor.ne[2] = 1;
    tensor.ne[3] = 1;
    tensor.nb[0] = ggml_type_size(type);
    tensor.nb[1] = tensor.nb[0] * tensor.ne[0];
    tensor.nb[2] = tensor.nb[1] * tensor.ne[1];
    tensor.nb[3] = tensor.nb[2] * tensor.ne[2];
    std::snprintf(tensor.name, sizeof(tensor.name), "%s", name);
}

static bool test_set_rows_uses_dst_root_owner_for_kv_view() {
    printf("\n=== Test: SET_ROWS routes KV view writes to dst root owner ===\n");

    alignas(64) std::uint8_t src_dev1[4096]   = {};
    alignas(64) std::uint8_t idx_dev1[4096]   = {};
    alignas(64) std::uint8_t kv_dev0[4096]    = {};
    alignas(64) std::uint8_t kv_dev1[4096]    = {};
    alignas(64) std::uint8_t stale_host[4096] = {};

    ggml_tensor src0;
    ggml_tensor idx;
    ggml_tensor root;
    ggml_tensor view;
    init_tensor(src0, "k_cur_l1", GGML_TYPE_F32);
    init_tensor(idx, "kv_pos_l1", GGML_TYPE_I32);
    init_tensor(root, "cache_k_l1", GGML_TYPE_F16);
    init_tensor(view, "cache_k_l1_view", GGML_TYPE_F16);

    ggml_tensor_extra_gpu src_extra{};
    ggml_tensor_extra_gpu idx_extra{};
    ggml_tensor_extra_gpu root_extra{};
    ggml_tensor_extra_gpu view_extra{};
    src0.extra = &src_extra;
    idx.extra  = &idx_extra;
    root.extra = &root_extra;
    view.extra = &view_extra;

    src_extra.set_data_device(1, src_dev1, GGML_LAYOUT_AOS, true);
    idx_extra.set_data_device(1, idx_dev1, GGML_LAYOUT_AOS, true);
    root_extra.set_data_device(0, kv_dev0, GGML_LAYOUT_AOS, true);
    root_extra.set_data_device(1, kv_dev1, GGML_LAYOUT_AOS, true);
    root.data = kv_dev1;

    view.view_src  = &root;
    view.view_offs = 192;
    view.data      = stale_host + 256;
    view.src[0]    = &src0;
    view.src[1]    = &idx;

    ggml_sycl_set_rows_plan plan{};
    TEST_ASSERT(ggml_sycl_plan_set_rows(&view, 0, &plan), "SET_ROWS plan should resolve");
    TEST_ASSERT(plan.owner_device == 1, "SET_ROWS must choose the dst/root device-1 owner");
    TEST_ASSERT(plan.src0_ptr == src_dev1, "src0 must resolve on the owner device");
    TEST_ASSERT(plan.index_ptr == idx_dev1, "index/control tensor must resolve on the owner device");
    TEST_ASSERT(plan.dst_ptr == kv_dev1 + view.view_offs, "dst view must resolve through the device-1 KV root");
    TEST_ASSERT(!plan.index_needs_staging, "device-resident owner index tensor should not be treated as raw host");

    return true;
}

static bool test_raw_control_indices_require_owner_staging() {
    printf("\n=== Test: raw SET_ROWS control indices are marked for owner staging ===\n");

    alignas(64) std::uint8_t src_dev1[4096] = {};
    alignas(64) std::uint8_t kv_dev1[4096]  = {};
    alignas(64) std::int32_t raw_idx[4]     = { 1, 2, 3, 4 };

    ggml_tensor src0;
    ggml_tensor idx;
    ggml_tensor dst;
    init_tensor(src0, "v_cur_l1", GGML_TYPE_F32);
    init_tensor(idx, "raw_kv_pos", GGML_TYPE_I32);
    init_tensor(dst, "cache_v_l1", GGML_TYPE_F16);

    ggml_tensor_extra_gpu src_extra{};
    ggml_tensor_extra_gpu dst_extra{};
    src0.extra = &src_extra;
    dst.extra  = &dst_extra;
    src_extra.set_data_device(1, src_dev1, GGML_LAYOUT_AOS, true);
    dst_extra.set_data_device(1, kv_dev1, GGML_LAYOUT_AOS, true);
    dst.data = kv_dev1;

    idx.data   = raw_idx;
    dst.src[0] = &src0;
    dst.src[1] = &idx;

    ggml_sycl_set_rows_plan plan{};
    TEST_ASSERT(ggml_sycl_plan_set_rows(&dst, 0, &plan), "SET_ROWS plan should resolve with raw control indices");
    TEST_ASSERT(plan.owner_device == 1, "SET_ROWS must still choose dst owner for raw controls");
    TEST_ASSERT(plan.index_ptr == raw_idx, "raw index pointer should be visible to the staging decision");
    TEST_ASSERT(plan.index_needs_staging, "raw host control indices must be staged before device execution");

    return true;
}

static bool test_wrong_device_inputs_record_source_owner_for_staging() {
    printf("\n=== Test: wrong-device SET_ROWS inputs record source owner for staging ===\n");

    alignas(64) std::uint8_t src_dev0[4096] = {};
    alignas(64) std::uint8_t idx_dev0[4096] = {};
    alignas(64) std::uint8_t kv_dev1[4096]  = {};

    ggml_sycl::alloc_registry::instance().register_alloc(src_dev0, sizeof(src_dev0), 0, ggml_sycl::alloc_type::DEVICE);
    ggml_sycl::alloc_registry::instance().register_alloc(idx_dev0, sizeof(idx_dev0), 0, ggml_sycl::alloc_type::DEVICE);
    ggml_sycl::alloc_registry::instance().register_alloc(kv_dev1, sizeof(kv_dev1), 1, ggml_sycl::alloc_type::DEVICE);

    ggml_tensor src0;
    ggml_tensor idx;
    ggml_tensor dst;
    init_tensor(src0, "k_cur_l1", GGML_TYPE_F32);
    init_tensor(idx, "kv_pos_l1", GGML_TYPE_I32);
    init_tensor(dst, "cache_k_l1", GGML_TYPE_F16);

    ggml_tensor_extra_gpu src_extra{};
    ggml_tensor_extra_gpu idx_extra{};
    ggml_tensor_extra_gpu dst_extra{};
    src0.extra = &src_extra;
    idx.extra  = &idx_extra;
    dst.extra  = &dst_extra;
    src_extra.set_data_device(0, src_dev0, GGML_LAYOUT_AOS, true);
    idx_extra.set_data_device(0, idx_dev0, GGML_LAYOUT_AOS, true);
    dst_extra.set_data_device(1, kv_dev1, GGML_LAYOUT_AOS, true);
    dst.data   = kv_dev1;
    dst.src[0] = &src0;
    dst.src[1] = &idx;

    ggml_sycl_set_rows_plan plan{};
    const bool              ok = ggml_sycl_plan_set_rows(&dst, 0, &plan);

    ggml_sycl::alloc_registry::instance().unregister_alloc(src_dev0);
    ggml_sycl::alloc_registry::instance().unregister_alloc(idx_dev0);
    ggml_sycl::alloc_registry::instance().unregister_alloc(kv_dev1);

    TEST_ASSERT(ok, "SET_ROWS plan should resolve wrong-device inputs");
    TEST_ASSERT(plan.owner_device == 1, "SET_ROWS owner should remain the dst KV owner");
    TEST_ASSERT(plan.src0_needs_staging, "wrong-device src0 must require staging");
    TEST_ASSERT(plan.index_needs_staging, "wrong-device indices must require staging");
    TEST_ASSERT(plan.src0_device == 0, "src0 staging must remember the source device");
    TEST_ASSERT(plan.index_device == 0, "index staging must remember the source device");

    return true;
}

static bool test_wrong_device_handles_win_over_stale_tensor_data() {
    printf("\n=== Test: wrong-device smart handles win over stale tensor data ===\n");

    alignas(64) std::uint8_t stale_src_host[4096] = {};
    alignas(64) std::uint8_t stale_idx_host[4096] = {};
    alignas(64) std::uint8_t src_dev0[4096]       = {};
    alignas(64) std::uint8_t idx_dev0[4096]       = {};
    alignas(64) std::uint8_t kv_dev1[4096]        = {};

    ggml_sycl::alloc_registry::instance().register_alloc(src_dev0, sizeof(src_dev0), 0, ggml_sycl::alloc_type::DEVICE);
    ggml_sycl::alloc_registry::instance().register_alloc(idx_dev0, sizeof(idx_dev0), 0, ggml_sycl::alloc_type::DEVICE);
    ggml_sycl::alloc_registry::instance().register_alloc(kv_dev1, sizeof(kv_dev1), 1, ggml_sycl::alloc_type::DEVICE);

    ggml_tensor src0;
    ggml_tensor idx;
    ggml_tensor dst;
    init_tensor(src0, "k_cur_l1", GGML_TYPE_F32);
    init_tensor(idx, "kv_pos_l1", GGML_TYPE_I32);
    init_tensor(dst, "cache_k_l1", GGML_TYPE_F16);

    ggml_tensor_extra_gpu src_extra{};
    ggml_tensor_extra_gpu idx_extra{};
    ggml_tensor_extra_gpu dst_extra{};
    src0.extra = &src_extra;
    idx.extra  = &idx_extra;
    dst.extra  = &dst_extra;
    src_extra.set_data_device(0, src_dev0, GGML_LAYOUT_AOS, true);
    idx_extra.set_data_device(0, idx_dev0, GGML_LAYOUT_AOS, true);
    dst_extra.set_data_device(1, kv_dev1, GGML_LAYOUT_AOS, true);
    src0.data  = stale_src_host;
    idx.data   = stale_idx_host;
    dst.data   = kv_dev1;
    dst.src[0] = &src0;
    dst.src[1] = &idx;

    ggml_sycl_set_rows_plan plan{};
    const bool              ok = ggml_sycl_plan_set_rows(&dst, 0, &plan);

    ggml_sycl::alloc_registry::instance().unregister_alloc(src_dev0);
    ggml_sycl::alloc_registry::instance().unregister_alloc(idx_dev0);
    ggml_sycl::alloc_registry::instance().unregister_alloc(kv_dev1);

    TEST_ASSERT(ok, "SET_ROWS plan should resolve tracked remote handles even when tensor->data is stale");
    TEST_ASSERT(plan.src0_ptr == src_dev0, "src0 staging source should be the tracked device-0 handle");
    TEST_ASSERT(plan.index_ptr == idx_dev0, "index staging source should be the tracked device-0 handle");
    TEST_ASSERT(plan.src0_needs_staging && plan.index_needs_staging, "remote handles must be staged to owner");
    TEST_ASSERT(plan.src0_device == 0 && plan.index_device == 0, "remote handle source devices should be preserved");

    return true;
}

static bool test_dst_root_handle_wins_over_stale_tensor_data() {
    printf("\n=== Test: dst root smart handle wins over stale tensor data ===\n");

    alignas(64) std::uint8_t src_dev1[4096]      = {};
    alignas(64) std::uint8_t idx_dev1[4096]      = {};
    alignas(64) std::uint8_t stale_root_dev0[4096] = {};
    alignas(64) std::uint8_t kv_dev1[4096]       = {};

    ggml_sycl::alloc_registry::instance().register_alloc(stale_root_dev0, sizeof(stale_root_dev0), 0,
                                                         ggml_sycl::alloc_type::DEVICE);
    ggml_sycl::alloc_registry::instance().register_alloc(kv_dev1, sizeof(kv_dev1), 1,
                                                         ggml_sycl::alloc_type::DEVICE);

    ggml_tensor src0;
    ggml_tensor idx;
    ggml_tensor root;
    ggml_tensor view;
    init_tensor(src0, "k_cur_l1", GGML_TYPE_F32);
    init_tensor(idx, "kv_pos_l1", GGML_TYPE_I32);
    init_tensor(root, "cache_k_l1", GGML_TYPE_F16);
    init_tensor(view, "cache_k_l1_view", GGML_TYPE_F16);

    ggml_tensor_extra_gpu src_extra{};
    ggml_tensor_extra_gpu idx_extra{};
    ggml_tensor_extra_gpu root_extra{};
    src0.extra = &src_extra;
    idx.extra  = &idx_extra;
    root.extra = &root_extra;
    src_extra.set_data_device(1, src_dev1, GGML_LAYOUT_AOS, true);
    idx_extra.set_data_device(1, idx_dev1, GGML_LAYOUT_AOS, true);
    root_extra.set_data_device(1, kv_dev1, GGML_LAYOUT_AOS, true);

    root.data     = stale_root_dev0;
    view.view_src = &root;
    view.view_offs = 128;
    view.data     = stale_root_dev0 + view.view_offs;
    view.src[0]   = &src0;
    view.src[1]   = &idx;

    ggml_sycl_set_rows_plan plan{};
    const bool              ok = ggml_sycl_plan_set_rows(&view, 0, &plan);

    ggml_sycl::alloc_registry::instance().unregister_alloc(stale_root_dev0);
    ggml_sycl::alloc_registry::instance().unregister_alloc(kv_dev1);

    TEST_ASSERT(ok, "SET_ROWS plan should resolve with stale dst root tensor->data");
    TEST_ASSERT(plan.owner_device == 1, "dst root smart handle must choose the KV owner over stale raw data");
    TEST_ASSERT(plan.dst_ptr == kv_dev1 + view.view_offs, "dst view must resolve through the smart-handle root");

    return true;
}

static bool test_untracked_device_usm_fails_closed() {
    printf("\n=== Test: untracked device USM fails closed ===\n");

    if (ggml_sycl_info().device_count <= 0) {
        printf("  SKIP: no SYCL devices visible\n");
        return true;
    }

    sycl::queue & q             = ggml_sycl_get_device(0).default_queue();
    void *        untracked_dev = sycl::malloc_device(4096, q);
    if (!untracked_dev) {
        printf("  SKIP: sycl::malloc_device returned null\n");
        return true;
    }

    alignas(64) std::uint8_t idx_host[4096] = {};
    alignas(64) std::uint8_t kv_dev0[4096]  = {};
    ggml_sycl::alloc_registry::instance().register_alloc(kv_dev0, sizeof(kv_dev0), 0, ggml_sycl::alloc_type::DEVICE);

    ggml_tensor src0;
    ggml_tensor idx;
    ggml_tensor dst;
    init_tensor(src0, "k_cur_untracked", GGML_TYPE_F32);
    init_tensor(idx, "kv_pos_l1", GGML_TYPE_I32);
    init_tensor(dst, "cache_k_l1", GGML_TYPE_F16);

    ggml_tensor_extra_gpu dst_extra{};
    dst.extra  = &dst_extra;
    src0.data  = untracked_dev;
    idx.data   = idx_host;
    dst.data   = kv_dev0;
    dst.src[0] = &src0;
    dst.src[1] = &idx;
    dst_extra.set_data_device(0, kv_dev0, GGML_LAYOUT_AOS, true);

    ggml_sycl_set_rows_plan plan{};
    const bool              ok = ggml_sycl_plan_set_rows(&dst, 0, &plan);

    ggml_sycl::alloc_registry::instance().unregister_alloc(kv_dev0);
    sycl::free(untracked_dev, q);

    TEST_ASSERT(!ok, "untracked device USM must fail closed instead of being treated as host-stageable");

    return true;
}

int main() {
    bool ok = true;
    ok &= test_set_rows_uses_dst_root_owner_for_kv_view();
    ok &= test_raw_control_indices_require_owner_staging();
    ok &= test_wrong_device_inputs_record_source_owner_for_staging();
    ok &= test_wrong_device_handles_win_over_stale_tensor_data();
    ok &= test_dst_root_handle_wins_over_stale_tensor_data();
    ok &= test_untracked_device_usm_fails_closed();

    printf("\nSYCL SET_ROWS owner routing tests: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

#endif
