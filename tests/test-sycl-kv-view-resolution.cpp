// Regression tests for SYCL KV/view pointer resolution.
//
// These tests use synthetic ggml_tensor objects and direct smart handles so they
// do not need real allocations on multiple GPUs. The contract under test is the
// resolver policy: views must resolve from the root handle for the requested
// physical device, and resolving one device must not poison another.

#include "ggml-sycl/common.hpp"
#include "ggml-sycl/ggml-sycl-test.hpp"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

void ggml_sycl_data_ptr_cache_new_graph();
int  ggml_sycl_test_select_f16_attention_device(const ggml_tensor * src0,
                                                const ggml_tensor * src1,
                                                const ggml_tensor * dst,
                                                int                 current_device);
bool ggml_sycl_test_f16_staged_view_resolves_through_root(ggml_tensor * view, int target_device, void * staged_base);
bool ggml_sycl_test_publish_f16_attention_dst(ggml_tensor * dst, int target_device, void * staged_base, size_t bytes);

struct ggml_sycl_test_consumer_device_plan {
    int  execution_device;
    int  src_owner[2];
    bool src_needs_staging[2];
};

bool ggml_sycl_test_plan_simple_consumer_device(const ggml_tensor *                   dst,
                                                int                                   current_device,
                                                int                                   forced_device,
                                                ggml_sycl_test_consumer_device_plan * out);
bool ggml_sycl_test_plan_real_simple_consumer_dispatch(const ggml_tensor *                   dst,
                                                       int                                   current_device,
                                                       ggml_sycl_test_consumer_device_plan * out);
bool ggml_sycl_test_binbcast_needs_raw_host_staging(const ggml_tensor * tensor, const void * resolved_ptr, int device);

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

static void init_tensor(ggml_tensor & tensor, const char * name) {
    std::memset(&tensor, 0, sizeof(tensor));
    tensor.type  = GGML_TYPE_F16;
    tensor.ne[0] = 16;
    tensor.ne[1] = 1;
    tensor.ne[2] = 1;
    tensor.ne[3] = 1;
    tensor.nb[0] = ggml_type_size(tensor.type);
    tensor.nb[1] = tensor.nb[0] * tensor.ne[0];
    tensor.nb[2] = tensor.nb[1] * tensor.ne[1];
    tensor.nb[3] = tensor.nb[2] * tensor.ne[2];
    std::snprintf(tensor.name, sizeof(tensor.name), "%s", name);
}

static ggml_sycl_device_info make_mock_sycl_info(bool direct_peer = false) {
    ggml_sycl_device_info info{};
    info.device_count    = 2;
    info.total_gpu_count = 2;
    for (int d = 0; d < info.total_gpu_count; ++d) {
        info.devices[d].cc                       = 1200;
        info.devices[d].total_vram               = 16ull * 1024ull * 1024ull * 1024ull;
        info.devices[d].free_vram_at_init        = info.devices[d].total_vram;
        info.devices[d].max_alloc_size           = info.devices[d].total_vram;
        info.devices[d].safe_max_alloc_size      = info.devices[d].total_vram;
        info.devices[d].supports_soa_reorder     = true;
        info.gpu_dpct_ids[d]                     = d;
        info.max_work_group_sizes[d]             = 256;
        info.default_tensor_split[d]             = d == 0 ? 0.5f : 1.0f;
        info.devices[d].xmx_caps.supported       = true;
        info.devices[d].xmx_caps.supports_int8   = true;
        info.devices[d].xmx_caps.compute_units   = d == 0 ? 160 : 96;
        info.devices[d].xmx_caps.global_mem_size = info.devices[d].total_vram;
    }
    for (int src = 0; src < info.total_gpu_count; ++src) {
        for (int dst = 0; dst < info.total_gpu_count; ++dst) {
            sycl_peer_link_info & link   = info.peer_links[src][dst];
            link.valid                   = true;
            link.src_device              = src;
            link.dst_device              = dst;
            link.same_device             = src == dst;
            link.same_backend            = true;
            link.same_platform           = true;
            link.same_sycl_context       = src == dst || direct_peer;
            link.level_zero              = true;
            link.l0_peer_query_supported = true;
            link.l0_can_access_peer      = src == dst || direct_peer;
            link.direct_copy_measured    = src != dst && direct_peer;
            std::snprintf(link.preferred_transfer, sizeof(link.preferred_transfer), "%s",
                          src == dst ? "same-device" : (direct_peer ? "direct" : "host-bounce"));
            std::snprintf(link.unsupported_reason, sizeof(link.unsupported_reason), "%s",
                          src == dst || direct_peer ? "" : "mock-no-p2p");
        }
    }
    return info;
}

static bool test_view_prefers_root_requested_device_over_stale_view_handle() {
    printf("\n=== Test: view resolves through root requested-device handle ===\n");

    alignas(64) std::uint8_t root_dev0[4096]       = {};
    alignas(64) std::uint8_t root_dev1[4096]       = {};
    alignas(64) std::uint8_t stale_view_host[4096] = {};

    ggml_tensor root;
    ggml_tensor view;
    init_tensor(root, "cache_k_l10_root");
    init_tensor(view, "cache_k_l10_view");

    ggml_tensor_extra_gpu root_extra{};
    ggml_tensor_extra_gpu view_extra{};
    root.extra = &root_extra;
    view.extra = &view_extra;

    root_extra.set_data_device(0, root_dev0, GGML_LAYOUT_AOS, true);
    root_extra.set_data_device(1, root_dev1, GGML_LAYOUT_AOS, true);

    view.view_src  = &root;
    view.view_offs = 192;
    view.data      = stale_view_host + 777;
    view_extra.set_data_device(1, stale_view_host + 333, GGML_LAYOUT_AOS, false);

    void * resolved = ggml_sycl_get_data_ptr(&view, 1);
    TEST_ASSERT(resolved == root_dev1 + view.view_offs,
                "device-1 view should resolve as root device-1 pointer plus view_offs");

    auto view_handle = view_extra.data_handle[1].resolve(1);
    TEST_ASSERT(view_handle.ptr == root_dev1 + view.view_offs,
                "view device-1 handle should be refreshed to root-derived pointer");
    TEST_ASSERT(view_handle.on_device, "view device-1 handle should be marked device-resident");

    return true;
}

static bool test_permuted_kv_view_resolves_root_device_and_fails_closed_elsewhere() {
    printf("\n=== Test: permuted KV view resolves root device and fails closed elsewhere ===\n");

    alignas(64) std::uint8_t root_dev1[4096]       = {};
    alignas(64) std::uint8_t stale_view_host[4096] = {};

    ggml_tensor root;
    ggml_tensor permuted_view;
    init_tensor(root, "cache_v_l12_root");
    init_tensor(permuted_view, "cache_v_l12_permuted_view");

    ggml_tensor_extra_gpu root_extra{};
    ggml_tensor_extra_gpu view_extra{};
    root.extra          = &root_extra;
    permuted_view.extra = &view_extra;

    root_extra.set_data_device(1, root_dev1, GGML_LAYOUT_AOS, true);
    view_extra.set_data_device(0, stale_view_host + 640, GGML_LAYOUT_AOS, false);

    permuted_view.op           = GGML_OP_PERMUTE;
    permuted_view.src[0]       = &root;
    permuted_view.view_src     = &root;
    permuted_view.view_offs    = 640;
    permuted_view.data         = stale_view_host + 640;
    permuted_view.ne[0]        = root.ne[1];
    permuted_view.ne[1]        = root.ne[0];
    permuted_view.nb[0]        = root.nb[1];
    permuted_view.nb[1]        = root.nb[0];
    permuted_view.op_params[0] = 1;
    permuted_view.op_params[1] = 0;
    permuted_view.op_params[2] = 2;
    permuted_view.op_params[3] = 3;

    ggml_sycl_data_ptr_cache_new_graph();
    void * wrong_device = ggml_sycl_get_data_ptr(&permuted_view, 0);
    void * owner_device = ggml_sycl_get_data_ptr(&permuted_view, 1);

    auto refreshed_owner = view_extra.data_handle[1].resolve(1);
    auto refreshed_wrong = view_extra.data_handle[1].resolve(0);

    TEST_ASSERT(wrong_device == nullptr, "permuted KV view must fail closed on the wrong device");
    TEST_ASSERT(owner_device == root_dev1 + permuted_view.view_offs,
                "permuted KV view should resolve as root device-1 pointer plus view_offs");
    TEST_ASSERT(refreshed_owner.ptr == root_dev1 + permuted_view.view_offs && refreshed_owner.on_device,
                "permuted KV view handle should be refreshed as device-1 root-derived pointer");
    TEST_ASSERT(!refreshed_wrong, "refreshed device-1 view handle must reject device-0 resolution");

    return true;
}

static bool test_flattened_view_uses_root_before_stale_leaf_handle() {
    printf("\n=== Test: flattened view walks to root before stale leaf handle ===\n");

    alignas(64) std::uint8_t root_dev1[4096]  = {};
    alignas(64) std::uint8_t stale_leaf[4096] = {};

    ggml_tensor root;
    ggml_tensor leaf;
    init_tensor(root, "cache_k_l10_root");
    init_tensor(leaf, "cache_k_l10_flattened_view");

    ggml_tensor_extra_gpu root_extra{};
    ggml_tensor_extra_gpu leaf_extra{};
    root.extra = &root_extra;
    leaf.extra = &leaf_extra;

    root_extra.set_data_device(1, root_dev1, GGML_LAYOUT_AOS, true);
    leaf_extra.set_data_device(1, stale_leaf + 256, GGML_LAYOUT_AOS, false);

    leaf.view_src  = &root;
    leaf.view_offs = 320;
    leaf.data      = stale_leaf + 256;

    void * resolved = ggml_sycl_get_data_ptr(&leaf, 1);
    TEST_ASSERT(resolved == root_dev1 + leaf.view_offs,
                "flattened view should use root device-1 pointer plus absolute view_offs");

    auto leaf_handle = leaf_extra.data_handle[1].resolve(1);
    TEST_ASSERT(leaf_handle.ptr == root_dev1 + leaf.view_offs,
                "flattened view handle should be refreshed to root-derived pointer");
    TEST_ASSERT(leaf_handle.on_device, "flattened view handle should be marked device-resident");

    return true;
}

static bool test_slow_cache_is_per_tensor_and_device() {
    printf("\n=== Test: slow-path pointer cache is keyed by tensor and device ===\n");

    alignas(64) std::uint8_t root_dev0[4096] = {};
    alignas(64) std::uint8_t root_dev1[4096] = {};

    ggml_tensor root;
    ggml_tensor view;
    init_tensor(root, "cache_v_l10_root");
    init_tensor(view, "cache_v_l10_view");

    ggml_tensor_extra_gpu root_extra{};
    root.extra = &root_extra;

    root_extra.set_data_device(0, root_dev0, GGML_LAYOUT_AOS, true);
    root_extra.set_data_device(1, root_dev1, GGML_LAYOUT_AOS, true);

    view.view_src  = &root;
    view.view_offs = 512;

    ggml_sycl_data_ptr_cache_new_graph();
    void * dev0_first  = ggml_sycl_get_data_ptr_slow(&view, 0);
    void * dev1_second = ggml_sycl_get_data_ptr_slow(&view, 1);
    void * dev0_again  = ggml_sycl_get_data_ptr_slow(&view, 0);

    TEST_ASSERT(dev0_first == root_dev0 + view.view_offs, "device-0 slow resolve mismatch");
    TEST_ASSERT(dev1_second == root_dev1 + view.view_offs,
                "device-1 slow resolve was poisoned by earlier device-0 resolve");
    TEST_ASSERT(dev0_again == dev0_first, "device-0 cached slow resolve changed unexpectedly");

    return true;
}

static bool test_slow_view_prefers_root_over_stale_view_handle() {
    printf("\n=== Test: slow view resolve ignores stale per-view handle ===\n");

    alignas(64) std::uint8_t root_dev1[4096]       = {};
    alignas(64) std::uint8_t stale_view_host[4096] = {};

    ggml_tensor root;
    ggml_tensor view;
    init_tensor(root, "cache_k_l11_root");
    init_tensor(view, "cache_k_l11_view");

    ggml_tensor_extra_gpu root_extra{};
    ggml_tensor_extra_gpu view_extra{};
    root.extra = &root_extra;
    view.extra = &view_extra;

    root_extra.set_data_device(1, root_dev1, GGML_LAYOUT_AOS, true);
    view_extra.set_data_device(1, stale_view_host + 256, GGML_LAYOUT_AOS, false);

    view.view_src  = &root;
    view.view_offs = 768;
    view.data      = stale_view_host + 256;

    ggml_sycl_data_ptr_cache_new_graph();
    void * resolved = ggml_sycl_get_data_ptr_slow(&view, 1);
    TEST_ASSERT(resolved == root_dev1 + view.view_offs,
                "slow view resolve should use root device-1 pointer plus view_offs");

    auto view_handle = view_extra.data_handle[1].resolve(1);
    TEST_ASSERT(view_handle.ptr == root_dev1 + view.view_offs,
                "slow view resolve should refresh stale view handle to root-derived pointer");
    TEST_ASSERT(view_handle.on_device, "slow view refreshed handle should be device-resident");

    return true;
}

static bool test_f16_attention_routing_prefers_kv_view_owner() {
    printf("\n=== Test: F16 attention routing prefers cache_k view owner ===\n");

    alignas(64) std::uint8_t q_dev0[4096]       = {};
    alignas(64) std::uint8_t dst_dev0[4096]     = {};
    alignas(64) std::uint8_t cache_k_dev1[4096] = {};

    ggml_tensor q;
    ggml_tensor cache_k_root;
    ggml_tensor cache_k_view;
    ggml_tensor dst;
    init_tensor(q, "Qcur-10");
    init_tensor(cache_k_root, "cache_k_l10");
    init_tensor(cache_k_view, "cache_k_l10_view");
    init_tensor(dst, "KQ-10");

    ggml_tensor_extra_gpu q_extra{};
    ggml_tensor_extra_gpu cache_k_root_extra{};
    ggml_tensor_extra_gpu cache_k_view_extra{};
    ggml_tensor_extra_gpu dst_extra{};
    q.extra            = &q_extra;
    cache_k_root.extra = &cache_k_root_extra;
    cache_k_view.extra = &cache_k_view_extra;
    dst.extra          = &dst_extra;

    q_extra.set_data_device(0, q_dev0, GGML_LAYOUT_AOS, true);
    dst_extra.set_data_device(0, dst_dev0, GGML_LAYOUT_AOS, true);
    cache_k_root_extra.set_data_device(1, cache_k_dev1, GGML_LAYOUT_AOS, true);

    cache_k_view.view_src  = &cache_k_root;
    cache_k_view.view_offs = 384;

    const int routed = ggml_sycl_test_select_f16_attention_device(&q, &cache_k_view, &dst, 0);
    TEST_ASSERT(routed == 1, "cache_k view owned by device 1 should route F16 attention to device 1");

    return true;
}

static bool test_f16_staged_view_override_resolves_through_root() {
    printf("\n=== Test: F16 staged cache_k view override resolves through root ===\n");

    alignas(64) std::uint8_t cache_k_dev1[4096] = {};
    alignas(64) std::uint8_t staged_dev1[4096]  = {};

    ggml_tensor cache_k_root;
    ggml_tensor cache_k_view;
    init_tensor(cache_k_root, "cache_k_l10");
    init_tensor(cache_k_view, "cache_k_l10_view");

    ggml_tensor_extra_gpu cache_k_root_extra{};
    ggml_tensor_extra_gpu cache_k_view_extra{};
    cache_k_root.extra = &cache_k_root_extra;
    cache_k_view.extra = &cache_k_view_extra;
    cache_k_root_extra.set_data_device(1, cache_k_dev1, GGML_LAYOUT_AOS, true);

    cache_k_view.view_src  = &cache_k_root;
    cache_k_view.view_offs = 384;

    TEST_ASSERT(ggml_sycl_test_f16_staged_view_resolves_through_root(&cache_k_view, 1, staged_dev1),
                "staged F16 cache_k view must resolve from staged root plus view_offs");

    void * restored = ggml_sycl_resolve_tensor_ptr(&cache_k_view, 1);
    TEST_ASSERT(restored == cache_k_dev1 + cache_k_view.view_offs,
                "staged F16 view override should restore the original root handle");

    return true;
}

static bool test_f16_remote_dst_publication_sets_producer_handle() {
    printf("\n=== Test: F16 remote dst publication records producing device ===\n");

    alignas(64) std::uint8_t stale_dst_dev0[4096] = {};
    alignas(64) std::uint8_t produced_dev1[4096]  = {};

    ggml_sycl::alloc_registry::instance().register_alloc(stale_dst_dev0, sizeof(stale_dst_dev0), 0,
                                                         ggml_sycl::alloc_type::DEVICE);
    ggml_sycl::alloc_registry::instance().register_alloc(produced_dev1, sizeof(produced_dev1), 1,
                                                         ggml_sycl::alloc_type::DEVICE);

    ggml_tensor dst;
    init_tensor(dst, "KQ-remote");
    ggml_tensor_extra_gpu dst_extra{};
    dst.extra                = &dst_extra;
    dst.data                 = stale_dst_dev0;
    dst_extra.data_device[0] = stale_dst_dev0;
    dst_extra.data_handle[0] = ggml_sycl::mem_handle::from_direct(stale_dst_dev0, GGML_LAYOUT_AOS,
                                                                  /*on_device=*/true, /*device=*/0);

    const bool published = ggml_sycl_test_publish_f16_attention_dst(&dst, 1, produced_dev1, sizeof(produced_dev1));
    void *     dev1      = ggml_sycl_resolve_tensor_ptr(&dst, 1);
    void *     dev0      = ggml_sycl_resolve_tensor_ptr(&dst, 0);
    auto       handle1   = dst_extra.data_handle[1].resolve(1);
    auto       wrong     = dst_extra.data_handle[1].resolve(0);

    ggml_sycl::alloc_registry::instance().unregister_alloc(stale_dst_dev0);
    ggml_sycl::alloc_registry::instance().unregister_alloc(produced_dev1);

    TEST_ASSERT(published, "F16 dst publication helper should accept a staged device output");
    TEST_ASSERT(dev1 == produced_dev1, "dst should resolve on producing device 1 after publication");
    TEST_ASSERT(handle1.ptr == produced_dev1 && handle1.on_device,
                "dst smart handle should identify the producing device allocation");
    TEST_ASSERT(!wrong, "published device-1 handle must fail closed when resolved from device 0");
    TEST_ASSERT(dev0 == nullptr, "publication must clear stale ctx-device dst output state");

    return true;
}

static bool test_published_remote_dst_is_authoritative_for_immediate_consumer() {
    printf("\n=== Test: published remote dst is authoritative for immediate consumer ===\n");

    alignas(64) std::uint8_t stale_dst_dev0[4096] = {};
    alignas(64) std::uint8_t produced_dev1[4096]  = {};
    alignas(64) std::uint8_t rhs_dev1[4096]       = {};

    ggml_sycl::alloc_registry::instance().register_alloc(stale_dst_dev0, sizeof(stale_dst_dev0), 0,
                                                         ggml_sycl::alloc_type::DEVICE);
    ggml_sycl::alloc_registry::instance().register_alloc(produced_dev1, sizeof(produced_dev1), 1,
                                                         ggml_sycl::alloc_type::DEVICE);
    ggml_sycl::alloc_registry::instance().register_alloc(rhs_dev1, sizeof(rhs_dev1), 1, ggml_sycl::alloc_type::DEVICE);

    ggml_tensor dst;
    ggml_tensor rhs;
    ggml_tensor add;
    init_tensor(dst, "KQ-remote");
    init_tensor(rhs, "bias-remote");
    init_tensor(add, "add-after-remote");
    add.op     = GGML_OP_ADD;
    add.src[0] = &dst;
    add.src[1] = &rhs;

    ggml_tensor_extra_gpu dst_extra{};
    ggml_tensor_extra_gpu rhs_extra{};
    dst.extra = &dst_extra;
    rhs.extra = &rhs_extra;

    dst.data                 = stale_dst_dev0;
    dst_extra.data_device[0] = stale_dst_dev0;
    dst_extra.data_handle[0] = ggml_sycl::mem_handle::from_direct(stale_dst_dev0, GGML_LAYOUT_AOS,
                                                                  /*on_device=*/true, /*device=*/0);
    rhs_extra.set_data_device(1, rhs_dev1, GGML_LAYOUT_AOS, true);

    const bool published = ggml_sycl_test_publish_f16_attention_dst(&dst, 1, produced_dev1, sizeof(produced_dev1));
    void *     dev0      = ggml_sycl_resolve_tensor_ptr(&dst, 0);
    void *     dev1      = ggml_sycl_resolve_tensor_ptr(&dst, 1);

    ggml_sycl_test_consumer_device_plan plan{};
    const bool                          planned = ggml_sycl_test_plan_simple_consumer_device(&add, 0, -1, &plan);

    ggml_sycl::alloc_registry::instance().unregister_alloc(stale_dst_dev0);
    ggml_sycl::alloc_registry::instance().unregister_alloc(produced_dev1);
    ggml_sycl::alloc_registry::instance().unregister_alloc(rhs_dev1);

    TEST_ASSERT(published, "remote dst publication should succeed");
    TEST_ASSERT(dev0 == nullptr, "immediate ctx-device resolve must not see stale dst output");
    TEST_ASSERT(dev1 == produced_dev1, "immediate producer-device resolve should see published output");
    TEST_ASSERT(planned, "simple consumer dispatch plan should resolve");
    TEST_ASSERT(plan.execution_device == 0, "host-bounce ADD dispatch should stay on the current device");
    TEST_ASSERT(plan.src_owner[0] == 1 && plan.src_owner[1] == 1, "ADD dispatch should preserve producer ownership");
    TEST_ASSERT(plan.src_needs_staging[0] && plan.src_needs_staging[1],
                "host-bounce ADD dispatch should stage remote producer inputs");

    return true;
}

static bool test_remote_view_dst_publication_clears_stale_view_slot() {
    printf("\n=== Test: remote view dst publication clears stale view slot ===\n");

    alignas(64) std::uint8_t stale_root_dev0[4096] = {};
    alignas(64) std::uint8_t stale_view_dev0[4096] = {};
    alignas(64) std::uint8_t produced_dev1[4096]   = {};

    ggml_sycl::alloc_registry::instance().register_alloc(stale_root_dev0, sizeof(stale_root_dev0), 0,
                                                         ggml_sycl::alloc_type::DEVICE);
    ggml_sycl::alloc_registry::instance().register_alloc(stale_view_dev0, sizeof(stale_view_dev0), 0,
                                                         ggml_sycl::alloc_type::DEVICE);
    ggml_sycl::alloc_registry::instance().register_alloc(produced_dev1, sizeof(produced_dev1), 1,
                                                         ggml_sycl::alloc_type::DEVICE);

    ggml_tensor root;
    ggml_tensor view;
    init_tensor(root, "KQ-root-remote");
    init_tensor(view, "KQ-view-remote");
    view.view_src  = &root;
    view.view_offs = 256;

    ggml_tensor_extra_gpu root_extra{};
    ggml_tensor_extra_gpu view_extra{};
    root.extra = &root_extra;
    view.extra = &view_extra;

    root.data                 = stale_root_dev0;
    view.data                 = stale_view_dev0 + 64;
    root_extra.data_device[0] = stale_root_dev0;
    root_extra.data_handle[0] = ggml_sycl::mem_handle::from_direct(stale_root_dev0, GGML_LAYOUT_AOS,
                                                                   /*on_device=*/true, /*device=*/0);
    view_extra.data_device[0] = stale_view_dev0 + 64;
    view_extra.data_handle[0] = ggml_sycl::mem_handle::from_direct(stale_view_dev0 + 64, GGML_LAYOUT_AOS,
                                                                   /*on_device=*/true, /*device=*/0);

    const bool published = ggml_sycl_test_publish_f16_attention_dst(&view, 1, produced_dev1, sizeof(produced_dev1));
    void *     dev0      = ggml_sycl_resolve_tensor_ptr(&view, 0);
    void *     dev1      = ggml_sycl_resolve_tensor_ptr(&view, 1);

    ggml_sycl::alloc_registry::instance().unregister_alloc(stale_root_dev0);
    ggml_sycl::alloc_registry::instance().unregister_alloc(stale_view_dev0);
    ggml_sycl::alloc_registry::instance().unregister_alloc(produced_dev1);

    TEST_ASSERT(published, "remote view dst publication should succeed");
    TEST_ASSERT(dev0 == nullptr, "remote view publication must clear stale ctx-device view state");
    TEST_ASSERT(dev1 == produced_dev1 + view.view_offs, "view should resolve to produced root plus view offset");
    TEST_ASSERT(view.data == produced_dev1 + view.view_offs,
                "view data pointer should track the published view output");

    return true;
}

static bool test_simple_consumer_prefers_remote_producer_owner_with_direct_peer() {
    printf("\n=== Test: simple consumer can execute on remote producer owner with direct peer ===\n");

    alignas(64) std::uint8_t produced_dev1[4096] = {};
    alignas(64) std::uint8_t rhs_dev1[4096]      = {};

    ggml_tensor lhs;
    ggml_tensor rhs;
    ggml_tensor add;
    init_tensor(lhs, "KQ-remote");
    init_tensor(rhs, "bias-remote");
    init_tensor(add, "add-remote");
    add.op     = GGML_OP_ADD;
    add.src[0] = &lhs;
    add.src[1] = &rhs;

    ggml_tensor_extra_gpu lhs_extra{};
    ggml_tensor_extra_gpu rhs_extra{};
    lhs.extra = &lhs_extra;
    rhs.extra = &rhs_extra;
    lhs_extra.set_data_device(1, produced_dev1, GGML_LAYOUT_AOS, true);
    rhs_extra.set_data_device(1, rhs_dev1, GGML_LAYOUT_AOS, true);

    ggml_sycl_test_consumer_device_plan plan{};
    TEST_ASSERT(ggml_sycl_test_plan_simple_consumer_device(&add, 0, -1, &plan), "ADD consumer plan should resolve");
    TEST_ASSERT(plan.execution_device == 1, "ADD should choose the producing remote device when inputs resolve there");
    TEST_ASSERT(plan.src_owner[0] == 1 && plan.src_owner[1] == 1, "ADD source owners should resolve to device 1");
    TEST_ASSERT(!plan.src_needs_staging[0] && !plan.src_needs_staging[1],
                "ADD inputs on the selected producing device should not be staged");

    return true;
}

static bool test_simple_consumer_ignores_nonexistent_device_slots() {
    printf("\n=== Test: simple consumer ignores stale non-routable device slots ===\n");

    alignas(64) std::uint8_t stale_high_device[4096] = {};
    alignas(64) std::uint8_t rhs_dev0[4096]          = {};

    const int stale_device = GGML_SYCL_MAX_DEVICES - 1;
    if (stale_device < 2) {
        printf("  SKIP: GGML_SYCL_MAX_DEVICES too small for stale high-device test\n");
        return true;
    }

    ggml_tensor lhs;
    ggml_tensor rhs;
    ggml_tensor add;
    init_tensor(lhs, "stale-high-device");
    init_tensor(rhs, "bias-local");
    init_tensor(add, "add-after-stale-high-device");
    add.op     = GGML_OP_ADD;
    add.src[0] = &lhs;
    add.src[1] = &rhs;

    ggml_tensor_extra_gpu lhs_extra{};
    ggml_tensor_extra_gpu rhs_extra{};
    lhs.extra = &lhs_extra;
    rhs.extra = &rhs_extra;

    lhs_extra.data_handle[stale_device] =
        ggml_sycl::mem_handle::from_direct(stale_high_device, GGML_LAYOUT_AOS, true, stale_device);
    lhs_extra.data_device[stale_device]      = stale_high_device;
    lhs_extra.data_device_size[stale_device] = sizeof(stale_high_device);
    rhs_extra.set_data_device(0, rhs_dev0, GGML_LAYOUT_AOS, true);

    ggml_sycl_test_consumer_device_plan plan{};
    TEST_ASSERT(ggml_sycl_test_plan_simple_consumer_device(&add, 0, -1, &plan),
                "ADD consumer plan should still resolve with a valid local rhs");
    TEST_ASSERT(plan.execution_device == 0, "ADD must not route to a stale non-routable device slot");
    TEST_ASSERT(plan.src_owner[0] == -1, "stale high-device source owner should be ignored");
    TEST_ASSERT(plan.src_owner[1] == 0, "valid local source owner should still resolve");

    return true;
}

static bool test_simple_consumer_prefers_current_activation_over_remote_weight() {
    printf("\n=== Test: simple consumer keeps mixed activation/weight op on current device ===\n");

    alignas(64) std::uint8_t activation_dev0[4096] = {};
    alignas(64) std::uint8_t weight_dev1[4096]     = {};

    ggml_tensor activation;
    ggml_tensor weight;
    ggml_tensor mul;
    init_tensor(activation, "norm-10");
    init_tensor(weight, "blk.10.post_attention_norm.weight");
    init_tensor(mul, "attn_post_norm-10");
    mul.op     = GGML_OP_MUL;
    mul.src[0] = &activation;
    mul.src[1] = &weight;

    ggml_tensor_extra_gpu activation_extra{};
    ggml_tensor_extra_gpu weight_extra{};
    activation.extra = &activation_extra;
    weight.extra     = &weight_extra;
    activation_extra.set_data_device(0, activation_dev0, GGML_LAYOUT_AOS, true);
    weight_extra.set_data_device(1, weight_dev1, GGML_LAYOUT_AOS, true);

    ggml_sycl_test_consumer_device_plan plan{};
    TEST_ASSERT(ggml_sycl_test_plan_simple_consumer_device(&mul, 0, -1, &plan), "MUL consumer plan should resolve");
    TEST_ASSERT(plan.execution_device == 0, "MUL should stay on current activation owner when owners are mixed");
    TEST_ASSERT(plan.src_owner[0] == 0 && plan.src_owner[1] == 1, "MUL source owners should be preserved");
    TEST_ASSERT(!plan.src_needs_staging[0], "current activation source should not be staged");
    TEST_ASSERT(plan.src_needs_staging[1], "remote broadcast weight should be staged to the activation device");

    return true;
}

static bool test_simple_consumer_stages_host_backed_broadcast_weight() {
    printf("\n=== Test: simple consumer stages host-backed broadcast weight ===\n");

    alignas(64) std::uint8_t activation_dev0[4096] = {};
    alignas(64) std::uint8_t weight_host[4096]     = {};

    ggml_tensor activation;
    ggml_tensor weight;
    ggml_tensor mul;
    init_tensor(activation, "norm-10");
    init_tensor(weight, "blk.10.post_attention_norm.weight");
    init_tensor(mul, "attn_post_norm-10");
    mul.op     = GGML_OP_MUL;
    mul.src[0] = &activation;
    mul.src[1] = &weight;

    ggml_tensor_extra_gpu activation_extra{};
    activation.extra = &activation_extra;
    activation.data  = activation_dev0;
    weight.data      = weight_host;
    activation_extra.set_data_device(0, activation_dev0, GGML_LAYOUT_AOS, true);

    ggml_sycl_test_consumer_device_plan plan{};
    TEST_ASSERT(ggml_sycl_test_plan_simple_consumer_device(&mul, 0, -1, &plan), "MUL host-weight plan should resolve");
    TEST_ASSERT(plan.execution_device == 0, "host-backed broadcast weight should execute with current activation");
    TEST_ASSERT(plan.src_owner[0] == 0, "activation owner should resolve to device 0");
    TEST_ASSERT(plan.src_owner[1] == -1, "host-backed broadcast weight should not claim a device owner");
    TEST_ASSERT(!plan.src_needs_staging[0], "device activation should not be staged");
    TEST_ASSERT(plan.src_needs_staging[1], "host-backed broadcast weight should be staged to the execution device");

    return true;
}

static bool test_simple_consumer_marks_staging_when_forced_elsewhere() {
    printf("\n=== Test: simple consumer marks explicit staging when forced off producer ===\n");

    alignas(64) std::uint8_t produced_dev1[4096] = {};

    ggml_tensor src;
    ggml_tensor cont;
    init_tensor(src, "KQ-remote");
    init_tensor(cont, "cont-local");
    cont.op     = GGML_OP_CONT;
    cont.src[0] = &src;

    ggml_tensor_extra_gpu src_extra{};
    src.extra = &src_extra;
    src_extra.set_data_device(1, produced_dev1, GGML_LAYOUT_AOS, true);

    ggml_sycl_test_consumer_device_plan plan{};
    TEST_ASSERT(ggml_sycl_test_plan_simple_consumer_device(&cont, 0, 0, &plan), "CONT forced plan should resolve");
    TEST_ASSERT(plan.execution_device == 0, "forced CONT plan should honor the requested local execution device");
    TEST_ASSERT(plan.src_owner[0] == 1, "CONT source owner should remain the producing device");
    TEST_ASSERT(plan.src_needs_staging[0], "CONT source must be explicitly staged when executing away from producer");

    return true;
}

static bool test_requested_device_rejects_wrong_owned_root_handle() {
    printf("\n=== Test: requested-device resolver rejects wrong-owned root handle ===\n");

    alignas(64) std::uint8_t root_dev0[4096] = {};

    ggml_tensor root;
    ggml_tensor view;
    init_tensor(root, "cache_k_l10_root");
    init_tensor(view, "cache_k_l10_view");

    ggml_tensor_extra_gpu root_extra{};
    ggml_tensor_extra_gpu view_extra{};
    root.extra = &root_extra;
    view.extra = &view_extra;

    root_extra.data_device[1] = root_dev0;
    root_extra.data_handle[1] = ggml_sycl::mem_handle::from_direct(root_dev0, GGML_LAYOUT_AOS,
                                                                   /*on_device=*/true, /*device=*/0);

    view.view_src  = &root;
    view.view_offs = 384;

    ggml_sycl_data_ptr_cache_new_graph();
    void * resolved = ggml_sycl_get_data_ptr(&view, 1);
    TEST_ASSERT(resolved != root_dev0 + view.view_offs,
                "device-1 resolver must not accept a root handle owned by device 0");
    TEST_ASSERT(resolved == nullptr, "wrong-owned root handle with no valid fallback should fail closed");

    return true;
}

static bool test_requested_device_rejects_wrong_owned_direct_tensor() {
    printf("\n=== Test: requested-device resolver rejects wrong-owned direct tensor ===\n");

    alignas(64) std::uint8_t tensor_dev1[4096] = {};

    ggml_tensor tensor;
    init_tensor(tensor, "direct_dev1_tensor");

    ggml_tensor_extra_gpu extra{};
    tensor.extra         = &extra;
    extra.data_device[0] = tensor_dev1;
    extra.data_handle[0] = ggml_sycl::mem_handle::from_direct(tensor_dev1, GGML_LAYOUT_AOS,
                                                              /*on_device=*/true, /*device=*/1);

    ggml_sycl_data_ptr_cache_new_graph();
    auto resolved = ggml_sycl_resolve(&tensor, 0);
    TEST_ASSERT(!resolved, "ggml_sycl_resolve(device 0) must reject a direct handle owned by device 1");
    TEST_ASSERT(ggml_sycl_get_data_ptr(&tensor, 0) == nullptr,
                "ggml_sycl_get_data_ptr(device 0) must reject a wrong-owned direct handle");

    return true;
}

static bool test_raw_tensor_data_device_fallback_requires_owner_match() {
    printf("\n=== Test: raw tensor data device fallback requires owner match ===\n");

    alignas(64) std::uint8_t tensor_dev1[4096] = {};

    ggml_sycl::alloc_registry::instance().register_alloc(tensor_dev1, sizeof(tensor_dev1), 1,
                                                         ggml_sycl::alloc_type::DEVICE);

    ggml_tensor tensor;
    init_tensor(tensor, "raw_dev1_tensor");
    tensor.data = tensor_dev1;

    ggml_sycl_data_ptr_cache_new_graph();
    void * wrong_device = ggml_sycl_get_data_ptr(&tensor, 0);
    void * owner_device = ggml_sycl_get_data_ptr(&tensor, 1);

    ggml_sycl::alloc_registry::instance().unregister_alloc(tensor_dev1);

    TEST_ASSERT(wrong_device == nullptr, "raw device pointer must not resolve on a non-owner device");
    TEST_ASSERT(owner_device == tensor_dev1, "raw device pointer should still resolve on its registered owner");

    return true;
}

static bool test_resolve_or_host_rejects_registered_wrong_device_data() {
    printf("\n=== Test: resolve-or-host rejects registered wrong-device data ===\n");

    alignas(64) std::uint8_t tensor_dev1[4096] = {};

    ggml_sycl::alloc_registry::instance().register_alloc(tensor_dev1, sizeof(tensor_dev1), 1,
                                                         ggml_sycl::alloc_type::DEVICE);

    ggml_tensor tensor;
    init_tensor(tensor, "raw_dev1_resolve_or_host");
    tensor.data = tensor_dev1;

    ggml_sycl_data_ptr_cache_new_graph();
    void * wrong_device = ggml_sycl_resolve_or_host_tensor_ptr(&tensor, 0);
    void * owner_device = ggml_sycl_resolve_or_host_tensor_ptr(&tensor, 1);

    ggml_sycl::alloc_registry::instance().unregister_alloc(tensor_dev1);

    TEST_ASSERT(wrong_device == nullptr, "resolve-or-host must not reinterpret wrong-device data as host memory");
    TEST_ASSERT(owner_device == tensor_dev1, "resolve-or-host should resolve registered device data on its owner");

    return true;
}

static bool test_untracked_device_usm_fails_common_resolver() {
    printf("\n=== Test: common resolver rejects untracked device USM ===\n");

    const char * run_runtime = std::getenv("GGML_SYCL_TEST_KV_VIEW_RUNTIME");
    if (!run_runtime || std::atoi(run_runtime) == 0) {
        printf("  SKIP: runtime USM probe requires GGML_SYCL_TEST_KV_VIEW_RUNTIME=1\n");
        return true;
    }

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

    ggml_tensor tensor;
    init_tensor(tensor, "untracked_device_usm");
    tensor.data = untracked_dev;

    ggml_sycl_data_ptr_cache_new_graph();
    void * resolved = ggml_sycl_get_data_ptr(&tensor, 0);
    void * host     = ggml_sycl_resolve_or_host_tensor_ptr(&tensor, 0);

    sycl::free(untracked_dev, q);

    TEST_ASSERT(resolved == nullptr, "common resolver must reject untracked device USM");
    TEST_ASSERT(host == nullptr, "resolve-or-host must not return untracked device USM as host memory");

    return true;
}

static bool test_binbcast_stages_only_raw_host_storage() {
    printf("\n=== Test: binbcast stages raw host storage but not tracked USM ===\n");

    alignas(64) std::uint8_t raw_host[4096] = {};

    ggml_tensor tensor;
    init_tensor(tensor, "raw_host_bias");
    tensor.data = raw_host;

    TEST_ASSERT(ggml_sycl_test_binbcast_needs_raw_host_staging(&tensor, raw_host, 0),
                "unregistered raw host tensor storage must be staged before a GPU binbcast kernel");

    ggml_sycl::alloc_registry::instance().register_alloc(raw_host, sizeof(raw_host), -1,
                                                         ggml_sycl::alloc_type::HOST_PINNED);
    TEST_ASSERT(ggml_sycl_test_binbcast_needs_raw_host_staging(&tensor, raw_host, 0),
                "raw tensor storage should be staged even when the backing allocation is host-pinned");
    ggml_sycl::alloc_registry::instance().unregister_alloc(raw_host);

    ggml_sycl::alloc_registry::instance().register_alloc(raw_host, sizeof(raw_host), 0,
                                                         ggml_sycl::alloc_type::DEVICE);
    TEST_ASSERT(!ggml_sycl_test_binbcast_needs_raw_host_staging(&tensor, raw_host, 1),
                "registered device USM should not be copied as CPU-readable raw host memory");
    ggml_sycl::alloc_registry::instance().unregister_alloc(raw_host);

    return true;
}

int main() {
    const ggml_sycl_device_info              no_peer_mock_info     = make_mock_sycl_info(false);
    const ggml_sycl_device_info              direct_peer_mock_info = make_mock_sycl_info(true);
    ggml_sycl::test_sycl_info_override_guard info_guard(no_peer_mock_info);

    bool ok = true;
    ok &= test_view_prefers_root_requested_device_over_stale_view_handle();
    ok &= test_permuted_kv_view_resolves_root_device_and_fails_closed_elsewhere();
    ok &= test_flattened_view_uses_root_before_stale_leaf_handle();
    ok &= test_slow_cache_is_per_tensor_and_device();
    ok &= test_slow_view_prefers_root_over_stale_view_handle();
    ok &= test_f16_attention_routing_prefers_kv_view_owner();
    ok &= test_f16_staged_view_override_resolves_through_root();
    ok &= test_f16_remote_dst_publication_sets_producer_handle();
    ok &= test_published_remote_dst_is_authoritative_for_immediate_consumer();
    ok &= test_remote_view_dst_publication_clears_stale_view_slot();
    ggml_sycl::test_set_sycl_info_override(direct_peer_mock_info);
    ok &= test_simple_consumer_prefers_remote_producer_owner_with_direct_peer();
    ggml_sycl::test_set_sycl_info_override(no_peer_mock_info);
    ok &= test_simple_consumer_ignores_nonexistent_device_slots();
    ok &= test_simple_consumer_prefers_current_activation_over_remote_weight();
    ok &= test_simple_consumer_stages_host_backed_broadcast_weight();
    ok &= test_simple_consumer_marks_staging_when_forced_elsewhere();
    ok &= test_requested_device_rejects_wrong_owned_root_handle();
    ok &= test_requested_device_rejects_wrong_owned_direct_tensor();
    ok &= test_raw_tensor_data_device_fallback_requires_owner_match();
    ok &= test_resolve_or_host_rejects_registered_wrong_device_data();
    ok &= test_untracked_device_usm_fails_common_resolver();
    ok &= test_binbcast_stages_only_raw_host_storage();

    printf("\nSYCL KV view resolution tests: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

#endif
