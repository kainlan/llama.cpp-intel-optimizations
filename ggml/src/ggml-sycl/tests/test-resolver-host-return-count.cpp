#include "../common.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// The build is -DNDEBUG (Release), so assert() would compile away and the
// test would pass vacuously. Use an explicit check that always runs.
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

using ggml_sycl::offload_phase;

// The resolver counter only counts a host return of a staged source on a
// thread that is dispatching a device graph, and not inside a host executor.
// It touches no device: the tensors are synthetic, an F32 tensor never reaches
// the multi-GPU control-storage query, and the registry lookup is a map read.
static void init_tensor(ggml_tensor & tensor, const char * name, void * data) {
    std::memset(&tensor, 0, sizeof(tensor));
    tensor.type  = GGML_TYPE_F32;
    tensor.ne[0] = 16;
    tensor.ne[1] = 1;
    tensor.ne[2] = 1;
    tensor.ne[3] = 1;
    tensor.nb[0] = ggml_type_size(tensor.type);
    tensor.nb[1] = tensor.nb[0] * tensor.ne[0];
    tensor.nb[2] = tensor.nb[1] * tensor.ne[1];
    tensor.nb[3] = tensor.nb[2] * tensor.ne[2];
    tensor.data  = data;
    std::snprintf(tensor.name, sizeof(tensor.name), "%s", name);
}

static uint64_t count_of(offload_phase phase) {
    return ggml_sycl_resolver_host_returns_to_device(phase);
}

int main() {
    float host_bytes[16] = {};

    ggml_tensor staged;  // an unregistered host buffer: a prestage would copy it
    init_tensor(staged, "staged_src", host_bytes);
    CHECK(ggml_sycl_prestage_needs_device_copy(&staged, nullptr, 0) == sizeof(host_bytes),
          "setup: the predicate must count the host buffer");

    ggml_tensor input = staged;  // INPUT tensors have their own staging
    input.flags |= GGML_TENSOR_FLAG_INPUT;
    CHECK(ggml_sycl_prestage_needs_device_copy(&input, nullptr, 0) == 0, "setup: the predicate must not count INPUT");

    ggml_tensor view;  // no storage of its own: counted through the root it views
    init_tensor(view, "staged_view", nullptr);
    view.view_src = &staged;

    ggml_sycl::offload_stats_set_phase(offload_phase::TG);
    CHECK(g_ggml_sycl_device_dispatch_depth == 0, "setup: the thread starts outside any region");

    // 1. Outside any region a resolver return is not a device kernel's read.
    uint64_t tg = count_of(offload_phase::TG);
    ggml_sycl_resolver_count_host_return(&staged, 0);
    CHECK(count_of(offload_phase::TG) == tg, "case 1: a return outside the device-dispatch region must not count");

    // 2. Inside the device-dispatch region it counts, in the current phase's bucket.
    {
        ggml_sycl_device_dispatch_region dispatch;
        const uint64_t                   pp    = count_of(offload_phase::PP);
        const uint64_t                   other = count_of(offload_phase::UNKNOWN);
        ggml_sycl_resolver_count_host_return(&staged, 0);
        CHECK(count_of(offload_phase::TG) == tg + 1, "case 2: a return inside the region must count in TG");
        CHECK(count_of(offload_phase::PP) == pp && count_of(offload_phase::UNKNOWN) == other,
              "case 2: only the current phase's bucket may move");
        tg++;

        // 3. A host executor closes the region for its scope.
        {
            ggml_sycl_host_executor_region executor;
            ggml_sycl_resolver_count_host_return(&staged, 0);
            CHECK(count_of(offload_phase::TG) == tg, "case 3: a return inside a host executor must not count");
        }

        // 4. The region is back once the executor's scope ends.
        ggml_sycl_resolver_count_host_return(&staged, 0);
        CHECK(count_of(offload_phase::TG) == tg + 1, "case 4: the region must reopen after the executor's scope");
        tg++;

        // 5. Phase buckets: PP, and every other phase in one bucket.
        ggml_sycl::offload_stats_set_phase(offload_phase::PP);
        ggml_sycl_resolver_count_host_return(&staged, 0);
        CHECK(count_of(offload_phase::PP) == pp + 1, "case 5: a PP return must count in PP");
        ggml_sycl::offload_stats_set_phase(offload_phase::WARMUP);
        ggml_sycl_resolver_count_host_return(&staged, 0);
        CHECK(count_of(offload_phase::UNKNOWN) == other + 1 && count_of(offload_phase::LOAD) == other + 1,
              "case 5: a WARMUP return must count in the shared other bucket");
        ggml_sycl::offload_stats_set_phase(offload_phase::TG);

        // 6. Only staged sources count: INPUT does not, and a view counts through its root.
        ggml_sycl_resolver_count_host_return(&input, 0);
        CHECK(count_of(offload_phase::TG) == tg, "case 6: an INPUT source must not count");
        ggml_sycl_resolver_count_host_return(&view, 0);
        CHECK(count_of(offload_phase::TG) == tg + 1, "case 6: a view of a staged root must count");
        tg++;
    }

    // 7. The depth is restored: nothing counts after the regions close.
    CHECK(g_ggml_sycl_device_dispatch_depth == 0, "case 7: the depth must return to 0");
    ggml_sycl_resolver_count_host_return(&staged, 0);
    CHECK(count_of(offload_phase::TG) == tg, "case 7: a return after the regions close must not count");

    std::printf("PASS\n");
    return 0;
}
