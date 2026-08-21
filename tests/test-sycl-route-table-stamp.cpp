// Route-table stamp: a cached MoE route table may be consumed only when it is
// valid and BOTH generations match (llama.cpp perf-recovery epic, track B).
#include "ggml-sycl/moe-route-table.hpp"

#include <cstdio>
#define CHECK(cond, msg)                      \
    do {                                      \
        if (!(cond)) {                        \
            std::printf("FAILED: %s\n", msg); \
            return 1;                         \
        }                                     \
    } while (0)

int main() {
    ggml_sycl::moe_route_table_stamp s{};
    CHECK(!ggml_sycl_moe_route_table_current(s, 1, 1), "never-built table must not be current");
    s.valid                     = true;
    s.plan_generation           = 4;
    s.expert_storage_generation = 9;
    CHECK(ggml_sycl_moe_route_table_current(s, 4, 9), "matching stamps must be current");
    CHECK(!ggml_sycl_moe_route_table_current(s, 5, 9), "plan-generation bump must invalidate");
    CHECK(!ggml_sycl_moe_route_table_current(s, 4, 10), "storage-generation bump must invalidate");
    CHECK(!ggml_sycl_moe_route_table_current(s, 5, 10), "double bump must invalidate");
    s.valid = false;
    CHECK(!ggml_sycl_moe_route_table_current(s, 4, 9), "explicit invalidation must hold");
    std::printf("OK: route-table stamp semantics\n");
    return 0;
}
