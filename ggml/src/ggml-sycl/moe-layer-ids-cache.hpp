#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

// Graph-local, worker-affine MoE routing state. The map and vector allocations
// deliberately survive graph boundaries; only values that are invalid for the
// next graph are reset. Keeping the reset synchronous and caller-thread-local
// is required because graph dispatch may return to the same persistent worker.
struct moe_layer_ids_cache_entry {
    std::vector<int32_t> ids_host;

    struct cached_partition {
        std::vector<size_t> cpu_indices;

        struct sec_info {
            size_t ci;
            int    device_id;
        };

        std::vector<sec_info> sec_indices;
        std::vector<size_t>   gpu0_cached_indices;
        bool                  valid = false;
    } partition;
};

using moe_layer_ids_cache = std::unordered_map<int, moe_layer_ids_cache_entry>;

static inline void ggml_sycl_moe_layer_ids_cache_reset_entry(moe_layer_ids_cache_entry & entry) {
    entry.ids_host.clear();
    entry.partition.cpu_indices.clear();
    entry.partition.sec_indices.clear();
    entry.partition.gpu0_cached_indices.clear();
    entry.partition.valid = false;
}

// Shared graph-boundary operation used by production and the host-only worker
// fixture. Do not replace this with map::clear(): bounded reuse relies on
// retaining map nodes, buckets, and each vector's allocation.
static inline void ggml_sycl_moe_layer_ids_cache_new_graph(moe_layer_ids_cache & cache) {
    for (auto & [_, entry] : cache) {
        ggml_sycl_moe_layer_ids_cache_reset_entry(entry);
    }
}
