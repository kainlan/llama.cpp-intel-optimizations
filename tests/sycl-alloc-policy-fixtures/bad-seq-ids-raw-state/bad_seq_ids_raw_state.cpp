#include <cstdint>
#include <cstddef>

struct bad_seq_ids_cache {
    const int32_t * q_seq_ids  = nullptr;
    const int32_t * kv_seq_ids = nullptr;
    size_t          q_count    = 0;
    size_t          kv_count   = 0;
};

static thread_local bad_seq_ids_cache g_sycl_seq_ids_cache;

void bad_set_seq_ids_host(const int32_t * q_seq_ids, size_t q_count, const int32_t * kv_seq_ids, size_t kv_count) {
    g_sycl_seq_ids_cache.q_seq_ids  = q_seq_ids;
    g_sycl_seq_ids_cache.q_count    = q_count;
    g_sycl_seq_ids_cache.kv_seq_ids = kv_seq_ids;
    g_sycl_seq_ids_cache.kv_count   = kv_count;
}
