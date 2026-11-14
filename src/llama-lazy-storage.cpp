#include "llama-lazy-storage.h"

#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
ggml_sycl_lazy_table_info llama_lazy_storage_table_info(const llama_lazy_storage_table & t) {
    ggml_sycl_lazy_table_info i = {};
    i.name                      = t.name.c_str();
    i.type                      = t.type;
    i.rows_per_token            = t.attrs.rows_per_token;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        i.ne[d] = t.ne[d];
    }
    i.row_size       = t.row_size;
    i.consumer_layer = t.attrs.consumer_layer;
    i.file_index     = t.value.file_idx;
    i.file_offset    = t.value.offs;
    i.size           = t.value.size;
    i.storage_only   = t.attrs.storage_only;
    return i;
}
#endif
