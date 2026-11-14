// The converter that hands a STORAGE row of the lazy-range record to the SYCL backend as a lazy table.
// Host-only: no backend, device or model.

#include "llama-lazy-storage.h"

#include <cstdio>
#include <cstring>

static int g_fails = 0;
#define CHECK(c, ...)                          \
    do {                                       \
        if (!(c)) {                            \
            std::printf("FAIL: " __VA_ARGS__); \
            std::printf("\n");                 \
            ++g_fails;                         \
        }                                      \
    } while (0)

static void check_converter() {
#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
    llama_lazy_storage_table t = {};
    t.name                     = "per_layer_token_embd.weight";
    t.type                     = GGML_TYPE_Q8_0;
    t.ne[0]                    = 160;
    t.ne[1]                    = 1000;
    t.ne[2] = t.ne[3]                 = 1;
    t.row_size                        = 170;
    t.value                           = { 2, 4032, 170000 };
    t.attrs                           = { 16, 1, true };
    const ggml_sycl_lazy_table_info i = llama_lazy_storage_table_info(t);
    CHECK(std::strcmp(i.name, t.name.c_str()) == 0 && i.name == t.name.c_str(), "name");
    CHECK(i.type == t.type && i.ne[0] == 160 && i.ne[1] == 1000 && i.row_size == 170, "shape");
    CHECK(i.file_index == 2 && i.file_offset == 4032 && i.size == 170000, "place");
    CHECK(i.rows_per_token == 16 && i.consumer_layer == 1 && i.storage_only, "attrs");
#else
    std::printf("SKIP converter: no SYCL in this build\n");
#endif
}

int main() {
    check_converter();
    return g_fails == 0 ? 0 : 1;
}
