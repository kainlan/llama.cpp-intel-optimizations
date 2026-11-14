// The converter that hands a STORAGE row of the lazy-range record to the SYCL backend as a lazy table.
// Host-only: no backend, device or model.

#include "llama-impl.h"
#include "llama-lazy-storage.h"
#include "llama-model.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>

static int g_fails = 0;
#define CHECK(c, ...)                          \
    do {                                       \
        if (!(c)) {                            \
            std::printf("FAIL: " __VA_ARGS__); \
            std::printf("\n");                 \
            ++g_fails;                         \
        }                                      \
    } while (0)

// False when this build has no SYCL backend and nothing was checked.
static bool check_converter() {
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
    return true;
#else
    return false;
#endif
}

static int         g_closed = 0;
static int         g_handle = 0;
static std::string g_log;

static void * fake_open(llama_sycl_model_token, const llama_lazy_storage_table &, const char *) {
    return &g_handle;
}
static int fake_gather(void *, const int32_t * rows, size_t n, void * dst, size_t) {
    for (size_t i = 0; i < n; ++i) {
        if (rows[i] == 99) {
            return EIO;
        }
        std::memset(static_cast<uint8_t *>(dst) + i * 4, rows[i], 4);
    }
    return 0;
}
static void fake_counters(const void *, uint64_t out[4]) {
    out[0] = 1;
    out[1] = 2;
    out[2] = 0;
    out[3] = 2;
}
static void fake_close(void * h) {
    g_closed += h == &g_handle;
}

// The handle on fake procs: it refuses a gather before the open, forwards after it, prints its counters once and
// closes the backend handle exactly once.
static void check_handle() {
    llama_log_set([](ggml_log_level, const char * t, void *) { g_log += t; }, nullptr);
    const llama_lazy_storage_procs procs = { fake_open, fake_gather, fake_counters, fake_close };
    llama_lazy_storage_table       table = {};
    table.name                           = "per_layer_token_embd.weight";
    table.type                           = GGML_TYPE_I8;
    table.ne[0]                          = 4;
    table.ne[1]                          = 100;
    table.row_size                       = 4;
    {
        llama_lazy_storage s(table, "/models/x.gguf", procs);
        uint8_t            dst[8];
        const int32_t      rows[2] = { 1, 2 };
        CHECK(s.gather(rows, 2, dst, sizeof(dst)) == ENODEV, "a gather before the open did not return ENODEV");
        llama_sycl_model_token token;
        token.model_id        = 1;
        token.load_txn_id     = 1;
        token.slot            = 0;
        token.slot_generation = 1;
        CHECK(s.open(token), "the open failed");
        CHECK(s.gather(rows, 2, dst, sizeof(dst)) == 0 && dst[4] == 2, "the gather was not forwarded");
        const int32_t bad[1] = { 99 };
        CHECK(s.gather(bad, 1, dst, 4) == EIO, "a failed gather did not return its errno");
    }
    llama_log_set(nullptr, nullptr);
    CHECK(g_closed == 1, "closed %d times", g_closed);
    CHECK(g_log.find("[LAZY-STORAGE] per_layer_token_embd.weight hits=1 misses=2 evictions=0 submitted=2") !=
              std::string::npos,
          "no counters line: %s", g_log.c_str());
}

int main() {
    if (!check_converter()) {
        std::printf("SKIP: no SYCL in this build, so nothing was checked\n");
        return 77;
    }
    check_handle();
    return g_fails == 0 ? 0 : 1;
}
