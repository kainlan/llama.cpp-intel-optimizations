// The converter that hands a STORAGE row of the lazy-range record to the SYCL backend as a lazy table, and the handle
// libllama keeps on an opened table, checked on fake procs. Host-only: no backend, device or model; the handle checks
// run in a build without SYCL too.

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

static int         g_closed     = 0;
static int         g_opened     = 0;
static int         g_level      = -1;
static int         g_handle     = 0;
static bool        g_open_fails = false;
static std::string g_log;

static const char * k_name = "per_layer_token_embd.weight";
static const char * k_path = "/models/x.gguf";

// The fakes verify what the class forwards: a wrong token, name, path, handle or size is a refusal the checks see.
static void * fake_open(llama_sycl_model_token token, const llama_lazy_storage_table & table, const char * file_path) {
    ++g_opened;
    if (g_open_fails || token.model_id != 1 || token.load_txn_id != 1 || token.slot != 0 ||
        token.slot_generation != 1 || table.name != k_name || file_path == nullptr ||
        std::strcmp(file_path, k_path) != 0) {
        return nullptr;
    }
    return &g_handle;
}

static int fake_gather(void * handle, const int32_t * rows, size_t n, void * dst, size_t dst_bytes) {
    if (handle != &g_handle || dst_bytes < n * 4) {
        return EINVAL;
    }
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
    out[2] = 3;
    out[3] = 4;
}

static void fake_close(void * h) {
    g_closed += h == &g_handle;
}

static const char * k_counters_line =
    "[LAZY-STORAGE] per_layer_token_embd.weight hits=1 misses=2 evictions=3 submitted=4";

static size_t count_log(const char * needle) {
    size_t found = 0;
    for (size_t at = g_log.find(needle); at != std::string::npos; at = g_log.find(needle, at + 1)) {
        ++found;
    }
    return found;
}

static void log_capture() {
    g_log.clear();
    g_level      = -1;
    g_opened     = 0;
    g_closed     = 0;
    g_open_fails = false;
    llama_log_set(
        [](ggml_log_level level, const char * t, void *) {
            if (std::strstr(t, "[LAZY-STORAGE]") != nullptr) {
                g_level = level;
            }
            g_log += t;
        },
        nullptr);
}

static llama_lazy_storage_table make_table() {
    llama_lazy_storage_table table = {};
    table.name                     = k_name;
    table.type                     = GGML_TYPE_I8;
    table.ne[0]                    = 4;
    table.ne[1]                    = 100;
    table.row_size                 = 4;
    return table;
}

static llama_sycl_model_token make_token() {
    llama_sycl_model_token token;
    token.model_id        = 1;
    token.load_txn_id     = 1;
    token.slot            = 0;
    token.slot_generation = 1;
    return token;
}

// The handle on fake procs: it refuses a gather before the open, forwards after it, prints its counters once and
// closes the backend handle exactly once. Each case captures its own log.
static void check_handle() {
    const llama_lazy_storage_procs procs   = { fake_open, fake_gather, fake_counters, fake_close };
    const llama_lazy_storage_table table   = make_table();
    const int32_t                  rows[2] = { 1, 2 };
    uint8_t                        dst[8]  = {};

    log_capture();
    {
        llama_lazy_storage s(table, k_path, procs);
        CHECK(s.gather(rows, 2, dst, sizeof(dst)) == ENODEV, "a gather before the open did not return ENODEV");
        const llama_sycl_model_token token = make_token();
        CHECK(s.open(token), "the open failed");
        CHECK(s.gather(rows, 2, dst, sizeof(dst)) == 0 && dst[0] == 1 && dst[4] == 2, "the gather was not forwarded");
        CHECK(s.open(token) && g_opened == 1, "a second open reached the backend: %d opens", g_opened);
        CHECK(g_log.find("already open") != std::string::npos, "a second open did not WARN");
        const int32_t bad[1] = { 99 };
        CHECK(s.gather(bad, 1, dst, 4) == EIO, "a failed gather did not return its errno");
    }
    llama_log_set(nullptr, nullptr);
    CHECK(g_closed == 1, "closed %d times", g_closed);
    CHECK(count_log(k_counters_line) == 1, "the counters line was not printed exactly once: %s", g_log.c_str());
    CHECK(g_level == GGML_LOG_LEVEL_WARN, "the counters line was logged at level %d, not WARN", g_level);

    // Never opened: nothing to close and nothing to report.
    log_capture();
    {
        llama_lazy_storage s(table, k_path, procs);
    }
    llama_log_set(nullptr, nullptr);
    CHECK(g_closed == 0 && g_opened == 0, "a never-opened storage reached the backend");
    CHECK(count_log("[LAZY-STORAGE]") == 0, "a never-opened storage printed a counters line: %s", g_log.c_str());

    // The backend refuses the open: the handle stays unopened.
    log_capture();
    g_open_fails = true;
    {
        llama_lazy_storage s(table, k_path, procs);
        CHECK(!s.open(make_token()), "a refused open returned true");
        CHECK(s.gather(rows, 2, dst, sizeof(dst)) == ENODEV, "a gather after a refused open did not return ENODEV");
    }
    llama_log_set(nullptr, nullptr);
    CHECK(g_closed == 0, "a refused open was closed %d times", g_closed);
    CHECK(count_log("[LAZY-STORAGE]") == 0, "a refused open printed a counters line: %s", g_log.c_str());

    // No backend: all four procs null.
    log_capture();
    {
        llama_lazy_storage s(table, k_path, llama_lazy_storage_procs{});
        CHECK(!s.open(make_token()), "an open with no procs returned true");
        CHECK(g_log.find("no backend") != std::string::npos, "an open with no procs did not WARN");
        CHECK(s.gather(rows, 2, dst, sizeof(dst)) == ENODEV, "a gather with no procs did not return ENODEV");
    }
    llama_log_set(nullptr, nullptr);

    // One proc missing: a handle that could not be closed must never exist.
    log_capture();
    {
        llama_lazy_storage_procs partial = procs;
        partial.close                    = nullptr;
        llama_lazy_storage s(table, k_path, partial);
        CHECK(!s.open(make_token()), "an open with no close proc returned true");
        CHECK(g_opened == 0, "an open with no close proc reached the backend");
    }
    llama_log_set(nullptr, nullptr);
    CHECK(count_log("[LAZY-STORAGE]") == 0, "a storage with no close proc printed a counters line");
}

int main() {
    // The handle does not depend on the backend, so it is checked in every build.
    check_handle();
    const bool converter_ran = check_converter();
    if (g_fails != 0) {
        return 1;
    }
    if (!converter_ran) {
        std::printf("SKIP: no SYCL in this build, so only the converter was skipped; the handle checks ran\n");
        return 77;
    }
    return 0;
}
