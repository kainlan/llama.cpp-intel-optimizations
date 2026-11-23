// The converter that hands a STORAGE row of the lazy-range record to the SYCL backend as a lazy table, and the handle
// libllama keeps on an opened table with its gather of rows as f32, checked on fake procs. Host-only: no backend,
// device or model; the handle checks run in a build without SYCL too.

#include "llama-lazy-storage.h"
#include "llama-model.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

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
        CHECK(count_log("no backend") == 1, "an open with no procs did not WARN exactly once");
        CHECK(s.gather(rows, 2, dst, sizeof(dst)) == ENODEV, "a gather with no procs did not return ENODEV");
    }
    llama_log_set(nullptr, nullptr);

    // One proc missing, each of the four in turn: a handle that could not be read, counted or closed must never exist.
    const char * const names[4] = { "open", "gather", "counters", "close" };
    for (int missing = 0; missing < 4; ++missing) {
        log_capture();
        {
            llama_lazy_storage_procs partial = procs;
            if (missing == 0) {
                partial.open = nullptr;
            } else if (missing == 1) {
                partial.gather = nullptr;
            } else if (missing == 2) {
                partial.counters = nullptr;
            } else {
                partial.close = nullptr;
            }
            llama_lazy_storage s(table, k_path, partial);
            CHECK(!s.open(make_token()), "an open with no %s proc returned true", names[missing]);
            CHECK(g_opened == 0, "an open with no %s proc reached the backend", names[missing]);
            CHECK(count_log("no backend") == 1, "an open with no %s proc did not WARN exactly once", names[missing]);
            CHECK(s.gather(rows, 2, dst, sizeof(dst)) == ENODEV, "a gather with no %s proc did not return ENODEV",
                  names[missing]);
        }
        llama_log_set(nullptr, nullptr);
        CHECK(count_log("[LAZY-STORAGE]") == 0, "a storage with no %s proc printed a counters line", names[missing]);
    }
}

// gather_f32 on fake procs whose gather writes known bytes for each requested row: a Q8_0 table, whose values are
// checked against d * q worked out here rather than by the conversion under test, and an F32 table, read as it is.
static ggml_type g_typed_type      = GGML_TYPE_COUNT;
static int64_t   g_typed_ne0       = 0;
static size_t    g_typed_row_size  = 0;
static int       g_typed_fail      = 0;  // the errno the fake gather returns, or 0
static int       g_typed_calls     = 0;
static size_t    g_typed_dst_bytes = 0;

// Q8_0 rows of two blocks; each block's scale is a power of two, exact in fp16 and in f32, so d * q is exact too.
static const int64_t  k_q8_ne0        = 64;
static const size_t   k_q8_row_size   = 2 * (sizeof(uint16_t) + 32);
static const uint16_t k_q8_d_bits[4]  = { 0x3400, 0x3800, 0x3C00, 0x4000 };
static const float    k_q8_d_value[4] = { 0.25f, 0.5f, 1.0f, 2.0f };

static int q8_scale(int32_t row, int64_t block) {
    return (int) ((row + block) % 4);
}

static int8_t q8_quant(int32_t row, int64_t block, int64_t j) {
    return (int8_t) ((row * 7 + block * 3 + j) % 51 - 25);
}

static float q8_value(int32_t row, int64_t k) {
    const int64_t block = k / 32;
    return k_q8_d_value[q8_scale(row, block)] * (float) q8_quant(row, block, k % 32);
}

static const int64_t k_f32_ne0 = 8;

static float f32_value(int32_t row, int64_t k) {
    return (float) row * 16.0f + (float) k * 0.25f;
}

static int fake_gather_typed(void * handle, const int32_t * rows, size_t n, void * dst, size_t dst_bytes) {
    ++g_typed_calls;
    g_typed_dst_bytes = dst_bytes;
    // the backend's own refusals: another handle, or a destination shorter than n rows
    if (handle != &g_handle || dst == nullptr || dst_bytes < n * g_typed_row_size) {
        return EINVAL;
    }
    if (g_typed_fail != 0) {
        return g_typed_fail;
    }
    for (size_t i = 0; i < n; ++i) {
        uint8_t * row = static_cast<uint8_t *>(dst) + i * g_typed_row_size;
        if (g_typed_type == GGML_TYPE_F32) {
            for (int64_t k = 0; k < g_typed_ne0; ++k) {
                const float v = f32_value(rows[i], k);
                std::memcpy(row + k * sizeof(float), &v, sizeof(float));
            }
        } else {
            for (int64_t b = 0; b < g_typed_ne0 / 32; ++b) {
                uint8_t * block = row + b * (sizeof(uint16_t) + 32);
                std::memcpy(block, &k_q8_d_bits[q8_scale(rows[i], b)], sizeof(uint16_t));
                for (int64_t j = 0; j < 32; ++j) {
                    block[sizeof(uint16_t) + j] = (uint8_t) q8_quant(rows[i], b, j);
                }
            }
        }
    }
    return 0;
}

static llama_lazy_storage_table make_typed_table(ggml_type type, int64_t ne0, size_t row_size) {
    llama_lazy_storage_table table = make_table();
    table.type                     = type;
    table.ne[0]                    = ne0;
    table.ne[2] = table.ne[3] = 1;
    table.row_size            = row_size;
    g_typed_type              = type;
    g_typed_ne0               = ne0;
    g_typed_row_size          = row_size;
    g_typed_fail              = 0;
    g_typed_calls             = 0;
    g_typed_dst_bytes         = 0;
    return table;
}

// Every value of `out` is the expected value of its row, in the order the rows were asked for.
static bool values_match(const std::vector<float> & out, const int32_t * rows, size_t n, int64_t ne0, bool f32) {
    for (size_t i = 0; i < n; ++i) {
        for (int64_t k = 0; k < ne0; ++k) {
            const float want = f32 ? f32_value(rows[i], k) : q8_value(rows[i], k);
            if (out[i * ne0 + k] != want) {
                std::printf("  row %zu (id %d) value %lld: %g, want %g\n", i, rows[i], (long long) k,
                            (double) out[i * ne0 + k], (double) want);
                return false;
            }
        }
    }
    return true;
}

static void check_gather_f32() {
    const llama_lazy_storage_procs procs = { fake_open, fake_gather_typed, fake_counters, fake_close };

    // a Q8_0 table: rows out of order and one repeated, converted in the order asked
    log_capture();
    {
        llama_lazy_storage s(make_typed_table(GGML_TYPE_Q8_0, k_q8_ne0, k_q8_row_size), k_path, procs);
        CHECK(s.can_gather_f32(), "a Q8_0 table cannot be gathered as f32");
        CHECK(s.open(make_token()), "the Q8_0 open failed");
        const int32_t        rows[4] = { 5, 2, 9, 2 };
        std::vector<uint8_t> raw;
        std::vector<float>   out;
        const int            rc = s.gather_f32(rows, 4, raw, out);
        CHECK(rc == 0, "the Q8_0 gather_f32 returned %d", rc);
        CHECK(g_typed_calls == 1 && g_typed_dst_bytes == 4 * k_q8_row_size,
              "the Q8_0 rows were not read once into exactly 4 rows of bytes: %d call(s), %zu B", g_typed_calls,
              g_typed_dst_bytes);
        CHECK(out.size() == 4 * (size_t) k_q8_ne0, "the Q8_0 gather_f32 gave %zu values, want %lld", out.size(),
              (long long) (4 * k_q8_ne0));
        CHECK(out.size() == 4 * (size_t) k_q8_ne0 && values_match(out, rows, 4, k_q8_ne0, false),
              "the Q8_0 values are not d * q of the rows asked for, in order");

        // a warm call with fewer rows: the scratch is reused, not reallocated, and holds only the new rows
        const int32_t   more[6] = { 1, 2, 3, 4, 5, 6 };
        const int       rc6     = s.gather_f32(more, 6, raw, out);
        const uint8_t * raw_at  = raw.data();
        const float *   out_at  = out.data();
        const size_t    raw_cap = raw.capacity();
        const size_t    out_cap = out.capacity();

        const int32_t fewer[2] = { 8, 0 };
        const int     rc2      = s.gather_f32(fewer, 2, raw, out);
        CHECK(rc6 == 0 && rc2 == 0, "the reuse calls returned %d and %d", rc6, rc2);
        CHECK(raw.data() == raw_at && out.data() == out_at && raw.capacity() == raw_cap && out.capacity() == out_cap,
              "a call with fewer rows reallocated its scratch");
        CHECK(out.size() == 2 * (size_t) k_q8_ne0 && values_match(out, fewer, 2, k_q8_ne0, false),
              "a call with fewer rows did not return exactly its own rows: %zu values", out.size());

        // a failed read is returned as it came
        g_typed_fail = EIO;
        CHECK(s.gather_f32(rows, 4, raw, out) == EIO, "a failed Q8_0 read did not return EIO");
    }
    llama_log_set(nullptr, nullptr);

    // an F32 table: its rows are the values, read straight into `out`, with nothing read into `raw`
    log_capture();
    {
        llama_lazy_storage s(make_typed_table(GGML_TYPE_F32, k_f32_ne0, k_f32_ne0 * sizeof(float)), k_path, procs);
        CHECK(s.can_gather_f32(), "an F32 table cannot be gathered as f32");
        CHECK(s.open(make_token()), "the F32 open failed");
        const int32_t        rows[3] = { 3, 0, 7 };
        std::vector<uint8_t> raw;
        std::vector<float>   out;
        const int            rc = s.gather_f32(rows, 3, raw, out);
        CHECK(rc == 0, "the F32 gather_f32 returned %d", rc);
        CHECK(g_typed_calls == 1 && g_typed_dst_bytes == 3 * k_f32_ne0 * sizeof(float),
              "the F32 rows were not read once into exactly 3 rows of floats: %d call(s), %zu B", g_typed_calls,
              g_typed_dst_bytes);
        CHECK(out.size() == 3 * (size_t) k_f32_ne0 && values_match(out, rows, 3, k_f32_ne0, true),
              "the F32 values are not the rows asked for, in order: %zu values", out.size());
        CHECK(raw.capacity() == 0, "an F32 gather grew the raw scratch to %zu B", raw.capacity());

        g_typed_fail = EIO;
        CHECK(s.gather_f32(rows, 3, raw, out) == EIO, "a failed F32 read did not return EIO");
    }
    llama_log_set(nullptr, nullptr);

    // a type with no conversion: refused before anything is read
    log_capture();
    {
        llama_lazy_storage s(make_typed_table(GGML_TYPE_I32, 4, 4 * sizeof(int32_t)), k_path, procs);
        CHECK(!s.can_gather_f32(), "an I32 table claims it can be gathered as f32");
        CHECK(s.open(make_token()), "the I32 open failed");
        const int32_t        rows[2] = { 1, 2 };
        std::vector<uint8_t> raw;
        std::vector<float>   out;
        const int            rc = s.gather_f32(rows, 2, raw, out);
        CHECK(rc == ENOTSUP && g_typed_calls == 0,
              "an I32 gather_f32 returned %d after %d read(s), want ENOTSUP and none", rc, g_typed_calls);
    }
    llama_log_set(nullptr, nullptr);

    // a handle that was never opened answers what gather() answers, without reaching the backend
    log_capture();
    {
        llama_lazy_storage   s(make_typed_table(GGML_TYPE_Q8_0, k_q8_ne0, k_q8_row_size), k_path, procs);
        const int32_t        rows[1] = { 1 };
        std::vector<uint8_t> raw;
        std::vector<float>   out;
        const int            rc = s.gather_f32(rows, 1, raw, out);
        CHECK(rc == ENODEV && g_typed_calls == 0, "an unopened gather_f32 returned %d after %d read(s), want ENODEV",
              rc, g_typed_calls);
    }
    llama_log_set(nullptr, nullptr);
}

int main() {
    // The handle does not depend on the backend, so it is checked in every build.
    check_handle();
    check_gather_f32();
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
