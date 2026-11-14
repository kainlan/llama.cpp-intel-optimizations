#include "llama-lazy-storage.h"

#include "llama-impl.h"
#include "llama-model.h"

#include <cerrno>
#include <cstring>
#include <utility>

llama_lazy_storage::llama_lazy_storage(const llama_lazy_storage_table & table,
                                       std::string                      file_path,
                                       const llama_lazy_storage_procs & procs) :
    table_(table),
    file_path_(std::move(file_path)),
    procs_(procs) {}

llama_lazy_storage::~llama_lazy_storage() {
    if (handle_ == nullptr) {
        return;
    }
    // WARN so that a normal run shows the line: it is the evidence that the table was read through the backend.
    if (procs_.counters != nullptr) {
        uint64_t c[4] = {};
        procs_.counters(handle_, c);
        LLAMA_LOG_WARN("[LAZY-STORAGE] %s hits=%llu misses=%llu evictions=%llu submitted=%llu\n", table_.name.c_str(),
                       (unsigned long long) c[0], (unsigned long long) c[1], (unsigned long long) c[2],
                       (unsigned long long) c[3]);
    }
    if (procs_.close != nullptr) {
        procs_.close(handle_);
    }
    handle_ = nullptr;
}

bool llama_lazy_storage::open(const llama_sycl_model_token & token) {
    if (procs_.open == nullptr) {
        LLAMA_LOG_WARN("%s: no backend to open the storage table %s\n", __func__, table_.name.c_str());
        return false;
    }
    if (handle_ != nullptr) {
        return true;
    }
    handle_ = procs_.open(token, table_, file_path_.c_str());
    return handle_ != nullptr;
}

int llama_lazy_storage::gather(const int32_t * rows, size_t n, void * dst, size_t dst_bytes) const {
    if (handle_ == nullptr || procs_.gather == nullptr) {
        return ENODEV;
    }
    return procs_.gather(handle_, rows, n, dst, dst_bytes);
}

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

#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
namespace {

// The backend's procs, resolved once by name from the SYCL registration; null members when it has none.
struct llama_lazy_storage_backend_procs {
    ggml_sycl_lazy_storage * (*open)(ggml_sycl_model_token, const ggml_sycl_lazy_table_info *, const char *) = nullptr;
    int (*gather)(ggml_sycl_lazy_storage *, const int32_t *, size_t, void *, size_t)                         = nullptr;
    void (*counters)(const ggml_sycl_lazy_storage *, uint64_t[4])                                            = nullptr;
    void (*close)(ggml_sycl_lazy_storage *)                                                                  = nullptr;
};

llama_lazy_storage_backend_procs & llama_lazy_storage_backend() {
    static llama_lazy_storage_backend_procs procs;
    return procs;
}

void * llama_lazy_storage_adapt_open(llama_sycl_model_token           token,
                                     const llama_lazy_storage_table & table,
                                     const char *                     file_path) {
    ggml_sycl_model_token t              = {};
    t.model_id                           = token.model_id;
    t.load_txn_id                        = token.load_txn_id;
    t.slot                               = token.slot;
    t.slot_generation                    = token.slot_generation;
    // The info points into `table`, which the caller keeps alive and unmoved across this call.
    const ggml_sycl_lazy_table_info info = llama_lazy_storage_table_info(table);
    return llama_lazy_storage_backend().open(t, &info, file_path);
}

int llama_lazy_storage_adapt_gather(void * handle, const int32_t * rows, size_t n, void * dst, size_t dst_bytes) {
    return llama_lazy_storage_backend().gather(static_cast<ggml_sycl_lazy_storage *>(handle), rows, n, dst, dst_bytes);
}

void llama_lazy_storage_adapt_counters(const void * handle, uint64_t out[4]) {
    llama_lazy_storage_backend().counters(static_cast<const ggml_sycl_lazy_storage *>(handle), out);
}

void llama_lazy_storage_adapt_close(void * handle) {
    llama_lazy_storage_backend().close(static_cast<ggml_sycl_lazy_storage *>(handle));
}

}  // namespace
#endif

llama_lazy_storage_procs llama_lazy_storage_sycl_procs() {
    llama_lazy_storage_procs result = {};
#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
    const size_t dev_count = ggml_backend_dev_count();
    for (size_t i = 0; i < dev_count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (!reg || std::strcmp(ggml_backend_reg_name(reg), "SYCL") != 0) {
            continue;
        }
        llama_lazy_storage_backend_procs & b = llama_lazy_storage_backend();

        b.open = reinterpret_cast<decltype(b.open)>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_lazy_storage_open"));
        b.gather = reinterpret_cast<decltype(b.gather)>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_lazy_storage_gather"));
        b.counters = reinterpret_cast<decltype(b.counters)>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_lazy_storage_counters"));
        b.close = reinterpret_cast<decltype(b.close)>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_lazy_storage_close"));
        if (b.open && b.gather && b.counters && b.close) {
            result.open     = llama_lazy_storage_adapt_open;
            result.gather   = llama_lazy_storage_adapt_gather;
            result.counters = llama_lazy_storage_adapt_counters;
            result.close    = llama_lazy_storage_adapt_close;
        }
        break;
    }
#endif
    return result;
}
