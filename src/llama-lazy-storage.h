#pragma once

#include "llama-lazy-range-record.h"

#include <cstddef>
#include <cstdint>
#include <string>

struct llama_sycl_model_token;

// The backend's four storage procs, as libllama calls them; the handle is the backend's, opaque here.
struct llama_lazy_storage_procs {
    void * (*open)(llama_sycl_model_token token, const llama_lazy_storage_table & table, const char * file_path);
    int (*gather)(void * handle, const int32_t * rows, size_t n, void * dst, size_t dst_bytes);
    // out[0] hits, out[1] misses, out[2] evictions, out[3] waiters submitted for misses (not disk reads)
    void (*counters)(const void * handle, uint64_t out[4]);
    void (*close)(void * handle);
};

// The SYCL backend's procs, adapted; all null when the backend lacks any of them, or in a build without SYCL.
llama_lazy_storage_procs llama_lazy_storage_sycl_procs();

// A STORAGE table the model's graph reads through set_input: created before the load's measure, so the graph builds
// its F32 input instead of a gather from the table, and opened once the load has a model token. Not copyable: the
// backend handle is closed exactly once.
class llama_lazy_storage {
  public:
    llama_lazy_storage(const llama_lazy_storage_table & table,
                       std::string                      file_path,
                       const llama_lazy_storage_procs & procs);
    ~llama_lazy_storage();
    llama_lazy_storage(const llama_lazy_storage &)             = delete;
    llama_lazy_storage & operator=(const llama_lazy_storage &) = delete;

    bool open(const llama_sycl_model_token & token);  // false, after a WARN, when the backend refuses
    int  gather(const int32_t * rows, size_t n, void * dst, size_t dst_bytes) const;  // 0, ENODEV unopened, or errno

    const std::string & name() const { return table_.name; }

    ggml_type type() const { return table_.type; }

    int64_t ne0() const { return table_.ne[0]; }

    size_t row_size() const { return table_.row_size; }

  private:
    llama_lazy_storage_table table_;
    std::string              file_path_;
    llama_lazy_storage_procs procs_;
    void *                   handle_ = nullptr;
};

#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
#    include "ggml-sycl.h"

// The backend's view of a STORAGE table, for the planner's inventory and for the open proc: one conversion, so the two
// cannot describe one table differently. `name` points into `table.name`: the table must outlive the info and must
// not be moved or modified while the info is in use.
ggml_sycl_lazy_table_info llama_lazy_storage_table_info(const llama_lazy_storage_table & table);
#endif
