#pragma once

#include "llama-lazy-range-record.h"

#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
#    include "ggml-sycl.h"

// The backend's view of a STORAGE table, for the planner's inventory and for the open proc: one conversion, so the two
// cannot describe one table differently. `name` points into `table.name`: the table must outlive the info and must not be
// moved or modified while the info is in use.
ggml_sycl_lazy_table_info llama_lazy_storage_table_info(const llama_lazy_storage_table & table);
#endif
