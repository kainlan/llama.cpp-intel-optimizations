#pragma once

// The per-model lazy-range record (llama.cpp-5efe C1; design sycl-mmap-host-tier.md §3.1).
//
// One row per TENSOR_READ_LAZY tensor the architecture declares and the model file
// carries, whatever any consumer decides about it. A row holds the tensor's
// (file, offset, size), whether it is eligible to be read lazily, who defers its
// read, and the attributes of its consumer. The pre-fill writes every field once,
// before the early placement plan, and a later write aborts; consumers read the
// record and keep no copy of the ranges or the decision.

#include "llama-arch.h"
#include "llama-hparams.h"
#include "llama-mmap.h"
#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

// who reads a tensor's rows on demand
enum llama_lazy_deferral {
    LLAMA_LAZY_DEFERRAL_EAGER     = 0,  // not lazy: the tensor is an ordinary weight
    LLAMA_LAZY_DEFERRAL_LAZY_READ = 1,  // lazy, read through the CPU mapping (lazy_read)
    LLAMA_LAZY_DEFERRAL_STORAGE   = 2,  // lazy, owned by the SYCL STORAGE tier (llama.cpp-th32 T6)
};

const char * llama_lazy_deferral_name(enum llama_lazy_deferral deferral);

// what the consumer of a lazy table needs to know about it
struct llama_lazy_tensor_attrs {
    uint32_t rows_per_token;  // 0 = not known for this arch
    int32_t  consumer_layer;  // -1 = not known for this arch
};

// the table's place in its file, handed out by value
struct llama_lazy_range_value {
    uint32_t file_idx;
    size_t   offs;
    size_t   size;
};

// a declared tensor the model file carries, as the pre-fill reads it from the loader
struct llama_lazy_range_source {
    uint32_t file_idx;
    size_t   offs;
    size_t   size;
};

struct llama_lazy_range_row {
    std::string              name;
    llama_lazy_range_value   value;
    bool                     lazy_eligible;
    enum llama_lazy_deferral deferral;
    llama_lazy_tensor_attrs  attrs;
};

class llama_lazy_range_record {
  public:
    // Set the rows once. Aborts when the record is already filled: rows, the bit,
    // `deferral` and the attributes are immutable afterwards, so a by-value copy
    // taken at plan publish stays equal to the record for the model's lifetime.
    void fill(std::vector<llama_lazy_range_row> rows);

    bool filled() const { return filled_; }

    const std::vector<llama_lazy_range_row> & rows() const { return rows_; }

    // per-tensor queries; has() is false for an undeclared name, and the other
    // accessors abort on one
    bool                     has(const std::string & name) const;
    bool                     eligible(const std::string & name) const;
    enum llama_lazy_deferral deferral(const std::string & name) const;
    llama_lazy_tensor_attrs  attrs(const std::string & name) const;
    llama_lazy_range_value   value(const std::string & name) const;

    // sorted, merged [first, last) byte ranges of one file's rows by deferral
    llama_mmap::ranges deferred_for_file(uint32_t file_idx) const;  // LAZY_READ rows only
    llama_mmap::ranges storage_for_file(uint32_t file_idx) const;   // STORAGE rows only
    llama_mmap::ranges excluded_for_file(uint32_t file_idx) const;  // their union

  private:
    const llama_lazy_range_row * find(const std::string & name) const;
    llama_mmap::ranges           ranges_for_file(uint32_t file_idx, bool lazy_read, bool storage) const;

    std::vector<llama_lazy_range_row> rows_;
    bool                              filled_ = false;
};

// looks a declared tensor up in the model file(s); false when the file does not carry it
using llama_lazy_range_find_fn = std::function<bool(const std::string & name, llama_lazy_range_source & out)>;

// The pre-fill. Fills `record` once, before the early plan and before any
// create_tensor() call, with one row per name in `declared` (a name, or a pattern
// with one "%d" for the block index) that `find` returns.
//
//   lazy_eligible = llama_lazy_eligible(taker ? mode_pre : mode_post, size,
//                                       taker ? true : mmap_supported)
//   deferral      = STORAGE   when eligible and `taker`
//                   LAZY_READ when eligible and not `taker`
//                   EAGER     otherwise
//
// `taker` is the model's fact "its weights are SYCL-owned and the hooks resolve"
// (sycl_model_loading_guard.active), not the process-level "a SYCL backend is
// registered". The taker reads a table through its own O_DIRECT fd and never
// through a mapping, so the load-mode fallback, which exists because the CPU lazy
// path needs a mapping, must not disengage it: it reads the mode as requested
// (`mode_pre`). Off the taker, lazy_read really reads with `mode_post`.
//
// Prints one INFO line per model and one per row (read by the lead's witnesses).
void llama_lazy_range_record_prefill(llama_lazy_range_record &        record,
                                     enum llm_arch                    arch,
                                     const llama_hparams &            hparams,
                                     const std::vector<std::string> & declared,
                                     const llama_lazy_range_find_fn & find,
                                     enum llama_lazy_mode             mode_pre,
                                     enum llama_lazy_mode             mode_post,
                                     bool                             taker,
                                     bool                             mmap_supported);

// The early plan's inventory filter: true for a tensor it must not place, stage or
// count as an ordinary weight. Only a STORAGE row; an ineligible one is the ordinary
// weight it is.
bool llama_lazy_early_plan_excludes(const llama_lazy_range_record & record, const std::string & name);

// The drift check of create_tensor(): a TENSOR_READ_LAZY call must have a row, and a
// call without the flag must not. Aborts naming the tensor and the architecture, in
// every mode, whatever the row's eligibility and deferral.
void llama_lazy_range_check_create(const llama_lazy_range_record & record,
                                   enum llm_arch                   arch,
                                   const std::string &             name,
                                   bool                            flagged);

// The STORAGE refusal of create_tensor(): a TENSOR_READ_LAZY tensor whose row is
// STORAGE belongs to the SYCL storage tier, and this tree has no consumer for it, so
// the load refuses it by throwing std::runtime_error naming the tensor, before
// buft_for_tensor() can make it an ordinary input-layer weight in host memory. Silent
// on a LAZY_READ or EAGER row; aborts on a name without a row.
void llama_lazy_range_check_storage(const llama_lazy_range_record & record, const std::string & name);
