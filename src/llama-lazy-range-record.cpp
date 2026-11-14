#include "llama-lazy-range-record.h"

#include "llama-impl.h"
#include "llama-model-loader.h"
#include "llama-model.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

const char * llama_lazy_deferral_name(enum llama_lazy_deferral deferral) {
    switch (deferral) {
        case LLAMA_LAZY_DEFERRAL_EAGER:
            return "EAGER";
        case LLAMA_LAZY_DEFERRAL_LAZY_READ:
            return "LAZY_READ";
        case LLAMA_LAZY_DEFERRAL_STORAGE:
            return "STORAGE";
    }
    return "?";
}

static const char * llama_lazy_mode_name(enum llama_lazy_mode mode) {
    switch (mode) {
        case LLAMA_LAZY_MODE_OFF:
            return "off";
        case LLAMA_LAZY_MODE_AUTO:
            return "auto";
        case LLAMA_LAZY_MODE_ON:
            return "on";
    }
    return "?";
}

//
// llama_lazy_range_record
//

void llama_lazy_range_record::fill(std::vector<llama_lazy_range_row> rows) {
    if (filled_) {
        GGML_ABORT(
            "lazy-range record: written twice; rows, the eligibility bit, `deferral` and the attributes are "
            "set once, by the pre-fill");
    }
    rows_   = std::move(rows);
    filled_ = true;
}

const llama_lazy_range_row * llama_lazy_range_record::find(const std::string & name) const {
    for (const llama_lazy_range_row & row : rows_) {
        if (row.name == name) {
            return &row;
        }
    }
    return nullptr;
}

bool llama_lazy_range_record::has(const std::string & name) const {
    return find(name) != nullptr;
}

bool llama_lazy_range_record::eligible(const std::string & name) const {
    const llama_lazy_range_row * row = find(name);
    GGML_ASSERT(row != nullptr && "lazy-range record: no row for the tensor");
    return row->lazy_eligible;
}

enum llama_lazy_deferral llama_lazy_range_record::deferral(const std::string & name) const {
    const llama_lazy_range_row * row = find(name);
    GGML_ASSERT(row != nullptr && "lazy-range record: no row for the tensor");
    return row->deferral;
}

llama_lazy_tensor_attrs llama_lazy_range_record::attrs(const std::string & name) const {
    const llama_lazy_range_row * row = find(name);
    GGML_ASSERT(row != nullptr && "lazy-range record: no row for the tensor");
    return row->attrs;
}

llama_lazy_range_value llama_lazy_range_record::value(const std::string & name) const {
    const llama_lazy_range_row * row = find(name);
    GGML_ASSERT(row != nullptr && "lazy-range record: no row for the tensor");
    return row->value;
}

llama_mmap::ranges llama_lazy_range_record::ranges_for_file(uint32_t file_idx, bool lazy_read, bool storage) const {
    llama_mmap::ranges ranges;
    for (const llama_lazy_range_row & row : rows_) {
        if (row.value.file_idx != file_idx) {
            continue;
        }
        const bool take = (lazy_read && row.deferral == LLAMA_LAZY_DEFERRAL_LAZY_READ) ||
                          (storage && row.deferral == LLAMA_LAZY_DEFERRAL_STORAGE);
        if (take) {
            ranges.emplace_back(row.value.offs, row.value.offs + row.value.size);
        }
    }

    // sorted and merged: back-to-back and overlapping ranges become one
    std::sort(ranges.begin(), ranges.end());
    llama_mmap::ranges merged;
    for (const auto & range : ranges) {
        if (!merged.empty() && range.first <= merged.back().second) {
            merged.back().second = std::max(merged.back().second, range.second);
        } else {
            merged.push_back(range);
        }
    }
    return merged;
}

llama_mmap::ranges llama_lazy_range_record::deferred_for_file(uint32_t file_idx) const {
    return ranges_for_file(file_idx, true, false);
}

llama_mmap::ranges llama_lazy_range_record::storage_for_file(uint32_t file_idx) const {
    return ranges_for_file(file_idx, false, true);
}

llama_mmap::ranges llama_lazy_range_record::excluded_for_file(uint32_t file_idx) const {
    return ranges_for_file(file_idx, true, true);
}

//
// the pre-fill, the early-plan filter and the create_tensor check
//

// defined beside llama_lazy_range_check_storage(), the refusal it bridges
static bool llama_lazy_storage_consumer_present(enum llm_arch arch);

// "per_layer_token_embd.weight" -> itself; "blk.%d.x.weight" -> one name per block
static std::vector<std::string> llama_lazy_expand_declared(const std::string & declared, uint32_t n_layer) {
    const size_t pos = declared.find("%d");
    if (pos == std::string::npos) {
        return { declared };
    }
    std::vector<std::string> names;
    for (uint32_t il = 0; il < n_layer; ++il) {
        names.push_back(declared.substr(0, pos) + std::to_string(il) + declared.substr(pos + 2));
    }
    return names;
}

void llama_lazy_range_record_prefill(llama_lazy_range_record &        record,
                                     enum llm_arch                    arch,
                                     const llama_hparams &            hparams,
                                     const std::vector<std::string> & declared,
                                     const llama_lazy_range_find_fn & find,
                                     enum llama_lazy_mode             mode_pre,
                                     enum llama_lazy_mode             mode_post,
                                     bool                             taker,
                                     bool                             mmap_supported) {
    // the taker reads the mode as requested: its reads go through its own O_DIRECT fd,
    // and the load-mode fallback exists because the CPU lazy path needs a mapping, so
    // it must not disengage the taker. Off the taker, lazy_read reads with mode_post.
    const enum llama_lazy_mode mode = taker ? mode_pre : mode_post;

    LLAMA_LOG_INFO("%s: lazy-range record: mode_pre=%s mode_post=%s taker=%d\n", __func__,
                   llama_lazy_mode_name(mode_pre), llama_lazy_mode_name(mode_post), taker ? 1 : 0);

    std::vector<llama_lazy_range_row> rows;
    for (const std::string & pattern : declared) {
        for (const std::string & name : llama_lazy_expand_declared(pattern, hparams.n_layer_all)) {
            llama_lazy_range_source src = {};
            if (!find(name, src)) {
                continue;
            }

            bool eligible = llama_lazy_eligible(mode, src.size, taker ? true : mmap_supported);

            // the default mode must load: without a STORAGE consumer the table is the ordinary weight
            // `-lzm off` makes it
            if (taker && eligible && mode == LLAMA_LAZY_MODE_AUTO && !llama_lazy_storage_consumer_present(arch)) {
                eligible = false;
                LLAMA_LOG_WARN(
                    "%s: tensor %s (size = %zu MiB) is loaded as an ordinary weight (read in full, placed by the "
                    "plan): this build has no storage tier for a SYCL-owned table, so -lzm on, which asks for it, "
                    "fails the load\n",
                    __func__, name.c_str(), src.size / 1024 / 1024);
            }

            // add()'s old warning, moved here: a table that would have been lazy,
            // read in full because this platform cannot map the file
            if (!taker && !mmap_supported && llama_lazy_eligible(mode, src.size, true)) {
                LLAMA_LOG_WARN("%s: mmap is not available, so tensor %s (size = %zu MiB) is loaded into RAM in full\n",
                               __func__, name.c_str(), src.size / 1024 / 1024);
            }

            llama_lazy_range_row row = {};
            row.name                 = name;
            row.value                = { src.file_idx, src.offs, src.size };
            row.lazy_eligible        = eligible;
            row.deferral             = !eligible ? LLAMA_LAZY_DEFERRAL_EAGER :
                                       taker     ? LLAMA_LAZY_DEFERRAL_STORAGE :
                                                   LLAMA_LAZY_DEFERRAL_LAZY_READ;
            row.attrs                = llama_model_lazy_tensor_attrs(arch, hparams, name);

            LLAMA_LOG_INFO("%s: lazy-range record: %s (%zu MiB) eligible=%d deferral=%s taker=%d mode=%s\n", __func__,
                           name.c_str(), src.size / 1024 / 1024, eligible ? 1 : 0,
                           llama_lazy_deferral_name(row.deferral), taker ? 1 : 0, llama_lazy_mode_name(mode));

            rows.push_back(std::move(row));
        }
    }

    record.fill(std::move(rows));
}

bool llama_lazy_early_plan_excludes(const llama_lazy_range_record & record, const std::string & name) {
    return record.has(name) && record.deferral(name) == LLAMA_LAZY_DEFERRAL_STORAGE;
}

void llama_lazy_range_check_create(const llama_lazy_range_record & record,
                                   enum llm_arch                   arch,
                                   const std::string &             name,
                                   bool                            flagged) {
    const bool declared = record.has(name);
    if (flagged && !declared) {
        GGML_ABORT(
            "lazy-range record: tensor %s of arch %s is created with TENSOR_READ_LAZY but has no row; "
            "llama_model_lazy_tensor_names() does not declare it",
            name.c_str(), llm_arch_name(arch));
    }
    if (!flagged && declared) {
        GGML_ABORT("lazy-range record: tensor %s of arch %s has a row but is created without TENSOR_READ_LAZY",
                   name.c_str(), llm_arch_name(arch));
    }
}

// Whether `arch`'s graph consumes a STORAGE row. None does yet: llama_lazy_range_check_storage() below refuses a
// STORAGE row for an arch without one, and the pre-fill does not make one from the default mode (AUTO). An arch
// gains its consumer by returning true here, which moves the `consumer` flag of
// tests/test-llama-lazy-range-prefill.cpp. Once every arch has one, this function goes, and with it the bridge in
// the pre-fill and the refusal below; the pin test_the_auto_bridge_lives_exactly_as_long_as_the_refusal in
// tests/test-llama-lazy-range-record-source.py holds those three together.
static bool llama_lazy_storage_consumer_present(enum llm_arch arch) {
    GGML_UNUSED(arch);
    return false;
}

std::vector<llama_lazy_storage_table> llama_lazy_range_storage_tables(const llama_lazy_range_record &    record,
                                                                      const llama_lazy_range_tensor_fn & tensor) {
    std::vector<llama_lazy_storage_table> out;
    for (const llama_lazy_range_row & row : record.rows()) {
        if (row.deferral != LLAMA_LAZY_DEFERRAL_STORAGE) {
            continue;
        }
        const ggml_tensor * t = tensor(row.name);
        if (t == nullptr) {
            GGML_ABORT("lazy-range record: STORAGE row %s has no tensor in the model file", row.name.c_str());
        }
        llama_lazy_storage_table s = {};
        s.name                     = row.name;
        s.type                     = t->type;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            s.ne[d] = t->ne[d];
        }
        s.row_size = ggml_row_size(t->type, t->ne[0]);
        s.value    = row.value;
        s.attrs    = row.attrs;
        out.push_back(std::move(s));
    }
    return out;
}

void llama_lazy_range_check_storage(const llama_lazy_range_record & record,
                                    enum llm_arch                   arch,
                                    const std::string &             name) {
    if (record.deferral(name) == LLAMA_LAZY_DEFERRAL_STORAGE && !llama_lazy_storage_consumer_present(arch)) {
        throw std::runtime_error(format("lazy-range record: %s is STORAGE and arch %s has no STORAGE consumer",
                                        name.c_str(), llm_arch_name(arch)));
    }
}
