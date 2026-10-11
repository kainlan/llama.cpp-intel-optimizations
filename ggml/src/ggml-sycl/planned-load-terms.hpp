// The planned load terms by owner (llama.cpp-ihk6, llama.cpp-45qo).
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// A load reserves two planned RUNTIME terms per device (llama.cpp-p6i0): the compute term (the scheduler's compute
// buffer) and the state term (the context memory on the device's plain buffer type). Several models can be live on
// one device, so each owner keeps its own contribution and the term a device reads is the merge, the largest
// contribution of the owners on it. Canonical contract 12.4 keys every reset by an explicit owner; here the key is the
// owner's model slot and slot generation. The slot is reserved and its generation minted when the load transaction
// begins (lifecycle::Registry::begin_outer), before any reservation runs, so a load in flight and the model it commits
// to have the same key and nothing moves at commit. An owner's contribution leaves only by its own key: the load's
// rollback, or the slot's release at teardown.
//
// The dense scratch plan's inputs live here too (llama.cpp-45qo): the Q8_1 src1 bytes per token, the f16 dequant
// weight maximum and src1 bytes per token, the n_ubatch they are sized at, and a flag per buffer for a figure the
// sizing rejected. unified-cache.cpp sizes the plan from the merge of each input, as it did from the device-global
// inputs these replace, and a flag reads set while any owner that raised it is live.
//
// A slot is reserved again only after its release, so once a newer generation of a slot stores, every contribution an
// older generation left, on any device, is one its owner's rollback or release did not drop. The store sweeps them all,
// not only the device it stores on: a stale term left on a device the new owner never reserves on would otherwise stand
// in that device's merge with no owner left to drop it.
//
// An unlocked value type with no SYCL dependency: unified-cache.cpp holds the process's table under a leaf mutex and
// publishes the merge to the atomics its readers load, and tests/test-planned-load-terms.cpp drives it directly.

#pragma once

#include "ggml-sycl.h"
#include "model-lifecycle.hpp"

#include <cstddef>
#include <cstdint>

namespace ggml_sycl {

// Every term merges the same way, as the largest contribution of the owners on the device.
enum class planned_load_term {
    COMPUTE,                           // the scheduler's compute buffer (llama.cpp-p6i0)
    STATE,                             // the context memory on the device's plain buffer type (llama.cpp-p6i0)
    MMQ_SRC1_BYTES_PER_TOKEN,          // the dense Q8_1 src1 buffer's bytes per token (llama.cpp-479i)
    DEQUANT_F16_WEIGHT_BYTES,          // the dense f16 dequant src0 copy: the largest weight it holds
    DEQUANT_F16_SRC1_BYTES_PER_TOKEN,  // the dense f16 dequant src1 copy's bytes per token
    DENSE_SCRATCH_N_UBATCH,            // the n_ubatch both dense scratch plans are sized at (llama.cpp-kpjw)
    MMQ_SRC1_REJECTED,                 // 1: the Q8_1 src1 figure overflowed its sizing
    DEQUANT_F16_REJECTED,              // 1: the f16 dequant figure overflowed its sizing
};

constexpr int planned_load_term_count = 8;

// One term and the value an owner stores for it.
struct planned_load_term_value {
    planned_load_term term;
    size_t            value;
};

// One owner's contribution on one device. generation 0: no owner holds it (no slot generation is ever 0).
struct planned_load_term_contribution {
    uint64_t generation                      = 0;
    size_t   values[planned_load_term_count] = {};
};

// What a store swept: per device, the generation of the older contribution it removed for the owner's slot (0: none).
struct planned_load_terms_replaced {
    uint64_t generation[GGML_SYCL_MAX_DEVICES] = {};
};

class planned_load_terms {
  public:
    // Store `values[0..n_values)` as `owner`'s terms on `device`, each replacing the value that owner stored for it
    // before: a re-measure within one owner replaces, it does not merge with itself. All or nothing: false, storing
    // nothing, for a device out of range, a slot that is not a model slot (lifecycle::buffer_owner_slot and
    // lifecycle::no_model_slot among them), generation 0, an owner older than a contribution the slot holds on any
    // device, a null or empty value list, or a term out of range. A newer owner of the slot removes every older
    // contribution of the slot, on every device, whole; the generation removed on each device is written to `replaced`.
    bool set(int                             device,
             lifecycle::SlotToken            owner,
             const planned_load_term_value * values,
             int                             n_values,
             planned_load_terms_replaced *   replaced = nullptr) {
        if (replaced != nullptr) {
            *replaced = planned_load_terms_replaced{};
        }
        if (!device_in_range(device) || !is_model_owner(owner) || values == nullptr || n_values <= 0) {
            return false;
        }
        for (int i = 0; i < n_values; ++i) {
            if (!term_in_range(values[i].term)) {
                return false;
            }
        }
        for (int d = 0; d < GGML_SYCL_MAX_DEVICES; ++d) {
            if (contributions_[d][owner.slot].generation > owner.generation) {
                return false;
            }
        }
        (void) clear_slot(owner.slot, owner.generation, slot_generations::OLDER, replaced);
        planned_load_term_contribution & c = contributions_[device][owner.slot];
        c.generation                       = owner.generation;
        for (int i = 0; i < n_values; ++i) {
            c.values[static_cast<int>(values[i].term)] = values[i].value;
        }
        return true;
    }

    // One term: set() with a list of one.
    bool set(int                           device,
             lifecycle::SlotToken          owner,
             planned_load_term             term,
             size_t                        bytes,
             planned_load_terms_replaced * replaced = nullptr) {
        const planned_load_term_value value = { term, bytes };
        return set(device, owner, &value, 1, replaced);
    }

    // Remove `owner`'s contributions on every device. Another owner's, and a newer generation of the same slot's, are
    // left alone. True when something was removed.
    bool drop(lifecycle::SlotToken owner) {
        if (!is_model_owner(owner)) {
            return false;
        }
        return clear_slot(owner.slot, owner.generation, slot_generations::EXACT, nullptr);
    }

    // The merge over the owners on `device`: the largest contribution of `term`, 0 with none (or out of range).
    size_t merged(int device, planned_load_term term) const { return merged_without_slot(device, term, no_slot); }

    // What merged(device, term) would read once `owner` stored `value` for `term` on `device`: the owner's own earlier
    // value, and anything an older generation of its slot left, would be replaced, so neither counts. Lets a caller
    // validate the merged figure before it stores. 0 for a device or term out of range.
    size_t merged_if_set(int device, lifecycle::SlotToken owner, planned_load_term term, size_t value) const {
        if (!device_in_range(device) || !term_in_range(term)) {
            return 0;
        }
        const size_t others = merged_without_slot(device, term, owner.slot);
        return value > others ? value : others;
    }

    // `owner`'s own value of `term` on `device`, into `*out`. False, writing 0, when the owner holds no contribution
    // there (another generation of its slot is another owner) or the arguments are out of range.
    bool get(int device, lifecycle::SlotToken owner, planned_load_term term, size_t * out) const {
        if (out == nullptr) {
            return false;
        }
        *out = 0;
        if (!device_in_range(device) || !is_model_owner(owner) || !term_in_range(term)) {
            return false;
        }
        const planned_load_term_contribution & c = contributions_[device][owner.slot];
        if (c.generation != owner.generation) {
            return false;
        }
        *out = c.values[static_cast<int>(term)];
        return true;
    }

  private:
    enum class slot_generations { EXACT, OLDER };

    static constexpr uint32_t no_slot = UINT32_MAX;

    static bool device_in_range(int device) { return device >= 0 && device < GGML_SYCL_MAX_DEVICES; }

    static bool term_in_range(planned_load_term term) {
        return static_cast<int>(term) >= 0 && static_cast<int>(term) < planned_load_term_count;
    }

    static bool is_model_owner(lifecycle::SlotToken owner) {
        return owner.slot < lifecycle::model_slot_count && owner.generation != 0;
    }

    // Clear `slot`'s contribution on every device where it is held by `generation` (EXACT, a drop) or by a generation
    // older than it (OLDER, a newer owner's sweep), writing each cleared generation to `replaced`. One walk for both,
    // so a drop and a sweep cannot come to disagree about which devices a slot's contribution can sit on.
    bool clear_slot(uint32_t                      slot,
                    uint64_t                      generation,
                    slot_generations              which,
                    planned_load_terms_replaced * replaced) {
        bool cleared = false;
        for (int d = 0; d < GGML_SYCL_MAX_DEVICES; ++d) {
            planned_load_term_contribution & c     = contributions_[d][slot];
            const bool                       older = c.generation != 0 && c.generation < generation;
            if (which == slot_generations::EXACT ? c.generation != generation : !older) {
                continue;
            }
            if (replaced != nullptr) {
                replaced->generation[d] = c.generation;
            }
            c       = planned_load_term_contribution{};
            cleared = true;
        }
        return cleared;
    }

    // The largest `term` over the owners on `device`, leaving out `skipped` (no_slot: leave out none).
    size_t merged_without_slot(int device, planned_load_term term, uint32_t skipped) const {
        if (!device_in_range(device) || !term_in_range(term)) {
            return 0;
        }
        size_t out = 0;
        for (uint32_t slot = 0; slot < lifecycle::model_slot_count; ++slot) {
            const planned_load_term_contribution & c = contributions_[device][slot];
            if (slot == skipped || c.generation == 0) {
                continue;
            }
            const size_t value = c.values[static_cast<int>(term)];
            if (value > out) {
                out = value;
            }
        }
        return out;
    }

    planned_load_term_contribution contributions_[GGML_SYCL_MAX_DEVICES][lifecycle::model_slot_count]{};
};

}  // namespace ggml_sycl
