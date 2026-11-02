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
// sizing rejected. unified-cache.cpp sizes each buffer's plan from the merge of each input over the owners whose figure
// for that buffer was accepted (merged_accepted): one model's overflow does not invalidate the plan of a model already
// running (llama.cpp-kpjw), and the rejected owner adds nothing to that buffer's plan while it is live.
//
// An owner's planned n_ubatch is the largest of the values its live backend contexts last published, each context's
// own entry replaced by its own later publish, or the load's n_ubatch when no context is live (planned_n_ubatch).
// Neither a latch nor a floor: the auto-ubatch ladder publishes each rung for real, so a rung that wins the transaction
// and then loses elsewhere must not leave the plan above the rung the ladder settles on, and a context running below
// the load's n_ubatch must not be held to it.
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
    DENSE_SCRATCH_LOAD_N_UBATCH,       // the n_ubatch the load planned at: the plan's with no live context
    MMQ_SRC1_REJECTED,                 // 1: the Q8_1 src1 figure overflowed its sizing
    DEQUANT_F16_REJECTED,              // 1: the f16 dequant figure overflowed its sizing
    COUNT,                             // not a term: the number of terms, so a term added above is counted with it
};

constexpr int planned_load_term_count = static_cast<int>(planned_load_term::COUNT);

// One term and the value an owner stores for it.
struct planned_load_term_value {
    planned_load_term term;
    size_t            value;
};

// The n_ubatch one backend context last published for its owner's dense scratch plan. context 0: a free entry (the
// backend's context ids are minted from 1).
struct planned_context_n_ubatch {
    uint64_t context  = 0;
    size_t   n_ubatch = 0;
};

// How many backend contexts of one model on one device can each record a published n_ubatch. A context past it is
// refused by name (context_room) rather than left out of the plan its buffers are sized from. One llama_context is
// one backend context per device. The in-tree vehicles create few per model: test-thread-safety registers -np 4 (four
// contexts per model), llama-server makes one context (its slots are sequences of it) plus one draft context when
// speculative decoding is on, and the speculative code makes at most one draft context from a model. 8 covers those
// with room; a by-name refusal for the ninth context of a model means raise this constant.
constexpr int planned_context_capacity = 8;

// One owner's contribution on one device. generation 0: no owner holds it (no slot generation is ever 0). The context
// entries leave with the contribution, so a drop or a newer owner's sweep clears them too.
struct planned_load_term_contribution {
    uint64_t                 generation                         = 0;
    size_t                   values[planned_load_term_count]    = {};
    planned_context_n_ubatch contexts[planned_context_capacity] = {};
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
    size_t merged(int device, planned_load_term term) const { return merged_over(device, term, no_slot, no_filter); }

    // The merge over the owners on `device` whose `rejected` flag term reads 0: an owner whose figure was rejected
    // contributes none of its terms, so its overflow leaves the merge of the owners already running as it was. 0 with
    // none, or for an argument out of range.
    size_t merged_accepted(int device, planned_load_term term, planned_load_term rejected) const {
        if (!term_in_range(rejected)) {
            return 0;
        }
        return merged_over(device, term, no_slot, static_cast<int>(rejected));
    }

    // What merged_accepted(device, term, rejected) would read once `owner` stored `value` for `term` as an accepted
    // figure: the owner's own earlier value and flag, and anything an older generation of its slot left, would be
    // replaced, so none of them counts. Lets a caller validate the merged figure before it stores. 0 for an argument
    // out of range.
    size_t merged_accepted_if_set(int                  device,
                                  lifecycle::SlotToken owner,
                                  planned_load_term    term,
                                  size_t               value,
                                  planned_load_term    rejected) const {
        if (!device_in_range(device) || !term_in_range(term) || !term_in_range(rejected)) {
            return 0;
        }
        const size_t others = merged_over(device, term, owner.slot, static_cast<int>(rejected));
        return value > others ? value : others;
    }

    // Record `n_ubatch` as the value backend context `context` last published for `owner` on `device`. The context's
    // own earlier value is replaced, so the auto-ubatch ladder's settle lowers what a losing rung raised; every other
    // context's value stays. False, recording nothing, when the owner holds no contribution there (another generation
    // of its slot is another owner), for context 0 or n_ubatch 0, an argument out of range, or no room (context_room).
    bool set_context_n_ubatch(int device, lifecycle::SlotToken owner, uint64_t context, size_t n_ubatch) {
        planned_load_term_contribution * c = owned(device, owner);
        if (c == nullptr || context == 0 || n_ubatch == 0) {
            return false;
        }
        int i = context_index(*c, context);
        if (i < 0) {
            i = context_index(*c, 0);
        }
        if (i < 0) {
            return false;
        }
        c->contexts[i] = planned_context_n_ubatch{ context, n_ubatch };
        return true;
    }

    // Whether `context` can record a value for `owner` on `device`: it holds an entry there, one is free, or the owner
    // holds no contribution there (nothing is recorded then, and nothing is refused for want of room). Context 0 is
    // never a backend context's id and set_context_n_ubatch refuses it, so it has no room either.
    bool context_room(int device, lifecycle::SlotToken owner, uint64_t context) const {
        const planned_load_term_contribution * c = owned(device, owner);
        return c == nullptr || (context != 0 && (context_index(*c, context) >= 0 || context_index(*c, 0) >= 0));
    }

    // Remove backend context `context`'s entry on `device` (the context released its hold or is destroyed), writing
    // the owner it was recorded for to `*owner` (may be null). A backend context publishes for one model, so at most
    // one owner holds it. True when an entry was removed.
    bool drop_context(int device, uint64_t context, lifecycle::SlotToken * owner) {
        if (!device_in_range(device) || context == 0) {
            return false;
        }
        for (uint32_t slot = 0; slot < lifecycle::model_slot_count; ++slot) {
            planned_load_term_contribution & c = contributions_[device][slot];
            const int                        i = c.generation != 0 ? context_index(c, context) : -1;
            if (i < 0) {
                continue;
            }
            c.contexts[i] = planned_context_n_ubatch{};
            if (owner != nullptr) {
                *owner = lifecycle::SlotToken{ slot, c.generation };
            }
            return true;
        }
        return false;
    }

    // The n_ubatch `owner`'s dense scratch is planned at on `device`: the largest value its live contexts last
    // published, or its DENSE_SCRATCH_LOAD_N_UBATCH when none is live. 0 when the owner holds no contribution there
    // or an argument is out of range.
    size_t planned_n_ubatch(int device, lifecycle::SlotToken owner) const {
        const planned_load_term_contribution * c = owned(device, owner);
        if (c == nullptr) {
            return 0;
        }
        size_t out = 0;
        for (const planned_context_n_ubatch & entry : c->contexts) {
            if (entry.context != 0 && entry.n_ubatch > out) {
                out = entry.n_ubatch;
            }
        }
        return out != 0 ? out : c->values[static_cast<int>(planned_load_term::DENSE_SCRATCH_LOAD_N_UBATCH)];
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

    static constexpr uint32_t no_slot   = UINT32_MAX;
    static constexpr int      no_filter = -1;

    static bool device_in_range(int device) { return device >= 0 && device < GGML_SYCL_MAX_DEVICES; }

    static bool term_in_range(planned_load_term term) {
        return static_cast<int>(term) >= 0 && static_cast<int>(term) < planned_load_term_count;
    }

    static bool is_model_owner(lifecycle::SlotToken owner) {
        return owner.slot < lifecycle::model_slot_count && owner.generation != 0;
    }

    // `owner`'s contribution on `device`, or null when another generation of its slot (or none) holds it or the
    // arguments are out of range.
    const planned_load_term_contribution * owned(int device, lifecycle::SlotToken owner) const {
        if (!device_in_range(device) || !is_model_owner(owner)) {
            return nullptr;
        }
        const planned_load_term_contribution & c = contributions_[device][owner.slot];
        return c.generation == owner.generation ? &c : nullptr;
    }

    planned_load_term_contribution * owned(int device, lifecycle::SlotToken owner) {
        return const_cast<planned_load_term_contribution *>(
            static_cast<const planned_load_terms *>(this)->owned(device, owner));
    }

    // The index of `c`'s entry recorded for `context` (0: the first free entry), or -1.
    static int context_index(const planned_load_term_contribution & c, uint64_t context) {
        for (int i = 0; i < planned_context_capacity; ++i) {
            if (c.contexts[i].context == context) {
                return i;
            }
        }
        return -1;
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

    // The largest `term` over the owners on `device`, leaving out `skipped` (no_slot: leave out none) and every owner
    // whose `filter` term is nonzero (no_filter: leave out none).
    size_t merged_over(int device, planned_load_term term, uint32_t skipped, int filter) const {
        if (!device_in_range(device) || !term_in_range(term)) {
            return 0;
        }
        size_t out = 0;
        for (uint32_t slot = 0; slot < lifecycle::model_slot_count; ++slot) {
            const planned_load_term_contribution & c = contributions_[device][slot];
            if (slot == skipped || c.generation == 0 || (filter != no_filter && c.values[filter] != 0)) {
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
