// The planned load terms by owner (llama.cpp-ihk6).
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

enum class planned_load_term { COMPUTE, STATE };

// One owner's contribution on one device. generation 0: no owner holds it (no slot generation is ever 0).
struct planned_load_term_contribution {
    uint64_t generation    = 0;
    size_t   compute_bytes = 0;
    size_t   state_bytes   = 0;
};

// What a store swept: per device, the generation of the older contribution it removed for the owner's slot (0: none).
struct planned_load_terms_replaced {
    uint64_t generation[GGML_SYCL_MAX_DEVICES] = {};
};

class planned_load_terms {
  public:
    // Store `bytes` as `owner`'s `term` on `device`, replacing the value that owner stored before: a re-measure within
    // one owner replaces, it does not merge with itself. False, storing nothing, for a device out of range, a slot that
    // is not a model slot (lifecycle::buffer_owner_slot and lifecycle::no_model_slot among them), generation 0, or an
    // owner older than a contribution the slot holds on any device. A newer owner of the slot removes every older
    // contribution of the slot, on every device, whole; the generation removed on each device is written to `replaced`.
    bool set(int                           device,
             lifecycle::SlotToken          owner,
             planned_load_term             term,
             size_t                        bytes,
             planned_load_terms_replaced * replaced = nullptr) {
        if (replaced != nullptr) {
            *replaced = planned_load_terms_replaced{};
        }
        if (device < 0 || device >= GGML_SYCL_MAX_DEVICES || owner.slot >= lifecycle::model_slot_count ||
            owner.generation == 0) {
            return false;
        }
        for (int d = 0; d < GGML_SYCL_MAX_DEVICES; ++d) {
            if (contributions_[d][owner.slot].generation > owner.generation) {
                return false;
            }
        }
        for (int d = 0; d < GGML_SYCL_MAX_DEVICES; ++d) {
            planned_load_term_contribution & old = contributions_[d][owner.slot];
            if (old.generation != 0 && old.generation < owner.generation) {
                if (replaced != nullptr) {
                    replaced->generation[d] = old.generation;
                }
                old = planned_load_term_contribution{};
            }
        }
        planned_load_term_contribution & c = contributions_[device][owner.slot];
        c.generation                       = owner.generation;
        if (term == planned_load_term::COMPUTE) {
            c.compute_bytes = bytes;
        } else {
            c.state_bytes = bytes;
        }
        return true;
    }

    // Remove `owner`'s contributions on every device. Another owner's, and a newer generation of the same slot's, are
    // left alone. True when something was removed.
    bool drop(lifecycle::SlotToken owner) {
        if (owner.slot >= lifecycle::model_slot_count || owner.generation == 0) {
            return false;
        }
        bool dropped = false;
        for (int device = 0; device < GGML_SYCL_MAX_DEVICES; ++device) {
            planned_load_term_contribution & c = contributions_[device][owner.slot];
            if (c.generation == owner.generation) {
                c       = planned_load_term_contribution{};
                dropped = true;
            }
        }
        return dropped;
    }

    // The merge over the owners on `device`: the largest contribution of `term`, 0 with none (or out of range).
    size_t merged(int device, planned_load_term term) const {
        if (device < 0 || device >= GGML_SYCL_MAX_DEVICES) {
            return 0;
        }
        size_t out = 0;
        for (uint32_t slot = 0; slot < lifecycle::model_slot_count; ++slot) {
            const planned_load_term_contribution & c = contributions_[device][slot];
            if (c.generation == 0) {
                continue;
            }
            const size_t bytes = term == planned_load_term::COMPUTE ? c.compute_bytes : c.state_bytes;
            if (bytes > out) {
                out = bytes;
            }
        }
        return out;
    }

  private:
    planned_load_term_contribution contributions_[GGML_SYCL_MAX_DEVICES][lifecycle::model_slot_count]{};
};

}  // namespace ggml_sycl
