#pragma once

// The log of which expert each recent move of a weight's expert storage generation changed. Every weight extra holds
// one (common.hpp, ggml_tensor_extra_gpu_weight_ext::moe_expert_storage_changes), and the hybrid decode record
// (moe-hybrid-record.hpp) reads it to re-check only the experts that moved since its decision.
//
// SYCL-free and free of project headers on purpose: common.hpp, which every SYCL source includes, includes it, and
// tests/test-sycl-moe-hybrid-record.cpp runs it without a device.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ggml_sycl {

// Which expert each recent move of a weight's expert storage generation changed. A holder of a decision made at an
// older generation (the hybrid decode record) asks which experts changed since then and re-checks only those. The log
// keeps the last `capacity` moves: it answers only for a window it saw whole, in which every move named one expert, so
// a move older than the log, one made without a note, or one that changed every expert leaves the caller to decide
// again from scratch.
struct moe_storage_change_log {
    static constexpr int32_t ALL      = -1;  // a move that changed every expert, or named none
    static constexpr size_t  capacity = 64;

    struct entry {
        uint64_t generation = 0;
        int32_t  expert     = ALL;
    };

    entry  ring[capacity] = {};
    size_t next           = 0;  // where the next move is logged
    size_t n_logged       = 0;  // moves logged, up to capacity

    // The generation moved to `generation`, changing `expert` (ALL: every expert).
    void note(uint64_t generation, int32_t expert) {
        ring[next] = { generation, expert };
        next       = (next + 1) % capacity;
        n_logged   = n_logged < capacity ? n_logged + 1 : capacity;
    }

    // The experts whose storage changed after generation `since` up to `now`, ascending and each once, into *out.
    // False when the log cannot say: the window is longer than the log holds, a move in it was not noted, or one
    // changed every expert.
    bool changed_since(uint64_t since, uint64_t now, std::vector<int32_t> * out) const {
        out->clear();
        if (now < since || now - since > n_logged) {
            return false;
        }
        size_t at = next;
        for (uint64_t g = now; g > since; --g) {
            at              = (at + capacity - 1) % capacity;
            const entry & e = ring[at];
            if (e.generation != g || e.expert < 0) {
                out->clear();
                return false;
            }
            out->push_back(e.expert);
        }
        std::sort(out->begin(), out->end());
        out->erase(std::unique(out->begin(), out->end()), out->end());
        return true;
    }
};

}  // namespace ggml_sycl
