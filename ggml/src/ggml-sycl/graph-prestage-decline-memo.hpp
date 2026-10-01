//
// Memo of the graph signatures a context declined to record because their inputs could not all be staged onto the
// device (llama.cpp-qhfp).
//
// The decline is decided by a pre-stage pass over the whole graph, so re-deciding on every token would repeat that
// pass for a graph that cannot change its answer. The memo lives on the context that declined, for three reasons:
//   * it is destroyed with the context, so no later context can inherit it (an address-keyed slot could, when a
//     new context lands where an old one lived);
//   * it holds several signatures, so a context that alternates a prompt-processing graph and a decode graph does
//     not make each evict the other and run the pass on every token;
//   * it is not a thread-local, so the thread that tears the context down is the one that frees it.
//
// A decline is not permanent. A structural one (an input that is host-resident for this placement) will decline
// again, at the cost of one pass; a transient one (a staging allocation that failed under memory pressure)
// recovers. After retry_after skipped tokens the memo forgets the signature, so the next token re-decides.
//
// Pure C++, so a host test can pin the retry and capacity behaviour.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

struct graph_prestage_decline_memo {
    // Tokens that skip the pass before the signature is judged afresh.
    static constexpr uint32_t retry_after = 256;
    // Signatures held at once; the oldest is dropped beyond this.
    static constexpr size_t   max_entries = 16;

    struct entry {
        uint64_t hash  = 0;
        uint32_t skips = 0;
    };

    std::vector<entry> entries;

    bool contains(uint64_t hash) const {
        for (const entry & e : entries) {
            if (e.hash == hash) {
                return true;
            }
        }
        return false;
    }

    // Record a decline. A signature already held starts its skip count again.
    void remember(uint64_t hash) {
        for (entry & e : entries) {
            if (e.hash == hash) {
                e.skips = 0;
                return;
            }
        }
        if (entries.size() >= max_entries) {
            entries.erase(entries.begin());
        }
        entries.push_back({ hash, 0 });
    }

    // One token asks whether `hash` is still declined. True means skip the pass. The retry_after-th ask forgets
    // the signature and answers false, so the caller re-decides on this token.
    bool skip(uint64_t hash) {
        for (size_t i = 0; i < entries.size(); ++i) {
            if (entries[i].hash != hash) {
                continue;
            }
            if (++entries[i].skips >= retry_after) {
                entries.erase(entries.begin() + (std::ptrdiff_t) i);
                return false;
            }
            return true;
        }
        return false;
    }

    void forget(uint64_t hash) {
        for (size_t i = 0; i < entries.size(); ++i) {
            if (entries[i].hash == hash) {
                entries.erase(entries.begin() + (std::ptrdiff_t) i);
                return;
            }
        }
    }
};
