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
// recovers. Asking skip() about a held signature answers "skip" retry_after - 1 times; the retry_after-th ask
// forgets the signature and answers "decide", so one token in retry_after re-runs the pass.
//
// Two key spaces feed the memo. Every recorder but one keys by ggml_sycl_graph_signature(cgraph); the dense
// split recorder keys by its own plan hash (graphs_key()). They are different hash functions over different
// inputs, so the dense one goes through dense_split_key(): a collision between the spaces is then a 2^-64 event
// rather than a structural overlap.
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
        uint64_t hash     = 0;
        uint32_t skips    = 0;
        // Token (graph-compute call) that last counted against this entry. A token can ask about one signature
        // more than once (the early check, then the recorder), and must still count once.
        uint64_t last_seq = 0;
    };

    // Tag for the dense split recorder's plan hash, so it cannot be mistaken for a graph signature.
    static uint64_t dense_split_key(uint64_t graphs_key) { return graphs_key ^ 0xD5E5B11700DECA11ULL; }

    std::vector<entry> entries;

    bool contains(uint64_t hash) const {
        for (const entry & e : entries) {
            if (e.hash == hash) {
                return true;
            }
        }
        return false;
    }

    // Record a decline made on token `seq`. A signature already held starts its skip count again.
    void remember(uint64_t hash, uint64_t seq) {
        for (entry & e : entries) {
            if (e.hash == hash) {
                e.skips    = 0;
                e.last_seq = seq;
                return;
            }
        }
        if (entries.size() >= max_entries) {
            entries.erase(entries.begin());
        }
        entries.push_back({ hash, 0, seq });
    }

    // Token `seq` asks whether `hash` is still declined. True means skip the pass. Each token counts once: a
    // second ask on the same `seq` (or on the token that declined) answers true without counting. The
    // retry_after-th counted token forgets the signature and answers false, so the caller re-decides on it. This
    // MUTATES the memo (it counts the ask), which is why the backend wrapper is called graph_prestage_skip_declined.
    bool skip(uint64_t hash, uint64_t seq) {
        for (size_t i = 0; i < entries.size(); ++i) {
            if (entries[i].hash != hash) {
                continue;
            }
            if (entries[i].last_seq == seq) {
                return true;
            }
            entries[i].last_seq = seq;
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
