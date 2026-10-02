//
// Test: the per-context memo of declined graph signatures (llama.cpp-qhfp)
//
// graph_prestage_or_decline declines a graph whose inputs cannot all be staged onto the device and remembers the
// signature so the pass is not repeated on every token. The memo used to be one thread_local (context pointer,
// hash) slot: every other signature evicted it, a new context at a freed context's address inherited it, and a
// decline was permanent even when the cause (a staging allocation) was transient. This pins the replacement.
// Host-only: the header includes no SYCL header.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "graph-prestage-decline-memo.hpp"

#include <cstdio>

// The build is -DNDEBUG (Release), so assert() would compile away and the test would pass vacuously.
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

int main() {
    // Each ask below is a new token unless a case says otherwise.
    uint64_t seq = 0;
    // An unknown signature is not declined.
    {
        graph_prestage_decline_memo memo;
        CHECK(!memo.skip(1, ++seq), "an unseen signature is not declined");
        CHECK(!memo.contains(1), "an unseen signature is not held");
    }

    // PP and decode signatures coexist: remembering one does not evict the other.
    {
        graph_prestage_decline_memo memo;
        memo.remember(10, ++seq);
        memo.remember(20, ++seq);
        CHECK(memo.skip(10, ++seq), "the first signature is still declined");
        CHECK(memo.skip(20, ++seq), "the second signature is declined too");
        CHECK(!memo.skip(30, ++seq), "a third signature is judged afresh");
    }

    // A decline is forgotten after retry_after skipped tokens, so the next token re-decides.
    {
        graph_prestage_decline_memo memo;
        memo.remember(7, ++seq);
        for (uint32_t i = 1; i < graph_prestage_decline_memo::retry_after; ++i) {
            CHECK(memo.skip(7, ++seq), "skipped while inside the retry window");
        }
        CHECK(!memo.skip(7, ++seq), "the retry_after-th ask forgets the decline");
        CHECK(!memo.contains(7), "a forgotten signature is not held");
        CHECK(!memo.skip(7, ++seq), "and stays forgotten until it is declined again");
        memo.remember(7, ++seq);
        CHECK(memo.skip(7, ++seq), "a fresh decline is skipped again");
    }

    // Declining again restarts the window.
    {
        graph_prestage_decline_memo memo;
        memo.remember(5, ++seq);
        for (uint32_t i = 0; i < graph_prestage_decline_memo::retry_after - 1; ++i) {
            (void) memo.skip(5, ++seq);
        }
        memo.remember(5, ++seq);
        CHECK(memo.entries.size() == 1, "re-declining a held signature does not add an entry");
        CHECK(memo.skip(5, ++seq), "re-declining restarts the retry window");
    }

    // A pre-stage that succeeds forgets an earlier decline.
    {
        graph_prestage_decline_memo memo;
        memo.remember(3, ++seq);
        memo.forget(3);
        CHECK(!memo.skip(3, ++seq), "a forgotten signature is not skipped");
        memo.forget(99);  // forgetting an unknown signature is harmless
    }

    // The memo is bounded: past max_entries the oldest signature goes.
    {
        graph_prestage_decline_memo memo;
        for (uint64_t h = 1; h <= graph_prestage_decline_memo::max_entries + 1; ++h) {
            memo.remember(h, ++seq);
        }
        CHECK(memo.entries.size() == graph_prestage_decline_memo::max_entries, "the memo stays at max_entries");
        CHECK(!memo.contains(1), "the oldest signature was dropped");
        CHECK(memo.contains(graph_prestage_decline_memo::max_entries + 1), "the newest signature is held");
    }

    // The dense split recorder's plan hash is tagged so it cannot collide with a graph signature of the same value.
    {
        graph_prestage_decline_memo memo;
        memo.remember(graph_prestage_decline_memo::dense_split_key(42), ++seq);
        CHECK(graph_prestage_decline_memo::dense_split_key(42) != 42, "the dense key differs from the raw value");
        CHECK(!memo.contains(42), "a graph signature equal to a dense plan hash is not declined by it");
        CHECK(memo.contains(graph_prestage_decline_memo::dense_split_key(42)), "the dense key itself is held");
    }

    // A token counts once however often it asks: the early check and the recorder both ask on one token.
    {
        graph_prestage_decline_memo memo;
        memo.remember(8, ++seq);
        const uint64_t declining_token = seq;
        CHECK(memo.skip(8, declining_token), "the declining token's own ask answers skip");
        CHECK(memo.entries[0].skips == 0, "the declining token's ask does not count");
        const uint64_t next = ++seq;
        for (int i = 0; i < 5; ++i) {
            CHECK(memo.skip(8, next), "repeat asks on one token answer skip");
        }
        CHECK(memo.entries[0].skips == 1, "five asks on one token count as one");
        // retry_after distinct tokens still forget it, however many asks each makes.
        for (uint32_t t = 1; t < graph_prestage_decline_memo::retry_after - 1; ++t) {
            const uint64_t tok = ++seq;
            CHECK(memo.skip(8, tok), "inside the window");
            CHECK(memo.skip(8, tok), "the second ask on a token still skips");
        }
        CHECK(!memo.skip(8, ++seq), "the retry_after-th distinct token forgets the decline");
    }

    std::printf("ok\n");
    return 0;
}
