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
    // An unknown signature is not declined.
    {
        graph_prestage_decline_memo memo;
        CHECK(!memo.skip(1), "an unseen signature is not declined");
        CHECK(!memo.contains(1), "an unseen signature is not held");
    }

    // PP and decode signatures coexist: remembering one does not evict the other.
    {
        graph_prestage_decline_memo memo;
        memo.remember(10);
        memo.remember(20);
        CHECK(memo.skip(10), "the first signature is still declined");
        CHECK(memo.skip(20), "the second signature is declined too");
        CHECK(!memo.skip(30), "a third signature is judged afresh");
    }

    // A decline is forgotten after retry_after skipped tokens, so the next token re-decides.
    {
        graph_prestage_decline_memo memo;
        memo.remember(7);
        for (uint32_t i = 1; i < graph_prestage_decline_memo::retry_after; ++i) {
            CHECK(memo.skip(7), "skipped while inside the retry window");
        }
        CHECK(!memo.skip(7), "the retry_after-th ask forgets the decline");
        CHECK(!memo.contains(7), "a forgotten signature is not held");
        CHECK(!memo.skip(7), "and stays forgotten until it is declined again");
        memo.remember(7);
        CHECK(memo.skip(7), "a fresh decline is skipped again");
    }

    // Declining again restarts the window.
    {
        graph_prestage_decline_memo memo;
        memo.remember(5);
        for (uint32_t i = 0; i < graph_prestage_decline_memo::retry_after - 1; ++i) {
            (void) memo.skip(5);
        }
        memo.remember(5);
        CHECK(memo.entries.size() == 1, "re-declining a held signature does not add an entry");
        CHECK(memo.skip(5), "re-declining restarts the retry window");
    }

    // A pre-stage that succeeds forgets an earlier decline.
    {
        graph_prestage_decline_memo memo;
        memo.remember(3);
        memo.forget(3);
        CHECK(!memo.skip(3), "a forgotten signature is not skipped");
        memo.forget(99);  // forgetting an unknown signature is harmless
    }

    // The memo is bounded: past max_entries the oldest signature goes.
    {
        graph_prestage_decline_memo memo;
        for (uint64_t h = 1; h <= graph_prestage_decline_memo::max_entries + 1; ++h) {
            memo.remember(h);
        }
        CHECK(memo.entries.size() == graph_prestage_decline_memo::max_entries, "the memo stays at max_entries");
        CHECK(!memo.contains(1), "the oldest signature was dropped");
        CHECK(memo.contains(graph_prestage_decline_memo::max_entries + 1), "the newest signature is held");
    }

    // The dense split recorder's plan hash is tagged so it cannot collide with a graph signature of the same value.
    {
        graph_prestage_decline_memo memo;
        memo.remember(graph_prestage_decline_memo::dense_split_key(42));
        CHECK(graph_prestage_decline_memo::dense_split_key(42) != 42, "the dense key differs from the raw value");
        CHECK(!memo.contains(42), "a graph signature equal to a dense plan hash is not declined by it");
        CHECK(memo.contains(graph_prestage_decline_memo::dense_split_key(42)), "the dense key itself is held");
    }

    std::printf("ok\n");
    return 0;
}
