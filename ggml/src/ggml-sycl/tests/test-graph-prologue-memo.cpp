//
// Test: the per-context prologue memo of a graph llama reuses
//
// The decode prologue walks the graph three times on every token: the structural signature, the keyed segment slot
// key and the input prestage. A graph llama reuses comes back with the same ggml_cgraph::uid and the same tensors, so
// the memo serves those walks from the first full pass under the same key. This pins the memo's switch, its key,
// what each part records and serves, the uid-0 rule, invalidation and the verify comparison. Host-only: the header
// includes no SYCL header and names no ggml type.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "graph-prologue-memo.hpp"

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

// The build is -DNDEBUG (Release), so assert() would compile away and the test would pass vacuously.
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

namespace gpm = ggml_sycl::graph_prologue_memo;
namespace gsc = ggml_sycl::graph_segment_cache;

namespace {

struct fake_tensor {
    int id;
};

struct fake_buffer {
    uint64_t     identity;
    const char * base;
};

using fake_memo   = gpm::memo<fake_tensor, const fake_buffer *>;
using fake_effect = gpm::prestage_effect<fake_tensor>;
using fake_ident  = gpm::buffer_identity<const fake_buffer *>;

gpm::key make_key(uint64_t uid) {
    gpm::key k;
    k.uid        = uid;
    k.n_nodes    = 100;
    k.n_leafs    = 3;
    k.device     = 0;
    k.is_decode  = true;
    k.tenant_gen = 7;
    return k;
}

gsc::key make_slot_key(uint64_t storage) {
    gsc::key k;
    k.signature = 11;
    k.names     = 22;
    k.storage   = storage;
    k.n_nodes   = 100;
    k.device    = 0;
    k.is_decode = true;
    return k;
}

uint64_t identity_of(const fake_buffer * b) {
    return b->identity;
}

const char * base_of(const fake_buffer * b) {
    return b->base;
}

}  // namespace

int main() {
    // The switch: unset or empty is on, the value is read as an integer, 0 is off, 2 is verify, anything else on.
    {
        CHECK(gpm::mode_parse(nullptr) == gpm::mode::ON, "unset is on");
        CHECK(gpm::mode_parse("") == gpm::mode::ON, "empty is on");
        CHECK(gpm::mode_parse("1") == gpm::mode::ON, "1 is on");
        CHECK(gpm::mode_parse("0") == gpm::mode::OFF, "0 is off");
        CHECK(gpm::mode_parse("off") == gpm::mode::OFF, "a value that reads as 0 is off");
        CHECK(gpm::mode_parse("2") == gpm::mode::VERIFY, "2 is verify");
        CHECK(gpm::mode_parse("3") == gpm::mode::ON, "any other number is on");
    }

    // A part recorded under a key is served under the same key, on the next call too.
    {
        fake_memo memo;
        memo.begin(make_key(5));
        CHECK(memo.signature() == nullptr, "nothing is served before a record");
        memo.record_signature(42);
        memo.begin(make_key(5));
        CHECK(memo.signature() != nullptr && *memo.signature() == 42, "the same key serves the recorded signature");
    }

    // Every field of the key matters: a call that differs in any one of them drops every part.
    {
        for (int field = 0; field < 6; ++field) {
            fake_memo memo;
            memo.begin(make_key(5));
            memo.record_signature(42);
            gpm::key other = make_key(5);
            switch (field) {
                case 0:
                    other.uid = 6;
                    break;
                case 1:
                    other.n_nodes = 99;
                    break;
                case 2:
                    other.n_leafs = 4;
                    break;
                case 3:
                    other.device = 1;
                    break;
                case 4:
                    other.is_decode = false;
                    break;
                default:
                    other.tenant_gen = 8;
                    break;
            }
            CHECK(!(other == make_key(5)), "the key compares the changed field");
            memo.begin(other);
            CHECK(memo.signature() == nullptr, "a key that differs in one field drops the parts");
            memo.begin(make_key(5));
            CHECK(memo.signature() == nullptr, "and they do not come back when the old key returns");
        }
    }

    // uid 0 means "not set": it never hits and records nothing, and it drops what an earlier key recorded.
    {
        fake_memo memo;
        memo.begin(make_key(0));
        memo.record_signature(42);
        memo.record_slot_key(make_slot_key(1), std::vector<fake_ident>());
        memo.record_prestage(true, std::vector<fake_effect>(1, fake_effect{}));
        memo.begin(make_key(0));
        CHECK(memo.signature() == nullptr, "uid 0 records no signature");
        CHECK(memo.slot_key() == nullptr, "uid 0 records no slot key");
        CHECK(memo.prestage_effects() == nullptr, "uid 0 records no prestage list");

        memo.begin(make_key(5));
        memo.record_signature(42);
        memo.begin(make_key(0));
        CHECK(memo.signature() == nullptr, "a uid-0 call drops the parts of the key before it");
    }

    // clear() drops every part and disarms until the next begin(); the counters survive it.
    {
        fake_memo memo;
        memo.begin(make_key(5));
        memo.record_signature(42);
        memo.note_hit();
        memo.clear();
        CHECK(memo.signature() == nullptr, "clear drops the signature");
        memo.record_signature(43);
        CHECK(memo.signature() == nullptr, "a cleared memo records nothing before the next begin");
        memo.begin(make_key(5));
        CHECK(memo.signature() == nullptr, "the next begin starts empty");
        memo.record_signature(44);
        CHECK(memo.signature() != nullptr && *memo.signature() == 44, "and records again");
        CHECK(memo.stats().hits == 1, "clear keeps the counters");
    }

    // The prestage list is recorded only for a pass that staged everything, and is served in its order.
    {
        fake_tensor a{ 1 };
        fake_tensor b{ 2 };
        fake_tensor c{ 3 };
        fake_memo   memo;
        memo.begin(make_key(5));
        const fake_effect effect_a{ &a, gpm::prestage_arm::INPUT, "leaf" };
        const fake_effect effect_b{ &b, gpm::prestage_arm::CACHE, "node_src" };
        const fake_effect effect_c{ &c, gpm::prestage_arm::CACHE, "node_src" };
        memo.record_prestage(false, std::vector<fake_effect>(1, effect_a));
        CHECK(memo.prestage_effects() == nullptr, "a pass that did not stage everything is not recorded");
        std::vector<fake_effect> walk;
        walk.push_back(effect_c);
        walk.push_back(effect_a);
        walk.push_back(effect_b);
        memo.record_prestage(true, std::move(walk));
        const std::vector<fake_effect> * held = memo.prestage_effects();
        CHECK(held != nullptr && held->size() == 3, "a pass that staged everything is recorded");
        CHECK((*held)[0].tensor == &c && (*held)[1].tensor == &a && (*held)[2].tensor == &b, "in its order");
        CHECK((*held)[1].arm == gpm::prestage_arm::INPUT && (*held)[0].arm == gpm::prestage_arm::CACHE,
              "with each tensor's arm");
        memo.record_signature(42);
        memo.drop_prestage();
        CHECK(memo.prestage_effects() == nullptr, "drop_prestage drops the list");
        CHECK(memo.signature() != nullptr, "and only the list");
        memo.record_prestage(true, {});
        CHECK(memo.prestage_effects() != nullptr && memo.prestage_effects()->empty(),
              "an empty list (a pass with no effect arm) is a valid record");
        memo.record_prestage(false, std::vector<fake_effect>(1, effect_a));
        CHECK(memo.prestage_effects() == nullptr, "a later pass that did not stage everything drops the held list");
    }

    // Two effects are equal by tensor and arm; the source label is only for the failure log.
    {
        fake_tensor       a{ 1 };
        fake_tensor       b{ 2 };
        const fake_effect a_leaf{ &a, gpm::prestage_arm::INPUT, "leaf" };
        const fake_effect a_src{ &a, gpm::prestage_arm::INPUT, "node_src" };
        const fake_effect b_leaf{ &b, gpm::prestage_arm::INPUT, "leaf" };
        const fake_effect a_cache{ &a, gpm::prestage_arm::CACHE, "leaf" };
        CHECK(a_leaf == a_src, "the source label does not take part");
        CHECK(!(a_leaf == b_leaf), "the tensor does");
        CHECK(!(a_leaf == a_cache), "the arm does");
    }

    // The slot key is served only while every distinct buffer the full walk met keeps its identity and its base.
    {
        fake_buffer x{ 100, reinterpret_cast<const char *>(0x1000) };
        fake_buffer y{ 200, reinterpret_cast<const char *>(0x2000) };
        fake_memo   memo;
        memo.begin(make_key(5));
        std::vector<fake_ident> met;
        met.push_back(fake_ident{ &x, 100, x.base });
        met.push_back(fake_ident{ &y, 200, y.base });
        memo.record_slot_key(make_slot_key(9), std::move(met));
        CHECK(memo.slot_key() != nullptr && *memo.slot_key() == make_slot_key(9), "the slot key is recorded");
        CHECK(memo.slot_key_buffers().size() == 2, "with the distinct buffers the walk met");
        CHECK(memo.slot_key_buffers_unchanged(identity_of, base_of), "unchanged buffers serve it");
        y.identity = 201;
        CHECK(!memo.slot_key_buffers_unchanged(identity_of, base_of), "a changed identity does not");
        y.identity = 200;
        x.base     = reinterpret_cast<const char *>(0x1800);
        CHECK(!memo.slot_key_buffers_unchanged(identity_of, base_of), "a changed base does not");
        x.base = reinterpret_cast<const char *>(0x1000);
        CHECK(memo.slot_key_buffers_unchanged(identity_of, base_of), "restored buffers serve it again");
    }

    // Verify: a comparison that agrees counts and keeps the memo; one that differs counts a mismatch and clears it.
    {
        fake_memo memo;
        memo.begin(make_key(5));
        memo.record_signature(42);
        CHECK(memo.note_verified(true), "an agreeing comparison answers true");
        CHECK(memo.signature() != nullptr, "and keeps the memo");
        CHECK(!memo.note_verified(false), "a differing comparison answers false");
        CHECK(memo.signature() == nullptr, "and clears the memo");
        CHECK(memo.stats().verified == 2 && memo.stats().mismatches == 1, "both are counted, the difference apart");
    }

    // The counters the teardown summary prints.
    {
        fake_memo memo;
        memo.note_hit();
        memo.note_hit();
        memo.note_miss();
        CHECK(memo.stats().hits == 2 && memo.stats().misses == 1, "hits and misses are counted");
        CHECK(memo.stats().verified == 0 && memo.stats().mismatches == 0, "with nothing verified");
    }

    // The mismatch line names which part differed.
    {
        CHECK(std::string(gpm::part_name(gpm::part::SIGNATURE)) == "signature", "the signature is named");
        CHECK(std::string(gpm::part_name(gpm::part::SLOT_KEY)) == "slot key", "the slot key is named");
        CHECK(std::string(gpm::part_name(gpm::part::PRESTAGE)) == "prestage effect list and answer",
              "the prestage is named");
    }

    std::printf("graph-prologue-memo: all checks passed\n");
    return 0;
}
