// Host tests of the chunk lease holder table (chunk-lease-holders.hpp), which GGML_SYCL_LEASE_HOLDERS=1 uses to name
// the code that holds each VRAM arena chunk lease.
//
// What this proves: the slot a take gets from its tag and call stack, which the cache counts the lease in and the
// release lowers. The same holder gets the same slot every time and from any thread, so a release lowers the slot its
// acquire raised; a holder's frames start at the address the tag recorded, so the cache's own frames never split one
// holder into two; a tag with no address, a full table and an unclaimed slot all answer slot 0 or no holder. What it
// does not prove: that the unwinder walks the backend's frames on a device run, or that the cache moves its counts
// only behind the flag (the source contract test-sycl-chunk-lease-holder-source.py pins that).

#include "chunk-lease-holders.hpp"

#include <cstdio>
#include <thread>
#include <vector>

using ggml_sycl::chunk_lease_holder;
using ggml_sycl::CHUNK_LEASE_HOLDER_FRAMES;
using ggml_sycl::CHUNK_LEASE_HOLDER_SLOTS;
using ggml_sycl::chunk_lease_holder_table;
using ggml_sycl::chunk_lease_site;
using ggml_sycl::chunk_lease_tag;

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
            return 1;                                                                     \
        }                                                                                 \
    } while (0)

// A fake return address: distinct, never dereferenced.
static void * addr(uintptr_t v) {
    return reinterpret_cast<void *>(v);
}

static chunk_lease_tag tag_at(const void *     pc,
                              chunk_lease_site site = chunk_lease_site::TENSOR_EXTRA,
                              bool             copy = true) {
    chunk_lease_tag tag;
    tag.site = site;
    tag.copy = copy;
    tag.pc   = pc;
    return tag;
}

static int case_null_address_is_slot_zero() {
    chunk_lease_holder_table table;
    void *                   stack[] = { addr(0x10), addr(0x20) };
    CHECK(table.claim(tag_at(nullptr), stack, 2) == 0);
    chunk_lease_holder h;
    CHECK(!table.read(0, h));
    CHECK(!table.read(1, h));  // nothing was claimed
    return 0;
}

static int case_frames_start_at_the_recorded_address() {
    chunk_lease_holder_table table;
    // The cache's own frames (0x10, 0x20) sit below the take's; the tag recorded 0x30.
    void *                   stack[] = { addr(0x10), addr(0x20), addr(0x30), addr(0x40), addr(0x50) };
    const uint16_t           slot    = table.claim(tag_at(addr(0x30)), stack, 5);
    CHECK(slot != 0);
    chunk_lease_holder h;
    CHECK(table.read(slot, h));
    CHECK(h.n_frames == 3);
    CHECK(h.frames[0] == addr(0x30) && h.frames[1] == addr(0x40) && h.frames[2] == addr(0x50));
    CHECK(h.site == chunk_lease_site::TENSOR_EXTRA && h.copy);
    // The same take from a different depth inside the cache (an inlined bump, another helper) is the same holder.
    void * deeper[] = { addr(0x11), addr(0x12), addr(0x13), addr(0x30), addr(0x40), addr(0x50) };
    CHECK(table.claim(tag_at(addr(0x30)), deeper, 6) == slot);
    return 0;
}

static int case_address_off_the_stack_is_the_address_alone() {
    chunk_lease_holder_table table;
    void *                   stack[] = { addr(0x10), addr(0x20) };
    const uint16_t           slot    = table.claim(tag_at(addr(0x99)), stack, 2);
    chunk_lease_holder       h;
    CHECK(table.read(slot, h));
    CHECK(h.n_frames == 1 && h.frames[0] == addr(0x99));
    // An unwinder that walked nothing gives the same holder.
    CHECK(table.claim(tag_at(addr(0x99)), stack, 0) == slot);
    return 0;
}

static int case_frames_are_capped() {
    chunk_lease_holder_table table;
    std::vector<void *>      stack;
    for (uintptr_t i = 1; i <= 20; ++i) {
        stack.push_back(addr(i * 0x100));
    }
    const uint16_t     slot = table.claim(tag_at(addr(0x100)), stack.data(), static_cast<int>(stack.size()));
    chunk_lease_holder h;
    CHECK(table.read(slot, h));
    CHECK(h.n_frames == CHUNK_LEASE_HOLDER_FRAMES);
    CHECK(h.frames[CHUNK_LEASE_HOLDER_FRAMES - 1] == addr(CHUNK_LEASE_HOLDER_FRAMES * 0x100));
    return 0;
}

static int case_distinct_holders_get_distinct_slots() {
    chunk_lease_holder_table table;
    void *                   a[]  = { addr(0x30), addr(0x40) };
    void *                   b[]  = { addr(0x30), addr(0x41) };  // same taker, another caller above it
    const uint16_t           sa   = table.claim(tag_at(addr(0x30)), a, 2);
    const uint16_t           sb   = table.claim(tag_at(addr(0x30)), b, 2);
    const uint16_t           ssrc = table.claim(tag_at(addr(0x30), chunk_lease_site::TENSOR_EXTRA, false), a, 2);
    const uint16_t           site = table.claim(tag_at(addr(0x30), chunk_lease_site::CHUNK_VIEW, true), a, 2);
    CHECK(sa != 0 && sb != 0 && ssrc != 0 && site != 0);
    CHECK(sa != sb && sa != ssrc && sa != site && sb != ssrc && sb != site && ssrc != site);
    CHECK(table.claim(tag_at(addr(0x30)), a, 2) == sa);
    return 0;
}

static int case_full_table_answers_slot_zero() {
    chunk_lease_holder_table table;
    for (uintptr_t i = 1; i < CHUNK_LEASE_HOLDER_SLOTS; ++i) {
        void * stack[] = { addr(i) };
        CHECK(table.claim(tag_at(addr(i)), stack, 1) == i);
    }
    void * stack[] = { addr(0xffff) };
    CHECK(table.claim(tag_at(addr(0xffff)), stack, 1) == 0);
    // A holder claimed before the table filled keeps its slot.
    void * first[] = { addr(1) };
    CHECK(table.claim(tag_at(addr(1)), first, 1) == 1);
    return 0;
}

static int case_threads_agree_on_a_slot() {
    chunk_lease_holder_table table;
    constexpr int            n_threads                 = 8;
    constexpr int            n_holders                 = 16;
    uint16_t                 got[n_threads][n_holders] = {};
    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; ++t) {
        threads.emplace_back([&table, &got, t] {
            for (int k = 0; k < n_holders; ++k) {
                const int h       = (k + t) % n_holders;  // each thread claims in another order
                void *    stack[] = { addr(0x1000 + h), addr(0x2000) };
                got[t][h]         = table.claim(tag_at(addr(0x1000 + h)), stack, 2);
            }
        });
    }
    for (std::thread & th : threads) {
        th.join();
    }
    for (int h = 0; h < n_holders; ++h) {
        CHECK(got[0][h] != 0);
        for (int t = 1; t < n_threads; ++t) {
            CHECK(got[t][h] == got[0][h]);
        }
        for (int o = 0; o < h; ++o) {
            CHECK(got[0][o] != got[0][h]);
        }
    }
    return 0;
}

int main() {
    struct test_case {
        const char * name;
        int (*fn)();
    };

    const test_case cases[] = {
        { "null_address_is_slot_zero",                  case_null_address_is_slot_zero                  },
        { "frames_start_at_the_recorded_address",       case_frames_start_at_the_recorded_address       },
        { "address_off_the_stack_is_the_address_alone", case_address_off_the_stack_is_the_address_alone },
        { "frames_are_capped",                          case_frames_are_capped                          },
        { "distinct_holders_get_distinct_slots",        case_distinct_holders_get_distinct_slots        },
        { "full_table_answers_slot_zero",               case_full_table_answers_slot_zero               },
        { "threads_agree_on_a_slot",                    case_threads_agree_on_a_slot                    },
    };
    for (const test_case & c : cases) {
        if (c.fn() != 0) {
            std::fprintf(stderr, "case %s failed\n", c.name);
            return 1;
        }
        std::printf("ok   %s\n", c.name);
    }
    std::printf("chunk-lease-holders: %zu cases passed\n", sizeof(cases) / sizeof(cases[0]));
    return 0;
}
