// Host tests of the pending-range primitive (llama.cpp-moua L4; moua design
// 2.3.3 A1-A3): the term-filtered record/clear/retag/query set, the draws
// inside an owner's own ranges, the keyed WEIGHT draw, replace_within_block's block
// core, and the device-wide queries.  The two TLSF placement calls it stands on
// (allocate_at, allocate_excluding) are pinned in tests/test-tlsf-allocator.cpp.
//
// Every arm is built so a plausible wrong implementation fails it: the cases
// that decide a term filter put two terms of ONE owner on the same TLSF, and
// the cases that decide a placement put a foreign range at a LOWER offset than
// the owner's, so a first fit that ignored the geometry would land in it.
//
// The build is -DNDEBUG, so CHECK is explicit and always runs.

// A TLSF_ASSERT that throws, so the assertion arms of the primitive can be tested.
struct tlsf_assert_error {};

#define TLSF_ASSERT(cond)              \
    do {                               \
        if (!(cond)) {                 \
            throw tlsf_assert_error{}; \
        }                              \
    } while (0)

#include "../pending-range.hpp"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(cond, msg)                                                         \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

using namespace ggml_sycl;

namespace {

constexpr size_t K  = 1024;
constexpr size_t MB = 1024 * 1024;

struct misuse_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

[[noreturn]] void throwing_handler(const char * detail) {
    throw misuse_error(detail);
}

// Run `fn`; true when it aborted through the misuse channel.
template <typename Fn> bool misuses(Fn fn) {
    pending_range_misuse_fn saved  = pending_range_misuse_handler();
    pending_range_misuse_handler() = throwing_handler;
    bool aborted                   = false;
    try {
        fn();
    } catch (const misuse_error &) {
        aborted = true;
    }
    pending_range_misuse_handler() = saved;
    return aborted;
}

// Run `fn`; true when it tripped a TLSF_ASSERT.
template <typename Fn> bool asserts(Fn fn) {
    try {
        fn();
    } catch (const tlsf_assert_error &) {
        return true;
    }
    return false;
}

// Run `fn`; the misuse line it aborted with, or "" when it did not abort.
template <typename Fn> std::string misuse_line_of(Fn fn) {
    pending_range_misuse_fn saved  = pending_range_misuse_handler();
    pending_range_misuse_handler() = throwing_handler;
    std::string line;
    try {
        fn();
    } catch (const misuse_error & e) {
        line = e.what();
    }
    pending_range_misuse_handler() = saved;
    return line;
}

pending_owner load_of(uint64_t id) {
    return { pending_owner_kind::LOAD, id };
}

pending_owner model_of(uint64_t id) {
    return { pending_owner_kind::MODEL, id };
}

pending_owner ctx_of(uint64_t id) {
    return { pending_owner_kind::CONTEXT, id };
}

pending_owner dev_of(uint64_t id) {
    return { pending_owner_kind::DEVICE, id };
}

void check_ok(const tlsf_allocator & t, const char * what) {
    if (!t.check_invariants()) {
        std::fprintf(stderr, "FAIL: check_invariants after %s\n", what);
        std::exit(1);
    }
}

size_t count_of(const pending_range_set & s, pending_owner o, pending_term_mask m) {
    return s.ranges(o, m).size();
}

size_t count_of(const pending_range_set & s, pending_owner o, pending_term t) {
    return count_of(s, o, pending_term_bit(t));
}

// ---- A1: record, clear, retag, filter rules ---------------------------------

// A {LOAD, txn} record replaces that owner's range of the same term; the early stage
// records twice and the TLSF holds one.  A context's records append.  A {DEVICE, d}
// VM_TAIL_SURPLUS record replaces; a {DEVICE, d} DEVICE_TERM record appends.
void case_record_replace_or_append() {
    pending_range_set s;
    s.record(load_of(7), pending_term::WEIGHT, 0, 4 * K);
    s.record(load_of(7), pending_term::WEIGHT, 8 * K, 4 * K);
    CHECK(count_of(s, load_of(7), pending_term::WEIGHT | pending_term::SCRATCH) == 1,
          "a LOAD record twice holds one range");
    CHECK(s.ranges(load_of(7), pending_term_bit(pending_term::WEIGHT))[0].offset == 8 * K,
          "the second LOAD record replaced the first");

    // The key is (txn, term): another term and another txn are untouched.
    s.record(load_of(7), pending_term::SCRATCH, 16 * K, 4 * K);
    s.record(load_of(8), pending_term::WEIGHT, 24 * K, 4 * K);
    s.record(load_of(7), pending_term::WEIGHT, 32 * K, 4 * K);
    CHECK(count_of(s, load_of(7), pending_term::SCRATCH) == 1, "a LOAD WEIGHT record leaves its SCRATCH range");
    CHECK(count_of(s, load_of(8), pending_term::WEIGHT) == 1, "a LOAD record leaves another txn's range");
    CHECK(count_of(s, load_of(7), pending_term::WEIGHT) == 1, "still one WEIGHT range for txn 7");

    // A context owner appends: a step-5 placement and a yield retag land on one TLSF, side by side.
    pending_range_set c;
    c.record(ctx_of(3), pending_term::REGION, 0, 4 * K);
    c.record(ctx_of(3), pending_term::REGION, 64 * K, 4 * K);
    CHECK(count_of(c, ctx_of(3), pending_term::REGION) == 2, "a CONTEXT owner's records append");

    pending_range_set d;
    d.record(dev_of(0), pending_term::VM_TAIL_SURPLUS, 0, 4 * K);
    d.record(dev_of(0), pending_term::VM_TAIL_SURPLUS, 8 * K, 4 * K);
    CHECK(count_of(d, dev_of(0), pending_term::VM_TAIL_SURPLUS) == 1, "a DEVICE VM_TAIL_SURPLUS record replaces");
    d.record(dev_of(0), pending_term::DEVICE_TERM, 0, 4 * K);
    d.record(dev_of(0), pending_term::DEVICE_TERM, 8 * K, 4 * K);
    CHECK(count_of(d, dev_of(0), pending_term::DEVICE_TERM) == 2, "a DEVICE DEVICE_TERM record appends");

    pending_range_set z;
    z.record(ctx_of(1), pending_term::REGION, 0, 0);
    CHECK(z.empty(), "a zero-size record is dropped");
}

// A filtered clear drops only the filtered terms, a second clear is a no-op, and a
// context's clear leaves 23mk's hold under the same owner.
void case_filtered_clear() {
    pending_range_set s;
    s.record(ctx_of(5), pending_term::REGION, 0, 4 * K);
    s.record(ctx_of(5), pending_term::ONEDNN_PP_A, 8 * K, 4 * K);
    s.record(ctx_of(6), pending_term::REGION, 16 * K, 4 * K);
    CHECK(s.clear(ctx_of(5), pending_term_bit(pending_term::REGION)) == 1, "the filtered clear drops the REGION range");
    CHECK(count_of(s, ctx_of(5), pending_term::ONEDNN_PP_A) == 1, "the same context's other-term hold stays whole");
    CHECK(count_of(s, ctx_of(6), pending_term::REGION) == 1, "another owner's range stays");
    CHECK(s.clear(ctx_of(5), pending_term_bit(pending_term::REGION)) == 0, "a second clear is a no-op");
    CHECK(s.clear(ctx_of(5), pending_term_bit(pending_term::REGION)) == 0, "and a third");
}

// The misuse rules: PENDING_TERM_ALL only under a {LOAD, txn} owner, an empty filter
// never, owner kind NONE never.
void case_filter_misuse() {
    pending_range_set s;
    s.record(ctx_of(1), pending_term::REGION, 0, 4 * K);
    s.record(load_of(1), pending_term::WEIGHT, 8 * K, 4 * K);
    CHECK(misuses([&] { s.clear(ctx_of(1), PENDING_TERM_ALL); }), "an unfiltered CONTEXT clear is a defect");
    CHECK(misuses([&] { s.clear(model_of(1), PENDING_TERM_ALL); }), "an unfiltered MODEL clear is a defect");
    CHECK(misuses([&] { s.clear(dev_of(1), PENDING_TERM_ALL); }), "an unfiltered DEVICE clear is a defect");
    CHECK(misuses([&] { s.retag(ctx_of(1), PENDING_TERM_ALL, model_of(1)); }),
          "an unfiltered CONTEXT retag is a defect");
    CHECK(misuses([&] { s.ranges(ctx_of(1), PENDING_TERM_ALL); }), "an unfiltered CONTEXT fit read is a defect");
    CHECK(misuses([&] { s.clear(ctx_of(1), 0); }), "an empty filter is a defect");
    CHECK(misuses([&] { s.ranges(load_of(1), 0); }), "an empty fit filter is a defect");
    CHECK(misuses([&] { s.clear(pending_owner{}, pending_term_bit(pending_term::WEIGHT)); }), "owner NONE is a defect");
    CHECK(misuses([&] { s.record(pending_owner{}, pending_term::WEIGHT, 0, 4 * K); }),
          "recording for owner NONE is a defect");
    CHECK(!misuses([&] { s.clear(load_of(1), PENDING_TERM_ALL); }), "PENDING_TERM_ALL is legal under a LOAD owner");
    CHECK(count_of(s, ctx_of(1), pending_term_bit(pending_term::REGION)) == 1, "no refused call changed a range");
}

// A4 retag then clear.  Load B holds a drawn WEIGHT range, an undrawn one, a SCRATCH
// range and a MODEL_TERM range.  The commit retags WEIGHT | MODEL_TERM | FIRST_CONTEXT
// to {MODEL, B}, then clears {LOAD, B} wholesale.
void case_retag_then_clear() {
    pending_range_set   s;
    tlsf_allocator      t(64 * MB);
    const pending_owner b = load_of(9);
    s.record(b, pending_term::WEIGHT, 0, 8 * MB);
    s.record(b, pending_term::SCRATCH, 16 * MB, 4 * MB);
    s.record(b, pending_term::MODEL_TERM, 24 * MB, 2 * MB);
    s.record(b, pending_term::FIRST_CONTEXT, 32 * MB, 2 * MB);
    // The drawn part: a keyed draw consumes the front of the WEIGHT range.
    const pending_within_result drawn = s.allocate_within_keyed(t, b, { 0, 3 * MB }, 3 * MB, 0);
    CHECK(drawn.ok() && drawn.offset == 0, "the drawn part carves the front of the WEIGHT range");
    s.record(b, pending_term::ARENA, 40 * MB, MB);  // a second range of another term, never retagged

    const pending_term_mask moves = pending_term::WEIGHT | pending_term::MODEL_TERM | pending_term::FIRST_CONTEXT;
    CHECK(s.retag(b, moves, model_of(9)) == 3, "the commit retags the three durable terms");
    CHECK(s.clear(b, PENDING_TERM_ALL) == 2, "the load's wholesale clear takes the rest");
    CHECK(count_of(s, b, pending_term_bit(pending_term::SCRATCH)) == 0, "the SCRATCH range is gone");
    CHECK(s.free_bytes(t, model_of(9), moves) == 5 * MB + 2 * MB + 2 * MB,
          "the drawn range's remainder, MODEL_TERM and FIRST_CONTEXT survive whole as {MODEL, B}");
    CHECK(s.all().size() == 3, "nothing is left under {LOAD, B}");
    // A lazy draw under the model owner lands inside the undrawn remainder.
    const pending_within_result lazy = s.allocate_within(t, model_of(9), pending_term::WEIGHT, MB, 256, 0, true);
    CHECK(lazy.ok() && lazy.offset == 3 * MB, "a lazy draw lands in the remainder of the WEIGHT range");
    // The unload clear names WEIGHT | FIRST_CONTEXT and leaves MODEL_TERM for 23mk's clear.
    s.clear(model_of(9), pending_term::WEIGHT | pending_term::FIRST_CONTEXT);
    CHECK(count_of(s, model_of(9), pending_term_bit(pending_term::MODEL_TERM)) == 1,
          "MODEL_TERM survives the unload clear");
    check_ok(t, "retag then clear");
}

// ---- A2: draws inside the owner's ranges ------------------------------------

// consume == false leaves the record whole so the owner can redraw after a free;
// consume == true trims the carved part.
void case_allocate_within_consume() {
    pending_range_set   s;
    tlsf_allocator      t(16 * MB);
    const pending_owner o = ctx_of(1);
    s.record(o, pending_term::ONEDNN_GRAPH_SCRATCH, 4 * MB, 4 * MB);

    pending_within_result a = s.allocate_within(t, o, pending_term::ONEDNN_GRAPH_SCRATCH, MB, 256, 0, false);
    CHECK(a.ok() && a.offset == 4 * MB, "the first draw takes the lowest free part of the range");
    CHECK(s.ranges(o, pending_term_bit(pending_term::ONEDNN_GRAPH_SCRATCH))[0].size == 4 * MB,
          "consume == false leaves the record untrimmed");
    pending_within_result b = s.allocate_within(t, o, pending_term::ONEDNN_GRAPH_SCRATCH, MB, 256, 0, false);
    CHECK(b.ok() && b.offset == 5 * MB, "the second draw takes the next part");
    t.free(a.offset);
    pending_within_result c = s.allocate_within(t, o, pending_term::ONEDNN_GRAPH_SCRATCH, MB, 256, 0, false);
    CHECK(c.ok() && c.offset == 4 * MB, "after a free inside an unconsumed range the owner redraws the same bytes");
    check_ok(t, "consume == false");

    pending_range_set   w;
    tlsf_allocator      t2(16 * MB);
    const pending_owner m = model_of(2);
    w.record(m, pending_term::WEIGHT, 4 * MB, 4 * MB);
    pending_within_result d = w.allocate_within(t2, m, pending_term::WEIGHT, MB, 256, 0, true);
    CHECK(d.ok() && d.offset == 4 * MB, "a weight draw takes the front");
    const std::vector<pending_range> left = w.ranges(m, pending_term_bit(pending_term::WEIGHT));
    CHECK(left.size() == 1 && left[0].offset == 5 * MB && left[0].size == 3 * MB,
          "consume == true trims the carved part");
    t2.free(d.offset);
    pending_within_result e = w.allocate_within(t2, m, pending_term::WEIGHT, MB, 256, 0, true);
    CHECK(e.ok() && e.offset == 5 * MB, "a consumed part is gone from the record even after the free");
    check_ok(t2, "consume == true");
}

// A miss is reported, not fallen through: a draw bigger than any free part of the
// owner's ranges returns MISS and allocates nothing, even though the TLSF has room.
void case_allocate_within_miss() {
    pending_range_set   s;
    tlsf_allocator      t(16 * MB);
    const pending_owner o = ctx_of(1);
    s.record(o, pending_term::REGION, 4 * MB, 2 * MB);
    const size_t                used_before = t.used();
    const pending_within_result miss        = s.allocate_within(t, o, pending_term::REGION, 3 * MB, 256, 0, true);
    CHECK(miss.status == pending_within_status::MISS && miss.offset == SIZE_MAX, "a draw larger than the range misses");
    CHECK(t.used() == used_before, "a miss allocates nothing outside the ranges");
    // The range is partly occupied by a foreign block: the free part is smaller than the draw.
    const size_t block = t.allocate_at(4 * MB, MB, 9);
    CHECK(block == 4 * MB, "an unrelated block takes the front of the range");
    CHECK(s.allocate_within(t, o, pending_term::REGION, 2 * MB, 256, 0, true).status == pending_within_status::MISS,
          "only the free part counts");
    CHECK(s.allocate_within(t, o, pending_term::REGION, MB, 256, 0, true).offset == 5 * MB,
          "the free part serves what fits");
    check_ok(t, "miss");
}

// The term decides the range: a WEIGHT draw never lands in a SCRATCH range of the
// same owner, even when the SCRATCH range is at the LOWER offset and the WEIGHT
// range is too small.
void case_allocate_within_term_isolation() {
    pending_range_set   s;
    tlsf_allocator      t(16 * MB);
    const pending_owner o = load_of(4);
    s.record(o, pending_term::SCRATCH, 0, 4 * MB);
    s.record(o, pending_term::WEIGHT, 8 * MB, MB);
    CHECK(s.allocate_within(t, o, pending_term::WEIGHT, 2 * MB, 256, 0, true).status == pending_within_status::MISS,
          "a WEIGHT draw that only the SCRATCH range could hold misses");
    CHECK(s.allocate_within(t, o, pending_term::WEIGHT, MB, 256, 0, true).offset == 8 * MB,
          "and lands in its own range");
    CHECK(s.allocate_within(t, o, pending_term::SCRATCH, MB, 256, 0, true).offset == 0,
          "a SCRATCH draw lands in the hold");
    CHECK(
        s.allocate_within(t, load_of(5), pending_term::WEIGHT, 256, 256, 0, true).status == pending_within_status::MISS,
        "another owner has no range to draw in");
    check_ok(t, "term isolation");
}

// ---- the keyed WEIGHT draw ---------------------------------------------------

void case_keyed_draw() {
    pending_range_set   s;
    tlsf_allocator      t(16 * MB);
    const pending_owner o = load_of(1);
    s.record(o, pending_term::WEIGHT, 2 * MB, 8 * MB);
    // Draws arrive in the reverse of plan order and still land on their keys.
    const pending_within_result hi = s.allocate_within_keyed(t, o, { 6 * MB, 2 * MB }, 2 * MB, 7);
    const pending_within_result lo = s.allocate_within_keyed(t, o, { 2 * MB, MB }, MB, 7);
    CHECK(hi.ok() && hi.offset == 6 * MB, "a keyed draw takes exactly its offset");
    CHECK(lo.ok() && lo.offset == 2 * MB, "and the order the draws arrive in cannot move an item");
    check_ok(t, "keyed draws");
    // A second draw of a drawn key finds it consumed, so it is outside the record.
    CHECK(s.allocate_within_keyed(t, o, { 2 * MB, MB }, MB, 7).status == pending_within_status::KEY_OUTSIDE_RANGE,
          "a drawn key is consumed");
    // A key whose bytes another block holds inside a still-recorded range.
    CHECK(t.allocate_at(8 * MB, MB, 9) == 8 * MB, "a block that bypassed the owner takes part of its range");
    CHECK(s.allocate_within_keyed(t, o, { 8 * MB, MB }, MB, 7).status == pending_within_status::KEY_OCCUPIED,
          "an occupied key");
    CHECK(s.allocate_within_keyed(t, o, { 3 * MB, MB }, 2 * MB, 7).status == pending_within_status::KEY_SIZE_MISMATCH,
          "a size that differs from the key's");
    CHECK(s.allocate_within_keyed(t, o, { 12 * MB, MB }, MB, 7).status == pending_within_status::KEY_OUTSIDE_RANGE,
          "a key outside the owner's range");
    CHECK(s.allocate_within_keyed(t, load_of(2), { 3 * MB, MB }, MB, 7).status ==
              pending_within_status::KEY_OUTSIDE_RANGE,
          "another owner's key");
    // A key in the owner's SCRATCH range is outside its WEIGHT ranges.
    s.record(o, pending_term::SCRATCH, 12 * MB, 2 * MB);
    CHECK(s.allocate_within_keyed(t, o, { 12 * MB, MB }, MB, 7).status == pending_within_status::KEY_OUTSIDE_RANGE,
          "a SCRATCH range is not a WEIGHT key's room");
    // The refusals changed nothing: the free part of the WEIGHT range is what the two draws left.
    CHECK(s.free_bytes(t, o, pending_term_bit(pending_term::WEIGHT)) == 3 * MB + MB, "refusals consume nothing");
    CHECK(count_of(s, o, pending_term::WEIGHT) == 2, "and trim nothing");
}

// Adjacent records of one owner and term are not coalesced.  A first-fit draw lies inside ONE
// record, so it misses a size that only the two together hold, while free_bytes() counts both; the
// keyed form names its bytes and reads the union of the owner's ranges.
void case_adjacent_ranges() {
    pending_range_set   u;
    tlsf_allocator      t(16 * MB);
    const pending_owner m = model_of(1);
    u.record(m, pending_term::WEIGHT, 0, 2 * MB);
    u.record(m, pending_term::WEIGHT, 2 * MB, 2 * MB);
    CHECK(u.free_bytes(t, m, pending_term_bit(pending_term::WEIGHT)) == 4 * MB, "free_bytes counts both records");
    CHECK(u.allocate_within(t, m, pending_term::WEIGHT, 3 * MB, 256, 0, false).status == pending_within_status::MISS,
          "a first-fit draw does not span two records");
    CHECK(t.used() == 0, "and the miss carved nothing");
    const pending_within_result k = u.allocate_within_keyed(t, m, { MB, 2 * MB }, 2 * MB, 7);
    CHECK(k.ok() && k.offset == MB, "a keyed draw straddling the two records is covered by their union");
    CHECK(u.free_bytes(t, m, pending_term_bit(pending_term::WEIGHT)) == 2 * MB, "and consumes the bytes from both");
    check_ok(t, "adjacent ranges");
}

// The carve rounds the key to the grain, so the range test must too (I1): a 300-byte key in a
// 300-byte range would carve 512 bytes and swallow the next owner's room.
void case_keyed_rounding_and_refusals() {
    pending_range_set   s;
    tlsf_allocator      t(16 * MB);
    const pending_owner o = load_of(1);
    s.record(o, pending_term::WEIGHT, 0, 300);
    s.record(ctx_of(2), pending_term::REGION, 300, 212);
    CHECK(s.allocate_within_keyed(t, o, { 0, 300 }, 300, 7).status == pending_within_status::KEY_OUTSIDE_RANGE,
          "a key whose rounded extent leaves the owner's range is refused");
    CHECK(t.used() == 0, "the refusal carved nothing");
    CHECK(count_of(s, ctx_of(2), pending_term::REGION) == 1 && s.all()[1].size == 212, "the foreign range is whole");
    s.record(o, pending_term::WEIGHT, 0, 512);
    const pending_within_result fit = s.allocate_within_keyed(t, o, { 0, 300 }, 300, 7);
    CHECK(fit.ok() && fit.offset == 0 && t.block_size_at(0) == 512, "a range that holds the rounded extent serves it");
    check_ok(t, "keyed rounding");

    CHECK(s.allocate_within_keyed(t, o, { 2 * K, 0 }, 0, 7).status == pending_within_status::ZERO_SIZE,
          "a zero-size key is its own result");
    pending_range_set g;
    g.record(model_of(3), pending_term::WEIGHT, 0, K);
    CHECK(g.allocate_within_keyed(t, model_of(3), { 100, 256 }, 256, 7).status == pending_within_status::KEY_OFF_GRAIN,
          "an off-grain key offset is its own result, not KEY_OCCUPIED");
    CHECK(g.allocate_within_keyed(t, model_of(3), { 0, SIZE_MAX }, SIZE_MAX, 7).status ==
              pending_within_status::SIZE_OVERFLOW,
          "a key whose rounding wraps is its own result");
    CHECK(g.allocate_within_keyed(t, model_of(3), { 256, SIZE_MAX - 300 }, SIZE_MAX - 300, 7).status ==
              pending_within_status::SIZE_OVERFLOW,
          "a key whose end wraps is its own result");
    CHECK(count_of(g, model_of(3), pending_term::WEIGHT) == 1, "none of them trimmed the range");

    // covered_by is byte for byte: a gap smaller than the grain between two ranges is a gap.
    pending_range_set   h;
    tlsf_allocator      t2(16 * K);
    const pending_owner m = model_of(4);
    h.record(m, pending_term::WEIGHT, 0, 1000);
    h.record(m, pending_term::WEIGHT, 1100, 948);
    CHECK(h.allocate_within_keyed(t2, m, { 0, 2048 }, 2048, 7).status == pending_within_status::KEY_OUTSIDE_RANGE,
          "a 100-byte gap between two ranges is not covered");
    CHECK(t2.used() == 0, "and nothing was carved");
    check_ok(t2, "gap");
}

// A draw of nothing and a draw that wraps are not plan misses (M5), and an alignment above the
// grain is a defect rather than a silent clamp (M1).
void case_allocate_within_bad_requests() {
    pending_range_set   s;
    tlsf_allocator      t(16 * MB);
    const pending_owner o = ctx_of(1);
    s.record(o, pending_term::REGION, 0, 4 * MB);
    CHECK(s.allocate_within(t, o, pending_term::REGION, 0, 256, 0, true).status == pending_within_status::ZERO_SIZE,
          "a zero-size draw");
    CHECK(s.allocate_within(t, o, pending_term::REGION, SIZE_MAX, 256, 0, true).status ==
              pending_within_status::SIZE_OVERFLOW,
          "a size that wraps");
    CHECK(s.allocate_within(t, o, pending_term::REGION, 5 * MB, 256, 0, true).status == pending_within_status::MISS,
          "a plain miss is still a miss");
    CHECK(asserts([&] { s.allocate_within(t, o, pending_term::REGION, MB, 4096, 0, true); }),
          "an alignment above the grain asserts");
    CHECK(t.used() == 0 && count_of(s, o, pending_term::REGION) == 1, "none of them drew or trimmed");
}

// The misuse line names the operation that aborted, so the abort line alone locates it (M4);
// the accounting reads apply the same filter rules as the fit reads (M2).
void case_misuse_names_the_operation() {
    pending_range_set s;
    tlsf_allocator    t(16 * MB);
    s.record(ctx_of(1), pending_term::REGION, 0, 4 * K);
    CHECK(misuse_line_of([&] { s.clear(pending_owner{}, 1); }).find("clear") == 0, "clear names itself");
    CHECK(misuse_line_of([&] { s.ranges(ctx_of(1), 0); }).find("ranges") == 0, "ranges names itself");
    CHECK(misuse_line_of([&] { s.retag(ctx_of(1), 1, pending_owner{}); }).find("retag target") == 0,
          "a NONE retag target names itself");
    CHECK(misuse_line_of([&] {
              s.allocate_within(t, pending_owner{}, pending_term::REGION, K, 256, 0, true);
          }).find("allocate_within") == 0,
          "allocate_within names itself");
    CHECK(misuse_line_of([&] {
              s.allocate_within_keyed(t, pending_owner{}, { 0, K }, K, 0);
          }).find("allocate_within_keyed") == 0,
          "the keyed draw names itself");
    CHECK(misuse_line_of([&] {
              s.replace_within_block(t, pending_owner{}, pending_term::WEIGHT, 0, 0, K, 0, 0);
          }).find("replace_within_block") == 0,
          "replace_within_block names itself");
    CHECK(misuse_line_of([&] { s.record(ctx_of(1), pending_term::REGION, SIZE_MAX - 10, 100); }).find("record") == 0,
          "a wrapping record aborts and names itself");
    CHECK(count_of(s, ctx_of(1), pending_term::REGION) == 1, "the wrapping record left the set alone");

    CHECK(misuses([&] { s.free_bytes(t, ctx_of(1), 0); }), "free_bytes with an empty filter is a defect");
    CHECK(misuses([&] { s.free_bytes(t, ctx_of(1), PENDING_TERM_ALL); }),
          "free_bytes with PENDING_TERM_ALL under a CONTEXT owner is a defect");
    CHECK(misuses([&] { s.free_bytes(t, pending_owner{}, 1); }), "free_bytes for owner NONE is a defect");
    CHECK(!misuses([&] { s.free_bytes(t, load_of(1), PENDING_TERM_ALL); }),
          "free_bytes under a LOAD owner may name ALL");
    const std::vector<pending_range_member> none;
    CHECK(misuses([&] { pending_bytes(none, ctx_of(1), 0); }), "pending_bytes with an empty filter is a defect");
    CHECK(misuses([&] { pending_bytes(none, ctx_of(1), PENDING_TERM_ALL); }),
          "pending_bytes with PENDING_TERM_ALL under a CONTEXT owner is a defect, even over no members");
    CHECK(misuses([&] { pending_bytes(none, pending_owner{}, 1); }), "pending_bytes for owner NONE is a defect");
    CHECK(pending_bytes(none, load_of(1), PENDING_TERM_ALL) == 0, "and a LOAD owner may name ALL");
}

// retag and ranges accept PENDING_TERM_ALL under a LOAD owner, like clear; a retag target of owner
// NONE is refused.
void case_all_under_load_and_retag_target() {
    pending_range_set s;
    s.record(load_of(1), pending_term::WEIGHT, 0, 4 * K);
    s.record(load_of(1), pending_term::SCRATCH, 8 * K, 4 * K);
    CHECK(!misuses([&] { s.ranges(load_of(1), PENDING_TERM_ALL); }), "ranges may name ALL under a LOAD owner");
    CHECK(s.ranges(load_of(1), PENDING_TERM_ALL).size() == 2, "and returns every term");
    CHECK(misuses([&] { s.retag(load_of(1), pending_term_bit(pending_term::WEIGHT), pending_owner{}); }),
          "a retag to owner NONE is a defect");
    CHECK(count_of(s, load_of(1), pending_term::WEIGHT) == 1, "and moved nothing");
    CHECK(!misuses([&] { s.retag(load_of(1), PENDING_TERM_ALL, model_of(1)); }),
          "retag may name ALL under a LOAD owner");
    CHECK(s.ranges(model_of(1), pending_term::WEIGHT | pending_term::SCRATCH).size() == 2,
          "retag with ALL moved both terms");
}

// allocate_excluding's caller contract (M7) and its zero-size excluded range.
void case_excluding_caller_contract() {
    tlsf_allocator t(16 * MB);
    using R = tlsf_allocator::excluded_range;
    CHECK(asserts([&] {
              t.allocate_excluding(
                  {
                      R{ 4096, SIZE_MAX }
              },
                  K, 256, 1);
          }),
          "an excluded range whose end wraps asserts");
    CHECK(t.used() == 0, "and allocated nothing");
    // The wrap is diagnosed before the request is looked at (r2 m3): a zero or overflowing size still asserts.
    CHECK(asserts([&] {
              t.allocate_excluding(
                  {
                      R{ 4096, SIZE_MAX }
              },
                  0, 256, 1);
          }),
          "a wrapping excluded range asserts for a zero-size request too");
    CHECK(asserts([&] {
              t.allocate_excluding(
                  {
                      R{ 4096, SIZE_MAX }
              },
                  SIZE_MAX, 256, 1);
          }),
          "and for a request whose rounding wraps");
    CHECK(t.allocate_excluding(
              {
                  R{ 256, 0 }
    },
              1024, 256, 1) == 0,
          "a zero-size excluded range excludes nothing");
    CHECK(t.allocate_excluding(
              {
                  R{ SIZE_MAX - 100, 100 }
    },
              K, 256, 1) == K,
          "a range ending at the top of the address space excludes only itself");
    check_ok(t, "excluding contract");
}

// The pieces of an off-grain region (I2): a tail block that absorbed a sub-grain remainder is
// restored at its recorded extent, and a replace whose rest is that tail carves it exactly.
void case_replace_off_grain_tail() {
    {
        // The carve fails (the new range reaches past the region end) and old comes back whole.
        pending_range_set   s;
        tlsf_allocator      t(1000);
        const pending_owner o   = model_of(1);
        const size_t        old = t.allocate_at(0, 768, 3);
        CHECK(old == 0 && t.block_size_at(0) == 1000, "old is a tail block that absorbed the 232-byte remainder");
        s.record(o, pending_term::WEIGHT, 0, 1024);
        const pending_replace_result r = s.replace_within_block(t, o, pending_term::WEIGHT, 0, 256, 768, 4, 6);
        CHECK(r.status == pending_replace_status::CARVE_FAILED, "the carve past the region end fails without aborting");
        CHECK(t.block_size_at(0) == 1000 && t.tag_at(0) == 3 && t.used() == 1000, "old is back at its exact extent");
        CHECK(s.all().size() == 1 && s.all()[0].offset == 0 && s.all()[0].size == 1024, "the record is untouched");
        check_ok(t, "off-grain restore");
    }
    {
        // A new range whose rounded end wraps the address space is refused before anything is released.
        pending_range_set   s;
        tlsf_allocator      t(16 * MB);
        const pending_owner o   = ctx_of(1);
        const size_t        old = t.allocate_at(4 * MB, MB, 3);
        s.record(o, pending_term::ONEDNN_PP_A, 4 * MB, MB);
        const size_t top = SIZE_MAX - 255;  // on the grain
        CHECK(s.replace_within_block(t, o, pending_term::ONEDNN_PP_A, old, top, 512, 4, 6).status ==
                  pending_replace_status::SIZE_OVERFLOW,
              "a new range whose end wraps the address space is named, not a range miss");
        CHECK(t.block_size_at(4 * MB) == MB && t.tag_at(4 * MB) == 3, "and old was never released");
    }
    {
        // Another term: the rest is the 488-byte tail, carved as one remainder block of its exact extent.
        pending_range_set   s;
        tlsf_allocator      t(1000);
        const pending_owner o   = ctx_of(1);
        const size_t        old = t.allocate_at(0, 768, 3);
        s.record(o, pending_term::ONEDNN_PP_A, 0, 1000);
        const pending_replace_result r = s.replace_within_block(t, o, pending_term::ONEDNN_PP_A, old, 0, 512, 4, 6);
        CHECK(r.ok() && r.new_offset == 0 && t.block_size_at(0) == 512, "the new block is the front 512 bytes");
        CHECK(r.remainder_offset == 512 && r.remainder_size == 488 && t.block_size_at(512) == 488 && t.tag_at(512) == 6,
              "the rest is one remainder block of the tail's exact extent");
        check_ok(t, "off-grain rest");
    }
    {
        // The new carve absorbs the 232-byte tail (r2 I1): the rest is empty, not a carve into the allocated
        // block.  Another term: no remainder, and the block is the whole 1000 bytes.
        pending_range_set   s;
        tlsf_allocator      t(1000);
        const pending_owner o   = ctx_of(1);
        const size_t        old = t.allocate_at(0, 768, 3);
        s.record(o, pending_term::ONEDNN_PP_A, 0, 1000);
        pending_replace_result r;
        CHECK(!asserts([&] { r = s.replace_within_block(t, o, pending_term::ONEDNN_PP_A, old, 0, 768, 4, 6); }),
              "a new range that absorbs the old tail does not abort");
        CHECK(r.ok() && r.new_offset == 0 && t.block_size_at(0) == 1000 && t.tag_at(0) == 4,
              "the new block is the whole 1000 bytes");
        CHECK(r.remainder_offset == SIZE_MAX && r.remainder_size == 0, "and there is no remainder block");
        CHECK(s.all().empty(), "and no range of the term is left");
        check_ok(t, "absorbed tail");
    }
    {
        // The same input under WEIGHT: no stale fragment is recorded over the allocated block.
        pending_range_set   s;
        tlsf_allocator      t(1000);
        const pending_owner o   = model_of(1);
        const size_t        old = t.allocate_at(0, 768, 3);
        s.record(o, pending_term::WEIGHT, 0, 1000);
        const pending_replace_result r = s.replace_within_block(t, o, pending_term::WEIGHT, old, 0, 768, 4, 6);
        CHECK(r.ok() && t.block_size_at(0) == 1000, "the new block is the whole 1000 bytes");
        CHECK(s.all().empty(), "no fragment is recorded over the allocated block");
        check_ok(t, "absorbed tail, WEIGHT");
    }
    {
        // The front remainder case: new {256,512} absorbs the tail, the rest is the 256 bytes before it.
        pending_range_set   s;
        tlsf_allocator      t(1000);
        const pending_owner o   = ctx_of(1);
        const size_t        old = t.allocate_at(0, 768, 3);
        s.record(o, pending_term::ONEDNN_PP_A, 0, 1000);
        const pending_replace_result r = s.replace_within_block(t, o, pending_term::ONEDNN_PP_A, old, 256, 512, 4, 6);
        CHECK(r.ok() && r.new_offset == 256 && t.block_size_at(256) == 744,
              "a one-piece rest is not refused for a tail the new block absorbs");
        CHECK(r.remainder_offset == 0 && r.remainder_size == 256 && t.block_size_at(0) == 256 && t.tag_at(0) == 6,
              "the rest is the front 256 bytes");
        check_ok(t, "absorbed tail, front rest");

        pending_range_set   w;
        tlsf_allocator      u(1000);
        const pending_owner m    = model_of(1);
        const size_t        old2 = u.allocate_at(0, 768, 3);
        w.record(m, pending_term::WEIGHT, 0, 1000);
        const pending_replace_result rw = w.replace_within_block(u, m, pending_term::WEIGHT, old2, 256, 512, 4, 6);
        CHECK(rw.ok() && w.all().size() == 1 && w.all()[0].offset == 0 && w.all()[0].size == 256,
              "WEIGHT records only the front fragment");
        check_ok(u, "absorbed tail, front rest, WEIGHT");
    }
    {
        // A zero size and a size whose rounding wraps are named, apart from a range miss (r2 m2).
        pending_range_set   s;
        tlsf_allocator      t(16 * MB);
        const pending_owner o   = ctx_of(1);
        const size_t        old = t.allocate_at(4 * MB, MB, 3);
        s.record(o, pending_term::ONEDNN_PP_A, 4 * MB, MB);
        CHECK(s.replace_within_block(t, o, pending_term::ONEDNN_PP_A, old, 4 * MB, 0, 4, 6).status ==
                  pending_replace_status::ZERO_SIZE,
              "a zero size is named");
        CHECK(s.replace_within_block(t, o, pending_term::ONEDNN_PP_A, old, 4 * MB, SIZE_MAX, 4, 6).status ==
                  pending_replace_status::SIZE_OVERFLOW,
              "a size whose rounding wraps is named");
        CHECK(t.block_size_at(4 * MB) == MB && t.tag_at(4 * MB) == 3, "and old was never released");
    }
}

// ---- other allocators honour the ranges --------------------------------------

// allocate_excluding(as_excluded()) never lands in a pending range, a range at a
// LOWER offset than the free space the request would otherwise take included.
void case_others_honour_ranges() {
    pending_range_set s;
    tlsf_allocator    t(16 * MB);
    s.record(load_of(1), pending_term::WEIGHT, 0, 4 * MB);
    s.record(ctx_of(2), pending_term::REGION, 6 * MB, 2 * MB);
    const size_t a = t.allocate_excluding(s.as_excluded(), MB, 256, 1);
    CHECK(a == 4 * MB, "the first fit skips the range at the lowest offset");
    const size_t b = t.allocate_excluding(s.as_excluded(), 2 * MB, 256, 1);
    CHECK(b == 8 * MB, "and the gap between the ranges is too small for a 2 MB block above it");
    const size_t c = t.allocate_excluding(s.as_excluded(), MB, 256, 1);
    CHECK(c == 5 * MB, "a request that fits between two ranges goes there");
    CHECK(t.allocate_excluding(s.as_excluded(), 7 * MB, 256, 1) == SIZE_MAX,
          "a request that fits only across a range misses");
    check_ok(t, "others honour ranges");
}

// ---- the device-wide queries -------------------------------------------------

void case_device_wide_queries() {
    // RUNTIME (zone 1): {CONTEXT,c}/REGION, {CONTEXT,c}/ONEDNN_PP_A, {LOAD,B}/SCRATCH, {MODEL,M}/MODEL_TERM.
    // Another zone (2): {LOAD,B}/WEIGHT.
    tlsf_allocator      runtime(32 * MB), weight(32 * MB), second(32 * MB);
    pending_range_set   rs, ws, ss;
    const pending_owner c = ctx_of(1), bl = load_of(2), m = model_of(3);
    rs.record(c, pending_term::REGION, 0, MB);
    rs.record(c, pending_term::ONEDNN_PP_A, 4 * MB, 2 * MB);
    rs.record(bl, pending_term::SCRATCH, 8 * MB, 4 * MB);
    rs.record(m, pending_term::MODEL_TERM, 16 * MB, 8 * MB);
    ws.record(bl, pending_term::WEIGHT, 0, 16 * MB);
    const std::vector<pending_range_member> all = {
        { 1, &runtime, &rs },
        { 2, &weight,  &ws },
        { 1, &second,  &ss }
    };

    CHECK(pending_bytes_excluding(all, 1, c, pending_term::ONEDNN_PP_A) == MB + 4 * MB + 8 * MB,
          "the other three RUNTIME ranges, and not the excluded (owner, term)");
    CHECK(pending_bytes_excluding(all, -1, c, pending_term::ONEDNN_PP_A) == MB + 4 * MB + 8 * MB + 16 * MB,
          "the every-zone form adds the WEIGHT range");
    CHECK(pending_bytes_excluding(all, 1, pending_owner{}, pending_term::ONEDNN_PP_A) == MB + 2 * MB + 4 * MB + 8 * MB,
          "owner NONE matches no range, so every range counts");
    // The exclusion is one (owner, term): excluding by owner alone would drop the REGION range too.
    CHECK(pending_bytes_excluding(all, 1, c, pending_term::REGION) == 2 * MB + 4 * MB + 8 * MB,
          "excluding REGION keeps the same owner's ONEDNN_PP_A hold");
    // The caller's own owner's pending_bytes misses the other owners.
    CHECK(pending_bytes(all, c, pending_term::REGION | pending_term::ONEDNN_PP_A) == 3 * MB,
          "pending_bytes is one owner's");
    CHECK(pending_bytes(all, bl, PENDING_TERM_ALL) == 4 * MB + 16 * MB, "a load's wholesale figure spans zones");
    // Occupied bytes are not free bytes.
    CHECK(runtime.allocate_at(0, 512 * K, 1) == 0, "a block takes half of the REGION range");
    CHECK(pending_bytes(all, c, pending_term_bit(pending_term::REGION)) == 512 * K, "free bytes, not range bytes");
    // Overlapping ranges count a free byte once.
    rs.record(ctx_of(9), pending_term::REGION, MB / 2, 2 * MB);
    CHECK(pending_bytes_excluding(all, 1, c, pending_term::ONEDNN_PP_A) == (MB + 3 * MB / 2) - MB / 2 + 4 * MB + 8 * MB,
          "two ranges that overlap count a free byte once");
}

// ---- A3: replace_within_block's block core ------------------------------------------

// An empty rest returns no remainder and carves nothing.
void case_replace_empty_rest() {
    pending_range_set s;
    tlsf_allocator    t(16 * MB);
    const size_t      old_off = t.allocate_at(4 * MB, 2 * MB, 3);
    s.record(ctx_of(1), pending_term::ONEDNN_PP_A, 4 * MB, 2 * MB);
    const size_t                 blocks_before = t.block_census().size();
    const pending_replace_result r =
        s.replace_within_block(t, ctx_of(1), pending_term::ONEDNN_PP_A, old_off, 4 * MB, 2 * MB, 5, 6);
    CHECK(r.ok() && r.new_offset == 4 * MB, "the new block takes old's place");
    CHECK(r.remainder_offset == SIZE_MAX && r.remainder_size == 0, "an empty rest returns no remainder");
    CHECK(t.block_census().size() == blocks_before, "and carves no zero-size block");
    CHECK(t.tag_at(4 * MB) == 5, "the new block carries the new tag");
    check_ok(t, "empty rest");
}

// A new range never lands in the owner's ranges of another term, and one outside
// old's block plus the owner's ranges of the term is refused untouched.
void case_replace_term_scope() {
    pending_range_set s;
    tlsf_allocator    t(16 * MB);
    const size_t      old_off = t.allocate_at(4 * MB, MB, 3);
    s.record(ctx_of(1), pending_term::ONEDNN_PP_A, 5 * MB, MB);     // the term's room, just above old
    s.record(ctx_of(1), pending_term::SET_ROWS_STAGE, 6 * MB, MB);  // another term of the same owner, beyond that
    // Growing old into its own term's range is fine.
    pending_replace_result ok =
        s.replace_within_block(t, ctx_of(1), pending_term::ONEDNN_PP_A, old_off, 4 * MB, 2 * MB, 5, 6);
    CHECK(ok.ok() && ok.new_offset == 4 * MB, "old grows into the owner's range of the same term");
    CHECK(count_of(s, ctx_of(1), pending_term_bit(pending_term::ONEDNN_PP_A)) == 0,
          "and the room it took leaves the record");
    check_ok(t, "growth");
    // Growing further into the other term's range is refused, and nothing moves.
    const size_t           used = t.used();
    pending_replace_result bad =
        s.replace_within_block(t, ctx_of(1), pending_term::ONEDNN_PP_A, 4 * MB, 4 * MB, 3 * MB, 5, 6);
    CHECK(bad.status == pending_replace_status::OUTSIDE_RANGE,
          "a new range that reaches another term's range is refused");
    CHECK(t.used() == used && t.block_size_at(4 * MB) == 2 * MB, "a refusal before the release changes nothing");
    CHECK(count_of(s, ctx_of(1), pending_term_bit(pending_term::SET_ROWS_STAGE)) == 1,
          "the other term's range is whole");
    // A range of another owner's is not the owner's room either.
    pending_range_set o;
    tlsf_allocator    t2(16 * MB);
    const size_t      old2 = t2.allocate_at(4 * MB, MB, 3);
    o.record(ctx_of(2), pending_term::ONEDNN_PP_A, 5 * MB, MB);
    CHECK(o.replace_within_block(t2, ctx_of(1), pending_term::ONEDNN_PP_A, old2, 4 * MB, 2 * MB, 5, 6).status ==
              pending_replace_status::OUTSIDE_RANGE,
          "another owner's range is not room for this owner");
    CHECK(o.replace_within_block(t2, ctx_of(1), pending_term::ONEDNN_PP_A, 12 * MB, 12 * MB, MB, 5, 6).status ==
              pending_replace_status::OLD_NOT_FOUND,
          "an offset that is no allocated block");
}

// A failed carve restores old whole: the new range's room is not free once old is
// released because a third block sits in the owner's range.
void case_replace_failed_carve_restores_old() {
    pending_range_set s;
    tlsf_allocator    t(16 * MB);
    const size_t      old_off = t.allocate_at(4 * MB, MB, 3);
    CHECK(t.allocate_at(5 * MB + 256 * K, 256 * K, 8) == 5 * MB + 256 * K, "a third block inside the owner's range");
    s.record(ctx_of(1), pending_term::ONEDNN_PP_A, 5 * MB, MB);
    const auto                   before = t.block_census();
    const pending_replace_result r =
        s.replace_within_block(t, ctx_of(1), pending_term::ONEDNN_PP_A, old_off, 4 * MB, 2 * MB, 5, 6);
    CHECK(r.status == pending_replace_status::CARVE_FAILED, "a carve over an allocated block fails");
    const auto after = t.check_invariants() ? t.block_census() : std::vector<tlsf_allocator::extent>();
    CHECK(after.size() == before.size(), "old's block comes back as it was");
    for (size_t i = 0; i < before.size(); ++i) {
        CHECK(before[i].offset == after[i].offset && before[i].size == after[i].size &&
                  before[i].free == after[i].free && before[i].tag == after[i].tag,
              "every block of the census, tag included, is as before");
    }
    CHECK(count_of(s, ctx_of(1), pending_term_bit(pending_term::ONEDNN_PP_A)) == 1 && s.all()[0].size == MB,
          "the owner's range is untrimmed");
}

// Remainder mode: a WEIGHT re-draw inside a larger old leaves BOTH fragments recorded
// {owner, WEIGHT} and free; another owner's allocation of their size misses them, and
// the owner's allocate_within lands in one.
void case_replace_remainder_mode() {
    pending_range_set            s;
    tlsf_allocator               t(32 * MB);
    const pending_owner          o       = model_of(1);
    const size_t                 old_off = t.allocate_at(8 * MB, 8 * MB, 3);
    const pending_replace_result r = s.replace_within_block(t, o, pending_term::WEIGHT, old_off, 10 * MB, 2 * MB, 4, 6);
    CHECK(r.ok() && r.new_offset == 10 * MB, "the re-draw lands inside old");
    CHECK(r.remainder_offset == SIZE_MAX && r.remainder_size == 0, "remainder mode returns no remainder block");
    const std::vector<pending_range> rest = s.ranges(o, pending_term_bit(pending_term::WEIGHT));
    CHECK(rest.size() == 2, "both fragments are recorded");
    CHECK(rest[0].offset == 8 * MB && rest[0].size == 2 * MB && rest[1].offset == 12 * MB && rest[1].size == 4 * MB,
          "the fragments are exactly before and after the new block, untrimmed");
    CHECK(s.free_bytes(t, o, pending_term_bit(pending_term::WEIGHT)) == 6 * MB, "and they are free bytes");
    // Another owner's allocation of a fragment's size cannot take either.
    const size_t foreign = t.allocate_excluding(s.as_excluded(), 2 * MB, 256, 9);
    CHECK(foreign != SIZE_MAX && (foreign + 2 * MB <= 8 * MB || foreign >= 16 * MB),
          "another owner's allocation lands outside both fragments");
    // The owner redraws inside one fragment.
    const pending_within_result again = s.allocate_within(t, o, pending_term::WEIGHT, 2 * MB, 256, 4, true);
    CHECK(again.ok() && again.offset == 8 * MB, "the owner's allocate_within lands in a fragment");
    check_ok(t, "remainder mode");

    // The owner's record already covers old (an untrimmed record): the fragments replace it, they do not
    // sit beside it, so a fragment is not recorded twice.
    pending_range_set c;
    tlsf_allocator    t2(32 * MB);
    const size_t      old2 = t2.allocate_at(8 * MB, 8 * MB, 3);
    c.record(o, pending_term::WEIGHT, 8 * MB, 8 * MB);
    CHECK(c.replace_within_block(t2, o, pending_term::WEIGHT, old2, 10 * MB, 2 * MB, 4, 6).ok(),
          "the re-draw inside a recorded old");
    const std::vector<pending_range> both = c.ranges(o, pending_term_bit(pending_term::WEIGHT));
    CHECK(both.size() == 2 && c.free_bytes(t2, o, pending_term_bit(pending_term::WEIGHT)) == 6 * MB,
          "each fragment is recorded once");
    check_ok(t2, "remainder mode over a record");
}

// Another term's rest is ONE block owned by the remainder handle; two pieces are refused.
void case_replace_other_term_rest() {
    pending_range_set s;
    tlsf_allocator    t(16 * MB);
    const size_t      old_off = t.allocate_at(4 * MB, 4 * MB, 3);
    s.record(ctx_of(1), pending_term::ONEDNN_PP_A, 4 * MB, 4 * MB);
    const pending_replace_result r =
        s.replace_within_block(t, ctx_of(1), pending_term::ONEDNN_PP_A, old_off, 4 * MB, MB, 5, 6);
    CHECK(r.ok() && r.new_offset == 4 * MB, "the new block takes old's front");
    CHECK(r.remainder_offset == 5 * MB && r.remainder_size == 3 * MB, "the rest is one remainder block above it");
    CHECK(t.tag_at(5 * MB) == 6 && t.block_size_at(5 * MB) == 3 * MB,
          "carved as an allocated block of the remainder tag");
    check_ok(t, "other term rest");
    // In the middle, the rest is two pieces: refused untouched.
    pending_range_set u;
    tlsf_allocator    t2(16 * MB);
    const size_t      old2 = t2.allocate_at(4 * MB, 4 * MB, 3);
    u.record(ctx_of(1), pending_term::ONEDNN_PP_A, 4 * MB, 4 * MB);
    CHECK(u.replace_within_block(t2, ctx_of(1), pending_term::ONEDNN_PP_A, old2, 5 * MB, MB, 5, 6).status ==
              pending_replace_status::REST_NOT_ONE,
          "a middle carve leaves two pieces, and only WEIGHT has the remainder mode for that");
    CHECK(t2.block_size_at(4 * MB) == 4 * MB, "refused before the release");
}

}  // namespace

int main() {
    case_record_replace_or_append();
    case_filtered_clear();
    case_filter_misuse();
    case_retag_then_clear();
    case_allocate_within_consume();
    case_allocate_within_miss();
    case_allocate_within_term_isolation();
    case_keyed_draw();
    case_adjacent_ranges();
    case_keyed_rounding_and_refusals();
    case_allocate_within_bad_requests();
    case_misuse_names_the_operation();
    case_all_under_load_and_retag_target();
    case_excluding_caller_contract();
    case_others_honour_ranges();
    case_device_wide_queries();
    case_replace_empty_rest();
    case_replace_term_scope();
    case_replace_failed_carve_restores_old();
    case_replace_remainder_mode();
    case_replace_other_term_rest();
    case_replace_off_grain_tail();
    std::printf("test-pending-range: all cases passed\n");
    return 0;
}
