// Host-side bookkeeping of the event-chained hybrid decode route's per-tensor record: why a held decision is stale,
// and the per-token count GGML_SYCL_MOE_HYBRID_RECORD_TRACE prints. ggml-sycl/moe-hybrid-record.hpp is SYCL-free,
// so these run without a device and link nothing.
#include "ggml-sycl/moe-hybrid-record.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(cond, msg)                      \
    do {                                      \
        if (!(cond)) {                        \
            std::printf("FAILED: %s\n", msg); \
            return 1;                         \
        }                                     \
    } while (0)

using ggml_sycl::moe_decode_direct_stamp;
using ggml_sycl::moe_hybrid_record_counts;
using ggml_sycl::moe_hybrid_record_miss;
using ggml_sycl::moe_hybrid_record_miss_classify;
using ggml_sycl::moe_hybrid_record_trace;

// A stamp the route settled ELIGIBLE at replan epoch 3, storage generation 7, for 8 selected rows.
static moe_decode_direct_stamp settled_stamp() {
    moe_decode_direct_stamp s{};
    ggml_sycl::moe_decode_direct_stamp_settle(s, 3, 7, 8, /*layout=*/0, ggml_sycl::MOE_DECODE_DIRECT_OUTCOME_ELIGIBLE);
    return s;
}

static int test_miss_classify() {
    const moe_decode_direct_stamp s = settled_stamp();
    CHECK(moe_hybrid_record_miss_classify(s, 3, 7, 8) == ggml_sycl::MOE_HYBRID_RECORD_MISS_NONE,
          "the inputs the decision was made for: the stamp answers");
    CHECK(moe_hybrid_record_miss_classify(s, 4, 7, 8) == ggml_sycl::MOE_HYBRID_RECORD_MISS_EPOCH,
          "a mid-load replan moved the epoch");
    CHECK(moe_hybrid_record_miss_classify(s, 3, 7, 4) == ggml_sycl::MOE_HYBRID_RECORD_MISS_ROWS,
          "the op selects another number of rows");
    CHECK(moe_hybrid_record_miss_classify(s, 3, 9, 8) == ggml_sycl::MOE_HYBRID_RECORD_MISS_GENERATION,
          "only the expert storage generation moved");
    // A miss on several inputs names the one a held record cannot be kept across: the epoch before the rows, the
    // rows before the generation.
    CHECK(moe_hybrid_record_miss_classify(s, 4, 9, 4) == ggml_sycl::MOE_HYBRID_RECORD_MISS_EPOCH,
          "epoch and generation and rows moved: the epoch is named");
    CHECK(moe_hybrid_record_miss_classify(s, 3, 9, 4) == ggml_sycl::MOE_HYBRID_RECORD_MISS_ROWS,
          "rows and generation moved: the rows are named");

    const moe_decode_direct_stamp fresh{};
    CHECK(moe_hybrid_record_miss_classify(fresh, 3, 7, 8) == ggml_sycl::MOE_HYBRID_RECORD_MISS_UNDECIDED,
          "a tensor never decided (or a dropped decision) has no stamp to compare");

    moe_decode_direct_stamp retried{};
    ggml_sycl::moe_decode_direct_stamp_settle(retried, 3, 7, 8, 0, ggml_sycl::MOE_DECODE_DIRECT_OUTCOME_RETRY);
    CHECK(moe_hybrid_record_miss_classify(retried, 3, 7, 8) == ggml_sycl::MOE_HYBRID_RECORD_MISS_RETRY,
          "a build that asked to be retried leaves no decision, and the miss says so");

    // The classifier agrees with the stamp's own test on every input it is asked about.
    for (uint64_t epoch = 2; epoch <= 4; ++epoch) {
        for (uint64_t gen = 6; gen <= 8; ++gen) {
            for (int64_t rows = 7; rows <= 9; ++rows) {
                const bool current = ggml_sycl::moe_decode_direct_stamp_current(s, epoch, gen, rows);
                const bool none =
                    moe_hybrid_record_miss_classify(s, epoch, gen, rows) == ggml_sycl::MOE_HYBRID_RECORD_MISS_NONE;
                CHECK(current == none, "NONE is exactly the stamp being current");
            }
        }
    }

    for (int m = 0; m < ggml_sycl::MOE_HYBRID_RECORD_MISS_COUNT; ++m) {
        const char * name = ggml_sycl::moe_hybrid_record_miss_name(static_cast<moe_hybrid_record_miss>(m));
        CHECK(name != nullptr && name[0] != '\0' && std::strcmp(name, "?") != 0, "every miss has a name");
    }
    return 0;
}

static int test_trace_tokens() {
    moe_hybrid_record_trace  t;
    uint64_t                 done_token = 99;
    moe_hybrid_record_counts done;

    // Token 0: blocks 0, 0, 1, 3 (gate and up of block 0 share a block number).
    CHECK(!t.lookup(0, &done_token, &done), "the first lookup ends no token");
    t.miss(ggml_sycl::MOE_HYBRID_RECORD_MISS_UNDECIDED, 0);
    t.build_outcome(ggml_sycl::MOE_DECODE_DIRECT_OUTCOME_ELIGIBLE);
    t.resolved(128);
    CHECK(!t.lookup(0, &done_token, &done), "the same block again ends no token");
    t.miss(ggml_sycl::MOE_HYBRID_RECORD_MISS_NONE, 0);
    CHECK(!t.lookup(1, &done_token, &done), "a later block ends no token");
    t.miss(ggml_sycl::MOE_HYBRID_RECORD_MISS_GENERATION, 3);
    t.build_outcome(ggml_sycl::MOE_DECODE_DIRECT_OUTCOME_REFUSED);
    t.resolved(128);
    CHECK(!t.lookup(3, &done_token, &done), "a later block ends no token");
    t.no_plan();

    // Token 1 starts at block 0: the lookup hands back token 0's counts and counts itself into token 1.
    CHECK(t.lookup(0, &done_token, &done), "a lower block starts a new token");
    CHECK(done_token == 0, "the token handed back is the one that ended");
    CHECK(done.lookups == 4, "token 0 made four lookups");
    CHECK(done.misses[ggml_sycl::MOE_HYBRID_RECORD_MISS_NONE] == 1, "one lookup the stamp answered");
    CHECK(done.misses[ggml_sycl::MOE_HYBRID_RECORD_MISS_UNDECIDED] == 1, "one lookup found no decision");
    CHECK(done.misses[ggml_sycl::MOE_HYBRID_RECORD_MISS_GENERATION] == 1, "one lookup missed on the generation");
    CHECK(done.generation_steps == 3, "the generation miss moved three steps");
    CHECK(done.no_plan == 1, "one lookup found no plan");
    CHECK(done.builds == 2 && done.built_experts == 256, "two builds resolved 128 experts each");
    CHECK(done.outcomes[ggml_sycl::MOE_DECODE_DIRECT_OUTCOME_ELIGIBLE] == 1 &&
              done.outcomes[ggml_sycl::MOE_DECODE_DIRECT_OUTCOME_REFUSED] == 1 &&
              done.outcomes[ggml_sycl::MOE_DECODE_DIRECT_OUTCOME_RETRY] == 0,
          "one build eligible, one refused");
    t.miss(ggml_sycl::MOE_HYBRID_RECORD_MISS_NONE, 0);

    CHECK(!t.lookup(5, &done_token, &done), "a later block ends no token");
    t.miss(ggml_sycl::MOE_HYBRID_RECORD_MISS_NONE, 0);
    CHECK(t.lookup(2, &done_token, &done), "a lower block starts token 2");
    CHECK(done_token == 1 && done.lookups == 2 && done.builds == 0 && done.no_plan == 0,
          "token 1 restarted from zero: two lookups, nothing built");
    CHECK(done.misses[ggml_sycl::MOE_HYBRID_RECORD_MISS_NONE] == 2, "both of token 1's lookups were current");
    CHECK(t.token == 2 && t.counts.lookups == 1, "token 2 holds the lookup that started it");
    return 0;
}

static int test_trace_line() {
    moe_hybrid_record_counts c;
    c.lookups                                                 = 144;
    c.misses[ggml_sycl::MOE_HYBRID_RECORD_MISS_NONE]          = 138;
    c.misses[ggml_sycl::MOE_HYBRID_RECORD_MISS_UNDECIDED]     = 1;
    c.misses[ggml_sycl::MOE_HYBRID_RECORD_MISS_RETRY]         = 0;
    c.misses[ggml_sycl::MOE_HYBRID_RECORD_MISS_EPOCH]         = 0;
    c.misses[ggml_sycl::MOE_HYBRID_RECORD_MISS_ROWS]          = 0;
    c.misses[ggml_sycl::MOE_HYBRID_RECORD_MISS_GENERATION]    = 5;
    c.generation_steps                                        = 15;
    c.no_plan                                                 = 0;
    c.builds                                                  = 6;
    c.built_experts                                           = 3072;
    c.outcomes[ggml_sycl::MOE_DECODE_DIRECT_OUTCOME_ELIGIBLE] = 6;

    char      line[512];
    const int n = ggml_sycl::moe_hybrid_record_line(line, sizeof(line), 41, c);
    CHECK(n > 0 && static_cast<size_t>(n) < sizeof(line), "the line fits its buffer");
    const std::string want =
        "token=41 lookups=144 current=138 no_plan=0 undecided=1 retry=0 epoch=0 rows=0 generation=5 "
        "generation_steps=15 builds=6 built_experts=3072 eligible=6 refused=0 retried=0";
    if (want != line) {
        std::printf("FAILED: line\n  got:  %s\n  want: %s\n", line, want.c_str());
        return 1;
    }

    // The widest values each field can hold still fit the buffer the backend prints from.
    moe_hybrid_record_counts wide;
    wide.lookups = wide.no_plan = wide.generation_steps = wide.builds = wide.built_experts = UINT64_MAX;
    for (uint64_t & m : wide.misses) {
        m = UINT64_MAX;
    }
    for (uint64_t & o : wide.outcomes) {
        o = UINT64_MAX;
    }
    const int wn = ggml_sycl::moe_hybrid_record_line(line, sizeof(line), UINT64_MAX, wide);
    CHECK(wn > 0 && static_cast<size_t>(wn) < sizeof(line), "the widest line fits the backend's 512-byte buffer");
    return 0;
}

// The weight's log of which expert each generation bump changed: a holder of a decision made at `since` asks which
// experts changed up to `now`, and the log answers only when it saw every bump in between and each named an expert.
static int test_change_log() {
    using ggml_sycl::moe_storage_change_log;
    moe_storage_change_log log;
    std::vector<int32_t>   ids = { 99 };

    // The weight's generation starts at 1 with no bump made.
    CHECK(log.changed_since(1, 1, &ids) && ids.empty(), "no bump since the decision: nothing changed");
    CHECK(!log.changed_since(1, 2, &ids), "a bump the log never saw has no answer");
    CHECK(!log.changed_since(3, 2, &ids), "a decision newer than the weight has no answer");

    // An evict of expert 5 (its storage record dropped, then its host copy remembered) and an admit of expert 7.
    log.note(2, 5);
    log.note(3, 5);
    log.note(4, 7);
    CHECK(log.changed_since(1, 4, &ids) && ids == std::vector<int32_t>({ 5, 7 }),
          "every bump since the decision, each expert once, ascending");
    CHECK(log.changed_since(3, 4, &ids) && ids == std::vector<int32_t>({ 7 }), "only the bumps after the decision");
    CHECK(log.changed_since(4, 4, &ids) && ids.empty(), "a decision at the newest generation: nothing changed");

    // A bump that changed every expert (a cleared store) answers nothing across it, and the bumps after it still do.
    log.note(5, moe_storage_change_log::ALL);
    CHECK(!log.changed_since(4, 5, &ids), "a bump naming every expert has no per-expert answer");
    CHECK(!log.changed_since(1, 5, &ids), "nor does any window that holds it");
    log.note(6, 3);
    CHECK(log.changed_since(5, 6, &ids) && ids == std::vector<int32_t>({ 3 }), "the bump after it answers alone");

    // A bump made without a note leaves a hole: a window across the hole has no answer, one after it does.
    log.note(8, 1);
    CHECK(!log.changed_since(6, 8, &ids), "generation 7 moved without a note: no answer across it");
    CHECK(log.changed_since(7, 8, &ids) && ids == std::vector<int32_t>({ 1 }), "the window after the hole answers");
    CHECK(!log.changed_since(8, 9, &ids), "a newest bump made without a note: no answer");

    // The log keeps the last `capacity` bumps. A window the log still holds answers, across the ring's wrap; a window
    // reaching past it does not.
    moe_storage_change_log ring;
    const uint64_t         first = 10;
    const uint64_t         last  = first + moe_storage_change_log::capacity + 3;
    for (uint64_t g = first; g <= last; ++g) {
        ring.note(g, static_cast<int32_t>(g % 5));
    }
    CHECK(ring.changed_since(last - moe_storage_change_log::capacity, last, &ids) &&
              ids == std::vector<int32_t>({ 0, 1, 2, 3, 4 }),
          "the widest window the log holds answers, across the wrap");
    CHECK(ring.changed_since(last - 2, last, &ids) && ids.size() == 2, "a narrow window at the end answers");
    CHECK(!ring.changed_since(last - moe_storage_change_log::capacity - 1, last, &ids),
          "one bump past the log's window: no answer");
    CHECK(!ring.changed_since(first - 1, last, &ids), "a decision older than every bump the log holds: no answer");
    return 0;
}

int main() {
    if (test_miss_classify() != 0 || test_trace_tokens() != 0 || test_trace_line() != 0 || test_change_log() != 0) {
        return 1;
    }
    std::printf("OK: moe hybrid decode record bookkeeping\n");
    return 0;
}
