// Host-side bookkeeping of the event-chained hybrid decode route's per-tensor record: why a held decision is stale,
// and the per-token count GGML_SYCL_MOE_HYBRID_RECORD_TRACE prints. ggml-sycl/moe-hybrid-record.hpp and the weight's
// change log it reads (ggml-sycl/moe-storage-change-log.hpp) are SYCL-free, so these run without a device and link
// nothing.
#include "ggml-sycl/cow-chunked-vector.hpp"
#include "ggml-sycl/moe-hybrid-record.hpp"
#include "ggml-sycl/moe-storage-change-log.hpp"

#include <cstdio>
#include <cstring>
#include <memory>
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
    CHECK(moe_hybrid_record_miss_classify(s, 3, 7, 8, false) == ggml_sycl::MOE_HYBRID_RECORD_MISS_NONE,
          "the inputs the decision was made for: the stamp answers");
    CHECK(moe_hybrid_record_miss_classify(s, 4, 7, 8, false) == ggml_sycl::MOE_HYBRID_RECORD_MISS_EPOCH,
          "a mid-load replan moved the epoch");
    CHECK(moe_hybrid_record_miss_classify(s, 3, 7, 4, false) == ggml_sycl::MOE_HYBRID_RECORD_MISS_ROWS,
          "the op selects another number of rows");
    CHECK(moe_hybrid_record_miss_classify(s, 3, 9, 8, false) == ggml_sycl::MOE_HYBRID_RECORD_MISS_GENERATION,
          "only the expert storage generation moved");
    // A miss on several inputs names the one a held record cannot be kept across: the epoch before the rows, the
    // rows before the generation.
    CHECK(moe_hybrid_record_miss_classify(s, 4, 9, 4, false) == ggml_sycl::MOE_HYBRID_RECORD_MISS_EPOCH,
          "epoch and generation and rows moved: the epoch is named");
    CHECK(moe_hybrid_record_miss_classify(s, 3, 9, 4, false) == ggml_sycl::MOE_HYBRID_RECORD_MISS_ROWS,
          "rows and generation moved: the rows are named");

    // A current stamp over a record an expert loan or evict withdrew a slot from is the generation move the drop made,
    // never the held decision: its table still names a retired weight. A replan is still named first.
    CHECK(
        moe_hybrid_record_miss_classify(s, 3, 7, 8, /*withdrawn=*/true) == ggml_sycl::MOE_HYBRID_RECORD_MISS_GENERATION,
        "a withdrawn record under a current stamp is a generation miss");
    CHECK(moe_hybrid_record_miss_classify(s, 4, 7, 8, /*withdrawn=*/true) == ggml_sycl::MOE_HYBRID_RECORD_MISS_EPOCH,
          "a withdrawn record under a replan is still the replan");
    // The log names nothing since the stamp's generation, so the lookup builds the record: it never stalls on it.
    {
        ggml_sycl::moe_storage_change_log log;
        log.note(7, 1);
        std::vector<int32_t> changed;
        CHECK(ggml_sycl::moe_hybrid_record_plan_patch(ggml_sycl::MOE_HYBRID_RECORD_MISS_GENERATION, s, true, log, 7,
                                                      &changed) == ggml_sycl::MOE_HYBRID_RECORD_PATCH_PLAN_UNLOGGED &&
                  changed.empty(),
              "a withdrawn record under a current stamp is built again");
    }

    const moe_decode_direct_stamp fresh{};
    CHECK(moe_hybrid_record_miss_classify(fresh, 3, 7, 8, false) == ggml_sycl::MOE_HYBRID_RECORD_MISS_UNDECIDED,
          "a tensor never decided (or a dropped decision) has no stamp to compare");

    moe_decode_direct_stamp retried{};
    ggml_sycl::moe_decode_direct_stamp_settle(retried, 3, 7, 8, 0, ggml_sycl::MOE_DECODE_DIRECT_OUTCOME_RETRY);
    CHECK(moe_hybrid_record_miss_classify(retried, 3, 7, 8, false) == ggml_sycl::MOE_HYBRID_RECORD_MISS_RETRY,
          "a build that asked to be retried leaves no decision, and the miss says so");

    // The classifier agrees with the stamp's own test on every input it is asked about.
    for (uint64_t epoch = 2; epoch <= 4; ++epoch) {
        for (uint64_t gen = 6; gen <= 8; ++gen) {
            for (int64_t rows = 7; rows <= 9; ++rows) {
                const bool current = ggml_sycl::moe_decode_direct_stamp_current(s, epoch, gen, rows);
                const bool none    = moe_hybrid_record_miss_classify(s, epoch, gen, rows, false) ==
                                  ggml_sycl::MOE_HYBRID_RECORD_MISS_NONE;
                CHECK(current == none, "NONE is exactly the stamp being current");
                CHECK(moe_hybrid_record_miss_classify(s, epoch, gen, rows, true) !=
                          ggml_sycl::MOE_HYBRID_RECORD_MISS_NONE,
                      "a withdrawn record is never held, whatever the stamp");
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
    c.patches                                                 = 4;
    c.patch_experts                                           = 9;
    c.patch_declined                                          = 1;
    c.patch_unlogged                                          = 0;
    c.withdrawn                                               = 2;

    char      line[768];
    const int n = ggml_sycl::moe_hybrid_record_line(line, sizeof(line), 41, c);
    CHECK(n > 0 && static_cast<size_t>(n) < sizeof(line), "the line fits its buffer");
    const std::string want =
        "token=41 lookups=144 current=138 no_plan=0 undecided=1 retry=0 epoch=0 rows=0 generation=5 "
        "generation_steps=15 builds=6 built_experts=3072 eligible=6 refused=0 retried=0 patches=4 patch_experts=9 "
        "patch_declined=1 patch_unlogged=0 withdrawn=2";
    if (want != line) {
        std::printf("FAILED: line\n  got:  %s\n  want: %s\n", line, want.c_str());
        return 1;
    }

    // The widest values each field can hold still fit the buffer the backend prints from.
    moe_hybrid_record_counts wide;
    wide.lookups = wide.no_plan = wide.generation_steps = wide.builds = wide.built_experts = UINT64_MAX;
    wide.patches = wide.patch_experts = wide.patch_declined = wide.patch_unlogged = wide.withdrawn = UINT64_MAX;
    for (uint64_t & m : wide.misses) {
        m = UINT64_MAX;
    }
    for (uint64_t & o : wide.outcomes) {
        o = UINT64_MAX;
    }
    const int wn = ggml_sycl::moe_hybrid_record_line(line, sizeof(line), UINT64_MAX, wide);
    CHECK(wn > 0 && static_cast<size_t>(wn) < sizeof(line), "the widest line fits the backend's 768-byte buffer");
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

// A lease the tests can count: the shared count says how many holders keep its storage, so a test sees a replaced
// lease kept by the retained list alone, and an untouched lease kept by the record.
struct fake_lease {
    std::shared_ptr<int> storage;

    bool valid() const { return storage != nullptr; }
};

using fake_slot = ggml_sycl::moe_hybrid_record_slot<fake_lease>;

static fake_lease lease_of(int id) {
    return fake_lease{ std::make_shared<int>(id) };
}

// Which lookups a patch answers: a miss on the generation alone, over a record the decision built, when the log can
// name the experts that moved. Every other miss, and a log that overflowed, builds the record again.
static int test_patch_plan() {
    using ggml_sycl::moe_storage_change_log;
    const moe_decode_direct_stamp held = settled_stamp();  // eligible, at generation 7
    moe_storage_change_log        log;
    for (uint64_t g = 2; g <= 7; ++g) {
        log.note(g, 0);
    }
    log.note(8, 5);
    log.note(9, 2);
    std::vector<int32_t> changed;

    CHECK(ggml_sycl::moe_hybrid_record_plan_patch(ggml_sycl::MOE_HYBRID_RECORD_MISS_GENERATION, held, true, log, 9,
                                                  &changed) == ggml_sycl::MOE_HYBRID_RECORD_PATCH_PLAN_EXPERTS &&
              changed == std::vector<int32_t>({ 2, 5 }),
          "a generation miss over a held record re-checks the experts the log names");
    const moe_hybrid_record_miss full[] = { ggml_sycl::MOE_HYBRID_RECORD_MISS_EPOCH,
                                            ggml_sycl::MOE_HYBRID_RECORD_MISS_ROWS,
                                            ggml_sycl::MOE_HYBRID_RECORD_MISS_UNDECIDED,
                                            ggml_sycl::MOE_HYBRID_RECORD_MISS_RETRY,
                                            ggml_sycl::MOE_HYBRID_RECORD_MISS_NONE };
    for (moe_hybrid_record_miss m : full) {
        CHECK(ggml_sycl::moe_hybrid_record_plan_patch(m, held, true, log, 9, &changed) ==
                      ggml_sycl::MOE_HYBRID_RECORD_PATCH_PLAN_NONE &&
                  changed.empty(),
              "a replan, a new row count or no decision builds the record again, whatever the log says");
    }
    CHECK(ggml_sycl::moe_hybrid_record_plan_patch(ggml_sycl::MOE_HYBRID_RECORD_MISS_GENERATION, held, false, log, 9,
                                                  &changed) == ggml_sycl::MOE_HYBRID_RECORD_PATCH_PLAN_NONE,
          "no held record: nothing to patch");
    moe_decode_direct_stamp refused{};
    ggml_sycl::moe_decode_direct_stamp_settle(refused, 3, 7, 8, 0, ggml_sycl::MOE_DECODE_DIRECT_OUTCOME_REFUSED);
    CHECK(ggml_sycl::moe_hybrid_record_plan_patch(ggml_sycl::MOE_HYBRID_RECORD_MISS_GENERATION, refused, true, log, 9,
                                                  &changed) == ggml_sycl::MOE_HYBRID_RECORD_PATCH_PLAN_NONE,
          "a refusal at the old generation is decided again, not patched");

    // The log overflowed: more moves since the decision than it holds.
    moe_storage_change_log overflowed;
    const uint64_t         now = 7 + moe_storage_change_log::capacity + 1;
    for (uint64_t g = 8; g <= now; ++g) {
        overflowed.note(g, 1);
    }
    CHECK(ggml_sycl::moe_hybrid_record_plan_patch(ggml_sycl::MOE_HYBRID_RECORD_MISS_GENERATION, held, true, overflowed,
                                                  now, &changed) == ggml_sycl::MOE_HYBRID_RECORD_PATCH_PLAN_UNLOGGED &&
              changed.empty(),
          "an overflowed log cannot name the moved experts: the record is built again");
    moe_storage_change_log cleared = log;
    cleared.note(10, moe_storage_change_log::ALL);
    CHECK(ggml_sycl::moe_hybrid_record_plan_patch(ggml_sycl::MOE_HYBRID_RECORD_MISS_GENERATION, held, true, cleared, 10,
                                                  &changed) == ggml_sycl::MOE_HYBRID_RECORD_PATCH_PLAN_UNLOGGED,
          "a move that changed every expert: the record is built again");
    return 0;
}

// A patch installs the moved experts' slots into the copy it publishes, and nothing else.
static int test_install() {
    // Four experts: 0 and 2 on the device, 1 and 3 on the host.
    int                           w[8]          = {};
    std::vector<const void *>     device_weight = { &w[0], nullptr, &w[2], nullptr };
    std::vector<const void *>     host_weight   = { nullptr, &w[1], nullptr, &w[3] };
    std::vector<fake_lease>       lease         = { lease_of(0), lease_of(1), lease_of(2), lease_of(3) };
    const std::vector<fake_lease> before        = lease;  // the held record's leases, which the copy shares

    // Expert 2 is evicted to the host, expert 1 admitted to the device.
    std::vector<fake_slot> fresh(2);
    fresh[0].device_weight = &w[5];
    fresh[0].lease         = lease_of(5);
    fresh[1].host_weight   = &w[6];
    fresh[1].lease         = lease_of(6);
    std::vector<fake_lease> replaced;
    CHECK(ggml_sycl::moe_hybrid_record_install(device_weight, host_weight, lease, { 1, 2 }, fresh, &replaced),
          "the install takes in-range ids paired with their slots");

    CHECK(device_weight[1] == &w[5] && host_weight[1] == nullptr && *lease[1].storage == 5,
          "the admitted expert's slot names its device weight and lease");
    CHECK(device_weight[2] == nullptr && host_weight[2] == &w[6] && *lease[2].storage == 6,
          "the evicted expert's slot names its host weight and lease");
    CHECK(device_weight[0] == &w[0] && host_weight[0] == nullptr && lease[0].storage == before[0].storage,
          "an expert that did not move keeps its slot and its lease");
    CHECK(device_weight[3] == nullptr && host_weight[3] == &w[3] && lease[3].storage == before[3].storage,
          "the other unmoved expert keeps its slot and its lease");
    CHECK(replaced.size() == 2 && replaced[0].storage == before[1].storage && replaced[1].storage == before[2].storage,
          "each replaced lease is handed to the retained list, in slot order");
    // The replaced leases are held by the retained list and the held record only: the copy dropped them.
    CHECK(before[1].storage.use_count() == 2 && before[2].storage.use_count() == 2,
          "the copy holds no replaced lease; the retained list keeps it until the queue passes");
    CHECK(before[0].storage.use_count() == 2, "an untouched lease is shared by the held record and the copy");

    // An id out of range changes nothing.
    std::vector<fake_slot> one(1);
    one[0].host_weight = &w[7];
    one[0].lease       = lease_of(7);
    std::vector<fake_lease> none;
    CHECK(!ggml_sycl::moe_hybrid_record_install(device_weight, host_weight, lease, { 4 }, one, &none) && none.empty() &&
              host_weight[3] == &w[3],
          "an id past the record refuses the install and leaves every slot");
    CHECK(!ggml_sycl::moe_hybrid_record_install(device_weight, host_weight, lease, { 0, 3 }, one, &none) &&
              none.empty() && device_weight[0] == &w[0],
          "slots that do not pair with the ids refuse the install");

    // An in-range id ahead of one past the record: the install refuses before it writes any slot, so the in-range
    // expert keeps its slot and its lease, and no lease reaches the retained list.
    std::vector<fake_slot> two(2);
    two[0].host_weight = &w[7];
    two[0].lease       = lease_of(7);
    two[1].host_weight = &w[4];
    two[1].lease       = lease_of(4);

    const fake_lease slot_1_lease = lease[1];
    CHECK(
        !ggml_sycl::moe_hybrid_record_install(device_weight, host_weight, lease, { 1, 9 }, two, &none) && none.empty(),
        "an id past the record refuses the install, whatever ids come before it");
    CHECK(device_weight[1] == &w[5] && host_weight[1] == nullptr && lease[1].storage == slot_1_lease.storage,
          "the refused install leaves the in-range expert's slot and lease as they were");
    CHECK(two[0].lease.valid() && *two[0].lease.storage == 7, "the refused install takes no fresh lease");
    return 0;
}

// The shape a record's slots make, and the build's own refusals over it, which a patch must pass too.
static int test_shape() {
    int                       w[4]          = {};
    std::vector<const void *> device_weight = { &w[0], nullptr, &w[2], nullptr };
    std::vector<const void *> host_weight   = { nullptr, &w[1], nullptr, &w[3] };
    std::vector<fake_lease>   lease         = { lease_of(0), lease_of(1), lease_of(2), lease_of(3) };

    ggml_sycl::moe_hybrid_record_shape s = ggml_sycl::moe_hybrid_record_shape_of(device_weight, host_weight, lease);
    CHECK(s.n_device == 2 && s.n_host == 2 && s.n_empty == 0, "two device slots, two host slots");
    CHECK(s.substitute == &w[0], "the host slots' table entries name the first device expert");
    CHECK(ggml_sycl::moe_hybrid_record_shape_refusal(/*all_host=*/false, s) == nullptr, "a split record stands");
    CHECK(ggml_sycl::moe_hybrid_record_shape_refusal(/*all_host=*/true, s) != nullptr,
          "an all-host record with a device slot does not");

    // Every expert moved to the host: the split record's shape is gone, so the record is built again.
    std::vector<const void *> no_device(4, nullptr);
    std::vector<const void *> all_on_host = { &w[0], &w[1], &w[2], &w[3] };

    s = ggml_sycl::moe_hybrid_record_shape_of(no_device, all_on_host, lease);
    CHECK(s.n_device == 0 && s.n_host == 4 && s.substitute == nullptr, "four host slots, no substitute");
    CHECK(ggml_sycl::moe_hybrid_record_shape_refusal(/*all_host=*/false, s) != nullptr,
          "a split record with no device slot does not stand");
    CHECK(ggml_sycl::moe_hybrid_record_shape_refusal(/*all_host=*/true, s) == nullptr,
          "an all-host record of host slots stands: the build admits every all-host tensor");

    // Every expert on the device: no CPU work for the route.
    s = ggml_sycl::moe_hybrid_record_shape_of(all_on_host, no_device, lease);
    CHECK(ggml_sycl::moe_hybrid_record_shape_refusal(/*all_host=*/false, s) != nullptr,
          "a record with no host slot does not stand");

    // A slot with no lease, or naming both weights, or neither, is empty, and a record with one does not stand.
    std::vector<fake_lease> withdrawn = lease;
    withdrawn[2]                      = fake_lease{};

    s = ggml_sycl::moe_hybrid_record_shape_of(device_weight, host_weight, withdrawn);
    CHECK(s.n_empty == 1 && ggml_sycl::moe_hybrid_record_shape_refusal(false, s) != nullptr,
          "a slot whose lease was withdrawn is empty");
    std::vector<const void *> both = host_weight;
    both[0]                        = &w[0];

    s = ggml_sycl::moe_hybrid_record_shape_of(device_weight, both, lease);
    CHECK(s.n_empty == 1 && ggml_sycl::moe_hybrid_record_shape_refusal(false, s) != nullptr,
          "a slot naming a device and a host weight is empty");
    s = ggml_sycl::moe_hybrid_record_shape_of(no_device, no_device, lease);
    CHECK(s.n_empty == 4 && ggml_sycl::moe_hybrid_record_shape_refusal(true, s) != nullptr,
          "slots naming no weight are empty");

    // Arrays of different lengths are no record's slots: the shape is empty, and neither mode stands on it.
    const std::vector<fake_lease> three_leases(lease.begin(), lease.begin() + 3);
    s = ggml_sycl::moe_hybrid_record_shape_of(device_weight, host_weight, three_leases);
    CHECK(s.n_empty > 0 && ggml_sycl::moe_hybrid_record_shape_refusal(false, s) != nullptr &&
              ggml_sycl::moe_hybrid_record_shape_refusal(true, s) != nullptr,
          "fewer leases than weights is an empty shape");
    const std::vector<const void *> three_device(device_weight.begin(), device_weight.begin() + 3);
    s = ggml_sycl::moe_hybrid_record_shape_of(three_device, host_weight, lease);
    CHECK(s.n_empty > 0 && ggml_sycl::moe_hybrid_record_shape_refusal(false, s) != nullptr,
          "fewer device weights than leases is an empty shape");
    const std::vector<const void *> three_host(host_weight.begin(), host_weight.begin() + 3);
    s = ggml_sycl::moe_hybrid_record_shape_of(device_weight, three_host, lease);
    CHECK(s.n_empty > 0 && ggml_sycl::moe_hybrid_record_shape_refusal(true, s) != nullptr,
          "fewer host weights than leases is an empty shape");
    return 0;
}

// The plan's part of the build's decision, which a patch asks again, since an expert move rewrites the plan without
// moving the replan epoch. A patch stands only on the frame the build would take now: the held record's all-host
// mode, and for a split record no moved expert at or before the one whose layouts gave the record its layout.
static int test_plan_frame() {
    CHECK(ggml_sycl::moe_hybrid_record_plan_refusal(/*has_host=*/false, 2, 2) != nullptr,
          "a tensor whose plan puts no expert on the host has no record");
    CHECK(ggml_sycl::moe_hybrid_record_plan_refusal(/*has_host=*/true, 3, 2) != nullptr,
          "a tensor whose plan puts an expert on another card has no record");
    CHECK(ggml_sycl::moe_hybrid_record_plan_refusal(/*has_host=*/true, 0, 0) == nullptr,
          "a plan with no device expert is the all-host shape, not a refusal");
    CHECK(ggml_sycl::moe_hybrid_record_plan_refusal(/*has_host=*/true, 2, 2) == nullptr,
          "a split plan on this card stands");

    // A split record whose layout came from expert 3.
    CHECK(ggml_sycl::moe_hybrid_record_patch_frame_holds(false, 3, false, 4),
          "moves after the layout's expert leave the build's layout scan where it stopped");
    CHECK(!ggml_sycl::moe_hybrid_record_patch_frame_holds(false, 3, false, 3),
          "a move of the layout's own expert sends the record to the build");
    CHECK(!ggml_sycl::moe_hybrid_record_patch_frame_holds(false, 3, false, 1),
          "a move before it can stop the build's scan sooner: the build decides");
    CHECK(!ggml_sycl::moe_hybrid_record_patch_frame_holds(false, 3, true, 5),
          "a plan that now puts no expert on a device is the build's all-host record");
    // An all-host record has no layout scan.
    CHECK(ggml_sycl::moe_hybrid_record_patch_frame_holds(true, -1, true, 0), "any move of an all-host record stands");
    CHECK(!ggml_sycl::moe_hybrid_record_patch_frame_holds(true, -1, false, 7),
          "a plan that now puts an expert on the device is the build's split record");
    return 0;
}

// A loan or an evict withdraws, from a copy of the held record, the slots that lease the device entry it retired. The
// copy keeps every other slot and lease, and once the held record is released only the drop's retention leases the
// retired entry, so its room comes back when the queue passes the retention's barrier.
static int test_withdraw() {
    // Four experts: 0 and 2 on the device, 1 and 3 on the host; the entry expert 2's lease names is retired.
    int                       w[4]          = {};
    std::vector<const void *> device_weight = { &w[0], nullptr, &w[2], nullptr };
    std::vector<const void *> host_weight   = { nullptr, &w[1], nullptr, &w[3] };
    std::vector<fake_lease>   lease         = { lease_of(0), lease_of(1), lease_of(2), lease_of(3) };
    const int *               retired       = lease[2].storage.get();

    const auto on_entry = [retired](const fake_lease & l) {
        return l.storage.get() == retired;
    };
    const auto no_entry = [](const fake_lease &) {
        return false;
    };

    std::vector<const void *> copy_device = device_weight;
    std::vector<const void *> copy_host   = host_weight;
    std::vector<fake_lease>   copy_lease  = lease;
    std::vector<fake_lease>   retained;
    std::vector<int32_t>      emptied;
    const size_t              n =
        ggml_sycl::moe_hybrid_record_withdraw(copy_device, copy_host, copy_lease, on_entry, &retained, &emptied);
    CHECK(n == 1 && emptied == std::vector<int32_t>({ 2 }) && retained.size() == 1,
          "the drop empties the one slot that leases the retired entry and names its expert");
    CHECK(copy_device[2] == nullptr && copy_host[2] == nullptr && !copy_lease[2].valid(),
          "the withdrawn slot names no weight and holds no lease");
    CHECK(copy_device[0] == &w[0] && copy_host[1] == &w[1] && copy_host[3] == &w[3] &&
              copy_lease[0].storage == lease[0].storage && copy_lease[1].storage == lease[1].storage &&
              copy_lease[3].storage == lease[3].storage,
          "every other slot keeps its weight and its lease");
    CHECK(device_weight[2] == &w[2] && lease[2].storage.get() == retired,
          "the held record is not written: a CPU job or a queued table copy may still read it");

    // The copy replaces the held record, which goes once nothing else holds it.
    lease.clear();
    CHECK(retained[0].storage.get() == retired && retained[0].storage.use_count() == 1,
          "with the held record gone, only the drop's retention leases the retired entry");
    CHECK(copy_lease[0].storage.use_count() == 1 && copy_lease[3].storage.use_count() == 1,
          "the copy alone holds the slots it kept");

    const ggml_sycl::moe_hybrid_record_shape s =
        ggml_sycl::moe_hybrid_record_shape_of(copy_device, copy_host, copy_lease);
    CHECK(s.n_empty == 1 && ggml_sycl::moe_hybrid_record_shape_refusal(/*all_host=*/false, s) != nullptr,
          "a copy with a withdrawn slot does not stand until a patch refills the slot");

    // An entry no slot leases empties nothing and names no expert.
    std::vector<fake_lease> none;
    std::vector<int32_t>    none_ids;
    CHECK(ggml_sycl::moe_hybrid_record_withdraw(copy_device, copy_host, copy_lease, no_entry, &none, &none_ids) == 0 &&
              none.empty() && none_ids.empty() && copy_lease[0].valid(),
          "an entry no slot leases leaves every slot");
    return 0;
}

// The drop's slot-direct form: an expert move's retired entry backs only its own expert's slot, so the drop empties
// that one slot when it leases the entry, and walks every slot (the form above) only when it does not.
static int test_withdraw_slot() {
    int                       w[4]          = {};
    std::vector<const void *> device_weight = { &w[0], nullptr, &w[2], nullptr };
    std::vector<const void *> host_weight   = { nullptr, &w[1], nullptr, &w[3] };
    std::vector<fake_lease>   lease         = { lease_of(0), lease_of(1), lease_of(2), lease_of(3) };
    const int *               retired       = lease[2].storage.get();
    const auto                on_entry      = [retired](const fake_lease & l) {
        return l.storage.get() == retired;
    };

    CHECK(ggml_sycl::moe_hybrid_record_slot_matches(lease, 2, on_entry), "the retired expert's own slot matches");
    CHECK(!ggml_sycl::moe_hybrid_record_slot_matches(lease, 1, on_entry), "another expert's slot does not");
    CHECK(!ggml_sycl::moe_hybrid_record_slot_matches(lease, 4, on_entry) &&
              !ggml_sycl::moe_hybrid_record_slot_matches(lease, -1, on_entry),
          "an expert outside the record has no slot to match");

    std::vector<fake_lease> retained;
    std::vector<int32_t>    emptied;
    CHECK(ggml_sycl::moe_hybrid_record_withdraw_slot(device_weight, host_weight, lease, 2, &retained, &emptied),
          "the slot is emptied");
    CHECK(retained.size() == 1 && retained[0].storage.get() == retired && emptied == std::vector<int32_t>({ 2 }),
          "its lease is retained and its expert named");
    CHECK(device_weight[2] == nullptr && host_weight[2] == nullptr && !lease[2].valid(),
          "the slot names no weight and holds no lease");
    CHECK(lease[0].valid() && lease[1].valid() && lease[3].valid() && device_weight[0] == &w[0] &&
              host_weight[1] == &w[1] && host_weight[3] == &w[3],
          "every other slot keeps its weight and its lease");
    CHECK(!ggml_sycl::moe_hybrid_record_withdraw_slot(device_weight, host_weight, lease, 2, &retained, &emptied) &&
              retained.size() == 1,
          "an empty slot has nothing to withdraw");
    CHECK(!ggml_sycl::moe_hybrid_record_withdraw_slot(device_weight, host_weight, lease, 9, &retained, &emptied),
          "an expert outside the record has no slot");
    return 0;
}

// The record keeps its leases in chunks shared between a record and the copies a patch or a drop publishes, so a copy
// that writes one slot clones that slot's chunk and shares the rest with the held record.
using chunked_leases = ggml_sycl::cow_chunked_vector<fake_lease, ggml_sycl::moe_hybrid_record_lease_chunk>;

static chunked_leases chunked_leases_of(size_t n) {
    chunked_leases out;
    for (size_t e = 0; e < n; ++e) {
        out.push_back(lease_of(static_cast<int>(e)));
    }
    return out;
}

static int test_lease_chunks_shared() {
    constexpr size_t          chunk = ggml_sycl::moe_hybrid_record_lease_chunk;
    constexpr size_t          n_as  = 3 * chunk;  // three chunks
    std::vector<int>          w(n_as + 1);
    std::vector<const void *> device_weight(n_as, nullptr);
    std::vector<const void *> host_weight(n_as, nullptr);
    for (size_t e = 0; e < n_as; ++e) {
        device_weight[e] = &w[e];
    }
    const chunked_leases held = chunked_leases_of(n_as);  // the published record's leases: never written
    const auto           kept = [&](const chunked_leases & copy) {
        return copy.chunks_cloned() == 1 && copy.shared_chunks_with(held) == 2;
    };

    // A patch moves the last expert (third chunk) to the host.
    {
        chunked_leases            copy = held;
        std::vector<const void *> dw = device_weight, hw = host_weight;
        std::vector<fake_slot>    fresh(1);
        fresh[0].host_weight = &w[n_as];
        fresh[0].lease       = lease_of(100);
        std::vector<fake_lease> replaced;
        CHECK(ggml_sycl::moe_hybrid_record_install(dw, hw, copy, { static_cast<int32_t>(n_as - 1) }, fresh, &replaced),
              "the patch installs the moved expert's slot in the chunked leases");
        CHECK(kept(copy), "the patch clones the moved slot's chunk alone and shares the other two");
        CHECK(replaced.size() == 1 && replaced[0].storage == held[n_as - 1].storage &&
                  *std::as_const(copy)[n_as - 1].storage == 100,
              "the replaced lease is retained, the held record still holds it, and the copy holds the new one");
        const ggml_sycl::moe_hybrid_record_shape s = ggml_sycl::moe_hybrid_record_shape_of(dw, hw, copy);
        CHECK(s.n_device == n_as - 1 && s.n_host == 1 && s.n_empty == 0 && copy.chunks_cloned() == 1,
              "reading the copy's shape clones nothing");
    }
    // A drop empties expert 3's slot (first chunk) directly.
    {
        chunked_leases            copy = held;
        std::vector<const void *> dw = device_weight, hw = host_weight;
        std::vector<fake_lease>   retained;
        std::vector<int32_t>      emptied;
        CHECK(ggml_sycl::moe_hybrid_record_withdraw_slot(dw, hw, copy, 3, &retained, &emptied),
              "the slot withdraw empties the slot in the chunked leases");
        CHECK(kept(copy), "the slot withdraw clones that slot's chunk alone");
        CHECK(held[3].valid() && !std::as_const(copy)[3].valid(), "the held record keeps the lease the copy dropped");
    }
    // A drop that walks every slot finds expert chunk + 1 (second chunk).
    {
        chunked_leases            copy = held;
        std::vector<const void *> dw = device_weight, hw = host_weight;
        const int *               retired  = held[chunk + 1].storage.get();
        const auto                on_entry = [retired](const fake_lease & l) {
            return l.storage.get() == retired;
        };
        std::vector<fake_lease> retained;
        std::vector<int32_t>    emptied;
        CHECK(ggml_sycl::moe_hybrid_record_withdraw(dw, hw, copy, on_entry, &retained, &emptied) == 1 &&
                  emptied == std::vector<int32_t>({ static_cast<int32_t>(chunk + 1) }),
              "the walk empties the one slot that leases the retired entry");
        CHECK(kept(copy), "the walk reads every slot but clones only the emptied slot's chunk");

        chunked_leases untouched = held;
        const auto     no_entry  = [](const fake_lease &) {
            return false;
        };
        CHECK(ggml_sycl::moe_hybrid_record_withdraw(dw, hw, untouched, no_entry, &retained, &emptied) == 0 &&
                  untouched.chunks_cloned() == 0 && untouched.shared_chunks_with(held) == 3,
              "a walk that empties nothing clones nothing");
    }
    return 0;
}

// The per-token counts of what the patches did, and of the slots loans and evicts had withdrawn from the records the
// lookups replaced: the one sign on a device that a drop matched a record slot by the entry it retired.
static int test_trace_patches() {
    moe_hybrid_record_trace t;
    t.patch(ggml_sycl::MOE_HYBRID_RECORD_PATCH_PLAN_EXPERTS, /*patched=*/true, 2);
    t.patch(ggml_sycl::MOE_HYBRID_RECORD_PATCH_PLAN_EXPERTS, /*patched=*/false, 3);
    t.patch(ggml_sycl::MOE_HYBRID_RECORD_PATCH_PLAN_UNLOGGED, /*patched=*/false, 0);
    CHECK(t.counts.patches == 1 && t.counts.patch_declined == 1 && t.counts.patch_unlogged == 1,
          "one patch stood, one declined, one found the log silent");
    CHECK(t.counts.patch_experts == 5, "the attempts asked for five experts");
    t.withdrawn(2);
    t.withdrawn(1);
    CHECK(t.counts.withdrawn == 3, "two replaced records had three withdrawn slots between them");
    return 0;
}

int main() {
    if (test_miss_classify() != 0 || test_trace_tokens() != 0 || test_trace_line() != 0 || test_change_log() != 0 ||
        test_patch_plan() != 0 || test_install() != 0 || test_shape() != 0 || test_plan_frame() != 0 ||
        test_withdraw() != 0 || test_withdraw_slot() != 0 || test_lease_chunks_shared() != 0 ||
        test_trace_patches() != 0) {
        return 1;
    }
    std::printf("OK: moe hybrid decode record bookkeeping\n");
    return 0;
}
