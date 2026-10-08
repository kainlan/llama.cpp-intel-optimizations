// MoE graph preload memo: per-tensor, per-device stamps of the preload's
// outcome, and the per-tensor verdict for planner-owned experts.
// ggml-sycl/moe-graph-preload-stamp.hpp is SYCL-free, so these run without a device.
#include "ggml-sycl/moe-graph-preload-stamp.hpp"

#include <cstdio>

#define CHECK(cond, msg)                      \
    do {                                      \
        if (!(cond)) {                        \
            std::printf("FAILED: %s\n", msg); \
            return 1;                         \
        }                                     \
    } while (0)

using ggml_sycl::moe_graph_preload_classify;
using ggml_sycl::moe_graph_preload_inputs;
using ggml_sycl::moe_graph_preload_split_add;
using ggml_sycl::moe_graph_preload_split_decide;
using ggml_sycl::moe_graph_preload_split_decision;
using ggml_sycl::moe_graph_preload_split_scan;
using ggml_sycl::moe_graph_preload_stamp;
using ggml_sycl::moe_graph_preload_stamp_current;
using ggml_sycl::moe_graph_preload_stamp_record;
using ggml_sycl::moe_graph_preload_tensor_verdict;

namespace {

moe_graph_preload_inputs decode_inputs(uint64_t generation) {
    moe_graph_preload_inputs in;
    in.replan_epoch       = 3;
    in.storage_generation = generation;
    in.n_tokens           = 1;
    in.device             = 0;
    in.host_tier_boundary = true;
    return in;
}

// A three-tensor split (gate, up, down) whose stamps were recorded with `recorded`.
struct split_fixture {
    moe_graph_preload_stamp  stamps[3];
    moe_graph_preload_inputs inputs[3];

    split_fixture() {
        for (int i = 0; i < 3; ++i) {
            inputs[i] = decode_inputs(10 + i);
        }
    }

    void record_all(bool refused) {
        for (int i = 0; i < 3; ++i) {
            moe_graph_preload_stamp_record(stamps[i], inputs[i], refused);
        }
    }

    moe_graph_preload_split_decision decide() const {
        moe_graph_preload_split_scan scan;
        for (int i = 0; i < 3; ++i) {
            moe_graph_preload_split_add(scan, stamps[i], inputs[i]);
        }
        return moe_graph_preload_split_decide(scan);
    }
};

int test_stamp_inputs() {
    moe_graph_preload_stamp        s;
    const moe_graph_preload_inputs in = decode_inputs(7);
    CHECK(!moe_graph_preload_stamp_current(s, in), "an unrecorded stamp is never current");
    moe_graph_preload_stamp_record(s, in, false);
    CHECK(moe_graph_preload_stamp_current(s, in), "the same inputs are current");

    moe_graph_preload_inputs other = in;
    other.replan_epoch++;
    CHECK(!moe_graph_preload_stamp_current(s, other), "a replan (epoch bump) makes the stamp stale");
    other = in;
    other.storage_generation++;
    CHECK(!moe_graph_preload_stamp_current(s, other), "an expert storage rewrite makes the stamp stale");
    other          = in;
    other.n_tokens = 4;
    CHECK(!moe_graph_preload_stamp_current(s, other), "a different routed-token count makes the stamp stale");
    other        = in;
    other.device = 1;
    CHECK(!moe_graph_preload_stamp_current(s, other), "another device's question is not answered by this stamp");
    other                    = in;
    other.host_tier_boundary = false;
    CHECK(!moe_graph_preload_stamp_current(s, other), "whether host tensors may be boundaries is part of the answer");
    return 0;
}

int test_split_decisions() {
    {
        moe_graph_preload_split_scan scan;
        CHECK(moe_graph_preload_split_decide(scan) == moe_graph_preload_split_decision::NO_MOE,
              "a split with no MUL_MAT_ID has nothing to prepare");
    }
    {
        split_fixture f;
        CHECK(f.decide() == moe_graph_preload_split_decision::RUN, "a split never prepared runs the preload");
        f.record_all(false);
        CHECK(f.decide() == moe_graph_preload_split_decision::KNOWN_OK,
              "a split whose tensors all succeeded under unchanged inputs skips the preload");
        f.inputs[2].storage_generation++;
        CHECK(f.decide() == moe_graph_preload_split_decision::RUN,
              "a storage change on any one tensor reruns the preload");
    }
    {
        split_fixture f;
        f.record_all(false);
        for (int i = 0; i < 3; ++i) {
            f.inputs[i].replan_epoch++;
        }
        CHECK(f.decide() == moe_graph_preload_split_decision::RUN, "a replan reruns the preload");
    }
    {
        split_fixture f;
        f.record_all(false);
        for (int i = 0; i < 3; ++i) {
            f.inputs[i].device = 1;
        }
        CHECK(f.decide() == moe_graph_preload_split_decision::RUN, "another device runs its own preload");
    }
    {
        // The preload stops at the first refused tensor, so later tensors have no stamp.
        split_fixture f;
        moe_graph_preload_stamp_record(f.stamps[0], f.inputs[0], false);
        moe_graph_preload_stamp_record(f.stamps[1], f.inputs[1], true);
        CHECK(f.decide() == moe_graph_preload_split_decision::REFUSED,
              "a current refusal keeps the split off graphs without rerunning the preload");
        f.inputs[1].storage_generation++;
        CHECK(f.decide() == moe_graph_preload_split_decision::RUN,
              "a storage change on the refused tensor re-opens the decision");
    }
    {
        split_fixture f;
        moe_graph_preload_stamp_record(f.stamps[0], f.inputs[0], true);
        f.inputs[0].replan_epoch++;
        CHECK(f.decide() == moe_graph_preload_split_decision::RUN, "a stale refusal is not a refusal");
    }
    return 0;
}

int test_tensor_verdicts() {
    const int64_t n = 512;
    CHECK(moe_graph_preload_classify(512, 0, 0, 0, n, false) == moe_graph_preload_tensor_verdict::TABLE,
          "every expert local: one pointer table");
    CHECK(moe_graph_preload_classify(0, 0, 512, 0, n, true) == moe_graph_preload_tensor_verdict::HOST_TIER_BOUNDARY,
          "every expert host-planned: a direct boundary node, not a veto");
    CHECK(moe_graph_preload_classify(0, 0, 512, 0, n, false) == moe_graph_preload_tensor_verdict::REFUSE,
          "an all-host tensor is refused where MoE nodes would be recorded into the graph");
    CHECK(moe_graph_preload_classify(300, 0, 212, 0, n, true) == moe_graph_preload_tensor_verdict::REFUSE,
          "a mixed local/host tensor is refused");
    CHECK(moe_graph_preload_classify(0, 512, 0, 0, n, true) == moe_graph_preload_tensor_verdict::REFUSE,
          "experts on another device are refused");
    CHECK(moe_graph_preload_classify(0, 0, 511, 1, n, true) == moe_graph_preload_tensor_verdict::REFUSE,
          "a missing expert is refused");
    return 0;
}

}  // namespace

int main() {
    int rc = 0;
    rc |= test_stamp_inputs();
    rc |= test_split_decisions();
    rc |= test_tensor_verdicts();
    if (rc == 0) {
        std::printf("test-sycl-moe-graph-preload-stamp: all checks passed\n");
    }
    return rc;
}
