// Dense two-card split decode policies (llama.cpp-m1or).
//
// Host-only: no SYCL queue, no device, no model. Both policies are pure
// functions of facts the backend gathers, which is what makes them testable.
//
// 1. simple_consumer_src_needs_staging. The case this exists for is
//    `local-control-leaf-is-not-staged`: in a Mistral split (B70 layers 0-29,
//    B50 layers 30-31) every device-0 ROPE and the embedding GET_ROWS were
//    routed -- staged, copied synchronously, drained -- only because their
//    position/token leaf lives in host USM, although that host USM belongs to
//    device 0's own context and a single-card run reads it in place. 61 routes
//    per decoded token. Cross-device staging must be untouched.
//
// 2. block_exec_first_failing_precheck. The classifier printed a literal
//    `executor=inactive`; the executor now reports the first gate that stopped
//    it, and these are the gates it evaluates.

#include "block-exec-gate.hpp"
#include "simple-consumer-route-policy.hpp"

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

using ggml_sycl::block_exec_first_failing_precheck;
using ggml_sycl::block_exec_gate;
using ggml_sycl::block_exec_gate_name;
using ggml_sycl::block_exec_precheck_inputs;
using ggml_sycl::simple_consumer_src_facts;
using ggml_sycl::simple_consumer_src_needs_staging;

static void check(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// inp_pos / inp_tokens as a device-0 graph sees them: an ownerless graph leaf
// flagged as input, in pinned host memory of device 0's context.
static simple_consumer_src_facts local_control_leaf() {
    simple_consumer_src_facts f{};
    f.owner                         = -1;
    f.is_leaf                       = true;
    f.is_control_input              = true;
    f.accessible_as_device_operand  = false;
    f.host_addressable_by_execution = true;
    return f;
}

static void test_local_control_leaf_is_not_staged() {
    check(!simple_consumer_src_needs_staging(local_control_leaf(), 0, 0, true),
          "local-control-leaf-is-not-staged: a device-0 op must read device 0's own host control leaf in place");
}

static void test_cross_device_control_leaf_is_staged() {
    // Device-1 ROPE (layers 30-31) inside device 0's graph: the leaf is in
    // device 0's context, so device 1 cannot read it. Staging stays.
    simple_consumer_src_facts f     = local_control_leaf();
    f.host_addressable_by_execution = false;
    check(simple_consumer_src_needs_staging(f, 1, 0, true),
          "cross-device-control-leaf-is-staged: a control leaf consumed on another device must still be staged");
    // Even if the other context could address it, a cross-device target keeps
    // its existing route: this change is scoped to the executing device.
    check(simple_consumer_src_needs_staging(local_control_leaf(), 1, 0, true),
          "cross-device-target-keeps-staging: the exemption applies only when the target is the executing device");
}

static void test_foreign_context_host_leaf_is_staged() {
    simple_consumer_src_facts f     = local_control_leaf();
    f.host_addressable_by_execution = false;
    check(simple_consumer_src_needs_staging(f, 0, 0, true),
          "foreign-context-host-leaf-is-staged: host memory of another context is not readable in place");
}

static void test_host_weight_is_staged() {
    // Placement decides the executor: a host-resident weight is never read by
    // a GPU over PCIe, even when its pinned memory is addressable.
    simple_consumer_src_facts f = local_control_leaf();
    f.is_control_input          = false;
    check(simple_consumer_src_needs_staging(f, 0, 0, true),
          "host-weight-is-staged: an addressable host weight is still not a control input");
}

static void test_device_owned_sources() {
    simple_consumer_src_facts remote{};
    remote.owner   = 1;
    remote.is_leaf = false;
    check(simple_consumer_src_needs_staging(remote, 0, 0, true),
          "remote-intermediate-is-staged: a device-1 activation consumed on device 0 must be staged");

    simple_consumer_src_facts local{};
    local.owner = 0;
    check(!simple_consumer_src_needs_staging(local, 0, 0, true),
          "local-intermediate-is-not-staged: a device-0 activation consumed on device 0 needs nothing");

    remote.accessible_as_device_operand = true;
    check(!simple_consumer_src_needs_staging(remote, 0, 0, true),
          "accessible-remote-is-not-staged: a source already usable as a device operand needs nothing");
}

static void test_unchanged_edges() {
    simple_consumer_src_facts f     = local_control_leaf();
    f.host_addressable_by_execution = false;
    check(!simple_consumer_src_needs_staging(f, 0, 0, false),
          "no-active-source: an op whose sources are all leaves is not routed for a host leaf");
    check(!simple_consumer_src_needs_staging(f, -1, 0, true), "no-execution-device: nothing to stage onto");

    simple_consumer_src_facts intermediate{};
    intermediate.owner   = -1;
    intermediate.is_leaf = false;
    check(!simple_consumer_src_needs_staging(intermediate, 0, 0, true),
          "ownerless-intermediate-is-not-staged: the host-leaf rule applies only to graph leaves");
}

static block_exec_precheck_inputs all_prechecks_pass() {
    block_exec_precheck_inputs in{};
    in.execute_enabled  = true;
    in.has_graph        = true;
    in.graph_recording  = false;
    in.has_plan         = true;
    in.candidate_blocks = 2;
    in.active_blocks    = 1;
    return in;
}

static void expect_gate(const block_exec_precheck_inputs & in, block_exec_gate want, const std::string & name) {
    const block_exec_gate got = block_exec_first_failing_precheck(in);
    check(got == want, name + " (got " + block_exec_gate_name(got) + ", want " + block_exec_gate_name(want) + ")");
}

static void test_block_exec_gates() {
    expect_gate(all_prechecks_pass(), ggml_sycl::BLOCK_EXEC_GATE_NONE, "all-prechecks-pass");

    // The dense split today, with the env default: several gates fail, and the
    // report names the one evaluated first.
    block_exec_precheck_inputs split = all_prechecks_pass();
    split.execute_enabled            = false;
    split.candidate_blocks           = 0;
    split.active_blocks              = 2;
    expect_gate(split, ggml_sycl::BLOCK_EXEC_GATE_EXECUTE_DISABLED, "dense-split-default-env");

    // The lead's pre-registered M2 line with GGML_SYCL_BLOCK_EXEC_EXECUTE=1:
    // `skip candidate execute plan=1 candidate_blocks=0 active_blocks=2`.
    split.execute_enabled = true;
    expect_gate(split, ggml_sycl::BLOCK_EXEC_GATE_CANDIDATE_BLOCKS, "dense-split-execute-enabled");

    block_exec_precheck_inputs in = all_prechecks_pass();
    in.active_blocks              = 2;
    expect_gate(in, ggml_sycl::BLOCK_EXEC_GATE_ACTIVE_MULTI_BLOCK, "active-multi-block");

    in          = all_prechecks_pass();
    in.has_plan = false;
    expect_gate(in, ggml_sycl::BLOCK_EXEC_GATE_NO_PLAN, "no-plan");

    in                 = all_prechecks_pass();
    in.graph_recording = true;
    in.has_plan        = false;
    expect_gate(in, ggml_sycl::BLOCK_EXEC_GATE_GRAPH_RECORDING, "recording-before-plan");

    in           = all_prechecks_pass();
    in.has_graph = false;
    expect_gate(in, ggml_sycl::BLOCK_EXEC_GATE_NO_GRAPH, "no-graph");

    for (int g = ggml_sycl::BLOCK_EXEC_GATE_NONE; g <= ggml_sycl::BLOCK_EXEC_GATE_EXECUTION_REJECTED; ++g) {
        const char * name = block_exec_gate_name(static_cast<block_exec_gate>(g));
        check(name != nullptr && std::strcmp(name, "unknown") != 0, "every gate has a name: " + std::to_string(g));
        for (int h = ggml_sycl::BLOCK_EXEC_GATE_NONE; h < g; ++h) {
            check(std::strcmp(name, block_exec_gate_name(static_cast<block_exec_gate>(h))) != 0,
                  "gate names are distinct: " + std::to_string(g));
        }
    }
}

int main() {
    struct test_case {
        const char * name;
        void (*fn)();
    };

    const test_case cases[] = {
        { "local-control-leaf-is-not-staged",    test_local_control_leaf_is_not_staged    },
        { "cross-device-control-leaf-is-staged", test_cross_device_control_leaf_is_staged },
        { "foreign-context-host-leaf-is-staged", test_foreign_context_host_leaf_is_staged },
        { "host-weight-is-staged",               test_host_weight_is_staged               },
        { "device-owned-sources",                test_device_owned_sources                },
        { "unchanged-edges",                     test_unchanged_edges                     },
        { "block-exec-gates",                    test_block_exec_gates                    },
    };

    int failed = 0;
    for (const test_case & tc : cases) {
        try {
            tc.fn();
            std::cout << "PASS " << tc.name << "\n";
        } catch (const std::exception & e) {
            std::cout << "FAIL " << tc.name << ": " << e.what() << "\n";
            failed++;
        }
    }
    std::cout << (failed == 0 ? "ALL PASS" : "FAILED") << " (" << failed << " failed)\n";
    return failed == 0 ? 0 : 1;
}
