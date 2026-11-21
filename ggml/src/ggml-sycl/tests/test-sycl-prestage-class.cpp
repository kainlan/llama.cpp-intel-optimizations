// What a graph prestage does with one source tensor (prestage-class.hpp).
//
// Host-only: no SYCL queue, no device, no model. The rule is a pure function of facts the backend gathers, and the
// view-root walk reads only ggml_tensor fields, so both are tested here on synthetic tensors.
//
// The case this exists for: when the plan has no CPU work, the CPU backend's compute buffer is the generic host buft,
// and the scheduler makes no device copy of an activation the CPU produces (an op the SYCL backend does not support).
// A SYCL split then reads it from host memory. Classed COPY, the prestage's cache arm would refuse it and the graph
// would never be recorded. An op output in host memory that the graph being staged does not produce is
// another split's per-token activation, so it takes the INPUT arm (stable staging, re-copied on every call) unless an
// earlier class claims it. An op output the graph produces itself (every intermediate, when the backend computes into
// host memory) stays COPY.

#include "prestage-class.hpp"

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

static void check(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static const char * class_name(ggml_sycl_prestage_class c) {
    switch (c) {
        case ggml_sycl_prestage_class::CONTROL:
            return "CONTROL";
        case ggml_sycl_prestage_class::WEIGHT:
            return "WEIGHT";
        case ggml_sycl_prestage_class::DEVICE:
            return "DEVICE";
        case ggml_sycl_prestage_class::INPUT:
            return "INPUT";
        case ggml_sycl_prestage_class::KV_HOST:
            return "KV_HOST";
        case ggml_sycl_prestage_class::COPY:
            return "COPY";
    }
    return "?";
}

static void check_class(const ggml_sycl_prestage_facts & facts, ggml_sycl_prestage_class want, const char * what) {
    const ggml_sycl_prestage_class got = ggml_sycl_prestage_class_of(facts);
    check(got == want, std::string(what) + ": want " + class_name(want) + ", got " + class_name(got));
}

static ggml_tensor tensor_with_op(ggml_op op) {
    ggml_tensor t;
    std::memset(&t, 0, sizeof(t));
    t.op = op;
    return t;
}

// A host-resident op output with no other fact set: what the CPU split leaves for a SYCL split to read.
static ggml_sycl_prestage_facts host_activation() {
    ggml_sycl_prestage_facts f{};
    f.root_is_op_output = true;
    return f;
}

static void test_host_op_output_is_input() {
    check_class(host_activation(), ggml_sycl_prestage_class::INPUT,
                "host-op-output-is-input: a host-resident op output is a per-token activation");
}

static void test_view_of_op_output_is_input() {
    // The view's own op is GGML_OP_VIEW whether it views a leaf or an op output; the root decides.
    ggml_tensor add  = tensor_with_op(GGML_OP_ADD);
    ggml_tensor view = tensor_with_op(GGML_OP_VIEW);
    view.view_src    = &add;
    check(ggml_sycl_prestage_root_is_op_output(&add), "root-walk: an ADD output is an op output");
    check(ggml_sycl_prestage_root_is_op_output(&view), "root-walk: a view of an ADD output is an op output");

    ggml_sycl_prestage_facts f = host_activation();
    f.root_is_op_output        = ggml_sycl_prestage_root_is_op_output(&view);
    check_class(f, ggml_sycl_prestage_class::INPUT, "view-of-op-output-is-input");
}

static void test_graph_own_op_output_stays_copy() {
    // With the backend's compute buffer in host memory every intermediate is a host op output. Staged as an input, it
    // would be copied before its producer ran.
    ggml_sycl_prestage_facts f = host_activation();
    f.root_is_graph_node       = true;
    check_class(f, ggml_sycl_prestage_class::COPY, "graph-own-op-output-stays-copy");
}

static void test_view_chain_reaches_the_root() {
    ggml_tensor add   = tensor_with_op(GGML_OP_ADD);
    ggml_tensor view1 = tensor_with_op(GGML_OP_VIEW);
    ggml_tensor view2 = tensor_with_op(GGML_OP_RESHAPE);
    ggml_tensor view3 = tensor_with_op(GGML_OP_PERMUTE);
    view1.view_src    = &add;
    view2.view_src    = &view1;
    view3.view_src    = &view2;
    check(ggml_sycl_prestage_view_root(&view3) == &add, "view-chain: three views resolve to the ADD output");
    check(ggml_sycl_prestage_root_is_op_output(&view3), "view-chain: the end of the chain is an op output");

    ggml_tensor leaf = tensor_with_op(GGML_OP_NONE);
    view1.view_src   = &leaf;
    check(ggml_sycl_prestage_view_root(&view3) == &leaf, "view-chain: three views resolve to the leaf");
    check(!ggml_sycl_prestage_root_is_op_output(&view3),
          "view-chain: a chain of views over a leaf is not an op output, although every link is an op");
}

static void test_input_flag_reaches_views() {
    // ggml_view_tensor() does not copy flags: the INPUT flag sits on the root, and every view of it is that input.
    ggml_tensor leaf  = tensor_with_op(GGML_OP_NONE);
    ggml_tensor view1 = tensor_with_op(GGML_OP_VIEW);
    ggml_tensor view2 = tensor_with_op(GGML_OP_RESHAPE);
    view1.view_src    = &leaf;
    view2.view_src    = &view1;
    check(!ggml_sycl_prestage_is_input(nullptr), "input-flag: no tensor is no input");
    check(!ggml_sycl_prestage_is_input(&view2), "input-flag: views of an unflagged leaf are not inputs");

    leaf.flags = GGML_TENSOR_FLAG_INPUT;
    check(ggml_sycl_prestage_is_input(&leaf), "input-flag: the flagged leaf is an input");
    check(ggml_sycl_prestage_is_input(&view2), "input-flag: a view two links from the flagged leaf is an input");

    leaf.flags  = 0;
    view1.flags = GGML_TENSOR_FLAG_INPUT;
    check(ggml_sycl_prestage_is_input(&view2), "input-flag: a flag on a middle link counts for the views above it");
    check(!ggml_sycl_prestage_is_input(&leaf), "input-flag: a flag on a view does not reach its root");
}

static void test_kv_write_is_not_an_activation() {
    // A KV write's result is an op that views the cache, a leaf. The root decides, so it never takes the op-output
    // rule, in the KV-host buft or anywhere else.
    ggml_tensor   kv_leaf  = tensor_with_op(GGML_OP_NONE);
    const ggml_op writes[] = { GGML_OP_SET_ROWS, GGML_OP_CPY };
    for (ggml_op op : writes) {
        ggml_tensor write = tensor_with_op(op);
        write.view_src    = &kv_leaf;
        check(!ggml_sycl_prestage_root_is_op_output(&write), "kv-write: the cache root is a leaf");

        ggml_sycl_prestage_facts f{};
        f.root_is_op_output = ggml_sycl_prestage_root_is_op_output(&write);
        f.is_kv_host        = true;
        check_class(f, ggml_sycl_prestage_class::KV_HOST, "kv-write-in-kv-host-stays-kv-host");
        f.is_kv_host = false;
        check_class(f, ggml_sycl_prestage_class::COPY, "kv-write-elsewhere-stays-copy");
    }
}

static void test_reported_bytes() {
    // The resolver host-return counter reads these bytes: a source an ordinary kernel reads from host memory in place.
    ggml_sycl_prestage_facts f{};
    check(ggml_sycl_prestage_reports_bytes(f), "reports-bytes: a COPY source");

    f = host_activation();
    check(ggml_sycl_prestage_reports_bytes(f), "reports-bytes: a host op output from another split");
    f.root_is_graph_node = true;
    check(ggml_sycl_prestage_reports_bytes(f),
          "reports-bytes: a host op output this graph produces (also the answer when the graph is unknown)");

    f          = host_activation();
    f.is_input = true;
    check(!ggml_sycl_prestage_reports_bytes(f), "reports-bytes: never a flagged INPUT, op output or not");

    f            = ggml_sycl_prestage_facts{};
    f.is_kv_host = true;
    check(!ggml_sycl_prestage_reports_bytes(f), "reports-bytes: not a KV-host source");
    f           = host_activation();
    f.is_weight = true;
    check(!ggml_sycl_prestage_reports_bytes(f), "reports-bytes: not a weight");
    f           = host_activation();
    f.is_device = true;
    check(!ggml_sycl_prestage_reports_bytes(f), "reports-bytes: not device USM");
    f            = host_activation();
    f.is_control = true;
    check(!ggml_sycl_prestage_reports_bytes(f), "reports-bytes: not control storage");
}

static void test_leaf_without_flag_stays_copy() {
    ggml_tensor leaf = tensor_with_op(GGML_OP_NONE);
    ggml_tensor view = tensor_with_op(GGML_OP_VIEW);
    view.view_src    = &leaf;
    check(!ggml_sycl_prestage_root_is_op_output(&leaf), "root-walk: a leaf is not an op output");
    check(!ggml_sycl_prestage_root_is_op_output(&view),
          "root-walk: a view of a leaf is not an op output, although the view's own op is VIEW");
    check(!ggml_sycl_prestage_root_is_op_output(nullptr), "root-walk: no tensor is not an op output");

    ggml_sycl_prestage_facts f{};
    f.root_is_op_output = ggml_sycl_prestage_root_is_op_output(&view);
    check_class(f, ggml_sycl_prestage_class::COPY, "leaf-without-flag-stays-copy");
}

static void test_flagged_input_is_input() {
    ggml_sycl_prestage_facts f{};
    f.is_input = true;
    check_class(f, ggml_sycl_prestage_class::INPUT, "flagged-input-is-input");
}

static void test_kv_host_stays_kv_host() {
    // A KV write (SET_ROWS, CPY) is an op whose result views the cache. The cache root is a leaf, but the rule must
    // not depend on that: KV-host state is persistent, never per-call staging, even with an op-output fact.
    ggml_sycl_prestage_facts f = host_activation();
    f.is_kv_host               = true;
    check_class(f, ggml_sycl_prestage_class::KV_HOST, "kv-host-stays-kv-host: op output in the KV-host buft");
    f.root_is_op_output = false;
    check_class(f, ggml_sycl_prestage_class::KV_HOST, "kv-host-stays-kv-host: KV-host leaf");
    // The flag keeps its place ahead of KV_HOST.
    f.is_input = true;
    check_class(f, ggml_sycl_prestage_class::INPUT, "flagged-input-ahead-of-kv-host");
}

static void test_weight_stays_weight() {
    ggml_sycl_prestage_facts f = host_activation();
    f.is_weight                = true;
    check_class(f, ggml_sycl_prestage_class::WEIGHT, "weight-stays-weight: with an op-output fact");
    f.is_input = true;
    check_class(f, ggml_sycl_prestage_class::WEIGHT, "weight-stays-weight: with the INPUT flag");
}

static void test_device_stays_device() {
    ggml_sycl_prestage_facts f = host_activation();
    f.is_device                = true;
    check_class(f, ggml_sycl_prestage_class::DEVICE, "device-stays-device: a device-resident op output");
    f.is_input   = true;
    f.is_kv_host = true;
    check_class(f, ggml_sycl_prestage_class::DEVICE, "device-stays-device: with the INPUT flag and the KV-host buft");
}

static void test_control_comes_first() {
    ggml_sycl_prestage_facts f = host_activation();
    f.is_control               = true;
    f.is_weight                = true;
    f.is_device                = true;
    f.is_input                 = true;
    f.is_kv_host               = true;
    check_class(f, ggml_sycl_prestage_class::CONTROL, "control-comes-first");
}

int main() {
    struct test_case {
        const char * name;
        void (*fn)();
    };

    const test_case cases[] = {
        { "host-op-output-is-input",        test_host_op_output_is_input        },
        { "view-of-op-output-is-input",     test_view_of_op_output_is_input     },
        { "graph-own-op-output-stays-copy", test_graph_own_op_output_stays_copy },
        { "view-chain-reaches-the-root",    test_view_chain_reaches_the_root    },
        { "input-flag-reaches-views",       test_input_flag_reaches_views       },
        { "kv-write-is-not-an-activation",  test_kv_write_is_not_an_activation  },
        { "reported-bytes",                 test_reported_bytes                 },
        { "leaf-without-flag-stays-copy",   test_leaf_without_flag_stays_copy   },
        { "flagged-input-is-input",         test_flagged_input_is_input         },
        { "kv-host-stays-kv-host",          test_kv_host_stays_kv_host          },
        { "weight-stays-weight",            test_weight_stays_weight            },
        { "device-stays-device",            test_device_stays_device            },
        { "control-comes-first",            test_control_comes_first            },
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
