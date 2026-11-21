// What a graph prestage does with one source tensor (prestage-class.hpp).
//
// Host-only: no SYCL queue, no device, no model. The rule is a pure function of facts the backend gathers, and the
// view-root walk reads only ggml_tensor fields, so both are tested here on synthetic tensors.
//
// The case this exists for: under spread expert placement the plan has no CPU work, the CPU backend's compute buffer
// is the generic host buft, and the scheduler makes no device copy of an activation the CPU produces. Qwen's mask ADD
// is such an activation: a SYCL split reads it from host memory. Classed COPY, the prestage's cache arm refused it and
// the decode graph was never recorded. An op output in host memory is another split's per-token activation, so it
// takes the INPUT arm (stable staging, re-copied on every call) unless an earlier class claims it.

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
        { "host-op-output-is-input",      test_host_op_output_is_input      },
        { "view-of-op-output-is-input",   test_view_of_op_output_is_input   },
        { "leaf-without-flag-stays-copy", test_leaf_without_flag_stays_copy },
        { "flagged-input-is-input",       test_flagged_input_is_input       },
        { "kv-host-stays-kv-host",        test_kv_host_stays_kv_host        },
        { "weight-stays-weight",          test_weight_stays_weight          },
        { "device-stays-device",          test_device_stays_device          },
        { "control-comes-first",          test_control_comes_first          },
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
