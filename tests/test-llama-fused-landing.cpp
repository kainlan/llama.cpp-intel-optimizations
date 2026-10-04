// llama_fused_cpu_landing_is_placement(): is a fused op that landed on the CPU a placement or a capability gap?
//
// Runs the shipped classifier (src/llama-fused-landing.h) against fake devices, buffers and tensors, so no backend
// and no model is loaded. The fake device's supports_op is switchable, which is what lets a case say "the device
// declined this op" independently of "the device has no kernel for it" -- the two things the SYCL backend's
// supports_op conflates when it declines for placement (host-demoted KV, planner-on-host weights).

#include "../ggml/src/ggml-backend-impl.h"
#include "../src/llama-fused-landing.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstdio>
#include <string>

static int g_cases  = 0;
static int g_failed = 0;

static void expect(const std::string & name, bool got, bool want) {
    ++g_cases;
    if (got != want) {
        ++g_failed;
    }
    printf("%s: %s (got %s, want %s)\n", got == want ? "PASS" : "FAIL", name.c_str(), got ? "placement" : "gap",
           want ? "placement" : "gap");
}

// fake device ------------------------------------------------------------------------------------------------------

static bool g_dev_supports_op   = false;
static int  g_supports_op_calls = 0;

static bool fake_supports_op(ggml_backend_dev_t, const ggml_tensor *) {
    ++g_supports_op_calls;
    return g_dev_supports_op;
}

static ggml_backend_device g_dev = {};

// fake buffer types and buffers ------------------------------------------------------------------------------------

static const char * fake_buft_name(ggml_backend_buffer_type_t) {
    return "fake";
}

static bool fake_buft_is_host_true(ggml_backend_buffer_type_t) {
    return true;
}

static bool fake_buft_is_host_false(ggml_backend_buffer_type_t) {
    return false;
}

static void fake_free_buffer(ggml_backend_buffer_t) {}

static ggml_backend_buffer_type g_host_buft   = {};
static ggml_backend_buffer_type g_device_buft = {};

static ggml_backend_buffer_t make_buffer(bool host, ggml_backend_buffer_usage usage) {
    ggml_backend_buffer_i iface = {};
    iface.free_buffer           = fake_free_buffer;
    ggml_backend_buffer_t buf   = ggml_backend_buffer_init(host ? &g_host_buft : &g_device_buft, iface, nullptr, 64);
    ggml_backend_buffer_set_usage(buf, usage);
    return buf;
}

// capability stub --------------------------------------------------------------------------------------------------

static bool                g_capable          = false;
static int                 g_capability_calls = 0;
static ggml_backend_dev_t  g_capability_dev   = nullptr;
static const ggml_tensor * g_capability_node  = nullptr;

static bool fake_capability(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ++g_capability_calls;
    g_capability_dev  = dev;
    g_capability_node = op;
    return g_capable;
}

int main() {
    g_dev.iface.supports_op = fake_supports_op;

    g_host_buft.iface.get_name   = fake_buft_name;
    g_host_buft.iface.is_host    = fake_buft_is_host_true;
    g_device_buft.iface.get_name = fake_buft_name;
    g_device_buft.iface.is_host  = fake_buft_is_host_false;

    ggml_init_params params = {};
    params.mem_size         = 32 * ggml_tensor_overhead();
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);

    // operands: a persistent KV-like tensor, a view of it, and an activation
    ggml_tensor * kv   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
    ggml_tensor * kv_v = ggml_view_1d(ctx, kv, 8, 0);
    ggml_tensor * act  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * node = ggml_add(ctx, kv_v, act);  // src[0] = view of kv, src[1] = act

    ggml_backend_buffer_t host_any     = make_buffer(true, GGML_BACKEND_BUFFER_USAGE_ANY);
    ggml_backend_buffer_t host_compute = make_buffer(true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_buffer_t dev_any      = make_buffer(false, GGML_BACKEND_BUFFER_USAGE_ANY);
    ggml_backend_buffer_t dev_compute  = make_buffer(false, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    auto place = [&](ggml_backend_buffer_t kv_buf, ggml_backend_buffer_t act_buf) {
        kv->buffer   = kv_buf;
        kv_v->buffer = kv_buf;
        act->buffer  = act_buf;
    };

    // ---- no capability query: the operand heuristic ----
    place(dev_any, dev_compute);
    g_dev_supports_op = true;
    expect("heuristic: the device supports the node", llama_fused_cpu_landing_is_placement(&g_dev, node, nullptr),
           true);

    g_dev_supports_op = false;
    expect("heuristic: unsupported, every operand in a device buffer is a gap",
           llama_fused_cpu_landing_is_placement(&g_dev, node, nullptr), false);

    place(host_any, dev_compute);
    expect("heuristic: unsupported, the persistent operand (a view of host KV) is in host memory",
           llama_fused_cpu_landing_is_placement(&g_dev, node, nullptr), true);

    place(host_compute, host_compute);
    expect("heuristic: unsupported, operands only in host COMPUTE scratch is a gap (control)",
           llama_fused_cpu_landing_is_placement(&g_dev, node, nullptr), false);

    place(nullptr, nullptr);
    expect("heuristic: unsupported, operands without a buffer yet (a reserve graph) is a gap",
           llama_fused_cpu_landing_is_placement(&g_dev, node, nullptr), false);

    expect("a layer with no device is not a gap", llama_fused_cpu_landing_is_placement(nullptr, node, nullptr), true);

    // ---- with the capability query: exact, and supports_op is not consulted ----
    // the planner-on-host case: the weight is SYCL-owned (not a host buffer), supports_op declines it for placement
    place(dev_any, dev_compute);
    g_dev_supports_op   = false;
    g_capable           = true;
    g_supports_op_calls = 0;
    g_capability_calls  = 0;
    expect("capability: a kernel exists but supports_op declined for placement (planner-on-host weight)",
           llama_fused_cpu_landing_is_placement(&g_dev, node, fake_capability), true);
    expect("capability: it asked the capability query with the layer's device and the node",
           g_capability_calls == 1 && g_capability_dev == &g_dev && g_capability_node == node, true);
    expect("capability: supports_op was not consulted", g_supports_op_calls == 0, true);

    g_capable = false;
    expect("capability: no kernel for the op, type or shape is a gap",
           llama_fused_cpu_landing_is_placement(&g_dev, node, fake_capability), false);

    // the heuristic would say placement here (host operand), the capability answer wins: a missing kernel is a gap
    place(host_any, dev_compute);
    g_dev_supports_op = true;
    expect("capability: no kernel is a gap even when an operand is host-resident and supports_op says yes",
           llama_fused_cpu_landing_is_placement(&g_dev, node, fake_capability), false);

    g_capable = true;
    place(host_any, dev_compute);
    expect("capability: kernel exists and host KV (flash attention over demoted layers) stays placement",
           llama_fused_cpu_landing_is_placement(&g_dev, node, fake_capability), true);

    expect("capability: a layer with no device is not a gap",
           llama_fused_cpu_landing_is_placement(nullptr, node, fake_capability), true);

    ggml_backend_buffer_free(host_any);
    ggml_backend_buffer_free(host_compute);
    ggml_backend_buffer_free(dev_any);
    ggml_backend_buffer_free(dev_compute);
    ggml_free(ctx);

    printf("\n%d cases, %d failed\n", g_cases, g_failed);
    return g_failed == 0 ? 0 : 1;
}
