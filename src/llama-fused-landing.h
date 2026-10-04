#pragma once

// Header-only and free of llama.cpp internals on purpose: tests/test-llama-fused-landing.cpp compiles this same
// function against fake devices, buffers and tensors.

#include "ggml-backend.h"
#include "ggml.h"

// A backend's capability-only supports_op: true when it has a kernel for the node, whether or not the data is
// planned onto that device. The SYCL backend exports it as "ggml_backend_sycl_supports_op_capability".
typedef bool (*llama_fused_capability_fn)(ggml_backend_dev_t dev, const ggml_tensor * op);

// A fused op landed on the CPU although its layer is assigned to `dev_layer`: is that PLACEMENT (the designed
// outcome -- the data lives where the CPU runs it) or a CAPABILITY gap (the layer's device has no kernel for it)?
// The two read the same in the scheduler's output and need opposite remedies, so the log must say which happened.
//
// With `capability` the answer is exact: a device that has the kernel was declined for placement, and one that
// lacks it has a gap. supports_op() cannot answer this -- backends also decline an op for placement (host-demoted
// KV, planner-on-host weights), so its "false" covers both.
//
// Without it (a backend that exports no capability query) the best available evidence is used: the device
// supporting the node, or a persistent operand -- the view source, in a host buffer whose usage is not COMPUTE --
// already living in host memory, both mean placement. Compute scratch says nothing about where the data was
// planned, so it never counts. This path cannot see a planner decision that moved a weight into a device-owned
// host tier (that operand's buffer is not a host buffer), which is why a backend that has the capability query
// should export it.
static inline bool llama_fused_cpu_landing_is_placement(ggml_backend_dev_t        dev_layer,
                                                        const ggml_tensor *       node,
                                                        llama_fused_capability_fn capability) {
    if (!dev_layer) {
        return true;
    }

    if (capability) {
        return capability(dev_layer, node);
    }

    if (ggml_backend_dev_supports_op(dev_layer, node)) {
        return true;
    }

    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        const ggml_tensor * src = node->src[i];
        if (!src) {
            continue;
        }

        const ggml_tensor * data = src->view_src ? src->view_src : src;
        if (data->buffer && ggml_backend_buffer_is_host(data->buffer) &&
            ggml_backend_buffer_get_usage(data->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
            return true;
        }
    }

    return false;
}
