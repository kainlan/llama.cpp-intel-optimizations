#pragma once

// The MoE executor's knobs and its once-per-change decision log (llama.cpp-36j0 Task 4). Plain values, no SYCL:
// test-moe-exec-policy pins them with no device. The decision itself is moe-exec-cost.hpp; this header only says
// what the user asked for (GGML_SYCL_MOE_EXEC, GGML_SYCL_MOE_EXEC_TRACE) and prints a layer's choice when it changes.

#include "ggml-sycl.h"  // GGML_SYCL_MAX_DEVICES
#include "ggml.h"
#include "moe-exec-cost.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace ggml_sycl {

// The one place each variable is spelled. The sycl_env_settings table and the parser both name them through these
// constants, so a rename or a typo cannot give the table and the parser different variables.
static constexpr const char * MOE_EXEC_ENV_POLICY = "GGML_SYCL_MOE_EXEC";
static constexpr const char * MOE_EXEC_ENV_TRACE  = "GGML_SYCL_MOE_EXEC_TRACE";

struct moe_exec_config {
    moe_exec_policy policy = moe_exec_policy::AUTO;
    int             trace  = 0;  // 0 off, 1 one INFO line per layer per micro-batch, 2 also the load-time copy probe
    const char *    bad_policy =
        nullptr;                 // the rejected GGML_SYCL_MOE_EXEC value (points into the environment), for one WARN
};

// getenv_fn is injected so a test needs no real environment. The ring has a fixed two regions, so there is no
// ring-size knob (owner ruling B, llama.cpp-36j0 c-3bde).
static inline moe_exec_config moe_exec_config_parse(moe_exec_getenv_fn getenv_fn) {
    moe_exec_config c;
    const char *    p = getenv_fn ? getenv_fn(MOE_EXEC_ENV_POLICY) : nullptr;
    if (p != nullptr && std::strcmp(p, "auto") != 0) {
        if (std::strcmp(p, "cpu") == 0) {
            c.policy = moe_exec_policy::CPU;
        } else if (std::strcmp(p, "ring") == 0) {
            c.policy = moe_exec_policy::RING;
        } else {
            c.bad_policy = p;
        }
    }
    const char * t = getenv_fn ? getenv_fn(MOE_EXEC_ENV_TRACE) : nullptr;
    if (t != nullptr) {
        char *     end   = nullptr;
        const long level = std::strtol(t, &end, 10);
        if (end != t && level > 0) {
            c.trace = level > 2 ? 2 : static_cast<int>(level);
        }
    }
    return c;
}

static inline const char * moe_exec_choice_name(moe_exec_choice c) {
    return c == moe_exec_choice::RING ? "ring" : "cpu";
}

static inline const char * moe_exec_reason_name(moe_exec_reason r) {
    switch (r) {
        case moe_exec_reason::FORCED:
            return "forced";
        case moe_exec_reason::CHEAPER:
            return "cheaper";
        case moe_exec_reason::RING_UNAVAILABLE:
            return "ring-unavailable";
        case moe_exec_reason::DECODE:
            return "decode";
        case moe_exec_reason::NO_HOST_EXPERTS:
            return "no-host-experts";
        default:
            return "none";
    }
}

// Remembers each (device, layer)'s last choice and prints a WARN line only when it differs from the previous
// micro-batch's (the first decision counts as a change), never per op. The counters cover every consulted
// layer-ubatch, so the teardown summary is complete whether or not a line printed.
struct moe_exec_decision_log {
    static constexpr int MAX_LAYERS = 512;  // Qwen3.8 has 48 layers, DeepSeek V4.1 61

    explicit moe_exec_decision_log(void (*s)(void * ctx, const char * line) = nullptr, void * c = nullptr) :
        sink(s),
        sink_ctx(c) {}

    void (*sink)(void * ctx, const char * line) = nullptr;  // tests capture through ctx; production is GGML_LOG_WARN
    void *   sink_ctx                           = nullptr;
    int8_t   last[GGML_SYCL_MAX_DEVICES][MAX_LAYERS] = {};  // 0 = none yet, else moe_exec_choice
    uint32_t n_ring                                  = 0;
    uint32_t n_cpu                                   = 0;
    uint32_t n_forced                                = 0;
    size_t   bytes_streamed                          = 0;

    // bytes is what this layer-ubatch streams through the ring (0 for a CPU choice).
    void note(int device, int layer, const moe_exec_decision & d, size_t bytes) {
        if (d.choice == moe_exec_choice::NOT_CONSULTED) {
            return;
        }
        GGML_ASSERT(device >= 0 && device < GGML_SYCL_MAX_DEVICES);
        GGML_ASSERT(layer >= 0 && layer < MAX_LAYERS);
        std::lock_guard<std::mutex> lock(mu);
        const bool                  ring = d.choice == moe_exec_choice::RING;
        if (ring) {
            ++n_ring;
            bytes_streamed += bytes;
        } else {
            ++n_cpu;
        }
        n_forced += d.reason == moe_exec_reason::FORCED ? 1 : 0;
        const int8_t now = static_cast<int8_t>(d.choice);
        if (last[device][layer] == now) {
            return;
        }
        last[device][layer] = now;
        char line[192];
        std::snprintf(line, sizeof(line), "[MOE-EXEC] layer %d: %s (p=%.2f t_cpu=%.0f us t_ring=%.0f us %s)", layer,
                      moe_exec_choice_name(d.choice), d.p_touch, d.t_cpu_us, d.t_ring_us,
                      moe_exec_reason_name(d.reason));
        if (sink != nullptr) {
            sink(sink_ctx, line);
        }
    }

    uint32_t layer_ubatches() const { return n_ring + n_cpu; }

    void summary(char * buf, size_t cap) const {
        std::snprintf(buf, cap, "[MOE-EXEC] summary: %u layer-ubatches ring, %u cpu, %u forced, %zu MiB streamed",
                      n_ring, n_cpu, n_forced, bytes_streamed >> 20);
    }

    void reset_counters() {
        std::lock_guard<std::mutex> lock(mu);
        n_ring = n_cpu = n_forced = 0;
        bytes_streamed            = 0;
    }

  private:
    std::mutex mu;
};

}  // namespace ggml_sycl
