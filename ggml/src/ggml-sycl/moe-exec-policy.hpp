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
    // 0 off; 1 the stage, stage-layouts and decision lines per consulted layer-ubatch, one executor line per
    // MUL_MAT_ID dispatch, one staged-route line per prompt (plus a refusals line when one was refused), and one
    // loan line when a boundary drain times out; 2 also the load-time copy probe
    int             trace          = 0;
    char            bad_policy[48] = {};  // the rejected GGML_SYCL_MOE_EXEC value, copied, for one WARN; empty if none
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
            std::snprintf(c.bad_policy, sizeof(c.bad_policy), "%s", p);
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
// micro-batch's (the first decision counts as a change, a reason change alone does not print), never per op. The
// counters cover every consulted layer-ubatch, so the teardown summary is complete whether or not a line printed.
struct moe_exec_decision_log {
    static constexpr int MAX_LAYERS = 512;  // Qwen3.8 has 48 layers, DeepSeek V4.1 61

    explicit moe_exec_decision_log(void (*s)(void * ctx, const char * line)  = nullptr,
                                   void * c                                  = nullptr,
                                   int    trace_level                        = 0,
                                   void (*ts)(void * ctx, const char * line) = nullptr) :
        sink(s),
        sink_ctx(c),
        trace(trace_level),
        trace_sink(ts) {}

    // Set only by the constructor, in this order. sink_ctx is handed to both sinks.
    void (*sink)(void *       ctx,
                 const char * line);  // the change line; tests capture through ctx; production is GGML_LOG_WARN
    void * sink_ctx;
    int    trace;                     // GGML_SYCL_MOE_EXEC_TRACE level: >= 1 prints one line per consulted layer-ubatch
    void (*trace_sink)(void * ctx, const char * line);      // production is GGML_LOG_INFO: opt-in, -lv 4 or -v
    int8_t   last[GGML_SYCL_MAX_DEVICES][MAX_LAYERS] = {};  // 0 = none yet, else moe_exec_choice
    uint32_t n_ring                                  = 0;
    uint32_t n_cpu                                   = 0;
    uint32_t n_forced                                = 0;
    size_t   bytes_streamed                          = 0;
    uint32_t n_windows = 0;  // executor windows the ring served this prompt (fed once per window by the ring's window_begin)
    uint32_t n_exposed = 0;  // of them, a layer's first UP window: copied with no earlier phase to hide under
    uint32_t n_misses  = 0;  // experts a window asked for that were not staged ahead, exposed windows excluded

    void note_window(bool exposed, uint32_t misses) {
        std::lock_guard<std::mutex> lock(mu);
        ++n_windows;
        n_exposed += exposed ? 1 : 0;
        n_misses += exposed ? 0 : misses;
    }

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
        if (trace >= 1 && trace_sink != nullptr) {
            char tline[224];
            std::snprintf(tline, sizeof(tline),
                          "[MOE-EXEC-TRACE] dev %d layer %d: %s (p=%.2f t_cpu=%.0f us t_ring=%.0f us %s)", device,
                          layer, moe_exec_choice_name(d.choice), d.p_touch, d.t_cpu_us, d.t_ring_us,
                          moe_exec_reason_name(d.reason));
            trace_sink(sink_ctx, tline);
        }
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

    // Formats the teardown line and zeroes the counters under one lock, so a note() cannot be cleared unprinted and
    // two concurrent frees cannot both print. False (buf untouched) when nothing was consulted.
    bool take_summary(char * buf, size_t cap) {
        std::lock_guard<std::mutex> lock(mu);
        if (n_ring + n_cpu == 0) {
            return false;
        }
        // clang-format off: the knobs gate pins this format as one contiguous literal
        std::snprintf(buf, cap,
                      "[MOE-EXEC] summary: %u layer-ubatches ring, %u cpu, %u forced, %zu MiB streamed, %u windows, %u exposed, %u misses",
                      n_ring, n_cpu, n_forced, bytes_streamed >> 20, n_windows, n_exposed, n_misses);
        // clang-format on
        n_ring = n_cpu = n_forced = 0;
        bytes_streamed            = 0;
        n_windows = n_exposed = n_misses = 0;
        return true;
    }

  private:
    mutable std::mutex mu;
};

}  // namespace ggml_sycl
