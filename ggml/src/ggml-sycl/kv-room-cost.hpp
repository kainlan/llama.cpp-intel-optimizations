#pragma once

// The KV context room's cost (llama.cpp-w013, llama.cpp-7fmt) as plain values: hold_kv_context_room
// (unified-cache.cpp) weighs every layer's room with these, and test-kv-room-cost reaches them with no device. This
// header must stay free of SYCL and unified-cache includes; unified-cache.hpp includes it.

#include "ggml.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

namespace ggml_sycl {

// The defaults of the room's four cost parameters (placement_kv_info::ctx_fill_pct, host_attn_gbps, cpu_expert_gbps
// and host_split_us). docs/backend/sycl-env-vars.md gives the measurement behind each one and the variable that
// overrides it.
constexpr double PLACEMENT_CTX_FILL_PCT_DEFAULT    = 50.0;   // percent of the requested context expected to be filled
constexpr double PLACEMENT_HOST_ATTN_GBPS_DEFAULT  = 30.0;   // GB/s at which the CPU reads a host layer's KV
constexpr double PLACEMENT_CPU_EXPERT_GBPS_DEFAULT = 40.0;   // GB/s for a routed-expert type with no measured rate
constexpr double PLACEMENT_HOST_SPLIT_US_DEFAULT   = 250.0;  // us per CPU split a host layer adds to a decode token

// llama.cpp-7fmt: one measured CPU expert rate and the measurement it came from, which the room's line prints after
// the types it priced. A later measurement replaces both together.
struct placement_cpu_expert_rate {
    double       gbps;    // 0 = not measured
    const char * source;  // the measurement: model or mix, date, ticket comment; nullptr when not measured
};

// llama.cpp-7fmt: the effective rate, in GB/s, at which a decode token reads a host-resident routed expert of `type`:
// the room's bytes times p_hit over the decode time the displaced experts added, measured end to end (ms per token
// with the room held minus a control with the same experts on the device; ticket comments c-ncj4, c-plz8). It is not
// the CPU kernel's throughput: it folds in the device waiting on the CPU leg. The catalog row of
// GGML_SYCL_PLAN_CPU_EXPERT_GBPS records each entry's host, driver, date, load and thread count.
constexpr placement_cpu_expert_rate placement_cpu_expert_rate_measured(ggml_type type) {
    switch (type) {
        case GGML_TYPE_MXFP4:
            return { 5.6, "gpt-oss-20b 2026-10-18 (c-ncj4)" };
        // Qwen3.8-Flash-Next GSQ-RCO's seven routed-expert types, measured on the B70 as one mix: its byte-weighted
        // harmonic mean is the measured rate, so each type gets the mix's rate. Another mix of them reads a rate no
        // measurement validated until per-type rates replace these entries (llama.cpp-073y).
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_Q2_0:
            return { 22.0, "qwen38-gsq-rco-mix 2026-10-18 (c-plz8)" };
        default:
            return { 0.0, nullptr };
    }
}

// llama.cpp-7fmt: the CPU splits a decode token pays for a layer whose KV the runtime demoted to the host, each a
// device drain plus synchronous scheduler copies. A layer with an indexer key cache runs SET_ROWS x3, TOP_K,
// FLASH_ATTN and a GET_ROWS on the CPU, interleaved with device ops: 3 splits (scheduler census: 36 CPU splits for
// Qwen3.8's 12 demoted attention layers). Any other layer with KV of its own runs its KV writes and FLASH_ATTN as one
// CPU run: 1 split (GPT-OSS 20B: 12 for 12 demoted layers).
constexpr uint32_t placement_host_attn_splits(bool has_indexer) {
    return has_indexer ? 3 : 1;
}

// Every rate the table holds is 0 (not measured) with no source, or finite and positive with the measurement it came
// from, since each divides and the room's line names it; a NaN, an inf or an unnamed entry fails the build.
constexpr bool placement_cpu_expert_rates_valid() {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const placement_cpu_expert_rate r = placement_cpu_expert_rate_measured(static_cast<ggml_type>(t));
        if (r.source == nullptr ? r.gbps != 0.0 : !(r.gbps > 0.0 && r.gbps <= std::numeric_limits<double>::max())) {
            return false;
        }
    }
    return true;
}

static_assert(placement_cpu_expert_rates_valid(),
              "a measured CPU expert rate must be finite and positive and name its measurement");
static_assert(PLACEMENT_HOST_SPLIT_US_DEFAULT > 0.0 && PLACEMENT_CPU_EXPERT_GBPS_DEFAULT > 0.0 &&
                  PLACEMENT_HOST_ATTN_GBPS_DEFAULT > 0.0,
              "the room's default costs must be positive");

static_assert(GGML_TYPE_COUNT < 64, "kv_room_expert_types keeps one mask bit per ggml_type and one for untyped");

// llama.cpp-7fmt: the routed-expert weights a device's room would push to the host, by type. B's CPU rate is their
// byte-weighted harmonic mean. GGML_TYPE_COUNT stands for a tensor whose type is not known.
struct kv_room_expert_types {
    size_t   bytes            = 0;    // every routed-expert byte
    size_t   unmeasured_bytes = 0;    // the bytes of the types with no measured rate
    double   measured_us      = 0.0;  // the measured types' bytes, each over its rate, in microseconds
    uint64_t measured_mask    = 0;    // bit t: type t is present and has a measured rate
    uint64_t unmeasured_mask  = 0;    // bit t: type t (GGML_TYPE_COUNT: untyped) is present and has none

    void add(ggml_type type, size_t n_bytes) {
        const bool   typed = type >= 0 && type < GGML_TYPE_COUNT;
        const double gbps  = typed ? placement_cpu_expert_rate_measured(type).gbps : 0.0;
        const int    bit   = typed ? static_cast<int>(type) : static_cast<int>(GGML_TYPE_COUNT);
        bytes += n_bytes;
        if (gbps > 0.0) {
            measured_us += static_cast<double>(n_bytes) / (gbps * 1e3);
            measured_mask |= uint64_t(1) << bit;
        } else {
            unmeasured_bytes += n_bytes;
            unmeasured_mask |= uint64_t(1) << bit;
        }
    }
};

// llama.cpp-7fmt: the rate B is read at, and whether the room is weighed at all.
struct kv_room_expert_rate {
    double               gbps       = 0.0;    // what B divides by
    bool                 priced     = true;   // false: every layer's room is refused without being weighed
    bool                 overridden = false;  // the cpu-expert override priced every type
    kv_room_expert_types types;
};

// llama.cpp-7fmt: B's rate for a device's routed experts. The cpu-expert override, when set, prices every type. Without
// it the rate is the byte-weighted harmonic mean of the measured rates, a type with none counting at cpu_expert_gbps
// (the fallback) for a context the caller named, as before. For the model's default context (the caller named no
// n_ctx) a type with no measured rate is not priced at the fallback: the room is refused for every layer, which is
// the placement a load with no requested context makes. That is the only way n_ctx_context_is_default reaches the
// decision, and it can only refuse. A device with no routed experts keeps the fallback, which a dense model never
// reads.
static inline kv_room_expert_rate kv_room_expert_rate_for(const kv_room_expert_types & types,
                                                          double                       cpu_expert_gbps,
                                                          bool                         cpu_expert_gbps_set,
                                                          bool                         n_ctx_context_is_default) {
    kv_room_expert_rate rate;
    rate.types      = types;
    rate.gbps       = cpu_expert_gbps;
    rate.overridden = cpu_expert_gbps_set;
    if (cpu_expert_gbps_set || types.bytes == 0) {
        return rate;
    }
    if (types.unmeasured_mask != 0 && n_ctx_context_is_default) {
        rate.priced = false;
        return rate;
    }
    const double read_us = types.measured_us + static_cast<double>(types.unmeasured_bytes) / (cpu_expert_gbps * 1e3);
    rate.gbps            = static_cast<double>(types.bytes) / (read_us * 1e3);
    return rate;
}

// llama.cpp-7fmt: a ggml type's name; ggml_type_name in the planner. This header calls nothing in ggml-base, so
// test-kv-room-cost links nothing.
typedef const char * (*kv_room_type_name_fn)(ggml_type type);

// The types a mask of kv_room_expert_types names, joined with "/"; GGML_TYPE_COUNT is "untyped". The unpriced room's
// WARN names the unmeasured types with it.
static inline std::string kv_room_type_names(uint64_t mask, kv_room_type_name_fn type_name) {
    std::string names;
    for (int t = 0; t <= GGML_TYPE_COUNT; ++t) {
        if (((mask >> t) & 1) == 0) {
            continue;
        }
        names += names.empty() ? "" : "/";
        names += t == GGML_TYPE_COUNT ? "untyped" : type_name(static_cast<ggml_type>(t));
    }
    return names;
}

// llama.cpp-7fmt: where B's rate came from, for the room's line: "override" when the cpu-expert override priced every
// type; otherwise each measurement after the types it priced, in type order ("iq2_xxs/iq3_xxs via qwen38-gsq-rco-mix
// 2026-10-18 (c-plz8)"), then, after "unmeasured", the types read at the fallback rate. Every measured type names its
// measurement (placement_cpu_expert_rates_valid).
static inline std::string kv_room_rate_source(const kv_room_expert_rate & rate, kv_room_type_name_fn type_name) {
    if (rate.overridden) {
        return "override";
    }
    if (rate.types.bytes == 0) {
        return "no routed experts";
    }
    std::string source;
    uint64_t    left = rate.types.measured_mask;
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        if (((left >> t) & 1) == 0) {
            continue;
        }
        const char * from = placement_cpu_expert_rate_measured(static_cast<ggml_type>(t)).source;
        uint64_t     same = 0;  // the types left that the same measurement priced, t among them
        for (int u = t; u < GGML_TYPE_COUNT; ++u) {
            const char * other = placement_cpu_expert_rate_measured(static_cast<ggml_type>(u)).source;
            if (((left >> u) & 1) != 0 && std::strcmp(other, from) == 0) {
                same |= uint64_t(1) << u;
            }
        }
        left &= ~same;
        source += source.empty() ? "" : "; ";
        source += kv_room_type_names(same, type_name) + " via " + from;
    }
    if (rate.types.unmeasured_mask != 0) {
        source += source.empty() ? "unmeasured " : "; unmeasured ";
        source += kv_room_type_names(rate.types.unmeasured_mask, type_name);
    }
    return source;
}

// llama.cpp-w013: the cells a host layer's attention is expected to read per decode token, F = n_ctx_context x
// ctx_fill_pct / 100, the percent clamped to [0, 100].
static inline uint32_t kv_room_filled_cells(uint32_t n_ctx_context, double ctx_fill_pct) {
    return static_cast<uint32_t>(static_cast<double>(n_ctx_context) * std::clamp(ctx_fill_pct, 0.0, 100.0) / 100.0);
}

// llama.cpp-w013, llama.cpp-7fmt: one layer's room, per decode token. The caller has checked both rates finite and
// positive and the split cost finite and not negative.
struct kv_room_cost_inputs {
    size_t   host_read_bytes = 0;  // the layer's KV at the expected fill: what the host reads per token
    uint32_t n_host_splits   = 0;  // the CPU splits a token pays with the layer's KV on the host
    size_t   room_bytes      = 0;  // the layer's room: the routed-expert bytes it displaces
    int      n_experts       = 0;  // routed experts per layer; 0 for a dense model
    int      n_expert_used   = 0;  // of those, the ones a token uses
    double   host_attn_gbps  = 0.0;
    double   host_split_us   = 0.0;
    double   cpu_expert_gbps = 0.0;   // kv_room_expert_rate::gbps
    bool     priced          = true;  // kv_room_expert_rate::priced
};

struct kv_room_cost_terms {
    double fill_us      = 0.0;  // A's read of the layer's KV at the expected fill
    double host_attn_us = 0.0;  // A: fill_us plus the splits, each host_split_us
    double expert_us    = 0.0;  // B. It counts every room byte as a displaced expert byte, even where the experts
                                // fit beside the room and none is displaced; weighing min(room, displaced) is a
                                // follow-up (llama.cpp-7fmt).
    bool   held         = false;
};

// One layer's room, weighed per decode token:
//   A = the host reads the layer's KV at the expected fill at host_attn_gbps, and the token pays the layer's CPU
//       splits, each host_split_us (llama.cpp-7fmt);
//   B = the room displaces as many bytes of routed experts, and a token reads each with probability
//       n_expert_used / n_expert, at the experts' CPU rate (kv_room_expert_rate_for). Uniform: first-fit displaces
//       the least popular triplets, so this is an upper bound on their hit rate.
// Held iff A > B, or always for a dense model, which displaces no expert; never for a room that is not priced. GB/s
// is 1e3 bytes per microsecond.
static inline kv_room_cost_terms kv_room_weigh(const kv_room_cost_inputs & in) {
    const bool   moe   = in.n_experts > 0 && in.n_expert_used > 0;
    const double p_hit = moe ? std::min(1.0, static_cast<double>(in.n_expert_used) / in.n_experts) : 0.0;

    kv_room_cost_terms cost;
    cost.fill_us      = static_cast<double>(in.host_read_bytes) / (in.host_attn_gbps * 1e3);
    cost.host_attn_us = cost.fill_us + static_cast<double>(in.n_host_splits) * in.host_split_us;
    cost.expert_us    = static_cast<double>(in.room_bytes) * p_hit / (in.cpu_expert_gbps * 1e3);
    cost.held         = !moe || (in.priced && cost.host_attn_us > cost.expert_us);
    return cost;
}

}  // namespace ggml_sycl
