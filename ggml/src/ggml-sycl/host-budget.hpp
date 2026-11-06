//
// The host budget and the resident working set (docs/design/sycl-mmap-host-tier.md §4.5, §7 C5): pure functions of
// the host's memory figures, so a host-only test compiles this header alone. It includes nothing from SYCL, the cache
// or ggml, reads no environment and logs nothing; its callers sample, log, and store.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

namespace ggml_sycl {

// P3's floor on this host (9g22's 24 GiB plus th32 §4.3's 5 GiB runtime margin): the only resident-room rule the
// tree has before §4.5.
constexpr size_t k_host_p3_floor = 29ull << 30;

// The budget a failed RAM detection leaves: uncapped, and never a value a later sum can wrap.
constexpr size_t k_host_budget_uncapped = SIZE_MAX / 4;

enum class host_budget_status : uint8_t {
    OK                 = 0,
    HEADROOM_EXHAUSTED = 1,  // MemAvailable at or below the headroom: the named refusal host-headroom-exhausted
    DETECTION_FAILED   = 2,  // MemTotal and MemAvailable both unread: the budget is uncapped
};

inline const char * host_budget_status_name(host_budget_status status) {
    switch (status) {
        case host_budget_status::OK:
            return "ok";
        case host_budget_status::HEADROOM_EXHAUSTED:
            return "host-headroom-exhausted";
        case host_budget_status::DETECTION_FAILED:
            return "detection-failed";
    }
    return "unknown";
}

// The budget's inputs: one sample's MemTotal and MemAvailable, the host-budget percentage, and an accepted
// GGML_SYCL_HOST_RESERVE_MB in bytes.
struct host_budget_inputs {
    size_t total_bytes     = 0;
    size_t available_bytes = 0;
    int    budget_pct      = 100;
    size_t pinned_bytes    = 0;
    size_t committed_bytes = 0;
    bool   override_set    = false;
    size_t override_bytes  = 0;
};

struct host_budget_result {
    size_t             budget         = 0;      // the budget the caller stores
    size_t             computed       = 0;      // the formula's value, before the override
    size_t             headroom       = 0;      // the memory the budget leaves the host
    host_budget_status status         = host_budget_status::OK;
    bool               override_binds = false;  // the override is the budget
};

// The [HOST-ARENA] formula as the pinned pool's constructor computes it: min(total x pct, MemAvailable - max(8 GiB,
// total/32)), with the cap MemAvailable/2 when MemAvailable is at or below that reserve; an accepted override is the
// whole budget.
inline host_budget_result host_budget_compute(const host_budget_inputs & in) {
    host_budget_result r;
    size_t             budget = 0;
    if (in.total_bytes > 0) {
        const int pct = std::min(std::max(in.budget_pct, 1), 100);
        budget        = static_cast<size_t>(static_cast<double>(in.total_bytes) * (static_cast<double>(pct) / 100.0));
        r.headroom    = std::max<size_t>(8ull << 30, in.total_bytes / 32);
    } else if (in.available_bytes > 0) {
        budget     = in.available_bytes;
        r.headroom = std::min<size_t>(8ull << 30, in.available_bytes / 4);
    } else {
        budget   = k_host_budget_uncapped;
        r.status = host_budget_status::DETECTION_FAILED;
    }
    if (in.available_bytes > 0 && r.headroom > 0) {
        const size_t available_cap =
            in.available_bytes > r.headroom ? in.available_bytes - r.headroom : in.available_bytes / 2;
        budget = std::min(budget, available_cap);
    }
    r.computed = budget;
    r.budget   = budget;
    if (in.override_set) {
        r.budget         = in.override_bytes;
        r.override_binds = true;
    }
    return r;
}

// The resident working set (§4.5): resident_room = MemAvailable_pre - expected_host - the headroom, and mapped_ws, the
// host-executed mapped bytes the kernel may drop. exceeds is the WARN. A locked mapping counts as any other.
struct host_working_set {
    bool   sampled           = false;  // MemAvailable_pre is a reading
    size_t mem_available_pre = 0;
    size_t expected_host     = 0;
    size_t headroom          = 0;
    size_t resident_room     = 0;
    size_t mapped_ws         = 0;
    bool   exceeds           = false;
};

inline host_working_set host_working_set_check(size_t total_bytes,
                                               size_t mem_available_pre,
                                               size_t expected_host,
                                               size_t mapped_ws,
                                               bool   locked) {
    (void) total_bytes;
    (void) locked;
    host_working_set ws;
    ws.sampled           = mem_available_pre != 0;
    ws.mem_available_pre = mem_available_pre;
    ws.expected_host     = expected_host;
    ws.headroom          = k_host_p3_floor;
    ws.mapped_ws         = mapped_ws;
    const size_t charged = expected_host + ws.headroom;
    ws.resident_room     = mem_available_pre > charged ? mem_available_pre - charged : 0;
    ws.exceeds           = ws.mapped_ws > ws.resident_room;
    return ws;
}

// The working set's figures, in the one order every line that prints them uses.
inline std::string host_working_set_figures(const host_working_set & ws) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "mapped_ws=%zu resident_room=%zu (MemAvailable_pre=%zu expected_host=%zu headroom=%zu)", ws.mapped_ws,
                  ws.resident_room, ws.mem_available_pre, ws.expected_host, ws.headroom);
    return buf;
}

// The WARN, or an empty string when the mapped bytes fit.
inline std::string host_working_set_warn_line(const host_working_set & ws) {
    if (!ws.exceeds) {
        return std::string();
    }
    return "[HOST-PLAN] mapped host weights exceed the page cache that can stay resident: " +
           host_working_set_figures(ws) + "\n";
}

// The plan report's working-set line: the plan's Totals carry no working-set figures.
inline std::string host_working_set_plan_line(const char * tag, const host_working_set & ws) {
    (void) tag;
    (void) ws;
    return std::string();
}

}  // namespace ggml_sycl
