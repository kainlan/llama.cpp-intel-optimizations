//
// The host budget and the resident working set (docs/design/sycl-mmap-host-tier.md §4.5, §7 C5): pure functions of
// the host's memory figures, so a host-only test compiles this header alone. It includes nothing from SYCL, the cache
// or ggml, reads no environment and logs nothing; its callers sample, log, and store.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// The HOST_BUDGET_MUTANT_* macros each plant one defect a registered mutant of the two host tests must kill; no
// production target defines them.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

namespace ggml_sycl {

// This host's measurement (§4.5): 9g22's 24 GiB P3 floor plus th32 §4.3's 5 GiB runtime margin.
constexpr size_t k_host_p3_floor = 29ull << 30;

// The budget a failed RAM detection leaves: uncapped, and never a value a later sum can wrap.
constexpr size_t k_host_budget_uncapped = SIZE_MAX / 4;

// The memory the budget leaves the host (§4.5): total/32, raised to this host's P3 floor but never past a quarter of
// the host, so the floor measured here does not become a universal cap on a smaller host.
inline size_t host_headroom(size_t total) {
#if defined(HOST_BUDGET_MUTANT_OLD_RESERVE)
    return std::max<size_t>(8ull << 30, total / 32);
#else
    return std::max(total / 32, std::min(k_host_p3_floor, total / 4));
#endif
}

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

// The budget's inputs: one sample's MemTotal and MemAvailable, the host-budget percentage, the pinned share and the
// bytes already charged (both 0 for the pinned pool's own cap), and an accepted GGML_SYCL_HOST_RESERVE_MB in bytes.
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
    bool               override_binds = false;  // the override is the smaller term, so it is the budget
};

namespace detail {
inline size_t host_budget_sat_sub(size_t a, size_t b) {
    return a > b ? a - b : 0;
}

inline size_t host_budget_sat_add(size_t a, size_t b) {
    return a > SIZE_MAX - b ? SIZE_MAX : a + b;
}
}  // namespace detail

// The one host budget (§4.5): min(total x pct - pinned, committed + MemAvailable - host_headroom(total)) while
// MemAvailable exceeds the headroom, and the refusal host-headroom-exhausted otherwise, which grants nothing past
// `committed`. It never rises when MemAvailable falls. An unknown MemTotal keeps the reserve min(8 GiB,
// MemAvailable/4) against MemAvailable itself; nothing read leaves it uncapped. The override is a cap: the budget is
// min(override, formula), so it can lower the budget and never raise it, and the refusal stands either way.
inline host_budget_result host_budget_compute(const host_budget_inputs & in) {
    host_budget_result r;
    if (in.total_bytes == 0 && in.available_bytes == 0) {
        r.status   = host_budget_status::DETECTION_FAILED;
        r.computed = k_host_budget_uncapped;
    } else {
        size_t pct_cap = 0;
        if (in.total_bytes > 0) {
            const size_t pct = static_cast<size_t>(std::min(std::max(in.budget_pct, 1), 100));
            pct_cap          = in.total_bytes / 100 * pct + in.total_bytes % 100 * pct / 100;
            r.headroom       = host_headroom(in.total_bytes);
        } else {
            pct_cap    = in.available_bytes;
            r.headroom = std::min<size_t>(8ull << 30, in.available_bytes / 4);
        }
        size_t headroom_free = detail::host_budget_sat_sub(in.available_bytes, r.headroom);
        if (in.available_bytes <= r.headroom) {
#if defined(HOST_BUDGET_MUTANT_CLIFF)
            headroom_free = in.available_bytes / 2;
#else
            r.status = host_budget_status::HEADROOM_EXHAUSTED;
#endif
        }
        const size_t share_term = detail::host_budget_sat_sub(pct_cap, in.pinned_bytes);
        const size_t sample_term =
            std::min(k_host_budget_uncapped, detail::host_budget_sat_add(in.committed_bytes, headroom_free));
        r.computed = std::min(share_term, sample_term);
    }
    r.budget = r.computed;
#if defined(HOST_BUDGET_MUTANT_OVERRIDE_REPLACES)
    if (in.override_set) {
#else
    if (in.override_set && in.override_bytes < r.computed) {
#endif
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
