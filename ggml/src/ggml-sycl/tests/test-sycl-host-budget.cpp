//
// The monotone host budget (docs/design/sycl-mmap-host-tier.md §4.5, §7 C5), host-only: the binary compiles
// host-budget.hpp alone, with no device, no unified cache, no SYCL module and no ggml.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// Arms:
//   (1) every row of the §4.5 table: the headroom and the budget, or the named refusal host-headroom-exhausted;
//   (2) MemAvailable swept from 0 to MemTotal in 64 MiB steps on 32, 62 and 246.9 GiB hosts: the budget never
//       decreases, a refusal counting as 0;
//   (3) GGML_SYCL_HOST_RESERVE_MB composes: above the formula the budget is the formula's, below it the override's, and
//       at zero headroom-free memory it is the named refusal either way;
//   (4) an unknown MemTotal keeps the min(8 GiB, MemAvailable/4) reserve and stays monotone; nothing read is uncapped;
//   (5) the refusal names itself with MemAvailable, MemTotal and the headroom it compared, and a budget that is not
//       the refusal names nothing.
// Every arm names its engagement check. An arm whose engagement check fails is VOID, and a VOID arm fails the binary
// like a FAIL. Every arm runs, so one failure does not hide the rest. Each registered mutant passes only on its arm's
// FAIL line.

#include "../host-budget.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using namespace ggml_sycl;

constexpr size_t MiB = 1ull << 20;
constexpr size_t GiB = 1ull << 30;
constexpr int    PCT = 90;  // the default host-budget percentage

size_t gib(double v) {
    return static_cast<size_t>(std::llround(v * static_cast<double>(GiB)));
}

std::string fmt_gib(size_t bytes) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.2f GiB", static_cast<double>(bytes) / static_cast<double>(GiB));
    return buf;
}

int g_failures = 0;

void fail(const char * arm, const std::string & what) {
    std::printf("FAIL %s: %s\n", arm, what.c_str());
    ++g_failures;
}

void void_arm(const char * arm, const std::string & why) {
    std::printf("VOID %s: %s\n", arm, why.c_str());
    ++g_failures;
}

host_budget_result budget_of(size_t total, size_t available, bool override_set = false, size_t override_bytes = 0) {
    host_budget_inputs in;
    in.total_bytes     = total;
    in.available_bytes = available;
    in.budget_pct      = PCT;
    in.override_set    = override_set;
    in.override_bytes  = override_bytes;
    return host_budget_compute(in);
}

// The budget as the sweep orders it: a refusal grants nothing.
size_t granted(const host_budget_result & r) {
    return r.status == host_budget_status::HEADROOM_EXHAUSTED ? 0 : r.budget;
}

bool near(size_t a, size_t b) {
    return (a > b ? a - b : b - a) <= MiB;
}

// (1) The §4.5 table: MemTotal, the headroom, MemAvailable, and the new budget (a negative want is the refusal).
void arm_table() {
    const char * arm = "(1) every row of the 4.5 table";

    struct row {
        double total, headroom, available, want;
    };

    const std::vector<row> rows = {
        { 246.9, 29.0, 150.8, 121.8 },
        { 246.9, 29.0, 154.8, 125.8 },
        { 246.9, 29.0, 29.5,  0.5   },
        { 246.9, 29.0, 28.9,  -1.0  },
        { 62.0,  15.5, 50.0,  34.5  },
        { 32.0,  8.0,  20.0,  12.0  },
        { 32.0,  8.0,  7.9,   -1.0  },
    };
    // Engagement: total x 90% never binds on these rows, so each budget is the formula's MemAvailable term.
    for (const row & t : rows) {
        if (static_cast<double>(gib(t.total)) * PCT / 100.0 < static_cast<double>(gib(t.available))) {
            void_arm(arm, "a row's total x pct binds, so it does not score the MemAvailable term");
            return;
        }
    }
    for (const row & t : rows) {
        const host_budget_result r = budget_of(gib(t.total), gib(t.available));
        char                     head[128];
        std::snprintf(head, sizeof(head), "total=%.1f GiB available=%.1f GiB", t.total, t.available);
        if (!near(r.headroom, gib(t.headroom))) {
            fail(arm,
                 std::string(head) + " has headroom " + fmt_gib(r.headroom) + ", want " + fmt_gib(gib(t.headroom)));
        }
        if (t.want < 0) {
            if (r.status != host_budget_status::HEADROOM_EXHAUSTED || r.budget != 0) {
                fail(arm, std::string(head) + " gives " + fmt_gib(r.budget) + " (status " +
                              host_budget_status_name(r.status) + "), want the refusal host-headroom-exhausted");
            }
        } else if (r.status != host_budget_status::OK || !near(r.budget, gib(t.want))) {
            fail(arm, std::string(head) + " gives " + fmt_gib(r.budget) + " (status " +
                          host_budget_status_name(r.status) + "), want " + fmt_gib(gib(t.want)));
        }
    }
}

// (2) The budget never rises when MemAvailable falls: swept upward, it never decreases.
void arm_sweep() {
    const char *              arm    = "(2) the budget never decreases as MemAvailable rises";
    const std::vector<double> totals = { 32.0, 62.0, 246.9 };
    for (double t : totals) {
        const size_t total   = gib(t);
        size_t       prev    = 0;
        size_t       prev_av = 0;
        size_t       steps   = 0;
        size_t       top     = 0;
        for (size_t av = 0; av <= total; av += 64 * MiB) {
            const size_t g = granted(budget_of(total, av));
            if (steps > 0 && g < prev) {
                char buf[256];
                std::snprintf(buf, sizeof(buf), "total=%.1f GiB: available %s gives %s, available %s gives %s", t,
                              fmt_gib(prev_av).c_str(), fmt_gib(prev).c_str(), fmt_gib(av).c_str(), fmt_gib(g).c_str());
                fail(arm, buf);
            }
            prev    = g;
            prev_av = av;
            top     = g;
            ++steps;
        }
        // Engagement: the sweep visited every step and ended on a granted budget.
        if (steps != total / (64 * MiB) + 1 || top == 0) {
            char buf[128];
            std::snprintf(buf, sizeof(buf), "total=%.1f GiB: %zu steps, last budget %s", t, steps,
                          fmt_gib(top).c_str());
            void_arm(arm, buf);
        }
    }
}

// (3) The override is a cap on the formula, never a replacement of it.
void arm_override() {
    const char * arm     = "(3) the override composes";
    const size_t total   = gib(246.9);
    const size_t avail   = gib(150.8);
    const size_t formula = budget_of(total, avail).budget;
    const size_t above   = gib(200.0);
    const size_t below   = gib(50.0);
    // Engagement: the formula's value lies between the two overrides.
    if (!(below < formula && formula < above)) {
        void_arm(arm, "the formula's " + fmt_gib(formula) + " does not lie between the two overrides");
        return;
    }
    const host_budget_result hi = budget_of(total, avail, true, above);
    if (hi.budget != formula || hi.override_binds) {
        fail(arm,
             "an override above the formula gives " + fmt_gib(hi.budget) + ", want the formula's " + fmt_gib(formula));
    }
    const host_budget_result lo = budget_of(total, avail, true, below);
    if (lo.budget != below || !lo.override_binds) {
        fail(arm,
             "an override below the formula gives " + fmt_gib(lo.budget) + ", want the override's " + fmt_gib(below));
    }
    const size_t floor_avail = gib(28.9);  // at or below this host's 29 GiB headroom
    for (size_t ov : { above, gib(0.25) }) {
        const host_budget_result r = budget_of(total, floor_avail, true, ov);
        if (r.status != host_budget_status::HEADROOM_EXHAUSTED || r.budget != 0) {
            fail(arm, "an override of " + fmt_gib(ov) + " at zero headroom-free memory gives " + fmt_gib(r.budget) +
                          " (status " + host_budget_status_name(r.status) +
                          "), want the refusal host-headroom-exhausted");
        }
    }
}

// (4) An unknown MemTotal keeps its min(8 GiB, MemAvailable/4) reserve; nothing read leaves the budget uncapped.
void arm_unknown_total() {
    const char *             arm = "(4) an unknown MemTotal keeps its reserve";
    const host_budget_result r   = budget_of(0, gib(16.0));
    // Engagement: the sample is read, so the budget is a number, not the uncapped sentinel.
    if (r.status == host_budget_status::DETECTION_FAILED) {
        void_arm(arm, "a 16 GiB reading with no MemTotal is treated as no reading");
        return;
    }
    if (r.budget != gib(12.0) || r.headroom != gib(4.0)) {
        fail(arm, "available=16 GiB gives " + fmt_gib(r.budget) + " with headroom " + fmt_gib(r.headroom) +
                      ", want 12.00 GiB with 4.00 GiB");
    }
    size_t prev = 0;
    // From 64 MiB: MemAvailable 0 with no MemTotal is nothing read, scored below.
    for (size_t av = 64 * MiB; av <= 64 * GiB; av += 64 * MiB) {
        const size_t g = granted(budget_of(0, av));
        if (g < prev) {
            fail(arm,
                 "available " + fmt_gib(av) + " gives " + fmt_gib(g) + ", below the previous step's " + fmt_gib(prev));
            break;
        }
        prev = g;
    }
    const host_budget_result none = budget_of(0, 0);
    if (none.status != host_budget_status::DETECTION_FAILED || none.budget != k_host_budget_uncapped) {
        fail(arm, "nothing read gives " + fmt_gib(none.budget) + " (status " + host_budget_status_name(none.status) +
                      "), want the uncapped sentinel");
    }
}

// (5) The reason every refused line appends: the refusal's name and its three figures, or nothing.
void arm_refusal_reason() {
    const char *             arm       = "(5) the refusal names its figures";
    const size_t             total     = gib(246.9);
    const size_t             available = gib(28.9);
    const host_budget_result refused   = budget_of(total, available);
    // Engagement: the §4.5 table's (246.9, 28.9) row is the refusal.
    if (refused.status != host_budget_status::HEADROOM_EXHAUSTED) {
        void_arm(arm, "total=246.9 GiB available=28.9 GiB is not the refusal");
        return;
    }
    const std::string want = " reason=host-headroom-exhausted MemAvailable=" + std::to_string(available) +
                             " MemTotal=" + std::to_string(total) + " headroom=" + std::to_string(refused.headroom);
    const std::string got = host_headroom_refusal_reason(true, available, total, refused.headroom);
    if (got != want) {
        fail(arm, "the refusal's reason is '" + got + "', want '" + want + "'");
    }
    const std::string none = host_headroom_refusal_reason(false, gib(150.8), total, refused.headroom);
    if (!none.empty()) {
        fail(arm, "a budget that is not the refusal names '" + none + "', want nothing");
    }
}

}  // namespace

int main() {
    arm_table();
    arm_sweep();
    arm_override();
    arm_unknown_total();
    arm_refusal_reason();
    if (g_failures != 0) {
        std::printf("test-sycl-host-budget: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test-sycl-host-budget: PASS\n");
    return 0;
}
