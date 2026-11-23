//
// The resident working-set WARN (docs/design/sycl-mmap-host-tier.md §4.5, §7 C5), host-only: the binary compiles
// host-budget.hpp alone, with no device, no unified cache, no SYCL module and no ggml.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// Arms:
//   (1) the WARN prints, with its figures, when mapped_ws exceeds resident_room, and not when it does not: one byte
//       past the room, at the room, a GiB under it, and nothing mapped;
//   (2) the plan-report line carries the same figures as the WARN, warned or not;
//   (3) resident_room uses the budget's headroom, not P3's 29 GiB: at MemTotal 62 GiB, MemAvailable_pre 50 GiB,
//       expected_host 4 GiB and mapped_ws 20 GiB it does not WARN, where P3's constant leaves 17 GiB and would; with
//       MemTotal unread the headroom is the budget's min(8 GiB, MemAvailable/4);
//   (4) a plan with no MemAvailable sample does not WARN, and its plan line says so.
//   (5) a locked mapping (-lm mmap+mlock) is charged and cannot be dropped: mapped_ws is 0, nothing WARNs, and the
//       plan line says locked.
// Every arm computes the working set as a plan does: the headroom of the budget computed from the sample, fed to
// host_working_set_check_headroom with that sample's MemAvailable. Every arm names its engagement check. An arm whose
// engagement check fails is VOID, and a VOID arm fails the binary like a FAIL. Every arm runs, so one failure does not
// hide the rest. Each registered mutant passes only on its arm's FAIL line.

#include "../host-budget.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>

namespace {

using namespace ggml_sycl;

constexpr size_t GiB = 1ull << 30;

size_t gib(double v) {
    return static_cast<size_t>(std::llround(v * static_cast<double>(GiB)));
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

bool contains(const std::string & hay, const std::string & needle) {
    return hay.find(needle) != std::string::npos;
}

// Every figure the WARN and the plan line print, each as name=value.
std::string missing_figure(const std::string & line, const host_working_set & ws) {
    const std::pair<const char *, size_t> figures[] = {
        { "mapped_ws=",        ws.mapped_ws         },
        { "resident_room=",    ws.resident_room     },
        { "MemAvailable_pre=", ws.mem_available_pre },
        { "expected_host=",    ws.expected_host     },
        { "headroom=",         ws.headroom          },
    };
    for (const auto & f : figures) {
        if (!contains(line, std::string(f.first) + std::to_string(f.second))) {
            return std::string(f.first) + std::to_string(f.second);
        }
    }
    return std::string();
}

// The working set a plan computes (design 5efe §4.5): placement_plan_log_working_set passes its planner input's
// headroom, which is host_budget_compute's for the sample the plan's budget came from.
host_working_set working_set_of(size_t total, size_t available, size_t expected_host, size_t mapped, bool locked) {
    host_budget_inputs in;
    in.total_bytes        = total;
    in.available_bytes    = available;
    const size_t headroom = host_budget_compute(in).headroom;
    return host_working_set_check_headroom(headroom, available, expected_host, mapped, locked);
}

const size_t k_total    = gib(246.9);
const size_t k_avail    = gib(150.8);
const size_t k_expected = gib(20.0);

// (1) The threshold: past the room it WARNs with every figure, at or under it, it does not.
void arm_threshold() {
    const char *           arm  = "(1) the WARN fires only past the room";
    const host_working_set base = working_set_of(k_total, k_avail, k_expected, 0, false);
    // Engagement: the room is a positive figure, so a byte past it and a GiB under it both exist.
    if (base.resident_room <= GiB) {
        void_arm(arm, "resident_room is " + std::to_string(base.resident_room) + ", too small to straddle");
        return;
    }
    const size_t room = base.resident_room;

    const host_working_set over = working_set_of(k_total, k_avail, k_expected, room + 1, false);
    const std::string      warn = host_working_set_warn_line(over);
    if (warn.empty()) {
        fail(arm, "mapped_ws " + std::to_string(room + 1) + " past resident_room " + std::to_string(room) +
                      " printed no WARN");
    } else {
        const std::string prefix = "[HOST-PLAN] mapped host weights exceed the page cache that can stay resident: ";
        if (warn.compare(0, prefix.size(), prefix) != 0) {
            fail(arm, "the WARN does not open with the design's text: " + warn);
        }
        const std::string missing = missing_figure(warn, over);
        if (!missing.empty()) {
            fail(arm, "the WARN does not print " + missing + ": " + warn);
        }
    }
    const size_t unders[] = { room, room - GiB, 0 };
    for (size_t mapped : unders) {
        const host_working_set ws = working_set_of(k_total, k_avail, k_expected, mapped, false);
        if (!host_working_set_warn_line(ws).empty()) {
            fail(arm, "mapped_ws " + std::to_string(mapped) + " at or below resident_room " +
                          std::to_string(ws.resident_room) + " printed the WARN");
        }
    }
}

// (2) The plan-report line carries the WARN's figures, and carries them when nothing warns too.
void arm_plan_line() {
    const char *           arm  = "(2) the plan line carries the WARN's figures";
    const host_working_set base = working_set_of(k_total, k_avail, k_expected, 0, false);
    const host_working_set over = working_set_of(k_total, k_avail, k_expected, base.resident_room + GiB, false);
    const std::string      warn = host_working_set_warn_line(over);
    // Engagement: the WARN fires on this plan, so it has figures to compare.
    if (warn.empty()) {
        void_arm(arm, "the over-room plan printed no WARN to compare with");
        return;
    }
    const std::string figures = host_working_set_figures(over);
    const char *      tags[]  = { "[PLACEMENT]", "[PLACEMENT-MULTI]" };
    for (const char * tag : tags) {
        const std::string line = host_working_set_plan_line(tag, over);
        if (!contains(warn, figures) || !contains(line, figures)) {
            fail(arm,
                 std::string(tag) + " plan line '" + line + "' does not carry the WARN's figures '" + figures + "'");
        }
        if (line.compare(0, std::string(tag).size(), tag) != 0) {
            fail(arm, std::string(tag) + " plan line does not open with its tag: '" + line + "'");
        }
    }
    const host_working_set fits = working_set_of(k_total, k_avail, k_expected, GiB, false);
    const std::string      line = host_working_set_plan_line("[PLACEMENT]", fits);
    const std::string      miss = missing_figure(line, fits);
    if (!miss.empty()) {
        fail(arm, "the plan line of a plan that fits does not print " + miss + ": '" + line + "'");
    }
}

// (3) resident_room subtracts the budget's headroom: 15.5 GiB on a 62 GiB host, not P3's 29, and 8 GiB with MemTotal
// unread.
void arm_headroom() {
    const char * arm   = "(3) resident_room uses the budget's headroom";
    const size_t total = gib(62.0);
    // Engagement: P3's constant would leave 50 - 4 - 29 = 17 GiB, under the 20 GiB mapped, so the point separates them.
    if (gib(50.0) - gib(4.0) - k_host_p3_floor >= gib(20.0)) {
        void_arm(arm, "P3's constant would not WARN at this point either");
        return;
    }
    const host_working_set ws = working_set_of(total, gib(50.0), gib(4.0), gib(20.0), false);
    if (!host_working_set_warn_line(ws).empty() || ws.headroom != gib(15.5)) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "at total=62.0 GiB available=50.0 GiB expected_host=4.0 GiB mapped_ws=20.0 GiB the WARN %s "
                      "(headroom=%zu resident_room=%zu, want headroom=%zu)",
                      ws.exceeds ? "fired" : "stayed silent", ws.headroom, ws.resident_room, gib(15.5));
        fail(arm, buf);
    }
    // MemTotal unread: the budget keeps min(8 GiB, MemAvailable/4), so the room is 50 - 4 - 8 = 38 GiB.
    const host_working_set unread = working_set_of(0, gib(50.0), gib(4.0), gib(20.0), false);
    if (!host_working_set_warn_line(unread).empty() || unread.headroom != gib(8.0)) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "at MemTotal unread available=50.0 GiB expected_host=4.0 GiB mapped_ws=20.0 GiB the WARN %s "
                      "(headroom=%zu resident_room=%zu, want headroom=%zu)",
                      unread.exceeds ? "fired" : "stayed silent", unread.headroom, unread.resident_room, gib(8.0));
        fail(arm, buf);
    }
}

// (4) No MemAvailable sample: no WARN, and the plan line names the missing sample.
void arm_no_sample() {
    const char *           arm = "(4) a plan with no sample does not WARN";
    const host_working_set ws  = working_set_of(k_total, 0, k_expected, gib(8.0), false);
    // Engagement: the plan maps bytes, so a WARN had something to report.
    if (ws.mapped_ws == 0) {
        void_arm(arm, "the plan maps nothing");
        return;
    }
    if (!host_working_set_warn_line(ws).empty()) {
        fail(arm, "a plan with MemAvailable_pre=0 printed the WARN");
    }
    const std::string line = host_working_set_plan_line("[PLACEMENT]", ws);
    if (!contains(line, "no-sample")) {
        fail(arm, "the plan line of a plan with no sample does not say no-sample: '" + line + "'");
    }
}

// (5) A locked mapping is charged, so it is not working set the kernel may drop: mapped_ws 0, no WARN, check=locked.
void arm_locked() {
    const char *           arm      = "(5) a locked mapping is not working set";
    const host_working_set unlocked = working_set_of(k_total, k_avail, k_expected, gib(200.0), false);
    // Engagement: the same bytes unlocked exceed the room, so a WARN had something to report.
    if (!unlocked.exceeds) {
        void_arm(arm, "200 GiB unlocked does not exceed the room");
        return;
    }
    const host_working_set ws   = working_set_of(k_total, k_avail, k_expected, gib(200.0), true);
    const std::string      line = host_working_set_plan_line("[PLACEMENT]", ws);
    if (ws.mapped_ws != 0 || !host_working_set_warn_line(ws).empty() || !contains(line, "check=locked")) {
        fail(arm, "200 GiB locked gives mapped_ws " + std::to_string(ws.mapped_ws) + ", WARN " +
                      (host_working_set_warn_line(ws).empty() ? "silent" : "printed") + ", plan line '" + line +
                      "', want mapped_ws 0, no WARN and check=locked");
    }
}

}  // namespace

int main() {
    arm_threshold();
    arm_plan_line();
    arm_headroom();
    arm_no_sample();
    arm_locked();
    if (g_failures != 0) {
        std::printf("test-sycl-host-working-set-warn: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test-sycl-host-working-set-warn: PASS\n");
    return 0;
}
