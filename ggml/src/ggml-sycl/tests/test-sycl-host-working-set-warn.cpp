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
//   (3) resident_room uses host_headroom(total), not P3's 29 GiB: at MemTotal 62 GiB, MemAvailable_pre 50 GiB,
//       expected_host 4 GiB and mapped_ws 20 GiB it does not WARN, where P3's constant leaves 17 GiB and would;
//   (4) a plan with no MemAvailable sample does not WARN, and its plan line says so.
// Every arm names its engagement check. An arm whose engagement check fails is VOID, and a VOID arm fails the binary
// like a FAIL. Every arm runs, so one failure does not hide the rest. Each registered mutant passes only on its arm's
// FAIL line.

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

const size_t k_total    = gib(246.9);
const size_t k_avail    = gib(150.8);
const size_t k_expected = gib(20.0);

// (1) The threshold: past the room it WARNs with every figure, at or under it, it does not.
void arm_threshold() {
    const char *           arm  = "(1) the WARN fires only past the room";
    const host_working_set base = host_working_set_check(k_total, k_avail, k_expected, 0);
    // Engagement: the room is a positive figure, so a byte past it and a GiB under it both exist.
    if (base.resident_room <= GiB) {
        void_arm(arm, "resident_room is " + std::to_string(base.resident_room) + ", too small to straddle");
        return;
    }
    const size_t room = base.resident_room;

    const host_working_set over = host_working_set_check(k_total, k_avail, k_expected, room + 1);
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
        const host_working_set ws = host_working_set_check(k_total, k_avail, k_expected, mapped);
        if (!host_working_set_warn_line(ws).empty()) {
            fail(arm, "mapped_ws " + std::to_string(mapped) + " at or below resident_room " +
                          std::to_string(ws.resident_room) + " printed the WARN");
        }
    }
}

// (2) The plan-report line carries the WARN's figures, and carries them when nothing warns too.
void arm_plan_line() {
    const char *           arm  = "(2) the plan line carries the WARN's figures";
    const host_working_set base = host_working_set_check(k_total, k_avail, k_expected, 0);
    const host_working_set over = host_working_set_check(k_total, k_avail, k_expected, base.resident_room + GiB);
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
    const host_working_set fits = host_working_set_check(k_total, k_avail, k_expected, GiB);
    const std::string      line = host_working_set_plan_line("[PLACEMENT]", fits);
    const std::string      miss = missing_figure(line, fits);
    if (!miss.empty()) {
        fail(arm, "the plan line of a plan that fits does not print " + miss + ": '" + line + "'");
    }
}

// (3) resident_room subtracts host_headroom(total): 15.5 GiB on a 62 GiB host, not P3's 29.
void arm_headroom() {
    const char * arm   = "(3) resident_room uses host_headroom(total)";
    const size_t total = gib(62.0);
    // Engagement: P3's constant would leave 50 - 4 - 29 = 17 GiB, under the 20 GiB mapped, so the point separates them.
    if (gib(50.0) - gib(4.0) - k_host_p3_floor >= gib(20.0)) {
        void_arm(arm, "P3's constant would not WARN at this point either");
        return;
    }
    const host_working_set ws = host_working_set_check(total, gib(50.0), gib(4.0), gib(20.0));
    if (!host_working_set_warn_line(ws).empty() || ws.headroom != gib(15.5)) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "at total=62.0 GiB available=50.0 GiB expected_host=4.0 GiB mapped_ws=20.0 GiB the WARN %s "
                      "(headroom=%zu resident_room=%zu, want headroom=%zu)",
                      ws.exceeds ? "fired" : "stayed silent", ws.headroom, ws.resident_room, gib(15.5));
        fail(arm, buf);
    }
}

// (4) No MemAvailable sample: no WARN, and the plan line names the missing sample.
void arm_no_sample() {
    const char *           arm = "(4) a plan with no sample does not WARN";
    const host_working_set ws  = host_working_set_check(k_total, 0, k_expected, gib(8.0));
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

}  // namespace

int main() {
    arm_threshold();
    arm_plan_line();
    arm_headroom();
    arm_no_sample();
    if (g_failures != 0) {
        std::printf("test-sycl-host-working-set-warn: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test-sycl-host-working-set-warn: PASS\n");
    return 0;
}
