// The staging probe's verdict table (tests/sycl-staging-probe-verdict.h), driven over the readings that
// separate its outcomes and over every threshold edge. Host-only: no SYCL, no device.
//
// The table is the point. The probe's first two verdicts were read from device-written host-USM words (a
// release flag the gate polled, a marker the host read); discrete Battlemage gives neither a visibility
// guarantee, so the verdict is now a function of three host-clock readings against a calibrated busy time.
// Each row below names the reading it must give.

#include "sycl-staging-probe-verdict.h"

#include <cstdio>
#include <cstring>

namespace {

int g_failures = 0;

void expect(double busy, double ret, double done, staging_probe_verdict want, const char * why) {
    const staging_probe_verdict got = staging_probe_classify(busy, ret, done);
    if (got != want) {
        std::fprintf(stderr, "FAIL: busy=%.1f return=%.1f done=%.1f gave %s, want %s (%s)\n", busy, ret, done,
                     staging_probe_verdict_name(got), staging_probe_verdict_name(want), why);
        ++g_failures;
    }
}

}  // namespace

int main() {
    using V = staging_probe_verdict;
    // Unusable readings classify nothing.
    expect(0.0, 1.0, 2.0, V::UNDECIDED, "no calibrated busy time");
    expect(-5.0, 1.0, 2.0, V::UNDECIDED, "negative busy time");
    expect(400.0, -1.0, 2.0, V::UNDECIDED, "negative return time");
    expect(400.0, 10.0, 5.0, V::UNDECIDED, "done before the submit returned");

    // The three pre-registered outcomes, at busy = 400 ms.
    expect(400.0, 0.2, 405.0, V::RETURNS_BEFORE, "submit returned at once, leg 2 finished after the hold");
    expect(400.0, 399.0, 405.0, V::BLOCKS, "submit returned only when the hold ended");
    expect(400.0, 0.2, 5.0, V::VOID, "leg 2 finished long before the hold could have: it did not hold");
    expect(400.0, 5.0, 5.0, V::VOID, "returned and finished at once: void, never returns_before");

    // The threshold edges (0.50 return, 0.90 blocks, 0.90 void).
    expect(400.0, 199.9, 400.0, V::RETURNS_BEFORE, "just under half the busy time");
    expect(400.0, 200.0, 400.0, V::UNDECIDED, "exactly half: in between");
    expect(400.0, 359.9, 400.0, V::UNDECIDED, "just under 0.90: in between");
    expect(400.0, 360.0, 400.0, V::BLOCKS, "exactly 0.90 of the busy time");
    expect(400.0, 100.0, 359.9, V::VOID, "done just under 0.90 of the busy time");
    expect(400.0, 100.0, 360.0, V::RETURNS_BEFORE, "done exactly at 0.90: held");

    // The names the lead greps for.
    if (std::strcmp(staging_probe_verdict_name(V::RETURNS_BEFORE), "returns_before") != 0 ||
        std::strcmp(staging_probe_verdict_name(V::BLOCKS), "blocks") != 0 ||
        std::strcmp(staging_probe_verdict_name(V::VOID), "void") != 0 ||
        std::strcmp(staging_probe_verdict_name(V::UNDECIDED), "undecided") != 0) {
        std::fprintf(stderr, "FAIL: a verdict name changed\n");
        ++g_failures;
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("staging probe verdict table: ok\n");
    return 0;
}
