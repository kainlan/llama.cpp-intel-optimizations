// The staging probe's verdict table (tests/sycl-staging-probe-verdict.h), driven over every
// (marker x release_sent x leg2_returned) combination. Host-only: no SYCL, no device.
//
// The table is the point. The probe's original verdict tested the marker first, so a submit that
// blocked until the 2 s release, with leg 1 then finishing, read "void" or "blocks" depending on a
// race between the marker kernel and the return. Each row below names the reading it must give.

#include "sycl-staging-probe-verdict.h"

#include <cstdio>
#include <cstring>

namespace {

int g_failures = 0;

void expect(bool marker, bool release, bool returned, staging_probe_verdict want, const char * why) {
    const staging_probe_verdict got = staging_probe_classify(marker, release, returned);
    if (got != want) {
        std::fprintf(stderr, "FAIL: marker=%d release_sent=%d leg2_returned=%d gave %s, want %s (%s)\n", marker ? 1 : 0,
                     release ? 1 : 0, returned ? 1 : 0, staging_probe_verdict_name(got),
                     staging_probe_verdict_name(want), why);
        ++g_failures;
    }
}

}  // namespace

int main() {
    using V = staging_probe_verdict;
    // A reading taken before the submit returned classifies nothing, whatever the flags say.
    for (int m = 0; m < 2; ++m) {
        for (int r = 0; r < 2; ++r) {
            expect(m != 0, r != 0, false, V::UNDECIDED, "not returned");
        }
    }
    expect(false, false, true, V::RETURNS_BEFORE, "gate held, submit returned first");
    expect(true, false, true, V::VOID, "marker set with the release unsent: the gate did not hold");
    expect(false, true, true, V::BLOCKS, "returned only after the timed release, leg 1 not yet marked");
    expect(true, true, true, V::BLOCKS, "returned after the timed release and leg 1 finished: still blocks, not void");

    // The names the lead greps for.
    if (std::strcmp(staging_probe_verdict_name(V::RETURNS_BEFORE), "returns_before") != 0 ||
        std::strcmp(staging_probe_verdict_name(V::BLOCKS), "blocks") != 0 ||
        std::strcmp(staging_probe_verdict_name(V::VOID), "void") != 0) {
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
