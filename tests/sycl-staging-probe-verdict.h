#pragma once

// The staging probe's verdict, as a pure function of three readings taken at the instant leg 2's
// submit returned. Kept out of the probe's main() so a host-only test can drive the whole table.
//
//   marker_set        the host-USM marker a kernel chained behind leg 1 writes when leg 1 completed
//   release_sent      the gate's release flag had already been sent when the submit returned. The
//                     watchdog only sends it ahead of the return when its 2 s timeout fired first, so
//                     this reads "the submit did not return until the timed release"
//   leg2_returned     the submit had returned (a reading taken before it returned classifies nothing)
//
// The three outcomes G0 pre-registers (design row 113):
//   RETURNS_BEFORE  the submit returned while the gate still held: marker unset, release not sent
//   BLOCKS          the submit returned only after the timed release: whatever the marker says, the
//                   runtime held the submitting thread until the gate opened. Leg 1 finishing right
//                   after the release is expected, so a set marker here is not a failed gate
//   VOID            the marker was set although the release had NOT been sent: leg 1 completed with
//                   the gate shut, so the gate did not hold and the run proves nothing
// UNDECIDED is the fourth value for a reading taken before the submit returned.

enum class staging_probe_verdict {
    RETURNS_BEFORE,
    BLOCKS,
    VOID,
    UNDECIDED,
};

inline staging_probe_verdict staging_probe_classify(bool marker_set, bool release_sent, bool leg2_returned) {
    if (!leg2_returned) {
        return staging_probe_verdict::UNDECIDED;
    }
    if (release_sent) {
        return staging_probe_verdict::BLOCKS;
    }
    return marker_set ? staging_probe_verdict::VOID : staging_probe_verdict::RETURNS_BEFORE;
}

inline const char * staging_probe_verdict_name(staging_probe_verdict v) {
    switch (v) {
        case staging_probe_verdict::RETURNS_BEFORE:
            return "returns_before";
        case staging_probe_verdict::BLOCKS:
            return "blocks";
        case staging_probe_verdict::VOID:
            return "void";
        case staging_probe_verdict::UNDECIDED:
            break;
    }
    return "undecided";
}
