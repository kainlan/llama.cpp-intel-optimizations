#pragma once

// The staging probe's verdict, as a pure function of three host-clock readings (milliseconds from the
// instant the calibrated busy kernel was submitted), so a host-only test can drive the whole table.
//
//   busy_ms     the calibrated busy kernel's measured run time alone (the hold on leg 1)
//   return_ms   when leg 2's submit returned
//   done_ms     when leg 2 had completed
//
// Leg 1 is ordered behind the busy kernel on the source queue, so leg 1 cannot complete before about busy_ms,
// and leg 2 depends on leg 1. The outcomes G0 pre-registers (design row 113):
//   RETURNS_BEFORE  return_ms < k_returns_before_frac * busy_ms: the submit returned while leg 1 was still
//                   held, so it did not wait for it
//   BLOCKS          return_ms >= k_blocks_frac * busy_ms: the submit returned only about when leg 1 could
//                   have completed, so the runtime held the submitting thread for the dependency
//   VOID            done_ms < k_void_done_frac * busy_ms: leg 2 completed before the busy kernel could have
//                   finished, so leg 1 was not held behind it and the run proves nothing
//   UNDECIDED       in between, or an unusable reading (no busy time, or done before return): not a verdict
// VOID is tested first, so a hold that did not hold never reads as returns_before.
//
// No reading here is a device write observed by the host and none is a device read of a word the host
// writes: discrete Battlemage has no usm_atomic_host_allocations, so neither has a visibility guarantee.

constexpr double k_staging_probe_returns_before_frac = 0.50;
constexpr double k_staging_probe_blocks_frac         = 0.90;
constexpr double k_staging_probe_void_done_frac      = 0.90;

enum class staging_probe_verdict {
    RETURNS_BEFORE,
    BLOCKS,
    VOID,
    UNDECIDED,
};

inline staging_probe_verdict staging_probe_classify(double busy_ms, double return_ms, double done_ms) {
    if (!(busy_ms > 0.0) || return_ms < 0.0 || done_ms < return_ms) {
        return staging_probe_verdict::UNDECIDED;
    }
    if (done_ms < k_staging_probe_void_done_frac * busy_ms) {
        return staging_probe_verdict::VOID;
    }
    if (return_ms < k_staging_probe_returns_before_frac * busy_ms) {
        return staging_probe_verdict::RETURNS_BEFORE;
    }
    if (return_ms >= k_staging_probe_blocks_frac * busy_ms) {
        return staging_probe_verdict::BLOCKS;
    }
    return staging_probe_verdict::UNDECIDED;
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
