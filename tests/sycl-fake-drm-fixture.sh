#!/usr/bin/env bash
# Shared fake-sysfs/DRM fixture builders for the SYCL bash test suites that
# exercise PCI/DRM-derived card selection (tests/test-bench-guard.sh,
# tests/test-sycl-decode-mode-capture.sh, tests/test-sycl-gpu-preflight.sh).
#
# All three suites used to carry independent hand-rolled copies of
# mk_pci_dev/mk_drmroot that had drifted to three different signatures (a
# 5-param mk_pci_dev with throttle/act_freq, a 3-param one that always wrote
# 0/0, and a 2-param one with no freq node support at all) despite building
# the identical fake PCI/DRM topology. This file is the superset of all
# three: every existing call site in every sourcing suite works against it
# unchanged (llama.cpp-blut).
#
# Requires $T (the sourcing suite's own scratch tmpdir, already created and
# trapped for cleanup) to be set before mk_drmroot is called. Source by
# BASH_SOURCE-relative path, not by an assumed cwd, e.g.:
#   # shellcheck source=sycl-fake-drm-fixture.sh
#   source "$(dirname "${BASH_SOURCE[0]}")/sycl-fake-drm-fixture.sh"

# mk_pci_dev DEVROOT PCI_ADDR [with_freq [throttle act_freq]] -- create a
# fake sysfs PCI device directory DEVROOT/PCI_ADDR with vendor=0x8086 and
# class=0x030000 (an Intel display controller, discrete or integrated
# depending on PCI_ADDR). Pass "with_freq" as the third argument to also
# populate a tile0/gt0/freq0/throttle/status + act_freq tree under it --
# needed only for a device a test expects the guard to actually SELECT and
# run against; skip it for an iGPU or any decoy the guard must
# exclude/refuse before ever deriving FREQ from it, and for any caller that
# does not care about freq nodes at all. throttle/act_freq (4th/5th args)
# default to 0/0.
mk_pci_dev() {
    local devroot="$1" addr="$2" with_freq="${3:-}" throttle="${4:-0}" act_freq="${5:-0}"
    mkdir -p "$devroot/$addr"
    echo 0x8086 > "$devroot/$addr/vendor"
    echo 0x030000 > "$devroot/$addr/class"
    if [ -n "$with_freq" ]; then
        mkdir -p "$devroot/$addr/tile0/gt0/freq0/throttle"
        echo "$throttle" > "$devroot/$addr/tile0/gt0/freq0/throttle/status"
        echo "$act_freq" > "$devroot/$addr/tile0/gt0/freq0/act_freq"
    fi
}

# mk_drmroot: two discrete cards, card-order != PCI-order (card0 -> the
# HIGHER address 09:00.0, card2 -> the LOWER address 04:00.0), an
# integrated GPU (card1 -> 00:02.0) that must be excluded, and a connector
# entry (card0-DP-1, sharing card2's device symlink) that must be ignored by
# the card[0-9]+ filter alone -- same shape in every sourcing suite. Its own
# device root ($T/devices-lz01) is private to this fixture; a caller that
# needs an independent device root builds its own rather than reusing this
# one, so reordering test cases can never turn one case into another by
# accident.
mk_drmroot() {
    local d="$T/drmroot" devroot="$T/devices-lz01"
    rm -rf "$d" "$devroot"
    mk_pci_dev "$devroot" 0000:09:00.0 with_freq
    mk_pci_dev "$devroot" 0000:00:02.0
    mk_pci_dev "$devroot" 0000:04:00.0 with_freq
    mkdir -p "$d/card0" "$d/card1" "$d/card2" "$d/card0-DP-1"
    ln -s "$devroot/0000:09:00.0" "$d/card0/device"
    ln -s "$devroot/0000:00:02.0" "$d/card1/device"
    ln -s "$devroot/0000:04:00.0" "$d/card2/device"
    ln -s "$devroot/0000:04:00.0" "$d/card0-DP-1/device"
}
