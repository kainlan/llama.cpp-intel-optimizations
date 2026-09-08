#!/usr/bin/env bash
# Unit tests for scripts/sycl-gpu-preflight.sh's live B50 PCI-address
# derivation (llama.cpp-o4fs). Before this fix, sycl_preflight_b50_sysfs_bad
# hardcoded /sys/bus/pci/devices/0000:07:00.0 for the B50 -- stale against
# the current boot's 0000:09:00.0 address (the discrete cards moved at the
# 2026-09-05 boot; see CLAUDE.md) -- and had no way to be exercised against
# a fake sysfs tree at all, so every case below is RED against the pre-fix
# script: SYCL_PREFLIGHT_DRM_ROOT/SYCL_PREFLIGHT_PCI_ROOT did not exist, so
# sycl_preflight_b50_pci_address (itself new) is undefined, and
# sycl_preflight_b50_sysfs_bad ignored both env vars entirely and always
# consulted the real host's /sys/bus/pci/devices/0000:07:00.0.
#
# Fixture style mirrors tests/test-bench-guard.sh's own mk_pci_dev/mk_drmroot
# (card-order != PCI-order, a connector entry that must be ignored) -- kept
# as a local copy per this project's existing convention of each test file
# owning its own fixture builders (test-sycl-decode-mode-capture.sh does the
# same for its own --drm-root coverage).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFLIGHT="$ROOT_DIR/scripts/sycl-gpu-preflight.sh"
[ -f "$PREFLIGHT" ] || { echo "SKIP: sycl-gpu-preflight.sh not present"; exit 77; }

# shellcheck disable=SC1090,SC1091
source "$PREFLIGHT"

T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
fail=0

mk_pci_dev() { # $1=devroot $2=addr [with_freq]
    local devroot="$1" addr="$2" with_freq="${3:-}"
    mkdir -p "$devroot/$addr"
    echo 0x8086 > "$devroot/$addr/vendor"
    echo 0x030000 > "$devroot/$addr/class"
    if [ -n "$with_freq" ]; then
        mkdir -p "$devroot/$addr/tile0/gt0/freq0/throttle"
        echo 0 > "$devroot/$addr/tile0/gt0/freq0/throttle/status"
        echo 0 > "$devroot/$addr/tile0/gt0/freq0/act_freq"
    fi
}

# mk_drmroot: two discrete cards, card-order != PCI-order (card0 -> the
# HIGHER address 09:00.0, card2 -> the LOWER address 04:00.0), an
# integrated GPU (card1 -> 00:02.0) that must be excluded, and a connector
# entry (card0-DP-1, sharing card2's device symlink) that must be ignored
# by the card[0-9]+ filter alone -- same shape as
# tests/test-bench-guard.sh's own mk_drmroot fixture.
mk_drmroot() {
    local d="$T/drmroot" devroot="$T/devices"
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

# mk_drmroot_single: only ONE discrete card -- a single-GPU host, so
# index 1 (the B50) is out of range and sycl_preflight_b50_pci_address
# must fail to derive anything.
mk_drmroot_single() {
    local d="$T/drmroot-single" devroot="$T/devices-single"
    rm -rf "$d" "$devroot"
    mk_pci_dev "$devroot" 0000:04:00.0 with_freq
    mkdir -p "$d/card0"
    ln -s "$devroot/0000:04:00.0" "$d/card0/device"
}

mk_pci_state() { # $1=pciroot $2=addr $3=enable $4=power_state [$5=runtime_status]
    local root="$1" addr="$2" enable="$3" pstate="$4" rstatus="${5:-}"
    mkdir -p "$root/$addr/power"
    printf '%s\n' "$enable" > "$root/$addr/enable"
    printf '%s\n' "$pstate" > "$root/$addr/power_state"
    [ -z "$rstatus" ] || printf '%s\n' "$rstatus" > "$root/$addr/power/runtime_status"
}

# --- sycl_preflight_b50_pci_address: derivation on a card-order != PCI-order
# fixture, mirroring bench-guard.sh's own level_zero:0/level_zero:1 checks ---

mk_drmroot
# `|| true` on every sycl_preflight_b50_pci_address call below: a derivation
# FAILURE is an expected, checked outcome here (asserted via the empty
# result), never a script-ending error -- without this, `set -e` treats
# `var=$(cmd)` the same as a bare failing command and exits the WHOLE test
# script silently the instant a derivation-failure case runs, before the
# corresponding FAIL check ever gets a chance to fire (confirmed directly:
# omitting `|| true` here made this file exit 1 with no output at all).
got="$(SYCL_PREFLIGHT_DRM_ROOT="$T/drmroot" sycl_preflight_b50_pci_address)" || true
[ "$got" = "0000:09:00.0" ] \
    || { echo "FAIL: sycl_preflight_b50_pci_address must derive 0000:09:00.0 (the HIGHER PCI address; card0->09, card2->04) for index 1 (B50), got '$got'"; fail=1; }

# --- out-of-range: a single-discrete-GPU fixture cannot derive a B50 ---

mk_drmroot_single
got_single="$(SYCL_PREFLIGHT_DRM_ROOT="$T/drmroot-single" sycl_preflight_b50_pci_address)" || true
[ -z "$got_single" ] \
    || { echo "FAIL: sycl_preflight_b50_pci_address must return empty on a single-discrete-GPU fixture (out of range for index 1), got '$got_single'"; fail=1; }

# --- sycl_preflight_b50_sysfs_bad: cannot-derive case treated as bad, same
# conservative stance the old fixed-address form took toward a genuinely
# missing sysfs entry ([[ -e "$b50" ]] || return 0) ---

if SYCL_PREFLIGHT_DRM_ROOT="$T/drmroot-single" sycl_preflight_b50_sysfs_bad; then
    :
else
    echo "FAIL: sycl_preflight_b50_sysfs_bad must report bad (return 0) when no second discrete card can be derived"
    fail=1
fi

# --- sycl_preflight_b50_sysfs_bad: a GOOD B50 (enabled, D0) at the derived
# address must NOT be reported bad ---

mk_pci_state "$T/pciroot-good" 0000:09:00.0 1 D0
if SYCL_PREFLIGHT_DRM_ROOT="$T/drmroot" SYCL_PREFLIGHT_PCI_ROOT="$T/pciroot-good" sycl_preflight_b50_sysfs_bad; then
    echo "FAIL: sycl_preflight_b50_sysfs_bad must NOT report bad for an enabled, D0, non-suspended B50"
    fail=1
fi

# --- sycl_preflight_b50_sysfs_bad: a BAD B50 (disabled, D3hot, NOT
# suspended) at the derived address must be reported bad ---

mk_pci_state "$T/pciroot-bad" 0000:09:00.0 0 D3hot active
if SYCL_PREFLIGHT_DRM_ROOT="$T/drmroot" SYCL_PREFLIGHT_PCI_ROOT="$T/pciroot-bad" sycl_preflight_b50_sysfs_bad; then
    :
else
    echo "FAIL: sycl_preflight_b50_sysfs_bad must report bad for a disabled, non-D0, non-suspended B50"
    fail=1
fi

# --- sycl_preflight_b50_sysfs_bad: idle-suspend exemption still applies once
# the address is live-derived -- runtime_status=suspended must NOT be bad
# even with enable=0/power_state=D3hot (runtime PM parking an idle card) ---

mk_pci_state "$T/pciroot-suspended" 0000:09:00.0 0 D3hot suspended
if SYCL_PREFLIGHT_DRM_ROOT="$T/drmroot" SYCL_PREFLIGHT_PCI_ROOT="$T/pciroot-suspended" sycl_preflight_b50_sysfs_bad; then
    echo "FAIL: sycl_preflight_b50_sysfs_bad must NOT report bad for a runtime-suspended B50 (idle-suspend exemption)"
    fail=1
fi

# --- keep existing behaviour when the selector cannot use the B50:
# sycl_preflight_selector_may_use_b50 is unchanged by this fix and must
# still gate a B70-only selector away from the B50 check entirely ---

if sycl_preflight_selector_may_use_b50 "level_zero:0"; then
    echo "FAIL: sycl_preflight_selector_may_use_b50 must return false for a B70-only selector (level_zero:0)"
    fail=1
fi
if ! sycl_preflight_selector_may_use_b50 "level_zero:1"; then
    echo "FAIL: sycl_preflight_selector_may_use_b50 must return true for level_zero:1"
    fail=1
fi

[ "$fail" -eq 0 ] && echo "OK: sycl-gpu-preflight B50 live derivation" || exit 1
