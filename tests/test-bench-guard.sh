#!/usr/bin/env bash
# Unit tests for scripts/bench-guard.sh preflight, against a fake sysfs tree.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GUARD="$ROOT_DIR/scripts/bench-guard.sh"
[ -x "$GUARD" ] || { echo "SKIP: bench-guard.sh not present"; exit 77; }

T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
fail=0

mk_tree() { # $1=throttle $2=act_freq
    local d="$T/sys/class/drm/card9/device/tile0/gt0/freq0"
    mkdir -p "$d/throttle"
    echo "$1" > "$d/throttle/status"
    echo "$2" > "$d/act_freq"
}
mk_meminfo() { printf 'MemAvailable: 190000000 kB\nShmem: %s kB\n' "$1" > "$T/meminfo"; }
# run_guard <pgrep-cmd> [extra guard flags...] -- forwards anything after the
# pgrep-cmd straight through to bench-guard.sh (e.g. --df-cmd), so callers
# that need an extra override don't have to spell out the whole invocation.
# Defaults --df-cmd to `true` (tmpfs=0kB) so the suite is hermetic against
# THIS host's real tmpfs usage; a later --df-cmd in "$@" overrides it, since
# bench-guard.sh's arg parser keeps the last occurrence of a repeated flag.
run_guard() {
    local pgrep="$1"; shift
    "$GUARD" --sysfs-card "$T/sys/class/drm/card9" --meminfo "$T/meminfo" \
              --pgrep-cmd "$pgrep" --max-wait 1 --df-cmd true "$@" -- true
}

# Assert an EXACT status, never merely non-zero (mirrors
# tests/test-sycl-device-guard-symmetry-policy.sh's expect_status). Usage:
#   expect_status <want> <description> -- <command...>
expect_status() {
    local want="$1" what="$2"
    shift 2
    [ "$1" = "--" ] || { echo "expect_status: expected -- before command" >&2; exit 2; }
    shift
    local rc=0
    "$@" >/dev/null 2>&1 || rc=$?
    if [ "$rc" -ne "$want" ]; then
        echo "FAIL: expected $what to exit $want, got $rc" >&2
        fail=1
    fi
}

mk_tree 1 0; mk_meminfo 3000000
expect_status 3 "throttled card must refuse" -- run_guard "false"

mk_tree 0 1800
expect_status 3 "active card (act_freq!=0) must refuse" -- run_guard "false"

mk_tree 0 0
expect_status 3 "stale GPU tenant must refuse" -- run_guard "echo 1234 llama-bench"

mk_meminfo 30000000
expect_status 3 "high Shmem must refuse" -- run_guard "false"

mk_meminfo 3000000
expect_status 0 "clean host must run" -- run_guard "false"

# tmpfs files are not GPU-BO shmem: Shmem 30 GB with 29 GB of tmpfs files must run
mk_meminfo 30000000
printf 'Filesystem 1K-blocks Used Available Use%% Mounted on\ntmpfs 33554432 29000000 4554432 87%% /tmp\n' > "$T/df.txt"
expect_status 0 "high Shmem explained by tmpfs must run" -- run_guard "false" --df-cmd "cat $T/df.txt"

# A FAILING df-cmd must be treated as tmpfs=0 -- fail closed toward the
# pre-tmpfs-subtraction behaviour, never silently zero out the ceiling check.
mk_meminfo 3000000
expect_status 0 "failing df-cmd + low Shmem must still run (tmpfs=0)" -- run_guard "false" --df-cmd false

mk_meminfo 30000000
out="$("$GUARD" --sysfs-card "$T/sys/class/drm/card9" --meminfo "$T/meminfo" --pgrep-cmd false --df-cmd false --max-wait 1 -- true 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 3 ] || { echo "FAIL: expected failing df-cmd + high Shmem to exit 3, got $rc"; fail=1; }
echo "$out" | grep -q "minus tmpfs 0 kB" || { echo "FAIL: refusal message must show 'minus tmpfs 0 kB' (got: $out)"; fail=1; }

# The clamp branch: tmpfs usage that meets or exceeds Shmem must clamp
# effective Shmem to 0 (not go negative) and emit an informational note on
# stderr, without refusing -- Shmem 3,000,000 kB is comfortably under the
# ceiling once clamped.
mk_meminfo 3000000
printf 'Filesystem 1K-blocks Used Available Use%% Mounted on\ntmpfs 8000000 5000000 3000000 63%% /tmp\n' > "$T/df-clamp.txt"
out="$(run_guard "false" --df-cmd "cat $T/df-clamp.txt" 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 0 ] || { echo "FAIL: expected clamp branch (tmpfs >= Shmem) to still run, got $rc"; fail=1; }
echo "$out" | grep -q "tmpfs used (5000000 kB) meets or exceeds Shmem (3000000 kB); effective Shmem clamped to 0" || { echo "FAIL: clamp note missing (got: $out)"; fail=1; }

# Selector-to-PCI derivation must be an EXACT match. level_zero:0,1 (and
# anything else that isn't precisely "level_zero:0" or "level_zero:1") must
# NOT glob-match one of them -- it must fall through to the explicit
# "cannot derive card" refusal. No --sysfs-card/--pci override here, so this
# exercises the real derivation branch, not the test-fixture bypass.
expect_status 3 "level_zero:0,1 selector must not derive a card" -- \
    env ONEAPI_DEVICE_SELECTOR=level_zero:0,1 "$GUARD" \
        --meminfo "$T/meminfo" --pgrep-cmd false --df-cmd true --max-wait 1 -- true

# --- Task A2: run + verdict stamping ---

# --log captures stdout+stderr behind a VALID header on a clean run.
mk_tree 0 0; mk_meminfo 3000000
"$GUARD" --sysfs-card "$T/sys/class/drm/card9" --meminfo "$T/meminfo" --pgrep-cmd "false" --df-cmd true \
         --max-wait 1 --log "$T/run.log" -- sh -c "echo bench-output" || fail=1
head -1 "$T/run.log" | grep -q "bench-guard: VALID" || { echo "FAIL: VALID stamp missing"; fail=1; }
grep -q "bench-output" "$T/run.log" || { echo "FAIL: output not captured"; fail=1; }
# The header must carry raw/tmpfs/net separately, not a bare net figure under
# the old field name -- a mutant reverting to `pre_shmem=...kB` must go RED.
head -1 "$T/run.log" | grep -qE 'pre_shmem_raw=[0-9]+kB pre_tmpfs=[0-9]+kB pre_shmem_eff=[0-9]+kB' \
    || { echo "FAIL: header missing pre_shmem_raw/pre_tmpfs/pre_shmem_eff fields"; fail=1; }
head -1 "$T/run.log" | grep -qE 'post_shmem_raw=[0-9]+kB post_tmpfs=[0-9]+kB post_shmem_eff=[0-9]+kB' \
    || { echo "FAIL: header missing post_shmem_raw/post_tmpfs/post_shmem_eff fields"; fail=1; }

# Distinct fake Shmem/tmpfs values must reach the header arithmetically
# correct: Shmem 12,000,000 kB minus tmpfs Used 4,000,000 kB = 8,000,000 kB.
mk_tree 0 0; mk_meminfo 12000000
printf 'Filesystem 1K-blocks Used Available Use%% Mounted on\ntmpfs 20000000 4000000 16000000 20%% /tmp\n' > "$T/df-header.txt"
"$GUARD" --sysfs-card "$T/sys/class/drm/card9" --meminfo "$T/meminfo" --pgrep-cmd "false" \
         --df-cmd "cat $T/df-header.txt" --max-wait 1 --log "$T/run-header.log" -- true || fail=1
head -1 "$T/run-header.log" | grep -q "pre_shmem_eff=8000000kB" \
    || { echo "FAIL: expected pre_shmem_eff=8000000kB in header (got: $(head -1 "$T/run-header.log"))"; fail=1; }

# A wrapped command that grows Shmem (rewrites the fake meminfo file mid-run)
# must stamp SUSPECT, even though the command itself succeeds.
mk_tree 0 0; mk_meminfo 3000000
"$GUARD" --sysfs-card "$T/sys/class/drm/card9" --meminfo "$T/meminfo" --pgrep-cmd "false" --df-cmd true --max-wait 1 \
         --log "$T/run2.log" -- sh -c "printf 'MemAvailable: 1 kB\nShmem: 99999999 kB\n' > '$T/meminfo'" || fail=1
head -1 "$T/run2.log" | grep -q "SUSPECT" || { echo "FAIL: Shmem growth must stamp SUSPECT"; fail=1; }

# Exit-code mirroring: bench-guard's own exit code must equal the wrapped
# command's, in BOTH the --log and no-log paths -- never the verdict.
mk_tree 0 0; mk_meminfo 3000000
rc=0
"$GUARD" --sysfs-card "$T/sys/class/drm/card9" --meminfo "$T/meminfo" --pgrep-cmd "false" --df-cmd true \
         --max-wait 1 --log "$T/run3.log" -- sh -c "exit 7" || rc=$?
[ "$rc" -eq 7 ] || { echo "FAIL: --log path must mirror wrapped command exit code (got $rc, want 7)"; fail=1; }

mk_tree 0 0; mk_meminfo 3000000
rc=0
"$GUARD" --sysfs-card "$T/sys/class/drm/card9" --meminfo "$T/meminfo" --pgrep-cmd "false" --df-cmd true \
         --max-wait 1 -- sh -c "exit 7" >/dev/null 2>&1 || rc=$?
[ "$rc" -eq 7 ] || { echo "FAIL: no-log path must mirror wrapped command exit code (got $rc, want 7)"; fail=1; }

# --budget must parse and default the timeout without breaking a clean run.
mk_tree 0 0; mk_meminfo 3000000
expect_status 0 "clean host must run with --budget set" -- \
    "$GUARD" --sysfs-card "$T/sys/class/drm/card9" --meminfo "$T/meminfo" --pgrep-cmd "false" --df-cmd true \
             --max-wait 1 --budget 5 -- true

# A fixture tree missing act_freq must refuse with a clear message, not die
# on cat's raw (unguarded) failure under `set -e` (A1-review FYI fix).
mk_tree 0 0; rm -f "$T/sys/class/drm/card9/device/tile0/gt0/freq0/act_freq"; mk_meminfo 3000000
expect_status 3 "missing act_freq sysfs must refuse cleanly" -- run_guard "false"

# --journalctl-cmd is fakeable like every other probe: a fake command that
# emits a "GT reset" line must stamp SUSPECT, even on an otherwise-clean run.
mk_tree 0 0; mk_meminfo 3000000
"$GUARD" --sysfs-card "$T/sys/class/drm/card9" --meminfo "$T/meminfo" --pgrep-cmd "false" --df-cmd true --max-wait 1 \
         --journalctl-cmd "echo kernel: xe 0000:03:00.0: GT reset triggered" \
         --log "$T/run4.log" -- true || fail=1
head -1 "$T/run4.log" | grep -q "SUSPECT" || { echo "FAIL: kernel GT-reset line must stamp SUSPECT"; fail=1; }

# Timeout kill: a wrapped command that outlives --budget must be killed
# (rc 124, mirrored by the guard) and stamped SUSPECT with the reason.
mk_tree 0 0; mk_meminfo 3000000
rc=0
"$GUARD" --sysfs-card "$T/sys/class/drm/card9" --meminfo "$T/meminfo" --pgrep-cmd "false" --df-cmd true --max-wait 1 \
         --budget 1 --log "$T/run5.log" -- sh -c "sleep 5" || rc=$?
[ "$rc" -eq 124 ] || { echo "FAIL: timeout-killed command must exit 124 (got $rc)"; fail=1; }
head -1 "$T/run5.log" | grep -q "timeout-killed:rc=124" || { echo "FAIL: timeout kill must stamp SUSPECT with timeout-killed:rc=124"; fail=1; }

# --- Live DRM/PCI derivation (llama.cpp-imns) ---
#
# Builds a fake --drm-root whose card-NUMBER order deliberately differs from
# PCI-address order (card0 -> 09:00.0, card2 -> 04:00.0), includes an
# integrated GPU (card1 -> 00:02.0) that must be excluded, and a connector
# entry (card0-DP-1) that must be ignored by name alone -- it gets a real
# `device` symlink to the SAME discrete device as card2 (0000:04:00.0), so
# only the card[0-9]+ filter (not an absent symlink) is what skips it.
# card0-DP-1 sorts lexically BEFORE card2 in glob order ('-' < '2'), which is
# exactly what caught review round 1: an earlier, unfiltered second scan for
# the sysfs card matching a derived PCI address matched card0-DP-1 first and
# bound SYSFS_CARD to a connector instead of card2. The card= assertions
# below (not just pci=) are what a regression of that would fail.
mk_drmroot() {
    local d="$T/drmroot"
    rm -rf "$d" "$T/devices"
    mkdir -p "$T/devices/0000:09:00.0/tile0/gt0/freq0/throttle"
    echo 0 > "$T/devices/0000:09:00.0/tile0/gt0/freq0/throttle/status"
    echo 0 > "$T/devices/0000:09:00.0/tile0/gt0/freq0/act_freq"
    echo 0x8086 > "$T/devices/0000:09:00.0/vendor"
    echo 0x030000 > "$T/devices/0000:09:00.0/class"

    mkdir -p "$T/devices/0000:00:02.0"
    echo 0x8086 > "$T/devices/0000:00:02.0/vendor"
    echo 0x030000 > "$T/devices/0000:00:02.0/class"

    mkdir -p "$T/devices/0000:04:00.0/tile0/gt0/freq0/throttle"
    echo 0 > "$T/devices/0000:04:00.0/tile0/gt0/freq0/throttle/status"
    echo 0 > "$T/devices/0000:04:00.0/tile0/gt0/freq0/act_freq"
    echo 0x8086 > "$T/devices/0000:04:00.0/vendor"
    echo 0x030000 > "$T/devices/0000:04:00.0/class"

    mkdir -p "$d/card0" "$d/card1" "$d/card2" "$d/card0-DP-1"
    ln -s "$T/devices/0000:09:00.0" "$d/card0/device"
    ln -s "$T/devices/0000:00:02.0" "$d/card1/device"
    ln -s "$T/devices/0000:04:00.0" "$d/card2/device"
    ln -s "$T/devices/0000:04:00.0" "$d/card0-DP-1/device"
}

mk_drmroot; mk_meminfo 3000000
env ONEAPI_DEVICE_SELECTOR=level_zero:0 "$GUARD" --drm-root "$T/drmroot" --meminfo "$T/meminfo" \
    --pgrep-cmd false --df-cmd true --max-wait 1 --log "$T/run-lz0.log" -- true || fail=1
head -1 "$T/run-lz0.log" | grep -q "pci=0000:04:00.0" \
    || { echo "FAIL: level_zero:0 must resolve to the LOWER PCI address 0000:04:00.0 (card0->09, card2->04; got: $(head -1 "$T/run-lz0.log"))"; fail=1; }
head -1 "$T/run-lz0.log" | grep -q "card=$T/drmroot/card2" \
    || { echo "FAIL: level_zero:0 must bind card=$T/drmroot/card2, not the card0-DP-1 connector (got: $(head -1 "$T/run-lz0.log"))"; fail=1; }

mk_meminfo 3000000
env ONEAPI_DEVICE_SELECTOR=level_zero:1 "$GUARD" --drm-root "$T/drmroot" --meminfo "$T/meminfo" \
    --pgrep-cmd false --df-cmd true --max-wait 1 --log "$T/run-lz1.log" -- true || fail=1
head -1 "$T/run-lz1.log" | grep -q "pci=0000:09:00.0" \
    || { echo "FAIL: level_zero:1 must resolve to the HIGHER PCI address 0000:09:00.0 (got: $(head -1 "$T/run-lz1.log"))"; fail=1; }
head -1 "$T/run-lz1.log" | grep -q "card=$T/drmroot/card0" \
    || { echo "FAIL: level_zero:1 must bind card=$T/drmroot/card0 (got: $(head -1 "$T/run-lz1.log"))"; fail=1; }

out="$(env ONEAPI_DEVICE_SELECTOR=level_zero:2 "$GUARD" --drm-root "$T/drmroot" --meminfo "$T/meminfo" \
    --pgrep-cmd false --df-cmd true --max-wait 1 -- true 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 3 ] || { echo "FAIL: level_zero:2 (out of range) must refuse with exit 3, got $rc"; fail=1; }
echo "$out" | grep -q "0000:04:00.0" && echo "$out" | grep -q "0000:09:00.0" \
    || { echo "FAIL: out-of-range refusal must name both discrete cards found (got: $out)"; fail=1; }

mkdir -p "$T/drmroot-igpu-only/card1"
ln -s "$T/devices/0000:00:02.0" "$T/drmroot-igpu-only/card1/device"
expect_status 3 "a DRM root with only an integrated GPU must refuse" -- \
    env ONEAPI_DEVICE_SELECTOR=level_zero:0 "$GUARD" --drm-root "$T/drmroot-igpu-only" --meminfo "$T/meminfo" \
        --pgrep-cmd false --df-cmd true --max-wait 1 -- true

expect_status 3 "level_zero:0,1 must not derive a card even with --drm-root set" -- \
    env ONEAPI_DEVICE_SELECTOR=level_zero:0,1 "$GUARD" --drm-root "$T/drmroot" --meminfo "$T/meminfo" \
        --pgrep-cmd false --df-cmd true --max-wait 1 -- true

# A dangling device symlink (target no longer resolves, e.g. a card
# removed or a hot-unplug race) must refuse loudly rather than silently
# excluding the card and shifting level_zero indices for the rest (review
# round 2 finding 1a). TWO cards, not one: card0's symlink is dangling,
# card1 (reusing the still-valid 0000:09:00.0 device from mk_drmroot) is
# fine -- this is what actually reproduces the reported bug. With only one
# (broken) card, "no discrete GPU found" would also refuse with exit 3,
# masking the real defect: on the pre-fix code this two-card tree instead
# silently drops card0 and hands level_zero:0 card1's address with rc=0
# and a VALID stamp (verified against the pre-round-2 guard before this
# fix landed).
rm -rf "$T/drmroot-dangling"
mkdir -p "$T/drmroot-dangling/card0" "$T/drmroot-dangling/card1"
ln -s "$T/devices/0000:99:00.0-does-not-exist" "$T/drmroot-dangling/card0/device"
ln -s "$T/devices/0000:09:00.0" "$T/drmroot-dangling/card1/device"
out="$(env ONEAPI_DEVICE_SELECTOR=level_zero:0 "$GUARD" --drm-root "$T/drmroot-dangling" --meminfo "$T/meminfo" \
    --pgrep-cmd false --df-cmd true --max-wait 1 -- true 2>&1)" && rc=0 || rc=$?
[ "$rc" -eq 3 ] || { echo "FAIL: a dangling device symlink must refuse with exit 3, got $rc (out: $out)"; fail=1; }
echo "$out" | grep -qi "dangling" || { echo "FAIL: dangling-symlink refusal must name the problem (got: $out)"; fail=1; }

# An EXISTING but UNREADABLE device/vendor file must also refuse loudly
# (review round 2 finding 1b), for the same two-card reason as above --
# skip under root, which reads any file regardless of permission bits, so
# chmod 000 would not reproduce this.
if [ "$(id -u)" -eq 0 ]; then
    echo "SKIP: unreadable-vendor-file case not reproducible as root (root bypasses permission bits)"
else
    rm -rf "$T/drmroot-unreadable-vendor" "$T/devices-unreadable"
    mkdir -p "$T/devices-unreadable/0000:04:00.0"
    echo 0x8086 > "$T/devices-unreadable/0000:04:00.0/vendor"
    chmod 000 "$T/devices-unreadable/0000:04:00.0/vendor"
    echo 0x030000 > "$T/devices-unreadable/0000:04:00.0/class"
    mkdir -p "$T/drmroot-unreadable-vendor/card0" "$T/drmroot-unreadable-vendor/card1"
    ln -s "$T/devices-unreadable/0000:04:00.0" "$T/drmroot-unreadable-vendor/card0/device"
    ln -s "$T/devices/0000:09:00.0" "$T/drmroot-unreadable-vendor/card1/device"
    out="$(env ONEAPI_DEVICE_SELECTOR=level_zero:0 "$GUARD" --drm-root "$T/drmroot-unreadable-vendor" --meminfo "$T/meminfo" \
        --pgrep-cmd false --df-cmd true --max-wait 1 -- true 2>&1)" && rc=0 || rc=$?
    chmod 644 "$T/devices-unreadable/0000:04:00.0/vendor"   # so the EXIT trap's rm -rf can clean it up
    [ "$rc" -eq 3 ] || { echo "FAIL: an unreadable device/vendor file must refuse with exit 3, got $rc (out: $out)"; fail=1; }
    echo "$out" | grep -qi "not readable" || { echo "FAIL: unreadable-vendor refusal must name the problem (got: $out)"; fail=1; }
fi

[ "$fail" -eq 0 ] && echo "OK: all preflight refusals" || exit 1
