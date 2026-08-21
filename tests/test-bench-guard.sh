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
run_guard() { "$GUARD" --sysfs-card "$T/sys/class/drm/card9" --meminfo "$T/meminfo" \
              --pgrep-cmd "$1" --max-wait 1 -- true; }

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

# Selector-to-PCI derivation must be an EXACT match. level_zero:0,1 (and
# anything else that isn't precisely "level_zero:0" or "level_zero:1") must
# NOT glob-match one of them -- it must fall through to the explicit
# "cannot derive card" refusal. No --sysfs-card/--pci override here, so this
# exercises the real derivation branch, not the test-fixture bypass.
expect_status 3 "level_zero:0,1 selector must not derive a card" -- \
    env ONEAPI_DEVICE_SELECTOR=level_zero:0,1 "$GUARD" \
        --meminfo "$T/meminfo" --pgrep-cmd false --max-wait 1 -- true

# --- Task A2: run + verdict stamping ---

# --log captures stdout+stderr behind a VALID header on a clean run.
mk_tree 0 0; mk_meminfo 3000000
"$GUARD" --sysfs-card "$T/sys/class/drm/card9" --meminfo "$T/meminfo" --pgrep-cmd "false" \
         --max-wait 1 --log "$T/run.log" -- sh -c "echo bench-output" || fail=1
head -1 "$T/run.log" | grep -q "bench-guard: VALID" || { echo "FAIL: VALID stamp missing"; fail=1; }
grep -q "bench-output" "$T/run.log" || { echo "FAIL: output not captured"; fail=1; }

# A wrapped command that grows Shmem (rewrites the fake meminfo file mid-run)
# must stamp SUSPECT, even though the command itself succeeds.
mk_tree 0 0; mk_meminfo 3000000
"$GUARD" --sysfs-card "$T/sys/class/drm/card9" --meminfo "$T/meminfo" --pgrep-cmd "false" --max-wait 1 \
         --log "$T/run2.log" -- sh -c "printf 'MemAvailable: 1 kB\nShmem: 99999999 kB\n' > '$T/meminfo'" || fail=1
head -1 "$T/run2.log" | grep -q "SUSPECT" || { echo "FAIL: Shmem growth must stamp SUSPECT"; fail=1; }

# Exit-code mirroring: bench-guard's own exit code must equal the wrapped
# command's, in BOTH the --log and no-log paths -- never the verdict.
mk_tree 0 0; mk_meminfo 3000000
rc=0
"$GUARD" --sysfs-card "$T/sys/class/drm/card9" --meminfo "$T/meminfo" --pgrep-cmd "false" \
         --max-wait 1 --log "$T/run3.log" -- sh -c "exit 7" || rc=$?
[ "$rc" -eq 7 ] || { echo "FAIL: --log path must mirror wrapped command exit code (got $rc, want 7)"; fail=1; }

mk_tree 0 0; mk_meminfo 3000000
rc=0
"$GUARD" --sysfs-card "$T/sys/class/drm/card9" --meminfo "$T/meminfo" --pgrep-cmd "false" \
         --max-wait 1 -- sh -c "exit 7" >/dev/null 2>&1 || rc=$?
[ "$rc" -eq 7 ] || { echo "FAIL: no-log path must mirror wrapped command exit code (got $rc, want 7)"; fail=1; }

# --budget must parse and default the timeout without breaking a clean run.
mk_tree 0 0; mk_meminfo 3000000
expect_status 0 "clean host must run with --budget set" -- \
    "$GUARD" --sysfs-card "$T/sys/class/drm/card9" --meminfo "$T/meminfo" --pgrep-cmd "false" \
             --max-wait 1 --budget 5 -- true

# A fixture tree missing act_freq must refuse with a clear message, not die
# on cat's raw (unguarded) failure under `set -e` (A1-review FYI fix).
mk_tree 0 0; rm -f "$T/sys/class/drm/card9/device/tile0/gt0/freq0/act_freq"; mk_meminfo 3000000
expect_status 3 "missing act_freq sysfs must refuse cleanly" -- run_guard "false"

[ "$fail" -eq 0 ] && echo "OK: all preflight refusals" || exit 1
