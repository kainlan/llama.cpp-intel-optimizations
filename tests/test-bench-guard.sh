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

[ "$fail" -eq 0 ] && echo "OK: all preflight refusals" || exit 1
