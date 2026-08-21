#!/usr/bin/env bash
# Unit tests for scripts/bench-guard.sh preflight, against a fake sysfs tree.
set -u
GUARD="$(dirname "$0")/../scripts/bench-guard.sh"
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

mk_tree 1 0; mk_meminfo 3000000
run_guard "false" >/dev/null 2>&1; [ $? -eq 3 ] || { echo "FAIL: throttled must refuse"; fail=1; }
mk_tree 0 1800; run_guard "false" >/dev/null 2>&1; [ $? -eq 3 ] || { echo "FAIL: active card must refuse"; fail=1; }
mk_tree 0 0; run_guard "echo 1234 llama-bench" >/dev/null 2>&1; [ $? -eq 3 ] || { echo "FAIL: tenant must refuse"; fail=1; }
mk_meminfo 30000000; run_guard "false" >/dev/null 2>&1; [ $? -eq 3 ] || { echo "FAIL: high Shmem must refuse"; fail=1; }
mk_meminfo 3000000; run_guard "false" >/dev/null 2>&1; [ $? -eq 0 ] || { echo "FAIL: clean host must run"; fail=1; }
[ $fail -eq 0 ] && echo "OK: all preflight refusals" || exit 1
