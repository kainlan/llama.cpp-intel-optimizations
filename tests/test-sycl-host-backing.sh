#!/usr/bin/env bash
# Host-only test of scripts/sycl-host-backing.sh. Reads recorded /proc/meminfo
# snapshots; allocates nothing and touches no GPU.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIB="$ROOT_DIR/scripts/sycl-host-backing.sh"
FIX="$ROOT_DIR/tests/sycl-meminfo-fixtures"
[ -r "$LIB" ] || { echo "FAIL: $LIB is missing"; exit 1; }
# shellcheck disable=SC1090
source "$LIB"

T="$(mktemp -d)"; trap 'chmod -R u+rwX "$T" 2>/dev/null; rm -rf "$T"' EXIT
fail=0
cases=0

check_eq() { # $1=label $2=expected $3=actual
    cases=$((cases + 1))
    [ "$2" = "$3" ] || { echo "FAIL: $1: expected '$2', got '$3'"; fail=1; }
}
check_true() { # $1=label, rest = a test command
    local label="$1"; shift
    cases=$((cases + 1))
    "$@" || { echo "FAIL: $label"; fail=1; }
}

LOADED="$FIX/meminfo-qwen-loaded.txt"
IDLE="$FIX/meminfo-idle-tenant-held.txt"

# Both snapshots, to the kB. Loaded: a Qwen3.8 default-mode load on the B70.
# Idle: the same host with another tenant's 122 GiB model held in AnonPages.
check_eq "loaded sum" 68554068 "$(sycl_gpu_host_backing_kb "$LOADED")"
check_eq "idle sum" 13363740 "$(sycl_gpu_host_backing_kb "$IDLE")"
check_eq "loaded remainder" 56583232 "$(sycl_gpu_host_remainder_kb "$LOADED")"
check_eq "idle remainder" 1765960 "$(sycl_gpu_host_remainder_kb "$IDLE")"
check_eq "loaded GiB" 65.4 "$(sycl_gpu_host_backing_gib "$LOADED")"
check_eq "idle GiB" 12.7 "$(sycl_gpu_host_backing_gib "$IDLE")"

# The point of the library: Shmem alone is flat across the two snapshots, the
# sum is not.
sh_loaded="$(awk '/^Shmem:/{print $2}' "$LOADED")"
sh_idle="$(awk '/^Shmem:/{print $2}' "$IDLE")"
sh_diff=$((sh_loaded - sh_idle))
sum_diff=$(($(sycl_gpu_host_backing_kb "$LOADED") - $(sycl_gpu_host_backing_kb "$IDLE")))
check_true "Shmem differs by under 1 GiB" [ "$sh_diff" -lt 1048576 ]
check_true "sum differs by over 50 GiB" [ "$sum_diff" -gt $((50 * 1048576)) ]

# A missing field counts as 0: drop Hugetlb and Percpu (older kernels lack them).
grep -vE '^(Hugetlb|Percpu):' "$IDLE" > "$T/missing"
check_eq "missing fields count as 0" $((13363740 + 39072)) "$(sycl_gpu_host_backing_kb "$T/missing")"

# A negative remainder clamps to 0, so the sum is Shmem alone.
sed 's/^MemTotal:.*/MemTotal: 1000 kB/' "$IDLE" > "$T/neg"
check_eq "negative remainder clamps to 0" 0 "$(sycl_gpu_host_remainder_kb "$T/neg")"
check_eq "negative remainder leaves Shmem" "$sh_idle" "$(sycl_gpu_host_backing_kb "$T/neg")"

# Fail closed: an unreadable path, a directory, and a file with no MemTotal
# return non-zero and print nothing on stdout.
for p in "$T/no-such-file" "$T"; do
    out="$(sycl_gpu_host_backing_kb "$p" 2>/dev/null)" && rc=0 || rc=$?
    check_true "unreadable '$p' returns non-zero" [ "$rc" -ne 0 ]
    check_eq "unreadable '$p' prints nothing" "" "$out"
done
grep -v '^MemTotal:' "$IDLE" > "$T/no-total"
out="$(sycl_gpu_host_backing_kb "$T/no-total" 2>/dev/null)" && rc=0 || rc=$?
check_true "no MemTotal returns non-zero" [ "$rc" -ne 0 ]
check_eq "no MemTotal prints nothing" "" "$out"
out="$(sycl_gpu_host_backing_gib "$T/no-such-file" 2>/dev/null)" && rc=0 || rc=$?
check_true "gib on unreadable returns non-zero" [ "$rc" -ne 0 ]
check_eq "gib on unreadable prints nothing" "" "$out"

# The log line carries all three figures, and "unreadable" instead of a failure.
check_eq "log line" "Shmem: 11970836 kB HostBacking: 68554068 kB MemAvailable: 52242988 kB source=$LOADED" "$(sycl_gpu_host_backing_log_line "$LOADED")"
check_eq "log line, unreadable" "Shmem: unreadable kB HostBacking: unreadable kB MemAvailable: unreadable kB source=$T/no-such-file" "$(sycl_gpu_host_backing_log_line "$T/no-such-file")"

# One read gives both parts: "shmem_kb remainder_kb".
check_eq "parts, loaded" "11970836 56583232" "$(sycl_gpu_host_parts_kb "$LOADED")"
check_eq "parts, idle" "11597780 1765960" "$(sycl_gpu_host_parts_kb "$IDLE")"
out="$(sycl_gpu_host_parts_kb "$T/no-such-file" 2>/dev/null)" && rc=0 || rc=$?
check_true "parts on unreadable returns non-zero" [ "$rc" -ne 0 ]
check_eq "parts on unreadable prints nothing" "" "$out"

# The test seam: SYCL_HOST_BACKING_MEMINFO replaces /proc/meminfo as the default file, and the log
# line says so by printing source=<path>. With the default file the line carries no source.
check_eq "seam: backing" 68554068 "$(SYCL_HOST_BACKING_MEMINFO="$LOADED" sycl_gpu_host_backing_kb)"
check_eq "seam: parts" "11597780 1765960" "$(SYCL_HOST_BACKING_MEMINFO="$IDLE" sycl_gpu_host_parts_kb)"
check_eq "seam: log line names its source" \
    "Shmem: 11970836 kB HostBacking: 68554068 kB MemAvailable: 52242988 kB source=$LOADED" \
    "$(SYCL_HOST_BACKING_MEMINFO="$LOADED" sycl_gpu_host_backing_log_line)"
case "$(sycl_gpu_host_backing_log_line)" in *source=*) echo "FAIL: the default log line must not carry a source"; fail=1;; esac
cases=$((cases + 1))
out="$(SYCL_HOST_BACKING_MEMINFO="$T/no-such-file" sycl_gpu_host_backing_kb 2>/dev/null)" && rc=0 || rc=$?
check_true "seam pointing at an unreadable file fails closed" [ "$rc" -ne 0 ]

# Default file is /proc/meminfo: a number on any Linux host.
if [ -r /proc/meminfo ]; then
    case "$(sycl_gpu_host_backing_kb)" in
        ''|*[!0-9]*) echo "FAIL: default read of /proc/meminfo is not a number"; fail=1;;
    esac
    cases=$((cases + 1))
fi

[ "$fail" -eq 0 ] || exit 1
echo "OK: test-sycl-host-backing ($cases cases)"
