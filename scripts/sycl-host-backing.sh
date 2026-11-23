#!/usr/bin/env bash
# sycl-host-backing.sh: the one instrument for "how much host memory do the GPUs hold".
#
# SOURCE this file, never execute it directly:
#   source "$(dirname "${BASH_SOURCE[0]}")/sycl-host-backing.sh"
# It defines functions only, sets no shell options and is safe under `set -euo pipefail`.
#
# GPU host backing = Shmem + max(0, remainder), where
#   remainder = MemTotal - (MemFree + Buffers + Cached + SwapCached + AnonPages + Slab
#                           + PageTables + KernelStack + Percpu + Hugetlb)
# An iGPU, or a test-backend-ops run, leaves its host backing in Shmem. A discrete
# card's host-USM memory (pinned expert weights) is owned by the xe driver and is in no
# /proc/meminfo field and no process's RSS: Shmem stays flat and only the remainder
# moves. The sum sees both classes. It reads /proc/meminfo alone, so it is safe while a
# SYCL process hangs. Another tenant's model held in AnonPages does not enter the sum.
# Details: docs/backend/sycl-host-safety.md, section "GPU host backing".
#
# Every function takes an optional meminfo file (default /proc/meminfo) and prints kB (or GiB)
# on stdout. All the figures of one call come from one read of the file (one awk pass), so a
# decision figure and the figure logged beside it cannot disagree. Units are GiB (kB / 1048576)
# everywhere. A missing field counts as 0, except MemTotal: without it the remainder cannot be
# computed. A file that cannot be read, or has no MemTotal, returns 1 and prints nothing on stdout.
# Callers must treat that as "do not run" and must not substitute 0.

# sycl_gpu_host_state_kb [meminfo-file]: prints "shmem_kb remainder_kb memavailable_kb" from ONE
# read. The remainder is clamped to >= 0. Every other function below is built on this one.
sycl_gpu_host_state_kb() {
    local f="${1:-/proc/meminfo}"
    [ -f "$f" ] && [ -r "$f" ] || { echo "sycl-host-backing: cannot read $f" >&2; return 1; }
    awk '
        /^MemTotal:/   { total = $2; have = 1 }
        /^Shmem:/      { shm = $2 }
        /^MemAvailable:/ { avail = $2 }
        /^MemFree:/    { used += $2 }
        /^Buffers:/    { used += $2 }
        /^Cached:/     { used += $2 }
        /^SwapCached:/ { used += $2 }
        /^AnonPages:/  { used += $2 }
        /^Slab:/       { used += $2 }
        /^PageTables:/ { used += $2 }
        /^KernelStack:/ { used += $2 }
        /^Percpu:/     { used += $2 }
        /^Hugetlb:/    { used += $2 }
        END {
            if (!have) { print "sycl-host-backing: no MemTotal in " FILENAME > "/dev/stderr"; exit 1 }
            r = total - used
            print shm + 0, (r > 0 ? r : 0), avail + 0
        }' "$f"
}

# sycl_gpu_host_parts_kb [meminfo-file]: prints "shmem_kb remainder_kb" from ONE read.
sycl_gpu_host_parts_kb() {
    local st
    st="$(sycl_gpu_host_state_kb "$@")" || return 1
    echo "${st% *}"
}

# sycl_gpu_host_remainder_kb [meminfo-file]: the unaccounted remainder in kB, clamped to >= 0.
sycl_gpu_host_remainder_kb() {
    local parts
    parts="$(sycl_gpu_host_parts_kb "$@")" || return 1
    echo "${parts#* }"
}

# sycl_gpu_host_backing_kb [meminfo-file]: Shmem + remainder in kB.
sycl_gpu_host_backing_kb() {
    local parts
    parts="$(sycl_gpu_host_parts_kb "$@")" || return 1
    echo $(( ${parts% *} + ${parts#* } ))
}

# sycl_gpu_host_backing_gib [meminfo-file]: the same figure in GiB, one decimal.
sycl_gpu_host_backing_gib() {
    local kb
    kb="$(sycl_gpu_host_backing_kb "$@")" || return 1
    awk -v kb="$kb" 'BEGIN{printf "%.1f\n", kb / 1048576}'
}

# sycl_gpu_host_backing_log_line [meminfo-file]: one log line, "Shmem: N kB HostBacking: N kB
# MemAvailable: N kB". All three come from one read. Shmem stays in it as the iGPU signature. It
# always returns 0: a figure that cannot be read prints "unreadable", so a log line never decides
# anything.
sycl_gpu_host_backing_log_line() {
    local st shm rem av
    if st="$(sycl_gpu_host_state_kb "$@" 2>/dev/null)"; then
        read -r shm rem av <<< "$st"
        echo "Shmem: $shm kB HostBacking: $((shm + rem)) kB MemAvailable: $av kB"
    else
        echo "Shmem: unreadable kB HostBacking: unreadable kB MemAvailable: unreadable kB"
    fi
}
