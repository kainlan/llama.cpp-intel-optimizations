#!/usr/bin/env bash
# bench-guard: refuse to measure on a corrupted host; stamp logs VALID/SUSPECT.
# Default card mapping (this host, verify with --sysfs-card on other machines):
#   level_zero:0 = B70 = PCI 0000:03:00.0 ; level_zero:1 = B50 = PCI 0000:07:00.0
# Derivation is LIVE via the PCI device symlink, never the static card index
# (CLAUDE.md: DRM numbering moves across boots).
set -u
SYSFS_CARD="" MEMINFO=/proc/meminfo PGREP_CMD="" MAX_WAIT=360 PCI="" SELECTOR="${ONEAPI_DEVICE_SELECTOR:-}"
SHMEM_CEIL_KB=$((10*1024*1024))
while [ $# -gt 0 ]; do case "$1" in
    --sysfs-card) SYSFS_CARD="$2"; shift 2;;
    --meminfo)    MEMINFO="$2";    shift 2;;
    --pgrep-cmd)  PGREP_CMD="$2";  shift 2;;
    --max-wait)   MAX_WAIT="$2";   shift 2;;
    --pci)        PCI="$2";        shift 2;;
    --) shift; break;;
    *) echo "bench-guard: unknown arg $1" >&2; exit 2;;
esac; done
[ $# -gt 0 ] || { echo "bench-guard: no command" >&2; exit 2; }

refuse() { echo "bench-guard: REFUSED: $*" >&2; exit 3; }

if [ -z "$SYSFS_CARD" ]; then
    if [ -z "$PCI" ]; then case "$SELECTOR" in
        level_zero:0*) PCI="0000:03:00.0";;
        level_zero:1*) PCI="0000:07:00.0";;
        *) refuse "cannot derive card: set ONEAPI_DEVICE_SELECTOR, --pci, or --sysfs-card";;
    esac; fi
    for c in /sys/class/drm/card*; do
        [ "$(readlink -f "$c/device" 2>/dev/null | xargs -r basename)" = "$PCI" ] && SYSFS_CARD="$c" && break
    done
    [ -n "$SYSFS_CARD" ] || refuse "no DRM card for PCI $PCI"
fi
FREQ="$SYSFS_CARD/device/tile0/gt0/freq0"
[ -r "$FREQ/throttle/status" ] || FREQ="$SYSFS_CARD"   # test trees pass the freq0 parent directly
[ -r "$FREQ/throttle/status" ] || refuse "no throttle sysfs under $SYSFS_CARD"

tenants() {
    if [ -n "$PGREP_CMD" ]; then $PGREP_CMD 2>/dev/null; else pgrep -a -x 'llama-cli|llama-bench|llama-completion' 2>/dev/null; fi
}
t="$(tenants | grep -E 'llama' || true)"
[ -z "$t" ] || refuse "stale GPU tenant(s): $t"

shmem_kb() { awk '/^Shmem:/{print $2}' "$MEMINFO"; }
[ "$(shmem_kb)" -le "$SHMEM_CEIL_KB" ] || refuse "Shmem $(shmem_kb) kB above ceiling $SHMEM_CEIL_KB kB"

waited=0
while :; do
    st="$(cat "$FREQ/throttle/status")"; act="$(cat "$FREQ/act_freq")"
    [ "$st" = "0" ] && [ "$act" = "0" ] && break
    [ "$waited" -ge "$MAX_WAIT" ] && refuse "card busy/throttled after ${MAX_WAIT}s (throttle=$st act_freq=$act)"
    sleep 5; waited=$((waited+5))
done
exec "$@"
