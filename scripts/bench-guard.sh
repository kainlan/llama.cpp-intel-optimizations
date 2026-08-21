#!/usr/bin/env bash
# bench-guard: preflight-only measurement guard (perf-recovery epic, track A,
# task A1). Refuses to run the wrapped command when the target GPU is
# throttled/active, a stale llama-cli|llama-bench|llama-completion tenant is
# already running, or Shmem is elevated (CLAUDE.md: TTM shmem OOM history).
# All host probes go through overridable roots (--sysfs-card, --meminfo,
# --pgrep-cmd) so the logic is testable without hardware.
#
# Scope of THIS commit: preflight only. On a clean host it execs the wrapped
# command and exits with its status, unmodified. Run+postflight verdict
# stamping (VALID/SUSPECT headers on an archived --log) is task A2, landing
# on top of this file next -- not a gap here.
#
# Default card mapping (this host, verify with --sysfs-card on other machines):
#   level_zero:0 = B70 = PCI 0000:03:00.0 ; level_zero:1 = B50 = PCI 0000:07:00.0
# Derivation is LIVE via the PCI device symlink, never the static card index
# (CLAUDE.md: DRM numbering moves across boots). The selector match below is
# EXACT ("level_zero:0", not a "level_zero:0*" glob) so a multi-device or
# other-form selector (level_zero:0,1, level_zero:gpu, ...) falls through to
# the explicit "cannot derive card" refusal instead of silently picking one
# of the cards it names.
set -euo pipefail
SYSFS_CARD="" MEMINFO=/proc/meminfo PGREP_CMD="" MAX_WAIT=360 PCI="" SELECTOR="${ONEAPI_DEVICE_SELECTOR:-}"
SHMEM_CEIL_KB=$((10*1024*1024))
POLL_INTERVAL=5
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
        level_zero:0) PCI="0000:03:00.0";;
        level_zero:1) PCI="0000:07:00.0";;
        *) refuse "cannot derive card: ONEAPI_DEVICE_SELECTOR must be exactly level_zero:0 or level_zero:1 (got '$SELECTOR'); otherwise pass --pci or --sysfs-card";;
    esac; fi
    for c in /sys/class/drm/card*; do
        [ "$(readlink -f "$c/device" 2>/dev/null | xargs -r basename)" = "$PCI" ] && SYSFS_CARD="$c" && break
    done
    [ -n "$SYSFS_CARD" ] || refuse "no DRM card for PCI $PCI"
fi
FREQ="$SYSFS_CARD/device/tile0/gt0/freq0"
[ -r "$FREQ/throttle/status" ] || refuse "no throttle sysfs under $FREQ"

tenants() {
    if [ -n "$PGREP_CMD" ]; then $PGREP_CMD 2>/dev/null; else pgrep -a -x 'llama-cli|llama-bench|llama-completion' 2>/dev/null; fi
}
t="$(tenants | grep -E 'llama' || true)"
[ -z "$t" ] || refuse "stale GPU tenant(s): $t"

shmem_kb() { awk '/^Shmem:/{print $2}' "$MEMINFO"; }
[ "$(shmem_kb)" -le "$SHMEM_CEIL_KB" ] || refuse "Shmem $(shmem_kb) kB above ceiling $SHMEM_CEIL_KB kB"

# Poll throttle/act_freq up to --max-wait, checking the deadline BEFORE each
# sleep and capping each sleep to the time actually remaining -- so
# --max-wait 1 refuses in ~1s, not one full POLL_INTERVAL late.
waited=0
while :; do
    st="$(cat "$FREQ/throttle/status")"; act="$(cat "$FREQ/act_freq")"
    [ "$st" = "0" ] && [ "$act" = "0" ] && break
    [ "$waited" -ge "$MAX_WAIT" ] && refuse "card busy/throttled after ${MAX_WAIT}s (throttle=$st act_freq=$act)"
    remaining=$((MAX_WAIT - waited))
    interval=$POLL_INTERVAL
    [ "$remaining" -lt "$interval" ] && interval="$remaining"
    sleep "$interval"
    waited=$((waited + interval))
done
exec "$@"
