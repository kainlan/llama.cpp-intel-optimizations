#!/usr/bin/env bash
# bench-guard: preflight refuses to measure on a corrupted host; run+postflight
# stamps the archived log VALID/SUSPECT (perf-recovery epic, track A, tasks
# A1+A2). Refuses to run the wrapped command when the target GPU is
# throttled/active, a stale llama-cli|llama-bench|llama-completion tenant is
# already running, or Shmem is elevated (CLAUDE.md: TTM shmem OOM history).
# All host probes go through overridable roots (--sysfs-card, --meminfo,
# --pgrep-cmd) so the logic is testable without hardware.
#
# On a clean host: runs the wrapped command under `timeout -k 15 <budget>`
# (default budget 900s; --budget overrides -- load-bearing per CLAUDE.md, `-k`
# is what prevents a hung gate binary from outliving the wrapper, never remove
# it). With --log FILE, stdout+stderr are captured to FILE behind a
# VALID/SUSPECT verdict header (pre/post throttle + Shmem readings); without
# --log, the verdict is printed to stderr only. Either way bench-guard's own
# exit code always mirrors the wrapped command's, never the verdict.
#
# Postflight SUSPECT triggers: Shmem grew more than 5 GB across the run, the
# run was killed by the timeout (rc 124/137), or the kernel log shows a GT
# reset/guc_id/CAT error since the run started. A post-run throttle=1 reading
# is NOT by itself suspect -- the run's own power draw asserts it.
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
SHMEM_GROWTH_SUSPECT_KB=$((5*1024*1024))
POLL_INTERVAL=5
LOG="" BUDGET=900
while [ $# -gt 0 ]; do case "$1" in
    --sysfs-card) SYSFS_CARD="$2"; shift 2;;
    --meminfo)    MEMINFO="$2";    shift 2;;
    --pgrep-cmd)  PGREP_CMD="$2";  shift 2;;
    --max-wait)   MAX_WAIT="$2";   shift 2;;
    --pci)        PCI="$2";        shift 2;;
    --log)        LOG="$2";        shift 2;;
    --budget)     BUDGET="$2";     shift 2;;
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
# Guarded for the same reason as throttle/status above: under `set -e`, a
# bare `cat` on a fixture tree lacking act_freq would die on cat's own raw
# exit status instead of a "refuse" message (A1 review FYI).
[ -r "$FREQ/act_freq" ] || refuse "no act_freq sysfs under $FREQ"

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

pre_shmem="$(shmem_kb)"
pre_thr="$st"

# -e-safe capture: `cmd; rc=$?` would trip `set -e` the instant the wrapped
# command exits non-zero (its own exit status IS the simple command's
# status). `|| rc=$?` keeps the non-zero status without letting -e fire, and
# mirrors it in the guard's own exit below rather than the verdict.
rc=0
if [ -n "$LOG" ]; then
    tmp_out="$(mktemp)"
    timeout -k 15 "$BUDGET" "$@" >"$tmp_out" 2>&1 || rc=$?
else
    timeout -k 15 "$BUDGET" "$@" || rc=$?
fi

post_shmem="$(shmem_kb)"
post_thr="$(cat "$FREQ/throttle/status")"

verdict="VALID"
reasons=""
# Plain `if` blocks, not `&&` chains -- a trailing false `&&` under `set -e`
# would exit the script instead of just skipping the reason.
if [ $((post_shmem - pre_shmem)) -gt "$SHMEM_GROWTH_SUSPECT_KB" ]; then
    verdict="SUSPECT"
    reasons="$reasons shmem-grew:$((post_shmem - pre_shmem))kB"
fi
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    verdict="SUSPECT"
    reasons="$reasons timeout-killed:rc=$rc"
fi
# journalctl legitimately finds nothing (grep rc=1) on a clean run -- that's
# inside an `if` condition, which `set -e` already exempts from tripping.
if journalctl -k --since "10 minutes ago" --no-pager 2>/dev/null | grep -qiE 'GT reset|guc_id|CAT error'; then
    verdict="SUSPECT"
    reasons="$reasons kernel-gpu-fault"
fi

if [ -n "$LOG" ]; then
    {
        echo "# bench-guard: $verdict${reasons:+:$reasons} pre_throttle=$pre_thr post_throttle=$post_thr pre_shmem=${pre_shmem}kB post_shmem=${post_shmem}kB cmd: $*"
        cat "$tmp_out"
    } > "$LOG"
    rm -f "$tmp_out"
else
    echo "bench-guard: $verdict${reasons:+:$reasons}" >&2
fi
exit "$rc"
