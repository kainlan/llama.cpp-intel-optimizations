#!/usr/bin/env bash
# bench-guard: preflight refuses to measure on a corrupted host; run+postflight
# stamps the archived log VALID/SUSPECT (perf-recovery epic, track A, tasks
# A1+A2). Refuses to run the wrapped command when the target GPU is
# throttled/active, a stale llama-cli|llama-bench|llama-completion tenant is
# already running, or Shmem is elevated (CLAUDE.md: TTM shmem OOM history).
# All host probes go through overridable roots (--sysfs-card, --meminfo,
# --pgrep-cmd, --journalctl-cmd, --df-cmd -- the last three are test hooks,
# same as --meminfo/--sysfs-card, so the logic is testable without hardware).
# The Shmem ceiling check compares Shmem net of tmpfs file usage (summed
# "Used" from `df -k -t tmpfs`, override with --df-cmd): ordinary tmpfs files
# (/tmp, /dev/shm) count toward Shmem in /proc/meminfo alongside TTM GPU-BO
# backing, and the ceiling exists for the latter, not the former. If tmpfs
# usage meets or exceeds Shmem (swap-backed tmpfs pages, `none`-fstype rows
# `df` counts that Shmem doesn't, etc.) effective Shmem clamps to 0 rather
# than going negative, and a one-line note is printed to stderr (once, at
# preflight) so the clamp is never silent. The archived --log header stamps
# raw Shmem, tmpfs used, and the net figure separately at each sample point,
# never just the net.
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
# Card derivation is LIVE from DRM/PCI enumeration under --drm-root (default
# /sys/class/drm), never a fixed PCI address table (CLAUDE.md: DRM numbering
# AND PCI bus addresses move across boots -- verified 2026-09-05, when the
# discrete cards moved from 0000:03:00.0/07:00.0 to 0000:04:00.0/09:00.0).
# derive_pci_for_selector() below enumerates top-level cards (card[0-9]+
# only -- connector entries like card1-DP-1 are excluded by name), keeps
# discrete Intel display controllers (device/vendor 0x8086, device/class
# 0x0300*, PCI bus != 0000:00 so the integrated GPU is excluded), sorts the
# survivors by PCI address (lexical order on the zero-padded dddd:bb:dd.f
# string is bus/device/function order), and maps level_zero:N to the N-th
# survivor, zero-based. This assumes Level Zero orders the two discrete
# cards by ascending PCI address -- true on every boot observed so far (the
# B70 is always the lower address, i.e. level_zero:0). --pci ADDR and
# --sysfs-card DIR remain explicit overrides that bypass the derivation
# entirely. The selector match is EXACT (level_zero:<digit>, a single
# digit) so a multi-device or other-form selector (level_zero:0,1,
# level_zero:gpu, ...) falls through to the explicit "cannot derive card"
# refusal instead of silently picking one of the cards it names.
set -euo pipefail
SYSFS_CARD="" MEMINFO=/proc/meminfo PGREP_CMD="" DF_CMD="" MAX_WAIT=360 PCI="" SELECTOR="${ONEAPI_DEVICE_SELECTOR:-}" DRM_ROOT=/sys/class/drm
SHMEM_CEIL_KB=$((10*1024*1024))
SHMEM_GROWTH_SUSPECT_KB=$((5*1024*1024))
POLL_INTERVAL=5
LOG="" BUDGET=900 JOURNALCTL_CMD=""
while [ $# -gt 0 ]; do case "$1" in
    --sysfs-card)      SYSFS_CARD="$2";     shift 2;;
    --meminfo)         MEMINFO="$2";        shift 2;;
    --pgrep-cmd)       PGREP_CMD="$2";      shift 2;;
    --df-cmd)          DF_CMD="$2";         shift 2;;
    --max-wait)        MAX_WAIT="$2";       shift 2;;
    --pci)             PCI="$2";            shift 2;;
    --drm-root)        DRM_ROOT="$2";       shift 2;;
    --log)             LOG="$2";            shift 2;;
    --budget)          BUDGET="$2";         shift 2;;
    --journalctl-cmd)  JOURNALCTL_CMD="$2"; shift 2;;
    --) shift; break;;
    *) echo "bench-guard: unknown arg $1" >&2; exit 2;;
esac; done
[ $# -gt 0 ] || { echo "bench-guard: no command" >&2; exit 2; }

refuse() { echo "bench-guard: REFUSED: $*" >&2; exit 3; }

# derive_pci_for_selector IDX -- enumerate top-level DRM cards under
# $DRM_ROOT, keep discrete Intel display controllers, sort by PCI address,
# and print the IDX-th (zero-based) survivor's PCI address on stdout.
# Calls refuse() (exit 3) directly on any failure. Safe to invoke via a
# plain command-substitution assignment (PCI="$(derive_pci_for_selector
# ...)") even though that runs this function in a subshell: under `set -e`
# a non-zero exit status from a command-substitution assignment still
# aborts the whole script (unlike a pipeline stage, which pipefail is
# needed for), so refuse()'s "exit 3" propagates rather than being
# swallowed -- verified empirically, not folklore.
derive_pci_for_selector() {
    local idx="$1" c base pci vendor class
    local -a entries=()
    for c in "$DRM_ROOT"/card*; do
        [ -e "$c" ] || continue
        base="$(basename "$c")"
        [[ "$base" =~ ^card[0-9]+$ ]] || continue   # skip connectors (card1-DP-1, ...)
        pci="$(readlink -f "$c/device" 2>/dev/null | xargs -r basename)"
        [ -n "$pci" ] || continue
        [ -r "$c/device/vendor" ] || continue
        [ -r "$c/device/class" ] || continue
        vendor="$(cat "$c/device/vendor" 2>/dev/null || true)"
        [ "$vendor" = "0x8086" ] || continue
        class="$(cat "$c/device/class" 2>/dev/null || true)"
        case "$class" in 0x0300*) : ;; *) continue;; esac
        case "$pci" in 0000:00:*) continue;; esac    # exclude the integrated GPU
        entries+=("$base=$pci")
    done
    local -a sorted=()
    if [ "${#entries[@]}" -gt 0 ]; then
        # Lexical sort on field 2 (the PCI address) is bus/device/function
        # order for the zero-padded dddd:bb:dd.f form; force LC_ALL=C so a
        # non-C locale cannot reorder it.
        mapfile -t sorted < <(printf '%s\n' "${entries[@]}" | LC_ALL=C sort -t '=' -k2,2)
    fi
    local n="${#sorted[@]}"
    [ "$n" -gt 0 ] || refuse "no discrete Intel GPU display controller found under $DRM_ROOT"
    [ "$idx" -lt "$n" ] || refuse "level_zero:$idx is out of range; found $n discrete Intel GPU(s): ${sorted[*]}"
    printf '%s\n' "${sorted[$idx]}" | cut -d= -f2
}

if [ -z "$SYSFS_CARD" ]; then
    if [ -z "$PCI" ]; then
        case "$SELECTOR" in
            level_zero:[0-9]) : ;;
            *) refuse "cannot derive card: ONEAPI_DEVICE_SELECTOR must be exactly level_zero:<digit> (got '$SELECTOR'); otherwise pass --pci or --sysfs-card";;
        esac
        PCI="$(derive_pci_for_selector "${SELECTOR#level_zero:}")"
    fi
    for c in "$DRM_ROOT"/card*; do
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
# tmpfs_kb sits to the LEFT of a pipe (piped into awk): each pipeline stage
# runs in its own subshell, so a refuse()/exit called from inside DF_CMD (or
# a future rewrite of this helper) would only exit that subshell -- it would
# be swallowed, never reach the top-level script. Do not add refuse() here;
# keep this helper pure (probe in, number out).
tmpfs_kb() { { if [ -n "$DF_CMD" ]; then $DF_CMD 2>/dev/null; else df -k -t tmpfs 2>/dev/null; fi; } | awk 'NR>1{s+=$3} END{print s+0}'; }

# Sample raw Shmem + tmpfs usage ONCE and derive the effective figure, into
# the three SAMPLE_* globals -- never re-derive individually. Callers: the
# preflight ceiling check, and pre/post around the wrapped command. The clamp
# note (below) is printed only at the FIRST call (preflight), not at every
# sample point -- one run should print it at most once, not up to three
# times for the same underlying host condition.
SAMPLE_RAW=0 SAMPLE_TMPFS=0 SAMPLE_EFF=0
sample_shmem() {
    SAMPLE_RAW="$(shmem_kb)"
    # `|| true`: tmpfs_kb's internal pipe can return non-zero under pipefail
    # when DF_CMD itself fails (e.g. a fake `false` in tests) even though the
    # trailing awk still emits a valid "0" -- this is a *plain* assignment
    # (unlike the old `[ "$(eff_shmem_kb)" -le ... ]` form), so under set -e
    # a bare non-zero status here would abort the whole script instead of
    # just leaving SAMPLE_TMPFS at the awk-emitted fallback value.
    SAMPLE_TMPFS="$(tmpfs_kb)" || true
    if [ "$SAMPLE_RAW" -gt "$SAMPLE_TMPFS" ]; then
        SAMPLE_EFF=$((SAMPLE_RAW - SAMPLE_TMPFS))
    else
        SAMPLE_EFF=0
    fi
}

sample_shmem
raw_shmem="$SAMPLE_RAW" tmpfs_used="$SAMPLE_TMPFS" eff_shmem="$SAMPLE_EFF"
if [ "$tmpfs_used" -ge "$raw_shmem" ]; then
    echo "bench-guard: note: tmpfs used (${tmpfs_used} kB) meets or exceeds Shmem (${raw_shmem} kB); effective Shmem clamped to 0" >&2
fi
[ "$eff_shmem" -le "$SHMEM_CEIL_KB" ] || refuse "Shmem $raw_shmem kB minus tmpfs $tmpfs_used kB = $eff_shmem kB above ceiling $SHMEM_CEIL_KB kB"

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

sample_shmem
pre_shmem_raw="$SAMPLE_RAW" pre_tmpfs="$SAMPLE_TMPFS" pre_shmem_eff="$SAMPLE_EFF"
pre_thr="$st"

# -e-safe capture: `cmd; rc=$?` would trip `set -e` the instant the wrapped
# command exits non-zero (its own exit status IS the simple command's
# status). `|| rc=$?` keeps the non-zero status without letting -e fire, and
# mirrors it in the guard's own exit below rather than the verdict.
rc=0
if [ -n "$LOG" ]; then
    tmp_out="$(mktemp)"
    trap 'rm -f "$tmp_out"' EXIT
    timeout -k 15 "$BUDGET" "$@" >"$tmp_out" 2>&1 || rc=$?
else
    timeout -k 15 "$BUDGET" "$@" || rc=$?
fi

sample_shmem
post_shmem_raw="$SAMPLE_RAW" post_tmpfs="$SAMPLE_TMPFS" post_shmem_eff="$SAMPLE_EFF"
post_thr="$(cat "$FREQ/throttle/status")"

verdict="VALID"
reasons=""
# Plain `if` blocks, not `&&` chains -- a trailing false `&&` under `set -e`
# would exit the script instead of just skipping the reason.
if [ $((post_shmem_eff - pre_shmem_eff)) -gt "$SHMEM_GROWTH_SUSPECT_KB" ]; then
    verdict="SUSPECT"
    reasons="$reasons shmem-grew:$((post_shmem_eff - pre_shmem_eff))kB"
fi
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    verdict="SUSPECT"
    reasons="$reasons timeout-killed:rc=$rc"
fi
# journalctl legitimately finds nothing (grep rc=1) on a clean run -- that's
# inside an `if` condition, which `set -e` already exempts from tripping.
kernel_log() {
    if [ -n "$JOURNALCTL_CMD" ]; then $JOURNALCTL_CMD 2>/dev/null; else journalctl -k --since "10 minutes ago" --no-pager 2>/dev/null; fi
}
if kernel_log | grep -qiE 'GT reset|guc_id|CAT error'; then
    verdict="SUSPECT"
    reasons="$reasons kernel-gpu-fault"
fi

verdict_line="$verdict${reasons:+:$reasons}"
# pci=/card= record which device was guarded. When --sysfs-card was passed
# directly (bypassing derivation) and no --pci accompanied it, $PCI is
# still empty here -- stamp the literal "override" rather than a blank.
pci_for_log="${PCI:-override}"
if [ -n "$LOG" ]; then
    {
        echo "# bench-guard: $verdict_line pci=$pci_for_log card=$SYSFS_CARD pre_throttle=$pre_thr post_throttle=$post_thr pre_shmem_raw=${pre_shmem_raw}kB pre_tmpfs=${pre_tmpfs}kB pre_shmem_eff=${pre_shmem_eff}kB post_shmem_raw=${post_shmem_raw}kB post_tmpfs=${post_tmpfs}kB post_shmem_eff=${post_shmem_eff}kB cmd: $*"
        cat "$tmp_out"
    } > "$LOG"
else
    echo "bench-guard: $verdict_line" >&2
fi
exit "$rc"
