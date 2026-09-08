#!/usr/bin/env bash
# bench-guard: preflight refuses to measure on a corrupted host; run+postflight
# stamps the archived log VALID/SUSPECT (perf-recovery epic, track A, tasks
# A1+A2). Refuses to run the wrapped command when the target GPU is
# throttled/active, a stale llama-cli|llama-bench|llama-completion tenant is
# already running, or Shmem is elevated (CLAUDE.md: TTM shmem OOM history).
# All host probes go through overridable roots (--sysfs-card, --meminfo,
# --pgrep-cmd, --journalctl-cmd, --df-cmd, --drm-root -- the last four are
# test hooks, same as --meminfo/--sysfs-card, so the logic is testable
# without hardware).
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
# VALID/SUSPECT verdict header (pci/card identity, pre/post throttle + Shmem
# readings); without --log, the verdict plus the same pci=/card= identification
# is printed to stderr. Either way bench-guard's own exit code always mirrors
# the wrapped command's, never the verdict.
#
# Postflight SUSPECT triggers: Shmem grew more than 5 GB across the run, the
# run was killed by the timeout (rc 124/137), the wrapped command died by any
# other signal (rc >= 128, e.g. 139 SIGSEGV -- a crashed bench is never a
# valid measurement), or the kernel log shows a GT reset/guc_id/CAT error
# since the run started. The kernel-log check counts matching lines
# (`grep -c`, never `grep -q`) rather than testing for a match, because `-q`
# exits at the first hit and SIGPIPEs a still-writing `journalctl` producer;
# under `set -o pipefail` that reads as pipeline failure and takes the VALID
# branch on exactly the runs that have a fault to report (llama.cpp-m1ny).
# The match count is stamped in the reason (`kernel-gpu-fault:<N>`) and in the
# header. A post-run throttle=1 reading is NOT by itself suspect -- the run's
# own power draw asserts it.
#
# Card derivation is LIVE from DRM/PCI enumeration under --drm-root (default
# /sys/class/drm), never a fixed PCI address table (CLAUDE.md: DRM numbering
# AND PCI bus addresses move across boots -- verified 2026-09-05, when the
# discrete cards moved from 0000:03:00.0/07:00.0 to 0000:04:00.0/09:00.0).
# derive_card_for_selector() (below; see its own docstring for the
# enumerate/filter/sort mechanics) maps level_zero:N to the N-th discrete
# GPU by ascending PCI address, zero-based. This assumes Level Zero orders
# the discrete cards by ascending PCI address -- true on every boot
# observed so far (the B70 is always the lower address, i.e.
# level_zero:0). --pci ADDR and --sysfs-card DIR remain explicit overrides
# that bypass the derivation entirely. The selector match is EXACT
# (level_zero:<digit>, a single digit) so a multi-device or other-form
# selector (level_zero:0,1, level_zero:gpu, ...) falls through to the
# explicit "cannot derive card" refusal instead of silently picking one of
# the cards it names.
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

# Without this, a missing/unreadable --meminfo reaches shmem_kb()'s bare
# `awk` unguarded and the script dies on awk's own raw exit status (2) and
# stderr message ("awk: can't open file ...") instead of a clean refuse()
# -- the exact "bare command fails under set -e with no context" failure
# mode this file guards against everywhere else (llama.cpp-imns review
# round 5, finding F9).
[ -r "$MEMINFO" ] || refuse "no meminfo at $MEMINFO"

# is_top_level_card DIR -- true iff DIR exists and its basename matches
# card[0-9]+, i.e. a numbered top-level card rather than a connector entry
# (card1-DP-1, ...). Shared by derive_card_for_selector and
# find_card_by_pci so the two enumeration loops cannot drift onto
# different filters.
is_top_level_card() {
    local c="$1" base
    [ -e "$c" ] || return 1
    base="$(basename "$c")"
    [[ "$base" =~ ^card[0-9]+$ ]]
}

# derive_card_for_selector IDX -- enumerate top-level DRM cards under
# $DRM_ROOT (is_top_level_card), keep discrete Intel display controllers
# (device/vendor 0x8086, device/class 0x0300*), excluding PCI bus 00 on
# any domain, which is the integrated GPU, sort the survivors by PCI
# address (lexical order on the zero-padded dddd:bb:dd.f string is
# domain/bus/device/function order -- unlike the bus-00 exclusion, which
# matches a substring and needs no assumption about domain width, this
# lexical sort DOES assume every survivor's domain is padded to the same
# width, the way sysfs actually presents it; two domains of different
# widths would put the domain digits at different offsets, so the first
# differing character decides instead of the domain's value, and could
# misorder),
# and set DERIVED_CARD/DERIVED_PCI to
# the sysfs card path and PCI address of the IDX-th (zero-based) survivor.
# Calls refuse() (exit 3) directly on any failure. Called directly, never
# via a command substitution, so `set -e` applies to it exactly as it does
# to the rest of the script; every probe below that can genuinely fail (as
# opposed to a file legitimately not existing, which is a normal
# `continue`) is still checked with an explicit `if ! ...; then refuse
# ...; fi` purely to give a specific, actionable message -- an unguarded
# failure under `set -e` would abort the whole script with no context at
# all.
#
# A card that legitimately lacks a `device` link at all is excluded quietly
# -- only the ordering of the SURVIVING cards is index-significant, and the
# absence says nothing meaningful about whether this was ever a GPU. A card
# whose device/vendor or device/class file legitimately does not exist, or
# exists, IS READABLE, and simply holds an unexpected value (extra
# whitespace, an unrelated class like 0x038000), is ALSO excluded quietly
# the same way: everything after it shifts down by one index, exactly as
# if it were a different vendor, and there is no way from here to tell
# "this really is a different device" from "the file is subtly wrong".
#
# In contrast, a DANGLING device symlink (the card had one, but its target
# no longer resolves -- e.g. a device removed or a hot-unplug race), an
# UNREADABLE-OR-UNSEARCHABLE device DIRECTORY (the resolved target exists
# but cannot be read or entered, e.g. `chmod 000` on it -- this must be
# caught separately from the vendor/class file checks below, because an
# unsearchable directory makes `[ -e "$c/device/vendor" ]` itself return
# false, the same as a legitimately absent file, so those checks alone
# cannot see it), and an EXISTING-but-UNREADABLE vendor/class file (the
# file is there, we simply cannot read it, e.g. a permission change) are
# NOT quiet exclusions: all three refuse() loudly instead, because silently
# dropping any of them would change level_zero:N's meaning for every card
# after it with no visible signal -- exactly the silent index shift this
# whole derivation exists to prevent (llama.cpp-imns review round 2/3).
derive_card_for_selector() {
    local idx="$1" c base pci vendor class
    local -a entries=()
    for c in "$DRM_ROOT"/card*; do
        is_top_level_card "$c" || continue
        base="$(basename "$c")"
        if [ ! -e "$c/device" ] && [ ! -L "$c/device" ]; then
            continue   # no device link at all: legitimate skip
        fi
        if [ -L "$c/device" ] && [ ! -e "$c/device" ]; then
            refuse "$base's device symlink is dangling (its target no longer resolves); refusing rather than silently excluding it, which would shift level_zero indices for the remaining cards"
        fi
        if ! pci="$(readlink -f "$c/device" 2>/dev/null | xargs -r basename)"; then
            refuse "failed to resolve the device symlink under $c/device (readlink probe error)"
        fi
        [ -n "$pci" ] || continue
        if [ -d "$c/device" ] && { [ ! -r "$c/device" ] || [ ! -x "$c/device" ]; }; then
            refuse "$base's device directory ($pci) exists but is not readable/searchable; refusing rather than silently excluding it, which would shift level_zero indices for the remaining cards"
        fi
        if [ -e "$c/device/vendor" ] && [ ! -r "$c/device/vendor" ]; then
            refuse "$base's device/vendor exists but is not readable; refusing rather than silently excluding it, which would shift level_zero indices for the remaining cards"
        fi
        [ -r "$c/device/vendor" ] || continue
        if [ -e "$c/device/class" ] && [ ! -r "$c/device/class" ]; then
            refuse "$base's device/class exists but is not readable; refusing rather than silently excluding it, which would shift level_zero indices for the remaining cards"
        fi
        [ -r "$c/device/class" ] || continue
        if ! vendor="$(cat "$c/device/vendor" 2>/dev/null)"; then
            refuse "failed to read $c/device/vendor (probe error after it was confirmed readable)"
        fi
        [ "$vendor" = "0x8086" ] || continue
        if ! class="$(cat "$c/device/class" 2>/dev/null)"; then
            refuse "failed to read $c/device/class (probe error after it was confirmed readable)"
        fi
        case "$class" in 0x0300*) : ;; *) continue;; esac
        case "$pci" in *:00:*) continue;; esac    # exclude PCI bus 00 on any domain (any width) -- the integrated GPU
        entries+=("$base"$'\t'"$pci"$'\t'"$c")
    done
    local sorted_str=""
    if [ "${#entries[@]}" -gt 0 ]; then
        # Lexical sort on field 2 (the PCI address) is domain/bus/device/
        # function order for the zero-padded dddd:bb:dd.f form -- this
        # assumes uniform domain width across survivors (see the docstring
        # above); force LC_ALL=C so a non-C locale cannot reorder it. The
        # explicit `if !` (not relying on mapfile's own status, which never
        # sees a failure from inside a process substitution) is what makes
        # a sort probe failure refuse() instead of silently yielding zero
        # or partial candidates.
        if ! sorted_str="$(printf '%s\n' "${entries[@]}" | LC_ALL=C sort -t $'\t' -k2,2)"; then
            refuse "failed to sort discrete GPU candidates under $DRM_ROOT (sort probe error)"
        fi
    fi
    local -a sorted=()
    [ -z "$sorted_str" ] || mapfile -t sorted <<< "$sorted_str"
    local n="${#sorted[@]}"
    [ "$n" -gt 0 ] || refuse "no discrete Intel GPU display controller found under $DRM_ROOT"
    if [ "$idx" -ge "$n" ]; then
        local disp="" e b p
        for e in "${sorted[@]}"; do
            IFS=$'\t' read -r b p _ <<< "$e"
            disp="$disp${disp:+ }$b=$p"
        done
        refuse "level_zero:$idx is out of range; found $n discrete Intel GPU(s): $disp"
    fi
    local p path
    IFS=$'\t' read -r _ p path <<< "${sorted[$idx]}"
    DERIVED_CARD="$path"
    DERIVED_PCI="$p"
}

# find_card_by_pci PCI -- scan top-level DRM cards under $DRM_ROOT (same
# is_top_level_card filter as derive_card_for_selector, so a connector
# entry can't shadow the real card here either) and print the first
# matching sysfs card path, or an empty line if none matches. Always exits
# 0 (the caller decides what an empty result means), so it is safe to
# assign its output via plain command substitution without an explicit
# `|| ...` guard. Unlike derive_card_for_selector's guarded readlink, a
# readlink failure here is a silent skip rather than a refuse(): this
# function has no index semantics to protect -- there is one target PCI
# address, not an ordered list whose positions shift when an entry is
# dropped -- and a card this loop misses simply leaves SYSFS_CARD empty at
# the call site, which reaches the loud "no DRM card for PCI $PCI" refusal
# there.
find_card_by_pci() {
    local target_pci="$1" c found=""
    for c in "$DRM_ROOT"/card*; do
        is_top_level_card "$c" || continue
        if [ "$(readlink -f "$c/device" 2>/dev/null | xargs -r basename)" = "$target_pci" ]; then
            found="$c"
            break
        fi
    done
    printf '%s\n' "$found"
}

if [ -z "$SYSFS_CARD" ]; then
    if [ -z "$PCI" ]; then
        case "$SELECTOR" in
            level_zero:[0-9]) : ;;
            *) refuse "cannot derive card: ONEAPI_DEVICE_SELECTOR must be exactly level_zero:<digit> (got '$SELECTOR'); otherwise pass --pci or --sysfs-card";;
        esac
        # Derive BOTH the PCI address and the sysfs card path in the same
        # pass (into DERIVED_PCI/DERIVED_CARD) -- do not re-scan $DRM_ROOT
        # afterward to find the card for that PCI: an earlier, unfiltered
        # second scan matched a connector entry sharing the same device
        # symlink ahead of the real card in glob order (llama.cpp-imns
        # review round 1).
        DERIVED_CARD="" DERIVED_PCI=""
        derive_card_for_selector "${SELECTOR#level_zero:}"
        SYSFS_CARD="$DERIVED_CARD"
        PCI="$DERIVED_PCI"
    else
        # --pci was given explicitly (no --sysfs-card): still need to find
        # the matching sysfs card.
        SYSFS_CARD="$(find_card_by_pci "$PCI")"
    fi
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
elif [ "$rc" -ge 128 ]; then
    verdict="SUSPECT"
    reasons="$reasons signal:rc=$rc"
fi
# journalctl legitimately finds nothing (grep rc=1) on a clean run -- that's
# inside an `if` condition, which `set -e` already exempts from tripping.
# Count form, never `-q`: `-q` exits at the first match, which SIGPIPEs a
# still-writing journalctl and (under pipefail) reports pipeline failure
# instead of a match -- the exact case that must be caught (llama.cpp-m1ny).
# `grep -c` reads its input to completion, and `|| true` covers the
# legitimate zero-match case so `set -e` doesn't trip on grep's rc=1.
kernel_log() {
    if [ -n "$JOURNALCTL_CMD" ]; then $JOURNALCTL_CMD 2>/dev/null; else journalctl -k --since "10 minutes ago" --no-pager 2>/dev/null; fi
}
kf="$(kernel_log | grep -ciE 'GT reset|guc_id|CAT error' || true)"
if [ "${kf:-0}" -gt 0 ]; then
    verdict="SUSPECT"
    reasons="$reasons kernel-gpu-fault:${kf}"
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
    echo "bench-guard: $verdict_line pci=$pci_for_log card=$SYSFS_CARD" >&2
fi
exit "$rc"
