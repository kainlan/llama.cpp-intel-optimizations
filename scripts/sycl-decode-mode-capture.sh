#!/usr/bin/env bash
# sycl-decode-mode-capture: one-run capture for the B70 GPT-OSS ~28 vs ~39-40
# tok/s decode slow-mode (llama.cpp-gvu7, plan task S4). Wraps EXACTLY ONE
# llama-bench invocation -- never loops the bench, the ticket's rule is three
# separate runs by the caller, each its own capture directory -- and records,
# during that one run:
#   <dir>/bench.log     -- bench-guard.sh's own VALID/SUSPECT-stamped capture
#                           of the wrapped command's stdout+stderr.
#   <dir>/timeline.tsv   -- act_freq/cur_freq/throttle/reason_pl2 + RssAnon of
#                           the bench pid, sampled every 0.5s while it runs.
#   <dir>/host.txt       -- loadavg, ffmpeg tenant count, Shmem, MemAvailable,
#                           taken once before and once after the run, plus a
#                           trailing "bench_pid=... comm=... cmdline=..." line
#                           recording exactly what timeline.tsv's RssAnon
#                           column tracked (or "unknown" if never resolved).
#   <dir>/kprof*         -- GGML_SYCL_KERNEL_PROFILE CSV for the run.
#   <dir>/mode.txt       -- "tg128=<value> mode=slow|fast|unknown" (thresholds
#                           32/36 tok/s), the same line printed to stdout.
#
# All host-corruption preflight (throttled/active card, stale GPU tenant,
# elevated Shmem) is bench-guard.sh's, invoked here as a CHILD -- this script
# never re-implements that logic, only forwards its test hooks unchanged
# (--sysfs-card, --meminfo, --pgrep-cmd, --df-cmd, --journalctl-cmd,
# --max-wait) so the VALID/SUSPECT verdict and the REFUSED (exit 3) path come
# for free. On refusal, this script mirrors bench-guard's exit 3 and writes
# NO mode.txt -- there was no run to compute a mode from.
#
# Card derivation for the sampler is a live PCI-symlink lookup, the same one
# bench-guard.sh performs internally for its own preflight (0000:03:00.0 =
# B70, level_zero:0; 0000:07:00.0 = B50, level_zero:1) -- copied here rather
# than shared because bench-guard.sh does not expose its derived card to a
# caller. Passing --sysfs-card explicitly (as the test suite does) makes both
# this script and the bench-guard.sh child agree on the same fake tree.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_GUARD="$SCRIPT_DIR/bench-guard.sh"

OUT="" SYSFS_CARD="" MEMINFO="/proc/meminfo" PGREP_CMD="" DF_CMD="" JOURNALCTL_CMD="" MAX_WAIT=""

while [ $# -gt 0 ]; do case "$1" in
    --out)             OUT="$2";            shift 2;;
    --sysfs-card)      SYSFS_CARD="$2";     shift 2;;
    --meminfo)         MEMINFO="$2";        shift 2;;
    --pgrep-cmd)       PGREP_CMD="$2";      shift 2;;
    --df-cmd)          DF_CMD="$2";         shift 2;;
    --journalctl-cmd)  JOURNALCTL_CMD="$2"; shift 2;;
    --max-wait)        MAX_WAIT="$2";       shift 2;;
    --) shift; break;;
    *) echo "sycl-decode-mode-capture: unknown arg $1" >&2; exit 2;;
esac; done
[ -n "$OUT" ] || { echo "sycl-decode-mode-capture: --out DIR is required" >&2; exit 2; }
[ $# -gt 0 ] || { echo "sycl-decode-mode-capture: no bench command (pass it after --)" >&2; exit 2; }
[ -x "$BENCH_GUARD" ] || { echo "sycl-decode-mode-capture: $BENCH_GUARD not found or not executable" >&2; exit 2; }

mkdir -p "$OUT"
# host_snapshot (below) APPENDS; truncate once here so re-running a capture
# into an existing --out dir doesn't accumulate stale before/after blocks
# and audit lines from a prior run underneath the new ones.
: > "$OUT/host.txt"

# --- card derivation (mirrors bench-guard.sh; skipped when --sysfs-card is given) ---
if [ -z "$SYSFS_CARD" ]; then
    SELECTOR="${ONEAPI_DEVICE_SELECTOR:-}"
    case "$SELECTOR" in
        level_zero:0) PCI="0000:03:00.0";;
        level_zero:1) PCI="0000:07:00.0";;
        *) echo "sycl-decode-mode-capture: cannot derive card: ONEAPI_DEVICE_SELECTOR must be exactly level_zero:0 or level_zero:1 (got '$SELECTOR'); otherwise pass --sysfs-card" >&2; exit 3;;
    esac
    for c in /sys/class/drm/card*; do
        [ "$(readlink -f "$c/device" 2>/dev/null | xargs -r basename)" = "$PCI" ] && SYSFS_CARD="$c" && break
    done
    [ -n "$SYSFS_CARD" ] || { echo "sycl-decode-mode-capture: no DRM card for PCI $PCI" >&2; exit 3; }
fi
FREQ="$SYSFS_CARD/device/tile0/gt0/freq0"

# Forward the test hooks bench-guard.sh understands, unchanged, plus the
# --sysfs-card this script just derived (or was given) so both agree.
GUARD_ARGS=(--sysfs-card "$SYSFS_CARD" --meminfo "$MEMINFO")
[ -n "$PGREP_CMD" ] && GUARD_ARGS+=(--pgrep-cmd "$PGREP_CMD")
[ -n "$DF_CMD" ] && GUARD_ARGS+=(--df-cmd "$DF_CMD")
[ -n "$JOURNALCTL_CMD" ] && GUARD_ARGS+=(--journalctl-cmd "$JOURNALCTL_CMD")
[ -n "$MAX_WAIT" ] && GUARD_ARGS+=(--max-wait "$MAX_WAIT")

# read_field: sysfs sampling helper. A MISSING or unreadable sysfs file must
# produce a placeholder, never abort the sampler loop under `set -e`.
read_field() {
    local f="$1"
    if [ -r "$f" ]; then cat "$f" 2>/dev/null || echo "-"; else echo "-"; fi
}

ffmpeg_count() {
    local n
    if [ -n "$PGREP_CMD" ]; then
        n="$({ $PGREP_CMD 2>/dev/null || true; } | grep -c ffmpeg || true)"
    else
        n="$(pgrep -c ffmpeg 2>/dev/null || true)"
    fi
    [ -n "$n" ] && echo "$n" || echo 0
}

host_snapshot() {
    local label="$1"
    {
        echo "=== $label ==="
        echo "loadavg: $(cat /proc/loadavg 2>/dev/null || echo -)"
        echo "ffmpeg_count: $(ffmpeg_count)"
        echo "Shmem: $(awk '/^Shmem:/{print $2}' "$MEMINFO" 2>/dev/null || echo -) kB"
        echo "MemAvailable: $(awk '/^MemAvailable:/{print $2}' "$MEMINFO" 2>/dev/null || echo -) kB"
    } >> "$OUT/host.txt"
}

# find_bench_pid: STRUCTURAL leaf descent -- no comm/name filtering at all.
# Walks `pgrep -P` from guard_pid down to whatever is CURRENTLY the innermost
# live descendant and returns that. The wrapped command sits under
# `timeout -k 15 <budget>`, and `env VAR=1 CMD` execve()s CMD in place (no
# extra fork), so once that chain has formed its leaf IS the bench -- no
# special-casing of either wrapper is needed. Returns nothing if guard_pid
# currently has no descendants.
#
# Two name-based filters were tried here before this and both failed
# (llama.cpp-gvu7 review, hardware + hermetic testing):
#   - accepting the guard's own direct child unconditionally as a fallback
#     froze on `timeout` itself when the wrapped command hadn't forked yet,
#     reproduced on B70 hardware as a constant ~1.2 MB RssAnon across a whole
#     77-row run;
#   - requiring a literal "timeout" comm before descending (this function's
#     immediately preceding version) fixed that, but the reviewer then
#     showed bench-guard's OWN throttle/tenant poll loop forks a `sleep`
#     directly under the guard while it waits for a busy card -- a denylist
#     tuned for {timeout, env} does not reject "sleep", nor would it reject
#     "df"/"awk"/"cat" from bench-guard's other internal pipelines.
# Comm alone cannot distinguish the wrapped command from bench-guard's own
# subprocesses, because both are transient children somewhere under
# guard_pid at one point or another. Leaf descent sidesteps the question:
# it tracks WHERE the live tree currently bottoms out, and the caller
# re-derives that every sampling tick (see the re-resolve condition below)
# rather than trusting one resolution for the rest of the run -- so a
# transient false latch (e.g. onto bench-guard's own poll `sleep`) corrects
# itself within one tick once that process exits or grows a child of its
# own, instead of freezing on it.
find_bench_pid() {
    local pid="$1" child
    while :; do
        child="$(pgrep -P "$pid" 2>/dev/null | head -1 || true)"
        [ -n "$child" ] || break
        pid="$child"
    done
    if [ "$pid" != "$1" ]; then
        echo "$pid"
        return 0
    fi
    return 1
}

parse_tg128() {
    local log="$1" line value
    [ -r "$log" ] || { echo ""; return 0; }
    # Anchor to an actual markdown TABLE ROW (a line beginning with '|' that
    # has a "tg128" cell) -- a bare substring grep also matches bench-guard's
    # own header line, which echoes the full wrapped command including
    # GGML_SYCL_KERNEL_PROFILE_OUTPUT="$OUT/kprof": an --out directory named
    # e.g. b70-tg128-run1 puts the literal text "tg128" in that non-data
    # line too (llama.cpp-gvu7 review, reproduced hermetically).
    line="$(grep -E '^\|.*tg128' "$log" 2>/dev/null | tail -1 || true)"
    [ -n "$line" ] || { echo ""; return 0; }
    # Take the last non-empty '|'-delimited cell (the t/s column), which
    # looks like "39.52 ± 0.31".
    value="$(printf '%s\n' "$line" | awk -F'|' '{ for (i=NF; i>=1; i--) { s=$i; gsub(/^[ \t]+|[ \t]+$/, "", s); if (s != "") { print s; exit } } }')"
    [ -n "$value" ] || { echo ""; return 0; }
    value="$(printf '%s\n' "$value" | awk '{print $1}')"
    # Reject anything that is not a plain decimal number. compute_mode feeds
    # this to awk's `v + 0`, which silently coerces non-numeric garbage (a
    # crashed bench can leave a cell like "#") to 0 and reports mode=slow --
    # the exact verdict this capture exists to detect -- so a value that
    # fails this check must become "" (mode=unknown), never pass through.
    printf '%s\n' "$value" | grep -qE '^[0-9]+([.][0-9]+)?$' || { echo ""; return 0; }
    echo "$value"
}

compute_mode() {
    local v="$1"
    [ -n "$v" ] || { echo unknown; return 0; }
    awk -v v="$v" 'BEGIN { if (v + 0 < 32) print "slow"; else if (v + 0 > 36) print "fast"; else print "unknown"; }'
}

# --- run ---

host_snapshot before

"$BENCH_GUARD" --log "$OUT/bench.log" "${GUARD_ARGS[@]}" -- \
    env GGML_SYCL_KERNEL_PROFILE=1 GGML_SYCL_KERNEL_PROFILE_FORMAT=csv GGML_SYCL_KERNEL_PROFILE_OUTPUT="$OUT/kprof" "$@" &
guard_pid=$!

: > "$OUT/timeline.tsv"
printf 't_s\tact_freq\tcur_freq\tthrottle_status\treason_pl2\trss_anon_kb\n' >> "$OUT/timeline.tsv"

bench_pid="" bench_comm="-" bench_cmdline="-" bench_found_deep=0
start_ts="$(date +%s.%N)"
while kill -0 "$guard_pid" 2>/dev/null; do
    # Re-resolve on EVERY tick while the current candidate is not a settled
    # leaf: empty (never found, or the guard currently has no descendants),
    # STILL HAS A CHILD of its own (the live tree moved deeper since the
    # last tick -- e.g. bench-guard's poll `sleep` exited and `timeout`'s
    # chain has now formed beneath the guard, or the bench itself spawned a
    # subprocess), or dead. This is what makes a transient false latch (a
    # brief bench-guard internal helper, or its own throttle/tenant poll
    # `sleep`) self-correct within one sampling tick instead of freezing for
    # the run -- there is no attempt cap and no cached "resolved, stop
    # looking" state. A failed re-resolution attempt (no descendant found
    # THIS tick) deliberately does NOT clear bench_pid/bench_comm/
    # bench_cmdline: the bench typically dies of natural causes moments
    # before the guard itself exits (its own postflight still has to run),
    # and blanking the audit fields at that point would report
    # "bench_pid=unknown" alongside stale comm/cmdline from the process that
    # actually ran -- inconsistent and less useful than keeping the last
    # real resolution on record. RssAnon sampling below already degrades to
    # "-" on its own once /proc/<pid>/status is gone, so nothing here causes
    # a dead pid's memory to be misreported.
    if [ -z "$bench_pid" ] \
        || [ -n "$(pgrep -P "$bench_pid" 2>/dev/null | head -1 || true)" ] \
        || [ ! -r "/proc/$bench_pid/status" ]; then
        new_pid="$(find_bench_pid "$guard_pid" || true)"
        if [ -n "$new_pid" ]; then
            # A SUCCESSFUL resolution can still be the WRONG process: pure
            # leaf descent finds whatever currently has no children, and
            # `timeout` briefly satisfies that too, in the narrow window
            # after the wrapped command exits (and its slot is reaped) but
            # before `timeout` itself notices and exits -- observed directly
            # while testing this fix. `timeout` is always a DIRECT child of
            # the guard (depth 1); the wrapped command is always at least
            # one hop deeper (depth 2+, since `env VAR=1 CMD` execve()s CMD
            # in place with no extra hop of its own). So once a resolution
            # has EVER gone deeper than depth 1, a later resolution landing
            # back on a direct child of the guard means the real subtree has
            # emptied out (the bench finished), not that tracking should
            # regress to the wrapper -- ignore it and keep the last real
            # (deeper) resolution on record. Before anything has gone deep
            # yet, a depth-1 candidate is accepted normally (covers a
            # hypothetical future bench-guard.sh that runs the command
            # without an intermediate `timeout` layer at all).
            # Field 4 of /proc/PID/stat is PPID; this assumes comm (field 2,
            # parenthesised) has no embedded whitespace, true for every comm
            # this script ever sees (bash, timeout, env, sleep, llama-bench).
            new_ppid="$(awk '{print $4}' "/proc/$new_pid/stat" 2>/dev/null || true)"
            if [ "$new_ppid" = "$guard_pid" ] && [ "$bench_found_deep" -eq 1 ]; then
                :
            else
                bench_pid="$new_pid"
                bench_comm="$(cat "/proc/$bench_pid/comm" 2>/dev/null || echo -)"
                # -r guards against a shell-level redirection error hitting
                # stderr when a fast-dying candidate (e.g. a bench that
                # crashes almost immediately) exits in the gap between
                # resolution and this read -- `< missing-file` fails before
                # `tr`'s own `2>/dev/null` can apply, since input
                # redirection is set up before the command runs.
                if [ -r "/proc/$bench_pid/cmdline" ]; then
                    bench_cmdline="$(tr '\0' ' ' < "/proc/$bench_pid/cmdline" 2>/dev/null || true)"
                else
                    bench_cmdline=""
                fi
                bench_cmdline="${bench_cmdline:0:120}"
                [ -n "$bench_cmdline" ] || bench_cmdline="-"
                [ "$new_ppid" = "$guard_pid" ] || bench_found_deep=1
            fi
        fi
    fi

    now_ts="$(date +%s.%N)"
    t_s="$(awk -v a="$now_ts" -v b="$start_ts" 'BEGIN { printf "%.1f", a - b }')"
    act_freq="$(read_field "$FREQ/act_freq")"
    cur_freq="$(read_field "$FREQ/cur_freq")"
    throttle_status="$(read_field "$FREQ/throttle/status")"
    reason_pl2="$(read_field "$FREQ/throttle/reason_pl2")"
    rss_anon_kb="-"
    if [ -n "$bench_pid" ] && [ -r "/proc/$bench_pid/status" ]; then
        rss_anon_kb="$(awk '/^RssAnon:/{print $2}' "/proc/$bench_pid/status" 2>/dev/null || true)"
        [ -n "$rss_anon_kb" ] || rss_anon_kb="-"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$t_s" "$act_freq" "$cur_freq" "$throttle_status" "$reason_pl2" "$rss_anon_kb" >> "$OUT/timeline.tsv"

    sleep 0.5
done

rc=0
wait "$guard_pid" || rc=$?

host_snapshot after

# Auditability (llama.cpp-gvu7 review): record exactly what the timeline's
# RssAnon column tracked, so a reader can tell a resolved bench pid from a
# never-found one without re-deriving the process tree.
echo "bench_pid=${bench_pid:-unknown} comm=$bench_comm cmdline=$bench_cmdline" >> "$OUT/host.txt"

if [ "$rc" -eq 3 ]; then
    echo "sycl-decode-mode-capture: bench-guard refused (see $OUT/bench.log); no mode computed" >&2
    exit "$rc"
fi

tg128_raw="$(parse_tg128 "$OUT/bench.log")"
mode="$(compute_mode "$tg128_raw")"
tg128_display="${tg128_raw:-unknown}"
line="tg128=$tg128_display mode=$mode"
echo "$line"
echo "$line" > "$OUT/mode.txt"

exit "$rc"
