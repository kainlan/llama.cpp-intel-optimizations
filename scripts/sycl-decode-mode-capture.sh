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
#                           taken once before and once after the run.
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

# find_bench_pid: bench-guard.sh runs the wrapped command as its own child
# under `timeout -k 15 <budget>`, so the bench sits TWO levels below the
# guard's pid -- walk pgrep -P twice (guard -> timeout -> command) and take
# the first (only) descendant found at each hop. `timeout` forks exactly one
# child to run the wrapped command, so no name filter is needed to pick the
# right one -- and matching purely by process-tree ancestry (never a global
# name/pattern search) means this can never match the capture script's own
# command line either. Deliberately NOT matched by comm/basename: `env
# VAR=1 CMD` execve()s CMD in place (no extra fork) so a compiled binary like
# llama-bench keeps its own comm at this hop, but a shebang script (as used
# by the test fixture) reports comm="bash" here via binfmt_script's interpreter
# swap -- a name filter tuned for one would miss the other. Falls back to a
# direct child of the guard in case a future bench-guard.sh cuts out the
# intermediate `timeout` layer.
find_bench_pid() {
    local guard_pid="$1" tpid cpid
    for tpid in $(pgrep -P "$guard_pid" 2>/dev/null || true); do
        for cpid in $(pgrep -P "$tpid" 2>/dev/null || true); do
            echo "$cpid"; return 0
        done
    done
    for cpid in $(pgrep -P "$guard_pid" 2>/dev/null || true); do
        echo "$cpid"; return 0
    done
    return 1
}

parse_tg128() {
    local log="$1" line value
    [ -r "$log" ] || { echo ""; return 0; }
    line="$(grep -F 'tg128' "$log" 2>/dev/null | tail -1 || true)"
    [ -n "$line" ] || { echo ""; return 0; }
    # Markdown table row: take the last non-empty '|'-delimited cell (the t/s
    # column), which looks like "39.52 ± 0.31".
    value="$(printf '%s\n' "$line" | awk -F'|' '{ for (i=NF; i>=1; i--) { s=$i; gsub(/^[ \t]+|[ \t]+$/, "", s); if (s != "") { print s; exit } } }')"
    [ -n "$value" ] || { echo ""; return 0; }
    printf '%s\n' "$value" | awk '{print $1}'
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

bench_pid=""
pid_attempts=0
start_ts="$(date +%s.%N)"
while kill -0 "$guard_pid" 2>/dev/null; do
    # Give up looking for the pid after ~2s (4 attempts at the 0.5s sampling
    # cadence below); a run that never resolves it still proceeds, recording
    # "-" for RssAnon rather than failing the capture.
    if [ -z "$bench_pid" ] && [ "$pid_attempts" -lt 4 ]; then
        bench_pid="$(find_bench_pid "$guard_pid" || true)"
        pid_attempts=$((pid_attempts + 1))
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
