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
# under `timeout -k 15 <budget>`, so the bench sits below a `timeout` process
# that is ITSELF a direct child of the guard. Only ever descends through a
# direct child of the guard whose comm is literally "timeout" -- every OTHER
# direct child of the guard is ignored outright, never accepted and never
# descended into. This is deliberately strict, not merely a name filter on
# the final candidate: bench-guard.sh is a shell script that runs many of its
# own external helpers (mktemp, awk, cat, journalctl, pgrep, and -- caught by
# hardware testing, llama.cpp-gvu7 review -- the `rm -f "$tmp_out"` its own
# EXIT trap runs right as it finishes) as transient direct children of
# itself. Those are indistinguishable from the wrapped command by comm alone
# (none of them is named "timeout" or "env" either), so a filter that simply
# rejected "timeout"/"env" and accepted anything else, as an earlier version
# of this function did, would occasionally sample bench-guard's own trap
# cleanup instead of the bench -- reproduced on hardware as
# `comm=rm cmdline=rm -f /tmp/tmp.XXXXXX`. Requiring the `timeout` hop first
# means only the wrapped command's own subtree is ever eligible.
#
# Below the `timeout` hop, `env VAR=1 CMD` execve()s CMD in place (no extra
# fork) so a compiled binary like llama-bench keeps its own comm at that hop
# and is accepted immediately; a shebang script (as used by the test
# fixture) reports comm="bash" here via binfmt_script's interpreter swap,
# which is also accepted immediately (it is not "env") rather than descending
# further into whatever the script itself later forks (e.g. a `sleep` child)
# -- the bench's own top-level process is what RssAnon should track, not its
# leaf descendant.
#
# An earlier version also fell back to returning the guard's own direct
# child (i.e. "timeout" itself, unresolved further) whenever the first probe
# raced ahead of timeout forking its child, and then never re-tried, so
# every RssAnon sample for the whole run read the ~1.2 MB wrapper instead of
# an 11 GB model process. There is no such fallback here: a `timeout` hop
# with nothing (yet) beneath it simply yields no candidate this attempt, and
# the caller re-invokes this function every sampling tick until one appears.
find_bench_pid() {
    local guard_pid="$1" tpid p c comm depth
    local -a queue next
    for tpid in $(pgrep -P "$guard_pid" 2>/dev/null || true); do
        comm="$(cat "/proc/$tpid/comm" 2>/dev/null || echo "")"
        [ "$comm" = "timeout" ] || continue

        queue=("$tpid")
        depth=0
        while [ "$depth" -lt 5 ] && [ "${#queue[@]}" -gt 0 ]; do
            next=()
            for p in "${queue[@]}"; do
                for c in $(pgrep -P "$p" 2>/dev/null || true); do
                    comm="$(cat "/proc/$c/comm" 2>/dev/null || echo "")"
                    case "$comm" in
                        env) next+=("$c");;
                        *) echo "$c"; return 0;;
                    esac
                done
            done
            queue=("${next[@]}")
            depth=$((depth + 1))
        done
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

bench_pid="" bench_comm="-" bench_cmdline="-"
start_ts="$(date +%s.%N)"
while kill -0 "$guard_pid" 2>/dev/null; do
    # Re-resolve whenever we don't have a live candidate -- not capped at a
    # fixed attempt count, so a slow-to-fork command is still found later in
    # the run, and a candidate that exits mid-run (unexpected, but cheap to
    # handle) triggers a fresh resolution rather than sampling a dead pid.
    # Bounded only by the guard's own lifetime (the enclosing `while`); a run
    # that never resolves still proceeds, recording "-" for RssAnon. A failed
    # RE-resolution attempt (no candidate found THIS tick) deliberately does
    # NOT clear bench_pid/bench_comm/bench_cmdline: the bench typically dies
    # of natural causes moments before the guard itself exits (its own
    # postflight still has to run), and blanking the audit fields at that
    # point would report "bench_pid=unknown" alongside stale comm/cmdline
    # from the process that actually ran -- inconsistent and less useful than
    # keeping the last real resolution on record. RssAnon sampling below
    # already degrades to "-" on its own once /proc/<pid>/status is gone, so
    # nothing here causes a dead pid's memory to be misreported.
    if [ -z "$bench_pid" ] || ! kill -0 "$bench_pid" 2>/dev/null; then
        new_pid="$(find_bench_pid "$guard_pid" || true)"
        if [ -n "$new_pid" ]; then
            bench_pid="$new_pid"
            bench_comm="$(cat "/proc/$bench_pid/comm" 2>/dev/null || echo -)"
            bench_cmdline="$(tr '\0' ' ' < "/proc/$bench_pid/cmdline" 2>/dev/null || true)"
            bench_cmdline="${bench_cmdline:0:120}"
            [ -n "$bench_cmdline" ] || bench_cmdline="-"
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
