#!/usr/bin/env bash
# sycl-decode-mode-capture: one-run capture for the B70 GPT-OSS ~28 vs ~39-40
# tok/s decode slow-mode (llama.cpp-gvu7, plan task S4). Wraps EXACTLY ONE
# llama-bench invocation -- never loops the bench, the ticket's rule is three
# separate runs by the caller, each its own capture directory -- and records,
# during that one run:
#   <dir>/bench.log      -- bench-guard.sh's own VALID/SUSPECT-stamped
#                           capture of the wrapped command's stdout+stderr.
#   <dir>/timeline.tsv    -- act_freq/cur_freq/throttle/reason_pl2/bench_pid/
#                           RssAnon of the bench pid, sampled every
#                           TICK_SECONDS while it runs. bench_pid and
#                           RssAnon both read "-" until a candidate has been
#                           CONFIRMED -- the same pid resolved on
#                           CONFIRM_TICKS consecutive ticks -- so a
#                           transient bench-guard helper (its throttle/
#                           tenant poll sleep, df, journalctl, ...) is never
#                           attributed a row, even briefly. Consequence: a
#                           wrapped command shorter than roughly
#                           CONFIRM_TICKS*TICK_SECONDS (at least ~1.5s --
#                           more under load, since confirmation counts
#                           ticks, not elapsed time) is never confirmed at
#                           all, so its ENTIRE run gets an all-"-"
#                           bench_pid/rss_anon_kb column (host.txt's audit
#                           line still names it if resolved, since that
#                           commit happens independently of row
#                           attribution) -- acceptable for this script's
#                           stated purpose, llama-bench decode runs of tens
#                           of seconds.
#   <dir>/host.txt        -- loadavg, ffmpeg tenant count, Shmem,
#                           MemAvailable, taken once before and once after
#                           the run, plus a trailing
#                           "bench_pid=... comm=... cmdline=..." line
#                           recording exactly what timeline.tsv's RssAnon
#                           column tracked (or "unknown" if never resolved).
#   <dir>/kprof*          -- GGML_SYCL_KERNEL_PROFILE CSV for the run.
#   <dir>/mode.txt        -- "tg128=<value> mode=slow|fast|unknown"
#                           (thresholds TG128_SLOW_MAX/TG128_FAST_MIN), the
#                           same line printed to stdout.
#
# host.txt, bench.log, mode.txt, and timeline.tsv are ALL reset once at the
# very start of every invocation, before bench-guard.sh is even launched.
# host.txt and timeline.tsv need this because this script itself only
# appends to them afterward. bench.log and mode.txt need it for a subtler
# reason: their sole writers -- bench-guard.sh's own --log write, and this
# script's own final mode.txt write -- are BOTH skipped entirely on a
# preflight refusal (bench-guard.sh's refuse() path exits before ever
# touching --log; this script's own `exit 3` branch runs before its mode.txt
# write). Without an explicit reset, a refusal into a --out dir a PRIOR
# successful run already populated would silently leave that prior run's
# verdict in both files -- exactly the kind of stale artifact this script's
# own refusal message would then point the reader at (llama.cpp-gvu7 quality
# review; the round-1 nit that got host.txt's truncate added did not cover
# these two, since they follow a different rule: reset-and-conditionally-
# rewritten, not reset-and-always-rewritten).
#
# All host-corruption preflight (throttled/active card, stale GPU tenant,
# elevated Shmem) is bench-guard.sh's DECISION logic, invoked here as a
# CHILD -- this script never re-implements it, so the VALID/SUSPECT verdict
# and the REFUSED (exit 3) path come for free. All six of bench-guard.sh's
# own test hooks (--sysfs-card, --meminfo, --pgrep-cmd, --df-cmd,
# --journalctl-cmd, --max-wait) are forwarded to it unchanged, but three are
# ALSO read locally by this script for its own purposes, not merely handed
# through: --sysfs-card additionally derives FREQ for the sampler (the same
# card bench-guard.sh itself derives, so passing --sysfs-card keeps both in
# agreement); --meminfo is additionally read by host_snapshot for its
# Shmem/MemAvailable lines; --pgrep-cmd is additionally read by
# ffmpeg_count, which pipes its output through `grep -c ffmpeg` -- a
# DIFFERENT question than bench-guard.sh's own tenant-listing use of the
# same hook (`pgrep -a -x 'llama-cli|llama-bench|llama-completion'`). A
# --pgrep-cmd override built only to answer bench-guard.sh's tenant
# question (e.g. one that lists tenants but never mentions "ffmpeg") will
# silently make every host.txt ffmpeg_count read 0 rather than fail loudly
# -- pass a general process-listing command if ffmpeg counting matters to
# the caller, not one filtered to llama tenants (llama.cpp-gvu7 quality
# review).
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

# --- tunables (named, not inline literals, the way bench-guard.sh names its
# own: SHMEM_CEIL_KB, SHMEM_GROWTH_SUSPECT_KB, POLL_INTERVAL). Several
# comments throughout this script and its test do arithmetic against these
# (e.g. "CONFIRM_TICKS consecutive TICK_SECONDS ticks span >= 1.5s") -- a
# silent literal edit would make those claims wrong with nothing to catch
# it; a name change here is at least grep-able (llama.cpp-gvu7 quality
# review). ---
TG128_SLOW_MAX=32       # compute_mode: strictly below this is "slow"
TG128_FAST_MIN=36       # compute_mode: strictly above this is "fast"
TICK_SECONDS=0.5        # sampler cadence
CONFIRM_TICKS=3         # consecutive ticks the SAME pid must resolve to
                         # before it is trusted (see confirm_bench_pid)
CMDLINE_MAX_CHARS=120   # audit-line cmdline truncation

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
# Reset every file this script itself writes, before bench-guard.sh is even
# launched -- see the header comment above for why mode.txt and bench.log
# need an explicit `rm -f` (their writers don't run on every path) while
# host.txt and timeline.tsv only need a truncate (this script always
# rewrites them). Grouped together so "what does a fresh invocation reset"
# has one answer, in one place (llama.cpp-gvu7 quality review).
: > "$OUT/host.txt"
rm -f "$OUT/mode.txt" "$OUT/bench.log"
: > "$OUT/timeline.tsv"
printf 't_s\tact_freq\tcur_freq\tthrottle_status\treason_pl2\tbench_pid\trss_anon_kb\n' >> "$OUT/timeline.tsv"

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
# (confirm_bench_pid, below) re-derives that on EVERY sampling tick rather
# than trusting one resolution for the rest of the run.
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

# confirm_bench_pid: calls find_bench_pid fresh on EVERY invocation (no
# "skip if already have a candidate" shortcut) and requires the SAME pid on
# CONFIRM_TICKS consecutive calls before treating it as trustworthy. Reads
# and updates the global pending_pid/pending_confirms state, and sets the
# global confirmed_pid to the confirmed pid once the bar is met (every
# subsequent call while that same pid remains current) or to "" otherwise.
# The caller is responsible for deciding whether a freshly-confirmed pid is
# actually NEW (see commit_bench_pid) -- this function only answers "is the
# current leaf trustworthy right now". MUST be called directly, never via
# `$(confirm_bench_pid ...)`: command substitution forks a subshell, and a
# subshell's writes to pending_pid/pending_confirms/confirmed_pid never
# reach the caller -- the whole point of these being globals is that they
# persist ACROSS ticks in the same shell (caught directly during this
# refactor: capturing this function's output via `$(...)` silently kept
# every tick's confirmation state stuck at zero, since each call started
# and ended in its own throwaway subshell -- the sampler never confirmed
# anything, ever, and every row read "-").
#
# CONFIRM_TICKS is 3, not 2: an earlier version of this mechanism (an
# outright depth-based guard, rejecting only a candidate that is a direct
# child of the guard) missed bench-guard's OWN helpers that run DEEPER than
# that -- `SAMPLE_TMPFS="$(tmpfs_kb)"` puts `df` at depth 3 under the guard,
# and the postflight `kernel_log | grep` puts `journalctl` at depth 2, both
# measured directly -- so a depth check alone cannot tell them from the
# bench once the real bench has exited and either helper becomes the sole
# leaf for a moment. The FIRST fix tried here required only two consecutive
# ticks (the reviewer's own proposal), but empirically re-testing that
# against an adversarial ~0.8s helper (standing in for the ~10ms a real one
# measures) showed a genuine ~20% false-confirmation rate (1/5 on a first
# run, reproduced): two TICK_SECONDS-spaced samples CAN both land inside an
# 0.8s-long process's lifetime purely by phase-alignment luck. THREE
# consecutive ticks span a full 2*TICK_SECONDS = 1.0s, so a process cannot
# be observed at three points TICK_SECONDS apart unless it lives at least
# that long -- the REAL bound (round-3 review, measured on this fixed
# script): an 0.8s helper is excluded 8/8, a 1.4s helper 5/5, but a 2.5s
# helper IS confirmed 4/4 (it overwrites the audit line and two tail
# timeline rows per run). No practical exposure follows from that: real
# `journalctl -k ...` measured 0.01-0.02s and real `df -k -t tmpfs` measured
# ~0.00s, two orders of magnitude under the bound. The bench, which runs for
# many seconds, is unaffected either way.
#
# Confirmation is temporal, not structural, so re-resolution is never
# frozen once a candidate has been confirmed -- the guard's own PREFLIGHT
# `df` is depth 3 too, and freezing on "seen something deep once" would let
# a stale confirmed candidate block noticing the real bench replace it. If
# the >1.4s helper class ever needs closing rather than bounding: this
# script sets GGML_SYCL_KERNEL_PROFILE_OUTPUT on the wrapped command only
# (see the env call below), and bench-guard's own helpers inherit its
# environment without that variable, so requiring the string in
# /proc/PID/environ would identify the bench subtree independently of
# depth, comm, and timing -- not implemented, since nothing measured needs
# it yet.
confirm_bench_pid() {
    local guard_pid="$1" new_pid
    new_pid="$(find_bench_pid "$guard_pid" || true)"
    if [ -n "$new_pid" ] && [ "$new_pid" = "$pending_pid" ]; then
        pending_confirms=$((pending_confirms + 1))
    else
        pending_pid="$new_pid"
        if [ -n "$new_pid" ]; then pending_confirms=1; else pending_confirms=0; fi
    fi
    if [ "$pending_confirms" -ge "$CONFIRM_TICKS" ]; then
        confirmed_pid="$pending_pid"
    else
        confirmed_pid=""
    fi
}

# commit_bench_pid: records a freshly-confirmed pid into the global
# bench_pid/bench_comm/bench_cmdline/bench_reportable audit state. Called
# only when confirm_bench_pid reports a pid that differs from the
# currently-committed one.
#
# bench_reportable is a SEPARATE gate from confirmation, computed once here
# and cached (a pid's ppid never changes). CONFIRM_TICKS-tick confirmation
# alone does not keep bench-guard's own throttle/tenant poll `sleep` out of
# the timeline: unlike the ~10ms df/journalctl helpers confirm_bench_pid's
# own comment discusses, that `sleep` runs for the FULL poll interval (5s =
# 10 ticks) and so easily passes confirmation and gets committed to
# bench_pid while the card is still busy. `sleep` is always a direct child
# of the guard (depth 1); the wrapped command is always at least one hop
# deeper (depth 2+, since `env VAR=1 CMD` execve()s CMD in place with no
# extra hop of its own) -- so gating row attribution on depth specifically
# excludes it without touching discovery or confirmation at all. This is
# display-layer only: a depth-1 candidate can still be committed to the
# audit fields (so a run that ends before the real bench ever forms still
# reports something rather than "unknown"), it is just never given a
# timeline row (see the sampler loop below).
#
# This assumes bench-guard.sh keeps wrapping the command in `timeout` (its
# own header already pins `timeout -k 15` as load-bearing); if that ever
# stops holding, the bench itself lands at depth 1 and bench_reportable
# never becomes 1 -- confirmed by deleting `timeout` from a scratch
# bench-guard.sh and re-running (round-3 review): the timeline carries ZERO
# attributed rows for the whole capture while host.txt's audit line still
# names the real bench correctly. That fails CLOSED (data loss, an all-"-"
# timeline with a populated audit line), never misattribution (a wrong
# pid/comm on a row), and the two states remain distinguishable from the
# output alone.
commit_bench_pid() {
    local pid="$1" guard_pid="$2" bench_ppid
    bench_pid="$pid"
    bench_comm="$(cat "/proc/$bench_pid/comm" 2>/dev/null || echo -)"
    # -r guards against a shell-level redirection error hitting stderr when
    # a fast-dying candidate (e.g. a bench that crashes almost immediately)
    # exits in the gap between confirmation and this read -- `< missing-file`
    # fails before `tr`'s own `2>/dev/null` can apply, since input
    # redirection is set up before the command runs.
    if [ -r "/proc/$bench_pid/cmdline" ]; then
        bench_cmdline="$(tr '\0' ' ' < "/proc/$bench_pid/cmdline" 2>/dev/null || true)"
    else
        bench_cmdline=""
    fi
    bench_cmdline="${bench_cmdline:0:$CMDLINE_MAX_CHARS}"
    [ -n "$bench_cmdline" ] || bench_cmdline="-"
    # Field 4 of /proc/PID/stat is PPID; this assumes comm (field 2,
    # parenthesised) has no embedded whitespace, true for every comm this
    # script ever sees.
    bench_ppid="$(awk '{print $4}' "/proc/$bench_pid/stat" 2>/dev/null || true)"
    if [ -n "$bench_ppid" ] && [ "$bench_ppid" != "$guard_pid" ]; then
        bench_reportable=1
    else
        bench_reportable=0
    fi
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
    awk -v v="$v" -v slow="$TG128_SLOW_MAX" -v fast="$TG128_FAST_MIN" \
        'BEGIN { if (v + 0 < slow) print "slow"; else if (v + 0 > fast) print "fast"; else print "unknown"; }'
}

# --- run ---

host_snapshot before

"$BENCH_GUARD" --log "$OUT/bench.log" "${GUARD_ARGS[@]}" -- \
    env GGML_SYCL_KERNEL_PROFILE=1 GGML_SYCL_KERNEL_PROFILE_FORMAT=csv GGML_SYCL_KERNEL_PROFILE_OUTPUT="$OUT/kprof" "$@" &
guard_pid=$!

bench_pid="" bench_comm="-" bench_cmdline="-" bench_reportable=0
pending_pid="" pending_confirms=0 confirmed_pid=""
start_ts="$EPOCHREALTIME"
while kill -0 "$guard_pid" 2>/dev/null; do
    confirm_bench_pid "$guard_pid"
    if [ -n "$confirmed_pid" ] && [ "$confirmed_pid" != "$bench_pid" ]; then
        commit_bench_pid "$confirmed_pid" "$guard_pid"
    fi

    now_ts="$EPOCHREALTIME"
    t_s="$(awk -v a="$now_ts" -v b="$start_ts" 'BEGIN { printf "%.1f", a - b }')"
    act_freq="$(read_field "$FREQ/act_freq")"
    cur_freq="$(read_field "$FREQ/cur_freq")"
    throttle_status="$(read_field "$FREQ/throttle/status")"
    reason_pl2="$(read_field "$FREQ/throttle/reason_pl2")"
    row_bench_pid="-"
    rss_anon_kb="-"
    # Only a CONFIRMED, depth-2+ (bench_reportable) bench_pid is ever
    # attributed a row -- see commit_bench_pid's comment for why depth,
    # not just confirmation, is required.
    if [ -n "$bench_pid" ] && [ "$bench_reportable" -eq 1 ] && [ -r "/proc/$bench_pid/status" ]; then
        row_bench_pid="$bench_pid"
        rss_anon_kb="$(awk '/^RssAnon:/{print $2}' "/proc/$bench_pid/status" 2>/dev/null || true)"
        [ -n "$rss_anon_kb" ] || rss_anon_kb="-"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$t_s" "$act_freq" "$cur_freq" "$throttle_status" "$reason_pl2" "$row_bench_pid" "$rss_anon_kb" >> "$OUT/timeline.tsv"

    sleep "$TICK_SECONDS"
done

rc=0
wait "$guard_pid" || rc=$?

host_snapshot after

# Auditability (llama.cpp-gvu7 review): record exactly what the timeline's
# RssAnon column tracked, so a reader can tell a resolved bench pid from a
# never-found one without re-deriving the process tree.
echo "bench_pid=${bench_pid:-unknown} comm=$bench_comm cmdline=$bench_cmdline" >> "$OUT/host.txt"

if [ "$rc" -eq 3 ]; then
    # bench-guard's own refuse() message already reached stderr directly
    # (it is never redirected here) -- and, unlike a normal run, its
    # refuse() path never touches --log at all, so $OUT/bench.log does not
    # exist to point the reader at (removed at start, above; a stale prior
    # run's file would otherwise still be sitting there).
    echo "sycl-decode-mode-capture: bench-guard refused (see its message above); no mode computed" >&2
    exit "$rc"
fi

tg128_raw="$(parse_tg128 "$OUT/bench.log")"
mode="$(compute_mode "$tg128_raw")"
tg128_display="${tg128_raw:-unknown}"
line="tg128=$tg128_display mode=$mode"
echo "$line"
echo "$line" > "$OUT/mode.txt"

exit "$rc"
