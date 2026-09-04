#!/usr/bin/env bash
# Unit tests for scripts/sycl-decode-mode-capture.sh, against a fake sysfs
# tree and a fake bench command -- mirrors tests/test-bench-guard.sh's
# mk_tree/mk_meminfo/expect_status pattern. All host-corruption preflight
# (throttled/active card, stale tenant, elevated Shmem) belongs to
# bench-guard.sh, which the capture script invokes as a child and which has
# its own test suite (test-bench-guard.sh); this file exercises only the
# capture script itself: the sampler timeline, host.txt before/after blocks,
# tg128 parsing + mode computation, and refusal pass-through.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAPTURE="$ROOT_DIR/scripts/sycl-decode-mode-capture.sh"
GUARD="$ROOT_DIR/scripts/bench-guard.sh"
[ -x "$CAPTURE" ] || { echo "SKIP: sycl-decode-mode-capture.sh not present"; exit 77; }
[ -x "$GUARD" ] || { echo "SKIP: bench-guard.sh not present"; exit 77; }

T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
fail=0

# mk_tree: fake sysfs freq0 dir with ONLY throttle/status and act_freq -- like
# test-bench-guard.sh's own fixture, it deliberately never creates cur_freq or
# throttle/reason_pl2, so every run below also exercises the "missing sysfs
# file produces a placeholder" requirement without a dedicated fixture.
mk_tree() { # $1=throttle $2=act_freq
    local d="$T/sys/class/drm/card9/device/tile0/gt0/freq0"
    mkdir -p "$d/throttle"
    echo "$1" > "$d/throttle/status"
    echo "$2" > "$d/act_freq"
}
mk_meminfo() { printf 'MemAvailable: 190000000 kB\nShmem: %s kB\n' "$1" > "$T/meminfo"; }

expect_status() {
    local want="$1" what="$2"
    shift 2
    [ "$1" = "--" ] || { echo "expect_status: expected -- before command" >&2; exit 2; }
    shift
    local rc=0
    "$@" >/dev/null 2>&1 || rc=$?
    if [ "$rc" -ne "$want" ]; then
        echo "FAIL: expected $what to exit $want, got $rc" >&2
        fail=1
    fi
}

# mk_fake_bench: a fake "llama-bench" that prints a canned markdown tg128 row
# embedding $1 as the t/s figure, then sleeps 2s (an EXTERNAL `sleep`, so this
# script forks a child partway through its life) so the sampler (0.5s
# cadence) gets several samples. The capture script's pid discovery is pure
# structural leaf descent (no name/comm matching at all -- see
# find_bench_pid's comment in the script), so once this script forks `sleep`
# the leaf being tracked moves from this script to that child; that's fine
# for the tests below that only need SOME pid to be resolved (mode/tg128
# parsing, timeline shape, host.txt shape). It is deliberately NOT used for
# the pid/RSS regression check further down, which needs the tracked leaf to
# stay this script's own process throughout -- see mk_fake_bench_grow.
mk_fake_bench() {
    local tg="$1" path="$T/fakebench.sh"
    cat > "$path" <<EOF
#!/usr/bin/env bash
echo '| model | size | params | backend | ngl | test | t/s |'
echo '|---|---|---|---|---|---|---|'
echo "| gpt-oss 20B MXFP4 | 12.83 GiB | 20.91 B | SYCL | 99 | tg128 | $tg ± 0.31 |"
sleep 2
EOF
    chmod +x "$path"
    echo "$path"
}

# mk_fake_bench_grow: like mk_fake_bench, but NEVER forks any subprocess --
# it grows and holds a large string in its OWN bash memory (a doubling loop,
# pure `[`/`:`/arithmetic/string-concatenation builtins) instead of calling
# an external `sleep`, then busy-waits on the builtin $SECONDS variable
# instead. This is required for the pid/RSS regression check below: leaf
# descent tracks the CURRENT deepest live descendant, so a fixture that
# forks anything (mk_fake_bench's own `sleep 2`, for instance, or even a
# transient `$(head -c 64M /dev/zero | tr '\0' a)` pipeline) would have the
# tracked leaf move onto that child, no longer matching this script's own
# memory or its own pid written to FAKE_BENCH_PIDFILE. Doubling a 1-byte
# string 26 times reaches 64 MiB almost instantly and holds bash's own
# RssAnon around that scale throughout the busy-wait -- comfortably above
# both the ~1.2 MB a `timeout`/`env`/`sleep` wrapper reports and the 32768 kB
# (32 MiB) assertion threshold below, with margin (verified empirically
# during this fix: peak ~66 MB for a 64 MiB string, no child ever observed
# via `pgrep -P`).
mk_fake_bench_grow() {
    local tg="$1" path="$T/fakebench-grow.sh"
    cat > "$path" <<EOF
#!/usr/bin/env bash
if [ -n "\${FAKE_BENCH_PIDFILE:-}" ]; then echo "\$\$" > "\${FAKE_BENCH_PIDFILE}"; fi
s="x"
i=0
while [ "\$i" -lt 26 ]; do
    s="\$s\$s"
    i=\$((i + 1))
done
echo '| model | size | params | backend | ngl | test | t/s |'
echo '|---|---|---|---|---|---|---|'
echo "| gpt-oss 20B MXFP4 | 12.83 GiB | 20.91 B | SYCL | 99 | tg128 | $tg ± 0.31 |"
end=\$((SECONDS + 2))
while [ "\$SECONDS" -lt "\$end" ]; do :; done
EOF
    chmod +x "$path"
    echo "$path"
}

# mk_fake_bench_crash: exits nonzero with no markdown table at all --
# simulates a bench that crashed before producing results. Used to prove
# parse_tg128 reports "" (mode=unknown) rather than matching some unrelated
# line that happens to contain the substring "tg128".
mk_fake_bench_crash() {
    local path="$T/fakebench-crash.sh"
    cat > "$path" <<'EOF'
#!/usr/bin/env bash
echo "some crash output, no results table" >&2
exit 1
EOF
    chmod +x "$path"
    echo "$path"
}

# mk_fake_bench_nonnumeric: a well-formed markdown tg128 ROW whose t/s cell
# is garbage ("#") rather than a number -- the crash signature the review
# found in practice, distinct from mk_fake_bench_crash's "no row at all".
mk_fake_bench_nonnumeric() {
    local path="$T/fakebench-nonnumeric.sh"
    cat > "$path" <<'EOF'
#!/usr/bin/env bash
echo '| model | size | params | backend | ngl | test | t/s |'
echo '|---|---|---|---|---|---|---|'
echo '| gpt-oss 20B MXFP4 | 12.83 GiB | 20.91 B | SYCL | 99 | tg128 | # |'
EOF
    chmod +x "$path"
    echo "$path"
}

run_capture() { # $1=out-dir, remaining = extra CAPTURE args, then -- command
    "$CAPTURE" --sysfs-card "$T/sys/class/drm/card9" --meminfo "$T/meminfo" \
        --pgrep-cmd false --df-cmd true --journalctl-cmd true --max-wait 1 \
        --out "$1" "${@:2}"
}

# --- mode computation across the two thresholds and the middle band ---

mk_tree 0 0; mk_meminfo 3000000
out_slow="$T/out-slow"
bench="$(mk_fake_bench 28.0)"
run_capture "$out_slow" -- "$bench" || { echo "FAIL: slow-mode run failed"; fail=1; }
[ -f "$out_slow/mode.txt" ] || { echo "FAIL: mode.txt missing (slow)"; fail=1; }
grep -q "mode=slow" "$out_slow/mode.txt" 2>/dev/null || { echo "FAIL: expected mode=slow (got: $(cat "$out_slow/mode.txt" 2>/dev/null))"; fail=1; }
grep -q "tg128=28.0" "$out_slow/mode.txt" 2>/dev/null || { echo "FAIL: expected tg128=28.0 (got: $(cat "$out_slow/mode.txt" 2>/dev/null))"; fail=1; }

mk_tree 0 0; mk_meminfo 3000000
out_fast="$T/out-fast"
bench="$(mk_fake_bench 39.5)"
run_capture "$out_fast" -- "$bench" || { echo "FAIL: fast-mode run failed"; fail=1; }
grep -q "mode=fast" "$out_fast/mode.txt" 2>/dev/null || { echo "FAIL: expected mode=fast (got: $(cat "$out_fast/mode.txt" 2>/dev/null))"; fail=1; }

mk_tree 0 0; mk_meminfo 3000000
out_mid="$T/out-unknown"
bench="$(mk_fake_bench 34.0)"
run_capture "$out_mid" -- "$bench" || { echo "FAIL: mid-band run failed"; fail=1; }
grep -q "mode=unknown" "$out_mid/mode.txt" 2>/dev/null || { echo "FAIL: expected mode=unknown (got: $(cat "$out_mid/mode.txt" 2>/dev/null))"; fail=1; }

# --- timeline.tsv: >= 3 sampled rows over the fake bench's 2s sleep ---

[ -s "$out_slow/timeline.tsv" ] || { echo "FAIL: timeline.tsv missing/empty"; fail=1; }
rows=$(( $(wc -l < "$out_slow/timeline.tsv") - 1 ))
[ "$rows" -ge 3 ] || { echo "FAIL: expected >= 3 timeline data rows, got $rows"; fail=1; }
head -1 "$out_slow/timeline.tsv" | grep -qE '^t_s	act_freq	cur_freq	throttle_status	reason_pl2	rss_anon_kb$' \
    || { echo "FAIL: timeline.tsv header wrong: $(head -1 "$out_slow/timeline.tsv")"; fail=1; }

# cur_freq and throttle/reason_pl2 were never created by mk_tree above --
# every data row must show the "-" placeholder for both, never abort.
awk -F'\t' 'NR>1 { if ($3 != "-" || $5 != "-") bad=1 } END { exit bad ? 1 : 0 }' "$out_slow/timeline.tsv" \
    || { echo "FAIL: expected '-' placeholder for missing cur_freq/reason_pl2 sysfs files"; fail=1; }

# --- host.txt: before/after blocks with the required fields ---

[ -f "$out_slow/host.txt" ] || { echo "FAIL: host.txt missing"; fail=1; }
grep -q "=== before ===" "$out_slow/host.txt" || { echo "FAIL: host.txt missing before block"; fail=1; }
grep -q "=== after ===" "$out_slow/host.txt" || { echo "FAIL: host.txt missing after block"; fail=1; }
grep -q "loadavg:" "$out_slow/host.txt" || { echo "FAIL: host.txt missing loadavg"; fail=1; }
grep -q "ffmpeg_count:" "$out_slow/host.txt" || { echo "FAIL: host.txt missing ffmpeg_count"; fail=1; }
grep -q "Shmem:" "$out_slow/host.txt" || { echo "FAIL: host.txt missing Shmem"; fail=1; }
grep -q "MemAvailable:" "$out_slow/host.txt" || { echo "FAIL: host.txt missing MemAvailable"; fail=1; }
grep -qE '^bench_pid=[0-9]+ comm=' "$out_slow/host.txt" \
    || { echo "FAIL: host.txt missing a resolved bench_pid=<pid> comm=... audit line (got: $(grep '^bench_pid=' "$out_slow/host.txt" 2>/dev/null))"; fail=1; }

# --- regression: RssAnon must track the ACTUAL bench pid and its ACTUAL
# memory, never a wrapper (llama.cpp-gvu7 review, two rounds). Two earlier
# versions' pid discovery each latched onto the wrong process for the whole
# run and no prior assertion caught either: a fallback to the guard's own
# direct child froze on `timeout` (reproduced on B70 hardware as a constant
# ~1.2 MB RssAnon across 77 rows), and a {timeout,env} comm denylist latched
# onto bench-guard's OWN throttle/tenant poll `sleep` instead. "At least one
# non-'-' RssAnon sample" is trivially true of any of those wrappers too, so
# this replaces that control with two checks that are NOT: mk_fake_bench_grow
# writes its own real pid to FAKE_BENCH_PIDFILE and holds tens of MB of its
# own memory throughout (see that fixture's comment for why it never forks),
# so host.txt's audit line must equal that ground-truth pid, AND the
# timeline's peak RssAnon must clear a threshold no mere wrapper process
# could reach. ---

out_pid="$T/out-pidcheck"
mk_tree 0 0; mk_meminfo 3000000
bench="$(mk_fake_bench_grow 40.0)"
export FAKE_BENCH_PIDFILE="$T/fake-bench.pid"
run_capture "$out_pid" -- "$bench" || { echo "FAIL: pid-check run failed"; fail=1; }
unset FAKE_BENCH_PIDFILE
[ -s "$T/fake-bench.pid" ] || { echo "FAIL: fake bench never wrote its own pid"; fail=1; }
want_pid="$(cat "$T/fake-bench.pid")"
grep -q "^bench_pid=$want_pid comm=" "$out_pid/host.txt" \
    || { echo "FAIL: expected bench_pid=$want_pid in host.txt (got: $(grep '^bench_pid=' "$out_pid/host.txt" 2>/dev/null))"; fail=1; }

# 32768 kB (32 MiB): a `timeout`/`env`/`sleep` wrapper can never reach this
# (~1.2 MB observed), and the fixture's ~64 MiB steady-state RssAnon clears
# it with roughly 2x margin.
max_rss="$(awk -F'\t' 'NR>1 && $6 != "-" { v = $6 + 0; if (v > max) max = v } END { print max + 0 }' "$out_pid/timeline.tsv")"
[ "$max_rss" -gt 32768 ] \
    || { echo "FAIL: expected a sampled RssAnon > 32768 kB (fixture holds ~64 MiB); got max=$max_rss kB -- RssAnon may be tracking a wrapper, not the bench"; fail=1; }

# --- regression: proof B (llama.cpp-gvu7 review). While the card is
# (simulated) busy, bench-guard's OWN preflight throttle/tenant poll loop
# forks its own `sleep 5` directly under the guard -- structural leaf
# descent WILL transiently latch onto that (expected, self-correcting, not
# a bug: it is genuinely the deepest live process at that instant). The
# regression this guards is whether tracking permanently sticks there once
# the real chain forms, the way a comm denylist for {timeout, env} did (it
# does not reject "sleep") in an earlier, already-replaced version of this
# script. mk_tree_busy_then_free starts throttle/status=1 and flips it to 0
# via a background subshell after ~6s, mirroring the reviewer's own repro;
# bench-guard's fixed 5s poll interval means at least one of its own poll
# `sleep`s is observed before the card clears. --max-wait 20 gives ample
# budget past the ~10s worst case (two 5s poll cycles) before bench-guard
# itself would refuse.
mk_tree_busy_then_free() {
    local d="$T/sys/class/drm/card9/device/tile0/gt0/freq0"
    mkdir -p "$d/throttle"
    printf '1\n' > "$d/throttle/status"
    printf '1800\n' > "$d/act_freq"
    ( sleep 6; printf '0\n' > "$d/throttle/status"; printf '0\n' > "$d/act_freq" ) &
}

out_busy="$T/out-busycard"
mk_tree_busy_then_free
mk_meminfo 3000000
bench="$(mk_fake_bench_grow 40.0)"
export FAKE_BENCH_PIDFILE="$T/fake-bench-busy.pid"
"$CAPTURE" --sysfs-card "$T/sys/class/drm/card9" --meminfo "$T/meminfo" \
    --pgrep-cmd false --df-cmd true --journalctl-cmd true --max-wait 20 \
    --out "$out_busy" -- "$bench" || { echo "FAIL: busy-card run failed"; fail=1; }
unset FAKE_BENCH_PIDFILE
wait 2>/dev/null || true

[ -s "$T/fake-bench-busy.pid" ] || { echo "FAIL: fake bench never wrote its own pid (busy-card)"; fail=1; }
want_busy_pid="$(cat "$T/fake-bench-busy.pid")"
grep -q "^bench_pid=$want_busy_pid comm=" "$out_busy/host.txt" \
    || { echo "FAIL: expected bench_pid=$want_busy_pid in host.txt after busy-card recovery (got: $(grep '^bench_pid=' "$out_busy/host.txt" 2>/dev/null))"; fail=1; }
grep -qE '^bench_pid=[0-9]+ comm=(sleep|timeout) ' "$out_busy/host.txt" \
    && { echo "FAIL: host.txt still names a wrapper after busy-card recovery: $(grep '^bench_pid=' "$out_busy/host.txt")"; fail=1; }
max_rss_busy="$(awk -F'\t' 'NR>1 && $6 != "-" { v = $6 + 0; if (v > max) max = v } END { print max + 0 }' "$out_busy/timeline.tsv")"
[ "$max_rss_busy" -gt 32768 ] \
    || { echo "FAIL: expected a sampled RssAnon > 32768 kB after busy-card recovery; got max=$max_rss_busy kB"; fail=1; }

# --- tg128 parsing must be anchored to an actual markdown table row and
# validated as numeric (llama.cpp-gvu7 review): a bare substring grep over
# the whole log also matches bench-guard's own header line (which echoes the
# wrapped command, including GGML_SYCL_KERNEL_PROFILE_OUTPUT="$OUT/kprof" --
# an --out dir whose path happens to contain "tg128" puts that word in a
# non-data line too), and a non-numeric cell must not fall through
# compute_mode's `v + 0` coercion into a false mode=slow. ---

out_crash="$T/tg128-run1"    # path itself contains "tg128", by design
mk_tree 0 0; mk_meminfo 3000000
bench="$(mk_fake_bench_crash)"
crash_rc=0
run_capture "$out_crash" -- "$bench" || crash_rc=$?
[ "$crash_rc" -eq 1 ] || { echo "FAIL: expected the crashed bench's own exit code (1) to propagate, got $crash_rc"; fail=1; }
[ -f "$out_crash/mode.txt" ] || { echo "FAIL: mode.txt missing for a non-refusal crash"; fail=1; }
grep -qx "tg128=unknown mode=unknown" "$out_crash/mode.txt" \
    || { echo "FAIL: expected tg128=unknown mode=unknown for a no-results-table run whose --out path contains 'tg128' (got: $(cat "$out_crash/mode.txt" 2>/dev/null))"; fail=1; }

out_nonnum="$T/out-nonnumeric"
mk_tree 0 0; mk_meminfo 3000000
bench="$(mk_fake_bench_nonnumeric)"
run_capture "$out_nonnum" -- "$bench" || { echo "FAIL: non-numeric-cell run failed"; fail=1; }
grep -qx "tg128=unknown mode=unknown" "$out_nonnum/mode.txt" 2>/dev/null \
    || { echo "FAIL: expected tg128=unknown mode=unknown for a non-numeric t/s cell (got: $(cat "$out_nonnum/mode.txt" 2>/dev/null))"; fail=1; }

# --- exit status mirrors the underlying run (0 on a clean run) ---

mk_tree 0 0; mk_meminfo 3000000
bench="$(mk_fake_bench 40.0)"
expect_status 0 "clean run must exit 0" -- run_capture "$T/out-rc" -- "$bench"

# --- refusal: high Shmem (zero tmpfs) must propagate bench-guard's exit 3
# and must NOT write mode.txt -- there was no run to compute a mode from. ---

mk_tree 0 0; mk_meminfo 30000000
out_ref="$T/out-refused"
bench="$(mk_fake_bench 40.0)"
expect_status 3 "high-Shmem refusal must propagate as exit 3" -- run_capture "$out_ref" -- "$bench"
[ ! -f "$out_ref/mode.txt" ] || { echo "FAIL: refusal must not write mode.txt"; fail=1; }

[ "$fail" -eq 0 ] && echo "OK: sycl-decode-mode-capture" || exit 1
