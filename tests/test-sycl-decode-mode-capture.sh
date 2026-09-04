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
# embedding $1 as the t/s figure, then sleeps 2s so the sampler (0.5s
# cadence) gets several samples. The capture script's pid discovery walks the
# guard -> timeout -> command process tree by ancestry only (no name/comm
# match), so this bash-script fixture is found the same way a real
# llama-bench binary would be, comm quirks and all (a shebang script reports
# comm="bash" here, not its own basename -- see find_bench_pid's comment in
# the script).
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

# The bench pid must have been resolved for at least one sample (RssAnon not
# "-" throughout) -- proves --bench-name pid discovery actually worked, not
# just that the placeholder path was taken the whole time.
awk -F'\t' 'NR>1 && $6 != "-" { found=1 } END { exit found ? 0 : 1 }' "$out_slow/timeline.tsv" \
    || { echo "FAIL: expected at least one resolved RssAnon sample (bench pid never found)"; fail=1; }

# --- host.txt: before/after blocks with the required fields ---

[ -f "$out_slow/host.txt" ] || { echo "FAIL: host.txt missing"; fail=1; }
grep -q "=== before ===" "$out_slow/host.txt" || { echo "FAIL: host.txt missing before block"; fail=1; }
grep -q "=== after ===" "$out_slow/host.txt" || { echo "FAIL: host.txt missing after block"; fail=1; }
grep -q "loadavg:" "$out_slow/host.txt" || { echo "FAIL: host.txt missing loadavg"; fail=1; }
grep -q "ffmpeg_count:" "$out_slow/host.txt" || { echo "FAIL: host.txt missing ffmpeg_count"; fail=1; }
grep -q "Shmem:" "$out_slow/host.txt" || { echo "FAIL: host.txt missing Shmem"; fail=1; }
grep -q "MemAvailable:" "$out_slow/host.txt" || { echo "FAIL: host.txt missing MemAvailable"; fail=1; }

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
